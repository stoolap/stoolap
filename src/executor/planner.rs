// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Query Planner - Integrates statistics and cost-based optimization
//!
//! This module provides the QueryPlanner which coordinates between:
//! - Table statistics stored in system tables (sys_table_stats, sys_column_stats)
//! - Cost estimator for choosing access methods
//! - Zone maps for segment pruning
//! - Index selection for efficient access paths
//!
//! The planner is used by the executor to make informed decisions about:
//! - Whether to use an index vs sequential scan
//! - Which join algorithm to use
//! - Which segments can be skipped using zone maps

use std::sync::Arc;

use crate::common::deadline::Deadline;
use crate::common::names::eq_names;
use crate::common::StringMap;
use crate::storage::traits::transaction::{StatementTable, TableUse};

use crate::core::{Operator, Result, Value};
use crate::parser::ast::Expression;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::mvcc::version_store::VersionStore;
use crate::storage::mvcc::zonemap::TableZoneMap;
use crate::storage::statistics::{
    Histogram, HistogramOp, TableStats, SYS_COLUMN_STATS, SYS_TABLE_STATS,
};
use crate::storage::traits::{Engine, Table, Transaction};

/// Query planner that integrates statistics-based optimization
pub struct QueryPlanner {
    /// Reference to the storage engine for reading statistics
    engine: Arc<MVCCEngine>,
    /// Cache of table statistics to avoid repeated lookups
    stats_cache: std::sync::RwLock<StringMap<CachedStats>>,
    /// Monotonic base for the LRU access stamps
    base: crate::common::time_compat::Instant,
}

/// Default TTL for cached statistics (5 minutes)
/// After this time, stats are considered potentially stale and will be refreshed
const STATS_CACHE_TTL_SECS: u64 = 300;

/// Maximum number of tables to cache statistics for (LRU eviction threshold)
/// This prevents unbounded memory growth for databases with many tables
const MAX_STATS_CACHE_SIZE: usize = 1000;

/// Cached statistics for a table
struct CachedStats {
    table_stats: TableStats,
    column_stats: StringMap<ColumnStatsCache>,
    /// Timestamp when this cache entry was created
    cached_at: crate::common::time_compat::Instant,
    /// Engine stats epoch at creation: ANALYZE through any handle bumps
    /// the engine epoch, so entries cached by other executors go stale
    /// immediately instead of waiting out the TTL.
    stats_epoch: u64,
    /// Seconds since planner creation at last access (for LRU eviction).
    /// Atomic so cache hits can touch it under the read lock.
    last_accessed: std::sync::atomic::AtomicU64,
}

impl CachedStats {
    /// Check if this cache entry is stale (older than TTL, or superseded
    /// by an ANALYZE elsewhere)
    fn is_stale(&self, current_stats_epoch: u64) -> bool {
        self.stats_epoch != current_stats_epoch
            || self.cached_at.elapsed().as_secs() > STATS_CACHE_TTL_SECS
    }

    /// Update last accessed time
    fn touch(&self, now_secs: u64) {
        self.last_accessed
            .store(now_secs, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Cached column stats (simplified for internal use)
#[derive(Clone)]
pub struct ColumnStatsCache {
    /// Number of null values in the column
    pub null_count: u64,
    /// Number of distinct values in the column
    pub distinct_count: u64,
    /// Minimum value in the column
    pub min_value: Option<Value>,
    /// Maximum value in the column
    pub max_value: Option<Value>,
    /// Average width of values in bytes
    pub avg_width: usize,
    /// Histogram for range selectivity estimation
    pub histogram: Option<Histogram>,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(engine: Arc<MVCCEngine>) -> Self {
        Self {
            engine,
            stats_cache: std::sync::RwLock::new(StringMap::new()),
            base: crate::common::time_compat::Instant::now(),
        }
    }

    /// Seconds since planner creation (monotonic LRU stamp)
    fn now_secs(&self) -> u64 {
        self.base.elapsed().as_secs()
    }

    /// Invalidate cached statistics for a table
    ///
    /// Call this after ANALYZE to ensure fresh statistics are used.
    pub fn invalidate_stats_cache(&self, table_name: &str) {
        let mut cache = self.stats_cache.write().unwrap();
        cache.remove(&table_name.to_lowercase());
    }

    /// Clear all cached statistics
    pub fn clear_stats_cache(&self) {
        let mut cache = self.stats_cache.write().unwrap();
        cache.clear();
    }

    /// Get or load statistics for a table
    ///
    /// If cached stats are stale (older than TTL), they will be refreshed
    /// from the system tables.
    ///
    /// Returns None if:
    /// - No statistics have been collected (ANALYZE not run)
    /// - Statistics have row_count == 0 (empty/invalid stats)
    pub fn get_table_stats(&self, table_name: &str) -> Option<TableStats> {
        let key = table_name.to_lowercase();

        {
            let current_epoch = self.engine.stats_epoch();
            let cache = self.stats_cache.read().unwrap();
            if let Some(cached) = cache.get(&key) {
                // A fresh entry is authoritative either way: row_count == 0
                // is the cached "no stats collected" verdict, kept so every
                // statement does not re-probe the system tables until
                // ANALYZE invalidates the entry or the TTL expires.
                if !cached.is_stale(current_epoch) {
                    cached.touch(self.now_secs());
                    return (cached.table_stats.row_count > 0).then(|| cached.table_stats.clone());
                }
                // Stats are stale, will reload below
            }
        }

        // Load from system tables (will update cache)
        // Only return stats if they have valid row_count > 0
        self.load_stats_from_system_tables(table_name)
            .ok()
            .filter(|stats| stats.row_count > 0)
    }

    /// Get table statistics with fallback to runtime estimation
    ///
    /// If ANALYZE hasn't been run, computes basic statistics from the table.
    /// This ensures the optimizer always has some statistics to work with.
    pub fn get_table_stats_with_fallback(&self, table: &dyn Table) -> TableStats {
        let table_name = table.name();

        // Try to get analyzed stats first
        if let Some(stats) = self.get_table_stats(table_name) {
            if stats.row_count > 0 {
                return stats;
            }
        }

        // Fallback: compute basic stats from table (use hint for O(1))
        let row_count = table.row_count_hint() as u64;
        TableStats {
            table_name: table_name.to_string(),
            row_count,
            page_count: (row_count / 100).max(1), // Estimate ~100 rows per page
            avg_row_size: 100,                    // Default estimate
        }
    }

    /// Get column statistics
    ///
    /// If cached stats are stale (older than TTL), they will be refreshed.
    pub fn get_column_stats(
        &self,
        table_name: &str,
        column_name: &str,
    ) -> Option<ColumnStatsCache> {
        let table_key = table_name.to_lowercase();
        let col_key = column_name.to_lowercase();

        // Check cache first
        let should_reload = {
            let current_epoch = self.engine.stats_epoch();
            let cache = self.stats_cache.read().unwrap();
            if let Some(cached) = cache.get(&table_key) {
                if !cached.is_stale(current_epoch) {
                    cached.touch(self.now_secs());
                    return cached.column_stats.get(&col_key).cloned();
                }
                true // Stale, need to reload
            } else {
                true // Not in cache, need to load
            }
        };

        if should_reload {
            // Load stats which will populate cache
            let _ = self.load_stats_from_system_tables(table_name);
        }

        // Try cache again
        let cache = self.stats_cache.read().unwrap();
        cache.get(&table_key).and_then(|c| {
            c.touch(self.now_secs());
            c.column_stats.get(&col_key).cloned()
        })
    }

    /// Get zone maps for a table (from table, not system tables)
    /// Uses Arc to avoid cloning on high QPS workloads
    pub fn get_zone_maps(&self, table: &dyn Table) -> Option<std::sync::Arc<TableZoneMap>> {
        table.get_zone_maps()
    }

    /// Load statistics from system tables
    fn load_stats_from_system_tables(&self, table_name: &str) -> Result<TableStats> {
        // Capture the epoch BEFORE reading: if an ANALYZE on another handle
        // commits and bumps between the read and the cache insert, the
        // entry carries the pre-read epoch and the next access sees it as
        // stale instead of serving old stats stamped fresh for the TTL.
        let stats_epoch_before_read = self.engine.stats_epoch();
        let tx = self.engine.begin_transaction()?;

        // Check if system tables exist
        let tables = tx.list_tables()?;
        let has_table_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_TABLE_STATS));
        let has_column_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_COLUMN_STATS));

        // With no stats collected, cache the default (row_count == 0) entry
        // below as a negative verdict: get_table_stats treats a fresh zero
        // entry as authoritative, so statements stop re-probing the system
        // tables until ANALYZE invalidates the entry or the TTL expires.
        let (table_stats, column_stats) = if !has_table_stats {
            (TableStats::default(), StringMap::new())
        } else {
            let table_stats = self.read_table_stats(&*tx, table_name)?;
            let column_stats = if has_column_stats {
                self.read_column_stats(&*tx, table_name)?
            } else {
                StringMap::new()
            };
            (table_stats, column_stats)
        };

        // Cache the stats with current timestamp
        {
            let mut cache = self.stats_cache.write().unwrap();

            // LRU eviction: if cache is full, remove least recently used entries
            if cache.len() >= MAX_STATS_CACHE_SIZE {
                // Find the least recently used entry (oldest last_accessed time)
                if let Some(lru_key) = cache
                    .iter()
                    .min_by_key(|(_, v)| v.last_accessed.load(std::sync::atomic::Ordering::Relaxed))
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&lru_key);
                }
            }

            cache.insert(
                table_name.to_lowercase(),
                CachedStats {
                    table_stats: table_stats.clone(),
                    column_stats,
                    cached_at: crate::common::time_compat::Instant::now(),
                    stats_epoch: stats_epoch_before_read,
                    last_accessed: std::sync::atomic::AtomicU64::new(self.now_secs()),
                },
            );
        }

        Ok(table_stats)
    }

    /// Read table statistics from sys_table_stats
    ///
    /// Table schema is:
    /// id (0), table_name (1), row_count (2), page_count (3), avg_row_size (4), last_analyzed (5)
    fn read_table_stats(&self, tx: &dyn Transaction, table_name: &str) -> Result<TableStats> {
        let stats_table = match tx.get_read_table_at(SYS_TABLE_STATS, None, TableUse::Read) {
            Ok(t) => t,
            Err(_) => return Ok(TableStats::default()),
        };

        // Scan for this table's stats (all columns, no filter)
        let mut result = stats_table.scan(&[], None, Deadline::NONE)?;
        while result.next() {
            let row = result.row();
            // Check if this row is for our table (table_name is column 1)
            if let Some(Value::Text(name)) = row.get(1) {
                if eq_names(name, table_name) {
                    return Ok(TableStats {
                        table_name: table_name.to_string(),
                        row_count: row.get(2).and_then(|v| v.as_int64()).unwrap_or(0) as u64,
                        page_count: row.get(3).and_then(|v| v.as_int64()).unwrap_or(1) as u64,
                        avg_row_size: row.get(4).and_then(|v| v.as_int64()).unwrap_or(100) as u64,
                    });
                }
            }
        }

        // No stats found - return default
        Ok(TableStats::default())
    }

    /// Read column statistics from sys_column_stats
    ///
    /// Table schema is:
    /// id (0), table_name (1), column_name (2), null_count (3), distinct_count (4),
    /// min_value (5), max_value (6), avg_width (7), histogram (8)
    fn read_column_stats(
        &self,
        tx: &dyn Transaction,
        table_name: &str,
    ) -> Result<StringMap<ColumnStatsCache>> {
        let mut stats = StringMap::new();

        let stats_table = match tx.get_read_table_at(SYS_COLUMN_STATS, None, TableUse::Read) {
            Ok(t) => t,
            Err(_) => return Ok(stats),
        };

        // Scan for this table's column stats (all columns, no filter)
        let mut result = stats_table.scan(&[], None, Deadline::NONE)?;
        while result.next() {
            let row = result.row();
            // Check if this row is for our table (table_name is column 1)
            if let Some(Value::Text(name)) = row.get(1) {
                if eq_names(name, table_name) {
                    if let Some(Value::Text(col_name)) = row.get(2) {
                        // Parse histogram from JSON string if available
                        let histogram = row
                            .get(8)
                            .and_then(|v| match v {
                                Value::Text(s) => Some(s.to_string()),
                                _ => None,
                            })
                            .and_then(|s| Histogram::from_json(&s));

                        let col_stats = ColumnStatsCache {
                            null_count: row.get(3).and_then(|v| v.as_int64()).unwrap_or(0) as u64,
                            distinct_count: row.get(4).and_then(|v| v.as_int64()).unwrap_or(0)
                                as u64,
                            min_value: row.get(5).cloned(),
                            max_value: row.get(6).cloned(),
                            avg_width: row.get(7).and_then(|v| v.as_int64()).unwrap_or(8) as usize,
                            histogram,
                        };
                        stats.insert(col_name.to_lowercase().to_string(), col_stats);
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Estimate selectivity for a predicate
    fn estimate_selectivity(
        &self,
        op: Option<Operator>,
        value: Option<&Value>,
        col_stats: Option<&ColumnStatsCache>,
        table_stats: &TableStats,
    ) -> f64 {
        match (op, value, col_stats) {
            (Some(Operator::Eq), _, Some(stats)) if stats.distinct_count > 0 => {
                // Equality: 1/distinct_count
                1.0 / stats.distinct_count as f64
            }
            (Some(Operator::Eq), _, _) => {
                // Default equality selectivity
                0.1
            }
            (Some(Operator::Ne), _, Some(stats)) if stats.distinct_count > 0 => {
                // Not equal: 1 - 1/distinct_count
                1.0 - (1.0 / stats.distinct_count as f64)
            }
            (Some(Operator::Ne), _, _) => 0.9,
            (
                Some(Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte),
                Some(val),
                Some(stats),
            ) => {
                // Use histogram for accurate range selectivity if available
                if let Some(ref histogram) = stats.histogram {
                    let hist_op = match op {
                        Some(Operator::Lt) => HistogramOp::LessThan,
                        Some(Operator::Lte) => HistogramOp::LessThanOrEqual,
                        Some(Operator::Gt) => HistogramOp::GreaterThan,
                        Some(Operator::Gte) => HistogramOp::GreaterThanOrEqual,
                        _ => HistogramOp::Equal,
                    };
                    return histogram.estimate_selectivity(val, hist_op);
                }

                // Fall back to min/max based heuristic
                if let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) {
                    if min < max {
                        // Estimate position in range using linear interpolation
                        let position = Self::estimate_value_position(val, min, max);
                        match op {
                            Some(Operator::Lt | Operator::Lte) => {
                                if val <= min {
                                    0.01
                                } else if val >= max {
                                    0.99
                                } else {
                                    position.clamp(0.01, 0.99)
                                }
                            }
                            Some(Operator::Gt | Operator::Gte) => {
                                if val >= max {
                                    0.01
                                } else if val <= min {
                                    0.99
                                } else {
                                    (1.0 - position).clamp(0.01, 0.99)
                                }
                            }
                            _ => 0.33,
                        }
                    } else {
                        0.33
                    }
                } else {
                    0.33
                }
            }
            (Some(Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte), _, _) => {
                // Default range selectivity
                0.33
            }
            (Some(Operator::Like), _, _) => {
                // LIKE selectivity depends on pattern
                0.25
            }
            (Some(Operator::In), _, _) => {
                // IN selectivity
                0.2
            }
            (Some(Operator::NotIn), _, _) => {
                // NOT IN selectivity
                0.8
            }
            (Some(Operator::IsNull), _, Some(stats)) if table_stats.row_count > 0 => {
                stats.null_count as f64 / table_stats.row_count as f64
            }
            (Some(Operator::IsNotNull), _, Some(stats)) if table_stats.row_count > 0 => {
                1.0 - (stats.null_count as f64 / table_stats.row_count as f64)
            }
            _ => 1.0, // No selectivity reduction
        }
    }

    /// Estimate the position of a value within a range (0.0 to 1.0)
    /// Used for linear interpolation when histogram is not available
    fn estimate_value_position(value: &Value, min: &Value, max: &Value) -> f64 {
        match (min, max, value) {
            (Value::Integer(lo), Value::Integer(hi), Value::Integer(v)) => {
                if hi == lo {
                    0.5
                } else {
                    ((*v - *lo) as f64 / (*hi - *lo) as f64).clamp(0.0, 1.0)
                }
            }
            (Value::Float(lo), Value::Float(hi), Value::Float(v)) => {
                if (hi - lo).abs() < f64::EPSILON {
                    0.5
                } else {
                    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
                }
            }
            // Handle mixed integer/float comparisons
            (Value::Integer(lo), Value::Integer(hi), Value::Float(v)) => {
                let lo_f = *lo as f64;
                let hi_f = *hi as f64;
                if (hi_f - lo_f).abs() < f64::EPSILON {
                    0.5
                } else {
                    ((v - lo_f) / (hi_f - lo_f)).clamp(0.0, 1.0)
                }
            }
            (Value::Float(lo), Value::Float(hi), Value::Integer(v)) => {
                let v_f = *v as f64;
                if (hi - lo).abs() < f64::EPSILON {
                    0.5
                } else {
                    ((v_f - lo) / (hi - lo)).clamp(0.0, 1.0)
                }
            }
            _ => 0.5, // Default for non-comparable types
        }
    }

    /// Check if zone maps indicate that no rows can possibly match the expression
    ///
    /// Returns true if the entire scan can be skipped (zone maps show no match possible).
    /// Returns false if:
    /// - Zone maps are not available
    /// - Some segments might match
    /// - Expression cannot be evaluated against zone maps
    ///
    /// This enables early exit optimization for range queries on ordered data.
    pub fn can_prune_entire_scan(
        &self,
        table: &dyn Table,
        expr: &dyn crate::storage::expression::Expression,
    ) -> bool {
        let zone_maps = match table.get_zone_maps() {
            Some(zm) => zm,
            None => return false, // No zone maps, cannot prune
        };

        // The maps describe the committed state of one publish epoch, so they
        // answer only for a statement reading exactly that state: no commit
        // since, no older snapshot and no writes of its own
        let reads_their_state = zone_maps
            .epoch
            .is_some_and(|epoch| table.publish_epoch_if_quiet() == Some(epoch))
            && table.sees_newest()
            && !table.has_local_changes();
        if !reads_their_state {
            return false;
        }

        // Extract all comparisons from the expression
        let comparisons = expr.collect_comparisons();
        if comparisons.is_empty() {
            return false; // No simple comparisons to check
        }

        // For AND expressions: ALL comparisons must show no possible match
        // For a single comparison: check if any segment could match; a
        // comparison no zone map answers keeps the scan
        for (column, op, value) in comparisons {
            if zone_maps.may_match(column, op, value) != Some(false) {
                return false;
            }
        }

        // All comparisons indicate no segments match - can skip entire scan
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::zone_map_pruned();
        true
    }

    /// Get overall health of statistics for a table
    pub fn stats_health(&self, table_name: &str) -> StatsHealth {
        let table_stats = self.get_table_stats(table_name);

        match table_stats {
            Some(stats) => {
                if stats.row_count > 0 {
                    StatsHealth::Current
                } else {
                    StatsHealth::Missing
                }
            }
            None => StatsHealth::Missing,
        }
    }

    // =========================================================================
    // Cardinality Estimation for Scans
    // =========================================================================

    /// The arms of the share of `table_name`'s rows `predicate` keeps: a
    /// comparison of an indexed column with a value is counted by the index on
    /// each run, else estimated by the statistics ANALYZE collected, else by
    /// the default selectivities
    pub(crate) fn share_arms(&self, table_name: &str, predicate: &Expression) -> ShareArms {
        let table_stats = self.get_table_stats(table_name).unwrap_or_default();
        let store = self.engine.get_version_store(table_name).ok();
        let root = self.share_arm(table_name, predicate, (&table_stats, store.as_deref()));
        ShareArms {
            root,
            table_stats,
            table_name: table_name.to_string(),
            store,
        }
    }

    /// The share `arms` keeps on a run with `params`
    pub(crate) fn arms_share(&self, arms: &ShareArms, params: &[Value]) -> f64 {
        let run = Run {
            table_stats: &arms.table_stats,
            params,
            counting: (arms.store.as_deref()).map(|store| (arms.table_name.as_str(), store)),
            hot_rows: std::cell::OnceCell::new(),
        };
        self.arm_share(&arms.root, &run).clamp(0.0, 1.0)
    }

    fn arm_share(&self, arm: &ShareArm, run: &Run) -> f64 {
        match arm {
            ShareArm::Fixed(share) => *share,
            // AND multiplies the shares, as of independent terms
            ShareArm::And(sides) => self.arm_share(&sides[0], run) * self.arm_share(&sides[1], run),
            // OR adds them, less their overlap
            ShareArm::Or(sides) => {
                let (a, b) = (
                    self.arm_share(&sides[0], run),
                    self.arm_share(&sides[1], run),
                );
                (a + b - a * b).min(1.0)
            }
            ShareArm::Not(arm) => 1.0 - self.arm_share(arm, run),
            ShareArm::Compare(compared) => self.compared_share(compared, run),
        }
    }

    /// The share a comparison keeps: the index's count of the run's values,
    /// else the estimate of its column's statistics, else the default
    fn compared_share(&self, compared: &Compared, run: &Run) -> f64 {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let value = |at: usize| compared.values.get(at).and_then(|v| v.bind(run.params));
        match compared.compare {
            Compare::Op(op) => {
                let value = value(0);
                let indexed = value.and_then(|v| {
                    let (lower, upper) = match op {
                        Operator::Eq => (Included(v), Included(v)),
                        Operator::Lt => (Unbounded, Excluded(v)),
                        Operator::Lte => (Unbounded, Included(v)),
                        Operator::Gt => (Excluded(v), Unbounded),
                        Operator::Gte => (Included(v), Unbounded),
                        _ => return None,
                    };
                    self.counted_share(compared, run, lower, upper)
                });
                indexed.unwrap_or_else(|| {
                    let stats = compared.stats.as_ref();
                    self.estimate_selectivity(Some(op), value, stats, run.table_stats)
                })
            }
            Compare::In { not, listed } => {
                // Each member the index counts, when it counts every one
                let indexed = listed.then(|| {
                    (compared.values.iter())
                        .map(|member| {
                            let value = member.bind(run.params)?;
                            self.counted_share(compared, run, Included(value), Included(value))
                        })
                        .sum::<Option<f64>>()
                });
                let list_size = if listed {
                    compared.values.len() as f64
                } else {
                    5.0
                };
                let distinct = (compared.stats.as_ref())
                    .map(|s| s.distinct_count.max(1) as f64)
                    .unwrap_or(100.0);
                let share = indexed.flatten().unwrap_or(list_size / distinct).min(1.0);
                if not {
                    1.0 - share
                } else {
                    share
                }
            }
            Compare::Between { not } => {
                let (low, high) = (value(0), value(1));
                let indexed = low.zip(high).and_then(|(low, high)| {
                    self.counted_share(compared, run, Included(low), Included(high))
                });
                // Estimate as (high - low) / (max - min)
                let share = if let Some(share) = indexed {
                    share
                } else if let (Some(stats), Some(low), Some(high)) = (&compared.stats, low, high) {
                    if let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) {
                        let low_pos = Self::estimate_value_position(low, min, max);
                        let high_pos = Self::estimate_value_position(high, min, max);
                        (high_pos - low_pos).abs().clamp(0.01, 0.99)
                    } else {
                        0.25 // Default BETWEEN selectivity
                    }
                } else {
                    0.25
                };
                if not {
                    1.0 - share
                } else {
                    share
                }
            }
        }
    }

    /// The share of rows whose compared column lies from `lower` to `upper`,
    /// as its index counts it while all of the table's rows are hot; None
    /// otherwise, or for a range holding more than `RANGE_COUNT_CAP` rows,
    /// whose walk would cost more than the estimate saves
    fn counted_share(
        &self,
        compared: &Compared,
        run: &Run,
        lower: std::ops::Bound<&Value>,
        upper: std::ops::Bound<&Value>,
    ) -> Option<f64> {
        use std::ops::Bound::Included;
        let index = compared.index.as_deref()?;
        let (table_name, store) = run.counting?;
        let rows = (*run.hot_rows.get_or_init(|| {
            (self.engine.row_count_hint(table_name))
                .filter(|&rows| rows > 0 && rows == store.committed_row_count())
        }))?;
        let count = match (lower, upper) {
            (Included(low), Included(high)) if low == high => {
                let column = compared.column.as_deref()?;
                crate::storage::mvcc::table::index_equality_count(
                    index,
                    &store.schema(),
                    column,
                    low,
                    rows,
                )?
            }
            _ => index
                .count_capped(lower, upper, RANGE_COUNT_CAP)
                .filter(|&count| count <= RANGE_COUNT_CAP)?,
        };
        Some(count as f64 / rows as f64)
    }

    /// Whether `a = b` compares a column of `table_name` that no two rows
    /// share a value of, its primary key or a column of a unique index, with
    /// a value that reads no column
    fn unique_key_equality(&self, table_name: &str, a: &Expression, b: &Expression) -> bool {
        let reads_none = |expr: &Expression| {
            let mut refs = 0;
            crate::executor::utils::visit_column_refs(expr, &mut |_| refs += 1) && refs == 0
        };
        let column = match (self.extract_column_name(a), self.extract_column_name(b)) {
            (Some(column), None) if reads_none(b) => column,
            (None, Some(column)) if reads_none(a) => column,
            _ => return false,
        };
        let Ok(schema) = self.engine.get_table_schema(table_name) else {
            return false;
        };
        let is_pk = schema
            .pk_column_index()
            .is_some_and(|pk| schema.columns[pk].name_lower == column);
        is_pk
            || self
                .engine
                .get_version_store(table_name)
                .is_ok_and(|store| {
                    store
                        .get_index_by_column(column)
                        .is_some_and(|index| index.is_unique())
                })
    }

    /// The arm of `expr`'s share of `table_name`'s rows, its comparisons
    /// resolved to their columns' indexes and statistics
    fn share_arm(&self, table_name: &str, expr: &Expression, building: Building<'_>) -> ShareArm {
        use crate::parser::ast::{InfixOperator, PrefixOperator};
        let (table_stats, store) = building;
        // A comparison's share is fixed unless a run's parameter or its
        // index's count decides it
        let compared = |column: Option<&str>, compare: Compare, values: Vec<ValueSource>| {
            let compared = Compared {
                index: column
                    .zip(store)
                    .and_then(|(c, s)| s.get_index_by_column(c)),
                stats: (column.filter(|_| table_stats.row_count > 0))
                    .and_then(|c| self.get_column_stats(table_name, c)),
                column: column.map(str::to_string),
                compare,
                values,
            };
            let params = (compared.values.iter()).any(|v| matches!(v, ValueSource::Param(_)));
            if params || (compared.index.is_some() && compare.counts()) {
                return ShareArm::Compare(Box::new(compared));
            }
            let run = Run {
                table_stats,
                params: &[],
                counting: None,
                hot_rows: std::cell::OnceCell::new(),
            };
            ShareArm::Fixed(self.compared_share(&compared, &run))
        };
        let like = |pattern: &Expression, negated: bool| {
            let share = match self.extract_string_value(pattern) {
                Some(p) if !p.starts_with('%') => 0.1, // Prefix match is more selective
                _ => 0.25,                             // Suffix or contains
            };
            ShareArm::Fixed(if negated { 1.0 - share } else { share })
        };

        match expr {
            Expression::Infix(infix) => match infix.op_type {
                InfixOperator::And | InfixOperator::Or => {
                    let sides = Box::new([
                        self.share_arm(table_name, &infix.left, building),
                        self.share_arm(table_name, &infix.right, building),
                    ]);
                    if matches!(infix.op_type, InfixOperator::And) {
                        ShareArm::And(sides)
                    } else {
                        ShareArm::Or(sides)
                    }
                }
                InfixOperator::Is | InfixOperator::IsNot
                    if matches!(infix.right.as_ref(), Expression::NullLiteral(_)) =>
                {
                    let op = match infix.op_type {
                        InfixOperator::Is => Operator::IsNull,
                        _ => Operator::IsNotNull,
                    };
                    let column = self.extract_column_name(&infix.left);
                    compared(column, Compare::Op(op), Vec::new())
                }
                InfixOperator::Equal | InfixOperator::NotEqual => {
                    // A key no two rows share, equal to a value, keeps one
                    // row, a share no count of rows moves
                    let equal = matches!(infix.op_type, InfixOperator::Equal);
                    if equal && self.unique_key_equality(table_name, &infix.left, &infix.right) {
                        return ShareArm::Fixed(0.0);
                    }
                    let column = (self.extract_column_name(&infix.left))
                        .or_else(|| self.extract_column_name(&infix.right));
                    let value = match self.value_source(&infix.right) {
                        ValueSource::Unknown => self.value_source(&infix.left),
                        value => value,
                    };
                    let op = if equal { Operator::Eq } else { Operator::Ne };
                    compared(column, Compare::Op(op), vec![value])
                }
                InfixOperator::LessThan
                | InfixOperator::LessEqual
                | InfixOperator::GreaterThan
                | InfixOperator::GreaterEqual => {
                    let op = match infix.op_type {
                        InfixOperator::LessThan => Operator::Lt,
                        InfixOperator::LessEqual => Operator::Lte,
                        InfixOperator::GreaterThan => Operator::Gt,
                        _ => Operator::Gte,
                    };
                    let column = self.extract_column_name(&infix.left);
                    compared(
                        column,
                        Compare::Op(op),
                        vec![self.value_source(&infix.right)],
                    )
                }
                InfixOperator::Like | InfixOperator::ILike => like(&infix.right, false),
                InfixOperator::NotLike | InfixOperator::NotILike => like(&infix.right, true),
                // Default for other operators
                _ => ShareArm::Fixed(0.5),
            },
            Expression::In(in_expr) => {
                let members = match in_expr.right.as_ref() {
                    Expression::List(list) => Some(&list.elements[..]),
                    Expression::ExpressionList(list) => Some(&list.expressions[..]),
                    _ => None,
                };
                let values = (members.unwrap_or_default().iter())
                    .map(|member| self.value_source(member))
                    .collect();
                let compare = Compare::In {
                    not: in_expr.not,
                    listed: members.is_some(),
                };
                compared(self.extract_column_name(&in_expr.left), compare, values)
            }
            Expression::Between(between) => {
                let values = vec![
                    self.value_source(&between.lower),
                    self.value_source(&between.upper),
                ];
                let compare = Compare::Between { not: between.not };
                compared(self.extract_column_name(&between.expr), compare, values)
            }
            Expression::Like(like_expr) => {
                let negated = like_expr.operator.to_uppercase().contains("NOT");
                like(&like_expr.pattern, negated)
            }
            Expression::Prefix(prefix) if matches!(prefix.op_type, PrefixOperator::Not) => {
                ShareArm::Not(Box::new(self.share_arm(
                    table_name,
                    &prefix.right,
                    building,
                )))
            }
            // Unknown expressions - conservative estimate
            _ => ShareArm::Fixed(0.5),
        }
    }

    /// Extract column name from an expression
    fn extract_column_name<'e>(&self, expr: &'e Expression) -> Option<&'e str> {
        match expr {
            Expression::Identifier(id) => Some(id.value_lower.as_str()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value_lower.as_str()),
            _ => None,
        }
    }

    /// Extract a Value from a literal expression
    fn extract_value(&self, expr: &Expression) -> Option<Value> {
        match expr {
            Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
            Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
            Expression::StringLiteral(lit) => {
                Some(crate::executor::utils::string_literal_value(lit))
            }
            Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
            Expression::NullLiteral(_) => None, // NULL doesn't have a comparable value
            _ => None,
        }
    }

    /// Where a run finds `expr`'s value: a literal's own, or a parameter's
    fn value_source(&self, expr: &Expression) -> ValueSource {
        match expr {
            Expression::Parameter(p) => {
                (p.index.checked_sub(1)).map_or(ValueSource::Unknown, ValueSource::Param)
            }
            _ => (self.extract_value(expr)).map_or(ValueSource::Unknown, ValueSource::Literal),
        }
    }

    /// Extract string value from an expression
    fn extract_string_value(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::StringLiteral(lit) => Some(lit.value.to_string()),
            _ => None,
        }
    }
}

/// The most rows an estimate counts in a range before it estimates instead
const RANGE_COUNT_CAP: usize = 4096;

/// The share of a table's rows a predicate keeps, resolved once per
/// statistics epoch for runs to bind and count
pub(crate) struct ShareArms {
    root: ShareArm,
    table_stats: TableStats,
    table_name: String,
    store: Option<Arc<VersionStore>>,
}

impl std::fmt::Debug for ShareArms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShareArms")
            .field("table_name", &self.table_name)
            .finish_non_exhaustive()
    }
}

enum ShareArm {
    /// A share no run changes
    Fixed(f64),
    And(Box<[ShareArm; 2]>),
    Or(Box<[ShareArm; 2]>),
    Not(Box<ShareArm>),
    /// A comparison whose share a run's values or its index's count decide
    Compare(Box<Compared>),
}

/// A column compared with values, with the index and the statistics of the
/// column; `column` names it for an index that counts
struct Compared {
    compare: Compare,
    values: Vec<ValueSource>,
    column: Option<String>,
    index: Option<Arc<dyn crate::storage::traits::Index>>,
    stats: Option<ColumnStatsCache>,
}

#[derive(Clone, Copy)]
enum Compare {
    Op(Operator),
    Between {
        not: bool,
    },
    /// `listed` when the values are the members of a list
    In {
        not: bool,
        listed: bool,
    },
}

impl Compare {
    /// Whether an index counts the rows this comparison keeps
    fn counts(self) -> bool {
        !matches!(
            self,
            Compare::Op(Operator::Ne | Operator::IsNull | Operator::IsNotNull)
        )
    }
}

enum ValueSource {
    Literal(Value),
    /// The run's parameter at this position
    Param(usize),
    /// A value no estimate binds
    Unknown,
}

impl ValueSource {
    fn bind<'a>(&'a self, params: &'a [Value]) -> Option<&'a Value> {
        match self {
            ValueSource::Literal(value) => Some(value),
            ValueSource::Param(at) => params.get(*at).filter(|value| !value.is_null()),
            ValueSource::Unknown => None,
        }
    }
}

/// What share arms are resolved from: the table's statistics and its store
type Building<'a> = (&'a TableStats, Option<&'a VersionStore>);

/// What one share is estimated with: the table's statistics, the run's
/// parameters, and the table's store, whose indexes count while all of its
/// rows are hot, with those rows found once
struct Run<'a> {
    table_stats: &'a TableStats,
    params: &'a [Value],
    counting: Option<(&'a str, &'a VersionStore)>,
    hot_rows: std::cell::OnceCell<Option<usize>>,
}

/// Health status of table statistics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatsHealth {
    /// Statistics are current (recently analyzed)
    Current,
    /// Statistics exist but may be stale
    Stale,
    /// No statistics available
    Missing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selectivity_estimation() {
        let planner = QueryPlanner::new(Arc::new(MVCCEngine::in_memory()));

        // Test equality selectivity with distinct count
        let col_stats = ColumnStatsCache {
            null_count: 0,
            distinct_count: 100,
            min_value: Some(Value::Integer(1)),
            max_value: Some(Value::Integer(100)),
            avg_width: 8,
            histogram: None,
        };
        let table_stats = TableStats::default();

        let sel = planner.estimate_selectivity(
            Some(Operator::Eq),
            Some(&Value::Integer(50)),
            Some(&col_stats),
            &table_stats,
        );
        assert!((sel - 0.01).abs() < 0.001); // 1/100 = 0.01

        // Test no column stats
        let sel_no_stats = planner.estimate_selectivity(
            Some(Operator::Eq),
            Some(&Value::Integer(50)),
            None,
            &table_stats,
        );
        assert!((sel_no_stats - 0.1).abs() < 0.001); // Default
    }

    #[test]
    fn test_estimate_value_position_integers() {
        // Value in middle of range
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(50),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!((pos - 0.5).abs() < 0.001);

        // Value at start
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(0),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!(pos.abs() < 0.001);

        // Value at end
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(100),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!((pos - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_estimate_value_position_floats() {
        let pos = QueryPlanner::estimate_value_position(
            &Value::Float(0.75),
            &Value::Float(0.0),
            &Value::Float(1.0),
        );
        assert!((pos - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_estimate_value_position_equal_bounds() {
        // Equal bounds should return 0.5
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(50),
            &Value::Integer(50),
            &Value::Integer(50),
        );
        assert!((pos - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_estimate_value_position_clamping() {
        // Value below range should clamp to 0
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(-10),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!(pos.abs() < 0.001);

        // Value above range should clamp to 1
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(200),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!((pos - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_selectivity_range_operators() {
        let planner = QueryPlanner::new(Arc::new(MVCCEngine::in_memory()));

        let col_stats = ColumnStatsCache {
            null_count: 0,
            distinct_count: 100,
            min_value: Some(Value::Integer(1)),
            max_value: Some(Value::Integer(100)),
            avg_width: 8,
            histogram: None,
        };
        let table_stats = TableStats::default();

        // Test Greater Than - should be about 50% for value in middle
        let sel = planner.estimate_selectivity(
            Some(Operator::Gt),
            Some(&Value::Integer(50)),
            Some(&col_stats),
            &table_stats,
        );
        assert!(sel > 0.0 && sel < 1.0);

        // Test Less Than
        let sel = planner.estimate_selectivity(
            Some(Operator::Lt),
            Some(&Value::Integer(50)),
            Some(&col_stats),
            &table_stats,
        );
        assert!(sel > 0.0 && sel < 1.0);
    }

    #[test]
    fn test_selectivity_no_operator() {
        let planner = QueryPlanner::new(Arc::new(MVCCEngine::in_memory()));
        let table_stats = TableStats::default();

        // No operator should return default selectivity
        let sel = planner.estimate_selectivity(None, Some(&Value::Integer(50)), None, &table_stats);
        assert!((sel - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_health_missing() {
        let planner = QueryPlanner::new(Arc::new(MVCCEngine::in_memory()));

        // Non-existent table should return Missing
        let health = planner.stats_health("non_existent_table");
        assert!(matches!(health, StatsHealth::Missing));
    }

    #[test]
    fn test_estimate_value_position_mixed_types() {
        // Integer min/max with float value
        let pos = QueryPlanner::estimate_value_position(
            &Value::Float(50.5),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!(pos > 0.49 && pos < 0.52);

        // Float min/max with integer value
        let pos = QueryPlanner::estimate_value_position(
            &Value::Integer(75),
            &Value::Float(0.0),
            &Value::Float(100.0),
        );
        assert!((pos - 0.75).abs() < 0.001);
    }

    #[test]
    fn cached_stats_go_stale_on_epoch_change() {
        let entry = CachedStats {
            table_stats: TableStats::default(),
            column_stats: StringMap::new(),
            cached_at: crate::common::time_compat::Instant::now(),
            stats_epoch: 1,
            last_accessed: std::sync::atomic::AtomicU64::new(0),
        };
        assert!(!entry.is_stale(1));
        // ANALYZE through another handle bumps the engine epoch: the pinned
        // verdict (including a negative row_count == 0 one) must reload.
        assert!(entry.is_stale(2));
    }
}
