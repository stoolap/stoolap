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

use rustc_hash::FxHashMap;

use crate::core::{IndexType, Operator, Result, Value};
use crate::optimizer::feedback::{fingerprint_predicate, global_feedback_cache};
use crate::optimizer::workload::{EdgeAwarePlanner, EdgeJoinRecommendation};
use crate::optimizer::{
    AccessMethod, BuildSide, CostEstimator, JoinAlgorithm, JoinStats, PlanCost,
};
use crate::parser::ast::Expression;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::mvcc::zonemap::{PruneStats, TableZoneMap};
use crate::storage::statistics::{
    Histogram, HistogramOp, TableStats, SYS_COLUMN_STATS, SYS_TABLE_STATS,
};
use crate::storage::traits::{Engine, Table, Transaction};

/// Query planner that integrates statistics-based optimization
pub struct QueryPlanner {
    /// Reference to the storage engine for reading statistics
    engine: Arc<MVCCEngine>,
    /// Cost estimator for evaluating access methods
    cost_estimator: CostEstimator,
    /// Cache of table statistics to avoid repeated lookups
    stats_cache: std::sync::RwLock<FxHashMap<String, CachedStats>>,
}

/// Default TTL for cached statistics (5 minutes)
/// After this time, stats are considered potentially stale and will be refreshed
const STATS_CACHE_TTL_SECS: u64 = 300;

/// Maximum number of tables to cache statistics for (LRU eviction threshold)
/// This prevents unbounded memory growth for databases with many tables
const MAX_STATS_CACHE_SIZE: usize = 1000;

/// Cached statistics for a table
#[derive(Clone)]
struct CachedStats {
    table_stats: TableStats,
    column_stats: FxHashMap<String, ColumnStatsCache>,
    #[allow(dead_code)]
    zone_maps: Option<TableZoneMap>,
    /// Timestamp when this cache entry was created
    cached_at: std::time::Instant,
    /// Timestamp of last access (for LRU eviction)
    last_accessed: std::time::Instant,
}

impl CachedStats {
    /// Check if this cache entry is stale (older than TTL)
    fn is_stale(&self) -> bool {
        self.cached_at.elapsed().as_secs() > STATS_CACHE_TTL_SECS
    }

    /// Update last accessed time
    fn touch(&mut self) {
        self.last_accessed = std::time::Instant::now();
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
    #[allow(dead_code)]
    pub avg_width: usize,
    /// Histogram for range selectivity estimation
    pub histogram: Option<Histogram>,
}

/// Result of access method selection
#[derive(Debug, Clone)]
pub struct AccessPlan {
    /// Selected access method
    pub method: AccessMethod,
    /// Estimated cost
    pub cost: PlanCost,
    /// Zone map pruning info (segments to scan)
    pub prune_stats: Option<PruneStats>,
    /// Recommendation explanation
    pub explanation: String,
}

/// Result of join planning
#[derive(Debug, Clone)]
pub struct JoinPlan {
    /// Selected join algorithm
    pub algorithm: JoinAlgorithm,
    /// Estimated cost
    pub cost: PlanCost,
    /// Which table should be the build side (for hash join)
    pub build_side: Option<String>,
    /// Explanation of the choice
    pub explanation: String,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(engine: Arc<MVCCEngine>) -> Self {
        Self {
            engine,
            cost_estimator: CostEstimator::new(),
            stats_cache: std::sync::RwLock::new(FxHashMap::default()),
        }
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
    pub fn get_table_stats(&self, table_name: &str) -> Option<TableStats> {
        let key = table_name.to_lowercase();

        // Check cache first - use read lock then upgrade to write for LRU touch
        {
            let cache = self.stats_cache.read().unwrap();
            if let Some(cached) = cache.get(&key) {
                // Return cached stats if still fresh
                if !cached.is_stale() {
                    let result = cached.table_stats.clone();
                    // We need to touch the entry - drop read lock first
                    drop(cache);
                    // Update last_accessed for LRU
                    if let Ok(mut write_cache) = self.stats_cache.write() {
                        if let Some(entry) = write_cache.get_mut(&key) {
                            entry.touch();
                        }
                    }
                    return Some(result);
                }
                // Stats are stale, will reload below
            }
        }

        // Load from system tables (will update cache)
        self.load_stats_from_system_tables(table_name).ok()
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

        // Fallback: compute basic stats from table
        let row_count = table.row_count() as u64;
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
            let cache = self.stats_cache.read().unwrap();
            if let Some(cached) = cache.get(&table_key) {
                if !cached.is_stale() {
                    let result = cached.column_stats.get(&col_key).cloned();
                    // Touch the entry for LRU
                    drop(cache);
                    if let Ok(mut write_cache) = self.stats_cache.write() {
                        if let Some(entry) = write_cache.get_mut(&table_key) {
                            entry.touch();
                        }
                    }
                    return result;
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
        let result = cache
            .get(&table_key)
            .and_then(|c| c.column_stats.get(&col_key).cloned());

        // Touch the entry for LRU if found
        if result.is_some() {
            drop(cache);
            if let Ok(mut write_cache) = self.stats_cache.write() {
                if let Some(entry) = write_cache.get_mut(&table_key) {
                    entry.touch();
                }
            }
        }

        result
    }

    /// Get zone maps for a table (from table, not system tables)
    /// Uses Arc to avoid cloning on high QPS workloads
    pub fn get_zone_maps(&self, table: &dyn Table) -> Option<std::sync::Arc<TableZoneMap>> {
        table.get_zone_maps()
    }

    /// Load statistics from system tables
    fn load_stats_from_system_tables(&self, table_name: &str) -> Result<TableStats> {
        let tx = self.engine.begin_transaction()?;

        // Check if system tables exist
        let tables = tx.list_tables()?;
        let has_table_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_TABLE_STATS));
        let has_column_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_COLUMN_STATS));

        if !has_table_stats {
            // No statistics available - return default
            return Ok(TableStats::default());
        }

        // Read table statistics
        let table_stats = self.read_table_stats(&*tx, table_name)?;

        // Read column statistics if available
        let column_stats = if has_column_stats {
            self.read_column_stats(&*tx, table_name)?
        } else {
            FxHashMap::default()
        };

        // Cache the stats with current timestamp
        {
            let mut cache = self.stats_cache.write().unwrap();

            // LRU eviction: if cache is full, remove least recently used entries
            if cache.len() >= MAX_STATS_CACHE_SIZE {
                // Find the least recently used entry (oldest last_accessed time)
                if let Some(lru_key) = cache
                    .iter()
                    .min_by_key(|(_, v)| v.last_accessed)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&lru_key);
                }
            }

            let now = std::time::Instant::now();
            cache.insert(
                table_name.to_lowercase(),
                CachedStats {
                    table_stats: table_stats.clone(),
                    column_stats,
                    zone_maps: None, // Zone maps are stored in the table, not here
                    cached_at: now,
                    last_accessed: now,
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
        let stats_table = match tx.get_table(SYS_TABLE_STATS) {
            Ok(t) => t,
            Err(_) => return Ok(TableStats::default()),
        };

        // Scan for this table's stats (all columns, no filter)
        let mut result = stats_table.scan(&[], None)?;
        while result.next() {
            let row = result.row();
            // Check if this row is for our table (table_name is column 1)
            if let Some(Value::Text(name)) = row.get(1) {
                if name.eq_ignore_ascii_case(table_name) {
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
    ) -> Result<FxHashMap<String, ColumnStatsCache>> {
        let mut stats = FxHashMap::default();

        let stats_table = match tx.get_table(SYS_COLUMN_STATS) {
            Ok(t) => t,
            Err(_) => return Ok(stats),
        };

        // Scan for this table's column stats (all columns, no filter)
        let mut result = stats_table.scan(&[], None)?;
        while result.next() {
            let row = result.row();
            // Check if this row is for our table (table_name is column 1)
            if let Some(Value::Text(name)) = row.get(1) {
                if name.eq_ignore_ascii_case(table_name) {
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
                        stats.insert(col_name.to_lowercase(), col_stats);
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Choose the best access method for a table scan
    ///
    /// Considers:
    /// - Table statistics (row count, selectivity)
    /// - Available indexes
    /// - Zone maps for segment pruning
    /// - Predicate columns and operators
    pub fn choose_access_method(
        &self,
        table: &dyn Table,
        predicate_column: Option<&str>,
        predicate_op: Option<Operator>,
        predicate_value: Option<&Value>,
    ) -> AccessPlan {
        let table_name = table.name();

        // Get statistics (from cache or system tables)
        let table_stats = self.get_table_stats(table_name).unwrap_or_default();

        // Get zone map pruning info
        let prune_stats = if let (Some(col), Some(op), Some(val)) =
            (predicate_column, predicate_op, predicate_value)
        {
            table
                .get_zone_maps()
                .and_then(|zm| zm.get_prune_stats(col, op, val))
        } else {
            None
        };

        // Check for available indexes on predicate column and get the index
        let index_info = predicate_column.and_then(|col| {
            table.get_index_on_column(col).map(|idx| {
                let idx_type = idx.index_type();
                let idx_name = idx.name().to_string();
                (idx_name, idx_type)
            })
        });

        // Check if this is a range query (not equality)
        let is_range_query = matches!(
            predicate_op,
            Some(Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte)
        );

        // Determine if we can use the index
        // Hash indexes don't support range queries - skip them for non-equality predicates
        let usable_index = index_info.as_ref().and_then(|(name, idx_type)| {
            if *idx_type == IndexType::Hash && is_range_query {
                // Hash index cannot handle range queries - fall back to seq scan
                None
            } else {
                Some((name.clone(), *idx_type))
            }
        });

        // Check if predicate column is primary key
        let is_pk_lookup = predicate_column
            .map(|col| {
                table
                    .schema()
                    .columns
                    .iter()
                    .any(|c| c.name.eq_ignore_ascii_case(col) && c.primary_key)
            })
            .unwrap_or(false);

        // Get column stats for selectivity estimation
        let col_stats = predicate_column.and_then(|col| self.get_column_stats(table_name, col));

        // Estimate selectivity based on operator and value
        let selectivity = self.estimate_selectivity(
            predicate_op,
            predicate_value,
            col_stats.as_ref(),
            &table_stats,
        );

        // Use zone map info to further refine selectivity
        let adjusted_selectivity = if let Some(ref prune) = prune_stats {
            if prune.total_segments > 0 {
                let segment_selectivity =
                    prune.scanned_segments as f64 / prune.total_segments as f64;
                // Combined selectivity: segments * rows within segments
                selectivity * segment_selectivity
            } else {
                selectivity
            }
        } else {
            selectivity
        };

        // Choose access method based on cost
        let (method, cost, explanation) = if is_pk_lookup
            && matches!(predicate_op, Some(Operator::Eq))
        {
            // Primary key point lookup - always best
            let cost = self.cost_estimator.estimate_pk_lookup();
            (
                AccessMethod::PkLookup,
                cost,
                "Primary key equality lookup - O(1) access".to_string(),
            )
        } else if let Some((idx_name, idx_type)) = usable_index {
            if adjusted_selectivity < 0.15 {
                // Low selectivity - index scan is better
                let col_name = predicate_column.unwrap_or("unknown").to_string();
                let cost = self.cost_estimator.estimate_index_scan(
                    &table_stats,
                    None, // No detailed column stats for now
                    adjusted_selectivity,
                    &col_name,
                    idx_type,
                );
                (
                    AccessMethod::IndexScan {
                        index_name: idx_name,
                        column: col_name,
                        selectivity: adjusted_selectivity,
                        index_type: idx_type,
                    },
                    cost,
                    format!(
                        "{} index scan: ~{:.1}% selectivity",
                        idx_type.as_str().to_uppercase(),
                        adjusted_selectivity * 100.0
                    ),
                )
            } else {
                // High selectivity - sequential scan is better
                let cost = self.cost_estimator.estimate_seq_scan(&table_stats);
                let explanation = if let Some(ref prune) = prune_stats {
                    format!(
                        "Sequential scan with zone map pruning: scanning {}/{} segments ({:.1}% pruned)",
                        prune.scanned_segments,
                        prune.total_segments,
                        (prune.pruned_segments as f64 / prune.total_segments as f64) * 100.0
                    )
                } else {
                    format!(
                        "Sequential scan: {} rows, {} pages",
                        table_stats.row_count, table_stats.page_count
                    )
                };
                (AccessMethod::SeqScan, cost, explanation)
            }
        } else {
            // Sequential scan (possibly with zone map pruning)
            let cost = self.cost_estimator.estimate_seq_scan(&table_stats);
            let explanation = if let Some(ref prune) = prune_stats {
                format!(
                    "Sequential scan with zone map pruning: scanning {}/{} segments ({:.1}% pruned)",
                    prune.scanned_segments,
                    prune.total_segments,
                    (prune.pruned_segments as f64 / prune.total_segments as f64) * 100.0
                )
            } else {
                format!(
                    "Sequential scan: {} rows, {} pages",
                    table_stats.row_count, table_stats.page_count
                )
            };
            (AccessMethod::SeqScan, cost, explanation)
        };

        AccessPlan {
            method,
            cost,
            prune_stats,
            explanation,
        }
    }

    /// Choose the best join algorithm
    pub fn choose_join_algorithm(
        &self,
        left_table: &dyn Table,
        right_table: &dyn Table,
        join_column_left: &str,
        join_column_right: &str,
    ) -> JoinPlan {
        let left_stats = self.get_table_stats(left_table.name()).unwrap_or_default();
        let right_stats = self.get_table_stats(right_table.name()).unwrap_or_default();

        let left_col_stats = self.get_column_stats(left_table.name(), join_column_left);
        let right_col_stats = self.get_column_stats(right_table.name(), join_column_right);

        let join_stats = JoinStats {
            left_stats: left_stats.clone(),
            right_stats: right_stats.clone(),
            left_distinct: left_col_stats
                .as_ref()
                .map(|s| s.distinct_count)
                .unwrap_or(0),
            right_distinct: right_col_stats
                .as_ref()
                .map(|s| s.distinct_count)
                .unwrap_or(0),
        };

        // Evaluate different join algorithms
        let hash_cost = self.cost_estimator.estimate_hash_join(&join_stats);
        let nested_loop_cost = self.cost_estimator.estimate_nested_loop_join(&join_stats);
        // Assume unsorted inputs by default
        let merge_cost = self
            .cost_estimator
            .estimate_merge_join(&join_stats, false, false);

        let left_rows = left_stats.row_count;
        let right_rows = right_stats.row_count;

        // Choose the cheapest
        let (algorithm, cost, build_side, explanation) =
            if hash_cost.total <= nested_loop_cost.total && hash_cost.total <= merge_cost.total {
                // Hash join - build side should be smaller table
                let (build_name, bs, build_rows, probe_rows) = if left_rows <= right_rows {
                    (
                        left_table.name().to_string(),
                        BuildSide::Left,
                        left_rows,
                        right_rows,
                    )
                } else {
                    (
                        right_table.name().to_string(),
                        BuildSide::Right,
                        right_rows,
                        left_rows,
                    )
                };
                (
                    JoinAlgorithm::HashJoin {
                        build_side: bs,
                        build_rows,
                        probe_rows,
                    },
                    hash_cost,
                    Some(build_name),
                    format!(
                        "Hash join: build on smaller table (L:{} R:{} rows)",
                        left_rows, right_rows
                    ),
                )
            } else if merge_cost.total <= nested_loop_cost.total {
                (
                    JoinAlgorithm::MergeJoin {
                        left_rows,
                        right_rows,
                        left_sorted: false,
                        right_sorted: false,
                    },
                    merge_cost,
                    None,
                    "Merge join: efficient for sorted inputs".to_string(),
                )
            } else {
                (
                    JoinAlgorithm::NestedLoop {
                        outer_rows: left_rows,
                        inner_rows: right_rows,
                    },
                    nested_loop_cost,
                    None,
                    format!(
                        "Nested loop join: {} x {} = {} comparisons",
                        left_rows,
                        right_rows,
                        left_rows * right_rows
                    ),
                )
            };

        JoinPlan {
            algorithm,
            cost,
            build_side,
            explanation,
        }
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

    /// Get segments to scan based on zone maps
    ///
    /// Returns list of segment IDs that need to be scanned, or None if
    /// zone maps are not available or predicate cannot be evaluated.
    pub fn get_segments_to_scan(
        &self,
        table: &dyn Table,
        column: &str,
        op: Operator,
        value: &Value,
    ) -> Option<Vec<u32>> {
        table.get_segments_to_scan(column, op, value)
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

        // Check if zone maps are stale
        if zone_maps.is_stale() {
            return false; // Stale zone maps, don't trust them
        }

        // Extract all comparisons from the expression
        let comparisons = expr.collect_comparisons();
        if comparisons.is_empty() {
            return false; // No simple comparisons to check
        }

        // For AND expressions: ALL comparisons must show no possible match
        // For a single comparison: check if any segment could match
        for (column, op, value) in comparisons {
            if let Some(segments) = zone_maps.get_segments_to_scan(column, op, value) {
                if !segments.is_empty() {
                    return false; // At least one segment might match
                }
            } else {
                return false; // Cannot evaluate this comparison
            }
        }

        // All comparisons indicate no segments match - can skip entire scan
        true
    }

    /// Get zone map prune statistics for an expression
    ///
    /// Returns aggregated pruning statistics across all comparisons in the expression.
    /// Useful for EXPLAIN output to show zone map effectiveness.
    pub fn get_zone_map_prune_stats(
        &self,
        table: &dyn Table,
        expr: &dyn crate::storage::expression::Expression,
    ) -> Option<crate::storage::mvcc::zonemap::PruneStats> {
        let zone_maps = table.get_zone_maps()?;

        // Get the first comparison that has zone map info
        let comparisons = expr.collect_comparisons();
        for (column, op, value) in comparisons {
            if let Some(stats) = zone_maps.get_prune_stats(column, op, value) {
                return Some(stats);
            }
        }

        None
    }

    /// Clear cached statistics (call after ANALYZE or schema changes)
    pub fn invalidate_cache(&self, table_name: Option<&str>) {
        let mut cache = self.stats_cache.write().unwrap();
        if let Some(name) = table_name {
            cache.remove(&name.to_lowercase());
        } else {
            cache.clear();
        }
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

    /// Estimate the number of rows that will be returned by a scan with a predicate
    ///
    /// This method uses table statistics and column statistics to estimate
    /// selectivity of predicates. It also applies cardinality feedback corrections
    /// if available from previous query executions.
    ///
    /// # Arguments
    /// * `table_name` - Name of the table being scanned
    /// * `predicate` - Optional WHERE clause predicate
    ///
    /// # Returns
    /// Estimated number of rows, or None if stats are unavailable
    pub fn estimate_scan_rows(
        &self,
        table_name: &str,
        predicate: Option<&Expression>,
    ) -> Option<u64> {
        let table_stats = self.get_table_stats(table_name)?;
        let base_rows = table_stats.row_count;

        if base_rows == 0 {
            return Some(0);
        }

        let predicate = match predicate {
            Some(p) => p,
            None => return Some(base_rows), // Full table scan
        };

        // Estimate selectivity from the predicate
        let selectivity = self.estimate_predicate_selectivity(table_name, predicate, &table_stats);
        let estimated = ((base_rows as f64) * selectivity).max(1.0) as u64;

        // Apply feedback correction
        Some(self.estimate_with_feedback(table_name, Some(predicate), estimated))
    }

    /// Estimate selectivity of a predicate expression
    fn estimate_predicate_selectivity(
        &self,
        table_name: &str,
        expr: &Expression,
        table_stats: &TableStats,
    ) -> f64 {
        use crate::parser::ast::{InfixOperator, PrefixOperator};

        match expr {
            // Infix expressions (a AND b, a OR b, a = b, a IS NULL, etc.)
            Expression::Infix(infix) => {
                match infix.op_type {
                    // AND: multiply selectivities (assuming independence)
                    InfixOperator::And => {
                        let left_sel = self.estimate_predicate_selectivity(
                            table_name,
                            &infix.left,
                            table_stats,
                        );
                        let right_sel = self.estimate_predicate_selectivity(
                            table_name,
                            &infix.right,
                            table_stats,
                        );
                        left_sel * right_sel
                    }
                    // OR: use inclusion-exclusion principle
                    InfixOperator::Or => {
                        let left_sel = self.estimate_predicate_selectivity(
                            table_name,
                            &infix.left,
                            table_stats,
                        );
                        let right_sel = self.estimate_predicate_selectivity(
                            table_name,
                            &infix.right,
                            table_stats,
                        );
                        // P(A or B) = P(A) + P(B) - P(A and B)
                        (left_sel + right_sel - left_sel * right_sel).min(1.0)
                    }
                    // IS NULL
                    InfixOperator::Is => {
                        // Check if right side is NULL
                        if matches!(infix.right.as_ref(), Expression::NullLiteral(_)) {
                            let col_name = self.extract_column_name(&infix.left);
                            let col_stats =
                                col_name.and_then(|name| self.get_column_stats(table_name, &name));
                            self.estimate_selectivity(
                                Some(Operator::IsNull),
                                None,
                                col_stats.as_ref(),
                                table_stats,
                            )
                        } else {
                            0.5
                        }
                    }
                    // IS NOT NULL
                    InfixOperator::IsNot => {
                        if matches!(infix.right.as_ref(), Expression::NullLiteral(_)) {
                            let col_name = self.extract_column_name(&infix.left);
                            let col_stats =
                                col_name.and_then(|name| self.get_column_stats(table_name, &name));
                            self.estimate_selectivity(
                                Some(Operator::IsNotNull),
                                None,
                                col_stats.as_ref(),
                                table_stats,
                            )
                        } else {
                            0.5
                        }
                    }
                    // Comparison operators
                    InfixOperator::Equal => {
                        let col_name = self
                            .extract_column_name(&infix.left)
                            .or_else(|| self.extract_column_name(&infix.right));
                        let value = self
                            .extract_value(&infix.right)
                            .or_else(|| self.extract_value(&infix.left));
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Eq),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::NotEqual => {
                        let col_name = self
                            .extract_column_name(&infix.left)
                            .or_else(|| self.extract_column_name(&infix.right));
                        let value = self
                            .extract_value(&infix.right)
                            .or_else(|| self.extract_value(&infix.left));
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Ne),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::LessThan => {
                        let col_name = self.extract_column_name(&infix.left);
                        let value = self.extract_value(&infix.right);
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Lt),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::LessEqual => {
                        let col_name = self.extract_column_name(&infix.left);
                        let value = self.extract_value(&infix.right);
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Lte),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::GreaterThan => {
                        let col_name = self.extract_column_name(&infix.left);
                        let value = self.extract_value(&infix.right);
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Gt),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::GreaterEqual => {
                        let col_name = self.extract_column_name(&infix.left);
                        let value = self.extract_value(&infix.right);
                        let col_stats =
                            col_name.and_then(|name| self.get_column_stats(table_name, &name));
                        self.estimate_selectivity(
                            Some(Operator::Gte),
                            value.as_ref(),
                            col_stats.as_ref(),
                            table_stats,
                        )
                    }
                    InfixOperator::Like | InfixOperator::ILike => {
                        let pattern_str = self.extract_string_value(&infix.right);
                        match pattern_str {
                            Some(p) if !p.starts_with('%') => 0.1, // Prefix match is more selective
                            Some(_) => 0.25,                       // Suffix or contains
                            None => 0.25,
                        }
                    }
                    InfixOperator::NotLike | InfixOperator::NotILike => {
                        let pattern_str = self.extract_string_value(&infix.right);
                        let like_sel = match pattern_str {
                            Some(p) if !p.starts_with('%') => 0.1,
                            Some(_) => 0.25,
                            None => 0.25,
                        };
                        1.0 - like_sel
                    }
                    // Default for other operators
                    _ => 0.5,
                }
            }
            // IN expression
            Expression::In(in_expr) => {
                let col_name = self.extract_column_name(&in_expr.left);
                let col_stats = col_name.and_then(|name| self.get_column_stats(table_name, &name));

                // Get list size from the right side
                let list_size = match in_expr.right.as_ref() {
                    Expression::List(list) => list.elements.len() as f64,
                    Expression::ExpressionList(list) => list.expressions.len() as f64,
                    _ => 5.0, // Default assumption
                };
                let distinct = col_stats
                    .map(|s| s.distinct_count.max(1) as f64)
                    .unwrap_or(100.0);
                let in_selectivity = (list_size / distinct).min(1.0);

                if in_expr.not {
                    1.0 - in_selectivity
                } else {
                    in_selectivity
                }
            }
            // BETWEEN expression
            Expression::Between(between) => {
                let col_name = self.extract_column_name(&between.expr);
                let col_stats = col_name.and_then(|name| self.get_column_stats(table_name, &name));
                let low_val = self.extract_value(&between.lower);
                let high_val = self.extract_value(&between.upper);

                // Estimate as (high - low) / (max - min)
                let range_sel = if let (Some(ref stats), Some(low_v), Some(high_v)) =
                    (&col_stats, low_val, high_val)
                {
                    if let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) {
                        let low_pos = Self::estimate_value_position(&low_v, min, max);
                        let high_pos = Self::estimate_value_position(&high_v, min, max);
                        (high_pos - low_pos).abs().clamp(0.01, 0.99)
                    } else {
                        0.25 // Default BETWEEN selectivity
                    }
                } else {
                    0.25
                };

                if between.not {
                    1.0 - range_sel
                } else {
                    range_sel
                }
            }
            // LIKE expression (standalone)
            Expression::Like(like_expr) => {
                let is_negated = like_expr.operator.to_uppercase().contains("NOT");
                let pattern_str = self.extract_string_value(&like_expr.pattern);
                let base_sel = match pattern_str {
                    Some(p) if !p.starts_with('%') => 0.1,
                    Some(_) => 0.25,
                    None => 0.25,
                };
                if is_negated {
                    1.0 - base_sel
                } else {
                    base_sel
                }
            }
            // Prefix expressions (NOT x, -x)
            Expression::Prefix(prefix) => match prefix.op_type {
                PrefixOperator::Not => {
                    1.0 - self.estimate_predicate_selectivity(
                        table_name,
                        &prefix.right,
                        table_stats,
                    )
                }
                _ => 0.5,
            },
            // Unknown expressions - conservative estimate
            _ => 0.5,
        }
    }

    /// Extract column name from an expression
    fn extract_column_name(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value_lower.clone()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value_lower.clone()),
            _ => None,
        }
    }

    /// Extract a Value from a literal expression
    fn extract_value(&self, expr: &Expression) -> Option<Value> {
        match expr {
            Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
            Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
            Expression::StringLiteral(lit) => Some(Value::Text(lit.value.clone().into())),
            Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
            Expression::NullLiteral(_) => None, // NULL doesn't have a comparable value
            _ => None,
        }
    }

    /// Extract string value from an expression
    fn extract_string_value(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::StringLiteral(lit) => Some(lit.value.clone()),
            _ => None,
        }
    }

    // =========================================================================
    // Cardinality Feedback Integration
    // =========================================================================

    /// Estimate row count with cardinality feedback correction
    ///
    /// This method combines statistics-based estimation with learned corrections
    /// from previous query executions. When similar predicates have been executed
    /// before, the correction factor improves accuracy.
    ///
    /// # Arguments
    /// * `table_name` - Name of the table being scanned
    /// * `predicate` - The WHERE clause predicate (for fingerprinting)
    /// * `base_estimate` - Initial row count estimate from statistics
    ///
    /// # Returns
    /// Corrected row count estimate
    pub fn estimate_with_feedback(
        &self,
        table_name: &str,
        predicate: Option<&Expression>,
        base_estimate: u64,
    ) -> u64 {
        let predicate = match predicate {
            Some(p) => p,
            None => return base_estimate, // No predicate, no feedback
        };

        // Get fingerprint for this predicate pattern
        let fingerprint = fingerprint_predicate(table_name, predicate);

        // Look up and apply any learned correction
        let feedback_cache = global_feedback_cache();
        feedback_cache.apply_correction(table_name, fingerprint, base_estimate)
    }

    /// Record cardinality feedback after query execution
    ///
    /// This method stores the difference between estimated and actual row counts,
    /// enabling future queries with similar predicates to benefit from the correction.
    ///
    /// # Arguments
    /// * `table_name` - Name of the table that was scanned
    /// * `predicate` - The WHERE clause predicate (for fingerprinting)
    /// * `column_name` - Optional column name for more specific feedback
    /// * `estimated_rows` - Row count estimate used during planning
    /// * `actual_rows` - Actual row count observed during execution
    pub fn record_feedback(
        &self,
        table_name: &str,
        predicate: &Expression,
        column_name: Option<String>,
        estimated_rows: u64,
        actual_rows: u64,
    ) {
        // Only record if there's meaningful difference (avoid noise from perfect estimates)
        if estimated_rows == actual_rows {
            return;
        }

        // Only record if actual rows are significant (avoid learning from tiny results)
        if actual_rows < 10 && estimated_rows < 10 {
            return;
        }

        let fingerprint = fingerprint_predicate(table_name, predicate);
        let feedback_cache = global_feedback_cache();
        feedback_cache.record_feedback(
            table_name,
            fingerprint,
            column_name,
            estimated_rows,
            actual_rows,
        );
    }

    /// Get the correction factor for a predicate (for debugging/EXPLAIN)
    ///
    /// Returns 1.0 if no feedback is available or if feedback is not yet reliable.
    pub fn get_feedback_correction(&self, table_name: &str, predicate: &Expression) -> f64 {
        let fingerprint = fingerprint_predicate(table_name, predicate);
        let feedback_cache = global_feedback_cache();
        feedback_cache.get_correction(table_name, fingerprint)
    }
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

/// Runtime join algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeJoinAlgorithm {
    /// Hash join: O(N + M) - best for large unsorted inputs with equality keys
    HashJoin,
    /// Merge join: O(N + M) - best when inputs are already sorted on join keys
    MergeJoin,
    /// Nested loop: O(N * M) - best for small tables or no equality keys
    NestedLoop,
}

/// Runtime join decision result
#[derive(Debug, Clone)]
pub struct RuntimeJoinDecision {
    /// Selected join algorithm
    pub algorithm: RuntimeJoinAlgorithm,
    /// Should swap build/probe sides for better performance (hash join)
    /// or outer/inner sides (nested loop)
    pub swap_sides: bool,
    /// Explanation for logging/debugging
    pub explanation: String,
}

impl RuntimeJoinDecision {
    /// Check if hash join was selected
    pub fn use_hash_join(&self) -> bool {
        self.algorithm == RuntimeJoinAlgorithm::HashJoin
    }

    /// Check if merge join was selected
    pub fn use_merge_join(&self) -> bool {
        self.algorithm == RuntimeJoinAlgorithm::MergeJoin
    }

    /// Check if nested loop was selected
    pub fn use_nested_loop(&self) -> bool {
        self.algorithm == RuntimeJoinAlgorithm::NestedLoop
    }
}

impl QueryPlanner {
    /// Make a runtime join algorithm decision based on actual row counts
    ///
    /// This is called during execution with the actual materialized row counts,
    /// enabling adaptive decisions that account for runtime conditions.
    /// Also consults the EdgeAwarePlanner for workload-learned hints.
    ///
    /// # Arguments
    /// * `left_rows` - Actual row count from left side
    /// * `right_rows` - Actual row count from right side
    /// * `has_equality_keys` - Whether join has equality conditions (a.x = b.x)
    ///
    /// # Returns
    /// Decision on which algorithm to use and whether to swap sides
    pub fn plan_runtime_join(
        &self,
        left_rows: usize,
        right_rows: usize,
        has_equality_keys: bool,
    ) -> RuntimeJoinDecision {
        self.plan_runtime_join_with_sort_info(
            left_rows,
            right_rows,
            has_equality_keys,
            false,
            false,
        )
    }

    /// Make a runtime join algorithm decision with sort information
    ///
    /// Extended version that also considers whether inputs are pre-sorted,
    /// which enables merge join optimization.
    ///
    /// # Arguments
    /// * `left_rows` - Actual row count from left side
    /// * `right_rows` - Actual row count from right side
    /// * `has_equality_keys` - Whether join has equality conditions (a.x = b.x)
    /// * `left_sorted` - Whether left input is sorted on join keys
    /// * `right_sorted` - Whether right input is sorted on join keys
    pub fn plan_runtime_join_with_sort_info(
        &self,
        left_rows: usize,
        right_rows: usize,
        has_equality_keys: bool,
        left_sorted: bool,
        right_sorted: bool,
    ) -> RuntimeJoinDecision {
        // For small tables, nested loop is faster (no hash table overhead)
        // PostgreSQL uses similar thresholds
        const NESTED_LOOP_MAX: usize = 200;
        const HASH_JOIN_MIN_BENEFIT: usize = 50;
        const ESTIMATED_BYTES_PER_ROW: u64 = 100;
        // Merge join is preferred over hash when both inputs are sorted
        // and tables are large enough to benefit from avoiding hash overhead
        const MERGE_JOIN_MIN_ROWS: usize = 500;

        let total_rows = left_rows + right_rows;
        let product = left_rows.saturating_mul(right_rows);

        // Case 1: No equality keys - must use nested loop
        if !has_equality_keys {
            return RuntimeJoinDecision {
                algorithm: RuntimeJoinAlgorithm::NestedLoop,
                swap_sides: false,
                explanation: "Nested loop: no equality join keys".to_string(),
            };
        }

        // Consult EdgeAwarePlanner for workload-learned hints
        let edge_planner = EdgeAwarePlanner::from_global();
        let (build_rows_u64, probe_rows_u64) = if right_rows < left_rows {
            (right_rows as u64, left_rows as u64)
        } else {
            (left_rows as u64, right_rows as u64)
        };

        let edge_recommendation = edge_planner.recommend_join_for_edge(
            build_rows_u64,
            probe_rows_u64,
            ESTIMATED_BYTES_PER_ROW,
        );

        // Check if edge constraints force a specific algorithm
        match edge_recommendation {
            EdgeJoinRecommendation::ForceNestedLoop { reason } => {
                return RuntimeJoinDecision {
                    algorithm: RuntimeJoinAlgorithm::NestedLoop,
                    swap_sides: false,
                    explanation: format!("Nested loop (edge constraint): {}", reason),
                };
            }
            EdgeJoinRecommendation::PreferNestedLoop { reason } => {
                // Edge mode prefers nested loop - use lower threshold
                if left_rows <= NESTED_LOOP_MAX * 2 && right_rows <= NESTED_LOOP_MAX * 2 {
                    return RuntimeJoinDecision {
                        algorithm: RuntimeJoinAlgorithm::NestedLoop,
                        swap_sides: false,
                        explanation: format!("Nested loop (edge hint): {}", reason),
                    };
                }
            }
            EdgeJoinRecommendation::PreferHashJoin { .. } => {
                // Edge mode prefers hash join - skip nested loop checks for medium tables
                // But still consider merge join if both inputs are sorted
                if total_rows > NESTED_LOOP_MAX && !(left_sorted && right_sorted) {
                    let swap = right_rows < left_rows;
                    let (build, probe) = if swap {
                        (right_rows, left_rows)
                    } else {
                        (left_rows, right_rows)
                    };
                    return RuntimeJoinDecision {
                        algorithm: RuntimeJoinAlgorithm::HashJoin,
                        swap_sides: swap,
                        explanation: format!(
                            "Hash join (batch workload): build {} rows, probe {} rows",
                            build, probe
                        ),
                    };
                }
            }
            EdgeJoinRecommendation::UseDefault => {
                // Fall through to standard cost-based decision
            }
        }

        // Case 2: Both sides tiny - nested loop is faster
        if left_rows <= NESTED_LOOP_MAX && right_rows <= NESTED_LOOP_MAX {
            return RuntimeJoinDecision {
                algorithm: RuntimeJoinAlgorithm::NestedLoop,
                swap_sides: false,
                explanation: format!(
                    "Nested loop: both sides small ({}x{} = {} comparisons)",
                    left_rows, right_rows, product
                ),
            };
        }

        // Case 3: One side empty - no benefit from hash/merge join
        if left_rows == 0 || right_rows == 0 {
            return RuntimeJoinDecision {
                algorithm: RuntimeJoinAlgorithm::NestedLoop,
                swap_sides: false,
                explanation: "Nested loop: one side empty".to_string(),
            };
        }

        // Case 4: Merge join when both inputs are already sorted
        // Merge join is O(N + M) like hash join, but avoids hash table overhead
        // It's optimal when both inputs are pre-sorted on join keys
        if left_sorted && right_sorted && total_rows >= MERGE_JOIN_MIN_ROWS {
            return RuntimeJoinDecision {
                algorithm: RuntimeJoinAlgorithm::MergeJoin,
                swap_sides: false,
                explanation: format!(
                    "Merge join: both inputs sorted ({} + {} rows)",
                    left_rows, right_rows
                ),
            };
        }

        // Case 5: Hash join cost analysis
        // Hash join is O(N + M) vs nested loop O(N * M)
        // But hash join has setup cost, so only use when beneficial
        let hash_cost = total_rows as f64;
        let nested_cost = product as f64;

        if nested_cost < hash_cost + HASH_JOIN_MIN_BENEFIT as f64 {
            // Nested loop is cheaper even accounting for setup
            return RuntimeJoinDecision {
                algorithm: RuntimeJoinAlgorithm::NestedLoop,
                swap_sides: false,
                explanation: format!(
                    "Nested loop: cheaper than hash ({} < {} + setup)",
                    product, total_rows
                ),
            };
        }

        // Case 6: Use hash join with smaller side as build side
        let swap = right_rows < left_rows;
        let (build_rows, probe_rows) = if swap {
            (right_rows, left_rows)
        } else {
            (left_rows, right_rows)
        };

        RuntimeJoinDecision {
            algorithm: RuntimeJoinAlgorithm::HashJoin,
            swap_sides: swap,
            explanation: format!(
                "Hash join: build {} rows, probe {} rows (swap={})",
                build_rows, probe_rows, swap
            ),
        }
    }
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
}
