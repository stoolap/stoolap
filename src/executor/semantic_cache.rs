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

//! Semantic Query Caching with Predicate Subsumption
//!
//! This module implements intelligent query result caching that goes beyond simple
//! string matching. It detects when a new query's results can be computed by filtering
//! cached results from a previous query, without re-executing against storage.
//!
//! # Key Insight
//!
//! If Query A's predicate P_A is LESS RESTRICTIVE than Query B's predicate P_B,
//! then B's results are a SUBSET of A's results:
//!
//! ```text
//! Query A: SELECT * FROM orders WHERE amount > 100  (cached: 500 rows)
//! Query B: SELECT * FROM orders WHERE amount > 150  (new query)
//!
//! Since amount > 150 is STRICTER than amount > 100,
//! B's results ⊆ A's results
//!
//! Instead of scanning storage: Filter A's cached 500 rows → B's results
//! ```
//!
//! # Supported Subsumption Patterns
//!
//! 1. **Numeric Range Tightening:**
//!    - `col > 100` (cached) → `col > 150` (new): Filter cached
//!    - `col < 500` (cached) → `col < 300` (new): Filter cached
//!    - `col BETWEEN 100 AND 500` (cached) → `col BETWEEN 200 AND 400` (new): Filter cached
//!
//! 2. **Equality Subset:**
//!    - `col IN (1,2,3,4,5)` (cached) → `col IN (2,3)` (new): Filter cached
//!
//! 3. **AND Conjunction Strengthening:**
//!    - `A` (cached) → `A AND B` (new): Filter cached
//!    - `A AND B` (cached) → `A AND B AND C` (new): Filter cached
//!
//! # Not Supported (Triggers Re-execution)
//!
//! - OR predicates (may expand result set)
//! - Different tables
//! - Different column sets
//! - Non-comparable predicates (LIKE, function calls)
//!
//! # Transaction Isolation Considerations
//!
//! **Important:** The semantic cache is currently global and does not account for
//! MVCC transaction isolation. This means:
//!
//! - Cache entries are shared across all transactions
//! - A transaction might see cached results from another transaction's read
//! - Cache invalidation on DML ensures committed changes are reflected
//!
//! This is safe for:
//! - Single-connection usage
//! - Read-only workloads
//! - Scenarios where eventual consistency is acceptable
//!
//! For strict serializable isolation with concurrent writes, consider:
//! - Disabling the cache during critical transactions
//! - Using explicit cache invalidation between operations
//!
//! Future enhancement: Per-transaction cache scoping with timestamp-based invalidation

use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::core::{Result, Row};
use crate::functions::FunctionRegistry;
use crate::parser::ast::{Expression, InfixOperator};

use super::expression::ExpressionEval;
use super::utils::{expressions_equivalent, extract_and_conditions, extract_column_name};

/// Maximum number of cached query results per table+column combination.
///
/// This limits memory usage by bounding how many distinct query patterns
/// can be cached for each table. When this limit is reached, the least
/// recently used (LRU) entries are evicted.
///
/// Default: 64 entries
pub const DEFAULT_SEMANTIC_CACHE_SIZE: usize = 64;

/// Time-to-live for cached results in seconds.
///
/// Cached query results are automatically evicted after this duration
/// to prevent serving stale data. This provides a safety net beyond
/// explicit invalidation on DML operations.
///
/// Default: 300 seconds (5 minutes)
pub const DEFAULT_CACHE_TTL_SECS: u64 = 300;

/// Maximum number of rows to cache per query result.
///
/// Query results exceeding this threshold are not cached to prevent
/// memory bloat. This is particularly important for queries that may
/// return large result sets.
///
/// Default: 100,000 rows
pub const DEFAULT_MAX_CACHED_ROWS: usize = 100_000;

/// Global maximum total rows across all cache entries.
///
/// This prevents unbounded memory growth when many tables/column patterns
/// are cached. When exceeded, entries are evicted using LRU across all
/// tables until under the limit.
///
/// Default: 1,000,000 rows (approximately 100-500MB depending on row size)
pub const DEFAULT_MAX_GLOBAL_CACHED_ROWS: usize = 1_000_000;

/// Fingerprint for a cacheable query pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryFingerprint {
    /// Table name (lowercase)
    pub table_name: String,
    /// Selected columns (sorted for comparison)
    pub columns: Vec<String>,
    /// Hash of the predicate structure (not values)
    pub predicate_structure_hash: u64,
}

impl QueryFingerprint {
    /// Create a fingerprint for a simple table scan
    pub fn new(table_name: &str, columns: Vec<String>) -> Self {
        Self {
            table_name: table_name.to_lowercase(),
            columns,
            predicate_structure_hash: 0,
        }
    }

    /// Create a fingerprint with predicate structure
    pub fn with_predicate(table_name: &str, columns: Vec<String>, predicate: &Expression) -> Self {
        Self {
            table_name: table_name.to_lowercase(),
            columns,
            predicate_structure_hash: hash_predicate_structure(predicate),
        }
    }
}

/// Cached query result with metadata
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// The fingerprint identifying this query pattern
    pub fingerprint: QueryFingerprint,
    /// Column names in order
    pub column_names: Vec<String>,
    /// Cached rows (Arc for zero-copy sharing on cache hits)
    pub rows: Arc<Vec<Row>>,
    /// The original WHERE predicate (for subsumption checking)
    pub predicate: Option<Expression>,
    /// When this entry was cached
    pub cached_at: Instant,
    /// Last access time (for LRU eviction)
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
}

impl CachedResult {
    /// Create a new cached result
    pub fn new(
        fingerprint: QueryFingerprint,
        column_names: Vec<String>,
        rows: Vec<Row>,
        predicate: Option<Expression>,
    ) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            column_names,
            rows: Arc::new(rows), // Wrap in Arc for zero-copy sharing
            predicate,
            cached_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    /// Create a new cached result with pre-wrapped Arc (avoids clone)
    ///
    /// Use this when the caller already has an Arc<Vec<Row>> to avoid
    /// an extra allocation and copy.
    pub fn new_with_arc(
        fingerprint: QueryFingerprint,
        column_names: Vec<String>,
        rows: Arc<Vec<Row>>,
        predicate: Option<Expression>,
    ) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            column_names,
            rows, // Use Arc directly - no additional wrapping
            predicate,
            cached_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    /// Check if this cached result has expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }

    /// Record an access to this cached result
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Subsumption relationship between predicates
#[derive(Debug, Clone)]
pub enum SubsumptionResult {
    /// New predicate is stricter - can filter cached results
    Subsumed {
        /// The additional filter to apply to cached rows
        filter: Box<Expression>,
    },
    /// Predicates are identical - use cached results directly
    Identical,
    /// Cannot determine subsumption - must re-execute
    NoSubsumption,
}

/// Semantic Query Cache
///
/// Intelligently caches query results and detects when new queries
/// can be answered by filtering cached results.
pub struct SemanticCache {
    /// Cached results: table_name -> (column_key -> Vec<CachedResult>)
    /// Nested structure enables O(1) table invalidation
    cache: RwLock<FxHashMap<String, FxHashMap<String, Vec<CachedResult>>>>,
    /// Maximum cache size per table+column combination
    max_size: usize,
    /// Cache TTL
    ttl: Duration,
    /// Maximum rows to cache per query
    max_rows: usize,
    /// Maximum total rows across all cache entries (prevents unbounded growth)
    max_global_rows: usize,
    /// Current total row count across all entries (for global limit enforcement)
    global_row_count: AtomicU64,
    /// Statistics (lock-free atomics)
    stats: SemanticCacheStats,
}

/// Statistics for the semantic cache (lock-free with atomics)
#[derive(Debug, Default)]
pub struct SemanticCacheStats {
    /// Total cache hits (exact or subsumption)
    pub hits: AtomicU64,
    /// Exact match hits
    pub exact_hits: AtomicU64,
    /// Subsumption hits (filtered from cached)
    pub subsumption_hits: AtomicU64,
    /// Cache misses
    pub misses: AtomicU64,
    /// Entries evicted due to TTL
    pub ttl_evictions: AtomicU64,
    /// Entries evicted due to size limit
    pub size_evictions: AtomicU64,
    /// Lock acquisition failures (poisoned lock from panics)
    /// If this is non-zero, a previous operation panicked while holding the lock
    pub lock_failures: AtomicU64,
}

/// Snapshot of cache statistics (plain values for reading)
#[derive(Debug, Clone, Default)]
pub struct SemanticCacheStatsSnapshot {
    /// Total cache hits (exact or subsumption)
    pub hits: u64,
    /// Exact match hits
    pub exact_hits: u64,
    /// Subsumption hits (filtered from cached)
    pub subsumption_hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Entries evicted due to TTL
    pub ttl_evictions: u64,
    /// Entries evicted due to size limit
    pub size_evictions: u64,
    /// Lock acquisition failures (indicates previous panic)
    pub lock_failures: u64,
}

/// Result of a cache lookup
#[derive(Debug)]
pub enum CacheLookupResult {
    /// Exact match found (Arc for zero-copy sharing)
    ExactHit(Arc<Vec<Row>>),
    /// Subsumption match found - apply filter to get results
    SubsumptionHit {
        /// Rows to filter (Arc for zero-copy sharing)
        rows: Arc<Vec<Row>>,
        /// Filter predicate to apply
        filter: Box<Expression>,
        /// Column names for evaluation context
        columns: Vec<String>,
    },
    /// No match found
    Miss,
}

impl SemanticCache {
    /// Create a new semantic cache with default settings
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_SEMANTIC_CACHE_SIZE,
            Duration::from_secs(DEFAULT_CACHE_TTL_SECS),
            DEFAULT_MAX_CACHED_ROWS,
            DEFAULT_MAX_GLOBAL_CACHED_ROWS,
        )
    }

    /// Create a semantic cache with custom configuration
    pub fn with_config(
        max_size: usize,
        ttl: Duration,
        max_rows: usize,
        max_global_rows: usize,
    ) -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
            max_size,
            ttl,
            max_rows,
            max_global_rows,
            global_row_count: AtomicU64::new(0),
            stats: SemanticCacheStats::default(),
        }
    }

    /// Look up a query in the cache
    ///
    /// Returns:
    /// - `ExactHit` if an identical query is cached
    /// - `SubsumptionHit` if a broader query is cached and can be filtered
    /// - `Miss` if no usable cache entry exists
    pub fn lookup(
        &self,
        table_name: &str,
        columns: &[String],
        predicate: Option<&Expression>,
    ) -> CacheLookupResult {
        let (table_key, column_key) = Self::cache_keys(table_name, columns);

        // First pass: read-only search
        let hit_info = {
            let cache = match self.cache.read() {
                Ok(c) => c,
                Err(_) => {
                    self.stats.lock_failures.fetch_add(1, Ordering::Relaxed);
                    return CacheLookupResult::Miss;
                }
            };

            // Navigate nested structure: table -> columns -> entries
            let table_cache = match cache.get(&table_key) {
                Some(tc) => tc,
                None => {
                    drop(cache);
                    self.record_miss();
                    return CacheLookupResult::Miss;
                }
            };

            let entries = match table_cache.get(&column_key) {
                Some(e) => e,
                None => {
                    drop(cache);
                    self.record_miss();
                    return CacheLookupResult::Miss;
                }
            };

            // Try to find a usable cached result
            // Store (index, predicate_hash) for TOCTOU-safe access update
            let mut found = None;
            for (idx, entry) in entries.iter().enumerate() {
                // Skip expired entries
                if entry.is_expired(self.ttl) {
                    continue;
                }

                // Check if columns match
                if entry.column_names != columns {
                    continue;
                }

                // Check predicate relationship
                match check_subsumption(entry.predicate.as_ref(), predicate) {
                    SubsumptionResult::Identical => {
                        let rows = entry.rows.clone();
                        let hash = entry.fingerprint.predicate_structure_hash;
                        found = Some((idx, hash, CacheLookupResult::ExactHit(rows)));
                        break;
                    }
                    SubsumptionResult::Subsumed { filter } => {
                        let rows = entry.rows.clone();
                        let columns = entry.column_names.clone();
                        let hash = entry.fingerprint.predicate_structure_hash;
                        found = Some((
                            idx,
                            hash,
                            CacheLookupResult::SubsumptionHit {
                                rows,
                                filter,
                                columns,
                            },
                        ));
                        break;
                    }
                    SubsumptionResult::NoSubsumption => {
                        // Try next entry
                        continue;
                    }
                }
            }
            found
        }; // Read lock released here

        match hit_info {
            Some((idx, expected_hash, result)) => {
                // Update access time with write lock
                // TOCTOU safety: verify the entry's fingerprint hash matches
                // If entry was evicted/replaced, skip update (benign miss)
                if let Ok(mut cache) = self.cache.write() {
                    if let Some(table_cache) = cache.get_mut(&table_key) {
                        if let Some(entries) = table_cache.get_mut(&column_key) {
                            if let Some(entry) = entries.get_mut(idx) {
                                // Only update if it's still the same entry
                                if entry.fingerprint.predicate_structure_hash == expected_hash {
                                    entry.record_access();
                                }
                            }
                        }
                    }
                }
                // Record hit stats
                match &result {
                    CacheLookupResult::ExactHit(_) => self.record_exact_hit(),
                    CacheLookupResult::SubsumptionHit { .. } => self.record_subsumption_hit(),
                    CacheLookupResult::Miss => {}
                }
                result
            }
            None => {
                self.record_miss();
                CacheLookupResult::Miss
            }
        }
    }

    /// Insert a query result into the cache
    pub fn insert(
        &self,
        table_name: &str,
        columns: Vec<String>,
        rows: Vec<Row>,
        predicate: Option<Expression>,
    ) {
        let new_row_count = rows.len();

        // Don't cache if too many rows in this single result
        if new_row_count > self.max_rows {
            return;
        }

        let (table_key, column_key) = Self::cache_keys(table_name, &columns);
        let fingerprint = match &predicate {
            Some(p) => QueryFingerprint::with_predicate(table_name, columns.clone(), p),
            None => QueryFingerprint::new(table_name, columns.clone()),
        };

        let entry = CachedResult::new(fingerprint, columns, rows, predicate);

        let mut cache = match self.cache.write() {
            Ok(c) => c,
            Err(_) => {
                self.stats.lock_failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        // Check global limit and evict across other tables first if needed
        let current_global = self.global_row_count.load(Ordering::Relaxed) as usize;
        if current_global + new_row_count > self.max_global_rows {
            let rows_to_free =
                (current_global + new_row_count).saturating_sub(self.max_global_rows);
            self.evict_global_lru(&mut cache, rows_to_free, &table_key);
        }

        // Navigate to nested entries: table -> columns -> entries
        let table_cache = cache.entry(table_key).or_default();
        let entries = table_cache.entry(column_key).or_default();

        // Evict expired entries and track row count reduction
        let mut rows_freed: usize = 0;
        let before_len = entries.len();
        entries.retain(|e| {
            if e.is_expired(self.ttl) {
                rows_freed += e.rows.len();
                false
            } else {
                true
            }
        });
        let evicted = before_len - entries.len();
        if evicted > 0 {
            self.stats
                .ttl_evictions
                .fetch_add(evicted as u64, Ordering::Relaxed);
        }

        // Evict if at per-table capacity (LRU)
        while entries.len() >= self.max_size {
            // Find least recently used
            if let Some((idx, _)) = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| (e.last_accessed, e.access_count))
            {
                rows_freed += entries[idx].rows.len();
                entries.remove(idx);
                self.stats.size_evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }

        // Update global row count (subtract freed, add new)
        if rows_freed > 0 {
            self.global_row_count
                .fetch_sub(rows_freed as u64, Ordering::Relaxed);
        }

        // Add the new entry and update global count
        self.global_row_count
            .fetch_add(new_row_count as u64, Ordering::Relaxed);
        entries.push(entry);
    }

    /// Insert a query result into the cache using a pre-wrapped Arc
    ///
    /// This variant allows sharing the Arc between the cache and the caller,
    /// avoiding a full Vec clone when both need access to the same data.
    pub fn insert_arc(
        &self,
        table_name: &str,
        columns: Vec<String>,
        rows: Arc<Vec<Row>>,
        predicate: Option<Expression>,
    ) {
        let new_row_count = rows.len();

        // Don't cache if too many rows in this single result
        if new_row_count > self.max_rows {
            return;
        }

        let (table_key, column_key) = Self::cache_keys(table_name, &columns);
        let fingerprint = match &predicate {
            Some(p) => QueryFingerprint::with_predicate(table_name, columns.clone(), p),
            None => QueryFingerprint::new(table_name, columns.clone()),
        };

        let entry = CachedResult::new_with_arc(fingerprint, columns, rows, predicate);

        let mut cache = match self.cache.write() {
            Ok(c) => c,
            Err(_) => {
                self.stats.lock_failures.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        // Check global limit and evict across other tables first if needed
        let current_global = self.global_row_count.load(Ordering::Relaxed) as usize;
        if current_global + new_row_count > self.max_global_rows {
            let rows_to_free =
                (current_global + new_row_count).saturating_sub(self.max_global_rows);
            self.evict_global_lru(&mut cache, rows_to_free, &table_key);
        }

        // Navigate to nested entries: table -> columns -> entries
        let table_cache = cache.entry(table_key).or_default();
        let entries = table_cache.entry(column_key).or_default();

        // Evict expired entries and track row count reduction
        let mut rows_freed: usize = 0;
        let before_len = entries.len();
        entries.retain(|e| {
            if e.is_expired(self.ttl) {
                rows_freed += e.rows.len();
                false
            } else {
                true
            }
        });
        let evicted = before_len - entries.len();
        if evicted > 0 {
            self.stats
                .ttl_evictions
                .fetch_add(evicted as u64, Ordering::Relaxed);
        }

        // Evict if at per-table capacity (LRU)
        while entries.len() >= self.max_size {
            if let Some((idx, _)) = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| (e.last_accessed, e.access_count))
            {
                rows_freed += entries[idx].rows.len();
                entries.remove(idx);
                self.stats.size_evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }

        // Update global row count (subtract freed, add new)
        if rows_freed > 0 {
            self.global_row_count
                .fetch_sub(rows_freed as u64, Ordering::Relaxed);
        }

        // Add the new entry and update global count
        self.global_row_count
            .fetch_add(new_row_count as u64, Ordering::Relaxed);
        entries.push(entry);
    }

    /// Evict entries across all tables using global LRU until rows_to_free rows are freed
    fn evict_global_lru(
        &self,
        cache: &mut FxHashMap<String, FxHashMap<String, Vec<CachedResult>>>,
        mut rows_to_free: usize,
        skip_table: &str,
    ) {
        while rows_to_free > 0 {
            // Find the globally oldest entry across all tables
            let mut oldest: Option<(String, String, usize, Instant, u64, usize)> = None;

            for (table_key, table_cache) in cache.iter() {
                if table_key == skip_table {
                    continue; // Don't evict from the table we're inserting into
                }
                for (col_key, entries) in table_cache.iter() {
                    for (idx, entry) in entries.iter().enumerate() {
                        let dominated = match &oldest {
                            None => true,
                            Some((_, _, _, last_acc, acc_count, _)) => {
                                (entry.last_accessed, entry.access_count) < (*last_acc, *acc_count)
                            }
                        };
                        if dominated {
                            oldest = Some((
                                table_key.clone(),
                                col_key.clone(),
                                idx,
                                entry.last_accessed,
                                entry.access_count,
                                entry.rows.len(),
                            ));
                        }
                    }
                }
            }

            match oldest {
                Some((table_key, col_key, idx, _, _, row_count)) => {
                    if let Some(table_cache) = cache.get_mut(&table_key) {
                        if let Some(entries) = table_cache.get_mut(&col_key) {
                            entries.remove(idx);
                            self.global_row_count
                                .fetch_sub(row_count as u64, Ordering::Relaxed);
                            self.stats.size_evictions.fetch_add(1, Ordering::Relaxed);
                            rows_to_free = rows_to_free.saturating_sub(row_count);

                            // Clean up empty structures
                            if entries.is_empty() {
                                table_cache.remove(&col_key);
                            }
                        }
                        if table_cache.is_empty() {
                            cache.remove(&table_key);
                        }
                    }
                }
                None => break, // No more entries to evict
            }
        }
    }

    /// Invalidate all cache entries for a table (O(1) operation)
    pub fn invalidate_table(&self, table_name: &str) {
        let table_key = table_name.to_lowercase();
        match self.cache.write() {
            Ok(mut cache) => {
                // Count rows being removed for global tracking
                if let Some(table_cache) = cache.get(&table_key) {
                    let rows_removed: usize = table_cache
                        .values()
                        .flat_map(|entries| entries.iter())
                        .map(|e| e.rows.len())
                        .sum();
                    if rows_removed > 0 {
                        self.global_row_count
                            .fetch_sub(rows_removed as u64, Ordering::Relaxed);
                    }
                }
                // O(1) removal: just remove the entire table entry
                cache.remove(&table_key);
            }
            Err(_) => {
                self.stats.lock_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Clear the entire cache and reset statistics
    pub fn clear(&self) {
        match self.cache.write() {
            Ok(mut cache) => cache.clear(),
            Err(_) => {
                self.stats.lock_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Reset global row count
        self.global_row_count.store(0, Ordering::Relaxed);
        // Reset all atomic stats counters
        self.stats.hits.store(0, Ordering::Relaxed);
        self.stats.exact_hits.store(0, Ordering::Relaxed);
        self.stats.subsumption_hits.store(0, Ordering::Relaxed);
        self.stats.misses.store(0, Ordering::Relaxed);
        self.stats.ttl_evictions.store(0, Ordering::Relaxed);
        self.stats.size_evictions.store(0, Ordering::Relaxed);
        // Note: lock_failures is NOT reset - it indicates historical panics
    }

    /// Get cache statistics as a snapshot
    pub fn stats(&self) -> SemanticCacheStatsSnapshot {
        SemanticCacheStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            exact_hits: self.stats.exact_hits.load(Ordering::Relaxed),
            subsumption_hits: self.stats.subsumption_hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            ttl_evictions: self.stats.ttl_evictions.load(Ordering::Relaxed),
            size_evictions: self.stats.size_evictions.load(Ordering::Relaxed),
            lock_failures: self.stats.lock_failures.load(Ordering::Relaxed),
        }
    }

    /// Get the number of cached entries
    pub fn size(&self) -> usize {
        self.cache
            .read()
            .map(|c| {
                // Sum entries across all tables and column combinations
                c.values()
                    .map(|table_cache| table_cache.values().map(|v| v.len()).sum::<usize>())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Filter cached rows using a predicate
    ///
    /// This is used when a subsumption match is found to filter
    /// the broader cached result down to the stricter query's result.
    ///
    /// CRITICAL: This function now returns Result to properly propagate compilation errors.
    /// Previously, compilation failures silently returned unfiltered data which was incorrect.
    pub fn filter_rows(
        rows: Vec<Row>,
        filter: &Expression,
        columns: &[String],
        _function_registry: &FunctionRegistry,
    ) -> Result<Vec<Row>> {
        let columns_vec: Vec<String> = columns.to_vec();
        // CRITICAL: Propagate compilation errors instead of returning unfiltered data
        let mut eval = ExpressionEval::compile(filter, &columns_vec)?;

        Ok(rows.into_iter().filter(|row| eval.eval_bool(row)).collect())
    }

    // Private helpers

    /// Returns (table_key, column_key) for the nested cache structure
    fn cache_keys(table_name: &str, columns: &[String]) -> (String, String) {
        // Use null byte as delimiter for column key since it's invalid in SQL identifiers
        // This prevents collision between columns like ["a,b", "c"] and ["a", "b,c"]
        let table_key = table_name.to_lowercase();
        let mut sorted_cols = columns.to_vec();
        sorted_cols.sort();
        let column_key = sorted_cols.join("\0");
        (table_key, column_key)
    }

    fn record_exact_hit(&self) {
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_subsumption_hit(&self) {
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        self.stats.subsumption_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for SemanticCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash the structure of a predicate (ignoring literal values)
///
/// This allows matching predicates like:
/// - `col > 100` and `col > 200` (same structure, different values)
fn hash_predicate_structure(expr: &Expression) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_expr_structure(expr, &mut hasher);
    hasher.finish()
}

fn hash_expr_structure(expr: &Expression, hasher: &mut DefaultHasher) {
    match expr {
        Expression::Identifier(ident) => {
            0u8.hash(hasher);
            ident.value_lower.hash(hasher);
        }
        Expression::QualifiedIdentifier(qi) => {
            1u8.hash(hasher);
            qi.qualifier.value_lower.hash(hasher);
            qi.name.value_lower.hash(hasher);
        }
        Expression::IntegerLiteral(_) => {
            2u8.hash(hasher);
        }
        Expression::FloatLiteral(_) => {
            3u8.hash(hasher);
        }
        Expression::StringLiteral(_) => {
            4u8.hash(hasher);
        }
        Expression::BooleanLiteral(_) => {
            5u8.hash(hasher);
        }
        Expression::NullLiteral(_) => {
            6u8.hash(hasher);
        }
        Expression::Infix(infix) => {
            7u8.hash(hasher);
            format!("{:?}", infix.op_type).hash(hasher);
            hash_expr_structure(&infix.left, hasher);
            hash_expr_structure(&infix.right, hasher);
        }
        Expression::Prefix(prefix) => {
            8u8.hash(hasher);
            prefix.operator.hash(hasher);
            hash_expr_structure(&prefix.right, hasher);
        }
        Expression::Between(between) => {
            9u8.hash(hasher);
            hash_expr_structure(&between.expr, hasher);
        }
        Expression::In(in_expr) => {
            10u8.hash(hasher);
            hash_expr_structure(&in_expr.left, hasher);
        }
        Expression::FunctionCall(func) => {
            11u8.hash(hasher);
            func.function.to_lowercase().hash(hasher);
            func.arguments.len().hash(hasher);
        }
        Expression::Case(_) => {
            12u8.hash(hasher);
        }
        Expression::List(list) => {
            13u8.hash(hasher);
            list.elements.len().hash(hasher);
        }
        _ => {
            // Default case for other expressions
            255u8.hash(hasher);
        }
    }
}

/// Check if new_predicate is subsumed by cached_predicate
///
/// Returns:
/// - `Identical` if predicates are semantically equivalent
/// - `Subsumed { filter }` if new_predicate is stricter than cached_predicate
/// - `NoSubsumption` if we can't determine a subsumption relationship
pub fn check_subsumption(
    cached_predicate: Option<&Expression>,
    new_predicate: Option<&Expression>,
) -> SubsumptionResult {
    match (cached_predicate, new_predicate) {
        // Both None - identical (full table scan)
        (None, None) => SubsumptionResult::Identical,

        // Cached has predicate, new has none - new is broader (cannot use cache)
        (Some(_), None) => SubsumptionResult::NoSubsumption,

        // Cached has no predicate (full scan), new has predicate
        // New is stricter - can filter cached full scan
        (None, Some(new_pred)) => SubsumptionResult::Subsumed {
            filter: Box::new(new_pred.clone()),
        },

        // Both have predicates - analyze relationship
        (Some(cached), Some(new)) => check_predicate_subsumption(cached, new),
    }
}

/// Analyze predicate subsumption for two predicates
fn check_predicate_subsumption(cached: &Expression, new: &Expression) -> SubsumptionResult {
    // First check if predicates are structurally identical
    if expressions_equivalent(cached, new) {
        return SubsumptionResult::Identical;
    }

    // Check for range tightening: cached is broader range
    if let Some(result) = check_range_subsumption(cached, new) {
        return result;
    }

    // Check for AND strengthening: new adds more conditions
    if let Some(result) = check_and_subsumption(cached, new) {
        return result;
    }

    // Check for IN list subsumption: new has smaller IN list
    if let Some(result) = check_in_subsumption(cached, new) {
        return result;
    }

    SubsumptionResult::NoSubsumption
}

/// Check for numeric range subsumption
///
/// Examples:
/// - cached: `col > 100`, new: `col > 150` → Subsumed (new is stricter)
/// - cached: `col < 500`, new: `col < 300` → Subsumed (new is stricter)
fn check_range_subsumption(cached: &Expression, new: &Expression) -> Option<SubsumptionResult> {
    // Both must be infix comparisons
    let (cached_infix, new_infix) = match (cached, new) {
        (Expression::Infix(c), Expression::Infix(n)) => (c, n),
        _ => return None,
    };

    // Must be comparing same column to a literal
    let cached_col = extract_column_name(&cached_infix.left)?;
    let new_col = extract_column_name(&new_infix.left)?;

    if cached_col.to_lowercase() != new_col.to_lowercase() {
        return None;
    }

    let cached_val = extract_numeric_value(&cached_infix.right)?;
    let new_val = extract_numeric_value(&new_infix.right)?;

    // Check operator relationship
    match (&cached_infix.op_type, &new_infix.op_type) {
        // Greater than: new > cached means new is stricter
        (InfixOperator::GreaterThan, InfixOperator::GreaterThan)
        | (InfixOperator::GreaterThan, InfixOperator::GreaterEqual) => {
            if new_val > cached_val {
                Some(SubsumptionResult::Subsumed {
                    filter: Box::new(new.clone()),
                })
            } else if (new_val - cached_val).abs() < f64::EPSILON {
                Some(SubsumptionResult::Identical)
            } else {
                None
            }
        }

        (InfixOperator::GreaterEqual, InfixOperator::GreaterThan)
        | (InfixOperator::GreaterEqual, InfixOperator::GreaterEqual) => {
            if new_val >= cached_val {
                if (new_val - cached_val).abs() < f64::EPSILON
                    && matches!(cached_infix.op_type, InfixOperator::GreaterEqual)
                    && matches!(new_infix.op_type, InfixOperator::GreaterEqual)
                {
                    Some(SubsumptionResult::Identical)
                } else {
                    Some(SubsumptionResult::Subsumed {
                        filter: Box::new(new.clone()),
                    })
                }
            } else {
                None
            }
        }

        // Less than: new < cached means new is stricter
        (InfixOperator::LessThan, InfixOperator::LessThan)
        | (InfixOperator::LessThan, InfixOperator::LessEqual) => {
            if new_val < cached_val {
                Some(SubsumptionResult::Subsumed {
                    filter: Box::new(new.clone()),
                })
            } else if (new_val - cached_val).abs() < f64::EPSILON {
                Some(SubsumptionResult::Identical)
            } else {
                None
            }
        }

        (InfixOperator::LessEqual, InfixOperator::LessThan)
        | (InfixOperator::LessEqual, InfixOperator::LessEqual) => {
            if new_val <= cached_val {
                if (new_val - cached_val).abs() < f64::EPSILON
                    && matches!(cached_infix.op_type, InfixOperator::LessEqual)
                    && matches!(new_infix.op_type, InfixOperator::LessEqual)
                {
                    Some(SubsumptionResult::Identical)
                } else {
                    Some(SubsumptionResult::Subsumed {
                        filter: Box::new(new.clone()),
                    })
                }
            } else {
                None
            }
        }

        // Equality: values must match for identity
        (InfixOperator::Equal, InfixOperator::Equal) => {
            if (new_val - cached_val).abs() < f64::EPSILON {
                Some(SubsumptionResult::Identical)
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Check for AND conjunction subsumption
///
/// If new = cached AND extra_condition, then new is subsumed
fn check_and_subsumption(cached: &Expression, new: &Expression) -> Option<SubsumptionResult> {
    // New must be an AND
    let new_infix = match new {
        Expression::Infix(infix) if matches!(infix.op_type, InfixOperator::And) => infix,
        _ => return None,
    };

    // Check if cached is equivalent to one side of the AND
    if expressions_equivalent(cached, &new_infix.left)
        || expressions_equivalent(cached, &new_infix.right)
    {
        // New is cached AND something_else → new is stricter
        return Some(SubsumptionResult::Subsumed {
            filter: Box::new(new.clone()),
        });
    }

    // Check if cached is also an AND and new extends it
    if let Expression::Infix(cached_infix) = cached {
        if matches!(cached_infix.op_type, InfixOperator::And) {
            // Extract conditions from both
            let cached_conditions = extract_and_conditions(cached);
            let new_conditions = extract_and_conditions(new);

            // New must contain all of cached's conditions
            let all_cached_present = cached_conditions.iter().all(|cc| {
                new_conditions
                    .iter()
                    .any(|nc| expressions_equivalent(cc, nc))
            });

            if all_cached_present && new_conditions.len() > cached_conditions.len() {
                return Some(SubsumptionResult::Subsumed {
                    filter: Box::new(new.clone()),
                });
            }
        }
    }

    None
}

/// Check for IN list subsumption
///
/// If new IN list is subset of cached IN list, new is subsumed
fn check_in_subsumption(cached: &Expression, new: &Expression) -> Option<SubsumptionResult> {
    let (cached_in, new_in) = match (cached, new) {
        (Expression::In(c), Expression::In(n)) => (c, n),
        _ => return None,
    };

    // Must be same column
    if !expressions_equivalent(&cached_in.left, &new_in.left) {
        return None;
    }

    // Extract values from both IN lists
    let cached_values = extract_in_values(&cached_in.right)?;
    let new_values = extract_in_values(&new_in.right)?;

    // Check if new is subset of cached
    let is_subset = new_values
        .iter()
        .all(|nv| cached_values.iter().any(|cv| values_equal(cv, nv)));

    if is_subset {
        if new_values.len() == cached_values.len() {
            Some(SubsumptionResult::Identical)
        } else {
            Some(SubsumptionResult::Subsumed {
                filter: Box::new(new.clone()),
            })
        }
    } else {
        None
    }
}

/// Numeric value extracted from a literal
#[derive(Debug, Clone, Copy)]
enum NumericValue {
    Integer(i64),
    Float(f64),
}

impl NumericValue {
    fn as_f64(&self) -> f64 {
        match self {
            NumericValue::Integer(i) => *i as f64,
            NumericValue::Float(f) => *f,
        }
    }
}

impl PartialEq for NumericValue {
    fn eq(&self, other: &Self) -> bool {
        // For database predicate comparison, exact integer comparison is preferred,
        // but for floats we use a reasonable tolerance (1e-10) rather than f64::EPSILON
        // which is too strict (~2.2e-16) and can cause false negatives.
        match (self, other) {
            (NumericValue::Integer(a), NumericValue::Integer(b)) => a == b,
            _ => (self.as_f64() - other.as_f64()).abs() < 1e-10,
        }
    }
}

/// Extract values from an IN expression's right side
fn extract_in_values(expr: &Expression) -> Option<Vec<NumericValue>> {
    match expr {
        Expression::List(list) => Some(
            list.elements
                .iter()
                .filter_map(|e| match e {
                    Expression::IntegerLiteral(lit) => Some(NumericValue::Integer(lit.value)),
                    Expression::FloatLiteral(lit) => Some(NumericValue::Float(lit.value)),
                    _ => None,
                })
                .collect(),
        ),
        Expression::ExpressionList(list) => Some(
            list.expressions
                .iter()
                .filter_map(|e| match e {
                    Expression::IntegerLiteral(lit) => Some(NumericValue::Integer(lit.value)),
                    Expression::FloatLiteral(lit) => Some(NumericValue::Float(lit.value)),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Compare two NumericValues for equality
fn values_equal(a: &NumericValue, b: &NumericValue) -> bool {
    a == b
}

/// Extract numeric value from a literal expression
fn extract_numeric_value(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(lit.value as f64),
        Expression::FloatLiteral(lit) => Some(lit.value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use crate::parser::ast::{Identifier, InExpression, InfixExpression, ListExpression};
    use crate::parser::token::{Position, Token, TokenType};

    fn make_token() -> Token {
        Token {
            token_type: TokenType::Integer,
            literal: "".to_string(),
            position: Position::new(0, 1, 1),
            error: None,
        }
    }

    fn make_identifier(name: &str) -> Expression {
        Expression::Identifier(Identifier::new(make_token(), name.to_string()))
    }

    fn make_int_literal(val: i64) -> Expression {
        Expression::IntegerLiteral(crate::parser::ast::IntegerLiteral {
            token: make_token(),
            value: val,
        })
    }

    fn make_gt(col: &str, val: i64) -> Expression {
        Expression::Infix(InfixExpression::new(
            make_token(),
            Box::new(make_identifier(col)),
            ">".to_string(),
            Box::new(make_int_literal(val)),
        ))
    }

    fn make_lt(col: &str, val: i64) -> Expression {
        Expression::Infix(InfixExpression::new(
            make_token(),
            Box::new(make_identifier(col)),
            "<".to_string(),
            Box::new(make_int_literal(val)),
        ))
    }

    fn make_and(left: Expression, right: Expression) -> Expression {
        Expression::Infix(InfixExpression::new(
            make_token(),
            Box::new(left),
            "AND".to_string(),
            Box::new(right),
        ))
    }

    fn make_in(col: &str, values: Vec<i64>) -> Expression {
        Expression::In(InExpression {
            token: make_token(),
            left: Box::new(make_identifier(col)),
            right: Box::new(Expression::List(ListExpression {
                token: make_token(),
                elements: values.into_iter().map(make_int_literal).collect(),
            })),
            not: false,
        })
    }

    #[test]
    fn test_identical_predicates() {
        let pred1 = make_gt("amount", 100);
        let pred2 = make_gt("amount", 100);

        match check_subsumption(Some(&pred1), Some(&pred2)) {
            SubsumptionResult::Identical => {}
            other => panic!("Expected Identical, got {:?}", other),
        }
    }

    #[test]
    fn test_range_subsumption_greater_than() {
        // cached: amount > 100, new: amount > 150
        let cached = make_gt("amount", 100);
        let new = make_gt("amount", 150);

        match check_subsumption(Some(&cached), Some(&new)) {
            SubsumptionResult::Subsumed { .. } => {}
            other => panic!("Expected Subsumed, got {:?}", other),
        }

        // Reverse should not work: cached: amount > 150, new: amount > 100
        match check_subsumption(Some(&new), Some(&cached)) {
            SubsumptionResult::NoSubsumption => {}
            other => panic!("Expected NoSubsumption, got {:?}", other),
        }
    }

    #[test]
    fn test_range_subsumption_less_than() {
        // cached: amount < 500, new: amount < 300
        let cached = make_lt("amount", 500);
        let new = make_lt("amount", 300);

        match check_subsumption(Some(&cached), Some(&new)) {
            SubsumptionResult::Subsumed { .. } => {}
            other => panic!("Expected Subsumed, got {:?}", other),
        }
    }

    #[test]
    fn test_and_subsumption() {
        // cached: amount > 100, new: amount > 100 AND status > 0
        let cached = make_gt("amount", 100);
        let status_check = make_gt("status", 0);
        let new = make_and(make_gt("amount", 100), status_check);

        match check_subsumption(Some(&cached), Some(&new)) {
            SubsumptionResult::Subsumed { .. } => {}
            other => panic!("Expected Subsumed, got {:?}", other),
        }
    }

    #[test]
    fn test_in_subsumption() {
        // cached: id IN (1,2,3,4,5), new: id IN (2,3)
        let cached = make_in("id", vec![1, 2, 3, 4, 5]);
        let new = make_in("id", vec![2, 3]);

        match check_subsumption(Some(&cached), Some(&new)) {
            SubsumptionResult::Subsumed { .. } => {}
            other => panic!("Expected Subsumed, got {:?}", other),
        }
    }

    #[test]
    fn test_no_predicate_to_predicate() {
        // cached: full scan, new: amount > 100
        let new = make_gt("amount", 100);

        match check_subsumption(None, Some(&new)) {
            SubsumptionResult::Subsumed { .. } => {}
            other => panic!("Expected Subsumed, got {:?}", other),
        }
    }

    #[test]
    fn test_cache_basic() {
        let cache = SemanticCache::new();

        // Insert a result
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::Integer(200)]),
            Row::from_values(vec![Value::Integer(2), Value::Integer(300)]),
            Row::from_values(vec![Value::Integer(3), Value::Integer(400)]),
        ];

        cache.insert(
            "orders",
            vec!["id".to_string(), "amount".to_string()],
            rows.clone(),
            Some(make_gt("amount", 100)),
        );

        assert_eq!(cache.size(), 1);

        // Lookup with identical predicate
        match cache.lookup(
            "orders",
            &["id".to_string(), "amount".to_string()],
            Some(&make_gt("amount", 100)),
        ) {
            CacheLookupResult::ExactHit(cached_rows) => {
                assert_eq!(cached_rows.len(), 3);
            }
            other => panic!("Expected ExactHit, got {:?}", other),
        }

        let stats = cache.stats();
        assert_eq!(stats.exact_hits, 1);
    }

    #[test]
    fn test_cache_subsumption_lookup() {
        let cache = SemanticCache::new();

        // Cache result for amount > 100
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::Integer(150)]),
            Row::from_values(vec![Value::Integer(2), Value::Integer(200)]),
            Row::from_values(vec![Value::Integer(3), Value::Integer(300)]),
        ];

        cache.insert(
            "orders",
            vec!["id".to_string(), "amount".to_string()],
            rows,
            Some(make_gt("amount", 100)),
        );

        // Lookup with stricter predicate: amount > 180
        match cache.lookup(
            "orders",
            &["id".to_string(), "amount".to_string()],
            Some(&make_gt("amount", 180)),
        ) {
            CacheLookupResult::SubsumptionHit { rows, .. } => {
                assert_eq!(rows.len(), 3); // All cached rows returned
            }
            other => panic!("Expected SubsumptionHit, got {:?}", other),
        }

        let stats = cache.stats();
        assert_eq!(stats.subsumption_hits, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = SemanticCache::new();

        cache.insert(
            "orders",
            vec!["id".to_string()],
            vec![Row::from_values(vec![Value::Integer(1)])],
            None,
        );

        assert_eq!(cache.size(), 1);

        cache.invalidate_table("orders");
        assert_eq!(cache.size(), 0);
    }
}
