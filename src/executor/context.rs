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

//! Execution Context
//!
//! This module provides the execution context for SQL queries, including
//! parameter handling, transaction state, and query options.

use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::core::{Result, Row, Value};

// Cache for scalar subquery results to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Uses SQL string as key (not hash) to avoid collision risk.
thread_local! {
    static SCALAR_SUBQUERY_CACHE: RefCell<FxHashMap<String, Value>> = RefCell::new(FxHashMap::default());
}

/// Clear the scalar subquery cache. Should be called at the start of each top-level query.
pub fn clear_scalar_subquery_cache() {
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit(); // Release capacity to avoid memory bloat in long-running apps
    });
}

/// Get a cached scalar subquery result by SQL string key.
pub fn get_cached_scalar_subquery(key: &str) -> Option<Value> {
    SCALAR_SUBQUERY_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache a scalar subquery result.
pub fn cache_scalar_subquery(key: String, value: Value) {
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, value);
    });
}

// Cache for IN subquery results to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Uses SQL string as key (not hash) to avoid collision risk.
thread_local! {
    static IN_SUBQUERY_CACHE: RefCell<FxHashMap<String, Vec<Value>>> = RefCell::new(FxHashMap::default());
}

/// Clear the IN subquery cache. Should be called at the start of each top-level query.
pub fn clear_in_subquery_cache() {
    IN_SUBQUERY_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit(); // Release capacity to avoid memory bloat
    });
}

/// Get a cached IN subquery result by SQL string key.
pub fn get_cached_in_subquery(key: &str) -> Option<Vec<Value>> {
    IN_SUBQUERY_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache an IN subquery result.
pub fn cache_in_subquery(key: String, values: Vec<Value>) {
    IN_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, values);
    });
}

// Cache for semi-join (EXISTS) hash sets to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Uses the inner query SQL + predicate as key (format: "SEMI:table:column:predicate").
use ahash::{AHashMap, AHashSet};
thread_local! {
    static SEMI_JOIN_CACHE: RefCell<FxHashMap<String, Arc<AHashSet<Value>>>> = RefCell::new(FxHashMap::default());
}

/// Clear the semi-join cache completely.
/// NOTE: This is now only used for explicit cache clearing (e.g., after DDL operations).
/// For DML operations, use `invalidate_semi_join_cache_for_table` instead.
pub fn clear_semi_join_cache() {
    SEMI_JOIN_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
}

/// Invalidate semi-join cache entries for a specific table.
/// Should be called after INSERT, UPDATE, DELETE, or TRUNCATE on a table.
/// The cache key format is "SEMI:table:column:predicate", so we match on "SEMI:table:".
#[inline]
pub fn invalidate_semi_join_cache_for_table(table_name: &str) {
    SEMI_JOIN_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        if c.is_empty() {
            return;
        }
        c.retain(|key, _| {
            // Key format: "SEMI:table:column:predicate"
            // Use zero-allocation comparison with eq_ignore_ascii_case
            if let Some(rest) = key.strip_prefix("SEMI:") {
                if let Some(colon_pos) = rest.find(':') {
                    let key_table = &rest[..colon_pos];
                    // Keep entry if table names DON'T match (case-insensitive)
                    return !key_table.eq_ignore_ascii_case(table_name);
                }
            }
            // Keep entries that don't match expected format
            true
        });
    });
}

/// Get a cached semi-join hash set by key.
pub fn get_cached_semi_join(key: &str) -> Option<Arc<AHashSet<Value>>> {
    SEMI_JOIN_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache a semi-join hash set result.
pub fn cache_semi_join(key: String, values: AHashSet<Value>) {
    SEMI_JOIN_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, Arc::new(values));
    });
}

/// Cache a semi-join hash set result (Arc version for zero-copy).
pub fn cache_semi_join_arc(key: String, values: Arc<AHashSet<Value>>) {
    SEMI_JOIN_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, values);
    });
}

// Cache for EXISTS predicate filters to avoid re-compilation per row.
// The key is the predicate expression string (after alias stripping).
// The value is the compiled RowFilter.
use super::expression::RowFilter;
thread_local! {
    static EXISTS_PREDICATE_CACHE: RefCell<FxHashMap<String, RowFilter>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS predicate cache. Should be called at the start of each top-level query.
pub fn clear_exists_predicate_cache() {
    EXISTS_PREDICATE_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get a cached EXISTS predicate filter by key.
pub fn get_cached_exists_predicate(key: &str) -> Option<RowFilter> {
    EXISTS_PREDICATE_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache an EXISTS predicate filter.
pub fn cache_exists_predicate(key: String, filter: RowFilter) {
    EXISTS_PREDICATE_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, filter);
    });
}

// Cache for EXISTS index lookups to avoid re-fetching per row.
// The key is "table_name:column_name", the value is the index reference.
use crate::storage::traits::Index;
thread_local! {
    static EXISTS_INDEX_CACHE: RefCell<FxHashMap<String, std::sync::Arc<dyn Index>>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS index cache. Should be called at the start of each top-level query.
pub fn clear_exists_index_cache() {
    EXISTS_INDEX_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get a cached EXISTS index by key.
pub fn get_cached_exists_index(key: &str) -> Option<std::sync::Arc<dyn Index>> {
    EXISTS_INDEX_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache an EXISTS index.
pub fn cache_exists_index(key: String, index: std::sync::Arc<dyn Index>) {
    EXISTS_INDEX_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, index);
    });
}

/// Type alias for row fetcher function used in EXISTS/COUNT optimization.
pub type RowFetcher = Box<dyn Fn(&[i64]) -> Vec<(i64, crate::core::Row)> + Send + Sync>;

/// Type alias for row counter function used in COUNT(*) optimization.
/// This only counts visible rows without cloning their data.
pub type RowCounter = Box<dyn Fn(&[i64]) -> usize + Send + Sync>;

// Cache for EXISTS row fetchers to avoid repeated version store lookups.
// The key is the table name, the value is the row fetcher function.
thread_local! {
    static EXISTS_FETCHER_CACHE: RefCell<FxHashMap<String, std::sync::Arc<RowFetcher>>> = RefCell::new(FxHashMap::default());
}

// Cache for COUNT row counters to avoid repeated version store lookups.
// The key is the table name, the value is the row counter function.
thread_local! {
    static COUNT_COUNTER_CACHE: RefCell<FxHashMap<String, std::sync::Arc<RowCounter>>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS row fetcher cache. Should be called at the start of each top-level query.
pub fn clear_exists_fetcher_cache() {
    EXISTS_FETCHER_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Clear the COUNT row counter cache. Should be called at the start of each top-level query.
pub fn clear_count_counter_cache() {
    COUNT_COUNTER_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get a cached EXISTS row fetcher by table name.
pub fn get_cached_exists_fetcher(key: &str) -> Option<std::sync::Arc<RowFetcher>> {
    EXISTS_FETCHER_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Get a cached COUNT row counter by table name.
pub fn get_cached_count_counter(key: &str) -> Option<std::sync::Arc<RowCounter>> {
    COUNT_COUNTER_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache an EXISTS row fetcher.
pub fn cache_exists_fetcher(key: String, fetcher: RowFetcher) {
    EXISTS_FETCHER_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, std::sync::Arc::new(fetcher));
    });
}

/// Cache a COUNT row counter.
pub fn cache_count_counter(key: String, counter: RowCounter) {
    COUNT_COUNTER_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, std::sync::Arc::new(counter));
    });
}

// Cache for table schema column names to avoid repeated get_table_schema() calls.
// The key is the table name, the value is the list of column names.
thread_local! {
    static EXISTS_SCHEMA_CACHE: RefCell<FxHashMap<String, std::sync::Arc<Vec<String>>>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS schema cache. Should be called at the start of each top-level query.
pub fn clear_exists_schema_cache() {
    EXISTS_SCHEMA_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get cached table column names by table name.
pub fn get_cached_exists_schema(key: &str) -> Option<std::sync::Arc<Vec<String>>> {
    EXISTS_SCHEMA_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache table column names.
pub fn cache_exists_schema(key: String, columns: Vec<String>) {
    EXISTS_SCHEMA_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, std::sync::Arc::new(columns));
    });
}

// Cache for pre-computed EXISTS predicate cache keys to avoid expensive format!("{:?}") on every probe.
// The key is the subquery pointer address (usize), the value is the predicate cache key.
thread_local! {
    static EXISTS_PRED_KEY_CACHE: RefCell<FxHashMap<usize, String>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS predicate key cache.
pub fn clear_exists_pred_key_cache() {
    EXISTS_PRED_KEY_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get cached predicate cache key by subquery pointer address.
#[inline]
pub fn get_cached_exists_pred_key(subquery_ptr: usize) -> Option<String> {
    EXISTS_PRED_KEY_CACHE.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Cache a predicate cache key.
#[inline]
pub fn cache_exists_pred_key(subquery_ptr: usize, pred_key: String) {
    EXISTS_PRED_KEY_CACHE.with(|cache| {
        cache.borrow_mut().insert(subquery_ptr, pred_key);
    });
}

// Cache for batch aggregate subquery results (e.g., COUNT(*) GROUP BY user_id).
// Thread-local to avoid synchronization overhead.
// The key is a stable identifier for the subquery, the value is a map from group key to aggregate value.
thread_local! {
    static BATCH_AGGREGATE_CACHE: RefCell<FxHashMap<String, Arc<AHashMap<Value, Value>>>> = RefCell::new(FxHashMap::default());
}

/// Clear the batch aggregate cache. Should be called at the start of each top-level query.
pub fn clear_batch_aggregate_cache() {
    BATCH_AGGREGATE_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
}

/// Get a cached batch aggregate result map by subquery identifier.
pub fn get_cached_batch_aggregate(key: &str) -> Option<Arc<AHashMap<Value, Value>>> {
    BATCH_AGGREGATE_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache a batch aggregate result map.
pub fn cache_batch_aggregate(key: String, values: AHashMap<Value, Value>) {
    BATCH_AGGREGATE_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, Arc::new(values));
    });
}

/// Pre-computed info for batch aggregate lookups to avoid per-row allocations.
#[derive(Clone)]
pub struct BatchAggregateLookupInfo {
    /// The cache key for the batch aggregate results
    pub cache_key: String,
    /// The outer column name (lowercase) to look up in outer_row
    pub outer_column_lower: String,
    /// Optional qualified outer column name (e.g., "u.id")
    pub outer_qualified_lower: Option<String>,
    /// Whether this is a COUNT expression (returns 0 for missing keys)
    pub is_count: bool,
}

// Cache for batch aggregate lookup info to avoid recomputing per row.
// The key is the subquery SQL string, the value is the pre-computed lookup info.
thread_local! {
    static BATCH_AGGREGATE_INFO_CACHE: RefCell<FxHashMap<String, Option<BatchAggregateLookupInfo>>> = RefCell::new(FxHashMap::default());
}

/// Clear the batch aggregate info cache.
pub fn clear_batch_aggregate_info_cache() {
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
}

/// Get cached batch aggregate lookup info by subquery string.
pub fn get_cached_batch_aggregate_info(
    subquery_key: &str,
) -> Option<Option<BatchAggregateLookupInfo>> {
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| cache.borrow().get(subquery_key).cloned())
}

/// Cache batch aggregate lookup info (None means not batchable).
pub fn cache_batch_aggregate_info(subquery_key: String, info: Option<BatchAggregateLookupInfo>) {
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| {
        cache.borrow_mut().insert(subquery_key, info);
    });
}

use crate::parser::ast::Expression;

/// Pre-computed info for index nested loop EXISTS lookups to avoid per-row string operations.
/// This caches the pre-computed lowercase column names for O(1) outer row lookups.
#[derive(Clone)]
pub struct ExistsCorrelationInfo {
    /// The outer column name in original case
    pub outer_column: String,
    /// The outer table name (optional)
    pub outer_table: Option<String>,
    /// The inner column name
    pub inner_column: String,
    /// The inner table name
    pub inner_table: String,
    /// Pre-computed lowercase outer column name for fast HashMap lookup
    pub outer_column_lower: String,
    /// Pre-computed qualified outer column name (e.g., "u.id") in lowercase
    pub outer_qualified_lower: Option<String>,
    /// The additional predicate beyond the correlation (if any)
    pub additional_predicate: Option<Expression>,
    /// Pre-computed index cache key ("table:column") to avoid per-probe format! allocation
    pub index_cache_key: String,
}

// Cache for EXISTS correlation info to avoid per-row extraction.
// The key is the subquery pointer address (usize), avoiding format! allocation.
thread_local! {
    static EXISTS_CORRELATION_CACHE: RefCell<FxHashMap<usize, Option<ExistsCorrelationInfo>>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS correlation cache.
pub fn clear_exists_correlation_cache() {
    EXISTS_CORRELATION_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Get cached EXISTS correlation info by subquery pointer address.
#[inline]
pub fn get_cached_exists_correlation(subquery_ptr: usize) -> Option<Option<ExistsCorrelationInfo>> {
    EXISTS_CORRELATION_CACHE.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Cache EXISTS correlation info (None means correlation not extractable).
#[inline]
pub fn cache_exists_correlation(subquery_ptr: usize, info: Option<ExistsCorrelationInfo>) {
    EXISTS_CORRELATION_CACHE.with(|cache| {
        cache.borrow_mut().insert(subquery_ptr, info);
    });
}

/// Execution context for SQL queries
///
/// The execution context carries state and configuration for query execution,
/// including parameters, transaction state, and cancellation support.
///
/// Note: This struct uses Arc for immutable shared data to make cloning cheap
/// during correlated subquery processing where context is cloned per row.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Query parameters ($1, $2, etc.) - wrapped in Arc for cheap cloning
    params: Arc<Vec<Value>>,
    /// Named parameters (:name) - wrapped in Arc for cheap cloning
    named_params: Arc<FxHashMap<String, Value>>,
    /// Whether to use auto-commit for DML statements
    auto_commit: bool,
    /// Cancellation flag
    cancelled: Arc<AtomicBool>,
    /// Current database/schema name - wrapped in Arc for cheap cloning
    current_database: Arc<Option<String>>,
    /// Session variables (SET key = value) - wrapped in Arc for cheap cloning
    session_vars: Arc<AHashMap<String, Value>>,
    /// Query timeout in milliseconds (0 = no timeout)
    timeout_ms: u64,
    /// Current view nesting depth (for detecting infinite recursion)
    view_depth: usize,
    /// Query execution depth (0 = top-level query, >0 = subquery/nested)
    /// Used to ensure TimeoutGuard is only created once at the top level
    pub(crate) query_depth: usize,
    /// Outer row context for correlated subqueries
    /// Maps column name (lowercase) to value from the outer query
    /// Uses FxHashMap for faster hashing
    /// pub(crate) to allow taking ownership back for reuse in optimized loops
    pub(crate) outer_row: Option<FxHashMap<String, Value>>,
    /// Outer row column names (for qualified identifier resolution) - wrapped in Arc
    outer_columns: Option<Arc<Vec<String>>>,
    /// CTE data for subqueries to reference CTEs from outer query
    /// Maps CTE name (lowercase) to (columns, rows)
    cte_data: Option<Arc<CteDataMap>>,
    /// Current transaction ID for CURRENT_TRANSACTION_ID() function
    transaction_id: Option<u64>,
}

/// Type alias for CTE data map to reduce type complexity
/// Uses Arc<Vec<Row>> to enable zero-copy sharing of CTE results with joins
type CteDataMap = FxHashMap<String, (Vec<String>, Arc<Vec<Row>>)>;

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new empty execution context
    pub fn new() -> Self {
        Self {
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            auto_commit: true,
            cancelled: Arc::new(AtomicBool::new(false)),
            current_database: Arc::new(None),
            session_vars: Arc::new(AHashMap::new()),
            timeout_ms: 0,
            view_depth: 0,
            query_depth: 0,
            outer_row: None,
            outer_columns: None,
            cte_data: None,
            transaction_id: None,
        }
    }

    /// Create an execution context with positional parameters
    pub fn with_params(params: Vec<Value>) -> Self {
        Self {
            params: Arc::new(params),
            ..Self::new()
        }
    }

    /// Create an execution context with named parameters
    /// Accepts std::collections::HashMap for API compatibility
    pub fn with_named_params(named_params: std::collections::HashMap<String, Value>) -> Self {
        // Convert HashMap to FxHashMap (more efficient for lookups)
        let fx_params: FxHashMap<String, Value> = named_params.into_iter().collect();
        Self {
            named_params: Arc::new(fx_params),
            ..Self::new()
        }
    }

    /// Get a positional parameter by index (1-based)
    pub fn get_param(&self, index: usize) -> Option<&Value> {
        if index == 0 || index > self.params.len() {
            None
        } else {
            self.params.get(index - 1)
        }
    }

    /// Get a named parameter by name
    pub fn get_named_param(&self, name: &str) -> Option<&Value> {
        self.named_params.get(name)
    }

    /// Get all positional parameters
    pub fn params(&self) -> &[Value] {
        &self.params
    }

    /// Get the params Arc for zero-copy sharing.
    /// Used by evaluator bridge to avoid cloning params.
    pub fn params_arc(&self) -> &Arc<Vec<Value>> {
        &self.params
    }

    /// Get all named parameters
    pub fn named_params(&self) -> &FxHashMap<String, Value> {
        &self.named_params
    }

    /// Get the named_params Arc for zero-copy sharing.
    /// Used by evaluator bridge to avoid cloning params.
    pub fn named_params_arc(&self) -> &Arc<FxHashMap<String, Value>> {
        &self.named_params
    }

    /// Get the number of positional parameters
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Set positional parameters
    pub fn set_params(&mut self, params: Vec<Value>) {
        self.params = Arc::new(params);
    }

    /// Add a positional parameter
    pub fn add_param(&mut self, value: Value) {
        Arc::make_mut(&mut self.params).push(value);
    }

    /// Set a named parameter
    pub fn set_named_param(&mut self, name: impl Into<String>, value: Value) {
        Arc::make_mut(&mut self.named_params).insert(name.into(), value);
    }

    /// Check if auto-commit is enabled
    pub fn auto_commit(&self) -> bool {
        self.auto_commit
    }

    /// Set auto-commit mode
    pub fn set_auto_commit(&mut self, auto_commit: bool) {
        self.auto_commit = auto_commit;
    }

    /// Check if the query has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Cancel the query
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Get a cancellation handle that can be used from another thread
    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            cancelled: self.cancelled.clone(),
        }
    }

    /// Get the current database/schema name
    pub fn current_database(&self) -> Option<&str> {
        self.current_database.as_ref().as_deref()
    }

    /// Set the current database/schema name
    pub fn set_current_database(&mut self, database: impl Into<String>) {
        self.current_database = Arc::new(Some(database.into()));
    }

    /// Get a session variable
    pub fn get_session_var(&self, name: &str) -> Option<&Value> {
        self.session_vars.get(name)
    }

    /// Set a session variable
    pub fn set_session_var(&mut self, name: impl Into<String>, value: Value) {
        Arc::make_mut(&mut self.session_vars).insert(name.into(), value);
    }

    /// Get the query timeout in milliseconds
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    /// Set the query timeout in milliseconds
    pub fn set_timeout_ms(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }

    /// Check if a timeout has been set
    pub fn has_timeout(&self) -> bool {
        self.timeout_ms > 0
    }

    /// Get the current view nesting depth
    pub fn view_depth(&self) -> usize {
        self.view_depth
    }

    /// Create a new context with incremented view depth.
    /// Used when executing nested views to track recursion depth.
    /// Also increments query_depth since views are nested queries.
    pub fn with_incremented_view_depth(&self) -> Self {
        Self {
            params: self.params.clone(),
            named_params: self.named_params.clone(),
            auto_commit: self.auto_commit,
            cancelled: self.cancelled.clone(),
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            timeout_ms: self.timeout_ms,
            view_depth: self.view_depth + 1,
            query_depth: self.query_depth + 1, // Views are nested queries
            outer_row: self.outer_row.clone(),
            outer_columns: self.outer_columns.clone(),
            cte_data: self.cte_data.clone(),
            transaction_id: self.transaction_id,
        }
    }

    /// Create a new context with incremented query depth.
    /// Used when executing subqueries to ensure TimeoutGuard is only created at the top level.
    pub fn with_incremented_query_depth(&self) -> Self {
        Self {
            params: self.params.clone(),
            named_params: self.named_params.clone(),
            auto_commit: self.auto_commit,
            cancelled: self.cancelled.clone(),
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            timeout_ms: self.timeout_ms,
            view_depth: self.view_depth,
            query_depth: self.query_depth + 1,
            outer_row: self.outer_row.clone(),
            outer_columns: self.outer_columns.clone(),
            cte_data: self.cte_data.clone(),
            transaction_id: self.transaction_id,
        }
    }

    /// Get the outer row context for correlated subqueries
    pub fn outer_row(&self) -> Option<&FxHashMap<String, Value>> {
        self.outer_row.as_ref()
    }

    /// Get the outer row columns for correlated subqueries
    pub fn outer_columns(&self) -> Option<&[String]> {
        self.outer_columns.as_ref().map(|v| v.as_slice())
    }

    /// Create a new context with outer row context for correlated subqueries.
    /// The outer_row maps lowercase column names to their values.
    /// NOTE: This is now cheap to clone due to Arc wrapping of immutable fields.
    pub fn with_outer_row(
        &self,
        outer_row: FxHashMap<String, Value>,
        outer_columns: Arc<Vec<String>>,
    ) -> Self {
        Self {
            params: self.params.clone(),             // Arc clone = cheap
            named_params: self.named_params.clone(), // Arc clone = cheap
            auto_commit: self.auto_commit,
            cancelled: self.cancelled.clone(), // Arc clone = cheap
            current_database: self.current_database.clone(), // Arc clone = cheap
            session_vars: self.session_vars.clone(), // Arc clone = cheap
            timeout_ms: self.timeout_ms,
            view_depth: self.view_depth,
            query_depth: self.query_depth + 1, // Increment for subquery
            outer_row: Some(outer_row),
            outer_columns: Some(outer_columns), // Arc clone = cheap
            cte_data: self.cte_data.clone(),    // Arc clone = cheap
            transaction_id: self.transaction_id,
        }
    }

    /// Get CTE data by name (case-insensitive)
    /// Returns Arc reference to enable zero-copy sharing with joins
    pub fn get_cte(&self, name: &str) -> Option<(&Vec<String>, &Arc<Vec<Row>>)> {
        self.cte_data.as_ref().and_then(|data| {
            data.get(&name.to_lowercase())
                .map(|(cols, rows)| (cols, rows))
        })
    }

    /// Check if context has CTE data
    pub fn has_cte(&self, name: &str) -> bool {
        self.cte_data
            .as_ref()
            .is_some_and(|data| data.contains_key(&name.to_lowercase()))
    }

    /// Create a new context with CTE data for subqueries to reference
    /// Takes an Arc to avoid cloning large CTE datasets
    pub fn with_cte_data(&self, cte_data: Arc<CteDataMap>) -> Self {
        Self {
            params: self.params.clone(),
            named_params: self.named_params.clone(),
            auto_commit: self.auto_commit,
            cancelled: self.cancelled.clone(),
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            timeout_ms: self.timeout_ms,
            view_depth: self.view_depth,
            query_depth: self.query_depth,
            outer_row: self.outer_row.clone(),
            outer_columns: self.outer_columns.clone(),
            cte_data: Some(cte_data),
            transaction_id: self.transaction_id,
        }
    }

    /// Get the current transaction ID
    pub fn transaction_id(&self) -> Option<u64> {
        self.transaction_id
    }

    /// Set the transaction ID
    pub fn set_transaction_id(&mut self, txn_id: u64) {
        self.transaction_id = Some(txn_id);
    }

    /// Create a new context with a transaction ID
    pub fn with_transaction_id(&self, txn_id: u64) -> Self {
        Self {
            params: self.params.clone(),
            named_params: self.named_params.clone(),
            auto_commit: self.auto_commit,
            cancelled: self.cancelled.clone(),
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            timeout_ms: self.timeout_ms,
            view_depth: self.view_depth,
            query_depth: self.query_depth,
            outer_row: self.outer_row.clone(),
            outer_columns: self.outer_columns.clone(),
            cte_data: self.cte_data.clone(),
            transaction_id: Some(txn_id),
        }
    }

    /// Check for cancellation and return an error if cancelled
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(crate::core::Error::QueryCancelled)
        } else {
            Ok(())
        }
    }
}

/// Handle for cancelling a query from another thread
#[derive(Debug, Clone)]
pub struct CancellationHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Cancel the query
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Check if the query has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Global Timeout Manager
// ============================================================================
//
// Uses a single background thread to manage all query timeouts efficiently.
// This avoids spawning a new thread for each query with a timeout.

/// Entry in the timeout priority queue
struct TimeoutEntry {
    /// When the timeout expires
    deadline: Instant,
    /// Unique ID for this timeout (for cancellation)
    id: u64,
    /// Handle to cancel the query
    cancel_handle: CancellationHandle,
    /// Whether this timeout has been cancelled (query completed)
    cancelled: Arc<AtomicBool>,
}

impl PartialEq for TimeoutEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.id == other.id
    }
}

impl Eq for TimeoutEntry {}

impl PartialOrd for TimeoutEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimeoutEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap becomes a min-heap (earliest deadline first)
        other.deadline.cmp(&self.deadline)
    }
}

/// Global timeout manager state
struct TimeoutManagerState {
    /// Priority queue of pending timeouts (min-heap by deadline)
    timeouts: BinaryHeap<TimeoutEntry>,
    /// Whether the manager is shutting down
    shutdown: bool,
}

/// Global timeout manager that handles all query timeouts in a single thread
struct TimeoutManager {
    /// Shared state protected by mutex
    state: Mutex<TimeoutManagerState>,
    /// Condition variable to wake the timer thread
    condvar: Condvar,
    /// Counter for generating unique timeout IDs
    next_id: AtomicU64,
}

impl TimeoutManager {
    /// Create a new timeout manager and spawn its background thread
    fn new() -> Arc<Self> {
        let manager = Arc::new(Self {
            state: Mutex::new(TimeoutManagerState {
                timeouts: BinaryHeap::new(),
                shutdown: false,
            }),
            condvar: Condvar::new(),
            next_id: AtomicU64::new(1),
        });

        // Spawn the background timer thread
        let manager_clone = Arc::clone(&manager);
        std::thread::Builder::new()
            .name("stoolap-timeout-manager".to_string())
            .spawn(move || {
                manager_clone.run();
            })
            .expect("Failed to spawn timeout manager thread");

        manager
    }

    /// Background thread loop
    fn run(&self) {
        loop {
            let mut state = self.state.lock().unwrap();

            // Check for shutdown
            if state.shutdown && state.timeouts.is_empty() {
                return;
            }

            // Process expired timeouts
            let now = Instant::now();
            while let Some(entry) = state.timeouts.peek() {
                if entry.deadline <= now {
                    let entry = state.timeouts.pop().unwrap();
                    // Only cancel if the timeout wasn't already cancelled
                    if !entry.cancelled.load(Ordering::Relaxed) {
                        entry.cancel_handle.cancel();
                    }
                } else {
                    break;
                }
            }

            // Calculate wait time until next timeout
            let wait_duration = if let Some(entry) = state.timeouts.peek() {
                entry.deadline.saturating_duration_since(now)
            } else {
                // No timeouts pending, wait indefinitely for new work
                Duration::from_secs(3600) // 1 hour max wait
            };

            // Wait for new work or timeout
            if wait_duration.is_zero() {
                continue; // Immediately process
            }
            let (new_state, _timeout_result) =
                self.condvar.wait_timeout(state, wait_duration).unwrap();
            state = new_state;

            // Re-check shutdown after waking
            if state.shutdown && state.timeouts.is_empty() {
                return;
            }
        }
    }

    /// Register a new timeout, returns the timeout ID
    fn register(
        &self,
        timeout_ms: u64,
        cancel_handle: CancellationHandle,
        cancelled: Arc<AtomicBool>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);

        let entry = TimeoutEntry {
            deadline,
            id,
            cancel_handle,
            cancelled,
        };

        let mut state = self.state.lock().unwrap();
        let was_empty = state.timeouts.is_empty();
        let is_earliest = state.timeouts.peek().is_none_or(|e| deadline < e.deadline);

        state.timeouts.push(entry);

        // Wake the timer thread if this is the new earliest deadline
        if was_empty || is_earliest {
            self.condvar.notify_one();
        }

        id
    }
}

/// Get or create the global timeout manager
fn global_timeout_manager() -> &'static Arc<TimeoutManager> {
    use std::sync::OnceLock;
    static MANAGER: OnceLock<Arc<TimeoutManager>> = OnceLock::new();
    MANAGER.get_or_init(TimeoutManager::new)
}

/// Guard that automatically cancels a query after a timeout.
/// Uses a global timeout manager for efficient handling of many concurrent timeouts.
pub struct TimeoutGuard {
    /// Flag to signal that the query completed (timeout should be ignored)
    cancelled: Arc<AtomicBool>,
}

impl TimeoutGuard {
    /// Create a new timeout guard that will cancel the query after timeout_ms.
    /// Returns None if timeout_ms is 0 (no timeout).
    pub fn new(ctx: &ExecutionContext) -> Option<Self> {
        let timeout_ms = ctx.timeout_ms();
        if timeout_ms == 0 {
            return None;
        }

        let cancel_handle = ctx.cancellation_handle();
        let cancelled = Arc::new(AtomicBool::new(false));

        // Register with the global timeout manager
        global_timeout_manager().register(timeout_ms, cancel_handle, Arc::clone(&cancelled));

        Some(Self { cancelled })
    }
}

impl Drop for TimeoutGuard {
    fn drop(&mut self) {
        // Mark this timeout as cancelled so the manager ignores it
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Builder for ExecutionContext
pub struct ExecutionContextBuilder {
    ctx: ExecutionContext,
}

impl ExecutionContextBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            ctx: ExecutionContext::new(),
        }
    }

    /// Add positional parameters
    pub fn params(mut self, params: Vec<Value>) -> Self {
        self.ctx.params = Arc::new(params);
        self
    }

    /// Add a positional parameter
    pub fn param(self, value: Value) -> Self {
        let mut v = (*self.ctx.params).clone();
        v.push(value);
        Self {
            ctx: ExecutionContext {
                params: Arc::new(v),
                ..self.ctx
            },
        }
    }

    /// Add a named parameter
    pub fn named_param(self, name: impl Into<String>, value: Value) -> Self {
        Self {
            ctx: ExecutionContext {
                named_params: Arc::new({
                    let mut m = (*self.ctx.named_params).clone();
                    m.insert(name.into(), value);
                    m
                }),
                ..self.ctx
            },
        }
    }

    /// Set auto-commit mode
    pub fn auto_commit(mut self, auto_commit: bool) -> Self {
        self.ctx.auto_commit = auto_commit;
        self
    }

    /// Set the current database
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.ctx.current_database = Arc::new(Some(database.into()));
        self
    }

    /// Set a session variable
    pub fn session_var(self, name: impl Into<String>, value: Value) -> Self {
        Self {
            ctx: ExecutionContext {
                session_vars: Arc::new({
                    let mut m = (*self.ctx.session_vars).clone();
                    m.insert(name.into(), value);
                    m
                }),
                ..self.ctx
            },
        }
    }

    /// Set the query timeout
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.ctx.timeout_ms = timeout_ms;
        self
    }

    /// Build the execution context
    pub fn build(self) -> ExecutionContext {
        self.ctx
    }
}

impl Default for ExecutionContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_context_new() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.param_count(), 0);
        assert!(ctx.auto_commit());
        assert!(!ctx.is_cancelled());
    }

    #[test]
    fn test_context_with_params() {
        let ctx = ExecutionContext::with_params(vec![Value::Integer(1), Value::text("hello")]);
        assert_eq!(ctx.param_count(), 2);
        assert_eq!(ctx.get_param(1), Some(&Value::Integer(1)));
        assert_eq!(ctx.get_param(2), Some(&Value::text("hello")));
        assert_eq!(ctx.get_param(0), None); // 0 is invalid
        assert_eq!(ctx.get_param(3), None); // Out of bounds
    }

    #[test]
    fn test_context_named_params() {
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::text("Alice"));
        params.insert("age".to_string(), Value::Integer(30));

        let ctx = ExecutionContext::with_named_params(params);
        assert_eq!(ctx.get_named_param("name"), Some(&Value::text("Alice")));
        assert_eq!(ctx.get_named_param("age"), Some(&Value::Integer(30)));
        assert_eq!(ctx.get_named_param("unknown"), None);
    }

    #[test]
    fn test_context_cancellation() {
        let ctx = ExecutionContext::new();
        assert!(!ctx.is_cancelled());

        let handle = ctx.cancellation_handle();
        assert!(!handle.is_cancelled());

        handle.cancel();
        assert!(ctx.is_cancelled());
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_context_check_cancelled() {
        let ctx = ExecutionContext::new();
        assert!(ctx.check_cancelled().is_ok());

        ctx.cancel();
        assert!(ctx.check_cancelled().is_err());
    }

    #[test]
    fn test_context_session_vars() {
        let mut ctx = ExecutionContext::new();
        ctx.set_session_var("timezone", Value::text("UTC"));

        assert_eq!(ctx.get_session_var("timezone"), Some(&Value::text("UTC")));
        assert_eq!(ctx.get_session_var("unknown"), None);
    }

    #[test]
    fn test_context_builder() {
        let ctx = ExecutionContextBuilder::new()
            .params(vec![Value::Integer(1)])
            .param(Value::Integer(2))
            .named_param("name", Value::text("test"))
            .auto_commit(false)
            .database("mydb")
            .timeout_ms(5000)
            .build();

        assert_eq!(ctx.param_count(), 2);
        assert_eq!(ctx.get_param(1), Some(&Value::Integer(1)));
        assert_eq!(ctx.get_param(2), Some(&Value::Integer(2)));
        assert_eq!(ctx.get_named_param("name"), Some(&Value::text("test")));
        assert!(!ctx.auto_commit());
        assert_eq!(ctx.current_database(), Some("mydb"));
        assert_eq!(ctx.timeout_ms(), 5000);
    }
}
