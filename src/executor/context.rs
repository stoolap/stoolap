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
use std::collections::{BinaryHeap, HashMap};
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
    session_vars: Arc<HashMap<String, Value>>,
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
type CteDataMap = FxHashMap<String, (Vec<String>, Vec<Row>)>;

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
            session_vars: Arc::new(HashMap::new()),
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
    pub fn get_cte(&self, name: &str) -> Option<(&Vec<String>, &Vec<Row>)> {
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
