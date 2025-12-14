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

//! SQL Executor
//!
//! This module provides the SQL query execution engine for Stoolap.
//!
//! # Architecture
//!
//! The executor follows a composable result pipeline pattern where each
//! operation wraps an underlying result to transform the data:
//!
//! ```text
//! Table.Scan()
//!   ↓
//! FilteredResult (WHERE clause)
//!   ↓
//! JoinResult (JOIN operations)
//!   ↓
//! AggregateResult (GROUP BY)
//!   ↓
//! OrderedResult (ORDER BY)
//!   ↓
//! LimitedResult (LIMIT/OFFSET)
//!   ↓
//! User Application
//! ```
//!
//! # Components
//!
//! - [`Executor`] - Main executor orchestrating query execution
//! - [`Evaluator`] - Expression evaluation engine
//! - [`ExecResult`] - Base result type for DML operations
//! - Various result wrappers for query pipeline

pub mod context;
pub mod expression;
pub mod parallel;
pub mod pattern_cache;
pub mod planner;
pub mod query_cache;
pub mod result;
pub mod semantic_cache;
pub mod statistics;

mod aggregation;
mod cte;
mod ddl;
mod dml;
mod explain;
mod join;
pub mod pushdown;
mod query;
mod set_ops;
mod show;
mod subquery;
pub mod utils;
mod window;

use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

use crate::core::{Error, Result, Value};
use crate::functions::FunctionRegistry;
use crate::parser::ast::{Program, Statement};
use crate::parser::Parser;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};

pub use context::{ExecutionContext, TimeoutGuard};
pub use expression::{
    CompileContext, CompileError, CompiledEvaluator, ExecuteContext, ExprCompiler, ExprVM,
    Program as ExprProgram,
};
pub use parallel::{
    hash_row_by_keys,
    parallel_hash_build,
    parallel_hash_join,
    parallel_hash_probe,
    parallel_order_by,
    parallel_order_by_fn,
    verify_key_match,
    JoinType as ParallelJoinType,
    ParallelConfig,
    ParallelHashTable,
    ParallelJoinResult,
    ParallelStats,
    SortDirection,
    SortSpec,
    DEFAULT_PARALLEL_CHUNK_SIZE,
    // Parallel threshold constants - single source of truth
    DEFAULT_PARALLEL_FILTER_THRESHOLD,
    DEFAULT_PARALLEL_JOIN_THRESHOLD,
    DEFAULT_PARALLEL_SORT_THRESHOLD,
};
pub use planner::{
    AccessPlan, ColumnStatsCache, JoinPlan, QueryPlanner, RuntimeJoinAlgorithm,
    RuntimeJoinDecision, StatsHealth,
};
pub use query_cache::{CacheStats, CachedQueryPlan, QueryCache, DEFAULT_CACHE_SIZE};
pub use result::{ExecResult, ExecutorMemoryResult};
pub use semantic_cache::{
    CacheLookupResult, CachedResult, QueryFingerprint, SemanticCache, SemanticCacheStats,
    SemanticCacheStatsSnapshot, SubsumptionResult, DEFAULT_CACHE_TTL_SECS, DEFAULT_MAX_CACHED_ROWS,
    DEFAULT_SEMANTIC_CACHE_SIZE,
};

/// Active transaction state for explicit transaction control (BEGIN/COMMIT/ROLLBACK)
struct ActiveTransaction {
    /// The transaction object
    transaction: Box<dyn Transaction>,
    /// Tables accessed within this transaction (cached for proper commit/rollback)
    tables: FxHashMap<String, Box<dyn Table>>,
}

/// SQL Query Executor
///
/// The executor is the main entry point for executing SQL statements.
/// It coordinates between the parser, storage engine, and function registry.
pub struct Executor {
    /// Storage engine
    engine: Arc<MVCCEngine>,
    /// Function registry for scalar, aggregate, and window functions
    function_registry: Arc<FunctionRegistry>,
    /// Default isolation level for transactions
    default_isolation_level: crate::core::IsolationLevel,
    /// Query cache for parsed statements
    query_cache: QueryCache,
    /// Semantic cache for query results with subsumption detection
    semantic_cache: SemanticCache,
    /// Active transaction for explicit transaction control (BEGIN/COMMIT/ROLLBACK)
    active_transaction: Mutex<Option<ActiveTransaction>>,
    /// Query planner for cost-based optimization (lazily initialized)
    query_planner: std::sync::OnceLock<QueryPlanner>,
}

impl Executor {
    /// Create a new executor with the given storage engine
    pub fn new(engine: Arc<MVCCEngine>) -> Self {
        Self {
            engine,
            function_registry: Arc::new(FunctionRegistry::new()),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache: SemanticCache::default(),
            active_transaction: Mutex::new(None),
            query_planner: std::sync::OnceLock::new(),
        }
    }

    /// Create a new executor with a custom function registry
    pub fn with_function_registry(
        engine: Arc<MVCCEngine>,
        function_registry: Arc<FunctionRegistry>,
    ) -> Self {
        Self {
            engine,
            function_registry,
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache: SemanticCache::default(),
            active_transaction: Mutex::new(None),
            query_planner: std::sync::OnceLock::new(),
        }
    }

    /// Create a new executor with a custom cache size
    pub fn with_cache_size(engine: Arc<MVCCEngine>, cache_size: usize) -> Self {
        Self {
            engine,
            function_registry: Arc::new(FunctionRegistry::new()),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::new(cache_size),
            semantic_cache: SemanticCache::default(),
            active_transaction: Mutex::new(None),
            query_planner: std::sync::OnceLock::new(),
        }
    }

    /// Check if there is an active explicit transaction
    pub fn has_active_transaction(&self) -> bool {
        self.active_transaction.lock().unwrap().is_some()
    }

    /// Get the query planner (lazily initialized)
    fn get_query_planner(&self) -> &QueryPlanner {
        self.query_planner
            .get_or_init(|| QueryPlanner::new(Arc::clone(&self.engine)))
    }

    /// Get or create a table within the active transaction
    /// Returns (table, should_auto_commit) where should_auto_commit is false if there's an active transaction
    #[allow(dead_code)]
    fn get_table_for_dml(&self, table_name: &str) -> Result<(Box<dyn Table>, bool)> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(ref mut tx_state) = *active_tx {
            // There's an active transaction - use it
            let table_name_lower = table_name.to_lowercase();

            // Check if we already have this table cached
            if tx_state.tables.contains_key(&table_name_lower) {
                // We need to get the table from the transaction again since we can't clone Box<dyn Table>
                let table = tx_state.transaction.get_table(table_name)?;
                return Ok((table, false));
            }

            // Get the table from the transaction and cache it
            let table = tx_state.transaction.get_table(table_name)?;

            // Store a reference indicator that this table is active in the transaction
            // Note: We can't cache the actual table as Box<dyn Table> isn't Clone
            // But we can get a fresh handle each time - the key is using the same transaction
            tx_state.tables.insert(
                table_name_lower.clone(),
                tx_state.transaction.get_table(table_name)?,
            );

            Ok((table, false))
        } else {
            // No active transaction - create a new one with auto-commit
            let tx = self.engine.begin_transaction()?;
            let table = tx.get_table(table_name)?;
            Ok((table, true))
        }
    }

    /// Start a transaction and get a table, returning transaction and table
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    fn start_transaction_for_dml(
        &self,
        table_name: &str,
    ) -> Result<(Option<Box<dyn Transaction>>, Box<dyn Table>, bool)> {
        let active_tx = self.active_transaction.lock().unwrap();

        if active_tx.is_some() {
            // There's an active transaction - we'll use the cached version
            drop(active_tx);
            let (table, auto_commit) = self.get_table_for_dml(table_name)?;
            Ok((None, table, auto_commit))
        } else {
            // No active transaction - create a new one with auto-commit
            drop(active_tx);
            let tx = self.engine.begin_transaction()?;
            let table = tx.get_table(table_name)?;
            Ok((Some(tx), table, true))
        }
    }

    /// Set the default isolation level for new transactions
    pub fn set_default_isolation_level(&mut self, level: crate::core::IsolationLevel) {
        self.default_isolation_level = level;
    }

    /// Get the storage engine
    pub fn engine(&self) -> &Arc<MVCCEngine> {
        &self.engine
    }

    /// Get the function registry
    pub fn function_registry(&self) -> &Arc<FunctionRegistry> {
        &self.function_registry
    }

    /// Execute a SQL query string
    ///
    /// This is the main entry point for executing SQL statements.
    /// It parses the query and executes each statement in order.
    /// Uses the query cache to avoid re-parsing identical queries.
    pub fn execute(&self, sql: &str) -> Result<Box<dyn QueryResult>> {
        let ctx = ExecutionContext::new();
        self.execute_cached(sql, &ctx)
    }

    /// Execute a SQL query with positional parameters
    ///
    /// Parameters are substituted for $1, $2, etc. placeholders in the query.
    /// Uses the query cache for efficient re-execution of parameterized queries.
    pub fn execute_with_params(&self, sql: &str, params: &[Value]) -> Result<Box<dyn QueryResult>> {
        let ctx = ExecutionContext::with_params(params.to_vec());
        self.execute_cached(sql, &ctx)
    }

    /// Execute a SQL query with named parameters
    ///
    /// Parameters are substituted for :name placeholders in the query.
    /// Uses the query cache for efficient re-execution of parameterized queries.
    pub fn execute_with_named_params(
        &self,
        sql: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<Box<dyn QueryResult>> {
        let ctx = ExecutionContext::with_named_params(params);
        self.execute_cached(sql, &ctx)
    }

    /// Execute a SQL query with a full execution context
    /// Uses the query cache for efficient re-execution.
    pub fn execute_with_context(
        &self,
        sql: &str,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_cached(sql, ctx)
    }

    /// Execute a SQL query using the query cache
    ///
    /// This method first checks the cache for a previously parsed statement.
    /// If found, it uses the cached AST. Otherwise, it parses the query
    /// and caches the result for future use.
    fn execute_cached(&self, sql: &str, ctx: &ExecutionContext) -> Result<Box<dyn QueryResult>> {
        // Try to get from cache
        if let Some(cached) = self.query_cache.get(sql) {
            // Validate parameter count if query has parameters
            if cached.has_params {
                let provided = ctx.params().len();
                if provided < cached.param_count {
                    return Err(Error::internal(format!(
                        "Query requires {} parameters but only {} provided",
                        cached.param_count, provided
                    )));
                }
            }

            // Execute the cached statement
            return self.execute_statement(&cached.statement, ctx);
        }

        // Parse the query
        let mut parser = Parser::new(sql);
        let mut program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;

        // Cache single-statement queries and execute directly from cache
        if program.statements.len() == 1 {
            // Take ownership of the statement to avoid clone
            let stmt = program.statements.pop().unwrap();
            let (has_params, param_count) = count_parameters(&stmt);
            let stmt_arc = std::sync::Arc::new(stmt);
            self.query_cache
                .put(sql, stmt_arc.clone(), has_params, param_count);
            // Execute directly from the Arc (no clone needed)
            return self.execute_statement(&stmt_arc, ctx);
        }

        self.execute_program_with_context(&program, ctx)
    }

    /// Get the query cache
    pub fn query_cache(&self) -> &QueryCache {
        &self.query_cache
    }

    /// Get query cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.query_cache.stats()
    }

    /// Clear the query cache
    pub fn clear_cache(&self) {
        self.query_cache.clear();
    }

    /// Get the semantic cache
    pub fn semantic_cache(&self) -> &SemanticCache {
        &self.semantic_cache
    }

    /// Get semantic cache statistics
    pub fn semantic_cache_stats(&self) -> SemanticCacheStatsSnapshot {
        self.semantic_cache.stats()
    }

    /// Clear the semantic cache
    pub fn clear_semantic_cache(&self) {
        self.semantic_cache.clear();
    }

    /// Invalidate semantic cache for a specific table
    ///
    /// Call this after INSERT, UPDATE, DELETE, or TRUNCATE on a table.
    pub fn invalidate_semantic_cache(&self, table_name: &str) {
        self.semantic_cache.invalidate_table(table_name);
    }

    /// Execute a parsed program
    pub fn execute_program(&self, program: &Program) -> Result<Box<dyn QueryResult>> {
        let ctx = ExecutionContext::new();
        self.execute_program_with_context(program, &ctx)
    }

    /// Execute a parsed program with context
    pub fn execute_program_with_context(
        &self,
        program: &Program,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        if program.statements.is_empty() {
            return Ok(Box::new(ExecResult::empty()));
        }

        let mut last_result: Option<Box<dyn QueryResult>> = None;

        for statement in &program.statements {
            last_result = Some(self.execute_statement(statement, ctx)?);
        }

        Ok(last_result.unwrap())
    }

    /// Execute a single statement
    pub fn execute_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // If there's an active transaction, inject the transaction ID into the context
        // This enables CURRENT_TRANSACTION_ID() function to return the correct value
        let ctx = {
            let active_tx = self.active_transaction.lock().unwrap();
            if let Some(ref tx_state) = *active_tx {
                let txn_id = tx_state.transaction.id();
                ctx.with_transaction_id(txn_id as u64)
            } else {
                ctx.clone()
            }
        };

        match statement {
            // DDL statements
            Statement::CreateTable(stmt) => self.execute_create_table(stmt, &ctx),
            Statement::DropTable(stmt) => self.execute_drop_table(stmt, &ctx),
            Statement::CreateIndex(stmt) => self.execute_create_index(stmt, &ctx),
            Statement::DropIndex(stmt) => self.execute_drop_index(stmt, &ctx),
            Statement::AlterTable(stmt) => self.execute_alter_table(stmt, &ctx),
            Statement::CreateView(stmt) => self.execute_create_view(stmt, &ctx),
            Statement::DropView(stmt) => self.execute_drop_view(stmt, &ctx),
            Statement::CreateColumnarIndex(stmt) => self.execute_create_columnar_index(stmt, &ctx),
            Statement::DropColumnarIndex(stmt) => self.execute_drop_columnar_index(stmt, &ctx),

            // DML statements
            Statement::Insert(stmt) => self.execute_insert(stmt, &ctx),
            Statement::Update(stmt) => self.execute_update(stmt, &ctx),
            Statement::Delete(stmt) => self.execute_delete(stmt, &ctx),
            Statement::Truncate(stmt) => self.execute_truncate(stmt, &ctx),

            // Query statements
            Statement::Select(stmt) => self.execute_select(stmt, &ctx),

            // Transaction control
            Statement::Begin(stmt) => self.execute_begin(stmt, &ctx),
            Statement::Commit(stmt) => self.execute_commit_stmt(stmt, &ctx),
            Statement::Rollback(stmt) => self.execute_rollback_stmt(stmt, &ctx),
            Statement::Savepoint(stmt) => self.execute_savepoint(stmt, &ctx),

            // Utility statements
            Statement::Set(stmt) => self.execute_set(stmt, &ctx),
            Statement::ShowTables(stmt) => self.execute_show_tables(stmt, &ctx),
            Statement::ShowViews(stmt) => self.execute_show_views(stmt, &ctx),
            Statement::ShowCreateTable(stmt) => self.execute_show_create_table(stmt, &ctx),
            Statement::ShowCreateView(stmt) => self.execute_show_create_view(stmt, &ctx),
            Statement::ShowIndexes(stmt) => self.execute_show_indexes(stmt, &ctx),
            Statement::Describe(stmt) => self.execute_describe(stmt, &ctx),
            Statement::Pragma(stmt) => self.execute_pragma(stmt, &ctx),
            Statement::Expression(stmt) => self.execute_expression_stmt(stmt, &ctx),
            Statement::Explain(stmt) => self.execute_explain(stmt, &ctx),
            Statement::Analyze(stmt) => self.execute_analyze(stmt, &ctx),
        }
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.engine.begin_transaction()
    }

    /// Begin a new transaction with a specific isolation level
    pub fn begin_transaction_with_isolation(
        &self,
        isolation: crate::core::IsolationLevel,
    ) -> Result<Box<dyn Transaction>> {
        let mut tx = self.engine.begin_transaction()?;
        let _ = tx.set_isolation_level(isolation);
        Ok(tx)
    }
}

/// Count the number of parameter placeholders in a statement
///
/// Returns (has_params, max_param_index)
fn count_parameters(stmt: &Statement) -> (bool, usize) {
    use crate::parser::ast::*;

    struct ParamCounter {
        max_index: usize,
        has_positional: bool,
    }

    impl ParamCounter {
        fn new() -> Self {
            Self {
                max_index: 0,
                has_positional: false,
            }
        }

        fn visit_expr(&mut self, expr: &Expression) {
            match expr {
                Expression::Parameter(param) => {
                    if param.index > 0 {
                        self.max_index = self.max_index.max(param.index);
                    } else {
                        // Positional parameter (?)
                        self.has_positional = true;
                    }
                }
                Expression::Infix(infix) => {
                    self.visit_expr(&infix.left);
                    self.visit_expr(&infix.right);
                }
                Expression::Prefix(prefix) => {
                    self.visit_expr(&prefix.right);
                }
                Expression::FunctionCall(func) => {
                    for arg in &func.arguments {
                        self.visit_expr(arg);
                    }
                }
                Expression::Case(case) => {
                    if let Some(val) = &case.value {
                        self.visit_expr(val);
                    }
                    for when in &case.when_clauses {
                        self.visit_expr(&when.condition);
                        self.visit_expr(&when.then_result);
                    }
                    if let Some(el) = &case.else_value {
                        self.visit_expr(el);
                    }
                }
                Expression::In(in_expr) => {
                    self.visit_expr(&in_expr.left);
                    self.visit_expr(&in_expr.right);
                }
                Expression::Between(between) => {
                    self.visit_expr(&between.expr);
                    self.visit_expr(&between.lower);
                    self.visit_expr(&between.upper);
                }
                Expression::Cast(cast) => {
                    self.visit_expr(&cast.expr);
                }
                Expression::ScalarSubquery(subq) => {
                    self.visit_select(&subq.subquery);
                }
                Expression::Exists(exists) => {
                    self.visit_select(&exists.subquery);
                }
                Expression::List(list) => {
                    for item in &list.elements {
                        self.visit_expr(item);
                    }
                }
                Expression::ExpressionList(list) => {
                    for item in &list.expressions {
                        self.visit_expr(item);
                    }
                }
                Expression::Aliased(aliased) => {
                    self.visit_expr(&aliased.expression);
                }
                Expression::Window(window) => {
                    // WindowExpression.function is Box<FunctionCall>, visit its arguments
                    for arg in &window.function.arguments {
                        self.visit_expr(arg);
                    }
                }
                _ => {}
            }
        }

        fn visit_select(&mut self, select: &SelectStatement) {
            // Visit columns
            for col in &select.columns {
                self.visit_expr(col);
            }
            // Visit table expression (may contain subqueries)
            if let Some(table_expr) = &select.table_expr {
                self.visit_expr(table_expr);
            }
            // Visit where clause
            if let Some(where_clause) = &select.where_clause {
                self.visit_expr(where_clause);
            }
            // Visit group by
            for group in &select.group_by.columns {
                self.visit_expr(group);
            }
            // Visit having
            if let Some(having) = &select.having {
                self.visit_expr(having);
            }
        }
    }

    let mut counter = ParamCounter::new();

    match stmt {
        Statement::Select(select) => counter.visit_select(select),
        Statement::Insert(insert) => {
            for row in &insert.values {
                for expr in row {
                    counter.visit_expr(expr);
                }
            }
        }
        Statement::Update(update) => {
            for expr in update.updates.values() {
                counter.visit_expr(expr);
            }
            if let Some(where_clause) = &update.where_clause {
                counter.visit_expr(where_clause);
            }
        }
        Statement::Delete(delete) => {
            if let Some(where_clause) = &delete.where_clause {
                counter.visit_expr(where_clause);
            }
        }
        _ => {}
    }

    let has_params = counter.max_index > 0 || counter.has_positional;
    let param_count = if counter.has_positional {
        // For positional params, we can't know the count statically
        0
    } else {
        counter.max_index
    };

    (has_params, param_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mvcc::engine::MVCCEngine;

    fn create_test_executor() -> Executor {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        Executor::new(Arc::new(engine))
    }

    #[test]
    fn test_executor_creation() {
        let executor = create_test_executor();
        assert!(executor.function_registry().exists("COUNT"));
        assert!(executor.function_registry().exists("UPPER"));
    }

    #[test]
    fn test_empty_program() {
        let executor = create_test_executor();
        let result = executor.execute("").unwrap();
        assert_eq!(result.columns().len(), 0);
    }

    #[test]
    fn test_create_table() {
        let executor = create_test_executor();
        let result = executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);
    }

    #[test]
    fn test_insert_and_select() {
        let executor = create_test_executor();

        // Create table
        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();

        // Insert data
        let result = executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);

        // Select data
        let mut result = executor.execute("SELECT * FROM users").unwrap();
        let columns = result.columns();
        assert_eq!(columns.len(), 2);

        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("Alice")));

        assert!(!result.next());
    }

    #[test]
    fn test_parameterized_query() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
            .unwrap();

        let mut result = executor
            .execute_with_params("SELECT * FROM users WHERE id = $1", &[Value::Integer(1)])
            .unwrap();

        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert!(!result.next());
    }

    #[test]
    fn test_query_cache_basic() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();

        // First execution - should parse and cache
        let stats_before = executor.cache_stats();
        executor.execute("SELECT * FROM users").unwrap();
        let stats_after = executor.cache_stats();
        assert!(stats_after.size > stats_before.size);

        // Second execution - should use cache
        let size_before = executor.cache_stats().size;
        executor.execute("SELECT * FROM users").unwrap();
        let size_after = executor.cache_stats().size;
        assert_eq!(size_before, size_after); // No new entries
    }

    #[test]
    fn test_query_cache_parameterized() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        executor
            .execute("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
            .unwrap();

        // Execute with different parameters - should reuse cached plan
        let query = "SELECT * FROM users WHERE id = $1";

        // First execution
        let mut result = executor
            .execute_with_params(query, &[Value::Integer(1)])
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));

        // Second execution with different param - should use cache
        let mut result = executor
            .execute_with_params(query, &[Value::Integer(2)])
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));
    }

    #[test]
    fn test_query_cache_clear() {
        let executor = create_test_executor();

        executor.execute("SELECT 1").unwrap();
        executor.execute("SELECT 2").unwrap();
        assert!(executor.cache_stats().size > 0);

        executor.clear_cache();
        assert_eq!(executor.cache_stats().size, 0);
    }

    #[test]
    fn test_query_cache_whitespace_normalization() {
        let executor = create_test_executor();

        executor.execute("SELECT  1").unwrap();
        let size = executor.cache_stats().size;

        // Same query with different whitespace should hit cache
        executor.execute("SELECT 1").unwrap();
        assert_eq!(executor.cache_stats().size, size);
    }
}
