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
//! FilteredResult (WHERE clause with pre-compiled RowFilter)
//!   ↓
//! HashJoinOperator (JOIN operations)
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
//! - [`ExprVM`] - Expression virtual machine for evaluation
//! - [`ExecResult`] - Base result type for DML operations
//! - Various result wrappers for query pipeline

pub mod context;
pub mod expression;
pub mod hash_table;
pub mod join_executor;
pub mod operator;
pub mod operators;
pub mod parallel;
pub mod planner;
pub mod query_cache;
pub mod result;
pub mod semantic_cache;
pub mod statistics;

mod aggregation;
mod copy;
mod cte;
mod ddl;
mod dml;
mod dml_fast_path;
mod explain;
pub(crate) mod expr_converter;
mod foreign_key;
mod index_optimizer;
mod pk_fast_path;
pub mod pushdown;
mod query;
mod query_classification;
mod set_ops;
mod show;
mod subquery;
pub mod utils;
mod window;

use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::api::params::ParamVec;
use crate::core::{Error, Result, Value};
use crate::functions::FunctionRegistry;

/// Default function registry - shared across all executors to avoid per-database allocation
static DEFAULT_FUNCTION_REGISTRY: OnceLock<Arc<FunctionRegistry>> = OnceLock::new();

/// Get the default function registry (lazily initialized, shared across all executors)
#[inline]
fn default_function_registry() -> Arc<FunctionRegistry> {
    DEFAULT_FUNCTION_REGISTRY
        .get_or_init(|| Arc::new(FunctionRegistry::new()))
        .clone()
}
use crate::parser::ast::{Program, Statement};
use crate::parser::Parser;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};

pub use context::{clear_all_thread_local_caches, ExecutionContext, TimeoutGuard};
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
    ColumnStatsCache, QueryPlanner, RuntimeJoinAlgorithm, RuntimeJoinDecision, StatsHealth,
};
use query_cache::CompiledExecution;
pub use query_cache::{CacheStats, CachedPlanRef, CachedQueryPlan, QueryCache, DEFAULT_CACHE_SIZE};
pub use query_classification::clear_classification_cache;
pub use result::{ColumnarResult, ExecResult, ExecutorResult};
pub use semantic_cache::{
    CacheLookupResult, CachedResult, QueryFingerprint, SemanticCache, SemanticCacheStats,
    SemanticCacheStatsSnapshot, SubsumptionResult, DEFAULT_CACHE_TTL_SECS, DEFAULT_MAX_CACHED_ROWS,
    DEFAULT_SEMANTIC_CACHE_SIZE,
};

// New streaming operator infrastructure
pub use hash_table::{hash_row_keys, verify_key_equality, JoinHashTable};
pub use join_executor::{JoinAnalysis, JoinExecutor, JoinRequest, JoinResult};
pub use operator::{
    ColumnInfo, CompositeRow, EmptyOperator, MaterializedOperator, Operator, RowRef,
};
pub use operators::{
    BatchIndexNestedLoopJoinOperator, HashJoinOperator, IndexLookupStrategy,
    IndexNestedLoopJoinOperator, JoinProjection, JoinSide, JoinType, MergeJoinOperator,
    NestedLoopJoinOperator,
};
pub use utils::{compute_join_projection, extract_join_keys_and_residual, JoinProjectionIndices};

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
            function_registry: default_function_registry(),
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
            function_registry: default_function_registry(),
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
    /// Note: Callers should try try_fast_path_with_params() first before calling this.
    pub fn execute_with_params(&self, sql: &str, params: ParamVec) -> Result<Box<dyn QueryResult>> {
        let ctx = ExecutionContext::with_params(params);
        self.execute_cached(sql, &ctx)
    }

    /// Try fast path execution with borrowed params slice
    /// Returns None if fast path doesn't apply, Some(result) otherwise
    pub fn try_fast_path_with_params(
        &self,
        sql: &str,
        params: &[Value],
    ) -> Option<Result<Box<dyn QueryResult>>> {
        if let Err(error) = self.engine.check_health() {
            return Some(Err(error));
        }
        // Quick reject: if in explicit transaction, skip fast path
        {
            let active_tx = match self.active_transaction.try_lock() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            if active_tx.is_some() {
                return None;
            }
        }

        // Try to get from cache
        let cached = self.query_cache.get(sql)?;

        // Validate parameter count
        if cached.has_params && params.len() < cached.param_count {
            return None; // Let normal path handle error
        }

        // Unsupported statement kinds use normal dispatch. Do not register
        // and immediately discard a read epoch on every prepared INSERT.
        if !matches!(
            cached.statement.as_ref(),
            Statement::Select(_) | Statement::Update(_) | Statement::Delete(_)
        ) {
            return None;
        }

        // These borrowed-parameter helpers only consume an already compiled
        // matching plan. Unknown/stale plans must return to normal dispatch,
        // where compilation happens, without creating an unused epoch first.
        {
            let compiled = cached.compiled.read().ok()?;
            let cached_epoch = match (cached.statement.as_ref(), &*compiled) {
                (Statement::Select(_), CompiledExecution::PkLookup(plan)) => plan.cached_epoch,
                (Statement::Update(_), CompiledExecution::PkUpdate(plan)) => plan.cached_epoch,
                (Statement::Delete(_), CompiledExecution::PkDelete(plan)) => plan.cached_epoch,
                _ => return None,
            };
            if self.engine.schema_epoch() != cached_epoch {
                return None;
            }
        }

        // The borrowed-parameter entry point bypasses normal dispatch, but
        // still registers the same statement visibility boundary.
        let mut ctx = ExecutionContext::new();
        let epoch = match self.engine.capture_read_epoch() {
            Ok(epoch) => epoch,
            Err(error) => return Some(Err(error)),
        };
        if let Some(epoch) = &epoch {
            ctx = ctx.with_read_epoch(epoch.clone());
        }
        let result = match cached.statement.as_ref() {
            Statement::Select(stmt) => {
                self.try_fast_pk_lookup_with_params(stmt, params, &ctx, &cached.compiled)
            }
            Statement::Update(stmt) => {
                self.try_fast_pk_update_with_params(stmt, params, &ctx, &cached.compiled)
            }
            Statement::Delete(stmt) => {
                self.try_fast_pk_delete_with_params(stmt, params, &ctx, &cached.compiled)
            }
            _ => None,
        };
        result.map(|result| {
            result.map(|result| match epoch {
                Some(epoch) if !result.columns().is_empty() => {
                    Box::new(result::EpochResult::new(result, epoch)) as Box<dyn QueryResult>
                }
                _ => result,
            })
        })
    }

    /// Execute a SQL query with named parameters
    ///
    /// Parameters are substituted for :name placeholders in the query.
    /// Uses the query cache for efficient re-execution of parameterized queries.
    pub fn execute_with_named_params(
        &self,
        sql: &str,
        params: FxHashMap<String, Value>,
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
        self.engine.check_health()?;
        // Try to get from cache
        if let Some(cached) = self.query_cache.get(sql) {
            return self.execute_with_cached_plan(&cached, ctx);
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
            let cached_plan = self
                .query_cache
                .put(sql, stmt_arc.clone(), has_params, param_count);

            return self.execute_with_cached_plan(&cached_plan, ctx);
        }

        self.execute_program_with_context(&program, ctx)
    }

    /// Current explicit-transaction id, if one is active.
    #[inline]
    fn active_txn_id(&self) -> Option<i64> {
        self.active_transaction
            .lock()
            .unwrap()
            .as_ref()
            .map(|t| t.transaction.id())
    }

    /// Run the compiled fast-path battery for a single cached statement.
    /// Callers dispatch INSERT before calling this (its compiled path
    /// handles the active transaction itself).
    ///
    /// `in_txn` gates the SELECT/UPDATE/DELETE probes: they read committed
    /// state only, so inside an explicit transaction they must not fire.
    #[inline]
    fn try_compiled_fast_paths(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        compiled: &Arc<std::sync::RwLock<query_cache::CompiledExecution>>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        match statement {
            Statement::Select(stmt) if !in_txn => {
                if let Some(result) = self.try_fast_pk_lookup_compiled(stmt, ctx, compiled) {
                    return Some(result);
                }
                if let Some(result) = self.try_fast_count_distinct_compiled(stmt, ctx, compiled) {
                    return Some(result);
                }
                self.try_fast_count_star_compiled(stmt, ctx, compiled)
            }
            Statement::Update(stmt) if !in_txn => {
                self.try_fast_pk_update_compiled(stmt, ctx, compiled)
            }
            Statement::Delete(stmt) if !in_txn => {
                self.try_fast_pk_delete_compiled(stmt, ctx, compiled)
            }
            _ => None,
        }
    }

    pub(crate) fn fetch_rows_in_context(
        &self,
        table_name: &str,
        row_ids: &[i64],
        ctx: &ExecutionContext,
    ) -> Result<crate::core::RowVec> {
        let table = self.table_in_context(table_name, ctx)?;
        table.fetch_rows_by_ids(
            row_ids,
            &crate::storage::expression::logical::ConstBoolExpr::true_expr(),
        )
    }

    pub(crate) fn table_in_context(
        &self,
        name: &str,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn Table>> {
        let active = self
            .active_transaction
            .lock()
            .map_err(|_| Error::internal("active transaction lock is poisoned"))?;
        if let Some(tx) = active.as_ref() {
            return ctx.get_table(tx.transaction.as_ref(), name);
        }
        drop(active);
        if let Some(snapshot) = ctx.statement_snapshot() {
            return snapshot.get_table(name);
        }
        let tx = self.engine.begin_transaction()?;
        ctx.get_table(tx.as_ref(), name)
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
        self.with_statement_epoch(statement, ctx, |ctx| {
            self.execute_statement_inner(statement, ctx, self.active_txn_id(), None)
        })
    }

    /// Capture exactly once before any compiled or regular read. Context
    /// clones used for nested queries share this lease; each top-level call
    /// starts with its caller's unmodified context.
    fn with_statement_epoch<F>(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        action: F,
    ) -> Result<Box<dyn QueryResult>>
    where
        F: FnOnce(&ExecutionContext) -> Result<Box<dyn QueryResult>>,
    {
        if matches!(
            statement,
            Statement::Begin(_)
                | Statement::Commit(_)
                | Statement::Rollback(_)
                | Statement::Savepoint(_)
                | Statement::ReleaseSavepoint(_)
                | Statement::Truncate(_)
        ) {
            return action(ctx);
        }
        if let Some(epoch) = ctx.read_epoch() {
            let result = action(ctx)?;
            return if ctx.query_depth == 0 && !result.columns().is_empty() {
                Ok(Box::new(result::EpochResult::new(result, epoch.clone())))
            } else {
                Ok(result)
            };
        }
        let (epoch, txn_id) = {
            let active = self
                .active_transaction
                .lock()
                .map_err(|_| Error::internal("active transaction lock is poisoned"))?;
            match active.as_ref() {
                Some(tx) => (
                    tx.transaction.capture_read_epoch()?,
                    Some(tx.transaction.id()),
                ),
                None => (self.engine.capture_read_epoch()?, None),
            }
        };
        let Some(epoch) = epoch else {
            return action(ctx);
        };
        let mut statement_ctx = ctx.with_read_epoch(epoch.clone());
        if let Some(txn_id) = txn_id {
            statement_ctx.set_transaction_id(txn_id as u64);
        }
        let result = action(&statement_ctx)?;
        if result.columns().is_empty() {
            Ok(result)
        } else {
            Ok(Box::new(result::EpochResult::new(result, epoch)))
        }
    }

    /// Execute a single statement with pre-captured transaction state.
    ///
    /// `plan_slots` is Some for statements coming through the query cache:
    /// it carries the plan's classification slot, and doubles as the
    /// "compiled probe battery already ran" signal (the battery ran or was
    /// skipped for an explicit transaction, where the uncompiled probe
    /// would bail too).
    fn execute_statement_inner(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        active_txn_id: Option<i64>,
        plan_classification: Option<
            &std::sync::OnceLock<Arc<query_classification::QueryClassification>>,
        >,
    ) -> Result<Box<dyn QueryResult>> {
        self.engine.check_health()?;
        // If there's an active transaction, inject the transaction ID into the context
        // This enables CURRENT_TRANSACTION_ID() function to return the correct value
        let ctx_with_txn;
        let ctx = match active_txn_id {
            Some(txn_id) => {
                ctx_with_txn = ctx.with_transaction_id(txn_id as u64);
                &ctx_with_txn
            }
            None => ctx,
        };

        match statement {
            // DDL statements
            Statement::CreateTable(stmt) => self.execute_create_table(stmt, ctx),
            Statement::DropTable(stmt) => self.execute_drop_table(stmt, ctx),
            Statement::CreateIndex(stmt) => self.execute_create_index(stmt, ctx),
            Statement::DropIndex(stmt) => self.execute_drop_index(stmt, ctx),
            Statement::AlterTable(stmt) => self.execute_alter_table(stmt, ctx),
            Statement::CreateView(stmt) => self.execute_create_view(stmt, ctx),
            Statement::DropView(stmt) => self.execute_drop_view(stmt, ctx),

            // DML statements
            Statement::Insert(stmt) => self.execute_insert(stmt, ctx),
            Statement::Update(stmt) => self.execute_update(stmt, ctx),
            Statement::Delete(stmt) => self.execute_delete(stmt, ctx),
            Statement::Truncate(stmt) => self.execute_truncate(stmt, ctx),

            // Query statements - try fast-path first for simple PK lookups
            Statement::Select(stmt) => {
                // Fast-path for simple PK lookups (bypasses full planner);
                // cached plans already ran the compiled probe battery.
                if plan_classification.is_none() {
                    if let Some(result) = self.try_fast_pk_lookup(stmt, ctx) {
                        return result;
                    }
                }
                // Fall back to full query execution
                self.execute_select_with_plan(stmt, ctx, plan_classification)
            }

            // Transaction control
            Statement::Begin(stmt) => self.execute_begin(stmt, ctx),
            Statement::Commit(stmt) => self.execute_commit_stmt(stmt, ctx),
            Statement::Rollback(stmt) => self.execute_rollback_stmt(stmt, ctx),
            Statement::Savepoint(stmt) => self.execute_savepoint(stmt, ctx),
            Statement::ReleaseSavepoint(stmt) => self.execute_release_savepoint(stmt, ctx),

            // Utility statements
            Statement::Set(stmt) => self.execute_set(stmt, ctx),
            Statement::ShowTables(stmt) => self.execute_show_tables(stmt, ctx),
            Statement::ShowViews(stmt) => self.execute_show_views(stmt, ctx),
            Statement::ShowCreateTable(stmt) => self.execute_show_create_table(stmt, ctx),
            Statement::ShowCreateView(stmt) => self.execute_show_create_view(stmt, ctx),
            Statement::ShowIndexes(stmt) => self.execute_show_indexes(stmt, ctx),
            Statement::Describe(stmt) => self.execute_describe(stmt, ctx),
            Statement::Pragma(stmt) => self.execute_pragma(stmt, ctx),
            Statement::Expression(stmt) => self.execute_expression_stmt(stmt, ctx),
            Statement::Explain(stmt) => self.execute_explain(stmt, ctx),
            Statement::Analyze(stmt) => self.execute_analyze(stmt, ctx),
            Statement::Vacuum(stmt) => self.execute_vacuum(stmt, ctx),
            Statement::Copy(stmt) => self.execute_copy(stmt, ctx),
        }
    }

    /// Explicit transactions retain earlier successful statements on an error.
    /// Autocommit already owns an entire disposable storage transaction.
    fn with_statement_rollback<F>(&self, action: F) -> Result<Box<dyn QueryResult>>
    where
        F: FnOnce() -> Result<Box<dyn QueryResult>>,
    {
        let checkpoint = {
            let mut active = self
                .active_transaction
                .lock()
                .map_err(|_| Error::internal("active transaction lock is poisoned"))?;
            match active.as_mut() {
                Some(tx) => tx.transaction.begin_statement()?,
                None => false,
            }
        };
        let result = action();
        if checkpoint {
            let Ok(mut active) = self.active_transaction.lock() else {
                panic!("active transaction lock is poisoned during statement cleanup");
            };
            if let Some(tx) = active.as_mut() {
                tx.transaction.finish_statement(result.is_ok());
            }
        }
        result
    }

    /// Install an external storage transaction as the active transaction.
    ///
    /// Used by the programmatic Transaction API to delegate SELECT queries
    /// to the full executor pipeline (aggregates, JOINs, window functions, etc.)
    /// while keeping the transaction's uncommitted changes visible.
    pub fn install_transaction(&self, tx: Box<dyn Transaction>) {
        let mut active_tx = self.active_transaction.lock().unwrap();
        *active_tx = Some(ActiveTransaction {
            transaction: tx,
            tables: FxHashMap::default(),
        });
    }

    /// Take back the storage transaction from the active transaction slot.
    ///
    /// Returns the transaction so the caller can continue using it for
    /// further DML operations after the SELECT delegation completes.
    pub fn take_transaction(&self) -> Option<Box<dyn Transaction>> {
        let mut active_tx = self.active_transaction.lock().unwrap();
        active_tx.take().map(|at| at.transaction)
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
        self.engine.begin_transaction_with_level(isolation)
    }

    /// Get or create a cached plan for a SQL statement.
    ///
    /// Parses the SQL and caches the plan if not already cached.
    /// Returns a lightweight CachedPlanRef that can be stored and reused
    /// for repeated execution without re-parsing or cache lookup overhead.
    pub fn get_or_create_plan(&self, sql: &str) -> Result<CachedPlanRef> {
        if let Some(plan) = self.query_cache.get(sql) {
            return Ok(plan);
        }
        let mut parser = Parser::new(sql);
        let mut program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;
        if program.statements.len() != 1 {
            return Err(Error::parse(
                "Prepared statements must contain exactly one statement",
            ));
        }
        let stmt = program.statements.pop().unwrap();
        let (has_params, param_count) = count_parameters(&stmt);
        Ok(self
            .query_cache
            .put(sql, Arc::new(stmt), has_params, param_count))
    }

    /// Execute a pre-cached plan directly, skipping cache lookup.
    ///
    /// This is the fast path for prepared statements: the caller holds a
    /// `CachedPlanRef` obtained from `get_or_create_plan()` and passes it
    /// here on every execution, avoiding normalize + hash + RwLock read
    /// per call.
    pub fn execute_with_cached_plan(
        &self,
        plan: &CachedPlanRef,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.with_statement_epoch(&plan.statement, ctx, |ctx| {
            self.execute_cached_plan_inner(plan, ctx)
        })
    }

    fn execute_cached_plan_inner(
        &self,
        plan: &CachedPlanRef,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.engine.check_health()?;
        // Validate parameter count
        if plan.has_params {
            let provided = ctx.params().len();
            if provided < plan.param_count {
                return Err(Error::internal(format!(
                    "Query requires {} parameters but only {} provided",
                    plan.param_count, provided
                )));
            }
        }

        // INSERT handles the active transaction itself; dispatch before the
        // txn-state capture so prepared INSERT takes the mutex only once.
        if let Statement::Insert(stmt) = plan.statement.as_ref() {
            return self.execute_insert_with_compiled_cache(stmt, ctx, &plan.compiled);
        }

        let active_txn_id = self.active_txn_id();
        if let Some(result) = self.try_compiled_fast_paths(
            plan.statement.as_ref(),
            ctx,
            &plan.compiled,
            active_txn_id.is_some(),
        ) {
            return result;
        }

        self.execute_statement_inner(
            &plan.statement,
            ctx,
            active_txn_id,
            Some(&plan.classification),
        )
    }
}

/// Count the number of parameter placeholders in a statement
///
/// Returns (has_params, max_param_index)
pub(crate) fn count_parameters(stmt: &Statement) -> (bool, usize) {
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

    fn epoch_statement() -> Statement {
        Parser::new("SELECT 1")
            .parse_program()
            .unwrap()
            .statements
            .remove(0)
    }

    #[test]
    fn statement_epoch_is_shared_by_nested_and_derived_contexts() {
        let executor = create_test_executor();
        let registry = executor.engine.registry();
        let statement = epoch_statement();
        executor
            .with_statement_epoch(&statement, &ExecutionContext::new(), |ctx| {
                let epoch = ctx.read_epoch().unwrap();
                let (writer, _) = registry.begin_transaction();
                registry.complete_commit(writer);
                assert!(!epoch.is_visible(writer));
                for nested in [
                    ctx.with_incremented_query_depth(),
                    ctx.with_incremented_view_depth(),
                    ctx.with_transaction_id(99),
                ] {
                    executor.with_statement_epoch(&statement, &nested, |nested| {
                        assert!(epoch.same_epoch(nested.read_epoch().unwrap()));
                        assert!(!nested.read_epoch().unwrap().is_visible(writer));
                        Ok(Box::new(ExecResult::empty()))
                    })?;
                }
                Ok(Box::new(ExecResult::empty()))
            })
            .unwrap();
        assert_eq!(registry.oldest_retention_horizon(), None);
    }

    #[test]
    fn statement_epoch_refreshes_rc_and_preserves_si_begin() {
        for isolation in [
            crate::IsolationLevel::ReadCommitted,
            crate::IsolationLevel::SnapshotIsolation,
        ] {
            let executor = create_test_executor();
            let registry = executor.engine.registry();
            let transaction = executor
                .begin_transaction_with_isolation(isolation)
                .unwrap();
            let txn_id = transaction.id();
            executor.install_transaction(transaction);
            let statement = epoch_statement();
            let mut first_cutoff = 0;
            executor
                .with_statement_epoch(&statement, &ExecutionContext::new(), |ctx| {
                    first_cutoff = ctx.read_epoch().unwrap().cutoff();
                    assert_eq!(ctx.transaction_id(), Some(txn_id as u64));
                    Ok(Box::new(ExecResult::empty()))
                })
                .unwrap();
            let (writer, _) = registry.begin_transaction();
            registry.complete_commit(writer);
            executor
                .with_statement_epoch(&statement, &ExecutionContext::new(), |ctx| {
                    let epoch = ctx.read_epoch().unwrap();
                    if isolation == crate::IsolationLevel::ReadCommitted {
                        assert!(epoch.cutoff() > first_cutoff);
                        assert!(epoch.is_visible(writer));
                    } else {
                        assert_eq!(epoch.cutoff(), first_cutoff);
                        assert!(!epoch.is_visible(writer));
                    }
                    Ok(Box::new(ExecResult::empty()))
                })
                .unwrap();
            executor.execute("ROLLBACK").unwrap();
            assert_eq!(registry.oldest_retention_horizon(), None);
        }
    }

    #[test]
    fn borrowed_fast_path_rejects_insert_before_epoch_registration() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE epoch_insert (id INTEGER PRIMARY KEY)")
            .unwrap();
        let sql = "INSERT INTO epoch_insert VALUES ($1)";
        executor
            .execute_with_params(sql, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        let registry = executor.engine.registry();
        let before = registry.capture_read_epoch().cache_identity().1;
        assert!(executor
            .try_fast_path_with_params(sql, &[Value::Integer(2)])
            .is_none());
        let after = registry.capture_read_epoch().cache_identity().1;
        assert_eq!(
            after,
            before + 1,
            "a rejected dispatch must not register an epoch"
        );
        assert_eq!(registry.oldest_retention_horizon(), None);

        // The supported compiled SELECT still captures exactly one statement
        // epoch. Its independent read transaction also registers a begin slot.
        let select = "SELECT * FROM epoch_insert WHERE id = $1";
        executor
            .execute_with_params(select, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        let before_select = registry.capture_read_epoch().cache_identity().1;
        let mut result = executor
            .try_fast_path_with_params(select, &[Value::Integer(1)])
            .unwrap()
            .unwrap();
        assert_eq!(
            registry.capture_read_epoch().cache_identity().1,
            before_select + 3,
            "compiled lookup must share one statement lease plus its transaction begin slot"
        );
        assert!(registry.has_retention_obligations());
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));
        result.close().unwrap();
        assert!(!registry.has_retention_obligations());
    }

    #[test]
    fn borrowed_compiled_preflight_preserves_normal_compilation_and_schema_refresh() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE epoch_lookup (id INTEGER PRIMARY KEY, n INTEGER)")
            .unwrap();
        executor
            .execute("INSERT INTO epoch_lookup VALUES (1, 10)")
            .unwrap();
        let registry = executor.engine.registry();
        let query = "SELECT * FROM epoch_lookup WHERE id = $1";
        executor
            .execute_with_params(query, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        let cached = executor.query_cache.get(query).unwrap();
        for state in [
            CompiledExecution::Unknown,
            CompiledExecution::NotOptimizable(executor.engine.schema_epoch()),
        ] {
            *cached.compiled.write().unwrap() = state;
            let before = registry.capture_read_epoch().cache_identity().1;
            assert!(executor
                .try_fast_path_with_params(query, &[Value::Integer(1)])
                .is_none());
            assert_eq!(registry.capture_read_epoch().cache_identity().1, before + 1);
            let mut result = executor
                .execute_with_params(query, smallvec::smallvec![Value::Integer(1)])
                .unwrap();
            assert!(result.next());
            assert_eq!(result.row().get(1), Some(&Value::Integer(10)));
            result.close().unwrap();
        }
        // Unknown compilation still populates the shared slot; a later schema
        // change routes through normal recompilation instead of getting stuck.
        *cached.compiled.write().unwrap() = CompiledExecution::Unknown;
        executor
            .execute_with_params(query, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        assert!(matches!(
            &*cached.compiled.read().unwrap(),
            CompiledExecution::PkLookup(_)
        ));
        executor
            .execute("ALTER TABLE epoch_lookup ADD COLUMN extra INTEGER DEFAULT 7")
            .unwrap();
        let before = registry.capture_read_epoch().cache_identity().1;
        assert!(executor
            .try_fast_path_with_params(query, &[Value::Integer(1)])
            .is_none());
        assert_eq!(registry.capture_read_epoch().cache_identity().1, before + 1);
        let mut result = executor
            .execute_with_params(query, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(2), Some(&Value::Integer(7)));
        result.close().unwrap();
        assert!(executor
            .try_fast_path_with_params(query, &[Value::Integer(1)])
            .is_some());
        assert!(!registry.has_retention_obligations());
    }

    #[test]
    fn statement_result_retains_epoch_until_drop_or_close() {
        let executor = create_test_executor();
        let registry = executor.engine.registry();
        let mut result = executor.execute("SELECT 1").unwrap();
        assert!(registry.oldest_retention_horizon().is_some());
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));
        result.close().unwrap();
        assert_eq!(registry.oldest_retention_horizon(), None);
        assert!(!result.next());
        let result = executor.execute("SELECT 2").unwrap();
        assert!(registry.oldest_retention_horizon().is_some());
        drop(result);
        assert_eq!(registry.oldest_retention_horizon(), None);
    }

    #[test]
    fn statement_epoch_error_releases_lease() {
        let executor = create_test_executor();
        let registry = executor.engine.registry();
        let error =
            executor.with_statement_epoch(&epoch_statement(), &ExecutionContext::new(), |_ctx| {
                Err(Error::internal("statement failed"))
            });
        assert!(error.is_err());
        assert_eq!(registry.oldest_retention_horizon(), None);
    }

    #[test]
    fn statement_cache_epoch_separates_engines_without_pinning_horizons() {
        use super::context::{
            cache_scalar_subquery, ensure_statement_cache_epoch, get_cached_scalar_subquery,
        };
        let first = crate::storage::mvcc::TransactionRegistry::new();
        let second = crate::storage::mvcc::TransactionRegistry::new();
        let first_ctx = ExecutionContext::new().with_read_epoch(first.capture_read_epoch());
        let second_ctx = ExecutionContext::new().with_read_epoch(second.capture_read_epoch());
        ensure_statement_cache_epoch(&first_ctx);
        cache_scalar_subquery(
            "same key".into(),
            smallvec::SmallVec::new(),
            Value::Integer(1),
        );
        ensure_statement_cache_epoch(&first_ctx.with_incremented_query_depth());
        assert_eq!(
            get_cached_scalar_subquery("same key"),
            Some(Value::Integer(1))
        );
        ensure_statement_cache_epoch(&second_ctx);
        assert_eq!(get_cached_scalar_subquery("same key"), None);
        cache_scalar_subquery(
            "same key".into(),
            smallvec::SmallVec::new(),
            Value::Integer(2),
        );
        // Resuming an older lazy query must not consume the newer query's memo.
        ensure_statement_cache_epoch(&first_ctx);
        assert_eq!(get_cached_scalar_subquery("same key"), None);
        drop(first_ctx);
        drop(second_ctx);
        assert_eq!(first.oldest_retention_horizon(), None);
        assert_eq!(second.oldest_retention_horizon(), None);
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
            .execute_with_params(
                "SELECT * FROM users WHERE id = $1",
                smallvec::smallvec![Value::Integer(1)],
            )
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
            .execute_with_params(query, smallvec::smallvec![Value::Integer(1)])
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));

        // Second execution with different param - should use cache
        let mut result = executor
            .execute_with_params(query, smallvec::smallvec![Value::Integer(2)])
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
