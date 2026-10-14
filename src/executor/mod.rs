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
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

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

/// The statement gate held for one entry into the executor
enum StatementGate<'a> {
    Shared { _read: RwLockReadGuard<'a, ()> },
    Exclusive { _write: RwLockWriteGuard<'a, ()> },
}

/// Whether `statement` changes this session's state, so it runs alone.
/// EXPLAIN ANALYZE runs the statement it explains.
fn changes_session(statement: &Statement) -> bool {
    matches!(
        statement,
        Statement::Begin(_)
            | Statement::Commit(_)
            | Statement::Rollback(_)
            | Statement::Savepoint(_)
            | Statement::ReleaseSavepoint(_)
            | Statement::Set(_)
            | Statement::Pragma(_)
            | Statement::Explain(_)
    )
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
    /// Default isolation level for transactions, as its `u8` value
    default_isolation_level: AtomicU8,
    /// Query cache for parsed statements
    query_cache: QueryCache,
    /// Semantic cache for query results with subsumption detection
    semantic_cache: SemanticCache,
    /// Taken before `active_transaction`: shared by statements that run
    /// outside an explicit transaction, exclusive for the rest
    statement_gate: RwLock<()>,
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
            default_isolation_level: AtomicU8::new(
                crate::core::IsolationLevel::ReadCommitted as u8,
            ),
            query_cache: QueryCache::default(),
            semantic_cache: SemanticCache::default(),
            statement_gate: RwLock::new(()),
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
            default_isolation_level: AtomicU8::new(
                crate::core::IsolationLevel::ReadCommitted as u8,
            ),
            query_cache: QueryCache::default(),
            semantic_cache: SemanticCache::default(),
            statement_gate: RwLock::new(()),
            active_transaction: Mutex::new(None),
            query_planner: std::sync::OnceLock::new(),
        }
    }

    /// Create a new executor with a custom cache size
    pub fn with_cache_size(engine: Arc<MVCCEngine>, cache_size: usize) -> Self {
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: AtomicU8::new(
                crate::core::IsolationLevel::ReadCommitted as u8,
            ),
            query_cache: QueryCache::new(cache_size),
            semantic_cache: SemanticCache::default(),
            statement_gate: RwLock::new(()),
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

    /// Set the default isolation level for new transactions
    pub fn set_default_isolation_level(&self, level: crate::core::IsolationLevel) {
        self.default_isolation_level
            .store(level as u8, Ordering::Relaxed);
    }

    /// The gate for one entry: shared while this session has no explicit
    /// transaction and `exclusive` is false, exclusive otherwise
    fn enter(&self, exclusive: bool) -> StatementGate<'_> {
        if !exclusive {
            if let Some(read) = self.enter_shared() {
                return StatementGate::Shared { _read: read };
            }
        }
        #[cfg(feature = "test-failpoints")]
        if let Err(std::sync::TryLockError::WouldBlock) = self.statement_gate.try_write() {
            crate::test_failpoints::statement_gate_blocked();
        }
        StatementGate::Exclusive {
            _write: self
                .statement_gate
                .write()
                .unwrap_or_else(PoisonError::into_inner),
        }
    }

    /// The shared gate, or None while an explicit transaction is open
    fn enter_shared(&self) -> Option<RwLockReadGuard<'_, ()>> {
        #[cfg(feature = "test-failpoints")]
        if let Err(std::sync::TryLockError::WouldBlock) = self.statement_gate.try_read() {
            crate::test_failpoints::statement_gate_blocked();
        }
        let read = self
            .statement_gate
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let no_transaction = self
            .active_transaction
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none();
        no_transaction.then_some(read)
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
        // Quick reject: if in explicit transaction, skip fast path
        let _gate = self.enter_shared()?;

        // Try to get from cache
        let cached = self.query_cache.get(sql)?;

        // Validate parameter count
        if cached.has_params && params.len() < cached.param_count {
            return None; // Let normal path handle error
        }

        // Try compiled fast paths based on statement type
        match cached.statement.as_ref() {
            Statement::Select(stmt) => {
                self.try_fast_pk_lookup_with_params(stmt, params, &cached.compiled)
            }
            Statement::Update(stmt) => {
                self.try_fast_pk_update_with_params(stmt, params, &cached.compiled)
            }
            Statement::Delete(stmt) => {
                self.try_fast_pk_delete_with_params(stmt, params, &cached.compiled)
            }
            _ => None,
        }
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

            let gate = self.enter(changes_session(&cached.statement));

            // INSERT handles the active transaction itself; dispatch before
            // the txn-state capture so the hot prepared-INSERT path takes
            // the transaction mutex only once.
            if let Statement::Insert(stmt) = cached.statement.as_ref() {
                return self.execute_insert_with_compiled_cache(stmt, ctx, &cached.compiled);
            }

            // Capture transaction state once for the rest of the statement:
            // the probe battery and the ctx txn-id injection both need it,
            // and repeated mutex acquisitions dominate the fixed cost of
            // cached statements.
            let active_txn_id = self.gated_txn_id(&gate);

            if let Some(result) = self.try_compiled_fast_paths(
                cached.statement.as_ref(),
                ctx,
                &cached.compiled,
                active_txn_id.is_some(),
            ) {
                return result;
            }

            // Execute the cached statement (standard path)
            return self.execute_statement_inner(
                &cached.statement,
                ctx,
                active_txn_id,
                Some(&cached.classification),
                Some(&cached.compiled),
            );
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

            let gate = self.enter(changes_session(&stmt_arc));

            if let Statement::Insert(stmt) = stmt_arc.as_ref() {
                return self.execute_insert_with_compiled_cache(stmt, ctx, &cached_plan.compiled);
            }

            let active_txn_id = self.gated_txn_id(&gate);

            if let Some(result) = self.try_compiled_fast_paths(
                stmt_arc.as_ref(),
                ctx,
                &cached_plan.compiled,
                active_txn_id.is_some(),
            ) {
                return result;
            }

            // Execute directly from the Arc (no clone needed)
            return self.execute_statement_inner(
                &stmt_arc,
                ctx,
                active_txn_id,
                Some(&cached_plan.classification),
                Some(&cached_plan.compiled),
            );
        }

        // One gate for the whole program, so no other statement on this
        // session runs between its statements
        let _gate = self.enter(true);
        self.run_program(&program, ctx)
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

    /// The explicit-transaction id under `gate`: none can be open while the
    /// gate is shared, since opening one takes it exclusively
    #[inline]
    fn gated_txn_id(&self, gate: &StatementGate<'_>) -> Option<i64> {
        match gate {
            StatementGate::Shared { .. } => None,
            StatementGate::Exclusive { .. } => self.active_txn_id(),
        }
    }

    /// Run the compiled fast-path battery for a single cached statement.
    /// Callers dispatch INSERT before calling this (its compiled path
    /// handles the active transaction itself).
    ///
    /// Inside an explicit transaction (`in_txn`) only the PK lookup runs, read
    /// through the transaction; the COUNT and UPDATE/DELETE probes read
    /// committed state only and must not fire.
    #[inline]
    fn try_compiled_fast_paths(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        compiled: &Arc<std::sync::RwLock<query_cache::CompiledExecution>>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        match statement {
            Statement::Select(stmt) => {
                if let Some(result) = self.try_fast_pk_lookup_compiled(stmt, ctx, compiled, in_txn)
                {
                    return Some(result);
                }
                if in_txn {
                    return None;
                }
                if let Some(result) = self.try_fast_count_distinct_compiled(stmt, compiled) {
                    return Some(result);
                }
                self.try_fast_count_star_compiled(stmt, compiled)
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

    /// The stamp a semantic cache read of `table_name` happens under now, or
    /// None while a commit is publishing to it or its table is being replaced
    fn semantic_cache_stamp(&self, table_name: &str) -> Option<semantic_cache::CacheStamp> {
        let schema = self.engine.schema_epoch();
        let store = self.engine.get_version_store(table_name).ok()?;
        let publish = store.publish_epoch_if_quiet()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::cache_stamp_publish_read();
        // DDL bumps the schema epoch after swapping the store, so an unchanged
        // epoch means the publish epoch belongs to this table's store
        (self.engine.schema_epoch() == schema)
            .then_some(semantic_cache::CacheStamp { publish, schema })
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
        let _gate = self.enter(true);
        self.run_program(program, ctx)
    }

    /// Runs every statement of `program` under a gate the caller holds
    fn run_program(
        &self,
        program: &Program,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        if program.statements.is_empty() {
            return Ok(Box::new(ExecResult::empty()));
        }

        let mut last_result: Option<Box<dyn QueryResult>> = None;

        for statement in &program.statements {
            last_result = Some(self.run_statement(statement, ctx)?);
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::program_statement_done();
        }

        Ok(last_result.unwrap())
    }

    /// Execute a single statement
    pub fn execute_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let _gate = self.enter(changes_session(statement));
        self.run_statement(statement, ctx)
    }

    /// Runs one statement under a gate the caller holds
    pub(crate) fn run_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_statement_inner(statement, ctx, self.active_txn_id(), None, None)
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
        compiled: Option<&std::sync::RwLock<query_cache::CompiledExecution>>,
    ) -> Result<Box<dyn QueryResult>> {
        // A SELECT clears them where its top level starts
        if !matches!(statement, Statement::Select(_)) {
            context::clear_statement_caches();
        }
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
            Statement::Insert(stmt) => self.execute_insert(stmt, ctx, compiled),
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
                self.execute_select_with_plan(stmt, ctx, plan_classification, compiled)
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

    /// Install an external storage transaction as the active transaction.
    ///
    /// Used by the programmatic Transaction API to delegate SELECT queries
    /// to the full executor pipeline (aggregates, JOINs, window functions, etc.)
    /// while keeping the transaction's uncommitted changes visible.
    pub fn install_transaction(&self, tx: Box<dyn Transaction>) {
        let _gate = self.enter(true);
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
        let _gate = self.enter(true);
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

        let gate = self.enter(changes_session(&plan.statement));

        // INSERT handles the active transaction itself; dispatch before the
        // txn-state capture so prepared INSERT takes the mutex only once.
        if let Statement::Insert(stmt) = plan.statement.as_ref() {
            return self.execute_insert_with_compiled_cache(stmt, ctx, &plan.compiled);
        }

        let active_txn_id = self.gated_txn_id(&gate);
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
            Some(&plan.compiled),
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

        executor.execute("SELECT 1").unwrap();
        let size = executor.cache_stats().size;

        // Whitespace around the query shares its plan; whitespace inside does not
        executor.execute("  SELECT 1\n").unwrap();
        assert_eq!(executor.cache_stats().size, size);
        executor.execute("SELECT  1").unwrap();
        assert_eq!(executor.cache_stats().size, size + 1);
    }
}
