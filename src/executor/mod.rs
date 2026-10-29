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
mod explain;
pub(crate) mod expr_key;
mod foreign_key;
mod grouped_aggregate;
mod index_optimizer;
mod plan_reads;
pub(crate) mod planned;
pub(crate) mod planned_aggregate;
pub(crate) mod planned_dml;
mod prepared_join;
pub(crate) mod prepared_select;
pub(crate) mod projection;
pub mod pushdown;
mod query;
pub(crate) mod query_classification;
pub(crate) mod row_count;
mod set_ops;
mod show;
mod subquery;
pub mod utils;
mod volatility;
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
use crate::parser::ast::{ParamSlots, Program, Statement};
use crate::parser::Parser;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::traits::transaction::TableUse;
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};
use planned_dml::DmlStatement;

#[cfg(any(test, feature = "test-failpoints"))]
pub(crate) use aggregation::AggregatePath;
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
pub use query_cache::{
    CacheStats, CachedPlanRef, CachedQueryPlan, NestedPlan, QueryCache, DEFAULT_CACHE_SIZE,
};
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
pub use operator::{CompositeRow, EmptyOperator, MaterializedOperator, Operator, RowRef};
pub use operators::{
    BatchIndexNestedLoopJoinOperator, HashJoinOperator, IndexLookupStrategy,
    IndexNestedLoopJoinOperator, JoinProjection, JoinSide, JoinType, NestedLoopJoinOperator,
};
pub use utils::{compute_join_projection, extract_join_keys_and_residual, JoinProjectionIndices};

/// Active transaction state for explicit transaction control (BEGIN/COMMIT/ROLLBACK)
struct ActiveTransaction {
    /// The transaction object
    transaction: Box<dyn Transaction>,
    /// Tables accessed within this transaction (cached for proper commit/rollback)
    tables: FxHashMap<String, Box<dyn Table>>,
}

/// The session's explicit transaction, with whether one is open readable
/// without the lock: every guard writes it back as it is released
struct TransactionSlot {
    slot: Mutex<Option<ActiveTransaction>>,
    open: std::sync::atomic::AtomicBool,
}

impl TransactionSlot {
    fn new(transaction: Option<ActiveTransaction>) -> Self {
        Self {
            open: std::sync::atomic::AtomicBool::new(transaction.is_some()),
            slot: Mutex::new(transaction),
        }
    }

    fn lock(&self) -> std::sync::LockResult<SlotGuard<'_>> {
        match self.slot.lock() {
            Ok(guard) => Ok(SlotGuard {
                guard,
                open: &self.open,
            }),
            Err(poisoned) => Err(PoisonError::new(SlotGuard {
                guard: poisoned.into_inner(),
                open: &self.open,
            })),
        }
    }

    /// Whether a transaction is open, as the last released guard left it
    fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }
}

struct SlotGuard<'a> {
    guard: std::sync::MutexGuard<'a, Option<ActiveTransaction>>,
    open: &'a std::sync::atomic::AtomicBool,
}

impl std::ops::Deref for SlotGuard<'_> {
    type Target = Option<ActiveTransaction>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl std::ops::DerefMut for SlotGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        self.open.store(self.guard.is_some(), Ordering::Release);
    }
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

/// Whether `statement` reads committed rows beyond the target it writes: a
/// SELECT, or DML with a query, a subquery or a CTE. Its reads then go
/// through one statement view; the write and its target keep the newest
fn reads_through_a_view(statement: &Statement) -> bool {
    use crate::parser::ast::Expression;
    let any = |exprs: &[Expression]| exprs.iter().any(Executor::has_subqueries);
    let maybe =
        |expr: &Option<Box<Expression>>| expr.as_deref().is_some_and(Executor::has_subqueries);
    match statement {
        Statement::Select(_) => true,
        Statement::Insert(stmt) => insert_reads_through_a_view(stmt),
        Statement::Update(stmt) => {
            stmt.with.is_some()
                || maybe(&stmt.where_clause)
                || stmt.updates.values().any(Executor::has_subqueries)
                || any(&stmt.returning)
        }
        Statement::Delete(stmt) => {
            stmt.with.is_some() || maybe(&stmt.where_clause) || any(&stmt.returning)
        }
        _ => false,
    }
}

/// [`reads_through_a_view`] for an INSERT: a query source, or a subquery in
/// its values, its conflict update or its RETURNING
pub(crate) fn insert_reads_through_a_view(stmt: &crate::parser::ast::InsertStatement) -> bool {
    let any = |exprs: &[crate::parser::ast::Expression]| exprs.iter().any(Executor::has_subqueries);
    stmt.select.is_some()
        || stmt.values.iter().any(|row| any(row))
        || any(&stmt.update_expressions)
        || stmt
            .update_where
            .as_deref()
            .is_some_and(Executor::has_subqueries)
        || any(&stmt.returning)
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
    /// This session's isolation level as its `u8` value, `LEVEL_UNSET`
    /// until the session sets one
    default_isolation_level: AtomicU8,
    /// Query cache for parsed statements, shared with the executors of
    /// this session's transactions
    query_cache: Arc<QueryCache>,
    /// Semantic cache for query results with subsumption detection
    semantic_cache: Arc<SemanticCache>,
    /// Taken before `active_transaction`: shared by statements that run
    /// outside an explicit transaction, exclusive for the rest
    statement_gate: RwLock<()>,
    /// Active transaction for explicit transaction control (BEGIN/COMMIT/ROLLBACK)
    active_transaction: TransactionSlot,
    /// Query planner for cost-based optimization (lazily initialized)
    query_planner: std::sync::OnceLock<QueryPlanner>,
    /// The memory of views this executor took, for a capture to take again
    /// once no handle holds it; never locked across a capture
    view_spares: parking_lot::Mutex<
        smallvec::SmallVec<[Arc<crate::storage::mvcc::registry::ViewState>; VIEW_SPARES]>,
    >,
}

/// The view memories an executor keeps for its next statements
const VIEW_SPARES: usize = 4;

/// A session that has not set an isolation level runs at the engine's
const LEVEL_UNSET: u8 = u8::MAX;

/// The runs a SELECT gets, the first included, when a truncate or drop
/// overtakes its view
const STATEMENT_RUNS: usize = 3;

impl Executor {
    /// Create a new executor with the given storage engine
    pub fn new(engine: Arc<MVCCEngine>) -> Self {
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: AtomicU8::new(LEVEL_UNSET),
            query_cache: Arc::new(QueryCache::default()),
            semantic_cache: Arc::new(SemanticCache::default()),
            statement_gate: RwLock::new(()),
            active_transaction: TransactionSlot::new(None),
            query_planner: std::sync::OnceLock::new(),
            view_spares: parking_lot::Mutex::new(smallvec::SmallVec::new()),
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
            default_isolation_level: AtomicU8::new(LEVEL_UNSET),
            query_cache: Arc::new(QueryCache::default()),
            semantic_cache: Arc::new(SemanticCache::default()),
            statement_gate: RwLock::new(()),
            active_transaction: TransactionSlot::new(None),
            query_planner: std::sync::OnceLock::new(),
            view_spares: parking_lot::Mutex::new(smallvec::SmallVec::new()),
        }
    }

    /// An executor that runs `tx` as its open transaction, sharing this
    /// one's plans, compiled paths and result cache
    pub(crate) fn for_transaction(&self, tx: Box<dyn Transaction>) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            function_registry: Arc::clone(&self.function_registry),
            default_isolation_level: AtomicU8::new(
                self.default_isolation_level.load(Ordering::Relaxed),
            ),
            query_cache: Arc::clone(&self.query_cache),
            semantic_cache: Arc::clone(&self.semantic_cache),
            statement_gate: RwLock::new(()),
            active_transaction: TransactionSlot::new(Some(ActiveTransaction {
                transaction: tx,
                tables: FxHashMap::default(),
            })),
            query_planner: std::sync::OnceLock::new(),
            view_spares: parking_lot::Mutex::new(smallvec::SmallVec::new()),
        }
    }

    /// Get the query planner (lazily initialized)
    fn get_query_planner(&self) -> &QueryPlanner {
        self.query_planner
            .get_or_init(|| QueryPlanner::new(Arc::clone(&self.engine)))
    }

    /// Set this session's isolation level: the level of every transaction
    /// it begins without naming one, autocommit statements included. Waits
    /// for the session's running statements, which read the level more than once
    pub fn set_default_isolation_level(&self, level: crate::core::IsolationLevel) {
        let _gate = self.enter(true);
        self.store_isolation_level(level);
    }

    /// Writes the session's isolation level; the caller holds the gate
    /// exclusively
    fn store_isolation_level(&self, level: crate::core::IsolationLevel) {
        self.default_isolation_level
            .store(level as u8, Ordering::Relaxed);
    }

    /// This session's isolation level; the engine's until the session sets one
    pub fn isolation_level(&self) -> crate::core::IsolationLevel {
        let level = self.default_isolation_level.load(Ordering::Relaxed);
        crate::core::IsolationLevel::from_u8(level)
            .unwrap_or_else(|| self.engine.get_isolation_level())
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
        (!self.active_transaction.is_open()).then_some(read)
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
        self.execute_cached(sql, std::borrow::Cow::Owned(ExecutionContext::new()))
    }

    /// Execute a SQL query with positional parameters
    ///
    /// Parameters are substituted for $1, $2, etc. placeholders in the query.
    /// Uses the query cache for efficient re-execution of parameterized queries.
    /// Note: Callers should try try_fast_path_with_params() first before calling this.
    pub fn execute_with_params(&self, sql: &str, params: ParamVec) -> Result<Box<dyn QueryResult>> {
        self.execute_cached(
            sql,
            std::borrow::Cow::Owned(ExecutionContext::with_params(params)),
        )
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
        self.param_fast_paths(&cached, params)
    }

    /// [`Self::try_fast_path_with_params`] for a plan the caller holds
    pub(crate) fn try_plan_fast_path(
        &self,
        plan: &CachedPlanRef,
        params: &[Value],
    ) -> Option<Result<Box<dyn QueryResult>>> {
        let _gate = self.enter_shared()?;
        self.param_fast_paths(plan, params)
    }

    /// The compiled PK paths that take borrowed params, under a shared gate
    /// the caller holds; None when none applies
    fn param_fast_paths(
        &self,
        plan: &CachedPlanRef,
        params: &[Value],
    ) -> Option<Result<Box<dyn QueryResult>>> {
        if let Err(e) = check_param_count(&plan.params, params.len()) {
            return Some(Err(e));
        }

        // Try compiled fast paths based on statement type
        self.fast_path_until_held(&plan.statement, None, || match plan.statement.as_ref() {
            Statement::Select(stmt) => {
                self.run_select_plan(stmt, plan.compiled.slot(), None, params, false)
            }
            Statement::Update(stmt) => {
                let stmt = DmlStatement::Update(stmt);
                self.run_dml_plan(stmt, plan.compiled.slot(), None, params, false)
            }
            Statement::Delete(stmt) => {
                let stmt = DmlStatement::Delete(stmt);
                self.run_dml_plan(stmt, plan.compiled.slot(), None, params, false)
            }
            _ => None,
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
        self.execute_named_with(sql, &mut value_in(&params))
    }

    /// [`Self::execute_with_named_params`] with each value read by its name
    /// from `value_of`
    pub(crate) fn execute_named_with(
        &self,
        sql: &str,
        value_of: &mut impl FnMut(&str) -> Result<Option<Value>>,
    ) -> Result<Box<dyn QueryResult>> {
        match self.plan_or_program(sql)? {
            Ok(plan) => {
                let values = bind_named(&plan.params, value_of)?;
                if let Some(result) = self.try_plan_fast_path(&plan, &values) {
                    return result;
                }
                let ctx = ExecutionContext::with_params(values);
                self.execute_plan_in(&plan, std::borrow::Cow::Owned(ctx))
            }
            Err(program) => {
                let contexts = bind_program(&program, value_of)?;
                let _gate = self.enter(true);
                self.run_statements(&program, |i| &contexts[i])
            }
        }
    }

    /// Execute a SQL query with a full execution context
    /// Uses the query cache for efficient re-execution.
    pub fn execute_with_context(
        &self,
        sql: &str,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_cached(sql, std::borrow::Cow::Borrowed(ctx))
    }

    /// [`Self::execute_with_context`] for a context the caller gives up: a
    /// statement's view goes into it without a copy
    pub(crate) fn execute_with_owned_context(
        &self,
        sql: &str,
        ctx: ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_cached(sql, std::borrow::Cow::Owned(ctx))
    }

    /// Execute a SQL query using the query cache
    ///
    /// This method first checks the cache for a previously parsed statement.
    /// If found, it uses the cached AST. Otherwise, it parses the query
    /// and caches the result for future use.
    fn execute_cached(
        &self,
        sql: &str,
        ctx: std::borrow::Cow<'_, ExecutionContext>,
    ) -> Result<Box<dyn QueryResult>> {
        if let Some(plan) = self.query_cache.get(sql) {
            return self.execute_plan_in(&plan, ctx);
        }
        match self.plan_or_program(sql)? {
            Ok(plan) => self.execute_plan_in(&plan, ctx),
            Err(program) => {
                // One gate for the whole program, so no other statement on
                // this session runs between its statements
                let _gate = self.enter(true);
                self.run_program(&program, &ctx)
            }
        }
    }

    /// Current explicit-transaction id, if one is active.
    #[inline]
    pub(crate) fn active_txn_id(&self) -> Option<i64> {
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

    /// The open transaction's id under `gate`, and `ctx` carrying it for
    /// CURRENT_TRANSACTION_ID() on the compiled and general paths; INSERT
    /// reads the id in the lock section that takes its table
    fn gated_txn_ctx<'c>(
        &self,
        gate: &StatementGate<'_>,
        ctx: std::borrow::Cow<'c, ExecutionContext>,
    ) -> (Option<i64>, std::borrow::Cow<'c, ExecutionContext>) {
        let txn_id = self.gated_txn_id(gate);
        match txn_id {
            Some(id) => {
                let mut ctx = ctx;
                ctx.to_mut().set_transaction_id(id as u64);
                (txn_id, ctx)
            }
            None => (None, ctx),
        }
    }

    /// Run the compiled fast-path battery for a single cached statement.
    /// Callers dispatch INSERT before calling this (its compiled path
    /// handles the active transaction itself).
    ///
    /// Inside an explicit transaction (`in_txn`) the PK lookup, UPDATE and
    /// DELETE run through the transaction; the COUNT probes read committed
    /// state only and must not fire.
    ///
    /// A SELECT whose own view a truncate or drop overtook runs again, under
    /// the same bound as the regular path
    #[inline]
    fn compiled_fast_paths_until_held(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        compiled: &std::sync::RwLock<query_cache::CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        self.fast_path_until_held(statement, Some(ctx), || {
            self.try_compiled_fast_paths(statement, ctx, compiled, in_txn)
        })
    }

    /// Runs a fast path `attempt` for `statement` again, under the bound
    /// the regular path has, while a truncate or drop overtakes the view of
    /// a SELECT that took its own; `ctx` is None for borrowed parameters
    fn fast_path_until_held(
        &self,
        statement: &Statement,
        ctx: Option<&ExecutionContext>,
        mut attempt: impl FnMut() -> Option<Result<Box<dyn QueryResult>>>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        let mut runs = 1;
        loop {
            let result = attempt();
            let again = matches!(result, Some(Err(Error::ViewOvertaken)))
                && runs < STATEMENT_RUNS
                && matches!(statement, Statement::Select(_))
                && ctx.is_none_or(|ctx| ctx.statement_view().is_none())
                && !self.function_registry.has_registered_functions();
            if !again {
                return result;
            }
            let waited = match ctx {
                Some(ctx) => self.wait_out_destruction(ctx),
                None => self.wait_out_destruction(&ExecutionContext::new()),
            };
            if let Err(e) = waited {
                return Some(Err(e));
            }
            runs += 1;
        }
    }

    /// Waits, holding no lock, until no truncate or drop is taking rows
    /// away, so a view taken next is not overtaken from its start. A
    /// cancelled or timed out statement stops waiting
    fn wait_out_destruction(&self, ctx: &ExecutionContext) -> Result<()> {
        let registry = self.engine.registry_ref();
        let mut noted = false;
        while registry.destruction_running() {
            ctx.check_cancelled()?;
            if !noted {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::destruction_waited();
                noted = true;
            }
            std::thread::yield_now();
        }
        ctx.check_cancelled()
    }

    #[inline]
    fn try_compiled_fast_paths(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        compiled: &std::sync::RwLock<query_cache::CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        match statement {
            Statement::Select(stmt) => {
                if let Some(result) =
                    self.run_select_plan(stmt, compiled, Some(ctx), ctx.params(), in_txn)
                {
                    return Some(result);
                }
                // The aggregate plan reads committed state: a view the context
                // holds is read through by the regular path
                if in_txn || ctx.statement_view().is_some() {
                    return None;
                }
                self.run_planned_aggregate(stmt, compiled, ctx)
            }
            Statement::Update(stmt) => {
                let stmt = DmlStatement::Update(stmt);
                self.run_dml_plan(stmt, compiled, Some(ctx), ctx.params(), in_txn)
            }
            Statement::Delete(stmt) => {
                let stmt = DmlStatement::Delete(stmt);
                self.run_dml_plan(stmt, compiled, Some(ctx), ctx.params(), in_txn)
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

    /// The stamp a semantic cache read of `table_name` happens under now, or
    /// None while a commit is publishing to it, its table is being replaced,
    /// or `view` misses a commit it published: a stamp names the newest state.
    /// `read` takes what it needs of the table's store in the same state
    fn semantic_cache_stamp<R>(
        &self,
        table_name: &str,
        view: Option<&crate::storage::mvcc::registry::StatementView>,
        read: impl FnOnce(&crate::storage::mvcc::version_store::VersionStore) -> R,
    ) -> Option<(semantic_cache::CacheStamp, R)> {
        let schema = self.engine.schema_epoch();
        let store = self.engine.get_version_store(table_name).ok()?;
        let publish = store.publish_epoch_if_quiet()?;
        if view.is_some_and(|view| !store.view_sees_newest(view)) {
            return None;
        }
        let read = read(&store);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::cache_stamp_publish_read();
        // DDL bumps the schema epoch after swapping the store, so an unchanged
        // epoch means the publish epoch belongs to this table's store
        (self.engine.schema_epoch() == schema)
            .then_some((semantic_cache::CacheStamp { publish, schema }, read))
    }

    /// The stamp a cached subquery result over `tables` is read under now,
    /// or None while a commit is publishing to one of them, one of them is
    /// not a table of this engine, or `view` misses a commit one published
    pub(crate) fn subquery_cache_stamp<T: AsRef<str>>(
        &self,
        tables: &[T],
        view: Option<&crate::storage::mvcc::registry::StatementView>,
    ) -> Option<context::SubqueryStamp> {
        let schema = self.engine.schema_epoch();
        let mut publish = smallvec::SmallVec::with_capacity(tables.len());
        for table in tables {
            let store = self.engine.get_version_store(table.as_ref()).ok()?;
            publish.push(store.publish_epoch_if_quiet()?);
            if view.is_some_and(|view| !store.view_sees_newest(view)) {
                return None;
            }
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::cache_stamp_publish_read();
        // An unchanged schema epoch means each publish epoch is its table's store's
        (self.engine.schema_epoch() == schema).then(|| context::SubqueryStamp {
            engine: self.engine.instance(),
            schema,
            publish,
        })
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
        for slots in &program.params {
            check_param_count(slots, ctx.params().len())?;
        }
        self.run_statements(program, |_| ctx)
    }

    /// Runs every statement of `program`, the i-th in `ctx_of(i)`, under a
    /// gate the caller holds; the parameters are checked
    fn run_statements<'c>(
        &self,
        program: &Program,
        ctx_of: impl Fn(usize) -> &'c ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut last_result: Box<dyn QueryResult> = Box::new(ExecResult::empty());
        for (i, statement) in program.statements.iter().enumerate() {
            last_result = self.run_statement(statement, ctx_of(i))?;
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::program_statement_done();
        }
        Ok(last_result)
    }

    /// Execute a single statement; its parameters are checked against the
    /// count the parser wrote on it
    pub fn execute_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let slots = ParamSlots {
            count: statement.param_count(),
            names: None,
        };
        check_param_count(&slots, ctx.params().len())?;
        let _gate = self.enter(changes_session(statement));
        self.run_statement(statement, ctx)
    }

    /// Runs one statement under a gate the caller holds
    pub(crate) fn run_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_statement_inner(
            statement,
            std::borrow::Cow::Borrowed(ctx),
            self.active_txn_id(),
            None,
            None,
            None,
        )
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
        ctx: std::borrow::Cow<'_, ExecutionContext>,
        active_txn_id: Option<i64>,
        plan_classification: Option<
            &std::sync::OnceLock<Arc<query_classification::QueryClassification>>,
        >,
        compiled: Option<&NestedPlan>,
        counts: Option<&query_cache::CountsSlot>,
    ) -> Result<Box<dyn QueryResult>> {
        // A SELECT clears them where its top level starts
        if !matches!(statement, Statement::Select(_)) {
            context::clear_statement_caches();
        }
        // If there's an active transaction, inject the transaction ID into the context
        // This enables CURRENT_TRANSACTION_ID() function to return the correct value
        // and the statement's view goes into the same context: one the caller
        // owns takes both in place, a borrowed one is copied once
        let select = matches!(statement, Statement::Select(_));
        // A timed SELECT's deadline counts from before its first view
        let started =
            (select && ctx.timeout_ms() > 0).then(crate::common::time_compat::Instant::now);
        // A whole answer the semantic cache holds needs no view
        let mut semantic_miss = None;
        if let (Statement::Select(stmt), None, Some(slot)) =
            (statement, active_txn_id, plan_classification)
        {
            let classification =
                slot.get_or_init(|| query_classification::get_classification(stmt));
            match self.semantic_cache_before_view(stmt, &ctx, classification)? {
                query::EarlySemantic::Hit(result) => return Ok(result),
                query::EarlySemantic::Miss(miss) => semantic_miss = Some(miss),
                query::EarlySemantic::NotTried => {}
            }
        }
        let view = self.statement_view(reads_through_a_view(statement), &ctx, active_txn_id);
        let took_view = view.is_some();
        let ctx = match (active_txn_id, view) {
            (None, None) => ctx,
            (txn_id, view) => {
                let mut owned = ctx.into_owned();
                if let Some(txn_id) = txn_id {
                    owned = owned.into_transaction_id(txn_id as u64);
                }
                // A SELECT writes no table; one given neither reads through
                // a transaction of its own anyway
                if select {
                    owned = owned.with_table_use(TableUse::Read);
                }
                if let Some(view) = view {
                    owned = owned.into_statement_view(view);
                    if let Some(miss) = semantic_miss {
                        owned = owned.into_semantic_miss(miss);
                    }
                }
                std::borrow::Cow::Owned(owned)
            }
        };
        if select {
            let started = started.filter(|_| took_view);
            return self.select_until_held(
                statement,
                ctx,
                took_view,
                started,
                plan_classification,
                compiled,
                counts,
            );
        }
        self.dispatch_statement(statement, &ctx, plan_classification, compiled, None)
    }

    /// A SELECT, run again with a fresh view when a truncate or drop
    /// overtakes the one it took: at most three runs in all, inside the
    /// deadline of the first. A view the caller gave, or one a function
    /// registered after the built-ins may not survive running twice, ends
    /// it with the error instead
    #[allow(clippy::too_many_arguments)]
    fn select_until_held(
        &self,
        statement: &Statement,
        mut ctx: std::borrow::Cow<'_, ExecutionContext>,
        took_view: bool,
        started: Option<crate::common::time_compat::Instant>,
        plan_classification: Option<
            &std::sync::OnceLock<Arc<query_classification::QueryClassification>>,
        >,
        compiled: Option<&NestedPlan>,
        counts: Option<&query_cache::CountsSlot>,
    ) -> Result<Box<dyn QueryResult>> {
        let mut _deadline = None;
        for attempt in 1..=STATEMENT_RUNS {
            let result =
                self.dispatch_statement(statement, &ctx, plan_classification, compiled, counts);
            // A result read whole through an overtaken view may hold what went
            let overtaken = match &result {
                Err(Error::ViewOvertaken) => true,
                Ok(_) => ctx.view_held().is_err(),
                Err(_) => false,
            };
            if !overtaken {
                return result;
            }
            drop(result);
            if !took_view
                || attempt == STATEMENT_RUNS
                || self.function_registry.has_registered_functions()
            {
                return Err(Error::ViewOvertaken);
            }
            if let (Some(started), None) = (started, &_deadline) {
                let spent = started.elapsed().as_millis() as u64;
                let Some(left) = ctx.timeout_ms().checked_sub(spent).filter(|&ms| ms > 0) else {
                    return Err(Error::QueryCancelled);
                };
                _deadline = TimeoutGuard::for_ms(&ctx, left);
            }
            self.wait_out_destruction(&ctx)?;
            let statement_ctx = ctx.to_mut();
            drop(statement_ctx.take_statement_view());
            statement_ctx.set_statement_view(self.capture_view());
        }
        Err(Error::ViewOvertaken)
    }

    /// Runs `statement` in `ctx`, which holds its transaction and its view
    fn dispatch_statement(
        &self,
        statement: &Statement,
        ctx: &ExecutionContext,
        plan_classification: Option<
            &std::sync::OnceLock<Arc<query_classification::QueryClassification>>,
        >,
        compiled: Option<&NestedPlan>,
        counts: Option<&query_cache::CountsSlot>,
    ) -> Result<Box<dyn QueryResult>> {
        match statement {
            // DDL statements
            Statement::CreateTable(stmt) => self.execute_create_table(stmt, ctx, compiled),
            Statement::DropTable(stmt) => self.execute_drop_table(stmt, ctx),
            Statement::CreateIndex(stmt) => self.execute_create_index(stmt, ctx),
            Statement::DropIndex(stmt) => self.execute_drop_index(stmt, ctx),
            Statement::AlterTable(stmt) => self.execute_alter_table(stmt, ctx),
            Statement::CreateView(stmt) => self.execute_create_view(stmt, ctx),
            Statement::DropView(stmt) => self.execute_drop_view(stmt, ctx),

            // DML statements
            Statement::Insert(stmt) => self.execute_insert(stmt, ctx, compiled),
            Statement::Update(stmt) => self.execute_update(stmt, ctx, compiled),
            Statement::Delete(stmt) => self.execute_delete(stmt, ctx, compiled),
            Statement::Truncate(stmt) => self.execute_truncate(stmt, ctx),

            Statement::Select(stmt) => {
                // A statement run as a parsed tree is planned by its structure;
                // a cached plan's statement was planned before it got here
                let tree_plan = plan_classification
                    .is_none()
                    .then(|| self.query_cache.statement_plan(stmt))
                    .flatten();
                if let Some(plan) = &tree_plan {
                    let in_txn = self.active_transaction.is_open();
                    let params = ctx.params();
                    let planned =
                        self.run_select_plan(stmt, plan.slot(), Some(ctx), params, in_txn);
                    if let Some(result) = planned {
                        return result;
                    }
                }
                let compiled = compiled.or(tree_plan.as_deref());
                self.execute_select_with_plan(stmt, ctx, plan_classification, compiled, counts)
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
            Statement::Explain(stmt) => self.execute_explain(stmt, ctx, compiled),
            Statement::Analyze(stmt) => self.execute_analyze(stmt, ctx),
            Statement::Vacuum(stmt) => self.execute_vacuum(stmt, ctx),
            Statement::Copy(stmt) => self.execute_copy(stmt, ctx),
        }
    }

    /// The view a read committed statement's reads see, taken before its
    /// first read; None when the statement already has one, reads nothing
    /// through one, or runs in a snapshot, which reads at its own point. A
    /// statement snapshot the context carries gives its view, or is such a
    /// point unless it reads each commit as it lands
    pub(crate) fn statement_view(
        &self,
        reads: bool,
        ctx: &ExecutionContext,
        active_txn_id: Option<i64>,
    ) -> Option<crate::storage::mvcc::registry::StatementView> {
        if !reads || ctx.statement_view().is_some() {
            return None;
        }
        let at_its_point = ctx
            .statement_snapshot()
            .is_some_and(|snapshot| !snapshot.reads_latest());
        if at_its_point {
            return None;
        }
        let level = match active_txn_id {
            Some(txn_id) => self.engine.registry_ref().get_isolation_level(txn_id),
            None => self.isolation_level(),
        };
        (level == crate::core::IsolationLevel::ReadCommitted).then(|| self.capture_view())
    }

    /// A view of the commits complete now, in the memory of one this executor
    /// took before when no handle holds that any more
    pub(crate) fn capture_view(&self) -> crate::storage::mvcc::registry::StatementView {
        let spare = {
            let mut spares = self.view_spares.lock();
            let free = spares
                .iter_mut()
                .position(|state| Arc::get_mut(state).is_some());
            free.map(|at| spares.swap_remove(at))
        };
        let view = self.engine.registry_ref().capture_view_reusing(spare);
        {
            let mut spares = self.view_spares.lock();
            if spares.len() < VIEW_SPARES {
                spares.push(view.spare());
            }
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::statement_view_taken();
        view
    }

    /// A view of the commits complete now, for an autocommit read under read
    /// committed; None under snapshot isolation, which reads at its own point
    pub(crate) fn autocommit_view(&self) -> Option<crate::storage::mvcc::registry::StatementView> {
        (self.isolation_level() == crate::core::IsolationLevel::ReadCommitted)
            .then(|| self.capture_view())
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

    /// Begin a new transaction at this session's isolation level
    pub fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.engine
            .begin_transaction_with_level(self.isolation_level())
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
        self.plan_or_program(sql)?
            .map_err(|_| Error::parse("Prepared statements must contain exactly one statement"))
    }

    /// The cached plan of single-statement `sql`, made and cached on first
    /// use, or the parsed program when `sql` holds any other count
    pub(crate) fn plan_or_program(
        &self,
        sql: &str,
    ) -> Result<std::result::Result<CachedPlanRef, Program>> {
        if let Some(plan) = self.query_cache.get(sql) {
            return Ok(Ok(plan));
        }
        let mut parser = Parser::new(sql);
        let mut program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;
        if program.statements.len() != 1 {
            return Ok(Err(program));
        }
        let (Some(stmt), Some(params)) = (program.statements.pop(), program.params.pop()) else {
            return Ok(Err(program));
        };
        Ok(Ok(self.query_cache.put(sql, Arc::new(stmt), params)))
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
        self.execute_plan_in(plan, std::borrow::Cow::Borrowed(ctx))
    }

    /// [`Self::execute_with_cached_plan`] for a context the caller gives up:
    /// a statement's view goes into it without a copy
    pub(crate) fn execute_with_cached_plan_owned(
        &self,
        plan: &CachedPlanRef,
        ctx: ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_plan_in(plan, std::borrow::Cow::Owned(ctx))
    }

    fn execute_plan_in(
        &self,
        plan: &CachedPlanRef,
        ctx: std::borrow::Cow<'_, ExecutionContext>,
    ) -> Result<Box<dyn QueryResult>> {
        check_param_count(&plan.params, ctx.params().len())?;
        let gate = self.enter(changes_session(&plan.statement));
        // The fast paths read tables by name: CTEs the context brings are
        // read by the plans, which hold the CTE names they bound
        let held = !ctx.brings_ctes();
        if let (Statement::Insert(stmt), true) = (plan.statement.as_ref(), held) {
            return self.execute_insert_with_compiled_cache(stmt, &ctx, &plan.compiled);
        }
        let (active_txn_id, ctx) = self.gated_txn_ctx(&gate, ctx);
        if held {
            if let Some(result) = self.compiled_fast_paths_until_held(
                plan.statement.as_ref(),
                &ctx,
                plan.compiled.slot(),
                active_txn_id.is_some(),
            ) {
                return result;
            }
        }

        self.execute_statement_inner(
            &plan.statement,
            ctx,
            active_txn_id,
            Some(&plan.classification),
            Some(&*plan.compiled),
            plan.counts.as_deref(),
        )
    }
}

/// Refuses a statement taking `slots` that is given `provided` values: a
/// parameter without a value is an error, never NULL
#[inline]
pub(crate) fn check_param_count(slots: &ParamSlots, provided: usize) -> Result<()> {
    if provided < slots.count {
        return Err(too_few_params(slots, provided));
    }
    Ok(())
}

/// The error for the first slot of `slots` past the `provided` values
#[cold]
fn too_few_params(slots: &ParamSlots, provided: usize) -> Error {
    match slots.names.as_deref().and_then(|names| names.get(provided)) {
        Some(name) => missing_name(name),
        None => Error::invalid_argument(format!(
            "Query requires {} parameters but only {} provided",
            slots.count, provided
        )),
    }
}

#[cold]
fn missing_name(name: &str) -> Error {
    Error::invalid_argument(format!("named parameter :{} has no value", name))
}

/// The values of a statement taking `slots`, in slot order, each read by
/// its name from `value_of`, which may be asked for one name again by a
/// later statement
pub(crate) fn bind_named(
    slots: &ParamSlots,
    value_of: &mut impl FnMut(&str) -> Result<Option<Value>>,
) -> Result<ParamVec> {
    let Some(names) = &slots.names else {
        check_param_count(slots, 0)?;
        return Ok(ParamVec::new());
    };
    let mut values = ParamVec::with_capacity(names.len());
    for name in names.iter() {
        match value_of(name)? {
            Some(value) => values.push(value),
            None => return Err(missing_name(name)),
        }
    }
    Ok(values)
}

/// A context for each statement of `program`, its slots bound by name from
/// `value_of`; every statement is bound before any runs
pub(crate) fn bind_program(
    program: &Program,
    value_of: &mut impl FnMut(&str) -> Result<Option<Value>>,
) -> Result<Vec<ExecutionContext>> {
    program
        .params
        .iter()
        .map(|slots| bind_named(slots, value_of).map(ExecutionContext::with_params))
        .collect()
}

/// The `value_of` of a map of named values
pub(crate) fn value_in(
    params: &FxHashMap<String, Value>,
) -> impl FnMut(&str) -> Result<Option<Value>> + '_ {
    |name| Ok(params.get(name).cloned())
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

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn a_view_memory_is_taken_again_only_once_its_last_handle_unregistered() {
        type Memory = *const crate::storage::mvcc::registry::ViewState;
        let executor = std::rc::Rc::new(create_test_executor());
        let first = executor.capture_view();
        let first_memory: Memory = Arc::as_ptr(&first.spare());
        let seen = std::rc::Rc::new(std::cell::Cell::new(None::<Memory>));
        let (inner, slot) = (std::rc::Rc::clone(&executor), std::rc::Rc::clone(&seen));
        crate::test_failpoints::on_last_view_lease_released(move || {
            let second = inner.capture_view();
            slot.set(Some(Arc::as_ptr(&second.spare())));
        });
        drop(first);
        let second_memory = seen.get().expect("the last handle's release ran the hook");
        assert_ne!(
            second_memory, first_memory,
            "a capture took the memory of a view still registered"
        );
        let third = executor.capture_view();
        assert_eq!(
            Arc::as_ptr(&third.spare()),
            first_memory,
            "a capture took new memory though the first view's was free"
        );
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
