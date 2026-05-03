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
pub mod pattern_cache;
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
use crate::storage::traits::{
    QueryResult, ReadEngine, ReadTransaction, WriteTable, WriteTransaction,
};

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
    transaction: Box<dyn WriteTransaction>,
    /// Tables accessed within this transaction (cached for proper commit/rollback)
    tables: FxHashMap<String, Box<dyn WriteTable>>,
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
    /// Semantic cache for query results with subsumption detection.
    ///
    /// Held as `Arc<SemanticCache>` so every per-handle `Executor` for
    /// the same `EngineEntry` shares one cache. DML invalidation on any
    /// handle reaches every sibling reader. A per-handle cache would
    /// silently serve stale rows after a peer's commit (DML invalidates
    /// only the writer's local map; the reader sees the pre-update
    /// cached result).
    semantic_cache: Arc<SemanticCache>,
    /// Active transaction for explicit transaction control (BEGIN/COMMIT/ROLLBACK)
    active_transaction: Mutex<Option<ActiveTransaction>>,
    /// Query planner for cost-based optimization.
    ///
    /// Held as `Arc<QueryPlanner>` so every per-handle `Executor` for the
    /// same `EngineEntry` shares one planner. ANALYZE invalidates the
    /// planner's stats cache, and a per-handle planner would leave
    /// sibling handles on pre-ANALYZE estimates until the 5-minute TTL
    /// expires. Public Executor constructors (called by external Rust
    /// callers) build a fresh planner since they aren't tied to an
    /// engine entry.
    query_planner: Arc<QueryPlanner>,
    /// True for executors backing a `ReadOnlyDatabase`. DML helper paths
    /// (`get_table_for_dml`, `start_transaction_for_dml`) refuse to begin
    /// auto-commit transactions when this is set, providing defense-in-depth
    /// against any path that bypasses the parser-level write gate.
    read_only: bool,
}

impl Executor {
    /// Create a new executor with the given storage engine.
    ///
    /// `read_only` is inferred from the engine: if the engine was opened
    /// read-only (`?read_only=true` / `?mode=ro`), the executor refuses
    /// to begin writable auto-commit transactions in DML helper paths.
    /// Without this inference, an external caller could obtain an
    /// `Arc<MVCCEngine>` from a read-only `Database::engine()` and use
    /// `Executor::new` to construct a writable executor on top, reaching
    /// `MVCCEngine::begin_writable_transaction_internal` and writing
    /// through (with partial-mutation hazards on top of the contract
    /// violation, since the engine's WAL is opened read-only).
    pub fn new(engine: Arc<MVCCEngine>) -> Self {
        let read_only = engine.is_read_only_mode();
        let query_planner = Arc::new(QueryPlanner::new(Arc::clone(&engine)));
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache: Arc::new(SemanticCache::default()),
            active_transaction: Mutex::new(None),
            query_planner,
            read_only,
        }
    }

    /// Create a new executor configured for a read-only database.
    ///
    /// In addition to the parser-level write gate that
    /// [`crate::api::ReadOnlyDatabase`] applies before dispatching any
    /// SQL, this executor refuses to begin auto-commit transactions in
    /// DML helper paths (`get_table_for_dml`, `start_transaction_for_dml`).
    /// Together they prevent any write code path from running on a
    /// read-only handle, even if the parser gate is bypassed.
    pub fn new_read_only(engine: Arc<MVCCEngine>) -> Self {
        let query_planner = Arc::new(QueryPlanner::new(Arc::clone(&engine)));
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache: Arc::new(SemanticCache::default()),
            active_transaction: Mutex::new(None),
            query_planner,
            read_only: true,
        }
    }

    /// Create a per-handle executor that shares its semantic cache and
    /// query planner with every sibling executor for the same `EngineEntry`.
    ///
    /// Used by `Database::share_entry` / `ReadOnlyDatabase::from_entry`
    /// so DML invalidation on one handle's cache and ANALYZE on one
    /// handle's planner reach every sibling reader. Without sharing, a
    /// sibling would keep serving pre-update cached rows (semantic cache)
    /// or pre-ANALYZE plan estimates (planner) after a peer's commit.
    pub(crate) fn with_shared_semantic_cache(
        engine: Arc<MVCCEngine>,
        semantic_cache: Arc<SemanticCache>,
        query_planner: Arc<QueryPlanner>,
    ) -> Self {
        let read_only = engine.is_read_only_mode();
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache,
            active_transaction: Mutex::new(None),
            query_planner,
            read_only,
        }
    }

    /// Read-only variant of [`Self::with_shared_semantic_cache`].
    pub(crate) fn with_shared_semantic_cache_read_only(
        engine: Arc<MVCCEngine>,
        semantic_cache: Arc<SemanticCache>,
        query_planner: Arc<QueryPlanner>,
    ) -> Self {
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache,
            active_transaction: Mutex::new(None),
            query_planner,
            read_only: true,
        }
    }

    /// Create a new executor with a custom function registry.
    /// `read_only` is inferred from the engine; see [`Self::new`] for details.
    pub fn with_function_registry(
        engine: Arc<MVCCEngine>,
        function_registry: Arc<FunctionRegistry>,
    ) -> Self {
        let read_only = engine.is_read_only_mode();
        let query_planner = Arc::new(QueryPlanner::new(Arc::clone(&engine)));
        Self {
            engine,
            function_registry,
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::default(),
            semantic_cache: Arc::new(SemanticCache::default()),
            active_transaction: Mutex::new(None),
            query_planner,
            read_only,
        }
    }

    /// Create a new executor with a custom cache size.
    /// `read_only` is inferred from the engine; see [`Self::new`] for details.
    pub fn with_cache_size(engine: Arc<MVCCEngine>, cache_size: usize) -> Self {
        let read_only = engine.is_read_only_mode();
        let query_planner = Arc::new(QueryPlanner::new(Arc::clone(&engine)));
        Self {
            engine,
            function_registry: default_function_registry(),
            default_isolation_level: crate::core::IsolationLevel::ReadCommitted,
            query_cache: QueryCache::new(cache_size),
            semantic_cache: Arc::new(SemanticCache::default()),
            active_transaction: Mutex::new(None),
            query_planner,
            read_only,
        }
    }

    /// Returns true if this executor is configured for read-only access.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Check if there is an active explicit transaction
    pub fn has_active_transaction(&self) -> bool {
        self.active_transaction.lock().unwrap().is_some()
    }

    /// Get the query planner.
    ///
    /// Eagerly constructed at executor creation; shared across sibling
    /// executors via `Arc` so ANALYZE on one handle invalidates the
    /// stats cache observed by every reader.
    fn get_query_planner(&self) -> &QueryPlanner {
        &self.query_planner
    }

    /// Get or create a table within the active transaction
    /// Returns (table, should_auto_commit) where should_auto_commit is false if there's an active transaction
    #[allow(dead_code)]
    fn get_table_for_dml(&self, table_name: &str) -> Result<(Box<dyn WriteTable>, bool)> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(ref mut tx_state) = *active_tx {
            // There's an active transaction - use it
            let table_name_lower = table_name.to_lowercase();

            // Check if we already have this table cached
            if tx_state.tables.contains_key(&table_name_lower) {
                // We need to get the table from the transaction again since we can't clone Box<dyn WriteTable>
                let table = tx_state.transaction.get_table(table_name)?;
                return Ok((table, false));
            }

            // Get the table from the transaction and cache it
            let table = tx_state.transaction.get_table(table_name)?;

            // Store a reference indicator that this table is active in the transaction
            // Note: We can't cache the actual table as Box<dyn WriteTable> isn't Clone
            // But we can get a fresh handle each time - the key is using the same transaction
            tx_state.tables.insert(
                table_name_lower.clone(),
                tx_state.transaction.get_table(table_name)?,
            );

            Ok((table, false))
        } else {
            // No active transaction - create a new one with auto-commit.
            // Defense-in-depth: refuse to begin a writable auto-commit txn
            // on a read-only executor. The parser-level write gate on
            // ReadOnlyDatabase should prevent ever reaching here, but this
            // is the second line of defence at the executor surface.
            if self.read_only {
                return Err(crate::core::Error::read_only_violation_at(
                    "executor",
                    "DML auto-commit",
                ));
            }
            let tx = self.engine.begin_writable_transaction_internal()?;
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
    ) -> Result<(Option<Box<dyn WriteTransaction>>, Box<dyn WriteTable>, bool)> {
        let active_tx = self.active_transaction.lock().unwrap();

        if active_tx.is_some() {
            // There's an active transaction - we'll use the cached version
            drop(active_tx);
            let (table, auto_commit) = self.get_table_for_dml(table_name)?;
            Ok((None, table, auto_commit))
        } else {
            // No active transaction - create a new one with auto-commit.
            // Defense-in-depth (see get_table_for_dml).
            drop(active_tx);
            if self.read_only {
                return Err(crate::core::Error::read_only_violation_at(
                    "executor",
                    "DML auto-commit",
                ));
            }
            let tx = self.engine.begin_writable_transaction_internal()?;
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

        // On a read-only executor, the PK Update / PK Delete fast paths
        // would otherwise mutate storage without going through
        // `execute_cached` (which has its own read-only check). Reject
        // them here so the parser-level write gate is honoured even on
        // the fast path. The cached AST has already been parsed once.
        if self.read_only {
            if let Some(reason) = cached.statement.write_reason() {
                return Some(Err(crate::core::Error::read_only_violation_at(
                    "parser", reason,
                )));
            }
        }

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
            // On a read-only executor, refuse any cached statement that
            // mutates persistent state. The check uses the parsed-once
            // cached AST — no extra parse on the hot path.
            if self.read_only {
                if let Some(reason) = cached.statement.write_reason() {
                    return Err(crate::core::Error::read_only_violation_at("parser", reason));
                }
            }

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

            // Try compiled fast paths based on statement type
            match cached.statement.as_ref() {
                Statement::Select(stmt) => {
                    // Try PK lookup fast path first
                    if let Some(result) =
                        self.try_fast_pk_lookup_compiled(stmt, ctx, &cached.compiled)
                    {
                        return result;
                    }
                    // Try COUNT(DISTINCT col) fast path
                    if let Some(result) =
                        self.try_fast_count_distinct_compiled(stmt, &cached.compiled)
                    {
                        return result;
                    }
                    // Try COUNT(*) fast path
                    if let Some(result) = self.try_fast_count_star_compiled(stmt, &cached.compiled)
                    {
                        return result;
                    }
                }
                Statement::Update(stmt) => {
                    if let Some(result) =
                        self.try_fast_pk_update_compiled(stmt, ctx, &cached.compiled)
                    {
                        return result;
                    }
                }
                Statement::Delete(stmt) => {
                    if let Some(result) =
                        self.try_fast_pk_delete_compiled(stmt, ctx, &cached.compiled)
                    {
                        return result;
                    }
                }
                // INSERT: Use compiled cache to avoid recomputing schema metadata
                Statement::Insert(stmt) => {
                    return self.execute_insert_with_compiled_cache(stmt, ctx, &cached.compiled);
                }
                _ => {}
            }

            // Execute the cached statement (standard path)
            return self.execute_statement(&cached.statement, ctx);
        }

        // Parse the query
        let mut parser = Parser::new(sql);
        let mut program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;

        // On a read-only executor, refuse any statement in the program that
        // mutates persistent state. Single fresh parse — checks happen on
        // the parsed AST without re-parsing.
        if self.read_only {
            for stmt in &program.statements {
                if let Some(reason) = stmt.write_reason() {
                    return Err(crate::core::Error::read_only_violation_at("parser", reason));
                }
            }
        }

        // Cache single-statement queries and execute directly from cache
        if program.statements.len() == 1 {
            // Take ownership of the statement to avoid clone
            let stmt = program.statements.pop().unwrap();
            let (has_params, param_count) = count_parameters(&stmt);
            let stmt_arc = std::sync::Arc::new(stmt);
            let cached_plan = self
                .query_cache
                .put(sql, stmt_arc.clone(), has_params, param_count);

            // Try compiled fast paths based on statement type
            match stmt_arc.as_ref() {
                Statement::Select(select) => {
                    // Try PK lookup fast path first
                    if let Some(result) =
                        self.try_fast_pk_lookup_compiled(select, ctx, &cached_plan.compiled)
                    {
                        return result;
                    }
                    // Try COUNT(DISTINCT col) fast path
                    if let Some(result) =
                        self.try_fast_count_distinct_compiled(select, &cached_plan.compiled)
                    {
                        return result;
                    }
                    // Try COUNT(*) fast path
                    if let Some(result) =
                        self.try_fast_count_star_compiled(select, &cached_plan.compiled)
                    {
                        return result;
                    }
                }
                Statement::Update(update) => {
                    if let Some(result) =
                        self.try_fast_pk_update_compiled(update, ctx, &cached_plan.compiled)
                    {
                        return result;
                    }
                }
                Statement::Delete(delete) => {
                    if let Some(result) =
                        self.try_fast_pk_delete_compiled(delete, ctx, &cached_plan.compiled)
                    {
                        return result;
                    }
                }
                // INSERT: Use compiled cache to avoid recomputing schema metadata
                Statement::Insert(insert) => {
                    return self.execute_insert_with_compiled_cache(
                        insert,
                        ctx,
                        &cached_plan.compiled,
                    );
                }
                _ => {}
            }

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

    /// SWMR v2 Phase G: invalidate cached query plans for one table
    /// only. Used by `ReadOnlyDatabase::refresh` so a writer commit
    /// against table A doesn't blow away cached plans for unrelated
    /// tables B, C, … on the reader side. Same fallback semantics as
    /// `QueryCache::invalidate_table` (per-plan scan of its compiled
    /// lookup table name + query text heuristic).
    pub fn invalidate_query_cache_for_table(&self, table_name: &str) {
        self.query_cache.invalidate_table(table_name);
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
        // Read-only check at the AST entrypoint. SQL-string callers
        // (`execute`, `execute_with_params`) check earlier in the
        // parse/cache path; AST callers (`execute_program`,
        // `execute_statement`, FFI callers, etc.) bypass that, so the
        // gate must live here too.
        if self.read_only {
            if let Some(reason) = statement.write_reason() {
                return Err(crate::core::Error::read_only_violation_at("parser", reason));
            }
        }

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

            // DML statements
            Statement::Insert(stmt) => self.execute_insert(stmt, &ctx),
            Statement::Update(stmt) => self.execute_update(stmt, &ctx),
            Statement::Delete(stmt) => self.execute_delete(stmt, &ctx),
            Statement::Truncate(stmt) => self.execute_truncate(stmt, &ctx),

            // Query statements - try fast-path first for simple PK lookups
            Statement::Select(stmt) => {
                // Fast-path for simple PK lookups (bypasses full planner)
                if let Some(result) = self.try_fast_pk_lookup(stmt, &ctx) {
                    return result;
                }
                // Fall back to full query execution
                self.execute_select(stmt, &ctx)
            }

            // Transaction control
            Statement::Begin(stmt) => self.execute_begin(stmt, &ctx),
            Statement::Commit(stmt) => self.execute_commit_stmt(stmt, &ctx),
            Statement::Rollback(stmt) => self.execute_rollback_stmt(stmt, &ctx),
            Statement::Savepoint(stmt) => self.execute_savepoint(stmt, &ctx),
            Statement::ReleaseSavepoint(stmt) => self.execute_release_savepoint(stmt, &ctx),

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
            Statement::Vacuum(stmt) => self.execute_vacuum(stmt, &ctx),
            Statement::Copy(stmt) => self.execute_copy(stmt, &ctx),
        }
    }

    /// Install an external storage transaction as the active transaction.
    ///
    /// Used by the programmatic Transaction API to delegate SELECT queries
    /// to the full executor pipeline (aggregates, JOINs, window functions, etc.)
    /// while keeping the transaction's uncommitted changes visible.
    pub fn install_transaction(&self, tx: Box<dyn WriteTransaction>) {
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
    pub fn take_transaction(&self) -> Option<Box<dyn WriteTransaction>> {
        let mut active_tx = self.active_transaction.lock().unwrap();
        active_tx.take().map(|at| at.transaction)
    }

    /// Begin a new transaction
    ///
    /// Refuses on a read-only executor. Without this guard, a `Database`
    /// opened with `?read_only=true` could call `db.begin()?` and receive
    /// a writable `Box<dyn WriteTransaction>` that bypasses every other
    /// gate (parser, DML auto-commit). Read-intent internal paths
    /// (subquery, SHOW, planner stats, INL join, etc.) use
    /// `ReadEngine::begin_read_transaction` and never reach this method.
    /// Write-intent internal paths (DML, DDL, COPY, ANALYZE) use
    /// `MVCCEngine::begin_writable_transaction_internal` directly. The
    /// public Engine trait method (`<MVCCEngine as Engine>::begin_transaction`)
    /// gates on `is_read_only_mode`, so external callers going through
    /// `Database::engine().begin_transaction()` get the same refusal.
    pub fn begin_transaction(&self) -> Result<Box<dyn WriteTransaction>> {
        if self.read_only {
            return Err(crate::core::Error::read_only_violation_at(
                "executor", "BEGIN",
            ));
        }
        self.engine.begin_writable_transaction_internal()
    }

    /// Begin a new transaction with a specific isolation level
    ///
    /// Same read-only guard as [`Self::begin_transaction`]: a writable
    /// transaction is never handed out from a read-only executor.
    pub fn begin_transaction_with_isolation(
        &self,
        isolation: crate::core::IsolationLevel,
    ) -> Result<Box<dyn WriteTransaction>> {
        if self.read_only {
            return Err(crate::core::Error::read_only_violation_at(
                "executor", "BEGIN",
            ));
        }
        let mut tx = self.engine.begin_writable_transaction_internal()?;
        let _ = tx.set_isolation_level(isolation);
        Ok(tx)
    }

    /// Begin a new read-only transaction.
    ///
    /// Returns `Box<dyn ReadTransaction>`, which has no path to writable
    /// table handles, DDL, or any mutating engine operation. Callers
    /// holding the returned trait object cannot escalate to write
    /// access by construction.
    ///
    /// Used by future read-only execution paths
    /// (e.g. `ReadOnlyDatabase::begin`) and by external Rust callers who
    /// want compile-time read-only enforcement on a transaction obtained
    /// from a writable engine.
    pub fn begin_read_transaction(&self) -> Result<Box<dyn ReadTransaction>> {
        ReadEngine::begin_read_transaction(self.engine.as_ref())
    }

    /// Begin a new read-only transaction with a specific isolation level.
    ///
    /// See [`Self::begin_read_transaction`] for the contract.
    pub fn begin_read_transaction_with_isolation(
        &self,
        isolation: crate::core::IsolationLevel,
    ) -> Result<Box<dyn ReadTransaction>> {
        ReadEngine::begin_read_transaction_with_level(self.engine.as_ref(), isolation)
    }

    /// Get or create a cached plan for a SQL statement.
    ///
    /// Parses the SQL and caches the plan if not already cached.
    /// Returns a lightweight CachedPlanRef that can be stored and reused
    /// for repeated execution without re-parsing or cache lookup overhead.
    pub fn get_or_create_plan(&self, sql: &str) -> Result<CachedPlanRef> {
        if let Some(plan) = self.query_cache.get(sql) {
            // Even on a cache hit, refuse to hand out a write-intent plan
            // from a read-only executor. The plan was cached by an earlier
            // (writable) caller; the executor wrapping that plan now is
            // read-only and would refuse at execute time anyway. Failing
            // here gives the caller the same diagnostic earlier and
            // matches the behaviour of `prepare()`.
            if self.read_only {
                if let Some(reason) = plan.statement.write_reason() {
                    return Err(crate::core::Error::read_only_violation_at("parser", reason));
                }
            }
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
        // Read-only refusal at parse time. Without this, write SQL would
        // parse and cache successfully against a read-only executor,
        // and the refusal would only fire later at `execute_plan` /
        // `query_plan` — confusing to debug.
        if self.read_only {
            if let Some(reason) = stmt.write_reason() {
                return Err(crate::core::Error::read_only_violation_at("parser", reason));
            }
        }
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
        // Read-only check on the prepared-statement path. Like
        // `try_fast_path_with_params`, this entrypoint bypasses
        // `execute_cached`, so the read-only enforcement must live here too.
        if self.read_only {
            if let Some(reason) = plan.statement.write_reason() {
                return Err(crate::core::Error::read_only_violation_at("parser", reason));
            }
        }

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

        // Try compiled fast paths based on statement type
        match plan.statement.as_ref() {
            Statement::Select(stmt) => {
                if let Some(result) = self.try_fast_pk_lookup_compiled(stmt, ctx, &plan.compiled) {
                    return result;
                }
                if let Some(result) = self.try_fast_count_distinct_compiled(stmt, &plan.compiled) {
                    return result;
                }
                if let Some(result) = self.try_fast_count_star_compiled(stmt, &plan.compiled) {
                    return result;
                }
            }
            Statement::Update(stmt) => {
                if let Some(result) = self.try_fast_pk_update_compiled(stmt, ctx, &plan.compiled) {
                    return result;
                }
            }
            Statement::Delete(stmt) => {
                if let Some(result) = self.try_fast_pk_delete_compiled(stmt, ctx, &plan.compiled) {
                    return result;
                }
            }
            Statement::Insert(stmt) => {
                return self.execute_insert_with_compiled_cache(stmt, ctx, &plan.compiled);
            }
            _ => {}
        }

        self.execute_statement(&plan.statement, ctx)
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

        executor.execute("SELECT  1").unwrap();
        let size = executor.cache_stats().size;

        // Same query with different whitespace should hit cache
        executor.execute("SELECT 1").unwrap();
        assert_eq!(executor.cache_stats().size, size);
    }
}
