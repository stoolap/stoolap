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

//! Query Cache for parsed SQL statements
//!
//! This module provides a cache for previously parsed SQL queries,
//! storing the parse tree to avoid the overhead of parsing the same
//! query multiple times.
//!
//! # Example
//!
//! ```ignore
//! let cache = QueryCache::new(1000);
//!
//! // First query - will be parsed and cached
//! if let Some(plan) = cache.get("SELECT * FROM users") {
//!     // Use cached plan
//! } else {
//!     // Parse and cache
//!     let stmt = parse(sql);
//!     cache.put(sql, stmt, ParamSlots::default());
//! }
//!
//! // Second identical query - retrieved from cache
//! let plan = cache.get("SELECT * FROM users").unwrap();
//! ```

use crate::common::time_compat::Instant;
use std::sync::{Arc, RwLock};

use crate::api::params::ParamVec;
use crate::common::{CompactArc, SmartString};
use rustc_hash::FxHashMap;

use crate::parser::ast::{ParamSlots, SelectStatement, Statement, WithClause};
use crate::storage::mvcc::version_store::LayoutToken;

use super::expr_key::ExprKey;
use super::query_classification::QueryClassification;
use super::shared_literals::SharedPlan;
use smallvec::SmallVec;

/// When a plan holds: at the schema epoch it last held at, or at a later
/// one while each table and view it read last changed when it did; a plan
/// that does not know what it read holds at its epoch only
#[derive(Debug)]
pub struct PlanValidity {
    epoch: std::sync::atomic::AtomicU64,
    reads: Option<Arc<[(SmartString, u64)]>>,
}

impl Clone for PlanValidity {
    fn clone(&self) -> Self {
        Self {
            epoch: std::sync::atomic::AtomicU64::new(self.epoch()),
            reads: self.reads.clone(),
        }
    }
}

impl PlanValidity {
    /// A plan made at `epoch` that read `reads`, each name with the epoch
    /// of its last change, or that does not know what it read
    pub(crate) fn new(epoch: u64, reads: Option<Arc<[(SmartString, u64)]>>) -> Self {
        Self {
            epoch: std::sync::atomic::AtomicU64::new(epoch),
            reads,
        }
    }

    /// The schema epoch the plan last held at
    #[inline]
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the plan holds at `epoch`, `engine`'s current one
    #[inline]
    pub(crate) fn holds<E: crate::storage::traits::Engine + ?Sized>(
        &self,
        engine: &E,
        epoch: u64,
    ) -> bool {
        self.epoch() == epoch || self.holds_again(engine, epoch)
    }

    #[cold]
    fn holds_again<E: crate::storage::traits::Engine + ?Sized>(
        &self,
        engine: &E,
        epoch: u64,
    ) -> bool {
        let Some(reads) = &self.reads else {
            return false;
        };
        let held = reads
            .iter()
            .all(|(name, at)| engine.name_epoch(name) == *at);
        if held {
            self.epoch
                .fetch_max(epoch, std::sync::atomic::Ordering::Relaxed);
        }
        held
    }
}

/// How to extract the PK value for a compiled PK update or delete
#[derive(Debug, Clone)]
pub enum PkValueSource {
    /// Value comes from a positional parameter (0-indexed)
    Parameter(usize),
    /// Value is a literal integer
    Literal(i64),
}

/// Positions compiled under `token` run on `table` only while it still
/// holds its rows in that layout
pub(crate) fn check_compiled_token(
    table: &dyn crate::storage::traits::Table,
    token: LayoutToken,
) -> crate::core::Result<()> {
    if table.layout_token() != token {
        return Err(crate::core::Error::SchemaChanged {
            table: table.name().to_string(),
        });
    }
    Ok(())
}

/// A fast path whose table moved off its compiled layout before it wrote
/// anything hands the statement to the normal path, which resolves anew
pub(crate) fn or_normal_path(
    result: crate::core::Result<Box<dyn crate::storage::traits::QueryResult>>,
) -> Option<crate::core::Result<Box<dyn crate::storage::traits::QueryResult>>> {
    match result {
        Err(crate::core::Error::SchemaChanged { .. }) => None,
        result => Some(result),
    }
}

/// Pre-compiled update column assignment
#[derive(Debug, Clone)]
pub struct CompiledUpdateColumn {
    /// Column index in the schema
    pub column_idx: usize,
    /// Column data type for coercion
    pub column_type: crate::core::DataType,
    /// How to get the new value (literal or parameter)
    pub value_source: UpdateValueSource,
    /// Vector dimensions of the target column (0 for non-vector columns);
    /// updates must validate like the full path (VectorDimensionMismatch)
    pub column_vector_dims: u16,
}

/// How to get the update value
#[derive(Clone)]
pub enum UpdateValueSource {
    /// Value is a literal
    Literal(crate::core::Value),
    /// Value comes from a positional parameter (0-indexed)
    Parameter(usize),
    /// Value is a row expression (e.g. SET n = n + 1), compiled once per
    /// plan and executed per update against the current row
    Expression(crate::executor::expression::SharedProgram),
}

impl std::fmt::Debug for UpdateValueSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateValueSource::Literal(v) => f.debug_tuple("Literal").field(v).finish(),
            UpdateValueSource::Parameter(i) => f.debug_tuple("Parameter").field(i).finish(),
            UpdateValueSource::Expression(_) => f.write_str("Expression(<compiled>)"),
        }
    }
}

/// Pre-compiled state for INSERT statements
/// Caches schema-derived information to avoid recomputation on every INSERT execution
#[derive(Clone)]
pub struct CompiledInsert {
    /// Table name (already lowercased, SmartString for inline storage)
    pub table_name: SmartString,
    /// Column indices for INSERT (which schema columns to populate)
    pub column_indices: Arc<Vec<usize>>,
    /// Column types for the INSERT columns (for type coercion)
    pub column_types: Arc<Vec<crate::core::DataType>>,
    /// Vector dimensions for each INSERT column (0 for non-vector columns)
    pub column_vector_dims: Arc<Vec<u16>>,
    /// Column names for error messages (Arc for zero-copy sharing)
    pub column_names: Arc<Vec<SmartString>>,
    /// Per-row default programs: Some for non-literal defaults, which
    /// must execute per row (DEFAULT (RANDOM()) differs per row)
    pub default_programs: Arc<Vec<crate::executor::dml::DefaultSlot>>,
    /// Constant default values for all columns (null_unknown when the
    /// column has no default or a per-row program)
    pub default_row_template: Arc<Vec<crate::core::Value>>,
    /// CHECK constraints, pre-compiled once per plan. Re-parsing the
    /// check SQL per row costs a tokenizer plus parser run per inserted
    /// row per constraint.
    pub(crate) compiled_checks: Arc<Vec<crate::executor::dml::CompiledCheck>>,
    /// When the compiled state holds
    pub validity: PlanValidity,
    /// The table instance and layout the indices and templates describe
    pub token: LayoutToken,
    /// Whether the statement reads beyond its target, so its reads go
    /// through a statement view: a query source or a subquery
    pub reads_through_view: bool,
    /// Whether its VALUES or RETURNING subqueries run as prepared blocks
    pub reads_prepared: bool,
    /// Those blocks, once prepared for the current schema
    pub(crate) prepared: Option<Arc<super::prepared_select::PreparedDml>>,
}

impl CompiledInsert {
    /// What one execution runs from: the cached state without the names it
    /// read, which stay with the cached one, at the epoch it holds at
    pub(crate) fn copied(&self) -> Self {
        Self {
            table_name: self.table_name.clone(),
            column_indices: Arc::clone(&self.column_indices),
            column_types: Arc::clone(&self.column_types),
            column_vector_dims: Arc::clone(&self.column_vector_dims),
            column_names: Arc::clone(&self.column_names),
            default_programs: Arc::clone(&self.default_programs),
            default_row_template: Arc::clone(&self.default_row_template),
            compiled_checks: Arc::clone(&self.compiled_checks),
            validity: PlanValidity::new(self.validity.epoch(), None),
            token: self.token,
            reads_through_view: self.reads_through_view,
            reads_prepared: self.reads_prepared,
            prepared: self.prepared.clone(),
        }
    }
}

/// Compiled parts of an INSERT that handles conflicts (ON DUPLICATE KEY
/// UPDATE, DO NOTHING)
#[derive(Clone)]
pub struct CompiledConflictInsert {
    /// CHECK constraints of the inserted rows
    pub(crate) checks: Arc<Vec<super::dml::CompiledCheck>>,
    /// ON DUPLICATE KEY UPDATE clause
    pub(crate) upsert: Option<Arc<super::dml::CompiledUpsert>>,
    /// Table name (already lowercased)
    pub table_name: SmartString,
    /// When the compiled state holds
    pub validity: PlanValidity,
    /// Layout of the table handle these were compiled against
    pub token: LayoutToken,
}

impl std::fmt::Debug for CompiledConflictInsert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledConflictInsert")
            .field("table_name", &self.table_name)
            .field("validity", &self.validity)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for CompiledInsert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledInsert")
            .field("table_name", &self.table_name)
            .field("column_indices", &self.column_indices)
            .field("validity", &self.validity)
            .field("compiled_checks", &self.compiled_checks.len())
            .finish_non_exhaustive()
    }
}

/// The plan of a statement, and those of the statements it runs inside
/// itself, each held in the plan of the one that runs it and dropped with it
#[derive(Debug, Default)]
pub struct NestedPlan {
    plan: RwLock<CompiledExecution>,
    /// A SELECT's derived FROM's, then its later set operation branches',
    /// then its CTEs'; a DML statement's CTEs'; or the one statement an
    /// INSERT, a CREATE TABLE or an EXPLAIN ANALYZE runs. Made when one first runs
    nested: std::sync::OnceLock<Box<[NestedPlan]>>,
}

impl NestedPlan {
    /// The compiled state of this statement itself
    pub fn slot(&self) -> &RwLock<CompiledExecution> {
        &self.plan
    }

    /// The `count` plans nested in this one
    fn nested(&self, count: usize) -> &[NestedPlan] {
        self.nested
            .get_or_init(|| (0..count).map(|_| NestedPlan::default()).collect())
    }

    /// The plans nested in `stmt`'s, this being that plan
    fn of_select(&self, stmt: &SelectStatement) -> &[NestedPlan] {
        let ctes = stmt.with.as_ref().map_or(0, |with| with.ctes.len());
        self.nested(1 + stmt.set_operations.len() + ctes)
    }

    /// The plan of `stmt`'s derived FROM
    pub(crate) fn from(&self, stmt: &SelectStatement) -> Option<&NestedPlan> {
        self.of_select(stmt).first()
    }

    /// The plan of `stmt`'s set operation branch at `index`
    pub(crate) fn branch(&self, stmt: &SelectStatement, index: usize) -> Option<&NestedPlan> {
        self.branches(stmt).get(index)
    }

    /// The plans of `stmt`'s set operation branches past the first, in order
    pub(crate) fn branches(&self, stmt: &SelectStatement) -> &[NestedPlan] {
        let nested = self.of_select(stmt);
        nested.get(1..1 + stmt.set_operations.len()).unwrap_or(&[])
    }

    /// The plans of the bodies of `stmt`'s CTEs, in order
    pub(crate) fn ctes(&self, stmt: &SelectStatement) -> &[NestedPlan] {
        let nested = self.of_select(stmt);
        nested.get(1 + stmt.set_operations.len()..).unwrap_or(&[])
    }

    /// The plans of the bodies of `with`'s CTEs, `with` being a DML
    /// statement's and this its plan
    pub(crate) fn dml_ctes(&self, with: &WithClause) -> &[NestedPlan] {
        self.nested(with.ctes.len())
    }

    /// The plan of the one statement this statement runs: an INSERT's
    /// SELECT, a CREATE TABLE's AS SELECT or an EXPLAIN ANALYZE's statement
    pub(crate) fn source(&self) -> Option<&NestedPlan> {
        self.nested(1).first()
    }

    /// The plans nested in this one that a run has made; none before one
    pub(crate) fn made(&self) -> &[NestedPlan] {
        self.nested.get().map_or(&[], |nested| &nested[..])
    }
}

/// What a statement's slot holds: a fast path's compiled state, a planned
/// statement or a prepared plan
#[derive(Debug, Clone, Default)]
pub enum CompiledExecution {
    /// Not analyzed yet
    #[default]
    Unknown,
    /// Analyzed but doesn't qualify for any fast path, while the validity holds
    NotOptimizable(PlanValidity),
    /// An UPDATE or DELETE that reads the row its primary key names
    PlannedDml(Arc<super::planned_dml::PlannedDml>),
    /// Cached INSERT compilation (schema-derived info)
    Insert(CompiledInsert),
    /// Compiled INSERT with conflict handling
    ConflictInsert(CompiledConflictInsert),
    /// A statement without WHERE whose one row is one count the table keeps
    /// or an index's bound
    GlobalAggregate(Arc<super::planned_aggregate::PlannedAggregate>),
    /// The programs a join statement compiles (general SELECT path)
    PreparedJoin(Arc<super::prepared_join::PreparedJoin>),
    /// A statement of the prepared family (general SELECT path)
    PreparedSelect(Arc<super::prepared_select::PreparedSelect>),
    /// An UPDATE or DELETE whose subqueries run prepared
    PreparedDml(Arc<super::prepared_select::PreparedDml>),
    /// A grouped statement whose SELECT list or HAVING subqueries run prepared
    PreparedGrouped(Arc<super::prepared_select::PreparedGrouped>),
}

/// Default cache size (number of cached plans)
pub const DEFAULT_CACHE_SIZE: usize = 1000;

/// Lightweight reference to a cached plan for query execution.
/// Contains only what's needed to execute: the immutable statement and param info.
#[derive(Debug, Clone)]
pub struct CachedPlanRef {
    /// The parsed AST (cheap Arc clone)
    pub statement: Arc<Statement>,
    /// The parameter slots the statement takes
    pub params: ParamSlots,
    /// Shared reference to compiled execution state (lazily populated)
    pub compiled: Arc<NestedPlan>,
    /// Classification of the SELECT, resolved once per plan (lazily populated)
    pub classification: Arc<std::sync::OnceLock<Arc<QueryClassification>>>,
    /// The prepared subqueries of a SELECT's LIMIT and OFFSET; None
    /// without one
    pub counts: Option<Arc<CountsSlot>>,
    /// Whether the statement reads a table AS OF anywhere
    pub holds_as_of: bool,
}

impl CachedPlanRef {
    /// A plan of `statement`, nothing compiled or classified yet
    pub(crate) fn new(statement: Arc<Statement>, params: ParamSlots) -> Self {
        let counts = counts_slot(&statement);
        let holds_as_of = super::statement_holds_as_of(&statement);
        Self {
            statement,
            params,
            compiled: Arc::new(NestedPlan::default()),
            classification: Arc::new(std::sync::OnceLock::new()),
            counts,
            holds_as_of,
        }
    }
}

/// What a plan keeps for a SELECT whose LIMIT or OFFSET holds a subquery
#[derive(Debug)]
pub struct CountsSlot {
    /// The subqueries' prepared blocks
    pub(crate) compiled: NestedPlan,
    /// The identity of the LIMIT and of the OFFSET expression
    pub(crate) keys: [Option<super::expr_key::ExprKey>; 2],
}

/// The slot of `statement`, for a SELECT whose LIMIT or OFFSET holds a subquery
fn counts_slot(statement: &Statement) -> Option<Arc<CountsSlot>> {
    let Statement::Select(select) = statement else {
        return None;
    };
    let counts = [select.limit.as_deref(), select.offset.as_deref()];
    counts
        .into_iter()
        .flatten()
        .any(super::Executor::has_subqueries)
        .then(|| {
            Arc::new(CountsSlot {
                compiled: NestedPlan::default(),
                keys: counts.map(|count| count.map(super::expr_key::ExprKey::of)),
            })
        })
}

/// Represents a parsed and prepared statement stored in the cache
#[derive(Debug, Clone)]
pub struct CachedQueryPlan {
    /// The parsed AST (wrapped in Arc for cheap cloning - statements are immutable)
    pub statement: Arc<Statement>,
    /// Last time this plan was used (monotonic)
    pub last_used: Instant,
    /// Number of times this plan has been used
    pub usage_count: u64,
    /// The parameter slots the statement takes
    pub params: ParamSlots,
    /// Compiled execution state (lazily populated on first execution)
    pub compiled: Arc<NestedPlan>,
    /// Classification of the SELECT, resolved once per plan (lazily populated)
    pub classification: Arc<std::sync::OnceLock<Arc<QueryClassification>>>,
    /// The prepared subqueries of a SELECT's LIMIT and OFFSET; None
    /// without one
    pub counts: Option<Arc<CountsSlot>>,
    /// Whether the statement reads a table AS OF anywhere
    pub holds_as_of: bool,
}

/// Query cache for parsed SQL statements
///
/// Provides thread-safe caching of parsed SQL queries to avoid
/// the overhead of parsing the same query multiple times.
pub struct QueryCache {
    /// Cached plans indexed by normalized query text (FxHash for fast string hashing)
    plans: RwLock<FxHashMap<SmartString, TextPlan>>,
    /// Maximum number of cached plans
    max_size: usize,
    /// Factor to determine how many plans to prune when cache is full (0.0-1.0)
    prune_factor: f64,
    /// The plan of each SELECT run as a parsed tree, by its structure
    statements: parking_lot::Mutex<StatementPlans>,
    /// The plans texts differing only in their shared literals run, by the
    /// texts' skeleton
    shared: parking_lot::Mutex<lru::LruCache<String, SmallVec<[Arc<SharedPlan>; 1]>>>,
}

/// The values a text's shared literals fill a shared plan's hidden slots with
pub(crate) type SharedValues = CompactArc<ParamVec>;

/// What the cache keeps for a text
enum TextPlan {
    Own(CachedQueryPlan),
    /// A plan shared by its skeleton, with the values of this text
    Shared(Arc<SharedPlan>, SharedValues, Instant),
}

impl TextPlan {
    /// When it was kept and how often it was used, oldest and least used
    /// pruned first
    fn age(&self) -> (Instant, u64) {
        match self {
            TextPlan::Own(plan) => (plan.last_used, plan.usage_count),
            TextPlan::Shared(_, _, kept) => (*kept, 1),
        }
    }
}

/// The plan a text runs
pub(crate) enum CachedText {
    Own(CachedPlanRef),
    Shared(Arc<SharedPlan>, SharedValues),
}

/// The plans one skeleton keeps, each for other values of its fixed literals
const SHARED_PER_SKELETON: usize = 8;

/// The plans of SELECTs run as parsed trees: by the tree's address, with the
/// tree they were made for, and by the tree's structure
struct StatementPlans {
    by_address: lru::LruCache<usize, (SelectStatement, Arc<NestedPlan>)>,
    by_structure: lru::LruCache<ExprKey, Arc<NestedPlan>>,
}

impl QueryCache {
    /// Create a new query cache with the given maximum size
    pub fn new(max_size: usize) -> Self {
        let capacity = std::num::NonZeroUsize::new(max_size).unwrap_or(std::num::NonZeroUsize::MIN);
        Self {
            plans: RwLock::new(FxHashMap::default()),
            max_size,
            prune_factor: 0.2, // Prune 20% of entries when cache is full
            statements: parking_lot::Mutex::new(StatementPlans {
                by_address: lru::LruCache::new(capacity),
                by_structure: lru::LruCache::new(capacity),
            }),
            shared: parking_lot::Mutex::new(lru::LruCache::new(capacity)),
        }
    }

    /// The plan of the SELECT `stmt` run as a parsed tree: the one made for
    /// this tree when it is the one run here before, else the one of its
    /// structure; None when its structure has no key
    pub(crate) fn statement_plan(&self, stmt: &SelectStatement) -> Option<Arc<NestedPlan>> {
        let address = std::ptr::from_ref(stmt) as usize;
        if let Some((held, plan)) = self.statements.lock().by_address.get(&address) {
            if held == stmt {
                return Some(Arc::clone(plan));
            }
        }
        let key = ExprKey::of_statement(stmt)?;
        let mut plans = self.statements.lock();
        let plan = Arc::clone(
            plans
                .by_structure
                .get_or_insert(key, || Arc::new(NestedPlan::default())),
        );
        plans
            .by_address
            .put(address, (stmt.clone(), Arc::clone(&plan)));
        Some(plan)
    }

    /// Create a new query cache with default size
    pub fn default_sized() -> Self {
        Self::new(DEFAULT_CACHE_SIZE)
    }

    /// Get a cached plan for a query if available
    ///
    /// Returns a cheap Arc clone of the cached statement and metadata.
    /// The Statement is immutable and shared via Arc.
    ///
    /// OPTIMIZATION: Only uses read lock for cache hits (no write lock for stats).
    /// Stats are only updated on cache misses via put().
    pub fn get(&self, query: &str) -> Option<CachedPlanRef> {
        match self.lookup(query)? {
            CachedText::Own(plan) => Some(plan),
            CachedText::Shared(..) => None,
        }
    }

    /// The plan the text `query` ran before: its own, or a shared one with
    /// the values of its shared literals
    pub(crate) fn lookup(&self, query: &str) -> Option<CachedText> {
        // Only use read lock - skip stats update for performance
        let plans = self.plans.read().ok()?;
        let plan = match plans.get(normalize_query(query))? {
            TextPlan::Own(plan) => plan,
            TextPlan::Shared(plan, values, _) => {
                return Some(CachedText::Shared(Arc::clone(plan), values.clone()))
            }
        };

        // Only clone the Arc (cheap) and copy the small fields
        Some(CachedText::Own(CachedPlanRef {
            statement: plan.statement.clone(),
            params: plan.params.clone(),
            compiled: plan.compiled.clone(), // Share compiled state
            classification: plan.classification.clone(),
            counts: plan.counts.clone(),
            holds_as_of: plan.holds_as_of,
        }))
    }

    /// Add a plan to the cache
    ///
    /// Returns a lightweight reference to the cached plan (CachedPlanRef).
    /// This avoids cloning SmartStrings since callers only need the statement
    /// and compiled execution state.
    pub fn put(&self, query: &str, statement: Arc<Statement>, params: ParamSlots) -> CachedPlanRef {
        let normalized_key = SmartString::new(normalize_query(query));

        // Create the compiled state upfront - shared between stored plan and returned ref
        let plan = CachedPlanRef::new(statement, params);

        if let Ok(mut plans) = self.plans.write() {
            // Check if we need to prune the cache
            if plans.len() >= self.max_size {
                self.prune_cache(&mut plans);
            }

            plans.insert(
                normalized_key,
                TextPlan::Own(CachedQueryPlan {
                    statement: plan.statement.clone(),
                    last_used: Instant::now(),
                    usage_count: 1,
                    params: plan.params.clone(),
                    compiled: plan.compiled.clone(), // Arc clone - cheap
                    classification: plan.classification.clone(),
                    counts: plan.counts.clone(),
                    holds_as_of: plan.holds_as_of,
                }),
            );
        }

        // Return lightweight reference - only Arc clones, no SmartString clones
        plan
    }

    /// The first plan kept for the texts of `skeleton` that `bind` binds,
    /// with what it bound
    pub(crate) fn find_shared<T>(
        &self,
        skeleton: &str,
        mut bind: impl FnMut(&SharedPlan) -> Option<T>,
    ) -> Option<(Arc<SharedPlan>, T)> {
        let mut shared = self.shared.lock();
        let plans = shared.get(skeleton)?;
        plans
            .iter()
            .find_map(|plan| bind(plan).map(|bound| (Arc::clone(plan), bound)))
    }

    /// Keeps `plan` for the texts of `skeleton`, beside at most
    /// [`SHARED_PER_SKELETON`] others whose fixed literals differ
    pub(crate) fn put_shared(&self, skeleton: String, plan: SharedPlan) -> Arc<SharedPlan> {
        let plan = Arc::new(plan);
        let mut shared = self.shared.lock();
        let plans = shared.get_or_insert_mut(skeleton, SmallVec::new);
        if plans.len() == SHARED_PER_SKELETON {
            plans.remove(0);
        }
        plans.push(Arc::clone(&plan));
        plan
    }

    /// Keeps that the text `query` runs `plan` with `values`
    pub(crate) fn put_shared_text(
        &self,
        query: &str,
        plan: &Arc<SharedPlan>,
        values: &SharedValues,
    ) {
        let key = SmartString::new(normalize_query(query));
        if let Ok(mut plans) = self.plans.write() {
            if plans.len() >= self.max_size {
                self.prune_cache(&mut plans);
            }
            let text = TextPlan::Shared(Arc::clone(plan), values.clone(), Instant::now());
            plans.insert(key, text);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut plans) = self.plans.write() {
            plans.clear();
        }
        let mut statements = self.statements.lock();
        statements.by_address.clear();
        statements.by_structure.clear();
        self.shared.lock().clear();
    }

    /// Get the number of plans in the cache
    pub fn size(&self) -> usize {
        self.plans.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let plans = match self.plans.read() {
            Ok(p) => p,
            Err(_) => {
                return CacheStats {
                    size: 0,
                    max_size: self.max_size,
                    total_usage: 0,
                    avg_usage: 0.0,
                }
            }
        };

        let size = plans.len();
        let total_usage: u64 = plans.values().map(|p| p.age().1).sum();
        let avg_usage = if size > 0 {
            total_usage as f64 / size as f64
        } else {
            0.0
        };

        CacheStats {
            size,
            max_size: self.max_size,
            total_usage,
            avg_usage,
        }
    }

    /// Prune the least recently used entries when the cache is full
    fn prune_cache(&self, plans: &mut FxHashMap<SmartString, TextPlan>) {
        // Calculate how many entries to remove
        let num_to_remove = ((self.max_size as f64) * self.prune_factor).ceil() as usize;
        let num_to_remove = num_to_remove.max(1);

        if plans.len() <= num_to_remove {
            return;
        }

        // Build a list of references sorted by last used time and usage count
        // Use references to avoid cloning all keys
        let mut entries: Vec<(&SmartString, Instant, u64)> = plans
            .iter()
            .map(|(k, p)| {
                let (last_used, usage_count) = p.age();
                (k, last_used, usage_count)
            })
            .collect();

        // The oldest first, then the least used: only which ones, not their order
        entries.select_nth_unstable_by(num_to_remove - 1, |a, b| {
            a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2))
        });

        // Collect only the keys to remove (clone only what we need)
        let keys_to_remove: Vec<SmartString> = entries
            .into_iter()
            .take(num_to_remove)
            .map(|(k, _, _)| k.clone())
            .collect();

        // Remove the oldest/least used entries
        for key in keys_to_remove {
            plans.remove(&key);
        }
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::default_sized()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached plans
    pub size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Total usage count across all cached plans
    pub total_usage: u64,
    /// Average usage per cached plan
    pub avg_usage: f64,
}

/// The cache key of a query: its text without leading and trailing whitespace.
/// Whitespace inside is kept, since in a literal or a comment it changes the query.
#[inline]
fn normalize_query(query: &str) -> &str {
    query.trim_matches(|c: char| c.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Expression, GroupByClause, SelectStatement, StarExpression};
    use crate::parser::token::{Position, Token, TokenType};

    fn dummy_token() -> Token {
        Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1))
    }

    fn star_token() -> Token {
        Token::new(TokenType::Operator, "*", Position::new(0, 1, 1))
    }

    fn create_test_statement() -> Arc<Statement> {
        Arc::new(Statement::Select(SelectStatement {
            param_count: 0,
            token: dummy_token(),
            with: None,
            distinct: false,
            distinct_on: vec![],
            columns: vec![Expression::Star(StarExpression {
                token: star_token(),
            })],
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
            prepared: None,
        }))
    }

    #[test]
    fn test_cache_put_get() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        // Put in cache
        cache.put("SELECT * FROM users", stmt.clone(), ParamSlots::default());
        assert_eq!(cache.size(), 1);

        // Get from cache
        let plan = cache.get("SELECT * FROM users");
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.params, ParamSlots::default());
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryCache::new(100);

        let plan = cache.get("SELECT * FROM users");
        assert!(plan.is_none());
    }

    #[test]
    fn test_cache_usage_count() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users", stmt, ParamSlots::default());

        // Get multiple times - OPTIMIZATION: get() no longer updates stats
        // to avoid write lock contention
        for _ in 0..5 {
            cache.get("SELECT * FROM users");
        }

        // Usage count is only incremented on put(), not get() (for performance)
        let stats = cache.stats();
        assert_eq!(stats.total_usage, 1); // Only initial put = 1
    }

    #[test]
    fn test_cache_clear() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users", stmt, ParamSlots::default());
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_pruning() {
        let cache = QueryCache::new(5);
        let stmt = create_test_statement();

        // Fill the cache
        for i in 0..10 {
            let query = format!("SELECT * FROM table{}", i);
            cache.put(&query, stmt.clone(), ParamSlots::default());
        }

        // Cache should have pruned some entries
        assert!(cache.size() <= 5);
    }

    #[test]
    fn test_normalize_query() {
        assert_eq!(
            normalize_query(" \n SELECT * FROM users\t "),
            "SELECT * FROM users"
        );
        assert_eq!(normalize_query("SELECT 'a  b'"), "SELECT 'a  b'");
        assert_eq!(
            normalize_query("SELECT 1 -- note\n+ 1"),
            "SELECT 1 -- note\n+ 1"
        );
        assert_eq!(normalize_query(" \t\n"), "");
    }

    #[test]
    fn test_normalize_query_utf8() {
        assert_eq!(
            normalize_query("  SELECT * FROM t WHERE city = '東京' AND emoji = '🎉'\n"),
            "SELECT * FROM t WHERE city = '東京' AND emoji = '🎉'"
        );
    }

    #[test]
    fn test_normalized_cache_hit() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users", stmt, ParamSlots::default());

        assert!(cache.get("  SELECT * FROM users\n").is_some());
        assert!(
            cache.get("SELECT  * FROM users").is_none(),
            "whitespace inside the query is part of the key"
        );
    }

    #[test]
    fn test_parameterized_query() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        let params = ParamSlots {
            count: 1,
            names: None,
        };
        cache.put("SELECT * FROM users WHERE id = $1", stmt, params.clone());

        let plan = cache.get("SELECT * FROM users WHERE id = $1").unwrap();
        assert_eq!(plan.params, params);
    }

    #[test]
    fn test_cache_stats() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT 1", stmt.clone(), ParamSlots::default());
        cache.put("SELECT 2", stmt.clone(), ParamSlots::default());

        // Access first query more - get() no longer updates stats (OPTIMIZATION)
        for _ in 0..5 {
            cache.get("SELECT 1");
        }

        let stats = cache.stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.max_size, 100);
        assert_eq!(stats.total_usage, 2); // Only 2 puts (gets don't update stats)
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(QueryCache::new(1000));
        let stmt = create_test_statement();

        // Pre-populate
        cache.put("SELECT * FROM users", stmt.clone(), ParamSlots::default());

        let mut handles = vec![];

        // Spawn multiple reader threads
        for _ in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    cache.get("SELECT * FROM users");
                }
            }));
        }

        // Spawn writer threads
        for i in 0..5 {
            let cache = Arc::clone(&cache);
            let stmt = stmt.clone();
            handles.push(thread::spawn(move || {
                for j in 0..20 {
                    let query = format!("SELECT * FROM table{}_{}", i, j);
                    cache.put(&query, stmt.clone(), ParamSlots::default());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should still be functional
        assert!(cache.get("SELECT * FROM users").is_some());
    }
}
