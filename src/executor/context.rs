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

use crate::common::deadline::Deadline;
pub use crate::common::deadline::DeadlineSite;
use crate::common::time_compat::Instant;
use crate::storage::mvcc::commit_log::HistoricalCut;
use crate::storage::mvcc::registry::StatementView;
use crate::storage::traits::transaction::{StatementTable, TableUse};
use lru::LruCache;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, LazyLock};

use crate::api::params::ParamVec;
use crate::common::{CompactArc, StringMap};
use crate::core::predicate::PredicateSet;
use crate::core::{Result, Row, Value};

// Static defaults for ExecutionContext to avoid allocations for empty values.
// These are shared across all contexts and only require Arc refcount bump on clone.
static EMPTY_PARAMS: LazyLock<CompactArc<ParamVec>> =
    LazyLock::new(|| CompactArc::new(ParamVec::new()));
static EMPTY_DATABASE: LazyLock<Arc<Option<String>>> = LazyLock::new(|| Arc::new(None));
static EMPTY_SESSION_VARS: LazyLock<Arc<AHashMap<String, Value>>> =
    LazyLock::new(|| Arc::new(AHashMap::new()));

use smallvec::SmallVec;

/// What a cached subquery result was read under: its engine, the schema
/// epoch and the publish epoch of each table it reads, all taken before the
/// read. Any commit, DDL or other database changes one of them
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubqueryStamp {
    pub engine: u64,
    pub schema: u64,
    pub publish: SmallVec<[u64; 2]>,
}

use crate::parser::ast::{Expression, SelectStatement, WindowFrameBound};

/// The tables a cached subquery result is checked against
pub type CacheTables = SmallVec<[CompactArc<str>; 2]>;

/// The tables `stmt` reads, its nested subqueries included, when that is
/// all it reads: plain tables and joins, with no CTE, set operation, AS OF
/// or table function anywhere in it. A cached result of anything else
/// cannot be checked
pub fn subquery_cache_tables(stmt: &SelectStatement) -> Option<CacheTables> {
    let mut tables = SmallVec::new();
    statement_tables(stmt, &mut tables)?;
    Some(tables)
}

/// Adds the tables `stmt` reads to `tables`; None when it reads anything else
fn statement_tables(stmt: &SelectStatement, tables: &mut CacheTables) -> Option<()> {
    if stmt.with.is_some() || !stmt.set_operations.is_empty() {
        return None;
    }
    let grouping_sets = match &stmt.group_by.modifier {
        crate::parser::ast::GroupByModifier::GroupingSets(sets) => sets.as_slice(),
        _ => &[],
    };
    let clauses = stmt
        .columns
        .iter()
        .chain(stmt.distinct_on.iter())
        .chain(stmt.where_clause.as_deref())
        .chain(stmt.group_by.columns.iter())
        .chain(grouping_sets.iter().flatten())
        .chain(stmt.having.as_deref())
        .chain(stmt.order_by.iter().map(|o| &o.expression))
        .chain(stmt.limit.as_deref())
        .chain(stmt.offset.as_deref());
    for expr in clauses {
        expr_tables(expr, tables)?;
    }
    for window in &stmt.window_defs {
        exprs_tables(&window.partition_by, tables)?;
        for order in &window.order_by {
            expr_tables(&order.expression, tables)?;
        }
        if let Some(frame) = &window.frame {
            bound_tables(&frame.start, tables)?;
            if let Some(end) = &frame.end {
                bound_tables(end, tables)?;
            }
        }
    }
    match stmt.table_expr.as_deref() {
        Some(source) => source_tables(source, tables),
        None => Some(()),
    }
}

/// The tables of a FROM made of plain tables and joins; None for any other source
fn source_tables(source: &Expression, tables: &mut CacheTables) -> Option<()> {
    match source {
        Expression::TableSource(ts) if ts.as_of.is_none() => {
            let name = ts.name.value_lower.as_str();
            if !tables.iter().any(|kept| &**kept == name) {
                tables.push(CompactArc::from(name));
            }
            Some(())
        }
        Expression::JoinSource(js) => {
            if let Some(condition) = js.condition.as_deref() {
                expr_tables(condition, tables)?;
            }
            source_tables(&js.left, tables)?;
            source_tables(&js.right, tables)
        }
        _ => None,
    }
}

/// Adds the tables `expr`'s subqueries read to `tables`; None for one the
/// cache cannot check, or a table source where an expression stands. Every
/// kind is named, so a new one is decided here before it compiles
pub(crate) fn expr_tables(expr: &Expression, tables: &mut CacheTables) -> Option<()> {
    match expr {
        Expression::ScalarSubquery(s) => statement_tables(&s.subquery, tables),
        Expression::Exists(e) => statement_tables(&e.subquery, tables),
        Expression::AllAny(a) => {
            expr_tables(&a.left, tables)?;
            statement_tables(&a.subquery, tables)
        }
        Expression::TableSource(_)
        | Expression::JoinSource(_)
        | Expression::SubquerySource(_)
        | Expression::ValuesSource(_)
        | Expression::CteReference(_)
        | Expression::FunctionTableSource(_) => None,
        Expression::Prefix(p) => expr_tables(&p.right, tables),
        Expression::Infix(i) => {
            expr_tables(&i.left, tables)?;
            expr_tables(&i.right, tables)
        }
        Expression::List(l) => exprs_tables(&l.elements, tables),
        Expression::ExpressionList(l) => exprs_tables(&l.expressions, tables),
        Expression::Distinct(d) => expr_tables(&d.expr, tables),
        Expression::In(i) => {
            expr_tables(&i.left, tables)?;
            expr_tables(&i.right, tables)
        }
        Expression::InHashSet(h) => expr_tables(&h.column, tables),
        Expression::QuantifiedSet(q) => expr_tables(&q.left, tables),
        Expression::Between(b) => {
            expr_tables(&b.expr, tables)?;
            expr_tables(&b.lower, tables)?;
            expr_tables(&b.upper, tables)
        }
        Expression::Like(l) => {
            expr_tables(&l.left, tables)?;
            expr_tables(&l.pattern, tables)?;
            match l.escape.as_deref() {
                Some(escape) => expr_tables(escape, tables),
                None => Some(()),
            }
        }
        Expression::Case(c) => {
            if let Some(value) = c.value.as_deref() {
                expr_tables(value, tables)?;
            }
            for when in &c.when_clauses {
                expr_tables(&when.condition, tables)?;
                expr_tables(&when.then_result, tables)?;
            }
            match c.else_value.as_deref() {
                Some(other) => expr_tables(other, tables),
                None => Some(()),
            }
        }
        Expression::Cast(c) => expr_tables(&c.expr, tables),
        Expression::FunctionCall(f) => call_tables(f, tables),
        Expression::Aliased(a) => expr_tables(&a.expression, tables),
        Expression::Window(w) => {
            call_tables(&w.function, tables)?;
            exprs_tables(&w.partition_by, tables)?;
            for order in &w.order_by {
                expr_tables(&order.expression, tables)?;
            }
            if let Some(frame) = &w.frame {
                bound_tables(&frame.start, tables)?;
                if let Some(end) = &frame.end {
                    bound_tables(end, tables)?;
                }
            }
            Some(())
        }
        Expression::Identifier(_)
        | Expression::QualifiedIdentifier(_)
        | Expression::Column(_)
        | Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::IntervalLiteral(_)
        | Expression::Parameter(_)
        | Expression::Star(_)
        | Expression::QualifiedStar(_)
        | Expression::Default(_) => Some(()),
    }
}

/// [`expr_tables`] for each of `exprs`
fn exprs_tables(exprs: &[Expression], tables: &mut CacheTables) -> Option<()> {
    exprs.iter().try_for_each(|expr| expr_tables(expr, tables))
}

/// [`expr_tables`] for a call's arguments, FILTER and ORDER BY
fn call_tables(call: &crate::parser::ast::FunctionCall, tables: &mut CacheTables) -> Option<()> {
    exprs_tables(&call.arguments, tables)?;
    if let Some(filter) = call.filter.as_deref() {
        expr_tables(filter, tables)?;
    }
    call.order_by
        .iter()
        .try_for_each(|order| expr_tables(&order.expression, tables))
}

/// [`expr_tables`] for the offset of a window frame bound
fn bound_tables(bound: &WindowFrameBound, tables: &mut CacheTables) -> Option<()> {
    match bound {
        WindowFrameBound::Preceding(expr) | WindowFrameBound::Following(expr) => {
            expr_tables(expr, tables)
        }
        WindowFrameBound::CurrentRow
        | WindowFrameBound::UnboundedPreceding
        | WindowFrameBound::UnboundedFollowing => Some(()),
    }
}

use ahash::AHashMap;
use std::hash::{Hash, Hasher};

/// Answers a block cache keeps per thread, the least recently used dropped first
const BLOCK_ANSWER_CACHE_SIZE: usize = 256;

/// Whether two parameter values are one input: of one type, a NULL of one
/// type and a FLOAT of one bit pattern. `Value`'s equality is wider (INTEGER
/// 1 is FLOAT 1.0, NULLs of any type meet), which `TYPEOF` tells apart
fn same_parameter(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null(x), Value::Null(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        _ => std::mem::discriminant(a) == std::mem::discriminant(b) && a == b,
    }
}

/// The hash that agrees with `same_parameter`
fn hash_parameter<H: Hasher>(value: &Value, state: &mut H) {
    std::mem::discriminant(value).hash(state);
    match value {
        Value::Null(data_type) => data_type.hash(state),
        Value::Float(f) => f.to_bits().hash(state),
        _ => value.hash(state),
    }
}

/// What the candidates of one key of a prepared EXISTS block met: a row
/// that passed, or else the first error one of them raised
#[derive(Debug, Clone)]
pub enum ExistsOutcome {
    Passed,
    Failed(Box<crate::core::Error>),
}

/// A prepared EXISTS block's candidates, grouped by key, and whether a
/// key holds an error
#[derive(Debug, Default)]
pub struct ExistsKeys {
    pub outcomes: FxHashMap<SmallVec<[Value; 1]>, ExistsOutcome>,
    pub failed: bool,
    /// The row ids of the keys that passed, which a NOT EXISTS read by the
    /// primary key leaves out, collected once for every run
    pub left_out: std::sync::OnceLock<crate::common::I64Set>,
}

/// A value block's members as one test asks them
#[derive(Debug)]
pub enum MemberSet {
    /// The IN members of one key of a correlated block
    In(PredicateSet),
    /// An uncorrelated block's IN members, shared with the member tests a
    /// program is bound to
    SharedIn(CompactArc<PredicateSet>),
    Quantified(super::expression::QuantifiedMembers),
    /// A row block's members, which a row of their width is tested against
    Rows(super::expression::RowMembers),
}

/// A value block's members for a whole run, and the first error one of
/// its candidates raised
#[derive(Debug)]
pub struct KeptMembers {
    pub set: MemberSet,
    pub failed: Option<crate::core::Error>,
}

impl KeptMembers {
    /// `value` tested against the members: a member that settles the test
    /// (TRUE for IN and ANY, FALSE for ALL) wins; else the first error a
    /// candidate raised; else what the members say
    #[inline]
    pub(crate) fn answer(&self, value: &crate::core::Value) -> crate::core::Result<Option<bool>> {
        let (answer, decides) = match &self.set {
            MemberSet::In(set) => (set.contains(value), true),
            MemberSet::SharedIn(set) => (set.contains(value), true),
            MemberSet::Quantified(members) => (members.answer(value), !members.parts().1),
            MemberSet::Rows(_) => return Err(one_value_against_rows()),
        };
        self.settle(answer, decides)
    }

    /// `row` tested against a row block's members as [`Self::answer`] tests
    /// a value
    pub(crate) fn answer_row(&self, row: &[Value]) -> crate::core::Result<Option<bool>> {
        let MemberSet::Rows(members) = &self.set else {
            return Err(one_value_against_rows());
        };
        match (members.decides(), &self.failed) {
            (Some(decides), _) => self.settle(members.answer(row)?, decides),
            // A row compared with the one member has no member that decides
            (None, Some(e)) => Err(e.clone()),
            (None, None) => members.answer(row),
        }
    }

    /// `tested`, one value or a row, as [`Self::answer`] or
    /// [`Self::answer_row`] tests it
    #[inline]
    pub(crate) fn answer_values(&self, tested: &[Value]) -> crate::core::Result<Option<bool>> {
        match tested {
            [value] => self.answer(value),
            row => self.answer_row(row),
        }
    }

    #[inline]
    fn settle(&self, answer: Option<bool>, decides: bool) -> crate::core::Result<Option<bool>> {
        match (&self.failed, answer) {
            (_, Some(answer)) if answer == decides => Ok(Some(answer)),
            (Some(e), _) => Err(e.clone()),
            (None, answer) => Ok(answer),
        }
    }
}

#[cold]
fn one_value_against_rows() -> crate::core::Error {
    crate::core::Error::internal("a value and a block's members of other widths")
}

/// A correlated IN block's members grouped by key
pub type MemberKeys = FxHashMap<SmallVec<[Value; 1]>, KeptMembers>;

/// A value block's members as one answer keeps them, and the row ids a
/// NOT IN read by the primary key leaves out, collected once for every run
#[derive(Debug)]
pub struct AnsweredMembers {
    pub members: KeptMembers,
    pub left_out: std::sync::OnceLock<crate::common::I64Set>,
}

impl From<KeptMembers> for AnsweredMembers {
    fn from(members: KeptMembers) -> Self {
        Self {
            members,
            left_out: std::sync::OnceLock::new(),
        }
    }
}

impl std::ops::Deref for AnsweredMembers {
    type Target = KeptMembers;

    fn deref(&self) -> &KeptMembers {
        &self.members
    }
}

/// A prepared block's answer for a whole run, or for one call of its parent
#[derive(Debug, Clone)]
pub enum BlockAnswer {
    /// A scalar block's value
    Value(Value),
    /// An EXISTS block's value
    Once(bool),
    Keyed(CompactArc<ExistsKeys>),
    /// A value block's members
    Members(CompactArc<AnsweredMembers>),
    KeyedMembers(CompactArc<MemberKeys>),
    /// The candidates a keyed block's probes may still read over tables
    /// unchanged since its stamp, shared by every run that probes them
    ProbeBudget(Arc<AtomicUsize>),
}

/// What a prepared EXISTS answer is of: the block's identity, the scopes
/// around it its names bind in, and the statement's parameter values it
/// reads: those at `read`, or every one
pub(crate) struct BlockOrigin<'a> {
    pub block_key: super::expr_key::ExprKey,
    pub scopes: &'a Arc<str>,
    pub params: &'a [Value],
    pub read: Option<&'a [usize]>,
}

/// An `BlockOrigin` as its cache entry keeps it
struct KeptBlockOrigin {
    block_key: super::expr_key::ExprKey,
    scopes: Arc<str>,
    params: Vec<Value>,
}

impl BlockOrigin<'_> {
    /// The parameter values the answer depends on, in order
    fn read_params(&self) -> impl Iterator<Item = &Value> {
        let all = self.read.is_none().then_some(self.params.iter());
        let read = self.read.into_iter().flatten().map(|at| {
            self.params
                .get(*at)
                .unwrap_or(&crate::core::value::NULL_VALUE)
        });
        all.into_iter().flatten().chain(read)
    }

    /// The cache key, taken without allocation
    pub fn key(&self) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        self.block_key.hash(&mut hasher);
        self.scopes.hash(&mut hasher);
        self.read_params().count().hash(&mut hasher);
        for value in self.read_params() {
            hash_parameter(value, &mut hasher);
        }
        hasher.finish()
    }

    fn is(&self, kept: &KeptBlockOrigin) -> bool {
        self.block_key == kept.block_key
            && (Arc::ptr_eq(self.scopes, &kept.scopes) || **self.scopes == *kept.scopes)
            && kept.params.len() == self.read_params().count()
            && kept
                .params
                .iter()
                .zip(self.read_params())
                .all(|(kept, value)| same_parameter(kept, value))
    }
}

thread_local! {
    static BLOCK_ANSWER_CACHE: RefCell<
        LruCache<u64, (KeptBlockOrigin, SubqueryStamp, BlockAnswer)>,
    > = RefCell::new(LruCache::new(NonZeroUsize::new(BLOCK_ANSWER_CACHE_SIZE).unwrap()));
}

/// Clear the prepared EXISTS answer cache completely
pub fn clear_block_answer_cache() {
    BLOCK_ANSWER_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// The answer kept for `origin` under `stamp`; one read under another
/// stamp is dropped
pub(crate) fn get_cached_block_answer(
    origin: &BlockOrigin<'_>,
    key: u64,
    stamp: &SubqueryStamp,
) -> Option<BlockAnswer> {
    BLOCK_ANSWER_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        match c.get(&key) {
            Some((kept, kept_stamp, answer)) if origin.is(kept) => {
                if kept_stamp == stamp {
                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::subquery_cache_hit();
                    return Some(answer.clone());
                }
                c.pop(&key);
                None
            }
            _ => None,
        }
    })
}

/// Keep an answer under the stamp taken before its read
pub(crate) fn cache_block_answer(
    origin: &BlockOrigin<'_>,
    key: u64,
    stamp: SubqueryStamp,
    answer: BlockAnswer,
) {
    let kept = KeptBlockOrigin {
        block_key: origin.block_key,
        scopes: Arc::clone(origin.scopes),
        params: origin.read_params().cloned().collect(),
    };
    BLOCK_ANSWER_CACHE.with(|cache| cache.borrow_mut().put(key, (kept, stamp, answer)));
}

/// Clear ALL thread-local caches to release memory.
/// Call this when a database is dropped to prevent memory leaks.
pub fn clear_all_thread_local_caches() {
    clear_block_answer_cache();

    // Clear storage expression caches (regex patterns)
    crate::core::pattern::clear_regex_cache();

    // Clear RowVec and RowIdVec thread-local pools
    crate::core::row_vec::clear_row_vec_pool();
    crate::core::row_vec::clear_row_id_vec_pool();

    // Clear global transaction version map pools
    crate::storage::mvcc::clear_version_map_pools();

    // Clear global LRU caches
    super::query_classification::clear_classification_cache();
}

/// Execution context for SQL queries
///
/// The execution context carries state and configuration for query execution,
/// including parameters, transaction state, and the call's deadline.
///
/// Note: This struct uses Arc for immutable shared data to make cloning cheap
/// during correlated subquery processing where context is cloned per row.
/// A transaction shared by every read of one statement
#[derive(Clone)]
pub struct StatementSnapshot(Arc<SnapshotState>);

struct SnapshotState {
    reads: SnapshotReads,
    /// Whether its reads see each commit as it lands (READ COMMITTED with
    /// no statement view), so a result read through it is no older than a
    /// stamp taken before the read
    reads_latest: bool,
    /// What is written through its transaction: nothing for the executor's
    /// own, maybe anything for one a caller gave through `get_table`
    table_use: TableUse,
}

/// What a statement snapshot opens its tables through
enum SnapshotReads {
    /// A transaction, locked without allocating, and the statement view
    /// its tables are read through when the statement has one
    Transaction {
        transaction: parking_lot::Mutex<Box<dyn crate::storage::traits::Transaction>>,
        view: Option<StatementView>,
    },
    /// The view alone, for reads that write nothing: the engine opens the
    /// tables, and the registry counts the reads as a transaction
    View {
        view: StatementView,
        engine: Arc<crate::storage::mvcc::engine::MVCCEngine>,
        _admitted: crate::storage::mvcc::registry::ReaderAdmission,
    },
}

impl StatementSnapshot {
    /// The statement's reads through `transaction`, begun at `isolation`,
    /// and through `view` when the statement has one
    pub fn new(
        transaction: Box<dyn crate::storage::traits::Transaction>,
        isolation: crate::core::IsolationLevel,
        view: Option<StatementView>,
    ) -> Self {
        let reads_latest =
            isolation == crate::core::IsolationLevel::ReadCommitted && view.is_none();
        let transaction = parking_lot::Mutex::new(transaction);
        Self(Arc::new(SnapshotState {
            reads: SnapshotReads::Transaction { transaction, view },
            reads_latest,
            table_use: TableUse::Write,
        }))
    }

    /// The executor's own reads on `engine`, at `isolation`: nothing is
    /// written through them, so a table opens without a local store, and
    /// under a view without a transaction
    pub(crate) fn for_reads(
        engine: &Arc<crate::storage::mvcc::engine::MVCCEngine>,
        isolation: crate::core::IsolationLevel,
        view: Option<StatementView>,
    ) -> crate::core::Result<Self> {
        use crate::storage::traits::Engine;
        let reads_latest =
            isolation == crate::core::IsolationLevel::ReadCommitted && view.is_none();
        let reads = match view {
            Some(view) => SnapshotReads::View {
                view,
                engine: Arc::clone(engine),
                _admitted: engine.registry_ref().admit_reader().ok_or_else(|| {
                    crate::core::Error::internal(
                        "transaction registry is not accepting new transactions",
                    )
                })?,
            },
            None => SnapshotReads::Transaction {
                transaction: parking_lot::Mutex::new(
                    engine.begin_transaction_with_level(isolation)?,
                ),
                view: None,
            },
        };
        Ok(Self(Arc::new(SnapshotState {
            reads,
            reads_latest,
            table_use: TableUse::Read,
        })))
    }

    /// Whether its reads see each commit as it lands
    pub(crate) fn reads_latest(&self) -> bool {
        self.0.reads_latest
    }

    /// Whether nothing writes through its transaction, so a commit stamp
    /// covers every row its reads can see
    pub(crate) fn writes_nothing(&self) -> bool {
        self.0.table_use == TableUse::Read
    }

    /// The statement view its tables are read through
    pub(crate) fn view(&self) -> Option<&StatementView> {
        match &self.0.reads {
            SnapshotReads::Transaction { view, .. } => view.as_ref(),
            SnapshotReads::View { view, .. } => Some(view),
        }
    }

    /// The table `name`; read only when the snapshot reads through its view
    pub fn get_table(
        &self,
        name: &str,
    ) -> crate::core::Result<Box<dyn crate::storage::traits::Table>> {
        match &self.0.reads {
            SnapshotReads::Transaction { transaction, .. } => transaction.lock().get_table(name),
            SnapshotReads::View { view, engine, .. } => engine.read_table_in_view(name, view),
        }
    }

    /// The table read through `view`, None reading the newest committed
    /// state, or the snapshot's own view when it reads through one
    pub(crate) fn statement_table_at(
        &self,
        name: &str,
        view: Option<&StatementView>,
    ) -> crate::core::Result<Box<dyn crate::storage::traits::Table>> {
        match &self.0.reads {
            SnapshotReads::Transaction { transaction, .. } => {
                transaction
                    .lock()
                    .get_read_table_at(name, view, self.0.table_use)
            }
            SnapshotReads::View {
                view: own, engine, ..
            } => engine.read_table_in_view(name, view.unwrap_or(own)),
        }
    }

    /// Every row of `name` as it stood at `point`
    pub(crate) fn rows_at(
        &self,
        name: &str,
        point: &crate::storage::mvcc::commit_log::ReadPoint,
    ) -> crate::core::Result<crate::core::RowVec> {
        match &self.0.reads {
            SnapshotReads::Transaction { transaction, .. } => {
                transaction.lock().rows_at(name, point)
            }
            SnapshotReads::View { view, engine, .. } => {
                engine.read_table_in_view(name, view)?.rows_at(point, None)
            }
        }
    }
}

impl std::fmt::Debug for StatementSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StatementSnapshot")
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Query parameters ($1, $2, etc.) - wrapped in Arc for cheap cloning
    params: CompactArc<ParamVec>,
    /// Whether to use auto-commit for DML statements
    auto_commit: bool,
    /// Current database/schema name - wrapped in Arc for cheap cloning
    current_database: Arc<Option<String>>,
    /// Session variables (SET key = value) - wrapped in Arc for cheap cloning
    session_vars: Arc<AHashMap<String, Value>>,
    /// When the call stops, shared by every statement it runs
    deadline: Deadline,
    /// Current view nesting depth (for detecting infinite recursion)
    view_depth: usize,
    /// Query execution depth (0 = top-level query, >0 = subquery/nested)
    pub(crate) query_depth: usize,
    /// CTE data for subqueries to reference CTEs from outer query
    /// Maps CTE name (lowercase) to (columns, rows)
    cte_data: Option<Arc<CteDataMap>>,
    /// Current transaction ID for CURRENT_TRANSACTION_ID() function
    transaction_id: Option<u64>,
    /// One transaction for every fetch of a statement that has no explicit
    /// transaction, so its reads share a snapshot
    statement_snapshot: Option<StatementSnapshot>,
    /// The commits a read committed statement's reads see, taken once for
    /// the statement; None reads the newest committed state
    statement_view: Option<StatementView>,
    /// What the statement writes beside the tables its reads open: reads
    /// open a handle that registers nothing where it cannot write
    table_use: TableUse,
    /// A semantic cache miss at the statement's start, for its scan
    semantic_miss: Option<SemanticMiss>,
    /// Whether every parameter is a literal of the statement's text, which
    /// the semantic cache reads as the literal it was
    literal_params: bool,
    /// The LIMIT and OFFSET subqueries a statement read before its rows,
    /// behind an Arc, as few statements have one
    read_counts: Option<Arc<ReadCounts>>,
    /// The bounds every AS OF of the statement resolves against, taken at
    /// its start; behind an Arc, as few statements have one
    historical_cut: Option<Arc<HistoricalCut>>,
}

/// The counts a statement at `depth` read before its rows
#[derive(Debug)]
pub(crate) struct ReadCounts {
    pub(crate) depth: usize,
    pub(crate) counts: smallvec::SmallVec<[ReadCount; 2]>,
}

/// One count of `clause`, LIMIT or OFFSET, by its expression: the
/// statement's own at `address`, or a copy built for a rebuilt statement,
/// by `key`
#[derive(Debug)]
pub(crate) struct ReadCount {
    pub(crate) clause: &'static str,
    pub(crate) address: usize,
    pub(crate) key: super::expr_key::ExprKey,
    pub(crate) count: usize,
}

/// What a semantic cache lookup that missed before the statement's view
/// leaves the scan of that statement: its WHERE, simplified once, and no
/// second lookup
#[derive(Debug, Clone)]
pub(crate) struct SemanticMiss {
    /// The address of the `SelectStatement` the lookup was for
    pub(crate) statement: std::num::NonZeroUsize,
    /// The simplified WHERE, when simplifying changed it
    pub(crate) simplified: Option<Arc<crate::parser::ast::Expression>>,
}

/// Type alias for CTE data: (columns, rows) with Arc for zero-copy sharing
/// Uses Vec<(i64, Row)> for rows - same structure as RowVec but Arc-shareable
type CteData = (CompactArc<Vec<String>>, CompactArc<Vec<(i64, Row)>>);

/// Type alias for CTE data map to reduce type complexity
/// Uses CompactArc<Vec<String>> for columns and CompactArc<Vec<(i64, Row)>> for rows
/// to enable zero-copy sharing of CTE results with joins
type CteDataMap = StringMap<CteData>;

/// The names and columns of the CTEs a context brought when a plan was made,
/// None for none
#[derive(Debug)]
pub struct CteScope(Option<Box<[ScopedCte]>>);

/// A CTE's name and its columns
type ScopedCte = (String, CompactArc<Vec<String>>);

impl CteScope {
    /// Whether `ctx` brings these CTEs and no other, each with its columns
    pub(crate) fn holds(&self, ctx: &ExecutionContext) -> bool {
        let data = ctx.cte_data.as_deref().filter(|data| !data.is_empty());
        match (&self.0, data) {
            (None, None) => true,
            (Some(held), Some(data)) => {
                held.len() == data.len()
                    && held.iter().all(|(name, columns)| {
                        data.get(name)
                            .is_some_and(|(now, _)| now.as_slice() == columns.as_slice())
                    })
            }
            _ => false,
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new empty execution context
    /// Uses static defaults for empty collections to avoid allocations
    pub fn new() -> Self {
        Self {
            params: EMPTY_PARAMS.clone(),
            auto_commit: true,
            current_database: EMPTY_DATABASE.clone(),
            session_vars: EMPTY_SESSION_VARS.clone(),
            deadline: Deadline::NONE,
            view_depth: 0,
            query_depth: 0,
            cte_data: None,
            transaction_id: None,
            statement_snapshot: None,
            statement_view: None,
            table_use: TableUse::Write,
            semantic_miss: None,
            literal_params: false,
            read_counts: None,
            historical_cut: None,
        }
    }

    /// Create an execution context with positional parameters
    pub fn with_params(params: ParamVec) -> Self {
        Self {
            params: CompactArc::new(params),
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

    /// Get all positional parameters
    pub fn params(&self) -> &[Value] {
        &self.params
    }

    /// Get the params Arc for zero-copy sharing.
    /// Used by evaluator bridge to avoid cloning params.
    pub fn params_arc(&self) -> &CompactArc<ParamVec> {
        &self.params
    }

    /// Get the number of positional parameters
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Set positional parameters
    pub fn set_params(&mut self, params: ParamVec) {
        self.params = CompactArc::new(params);
    }

    /// Set positional parameters shared with another holder
    pub(crate) fn set_params_arc(&mut self, params: CompactArc<ParamVec>) {
        self.params = params;
    }

    /// Marks every parameter as a literal of the statement's text, or not
    pub(crate) fn set_literal_params(&mut self, literal: bool) {
        self.literal_params = literal;
    }

    /// Whether every parameter is a literal of the statement's text
    pub(crate) fn literal_params(&self) -> bool {
        self.literal_params
    }

    /// Add a positional parameter
    pub fn add_param(&mut self, value: Value) {
        CompactArc::make_mut(&mut self.params).push(value);
    }

    /// Check if auto-commit is enabled
    pub fn auto_commit(&self) -> bool {
        self.auto_commit
    }

    /// Set auto-commit mode
    pub fn set_auto_commit(&mut self, auto_commit: bool) {
        self.auto_commit = auto_commit;
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

    /// The instant past which the call's statements stop, if any
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline.instant()
    }

    /// Stop the call's statements past `at`; None runs them to their end
    pub fn set_deadline(&mut self, at: Option<Instant>) {
        self.deadline = Deadline::at(at);
    }

    /// Stop the call's statements `timeout_ms` from now; 0 sets no deadline
    pub fn set_timeout_ms(&mut self, timeout_ms: u64) {
        self.deadline = Deadline::after_ms(timeout_ms);
    }

    /// The call's deadline, which a result read after the call keeps
    #[inline]
    pub(crate) fn deadline_of_call(&self) -> Deadline {
        self.deadline
    }

    /// Count `units` of work at `site` against the deadline
    #[inline]
    pub(crate) fn charge(&self, site: DeadlineSite, units: usize) -> Result<()> {
        self.deadline.charge(site, units)
    }

    /// Fail when the deadline has passed, reading the clock now
    #[inline]
    pub(crate) fn check_deadline(&self) -> Result<()> {
        self.deadline.check()
    }

    /// Get the current view nesting depth
    pub fn view_depth(&self) -> usize {
        self.view_depth
    }

    /// Create a new context with incremented view depth.
    /// Used when executing nested views to track recursion depth.
    /// Also increments query_depth since views are nested queries.
    /// A view's query reads none of the CTEs of the statement reading it,
    /// and none of its parameters, which a view cannot take.
    pub fn with_incremented_view_depth(&self) -> Self {
        Self {
            params: EMPTY_PARAMS.clone(),
            auto_commit: self.auto_commit,
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            deadline: self.deadline,
            view_depth: self.view_depth + 1,
            query_depth: self.query_depth + 1, // Views are nested queries
            cte_data: None,
            transaction_id: self.transaction_id,
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            literal_params: false,
            read_counts: None,
            historical_cut: None,
        }
    }

    /// Create a new context with incremented query depth.
    /// Used when executing subqueries.
    pub fn with_incremented_query_depth(&self) -> Self {
        Self {
            params: self.params.clone(),
            auto_commit: self.auto_commit,
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            deadline: self.deadline,
            view_depth: self.view_depth,
            query_depth: self.query_depth + 1,
            cte_data: self.cte_data.clone(),
            transaction_id: self.transaction_id,
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            literal_params: false,
            read_counts: None,
            historical_cut: self.historical_cut.clone(),
        }
    }

    /// Get CTE data by name that is already lowercase.
    /// Use this when the name is known to be lowercase (e.g., from value_lower fields)
    /// to avoid redundant to_lowercase() allocation.
    #[inline]
    pub fn get_cte_by_lower(&self, name_lower: &str) -> Option<&CteData> {
        self.cte_data.as_ref().and_then(|data| data.get(name_lower))
    }

    /// Check if context has CTE data
    pub fn has_cte(&self, name: &str) -> bool {
        self.cte_data
            .as_ref()
            .is_some_and(|data| data.contains_key(&name.to_lowercase()))
    }

    /// Whether this context brings CTEs a statement run in it may read
    #[inline]
    pub(crate) fn brings_ctes(&self) -> bool {
        self.cte_data
            .as_deref()
            .is_some_and(|data| !data.is_empty())
    }

    /// The CTEs this context brings, shared
    pub(crate) fn cte_data(&self) -> Option<&Arc<CteDataMap>> {
        self.cte_data.as_ref()
    }

    /// The names and columns of the CTEs this context brings
    pub(crate) fn cte_scope(&self) -> CteScope {
        CteScope(
            self.cte_data
                .as_deref()
                .filter(|data| !data.is_empty())
                .map(|data| {
                    data.iter()
                        .map(|(name, (columns, _))| (name.clone(), columns.clone()))
                        .collect()
                }),
        )
    }

    /// Check if context has CTE data by name that is already lowercase.
    /// Use this when the name is known to be lowercase to avoid allocation.
    #[inline]
    pub fn has_cte_by_lower(&self, name_lower: &str) -> bool {
        self.cte_data
            .as_ref()
            .is_some_and(|data| data.contains_key(name_lower))
    }

    /// Create a new context with CTE data for subqueries to reference
    /// Takes an Arc to avoid cloning large CTE datasets
    pub fn with_cte_data(&self, cte_data: Arc<CteDataMap>) -> Self {
        Self {
            params: self.params.clone(),
            auto_commit: self.auto_commit,
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            deadline: self.deadline,
            view_depth: self.view_depth,
            query_depth: self.query_depth,
            cte_data: Some(cte_data),
            transaction_id: self.transaction_id,
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            literal_params: false,
            read_counts: self.read_counts.clone(),
            historical_cut: self.historical_cut.clone(),
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
    /// The statement-scoped snapshot, if the statement carries one
    pub fn statement_snapshot(&self) -> Option<&StatementSnapshot> {
        self.statement_snapshot.as_ref()
    }

    /// A copy of this context whose reads go through `snapshot`
    pub fn with_statement_snapshot(&self, snapshot: StatementSnapshot) -> Self {
        let mut ctx = self.clone();
        ctx.statement_snapshot = Some(snapshot);
        ctx
    }

    /// The statement view the statement's reads see, if it has one: its own,
    /// else the one its statement snapshot reads through
    pub fn statement_view(&self) -> Option<&StatementView> {
        self.statement_view
            .as_ref()
            .or_else(|| self.statement_snapshot.as_ref()?.view())
    }

    /// Err once a truncate or drop overtook the statement's view: what the
    /// statement read through it may be gone
    pub(crate) fn view_held(&self) -> crate::core::Result<()> {
        self.statement_view().map_or(Ok(()), StatementView::held)
    }

    /// A copy of this context whose reads see `view`
    pub fn with_statement_view(&self, view: StatementView) -> Self {
        self.clone().into_statement_view(view)
    }

    /// This context, a statement's own copy, with its reads seeing `view`
    pub(crate) fn into_statement_view(mut self, view: StatementView) -> Self {
        self.statement_view = Some(view);
        self
    }

    /// The bounds the statement's AS OF reads resolve against, if it took them
    pub(crate) fn historical_cut(&self) -> Option<HistoricalCut> {
        self.historical_cut.as_deref().copied()
    }

    /// This context with the statement's AS OF bounds
    pub(crate) fn into_historical_cut(mut self, cut: HistoricalCut) -> Self {
        self.historical_cut = Some(Arc::new(cut));
        self
    }

    /// Lets go of the statement's view, in its own copy, before it takes
    /// another: the old one goes first, so it keeps nothing past this
    pub(crate) fn take_statement_view(&mut self) -> Option<StatementView> {
        self.statement_view.take()
    }

    /// Sets the statement's view, in its own copy
    pub(crate) fn set_statement_view(&mut self, view: StatementView) {
        self.statement_view = Some(view);
    }

    /// What the statement writes beside the tables its reads open
    pub(crate) fn table_use(&self) -> TableUse {
        self.table_use
    }

    /// This context for a statement that writes what `table_use` says,
    /// beside the tables its reads open
    pub(crate) fn with_table_use(mut self, table_use: TableUse) -> Self {
        self.table_use = table_use;
        self
    }

    /// The semantic cache miss the start of `statement` left, if it is the
    /// statement this context's lookup was for
    pub(crate) fn semantic_miss(
        &self,
        statement: &crate::parser::ast::SelectStatement,
    ) -> Option<&SemanticMiss> {
        let address = statement as *const _ as usize;
        self.semantic_miss
            .as_ref()
            .filter(|miss| miss.statement.get() == address)
    }

    /// This context, a statement's own copy, carrying the miss `miss`
    pub(crate) fn into_semantic_miss(mut self, miss: SemanticMiss) -> Self {
        self.semantic_miss = Some(miss);
        self
    }

    /// The count a statement at this depth read for its `clause`, LIMIT or
    /// OFFSET, written `expr`, before its rows, if it read one
    pub(crate) fn read_count(
        &self,
        expr: &crate::parser::ast::Expression,
        clause: &'static str,
    ) -> Option<usize> {
        let read = self.read_counts.as_deref()?;
        if read.depth != self.query_depth {
            return None;
        }
        let mut counts = read.counts.iter().filter(|read| read.clause == clause);
        let address = expr as *const _ as usize;
        if let Some(read) = counts.clone().find(|read| read.address == address) {
            return Some(read.count);
        }
        let key = super::expr_key::ExprKey::of(expr);
        counts.find(|read| read.key == key).map(|read| read.count)
    }

    /// This context reading `params`: the statement's own, then the values
    /// its run computed once
    pub(crate) fn with_slot_params(&self, params: ParamVec) -> Self {
        let mut ctx = self.clone();
        ctx.params = CompactArc::new(params);
        ctx
    }

    /// This context without the counts read for its statement, for a
    /// statement of its own at the same depth; None when it carries none
    pub(crate) fn without_read_counts(&self) -> Option<Self> {
        self.read_counts.as_ref()?;
        let mut ctx = self.clone();
        ctx.read_counts = None;
        Some(ctx)
    }

    /// This context carrying `counts`, read for a statement at its depth
    pub(crate) fn with_read_counts(&self, counts: ReadCounts) -> Self {
        let mut ctx = self.clone();
        ctx.read_counts = Some(Arc::new(counts));
        ctx
    }

    pub fn with_transaction_id(&self, txn_id: u64) -> Self {
        Self {
            params: self.params.clone(),
            auto_commit: self.auto_commit,
            current_database: self.current_database.clone(),
            session_vars: self.session_vars.clone(),
            deadline: self.deadline,
            view_depth: self.view_depth,
            query_depth: self.query_depth,
            cte_data: self.cte_data.clone(),
            transaction_id: Some(txn_id),
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            table_use: self.table_use,
            semantic_miss: self.semantic_miss.clone(),
            literal_params: self.literal_params,
            read_counts: self.read_counts.clone(),
            historical_cut: self.historical_cut.clone(),
        }
    }

    /// This context, a statement's own copy, run in transaction `txn_id`
    pub(crate) fn into_transaction_id(mut self, txn_id: u64) -> Self {
        self.transaction_id = Some(txn_id);
        self
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
    pub fn params(mut self, params: ParamVec) -> Self {
        self.ctx.params = CompactArc::new(params);
        self
    }

    /// Add a positional parameter
    pub fn param(self, value: Value) -> Self {
        let mut v = (*self.ctx.params).clone();
        v.push(value);
        Self {
            ctx: ExecutionContext {
                params: CompactArc::new(v),
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
        self.ctx.set_timeout_ms(timeout_ms);
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
    use crate::common::deadline::CHARGE_BEAT;

    #[test]
    fn test_context_new() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.param_count(), 0);
        assert!(ctx.auto_commit());
        assert!(ctx.deadline().is_none());
    }

    fn ids(table: &dyn crate::storage::traits::Table) -> Vec<i64> {
        let mut ids: Vec<i64> = table
            .collect_all_rows(None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn table_of_one_row(name: &str) -> crate::api::Database {
        let db = crate::api::Database::open(&format!("memory://{name}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db
    }

    #[test]
    fn a_statement_table_at_a_view_reads_the_commits_complete_before_it() {
        use crate::storage::traits::Engine;
        let db = table_of_one_row("ctx_statement_table_at_view");
        let view = db.engine().registry().capture_view();
        db.execute("INSERT INTO t VALUES (2)", ()).unwrap();
        let txn = db.engine().begin_transaction().unwrap();
        let at_view = txn.get_statement_table_at("t", Some(&view)).unwrap();
        assert_eq!(
            ids(&*at_view),
            vec![1],
            "the view excludes the later insert"
        );
        let newest = txn.get_statement_table_at("t", None).unwrap();
        assert_eq!(ids(&*newest), vec![1, 2]);
    }

    #[test]
    fn a_snapshot_transaction_takes_no_statement_table_at_a_view() {
        use crate::storage::traits::Engine;
        let db = table_of_one_row("ctx_statement_table_at_view_snapshot");
        let view = db.engine().registry().capture_view();
        let txn = db
            .engine()
            .begin_transaction_with_level(crate::core::IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert!(txn.get_statement_table_at("t", Some(&view)).is_err());
        assert!(txn.get_statement_table_at("t", None).is_ok());
    }

    #[test]
    fn a_statement_view_is_kept_by_the_contexts_made_from_it() {
        let db = table_of_one_row("ctx_statement_view_kept");
        let view = db.engine().registry().capture_view();
        let ctx = ExecutionContext::new().with_statement_view(view.clone());
        let derived = [
            ctx.with_incremented_view_depth(),
            ctx.with_incremented_query_depth(),
            ctx.with_cte_data(Arc::new(CteDataMap::default())),
            ctx.with_transaction_id(1),
            ctx.clone(),
        ];
        for derived in &derived {
            assert!(derived
                .statement_view()
                .is_some_and(|kept| Arc::ptr_eq(&kept.spare(), &view.spare())));
        }
        assert!(ExecutionContext::new().statement_view().is_none());
    }

    #[test]
    fn test_context_with_params() {
        let ctx = ExecutionContext::with_params(smallvec::smallvec![
            Value::Integer(1),
            Value::text("hello")
        ]);
        assert_eq!(ctx.param_count(), 2);
        assert_eq!(ctx.get_param(1), Some(&Value::Integer(1)));
        assert_eq!(ctx.get_param(2), Some(&Value::text("hello")));
        assert_eq!(ctx.get_param(0), None); // 0 is invalid
        assert_eq!(ctx.get_param(3), None); // Out of bounds
    }

    #[test]
    fn a_passed_deadline_fails_its_check_and_every_charge() {
        let mut ctx = ExecutionContext::new();
        ctx.set_deadline(Some(Instant::now()));
        assert!(ctx.check_deadline().is_err());
        for _ in 0..3 {
            assert!(ctx.charge(DeadlineSite::Scan, 1).is_err());
        }
    }

    #[test]
    fn a_context_without_a_deadline_never_stops() {
        let ctx = ExecutionContext::new();
        assert!(ctx.check_deadline().is_ok());
        for _ in 0..3 * CHARGE_BEAT {
            assert!(ctx.charge(DeadlineSite::Scan, 1).is_ok());
        }
    }

    #[test]
    fn a_deadline_ahead_is_read_once_a_beat() {
        let mut ctx = ExecutionContext::new();
        ctx.set_timeout_ms(60_000);
        assert!(ctx.check_deadline().is_ok());
        for _ in 0..3 * CHARGE_BEAT {
            assert!(ctx.charge(DeadlineSite::Scan, 1).is_ok());
        }
    }

    #[test]
    fn a_deadline_ahead_never_spares_a_passed_one_on_its_thread() {
        let mut ahead = ExecutionContext::new();
        ahead.set_timeout_ms(60_000);
        let mut passed = ExecutionContext::new();
        passed.set_deadline(Some(Instant::now()));
        for _ in 0..3 {
            assert!(ahead.charge(DeadlineSite::Scan, 1).is_ok());
            assert!(
                passed.charge(DeadlineSite::Scan, 1).is_err(),
                "a passed deadline charged against another's beat"
            );
        }
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
            .params(smallvec::smallvec![Value::Integer(1)])
            .param(Value::Integer(2))
            .auto_commit(false)
            .database("mydb")
            .timeout_ms(5000)
            .build();

        assert_eq!(ctx.param_count(), 2);
        assert_eq!(ctx.get_param(1), Some(&Value::Integer(1)));
        assert_eq!(ctx.get_param(2), Some(&Value::Integer(2)));
        assert!(!ctx.auto_commit());
        assert_eq!(ctx.current_database(), Some("mydb"));
        assert!(ctx.deadline().is_some());
    }
}
