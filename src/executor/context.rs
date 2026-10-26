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

use crate::common::time_compat::Instant;
use crate::storage::mvcc::registry::StatementView;
use crate::storage::traits::transaction::{StatementTable, TableUse};
use lru::LruCache;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

// Cache size limits for subquery caches to prevent unbounded memory growth.
// These are per-thread limits since the caches are thread-local.
const SCALAR_SUBQUERY_CACHE_SIZE: usize = 128;
const IN_SUBQUERY_CACHE_SIZE: usize = 128;
const SEMI_JOIN_CACHE_SIZE: usize = 256;

use crate::api::params::ParamVec;
use crate::common::{CompactArc, SmartString, StringMap};
use crate::core::predicate::PredicateSet;
use crate::core::{Result, Row, Value, ValueMap};

// Static defaults for ExecutionContext to avoid allocations for empty values.
// These are shared across all contexts and only require Arc refcount bump on clone.
// Note: cancelled is NOT shared - each context needs its own cancellation flag.
static EMPTY_PARAMS: LazyLock<CompactArc<ParamVec>> =
    LazyLock::new(|| CompactArc::new(ParamVec::new()));
static EMPTY_DATABASE: LazyLock<Arc<Option<String>>> = LazyLock::new(|| Arc::new(None));
static EMPTY_SESSION_VARS: LazyLock<Arc<AHashMap<String, Value>>> =
    LazyLock::new(|| Arc::new(AHashMap::new()));

// Cache for scalar subquery results to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Uses SQL string as key (not hash) to avoid collision risk.
// Each result serves only a lookup under the stamp it was read under.
// LRU-bounded to prevent unbounded memory growth.
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

/// Cached scalar subquery entry: the stamp it was read under, and the value
type ScalarSubqueryCacheEntry = (SubqueryStamp, Value);

thread_local! {
    static SCALAR_SUBQUERY_CACHE: RefCell<LruCache<String, ScalarSubqueryCacheEntry>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(SCALAR_SUBQUERY_CACHE_SIZE).unwrap()));
}

/// Clear the scalar subquery cache completely.
pub fn clear_scalar_subquery_cache() {
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// The cached scalar subquery result of `key` read under `stamp`; an
/// entry read under another stamp is dropped
pub fn get_cached_scalar_subquery(key: &str, stamp: &SubqueryStamp) -> Option<Value> {
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        match c.get(key) {
            Some((kept, value)) if kept == stamp => {
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::subquery_cache_hit();
                Some(value.clone())
            }
            Some(_) => {
                c.pop(key);
                None
            }
            None => None,
        }
    })
}

/// Cache a scalar subquery result under the stamp taken before its read
pub fn cache_scalar_subquery(key: String, stamp: SubqueryStamp, value: Value) {
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().put(key, (stamp, value));
    });
}

// Cache for IN subquery results to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Uses SQL string as key (not hash) to avoid collision risk.
// Each result serves only a lookup under the stamp it was read under.
// LRU-bounded to prevent unbounded memory growth.

/// Cached IN subquery entry: (the stamp it was read under, result values,
/// lazily built shared hash set for single-column IN)
type InSubqueryCacheEntry = (SubqueryStamp, Vec<Value>, Option<CompactArc<PredicateSet>>);

thread_local! {
    static IN_SUBQUERY_CACHE: RefCell<LruCache<String, InSubqueryCacheEntry>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(IN_SUBQUERY_CACHE_SIZE).unwrap()));
}

/// Clear the IN subquery cache completely.
pub fn clear_in_subquery_cache() {
    IN_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// The cache entry of `key` read under `stamp`, dropping one read under another
fn with_in_subquery_entry<T>(
    key: &str,
    stamp: &SubqueryStamp,
    read: impl FnOnce(&mut InSubqueryCacheEntry) -> T,
) -> Option<T> {
    IN_SUBQUERY_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        match c.get_mut(key) {
            Some(entry) if entry.0 == *stamp => {
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::subquery_cache_hit();
                Some(read(entry))
            }
            Some(_) => {
                c.pop(key);
                None
            }
            None => None,
        }
    })
}

/// The cached IN subquery result of `key` read under `stamp`
pub fn get_cached_in_subquery(key: &str, stamp: &SubqueryStamp) -> Option<Vec<Value>> {
    with_in_subquery_entry(key, stamp, |entry| entry.1.clone())
}

/// Get a cached IN subquery result as a shared hash set, building and
/// memoizing the set inside the entry on first request. Later hits are
/// an Arc clone: no value clones, no set rebuild.
pub fn get_cached_in_subquery_set(
    key: &str,
    stamp: &SubqueryStamp,
) -> Option<CompactArc<PredicateSet>> {
    with_in_subquery_entry(key, stamp, |entry| {
        if entry.2.is_none() {
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::in_subquery_set_built();
            let members = CompactArc::new(entry.1.iter().cloned().collect());
            entry.2 = Some(CompactArc::new(PredicateSet::of(members)));
        }
        entry.2.clone()
    })
    .flatten()
}

/// Cache an IN subquery result under the stamp taken before its read. A
/// result holding a NULL is not kept: the negated form of the same subquery
/// reads this cache and keeps no row for such a set, which its readers
/// decide on a fresh list alone
pub fn cache_in_subquery(key: String, stamp: SubqueryStamp, values: Vec<Value>) {
    if values.iter().any(Value::is_null) {
        return;
    }
    IN_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().put(key, (stamp, values, None));
    });
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

// Cache for semi-join (EXISTS) hash sets to avoid re-execution.
// Thread-local to avoid synchronization overhead.
// Keyed by a u64 hash, checked against the entry's origin on every hit.
// LRU-bounded to prevent unbounded memory growth.
use ahash::AHashMap;
use std::hash::{Hash, Hasher};

/// What a semi-join set is taken for: the inner table, column and alias,
/// the inner filter, and the statement's parameter values
pub struct SemiJoinOrigin<'a> {
    pub table: &'a str,
    pub column: &'a str,
    pub alias: Option<&'a str>,
    pub filter: Option<&'a Arc<crate::parser::ast::Expression>>,
    pub params: &'a [Value],
}

/// A `SemiJoinOrigin` as its cache entry keeps it
struct KeptOrigin {
    table: CompactArc<str>,
    column: String,
    alias: Option<String>,
    filter: Option<Arc<crate::parser::ast::Expression>>,
    params: Vec<Value>,
}

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

impl SemiJoinOrigin<'_> {
    /// The cache key: a hash of every part, taken without allocation
    pub fn key(&self) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        self.table.hash(&mut hasher);
        self.column.hash(&mut hasher);
        self.alias.hash(&mut hasher);
        self.filter
            .map(|filter| super::expression::compute_expression_hash(filter))
            .hash(&mut hasher);
        self.params.len().hash(&mut hasher);
        for value in self.params {
            hash_parameter(value, &mut hasher);
        }
        hasher.finish()
    }

    fn is(&self, kept: &KeptOrigin) -> bool {
        let same_filter = match (self.filter, &kept.filter) {
            (None, None) => true,
            (Some(filter), Some(kept)) => Arc::ptr_eq(filter, kept) || **filter == **kept,
            _ => false,
        };
        same_filter
            && *kept.table == *self.table
            && kept.column == self.column
            && kept.alias.as_deref() == self.alias
            && kept.params.len() == self.params.len()
            && kept
                .params
                .iter()
                .zip(self.params)
                .all(|(kept, value)| same_parameter(kept, value))
    }

    fn keep(&self) -> KeptOrigin {
        KeptOrigin {
            table: CompactArc::from(self.table),
            column: self.column.to_string(),
            alias: self.alias.map(str::to_string),
            filter: self.filter.cloned(),
            params: self.params.to_vec(),
        }
    }
}

/// Cached semi-join entry: its origin, checked on a hit, the stamp the set
/// was read under, and the set
type SemiJoinCacheEntry = (KeptOrigin, SubqueryStamp, CompactArc<PredicateSet>);

thread_local! {
    static SEMI_JOIN_CACHE: RefCell<LruCache<u64, SemiJoinCacheEntry>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(SEMI_JOIN_CACHE_SIZE).unwrap()));
}

/// Clear the semi-join cache completely.
pub fn clear_semi_join_cache() {
    SEMI_JOIN_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// The cached semi-join set taken for `origin` under `stamp`, under its
/// `key`; an entry read under another stamp is dropped
#[inline]
pub fn get_cached_semi_join(
    origin: &SemiJoinOrigin<'_>,
    key: u64,
    stamp: &SubqueryStamp,
) -> Option<CompactArc<PredicateSet>> {
    SEMI_JOIN_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        match c.get(&key) {
            Some((kept, kept_stamp, set)) if origin.is(kept) => {
                if kept_stamp == stamp {
                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::subquery_cache_hit();
                    return Some(CompactArc::clone(set));
                }
                c.pop(&key);
                None
            }
            _ => None,
        }
    })
}

/// Cache a semi-join set under the stamp taken before its read
#[inline]
pub fn cache_semi_join_arc(
    origin: &SemiJoinOrigin<'_>,
    key: u64,
    stamp: SubqueryStamp,
    values: CompactArc<PredicateSet>,
) {
    SEMI_JOIN_CACHE.with(|cache| {
        cache.borrow_mut().put(key, (origin.keep(), stamp, values));
    });
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
        match &self.set {
            MemberSet::Rows(members) => self.settle(members.answer(row), members.decides()),
            _ => Err(one_value_against_rows()),
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

/// A prepared block's answer for a whole run, or for one call of its parent
#[derive(Debug, Clone)]
pub enum BlockAnswer {
    /// A scalar block's value
    Value(Value),
    /// An EXISTS block's value
    Once(bool),
    Keyed(CompactArc<ExistsKeys>),
    /// A value block's members
    Members(CompactArc<KeptMembers>),
    KeyedMembers(CompactArc<MemberKeys>),
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
    > = RefCell::new(LruCache::new(NonZeroUsize::new(SEMI_JOIN_CACHE_SIZE).unwrap()));
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

// Cache for table schema column names to avoid repeated get_table_schema() calls.
// The key is the table name, the value is the list of column names.
thread_local! {
    static EXISTS_SCHEMA_CACHE: RefCell<FxHashMap<String, ExistsColumns>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS schema cache. Should be called at the start of each top-level query.
pub fn clear_exists_schema_cache() {
    EXISTS_SCHEMA_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// An EXISTS probe table's column names and the token their positions hold for
pub type ExistsColumns = (
    CompactArc<Vec<String>>,
    crate::storage::mvcc::version_store::LayoutToken,
);

/// Get cached table column names by table name.
pub fn get_cached_exists_schema(key: &str) -> Option<ExistsColumns> {
    EXISTS_SCHEMA_CACHE.with(|cache| cache.borrow().get(key).cloned())
}

/// Cache table column names (takes Arc for zero-copy sharing).
pub fn cache_exists_schema(key: String, columns: ExistsColumns) {
    EXISTS_SCHEMA_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, columns);
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
// The key is the subquery's address, as for the lookup info; the value maps a group key to its aggregate.
thread_local! {
    static BATCH_AGGREGATE_CACHE: RefCell<FxHashMap<usize, CompactArc<BatchAggregate>>> = RefCell::new(FxHashMap::default());
}

/// A batch aggregate's results by group key, with the kinds of the keys
pub struct BatchAggregate {
    pub values: ValueMap<Value>,
    /// As `core::predicate::kind` names them
    pub kinds: u8,
}

/// Clear the batch aggregate cache. Should be called at the start of each top-level query.
pub fn clear_batch_aggregate_cache() {
    BATCH_AGGREGATE_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
}

/// Get a cached batch aggregate result map by subquery address.
pub fn get_cached_batch_aggregate(subquery_ptr: usize) -> Option<CompactArc<BatchAggregate>> {
    BATCH_AGGREGATE_CACHE.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Cache a batch aggregate result map.
pub fn cache_batch_aggregate(subquery_ptr: usize, aggregate: BatchAggregate) {
    BATCH_AGGREGATE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(subquery_ptr, CompactArc::new(aggregate));
    });
}

/// Pre-computed info for batch aggregate lookups to avoid per-row allocations.
#[derive(Clone)]
pub struct BatchAggregateLookupInfo {
    /// The outer column name (lowercase) to look up in outer_row
    pub outer_column_lower: String,
    /// Optional qualified outer column name (e.g., "u.id")
    pub outer_qualified_lower: Option<String>,
    /// Whether this is a COUNT expression (returns 0 for missing keys)
    pub is_count: bool,
}

// Cache for batch aggregate lookup info to avoid recomputing per row.
// The key is the subquery pointer address (usize), avoiding expensive to_string() per row.
// Value is Arc-wrapped to avoid cloning strings on every lookup.
thread_local! {
    static BATCH_AGGREGATE_INFO_CACHE: RefCell<FxHashMap<usize, Option<Arc<BatchAggregateLookupInfo>>>> = RefCell::new(FxHashMap::default());
}

/// Clear the batch aggregate info cache.
pub fn clear_batch_aggregate_info_cache() {
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
}

/// Get cached batch aggregate lookup info by subquery pointer address.
/// Returns Arc to avoid cloning strings on every lookup.
#[inline]
pub fn get_cached_batch_aggregate_info(
    subquery_ptr: usize,
) -> Option<Option<Arc<BatchAggregateLookupInfo>>> {
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Cache batch aggregate lookup info and return the Arc-wrapped version.
/// Returns None if info was None (not batchable).
#[inline]
pub fn cache_batch_aggregate_info(
    subquery_ptr: usize,
    info: Option<BatchAggregateLookupInfo>,
) -> Option<Arc<BatchAggregateLookupInfo>> {
    let arc_info = info.map(Arc::new);
    let result = arc_info.clone();
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| {
        cache.borrow_mut().insert(subquery_ptr, arc_info);
    });
    result
}

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
}

// Cache for EXISTS correlation info to avoid per-row extraction.
// The key is the subquery pointer address (usize), avoiding format! allocation.
// Value is Arc-wrapped to avoid cloning strings on every lookup.
thread_local! {
    static EXISTS_CORRELATION_CACHE: RefCell<FxHashMap<usize, Option<Arc<ExistsCorrelationInfo>>>> = RefCell::new(FxHashMap::default());
}

/// Clear the EXISTS correlation cache.
pub fn clear_exists_correlation_cache() {
    EXISTS_CORRELATION_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

// The semi-join set an EXISTS answers a refused probe from, built on its
// first refusal in the statement, by subquery address.
thread_local! {
    static EXISTS_PROBE_FALLBACKS: RefCell<FxHashMap<usize, CompactArc<PredicateSet>>> = RefCell::new(FxHashMap::default());
}

/// Get the probe fallback kept for a subquery address.
pub fn get_exists_probe_fallback(subquery_ptr: usize) -> Option<CompactArc<PredicateSet>> {
    EXISTS_PROBE_FALLBACKS.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Keep the probe fallback for a subquery address.
pub fn cache_exists_probe_fallback(subquery_ptr: usize, set: CompactArc<PredicateSet>) {
    EXISTS_PROBE_FALLBACKS.with(|cache| {
        cache.borrow_mut().insert(subquery_ptr, set);
    });
}

/// Clear the caches one statement keeps by subquery address or for its own
/// reads. Called when a statement starts, since a later statement may reuse
/// an address and must not read an earlier statement's results.
pub fn clear_statement_caches() {
    clear_exists_predicate_cache();
    clear_exists_schema_cache();
    clear_exists_pred_key_cache();
    clear_exists_correlation_cache();
    EXISTS_PROBE_FALLBACKS.with(|cache| cache.borrow_mut().clear());
    clear_batch_aggregate_cache();
    clear_batch_aggregate_info_cache();
}

/// Clear ALL thread-local caches to release memory.
/// Call this when a database is dropped to prevent memory leaks.
/// This also shrinks all cache capacities to zero where applicable.
pub fn clear_all_thread_local_caches() {
    // Clear LRU-bounded caches (no shrink_to_fit needed - fixed capacity)
    SCALAR_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
    IN_SUBQUERY_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
    SEMI_JOIN_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
    clear_block_answer_cache();
    // Clear and shrink unbounded caches
    EXISTS_PREDICATE_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    EXISTS_SCHEMA_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    EXISTS_PRED_KEY_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    BATCH_AGGREGATE_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    BATCH_AGGREGATE_INFO_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    EXISTS_CORRELATION_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });
    EXISTS_PROBE_FALLBACKS.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clear();
        c.shrink_to_fit();
    });

    // Clear storage expression caches (regex patterns)
    crate::core::pattern::clear_regex_cache();

    // Clear RowVec and RowIdVec thread-local pools
    crate::core::row_vec::clear_row_vec_pool();
    crate::core::row_vec::clear_row_id_vec_pool();

    // Clear global transaction version map pools
    crate::storage::mvcc::clear_version_map_pools();

    // Clear global LRU caches
    super::expression::clear_program_cache();
    super::query_classification::clear_classification_cache();
}

/// Get cached EXISTS correlation info by subquery pointer address.
/// Returns Arc to avoid cloning strings on every lookup.
#[inline]
pub fn get_cached_exists_correlation(
    subquery_ptr: usize,
) -> Option<Option<Arc<ExistsCorrelationInfo>>> {
    EXISTS_CORRELATION_CACHE.with(|cache| cache.borrow().get(&subquery_ptr).cloned())
}

/// Cache EXISTS correlation info and return the Arc-wrapped version.
/// Returns None if info was None (correlation not extractable).
#[inline]
pub fn cache_exists_correlation(
    subquery_ptr: usize,
    info: Option<ExistsCorrelationInfo>,
) -> Option<Arc<ExistsCorrelationInfo>> {
    let arc_info = info.map(Arc::new);
    let result = arc_info.clone();
    EXISTS_CORRELATION_CACHE.with(|cache| {
        cache.borrow_mut().insert(subquery_ptr, arc_info);
    });
    result
}

/// Execution context for SQL queries
///
/// The execution context carries state and configuration for query execution,
/// including parameters, transaction state, and cancellation support.
///
/// Note: This struct uses Arc for immutable shared data to make cloning cheap
/// during correlated subquery processing where context is cloned per row.
/// A transaction shared by every read of one statement
#[derive(Clone)]
pub struct StatementSnapshot(Arc<SnapshotState>);

struct SnapshotState {
    transaction: Mutex<Box<dyn crate::storage::traits::Transaction>>,
    /// Whether its reads see each commit as it lands (READ COMMITTED with
    /// no statement view), so a result read through it is no older than a
    /// stamp taken before the read
    reads_latest: bool,
    /// The statement view its tables are read through
    view: Option<StatementView>,
    /// What is written through its transaction: nothing for the executor's
    /// own, maybe anything for one a caller gave through `get_table`
    table_use: TableUse,
    /// The first table the statement's correlated probes read, read without a lock
    first_probe_table: std::sync::OnceLock<ProbeTable>,
    /// Any further tables they read, taken once each
    probe_tables: Mutex<SmallVec<[ProbeTable; 1]>>,
}

type ProbeTable = (SmartString, Box<dyn crate::storage::traits::Table>);

impl StatementSnapshot {
    /// The statement's reads through `transaction`, begun at `isolation`,
    /// and through `view` when the statement has one
    pub fn new(
        transaction: Box<dyn crate::storage::traits::Transaction>,
        isolation: crate::core::IsolationLevel,
        view: Option<StatementView>,
    ) -> Self {
        Self::with_use(transaction, isolation, view, TableUse::Write)
    }

    /// [`Self::new`] for the executor's own reads: nothing writes through
    /// the transaction, so its tables open without a local store
    pub(crate) fn for_reads(
        transaction: Box<dyn crate::storage::traits::Transaction>,
        isolation: crate::core::IsolationLevel,
        view: Option<StatementView>,
    ) -> Self {
        Self::with_use(transaction, isolation, view, TableUse::Read)
    }

    fn with_use(
        transaction: Box<dyn crate::storage::traits::Transaction>,
        isolation: crate::core::IsolationLevel,
        view: Option<StatementView>,
        table_use: TableUse,
    ) -> Self {
        Self(Arc::new(SnapshotState {
            transaction: Mutex::new(transaction),
            reads_latest: isolation == crate::core::IsolationLevel::ReadCommitted && view.is_none(),
            view,
            table_use,
            first_probe_table: std::sync::OnceLock::new(),
            probe_tables: Mutex::new(SmallVec::new()),
        }))
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
        self.0.view.as_ref()
    }

    pub fn get_table(
        &self,
        name: &str,
    ) -> crate::core::Result<Box<dyn crate::storage::traits::Table>> {
        self.0
            .transaction
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_table(name)
    }

    /// The table the executor's own statements use, without the write gate,
    /// read through the statement's view when it has one
    pub(crate) fn statement_table(
        &self,
        name: &str,
    ) -> crate::core::Result<Box<dyn crate::storage::traits::Table>> {
        self.statement_table_at(name, self.0.view.as_ref())
    }

    /// The table read through `view`, None reading the newest committed state
    pub(crate) fn statement_table_at(
        &self,
        name: &str,
        view: Option<&StatementView>,
    ) -> crate::core::Result<Box<dyn crate::storage::traits::Table>> {
        self.0
            .transaction
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_read_table_at(name, view, self.0.table_use)
    }

    /// Runs `f` on the table every correlated probe of the statement reads,
    /// taken on the first. None when there is no such table, as for a view
    pub(crate) fn with_probe_table<T>(
        &self,
        name: &str,
        f: impl FnOnce(&dyn crate::storage::traits::Table) -> crate::core::Result<T>,
    ) -> crate::core::Result<Option<T>> {
        let first = &self.0.first_probe_table;
        if first.get().is_none() {
            let table = match self.statement_table(name) {
                Err(crate::core::Error::TableNotFound(_)) => return Ok(None),
                table => table?,
            };
            // A probe on another thread may have set it first; either serves
            let _ = first.set((SmartString::from(name), table));
        }
        if let Some((n, table)) = first.get() {
            if n.eq_ignore_ascii_case(name) {
                return f(&**table).map(Some);
            }
        }
        let mut tables = self
            .0
            .probe_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let at = match tables
            .iter()
            .position(|(n, _)| n.eq_ignore_ascii_case(name))
        {
            Some(at) => at,
            None => {
                let table = match self.statement_table(name) {
                    Err(crate::core::Error::TableNotFound(_)) => return Ok(None),
                    table => table?,
                };
                tables.push((SmartString::from(name), table));
                tables.len() - 1
            }
        };
        f(&*tables[at].1).map(Some)
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
    /// Uses FxHashMap<CompactArc<str>, Value> for zero-cost key cloning in hot loops
    /// pub(crate) to allow taking ownership back for reuse in optimized loops
    pub(crate) outer_row: Option<FxHashMap<CompactArc<str>, Value>>,
    /// Outer row column names (for qualified identifier resolution) - wrapped in Arc
    outer_columns: Option<CompactArc<Vec<String>>>,
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
    /// A DML target selected through a query: the scan of this table at
    /// this query depth reads the newest state, the query's subqueries the
    /// view. Behind an Arc, as few statements set it and every one carries it
    newest_target: Option<Arc<(SmartString, usize)>>,
    /// What the statement writes beside the tables its reads open: reads
    /// open a handle that registers nothing where it cannot write
    table_use: TableUse,
    /// A semantic cache miss at the statement's start, for its scan
    semantic_miss: Option<SemanticMiss>,
    /// The LIMIT and OFFSET subqueries a statement read before its rows,
    /// behind an Arc, as few statements have one
    read_counts: Option<Arc<ReadCounts>>,
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
            cancelled: Arc::new(AtomicBool::new(false)), // Each context needs own flag
            current_database: EMPTY_DATABASE.clone(),
            session_vars: EMPTY_SESSION_VARS.clone(),
            timeout_ms: 0,
            view_depth: 0,
            query_depth: 0,
            outer_row: None,
            outer_columns: None,
            cte_data: None,
            transaction_id: None,
            statement_snapshot: None,
            statement_view: None,
            newest_target: None,
            table_use: TableUse::Write,
            semantic_miss: None,
            read_counts: None,
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

    /// Check if the query has been cancelled
    pub fn is_cancelled(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if let Some(manager) = TIMEOUT_MANAGER.get() {
            manager.poll();
        }
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
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            newest_target: self.newest_target.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            read_counts: None,
        }
    }

    /// Create a new context with incremented query depth.
    /// Used when executing subqueries to ensure TimeoutGuard is only created at the top level.
    pub fn with_incremented_query_depth(&self) -> Self {
        Self {
            params: self.params.clone(),
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
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            newest_target: self.newest_target.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            read_counts: None,
        }
    }

    /// Get the outer row context for correlated subqueries
    pub fn outer_row(&self) -> Option<&FxHashMap<CompactArc<str>, Value>> {
        self.outer_row.as_ref()
    }

    /// Get the outer row columns for correlated subqueries
    pub fn outer_columns(&self) -> Option<&[String]> {
        self.outer_columns.as_ref().map(|v| v.as_slice())
    }

    /// Create a new context with outer row context for correlated subqueries.
    /// The outer_row maps lowercase column names (as CompactArc<str>) to their values.
    /// NOTE: This is now cheap to clone due to Arc wrapping of immutable fields.
    pub fn with_outer_row(
        &self,
        outer_row: FxHashMap<CompactArc<str>, Value>,
        outer_columns: CompactArc<Vec<String>>,
    ) -> Self {
        // A subquery inside a subquery reads the row its grandparent sits
        // on as well, with the nearer scope winning where both name a column
        let outer_row = match &self.outer_row {
            Some(grandparent) if !grandparent.is_empty() => {
                let mut merged = grandparent.clone();
                merged.extend(outer_row);
                merged
            }
            _ => outer_row,
        };
        Self {
            params: self.params.clone(), // Arc clone = cheap
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
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            newest_target: self.newest_target.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            read_counts: None,
        }
    }

    /// Get CTE data by name (case-insensitive)
    /// Returns Arc references to enable zero-copy sharing with joins
    pub fn get_cte(&self, name: &str) -> Option<&CteData> {
        self.cte_data
            .as_ref()
            .and_then(|data| data.get(&name.to_lowercase()))
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
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            newest_target: self.newest_target.clone(),
            table_use: self.table_use,
            semantic_miss: None,
            read_counts: self.read_counts.clone(),
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

    /// Lets go of the statement's view, in its own copy, before it takes
    /// another: the old one goes first, so it keeps nothing past this
    pub(crate) fn take_statement_view(&mut self) -> Option<StatementView> {
        self.statement_view.take()
    }

    /// Sets the statement's view, in its own copy
    pub(crate) fn set_statement_view(&mut self, view: StatementView) {
        self.statement_view = Some(view);
    }

    /// A copy of this context in which the query run at this depth selects
    /// the rows of `table` a DML writes: its scan of the table reads the
    /// newest state, as the write does
    pub(crate) fn with_newest_target(&self, table: &str) -> Self {
        let mut ctx = self.clone();
        ctx.newest_target = Some(Arc::new((SmartString::from(table), self.query_depth)));
        ctx
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

    /// The view a scan of `table` at this depth reads through: none for the
    /// DML target a query selects, else the statement's
    pub(crate) fn scan_view(&self, table: &str) -> Option<&StatementView> {
        let target = self.newest_target.as_deref().is_some_and(|(name, depth)| {
            *depth == self.query_depth && name.eq_ignore_ascii_case(table)
        });
        if target {
            None
        } else {
            self.statement_view()
        }
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
            statement_snapshot: self.statement_snapshot.clone(),
            statement_view: self.statement_view.clone(),
            newest_target: self.newest_target.clone(),
            table_use: self.table_use,
            semantic_miss: self.semantic_miss.clone(),
            read_counts: self.read_counts.clone(),
        }
    }

    /// This context, a statement's own copy, run in transaction `txn_id`
    pub(crate) fn into_transaction_id(mut self, txn_id: u64) -> Self {
        self.transaction_id = Some(txn_id);
        self
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
    #[cfg(not(target_arch = "wasm32"))]
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
                #[cfg(not(target_arch = "wasm32"))]
                shutdown: false,
            }),
            condvar: Condvar::new(),
            next_id: AtomicU64::new(1),
        });

        // Spawn the background timer thread
        #[cfg(not(target_arch = "wasm32"))]
        {
            let manager_clone = Arc::clone(&manager);
            std::thread::Builder::new()
                .name("stoolap-timeout-manager".to_string())
                .spawn(move || {
                    manager_clone.run();
                })
                .expect("Failed to spawn timeout manager thread");
        }

        manager
    }

    /// Cancels every query whose deadline is at or before `now`
    fn expire_due(state: &mut TimeoutManagerState, now: Instant) {
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
    }

    /// Without a timer thread: drops finished timeouts, then expires due ones
    #[cfg(target_arch = "wasm32")]
    fn poll(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        while state
            .timeouts
            .peek()
            .is_some_and(|e| e.cancelled.load(Ordering::Relaxed))
        {
            state.timeouts.pop();
        }
        if !state.timeouts.is_empty() {
            Self::expire_due(&mut state, Instant::now());
        }
    }

    /// Background thread loop
    #[cfg(not(target_arch = "wasm32"))]
    fn run(&self) {
        loop {
            let mut state = self.state.lock().unwrap();

            // Check for shutdown
            if state.shutdown && state.timeouts.is_empty() {
                return;
            }

            // Process expired timeouts
            let now = Instant::now();
            Self::expire_due(&mut state, now);

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
    TIMEOUT_MANAGER.get_or_init(TimeoutManager::new)
}

static TIMEOUT_MANAGER: std::sync::OnceLock<Arc<TimeoutManager>> = std::sync::OnceLock::new();

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
        Self::for_ms(ctx, ctx.timeout_ms())
    }

    /// A guard cancelling `ctx`'s query `timeout_ms` from now; None for 0
    pub(crate) fn for_ms(ctx: &ExecutionContext, timeout_ms: u64) -> Option<Self> {
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
    use crate::core::DataType;
    use rustc_hash::FxHashMap;

    #[test]
    fn test_context_new() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.param_count(), 0);
        assert!(ctx.auto_commit());
        assert!(!ctx.is_cancelled());
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
            ctx.with_outer_row(FxHashMap::default(), CompactArc::new(Vec::new())),
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
        assert_eq!(ctx.timeout_ms(), 5000);
    }

    #[test]
    fn a_semi_join_set_answers_only_its_own_origin_under_a_shared_key() {
        let origin = |table, params| SemiJoinOrigin {
            table,
            column: "bid",
            alias: None,
            filter: None,
            params,
        };
        let taken = origin("semi_join_origin_test", &[Value::Integer(10)]);
        let key = taken.key();
        let set = || {
            CompactArc::new(PredicateSet::of(CompactArc::new(
                [Value::Integer(1)].into_iter().collect(),
            )))
        };
        let stamp = |publish| SubqueryStamp {
            engine: 0,
            schema: 0,
            publish: smallvec::smallvec![publish],
        };
        cache_semi_join_arc(&taken, key, stamp(1), set());
        assert!(get_cached_semi_join(&taken, key, &stamp(1)).is_some());
        for other in [
            origin("semi_join_origin_test", &[Value::Integer(20)]),
            origin("semi_join_origin_test", &[Value::Float(10.0)]),
            origin("semi_join_origin_other", &[Value::Integer(10)]),
        ] {
            assert!(
                get_cached_semi_join(&other, key, &stamp(1)).is_none(),
                "a set kept under the key answered another origin"
            );
        }
        assert!(
            get_cached_semi_join(&taken, key, &stamp(2)).is_none(),
            "a set read under one stamp answered another"
        );
        assert!(
            get_cached_semi_join(&taken, key, &stamp(1)).is_none(),
            "a set seen under a newer stamp stayed"
        );

        let null = origin("semi_join_origin_test", &[Value::Null(DataType::Integer)]);
        let null_key = null.key();
        let other_null = origin("semi_join_origin_test", &[Value::Null(DataType::Float)]);
        assert_ne!(null_key, other_null.key());
        cache_semi_join_arc(&null, null_key, stamp(1), set());
        assert!(get_cached_semi_join(&other_null, null_key, &stamp(1)).is_none());
        clear_semi_join_cache();
    }
}
