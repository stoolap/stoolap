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

//! The order a run of INNER joins over base tables is joined in: from the
//! table estimated to keep the fewest rows, each next table one its join
//! conditions reach at the least estimated cost, taken only when the whole
//! order is estimated at most half the cost of the written one

use crate::common::SmartString;
use crate::parser::ast::{
    Expression, InfixOperator, JoinTableSource, SelectStatement, SimpleTableSource,
};
use crate::storage::traits::Engine;

use std::sync::Arc;

use super::context::ExecutionContext;
use super::planner::QueryPlanner;
use super::query::CteNames;
use super::query_cache::NestedPlan;
use super::utils::{combine_predicates_with_and, extract_and_conditions, visit_column_refs};
use super::Executor;

/// How many times cheaper the chosen order must be estimated than the
/// written one before the written one is left
const DOMINANCE: f64 = 2.0;

/// The order a statement's plan holds for its joins
#[derive(Debug, Default)]
pub(crate) enum HeldOrder {
    #[default]
    Undecided,
    /// The statement is itself an order decided, run as it stands
    Fixed,
    /// The statement reordered under the key, or None to run it as written
    Decided(OrderKey, Option<Arc<OrderedPlan>>),
}

/// A reordered statement with its own plan
#[derive(Debug)]
pub(crate) struct OrderedPlan {
    pub order: JoinOrder,
    pub plan: NestedPlan,
}

/// A statement whose joins run in another order than written
#[derive(Debug)]
pub(crate) struct JoinOrder {
    pub stmt: SelectStatement,
    /// The joined tables by alias, as written and as joined
    pub written: Vec<SmartString>,
    pub chosen: Vec<SmartString>,
}

/// What a memo of the order holds it under: the schema and statistics
/// epochs, and the log2 bucket of each joined table's rows
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct OrderKey {
    schema: u64,
    stats: u64,
    buckets: Vec<u32>,
}

impl OrderKey {
    /// Whether an order decided under `self` still holds under `now`: the
    /// same epochs, and no table's rows moved by two buckets
    pub(crate) fn holds(&self, now: &OrderKey) -> bool {
        self.schema == now.schema
            && self.stats == now.stats
            && self.buckets.len() == now.buckets.len()
            && (self.buckets.iter().zip(&now.buckets)).all(|(a, b)| a.abs_diff(*b) < 2)
    }
}

/// One joined table of the reordered run
struct Leaf<'s> {
    source: &'s SimpleTableSource,
    alias: &'s str,
    schema: crate::common::CompactArc<crate::core::Schema>,
    /// Rows the table holds, and of them the rows its own WHERE terms keep
    total: f64,
    kept: f64,
    /// Whether every row is hot, so its indexes count all of them
    all_hot: bool,
}

/// An ON term of the run, with the leaves it reads
struct Term<'s> {
    expr: &'s Expression,
    leaves: Vec<usize>,
}

/// An equality between a column of each of two leaves
struct Edge {
    ends: [(usize, SmartString); 2],
}

/// The facts a step's estimate reads of one side of an edge
struct Column {
    /// Whether no two rows share a value: a primary key or a unique index
    unique: bool,
    /// Whether an index or the primary key answers a probe for a value
    probed: bool,
    distinct: Option<f64>,
}

impl Executor {
    /// The reordered statement `stmt` runs as, decided once per key in the
    /// plan `compiled` holds for it; None to run it as written, found
    /// without a lock or a count for a statement with no run of joins
    pub(crate) fn ordered_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        compiled: Option<&NestedPlan>,
    ) -> Option<Arc<OrderedPlan>> {
        let (_, sources) = join_run(stmt)?;
        let decide = || {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::join_order_decided();
            let order = self.reorder_joins(stmt, CteNames::Context(ctx))?;
            let plan = NestedPlan::default();
            *plan.join_order().write().ok()? = HeldOrder::Fixed;
            Some(Arc::new(OrderedPlan { order, plan }))
        };
        let Some(compiled) = compiled else {
            return decide();
        };
        if matches!(*compiled.join_order().read().ok()?, HeldOrder::Fixed) {
            return None;
        }
        let key = self.join_order_key(&sources)?;
        if let HeldOrder::Decided(held, plan) = &*compiled.join_order().read().ok()? {
            if held.holds(&key) {
                return plan.clone();
            }
        }
        let plan = decide();
        *compiled.join_order().write().ok()? = HeldOrder::Decided(key, plan.clone());
        plan
    }

    /// The memo key of the order of a run of joins over `sources`
    fn join_order_key(&self, sources: &[&SimpleTableSource]) -> Option<OrderKey> {
        let buckets = (sources.iter())
            .map(|source| {
                let rows = self.engine.row_count_hint(&source.name.value_lower)?;
                Some(usize::BITS - rows.leading_zeros())
            })
            .collect::<Option<Vec<u32>>>()?;
        Some(OrderKey {
            schema: self.engine.schema_epoch(),
            stats: self.engine.stats_epoch(),
            buckets,
        })
    }

    /// `stmt` with its run of INNER joins in the order estimated cheapest,
    /// when it is estimated at most half the cost of the written order;
    /// None when the statement has no such run, a table of it cannot be
    /// estimated, or the written order is kept
    pub(crate) fn reorder_joins(
        &self,
        stmt: &SelectStatement,
        ctes: CteNames<'_>,
    ) -> Option<JoinOrder> {
        let (run, sources) = join_run(stmt)?;
        if !reads_plain_names(stmt) {
            return None;
        }
        let leaves = self.leaves(&sources, stmt, ctes)?;
        let terms = run_terms(run, &leaves)?;
        let edges: Vec<Edge> = terms.iter().filter_map(|t| edge(t, &leaves)).collect();
        let columns = |leaf: usize, column: &str| self.column_facts(&leaves[leaf], column);
        let written: Vec<usize> = (0..leaves.len()).collect();
        let written_cost = order_cost(&written, &leaves, &edges, &columns)?;
        let chosen = greedy_order(&leaves, &edges, &columns)?;
        let chosen_cost = order_cost(&chosen, &leaves, &edges, &columns)?;
        if chosen == written || chosen_cost * DOMINANCE > written_cost {
            return None;
        }
        let joined = rebuild(run, &chosen, &leaves, &terms)?;
        let mut reordered = stmt.clone();
        reordered.table_expr = Some(Box::new(replace_run(
            stmt.table_expr.as_deref()?,
            run,
            joined,
        )));
        let alias = |&leaf: &usize| SmartString::from(leaves[leaf].alias);
        Some(JoinOrder {
            stmt: reordered,
            written: written.iter().map(alias).collect(),
            chosen: chosen.iter().map(alias).collect(),
        })
    }

    /// Each base table of the run with its rows and the rows its own WHERE
    /// terms keep, as the join side estimates count them
    fn leaves<'s>(
        &self,
        sources: &[&'s SimpleTableSource],
        stmt: &SelectStatement,
        ctes: CteNames<'_>,
    ) -> Option<Vec<Leaf<'s>>> {
        let planner = self.get_query_planner();
        let wheres = stmt.where_clause.as_deref().map(extract_and_conditions);
        let mut leaves = Vec::with_capacity(sources.len());
        for &source in sources {
            if ctes.hides(&source.name.value_lower) {
                return None;
            }
            leaves.push(self.leaf(source)?);
        }
        // Each WHERE term that reads one leaf alone narrows that leaf
        for term in wheres.into_iter().flatten() {
            if let Some([leaf]) = term_leaves(term, &leaves).as_deref() {
                let leaf = &mut leaves[*leaf];
                let arms = planner.share_arms(&leaf.source.name.value_lower, term);
                leaf.kept *= planner.arms_share(&arms, &[]);
            }
        }
        Some(leaves)
    }

    /// The base table `source` names, all of its rows kept; None for one
    /// read AS OF or that the engine does not hold
    fn leaf<'s>(&self, source: &'s SimpleTableSource) -> Option<Leaf<'s>> {
        let table = source.name.value_lower.as_str();
        if source.as_of.is_some() {
            return None;
        }
        let schema = self.engine.get_table_schema(table).ok()?;
        let store = self.engine.get_version_store(table).ok()?;
        let rows = self.engine.row_count_hint(table)?;
        let alias = source.alias.as_ref().unwrap_or(&source.name);
        Some(Leaf {
            source,
            alias: alias.value_lower.as_str(),
            schema,
            total: rows as f64,
            kept: rows as f64,
            all_hot: rows == store.committed_row_count(),
        })
    }

    /// The rows the equality join `join` of two base tables gives, `kept`
    /// rows of each side kept by its own conditions, by the key estimated to
    /// keep the fewest; None for another join
    pub(crate) fn equality_join_rows(&self, join: &JoinTableSource, kept: [f64; 2]) -> Option<f64> {
        fn source(side: &Expression) -> Option<&SimpleTableSource> {
            match side {
                Expression::TableSource(source) => Some(source),
                _ => None,
            }
        }
        let mut leaves = vec![
            self.leaf(source(&join.left)?)?,
            self.leaf(source(&join.right)?)?,
        ];
        for (leaf, kept) in leaves.iter_mut().zip(kept) {
            leaf.kept = kept;
        }
        let terms = (extract_and_conditions(join.condition.as_deref()?).into_iter())
            .map(|expr| {
                Some(Term {
                    expr,
                    leaves: term_leaves(expr, &leaves)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let columns = |leaf: usize, column: &str| self.column_facts(&leaves[leaf], column);
        (terms.iter().filter_map(|term| edge(term, &leaves)))
            .map(|edge| {
                let [(a, a_column), (b, b_column)] = &edge.ends;
                let ends = [(*a, a_column.as_str(), true), (*b, b_column.as_str(), true)];
                edge_rows(ends, [kept[*a], kept[*b]], &leaves, &columns)
            })
            .min_by(f64::total_cmp)
    }

    /// The facts a join step reads of `column` of `leaf`
    fn column_facts(&self, leaf: &Leaf<'_>, column: &str) -> Column {
        let table = leaf.source.name.value_lower.as_str();
        let pk = (leaf.schema.pk_column_index())
            .is_some_and(|at| leaf.schema.columns[at].name_lower == column);
        let index = (self.engine.get_version_store(table).ok())
            .and_then(|store| store.get_index_by_column(column));
        let unique = pk || index.as_ref().is_some_and(|index| index.is_unique());
        let distinct = if unique {
            Some(leaf.total)
        } else {
            let counted = (index.as_ref())
                .filter(|_| leaf.all_hot)
                .and_then(|index| index.get_distinct_count_excluding_null());
            let analyzed = || {
                (self.get_query_planner().get_column_stats(table, column))
                    .map(|stats| stats.distinct_count as usize)
                    .filter(|&distinct| distinct > 0)
            };
            counted.or_else(analyzed).map(|distinct| distinct as f64)
        };
        Column {
            unique,
            probed: pk || index.is_some(),
            distinct,
        }
    }
}

/// The run of INNER joins at the bottom of `stmt`'s left-deep join tree, and
/// its base tables in written order: three or more, each joined by an ON
fn join_run(stmt: &SelectStatement) -> Option<(&JoinTableSource, Vec<&SimpleTableSource>)> {
    if stmt.with.is_some() || !stmt.set_operations.is_empty() || stmt.prepared.is_some() {
        return None;
    }
    let Expression::JoinSource(top) = stmt.table_expr.as_deref()? else {
        return None;
    };
    // The joins down the left spine, top first
    let mut spine = vec![&**top];
    while let Expression::JoinSource(left) = spine.last()?.left.as_ref() {
        spine.push(left);
    }
    let Expression::TableSource(first) = spine.last()?.left.as_ref() else {
        return None;
    };
    let inner = |join: &JoinTableSource| {
        join.join_type == "INNER"
            && join.condition.is_some()
            && join.using_columns.is_empty()
            && matches!(join.right.as_ref(), Expression::TableSource(_))
    };
    let below = spine.iter().rev().take_while(|join| inner(join)).count();
    if below < 2 {
        return None;
    }
    let run = spine[spine.len() - below];
    let mut sources = vec![&**first];
    for join in spine.iter().rev().take(below) {
        if let Expression::TableSource(right) = join.right.as_ref() {
            sources.push(right);
        }
    }
    Some((run, sources))
}

/// Whether every column reference of `stmt` is one a name walk sees: no
/// star, no window definition, and no part such as a subquery whose names
/// could resolve by the tables' positions
fn reads_plain_names(stmt: &SelectStatement) -> bool {
    let star = (stmt.columns.iter())
        .any(|c| matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)));
    !star
        && stmt.window_defs.is_empty()
        && clauses(stmt).all(|expr| visit_column_refs(expr, &mut |_| {}))
}

/// Every expression of `stmt` that names columns, its join conditions included
fn clauses(stmt: &SelectStatement) -> impl Iterator<Item = &Expression> {
    let mut conditions = Vec::new();
    let mut at = stmt.table_expr.as_deref();
    while let Some(Expression::JoinSource(join)) = at {
        conditions.extend(join.condition.as_deref());
        at = Some(join.left.as_ref());
    }
    (stmt.columns.iter())
        .chain(&stmt.distinct_on)
        .chain(stmt.where_clause.as_deref())
        .chain(&stmt.group_by.columns)
        .chain(stmt.having.as_deref())
        .chain(stmt.order_by.iter().map(|o| &o.expression))
        .chain(conditions)
}

/// The leaves `expr` reads, each once in leaf order; None when it reads a
/// name that is no leaf's column
fn term_leaves(expr: &Expression, leaves: &[Leaf<'_>]) -> Option<Vec<usize>> {
    let mut read = vec![false; leaves.len()];
    let mut known = true;
    visit_column_refs(expr, &mut |name| {
        let at = match name {
            Expression::QualifiedIdentifier(q) => {
                (leaves.iter()).position(|leaf| leaf.alias == q.qualifier.value_lower.as_str())
            }
            Expression::Identifier(id) => {
                (leaves.iter()).position(|leaf| leaf.schema.has_column(&id.value_lower))
            }
            _ => None,
        };
        match at {
            Some(at) => read[at] = true,
            None => known = false,
        }
    });
    known.then(|| (0..leaves.len()).filter(|&at| read[at]).collect())
}

/// Every ON term of the run, with the leaves it reads; None when one reads
/// a name the run's leaves do not hold
fn run_terms<'s>(run: &'s JoinTableSource, leaves: &[Leaf<'_>]) -> Option<Vec<Term<'s>>> {
    let mut terms = Vec::new();
    let mut at = Some(run);
    while let Some(join) = at {
        for expr in extract_and_conditions(join.condition.as_deref()?) {
            terms.push(Term {
                expr,
                leaves: term_leaves(expr, leaves)?,
            });
        }
        at = match join.left.as_ref() {
            Expression::JoinSource(left) => Some(left),
            _ => None,
        };
    }
    Some(terms)
}

/// The equality `term` states between a column of each of two leaves
fn edge(term: &Term<'_>, leaves: &[Leaf<'_>]) -> Option<Edge> {
    let Expression::Infix(infix) = term.expr else {
        return None;
    };
    if !matches!(infix.op_type, InfixOperator::Equal) || term.leaves.len() != 2 {
        return None;
    }
    let end = |side: &Expression| -> Option<(usize, SmartString)> {
        let leaves_of = term_leaves(side, leaves)?;
        let name = match side {
            Expression::Identifier(id) => id.value_lower.clone(),
            Expression::QualifiedIdentifier(q) => q.name.value_lower.clone(),
            _ => return None,
        };
        Some((*leaves_of.first()?, name))
    };
    let ends = [end(&infix.left)?, end(&infix.right)?];
    (ends[0].0 != ends[1].0).then_some(Edge { ends })
}

/// The cheapest step joining `next` to the leaves `placed` holds, with
/// `rows` rows so far: its estimated cost and the rows it gives; None when
/// no equality reaches `next`
fn step(
    placed: &[bool],
    rows: f64,
    next: usize,
    leaves: &[Leaf<'_>],
    edges: &[Edge],
    columns: &dyn Fn(usize, &str) -> Column,
) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for edge in edges {
        let [a, b] = &edge.ends;
        let ((from, from_column), (to, to_column)) = match (a.0 == next, b.0 == next) {
            (true, false) if placed[b.0] => (b, a),
            (false, true) if placed[a.0] => (a, b),
            _ => continue,
        };
        // The rows so far may repeat a key their table holds once
        let ends = [
            (*from, from_column.as_str(), false),
            (*to, to_column.as_str(), true),
        ];
        let out = edge_rows(ends, [rows, leaves[next].kept], leaves, columns);
        // A probe reads a key per row so far; otherwise the table is read whole
        let read = if columns(*to, to_column).probed {
            rows
        } else {
            leaves[next].total
        };
        let cost = read + out;
        if best.is_none_or(|(least, _)| cost < least) {
            best = Some((cost, out));
        }
    }
    best
}

/// The rows an equality between `ends`, each a leaf, its column and whether
/// its key's uniqueness bounds the join, gives of `rows` rows a side, by
/// `QueryPlanner::join_rows`; a distinct count not known is its table's rows
fn edge_rows(
    ends: [(usize, &str, bool); 2],
    rows: [f64; 2],
    leaves: &[Leaf<'_>],
    columns: &dyn Fn(usize, &str) -> Column,
) -> f64 {
    let facts = ends.map(|(leaf, column, _)| columns(leaf, column));
    let distinct = [0, 1].map(|end| facts[end].distinct.unwrap_or(leaves[ends[end].0].total));
    let unique = [0, 1].map(|end| ends[end].2 && facts[end].unique);
    QueryPlanner::join_rows(rows, distinct, unique)
}

/// The estimated cost of joining `order`'s leaves in that order: the rows
/// each step reads and gives; None when a step has no equality to join by
fn order_cost(
    order: &[usize],
    leaves: &[Leaf<'_>],
    edges: &[Edge],
    columns: &dyn Fn(usize, &str) -> Column,
) -> Option<f64> {
    let mut placed = vec![false; leaves.len()];
    let first = *order.first()?;
    placed[first] = true;
    let (mut cost, mut rows) = (leaves[first].total, leaves[first].kept);
    for &next in &order[1..] {
        let (step_cost, out) = step(&placed, rows, next, leaves, edges, columns)?;
        cost += step_cost;
        rows = out;
        placed[next] = true;
    }
    Some(cost)
}

/// The order from the leaf estimated to keep the fewest rows, each next one
/// the cheapest step reachable; ties keep the written order
fn greedy_order(
    leaves: &[Leaf<'_>],
    edges: &[Edge],
    columns: &dyn Fn(usize, &str) -> Column,
) -> Option<Vec<usize>> {
    let first = (0..leaves.len()).min_by(|&a, &b| leaves[a].kept.total_cmp(&leaves[b].kept))?;
    let mut order = vec![first];
    let mut placed = vec![false; leaves.len()];
    placed[first] = true;
    let mut rows = leaves[first].kept;
    while order.len() < leaves.len() {
        let (next, (_, out)) = (0..leaves.len())
            .filter(|&leaf| !placed[leaf])
            .filter_map(|leaf| Some((leaf, step(&placed, rows, leaf, leaves, edges, columns)?)))
            .min_by(|(_, (a, _)), (_, (b, _))| a.total_cmp(b))?;
        order.push(next);
        placed[next] = true;
        rows = out;
    }
    Some(order)
}

/// The run's joins in `order`, each ON holding the terms whose last leaf it
/// joins; None when a join would have no term
fn rebuild(
    run: &JoinTableSource,
    order: &[usize],
    leaves: &[Leaf<'_>],
    terms: &[Term<'_>],
) -> Option<Expression> {
    let position = |leaf: usize| order.iter().position(|&at| at == leaf);
    let mut on: Vec<Vec<Expression>> = vec![Vec::new(); order.len()];
    for term in terms {
        let last = term
            .leaves
            .iter()
            .map(|&leaf| position(leaf))
            .max()
            .flatten();
        on[last.unwrap_or(0).max(1)].push(term.expr.clone());
    }
    let leaf = |at: usize| Expression::TableSource(Box::new(leaves[order[at]].source.clone()));
    let mut joined = leaf(0);
    for (at, terms) in on.into_iter().enumerate().skip(1) {
        joined = Expression::JoinSource(Box::new(JoinTableSource {
            token: run.token.clone(),
            left: Box::new(joined),
            join_type: SmartString::const_new("INNER"),
            right: Box::new(leaf(at)),
            condition: Some(Box::new(combine_predicates_with_and(terms)?)),
            using_columns: Vec::new(),
        }));
    }
    Some(joined)
}

/// `tree` with the join `run` in it replaced by `joined`
fn replace_run(tree: &Expression, run: &JoinTableSource, joined: Expression) -> Expression {
    match tree {
        Expression::JoinSource(join) if std::ptr::eq(&**join, run) => joined,
        Expression::JoinSource(join) => {
            let mut join = join.clone();
            join.left = Box::new(replace_run(&join.left, run, joined));
            Expression::JoinSource(join)
        }
        other => other.clone(),
    }
}
