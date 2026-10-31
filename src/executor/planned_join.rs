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

//! The one place a join's way of running is decided: `plan_join` over the
//! statement, its schemas and its indexes, and the arms a run picks from

use std::sync::Arc;

use crate::core::Result;
use crate::parser::ast::{Expression, InfixOperator, JoinTableSource, SelectStatement};
use crate::planner::join::{IndexProbe, JoinPlan, ProbeIndex};
use crate::storage::traits::transaction::{StatementTable, TableUse};
use crate::storage::traits::Engine;

use super::operators::index_nested_loop::IndexLookupStrategy;
use super::query::{CteNames, HOT_INNER_ROWS_PER_KEY, SMALL_OUTER_ROWS};
use super::query_classification::QueryClassification;
use super::utils::{bare_name, probes_by_equality, qualified, split_qualified};
use super::Executor;

/// What a join's plan is decided from: the statement, and its WHERE as the
/// join split it over the sides
pub(crate) struct JoinShape<'a> {
    pub stmt: &'a SelectStatement,
    pub classification: &'a QueryClassification,
    pub join: &'a JoinTableSource,
    /// The ON as the join runs it, with what the blocks moved into it
    pub condition: Option<&'a Expression>,
    pub join_type: &'a str,
    pub aliases: [Option<&'a str>; 2],
    /// Each side's conditions, left then right
    pub filters: [Option<&'a Expression>; 2],
    pub cross_filter: Option<&'a Expression>,
    /// Which sides the statement's blocks read as rows
    pub rows_sides: [bool; 2],
    /// A correlated WHERE run per joined row, outside the blocks
    pub per_row_where: bool,
    /// Whether the statement's blocks run its subqueries
    pub blocks: bool,
    pub ctes: CteNames<'a>,
}

impl Executor {
    /// The plan of the join `shape` describes
    pub(crate) fn plan_join(&self, shape: &JoinShape<'_>) -> Result<JoinPlan> {
        let condition = shape.join.condition.as_deref();
        let reduction = match shape.per_row_where {
            true => None,
            false => self.reduction_keys(shape),
        };
        let probes = match shape.per_row_where {
            true => [None, None],
            false => self.join_probes(
                shape.ctes,
                [&shape.join.left, &shape.join.right],
                shape.condition,
                shape.join_type,
                shape.aliases,
                shape.rows_sides,
            ),
        };
        let filters_have_subquery = !shape.blocks
            && [shape.filters[0], shape.filters[1], shape.cross_filter]
                .into_iter()
                .flatten()
                .any(Self::filter_has_subquery);
        let grouped_index = reduction.is_some()
            && probes[0].is_some()
            && self
                .grouped_index_join_applies(
                    shape.stmt,
                    shape.join_type,
                    condition,
                    shape.filters[1].is_some(),
                    filters_have_subquery,
                    None,
                )?
                .is_some();
        Ok(JoinPlan {
            reduction,
            grouped_index,
            probes,
            limit_hash: Self::limit_hash(shape),
        })
    }

    /// The probe each of the `sides` of a join on `condition` offers as the
    /// inner side of an index join, the right side's first; none for a side
    /// read as rows
    pub(crate) fn join_probes(
        &self,
        ctes: CteNames<'_>,
        [left, right]: [&Expression; 2],
        condition: Option<&Expression>,
        join_type: &str,
        [left_alias, right_alias]: [Option<&str>; 2],
        rows_sides: [bool; 2],
    ) -> [Option<IndexProbe>; 2] {
        let probe = |inner: &Expression, aliases, rows: bool| match rows {
            true => None,
            false => self.index_probe(ctes, inner, condition, join_type, aliases),
        };
        [
            probe(right, [left_alias, right_alias], rows_sides[1]),
            probe(left, [right_alias, left_alias], rows_sides[0]),
        ]
    }

    /// The key columns, left then right, of a GROUP BY + LIMIT join whose
    /// left side can be fetched in a chunk and the right side narrowed by
    /// its keys: an INNER or LEFT join, a plain GROUP BY over the left
    /// side's columns covering its primary key, a LIMIT without ORDER BY,
    /// window or DISTINCT, and a simple equality key
    fn reduction_keys(&self, shape: &JoinShape<'_>) -> Option<(String, String)> {
        let (stmt, join_type) = (shape.stmt, shape.join_type);
        let is_inner = join_type == "INNER";
        let is_left = join_type.contains("LEFT") && !join_type.contains("FULL");
        if !is_inner && !is_left {
            return None;
        }
        // ROLLUP, CUBE and GROUPING SETS add groups of many left rows, which
        // a cut left side would count short
        if stmt.group_by.columns.is_empty()
            || stmt.group_by.modifier != crate::parser::ast::GroupByModifier::None
        {
            return None;
        }
        // Without ORDER BY the groups come in no order, so any LIMIT of
        // them is an answer
        if stmt.limit.is_none() || !stmt.order_by.is_empty() {
            return None;
        }
        // A window function or DISTINCT runs on the grouped rows afterwards and
        // needs every group, not a chunk that already meets the limit
        if shape.classification.has_window_functions
            || stmt.distinct
            || !stmt.distinct_on.is_empty()
        {
            return None;
        }
        // Each group must come from one left row, or a partial chunk would
        // give partial aggregates: GROUP BY has to cover the left primary key
        let left_alias = shape.aliases[0]?;
        let left_table = match shape.join.left.as_ref() {
            Expression::TableSource(ts) => ts.name.value.as_str(),
            _ => return None,
        };
        let left_schema = self.engine.get_table_schema(left_table).ok()?;
        let pk_columns = left_schema.primary_key_columns();
        let pk_covered = !pk_columns.is_empty()
            && pk_columns.iter().all(|pk| {
                stmt.group_by.columns.iter().any(|col| match col {
                    Expression::QualifiedIdentifier(qid) => {
                        qid.qualifier.value.eq_ignore_ascii_case(left_alias)
                            && qid.name.value.eq_ignore_ascii_case(&pk.name)
                    }
                    Expression::Identifier(id) => id.value.eq_ignore_ascii_case(&pk.name),
                    _ => false,
                })
            });
        if !pk_covered {
            return None;
        }
        // An unqualified GROUP BY column could be either side's; it is
        // taken as the left side's
        let of_left = |col: &Expression| match col {
            Expression::QualifiedIdentifier(qid) => {
                qid.qualifier.value.eq_ignore_ascii_case(left_alias)
            }
            Expression::Identifier(_) => true,
            _ => false,
        };
        if !stmt.group_by.columns.iter().all(of_left) {
            return None;
        }
        let (left_key, right_key) = simple_join_key(shape.join.condition.as_deref()?)?;
        let left_alias_lower = left_alias.to_lowercase();
        let qualified_by_left = |key: &str| {
            split_qualified(&key.to_lowercase()).is_some_and(|(q, _)| q == left_alias_lower)
        };
        if qualified_by_left(&left_key) {
            Some((left_key, right_key))
        } else if qualified_by_left(&right_key) {
            Some((right_key, left_key))
        } else {
            None
        }
    }

    /// How an index join finds the rows of `inner_expr` for a key: its
    /// primary key, an index on the key column, or, when neither has it,
    /// the table's candidates for the key, which a run takes only where
    /// they pay. `aliases` are the outer side's and the inner side's. Only
    /// an INNER or LEFT join probes, with an equality between the sides
    fn index_probe(
        &self,
        ctes: CteNames<'_>,
        inner_expr: &Expression,
        condition: Option<&Expression>,
        join_type: &str,
        [outer_alias, inner_alias]: [Option<&str>; 2],
    ) -> Option<IndexProbe> {
        // A RIGHT join would track every unmatched inner row
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            return None;
        }
        // A table, or a subquery that passes one through under its alias
        let (table_name, effective_alias) = match inner_expr {
            Expression::TableSource(_) => (index_join_table_name(inner_expr)?, None),
            Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                Expression::TableSource(_) => (index_join_table_name(inner_expr)?, None),
                Expression::SubquerySource(sq) => (
                    self.passthrough_table(&sq.subquery)?,
                    sq.alias.as_ref().map(|a| a.value.to_string()),
                ),
                _ => return None,
            },
            Expression::SubquerySource(sq) => (
                self.passthrough_table(&sq.subquery)?,
                sq.alias.as_ref().map(|a| a.value.to_string()),
            ),
            _ => return None,
        };
        // A CTE of the same name hides the table and has no index
        if ctes.hides(&table_name) {
            return None;
        }
        let (left_col, right_col) = simple_join_key(condition?)?;
        let inner_alias_lower = effective_alias
            .as_deref()
            .or(inner_alias)
            .unwrap_or(&table_name)
            .to_lowercase();
        let qualified_by = |col: &str, alias: &str| {
            split_qualified(&col.to_lowercase()).is_some_and(|(qualifier, _)| qualifier == alias)
        };
        let (inner_col, outer_col) = if qualified_by(&left_col, &inner_alias_lower) {
            (left_col, right_col)
        } else if qualified_by(&right_col, &inner_alias_lower) {
            (right_col, left_col)
        } else {
            return None;
        };
        // An equality between two columns of the inner table joins nothing
        if qualified_by(&outer_col, &inner_alias_lower) {
            return None;
        }
        if outer_alias.is_some_and(|alias| !qualified_by(&outer_col, &alias.to_lowercase())) {
            return None;
        }
        let inner_col = bare_name(&inner_col).to_string();
        let txn = self.begin_transaction().ok()?;
        let table = txn
            .get_read_table_at(&table_name, None, TableUse::Read)
            .ok()?;
        let schema = table.schema();
        let is_pk = schema
            .pk_column_index()
            .is_some_and(|pk| schema.columns[pk].name_lower == inner_col.to_lowercase());
        let index = if is_pk {
            ProbeIndex::PrimaryKey
        } else if let Some(index) = table.lookup_index_on_column(&inner_col) {
            ProbeIndex::Secondary(index)
        } else {
            // A narrowing key must be the outer side's; an inner column's
            // name is unsure
            let outer_is_sure = outer_alias.is_some()
                || split_qualified(&outer_col).is_some()
                || !schema
                    .columns
                    .iter()
                    .any(|c| c.name_lower.eq_ignore_ascii_case(&outer_col));
            ProbeIndex::Equality { outer_is_sure }
        };
        Some(IndexProbe {
            table: table_name,
            index,
            inner_col,
            outer_col,
        })
    }

    /// The lookup `probe` runs with, when this run takes it. The table's
    /// candidates answer a probe while they can, deciding per probe: a join
    /// bounded by a limit asks them of a column a probe reads, and a small
    /// outer narrows a large table, a sealed one through its side file, or
    /// the scan of a column no probe reads, from an outer side whose size
    /// only running it tells
    pub(crate) fn probe_strategy(
        &self,
        probe: &IndexProbe,
        bounded: bool,
        small_outer: bool,
        outer_materialized: bool,
    ) -> Option<(String, IndexLookupStrategy, String, String)> {
        let strategy = match &probe.index {
            ProbeIndex::PrimaryKey => IndexLookupStrategy::PrimaryKey,
            ProbeIndex::Secondary(index) => IndexLookupStrategy::SecondaryIndex(Arc::clone(index)),
            ProbeIndex::Equality { outer_is_sure } => {
                let txn = self.begin_transaction().ok()?;
                let table = txn
                    .get_read_table_at(&probe.table, None, TableUse::Read)
                    .ok()?;
                let large = table.row_count_hint()
                    >= SMALL_OUTER_ROWS.saturating_mul(HOT_INNER_ROWS_PER_KEY);
                let probes = probes_by_equality(&*table, &probe.inner_col);
                let narrows =
                    large || (probes && table.has_sealed_rows()) || (outer_materialized && !probes);
                if !((bounded && probes) || (small_outer && *outer_is_sure && narrows)) {
                    return None;
                }
                return Some((
                    probe.table.clone(),
                    IndexLookupStrategy::TableEquality {
                        column: probe.inner_col.clone(),
                    },
                    probe.inner_col.clone(),
                    probe.outer_col.clone(),
                ));
            }
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::join_index_chosen();
        Some((
            probe.table.clone(),
            strategy,
            probe.inner_col.clone(),
            probe.outer_col.clone(),
        ))
    }

    /// Whether a small LIMIT streams the join through a hash table: an
    /// INNER join with an equality, nothing that orders, groups, aggregates
    /// or merges the joined rows, and no condition across sides
    fn limit_hash(shape: &JoinShape<'_>) -> bool {
        let (stmt, class, join_type) = (shape.stmt, shape.classification, shape.join_type);
        !join_type.contains("FULL")
            && !join_type.contains("LEFT")
            && !join_type.contains("RIGHT")
            && !class.has_order_by
            && !class.has_group_by
            && !class.has_aggregation
            && !class.has_window_functions
            && !stmt.distinct
            && stmt.distinct_on.is_empty()
            && shape.cross_filter.is_none()
            && shape.join.condition.as_deref().is_some_and(has_equality)
    }

    /// Whether a LIMIT hash join builds on its left side: the side the
    /// statistics estimate smaller under its conditions, read per run since
    /// ANALYZE moves the estimates without changing the plan
    pub(crate) fn limit_hash_builds_left(
        &self,
        [left, right]: [&Expression; 2],
        [left_filter, right_filter]: [Option<&Expression>; 2],
    ) -> bool {
        self.estimate_side_rows(left, left_filter) < self.estimate_side_rows(right, right_filter)
    }

    /// The rows a base table side is estimated to give under `filter`;
    /// u64::MAX for any other side
    fn estimate_side_rows(&self, side: &Expression, filter: Option<&Expression>) -> u64 {
        let table = match side {
            Expression::TableSource(ts) => ts.name.value_lower.as_str(),
            Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                Expression::TableSource(ts) => ts.name.value_lower.as_str(),
                _ => return u64::MAX,
            },
            _ => return u64::MAX,
        };
        self.get_query_planner()
            .estimate_scan_rows(table, filter)
            .unwrap_or(u64::MAX)
    }

    /// The table a subquery passes through: one table, every column in its
    /// order under its own name, and no clause that filters, orders, groups,
    /// limits or merges, since an index join reads the table itself
    fn passthrough_table(&self, stmt: &SelectStatement) -> Option<String> {
        if stmt.with.is_some()
            || !stmt.set_operations.is_empty()
            || !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
            || !stmt.order_by.is_empty()
            || stmt.limit.is_some()
            || stmt.offset.is_some()
            || stmt.where_clause.is_some()
            || stmt.distinct
            || !stmt.window_defs.is_empty()
        {
            return None;
        }
        let table = index_join_table_name(stmt.table_expr.as_ref()?)?;
        if !matches!(
            stmt.columns.as_slice(),
            [Expression::Star(_)] | [Expression::QualifiedStar(_)]
        ) {
            let schema = self.engine.get_table_schema(&table).ok()?;
            // The output name as written, since the probe names the column
            // as the schema spells it
            let own_name = |column: &Expression| match column {
                Expression::Identifier(id) => Some(id.value.to_string()),
                Expression::QualifiedIdentifier(q) => Some(q.name.value.to_string()),
                Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                    Expression::Identifier(id) if id.value_lower == aliased.alias.value_lower => {
                        Some(aliased.alias.value.to_string())
                    }
                    Expression::QualifiedIdentifier(q)
                        if q.name.value_lower == aliased.alias.value_lower =>
                    {
                        Some(aliased.alias.value.to_string())
                    }
                    _ => None,
                },
                _ => None,
            };
            let same = stmt.columns.len() == schema.columns.len()
                && stmt
                    .columns
                    .iter()
                    .zip(&schema.columns)
                    .all(|(column, table_column)| {
                        own_name(column).is_some_and(|name| name == table_column.name)
                    });
            if !same {
                return None;
            }
        }
        Some(table)
    }
}

/// The first equality between two columns in `condition`'s AND chain,
/// each as the ON names it
fn simple_join_key(condition: &Expression) -> Option<(String, String)> {
    match condition {
        Expression::Infix(infix) if infix.op_type == InfixOperator::Equal => {
            Some((column_name(&infix.left)?, column_name(&infix.right)?))
        }
        Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
            simple_join_key(&infix.left).or_else(|| simple_join_key(&infix.right))
        }
        _ => None,
    }
}

/// A column as the ON names it, qualified when it is
fn column_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::QualifiedIdentifier(qid) => {
            Some(qualified(&qid.qualifier.value, &qid.name.value))
        }
        Expression::Identifier(id) => Some(id.value.to_string()),
        _ => None,
    }
}

/// Whether `expr` holds an `=` at its top or in its AND and OR branches
fn has_equality(expr: &Expression) -> bool {
    match expr {
        Expression::Infix(infix) => {
            let op = infix.operator.as_str();
            if op == "=" {
                return true;
            }
            if op.eq_ignore_ascii_case("AND") || op.eq_ignore_ascii_case("OR") {
                return has_equality(&infix.left) || has_equality(&infix.right);
            }
            false
        }
        _ => false,
    }
}

/// The table an index join can probe on a side that names one: never a table
/// read AS OF, whose past rows no index holds
fn index_join_table_name(side: &Expression) -> Option<String> {
    let table = match side {
        Expression::TableSource(ts) => ts,
        Expression::Aliased(aliased) => match aliased.expression.as_ref() {
            Expression::TableSource(ts) => ts,
            _ => return None,
        },
        _ => return None,
    };
    table
        .as_of
        .is_none()
        .then(|| table.name.value_lower.to_string())
}

/// The side an index join probes, given what the right table offers
/// (`unswapped`): an INNER join probes its left table instead from a right
/// side that is not a base table (a subquery, a CTE, a view), or for a
/// primary key when only the right side is filtered. Some((probe, true))
/// when the sides swap
pub(crate) fn pick_index_join_side<T>(
    join_type: &str,
    right_is_table: bool,
    right_filtered: bool,
    left_filtered: bool,
    unswapped: Option<T>,
    probe_left: impl FnOnce() -> Option<T>,
    is_primary_key: impl Fn(&T) -> bool,
) -> Option<(T, bool)> {
    // Only an INNER join reads the same with its sides exchanged; a LEFT
    // join would go on preserving the side it was given
    if join_type != "INNER" {
        return unswapped.map(|probe| (probe, false));
    }
    if unswapped.is_none() && !right_is_table {
        return probe_left().map(|probe| (probe, true));
    }
    if right_filtered && !left_filtered {
        if let Some(probe) = probe_left().filter(&is_primary_key) {
            return Some((probe, true));
        }
    }
    unswapped.map(|probe| (probe, false))
}
