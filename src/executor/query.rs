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

//! SELECT Query Execution
//!
//! This module implements SELECT query execution including:
//! - Simple table scans
//! - WHERE clause filtering
//! - Column projection
//! - ORDER BY sorting
//! - LIMIT/OFFSET
//! - DISTINCT
//! - Aggregate functions and GROUP BY
//! - JOIN operations

use std::cmp::Ordering;
use std::sync::Arc;

use crate::common::SmartString;
use crate::storage::traits::transaction::{StatementTable, TableUse};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::common::{CompactArc, CompactVec, StringMap};
use crate::core::predicate::{kind, sql_compare, sql_eq, KeySide};
use crate::core::{DataType, Error, Result, Row, RowVec, Schema, Value, ValueSet};
use crate::optimizer::ExpressionSimplifier;
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::mvcc::engine::ViewDefinition;
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::{Engine, QueryResult, ScanAccess};

/// Maximum depth for nested views to prevent stack overflow
const MAX_VIEW_DEPTH: usize = 32;

/// Threshold below which streaming NOT EXISTS with early termination outperforms
/// bulk anti-join materialization. Queries with LIMIT below this value use the
/// streaming InHashSet path which can terminate early, while queries without LIMIT
/// (or with larger LIMIT) use bulk anti-join which is faster for full scans.
/// Based on benchmarking where streaming wins when result set is small enough
/// to benefit from early termination.
const ANTI_JOIN_LIMIT_THRESHOLD: i64 = 10000;

/// Deferred projection info: (column_indices, output_column_names)
/// Used when projection can be deferred until after ORDER BY + LIMIT
type DeferredProjection = (Vec<usize>, Vec<String>);

/// What one output column of a SELECT holds
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColumnSource {
    /// The SELECT item that gives it; None for an input column appended
    /// after the list for ORDER BY or DISTINCT ON to read
    pub(crate) item: Option<usize>,
    /// The input column it is a copy of, when it is one
    pub(crate) input: Option<usize>,
}

/// Where each output column of a SELECT comes from, over the input columns
/// it was projected from: the one record an ORDER BY name is bound by
#[derive(Clone, Debug, Default)]
pub(crate) struct OutputLayout {
    /// Shared with the source that names them, where it holds them shared
    pub(crate) input_columns: CompactArc<Vec<String>>,
    /// Held inline up to eight columns, so a short layout costs no allocation
    pub(crate) sources: LayoutSources,
}

pub(crate) type LayoutSources = smallvec::SmallVec<[ColumnSource; 8]>;

impl OutputLayout {
    /// The layout of rows that are the input `columns` themselves, as a
    /// projection deferred past the sort leaves them
    pub(crate) fn of_input(columns: &[String]) -> Self {
        Self::of_shared_input(&CompactArc::new(columns.to_vec()))
    }

    /// `of_input` over input columns a source already holds shared
    pub(crate) fn of_shared_input(columns: &CompactArc<Vec<String>>) -> Self {
        Self {
            input_columns: CompactArc::clone(columns),
            sources: (0..columns.len())
                .map(|idx| ColumnSource {
                    item: None,
                    input: Some(idx),
                })
                .collect(),
        }
    }

    /// How many columns the first `items` SELECT items show, before those
    /// appended for the sort keys, a hidden key item among them
    fn shown(&self, items: usize) -> usize {
        self.sources
            .iter()
            .take_while(|source| source.item.is_some_and(|item| item < items))
            .count()
    }

    /// The output column of SELECT item `item`, its first under a star
    fn item_position(&self, item: usize) -> Option<usize> {
        self.sources.iter().position(|s| s.item == Some(item))
    }

    /// The output position of what `key` reads: the column its SELECT item
    /// gives, or a column holding its input column unchanged
    fn position(&self, key: &OrderKey) -> Option<usize> {
        Self::position_over(&self.sources, &self.input_columns, key)
    }

    /// [`Self::position`] of a layout of `sources` over `input_columns`
    fn position_over(
        sources: &[ColumnSource],
        input_columns: &[String],
        key: &OrderKey,
    ) -> Option<usize> {
        match *key {
            OrderKey::Item(item) => sources.iter().position(|s| s.item == Some(item)),
            OrderKey::Input(qualifier, name) => {
                let input = input_column_index(input_columns, qualifier, name)?;
                sources.iter().position(|s| s.input == Some(input))
            }
            OrderKey::Ordinal(_) | OrderKey::Expression => None,
        }
    }

    /// Each name `stmt`'s sort keys read, into `slots` at the column this
    /// layout gives it; an error for a qualified name no input column
    /// answers, which no output name stands in for
    fn sort_slots(
        &self,
        stmt: &SelectStatement,
        slots: &mut FxHashMap<ExprKey, usize>,
    ) -> Result<()> {
        let mut missing = None;
        visit_order_by_inputs(stmt, &mut |column, key| {
            if let Some(position) = self.position(&key) {
                slots.entry(ExprKey::of(column)).or_insert(position);
            } else if let OrderKey::Input(Some(qualifier), name) = key {
                if input_column_index(&self.input_columns, Some(qualifier), name).is_none() {
                    missing.get_or_insert_with(|| format!("{qualifier}.{name}"));
                }
            }
        });
        match missing {
            Some(name) => Err(super::aggregation::ungrouped_column(&name)),
            None => Ok(()),
        }
    }
}

/// What one ORDER BY item sorts by, bound once by one rule: a bare name a
/// SELECT alias over anything but that column carries sorts by the alias's
/// column, any other plain name by its input column, an ordinal by the
/// output column at that position
enum OrderKey<'e> {
    Ordinal(usize),
    Item(usize),
    Input(Option<&'e str>, &'e str),
    /// Anything else: read from a column that computes it, or evaluated
    Expression,
}

/// The binding of ORDER BY `expr` of `stmt`
fn bind_order_key<'e>(expr: &'e Expression, stmt: &SelectStatement) -> OrderKey<'e> {
    match expr {
        Expression::IntegerLiteral(lit) => {
            OrderKey::Ordinal((lit.value as usize).saturating_sub(1))
        }
        Expression::Identifier(id) if order_by_names_an_alias(expr, stmt) => stmt
            .columns
            .iter()
            .position(|item| {
                matches!(item, Expression::Aliased(aliased) if aliased.alias.value_lower == id.value_lower)
            })
            .map_or(OrderKey::Expression, OrderKey::Item),
        Expression::Identifier(id) => OrderKey::Input(None, &id.value_lower),
        Expression::QualifiedIdentifier(qid) if reads_input(stmt.table_expr.as_deref(), qid) => {
            OrderKey::Input(Some(&qid.qualifier.value_lower), &qid.name.value_lower)
        }
        _ => OrderKey::Expression,
    }
}

/// Whether ORDER BY `expr` of `stmt` is read from a column computing it,
/// not by a name or a position
pub(crate) fn order_key_is_expression(expr: &Expression, stmt: &SelectStatement) -> bool {
    matches!(bind_order_key(expr, stmt), OrderKey::Expression)
}

/// Whether a qualified name reads the input of a query whose FROM is `from`:
/// one whose qualifier no entry shows is an enclosing query's column
fn reads_input(from: Option<&Expression>, qid: &QualifiedIdentifier) -> bool {
    from.is_some_and(|from| crate::parser::scope::from_shows(from, &qid.qualifier.value_lower))
}

/// `stmt`'s sort keys: its ORDER BY keys, then its DISTINCT ON keys, which
/// are bound, carried and read as ORDER BY keys are. A set operation's
/// branch carries no DISTINCT ON key: one reads its first branch's items
fn sort_keys(stmt: &SelectStatement) -> impl Iterator<Item = &Expression> {
    let carried = stmt.set_operations.is_empty();
    stmt.order_by
        .iter()
        .map(|ob| &ob.expression)
        .chain(stmt.distinct_on.iter().filter(move |_| carried))
}

/// Each input column `stmt`'s sort keys read by their binding, with the
/// reference that names it: a plain name bound to its input column, and
/// every name inside an expression; false when an expression reads more
/// than its names say, as a subquery does
fn visit_order_by_inputs<'e>(
    stmt: &'e SelectStatement,
    visit: &mut dyn FnMut(&'e Expression, OrderKey<'e>),
) -> bool {
    let mut read_all = true;
    for expr in sort_keys(stmt) {
        match bind_order_key(expr, stmt) {
            key @ OrderKey::Input(..) => visit(expr, key),
            OrderKey::Expression => {
                read_all &= visit_expression_inputs(expr, stmt.table_expr.as_deref(), visit)
            }
            OrderKey::Ordinal(_) | OrderKey::Item(_) => {}
        }
    }
    read_all
}

/// Each name inside a sort expression of a query whose FROM is `from`,
/// bound as an input column; false as `visit_column_refs` says
fn visit_expression_inputs<'e>(
    expr: &'e Expression,
    from: Option<&Expression>,
    visit: &mut dyn FnMut(&'e Expression, OrderKey<'e>),
) -> bool {
    visit_column_refs(expr, &mut |column| match column {
        Expression::Identifier(id) => visit(column, OrderKey::Input(None, &id.value_lower)),
        Expression::QualifiedIdentifier(qid) if reads_input(from, qid) => visit(
            column,
            OrderKey::Input(Some(&qid.qualifier.value_lower), &qid.name.value_lower),
        ),
        _ => {}
    })
}

/// The input columns `stmt`'s sort keys read, as indexes into `columns`;
/// every one of them when its names do not say all it reads
pub(crate) fn order_by_inputs(stmt: &SelectStatement, columns: &[String]) -> Vec<usize> {
    let mut inputs = Vec::new();
    let read_all = visit_order_by_inputs(stmt, &mut |_, key| {
        if let OrderKey::Input(qualifier, name) = key {
            if let Some(input) = input_column_index(columns, qualifier, name) {
                if !inputs.contains(&input) {
                    inputs.push(input);
                }
            }
        }
    });
    if read_all {
        inputs
    } else {
        (0..columns.len()).collect()
    }
}

/// The input column a plain column reference reads, as the projection finds
/// it: a qualified name by [`by_qualifier`]; a bare name, then a qualified
/// column whose last part is the bare name. The names come lowercased
fn input_column_index(columns: &[String], qualifier: Option<&str>, name: &str) -> Option<usize> {
    let bare = |name: &str| {
        columns.iter().position(|c| eq_folded(c, name)).or_else(|| {
            columns
                .iter()
                .position(|c| split_qualified(c).is_some_and(|(_, last)| eq_folded(last, name)))
        })
    };
    let Some(qualifier) = qualifier else {
        return bare(name);
    };
    by_qualifier(
        columns
            .iter()
            .position(|c| eq_folded_qualified(c, qualifier, name)),
        qualifier,
        name,
        bare,
        |qualifier| {
            columns
                .iter()
                .any(|c| split_qualified(c).is_some_and(|(q, _)| eq_folded(q, qualifier)))
        },
        |index| split_qualified(&columns[index]).is_none(),
    )
}

/// `name` as `eq_folded` takes it: borrowed when ASCII, which it compares
/// ignoring case, or when lowercasing changes none of its characters
fn folded(name: &str) -> std::borrow::Cow<'_, str> {
    let changes = |c: char| {
        let mut lower = c.to_lowercase();
        lower.next() != Some(c) || lower.next().is_some()
    };
    if !name.is_ascii() && name.chars().any(changes) {
        std::borrow::Cow::Owned(name.to_lowercase())
    } else {
        std::borrow::Cow::Borrowed(name)
    }
}

/// What a SELECT's rows carry to its ORDER BY: the projection deferred past
/// ORDER BY and LIMIT, when it is, and the layout of the columns
#[derive(Default)]
pub(crate) struct SelectShape {
    deferred: Option<DeferredProjection>,
    layout: Option<LaidOut>,
    /// How many columns the statement shows, when a star leaves that unknown
    /// and the ORDER BY keys follow them
    visible: Option<usize>,
}

/// A layout the rows carry: built for this execution, or the one a
/// prepared plan holds for all of them. Boxing the built one would be the
/// allocation this type exists to avoid
#[allow(clippy::large_enum_variant)]
pub(crate) enum LaidOut {
    Built(OutputLayout),
    Shared(CompactArc<OutputLayout>),
}

impl std::ops::Deref for LaidOut {
    type Target = OutputLayout;

    fn deref(&self) -> &OutputLayout {
        match self {
            LaidOut::Built(layout) => layout,
            LaidOut::Shared(layout) => layout,
        }
    }
}

impl SelectShape {
    /// The shape of columns laid out as `layout`, with no deferred
    /// projection; none without a layout, as a statement with no ORDER BY
    fn laid_out(layout: Option<OutputLayout>) -> Option<Self> {
        layout.map(|layout| Self {
            deferred: None,
            layout: Some(LaidOut::Built(layout)),
            visible: None,
        })
    }
}

/// Type alias for select execution results: (result, column_names, limit_offset_applied, shape)
/// Using CompactArc<Vec<String>> for column names enables zero-copy sharing across query execution.
type SelectOutput = (
    Box<dyn QueryResult>,
    CompactArc<Vec<String>>,
    bool,
    Option<SelectShape>,
);
type SelectResult = Result<SelectOutput>;

/// An outer side fetched for a join: the result, its qualified columns,
/// whether the limit and offset were applied, and the layout of a table
type OuterFetch = (Box<dyn QueryResult>, Vec<String>, bool, Option<LayoutToken>);

/// The layout a table's rows were read in and the schema they were read with
type ReadSchema = (LayoutToken, CompactArc<Schema>);

/// How a join reads one of its sides: its conditions as the WHERE gives
/// them, and the statement's run with the side's number when the blocks
/// hold those conditions
#[derive(Clone, Copy)]
struct SideRead<'a> {
    filter: Option<&'a Expression>,
    run: Option<(StatementRun<'a>, usize)>,
    /// The join's plan of the side, when the side runs as a statement
    plan: Option<&'a NestedPlan>,
    /// A side that is a join keeps both columns of a pair it merges
    apart: bool,
}

/// A WHERE asked of joined rows, bound to what the statement's run holds
/// and bound again when a row reads a value it asks, so a value no row
/// reaches is never read
struct JoinedWhere<'r, 'p> {
    filter: RowFilter,
    bound: Option<(Bound, &'r super::prepared_select::PreparedRunner<'p>)>,
    ctx: &'r ExecutionContext,
}

impl<'r, 'p> JoinedWhere<'r, 'p> {
    fn new(
        program: SharedProgram,
        run: Option<&'r super::prepared_select::PreparedRunner<'p>>,
        ctx: &'r ExecutionContext,
    ) -> Self {
        let mut bound = run
            .filter(|_| program.has_subqueries())
            .map(|run| (run.bind(&program), run));
        let program = match bound.as_mut() {
            Some((bound, run)) => CompactArc::clone(run.current(bound)),
            None => program,
        };
        Self {
            filter: RowFilter::from_program(program).with_context(ctx),
            bound,
            ctx,
        }
    }

    fn matches(&mut self, row: &Row, blocks: Option<&dyn SubqueryExecutor>) -> Result<bool> {
        let passed = self.filter.matches_checked_with(row, blocks)?;
        if let Some((bound, run)) = self.bound.as_mut() {
            if run.rebound(bound) {
                let program = CompactArc::clone(run.current(bound));
                self.filter = RowFilter::from_program(program).with_context(self.ctx);
            }
        }
        Ok(passed)
    }
}

/// A semantic cache answer and its output columns
type CachedAnswer = (Box<dyn QueryResult>, CompactArc<Vec<String>>);

/// What a semantic cache lookup before a statement's view came to
pub(crate) enum EarlySemantic {
    /// The statement's whole answer
    Hit(Box<dyn QueryResult>),
    /// A miss, which the statement's scan takes over
    Miss(SemanticMiss),
    /// Not a statement the lookup can answer alone
    NotTried,
}

use super::context::{
    clear_statement_caches, ExecutionContext, SemanticMiss, StatementSnapshot, TimeoutGuard,
};
use super::expr_key::ExprKey;
use super::expression::SubqueryExecutor;
use super::expression::{
    compile_expression, CompileContext, CompiledEvaluator, ExecuteContext, ExprCompiler, ExprVM,
    ExpressionEval, JoinFilter, RowFilter, SharedProgram,
};
use super::hash_table::JoinHashTable;
use super::join_executor::{JoinExecutor, JoinPath, StatementRun, StreamingJoinRequest};
use super::operator::{Operator, QueryResultOperator};
use super::operators::hash_join::JoinType as OperatorJoinType;
use super::operators::index_nested_loop::{
    BatchIndexNestedLoopJoinOperator, IndexLookupStrategy, IndexNestedLoopJoinOperator,
};
use super::operators::BloomFilterOperator;
use super::parallel::{self, ParallelConfig};
use super::planned_aggregate::{bounds, AggregateRead};
use super::planned_join::{pick_index_join_side, JoinShape};
use super::prepared_join::{
    BuiltPrograms, JoinProgramKey, JoinSide, JoinSides, PreparedJoin, WhereSite,
};
use super::prepared_select::{
    ambiguous_column, join_names_ambiguous, joined_aliases, set_operation_key_column,
    set_operation_key_item, Bound, JoinBlocks, OutOfFamily, PreparedProjection,
};
use super::projection::{Concat, Projection};
use super::pushdown;
use super::query_cache::{CachedPlanRef, CompiledExecution, NestedPlan};
use super::query_classification::{get_classification, QueryClassification};
use super::result::{
    AliasedResult, DistinctOnResult, DistinctResult, ExecResult, ExecutorResult, FilteredResult,
    LimitedResult, OrderLimitPath, OrderedResult, ProjectedResult, ProjectionResult,
    RadixOrderSpec, ScannerResult, TopN, TopNResult,
};
use super::row_count::{row_counts, rows_needed, static_row_counts, RowCounts};
use super::utils::compute_join_projection;
use super::utils::{
    add_table_qualifier, bare_name, build_column_index_map, by_qualifier, collect_table_qualifiers,
    collect_unqualified_columns, column_kinds, combine_predicates_with_and, dummy_token, eq_folded,
    eq_folded_qualified, expression_contains_aggregate, extract_and_conditions,
    extract_base_column_name, extract_join_keys_and_residual, filter_references_column,
    flatten_and_predicates, flip_operator, for_each_qualified_star_column,
    get_table_alias_from_expr, index_join_residual, infix_to_operator, item_name, named_as_written,
    order_by_names_an_alias, qualified, qualified_index, shown_name, split_qualified,
    star_takes_every_column, string_literal_value, strip_table_qualifier, substitute_filter_column,
    visit_column_refs,
};
use super::volatility::{expression_volatility, statement_constant};
use super::window::{WindowOutput, WindowPreGroupedState, WindowPreSortedState, WindowRows};
use super::Executor;
use crate::functions::{global_registry, FunctionRegistry, Volatility};
use crate::optimizer::bloom::BloomFilterBuilder;
use crate::planner::distinct_values_column;
use crate::planner::join::IndexProbe;
use crate::storage::volume::side_index::{resolve_known, Domain, Resolved};

/// The layout an ORDER BY or DISTINCT ON projection produces
pub(super) struct OrderByProjection<'e> {
    pub(super) col_index_map_lower: StringMap<usize>,
    /// The column each SELECT expression reads, when it is a plain column
    pub(super) select_column_indices: Vec<Option<usize>>,
    pub(super) extra_order_indices: Vec<usize>,
    pub(super) computed_order_by: Vec<&'e Expression>,
    pub(super) computed_distinct_on: Vec<&'e Expression>,
    pub(super) extra_names: Vec<String>,
    /// The source of each appended column, in the order of `extra_names`
    pub(super) extra_sources: Vec<ColumnSource>,
}

/// Pre-computed column name mappings for correlated subqueries.
/// Uses CompactArc<str> for zero-cost cloning in the per-row inner loop.
struct ColumnKeyMapping {
    /// Column index in the row
    index: usize,
    /// Lowercase column name (e.g., "id")
    col_lower: CompactArc<str>,
    /// Qualified name with table alias (e.g., "c.id")
    qualified_name: Option<CompactArc<str>>,
    /// Unqualified part if original had a dot (e.g., "id" from "table.id")
    unqualified_part: Option<CompactArc<str>>,
}

impl ColumnKeyMapping {
    /// Build column key mappings from column names and optional table alias.
    /// This pre-computes all the string transformations needed for correlated subquery
    /// outer row context, avoiding per-row allocations.
    fn build_mappings(columns: &[String], table_alias: Option<&str>) -> Vec<ColumnKeyMapping> {
        columns
            .iter()
            .enumerate()
            .map(|(i, col_name)| {
                let col_lower: CompactArc<str> = CompactArc::from(col_name.to_lowercase().as_str());
                let unqualified_part = split_qualified(col_name)
                    .map(|(_, name)| CompactArc::from(name.to_lowercase().as_str()));
                let qualified_name = table_alias
                    .filter(|_| unqualified_part.is_none())
                    .map(|alias| CompactArc::from(qualified(alias, &col_lower).as_str()));
                ColumnKeyMapping {
                    index: i,
                    col_lower,
                    qualified_name,
                    unqualified_part,
                }
            })
            .collect()
    }
}

/// A later outer chunk continues the first only as a bounded fetch of the
/// same source in the same layout: a table function has no layout and
/// continues as itself, a table moved to another layout or name does not
fn same_outer_source(
    outer: &Expression,
    first: Option<LayoutToken>,
    now: Option<LayoutToken>,
    bounded: bool,
) -> Result<()> {
    if bounded && now == first {
        return Ok(());
    }
    Err(Error::SchemaChanged {
        table: super::explain::extract_table_name(outer).unwrap_or_default(),
    })
}

/// The next left chunk of a GROUP BY + LIMIT join after `fetched` rows gave
/// `found` of the `want` groups: twice what the rate so far needs, from 32
/// rows up to three times those fetched
fn next_left_chunk(fetched: usize, found: usize, want: usize) -> usize {
    let missing = want.saturating_sub(found);
    let estimate = match found {
        0 => usize::MAX,
        _ => ((missing as f64 * fetched as f64 / found as f64) * 2.0).ceil() as usize,
    };
    estimate.clamp(32, fetched.saturating_mul(3).max(32))
}

/// The CTE names that hide tables of the same name for a join's sides: the
/// ones execution registered, or the ones EXPLAIN reads off the WITH
#[derive(Clone, Copy)]
pub(crate) enum CteNames<'a> {
    Context(&'a ExecutionContext),
    Listed(&'a [String]),
}

impl CteNames<'_> {
    pub(super) fn hides(self, name_lower: &str) -> bool {
        match self {
            CteNames::Context(ctx) => ctx.get_cte_by_lower(name_lower).is_some(),
            CteNames::Listed(names) => names.iter().any(|name| name == name_lower),
        }
    }
}

/// The joined rows of `all_columns`, each NATURAL or USING pair shown once:
/// `excluded` holds each pair's right column and `renames` its left one by
/// its bare name. A SELECT * drops the right column and names the left bare;
/// an explicit select list finds the left by its bare name too, as a column
/// of its own beside the qualified ones, so `a`, `x.a` and `x.*` all see it
fn merged_columns(
    stmt: &SelectStatement,
    all_columns: &[String],
    filtered_rows: RowVec,
    (excluded_column_indices, join_col_renames): (&[usize], &[(usize, String)]),
) -> (Vec<String>, RowVec) {
    if excluded_column_indices.is_empty() && join_col_renames.is_empty() {
        return (all_columns.to_vec(), filtered_rows);
    }
    // Check if SELECT * is used (need to deduplicate columns)
    let has_star = stmt
        .columns
        .iter()
        .any(|c| matches!(c, Expression::Star(_)));

    if has_star {
        // Create a set for O(1) lookup
        let excluded_set: FxHashSet<usize> = excluded_column_indices.iter().copied().collect();

        // Create rename map for join columns
        let rename_map: FxHashMap<usize, String> = join_col_renames.iter().cloned().collect();

        // Build list of indices to keep (all except excluded)
        let kept_indices: Vec<usize> = (0..all_columns.len())
            .filter(|i| !excluded_set.contains(i))
            .collect();

        // Filter columns and apply renames for join columns
        let filtered_columns: Vec<String> = kept_indices
            .iter()
            .map(|&i| {
                if let Some(base_name) = rename_map.get(&i) {
                    // Use unqualified name for join columns per SQL standard
                    base_name.clone()
                } else {
                    all_columns[i].clone()
                }
            })
            .collect();

        // Filter row values using clone_subset, preserving row IDs
        let filtered_rows: RowVec = filtered_rows
            .into_iter()
            .map(|(row_id, row)| (row_id, row.clone_subset(&kept_indices)))
            .collect();

        (filtered_columns, filtered_rows)
    } else {
        let mut columns = all_columns.to_vec();
        let mut rows = filtered_rows;
        for (idx, base_name) in join_col_renames {
            columns.push(base_name.clone());
            for (_, row) in rows.iter_mut() {
                let value = row.get(*idx).cloned().unwrap_or_else(Value::null_unknown);
                row.push(value);
            }
        }
        (columns, rows)
    }
}

/// `all_columns`, the names of joined rows holding each of the pairs
/// `merges` apart, with each pair's left column named bare, as the
/// merged column a bare name reads
fn merged_names(all_columns: &[String], merges: &[(usize, usize)]) -> Vec<String> {
    let mut named = all_columns.to_vec();
    for &(left, _) in merges {
        named[left] = bare_name(&all_columns[left]).to_string();
    }
    named
}

/// Into `excluded` and `renames`, as [`merged_columns`] takes them, the pairs
/// `held`'s joined sides merge, which the blocks read apart, over `columns`,
/// for `stmt`'s star or a qualified star; the left columns a clause names
/// bare otherwise, to name bare in place. A column already named is named once
fn side_pairs(
    stmt: &SelectStatement,
    held: &JoinBlocks,
    columns: &[String],
    (excluded, renames): (&mut Vec<usize>, &mut Vec<(usize, String)>),
) -> Vec<(usize, String)> {
    let star = stmt
        .columns
        .iter()
        .any(|c| matches!(c, Expression::Star(_)));
    let qualified_star = stmt
        .columns
        .iter()
        .any(|c| matches!(c, Expression::QualifiedStar(_)));
    let mut in_place = Vec::new();
    for (left, right) in held.side_merges() {
        if star {
            excluded.push(right);
        }
        let name = bare_name(&columns[left]);
        if renames.iter().any(|(named, _)| *named == left) || !(star || names_bare(stmt, name)) {
            continue;
        }
        match star || qualified_star {
            true => renames.push((left, name.to_string())),
            false => in_place.push((left, name.to_string())),
        }
    }
    in_place
}

/// `columns` and `rows`, joined rows of a join path that does not run
/// [`merged_columns`] itself, with the pairs of `held`'s joined sides shown
/// as the join's own are; as they are without `held`
fn merged_side_rows(
    stmt: &SelectStatement,
    held: Option<&JoinBlocks>,
    columns: Vec<String>,
    rows: RowVec,
) -> (Vec<String>, RowVec) {
    let Some(held) = held else {
        return (columns, rows);
    };
    let (mut excluded, mut renames) = (Vec::new(), Vec::new());
    let in_place = side_pairs(stmt, held, &columns, (&mut excluded, &mut renames));
    let (mut columns, rows) = match excluded.is_empty() && renames.is_empty() {
        true => (columns, rows),
        false => merged_columns(stmt, &columns, rows, (&excluded, &renames)),
    };
    for (at, name) in in_place {
        columns[at] = name;
    }
    (columns, rows)
}

/// Whether a clause of `stmt` the projection, grouping or sort reads by name
/// names `name` bare, or reads more than its names say
fn names_bare(stmt: &SelectStatement, name: &str) -> bool {
    if !stmt.window_defs.is_empty() {
        return true;
    }
    let mut named = false;
    let clauses = stmt
        .columns
        .iter()
        .chain(&stmt.group_by.columns)
        .chain(stmt.having.as_deref())
        .chain(stmt.order_by.iter().map(|ob| &ob.expression))
        .chain(&stmt.distinct_on);
    for clause in clauses {
        let read_all = visit_column_refs(clause, &mut |column| {
            if let Expression::Identifier(id) = column {
                named |= eq_folded(name, &id.value_lower);
            }
        });
        if !read_all || named {
            return true;
        }
    }
    false
}

/// The equalities a NATURAL join or a USING list makes of the sides'
/// columns, each spelled by `column(right, index)`, the right columns
/// SELECT * leaves out, and the left columns it names unqualified; none
/// for any other join
fn natural_join_condition(
    join_type: &str,
    join_source: &JoinTableSource,
    left_columns: &[String],
    right_columns: &[String],
    column: impl Fn(bool, usize) -> Expression,
) -> (Option<Expression>, Vec<usize>, Vec<(usize, String)>) {
    if !join_type.contains("NATURAL") && join_source.using_columns.is_empty() {
        return (None, Vec::new(), Vec::new());
    }
    // Find common columns between left and right tables
    // For this, we extract the base column name (without table qualifier)
    let left_base_cols: Vec<(usize, String)> = left_columns
        .iter()
        .enumerate()
        .map(|(i, c)| (i, extract_base_column_name(c)))
        .collect();

    let right_base_cols: Vec<(usize, String)> = right_columns
        .iter()
        .enumerate()
        .map(|(i, c)| (i, extract_base_column_name(c)))
        .collect();

    // Determine which columns to match
    // For NATURAL JOIN: all common columns
    // For USING clause: only specified columns
    let using_col_names: Vec<String> = join_source
        .using_columns
        .iter()
        .map(|c| c.value_lower.to_string())
        .collect();

    // Find matching column pairs and track excluded right-side columns
    // Also track left columns to rename to unqualified names per SQL standard
    let mut conditions: Vec<Expression> = Vec::new();
    let mut excluded_right_indices: Vec<usize> = Vec::new();
    let mut join_column_renames: Vec<(usize, String)> = Vec::new(); // (left_idx, base_name)

    // Each name the join merges is one column of each side, its first of
    // that name, as the bare name reads it in a side that is a join
    for (left_idx, left_base) in &left_base_cols {
        // For NATURAL JOIN, match all common columns
        // For USING, only match specified columns
        let named = using_col_names.is_empty() || using_col_names.contains(left_base);
        if !named
            || join_column_renames
                .iter()
                .any(|(_, name)| name == left_base)
        {
            continue;
        }
        let Some((right_idx, _)) = right_base_cols.iter().find(|(_, right)| right == left_base)
        else {
            continue;
        };

        // Track right-side columns to exclude from SELECT *
        // The index in all_columns is left_columns.len() + right_idx
        excluded_right_indices.push(left_columns.len() + *right_idx);
        // Track left column to rename to unqualified name
        join_column_renames.push((*left_idx, left_base.clone()));

        // Create equality condition: left_col = right_col
        conditions.push(Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Operator, "=", Position::default()),
            Box::new(column(false, *left_idx)),
            "=".to_string(),
            Box::new(column(true, *right_idx)),
        )));
    }

    // Combine conditions with AND
    if conditions.is_empty() {
        (None, Vec::new(), Vec::new())
    } else {
        let mut combined = conditions.remove(0);
        for cond in conditions {
            combined = Expression::Infix(InfixExpression::new(
                Token::new(TokenType::Keyword, "AND", Position::default()),
                Box::new(combined),
                "AND".to_string(),
                Box::new(cond),
            ));
        }
        (Some(combined), excluded_right_indices, join_column_renames)
    }
}

/// A join side as its WHERE is split over it: one source by its name, with
/// its columns when they are known, or a joined side by the name and the
/// columns of each table it joins
#[derive(Clone, Copy)]
pub(crate) enum SplitSide<'a> {
    Source(Option<&'a str>, Option<&'a Schema>),
    Joined(&'a [(SmartString, CompactArc<Schema>)]),
}

impl<'a> SplitSide<'a> {
    /// `side` as the blocks read it, `alias` its name when it is one source
    pub(crate) fn of(alias: Option<&'a str>, side: &'a JoinSide) -> Self {
        match side.leaves.as_slice() {
            [(_, schema)] => SplitSide::Source(alias, Some(schema)),
            leaves => SplitSide::Joined(leaves),
        }
    }

    /// Both sides of a join: as `sides` reads them when known, else each by
    /// its name, with its columns when it is a base table
    pub(crate) fn both(
        sides: Option<&'a JoinSides>,
        aliases: [Option<&'a str>; 2],
        schemas: [Option<&'a Schema>; 2],
    ) -> [Self; 2] {
        match sides {
            Some([left, right]) => [
                SplitSide::of(aliases[0], left),
                SplitSide::of(aliases[1], right),
            ],
            None => [
                SplitSide::Source(aliases[0], schemas[0]),
                SplitSide::Source(aliases[1], schemas[1]),
            ],
        }
    }

    fn named(&self) -> bool {
        !matches!(self, SplitSide::Source(None, _))
    }

    /// Whether `qualifier`, lowercase, names a table of the side
    fn qualified_by(&self, qualifier: &str) -> bool {
        let names = |name: &str| {
            name.chars()
                .flat_map(char::to_lowercase)
                .eq(qualifier.chars())
        };
        match self {
            SplitSide::Source(alias, _) => alias.is_some_and(names),
            SplitSide::Joined(leaves) => leaves.iter().any(|(name, _)| names(name)),
        }
    }

    /// Whether the side has a column `name`; None when its columns are not
    /// known
    fn has_column(&self, name: &str) -> Option<bool> {
        match self {
            SplitSide::Source(_, schema) => schema.map(|s| s.has_column(name)),
            SplitSide::Joined(leaves) => Some(leaves.iter().any(|(_, s)| s.has_column(name))),
        }
    }

    /// A filter pushed to the side as its read takes it: one source reads
    /// its columns by their bare names, a joined side by its tables'
    fn pushed(&self, filter: &Expression) -> Expression {
        match self {
            SplitSide::Source(Some(alias), _) => strip_table_qualifier(filter, alias),
            _ => filter.clone(),
        }
    }
}

/// A join's WHERE split as execution pushes it: a filter for each side it
/// may narrow, never the side an outer join pads with NULLs, and the rest,
/// asked of the joined rows
pub(crate) fn split_join_where(
    where_clause: &Expression,
    [left, right]: [SplitSide<'_>; 2],
    join_type: &str,
) -> (Option<Expression>, Option<Expression>, Option<Expression>) {
    if !left.named() || !right.named() {
        return (None, None, Some(where_clause.clone()));
    }
    let (l, r, c) = partition_where_for_join(where_clause, [left, right]);

    // For OUTER JOINs, we can't push filters to the NULL-padded side
    // because rows that don't match need to appear with NULLs
    let can_push_left = !join_type.contains("RIGHT") && !join_type.contains("FULL");
    let can_push_right = !join_type.contains("LEFT") && !join_type.contains("FULL");

    let safe_left = if can_push_left {
        l.as_ref().map(|f| left.pushed(f))
    } else {
        None
    };

    let safe_right = if can_push_right {
        r.as_ref().map(|f| right.pushed(f))
    } else {
        None
    };

    // Any filters we couldn't push need to be applied post-join
    // If we pushed a filter, don't include it in remaining; otherwise include it
    let unpushed_left = if !can_push_left { l } else { None };
    let unpushed_right = if !can_push_right { r } else { None };

    let remaining = match (unpushed_left, unpushed_right, c) {
        (Some(l), Some(r), Some(c)) => combine_predicates_with_and(vec![l, r, c]),
        (Some(l), Some(r), None) => combine_predicates_with_and(vec![l, r]),
        (Some(l), None, Some(c)) => combine_predicates_with_and(vec![l, c]),
        (None, Some(r), Some(c)) => combine_predicates_with_and(vec![r, c]),
        (Some(l), None, None) => Some(l),
        (None, Some(r), None) => Some(r),
        (None, None, c) => c,
    };

    (safe_left, safe_right, remaining)
}

/// The input column a SELECT item names, its alias set aside: a bare name,
/// or a qualified one by its full name and then its bare one; the error for
/// a name no column has; None for any other expression, a quoted name no
/// column has among them
fn select_column(expr: &Expression, col_index_map: &StringMap<usize>) -> Option<Result<usize>> {
    match unaliased_item(expr) {
        Expression::Identifier(id)
            if id.quoted && !col_index_map.contains_key(id.value_lower.as_str()) =>
        {
            None
        }
        Expression::Identifier(id) => Some(
            col_index_map
                .get(id.value_lower.as_str())
                .copied()
                .ok_or_else(|| Error::ColumnNotFound(id.value.to_string())),
        ),
        Expression::QualifiedIdentifier(qid) => Some(
            qualified_index(
                col_index_map,
                &qid.qualifier.value_lower,
                &qid.name.value_lower,
            )
            .ok_or_else(|| {
                Error::ColumnNotFound(format!("{}.{}", qid.qualifier.value, qid.name.value))
            }),
        ),
        _ => None,
    }
}

// Argument source for inline function evaluation
#[derive(Debug, Clone)]
enum ArgSource {
    Column(usize),
    Const(Value),
}

// Simple condition for inline CASE: column comparisons
#[derive(Debug, Clone)]
enum CaseCondition {
    Equals { col_idx: usize, value: Value },
    NotEquals { col_idx: usize, value: Value },
    GreaterThan { col_idx: usize, value: Value },
    GreaterOrEqual { col_idx: usize, value: Value },
    LessThan { col_idx: usize, value: Value },
    LessOrEqual { col_idx: usize, value: Value },
    IsNull { col_idx: usize },
}

// A single WHEN branch for inline CASE
#[derive(Debug, Clone)]
struct CaseBranch {
    condition: CaseCondition,
    result: ArgSource,
}

/// How one SELECT expression of an uncorrelated projection is evaluated
#[derive(Debug, Clone)]
enum ExprAction {
    SimpleColumn(usize),
    StarExpand,
    /// The columns a qualified star takes, found once
    QualifiedStarExpand(Vec<usize>),
    /// Inline COALESCE - bypasses VM for 7x speedup
    Coalesce(smallvec::SmallVec<[ArgSource; 4]>),
    /// Inline CASE - bypasses VM for simple equality/null checks
    Case {
        branches: smallvec::SmallVec<[CaseBranch; 4]>,
        else_result: Option<ArgSource>,
    },
    /// Columns, constants and parameters joined by `||`
    Concat(Concat),
    Compiled(SharedProgram),
}

impl ExprAction {
    /// The action running `program`: a concatenation runs without the VM,
    /// see [`Concat`]
    fn of_program(program: SharedProgram) -> Self {
        match Concat::of(&program) {
            Some(concat) => ExprAction::Concat(concat),
            None => ExprAction::Compiled(program),
        }
    }
}

/// How a SELECT list without correlated subqueries projects its input rows
#[derive(Debug)]
enum SelectProjection {
    /// The rows are their own projection
    Whole,
    /// The columns at these positions, cloned
    Subset(Vec<usize>),
    /// The columns at these positions, moved; none appears twice
    Take(Vec<usize>),
    Actions(Vec<ExprAction>),
}

/// What a projection keeping ORDER BY and DISTINCT ON values reads per row
#[derive(Debug)]
enum OrderedProjection {
    /// Columns only: the SELECT list's, every one a column, then the extra ORDER BY ones
    Columns {
        select: Vec<Option<usize>>,
        extra: Vec<usize>,
    },
    /// Each output value's source; a failed compile is raised by the first row
    Items(Vec<Result<ItemSource>>),
}

/// A join statement's output over joined rows of one layout: how each row
/// is projected, the output column names, and where each comes from
#[derive(Debug)]
pub(crate) struct JoinOutput {
    projection: JoinProjection,
    columns: CompactArc<Vec<String>>,
    layout: Option<CompactArc<OutputLayout>>,
    /// The columns shown, when ORDER BY keys follow a star's
    visible: Option<usize>,
}

#[derive(Debug)]
enum OutputProjection {
    Select(SelectProjection),
    /// Keeps the ORDER BY and DISTINCT ON values the SELECT list leaves out
    Ordered(OrderedProjection),
}

#[derive(Debug)]
enum JoinProjection {
    General(OutputProjection),
    /// A SELECT list or ORDER BY holding subqueries the statement's blocks run
    Prepared(PreparedProjection),
}

/// The columns of the row a statement runs for, which its programs read by name
pub(crate) fn outer_column_names(ctx: &ExecutionContext) -> Option<Vec<String>> {
    ctx.outer_row()
        .filter(|outer| !outer.is_empty())
        .map(|outer| outer.keys().map(|k| k.to_string()).collect())
}

/// Each lowercase column name, and each qualified one's bare name, to its position
fn lower_index_map(
    all_columns: &[String],
    all_columns_lower: Option<&[String]>,
) -> StringMap<usize> {
    let computed_lower: Vec<String>;
    let columns_lower: &[String] = match all_columns_lower {
        Some(lower) => lower,
        None => {
            computed_lower = all_columns.iter().map(|c| c.to_lowercase()).collect();
            &computed_lower
        }
    };
    // Pre-size HashMap to avoid rehashing (estimate 1.5x for qualified names)
    let mut col_index_map_lower: StringMap<usize> =
        StringMap::with_capacity(all_columns.len() * 3 / 2);
    for (i, lower) in columns_lower.iter().enumerate() {
        col_index_map_lower.insert(lower.clone(), i);
        // Also add unqualified column names for qualified columns
        if let Some((_, column_part)) = split_qualified(lower) {
            col_index_map_lower
                .entry(column_part.to_string())
                .or_insert(i);
        }
    }
    col_index_map_lower
}

/// Where one projected value comes from: a column of the input row, or a
/// program compiled once for the statement
#[derive(Debug, Clone)]
enum ItemSource {
    Column(usize),
    Program(SharedProgram),
}

/// `items` with each program `program_of` finds bound to `params`, None
/// when none binds: a plan may be held, so each run binds its own copy.
/// Out of line, so the row loop after it compiles as it would without it
#[inline(never)]
fn bound_items<T: Clone>(
    items: &[T],
    params: &[Value],
    program_of: impl Fn(&T) -> Option<&SharedProgram>,
    with_program: impl Fn(SharedProgram) -> T,
) -> Option<Vec<T>> {
    let mut bound: Option<Vec<T>> = None;
    for (at, item) in items.iter().enumerate() {
        if let Some(program) = program_of(item).and_then(|program| program.bind(params)) {
            bound.get_or_insert_with(|| items.to_vec())[at] =
                with_program(CompactArc::new(program));
        }
    }
    bound
}

/// Where a projection puts the rows it builds
trait RowSink {
    /// Room for a row of `width` values
    fn storage(&mut self, width: usize) -> CompactVec<Value>;
    fn put(&mut self, id: i64, row: Row);
}

impl RowSink for RowVec {
    #[inline]
    fn storage(&mut self, width: usize) -> CompactVec<Value> {
        CompactVec::with_capacity(width)
    }

    #[inline]
    fn put(&mut self, id: i64, row: Row) {
        self.push((id, row));
    }
}

/// The rows an ORDER BY LIMIT keeps of a projection; a row it does not
/// keep lends its storage to the next
struct TopSink<'t, F: Fn(&Row, &Row) -> Ordering> {
    top: &'t mut TopN<F>,
    spare: Option<Row>,
}

impl<F: Fn(&Row, &Row) -> Ordering> RowSink for TopSink<'_, F> {
    #[inline]
    fn storage(&mut self, width: usize) -> CompactVec<Value> {
        match self.spare.take() {
            Some(row) => {
                let mut values = CompactVec::from_vec(row.into_values());
                values.clear();
                values
            }
            None => CompactVec::with_capacity(width),
        }
    }

    #[inline]
    fn put(&mut self, _id: i64, row: Row) {
        self.spare = self.top.push(row);
    }
}

#[cold]
fn view_parameters() -> Error {
    Error::InvalidArgument(crate::parser::parser::VIEW_TAKES_NO_PARAMETERS.to_string())
}

/// A view's query reads no table AS OF
fn check_view_reads_now(statement: &Statement) -> Result<()> {
    match statement {
        Statement::Select(select) if super::prepared_select::holds_as_of(select) => Err(
            Error::invalid_argument("a view's query cannot read a table AS OF"),
        ),
        _ => Ok(()),
    }
}

/// A SELECT item without its aliases
fn unaliased_item(mut expr: &Expression) -> &Expression {
    while let Expression::Aliased(aliased) = expr {
        expr = &aliased.expression;
    }
    expr
}

fn is_star_item(expr: &Expression) -> bool {
    matches!(expr, Expression::Star(_) | Expression::QualifiedStar(_))
}

/// Whether `each` holds for one of a WHERE's AND terms that is a statement
/// constant over the functions of `registry`, so no column reads as the
/// text of its name; asked in order
pub(crate) fn any_constant_term<'e>(
    expr: &'e Expression,
    registry: &FunctionRegistry,
    each: &mut impl FnMut(&'e Expression) -> bool,
) -> bool {
    if let Expression::Infix(infix) = expr {
        if infix.op_type == InfixOperator::And {
            return any_constant_term(&infix.left, registry, each)
                || any_constant_term(&infix.right, registry, each);
        }
    }
    statement_constant(expr, registry) && each(expr)
}

/// Whether one of a WHERE's AND terms is a statement constant: only such a
/// WHERE can be [`true_for_no_row`]
pub(crate) fn has_constant_term(expr: &Expression, registry: &FunctionRegistry) -> bool {
    any_constant_term(expr, registry, &mut |_| true)
}

/// Whether a WHERE `expr` keeps no row: one of its constant AND terms
/// evaluates to false or NULL with the functions of `registry`
pub(crate) fn true_for_no_row(
    expr: &Expression,
    ctx: &ExecutionContext,
    registry: &FunctionRegistry,
) -> bool {
    any_constant_term(expr, registry, &mut |term| {
        matches!(
            super::expression::evaluate_once_in(term, Some(ctx), registry),
            Ok(Value::Boolean(false) | Value::Null(_))
        )
    })
}

/// Partition WHERE clause predicates for JOIN filter pushdown.
/// Returns (left_filter, right_filter, cross_table_filter).
/// - left_filter: predicates referencing only left table
/// - right_filter: predicates referencing only right table
/// - cross_table_filter: predicates referencing both tables (must be applied post-join)
///
/// An unqualified column belongs to a side only when both sides' columns
/// are known and only that one has it; otherwise its predicate stays post-join
fn partition_where_for_join(
    where_clause: &Expression,
    [left, right]: [SplitSide<'_>; 2],
) -> (Option<Expression>, Option<Expression>, Option<Expression>) {
    // Collect AND-ed predicates
    let predicates = flatten_and_predicates(where_clause);

    let mut left_preds = Vec::new();
    let mut right_preds = Vec::new();
    let mut cross_preds = Vec::new();

    for pred in predicates {
        let qualifiers = collect_table_qualifiers(&pred);
        let mut unqualified = FxHashSet::default();
        collect_unqualified_columns(&pred, &mut unqualified);

        let mut refs_left = qualifiers.iter().any(|q| left.qualified_by(q));
        let mut refs_right = qualifiers.iter().any(|q| right.qualified_by(q));
        let mut unowned = false;
        for column in &unqualified {
            match (left.has_column(column), right.has_column(column)) {
                (Some(true), Some(false)) => refs_left = true,
                (Some(false), Some(true)) => refs_right = true,
                _ => unowned = true,
            }
        }

        // The qualifiers inside a subquery are not collected, so a correlated
        // one may read either side and is evaluated on the joined row, as is
        // a predicate that draws a new value for every row it is asked of
        if (refs_left && refs_right)
            || unowned
            || Executor::has_correlated_subqueries(&pred)
            || expression_volatility(&pred, global_registry()) == Volatility::PerCall
        {
            // References both tables - must be applied post-join
            cross_preds.push(pred);
        } else if refs_left {
            // Only references left table - push to left
            left_preds.push(pred);
        } else if refs_right {
            // Only references right table - push to right
            right_preds.push(pred);
        } else {
            // No table references (constants) - push to left arbitrarily
            left_preds.push(pred);
        }
    }

    (
        combine_predicates_with_and(left_preds),
        combine_predicates_with_and(right_preds),
        combine_predicates_with_and(cross_preds),
    )
}

/// The index join operator a statement's join runs on, for EXPLAIN
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum JoinIndexUse {
    None,
    /// `rows` marks a side the blocks read as rows, which no index probes
    Streaming {
        bounded: bool,
        rows: [bool; 2],
    },
    Grouped,
}

/// A GROUP BY + LIMIT join whose left side is fetched in a limited chunk
struct SemijoinReduction {
    cap: usize,
    want: usize,
    offset: usize,
    limit: usize,
    left_key_col: String,
    right_key_col: String,
}

/// The left chunk a reduction pass read: `read` of the `cap` rows it asked
/// for, the last of them, the layout it read them in and the groups the
/// statement wants; `moved` when a commit moved the rows passes before read
struct ReductionPass {
    cap: usize,
    read: usize,
    bounded: bool,
    last: Option<Row>,
    token: Option<LayoutToken>,
    want: usize,
    moved: bool,
}

impl ReductionPass {
    fn moved() -> Self {
        ReductionPass {
            cap: 0,
            read: 0,
            bounded: false,
            last: None,
            token: None,
            want: 0,
            moved: true,
        }
    }
}

/// The left chunk a GROUP BY + LIMIT join reads: `cap` rows after the
/// `offset` the chunks before it read, the last of which was `boundary`
struct LeftWindow {
    offset: usize,
    cap: usize,
    boundary: Option<Row>,
}

/// The most rows an outer side may hold for a join without a LIMIT to
/// narrow its inner side by the inner table's candidates
pub(super) const SMALL_OUTER_ROWS: usize = 1024;

/// Inner rows a table without sealed rows must hold per row of the largest
/// small outer before a small outer narrows it: a key read costs about what
/// its scan pays for 17 rows, 45 under a pushed filter
pub(super) const HOT_INNER_ROWS_PER_KEY: usize = 64;

/// A small outer side the join ran and handed on to the standard join
struct PreparedSide {
    rows: PreparedRows,
    columns: Vec<String>,
    /// The side is the join's right side
    is_right: bool,
}

/// A prepared side's rows: unread, read in part with the rest still in
/// its result, or held for more than one reader
enum PreparedRows {
    Unread(Box<dyn QueryResult>),
    Partial(Vec<Row>, Box<dyn QueryResult>),
    Shared(CompactArc<Vec<Row>>),
}

impl PreparedRows {
    /// The rows of `result` held when there are at most `SMALL_OUTER_ROWS`,
    /// else the result as far as it was read. A result that knows its length
    /// is taken whole or not at all; any other is read one row past the cap
    fn read_small(mut result: Box<dyn QueryResult>) -> Result<Self> {
        match result.exact_len() {
            Some(len) if len > SMALL_OUTER_ROWS => return Ok(PreparedRows::Unread(result)),
            Some(_) => {
                if let Some(rows) = result.try_into_arc_rows() {
                    return Ok(PreparedRows::Shared(rows));
                }
                return Ok(PreparedRows::Unread(result));
            }
            None => {}
        }
        let mut head = Vec::new();
        while head.len() <= SMALL_OUTER_ROWS && result.next() {
            head.push(result.take_row());
        }
        if let Some(error) = result.last_error() {
            return Err(error);
        }
        if head.len() <= SMALL_OUTER_ROWS {
            Ok(PreparedRows::Shared(CompactArc::new(head)))
        } else {
            Ok(PreparedRows::Partial(head, result))
        }
    }

    fn into_rows(self) -> Result<CompactArc<Vec<Row>>> {
        match self {
            PreparedRows::Unread(result) => Executor::materialize_result_arc(result),
            PreparedRows::Partial(mut rows, mut rest) => {
                while rest.next() {
                    rows.push(rest.take_row());
                }
                if let Some(error) = rest.last_error() {
                    return Err(error);
                }
                Ok(CompactArc::new(rows))
            }
            PreparedRows::Shared(rows) => Ok(rows),
        }
    }
}

impl Executor {
    /// Multi-column ORDER BY comparator over precomputed sort keys, with
    /// NULLS FIRST/LAST semantics (default: NULLS LAST for ASC, FIRST for
    /// DESC)
    fn compare_by_sort_keys_impl(
        a_keys: &[Value],
        b_keys: &[Value],
        order_by: &[crate::parser::ast::OrderByExpression],
        num_order_cols: usize,
    ) -> Ordering {
        for (i, ob) in order_by.iter().take(num_order_cols).enumerate() {
            let cmp = crate::core::value::compare_order_key(
                a_keys.get(i),
                b_keys.get(i),
                ob.ascending,
                ob.nulls_first,
            );
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Execute a SELECT statement
    pub(crate) fn execute_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_select_with_plan(stmt, ctx, None, None)
    }

    /// `ctx` with the snapshot a statement reading its LIMIT or OFFSET
    /// subqueries, or its CTEs before its prepared subqueries, reads; None in
    /// an explicit transaction, read through itself
    #[cold]
    #[inline(never)]
    fn with_counts_snapshot(&self, ctx: &ExecutionContext) -> Result<Option<ExecutionContext>> {
        let in_transaction = self
            .active_transaction
            .lock()
            .map_err(|_| Error::internal("active transaction lock poisoned"))?
            .is_some();
        if in_transaction {
            return Ok(None);
        }
        let snapshot = self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?;
        Ok(Some(ctx.with_statement_snapshot(snapshot)))
    }

    /// `ctx` with `stmt`'s LIMIT and OFFSET read once when one draws a new
    /// value per call and none holds a subquery, which is read after the
    /// statement's CTEs instead; None otherwise
    pub(crate) fn read_volatile_counts(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Option<ExecutionContext>> {
        let counts = [stmt.limit.as_deref(), stmt.offset.as_deref()];
        let volatile = counts.into_iter().flatten().any(|count| {
            !matches!(
                count,
                Expression::IntegerLiteral(_) | Expression::Parameter(_)
            ) && expression_volatility(count, &self.function_registry) == Volatility::PerCall
        });
        if !volatile || counts.into_iter().flatten().any(Self::has_subqueries) {
            return Ok(None);
        }
        self.read_row_counts(stmt, ctx, None).map(Some)
    }

    /// `ctx` with the counts the subqueries of `stmt`'s LIMIT and OFFSET
    /// answer, read once before the statement reads a row, under its
    /// snapshot; their blocks are held in `stmt`'s plan `compiled`
    #[cold]
    #[inline(never)]
    fn read_row_counts(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        compiled: Option<&NestedPlan>,
    ) -> Result<ExecutionContext> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::row_counts_read();
        let clauses = [
            (stmt.limit.as_deref(), "LIMIT"),
            (stmt.offset.as_deref(), "OFFSET"),
        ];
        let counts = clauses.iter().filter_map(|(count, _)| *count);
        // Counts without a subquery are evaluated as they are written
        if !counts.clone().any(Self::has_subqueries) {
            let mut read = smallvec::SmallVec::new();
            for (count, clause) in clauses {
                if let Some(count) = count {
                    read.push(super::context::ReadCount {
                        clause,
                        address: count as *const Expression as usize,
                        key: ExprKey::of(count),
                        count: super::row_count::row_count(count, ctx, clause)?,
                    });
                }
            }
            return Ok(ctx.with_read_counts(super::context::ReadCounts {
                depth: ctx.query_depth,
                counts: read,
            }));
        }
        if let Err(reason) = self.dml_family(None, counts, ctx) {
            return Err(Error::invalid_argument(format!(
                "a LIMIT or OFFSET subquery is not supported here: {reason}"
            )));
        }
        // A count is read before any row: its scope is a table without columns
        let slot = compiled.map(|plan| plan.counts(stmt));
        let values = clauses.map(|(count, _)| count.map_or(&[][..], std::slice::from_ref));
        let (prepared, tables) = self.open_prepared_dml(
            slot.map(|slot| slot.compiled.slot()),
            &super::prepared_select::NO_COLUMNS,
            &SmartString::default(),
            None,
            &[],
            &values,
            None,
            None,
            None,
            ctx,
        )?;
        let run = super::prepared_select::DmlRunner::new(self, &prepared, ctx, tables, 0)?;
        let mut vm = super::expression::ExprVM::new();
        let no_row = Row::new();
        let exec = run.context(&no_row);
        // A plan keeps the expressions' identities; only a statement run
        // without one hashes them here
        let keys = match slot {
            Some(slot) => slot.keys,
            None => clauses.map(|(count, _)| count.map(ExprKey::of)),
        };
        let mut counts = smallvec::SmallVec::new();
        for (((count, clause), key), programs) in clauses.iter().zip(keys).zip(run.values()) {
            if let (Some(count), Some(key), Some(Some(program))) = (count, key, programs.first()) {
                let value = vm.execute(program, &exec)?;
                counts.push(super::context::ReadCount {
                    clause,
                    address: *count as *const Expression as usize,
                    key,
                    count: super::row_count::count_of(value, clause)?,
                });
            }
        }
        drop(run);
        Ok(ctx.with_read_counts(super::context::ReadCounts {
            depth: ctx.query_depth,
            counts,
        }))
    }

    /// `execute_select` with the cached plan's classification slot, so
    /// repeated executions skip the AST hash and the global
    /// classification-cache mutex after the first resolution.
    pub(crate) fn execute_select_with_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        plan_classification: Option<&std::sync::OnceLock<Arc<QueryClassification>>>,
        compiled: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        let result = self.select_with_plan(stmt, ctx, plan_classification, compiled)?;
        // A column read through a FROM entry shows as `entry.name`
        if !result
            .columns()
            .iter()
            .any(|c| split_qualified(c).is_some())
        {
            return Ok(result);
        }
        let shown = result
            .columns()
            .iter()
            .map(|c| shown_name(c).into_owned())
            .collect();
        Ok(Box::new(AliasedResult::renamed(result, shown)))
    }

    fn select_with_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        plan_classification: Option<&std::sync::OnceLock<Arc<QueryClassification>>>,
        compiled: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        // Start timeout guard ONLY at the top level (query_depth == 0).
        // For nested queries (subqueries, views), the parent's TimeoutGuard handles timeout.
        // This ensures the timeout applies to the entire query, not each nested call.
        let _timeout_guard = if ctx.query_depth == 0 {
            // Clear query-local caches at top-level.
            // NOTE: scalar, IN, and semi-join caches are NOT cleared here - they're
            // invalidated per-table when data changes (INSERT, UPDATE, DELETE, TRUNCATE)
            // to enable cross-query caching for repeated subqueries.
            clear_statement_caches();
            TimeoutGuard::new(ctx)
        } else {
            None
        };

        // Check for cancellation at entry point
        ctx.check_cancelled()?;

        // Validate: aggregate functions are not allowed in WHERE clause
        if let Some(ref where_clause) = stmt.where_clause {
            if expression_contains_aggregate(where_clause) {
                return Err(Error::invalid_argument(
                    "aggregate functions are not allowed in WHERE clause (use HAVING instead)",
                ));
            }
        }

        // A LIMIT or OFFSET subquery reads the snapshot the statement's CTEs
        // and rows read, so it is taken before the CTEs run
        let reads_blocks = [stmt.limit.as_deref(), stmt.offset.as_deref()]
            .into_iter()
            .flatten()
            .any(Self::has_subqueries);
        let snapshot_ctx;
        let ctx = if reads_blocks && ctx.statement_snapshot().is_none() {
            snapshot_ctx = self.with_counts_snapshot(ctx)?;
            snapshot_ctx.as_ref().unwrap_or(ctx)
        } else {
            ctx
        };
        // Without a subquery a count reads no CTE, so it is read before the
        // CTEs, whose statement may take a path of its own
        let volatile_ctx = self.read_volatile_counts(stmt, ctx)?;
        let ctx = volatile_ctx.as_ref().unwrap_or(ctx);

        // Check for CTEs (WITH clause). A statement with one keeps every
        // subquery for the aggregation entry, which runs with the CTEs bound
        // A statement of the prepared or grouped family keeps its plan: its
        // CTEs inline as the CTE path inlines them, or are read from the
        // context, materialized under the snapshot its subqueries read and
        // cut as the CTE path cuts them
        if self.has_cte(stmt) {
            let grouped = || {
                Self::grouped_family_reach(stmt, &get_classification(stmt))
                    && self.grouped_family(stmt, Some(ctx)).is_ok()
            };
            let with = stmt
                .with
                .as_ref()
                .ok_or_else(|| Error::internal("a statement with CTEs without WITH"))?;
            let slot = compiled.map(NestedPlan::slot);
            if !self.holds_prepared_select(slot, ctx)
                && !self.holds_prepared_grouped(slot, ctx)
                && self.prepared_family(stmt, Some(ctx)).is_err()
                && !(slot.is_some()
                    && self.reads_source(stmt, ctx)
                    && self.planned_family(stmt, Some(ctx)).is_ok())
                && !grouped()
                && !self.run_keeps_join_plan(slot, stmt, with, ctx)
            {
                return self.execute_select_with_ctes(stmt, ctx, compiled);
            }
            if let Some(inlined) = self.try_inline_ctes(stmt, with) {
                return self.execute_select_with_plan(&inlined, ctx, None, compiled);
            }
            let snapshot = match ctx.statement_snapshot() {
                Some(_) => None,
                None => self.with_counts_snapshot(ctx)?,
            };
            let ctx = snapshot.as_ref().unwrap_or(ctx);
            return self.run_over_cut_ctes(stmt, with, compiled, ctx, |ctes| {
                let ctx = ctx.with_cte_data(ctes.data());
                self.select_rows_with_plan(stmt, &ctx, reads_blocks, plan_classification, compiled)
            });
        }
        self.select_rows_with_plan(stmt, ctx, reads_blocks, plan_classification, compiled)
    }

    /// `select_with_plan` past its CTEs, which `ctx` holds
    fn select_rows_with_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        reads_blocks: bool,
        plan_classification: Option<&std::sync::OnceLock<Arc<QueryClassification>>>,
        compiled: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        // The subquery is read once, before any path reads a row, with the
        // CTEs of the statement bound
        let counts_ctx;
        let ctx = if reads_blocks {
            counts_ctx = self.read_row_counts(stmt, ctx, compiled)?;
            &counts_ctx
        } else {
            ctx
        };

        // OPTIMIZATION: Get cached query classification ONCE at entry point
        // This classification is passed through the call chain to avoid
        // redundant hash computations and cache lookups (was 10+ calls per query)
        let classification = match plan_classification {
            Some(slot) => slot.get_or_init(|| get_classification(stmt)).clone(),
            None => get_classification(stmt),
        };

        // A statement of the prepared family runs on that path and no other,
        // and so does one over a single source with a plan slot to keep
        let slot = compiled.map(NestedPlan::slot);
        // A nested statement is planned into its slot as the router plans
        // the statement it runs
        let held = self.holds_prepared_select(slot, ctx)
            || slot.is_some_and(|slot| {
                self.with_planned(stmt, slot, plan_classification, Some(ctx), |_| ());
                self.holds_prepared_select(Some(slot), ctx)
            });
        let in_family = held
            || ((classification.where_has_subqueries || classification.output_has_subqueries)
                && self.prepared_family(stmt, Some(ctx)).is_ok())
            || (slot.is_some()
                && self.reads_source(stmt, ctx)
                && self.planned_family(stmt, Some(ctx)).is_ok());
        // A grouped or windowed statement whose clauses hold subqueries runs
        // them as its blocks
        let grouped_family = Self::grouped_family_reach(stmt, &classification)
            && !in_family
            && (self.holds_prepared_grouped(slot, ctx)
                || self.grouped_family(stmt, Some(ctx)).is_ok());

        // A SELECT * the runner answers is kept in the semantic cache as the
        // general path keeps it, under a stamp taken before its snapshot
        let semantic_stamp = match (in_family, stmt.table_expr.as_deref()) {
            (true, Some(Expression::TableSource(source)))
                if !self.active_transaction.is_open()
                    && Self::semantic_writable(stmt, ctx, &classification, source) =>
            {
                self.semantic_write_stamp(&source.name.value_lower, ctx)
                    .map(|stamp| (source, stamp))
            }
            _ => None,
        };

        // Correlated probes read the statement's tables through one transaction
        let snapshot_ctx;
        let ctx = if (classification.where_has_correlated_subqueries
            || classification.select_has_correlated_subqueries
            || classification.order_by_has_correlated_subqueries
            || in_family
            || grouped_family)
            && ctx.statement_snapshot().is_none()
            && !self.active_transaction.is_open()
        {
            let snapshot =
                self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        // The LIMIT and OFFSET are read before any path reads a row, a grouped
        // statement's source included
        let RowCounts { limit, offset } = row_counts(stmt, ctx)?;

        // An uncorrelated subquery inside an aggregate's FILTER or argument
        // folds before the aggregates are parsed, since every aggregation
        // path reads the statement's own columns; a correlated one is
        // resolved per input row at the aggregation entry
        let bound_stmt;
        // A statement holding this execution's values keeps nothing in its plan
        let mut folded_select = false;
        // A join prepares its grouped blocks once it has laid out its rows
        let grouped_join =
            grouped_family && matches!(stmt.table_expr.as_deref(), Some(Expression::JoinSource(_)));
        let grouped = match grouped_family && !grouped_join {
            true => self.prepared_grouped(compiled, stmt, ctx)?,
            false => None,
        };
        let stmt = if grouped.as_ref().is_none_or(|run| run.held.is_none())
            && !grouped_join
            && Self::folds_select_subqueries(stmt, &classification)
        {
            match self.try_process_select_subqueries(&stmt.columns, ctx)? {
                Some(columns) => {
                    // A folded item keeps the name it is written under
                    let columns = columns
                        .into_iter()
                        .zip(&stmt.columns)
                        .map(|(folded, written)| named_as_written(folded, written))
                        .collect();
                    bound_stmt = SelectStatement {
                        columns,
                        ..stmt.clone()
                    };
                    folded_select = true;
                    &bound_stmt
                }
                None => stmt,
            }
        } else {
            stmt
        };

        // Execute the main query
        // The third return value indicates if LIMIT/OFFSET was already applied (by storage-level pushdown)
        // The fourth return value contains deferred projection info if applicable
        // A pure UNION ALL without ORDER BY or DISTINCT needs LIMIT + OFFSET
        // rows at most from each branch, so that many bound the first
        // branch and the set operation; the whole is cut below
        let all_union_all = stmt
            .set_operations
            .iter()
            .all(|op| matches!(op.operation, SetOperationType::UnionAll));
        let set_limit = if all_union_all && stmt.order_by.is_empty() && !stmt.distinct {
            // A sum past i64 is no bound the branch can take
            limit
                .and_then(|l| l.checked_add(offset))
                .filter(|bound| i64::try_from(*bound).is_ok())
        } else {
            None
        };

        // ORDER BY, LIMIT and OFFSET belong to the whole set operation, so
        // the first branch runs without them, apart from that bound. With
        // no ORDER BY or OFFSET the statement's own LIMIT is that bound,
        // so the statement runs as it is, without a copy
        let branch_stmt;
        let branch = if stmt.set_operations.is_empty()
            || (stmt.order_by.is_empty()
                && stmt.offset.is_none()
                && (stmt.limit.is_none() || set_limit.is_some()))
        {
            stmt
        } else {
            branch_stmt = SelectStatement {
                order_by: Vec::new(),
                limit: set_limit.map(|l| {
                    Box::new(Expression::IntegerLiteral(IntegerLiteral {
                        token: dummy_token(&l.to_string(), TokenType::Integer),
                        value: l as i64,
                    }))
                }),
                offset: None,
                ..stmt.clone()
            };
            &branch_stmt
        };
        // A DISTINCT ON key the aggregation or window path computes rides along
        // as a hidden item, which the visible column count trims after
        let keyed_stmt;
        let keyed_classification;
        let written_items = branch.columns.len();
        let (branch, branch_classification) =
            match Self::with_distinct_on_items(branch, &classification) {
                Some(keyed) => {
                    keyed_stmt = keyed;
                    keyed_classification = get_classification(&keyed_stmt);
                    (&keyed_stmt, &keyed_classification)
                }
                None => (branch, &classification),
            };
        // The columns past the visible ones: the carried keys, and the ORDER BY
        // windows no item, a carried key included, computes
        let hidden_keys = branch.columns.len() - written_items;
        let hidden_windows = match branch_classification.has_window_functions {
            true => Self::hidden_order_by_windows(branch).len(),
            false => 0,
        };
        // A prepared statement knows how many columns it shows, a star included
        let mut prepared_visible = None;
        let mut distinct_applied = false;
        // The WHERE as the cache reads and keeps it, simplified by the
        // statement's start or here
        let simplified = match (semantic_stamp, stmt.where_clause.as_deref()) {
            (Some(_), Some(where_expr)) if ctx.semantic_miss(stmt).is_none() => {
                ExpressionSimplifier::new().try_simplify(where_expr)
            }
            _ => None,
        };
        // A statement of the family the start did not look up, a nested one,
        // reads the cache first, as the table scan does, unless its snapshot
        // keeps a point of its own, which the stamp may predate
        let cached = match (in_family, semantic_stamp, stmt.where_clause.as_deref()) {
            (true, Some((source, stamp)), Some(where_expr))
                if ctx.semantic_miss(stmt).is_none() && Self::reads_cacheable_rows(ctx) =>
            {
                let predicate = simplified.as_ref().unwrap_or(where_expr);
                let params = classification.where_has_parameters.then(|| ctx.params());
                self.semantic_source_answer(source, stmt, predicate, params, stamp, ctx)?
            }
            _ => None,
        };
        let executed = if let Some(answer) = cached {
            answer
        } else if in_family {
            // DISTINCT reads every row before a LIMIT counts them; a set
            // operation's branch takes the bound its whole one allows
            let (limit, offset) = match (stmt.set_operations.is_empty(), stmt.distinct) {
                (false, _) => (set_limit, 0),
                (true, true) => (None, 0),
                (true, false) => (limit, offset),
            };
            // An answer kept for the cache is the WHERE's whole one, read
            // only over a table small enough for the cache to take it; the
            // caller then skips the OFFSET's rows
            let semantic_stamp = semantic_stamp.filter(|(source, _)| {
                self.prepared_table(&source.name.value_lower, ctx, None)
                    .is_ok_and(|table| Self::semantic_rows_fit(&*table))
            });
            let offset = if semantic_stamp.is_some() { 0 } else { offset };
            let (result, columns, applied, visible, layout, distinct) =
                self.execute_prepared_select(stmt, ctx, compiled, limit, offset)?;
            let result = match (semantic_stamp, stmt.where_clause.as_deref()) {
                (Some((source, stamp)), Some(where_expr)) => {
                    let rows = Self::materialize_result(result)?;
                    let predicate = match ctx.semantic_miss(stmt) {
                        Some(miss) => miss.simplified.as_deref().unwrap_or(where_expr),
                        None => simplified.as_ref().unwrap_or(where_expr),
                    };
                    let table_name = &source.name.value_lower;
                    let params = classification.where_has_parameters.then(|| ctx.params());
                    self.semantic_write(
                        table_name,
                        &columns,
                        rows.rows(),
                        predicate,
                        params,
                        stamp,
                    );
                    Box::new(ExecutorResult::with_arc_columns(columns.clone(), rows))
                }
                _ => result,
            };
            prepared_visible = Some(visible);
            distinct_applied = distinct;
            let shape = SelectShape {
                deferred: None,
                layout: layout.map(LaidOut::Shared),
                visible: None,
            };
            (result, columns, applied, Some(shape))
        } else if let Some(grouped) = grouped {
            self.execute_prepared_grouped(grouped, branch, ctx)?
        } else {
            let compiled = compiled.filter(|_| !folded_select);
            self.execute_select_internal(branch, ctx, branch_classification, compiled)?
        };
        self.finish_select(
            (stmt, ctx, &classification, compiled),
            (executed, prepared_visible, distinct_applied),
            (limit, offset, set_limit),
            (written_items, hidden_keys, hidden_windows),
        )
    }

    /// `stmt`'s rows from what its executor gave, `executed`, with the
    /// output columns a prepared statement shows and whether it applied the
    /// DISTINCT: its DISTINCT, set operations, ORDER BY, DISTINCT ON and
    /// LIMIT, each unless the executor applied it, and only its own columns
    fn finish_select(
        &self,
        (stmt, ctx, classification, compiled): (
            &SelectStatement,
            &ExecutionContext,
            &QueryClassification,
            Option<&NestedPlan>,
        ),
        (executed, prepared_visible, distinct_applied): (SelectOutput, Option<usize>, bool),
        (limit, offset, set_limit): (Option<usize>, usize, Option<usize>),
        (written_items, hidden_keys, hidden_windows): (usize, usize, usize),
    ) -> Result<Box<dyn QueryResult>> {
        let (mut result, columns, limit_offset_applied, shape) = executed;
        let SelectShape {
            deferred: deferred_projection,
            layout,
            visible,
        } = shape.unwrap_or_default();
        let prepared_visible = prepared_visible.or(visible);

        // Count expected SELECT columns (before any extra ORDER BY columns).
        // Beside a star the count is unknown: the layout says which columns
        // a SELECT item gives; without one, the only columns past the
        // visible ones are the window functions ORDER BY had computed for
        // it, so those are counted off the end
        let expected_columns =
            match prepared_visible.unwrap_or_else(|| self.count_select_columns(stmt)) {
                0 if stmt.set_operations.is_empty() && layout.is_some() => {
                    layout.as_deref().map_or(0, |l| l.shown(written_items))
                }
                0 if hidden_windows + hidden_keys > 0 => {
                    let hidden = hidden_windows + hidden_keys;
                    if hidden > 0 {
                        columns.len().saturating_sub(hidden)
                    } else {
                        0
                    }
                }
                counted => counted,
            };
        // Beside a star only the rows know the width an ORDER BY position
        // counts in: the columns before those appended for the sort
        if stmt.columns.iter().any(is_star_item) {
            let width = match (&deferred_projection, expected_columns) {
                (Some((_, names)), _) => names.len(),
                (None, 0) => columns.len(),
                (None, shown) => shown,
            };
            if let Some(position) = crate::parser::scope::order_position_outside(stmt, Some(width))
            {
                return Err(Error::invalid_argument(
                    crate::parser::scope::order_position_message(position.value),
                ));
            }
        }

        // Apply DISTINCT (skip for DISTINCT ON — it's applied after ORDER BY)
        // When ORDER BY references columns not in SELECT, we add extra columns for sorting.
        // DISTINCT should only consider the original SELECT columns, not the extra ORDER BY columns.
        // A set operation's DISTINCT is its first SELECT's, so it comes first
        if stmt.distinct && stmt.distinct_on.is_empty() && !distinct_applied {
            if columns.len() > expected_columns && expected_columns > 0 {
                // Extra ORDER BY columns present - only hash SELECT columns for distinctness
                result = Box::new(DistinctResult::with_column_count(
                    result,
                    Some(expected_columns),
                ));
            } else {
                result = Box::new(DistinctResult::new(result));
            }
        }

        // Apply set operations (UNION, INTERSECT, EXCEPT)
        // Pass limit+offset to enable early termination for UNION ALL
        let mut limit_offset_applied = limit_offset_applied;
        if !stmt.set_operations.is_empty() {
            result = self.execute_set_operations(result, stmt, compiled, ctx, set_limit)?;

            // The set operation gathers LIMIT + OFFSET rows at most; the
            // OFFSET is skipped and the LIMIT cut below, on the whole
            limit_offset_applied = false;
        }

        // Apply ORDER BY (with TOP-N optimization if LIMIT is present)
        // Note: LIMIT/OFFSET was already evaluated earlier for set operations optimization
        // Skip ORDER BY if storage-level optimization already applied sorting + LIMIT/OFFSET
        // classification was already obtained at entry point and passed through
        // A DISTINCT ON key reads only where the first branch put its items,
        // which a set operation's rows keep
        let item_layout = layout.as_deref();
        // A name two of a set operation's columns carry is ambiguous, as in a derived table
        if !stmt.set_operations.is_empty() {
            // Beside a star over a join the count is unknown: every column is shown
            let shown = match expected_columns {
                0 => &columns[..],
                count => &columns[..count.min(columns.len())],
            };
            let reads = |c: &str, name: &str| {
                eq_folded(c, name)
                    || split_qualified(c).is_some_and(|(_, last)| eq_folded(last, name))
            };
            let mut twice = None;
            let registry = &self.function_registry;
            for ob in &stmt.order_by {
                if set_operation_key_column(&ob.expression, stmt, registry).is_some() {
                    continue;
                }
                visit_column_refs(&ob.expression, &mut |column| {
                    if let Expression::Identifier(id) = column {
                        let named = shown.iter().filter(|c| reads(c, &id.value_lower));
                        if twice.is_none() && named.count() > 1 {
                            twice = Some(id);
                        }
                    }
                });
            }
            if let Some(id) = twice {
                return Err(ambiguous_column(&id.value));
            }
        }
        if !stmt.order_by.is_empty() && !limit_offset_applied {
            // A set operation's rows come from other branches too: there a
            // column stands for a sort key only when aliased or an aggregate,
            // as on main, and aggregates are not read inside keys
            let set_operation = !stmt.set_operations.is_empty();
            // The rows' layout names each column's source; a set operation's
            // rows are the branches' too, so there only output names stand
            let layout = layout.as_ref().filter(|_| !set_operation);
            // Mapped once: the correlated path below asks for every row
            let (mapped_columns, aggregate_slots) =
                self.order_by_columns(stmt, &columns, layout.map(|l| &**l), set_operation);

            // Check if any ORDER BY expression needs evaluation (not just column refs or position)
            let has_complex_order_by = mapped_columns.iter().any(Option::is_none);

            // If ORDER BY has complex expressions, evaluate them and sort by keys
            if has_complex_order_by {
                // Materialize current result if needed
                let mut rows = RowVec::new();
                let mut row_id_counter = 0i64;
                while result.next() {
                    rows.push((row_id_counter, result.take_row()));
                    row_id_counter += 1;
                }
                if let Some(err) = result.last_error() {
                    return Err(err);
                }

                // Create evaluator for ORDER BY expressions
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::general_path_reached(
                    crate::test_failpoints::GeneralPath::OrderByEvaluator,
                );
                // What a sort expression reads from the row, for every
                // evaluator below: an aggregate the column computing it, a
                // name the input column at the position the layout gives it
                let mut sort_slots = aggregate_slots;
                if let Some(layout) = layout {
                    layout.sort_slots(stmt, &mut sort_slots)?;
                }
                let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                evaluator = evaluator.with_context(ctx);
                evaluator.init_columns(&columns);
                evaluator.add_aggregate_aliases(&sort_slots);

                // Check if any ORDER BY expression contains a correlated subquery
                // Use cached classification to avoid expensive AST traversal
                let has_correlated_order_by = classification.order_by_has_correlated_subqueries;

                // OPTIMIZATION: Instead of cloning rows and appending sort keys,
                // compute sort keys separately and use index-based sorting.
                // This avoids O(n * row_size) cloning overhead.
                let num_order_cols = stmt.order_by.len();

                // A subquery that reads nothing of the row is run once here:
                // its compiled placeholder cannot run on its own
                let order_exprs: Vec<std::borrow::Cow<Expression>> = stmt
                    .order_by
                    .iter()
                    .map(|ob| {
                        if Self::has_subqueries(&ob.expression)
                            && !Self::has_correlated_subqueries(&ob.expression)
                        {
                            self.process_where_subqueries(&ob.expression, ctx)
                                .map(std::borrow::Cow::Owned)
                        } else {
                            Ok(std::borrow::Cow::Borrowed(&ob.expression))
                        }
                    })
                    .collect::<Result<_>>()?;

                // Compute sort keys for each row: Vec<Vec<Value>>
                // Each inner Vec contains the evaluated ORDER BY expressions for that row
                let sort_keys: Vec<Vec<Value>> = if has_correlated_order_by {
                    // For correlated subqueries, we need to process per-row with outer context
                    let columns_arc = CompactArc::clone(&columns);
                    // Extract table alias for qualified column names
                    let order_table_alias: Option<SmartString> =
                        stmt.table_expr.as_ref().and_then(|te| match te.as_ref() {
                            crate::parser::ast::Expression::TableSource(source) => source
                                .alias
                                .as_ref()
                                .map(|a| a.value_lower.clone())
                                .or_else(|| Some(source.name.value_lower.clone())),
                            crate::parser::ast::Expression::Aliased(aliased) => {
                                Some(aliased.alias.value_lower.clone())
                            }
                            _ => None,
                        });

                    // OPTIMIZATION: Pre-compute lowercase column names once before row loop
                    // This avoids per-row to_lowercase() calls. Use CompactArc<str> for zero-cost clone.
                    let columns_lower: Vec<CompactArc<str>> = columns
                        .iter()
                        .map(|c| CompactArc::from(c.to_lowercase().as_str()))
                        .collect();
                    // Each name the outer row answers to, with its position:
                    // the output names, then each input column's names at
                    // the position the layout gives it, which win
                    let mut outer_names: Vec<(CompactArc<str>, usize)> = Vec::new();
                    let mut name_at = |name: &str, position: usize| {
                        outer_names.push((CompactArc::from(name), position));
                        if let Some(alias) = &order_table_alias {
                            if split_qualified(name).is_none() {
                                outer_names.push((
                                    CompactArc::from(qualified(alias, name).as_str()),
                                    position,
                                ));
                            }
                        }
                    };
                    for (position, name) in columns_lower.iter().enumerate() {
                        name_at(name, position);
                    }
                    if let Some(layout) = layout {
                        for (input, name) in layout.input_columns.iter().enumerate() {
                            let Some(position) =
                                layout.sources.iter().position(|s| s.input == Some(input))
                            else {
                                continue;
                            };
                            let name = name.to_lowercase();
                            name_at(&name, position);
                            if let Some((_, bare)) = split_qualified(&name) {
                                name_at(bare, position);
                            }
                        }
                    }

                    rows.iter()
                        .map(|(_, row)| {
                            // Build outer row context from current row
                            let mut outer_row_map: FxHashMap<CompactArc<str>, Value> =
                                FxHashMap::default();
                            for (name, position) in &outer_names {
                                let val =
                                    row.get(*position).cloned().unwrap_or(Value::null_unknown());
                                outer_row_map.insert(name.clone(), val);
                            }

                            // Create context with outer row for correlated subquery evaluation
                            let correlated_ctx =
                                ctx.with_outer_row(outer_row_map, columns_arc.clone());

                            evaluator.set_row_array(row);
                            stmt.order_by
                                .iter()
                                .zip(order_exprs.iter())
                                .zip(mapped_columns.iter())
                                .map(|((ob, order_expr), mapped)| {
                                    // A key the projection already put in a
                                    // column is read from it
                                    if let Some(idx) = *mapped {
                                        return Ok(row
                                            .get(idx)
                                            .cloned()
                                            .unwrap_or_else(Value::null_unknown));
                                    }
                                    // Try processing correlated subqueries first
                                    if Self::has_correlated_subqueries(&ob.expression) {
                                        let processed_expr = self.process_correlated_expression(
                                            &ob.expression,
                                            &correlated_ctx,
                                        )?;
                                        let mut corr_eval =
                                            CompiledEvaluator::new(&self.function_registry);
                                        corr_eval.init_columns(&columns);
                                        corr_eval.add_aggregate_aliases(&sort_slots);
                                        corr_eval.set_row_array(row);
                                        corr_eval = corr_eval.with_context(&correlated_ctx);
                                        corr_eval.evaluate(&processed_expr)
                                    } else {
                                        evaluator.evaluate(order_expr)
                                    }
                                })
                                .collect::<Result<Vec<Value>>>()
                        })
                        .collect::<Result<_>>()?
                } else {
                    // A key the projection already put in a column is read
                    // from it: the row no longer carries what it was
                    // computed from. The rest are compiled once, so per row
                    // only the VM runs (no row clone, no expression re-hash)
                    #[allow(clippy::type_complexity)]
                    let key_sources: Vec<(
                        Option<usize>,
                        Option<super::expression::SharedProgram>,
                    )> = order_exprs
                        .iter()
                        .zip(mapped_columns.iter())
                        .map(|(order_expr, mapped)| {
                            Ok(match *mapped {
                                Some(idx) => (Some(idx), None),
                                None => (None, Some(evaluator.compile_cached(order_expr)?)),
                            })
                        })
                        .collect::<Result<_>>()?;
                    rows.iter()
                        .map(|(_, row)| {
                            key_sources
                                .iter()
                                .map(|source| match source {
                                    (Some(idx), _) => Ok(row
                                        .get(*idx)
                                        .cloned()
                                        .unwrap_or_else(Value::null_unknown)),
                                    (None, Some(p)) => evaluator.evaluate_program(p, row),
                                    (None, None) => Ok(Value::null_unknown()),
                                })
                                .collect::<Result<Vec<Value>>>()
                        })
                        .collect::<Result<_>>()?
                };

                // Create indices and sort them based on sort_keys
                let mut indices: Vec<usize> = (0..rows.len()).collect();

                // Top-K: with LIMIT (and no DISTINCT ON, which dedups after
                // the sort), select the needed prefix in O(n) and sort only
                // that, instead of sorting all n rows
                let top_k = if stmt.distinct_on.is_empty() {
                    limit
                        .map(|l| l.saturating_add(offset))
                        .filter(|&k| k > 0 && k < indices.len())
                } else {
                    None
                };
                if let Some(k) = top_k {
                    indices.select_nth_unstable_by(k - 1, |&a_idx, &b_idx| {
                        Self::compare_by_sort_keys_impl(
                            &sort_keys[a_idx],
                            &sort_keys[b_idx],
                            &stmt.order_by,
                            num_order_cols,
                        )
                    });
                    indices.truncate(k);
                }

                // Use sort_unstable_by for ~10-20% speedup (stability not needed for ORDER BY)
                indices.sort_unstable_by(|&a_idx, &b_idx| {
                    Self::compare_by_sort_keys_impl(
                        &sort_keys[a_idx],
                        &sort_keys[b_idx],
                        &stmt.order_by,
                        num_order_cols,
                    )
                });

                // Reorder rows using sorted indices
                // OPTIMIZATION: For LIMIT queries without DISTINCT ON, only collect needed rows
                // For full results, use in-place cycle-based permutation
                // When DISTINCT ON is active, skip early LIMIT (applied after dedup)
                let has_distinct_on = !stmt.distinct_on.is_empty();
                let final_rows: RowVec = if (limit.is_some() || offset > 0) && !has_distinct_on {
                    OrderLimitPath::ComputedKeys.taken();
                    // With LIMIT/OFFSET: Only collect the rows we actually need
                    let take_count = limit.unwrap_or(usize::MAX);
                    indices
                        .into_iter()
                        .skip(offset)
                        .take(take_count)
                        .enumerate()
                        .map(|(new_idx, i)| (new_idx as i64, std::mem::take(&mut rows[i].1)))
                        .collect()
                } else {
                    // No LIMIT or DISTINCT ON active: Use in-place cycle-based permutation
                    let n = rows.len();
                    for start in 0..n {
                        // Skip if already in correct position or already processed
                        if indices[start] == start || indices[start] == usize::MAX {
                            continue;
                        }

                        // Follow the cycle
                        let mut current = start;
                        loop {
                            let target = indices[current];
                            if target == start {
                                // Cycle complete
                                indices[current] = usize::MAX; // Mark as processed
                                break;
                            }
                            rows.swap(current, target);
                            indices[current] = usize::MAX; // Mark as processed
                            current = target;
                        }
                    }
                    rows
                };

                // Project to expected columns if needed
                // When DISTINCT ON is active, keep extra columns until after dedup
                let mut result_rows = final_rows;
                let needs_extra_col_removal =
                    columns.len() > expected_columns && expected_columns > 0;
                if needs_extra_col_removal && !has_distinct_on {
                    for (_, row) in result_rows.iter_mut() {
                        row.truncate(expected_columns);
                    }
                }

                // Use original column names if expected_columns matches
                let output_columns = if needs_extra_col_removal && !has_distinct_on {
                    CompactArc::new(columns[..expected_columns].to_vec())
                } else {
                    CompactArc::clone(&columns)
                };
                let mut result: Box<dyn QueryResult> = Box::new(ExecutorResult::with_arc_columns(
                    CompactArc::clone(&output_columns),
                    result_rows,
                ));

                // Apply DISTINCT ON if active
                if has_distinct_on {
                    result = self.distinct_on_result(stmt, result, item_layout, ctx)?;

                    // Remove extra columns after DISTINCT ON dedup
                    if needs_extra_col_removal {
                        result = Box::new(ProjectedResult::new(result, expected_columns));
                    }

                    // Apply LIMIT/OFFSET after DISTINCT ON
                    if limit.is_some() || offset > 0 {
                        result = Box::new(LimitedResult::new(result, limit, offset));
                    }
                }

                return Ok(result);
            }

            // Pre-compute column indices to avoid string comparisons during sort
            // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocations
            // Tuple: (col_idx, ascending, nulls_first)
            let order_specs: Vec<(Option<usize>, bool, Option<bool>)> = stmt
                .order_by
                .iter()
                .zip(mapped_columns.iter())
                .map(|(ob, mapped)| (*mapped, ob.ascending, ob.nulls_first))
                .collect();

            // TOP-N OPTIMIZATION: Use bounded heap when LIMIT is present
            // This is O(n log k) instead of O(n log n), where k = limit
            // Skip when DISTINCT ON is active: DISTINCT ON must happen between ORDER BY and LIMIT
            if let Some(lim) = limit {
                if stmt.distinct_on.is_empty() {
                    // Use TopNResult for ORDER BY + LIMIT (5-50x faster for large datasets)
                    result = Box::new(TopNResult::new(
                        result,
                        move |a, b| Self::compare_rows_with_indices(a, b, &order_specs),
                        lim,
                        offset,
                    )?);

                    // Apply deferred projection if applicable
                    // This reduces allocations from O(matched_rows) to O(limit)
                    match deferred_projection {
                        Some(_) => OrderLimitPath::DeferredProjection.taken(),
                        None => OrderLimitPath::TopN.taken(),
                    }
                    if let Some((col_indices, output_names)) = deferred_projection {
                        result = Box::new(ProjectionResult::new(
                            result,
                            Projection::columns(col_indices),
                            output_names,
                        ));
                    } else if columns.len() > expected_columns && expected_columns > 0 {
                        // Remove extra ORDER BY columns if needed (no deferred projection)
                        result = Box::new(ProjectedResult::new(result, expected_columns));
                    }

                    // LIMIT/OFFSET already applied by TopNResult
                    return Ok(result);
                }
                // DISTINCT ON with LIMIT: fall through to full sort
                // (DISTINCT ON is applied after ORDER BY, LIMIT after DISTINCT ON)
            }
            {
                // No LIMIT - use full sort
                // OPTIMIZATION: Try radix sort for integer columns (O(n) vs O(n log n))
                // Build RadixOrderSpec only if all columns have valid indices
                let radix_specs: Vec<RadixOrderSpec> = order_specs
                    .iter()
                    .filter_map(|(col_idx, ascending, nulls_first)| {
                        col_idx.map(|idx| RadixOrderSpec {
                            col_idx: idx,
                            ascending: *ascending,
                            nulls_first: *nulls_first,
                        })
                    })
                    .collect();

                // Use radix sort if all columns have valid indices
                if radix_specs.len() == order_specs.len() {
                    // All columns have valid indices - try radix sort
                    result = Box::new(OrderedResult::new_radix(
                        result,
                        &radix_specs,
                        move |a, b| Self::compare_rows_with_indices(a, b, &order_specs),
                    )?);
                } else {
                    // Some columns missing - use comparison sort
                    result = Box::new(OrderedResult::new(result, move |a, b| {
                        Self::compare_rows_with_indices(a, b, &order_specs)
                    })?);
                }
            }
        }

        // Apply DISTINCT ON before removing extra columns, so keys can reference
        // columns not in SELECT (they may be among the extra ORDER BY columns or
        // the full source columns).
        if !stmt.distinct_on.is_empty() {
            result = self.distinct_on_result(stmt, result, item_layout, ctx)?;
        }

        // Remove extra ORDER BY columns that were added for sorting
        // This happens when ORDER BY references columns not in SELECT
        if columns.len() > expected_columns && expected_columns > 0 {
            result = Box::new(ProjectedResult::new(result, expected_columns));
        }

        // Apply LIMIT/OFFSET (only if not already applied by TopNResult or storage-level pushdown)
        if !limit_offset_applied && (limit.is_some() || offset > 0) {
            result = Box::new(LimitedResult::new(result, limit, offset));
        }

        Ok(result)
    }

    /// `result` deduplicated on `stmt`'s DISTINCT ON keys, each bound and
    /// read as an ORDER BY key is: from the column holding it, else computed
    /// for each row as the sort computes one
    fn distinct_on_result(
        &self,
        stmt: &SelectStatement,
        mut result: Box<dyn QueryResult>,
        layout: Option<&OutputLayout>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let columns = CompactArc::new(result.columns().to_vec());
        let set_operation = !stmt.set_operations.is_empty();
        // A set operation's rows keep where its first branch put its items,
        // which the layout says; only without one do output names stand
        let (mut mapped, mut slots) = self.key_columns(
            stmt,
            stmt.distinct_on.iter(),
            &columns,
            layout,
            set_operation && layout.is_none(),
        );
        // A position names one of the columns the SELECT list shows
        let shown = layout.map_or(columns.len(), |l| l.shown(stmt.columns.len()));
        for (column, key) in mapped.iter_mut().zip(&stmt.distinct_on) {
            if let Expression::IntegerLiteral(lit) = key {
                *column = column.filter(|&c| lit.value >= 1 && c < shown);
            }
        }
        if let Some(indices) = mapped.iter().copied().collect::<Option<Vec<_>>>() {
            return Ok(Box::new(DistinctOnResult::new(result, indices)));
        }
        if let Some(layout) = layout {
            layout.sort_slots(stmt, &mut slots)?;
        }
        let outer = outer_column_names(ctx);
        // Each key no column holds is computed into a column appended to the
        // row, which the dedup reads and the projection below drops
        let width = columns.len();
        let mut programs: Vec<SharedProgram> = Vec::new();
        let mut indices = Vec::with_capacity(mapped.len());
        for (key, column) in stmt.distinct_on.iter().zip(mapped) {
            if let Some(column) = column {
                indices.push(column);
                continue;
            }
            // A set operation's branch carries no key column for its rows
            if set_operation || matches!(bind_order_key(key, stmt), OrderKey::Ordinal(_)) {
                return Err(Error::invalid_argument(format!(
                    "DISTINCT ON key {key} names no column"
                )));
            }
            // A subquery that reads nothing of the row is run once here, as
            // an ORDER BY key's is
            let key = match Self::has_subqueries(key) && !Self::has_correlated_subqueries(key) {
                true => std::borrow::Cow::Owned(self.process_where_subqueries(key, ctx)?),
                false => std::borrow::Cow::Borrowed(key),
            };
            indices.push(width + programs.len());
            programs.push(super::aggregation::compile_over_slots(
                &key,
                &columns,
                &self.function_registry,
                Some(&slots),
                outer.as_deref(),
                false,
                Some(ctx.params()),
            )?);
        }
        let mut vm = ExprVM::new();
        let mut rows = RowVec::new();
        let mut row_id = 0i64;
        while result.next() {
            let mut row = result.take_row();
            for program in &programs {
                let value = vm.execute(program, &super::aggregation::slot_context(&row, ctx))?;
                row.push(value);
            }
            rows.push((row_id, row));
            row_id += 1;
        }
        if let Some(err) = result.last_error() {
            return Err(err);
        }
        let mut keyed_columns = columns.to_vec();
        keyed_columns.extend(
            stmt.distinct_on
                .iter()
                .zip(&indices)
                .filter(|&(_, &index)| index >= width)
                .map(|(key, _)| ExprKey::of(key).hidden_column_name()),
        );
        let keyed = Box::new(ExecutorResult::new(keyed_columns, rows));
        Ok(Box::new(ProjectedResult::new(
            Box::new(DistinctOnResult::new(keyed, indices)),
            width,
        )))
    }

    /// The SELECT item computing `key`, aliased or not
    fn item_computing(stmt: &SelectStatement, key: &Expression) -> Option<usize> {
        stmt.columns
            .iter()
            .position(|c| ExprKey::of(unaliased_item(c)) == ExprKey::of(key))
    }

    /// `stmt` with each DISTINCT ON key that is no SELECT position or alias
    /// appended as a hidden item named by its identity, when the aggregation
    /// or window path computes its rows: a key is then computed where its row
    /// is, reading groups, aggregates and windows as a SELECT item does
    pub(crate) fn with_distinct_on_items(
        stmt: &SelectStatement,
        classification: &QueryClassification,
    ) -> Option<SelectStatement> {
        if stmt.distinct_on.is_empty()
            || !stmt.set_operations.is_empty()
            || !(classification.has_aggregation || classification.has_window_functions)
        {
            return None;
        }
        // A position past the SELECT list stays an error, not a hidden item;
        // beside a star the list's width is the star's, which ORDER BY reads
        let star = stmt.columns.iter().any(is_star_item);
        let past_list = |expr: &Expression| {
            matches!(expr, Expression::IntegerLiteral(lit)
                if usize::try_from(lit.value).map_or(true, |p| p > stmt.columns.len()))
        };
        let order_by = stmt
            .order_by
            .iter()
            .map(|ob| &ob.expression)
            .filter(|_| !star);
        if super::aggregation::grouping_items(stmt)
            .into_iter()
            .chain(order_by)
            .any(past_list)
        {
            return None;
        }
        let mut columns = stmt.columns.clone();
        for key in &stmt.distinct_on {
            // A key the SELECT list computes is read from that item
            if Self::item_computing(stmt, key).is_some() {
                continue;
            }
            let alias = |id: &Identifier| {
                stmt.columns.iter().any(|c| {
                    matches!(c, Expression::Aliased(a) if a.alias.value_lower == id.value_lower)
                })
            };
            match key {
                Expression::IntegerLiteral(_) => continue,
                Expression::Identifier(id) if alias(id) => continue,
                _ => {}
            }
            let name = ExprKey::of(key).hidden_column_name();
            if columns
                .iter()
                .any(|c| matches!(c, Expression::Aliased(a) if a.alias.value == name))
            {
                continue;
            }
            let token = dummy_token(&name, TokenType::Identifier);
            columns.push(Expression::Aliased(AliasedExpression {
                token: token.clone(),
                expression: Box::new(key.clone()),
                alias: Identifier::new(token, name),
            }));
        }
        (columns.len() > stmt.columns.len()).then(|| SelectStatement {
            columns,
            ..stmt.clone()
        })
    }

    /// Count the number of columns in the SELECT clause
    pub(crate) fn count_select_columns(&self, stmt: &SelectStatement) -> usize {
        // Check for SELECT * or SELECT t.* anywhere in the select list
        // If there's any Star or QualifiedStar, we can't determine the exact count
        // without knowing the table columns, so return 0 to disable projection truncation
        for col in &stmt.columns {
            if matches!(col, Expression::Star(_) | Expression::QualifiedStar(_)) {
                return 0; // Don't project - star expansion makes count unknown
            }
        }
        stmt.columns.len()
    }

    /// Execute the core SELECT logic
    /// Returns (result, columns, limit_offset_applied)
    /// The third value indicates if LIMIT/OFFSET was already applied at the storage level
    pub(crate) fn execute_select_internal(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        compiled: Option<&NestedPlan>,
    ) -> SelectResult {
        // Get table source
        let table_expr = match &stmt.table_expr {
            Some(expr) => expr.as_ref(),
            None => {
                // SELECT without FROM (e.g., SELECT 1+1)
                return self.execute_expression_select(stmt, ctx, classification);
            }
        };

        // Execute based on table source type
        match table_expr {
            Expression::TableSource(table_source) => {
                // Check if this is a CTE from context (for subqueries referencing outer CTEs)
                let table_name = &table_source.name.value_lower;
                if let Some((columns, rows)) = ctx.get_cte_by_lower(table_name) {
                    if table_source.as_of.is_some() {
                        return Err(Self::as_of_not_a_table(table_name));
                    }
                    // Execute query against CTE data
                    // Dereference Arc to get &Vec<(i64, Row)>, then wrap in RowVec
                    return self.execute_query_on_memory_result(
                        stmt,
                        ctx,
                        columns.to_vec(),
                        RowVec::from_vec((**rows).clone()),
                    );
                }

                // Check if this is actually a view (single lookup, no double RwLock acquisition)
                if let Some(view_def) = self.engine.get_view_lowercase(table_name)? {
                    if table_source.as_of.is_some() {
                        return Err(Self::as_of_not_a_table(table_name));
                    }
                    return self.execute_view_query(&view_def, stmt, ctx, classification);
                }
                self.execute_simple_table_scan(table_source, stmt, ctx, classification, None, None)
            }
            Expression::JoinSource(join_source) => {
                let slot = compiled.map(NestedPlan::slot);
                self.execute_join_source(join_source, stmt, ctx, classification, slot, false)
            }
            Expression::SubquerySource(subquery_source) => {
                let plan = compiled.and_then(|plan| plan.from(stmt));
                self.execute_subquery_source(subquery_source, stmt, ctx, classification, plan)
            }
            Expression::ValuesSource(values_source) => {
                self.execute_values_source(values_source, stmt, ctx, classification)
            }
            Expression::FunctionTableSource(tvf_source) => {
                self.execute_tvf_source(tvf_source, stmt, ctx, classification)
            }
            _ => Err(Error::NotSupported(
                "Unsupported FROM clause type".to_string(),
            )),
        }
    }

    /// Evaluate a TVF source: look up function, evaluate args, generate rows, determine columns.
    /// Execute a TVF for use in a join context: generate rows, build qualified columns,
    /// and apply an optional filter. Shared by execute_table_expression_with_filter
    /// and execute_table_expression_with_filter_limit.
    fn execute_tvf_for_join(
        &self,
        tvf_source: &FunctionTableSource,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        // Only push limit to TVF generation when no filter will discard rows afterwards.
        // With a filter, we must generate all rows first, then filter, then let
        // downstream processing handle the limit. The flag says whether the
        // limit and offset were applied here. A filter that keeps no row
        // generates none.
        let tvf_limit = match filter {
            Some(filter) if true_for_no_row(filter, ctx, &self.function_registry) => Some(0),
            Some(_) => None,
            None => limit.map(|l| l.saturating_add(offset)),
        };

        // Extract range bounds from the filter to narrow TVF generation range.
        // This avoids materializing millions of rows when a selective predicate exists.
        let range_hint = filter.and_then(|filter_expr| {
            let col_name: SmartString = if !tvf_source.column_aliases.is_empty() {
                tvf_source.column_aliases[0].value_lower.clone()
            } else {
                SmartString::from("value")
            };
            Self::tvf_range_hint(filter_expr, &col_name, ctx)
        });

        let (mut result_rows, column_names) =
            Self::evaluate_tvf_with_range(tvf_source, None, ctx, tvf_limit, range_hint)?;

        let table_alias = tvf_source
            .alias
            .as_ref()
            .map(|a| a.value.to_string())
            .unwrap_or_else(|| tvf_source.function.value.to_string());

        let qualified_columns: Vec<String> = column_names
            .iter()
            .map(|col| qualified(&table_alias, col))
            .collect();

        if let Some(filter_expr) = filter {
            let row_filter = RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
            row_filter.retain_checked(&mut result_rows)?;
        }
        let bounded = tvf_limit.is_some();
        if bounded && offset > 0 {
            let mut rows = result_rows.into_vec();
            rows.drain(..offset.min(rows.len()));
            result_rows = RowVec::from_vec(rows);
        }

        let result: Box<dyn QueryResult> = Box::new(ExecutorResult::new(column_names, result_rows));
        Ok((result, qualified_columns, bounded))
    }

    /// LIMIT + OFFSET of a positive LIMIT, for fetch paths that stop early and
    /// leave OFFSET to be applied afterwards; None when either is not a count,
    /// which the general path reports
    pub(crate) fn fetch_limit(stmt: &SelectStatement, ctx: &ExecutionContext) -> Option<usize> {
        let counts = row_counts(stmt, ctx).ok()?;
        let limit = counts.limit.filter(|limit| *limit > 0)?;
        Some(limit.saturating_add(counts.offset))
    }

    /// Execute a table-valued function source (e.g., generate_series(1, 10))
    fn execute_tvf_source(
        &self,
        tvf_source: &FunctionTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &Arc<QueryClassification>,
    ) -> SelectResult {
        // Push LIMIT down to TVF generation only when nothing can filter/reshape/reorder:
        // WHERE, GROUP BY, HAVING, ORDER BY, DISTINCT, aggregation, window functions, set operations
        let limit_hint = if stmt.where_clause.is_none()
            && stmt.group_by.columns.is_empty()
            && stmt.having.is_none()
            && stmt.order_by.is_empty()
            && !stmt.distinct
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_set_operations
        {
            rows_needed(stmt, ctx)
        } else {
            None
        };

        let (result_rows, columns) = self.tvf_rows(
            tvf_source,
            None,
            stmt.where_clause.as_deref(),
            limit_hint,
            ctx,
        )?;

        // Delegate to execute_query_on_memory_result which handles
        // WHERE, ORDER BY, LIMIT, GROUP BY, aggregation, window functions
        self.execute_query_on_memory_result(stmt, ctx, columns, result_rows)
    }

    /// The rows and columns a TVF source gives a statement whose WHERE is
    /// `where_clause`: none when it holds for no row, else the range its
    /// conjuncts on the TVF's column allow, up to `limit`
    pub(super) fn tvf_rows(
        &self,
        tvf_source: &FunctionTableSource,
        arguments: Option<&dyn Fn() -> Result<Vec<Value>>>,
        where_clause: Option<&Expression>,
        limit: Option<usize>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        // Short-circuit: if WHERE is a constant-false expression (e.g., 1=0),
        // skip TVF generation entirely and return empty result.
        if let Some(where_clause) = where_clause {
            if true_for_no_row(where_clause, ctx, &self.function_registry) {
                return Ok((RowVec::new(), Self::tvf_columns(tvf_source)?));
            }
        }

        // Extract range bounds from WHERE clause to narrow TVF generation range.
        // This avoids materializing millions of rows when a selective predicate exists.
        let range_hint = where_clause.and_then(|where_clause| {
            // Determine the TVF value column name (from alias or default "value")
            let col_name: SmartString = if !tvf_source.column_aliases.is_empty() {
                tvf_source.column_aliases[0].value_lower.clone()
            } else {
                SmartString::from("value")
            };
            Self::tvf_range_hint(where_clause, &col_name, ctx)
        });

        Self::evaluate_tvf_with_range(tvf_source, arguments, ctx, limit, range_hint)
    }

    /// The integers the conjuncts of `filter` on the TVF's column allow it to
    /// generate. A hint: the filter runs on every row generated, so a conjunct
    /// the keys cannot answer narrows nothing; None when none can
    fn tvf_range_hint(
        filter: &Expression,
        col_name: &str,
        ctx: &ExecutionContext,
    ) -> Option<Resolved> {
        let constant = |expr: &Expression| {
            if !statement_constant(expr, global_registry()) {
                return None;
            }
            super::expression::evaluate_once(expr, Some(ctx)).ok()
        };
        let mut comparisons = Vec::new();
        for conjunct in extract_and_conditions(filter) {
            match conjunct {
                Expression::Infix(infix) => {
                    let Some(op) = infix_to_operator(infix.op_type) else {
                        continue;
                    };
                    let (op, value) = if Self::is_tvf_column(&infix.left, col_name) {
                        (op, &*infix.right)
                    } else if Self::is_tvf_column(&infix.right, col_name) {
                        (flip_operator(op), &*infix.left)
                    } else {
                        continue;
                    };
                    comparisons.extend(constant(value).map(|value| (op, value)));
                }
                Expression::Between(between)
                    if !between.not && Self::is_tvf_column(&between.expr, col_name) =>
                {
                    let (gte, lte) = (crate::core::Operator::Gte, crate::core::Operator::Lte);
                    comparisons.extend(constant(&between.lower).map(|v| (gte, v)));
                    comparisons.extend(constant(&between.upper).map(|v| (lte, v)));
                }
                _ => {}
            }
        }
        resolve_known(
            Domain::Integer,
            comparisons.iter().map(|(op, value)| (*op, value)),
        )
    }

    /// Check if an expression refers to the TVF's output column.
    fn is_tvf_column(expr: &Expression, col_name: &str) -> bool {
        match expr {
            Expression::Identifier(id) => id.value_lower.eq_ignore_ascii_case(col_name),
            Expression::QualifiedIdentifier(qi) => {
                qi.name.value_lower.eq_ignore_ascii_case(col_name)
            }
            _ => false,
        }
    }

    /// Evaluate a TVF with optional range clamping on start/stop args.
    /// The table-valued function a TVF source calls
    fn tvf_of(
        tvf_source: &FunctionTableSource,
    ) -> Result<Box<dyn crate::functions::tvf::TableValuedFunction>> {
        let func_name = &tvf_source.function.value;
        crate::functions::global_registry()
            .get_tvf(func_name)
            .ok_or_else(|| {
                Error::NotSupported(format!("Unknown table-valued function: {}", func_name))
            })
    }

    /// The columns a TVF source gives: its aliases, or the function's own
    pub(super) fn tvf_columns(tvf_source: &FunctionTableSource) -> Result<Vec<String>> {
        Ok(Self::tvf_source_columns(
            tvf_source,
            &*Self::tvf_of(tvf_source)?,
        ))
    }

    fn tvf_source_columns(
        tvf_source: &FunctionTableSource,
        tvf: &dyn crate::functions::tvf::TableValuedFunction,
    ) -> Vec<String> {
        if !tvf_source.column_aliases.is_empty() {
            tvf_source
                .column_aliases
                .iter()
                .map(|id| id.value.to_string())
                .collect()
        } else {
            tvf.column_names()
        }
    }

    /// The rows of `tvf_source` over the values `arguments` computes, or
    /// over its arguments as written
    fn evaluate_tvf_with_range(
        tvf_source: &FunctionTableSource,
        arguments: Option<&dyn Fn() -> Result<Vec<Value>>>,
        ctx: &ExecutionContext,
        limit: Option<usize>,
        range_hint: Option<Resolved>,
    ) -> Result<(RowVec, Vec<String>)> {
        let tvf = Self::tvf_of(tvf_source)?;

        let mut arg_values = match arguments {
            Some(compute) => compute()?,
            None => {
                // Evaluate arguments to Values
                let mut values = Vec::with_capacity(tvf_source.arguments.len());
                for arg in &tvf_source.arguments {
                    values.push(super::expression::evaluate_once(arg, Some(ctx))?);
                }
                values
            }
        };

        if let Some(range) = range_hint {
            // An empty range narrows past both ends
            let (low, high) = match range {
                Resolved::Key(k) => (k, k),
                Resolved::Range(low, high) => (low, high),
                Resolved::Empty => (i64::MAX, i64::MIN),
                Resolved::Fallback => (i64::MIN, i64::MAX),
            };
            tvf.narrow(&mut arg_values, low, high);
        }

        let result_rows = tvf.generate(&arg_values, limit)?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::tvf_rows_generated(result_rows.len());

        Ok((result_rows, Self::tvf_source_columns(tvf_source, &*tvf)))
    }

    /// Execute SELECT without FROM (expressions only)
    fn execute_expression_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

        // The one row without columns, unless the WHERE keeps it out
        let mut passes = true;
        if let Some(where_clause) = &stmt.where_clause {
            // Pre-process subqueries in WHERE clause (EXISTS, IN, ALL/ANY, scalar subqueries)
            let processed_where = self.process_where_subqueries(where_clause, ctx)?;
            // A parent row binds the outer references of a FROM-less subquery
            let processed_where = match ctx.outer_row() {
                Some(outer) => super::utils::substitute_outer_references(&processed_where, outer),
                None => processed_where,
            };
            let where_result = super::expression::evaluate_once(&processed_where, Some(ctx))?;
            passes = ExprVM::passes(&where_result);
        }

        // Aggregates and windows read the row the WHERE passed, as in
        // SELECT SUM(3+5); over none, an aggregate still answers one row
        if classification.has_aggregation || classification.has_window_functions {
            let mut rows = RowVec::new();
            if passes {
                rows.push((0, Row::from_values(vec![])));
            }
            return self.grouped_select(stmt, ctx, classification, rows, &[], None);
        }
        if !passes {
            // Still need to determine column names for the schema
            let columns = CompactArc::new(stmt.columns.iter().map(item_name).collect());
            let result =
                ExecutorResult::with_arc_columns(CompactArc::clone(&columns), RowVec::new());
            let shape = SelectShape::laid_out(Self::result_layout(stmt, &[], None, &columns));
            return Ok((Box::new(result), columns, false, shape));
        }

        // Process scalar subqueries in SELECT columns first (single-pass)
        let processed_columns = self.try_process_select_subqueries(&stmt.columns, ctx)?;
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (col_expr, written) in columns_to_use.iter().zip(&stmt.columns) {
            // Named as written, not as its subqueries were read
            columns.push(item_name(written));

            // Evaluate expression, with the parent row binding outer references
            let bound = ctx
                .outer_row()
                .map(|outer| super::utils::substitute_outer_references(col_expr, outer));
            let value =
                super::expression::evaluate_once(bound.as_ref().unwrap_or(col_expr), Some(ctx))?;
            values.push(value);
        }

        let row = Row::from_values(values);
        let columns = CompactArc::new(columns);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&columns), rows);

        let shape = SelectShape::laid_out(Self::result_layout(stmt, &[], None, &columns));
        Ok((Box::new(result), columns, false, shape))
    }

    /// Execute a query against in-memory data (for CTEs referenced from context)
    fn execute_query_on_memory_result(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        columns: Vec<String>,
        rows: RowVec,
    ) -> SelectResult {
        // ORDER BY is the caller's, which binds it by the layout, and so is
        // LIMIT after it; DISTINCT reads every row before a LIMIT counts them
        let skip_order_limit = !stmt.order_by.is_empty() || stmt.distinct;

        let (result_cols, result_rows, order_limit_applied, layout) =
            self.execute_query_on_cte_result_inner(stmt, ctx, columns, rows, skip_order_limit)?;
        let shape = SelectShape::laid_out(layout);
        let result_cols = CompactArc::new(result_cols);
        Ok((
            Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(&result_cols),
                result_rows,
            )),
            result_cols,
            order_limit_applied,
            shape,
        ))
    }

    /// Check if ORDER BY or DISTINCT ON references columns not in SELECT
    /// OPTIMIZATION: Use HashSet for O(1) lookup and eq_ignore_ascii_case to avoid allocations
    fn order_by_needs_extra_columns(&self, stmt: &SelectStatement, all_columns: &[String]) -> bool {
        if stmt.order_by.is_empty() && stmt.distinct_on.is_empty() {
            return false;
        }

        // Check if SELECT contains * or t.* - it includes all columns, so ORDER BY is always covered
        // This handles both "SELECT *" and "SELECT *, expr" cases
        let has_star = stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::Star(_)));
        if has_star {
            return false;
        }

        // For t.*, check if all ORDER BY columns are covered by the qualified star
        let has_qualified_star = stmt.columns.iter().any(|c| {
            if let Expression::QualifiedStar(qs) = c {
                // Check if the sort keys match this qualifier
                sort_keys(stmt).all(|key| {
                    if let Expression::QualifiedIdentifier(qid) = key {
                        qid.qualifier.value_lower == qs.qualifier.to_lowercase()
                    } else if let Expression::Identifier(_) = key {
                        // Simple identifier might be covered by qualified star
                        true
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        });
        if has_qualified_star {
            return false;
        }

        // Get SELECT column names (lowercase) using HashSet for O(1) lookup
        // Include both unqualified and fully qualified names for disambiguation
        let mut select_columns: FxHashSet<String> = stmt
            .columns
            .iter()
            .filter_map(|expr| self.extract_select_column_name(expr))
            .map(|s| s.to_lowercase())
            .collect();
        // Also add qualified names for join column disambiguation
        for expr in &stmt.columns {
            match expr {
                Expression::QualifiedIdentifier(qi) => {
                    select_columns
                        .insert(qualified(&qi.qualifier.value_lower, &qi.name.value_lower));
                }
                Expression::Aliased(a) => {
                    if let Expression::QualifiedIdentifier(qi) = &*a.expression {
                        select_columns
                            .insert(qualified(&qi.qualifier.value_lower, &qi.name.value_lower));
                    }
                }
                _ => {}
            }
        }

        // A correlated subquery in a sort key reads the parent row, all of
        // it, so every column is carried until the sort has run
        if sort_keys(stmt).any(Self::has_correlated_subqueries) {
            return true;
        }

        // Check if any ORDER BY or DISTINCT ON key is not in SELECT
        for key in sort_keys(stmt) {
            match key {
                Expression::Identifier(id)
                    if !select_columns.contains(id.value_lower.as_str())
                        && all_columns
                            .iter()
                            .any(|c| c.eq_ignore_ascii_case(id.value_lower.as_str())) =>
                {
                    return true;
                }
                Expression::QualifiedIdentifier(qi)
                    if reads_input(stmt.table_expr.as_deref(), qi) =>
                {
                    let full_name = qualified(&qi.qualifier.value_lower, &qi.name.value_lower);
                    if !select_columns.contains(full_name.as_str())
                        && all_columns.iter().any(|c| {
                            c.eq_ignore_ascii_case(&full_name)
                                || c.eq_ignore_ascii_case(qi.name.value_lower.as_str())
                        })
                    {
                        return true;
                    }
                }
                Expression::IntegerLiteral(_) => {}
                // An expression is read from a column of its own, unless a
                // SELECT item computes it. A subquery reads
                // the parent row per row, which the sort does for itself
                other
                    if !matches!(
                        other,
                        Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                    ) && !Self::has_subqueries(other) =>
                {
                    let key = ExprKey::of(other);
                    if !stmt
                        .columns
                        .iter()
                        .any(|item| ExprKey::of(unaliased_item(item)) == key)
                    {
                        return true;
                    }
                }
                // One the sort evaluates reads its input columns from the row
                other if Self::has_subqueries(other) => {
                    let mut names_input = false;
                    visit_expression_inputs(other, stmt.table_expr.as_deref(), &mut |_, key| {
                        if let OrderKey::Input(qualifier, name) = key {
                            names_input |=
                                input_column_index(all_columns, qualifier, name).is_some();
                        }
                    });
                    if names_input {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }

    /// Extract column name from expression for ORDER BY handling
    #[allow(clippy::only_used_in_recursion)]
    fn extract_select_column_name(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value.to_string()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value.to_string()),
            Expression::Aliased(aliased) => self.extract_select_column_name(&aliased.expression),
            Expression::Star(_) | Expression::QualifiedStar(_) => None, // SELECT * or t.* includes all columns
            _ => None,
        }
    }

    /// Check if deferred projection optimization is applicable.
    ///
    /// For ORDER BY + LIMIT queries where SELECT columns are simple column references,
    /// we can defer projection until after sorting and limiting. This reduces allocations
    /// from O(matched_rows) to O(limit).
    ///
    /// Returns (column_indices, output_column_names) if optimization applies, None otherwise.
    fn get_deferred_projection_info(
        &self,
        stmt: &SelectStatement,
        source_columns_lower: &[String],
        classification: &std::sync::Arc<QueryClassification>,
    ) -> Option<(Vec<usize>, Vec<String>)> {
        // Must have ORDER BY + LIMIT
        if stmt.order_by.is_empty() || stmt.limit.is_none() {
            return None;
        }

        // No aggregation or window functions (these are handled separately)
        if classification.has_aggregation || classification.has_window_functions {
            return None;
        }

        // No DISTINCT — it reads the columns the SELECT asked for, and
        // deferring the projection would leave it reading the whole source
        // row, where rows that project alike are still telling apart
        if classification.has_distinct_on || classification.has_distinct {
            return None;
        }

        // ORDER BY columns must exist in source columns (so sorting can happen before projection)
        for ob in &stmt.order_by {
            if order_by_names_an_alias(&ob.expression, stmt) {
                return None;
            }
            let col_exists = match &ob.expression {
                Expression::Identifier(id) => source_columns_lower
                    .iter()
                    .any(|c| c == id.value_lower.as_str()),
                Expression::QualifiedIdentifier(qid) => source_columns_lower
                    .iter()
                    .any(|c| c == qid.name.value_lower.as_str()),
                _ => false, // Complex expression - can't evaluate on source columns
            };
            if !col_exists {
                return None;
            }
        }

        // Calculate projection indices - all SELECT columns must be simple column references
        let mut indices = Vec::with_capacity(stmt.columns.len());
        let mut output_names = Vec::with_capacity(stmt.columns.len());

        for expr in &stmt.columns {
            // A star does not benefit; a computed item needs the evaluator
            let column = match unaliased_item(expr) {
                Expression::Identifier(id) => id.value_lower.as_str(),
                Expression::QualifiedIdentifier(qid) => qid.name.value_lower.as_str(),
                _ => return None,
            };
            indices.push(source_columns_lower.iter().position(|c| c == column)?);
            output_names.push(item_name(expr));
        }

        Some((indices, output_names))
    }

    /// The answer of a DISTINCT statement the plan's rule reads the distinct
    /// values of, for one run without a plan; None when the rule does not
    /// take it or the table keeps no values to read
    fn distinct_values_answer(
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        classification: &QueryClassification,
    ) -> Result<Option<Box<dyn crate::storage::traits::QueryResult>>> {
        let Some(column) = distinct_values_column(stmt, classification, table.schema()) else {
            return Ok(None);
        };
        let Some(values) = Self::distinct_column_values(table, &column)? else {
            return Ok(None);
        };
        let rows: RowVec = values
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as i64, Row::from_values(vec![v])))
            .collect();
        let name = item_name(&stmt.columns[0]);
        Ok(Some(Box::new(ExecutorResult::new(vec![name], rows))))
    }

    /// The distinct values of `column`, lowercase, the table keeps in its
    /// index or volumes; None when it keeps none
    pub(crate) fn distinct_column_values(
        table: &dyn crate::storage::traits::Table,
        column: &str,
    ) -> Result<Option<Vec<Value>>> {
        let values = match table.get_partition_values(column)? {
            Some(values) => Some(values),
            None => match table.schema().column_index_map().get(column) {
                Some(&col_idx) => table.compute_distinct_values(col_idx)?,
                None => None,
            },
        };
        if values.is_some() {
            super::result::DistinctPath::Values.taken();
        }
        Ok(values)
    }

    /// The semantic cache answer to a read committed autocommit `SELECT *`
    /// over one base table, looked up before the statement takes its view.
    /// Only a statement nothing completes or reshapes after its scan is tried
    pub(crate) fn semantic_cache_before_view(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
    ) -> Result<EarlySemantic> {
        let source = match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(source)) if source.as_of.is_none() => source,
            _ => return Ok(EarlySemantic::NotTried),
        };
        let Some(where_expr) = stmt.where_clause.as_deref() else {
            return Ok(EarlySemantic::NotTried);
        };
        if !classification.semantic_cacheable(ctx.literal_params())
            || classification.has_offset
            || classification.has_cte
            || classification.has_set_operations
            || classification.has_having
            || classification.has_distinct_on
            || ctx.statement_view().is_some()
            || ctx.statement_snapshot().is_some()
            || ctx.outer_row().is_some()
            || ctx.timeout_ms() > 0
            || ctx.query_depth > 0
            || expression_contains_aggregate(where_expr)
            || self.isolation_level() != crate::core::IsolationLevel::ReadCommitted
        {
            return Ok(EarlySemantic::NotTried);
        }
        let table_name = source.name.value_lower.as_str();
        if ctx.has_cte_by_lower(table_name) || self.engine.get_view_lowercase(table_name)?.is_some()
        {
            return Ok(EarlySemantic::NotTried);
        }
        ctx.check_cancelled()?;
        let Some(statement) = std::num::NonZeroUsize::new(stmt as *const _ as usize) else {
            return Ok(EarlySemantic::NotTried);
        };
        let miss = SemanticMiss {
            statement,
            simplified: ExpressionSimplifier::new()
                .try_simplify(where_expr)
                .map(Arc::new),
        };
        let Some((stamp, all_columns)) =
            self.semantic_cache_stamp(table_name, None, |store| store.schema().column_names_arc())
        else {
            return Ok(EarlySemantic::Miss(miss));
        };
        let alias = source
            .alias
            .as_ref()
            .map_or(table_name, |alias| alias.value_lower.as_str());
        let predicate = miss.simplified.as_deref().unwrap_or(where_expr);
        let params = classification.where_has_parameters.then(|| ctx.params());
        match self.semantic_cache_answer(
            table_name,
            stmt,
            &all_columns,
            Some(alias),
            predicate,
            params,
            stamp,
        )? {
            Some((result, _)) => Ok(EarlySemantic::Hit(result)),
            None => Ok(EarlySemantic::Miss(miss)),
        }
    }

    /// The cached rows of `SELECT *` over `table_name` filtered by
    /// `where_expr` under `stamp`, with their output columns; None on a miss
    /// Whether the semantic cache keeps the answer of `stmt`, a SELECT * read
    /// from `source` outside an explicit transaction, by reads no older than
    /// a stamp taken before them, as a cached subquery's are; a CTE of the
    /// same name is not the table the stamp names
    pub(crate) fn semantic_writable(
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
        source: &SimpleTableSource,
    ) -> bool {
        Self::reads_cacheable_rows(ctx)
            && !ctx.has_cte_by_lower(&source.name.value_lower)
            && ctx.outer_row().is_none()
            && source.as_of.is_none()
            && stmt.where_clause.is_some()
            && classification.semantic_cacheable(ctx.literal_params())
    }

    /// Whether `table`'s rows may fit a semantic cache entry, so collecting
    /// an answer over it for the cache can pay off; an upper bound, so a
    /// larger table's answers stream
    pub(crate) fn semantic_rows_fit(table: &dyn crate::storage::traits::Table) -> bool {
        table.row_count_hint() <= super::semantic_cache::DEFAULT_MAX_CACHED_ROWS
    }

    /// The stamp a semantic cache entry of `table_name` is written under,
    /// taken before the rows it keeps are read, so an entry is never older
    /// than its stamp; None while the cache cannot take one
    pub(crate) fn semantic_write_stamp(
        &self,
        table_name: &str,
        ctx: &ExecutionContext,
    ) -> Option<super::semantic_cache::CacheStamp> {
        self.semantic_cache_stamp(table_name, ctx.scan_view(table_name), |_| ())
            .map(|(stamp, ())| stamp)
    }

    /// Keeps `rows`, the answer over the columns `columns` of `table_name`
    /// to a WHERE the cache's lookup reads as `predicate`, under `stamp`;
    /// `params` are the values of a predicate holding parameters
    pub(crate) fn semantic_write<'r>(
        &self,
        table_name: &str,
        columns: &[String],
        rows: impl Iterator<Item = &'r Row>,
        predicate: &Expression,
        params: Option<&[Value]>,
        stamp: super::semantic_cache::CacheStamp,
    ) {
        let predicate = match params {
            Some(params) => match super::prepared_select::parameters_as_literals(predicate, params)
            {
                Some(bound) => bound,
                None => return,
            },
            None => predicate.clone(),
        };
        self.semantic_cache.insert(
            table_name,
            columns.to_vec(),
            rows.cloned().collect(),
            Some(predicate),
            stamp,
        );
    }

    /// The cached answer to `stmt`, a SELECT * of `source` filtered by
    /// `where_expr`, shaped as the table scan shapes the answers it takes
    /// from the cache; None on a miss
    fn semantic_source_answer(
        &self,
        source: &SimpleTableSource,
        stmt: &SelectStatement,
        where_expr: &Expression,
        params: Option<&[Value]>,
        stamp: super::semantic_cache::CacheStamp,
        ctx: &ExecutionContext,
    ) -> Result<Option<SelectOutput>> {
        let table_name = &source.name.value_lower;
        let table = self.prepared_table(table_name, ctx, None)?;
        let all_columns = table.schema().column_names_arc();
        let alias = source.alias.as_ref().map_or(table_name, |a| &a.value_lower);
        let Some((result, columns)) = self.semantic_cache_answer(
            table_name,
            stmt,
            &all_columns,
            Some(alias),
            where_expr,
            params,
            stamp,
        )?
        else {
            return Ok(None);
        };
        let shape = SelectShape::laid_out(Self::shared_result_layout(
            stmt,
            &all_columns,
            Some(alias),
            &columns,
        ));
        Ok(Some((result, columns, false, shape)))
    }

    /// The cached answer to `stmt` whose WHERE the cache reads as
    /// `where_expr`; `params` are the values of one holding parameters,
    /// which the cache reads as the literals they are written as
    #[allow(clippy::too_many_arguments)]
    fn semantic_cache_answer(
        &self,
        table_name: &str,
        stmt: &SelectStatement,
        all_columns: &[String],
        table_alias: Option<&str>,
        where_expr: &Expression,
        params: Option<&[Value]>,
        stamp: super::semantic_cache::CacheStamp,
    ) -> Result<Option<CachedAnswer>> {
        use super::semantic_cache::{CacheLookupResult, SemanticCache};
        let bound;
        let where_expr = match params {
            Some(params) => {
                match super::prepared_select::parameters_as_literals(where_expr, params) {
                    Some(expr) => {
                        bound = expr;
                        &bound
                    }
                    None => return Ok(None),
                }
            }
            None => where_expr,
        };
        let output_columns = || {
            CompactArc::new(self.get_output_column_names(&stmt.columns, all_columns, table_alias))
        };
        match self
            .semantic_cache
            .lookup(table_name, all_columns, Some(where_expr), stamp)
        {
            CacheLookupResult::ExactHit(rows) => {
                // Zero-copy sharing of the cached rows
                let columns = output_columns();
                let result =
                    ExecutorResult::with_arc_columns_shared_rows(CompactArc::clone(&columns), rows);
                Ok(Some((Box::new(result), columns)))
            }
            CacheLookupResult::SubsumptionHit {
                rows,
                filter,
                columns,
            } => {
                let filtered: RowVec = SemanticCache::filter_rows(
                    (*rows).clone(),
                    &filter,
                    &columns,
                    &self.function_registry,
                )?
                .into_iter()
                .enumerate()
                .map(|(i, row)| (i as i64, row))
                .collect();
                let columns = output_columns();
                let result =
                    ExecutorResult::with_arc_columns(CompactArc::clone(&columns), filtered);
                Ok(Some((Box::new(result), columns)))
            }
            CacheLookupResult::Miss => Ok(None),
        }
    }

    /// A table scan's rows projected by the SELECT list, the names of the
    /// output columns and, for an ORDER BY, their layout; with `order_by_extra`, the ORDER BY
    /// columns the list lacks follow, for execute_select to sort by and drop.
    /// True when only the rows an ORDER BY LIMIT keeps were kept, in order
    #[allow(clippy::too_many_arguments)]
    fn project_scanned_rows(
        &self,
        stmt: &SelectStatement,
        rows: RowVec,
        all_columns: &CompactArc<Vec<String>>,
        all_columns_lower: &[String],
        table_alias: Option<&str>,
        order_by_extra: bool,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>, Option<OutputLayout>, bool)> {
        let mut output_columns =
            self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        if order_by_extra {
            let (rows, plan, extra_names, layout) = self.ordered_rows_plan(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                stmt.table_expr.as_deref(),
                rows,
                all_columns,
                table_alias,
                ctx,
            )?;
            output_columns.extend(extra_names);
            let plan = OutputProjection::Ordered(plan);
            let (projected_rows, applied) =
                self.run_output_projection(stmt, &plan, rows, &output_columns, Some(&layout), ctx)?;
            return Ok((projected_rows, output_columns, Some(layout), applied));
        }
        let layout = Self::shared_output_layout(stmt, all_columns, table_alias);
        if Self::has_correlated_select_subqueries(&stmt.columns) {
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                all_columns,
                Some(all_columns_lower),
                ctx,
                table_alias,
            )?;
            return Ok((projected_rows, output_columns, layout, false));
        }
        let plan = OutputProjection::Select(self.select_rows_plan(
            &stmt.columns,
            all_columns,
            Some(all_columns_lower),
            ctx,
            table_alias,
        )?);
        let (projected_rows, applied) =
            self.run_output_projection(stmt, &plan, rows, &output_columns, layout.as_ref(), ctx)?;
        Ok((projected_rows, output_columns, layout, applied))
    }

    /// The rows `plan` projects of `columns`, or only those the statement's
    /// ORDER BY LIMIT keeps, in order, when its sort reads them alone; true then
    fn run_output_projection(
        &self,
        stmt: &SelectStatement,
        plan: &OutputProjection,
        rows: RowVec,
        columns: &[String],
        layout: Option<&OutputLayout>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, bool)> {
        let Some(mut top) = self.projected_top(stmt, columns, layout, ctx)? else {
            let rows = match plan {
                OutputProjection::Select(plan) => Self::run_select_projection(plan, rows, ctx)?,
                OutputProjection::Ordered(plan) => Self::run_ordered_projection(plan, rows, ctx)?,
            };
            return Ok((rows, false));
        };
        OrderLimitPath::ProjectedTop.taken();
        let mut kept = TopSink {
            top: &mut top,
            spare: None,
        };
        match plan {
            OutputProjection::Select(plan) => {
                Self::project_select_into(plan, rows, ctx, &mut kept)?
            }
            OutputProjection::Ordered(plan) => {
                Self::project_ordered_into(plan, rows, ctx, &mut kept)?
            }
        }
        Ok((top.into_row_vec(), true))
    }

    /// A table scan's answer when no row can match: no rows, taking the
    /// aggregate, window or projection path scanned rows would take
    #[allow(clippy::too_many_arguments)]
    fn finish_without_rows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        all_columns: &CompactArc<Vec<String>>,
        all_columns_lower: &[String],
        table_alias: Option<&str>,
        classification: &QueryClassification,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> SelectResult {
        let (result, layout) = match (
            classification.has_aggregation,
            classification.has_window_functions,
        ) {
            (false, false) => {
                let (rows, columns, layout, applied) = self.project_scanned_rows(
                    stmt,
                    RowVec::new(),
                    all_columns,
                    all_columns_lower,
                    table_alias,
                    self.order_by_needs_extra_columns(stmt, all_columns),
                    ctx,
                )?;
                let columns = CompactArc::new(columns);
                let result = ExecutorResult::with_arc_columns(CompactArc::clone(&columns), rows);
                return Ok((
                    Box::new(result),
                    columns,
                    applied,
                    SelectShape::laid_out(layout),
                ));
            }
            _ => self.group_and_window_rows(
                stmt,
                ctx,
                classification,
                RowVec::new(),
                all_columns,
                blocks,
                |out| Self::shared_result_layout(stmt, all_columns, table_alias, out),
            )?,
        };
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, SelectShape::laid_out(layout)))
    }

    /// Execute a simple table scan
    fn execute_simple_table_scan(
        &self,
        table_source: &SimpleTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        read_out: Option<&mut Option<ReadSchema>>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> SelectResult {
        // A table read AS OF is a row source, read once for the statement
        if let Some(as_of) = &table_source.as_of {
            let (columns, rows) = self.as_of_rows(&table_source.name.value_lower, as_of, ctx)?;
            return self.execute_query_on_memory_result(stmt, ctx, columns, rows);
        }
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &table_source.name.value_lower;

        // Check if there's an active explicit transaction
        let active_tx = self.active_transaction.lock().unwrap();
        let in_explicit_transaction = active_tx.is_some();

        // Only a cacheable SELECT * takes a stamp: no transaction, parameters
        // or row-shaping clauses, and a view that sees the stamped state.
        // Taken before the transaction or snapshot the scan reads under, so
        // a cached result is never older than its stamp
        let cacheable = !in_explicit_transaction
            && Self::semantic_writable(stmt, ctx, classification, table_source);
        let cache_stamp = match cacheable {
            true => self.semantic_write_stamp(table_name, ctx),
            false => None,
        };
        // The statement's start looked the cache up and simplified the WHERE
        let early_miss = ctx.semantic_miss(stmt).filter(|_| cacheable);

        // Get table from active transaction or create a new one
        let (table, _standalone_tx) = if let Some(ref tx_state) = *active_tx {
            // Use the active transaction - this allows seeing uncommitted changes
            let table = tx_state
                .transaction
                .get_read_table_at(table_name, ctx.scan_view(table_name), ctx.table_use())
                .map_err(|e| {
                    if matches!(e, Error::TableNotFound(_)) {
                        Error::TableOrViewNotFound(table_name.to_string())
                    } else {
                        e
                    }
                })?;
            drop(active_tx); // Release lock before doing work
            (table, None)
        } else {
            drop(active_tx); // Release lock before creating new transaction
                             // No active transaction: a statement-scoped snapshot, when the
                             // statement carries one, keeps every fetch on one transaction;
                             // otherwise a standalone transaction
            if let Some(snapshot) = ctx.statement_snapshot() {
                let table = snapshot
                    .statement_table_at(table_name, ctx.scan_view(table_name))
                    .map_err(|e| {
                        if matches!(e, Error::TableNotFound(_)) {
                            Error::TableOrViewNotFound(table_name.to_string())
                        } else {
                            e
                        }
                    })?;
                (table, None)
            } else {
                let tx = self.begin_transaction()?;

                // The transaction is this read's own
                let table = tx
                    .get_read_table_at(table_name, ctx.scan_view(table_name), TableUse::Read)
                    .map_err(|e| {
                        if matches!(e, Error::TableNotFound(_)) {
                            Error::TableOrViewNotFound(table_name.to_string())
                        } else {
                            e
                        }
                    })?;
                (table, Some(tx))
            }
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::select_table_opened();
        // The layout this handle's reads are checked against, and its schema
        if let Some(out) = read_out {
            *out = Some((table.layout_token(), table.schema_arc()));
        }

        // Build column list from schema: refcount bump, not a per-statement
        // deep clone of every column name
        let all_columns = table.schema().column_names_arc();
        // Get pre-cached lowercase column names to avoid per-query to_lowercase() calls
        let all_columns_lower = table.schema().column_names_lower_arc();

        // Get table alias for correlated subquery support
        let table_alias: Option<String> = table_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.to_string())
            .or_else(|| Some(table_name.to_string()));
        // The shape of a result this scan answers whole, for its ORDER BY
        let shape_of = |columns: &[String]| {
            SelectShape::laid_out(Self::shared_result_layout(
                stmt,
                &all_columns,
                table_alias.as_deref(),
                columns,
            ))
        };

        // classification is passed from caller to avoid redundant cache lookups

        // A WHERE that keeps no row reads no row; one with a subquery or a
        // parent row to bind is left to the scan, which prepares both
        if let Some(where_clause) = stmt.where_clause.as_deref().filter(|w| {
            !classification.where_has_subqueries
                && ctx.outer_row().is_none()
                && true_for_no_row(w, ctx, &self.function_registry)
        }) {
            Self::bind_empty_where(stmt, where_clause, &all_columns)?;
            return self.finish_without_rows(
                stmt,
                ctx,
                &all_columns,
                &all_columns_lower,
                table_alias.as_deref(),
                classification,
                blocks,
            );
        }

        // AGGREGATION PUSHDOWN: Try to compute simple aggregates directly on storage
        // This avoids all row materialization for queries like:
        // - SELECT COUNT(*) FROM table
        // - SELECT SUM(col), MIN(col), MAX(col) FROM table
        // Must check before any row collection happens
        if classification.has_aggregation && !classification.has_window_functions {
            // One row or the groups, from the source the rules pick
            let indexed = |column: &str| bounds(table.get_index_on_column(column));
            let plan =
                self.aggregate_read_plan(stmt, classification, ctx, table.schema(), indexed)?;
            let result = match &plan {
                Some(plan) if matches!(plan.read, AggregateRead::Global { .. }) => {
                    let run = (classification.as_ref(), plan);
                    self.global_aggregate(table.as_ref(), stmt, run, ctx, blocks)?
                }
                Some(plan) => {
                    self.grouped_aggregate(table.as_ref(), stmt, plan, ctx, &all_columns)?
                }
                None => None,
            };
            if let Some(result) = result {
                let columns = CompactArc::new(result.columns().to_vec());
                let shape = shape_of(&columns);
                return Ok((result, columns, false, shape));
            }
        }

        // A DISTINCT the plan's rule takes reads its column's distinct values
        if classification.has_distinct {
            if let Some(result) =
                Self::distinct_values_answer(table.as_ref(), stmt, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                let shape = shape_of(&columns);
                return Ok((result, columns, false, shape));
            }
        }

        // Check if we need aggregation/window functions (need all columns)
        let needs_all_columns =
            classification.has_aggregation || classification.has_window_functions;

        // Check if ORDER BY references columns not in SELECT
        let order_by_needs_extra_columns = self.order_by_needs_extra_columns(stmt, &all_columns);

        // Build alias map from SELECT columns for alias substitution in WHERE clause.
        // Exclude aliases that shadow real table columns (WHERE references columns, not aliases).
        let alias_map = Self::build_alias_map_excluding(&stmt.columns, Some(&all_columns));

        // Note: Projection pushdown was found to be SLOWER due to extra allocations
        // We now always fetch all columns and project in the executor
        let _ = needs_all_columns; // Silence unused warning - kept for documentation

        // The WHERE the scan pushes down, which the statement's start may have
        // simplified already
        let early = early_miss.filter(|_| alias_map.is_empty() && stmt.where_clause.is_some());
        let rewritten_where = match early {
            Some(_) => None,
            None => Self::scan_where(stmt, &alias_map),
        };
        let where_to_use: Option<&Expression> = rewritten_where
            .as_deref()
            .or_else(|| early.and_then(|miss| miss.simplified.as_deref()))
            .or(stmt.where_clause.as_deref());

        // Check if this query might reference outer columns (correlated)
        let has_outer_context = ctx.outer_row().is_some();

        // A parent row's columns are bound into a WHERE holding subqueries,
        // the subqueries included, so one that reads only the parent row
        // folds once instead of running for every row of this table
        let bound_where: Option<Expression> = match (where_to_use, ctx.outer_row()) {
            (Some(where_expr), Some(outer_row)) if classification.where_has_subqueries => {
                let inner = [table_alias.as_deref().unwrap_or(table_name)];
                let scope = super::utils::InnerScope {
                    tables: &inner,
                    schema: Some(table.schema()),
                };
                Some(super::utils::substitute_outer_references_in_scope(
                    where_expr, outer_row, &scope,
                ))
            }
            _ => None,
        };
        let where_to_use: Option<&Expression> = bound_where.as_ref().or(where_to_use);
        let where_still_correlated = classification.where_has_correlated_subqueries
            && where_to_use.is_some_and(Self::has_correlated_subqueries);

        // SEMANTIC CACHE: a stamp exists only for an eligible query
        let cache_eligible = cache_stamp.is_some() && where_to_use.is_some();

        // Try cache lookup for eligible queries the statement's start did not
        if let Some(stamp) = cache_stamp.filter(|_| cache_eligible && early_miss.is_none()) {
            if let Some(where_expr) = where_to_use {
                if let Some((result, output_columns)) = self.semantic_cache_answer(
                    table_name,
                    stmt,
                    &all_columns,
                    table_alias.as_deref(),
                    where_expr,
                    classification.where_has_parameters.then(|| ctx.params()),
                    stamp,
                )? {
                    let shape = shape_of(&output_columns);
                    return Ok((result, output_columns, false, shape));
                }
            }
        }

        let (storage_expr, needs_memory_filter) = if let Some(where_expr) = where_to_use {
            // If there are subqueries, we must filter in memory
            // Use cached classification to avoid AST traversal
            if classification.where_has_subqueries {
                // The conjuncts without a subquery still narrow the scan;
                // the whole WHERE is evaluated in memory afterwards
                let (storage_expr, _) =
                    pushdown::try_pushdown(where_expr, table.schema(), Some(ctx));
                (storage_expr, true)
            } else if has_outer_context {
                // If we have outer row context, this is a correlated subquery
                // OPTIMIZATION: Substitute outer references with their actual values
                // to enable predicate pushdown for index usage.
                // Example: WHERE o.user_id = u.id -> WHERE o.user_id = 42
                if let Some(outer_row) = ctx.outer_row() {
                    let schema = table.schema();
                    let inner = [table_alias.as_deref().unwrap_or(table_name)];
                    let scope = super::utils::InnerScope {
                        tables: &inner,
                        schema: Some(schema),
                    };
                    let substituted_expr = super::utils::substitute_outer_references_in_scope(
                        where_expr, outer_row, &scope,
                    );
                    // Try to push down the substituted expression
                    let (storage_expr, needs_filter) =
                        pushdown::try_pushdown(&substituted_expr, schema, Some(ctx));
                    if storage_expr.is_some() {
                        // Successfully converted - can use index!
                        // Still need memory filter for any non-pushable parts
                        (storage_expr, needs_filter)
                    } else {
                        // Couldn't push down, fall back to memory filter
                        (None, true)
                    }
                } else {
                    // No outer row available, force in-memory evaluation
                    (None, true)
                }
            } else {
                // Try PARTIAL PUSHDOWN: Extract pushable conjuncts from the WHERE clause.
                // This allows queries like `WHERE indexed_col = 5 AND complex_func(x) > 0`
                // to use the index for `indexed_col = 5` while filtering `complex_func(x) > 0` in memory.
                let schema = table.schema();
                pushdown::try_pushdown(where_expr, schema, Some(ctx))
            }
        } else {
            (None, false)
        };

        // ZONE MAP PRUNING: Short-circuit if zone maps indicate no rows can match
        // This checks min/max statistics per segment to skip entire scan when
        // the WHERE clause predicates are outside all segment ranges
        // IMPORTANT: Don't short-circuit if we have aggregation, window functions, or GROUP BY
        // because aggregation on empty results needs to produce output (e.g., COUNT=0)
        let has_aggregation_or_grouping = classification.has_aggregation
            || classification.has_window_functions
            || classification.has_group_by;

        if !has_aggregation_or_grouping {
            if let Some(ref expr) = storage_expr {
                if self
                    .get_query_planner()
                    .can_prune_entire_scan(&*table, expr.as_ref())
                {
                    return self.finish_without_rows(
                        stmt,
                        ctx,
                        &all_columns,
                        &all_columns_lower,
                        table_alias.as_deref(),
                        classification,
                        blocks,
                    );
                }
            }
        }

        // FAST PATH: IN subquery index optimization
        // For queries like `SELECT * FROM table WHERE id IN (SELECT col FROM other_table WHERE ...)`
        // where 'id' has an index or is PRIMARY KEY, probe directly instead of scanning all rows
        // Left to the normal pipeline: aggregation and window functions (this
        // projection cannot compile them) and an ORDER BY key that is not
        // projected (the sorter would not find it).
        if needs_memory_filter
            && !has_outer_context
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !order_by_needs_extra_columns
        {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns, limit_applied)) = self
                    .try_in_subquery_index_optimization(
                        stmt,
                        where_expr,
                        &*table,
                        &all_columns,
                        table_alias.as_deref(),
                        ctx,
                        classification,
                    )?
                {
                    let shape = shape_of(&columns);
                    return Ok((result, columns, limit_applied, shape));
                }
            }
        }

        // FAST PATH: IN list literal index optimization
        // For queries like `SELECT * FROM table WHERE id IN (1, 2, 3, 5, 8)`
        // where 'id' has an index or is PRIMARY KEY, probe directly instead of scanning all rows.
        // Tried even when the IN list pushed down: pushdown turns it into a
        // min..max range, which on spread-out values scans the whole table.
        // Tried for every primary-key IN list, since pushdown would turn it
        // into a min..max range; for other indexed columns only when a memory
        // filter is needed anyway. Left to the normal pipeline: aggregation
        // and window functions (this projection cannot compile them) and an
        // ORDER BY key that is not projected (the sorter would not find it).
        if !has_outer_context
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !order_by_needs_extra_columns
        {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns, limit_applied)) = self
                    .try_in_list_index_optimization(
                        stmt,
                        where_expr,
                        &*table,
                        &all_columns,
                        table_alias.as_deref(),
                        ctx,
                        classification,
                        needs_memory_filter,
                    )?
                {
                    let shape = shape_of(&columns);
                    return Ok((result, columns, limit_applied, shape));
                }
            }
        }

        // FAST PATH: LIMIT pushdown optimization
        // For simple queries like `SELECT * FROM table LIMIT 10` or
        // `SELECT * FROM table WHERE indexed_col = value LIMIT 10` without ORDER BY,
        // we can stop scanning early at the storage layer
        let can_pushdown_limit = classification.has_limit
            && !classification.has_order_by
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_distinct
            && !needs_memory_filter; // Allow with storage_expr (WHERE on indexed columns)

        if can_pushdown_limit {
            let counts = row_counts(stmt, ctx)?;
            let (limit, offset) = (counts.limit.unwrap_or(usize::MAX), counts.offset);

            // Pass storage_expr to filter while scanning with limit
            // Use unordered version since ORDER BY is already confirmed empty by can_pushdown_limit
            // This enables true early termination without sorting overhead
            // Now returns RowVec directly with row IDs preserved
            let plan = storage_expr
                .as_deref()
                .and_then(|e| table.limited_index_plan(e));
            let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
            let rows = table.collect_access_rows_with_limit_unordered(
                access,
                storage_expr.as_deref(),
                limit,
                offset,
            )?;

            // Project rows according to SELECT expressions
            // Note: collect_rows_with_limit always returns full rows (all columns),
            // so we must always project here regardless of scanner_handled_projection
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                &all_columns,
                Some(&all_columns_lower),
                ctx,
                table_alias.as_deref(),
            )?;
            let output_columns = CompactArc::new(self.get_output_column_names(
                &stmt.columns,
                &all_columns,
                table_alias.as_deref(),
            ));

            let result = ExecutorResult::with_arc_columns(
                CompactArc::clone(&output_columns),
                projected_rows,
            );
            // LIMIT/OFFSET already applied at storage level
            return Ok((Box::new(result), output_columns, true, None));
        }

        // STREAMING PATH: For simple queries without aggregation/window/ORDER BY
        // Use streaming result to avoid materializing all rows into Vec
        //
        // For cache-eligible queries, we disable streaming ONLY if the table is small
        // enough to potentially benefit from caching. Large tables (>100K rows) would
        // exceed the cache limit anyway, so we allow streaming for them.
        //
        // OPTIMIZATION: Use row_count_hint() which is O(1) instead of row_count() which
        // is O(n) with visibility checks. The hint returns an upper bound (versions.len())
        // which is safe for this decision - if hint > threshold, actual count is also large.
        let should_disable_streaming_for_cache = cache_eligible && Self::semantic_rows_fit(&*table);

        let can_use_streaming = !classification.has_order_by
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !order_by_needs_extra_columns
            && !has_outer_context // Can't stream with outer context (correlated subqueries)
            && !should_disable_streaming_for_cache; // Only disable streaming for small cacheable queries

        // Check for CORRELATED subqueries in WHERE - these require per-row evaluation
        // Non-correlated scalar subqueries (like SELECT AVG(...)) are OK - they're evaluated once
        // and the result is used as a literal value for all rows
        let has_correlated_where_subqueries = classification.where_has_correlated_subqueries;

        // CRITICAL OPTIMIZATION: When needs_memory_filter is true AND we have LIMIT,
        // skip the streaming path. The streaming path pre-materializes ALL rows from
        // storage (via collect_visible_rows) before FilteredResult applies the filter.
        // This defeats early termination for LIMIT queries.
        // Fall through to the parallel/sequential path which has proper early termination.
        let skip_streaming_for_memory_filter_with_limit =
            needs_memory_filter && stmt.limit.is_some() && stmt.order_by.is_empty();

        if can_use_streaming
            && !has_correlated_where_subqueries
            && !skip_streaming_for_memory_filter_with_limit
        {
            // A SELECT list without a block streams; its blocks need processing first
            if !classification.select_has_scalar_subqueries {
                let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
                // A table that reads by need materializes only the columns
                // the statement names
                let needed = (table.narrows_scans()
                    && !has_outer_context
                    && !classification.where_has_subqueries)
                    .then(|| super::utils::needed_columns(stmt, where_to_use, &all_columns_lower))
                    .flatten();
                let plan = storage_expr.as_deref().and_then(|e| table.index_plan(e));
                let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                let scanner = table.scan_access(
                    access,
                    &column_idx_vec,
                    needed.as_deref(),
                    storage_expr.as_deref(),
                )?;

                let mut result: Box<dyn QueryResult> =
                    Box::new(ScannerResult::new(scanner, all_columns.clone()));

                // If we need memory filtering (complex WHERE that couldn't be pushed down)
                if needs_memory_filter {
                    if let Some(where_expr) = where_to_use {
                        // Check if there are non-correlated scalar subqueries to process
                        // These need to be evaluated once before streaming
                        let has_scalar_subqueries = classification.where_has_subqueries
                            && !classification.where_has_correlated_subqueries;

                        let filter = if has_scalar_subqueries {
                            // Process scalar subqueries to resolve them to literal values
                            let processed = self.process_where_subqueries(where_expr, ctx)?;
                            RowFilter::new(&processed, &all_columns)?.with_context(ctx)
                        } else {
                            // No scalar subqueries - use expression directly
                            RowFilter::new(where_expr, &all_columns)?.with_context(ctx)
                        };
                        result = Box::new(FilteredResult::from_filter(result, filter));
                    }
                }

                // A lone star is the scan's own rows, under the schema's names
                if matches!(stmt.columns.as_slice(), [Expression::Star(_)]) {
                    let shape = shape_of(&all_columns);
                    return Ok((result, all_columns, false, shape));
                }
                let projection =
                    Projection::compile(&stmt.columns, &all_columns, table_alias.as_deref())?;
                let output_columns = CompactArc::new(self.get_output_column_names(
                    &stmt.columns,
                    &all_columns,
                    table_alias.as_deref(),
                ));
                // The scan's own rows, under their own names, need no projection
                if !(projection.is_columns_in_order(all_columns.len())
                    && *output_columns == all_columns[..])
                {
                    result = Box::new(
                        ProjectionResult::new(result, projection, (*output_columns).clone())
                            .with_context(ctx),
                    );
                }
                // LIMIT/OFFSET NOT applied yet - streaming path
                let shape = shape_of(&output_columns);
                return Ok((result, output_columns, false, shape));
            }
        }

        // Collect rows - choose optimal path based on query type
        // PARALLEL EXECUTION: Use parallel filtering for large datasets
        let parallel_config = ParallelConfig::default();

        let rows_result = if needs_memory_filter {
            // Path 1: Need in-memory filtering (subqueries or complex expressions)
            // For memory filter, we need all columns to evaluate the WHERE clause
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            let needed = (table.narrows_scans()
                && !has_outer_context
                && !classification.has_window_functions
                && !classification.select_has_scalar_subqueries
                && !classification.where_has_subqueries)
                .then(|| super::utils::needed_columns(stmt, where_to_use, &all_columns_lower))
                .flatten();

            // OPTIMIZATION: Delay scanner creation until we know we need all rows.
            // For early termination path, we'll collect rows directly with a limit.
            // This avoids materializing all rows upfront when we only need a few.

            // Check if WHERE contains correlated subqueries
            // Use cached classification to avoid expensive AST traversal
            let has_correlated = where_still_correlated;

            // Check if WHERE contains any subqueries (correlated or not)
            // Use cached classification to avoid redundant traversal of the expression tree
            let has_subqueries = classification.where_has_subqueries;

            // SEMI-JOIN OPTIMIZATION: Try to optimize correlated EXISTS subqueries
            // This transforms EXISTS (SELECT ... WHERE outer.col = inner.col AND ...)
            // into: outer.col IN (SELECT DISTINCT inner_col FROM inner WHERE ...)
            // This changes O(outer × inner) to O(inner + outer) - massive performance win!
            let (processed_where, has_correlated) = if has_correlated {
                if let Some(where_expr) = where_to_use {
                    // Get outer table names for semi-join detection
                    let outer_tables = Self::collect_outer_table_names(&stmt.table_expr);

                    // Extract limit value for optimization decision
                    // With small LIMIT, index-nested-loop with early termination is faster
                    let outer_limit = row_counts(stmt, ctx)
                        .ok()
                        .and_then(|counts| counts.limit)
                        .and_then(|limit| i64::try_from(limit).ok());

                    // ANTI-JOIN OPTIMIZATION: For pure NOT EXISTS without LIMIT, use HashJoinOperator::Anti
                    // This is faster than InHashSet because:
                    // 1. Bulk hash table build and probe (no per-row expression evaluation)
                    // 2. Better cache efficiency
                    // 3. Direct row iteration without VM overhead
                    //
                    // HOWEVER: For queries with LIMIT, we skip anti-join because it materializes
                    // ALL outer rows before applying LIMIT. The streaming InHashSet path can stop
                    // early once LIMIT is reached, making it faster for limited result sets.
                    if let Some(not_exists_info) =
                        Self::try_extract_not_exists_info(where_expr, &outer_tables)
                    {
                        // Check if NOT EXISTS is the ONLY predicate (pure anti-join case)
                        let is_pure_not_exists = matches!(where_expr, Expression::Prefix(_));

                        // Only use anti-join for queries without LIMIT (or very large LIMIT)
                        // For LIMIT queries, the streaming InHashSet path is faster due to early termination
                        let has_limit = outer_limit.is_some()
                            && outer_limit.unwrap() < ANTI_JOIN_LIMIT_THRESHOLD;

                        // This path projects the anti-join rows and returns
                        // them, so it can only answer a plain projection:
                        // anything that shapes the result afterwards belongs
                        // to the general path
                        let is_plain_projection = !classification.has_aggregation
                            && !classification.has_window_functions
                            && !classification.has_group_by
                            && !classification.has_having
                            && !classification.has_order_by
                            && !classification.has_distinct
                            && !classification.has_distinct_on;
                        // Few outer rows probe the inner index each instead
                        let use_anti_join = is_pure_not_exists
                            && !has_limit
                            && is_plain_projection
                            && !self.should_use_index_nested_loop(
                                &not_exists_info,
                                None,
                                Some(table.row_count_hint()),
                                ctx,
                            );
                        // Decided before any row is read: an inner WHERE storage
                        // cannot answer whole leaves NOT EXISTS to the general path
                        let inner_filter = if use_anti_join {
                            self.anti_join_inner_filter(&not_exists_info, ctx)
                        } else {
                            None
                        };

                        // The inner rows as this statement sees them, or none
                        // when the inner name is not a table it can read so
                        let inner = match inner_filter {
                            Some(filter) => {
                                self.anti_join_inner_rows(&not_exists_info, filter.as_deref(), ctx)?
                            }
                            None => None,
                        };

                        if let Some((inner_rows, inner_key)) = inner {
                            #[cfg(feature = "test-failpoints")]
                            crate::test_failpoints::general_path_reached(
                                crate::test_failpoints::GeneralPath::AntiJoin,
                            );
                            // Materialize outer table rows
                            let mut outer_rows = table.collect_all_rows(storage_expr.as_deref())?;

                            // Execute anti-join
                            let anti_join_result = self.execute_anti_join(
                                &not_exists_info,
                                CompactArc::new(outer_rows.drain_rows().collect()),
                                &all_columns,
                                inner_rows,
                                inner_key,
                            )?;

                            // Project and return result
                            let projected_rows = self.project_rows_with_alias(
                                &stmt.columns,
                                anti_join_result,
                                &all_columns,
                                Some(&all_columns_lower),
                                ctx,
                                table_alias.as_deref(),
                            )?;
                            let output_columns = CompactArc::new(self.get_output_column_names(
                                &stmt.columns,
                                &all_columns,
                                table_alias.as_deref(),
                            ));

                            // LIMIT and OFFSET, parameters included, are the caller's to apply
                            let result = ExecutorResult::with_arc_columns(
                                CompactArc::clone(&output_columns),
                                projected_rows,
                            );
                            let shape = shape_of(&output_columns);
                            return Ok((Box::new(result), output_columns, false, shape));
                        }
                    }

                    // Try semi-join optimizations for both EXISTS and IN subqueries
                    // These transform O(outer × inner) to O(inner + outer)
                    // Avoid cloning upfront - only clone if no optimization succeeds

                    // 1. Try EXISTS semi-join optimization
                    let exists_optimized = self.try_optimize_exists_to_semi_join(
                        where_expr,
                        ctx,
                        &outer_tables,
                        outer_limit,
                        Some(table.row_count_hint()),
                    )?;

                    // 2. Try IN semi-join optimization (on EXISTS result or original)
                    let expr_for_in = exists_optimized.as_ref().unwrap_or(where_expr);
                    let in_optimized =
                        self.try_optimize_in_to_semi_join(expr_for_in, ctx, &outer_tables)?;

                    // Determine final expression without unnecessary clones
                    let (current_expr, any_optimized) = match (exists_optimized, in_optimized) {
                        (_, Some(in_opt)) => (in_opt, true),
                        (Some(exists_opt), None) => (exists_opt, true),
                        (None, None) => (where_expr.clone(), false), // Clone only when needed
                    };

                    // Check if there are still correlated subqueries after optimizations
                    let still_correlated = Self::has_correlated_subqueries(&current_expr);

                    if any_optimized && !still_correlated {
                        // All correlated subqueries were optimized away; a
                        // subquery the rewrite refused is read once here
                        let processed = if Self::has_subqueries(&current_expr) {
                            self.process_where_subqueries(&current_expr, ctx)?
                        } else {
                            current_expr
                        };
                        (Some(processed), false)
                    } else if any_optimized {
                        // Some optimizations applied but still have correlated parts
                        (Some(current_expr), true)
                    } else {
                        // No optimizations applied - keep original for per-row processing
                        (Some(current_expr), true)
                    }
                } else {
                    (None, false)
                }
            } else if let Some(where_expr) = where_to_use {
                if has_subqueries {
                    // Pre-process uncorrelated subqueries once
                    (Some(self.process_where_subqueries(where_expr, ctx)?), false)
                } else {
                    (Some(where_expr.clone()), false)
                }
            } else {
                (None, false)
            };

            // FAST PATH: InHashSet index optimization (from EXISTS → semi-join transformation)
            // If EXISTS was transformed to InHashSet and the column is PK/indexed,
            // probe directly instead of scanning all rows
            // Skip if there are correlated subqueries in SELECT columns
            // Skip if query has aggregation: projection cannot compile aggregate functions
            // Use cached classification to avoid AST traversal
            if !has_correlated
                && !classification.has_group_by
                && !classification.has_aggregation
                && !classification.has_window_functions
                && !classification.select_has_correlated_subqueries
                && !self.order_by_needs_extra_columns(stmt, &all_columns)
            {
                if let Some(ref where_expr) = processed_where {
                    if let Some((result, columns, limit_applied)) = self
                        .try_in_hashset_index_optimization(
                            stmt,
                            where_expr,
                            &*table,
                            &all_columns,
                            table_alias.as_deref(),
                            ctx,
                        )?
                    {
                        let shape = shape_of(&columns);
                        return Ok((result, columns, limit_applied, shape));
                    }
                }
            }

            // Check if we can use the PARALLEL PATH:
            // For simple WHERE without subqueries, collect all rows first
            // then filter in parallel. This is much faster for large tables.
            // CRITICAL: Cannot use parallel path when there's outer context because
            // the WHERE clause may reference outer columns (e.g., products.id in
            // a correlated subquery like WHERE product_id = products.id)
            let use_parallel_path =
                !has_correlated && !has_outer_context && processed_where.is_some();

            if use_parallel_path {
                let where_expr = processed_where.as_ref().unwrap();

                // EARLY TERMINATION: For LIMIT queries without ORDER BY, GROUP BY,
                // aggregation, or window functions, use streaming filter with early
                // termination. This is critical for NOT IN performance.
                let can_early_terminate = !classification.has_order_by
                    && !classification.has_group_by
                    && !classification.has_aggregation
                    && !classification.has_window_functions
                    && !stmt.distinct
                    && stmt.distinct_on.is_empty();

                let early_termination_target = if can_early_terminate {
                    let counts = row_counts(stmt, ctx)?;
                    counts.limit.map(|limit| (counts.offset, limit))
                } else {
                    None
                };

                // Use early termination path if we have a target
                if let Some((early_offset, early_limit)) = early_termination_target {
                    let target = early_offset.saturating_add(early_limit);
                    // OPTIMIZATION: For memory-filter + LIMIT, use iterative fetching
                    // with early termination. We fetch in batches to avoid loading
                    // all rows when only a few are needed.
                    let mut result_rows = RowVec::with_capacity(target.min(1000));
                    let plan = storage_expr
                        .as_deref()
                        .and_then(|e| table.limited_index_plan(e));
                    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);

                    if needs_memory_filter {
                        // PARTIAL PUSHDOWN: Storage handles some filtering, memory handles rest.
                        // Use batched fetching with increasing batch sizes to minimize work.
                        let mut batch_size = target.max(100); // Start with at least target rows
                        let mut offset = 0usize;

                        // Pre-compile filter ONCE outside the loop
                        let mut memory_eval =
                            ExpressionEval::compile(where_expr, &all_columns)?.with_context(ctx);

                        loop {
                            // Fetch batch from storage with storage filter + limit
                            // Now returns RowVec directly with row IDs preserved
                            let batch = table.collect_access_rows_with_limit(
                                access,
                                storage_expr.as_deref(),
                                batch_size,
                                offset,
                            )?;

                            if batch.is_empty() {
                                break; // No more rows from storage
                            }

                            // Apply full WHERE filter (includes both pushed and non-pushed parts)
                            for (row_id, row) in batch {
                                if memory_eval.eval_bool_checked(&row)? {
                                    result_rows.push((row_id, row));
                                    if result_rows.len() >= target {
                                        break;
                                    }
                                }
                            }

                            if result_rows.len() >= target {
                                break; // Got enough rows
                            }

                            // Need more rows - increase batch size and offset
                            offset += batch_size;
                            batch_size *= 2; // Exponential backoff
                        }
                    } else {
                        // NO MEMORY FILTER NEEDED: Full filter pushed to storage.
                        // Use storage-level limit directly. Returns RowVec directly.
                        result_rows = table.collect_access_rows_with_limit(
                            access,
                            storage_expr.as_deref(),
                            target,
                            0,
                        )?;
                    }

                    // The rows OFFSET skips were collected too; drop them and cut to
                    // LIMIT here, since this path reports both as applied
                    let mut trimmed = result_rows.into_vec();
                    trimmed.drain(..early_offset.min(trimmed.len()));
                    trimmed.truncate(early_limit);
                    let result_rows = RowVec::from_vec(trimmed);
                    // Project rows and return early - LIMIT/OFFSET already applied
                    let projected_rows = self.project_rows_with_alias(
                        &stmt.columns,
                        result_rows,
                        &all_columns,
                        Some(&all_columns_lower),
                        ctx,
                        table_alias.as_deref(),
                    )?;
                    let output_columns = CompactArc::new(self.get_output_column_names(
                        &stmt.columns,
                        &all_columns,
                        table_alias.as_deref(),
                    ));
                    let result = ExecutorResult::with_arc_columns(
                        CompactArc::clone(&output_columns),
                        projected_rows,
                    );
                    return Ok((Box::new(result), output_columns, true, None)); // true = LIMIT applied
                }

                // Normal path: collect all rows first (for ORDER BY, aggregation, etc.)
                // Without a storage filter there is no index opportunity,
                // so the table hands back the materialized RowVec directly
                // (with real row ids) instead of a per-row scanner drain.
                // With one, the access rule picks the scan's keys or indexes.
                let all_rows = if storage_expr.is_none() {
                    table.collect_all_rows(None)?
                } else {
                    let plan = storage_expr.as_deref().and_then(|e| table.index_plan(e));
                    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                    let mut scanner = table.scan_access(
                        access,
                        &column_idx_vec,
                        needed.as_deref(),
                        storage_expr.as_deref(),
                    )?;
                    let mut all_rows =
                        RowVec::with_capacity(scanner.estimated_count().unwrap_or(64));
                    while scanner.next() {
                        all_rows.push(scanner.take_row_with_id());
                    }
                    if let Some(error) = scanner.err() {
                        return Err(error.clone());
                    }
                    all_rows
                };

                // Apply parallel filtering if we have enough rows
                // CRITICAL: Propagate errors with ? instead of silently swallowing them
                let filtered = parallel::parallel_filter(
                    all_rows,
                    where_expr,
                    &all_columns,
                    &self.function_registry,
                    &parallel_config,
                    ctx,
                )?;
                (filtered, None, None)
            } else {
                // SEQUENTIAL PATH: For correlated subqueries or complex cases
                // Create evaluator once and reuse for all rows
                let mut eval = if processed_where.is_some() {
                    let mut e = CompiledEvaluator::new(&self.function_registry);
                    e = e.with_context(ctx);
                    e.init_columns(&all_columns);
                    Some(e)
                } else {
                    None
                };

                // OPTIMIZATION: Pre-compute column name mappings outside the loop
                // This avoids repeated to_lowercase() and format!() calls per row
                let column_keys: Option<Vec<ColumnKeyMapping>> = if has_correlated {
                    Some(ColumnKeyMapping::build_mappings(
                        &all_columns,
                        table_alias.as_deref(),
                    ))
                } else {
                    None
                };

                // OPTIMIZATION: Pre-allocate outer_row_map with capacity and reuse
                let base_capacity = all_columns.len() * 2 + ctx.outer_row().map_or(0, |m| m.len());
                let mut outer_row_map: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
                outer_row_map.reserve(base_capacity);

                // OPTIMIZATION: Wrap all_columns in Arc once, reuse for all rows (only if needed)
                let all_columns_arc: Option<CompactArc<Vec<String>>> = if has_correlated {
                    Some(all_columns.clone())
                } else {
                    None
                };

                // LIMIT EARLY TERMINATION: For correlated subqueries without ORDER BY,
                // we can stop as soon as we have enough matching rows.
                // This turns O(outer_size) EXISTS evaluations into O(LIMIT) evaluations.
                let early_termination_target: Option<usize> =
                    if has_correlated && stmt.order_by.is_empty() {
                        rows_needed(stmt, ctx)
                    } else {
                        None
                    };

                // Create scanner for the correlated subquery path
                // For correlated subqueries, we can't push down the WHERE clause to storage
                // because it depends on outer row values that change per row
                let plan = storage_expr.as_deref().and_then(|e| table.index_plan(e));
                let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                let mut scanner = table.scan_access(
                    access,
                    &column_idx_vec,
                    needed.as_deref(),
                    storage_expr.as_deref(),
                )?;

                // Pre-allocate to reduce reallocations - 64 avoids first 6 grow operations
                let mut rows: RowVec = RowVec::with_capacity(64);
                let mut row_count = 0u64;
                while scanner.next() {
                    // Check for cancellation every 100 rows (more frequent for slow queries)
                    row_count += 1;
                    if row_count.is_multiple_of(100) {
                        ctx.check_cancelled()?;
                    }

                    let row = scanner.take_row();

                    // Apply in-memory WHERE filter if needed (for complex expressions or subqueries)
                    if let (Some(ref where_expr), Some(ref mut evaluator)) =
                        (&processed_where, &mut eval)
                    {
                        evaluator.set_row_array(&row);

                        // For correlated subqueries, process per-row with outer context
                        if has_correlated {
                            #[cfg(feature = "test-failpoints")]
                            crate::test_failpoints::general_path_reached(
                                crate::test_failpoints::GeneralPath::CorrelatedScan,
                            );
                            // OPTIMIZATION: Clear and reuse outer_row_map instead of creating new
                            outer_row_map.clear();

                            // OPTIMIZATION: Copy parent outer context if exists (for nested correlated subqueries)
                            // Parent context doesn't change per row, but we need to clone since
                            // std::mem::take moves the map out each iteration
                            if let Some(parent_outer_row) = ctx.outer_row() {
                                outer_row_map.extend(
                                    parent_outer_row.iter().map(|(k, v)| (k.clone(), v.clone())),
                                );
                            }

                            // Use pre-computed column mappings
                            if let Some(ref keys) = column_keys {
                                for mapping in keys {
                                    if let Some(value) = row.get(mapping.index) {
                                        // OPTIMIZATION: Clone value once and reuse for all key insertions
                                        // Previously we cloned 2-3 times per column
                                        let cloned_value = value.clone();

                                        // Insert with unqualified part first (if column had a dot)
                                        if let Some(ref upart) = mapping.unqualified_part {
                                            outer_row_map
                                                .insert(upart.clone(), cloned_value.clone());
                                        }

                                        // Insert with qualified name if available
                                        if let Some(ref qname) = mapping.qualified_name {
                                            outer_row_map
                                                .insert(qname.clone(), cloned_value.clone());
                                        }

                                        // Insert with lowercase column name (move, no clone)
                                        outer_row_map
                                            .insert(mapping.col_lower.clone(), cloned_value);
                                    }
                                }
                            }

                            // Create context with outer row (cheap due to Arc)
                            // SAFETY: all_columns_arc is always Some when has_correlated is true
                            let mut correlated_ctx = ctx.with_outer_row(
                                std::mem::take(&mut outer_row_map),
                                all_columns_arc.clone().unwrap(), // Arc clone = cheap
                            );

                            // FAST PATH: If WHERE is just EXISTS or NOT EXISTS, evaluate directly
                            // without creating AST nodes. This saves ~2-3μs per row.
                            let (result, used_evaluator) = if let Expression::Exists(exists) =
                                where_expr
                            {
                                let r = self
                                    .execute_exists_subquery(&exists.subquery, &correlated_ctx)?;
                                (r, false)
                            } else if let Expression::Prefix(prefix) = where_expr {
                                if prefix.operator.eq_ignore_ascii_case("NOT") {
                                    if let Expression::Exists(exists) = prefix.right.as_ref() {
                                        let r = !self.execute_exists_subquery(
                                            &exists.subquery,
                                            &correlated_ctx,
                                        )?;
                                        (r, false)
                                    } else {
                                        // Not a simple NOT EXISTS, use standard path
                                        let processed = self.process_correlated_where(
                                            where_expr,
                                            &correlated_ctx,
                                        )?;
                                        ctx.check_cancelled()?;
                                        evaluator.set_outer_row_owned(
                                            correlated_ctx.outer_row.take().unwrap_or_default(),
                                        );
                                        evaluator.set_row_array(&row);
                                        let r = evaluator.evaluate_bool(&processed)?;
                                        (r, true)
                                    }
                                } else {
                                    // Not EXISTS/NOT EXISTS, use standard path
                                    let processed =
                                        self.process_correlated_where(where_expr, &correlated_ctx)?;
                                    ctx.check_cancelled()?;
                                    evaluator.set_outer_row_owned(
                                        correlated_ctx.outer_row.take().unwrap_or_default(),
                                    );
                                    evaluator.set_row_array(&row);
                                    let r = evaluator.evaluate_bool(&processed)?;
                                    (r, true)
                                }
                            } else {
                                // Complex WHERE expression, use standard path
                                let processed =
                                    self.process_correlated_where(where_expr, &correlated_ctx)?;

                                // Check for cancellation after processing each correlated subquery
                                // This is critical for slow correlated subqueries
                                ctx.check_cancelled()?;

                                // OPTIMIZATION: Take ownership instead of cloning - avoids HashMap clone
                                evaluator.set_outer_row_owned(
                                    correlated_ctx.outer_row.take().unwrap_or_default(),
                                );
                                evaluator.set_row_array(&row);

                                (evaluator.evaluate_bool(&processed)?, true)
                            };

                            // Take back the map for reuse
                            if used_evaluator {
                                outer_row_map = evaluator.take_outer_row();
                            } else {
                                outer_row_map = correlated_ctx.outer_row.take().unwrap_or_default();
                            }

                            if !result {
                                continue;
                            }
                        } else {
                            // Standard evaluation for non-correlated subqueries
                            if !evaluator.evaluate_bool(where_expr)? {
                                continue;
                            }
                        }
                    }

                    rows.push((row_count as i64, row));

                    // LIMIT EARLY TERMINATION: Stop if we have enough rows
                    if let Some(target) = early_termination_target {
                        if rows.len() >= target {
                            break;
                        }
                    }
                }
                if let Some(error) = scanner.err() {
                    return Err(error.clone());
                }
                (rows, None, None)
            }
        } else if storage_expr.is_some() {
            // Path 2: WHERE clause with pushdown - use scanner for index optimization
            // Note: We fetch all columns here because downstream projection uses all_columns indices
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            // A table that reads by need materializes only the columns the
            // statement names; anything a name cannot stand for reads all
            let needed = (table.narrows_scans()
                && !has_outer_context
                && !classification.has_window_functions
                && !classification.select_has_scalar_subqueries
                && !classification.where_has_subqueries)
                .then(|| super::utils::needed_columns(stmt, where_to_use, &all_columns_lower))
                .flatten();
            let plan = storage_expr.as_deref().and_then(|e| table.index_plan(e));
            let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
            let mut scanner = table.scan_access(
                access,
                &column_idx_vec,
                needed.as_deref(),
                storage_expr.as_deref(),
            )?;

            // OPTIMIZATION: Use take_row() to avoid cloning each row
            // Pre-allocate to reduce reallocations - 64 avoids first 6 grow operations
            let mut rows = RowVec::with_capacity(64);
            let mut row_count = 0u64;
            while scanner.next() {
                row_count += 1;
                if row_count.is_multiple_of(100) {
                    ctx.check_cancelled()?;
                }
                rows.push((row_count as i64, scanner.take_row()));
            }
            if let Some(e) = scanner.err() {
                return Err(e.clone());
            }
            (rows, None, None)
        } else {
            // Path 3: Full scan without WHERE - use collect_all_rows
            // Projection is handled later by the executor which is more efficient
            //
            // Window functions read through an index as the plan's rule
            // chooses, else every row
            let read = (classification.has_window_functions && !classification.has_aggregation)
                .then(|| {
                    Self::window_read(
                        stmt,
                        table.schema(),
                        |column| table.get_index_on_column(column).is_some(),
                        blocks.is_some(),
                    )
                })
                .flatten();
            let rows = match read {
                Some(read) => {
                    self.read_window_rows(&read, (stmt, None), ctx, table.as_ref(), &all_columns)?
                }
                None => None,
            };
            match rows {
                Some(WindowRows::Answer(result)) => {
                    let columns = CompactArc::new(result.columns().to_vec());
                    return Ok((result, columns, false, None));
                }
                Some(WindowRows::Rows(rows, sorted, grouped)) => (rows, sorted, grouped),
                None => (table.collect_all_rows(None)?, None, None),
            }
        };

        // Destructure: rows and optional window optimization states
        let (rows, window_presorted_state, window_pregrouped_state): (
            RowVec,
            Option<WindowPreSortedState>,
            Option<WindowPreGroupedState>,
        ) = rows_result;

        // Record cardinality feedback for future estimate improvements
        // This helps the optimizer learn from actual query execution
        if let Some(where_expr) = where_to_use {
            let actual_rows = rows.len() as u64;
            // Only record feedback if we have a meaningful predicate and enough rows
            if actual_rows >= 10 || rows.is_empty() {
                if let Some(estimated_rows) = self
                    .get_query_planner()
                    .estimate_scan_rows(table_name, Some(where_expr))
                {
                    self.get_query_planner().record_feedback(
                        table_name,
                        where_expr,
                        None, // Column-specific feedback not yet implemented
                        estimated_rows,
                        actual_rows,
                    );
                }
            }
        }

        // Handle the combination of window functions and aggregation
        // Order: Aggregation first (GROUP BY), then window functions
        let has_window = classification.has_window_functions;
        let has_agg = classification.has_aggregation;

        if has_agg {
            let (result, layout) = self.group_and_window_rows(
                stmt,
                ctx,
                classification,
                rows,
                &all_columns,
                blocks,
                |out| Self::shared_result_layout(stmt, &all_columns, table_alias.as_deref(), out),
            )?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, SelectShape::laid_out(layout)));
        }

        // Check if we need window functions only (no aggregation)
        if has_window {
            let (result, layout) = self.window_over(
                stmt,
                ctx,
                (&rows, &all_columns),
                (window_presorted_state, window_pregrouped_state),
                blocks,
            )?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, SelectShape::laid_out(layout)));
        }

        // Project rows according to SELECT expressions
        // Check if deferred projection is applicable (ORDER BY + LIMIT with simple columns)
        // This reduces allocations from O(matched_rows) to O(limit)
        let deferred_projection_info =
            self.get_deferred_projection_info(stmt, &all_columns_lower, classification);

        let (projected_rows, output_columns, shape, applied) = match deferred_projection_info {
            Some((col_indices, output_names)) if !order_by_needs_extra_columns => {
                // DEFERRED PROJECTION: Skip projection now, do it after ORDER BY + LIMIT
                // Return all source columns; projection will be applied after TopNResult
                (
                    rows,
                    all_columns.to_vec(),
                    SelectShape {
                        deferred: Some((col_indices, output_names)),
                        layout: Some(LaidOut::Built(OutputLayout::of_shared_input(&all_columns))),
                        visible: None,
                    },
                    false,
                )
            }
            _ => {
                let (projected_rows, output_columns, layout, applied) = self.project_scanned_rows(
                    stmt,
                    rows,
                    &all_columns,
                    &all_columns_lower,
                    table_alias.as_deref(),
                    order_by_needs_extra_columns,
                    ctx,
                )?;
                (
                    projected_rows,
                    output_columns,
                    SelectShape::laid_out(layout).unwrap_or_default(),
                    applied,
                )
            }
        };

        // SEMANTIC CACHE: Insert result for eligible queries
        // For SELECT * queries, cache the raw rows before returning
        //
        // Note: The cache stores Vec<Row>, so we extract rows from RowVec for caching.
        // The result keeps the original RowVec.
        // Skip caching when deferred projection is used (rows are not projected yet)
        if let Some(stamp) = cache_stamp.filter(|_| cache_eligible && shape.deferred.is_none()) {
            if let Some(where_expr) = where_to_use {
                self.semantic_write(
                    table_name,
                    &all_columns,
                    projected_rows.rows(),
                    where_expr,
                    classification.where_has_parameters.then(|| ctx.params()),
                    stamp,
                );
            }
        }

        let output_columns = CompactArc::new(output_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, applied, Some(shape)))
    }

    /// Execute a JOIN source. A GROUP BY + LIMIT join reads its left side in
    /// chunks, each group from one left row: while the groups fall short of
    /// the ones the statement wants and a chunk came full, the next chunk is
    /// joined and its groups follow. Run `apart`, it and each join it holds
    /// keep both columns of a pair they merge
    fn execute_join_source(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
        apart: bool,
    ) -> SelectResult {
        let reduction = std::cell::Cell::new(None);
        let mut window: Option<(LeftWindow, bool)> = None;
        // After a commit moved the chunks' boundary, under a snapshot that
        // keeps the rows
        let mut consistent_ctx = None;
        // The first chunk's columns, flag and shape, the groups so far and
        // the layout the chunks read the left table in
        type Head = (CompactArc<Vec<String>>, bool, Option<SelectShape>);
        let mut groups: Option<(Head, RowVec, Option<LayoutToken>)> = None;
        let mut fetched = 0usize;
        loop {
            let run_ctx = consistent_ctx.as_ref().unwrap_or(ctx);
            let sent = window.take();
            let asked = sent.as_ref().map_or(0, |(window, _)| window.cap);
            let output = self.execute_join_source_capped(
                join_source,
                stmt,
                run_ctx,
                classification,
                compiled,
                (sent, apart),
                &reduction,
            )?;
            let Some(pass) = reduction.take() else {
                return Ok(output);
            };
            if pass.moved {
                let all = fetched.saturating_add(asked);
                let in_transaction = self.active_transaction.lock().is_ok_and(|tx| tx.is_some());
                let cap = match in_transaction {
                    true => i64::MAX as usize,
                    false => {
                        let snapshot = self.new_statement_snapshot(
                            crate::core::IsolationLevel::SnapshotIsolation,
                            ctx.statement_view(),
                        )?;
                        consistent_ctx = Some(ctx.with_statement_snapshot(snapshot));
                        all.min(i64::MAX as usize)
                    }
                };
                let restart = LeftWindow {
                    offset: 0,
                    cap,
                    boundary: None,
                };
                window = Some((restart, consistent_ctx.is_some()));
                groups = None;
                fetched = 0;
                continue;
            }
            fetched = fetched.saturating_add(pass.read);
            let full = pass.bounded && pass.read == pass.cap;
            let (result, columns, flag, shape) = output;
            let (head, mut rows, token) = match groups.take() {
                // A first chunk that ends the read or gives enough groups is
                // the answer as it came
                None if !full || result.exact_len().is_some_and(|n| n >= pass.want) => {
                    return Ok((result, columns, flag, shape));
                }
                None => (
                    (columns, flag, shape),
                    Self::materialize_result(result)?,
                    pass.token,
                ),
                Some((head, mut rows, token)) => {
                    same_outer_source(&join_source.left, token, pass.token, true)?;
                    for (_, row) in Self::materialize_result(result)? {
                        rows.push((rows.len() as i64, row));
                    }
                    (head, rows, token)
                }
            };
            if !full || rows.len() >= pass.want {
                let (columns, flag, shape) = head;
                let result = ExecutorResult::with_arc_columns(CompactArc::clone(&columns), rows);
                return Ok((Box::new(result), columns, flag, shape));
            }
            let next = LeftWindow {
                offset: fetched,
                cap: next_left_chunk(fetched, rows.len(), pass.want).min(i64::MAX as usize),
                boundary: pass.last,
            };
            window = Some((next, consistent_ctx.is_some()));
            groups = Some((head, std::mem::take(&mut rows), token));
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::join_next_chunk();
        }
    }

    /// The join programs `slot` holds for `stmt`, the join's statement, as
    /// the schemas now are and for the CTEs `ctx` binds; a slot holding none
    /// for them holds an empty set from now on
    fn prepared_join(
        &self,
        slot: &std::sync::RwLock<CompiledExecution>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<std::sync::Arc<PreparedJoin>> {
        #[cfg(feature = "test-failpoints")]
        if crate::test_failpoints::prepared_families_off() {
            return None;
        }
        let epoch = self.engine.schema_epoch();
        let current = |guard: &CompiledExecution| match guard {
            CompiledExecution::PreparedJoin(held)
                if held.validity.holds(&*self.engine, epoch) && held.cte_scope.holds(ctx) =>
            {
                Some(std::sync::Arc::clone(held))
            }
            _ => None,
        };
        if let Some(held) = slot.read().ok().and_then(|guard| current(&guard)) {
            return Some(held);
        }
        let mut guard = slot.write().ok()?;
        if let Some(held) = current(&guard) {
            return Some(held);
        }
        let validity = self.select_validity(stmt, epoch);
        let held = std::sync::Arc::new(PreparedJoin::new(validity, ctx.cte_scope()));
        *guard = CompiledExecution::PreparedJoin(std::sync::Arc::clone(&held));
        Some(held)
    }

    /// Whether a grouped or windowed statement reads its SELECT list's
    /// uncorrelated subqueries before it runs, keeping nothing in its plan
    pub(crate) fn folds_select_subqueries(
        stmt: &SelectStatement,
        classification: &QueryClassification,
    ) -> bool {
        (classification.has_aggregation || classification.has_window_functions)
            && stmt.columns.iter().any(Self::has_subqueries)
            && !Self::has_correlated_select_subqueries(&stmt.columns)
    }

    /// Whether the join's ON is an INNER join's written one, which a WHERE
    /// condition may join without changing what it answers
    fn inner_on(join_source: &JoinTableSource, join_type: &str) -> bool {
        join_source.condition.is_some()
            && join_source.using_columns.is_empty()
            && !["LEFT", "RIGHT", "FULL", "NATURAL", "CROSS"]
                .iter()
                .any(|kind| join_type.contains(kind))
    }

    /// Whether a join keeps its plan, its CTEs bound or inlined, so its
    /// blocks may run; a join with a CTE not named before it runs stays
    /// there; a view only its run names runs when the caller `runs`
    pub(crate) fn join_keeps_plan(
        &self,
        stmt: &SelectStatement,
        with: &WithClause,
        runs: bool,
    ) -> bool {
        matches!(stmt.table_expr.as_deref(), Some(Expression::JoinSource(_)))
            && self.named_ctes(with, runs).is_ok()
    }

    /// `join_keeps_plan` for a run, found once by the join `slot` holds for
    /// this context
    fn run_keeps_join_plan(
        &self,
        slot: Option<&std::sync::RwLock<CompiledExecution>>,
        stmt: &SelectStatement,
        with: &WithClause,
        ctx: &ExecutionContext,
    ) -> bool {
        let epoch = self.engine.schema_epoch();
        let held = slot
            .and_then(|slot| slot.read().ok())
            .and_then(|guard| match &*guard {
                CompiledExecution::PreparedJoin(held)
                    if held.validity.holds(&*self.engine, epoch) && held.cte_scope.holds(ctx) =>
                {
                    Some(std::sync::Arc::clone(held))
                }
                _ => None,
            });
        match held {
            Some(held) => held.ctes_named(|| self.join_keeps_plan(stmt, with, true)),
            None => self.join_keeps_plan(stmt, with, true),
        }
    }

    /// `join_side_schemas` without running anything, its CTEs `with` named
    /// by their bodies, for a caller that reports a join and does not run it
    pub(crate) fn unrun_join_sides(
        &self,
        join: &JoinTableSource,
        with: Option<&WithClause>,
    ) -> Option<JoinSides> {
        let ctx = match with {
            Some(with) => self.named_ctes(with, false).ok()?,
            None => ExecutionContext::new(),
        };
        self.join_side_schemas(join, &ctx, false)
    }

    /// Each side's columns as the join's blocks read them, a base table's
    /// schema or the names a FROM gives any other source, and whether it is
    /// read as rows; a side that is a join, read as rows, has its tables'
    /// columns side by side; a table read AS OF is read as rows. None when a
    /// source is not named without reading it; a view only its run names
    /// runs when the caller `runs`
    pub(crate) fn join_side_schemas(
        &self,
        join_source: &JoinTableSource,
        ctx: &ExecutionContext,
        runs: bool,
    ) -> Option<JoinSides> {
        Some([
            self.join_side(&join_source.left, ctx, runs)?,
            self.join_side(&join_source.right, ctx, runs)?,
        ])
    }

    fn join_side(&self, expr: &Expression, ctx: &ExecutionContext, runs: bool) -> Option<JoinSide> {
        match expr {
            // A table read AS OF is a row source in the table's columns; a
            // view or a CTE of the name has no history to read
            Expression::TableSource(ts) if ts.as_of.is_some() => {
                let schema = self.base_table_side_schema(expr, CteNames::Context(ctx))?;
                let name = SmartString::from(get_table_alias_from_expr(expr).unwrap_or_default());
                Some(JoinSide {
                    leaves: smallvec::smallvec![(name, schema)],
                    rows: true,
                })
            }
            Expression::TableSource(_)
            | Expression::SubquerySource(_)
            | Expression::ValuesSource(_)
            | Expression::FunctionTableSource(_) => {
                let name = SmartString::from(get_table_alias_from_expr(expr).unwrap_or_default());
                let (schema, rows) = match self.base_table_side_schema(expr, CteNames::Context(ctx))
                {
                    Some(schema) => (schema, false),
                    None => (self.join_side_schema(expr, ctx, runs).ok()?, true),
                };
                Some(JoinSide {
                    leaves: smallvec::smallvec![(name, schema)],
                    rows,
                })
            }
            // Its rows lay out each table's columns in turn, both of a pair
            // a NATURAL or USING join merges among them
            Expression::JoinSource(join) => {
                let mut leaves = smallvec::SmallVec::new();
                for side in [&join.left, &join.right] {
                    let side = self.join_side(side, ctx, runs)?;
                    if side.leaves.iter().any(|(name, _)| name.is_empty()) {
                        return None;
                    }
                    leaves.extend(side.leaves);
                }
                Some(JoinSide { leaves, rows: true })
            }
            _ => None,
        }
    }

    /// Whether the ON of `join`, or of a join it holds as a side, holds a
    /// subquery
    pub(crate) fn ons_hold_subqueries(join: &JoinTableSource) -> bool {
        let nested = |side: &Expression| match side {
            Expression::JoinSource(join) => Self::ons_hold_subqueries(join),
            _ => false,
        };
        join.condition.as_deref().is_some_and(Self::has_subqueries)
            || nested(&join.left)
            || nested(&join.right)
    }

    /// Whether a join, `sides` as `join_side_schemas` finds them, runs on its
    /// prepared blocks: named sides, and subqueries in
    /// its ONs, a joined side's included, its WHERE and its output all of the
    /// forms the blocks run; or why not
    pub(crate) fn join_blocks_family(
        &self,
        stmt: &SelectStatement,
        join_source: &JoinTableSource,
        join_type: &str,
        (left_alias, right_alias): (Option<&str>, Option<&str>),
        sides: Option<&JoinSides>,
    ) -> std::result::Result<(), OutOfFamily> {
        #[cfg(feature = "test-failpoints")]
        if crate::test_failpoints::prepared_families_off() {
            return Err("a test refusing the prepared families");
        }
        let Some(sides @ [left, right]) = sides else {
            return Err("a side not named before it is read");
        };
        // The blocks qualify each side's columns by its name, a joined
        // side's by its tables'
        let named = |alias: Option<&str>, side: &JoinSide| alias.is_some() || side.leaves.len() > 1;
        if !named(left_alias, left) || !named(right_alias, right) {
            return Err("a side without a name");
        }
        let split = [
            SplitSide::of(left_alias, left),
            SplitSide::of(right_alias, right),
        ];
        if let Some(on) = join_source
            .condition
            .as_deref()
            .filter(|on| Self::has_subqueries(on))
        {
            self.join_where_family(on)?;
        }
        // A joined side runs as a join of its own, its ON's subqueries as
        // its blocks
        for side in [&*join_source.left, &*join_source.right] {
            if let Expression::JoinSource(join) = side {
                self.joined_side_family(join)?;
            }
        }
        self.join_output_family(stmt)?;
        // A join without a subquery reads a name both sources have from its
        // left source, which its blocks refuse as ambiguous
        let classification = get_classification(stmt);
        let holds_subquery = classification.where_has_subqueries
            || classification.output_has_subqueries
            || Self::ons_hold_subqueries(join_source);
        // Each WHERE part as its site reads it, the SELECT aliases it names
        // opened over the joined columns the sides lay out
        let columns: Vec<String> = sides
            .iter()
            .flat_map(|side| side.columns())
            .map(|(table, name)| qualified(table, name))
            .collect();
        if !holds_subquery {
            let opened_where = stmt
                .where_clause
                .as_deref()
                .map(|w| Self::with_select_aliases(stmt, w, &columns).unwrap_or_else(|| w.clone()));
            let clauses: smallvec::SmallVec<[&Expression; 2]> = join_source
                .condition
                .as_deref()
                .into_iter()
                .chain(opened_where.as_ref())
                .collect();
            if join_names_ambiguous(join_source, sides, &clauses) {
                return Err("a join without a subquery naming a column both sources have");
            }
        }
        let Some(where_clause) = stmt.where_clause.as_deref() else {
            return Ok(());
        };
        let (left, right, cross) = split_join_where(where_clause, split, join_type);
        let opened = |part: Option<Expression>| {
            part.map(|part| Self::with_select_aliases(stmt, &part, &columns).unwrap_or(part))
        };
        let parts = [opened(left), opened(right), opened(cross)];
        for part in parts.iter().flatten() {
            if Self::has_subqueries(part) {
                self.join_where_family(part)?;
            }
        }
        Ok(())
    }

    /// Why the subqueries of the ON of `join`, a joined side, or of a join it
    /// holds are not all of the forms the prepared blocks run
    fn joined_side_family(&self, join: &JoinTableSource) -> std::result::Result<(), OutOfFamily> {
        if let Some(on) = join
            .condition
            .as_deref()
            .filter(|on| Self::has_subqueries(on))
        {
            self.join_where_family(on)?;
        }
        for side in [&*join.left, &*join.right] {
            if let Expression::JoinSource(join) = side {
                self.joined_side_family(join)?;
            }
        }
        Ok(())
    }

    /// Whether a join's output groups its rows, so the grouped blocks or the
    /// general path run its subqueries after the join, never its join blocks
    fn grouped_join_output(stmt: &SelectStatement) -> bool {
        let classification = get_classification(stmt);
        classification.has_aggregation && !classification.has_window_functions
    }

    /// Why the subqueries of the SELECT list, the ORDER BY or the DISTINCT ON
    /// of a join are not all of the forms the prepared blocks run over the
    /// joined rows
    fn join_output_family(&self, stmt: &SelectStatement) -> std::result::Result<(), OutOfFamily> {
        if Self::grouped_join_output(stmt) {
            return Ok(());
        }
        let classification = get_classification(stmt);
        let items = stmt
            .columns
            .iter()
            .filter(|c| !matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)))
            .map(unaliased_item);
        let keys = stmt
            .order_by
            .iter()
            .map(|ob| &ob.expression)
            .chain(&stmt.distinct_on);
        let mut found = false;
        for expr in items.chain(keys) {
            if Self::has_subqueries(expr) {
                self.join_where_family(expr)?;
                found = true;
            }
        }
        if !found {
            return Ok(());
        }
        if classification.has_aggregation
            || classification.has_window_functions
            || !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
        {
            return Err("a grouped statement whose output holds a subquery");
        }
        Ok(())
    }

    /// The program of the WHERE `cross` a join asks at `site` of its joined
    /// rows named `all_columns`, held with the plan when it has one. With
    /// the statement's `blocks`, the WHERE they rewrote; else `cross` as
    /// written, its SELECT aliases opened here, once
    #[allow(clippy::too_many_arguments)]
    fn join_where(
        &self,
        prepared: Option<&PreparedJoin>,
        site: WhereSite,
        stmt: &SelectStatement,
        cross: &Expression,
        all_columns: &[String],
        blocks: Option<&JoinBlocks>,
    ) -> Result<SharedProgram> {
        let compile = || match blocks.and_then(|blocks| Some((blocks, blocks.cross.as_ref()?))) {
            Some((blocks, cross)) => blocks.compile(cross, false, &self.function_registry),
            None => {
                let resolved = Self::with_select_aliases(stmt, cross, all_columns);
                compile_expression(resolved.as_ref().unwrap_or(cross), all_columns)
            }
        };
        match prepared {
            Some(prepared) => prepared.where_program(site, all_columns, compile),
            None => compile(),
        }
    }

    /// The statement's blocks, prepared once with its plan from its ON, the
    /// WHERE `cross` it asks of the joined rows and the WHERE's conditions on
    /// one side, `side_filters`; none without them
    #[allow(clippy::too_many_arguments)]
    fn statement_blocks(
        &self,
        prepared: Option<&PreparedJoin>,
        stmt: &SelectStatement,
        (join_source, join_type): (&JoinTableSource, &str),
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        (left_alias, right_alias): (Option<&str>, Option<&str>),
        sides: Option<&JoinSides>,
        ctx: &ExecutionContext,
    ) -> Result<Option<std::sync::Arc<JoinBlocks>>> {
        let (Some(prepared), Some(sides)) = (prepared, sides) else {
            return Ok(None);
        };
        let [left_side, right_side] = sides;
        prepared.join_blocks(|| {
            // Each column by its table and its own name, which may hold a dot
            let spelled: [Vec<(&str, &str)>; 2] = [
                left_side.columns().collect(),
                right_side.columns().collect(),
            ];
            // The WHERE's SELECT aliases open over the columns the sides lay
            // out, as the decision to run blocks read them
            let columns: Vec<String> = spelled
                .iter()
                .flatten()
                .map(|(table, name)| qualified(table, name))
                .collect();
            // The equalities of a NATURAL join or a USING list stand for its ON
            let (left, right) = columns.split_at(spelled[0].len());
            let natural =
                natural_join_condition(join_type, join_source, left, right, |right, i| {
                    let (table, name) = spelled[usize::from(right)][i];
                    Expression::QualifiedIdentifier(QualifiedIdentifier {
                        qualifier: Identifier::new(
                            Token::new(TokenType::Identifier, table, Position::default()),
                            table,
                        ),
                        name: Identifier::new(
                            Token::new(TokenType::Identifier, name, Position::default()),
                            name,
                        ),
                    })
                })
                .0;
            let on = natural.as_ref().or(join_source.condition.as_deref());
            let opened = cross.map(|cross| {
                Self::with_select_aliases(stmt, cross, &columns).unwrap_or_else(|| cross.clone())
            });
            // A side's conditions come without their qualifier; it is put
            // back, as the index join puts it back on its inner side's. A
            // joined side's keep their tables'
            let qualified = |filter: Option<&Expression>, alias: Option<&str>| {
                filter.map(|f| match alias {
                    Some(alias) => add_table_qualifier(f, alias),
                    None => f.clone(),
                })
            };
            let [left_filter, right_filter] = side_filters;
            let left_filter = qualified(left_filter, left_alias);
            let right_filter = qualified(right_filter, right_alias);
            // The SELECT items and sort keys whose subqueries the blocks run
            let output = !Self::grouped_join_output(stmt);
            let select: Vec<Option<&Expression>> = stmt
                .columns
                .iter()
                .map(|c| Some(unaliased_item(c)).filter(|e| output && Self::has_subqueries(e)))
                .collect();
            let keys: Vec<Option<&Expression>> = stmt
                .order_by
                .iter()
                .map(|ob| Some(&ob.expression).filter(|e| output && Self::has_subqueries(e)))
                .collect();
            let distinct: Vec<Option<&Expression>> = stmt
                .distinct_on
                .iter()
                .map(|key| Some(key).filter(|e| output && Self::has_subqueries(e)))
                .collect();
            self.prepare_join_blocks(
                (join_source, on, Self::inner_on(join_source, join_type)),
                opened.as_ref(),
                [left_filter.as_ref(), right_filter.as_ref()],
                [&select, &keys, &distinct],
                |star| joined_aliases(stmt, &columns, star),
                sides,
                columns.len(),
                ctx,
            )
        })
    }

    /// `predicate` of `join` with the parent row's values in place of the
    /// columns it names outside the joined tables; None with no parent row
    /// or no such column. A conjunct holding a subquery is left for the
    /// per-row path, which binds it
    fn bind_to_parent_row(
        join: &JoinTableSource,
        predicate: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<Expression> {
        let outer = ctx.outer_row()?;
        let mut tables = smallvec::SmallVec::new();
        Self::collect_table_names_from_source(&join.left, &mut tables);
        Self::collect_table_names_from_source(&join.right, &mut tables);
        if !Self::references_outer_columns(predicate, &tables) {
            return None;
        }
        let scope = super::utils::InnerScope {
            tables: &tables,
            schema: None,
        };
        let bound = flatten_and_predicates(predicate)
            .into_iter()
            .map(|pred| {
                if Self::has_subqueries(&pred) {
                    pred
                } else {
                    super::utils::substitute_outer_references_in_scope(&pred, outer, &scope)
                }
            })
            .collect();
        combine_predicates_with_and(bound)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_join_source_capped(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
        (left_window, apart): (Option<(LeftWindow, bool)>, bool),
        reduction: &std::cell::Cell<Option<ReductionPass>>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

        // A parent row's column named through the parent's table is bound in
        // the ON and the WHERE before either is read, or the bare name it
        // falls back to would read a joined column of that name
        let bound_source;
        let join_source = match join_source
            .condition
            .as_deref()
            .filter(|_| ctx.outer_row().is_some())
        {
            Some(on) => {
                // An uncorrelated subquery folds first, as it would below,
                // so the parent's columns beside it are bound too
                let folded = match Self::has_subqueries(on) && !Self::has_correlated_subqueries(on)
                {
                    true => Some(self.process_where_subqueries(on, ctx)?),
                    false => None,
                };
                let bound =
                    Self::bind_to_parent_row(join_source, folded.as_ref().unwrap_or(on), ctx);
                match bound.or(folded) {
                    Some(condition) => {
                        bound_source = JoinTableSource {
                            condition: Some(Box::new(condition)),
                            ..join_source.clone()
                        };
                        &bound_source
                    }
                    None => join_source,
                }
            }
            None => join_source,
        };

        // Get table aliases for filter pushdown
        let left_alias = get_table_alias_from_expr(&join_source.left);
        let right_alias = get_table_alias_from_expr(&join_source.right);

        // Determine join type early for filter pushdown decisions
        let join_type = join_source.join_type.to_uppercase();
        let left_schema = self.base_table_side_schema(&join_source.left, CteNames::Context(ctx));
        let right_schema = self.base_table_side_schema(&join_source.right, CteNames::Context(ctx));

        // The plan holds what the join compiles, unless this execution read a
        // value into its clauses: a folded subquery or an outer row. A join
        // whose ON and WHERE subqueries the blocks run runs them so; any
        // other reads them per execution
        let on_subqueries = join_source
            .condition
            .as_deref()
            .is_some_and(Self::has_subqueries);
        let plan = compiled
            .filter(|_| ctx.outer_row().is_none())
            .and_then(|slot| self.prepared_join(slot, stmt, ctx));
        let block_sides = plan.as_ref().and_then(|plan| {
            plan.sides(|| self.join_side_schemas(join_source, ctx, true))
                .cloned()
        });
        let blocks_where = plan.as_ref().is_some_and(|plan| {
            plan.blocks_where(|| {
                self.join_blocks_family(
                    stmt,
                    join_source,
                    &join_type,
                    (left_alias.as_deref(), right_alias.as_deref()),
                    block_sides.as_ref(),
                )
                .is_ok()
            })
        });
        // Outside the blocks, an uncorrelated subquery in the ON clause is read
        // once, the way the WHERE clause reads one, so every path below
        // compares each pair against a value rather than against something it
        // never evaluated
        let processed_on: Option<Expression> = match join_source.condition.as_deref() {
            Some(cond)
                if !blocks_where && on_subqueries && !Self::has_correlated_subqueries(cond) =>
            {
                Some(self.process_where_subqueries(cond, ctx)?)
            }
            _ => None,
        };
        let on_condition: Option<&Expression> =
            processed_on.as_ref().or(join_source.condition.as_deref());
        let grouped_plan = plan
            .as_ref()
            .filter(|_| Self::grouped_family_reach(stmt, classification))
            .cloned();
        let prepared = plan.as_deref().filter(|_| {
            processed_on.is_none()
                && ((!classification.where_has_subqueries && !on_subqueries) || blocks_where)
        });
        // The sides and the blocks read one snapshot, as correlated probes do
        let snapshot_ctx;
        let ctx = if blocks_where
            && ctx.statement_snapshot().is_none()
            && self.active_transaction.lock().is_ok_and(|tx| tx.is_none())
        {
            let snapshot =
                self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        // Partition WHERE clause predicates for pushdown
        // Note: For OUTER JOINs, we must be careful:
        // - LEFT JOIN: Can push filters to left (preserved), but NOT to right (may have NULLs)
        // - RIGHT JOIN: Can push filters to right (preserved), but NOT to left
        // - FULL OUTER JOIN: Cannot push filters to either side
        // Unless the plan runs them as blocks, uncorrelated subqueries in the
        // WHERE fold to their values before the filters are split, so every
        // join path sees a plain predicate, and a correlated one left after
        // the join is evaluated per joined row
        let correlated_where = classification.where_has_correlated_subqueries;
        let where_folds = classification.where_has_subqueries && !blocks_where;
        let where_for_join: Option<Expression> = match &stmt.where_clause {
            Some(where_clause) if where_folds => {
                // Each conjunct folds on its own, so an uncorrelated one folds
                // next to a correlated one
                let mut folded = Vec::new();
                for pred in flatten_and_predicates(where_clause) {
                    if Self::has_subqueries(&pred) && !Self::has_correlated_subqueries(&pred) {
                        folded.push(self.process_where_subqueries(&pred, ctx)?);
                    } else {
                        folded.push(pred);
                    }
                }
                combine_predicates_with_and(folded)
            }
            Some(where_clause) => Some((**where_clause).clone()),
            None => None,
        };
        // Bound before the predicates are split, or the side a predicate is
        // pushed to would read the bare name it falls back to as its own
        let where_for_join = match where_for_join {
            Some(where_clause) if ctx.outer_row().is_some() => Some(
                Self::bind_to_parent_row(join_source, &where_clause, ctx).unwrap_or(where_clause),
            ),
            where_for_join => where_for_join,
        };
        // The WHERE splits over the columns each side is read with, a joined
        // side's by its tables'
        let split = SplitSide::both(
            block_sides.as_ref(),
            [left_alias.as_deref(), right_alias.as_deref()],
            [left_schema.as_deref(), right_schema.as_deref()],
        );
        let (left_filter, right_filter, cross_filter) = match where_for_join.as_ref() {
            Some(where_clause) => split_join_where(where_clause, split, &join_type),
            None => (None, None, None),
        };
        // The statement's blocks, and the one run of them every site calls
        let blocks = match blocks_where {
            true => self.statement_blocks(
                prepared,
                stmt,
                (join_source, join_type.as_str()),
                cross_filter.as_ref(),
                [left_filter.as_ref(), right_filter.as_ref()],
                (left_alias.as_deref(), right_alias.as_deref()),
                block_sides.as_ref(),
                ctx,
            )?,
            false => None,
        };
        // Each side reads the conditions the blocks left it; the joined row
        // asks the rest, the ones the blocks moved among them
        let where_split = left_filter.is_some() || right_filter.is_some();
        let (left_filter, right_filter) = match blocks.as_deref().and_then(|b| b.written.as_ref()) {
            Some([left, right]) => (left.clone(), right.clone()),
            None => (left_filter, right_filter),
        };
        // A side read as rows reads its conditions without a block; the
        // blocks ask those of its others that bind as it is read, and the
        // rest of the joined row with the WHERE's conditions on both sides
        let rows_sides = blocks.as_deref().map_or([false; 2], |held| held.rows_sides);
        // A joined side runs all of its conditions as a join of its own, not
        // under the plan it holds when the values this run read are in them
        let joined = [&join_source.left, &join_source.right]
            .map(|side| matches!(**side, Expression::JoinSource(_)));
        let reads_plain = [0, 1].map(|i| !joined[i] && rows_sides[i]);
        let side_plan = |side: usize| {
            plan.as_deref()
                .map(|plan| plan.side_plan(side))
                .filter(|_| !(joined[side] && where_folds))
        };
        let plain = |filter: &Option<Expression>, rows: bool| match rows {
            true => filter.as_ref().and_then(|filter| {
                combine_predicates_with_and(
                    flatten_and_predicates(filter)
                        .into_iter()
                        .filter(|part| !Self::has_subqueries(part))
                        .collect(),
                )
            }),
            false => None,
        };
        let (left_plain, right_plain) = (
            plain(&left_filter, reads_plain[0]),
            plain(&right_filter, reads_plain[1]),
        );
        // As the blocks prepared them: an INNER join's ON with them, else its
        // WHERE on the joined row
        let moved_on;
        let (on_condition, cross_filter) =
            match blocks.as_deref().and_then(|held| held.moved.clone()) {
                Some(moved) if Self::inner_on(join_source, &join_type) => {
                    moved_on = combine_predicates_with_and(
                        on_condition.cloned().into_iter().chain([moved]).collect(),
                    );
                    (moved_on.as_ref(), cross_filter)
                }
                Some(moved) => (
                    on_condition,
                    combine_predicates_with_and(cross_filter.into_iter().chain([moved]).collect()),
                ),
                None => (on_condition, cross_filter),
            };
        let blocks_run = match blocks.as_deref() {
            Some(blocks) => Some(self.join_runner(blocks, ctx)?.ok_or_else(|| {
                Error::internal("a table's columns changed while the statement prepared")
            })?),
            None => None,
        };
        let statement_run = match (blocks.as_deref(), blocks_run.as_ref()) {
            (Some(blocks), Some(run)) => Some(StatementRun { blocks, run }),
            _ => None,
        };
        // A joined side whose merged pairs the blocks read apart has each pair
        // shown and named as the join's own, which only the standard path does
        let side_merged = blocks
            .as_deref()
            .is_some_and(|held| held.side_merges().next().is_some());
        // An output the blocks run reads every pair apart and shows it merged
        let output_apart = !classification.has_aggregation
            && !classification.has_window_functions
            && blocks.as_deref().is_some_and(JoinBlocks::runs_output);
        let side_merged = side_merged && !output_apart;
        let left_read = SideRead {
            filter: match reads_plain[0] {
                true => left_plain.as_ref(),
                false => left_filter.as_ref(),
            },
            run: statement_run.map(|run| (run, 0)),
            plan: side_plan(0),
            apart,
        };
        let right_read = SideRead {
            filter: match reads_plain[1] {
                true => right_plain.as_ref(),
                false => right_filter.as_ref(),
            },
            run: statement_run.map(|run| (run, 1)),
            plan: side_plan(1),
            apart,
        };
        let where_blocks = statement_run.and_then(|run| run.site(false));
        let where_blocks = where_blocks
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        let pair_blocks = statement_run.and_then(|run| run.site(true));
        let pair_blocks = pair_blocks
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        // The result a held output builds, its blocks told how many rows ask
        let output_result = |output: &JoinOutput, rows: RowVec, applied: bool| {
            if let Some(run) = &blocks_run {
                run.ask_rows(Some(rows.len()));
            }
            Self::join_output_result(output, rows, applied, ctx, blocks_run.as_ref())
        };
        // A WHERE that keeps no row joins no rows whatever the join type; the
        // sides run with no rows only to name their columns
        let never = where_for_join
            .as_ref()
            .filter(|_| !classification.where_has_subqueries)
            .is_some_and(|w| true_for_no_row(w, ctx, &self.function_registry));

        // Outside the blocks a correlated WHERE runs per row on the general path
        let per_row_where = correlated_where && statement_run.is_none();
        let shape = JoinShape {
            stmt,
            classification,
            join: join_source,
            condition: on_condition,
            join_type: &join_type,
            aliases: [left_alias.as_deref(), right_alias.as_deref()],
            filters: [left_filter.as_ref(), right_filter.as_ref()],
            cross_filter: cross_filter.as_ref(),
            rows_sides,
            per_row_where,
            blocks: statement_run.is_some(),
            ctes: CteNames::Context(ctx),
        };
        let planned;
        let join_plan = match prepared {
            Some(prepared) => prepared.join_plan(|| self.plan_join(&shape))?,
            None => {
                planned = self.plan_join(&shape)?;
                &planned
            }
        };
        // A GROUP BY + LIMIT reads the left side in a chunk, then the right
        // side narrowed by its keys
        let semijoin_limit = join_plan
            .reduction
            .as_ref()
            .and_then(|keys| Self::semijoin_reduction(stmt, &join_type, keys, Some(ctx)));

        let (left_rows, left_columns, right_rows, right_columns) = if never {
            // Not every side honours the filter or the limit, so its rows are
            // dropped and only its columns kept. The filter reads no column,
            // as a side knows only its own
            let none = Expression::BooleanLiteral(BooleanLiteral {
                token: dummy_token("FALSE", TokenType::Keyword),
                value: false,
            });
            // Each read as the blocks check it, without the side's own filter
            let columns_of = |expr: &Expression, side: SideRead<'_>| {
                let side = SideRead {
                    filter: None,
                    ..side
                };
                self.read_join_side(expr, ctx, side, Some(&none), Some(0), 0)
                    .map(|(_, columns, _, _)| columns)
            };
            let left_cols = columns_of(&join_source.left, left_read)?;
            let right_cols = columns_of(&join_source.right, right_read)?;
            if let Some(where_clause) = &where_for_join {
                let joined: Vec<String> = left_cols.iter().chain(&right_cols).cloned().collect();
                Self::bind_empty_where(stmt, where_clause, &joined)?;
            }
            (
                CompactArc::new(Vec::new()),
                left_cols,
                CompactArc::new(Vec::new()),
                right_cols,
            )
        } else if let Some(plan) = semijoin_limit {
            // A later chunk goes on with the reduction an earlier one took
            let grouped_probe = join_plan.probes[0]
                .as_ref()
                .filter(|_| join_plan.grouped_index && left_window.is_none());
            if let Some(probe) = grouped_probe {
                if let Some(result) = self.try_grouped_index_join(
                    stmt,
                    ctx,
                    join_source,
                    &join_type,
                    right_alias.as_deref(),
                    left_read,
                    right_filter.as_ref(),
                    cross_filter.as_ref(),
                    prepared,
                    (&plan, probe),
                )? {
                    return Ok(result);
                }
            }
            // Semi-join reduction for INNER/LEFT JOIN + GROUP BY: fetch the left
            // side with the limit, then the right side through an IN filter on
            // the join key (uses the index)
            let SemijoinReduction {
                cap,
                want,
                left_key_col,
                right_key_col,
                ..
            } = plan;
            let (window, consistent) = left_window.unwrap_or((
                LeftWindow {
                    offset: 0,
                    cap,
                    boundary: None,
                },
                false,
            ));
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::join_side_ran();
            let Some((left_result, left_cols, left_bounded, left_token)) =
                self.left_chunk(&join_source.left, ctx, left_read, &window, consistent)?
            else {
                reduction.set(Some(ReductionPass::moved()));
                let none = ExecutorResult::new(Vec::new(), RowVec::new());
                return Ok((Box::new(none), CompactArc::new(Vec::new()), false, None));
            };
            // A read after its boundary row goes on from where it stands
            let left_rows = match window.offset {
                0 => Self::materialize_result_arc(left_result)?,
                _ => CompactArc::new(
                    Self::materialize_result(left_result)?
                        .into_iter()
                        .map(|(_, row)| row)
                        .collect(),
                ),
            };
            let left_key_idx = Self::find_column_index_by_name(&left_key_col, &left_cols);
            let join_key_values: Vec<Value> = if let Some(idx) = left_key_idx {
                left_rows
                    .iter()
                    .filter_map(|row| row.get(idx).cloned())
                    .filter(|v| !v.is_null())
                    .collect()
            } else {
                Vec::new()
            };
            let keys_in = match join_key_values.is_empty() {
                true => None,
                false => self.build_in_filter_expression(&right_key_col, &join_key_values),
            };
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::join_side_ran();
            let (right_result, right_cols, _, _) = self.read_join_side(
                &join_source.right,
                ctx,
                right_read,
                keys_in.as_ref(),
                None,
                0,
            )?;
            let right_rows = Self::materialize_result_arc(right_result)?;
            reduction.set(Some(ReductionPass {
                cap: window.cap,
                read: left_rows.len(),
                bounded: left_bounded,
                last: left_rows.last().cloned(),
                token: left_token,
                want,
                moved: false,
            }));
            (left_rows, left_cols, right_rows, right_cols)
        } else {
            // Check if we can use Index Nested Loop Join
            // This optimization avoids materializing the right side entirely.
            // The join is chosen alike whatever reads its rows: an aggregate
            // or a window takes the index join's rows as it takes the hash join's
            let pushed_limit = self.pushed_join_limit(stmt, Some(ctx), classification, &join_type);
            // An INNER join without a limit narrows its inner side by a table's
            // candidates only for an outer side found small once it runs
            let small_outer = pushed_limit.is_none() && join_type == "INNER";
            let picked = {
                let left_is_table = left_schema.is_some();
                let right_is_table = right_schema.is_some();
                // The probe a side offers, when this run takes it
                let probe = |side: usize, outer_materialized: bool| {
                    join_plan.probes[side].as_ref().and_then(|probe| {
                        let bounded = pushed_limit.is_some();
                        self.probe_strategy(probe, bounded, small_outer, outer_materialized)
                    })
                };
                pick_index_join_side(
                    &join_type,
                    right_is_table,
                    right_filter.is_some(),
                    left_filter.is_some(),
                    probe(0, !left_is_table),
                    || probe(1, !right_is_table),
                    |(_, strategy, _, _)| matches!(strategy, IndexLookupStrategy::PrimaryKey),
                )
            };
            // A swapped join reads the right side as its outer, with its filter
            let (index_nl_info, nl_right_filter, swapped) = match picked {
                Some((info, true)) => (Some(info), left_filter.clone(), true),
                picked => (picked.map(|(info, _)| info), right_filter.clone(), false),
            };

            // A join the inner table stops answering leaves through the label
            // to the hash join below, with the side it ran and its context
            let mut prepared_side: Option<PreparedSide> = None;
            let mut prepared_ctx: Option<ExecutionContext> = None;
            // The small outer's keys the inner side's read is narrowed by
            let mut inner_keys: Option<Expression> = None;
            'index_nl: {
                let Some((table_name, lookup_strategy, inner_col, outer_col)) = index_nl_info
                else {
                    break 'index_nl;
                };
                let small_outer = small_outer
                    && matches!(lookup_strategy, IndexLookupStrategy::TableEquality { .. });
                // Index Nested Loop path: stream outer side for early termination
                // When swapped, execute right side as outer (with original right filter, now in outer_read)
                let outer_expr = if swapped {
                    &join_source.right
                } else {
                    &join_source.left
                };

                // JOIN KEY EQUIVALENCE OPTIMIZATION:
                // When right filter references the inner join key column, we can push an
                // equivalent filter to the outer side. This dramatically reduces iterations.
                //
                // Example: SELECT * FROM users u JOIN orders o ON u.id = o.user_id WHERE o.user_id IN (1,2,3)
                //
                // Without optimization: Scan ALL 10000 users, lookup orders for each, filter by user_id
                // With optimization: Scan only users with id IN (1,2,3), then lookup their orders
                //
                // The join condition u.id = o.user_id means:
                //   Filter "o.user_id IN (1,2,3)" is equivalent to "u.id IN (1,2,3)" for join results
                // A small outer keeps the filter the standard join gives it
                // The equivalence holds only for two keys of one type or both
                // numbers: a comparison reads a constant against each by its own type
                let same_family = || {
                    let column_type = |expr: &Expression, column: &str| {
                        let name = bare_name(column);
                        self.base_table_side_schema(expr, CteNames::Context(ctx))
                            .and_then(|schema| schema.get_column_type(name))
                    };
                    let inner_expr = if swapped {
                        &join_source.left
                    } else {
                        &join_source.right
                    };
                    match (
                        column_type(inner_expr, &inner_col),
                        column_type(outer_expr, &outer_col),
                    ) {
                        (Some(inner), Some(outer)) => {
                            inner == outer || (inner.is_numeric() && outer.is_numeric())
                        }
                        _ => false,
                    }
                };
                // The inner filter's conjuncts a subquery holds stay with the
                // blocks that read them on their own side
                let transferable = nl_right_filter.as_ref().and_then(|right_f| {
                    combine_predicates_with_and(
                        flatten_and_predicates(right_f)
                            .into_iter()
                            .filter(|pred| !Self::has_subqueries(pred))
                            .collect(),
                    )
                });
                let outer_extra = if small_outer {
                    None
                } else if let Some(ref right_f) = transferable {
                    // Check if right filter references the inner join key column
                    let references = filter_references_column(right_f, &inner_col);
                    if references && same_family() {
                        // Create equivalent filter for outer side by substituting the column
                        substitute_filter_column(right_f, &inner_col, &outer_col)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let outer_read = if swapped { right_read } else { left_read };

                // The limit the join is bounded by, decided with the strategy
                let join_limit = pushed_limit;

                // The outer side is fetched in chunks: the first one is the limit
                // plus an eighth of slack, so the odd outer row without a match
                // needs no second fetch, and the join below fetches more when it
                // stops short. Only a plain table takes the limit and can be
                // fetched again from an offset; the fetch says whether it did.
                // One snapshot serves every outer fetch and the inner probes
                // Read committed is the cheap default; snapshot isolation costs
                // about two microseconds per join, so it is taken only when a
                // continuation finds that a commit moved rows under it
                let mut snapshot = match ctx.statement_snapshot() {
                    Some(snapshot) => snapshot.clone(),
                    None => {
                        self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?
                    }
                };
                let mut ctx_join = ctx.with_statement_snapshot(snapshot.clone());
                if small_outer {
                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::join_side_ran();
                    let (result, columns, _, _) =
                        self.read_join_side(outer_expr, &ctx_join, outer_read, None, None, 0)?;
                    // A small outer's keys narrow the inner side to the rows
                    // its candidates name; the standard join runs either way
                    let rows = match PreparedRows::read_small(result)? {
                        PreparedRows::Shared(rows) => {
                            let inner_alias = if swapped {
                                left_alias.as_deref().unwrap_or(&table_name)
                            } else {
                                right_alias.as_deref().unwrap_or(&table_name)
                            };
                            // A literal list for the IN list path: integer keys narrow
                            // any column, others only one of their own type, NaN none
                            let mut inner_type = None;
                            let mut inner_type = || {
                                *inner_type.get_or_insert_with(|| {
                                    let inner_expr = if swapped {
                                        &join_source.left
                                    } else {
                                        &join_source.right
                                    };
                                    let name = bare_name(&inner_col);
                                    self.base_table_side_schema(inner_expr, CteNames::Context(ctx))
                                        .and_then(|schema| schema.get_column_type(name))
                                })
                            };
                            let keys = Self::find_column_index_by_name(&outer_col, &columns)
                                .and_then(|idx| {
                                    let mut seen = ValueSet::default();
                                    let mut keys = Vec::new();
                                    for key in rows.iter().filter_map(|row| row.get(idx)) {
                                        let narrows = match key {
                                            Value::Null(_) => continue,
                                            Value::Integer(_) => true,
                                            Value::Text(_) => inner_type() == Some(DataType::Text),
                                            Value::Float(f) => {
                                                !f.is_nan() && inner_type() == Some(DataType::Float)
                                            }
                                            Value::Timestamp(_) => {
                                                inner_type() == Some(DataType::Timestamp)
                                            }
                                            _ => false,
                                        };
                                        if !narrows {
                                            return None;
                                        }
                                        if seen.insert(key.clone()) {
                                            keys.push(key.clone());
                                        }
                                    }
                                    Some(keys)
                                });
                            // The alias and the column stay apart: either may hold a dot
                            let column = Expression::QualifiedIdentifier(QualifiedIdentifier {
                                qualifier: Identifier::new(
                                    Token::new(
                                        TokenType::Identifier,
                                        inner_alias,
                                        Position::default(),
                                    ),
                                    inner_alias.to_string(),
                                ),
                                name: Identifier::new(
                                    Token::new(
                                        TokenType::Identifier,
                                        &inner_col,
                                        Position::default(),
                                    ),
                                    inner_col.clone(),
                                ),
                            });
                            let members =
                                keys.and_then(|keys| Self::in_list_expression(column, &keys));
                            if let Some(members) = members {
                                inner_keys = Some(members);
                                #[cfg(any(test, feature = "test-failpoints"))]
                                crate::test_failpoints::small_outer_joined();
                            }
                            PreparedRows::Shared(rows)
                        }
                        rows => rows,
                    };
                    prepared_side = Some(PreparedSide {
                        rows,
                        columns,
                        is_right: swapped,
                    });
                    prepared_ctx = Some(ctx_join);
                    break 'index_nl;
                }
                // The inner table opens before the outer side is read, so one
                // whose index is no longer the planned one joins by hash with
                // nothing read twice
                #[cfg(feature = "test-failpoints")]
                if statement_run.is_some() {
                    crate::test_failpoints::join_side_opening();
                }
                let Some(inner_table) = self.index_join_inner(
                    &snapshot,
                    ctx,
                    &table_name,
                    &lookup_strategy,
                    &inner_col,
                    None,
                )?
                else {
                    break 'index_nl;
                };
                // An explicit transaction reads through its own transaction, whose
                // isolation may let a commit move rows between two fetches, so
                // the outer side is fetched whole there
                let in_explicit_transaction = self.active_transaction.lock().unwrap().is_some();
                let mut outer_limit = if in_explicit_transaction {
                    None
                } else {
                    join_limit.map(|l| {
                        let l = l as usize;
                        l.saturating_add(l / 16).max(100)
                    })
                };

                // Execute outer side with limit optimization for true early termination
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::join_side_ran();
                let (outer_result, outer_cols, outer_bounded, outer_token) = self.read_join_side(
                    outer_expr,
                    &ctx_join,
                    outer_read,
                    outer_extra.as_ref(),
                    outer_limit,
                    0,
                )?;
                if !outer_bounded {
                    outer_limit = None;
                }

                // Find the outer key index in outer columns
                // OPTIMIZATION: Pre-compute lowercase column names to avoid per-column to_lowercase()
                let outer_cols_lower: Vec<String> =
                    outer_cols.iter().map(|c| c.to_lowercase()).collect();
                let outer_col_lower = outer_col.to_lowercase();
                let outer_key_idx = outer_cols_lower
                    .iter()
                    .position(|c| c == &outer_col_lower)
                    .or_else(|| {
                        // Try unqualified match
                        let outer_unqualified = bare_name(&outer_col_lower);
                        outer_cols_lower
                            .iter()
                            .position(|c_lower| bare_name(c_lower) == outer_unqualified)
                    });

                if let Some(outer_idx) = outer_key_idx {
                    let inner_token = inner_table.layout_token();
                    let inner_schema = inner_table.schema();
                    if let Some(run) = statement_run {
                        run.blocks
                            .side_fits(usize::from(!swapped), &inner_table.schema_arc())?;
                    }
                    // The WHERE's conditions on the inner table, fetched through
                    // when its scan takes them whole, else asked with the residual
                    let inner_taken = match statement_run {
                        Some(run) => self.join_side_filter(
                            run.blocks,
                            run.run,
                            usize::from(!swapped),
                            &*inner_table,
                            &ctx_join,
                        )?,
                        None => None,
                    };
                    let taken = inner_taken.is_some();
                    #[cfg(feature = "test-failpoints")]
                    if taken {
                        crate::test_failpoints::inner_fetch_filtered();
                    }

                    // Build inner columns list (qualified)
                    // When swapped, inner table alias is the original left alias
                    let inner_alias = if swapped {
                        left_alias.as_deref().unwrap_or(&table_name)
                    } else {
                        right_alias.as_deref().unwrap_or(&table_name)
                    };
                    let inner_cols: Vec<String> = inner_schema
                        .columns
                        .iter()
                        .map(|col| qualified(inner_alias, &col.name))
                        .collect();

                    // Joined rows read the sides in the statement's order
                    let (left_cols, right_cols) = if swapped {
                        (&inner_cols, &outer_cols)
                    } else {
                        (&outer_cols, &inner_cols)
                    };
                    let all_columns = {
                        let mut all = left_cols.clone();
                        all.extend(right_cols.iter().cloned());
                        all
                    };

                    // The probe answers one equality out of the ON clause, so
                    // the whole clause is asked of each pair it returns, next
                    // to the filter the WHERE put on the inner table. The
                    // operator asks it where the pair is formed, so a pair it
                    // turns away leaves an outer row still unmatched. The
                    // filter is built per pass since the operator owns it
                    // The inner key column, checked on each fetched row in
                    // place of the equality the residual filter leaves out
                    let inner_key_idx = inner_cols
                        .iter()
                        .position(|c| bare_name(c).eq_ignore_ascii_case(&inner_col));
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::join_inner_opened();
                    // The program is held with the plan, by the key equality
                    // it leaves out and the columns it reads
                    let compile_residual = || -> Result<BuiltPrograms> {
                        // With the statement's blocks, the clauses they
                        // rewrote, over the scope's names for each side
                        let (on, outer_names, inner_names, inner_filter) = match blocks.as_deref() {
                            Some(blocks) => {
                                let (outer, inner) = match swapped {
                                    true => (&blocks.right, &blocks.left),
                                    false => (&blocks.left, &blocks.right),
                                };
                                let inner_filter = match taken {
                                    true => None,
                                    false => blocks.sides[usize::from(!swapped)].clone(),
                                };
                                (blocks.on.as_ref(), outer, inner, inner_filter)
                            }
                            None => (
                                on_condition,
                                &outer_cols,
                                &inner_cols,
                                nl_right_filter
                                    .as_ref()
                                    .map(|rf| add_table_qualifier(rf, inner_alias)),
                            ),
                        };
                        // The equality the probe answers is left out, so
                        // only the rest of the clause is asked of each pair
                        let rest = on.and_then(|on| match swapped {
                            true => index_join_residual(
                                on,
                                inner_names,
                                outer_names,
                                (inner_key_idx, Some(outer_idx)),
                            ),
                            false => index_join_residual(
                                on,
                                outer_names,
                                inner_names,
                                (Some(outer_idx), inner_key_idx),
                            ),
                        });
                        let combined = match (rest, inner_filter) {
                            (Some(on), Some(f)) => Some(Expression::Infix(InfixExpression::new(
                                Token::new(TokenType::Operator, "AND", Position::default()),
                                Box::new(on),
                                "AND".to_string(),
                                Box::new(f),
                            ))),
                            (Some(on), None) => Some(on),
                            (None, Some(f)) => Some(f),
                            (None, None) => None,
                        };
                        combined
                            .map(|expr| match blocks.as_deref() {
                                Some(blocks) => {
                                    blocks.compile(&expr, true, &self.function_registry)
                                }
                                None => JoinFilter::new(
                                    &expr,
                                    left_cols,
                                    right_cols,
                                    &self.function_registry,
                                )
                                .map(|f| CompactArc::clone(f.program())),
                            })
                            .into_iter()
                            .collect()
                    };
                    let build_residual_filter = || -> Result<Option<JoinFilter>> {
                        let read = |programs: &[SharedProgram]| {
                            programs.first().map(|p| {
                                JoinFilter::from_program(CompactArc::clone(p)).with_context(ctx)
                            })
                        };
                        Ok(match &prepared {
                            Some(prepared) => prepared.programs(
                                |held| {
                                    matches!(held, JoinProgramKey::IndexResidual {
                                        swapped: s, outer_col: oc, inner_col: ic,
                                        outer_cols: ocs, inner_cols: ics, inner_taken: t,
                                    } if *s == swapped && *oc == outer_col && *ic == inner_col
                                        && *ocs == outer_cols && *ics == inner_cols && *t == taken)
                                },
                                || JoinProgramKey::IndexResidual {
                                    swapped,
                                    outer_col: outer_col.clone(),
                                    inner_col: inner_col.clone(),
                                    outer_cols: outer_cols.clone(),
                                    inner_cols: inner_cols.clone(),
                                    inner_taken: taken,
                                },
                                compile_residual,
                                read,
                            )?,
                            None => read(&compile_residual()?),
                        })
                    };

                    // Execute Index Nested Loop Join using operators
                    // Use batch version for NO LIMIT (reduces lock overhead from O(N) to O(1))
                    // Use streaming version for LIMIT queries (supports early termination)

                    let op_join_type = OperatorJoinType::parse(&join_type);

                    // Try to push projection into the join operator for ~2.3x speedup
                    // Eligibility: no cross filter, no ORDER BY, DISTINCT ON keys
                    // the projection carries, no aggregation/window, simple
                    // column references only
                    // Note: JoinProjection uses ColumnSource enum to preserve SELECT column order
                    // regardless of whether the join was swapped (outer/inner assignment)
                    let projection_pushdown = if cross_filter.is_none()
                        && stmt.order_by.is_empty()
                        && !classification.has_aggregation
                        && !classification.has_window_functions
                    {
                        compute_join_projection(&stmt.columns, &outer_cols, &inner_cols)
                    } else {
                        None
                    };
                    // Where each projected column comes from in the joined
                    // row, for DISTINCT ON to read the projected rows by
                    let pushdown_sources: Option<LayoutSources> = projection_pushdown
                        .as_ref()
                        .filter(|_| !stmt.distinct_on.is_empty())
                        .map(|proj| {
                            use crate::executor::operators::index_nested_loop::ColumnSource as Side;
                            proj.columns
                                .iter()
                                .enumerate()
                                .map(|(item, side)| ColumnSource {
                                    item: Some(item),
                                    input: Some(match (*side, swapped) {
                                        (Side::Outer(idx), false) | (Side::Inner(idx), true) => idx,
                                        (Side::Outer(idx), true) => inner_cols.len() + idx,
                                        (Side::Inner(idx), false) => outer_cols.len() + idx,
                                    }),
                                })
                                .collect()
                        });
                    let projection_pushdown = projection_pushdown.filter(|_| {
                        pushdown_sources.as_ref().is_none_or(|sources| {
                            stmt.distinct_on
                                .iter()
                                .all(|key| match bind_order_key(key, stmt) {
                                    OrderKey::Ordinal(_) => true,
                                    key => OutputLayout::position_over(sources, &all_columns, &key)
                                        .is_some(),
                                })
                        })
                    });

                    // A cross-table WHERE filter counts toward LIMIT, so it is
                    // applied per joined row rather than after collection
                    let mut cross_row_filter = match cross_filter {
                        Some(ref cross) => Some(JoinedWhere::new(
                            self.join_where(
                                prepared,
                                WhereSite::IndexJoin,
                                stmt,
                                cross,
                                &all_columns,
                                blocks.as_deref(),
                            )?,
                            blocks_run.as_ref(),
                            ctx,
                        )),
                        None => None,
                    };
                    if let Some(run) = &blocks_run {
                        run.ask_rows(outer_result.estimated_count());
                    }
                    let mut result_rows = RowVec::new();
                    // A primary key join without a limit fetches its rows in one
                    // batch; an index lookup streams, which costs less than the
                    // batch's key maps for it
                    let batch = join_limit.is_none()
                        && matches!(lookup_strategy, IndexLookupStrategy::PrimaryKey);
                    if !batch {
                        // Streaming INL for early termination with LIMIT. When the
                        // joined rows stop short of the limit and the outer chunk
                        // was full, the next pass continues with the outer rows
                        // after the chunk, sized from the match rate so far
                        let mut outer_result = outer_result;
                        let mut inner_table = Some(inner_table);
                        let mut fetched = 0usize;
                        let mut consistent = false;
                        loop {
                            let outer_op: Box<dyn Operator> =
                                Box::new(QueryResultOperator::new(outer_result, outer_cols.len()));
                            let table = match inner_table.take() {
                                Some(table) => table,
                                None => match self.index_join_inner(
                                    &snapshot,
                                    ctx,
                                    &table_name,
                                    &lookup_strategy,
                                    &inner_col,
                                    Some(inner_token),
                                )? {
                                    Some(table) => table,
                                    // The rows joined so far are dropped too
                                    None => break 'index_nl,
                                },
                            };
                            JoinPath::IndexNestedLoop.taken();
                            let mut op = IndexNestedLoopJoinOperator::new(
                                outer_op,
                                table,
                                inner_cols.len(),
                                op_join_type,
                                outer_idx,
                                lookup_strategy.clone(),
                                build_residual_filter()?,
                            )
                            .with_inner_key(inner_key_idx)
                            .with_blocks(pair_blocks);
                            if let Some(Some(filter)) = &inner_taken {
                                op = op.with_inner_filter(filter.clone_box());
                            }
                            if swapped {
                                op = op.with_outer_on_right();
                            }
                            if let Some(ref proj) = projection_pushdown {
                                op = op.with_projection(proj.columns.clone());
                            }
                            Self::collect_join_rows(
                                &mut op,
                                join_limit,
                                cross_row_filter.as_mut(),
                                where_blocks,
                                &mut result_rows,
                            )?;
                            // A probe the inner table could not answer: the rows
                            // joined so far are dropped, and the whole join runs
                            // once more on the hash path
                            if op.needs_fallback() {
                                break 'index_nl;
                            }
                            let lim = join_limit.unwrap_or(0) as usize;
                            let cap = match outer_limit {
                                Some(cap) if op.outer_rows_seen() == cap => cap,
                                _ => break,
                            };
                            if result_rows.len() >= lim {
                                break;
                            }
                            let boundary = op.last_outer_row().cloned();
                            fetched = fetched.saturating_add(cap);
                            let missing = lim - result_rows.len();
                            let rate = result_rows.len() as f64 / fetched as f64;
                            let estimate = if rate > 0.0 {
                                ((missing as f64 / rate) * 2.0).ceil() as usize
                            } else {
                                usize::MAX
                            };
                            let next = estimate.clamp(32, fetched.saturating_mul(3));
                            outer_limit = Some(next);
                            #[cfg(feature = "test-failpoints")]
                            crate::test_failpoints::join_next_chunk();
                            if consistent {
                                let (more, _, bounded, token) = self.read_join_side(
                                    outer_expr,
                                    &ctx_join,
                                    outer_read,
                                    outer_extra.as_ref(),
                                    Some(next),
                                    fetched,
                                )?;
                                same_outer_source(outer_expr, outer_token, token, bounded)?;
                                outer_result = more;
                                continue;
                            }
                            // Under read committed a commit may have moved rows before
                            // the boundary. Fetch one row of overlap: when it is still
                            // the row the chunk ended on, the rows after it are the
                            // continuation; otherwise the join restarts on snapshot
                            // isolation, where the chunks line up by construction
                            let (mut more, _, bounded, token) = self.read_join_side(
                                outer_expr,
                                &ctx_join,
                                outer_read,
                                outer_extra.as_ref(),
                                Some(next + 1),
                                fetched - 1,
                            )?;
                            same_outer_source(outer_expr, outer_token, token, bounded)?;
                            let unmoved = more.next()
                                && boundary
                                    .as_ref()
                                    .is_some_and(|row| row.as_slice() == more.row().as_slice());
                            if unmoved {
                                outer_result = more;
                                continue;
                            }
                            snapshot = self.new_statement_snapshot(
                                crate::core::IsolationLevel::SnapshotIsolation,
                                ctx.statement_view(),
                            )?;
                            ctx_join = ctx.with_statement_snapshot(snapshot.clone());
                            consistent = true;
                            result_rows.clear();
                            outer_limit = Some(fetched.saturating_add(next));
                            fetched = 0;
                            let (restarted, _, bounded, token) = self.read_join_side(
                                outer_expr,
                                &ctx_join,
                                outer_read,
                                outer_extra.as_ref(),
                                outer_limit,
                                0,
                            )?;
                            same_outer_source(outer_expr, outer_token, token, bounded)?;
                            outer_result = restarted;
                        }
                    } else {
                        // Batch INL for NO LIMIT - single batch fetch, O(1) lock overhead
                        let outer_op: Box<dyn Operator> =
                            Box::new(QueryResultOperator::new(outer_result, outer_cols.len()));
                        JoinPath::BatchIndexNestedLoop.taken();
                        let mut op = BatchIndexNestedLoopJoinOperator::new(
                            outer_op,
                            inner_table,
                            inner_cols.len(),
                            op_join_type,
                            outer_idx,
                            lookup_strategy.clone(),
                            build_residual_filter()?,
                        )
                        .with_inner_key(inner_key_idx)
                        .with_blocks(pair_blocks);
                        if let Some(Some(filter)) = inner_taken {
                            op = op.with_inner_filter(filter);
                        }
                        if swapped {
                            op = op.with_outer_on_right();
                        }
                        if let Some(ref proj) = projection_pushdown {
                            op = op.with_projection(proj.columns.clone());
                        }
                        Self::collect_join_rows(
                            &mut op,
                            None,
                            cross_row_filter.as_mut(),
                            where_blocks,
                            &mut result_rows,
                        )?;
                    }

                    let mut final_rows = result_rows;

                    // Check for aggregation/window functions that need special handling
                    let has_agg = classification.has_aggregation;
                    let has_window = classification.has_window_functions;

                    // Apply ORDER BY if present. Aggregation/window queries
                    // order after aggregating, so they see the untruncated
                    // rows. An ORDER BY expression the join-local
                    // evaluator cannot resolve (SELECT aliases, positions)
                    // also defers ordering to the standard path.
                    // An ordinal, or a name bound to an alias's column, reads
                    // the output, which only the standard path holds
                    let reads_output = stmt.order_by.iter().any(|ob| {
                        matches!(
                            bind_order_key(&ob.expression, stmt),
                            OrderKey::Ordinal(_) | OrderKey::Item(_)
                        )
                    });
                    let mut order_limit_applied = false;
                    // DISTINCT merges rows before the LIMIT counts them, so
                    // its ordering and LIMIT are left to the standard path
                    let distinct = stmt.distinct || !stmt.distinct_on.is_empty();
                    if !has_agg
                        && !has_window
                        && !reads_output
                        && !distinct
                        && !stmt.order_by.is_empty()
                    {
                        // Compute sort keys and indices; compile and
                        // evaluation errors both defer to the standard path
                        let compile = || -> Result<BuiltPrograms> {
                            let outer_columns = outer_column_names(ctx);
                            let mut compile_ctx =
                                CompileContext::new(&all_columns, &self.function_registry);
                            if let Some(outer) = &outer_columns {
                                compile_ctx = compile_ctx.with_outer_columns(outer);
                            }
                            stmt.order_by
                                .iter()
                                .map(|ob| {
                                    ExprCompiler::new(&compile_ctx)
                                        .compile(&ob.expression)
                                        .map(CompactArc::new)
                                        .map_err(|e| {
                                            Error::internal(format!("Compile error: {}", e))
                                        })
                                })
                                .collect()
                        };
                        let keys_of = |programs: &[SharedProgram]| {
                            let mut vm = ExprVM::new();
                            let mut sort_keys: Vec<Vec<Value>> =
                                Vec::with_capacity(final_rows.len());
                            for (_, row) in final_rows.iter() {
                                let exec = super::expression::execute_context(row, Some(ctx));
                                let mut keys = Vec::with_capacity(programs.len());
                                for p in programs {
                                    keys.push(vm.execute(p, &exec).ok()?);
                                }
                                sort_keys.push(keys);
                            }
                            Some(sort_keys)
                        };
                        let sort_keys = match &prepared {
                            Some(prepared) => prepared.programs(
                                |held| {
                                    matches!(held, JoinProgramKey::OrderKeys { columns }
                                        if *columns == all_columns)
                                },
                                || JoinProgramKey::OrderKeys {
                                    columns: all_columns.clone(),
                                },
                                compile,
                                keys_of,
                            ),
                            None => compile().map(|programs| keys_of(&programs)),
                        };
                        if let Ok(Some(sort_keys)) = sort_keys {
                            // Sort by indices with the shared NULL-aware
                            // comparator (the standard path's semantics)
                            let mut indices: Vec<usize> = (0..final_rows.len()).collect();
                            indices.sort_unstable_by(|&a, &b| {
                                Self::compare_by_sort_keys_impl(
                                    &sort_keys[a],
                                    &sort_keys[b],
                                    &stmt.order_by,
                                    stmt.order_by.len(),
                                )
                            });

                            // Apply LIMIT/OFFSET while reordering: with a
                            // limit only the needed rows move; otherwise the
                            // permutation happens in place instead of
                            // deep-cloning every joined row
                            let counts = row_counts(stmt, ctx)?;
                            let offset = counts.offset;
                            let limit = counts.limit.unwrap_or(usize::MAX);

                            // Sorted here, and LIMIT/OFFSET applied below, so
                            // the standard path must not apply OFFSET again
                            order_limit_applied = true;

                            if offset > 0 || limit != usize::MAX {
                                OrderLimitPath::JoinIndexSort.taken();
                                final_rows = indices
                                    .into_iter()
                                    .skip(offset)
                                    .take(limit)
                                    .map(|i| {
                                        (final_rows[i].0, std::mem::take(&mut final_rows[i].1))
                                    })
                                    .collect();
                            } else {
                                // In-place cycle-based permutation (same
                                // pattern as the single-table ORDER BY path)
                                let n = final_rows.len();
                                let mut indices = indices;
                                for start in 0..n {
                                    if indices[start] == start || indices[start] == usize::MAX {
                                        continue;
                                    }
                                    let mut current = start;
                                    loop {
                                        let target = indices[current];
                                        if target == start {
                                            indices[current] = usize::MAX;
                                            break;
                                        }
                                        final_rows.swap(current, target);
                                        indices[current] = usize::MAX;
                                        current = target;
                                    }
                                }
                            }
                        }
                    }

                    let merging = blocks
                        .as_deref()
                        .filter(|_| side_merged && !apart && projection_pushdown.is_none());
                    let (all_columns, final_rows) =
                        merged_side_rows(stmt, merging, all_columns, final_rows);
                    let held = if has_agg || has_window || projection_pushdown.is_some() {
                        None
                    } else {
                        self.held_join_output(
                            prepared,
                            stmt,
                            &all_columns,
                            order_limit_applied,
                            blocks.as_deref(),
                            ctx,
                        )?
                    };
                    if has_agg || has_window {
                        return self.grouped_join_select(
                            grouped_plan.as_deref(),
                            stmt,
                            ctx,
                            classification,
                            final_rows,
                            &all_columns,
                            left_cols.len(),
                        );
                    } else if let Some(output) = held.as_deref().and_then(Option::as_ref) {
                        return output_result(output, final_rows, order_limit_applied);
                    } else if let Some(proj) = projection_pushdown {
                        // Projection was pushed down - rows are already projected
                        let output_columns = CompactArc::new(proj.output_columns);
                        let result = ExecutorResult::with_arc_columns(
                            CompactArc::clone(&output_columns),
                            final_rows,
                        );
                        return Ok((
                            Box::new(result),
                            output_columns,
                            order_limit_applied,
                            SelectShape::laid_out(pushdown_sources.map(|sources| OutputLayout {
                                input_columns: CompactArc::new(all_columns),
                                sources,
                            })),
                        ));
                    } else if !order_limit_applied
                        && self.order_by_needs_extra_columns(stmt, &all_columns)
                    {
                        // Ordering left to the standard path keeps the ORDER BY
                        // and DISTINCT ON columns the SELECT does not hold
                        let (projected_rows, extra_names, layout) = self
                            .project_rows_with_order_by(
                                &stmt.columns,
                                &stmt.order_by,
                                &stmt.distinct_on,
                                stmt.table_expr.as_deref(),
                                final_rows,
                                &CompactArc::new(all_columns.clone()),
                                None,
                                ctx,
                            )?;
                        let mut output_columns =
                            self.get_output_column_names(&stmt.columns, &all_columns, None);
                        output_columns.extend(extra_names);
                        let output_columns = CompactArc::new(output_columns);
                        let result = ExecutorResult::with_arc_columns(
                            CompactArc::clone(&output_columns),
                            projected_rows,
                        );
                        return Ok((
                            Box::new(result),
                            output_columns,
                            false,
                            SelectShape::laid_out(Some(layout)),
                        ));
                    } else {
                        // Project rows according to SELECT expressions
                        let projected_rows = self.project_rows_with_alias(
                            &stmt.columns,
                            final_rows,
                            &all_columns,
                            None,
                            ctx,
                            None,
                        )?;
                        let output_columns = CompactArc::new(self.get_output_column_names(
                            &stmt.columns,
                            &all_columns,
                            None,
                        ));

                        // Return with projected results
                        let result = ExecutorResult::with_arc_columns(
                            CompactArc::clone(&output_columns),
                            projected_rows,
                        );
                        let layout = Self::output_layout(stmt, &all_columns, None);
                        return Ok((
                            Box::new(result),
                            output_columns,
                            order_limit_applied,
                            SelectShape::laid_out(layout),
                        ));
                    }
                }
            }

            // =================================================================
            // STREAMING HASH JOIN OPTIMIZATION
            // =================================================================
            // For queries with small LIMIT, use streaming execution to avoid
            // materializing the probe side. This enables true early termination.
            //
            // Eligibility:
            // - LIMIT ≤ 100 (small limit benefits from early termination)
            // - INNER JOIN (simpler, no NULL padding needed)
            // - Has equality keys (hash join applicable)
            // - No ORDER BY, GROUP BY, aggregation (would negate early termination)
            // - No window function, which only the standard path evaluates
            // - No DISTINCT, which reads every joined row before the LIMIT
            // - No cross-table predicates (applied after join)

            // OFFSET rows are skipped after collection, so the early-termination
            // budget is LIMIT + OFFSET, for a LIMIT of at most 100
            let streaming = match join_plan.limit_hash {
                true => row_counts(stmt, ctx).ok().and_then(|counts| {
                    let limit = counts.limit.filter(|limit| *limit <= 100)?;
                    let build_left = self.limit_hash_builds_left(
                        [&join_source.left, &join_source.right],
                        [left_filter.as_ref(), right_filter.as_ref()],
                    );
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::limit_hash_built(build_left);
                    Some((limit.saturating_add(counts.offset) as u64, build_left))
                }),
                false => None,
            };

            // Use streaming path if eligible; sides it ran but could not join
            // by hash go on to the standard path
            let streamed = 'streaming: {
                let Some((limit, build_left)) = streaming else {
                    break 'streaming None;
                };
                // Execute both sides
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::join_side_ran();
                let (left_result, left_cols, _, _) =
                    self.read_join_side(&join_source.left, ctx, left_read, None, None, 0)?;

                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::join_side_ran();
                let (right_result, right_cols, _, _) =
                    self.read_join_side(&join_source.right, ctx, right_read, None, None, 0)?;

                // Extract join keys BEFORE moving columns (uses original left/right positions)
                let (left_key_indices, right_key_indices, _) = if let Some(cond) = on_condition {
                    extract_join_keys_and_residual(cond, &left_cols, &right_cols)
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };

                // Build combined columns (always left-first for consistent output schema)
                let mut all_cols = left_cols.clone();
                all_cols.extend(right_cols.iter().cloned());

                // Materialize the smaller side as build, stream the larger as probe
                let (build_rows, build_cols, probe_result, probe_cols, build_is_left) =
                    if build_left {
                        let build = Self::materialize_result_arc(left_result)?;
                        (build, left_cols, right_result, right_cols, true)
                    } else {
                        let build = Self::materialize_result_arc(right_result)?;
                        (build, right_cols, left_result, left_cols, false)
                    };

                // Map key indices based on which side is build
                let (build_key_indices, probe_key_indices) = if build_is_left {
                    (left_key_indices, right_key_indices)
                } else {
                    (right_key_indices, left_key_indices)
                };

                // A key pair where a number may meet a TEXT is no hash key.
                // A base table side's schema says what it holds, else the
                // build rows do. One key over a build column of one family
                // has a probe key of the other answered as it streams; any
                // other such pair has the probe read whole and joined by
                // the standard path
                let (build_expr, probe_expr) = if build_is_left {
                    (&join_source.left, &join_source.right)
                } else {
                    (&join_source.right, &join_source.left)
                };
                let build_schema = self.base_table_side_schema(build_expr, CteNames::Context(ctx));
                let probe_schema = self.base_table_side_schema(probe_expr, CteNames::Context(ctx));

                // Skip bloom filter for small LIMIT queries - the overhead of building
                // a bloom filter for thousands of rows isn't worth it when we only
                // need a few results. With streaming + early termination, we don't
                // probe many rows anyway.
                let skip_bloom_for_small_limit = limit <= 500;
                let wants_bloom = build_rows.len() >= 100
                    && !build_key_indices.is_empty()
                    && !skip_bloom_for_small_limit;
                let new_bloom = || {
                    BloomFilterBuilder::new(
                        "join_key".to_string(),
                        "build".to_string(),
                        build_rows.len(),
                    )
                };
                // One key no schema types: the pass that builds its hash table
                // tells what the build column holds
                let typed = |idx: usize| {
                    let name = bare_name(&build_cols[idx]);
                    build_schema
                        .as_ref()
                        .and_then(|s| s.get_column_type(name))
                        .is_some()
                };
                let one_key_built = match build_key_indices.as_slice() {
                    &[build_idx] if !typed(build_idx) => {
                        let mut builder = wants_bloom.then(new_bloom);
                        let (table, kinds) =
                            JoinHashTable::build_one_key(&build_rows, build_idx, builder.as_mut());
                        Some((table, builder, kinds))
                    }
                    _ => None,
                };
                let mut cross_key = None;
                let mut unverified = false;
                for (&build_idx, &probe_idx) in build_key_indices.iter().zip(&probe_key_indices) {
                    let build = match &one_key_built {
                        Some((_, _, kinds)) => *kinds,
                        None => column_kinds(
                            &build_rows,
                            &build_cols,
                            build_schema.as_deref(),
                            build_idx,
                        ),
                    };
                    if build & (kind::TEXT | kind::READ_FROM_TEXT) == 0 {
                        continue;
                    }
                    let probe_column = &probe_cols[probe_idx];
                    let name = bare_name(probe_column);
                    let crosses = probe_schema
                        .as_ref()
                        .and_then(|s| s.get_column_type(name))
                        .is_none_or(|probe| kind::crosses(build, kind::of_type(probe)));
                    match KeySide::of_kinds(build) {
                        Some(side) if crosses && build_key_indices.len() == 1 => {
                            cross_key = Some(side);
                        }
                        _ if crosses => unverified = true,
                        _ => {}
                    }
                }
                if unverified {
                    let probe_rows = Self::materialize_result_arc(probe_result)?;
                    break 'streaming Some(if build_is_left {
                        (build_rows, build_cols, probe_rows, probe_cols)
                    } else {
                        (probe_rows, probe_cols, build_rows, build_cols)
                    });
                }

                // Convert probe side to streaming operator
                let probe_source: Box<dyn Operator> =
                    Box::new(QueryResultOperator::new(probe_result, probe_cols.len()));

                // Build hash table and bloom filter together in a single pass
                // This avoids iterating build_rows twice (once for bloom, once for hash table)
                // A probe key the comparison reads as another type would miss
                // a bloom filter of the build keys as they are
                let (bloom_filter, pre_built_hash_table) =
                    if let Some((table, builder, _)) = one_key_built {
                        let bloom = builder
                            .filter(|_| cross_key.is_none())
                            .map(BloomFilterBuilder::build)
                            .filter(|bf| bf.is_effective());
                        (bloom, Some(table))
                    } else if wants_bloom && cross_key.is_none() {
                        let mut builder = new_bloom();
                        // Single-pass: build hash table and populate bloom filter
                        let hash_table = JoinHashTable::build_with_bloom(
                            &build_rows,
                            &build_key_indices,
                            &mut builder,
                        );
                        let bf = builder.build();
                        let bloom = if bf.is_effective() { Some(bf) } else { None };
                        (bloom, Some(hash_table))
                    } else if !build_key_indices.is_empty() {
                        // No bloom filter, but still pre-build hash table
                        let hash_table = JoinHashTable::build(&build_rows, &build_key_indices);
                        (None, Some(hash_table))
                    } else {
                        (None, None)
                    };

                // Wrap probe with bloom filter if available
                let probe_source: Box<dyn Operator> = match bloom_filter {
                    Some(bf) if !probe_key_indices.is_empty() => Box::new(
                        BloomFilterOperator::new(probe_source, bf, probe_key_indices),
                    ),
                    _ => probe_source,
                };

                // Execute streaming hash join
                let join_executor = JoinExecutor::new();
                let streaming_request = StreamingJoinRequest {
                    build_rows,
                    build_columns: &build_cols,
                    probe_source,
                    probe_columns: probe_cols.clone(),
                    condition: on_condition,
                    join_type: &join_type,
                    build_is_left,
                    limit: Some(limit),
                    ctx,
                    pre_built_hash_table,
                    cross_key,
                    prepared,
                    blocks: statement_run,
                };

                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::streaming_join_streamed();
                let join_result = join_executor.execute_streaming(streaming_request)?;
                let merging = blocks.as_deref().filter(|_| side_merged && !apart);
                let (all_cols, joined_rows) =
                    merged_side_rows(stmt, merging, all_cols, join_result.rows);
                let held = self.held_join_output(
                    prepared,
                    stmt,
                    &all_cols,
                    false,
                    blocks.as_deref(),
                    ctx,
                )?;
                if let Some(output) = held.as_deref().and_then(Option::as_ref) {
                    return output_result(output, joined_rows, false);
                }

                // Project and return results
                let projected_rows = self.project_rows_with_alias(
                    &stmt.columns,
                    joined_rows,
                    &all_cols,
                    None,
                    ctx,
                    None,
                )?;
                let output_columns =
                    CompactArc::new(self.get_output_column_names(&stmt.columns, &all_cols, None));

                let result = ExecutorResult::with_arc_columns(
                    CompactArc::clone(&output_columns),
                    projected_rows,
                );
                return Ok((Box::new(result), output_columns, false, None));
            };

            // =================================================================
            // STANDARD PATH: Materialize both sides
            // =================================================================

            if let Some(sides) = streamed {
                sides
            } else {
                // Execute both sides; a side the index join already ran is taken
                // as it left it, and the other side is read in the same context
                let side_ctx = prepared_ctx.as_ref().unwrap_or(ctx);
                let run_side =
                    |expr: &Expression, read: SideRead<'_>, keys: Option<&Expression>| {
                        #[cfg(any(test, feature = "test-failpoints"))]
                        crate::test_failpoints::join_side_ran();
                        let (result, columns, _, _) =
                            self.read_join_side(expr, side_ctx, read, keys, None, 0)?;
                        Ok::<_, Error>((Self::materialize_result_arc(result)?, columns))
                    };
                match prepared_side.take() {
                    Some(PreparedSide {
                        rows,
                        columns,
                        is_right: true,
                    }) => {
                        let (left_rows, left_cols) =
                            run_side(&join_source.left, left_read, inner_keys.as_ref())?;
                        (left_rows, left_cols, rows.into_rows()?, columns)
                    }
                    Some(PreparedSide {
                        rows,
                        columns,
                        is_right: false,
                    }) => {
                        let (right_rows, right_cols) =
                            run_side(&join_source.right, right_read, inner_keys.as_ref())?;
                        (rows.into_rows()?, columns, right_rows, right_cols)
                    }
                    None => {
                        let (left_rows, left_cols) = run_side(&join_source.left, left_read, None)?;
                        let (right_rows, right_cols) =
                            run_side(&join_source.right, right_read, None)?;
                        (left_rows, left_cols, right_rows, right_cols)
                    }
                }
            }
        };

        // Combine column names (qualified with table aliases)
        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.clone());

        // Handle NATURAL JOIN or USING clause by automatically finding common columns
        let natural_join_condition = natural_join_condition(
            &join_type,
            join_source,
            &left_columns,
            &right_columns,
            |right, i| {
                let name = if right {
                    &right_columns[i]
                } else {
                    &left_columns[i]
                };
                Expression::Identifier(Identifier::new(
                    Token::new(TokenType::Identifier, name.clone(), Position::default()),
                    name.clone(),
                ))
            },
        );

        // Destructure the tuple: (condition, excluded_column_indices, column_renames)
        let (natural_join_cond, mut excluded_column_indices, mut join_col_renames) =
            natural_join_condition;
        // Run apart, it keeps both columns of each pair it merges
        if apart || output_apart {
            excluded_column_indices.clear();
            join_col_renames.clear();
        }
        let side_names = match blocks.as_deref().filter(|_| side_merged && !apart) {
            Some(held) => {
                let lists = (&mut excluded_column_indices, &mut join_col_renames);
                side_pairs(stmt, held, &all_columns, lists)
            }
            None => Vec::new(),
        };

        // Use natural join condition if present, otherwise use explicit
        // condition, whose subqueries were read at the top of the function
        let effective_condition = natural_join_cond.as_ref().or(on_condition);

        // =================================================================
        // Execute JOIN using streaming JoinExecutor
        // =================================================================
        // JoinExecutor handles:
        // - Algorithm selection (Hash Join, Merge Join, Nested Loop)
        // - Build/probe side optimization for hash joins
        // - Merge join for pre-sorted inputs
        // - Residual filter application (non-equality conditions)
        // - Early termination with LIMIT

        // Compute LIMIT for early termination pushdown
        // Safe to push when: no ORDER BY, no GROUP BY/aggregation, no window
        // function or DISTINCT, which read every joined row, no FULL OUTER
        let can_push_limit = !join_type.contains("FULL")
            && !classification.has_order_by
            && cross_filter.is_none()
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !stmt.distinct
            && stmt.distinct_on.is_empty();

        let join_limit = if can_push_limit {
            // OFFSET rows are skipped after collection, so early termination
            // must gather LIMIT + OFFSET rows
            rows_needed(stmt, ctx).map(|rows| rows as u64)
        } else {
            None
        };

        // Execute join using JoinExecutor (takes ownership of rows)
        let join_executor = JoinExecutor::new();
        let join_request = super::join_executor::JoinRequest {
            left_rows,
            right_rows,
            left_columns: &left_columns,
            right_columns: &right_columns,
            condition: effective_condition,
            join_type: &join_type,
            limit: join_limit,
            ctx,
            left_schema: self.base_table_side_schema(&join_source.left, CteNames::Context(ctx)),
            right_schema: self.base_table_side_schema(&join_source.right, CteNames::Context(ctx)),
            prepared,
            blocks: statement_run,
        };

        let join_result = join_executor.execute(join_request)?;
        let result_rows = join_result.rows;

        // Apply remaining WHERE clause if present (after filter pushdown)
        // IMPORTANT: When predicates were pushed to left/right (left_filter or right_filter is Some),
        // we should ONLY apply cross_filter (predicates that reference both tables).
        // If cross_filter is None and any pushdown happened, we don't need post-join filtering.
        // Only use stmt.where_clause when NO pushdown happened at all.
        let did_any_pushdown = where_split;
        let effective_where = if did_any_pushdown {
            // Pushdown happened - only apply cross predicates (if any)
            cross_filter.clone()
        } else {
            // No pushdown - apply full WHERE clause
            where_for_join.clone()
        };

        // Apply WHERE clause if present
        let filtered_rows = if let Some(where_expr) = effective_where {
            if per_row_where {
                let resolved = Self::with_select_aliases(stmt, &where_expr, &all_columns)
                    .unwrap_or(where_expr);
                let where_clause = &resolved;
                // Each joined row is the outer row of the subqueries in the
                // WHERE, so they are resolved and the predicate compiled per row
                let keys = ColumnKeyMapping::build_mappings(&all_columns, None);
                let outer_columns = CompactArc::new(all_columns.clone());
                let joined_qualifiers: Vec<String> = {
                    let mut qualifiers: Vec<String> = all_columns
                        .iter()
                        .filter_map(|c| split_qualified(c).map(|(q, _)| q.to_lowercase()))
                        .collect();
                    qualifiers.sort_unstable();
                    qualifiers.dedup();
                    qualifiers
                };
                let joined_qualifiers: Vec<&str> =
                    joined_qualifiers.iter().map(String::as_str).collect();
                let mut filtered = RowVec::with_capacity(result_rows.len());
                for (id, row) in result_rows {
                    // The row of a query around this one stays visible to the subqueries
                    let mut outer: FxHashMap<CompactArc<str>, Value> =
                        ctx.outer_row().cloned().unwrap_or_default();
                    for key in &keys {
                        if let Some(value) = row.get(key.index) {
                            if let Some(unqualified) = &key.unqualified_part {
                                outer.insert(unqualified.clone(), value.clone());
                            }
                            outer.insert(key.col_lower.clone(), value.clone());
                        }
                    }
                    let row_ctx = ctx.with_outer_row(outer, outer_columns.clone());
                    let processed = self.process_correlated_where(where_clause, &row_ctx)?;
                    ctx.check_cancelled()?;
                    // A column of the query around this one is not in the joined
                    // row, so a reference through the parent's table takes its
                    // value from the context; one through a joined table's name
                    // is the row's own, whatever bare name it shares
                    let bound = match row_ctx.outer_row() {
                        Some(outer) => {
                            let scope = super::utils::InnerScope {
                                tables: &joined_qualifiers,
                                schema: None,
                            };
                            super::utils::substitute_outer_references_in_scope(
                                &processed, outer, &scope,
                            )
                        }
                        None => processed,
                    };
                    if RowFilter::new(&bound, &all_columns)?
                        .with_context(&row_ctx)
                        .matches_checked(&row)?
                    {
                        filtered.push((id, row));
                    }
                }
                filtered
            } else {
                // Its subqueries run as the plan's blocks, or were folded
                // before the filters were split
                let program = self.join_where(
                    prepared,
                    WhereSite::AfterJoin,
                    stmt,
                    &where_expr,
                    &all_columns,
                    blocks.as_deref(),
                )?;
                let mut where_filter = JoinedWhere::new(program, blocks_run.as_ref(), ctx);
                if let Some(run) = &blocks_run {
                    run.ask_rows(Some(result_rows.len()));
                }
                let mut filtered = RowVec::with_capacity(result_rows.len());
                for (id, row) in result_rows {
                    if where_filter.matches(&row, where_blocks)? {
                        filtered.push((id, row));
                    }
                }
                filtered
            }
        } else {
            result_rows
        };

        let (mut final_columns, final_rows) = merged_columns(
            stmt,
            &all_columns,
            filtered_rows,
            (&excluded_column_indices, &join_col_renames),
        );
        for (at, name) in side_names {
            final_columns[at] = name;
        }

        if classification.has_aggregation || classification.has_window_functions {
            return self.grouped_join_select(
                grouped_plan.as_deref(),
                stmt,
                ctx,
                classification,
                final_rows,
                &final_columns,
                left_columns.len(),
            );
        }

        let held = self.held_join_output(
            prepared,
            stmt,
            &final_columns,
            false,
            blocks.as_deref(),
            ctx,
        )?;
        if let Some(output) = held.as_deref().and_then(Option::as_ref) {
            return output_result(output, final_rows, false);
        }

        // Check if ORDER BY or DISTINCT ON references columns not in SELECT
        let join_needs_extra_columns = self.order_by_needs_extra_columns(stmt, &final_columns);

        // Project rows according to SELECT expressions
        let (projected_rows, output_columns, layout) = if join_needs_extra_columns {
            // Use projection that preserves extra ORDER BY / DISTINCT ON columns
            let (projected_rows, extra_names, layout) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                stmt.table_expr.as_deref(),
                final_rows,
                &CompactArc::new(final_columns.clone()),
                None,
                ctx,
            )?;
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &final_columns, None);
            output_columns.extend(extra_names);
            (
                projected_rows,
                CompactArc::new(output_columns),
                Some(layout),
            )
        } else {
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                final_rows,
                &final_columns,
                None,
                ctx,
                None,
            )?;
            // Determine output column names
            // Note: For JOIN results, columns are already qualified (e.g., "a.id", "b.id"),
            // so we pass None for table_alias - the prefix matching will work
            let output_columns =
                CompactArc::new(self.get_output_column_names(&stmt.columns, &final_columns, None));
            let layout = Self::output_layout(stmt, &final_columns, None);
            (projected_rows, output_columns, layout)
        };

        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((
            Box::new(result),
            output_columns,
            false,
            SelectShape::laid_out(layout),
        ))
    }

    /// Keep the rows of an in-memory source that pass a WHERE holding a
    /// correlated subquery, each row bound as the subquery's outer row
    pub(crate) fn filter_rows_by_correlated_where(
        &self,
        where_expr: &Expression,
        rows: RowVec,
        columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        let mut evaluator = CompiledEvaluator::new(&self.function_registry).with_context(ctx);
        evaluator.init_columns(columns);
        let column_keys = ColumnKeyMapping::build_mappings(columns, table_alias);
        let columns_arc = CompactArc::new(columns.to_vec());
        let mut outer_row_map: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
        let mut kept = RowVec::with_capacity(rows.len());
        for (id, row) in rows {
            outer_row_map.clear();
            for mapping in &column_keys {
                if let Some(value) = row.get(mapping.index) {
                    if let Some(ref upart) = mapping.unqualified_part {
                        outer_row_map.insert(upart.clone(), value.clone());
                    }
                    if let Some(ref qname) = mapping.qualified_name {
                        outer_row_map.insert(qname.clone(), value.clone());
                    }
                    outer_row_map.insert(mapping.col_lower.clone(), value.clone());
                }
            }
            let mut correlated_ctx =
                ctx.with_outer_row(std::mem::take(&mut outer_row_map), columns_arc.clone());
            let processed = self.process_correlated_where(where_expr, &correlated_ctx)?;
            ctx.check_cancelled()?;
            evaluator.set_outer_row_owned(correlated_ctx.outer_row.take().unwrap_or_default());
            evaluator.set_row_array(&row);
            let keep = evaluator.evaluate_bool(&processed)?;
            outer_row_map = evaluator.take_outer_row();
            if keep {
                kept.push((id, row));
            }
        }
        Ok(kept)
    }

    /// Execute a subquery source
    fn execute_subquery_source(
        &self,
        subquery_source: &SubqueryTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        plan: Option<&NestedPlan>,
    ) -> SelectResult {
        // classification is passed from caller for the outer stmt
        // Note: The inner subquery will get its own classification via execute_select

        // Execute subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let body = &subquery_source.subquery;
        let result = self.execute_select_with_plan(body, &subquery_ctx, None, plan)?;
        let columns = result.columns().to_vec();

        // OPTIMIZATION: For simple GROUP BY aggregation without WHERE clause, try streaming
        // directly to aggregation HashMap without materializing all rows first.
        // This reduces memory allocations from O(N) to O(groups).
        if stmt.where_clause.is_none()
            && classification.has_aggregation
            && !classification.has_window_functions
        {
            let result = match self.try_streaming_derived_table_aggregation(
                result,
                stmt,
                ctx,
                classification,
            )? {
                Ok(agg_result) => {
                    let out_columns = CompactArc::new(agg_result.columns().to_vec());
                    let alias = subquery_source
                        .alias
                        .as_ref()
                        .map(|a| a.value_lower.as_str());
                    let layout = Self::result_layout(stmt, &columns, alias, &out_columns);
                    return Ok((
                        agg_result,
                        out_columns,
                        false,
                        SelectShape::laid_out(layout),
                    ));
                }
                Err(unread) => unread,
            };

            // Fall through to materialization path
            let mut rows = RowVec::new();
            let mut result = result;
            let mut row_id = 0i64;
            while result.next() {
                rows.push((row_id, result.take_row()));
                row_id += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }

            let agg_result =
                self.execute_select_with_aggregation(stmt, ctx, rows, &columns, None)?;
            let out_columns = CompactArc::new(agg_result.columns().to_vec());
            let alias = subquery_source
                .alias
                .as_ref()
                .map(|a| a.value_lower.as_str());
            let layout = Self::result_layout(stmt, &columns, alias, &out_columns);
            return Ok((
                agg_result,
                out_columns,
                false,
                SelectShape::laid_out(layout),
            ));
        }

        // Materialize the subquery result directly with synthetic row IDs
        let mut rows = RowVec::new();
        let mut result = result;
        let mut row_id = 0i64;
        while result.next() {
            rows.push((row_id, result.take_row()));
            row_id += 1;
        }
        if let Some(err) = result.last_error() {
            return Err(err);
        }

        // Apply WHERE clause if present. A parent row's column named through
        // its own table is bound first, or the bare name it would fall
        // back to could be this source's own
        let filtered_rows: RowVec = if let Some(ref where_clause) = stmt.where_clause {
            let bound_where;
            let where_clause: &Expression = match ctx.outer_row() {
                Some(outer) => {
                    let inner: Vec<&str> = subquery_source
                        .alias
                        .iter()
                        .map(|a| a.value_lower.as_str())
                        .collect();
                    let scope = super::utils::InnerScope {
                        tables: &inner,
                        schema: None,
                    };
                    bound_where = super::utils::substitute_outer_references_in_scope(
                        where_clause,
                        outer,
                        &scope,
                    );
                    &bound_where
                }
                None => where_clause,
            };
            if Self::has_correlated_subqueries(where_clause) {
                let alias = subquery_source
                    .alias
                    .as_ref()
                    .map(|a| a.value_lower.as_str());
                self.filter_rows_by_correlated_where(where_clause, rows, &columns, alias, ctx)?
            } else {
                let where_clause = self.answered_where(where_clause, ctx)?;
                let where_filter = RowFilter::new(&where_clause, &columns)?.with_context(ctx);

                let mut filtered = RowVec::with_capacity(rows.len());
                for (id, row) in rows {
                    if where_filter.matches_checked(&row)? {
                        filtered.push((id, row));
                    }
                }
                filtered
            }
        } else {
            rows
        };

        let subquery_alias = subquery_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str());

        if classification.has_aggregation || classification.has_window_functions {
            return self.grouped_select(
                stmt,
                ctx,
                classification,
                filtered_rows,
                &columns,
                subquery_alias,
            );
        }

        // Project rows according to SELECT expressions; a column ORDER BY
        // or DISTINCT ON reads that the select list leaves out rides along
        let mut output_columns =
            self.get_output_column_names(&stmt.columns, &columns, subquery_alias);
        let (projected_rows, layout, applied) = if self.order_by_needs_extra_columns(stmt, &columns)
        {
            let (rows, plan, extra_names, layout) = self.ordered_rows_plan(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                stmt.table_expr.as_deref(),
                filtered_rows,
                &CompactArc::new(columns.clone()),
                subquery_alias,
                ctx,
            )?;
            output_columns.extend(extra_names);
            let plan = OutputProjection::Ordered(plan);
            let (projected_rows, applied) =
                self.run_output_projection(stmt, &plan, rows, &output_columns, Some(&layout), ctx)?;
            (projected_rows, Some(layout), applied)
        } else if Self::has_correlated_select_subqueries(&stmt.columns) {
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                filtered_rows,
                &columns,
                None,
                ctx,
                subquery_alias,
            )?;
            let layout = Self::output_layout(stmt, &columns, subquery_alias);
            (projected_rows, layout, false)
        } else {
            let plan = OutputProjection::Select(self.select_rows_plan(
                &stmt.columns,
                &columns,
                None,
                ctx,
                subquery_alias,
            )?);
            let layout = Self::output_layout(stmt, &columns, subquery_alias);
            let (projected_rows, applied) = self.run_output_projection(
                stmt,
                &plan,
                filtered_rows,
                &output_columns,
                layout.as_ref(),
                ctx,
            )?;
            (projected_rows, layout, applied)
        };

        let output_columns = CompactArc::new(output_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((
            Box::new(result),
            output_columns,
            applied,
            SelectShape::laid_out(layout),
        ))
    }

    /// Parse a view's SQL query and return its cached plan.
    /// Uses the query cache to avoid re-parsing the same view query multiple times.
    /// The plan's statement is a SELECT - caller should extract it via pattern match.
    pub(super) fn parse_view_statement(&self, view_query: &str) -> Result<CachedPlanRef> {
        // Try to get from cache first (cheap Arc clone, no Statement clone)
        if let Some(cached) = self.query_cache.get(view_query) {
            // Validate it's a SELECT statement
            if !matches!(cached.statement.as_ref(), Statement::Select(_)) {
                return Err(Error::InvalidArgument(
                    "View definition is not a SELECT statement".to_string(),
                ));
            }
            if cached.params.count > 0 {
                return Err(view_parameters());
            }
            check_view_reads_now(&cached.statement)?;
            return Ok(cached);
        }

        // Parse the view's query
        let statements = crate::parser::parse_written_sql(view_query)
            .map_err(|e| Error::InvalidArgument(format!("Failed to parse view query: {}", e)))?;

        // The view query should be a single SELECT statement
        if statements.len() != 1 {
            return Err(Error::InvalidArgument(
                "View definition must contain exactly one statement".to_string(),
            ));
        }

        // Validate it's a SELECT statement
        if !matches!(&statements[0], Statement::Select(_)) {
            return Err(Error::InvalidArgument(
                "View definition is not a SELECT statement".to_string(),
            ));
        }

        if statements[0].param_count() > 0 {
            return Err(view_parameters());
        }
        check_view_reads_now(&statements[0])?;

        // Wrap in Arc and cache (single allocation, no clone)
        let stmt_arc = Arc::new(statements.into_iter().next().unwrap());

        // Cache the parsed statement for future use
        Ok(self
            .query_cache
            .put(view_query, stmt_arc, ParamSlots::default()))
    }

    /// Refuse a view read one view too deep, before its nesting overflows the stack
    pub(super) fn check_view_depth(ctx: &ExecutionContext) -> Result<()> {
        if ctx.view_depth() >= MAX_VIEW_DEPTH {
            return Err(Error::InvalidArgument(format!(
                "Maximum view nesting depth ({}) exceeded",
                MAX_VIEW_DEPTH
            )));
        }
        Ok(())
    }

    /// The rows of a view's own query, one view deeper than `ctx`
    pub(super) fn execute_view_body(
        &self,
        view_def: &ViewDefinition,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        Self::check_view_depth(ctx)?;

        // Parse the view's query using cache (an Arc clone of its plan, no clone)
        let view_plan = self.parse_view_statement(&view_def.query)?;
        let view_select = match view_plan.statement.as_ref() {
            Statement::Select(s) => s,
            _ => unreachable!("parse_view_statement validates this is a SELECT"),
        };

        // Execute the view's query with incremented depth, under its own plan
        let nested_ctx = ctx.with_incremented_view_depth();
        let result = self.execute_select_with_plan(
            view_select,
            &nested_ctx,
            Some(&view_plan.classification),
            Some(&view_plan.compiled),
        )?;
        let columns = match view_def.columns() {
            Some(columns) => columns,
            None => view_def.record_columns(result.columns().to_vec()),
        };
        if columns.len() != result.columns().len() {
            return Err(Error::InvalidArgument(format!(
                "view '{}' has {} columns but its query now gives {}: create the view again",
                view_def.original_name,
                columns.len(),
                result.columns().len()
            )));
        }
        if columns == result.columns() {
            return Ok(result);
        }
        Ok(Box::new(AliasedResult::renamed(result, columns.to_vec())))
    }

    /// Execute a view as a subquery
    fn execute_view_query(
        &self,
        view_def: &ViewDefinition,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // classification is passed from caller for the outer stmt
        // Note: The view's inner query will get its own classification via execute_select
        let result = self.execute_view_body(view_def, ctx)?;
        let view_columns = result.columns().to_vec();

        // Apply outer query's WHERE clause if present
        // OPTIMIZATION: FilteredResult owns a pre-compiled RowFilter and reuses it for each row,
        // avoiding repeated expression compilation per row.
        // CRITICAL: Must pass ctx for parameter resolution ($1, named params, etc.)
        let mut result: Box<dyn QueryResult> = result;
        let view_alias = stmt
            .table_expr
            .as_deref()
            .and_then(super::utils::get_table_alias_from_expr)
            .map(|name| name.to_lowercase());
        if let Some(ref where_clause) = stmt.where_clause {
            // A parent row's column named through its own table is bound
            // first, or the bare name it would fall back to could be the
            // view's own
            let bound_where;
            let where_clause: &Expression = match ctx.outer_row() {
                Some(outer) => {
                    let inner: Vec<String> = stmt
                        .table_expr
                        .as_deref()
                        .and_then(super::utils::get_table_alias_from_expr)
                        .map(|name| name.to_lowercase())
                        .into_iter()
                        .collect();
                    let inner: Vec<&str> = inner.iter().map(String::as_str).collect();
                    let scope = super::utils::InnerScope {
                        tables: &inner,
                        schema: None,
                    };
                    bound_where = super::utils::substitute_outer_references_in_scope(
                        where_clause,
                        outer,
                        &scope,
                    );
                    &bound_where
                }
                None => where_clause,
            };
            if Self::has_correlated_subqueries(where_clause) {
                let rows = Self::materialize_result(result)?;
                let rows = self.filter_rows_by_correlated_where(
                    where_clause,
                    rows,
                    &view_columns,
                    view_alias.as_deref(),
                    ctx,
                )?;
                result = Box::new(ExecutorResult::new(view_columns.clone(), rows));
            } else {
                let where_clause = self.answered_where(where_clause, ctx)?;
                let filter = RowFilter::new(&where_clause, &view_columns)?.with_context(ctx);
                result = Box::new(FilteredResult::from_filter(result, filter));
            }
        }

        // A correlated subquery in the select list reads each view row as
        // its outer row, the way it does over a derived table
        if Self::has_correlated_select_subqueries(&stmt.columns)
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !self.order_by_needs_extra_columns(stmt, &view_columns)
        {
            let rows = Self::materialize_result(result)?;
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                &view_columns,
                None,
                ctx,
                view_alias.as_deref(),
            )?;
            let output_columns = CompactArc::new(self.get_output_column_names(
                &stmt.columns,
                &view_columns,
                view_alias.as_deref(),
            ));
            let result = ExecutorResult::with_arc_columns(
                CompactArc::clone(&output_columns),
                projected_rows,
            );
            let layout = Self::output_layout(stmt, &view_columns, None);
            return Ok((
                Box::new(result),
                output_columns,
                false,
                SelectShape::laid_out(layout),
            ));
        }

        // Aggregation and window functions run over the view's rows, materialized
        if classification.has_aggregation || classification.has_window_functions {
            let mut rows = RowVec::with_capacity(64);
            let mut idx = 0i64;
            while result.next() {
                rows.push((idx, result.take_row()));
                idx += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }
            return self.grouped_select(
                stmt,
                ctx,
                classification,
                rows,
                &view_columns,
                view_alias.as_deref(),
            );
        }

        // Handle projection: apply outer query's column selection
        // Check if SELECT * or t.* - if so, return all view columns
        let is_select_star = stmt.columns.len() == 1
            && star_takes_every_column(&stmt.columns[0], &view_columns, view_alias.as_deref());

        if is_select_star {
            // For SELECT *, just return the view result with WHERE applied
            // DISTINCT, ORDER BY, LIMIT/OFFSET are handled by execute_select
            let layout = Self::output_layout(stmt, &view_columns, view_alias.as_deref());
            return Ok((
                result,
                CompactArc::new(view_columns),
                false,
                SelectShape::laid_out(layout),
            ));
        }

        // A column ORDER BY or DISTINCT ON reads that the select list leaves
        // out rides along, the way it does for a derived table
        if self.order_by_needs_extra_columns(stmt, &view_columns) {
            let mut result = result;
            let mut rows = RowVec::new();
            let mut row_id = 0i64;
            while result.next() {
                rows.push((row_id, result.take_row()));
                row_id += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }
            let (rows, plan, extra_names, layout) = self.ordered_rows_plan(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                stmt.table_expr.as_deref(),
                rows,
                &CompactArc::new(view_columns.clone()),
                view_alias.as_deref(),
                ctx,
            )?;
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &view_columns, None);
            output_columns.extend(extra_names);
            let plan = OutputProjection::Ordered(plan);
            let (projected_rows, applied) =
                self.run_output_projection(stmt, &plan, rows, &output_columns, Some(&layout), ctx)?;
            let output_columns = CompactArc::new(output_columns);
            let result = ExecutorResult::with_arc_columns(
                CompactArc::clone(&output_columns),
                projected_rows,
            );
            return Ok((
                Box::new(result),
                output_columns,
                applied,
                SelectShape::laid_out(Some(layout)),
            ));
        }

        // Determine if we have any complex expressions (not just column references)
        // Column references are checked against the view here; any other item
        // compiles into the projection
        let mut has_complex_expressions = false;
        let mut column_indices = Vec::with_capacity(stmt.columns.len());
        let mut output_columns = Vec::with_capacity(stmt.columns.len());
        let view_alias = view_alias.as_deref();

        // First pass: check if all are simple column references and build indices
        for col in &stmt.columns {
            match col {
                Expression::Star(_) => {
                    // Expand all columns
                    for (idx, name) in view_columns.iter().enumerate() {
                        column_indices.push(idx);
                        output_columns.push(name.clone());
                    }
                }
                Expression::QualifiedStar(qs) => {
                    for_each_qualified_star_column(
                        &view_columns,
                        &qs.qualifier,
                        view_alias,
                        |idx, start| {
                            column_indices.push(idx);
                            output_columns.push(view_columns[idx][start..].to_string());
                        },
                    );
                }
                Expression::Identifier(id) => {
                    // Simple column reference
                    output_columns.push(id.value.to_string());
                    let name_lower = &id.value_lower;
                    if let Some(idx) = view_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(name_lower))
                    {
                        column_indices.push(idx);
                    } else if id.quoted {
                        // A quoted name no column has is its text, which
                        // the compiled projection reads as such
                        has_complex_expressions = true;
                    } else {
                        return Err(Error::ColumnNotFound(id.value.to_string()));
                    }
                }
                Expression::Aliased(aliased) => {
                    output_columns.push(aliased.alias.value.to_string());
                    // Check if inner is a simple identifier
                    if let Expression::Identifier(id) = aliased.expression.as_ref() {
                        let name_lower = &id.value_lower;
                        if let Some(idx) = view_columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(name_lower))
                        {
                            column_indices.push(idx);
                        } else if id.quoted {
                            has_complex_expressions = true;
                        } else {
                            return Err(Error::ColumnNotFound(id.value.to_string()));
                        }
                    } else {
                        // Complex expression in alias - compiled into the projection
                        has_complex_expressions = true;
                    }
                }
                _ => {
                    // Any other expression type requires evaluation
                    has_complex_expressions = true;
                    // We'll compute the output column name later
                    output_columns.push(item_name(col));
                }
            }
        }

        // Apply projection based on complexity
        let result: Box<dyn QueryResult> = if has_complex_expressions {
            // Rebuild output columns to handle Star expansion
            let mut final_output_columns = Vec::with_capacity(stmt.columns.len());
            for col in &stmt.columns {
                match col {
                    Expression::Star(_) => {
                        for name in &view_columns {
                            final_output_columns.push(name.clone());
                        }
                    }
                    Expression::QualifiedStar(qs) => {
                        for_each_qualified_star_column(
                            &view_columns,
                            &qs.qualifier,
                            view_alias,
                            |idx, start| {
                                final_output_columns.push(view_columns[idx][start..].to_string())
                            },
                        );
                    }
                    Expression::Aliased(aliased) => {
                        final_output_columns.push(aliased.alias.value.to_string());
                    }
                    _ => {
                        final_output_columns.push(item_name(col));
                    }
                }
            }

            let projection = Projection::compile(&stmt.columns, result.columns(), view_alias)?;
            Box::new(
                ProjectionResult::new(result, projection, final_output_columns.clone())
                    .with_context(ctx),
            )
        } else {
            Box::new(ProjectionResult::new(
                result,
                Projection::columns(column_indices),
                output_columns.clone(),
            ))
        };

        // Rebuild output_columns if we used complex expressions
        let final_columns = if has_complex_expressions {
            let mut cols = Vec::with_capacity(stmt.columns.len());
            for col in &stmt.columns {
                match col {
                    Expression::Star(_) => {
                        for name in &view_columns {
                            cols.push(name.clone());
                        }
                    }
                    Expression::QualifiedStar(qs) => {
                        for_each_qualified_star_column(
                            &view_columns,
                            &qs.qualifier,
                            view_alias,
                            |idx, start| cols.push(view_columns[idx][start..].to_string()),
                        );
                    }
                    Expression::Aliased(aliased) => {
                        cols.push(aliased.alias.value.to_string());
                    }
                    _ => {
                        cols.push(item_name(col));
                    }
                }
            }
            cols
        } else {
            output_columns
        };

        let layout = Self::output_layout(stmt, &view_columns, view_alias);
        Ok((
            result,
            CompactArc::new(final_columns),
            false,
            SelectShape::laid_out(layout),
        ))
    }

    /// The column names of a VALUES source
    pub(super) fn values_columns(values_source: &ValuesTableSource) -> Vec<String> {
        // Determine column names
        let num_columns = if values_source.rows.is_empty() {
            0
        } else {
            values_source.rows[0].len()
        };

        if !values_source.column_aliases.is_empty() {
            // Use provided column aliases
            values_source
                .column_aliases
                .iter()
                .map(|id| id.value.to_string())
                .collect()
        } else {
            // Generate default column names: column1, column2, ...
            (1..=num_columns).map(|i| format!("column{}", i)).collect()
        }
    }

    /// One row of a VALUES source, each value evaluated
    pub(super) fn values_row(row_exprs: &[Expression], ctx: &ExecutionContext) -> Result<Row> {
        let mut row_values = Vec::with_capacity(row_exprs.len());
        for expr in row_exprs {
            let value = ExpressionEval::compile(expr, &[])?
                .with_context(ctx)
                .eval_slice(&Row::new())?;
            row_values.push(value);
        }
        Ok(Row::from_values(row_values))
    }

    /// Execute a VALUES source (e.g., (VALUES (1, 'a'), (2, 'b')) AS t(col1, col2))
    fn execute_values_source(
        &self,
        values_source: &ValuesTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

        // Get the table alias for qualified name resolution
        let table_alias = values_source
            .alias
            .as_ref()
            .map(|a| a.value.to_string())
            .unwrap_or_default();

        let column_names = Self::values_columns(values_source);

        // Build a map of both simple and qualified column names to indices
        let mut col_index_map = build_column_index_map(&column_names);

        // Also add qualified names (e.g., "v.id" for table alias "v")
        // OPTIMIZATION: Pre-compute lowercase table alias once outside the loop
        if !table_alias.is_empty() {
            let alias_lower = table_alias.to_lowercase();
            for (i, name) in column_names.iter().enumerate() {
                col_index_map.insert(qualified(&alias_lower, &name.to_lowercase()), i);
            }
        }

        // OPTIMIZATION: Pre-create RowFilters outside the loop if WHERE clause exists
        let (where_filter, qualified_filter) = if let Some(ref where_clause) = stmt.where_clause {
            // Pre-compute qualified column names once
            let qualified_cols: Vec<String> = column_names
                .iter()
                .map(|c| qualified(&table_alias, c))
                .collect();

            // Create filters for simple and qualified column names
            let filter = RowFilter::new(where_clause, &column_names)?.with_context(ctx);
            let qual_filter = RowFilter::new(where_clause, &qualified_cols)?.with_context(ctx);

            (Some(filter), Some(qual_filter))
        } else {
            (None, None)
        };

        // Evaluate all rows
        let mut result_rows = RowVec::with_capacity(values_source.rows.len());
        let mut row_id = 0i64;
        for row_exprs in &values_source.rows {
            let row = Self::values_row(row_exprs, ctx)?;

            // Apply WHERE clause filtering
            if let (Some(wf), Some(qf)) = (&where_filter, &qualified_filter) {
                // Try with simple column names first, then qualified
                if wf.matches_checked(&row)? || qf.matches_checked(&row)? {
                    result_rows.push((row_id, row));
                    row_id += 1;
                }
            } else {
                result_rows.push((row_id, row));
                row_id += 1;
            }
        }

        let values_alias = values_source.alias.as_ref().map(|a| a.value_lower.as_str());

        if classification.has_aggregation || classification.has_window_functions {
            return self.grouped_select(
                stmt,
                ctx,
                classification,
                result_rows,
                &column_names,
                values_alias,
            );
        }

        let layout = Self::output_layout(stmt, &column_names, values_alias);

        // Create the result - use CompactArc for column names
        let column_names_arc = CompactArc::new(column_names);
        let values_result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&column_names_arc), result_rows);

        // If the SELECT has projections, apply them
        if stmt.columns.len() == 1
            && star_takes_every_column(&stmt.columns[0], &column_names_arc, values_alias)
        {
            // SELECT * or t.* - return all columns
            return Ok((
                Box::new(values_result),
                column_names_arc,
                false,
                SelectShape::laid_out(layout),
            ));
        }

        // Apply column projection
        let mut projected_columns = Vec::new();
        let mut projected_rows = RowVec::new();
        let mut proj_row_id = 0i64;

        // Determine output columns
        for col_expr in &stmt.columns {
            match col_expr {
                Expression::Star(_) => {
                    projected_columns.extend(column_names_arc.iter().cloned());
                }
                Expression::QualifiedStar(qs) => {
                    for_each_qualified_star_column(
                        &column_names_arc,
                        &qs.qualifier,
                        values_alias,
                        |idx, start| {
                            projected_columns.push(column_names_arc[idx][start..].to_string())
                        },
                    );
                }
                other => projected_columns.push(item_name(other)),
            }
        }

        // Build extended columns for evaluator - just use simple column names
        // The evaluator handles qualified references by stripping the qualifier
        let extended_columns = column_names_arc.to_vec();

        // OPTIMIZATION: Check if we need an evaluator for complex expressions
        let needs_evaluator = stmt.columns.iter().any(|c| {
            matches!(c, Expression::Aliased(_))
                || !matches!(
                    c,
                    Expression::Star(_)
                        | Expression::QualifiedStar(_)
                        | Expression::Identifier(_)
                        | Expression::QualifiedIdentifier(_)
                )
        });

        let mut proj_eval = if needs_evaluator {
            let mut eval = CompiledEvaluator::new(&self.function_registry);
            eval = eval.with_context(ctx);
            eval.init_columns(&extended_columns);
            Some(eval)
        } else {
            None
        };

        // OPTIMIZATION: Pre-compute qualified column lookups to avoid format! per row
        let mut qualified_col_indices: Vec<Option<usize>> = Vec::new();
        for col_expr in &stmt.columns {
            if let Expression::QualifiedIdentifier(qi) = col_expr {
                qualified_col_indices.push(qualified_index(
                    &col_index_map,
                    &qi.qualifier.value_lower,
                    &qi.name.value_lower,
                ));
            } else {
                qualified_col_indices.push(None);
            }
        }
        // A name no column has is compiled once, read as the compiler reads
        // it: the enclosing row's column, a quoted name's text, or an error
        let outer_columns = outer_column_names(ctx);
        let mut unbound_ctx = CompileContext::new(&extended_columns, &self.function_registry);
        if let Some(outer) = &outer_columns {
            unbound_ctx = unbound_ctx.with_outer_columns(outer);
        }
        let unbound: Vec<Option<SharedProgram>> = stmt
            .columns
            .iter()
            .zip(&qualified_col_indices)
            .map(|(col_expr, qualified_idx)| {
                let bound = match col_expr {
                    Expression::Identifier(id) => {
                        col_index_map.contains_key(id.value_lower.as_str())
                    }
                    Expression::QualifiedIdentifier(_) => qualified_idx.is_some(),
                    _ => return Ok(None),
                };
                if bound {
                    return Ok(None);
                }
                ExprCompiler::new(&unbound_ctx)
                    .compile(col_expr)
                    .map(|program| Some(CompactArc::new(program)))
                    .map_err(|e| Error::internal(format!("Compile error: {}", e)))
            })
            .collect::<Result<_>>()?;
        let mut unbound_vm = ExprVM::new();

        // Materialize and project
        // OPTIMIZATION: Pre-compute output size to avoid reallocation per row
        let output_cols = projected_columns.len();
        // The columns each qualified star takes, found once for every row
        let star_columns: Vec<Vec<usize>> = stmt
            .columns
            .iter()
            .map(|col_expr| {
                let mut taken = Vec::new();
                if let Expression::QualifiedStar(qs) = col_expr {
                    for_each_qualified_star_column(
                        &column_names_arc,
                        &qs.qualifier,
                        values_alias,
                        |idx, _| taken.push(idx),
                    );
                }
                taken
            })
            .collect();
        let mut result_box: Box<dyn QueryResult> = Box::new(values_result);
        while result_box.next() {
            let row = result_box.row();
            let mut new_values = Vec::with_capacity(output_cols);

            for (col_idx, col_expr) in stmt.columns.iter().enumerate() {
                if let Some(program) = &unbound[col_idx] {
                    let mut exec =
                        ExecuteContext::with_common_params(row, ctx.params(), ctx.transaction_id());
                    if let Some(outer) = ctx.outer_row() {
                        exec = exec.with_outer_row(outer);
                    }
                    new_values.push(unbound_vm.execute(program, &exec)?);
                    continue;
                }
                match col_expr {
                    Expression::Star(_) => {
                        // Extend with all values from the row
                        new_values.extend(row.iter().cloned());
                    }
                    Expression::QualifiedStar(_) => {
                        new_values.extend(
                            star_columns[col_idx].iter().map(|&idx| {
                                row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
                            }),
                        );
                    }
                    Expression::Identifier(id) => {
                        // Use pre-computed lowercase
                        if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                            if let Some(val) = row.get(idx) {
                                new_values.push(val.clone());
                            } else {
                                new_values.push(Value::null_unknown());
                            }
                        } else {
                            new_values.push(Value::null_unknown());
                        }
                    }
                    Expression::QualifiedIdentifier(_) => {
                        // Use pre-computed index
                        if let Some(idx) = qualified_col_indices[col_idx] {
                            if let Some(val) = row.get(idx) {
                                new_values.push(val.clone());
                            } else {
                                new_values.push(Value::null_unknown());
                            }
                        } else {
                            new_values.push(Value::null_unknown());
                        }
                    }
                    Expression::Aliased(a) => {
                        // Evaluate the underlying expression
                        let eval = proj_eval.as_mut().unwrap();
                        eval.set_row_array(row);

                        // Check if expression contains EXISTS subqueries
                        let expr_to_eval = if Self::has_subqueries(&a.expression) {
                            // Check if it's correlated (references outer columns)
                            if Self::has_correlated_subqueries(&a.expression) {
                                // Build outer row context for correlated subquery
                                let mut outer_row_map: FxHashMap<CompactArc<str>, Value> =
                                    FxHashMap::default();
                                for (i, col_name) in extended_columns.iter().enumerate() {
                                    if let Some(value) = row.get(i) {
                                        outer_row_map.insert(
                                            CompactArc::from(col_name.to_lowercase().as_str()),
                                            value.clone(),
                                        );
                                    }
                                }
                                let correlated_ctx = ctx.with_outer_row(
                                    outer_row_map,
                                    CompactArc::new(extended_columns.clone()),
                                );
                                self.process_correlated_where(&a.expression, &correlated_ctx)?
                            } else {
                                self.process_where_subqueries(&a.expression, ctx)?
                            }
                        } else {
                            (*a.expression).clone()
                        };

                        let val = eval.evaluate(&expr_to_eval)?;
                        new_values.push(val);
                    }
                    other => {
                        // Evaluate the expression
                        let eval = proj_eval.as_mut().unwrap();
                        eval.set_row_array(row);

                        // Check if expression contains EXISTS subqueries
                        let expr_to_eval = if Self::has_subqueries(other) {
                            // Check if it's correlated (references outer columns)
                            if Self::has_correlated_subqueries(other) {
                                // Build outer row context for correlated subquery
                                let mut outer_row_map: FxHashMap<CompactArc<str>, Value> =
                                    FxHashMap::default();
                                for (i, col_name) in extended_columns.iter().enumerate() {
                                    if let Some(value) = row.get(i) {
                                        outer_row_map.insert(
                                            CompactArc::from(col_name.to_lowercase().as_str()),
                                            value.clone(),
                                        );
                                    }
                                }
                                let correlated_ctx = ctx.with_outer_row(
                                    outer_row_map,
                                    CompactArc::new(extended_columns.clone()),
                                );
                                self.process_correlated_where(other, &correlated_ctx)?
                            } else {
                                self.process_where_subqueries(other, ctx)?
                            }
                        } else {
                            other.clone()
                        };

                        let val = eval.evaluate(&expr_to_eval)?;
                        new_values.push(val);
                    }
                }
            }

            projected_rows.push((proj_row_id, Row::from_values(new_values)));
            proj_row_id += 1;
        }

        let projected_columns_arc = CompactArc::new(projected_columns);
        let final_result = ExecutorResult::with_arc_columns(
            CompactArc::clone(&projected_columns_arc),
            projected_rows,
        );
        Ok((
            Box::new(final_result),
            projected_columns_arc,
            false,
            SelectShape::laid_out(layout),
        ))
    }

    /// The statement a joined side runs as: a SELECT * of its join, with
    /// `filter` as its WHERE
    fn joined_side_statement(
        join: &JoinTableSource,
        filter: Option<&Expression>,
    ) -> SelectStatement {
        SelectStatement {
            param_count: 0,
            token: dummy_token("SELECT", TokenType::Keyword),
            distinct: false,
            distinct_on: vec![],
            columns: vec![Expression::Star(StarExpression {
                token: dummy_token("*", TokenType::Punctuator),
            })],
            with: None,
            table_expr: Some(Box::new(Expression::JoinSource(Box::new(join.clone())))),
            where_clause: filter.cloned().map(Box::new),
            group_by: GroupByClause::default(),
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
            prepared: None,
        }
    }

    /// The rows of `join`, a side that is itself a join, each table's columns
    /// in turn: a SELECT * of it where `filter` holds, held in `compiled`
    /// when it has a plan. Read `apart`, a NATURAL or USING join in it keeps
    /// both columns of a pair
    fn joined_side_rows(
        &self,
        join: &JoinTableSource,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
        apart: bool,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        let select_all = Self::joined_side_statement(join, filter);
        let classification = get_classification(&select_all);
        let (result, columns, _, _) =
            self.execute_join_source(join, &select_all, ctx, &classification, compiled, apart)?;
        Ok((result, columns.to_vec()))
    }

    /// Execute a table expression with optional filter pushdown
    /// This is used by JOIN to push WHERE predicates to individual table scans
    pub(crate) fn execute_table_expression_with_filter(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        read_out: Option<&mut Option<ReadSchema>>,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        match expr {
            Expression::TableSource(ts) => {
                // Check if this is a CTE from context (for subqueries referencing outer CTEs)
                let table_name = &ts.name.value_lower;
                if let Some((columns, rows)) = ctx.get_cte_by_lower(table_name) {
                    if ts.as_of.is_some() {
                        return Err(Self::as_of_not_a_table(table_name));
                    }
                    // Get the alias for column prefixing
                    let table_alias = ts
                        .alias
                        .as_ref()
                        .map(|a| a.value.clone())
                        .unwrap_or_else(|| ts.name.value.clone());
                    let qualified_columns: Vec<String> = columns
                        .iter()
                        .map(|col| qualified(&table_alias, col))
                        .collect();

                    // Apply filter to CTE data if present, or convert to RowVec
                    let result: Box<dyn QueryResult> = if let Some(filter_expr) = filter {
                        // Need to clone matching rows
                        // Must apply execution context for parameter support
                        // Use qualified_columns for filter column resolution since WHERE clause
                        // has qualified names like "ds.avg_salary" not just "avg_salary"
                        let row_filter =
                            RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
                        // CTE stores CompactArc<Vec<(i64, Row)>>, filter on the Row part
                        let mut filtered_rows = RowVec::with_capacity(rows.len());
                        for (id, row) in rows.iter() {
                            if row_filter.matches_checked(row)? {
                                filtered_rows.push((*id, row.clone()));
                            }
                        }
                        Box::new(super::result::ExecutorResult::new(
                            columns.to_vec(),
                            filtered_rows,
                        ))
                    } else {
                        // Convert CTE CompactArc<Vec<(i64, Row)>> to RowVec
                        let rows_vec = RowVec::from_vec((**rows).clone());
                        Box::new(super::result::ExecutorResult::new(
                            columns.to_vec(),
                            rows_vec,
                        ))
                    };

                    return Ok((result, qualified_columns));
                }

                // Check if this is actually a view (for JOINs that reference views)
                if let Some(view_def) = self.engine.get_view_lowercase(table_name)? {
                    if ts.as_of.is_some() {
                        return Err(Self::as_of_not_a_table(table_name));
                    }
                    let result = self.execute_view_body(&view_def, ctx)?;
                    let columns = result.columns().to_vec();

                    // Prefix column names with view alias (or view name if no alias)
                    let table_alias = ts
                        .alias
                        .as_ref()
                        .map(|a| a.value.clone())
                        .unwrap_or_else(|| ts.name.value.clone());
                    let qualified_columns: Vec<String> = columns
                        .iter()
                        .map(|col| qualified(&table_alias, col))
                        .collect();
                    // The filter pushed to this side applies to the view's
                    // rows the way it does to a CTE's
                    let result: Box<dyn QueryResult> = match filter {
                        Some(filter_expr) => {
                            let row_filter =
                                RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
                            Box::new(FilteredResult::from_filter(result, row_filter))
                        }
                        None => result,
                    };
                    return Ok((result, qualified_columns));
                }

                // Create a SELECT * statement with optional WHERE clause
                let select_all = SelectStatement {
                    param_count: 0,
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
                    distinct_on: vec![],
                    columns: vec![Expression::Star(StarExpression {
                        token: dummy_token("*", TokenType::Punctuator),
                    })],
                    with: None,
                    table_expr: Some(Box::new(Expression::TableSource(ts.clone()))),
                    where_clause: filter.map(|f| Box::new(f.clone())),
                    group_by: GroupByClause::default(),
                    having: None,
                    window_defs: vec![],
                    order_by: vec![],
                    limit: None,
                    offset: None,
                    set_operations: vec![],
                    prepared: None,
                };
                // Get classification for the synthetic SELECT statement
                let classification = get_classification(&select_all);
                let (result, columns, _, _) = self.execute_simple_table_scan(
                    ts,
                    &select_all,
                    ctx,
                    &classification,
                    read_out,
                    None,
                )?;

                // Prefix column names with table alias (or table name if no alias)
                // This is needed for proper qualified identifier resolution in JOINs
                let table_alias = ts
                    .alias
                    .as_ref()
                    .map(|a| a.value.clone())
                    .unwrap_or_else(|| ts.name.value.clone());

                let qualified_columns: Vec<String> = columns
                    .iter()
                    .map(|col| qualified(&table_alias, col))
                    .collect();

                Ok((result, qualified_columns))
            }
            Expression::JoinSource(js) => self.joined_side_rows(js, ctx, filter, None, false),
            Expression::SubquerySource(ss) => self.derived_side_rows(ss, ctx, filter, None),
            Expression::ValuesSource(vs) => {
                // A SELECT * of the VALUES, with the filter pushed to this side
                let select_all = SelectStatement {
                    param_count: 0,
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
                    distinct_on: vec![],
                    columns: vec![Expression::Star(StarExpression {
                        token: dummy_token("*", TokenType::Punctuator),
                    })],
                    with: None,
                    table_expr: Some(Box::new(Expression::ValuesSource(vs.clone()))),
                    where_clause: filter.map(|f| Box::new(f.clone())),
                    group_by: GroupByClause::default(),
                    having: None,
                    window_defs: vec![],
                    order_by: vec![],
                    limit: None,
                    offset: None,
                    set_operations: vec![],
                    prepared: None,
                };
                // Get classification for the synthetic SELECT statement
                let classification = get_classification(&select_all);
                let (result, columns, _, _) =
                    self.execute_values_source(vs, &select_all, ctx, &classification)?;
                // Its columns by its alias, as a table's are
                let columns = match &vs.alias {
                    Some(alias) => columns.iter().map(|c| qualified(&alias.value, c)).collect(),
                    None => columns.to_vec(),
                };
                Ok((result, columns))
            }
            Expression::FunctionTableSource(tvf_source) => self
                .execute_tvf_for_join(tvf_source, ctx, filter, None, 0)
                .map(|(result, columns, _)| (result, columns)),
            _ => Err(Error::NotSupported(
                "Unsupported table expression type".to_string(),
            )),
        }
    }

    /// The `window` of a GROUP BY + LIMIT join's left side `expr`. Unless
    /// read `consistent`, its boundary row is read again first, and a commit
    /// that moved it gives None
    fn left_chunk(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        side: SideRead<'_>,
        window: &LeftWindow,
        consistent: bool,
    ) -> Result<Option<OuterFetch>> {
        if consistent || window.offset == 0 {
            let read = Some(window.cap);
            return self
                .read_join_side(expr, ctx, side, None, read, window.offset)
                .map(Some);
        }
        let Some(boundary) = &window.boundary else {
            return Ok(None);
        };
        let read = Some(window.cap.saturating_add(1));
        let (mut more, columns, bounded, token) =
            self.read_join_side(expr, ctx, side, None, read, window.offset - 1)?;
        let unmoved = more.next() && more.row().as_slice() == boundary.as_slice();
        Ok(unmoved.then_some((more, columns, bounded, token)))
    }

    /// Execute a table expression with optional filter, row limit and row offset.
    /// The flag says whether the limit and offset were applied: only a plain
    /// table takes them; a CTE, a view, a subquery or a table function comes
    /// back whole and cannot be fetched again from an offset. A plain table
    /// also gives the layout its rows were read in.
    /// A join side's rows with the `extra` conditions the join adds: read
    /// through the statement's blocks when they hold the side's conditions,
    /// else with its filter, from the `offset`-th up to `limit` rows
    fn read_join_side(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        side: SideRead<'_>,
        extra: Option<&Expression>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<OuterFetch> {
        // A side the blocks read as rows is read by the one read below, as the
        // general path reads it, with its conditions without a block; the
        // ones with a block that it keeps are asked of the rows as they come,
        // which the read's limit cannot count
        let rows = side
            .run
            .filter(|(run, index)| run.blocks.rows_sides[*index]);
        let side = match rows {
            Some(_) => SideRead { run: None, ..side },
            None => side,
        };
        let (limit, offset) = match rows.filter(|(run, index)| run.blocks.side_asks(*index)) {
            Some(_) => (None, 0),
            None => (limit, offset),
        };
        if let (Some((run, index)), Expression::TableSource(ts)) = (side.run, expr) {
            let alias = ts.alias.as_ref().unwrap_or(&ts.name).value.as_str();
            let table = (ts.name.value_lower.as_str(), alias);
            let read =
                self.join_side_rows(run.blocks, run.run, index, table, ctx, extra, limit, offset)?;
            if let Some((result, token)) = read {
                let columns = result
                    .columns()
                    .iter()
                    .map(|c| qualified(alias, c))
                    .collect();
                return Ok((result, columns, limit.is_some(), Some(token)));
            }
        }
        let both;
        let filter = match (side.filter, extra) {
            (Some(filter), Some(extra)) => {
                both = combine_predicates_with_and(vec![filter.clone(), extra.clone()]);
                both.as_ref()
            }
            (filter, extra) => filter.or(extra),
        };
        #[cfg(feature = "test-failpoints")]
        if side.run.is_some() {
            crate::test_failpoints::join_side_opening();
        }
        let mut read = None;
        let read_out = side.run.is_some().then_some(&mut read);
        // A joined side runs as a join of its own, under the plan it holds,
        // both columns of a pair it merges kept as the blocks read them
        let nested = match (rows, expr) {
            (Some(_), Expression::JoinSource(join)) => Some((join, true)),
            (None, Expression::JoinSource(join)) => Some((join, side.apart)),
            _ => None,
        };
        // A derived table runs under the plan it holds
        let derived = match expr {
            Expression::SubquerySource(ss) => Some((ss, side.plan)),
            _ => None,
        };
        let fetched = match (nested, derived) {
            (Some((join, apart)), _) => {
                let compiled = side.plan.map(NestedPlan::slot);
                let (result, columns) =
                    self.joined_side_rows(join, ctx, side.filter, compiled, apart)?;
                (result, columns, false, None)
            }
            (None, Some((ss, plan))) => {
                let (result, columns) = self.derived_side_rows(ss, ctx, filter, plan)?;
                (result, columns, false, None)
            }
            (None, None) => self.execute_table_expression_with_filter_limit(
                expr, ctx, filter, limit, offset, read_out,
            )?,
        };
        if let Some((run, index)) = rows {
            let (result, columns, bounded, _) = fetched;
            let result =
                self.join_source_side_rows(run.blocks, run.run, index, (expr, result), ctx)?;
            return Ok((result, columns, bounded, None));
        }
        if let Some((run, index)) = side.run {
            let (_, schema) =
                read.ok_or_else(|| Error::internal("a join side read without its schema"))?;
            run.blocks.side_fits(index, &schema)?;
        }
        Ok(fetched)
    }

    /// A derived table's rows and its columns qualified by its alias, those
    /// `filter` passes, its statement run under `plan` when it holds one
    fn derived_side_rows(
        &self,
        ss: &SubqueryTableSource,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        plan: Option<&NestedPlan>,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        // Execute subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let result = self.execute_select_with_plan(&ss.subquery, &subquery_ctx, None, plan)?;
        let columns = result.columns().to_vec();

        // Prefix column names with subquery alias (required for proper ON condition resolution)
        // Without this, ON a.id = b.id cannot resolve qualified column names
        let qualified_columns: Vec<String> = if let Some(alias) = &ss.alias {
            columns
                .iter()
                .map(|col| qualified(&alias.value, col))
                .collect()
        } else {
            columns.clone()
        };

        // Apply filter to subquery result if present
        // This is needed when WHERE clause conditions are pushed down to subquery sources
        if let Some(filter_expr) = filter {
            // Use qualified_columns for filter column resolution since WHERE clause
            // has qualified names like "ds.avg_salary" not just "avg_salary"
            let row_filter = RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
            // Materialize and filter - materialized is RowVec = Vec<(i64, Row)>
            let mut materialized = Self::materialize_result(result)?;
            row_filter.retain_checked(&mut materialized)?;
            let filtered_result: Box<dyn QueryResult> =
                Box::new(super::result::ExecutorResult::new(columns, materialized));
            return Ok((filtered_result, qualified_columns));
        }

        Ok((result, qualified_columns))
    }

    /// `read_out` takes the layout and schema a plain table's rows were read
    /// with
    pub(crate) fn execute_table_expression_with_filter_limit(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        row_limit: Option<usize>,
        row_offset: usize,
        read_out: Option<&mut Option<ReadSchema>>,
    ) -> Result<OuterFetch> {
        // Handle FunctionTableSource with limit passthrough
        let tvf_source = match expr {
            Expression::FunctionTableSource(fts) => Some(fts.as_ref()),
            Expression::Aliased(aliased) => {
                if let Expression::FunctionTableSource(fts) = aliased.expression.as_ref() {
                    Some(fts.as_ref())
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(tvf_source) = tvf_source {
            return self
                .execute_tvf_for_join(tvf_source, ctx, filter, row_limit, row_offset)
                .map(|(result, columns, bounded)| (result, columns, bounded, None));
        }

        // Extract TableSource from the expression (handles both direct and aliased)
        let (ts, custom_alias) = match expr {
            Expression::TableSource(ts) => (ts, None),
            Expression::Aliased(aliased) => {
                if let Expression::TableSource(ts) = aliased.expression.as_ref() {
                    (ts, Some(aliased.alias.value.clone()))
                } else {
                    // Not a table source, fall back to standard execution
                    return self
                        .execute_table_expression_with_filter(expr, ctx, filter, None)
                        .map(|(result, columns)| (result, columns, false, None));
                }
            }
            _ => {
                // Not a table source, fall back to standard execution
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter, None)
                    .map(|(result, columns)| (result, columns, false, None));
            }
        };

        // Only optimize if we have a limit
        if let Some(limit) = row_limit {
            let table_name = ts.name.value_lower.as_str();

            // Skip if this is a CTE reference - CTEs are already materialized
            if ctx.get_cte_by_lower(table_name).is_some() {
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter, None)
                    .map(|(result, columns)| (result, columns, false, None));
            }

            // Skip if this is a view
            if self.engine.get_view_lowercase(table_name)?.is_some() {
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter, None)
                    .map(|(result, columns)| (result, columns, false, None));
            }

            // Create a SELECT * statement with WHERE clause AND LIMIT
            // This triggers the LIMIT pushdown optimization in execute_simple_table_scan
            let select_all = SelectStatement {
                param_count: 0,
                token: dummy_token("SELECT", TokenType::Keyword),
                distinct: false,
                distinct_on: vec![],
                columns: vec![Expression::Star(StarExpression {
                    token: dummy_token("*", TokenType::Punctuator),
                })],
                with: None,
                table_expr: Some(Box::new(Expression::TableSource(ts.clone()))),
                where_clause: filter.map(|f| Box::new(f.clone())),
                group_by: GroupByClause::default(),
                having: None,
                window_defs: vec![],
                order_by: vec![],
                limit: Some(Box::new(Expression::IntegerLiteral(
                    crate::parser::ast::IntegerLiteral {
                        token: dummy_token(&limit.to_string(), TokenType::Integer),
                        value: limit as i64,
                    },
                ))),
                offset: (row_offset > 0).then(|| {
                    Box::new(Expression::IntegerLiteral(
                        crate::parser::ast::IntegerLiteral {
                            token: dummy_token(&row_offset.to_string(), TokenType::Integer),
                            value: row_offset as i64,
                        },
                    ))
                }),
                set_operations: vec![],
                prepared: None,
            };
            // Get classification for the synthetic SELECT statement
            let classification = get_classification(&select_all);
            // The scan says whether it applied the synthetic LIMIT and OFFSET. A
            // path that did not may still have stopped early at their sum, so
            // the pair is applied here, as the select layer would
            let mut opened = None;
            let (result, columns, applied, _) = self.execute_simple_table_scan(
                ts,
                &select_all,
                ctx,
                &classification,
                Some(&mut opened),
                None,
            )?;
            let token = opened.as_ref().map(|(token, _)| *token);
            if let Some(out) = read_out {
                *out = opened;
            }
            let result: Box<dyn QueryResult> = if applied {
                result
            } else {
                Box::new(LimitedResult::new(result, Some(limit), row_offset))
            };

            // Prefix column names with table alias (or table name if no alias)
            // Use custom_alias from Aliased expression if provided
            let table_alias = custom_alias.unwrap_or_else(|| {
                ts.alias
                    .as_ref()
                    .map(|a| a.value.clone())
                    .unwrap_or_else(|| ts.name.value.clone())
            });

            let qualified_columns: Vec<String> = columns
                .iter()
                .map(|col| qualified(&table_alias, col))
                .collect();

            return Ok((result, qualified_columns, true, token));
        }

        // Fall back to standard execution for other cases
        self.execute_table_expression_with_filter(expr, ctx, filter, read_out)
            .map(|(result, columns)| (result, columns, false, None))
    }

    /// Materialize a result into a RowVec
    pub(crate) fn materialize_result(mut result: Box<dyn QueryResult>) -> Result<RowVec> {
        // Pre-allocate based on estimate to avoid reallocations
        let mut rows = if let Some(estimate) = result.estimated_count() {
            RowVec::with_capacity(estimate)
        } else {
            RowVec::new()
        };
        let mut row_id = 0i64;
        while result.next() {
            rows.push((row_id, result.take_row()));
            row_id += 1;
        }
        // Surface runtime filter errors (e.g., invalid parameterized REGEXP)
        if let Some(err) = result.last_error() {
            return Err(err);
        }
        Ok(rows)
    }

    /// The inner table of a join: from the explicit transaction when one is
    /// active, so the join sees what the transaction changed, otherwise from
    /// the statement snapshot the outer side reads through; through `view`,
    /// the statement's, which writes what `table_use` says beside it
    pub(super) fn join_table(
        &self,
        snapshot: &StatementSnapshot,
        name: &str,
        view: Option<&crate::storage::mvcc::registry::StatementView>,
        table_use: TableUse,
    ) -> Result<Box<dyn crate::storage::traits::Table>> {
        if let Some(active) = self.active_transaction.lock().unwrap().as_ref() {
            return active.transaction.get_read_table_at(name, view, table_use);
        }
        snapshot.statement_table_at(name, view)
    }

    /// The inner table of an index join for one pass: a later pass reads it
    /// in the layout of the first, and the strategy planned on another
    /// handle must still name its key. None when it does not
    fn index_join_inner(
        &self,
        snapshot: &StatementSnapshot,
        ctx: &ExecutionContext,
        name: &str,
        strategy: &IndexLookupStrategy,
        inner_key_col: &str,
        first: Option<LayoutToken>,
    ) -> Result<Option<Box<dyn crate::storage::traits::Table>>> {
        let table = self.join_table(snapshot, name, ctx.statement_view(), ctx.table_use())?;
        if first.is_some_and(|token| table.layout_token() != token) {
            return Err(Error::SchemaChanged {
                table: name.to_string(),
            });
        }
        let schema = table.schema();
        let is_pk = schema.pk_column_index().is_some_and(|pk| {
            schema.columns[pk]
                .name_lower
                .eq_ignore_ascii_case(inner_key_col)
        });
        let holds = match strategy {
            IndexLookupStrategy::PrimaryKey => is_pk,
            // The same object: a recreated index shares no pointer with it
            IndexLookupStrategy::SecondaryIndex(planned) => {
                !is_pk
                    && table
                        .lookup_index_on_column(inner_key_col)
                        .is_some_and(|index| {
                            std::ptr::addr_eq(Arc::as_ptr(&index), Arc::as_ptr(planned))
                        })
            }
            // Resolved by the table on each probe
            IndexLookupStrategy::TableEquality { .. } => true,
        };
        Ok(holds.then_some(table))
    }

    /// One transaction for every read of a statement. Under the statement's
    /// view its reads already keep one point, so the view stands in for a
    /// snapshot isolation that is asked for
    pub(super) fn new_statement_snapshot(
        &self,
        isolation: crate::core::IsolationLevel,
        view: Option<&crate::storage::mvcc::registry::StatementView>,
    ) -> Result<StatementSnapshot> {
        let isolation = if view.is_some() {
            crate::core::IsolationLevel::ReadCommitted
        } else {
            isolation
        };
        let transaction = self.engine.begin_transaction_with_level(isolation)?;
        Ok(StatementSnapshot::for_reads(
            transaction,
            isolation,
            view.cloned(),
        ))
    }

    /// The aggregation and window functions of `stmt` over a source's rows,
    /// the one dispatch every source hands its rows to: GROUP BY and then
    /// the windows over the groups, the windows alone, or the aggregation
    /// alone, whose layout `aggregate_layout` builds from its columns, its
    /// subqueries answered by the statement's `blocks`
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn group_and_window_rows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
        rows: RowVec,
        columns: &[String],
        blocks: Option<&dyn SubqueryExecutor>,
        aggregate_layout: impl FnOnce(&[String]) -> Option<OutputLayout>,
    ) -> Result<WindowOutput> {
        match (
            classification.has_aggregation,
            classification.has_window_functions,
        ) {
            (true, true) => {
                self.execute_aggregation_then_windows(stmt, ctx, &rows, columns, blocks)
            }
            (false, true) => {
                self.execute_select_with_window_functions(stmt, ctx, &rows, columns, blocks)
            }
            _ => {
                let result =
                    self.execute_select_with_aggregation(stmt, ctx, rows, columns, blocks)?;
                let layout = aggregate_layout(result.columns());
                Ok((result, layout))
            }
        }
    }

    /// `group_and_window_rows` as the SELECT's output, its layout over the
    /// input `columns` named under `table_alias`
    fn grouped_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
        rows: RowVec,
        columns: &[String],
        table_alias: Option<&str>,
    ) -> SelectResult {
        let (result, layout) =
            self.group_and_window_rows(stmt, ctx, classification, rows, columns, None, |out| {
                Self::result_layout(stmt, columns, table_alias, out)
            })?;
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, SelectShape::laid_out(layout)))
    }

    /// A join's grouped output over its joined `rows` of `columns`, the first
    /// `left_width` its left side's as read: through the grouped blocks when
    /// the statement belongs to their family, held with the `plan`; else as
    /// any grouped statement groups its rows
    #[allow(clippy::too_many_arguments)]
    fn grouped_join_select(
        &self,
        plan: Option<&PreparedJoin>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
        rows: RowVec,
        columns: &[String],
        left_width: usize,
    ) -> SelectResult {
        let in_family = Self::grouped_family_reach(stmt, classification) && {
            let decide = || self.grouped_family(stmt, Some(ctx)).is_ok();
            plan.map_or_else(decide, |plan| plan.grouped(decide))
        };
        if in_family {
            let build = || {
                let mut leaves = Vec::new();
                let mut left_leaves = 0;
                if let Some(Expression::JoinSource(join)) = stmt.table_expr.as_deref() {
                    self.join_leaves(&join.left, ctx, &mut leaves);
                    left_leaves = leaves.len();
                    self.join_leaves(&join.right, ctx, &mut leaves);
                }
                let sides = (left_leaves, left_width);
                self.prepare_grouped_join(stmt, columns, (&leaves, sides), ctx)
            };
            let held = match plan {
                Some(plan) => plan.grouped_blocks(columns, ctx.params().len(), build)?,
                None => std::sync::Arc::new(build()?),
            };
            if let Some(held) = held.as_ref() {
                let run = self.grouped_runner(held, ctx)?.ok_or_else(|| {
                    Error::internal("a table's columns changed while the statement prepared")
                })?;
                return self.group_held_rows(held, &run, ctx, rows, columns, |out| {
                    Self::result_layout(&held.stmt, columns, None, out)
                });
            }
        }
        self.grouped_select(stmt, ctx, classification, rows, columns, None)
    }

    /// Each source the join `expr` reads, left to right, with its column
    /// names when known before the join reads it, a table's or a CTE's, and
    /// their count, also known for a derived table selecting no star and a
    /// VALUES list
    fn join_leaves(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        leaves: &mut Vec<super::prepared_select::JoinLeaf>,
    ) {
        if let Expression::JoinSource(join) = expr {
            self.join_leaves(&join.left, ctx, leaves);
            self.join_leaves(&join.right, ctx, leaves);
            return;
        }
        let columns = match expr {
            Expression::TableSource(ts) => match ctx.get_cte_by_lower(&ts.name.value_lower) {
                Some((columns, _)) => Some(columns.to_vec()),
                None => self
                    .base_table_side_schema(expr, CteNames::Context(ctx))
                    .map(|schema| schema.column_names_arc().to_vec()),
            },
            _ => None,
        };
        let width = match expr {
            _ if columns.is_some() => columns.as_ref().map(Vec::len),
            Expression::SubquerySource(ss) => {
                let items = &ss.subquery.columns;
                let star = |c: &Expression| {
                    matches!(c, Expression::Star(_) | Expression::QualifiedStar(_))
                };
                (!items.iter().any(star)).then_some(items.len())
            }
            Expression::ValuesSource(values) => values.rows.first().map(Vec::len),
            _ => None,
        };
        leaves.push(super::prepared_select::JoinLeaf {
            name: get_table_alias_from_expr(expr)
                .map(|name| name.to_lowercase())
                .unwrap_or_default(),
            columns,
            width,
        });
    }

    /// `rows` of `columns` grouped as `held` prepared them, asking the
    /// `run`'s blocks
    fn group_held_rows(
        &self,
        held: &super::prepared_select::PreparedGrouped,
        run: &super::prepared_select::PreparedRunner<'_>,
        ctx: &ExecutionContext,
        rows: RowVec,
        columns: &[String],
        layout: impl FnOnce(&[String]) -> Option<OutputLayout>,
    ) -> SelectResult {
        let slots_ctx;
        let ctx = match held.reads_run_slots() {
            true => {
                slots_ctx = ctx
                    .with_slot_params(run.params().unwrap_or_default().iter().cloned().collect());
                &slots_ctx
            }
            false => ctx,
        };
        let (result, layout) = self.group_and_window_rows(
            &held.stmt,
            ctx,
            &held.classification,
            rows,
            columns,
            Some(run),
            layout,
        )?;
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, SelectShape::laid_out(layout)))
    }

    /// A grouped statement whose subqueries run as `grouped` prepared them:
    /// over a source's rows, or a table's, the general aggregation asking
    /// the blocks of each group
    fn execute_prepared_grouped(
        &self,
        grouped: super::prepared_select::GroupedRun<'_>,
        branch: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> SelectResult {
        let Some(held) = grouped.held.clone() else {
            // The general path groups the rows the source gave, as `branch`
            let Some((columns, rows)) = grouped.source_rows(None, ctx)? else {
                return Err(Error::internal("a grouped statement without its rows"));
            };
            return self.execute_query_on_memory_result(branch, ctx, columns.to_vec(), rows);
        };
        let held = &*held;
        let run = self.grouped_runner(held, ctx)?.ok_or_else(|| {
            Error::internal("a table's columns changed while the statement prepared")
        })?;
        let alias = grouped.scope_name().map(SmartString::from);
        if let Some((columns, rows)) = grouped.source_rows(Some(&run), ctx)? {
            return self.group_held_rows(held, &run, ctx, rows, &columns, |out| {
                Self::shared_result_layout(&held.stmt, &columns, alias.as_deref(), out)
            });
        }
        let Some(Expression::TableSource(table_source)) = held.stmt.table_expr.as_deref() else {
            return Err(Error::internal("a grouped statement without its table"));
        };
        // A value read once for the statement reads its run slot as a parameter
        let slots_ctx;
        let ctx = match held.reads_run_slots() {
            true => {
                slots_ctx = ctx
                    .with_slot_params(run.params().unwrap_or_default().iter().cloned().collect());
                &slots_ctx
            }
            false => ctx,
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::grouped_table_opening();
        if held.reads_through_where() {
            let name = &table_source.name.value_lower;
            let (columns, rows) = held.passing_rows(self, &run, name, ctx)?;
            let alias = table_source.alias.as_ref().map_or(name, |a| &a.value_lower);
            let (result, layout) = self.group_and_window_rows(
                &held.stmt,
                ctx,
                &held.classification,
                rows,
                &columns,
                Some(&run),
                |out| Self::shared_result_layout(&held.stmt, &columns, Some(alias), out),
            )?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, SelectShape::laid_out(layout)));
        }
        let mut read = None;
        let result = self.execute_simple_table_scan(
            table_source,
            &held.stmt,
            ctx,
            &held.classification,
            Some(&mut read),
            Some(&run),
        );
        if let Some((_, schema)) = &read {
            held.table_fits(schema)?;
        }
        result
    }

    /// True when a filter carries a subquery anywhere inside it. The grouped
    /// stream compiles its filters raw, where a subquery has no executor
    pub(super) fn filter_has_subquery(expr: &Expression) -> bool {
        use crate::parser::ast::Expression as E;
        let any = |exprs: &[Expression]| exprs.iter().any(Self::filter_has_subquery);
        match expr {
            E::ScalarSubquery(_) | E::Exists(_) | E::AllAny(_) => true,
            E::Identifier(_)
            | E::QualifiedIdentifier(_)
            | E::Column(_)
            | E::IntegerLiteral(_)
            | E::FloatLiteral(_)
            | E::StringLiteral(_)
            | E::BooleanLiteral(_)
            | E::NullLiteral(_)
            | E::IntervalLiteral(_)
            | E::Parameter(_)
            | E::Star(_)
            | E::QualifiedStar(_)
            | E::Default(_) => false,
            E::Prefix(p) => Self::filter_has_subquery(&p.right),
            E::Infix(i) => {
                Self::filter_has_subquery(&i.left) || Self::filter_has_subquery(&i.right)
            }
            E::List(l) => any(&l.elements),
            E::ExpressionList(l) => any(&l.expressions),
            E::Distinct(d) => Self::filter_has_subquery(&d.expr),
            E::In(i) => Self::filter_has_subquery(&i.left) || Self::filter_has_subquery(&i.right),
            E::InHashSet(i) => Self::filter_has_subquery(&i.column),
            E::QuantifiedSet(q) => Self::filter_has_subquery(&q.left),
            E::Between(b) => {
                Self::filter_has_subquery(&b.expr)
                    || Self::filter_has_subquery(&b.lower)
                    || Self::filter_has_subquery(&b.upper)
            }
            E::Like(l) => {
                Self::filter_has_subquery(&l.left)
                    || Self::filter_has_subquery(&l.pattern)
                    || l.escape.as_deref().is_some_and(Self::filter_has_subquery)
            }
            E::Case(c) => {
                c.value.as_deref().is_some_and(Self::filter_has_subquery)
                    || c.when_clauses.iter().any(|w| {
                        Self::filter_has_subquery(&w.condition)
                            || Self::filter_has_subquery(&w.then_result)
                    })
                    || c.else_value
                        .as_deref()
                        .is_some_and(Self::filter_has_subquery)
            }
            E::Cast(c) => Self::filter_has_subquery(&c.expr),
            E::FunctionCall(f) => Self::function_has_subquery(f),
            E::Aliased(a) => Self::filter_has_subquery(&a.expression),
            E::Window(w) => {
                Self::function_has_subquery(&w.function)
                    || any(&w.partition_by)
                    || w.order_by
                        .iter()
                        .any(|o| Self::filter_has_subquery(&o.expression))
            }
            E::TableSource(_)
            | E::JoinSource(_)
            | E::SubquerySource(_)
            | E::ValuesSource(_)
            | E::CteReference(_)
            | E::FunctionTableSource(_) => true,
        }
    }

    fn function_has_subquery(f: &crate::parser::ast::FunctionCall) -> bool {
        f.arguments.iter().any(Self::filter_has_subquery)
            || f.filter.as_deref().is_some_and(Self::filter_has_subquery)
            || f.order_by
                .iter()
                .any(|o| Self::filter_has_subquery(&o.expression))
    }

    /// A GROUP BY + LIMIT join whose groups are the left rows: the right side
    /// is probed per left row through its index, each group is folded as the
    /// joined rows stream past, HAVING runs on the group at once, and the
    /// scan stops after LIMIT + OFFSET groups. Nothing is materialised or
    /// hashed and the answer is complete by construction. None when the shape
    /// needs something this path does not do; the caller then reduces and
    /// hashes as before.
    #[allow(clippy::too_many_arguments)]
    fn try_grouped_index_join(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        join_source: &JoinTableSource,
        join_type: &str,
        right_alias: Option<&str>,
        left: SideRead<'_>,
        right_filter: Option<&Expression>,
        cross_filter: Option<&Expression>,
        prepared: Option<&PreparedJoin>,
        (plan, probe): (&SemijoinReduction, &IndexProbe),
    ) -> Result<Option<SelectOutput>> {
        // The statement's blocks run the filters' subqueries, on each site
        let run = left.run.map(|(run, _)| run);
        let filters_have_subquery = run.is_none()
            && [left.filter, right_filter, cross_filter]
                .into_iter()
                .flatten()
                .any(Self::filter_has_subquery);
        let Some((aggregations, group_names)) = self.grouped_index_join_applies(
            stmt,
            join_type,
            join_source.condition.as_deref(),
            right_filter.is_some(),
            filters_have_subquery,
            Some(ctx),
        )?
        else {
            return Ok(None);
        };
        let Some((table_name, lookup_strategy, inner_key_col, outer_key_col)) =
            self.probe_strategy(probe, true, false, false)
        else {
            return Ok(None);
        };

        // One transaction for the outer fetches and the inner probes; inside an
        // explicit transaction the outer side is read whole
        let mut snapshot = match ctx.statement_snapshot() {
            Some(snapshot) => snapshot.clone(),
            None => self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?,
        };
        let mut ctx_join = ctx.with_statement_snapshot(snapshot.clone());
        // Opened before the outer side is read, so an index no longer the
        // planned one leaves the reduction nothing read twice
        let Some(inner_table) = self.index_join_inner(
            &snapshot,
            ctx,
            &table_name,
            &lookup_strategy,
            &inner_key_col,
            None,
        )?
        else {
            return Ok(None);
        };
        // The outer side is the left table, read whole row: its columns are
        // named from its schema, so every column below is found before it is
        // read, and a read under another schema is a schema change
        let (Expression::TableSource(left_ts), Some(left_schema)) = (
            join_source.left.as_ref(),
            self.base_table_side_schema(&join_source.left, CteNames::Context(ctx)),
        ) else {
            return Ok(None);
        };
        let left_alias = left_ts
            .alias
            .as_ref()
            .unwrap_or(&left_ts.name)
            .value
            .as_str();
        let outer_cols: Vec<String> = left_schema
            .columns
            .iter()
            .map(|col| qualified(left_alias, &col.name))
            .collect();
        let unqualified = |name: &str| bare_name(name).to_string();
        let outer_key_lower = outer_key_col.to_lowercase();
        let outer_key_unqualified = unqualified(&outer_key_lower);
        let Some(outer_key_idx) = outer_cols
            .iter()
            .position(|c| c.to_lowercase() == outer_key_lower)
            .or_else(|| {
                outer_cols
                    .iter()
                    .position(|c| unqualified(&c.to_lowercase()) == outer_key_unqualified)
            })
        else {
            return Ok(None);
        };
        if let Some(run) = run {
            run.blocks.side_fits(1, &inner_table.schema_arc())?;
        }
        let inner_token = inner_table.layout_token();
        let inner_alias = right_alias.unwrap_or(&table_name);
        let inner_cols: Vec<String> = inner_table
            .schema()
            .columns
            .iter()
            .map(|col| qualified(inner_alias, &col.name))
            .collect();
        // The inner key column, checked on each fetched row: the ids name the
        // rows the index held for the key, not the rows' keys now
        let inner_key_idx = inner_cols
            .iter()
            .position(|c| bare_name(c).eq_ignore_ascii_case(&inner_key_col));
        let outer_len = outer_cols.len();
        let mut all_columns = outer_cols;
        all_columns.extend(inner_cols.iter().cloned());

        // Where each group column and aggregate argument lives in the joined row
        let mut group_idx = Vec::with_capacity(group_names.len());
        for name in &group_names {
            match Self::find_column_index_by_name(name, &all_columns) {
                Some(idx) => group_idx.push(idx),
                None => return Ok(None),
            }
        }
        let mut agg_idx = Vec::with_capacity(aggregations.len());
        for agg in &aggregations {
            if agg.column == "*" {
                agg_idx.push(None);
            } else {
                match Self::find_column_index_by_name(&agg.column, &all_columns) {
                    Some(idx) => agg_idx.push(Some(idx)),
                    None => return Ok(None),
                }
            }
        }
        let in_explicit_transaction = self.active_transaction.lock().unwrap().is_some();
        let mut outer_limit = if in_explicit_transaction {
            None
        } else {
            Some(plan.cap)
        };
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::join_side_ran();
        let (outer_result, outer_cols, outer_bounded, outer_token) =
            self.read_join_side(&join_source.left, &ctx_join, left, None, outer_limit, 0)?;
        if outer_cols[..] != all_columns[..outer_len] {
            return Err(Error::SchemaChanged {
                table: left_ts.name.value.to_string(),
            });
        }
        if !outer_bounded {
            outer_limit = None;
        }
        // Without a cross filter the operator builds only the columns the
        // groups and aggregates read
        let projection = if cross_filter.is_none() {
            use crate::executor::operators::index_nested_loop::ColumnSource;
            let mut sources: Vec<ColumnSource> = Vec::new();
            let mut place = |idx: usize| -> usize {
                let source = if idx < outer_cols.len() {
                    ColumnSource::Outer(idx)
                } else {
                    ColumnSource::Inner(idx - outer_cols.len())
                };
                match sources.iter().position(|s| *s == source) {
                    Some(at) => at,
                    None => {
                        sources.push(source);
                        sources.len() - 1
                    }
                }
            };
            for idx in group_idx.iter_mut() {
                *idx = place(*idx);
            }
            for idx in agg_idx.iter_mut().flatten() {
                *idx = place(*idx);
            }
            Some(sources)
        } else {
            None
        };
        let mut result_columns: Vec<String> = group_names.clone();
        for agg in &aggregations {
            result_columns.push(
                agg.alias
                    .clone()
                    .unwrap_or_else(|| agg.get_expression_name()),
            );
        }
        let having = match &stmt.having {
            Some(having) => {
                let processed = self.process_where_subqueries(having, ctx)?;
                let group_items: Vec<crate::executor::aggregation::GroupByItem> = group_names
                    .iter()
                    .map(|name| crate::executor::aggregation::GroupByItem::Column(name.clone()))
                    .collect();
                let slots = crate::executor::aggregation::aggregate_slot_keys(
                    &group_items,
                    &aggregations,
                    &stmt.columns,
                );
                Some(
                    RowFilter::with_aliases(&processed, &result_columns, &slots)?.with_context(ctx),
                )
            }
            None => None,
        };
        let mut cross = match cross_filter {
            Some(cross) => {
                let blocks = run.map(|run| run.blocks);
                let program = self.join_where(
                    prepared,
                    WhereSite::IndexJoin,
                    stmt,
                    cross,
                    &all_columns,
                    blocks,
                )?;
                Some(JoinedWhere::new(program, run.map(|run| run.run), ctx))
            }
            None => None,
        };
        let cross_site = run.and_then(|run| run.site(false));
        let cross_blocks = cross_site
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        let pair_site = run.and_then(|run| run.site(true));
        let pair_blocks = pair_site.as_ref().map(|site| site as &dyn SubqueryExecutor);
        // A right-side filter the storage layer takes whole runs inside the
        // inner fetch, built once so every chunk sees the same predicate; a
        // partial or refused one stays a residual on the joined row
        let inner_filter = right_filter
            .filter(|rf| run.is_none() || !Self::has_subqueries(rf))
            .and_then(|rf| {
                let rf = crate::executor::utils::strip_table_qualifier(rf, inner_alias);
                match pushdown::try_pushdown(&rf, inner_table.schema(), Some(ctx)) {
                    (Some(expr), false) => Some(expr),
                    _ => None,
                }
            });
        // The probe answers one equality, so the rest of the ON clause and a
        // right-side filter the fetch did not take are asked of each pair;
        // with the statement's blocks, the clauses they rewrote
        let (on, outer_names, inner_names, right) = match run {
            Some(run) => (
                run.blocks.on.as_ref(),
                &run.blocks.left,
                &run.blocks.right,
                run.blocks.sides[1].clone(),
            ),
            None => (
                join_source.condition.as_deref(),
                &outer_cols,
                &inner_cols,
                right_filter.map(|rf| add_table_qualifier(rf, inner_alias)),
            ),
        };
        let residual = combine_predicates_with_and(
            on.and_then(|on| {
                // The left side is the outer one here
                index_join_residual(
                    on,
                    outer_names,
                    inner_names,
                    (Some(outer_key_idx), inner_key_idx),
                )
            })
            .into_iter()
            .chain(right.filter(|_| inner_filter.is_none()))
            .collect(),
        );
        let build_residual_filter = || {
            residual
                .as_ref()
                .map(|expr| match run {
                    Some(run) => run
                        .blocks
                        .compile(expr, true, &self.function_registry)
                        .map(|p| JoinFilter::from_program(p).with_context(ctx)),
                    None => {
                        JoinFilter::new(expr, &outer_cols, &inner_cols, &self.function_registry)
                            .map(|f| f.with_context(ctx))
                    }
                })
                .transpose()
        };
        let mut funcs: Vec<crate::functions::aggregate::AggregateState> =
            Vec::with_capacity(aggregations.len());
        for agg in &aggregations {
            funcs.push(
                agg.state(&self.function_registry)
                    .ok_or_else(|| Error::internal(format!("aggregate {} vanished", agg.name)))?,
            );
        }
        let count_star = Value::Integer(1);
        let op_join_type = OperatorJoinType::parse(join_type);
        let want = plan.want;
        let failed = crate::executor::aggregation::extra_args_failure(&aggregations);

        // Fold one finished group into an output row; true when HAVING kept it
        let finish = |first: &Row,
                      funcs: &mut [crate::functions::aggregate::AggregateState],
                      groups: &mut RowVec|
         -> Result<bool> {
            if let Some(error) = failed {
                return Err(error.clone());
            }
            let mut values = Vec::with_capacity(result_columns.len());
            for &idx in &group_idx {
                values.push(first.get(idx).cloned().unwrap_or_else(Value::null_unknown));
            }
            for func in funcs.iter() {
                values.push(func.result()?);
            }
            let row = Row::from_values(values);
            if let Some(having) = &having {
                if !having.matches_checked(&row)? {
                    return Ok(false);
                }
            }
            groups.push((groups.len() as i64, row));
            Ok(true)
        };

        let mut groups = RowVec::new();
        let mut outer_result = outer_result;
        let mut inner_table = Some(inner_table);
        let mut fetched = 0usize;
        let mut consistent = false;
        loop {
            let outer_op: Box<dyn Operator> =
                Box::new(QueryResultOperator::new(outer_result, outer_cols.len()));
            let table = match inner_table.take() {
                Some(table) => table,
                None => match self.index_join_inner(
                    &snapshot,
                    ctx,
                    &table_name,
                    &lookup_strategy,
                    &inner_key_col,
                    Some(inner_token),
                )? {
                    Some(table) => table,
                    None => return Ok(None),
                },
            };
            JoinPath::GroupedIndexJoin.taken();
            let mut op = IndexNestedLoopJoinOperator::new(
                outer_op,
                table,
                inner_cols.len(),
                op_join_type,
                outer_key_idx,
                lookup_strategy.clone(),
                build_residual_filter()?,
            )
            .with_inner_key(inner_key_idx)
            .with_blocks(pair_blocks);
            if let Some(filter) = &inner_filter {
                op = op.with_inner_filter(filter.clone());
            }
            if let Some(sources) = &projection {
                op = op.with_projection(sources.clone());
            }
            op.open()?;
            // The group columns cover the left primary key, so a change in them
            // is a new left row
            let mut current: Option<Row> = None;
            let mut enough = false;
            while let Some(row_ref) = op.next()? {
                let row = row_ref.into_owned();
                if let Some(cross) = cross.as_mut() {
                    if !cross.matches(&row, cross_blocks)? {
                        continue;
                    }
                }
                let new_group = match &current {
                    Some(first) => group_idx.iter().any(|&idx| first.get(idx) != row.get(idx)),
                    None => true,
                };
                if new_group {
                    if let Some(first) = current.take() {
                        if finish(&first, &mut funcs, &mut groups)? && groups.len() >= want {
                            enough = true;
                            break;
                        }
                    }
                    for func in funcs.iter_mut() {
                        func.reset();
                    }
                    current = Some(row.clone());
                }
                for (i, func) in funcs.iter_mut().enumerate() {
                    let value = match agg_idx[i] {
                        Some(idx) => row.get(idx),
                        None => Some(&count_star),
                    };
                    if let Some(value) = value {
                        func.accumulate(value);
                    }
                }
            }
            if !enough {
                if let Some(first) = current.take() {
                    finish(&first, &mut funcs, &mut groups)?;
                }
            }
            op.close()?;
            // A probe the inner table could not answer: the groups so far are
            // no answer, and the caller reduces and hashes instead
            if op.needs_fallback() {
                return Ok(None);
            }
            if enough || groups.len() >= want {
                break;
            }
            let cap = match outer_limit {
                Some(cap) if op.outer_rows_seen() == cap => cap,
                _ => break,
            };
            let boundary = op.last_outer_row().cloned();
            fetched = fetched.saturating_add(cap);
            let next = next_left_chunk(fetched, groups.len(), want);
            outer_limit = Some(next);
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::join_next_chunk();
            let window = LeftWindow {
                offset: fetched,
                cap: next,
                boundary,
            };
            if let Some((more, _, bounded, token)) =
                self.left_chunk(&join_source.left, &ctx_join, left, &window, consistent)?
            {
                same_outer_source(&join_source.left, outer_token, token, bounded)?;
                outer_result = more;
                continue;
            }
            snapshot = self.new_statement_snapshot(
                crate::core::IsolationLevel::SnapshotIsolation,
                ctx.statement_view(),
            )?;
            ctx_join = ctx.with_statement_snapshot(snapshot.clone());
            consistent = true;
            groups.clear();
            outer_limit = Some(fetched.saturating_add(next));
            fetched = 0;
            let (restarted, _, bounded, token) =
                self.read_join_side(&join_source.left, &ctx_join, left, None, outer_limit, 0)?;
            same_outer_source(&join_source.left, outer_token, token, bounded)?;
            outer_result = restarted;
        }

        let (final_columns, final_rows) = self.apply_post_aggregation_expressions(
            stmt,
            ctx,
            &all_columns,
            &aggregations,
            result_columns,
            groups,
            None,
        )?;
        let mut rows = final_rows.into_vec();
        rows.drain(..plan.offset.min(rows.len()));
        rows.truncate(plan.limit);
        let columns = CompactArc::new(final_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&columns), RowVec::from_vec(rows));
        Ok(Some((Box::new(result), columns, true, None)))
    }

    /// Runs a join operator to completion or to `limit` rows. The cross-table
    /// filter is applied per row so filtered rows never count toward the limit.
    fn collect_join_rows(
        join_op: &mut dyn Operator,
        limit: Option<u64>,
        mut cross: Option<&mut JoinedWhere<'_, '_>>,
        blocks: Option<&dyn SubqueryExecutor>,
        out: &mut RowVec,
    ) -> Result<()> {
        join_op.open()?;
        while let Some(row_ref) = join_op.next()? {
            let row = row_ref.into_owned();
            if let Some(filter) = cross.as_deref_mut() {
                if !filter.matches(&row, blocks)? {
                    continue;
                }
            }
            out.push((out.len() as i64, row));
            if limit.is_some_and(|lim| out.len() >= lim as usize) {
                break;
            }
        }
        join_op.close()
    }

    /// Materialize a result into an CompactArc<Vec<Row>> for zero-copy sharing with joins
    ///
    /// This method first tries to extract an Arc directly from the result (e.g., from CTE cache),
    /// falling back to iterating and collecting rows if not possible.
    pub(crate) fn materialize_result_arc(
        mut result: Box<dyn QueryResult>,
    ) -> Result<CompactArc<Vec<Row>>> {
        // Try to extract Arc directly (zero-copy path for CTEs)
        if let Some(arc_rows) = result.try_into_arc_rows() {
            return Ok(arc_rows);
        }

        // Pre-allocate based on estimate to avoid reallocations
        let mut rows = if let Some(estimate) = result.estimated_count() {
            Vec::with_capacity(estimate)
        } else {
            Vec::new()
        };
        while result.next() {
            rows.push(result.take_row());
        }
        // Surface runtime filter errors (e.g., invalid parameterized REGEXP)
        if let Some(err) = result.last_error() {
            return Err(err);
        }
        Ok(CompactArc::new(rows))
    }

    /// Project rows with optional table alias for correlated subquery support
    ///
    /// If `all_columns_lower` is provided, it will be used directly instead of
    /// computing lowercase column names per-query. Pass pre-cached values from
    /// `schema.column_names_lower_arc()` for optimal performance.
    pub(crate) fn project_rows_with_alias(
        &self,
        select_exprs: &[Expression],
        rows: RowVec,
        all_columns: &[String],
        all_columns_lower: Option<&[String]>,
        ctx: &ExecutionContext,
        table_alias: Option<&str>,
    ) -> Result<RowVec> {
        // Correlated subqueries are evaluated per row
        if Self::has_correlated_select_subqueries(select_exprs) {
            return self.project_rows_correlated(
                select_exprs,
                rows,
                all_columns,
                all_columns_lower,
                ctx,
                table_alias,
            );
        }
        let plan = self.select_rows_plan(
            select_exprs,
            all_columns,
            all_columns_lower,
            ctx,
            table_alias,
        )?;
        Self::run_select_projection(&plan, rows, ctx)
    }

    /// The plan of [`Self::project_rows_with_alias`] for a SELECT list
    /// without correlated subqueries
    fn select_rows_plan(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
        all_columns_lower: Option<&[String]>,
        ctx: &ExecutionContext,
        table_alias: Option<&str>,
    ) -> Result<SelectProjection> {
        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        let processed = self.try_process_select_subqueries(select_exprs, ctx)?;
        self.select_projection(
            processed.as_deref().unwrap_or(select_exprs),
            all_columns,
            all_columns_lower,
            table_alias,
        )
    }

    /// How a SELECT list without correlated subqueries projects rows of
    /// `all_columns`, found before any row
    fn select_projection(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
        all_columns_lower: Option<&[String]>,
        table_alias: Option<&str>,
    ) -> Result<SelectProjection> {
        // Check if this is SELECT * (no projection needed)
        // Note: QualifiedStar (t.*) DOES need projection to filter columns
        if select_exprs.len() == 1 && matches!(&select_exprs[0], Expression::Star(_)) {
            return Ok(SelectProjection::Whole);
        }

        // Handle SELECT t.* - filter to only columns matching the qualifier
        if select_exprs.len() == 1 {
            if let Expression::QualifiedStar(qs) = &select_exprs[0] {
                let mut indices_to_use = Vec::new();
                for_each_qualified_star_column(
                    all_columns,
                    &qs.qualifier,
                    table_alias,
                    |idx, _| indices_to_use.push(idx),
                );

                // OPTIMIZATION: If selecting all columns in order (identity), pass through directly
                let is_identity = indices_to_use.len() == all_columns.len()
                    && indices_to_use.iter().enumerate().all(|(i, &idx)| i == idx);

                if is_identity {
                    return Ok(SelectProjection::Whole);
                }
                return Ok(SelectProjection::Subset(indices_to_use));
            }
        }

        // OPTIMIZATION: For fast path check, use linear search to avoid HashMap allocation.
        // Linear search with eq_ignore_ascii_case is faster than HashMap for small N
        // due to cache locality and zero allocation overhead.
        // HashMap will be built lazily only if we need project_rows_optimized.

        // Helper function for column lookup using linear search
        let find_column_index = |name: &str, columns: &[String]| -> Option<usize> {
            // Linear search with case-insensitive comparison
            columns
                .iter()
                .position(|c| c.eq_ignore_ascii_case(name))
                .or_else(|| {
                    // Try unqualified match for qualified column names
                    columns.iter().position(|c| {
                        split_qualified(c).is_some_and(|(_, bare)| bare.eq_ignore_ascii_case(name))
                    })
                })
        };
        let find_qualified = |qid: &QualifiedIdentifier| {
            input_column_index(
                all_columns,
                Some(&qid.qualifier.value_lower),
                &qid.name.value_lower,
            )
        };

        // Fast path: Check if all expressions are simple column references
        // If so, we can use direct index-based projection without creating Evaluators
        let mut simple_column_indices: Vec<usize> = Vec::with_capacity(select_exprs.len());
        let mut all_simple = true;

        for expr in select_exprs.iter() {
            match expr {
                Expression::Star(_) => {
                    // Expand all columns
                    for idx in 0..all_columns.len() {
                        simple_column_indices.push(idx);
                    }
                }
                Expression::QualifiedStar(qs) => {
                    for_each_qualified_star_column(
                        all_columns,
                        &qs.qualifier,
                        table_alias,
                        |idx, _| simple_column_indices.push(idx),
                    );
                }
                Expression::Identifier(id) => {
                    // Use linear search for fast path check
                    if let Some(idx) = find_column_index(&id.value_lower, all_columns) {
                        simple_column_indices.push(idx);
                    } else {
                        all_simple = false;
                        break;
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    if let Some(idx) = find_qualified(qid) {
                        simple_column_indices.push(idx);
                    } else {
                        all_simple = false;
                        break;
                    }
                }
                Expression::Aliased(aliased) => {
                    // Check if the inner expression is a simple column reference
                    match &*aliased.expression {
                        Expression::Identifier(id) => {
                            if let Some(idx) = find_column_index(&id.value_lower, all_columns) {
                                simple_column_indices.push(idx);
                            } else {
                                all_simple = false;
                                break;
                            }
                        }
                        Expression::QualifiedIdentifier(qid) => {
                            if let Some(idx) = find_qualified(qid) {
                                simple_column_indices.push(idx);
                            } else {
                                all_simple = false;
                                break;
                            }
                        }
                        _ => {
                            all_simple = false;
                            break;
                        }
                    }
                }
                _ => {
                    all_simple = false;
                    break;
                }
            }
        }

        // Use fast path if all columns are simple references
        // IMPORTANT: Only use take_columns if no index appears twice, otherwise
        // the second take of the same index would get null (value already moved)
        if all_simple && !simple_column_indices.is_empty() {
            // Check for duplicate indices (can happen with ambiguous column names after JOIN)
            let mut seen_indices: FxHashSet<usize> = FxHashSet::default();
            let has_duplicates = simple_column_indices
                .iter()
                .any(|&idx| !seen_indices.insert(idx));

            if !has_duplicates {
                // OPTIMIZATION: Check if this is an identity projection (all columns in order)
                // If so, skip the map entirely - no transformation needed
                let is_identity = simple_column_indices.len() == all_columns.len()
                    && simple_column_indices
                        .iter()
                        .enumerate()
                        .all(|(i, &idx)| i == idx);

                if is_identity {
                    return Ok(SelectProjection::Whole);
                }

                // OPTIMIZATION: Use take_columns to move values instead of cloning
                return Ok(SelectProjection::Take(simple_column_indices));
            }
            // Fall through to slow path if there are duplicates
        }

        // OPTIMIZATION: Pre-compute table alias lowercase
        let table_alias_lower: Option<String> = table_alias.map(|a| a.to_lowercase());
        let col_index_map_lower = lower_index_map(all_columns, all_columns_lower);

        // Use the optimized path with pre-compiled expressions
        Ok(SelectProjection::Actions(self.projection_actions(
            select_exprs,
            all_columns,
            &col_index_map_lower,
            table_alias_lower.as_deref(),
        )?))
    }

    /// The rows `plan` projects
    fn run_select_projection(
        plan: &SelectProjection,
        rows: RowVec,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        if let SelectProjection::Whole = plan {
            return Ok(rows);
        }
        let mut projected = RowVec::with_capacity(rows.len());
        Self::project_select_into(plan, rows, ctx, &mut projected)?;
        Ok(projected)
    }

    /// The rows `plan` projects, put into `out`
    fn project_select_into(
        plan: &SelectProjection,
        rows: RowVec,
        ctx: &ExecutionContext,
        out: &mut impl RowSink,
    ) -> Result<()> {
        match plan {
            SelectProjection::Whole => {
                for (id, row) in rows.into_iter() {
                    out.put(id, row);
                }
            }
            SelectProjection::Subset(indices) => {
                // Project rows to only include matching columns
                for (id, row) in rows.into_iter() {
                    out.put(id, row.clone_subset(indices));
                }
            }
            SelectProjection::Take(indices) => {
                for (id, row) in rows.into_iter() {
                    out.put(id, row.take_columns(indices));
                }
            }
            SelectProjection::Actions(actions) => {
                let bound = bound_items(
                    actions,
                    ctx.params(),
                    |action| match action {
                        ExprAction::Compiled(program) => Some(program),
                        ExprAction::Concat(concat) => concat.source(),
                        _ => None,
                    },
                    ExprAction::of_program,
                );
                return Self::run_projection_actions(
                    bound.as_deref().unwrap_or(actions),
                    rows,
                    ctx,
                    out,
                );
            }
        }
        Ok(())
    }

    /// Project rows whose SELECT list holds a correlated subquery, binding
    /// each row as the subquery's outer row
    fn project_rows_correlated(
        &self,
        select_exprs: &[Expression],
        rows: RowVec,
        all_columns: &[String],
        all_columns_lower: Option<&[String]>,
        ctx: &ExecutionContext,
        table_alias: Option<&str>,
    ) -> Result<RowVec> {
        let col_index_map_lower = lower_index_map(all_columns, all_columns_lower);

        // Correlated subquery path: expressions change per-row, use CompiledEvaluator
        let mut projected = RowVec::with_capacity(rows.len());

        // Create evaluator once and reuse for all rows
        let mut evaluator = CompiledEvaluator::new(&self.function_registry);
        evaluator = evaluator.with_context(ctx);
        evaluator.init_columns(all_columns);

        // OPTIMIZATION: Pre-compute column name mappings outside the loop for correlated subqueries
        let column_keys = ColumnKeyMapping::build_mappings(all_columns, table_alias);

        // OPTIMIZATION: Pre-allocate outer_row_map with capacity and reuse
        // Uses CompactArc<str> keys for zero-cost cloning in the per-row loop
        let base_capacity = all_columns.len() * 2;
        let mut outer_row_map: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
        outer_row_map.reserve(base_capacity);

        // OPTIMIZATION: Wrap all_columns in CompactArc once, reuse for all rows
        let all_columns_arc: CompactArc<Vec<String>> = CompactArc::new(all_columns.to_vec());

        // The columns each qualified star takes, found once for every row
        let qualified_star_cache: FxHashMap<&str, Vec<usize>> = select_exprs
            .iter()
            .filter_map(|expr| {
                let Expression::QualifiedStar(qs) = expr else {
                    return None;
                };
                let mut taken = Vec::new();
                for_each_qualified_star_column(
                    all_columns,
                    &qs.qualifier,
                    table_alias,
                    |idx, _| taken.push(idx),
                );
                Some((qs.qualifier.as_str(), taken))
            })
            .collect();

        for (id, row) in rows.into_iter() {
            // Use CompactVec directly instead of Vec to avoid Vec->CompactVec conversion
            let mut values: CompactVec<Value> = CompactVec::with_capacity(select_exprs.len());

            evaluator.set_row_array(&row);

            // OPTIMIZATION: Clear and reuse outer_row_map instead of creating new
            outer_row_map.clear();

            // Use pre-computed column mappings
            for mapping in &column_keys {
                if let Some(value) = row.get(mapping.index) {
                    // OPTIMIZATION: Clone value once and reuse for all key insertions
                    let cloned_value = value.clone();

                    // Insert with unqualified part first (if column had a dot)
                    if let Some(ref upart) = mapping.unqualified_part {
                        outer_row_map.insert(upart.clone(), cloned_value.clone());
                    }

                    // Insert with qualified name if available
                    if let Some(ref qname) = mapping.qualified_name {
                        outer_row_map.insert(qname.clone(), cloned_value.clone());
                    }

                    // Insert with lowercase column name (move, no clone)
                    outer_row_map.insert(mapping.col_lower.clone(), cloned_value);
                }
            }

            // Create context with outer row (cheap due to Arc)
            let correlated_ctx = ctx.with_outer_row(
                std::mem::take(&mut outer_row_map),
                all_columns_arc.clone(), // Arc clone = cheap
            );

            // Process correlated SELECT expressions for this row
            let processed_exprs: Result<Vec<Expression>> = select_exprs
                .iter()
                .map(|expr| self.process_correlated_expression(expr, &correlated_ctx))
                .collect();

            // Take back the map for reuse
            outer_row_map = correlated_ctx.outer_row.unwrap_or_default();

            let exprs_to_eval = processed_exprs?;

            for expr in exprs_to_eval.iter() {
                match expr {
                    Expression::Star(_) => {
                        // Expand all columns
                        for val in row.iter() {
                            values.push(val.clone());
                        }
                    }
                    Expression::QualifiedStar(qs) => {
                        for &idx in qualified_star_cache
                            .get(qs.qualifier.as_str())
                            .into_iter()
                            .flatten()
                        {
                            if let Some(val) = row.get(idx) {
                                values.push(val.clone());
                            }
                        }
                    }
                    _ => {
                        let value = self.evaluate_select_expr(
                            &mut evaluator,
                            expr,
                            &row,
                            &col_index_map_lower,
                        )?;
                        values.push(value);
                    }
                }
            }

            projected.push((id, Row::from_compact_vec(values)));
        }

        Ok(projected)
    }

    /// How each expression of a non-correlated projection is evaluated:
    /// complex ones are compiled once, before any row
    fn projection_actions(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
        col_index_map_lower: &StringMap<usize>,
        table_alias_lower: Option<&str>,
    ) -> Result<Vec<ExprAction>> {
        // Helper to try building inline COALESCE action (bypasses VM for 7x speedup)
        fn try_build_coalesce_action(
            func: &crate::parser::ast::FunctionCall,
            col_index_map_lower: &StringMap<usize>,
        ) -> Option<ExprAction> {
            if !func.function.eq_ignore_ascii_case("COALESCE") {
                return None;
            }
            if func.arguments.is_empty() {
                return None;
            }

            let mut args: smallvec::SmallVec<[ArgSource; 4]> = smallvec::SmallVec::new();
            for arg in &func.arguments {
                match arg {
                    Expression::Identifier(id) => {
                        let idx = *col_index_map_lower.get(id.value_lower.as_str())?;
                        args.push(ArgSource::Column(idx));
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        let idx = qualified_index(
                            col_index_map_lower,
                            &qid.qualifier.value_lower,
                            &qid.name.value_lower,
                        )?;
                        args.push(ArgSource::Column(idx));
                    }
                    // Handle all literal types
                    Expression::IntegerLiteral(lit) => {
                        args.push(ArgSource::Const(Value::Integer(lit.value)));
                    }
                    Expression::FloatLiteral(lit) => {
                        args.push(ArgSource::Const(Value::Float(lit.value)));
                    }
                    Expression::StringLiteral(lit) => {
                        args.push(ArgSource::Const(string_literal_value(lit)));
                    }
                    Expression::BooleanLiteral(lit) => {
                        args.push(ArgSource::Const(Value::Boolean(lit.value)));
                    }
                    Expression::NullLiteral(_) => {
                        args.push(ArgSource::Const(Value::null_unknown()));
                    }
                    _ => return None, // Complex arg - fall back to VM
                }
            }
            Some(ExprAction::Coalesce(args))
        }

        // Helper to extract column index from identifier expressions
        fn get_col_idx(expr: &Expression, col_index_map_lower: &StringMap<usize>) -> Option<usize> {
            match expr {
                Expression::Identifier(id) => {
                    col_index_map_lower.get(id.value_lower.as_str()).copied()
                }
                Expression::QualifiedIdentifier(qid) => qualified_index(
                    col_index_map_lower,
                    &qid.qualifier.value_lower,
                    &qid.name.value_lower,
                ),
                _ => None,
            }
        }

        // Helper to convert expression to ArgSource (column or literal)
        fn expr_to_arg_source(
            expr: &Expression,
            col_index_map_lower: &StringMap<usize>,
        ) -> Option<ArgSource> {
            match expr {
                Expression::Identifier(id) => {
                    let idx = *col_index_map_lower.get(id.value_lower.as_str())?;
                    Some(ArgSource::Column(idx))
                }
                Expression::QualifiedIdentifier(qid) => {
                    let idx = qualified_index(
                        col_index_map_lower,
                        &qid.qualifier.value_lower,
                        &qid.name.value_lower,
                    )?;
                    Some(ArgSource::Column(idx))
                }
                Expression::IntegerLiteral(lit) => {
                    Some(ArgSource::Const(Value::Integer(lit.value)))
                }
                Expression::FloatLiteral(lit) => Some(ArgSource::Const(Value::Float(lit.value))),
                Expression::StringLiteral(lit) => Some(ArgSource::Const(string_literal_value(lit))),
                Expression::BooleanLiteral(lit) => {
                    Some(ArgSource::Const(Value::Boolean(lit.value)))
                }
                Expression::NullLiteral(_) => Some(ArgSource::Const(Value::null_unknown())),
                _ => None,
            }
        }

        // Helper to convert expression to Value (for condition comparison)
        fn expr_to_value(expr: &Expression) -> Option<Value> {
            match expr {
                Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
                Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
                Expression::StringLiteral(lit) => Some(string_literal_value(lit)),
                Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
                Expression::NullLiteral(_) => Some(Value::null_unknown()),
                _ => None,
            }
        }

        // Helper to try building inline CASE action (bypasses VM)
        fn try_build_case_action(
            case: &crate::parser::ast::CaseExpression,
            col_index_map_lower: &StringMap<usize>,
        ) -> Option<ExprAction> {
            // Only handle searched CASE (no value expression)
            if case.value.is_some() {
                return None;
            }

            let mut branches: smallvec::SmallVec<[CaseBranch; 4]> = smallvec::SmallVec::new();

            for when_clause in &case.when_clauses {
                // Parse condition: support comparisons and IS NULL
                let condition = match &when_clause.condition {
                    Expression::Infix(infix) => {
                        // Handle column op literal (e.g., balance > 50000)
                        if let Some(col_idx) = get_col_idx(&infix.left, col_index_map_lower) {
                            if let Some(value) = expr_to_value(&infix.right) {
                                match infix.operator.as_str() {
                                    "=" => CaseCondition::Equals { col_idx, value },
                                    "!=" | "<>" => CaseCondition::NotEquals { col_idx, value },
                                    ">" => CaseCondition::GreaterThan { col_idx, value },
                                    ">=" => CaseCondition::GreaterOrEqual { col_idx, value },
                                    "<" => CaseCondition::LessThan { col_idx, value },
                                    "<=" => CaseCondition::LessOrEqual { col_idx, value },
                                    _ if infix.operator.eq_ignore_ascii_case("IS") => {
                                        if matches!(&*infix.right, Expression::NullLiteral(_)) {
                                            CaseCondition::IsNull { col_idx }
                                        } else {
                                            return None;
                                        }
                                    }
                                    _ => return None,
                                }
                            } else if infix.operator.eq_ignore_ascii_case("IS") {
                                if matches!(&*infix.right, Expression::NullLiteral(_)) {
                                    CaseCondition::IsNull { col_idx }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        // Handle literal op column (reversed: 50000 < balance)
                        } else {
                            let col_idx = get_col_idx(&infix.right, col_index_map_lower)?;
                            let value = expr_to_value(&infix.left)?;
                            // Reverse the operator since column is on right
                            match infix.operator.as_str() {
                                "=" => CaseCondition::Equals { col_idx, value },
                                "!=" | "<>" => CaseCondition::NotEquals { col_idx, value },
                                ">" => CaseCondition::LessThan { col_idx, value }, // 5 > col means col < 5
                                ">=" => CaseCondition::LessOrEqual { col_idx, value }, // 5 >= col means col <= 5
                                "<" => CaseCondition::GreaterThan { col_idx, value }, // 5 < col means col > 5
                                "<=" => CaseCondition::GreaterOrEqual { col_idx, value }, // 5 <= col means col >= 5
                                _ => return None,
                            }
                        }
                    }
                    _ => return None, // Complex condition - fall back to VM
                };

                // Parse result
                let result = expr_to_arg_source(&when_clause.then_result, col_index_map_lower)?;

                branches.push(CaseBranch { condition, result });
            }

            // Parse ELSE
            let else_result = match &case.else_value {
                Some(expr) => Some(expr_to_arg_source(expr, col_index_map_lower)?),
                None => None,
            };

            Some(ExprAction::Case {
                branches,
                else_result,
            })
        }

        // Analyze and pre-compile all expressions ONCE before the row loop
        let mut actions: Vec<ExprAction> = Vec::with_capacity(select_exprs.len());

        for expr in select_exprs.iter() {
            let action = match expr {
                Expression::Star(_) => ExprAction::StarExpand,
                Expression::QualifiedStar(qs) => {
                    let mut taken = Vec::new();
                    for_each_qualified_star_column(
                        all_columns,
                        &qs.qualifier,
                        table_alias_lower,
                        |idx, _| taken.push(idx),
                    );
                    ExprAction::QualifiedStarExpand(taken)
                }
                Expression::Identifier(id) => {
                    if let Some(&idx) = col_index_map_lower.get(id.value_lower.as_str()) {
                        ExprAction::SimpleColumn(idx)
                    } else {
                        // Unknown identifier - compile as expression (might be alias)
                        let program = compile_expression(expr, all_columns)?;
                        ExprAction::Compiled(program)
                    }
                }
                Expression::QualifiedIdentifier(qid) => match qualified_index(
                    col_index_map_lower,
                    &qid.qualifier.value_lower,
                    &qid.name.value_lower,
                ) {
                    Some(idx) => ExprAction::SimpleColumn(idx),
                    None => ExprAction::Compiled(compile_expression(expr, all_columns)?),
                },
                Expression::Aliased(aliased) => {
                    // Recurse into the inner expression
                    match &*aliased.expression {
                        Expression::Identifier(id) => {
                            if let Some(&idx) = col_index_map_lower.get(id.value_lower.as_str()) {
                                ExprAction::SimpleColumn(idx)
                            } else {
                                let program = compile_expression(&aliased.expression, all_columns)?;
                                ExprAction::Compiled(program)
                            }
                        }
                        Expression::QualifiedIdentifier(qid) => match qualified_index(
                            col_index_map_lower,
                            &qid.qualifier.value_lower,
                            &qid.name.value_lower,
                        ) {
                            Some(idx) => ExprAction::SimpleColumn(idx),
                            None => ExprAction::Compiled(compile_expression(
                                &aliased.expression,
                                all_columns,
                            )?),
                        },
                        Expression::FunctionCall(func) => {
                            // Try inline COALESCE optimization
                            if let Some(action) =
                                try_build_coalesce_action(func, col_index_map_lower)
                            {
                                action
                            } else {
                                let program = compile_expression(&aliased.expression, all_columns)?;
                                ExprAction::Compiled(program)
                            }
                        }
                        Expression::Case(case) => {
                            // Try inline CASE optimization
                            if let Some(action) = try_build_case_action(case, col_index_map_lower) {
                                action
                            } else {
                                let program = compile_expression(&aliased.expression, all_columns)?;
                                ExprAction::Compiled(program)
                            }
                        }
                        _ => {
                            // Compile the inner expression (not the Aliased wrapper)
                            let program = compile_expression(&aliased.expression, all_columns)?;
                            ExprAction::Compiled(program)
                        }
                    }
                }
                Expression::FunctionCall(func) => {
                    // Try inline COALESCE optimization
                    if let Some(action) = try_build_coalesce_action(func, col_index_map_lower) {
                        action
                    } else {
                        let program = compile_expression(expr, all_columns)?;
                        ExprAction::Compiled(program)
                    }
                }
                Expression::Case(case) => {
                    // Try inline CASE optimization
                    if let Some(action) = try_build_case_action(case, col_index_map_lower) {
                        action
                    } else {
                        let program = compile_expression(expr, all_columns)?;
                        ExprAction::Compiled(program)
                    }
                }
                _ => {
                    // Complex expression - compile it
                    let program = compile_expression(expr, all_columns)?;
                    ExprAction::Compiled(program)
                }
            };
            let action = match action {
                ExprAction::Compiled(program) => ExprAction::of_program(program),
                action => action,
            };
            actions.push(action);
        }
        Ok(actions)
    }

    /// The rows `actions` project, run directly on the VM. Out of line: its
    /// row loop keeps its own registers in any caller
    #[inline(never)]
    fn run_projection_actions(
        actions: &[ExprAction],
        rows: RowVec,
        ctx: &ExecutionContext,
        projected: &mut impl RowSink,
    ) -> Result<()> {
        // Pre-fetch parameters for VM context
        let params = ctx.params();
        let transaction_id = ctx.transaction_id();

        // Create reusable VM
        let mut vm = ExprVM::new();

        // Check if all actions are SimpleColumn (common fast path)
        let all_simple_columns = actions
            .iter()
            .all(|a| matches!(a, ExprAction::SimpleColumn(_)));

        // Project rows using pre-analyzed actions
        if all_simple_columns {
            // Super fast path: just extract column indices and use take_columns
            let indices: Vec<usize> = actions
                .iter()
                .filter_map(|a| {
                    if let ExprAction::SimpleColumn(idx) = a {
                        Some(*idx)
                    } else {
                        None
                    }
                })
                .collect();

            for (id, row) in rows.into_iter() {
                projected.put(id, row.take_columns(&indices));
            }
        } else {
            // General path: mixed actions
            // Pre-compute capacity outside loop to avoid repeated len() calls
            let values_capacity = actions.len();

            for (id, row) in rows.into_iter() {
                // Use CompactVec directly instead of Vec to avoid Vec->CompactVec conversion
                let mut values = projected.storage(values_capacity);

                // Build VM context with all params in one call (faster than builder chain)
                let vm_ctx = ExecuteContext::with_common_params(&row, params, transaction_id);

                for action in actions {
                    match action {
                        ExprAction::SimpleColumn(idx) => {
                            // Direct index access with bounds check
                            if let Some(v) = row.get(*idx) {
                                values.push(v.clone());
                            } else {
                                values.push(Value::null_unknown());
                            }
                        }
                        ExprAction::StarExpand => {
                            for val in row.iter() {
                                values.push(val.clone());
                            }
                        }
                        ExprAction::QualifiedStarExpand(taken) => {
                            for &idx in taken {
                                if let Some(val) = row.get(idx) {
                                    values.push(val.clone());
                                }
                            }
                        }
                        ExprAction::Coalesce(args) => {
                            // Inline COALESCE: direct loop avoids iterator overhead
                            let mut found = false;
                            for arg in args.iter() {
                                let val: Option<&Value> = match arg {
                                    ArgSource::Column(idx) => row.get(*idx),
                                    ArgSource::Const(v) => Some(v),
                                };
                                if let Some(v) = val {
                                    if !v.is_null() {
                                        values.push(v.clone());
                                        found = true;
                                        break;
                                    }
                                }
                            }
                            if !found {
                                values.push(Value::null_unknown());
                            }
                        }
                        ExprAction::Case {
                            branches,
                            else_result,
                        } => {
                            // Inline CASE: direct loop avoids iterator overhead
                            let mut matched = false;
                            for branch in branches.iter() {
                                let col_val: Option<&Value> = match &branch.condition {
                                    CaseCondition::Equals { col_idx, .. }
                                    | CaseCondition::NotEquals { col_idx, .. }
                                    | CaseCondition::GreaterThan { col_idx, .. }
                                    | CaseCondition::GreaterOrEqual { col_idx, .. }
                                    | CaseCondition::LessThan { col_idx, .. }
                                    | CaseCondition::LessOrEqual { col_idx, .. }
                                    | CaseCondition::IsNull { col_idx } => row.get(*col_idx),
                                };
                                let cond_matches = match (&branch.condition, col_val) {
                                    (CaseCondition::Equals { value, .. }, Some(v)) => {
                                        sql_eq(v, value) == Some(true)
                                    }
                                    (CaseCondition::NotEquals { value, .. }, Some(v)) => {
                                        sql_eq(v, value) == Some(false)
                                    }
                                    (CaseCondition::GreaterThan { value, .. }, Some(v)) => {
                                        sql_compare(v, value).is_some_and(|o| o.is_gt())
                                    }
                                    (CaseCondition::GreaterOrEqual { value, .. }, Some(v)) => {
                                        sql_compare(v, value).is_some_and(|o| o.is_ge())
                                    }
                                    (CaseCondition::LessThan { value, .. }, Some(v)) => {
                                        sql_compare(v, value).is_some_and(|o| o.is_lt())
                                    }
                                    (CaseCondition::LessOrEqual { value, .. }, Some(v)) => {
                                        sql_compare(v, value).is_some_and(|o| o.is_le())
                                    }
                                    (CaseCondition::IsNull { .. }, Some(v)) => v.is_null(),
                                    (_, None) => false,
                                };
                                if cond_matches {
                                    match &branch.result {
                                        ArgSource::Column(idx) => {
                                            if let Some(v) = row.get(*idx) {
                                                values.push(v.clone());
                                            } else {
                                                values.push(Value::null_unknown());
                                            }
                                        }
                                        ArgSource::Const(v) => values.push(v.clone()),
                                    }
                                    matched = true;
                                    break;
                                }
                            }
                            if !matched {
                                // No branch matched - use ELSE or NULL
                                match else_result {
                                    Some(ArgSource::Column(idx)) => {
                                        if let Some(v) = row.get(*idx) {
                                            values.push(v.clone());
                                        } else {
                                            values.push(Value::null_unknown());
                                        }
                                    }
                                    Some(ArgSource::Const(v)) => values.push(v.clone()),
                                    None => values.push(Value::null_unknown()),
                                }
                            }
                        }
                        ExprAction::Concat(concat) => values.push(concat.value(&vm_ctx)?),
                        ExprAction::Compiled(program) => {
                            let value = vm.execute(program, &vm_ctx)?;
                            values.push(value);
                        }
                    }
                }

                projected.put(id, Row::from_compact_vec(values));
            }
        }

        Ok(())
    }

    /// Project rows including ORDER BY columns not in SELECT
    /// Returns rows with SELECT columns followed by ORDER BY columns
    /// True when an expression names a SELECT alias that no source column
    /// carries. Inside an expression a source column wins over an alias of
    /// the same name, so only a name the source lacks counts here.
    fn reads_a_select_alias(
        expr: &Expression,
        aliases: &FxHashSet<String>,
        source_columns: &crate::common::StringMap<usize>,
    ) -> bool {
        let check = |e: &Expression| Self::reads_a_select_alias(e, aliases, source_columns);
        let only_an_alias =
            |name: &str| aliases.contains(name) && !source_columns.contains_key(name);

        match expr {
            Expression::Identifier(id) => only_an_alias(id.value_lower.as_str()),
            // An alias is never qualified
            Expression::QualifiedIdentifier(_) => false,
            Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => false,
            Expression::Prefix(p) => check(&p.right),
            Expression::Infix(inf) => check(&inf.left) || check(&inf.right),
            Expression::FunctionCall(fc) => fc.arguments.iter().any(check),
            Expression::Cast(c) => check(&c.expr),
            Expression::Aliased(a) => check(&a.expression),
            Expression::Case(case) => {
                case.value.as_ref().is_some_and(|v| check(v))
                    || case
                        .when_clauses
                        .iter()
                        .any(|w| check(&w.condition) || check(&w.then_result))
                    || case.else_value.as_ref().is_some_and(|e| check(e))
            }
            Expression::Between(b) => check(&b.expr) || check(&b.lower) || check(&b.upper),
            Expression::In(i) => check(&i.left) || check(&i.right),
            Expression::List(list) => list.elements.iter().any(check),
            Expression::ExpressionList(list) => list.expressions.iter().any(check),
            Expression::Like(l) => check(&l.left) || check(&l.pattern),
            Expression::Distinct(d) => check(&d.expr),
            Expression::Window(w) => w.function.arguments.iter().any(check),
            // Anything else is left to the sort, which reads the projected row
            _ => true,
        }
    }

    /// Evaluate each select expression holding a correlated subquery per
    /// row into a column appended to the rows, and name that column where
    /// the expression stood, so a projection reads a plain column there
    fn bind_correlated_select_columns(
        &self,
        select_exprs: &[Expression],
        rows: RowVec,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<Expression>, RowVec, Vec<String>)> {
        let correlated: Vec<usize> = select_exprs
            .iter()
            .enumerate()
            .filter(|(_, expr)| Self::has_correlated_subqueries(expr))
            .map(|(i, _)| i)
            .collect();
        let bare: Vec<Expression> = correlated
            .iter()
            .map(|&i| match &select_exprs[i] {
                Expression::Aliased(aliased) => (*aliased.expression).clone(),
                expr => expr.clone(),
            })
            .collect();
        let values = self.project_rows_with_alias(
            &bare,
            RowVec::from_vec((*rows).clone()),
            all_columns,
            None,
            ctx,
            table_alias,
        )?;
        let mut columns = all_columns.to_vec();
        let mut exprs = select_exprs.to_vec();
        // A source column may carry any name, so the bound column takes the
        // first name no source column has, qualified or bare
        let taken = |name: &str, columns: &[String]| {
            columns.iter().any(|column| {
                column.eq_ignore_ascii_case(name) || bare_name(column).eq_ignore_ascii_case(name)
            })
        };
        let mut next = 0usize;
        for &i in &correlated {
            let name = loop {
                let candidate = format!("__correlated_{next}");
                next += 1;
                if !taken(&candidate, &columns) {
                    break candidate;
                }
            };
            let column = Expression::Identifier(Identifier::new(
                dummy_token(&name, TokenType::Identifier),
                name.clone(),
            ));
            exprs[i] = match &select_exprs[i] {
                Expression::Aliased(aliased) => Expression::Aliased(AliasedExpression {
                    token: aliased.token.clone(),
                    expression: Box::new(column),
                    alias: aliased.alias.clone(),
                }),
                _ => column,
            };
            columns.push(name);
        }
        let rows = rows
            .into_iter()
            .zip(values)
            .map(|((id, row), (_, extra))| {
                let mut values = row.into_values();
                values.extend(extra.into_values());
                (id, Row::from_values(values))
            })
            .collect();
        Ok((exprs, rows, columns))
    }

    #[allow(clippy::too_many_arguments)]
    fn project_rows_with_order_by(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        distinct_on: &[Expression],
        from: Option<&Expression>,
        rows: RowVec,
        all_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>, OutputLayout)> {
        let (rows, plan, extra_names, layout) = self.ordered_rows_plan(
            select_exprs,
            order_by,
            distinct_on,
            from,
            rows,
            all_columns,
            table_alias,
            ctx,
        )?;
        let projected = Self::run_ordered_projection(&plan, rows, ctx)?;
        Ok((projected, extra_names, layout))
    }

    /// The plan of [`Self::project_rows_with_order_by`], with the rows it
    /// projects: a correlated SELECT item is bound into them first
    #[allow(clippy::too_many_arguments)]
    fn ordered_rows_plan(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        distinct_on: &[Expression],
        from: Option<&Expression>,
        mut rows: RowVec,
        all_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, OrderedProjection, Vec<String>, OutputLayout)> {
        // A correlated select expression is evaluated per row first, and
        // read below as the column it was bound to
        let mut bound_exprs = None;
        let mut bound_columns = None;
        if Self::has_correlated_select_subqueries(select_exprs) {
            let (exprs, bound_rows, columns) = self.bind_correlated_select_columns(
                select_exprs,
                rows,
                all_columns,
                table_alias,
                ctx,
            )?;
            bound_exprs = Some(exprs);
            rows = bound_rows;
            bound_columns = Some(CompactArc::new(columns));
        }
        let select_exprs: &[Expression] = bound_exprs.as_deref().unwrap_or(select_exprs);
        let shared_columns = bound_columns.as_ref().unwrap_or(all_columns);

        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        let processed = self.try_process_select_subqueries(select_exprs, ctx)?;
        let select_exprs = match &processed {
            Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
            None => std::borrow::Cow::Borrowed(select_exprs),
        };

        let (plan, extra_names, layout) = self.ordered_projection(
            &select_exprs,
            order_by,
            distinct_on,
            from,
            shared_columns,
            table_alias,
            ctx,
        );
        Ok((rows, plan, extra_names, layout))
    }

    /// What a projection keeping the ORDER BY and DISTINCT ON values the
    /// SELECT list leaves out reads per row, the names it appends, and
    /// where each output column comes from
    #[allow(clippy::too_many_arguments)]
    fn ordered_projection(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        distinct_on: &[Expression],
        from: Option<&Expression>,
        shared_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> (OrderedProjection, Vec<String>, OutputLayout) {
        let all_columns: &[String] = shared_columns;
        let OrderByProjection {
            col_index_map_lower,
            select_column_indices,
            extra_order_indices,
            computed_order_by,
            computed_distinct_on,
            extra_names,
            extra_sources,
        } = self.order_by_projection(
            select_exprs,
            order_by,
            distinct_on,
            from,
            all_columns,
            false,
        );
        let mut layout = Self::shared_select_layout(
            select_exprs,
            CompactArc::clone(shared_columns),
            table_alias,
        );
        layout.sources.extend(extra_sources);

        // Check if we can use fast path (all simple column refs, no computed DISTINCT ON)
        let all_simple = select_column_indices.iter().all(|idx| idx.is_some())
            && computed_distinct_on.is_empty()
            && computed_order_by.is_empty();

        if all_simple {
            let columns = OrderedProjection::Columns {
                select: select_column_indices,
                extra: extra_order_indices,
            };
            return (columns, extra_names, layout);
        }

        // Each value's source is found once for the statement; an error
        // is raised by the first row that reads it, as evaluating raised it
        // A statement run for an outer row reads that row's columns by name
        let outer_columns = outer_column_names(ctx);
        let mut compile_ctx = CompileContext::new(all_columns, &self.function_registry);
        if let Some(outer) = &outer_columns {
            compile_ctx = compile_ctx.with_outer_columns(outer);
        }
        let compile = |expr: &Expression| -> Result<SharedProgram> {
            ExprCompiler::new(&compile_ctx)
                .compile(expr)
                .map(CompactArc::new)
                .map_err(|e| Error::internal(format!("Compile error: {}", e)))
        };
        let mut sources: Vec<Result<ItemSource>> = Vec::with_capacity(select_exprs.len());
        for expr in select_exprs {
            match expr {
                Expression::Star(_) => {
                    sources.extend((0..all_columns.len()).map(|i| Ok(ItemSource::Column(i))))
                }
                Expression::QualifiedStar(star) => for_each_qualified_star_column(
                    all_columns,
                    &star.qualifier,
                    table_alias,
                    |i, _| sources.push(Ok(ItemSource::Column(i))),
                ),
                _ => sources.push(match select_column(expr, &col_index_map_lower) {
                    Some(idx) => idx.map(ItemSource::Column),
                    None => compile(unaliased_item(expr)).map(ItemSource::Program),
                }),
            }
        }
        sources.extend(
            extra_order_indices
                .iter()
                .map(|&idx| Ok(ItemSource::Column(idx))),
        );
        sources.extend(
            computed_order_by
                .iter()
                .chain(&computed_distinct_on)
                .map(|expr| compile(expr).map(ItemSource::Program)),
        );
        (OrderedProjection::Items(sources), extra_names, layout)
    }

    /// The output `stmt` builds from joined rows of `all_columns`, ordered
    /// already when `sorted`, its subqueries run by the statement's
    /// `blocks`; none for a SELECT list holding a subquery the blocks do
    /// not run, whose values each execution finds again
    fn join_output(
        &self,
        stmt: &SelectStatement,
        all_columns: &[String],
        sorted: bool,
        blocks: Option<&JoinBlocks>,
        ctx: &ExecutionContext,
    ) -> Result<Option<JoinOutput>> {
        if let Some(blocks) = blocks.filter(|blocks| blocks.runs_output()) {
            debug_assert_eq!(
                blocks.all.len(),
                all_columns.len(),
                "joined rows of {all_columns:?} for blocks over another layout"
            );
            return self
                .blocks_output(stmt, all_columns, sorted, blocks)
                .map(Some);
        }
        if stmt.columns.iter().any(Self::has_subqueries) {
            return Ok(None);
        }
        let mut columns = self.get_output_column_names(&stmt.columns, all_columns, None);
        if !sorted && self.order_by_needs_extra_columns(stmt, all_columns) {
            let (plan, extra_names, layout) = self.ordered_projection(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                stmt.table_expr.as_deref(),
                &CompactArc::new(all_columns.to_vec()),
                None,
                ctx,
            );
            columns.extend(extra_names);
            return Ok(Some(JoinOutput {
                projection: JoinProjection::General(OutputProjection::Ordered(plan)),
                columns: CompactArc::new(columns),
                layout: Some(CompactArc::new(layout)),
                visible: None,
            }));
        }
        let plan = self.select_projection(&stmt.columns, all_columns, None, None)?;
        Ok(Some(JoinOutput {
            projection: JoinProjection::General(OutputProjection::Select(plan)),
            columns: CompactArc::new(columns),
            layout: Self::output_layout(stmt, all_columns, None).map(CompactArc::new),
            visible: None,
        }))
    }

    /// The output of a join whose SELECT list or ORDER BY holds subqueries
    /// the statement's `blocks` run, projected as a prepared statement's is
    fn blocks_output(
        &self,
        stmt: &SelectStatement,
        all_columns: &[String],
        sorted: bool,
        blocks: &JoinBlocks,
    ) -> Result<JoinOutput> {
        // The rows hold each merged pair apart: a bare name reads its left
        // column, and a star shows that one alone, as the general path does
        let named = merged_names(all_columns, blocks.merges());
        let shown = blocks.star_columns();
        let star = Some((&shown[..], &named[..]));
        let (projection, extra_names, extra_sources) = PreparedProjection::prepare_joined(
            self,
            stmt,
            (&named, all_columns),
            &shown,
            sorted,
            blocks,
        )?;
        let mut columns = Self::output_columns(&stmt.columns, all_columns, None, star, None);
        let layout = || {
            let mut sources = LayoutSources::new();
            Self::output_columns(&stmt.columns, all_columns, None, star, Some(&mut sources));
            OutputLayout {
                input_columns: CompactArc::new(named.clone()),
                sources,
            }
        };
        if extra_names.is_empty() {
            let keyed = !stmt.order_by.is_empty() || !stmt.distinct_on.is_empty();
            return Ok(JoinOutput {
                projection: JoinProjection::Prepared(projection),
                columns: CompactArc::new(columns),
                layout: keyed.then(|| CompactArc::new(layout())),
                visible: None,
            });
        }
        let visible = columns.len();
        let mut layout = layout();
        layout.sources.extend(extra_sources);
        columns.extend(extra_names);
        Ok(JoinOutput {
            projection: JoinProjection::Prepared(projection),
            columns: CompactArc::new(columns),
            layout: Some(CompactArc::new(layout)),
            visible: Some(visible),
        })
    }

    /// The output the statement's plan holds for joined rows of
    /// `all_columns`, its subqueries run by `blocks`; none without a plan
    fn held_join_output(
        &self,
        prepared: Option<&PreparedJoin>,
        stmt: &SelectStatement,
        all_columns: &[String],
        sorted: bool,
        blocks: Option<&JoinBlocks>,
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<Option<JoinOutput>>>> {
        prepared
            .map(|prepared| {
                prepared.output(all_columns, sorted, || {
                    self.join_output(stmt, all_columns, sorted, blocks, ctx)
                })
            })
            .transpose()
    }

    /// The result `output` builds from joined rows, its subqueries answered
    /// by the statement's `run`; `applied` when the join applied the LIMIT
    /// and OFFSET
    fn join_output_result(
        output: &JoinOutput,
        rows: RowVec,
        applied: bool,
        ctx: &ExecutionContext,
        run: Option<&super::prepared_select::PreparedRunner<'_>>,
    ) -> SelectResult {
        let rows = match &output.projection {
            JoinProjection::General(OutputProjection::Select(plan)) => {
                Self::run_select_projection(plan, rows, ctx)?
            }
            JoinProjection::General(OutputProjection::Ordered(plan)) => {
                Self::run_ordered_projection(plan, rows, ctx)?
            }
            JoinProjection::Prepared(projection) => {
                let run =
                    run.ok_or_else(|| Error::internal("a join's blocks output without a run"))?;
                projection.project(rows, run)?
            }
        };
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&output.columns), rows);
        let shape = output.layout.as_ref().map(|layout| SelectShape {
            deferred: None,
            layout: Some(LaidOut::Shared(CompactArc::clone(layout))),
            visible: output.visible,
        });
        Ok((
            Box::new(result),
            CompactArc::clone(&output.columns),
            applied,
            shape,
        ))
    }

    /// The rows `plan` projects, each with its position as its id
    fn run_ordered_projection(
        plan: &OrderedProjection,
        rows: RowVec,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        let mut projected = RowVec::with_capacity(rows.len());
        Self::project_ordered_into(plan, rows, ctx, &mut projected)?;
        Ok(projected)
    }

    /// The rows `plan` projects, put into `projected` with their positions
    fn project_ordered_into(
        plan: &OrderedProjection,
        mut rows: RowVec,
        ctx: &ExecutionContext,
        projected: &mut impl RowSink,
    ) -> Result<()> {
        let params = ctx.params();
        match plan {
            OrderedProjection::Columns { select, extra } => {
                for (row_id, row) in rows.drain_rows().enumerate() {
                    let mut values = projected.storage(select.len() + extra.len()).into_vec();
                    for &idx in select.iter().flatten().chain(extra) {
                        values.push(row.get(idx).cloned().unwrap_or(Value::null_unknown()));
                    }
                    projected.put(row_id as i64, Row::from_values(values));
                }
            }
            OrderedProjection::Items(sources) => {
                // Slow path: Use Evaluator for complex expressions
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::general_path_reached(
                    crate::test_failpoints::GeneralPath::ProjectionEvaluator,
                );
                let bound = bound_items(
                    sources,
                    params,
                    |source| match source {
                        Ok(ItemSource::Program(program)) => Some(program),
                        _ => None,
                    },
                    |program| Ok(ItemSource::Program(program)),
                );
                let sources = bound.as_deref().unwrap_or(sources);
                Self::project_items_into(sources, rows, ctx, params, projected)?;
            }
        }
        Ok(())
    }

    /// The rows `sources` project, read with `params`, put into `projected`
    /// with their positions
    fn project_items_into(
        sources: &[Result<ItemSource>],
        mut rows: RowVec,
        ctx: &ExecutionContext,
        params: &[Value],
        projected: &mut impl RowSink,
    ) -> Result<()> {
        let mut vm = ExprVM::new();
        let (transaction_id, outer) = (ctx.transaction_id(), ctx.outer_row());
        for (row_id, row) in rows.drain_rows().enumerate() {
            let mut exec = ExecuteContext::with_common_params(&row, params, transaction_id);
            if let Some(outer) = outer {
                exec = exec.with_outer_row(outer);
            }
            let mut values = projected.storage(sources.len());
            for source in sources {
                values.push(match source {
                    Ok(ItemSource::Column(idx)) => {
                        row.get(*idx).cloned().unwrap_or(Value::null_unknown())
                    }
                    Ok(ItemSource::Program(program)) => vm.execute(program, &exec)?,
                    Err(e) => return Err(e.clone()),
                });
            }
            projected.put(row_id as i64, Row::from_compact_vec(values));
        }
        Ok(())
    }

    /// Which columns an ORDER BY or DISTINCT ON projection reads by index,
    /// which it computes, and the names of what it appends after the SELECT;
    /// a key holding a subquery is computed too with `computes_subqueries`
    pub(super) fn order_by_projection<'e>(
        &self,
        select_exprs: &'e [Expression],
        order_by: &'e [crate::parser::ast::OrderByExpression],
        distinct_on: &'e [Expression],
        from: Option<&Expression>,
        all_columns: &[String],
        computes_subqueries: bool,
    ) -> OrderByProjection<'e> {
        // Build column index map ONCE with FxHashMap for O(1) lookup
        let col_index_map_lower = build_column_index_map(all_columns);
        // The names below are read for the keys only
        let keyed = !order_by.is_empty() || !distinct_on.is_empty();

        // Get SELECT column names (lowercase) for checking duplicates
        let select_column_names: Vec<String> = select_exprs
            .iter()
            .filter(|_| keyed)
            .flat_map(|expr| {
                let mut names = Vec::new();
                // Add unqualified name
                if let Some(name) = self.extract_select_column_name(expr) {
                    names.push(name.to_lowercase());
                }
                // Also add fully qualified name for disambiguation
                match expr {
                    Expression::QualifiedIdentifier(qi) => {
                        names.push(qualified(&qi.qualifier.value_lower, &qi.name.value_lower));
                    }
                    Expression::Aliased(a) => {
                        if let Expression::QualifiedIdentifier(qi) = &*a.expression {
                            names.push(qualified(&qi.qualifier.value_lower, &qi.name.value_lower));
                        }
                    }
                    _ => {}
                }
                names
            })
            .collect();

        // The names the projection alone brings into scope
        let select_aliases: FxHashSet<String> = select_exprs
            .iter()
            .filter(|_| keyed)
            .filter_map(|e| match e {
                Expression::Aliased(a) => Some(a.alias.value_lower.to_string()),
                _ => None,
            })
            .collect();

        // Find ORDER BY and DISTINCT ON columns not in SELECT
        let mut extra_order_indices: Vec<usize> = Vec::new();
        // A key expression is read from a column of its own, so the sort
        // and DISTINCT ON have something to read once the projection has run
        let mut computed_order_by: Vec<&Expression> = Vec::new();
        let mut computed_distinct_on: Vec<&Expression> = Vec::new();
        let keys = order_by
            .iter()
            .map(|ob| (&ob.expression, false))
            .chain(distinct_on.iter().map(|expr| (expr, true)));
        for (expr, distinct) in keys {
            match expr {
                Expression::Identifier(id)
                    if !select_column_names
                        .iter()
                        .any(|s| s == id.value_lower.as_str()) =>
                {
                    if let Some(&idx) = col_index_map_lower.get(id.value_lower.as_str()) {
                        if !extra_order_indices.contains(&idx) {
                            extra_order_indices.push(idx);
                        }
                    }
                }
                Expression::QualifiedIdentifier(qi) if reads_input(from, qi) => {
                    let full_name = qualified(&qi.qualifier.value_lower, &qi.name.value_lower);
                    // Only match against full qualified name in SELECT to avoid ambiguity
                    if !select_column_names.contains(&full_name) {
                        let idx_opt = qualified_index(
                            &col_index_map_lower,
                            &qi.qualifier.value_lower,
                            &qi.name.value_lower,
                        );
                        if let Some(idx) = idx_opt {
                            if !extra_order_indices.contains(&idx) {
                                extra_order_indices.push(idx);
                            }
                        }
                    }
                }
                // A subquery is read per row against the parent, which the
                // sort does for itself, so it is not read from a column here.
                // Neither is an expression naming an alias, which the row
                // only carries once the projection has run
                other
                    if !matches!(
                        other,
                        Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                    ) && match Self::has_subqueries(other) {
                        // The blocks' family keeps out a key naming an alias
                        true => computes_subqueries,
                        false => !Self::reads_a_select_alias(
                            other,
                            &select_aliases,
                            &col_index_map_lower,
                        ),
                    } =>
                {
                    let key = ExprKey::of(other);
                    if !select_exprs
                        .iter()
                        .any(|item| ExprKey::of(unaliased_item(item)) == key)
                        && !computed_order_by
                            .iter()
                            .chain(&computed_distinct_on)
                            .any(|e| ExprKey::of(e) == key)
                    {
                        match distinct {
                            true => computed_distinct_on.push(other),
                            false => computed_order_by.push(other),
                        }
                    }
                }
                // An expression the sort evaluates carries the input columns
                // it names, which the sort reads by the layout
                other if Self::has_subqueries(other) => {
                    visit_expression_inputs(other, from, &mut |_, key| {
                        if let OrderKey::Input(qualifier, name) = key {
                            if let Some(idx) = input_column_index(all_columns, qualifier, name) {
                                if !extra_order_indices.contains(&idx) {
                                    extra_order_indices.push(idx);
                                }
                            }
                        }
                    });
                }
                _ => {}
            }
        }

        // A correlated subquery in a sort key reads the parent row, so every
        // column rides along until the sort has run
        if !computes_subqueries
            && order_by
                .iter()
                .map(|ob| &ob.expression)
                .chain(distinct_on)
                .any(Self::has_correlated_subqueries)
        {
            for idx in 0..all_columns.len() {
                if !extra_order_indices.contains(&idx) {
                    extra_order_indices.push(idx);
                }
            }
        }

        // Build column indices for SELECT expressions
        let mut select_column_indices: Vec<Option<usize>> = Vec::with_capacity(select_exprs.len());
        for expr in select_exprs.iter() {
            match expr {
                Expression::Identifier(id) => {
                    select_column_indices
                        .push(col_index_map_lower.get(id.value_lower.as_str()).copied());
                }
                Expression::QualifiedIdentifier(qid) => {
                    select_column_indices.push(qualified_index(
                        &col_index_map_lower,
                        &qid.qualifier.value_lower,
                        &qid.name.value_lower,
                    ));
                }
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => {
                        select_column_indices
                            .push(col_index_map_lower.get(id.value_lower.as_str()).copied());
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        select_column_indices.push(qualified_index(
                            &col_index_map_lower,
                            &qid.qualifier.value_lower,
                            &qid.name.value_lower,
                        ));
                    }
                    _ => select_column_indices.push(None),
                },
                _ => select_column_indices.push(None),
            }
        }

        // The names of the columns appended below, in the order they are
        // appended, so the caller can put them on the end of its own list
        let mut extra_names: Vec<String> = extra_order_indices
            .iter()
            .map(|idx| all_columns[*idx].clone())
            .collect();
        extra_names.extend(
            computed_order_by
                .iter()
                .chain(&computed_distinct_on)
                .map(|expr| ExprKey::of(expr).hidden_column_name()),
        );
        let computed = computed_order_by.len() + computed_distinct_on.len();
        let extra_sources = extra_order_indices
            .iter()
            .map(|&idx| Some(idx))
            .chain(std::iter::repeat_n(None, computed))
            .map(|input| ColumnSource { item: None, input })
            .collect();

        OrderByProjection {
            col_index_map_lower,
            select_column_indices,
            extra_order_indices,
            computed_order_by,
            computed_distinct_on,
            extra_names,
            extra_sources,
        }
    }

    /// Evaluate a single SELECT expression
    /// OPTIMIZATION: Takes pre-computed column index map for O(1) lookup instead of O(N) linear search
    fn evaluate_select_expr(
        &self,
        evaluator: &mut CompiledEvaluator,
        expr: &Expression,
        row: &Row,
        col_index_map: &StringMap<usize>,
    ) -> Result<Value> {
        match select_column(expr, col_index_map) {
            Some(idx) => Ok(row.get(idx?).cloned().unwrap_or(Value::null_unknown())),
            None => evaluator.evaluate(unaliased_item(expr)),
        }
    }

    /// Get output column names from SELECT expressions
    pub(crate) fn get_output_column_names(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
    ) -> Vec<String> {
        Self::output_columns(select_exprs, all_columns, table_alias, None, None)
    }

    /// The layout of the output columns `select_exprs` give over the input
    /// `all_columns`, from the expansion that names them
    pub(crate) fn select_layout(
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
    ) -> OutputLayout {
        Self::shared_select_layout(
            select_exprs,
            CompactArc::new(all_columns.to_vec()),
            table_alias,
        )
    }

    /// `select_layout` over input columns a source already holds shared
    pub(crate) fn shared_select_layout(
        select_exprs: &[Expression],
        all_columns: CompactArc<Vec<String>>,
        table_alias: Option<&str>,
    ) -> OutputLayout {
        OutputLayout {
            sources: Self::select_sources(select_exprs, &all_columns, table_alias),
            input_columns: all_columns,
        }
    }

    /// The source of each output column `select_exprs` give over the input
    /// `all_columns`, without the input names a layout holds
    pub(crate) fn select_sources(
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
    ) -> LayoutSources {
        let mut sources = LayoutSources::new();
        Self::output_columns(
            select_exprs,
            all_columns,
            table_alias,
            None,
            Some(&mut sources),
        );
        sources
    }

    /// The layout of `stmt`'s output columns over the input `all_columns`,
    /// for its ORDER BY and DISTINCT ON to read; None without either
    pub(crate) fn output_layout(
        stmt: &SelectStatement,
        all_columns: &[String],
        table_alias: Option<&str>,
    ) -> Option<OutputLayout> {
        (!stmt.order_by.is_empty() || !stmt.distinct_on.is_empty())
            .then(|| Self::select_layout(&stmt.columns, all_columns, table_alias))
    }

    /// `output_layout` over input columns a source already holds shared
    pub(crate) fn shared_output_layout(
        stmt: &SelectStatement,
        all_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
    ) -> Option<OutputLayout> {
        (!stmt.order_by.is_empty() || !stmt.distinct_on.is_empty()).then(|| {
            Self::shared_select_layout(&stmt.columns, CompactArc::clone(all_columns), table_alias)
        })
    }

    /// The layout of a grouped or windowed result of `stmt`: its first
    /// columns those of the SELECT list over `all_columns`, the rest appended
    /// for ORDER BY, a hidden key computing an expression and any other
    /// column carrying the input column it is named after. None without an
    /// ORDER BY or a DISTINCT ON
    pub(crate) fn result_layout(
        stmt: &SelectStatement,
        all_columns: &[String],
        table_alias: Option<&str>,
        result_columns: &[String],
    ) -> Option<OutputLayout> {
        let layout = Self::output_layout(stmt, all_columns, table_alias)?;
        Some(Self::appended_layout(layout, all_columns, result_columns))
    }

    /// `result_layout` over input columns a source already holds shared
    pub(crate) fn shared_result_layout(
        stmt: &SelectStatement,
        all_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
        result_columns: &[String],
    ) -> Option<OutputLayout> {
        let layout = Self::shared_output_layout(stmt, all_columns, table_alias)?;
        Some(Self::appended_layout(layout, all_columns, result_columns))
    }

    /// `layout` with a source for each of `result_columns` past its own
    fn appended_layout(
        mut layout: OutputLayout,
        all_columns: &[String],
        result_columns: &[String],
    ) -> OutputLayout {
        let appended = result_columns
            .get(layout.sources.len()..)
            .unwrap_or_default();
        for name in appended {
            let input = match ExprKey::from_hidden_column_name(name) {
                Some(_) => None,
                None => match split_qualified(&folded(name)) {
                    Some((qualifier, bare)) => {
                        input_column_index(all_columns, Some(qualifier), bare)
                    }
                    None => input_column_index(all_columns, None, name),
                },
            };
            layout.sources.push(ColumnSource { item: None, input });
        }
        layout
    }

    /// The names of the output columns `select_exprs` give over the input
    /// `all_columns`, a star expanded, and into `sources` the source of each.
    /// With `star`, a `*` shows the columns at its positions, by its names
    fn output_columns(
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
        star: Option<(&[usize], &[String])>,
        mut sources: Option<&mut LayoutSources>,
    ) -> Vec<String> {
        // A layout reads the sources alone: no name is built for it
        let laid_out = sources.is_some();
        let mut names = Vec::with_capacity(if laid_out { 0 } else { select_exprs.len() });
        let mut source = |item: usize, input: Option<usize>| {
            if let Some(sources) = sources.as_deref_mut() {
                sources.push(ColumnSource {
                    item: Some(item),
                    input,
                });
            }
        };

        for (i, expr) in select_exprs.iter().enumerate() {
            match expr {
                Expression::Star(_) => {
                    // SELECT * - add all column names
                    match star {
                        Some((shown, named)) => {
                            if !laid_out {
                                names.extend(shown.iter().map(|&at| named[at].clone()));
                            }
                            shown.iter().for_each(|&at| source(i, Some(at)));
                        }
                        None => {
                            if !laid_out {
                                names.extend(all_columns.iter().cloned());
                            }
                            (0..all_columns.len()).for_each(|idx| source(i, Some(idx)));
                        }
                    }
                    continue;
                }
                Expression::QualifiedStar(qs) => {
                    // SELECT t.*: the columns t names, each under its bare name
                    for_each_qualified_star_column(
                        all_columns,
                        &qs.qualifier,
                        table_alias,
                        |idx, start| {
                            if !laid_out {
                                names.push(all_columns[idx][start..].to_string());
                            }
                            source(i, Some(idx));
                        },
                    );
                    continue;
                }
                _ => {}
            }
            // A plain column, aliased or not, is a copy of its input column
            if laid_out {
                let copied = match expr {
                    Expression::Aliased(aliased) => aliased.expression.as_ref(),
                    other => other,
                };
                let input = match copied {
                    Expression::Identifier(id) => {
                        input_column_index(all_columns, None, &id.value_lower)
                    }
                    Expression::QualifiedIdentifier(qid) => input_column_index(
                        all_columns,
                        Some(&qid.qualifier.value_lower),
                        &qid.name.value_lower,
                    ),
                    _ => None,
                };
                source(i, input);
                continue;
            }
            names.push(item_name(expr));
        }

        names
    }

    /// Each ORDER BY item's column among `columns`, the rows `layout` laid
    /// out, None when no column holds it; and the aggregate columns a sort
    /// key may read by identity
    pub(crate) fn order_by_columns(
        &self,
        stmt: &SelectStatement,
        columns: &[String],
        layout: Option<&OutputLayout>,
        set_operation: bool,
    ) -> (Vec<Option<usize>>, FxHashMap<ExprKey, usize>) {
        let keys = stmt.order_by.iter().map(|ob| &ob.expression);
        self.key_columns(stmt, keys, columns, layout, set_operation)
    }

    /// `order_by_columns` for any of `stmt`'s sort keys
    fn key_columns<'k>(
        &self,
        stmt: &SelectStatement,
        keys: impl Iterator<Item = &'k Expression>,
        columns: &[String],
        layout: Option<&OutputLayout>,
        set_operation: bool,
    ) -> (Vec<Option<usize>>, FxHashMap<ExprKey, usize>) {
        // Each column by the identity of what computed it: the SELECT
        // list in order (unless a star moves the positions), then the
        // aggregates carried only for this sort
        let mut column_slots: FxHashMap<ExprKey, usize> = FxHashMap::default();
        // Only aggregates stand in for themselves inside a sort key: a
        // literal or a name there is read as written
        let mut aggregate_slots: FxHashMap<ExprKey, usize> = FxHashMap::default();
        let star = stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)));
        for (i, col) in stmt.columns.iter().enumerate() {
            // No sort key names a star
            if matches!(col, Expression::Star(_) | Expression::QualifiedStar(_)) {
                continue;
            }
            if set_operation && !set_operation_key_item(col, &self.function_registry) {
                continue;
            }
            // An item's column by the layout; without one, by its index
            // while no star moves the positions
            let Some(position) = (match layout {
                Some(layout) => layout.item_position(i),
                None => (!star).then_some(i),
            }) else {
                continue;
            };
            let key = ExprKey::of(col);
            column_slots.entry(key).or_insert(position);
            let inner = match col {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                other => other,
            };
            if !set_operation
                && matches!(inner, Expression::FunctionCall(f) if self.function_registry.is_aggregate(&f.function))
            {
                aggregate_slots.entry(key).or_insert(position);
            }
        }
        for (i, name) in columns.iter().enumerate() {
            if let Some(key) = ExprKey::from_hidden_column_name(name) {
                column_slots.entry(key).or_insert(i);
                aggregate_slots.entry(key).or_insert(i);
            }
        }

        // An expression an item computes, or one the projection carried in a
        // column named after its identity, never an output name spelled
        // like it
        let find_column_for =
            |expr: &Expression| -> Option<usize> { column_slots.get(&ExprKey::of(expr)).copied() };

        // Each ORDER BY item by its binding: an ordinal by position, a
        // plain name by the column the layout says holds what it reads,
        // a name the input lacks or anything else by a column computing it
        let try_map_to_column = |expr: &Expression| -> Option<usize> {
            let key = bind_order_key(expr, stmt);
            match (&key, layout) {
                (OrderKey::Ordinal(position), _) => Some(*position),
                (OrderKey::Expression, _) => find_column_for(expr),
                (_, Some(layout)) => layout.position(&key).or_else(|| find_column_for(expr)),
                // A set operation's rows carry only output names
                (OrderKey::Item(_) | OrderKey::Input(..), None) => {
                    debug_assert!(
                        set_operation,
                        "rows reached ORDER BY {expr} without a layout"
                    );
                    let name = match expr {
                        Expression::QualifiedIdentifier(qid) => &qid.name.value_lower,
                        Expression::Identifier(id) => &id.value_lower,
                        _ => return None,
                    };
                    columns.iter().position(|c| c.eq_ignore_ascii_case(name))
                }
            }
        };
        let mapped = keys.map(try_map_to_column).collect();
        (mapped, aggregate_slots)
    }

    /// The top-N the statement's ORDER BY LIMIT keeps of projected rows of
    /// `columns`, when its sort would read them by these columns alone, as
    /// execute_select sorts them; None when the sort comes later or reads more
    fn projected_top(
        &self,
        stmt: &SelectStatement,
        columns: &[String],
        layout: Option<&OutputLayout>,
        ctx: &ExecutionContext,
    ) -> Result<Option<TopN<impl Fn(&Row, &Row) -> Ordering>>> {
        if stmt.order_by.is_empty()
            || stmt.distinct
            || !stmt.distinct_on.is_empty()
            || !stmt.set_operations.is_empty()
        {
            return Ok(None);
        }
        let (
            Some(layout),
            RowCounts {
                limit: Some(limit),
                offset,
            },
        ) = (layout, row_counts(stmt, ctx)?)
        else {
            return Ok(None);
        };
        let (mapped, _) = self.order_by_columns(stmt, columns, Some(layout), false);
        let order_specs = stmt
            .order_by
            .iter()
            .zip(mapped)
            .map(|(ob, mapped)| mapped.map(|c| (Some(c), ob.ascending, ob.nulls_first)))
            .collect::<Option<Vec<_>>>();
        Ok(order_specs.map(|order_specs| {
            TopN::new(
                move |a: &Row, b: &Row| Self::compare_rows_with_indices(a, b, &order_specs),
                limit,
                offset,
            )
        }))
    }

    /// Compare two rows for ORDER BY using pre-computed column indices
    /// This avoids string allocations during sort comparisons
    #[inline]
    pub(crate) fn compare_rows_with_indices(
        a: &Row,
        b: &Row,
        order_specs: &[(Option<usize>, bool, Option<bool>)],
    ) -> Ordering {
        for (col_idx, ascending, nulls_first) in order_specs {
            if let Some(idx) = col_idx {
                let cmp = crate::core::value::compare_order_key(
                    a.get(*idx),
                    b.get(*idx),
                    *ascending,
                    *nulls_first,
                );
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
        }

        Ordering::Equal
    }

    /// Execute BEGIN statement
    /// Note: Nested BEGIN is a no-op when a transaction is already active
    pub(crate) fn execute_begin(
        &self,
        stmt: &BeginStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use super::ActiveTransaction;

        let mut active_tx = self.active_transaction.lock().unwrap();

        // If there's already an active transaction, treat as no-op
        if active_tx.is_some() {
            return Ok(Box::new(ExecResult::empty()));
        }

        // Start a new transaction, with isolation level if specified
        let transaction = if let Some(ref level) = stmt.isolation_level {
            let isolation = Self::parse_isolation_level(level)?;
            self.engine.begin_transaction_with_level(isolation)?
        } else {
            self.begin_transaction()?
        };

        *active_tx = Some(ActiveTransaction {
            transaction,
            tables: FxHashMap::default(),
        });

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute COMMIT statement
    pub(crate) fn execute_commit_stmt(
        &self,
        _stmt: &CommitStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(mut tx_state) = active_tx.take() {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            tx_state.transaction.commit()?;
            Ok(Box::new(ExecResult::empty()))
        } else {
            // No active transaction - this is a no-op (auto-commit mode)
            Ok(Box::new(ExecResult::empty()))
        }
    }

    /// Execute ROLLBACK statement
    pub(crate) fn execute_rollback_stmt(
        &self,
        stmt: &RollbackStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        // Check if this is a ROLLBACK TO SAVEPOINT
        if let Some(ref savepoint_name) = stmt.savepoint_name {
            if let Some(ref mut tx_state) = *active_tx {
                tx_state
                    .transaction
                    .rollback_to_savepoint(&savepoint_name.value)?;

                Ok(Box::new(ExecResult::empty()))
            } else {
                Err(Error::internal(
                    "ROLLBACK TO SAVEPOINT can only be used within a transaction",
                ))
            }
        } else {
            drop(active_tx);
            self.rollback_active()?;
            Ok(Box::new(ExecResult::empty()))
        }
    }

    /// Rolls back and ends the open transaction, its tables first; a no-op
    /// when none is open
    pub(crate) fn rollback_active(&self) -> Result<()> {
        let taken = self
            .active_transaction
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(mut tx_state) = taken {
            for (_name, mut table) in tx_state.tables.drain() {
                table.rollback();
            }
            tx_state.transaction.rollback()?;
        }
        Ok(())
    }

    /// Execute SET statement
    pub(crate) fn execute_set(
        &self,
        stmt: &SetStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let name = stmt.name.value.to_uppercase();

        match name.as_str() {
            "ISOLATION_LEVEL" | "TRANSACTION_ISOLATION" => {
                // Extract the value as a string
                let level_str = match &stmt.value {
                    Expression::StringLiteral(lit) => lit.value.to_uppercase(),
                    Expression::Identifier(id) => id.value.to_uppercase(),
                    _ => {
                        return Err(Error::internal(
                            "SET isolation_level requires a string value (e.g., 'READ COMMITTED', 'SNAPSHOT')",
                        ));
                    }
                };

                let isolation = Self::parse_isolation_level(&level_str)?;

                // If there's an active transaction, set on the transaction
                let mut active_tx = self.active_transaction.lock().unwrap();
                if let Some(ref mut tx_state) = *active_tx {
                    tx_state.transaction.set_isolation_level(isolation)?;
                } else {
                    self.store_isolation_level(isolation);
                }

                Ok(Box::new(ExecResult::empty()))
            }
            _ => {
                // Silently accept unknown SET variables for compatibility
                Ok(Box::new(ExecResult::empty()))
            }
        }
    }

    /// Parse an isolation level string into an IsolationLevel enum
    fn parse_isolation_level(level: &str) -> Result<crate::core::IsolationLevel> {
        use crate::core::IsolationLevel;
        match level {
            "READ COMMITTED" => Ok(IsolationLevel::ReadCommitted),
            "SNAPSHOT" | "SERIALIZABLE" | "REPEATABLE READ" => {
                Ok(IsolationLevel::SnapshotIsolation)
            }
            "READ UNCOMMITTED" => {
                // Treat as READ COMMITTED (closest supported level)
                Ok(IsolationLevel::ReadCommitted)
            }
            _ => Err(Error::internal(format!(
                "unsupported isolation level: '{}'. Supported: READ COMMITTED, SNAPSHOT, REPEATABLE READ, SERIALIZABLE",
                level
            ))),
        }
    }

    /// Execute PRAGMA statement
    pub(crate) fn execute_pragma(
        &self,
        stmt: &PragmaStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let pragma_name = stmt.name.value.to_uppercase();

        match pragma_name.as_str() {
            "SNAPSHOT" => {
                // PRAGMA SNAPSHOT: Create a full backup snapshot of all tables.
                // Writes .bin files to snapshots/ directory. keep_snapshots limits retention.
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA SNAPSHOT does not accept values"));
                }

                {
                    let active_tx = self.active_transaction.lock().unwrap();
                    if active_tx.is_some() {
                        return Err(Error::internal(
                            "PRAGMA SNAPSHOT cannot run inside a transaction. \
                             Commit or rollback first.",
                        ));
                    }
                }

                self.engine.create_snapshot()?;

                let columns = vec!["result".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::text("Snapshot created successfully")]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "CHECKPOINT" => {
                // PRAGMA CHECKPOINT: Run the checkpoint cycle (seal hot to volumes,
                // persist manifests, compact, WAL truncate).
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA CHECKPOINT does not accept values"));
                }

                {
                    let active_tx = self.active_transaction.lock().unwrap();
                    if active_tx.is_some() {
                        return Err(Error::internal(
                            "PRAGMA CHECKPOINT cannot run inside a transaction. \
                             Commit or rollback first.",
                        ));
                    }
                }

                self.engine.force_checkpoint_cycle()?;

                let columns = vec!["result".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::text("Checkpoint completed successfully")]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "RESTORE" => {
                // PRAGMA RESTORE: Restore database from latest backup snapshot.
                // PRAGMA RESTORE = 'YYYYMMDD-HHMMSS.fff': Restore from specific snapshot.
                {
                    let active_tx = self.active_transaction.lock().unwrap();
                    if active_tx.is_some() {
                        return Err(Error::internal(
                            "PRAGMA RESTORE cannot run inside a transaction. \
                             Commit or rollback first.",
                        ));
                    }
                }

                let timestamp = if let Some(ref value) = stmt.value {
                    Some(self.extract_pragma_string_value(value)?)
                } else {
                    None
                };

                let result_msg = self.engine.restore_snapshot(timestamp.as_deref())?;

                // Clear all query caches since all data has changed
                self.semantic_cache.clear();
                self.query_cache.clear();
                crate::executor::context::clear_scalar_subquery_cache();
                crate::executor::context::clear_in_subquery_cache();
                crate::executor::context::clear_semi_join_cache();
                crate::executor::context::clear_block_answer_cache();

                let columns = vec!["result".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((0, Row::from_values(vec![Value::text(&result_msg)])));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "DEDUP_SEGMENTS" => {
                let columns = vec!["message".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::text(
                        "Dedup segments is no longer needed (handled automatically)",
                    )]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "SNAPSHOT_INTERVAL" | "CHECKPOINT_INTERVAL" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    // Set mode: PRAGMA checkpoint_interval = 60
                    let new_value = self.extract_pragma_int_value(value)?;
                    if new_value < 0 {
                        return Err(Error::internal("checkpoint_interval must be non-negative"));
                    }
                    let mut new_config = config.clone();
                    new_config.persistence.checkpoint_interval = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    // Read mode: PRAGMA checkpoint_interval
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.checkpoint_interval as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "COMPACT_THRESHOLD" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    if new_value < 0 {
                        return Err(Error::internal("compact_threshold must be non-negative"));
                    }
                    let mut new_config = config.clone();
                    new_config.persistence.compact_threshold = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.compact_threshold as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "TARGET_VOLUME_ROWS" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    if new_value < 65536 {
                        return Err(Error::internal("target_volume_rows must be at least 65536"));
                    }
                    let mut new_config = config.clone();
                    new_config.persistence.target_volume_rows = new_value as usize;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.target_volume_rows as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "HOT_MAX_ROWS" | "HOT_MAX_BYTES" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];
                let is_rows = pragma_name == "HOT_MAX_ROWS";

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    if new_value < 0 {
                        return Err(Error::internal(format!(
                            "{} must be non-negative",
                            pragma_name.to_lowercase()
                        )));
                    }
                    let mut new_config = config.clone();
                    if is_rows {
                        new_config.persistence.hot_max_rows = new_value as usize;
                    } else {
                        new_config.persistence.hot_max_bytes = new_value as usize;
                    }
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let current = if is_rows {
                        config.persistence.hot_max_rows
                    } else {
                        config.persistence.hot_max_bytes
                    };
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(current as i64)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "MEMORY_STATS" => {
                if stmt.value.is_some() {
                    return Err(Error::internal(
                        "PRAGMA MEMORY_STATS does not accept values",
                    ));
                }

                let columns = vec![
                    "table_name".to_string(),
                    "hot_rows".to_string(),
                    "hot_bytes".to_string(),
                    "arena_slots".to_string(),
                    "arena_capacity_bytes".to_string(),
                    "chain_entries".to_string(),
                    "volume_bytes".to_string(),
                    "admission_waits".to_string(),
                ];

                let stats = self.engine.memory_stats();
                let mut rows = RowVec::with_capacity(stats.len());
                for (i, stat) in stats.into_iter().enumerate() {
                    rows.push((
                        i as i64,
                        Row::from_values(vec![
                            Value::text(&stat.table_name),
                            Value::Integer(stat.hot_rows as i64),
                            Value::Integer(stat.hot_bytes as i64),
                            Value::Integer(stat.arena_slots as i64),
                            Value::Integer(stat.arena_capacity_bytes as i64),
                            Value::Integer(stat.chain_entries as i64),
                            Value::Integer(stat.volume_bytes as i64),
                            Value::Integer(stat.admission_waits as i64),
                        ]),
                    ));
                }
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "SYNC_MODE" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if stmt.value.is_some() {
                    return Err(Error::NotSupported(
                        "sync_mode cannot be changed at runtime. Set it in the connection string: file:///path?sync_mode=none|normal|full".to_string(),
                    ));
                }
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::Integer(config.persistence.sync_mode as i64)]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "WAL_FLUSH_TRIGGER" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if stmt.value.is_some() {
                    return Err(Error::NotSupported(
                        "wal_flush_trigger cannot be changed at runtime. Set it in the connection string: file:///path?wal_flush_trigger=N".to_string(),
                    ));
                }
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::Integer(
                        config.persistence.wal_flush_trigger as i64,
                    )]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "KEEP_SNAPSHOTS" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    if new_value < 0 {
                        return Err(Error::internal("keep_snapshots must be non-negative"));
                    }
                    let mut new_config = config.clone();
                    new_config.persistence.keep_snapshots = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.keep_snapshots as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "HEAP_PROFILE" => {
                // Writes the heap profile, in gzipped pprof format, to the
                // file named; a build with the heap-profile feature only
                let Some(ref value) = stmt.value else {
                    return Err(Error::internal("PRAGMA HEAP_PROFILE needs a file path"));
                };
                let path = self.extract_pragma_string_value(value)?;
                let bytes = crate::heap_profile::write_heap_profile(&path)?;
                let mut rows = RowVec::with_capacity(1);
                rows.push((0, Row::from_values(vec![Value::Integer(bytes as i64)])));
                Ok(Box::new(ExecutorResult::new(
                    vec!["heap_profile_bytes".to_string()],
                    rows,
                )))
            }
            "GROUP_CACHE_MB" => {
                // Budget of the decoded row-group column cache, in megabytes;
                // zero disables it
                use crate::storage::volume::group_cache::DECODED_GROUPS;
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];
                if let Some(ref value) = stmt.value {
                    let mb = self.extract_pragma_int_value(value)?;
                    if mb < 0 {
                        return Err(Error::internal("group_cache_mb must not be negative"));
                    }
                    DECODED_GROUPS.set_budget_bytes((mb as usize).saturating_mul(1024 * 1024));
                }
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::Integer(
                        (DECODED_GROUPS.budget_bytes() / (1024 * 1024)) as i64,
                    )]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "INDEX_CACHE_MB" | "INDEX_BUILD_MB" => {
                // Budgets of the secondary index side files, in megabytes:
                // the pages readers hold and cache, and the workspaces of
                // the builds in flight; a request over a budget is refused
                use crate::storage::volume::secondary::{INDEX_BUILDS, INDEX_PAGES};
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];
                let pages = pragma_name == "INDEX_CACHE_MB";
                if let Some(ref value) = stmt.value {
                    let mb = self.extract_pragma_int_value(value)?;
                    if mb < 0 {
                        return Err(Error::internal(format!(
                            "{} must not be negative",
                            pragma_name.to_lowercase()
                        )));
                    }
                    let bytes = (mb as u64).saturating_mul(1024 * 1024);
                    if pages {
                        INDEX_PAGES.set_budget_bytes(bytes);
                    } else {
                        INDEX_BUILDS.set_budget_bytes(bytes);
                    }
                }
                let budget = if pages {
                    INDEX_PAGES.stats().budget_bytes
                } else {
                    INDEX_BUILDS.stats().budget_bytes
                };
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::Integer((budget / (1024 * 1024)) as i64)]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "INDEX_BACKFILL" => {
                // PRAGMA INDEX_BACKFILL [= n]: builds side files for up to n
                // volumes the query path cannot serve, all of them by default
                let limit = match stmt.value {
                    Some(ref value) => {
                        let n = self.extract_pragma_int_value(value)?;
                        if n < 0 {
                            return Err(Error::internal("index_backfill must not be negative"));
                        }
                        n as usize
                    }
                    None => usize::MAX,
                };
                let report = self.engine.backfill_side_files(limit)?;
                let columns: Vec<String> = [
                    "examined",
                    "built",
                    "refused",
                    "discarded",
                    "failed",
                    "left",
                ]
                .iter()
                .map(|c| c.to_string())
                .collect();
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![
                        Value::Integer(report.examined as i64),
                        Value::Integer(report.built as i64),
                        Value::Integer(report.refused as i64),
                        Value::Integer(report.discarded as i64),
                        Value::Integer(report.failed as i64),
                        Value::Integer(report.left as i64),
                    ]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "INDEX_READ_STATS" => {
                // One row: what the reads through side files did since the
                // process started
                if stmt.value.is_some() {
                    return Err(Error::internal(
                        "PRAGMA INDEX_READ_STATS does not accept values",
                    ));
                }
                let snapshot = crate::storage::volume::secondary::READS.snapshot();
                let columns: Vec<String> =
                    snapshot.iter().map(|(name, _)| name.to_string()).collect();
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(
                        snapshot
                            .iter()
                            .map(|(_, value)| Value::Integer(*value as i64))
                            .collect(),
                    ),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "INDEX_STATS" => {
                // One row per ledger of the secondary index side files
                use crate::storage::volume::secondary::{
                    BUILDS_FAILED, INDEX_BUILDS, INDEX_DIRECTORIES, INDEX_PAGES, SIDES_DISCARDED,
                };
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA INDEX_STATS does not accept values"));
                }
                let columns = vec![
                    "ledger".to_string(),
                    "budget_bytes".to_string(),
                    "charged_bytes".to_string(),
                    "peak_bytes".to_string(),
                    "refused".to_string(),
                    "cached_pages".to_string(),
                    "loads".to_string(),
                    "hits".to_string(),
                    "evictions".to_string(),
                    "over_budget".to_string(),
                    "builds_failed".to_string(),
                    "sides_discarded".to_string(),
                ];
                let pages = INDEX_PAGES.stats();
                let builds = INDEX_BUILDS.stats();
                let directories = INDEX_DIRECTORIES.stats();
                let ledger = |name: &str,
                              budget: u64,
                              charged: usize,
                              peak: usize,
                              refused: u64,
                              rest: [i64; 7]| {
                    let mut values = vec![
                        Value::text(name),
                        Value::Integer(budget as i64),
                        Value::Integer(charged as i64),
                        Value::Integer(peak as i64),
                        Value::Integer(refused as i64),
                    ];
                    values.extend(rest.into_iter().map(Value::Integer));
                    Row::from_values(values)
                };
                let mut rows = RowVec::with_capacity(3);
                rows.push((
                    0,
                    ledger(
                        "index_cache",
                        pages.budget_bytes,
                        pages.charged_bytes,
                        pages.peak_bytes,
                        pages.refused,
                        [
                            pages.cached_pages as i64,
                            pages.loads as i64,
                            pages.hits as i64,
                            pages.evictions as i64,
                            pages.over_budget as i64,
                            0,
                            0,
                        ],
                    ),
                ));
                rows.push((
                    1,
                    ledger(
                        "index_builds",
                        builds.budget_bytes,
                        builds.charged_bytes,
                        builds.peak_bytes,
                        builds.refused,
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            BUILDS_FAILED.load(std::sync::atomic::Ordering::Relaxed) as i64,
                            SIDES_DISCARDED.load(std::sync::atomic::Ordering::Relaxed) as i64,
                        ],
                    ),
                ));
                rows.push((
                    2,
                    ledger(
                        "index_directories",
                        directories.budget_bytes,
                        directories.charged_bytes,
                        directories.peak_bytes,
                        directories.refused,
                        [0; 7],
                    ),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "GROUP_CACHE_STATS" => {
                if stmt.value.is_some() {
                    return Err(Error::internal(
                        "PRAGMA GROUP_CACHE_STATS does not accept values",
                    ));
                }
                let stats = crate::storage::volume::group_cache::DECODED_GROUPS.stats();
                let columns = vec![
                    "budget_bytes".to_string(),
                    "bytes".to_string(),
                    "entries".to_string(),
                    "hits".to_string(),
                    "misses".to_string(),
                ];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![
                        Value::Integer(stats.budget_bytes as i64),
                        Value::Integer(stats.bytes as i64),
                        Value::Integer(stats.entries as i64),
                        Value::Integer(stats.hits as i64),
                        Value::Integer(stats.misses as i64),
                    ]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "VOLUME_STATS" => {
                if stmt.value.is_some() {
                    return Err(Error::internal(
                        "PRAGMA VOLUME_STATS does not accept values",
                    ));
                }

                let columns = vec![
                    "table_name".to_string(),
                    "segment_id".to_string(),
                    "tier".to_string(),
                    "row_count".to_string(),
                    "memory_bytes".to_string(),
                    "idle_cycles".to_string(),
                    "tombstones".to_string(),
                ];

                let stats = self.engine.volume_stats();
                let mut rows = RowVec::with_capacity(stats.len());
                for (i, (table, seg_id, tier, row_count, mem, idle, ts)) in
                    stats.into_iter().enumerate()
                {
                    rows.push((
                        i as i64,
                        Row::from_values(vec![
                            Value::text(&table),
                            Value::Integer(seg_id as i64),
                            Value::text(tier),
                            Value::Integer(row_count as i64),
                            Value::Integer(mem as i64),
                            Value::Integer(idle as i64),
                            Value::Integer(ts as i64),
                        ]),
                    ));
                }
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "VACUUM" => {
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA VACUUM does not accept values"));
                }
                // Delegate to the unified VACUUM implementation (no table filter)
                let vacuum_stmt = crate::parser::ast::VacuumStatement {
                    token: stmt.token.clone(),
                    table_name: None,
                };
                self.execute_vacuum(&vacuum_stmt, ctx)
            }
            _ => {
                // Unknown pragma - return empty result for compatibility
                Ok(Box::new(ExecResult::empty()))
            }
        }
    }

    /// Execute VACUUM statement — manual cleanup of deleted rows and index compaction
    ///
    /// VACUUM aggressively reclaims storage by removing deleted rows, pruning old
    /// version chains, and cleaning up stale transaction metadata. Unlike background
    /// cleanup (which uses a 5-minute retention), VACUUM uses zero retention so all
    /// historical versions not needed by active transactions are removed.
    ///
    /// Safety: `cleanup_deleted_rows` and `cleanup_old_previous_versions_with_retention`
    /// check active transaction visibility before removing any version, so concurrent
    /// readers are never disrupted. However, AS OF TIMESTAMP queries that reference
    /// timestamps before the VACUUM will no longer work — this is intentional.
    pub(crate) fn execute_vacuum(
        &self,
        stmt: &VacuumStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // VACUUM must not run inside an explicit transaction
        let active_tx = self.active_transaction.lock().unwrap();
        if active_tx.is_some() {
            return Err(Error::internal(
                "VACUUM cannot be executed within an active transaction",
            ));
        }
        drop(active_tx);

        let retention = std::time::Duration::ZERO;
        let table_name = stmt.table_name.as_ref().map(|id| id.value_lower.as_str());

        let cleaned = self.engine.vacuum(table_name, retention)?;

        let columns = vec![
            "deleted_rows_cleaned".to_string(),
            "old_versions_cleaned".to_string(),
            "transactions_cleaned".to_string(),
        ];
        let mut rows = RowVec::with_capacity(1);
        rows.push((
            0,
            Row::from_values(vec![
                Value::Integer(cleaned.0 as i64),
                Value::Integer(cleaned.1 as i64),
                Value::Integer(cleaned.2 as i64),
            ]),
        ));
        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Extract integer value from PRAGMA value expression
    fn extract_pragma_int_value(&self, value: &crate::parser::Expression) -> Result<i64> {
        match value {
            crate::parser::Expression::IntegerLiteral(lit) => Ok(lit.value),
            crate::parser::Expression::FloatLiteral(lit) => Ok(lit.value as i64),
            _ => Err(Error::internal("PRAGMA value must be an integer")),
        }
    }

    fn extract_pragma_string_value(&self, value: &crate::parser::Expression) -> Result<String> {
        match value {
            crate::parser::Expression::StringLiteral(lit) => Ok(lit.value.to_string()),
            _ => Err(Error::internal(
                "PRAGMA value must be a quoted string, e.g. 'YYYYMMDD-HHMMSS.fff'",
            )),
        }
    }

    /// Execute SAVEPOINT statement
    pub(crate) fn execute_savepoint(
        &self,
        stmt: &SavepointStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(ref mut tx_state) = *active_tx {
            // Create savepoint in the transaction
            tx_state
                .transaction
                .create_savepoint(&stmt.savepoint_name.value)?;
            Ok(Box::new(ExecResult::empty()))
        } else {
            // No active transaction - return error
            Err(Error::internal(
                "SAVEPOINT can only be used within a transaction (after BEGIN)",
            ))
        }
    }

    /// Execute RELEASE SAVEPOINT statement
    pub(crate) fn execute_release_savepoint(
        &self,
        stmt: &ReleaseSavepointStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(ref mut tx_state) = *active_tx {
            tx_state
                .transaction
                .release_savepoint(&stmt.savepoint_name.value)?;
            Ok(Box::new(ExecResult::empty()))
        } else {
            Err(Error::internal(
                "RELEASE SAVEPOINT can only be used within a transaction (after BEGIN)",
            ))
        }
    }

    /// Execute an expression statement (SELECT 1+1)
    pub(crate) fn execute_expression_stmt(
        &self,
        stmt: &ExpressionStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let value = super::expression::evaluate_once(&stmt.expression, Some(ctx))?;
        let columns = vec!["result".to_string()];
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, Row::from_values(vec![value])));

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Every row of table `name` read AS OF, every column in the table's
    /// order, read through the statement's transaction at the point `as_of`
    /// names under the statement's cut
    pub(crate) fn as_of_rows(
        &self,
        name: &str,
        as_of: &AsOfClause,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, RowVec)> {
        let point = self.as_of_point(as_of, ctx)?;
        let columns = self
            .engine
            .get_table_schema(name)
            .map_err(|e| match e {
                Error::TableNotFound(_) => Error::TableOrViewNotFound(name.to_string()),
                other => other,
            })?
            .column_names_owned()
            .to_vec();
        let active = self
            .active_transaction
            .lock()
            .map_err(|_| Error::internal("active transaction lock poisoned"))?;
        let rows = match (active.as_ref(), ctx.statement_snapshot()) {
            (Some(tx), _) => tx.transaction.rows_at(name, &point)?,
            (None, Some(snapshot)) => {
                drop(active);
                snapshot.rows_at(name, &point)?
            }
            (None, None) => {
                drop(active);
                self.begin_transaction()?.rows_at(name, &point)?
            }
        };
        Ok((columns, rows))
    }

    /// The error for AS OF on `name`, a view or a CTE: AS OF reads a table
    pub(crate) fn as_of_not_a_table(name: &str) -> Error {
        Error::invalid_argument(format!(
            "AS OF reads a table, and '{name}' is a view or a CTE"
        ))
    }

    /// The point an AS OF value names, resolved against the statement's cut
    fn as_of_point(
        &self,
        as_of: &AsOfClause,
        ctx: &ExecutionContext,
    ) -> Result<crate::storage::mvcc::commit_log::ReadPoint> {
        let value = super::expression::evaluate_once(&as_of.value, Some(ctx))?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::as_of_resolving();
        let registry = self.engine.registry();
        // A statement entered without one, through a caller's own plan, takes its own
        let cut = ctx
            .historical_cut()
            .unwrap_or_else(|| registry.historical_cut());
        match as_of.as_of_type.to_uppercase().as_str() {
            "TRANSACTION" => match value {
                Value::Integer(txn_id) => registry.point_of_transaction(txn_id, cut),
                _ => Err(Error::invalid_argument(
                    "AS OF TRANSACTION requires integer value",
                )),
            },
            "TIMESTAMP" => {
                // A time outside the TIMESTAMP range is an error, not an empty answer
                let nanos = match value {
                    Value::Timestamp(ts) => crate::core::value::timestamp_nanos(&ts)?,
                    Value::Text(s) => {
                        crate::core::value::timestamp_nanos(&crate::core::parse_timestamp(&s)?)?
                    }
                    // Nanoseconds since the epoch
                    Value::Integer(i) => i,
                    _ => {
                        return Err(Error::invalid_argument(
                            "AS OF TIMESTAMP requires timestamp or string value",
                        ))
                    }
                };
                registry.point_at_time(nanos, cut)
            }
            _ => Err(Error::invalid_argument(format!(
                "Unsupported AS OF type: {}",
                as_of.as_of_type
            ))),
        }
    }

    /// The WHERE a table scan reads `stmt`'s rows by when resolving the
    /// SELECT aliases of `aliases` it names, then simplifying, changes it;
    /// None when the scan reads the WHERE as written. Inlined into the table
    /// scan, it costs that scan's loop more per row
    #[inline(never)]
    pub(crate) fn scan_where(
        stmt: &SelectStatement,
        aliases: &FxHashMap<String, &Expression>,
    ) -> Option<Box<Expression>> {
        let where_clause = stmt.where_clause.as_deref()?;
        let resolved =
            (!aliases.is_empty()).then(|| Self::substitute_aliases(where_clause, aliases));
        let read = resolved.as_ref().unwrap_or(where_clause);
        ExpressionSimplifier::new()
            .try_simplify(read)
            .or(resolved)
            .map(Box::new)
    }

    /// Build a map of column aliases to their underlying expressions from SELECT columns
    /// Build alias map, optionally excluding aliases that shadow real table columns.
    ///
    /// In SQL, WHERE is evaluated before SELECT, so WHERE references table columns,
    /// not SELECT aliases. When an alias name matches a real column name, the column
    /// must take priority in WHERE context. Pass `base_columns` to exclude such aliases.
    pub(super) fn build_alias_map_excluding<'a>(
        columns: &'a [Expression],
        base_columns: Option<&[String]>,
    ) -> FxHashMap<String, &'a Expression> {
        // Count aliases first to pre-size HashMap and avoid rehashing
        let alias_count = columns
            .iter()
            .filter(|e| matches!(e, Expression::Aliased(_)))
            .count();

        if alias_count == 0 {
            return FxHashMap::default();
        }

        let mut alias_map = FxHashMap::with_capacity_and_hasher(alias_count, Default::default());
        for col_expr in columns {
            if let Expression::Aliased(aliased) = col_expr {
                let alias_name = aliased.alias.value_lower.to_string();

                // Skip aliases that shadow real table column names.
                // In standard SQL, WHERE references FROM columns, not SELECT aliases.
                // When alias "time" shadows column "time", WHERE time >= '...' must
                // reference the column, not the aliased expression.
                if let Some(base_cols) = base_columns {
                    // Check both fully qualified ("t.time") and unqualified ("time") names.
                    // Join columns are qualified (e.g., "t1.time"), but aliases are
                    // unqualified (e.g., "time"), so we must also compare against
                    // the part after the dot.
                    if base_cols.iter().any(|c| {
                        c.eq_ignore_ascii_case(&alias_name)
                            || bare_name(c).eq_ignore_ascii_case(&alias_name)
                    }) {
                        continue;
                    }
                }

                // Store reference instead of clone - avoids expensive Expression clone/drop
                alias_map.insert(alias_name, aliased.expression.as_ref());
            }
        }
        alias_map
    }

    /// A predicate over the join's `columns` with the SELECT aliases it names
    /// spelled as their expressions, None when there is none to spell; an
    /// alias a column of the join holds stays that column
    fn with_select_aliases(
        stmt: &SelectStatement,
        predicate: &Expression,
        columns: &[String],
    ) -> Option<Expression> {
        let alias_map = Self::build_alias_map_excluding(&stmt.columns, Some(columns));
        (!alias_map.is_empty()).then(|| Self::substitute_aliases(predicate, &alias_map))
    }

    /// Binds a WHERE that keeps no row over the `columns` it would filter,
    /// as the filter of their rows would, without reading one: a column it
    /// cannot find is the error the filter gives
    fn bind_empty_where(
        stmt: &SelectStatement,
        where_clause: &Expression,
        columns: &[String],
    ) -> Result<()> {
        let resolved = Self::with_select_aliases(stmt, where_clause, columns);
        RowFilter::new(resolved.as_ref().unwrap_or(where_clause), columns).map(|_| ())
    }

    /// Substitute column aliases in an expression with their underlying expressions
    pub(super) fn substitute_aliases(
        expr: &Expression,
        alias_map: &FxHashMap<String, &Expression>,
    ) -> Expression {
        match expr {
            Expression::Identifier(id) => {
                // Check if this identifier is an alias - use pre-computed lowercase
                if let Some(original) = alias_map.get(id.value_lower.as_str()) {
                    // Clone only when we actually find an alias match
                    return (*original).clone();
                }
                expr.clone()
            }
            Expression::Infix(infix) => Expression::Infix(InfixExpression {
                token: infix.token.clone(),
                left: Box::new(Self::substitute_aliases(&infix.left, alias_map)),
                operator: infix.operator.clone(),
                op_type: infix.op_type,
                right: Box::new(Self::substitute_aliases(&infix.right, alias_map)),
            }),
            Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression {
                token: prefix.token.clone(),
                operator: prefix.operator.clone(),
                op_type: prefix.op_type,
                right: Box::new(Self::substitute_aliases(&prefix.right, alias_map)),
            }),
            Expression::Between(between) => Expression::Between(BetweenExpression {
                token: between.token.clone(),
                expr: Box::new(Self::substitute_aliases(&between.expr, alias_map)),
                lower: Box::new(Self::substitute_aliases(&between.lower, alias_map)),
                upper: Box::new(Self::substitute_aliases(&between.upper, alias_map)),
                not: between.not,
            }),
            Expression::In(in_expr) => Expression::In(InExpression {
                token: in_expr.token.clone(),
                left: Box::new(Self::substitute_aliases(&in_expr.left, alias_map)),
                right: Box::new(Self::substitute_aliases(&in_expr.right, alias_map)),
                not: in_expr.not,
            }),
            Expression::AllAny(all_any) => Expression::AllAny(AllAnyExpression {
                token: all_any.token.clone(),
                left: Box::new(Self::substitute_aliases(&all_any.left, alias_map)),
                operator: all_any.operator.clone(),
                all_any_type: all_any.all_any_type,
                subquery: all_any.subquery.clone(),
            }),
            Expression::FunctionCall(func) => Expression::FunctionCall(Box::new(FunctionCall {
                token: func.token.clone(),
                function: func.function.clone(),
                arguments: func
                    .arguments
                    .iter()
                    .map(|e| Self::substitute_aliases(e, alias_map))
                    .collect(),
                is_distinct: func.is_distinct,
                order_by: func.order_by.clone(),
                filter: func.filter.clone(),
            })),
            Expression::Case(case) => Expression::Case(Box::new(CaseExpression {
                token: case.token.clone(),
                value: case
                    .value
                    .as_ref()
                    .map(|e| Box::new(Self::substitute_aliases(e, alias_map))),
                when_clauses: case
                    .when_clauses
                    .iter()
                    .map(|wc| WhenClause {
                        token: wc.token.clone(),
                        condition: Self::substitute_aliases(&wc.condition, alias_map),
                        then_result: Self::substitute_aliases(&wc.then_result, alias_map),
                    })
                    .collect(),
                else_value: case
                    .else_value
                    .as_ref()
                    .map(|e| Box::new(Self::substitute_aliases(e, alias_map))),
            })),
            Expression::List(list) => Expression::List(Box::new(ListExpression {
                token: list.token.clone(),
                elements: list
                    .elements
                    .iter()
                    .map(|e| Self::substitute_aliases(e, alias_map))
                    .collect(),
            })),
            Expression::Like(like) => Expression::Like(LikeExpression {
                token: like.token.clone(),
                left: Box::new(Self::substitute_aliases(&like.left, alias_map)),
                pattern: Box::new(Self::substitute_aliases(&like.pattern, alias_map)),
                operator: like.operator.clone(),
                escape: like
                    .escape
                    .as_ref()
                    .map(|e| Box::new(Self::substitute_aliases(e, alias_map))),
            }),
            // For other expression types, return as-is
            _ => expr.clone(),
        }
    }

    /// LIMIT and OFFSET of a GROUP BY + LIMIT join as (initial left chunk, rows
    /// expected after OFFSET). The first chunk is doubled for an INNER JOIN,
    /// where a left row without a match forms no group, and doubled again
    /// under HAVING, which drops groups after the fact. The counts are read
    /// with `ctx`'s parameters; without `ctx` only a written count is read
    fn semijoin_caps(
        stmt: &SelectStatement,
        is_inner: bool,
        ctx: Option<&ExecutionContext>,
    ) -> Option<(usize, usize, usize, usize)> {
        let counts = match ctx {
            Some(ctx) => row_counts(stmt, ctx).ok()?,
            None => static_row_counts(stmt)?,
        };
        let (limit, offset) = (counts.limit.filter(|limit| *limit > 0)?, counts.offset);
        let want = limit.saturating_add(offset);
        let mut cap = want;
        if is_inner {
            cap = cap.saturating_mul(2);
        }
        if stmt.having.is_some() {
            cap = cap.saturating_mul(2);
        }
        // The chunk becomes a LIMIT literal, which is an i64
        Some((cap.min(i64::MAX as usize), want, offset, limit))
    }

    /// The reduction a run of a plan with reduction `keys` takes: the first
    /// left chunk is the limit, doubled for an INNER JOIN since some left
    /// rows may not have matches and doubled again under HAVING; the join
    /// reruns with twice the chunk when the result stops short. `ctx` gives
    /// the statement's parameters when it runs; EXPLAIN gives none
    fn semijoin_reduction(
        stmt: &SelectStatement,
        join_type: &str,
        (left_key_col, right_key_col): &(String, String),
        ctx: Option<&ExecutionContext>,
    ) -> Option<SemijoinReduction> {
        let (cap, want, offset, limit) = Self::semijoin_caps(stmt, join_type == "INNER", ctx)?;
        Some(SemijoinReduction {
            cap,
            want,
            offset,
            limit,
            left_key_col: left_key_col.clone(),
            right_key_col: right_key_col.clone(),
        })
    }

    /// The limit the executor pushes into a join of `stmt`: the statement's
    /// LIMIT plus OFFSET when nothing orders, groups or aggregates the joined
    /// rows and both evaluate; None when the join runs unbounded. Without a
    /// context, as EXPLAIN asks, only literal counts are read
    pub(crate) fn pushed_join_limit(
        &self,
        stmt: &SelectStatement,
        ctx: Option<&ExecutionContext>,
        classification: &QueryClassification,
        join_type: &str,
    ) -> Option<u64> {
        // A window function and DISTINCT read every joined row before the
        // LIMIT counts any
        if join_type.contains("FULL")
            || classification.has_order_by
            || classification.has_group_by
            || classification.has_aggregation
            || classification.has_window_functions
            || stmt.distinct
            || !stmt.distinct_on.is_empty()
        {
            return None;
        }
        // OFFSET rows are skipped after collection, so early termination
        // must gather LIMIT + OFFSET rows
        let rows = match ctx {
            Some(ctx) => rows_needed(stmt, ctx)?,
            None => {
                let counts = static_row_counts(stmt)?;
                counts.limit?.saturating_add(counts.offset)
            }
        };
        Some(rows as u64)
    }

    /// Whether the grouped index join takes the statement's shape: an INNER
    /// join, or one whose right side carries no filter; plain aggregates the
    /// registry knows, without expressions, FILTER, ORDER BY or hidden
    /// ones; a plain GROUP BY over columns; no subquery in the filters.
    /// The executor and EXPLAIN ask the same question; the executor passes
    /// `ctx`, whose parameters the calls it runs read
    pub(super) fn grouped_index_join_applies(
        &self,
        stmt: &SelectStatement,
        join_type: &str,
        join_condition: Option<&Expression>,
        right_filter_present: bool,
        filters_have_subquery: bool,
        ctx: Option<&ExecutionContext>,
    ) -> Result<
        Option<(
            Vec<crate::executor::aggregation::SqlAggregateFunction>,
            Vec<String>,
        )>,
    > {
        // A right-side filter on a LEFT JOIN changes what an unmatched left row
        // means; the reduction handles that shape
        if join_type != "INNER" && right_filter_present {
            return Ok(None);
        }
        // A HAVING that reads the group it is asked about has to be run once
        // per group with that group as the outer row, which the general
        // aggregation does and this streaming path does not
        // The ON clause compiles into the residual, where a subquery has no executor
        if stmt.group_by.modifier != crate::parser::ast::GroupByModifier::None
            || filters_have_subquery
            || join_condition.is_some_and(Self::filter_has_subquery)
            || stmt
                .having
                .as_deref()
                .is_some_and(Self::has_correlated_subqueries)
        {
            return Ok(None);
        }
        let (aggregations, _) = self.parse_aggregations(stmt, ctx)?;
        // The operator reads columns and keeps each call's DISTINCT and
        // extra arguments in its state
        let supported = crate::executor::aggregation::Modifiers::DISTINCT
            .and(crate::executor::aggregation::Modifiers::EXTRA_ARGS);
        if aggregations.iter().any(|agg| {
            !agg.answered_with(supported)
                || agg.hidden
                || self.function_registry.get_aggregate(&agg.name).is_none()
        }) {
            return Ok(None);
        }
        // The group columns, which the operator then groups by: parsed once
        // here for both the shape and the run
        let mut group_names = Vec::new();
        for item in self.parse_group_by(stmt, &[])? {
            match item {
                crate::executor::aggregation::GroupByItem::Column(name) => group_names.push(name),
                _ => return Ok(None),
            }
        }
        Ok(Some((aggregations, group_names)))
    }

    /// Which index join operator, if any, a join in the FROM clause of
    /// `stmt` runs on: the grouped index join when the reduction plan bounds
    /// the join and its shape is the grouped operator's, no index join for
    /// the other shapes of that plan, else the streaming or batch index join,
    /// bounded when the executor pushes the limit. EXPLAIN asks this so it
    /// reports the strategy execution would choose; `blocks` when the join
    /// runs its subqueries as prepared blocks.
    pub(crate) fn join_index_use(&self, stmt: &SelectStatement, blocks: bool) -> JoinIndexUse {
        let Some(Expression::JoinSource(join)) = stmt.table_expr.as_deref() else {
            return JoinIndexUse::None;
        };
        let classification = get_classification(stmt);
        let join_type = join.join_type.to_uppercase();
        // A WHERE that reaches into the outer query outside the blocks keeps
        // both operators out
        if classification.where_has_correlated_subqueries && !blocks {
            return JoinIndexUse::None;
        }
        let ctes = super::explain::cte_names(stmt);
        let schemas = [&join.left, &join.right]
            .map(|side| self.base_table_side_schema(side, CteNames::Listed(&ctes)));
        let rows = [0, 1].map(|side| blocks && schemas[side].is_none());
        let aliases = [&join.left, &join.right].map(|side| get_table_alias_from_expr(side));
        let sides = self.unrun_join_sides(join, stmt.with.as_ref());
        let (left, right, cross) = match stmt.where_clause.as_deref() {
            Some(where_clause) => split_join_where(
                where_clause,
                SplitSide::both(
                    sides.as_ref(),
                    [aliases[0].as_deref(), aliases[1].as_deref()],
                    [schemas[0].as_deref(), schemas[1].as_deref()],
                ),
                &join_type,
            ),
            None => (None, None, None),
        };
        let shape = JoinShape {
            stmt,
            classification: &classification,
            join,
            condition: join.condition.as_deref(),
            join_type: &join_type,
            aliases: [aliases[0].as_deref(), aliases[1].as_deref()],
            filters: [left.as_ref(), right.as_ref()],
            cross_filter: cross.as_ref(),
            rows_sides: rows,
            per_row_where: false,
            blocks,
            ctes: CteNames::Listed(&ctes),
        };
        let Ok(plan) = self.plan_join(&shape) else {
            return JoinIndexUse::None;
        };
        let reduced = plan
            .reduction
            .as_ref()
            .is_some_and(|keys| Self::semijoin_reduction(stmt, &join_type, keys, None).is_some());
        if !reduced {
            let bounded = self
                .pushed_join_limit(stmt, None, &classification, &join_type)
                .is_some();
            return JoinIndexUse::Streaming { bounded, rows };
        }
        match plan.grouped_index {
            true => JoinIndexUse::Grouped,
            false => JoinIndexUse::None,
        }
    }

    /// The schema of a join side that reads a base table directly; None for
    /// a CTE, a view, a subquery or any other source
    pub(crate) fn base_table_side_schema(
        &self,
        expr: &Expression,
        ctes: CteNames<'_>,
    ) -> Option<CompactArc<Schema>> {
        let ts = match expr {
            Expression::TableSource(ts) => ts,
            Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                Expression::TableSource(ts) => ts,
                _ => return None,
            },
            _ => return None,
        };
        if ctes.hides(&ts.name.value_lower) {
            return None;
        }
        self.engine.get_table_schema(&ts.name.value_lower).ok()
    }

    /// Build an IN filter expression: column_name IN (v1, v2, ..., vN)
    pub(crate) fn build_in_filter_expression(
        &self,
        column_name: &str,
        values: &[Value],
    ) -> Option<Expression> {
        // A timestamp member compiled into a filter over a derived side reads
        // as TEXT and matches nothing, so those keys keep the unfiltered read
        if values.is_empty() || values.iter().any(|v| matches!(v, Value::Timestamp(_))) {
            return None;
        }

        // Parse the column name to handle qualified names like "o.user_id"
        let col_expr = if let Some((qualifier, name)) = split_qualified(column_name) {
            Expression::QualifiedIdentifier(QualifiedIdentifier {
                qualifier: Identifier::new(
                    Token::new(TokenType::Identifier, qualifier, Position::default()),
                    qualifier.to_string(),
                ),
                name: Identifier::new(
                    Token::new(TokenType::Identifier, name, Position::default()),
                    name.to_string(),
                ),
            })
        } else {
            Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, column_name, Position::default()),
                column_name.to_string(),
            ))
        };
        Self::in_list_expression(col_expr, values)
    }

    /// `col_expr IN (values...)`, the values written as literals, a
    /// timestamp as a typed one to the nanosecond; None for no values, or
    /// for a value no literal here writes exactly
    fn in_list_expression(col_expr: Expression, values: &[Value]) -> Option<Expression> {
        if values.is_empty() {
            return None;
        }

        // Build list of value literals.
        // IntegerLiteral uses "0" token (Display uses self.value, not token.literal).
        // FloatLiteral must use f.to_string() (Display uses token.literal).
        let mut value_exprs: Vec<Expression> = Vec::with_capacity(values.len());
        for v in values {
            value_exprs.push(match v {
                Value::Integer(i) => Expression::IntegerLiteral(IntegerLiteral {
                    token: Token::new(TokenType::Integer, "0", Position::default()),
                    value: *i,
                }),
                Value::Float(f) => Expression::FloatLiteral(FloatLiteral {
                    token: Token::new(TokenType::Float, f.to_string(), Position::default()),
                    value: *f,
                }),
                Value::Text(s) => Expression::StringLiteral(StringLiteral {
                    token: Token::new(TokenType::String, s.as_str(), Position::default()),
                    value: s.as_str().into(),
                    type_hint: None,
                }),
                Value::Timestamp(t) => {
                    let text = t.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
                    Expression::StringLiteral(StringLiteral {
                        token: Token::new(TokenType::String, &text, Position::default()),
                        value: text.as_str().into(),
                        type_hint: Some("TIMESTAMP".into()),
                    })
                }
                Value::Boolean(b) => Expression::BooleanLiteral(BooleanLiteral {
                    token: Token::new(
                        TokenType::Keyword,
                        if *b { "true" } else { "false" },
                        Position::default(),
                    ),
                    value: *b,
                }),
                _ => return None,
            });
        }

        // Create IN expression
        Some(Expression::In(InExpression {
            token: Token::new(TokenType::Keyword, "IN", Position::default()),
            left: Box::new(col_expr),
            right: Box::new(Expression::ExpressionList(Box::new(ExpressionList {
                token: Token::new(TokenType::Punctuator, "(", Position::default()),
                expressions: value_exprs,
            }))),
            not: false,
        }))
    }

    /// Find column index by name (case-insensitive), supporting qualified names like "t.col"
    fn find_column_index_by_name(col_name: &str, columns: &[String]) -> Option<usize> {
        let col_lower = col_name.to_lowercase();
        columns.iter().position(|c| {
            let c_lower = c.to_lowercase();
            // Exact match
            if c_lower == col_lower {
                return true;
            }
            // Match qualified to qualified: "t.col" matches "t.col"
            // Match qualified to unqualified: "t.col" -> check if "col" matches
            match (split_qualified(&col_lower), split_qualified(&c_lower)) {
                // Both qualified, which the exact match above compared
                (Some(_), Some(_)) => false,
                (Some((_, unqualified)), None) => c_lower == unqualified,
                (None, Some((_, bare))) => bare == col_lower,
                (None, None) => false,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mvcc::engine::MVCCEngine;
    use std::sync::Arc;

    fn create_test_executor() -> Executor {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        Executor::new(Arc::new(engine))
    }

    #[test]
    fn test_show_tables() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            .unwrap();

        let mut result = executor.execute("SHOW TABLES").unwrap();
        assert_eq!(result.columns().len(), 1);

        let mut found = false;
        while result.next() {
            let row = result.row();
            if let Some(Value::Text(name)) = row.get(0) {
                if &**name == "test" {
                    found = true;
                }
            }
        }
        assert!(found);
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn a_statement_snapshot_is_protected_with_its_sequence() {
        let engine = Arc::new(MVCCEngine::in_memory());
        engine.open_engine().unwrap();
        let executor = Executor::new(Arc::clone(&engine));
        executor
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
            .unwrap();
        executor.execute("INSERT INTO t VALUES (1, 0)").unwrap();
        let writer = Executor::new(Arc::clone(&engine));
        crate::test_failpoints::after_transaction_begun(move || {
            writer.execute("UPDATE t SET v = 10 WHERE id = 1").unwrap();
        });
        let snapshot = executor
            .new_statement_snapshot(crate::core::IsolationLevel::SnapshotIsolation, None)
            .unwrap();
        let rows = snapshot
            .get_table("t")
            .unwrap()
            .collect_rows_by_ids(&[1])
            .unwrap();
        let seen: Vec<Value> = rows
            .iter()
            .map(|(_, row)| row.get(1).cloned().unwrap())
            .collect();
        assert_eq!(
            seen,
            vec![Value::Integer(0)],
            "the statement snapshot saw a commit made after its sequence"
        );
    }
}
