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
use crate::storage::index::id_list::GroupIds;
use crate::storage::mvcc::engine::ViewDefinition;
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::{Engine, QueryResult};

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

    /// The output column of SELECT item `item`, its first under a star
    fn item_position(&self, item: usize) -> Option<usize> {
        self.sources.iter().position(|s| s.item == Some(item))
    }

    /// The output position of what `key` reads: the column its SELECT item
    /// gives, or a column holding its input column unchanged
    fn position(&self, key: &OrderKey) -> Option<usize> {
        match *key {
            OrderKey::Item(item) => self.item_position(item),
            OrderKey::Input(qualifier, name) => {
                let input = input_column_index(&self.input_columns, qualifier, name)?;
                self.sources.iter().position(|s| s.input == Some(input))
            }
            OrderKey::Ordinal(_) | OrderKey::Expression => None,
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
        Expression::QualifiedIdentifier(qid) => {
            OrderKey::Input(Some(&qid.qualifier.value_lower), &qid.name.value_lower)
        }
        _ => OrderKey::Expression,
    }
}

/// Each input column `stmt`'s ORDER BY reads by its binding, with the
/// reference that names it: a plain name bound to its input column, and
/// every name inside an expression; false when an expression reads more
/// than its names say, as a subquery does
fn visit_order_by_inputs<'e>(
    stmt: &'e SelectStatement,
    visit: &mut dyn FnMut(&'e Expression, OrderKey<'e>),
) -> bool {
    let mut read_all = true;
    for ob in &stmt.order_by {
        match bind_order_key(&ob.expression, stmt) {
            key @ OrderKey::Input(..) => visit(&ob.expression, key),
            OrderKey::Expression => read_all &= visit_expression_inputs(&ob.expression, visit),
            OrderKey::Ordinal(_) | OrderKey::Item(_) => {}
        }
    }
    read_all
}

/// Each name inside a sort expression, bound as an input column; false as
/// `visit_column_refs` says
fn visit_expression_inputs<'e>(
    expr: &'e Expression,
    visit: &mut dyn FnMut(&'e Expression, OrderKey<'e>),
) -> bool {
    visit_column_refs(expr, &mut |column| match column {
        Expression::Identifier(id) => visit(column, OrderKey::Input(None, &id.value_lower)),
        Expression::QualifiedIdentifier(qid) => visit(
            column,
            OrderKey::Input(Some(&qid.qualifier.value_lower), &qid.name.value_lower),
        ),
        _ => {}
    })
}

/// The input columns `stmt`'s ORDER BY reads, as indexes into `columns`;
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
/// it: the qualified name, then the bare name, then a qualified column whose
/// last part is the bare name. The names come lowercased
fn input_column_index(columns: &[String], qualifier: Option<&str>, name: &str) -> Option<usize> {
    qualifier
        .and_then(|qualifier| {
            columns
                .iter()
                .position(|c| eq_folded_qualified(c, qualifier, name))
        })
        .or_else(|| columns.iter().position(|c| eq_folded(c, name)))
        .or_else(|| {
            columns.iter().position(|c| {
                c.rsplit_once('.')
                    .is_some_and(|(_, last)| eq_folded(last, name))
            })
        })
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
use super::operator::{ColumnInfo, Operator, QueryResultOperator};
use super::operators::hash_join::JoinType as OperatorJoinType;
use super::operators::index_nested_loop::{
    BatchIndexNestedLoopJoinOperator, IndexLookupStrategy, IndexNestedLoopJoinOperator,
};
use super::operators::BloomFilterOperator;
use super::parallel::{self, ParallelConfig};
use super::prepared_join::{BuiltPrograms, JoinProgramKey, PreparedJoin, WhereSite};
use super::prepared_select::{names_any_within, JoinBlocks, OutOfFamily};
use super::pushdown;
use super::query_cache::CompiledExecution;
use super::query_classification::{get_classification, QueryClassification};
use super::result::{
    DistinctOnResult, DistinctResult, ExecResult, ExecutorResult, ExprMappedResult, FilteredResult,
    LimitedResult, OrderLimitPath, OrderedResult, ProjectedResult, RadixOrderSpec, ScannerResult,
    StreamingProjectionResult, TopNResult,
};
use super::row_count::{row_counts, rows_needed, static_row_counts, RowCounts};
use super::utils::compute_join_projection;
use super::utils::{
    add_table_qualifier, build_column_index_map, collect_table_qualifiers,
    collect_unqualified_columns, column_kinds, combine_predicates_with_and, dummy_token, eq_folded,
    eq_folded_qualified, expression_contains_aggregate, extract_and_conditions,
    extract_base_column_name, extract_join_keys_and_residual, filter_references_column,
    flatten_and_predicates, flip_operator, for_each_qualified_star_column,
    get_table_alias_from_expr, index_join_residual, infix_to_operator, order_by_names_an_alias,
    star_takes_every_column, string_literal_value, strip_table_qualifier, substitute_filter_column,
    visit_column_refs,
};
use super::volatility::{expression_volatility, statement_constant};
use super::window::{WindowOutput, WindowPreGroupedState, WindowPreSortedState};
use super::Executor;
use crate::functions::{global_registry, Volatility};
use crate::optimizer::bloom::BloomFilterBuilder;
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
                let qualified_name = table_alias
                    .map(|alias| CompactArc::from(format!("{}.{}", alias, col_lower).as_str()));
                let unqualified_part = col_name.rfind('.').map(|dot_idx| {
                    CompactArc::from(col_name[dot_idx + 1..].to_lowercase().as_str())
                });
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

/// The CTE names that hide tables of the same name for a join's sides: the
/// ones execution registered, or the ones EXPLAIN reads off the WITH
#[derive(Clone, Copy)]
pub(crate) enum CteNames<'a> {
    Context(&'a ExecutionContext),
    Listed(&'a [String]),
}

impl CteNames<'_> {
    fn hides(self, name_lower: &str) -> bool {
        match self {
            CteNames::Context(ctx) => ctx.get_cte_by_lower(name_lower).is_some(),
            CteNames::Listed(names) => names.iter().any(|name| name == name_lower),
        }
    }
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
    for (left_idx, left_base) in &left_base_cols {
        for (right_idx, right_base) in &right_base_cols {
            // For NATURAL JOIN, match all common columns
            // For USING, only match specified columns
            let should_match = if !using_col_names.is_empty() {
                // USING clause - match only specified columns
                using_col_names.contains(left_base) && left_base == right_base
            } else {
                // NATURAL JOIN - match all common columns
                left_base == right_base
            };

            if should_match {
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
        }
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

/// A join's WHERE split as execution pushes it: a filter for each side it
/// may narrow, never the side an outer join pads with NULLs, and the rest,
/// asked of the joined rows. The schemas are the sides that are base tables
pub(crate) fn split_join_where(
    where_clause: &Expression,
    left_alias: Option<&str>,
    right_alias: Option<&str>,
    schemas: (Option<&Schema>, Option<&Schema>),
    join_type: &str,
) -> (Option<Expression>, Option<Expression>, Option<Expression>) {
    let (Some(left_a), Some(right_a)) = (left_alias, right_alias) else {
        return (None, None, Some(where_clause.clone()));
    };
    let (l, r, c) = partition_where_for_join(where_clause, left_a, right_a, schemas);

    // For OUTER JOINs, we can't push filters to the NULL-padded side
    // because rows that don't match need to appear with NULLs
    let can_push_left = !join_type.contains("RIGHT") && !join_type.contains("FULL");
    let can_push_right = !join_type.contains("LEFT") && !join_type.contains("FULL");

    let safe_left = if can_push_left {
        l.as_ref().map(|f| strip_table_qualifier(f, left_a))
    } else {
        None
    };

    let safe_right = if can_push_right {
        r.as_ref().map(|f| strip_table_qualifier(f, right_a))
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
/// a name no column has; None for any other expression
fn select_column(expr: &Expression, col_index_map: &StringMap<usize>) -> Option<Result<usize>> {
    match unaliased_item(expr) {
        Expression::Identifier(id) => Some(
            col_index_map
                .get(id.value_lower.as_str())
                .copied()
                .ok_or_else(|| Error::ColumnNotFound(id.value.to_string())),
        ),
        Expression::QualifiedIdentifier(qid) => {
            // The full name first: two tables in a join may share a column name
            let full_name = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
            Some(
                col_index_map
                    .get(&full_name)
                    .or_else(|| col_index_map.get(qid.name.value_lower.as_str()))
                    .copied()
                    .ok_or_else(|| {
                        Error::ColumnNotFound(format!("{}.{}", qid.qualifier.value, qid.name.value))
                    }),
            )
        }
        _ => None,
    }
}

// Argument source for inline function evaluation
#[derive(Debug)]
enum ArgSource {
    Column(usize),
    Const(Value),
}

// Simple condition for inline CASE: column comparisons
#[derive(Debug)]
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
#[derive(Debug)]
struct CaseBranch {
    condition: CaseCondition,
    result: ArgSource,
}

/// How one SELECT expression of an uncorrelated projection is evaluated
#[derive(Debug)]
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
    /// Inline string concatenation (||) - bypasses VM
    Concat(smallvec::SmallVec<[ArgSource; 6]>),
    Compiled(SharedProgram),
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
    projection: OutputProjection,
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

impl JoinOutput {
    /// This output with its member tests bound to the sets `run` read; None
    /// when no program tests a block's members
    fn bound_to(&self, run: &super::prepared_select::PreparedRunner<'_>) -> Option<JoinOutput> {
        let OutputProjection::Ordered(OrderedProjection::Items(sources)) = &self.projection else {
            return None;
        };
        let asks = |source: &Result<ItemSource>| matches!(source, Ok(ItemSource::Program(p)) if p.has_subqueries());
        if !sources.iter().any(asks) {
            return None;
        }
        let sources = sources
            .iter()
            .map(|source| match source {
                Ok(ItemSource::Program(p)) => Ok(ItemSource::Program(
                    run.bound_program(p).unwrap_or_else(|| CompactArc::clone(p)),
                )),
                Ok(ItemSource::Column(i)) => Ok(ItemSource::Column(*i)),
                Err(e) => Err(e.clone()),
            })
            .collect();
        Some(JoinOutput {
            projection: OutputProjection::Ordered(OrderedProjection::Items(sources)),
            columns: CompactArc::clone(&self.columns),
            layout: self.layout.clone(),
            visible: self.visible,
        })
    }
}

/// The columns of the row a statement runs for, which its programs read by name
fn outer_column_names(ctx: &ExecutionContext) -> Option<Vec<String>> {
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
        if let Some(dot_idx) = lower.rfind('.') {
            let column_part = &lower[dot_idx + 1..];
            col_index_map_lower
                .entry(column_part.to_string())
                .or_insert(i);
        }
    }
    col_index_map_lower
}

/// Where one projected value comes from: a column of the input row, or a
/// program compiled once for the statement
#[derive(Debug)]
enum ItemSource {
    Column(usize),
    Program(SharedProgram),
}

#[cold]
fn view_parameters() -> Error {
    Error::InvalidArgument(crate::parser::parser::VIEW_TAKES_NO_PARAMETERS.to_string())
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

/// Whether a WHERE `expr` keeps no row: one of its AND terms is a statement
/// constant, so no column reads as the text of its name, that evaluates to
/// false or NULL
fn true_for_no_row(expr: &Expression, ctx: &ExecutionContext) -> bool {
    if let Expression::Infix(infix) = expr {
        if infix.op_type == InfixOperator::And {
            return true_for_no_row(&infix.left, ctx) || true_for_no_row(&infix.right, ctx);
        }
    }
    statement_constant(expr, global_registry())
        && matches!(
            super::expression::evaluate_once(expr, Some(ctx)),
            Ok(Value::Boolean(false) | Value::Null(_))
        )
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

/// The table an index join can probe on a side that names one: never a table
/// read AS OF, whose past rows no index holds
pub(crate) fn index_join_table_name(side: &Expression) -> Option<String> {
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

/// Partition WHERE clause predicates for JOIN filter pushdown.
/// Returns (left_filter, right_filter, cross_table_filter).
/// - left_filter: predicates referencing only left table
/// - right_filter: predicates referencing only right table
/// - cross_table_filter: predicates referencing both tables (must be applied post-join)
///
/// An unqualified column belongs to a side only when both sides are base
/// tables and only that one has it; otherwise its predicate stays post-join
fn partition_where_for_join(
    where_clause: &Expression,
    left_alias: &str,
    right_alias: &str,
    (left_schema, right_schema): (Option<&Schema>, Option<&Schema>),
) -> (Option<Expression>, Option<Expression>, Option<Expression>) {
    let left_alias_lower = left_alias.to_lowercase();
    let right_alias_lower = right_alias.to_lowercase();

    // Collect AND-ed predicates
    let predicates = flatten_and_predicates(where_clause);

    let mut left_preds = Vec::new();
    let mut right_preds = Vec::new();
    let mut cross_preds = Vec::new();

    for pred in predicates {
        let qualifiers = collect_table_qualifiers(&pred);
        let mut unqualified = FxHashSet::default();
        collect_unqualified_columns(&pred, &mut unqualified);

        let mut refs_left = qualifiers.contains(&left_alias_lower);
        let mut refs_right = qualifiers.contains(&right_alias_lower);
        let mut unowned = false;
        for column in &unqualified {
            match (
                left_schema.map(|s| s.has_column(column)),
                right_schema.map(|s| s.has_column(column)),
            ) {
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
    Streaming { bounded: bool },
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

/// The left chunk the last reduction pass used; execute_join_source retries
/// with twice the chunk when the final rows stop short of the limit
#[derive(Clone, Copy)]
struct ReductionPass {
    cap: usize,
    chunk_full: bool,
    limit: usize,
}

/// The most rows an outer side may hold for a join without a LIMIT to
/// narrow its inner side by the inner table's candidates
const SMALL_OUTER_ROWS: usize = 1024;

/// Inner rows a table without sealed rows must hold per row of the largest
/// small outer before a small outer narrows it: a key read costs about what
/// its scan pays for 17 rows, 45 under a pushed filter
const HOT_INNER_ROWS_PER_KEY: usize = 64;

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
        self.execute_select_with_plan(stmt, ctx, None, None, None)
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

    /// `ctx` with the counts the subqueries of `stmt`'s LIMIT and OFFSET
    /// answer, read once before the statement reads a row, under its
    /// snapshot; their blocks are held in `slot`
    #[cold]
    #[inline(never)]
    fn read_row_counts(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        slot: Option<&super::query_cache::CountsSlot>,
    ) -> Result<ExecutionContext> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::row_counts_read();
        let clauses = [
            (stmt.limit.as_deref(), "LIMIT"),
            (stmt.offset.as_deref(), "OFFSET"),
        ];
        let counts = clauses.iter().filter_map(|(count, _)| *count);
        if let Err(reason) = self.dml_family(None, counts, ctx) {
            return Err(Error::invalid_argument(format!(
                "a LIMIT or OFFSET subquery is not supported here: {reason}"
            )));
        }
        // A count is read before any row: its scope is a table without columns
        let values = clauses.map(|(count, _)| count.map_or(&[][..], std::slice::from_ref));
        let (prepared, tables) = self.open_prepared_dml(
            slot.map(|slot| &slot.compiled),
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
                let value = vm.execute_cow(program, &exec)?;
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
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
        counts: Option<&super::query_cache::CountsSlot>,
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
        let reads_counts = [stmt.limit.as_deref(), stmt.offset.as_deref()]
            .into_iter()
            .flatten()
            .any(Self::has_subqueries);
        let snapshot_ctx;
        let ctx = if reads_counts && ctx.statement_snapshot().is_none() {
            snapshot_ctx = self.with_counts_snapshot(ctx)?;
            snapshot_ctx.as_ref().unwrap_or(ctx)
        } else {
            ctx
        };

        // Check for CTEs (WITH clause). A statement with one keeps every
        // subquery for the aggregation entry, which runs with the CTEs bound
        // A statement of the prepared or grouped family keeps its plan: its
        // CTEs inline as the CTE path inlines them, or are read from the
        // context, materialized under the snapshot its subqueries read
        let ctes_ctx;
        let ctx = if self.has_cte(stmt) {
            let grouped = || {
                Self::grouped_family_reach(stmt, &get_classification(stmt))
                    && self.grouped_family(stmt, Some(ctx)).is_ok()
            };
            if !self.holds_prepared_select(compiled, ctx)
                && !self.holds_prepared_grouped(compiled, ctx)
                && self.prepared_family(stmt, Some(ctx)).is_err()
                && !grouped()
            {
                return self.execute_select_with_ctes(stmt, ctx);
            }
            let with = stmt
                .with
                .as_ref()
                .ok_or_else(|| Error::internal("a statement with CTEs without WITH"))?;
            if let Some(inlined) = self.try_inline_ctes(stmt, with) {
                return self.execute_select_with_plan(&inlined, ctx, None, compiled, None);
            }
            let snapshot = match ctx.statement_snapshot() {
                Some(_) => None,
                None => self.with_counts_snapshot(ctx)?,
            };
            let ctx = snapshot.as_ref().unwrap_or(ctx);
            ctes_ctx = ctx.with_cte_data(self.materialize_ctes(with, ctx, None)?.data());
            &ctes_ctx
        } else {
            ctx
        };

        // The subquery is read once, before any path reads a row, with the
        // CTEs of the statement bound
        let counts_ctx;
        let ctx = if reads_counts {
            counts_ctx = self.read_row_counts(stmt, ctx, counts)?;
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

        // A statement of the prepared family runs on that path and no other;
        // one without FROM never is
        let in_family = (classification.where_has_subqueries
            || classification.output_has_subqueries)
            && stmt.table_expr.is_some()
            && (self.holds_prepared_select(compiled, ctx)
                || self.prepared_family(stmt, Some(ctx)).is_ok());
        // A grouped or windowed statement whose clauses hold subqueries runs
        // them as its blocks
        let grouped_family = Self::grouped_family_reach(stmt, &classification)
            && !in_family
            && (self.holds_prepared_grouped(compiled, ctx)
                || self.grouped_family(stmt, Some(ctx)).is_ok());

        // Correlated probes read the statement's tables through one transaction
        let snapshot_ctx;
        let ctx = if (classification.where_has_correlated_subqueries
            || classification.select_has_correlated_subqueries
            || classification.order_by_has_correlated_subqueries
            || in_family
            || grouped_family)
            && ctx.statement_snapshot().is_none()
            && self.active_transaction.lock().unwrap().is_none()
        {
            let snapshot =
                self.new_statement_snapshot(self.isolation_level(), ctx.statement_view())?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

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

        // Evaluate LIMIT/OFFSET early (needed for set operations optimization)
        let RowCounts { limit, offset } = row_counts(stmt, ctx)?;

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
        let (mut result, columns, limit_offset_applied, shape) = if in_family {
            // DISTINCT reads every row before a LIMIT counts them; a set
            // operation's branch takes the bound its whole one allows
            let (limit, offset) = match (stmt.set_operations.is_empty(), stmt.distinct) {
                (false, _) => (set_limit, 0),
                (true, true) => (None, 0),
                (true, false) => (limit, offset),
            };
            let (result, columns, applied, visible, layout, distinct) =
                self.execute_prepared_select(stmt, ctx, compiled, limit, offset)?;
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
        let SelectShape {
            deferred: deferred_projection,
            layout,
            visible,
        } = shape.unwrap_or_default();
        let prepared_visible = prepared_visible.or(visible);

        // Apply set operations (UNION, INTERSECT, EXCEPT)
        // Pass limit+offset to enable early termination for UNION ALL
        let mut limit_offset_applied = limit_offset_applied;
        if !stmt.set_operations.is_empty() {
            result = self.execute_set_operations(result, &stmt.set_operations, ctx, set_limit)?;

            // The set operation gathers LIMIT + OFFSET rows at most; the
            // OFFSET is skipped and the LIMIT cut below, on the whole
            limit_offset_applied = false;
        }

        // Count expected SELECT columns (before any extra ORDER BY columns).
        // Beside a star the count is unknown, but the only columns past
        // the visible ones are then the window functions ORDER BY had
        // computed for it, so those are counted off the end
        let expected_columns =
            match prepared_visible.unwrap_or_else(|| self.count_select_columns(stmt)) {
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

        // Apply DISTINCT (skip for DISTINCT ON — it's applied after ORDER BY)
        // When ORDER BY references columns not in SELECT, we add extra columns for sorting.
        // DISTINCT should only consider the original SELECT columns, not the extra ORDER BY columns.
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

        // Apply ORDER BY (with TOP-N optimization if LIMIT is present)
        // Note: LIMIT/OFFSET was already evaluated earlier for set operations optimization
        // Skip ORDER BY if storage-level optimization already applied sorting + LIMIT/OFFSET
        // classification was already obtained at entry point and passed through
        // A DISTINCT ON key reads only where the first branch put its items,
        // which a set operation's rows keep
        let item_layout = layout.as_deref();
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
                    visit_order_by_inputs(stmt, &mut |column, key| {
                        if let Some(position) = layout.position(&key) {
                            sort_slots.entry(ExprKey::of(column)).or_insert(position);
                        }
                    });
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
                            if !name.contains('.') {
                                outer_names.push((
                                    CompactArc::from(format!("{}.{}", alias, name).as_str()),
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
                            if let Some((_, bare)) = name.rsplit_once('.') {
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
                    result = Self::distinct_on_result(stmt, result, item_layout, expected_columns)?;

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
                        result = Box::new(StreamingProjectionResult::new(
                            result,
                            col_indices,
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
            result = Self::distinct_on_result(stmt, result, item_layout, expected_columns)?;
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

    /// `result` deduplicated on `stmt`'s DISTINCT ON keys, each read from the
    /// column `resolve_distinct_on_indices` binds it to
    fn distinct_on_result(
        stmt: &SelectStatement,
        result: Box<dyn QueryResult>,
        layout: Option<&OutputLayout>,
        visible: usize,
    ) -> Result<Box<dyn QueryResult>> {
        let columns = result.columns().to_vec();
        let visible = if visible == 0 { columns.len() } else { visible };
        // An item's column by the layout; without one, counted from the front,
        // or from the end of the visible columns for one after every star
        let item_column = |key: &Expression| {
            let item = Self::item_computing(stmt, key)?;
            if let Some(layout) = layout {
                return layout.item_position(item);
            }
            let first_star = stmt.columns.iter().position(is_star_item);
            let last_star = stmt.columns.iter().rposition(is_star_item);
            match (first_star, last_star) {
                (Some(first), Some(last)) if first < item && item < last => None,
                (_, Some(last)) if last < item => visible.checked_sub(stmt.columns.len() - item),
                _ => Some(item),
            }
        };
        let indices = Self::resolve_distinct_on_indices(
            &stmt.distinct_on,
            &columns,
            Some(&stmt.columns),
            &item_column,
        );
        let indices = stmt
            .distinct_on
            .iter()
            .zip(indices)
            .map(|(key, index)| {
                index.ok_or_else(|| {
                    Error::invalid_argument(format!("DISTINCT ON key {key} names no column"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(DistinctOnResult::new(result, indices)))
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

    /// Resolve DISTINCT ON expressions to column indices in the result set,
    /// None for a key no column holds; a key carried as a hidden column is
    /// found by its identity.
    /// Also checks original SELECT expressions to handle aliased columns
    /// (e.g., `SELECT DISTINCT ON (customer) customer AS c` — matches `customer`
    /// against the original identifier behind the `c` alias).
    fn resolve_distinct_on_indices(
        distinct_on: &[Expression],
        result_columns: &[String],
        select_exprs: Option<&[Expression]>,
        item_column: &dyn Fn(&Expression) -> Option<usize>,
    ) -> Vec<Option<usize>> {
        distinct_on
            .iter()
            .map(|expr| {
                let hidden = || {
                    let key = ExprKey::of(expr);
                    result_columns
                        .iter()
                        .position(|c| ExprKey::from_hidden_column_name(c) == Some(key))
                };
                let bound = match expr {
                    Expression::Identifier(id) => {
                        // First: direct match against result column names
                        let direct = result_columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&id.value_lower));
                        if direct.is_some() {
                            return direct;
                        }
                        // Second: check if any SELECT expression is an alias over this identifier
                        if let Some(sel_exprs) = select_exprs {
                            for (i, sel) in sel_exprs.iter().enumerate() {
                                if let Expression::Aliased(aliased) = sel {
                                    match &*aliased.expression {
                                        Expression::Identifier(sel_id)
                                            if sel_id.value_lower == id.value_lower =>
                                        {
                                            return Some(i);
                                        }
                                        Expression::QualifiedIdentifier(qid)
                                            if qid.name.value_lower == id.value_lower =>
                                        {
                                            return Some(i);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        None
                    }
                    Expression::QualifiedIdentifier(qi) => {
                        // Match table.column — try full qualified name first
                        let full = format!("{}.{}", qi.qualifier.value, qi.name.value);
                        let qualified_match = result_columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&full));
                        if qualified_match.is_some() {
                            return qualified_match;
                        }
                        // Second: check SELECT expressions for the source column behind
                        // aliases or projections (e.g., c.name AS customer, or bare c.name
                        // projected as "name")
                        if let Some(sel_exprs) = select_exprs {
                            for (i, sel) in sel_exprs.iter().enumerate() {
                                match sel {
                                    Expression::QualifiedIdentifier(sel_qi)
                                        if sel_qi.qualifier.value_lower
                                            == qi.qualifier.value_lower
                                            && sel_qi.name.value_lower == qi.name.value_lower =>
                                    {
                                        return Some(i);
                                    }
                                    Expression::Aliased(aliased) => {
                                        if let Expression::QualifiedIdentifier(sel_qi) =
                                            &*aliased.expression
                                        {
                                            if sel_qi.qualifier.value_lower
                                                == qi.qualifier.value_lower
                                                && sel_qi.name.value_lower == qi.name.value_lower
                                            {
                                                return Some(i);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        // Last resort: unqualified only if unambiguous (exactly one match)
                        let unqualified_matches: Vec<usize> = result_columns
                            .iter()
                            .enumerate()
                            .filter(|(_, c)| c.eq_ignore_ascii_case(&qi.name.value))
                            .map(|(i, _)| i)
                            .collect();
                        if unqualified_matches.len() == 1 {
                            Some(unqualified_matches[0])
                        } else {
                            None
                        }
                    }
                    Expression::IntegerLiteral(lit) => {
                        // 1-based positional reference
                        let pos = lit.value as usize;
                        if pos >= 1 && pos <= result_columns.len() {
                            Some(pos - 1)
                        } else {
                            None
                        }
                    }
                    _ => {
                        // The SELECT column computing the same expression, else the
                        // hidden column carried for it, as ORDER BY reads them, else
                        // the column named after it
                        let key = ExprKey::of(expr);
                        item_column(expr)
                            .or_else(|| {
                                result_columns
                                    .iter()
                                    .position(|c| ExprKey::from_hidden_column_name(c) == Some(key))
                            })
                            .or_else(|| {
                                let expr_str = expr.to_string();
                                result_columns.iter().position(|c| *c == expr_str)
                            })
                    }
                };
                bound.or_else(hidden)
            })
            .collect()
    }

    /// Count the number of columns in the SELECT clause
    fn count_select_columns(&self, stmt: &SelectStatement) -> usize {
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
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
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
                    return self.execute_view_query(&view_def, stmt, ctx, classification);
                }
                self.execute_simple_table_scan(table_source, stmt, ctx, classification, None, None)
            }
            Expression::JoinSource(join_source) => {
                self.execute_join_source(join_source, stmt, ctx, classification, compiled)
            }
            Expression::SubquerySource(subquery_source) => {
                self.execute_subquery_source(subquery_source, stmt, ctx, classification)
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
            Some(filter) if true_for_no_row(filter, ctx) => Some(0),
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
            Self::evaluate_tvf_with_range(tvf_source, ctx, tvf_limit, range_hint)?;

        let table_alias = tvf_source
            .alias
            .as_ref()
            .map(|a| a.value.to_string())
            .unwrap_or_else(|| tvf_source.function.value.to_string());

        let qualified_columns: Vec<String> = column_names
            .iter()
            .map(|col| format!("{}.{}", table_alias, col))
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
    fn fetch_limit(stmt: &SelectStatement, ctx: &ExecutionContext) -> Option<usize> {
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

        let (result_rows, columns) =
            Self::tvf_rows(tvf_source, stmt.where_clause.as_deref(), limit_hint, ctx)?;

        // Delegate to execute_query_on_memory_result which handles
        // WHERE, ORDER BY, LIMIT, GROUP BY, aggregation, window functions
        self.execute_query_on_memory_result(stmt, ctx, columns, result_rows)
    }

    /// The rows and columns a TVF source gives a statement whose WHERE is
    /// `where_clause`: none when it holds for no row, else the range its
    /// conjuncts on the TVF's column allow, up to `limit`
    pub(super) fn tvf_rows(
        tvf_source: &FunctionTableSource,
        where_clause: Option<&Expression>,
        limit: Option<usize>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        // Short-circuit: if WHERE is a constant-false expression (e.g., 1=0),
        // skip TVF generation entirely and return empty result.
        if let Some(where_clause) = where_clause {
            if true_for_no_row(where_clause, ctx) {
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

        Self::evaluate_tvf_with_range(tvf_source, ctx, limit, range_hint)
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

    fn evaluate_tvf_with_range(
        tvf_source: &FunctionTableSource,
        ctx: &ExecutionContext,
        limit: Option<usize>,
        range_hint: Option<Resolved>,
    ) -> Result<(RowVec, Vec<String>)> {
        let tvf = Self::tvf_of(tvf_source)?;

        // Evaluate arguments to Values
        let mut arg_values = Vec::with_capacity(tvf_source.arguments.len());
        for arg in &tvf_source.arguments {
            arg_values.push(super::expression::evaluate_once(arg, Some(ctx))?);
        }

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

        // Check WHERE clause first - if it evaluates to false, return empty result
        if let Some(where_clause) = &stmt.where_clause {
            // Pre-process subqueries in WHERE clause (EXISTS, IN, ALL/ANY, scalar subqueries)
            let processed_where = self.process_where_subqueries(where_clause, ctx)?;
            // A parent row binds the outer references of a FROM-less subquery
            let processed_where = match ctx.outer_row() {
                Some(outer) => super::utils::substitute_outer_references(&processed_where, outer),
                None => processed_where,
            };
            let where_result = super::expression::evaluate_once(&processed_where, Some(ctx))?;
            let passes = match where_result {
                Value::Boolean(b) => b,
                Value::Null(_) => false, // NULL in WHERE is treated as false
                _ => true,
            };
            if !passes {
                // WHERE clause is false, return empty result
                // Still need to determine column names for the schema
                let mut columns: Vec<String> = Vec::new();
                for (i, col_expr) in stmt.columns.iter().enumerate() {
                    let col_name = if let Expression::Aliased(aliased) = col_expr {
                        aliased.alias.value.to_string()
                    } else {
                        format!("column{}", i + 1)
                    };
                    columns.push(col_name);
                }
                let columns = CompactArc::new(columns);
                let result =
                    ExecutorResult::with_arc_columns(CompactArc::clone(&columns), RowVec::new());
                let shape = SelectShape::laid_out(Self::result_layout(stmt, &[], None, &columns));
                return Ok((Box::new(result), columns, false, shape));
            }
        }

        // Aggregates and windows read one empty row, as in SELECT SUM(3+5)
        if classification.has_aggregation || classification.has_window_functions {
            let mut dummy_rows = RowVec::new();
            dummy_rows.push((0, Row::from_values(vec![])));
            return self.grouped_select(stmt, ctx, classification, dummy_rows, &[], None);
        }

        // Process scalar subqueries in SELECT columns first (single-pass)
        let processed_columns = self.try_process_select_subqueries(&stmt.columns, ctx)?;
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (i, col_expr) in columns_to_use.iter().enumerate() {
            // Get column name
            let col_name = if let Expression::Aliased(aliased) = col_expr {
                aliased.alias.value.to_string()
            } else {
                format!("column{}", i + 1)
            };
            columns.push(col_name);

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
                // Check if ORDER BY columns match this qualifier
                stmt.order_by.iter().all(|ob| {
                    if let Expression::QualifiedIdentifier(qid) = &ob.expression {
                        qid.qualifier.value_lower == qs.qualifier.to_lowercase()
                    } else if let Expression::Identifier(_) = &ob.expression {
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
                    select_columns.insert(format!(
                        "{}.{}",
                        qi.qualifier.value_lower, qi.name.value_lower
                    ));
                }
                Expression::Aliased(a) => {
                    if let Expression::QualifiedIdentifier(qi) = &*a.expression {
                        select_columns.insert(format!(
                            "{}.{}",
                            qi.qualifier.value_lower, qi.name.value_lower
                        ));
                    }
                }
                _ => {}
            }
        }

        // A correlated subquery in ORDER BY reads the parent row, all of
        // it, so every column is carried until the sort has run
        if stmt
            .order_by
            .iter()
            .any(|ob| Self::has_correlated_subqueries(&ob.expression))
        {
            return true;
        }

        // Check if any ORDER BY column is not in SELECT
        for ob in &stmt.order_by {
            match &ob.expression {
                Expression::Identifier(id)
                    if !select_columns.contains(id.value_lower.as_str())
                        && all_columns
                            .iter()
                            .any(|c| c.eq_ignore_ascii_case(id.value_lower.as_str())) =>
                {
                    return true;
                }
                Expression::QualifiedIdentifier(qi) => {
                    let full_name = format!("{}.{}", qi.qualifier.value_lower, qi.name.value_lower);
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
                // An expression is read from a column of its own, unless the
                // SELECT list already has it under a name. A subquery reads
                // the parent row per row, which the sort does for itself
                other
                    if !matches!(
                        other,
                        Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                    ) && !Self::has_subqueries(other) =>
                {
                    let name = other.to_string();
                    let key = ExprKey::of(other);
                    let aliased = stmt.columns.iter().any(|expr| match expr {
                        Expression::Aliased(a) => ExprKey::of(&a.expression) == key,
                        _ => false,
                    });
                    if !aliased && !select_columns.contains(name.to_lowercase().as_str()) {
                        return true;
                    }
                }
                // One the sort evaluates reads its input columns from the row
                other if Self::has_subqueries(other) => {
                    let mut names_input = false;
                    visit_expression_inputs(other, &mut |_, key| {
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

        // Check if any DISTINCT ON column is not in SELECT
        for expr in &stmt.distinct_on {
            match expr {
                Expression::Identifier(id) => {
                    if !select_columns.contains(id.value_lower.as_str())
                        && all_columns
                            .iter()
                            .any(|c| c.eq_ignore_ascii_case(id.value_lower.as_str()))
                    {
                        return true;
                    }
                }
                Expression::QualifiedIdentifier(qi) => {
                    let full_name = format!("{}.{}", qi.qualifier.value_lower, qi.name.value_lower);
                    // For qualified identifiers, only match the fully qualified name
                    // to avoid ambiguity when both tables have the same column name
                    if !select_columns.contains(full_name.as_str())
                        && all_columns.iter().any(|c| {
                            c.eq_ignore_ascii_case(&full_name)
                                || c.eq_ignore_ascii_case(qi.name.value_lower.as_str())
                        })
                    {
                        return true;
                    }
                }
                _ => {
                    // Computed expression (e.g., amount + 0) — always needs extra column
                    return true;
                }
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
        source_columns: &[String],
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
            match expr {
                Expression::Identifier(id) => {
                    let idx = source_columns_lower
                        .iter()
                        .position(|c| c == id.value_lower.as_str())?;
                    indices.push(idx);
                    output_names.push(source_columns[idx].clone());
                }
                Expression::QualifiedIdentifier(qid) => {
                    let idx = source_columns_lower
                        .iter()
                        .position(|c| c == qid.name.value_lower.as_str())?;
                    indices.push(idx);
                    output_names.push(source_columns[idx].clone());
                }
                Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                    Expression::Identifier(id) => {
                        let idx = source_columns_lower
                            .iter()
                            .position(|c| c == id.value_lower.as_str())?;
                        indices.push(idx);
                        output_names.push(aliased.alias.value.to_string());
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        let idx = source_columns_lower
                            .iter()
                            .position(|c| c == qid.name.value_lower.as_str())?;
                        indices.push(idx);
                        output_names.push(aliased.alias.value.to_string());
                    }
                    _ => return None, // Complex expression
                },
                Expression::Star(_) | Expression::QualifiedStar(_) => {
                    return None; // SELECT * doesn't benefit
                }
                _ => return None, // Function call, arithmetic, etc.
            }
        }

        Some((indices, output_names))
    }

    /// Try to get distinct values directly from an index
    ///
    /// This optimization works for queries like:
    /// - SELECT DISTINCT col FROM table (where col is indexed)
    ///
    /// Conditions:
    /// - Single column in SELECT (not *, not expression)
    /// - No WHERE clause
    /// - No GROUP BY, HAVING
    /// - No ORDER BY (could be extended later)
    /// - No LIMIT/OFFSET (could be extended later)
    /// - The column must have an index
    fn try_distinct_pushdown(
        &self,
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        all_columns: &[String],
        classification: &std::sync::Arc<QueryClassification>,
    ) -> Result<Option<Box<dyn crate::storage::traits::QueryResult>>> {
        // Must be DISTINCT
        if !stmt.distinct {
            return Ok(None);
        }

        // DISTINCT ON uses key-based dedup, not full-row dedup — cannot use index pushdown
        if !stmt.distinct_on.is_empty() {
            return Ok(None);
        }

        // Quick eligibility checks using cached classification
        if classification.has_where {
            return Ok(None);
        }
        if classification.has_group_by {
            return Ok(None);
        }
        if classification.has_having {
            return Ok(None);
        }
        if classification.has_aggregation {
            return Ok(None);
        }
        if classification.has_window_functions {
            return Ok(None);
        }
        // ORDER BY is OK - we can sort the distinct values after

        // Must have exactly one column
        if stmt.columns.len() != 1 {
            return Ok(None);
        }

        // Get the column name (must be a simple identifier, not an expression)
        let column_name = match &stmt.columns[0] {
            Expression::Identifier(id) => id.value_lower.to_string(),
            Expression::QualifiedIdentifier(qid) => qid.name.value_lower.to_string(),
            Expression::Aliased(aliased) => match &*aliased.expression {
                Expression::Identifier(id) => id.value_lower.to_string(),
                Expression::QualifiedIdentifier(qid) => qid.name.value_lower.to_string(),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };

        // Verify this column exists in the table
        let column_exists = all_columns
            .iter()
            .any(|c| c.eq_ignore_ascii_case(&column_name));
        if !column_exists {
            return Ok(None);
        }

        // Try to get distinct values from the index
        let distinct_values = match table.get_partition_values(&column_name)? {
            Some(values) => Some(values),
            None => match table.schema().column_index_map().get(&column_name) {
                Some(&col_idx) => table.compute_distinct_values(col_idx)?,
                None => None,
            },
        };

        if let Some(distinct_values) = distinct_values {
            // Build output column name (use alias if present)
            let output_name = match &stmt.columns[0] {
                Expression::Aliased(aliased) => aliased.alias.value.to_string(),
                Expression::Identifier(id) => id.value.to_string(),
                Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
                _ => column_name,
            };

            // Convert values to rows
            let rows: RowVec = distinct_values
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i as i64, Row::from_values(vec![v])))
                .collect();

            let result = ExecutorResult::new(vec![output_name], rows);
            return Ok(Some(Box::new(result)));
        }

        Ok(None)
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
        if !classification.semantic_cacheable()
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
        match self.semantic_cache_answer(
            table_name,
            stmt,
            &all_columns,
            Some(alias),
            predicate,
            stamp,
        )? {
            Some((result, _)) => Ok(EarlySemantic::Hit(result)),
            None => Ok(EarlySemantic::Miss(miss)),
        }
    }

    /// The cached rows of `SELECT *` over `table_name` filtered by
    /// `where_expr` under `stamp`, with their output columns; None on a miss
    fn semantic_cache_answer(
        &self,
        table_name: &str,
        stmt: &SelectStatement,
        all_columns: &[String],
        table_alias: Option<&str>,
        where_expr: &Expression,
        stamp: super::semantic_cache::CacheStamp,
    ) -> Result<Option<CachedAnswer>> {
        use super::semantic_cache::{CacheLookupResult, SemanticCache};
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
    /// columns the list lacks follow, for execute_select to sort by and drop
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
    ) -> Result<(RowVec, Vec<String>, Option<OutputLayout>)> {
        let mut output_columns =
            self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        if order_by_extra {
            let (projected_rows, extra_names, layout) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                rows,
                all_columns,
                table_alias,
                ctx,
            )?;
            output_columns.extend(extra_names);
            return Ok((projected_rows, output_columns, Some(layout)));
        }
        let projected_rows = self.project_rows_with_alias(
            &stmt.columns,
            rows,
            all_columns,
            Some(all_columns_lower),
            ctx,
            table_alias,
        )?;
        let layout = Self::shared_output_layout(stmt, all_columns, table_alias);
        Ok((projected_rows, output_columns, layout))
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
                let (rows, columns, layout) = self.project_scanned_rows(
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
                    false,
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
            && ctx.statement_snapshot().is_none()
            && ctx.outer_row().is_none()
            && table_source.as_of.is_none()
            && stmt.where_clause.is_some()
            && classification.semantic_cacheable();
        let cache_stamp = if cacheable {
            self.semantic_cache_stamp(table_name, ctx.scan_view(table_name), |_| ())
                .map(|(stamp, ())| stamp)
        } else {
            None
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
                if let Some(ref as_of) = table_source.as_of {
                    let tx = self.begin_transaction()?;
                    return self.execute_temporal_query(
                        table_name,
                        as_of,
                        stmt,
                        ctx,
                        &*tx,
                        classification,
                    );
                }
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

                // Check for AS OF (temporal query)
                if let Some(ref as_of) = table_source.as_of {
                    return self.execute_temporal_query(
                        table_name,
                        as_of,
                        stmt,
                        ctx,
                        &*tx,
                        classification,
                    );
                }

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
                && true_for_no_row(w, ctx)
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
            // MIN or MAX of an indexed column alone, with no WHERE or GROUP BY,
            // reads the index's bound instead of every row
            if !classification.has_where && !classification.has_group_by && stmt.having.is_none() {
                if let Some((result, columns)) =
                    self.try_min_max_index_optimization(stmt, &*table, &all_columns)?
                {
                    let shape = shape_of(&columns);
                    return Ok((result, columns, false, shape));
                }
            }

            // First try global aggregation pushdown (no GROUP BY, no WHERE)
            if let Some(result) =
                self.try_aggregation_pushdown(table.as_ref(), stmt, ctx, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                let shape = shape_of(&columns);
                return Ok((result, columns, false, shape));
            }

            // Try filtered aggregation pushdown (WHERE + aggregates, no GROUP BY)
            // Pushes both the filter and aggregation into the storage layer
            if classification.has_where {
                if let Some(result) = self.try_filtered_aggregation_pushdown(
                    table.as_ref(),
                    stmt,
                    ctx,
                    classification,
                    &all_columns,
                )? {
                    let columns = CompactArc::new(result.columns().to_vec());
                    let shape = shape_of(&columns);
                    return Ok((result, columns, false, shape));
                }
            }

            // Try storage-level GROUP BY aggregation (no WHERE clause)
            // This computes aggregates directly from arena without materializing Row objects
            if classification.has_group_by && !classification.has_where {
                if let Some(result) = self.try_storage_aggregation(
                    table.as_ref(),
                    stmt,
                    ctx,
                    &all_columns,
                    classification,
                ) {
                    let result = result?;
                    let columns = CompactArc::new(result.columns().to_vec());
                    let shape = shape_of(&columns);
                    return Ok((result, columns, false, shape));
                }
            }

            // Try streaming aggregation for expressions like AVG(col) * 100
            // This avoids collecting all rows by streaming through a scanner
            if let Some(result) =
                self.try_streaming_global_aggregation(table.as_ref(), stmt, ctx, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                let shape = shape_of(&columns);
                return Ok((result, columns, false, shape));
            }
        }

        // DISTINCT PUSHDOWN: Try to get distinct values directly from index
        // This avoids all row materialization for queries like:
        // - SELECT DISTINCT col FROM table (where col is indexed)
        // Returns distinct values in O(unique values) instead of O(rows)
        if classification.has_distinct {
            if let Some(result) =
                self.try_distinct_pushdown(table.as_ref(), stmt, &all_columns, classification)?
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

        // Substitute column aliases in WHERE clause if any - avoid clone when no substitution needed
        let resolved_where_clause: Option<Box<Expression>> = if !alias_map.is_empty() {
            stmt.where_clause
                .as_ref()
                .map(|where_expr| Box::new(Self::substitute_aliases(where_expr, &alias_map)))
        } else {
            None // Will use stmt.where_clause directly via reference when possible
        };

        // Note: Projection pushdown was found to be SLOWER due to extra allocations
        // We now always fetch all columns and project in the executor
        let _ = needs_all_columns; // Silence unused warning - kept for documentation

        // Try to build a storage expression for WHERE clause pushdown
        // This enables index usage and avoids full table scans
        // Use resolved_where_clause if we had alias substitutions, otherwise use original
        let where_to_use: Option<&Expression> = resolved_where_clause
            .as_deref()
            .or(stmt.where_clause.as_deref());

        // Apply expression simplification for constant folding and boolean optimization
        // This can eliminate redundant comparisons (e.g., 1=1 -> true, true AND x -> x)
        // Uses try_simplify() to avoid cloning expressions that can't be simplified (like EXISTS)
        let (simplified_where, where_to_use): (Option<Expression>, Option<&Expression>) =
            if let (Some(miss), Some(where_expr), None) =
                (early_miss, where_to_use, &resolved_where_clause)
            {
                (None, Some(miss.simplified.as_deref().unwrap_or(where_expr)))
            } else if let Some(where_expr) = where_to_use {
                let mut simplifier = ExpressionSimplifier::new();
                // try_simplify returns None if no changes, avoiding unnecessary cloning
                if let Some(simplified) = simplifier.try_simplify(where_expr) {
                    (Some(simplified), None) // Will be fixed up below
                } else {
                    // No simplification happened, use original (no clone occurred)
                    (None, Some(where_expr))
                }
            } else {
                (None, None)
            };

        // If we simplified, borrow from the simplified version
        let where_to_use: Option<&Expression> = where_to_use.or(simplified_where.as_ref());

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

        // FAST PATH: Streaming GROUP BY optimization using B-tree index
        // For queries like `SELECT user_id, SUM(amount) FROM orders GROUP BY user_id`
        // where user_id has a B-tree index, we iterate through the index in sorted order
        // and aggregate each group without using a hash map (SQLite-style sorted GROUP BY)
        if storage_expr.is_none()
            && !needs_memory_filter
            && classification.has_group_by
            && classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_order_by
        // No ORDER BY to avoid re-sorting
        {
            if let Some((result, columns)) =
                self.try_streaming_group_by(stmt, &*table, &all_columns, ctx)?
            {
                let shape = shape_of(&columns);
                return Ok((result, columns, false, shape));
            }
        }

        // FAST PATH: ORDER BY + LIMIT optimization (TOP-N)
        // For queries like `SELECT * FROM table ORDER BY indexed_col LIMIT 10`,
        // use index to get rows in sorted order directly, avoiding full table sort
        if classification.has_limit
            && stmt.order_by.len() == 1
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_distinct
            && storage_expr.is_none()
            && !needs_memory_filter
        {
            if let Some((result, columns)) = self.try_order_by_index_optimization(
                stmt,
                &*table,
                &all_columns,
                table_alias.as_deref(),
                ctx,
            )? {
                // Note: ORDER BY + LIMIT already handles LIMIT at storage level
                OrderLimitPath::IndexOrdered.taken();
                return Ok((result, columns, true, None));
            }
        }

        // FAST PATH: ORDER BY one column + LIMIT answered by the table's top-k
        // scan with the pushed-down WHERE, so only the rows near the bound are read
        if classification.has_limit
            && stmt.order_by.len() == 1
            && stmt.distinct_on.is_empty()
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_distinct
            && !needs_memory_filter
        {
            if let Some((result, columns)) = self.try_top_k_scan_optimization(
                stmt,
                &*table,
                storage_expr.as_deref(),
                &all_columns,
                table_alias.as_deref(),
                ctx,
            )? {
                OrderLimitPath::TopKScan.taken();
                return Ok((result, columns, true, None));
            }
        }

        // FAST PATH: Vector search optimization (HNSW index or parallel brute-force)
        // For queries like `SELECT id, VEC_DISTANCE_L2(embedding, '...') AS dist
        //   FROM documents ORDER BY dist LIMIT 10`
        // Supports WHERE: brute-force pre-filters rows; HNSW oversamples then post-filters
        // Skip when transaction has local changes — HNSW index and collect_rows_by_ids
        // only see committed data, so local INSERTs/UPDATEs would be missed.
        if classification.has_limit
            && stmt.order_by.len() == 1
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_distinct
            && !table.has_local_changes()
        {
            if let Some((result, columns)) =
                self.try_vector_search_optimization(stmt, &*table, &all_columns, ctx)?
            {
                OrderLimitPath::VectorSearch.taken();
                return Ok((result, columns, true, None));
            }
        }

        // FAST PATH: Keyset pagination optimization
        // For queries like `SELECT * FROM table WHERE id > X ORDER BY id LIMIT Y`,
        // use the PK's ordering to start iteration from X directly.
        // This provides O(limit) complexity instead of O(n) for full scans.
        // Note: OFFSET is not supported - queries with OFFSET fall through to regular execution.
        if classification.has_limit
            && stmt.offset.is_none()
            && stmt.order_by.len() == 1
            && !classification.has_group_by
            && !classification.has_aggregation
            && !classification.has_window_functions
            && !classification.has_distinct
            && !needs_memory_filter
        {
            if let Some((result, columns)) = self.try_keyset_pagination_optimization(
                stmt,
                where_to_use,
                &*table,
                &all_columns,
                table_alias.as_deref(),
                ctx,
            )? {
                OrderLimitPath::KeysetPagination.taken();
                return Ok((result, columns, true, None));
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
            let rows =
                table.collect_rows_with_limit_unordered(storage_expr.as_deref(), limit, offset)?;

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
        let should_disable_streaming_for_cache = cache_eligible && {
            let table_row_count = table.row_count_hint();
            table_row_count <= super::semantic_cache::DEFAULT_MAX_CACHED_ROWS
        };

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
            // Check if we have simple column projection (no complex expressions)
            let simple_projection = self.get_simple_projection_indices(&stmt.columns, &all_columns);

            if let Some((column_indices, output_columns)) = simple_projection {
                // All columns are simple references - we can stream!
                let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
                // A table that reads by need materializes only the columns
                // the statement names
                let needed = (table.narrows_scans()
                    && !has_outer_context
                    && !classification.where_has_subqueries)
                    .then(|| super::utils::needed_columns(stmt, where_to_use, &all_columns_lower))
                    .flatten();
                let scanner = match needed {
                    Some(mask) => {
                        table.scan_needed(&column_idx_vec, &mask, storage_expr.as_deref())?
                    }
                    None => table.scan(&column_idx_vec, storage_expr.as_deref())?,
                };

                // Wrap scanner in ScannerResult
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

                // Apply projection if needed (not SELECT *)
                // OPTIMIZATION: Check if projection is identity without allocating a Vec
                let is_identity_projection = column_indices.len() == all_columns.len()
                    && column_indices.iter().enumerate().all(|(i, &idx)| idx == i);
                // Check if column names differ (aliases)
                let names_differ = output_columns.len() != all_columns.len()
                    || output_columns
                        .iter()
                        .zip(all_columns.iter())
                        .any(|(out, all)| out != all);
                // Need StreamingProjectionResult if either:
                // 1. Non-identity projection (different columns or reordered)
                // 2. Column names differ (aliases like "SELECT id AS a")
                if !column_indices.is_empty() && (!is_identity_projection || names_differ) {
                    let output_columns = CompactArc::new(output_columns);
                    result = Box::new(StreamingProjectionResult::new(
                        result,
                        column_indices,
                        (*output_columns).clone(),
                    ));
                    // LIMIT/OFFSET NOT applied yet - streaming path
                    let shape = shape_of(&output_columns);
                    return Ok((result, output_columns, false, shape));
                } else {
                    // SELECT * - no projection needed
                    // LIMIT/OFFSET NOT applied yet - streaming path
                    let shape = shape_of(&output_columns);
                    return Ok((result, CompactArc::new(output_columns), false, shape));
                }
            } else if !classification.select_has_scalar_subqueries {
                // STREAMING WITH EXPRESSIONS: Use ExprMappedResult for complex projections
                // This avoids batch allocation when SELECT contains expressions like CASE
                // NOTE: Only use this path when SELECT doesn't have subqueries (which need special processing)
                let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
                let needed = (table.narrows_scans()
                    && !has_outer_context
                    && !classification.has_window_functions
                    && !classification.where_has_subqueries)
                    .then(|| super::utils::needed_columns(stmt, where_to_use, &all_columns_lower))
                    .flatten();
                let scanner = match needed {
                    Some(mask) => {
                        table.scan_needed(&column_idx_vec, &mask, storage_expr.as_deref())?
                    }
                    None => table.scan(&column_idx_vec, storage_expr.as_deref())?,
                };

                let mut result: Box<dyn QueryResult> =
                    Box::new(ScannerResult::new(scanner, all_columns.clone()));

                // If we need memory filtering
                if needs_memory_filter {
                    if let Some(where_expr) = where_to_use {
                        let has_scalar_subqueries = classification.where_has_subqueries
                            && !classification.where_has_correlated_subqueries;

                        let filter = if has_scalar_subqueries {
                            let processed = self.process_where_subqueries(where_expr, ctx)?;
                            RowFilter::new(&processed, &all_columns)?.with_context(ctx)
                        } else {
                            RowFilter::new(where_expr, &all_columns)?.with_context(ctx)
                        };
                        result = Box::new(FilteredResult::from_filter(result, filter));
                    }
                }

                // Use ExprMappedResult for expression-based projection with buffer reuse
                let output_columns = self.get_output_column_names(
                    &stmt.columns,
                    &all_columns,
                    table_alias.as_deref(),
                );
                let output_columns = CompactArc::new(output_columns);

                result = Box::new(
                    ExprMappedResult::new(
                        result,
                        stmt.columns.clone(),
                        (*output_columns).clone(),
                        table_alias.as_deref(),
                    )?
                    .with_context(ctx),
                );

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
                            let batch = table.collect_rows_with_limit(
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
                        result_rows =
                            table.collect_rows_with_limit(storage_expr.as_deref(), target, 0)?;
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
                // With one, scan() keeps its PK/index fast paths.
                let all_rows = if storage_expr.is_none() {
                    table.collect_all_rows(None)?
                } else {
                    let mut scanner = match needed.as_deref() {
                        Some(mask) => {
                            table.scan_needed(&column_idx_vec, mask, storage_expr.as_deref())?
                        }
                        None => table.scan(&column_idx_vec, storage_expr.as_deref())?,
                    };
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
                let mut scanner = match needed.as_deref() {
                    Some(mask) => {
                        table.scan_needed(&column_idx_vec, mask, storage_expr.as_deref())?
                    }
                    None => table.scan(&column_idx_vec, storage_expr.as_deref())?,
                };

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
            let mut scanner = match needed {
                Some(mask) => table.scan_needed(&column_idx_vec, &mask, storage_expr.as_deref())?,
                None => table.scan(&column_idx_vec, storage_expr.as_deref())?,
            };

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
            // OPTIMIZATION: For window functions, check if we can use index-based fetching:
            // 1. PARTITION BY on indexed column -> fetch rows grouped by partition
            // 2. ORDER BY on indexed column -> fetch rows in sorted order
            let has_window = classification.has_window_functions;
            let has_agg = classification.has_aggregation;

            if has_window && !has_agg {
                // First try PARTITION BY optimization (bigger speedup, avoids O(n) hashing)
                if let Some(partition_col) = Self::extract_window_partition_info(stmt) {
                    let col_lower = partition_col.to_lowercase();
                    let schema = table.schema();
                    let pk_columns = schema.primary_key_columns();
                    let is_pk = pk_columns.len() == 1 && pk_columns[0].name_lower == col_lower;
                    let has_index = is_pk || table.get_index_on_column(&partition_col).is_some();

                    if has_index {
                        // OPTIMIZATION: If we have LIMIT without ORDER BY, use lazy partition fetching
                        // This avoids fetching all partitions when only a few are needed
                        // NOTE: Cannot use this optimization if there's a top-level ORDER BY
                        // because we need all rows to sort before applying LIMIT
                        let has_order_by = !stmt.order_by.is_empty();
                        // DISTINCT runs after the fetch, so it keeps the LIMIT out too
                        let distinct = stmt.distinct || !stmt.distinct_on.is_empty();
                        if !has_order_by
                            && !distinct
                            && blocks.is_none()
                            && !Self::windows_hold_subquery(stmt)
                        {
                            if let Some(limit_val) = Self::fetch_limit(stmt, ctx) {
                                // Use lazy partition fetching - returns early!
                                let result = self
                                    .execute_select_with_window_functions_lazy_partition(
                                        stmt,
                                        ctx,
                                        table.as_ref(),
                                        &all_columns,
                                        &partition_col,
                                        limit_val,
                                    )?;
                                if let Some(query_result) = result {
                                    let columns = CompactArc::new(query_result.columns().to_vec());
                                    return Ok((query_result, columns, false, None));
                                }
                                // Fall through when the optimization is unavailable.
                            }
                        }

                        // Regular path: Fetch rows grouped by partition (no hash grouping needed)
                        if let Some(grouped_data) =
                            table.collect_rows_grouped_by_partition(&partition_col)?
                        {
                            // Flatten rows and build partition map
                            let mut all_rows = RowVec::new();
                            let mut partition_map: rustc_hash::FxHashMap<
                                smallvec::SmallVec<[Value; 4]>,
                                Vec<usize>,
                            > = rustc_hash::FxHashMap::default();

                            for (partition_value, partition_rows) in grouped_data {
                                let start_idx = all_rows.len();
                                let partition_size = partition_rows.len();
                                // Extend RowVec with (row_id, Row) tuples
                                for item in partition_rows {
                                    all_rows.push(item);
                                }

                                // Build partition key and indices
                                let key: smallvec::SmallVec<[Value; 4]> =
                                    smallvec::smallvec![partition_value];
                                let indices: Vec<usize> =
                                    (start_idx..start_idx + partition_size).collect();
                                partition_map.insert(key, indices);
                            }

                            (
                                all_rows,
                                None,
                                Some(WindowPreGroupedState {
                                    partition_map,
                                    partition_column: col_lower.clone(),
                                }),
                            )
                        } else {
                            (table.collect_all_rows(None)?, None, None)
                        }
                    } else {
                        (table.collect_all_rows(None)?, None, None)
                    }
                }
                // Then try ORDER BY optimization (avoids sorting)
                else if let Some((col_name, ascending)) = Self::extract_window_order_info(stmt) {
                    let col_lower = col_name.to_lowercase();
                    let schema = table.schema();
                    let pk_columns = schema.primary_key_columns();
                    let is_pk = pk_columns.len() == 1 && pk_columns[0].name_lower == col_lower;
                    // Rows fetched in index order are taken as sorted, but the index
                    // orders NULLs its own way, so only a column without NULLs may
                    // skip the window's own sort
                    let no_nulls = is_pk
                        || schema
                            .columns
                            .iter()
                            .any(|c| c.name_lower == col_lower && !c.nullable);
                    let has_index =
                        no_nulls && (is_pk || table.get_index_on_column(&col_name).is_some());

                    if has_index {
                        // OPTIMIZATION: If we have LIMIT without top-level ORDER BY,
                        // push the limit down to fetch only needed rows.
                        // This is safe ONLY for window functions that don't depend on total row count.
                        // Safe: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, LAST_VALUE
                        // Unsafe: NTILE, PERCENT_RANK, CUME_DIST (need total count)
                        let has_order_by = !stmt.order_by.is_empty();
                        let is_window_safe =
                            Self::is_window_safe_for_limit_pushdown(stmt, &col_name, ascending);
                        let distinct = stmt.distinct || !stmt.distinct_on.is_empty();
                        let fetch_limit = if !has_order_by && !distinct && is_window_safe {
                            Self::fetch_limit(stmt, ctx).unwrap_or(usize::MAX)
                        } else {
                            usize::MAX
                        };

                        // Fetch rows in sorted order from the index (no re-fetch needed)
                        if let Some(sorted_rows) = table.collect_rows_ordered_by_index(
                            &col_name,
                            ascending,
                            !ascending,
                            fetch_limit,
                            0,
                        )? {
                            (
                                sorted_rows,
                                Some(WindowPreSortedState {
                                    column: col_lower,
                                    ascending,
                                }),
                                None,
                            )
                        } else {
                            (table.collect_all_rows(None)?, None, None)
                        }
                    } else {
                        (table.collect_all_rows(None)?, None, None)
                    }
                } else {
                    (table.collect_all_rows(None)?, None, None)
                }
            } else {
                (table.collect_all_rows(None)?, None, None)
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
            // Use optimized paths if rows were pre-fetched with index optimization
            let (result, layout) = if let Some(pregrouped) = window_pregrouped_state {
                // PARTITION BY optimization: rows are already grouped by partition
                self.execute_select_with_window_functions_pregrouped(
                    stmt,
                    ctx,
                    &rows,
                    &all_columns,
                    pregrouped,
                    blocks,
                )?
            } else if window_presorted_state.is_some() {
                // ORDER BY optimization: rows are already sorted
                self.execute_select_with_window_functions_presorted(
                    stmt,
                    ctx,
                    &rows,
                    &all_columns,
                    window_presorted_state,
                    blocks,
                )?
            } else {
                // Default path: no optimization
                self.execute_select_with_window_functions(stmt, ctx, &rows, &all_columns, blocks)?
            };
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, SelectShape::laid_out(layout)));
        }

        // Project rows according to SELECT expressions
        // Check if deferred projection is applicable (ORDER BY + LIMIT with simple columns)
        // This reduces allocations from O(matched_rows) to O(limit)
        let deferred_projection_info = self.get_deferred_projection_info(
            stmt,
            &all_columns_lower,
            &all_columns,
            classification,
        );

        let (projected_rows, output_columns, shape) = match deferred_projection_info {
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
                )
            }
            _ => {
                let (projected_rows, output_columns, layout) = self.project_scanned_rows(
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
                // Clone rows for cache (cache needs Vec<Row>)
                let rows_for_cache: Vec<Row> = projected_rows.rows().cloned().collect();
                self.semantic_cache.insert(
                    table_name,
                    all_columns.to_vec(),
                    rows_for_cache,
                    Some(where_expr.clone()),
                    stamp,
                );
            }
        }

        let output_columns = CompactArc::new(output_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, false, Some(shape)))
    }

    /// Execute a JOIN source. A GROUP BY + LIMIT join fetches its left side in
    /// a limited chunk; when the final rows stop short of the limit and the
    /// chunk was full, the join runs again with a larger chunk
    fn execute_join_source(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
    ) -> SelectResult {
        let reduction = std::cell::Cell::new(None);
        let mut left_cap = None;
        loop {
            let (result, columns, flag, deferred) = self.execute_join_source_capped(
                join_source,
                stmt,
                ctx,
                classification,
                compiled,
                left_cap,
                &reduction,
            )?;
            let pass = match reduction.get() {
                Some(pass) if pass.chunk_full => pass,
                _ => return Ok((result, columns, flag, deferred)),
            };
            let (result, short) = match result.exact_len() {
                Some(len) => (result, len < pass.limit),
                None => {
                    let result_columns = result
                        .columns_arc()
                        .unwrap_or_else(|| CompactArc::new(result.columns().to_vec()));
                    let rows = Self::materialize_result_arc(result)?;
                    let short = rows.len() < pass.limit;
                    let result: Box<dyn QueryResult> = Box::new(
                        ExecutorResult::with_arc_columns_shared_rows(result_columns, rows),
                    );
                    (result, short)
                }
            };
            if !short {
                return Ok((result, columns, flag, deferred));
            }
            left_cap = Some(pass.cap.saturating_mul(2).min(i64::MAX as usize));
        }
    }

    /// The join programs `slot` holds for the current schema epoch; a slot
    /// holding none for it holds an empty set from now on
    fn prepared_join(
        &self,
        slot: &std::sync::RwLock<CompiledExecution>,
    ) -> Option<std::sync::Arc<PreparedJoin>> {
        let epoch = self.engine.schema_epoch();
        let current = |guard: &CompiledExecution| match guard {
            CompiledExecution::PreparedJoin(held) if held.cached_epoch == epoch => {
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
        let held = std::sync::Arc::new(PreparedJoin::new(epoch));
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

    /// Whether a join of two base tables holds subqueries in its ON or its
    /// WHERE, all of the forms the prepared blocks run, or why not
    pub(crate) fn join_blocks_family(
        &self,
        stmt: &SelectStatement,
        join_source: &JoinTableSource,
        join_type: &str,
        (left_alias, right_alias): (Option<&str>, Option<&str>),
        schemas: (Option<&Schema>, Option<&Schema>),
    ) -> std::result::Result<(), OutOfFamily> {
        let plain_table =
            |side: &Expression| matches!(side, Expression::TableSource(ts) if ts.as_of.is_none());
        if !plain_table(&join_source.left) || !plain_table(&join_source.right) {
            return Err("a side that is not a table read without AS OF");
        }
        if schemas.0.is_none() || schemas.1.is_none() {
            return Err("a side that names a CTE");
        }
        let on = join_source.condition.as_deref();
        let on_subqueries = on.is_some_and(Self::has_subqueries);
        if let Some(on) = on.filter(|_| on_subqueries) {
            self.join_where_family(on)?;
        }
        let output_subqueries = self.join_output_family(stmt, join_source, join_type)?;
        let Some(where_clause) = stmt.where_clause.as_deref() else {
            return match on_subqueries || output_subqueries {
                true => Ok(()),
                false => Err("no ON, WHERE, SELECT list or ORDER BY holds a subquery"),
            };
        };
        let (left, right, cross) =
            split_join_where(where_clause, left_alias, right_alias, schemas, join_type);
        // Each part as its site reads it, the SELECT aliases it names opened
        // over the joined columns the sides lay out
        let mut columns = Vec::new();
        for (alias, schema) in [(left_alias, schemas.0), (right_alias, schemas.1)] {
            if let (Some(alias), Some(schema)) = (alias, schema) {
                columns.extend(schema.columns.iter().map(|c| format!("{alias}.{}", c.name)));
            }
        }
        let opened = |part: Option<Expression>| {
            part.map(|part| Self::with_select_aliases(stmt, &part, &columns).unwrap_or(part))
        };
        let parts = [opened(left), opened(right), opened(cross)];
        let mut subqueries = on_subqueries || output_subqueries;
        for part in parts.iter().flatten() {
            if Self::has_subqueries(part) {
                self.join_where_family(part)?;
                subqueries = true;
            }
        }
        match subqueries {
            true => Ok(()),
            false => Err("no ON, WHERE, SELECT list or ORDER BY holds a subquery"),
        }
    }

    /// Whether a join's output groups its rows, so the grouped blocks or the
    /// general path run its subqueries after the join, never its join blocks
    fn grouped_join_output(stmt: &SelectStatement) -> bool {
        let classification = get_classification(stmt);
        classification.has_aggregation && !classification.has_window_functions
    }

    /// Whether the SELECT list or the ORDER BY of a join holds subqueries,
    /// all of the forms the prepared blocks run over the joined rows, or
    /// why the blocks cannot run them
    fn join_output_family(
        &self,
        stmt: &SelectStatement,
        join_source: &JoinTableSource,
        join_type: &str,
    ) -> std::result::Result<bool, OutOfFamily> {
        if Self::grouped_join_output(stmt) {
            return Ok(false);
        }
        let classification = get_classification(stmt);
        let items = stmt
            .columns
            .iter()
            .filter(|c| !matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)))
            .map(unaliased_item);
        let keys = stmt.order_by.iter().map(|ob| &ob.expression);
        let mut found = false;
        for expr in items.chain(keys) {
            if Self::has_subqueries(expr) {
                self.join_where_family(expr)?;
                found = true;
            }
        }
        if !found {
            return Ok(false);
        }
        if join_type.contains("NATURAL") || !join_source.using_columns.is_empty() {
            return Err("a NATURAL or USING join whose output holds a subquery");
        }
        if classification.has_aggregation
            || classification.has_window_functions
            || !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
        {
            return Err("a grouped statement whose output holds a subquery");
        }
        let aliases: Vec<&str> = stmt
            .columns
            .iter()
            .filter_map(|c| match c {
                Expression::Aliased(a) => Some(a.alias.value_lower.as_str()),
                _ => None,
            })
            .collect();
        if stmt.order_by.iter().any(|ob| {
            Self::has_subqueries(&ob.expression)
                && names_any_within(&ob.expression, &|name| aliases.contains(&name))
        }) {
            return Err("an ORDER BY subquery that names a SELECT alias");
        }
        Ok(true)
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
    /// one side, `side_filters`, with the views' rows preparing read; none
    /// without them
    #[allow(clippy::too_many_arguments)]
    fn statement_blocks(
        &self,
        prepared: Option<&PreparedJoin>,
        stmt: &SelectStatement,
        (join_source, join_type): (&JoinTableSource, &str),
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        (left_alias, right_alias): (Option<&str>, Option<&str>),
        (left_schema, right_schema): (Option<&Schema>, Option<&Schema>),
        ctx: &ExecutionContext,
    ) -> Result<
        Option<(
            std::sync::Arc<JoinBlocks>,
            super::prepared_select::FirstReads,
        )>,
    > {
        let (
            Some(prepared),
            Some(left_alias),
            Some(right_alias),
            Some(left_schema),
            Some(right_schema),
        ) = (prepared, left_alias, right_alias, left_schema, right_schema)
        else {
            return Ok(None);
        };
        prepared.join_blocks(|| {
            // The WHERE's SELECT aliases open over the columns the sides lay
            // out, as the decision to run blocks read them
            let columns: Vec<String> = [(left_alias, left_schema), (right_alias, right_schema)]
                .iter()
                .flat_map(|(alias, schema)| {
                    schema
                        .columns
                        .iter()
                        .map(move |c| format!("{alias}.{}", c.name))
                })
                .collect();
            // The equalities of a NATURAL join or a USING list stand for its ON
            let (left, right) = columns.split_at(left_schema.columns.len());
            // Each column by its side and its own name, which may hold a dot
            let natural =
                natural_join_condition(join_type, join_source, left, right, |right, i| {
                    let (alias, schema) = match right {
                        true => (right_alias, right_schema),
                        false => (left_alias, left_schema),
                    };
                    Expression::QualifiedIdentifier(QualifiedIdentifier {
                        qualifier: Identifier::new(
                            Token::new(TokenType::Identifier, alias, Position::default()),
                            alias,
                        ),
                        name: Identifier::new(
                            Token::new(
                                TokenType::Identifier,
                                schema.columns[i].name.as_str(),
                                Position::default(),
                            ),
                            schema.columns[i].name.as_str(),
                        ),
                    })
                })
                .0;
            let on = natural.as_ref().or(join_source.condition.as_deref());
            let opened = cross.map(|cross| {
                Self::with_select_aliases(stmt, cross, &columns).unwrap_or_else(|| cross.clone())
            });
            // A side's conditions come without their qualifier; it is put
            // back, as the index join puts it back on its inner side's
            let [left_filter, right_filter] = side_filters;
            let left_filter = left_filter.map(|f| add_table_qualifier(f, left_alias));
            let right_filter = right_filter.map(|f| add_table_qualifier(f, right_alias));
            let sides = [
                (
                    SmartString::from(left_alias.to_lowercase().as_str()),
                    left_schema,
                ),
                (
                    SmartString::from(right_alias.to_lowercase().as_str()),
                    right_schema,
                ),
            ];
            // The SELECT items and ORDER BY keys whose subqueries the blocks run
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
            self.prepare_join_blocks(
                on,
                opened.as_ref(),
                [left_filter.as_ref(), right_filter.as_ref()],
                [&select, &keys],
                sides,
                columns.len(),
                ctx,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_join_source_capped(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        compiled: Option<&std::sync::RwLock<CompiledExecution>>,
        left_cap_override: Option<usize>,
        reduction: &std::cell::Cell<Option<ReductionPass>>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

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
            .and_then(|slot| self.prepared_join(slot));
        let output_subqueries = stmt.columns.iter().any(Self::has_subqueries)
            || stmt
                .order_by
                .iter()
                .any(|ob| Self::has_subqueries(&ob.expression));
        let blocks_where =
            (classification.where_has_subqueries || on_subqueries || output_subqueries)
                && plan.as_ref().is_some_and(|plan| {
                    plan.blocks_where(|| {
                        self.join_blocks_family(
                            stmt,
                            join_source,
                            &join_type,
                            (left_alias.as_deref(), right_alias.as_deref()),
                            (left_schema.as_deref(), right_schema.as_deref()),
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
        let prepared = plan.filter(|_| {
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
        let where_for_join: Option<Expression> = match &stmt.where_clause {
            Some(where_clause) if classification.where_has_subqueries && !blocks_where => {
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
        // A parent row's column named through the parent's table is bound
        // before the predicates are split, or the side a predicate is
        // pushed to would read the bare name it falls back to as its own.
        // A predicate holding a subquery is left for the per-row path
        let where_for_join = match (where_for_join, ctx.outer_row()) {
            (Some(where_clause), Some(outer)) => {
                let mut inner = Vec::new();
                super::utils::collect_table_aliases(&join_source.left, &mut inner);
                super::utils::collect_table_aliases(&join_source.right, &mut inner);
                let inner: Vec<&str> = inner.iter().map(String::as_str).collect();
                let scope = super::utils::InnerScope {
                    tables: &inner,
                    schema: None,
                };
                let bound: Vec<Expression> = flatten_and_predicates(&where_clause)
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
            (where_clause, _) => where_clause,
        };
        let (left_filter, right_filter, cross_filter) = match where_for_join.as_ref() {
            Some(where_clause) => split_join_where(
                where_clause,
                left_alias.as_deref(),
                right_alias.as_deref(),
                (left_schema.as_deref(), right_schema.as_deref()),
                &join_type,
            ),
            None => (None, None, None),
        };
        // The statement's blocks, and the one run of them every site calls
        let blocks = match blocks_where {
            true => self.statement_blocks(
                prepared.as_deref(),
                stmt,
                (join_source, join_type.as_str()),
                cross_filter.as_ref(),
                [left_filter.as_ref(), right_filter.as_ref()],
                (left_alias.as_deref(), right_alias.as_deref()),
                (left_schema.as_deref(), right_schema.as_deref()),
                ctx,
            )?,
            false => None,
        };
        let (blocks, first) = blocks.unzip();
        let blocks_run = match blocks.as_deref() {
            Some(blocks) => Some(
                self.join_runner(blocks, &first.unwrap_or_default(), ctx)?
                    .ok_or_else(|| {
                        Error::internal("a table's columns changed while the statement prepared")
                    })?,
            ),
            None => None,
        };
        let statement_run = match (blocks.as_deref(), blocks_run.as_ref()) {
            (Some(blocks), Some(run)) => Some(StatementRun { blocks, run }),
            _ => None,
        };
        let left_read = SideRead {
            filter: left_filter.as_ref(),
            run: statement_run.map(|run| (run, 0)),
        };
        let right_read = SideRead {
            filter: right_filter.as_ref(),
            run: statement_run.map(|run| (run, 1)),
        };
        let where_blocks = statement_run.map(|run| run.site(false));
        let where_blocks = where_blocks
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        let pair_blocks = statement_run.map(|run| run.site(true));
        let pair_blocks = pair_blocks
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        // The result a held output builds, its blocks told how many rows ask
        // and its member tests bound to the sets the run read
        let output_result = |output: &JoinOutput, rows: RowVec, applied: bool| {
            let bound = blocks_run.as_ref().and_then(|run| {
                run.ask_rows(Some(rows.len()));
                output.bound_to(run)
            });
            let output = bound.as_ref().unwrap_or(output);
            Self::join_output_result(output, rows, applied, ctx, where_blocks)
        };
        // A WHERE that keeps no row joins no rows whatever the join type; the
        // sides run with no rows only to name their columns
        let never = where_for_join
            .as_ref()
            .filter(|_| !classification.where_has_subqueries)
            .is_some_and(|w| true_for_no_row(w, ctx));

        // Semi-join reduction optimization for LEFT JOIN + GROUP BY + LIMIT
        // Pattern: LEFT JOIN + GROUP BY on left columns only + LIMIT N + no ORDER BY
        // Optimization: limit left side first, filter right side with IN clause (uses index)
        // This reduces materialization from O(L + R) to O(N + N*avg_matches)
        let left_table = match join_source.left.as_ref() {
            Expression::TableSource(ts) => Some(ts.name.value.as_str()),
            _ => None,
        };
        // Outside the blocks a correlated WHERE runs per row on the general path
        let per_row_where = correlated_where && statement_run.is_none();
        let semijoin_limit = if per_row_where {
            None
        } else {
            self.get_semijoin_reduction_limit(
                &join_type,
                stmt,
                left_alias.as_deref(),
                left_table,
                &join_source.condition,
                classification,
            )
        };

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
            if let Some(result) = self.try_grouped_index_join(
                stmt,
                ctx,
                join_source,
                &join_type,
                left_alias.as_deref(),
                right_alias.as_deref(),
                left_read,
                right_filter.as_ref(),
                cross_filter.as_ref(),
                prepared.as_deref(),
                &plan,
            )? {
                return Ok(result);
            }
            // Semi-join reduction for INNER/LEFT JOIN + GROUP BY: fetch the left
            // side with the limit, then the right side through an IN filter on
            // the join key (uses the index)
            let SemijoinReduction {
                cap,
                limit,
                left_key_col,
                right_key_col,
                ..
            } = plan;
            let left_cap = left_cap_override.unwrap_or(cap);
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::join_side_ran();
            let (left_result, left_cols, left_bounded, _) =
                self.read_join_side(&join_source.left, ctx, left_read, None, Some(left_cap), 0)?;
            let left_rows = Self::materialize_result_arc(left_result)?;
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
                cap: left_cap,
                chunk_full: left_bounded && left_rows.len() == left_cap,
                limit,
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
            let picked = if per_row_where {
                None
            } else {
                let ctes = CteNames::Context(ctx);
                let left_is_table = left_schema.is_some();
                let right_is_table = right_schema.is_some();
                let unswapped = self.check_index_nested_loop_opportunity(
                    ctes,
                    &join_source.right,
                    on_condition,
                    &join_type,
                    left_alias.as_deref(),
                    right_alias.as_deref(),
                    pushed_limit.is_some(),
                    small_outer,
                    !left_is_table,
                );
                pick_index_join_side(
                    &join_type,
                    right_is_table,
                    right_filter.is_some(),
                    left_filter.is_some(),
                    unswapped,
                    || {
                        self.check_index_nested_loop_opportunity(
                            ctes,
                            &join_source.left,
                            on_condition,
                            &join_type,
                            right_alias.as_deref(),
                            left_alias.as_deref(),
                            pushed_limit.is_some(),
                            small_outer,
                            !right_is_table,
                        )
                    },
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
                        let name = column.rsplit('.').next().unwrap_or(column);
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
                                    let name = inner_col.rsplit('.').next().unwrap_or(&inner_col);
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
                        let outer_unqualified = outer_col_lower
                            .rfind('.')
                            .map(|p| &outer_col_lower[p + 1..])
                            .unwrap_or(&outer_col_lower);
                        outer_cols_lower.iter().position(|c_lower| {
                            let c_unqualified = c_lower
                                .rfind('.')
                                .map(|p| &c_lower[p + 1..])
                                .unwrap_or(c_lower);
                            c_unqualified == outer_unqualified
                        })
                    });

                if let Some(outer_idx) = outer_key_idx {
                    // Get inner table for schema and row fetching
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
                        .map(|col| format!("{}.{}", inner_alias, col.name))
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
                    let inner_key_idx = inner_cols.iter().position(|c| {
                        c.rsplit('.')
                            .next()
                            .is_some_and(|name| name.eq_ignore_ascii_case(&inner_col))
                    });
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
                        let rest = on.and_then(|on| {
                            index_join_residual(
                                on,
                                outer_names,
                                inner_names,
                                outer_idx,
                                inner_key_idx,
                            )
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
                    // Eligibility: no cross filter, no ORDER BY,
                    // no aggregation/window, simple column references only
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

                    // A cross-table WHERE filter counts toward LIMIT, so it is
                    // applied per joined row rather than after collection
                    let cross_row_filter = match cross_filter {
                        Some(ref cross) => Some(
                            RowFilter::from_program(self.join_where(
                                prepared.as_deref(),
                                WhereSite::IndexJoin,
                                stmt,
                                cross,
                                &all_columns,
                                blocks.as_deref(),
                            )?)
                            .with_context(ctx),
                        ),
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
                            let outer_op: Box<dyn Operator> = Box::new(QueryResultOperator::new(
                                outer_result,
                                outer_cols.clone(),
                            ));
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
                                inner_cols.iter().map(ColumnInfo::new).collect(),
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
                                let projected_schema: Vec<ColumnInfo> =
                                    proj.output_columns.iter().map(ColumnInfo::new).collect();
                                op = op.with_projection(proj.columns.clone(), projected_schema);
                            }
                            Self::collect_join_rows(
                                &mut op,
                                join_limit,
                                cross_row_filter.as_ref(),
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
                            Box::new(QueryResultOperator::new(outer_result, outer_cols.clone()));
                        JoinPath::BatchIndexNestedLoop.taken();
                        let mut op = BatchIndexNestedLoopJoinOperator::new(
                            outer_op,
                            inner_table,
                            inner_cols.iter().map(ColumnInfo::new).collect(),
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
                            let projected_schema: Vec<ColumnInfo> =
                                proj.output_columns.iter().map(ColumnInfo::new).collect();
                            op = op.with_projection(proj.columns.clone(), projected_schema);
                        }
                        Self::collect_join_rows(
                            &mut op,
                            None,
                            cross_row_filter.as_ref(),
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
                                    keys.push(vm.execute_cow(p, &exec).ok()?);
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

                    let held = if has_agg || has_window || projection_pushdown.is_some() {
                        None
                    } else {
                        self.held_join_output(
                            prepared.as_deref(),
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
                        return Ok((Box::new(result), output_columns, order_limit_applied, None));
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

            let streaming_limit = if !join_type.contains("FULL")
                && !join_type.contains("LEFT")
                && !join_type.contains("RIGHT")
                && !classification.has_order_by
                && !classification.has_group_by
                && !classification.has_aggregation
                && !classification.has_window_functions
                && !stmt.distinct
                && stmt.distinct_on.is_empty()
                && cross_filter.is_none()
            {
                // OFFSET rows are skipped after collection, so the
                // early-termination budget is LIMIT + OFFSET, for a LIMIT of
                // at most 100
                row_counts(stmt, ctx).ok().and_then(|counts| {
                    let limit = counts.limit.filter(|limit| *limit <= 100)?;
                    Some(limit.saturating_add(counts.offset) as u64)
                })
            } else {
                None
            };

            // Check if we have equality keys for hash join using AST analysis
            let has_equality_keys_for_streaming = join_source
                .condition
                .as_ref()
                .map(|cond| Self::has_equality_condition(cond))
                .unwrap_or(false);

            // Use streaming path if eligible; sides it ran but could not join
            // by hash go on to the standard path
            let streamed = 'streaming: {
                let Some(limit) = streaming_limit else {
                    break 'streaming None;
                };
                if has_equality_keys_for_streaming {
                    // Estimate cardinalities to choose optimal build side (smaller = build)
                    let left_card = self
                        .estimate_table_expr_cardinality(&join_source.left, left_filter.as_ref());
                    let right_card = self
                        .estimate_table_expr_cardinality(&join_source.right, right_filter.as_ref());
                    let build_left = left_card < right_card;

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
                    let (left_key_indices, right_key_indices, _) = if let Some(cond) = on_condition
                    {
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
                    let build_schema =
                        self.base_table_side_schema(build_expr, CteNames::Context(ctx));
                    let probe_schema =
                        self.base_table_side_schema(probe_expr, CteNames::Context(ctx));

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
                        let name = build_cols[idx]
                            .rsplit('.')
                            .next()
                            .unwrap_or(&build_cols[idx]);
                        build_schema
                            .as_ref()
                            .and_then(|s| s.get_column_type(name))
                            .is_some()
                    };
                    let one_key_built = match build_key_indices.as_slice() {
                        &[build_idx] if !typed(build_idx) => {
                            let mut builder = wants_bloom.then(new_bloom);
                            let (table, kinds) = JoinHashTable::build_one_key(
                                &build_rows,
                                build_idx,
                                builder.as_mut(),
                            );
                            Some((table, builder, kinds))
                        }
                        _ => None,
                    };
                    let mut cross_key = None;
                    let mut unverified = false;
                    for (&build_idx, &probe_idx) in build_key_indices.iter().zip(&probe_key_indices)
                    {
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
                        let name = probe_column.rsplit('.').next().unwrap_or(probe_column);
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
                        Box::new(QueryResultOperator::new(probe_result, probe_cols.clone()));

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
                        prepared: prepared.as_deref(),
                        blocks: statement_run,
                    };

                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::streaming_join_streamed();
                    let join_result = join_executor.execute_streaming(streaming_request)?;
                    let held = self.held_join_output(
                        prepared.as_deref(),
                        stmt,
                        &all_cols,
                        false,
                        blocks.as_deref(),
                        ctx,
                    )?;
                    if let Some(output) = held.as_deref().and_then(Option::as_ref) {
                        return output_result(output, join_result.rows, false);
                    }

                    // Project and return results
                    let projected_rows = self.project_rows_with_alias(
                        &stmt.columns,
                        join_result.rows,
                        &all_cols,
                        None,
                        ctx,
                        None,
                    )?;
                    let output_columns = CompactArc::new(self.get_output_column_names(
                        &stmt.columns,
                        &all_cols,
                        None,
                    ));

                    let result = ExecutorResult::with_arc_columns(
                        CompactArc::clone(&output_columns),
                        projected_rows,
                    );
                    return Ok((Box::new(result), output_columns, false, None));
                }
                None
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
        let (natural_join_cond, excluded_column_indices, join_col_renames) = natural_join_condition;

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
            prepared: prepared.as_deref(),
            blocks: statement_run,
        };

        let join_result = join_executor.execute(join_request)?;
        let result_rows = join_result.rows;

        // Apply remaining WHERE clause if present (after filter pushdown)
        // IMPORTANT: When predicates were pushed to left/right (left_filter or right_filter is Some),
        // we should ONLY apply cross_filter (predicates that reference both tables).
        // If cross_filter is None and any pushdown happened, we don't need post-join filtering.
        // Only use stmt.where_clause when NO pushdown happened at all.
        let did_any_pushdown = left_filter.is_some() || right_filter.is_some();
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
                        .filter_map(|c| c.split_once('.').map(|(q, _)| q.to_lowercase()))
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
                    prepared.as_deref(),
                    WhereSite::AfterJoin,
                    stmt,
                    &where_expr,
                    &all_columns,
                    blocks.as_deref(),
                )?;
                let where_filter = RowFilter::from_program(program).with_context(ctx);
                if let Some(run) = &blocks_run {
                    run.ask_rows(Some(result_rows.len()));
                }
                let mut filtered = RowVec::with_capacity(result_rows.len());
                for (id, row) in result_rows {
                    if where_filter.matches_checked_with(&row, where_blocks)? {
                        filtered.push((id, row));
                    }
                }
                filtered
            }
        } else {
            result_rows
        };

        // For NATURAL JOIN and JOIN USING, filter out duplicate columns from right side
        // when SELECT * is used. We need to remove these columns from both all_columns
        // and the row values to maintain consistency.
        // Also rename join columns to unqualified names per SQL standard.
        let (final_columns, final_rows) = if !excluded_column_indices.is_empty() {
            // Check if SELECT * is used (need to deduplicate columns)
            let has_star = stmt
                .columns
                .iter()
                .any(|c| matches!(c, Expression::Star(_)));

            if has_star {
                // Create a set for O(1) lookup
                let excluded_set: FxHashSet<usize> =
                    excluded_column_indices.iter().copied().collect();

                // Create rename map for join columns
                let rename_map: FxHashMap<usize, String> =
                    join_col_renames.iter().cloned().collect();

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
                // An explicit select list finds the join column by its
                // bare name too, as a column of its own beside the
                // qualified ones, so `a`, `x.a` and `x.*` all see it
                let mut columns = all_columns.clone();
                let mut rows = filtered_rows;
                for (idx, base_name) in &join_col_renames {
                    columns.push(base_name.clone());
                    for (_, row) in rows.iter_mut() {
                        let value = row.get(*idx).cloned().unwrap_or_else(Value::null_unknown);
                        row.push(value);
                    }
                }
                (columns, rows)
            }
        } else {
            (all_columns.clone(), filtered_rows)
        };

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
            prepared.as_deref(),
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
    ) -> SelectResult {
        // classification is passed from caller for the outer stmt
        // Note: The inner subquery will get its own classification via execute_select

        // Execute subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let result = self.execute_select(&subquery_source.subquery, &subquery_ctx)?;
        let columns = result.columns().to_vec();

        // OPTIMIZATION: For simple GROUP BY aggregation without WHERE clause, try streaming
        // directly to aggregation HashMap without materializing all rows first.
        // This reduces memory allocations from O(N) to O(groups).
        if stmt.where_clause.is_none()
            && classification.has_aggregation
            && !classification.has_window_functions
        {
            if let Some(agg_result) =
                self.try_streaming_derived_table_aggregation(result, stmt, ctx, classification)?
            {
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
            // Streaming not applicable - need to re-execute the subquery
            // (the streaming function consumed the result iterator)
            let result = self.execute_select(&subquery_source.subquery, &subquery_ctx)?;

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
                let where_filter = RowFilter::new(where_clause, &columns)?.with_context(ctx);

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
        let (projected_rows, output_columns, layout) =
            if self.order_by_needs_extra_columns(stmt, &columns) {
                let (projected_rows, extra_names, layout) = self.project_rows_with_order_by(
                    &stmt.columns,
                    &stmt.order_by,
                    &stmt.distinct_on,
                    filtered_rows,
                    &CompactArc::new(columns.clone()),
                    subquery_alias,
                    ctx,
                )?;
                let mut output_columns =
                    self.get_output_column_names(&stmt.columns, &columns, subquery_alias);
                output_columns.extend(extra_names);
                (
                    projected_rows,
                    CompactArc::new(output_columns),
                    Some(layout),
                )
            } else {
                let projected_rows = self.project_rows_with_alias(
                    &stmt.columns,
                    filtered_rows,
                    &columns,
                    None,
                    ctx,
                    subquery_alias,
                )?;
                let output_columns = CompactArc::new(self.get_output_column_names(
                    &stmt.columns,
                    &columns,
                    subquery_alias,
                ));
                let layout = Self::output_layout(stmt, &columns, subquery_alias);
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

    /// Parse a view's SQL query and return the cached Statement.
    /// Uses the query cache to avoid re-parsing the same view query multiple times.
    /// Returns Arc<Statement> - caller should extract SelectStatement via pattern match.
    pub(super) fn parse_view_statement(&self, view_query: &str) -> Result<Arc<Statement>> {
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
            return Ok(cached.statement);
        }

        // Parse the view's query
        let statements = crate::parser::parse_sql(view_query)
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

        // Wrap in Arc and cache (single allocation, no clone)
        let stmt_arc = Arc::new(statements.into_iter().next().unwrap());

        // Cache the parsed statement for future use
        self.query_cache
            .put(view_query, stmt_arc.clone(), ParamSlots::default());

        Ok(stmt_arc)
    }

    /// The rows of a view's own query, one view deeper than `ctx`
    pub(super) fn execute_view_body(
        &self,
        view_def: &ViewDefinition,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Check view depth to prevent stack overflow from deeply nested views
        let depth = ctx.view_depth();
        if depth >= MAX_VIEW_DEPTH {
            return Err(Error::InvalidArgument(format!(
                "Maximum view nesting depth ({}) exceeded",
                MAX_VIEW_DEPTH
            )));
        }

        // Parse the view's query using cache (returns Arc<Statement>, no clone)
        let view_stmt = self.parse_view_statement(&view_def.query)?;
        let view_select = match view_stmt.as_ref() {
            Statement::Select(s) => s,
            _ => unreachable!("parse_view_statement validates this is a SELECT"),
        };

        // Execute the view's query with incremented depth
        let nested_ctx = ctx.with_incremented_view_depth();
        self.execute_select(view_select, &nested_ctx)
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
                let filter = RowFilter::new(where_clause, &view_columns)?.with_context(ctx);
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
            let (projected_rows, extra_names, layout) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                rows,
                &CompactArc::new(view_columns.clone()),
                view_alias.as_deref(),
                ctx,
            )?;
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &view_columns, None);
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
        }

        // Determine if we have any complex expressions (not just column references)
        // If all expressions are simple column references, use fast StreamingProjectionResult
        // Otherwise, use ExprMappedResult with pre-compiled expression evaluation
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
                        } else {
                            return Err(Error::ColumnNotFound(id.value.to_string()));
                        }
                    } else {
                        // Complex expression in alias - need ExprMappedResult
                        has_complex_expressions = true;
                    }
                }
                _ => {
                    // Any other expression type requires evaluation
                    has_complex_expressions = true;
                    // We'll compute the output column name later
                    let col_name = Self::get_expression_column_name(col);
                    output_columns.push(col_name);
                }
            }
        }

        // Apply projection based on complexity
        // OPTIMIZATION: ExprMappedResult owns the Evaluator and reuses it for each row,
        // avoiding 7 HashMap allocations per row that the closure-based approach had.
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
                        final_output_columns.push(Self::get_expression_column_name(col));
                    }
                }
            }

            Box::new(
                ExprMappedResult::new(
                    result,
                    stmt.columns.clone(),
                    final_output_columns.clone(),
                    view_alias,
                )?
                .with_context(ctx),
            )
        } else {
            // Simple column references only: use fast StreamingProjectionResult
            Box::new(StreamingProjectionResult::new(
                result,
                column_indices,
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
                        cols.push(Self::get_expression_column_name(col));
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
                let name_lower = name.to_lowercase();
                let mut qualified = String::with_capacity(alias_lower.len() + 1 + name_lower.len());
                qualified.push_str(&alias_lower);
                qualified.push('.');
                qualified.push_str(&name_lower);
                col_index_map.insert(qualified, i);
            }
        }

        // OPTIMIZATION: Pre-create RowFilters outside the loop if WHERE clause exists
        let (where_filter, qualified_filter) = if let Some(ref where_clause) = stmt.where_clause {
            // Pre-compute qualified column names once
            let qualified_cols: Vec<String> = column_names
                .iter()
                .map(|c| format!("{}.{}", table_alias, c))
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
        for (i, col_expr) in stmt.columns.iter().enumerate() {
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
                Expression::Aliased(a) => {
                    projected_columns.push(a.alias.value.to_string());
                }
                Expression::Identifier(id) => {
                    projected_columns.push(id.value.to_string());
                }
                Expression::QualifiedIdentifier(qi) => {
                    projected_columns.push(qi.name.value.to_string());
                }
                _ => {
                    projected_columns.push(format!("expr{}", i + 1));
                }
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
                // Build qualified key once
                let mut key = String::with_capacity(
                    qi.qualifier.value_lower.len() + 1 + qi.name.value_lower.len(),
                );
                key.push_str(&qi.qualifier.value_lower);
                key.push('.');
                key.push_str(&qi.name.value_lower);
                qualified_col_indices.push(
                    col_index_map
                        .get(&key)
                        .copied()
                        .or_else(|| col_index_map.get(qi.name.value_lower.as_str()).copied()),
                );
            } else {
                qualified_col_indices.push(None);
            }
        }

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
                    // Get the alias for column prefixing
                    let table_alias = ts
                        .alias
                        .as_ref()
                        .map(|a| a.value.clone())
                        .unwrap_or_else(|| ts.name.value.clone());
                    let qualified_columns: Vec<String> = columns
                        .iter()
                        .map(|col| format!("{}.{}", table_alias, col))
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
                        .map(|col| format!("{}.{}", table_alias, col))
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
                    .map(|col| format!("{}.{}", table_alias, col))
                    .collect();

                Ok((result, qualified_columns))
            }
            Expression::JoinSource(js) => {
                let select_all = SelectStatement {
                    param_count: 0,
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
                    distinct_on: vec![],
                    columns: vec![Expression::Star(StarExpression {
                        token: dummy_token("*", TokenType::Punctuator),
                    })],
                    with: None,
                    table_expr: Some(Box::new(Expression::JoinSource(js.clone()))),
                    where_clause: None,
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
                    self.execute_join_source(js, &select_all, ctx, &classification, None)?;
                Ok((result, columns.to_vec()))
            }
            Expression::SubquerySource(ss) => {
                // Execute subquery with incremented depth to avoid creating new TimeoutGuard
                let subquery_ctx = ctx.with_incremented_query_depth();
                let result = self.execute_select(&ss.subquery, &subquery_ctx)?;
                let columns = result.columns().to_vec();

                // Prefix column names with subquery alias (required for proper ON condition resolution)
                // Without this, ON a.id = b.id cannot resolve qualified column names
                let qualified_columns: Vec<String> = if let Some(alias) = &ss.alias {
                    columns
                        .iter()
                        .map(|col| format!("{}.{}", alias.value, col))
                        .collect()
                } else {
                    columns.clone()
                };

                // Apply filter to subquery result if present
                // This is needed when WHERE clause conditions are pushed down to subquery sources
                if let Some(filter_expr) = filter {
                    // Use qualified_columns for filter column resolution since WHERE clause
                    // has qualified names like "ds.avg_salary" not just "avg_salary"
                    let row_filter =
                        RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
                    // Materialize and filter - materialized is RowVec = Vec<(i64, Row)>
                    let mut materialized = Self::materialize_result(result)?;
                    row_filter.retain_checked(&mut materialized)?;
                    let filtered_result: Box<dyn QueryResult> =
                        Box::new(super::result::ExecutorResult::new(columns, materialized));
                    return Ok((filtered_result, qualified_columns));
                }

                Ok((result, qualified_columns))
            }
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
                Ok((result, columns.to_vec()))
            }
            Expression::FunctionTableSource(tvf_source) => {
                Self::execute_tvf_for_join(tvf_source, ctx, filter, None, 0)
                    .map(|(result, columns, _)| (result, columns))
            }
            _ => Err(Error::NotSupported(
                "Unsupported table expression type".to_string(),
            )),
        }
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
        if let (Some((run, index)), Expression::TableSource(ts)) = (side.run, expr) {
            let alias = ts.alias.as_ref().unwrap_or(&ts.name).value.as_str();
            let table = (ts.name.value_lower.as_str(), alias);
            let read =
                self.join_side_rows(run.blocks, run.run, index, table, ctx, extra, limit, offset)?;
            if let Some((result, token)) = read {
                let columns = result
                    .columns()
                    .iter()
                    .map(|c| format!("{alias}.{c}"))
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
        let fetched = self.execute_table_expression_with_filter_limit(
            expr, ctx, filter, limit, offset, read_out,
        )?;
        if let Some((run, index)) = side.run {
            let (_, schema) =
                read.ok_or_else(|| Error::internal("a join side read without its schema"))?;
            run.blocks.side_fits(index, &schema)?;
        }
        Ok(fetched)
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
            return Self::execute_tvf_for_join(tvf_source, ctx, filter, row_limit, row_offset)
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
                .map(|col| format!("{}.{}", table_alias, col))
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
            // The views' rows preparing read, which this run opens
            let mut first = super::prepared_select::FirstReads::new();
            let mut build = || {
                let mut leaves = Vec::new();
                let mut left_leaves = 0;
                if let Some(Expression::JoinSource(join)) = stmt.table_expr.as_deref() {
                    self.join_leaves(&join.left, ctx, &mut leaves);
                    left_leaves = leaves.len();
                    self.join_leaves(&join.right, ctx, &mut leaves);
                }
                let sides = (left_leaves, left_width);
                self.prepare_grouped_join(stmt, columns, (&leaves, sides), ctx, &mut first)
            };
            let held = match plan {
                Some(plan) => plan.grouped_blocks(columns, ctx.params().len(), build)?,
                None => std::sync::Arc::new(build()?),
            };
            if let Some(held) = held.as_ref() {
                let run = self.grouped_runner(held, &first, ctx)?.ok_or_else(|| {
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

    /// `rows` of `columns` grouped as `held` prepared them: reading its run
    /// slots as parameters when they are all its subqueries, else asking
    /// the `run`'s blocks
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
        let ctx = match held.runs_alone.is_some() || held.reads_slots {
            true => {
                slots_ctx = ctx
                    .with_slot_params(run.params().unwrap_or_default().iter().cloned().collect());
                &slots_ctx
            }
            false => ctx,
        };
        let blocks: Option<&dyn SubqueryExecutor> = match held.runs_alone {
            Some(_) => None,
            None => Some(run),
        };
        let (result, layout) = self.group_and_window_rows(
            &held.stmt,
            ctx,
            &held.classification,
            rows,
            columns,
            blocks,
            layout,
        )?;
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, SelectShape::laid_out(layout)))
    }

    /// A grouped statement whose subqueries run as `grouped` prepared them:
    /// over a source's rows, or a table's, the general aggregation asking
    /// the blocks of each group; over a table with every one a run slot, as
    /// any grouped statement reading them as parameters
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
        let run = self
            .grouped_runner(held, &grouped.first, ctx)?
            .ok_or_else(|| {
                Error::internal("a table's columns changed while the statement prepared")
            })?;
        let alias = grouped.scope_name().map(SmartString::from);
        if let Some((columns, rows)) = grouped.source_rows(Some(&run), ctx)? {
            return self.group_held_rows(held, &run, ctx, rows, &columns, |out| {
                Self::shared_result_layout(&held.stmt, &columns, alias.as_deref(), out)
            });
        }
        if let Some(plan) = &held.runs_alone {
            let params = run.params().unwrap_or_default().iter().cloned().collect();
            let ctx = ctx.with_slot_params(params);
            return self.execute_select_internal(
                &held.stmt,
                &ctx,
                &held.classification,
                Some(plan),
            );
        }
        let Some(Expression::TableSource(table_source)) = held.stmt.table_expr.as_deref() else {
            return Err(Error::internal("a grouped statement without its table"));
        };
        // An argument read once for its call reads its run slot as a parameter
        let slots_ctx;
        let ctx = match held.reads_slots {
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
    fn filter_has_subquery(expr: &Expression) -> bool {
        use crate::parser::ast::Expression as E;
        let any = |exprs: &[Expression]| exprs.iter().any(Self::filter_has_subquery);
        match expr {
            E::ScalarSubquery(_) | E::Exists(_) | E::AllAny(_) => true,
            E::Identifier(_)
            | E::QualifiedIdentifier(_)
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
        left_alias: Option<&str>,
        right_alias: Option<&str>,
        left: SideRead<'_>,
        right_filter: Option<&Expression>,
        cross_filter: Option<&Expression>,
        prepared: Option<&PreparedJoin>,
        plan: &SemijoinReduction,
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
        let Some((table_name, lookup_strategy, inner_key_col, outer_key_col)) = self
            .check_index_nested_loop_opportunity(
                CteNames::Context(ctx),
                &join_source.right,
                join_source.condition.as_deref(),
                join_type,
                left_alias,
                right_alias,
                true,
                false,
                false,
            )
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
        if !outer_bounded {
            outer_limit = None;
        }
        let unqualified = |name: &str| name.rfind('.').map_or(name, |p| &name[p + 1..]).to_string();
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
        if let Some(run) = run {
            run.blocks.side_fits(1, &inner_table.schema_arc())?;
        }
        let inner_token = inner_table.layout_token();
        let inner_alias = right_alias.unwrap_or(&table_name);
        let inner_cols: Vec<String> = inner_table
            .schema()
            .columns
            .iter()
            .map(|col| format!("{}.{}", inner_alias, col.name))
            .collect();
        // The inner key column, checked on each fetched row: the ids name the
        // rows the index held for the key, not the rows' keys now
        let inner_key_idx = inner_cols.iter().position(|c| {
            c.rsplit('.')
                .next()
                .is_some_and(|name| name.eq_ignore_ascii_case(&inner_key_col))
        });
        let all_columns: Vec<String> = outer_cols
            .iter()
            .chain(inner_cols.iter())
            .cloned()
            .collect();

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
        // Without a cross filter the operator builds only the columns the
        // groups and aggregates read
        let projection = if cross_filter.is_none() {
            use crate::executor::operators::index_nested_loop::ColumnSource;
            let mut sources: Vec<ColumnSource> = Vec::new();
            let mut names: Vec<String> = Vec::new();
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
                        names.push(all_columns[idx].clone());
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
            let schema: Vec<ColumnInfo> = names.iter().map(ColumnInfo::new).collect();
            Some((sources, schema))
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
        let cross = match cross_filter {
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
                Some(RowFilter::from_program(program).with_context(ctx))
            }
            None => None,
        };
        let cross_site = run.map(|run| run.site(false));
        let cross_blocks = cross_site
            .as_ref()
            .map(|site| site as &dyn SubqueryExecutor);
        let pair_site = run.map(|run| run.site(true));
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
                index_join_residual(on, outer_names, inner_names, outer_key_idx, inner_key_idx)
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
                Box::new(QueryResultOperator::new(outer_result, outer_cols.clone()));
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
                inner_cols.iter().map(ColumnInfo::new).collect(),
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
            if let Some((sources, schema)) = &projection {
                op = op.with_projection(sources.clone(), schema.clone());
            }
            op.open()?;
            // The group columns cover the left primary key, so a change in them
            // is a new left row
            let mut current: Option<Row> = None;
            let mut enough = false;
            while let Some(row_ref) = op.next()? {
                let row = row_ref.into_owned();
                if let Some(cross) = &cross {
                    if !cross.matches_checked_with(&row, cross_blocks)? {
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
            let missing = want - groups.len();
            let rate = groups.len() as f64 / fetched as f64;
            let estimate = if rate > 0.0 {
                ((missing as f64 / rate) * 2.0).ceil() as usize
            } else {
                usize::MAX
            };
            let next = estimate.clamp(32, fetched.saturating_mul(3).max(32));
            outer_limit = Some(next);
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::join_next_chunk();
            if consistent {
                let (more, _, bounded, token) = self.read_join_side(
                    &join_source.left,
                    &ctx_join,
                    left,
                    None,
                    Some(next),
                    fetched,
                )?;
                same_outer_source(&join_source.left, outer_token, token, bounded)?;
                outer_result = more;
                continue;
            }
            // Read committed: one row of overlap tells whether a commit moved the
            // boundary; if it did, restart on snapshot isolation
            let (mut more, _, bounded, token) = self.read_join_side(
                &join_source.left,
                &ctx_join,
                left,
                None,
                Some(next + 1),
                fetched - 1,
            )?;
            same_outer_source(&join_source.left, outer_token, token, bounded)?;
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
        cross: Option<&RowFilter>,
        blocks: Option<&dyn SubqueryExecutor>,
        out: &mut RowVec,
    ) -> Result<()> {
        join_op.open()?;
        while let Some(row_ref) = join_op.next()? {
            let row = row_ref.into_owned();
            if let Some(filter) = cross {
                if !filter.matches_checked_with(&row, blocks)? {
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
        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        let processed = self.try_process_select_subqueries(select_exprs, ctx)?;
        let plan = self.select_projection(
            processed.as_deref().unwrap_or(select_exprs),
            all_columns,
            all_columns_lower,
            table_alias,
        )?;
        Self::run_select_projection(&plan, rows, ctx)
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
                        if let Some(dot_idx) = c.rfind('.') {
                            c[dot_idx + 1..].eq_ignore_ascii_case(name)
                        } else {
                            false
                        }
                    })
                })
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
                    // Try full qualified name first (table.column)
                    let full_name =
                        format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    if let Some(idx) = find_column_index(&full_name, all_columns) {
                        simple_column_indices.push(idx);
                    } else if let Some(idx) = find_column_index(&qid.name.value_lower, all_columns)
                    {
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
                            let full_name =
                                format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                            if let Some(idx) = find_column_index(&full_name, all_columns) {
                                simple_column_indices.push(idx);
                            } else if let Some(idx) =
                                find_column_index(&qid.name.value_lower, all_columns)
                            {
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
        match plan {
            SelectProjection::Whole => Ok(rows),
            SelectProjection::Subset(indices) => {
                // Project rows to only include matching columns
                let mut projected = RowVec::with_capacity(rows.len());
                for (id, row) in rows.into_iter() {
                    projected.push((id, row.clone_subset(indices)));
                }
                Ok(projected)
            }
            SelectProjection::Take(indices) => {
                let mut projected = RowVec::with_capacity(rows.len());
                for (id, row) in rows.into_iter() {
                    projected.push((id, row.take_columns(indices)));
                }
                Ok(projected)
            }
            SelectProjection::Actions(actions) => Self::run_projection_actions(actions, rows, ctx),
        }
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
                        // Try full name first, then just column name
                        let full_name =
                            format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                        if let Some(&idx) = col_index_map_lower.get(&full_name) {
                            args.push(ArgSource::Column(idx));
                        } else if let Some(&idx) =
                            col_index_map_lower.get(qid.name.value_lower.as_str())
                        {
                            args.push(ArgSource::Column(idx));
                        } else {
                            return None; // Unknown column
                        }
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
                Expression::QualifiedIdentifier(qid) => {
                    let full = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    col_index_map_lower
                        .get(&full)
                        .or_else(|| col_index_map_lower.get(qid.name.value_lower.as_str()))
                        .copied()
                }
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
                    let full = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    let idx = col_index_map_lower
                        .get(&full)
                        .or_else(|| col_index_map_lower.get(qid.name.value_lower.as_str()))?;
                    Some(ArgSource::Column(*idx))
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

        // Helper to flatten nested || concatenations into a list
        fn flatten_concat(
            expr: &Expression,
            col_index_map_lower: &StringMap<usize>,
            parts: &mut smallvec::SmallVec<[ArgSource; 6]>,
        ) -> bool {
            match expr {
                Expression::Infix(infix) if infix.operator == "||" => {
                    // Recursively flatten left and right
                    flatten_concat(&infix.left, col_index_map_lower, parts)
                        && flatten_concat(&infix.right, col_index_map_lower, parts)
                }
                _ => {
                    // Try to convert to ArgSource
                    if let Some(arg) = expr_to_arg_source(expr, col_index_map_lower) {
                        parts.push(arg);
                        true
                    } else {
                        false
                    }
                }
            }
        }

        // Helper to try building inline Concat action (bypasses VM)
        fn try_build_concat_action(
            expr: &Expression,
            col_index_map_lower: &StringMap<usize>,
        ) -> Option<ExprAction> {
            // Only handle || operator
            if let Expression::Infix(infix) = expr {
                if infix.operator == "||" {
                    let mut parts: smallvec::SmallVec<[ArgSource; 6]> = smallvec::SmallVec::new();
                    if flatten_concat(expr, col_index_map_lower, &mut parts) && parts.len() >= 2 {
                        return Some(ExprAction::Concat(parts));
                    }
                }
            }
            None
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
                Expression::QualifiedIdentifier(qid) => {
                    let full_name =
                        format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    if let Some(&idx) = col_index_map_lower.get(&full_name) {
                        ExprAction::SimpleColumn(idx)
                    } else if let Some(&idx) =
                        col_index_map_lower.get(qid.name.value_lower.as_str())
                    {
                        ExprAction::SimpleColumn(idx)
                    } else {
                        let program = compile_expression(expr, all_columns)?;
                        ExprAction::Compiled(program)
                    }
                }
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
                        Expression::QualifiedIdentifier(qid) => {
                            let full_name =
                                format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                            if let Some(&idx) = col_index_map_lower.get(&full_name) {
                                ExprAction::SimpleColumn(idx)
                            } else if let Some(&idx) =
                                col_index_map_lower.get(qid.name.value_lower.as_str())
                            {
                                ExprAction::SimpleColumn(idx)
                            } else {
                                let program = compile_expression(&aliased.expression, all_columns)?;
                                ExprAction::Compiled(program)
                            }
                        }
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
                        Expression::Infix(_) => {
                            // Try inline concat optimization for ||
                            if let Some(action) =
                                try_build_concat_action(&aliased.expression, col_index_map_lower)
                            {
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
                Expression::Infix(_) => {
                    // Try inline concat optimization for ||
                    if let Some(action) = try_build_concat_action(expr, col_index_map_lower) {
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
            actions.push(action);
        }
        Ok(actions)
    }

    /// The rows `actions` project, run directly on the VM
    fn run_projection_actions(
        actions: &[ExprAction],
        rows: RowVec,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
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
        let mut projected = RowVec::with_capacity(rows.len());

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
                projected.push((id, row.take_columns(&indices)));
            }
        } else {
            // General path: mixed actions
            // Pre-compute capacity outside loop to avoid repeated len() calls
            let values_capacity = actions.len();

            for (id, row) in rows.into_iter() {
                // Use CompactVec directly instead of Vec to avoid Vec->CompactVec conversion
                let mut values: CompactVec<Value> = CompactVec::with_capacity(values_capacity);

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
                        ExprAction::Concat(parts) => {
                            // First pass: calculate exact length for text-only (common case)
                            // and check for nulls/non-text values
                            let mut total_len = 0usize;
                            let mut any_null = false;
                            let mut all_text = true;
                            for part in parts.iter() {
                                let val: Option<&Value> = match part {
                                    ArgSource::Column(idx) => row.get(*idx),
                                    ArgSource::Const(v) => Some(v),
                                };
                                match val {
                                    Some(Value::Text(s)) => total_len += s.len(),
                                    Some(Value::Null(_)) | None => {
                                        any_null = true;
                                        break;
                                    }
                                    // A later part may still be NULL
                                    Some(_) => all_text = false,
                                }
                            }

                            if any_null {
                                values.push(Value::null_unknown());
                            } else if all_text {
                                // Fast path: all text, exact capacity, no shrink_to_fit realloc
                                let mut result = String::with_capacity(total_len);
                                for part in parts.iter() {
                                    let val: Option<&Value> = match part {
                                        ArgSource::Column(idx) => row.get(*idx),
                                        ArgSource::Const(v) => Some(v),
                                    };
                                    if let Some(Value::Text(s)) = val {
                                        result.push_str(s.as_str());
                                    }
                                }
                                // len == capacity, so into_boxed_str is O(1)
                                values.push(Value::Text(result.into()));
                            } else {
                                // Slow path: mixed types, written as Op::Concat writes
                                // them; use Arc to avoid shrink_to_fit
                                let mut result = String::with_capacity(64);
                                for part in parts.iter() {
                                    let val: Option<&Value> = match part {
                                        ArgSource::Column(idx) => row.get(*idx),
                                        ArgSource::Const(v) => Some(v),
                                    };
                                    match val {
                                        Some(Value::Text(s)) => result.push_str(s.as_str()),
                                        Some(v) => {
                                            use std::fmt::Write;
                                            let _ = write!(result, "{}", v);
                                        }
                                        None => {}
                                    }
                                }
                                values.push(Value::Text(SmartString::from_string_shared(result)));
                            }
                        }
                        ExprAction::Compiled(program) => {
                            let value = vm.execute_cow(program, &vm_ctx)?;
                            values.push(value);
                        }
                    }
                }

                projected.push((id, Row::from_compact_vec(values)));
            }
        }

        Ok(projected)
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
                let bare = column.rsplit('.').next().unwrap_or(column);
                column.eq_ignore_ascii_case(name) || bare.eq_ignore_ascii_case(name)
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
        mut rows: RowVec,
        all_columns: &CompactArc<Vec<String>>,
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>, OutputLayout)> {
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
            shared_columns,
            table_alias,
            ctx,
        );
        let projected = Self::run_ordered_projection(&plan, rows, ctx, None)?;
        Ok((projected, extra_names, layout))
    }

    /// What a projection keeping the ORDER BY and DISTINCT ON values the
    /// SELECT list leaves out reads per row, the names it appends, and
    /// where each output column comes from
    fn ordered_projection(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        distinct_on: &[Expression],
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
        } = self.order_by_projection(select_exprs, order_by, distinct_on, all_columns, false);
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
        let sources: Vec<Result<ItemSource>> = select_exprs
            .iter()
            .map(|expr| match select_column(expr, &col_index_map_lower) {
                Some(idx) => idx.map(ItemSource::Column),
                None => compile(unaliased_item(expr)).map(ItemSource::Program),
            })
            .chain(
                extra_order_indices
                    .iter()
                    .map(|&idx| Ok(ItemSource::Column(idx))),
            )
            .chain(
                computed_order_by
                    .iter()
                    .chain(&computed_distinct_on)
                    .map(|expr| compile(expr).map(ItemSource::Program)),
            )
            .collect();
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
        let runs_output = |blocks: &&JoinBlocks| {
            blocks
                .select
                .iter()
                .chain(&blocks.order_by)
                .any(Option::is_some)
        };
        if let Some(blocks) = blocks.filter(runs_output) {
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
                &CompactArc::new(all_columns.to_vec()),
                None,
                ctx,
            );
            columns.extend(extra_names);
            return Ok(Some(JoinOutput {
                projection: OutputProjection::Ordered(plan),
                columns: CompactArc::new(columns),
                layout: Some(CompactArc::new(layout)),
                visible: None,
            }));
        }
        let plan = self.select_projection(&stmt.columns, all_columns, None, None)?;
        Ok(Some(JoinOutput {
            projection: OutputProjection::Select(plan),
            columns: CompactArc::new(columns),
            layout: Self::output_layout(stmt, all_columns, None).map(CompactArc::new),
            visible: None,
        }))
    }

    /// The output of a join whose SELECT list or ORDER BY holds subqueries
    /// the statement's `blocks` run: each item and each key that holds one
    /// compiled as the blocks rewrote it, each key a column of its own
    fn blocks_output(
        &self,
        stmt: &SelectStatement,
        all_columns: &[String],
        sorted: bool,
        blocks: &JoinBlocks,
    ) -> Result<JoinOutput> {
        let registry = &self.function_registry;
        let plain = CompileContext::new(all_columns, registry);
        let compile = |rewritten: Option<&Expression>, expr: &Expression| {
            match rewritten {
                Some(rewritten) => blocks.compile(rewritten, false, registry),
                None => ExprCompiler::new(&plain)
                    .compile(expr)
                    .map(CompactArc::new)
                    .map_err(|e| Error::internal(format!("Compile error: {}", e))),
            }
            .map(ItemSource::Program)
        };
        let index_map = build_column_index_map(all_columns);
        let mut sources: Vec<Result<ItemSource>> = Vec::with_capacity(stmt.columns.len());
        for (item, rewritten) in stmt.columns.iter().zip(&blocks.select) {
            match item {
                Expression::Star(_) => {
                    sources.extend((0..all_columns.len()).map(|i| Ok(ItemSource::Column(i))))
                }
                Expression::QualifiedStar(star) => {
                    for_each_qualified_star_column(all_columns, &star.qualifier, None, |i, _| {
                        sources.push(Ok(ItemSource::Column(i)))
                    })
                }
                _ => sources.push(match (rewritten, select_column(item, &index_map)) {
                    (None, Some(index)) => index.map(ItemSource::Column),
                    (rewritten, _) => compile(rewritten.as_ref(), unaliased_item(item)),
                }),
            }
        }
        let mut columns = self.get_output_column_names(&stmt.columns, all_columns, None);
        let keys_hold_subqueries = blocks.order_by.iter().any(Option::is_some);
        if sorted || !(keys_hold_subqueries || self.order_by_needs_extra_columns(stmt, all_columns))
        {
            return Ok(JoinOutput {
                projection: OutputProjection::Ordered(OrderedProjection::Items(sources)),
                columns: CompactArc::new(columns),
                layout: Self::output_layout(stmt, all_columns, None).map(CompactArc::new),
                visible: None,
            });
        }
        let visible = columns.len();
        let plan = self.order_by_projection(
            &stmt.columns,
            &stmt.order_by,
            &stmt.distinct_on,
            all_columns,
            true,
        );
        sources.extend(
            plan.extra_order_indices
                .iter()
                .map(|&i| Ok(ItemSource::Column(i))),
        );
        for key in &plan.computed_order_by {
            let at = stmt
                .order_by
                .iter()
                .position(|ob| std::ptr::eq(&ob.expression, *key));
            let rewritten = at.and_then(|at| blocks.order_by[at].as_ref());
            sources.push(compile(rewritten, key));
        }
        for expr in &plan.computed_distinct_on {
            sources.push(compile(None, expr));
        }
        let mut layout =
            Self::shared_select_layout(&stmt.columns, CompactArc::new(all_columns.to_vec()), None);
        layout.sources.extend(plan.extra_sources);
        columns.extend(plan.extra_names);
        Ok(JoinOutput {
            projection: OutputProjection::Ordered(OrderedProjection::Items(sources)),
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
    /// by the statement's `blocks`; `applied` when the join applied the
    /// LIMIT and OFFSET
    fn join_output_result(
        output: &JoinOutput,
        rows: RowVec,
        applied: bool,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> SelectResult {
        let rows = match &output.projection {
            OutputProjection::Select(plan) => Self::run_select_projection(plan, rows, ctx)?,
            OutputProjection::Ordered(plan) => {
                Self::run_ordered_projection(plan, rows, ctx, blocks)?
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

    /// The rows `plan` projects, each with its position as its id, its
    /// subqueries answered by the statement's `blocks`
    fn run_ordered_projection(
        plan: &OrderedProjection,
        mut rows: RowVec,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<RowVec> {
        let params = blocks.and_then(|b| b.params()).unwrap_or(ctx.params());
        let mut projected = RowVec::with_capacity(rows.len());
        match plan {
            OrderedProjection::Columns { select, extra } => {
                for (row_id, row) in rows.drain_rows().enumerate() {
                    let mut values = Vec::with_capacity(select.len() + extra.len());
                    for &idx in select.iter().flatten().chain(extra) {
                        values.push(row.get(idx).cloned().unwrap_or(Value::null_unknown()));
                    }
                    projected.push((row_id as i64, Row::from_values(values)));
                }
            }
            OrderedProjection::Items(sources) => {
                // Slow path: Use Evaluator for complex expressions
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::general_path_reached(
                    crate::test_failpoints::GeneralPath::ProjectionEvaluator,
                );
                let mut vm = ExprVM::new();
                let (transaction_id, outer) = (ctx.transaction_id(), ctx.outer_row());
                for (row_id, row) in rows.drain_rows().enumerate() {
                    let mut exec = ExecuteContext::with_common_params(&row, params, transaction_id);
                    if let Some(outer) = outer {
                        exec = exec.with_outer_row(outer);
                    }
                    if let Some(blocks) = blocks {
                        exec = exec.with_subquery_executor(blocks);
                    }
                    let mut values: CompactVec<Value> = CompactVec::with_capacity(sources.len());
                    for source in sources {
                        values.push(match source {
                            Ok(ItemSource::Column(idx)) => {
                                row.get(*idx).cloned().unwrap_or(Value::null_unknown())
                            }
                            Ok(ItemSource::Program(program)) => vm.execute_cow(program, &exec)?,
                            Err(e) => return Err(e.clone()),
                        });
                    }
                    projected.push((row_id as i64, Row::from_compact_vec(values)));
                }
            }
        }
        Ok(projected)
    }

    /// Which columns an ORDER BY or DISTINCT ON projection reads by index,
    /// which it computes, and the names of what it appends after the SELECT;
    /// a key holding a subquery is computed too with `computes_subqueries`
    pub(super) fn order_by_projection<'e>(
        &self,
        select_exprs: &'e [Expression],
        order_by: &'e [crate::parser::ast::OrderByExpression],
        distinct_on: &'e [Expression],
        all_columns: &[String],
        computes_subqueries: bool,
    ) -> OrderByProjection<'e> {
        // Build column index map ONCE with FxHashMap for O(1) lookup
        let col_index_map_lower = build_column_index_map(all_columns);

        // Get SELECT column names (lowercase) for checking duplicates
        let select_column_names: Vec<String> = select_exprs
            .iter()
            .flat_map(|expr| {
                let mut names = Vec::new();
                // Add unqualified name
                if let Some(name) = self.extract_select_column_name(expr) {
                    names.push(name.to_lowercase());
                }
                // Also add fully qualified name for disambiguation
                match expr {
                    Expression::QualifiedIdentifier(qi) => {
                        names.push(format!(
                            "{}.{}",
                            qi.qualifier.value_lower, qi.name.value_lower
                        ));
                    }
                    Expression::Aliased(a) => {
                        if let Expression::QualifiedIdentifier(qi) = &*a.expression {
                            names.push(format!(
                                "{}.{}",
                                qi.qualifier.value_lower, qi.name.value_lower
                            ));
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
            .filter_map(|e| match e {
                Expression::Aliased(a) => Some(a.alias.value_lower.to_string()),
                _ => None,
            })
            .collect();

        // Find ORDER BY columns not in SELECT
        let mut extra_order_indices: Vec<usize> = Vec::new();
        // An ORDER BY expression is read from a column of its own, so the
        // sort has something to read once the projection has run
        let mut computed_order_by: Vec<&Expression> = Vec::new();
        for ob in order_by {
            match &ob.expression {
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
                Expression::QualifiedIdentifier(qi) => {
                    let full_name = format!("{}.{}", qi.qualifier.value_lower, qi.name.value_lower);
                    // Only match against full qualified name in SELECT to avoid ambiguity
                    if !select_column_names.contains(&full_name) {
                        // Prefer fully qualified lookup
                        let idx_opt = col_index_map_lower
                            .get(full_name.as_str())
                            .or_else(|| col_index_map_lower.get(qi.name.value_lower.as_str()));
                        if let Some(&idx) = idx_opt {
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
                    let name = other.to_string();
                    let key = ExprKey::of(other);
                    if !select_column_names
                        .iter()
                        .any(|s| s.eq_ignore_ascii_case(&name))
                        && !computed_order_by.iter().any(|e| ExprKey::of(e) == key)
                    {
                        computed_order_by.push(other);
                    }
                }
                // An expression the sort evaluates carries the input columns
                // it names, which the sort reads by the layout
                other if Self::has_subqueries(other) => {
                    visit_expression_inputs(other, &mut |_, key| {
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

        // A correlated subquery in ORDER BY reads the parent row, so every
        // column rides along until the sort has run
        if !computes_subqueries
            && order_by
                .iter()
                .any(|ob| Self::has_correlated_subqueries(&ob.expression))
        {
            for idx in 0..all_columns.len() {
                if !extra_order_indices.contains(&idx) {
                    extra_order_indices.push(idx);
                }
            }
        }

        // Find DISTINCT ON columns not in SELECT and not already in extra ORDER BY columns
        // Also collect computed DISTINCT ON expressions that need evaluation
        let mut computed_distinct_on: Vec<&Expression> = Vec::new();
        for expr in distinct_on {
            match expr {
                Expression::Identifier(id) => {
                    if !select_column_names
                        .iter()
                        .any(|s| s == id.value_lower.as_str())
                    {
                        if let Some(&idx) = col_index_map_lower.get(id.value_lower.as_str()) {
                            if !extra_order_indices.contains(&idx) {
                                extra_order_indices.push(idx);
                            }
                        }
                    }
                }
                Expression::QualifiedIdentifier(qi) => {
                    let full_name = format!("{}.{}", qi.qualifier.value_lower, qi.name.value_lower);
                    // For qualified identifiers, only match against the full qualified name
                    // in SELECT to avoid ambiguity (e.g., c.name vs p.name)
                    if !select_column_names.contains(&full_name) {
                        // Prefer fully qualified lookup to avoid binding to wrong column
                        let idx_opt = col_index_map_lower
                            .get(full_name.as_str())
                            .or_else(|| col_index_map_lower.get(qi.name.value_lower.as_str()));
                        if let Some(&idx) = idx_opt {
                            if !extra_order_indices.contains(&idx) {
                                extra_order_indices.push(idx);
                            }
                        }
                    }
                }
                _ => {
                    // Computed expression — needs evaluation and appending
                    computed_distinct_on.push(expr);
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
                    // Prefer full qualified name to handle same-named columns across tables
                    let full = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    select_column_indices.push(
                        col_index_map_lower
                            .get(full.as_str())
                            .or_else(|| col_index_map_lower.get(qid.name.value_lower.as_str()))
                            .copied(),
                    );
                }
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => {
                        select_column_indices
                            .push(col_index_map_lower.get(id.value_lower.as_str()).copied());
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        let full =
                            format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                        select_column_indices.push(
                            col_index_map_lower
                                .get(full.as_str())
                                .or_else(|| col_index_map_lower.get(qid.name.value_lower.as_str()))
                                .copied(),
                        );
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
        extra_names.extend(computed_order_by.iter().map(|expr| expr.to_string()));
        extra_names.extend(computed_distinct_on.iter().map(|expr| expr.to_string()));
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
        Self::output_columns(select_exprs, all_columns, table_alias, None)
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
        Self::output_columns(select_exprs, all_columns, table_alias, Some(&mut sources));
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
                None => match folded(name).rsplit_once('.') {
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
    /// `all_columns`, a star expanded, and into `sources` the source of each
    fn output_columns(
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
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
                    if !laid_out {
                        names.extend(all_columns.iter().cloned());
                    }
                    (0..all_columns.len()).for_each(|idx| source(i, Some(idx)));
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
            let name = match expr {
                Expression::Identifier(id) => id.value.to_string(),
                Expression::QualifiedIdentifier(qid) => {
                    // For qualified identifiers, just use the base column name
                    // Ambiguity in output is only an issue if the same base name appears
                    // multiple times in the SELECT clause itself (handled below)
                    qid.name.value.to_string()
                }
                Expression::Aliased(aliased) => aliased.alias.value.to_string(),
                Expression::FunctionCall(func) => {
                    // Use function name as column name
                    func.function.to_string()
                }
                Expression::Cast(cast) => {
                    // Try to derive name from inner expression
                    match &*cast.expr {
                        Expression::Identifier(id) => id.value.to_string(),
                        _ => format!("CAST(expr{})", i + 1),
                    }
                }
                _ => format!("expr{}", i + 1),
            };
            names.push(name);
        }

        names
    }

    /// Get simple column projection indices (returns None if any expression is complex)
    ///
    /// Returns (column_indices, output_names) for simple SELECT with only column references.
    /// Returns None if there are computed expressions that require Evaluator.
    fn get_simple_projection_indices(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
    ) -> Option<(Vec<usize>, Vec<String>)> {
        // SELECT * is a simple case; t.* needs the alias this path lacks
        if select_exprs.len() == 1 && matches!(&select_exprs[0], Expression::Star(_)) {
            return Some(((0..all_columns.len()).collect(), all_columns.to_vec()));
        }

        let col_index_map = build_column_index_map(all_columns);

        let mut indices = Vec::with_capacity(select_exprs.len());
        let mut names = Vec::with_capacity(select_exprs.len());

        for expr in select_exprs {
            match expr {
                Expression::Identifier(id) => {
                    if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                        indices.push(idx);
                        names.push(id.value.to_string());
                    } else {
                        return None;
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    if let Some(&idx) = col_index_map.get(qid.name.value_lower.as_str()) {
                        indices.push(idx);
                        names.push(qid.name.value.to_string());
                    } else {
                        return None;
                    }
                }
                Expression::Aliased(aliased) => {
                    // Check if inner expression is simple
                    match &*aliased.expression {
                        Expression::Identifier(id) => {
                            if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                                indices.push(idx);
                                names.push(aliased.alias.value.to_string());
                            } else {
                                return None;
                            }
                        }
                        Expression::QualifiedIdentifier(qid) => {
                            if let Some(&idx) = col_index_map.get(qid.name.value_lower.as_str()) {
                                indices.push(idx);
                                names.push(aliased.alias.value.to_string());
                            } else {
                                return None;
                            }
                        }
                        _ => return None, // Complex expression in alias
                    }
                }
                _ => return None, // Complex expression
            }
        }

        Some((indices, names))
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
            if set_operation
                && !matches!(col, Expression::Aliased(_))
                && !matches!(col, Expression::FunctionCall(f) if self.function_registry.is_aggregate(&f.function))
            {
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

        // An expression the projection put in a column of its own, named
        // after it, or one the slots above hold
        let find_column_for = |expr: &Expression| -> Option<usize> {
            column_slots.get(&ExprKey::of(expr)).copied().or_else(|| {
                let name = expr.to_string();
                columns.iter().position(|c| *c == name)
            })
        };

        // Each ORDER BY item by its binding: an ordinal by position, a
        // plain name by the column the layout says holds what it reads,
        // a name the input lacks or anything else by a column computing it
        let try_map_to_column = |ob: &crate::parser::ast::OrderByExpression| -> Option<usize> {
            let key = bind_order_key(&ob.expression, stmt);
            match (&key, layout) {
                (OrderKey::Ordinal(position), _) => Some(*position),
                (OrderKey::Expression, _) => find_column_for(&ob.expression),
                (_, Some(layout)) => layout
                    .position(&key)
                    .or_else(|| find_column_for(&ob.expression)),
                // A set operation's rows carry only output names
                (OrderKey::Item(_) | OrderKey::Input(..), None) => {
                    debug_assert!(
                        set_operation,
                        "rows reached ORDER BY {} without a layout",
                        ob.expression
                    );
                    let name = match &ob.expression {
                        Expression::QualifiedIdentifier(qid) => &qid.name.value_lower,
                        Expression::Identifier(id) => &id.value_lower,
                        _ => return None,
                    };
                    columns.iter().position(|c| c.eq_ignore_ascii_case(name))
                }
            }
        };
        let mapped = stmt.order_by.iter().map(try_map_to_column).collect();
        (mapped, aggregate_slots)
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

    /// Execute a temporal query (AS OF TRANSACTION or AS OF TIMESTAMP)
    ///
    /// This enables time-travel queries to see historical data.
    fn execute_temporal_query(
        &self,
        table_name: &str,
        as_of: &AsOfClause,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        tx: &dyn crate::storage::traits::Transaction,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

        // Get table schema
        let table = tx.get_read_table_at(table_name, None, ctx.table_use())?;
        let schema = table.schema().clone();
        let all_columns: Vec<String> = schema.column_names_owned().to_vec();
        // The name the statement gives the source: its alias, or the table's
        let source_alias = stmt
            .table_expr
            .as_deref()
            .and_then(get_table_alias_from_expr);
        let source_alias = source_alias.as_deref().unwrap_or(table_name);

        // Parse temporal value
        let temporal_value = self.parse_temporal_value(as_of, ctx)?;

        // Build columns to fetch
        let columns_to_fetch = self.build_columns_to_fetch(stmt, &all_columns)?;

        // Build storage expression if WHERE clause exists
        // Try to push down to storage, fall back to memory filter for complex expressions
        let (storage_expr, needs_memory_filter) = stmt
            .where_clause
            .as_ref()
            .map(|where_expr| pushdown::try_pushdown(where_expr, &schema, Some(ctx)))
            .unwrap_or((None, false));

        // Execute temporal query
        let result = tx.select_as_of(
            table_name,
            &columns_to_fetch,
            storage_expr.as_deref(),
            &as_of.as_of_type,
            temporal_value,
            Some(&all_columns),
        )?;

        // Get output columns from result
        let _output_columns = result.columns().to_vec();

        // Collect rows for further processing (projection, aggregation, etc.)
        // Use synthetic row IDs since temporal queries don't preserve original IDs
        let mut rows = RowVec::with_capacity(64);
        let mut result_iter = result;
        let mut row_id = 0i64;
        while result_iter.next() {
            rows.push((row_id, result_iter.take_row()));
            row_id += 1;
        }

        // Apply in-memory WHERE filter if storage expression couldn't handle it fully
        if needs_memory_filter {
            if let Some(where_expr) = &stmt.where_clause {
                let where_filter = RowFilter::new(where_expr, &all_columns)?.with_context(ctx);

                where_filter.retain_checked(&mut rows)?;
            }
        }

        // The shape of what this read answers, for its ORDER BY
        let shape_of = |columns: &[String]| {
            SelectShape::laid_out(Self::result_layout(
                stmt,
                &all_columns,
                Some(source_alias),
                columns,
            ))
        };

        if classification.has_aggregation || classification.has_window_functions {
            return self.grouped_select(
                stmt,
                ctx,
                classification,
                rows,
                &all_columns,
                Some(source_alias),
            );
        }

        // Project rows
        let (projected_rows, final_columns) =
            self.project_rows_for_select(stmt, rows, &all_columns, Some(source_alias), ctx)?;

        let shape = shape_of(&final_columns);
        let final_columns_arc = CompactArc::new(final_columns);
        Ok((
            Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(&final_columns_arc),
                projected_rows,
            )),
            final_columns_arc,
            false,
            shape,
        ))
    }

    /// Parse temporal value from AS OF clause
    fn parse_temporal_value(&self, as_of: &AsOfClause, ctx: &ExecutionContext) -> Result<i64> {
        let value = super::expression::evaluate_once(&as_of.value, Some(ctx))?;

        match as_of.as_of_type.to_uppercase().as_str() {
            "TRANSACTION" => {
                // Expect integer transaction ID
                match value {
                    Value::Integer(txn_id) => Ok(txn_id),
                    _ => Err(Error::invalid_argument(
                        "AS OF TRANSACTION requires integer value",
                    )),
                }
            }
            "TIMESTAMP" => {
                // Expect timestamp string or timestamp value
                match value {
                    Value::Timestamp(ts) => {
                        // Convert to nanoseconds since epoch
                        Ok(ts.timestamp_nanos_opt().unwrap_or(0))
                    }
                    Value::Text(s) => {
                        // Parse timestamp string
                        use chrono::{DateTime, NaiveDateTime, Utc};
                        let ts = if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
                            dt.with_timezone(&Utc)
                        } else if let Ok(ndt) =
                            NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                        {
                            DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc)
                        } else if let Ok(ndt) =
                            NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S%.f")
                        {
                            DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc)
                        } else {
                            return Err(Error::invalid_argument(format!(
                                "Invalid timestamp format: {}",
                                s
                            )));
                        };
                        Ok(ts.timestamp_nanos_opt().unwrap_or(0))
                    }
                    Value::Integer(i) => {
                        // Interpret as nanoseconds since epoch
                        Ok(i)
                    }
                    _ => Err(Error::invalid_argument(
                        "AS OF TIMESTAMP requires timestamp or string value",
                    )),
                }
            }
            _ => Err(Error::invalid_argument(format!(
                "Unsupported AS OF type: {}",
                as_of.as_of_type
            ))),
        }
    }

    /// Build columns to fetch for temporal query
    fn build_columns_to_fetch(
        &self,
        stmt: &SelectStatement,
        all_columns: &[String],
    ) -> Result<Vec<String>> {
        // Check for SELECT * or t.*
        if stmt.columns.len() == 1
            && matches!(
                &stmt.columns[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            )
        {
            return Ok(all_columns.to_vec());
        }

        // Extract column names from SELECT expressions
        let mut columns = Vec::new();
        for expr in &stmt.columns {
            match expr {
                Expression::Identifier(id) => columns.push(id.value.to_string()),
                Expression::QualifiedIdentifier(qid) => columns.push(qid.name.value.to_string()),
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => columns.push(id.value.to_string()),
                    Expression::QualifiedIdentifier(qid) => {
                        columns.push(qid.name.value.to_string())
                    }
                    _ => return Ok(all_columns.to_vec()), // Complex expression - need all columns
                },
                Expression::Star(_) | Expression::QualifiedStar(_) => {
                    return Ok(all_columns.to_vec())
                }
                _ => return Ok(all_columns.to_vec()), // Complex expression - need all columns
            }
        }

        // If WHERE clause exists, we might need additional columns
        if stmt.where_clause.is_some() {
            return Ok(all_columns.to_vec());
        }

        Ok(columns)
    }

    /// Project rows for SELECT (helper for temporal queries)
    fn project_rows_for_select(
        &self,
        stmt: &SelectStatement,
        rows: RowVec,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        // Check for SELECT * or a t.* that takes every column
        if stmt.columns.len() == 1
            && star_takes_every_column(&stmt.columns[0], all_columns, table_alias)
        {
            return Ok((rows, all_columns.to_vec()));
        }

        let projected =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, None, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);

        Ok((projected, output_columns))
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
                            || c.rsplit_once('.')
                                .is_some_and(|(_, base)| base.eq_ignore_ascii_case(&alias_name))
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

    /// Get a default column name for an expression
    ///
    /// Used when evaluating complex expressions that don't have an alias.
    /// Returns a reasonable name based on the expression type.
    fn get_expression_column_name(expr: &Expression) -> String {
        match expr {
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
            Expression::Aliased(aliased) => aliased.alias.value.to_string(),
            Expression::FunctionCall(func) => {
                // Use function name as column name
                func.function.to_string()
            }
            Expression::Infix(infix) => {
                // For arithmetic, use the operator as hint
                format!("expr_{}", infix.operator)
            }
            Expression::Prefix(prefix) => {
                format!("expr_{}", prefix.operator)
            }
            Expression::Case(_) => "case".to_string(),
            Expression::Cast(cast) => {
                format!("cast_{}", cast.type_name)
            }
            Expression::Star(_) => "*".to_string(),
            Expression::QualifiedStar(qs) => format!("{}.*", qs.qualifier),
            Expression::IntegerLiteral(lit) => format!("{}", lit.value),
            Expression::FloatLiteral(lit) => format!("{}", lit.value),
            Expression::StringLiteral(lit) => lit.value.to_string(),
            Expression::BooleanLiteral(lit) => format!("{}", lit.value),
            Expression::NullLiteral(_) => "NULL".to_string(),
            _ => "expr".to_string(),
        }
    }

    /// LIMIT and OFFSET of a GROUP BY + LIMIT join as (initial left chunk, rows
    /// expected after OFFSET). The first chunk is doubled for an INNER JOIN,
    /// where a left row without a match forms no group, and doubled again
    /// under HAVING, which drops groups after the fact
    fn semijoin_caps(
        stmt: &SelectStatement,
        is_inner: bool,
    ) -> Option<(usize, usize, usize, usize)> {
        // The plan is made without the statement's parameters: a count that
        // is not a literal keeps the reduction off
        let counts = static_row_counts(stmt)?;
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

    /// Check if semi-join reduction optimization can be applied and return its plan
    ///
    /// Conditions for semi-join reduction:
    /// 1. INNER JOIN or LEFT JOIN (not RIGHT or FULL)
    /// 2. GROUP BY references only left table columns
    /// 3. LIMIT is present with no ORDER BY (for correctness)
    /// 4. Single equality join condition (a.col = b.col)
    ///
    /// The first left chunk is the limit, doubled for an INNER JOIN since some
    /// left rows may not have matches and doubled again under HAVING; the join
    /// reruns with twice the chunk when the result stops short.
    fn get_semijoin_reduction_limit(
        &self,
        join_type: &str,
        stmt: &SelectStatement,
        left_alias: Option<&str>,
        left_table: Option<&str>,
        join_condition: &Option<Box<Expression>>,
        classification: &QueryClassification,
    ) -> Option<SemijoinReduction> {
        // Applies to INNER JOIN and LEFT JOIN (not RIGHT or FULL)
        let is_inner = join_type == "INNER";
        let is_left = join_type.contains("LEFT") && !join_type.contains("FULL");
        if !is_inner && !is_left {
            return None;
        }

        // Must have GROUP BY
        if stmt.group_by.columns.is_empty() {
            return None;
        }

        // Must have LIMIT with no ORDER BY (for correctness)
        // Order of grouped results without ORDER BY is undefined, so early termination is safe
        if stmt.limit.is_none() || !stmt.order_by.is_empty() {
            return None;
        }

        // A window function or DISTINCT runs on the grouped rows afterwards and
        // needs every group, not a chunk that already meets the limit
        if classification.has_window_functions || stmt.distinct || !stmt.distinct_on.is_empty() {
            return None;
        }
        let (cap, want, offset, limit) = Self::semijoin_caps(stmt, is_inner)?;
        // Each group must come from one left row, or a partial chunk would
        // give partial aggregates: GROUP BY has to cover the left primary key
        let left_alias = left_alias?;
        let left_schema = self.engine.get_table_schema(left_table?).ok()?;
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
        // Check that GROUP BY references only left table columns
        for col in &stmt.group_by.columns {
            if !self.column_references_table(col, left_alias) {
                return None;
            }
        }

        // Extract join key columns from simple equality condition: left.col = right.col
        let condition = join_condition.as_ref()?;
        let (left_key_col, right_key_col) = self.extract_simple_join_key(condition)?;

        // Verify left key references the left table
        let left_key_lower = left_key_col.to_lowercase();
        let left_alias_lower = left_alias.to_lowercase();
        let left_alias_len = left_alias_lower.len();
        // Inline prefix check: "alias." without format! allocation
        let left_matches = left_key_lower.len() > left_alias_len
            && left_key_lower.starts_with(&left_alias_lower)
            && left_key_lower.as_bytes()[left_alias_len] == b'.';
        if !left_matches {
            // Maybe it's right.col = left.col (swapped)
            let right_key_lower = right_key_col.to_lowercase();
            let right_matches = right_key_lower.len() > left_alias_len
                && right_key_lower.starts_with(&left_alias_lower)
                && right_key_lower.as_bytes()[left_alias_len] == b'.';
            if !right_matches {
                return None;
            }
            // Swap the keys
            return Some(SemijoinReduction {
                cap,
                want,
                offset,
                limit,
                left_key_col: right_key_col,
                right_key_col: left_key_col,
            });
        }

        Some(SemijoinReduction {
            cap,
            want,
            offset,
            limit,
            left_key_col,
            right_key_col,
        })
    }

    /// Check if a column expression references a specific table
    fn column_references_table(&self, expr: &Expression, table_alias: &str) -> bool {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                qid.qualifier.value.eq_ignore_ascii_case(table_alias)
            }
            Expression::Identifier(_) => {
                // Unqualified column - could be from any table, assume it's from the expected table
                // This is a conservative assumption
                true
            }
            _ => false,
        }
    }

    /// Extract join key column names from a simple equality condition
    /// Returns Some((left_col_qualified, right_col_qualified)) for patterns like a.id = b.user_id
    fn extract_simple_join_key(&self, condition: &Expression) -> Option<(String, String)> {
        match condition {
            Expression::Infix(infix) if infix.op_type == InfixOperator::Equal => {
                let left_col = self.extract_qualified_column_name(&infix.left)?;
                let right_col = self.extract_qualified_column_name(&infix.right)?;
                Some((left_col, right_col))
            }
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // For AND conditions, try to extract the first simple equality
                if let Some(result) = self.extract_simple_join_key(&infix.left) {
                    return Some(result);
                }
                self.extract_simple_join_key(&infix.right)
            }
            _ => None,
        }
    }

    /// Extract qualified column name from an identifier expression
    fn extract_qualified_column_name(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                Some(format!("{}.{}", qid.qualifier.value, qid.name.value))
            }
            Expression::Identifier(id) => Some(id.value.to_string()),
            _ => None,
        }
    }

    /// Check if Index Nested Loop Join can be used.
    ///
    /// Returns Some((table_name, index, inner_key_column, outer_key_column, outer_key_idx)) if:
    /// - Right side is a simple TableSource OR a simple passthrough SubquerySource
    /// - There's an equality join condition on a column that has an index on the right table
    /// - Join type is INNER or LEFT (not RIGHT or FULL)
    ///
    /// The outer_key_idx will be determined after materializing the outer side.
    #[allow(clippy::type_complexity)]
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
    fn grouped_index_join_applies(
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
        let left_table = match join.left.as_ref() {
            Expression::TableSource(ts) => Some(ts.name.value.as_str()),
            _ => None,
        };
        if self
            .get_semijoin_reduction_limit(
                &join_type,
                stmt,
                get_table_alias_from_expr(&join.left).as_deref(),
                left_table,
                &join.condition,
                &classification,
            )
            .is_none()
        {
            let bounded = self
                .pushed_join_limit(stmt, None, &classification, &join_type)
                .is_some();
            return JoinIndexUse::Streaming { bounded };
        }
        let (right_filter_present, filters_have_subquery) = match &stmt.where_clause {
            Some(where_clause) => {
                let ctes = super::explain::cte_names(stmt);
                let schema = |side| self.base_table_side_schema(side, CteNames::Listed(&ctes));
                let (left_schema, right_schema) = (schema(&join.left), schema(&join.right));
                let (_, right, _) = split_join_where(
                    where_clause,
                    get_table_alias_from_expr(&join.left).as_deref(),
                    get_table_alias_from_expr(&join.right).as_deref(),
                    (left_schema.as_deref(), right_schema.as_deref()),
                    &join_type,
                );
                (
                    right.is_some(),
                    !blocks && Self::filter_has_subquery(where_clause),
                )
            }
            None => (false, false),
        };
        match self.grouped_index_join_applies(
            stmt,
            &join_type,
            join.condition.as_deref(),
            right_filter_present,
            filters_have_subquery,
            None,
        ) {
            Ok(Some(_)) => JoinIndexUse::Grouped,
            _ => JoinIndexUse::None,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn check_index_nested_loop_opportunity(
        &self,
        ctes: CteNames<'_>,
        right_expr: &Expression,
        join_condition: Option<&Expression>,
        join_type: &str,
        left_alias: Option<&str>,
        right_alias: Option<&str>,
        bounded: bool,
        small_outer: bool,
        outer_materialized: bool,
    ) -> Option<(
        String,              // table_name
        IndexLookupStrategy, // lookup strategy (index or PK)
        String,              // inner_key_column (unqualified)
        String,              // outer_key_column (qualified)
    )> {
        // Only for INNER or LEFT joins (not RIGHT or FULL)
        // RIGHT joins would need to track all unmatched inner rows
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            return None;
        }

        // Extract table name and effective alias from right side
        // Supports: TableSource, Aliased TableSource, and simple SubquerySource
        let (table_name, effective_alias) = match right_expr {
            Expression::TableSource(_) => (index_join_table_name(right_expr)?, None),
            Expression::Aliased(aliased) => {
                match aliased.expression.as_ref() {
                    Expression::TableSource(_) => (index_join_table_name(right_expr)?, None),
                    Expression::SubquerySource(sq) => {
                        // Check if subquery is a simple passthrough
                        let underlying = self.extract_simple_subquery_table(&sq.subquery)?;
                        // Use the subquery's alias as the effective alias for join condition matching
                        let alias = sq.alias.as_ref().map(|a| a.value.to_string());
                        (underlying, alias)
                    }
                    _ => return None,
                }
            }
            Expression::SubquerySource(sq) => {
                // Check if subquery is a simple passthrough
                let underlying = self.extract_simple_subquery_table(&sq.subquery)?;
                let alias = sq.alias.as_ref().map(|a| a.value.to_string());
                (underlying, alias)
            }
            _ => return None,
        };

        let table_name_ref = &table_name;

        // A CTE of the same name hides the table and has no index
        if ctes.hides(table_name_ref) {
            return None;
        }

        // Need a join condition
        let condition = join_condition?;

        // Extract equality keys from the join condition
        let (left_col, right_col) = self.extract_simple_join_key(condition)?;

        // Determine which column belongs to which table
        let left_col_lower = left_col.to_lowercase();
        let right_col_lower = right_col.to_lowercase();

        // Get the right table alias - prefer effective_alias (from subquery), then right_alias, then table name
        let right_table_alias = effective_alias
            .as_deref()
            .or(right_alias)
            .unwrap_or(table_name_ref);

        // Check if left_col is from right table and right_col is from left table (swapped)
        // Pre-compute lowercase alias to avoid repeated format! allocations
        let right_alias_lower = right_table_alias.to_lowercase();
        let right_alias_len = right_alias_lower.len();
        // Inline prefix check helper
        let starts_with_alias = |col: &str, alias: &str, alias_len: usize| -> bool {
            col.len() > alias_len && col.starts_with(alias) && col.as_bytes()[alias_len] == b'.'
        };
        let (inner_col, outer_col) =
            if starts_with_alias(&left_col_lower, &right_alias_lower, right_alias_len) {
                // left_col is actually from right (inner) table
                (left_col.clone(), right_col.clone())
            } else if starts_with_alias(&right_col_lower, &right_alias_lower, right_alias_len) {
                // right_col is from right (inner) table - normal case
                (right_col.clone(), left_col.clone())
            } else {
                // Can't determine which column is from which table
                return None;
            };
        // An equality between two columns of the inner table joins nothing
        if starts_with_alias(
            &outer_col.to_lowercase(),
            &right_alias_lower,
            right_alias_len,
        ) {
            return None;
        }

        // Verify outer column is from left table (if we have left alias)
        if let Some(left_a) = left_alias {
            let outer_col_lower = outer_col.to_lowercase();
            let left_alias_lower = left_a.to_lowercase();
            let left_alias_len = left_alias_lower.len();
            if !starts_with_alias(&outer_col_lower, &left_alias_lower, left_alias_len) {
                // Outer column doesn't reference left table
                return None;
            }
        }

        // Extract unqualified column name for the inner (right) table
        let inner_col_unqualified = if let Some(dot_pos) = inner_col.rfind('.') {
            inner_col[dot_pos + 1..].to_string()
        } else {
            inner_col.clone()
        };

        // Try to get the table and check for index or PK
        let txn = self.begin_transaction().ok()?;
        let table = txn
            .get_read_table_at(table_name_ref, None, TableUse::Read)
            .ok()?;
        let schema = table.schema();

        // First check if inner column is the PRIMARY KEY (direct row_id lookup)
        let inner_col_lower = inner_col_unqualified.to_lowercase();
        if let Some(pk_idx) = schema.pk_column_index() {
            if schema.columns[pk_idx].name_lower == inner_col_lower {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::join_index_chosen();
                // It's a primary key lookup - most efficient!
                return Some((
                    table_name,
                    IndexLookupStrategy::PrimaryKey,
                    inner_col_unqualified,
                    outer_col,
                ));
            }
        }

        // Check if there's a secondary index on the inner column
        if let Some(index) = table.lookup_index_on_column(&inner_col_unqualified) {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::join_index_chosen();
            return Some((
                table_name,
                IndexLookupStrategy::SecondaryIndex(index),
                inner_col_unqualified,
                outer_col,
            ));
        }

        // A table that keeps rows outside its index still answers a bounded
        // probe from it while it can, deciding per probe; only a join the
        // executor bounds with a limit asks it, at execution and in EXPLAIN,
        // or one whose outer side may be small: on a table with sealed rows,
        // or a hot one large enough for the largest small outer
        // A narrowing key must be the outer side's; an inner column's name is unsure
        let outer_is_sure = left_alias.is_some()
            || outer_col.contains('.')
            || !schema
                .columns
                .iter()
                .any(|c| c.name_lower.eq_ignore_ascii_case(&outer_col));
        let large =
            table.row_count_hint() >= SMALL_OUTER_ROWS.saturating_mul(HOT_INNER_ROWS_PER_KEY);
        let probes = crate::executor::utils::probes_by_equality(&*table, &inner_col_unqualified);
        // A bounded join probes the column. A small outer never probes: its
        // keys narrow a large table, a sealed table through its side file,
        // or the scan of a column no probe reads, from a side whose size only
        // running it tells
        let narrows =
            large || (probes && table.has_sealed_rows()) || (outer_materialized && !probes);
        if (bounded && probes) || (small_outer && outer_is_sure && narrows) {
            return Some((
                table_name,
                IndexLookupStrategy::TableEquality {
                    column: inner_col_unqualified.clone(),
                },
                inner_col_unqualified,
                outer_col,
            ));
        }

        // No index or PK available
        None
    }

    /// Extract the underlying table name from a simple passthrough subquery.
    ///
    /// A "simple passthrough" subquery is one that:
    /// - Selects from a single table (no joins)
    /// - Has no WHERE, GROUP BY, ORDER BY, LIMIT, HAVING, DISTINCT
    /// - Has no CTEs, set operations, or window definitions
    ///
    /// Note: WHERE clauses are rejected because Index Nested Loop would
    /// bypass the subquery and query the raw table directly.
    ///
    /// Returns Some(table_name) if the subquery is simple, None otherwise.
    fn extract_simple_subquery_table(&self, stmt: &SelectStatement) -> Option<String> {
        // Must not have CTEs
        if stmt.with.is_some() {
            return None;
        }

        // Must not have set operations (UNION, INTERSECT, EXCEPT)
        if !stmt.set_operations.is_empty() {
            return None;
        }

        // Must not have GROUP BY
        if !stmt.group_by.columns.is_empty() {
            return None;
        }

        // Must not have HAVING
        if stmt.having.is_some() {
            return None;
        }

        // Must not have ORDER BY (would affect result order)
        if !stmt.order_by.is_empty() {
            return None;
        }

        // Must not have LIMIT or OFFSET (would affect which rows are returned)
        if stmt.limit.is_some() || stmt.offset.is_some() {
            return None;
        }

        // Must not have WHERE clause (Index NL would bypass it and query raw table)
        if stmt.where_clause.is_some() {
            return None;
        }

        // Must not be DISTINCT
        if stmt.distinct {
            return None;
        }

        // Must not have window definitions
        if !stmt.window_defs.is_empty() {
            return None;
        }

        // Must have a table expression that is a simple TableSource
        let table = index_join_table_name(stmt.table_expr.as_ref()?)?;

        // The probe returns the table's columns in the table's order, so the
        // subquery must select exactly those, each under its own name
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
        let col_expr = if let Some(dot_pos) = column_name.find('.') {
            let qualifier = &column_name[..dot_pos];
            let name = &column_name[dot_pos + 1..];
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
            if let Some(dot_pos) = col_lower.rfind('.') {
                let unqualified = &col_lower[dot_pos + 1..];
                if c_lower.contains('.') {
                    // Both qualified
                    c_lower == col_lower
                } else {
                    // col_name is qualified, c is unqualified
                    c_lower == unqualified
                }
            } else if let Some(c_dot) = c_lower.rfind('.') {
                // col_name is unqualified, c is qualified
                c_lower[c_dot + 1..] == *col_lower
            } else {
                false
            }
        })
    }

    /// Streaming GROUP BY optimization using B-tree index
    ///
    /// For queries like `SELECT user_id, SUM(amount) FROM orders GROUP BY user_id`
    /// where user_id has a B-tree index, we can iterate through the index in sorted
    /// order and aggregate each group without using a hash map. This is similar to
    /// SQLite's sorted GROUP BY approach.
    ///
    /// Benefits:
    /// - O(1) memory per group instead of O(groups) hash map
    /// - Better cache locality (sequential access)
    /// - Avoids hash computation overhead
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn try_streaming_group_by(
        &self,
        stmt: &SelectStatement,
        table: &dyn crate::storage::traits::Table,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, CompactArc<Vec<String>>)>> {
        // Rows a table that holds volumes may capture under the index lock,
        // and the bytes its keys and ids may take. Either bound stops the
        // capture, and a capture that stopped before the end of the index
        // answers only when the caller stopped on its own count.
        const CAPTURE_ROWS: usize = 4096;
        const CAPTURE_BYTES: usize = 1024 * 1024;

        if Self::aggregates_hold_subquery(stmt) {
            return Ok(None);
        }

        // Only single-column GROUP BY is supported for now
        if stmt.group_by.columns.len() != 1 {
            return Ok(None);
        }
        // ROLLUP, CUBE and GROUPING SETS add rows the walk has no state for
        if stmt.group_by.modifier != crate::parser::ast::GroupByModifier::None {
            return Ok(None);
        }

        let Some((group_col_name, false)) = self.single_group_column(stmt)? else {
            return Ok(None);
        };

        // The aggregate slots, as every aggregate path reads them; the select
        // list may read the group column and aggregate calls only
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt, Some(ctx))?;
        if aggregations.is_empty() {
            return Ok(None);
        }
        let readable = |expr: &Expression| {
            let inner = match expr {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                other => other,
            };
            match inner {
                Expression::Identifier(id) => id.value_lower == group_col_name.as_str(),
                Expression::FunctionCall(fc) => {
                    crate::executor::utils::is_aggregate_function(&fc.function)
                }
                _ => false,
            }
        };
        if !stmt.columns.iter().all(readable)
            || !non_agg_columns
                .iter()
                .all(|column| column.eq_ignore_ascii_case(&group_col_name))
        {
            return Ok(None);
        }

        // COUNT(*), and SUM, AVG, MIN and MAX over a column: each call's
        // column, None for COUNT(*)
        let mut agg_columns: Vec<Option<usize>> = Vec::with_capacity(aggregations.len());
        for agg in &aggregations {
            // This walk keeps no distinct state and reads each column whole
            if !agg.answered_with(crate::executor::aggregation::Modifiers::NONE) {
                return Ok(None);
            }
            let (func_name, col_name) = (&agg.name, &agg.column_lower);
            match func_name.as_str() {
                // Only COUNT(*) counts rows; COUNT(col) must skip NULLs,
                // which this streaming path cannot do.
                "COUNT" if col_name == "*" => agg_columns.push(None),
                "COUNT" => return Ok(None),
                "SUM" | "AVG" | "MIN" | "MAX" => {
                    // Find the column index for aggregate argument
                    match all_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(col_name))
                    {
                        Some(idx) => agg_columns.push(Some(idx)),
                        None => return Ok(None),
                    }
                }
                _ => return Ok(None), // Unsupported aggregate (STDDEV, VARIANCE, etc.)
            }
        }

        // OPTIMIZATION: Don't use streaming GROUP BY when row fetch is needed without LIMIT.
        // Streaming benefits from early termination (LIMIT), but without it, bulk fetch
        // of all rows is more efficient than per-group fetching, even with buffer reuse.
        let needs_row_fetch = agg_columns.iter().any(Option::is_some);

        // Only use streaming for COUNT-only queries or when LIMIT allows early termination
        if needs_row_fetch && stmt.limit.is_none() {
            return Ok(None); // Fall back to non-streaming (faster for SUM/AVG/MIN/MAX)
        }

        let Some(group_column) = all_columns
            .iter()
            .find(|c| c.eq_ignore_ascii_case(&group_col_name))
        else {
            return Ok(None);
        };
        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(group_column.clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));

        // HAVING over the slot row, compiled as every aggregate path compiles
        // it; a memory table walks under its index lock, so a HAVING that
        // calls a scalar function, which may sleep or write, runs elsewhere
        let having_filter = match stmt.having.as_deref() {
            Some(having) if Self::has_subqueries(having) => return Ok(None),
            Some(having) => {
                let slots = crate::executor::aggregation::aggregate_slot_keys(
                    &self.parse_group_by(stmt, all_columns)?,
                    &aggregations,
                    &stmt.columns,
                );
                let filter = crate::executor::aggregation::compile_over_slots(
                    having,
                    &slot_columns,
                    crate::functions::registry::global_registry(),
                    Some(&slots),
                    None,
                    false,
                    Some(ctx.params()),
                )?;
                use crate::executor::expression::Op;
                if filter
                    .ops()
                    .iter()
                    .any(|op| matches!(op, Op::CallScalar { .. } | Op::NativeFn1(_)))
                {
                    return Ok(None);
                }
                Some(filter)
            }
            None => None,
        };
        let mut having_vm = ExprVM::new();

        // Streaming aggregation: iterate through groups in sorted order
        let mut result_rows = RowVec::new();
        let mut result_row_id = 0i64;

        // With streaming aggregation we can stop once LIMIT + OFFSET groups pass
        // HAVING; OFFSET is applied by the caller and must find its groups here
        let limit_for_early_exit = rows_needed(stmt, ctx);

        // Use streaming callback to avoid upfront allocation of all groups.
        // This is more efficient because:
        // 1. No Value cloning until we need to keep the result
        // 2. No SmallVec->Vec conversion for row IDs
        // 3. Early termination stops iteration immediately
        // 4. Reusable row buffer avoids per-group allocations

        // Pre-allocate reusable buffer for row fetching (avoids alloc/dealloc per group)
        let mut row_buffer = crate::core::RowVec::with_capacity(256);
        // Create true expression once outside loop
        use crate::storage::expression::logical::ConstBoolExpr;
        let true_expr = ConstBoolExpr::true_expr();

        // One run of each call by the implementation every aggregate path
        // runs, reset for each group
        let mut runs = aggregations
            .iter()
            .map(|agg| Ok(agg.template(&self.function_registry)?.instance()))
            .collect::<Result<Vec<_>>>()?;
        let count_star = Value::Integer(1);

        let mut aggregate = |group_value: &Value, row_ids: GroupIds<'_>| -> Result<bool> {
            for run in runs.iter_mut() {
                run.reset();
            }

            if needs_row_fetch {
                // A page of ids at a time through the one row buffer; the
                // aggregate state carries across the group's pages
                for page in row_ids.pages() {
                    row_buffer.clear();
                    table.fetch_rows_by_ids_into(page, &true_expr, &mut row_buffer)?;
                    for (_row_id, row) in &row_buffer {
                        for (run, column) in runs.iter_mut().zip(&agg_columns) {
                            match column {
                                Some(idx) => {
                                    if let Some(value) = row.get(*idx) {
                                        run.accumulate(value);
                                    }
                                }
                                None => run.accumulate(&count_star),
                            }
                        }
                    }
                }
            } else {
                // COUNT(*) alone: the group's row count, no row read
                let rows = crate::functions::aggregate::CompiledAggregate::CountStar {
                    count: row_ids.len() as i64,
                };
                for run in runs.iter_mut() {
                    run.merge(&rows);
                }
            }

            // The slot row, then HAVING over it
            let mut values = Vec::with_capacity(1 + runs.len());
            values.push(group_value.clone());
            for run in &runs {
                values.push(run.result()?);
            }
            let row = Row::from_values(values);
            if let Some(filter) = &having_filter {
                if !having_vm.execute_bool_checked(
                    filter,
                    &crate::executor::aggregation::slot_context(&row, ctx),
                )? {
                    return Ok(true); // Continue to next group
                }
            }
            result_rows.push((result_row_id, row));
            result_row_id += 1;

            // Early termination: stop once we have LIMIT groups that passed HAVING
            if let Some(limit) = limit_for_early_exit {
                if result_rows.len() >= limit {
                    return Ok(false); // Stop iteration
                }
            }

            Ok(true) // Continue to next group
        };

        // The table owns whether its index can answer for this statement's
        // rows, and for how long it holds the index lock: a memory table
        // walks the index under that lock, a table holding volumes captures
        // a prefix of it first and releases the lock before the fetch above
        let walked = table.walk_btree_groups(
            &group_col_name,
            CAPTURE_ROWS,
            CAPTURE_BYTES,
            &mut aggregate,
        )?;
        // Nothing to walk, or the walk cannot answer for the statement's
        // rows: fall back to the regular GROUP BY
        if walked.is_none() {
            return Ok(None);
        }

        if let Some(keep) = limit_for_early_exit {
            result_rows.truncate(keep);
        }

        let slots = crate::executor::AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 1,
            source_columns: all_columns,
            blocks: None,
        };
        let result = self.finish_aggregation(
            crate::executor::AggregatePath::StreamingGroupBy,
            slots,
            ctx,
            slot_columns,
            result_rows,
        )?;
        let columns = CompactArc::new(result.columns().to_vec());
        Ok(Some((result, columns)))
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
