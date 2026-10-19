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
use crate::storage::traits::transaction::StatementTable;
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

/// Type alias for select execution results: (result, column_names, limit_offset_applied, deferred_projection)
/// Using CompactArc<Vec<String>> for column names enables zero-copy sharing across query execution.
/// The deferred_projection field is Some when projection should be applied after ORDER BY + LIMIT.
type SelectOutput = (
    Box<dyn QueryResult>,
    CompactArc<Vec<String>>,
    bool,
    Option<DeferredProjection>,
);
type SelectResult = Result<SelectOutput>;

/// An outer side fetched for a join: the result, its qualified columns,
/// whether the limit and offset were applied, and the layout of a table
type OuterFetch = (Box<dyn QueryResult>, Vec<String>, bool, Option<LayoutToken>);

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
use super::expression::{
    compile_expression, CompiledEvaluator, ExecuteContext, ExprVM, ExpressionEval, JoinFilter,
    RowFilter, SharedProgram,
};
use super::hash_table::JoinHashTable;
use super::join_executor::{JoinExecutor, StreamingJoinRequest};
use super::operator::{ColumnInfo, Operator, QueryResultOperator};
use super::operators::hash_join::JoinType as OperatorJoinType;
use super::operators::index_nested_loop::{
    BatchIndexNestedLoopJoinOperator, IndexLookupStrategy, IndexNestedLoopJoinOperator,
};
use super::operators::BloomFilterOperator;
use super::parallel::{self, ParallelConfig};
use super::pushdown;
use super::query_cache::{CompiledExecution, CompiledJoinResidual};
use super::query_classification::{get_classification, QueryClassification};
use super::result::{
    DistinctOnResult, DistinctResult, ExecResult, ExecutorResult, ExprMappedResult, FilteredResult,
    LimitedResult, OrderedResult, ProjectedResult, RadixOrderSpec, ScannerResult,
    StreamingProjectionResult, TopNResult,
};
use super::row_count::{row_counts, rows_needed, static_row_counts, RowCounts};
use super::utils::compute_join_projection;
use super::utils::{
    add_table_qualifier, build_column_index_map, collect_table_qualifiers,
    collect_unqualified_columns, column_kinds, combine_predicates_with_and, dummy_token,
    expression_contains_aggregate, extract_and_conditions, extract_base_column_name,
    extract_join_keys_and_residual, filter_references_column, flatten_and_predicates,
    flip_operator, get_table_alias_from_expr, index_join_residual, infix_to_operator,
    string_literal_value, strip_table_qualifier, substitute_filter_column,
};
use super::volatility::{expression_volatility, statement_constant};
use super::window::{WindowPreGroupedState, WindowPreSortedState};
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
        && ExpressionEval::compile(expr, &[]).is_ok_and(|eval| {
            matches!(
                eval.with_context(ctx).eval_slice(&Row::new()),
                Ok(Value::Boolean(false) | Value::Null(_))
            )
        })
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
        self.execute_select_with_plan(stmt, ctx, None, None)
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

        // Check for CTEs (WITH clause). A statement with one keeps every
        // subquery for the aggregation entry, which runs with the CTEs bound
        if self.has_cte(stmt) {
            return self.execute_select_with_ctes(stmt, ctx);
        }

        // OPTIMIZATION: Get cached query classification ONCE at entry point
        // This classification is passed through the call chain to avoid
        // redundant hash computations and cache lookups (was 10+ calls per query)
        let classification = match plan_classification {
            Some(slot) => slot.get_or_init(|| get_classification(stmt)).clone(),
            None => get_classification(stmt),
        };

        // A statement of the prepared family runs on that path and no other
        let in_family = classification.where_has_subqueries
            && (self.holds_prepared_select(compiled, ctx)
                || self.prepared_family(stmt, Some(ctx)).is_ok());

        // Correlated probes read the statement's tables through one transaction
        let snapshot_ctx;
        let ctx = if (classification.where_has_correlated_subqueries
            || classification.select_has_correlated_subqueries
            || classification.order_by_has_correlated_subqueries
            || in_family)
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
        let stmt = if (classification.has_aggregation || classification.has_window_functions)
            && stmt.columns.iter().any(Self::has_subqueries)
            && !Self::has_correlated_select_subqueries(&stmt.columns)
        {
            match self.try_process_select_subqueries(&stmt.columns, ctx)? {
                Some(columns) => {
                    bound_stmt = SelectStatement {
                        columns,
                        ..stmt.clone()
                    };
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
        // A prepared statement knows how many columns it shows, a star included
        let mut prepared_visible = None;
        let (mut result, columns, limit_offset_applied, deferred_projection) = if in_family {
            let (result, columns, applied, visible) =
                self.execute_prepared_select(stmt, ctx, compiled, limit, offset)?;
            prepared_visible = Some(visible);
            (result, columns, applied, None)
        } else {
            self.execute_select_internal(branch, ctx, &classification, compiled)?
        };

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
                0 if classification.has_window_functions => {
                    let hidden = Self::hidden_order_by_windows(stmt).len();
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
        if stmt.distinct && stmt.distinct_on.is_empty() {
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
        if !stmt.order_by.is_empty() && !limit_offset_applied {
            // Each column by the identity of what computed it: the SELECT
            // list in order (unless a star moves the positions), then the
            // aggregates carried only for this sort
            let mut column_slots: FxHashMap<ExprKey, usize> = FxHashMap::default();
            // Only aggregates stand in for themselves inside a sort key: a
            // literal or a name there is read as written
            let mut aggregate_slots: FxHashMap<ExprKey, usize> = FxHashMap::default();
            // A set operation's rows come from other branches too: there a
            // column stands for a sort key only when aliased or an aggregate,
            // as on main, and aggregates are not read inside keys
            let set_operation = !stmt.set_operations.is_empty();
            if !stmt
                .columns
                .iter()
                .any(|c| matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)))
            {
                for (i, col) in stmt.columns.iter().enumerate() {
                    if set_operation
                        && !matches!(col, Expression::Aliased(_))
                        && !matches!(col, Expression::FunctionCall(f) if self.function_registry.is_aggregate(&f.function))
                    {
                        continue;
                    }
                    let key = ExprKey::of(col);
                    column_slots.entry(key).or_insert(i);
                    let inner = match col {
                        Expression::Aliased(aliased) => aliased.expression.as_ref(),
                        other => other,
                    };
                    if !set_operation
                        && matches!(inner, Expression::FunctionCall(f) if self.function_registry.is_aggregate(&f.function))
                    {
                        aggregate_slots.entry(key).or_insert(i);
                    }
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

            // Check if ORDER BY expression can be mapped to existing column (handles aggregates)
            let try_map_to_column = |ob: &crate::parser::ast::OrderByExpression| -> Option<usize> {
                match &ob.expression {
                    Expression::Identifier(id) => {
                        // First, try matching against output column names
                        if let Some(pos) = columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&id.value_lower))
                        {
                            return Some(pos);
                        }
                        // Also check if this identifier matches the original expression of an aliased column
                        // e.g., SELECT val AS amount ... ORDER BY val should use the amount column
                        for (i, select_expr) in stmt.columns.iter().enumerate() {
                            if let Expression::Aliased(aliased) = select_expr {
                                // Check if the aliased expression is an identifier matching our ORDER BY
                                if let Expression::Identifier(aliased_id) = &*aliased.expression {
                                    if aliased_id.value_lower == id.value_lower {
                                        return Some(i);
                                    }
                                }
                                // Also check qualified identifier (table.column)
                                if let Expression::QualifiedIdentifier(qid) = &*aliased.expression {
                                    if qid.name.value_lower == id.value_lower {
                                        return Some(i);
                                    }
                                }
                            }
                        }
                        None
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        // Handle qualified column names like "c.name" for ORDER BY
                        // First, try matching against SELECT expressions directly
                        for (i, select_expr) in stmt.columns.iter().enumerate() {
                            match select_expr {
                                Expression::QualifiedIdentifier(sel_qid)
                                    // Direct match: ORDER BY c.name matches SELECT c.name
                                    if sel_qid.qualifier.value_lower == qid.qualifier.value_lower
                                        && sel_qid.name.value_lower == qid.name.value_lower =>
                                {
                                    return Some(i);
                                }
                                Expression::Aliased(aliased) => {
                                    // Check if the aliased expression matches
                                    if let Expression::QualifiedIdentifier(sel_qid) =
                                        aliased.expression.as_ref()
                                    {
                                        if sel_qid.qualifier.value_lower
                                            == qid.qualifier.value_lower
                                            && sel_qid.name.value_lower == qid.name.value_lower
                                        {
                                            return Some(i);
                                        }
                                    }
                                    // An alias whose name happens to equal the
                                    // column part only matches when no column
                                    // carries the qualified name (an alias is
                                    // never itself qualified), so it is tried
                                    // after the full-name lookup below
                                    let full_name = format!(
                                        "{}.{}",
                                        qid.qualifier.value_lower, qid.name.value_lower
                                    );
                                    if aliased.alias.value_lower == qid.name.value_lower
                                        && !columns
                                            .iter()
                                            .any(|c| c.eq_ignore_ascii_case(&full_name))
                                    {
                                        return Some(i);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Fallback: try full qualified name first, then unqualified
                        let full_name =
                            format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                        if let Some(pos) = columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&full_name))
                        {
                            return Some(pos);
                        }
                        columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&qid.name.value_lower))
                    }
                    Expression::IntegerLiteral(lit) => Some((lit.value as usize).saturating_sub(1)),
                    _ => find_column_for(&ob.expression),
                }
            };
            // Mapped once: the correlated path below asks for every row
            let mapped_columns: Vec<Option<usize>> =
                stmt.order_by.iter().map(try_map_to_column).collect();

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
                let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                evaluator = evaluator.with_context(ctx);
                evaluator.init_columns(&columns);
                // An aggregate inside a sort key reads the column computing it
                evaluator.add_aggregate_aliases(&aggregate_slots);

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
                    // Also pre-compute qualified names if alias is present
                    let qualified_names: Option<Vec<CompactArc<str>>> =
                        order_table_alias.as_ref().map(|alias| {
                            columns_lower
                                .iter()
                                .map(|c| CompactArc::from(format!("{}.{}", alias, c).as_str()))
                                .collect()
                        });

                    rows.iter()
                        .map(|(_, row)| {
                            // Build outer row context from current row
                            let mut outer_row_map: FxHashMap<CompactArc<str>, Value> =
                                FxHashMap::default();
                            for (idx, col_lower) in columns_lower.iter().enumerate() {
                                let val = row.get(idx).cloned().unwrap_or(Value::null_unknown());
                                // Use pre-computed lowercase and qualified names (Arc clone is cheap)
                                if let Some(ref qualified) = qualified_names {
                                    outer_row_map.insert(qualified[idx].clone(), val.clone());
                                    outer_row_map.insert(col_lower.clone(), val);
                                // move
                                } else {
                                    outer_row_map.insert(col_lower.clone(), val);
                                    // move directly, no clone
                                }
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
                    let result_cols = result.columns().to_vec();
                    let key_indices = Self::resolve_distinct_on_indices(
                        &stmt.distinct_on,
                        &result_cols,
                        Some(&stmt.columns),
                    );
                    result = Box::new(DistinctOnResult::new(result, key_indices));

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
            let result_columns = result.columns().to_vec();
            let key_indices = Self::resolve_distinct_on_indices(
                &stmt.distinct_on,
                &result_columns,
                Some(&stmt.columns),
            );
            result = Box::new(DistinctOnResult::new(result, key_indices));
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

    /// Resolve DISTINCT ON expressions to column indices in the result set.
    /// Also checks original SELECT expressions to handle aliased columns
    /// (e.g., `SELECT DISTINCT ON (customer) customer AS c` — matches `customer`
    /// against the original identifier behind the `c` alias).
    fn resolve_distinct_on_indices(
        distinct_on: &[Expression],
        result_columns: &[String],
        select_exprs: Option<&[Expression]>,
    ) -> Vec<usize> {
        distinct_on
            .iter()
            .filter_map(|expr| {
                match expr {
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
                        let full = format!("{}.{}", qi.qualifier, qi.name);
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
                        select_exprs
                            .filter(|sel| {
                                !sel.iter().any(|s| {
                                    matches!(s, Expression::Star(_) | Expression::QualifiedStar(_))
                                })
                            })
                            .and_then(|sel| sel.iter().position(|s| ExprKey::of(s) == key))
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
                }
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
                self.execute_simple_table_scan(table_source, stmt, ctx, classification, None)
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

        // Short-circuit: if WHERE is a constant-false expression (e.g., 1=0),
        // skip TVF generation entirely and return empty result.
        if let Some(ref where_clause) = stmt.where_clause {
            if true_for_no_row(where_clause, ctx) {
                let columns = if !tvf_source.column_aliases.is_empty() {
                    tvf_source
                        .column_aliases
                        .iter()
                        .map(|id| id.value.to_string())
                        .collect()
                } else {
                    vec!["value".to_string()]
                };
                return self.execute_query_on_memory_result(stmt, ctx, columns, RowVec::new());
            }
        }

        // Extract range bounds from WHERE clause to narrow TVF generation range.
        // This avoids materializing millions of rows when a selective predicate exists.
        let range_hint = Self::extract_tvf_range_hint(tvf_source, stmt, ctx);

        let (result_rows, columns) =
            Self::evaluate_tvf_with_range(tvf_source, ctx, limit_hint, range_hint)?;

        // Delegate to execute_query_on_memory_result which handles
        // WHERE, ORDER BY, LIMIT, GROUP BY, aggregation, window functions
        self.execute_query_on_memory_result(stmt, ctx, columns, result_rows)
    }

    /// Extract the integer range the WHERE clause allows a TVF to generate
    fn extract_tvf_range_hint(
        tvf_source: &FunctionTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<Resolved> {
        let where_clause = stmt.where_clause.as_ref()?;

        // Determine the TVF value column name (from alias or default "value")
        let col_name: SmartString = if !tvf_source.column_aliases.is_empty() {
            tvf_source.column_aliases[0].value_lower.clone()
        } else {
            SmartString::from("value")
        };

        Self::tvf_range_hint(where_clause, &col_name, ctx)
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
            ExpressionEval::compile(expr, &[])
                .ok()?
                .with_context(ctx)
                .eval_slice(&Row::new())
                .ok()
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
    fn evaluate_tvf_with_range(
        tvf_source: &FunctionTableSource,
        ctx: &ExecutionContext,
        limit: Option<usize>,
        range_hint: Option<Resolved>,
    ) -> Result<(RowVec, Vec<String>)> {
        use crate::functions::global_registry;

        let func_name = &tvf_source.function.value;
        let tvf = global_registry().get_tvf(func_name).ok_or_else(|| {
            Error::NotSupported(format!("Unknown table-valued function: {}", func_name))
        })?;

        // Evaluate arguments to Values
        let mut arg_values = Vec::with_capacity(tvf_source.arguments.len());
        for arg in &tvf_source.arguments {
            let value = ExpressionEval::compile(arg, &[])?
                .with_context(ctx)
                .eval_slice(&Row::new())?;
            arg_values.push(value);
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

        let columns: Vec<String> = if !tvf_source.column_aliases.is_empty() {
            tvf_source
                .column_aliases
                .iter()
                .map(|id| id.value.to_string())
                .collect()
        } else {
            tvf.column_names()
        };

        Ok((result_rows, columns))
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
            let where_result = ExpressionEval::compile(&processed_where, &[])?
                .with_context(ctx)
                .eval_slice(&Row::new())?;
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
                return Ok((Box::new(result), columns, false, None));
            }
        }

        // Check if we have aggregations - if so, use aggregation path with single dummy row
        // This handles cases like SELECT SUM(3+5) or SELECT COALESCE(SUM(1), 0)
        if classification.has_aggregation {
            // Create a single dummy row for aggregation to process
            let mut dummy_rows = RowVec::new();
            dummy_rows.push((0, Row::from_values(vec![])));
            let empty_columns: Vec<String> = vec![];
            let result =
                self.execute_select_with_aggregation(stmt, ctx, dummy_rows, &empty_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
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
            let value = ExpressionEval::compile(bound.as_ref().unwrap_or(col_expr), &[])?
                .with_context(ctx)
                .eval_slice(&Row::new())?;
            values.push(value);
        }

        let row = Row::from_values(values);
        let columns = CompactArc::new(columns);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&columns), rows);

        Ok((Box::new(result), columns, false, None))
    }

    /// Execute a query against in-memory data (for CTEs referenced from context)
    fn execute_query_on_memory_result(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        columns: Vec<String>,
        rows: RowVec,
    ) -> SelectResult {
        // Check if ORDER BY has complex expressions (e.g., -value, SUM(x), val*2)
        // that the CTE path's simple column-name-based sort can't handle.
        // If so, skip ORDER BY/LIMIT in the CTE path and let outer execute_select handle it.
        let has_complex_order_by = stmt.order_by.iter().any(|ob| {
            !matches!(
                &ob.expression,
                Expression::Identifier(_)
                    | Expression::QualifiedIdentifier(_)
                    | Expression::IntegerLiteral(_)
            )
        });

        // Also skip when DISTINCT is present: the outer execute_select applies
        // DISTINCT after projection but before LIMIT. If we apply LIMIT here first,
        // DISTINCT sees too few rows and produces wrong results.
        let skip_order_limit = has_complex_order_by || stmt.distinct;

        let (result_cols, result_rows, order_limit_applied) =
            self.execute_query_on_cte_result_inner(stmt, ctx, columns, rows, skip_order_limit)?;
        let result_cols = CompactArc::new(result_cols);
        Ok((
            Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(&result_cols),
                result_rows,
            )),
            result_cols,
            order_limit_applied,
            None,
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
                    let aliased = stmt.columns.iter().any(|expr| match expr {
                        Expression::Aliased(a) => a.expression.to_string() == name,
                        _ => false,
                    });
                    if !aliased && !select_columns.contains(name.to_lowercase().as_str()) {
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

    /// A table scan's rows projected by the SELECT list and the names of the
    /// output columns; with `order_by_extra`, the ORDER BY columns the list
    /// lacks follow, for execute_select to sort by and drop
    #[allow(clippy::too_many_arguments)]
    fn project_scanned_rows(
        &self,
        stmt: &SelectStatement,
        rows: RowVec,
        all_columns: &[String],
        all_columns_lower: &[String],
        table_alias: Option<&str>,
        order_by_extra: bool,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        let mut output_columns =
            self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        if order_by_extra {
            let (projected_rows, extra_names) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                rows,
                all_columns,
                table_alias,
                ctx,
            )?;
            output_columns.extend(extra_names);
            return Ok((projected_rows, output_columns));
        }
        let projected_rows = self.project_rows_with_alias(
            &stmt.columns,
            rows,
            all_columns,
            Some(all_columns_lower),
            ctx,
            table_alias,
        )?;
        Ok((projected_rows, output_columns))
    }

    /// A table scan's answer when no row can match: no rows, taking the
    /// aggregate, window or projection path scanned rows would take
    #[allow(clippy::too_many_arguments)]
    fn finish_without_rows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        all_columns: &[String],
        all_columns_lower: &[String],
        table_alias: Option<&str>,
        classification: &QueryClassification,
    ) -> SelectResult {
        let result = match (
            classification.has_aggregation,
            classification.has_window_functions,
        ) {
            (true, true) => self.execute_aggregation_then_windows(stmt, ctx, &[], all_columns)?,
            (false, true) => {
                self.execute_select_with_window_functions(stmt, ctx, &[], all_columns)?
            }
            (true, false) => {
                self.execute_select_with_aggregation(stmt, ctx, RowVec::new(), all_columns)?
            }
            (false, false) => {
                let (rows, columns) = self.project_scanned_rows(
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
                return Ok((Box::new(result), columns, false, None));
            }
        };
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, None))
    }

    /// Execute a simple table scan
    fn execute_simple_table_scan(
        &self,
        table_source: &SimpleTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        token_out: Option<&mut Option<LayoutToken>>,
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
                .get_statement_table_at(table_name, ctx.scan_view(table_name))
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

                let table = tx
                    .get_statement_table_at(table_name, ctx.scan_view(table_name))
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
        // The layout this handle's reads are checked against
        if let Some(out) = token_out {
            *out = Some(table.layout_token());
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
                    return Ok((result, columns, false, None));
                }
            }

            // First try global aggregation pushdown (no GROUP BY, no WHERE)
            if let Some(result) =
                self.try_aggregation_pushdown(table.as_ref(), stmt, ctx, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                return Ok((result, columns, false, None));
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
                    return Ok((result, columns, false, None));
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
                    return Ok((result, columns, false, None));
                }
            }

            // Try streaming aggregation for expressions like AVG(col) * 100
            // This avoids collecting all rows by streaming through a scanner
            if let Some(result) =
                self.try_streaming_global_aggregation(table.as_ref(), stmt, ctx, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                return Ok((result, columns, false, None));
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
                return Ok((result, columns, false, None));
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
                    return Ok((result, output_columns, false, None));
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
                return Ok((result, columns, false, None));
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
            if let Some((result, columns)) =
                self.try_order_by_index_optimization(stmt, &*table, &all_columns, ctx)?
            {
                // Note: ORDER BY + LIMIT already handles LIMIT at storage level
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
                    return Ok((result, columns, limit_applied, None));
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
                    return Ok((result, columns, limit_applied, None));
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
                    return Ok((result, output_columns, false, None));
                } else {
                    // SELECT * - no projection needed
                    // LIMIT/OFFSET NOT applied yet - streaming path
                    return Ok((result, CompactArc::new(output_columns), false, None));
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
                    ExprMappedResult::new(result, stmt.columns.clone(), (*output_columns).clone())?
                        .with_context(ctx),
                );

                return Ok((result, output_columns, false, None));
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
                            return Ok((Box::new(result), output_columns, false, None));
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
                        return Ok((result, columns, limit_applied, None));
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
                return Err(crate::core::Error::internal(format!("scan error: {}", e)));
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
                        if !has_order_by && !distinct && !Self::windows_hold_subquery(stmt) {
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

        if has_agg && has_window {
            // Both aggregation and window functions:
            // 1. First apply GROUP BY aggregation
            // 2. Then apply window functions on the aggregated result
            let result = self.execute_aggregation_then_windows(stmt, ctx, &rows, &all_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check if we need window functions only (no aggregation)
        if has_window {
            // Use optimized paths if rows were pre-fetched with index optimization
            let result = if let Some(pregrouped) = window_pregrouped_state {
                // PARTITION BY optimization: rows are already grouped by partition
                self.execute_select_with_window_functions_pregrouped(
                    stmt,
                    ctx,
                    &rows,
                    &all_columns,
                    pregrouped,
                )?
            } else if window_presorted_state.is_some() {
                // ORDER BY optimization: rows are already sorted
                self.execute_select_with_window_functions_presorted(
                    stmt,
                    ctx,
                    &rows,
                    &all_columns,
                    window_presorted_state,
                )?
            } else {
                // Default path: no optimization
                self.execute_select_with_window_functions(stmt, ctx, &rows, &all_columns)?
            };
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check if we need aggregation only (no window functions)
        if has_agg {
            let result = self.execute_select_with_aggregation(stmt, ctx, rows, &all_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
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

        let (projected_rows, output_columns, deferred_proj) = match deferred_projection_info {
            Some((col_indices, output_names)) if !order_by_needs_extra_columns => {
                // DEFERRED PROJECTION: Skip projection now, do it after ORDER BY + LIMIT
                // Return all source columns; projection will be applied after TopNResult
                (
                    rows,
                    all_columns.to_vec(),
                    Some((col_indices, output_names)),
                )
            }
            _ => {
                let (projected_rows, output_columns) = self.project_scanned_rows(
                    stmt,
                    rows,
                    &all_columns,
                    &all_columns_lower,
                    table_alias.as_deref(),
                    order_by_needs_extra_columns,
                    ctx,
                )?;
                (projected_rows, output_columns, None)
            }
        };

        // SEMANTIC CACHE: Insert result for eligible queries
        // For SELECT * queries, cache the raw rows before returning
        //
        // Note: The cache stores Vec<Row>, so we extract rows from RowVec for caching.
        // The result keeps the original RowVec.
        // Skip caching when deferred projection is used (rows are not projected yet)
        if let Some(stamp) = cache_stamp.filter(|_| cache_eligible && deferred_proj.is_none()) {
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
        Ok((Box::new(result), output_columns, false, deferred_proj))
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

        // A subquery in the ON clause is read once, the way the WHERE clause
        // reads one, so every path below compares each pair against a value
        // rather than against something it never evaluated
        let processed_on: Option<Expression> = match join_source.condition.as_deref() {
            Some(cond) if Self::has_subqueries(cond) && !Self::has_correlated_subqueries(cond) => {
                Some(self.process_where_subqueries(cond, ctx)?)
            }
            _ => None,
        };
        let on_condition: Option<&Expression> =
            processed_on.as_ref().or(join_source.condition.as_deref());

        // Determine join type early for filter pushdown decisions
        let join_type = join_source.join_type.to_uppercase();

        // Partition WHERE clause predicates for pushdown
        // Note: For OUTER JOINs, we must be careful:
        // - LEFT JOIN: Can push filters to left (preserved), but NOT to right (may have NULLs)
        // - RIGHT JOIN: Can push filters to right (preserved), but NOT to left
        // - FULL OUTER JOIN: Cannot push filters to either side
        // Uncorrelated subqueries in the WHERE fold to their values before
        // the filters are split, so every join path sees a plain predicate. A
        // correlated one pushed to one side is evaluated per row by that
        // side's scan; one left after the join is evaluated per joined row
        let correlated_where = classification.where_has_correlated_subqueries;
        let where_for_join: Option<Expression> = match &stmt.where_clause {
            Some(where_clause) if classification.where_has_subqueries => {
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
        let left_schema = self.base_table_side_schema(&join_source.left, CteNames::Context(ctx));
        let right_schema = self.base_table_side_schema(&join_source.right, CteNames::Context(ctx));
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
        let semijoin_limit = if correlated_where {
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
            let (_, left_cols, _, _) = self.execute_table_expression_with_filter_limit(
                &join_source.left,
                ctx,
                Some(&none),
                Some(0),
                0,
            )?;
            let (_, right_cols, _, _) = self.execute_table_expression_with_filter_limit(
                &join_source.right,
                ctx,
                Some(&none),
                Some(0),
                0,
            )?;
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
                left_filter.as_ref(),
                right_filter.as_ref(),
                cross_filter.as_ref(),
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
            let (left_result, left_cols, left_bounded, _) = self
                .execute_table_expression_with_filter_limit(
                    &join_source.left,
                    ctx,
                    left_filter.as_ref(),
                    Some(left_cap),
                    0,
                )?;
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
            let right_filter_with_in = if !join_key_values.is_empty() {
                let in_expr = self.build_in_filter_expression(&right_key_col, &join_key_values);
                match (right_filter.clone(), in_expr) {
                    (Some(existing), Some(in_filter)) => {
                        Some(Expression::Infix(InfixExpression::new(
                            Token::new(TokenType::Keyword, "AND", Position::default()),
                            Box::new(existing),
                            "AND".to_string(),
                            Box::new(in_filter),
                        )))
                    }
                    (None, Some(in_filter)) => Some(in_filter),
                    (existing, None) => existing,
                }
            } else {
                right_filter.clone()
            };
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::join_side_ran();
            let (right_result, right_cols) = self.execute_table_expression_with_filter(
                &join_source.right,
                ctx,
                right_filter_with_in.as_ref(),
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
            let picked = if correlated_where {
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
            let (index_nl_info, nl_left_filter, nl_right_filter, swapped) = match picked {
                Some((info, true)) => (Some(info), right_filter.clone(), left_filter.clone(), true),
                picked => (
                    picked.map(|(info, _)| info),
                    left_filter.clone(),
                    right_filter.clone(),
                    false,
                ),
            };

            // A join the inner table stops answering leaves through the label
            // to the hash join below, with the side it ran and its context
            let mut prepared_side: Option<PreparedSide> = None;
            let mut prepared_ctx: Option<ExecutionContext> = None;
            let mut reduced_inner: Option<Expression> = None;
            'index_nl: {
                let Some((table_name, lookup_strategy, inner_col, outer_col)) = index_nl_info
                else {
                    break 'index_nl;
                };
                let small_outer = small_outer
                    && matches!(lookup_strategy, IndexLookupStrategy::TableEquality { .. });
                // Index Nested Loop path: stream outer side for early termination
                // When swapped, execute right side as outer (with original right filter, now in nl_left_filter)
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
                let nl_left_filter = if small_outer {
                    nl_left_filter
                } else if let Some(ref right_f) = nl_right_filter {
                    // Check if right filter references the inner join key column
                    let references = filter_references_column(right_f, &inner_col);
                    if references && same_family() {
                        // Create equivalent filter for outer side by substituting the column
                        if let Some(outer_filter) =
                            substitute_filter_column(right_f, &inner_col, &outer_col)
                        {
                            // Combine with existing left filter if any
                            match nl_left_filter {
                                Some(existing) => Some(Expression::Infix(InfixExpression::new(
                                    Token::new(TokenType::Keyword, "AND", Position::default()),
                                    Box::new(existing),
                                    "AND".to_string(),
                                    Box::new(outer_filter),
                                ))),
                                None => Some(outer_filter),
                            }
                        } else {
                            nl_left_filter
                        }
                    } else {
                        nl_left_filter
                    }
                } else {
                    nl_left_filter
                };

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
                    None => self.new_statement_snapshot(
                        crate::core::IsolationLevel::ReadCommitted,
                        ctx.statement_view(),
                    )?,
                };
                let mut ctx_join = ctx.with_statement_snapshot(snapshot.clone());
                if small_outer {
                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::join_side_ran();
                    let (result, columns) = self.execute_table_expression_with_filter(
                        outer_expr,
                        &ctx_join,
                        nl_left_filter.as_ref(),
                    )?;
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
                                reduced_inner = combine_predicates_with_and(
                                    nl_right_filter
                                        .clone()
                                        .into_iter()
                                        .chain([members])
                                        .collect(),
                                );
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
                let (outer_result, outer_cols, outer_bounded, outer_token) = self
                    .execute_table_expression_with_filter_limit(
                        outer_expr,
                        &ctx_join,
                        nl_left_filter.as_ref(),
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

                    // Build all_columns to match physical row order (outer, inner)
                    // This avoids expensive per-row rotation - projections find columns by name
                    let all_columns = {
                        let mut all = outer_cols.clone();
                        all.extend(inner_cols.clone());
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
                    // The program lives on the plan, keyed by the schema epoch,
                    // the outer columns and the key equality it leaves out. A
                    // clause that read a subquery or an outer row this execution
                    // holds that value, so it is not kept
                    let residual_slot = compiled.filter(|_| {
                        processed_on.is_none()
                            && !classification.where_has_subqueries
                            && ctx.outer_row().is_none()
                    });
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::join_inner_opened();
                    let epoch = self.engine.schema_epoch();
                    let build_residual_filter = || -> Result<Option<JoinFilter>> {
                        if let Some(guard) = residual_slot.and_then(|slot| slot.read().ok()) {
                            if let CompiledExecution::JoinResidual(kept) = &*guard {
                                if kept.cached_epoch == epoch
                                    && kept.swapped == swapped
                                    && kept.outer_col == outer_col
                                    && kept.inner_col == inner_col
                                    && kept.outer_cols == outer_cols
                                    && kept.inner_cols == inner_cols
                                {
                                    return Ok(kept.program.as_ref().map(|program| {
                                        JoinFilter::from_program(CompactArc::clone(program))
                                            .with_context(ctx)
                                    }));
                                }
                            }
                        }
                        let inner_filter = nl_right_filter
                            .as_ref()
                            .map(|rf| add_table_qualifier(rf, inner_alias));
                        // The equality the probe answers is left out, so
                        // only the rest of the clause is asked of each pair
                        let rest = on_condition.and_then(|on| {
                            index_join_residual(
                                on,
                                &outer_cols,
                                &inner_cols,
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
                        let filter = combined
                            .map(|expr| {
                                JoinFilter::new(
                                    &expr,
                                    &outer_cols,
                                    &inner_cols,
                                    &self.function_registry,
                                )
                                .map(|f| f.with_context(ctx))
                            })
                            .transpose()?;
                        if let Some(mut guard) = residual_slot.and_then(|slot| slot.write().ok()) {
                            // A plan with two index joins keeps the first one's
                            // program rather than swapping them every execution
                            let taken = matches!(&*guard,
                                CompiledExecution::JoinResidual(kept) if kept.cached_epoch == epoch);
                            if !taken {
                                *guard = CompiledExecution::JoinResidual(CompiledJoinResidual {
                                    swapped,
                                    outer_cols: outer_cols.clone(),
                                    inner_cols: inner_cols.clone(),
                                    outer_col: outer_col.clone(),
                                    inner_col: inner_col.clone(),
                                    program: filter
                                        .as_ref()
                                        .map(|f| CompactArc::clone(f.program())),
                                    cached_epoch: epoch,
                                });
                            }
                        }
                        Ok(filter)
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
                        Some(ref cross) => {
                            let resolved = Self::with_select_aliases(stmt, cross, &all_columns);
                            let cross = resolved.as_ref().unwrap_or(cross);
                            Some(RowFilter::new(cross, &all_columns)?.with_context(ctx))
                        }
                        None => None,
                    };
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
                            let mut op = IndexNestedLoopJoinOperator::new(
                                outer_op,
                                table,
                                inner_cols.iter().map(ColumnInfo::new).collect(),
                                op_join_type,
                                outer_idx,
                                lookup_strategy.clone(),
                                build_residual_filter()?,
                            )
                            .with_inner_key(inner_key_idx);
                            if let Some(ref proj) = projection_pushdown {
                                let projected_schema: Vec<ColumnInfo> =
                                    proj.output_columns.iter().map(ColumnInfo::new).collect();
                                op = op.with_projection(proj.columns.clone(), projected_schema);
                            }
                            Self::collect_join_rows(
                                &mut op,
                                join_limit,
                                cross_row_filter.as_ref(),
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
                                let (more, _, bounded, token) = self
                                    .execute_table_expression_with_filter_limit(
                                        outer_expr,
                                        &ctx_join,
                                        nl_left_filter.as_ref(),
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
                            let (mut more, _, bounded, token) = self
                                .execute_table_expression_with_filter_limit(
                                    outer_expr,
                                    &ctx_join,
                                    nl_left_filter.as_ref(),
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
                            let (restarted, _, bounded, token) = self
                                .execute_table_expression_with_filter_limit(
                                    outer_expr,
                                    &ctx_join,
                                    nl_left_filter.as_ref(),
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
                        let mut op = BatchIndexNestedLoopJoinOperator::new(
                            outer_op,
                            inner_table,
                            inner_cols.iter().map(ColumnInfo::new).collect(),
                            op_join_type,
                            outer_idx,
                            lookup_strategy.clone(),
                            build_residual_filter()?,
                        )
                        .with_inner_key(inner_key_idx);
                        if let Some(ref proj) = projection_pushdown {
                            let projected_schema: Vec<ColumnInfo> =
                                proj.output_columns.iter().map(ColumnInfo::new).collect();
                            op = op.with_projection(proj.columns.clone(), projected_schema);
                        }
                        Self::collect_join_rows(
                            &mut op,
                            None,
                            cross_row_filter.as_ref(),
                            &mut result_rows,
                        )?;
                    }

                    let mut final_rows = result_rows;
                    // Columns are found by name, so rows keep the physical
                    // outer-then-inner order, except where `*` lists the
                    // columns in the order the statement wrote the sides
                    let star = stmt
                        .columns
                        .iter()
                        .any(|c| matches!(c, Expression::Star(_)));
                    let all_columns = if swapped && star && projection_pushdown.is_none() {
                        let outer_len = outer_cols.len();
                        for (_, row) in final_rows.iter_mut() {
                            let mut values = std::mem::take(row).into_values();
                            values.rotate_left(outer_len);
                            *row = Row::from_values(values);
                        }
                        let mut logical = inner_cols.clone();
                        logical.extend(outer_cols.iter().cloned());
                        logical
                    } else {
                        all_columns
                    };

                    // Check for aggregation/window functions that need special handling
                    let has_agg = classification.has_aggregation;
                    let has_window = classification.has_window_functions;

                    // Apply ORDER BY if present. Aggregation/window queries
                    // order after aggregating, so they see the untruncated
                    // rows. An ORDER BY expression the join-local
                    // evaluator cannot resolve (SELECT aliases, positions)
                    // also defers ordering to the standard path.
                    // A bare integer literal in ORDER BY is an output
                    // ordinal, which only the standard path resolves
                    let has_positional = stmt
                        .order_by
                        .iter()
                        .any(|ob| matches!(ob.expression, Expression::IntegerLiteral(_)));
                    let mut order_limit_applied = false;
                    // DISTINCT merges rows before the LIMIT counts them, so
                    // its ordering and LIMIT are left to the standard path
                    let distinct = stmt.distinct || !stmt.distinct_on.is_empty();
                    if !has_agg
                        && !has_window
                        && !has_positional
                        && !distinct
                        && !stmt.order_by.is_empty()
                    {
                        // Build sort specs by evaluating ORDER BY expressions
                        let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                        evaluator = evaluator.with_context(ctx);
                        evaluator.init_columns(&all_columns);

                        // Compute sort keys and indices; compile and
                        // evaluation errors both defer to the standard path
                        let mut keys_ok = true;
                        let mut programs = Vec::with_capacity(stmt.order_by.len());
                        for ob in &stmt.order_by {
                            match evaluator.compile_cached(&ob.expression) {
                                Ok(p) => programs.push(p),
                                Err(_) => {
                                    keys_ok = false;
                                    break;
                                }
                            }
                        }
                        let mut sort_keys: Vec<Vec<Value>> = Vec::with_capacity(final_rows.len());
                        if keys_ok {
                            'keys: for (_, row) in final_rows.iter() {
                                let mut keys = Vec::with_capacity(programs.len());
                                for p in &programs {
                                    match evaluator.evaluate_program(p, row) {
                                        Ok(v) => keys.push(v),
                                        Err(_) => {
                                            keys_ok = false;
                                            break 'keys;
                                        }
                                    }
                                }
                                sort_keys.push(keys);
                            }
                        }
                        if keys_ok {
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

                    if has_agg || has_window {
                        return self.aggregate_joined_rows(
                            stmt,
                            ctx,
                            classification,
                            final_rows,
                            &all_columns,
                        );
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
                        let (projected_rows, extra_names) = self.project_rows_with_order_by(
                            &stmt.columns,
                            &stmt.order_by,
                            &stmt.distinct_on,
                            final_rows,
                            &all_columns,
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
                        return Ok((Box::new(result), output_columns, false, None));
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
                        return Ok((Box::new(result), output_columns, order_limit_applied, None));
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
                    let (left_result, left_cols) = self.execute_table_expression_with_filter(
                        &join_source.left,
                        ctx,
                        left_filter.as_ref(),
                    )?;

                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::join_side_ran();
                    let (right_result, right_cols) = self.execute_table_expression_with_filter(
                        &join_source.right,
                        ctx,
                        right_filter.as_ref(),
                    )?;

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
                    let probe_source: Box<dyn Operator> = if let Some(ref bf) = bloom_filter {
                        if !probe_key_indices.is_empty() {
                            Box::new(BloomFilterOperator::new(
                                probe_source,
                                bf.clone(),
                                probe_key_indices.clone(),
                            ))
                        } else {
                            probe_source
                        }
                    } else {
                        probe_source
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
                        bloom_filter,
                        pre_built_hash_table,
                        cross_key,
                    };

                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::streaming_join_streamed();
                    let join_result = join_executor.execute_streaming(streaming_request)?;

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
                let run_side = |expr: &Expression, filter: Option<&Expression>| {
                    #[cfg(any(test, feature = "test-failpoints"))]
                    crate::test_failpoints::join_side_ran();
                    let (result, columns) =
                        self.execute_table_expression_with_filter(expr, side_ctx, filter)?;
                    Ok::<_, Error>((Self::materialize_result_arc(result)?, columns))
                };
                match prepared_side.take() {
                    Some(PreparedSide {
                        rows,
                        columns,
                        is_right: true,
                    }) => {
                        let filter = reduced_inner.as_ref().or(left_filter.as_ref());
                        let (left_rows, left_cols) = run_side(&join_source.left, filter)?;
                        (left_rows, left_cols, rows.into_rows()?, columns)
                    }
                    Some(PreparedSide {
                        rows,
                        columns,
                        is_right: false,
                    }) => {
                        let filter = reduced_inner.as_ref().or(right_filter.as_ref());
                        let (right_rows, right_cols) = run_side(&join_source.right, filter)?;
                        (rows.into_rows()?, columns, right_rows, right_cols)
                    }
                    None => {
                        let (left_rows, left_cols) =
                            run_side(&join_source.left, left_filter.as_ref())?;
                        let (right_rows, right_cols) =
                            run_side(&join_source.right, right_filter.as_ref())?;
                        (left_rows, left_cols, right_rows, right_cols)
                    }
                }
            }
        };

        // Combine column names (qualified with table aliases)
        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.clone());

        // Handle NATURAL JOIN or USING clause by automatically finding common columns
        let natural_join_condition =
            if join_type.contains("NATURAL") || !join_source.using_columns.is_empty() {
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
                            let left_col_name = left_columns[*left_idx].clone();
                            let right_col_name = right_columns[*right_idx].clone();
                            let left_col = Expression::Identifier(Identifier::new(
                                Token::new(
                                    TokenType::Identifier,
                                    left_col_name.clone(),
                                    Position::default(),
                                ),
                                left_col_name,
                            ));
                            let right_col = Expression::Identifier(Identifier::new(
                                Token::new(
                                    TokenType::Identifier,
                                    right_col_name.clone(),
                                    Position::default(),
                                ),
                                right_col_name,
                            ));
                            conditions.push(Expression::Infix(InfixExpression::new(
                                Token::new(TokenType::Operator, "=", Position::default()),
                                Box::new(left_col),
                                "=".to_string(),
                                Box::new(right_col),
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
            } else {
                (None, Vec::new(), Vec::new())
            };

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

        let resolved_where_clause = effective_where.map(|where_expr| {
            Box::new(
                Self::with_select_aliases(stmt, &where_expr, &all_columns).unwrap_or(where_expr),
            )
        });

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = resolved_where_clause {
            if correlated_where {
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
                // Uncorrelated subqueries were folded before the filters were split
                let where_filter = RowFilter::new(where_clause, &all_columns)?.with_context(ctx);
                let mut filtered = RowVec::with_capacity(result_rows.len());
                for (id, row) in result_rows {
                    if where_filter.matches_checked(&row)? {
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
            return self.aggregate_joined_rows(
                stmt,
                ctx,
                classification,
                final_rows,
                &final_columns,
            );
        }

        // Check if ORDER BY or DISTINCT ON references columns not in SELECT
        let join_needs_extra_columns = self.order_by_needs_extra_columns(stmt, &final_columns);

        // Project rows according to SELECT expressions
        let (projected_rows, output_columns) = if join_needs_extra_columns {
            // Use projection that preserves extra ORDER BY / DISTINCT ON columns
            let (projected_rows, extra_names) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                final_rows,
                &final_columns,
                None,
                ctx,
            )?;
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &final_columns, None);
            output_columns.extend(extra_names);
            (projected_rows, CompactArc::new(output_columns))
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
            (projected_rows, output_columns)
        };

        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, false, None))
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
                return Ok((agg_result, out_columns, false, None));
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

            let agg_result = self.execute_select_with_aggregation(stmt, ctx, rows, &columns)?;
            let out_columns = CompactArc::new(agg_result.columns().to_vec());
            return Ok((agg_result, out_columns, false, None));
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

        // Check if we need window functions
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &filtered_rows, &columns)?;
            let out_columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, out_columns, false, None));
        }

        // Check if we need aggregation
        if classification.has_aggregation {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &columns)?;
            let out_columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, out_columns, false, None));
        }

        let subquery_alias = subquery_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str());

        // Project rows according to SELECT expressions; a column ORDER BY
        // or DISTINCT ON reads that the select list leaves out rides along
        let (projected_rows, output_columns) = if self.order_by_needs_extra_columns(stmt, &columns)
        {
            let (projected_rows, extra_names) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                filtered_rows,
                &columns,
                subquery_alias,
                ctx,
            )?;
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &columns, subquery_alias);
            output_columns.extend(extra_names);
            (projected_rows, CompactArc::new(output_columns))
        } else {
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                filtered_rows,
                &columns,
                None,
                ctx,
                None,
            )?;
            let output_columns = CompactArc::new(self.get_output_column_names(
                &stmt.columns,
                &columns,
                subquery_alias,
            ));
            (projected_rows, output_columns)
        };

        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, false, None))
    }

    /// Parse a view's SQL query and return the cached Statement.
    /// Uses the query cache to avoid re-parsing the same view query multiple times.
    /// Returns Arc<Statement> - caller should extract SelectStatement via pattern match.
    fn parse_view_statement(&self, view_query: &str) -> Result<Arc<Statement>> {
        // Try to get from cache first (cheap Arc clone, no Statement clone)
        if let Some(cached) = self.query_cache.get(view_query) {
            // Validate it's a SELECT statement
            if !matches!(cached.statement.as_ref(), Statement::Select(_)) {
                return Err(Error::InvalidArgument(
                    "View definition is not a SELECT statement".to_string(),
                ));
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

        // Wrap in Arc and cache (single allocation, no clone)
        let stmt_arc = Arc::new(statements.into_iter().next().unwrap());

        // Cache the parsed statement for future use
        self.query_cache.put(view_query, stmt_arc.clone(), false, 0);

        Ok(stmt_arc)
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
        let result = self.execute_select(view_select, &nested_ctx)?;
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
            let output_columns =
                CompactArc::new(self.get_output_column_names(&stmt.columns, &view_columns, None));
            let result = ExecutorResult::with_arc_columns(
                CompactArc::clone(&output_columns),
                projected_rows,
            );
            return Ok((Box::new(result), output_columns, false, None));
        }

        // Handle aggregation: if outer query has aggregates, materialize view result and aggregate
        if classification.has_aggregation {
            // Materialize the view result into rows with synthetic IDs
            let mut rows = RowVec::with_capacity(64);
            let mut idx = 0i64;
            while result.next() {
                rows.push((idx, result.take_row()));
                idx += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }

            // Execute aggregation on the view's rows
            let agg_result =
                self.execute_select_with_aggregation(stmt, ctx, rows, &view_columns)?;
            let columns = CompactArc::new(agg_result.columns().to_vec());
            return Ok((agg_result, columns, false, None));
        }

        // Handle window functions: materialize view result and delegate
        if classification.has_window_functions {
            let mut rows = RowVec::with_capacity(64);
            let mut idx = 0i64;
            while result.next() {
                rows.push((idx, result.take_row()));
                idx += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }

            if classification.has_aggregation {
                // Aggregation + window functions: aggregate first, then apply window functions
                let result =
                    self.execute_aggregation_then_windows(stmt, ctx, &rows, &view_columns)?;
                let columns = CompactArc::new(result.columns().to_vec());
                return Ok((result, columns, false, None));
            }

            let result =
                self.execute_select_with_window_functions(stmt, ctx, &rows, &view_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Handle projection: apply outer query's column selection
        // Check if SELECT * or t.* - if so, return all view columns
        let is_select_star = stmt.columns.len() == 1
            && matches!(
                &stmt.columns[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            );

        if is_select_star {
            // For SELECT *, just return the view result with WHERE applied
            // DISTINCT, ORDER BY, LIMIT/OFFSET are handled by execute_select
            return Ok((result, CompactArc::new(view_columns), false, None));
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
            let (projected_rows, extra_names) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                &stmt.distinct_on,
                rows,
                &view_columns,
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
            return Ok((Box::new(result), output_columns, false, None));
        }

        // Determine if we have any complex expressions (not just column references)
        // If all expressions are simple column references, use fast StreamingProjectionResult
        // Otherwise, use ExprMappedResult with pre-compiled expression evaluation
        let mut has_complex_expressions = false;
        let mut column_indices = Vec::with_capacity(stmt.columns.len());
        let mut output_columns = Vec::with_capacity(stmt.columns.len());

        // OPTIMIZATION: Pre-compute lowercase view column names once
        let view_columns_lower: Vec<String> =
            view_columns.iter().map(|c| c.to_lowercase()).collect();

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
                    // Expand columns for specific table/alias
                    let qualifier_lower = qs.qualifier.to_lowercase();
                    let qualifier_len = qualifier_lower.len();
                    for (idx, col_lower) in view_columns_lower.iter().enumerate() {
                        // Inline prefix check: "qualifier." without format! allocation
                        if col_lower.len() > qualifier_len
                            && col_lower.starts_with(qualifier_lower.as_str())
                            && col_lower.as_bytes()[qualifier_len] == b'.'
                        {
                            column_indices.push(idx);
                            // Strip "qualifier." from the column name for the output
                            output_columns.push(view_columns[idx][qualifier_len + 1..].to_string());
                        }
                    }
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
                        let qualifier_lower = qs.qualifier.to_lowercase();
                        let qualifier_len = qualifier_lower.len();
                        // Use pre-computed lowercase columns from earlier
                        for (idx, col_lower) in view_columns_lower.iter().enumerate() {
                            // Inline prefix check: "qualifier." without format! allocation
                            if col_lower.len() > qualifier_len
                                && col_lower.starts_with(qualifier_lower.as_str())
                                && col_lower.as_bytes()[qualifier_len] == b'.'
                            {
                                // Strip "qualifier." from the column name for the output
                                final_output_columns
                                    .push(view_columns[idx][qualifier_len + 1..].to_string());
                            }
                        }
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
                ExprMappedResult::with_defaults(
                    result,
                    stmt.columns.clone(),
                    final_output_columns.clone(),
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
                    Expression::Star(_) | Expression::QualifiedStar(_) => {
                        for name in &view_columns {
                            cols.push(name.clone());
                        }
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

        Ok((result, CompactArc::new(final_columns), false, None))
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

        // Determine column names
        let num_columns = if values_source.rows.is_empty() {
            0
        } else {
            values_source.rows[0].len()
        };

        // Get the table alias for qualified name resolution
        let table_alias = values_source
            .alias
            .as_ref()
            .map(|a| a.value.to_string())
            .unwrap_or_default();

        let column_names: Vec<String> = if !values_source.column_aliases.is_empty() {
            // Use provided column aliases
            values_source
                .column_aliases
                .iter()
                .map(|id| id.value.to_string())
                .collect()
        } else {
            // Generate default column names: column1, column2, ...
            (1..=num_columns).map(|i| format!("column{}", i)).collect()
        };

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
            let mut row_values = Vec::with_capacity(row_exprs.len());
            for expr in row_exprs {
                let value = ExpressionEval::compile(expr, &[])?
                    .with_context(ctx)
                    .eval_slice(&Row::new())?;
                row_values.push(value);
            }
            let row = Row::from_values(row_values);

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

        // Check if we need window functions
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &result_rows, &column_names)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check if we need aggregation
        if classification.has_aggregation {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, result_rows, &column_names)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Create the result - use CompactArc for column names
        let column_names_arc = CompactArc::new(column_names);
        let values_result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&column_names_arc), result_rows);

        // If the SELECT has projections, apply them
        if stmt.columns.len() == 1
            && matches!(
                &stmt.columns[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            )
        {
            // SELECT * or t.* - return all columns
            return Ok((Box::new(values_result), column_names_arc, false, None));
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
                Expression::QualifiedStar(_) => {
                    // For single table, t.* is equivalent to *
                    projected_columns.extend(column_names_arc.iter().cloned());
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
        let mut result_box: Box<dyn QueryResult> = Box::new(values_result);
        while result_box.next() {
            let row = result_box.row();
            let mut new_values = Vec::with_capacity(output_cols);

            for (col_idx, col_expr) in stmt.columns.iter().enumerate() {
                match col_expr {
                    Expression::Star(_) | Expression::QualifiedStar(_) => {
                        // Extend with all values from the row
                        new_values.extend(row.iter().cloned());
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
        Ok((Box::new(final_result), projected_columns_arc, false, None))
    }

    /// Execute a table expression with optional filter pushdown
    /// This is used by JOIN to push WHERE predicates to individual table scans
    pub(crate) fn execute_table_expression_with_filter(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
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
                    // Check view depth to prevent stack overflow
                    let depth = ctx.view_depth();
                    if depth >= MAX_VIEW_DEPTH {
                        return Err(Error::InvalidArgument(format!(
                            "Maximum view nesting depth ({}) exceeded",
                            MAX_VIEW_DEPTH
                        )));
                    }

                    // Parse view query using cache (returns Arc<Statement>, no clone)
                    let view_stmt = self.parse_view_statement(&view_def.query)?;
                    let view_select = match view_stmt.as_ref() {
                        Statement::Select(s) => s,
                        _ => unreachable!("parse_view_statement validates this is a SELECT"),
                    };

                    // Execute with incremented depth
                    let nested_ctx = ctx.with_incremented_view_depth();
                    let result = self.execute_select(view_select, &nested_ctx)?;
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
                };
                // Get classification for the synthetic SELECT statement
                let classification = get_classification(&select_all);
                let (result, columns, _, _) =
                    self.execute_simple_table_scan(ts, &select_all, ctx, &classification, None)?;

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
                // Create a simple SELECT * statement to execute the VALUES
                let select_all = SelectStatement {
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
                    distinct_on: vec![],
                    columns: vec![Expression::Star(StarExpression {
                        token: dummy_token("*", TokenType::Punctuator),
                    })],
                    with: None,
                    table_expr: Some(Box::new(Expression::ValuesSource(vs.clone()))),
                    where_clause: None,
                    group_by: GroupByClause::default(),
                    having: None,
                    window_defs: vec![],
                    order_by: vec![],
                    limit: None,
                    offset: None,
                    set_operations: vec![],
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
    pub(crate) fn execute_table_expression_with_filter_limit(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        row_limit: Option<usize>,
        row_offset: usize,
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
                        .execute_table_expression_with_filter(expr, ctx, filter)
                        .map(|(result, columns)| (result, columns, false, None));
                }
            }
            _ => {
                // Not a table source, fall back to standard execution
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter)
                    .map(|(result, columns)| (result, columns, false, None));
            }
        };

        // Only optimize if we have a limit
        if let Some(limit) = row_limit {
            let table_name = ts.name.value_lower.as_str();

            // Skip if this is a CTE reference - CTEs are already materialized
            if ctx.get_cte_by_lower(table_name).is_some() {
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter)
                    .map(|(result, columns)| (result, columns, false, None));
            }

            // Skip if this is a view
            if self.engine.get_view_lowercase(table_name)?.is_some() {
                return self
                    .execute_table_expression_with_filter(expr, ctx, filter)
                    .map(|(result, columns)| (result, columns, false, None));
            }

            // Create a SELECT * statement with WHERE clause AND LIMIT
            // This triggers the LIMIT pushdown optimization in execute_simple_table_scan
            let select_all = SelectStatement {
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
            };
            // Get classification for the synthetic SELECT statement
            let classification = get_classification(&select_all);
            // The scan says whether it applied the synthetic LIMIT and OFFSET. A
            // path that did not may still have stopped early at their sum, so
            // the pair is applied here, as the select layer would
            let mut token = None;
            let (result, columns, applied, _) = self.execute_simple_table_scan(
                ts,
                &select_all,
                ctx,
                &classification,
                Some(&mut token),
            )?;
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
        self.execute_table_expression_with_filter(expr, ctx, filter)
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
    /// the statement's
    pub(super) fn join_table(
        &self,
        snapshot: &StatementSnapshot,
        name: &str,
        view: Option<&crate::storage::mvcc::registry::StatementView>,
    ) -> Result<Box<dyn crate::storage::traits::Table>> {
        if let Some(active) = self.active_transaction.lock().unwrap().as_ref() {
            return active.transaction.get_statement_table_at(name, view);
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
        let table = self.join_table(snapshot, name, ctx.statement_view())?;
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
        Ok(StatementSnapshot::new(
            transaction,
            isolation,
            view.cloned(),
        ))
    }

    /// The aggregation and window functions of `stmt` over a join's rows, the
    /// one tail every join operator hands its rows to
    fn aggregate_joined_rows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &QueryClassification,
        rows: RowVec,
        columns: &[String],
    ) -> SelectResult {
        let result = match (
            classification.has_aggregation,
            classification.has_window_functions,
        ) {
            // GROUP BY first, then the windows over the groups
            (true, true) => self.execute_aggregation_then_windows(stmt, ctx, &rows, columns)?,
            (false, true) => {
                self.execute_select_with_window_functions(stmt, ctx, &rows, columns)?
            }
            _ => self.execute_select_with_aggregation(stmt, ctx, rows, columns)?,
        };
        let columns = CompactArc::new(result.columns().to_vec());
        Ok((result, columns, false, None))
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
        left_filter: Option<&Expression>,
        right_filter: Option<&Expression>,
        cross_filter: Option<&Expression>,
        plan: &SemijoinReduction,
    ) -> Result<Option<SelectOutput>> {
        let filters_have_subquery = [left_filter, right_filter, cross_filter]
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
            None => self.new_statement_snapshot(
                crate::core::IsolationLevel::ReadCommitted,
                ctx.statement_view(),
            )?,
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
        let (outer_result, outer_cols, outer_bounded, outer_token) = self
            .execute_table_expression_with_filter_limit(
                &join_source.left,
                &ctx_join,
                left_filter,
                outer_limit,
                0,
            )?;
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
                let resolved = Self::with_select_aliases(stmt, cross, &all_columns);
                let cross = resolved.as_ref().unwrap_or(cross);
                Some(RowFilter::new(cross, &all_columns)?.with_context(ctx))
            }
            None => None,
        };
        // A right-side filter the storage layer takes whole runs inside the
        // inner fetch, built once so every chunk sees the same predicate; a
        // partial or refused one stays a residual on the joined row
        let inner_filter = right_filter.and_then(|rf| {
            let rf = crate::executor::utils::strip_table_qualifier(rf, inner_alias);
            match pushdown::try_pushdown(&rf, inner_table.schema(), Some(ctx)) {
                (Some(expr), false) => Some(expr),
                _ => None,
            }
        });
        // The probe answers one equality, so the rest of the ON clause and a
        // right-side filter the fetch did not take are asked of each pair
        let residual = combine_predicates_with_and(
            join_source
                .condition
                .as_deref()
                .and_then(|on| {
                    index_join_residual(on, &outer_cols, &inner_cols, outer_key_idx, inner_key_idx)
                })
                .into_iter()
                .chain(
                    right_filter
                        .filter(|_| inner_filter.is_none())
                        .map(|rf| add_table_qualifier(rf, inner_alias)),
                )
                .collect(),
        );
        let build_residual_filter = || {
            residual
                .as_ref()
                .map(|expr| {
                    JoinFilter::new(expr, &outer_cols, &inner_cols, &self.function_registry)
                        .map(|f| f.with_context(ctx))
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
            let mut op = IndexNestedLoopJoinOperator::new(
                outer_op,
                table,
                inner_cols.iter().map(ColumnInfo::new).collect(),
                op_join_type,
                outer_key_idx,
                lookup_strategy.clone(),
                build_residual_filter()?,
            )
            .with_inner_key(inner_key_idx);
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
                    if !cross.matches_checked(&row)? {
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
                let (more, _, bounded, token) = self.execute_table_expression_with_filter_limit(
                    &join_source.left,
                    &ctx_join,
                    left_filter,
                    Some(next),
                    fetched,
                )?;
                same_outer_source(&join_source.left, outer_token, token, bounded)?;
                outer_result = more;
                continue;
            }
            // Read committed: one row of overlap tells whether a commit moved the
            // boundary; if it did, restart on snapshot isolation
            let (mut more, _, bounded, token) = self.execute_table_expression_with_filter_limit(
                &join_source.left,
                &ctx_join,
                left_filter,
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
            let (restarted, _, bounded, token) = self.execute_table_expression_with_filter_limit(
                &join_source.left,
                &ctx_join,
                left_filter,
                outer_limit,
                0,
            )?;
            same_outer_source(&join_source.left, outer_token, token, bounded)?;
            outer_result = restarted;
        }

        let (final_columns, final_rows) = self.apply_post_aggregation_expressions(
            stmt,
            ctx,
            &aggregations,
            result_columns,
            groups,
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
        out: &mut RowVec,
    ) -> Result<()> {
        join_op.open()?;
        while let Some(row_ref) = join_op.next()? {
            let row = row_ref.into_owned();
            if let Some(filter) = cross {
                if !filter.matches_checked(&row)? {
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

    /// Project rows by evaluating SELECT expressions
    ///
    /// Optimized to:
    /// 1. Use FxHashMap for O(1) column lookup instead of O(n) position()
    /// 2. Avoid allocating lowercase strings per call
    /// 3. Use direct indexing for simple column references
    pub(crate) fn project_rows(
        &self,
        select_exprs: &[Expression],
        rows: RowVec,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        self.project_rows_with_alias(select_exprs, rows, all_columns, None, ctx, None)
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
        // Check if this is SELECT * (no projection needed)
        // Note: QualifiedStar (t.*) DOES need projection to filter columns
        if select_exprs.len() == 1 && matches!(&select_exprs[0], Expression::Star(_)) {
            return Ok(rows); // Pass through directly - pooled
        }

        // OPTIMIZATION: Compute lowercase columns once at the start
        // This avoids per-column to_lowercase() calls in loops below
        let computed_lower_early: Vec<String>;
        let columns_lower: &[String] = if let Some(lower) = all_columns_lower {
            lower
        } else {
            computed_lower_early = all_columns.iter().map(|c| c.to_lowercase()).collect();
            &computed_lower_early
        };

        // Handle SELECT t.* - filter to only columns matching the qualifier
        if select_exprs.len() == 1 {
            if let Expression::QualifiedStar(qs) = &select_exprs[0] {
                // Compute lowercase once and reuse
                let qualifier_lower: String = qs.qualifier.to_lowercase().into();
                let prefix_lower_len = qualifier_lower.len() + 1; // "qualifier."

                // Find indices of columns matching the qualifier (use pre-computed lowercase)
                let matching_indices: Vec<usize> = columns_lower
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| {
                        c.len() > prefix_lower_len
                            && c.starts_with(&qualifier_lower)
                            && c.as_bytes().get(qualifier_lower.len()) == Some(&b'.')
                    })
                    .map(|(i, _)| i)
                    .collect();

                // If no columns matched the prefix (single-table query), check if
                // the qualifier matches the table alias - if so, include all columns
                let indices_to_use = if matching_indices.is_empty() {
                    if let Some(alias) = table_alias {
                        if alias.eq_ignore_ascii_case(&qualifier_lower) {
                            (0..all_columns.len()).collect()
                        } else {
                            matching_indices
                        }
                    } else {
                        matching_indices
                    }
                } else {
                    matching_indices
                };

                // OPTIMIZATION: If selecting all columns in order (identity), pass through directly
                let is_identity = indices_to_use.len() == all_columns.len()
                    && indices_to_use.iter().enumerate().all(|(i, &idx)| i == idx);

                if is_identity {
                    return Ok(rows); // Pass through directly - no allocation needed
                }

                // Project rows to only include matching columns
                let mut projected = RowVec::with_capacity(rows.len());
                for (id, row) in rows.into_iter() {
                    projected.push((id, row.clone_subset(&indices_to_use)));
                }
                return Ok(projected);
            }
        }

        // Check if SELECT columns contain correlated scalar subqueries
        let has_correlated_select = Self::has_correlated_select_subqueries(select_exprs);

        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        // but only if they're NOT correlated (correlated ones must be evaluated per-row)
        let processed = if has_correlated_select {
            None
        } else {
            self.try_process_select_subqueries(select_exprs, ctx)?
        };
        let select_exprs_cow = match &processed {
            Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
            None => std::borrow::Cow::Borrowed(select_exprs),
        };
        let select_exprs = select_exprs_cow.as_ref();

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
                    // Expand columns for specific table/alias
                    // Compute lowercase once and reuse
                    let qualifier_lower: String = qs.qualifier.to_lowercase().into();
                    let mut found_any = false;
                    // Use pre-computed lowercase columns to avoid per-column to_lowercase()
                    for (idx, col_lower) in columns_lower.iter().enumerate() {
                        // Check if column starts with "qualifier."
                        if col_lower.len() > qualifier_lower.len()
                            && col_lower.starts_with(&qualifier_lower)
                            && col_lower.as_bytes().get(qualifier_lower.len()) == Some(&b'.')
                        {
                            simple_column_indices.push(idx);
                            found_any = true;
                        }
                    }
                    // If no columns matched the prefix (single-table query), check if
                    // the qualifier matches the table alias - if so, include all columns
                    if !found_any {
                        if let Some(alias) = table_alias {
                            if alias.eq_ignore_ascii_case(&qualifier_lower) {
                                for idx in 0..all_columns.len() {
                                    simple_column_indices.push(idx);
                                }
                            }
                        }
                    }
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
                    return Ok(rows); // Pass through directly - pooled
                }

                // OPTIMIZATION: Use take_columns to move values instead of cloning
                let mut projected = RowVec::with_capacity(rows.len());
                for (id, row) in rows.into_iter() {
                    projected.push((id, row.take_columns(&simple_column_indices)));
                }
                return Ok(projected);
            }
            // Fall through to slow path if there are duplicates
        }

        // OPTIMIZATION: Pre-compute table alias lowercase
        let table_alias_lower: Option<String> = table_alias.map(|a| a.to_lowercase());

        // Pre-size HashMap to avoid rehashing (estimate 1.5x for qualified names)
        // Use columns_lower which was computed early in the function
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

        // For non-correlated queries, use the optimized path with pre-compiled expressions
        if !has_correlated_select {
            return self.project_rows_optimized(
                select_exprs,
                rows,
                all_columns,
                columns_lower,
                &col_index_map_lower,
                table_alias_lower.as_deref(),
                ctx,
            );
        }

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

        // OPTIMIZATION: Pre-compute QualifiedStar lowercase qualifiers to avoid per-row to_lowercase()
        // Key: original qualifier string, Value: qualifier_lower (no format! allocation needed)
        let qualified_star_cache: FxHashMap<String, String> = select_exprs
            .iter()
            .filter_map(|expr| {
                if let Expression::QualifiedStar(qs) = expr {
                    Some((
                        qs.qualifier.to_string(),
                        qs.qualifier.to_lowercase().to_string(),
                    ))
                } else {
                    None
                }
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
                        // Expand columns for specific table/alias
                        // Use pre-computed cache to avoid per-row to_lowercase()
                        let qualifier_lower = qualified_star_cache
                            .get(qs.qualifier.as_str())
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        let qualifier_len = qualifier_lower.len();
                        let mut found_any = false;
                        for (idx, col_lower) in columns_lower.iter().enumerate() {
                            // Inline prefix check: "qualifier." without format! allocation
                            if col_lower.len() > qualifier_len
                                && col_lower.starts_with(qualifier_lower)
                                && col_lower.as_bytes()[qualifier_len] == b'.'
                            {
                                if let Some(val) = row.get(idx) {
                                    values.push(val.clone());
                                    found_any = true;
                                }
                            }
                        }
                        if !found_any {
                            if let Some(ref alias_lower) = table_alias_lower {
                                if alias_lower.as_str() == qualifier_lower {
                                    for val in row.iter() {
                                        values.push(val.clone());
                                    }
                                }
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

    /// Optimized projection for non-correlated queries.
    /// Pre-compiles all expressions once and uses direct VM execution.
    #[allow(clippy::too_many_arguments)]
    fn project_rows_optimized(
        &self,
        select_exprs: &[Expression],
        rows: RowVec,
        all_columns: &[String],
        all_columns_lower: &[String],
        col_index_map_lower: &StringMap<usize>,
        table_alias_lower: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        // Analyze expressions and pre-compile complex ones
        // Each element represents how to evaluate each SELECT expression:
        // - SimpleColumn: direct column index lookup
        // - StarExpand: expand all columns
        // - QualifiedStarExpand: expand columns for specific qualifier
        // - Coalesce: inline COALESCE evaluation (no VM overhead)
        // - Compiled: pre-compiled program to execute via VM

        // Argument source for inline function evaluation
        enum ArgSource {
            Column(usize),
            Const(Value),
        }

        // Simple condition for inline CASE: column comparisons
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
        struct CaseBranch {
            condition: CaseCondition,
            result: ArgSource,
        }

        enum ExprAction {
            SimpleColumn(usize),
            StarExpand,
            QualifiedStarExpand {
                qualifier_lower: String,
            },
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
                Expression::QualifiedStar(qs) => ExprAction::QualifiedStarExpand {
                    qualifier_lower: qs.qualifier.to_lowercase().to_string(),
                },
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

        // Pre-fetch parameters for VM context
        let params = ctx.params();
        let named_params = ctx.named_params();
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
            // Pre-compute named_params Option to avoid repeated checks
            let named_params_opt = if named_params.is_empty() {
                None
            } else {
                Some(named_params)
            };

            // Pre-compute capacity outside loop to avoid repeated len() calls
            let values_capacity = actions.len();

            for (id, row) in rows.into_iter() {
                // Use CompactVec directly instead of Vec to avoid Vec->CompactVec conversion
                let mut values: CompactVec<Value> = CompactVec::with_capacity(values_capacity);

                // Build VM context with all params in one call (faster than builder chain)
                let vm_ctx = ExecuteContext::with_common_params(
                    &row,
                    params,
                    named_params_opt,
                    transaction_id,
                );

                for action in &actions {
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
                        ExprAction::QualifiedStarExpand { qualifier_lower } => {
                            let qualifier_len = qualifier_lower.len();
                            let mut found_any = false;
                            for (idx, col_lower) in all_columns_lower.iter().enumerate() {
                                // Inline prefix check: "qualifier." without format! allocation
                                if col_lower.len() > qualifier_len
                                    && col_lower.starts_with(qualifier_lower)
                                    && col_lower.as_bytes()[qualifier_len] == b'.'
                                {
                                    if let Some(val) = row.get(idx) {
                                        values.push(val.clone());
                                        found_any = true;
                                    }
                                }
                            }
                            if !found_any {
                                if let Some(alias_lower) = table_alias_lower {
                                    if alias_lower == qualifier_lower {
                                        for val in row.iter() {
                                            values.push(val.clone());
                                        }
                                    }
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
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
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
            bound_columns = Some(columns);
        }
        let select_exprs: &[Expression] = bound_exprs.as_deref().unwrap_or(select_exprs);
        let all_columns: &[String] = bound_columns.as_deref().unwrap_or(all_columns);

        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        let processed = self.try_process_select_subqueries(select_exprs, ctx)?;
        let select_exprs = match &processed {
            Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
            None => std::borrow::Cow::Borrowed(select_exprs),
        };

        let OrderByProjection {
            col_index_map_lower,
            select_column_indices,
            extra_order_indices,
            computed_order_by,
            computed_distinct_on,
            extra_names,
        } = self.order_by_projection(&select_exprs, order_by, distinct_on, all_columns);

        // Check if we can use fast path (all simple column refs, no computed DISTINCT ON)
        let all_simple = select_column_indices.iter().all(|idx| idx.is_some())
            && computed_distinct_on.is_empty()
            && computed_order_by.is_empty();

        if all_simple {
            // Fast path
            let mut projected = RowVec::with_capacity(rows.len());
            let num_select_cols = select_column_indices.len();
            let num_extra_cols = extra_order_indices.len();

            for (row_id, row) in rows.drain_rows().enumerate() {
                let mut values = Vec::with_capacity(num_select_cols + num_extra_cols);
                // Add SELECT columns
                for idx in &select_column_indices {
                    values.push(
                        row.get(idx.unwrap())
                            .cloned()
                            .unwrap_or(Value::null_unknown()),
                    );
                }
                // Add extra ORDER BY columns
                for &idx in &extra_order_indices {
                    values.push(row.get(idx).cloned().unwrap_or(Value::null_unknown()));
                }
                projected.push((row_id as i64, Row::from_values(values)));
            }

            // Output column names will be computed in caller
            Ok((projected, extra_names))
        } else {
            // Slow path: Use Evaluator for complex expressions
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::general_path_reached(
                crate::test_failpoints::GeneralPath::ProjectionEvaluator,
            );
            let mut projected = RowVec::with_capacity(rows.len());

            // Create evaluator once and reuse for all rows
            let mut evaluator = CompiledEvaluator::new(&self.function_registry);
            evaluator = evaluator.with_context(ctx);
            evaluator.init_columns(all_columns);

            let total_extra =
                extra_order_indices.len() + computed_order_by.len() + computed_distinct_on.len();

            // OPTIMIZATION: Reuse col_index_map_lower for O(1) lookup
            for (row_id, row) in rows.drain_rows().enumerate() {
                let mut values = Vec::with_capacity(select_exprs.len() + total_extra);

                evaluator.set_row_array(&row);

                // Evaluate SELECT expressions
                for expr in select_exprs.iter() {
                    let value = self.evaluate_select_expr(
                        &mut evaluator,
                        expr,
                        &row,
                        &col_index_map_lower,
                    )?;
                    values.push(value);
                }

                // Add extra ORDER BY columns
                for &idx in &extra_order_indices {
                    values.push(row.get(idx).cloned().unwrap_or(Value::null_unknown()));
                }

                // Evaluate computed ORDER BY expressions
                for expr in &computed_order_by {
                    values.push(evaluator.evaluate(expr)?);
                }

                // Evaluate computed DISTINCT ON expressions
                for expr in &computed_distinct_on {
                    values.push(evaluator.evaluate(expr)?);
                }

                projected.push((row_id as i64, Row::from_values(values)));
            }

            Ok((projected, extra_names))
        }
    }

    /// Which columns an ORDER BY or DISTINCT ON projection reads by index,
    /// which it computes, and the names of what it appends after the SELECT
    pub(super) fn order_by_projection<'e>(
        &self,
        select_exprs: &'e [Expression],
        order_by: &'e [crate::parser::ast::OrderByExpression],
        distinct_on: &'e [Expression],
        all_columns: &[String],
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
                    ) && !Self::has_subqueries(other)
                        && !Self::reads_a_select_alias(
                            other,
                            &select_aliases,
                            &col_index_map_lower,
                        ) =>
                {
                    let name = other.to_string();
                    if !select_column_names
                        .iter()
                        .any(|s| s.eq_ignore_ascii_case(&name))
                        && !computed_order_by.iter().any(|e| e.to_string() == name)
                    {
                        computed_order_by.push(other);
                    }
                }
                _ => {}
            }
        }

        // A correlated subquery in ORDER BY reads the parent row, so every
        // column rides along until the sort has run
        if order_by
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

        OrderByProjection {
            col_index_map_lower,
            select_column_indices,
            extra_order_indices,
            computed_order_by,
            computed_distinct_on,
            extra_names,
        }
    }

    /// Evaluate a single SELECT expression
    /// OPTIMIZATION: Takes pre-computed column index map for O(1) lookup instead of O(N) linear search
    #[allow(clippy::only_used_in_recursion)]
    fn evaluate_select_expr(
        &self,
        evaluator: &mut CompiledEvaluator,
        expr: &Expression,
        row: &Row,
        col_index_map: &StringMap<usize>,
    ) -> Result<Value> {
        match expr {
            // Simple column reference - get from row by index using O(1) HashMap lookup
            Expression::Identifier(id) => {
                // Use pre-computed value_lower for O(1) lookup - no allocation!
                if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                    Ok(row.get(idx).cloned().unwrap_or(Value::null_unknown()))
                } else {
                    Err(Error::ColumnNotFound(id.value.to_string()))
                }
            }
            // Qualified identifier (table.column) - O(1) lookup
            Expression::QualifiedIdentifier(qid) => {
                // IMPORTANT: Try full qualified name FIRST to handle same-named columns
                // across different tables (e.g., t1.name vs t2.name in JOINs)
                let full_name = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                if let Some(&idx) = col_index_map.get(&full_name) {
                    Ok(row.get(idx).cloned().unwrap_or(Value::null_unknown()))
                } else if let Some(&idx) = col_index_map.get(qid.name.value_lower.as_str()) {
                    // Fallback to unqualified name for single-table queries or aliases
                    Ok(row.get(idx).cloned().unwrap_or(Value::null_unknown()))
                } else {
                    Err(Error::ColumnNotFound(format!(
                        "{}.{}",
                        qid.qualifier.value, qid.name.value
                    )))
                }
            }
            // Aliased expression - evaluate the inner expression
            Expression::Aliased(aliased) => {
                self.evaluate_select_expr(evaluator, &aliased.expression, row, col_index_map)
            }
            // For all other expressions (CAST, arithmetic, function calls, etc.)
            // use the evaluator
            _ => evaluator.evaluate(expr),
        }
    }

    /// Get output column names from SELECT expressions
    pub(crate) fn get_output_column_names(
        &self,
        select_exprs: &[Expression],
        all_columns: &[String],
        table_alias: Option<&str>,
    ) -> Vec<String> {
        let mut names = Vec::with_capacity(select_exprs.len());

        // OPTIMIZATION: Pre-compute lowercase column names once to avoid per-column to_lowercase()
        // Only compute if we have QualifiedStar expressions
        let all_columns_lower: Option<Vec<String>> = if select_exprs
            .iter()
            .any(|e| matches!(e, Expression::QualifiedStar(_)))
        {
            Some(all_columns.iter().map(|c| c.to_lowercase()).collect())
        } else {
            None
        };

        for (i, expr) in select_exprs.iter().enumerate() {
            match expr {
                Expression::Star(_) => {
                    // SELECT * - add all column names
                    names.extend(all_columns.iter().cloned());
                    continue;
                }
                Expression::QualifiedStar(qs) => {
                    // SELECT t.* - add columns matching the qualifier
                    let qualifier_lower = qs.qualifier.to_lowercase();
                    let qualifier_len = qualifier_lower.len();
                    let mut found_any = false;
                    // Use pre-computed lowercase columns
                    if let Some(ref cols_lower) = all_columns_lower {
                        for (idx, col_lower) in cols_lower.iter().enumerate() {
                            // Inline prefix check: "qualifier." without format! allocation
                            if col_lower.len() > qualifier_len
                                && col_lower.starts_with(qualifier_lower.as_str())
                                && col_lower.as_bytes()[qualifier_len] == b'.'
                            {
                                // Strip "qualifier." from the column name for the output
                                // e.g., "e.name" becomes "name"
                                names.push(all_columns[idx][qualifier_len + 1..].to_string());
                                found_any = true;
                            }
                        }
                    }
                    // If no columns matched the prefix (single-table query), check if
                    // the qualifier matches the table alias - if so, include all columns
                    if !found_any {
                        if let Some(alias) = table_alias {
                            if alias.eq_ignore_ascii_case(&qualifier_lower) {
                                names.extend(all_columns.iter().cloned());
                            }
                        }
                    }
                    continue;
                }
                _ => {}
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
        // SELECT * or t.* is a simple case
        if select_exprs.len() == 1
            && matches!(
                &select_exprs[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            )
        {
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

    /// Compare two rows for ORDER BY using pre-computed column indices
    /// This avoids string allocations during sort comparisons
    #[inline]
    fn compare_rows_with_indices(
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
                    self.set_default_isolation_level(isolation);
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
        let value = ExpressionEval::compile(&stmt.expression, &[])?
            .with_context(ctx)
            .eval_slice(&Row::new())?;
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
        let table = tx.get_statement_table(table_name)?;
        let schema = table.schema().clone();
        let all_columns: Vec<String> = schema.column_names_owned().to_vec();

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

        // Check for window functions
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &rows, &all_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check for aggregation
        if classification.has_aggregation {
            let result = self.execute_select_with_aggregation(stmt, ctx, rows, &all_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Project rows
        let (projected_rows, final_columns) =
            self.project_rows_for_select(stmt, rows, &all_columns, ctx)?;

        let final_columns_arc = CompactArc::new(final_columns);
        Ok((
            Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(&final_columns_arc),
                projected_rows,
            )),
            final_columns_arc,
            false,
            None,
        ))
    }

    /// Parse temporal value from AS OF clause
    fn parse_temporal_value(&self, as_of: &AsOfClause, ctx: &ExecutionContext) -> Result<i64> {
        let value = ExpressionEval::compile(&as_of.value, &[])?
            .with_context(ctx)
            .eval_slice(&Row::new())?;

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
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        // Check for SELECT * or t.*
        if stmt.columns.len() == 1
            && matches!(
                &stmt.columns[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            )
        {
            return Ok((rows, all_columns.to_vec()));
        }

        // Use existing project_rows method
        let projected = self.project_rows(&stmt.columns, rows, all_columns, ctx)?;
        // Note: This helper doesn't have table_alias available,
        // so we pass None. The prefix-based matching will still work for JOINs.
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, None);

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
    /// reports the strategy execution would choose.
    pub(crate) fn join_index_use(&self, stmt: &SelectStatement) -> JoinIndexUse {
        let Some(Expression::JoinSource(join)) = stmt.table_expr.as_deref() else {
            return JoinIndexUse::None;
        };
        let classification = get_classification(stmt);
        let join_type = join.join_type.to_uppercase();
        // A WHERE that reaches into the outer query keeps both operators out
        if classification.where_has_correlated_subqueries {
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
                (right.is_some(), Self::filter_has_subquery(where_clause))
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
        let table = txn.get_statement_table(table_name_ref).ok()?;
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

        // HAVING by the inline filter the other aggregate paths run, so every
        // group passes or fails by the same comparison
        let having_filter = match stmt.having.as_deref() {
            Some(having) => match crate::executor::try_parse_simple_having(having, &aggregations) {
                Some(filter) => Some(filter),
                None => return Ok(None),
            },
            None => None,
        };

        let Some(group_column) = all_columns
            .iter()
            .find(|c| c.eq_ignore_ascii_case(&group_col_name))
        else {
            return Ok(None);
        };
        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(group_column.clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));

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

            // One value per aggregate slot, then HAVING over them; only a
            // group that passes clones its group value
            let mut values = Vec::with_capacity(1 + runs.len());
            values.push(Value::null_unknown());
            for run in &runs {
                values.push(run.result()?);
            }
            if having_filter
                .as_ref()
                .is_some_and(|filter| !filter.passes(&values[1..]))
            {
                return Ok(true); // Continue to next group
            }
            values[0] = group_value.clone();
            result_rows.push((result_row_id, Row::from_values(values)));
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
