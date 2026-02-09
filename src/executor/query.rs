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
use rustc_hash::{FxHashMap, FxHashSet};

use crate::common::{CompactArc, CompactVec, StringMap};
use crate::core::{Error, Result, Row, RowVec, Value};
use crate::optimizer::ExpressionSimplifier;
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::mvcc::engine::ViewDefinition;
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
type SelectResult = Result<(
    Box<dyn QueryResult>,
    CompactArc<Vec<String>>,
    bool,
    Option<DeferredProjection>,
)>;

use super::context::{
    clear_batch_aggregate_cache, clear_batch_aggregate_info_cache, clear_count_counter_cache,
    clear_exists_correlation_cache, clear_exists_fetcher_cache, clear_exists_index_cache,
    clear_exists_pred_key_cache, clear_exists_predicate_cache, clear_exists_schema_cache,
    ExecutionContext, TimeoutGuard,
};
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
use super::query_classification::{get_classification, QueryClassification};
use super::result::{
    DistinctResult, ExecResult, ExecutorResult, ExprMappedResult, FilteredResult, LimitedResult,
    OrderedResult, ProjectedResult, RadixOrderSpec, ScannerResult, StreamingProjectionResult,
    TopNResult,
};
use super::utils::compute_join_projection;
use super::utils::{
    add_table_qualifier, build_column_index_map, collect_table_qualifiers,
    combine_predicates_with_and, compare_values, dummy_token, expression_contains_aggregate,
    extract_base_column_name, extract_join_keys_and_residual, filter_references_column,
    flatten_and_predicates, get_table_alias_from_expr, strip_table_qualifier,
    substitute_filter_column,
};
use super::window::{WindowPreGroupedState, WindowPreSortedState};
use super::Executor;
use crate::optimizer::bloom::BloomFilterBuilder;

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

/// Partition WHERE clause predicates for JOIN filter pushdown.
/// Returns (left_filter, right_filter, cross_table_filter).
/// - left_filter: predicates referencing only left table
/// - right_filter: predicates referencing only right table
/// - cross_table_filter: predicates referencing both tables (must be applied post-join)
fn partition_where_for_join(
    where_clause: &Expression,
    left_alias: &str,
    right_alias: &str,
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

        let refs_left = qualifiers.contains(&left_alias_lower);
        let refs_right = qualifiers.contains(&right_alias_lower);

        if refs_left && refs_right {
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

impl Executor {
    /// Execute a SELECT statement
    pub(crate) fn execute_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Start timeout guard ONLY at the top level (query_depth == 0).
        // For nested queries (subqueries, views), the parent's TimeoutGuard handles timeout.
        // This ensures the timeout applies to the entire query, not each nested call.
        let _timeout_guard = if ctx.query_depth == 0 {
            // Clear query-local caches at top-level.
            // NOTE: scalar, IN, and semi-join caches are NOT cleared here - they're
            // invalidated per-table when data changes (INSERT, UPDATE, DELETE, TRUNCATE)
            // to enable cross-query caching for repeated subqueries.
            clear_exists_predicate_cache();
            clear_exists_index_cache();
            clear_exists_fetcher_cache();
            clear_count_counter_cache();
            clear_exists_schema_cache();
            clear_exists_pred_key_cache();
            clear_exists_correlation_cache();
            clear_batch_aggregate_cache();
            clear_batch_aggregate_info_cache();
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

        // Check for CTEs (WITH clause)
        if self.has_cte(stmt) {
            return self.execute_select_with_ctes(stmt, ctx);
        }

        // OPTIMIZATION: Get cached query classification ONCE at entry point
        // This classification is passed through the call chain to avoid
        // redundant hash computations and cache lookups (was 10+ calls per query)
        let classification = get_classification(stmt);

        // Evaluate LIMIT/OFFSET early (needed for set operations optimization)
        let limit = if let Some(ref limit_expr) = stmt.limit {
            match ExpressionEval::compile(limit_expr, &[])?
                .with_context(ctx)
                .eval_slice(&Row::new())?
            {
                Value::Integer(l) => {
                    if l < 0 {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            l
                        )));
                    }
                    Some(l as usize)
                }
                Value::Float(f) => {
                    let l = f as i64;
                    if l < 0 {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            l
                        )));
                    }
                    Some(l as usize)
                }
                other => {
                    return Err(Error::ParseError(format!(
                        "LIMIT must be an integer, got {:?}",
                        other
                    )));
                }
            }
        } else {
            None
        };

        let offset = if let Some(ref offset_expr) = stmt.offset {
            match ExpressionEval::compile(offset_expr, &[])
                .ok()
                .and_then(|eval| eval.with_context(ctx).eval_slice(&Row::new()).ok())
            {
                Some(Value::Integer(o)) => {
                    if o < 0 {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            o
                        )));
                    }
                    o as usize
                }
                Some(Value::Float(f)) => {
                    let o = f as i64;
                    if o < 0 {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            o
                        )));
                    }
                    o as usize
                }
                Some(other) => {
                    return Err(Error::ParseError(format!(
                        "OFFSET must be an integer, got {:?}",
                        other
                    )));
                }
                None => 0,
            }
        } else {
            0
        };

        // Execute the main query
        // The third return value indicates if LIMIT/OFFSET was already applied (by storage-level pushdown)
        // The fourth return value contains deferred projection info if applicable
        let (mut result, columns, limit_offset_applied, deferred_projection) =
            self.execute_select_internal(stmt, ctx, &classification)?;

        // Apply set operations (UNION, INTERSECT, EXCEPT)
        // Pass limit+offset to enable early termination for UNION ALL
        let mut limit_offset_applied = limit_offset_applied;
        if !stmt.set_operations.is_empty() {
            // Only enable limit pushdown for pure UNION ALL (no dedup needed)
            let all_union_all = stmt
                .set_operations
                .iter()
                .all(|op| matches!(op.operation, SetOperationType::UnionAll));
            let set_limit = if all_union_all && stmt.order_by.is_empty() && !stmt.distinct {
                // Only push limit when there's no ORDER BY or DISTINCT that needs full result
                limit.map(|l| l + offset)
            } else {
                None
            };
            result = self.execute_set_operations(result, &stmt.set_operations, ctx, set_limit)?;

            // After set operations, reset limit_offset_applied since we have a new result
            // For UNION ALL with limit, we've already incorporated the limit
            limit_offset_applied = all_union_all && set_limit.is_some();
        }

        // Count expected SELECT columns (before any extra ORDER BY columns)
        let expected_columns = self.count_select_columns(stmt);

        // Apply DISTINCT
        // When ORDER BY references columns not in SELECT, we add extra columns for sorting.
        // DISTINCT should only consider the original SELECT columns, not the extra ORDER BY columns.
        if stmt.distinct {
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
            // Helper to format aggregate function call as column name
            let format_agg_column = |func: &crate::parser::ast::FunctionCall| -> String {
                if func.arguments.is_empty() {
                    format!("{}(*)", func.function)
                } else if let Some(arg) = func.arguments.first() {
                    match arg {
                        Expression::Identifier(id) => format!("{}({})", func.function, id.value),
                        Expression::Star(_) => format!("{}(*)", func.function),
                        _ => format!("{}(expr)", func.function),
                    }
                } else {
                    format!("{}()", func.function)
                }
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
                                Expression::QualifiedIdentifier(sel_qid) => {
                                    // Direct match: ORDER BY c.name matches SELECT c.name
                                    if sel_qid.qualifier.value_lower == qid.qualifier.value_lower
                                        && sel_qid.name.value_lower == qid.name.value_lower
                                    {
                                        return Some(i);
                                    }
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
                                    // Also check if the alias matches the base column name
                                    if aliased.alias.value_lower == qid.name.value_lower {
                                        return Some(i);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Fallback: try matching against column names
                        let base_name = qid.name.value_lower.clone();
                        columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&base_name))
                    }
                    Expression::IntegerLiteral(lit) => Some((lit.value as usize).saturating_sub(1)),
                    Expression::FunctionCall(func) => {
                        // Check if this is an aggregate function that exists as a column
                        let col_name = format_agg_column(func);
                        if let Some(pos) = columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(&col_name))
                        {
                            return Some(pos);
                        }
                        // Also check if any SELECT column is an aliased version of this expression
                        // e.g., SELECT SUM(amount) AS total ... ORDER BY SUM(amount)
                        for (i, select_expr) in stmt.columns.iter().enumerate() {
                            if let Expression::Aliased(aliased) = select_expr {
                                if let Expression::FunctionCall(sel_func) = &*aliased.expression {
                                    if sel_func.function.eq_ignore_ascii_case(&func.function) {
                                        // Compare arguments
                                        let sel_col_name = format_agg_column(sel_func);
                                        if sel_col_name.eq_ignore_ascii_case(&col_name) {
                                            return Some(i);
                                        }
                                    }
                                }
                            }
                        }
                        None
                    }
                    _ => {
                        // For any other expression (Infix, Prefix, Cast, etc.),
                        // check if it matches an aliased SELECT expression
                        // e.g., ORDER BY val * 2 when SELECT has val * 2 as doubled
                        let order_expr_str = format!("{}", ob.expression);
                        for (i, select_expr) in stmt.columns.iter().enumerate() {
                            if let Expression::Aliased(aliased) = select_expr {
                                let aliased_expr_str = format!("{}", aliased.expression);
                                // Compare the string representations of expressions
                                if order_expr_str == aliased_expr_str {
                                    return Some(i);
                                }
                            }
                        }
                        None
                    }
                }
            };

            // Check if any ORDER BY expression needs evaluation (not just column refs or position)
            let has_complex_order_by = stmt
                .order_by
                .iter()
                .any(|ob| try_map_to_column(ob).is_none());

            // If ORDER BY has complex expressions, evaluate them and sort by keys
            if has_complex_order_by {
                // Materialize current result if needed
                let mut rows = RowVec::new();
                let mut row_id_counter = 0i64;
                while result.next() {
                    rows.push((row_id_counter, result.take_row()));
                    row_id_counter += 1;
                }

                // Create evaluator for ORDER BY expressions
                let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                evaluator = evaluator.with_context(ctx);
                evaluator.init_columns(&columns);

                // Check if any ORDER BY expression contains a correlated subquery
                // Use cached classification to avoid expensive AST traversal
                let has_correlated_order_by = classification.order_by_has_correlated_subqueries;

                // OPTIMIZATION: Instead of cloning rows and appending sort keys,
                // compute sort keys separately and use index-based sorting.
                // This avoids O(n * row_size) cloning overhead.
                let num_order_cols = stmt.order_by.len();

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
                                .map(|ob| {
                                    // Try processing correlated subqueries first
                                    if Self::has_correlated_subqueries(&ob.expression) {
                                        match self.process_correlated_expression(
                                            &ob.expression,
                                            &correlated_ctx,
                                        ) {
                                            Ok(processed_expr) => {
                                                let mut corr_eval =
                                                    CompiledEvaluator::new(&self.function_registry);
                                                corr_eval.init_columns(&columns);
                                                corr_eval.set_row_array(row);
                                                corr_eval = corr_eval.with_context(&correlated_ctx);
                                                corr_eval
                                                    .evaluate(&processed_expr)
                                                    .unwrap_or(Value::null_unknown())
                                            }
                                            Err(_) => Value::null_unknown(),
                                        }
                                    } else {
                                        evaluator
                                            .evaluate(&ob.expression)
                                            .unwrap_or_else(|_| Value::null_unknown())
                                    }
                                })
                                .collect()
                        })
                        .collect()
                } else {
                    rows.iter()
                        .map(|(_, row)| {
                            evaluator.set_row_array(row);
                            stmt.order_by
                                .iter()
                                .map(|ob| {
                                    evaluator
                                        .evaluate(&ob.expression)
                                        .unwrap_or_else(|_| Value::null_unknown())
                                })
                                .collect()
                        })
                        .collect()
                };

                // Create indices and sort them based on sort_keys
                let mut indices: Vec<usize> = (0..rows.len()).collect();

                // Use sort_unstable_by for ~10-20% speedup (stability not needed for ORDER BY)
                indices.sort_unstable_by(|&a_idx, &b_idx| {
                    let a_keys = &sort_keys[a_idx];
                    let b_keys = &sort_keys[b_idx];

                    for i in 0..num_order_cols {
                        let ascending = stmt.order_by[i].ascending;
                        let nulls_first = stmt.order_by[i].nulls_first;
                        let a_val = a_keys.get(i);
                        let b_val = b_keys.get(i);

                        // Check if either value is NULL
                        let a_is_null =
                            a_val.is_none() || a_val.map(|v| v.is_null()).unwrap_or(true);
                        let b_is_null =
                            b_val.is_none() || b_val.map(|v| v.is_null()).unwrap_or(true);

                        // Handle NULL comparison
                        if a_is_null || b_is_null {
                            if a_is_null && b_is_null {
                                continue; // Both NULL, move to next column
                            }
                            // Default: NULLS LAST for ASC, NULLS FIRST for DESC
                            let nulls_come_first = nulls_first.unwrap_or(!ascending);
                            return if a_is_null {
                                if nulls_come_first {
                                    Ordering::Less
                                } else {
                                    Ordering::Greater
                                }
                            } else if nulls_come_first {
                                Ordering::Greater
                            } else {
                                Ordering::Less
                            };
                        }

                        let cmp = match (a_val, b_val) {
                            (Some(av), Some(bv)) => av.partial_cmp(bv).unwrap_or(Ordering::Equal),
                            _ => Ordering::Equal,
                        };
                        let cmp = if !ascending { cmp.reverse() } else { cmp };
                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }
                    Ordering::Equal
                });

                // Reorder rows using sorted indices
                // OPTIMIZATION: For LIMIT queries, only collect needed rows
                // For full results, use in-place cycle-based permutation
                let final_rows: RowVec = if limit.is_some() || offset > 0 {
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
                    // No LIMIT: Use in-place cycle-based permutation (no cloning!)
                    // This follows permutation cycles and swaps elements in place
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
                let mut result_rows = final_rows;
                if columns.len() > expected_columns && expected_columns > 0 {
                    for (_, row) in result_rows.iter_mut() {
                        row.truncate(expected_columns);
                    }
                }

                // Use original column names if expected_columns matches
                let output_columns = if expected_columns > 0 && expected_columns <= columns.len() {
                    CompactArc::new(columns[..expected_columns].to_vec())
                } else {
                    CompactArc::clone(&columns)
                };
                return Ok(Box::new(ExecutorResult::with_arc_columns(
                    output_columns,
                    result_rows,
                )));
            }

            // Pre-compute column indices to avoid string comparisons during sort
            // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocations
            // Tuple: (col_idx, ascending, nulls_first)
            let order_specs: Vec<(Option<usize>, bool, Option<bool>)> = stmt
                .order_by
                .iter()
                .map(|ob| (try_map_to_column(ob), ob.ascending, ob.nulls_first))
                .collect();

            // TOP-N OPTIMIZATION: Use bounded heap when LIMIT is present
            // This is O(n log k) instead of O(n log n), where k = limit
            if let Some(lim) = limit {
                // Use TopNResult for ORDER BY + LIMIT (5-50x faster for large datasets)
                result = Box::new(TopNResult::new(
                    result,
                    move |a, b| Self::compare_rows_with_indices(a, b, &order_specs),
                    lim,
                    offset,
                ));

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
            } else {
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
                    ));
                } else {
                    // Some columns missing - use comparison sort
                    result = Box::new(OrderedResult::new(result, move |a, b| {
                        Self::compare_rows_with_indices(a, b, &order_specs)
                    }));
                }
            }
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
                self.execute_simple_table_scan(table_source, stmt, ctx, classification)
            }
            Expression::JoinSource(join_source) => {
                self.execute_join_source(join_source, stmt, ctx, classification)
            }
            Expression::SubquerySource(subquery_source) => {
                self.execute_subquery_source(subquery_source, stmt, ctx, classification)
            }
            Expression::ValuesSource(values_source) => {
                self.execute_values_source(values_source, stmt, ctx, classification)
            }
            _ => Err(Error::NotSupportedMessage(
                "Unsupported FROM clause type".to_string(),
            )),
        }
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

            // Evaluate expression
            let value = ExpressionEval::compile(col_expr, &[])?
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
        // Use the CTE execution logic from cte.rs
        let (result_cols, result_rows) =
            self.execute_query_on_cte_result(stmt, ctx, columns, rows)?;
        let result_cols = CompactArc::new(result_cols);
        Ok((
            Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(&result_cols),
                result_rows,
            )),
            result_cols,
            false,
            None,
        ))
    }

    /// Check if ORDER BY references columns not in SELECT
    /// OPTIMIZATION: Use HashSet for O(1) lookup and eq_ignore_ascii_case to avoid allocations
    fn order_by_needs_extra_columns(&self, stmt: &SelectStatement, all_columns: &[String]) -> bool {
        if stmt.order_by.is_empty() {
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
        let select_columns: FxHashSet<String> = stmt
            .columns
            .iter()
            .filter_map(|expr| self.extract_select_column_name(expr))
            .map(|s| s.to_lowercase())
            .collect();

        // Check if any ORDER BY column is not in SELECT
        for ob in &stmt.order_by {
            if let Expression::Identifier(id) = &ob.expression {
                if !select_columns.contains(id.value_lower.as_str()) {
                    // Verify this column exists in the table (use eq_ignore_ascii_case to avoid allocation)
                    if all_columns
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(id.value_lower.as_str()))
                    {
                        return true;
                    }
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
                    if let Some(idx) = source_columns_lower
                        .iter()
                        .position(|c| c == id.value_lower.as_str())
                    {
                        indices.push(idx);
                        output_names.push(source_columns[idx].clone());
                    } else {
                        return None;
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    if let Some(idx) = source_columns_lower
                        .iter()
                        .position(|c| c == qid.name.value_lower.as_str())
                    {
                        indices.push(idx);
                        output_names.push(source_columns[idx].clone());
                    } else {
                        return None;
                    }
                }
                Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                    Expression::Identifier(id) => {
                        if let Some(idx) = source_columns_lower
                            .iter()
                            .position(|c| c == id.value_lower.as_str())
                        {
                            indices.push(idx);
                            output_names.push(aliased.alias.value.to_string());
                        } else {
                            return None;
                        }
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        if let Some(idx) = source_columns_lower
                            .iter()
                            .position(|c| c == qid.name.value_lower.as_str())
                        {
                            indices.push(idx);
                            output_names.push(aliased.alias.value.to_string());
                        } else {
                            return None;
                        }
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
        if let Some(distinct_values) = table.get_partition_values(&column_name) {
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

    /// Execute a simple table scan
    fn execute_simple_table_scan(
        &self,
        table_source: &SimpleTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &table_source.name.value_lower;

        // Check if there's an active explicit transaction
        let active_tx = self.active_transaction.lock().unwrap();
        let in_explicit_transaction = active_tx.is_some();

        // Get table from active transaction or create a new one
        let (table, _standalone_tx) = if let Some(ref tx_state) = *active_tx {
            // Use the active transaction - this allows seeing uncommitted changes
            let table = tx_state.transaction.get_table(table_name).map_err(|e| {
                if matches!(e, Error::TableNotFound) {
                    Error::TableOrViewNotFound(table_name.to_string())
                } else {
                    e
                }
            })?;
            drop(active_tx); // Release lock before doing work
            (table, None)
        } else {
            drop(active_tx); // Release lock before creating new transaction
                             // No active transaction - create a standalone transaction
            let tx = self.engine.begin_transaction()?;

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

            let table = tx.get_table(table_name).map_err(|e| {
                if matches!(e, Error::TableNotFound) {
                    Error::TableOrViewNotFound(table_name.to_string())
                } else {
                    e
                }
            })?;
            (table, Some(tx))
        };

        // Build column list from schema (using cached version to avoid repeated clones)
        let all_columns: Vec<String> = table.schema().column_names_owned().to_vec();
        // Get pre-cached lowercase column names to avoid per-query to_lowercase() calls
        let all_columns_lower = table.schema().column_names_lower_arc();

        // Get table alias for correlated subquery support
        let table_alias: Option<String> = table_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.to_string())
            .or_else(|| Some(table_name.to_string()));

        // classification is passed from caller to avoid redundant cache lookups

        // AGGREGATION PUSHDOWN: Try to compute simple aggregates directly on storage
        // This avoids all row materialization for queries like:
        // - SELECT COUNT(*) FROM table
        // - SELECT SUM(col), MIN(col), MAX(col) FROM table
        // Must check before any row collection happens
        if classification.has_aggregation && !classification.has_window_functions {
            // First try global aggregation pushdown (no GROUP BY)
            if let Some(result) =
                self.try_aggregation_pushdown(table.as_ref(), stmt, ctx, classification)?
            {
                let columns = CompactArc::new(result.columns().to_vec());
                return Ok((result, columns, false, None));
            }

            // Try storage-level GROUP BY aggregation (no WHERE clause)
            // This computes aggregates directly from arena without materializing Row objects
            if classification.has_group_by && !classification.has_where {
                if let Some(result) =
                    self.try_storage_aggregation(table.as_ref(), stmt, &all_columns, classification)
                {
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

        // Build alias map from SELECT columns for alias substitution in WHERE clause
        let alias_map = Self::build_alias_map(&stmt.columns);

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
            if let Some(where_expr) = where_to_use {
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

        // SEMANTIC CACHE: Check if we can serve this query from cache
        // Eligible queries: simple column projections with WHERE, no aggregation/window/grouping, no outer context
        // Use cached classification for is_select_star check
        let is_select_star = classification.is_select_star;

        let has_aggregation_window_grouping = classification.has_aggregation
            || classification.has_window_functions
            || classification.has_group_by;
        // Use cached classification for subquery detection (avoids AST traversal)
        let has_subqueries_in_where = classification.where_has_subqueries;

        // CRITICAL: Check if WHERE clause contains parameters ($1, $2, etc.)
        // Parameterized queries CANNOT be cached because:
        // 1. The cache stores results tied to specific parameter values
        // 2. But the AST only has parameter indices ($1), not actual values
        // 3. A cache "hit" would return wrong results for different parameter values
        // This was causing 100x slowdown for SELECT by ID queries due to:
        // - Cache misses on every lookup (unique predicates)
        // - Streaming disabled for cache-eligible queries
        // - Cache insertions (write locks) on every query execution
        // Use cached classification for parameter detection (avoids AST traversal)
        let has_parameters_in_where = classification.where_has_parameters;

        // Cache eligibility: We cache SELECT * queries because:
        // 1. The cache stores full table rows with their original column layout
        // 2. Subsumption detection works on full rows for filtering
        // 3. For non-SELECT * queries, we would need to project cached rows on hit
        //
        // CRITICAL: Disable caching during explicit transactions (BEGIN/COMMIT)
        // to preserve MVCC isolation guarantees. A transaction must see its own
        // consistent snapshot, not cached results from other transactions.
        let cache_eligible = is_select_star
            && where_to_use.is_some()
            && !has_aggregation_window_grouping
            && !has_outer_context
            && !has_subqueries_in_where
            && !has_parameters_in_where // Parameters can't be cached (values not in AST)
            && !classification.has_order_by
            && !classification.has_distinct
            && !classification.has_limit
            && !in_explicit_transaction; // MVCC safety: no caching in transactions

        // Try cache lookup for eligible queries
        if cache_eligible {
            if let Some(where_expr) = where_to_use {
                use super::semantic_cache::CacheLookupResult;

                match self
                    .semantic_cache
                    .lookup(table_name, &all_columns, Some(where_expr))
                {
                    CacheLookupResult::ExactHit(rows_arc) => {
                        // Exact cache hit - return cached rows with zero-copy sharing
                        let output_columns = CompactArc::new(self.get_output_column_names(
                            &stmt.columns,
                            &all_columns,
                            table_alias.as_deref(),
                        ));
                        let result = ExecutorResult::with_arc_columns_shared_rows(
                            CompactArc::clone(&output_columns),
                            rows_arc,
                        );
                        return Ok((Box::new(result), output_columns, false, None));
                    }
                    CacheLookupResult::SubsumptionHit {
                        rows: rows_arc,
                        filter,
                        columns,
                    } => {
                        // Subsumption hit - filter cached rows
                        // Clone the Vec since we need to filter (creates new Vec anyway)
                        use super::semantic_cache::SemanticCache;
                        let filtered_vec = SemanticCache::filter_rows(
                            (*rows_arc).clone(),
                            &filter,
                            &columns,
                            &self.function_registry,
                        )?;
                        // Convert Vec<Row> to RowVec
                        let filtered_rows: RowVec = filtered_vec
                            .into_iter()
                            .enumerate()
                            .map(|(i, row)| (i as i64, row))
                            .collect();
                        let output_columns = CompactArc::new(self.get_output_column_names(
                            &stmt.columns,
                            &all_columns,
                            table_alias.as_deref(),
                        ));
                        let result = ExecutorResult::with_arc_columns(
                            CompactArc::clone(&output_columns),
                            filtered_rows,
                        );
                        return Ok((Box::new(result), output_columns, false, None));
                    }
                    CacheLookupResult::Miss => {
                        // Cache miss - continue with normal execution
                        // Result will be inserted into cache below
                    }
                }
            }
        }

        let (storage_expr, needs_memory_filter) = if let Some(where_expr) = where_to_use {
            // If there are subqueries, we must filter in memory
            // Use cached classification to avoid AST traversal
            if classification.where_has_subqueries {
                (None, true)
            } else if has_outer_context {
                // If we have outer row context, this is a correlated subquery
                // OPTIMIZATION: Substitute outer references with their actual values
                // to enable predicate pushdown for index usage.
                // Example: WHERE o.user_id = u.id -> WHERE o.user_id = 42
                if let Some(outer_row) = ctx.outer_row() {
                    let substituted_expr =
                        super::utils::substitute_outer_references(where_expr, outer_row);
                    let schema = table.schema();
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
                    // Zone maps indicate no segments can match - return empty result
                    let output_columns = CompactArc::new(self.get_output_column_names(
                        &stmt.columns,
                        &all_columns,
                        table_alias.as_deref(),
                    ));
                    let result = ExecutorResult::with_arc_columns(
                        CompactArc::clone(&output_columns),
                        RowVec::new(),
                    );
                    return Ok((Box::new(result), output_columns, true, None));
                }
            }
        }

        // FAST PATH: MIN/MAX index optimization
        // For queries like `SELECT MIN(col) FROM table` or `SELECT MAX(col) FROM table`
        // without WHERE or GROUP BY, use the index directly (O(1) instead of O(n))
        if storage_expr.is_none() && !needs_memory_filter && !classification.has_group_by {
            if let Some((result, columns)) =
                self.try_min_max_index_optimization(stmt, &*table, &all_columns)?
            {
                return Ok((result, columns, false, None));
            }
        }

        // FAST PATH: COUNT(*) pushdown optimization
        // For queries like `SELECT COUNT(*) FROM table` without WHERE or GROUP BY,
        // use the table's row_count() method instead of scanning all rows
        if storage_expr.is_none() && !needs_memory_filter && !classification.has_group_by {
            if let Some((result, columns)) = self.try_count_star_optimization(stmt, &*table)? {
                return Ok((result, columns, false, None));
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
        if needs_memory_filter && !has_outer_context && !classification.has_group_by {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns)) = self.try_in_subquery_index_optimization(
                    stmt,
                    where_expr,
                    &*table,
                    &all_columns,
                    table_alias.as_deref(),
                    ctx,
                    classification,
                )? {
                    return Ok((result, columns, false, None));
                }
            }
        }

        // FAST PATH: IN list literal index optimization
        // For queries like `SELECT * FROM table WHERE id IN (1, 2, 3, 5, 8)`
        // where 'id' has an index or is PRIMARY KEY, probe directly instead of scanning all rows
        if needs_memory_filter && !has_outer_context && !classification.has_group_by {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns)) = self.try_in_list_index_optimization(
                    stmt,
                    where_expr,
                    &*table,
                    &all_columns,
                    table_alias.as_deref(),
                    ctx,
                    classification,
                )? {
                    return Ok((result, columns, false, None));
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
            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])?
                    .with_context(ctx)
                    .eval_slice(&Row::new())?
                {
                    Value::Integer(l) if l >= 0 => l as usize,
                    Value::Integer(l) => {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            l
                        )));
                    }
                    Value::Float(f) if f >= 0.0 => f as usize,
                    Value::Float(f) => {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            f
                        )));
                    }
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])?
                    .with_context(ctx)
                    .eval_slice(&Row::new())?
                {
                    Value::Integer(o) if o >= 0 => o as usize,
                    Value::Integer(o) => {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            o
                        )));
                    }
                    Value::Float(f) if f >= 0.0 => f as usize,
                    Value::Float(f) => {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            f
                        )));
                    }
                    _ => 0,
                }
            } else {
                0
            };

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
                let scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

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
                let scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

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

                result = Box::new(ExprMappedResult::new(
                    result,
                    stmt.columns.clone(),
                    (*output_columns).clone(),
                )?);

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

            // OPTIMIZATION: Delay scanner creation until we know we need all rows.
            // For early termination path, we'll collect rows directly with a limit.
            // This avoids materializing all rows upfront when we only need a few.

            // Check if WHERE contains correlated subqueries
            // Use cached classification to avoid expensive AST traversal
            let has_correlated = classification.where_has_subqueries
                && classification.where_has_correlated_subqueries;

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
                    let outer_limit = stmt.limit.as_ref().and_then(|limit_expr| {
                        if let Expression::IntegerLiteral(lit) = limit_expr.as_ref() {
                            Some(lit.value)
                        } else {
                            None
                        }
                    });

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
                        let use_anti_join = is_pure_not_exists && !has_limit;

                        if use_anti_join {
                            #[cfg(debug_assertions)]
                            eprintln!("[ANTI_JOIN] Using anti-join optimization for NOT EXISTS");

                            // Materialize outer table rows
                            let mut outer_rows = table.collect_all_rows(storage_expr.as_deref())?;

                            // Execute anti-join
                            let anti_join_result = self.execute_anti_join(
                                &not_exists_info,
                                CompactArc::new(outer_rows.drain_rows().collect()),
                                &all_columns,
                                ctx,
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

                            // Apply LIMIT if present
                            let final_rows = if let Some(limit_expr) = &stmt.limit {
                                if let Expression::IntegerLiteral(lit) = limit_expr.as_ref() {
                                    let limit = lit.value as usize;
                                    let offset = stmt
                                        .offset
                                        .as_ref()
                                        .and_then(|o| {
                                            if let Expression::IntegerLiteral(lit) = o.as_ref() {
                                                Some(lit.value as usize)
                                            } else {
                                                None
                                            }
                                        })
                                        .unwrap_or(0);
                                    projected_rows
                                        .into_iter()
                                        .skip(offset)
                                        .take(limit)
                                        .collect()
                                } else {
                                    projected_rows
                                }
                            } else {
                                projected_rows
                            };

                            let result = ExecutorResult::with_arc_columns(
                                CompactArc::clone(&output_columns),
                                final_rows,
                            );
                            return Ok((Box::new(result), output_columns, true, None));
                        }
                    }

                    // Try semi-join optimizations for both EXISTS and IN subqueries
                    // These transform O(outer × inner) to O(inner + outer)
                    // Avoid cloning upfront - only clone if no optimization succeeds

                    // 1. Try EXISTS semi-join optimization
                    let exists_optimized = self
                        .try_optimize_exists_to_semi_join(
                            where_expr,
                            ctx,
                            &outer_tables,
                            outer_limit,
                        )
                        .ok()
                        .flatten();

                    // 2. Try IN semi-join optimization (on EXISTS result or original)
                    let expr_for_in = exists_optimized.as_ref().unwrap_or(where_expr);
                    let in_optimized = self
                        .try_optimize_in_to_semi_join(expr_for_in, ctx, &outer_tables)
                        .ok()
                        .flatten();

                    // Determine final expression without unnecessary clones
                    let (current_expr, any_optimized) = match (exists_optimized, in_optimized) {
                        (_, Some(in_opt)) => (in_opt, true),
                        (Some(exists_opt), None) => (exists_opt, true),
                        (None, None) => (where_expr.clone(), false), // Clone only when needed
                    };

                    // Check if there are still correlated subqueries after optimizations
                    let still_correlated = Self::has_correlated_subqueries(&current_expr);

                    if any_optimized && !still_correlated {
                        // All correlated subqueries were optimized away
                        (Some(current_expr), false)
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
            // Use cached classification to avoid AST traversal
            if !has_correlated
                && !classification.has_group_by
                && !classification.select_has_correlated_subqueries
            {
                if let Some(ref where_expr) = processed_where {
                    if let Some((result, columns)) = self.try_in_hashset_index_optimization(
                        stmt,
                        where_expr,
                        &*table,
                        &all_columns,
                        table_alias.as_deref(),
                        ctx,
                    )? {
                        return Ok((result, columns, false, None));
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
                    && !classification.has_window_functions;

                let early_termination_target = if can_early_terminate {
                    let offset = stmt
                        .offset
                        .as_ref()
                        .and_then(|offset_expr| {
                            ExpressionEval::compile(offset_expr, &[])
                                .ok()
                                .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                                .and_then(|v| {
                                    if let Value::Integer(o) = v {
                                        Some(o.max(0) as usize)
                                    } else {
                                        None
                                    }
                                })
                        })
                        .unwrap_or(0);

                    stmt.limit.as_ref().and_then(|limit_expr| {
                        ExpressionEval::compile(limit_expr, &[])
                            .ok()
                            .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                            .and_then(|v| {
                                if let Value::Integer(l) = v {
                                    Some(offset + l.max(0) as usize)
                                } else {
                                    None
                                }
                            })
                    })
                } else {
                    None
                };

                // Use early termination path if we have a target
                if let Some(target) = early_termination_target {
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
                                if memory_eval.eval_bool(&row) {
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
                // Now we create the scanner since we need all rows
                let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;
                let mut all_rows = RowVec::new();
                while scanner.next() {
                    all_rows.push(scanner.take_row_with_id());
                }

                // Apply parallel filtering if we have enough rows
                // CRITICAL: Propagate errors with ? instead of silently swallowing them
                let filtered = parallel::parallel_filter(
                    all_rows,
                    where_expr,
                    &all_columns,
                    &self.function_registry,
                    &parallel_config,
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
                    Some(CompactArc::new(all_columns.clone()))
                } else {
                    None
                };

                // LIMIT EARLY TERMINATION: For correlated subqueries without ORDER BY,
                // we can stop as soon as we have enough matching rows.
                // This turns O(outer_size) EXISTS evaluations into O(LIMIT) evaluations.
                let early_termination_target: Option<usize> =
                    if has_correlated && stmt.order_by.is_empty() {
                        let offset = stmt
                            .offset
                            .as_ref()
                            .and_then(|offset_expr| {
                                ExpressionEval::compile(offset_expr, &[])
                                    .ok()
                                    .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                                    .and_then(|v| {
                                        if let Value::Integer(o) = v {
                                            Some(o.max(0) as usize)
                                        } else {
                                            None
                                        }
                                    })
                            })
                            .unwrap_or(0);

                        stmt.limit.as_ref().and_then(|limit_expr| {
                            ExpressionEval::compile(limit_expr, &[])
                                .ok()
                                .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                                .and_then(|v| {
                                    if let Value::Integer(l) = v {
                                        Some(offset + l.max(0) as usize)
                                    } else {
                                        None
                                    }
                                })
                        })
                    } else {
                        None
                    };

                // Create scanner for the correlated subquery path
                // For correlated subqueries, we can't push down the WHERE clause to storage
                // because it depends on outer row values that change per row
                let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

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
                (rows, None, None)
            }
        } else if storage_expr.is_some() {
            // Path 2: WHERE clause with pushdown - use scanner for index optimization
            // Note: We fetch all columns here because downstream projection uses all_columns indices
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

            // OPTIMIZATION: Use take_row() to avoid cloning each row
            // Pre-allocate to reduce reallocations - 64 avoids first 6 grow operations
            let mut rows = RowVec::with_capacity(64);
            let mut row_count = 0u64;
            while scanner.next() {
                // Check for cancellation every 100 rows
                row_count += 1;
                if row_count.is_multiple_of(100) {
                    ctx.check_cancelled()?;
                }
                // Use row_count as synthetic row ID for scanner-based paths
                rows.push((row_count as i64, scanner.take_row()));
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
                        if !has_order_by {
                            if let Some(limit_expr) = &stmt.limit {
                                if let Expression::IntegerLiteral(lit) = limit_expr.as_ref() {
                                    if lit.value > 0 {
                                        let limit_val = lit.value as usize;
                                        // Use lazy partition fetching - returns early!
                                        let result = self
                                            .execute_select_with_window_functions_lazy_partition(
                                                stmt,
                                                ctx,
                                                table.as_ref(),
                                                &all_columns,
                                                &partition_col,
                                                limit_val,
                                            );
                                        if let Ok(query_result) = result {
                                            let columns =
                                                CompactArc::new(query_result.columns().to_vec());
                                            return Ok((query_result, columns, false, None));
                                        }
                                        // Fall through to regular path if optimization fails
                                    }
                                }
                            }
                        }

                        // Regular path: Fetch rows grouped by partition (no hash grouping needed)
                        if let Some(grouped_data) =
                            table.collect_rows_grouped_by_partition(&partition_col)
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
                                Some(WindowPreGroupedState { partition_map }),
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
                    let has_index = is_pk || table.get_index_on_column(&col_name).is_some();

                    if has_index {
                        // OPTIMIZATION: If we have LIMIT without top-level ORDER BY,
                        // push the limit down to fetch only needed rows.
                        // This is safe ONLY for window functions that don't depend on total row count.
                        // Safe: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, LAST_VALUE
                        // Unsafe: NTILE, PERCENT_RANK, CUME_DIST (need total count)
                        let has_order_by = !stmt.order_by.is_empty();
                        let is_window_safe = Self::is_window_safe_for_limit_pushdown(stmt);
                        let fetch_limit = if !has_order_by && is_window_safe {
                            if let Some(limit_expr) = &stmt.limit {
                                if let Expression::IntegerLiteral(lit) = limit_expr.as_ref() {
                                    if lit.value > 0 {
                                        lit.value as usize
                                    } else {
                                        usize::MAX
                                    }
                                } else {
                                    usize::MAX
                                }
                            } else {
                                usize::MAX
                            }
                        } else {
                            usize::MAX
                        };

                        // Fetch rows in sorted order from the index (no re-fetch needed)
                        if let Some(sorted_rows) = table.collect_rows_ordered_by_index(
                            &col_name,
                            ascending,
                            fetch_limit,
                            0,
                        ) {
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
            let agg_result = self.execute_aggregation_for_window(stmt, ctx, &rows, &all_columns)?;
            let agg_columns = agg_result.0.clone();
            let agg_rows = agg_result.1;

            // Apply window functions on aggregated rows
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &agg_rows, &agg_columns)?;
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

        let (projected_rows, output_columns, deferred_proj) = if order_by_needs_extra_columns {
            // When ORDER BY references columns not in SELECT, we need to:
            // 1. Include those columns in the output (appended at end)
            // 2. Sort will happen in execute_select
            // 3. Extra columns will be projected out after sorting
            let (projected_rows, _) = self.project_rows_with_order_by(
                &stmt.columns,
                &stmt.order_by,
                rows,
                &all_columns,
                ctx,
            )?;
            // Get base column names
            let mut output_columns =
                self.get_output_column_names(&stmt.columns, &all_columns, table_alias.as_deref());
            // Append ORDER BY columns not in SELECT
            // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocations
            for ob in &stmt.order_by {
                if let Expression::Identifier(id) = &ob.expression {
                    if !output_columns
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(&id.value_lower))
                    {
                        output_columns.push(id.value.to_string());
                    }
                }
            }
            (projected_rows, output_columns, None)
        } else if let Some((col_indices, output_names)) = deferred_projection_info {
            // DEFERRED PROJECTION: Skip projection now, do it after ORDER BY + LIMIT
            // Return all source columns; projection will be applied after TopNResult
            (
                rows,
                all_columns.to_vec(),
                Some((col_indices, output_names)),
            )
        } else {
            // Standard projection
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                &all_columns,
                Some(&all_columns_lower),
                ctx,
                table_alias.as_deref(),
            )?;
            let output_columns =
                self.get_output_column_names(&stmt.columns, &all_columns, table_alias.as_deref());
            (projected_rows, output_columns, None)
        };

        // SEMANTIC CACHE: Insert result for eligible queries
        // For SELECT * queries, cache the raw rows before returning
        //
        // Note: The cache stores Vec<Row>, so we extract rows from RowVec for caching.
        // The result keeps the original RowVec.
        // Skip caching when deferred projection is used (rows are not projected yet)
        if cache_eligible && deferred_proj.is_none() {
            if let Some(where_expr) = where_to_use {
                // Clone rows for cache (cache needs Vec<Row>)
                let rows_for_cache: Vec<Row> = projected_rows.rows().cloned().collect();
                self.semantic_cache.insert(
                    table_name,
                    all_columns.clone(),
                    rows_for_cache,
                    Some(where_expr.clone()),
                );
            }
        }

        let output_columns = CompactArc::new(output_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, false, deferred_proj))
    }

    /// Execute a JOIN source
    fn execute_join_source(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> SelectResult {
        // classification is passed from caller to avoid redundant cache lookups

        // Get table aliases for filter pushdown
        let left_alias = get_table_alias_from_expr(&join_source.left);
        let right_alias = get_table_alias_from_expr(&join_source.right);

        // Determine join type early for filter pushdown decisions
        let join_type = join_source.join_type.to_uppercase();

        // Partition WHERE clause predicates for pushdown
        // Note: For OUTER JOINs, we must be careful:
        // - LEFT JOIN: Can push filters to left (preserved), but NOT to right (may have NULLs)
        // - RIGHT JOIN: Can push filters to right (preserved), but NOT to left
        // - FULL OUTER JOIN: Cannot push filters to either side
        let (left_filter, right_filter, cross_filter) = if let Some(ref where_clause) =
            stmt.where_clause
        {
            if let (Some(left_a), Some(right_a)) = (&left_alias, &right_alias) {
                let (l, r, c) = partition_where_for_join(where_clause, left_a, right_a);
                #[cfg(debug_assertions)]
                eprintln!("[FILTER_PARTITION] left_alias={}, right_alias={}, left_filter={:?}, right_filter={:?}, cross_filter={:?}",
                        left_a, right_a, l.as_ref().map(|_| "Some"), r.as_ref().map(|_| "Some"), c.as_ref().map(|_| "Some"));

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

                #[cfg(debug_assertions)]
                eprintln!(
                    "[FILTER_RESULT] safe_left={}, safe_right={}, remaining={}",
                    safe_left.is_some(),
                    safe_right.is_some(),
                    remaining.is_some()
                );
                (safe_left, safe_right, remaining)
            } else {
                (None, None, Some((**where_clause).clone()))
            }
        } else {
            (None, None, None)
        };
        #[cfg(debug_assertions)]
        eprintln!(
            "[FILTERS_FINAL] left_filter={}, right_filter={}, cross_filter={}",
            left_filter.is_some(),
            right_filter.is_some(),
            cross_filter.is_some()
        );

        // Semi-join reduction optimization for LEFT JOIN + GROUP BY + LIMIT
        // Pattern: LEFT JOIN + GROUP BY on left columns only + LIMIT N + no ORDER BY
        // Optimization: limit left side first, filter right side with IN clause (uses index)
        // This reduces materialization from O(L + R) to O(N + N*avg_matches)
        let semijoin_limit = self.get_semijoin_reduction_limit(
            &join_type,
            stmt,
            left_alias.as_deref(),
            &join_source.condition,
        );
        #[cfg(debug_assertions)]
        eprintln!(
            "[SEMIJOIN_CHECK] semijoin_limit={}",
            semijoin_limit.is_some()
        );

        let (left_rows, left_columns, right_rows, right_columns) = if let Some((
            limit_n,
            left_key_col,
            right_key_col,
        )) = semijoin_limit
        {
            #[cfg(debug_assertions)]
            eprintln!(
                "[SEMIJOIN_PATH] limit={}, left_key={}, right_key={}",
                limit_n, left_key_col, right_key_col
            );
            // Semi-join reduction for INNER/LEFT JOIN + GROUP BY
            // Step 1: Execute and materialize left side with limit (pushdown for efficiency)
            let (left_result, left_cols) = self.execute_table_expression_with_filter_limit(
                &join_source.left,
                ctx,
                left_filter.as_ref(),
                Some(limit_n),
            )?;
            let left_rows = Self::materialize_result_arc(left_result)?;

            // Step 2: Extract join key values from limited left rows
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

            // Step 3: Build combined filter for right side with IN clause
            let right_filter_with_in = if !join_key_values.is_empty() {
                // Create IN expression: right_key_col IN (v1, v2, ..., vN)
                let in_expr = self.build_in_filter_expression(&right_key_col, &join_key_values);
                // Combine with existing right filter if any
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

            // Step 4: Execute right side with IN filter (uses index on right_key_col)
            let (right_result, right_cols) = self.execute_table_expression_with_filter(
                &join_source.right,
                ctx,
                right_filter_with_in.as_ref(),
            )?;
            let right_rows = Self::materialize_result_arc(right_result)?;

            (left_rows, left_cols, right_rows, right_cols)
        } else {
            // Skip Index Nested Loop if query has aggregation or window functions
            // These require full result sets and can't use early termination
            let has_agg = classification.has_aggregation;
            let has_window = classification.has_window_functions;

            // Check if we can use Index Nested Loop Join
            // This optimization avoids materializing the right side entirely
            // NOTE: Don't use Index NL for aggregation/window queries - they need full results
            // and the current implementation falls through to standard path, causing double execution
            let index_nl_info = if has_agg || has_window {
                None
            } else {
                self.check_index_nested_loop_opportunity(
                    &join_source.right,
                    join_source.condition.as_ref().map(|c| c.as_ref()),
                    &join_type,
                    left_alias.as_deref(),
                    right_alias.as_deref(),
                )
            };

            // Subquery join optimization: When right side is a subquery (not a table)
            // and left side is a table with PK/index, use swapped Index NL.
            // This handles inlined CTEs: (table) JOIN (subquery) -> subquery outer, table inner
            let (index_nl_info, force_swap) = if index_nl_info.is_none()
                && !has_agg
                && !has_window
                && (join_type == "INNER" || join_type == "LEFT")
                && !matches!(join_source.right.as_ref(), Expression::TableSource(_))
                && !matches!(
                    join_source.right.as_ref(),
                    Expression::Aliased(a) if matches!(a.expression.as_ref(), Expression::TableSource(_))
                ) {
                // Right is subquery/CTE - check if left side has Index NL opportunity
                let left_as_inner = self.check_index_nested_loop_opportunity(
                    &join_source.left,
                    join_source.condition.as_ref().map(|c| c.as_ref()),
                    &join_type,
                    right_alias.as_deref(), // Swap aliases for the check
                    left_alias.as_deref(),
                );
                if left_as_inner.is_some() {
                    (left_as_inner, true) // Force swap
                } else {
                    (None, false)
                }
            } else {
                (index_nl_info, false)
            };

            // Join reordering optimization for INNER JOINs:
            // When one side has a filter, prefer putting filtered side as outer (left)
            // This reduces the number of probes into the inner table.
            // Swap if: right has filter, left doesn't, and swapped order gives Index NL on PK
            #[cfg(debug_assertions)]
            eprintln!("[JOIN_REORDER_PRE] force_swap={}, has_agg={}, has_window={}, join_type={}, right_filter={}, left_filter={}",
                force_swap, has_agg, has_window, join_type, right_filter.is_some(), left_filter.is_some());
            let (index_nl_info, nl_left_filter, nl_right_filter, swapped) = if force_swap {
                // Subquery join optimization: swap is forced (right is subquery, left is table)
                (
                    index_nl_info,
                    right_filter.clone(), // Subquery becomes outer, apply its filter
                    left_filter.clone(),  // Table becomes inner
                    true,
                )
            } else if !has_agg
                    && !has_window
                    && join_type == "INNER"
                    && right_filter.is_some()  // Right side has a filter
                    && left_filter.is_none()
            // Left side doesn't have a filter
            {
                #[cfg(debug_assertions)]
                eprintln!(
                    "[SWAP_CHECK] Checking swap for right_filter={:?}",
                    right_filter.as_ref().map(|f| f.to_string())
                );
                #[cfg(debug_assertions)]
                eprintln!("[SWAP_CONDITIONS] has_agg={}, has_window={}, join_type={}, right_filter={}, left_filter={}",
                    has_agg, has_window, join_type, right_filter.is_some(), left_filter.is_none());
                // Check if swapping gives Index NL opportunity with PK lookup
                // (which is more efficient than secondary index lookup)
                let swapped_info = self.check_index_nested_loop_opportunity(
                    &join_source.left, // Left becomes inner (right)
                    join_source.condition.as_ref().map(|c| c.as_ref()),
                    &join_type,
                    right_alias.as_deref(), // Swap aliases
                    left_alias.as_deref(),
                );

                // Prefer swapped if it gives PK lookup (most efficient)
                let prefer_swap = matches!(
                    &swapped_info,
                    Some((_, IndexLookupStrategy::PrimaryKey, _, _))
                );
                #[cfg(debug_assertions)]
                eprintln!(
                    "[SWAP_CHECK] swapped_info={:?}, prefer_swap={}",
                    swapped_info
                        .as_ref()
                        .map(|(t, _, i, o)| format!("{},{},{}", t, i, o)),
                    prefer_swap
                );

                if prefer_swap {
                    // Swap: right filter becomes outer filter
                    (
                        swapped_info,
                        right_filter.clone(),
                        left_filter.clone(),
                        true,
                    )
                } else {
                    (
                        index_nl_info,
                        left_filter.clone(),
                        right_filter.clone(),
                        false,
                    )
                }
            } else {
                (
                    index_nl_info,
                    left_filter.clone(),
                    right_filter.clone(),
                    false,
                )
            };

            #[cfg(debug_assertions)]
            eprintln!(
                "[INDEX_NL_CHECK] index_nl_info={:?}, nl_left_filter={:?}, nl_right_filter={:?}",
                index_nl_info
                    .as_ref()
                    .map(|(t, _, i, o)| format!("table={},inner={},outer={}", t, i, o)),
                nl_left_filter.as_ref().map(|_| "Some"),
                nl_right_filter.as_ref().map(|_| "Some")
            );
            if let Some((table_name, lookup_strategy, inner_col, outer_col)) = index_nl_info {
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
                let nl_left_filter = if let Some(ref right_f) = nl_right_filter {
                    // Check if right filter references the inner join key column
                    let references = filter_references_column(right_f, &inner_col);
                    #[cfg(debug_assertions)]
                    eprintln!("[JOIN_KEY_EQUIV] inner_col={:?}, outer_col={:?}, filter_references={}, right_filter={:?}",
                        inner_col, outer_col, references, right_f);
                    if references {
                        // Create equivalent filter for outer side by substituting the column
                        if let Some(outer_filter) =
                            substitute_filter_column(right_f, &inner_col, &outer_col)
                        {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[JOIN_KEY_EQUIV] SUCCESS! Created outer filter: {:?}",
                                outer_filter
                            );
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
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[JOIN_KEY_EQUIV] FAILED: substitute_filter_column returned None"
                            );
                            nl_left_filter
                        }
                    } else {
                        nl_left_filter
                    }
                } else {
                    nl_left_filter
                };

                // Compute join limit EARLY so we can use it for outer table optimization
                let can_push_limit = !join_type.contains("FULL")
                    && !classification.has_order_by
                    && !classification.has_group_by
                    && !classification.has_aggregation;

                let join_limit = if can_push_limit {
                    stmt.limit.as_ref().and_then(|limit_expr| {
                        ExpressionEval::compile(limit_expr, &[])
                            .ok()
                            .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                            .and_then(|v| match v {
                                Value::Integer(n) if n >= 0 => Some(n as u64),
                                _ => None,
                            })
                    })
                } else {
                    None
                };

                // For outer table, use a reasonable limit multiplier
                // We need enough rows to produce join_limit results, assuming some miss rate
                // Use 4x multiplier as heuristic (handles up to 75% miss rate)
                let outer_limit = join_limit.map(|l| (l as usize).saturating_mul(4).max(100));

                // Execute outer side with limit optimization for true early termination
                let (outer_result, outer_cols) = self.execute_table_expression_with_filter_limit(
                    outer_expr,
                    ctx,
                    nl_left_filter.as_ref(),
                    outer_limit,
                )?;

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
                    let txn = self.engine.begin_transaction()?;
                    let inner_table = txn.get_table(&table_name)?;
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

                    // Create residual filter from nl_right_filter if present
                    // This allows early termination during Index Nested Loop
                    let residual_filter = if let Some(ref rf) = nl_right_filter {
                        // Re-qualify the filter with inner table alias
                        let qualified_rf = add_table_qualifier(rf, inner_alias);
                        JoinFilter::new(
                            &qualified_rf,
                            &outer_cols,
                            &inner_cols,
                            &self.function_registry,
                        )
                        .ok()
                        .map(|f| f.with_context(ctx))
                    } else {
                        None
                    };

                    // Execute Index Nested Loop Join using operators
                    // Use batch version for NO LIMIT (reduces lock overhead from O(N) to O(1))
                    // Use streaming version for LIMIT queries (supports early termination)

                    // Convert to operator types
                    let outer_op: Box<dyn Operator> =
                        Box::new(QueryResultOperator::new(outer_result, outer_cols.clone()));
                    let inner_schema_info: Vec<ColumnInfo> =
                        inner_cols.iter().map(ColumnInfo::new).collect();
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

                    let mut join_op: Box<dyn Operator> = if join_limit.is_some() {
                        // Streaming INL for early termination with LIMIT
                        let op = IndexNestedLoopJoinOperator::new(
                            outer_op,
                            inner_table,
                            inner_schema_info,
                            op_join_type,
                            outer_idx,
                            lookup_strategy.clone(),
                            residual_filter,
                        );

                        // Apply projection pushdown if available
                        if let Some(ref proj) = projection_pushdown {
                            let projected_schema: Vec<ColumnInfo> =
                                proj.output_columns.iter().map(ColumnInfo::new).collect();
                            Box::new(op.with_projection(proj.columns.clone(), projected_schema))
                        } else {
                            Box::new(op)
                        }
                    } else {
                        // Batch INL for NO LIMIT - single batch fetch, O(1) lock overhead
                        let op = BatchIndexNestedLoopJoinOperator::new(
                            outer_op,
                            inner_table,
                            inner_schema_info,
                            op_join_type,
                            outer_idx,
                            lookup_strategy.clone(),
                            residual_filter,
                        );

                        // Apply projection pushdown if available
                        if let Some(ref proj) = projection_pushdown {
                            let projected_schema: Vec<ColumnInfo> =
                                proj.output_columns.iter().map(ColumnInfo::new).collect();
                            Box::new(op.with_projection(proj.columns.clone(), projected_schema))
                        } else {
                            Box::new(op)
                        }
                    };

                    // Execute and collect results with synthetic row IDs
                    join_op.open()?;
                    let mut result_rows = RowVec::new();
                    let mut row_id = 0i64;
                    while let Some(row_ref) = join_op.next()? {
                        result_rows.push((row_id, row_ref.into_owned()));
                        row_id += 1;
                        if let Some(lim) = join_limit {
                            if result_rows.len() >= lim as usize {
                                break;
                            }
                        }
                    }
                    join_op.close()?;

                    // No rotation needed - all_columns matches physical order
                    let mut final_rows = result_rows;

                    // Apply cross-table WHERE filters if any
                    if let Some(ref cross) = cross_filter {
                        let filter = RowFilter::new(cross, &all_columns)?.with_context(ctx);
                        final_rows.retain(|(_, row)| filter.matches(row));
                    }

                    // Apply ORDER BY if present
                    if !stmt.order_by.is_empty() {
                        // Build sort specs by evaluating ORDER BY expressions
                        let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                        evaluator = evaluator.with_context(ctx);
                        evaluator.init_columns(&all_columns);

                        // Compute sort keys and indices
                        let sort_keys: Vec<Vec<Value>> = final_rows
                            .iter()
                            .map(|(_, row)| {
                                evaluator.set_row_array(row);
                                stmt.order_by
                                    .iter()
                                    .map(|ob| {
                                        evaluator
                                            .evaluate(&ob.expression)
                                            .unwrap_or(Value::null_unknown())
                                    })
                                    .collect()
                            })
                            .collect();

                        // Sort by indices using sort_unstable_by for ~10-20% speedup
                        let mut indices: Vec<usize> = (0..final_rows.len()).collect();
                        indices.sort_unstable_by(|&a, &b| {
                            for (i, ob) in stmt.order_by.iter().enumerate() {
                                let av = &sort_keys[a][i];
                                let bv = &sort_keys[b][i];
                                let asc = ob.ascending;
                                let nulls_first = ob.nulls_first.unwrap_or(!asc);

                                let cmp = if av.is_null() && bv.is_null() {
                                    Ordering::Equal
                                } else if av.is_null() {
                                    if nulls_first {
                                        Ordering::Less
                                    } else {
                                        Ordering::Greater
                                    }
                                } else if bv.is_null() {
                                    if nulls_first {
                                        Ordering::Greater
                                    } else {
                                        Ordering::Less
                                    }
                                } else {
                                    compare_values(av, bv)
                                };

                                let cmp = if asc { cmp } else { cmp.reverse() };
                                if cmp != Ordering::Equal {
                                    return cmp;
                                }
                            }
                            Ordering::Equal
                        });

                        // Reorder rows
                        final_rows = indices.into_iter().map(|i| final_rows[i].clone()).collect();

                        // Apply LIMIT/OFFSET after sorting
                        let offset = stmt
                            .offset
                            .as_ref()
                            .and_then(|e| {
                                ExpressionEval::compile(e, &[])
                                    .ok()
                                    .and_then(|eval| {
                                        eval.with_context(ctx).eval_slice(&Row::new()).ok()
                                    })
                                    .and_then(|v| match v {
                                        Value::Integer(n) if n >= 0 => Some(n as usize),
                                        _ => None,
                                    })
                            })
                            .unwrap_or(0);

                        let limit = stmt
                            .limit
                            .as_ref()
                            .and_then(|e| {
                                ExpressionEval::compile(e, &[])
                                    .ok()
                                    .and_then(|eval| {
                                        eval.with_context(ctx).eval_slice(&Row::new()).ok()
                                    })
                                    .and_then(|v| match v {
                                        Value::Integer(n) if n >= 0 => Some(n as usize),
                                        _ => None,
                                    })
                            })
                            .unwrap_or(usize::MAX);

                        final_rows = final_rows.into_iter().skip(offset).take(limit).collect();
                    }

                    // Check for aggregation/window functions that need special handling
                    let has_agg = classification.has_aggregation;
                    let has_window = classification.has_window_functions;

                    if has_agg || has_window {
                        // Fall through to standard path for aggregation/window handling
                        // Don't return early - let the standard path handle these
                    } else if let Some(proj) = projection_pushdown {
                        // Projection was pushed down - rows are already projected
                        let output_columns = CompactArc::new(proj.output_columns);
                        let result = ExecutorResult::with_arc_columns(
                            CompactArc::clone(&output_columns),
                            final_rows,
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
                        return Ok((Box::new(result), output_columns, false, None));
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
            // - No cross-table predicates (applied after join)

            let streaming_limit = if !join_type.contains("FULL")
                && !join_type.contains("LEFT")
                && !join_type.contains("RIGHT")
                && !classification.has_order_by
                && !classification.has_group_by
                && !classification.has_aggregation
                && cross_filter.is_none()
            {
                // Compute limit value
                stmt.limit.as_ref().and_then(|limit_expr| {
                    ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                        .and_then(|v| match v {
                            Value::Integer(n) if (0..=100).contains(&n) => Some(n as u64),
                            _ => None,
                        })
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

            // Use streaming path if eligible
            if let Some(limit) = streaming_limit {
                if has_equality_keys_for_streaming {
                    // Estimate cardinalities to choose optimal build side (smaller = build)
                    let left_card = self
                        .estimate_table_expr_cardinality(&join_source.left, left_filter.as_ref());
                    let right_card = self
                        .estimate_table_expr_cardinality(&join_source.right, right_filter.as_ref());
                    let build_left = left_card < right_card;

                    // Execute both sides
                    let (left_result, left_cols) = self.execute_table_expression_with_filter(
                        &join_source.left,
                        ctx,
                        left_filter.as_ref(),
                    )?;

                    let (right_result, right_cols) = self.execute_table_expression_with_filter(
                        &join_source.right,
                        ctx,
                        right_filter.as_ref(),
                    )?;

                    // Extract join keys BEFORE moving columns (uses original left/right positions)
                    let (left_key_indices, right_key_indices, _) =
                        if let Some(cond) = join_source.condition.as_ref() {
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

                    // Convert probe side to streaming operator
                    let probe_source: Box<dyn Operator> =
                        Box::new(QueryResultOperator::new(probe_result, probe_cols.clone()));

                    // Build hash table and bloom filter together in a single pass
                    // This avoids iterating build_rows twice (once for bloom, once for hash table)
                    //
                    // Skip bloom filter for small LIMIT queries - the overhead of building
                    // a bloom filter for thousands of rows isn't worth it when we only
                    // need a few results. With streaming + early termination, we don't
                    // probe many rows anyway.
                    let skip_bloom_for_small_limit = limit <= 500;
                    let (bloom_filter, pre_built_hash_table) = if build_rows.len() >= 100
                        && !build_key_indices.is_empty()
                        && !skip_bloom_for_small_limit
                    {
                        let mut builder = BloomFilterBuilder::new(
                            "join_key".to_string(),
                            "build".to_string(),
                            build_rows.len(),
                        );
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
                        condition: join_source.condition.as_ref().map(|c| c.as_ref()),
                        join_type: &join_type,
                        build_is_left,
                        limit: Some(limit),
                        ctx,
                        bloom_filter,
                        pre_built_hash_table,
                    };

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
            }

            // =================================================================
            // STANDARD PATH: Materialize both sides
            // =================================================================

            // Execute both sides
            let (left_result, left_cols) = self.execute_table_expression_with_filter(
                &join_source.left,
                ctx,
                left_filter.as_ref(),
            )?;

            let (right_result, right_cols) = self.execute_table_expression_with_filter(
                &join_source.right,
                ctx,
                right_filter.as_ref(),
            )?;

            let left_rows = Self::materialize_result_arc(left_result)?;
            let right_rows = Self::materialize_result_arc(right_result)?;

            (left_rows, left_cols, right_rows, right_cols)
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

        // Use natural join condition if present, otherwise use explicit condition
        let effective_condition = natural_join_cond
            .as_ref()
            .or(join_source.condition.as_ref().map(|c| c.as_ref()));

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
        // Safe to push when: no ORDER BY, no GROUP BY/aggregation, no FULL OUTER
        let can_push_limit = !join_type.contains("FULL")
            && !classification.has_order_by
            && cross_filter.is_none()
            && !classification.has_group_by
            && !classification.has_aggregation;

        let join_limit = if can_push_limit {
            stmt.limit.as_ref().and_then(|limit_expr| {
                ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                    .and_then(|v| match v {
                        Value::Integer(n) if n >= 0 => Some(n as u64),
                        _ => None,
                    })
            })
        } else {
            None
        };

        // Get join algorithm decision from QueryPlanner
        // This uses cost-based optimization with edge-aware heuristics
        let (left_key_indices, right_key_indices, _) = if let Some(cond) = effective_condition {
            extract_join_keys_and_residual(cond, &left_columns, &right_columns)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let has_equality_keys = !left_key_indices.is_empty();

        // Check if inputs are sorted on join keys
        let left_sorted =
            has_equality_keys && super::utils::is_sorted_on_keys(&left_rows, &left_key_indices);
        let right_sorted =
            has_equality_keys && super::utils::is_sorted_on_keys(&right_rows, &right_key_indices);

        // Get algorithm decision from QueryPlanner
        let algorithm_decision = self.get_query_planner().plan_runtime_join_with_sort_info(
            left_rows.len(),
            right_rows.len(),
            has_equality_keys,
            left_sorted,
            right_sorted,
        );

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
            algorithm_hint: Some(&algorithm_decision),
        };

        let join_result = join_executor.execute(join_request)?;
        let result_rows = join_result.rows;

        // Build alias map for alias substitution
        let alias_map = Self::build_alias_map(&stmt.columns);

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
            stmt.where_clause.as_ref().map(|wc| (**wc).clone())
        };

        let resolved_where_clause = if !alias_map.is_empty() {
            effective_where
                .as_ref()
                .map(|where_expr| Box::new(Self::substitute_aliases(where_expr, &alias_map)))
        } else {
            effective_where.map(Box::new)
        };

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = resolved_where_clause {
            // Process subqueries (EXISTS, scalar subqueries) before evaluation
            // Use cached classification to avoid AST traversal
            let processed_where = if classification.where_has_subqueries {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Create RowFilter once and reuse
            let where_filter = RowFilter::new(&processed_where, &all_columns)?.with_context(ctx);

            result_rows
                .into_iter()
                .filter(|(_, row)| where_filter.matches(row))
                .collect()
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
                (all_columns.clone(), filtered_rows)
            }
        } else {
            (all_columns.clone(), filtered_rows)
        };

        let has_agg = classification.has_aggregation;
        let has_window = classification.has_window_functions;

        // Check if we have both aggregation and window functions
        if has_agg && has_window {
            // 1. First apply GROUP BY aggregation
            // 2. Then apply window functions on the aggregated result
            let agg_result =
                self.execute_aggregation_for_window(stmt, ctx, &final_rows, &final_columns)?;
            let agg_columns = agg_result.0.clone();
            let agg_rows = agg_result.1;

            // Apply window functions on aggregated rows (agg_rows is already RowVec with IDs)
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &agg_rows, &agg_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check if we need window functions only (no aggregation)
        if has_window {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &final_rows, &final_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Check if we need aggregation only
        if has_agg {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, final_rows, &final_columns)?;
            let columns = CompactArc::new(result.columns().to_vec());
            return Ok((result, columns, false, None));
        }

        // Project rows according to SELECT expressions
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

        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        Ok((Box::new(result), output_columns, false, None))
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
                self.try_streaming_derived_table_aggregation(result, stmt, classification)?
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

        // Apply WHERE clause if present
        let filtered_rows: RowVec = if let Some(ref where_clause) = stmt.where_clause {
            let where_filter = RowFilter::new(where_clause, &columns)?.with_context(ctx);

            rows.into_iter()
                .filter(|(_, row)| where_filter.matches(row))
                .collect()
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

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, filtered_rows, &columns, None, ctx, None)?;

        // Determine output column names
        let subquery_alias = subquery_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str());
        let output_columns =
            CompactArc::new(self.get_output_column_names(&stmt.columns, &columns, subquery_alias));

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
                return Err(Error::InvalidArgumentMessage(
                    "View definition is not a SELECT statement".to_string(),
                ));
            }
            return Ok(cached.statement);
        }

        // Parse the view's query
        let statements = crate::parser::parse_sql(view_query).map_err(|e| {
            Error::InvalidArgumentMessage(format!("Failed to parse view query: {}", e))
        })?;

        // The view query should be a single SELECT statement
        if statements.len() != 1 {
            return Err(Error::InvalidArgumentMessage(
                "View definition must contain exactly one statement".to_string(),
            ));
        }

        // Validate it's a SELECT statement
        if !matches!(&statements[0], Statement::Select(_)) {
            return Err(Error::InvalidArgumentMessage(
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
            return Err(Error::InvalidArgumentMessage(format!(
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
        let mut result: Box<dyn QueryResult> = result;
        if let Some(ref where_clause) = stmt.where_clause {
            let filter_expr = where_clause.as_ref().clone();
            result = Box::new(FilteredResult::with_defaults(result, filter_expr));
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

            // Execute aggregation on the view's rows
            let agg_result =
                self.execute_select_with_aggregation(stmt, ctx, rows, &view_columns)?;
            let columns = CompactArc::new(agg_result.columns().to_vec());
            return Ok((agg_result, columns, false, None));
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
                        return Err(Error::ColumnNotFoundNamed(id.value.to_string()));
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
                            return Err(Error::ColumnNotFoundNamed(id.value.to_string()));
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

            Box::new(ExprMappedResult::with_defaults(
                result,
                stmt.columns.clone(),
                final_output_columns.clone(),
            ))
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
                if wf.matches(&row) || qf.matches(&row) {
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

    /// Execute a table expression (for JOIN left/right sides)
    /// Returns columns prefixed with table alias for proper qualified name resolution
    pub(crate) fn execute_table_expression(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        self.execute_table_expression_with_filter(expr, ctx, None)
    }

    /// Execute a table expression with optional filter pushdown
    /// This is used by JOIN to push WHERE predicates to individual table scans
    pub(crate) fn execute_table_expression_with_filter(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        #[cfg(debug_assertions)]
        eprintln!(
            "[EXEC_TABLE_EXPR_FILTER] expr_type={:?}, filter={:?}",
            std::mem::discriminant(expr),
            filter.map(|f| f.to_string())
        );
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
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[CTE_FILTER] table={}, filter={:?}, columns={:?}, qualified={:?}, rows_before={}",
                            table_name, filter_expr.to_string(), columns, qualified_columns, rows.len()
                        );
                        let row_filter =
                            RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
                        // CTE stores CompactArc<Vec<(i64, Row)>>, filter on the Row part
                        let filtered_rows: RowVec = rows
                            .iter()
                            .filter(|(_, row)| row_filter.matches(row))
                            .cloned()
                            .collect();
                        #[cfg(debug_assertions)]
                        eprintln!("[CTE_FILTER] rows_after={}", filtered_rows.len());
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
                        return Err(Error::InvalidArgumentMessage(format!(
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
                    return Ok((result, qualified_columns));
                }

                // Create a SELECT * statement with optional WHERE clause
                let select_all = SelectStatement {
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
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
                    self.execute_simple_table_scan(ts, &select_all, ctx, &classification)?;

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
                    self.execute_join_source(js, &select_all, ctx, &classification)?;
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
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[SUBQUERY_FILTER] filter={:?}, columns={:?}, qualified={:?}",
                        filter_expr.to_string(),
                        columns,
                        qualified_columns
                    );
                    // Use qualified_columns for filter column resolution since WHERE clause
                    // has qualified names like "ds.avg_salary" not just "avg_salary"
                    let row_filter =
                        RowFilter::new(filter_expr, &qualified_columns)?.with_context(ctx);
                    // Materialize and filter - materialized is RowVec = Vec<(i64, Row)>
                    let mut materialized = Self::materialize_result(result)?;
                    materialized.retain(|(_, row)| row_filter.matches(row));
                    #[cfg(debug_assertions)]
                    eprintln!("[SUBQUERY_FILTER] rows_after={}", materialized.len());
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
            _ => Err(Error::NotSupportedMessage(
                "Unsupported table expression type".to_string(),
            )),
        }
    }

    /// Execute a table expression with optional filter and row limit.
    /// For simple table sources, adds LIMIT to enable true early termination.
    /// This is optimized for Index Nested Loop joins where we need only enough rows
    /// to produce the requested LIMIT results.
    pub(crate) fn execute_table_expression_with_filter_limit(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        filter: Option<&Expression>,
        row_limit: Option<usize>,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>)> {
        // Extract TableSource from the expression (handles both direct and aliased)
        let (ts, custom_alias) = match expr {
            Expression::TableSource(ts) => (ts, None),
            Expression::Aliased(aliased) => {
                if let Expression::TableSource(ts) = aliased.expression.as_ref() {
                    (ts, Some(aliased.alias.value.clone()))
                } else {
                    // Not a table source, fall back to standard execution
                    return self.execute_table_expression_with_filter(expr, ctx, filter);
                }
            }
            _ => {
                // Not a table source, fall back to standard execution
                return self.execute_table_expression_with_filter(expr, ctx, filter);
            }
        };

        // Only optimize if we have a limit
        if let Some(limit) = row_limit {
            let table_name = ts.name.value_lower.as_str();

            // Skip if this is a CTE reference - CTEs are already materialized
            if ctx.get_cte_by_lower(table_name).is_some() {
                return self.execute_table_expression_with_filter(expr, ctx, filter);
            }

            // Skip if this is a view
            if self.engine.get_view_lowercase(table_name)?.is_some() {
                return self.execute_table_expression_with_filter(expr, ctx, filter);
            }

            // Create a SELECT * statement with WHERE clause AND LIMIT
            // This triggers the LIMIT pushdown optimization in execute_simple_table_scan
            let select_all = SelectStatement {
                token: dummy_token("SELECT", TokenType::Keyword),
                distinct: false,
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
                offset: None,
                set_operations: vec![],
            };
            // Get classification for the synthetic SELECT statement
            let classification = get_classification(&select_all);
            let (result, columns, _, _) =
                self.execute_simple_table_scan(ts, &select_all, ctx, &classification)?;

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

            return Ok((result, qualified_columns));
        }

        // Fall back to standard execution for other cases
        #[cfg(debug_assertions)]
        eprintln!(
            "[EXEC_EXPR_LIMIT_FALLBACK] calling execute_table_expression_with_filter, filter={:?}",
            filter.map(|f| f.to_string())
        );
        self.execute_table_expression_with_filter(expr, ctx, filter)
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
        Ok(rows)
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
                        args.push(ArgSource::Const(Value::Text(lit.value.clone())));
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
                Expression::StringLiteral(lit) => {
                    Some(ArgSource::Const(Value::Text(lit.value.clone())))
                }
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
                Expression::StringLiteral(lit) => Some(Value::Text(lit.value.clone())),
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
                        } else if let Some(col_idx) = get_col_idx(&infix.right, col_index_map_lower)
                        {
                            if let Some(value) = expr_to_value(&infix.left) {
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
                            } else {
                                return None;
                            }
                        } else {
                            return None;
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
                                    (CaseCondition::Equals { value, .. }, Some(v)) => v == value,
                                    (CaseCondition::NotEquals { value, .. }, Some(v)) => v != value,
                                    (CaseCondition::GreaterThan { value, .. }, Some(v)) => {
                                        v > value
                                    }
                                    (CaseCondition::GreaterOrEqual { value, .. }, Some(v)) => {
                                        v >= value
                                    }
                                    (CaseCondition::LessThan { value, .. }, Some(v)) => v < value,
                                    (CaseCondition::LessOrEqual { value, .. }, Some(v)) => {
                                        v <= value
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
                                    Some(_) => {
                                        all_text = false;
                                        break;
                                    }
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
                                // Slow path: mixed types, use Arc to avoid shrink_to_fit
                                let mut result = String::with_capacity(64);
                                for part in parts.iter() {
                                    let val: Option<&Value> = match part {
                                        ArgSource::Column(idx) => row.get(*idx),
                                        ArgSource::Const(v) => Some(v),
                                    };
                                    match val {
                                        Some(Value::Text(s)) => result.push_str(s.as_str()),
                                        Some(Value::Integer(i)) => {
                                            use std::fmt::Write;
                                            let _ = write!(result, "{}", i);
                                        }
                                        Some(Value::Float(f)) => {
                                            use std::fmt::Write;
                                            let _ = write!(result, "{}", f);
                                        }
                                        Some(Value::Boolean(b)) => {
                                            result.push_str(if *b { "true" } else { "false" });
                                        }
                                        Some(v) => result.push_str(&v.to_string()),
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
    fn project_rows_with_order_by(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        mut rows: RowVec,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<(RowVec, Vec<String>)> {
        // Process scalar subqueries in SELECT columns before evaluation (single-pass)
        let processed = self.try_process_select_subqueries(select_exprs, ctx)?;
        let select_exprs = match &processed {
            Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
            None => std::borrow::Cow::Borrowed(select_exprs),
        };

        // Build column index map ONCE with FxHashMap for O(1) lookup
        let col_index_map_lower = build_column_index_map(all_columns);

        // Get SELECT column names (lowercase) for checking duplicates
        let select_column_names: Vec<String> = select_exprs
            .iter()
            .filter_map(|expr| self.extract_select_column_name(expr))
            .map(|s| s.to_lowercase())
            .collect();

        // Find ORDER BY columns not in SELECT
        let mut extra_order_indices: Vec<usize> = Vec::new();
        for ob in order_by {
            if let Expression::Identifier(id) = &ob.expression {
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
                    select_column_indices.push(
                        col_index_map_lower
                            .get(qid.name.value_lower.as_str())
                            .copied(),
                    );
                }
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => {
                        select_column_indices
                            .push(col_index_map_lower.get(id.value_lower.as_str()).copied());
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        select_column_indices.push(
                            col_index_map_lower
                                .get(qid.name.value_lower.as_str())
                                .copied(),
                        );
                    }
                    _ => select_column_indices.push(None),
                },
                _ => select_column_indices.push(None),
            }
        }

        // Check if we can use fast path (all simple column refs)
        let all_simple = select_column_indices.iter().all(|idx| idx.is_some());

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
            Ok((projected, vec![]))
        } else {
            // Slow path: Use Evaluator for complex expressions
            let mut projected = RowVec::with_capacity(rows.len());

            // Create evaluator once and reuse for all rows
            let mut evaluator = CompiledEvaluator::new(&self.function_registry);
            evaluator = evaluator.with_context(ctx);
            evaluator.init_columns(all_columns);

            // OPTIMIZATION: Reuse col_index_map_lower for O(1) lookup
            for (row_id, row) in rows.drain_rows().enumerate() {
                let mut values = Vec::with_capacity(select_exprs.len() + extra_order_indices.len());

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

                projected.push((row_id as i64, Row::from_values(values)));
            }

            Ok((projected, vec![]))
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
                    Err(Error::ColumnNotFoundNamed(id.value.to_string()))
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
                    Err(Error::ColumnNotFoundNamed(format!(
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
                let a_val = a.get(*idx);
                let b_val = b.get(*idx);

                // Check if either value is NULL
                let a_is_null = a_val.is_none() || a_val.map(|v| v.is_null()).unwrap_or(true);
                let b_is_null = b_val.is_none() || b_val.map(|v| v.is_null()).unwrap_or(true);

                // Handle NULL comparison
                if a_is_null || b_is_null {
                    if a_is_null && b_is_null {
                        continue; // Both NULL, move to next column
                    }
                    // Determine if NULLs should come first
                    // Default: NULLS LAST for ASC, NULLS FIRST for DESC
                    let nulls_come_first = nulls_first.unwrap_or(!*ascending);
                    let cmp = if a_is_null {
                        if nulls_come_first {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    } else {
                        // b_is_null
                        if nulls_come_first {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    };
                    return cmp;
                }

                // Both non-NULL - normal comparison
                let cmp = match (a_val, b_val) {
                    (Some(av), Some(bv)) => av.partial_cmp(bv).unwrap_or(Ordering::Equal),
                    _ => Ordering::Equal, // Already handled above
                };

                let cmp = if !*ascending { cmp.reverse() } else { cmp };

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
        _stmt: &BeginStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use super::ActiveTransaction;

        let mut active_tx = self.active_transaction.lock().unwrap();

        // If there's already an active transaction, treat as no-op
        if active_tx.is_some() {
            return Ok(Box::new(ExecResult::empty()));
        }

        // Start a new transaction
        let transaction = self.engine.begin_transaction()?;

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
                // Get the savepoint timestamp first
                let timestamp = tx_state
                    .transaction
                    .get_savepoint_timestamp(&savepoint_name.value)
                    .ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "savepoint '{}' does not exist",
                            savepoint_name.value
                        ))
                    })?;

                // Rollback each table's local changes that occurred after the savepoint
                for table in tx_state.tables.values() {
                    table.rollback_to_timestamp(timestamp);
                }

                // Rollback to savepoint in the transaction (removes the savepoint and any after it)
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
            // Full ROLLBACK - ends the transaction
            if let Some(mut tx_state) = active_tx.take() {
                // Rollback all tables first
                for (_name, mut table) in tx_state.tables.drain() {
                    table.rollback();
                }
                // Then rollback the transaction
                tx_state.transaction.rollback()?;
                Ok(Box::new(ExecResult::empty()))
            } else {
                // No active transaction - this is a no-op
                Ok(Box::new(ExecResult::empty()))
            }
        }
    }

    /// Execute SET statement
    pub(crate) fn execute_set(
        &self,
        _stmt: &SetStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute PRAGMA statement
    pub(crate) fn execute_pragma(
        &self,
        stmt: &PragmaStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let pragma_name = stmt.name.value.to_uppercase();

        match pragma_name.as_str() {
            "SNAPSHOT" => {
                // Handle PRAGMA SNAPSHOT - creates a manual snapshot
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA SNAPSHOT does not accept values"));
                }

                // Create a snapshot
                self.engine.create_snapshot()?;

                // Return success result
                let columns = vec!["result".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::text("Snapshot created successfully")]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "CHECKPOINT" => {
                // Alias for SNAPSHOT (SQLite-style)
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA CHECKPOINT does not accept values"));
                }

                self.engine.create_snapshot()?;

                let columns = vec!["result".to_string()];
                let mut rows = RowVec::with_capacity(1);
                rows.push((
                    0,
                    Row::from_values(vec![Value::text("Checkpoint created successfully")]),
                ));
                Ok(Box::new(ExecutorResult::new(columns, rows)))
            }
            "SNAPSHOT_INTERVAL" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    // Set mode: PRAGMA snapshot_interval = 60
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.snapshot_interval = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    // Read mode: PRAGMA snapshot_interval
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.snapshot_interval as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "KEEP_SNAPSHOTS" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
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
            "SYNC_MODE" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.sync_mode = match new_value {
                        0 => crate::storage::SyncMode::None,
                        1 => crate::storage::SyncMode::Normal,
                        2 => crate::storage::SyncMode::Full,
                        _ => {
                            return Err(Error::internal(
                                "sync_mode must be 0 (none), 1 (normal), or 2 (full)",
                            ))
                        }
                    };
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(config.persistence.sync_mode as i64)]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            "WAL_FLUSH_TRIGGER" => {
                let config = self.engine.config();
                let columns: Vec<String> = vec![pragma_name.to_lowercase().into()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.wal_flush_trigger = new_value as usize;
                    self.engine.update_engine_config(new_config)?;
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(new_value)])));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                } else {
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((
                        0,
                        Row::from_values(vec![Value::Integer(
                            config.persistence.wal_flush_trigger as i64,
                        )]),
                    ));
                    Ok(Box::new(ExecutorResult::new(columns, rows)))
                }
            }
            _ => {
                // Unknown pragma - return empty result for compatibility
                Ok(Box::new(ExecResult::empty()))
            }
        }
    }

    /// Extract integer value from PRAGMA value expression
    fn extract_pragma_int_value(&self, value: &crate::parser::Expression) -> Result<i64> {
        match value {
            crate::parser::Expression::IntegerLiteral(lit) => Ok(lit.value),
            crate::parser::Expression::FloatLiteral(lit) => Ok(lit.value as i64),
            _ => Err(Error::internal("PRAGMA value must be an integer")),
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
        let table = tx.get_table(table_name)?;
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

                rows.retain(|(_, row)| where_filter.matches(row));
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
    fn build_alias_map(columns: &[Expression]) -> FxHashMap<String, &Expression> {
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
                // Store reference instead of clone - avoids expensive Expression clone/drop
                alias_map.insert(alias_name, aliased.expression.as_ref());
            }
        }
        alias_map
    }

    /// Substitute column aliases in an expression with their underlying expressions
    fn substitute_aliases(
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

    /// Check if semi-join reduction optimization can be applied and return parameters
    /// Returns Some((limit_value, left_key_col, right_key_col)) if applicable, None otherwise
    ///
    /// Conditions for semi-join reduction:
    /// 1. INNER JOIN or LEFT JOIN (not RIGHT or FULL)
    /// 2. GROUP BY references only left table columns
    /// 3. LIMIT is present with no ORDER BY (for correctness)
    /// 4. Single equality join condition (a.col = b.col)
    ///
    /// For INNER JOIN, we over-fetch left rows (2x limit) since some may not have matches.
    /// For LEFT JOIN, exact limit is used since all left rows produce output.
    fn get_semijoin_reduction_limit(
        &self,
        join_type: &str,
        stmt: &SelectStatement,
        left_alias: Option<&str>,
        join_condition: &Option<Box<Expression>>,
    ) -> Option<(usize, String, String)> {
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

        // Check that GROUP BY references only left table columns
        let left_alias = left_alias?;
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
            let limit_value = stmt.limit.as_ref().and_then(|e| {
                ExpressionEval::compile(e, &[])
                    .ok()
                    .and_then(|mut eval| eval.eval_slice(&Row::new()).ok())
                    .and_then(|v| match v {
                        Value::Integer(n) if n > 0 => Some(n as usize),
                        _ => None,
                    })
            })?;
            // For INNER JOIN, over-fetch to account for rows without matches
            // Use 2x multiplier as a balance between accuracy and performance
            let adjusted_limit = if is_inner {
                limit_value * 2
            } else {
                limit_value
            };
            return Some((adjusted_limit, right_key_col, left_key_col));
        }

        // Get LIMIT value
        let limit_value = stmt.limit.as_ref().and_then(|e| {
            ExpressionEval::compile(e, &[])
                .ok()
                .and_then(|mut eval| eval.eval_slice(&Row::new()).ok())
                .and_then(|v| match v {
                    Value::Integer(n) if n > 0 => Some(n as usize),
                    _ => None,
                })
        })?;

        // For INNER JOIN, over-fetch to account for rows without matches
        // Use 2x multiplier as a balance between accuracy and performance
        let adjusted_limit = if is_inner {
            limit_value * 2
        } else {
            limit_value
        };

        Some((adjusted_limit, left_key_col, right_key_col))
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
    pub(crate) fn check_index_nested_loop_opportunity(
        &self,
        right_expr: &Expression,
        join_condition: Option<&Expression>,
        join_type: &str,
        left_alias: Option<&str>,
        right_alias: Option<&str>,
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
        // Note: Index NL does not support AS OF (temporal) queries - they need full table scan
        let (table_name, effective_alias) = match right_expr {
            Expression::TableSource(ts) => {
                // Disable Index NL for temporal (AS OF) queries
                if ts.as_of.is_some() {
                    return None;
                }
                (ts.name.value_lower.to_string(), None)
            }
            Expression::Aliased(aliased) => {
                match aliased.expression.as_ref() {
                    Expression::TableSource(ts) => {
                        // Disable Index NL for temporal (AS OF) queries
                        if ts.as_of.is_some() {
                            return None;
                        }
                        (ts.name.value_lower.to_string(), None)
                    }
                    Expression::SubquerySource(sq) => {
                        // Check if subquery is a simple passthrough
                        if let Some(underlying) = self.extract_simple_subquery_table(&sq.subquery) {
                            // Use the subquery's alias as the effective alias for join condition matching
                            let alias = sq.alias.as_ref().map(|a| a.value.to_string());
                            (underlying, alias)
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            Expression::SubquerySource(sq) => {
                // Check if subquery is a simple passthrough
                if let Some(underlying) = self.extract_simple_subquery_table(&sq.subquery) {
                    let alias = sq.alias.as_ref().map(|a| a.value.to_string());
                    (underlying, alias)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        let table_name_ref = &table_name;

        // Check if it's a CTE (CTEs don't have indexes)
        // We can't check CTEs here without context, but we'll verify when we try to get the table

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
        let txn = self.engine.begin_transaction().ok()?;
        let table = txn.get_table(table_name_ref).ok()?;
        let schema = table.schema();

        // First check if inner column is the PRIMARY KEY (direct row_id lookup)
        let inner_col_lower = inner_col_unqualified.to_lowercase();
        if let Some(pk_idx) = schema.pk_column_index() {
            if schema.columns[pk_idx].name_lower == inner_col_lower {
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
        if let Some(index) = table.get_index_on_column(&inner_col_unqualified) {
            return Some((
                table_name,
                IndexLookupStrategy::SecondaryIndex(index),
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
        let table_expr = stmt.table_expr.as_ref()?;
        match table_expr.as_ref() {
            Expression::TableSource(ts) => Some(ts.name.value_lower.to_string()),
            Expression::Aliased(aliased) => {
                if let Expression::TableSource(ts) = aliased.expression.as_ref() {
                    Some(ts.name.value_lower.to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Build an IN filter expression: column_name IN (v1, v2, ..., vN)
    pub(crate) fn build_in_filter_expression(
        &self,
        column_name: &str,
        values: &[Value],
    ) -> Option<Expression> {
        if values.is_empty() {
            return None;
        }

        // Parse the column name to handle qualified names like "o.user_id"
        let col_expr = if let Some(dot_pos) = column_name.find('.') {
            let qualifier = &column_name[..dot_pos];
            let name = &column_name[dot_pos + 1..];
            Expression::QualifiedIdentifier(QualifiedIdentifier {
                token: Token::new(TokenType::Identifier, qualifier, Position::default()),
                qualifier: Box::new(Identifier::new(
                    Token::new(TokenType::Identifier, qualifier, Position::default()),
                    qualifier.to_string(),
                )),
                name: Box::new(Identifier::new(
                    Token::new(TokenType::Identifier, name, Position::default()),
                    name.to_string(),
                )),
            })
        } else {
            Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, column_name, Position::default()),
                column_name.to_string(),
            ))
        };

        // Build list of value literals.
        // IntegerLiteral uses "0" token (Display uses self.value, not token.literal).
        // FloatLiteral must use f.to_string() (Display uses token.literal).
        let value_exprs: Vec<Expression> = values
            .iter()
            .map(|v| match v {
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
                Value::Boolean(b) => Expression::BooleanLiteral(BooleanLiteral {
                    token: Token::new(
                        TokenType::Keyword,
                        if *b { "true" } else { "false" },
                        Position::default(),
                    ),
                    value: *b,
                }),
                _ => Expression::NullLiteral(NullLiteral {
                    token: Token::new(TokenType::Keyword, "NULL", Position::default()),
                }),
            })
            .collect();

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

    /// Extract simple aggregations from SELECT columns
    /// Returns Vec of (function_name, column_name, optional_alias)
    fn extract_aggregations_simple(
        &self,
        stmt: &SelectStatement,
    ) -> Vec<(String, String, Option<String>)> {
        let mut result = Vec::new();

        for col in &stmt.columns {
            match col {
                Expression::FunctionCall(fc) => {
                    let func_upper = fc.function.to_uppercase();
                    if matches!(func_upper.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX") {
                        let col_name = if fc.arguments.is_empty() {
                            "*".to_string()
                        } else if let Some(Expression::Star(_)) = fc.arguments.first() {
                            "*".to_string()
                        } else if let Some(Expression::Identifier(id)) = fc.arguments.first() {
                            id.value_lower.to_string()
                        } else {
                            continue; // Skip complex expressions
                        };
                        result.push((func_upper.into(), col_name, None));
                    }
                }
                Expression::Aliased(aliased) => {
                    if let Expression::FunctionCall(fc) = aliased.expression.as_ref() {
                        let func_upper = fc.function.to_uppercase();
                        if matches!(func_upper.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX") {
                            let col_name = if fc.arguments.is_empty() {
                                "*".to_string()
                            } else if let Some(Expression::Star(_)) = fc.arguments.first() {
                                "*".to_string()
                            } else if let Some(Expression::Identifier(id)) = fc.arguments.first() {
                                id.value_lower.to_string()
                            } else {
                                continue; // Skip complex expressions
                            };
                            result.push((
                                func_upper.into(),
                                col_name,
                                Some(aliased.alias.value.to_string()),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        result
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
        use crate::core::IndexType;

        // Only single-column GROUP BY is supported for now
        if stmt.group_by.columns.len() != 1 {
            return Ok(None);
        }

        // Check if GROUP BY is a simple column reference
        let group_col_name: String = match &stmt.group_by.columns[0] {
            Expression::Identifier(id) => id.value_lower.to_string(),
            _ => return Ok(None),
        };

        // Check for B-tree or primary key index on GROUP BY column
        let btree_index = match table.get_index_on_column(&group_col_name) {
            Some(idx)
                if idx.index_type() == IndexType::BTree
                    || idx.index_type() == IndexType::PrimaryKey =>
            {
                idx
            }
            _ => return Ok(None),
        };

        // Extract aggregations from SELECT columns
        let aggregations = self.extract_aggregations_simple(stmt);
        if aggregations.is_empty() {
            return Ok(None);
        }

        // Check for simple aggregates (SUM, COUNT, AVG, MIN, MAX)
        #[derive(Clone, Copy)]
        enum StreamingAgg {
            Count,
            Sum(usize), // column index
            Avg(usize), // column index (computed as sum/count)
            Min(usize), // column index
            Max(usize), // column index
        }

        let mut simple_aggs: Vec<StreamingAgg> = Vec::with_capacity(aggregations.len());
        for (func_name, col_name, _alias) in &aggregations {
            match func_name.as_str() {
                "COUNT" => simple_aggs.push(StreamingAgg::Count),
                "SUM" | "AVG" | "MIN" | "MAX" => {
                    // Find the column index for aggregate argument
                    let col_idx = all_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(col_name));
                    match col_idx {
                        Some(idx) => match func_name.as_str() {
                            "SUM" => simple_aggs.push(StreamingAgg::Sum(idx)),
                            "AVG" => simple_aggs.push(StreamingAgg::Avg(idx)),
                            "MIN" => simple_aggs.push(StreamingAgg::Min(idx)),
                            "MAX" => simple_aggs.push(StreamingAgg::Max(idx)),
                            _ => unreachable!(),
                        },
                        None => return Ok(None),
                    }
                }
                _ => return Ok(None), // Unsupported aggregate (STDDEV, VARIANCE, etc.)
            }
        }

        // OPTIMIZATION: Don't use streaming GROUP BY when row fetch is needed without LIMIT.
        // Streaming benefits from early termination (LIMIT), but without it, bulk fetch
        // of all rows is more efficient than per-group fetching, even with buffer reuse.
        let needs_row_fetch = simple_aggs
            .iter()
            .any(|a| !matches!(a, StreamingAgg::Count));

        // Only use streaming for COUNT-only queries or when LIMIT allows early termination
        if needs_row_fetch && stmt.limit.is_none() {
            return Ok(None); // Fall back to non-streaming (faster for SUM/AVG/MIN/MAX)
        }

        // Check for simple HAVING filter
        let having_filter: Option<(usize, f64, bool)> = if let Some(ref having) = stmt.having {
            // Only support simple comparisons: agg_expr > constant
            match &**having {
                Expression::Infix(infix) => {
                    if infix.operator != ">" && infix.operator != ">=" {
                        return Ok(None);
                    }
                    // Left side should be an aggregate function
                    let agg_idx = match infix.left.as_ref() {
                        Expression::FunctionCall(fc) => {
                            let func_upper = fc.function.to_uppercase();
                            aggregations
                                .iter()
                                .position(|(name, _, _)| name == func_upper.as_str())
                        }
                        _ => None,
                    };
                    // Right side should be a constant
                    let threshold = match infix.right.as_ref() {
                        Expression::IntegerLiteral(n) => Some(n.value as f64),
                        Expression::FloatLiteral(f) => Some(f.value),
                        _ => None,
                    };
                    match (agg_idx, threshold) {
                        (Some(idx), Some(thresh)) => Some((idx, thresh, infix.operator == ">=")),
                        _ => return Ok(None), // Unsupported HAVING
                    }
                }
                _ => return Ok(None),
            }
        } else {
            None
        };

        // Build result columns
        let mut result_columns = Vec::with_capacity(1 + aggregations.len());
        result_columns.push(group_col_name.clone());
        for (func_name, col_name, alias) in &aggregations {
            let col_name = if let Some(ref a) = alias {
                a.clone()
            } else if col_name == "*" {
                format!("{}(*)", func_name)
            } else {
                format!("{}({})", func_name, col_name)
            };
            result_columns.push(col_name);
        }

        // Streaming aggregation: iterate through groups in sorted order
        let mut result_rows = RowVec::new();
        let mut result_row_id = 0i64;
        let num_aggs = simple_aggs.len();

        // Parse LIMIT for early termination
        // With streaming aggregation, we can stop once we have LIMIT groups that pass HAVING
        let limit_for_early_exit = stmt.limit.as_ref().and_then(|limit_expr| {
            ExpressionEval::compile(limit_expr, &[])
                .ok()
                .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()).ok())
                .and_then(|v| match v {
                    Value::Integer(n) if n > 0 => Some(n as usize),
                    _ => None,
                })
        });

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

        let iteration_result =
            btree_index.for_each_group(&mut |group_value: &Value, row_ids: &[i64]| {
                // Aggregate state: sums for SUM/AVG, min/max values, counts
                let mut agg_sums = vec![0.0f64; num_aggs];
                let mut agg_mins = vec![f64::MAX; num_aggs];
                let mut agg_maxs = vec![f64::MIN; num_aggs];
                let mut agg_has_value = vec![false; num_aggs];
                let mut counts = vec![0i64; num_aggs];

                // Optimization: For COUNT-only aggregates, use row_ids.len() directly
                let row_count = row_ids.len() as i64;

                if needs_row_fetch {
                    // Use the reusable buffer for row fetching
                    row_buffer.clear();
                    table.fetch_rows_by_ids_into(row_ids, &true_expr, &mut row_buffer);

                    for (_row_id, row) in &row_buffer {
                        for (i, agg) in simple_aggs.iter().enumerate() {
                            match agg {
                                StreamingAgg::Count => {
                                    counts[i] += 1;
                                }
                                StreamingAgg::Sum(col_idx) | StreamingAgg::Avg(col_idx) => {
                                    if let Some(value) = row.get(*col_idx) {
                                        match value {
                                            Value::Integer(v) => {
                                                agg_sums[i] += *v as f64;
                                                counts[i] += 1;
                                                agg_has_value[i] = true;
                                            }
                                            Value::Float(v) => {
                                                agg_sums[i] += v;
                                                counts[i] += 1;
                                                agg_has_value[i] = true;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                StreamingAgg::Min(col_idx) => {
                                    if let Some(value) = row.get(*col_idx) {
                                        let v = match value {
                                            Value::Integer(v) => Some(*v as f64),
                                            Value::Float(v) => Some(*v),
                                            _ => None,
                                        };
                                        if let Some(v) = v {
                                            if v < agg_mins[i] {
                                                agg_mins[i] = v;
                                            }
                                            agg_has_value[i] = true;
                                        }
                                    }
                                }
                                StreamingAgg::Max(col_idx) => {
                                    if let Some(value) = row.get(*col_idx) {
                                        let v = match value {
                                            Value::Integer(v) => Some(*v as f64),
                                            Value::Float(v) => Some(*v),
                                            _ => None,
                                        };
                                        if let Some(v) = v {
                                            if v > agg_maxs[i] {
                                                agg_maxs[i] = v;
                                            }
                                            agg_has_value[i] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Fast path: All aggregates are COUNT, no row fetch needed
                    for (i, agg) in simple_aggs.iter().enumerate() {
                        if matches!(agg, StreamingAgg::Count) {
                            counts[i] = row_count;
                        }
                    }
                }

                // Apply HAVING filter
                if let Some((agg_idx, threshold, inclusive)) = having_filter {
                    let agg_val = match simple_aggs[agg_idx] {
                        StreamingAgg::Count => counts[agg_idx] as f64,
                        StreamingAgg::Sum(_) | StreamingAgg::Avg(_) => {
                            if agg_has_value[agg_idx] {
                                match simple_aggs[agg_idx] {
                                    StreamingAgg::Avg(_) if counts[agg_idx] > 0 => {
                                        agg_sums[agg_idx] / counts[agg_idx] as f64
                                    }
                                    _ => agg_sums[agg_idx],
                                }
                            } else {
                                return Ok(true); // NULL doesn't pass HAVING, continue to next group
                            }
                        }
                        StreamingAgg::Min(_) => {
                            if agg_has_value[agg_idx] {
                                agg_mins[agg_idx]
                            } else {
                                return Ok(true); // Continue to next group
                            }
                        }
                        StreamingAgg::Max(_) => {
                            if agg_has_value[agg_idx] {
                                agg_maxs[agg_idx]
                            } else {
                                return Ok(true); // Continue to next group
                            }
                        }
                    };
                    let passes = if inclusive {
                        agg_val >= threshold
                    } else {
                        agg_val > threshold
                    };
                    if !passes {
                        return Ok(true); // Continue to next group
                    }
                }

                // Build result row - only clone group_value when we need to keep it
                let mut values = Vec::with_capacity(1 + num_aggs);
                values.push(group_value.clone());
                for (i, agg) in simple_aggs.iter().enumerate() {
                    let value = match agg {
                        StreamingAgg::Count => Value::Integer(counts[i]),
                        StreamingAgg::Sum(_) => {
                            if agg_has_value[i] {
                                Value::Float(agg_sums[i])
                            } else {
                                Value::null_unknown()
                            }
                        }
                        StreamingAgg::Avg(_) => {
                            if agg_has_value[i] && counts[i] > 0 {
                                Value::Float(agg_sums[i] / counts[i] as f64)
                            } else {
                                Value::null_unknown()
                            }
                        }
                        StreamingAgg::Min(_) => {
                            if agg_has_value[i] {
                                Value::Float(agg_mins[i])
                            } else {
                                Value::null_unknown()
                            }
                        }
                        StreamingAgg::Max(_) => {
                            if agg_has_value[i] {
                                Value::Float(agg_maxs[i])
                            } else {
                                Value::null_unknown()
                            }
                        }
                    };
                    values.push(value);
                }
                result_rows.push((result_row_id, Row::from_values(values)));
                result_row_id += 1;

                // Early termination: stop once we have LIMIT groups that passed HAVING
                if let Some(limit) = limit_for_early_exit {
                    if result_rows.len() >= limit {
                        return Ok(false); // Stop iteration
                    }
                }

                Ok(true) // Continue to next group
            });

        // Check if iteration was supported and succeeded
        match iteration_result {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(e),
            None => return Ok(None), // Fall back to regular GROUP BY
        }

        // Apply LIMIT if present
        if let Some(ref limit_expr) = stmt.limit {
            if let Ok(Value::Integer(n)) = ExpressionEval::compile(limit_expr, &[])
                .and_then(|e| e.with_context(ctx).eval_slice(&Row::new()))
            {
                if n >= 0 {
                    result_rows.truncate(n as usize);
                }
            }
        }

        let result_columns = CompactArc::new(result_columns);
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&result_columns), result_rows);
        Ok(Some((Box::new(result), result_columns)))
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
}
