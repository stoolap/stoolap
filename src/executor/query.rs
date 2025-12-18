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

use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::{Error, Result, Row, Value};
use crate::optimizer::ExpressionSimplifier;
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::mvcc::engine::ViewDefinition;
use crate::storage::traits::{Engine, QueryResult};

/// Maximum depth for nested views to prevent stack overflow
const MAX_VIEW_DEPTH: usize = 32;

use super::context::{
    cache_in_subquery, clear_exists_fetcher_cache, clear_exists_index_cache,
    clear_exists_pred_key_cache, clear_exists_predicate_cache, clear_exists_schema_cache,
    clear_in_subquery_cache, clear_scalar_subquery_cache, clear_semi_join_cache,
    get_cached_in_subquery, ExecutionContext, TimeoutGuard,
};
use super::expression::{CompiledEvaluator, ExpressionEval, JoinFilter, RowFilter};
use super::join::{self, build_column_index_map};
use super::parallel::{self, ParallelConfig};
use super::pushdown;
use super::result::{
    DistinctResult, ExecResult, ExecutorMemoryResult, ExprFilteredResult, ExprMappedResult,
    FilteredResult, LimitedResult, OrderedResult, ProjectedResult, RadixOrderSpec, ScannerResult,
    StreamingProjectionResult, TopNResult,
};
use super::utils::{
    add_table_qualifier, collect_table_qualifiers, combine_predicates_with_and, dummy_token,
    expression_contains_aggregate, expression_has_parameters, extract_base_column_name,
    flatten_and_predicates, get_table_alias_from_expr, strip_table_qualifier,
};
use super::window::{WindowPreGroupedState, WindowPreSortedState};
use super::Executor;
use crate::optimizer::aqe::{decide_join_algorithm, AqeJoinDecision, JoinAqeContext};
use crate::optimizer::{BuildSide, JoinAlgorithm};

/// Pre-computed column name mappings for correlated subqueries.
/// Avoids repeated string allocations per row in the inner loop.
struct ColumnKeyMapping {
    /// Column index in the row
    index: usize,
    /// Lowercase column name (e.g., "id")
    col_lower: String,
    /// Qualified name with table alias (e.g., "c.id")
    qualified_name: Option<String>,
    /// Unqualified part if original had a dot (e.g., "id" from "table.id")
    unqualified_part: Option<String>,
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
                let col_lower = col_name.to_lowercase();
                let qualified_name = table_alias.map(|alias| format!("{}.{}", alias, col_lower));
                let unqualified_part = col_name
                    .rfind('.')
                    .map(|dot_idx| col_name[dot_idx + 1..].to_lowercase());
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
            // Clear subquery caches at top-level to avoid stale results between queries
            clear_scalar_subquery_cache();
            clear_in_subquery_cache();
            clear_semi_join_cache();
            clear_exists_predicate_cache();
            clear_exists_index_cache();
            clear_exists_fetcher_cache();
            clear_exists_schema_cache();
            clear_exists_pred_key_cache();
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

        // Evaluate LIMIT/OFFSET early (needed for set operations optimization)
        let limit = if let Some(ref limit_expr) = stmt.limit {
            match ExpressionEval::compile(limit_expr, &[])?
                .with_context(ctx)
                .eval_slice(&[])?
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
                .and_then(|eval| eval.with_context(ctx).eval_slice(&[]).ok())
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
        let (mut result, columns, limit_offset_applied) =
            self.execute_select_internal(stmt, ctx)?;

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
                let mut rows: Vec<Row> = Vec::new();
                while result.next() {
                    rows.push(result.take_row());
                }

                // Create evaluator for ORDER BY expressions
                let mut evaluator = CompiledEvaluator::new(&self.function_registry);
                evaluator = evaluator.with_context(ctx);
                evaluator.init_columns(&columns);

                // Check if any ORDER BY expression contains a correlated subquery
                let has_correlated_order_by = stmt
                    .order_by
                    .iter()
                    .any(|ob| Self::has_correlated_subqueries(&ob.expression));

                // OPTIMIZATION: Instead of cloning rows and appending sort keys,
                // compute sort keys separately and use index-based sorting.
                // This avoids O(n * row_size) cloning overhead.
                let num_order_cols = stmt.order_by.len();

                // Compute sort keys for each row: Vec<Vec<Value>>
                // Each inner Vec contains the evaluated ORDER BY expressions for that row
                let sort_keys: Vec<Vec<Value>> = if has_correlated_order_by {
                    // For correlated subqueries, we need to process per-row with outer context
                    let columns_arc = std::sync::Arc::new(columns.clone());
                    // Extract table alias for qualified column names
                    let order_table_alias: Option<String> =
                        stmt.table_expr.as_ref().and_then(|te| match te.as_ref() {
                            crate::parser::ast::Expression::TableSource(source) => source
                                .alias
                                .as_ref()
                                .map(|a| a.value.to_lowercase())
                                .or_else(|| Some(source.name.value.to_lowercase())),
                            crate::parser::ast::Expression::Aliased(aliased) => {
                                Some(aliased.alias.value.to_lowercase())
                            }
                            _ => None,
                        });

                    rows.iter()
                        .map(|row| {
                            // Build outer row context from current row
                            let mut outer_row_map: FxHashMap<String, Value> = FxHashMap::default();
                            for (idx, col_name) in columns.iter().enumerate() {
                                let val = row.get(idx).cloned().unwrap_or(Value::null_unknown());
                                let col_lower = col_name.to_lowercase();
                                outer_row_map.insert(col_lower.clone(), val.clone());
                                if let Some(ref alias) = order_table_alias {
                                    outer_row_map.insert(format!("{}.{}", alias, col_lower), val);
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
                        .map(|row| {
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

                // Build order specs: (key_index, ascending, nulls_first)
                let order_specs: Vec<(usize, bool, Option<bool>)> = (0..num_order_cols)
                    .map(|i| (i, stmt.order_by[i].ascending, stmt.order_by[i].nulls_first))
                    .collect();

                indices.sort_by(|&a_idx, &b_idx| {
                    let a_keys = &sort_keys[a_idx];
                    let b_keys = &sort_keys[b_idx];

                    for &(key_idx, ascending, nulls_first) in &order_specs {
                        let a_val = a_keys.get(key_idx);
                        let b_val = b_keys.get(key_idx);

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
                // OPTIMIZATION: Use swap-based reordering to avoid extra allocations
                // when possible, but for simplicity use index-based collection
                let sorted_rows: Vec<Row> = indices.into_iter().map(|i| rows[i].clone()).collect();

                // Apply LIMIT/OFFSET
                let final_rows = if let Some(lim) = limit {
                    sorted_rows.into_iter().skip(offset).take(lim).collect()
                } else if offset > 0 {
                    sorted_rows.into_iter().skip(offset).collect()
                } else {
                    sorted_rows
                };

                // Project to expected columns if needed
                let mut result_rows = final_rows;
                if columns.len() > expected_columns && expected_columns > 0 {
                    for row in &mut result_rows {
                        row.truncate(expected_columns);
                    }
                }

                // Use original column names if expected_columns matches
                let output_columns = if expected_columns > 0 && expected_columns <= columns.len() {
                    columns[..expected_columns].to_vec()
                } else {
                    columns.clone()
                };
                return Ok(Box::new(ExecutorMemoryResult::new(
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

                // Remove extra ORDER BY columns if needed
                if columns.len() > expected_columns && expected_columns > 0 {
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
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        // Get table source
        let table_expr = match &stmt.table_expr {
            Some(expr) => expr.as_ref(),
            None => {
                // SELECT without FROM (e.g., SELECT 1+1)
                return self.execute_expression_select(stmt, ctx);
            }
        };

        // Execute based on table source type
        match table_expr {
            Expression::TableSource(table_source) => {
                // Check if this is a CTE from context (for subqueries referencing outer CTEs)
                let table_name = &table_source.name.value_lower;
                if let Some((columns, rows)) = ctx.get_cte(table_name) {
                    // Execute query against CTE data
                    return self.execute_query_on_memory_result(
                        stmt,
                        ctx,
                        columns.clone(),
                        rows.clone(),
                    );
                }

                // Check if this is actually a view (single lookup, no double RwLock acquisition)
                if let Some(view_def) = self.engine.get_view_lowercase(table_name)? {
                    return self.execute_view_query(&view_def, stmt, ctx);
                }
                self.execute_simple_table_scan(table_source, stmt, ctx)
            }
            Expression::JoinSource(join_source) => self.execute_join_source(join_source, stmt, ctx),
            Expression::SubquerySource(subquery_source) => {
                self.execute_subquery_source(subquery_source, stmt, ctx)
            }
            Expression::ValuesSource(values_source) => {
                self.execute_values_source(values_source, stmt, ctx)
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
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        // Check WHERE clause first - if it evaluates to false, return empty result
        if let Some(where_clause) = &stmt.where_clause {
            // Pre-process subqueries in WHERE clause (EXISTS, IN, ALL/ANY, scalar subqueries)
            let processed_where = self.process_where_subqueries(where_clause, ctx)?;
            let where_result = ExpressionEval::compile(&processed_where, &[])?
                .with_context(ctx)
                .eval_slice(&[])?;
            let passes = match where_result {
                Value::Boolean(b) => b,
                Value::Null(_) => false, // NULL in WHERE is treated as false
                _ => true,
            };
            if !passes {
                // WHERE clause is false, return empty result
                // Still need to determine column names for the schema
                let mut columns = Vec::new();
                for (i, col_expr) in stmt.columns.iter().enumerate() {
                    let col_name = if let Expression::Aliased(aliased) = col_expr {
                        aliased.alias.value.clone()
                    } else {
                        format!("column{}", i + 1)
                    };
                    columns.push(col_name);
                }
                let result = ExecutorMemoryResult::new(columns.clone(), vec![]);
                return Ok((Box::new(result), columns, false));
            }
        }

        // Check if we have aggregations - if so, use aggregation path with single dummy row
        // This handles cases like SELECT SUM(3+5) or SELECT COALESCE(SUM(1), 0)
        if self.has_aggregation(stmt) {
            // Create a single dummy row for aggregation to process
            let dummy_rows = vec![Row::from_values(vec![])];
            let empty_columns: Vec<String> = vec![];
            let result =
                self.execute_select_with_aggregation(stmt, ctx, dummy_rows, &empty_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Process scalar subqueries in SELECT columns first (single-pass)
        let processed_columns = self.try_process_select_subqueries(&stmt.columns, ctx)?;
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (i, col_expr) in columns_to_use.iter().enumerate() {
            // Get column name
            let col_name = if let Expression::Aliased(aliased) = col_expr {
                aliased.alias.value.clone()
            } else {
                format!("column{}", i + 1)
            };
            columns.push(col_name);

            // Evaluate expression
            let value = ExpressionEval::compile(col_expr, &[])?
                .with_context(ctx)
                .eval_slice(&[])?;
            values.push(value);
        }

        let row = Row::from_values(values);
        let result = ExecutorMemoryResult::new(columns.clone(), vec![row]);

        Ok((Box::new(result), columns, false))
    }

    /// Execute a query against in-memory data (for CTEs referenced from context)
    fn execute_query_on_memory_result(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        columns: Vec<String>,
        rows: Vec<Row>,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        // Use the CTE execution logic from cte.rs
        let (result_cols, result_rows) =
            self.execute_query_on_cte_result(stmt, ctx, columns, rows)?;
        Ok((
            Box::new(ExecutorMemoryResult::new(result_cols.clone(), result_rows)),
            result_cols,
            false,
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
                if !select_columns.contains(&id.value_lower) {
                    // Verify this column exists in the table (use eq_ignore_ascii_case to avoid allocation)
                    if all_columns
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(&id.value_lower))
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
            Expression::Identifier(id) => Some(id.value.clone()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value.clone()),
            Expression::Aliased(aliased) => self.extract_select_column_name(&aliased.expression),
            Expression::Star(_) | Expression::QualifiedStar(_) => None, // SELECT * or t.* includes all columns
            _ => None,
        }
    }

    /// Check if a SELECT statement has DISTINCT
    #[inline]
    fn has_distinct(&self, stmt: &SelectStatement) -> bool {
        stmt.distinct
    }

    /// Execute a simple table scan
    fn execute_simple_table_scan(
        &self,
        table_source: &SimpleTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
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
                return self.execute_temporal_query(table_name, as_of, stmt, ctx, &*tx);
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

        // Get table alias for correlated subquery support
        let table_alias: Option<String> = table_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.clone())
            .or_else(|| Some(table_name.clone()));

        // Check if we need aggregation/window functions (need all columns)
        let needs_all_columns = self.has_aggregation(stmt) || self.has_window_functions(stmt);

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
        let (simplified_where, where_to_use): (Option<Expression>, Option<&Expression>) =
            if let Some(where_expr) = where_to_use {
                let mut simplifier = ExpressionSimplifier::new();
                let simplified = simplifier.simplify(where_expr);
                if simplifier.was_simplified() {
                    (Some(simplified), None) // Will be fixed up below
                } else {
                    // No simplification happened, use original to avoid extra allocation
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
        let is_select_star =
            stmt.columns.len() == 1 && matches!(stmt.columns.first(), Some(Expression::Star(_)));

        let has_aggregation_window_grouping = self.has_aggregation(stmt)
            || self.has_window_functions(stmt)
            || !stmt.group_by.columns.is_empty();
        let has_subqueries_in_where = where_to_use.is_some_and(Self::has_subqueries);

        // CRITICAL: Check if WHERE clause contains parameters ($1, $2, etc.)
        // Parameterized queries CANNOT be cached because:
        // 1. The cache stores results tied to specific parameter values
        // 2. But the AST only has parameter indices ($1), not actual values
        // 3. A cache "hit" would return wrong results for different parameter values
        // This was causing 100x slowdown for SELECT by ID queries due to:
        // - Cache misses on every lookup (unique predicates)
        // - Streaming disabled for cache-eligible queries
        // - Cache insertions (write locks) on every query execution
        let has_parameters_in_where = where_to_use.is_some_and(expression_has_parameters);

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
            && stmt.order_by.is_empty()
            && !stmt.distinct
            && stmt.limit.is_none()
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
                        // Exact cache hit - return cached rows
                        // Try to unwrap Arc if we're the only owner, otherwise clone
                        let rows = std::sync::Arc::try_unwrap(rows_arc)
                            .unwrap_or_else(|arc| (*arc).clone());
                        let output_columns = self.get_output_column_names(
                            &stmt.columns,
                            &all_columns,
                            table_alias.as_deref(),
                        );
                        let result = ExecutorMemoryResult::new(output_columns.clone(), rows);
                        return Ok((Box::new(result), output_columns, false));
                    }
                    CacheLookupResult::SubsumptionHit {
                        rows: rows_arc,
                        filter,
                        columns,
                    } => {
                        // Subsumption hit - filter cached rows
                        // Clone the Vec since we need to filter (creates new Vec anyway)
                        use super::semantic_cache::SemanticCache;
                        let filtered_rows = SemanticCache::filter_rows(
                            (*rows_arc).clone(),
                            &filter,
                            &columns,
                            &self.function_registry,
                        )?;
                        let output_columns = self.get_output_column_names(
                            &stmt.columns,
                            &all_columns,
                            table_alias.as_deref(),
                        );
                        let result =
                            ExecutorMemoryResult::new(output_columns.clone(), filtered_rows);
                        return Ok((Box::new(result), output_columns, false));
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
            if Self::has_subqueries(where_expr) {
                (None, true)
            } else if has_outer_context {
                // If we have outer row context, this might be a correlated subquery
                // Force in-memory evaluation to use outer row values
                (None, true)
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
        let has_aggregation_or_grouping = self.has_aggregation(stmt)
            || self.has_window_functions(stmt)
            || !stmt.group_by.columns.is_empty();

        if !has_aggregation_or_grouping {
            if let Some(ref expr) = storage_expr {
                if self
                    .get_query_planner()
                    .can_prune_entire_scan(&*table, expr.as_ref())
                {
                    // Zone maps indicate no segments can match - return empty result
                    let output_columns = self.get_output_column_names(
                        &stmt.columns,
                        &all_columns,
                        table_alias.as_deref(),
                    );
                    let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
                    return Ok((Box::new(result), output_columns, true));
                }
            }
        }

        // FAST PATH: MIN/MAX index optimization
        // For queries like `SELECT MIN(col) FROM table` or `SELECT MAX(col) FROM table`
        // without WHERE or GROUP BY, use the index directly (O(1) instead of O(n))
        if storage_expr.is_none() && !needs_memory_filter && stmt.group_by.columns.is_empty() {
            if let Some((result, columns)) =
                self.try_min_max_index_optimization(stmt, &*table, &all_columns)?
            {
                return Ok((result, columns, false));
            }
        }

        // FAST PATH: COUNT(*) pushdown optimization
        // For queries like `SELECT COUNT(*) FROM table` without WHERE or GROUP BY,
        // use the table's row_count() method instead of scanning all rows
        if storage_expr.is_none() && !needs_memory_filter && stmt.group_by.columns.is_empty() {
            if let Some((result, columns)) = self.try_count_star_optimization(stmt, &*table)? {
                return Ok((result, columns, false));
            }
        }

        // FAST PATH: ORDER BY + LIMIT optimization (TOP-N)
        // For queries like `SELECT * FROM table ORDER BY indexed_col LIMIT 10`,
        // use index to get rows in sorted order directly, avoiding full table sort
        if stmt.limit.is_some()
            && stmt.order_by.len() == 1
            && stmt.group_by.columns.is_empty()
            && !self.has_aggregation(stmt)
            && !self.has_window_functions(stmt)
            && !self.has_distinct(stmt)
            && storage_expr.is_none()
            && !needs_memory_filter
        {
            if let Some((result, columns)) =
                self.try_order_by_index_optimization(stmt, &*table, &all_columns, ctx)?
            {
                // Note: ORDER BY + LIMIT already handles LIMIT at storage level
                return Ok((result, columns, true));
            }
        }

        // FAST PATH: IN subquery index optimization
        // For queries like `SELECT * FROM table WHERE id IN (SELECT col FROM other_table WHERE ...)`
        // where 'id' has an index or is PRIMARY KEY, probe directly instead of scanning all rows
        if needs_memory_filter && !has_outer_context && stmt.group_by.columns.is_empty() {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns)) = self.try_in_subquery_index_optimization(
                    stmt,
                    where_expr,
                    &*table,
                    &all_columns,
                    table_alias.as_deref(),
                    ctx,
                )? {
                    return Ok((result, columns, false));
                }
            }
        }

        // FAST PATH: IN list literal index optimization
        // For queries like `SELECT * FROM table WHERE id IN (1, 2, 3, 5, 8)`
        // where 'id' has an index or is PRIMARY KEY, probe directly instead of scanning all rows
        if needs_memory_filter && !has_outer_context && stmt.group_by.columns.is_empty() {
            if let Some(where_expr) = where_to_use {
                if let Some((result, columns)) = self.try_in_list_index_optimization(
                    stmt,
                    where_expr,
                    &*table,
                    &all_columns,
                    table_alias.as_deref(),
                    ctx,
                )? {
                    return Ok((result, columns, false));
                }
            }
        }

        // FAST PATH: LIMIT pushdown optimization
        // For simple queries like `SELECT * FROM table LIMIT 10` or
        // `SELECT * FROM table WHERE indexed_col = value LIMIT 10` without ORDER BY,
        // we can stop scanning early at the storage layer
        let can_pushdown_limit = stmt.limit.is_some()
            && stmt.order_by.is_empty()
            && stmt.group_by.columns.is_empty()
            && !self.has_aggregation(stmt)
            && !self.has_window_functions(stmt)
            && !self.has_distinct(stmt)
            && !needs_memory_filter; // Allow with storage_expr (WHERE on indexed columns)

        if can_pushdown_limit {
            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])?
                    .with_context(ctx)
                    .eval_slice(&[])?
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
                    .eval_slice(&[])?
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
            let rows =
                table.collect_rows_with_limit_unordered(storage_expr.as_deref(), limit, offset)?;

            // Project rows according to SELECT expressions
            // Note: collect_rows_with_limit always returns full rows (all columns),
            // so we must always project here regardless of scanner_handled_projection
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                &all_columns,
                ctx,
                table_alias.as_deref(),
            )?;
            let output_columns =
                self.get_output_column_names(&stmt.columns, &all_columns, table_alias.as_deref());
            let (projected_rows, output_columns) = (projected_rows, output_columns);

            let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
            // LIMIT/OFFSET already applied at storage level
            return Ok((Box::new(result), output_columns, true));
        }

        // STREAMING PATH: For simple queries without aggregation/window/ORDER BY
        // Use streaming result to avoid materializing all rows into Vec
        //
        // For cache-eligible queries, we disable streaming ONLY if the table is small
        // enough to potentially benefit from caching. Large tables (>100K rows) would
        // exceed the cache limit anyway, so we allow streaming for them.
        //
        // CRITICAL OPTIMIZATION: Only call row_count() when cache_eligible is true.
        // row_count() is O(n) and was causing 100x slowdown for parameterized queries
        // that don't even use the cache.
        let should_disable_streaming_for_cache = cache_eligible && {
            let table_row_count = table.row_count();
            table_row_count <= super::semantic_cache::DEFAULT_MAX_CACHED_ROWS
        };

        let can_use_streaming = stmt.order_by.is_empty()
            && stmt.group_by.columns.is_empty()
            && !self.has_aggregation(stmt)
            && !self.has_window_functions(stmt)
            && !order_by_needs_extra_columns
            && !has_outer_context // Can't stream with outer context (correlated subqueries)
            && !should_disable_streaming_for_cache; // Only disable streaming for small cacheable queries

        // Check for subqueries in WHERE - use the actual where clause (resolved or original)
        let has_where_subqueries = where_to_use.is_some_and(Self::has_subqueries);
        if can_use_streaming && !has_where_subqueries {
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
                        // Create a pre-compiled filter (RowFilter is Clone+Send+Sync)
                        let filter = RowFilter::new(where_expr, &all_columns)?.with_context(ctx);

                        // Create a filter predicate using the RowFilter
                        let predicate: Box<dyn Fn(&Row) -> bool + Send + Sync> =
                            Box::new(move |row: &Row| filter.matches(row));

                        result = Box::new(FilteredResult::new(result, predicate));
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
                    result = Box::new(StreamingProjectionResult::new(
                        result,
                        column_indices,
                        output_columns.clone(),
                    ));
                    // LIMIT/OFFSET NOT applied yet - streaming path
                    return Ok((result, output_columns, false));
                } else {
                    // SELECT * - no projection needed
                    // LIMIT/OFFSET NOT applied yet - streaming path
                    return Ok((result, output_columns, false));
                }
            }
        }

        // Collect rows - choose optimal path based on query type
        // PARALLEL EXECUTION: Use parallel filtering for large datasets
        let parallel_config = ParallelConfig::default();

        let rows_result = if needs_memory_filter {
            // Path 1: Need in-memory filtering (subqueries or complex expressions)
            // For memory filter, we need all columns to evaluate the WHERE clause
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

            // Check if WHERE contains correlated subqueries
            let has_correlated = if let Some(where_expr) = where_to_use {
                Self::has_subqueries(where_expr) && Self::has_correlated_subqueries(where_expr)
            } else {
                false
            };

            // Check if WHERE contains any subqueries (correlated or not)
            // Cache this result to avoid redundant traversal of the expression tree
            let has_subqueries = where_to_use.is_some_and(Self::has_subqueries);

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

                    // Try semi-join optimization
                    if let Ok(Some(optimized)) = self.try_optimize_exists_to_semi_join(
                        where_expr,
                        ctx,
                        &outer_tables,
                        outer_limit,
                    ) {
                        // Successfully transformed EXISTS to IN - no longer correlated!
                        // The optimized expression uses a hash set lookup instead of per-row subquery
                        (Some(optimized), false)
                    } else {
                        // Couldn't optimize - keep original for per-row processing
                        (Some(where_expr.clone()), true)
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
            if !has_correlated && stmt.group_by.columns.is_empty() {
                let has_correlated_select =
                    stmt.columns.iter().any(Self::has_correlated_subqueries);
                if !has_correlated_select {
                    if let Some(ref where_expr) = processed_where {
                        if let Some((result, columns)) = self.try_in_hashset_index_optimization(
                            stmt,
                            where_expr,
                            &*table,
                            &all_columns,
                            table_alias.as_deref(),
                            ctx,
                        )? {
                            return Ok((result, columns, false));
                        }
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

                // Collect all rows first (sequential - storage layer limitation)
                let mut all_rows = Vec::new();
                while scanner.next() {
                    all_rows.push(scanner.take_row());
                }

                // Apply parallel filtering if we have enough rows
                // CRITICAL: Propagate errors with ? instead of silently swallowing them
                if parallel_config.should_parallel_filter(all_rows.len()) {
                    (
                        parallel::parallel_filter(
                            all_rows,
                            where_expr,
                            &all_columns,
                            &self.function_registry,
                            &parallel_config,
                        )?,
                        None,
                        None,
                    )
                } else {
                    // Sequential filter for small datasets using pre-compiled expression
                    let mut eval =
                        ExpressionEval::compile(where_expr, &all_columns)?.with_context(ctx);

                    (
                        all_rows
                            .into_iter()
                            .filter(|row| eval.eval_bool(row))
                            .collect(),
                        None,
                        None,
                    )
                }
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
                let mut outer_row_map: FxHashMap<String, Value> = FxHashMap::default();
                outer_row_map.reserve(base_capacity);

                // OPTIMIZATION: Wrap all_columns in Arc once, reuse for all rows (only if needed)
                let all_columns_arc: Option<Arc<Vec<String>>> = if has_correlated {
                    Some(Arc::new(all_columns.clone()))
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
                                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
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
                                .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
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

                let mut rows = Vec::new();
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

                            // Copy parent outer context if exists (for nested correlated subqueries)
                            if let Some(parent_outer_row) = ctx.outer_row() {
                                outer_row_map.extend(
                                    parent_outer_row.iter().map(|(k, v)| (k.clone(), v.clone())),
                                );
                            }

                            // Use pre-computed column mappings
                            if let Some(ref keys) = column_keys {
                                for mapping in keys {
                                    if let Some(value) = row.get(mapping.index) {
                                        // Insert with lowercase column name
                                        outer_row_map
                                            .insert(mapping.col_lower.clone(), value.clone());

                                        // Insert with qualified name if available
                                        if let Some(ref qname) = mapping.qualified_name {
                                            outer_row_map.insert(qname.clone(), value.clone());
                                        }

                                        // Insert with unqualified part if column had a dot
                                        if let Some(ref upart) = mapping.unqualified_part {
                                            outer_row_map.insert(upart.clone(), value.clone());
                                        }
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

                    rows.push(row);

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
            // Column pruning is handled by Path 3 (collect_projected_rows) when no WHERE clause
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

            // OPTIMIZATION: Use take_row() to avoid cloning each row
            let mut rows = Vec::new();
            let mut row_count = 0u64;
            while scanner.next() {
                // Check for cancellation every 100 rows
                row_count += 1;
                if row_count.is_multiple_of(100) {
                    ctx.check_cancelled()?;
                }
                rows.push(scanner.take_row());
            }
            (rows, None, None)
        } else {
            // Path 3: Full scan without WHERE - use collect_all_rows
            // Note: collect_projected_rows was SLOWER due to extra allocations per row
            // Projection is handled later by the executor which is more efficient
            //
            // OPTIMIZATION: For window functions, check if we can use index-based fetching:
            // 1. PARTITION BY on indexed column -> fetch rows grouped by partition
            // 2. ORDER BY on indexed column -> fetch rows in sorted order
            let has_window = self.has_window_functions(stmt);
            let has_agg = self.has_aggregation(stmt);

            if has_window && !has_agg {
                // First try PARTITION BY optimization (bigger speedup, avoids O(n) hashing)
                if let Some(partition_col) = Self::extract_window_partition_info(stmt) {
                    let col_lower = partition_col.to_lowercase();
                    let schema = table.schema();
                    let pk_columns = schema.primary_key_columns();
                    let is_pk =
                        pk_columns.len() == 1 && pk_columns[0].name.to_lowercase() == col_lower;
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
                                            let columns = query_result.columns().to_vec();
                                            return Ok((query_result, columns, false));
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
                            let mut all_rows: Vec<Row> = Vec::new();
                            let mut partition_map: rustc_hash::FxHashMap<
                                smallvec::SmallVec<[Value; 4]>,
                                Vec<usize>,
                            > = rustc_hash::FxHashMap::default();

                            for (partition_value, partition_rows) in grouped_data {
                                let start_idx = all_rows.len();
                                let partition_size = partition_rows.len();
                                all_rows.extend(partition_rows);

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
                                    column: col_lower,
                                    partition_map,
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
                    let is_pk =
                        pk_columns.len() == 1 && pk_columns[0].name.to_lowercase() == col_lower;
                    let has_index = is_pk || table.get_index_on_column(&col_name).is_some();

                    if has_index {
                        // Fetch rows in sorted order from the index (no re-fetch needed)
                        if let Some(sorted_rows) =
                            table.collect_rows_ordered_by_index(&col_name, ascending, usize::MAX, 0)
                        {
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
            Vec<Row>,
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
                        None, // TODO: Extract column name from predicate if single-column
                        estimated_rows,
                        actual_rows,
                    );
                }
            }
        }

        // Handle the combination of window functions and aggregation
        // Order: Aggregation first (GROUP BY), then window functions
        let has_window = self.has_window_functions(stmt);
        let has_agg = self.has_aggregation(stmt);

        if has_agg && has_window {
            // Both aggregation and window functions:
            // 1. First apply GROUP BY aggregation
            // 2. Then apply window functions on the aggregated result
            let agg_result = self.execute_aggregation_for_window(stmt, ctx, rows, &all_columns)?;
            let agg_columns = agg_result.0.clone();
            let agg_rows = agg_result.1;

            // Apply window functions on aggregated rows
            let result =
                self.execute_select_with_window_functions(stmt, ctx, agg_rows, &agg_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check if we need window functions only (no aggregation)
        if has_window {
            // Use optimized paths if rows were pre-fetched with index optimization
            let result = if let Some(pregrouped) = window_pregrouped_state {
                // PARTITION BY optimization: rows are already grouped by partition
                self.execute_select_with_window_functions_pregrouped(
                    stmt,
                    ctx,
                    rows,
                    &all_columns,
                    pregrouped,
                )?
            } else if window_presorted_state.is_some() {
                // ORDER BY optimization: rows are already sorted
                self.execute_select_with_window_functions_presorted(
                    stmt,
                    ctx,
                    rows,
                    &all_columns,
                    window_presorted_state,
                )?
            } else {
                // Default path: no optimization
                self.execute_select_with_window_functions(stmt, ctx, rows, &all_columns)?
            };
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check if we need aggregation only (no window functions)
        if has_agg {
            let result = self.execute_select_with_aggregation(stmt, ctx, rows, &all_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Project rows according to SELECT expressions
        let (projected_rows, output_columns) = if order_by_needs_extra_columns {
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
                        output_columns.push(id.value.clone());
                    }
                }
            }
            (projected_rows, output_columns)
        } else {
            // Standard projection
            let projected_rows = self.project_rows_with_alias(
                &stmt.columns,
                rows,
                &all_columns,
                ctx,
                table_alias.as_deref(),
            )?;
            let output_columns =
                self.get_output_column_names(&stmt.columns, &all_columns, table_alias.as_deref());
            (projected_rows, output_columns)
        };

        // SEMANTIC CACHE: Insert result for eligible queries
        // For SELECT * queries, cache the raw rows before returning
        //
        // Optimization: We wrap in Arc and share between cache and result to avoid
        // full Vec clone. The cache stores Arc<Vec<Row>>, and we try_unwrap for the
        // result (avoiding clone if we're the only owner).
        let projected_rows = if cache_eligible {
            if let Some(where_expr) = where_to_use {
                let rows_arc = std::sync::Arc::new(projected_rows);
                self.semantic_cache.insert_arc(
                    table_name,
                    all_columns.clone(),
                    std::sync::Arc::clone(&rows_arc),
                    Some(where_expr.clone()),
                );
                // Try to reclaim ownership; if cache still holds a reference, clone
                std::sync::Arc::try_unwrap(rows_arc).unwrap_or_else(|arc| (*arc).clone())
            } else {
                projected_rows
            }
        } else {
            projected_rows
        };

        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok((Box::new(result), output_columns, false))
    }

    /// Execute a JOIN source
    fn execute_join_source(
        &self,
        join_source: &JoinTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
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
        let (left_filter, right_filter, cross_filter) =
            if let Some(ref where_clause) = stmt.where_clause {
                if let (Some(left_a), Some(right_a)) = (&left_alias, &right_alias) {
                    let (l, r, c) = partition_where_for_join(where_clause, left_a, right_a);

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
                } else {
                    (None, None, Some((**where_clause).clone()))
                }
            } else {
                (None, None, None)
            };

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

        let (left_rows, left_columns, right_rows, right_columns) =
            if let Some((limit_n, left_key_col, right_key_col)) = semijoin_limit {
                // Semi-join reduction for LEFT JOIN + GROUP BY
                // Step 1: Execute and materialize left side with limit
                let (left_result, left_cols) = self.execute_table_expression_with_filter(
                    &join_source.left,
                    ctx,
                    left_filter.as_ref(),
                )?;
                let mut left_rows = Self::materialize_result(left_result)?;
                left_rows.truncate(limit_n);

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
                let right_rows = Self::materialize_result(right_result)?;

                (left_rows, left_cols, right_rows, right_cols)
            } else {
                // Skip Index Nested Loop if query has aggregation or window functions
                // These require full result sets and can't use early termination
                let has_agg = self.has_aggregation(stmt);
                let has_window = self.has_window_functions(stmt);

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

                // Join reordering optimization for INNER JOINs:
                // When one side has a filter, prefer putting filtered side as outer (left)
                // This reduces the number of probes into the inner table.
                // Swap if: right has filter, left doesn't, and swapped order gives Index NL on PK
                let (index_nl_info, nl_left_filter, nl_right_filter, swapped) = if !has_agg
                        && !has_window
                        && join_type == "INNER"
                        && right_filter.is_some()  // Right side has a filter
                        && left_filter.is_none()
                // Left side doesn't have a filter
                {
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
                        Some((_, join::IndexLookupStrategy::PrimaryKey, _, _))
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

                if let Some((table_name, lookup_strategy, _inner_col, outer_col)) = index_nl_info {
                    // Index Nested Loop path: only materialize outer (left) side
                    // When swapped, execute right side as outer (with original right filter, now in nl_left_filter)
                    let outer_expr = if swapped {
                        &join_source.right
                    } else {
                        &join_source.left
                    };
                    let (outer_result, outer_cols) = self.execute_table_expression_with_filter(
                        outer_expr,
                        ctx,
                        nl_left_filter.as_ref(),
                    )?;
                    let outer_rows = Self::materialize_result(outer_result)?;

                    // Find the outer key index in outer columns
                    let outer_col_lower = outer_col.to_lowercase();
                    let outer_key_idx = outer_cols
                        .iter()
                        .position(|c| c.to_lowercase() == outer_col_lower)
                        .or_else(|| {
                            // Try unqualified match
                            let outer_unqualified = outer_col_lower
                                .rfind('.')
                                .map(|p| &outer_col_lower[p + 1..])
                                .unwrap_or(&outer_col_lower);
                            outer_cols.iter().position(|c| {
                                let c_lower = c.to_lowercase();
                                let c_unqualified = c_lower
                                    .rfind('.')
                                    .map(|p| &c_lower[p + 1..])
                                    .unwrap_or(&c_lower);
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

                        // Compute join limit if applicable
                        let has_group_by = !stmt.group_by.columns.is_empty();
                        let has_aggregation = self.has_aggregation(stmt);
                        let can_push_limit = !join_type.contains("FULL")
                            && stmt.order_by.is_empty()
                            && !has_group_by
                            && !has_aggregation;

                        let join_limit = if can_push_limit {
                            stmt.limit.as_ref().and_then(|limit_expr| {
                                ExpressionEval::compile(limit_expr, &[])
                                    .ok()
                                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                                    .and_then(|v| match v {
                                        Value::Integer(n) if n >= 0 => Some(n as u64),
                                        _ => None,
                                    })
                            })
                        } else {
                            None
                        };

                        // Build all columns for combined row
                        // When swapped, result order is outer(orig_right), inner(orig_left)
                        // But we need final order to be: orig_left, orig_right
                        let (all_columns, _orig_left_cols, _orig_right_cols) = if swapped {
                            // Swapped: outer=orig_right, inner=orig_left
                            let mut all = inner_cols.clone(); // orig_left first
                            all.extend(outer_cols.clone()); // orig_right second
                            (all, inner_cols.clone(), outer_cols.clone())
                        } else {
                            // Not swapped: outer=orig_left, inner=orig_right
                            let mut all = outer_cols.clone(); // orig_left first
                            all.extend(inner_cols.clone()); // orig_right second
                            (all, outer_cols.clone(), inner_cols.clone())
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
                        } else {
                            None
                        };

                        // Execute Index Nested Loop Join with residual filter
                        let result_rows = self.execute_index_nested_loop_join(
                            &outer_rows,
                            inner_table.as_ref(),
                            &lookup_strategy,
                            outer_idx,
                            residual_filter.as_ref(),
                            &join_type,
                            outer_cols.len(),
                            inner_cols.len(),
                            join_limit,
                        )?;

                        // When swapped, reorder columns from (outer, inner) to (orig_left, orig_right)
                        let result_rows = if swapped {
                            // Result is (outer=orig_right, inner=orig_left), need (orig_left, orig_right)
                            let _orig_left_count = inner_cols.len();
                            let orig_right_count = outer_cols.len();
                            result_rows
                                .into_iter()
                                .map(|row| {
                                    let values = row.as_slice();
                                    // Reorder: [outer(orig_right), inner(orig_left)] -> [orig_left, orig_right]
                                    let mut new_values = Vec::with_capacity(values.len());
                                    // First add inner values (which are orig_left)
                                    new_values.extend(values[orig_right_count..].iter().cloned());
                                    // Then add outer values (which are orig_right)
                                    new_values.extend(values[..orig_right_count].iter().cloned());
                                    Row::from_values(new_values)
                                })
                                .collect()
                        } else {
                            result_rows
                        };

                        // No need to apply right_filter again - it was applied during the join
                        let mut final_rows = result_rows;

                        // Apply cross-table WHERE filters if any
                        if let Some(ref cross) = cross_filter {
                            let filter = RowFilter::new(cross, &all_columns)?;
                            final_rows.retain(|row| filter.matches(row));
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
                                .map(|row| {
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

                            // Sort by indices
                            let mut indices: Vec<usize> = (0..final_rows.len()).collect();
                            indices.sort_by(|&a, &b| {
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
                                        join::compare_values(av, bv)
                                    };

                                    let cmp = if asc { cmp } else { cmp.reverse() };
                                    if cmp != Ordering::Equal {
                                        return cmp;
                                    }
                                }
                                Ordering::Equal
                            });

                            // Reorder rows
                            final_rows =
                                indices.into_iter().map(|i| final_rows[i].clone()).collect();

                            // Apply LIMIT/OFFSET after sorting
                            let offset = stmt
                                .offset
                                .as_ref()
                                .and_then(|e| {
                                    ExpressionEval::compile(e, &[])
                                        .ok()
                                        .and_then(|eval| {
                                            eval.with_context(ctx).eval_slice(&[]).ok()
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
                                            eval.with_context(ctx).eval_slice(&[]).ok()
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
                        let has_agg = self.has_aggregation(stmt);
                        let has_window = self.has_window_functions(stmt);

                        if has_agg || has_window {
                            // Fall through to standard path for aggregation/window handling
                            // Don't return early - let the standard path handle these
                        } else {
                            // Project rows according to SELECT expressions
                            let projected_rows =
                                self.project_rows(&stmt.columns, final_rows, &all_columns, ctx)?;
                            let output_columns =
                                self.get_output_column_names(&stmt.columns, &all_columns, None);

                            // Return with projected results
                            let result =
                                ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
                            return Ok((Box::new(result), output_columns, false));
                        }
                    }
                }

                // Standard path: execute both sides normally
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

                let left_rows = Self::materialize_result(left_result)?;
                let right_rows = Self::materialize_result(right_result)?;

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
                    .map(|c| c.value.to_lowercase())
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

        // Cost-based join algorithm selection using the query planner
        // The planner considers actual row counts for optimal algorithm choice
        let result_rows = if let Some(condition) = effective_condition {
            // Extract equality keys and non-equality residual conditions
            let (left_key_indices, right_key_indices, residual) =
                join::extract_join_keys_and_residual(condition, &left_columns, &right_columns);

            // Use the query planner for cost-based join decision
            let has_equality_keys = !left_key_indices.is_empty();

            // Detect if inputs are sorted on join keys for potential merge join optimization
            // This check is O(n) but enables O(n+m) merge join vs O(n*m) nested loop
            let left_sorted =
                has_equality_keys && join::is_sorted_on_keys(&left_rows, &left_key_indices);
            let right_sorted =
                has_equality_keys && join::is_sorted_on_keys(&right_rows, &right_key_indices);

            // Get estimated cardinalities from statistics (if available)
            // These are used by AQE to detect significant estimation errors
            let (estimated_left, estimated_right) = self.get_join_estimates(
                &join_source.left,
                &join_source.right,
                left_rows.len() as u64,
                right_rows.len() as u64,
            );

            let join_decision = self.get_query_planner().plan_runtime_join_with_sort_info(
                left_rows.len(),
                right_rows.len(),
                has_equality_keys,
                left_sorted,
                right_sorted,
            );

            // Convert planner's runtime algorithm to optimizer's JoinAlgorithm for AQE
            use crate::executor::planner::RuntimeJoinAlgorithm;
            let planned_algorithm = match join_decision.algorithm {
                RuntimeJoinAlgorithm::HashJoin => JoinAlgorithm::HashJoin {
                    build_side: if join_decision.swap_sides {
                        BuildSide::Right
                    } else {
                        BuildSide::Left
                    },
                    build_rows: if join_decision.swap_sides {
                        right_rows.len() as u64
                    } else {
                        left_rows.len() as u64
                    },
                    probe_rows: if join_decision.swap_sides {
                        left_rows.len() as u64
                    } else {
                        right_rows.len() as u64
                    },
                },
                RuntimeJoinAlgorithm::MergeJoin => JoinAlgorithm::MergeJoin {
                    left_rows: left_rows.len() as u64,
                    right_rows: right_rows.len() as u64,
                    left_sorted,
                    right_sorted,
                },
                RuntimeJoinAlgorithm::NestedLoop => JoinAlgorithm::NestedLoop {
                    outer_rows: left_rows.len() as u64,
                    inner_rows: right_rows.len() as u64,
                },
            };

            // Create AQE context with estimated vs actual cardinalities
            let aqe_context = JoinAqeContext::new(
                estimated_left,
                estimated_right,
                left_rows.len() as u64,
                right_rows.len() as u64,
                has_equality_keys,
                planned_algorithm.clone(),
            );

            // Let AQE decide if we should switch algorithms based on estimation error
            let aqe_decision = decide_join_algorithm(&aqe_context);

            // Determine final algorithm based on AQE decision
            let final_algorithm = match aqe_decision {
                AqeJoinDecision::KeepPlanned => join_decision.algorithm,
                AqeJoinDecision::SwitchToHashJoin { build_side: _ } => {
                    // AQE recommends hash join (build side handled by execute_hash_join)
                    RuntimeJoinAlgorithm::HashJoin
                }
                AqeJoinDecision::SwitchToNestedLoop => RuntimeJoinAlgorithm::NestedLoop,
            };

            // Compute safe LIMIT for pushdown to hash join
            // LIMIT can only be pushed when:
            // - It's an INNER JOIN, LEFT JOIN, or RIGHT JOIN (not FULL OUTER)
            // - No residual conditions to apply after join
            // - No ORDER BY (would need all rows to sort first)
            // - No remaining cross-table WHERE conditions
            // - No GROUP BY or aggregation (needs all rows to form correct groups)
            // Note: LEFT/RIGHT JOINs can early-terminate because each probe row
            // contributes to output (either matches or NULL-padded)
            let has_group_by = !stmt.group_by.columns.is_empty();
            let has_aggregation = self.has_aggregation(stmt);
            let can_push_limit = !join_type.contains("FULL")
                && residual.is_empty()
                && stmt.order_by.is_empty()
                && cross_filter.is_none()
                && !has_group_by
                && !has_aggregation;

            let join_limit = if can_push_limit {
                // Extract LIMIT value if present
                stmt.limit.as_ref().and_then(|limit_expr| {
                    ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                        .and_then(|v| match v {
                            Value::Integer(n) if n >= 0 => Some(n as u64),
                            _ => None,
                        })
                })
            } else {
                None
            };

            // Select join algorithm based on final decision (planner or AQE-overridden)
            match final_algorithm {
                RuntimeJoinAlgorithm::HashJoin => {
                    // Hash Join: O(N + M) instead of O(N * M)
                    // Supports multiple keys: ON a.x = b.x AND a.y = b.y
                    // Use execute_hash_join which handles:
                    // - Build side optimization (smaller table as build)
                    // - Outer join restrictions (LEFT/RIGHT/FULL joins need specific build sides)
                    // - Bloom filter optimization
                    // - Early termination when limit is pushed
                    let mut rows = self.execute_hash_join(
                        &left_rows,
                        &right_rows,
                        &left_key_indices,
                        &right_key_indices,
                        &join_type,
                        left_columns.len(),
                        right_columns.len(),
                        join_limit,
                    )?;

                    // Apply residual conditions (non-equality parts of ON clause)
                    // For OUTER JOINs, we need special handling: rows with NULLs from the
                    // padded side should be preserved even if residual conditions fail
                    if !residual.is_empty() {
                        self.apply_residual_conditions(
                            &mut rows,
                            &residual,
                            &all_columns,
                            &join_type,
                            left_columns.len(),
                            right_columns.len(),
                            ctx,
                        )?;
                    }
                    rows
                }
                RuntimeJoinAlgorithm::MergeJoin => {
                    // Merge Join: O(N + M) - optimal when inputs are sorted on join keys
                    // Currently triggered when both inputs are pre-sorted
                    let mut rows = self.execute_merge_join(
                        &left_rows,
                        &right_rows,
                        &left_key_indices,
                        &right_key_indices,
                        &join_type,
                        left_columns.len(),
                        right_columns.len(),
                    )?;

                    // Apply residual conditions
                    if !residual.is_empty() {
                        self.apply_residual_conditions(
                            &mut rows,
                            &residual,
                            &all_columns,
                            &join_type,
                            left_columns.len(),
                            right_columns.len(),
                            ctx,
                        )?;
                    }
                    rows
                }
                RuntimeJoinAlgorithm::NestedLoop => {
                    // Nested Loop: O(N * M) - used for small tables or non-equality joins
                    self.execute_nested_loop_join(
                        &left_rows,
                        &right_rows,
                        Some(condition),
                        &all_columns,
                        &left_columns,
                        &right_columns,
                        &join_type,
                        ctx,
                        join_limit,
                    )?
                }
            }
        } else {
            // CROSS JOIN - no condition
            // For CROSS JOIN, compute limit directly since we don't have join keys
            // Don't push limit when there's aggregation (needs all rows for grouping)
            let cross_has_group_by = !stmt.group_by.columns.is_empty();
            let cross_has_aggregation = self.has_aggregation(stmt);
            let cross_join_limit = if !join_type.contains("LEFT")
                && !join_type.contains("RIGHT")
                && !join_type.contains("FULL")
                && stmt.order_by.is_empty()
                && stmt.where_clause.is_none()
                && !cross_has_group_by
                && !cross_has_aggregation
            {
                stmt.limit.as_ref().and_then(|limit_expr| {
                    ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                        .and_then(|v| match v {
                            Value::Integer(n) if n >= 0 => Some(n as u64),
                            _ => None,
                        })
                })
            } else {
                None
            };
            self.execute_nested_loop_join(
                &left_rows,
                &right_rows,
                None,
                &all_columns,
                &left_columns,
                &right_columns,
                &join_type,
                ctx,
                cross_join_limit,
            )?
        };

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
            let processed_where = if Self::has_subqueries(where_clause) {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Create RowFilter once and reuse
            let where_filter = RowFilter::new(&processed_where, &all_columns)?.with_context(ctx);

            result_rows
                .into_iter()
                .filter(|row| where_filter.matches(row))
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

                // Filter row values using clone_subset
                let filtered_rows: Vec<Row> = filtered_rows
                    .into_iter()
                    .map(|row| row.clone_subset(&kept_indices))
                    .collect();

                (filtered_columns, filtered_rows)
            } else {
                (all_columns.clone(), filtered_rows)
            }
        } else {
            (all_columns.clone(), filtered_rows)
        };

        let has_agg = self.has_aggregation(stmt);
        let has_window = self.has_window_functions(stmt);

        // Check if we have both aggregation and window functions
        if has_agg && has_window {
            // 1. First apply GROUP BY aggregation
            // 2. Then apply window functions on the aggregated result
            let agg_result =
                self.execute_aggregation_for_window(stmt, ctx, final_rows, &final_columns)?;
            let agg_columns = agg_result.0.clone();
            let agg_rows = agg_result.1;

            // Apply window functions on aggregated rows
            let result =
                self.execute_select_with_window_functions(stmt, ctx, agg_rows, &agg_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check if we need window functions only (no aggregation)
        if has_window {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, final_rows, &final_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check if we need aggregation only
        if has_agg {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, final_rows, &final_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Project rows according to SELECT expressions
        let projected_rows = self.project_rows(&stmt.columns, final_rows, &final_columns, ctx)?;

        // Determine output column names
        // Note: For JOIN results, columns are already qualified (e.g., "a.id", "b.id"),
        // so we pass None for table_alias - the prefix matching will work
        let output_columns = self.get_output_column_names(&stmt.columns, &final_columns, None);

        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok((Box::new(result), output_columns, false))
    }

    /// Execute a subquery source
    fn execute_subquery_source(
        &self,
        subquery_source: &SubqueryTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        // Execute subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let result = self.execute_select(&subquery_source.subquery, &subquery_ctx)?;
        let columns = result.columns().to_vec();

        // Materialize the subquery result
        let rows = Self::materialize_result(result)?;

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            let where_filter = RowFilter::new(where_clause, &columns)?.with_context(ctx);

            rows.into_iter()
                .filter(|row| where_filter.matches(row))
                .collect()
        } else {
            rows
        };

        // Check if we need window functions
        if self.has_window_functions(stmt) {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, filtered_rows, &columns)?;
            let out_columns = result.columns().to_vec();
            return Ok((result, out_columns, false));
        }

        // Check if we need aggregation
        if self.has_aggregation(stmt) {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &columns)?;
            let out_columns = result.columns().to_vec();
            return Ok((result, out_columns, false));
        }

        // Project rows according to SELECT expressions
        let projected_rows = self.project_rows(&stmt.columns, filtered_rows, &columns, ctx)?;

        // Determine output column names
        let subquery_alias = subquery_source
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str());
        let output_columns = self.get_output_column_names(&stmt.columns, &columns, subquery_alias);

        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok((Box::new(result), output_columns, false))
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
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
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
        // OPTIMIZATION: ExprFilteredResult owns the Evaluator and reuses it for each row,
        // avoiding 7 HashMap allocations per row that the closure-based approach had.
        let mut result: Box<dyn QueryResult> = result;
        if let Some(ref where_clause) = stmt.where_clause {
            let filter_expr = where_clause.as_ref().clone();
            result = Box::new(ExprFilteredResult::with_defaults(result, filter_expr));
        }

        // Handle aggregation: if outer query has aggregates, materialize view result and aggregate
        if self.has_aggregation(stmt) {
            // Materialize the view result into rows
            let mut rows = Vec::new();
            while result.next() {
                rows.push(result.take_row());
            }

            // Execute aggregation on the view's rows
            let agg_result =
                self.execute_select_with_aggregation(stmt, ctx, rows, &view_columns)?;
            let columns = agg_result.columns().to_vec();
            return Ok((agg_result, columns, false));
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
            return Ok((result, view_columns, false));
        }

        // Determine if we have any complex expressions (not just column references)
        // If all expressions are simple column references, use fast StreamingProjectionResult
        // Otherwise, use MappedResult with expression evaluation
        let mut has_complex_expressions = false;
        let mut column_indices = Vec::with_capacity(stmt.columns.len());
        let mut output_columns = Vec::with_capacity(stmt.columns.len());

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
                    let prefix = format!("{}.", qs.qualifier);
                    let prefix_lower = prefix.to_lowercase();
                    for (idx, name) in view_columns.iter().enumerate() {
                        if name.to_lowercase().starts_with(&prefix_lower) {
                            column_indices.push(idx);
                            // Strip the prefix from the column name for the output
                            output_columns.push(name[prefix.len()..].to_string());
                        }
                    }
                }
                Expression::Identifier(id) => {
                    // Simple column reference
                    output_columns.push(id.value.clone());
                    let name_lower = &id.value_lower;
                    if let Some(idx) = view_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(name_lower))
                    {
                        column_indices.push(idx);
                    } else {
                        return Err(Error::ColumnNotFoundNamed(id.value.clone()));
                    }
                }
                Expression::Aliased(aliased) => {
                    output_columns.push(aliased.alias.value.clone());
                    // Check if inner is a simple identifier
                    if let Expression::Identifier(id) = aliased.expression.as_ref() {
                        let name_lower = &id.value_lower;
                        if let Some(idx) = view_columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(name_lower))
                        {
                            column_indices.push(idx);
                        } else {
                            return Err(Error::ColumnNotFoundNamed(id.value.clone()));
                        }
                    } else {
                        // Complex expression in alias - need MappedResult
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
                        let prefix = format!("{}.", qs.qualifier);
                        let prefix_lower = prefix.to_lowercase();
                        for name in &view_columns {
                            if name.to_lowercase().starts_with(&prefix_lower) {
                                // Strip the prefix from the column name for the output
                                final_output_columns.push(name[prefix.len()..].to_string());
                            }
                        }
                    }
                    Expression::Aliased(aliased) => {
                        final_output_columns.push(aliased.alias.value.clone());
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
                        cols.push(aliased.alias.value.clone());
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

        Ok((result, final_columns, false))
    }

    /// Execute a VALUES source (e.g., (VALUES (1, 'a'), (2, 'b')) AS t(col1, col2))
    fn execute_values_source(
        &self,
        values_source: &ValuesTableSource,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
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
            .map(|a| a.value.clone())
            .unwrap_or_default();

        let column_names: Vec<String> = if !values_source.column_aliases.is_empty() {
            // Use provided column aliases
            values_source
                .column_aliases
                .iter()
                .map(|id| id.value.clone())
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
        let mut result_rows = Vec::with_capacity(values_source.rows.len());
        for row_exprs in &values_source.rows {
            let mut row_values = Vec::with_capacity(row_exprs.len());
            for expr in row_exprs {
                let value = ExpressionEval::compile(expr, &[])?
                    .with_context(ctx)
                    .eval_slice(&[])?;
                row_values.push(value);
            }
            let row = Row::from_values(row_values);

            // Apply WHERE clause filtering
            if let (Some(wf), Some(qf)) = (&where_filter, &qualified_filter) {
                // Try with simple column names first, then qualified
                if wf.matches(&row) || qf.matches(&row) {
                    result_rows.push(row);
                }
            } else {
                result_rows.push(row);
            }
        }

        // Check if we need window functions
        if self.has_window_functions(stmt) {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, result_rows, &column_names)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check if we need aggregation
        if self.has_aggregation(stmt) {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, result_rows, &column_names)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Create the result
        let values_result = ExecutorMemoryResult::new(column_names.clone(), result_rows);

        // If the SELECT has projections, apply them
        if stmt.columns.len() == 1
            && matches!(
                &stmt.columns[0],
                Expression::Star(_) | Expression::QualifiedStar(_)
            )
        {
            // SELECT * or t.* - return all columns
            return Ok((Box::new(values_result), column_names, false));
        }

        // Apply column projection
        let mut projected_columns = Vec::new();
        let mut projected_rows = Vec::new();

        // Determine output columns
        for (i, col_expr) in stmt.columns.iter().enumerate() {
            match col_expr {
                Expression::Star(_) => {
                    projected_columns.extend(column_names.clone());
                }
                Expression::QualifiedStar(_) => {
                    // For single table, t.* is equivalent to *
                    projected_columns.extend(column_names.clone());
                }
                Expression::Aliased(a) => {
                    projected_columns.push(a.alias.value.clone());
                }
                Expression::Identifier(id) => {
                    projected_columns.push(id.value.clone());
                }
                Expression::QualifiedIdentifier(qi) => {
                    projected_columns.push(qi.name.value.clone());
                }
                _ => {
                    projected_columns.push(format!("expr{}", i + 1));
                }
            }
        }

        // Build extended columns for evaluator - just use simple column names
        // The evaluator handles qualified references by stripping the qualifier
        let extended_columns = column_names.clone();

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
                        .or_else(|| col_index_map.get(&qi.name.value_lower).copied()),
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
                        // OPTIMIZATION: Use extend_from_slice instead of iter().cloned()
                        new_values.extend_from_slice(row.as_slice());
                    }
                    Expression::Identifier(id) => {
                        // Use pre-computed lowercase
                        if let Some(&idx) = col_index_map.get(&id.value_lower) {
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
                                let mut outer_row_map: FxHashMap<String, Value> =
                                    FxHashMap::default();
                                for (i, col_name) in extended_columns.iter().enumerate() {
                                    if let Some(value) = row.get(i) {
                                        outer_row_map
                                            .insert(col_name.to_lowercase(), value.clone());
                                    }
                                }
                                let correlated_ctx = ctx.with_outer_row(
                                    outer_row_map,
                                    std::sync::Arc::new(extended_columns.clone()),
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
                                let mut outer_row_map: FxHashMap<String, Value> =
                                    FxHashMap::default();
                                for (i, col_name) in extended_columns.iter().enumerate() {
                                    if let Some(value) = row.get(i) {
                                        outer_row_map
                                            .insert(col_name.to_lowercase(), value.clone());
                                    }
                                }
                                let correlated_ctx = ctx.with_outer_row(
                                    outer_row_map,
                                    std::sync::Arc::new(extended_columns.clone()),
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

            projected_rows.push(Row::from_values(new_values));
        }

        let final_result = ExecutorMemoryResult::new(projected_columns.clone(), projected_rows);
        Ok((Box::new(final_result), projected_columns, false))
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
        match expr {
            Expression::TableSource(ts) => {
                // Check if this is a CTE from context (for subqueries referencing outer CTEs)
                let table_name = &ts.name.value_lower;
                if let Some((columns, rows)) = ctx.get_cte(table_name) {
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

                    // Apply filter to CTE data if present
                    let filtered_rows = if let Some(filter_expr) = filter {
                        let row_filter = RowFilter::new(filter_expr, columns)?;
                        rows.iter()
                            .filter(|row| row_filter.matches(row))
                            .cloned()
                            .collect()
                    } else {
                        rows.clone()
                    };

                    let result =
                        super::result::ExecutorMemoryResult::new(columns.clone(), filtered_rows);
                    return Ok((Box::new(result), qualified_columns));
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
                let (result, columns, _) = self.execute_simple_table_scan(ts, &select_all, ctx)?;

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
                let (result, columns, _) = self.execute_join_source(js, &select_all, ctx)?;
                Ok((result, columns))
            }
            Expression::SubquerySource(ss) => {
                // Execute subquery with incremented depth to avoid creating new TimeoutGuard
                let subquery_ctx = ctx.with_incremented_query_depth();
                let result = self.execute_select(&ss.subquery, &subquery_ctx)?;
                let columns = result.columns().to_vec();

                // Prefix column names with subquery alias (required for proper ON condition resolution)
                // Without this, ON a.id = b.id cannot resolve qualified column names
                if let Some(alias) = &ss.alias {
                    let qualified_columns: Vec<String> = columns
                        .iter()
                        .map(|col| format!("{}.{}", alias.value, col))
                        .collect();
                    return Ok((result, qualified_columns));
                }

                Ok((result, columns))
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
                let (result, columns, _) = self.execute_values_source(vs, &select_all, ctx)?;
                Ok((result, columns))
            }
            _ => Err(Error::NotSupportedMessage(
                "Unsupported table expression type".to_string(),
            )),
        }
    }

    /// Materialize a result into a vector of rows
    pub(crate) fn materialize_result(mut result: Box<dyn QueryResult>) -> Result<Vec<Row>> {
        let mut rows = Vec::new();
        while result.next() {
            rows.push(result.take_row());
        }
        Ok(rows)
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
        rows: Vec<Row>,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Row>> {
        self.project_rows_with_alias(select_exprs, rows, all_columns, ctx, None)
    }

    /// Project rows with optional table alias for correlated subquery support
    pub(crate) fn project_rows_with_alias(
        &self,
        select_exprs: &[Expression],
        rows: Vec<Row>,
        all_columns: &[String],
        ctx: &ExecutionContext,
        table_alias: Option<&str>,
    ) -> Result<Vec<Row>> {
        // Check if this is SELECT * (no projection needed)
        // Note: QualifiedStar (t.*) DOES need projection to filter columns
        if select_exprs.len() == 1 && matches!(&select_exprs[0], Expression::Star(_)) {
            return Ok(rows);
        }

        // Handle SELECT t.* - filter to only columns matching the qualifier
        if select_exprs.len() == 1 {
            if let Expression::QualifiedStar(qs) = &select_exprs[0] {
                let prefix = format!("{}.", qs.qualifier);
                let prefix_lower = prefix.to_lowercase();
                let qualifier_lower = qs.qualifier.to_lowercase();

                // Find indices of columns matching the qualifier
                let matching_indices: Vec<usize> = all_columns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.to_lowercase().starts_with(&prefix_lower))
                    .map(|(i, _)| i)
                    .collect();

                // If no columns matched the prefix (single-table query), check if
                // the qualifier matches the table alias - if so, include all columns
                let indices_to_use = if matching_indices.is_empty() {
                    if let Some(alias) = table_alias {
                        if alias.to_lowercase() == qualifier_lower {
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

                // Project rows to only include matching columns
                let projected: Vec<Row> = rows
                    .into_iter()
                    .map(|row| row.clone_subset(&indices_to_use))
                    .collect();

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

        // OPTIMIZATION: Pre-compute lowercase column names ONCE and reuse throughout
        // This avoids repeated to_lowercase() calls in col_index_map building,
        // QualifiedStar expansion, and row projection loop
        let all_columns_lower: Vec<String> = all_columns.iter().map(|c| c.to_lowercase()).collect();

        // Build column index map ONCE with FxHashMap for O(1) lookup
        // Use the pre-computed lowercase names
        let mut col_index_map_lower: FxHashMap<String, usize> = FxHashMap::default();
        for (i, lower) in all_columns_lower.iter().enumerate() {
            col_index_map_lower.insert(lower.clone(), i);

            // Also add unqualified column names for qualified columns (e.g., "table.column" -> "column")
            // This allows SELECT column FROM t1 JOIN t2 ON ... to work
            // Don't overwrite existing entries to handle ambiguous column names correctly
            if let Some(dot_idx) = lower.rfind('.') {
                let column_part = &lower[dot_idx + 1..];
                col_index_map_lower
                    .entry(column_part.to_string())
                    .or_insert(i);
            }
        }

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
                    let prefix_lower = format!("{}.", qs.qualifier.to_lowercase());
                    let qualifier_lower = qs.qualifier.to_lowercase();
                    let mut found_any = false;
                    // OPTIMIZATION: Use pre-computed all_columns_lower instead of to_lowercase() per column
                    for (idx, col_lower) in all_columns_lower.iter().enumerate() {
                        if col_lower.starts_with(&prefix_lower) {
                            simple_column_indices.push(idx);
                            found_any = true;
                        }
                    }
                    // If no columns matched the prefix (single-table query), check if
                    // the qualifier matches the table alias - if so, include all columns
                    if !found_any {
                        if let Some(alias) = table_alias {
                            if alias.to_lowercase() == qualifier_lower {
                                for idx in 0..all_columns.len() {
                                    simple_column_indices.push(idx);
                                }
                            }
                        }
                    }
                }
                Expression::Identifier(id) => {
                    // Use O(1) HashMap lookup instead of O(n) position()
                    if let Some(&idx) = col_index_map_lower.get(&id.value_lower) {
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
                    if let Some(&idx) = col_index_map_lower.get(&full_name) {
                        simple_column_indices.push(idx);
                    } else if let Some(&idx) = col_index_map_lower.get(&qid.name.value_lower) {
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
                            if let Some(&idx) = col_index_map_lower.get(&id.value_lower) {
                                simple_column_indices.push(idx);
                            } else {
                                all_simple = false;
                                break;
                            }
                        }
                        Expression::QualifiedIdentifier(qid) => {
                            let full_name =
                                format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                            if let Some(&idx) = col_index_map_lower.get(&full_name) {
                                simple_column_indices.push(idx);
                            } else if let Some(&idx) =
                                col_index_map_lower.get(&qid.name.value_lower)
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
                // OPTIMIZATION: Use take_columns to move values instead of cloning
                let projected: Vec<Row> = rows
                    .into_iter()
                    .map(|row| row.take_columns(&simple_column_indices))
                    .collect();
                return Ok(projected);
            }
            // Fall through to slow path if there are duplicates
        }

        // Slow path: Use Evaluator for complex expressions
        let mut projected = Vec::with_capacity(rows.len());

        // Create evaluator once and reuse for all rows
        let mut evaluator = CompiledEvaluator::new(&self.function_registry);
        evaluator = evaluator.with_context(ctx);
        evaluator.init_columns(all_columns);

        // OPTIMIZATION: Pre-compute column name mappings outside the loop for correlated subqueries
        let column_keys: Option<Vec<ColumnKeyMapping>> = if has_correlated_select {
            Some(ColumnKeyMapping::build_mappings(all_columns, table_alias))
        } else {
            None
        };

        // OPTIMIZATION: Pre-allocate outer_row_map with capacity and reuse
        let base_capacity = all_columns.len() * 2;
        let mut outer_row_map: FxHashMap<String, Value> = FxHashMap::default();
        if has_correlated_select {
            outer_row_map.reserve(base_capacity);
        }

        // OPTIMIZATION: Wrap all_columns in Arc once, reuse for all rows (only if needed)
        let all_columns_arc: Option<Arc<Vec<String>>> = if has_correlated_select {
            Some(Arc::new(all_columns.to_vec()))
        } else {
            None
        };

        // OPTIMIZATION: Pre-compute table alias lowercase (all_columns_lower computed earlier)
        let table_alias_lower: Option<String> = table_alias.map(|a| a.to_lowercase());

        // OPTIMIZATION: Reuse col_index_map_lower (already built above) for O(1) lookup
        // instead of O(N) linear search in evaluate_select_expr
        for row in rows {
            let mut values = Vec::with_capacity(select_exprs.len());

            evaluator.set_row_array(&row);

            // For correlated subqueries in SELECT, we need to process them per-row
            let exprs_to_eval: std::borrow::Cow<[Expression]> = if has_correlated_select {
                // OPTIMIZATION: Clear and reuse outer_row_map instead of creating new
                outer_row_map.clear();

                // Use pre-computed column mappings
                if let Some(ref keys) = column_keys {
                    for mapping in keys {
                        if let Some(value) = row.get(mapping.index) {
                            // Insert with lowercase column name
                            outer_row_map.insert(mapping.col_lower.clone(), value.clone());

                            // Insert with qualified name if available
                            if let Some(ref qname) = mapping.qualified_name {
                                outer_row_map.insert(qname.clone(), value.clone());
                            }

                            // Insert with unqualified part if column had a dot
                            if let Some(ref upart) = mapping.unqualified_part {
                                outer_row_map.insert(upart.clone(), value.clone());
                            }
                        }
                    }
                }

                // Create context with outer row (cheap due to Arc)
                // SAFETY: all_columns_arc is always Some when has_correlated_select is true
                let correlated_ctx = ctx.with_outer_row(
                    std::mem::take(&mut outer_row_map),
                    all_columns_arc.clone().unwrap(), // Arc clone = cheap
                );

                // Process correlated SELECT expressions for this row
                let processed_exprs: Result<Vec<Expression>> = select_exprs
                    .iter()
                    .map(|expr| self.process_correlated_expression(expr, &correlated_ctx))
                    .collect();

                // Take back the map for reuse
                outer_row_map = correlated_ctx.outer_row.unwrap_or_default();

                std::borrow::Cow::Owned(processed_exprs?)
            } else {
                std::borrow::Cow::Borrowed(select_exprs)
            };

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
                        let prefix_lower = format!("{}.", qs.qualifier.to_lowercase());
                        let qualifier_lower = qs.qualifier.to_lowercase();
                        let mut found_any = false;
                        // OPTIMIZATION: Use pre-computed all_columns_lower instead of to_lowercase() per row
                        for (idx, col_lower) in all_columns_lower.iter().enumerate() {
                            if col_lower.starts_with(&prefix_lower) {
                                if let Some(val) = row.get(idx) {
                                    values.push(val.clone());
                                    found_any = true;
                                }
                            }
                        }
                        // If no columns matched the prefix (single-table query), check if
                        // the qualifier matches the table alias - if so, include all columns
                        if !found_any {
                            // OPTIMIZATION: Use pre-computed table_alias_lower
                            if let Some(ref alias_lower) = table_alias_lower {
                                if *alias_lower == qualifier_lower {
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

            projected.push(Row::from_values(values));
        }

        Ok(projected)
    }

    /// Project rows including ORDER BY columns not in SELECT
    /// Returns rows with SELECT columns followed by ORDER BY columns
    fn project_rows_with_order_by(
        &self,
        select_exprs: &[Expression],
        order_by: &[crate::parser::ast::OrderByExpression],
        rows: Vec<Row>,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<Row>, Vec<String>)> {
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
                if !select_column_names.contains(&id.value_lower) {
                    if let Some(&idx) = col_index_map_lower.get(&id.value_lower) {
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
                    select_column_indices.push(col_index_map_lower.get(&id.value_lower).copied());
                }
                Expression::QualifiedIdentifier(qid) => {
                    select_column_indices
                        .push(col_index_map_lower.get(&qid.name.value_lower).copied());
                }
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => {
                        select_column_indices
                            .push(col_index_map_lower.get(&id.value_lower).copied());
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        select_column_indices
                            .push(col_index_map_lower.get(&qid.name.value_lower).copied());
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
            let mut projected = Vec::with_capacity(rows.len());
            let num_select_cols = select_column_indices.len();
            let num_extra_cols = extra_order_indices.len();

            for row in rows {
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
                projected.push(Row::from_values(values));
            }

            // Output column names will be computed in caller
            Ok((projected, vec![]))
        } else {
            // Slow path: Use Evaluator for complex expressions
            let mut projected = Vec::with_capacity(rows.len());

            // Create evaluator once and reuse for all rows
            let mut evaluator = CompiledEvaluator::new(&self.function_registry);
            evaluator = evaluator.with_context(ctx);
            evaluator.init_columns(all_columns);

            // OPTIMIZATION: Reuse col_index_map_lower for O(1) lookup
            for row in rows {
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

                projected.push(Row::from_values(values));
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
        col_index_map: &FxHashMap<String, usize>,
    ) -> Result<Value> {
        match expr {
            // Simple column reference - get from row by index using O(1) HashMap lookup
            Expression::Identifier(id) => {
                // Use pre-computed value_lower for O(1) lookup - no allocation!
                if let Some(&idx) = col_index_map.get(&id.value_lower) {
                    Ok(row.get(idx).cloned().unwrap_or(Value::null_unknown()))
                } else {
                    Err(Error::ColumnNotFoundNamed(id.value.clone()))
                }
            }
            // Qualified identifier (table.column) - O(1) lookup
            Expression::QualifiedIdentifier(qid) => {
                // IMPORTANT: Try full qualified name FIRST to handle same-named columns
                // across different tables (e.g., t1.name vs t2.name in JOINs)
                let full_name = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                if let Some(&idx) = col_index_map.get(&full_name) {
                    Ok(row.get(idx).cloned().unwrap_or(Value::null_unknown()))
                } else if let Some(&idx) = col_index_map.get(&qid.name.value_lower) {
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

        for (i, expr) in select_exprs.iter().enumerate() {
            match expr {
                Expression::Star(_) => {
                    // SELECT * - add all column names
                    names.extend(all_columns.iter().cloned());
                    continue;
                }
                Expression::QualifiedStar(qs) => {
                    // SELECT t.* - add columns matching the qualifier
                    let prefix = format!("{}.", qs.qualifier);
                    let prefix_lower = prefix.to_lowercase();
                    let qualifier_lower = qs.qualifier.to_lowercase();
                    let mut found_any = false;
                    for col in all_columns {
                        if col.to_lowercase().starts_with(&prefix_lower) {
                            // Strip the prefix from the column name for the output
                            // e.g., "e.name" becomes "name"
                            names.push(col[prefix.len()..].to_string());
                            found_any = true;
                        }
                    }
                    // If no columns matched the prefix (single-table query), check if
                    // the qualifier matches the table alias - if so, include all columns
                    if !found_any {
                        if let Some(alias) = table_alias {
                            if alias.to_lowercase() == qualifier_lower {
                                names.extend(all_columns.iter().cloned());
                            }
                        }
                    }
                    continue;
                }
                _ => {}
            }
            let name = match expr {
                Expression::Identifier(id) => id.value.clone(),
                Expression::QualifiedIdentifier(qid) => {
                    // For qualified identifiers, just use the base column name
                    // Ambiguity in output is only an issue if the same base name appears
                    // multiple times in the SELECT clause itself (handled below)
                    qid.name.value.clone()
                }
                Expression::Aliased(aliased) => aliased.alias.value.clone(),
                Expression::FunctionCall(func) => {
                    // Use function name as column name
                    func.function.clone()
                }
                Expression::Cast(cast) => {
                    // Try to derive name from inner expression
                    match &*cast.expr {
                        Expression::Identifier(id) => id.value.clone(),
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
                    if let Some(&idx) = col_index_map.get(&id.value_lower) {
                        indices.push(idx);
                        names.push(id.value.clone());
                    } else {
                        return None;
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    if let Some(&idx) = col_index_map.get(&qid.name.value_lower) {
                        indices.push(idx);
                        names.push(qid.name.value.clone());
                    } else {
                        return None;
                    }
                }
                Expression::Aliased(aliased) => {
                    // Check if inner expression is simple
                    match &*aliased.expression {
                        Expression::Identifier(id) => {
                            if let Some(&idx) = col_index_map.get(&id.value_lower) {
                                indices.push(idx);
                                names.push(aliased.alias.value.clone());
                            } else {
                                return None;
                            }
                        }
                        Expression::QualifiedIdentifier(qid) => {
                            if let Some(&idx) = col_index_map.get(&qid.name.value_lower) {
                                indices.push(idx);
                                names.push(aliased.alias.value.clone());
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
                let rows = vec![Row::from_values(vec![Value::text(
                    "Snapshot created successfully",
                )])];
                Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
            }
            "CHECKPOINT" => {
                // Alias for SNAPSHOT (SQLite-style)
                if stmt.value.is_some() {
                    return Err(Error::internal("PRAGMA CHECKPOINT does not accept values"));
                }

                self.engine.create_snapshot()?;

                let columns = vec!["result".to_string()];
                let rows = vec![Row::from_values(vec![Value::text(
                    "Checkpoint created successfully",
                )])];
                Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
            }
            "SNAPSHOT_INTERVAL" => {
                let config = self.engine.config();
                let columns = vec![pragma_name.to_lowercase()];

                if let Some(ref value) = stmt.value {
                    // Set mode: PRAGMA snapshot_interval = 60
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.snapshot_interval = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let rows = vec![Row::from_values(vec![Value::Integer(new_value)])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                } else {
                    // Read mode: PRAGMA snapshot_interval
                    let rows = vec![Row::from_values(vec![Value::Integer(
                        config.persistence.snapshot_interval as i64,
                    )])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                }
            }
            "KEEP_SNAPSHOTS" => {
                let config = self.engine.config();
                let columns = vec![pragma_name.to_lowercase()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.keep_snapshots = new_value as u32;
                    self.engine.update_engine_config(new_config)?;
                    let rows = vec![Row::from_values(vec![Value::Integer(new_value)])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                } else {
                    let rows = vec![Row::from_values(vec![Value::Integer(
                        config.persistence.keep_snapshots as i64,
                    )])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                }
            }
            "SYNC_MODE" => {
                let config = self.engine.config();
                let columns = vec![pragma_name.to_lowercase()];

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
                    let rows = vec![Row::from_values(vec![Value::Integer(new_value)])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                } else {
                    let rows = vec![Row::from_values(vec![Value::Integer(
                        config.persistence.sync_mode as i64,
                    )])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                }
            }
            "WAL_FLUSH_TRIGGER" => {
                let config = self.engine.config();
                let columns = vec![pragma_name.to_lowercase()];

                if let Some(ref value) = stmt.value {
                    let new_value = self.extract_pragma_int_value(value)?;
                    let mut new_config = config.clone();
                    new_config.persistence.wal_flush_trigger = new_value as usize;
                    self.engine.update_engine_config(new_config)?;
                    let rows = vec![Row::from_values(vec![Value::Integer(new_value)])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
                } else {
                    let rows = vec![Row::from_values(vec![Value::Integer(
                        config.persistence.wal_flush_trigger as i64,
                    )])];
                    Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
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
            .eval_slice(&[])?;
        let columns = vec!["result".to_string()];
        let rows = vec![Row::from_values(vec![value])];

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Try to optimize simple MIN/MAX aggregates using index
    ///
    /// For queries like `SELECT MIN(col) FROM table` or `SELECT MAX(col) FROM table`
    /// without WHERE or GROUP BY, use the index's O(1) min/max lookup instead of O(n) scan.
    #[allow(clippy::type_complexity)]
    fn try_min_max_index_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn crate::storage::traits::Table,
        _all_columns: &[String],
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Only optimize single MIN or MAX without DISTINCT
        if stmt.columns.len() != 1 {
            return Ok(None);
        }

        let col_expr = &stmt.columns[0];

        // Extract function info (handle aliased case too)
        let (func, alias) = match col_expr {
            Expression::FunctionCall(func) => (func, None),
            Expression::Aliased(aliased) => {
                if let Expression::FunctionCall(func) = aliased.expression.as_ref() {
                    (func, Some(aliased.alias.value.clone()))
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Check if it's MIN or MAX
        // OPTIMIZATION: func.function is already uppercase from parsing
        if func.function != "MIN" && func.function != "MAX" {
            return Ok(None);
        }

        // Don't optimize DISTINCT
        if func.is_distinct {
            return Ok(None);
        }

        // Extract column name
        if func.arguments.is_empty() {
            return Ok(None);
        }

        let column_name = match &func.arguments[0] {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            _ => return Ok(None),
        };

        // Try to get value from index
        let value = if func.function == "MIN" {
            table.get_index_min_value(&column_name)
        } else {
            table.get_index_max_value(&column_name)
        };

        if let Some(val) = value {
            // Build result
            let col_name = alias.unwrap_or_else(|| format!("{}({})", func.function, column_name));
            let columns = vec![col_name.clone()];
            let rows = vec![Row::from_values(vec![val])];
            let result: Box<dyn QueryResult> =
                Box::new(ExecutorMemoryResult::new(columns.clone(), rows));
            return Ok(Some((result, columns)));
        }

        Ok(None)
    }

    /// Try to optimize simple COUNT(*) queries using table row_count
    ///
    /// For queries like `SELECT COUNT(*) FROM table` without WHERE or GROUP BY,
    /// use the table's O(1) row_count() method instead of O(n) scan.
    #[allow(clippy::type_complexity)]
    fn try_count_star_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn crate::storage::traits::Table,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Only optimize single COUNT(*) without DISTINCT
        if stmt.columns.len() != 1 {
            return Ok(None);
        }

        let col_expr = &stmt.columns[0];

        // Extract function info (handle aliased case too)
        let (func, alias) = match col_expr {
            Expression::FunctionCall(func) => (func, None),
            Expression::Aliased(aliased) => {
                if let Expression::FunctionCall(func) = aliased.expression.as_ref() {
                    (func, Some(aliased.alias.value.clone()))
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Check if it's COUNT
        // OPTIMIZATION: func.function is already uppercase from parsing
        if func.function != "COUNT" {
            return Ok(None);
        }

        // Don't optimize DISTINCT
        if func.is_distinct {
            return Ok(None);
        }

        // Don't optimize if FILTER clause is present - requires row-by-row evaluation
        if func.filter.is_some() {
            return Ok(None);
        }

        // Must be COUNT(*) - either empty args or Star expression
        let is_count_star = func.arguments.is_empty()
            || (func.arguments.len() == 1 && matches!(func.arguments[0], Expression::Star(_)));

        if !is_count_star {
            return Ok(None);
        }

        // Use table's row_count method (O(1) instead of O(n))
        let count = table.row_count();

        // Build result
        let col_name = alias.unwrap_or_else(|| "COUNT(*)".to_string());
        let columns = vec![col_name.clone()];
        let rows = vec![Row::from_values(vec![Value::Integer(count as i64)])];
        let result: Box<dyn QueryResult> =
            Box::new(ExecutorMemoryResult::new(columns.clone(), rows));
        Ok(Some((result, columns)))
    }

    /// Try to optimize ORDER BY + LIMIT using index-ordered scan
    ///
    /// For queries like `SELECT * FROM table ORDER BY col LIMIT 10`,
    /// use the index to get rows in sorted order directly (O(limit) instead of O(n log n)).
    #[allow(clippy::type_complexity)]
    fn try_order_by_index_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn crate::storage::traits::Table,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Get the ORDER BY column name
        let order_by = &stmt.order_by[0];
        let column_name = match &order_by.expression {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            _ => return Ok(None), // Can't optimize complex ORDER BY expressions
        };

        // Determine sort order
        let ascending = order_by.ascending;

        // Evaluate limit and offset
        let limit = if let Some(ref limit_expr) = stmt.limit {
            match ExpressionEval::compile(limit_expr, &[])
                .ok()
                .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
            {
                Some(Value::Integer(l)) => l as usize,
                Some(Value::Float(f)) => f as usize,
                _ => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        let offset = if let Some(ref offset_expr) = stmt.offset {
            match ExpressionEval::compile(offset_expr, &[])
                .ok()
                .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
            {
                Some(Value::Integer(o)) => o as usize,
                Some(Value::Float(f)) => f as usize,
                _ => 0,
            }
        } else {
            0
        };

        // Try to use index-ordered scan
        if let Some(rows) =
            table.collect_rows_ordered_by_index(&column_name, ascending, limit, offset)
        {
            // Project rows according to SELECT expressions
            let projected_rows = self.project_rows(&stmt.columns, rows, all_columns, ctx)?;

            // Note: This optimization path doesn't have table_alias available,
            // so we pass None. The prefix-based matching will still work for JOINs.
            let output_columns = self.get_output_column_names(&stmt.columns, all_columns, None);

            let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
            return Ok(Some((Box::new(result), output_columns)));
        }

        Ok(None)
    }

    /// Extract window ORDER BY information for optimization
    /// Returns (column_name, ascending) if a simple optimizable case is found
    fn extract_window_order_info(stmt: &SelectStatement) -> Option<(String, bool)> {
        // Look for window functions in SELECT columns
        for col_expr in &stmt.columns {
            if let Some(info) = Self::find_window_order_in_expr(col_expr) {
                return Some(info);
            }
        }
        None
    }

    /// Find window ORDER BY info in an expression
    fn find_window_order_in_expr(expr: &Expression) -> Option<(String, bool)> {
        match expr {
            Expression::Window(window_expr) => {
                // Check for no PARTITION BY (single partition case)
                if !window_expr.partition_by.is_empty() {
                    return None; // Pre-sorting doesn't help with partitions
                }

                // Check if using a window reference (can't analyze those)
                if window_expr.window_ref.is_some() {
                    return None;
                }

                // Get ORDER BY info
                let order_by = &window_expr.order_by;

                // Only optimize if exactly one simple ORDER BY column
                if order_by.len() != 1 {
                    return None;
                }

                let order = &order_by[0];
                let column_name = match &order.expression {
                    Expression::Identifier(id) => id.value.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
                    _ => return None, // Complex expression, can't optimize
                };

                Some((column_name, order.ascending))
            }
            Expression::Aliased(aliased) => Self::find_window_order_in_expr(&aliased.expression),
            _ => None,
        }
    }

    /// Extract window PARTITION BY information for optimization
    /// Returns column_name if a simple single-column PARTITION BY is found
    fn extract_window_partition_info(stmt: &SelectStatement) -> Option<String> {
        // Look for window functions in SELECT columns
        for col_expr in &stmt.columns {
            if let Some(info) = Self::find_window_partition_in_expr(col_expr) {
                return Some(info);
            }
        }
        None
    }

    /// Find window PARTITION BY info in an expression
    fn find_window_partition_in_expr(expr: &Expression) -> Option<String> {
        match expr {
            Expression::Window(window_expr) => {
                // Only optimize single-column PARTITION BY
                if window_expr.partition_by.len() != 1 {
                    return None;
                }

                // Check if using a window reference (can't analyze those)
                if window_expr.window_ref.is_some() {
                    return None;
                }

                // Get PARTITION BY column
                let partition_col = &window_expr.partition_by[0];
                let column_name = match partition_col {
                    Expression::Identifier(id) => id.value.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
                    _ => return None, // Complex expression, can't optimize
                };

                Some(column_name)
            }
            Expression::Aliased(aliased) => {
                Self::find_window_partition_in_expr(&aliased.expression)
            }
            _ => None,
        }
    }

    /// IN subquery index optimization
    ///
    /// For queries like `SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE ...)`
    /// where `id` has an index, probe the index for each subquery value instead of scanning all rows.
    /// This is O(k log n) where k = subquery result size, vs O(n) for full table scan.
    #[allow(clippy::type_complexity)]
    fn try_in_subquery_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn crate::storage::traits::Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract IN subquery info: (column_name, subquery, is_negated, remaining_predicate)
        let (column_name, subquery, is_negated, remaining_predicate) =
            match Self::extract_in_subquery_info(where_expr) {
                Some(info) => info,
                None => return Ok(None),
            };

        // Skip correlated subqueries - they can't be pre-evaluated
        if Self::is_subquery_correlated(&subquery.subquery) {
            return Ok(None);
        }

        // Skip if SELECT columns have correlated subqueries (need per-row context)
        for col in &stmt.columns {
            if Self::has_correlated_subqueries(col) {
                return Ok(None);
            }
        }

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            schema.columns[pk_col_idx].name.to_lowercase() == column_name
        };

        // If not PK, check for index
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        // Execute the subquery to get values (with caching for non-correlated subqueries)
        let cache_key = subquery.subquery.to_string();
        let values = if let Some(cached) = get_cached_in_subquery(&cache_key) {
            cached
        } else {
            let subquery_ctx = ctx.with_incremented_query_depth();
            let mut result = self.execute_select(&subquery.subquery, &subquery_ctx)?;

            // Collect all values from the first column
            let mut values = Vec::new();
            while result.next() {
                let row = result.row();
                if !row.is_empty() {
                    values.push(
                        row.get(0)
                            .cloned()
                            .unwrap_or_else(crate::core::Value::null_unknown),
                    );
                }
            }
            // Cache for future use
            cache_in_subquery(cache_key, values.clone());
            values
        };

        if values.is_empty() {
            // Empty subquery result
            if is_negated {
                // NOT IN empty set = all rows match (fall through to normal scan)
                return Ok(None);
            } else {
                // IN empty set = no rows match
                let output_columns =
                    self.get_output_column_names(&stmt.columns, all_columns, table_alias);
                let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
                return Ok(Some((Box::new(result), output_columns)));
            }
        }

        // For NOT IN, we can't easily use the index (would need to exclude rows)
        // Fall back to normal scan
        if is_negated {
            return Ok(None);
        }

        // Collect row_ids: either from PK (direct) or from index probe
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            // PRIMARY KEY: the value IS the row_id (for INTEGER PK)
            for value in &values {
                if let Value::Integer(id) = value {
                    all_row_ids.push(*id);
                }
            }
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in &values {
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // Remove duplicates and sort for efficient batch fetch
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Process any subqueries in the remaining predicate
            let processed_remaining = if Self::has_subqueries(remaining) {
                self.process_where_subqueries(remaining, ctx)?
            } else {
                remaining.clone()
            };

            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter =
                crate::executor::expression::RowFilter::new(&processed_remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// IN list literal index optimization
    ///
    /// For queries like `SELECT * FROM users WHERE id IN (1, 2, 3, 5, 8)`
    /// where `id` has an index or is PRIMARY KEY, probe the index directly for each value
    /// instead of scanning all rows. This is O(k log n) where k = list size.
    #[allow(clippy::type_complexity)]
    fn try_in_list_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn crate::storage::traits::Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract IN list info: (column_name, values, is_negated, remaining_predicate)
        let (column_name, values, is_negated, remaining_predicate) =
            match Self::extract_in_list_info(where_expr, ctx) {
                Some(info) => info,
                None => return Ok(None),
            };

        // Skip if SELECT columns have correlated subqueries (need per-row context)
        for col in &stmt.columns {
            if Self::has_correlated_subqueries(col) {
                return Ok(None);
            }
        }

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            schema.columns[pk_col_idx].name.to_lowercase() == column_name
        };

        // If not PK, check for index
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        if values.is_empty() {
            // Empty IN list
            if is_negated {
                // NOT IN empty set = all rows match (fall through to normal scan)
                return Ok(None);
            } else {
                // IN empty set = no rows match
                let output_columns =
                    self.get_output_column_names(&stmt.columns, all_columns, table_alias);
                let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
                return Ok(Some((Box::new(result), output_columns)));
            }
        }

        // For NOT IN, we can't easily use the index (would need to exclude rows)
        // Fall back to normal scan
        if is_negated {
            return Ok(None);
        }

        // Collect row_ids: either from PK (direct) or from index probe
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            // PRIMARY KEY: the value IS the row_id (for INTEGER PK)
            for value in &values {
                if let Value::Integer(id) = value {
                    all_row_ids.push(*id);
                }
            }
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in &values {
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // Remove duplicates and sort for efficient batch fetch
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Process any subqueries in the remaining predicate
            let processed_remaining = if Self::has_subqueries(remaining) {
                self.process_where_subqueries(remaining, ctx)?
            } else {
                remaining.clone()
            };

            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter =
                crate::executor::expression::RowFilter::new(&processed_remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// Extract IN list literal information from a WHERE clause.
    /// Returns (column_name, values, is_negated, remaining_predicate)
    fn extract_in_list_info(
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<(String, Vec<Value>, bool, Option<Expression>)> {
        match expr {
            // Direct IN list: column IN (v1, v2, ...)
            Expression::In(in_expr) => {
                // Get the column name from the left side (lowercase for case-insensitive match)
                let column_name = match in_expr.left.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex left expressions
                };

                // Get the values from the right side (must be a literal list, not subquery)
                let values = match in_expr.right.as_ref() {
                    Expression::List(list) => {
                        // ListExpression has Vec<Expression>
                        Self::extract_literal_values(&list.elements, ctx)
                    }
                    Expression::ExpressionList(list) => {
                        // ExpressionList has Vec<Expression>
                        Self::extract_literal_values(&list.expressions, ctx)
                    }
                    _ => return None, // Not a literal list (might be subquery)
                };

                values.map(|v| (column_name, v, in_expr.not, None))
            }

            // IN list with AND: column IN (...) AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as IN list
                if let Some((col, vals, neg, _)) = Self::extract_in_list_info(&infix.left, ctx) {
                    return Some((col, vals, neg, Some((*infix.right).clone())));
                }
                // Try right side as IN list
                if let Some((col, vals, neg, _)) = Self::extract_in_list_info(&infix.right, ctx) {
                    return Some((col, vals, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
    }

    /// Extract literal values from a list of expressions.
    /// Returns None if any expression is not a literal (e.g., column reference, subquery).
    fn extract_literal_values(exprs: &[Expression], ctx: &ExecutionContext) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(exprs.len());
        for expr in exprs {
            // Try to evaluate as a constant expression
            match ExpressionEval::compile(expr, &[]) {
                Ok(compiled) => {
                    match compiled.with_context(ctx).eval_slice(&[]) {
                        Ok(val) => values.push(val),
                        Err(_) => return None, // Can't evaluate as constant
                    }
                }
                Err(_) => return None, // Can't compile (e.g., column reference)
            }
        }
        Some(values)
    }

    /// Extract IN subquery information from a WHERE clause.
    /// Returns (column_name, subquery, is_negated, remaining_predicate)
    fn extract_in_subquery_info(
        expr: &Expression,
    ) -> Option<(String, &ScalarSubquery, bool, Option<Expression>)> {
        match expr {
            // Direct IN subquery: column IN (SELECT ...)
            Expression::In(in_expr) => {
                // Get the column name from the left side (lowercase for case-insensitive match)
                let column_name = match in_expr.left.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex left expressions
                };

                // Get the subquery from the right side
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    return Some((column_name, subquery, in_expr.not, None));
                }
                None
            }

            // IN subquery with AND: column IN (SELECT ...) AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as IN subquery
                if let Some((col, sq, neg, _)) = Self::extract_in_subquery_info(&infix.left) {
                    return Some((col, sq, neg, Some((*infix.right).clone())));
                }
                // Try right side as IN subquery
                if let Some((col, sq, neg, _)) = Self::extract_in_subquery_info(&infix.right) {
                    return Some((col, sq, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
    }

    /// Try to optimize InHashSet expressions (from EXISTS → semi-join transformation).
    ///
    /// When EXISTS is transformed to InHashSet via semi-join, we can further optimize
    /// by probing the PRIMARY KEY or index directly instead of scanning all rows.
    ///
    /// For example: `SELECT * FROM users WHERE users.id IN {1, 2, 3} LIMIT 100`
    /// Instead of scanning all 10,000 users, probe the PK for ids 1, 2, 3 directly.
    #[allow(clippy::type_complexity)]
    fn try_in_hashset_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn crate::storage::traits::Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract InHashSet info: (column_name, values, is_negated, remaining_predicate)
        let (column_name, values, is_negated, remaining_predicate) =
            match Self::extract_in_hashset_info(where_expr) {
                Some(info) => info,
                None => return Ok(None),
            };

        // NOT IN is harder to optimize - fall back to normal scan
        if is_negated {
            return Ok(None);
        }

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            let pk_col_name = schema.columns[pk_col_idx].name.to_lowercase();
            pk_col_name == column_name
        };

        // If not PK, check for index
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        // EARLY LIMIT CHECK: Compute limit+offset before collecting row_ids
        // This allows us to stop collection early when there's no ORDER BY
        let early_termination_target = if stmt.order_by.is_empty() && remaining_predicate.is_none()
        {
            let offset = stmt
                .offset
                .as_ref()
                .and_then(|offset_expr| {
                    ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                        .and_then(|v| {
                            if let Value::Integer(o) = v {
                                Some(o.max(0) as usize)
                            } else {
                                None
                            }
                        })
                })
                .unwrap_or(0);

            let limit = stmt.limit.as_ref().and_then(|limit_expr| {
                ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    .and_then(|v| {
                        if let Value::Integer(l) = v {
                            Some(l.max(0) as usize)
                        } else {
                            None
                        }
                    })
            });

            limit.map(|l| offset + l)
        } else {
            None // Can't use early termination with ORDER BY or remaining predicate
        };

        // Collect row_ids: either from PK (direct) or from index probe
        // With early termination target, stop once we have enough
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            // PRIMARY KEY: the value IS the row_id (for INTEGER PK)
            for value in values.iter() {
                // Early termination check
                if let Some(target) = early_termination_target {
                    if all_row_ids.len() >= target {
                        break;
                    }
                }
                match value {
                    Value::Integer(id) => all_row_ids.push(*id),
                    Value::Float(f) => {
                        // Handle case where integer was stored as float
                        if f.fract() == 0.0 {
                            all_row_ids.push(*f as i64);
                        }
                    }
                    _ => {}
                }
            }
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in values.iter() {
                // Early termination check
                if let Some(target) = early_termination_target {
                    if all_row_ids.len() >= target {
                        break;
                    }
                }
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // If no row_ids found, return empty result
        if all_row_ids.is_empty() {
            let output_columns =
                self.get_output_column_names(&stmt.columns, all_columns, table_alias);
            let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
            return Ok(Some((Box::new(result), output_columns)));
        }

        // Remove duplicates and sort for efficient batch fetch
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter =
                crate::executor::expression::RowFilter::new(remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// Extract InHashSet information from a WHERE clause.
    /// Returns (column_name, hash_set_values, is_negated, remaining_predicate)
    #[allow(clippy::type_complexity)]
    fn extract_in_hashset_info(
        expr: &Expression,
    ) -> Option<(
        String,
        std::sync::Arc<ahash::AHashSet<Value>>,
        bool,
        Option<Expression>,
    )> {
        match expr {
            // Direct InHashSet: column IN {hash_set}
            Expression::InHashSet(in_hash) => {
                // Get the column name from the column expression (lowercase for case-insensitive match)
                let column_name = match in_hash.column.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex column expressions
                };

                Some((column_name, in_hash.values.clone(), in_hash.not, None))
            }

            // InHashSet with AND: column IN {hash_set} AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as InHashSet
                if let Some((col, vals, neg, _)) = Self::extract_in_hashset_info(&infix.left) {
                    return Some((col, vals, neg, Some((*infix.right).clone())));
                }
                // Try right side as InHashSet
                if let Some((col, vals, neg, _)) = Self::extract_in_hashset_info(&infix.right) {
                    return Some((col, vals, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
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
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
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
        let mut rows = Vec::new();
        let mut result_iter = result;
        while result_iter.next() {
            rows.push(result_iter.take_row());
        }

        // Apply in-memory WHERE filter if storage expression couldn't handle it fully
        if needs_memory_filter {
            if let Some(where_expr) = &stmt.where_clause {
                let where_filter = RowFilter::new(where_expr, &all_columns)?.with_context(ctx);

                rows.retain(|row| where_filter.matches(row));
            }
        }

        // Check for window functions
        if self.has_window_functions(stmt) {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, rows, &all_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Check for aggregation
        if self.has_aggregation(stmt) {
            let result = self.execute_select_with_aggregation(stmt, ctx, rows, &all_columns)?;
            let columns = result.columns().to_vec();
            return Ok((result, columns, false));
        }

        // Project rows
        let (projected_rows, final_columns) =
            self.project_rows_for_select(stmt, rows, &all_columns, ctx)?;

        Ok((
            Box::new(ExecutorMemoryResult::new(
                final_columns.clone(),
                projected_rows,
            )),
            final_columns,
            false,
        ))
    }

    /// Parse temporal value from AS OF clause
    fn parse_temporal_value(&self, as_of: &AsOfClause, ctx: &ExecutionContext) -> Result<i64> {
        let value = ExpressionEval::compile(&as_of.value, &[])?
            .with_context(ctx)
            .eval_slice(&[])?;

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
                Expression::Identifier(id) => columns.push(id.value.clone()),
                Expression::QualifiedIdentifier(qid) => columns.push(qid.name.value.clone()),
                Expression::Aliased(aliased) => match &*aliased.expression {
                    Expression::Identifier(id) => columns.push(id.value.clone()),
                    Expression::QualifiedIdentifier(qid) => columns.push(qid.name.value.clone()),
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
        rows: Vec<Row>,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<Row>, Vec<String>)> {
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
    fn build_alias_map(columns: &[Expression]) -> FxHashMap<String, Expression> {
        let mut alias_map = FxHashMap::default();
        for col_expr in columns {
            if let Expression::Aliased(aliased) = col_expr {
                let alias_name = aliased.alias.value_lower.clone();
                alias_map.insert(alias_name, (*aliased.expression).clone());
            }
        }
        alias_map
    }

    /// Substitute column aliases in an expression with their underlying expressions
    fn substitute_aliases(
        expr: &Expression,
        alias_map: &FxHashMap<String, Expression>,
    ) -> Expression {
        match expr {
            Expression::Identifier(id) => {
                // Check if this identifier is an alias - use pre-computed lowercase
                if let Some(original) = alias_map.get(&id.value_lower) {
                    return original.clone();
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
            Expression::FunctionCall(func) => Expression::FunctionCall(FunctionCall {
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
            }),
            Expression::Case(case) => Expression::Case(CaseExpression {
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
            }),
            Expression::List(list) => Expression::List(ListExpression {
                token: list.token.clone(),
                elements: list
                    .elements
                    .iter()
                    .map(|e| Self::substitute_aliases(e, alias_map))
                    .collect(),
            }),
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
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            Expression::Aliased(aliased) => aliased.alias.value.clone(),
            Expression::FunctionCall(func) => {
                // Use function name as column name
                func.function.clone()
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
            Expression::StringLiteral(lit) => lit.value.clone(),
            Expression::BooleanLiteral(lit) => format!("{}", lit.value),
            Expression::NullLiteral(_) => "NULL".to_string(),
            _ => "expr".to_string(),
        }
    }

    /// Check if semi-join reduction optimization can be applied and return parameters
    /// Returns Some((limit_value, left_key_col, right_key_col)) if applicable, None otherwise
    ///
    /// Conditions for semi-join reduction:
    /// 1. LEFT JOIN (preserved left side)
    /// 2. GROUP BY references only left table columns
    /// 3. LIMIT is present with no ORDER BY (or ORDER BY on left columns only)
    /// 4. Single equality join condition (a.col = b.col)
    fn get_semijoin_reduction_limit(
        &self,
        join_type: &str,
        stmt: &SelectStatement,
        left_alias: Option<&str>,
        join_condition: &Option<Box<Expression>>,
    ) -> Option<(usize, String, String)> {
        // Only applies to LEFT JOIN
        if !join_type.contains("LEFT") || join_type.contains("FULL") {
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
        if !left_key_lower.starts_with(&format!("{}.", left_alias.to_lowercase())) {
            // Maybe it's right.col = left.col (swapped)
            let right_key_lower = right_key_col.to_lowercase();
            if !right_key_lower.starts_with(&format!("{}.", left_alias.to_lowercase())) {
                return None;
            }
            // Swap the keys
            let limit_value = stmt.limit.as_ref().and_then(|e| {
                ExpressionEval::compile(e, &[])
                    .ok()
                    .and_then(|mut eval| eval.eval_slice(&[]).ok())
                    .and_then(|v| match v {
                        Value::Integer(n) if n > 0 => Some(n as usize),
                        _ => None,
                    })
            })?;
            return Some((limit_value, right_key_col, left_key_col));
        }

        // Get LIMIT value
        let limit_value = stmt.limit.as_ref().and_then(|e| {
            ExpressionEval::compile(e, &[])
                .ok()
                .and_then(|mut eval| eval.eval_slice(&[]).ok())
                .and_then(|v| match v {
                    Value::Integer(n) if n > 0 => Some(n as usize),
                    _ => None,
                })
        })?;

        Some((limit_value, left_key_col, right_key_col))
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
            Expression::Identifier(id) => Some(id.value.clone()),
            _ => None,
        }
    }

    /// Check if Index Nested Loop Join can be used.
    ///
    /// Returns Some((table_name, index, inner_key_column, outer_key_column, outer_key_idx)) if:
    /// - Right side is a simple TableSource (not CTE, view, subquery, or nested join)
    /// - There's an equality join condition on a column that has an index on the right table
    /// - Join type is INNER or LEFT (not RIGHT or FULL)
    ///
    /// The outer_key_idx will be determined after materializing the outer side.
    #[allow(clippy::type_complexity)]
    fn check_index_nested_loop_opportunity(
        &self,
        right_expr: &Expression,
        join_condition: Option<&Expression>,
        join_type: &str,
        left_alias: Option<&str>,
        right_alias: Option<&str>,
    ) -> Option<(
        String,                    // table_name
        join::IndexLookupStrategy, // lookup strategy (index or PK)
        String,                    // inner_key_column (unqualified)
        String,                    // outer_key_column (qualified)
    )> {
        // Only for INNER or LEFT joins (not RIGHT or FULL)
        // RIGHT joins would need to track all unmatched inner rows
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            return None;
        }

        // Right side must be a simple TableSource (possibly wrapped in Aliased)
        let table_source = match right_expr {
            Expression::TableSource(ts) => ts,
            Expression::Aliased(aliased) => {
                // Unwrap the alias to get the underlying TableSource
                match aliased.expression.as_ref() {
                    Expression::TableSource(ts) => ts,
                    _ => return None,
                }
            }
            _ => return None,
        };

        let table_name = &table_source.name.value_lower;

        // Check if it's a CTE (CTEs don't have indexes)
        // We can't check CTEs here without context, but we'll verify when we try to get the table

        // Need a join condition
        let condition = join_condition?;

        // Extract equality keys from the join condition
        let (left_col, right_col) = self.extract_simple_join_key(condition)?;

        // Determine which column belongs to which table
        let left_col_lower = left_col.to_lowercase();
        let right_col_lower = right_col.to_lowercase();

        // Get the right table alias (or table name if no alias)
        let right_table_alias = right_alias.unwrap_or(table_name);

        // Check if left_col is from right table and right_col is from left table (swapped)
        let (inner_col, outer_col) = if left_col_lower
            .starts_with(&format!("{}.", right_table_alias.to_lowercase()))
        {
            // left_col is actually from right (inner) table
            (left_col.clone(), right_col.clone())
        } else if right_col_lower.starts_with(&format!("{}.", right_table_alias.to_lowercase())) {
            // right_col is from right (inner) table - normal case
            (right_col.clone(), left_col.clone())
        } else {
            // Can't determine which column is from which table
            return None;
        };

        // Verify outer column is from left table (if we have left alias)
        if let Some(left_a) = left_alias {
            let outer_col_lower = outer_col.to_lowercase();
            if !outer_col_lower.starts_with(&format!("{}.", left_a.to_lowercase())) {
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
        let table = txn.get_table(table_name).ok()?;
        let schema = table.schema();

        // First check if inner column is the PRIMARY KEY (direct row_id lookup)
        let inner_col_lower = inner_col_unqualified.to_lowercase();
        if let Some(pk_idx) = schema.pk_column_index() {
            if schema.columns[pk_idx].name.to_lowercase() == inner_col_lower {
                // It's a primary key lookup - most efficient!
                return Some((
                    table_name.clone(),
                    join::IndexLookupStrategy::PrimaryKey,
                    inner_col_unqualified,
                    outer_col,
                ));
            }
        }

        // Check if there's a secondary index on the inner column
        if let Some(index) = table.get_index_on_column(&inner_col_unqualified) {
            return Some((
                table_name.clone(),
                join::IndexLookupStrategy::SecondaryIndex(index),
                inner_col_unqualified,
                outer_col,
            ));
        }

        // No index or PK available
        None
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

        // Build list of value literals
        let value_exprs: Vec<Expression> = values
            .iter()
            .map(|v| match v {
                Value::Integer(i) => Expression::IntegerLiteral(IntegerLiteral {
                    token: Token::new(TokenType::Integer, i.to_string(), Position::default()),
                    value: *i,
                }),
                Value::Float(f) => Expression::FloatLiteral(FloatLiteral {
                    token: Token::new(TokenType::Float, f.to_string(), Position::default()),
                    value: *f,
                }),
                Value::Text(s) => Expression::StringLiteral(StringLiteral {
                    token: Token::new(TokenType::String, s.to_string(), Position::default()),
                    value: s.to_string(),
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
            right: Box::new(Expression::ExpressionList(ExpressionList {
                token: Token::new(TokenType::Punctuator, "(", Position::default()),
                expressions: value_exprs,
            })),
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
