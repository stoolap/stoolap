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

use crate::core::{Error, Result, Row, Schema, Value};
use crate::optimizer::ExpressionSimplifier;
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::mvcc::engine::ViewDefinition;
use crate::storage::traits::{Engine, QueryResult, ScanPlan};

/// Maximum depth for nested views to prevent stack overflow
const MAX_VIEW_DEPTH: usize = 32;

/// Helper to create a dummy token for internal AST construction
fn dummy_token(literal: &str, token_type: TokenType) -> Token {
    Token::new(token_type, literal, Position::new(0, 1, 1))
}

use super::aggregation::expression_contains_aggregate;
use super::context::ExecutionContext;
use super::evaluator::Evaluator;
use super::parallel::{self, ParallelConfig};
use super::result::{
    DistinctResult, ExecResult, ExecutorMemoryResult, ExprFilteredResult, ExprMappedResult,
    FilteredResult, LimitedResult, OrderedResult, ProjectedResult, RadixOrderSpec, ScannerResult,
    StreamingProjectionResult, TopNResult,
};
use super::Executor;
use crate::optimizer::aqe::{decide_join_algorithm, AqeJoinDecision, JoinAqeContext};
use crate::optimizer::bloom::BloomFilter;
use crate::optimizer::feedback::{fingerprint_predicate, global_feedback_cache};
use crate::optimizer::{BuildSide, JoinAlgorithm};

/// Minimum build side size to use bloom filter optimization.
/// For small build sides, the bloom filter overhead isn't worth it.
const BLOOM_FILTER_MIN_BUILD_SIZE: usize = 100;

/// Extract the base (unqualified) column name from a potentially qualified column name.
/// For "table.column" returns "column", for "column" returns "column".
/// The result is always lowercase for case-insensitive comparisons.
#[inline]
fn extract_base_column_name(col_name: &str) -> String {
    if let Some(dot_idx) = col_name.rfind('.') {
        col_name[dot_idx + 1..].to_lowercase()
    } else {
        col_name.to_lowercase()
    }
}

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

impl Executor {
    /// Execute a SELECT statement
    pub(crate) fn execute_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
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

        // Execute the main query
        // The third return value indicates if LIMIT/OFFSET was already applied (by storage-level pushdown)
        let (mut result, columns, limit_offset_applied) =
            self.execute_select_internal(stmt, ctx)?;

        // Apply set operations (UNION, INTERSECT, EXCEPT)
        if !stmt.set_operations.is_empty() {
            result = self.execute_set_operations(result, &stmt.set_operations, ctx)?;
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

        // Evaluate LIMIT/OFFSET early (needed for TOP-N optimization)
        let limit = if let Some(ref limit_expr) = stmt.limit {
            let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
            match evaluator.evaluate(limit_expr) {
                Ok(Value::Integer(l)) => {
                    if l < 0 {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            l
                        )));
                    }
                    Some(l as usize)
                }
                Ok(Value::Float(f)) => {
                    // Convert float to integer (truncate)
                    let l = f as i64;
                    if l < 0 {
                        return Err(Error::ParseError(format!(
                            "LIMIT must be non-negative, got {}",
                            l
                        )));
                    }
                    Some(l as usize)
                }
                Ok(other) => {
                    return Err(Error::ParseError(format!(
                        "LIMIT must be an integer, got {:?}",
                        other
                    )));
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let offset = if let Some(ref offset_expr) = stmt.offset {
            let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
            match evaluator.evaluate(offset_expr) {
                Ok(Value::Integer(o)) => {
                    if o < 0 {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            o
                        )));
                    }
                    o as usize
                }
                Ok(Value::Float(f)) => {
                    // Convert float to integer (truncate)
                    let o = f as i64;
                    if o < 0 {
                        return Err(Error::ParseError(format!(
                            "OFFSET must be non-negative, got {}",
                            o
                        )));
                    }
                    o as usize
                }
                Ok(other) => {
                    return Err(Error::ParseError(format!(
                        "OFFSET must be an integer, got {:?}",
                        other
                    )));
                }
                Err(_) => 0,
            }
        } else {
            0
        };

        // Apply ORDER BY (with TOP-N optimization if LIMIT is present)
        if !stmt.order_by.is_empty() {
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
                let mut evaluator = Evaluator::new(&self.function_registry);
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
                            let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
                                rustc_hash::FxHashMap::default();
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
                                                    Evaluator::new(&self.function_registry);
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

    /// Execute set operations (UNION, INTERSECT, EXCEPT)
    fn execute_set_operations(
        &self,
        left_result: Box<dyn QueryResult>,
        set_ops: &[SetOperation],
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use rustc_hash::FxHashSet;

        // Materialize the left result
        let columns = left_result.columns().to_vec();
        let mut result_rows = Self::materialize_result(left_result)?;

        // Process each set operation in sequence
        for set_op in set_ops {
            // Execute the right side query
            let right_result = self.execute_select(&set_op.right, ctx)?;

            // Validate column count matches (SQL standard requirement)
            let right_col_count = right_result.columns().len();
            let left_col_count = columns.len();
            if left_col_count != right_col_count {
                return Err(crate::Error::internal(format!(
                    "each {} query must have the same number of columns: left has {}, right has {}",
                    match &set_op.operation {
                        SetOperationType::Union | SetOperationType::UnionAll => "UNION",
                        SetOperationType::Intersect | SetOperationType::IntersectAll => "INTERSECT",
                        SetOperationType::Except | SetOperationType::ExceptAll => "EXCEPT",
                    },
                    left_col_count,
                    right_col_count
                )));
            }

            let right_rows = Self::materialize_result(right_result)?;

            // Apply the set operation
            match &set_op.operation {
                SetOperationType::Union => {
                    // UNION: combine rows and remove duplicates
                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    let mut unique_rows = Vec::new();

                    // Add left rows (dedup)
                    for row in result_rows {
                        let hash = Self::hash_row(&row);
                        if seen.insert(hash) {
                            unique_rows.push(row);
                        }
                    }

                    // Add right rows (dedup)
                    for row in right_rows {
                        let hash = Self::hash_row(&row);
                        if seen.insert(hash) {
                            unique_rows.push(row);
                        }
                    }

                    result_rows = unique_rows;
                }
                SetOperationType::UnionAll => {
                    // UNION ALL: just concatenate (keep all duplicates)
                    result_rows.extend(right_rows);
                }
                SetOperationType::Intersect => {
                    // INTERSECT: keep only rows that exist in both (dedup)
                    let right_hashes: FxHashSet<u64> =
                        right_rows.iter().map(Self::hash_row).collect();

                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    result_rows.retain(|row| {
                        let hash = Self::hash_row(row);
                        right_hashes.contains(&hash) && seen.insert(hash)
                    });
                }
                SetOperationType::IntersectAll => {
                    // INTERSECT ALL: keep matching rows with multiplicity
                    let mut right_counts: rustc_hash::FxHashMap<u64, usize> =
                        rustc_hash::FxHashMap::default();
                    for row in &right_rows {
                        *right_counts.entry(Self::hash_row(row)).or_insert(0) += 1;
                    }

                    result_rows.retain(|row| {
                        let hash = Self::hash_row(row);
                        if let Some(count) = right_counts.get_mut(&hash) {
                            if *count > 0 {
                                *count -= 1;
                                return true;
                            }
                        }
                        false
                    });
                }
                SetOperationType::Except => {
                    // EXCEPT: keep left rows not in right (dedup)
                    let right_hashes: FxHashSet<u64> =
                        right_rows.iter().map(Self::hash_row).collect();

                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    result_rows.retain(|row| {
                        let hash = Self::hash_row(row);
                        !right_hashes.contains(&hash) && seen.insert(hash)
                    });
                }
                SetOperationType::ExceptAll => {
                    // EXCEPT ALL: remove matching rows with multiplicity
                    let mut right_counts: rustc_hash::FxHashMap<u64, usize> =
                        rustc_hash::FxHashMap::default();
                    for row in &right_rows {
                        *right_counts.entry(Self::hash_row(row)).or_insert(0) += 1;
                    }

                    result_rows.retain(|row| {
                        let hash = Self::hash_row(row);
                        if let Some(count) = right_counts.get_mut(&hash) {
                            if *count > 0 {
                                *count -= 1;
                                return false; // Remove this row
                            }
                        }
                        true // Keep this row
                    });
                }
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(columns, result_rows)))
    }

    /// Hash a row for set operations
    fn hash_row(row: &Row) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        for value in row.as_slice() {
            value.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Execute SELECT without FROM (expressions only)
    fn execute_expression_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Box<dyn QueryResult>, Vec<String>, bool)> {
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);

        // Check WHERE clause first - if it evaluates to false, return empty result
        if let Some(where_clause) = &stmt.where_clause {
            // Pre-process subqueries in WHERE clause (EXISTS, IN, ALL/ANY, scalar subqueries)
            let processed_where = self.process_where_subqueries(where_clause, ctx)?;
            let where_result = evaluator.evaluate(&processed_where)?;
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
            let value = evaluator.evaluate(col_expr)?;
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
        let select_columns: rustc_hash::FxHashSet<String> = stmt
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
                        );
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
                self.try_extract_pushable_conjuncts(where_expr, schema, Some(ctx))
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
            let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
            let limit = if let Some(ref limit_expr) = stmt.limit {
                match evaluator.evaluate(limit_expr) {
                    Ok(Value::Integer(l)) => l as usize,
                    Ok(Value::Float(f)) => f as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            let offset = if let Some(ref offset_expr) = stmt.offset {
                match evaluator.evaluate(offset_expr) {
                    Ok(Value::Integer(o)) => o as usize,
                    Ok(Value::Float(f)) => f as usize,
                    _ => 0,
                }
            } else {
                0
            };

            // Pass storage_expr to filter while scanning with limit
            let rows = table.collect_rows_with_limit(storage_expr.as_deref(), limit, offset)?;

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
        let table_row_count = table.row_count();
        let should_disable_streaming_for_cache =
            cache_eligible && table_row_count <= super::semantic_cache::DEFAULT_MAX_CACHED_ROWS;

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
                        // Create evaluator for the filter closure
                        let mut eval = Evaluator::new(&self.function_registry);
                        eval = eval.with_context(ctx);
                        eval.init_columns(&all_columns);
                        let where_clone = where_expr.clone();
                        let all_cols = all_columns.clone();

                        // Create a filter predicate using the evaluator
                        let predicate: Box<dyn Fn(&Row) -> bool + Send + Sync> =
                            Box::new(move |row: &Row| {
                                // Re-create evaluator for each call to avoid borrow issues
                                // This is a tradeoff: we sacrifice some performance for Send+Sync safety
                                let mut local_eval = Evaluator::with_defaults();
                                local_eval.init_columns(&all_cols);
                                local_eval.set_row_array(row);
                                local_eval.evaluate_bool(&where_clone).unwrap_or(false)
                            });

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

        let rows = if needs_memory_filter {
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

            // Prepare evaluator once for all rows (optimization)
            // For correlated subqueries, don't pre-process - we'll process per-row
            let processed_where = if let Some(where_expr) = where_to_use {
                if has_correlated {
                    // Keep original expression for correlated subqueries
                    Some(where_expr.clone())
                } else if has_subqueries {
                    // Pre-process uncorrelated subqueries once (reuse cached has_subqueries)
                    Some(self.process_where_subqueries(where_expr, ctx)?)
                } else {
                    Some(where_expr.clone())
                }
            } else {
                None
            };

            // Check if we can use the PARALLEL PATH:
            // For simple WHERE without subqueries, collect all rows first
            // then filter in parallel. This is much faster for large tables.
            let use_parallel_path = !has_correlated && !has_subqueries && processed_where.is_some();

            if use_parallel_path {
                let where_expr = processed_where.as_ref().unwrap();

                // Collect all rows first (sequential - storage layer limitation)
                let mut all_rows = Vec::new();
                while scanner.next() {
                    all_rows.push(scanner.take_row());
                }

                // Apply parallel filtering if we have enough rows
                if parallel_config.should_parallel_filter(all_rows.len()) {
                    parallel::parallel_filter(
                        all_rows,
                        where_expr,
                        &all_columns,
                        &self.function_registry,
                        &parallel_config,
                    )
                } else {
                    // Sequential filter for small datasets
                    let mut eval = Evaluator::new(&self.function_registry);
                    eval = eval.with_context(ctx);
                    eval.init_columns(&all_columns);

                    all_rows
                        .into_iter()
                        .filter(|row| {
                            eval.set_row_array(row);
                            eval.evaluate_bool(where_expr).unwrap_or(false)
                        })
                        .collect()
                }
            } else {
                // SEQUENTIAL PATH: For correlated subqueries or complex cases
                // Create evaluator once and reuse for all rows
                let mut eval = if processed_where.is_some() {
                    let mut e = Evaluator::new(&self.function_registry);
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
                let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
                    rustc_hash::FxHashMap::default();
                outer_row_map.reserve(base_capacity);

                // OPTIMIZATION: Wrap all_columns in Arc once, reuse for all rows (only if needed)
                let all_columns_arc: Option<Arc<Vec<String>>> = if has_correlated {
                    Some(Arc::new(all_columns.clone()))
                } else {
                    None
                };

                let mut rows = Vec::new();
                while scanner.next() {
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
                            let correlated_ctx = ctx.with_outer_row(
                                std::mem::take(&mut outer_row_map),
                                all_columns_arc.clone().unwrap(), // Arc clone = cheap
                            );

                            // Process the correlated subquery with outer row context
                            let processed =
                                self.process_correlated_where(where_expr, &correlated_ctx)?;

                            // OPTIMIZATION: Reuse outer evaluator, just update outer_row
                            evaluator.set_outer_row(correlated_ctx.outer_row());
                            evaluator.set_row_array(&row);

                            let result = evaluator.evaluate_bool(&processed)?;

                            // Take back the map for reuse
                            outer_row_map = correlated_ctx.outer_row.unwrap_or_default();

                            // Clear outer row from evaluator for next iteration
                            evaluator.clear_outer_row();

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
                }
                rows
            }
        } else if storage_expr.is_some() {
            // Path 2: WHERE clause with pushdown - use scanner for index optimization
            // Note: We fetch all columns here because downstream projection uses all_columns indices
            // Column pruning is handled by Path 3 (collect_projected_rows) when no WHERE clause
            let column_idx_vec: Vec<usize> = (0..all_columns.len()).collect();
            let mut scanner = table.scan(&column_idx_vec, storage_expr.as_deref())?;

            // OPTIMIZATION: Use take_row() to avoid cloning each row
            let mut rows = Vec::new();
            while scanner.next() {
                rows.push(scanner.take_row());
            }
            rows
        } else {
            // Path 3: Full scan without WHERE - use collect_all_rows
            // Note: collect_projected_rows was SLOWER due to extra allocations per row
            // Projection is handled later by the executor which is more efficient
            table.collect_all_rows(None)?
        };

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
            let result =
                self.execute_select_with_window_functions(stmt, ctx, rows, &all_columns)?;
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
        // Execute left side
        let (left_result, left_columns) = self.execute_table_expression(&join_source.left, ctx)?;

        // Execute right side
        let (right_result, right_columns) =
            self.execute_table_expression(&join_source.right, ctx)?;

        // Materialize both sides
        let left_rows = Self::materialize_result(left_result)?;
        let right_rows = Self::materialize_result(right_result)?;

        // Combine column names (qualified with table aliases)
        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.clone());

        // Determine join type
        let join_type = join_source.join_type.to_uppercase();

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
                Self::extract_join_keys_and_residual(condition, &left_columns, &right_columns);

            // Use the query planner for cost-based join decision
            let has_equality_keys = !left_key_indices.is_empty();

            // Detect if inputs are sorted on join keys for potential merge join optimization
            // This check is O(n) but enables O(n+m) merge join vs O(n*m) nested loop
            let left_sorted =
                has_equality_keys && Self::is_sorted_on_keys(&left_rows, &left_key_indices);
            let right_sorted =
                has_equality_keys && Self::is_sorted_on_keys(&right_rows, &right_key_indices);

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

            // Select join algorithm based on final decision (planner or AQE-overridden)
            match final_algorithm {
                RuntimeJoinAlgorithm::HashJoin => {
                    // Hash Join: O(N + M) instead of O(N * M)
                    // Supports multiple keys: ON a.x = b.x AND a.y = b.y
                    // Use execute_hash_join which handles:
                    // - Build side optimization (smaller table as build)
                    // - Outer join restrictions (LEFT/RIGHT/FULL joins need specific build sides)
                    // - Bloom filter optimization
                    let mut rows = self.execute_hash_join(
                        &left_rows,
                        &right_rows,
                        &left_key_indices,
                        &right_key_indices,
                        &join_type,
                        left_columns.len(),
                        right_columns.len(),
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
                        );
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
                        );
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
                    )?
                }
            }
        } else {
            // CROSS JOIN - no condition
            self.execute_nested_loop_join(
                &left_rows,
                &right_rows,
                None,
                &all_columns,
                &left_columns,
                &right_columns,
                &join_type,
                ctx,
            )?
        };

        // Build alias map and substitute aliases in WHERE clause
        let alias_map = Self::build_alias_map(&stmt.columns);
        let resolved_where_clause = if !alias_map.is_empty() {
            stmt.where_clause
                .as_ref()
                .map(|where_expr| Box::new(Self::substitute_aliases(where_expr, &alias_map)))
        } else {
            stmt.where_clause.clone()
        };

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = resolved_where_clause {
            // Process subqueries (EXISTS, scalar subqueries) before evaluation
            let processed_where = if Self::has_subqueries(where_clause) {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Create evaluator once and reuse
            let mut where_eval = Evaluator::new(&self.function_registry);
            where_eval = where_eval.with_context(ctx);
            where_eval.init_columns(&all_columns);

            let mut rows = Vec::new();
            for row in result_rows {
                where_eval.set_row_array(&row);
                if where_eval.evaluate_bool(&processed_where)? {
                    rows.push(row);
                }
            }
            rows
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
                let excluded_set: std::collections::HashSet<usize> =
                    excluded_column_indices.iter().copied().collect();

                // Create rename map for join columns
                let rename_map: std::collections::HashMap<usize, String> =
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
        let result = self.execute_select(&subquery_source.subquery, ctx)?;
        let columns = result.columns().to_vec();

        // Materialize the subquery result
        let rows = Self::materialize_result(result)?;

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            let mut where_eval = Evaluator::new(&self.function_registry);
            where_eval = where_eval.with_context(ctx);
            where_eval.init_columns(&columns);

            let mut filtered = Vec::new();
            for row in rows {
                where_eval.set_row_array(&row);
                if where_eval.evaluate_bool(where_clause)? {
                    filtered.push(row);
                }
            }
            filtered
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
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);

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
        let mut col_index_map: rustc_hash::FxHashMap<String, usize> = column_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.to_lowercase(), i))
            .collect();

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

        // OPTIMIZATION: Pre-create evaluators outside the loop if WHERE clause exists
        let (mut where_eval, mut qualified_eval, _qualified_columns) =
            if stmt.where_clause.is_some() {
                let mut eval = Evaluator::new(&self.function_registry);
                eval = eval.with_context(ctx);
                eval.init_columns(&column_names);

                // Pre-compute qualified column names once
                let qualified_cols: Vec<String> = column_names
                    .iter()
                    .map(|c| format!("{}.{}", table_alias, c))
                    .collect();
                let mut qual_eval = Evaluator::new(&self.function_registry);
                qual_eval = qual_eval.with_context(ctx);
                qual_eval.init_columns(&qualified_cols);

                (Some(eval), Some(qual_eval), qualified_cols)
            } else {
                (None, None, Vec::new())
            };

        // Evaluate all rows
        let mut result_rows = Vec::with_capacity(values_source.rows.len());
        for row_exprs in &values_source.rows {
            let mut row_values = Vec::with_capacity(row_exprs.len());
            for expr in row_exprs {
                let value = evaluator.evaluate(expr)?;
                row_values.push(value);
            }
            let row = Row::from_values(row_values);

            // Apply WHERE clause filtering
            if let Some(ref where_clause) = stmt.where_clause {
                let eval = where_eval.as_mut().unwrap();
                eval.set_row_array(&row);

                // Try to evaluate, but handle qualified identifiers
                match eval.evaluate_bool(where_clause) {
                    Ok(true) => result_rows.push(row),
                    Ok(false) => {} // Skip this row
                    Err(_) => {
                        // Try with table-qualified column names
                        let qual_eval = qualified_eval.as_mut().unwrap();
                        qual_eval.set_row_array(&row);

                        if qual_eval.evaluate_bool(where_clause).unwrap_or(false) {
                            result_rows.push(row);
                        }
                    }
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
            let mut eval = Evaluator::new(&self.function_registry);
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
                                let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
                                    rustc_hash::FxHashMap::default();
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
                                let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
                                    rustc_hash::FxHashMap::default();
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
                    let result =
                        super::result::ExecutorMemoryResult::new(columns.clone(), rows.clone());
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

                // Create a simple SELECT * statement
                let select_all = SelectStatement {
                    token: dummy_token("SELECT", TokenType::Keyword),
                    distinct: false,
                    columns: vec![Expression::Star(StarExpression {
                        token: dummy_token("*", TokenType::Punctuator),
                    })],
                    with: None,
                    table_expr: Some(Box::new(Expression::TableSource(ts.clone()))),
                    where_clause: None,
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
                let result = self.execute_select(&ss.subquery, ctx)?;
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

        // Build column index map ONCE with FxHashMap for O(1) lookup
        // Use lowercase keys for case-insensitive matching
        // Note: We could use schema.column_index_map() if we had access to the table here,
        // but project_rows is also called for CTEs which don't have a schema
        let mut col_index_map_lower: rustc_hash::FxHashMap<String, usize> =
            rustc_hash::FxHashMap::default();
        for (i, c) in all_columns.iter().enumerate() {
            let lower = c.to_lowercase();
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
                    let prefix = format!("{}.", qs.qualifier);
                    let prefix_lower = prefix.to_lowercase();
                    let qualifier_lower = qs.qualifier.to_lowercase();
                    let mut found_any = false;
                    for (idx, col) in all_columns.iter().enumerate() {
                        if col.to_lowercase().starts_with(&prefix_lower) {
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
            let mut seen_indices: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
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
        let mut evaluator = Evaluator::new(&self.function_registry);
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
        let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
            rustc_hash::FxHashMap::default();
        if has_correlated_select {
            outer_row_map.reserve(base_capacity);
        }

        // OPTIMIZATION: Wrap all_columns in Arc once, reuse for all rows (only if needed)
        let all_columns_arc: Option<Arc<Vec<String>>> = if has_correlated_select {
            Some(Arc::new(all_columns.to_vec()))
        } else {
            None
        };

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
                        let prefix = format!("{}.", qs.qualifier);
                        let prefix_lower = prefix.to_lowercase();
                        let qualifier_lower = qs.qualifier.to_lowercase();
                        let mut found_any = false;
                        for (idx, col) in all_columns.iter().enumerate() {
                            if col.to_lowercase().starts_with(&prefix_lower) {
                                if let Some(val) = row.get(idx) {
                                    values.push(val.clone());
                                    found_any = true;
                                }
                            }
                        }
                        // If no columns matched the prefix (single-table query), check if
                        // the qualifier matches the table alias - if so, include all columns
                        if !found_any {
                            if let Some(alias) = table_alias {
                                if alias.to_lowercase() == qualifier_lower {
                                    for val in row.iter() {
                                        values.push(val.clone());
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        let value = self.evaluate_select_expr(
                            &evaluator,
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
        let col_index_map_lower: rustc_hash::FxHashMap<String, usize> = all_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_lowercase(), i))
            .collect();

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
            let mut evaluator = Evaluator::new(&self.function_registry);
            evaluator = evaluator.with_context(ctx);
            evaluator.init_columns(all_columns);

            // OPTIMIZATION: Reuse col_index_map_lower for O(1) lookup
            for row in rows {
                let mut values = Vec::with_capacity(select_exprs.len() + extra_order_indices.len());

                evaluator.set_row_array(&row);

                // Evaluate SELECT expressions
                for expr in select_exprs.iter() {
                    let value =
                        self.evaluate_select_expr(&evaluator, expr, &row, &col_index_map_lower)?;
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
        evaluator: &Evaluator,
        expr: &Expression,
        row: &Row,
        col_index_map: &rustc_hash::FxHashMap<String, usize>,
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

        let col_index_map: rustc_hash::FxHashMap<String, usize> = all_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_lowercase(), i))
            .collect();

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

    /// Resolve SELECT column list to actual column names and indices
    #[allow(dead_code)]
    fn resolve_select_columns(
        &self,
        stmt: &SelectStatement,
        all_columns: &[String],
        _schema: &Schema,
    ) -> Result<(Vec<String>, Vec<usize>)> {
        let mut output_columns = Vec::new();
        let mut column_indices = Vec::new();

        for (i, col_expr) in stmt.columns.iter().enumerate() {
            match col_expr {
                Expression::Star(_) | Expression::QualifiedStar(_) => {
                    output_columns.extend(all_columns.iter().cloned());
                    column_indices.clear();
                    return Ok((output_columns, column_indices));
                }
                Expression::Identifier(id) => {
                    // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocations
                    if let Some(idx) = all_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(&id.value_lower))
                    {
                        output_columns.push(all_columns[idx].clone());
                        column_indices.push(idx);
                    } else {
                        return Err(Error::ColumnNotFoundNamed(id.value.clone()));
                    }
                }
                Expression::Aliased(aliased) => {
                    let name = aliased.alias.value.clone();
                    output_columns.push(name);
                }
                _ => {
                    output_columns.push(format!("expr{}", i + 1));
                }
            }
        }

        Ok((output_columns, column_indices))
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
            tables: rustc_hash::FxHashMap::default(),
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

    /// Execute SHOW TABLES statement
    pub(crate) fn execute_show_tables(
        &self,
        _stmt: &ShowTablesStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let tx = self.engine.begin_transaction()?;
        let tables = tx.list_tables()?;

        let columns = vec!["table_name".to_string()];
        let rows: Vec<Row> = tables
            .into_iter()
            .map(|name| Row::from_values(vec![Value::Text(std::sync::Arc::from(name.as_str()))]))
            .collect();

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute SHOW VIEWS statement
    pub(crate) fn execute_show_views(
        &self,
        _stmt: &ShowViewsStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let views = self.engine.list_views()?;

        let columns = vec!["view_name".to_string()];
        let rows: Vec<Row> = views
            .into_iter()
            .map(|name| Row::from_values(vec![Value::Text(std::sync::Arc::from(name.as_str()))]))
            .collect();

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute SHOW CREATE TABLE statement
    pub(crate) fn execute_show_create_table(
        &self,
        stmt: &ShowCreateTableStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;
        let schema = table.schema();

        // Get unique column names from indexes
        let mut unique_columns: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        if let Ok(indexes) = self.engine.list_table_indexes(table_name) {
            for index_name in indexes.keys() {
                if let Some(index) = table.get_index(index_name) {
                    // Only single-column unique indexes should be shown as UNIQUE constraint on column
                    let col_names = index.column_names();
                    if index.is_unique() && col_names.len() == 1 {
                        unique_columns.insert(col_names[0].to_lowercase());
                    }
                }
            }
        }

        // Build CREATE TABLE statement
        let mut create_sql = format!("CREATE TABLE {} (", table_name);
        let col_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let mut def = format!("{} {:?}", col.name, col.data_type);
                if col.primary_key {
                    def.push_str(" PRIMARY KEY");
                    if col.auto_increment {
                        def.push_str(" AUTO_INCREMENT");
                    }
                } else {
                    // Check if this column has a UNIQUE constraint
                    if unique_columns.contains(&col.name.to_lowercase()) {
                        def.push_str(" UNIQUE");
                    }
                    if !col.nullable {
                        def.push_str(" NOT NULL");
                    }
                }
                // Add DEFAULT if present
                if let Some(default_expr) = &col.default_expr {
                    def.push_str(&format!(" DEFAULT {}", default_expr));
                }
                // Add CHECK constraint if present
                if let Some(check) = &col.check_expr {
                    def.push_str(&format!(" CHECK ({})", check));
                }
                def
            })
            .collect();
        create_sql.push_str(&col_defs.join(", "));
        create_sql.push(')');

        let columns = vec!["Table".to_string(), "Create Table".to_string()];
        let rows = vec![Row::from_values(vec![
            Value::Text(std::sync::Arc::from(table_name.as_str())),
            Value::Text(std::sync::Arc::from(create_sql.as_str())),
        ])];

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute SHOW CREATE VIEW statement
    pub(crate) fn execute_show_create_view(
        &self,
        stmt: &ShowCreateViewStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let view_name = &stmt.view_name.value;

        // Get the view definition
        let view_def = self
            .engine
            .get_view(view_name)?
            .ok_or_else(|| Error::ViewNotFound(view_name.to_string()))?;

        // Build CREATE VIEW statement
        let create_sql = format!(
            "CREATE VIEW {} AS {}",
            view_def.original_name, view_def.query
        );

        let columns = vec!["View".to_string(), "Create View".to_string()];
        let rows = vec![Row::from_values(vec![
            Value::Text(std::sync::Arc::from(view_def.original_name.as_str())),
            Value::Text(std::sync::Arc::from(create_sql.as_str())),
        ])];

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute SHOW INDEXES statement
    pub(crate) fn execute_show_indexes(
        &self,
        stmt: &ShowIndexesStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;

        // Get a table reference to access indexes
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;

        // Get index info from version store through the table
        let index_names = {
            let indexes = self.engine.list_table_indexes(table_name)?;
            indexes.keys().cloned().collect::<Vec<_>>()
        };

        // Build rows with: table_name, index_name, column_name, index_type, is_unique
        let columns = vec![
            "table_name".to_string(),
            "index_name".to_string(),
            "column_name".to_string(),
            "index_type".to_string(),
            "is_unique".to_string(),
        ];

        let mut rows: Vec<Row> = Vec::new();
        for index_name in index_names {
            // Try to get index details from the table's underlying storage
            if let Some(index) = table.get_index(&index_name) {
                let column_names = index.column_names();
                // Show all columns for multi-column indexes
                let column_name = if column_names.len() > 1 {
                    format!("({})", column_names.join(", "))
                } else {
                    column_names
                        .first()
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                };
                let is_unique = index.is_unique();
                let index_type = index.index_type().as_str().to_uppercase();

                rows.push(Row::from_values(vec![
                    Value::Text(std::sync::Arc::from(table_name.as_str())),
                    Value::Text(std::sync::Arc::from(index_name.as_str())),
                    Value::Text(std::sync::Arc::from(column_name.as_str())),
                    Value::Text(std::sync::Arc::from(index_type)),
                    Value::Boolean(is_unique),
                ]));
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute DESCRIBE statement - shows table structure
    pub(crate) fn execute_describe(
        &self,
        stmt: &DescribeStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;
        let schema = table.schema();

        // Column headers: Field, Type, Null, Key, Default, Extra
        let columns = vec![
            "Field".to_string(),
            "Type".to_string(),
            "Null".to_string(),
            "Key".to_string(),
            "Default".to_string(),
            "Extra".to_string(),
        ];

        let mut rows: Vec<Row> = Vec::new();
        for col in &schema.columns {
            // Determine type string
            let type_str = format!("{:?}", col.data_type);

            // Determine nullability
            let null_str = if col.nullable { "YES" } else { "NO" };

            // Determine key type
            let key_str = if col.primary_key { "PRI" } else { "" };

            // Get default value if any
            let default_str = col
                .default_expr
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default();

            // Extra info (e.g., auto_increment equivalent)
            let extra_str =
                if col.primary_key && col.data_type == crate::core::types::DataType::Integer {
                    "auto_increment"
                } else {
                    ""
                };

            rows.push(Row::from_values(vec![
                Value::Text(std::sync::Arc::from(col.name.as_str())),
                Value::Text(std::sync::Arc::from(type_str.as_str())),
                Value::Text(std::sync::Arc::from(null_str)),
                Value::Text(std::sync::Arc::from(key_str)),
                Value::Text(std::sync::Arc::from(default_str.as_str())),
                Value::Text(std::sync::Arc::from(extra_str)),
            ]));
        }

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Execute EXPLAIN statement - shows query plan
    pub(crate) fn execute_explain(
        &self,
        stmt: &ExplainStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut plan_lines: Vec<String> = Vec::new();

        if stmt.analyze {
            // EXPLAIN ANALYZE: Execute the query and collect statistics
            let start = std::time::Instant::now();
            let mut result = self.execute_statement(&stmt.statement, ctx)?;

            // Count rows by iterating through the result
            let mut row_count = 0usize;
            while result.next() {
                row_count += 1;
            }
            let duration = start.elapsed();

            // Format duration nicely
            let time_str = if duration.as_secs() > 0 {
                format!("{:.2}s", duration.as_secs_f64())
            } else if duration.as_millis() > 0 {
                format!(
                    "{:.2}ms",
                    duration.as_millis() as f64 + (duration.as_micros() % 1000) as f64 / 1000.0
                )
            } else {
                format!("{:.2}µs", duration.as_micros() as f64)
            };

            // Record cardinality feedback for SELECT statements with WHERE
            if let Statement::Select(select) = &*stmt.statement {
                if let Some(ref where_clause) = select.where_clause {
                    // Try to get the table name and record feedback
                    if let Some(ref table_expr) = select.table_expr {
                        if let Some(table_name) = Self::extract_table_name(table_expr) {
                            // Compute predicate fingerprint
                            let predicate_hash = fingerprint_predicate(&table_name, where_clause);

                            // Get estimated row count from planner
                            let tx = self.engine.begin_transaction().ok();
                            let estimated_rows = tx
                                .as_ref()
                                .and_then(|tx| tx.get_table(&table_name).ok())
                                .map(|table| {
                                    let stats = self
                                        .get_query_planner()
                                        .get_table_stats_with_fallback(&*table);
                                    stats.row_count as usize
                                })
                                .unwrap_or(row_count);

                            // Record feedback to global cache
                            global_feedback_cache().record_feedback(
                                &table_name,
                                predicate_hash,
                                None, // column_name for more granular tracking
                                estimated_rows as u64,
                                row_count as u64,
                            );
                        }
                    }
                }
            }

            // Generate plan with actual statistics
            self.explain_statement_with_stats(
                &stmt.statement,
                &mut plan_lines,
                0,
                row_count,
                &time_str,
            );

            // Return the plan as a result
            let columns = vec!["plan".to_string()];
            let rows: Vec<Row> = plan_lines
                .into_iter()
                .map(|line| {
                    Row::from_values(vec![Value::Text(std::sync::Arc::from(line.as_str()))])
                })
                .collect();

            Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
        } else {
            // Regular EXPLAIN: Just show the plan without executing
            self.explain_statement(&stmt.statement, &mut plan_lines, 0);

            // Return as a single-column result
            let columns = vec!["plan".to_string()];
            let rows: Vec<Row> = plan_lines
                .into_iter()
                .map(|line| {
                    Row::from_values(vec![Value::Text(std::sync::Arc::from(line.as_str()))])
                })
                .collect();

            Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
        }
    }

    /// Generate EXPLAIN output with actual execution statistics
    fn explain_statement_with_stats(
        &self,
        stmt: &Statement,
        lines: &mut Vec<String>,
        indent: usize,
        row_count: usize,
        time_str: &str,
    ) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                lines.push(format!(
                    "{}SELECT (actual time={}, rows={})",
                    prefix, time_str, row_count
                ));
                self.explain_select_columns(select, lines, indent);

                // FROM clause with access plan
                if let Some(ref table_expr) = select.table_expr {
                    self.explain_table_expr_with_where_and_stats(
                        table_expr,
                        select.where_clause.as_deref(),
                        lines,
                        indent + 1,
                        row_count,
                    );
                }

                // GROUP BY
                if !select.group_by.columns.is_empty() {
                    let groups: Vec<String> = select
                        .group_by
                        .columns
                        .iter()
                        .map(|g| format!("{}", g))
                        .collect();
                    lines.push(format!("{}  Group: {}", prefix, groups.join(", ")));
                }

                // HAVING
                if let Some(ref having) = select.having {
                    lines.push(format!("{}  Having: {}", prefix, having));
                }

                // ORDER BY
                if !select.order_by.is_empty() {
                    let orders: Vec<String> = select
                        .order_by
                        .iter()
                        .map(|o| {
                            let dir = if !o.ascending { " DESC" } else { "" };
                            format!("{}{}", o.expression, dir)
                        })
                        .collect();
                    lines.push(format!("{}  Order: {}", prefix, orders.join(", ")));
                }

                // LIMIT/OFFSET
                if let Some(ref limit) = select.limit {
                    lines.push(format!("{}  Limit: {}", prefix, limit));
                }
                if let Some(ref offset) = select.offset {
                    lines.push(format!("{}  Offset: {}", prefix, offset));
                }
            }
            Statement::Insert(insert) => {
                lines.push(format!(
                    "{}INSERT INTO {} (actual time={}, rows={})",
                    prefix, insert.table_name, time_str, row_count
                ));
                if let Some(ref select) = insert.select {
                    lines.push(format!("{}  Source:", prefix));
                    self.explain_select(select, lines, indent + 2);
                } else {
                    lines.push(format!(
                        "{}  Values: {} row(s)",
                        prefix,
                        insert.values.len()
                    ));
                }
            }
            Statement::Update(update) => {
                lines.push(format!(
                    "{}UPDATE {} (actual time={}, rows={})",
                    prefix, update.table_name, time_str, row_count
                ));
                lines.push(format!(
                    "{}  Set: {} column(s)",
                    prefix,
                    update.updates.len()
                ));
                if let Some(ref where_clause) = update.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            Statement::Delete(delete) => {
                lines.push(format!(
                    "{}DELETE FROM {} (actual time={}, rows={})",
                    prefix, delete.table_name, time_str, row_count
                ));
                if let Some(ref where_clause) = delete.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            _ => {
                lines.push(format!(
                    "{}Statement: {} (actual time={}, rows={})",
                    prefix, stmt, time_str, row_count
                ));
            }
        }
    }

    /// Helper to show just the SELECT columns
    fn explain_select_columns(
        &self,
        select: &SelectStatement,
        lines: &mut Vec<String>,
        indent: usize,
    ) {
        let prefix = "  ".repeat(indent);

        // Show columns
        let col_count = select.columns.len();
        if col_count <= 5 {
            let cols: Vec<String> = select.columns.iter().map(|c| format!("{}", c)).collect();
            lines.push(format!("{}  Columns: {}", prefix, cols.join(", ")));
        } else {
            lines.push(format!("{}  Columns: {} column(s)", prefix, col_count));
        }
    }

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis and stats
    fn explain_table_expr_with_where_and_stats(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        row_count: usize,
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.engine.begin_transaction() {
                    if let Ok(table) = tx.get_table(&simple.name.value) {
                        // Build storage expression from WHERE clause for analysis
                        let storage_expr = if let Some(where_expr) = where_clause {
                            let schema = table.schema();
                            self.build_storage_expression_with_ctx(where_expr, schema, None)
                                .ok()
                        } else {
                            None
                        };

                        // Get the scan plan
                        let scan_plan = table.explain_scan(storage_expr.as_deref());

                        // For SeqScan, use the AST expression's Display format instead of storage expr Debug
                        // Check if parallel execution would be used based on TABLE's row count (not output rows)
                        // Parallel decision is based on input size, not filtered output
                        let parallel_config = parallel::ParallelConfig::default();
                        let table_row_count = table.row_count();
                        let would_use_parallel = where_clause.is_some()
                            && parallel_config.should_parallel_filter(table_row_count);

                        let scan_plan = match scan_plan {
                            ScanPlan::SeqScan {
                                table: tbl,
                                filter: _,
                            } if where_clause.is_some() => {
                                let filter_str = Some(format!("{}", where_clause.unwrap()));
                                if would_use_parallel {
                                    ScanPlan::ParallelSeqScan {
                                        table: tbl,
                                        filter: filter_str,
                                        workers: rayon::current_num_threads(),
                                    }
                                } else {
                                    ScanPlan::SeqScan {
                                        table: tbl,
                                        filter: filter_str,
                                    }
                                }
                            }
                            other => other,
                        };

                        // Format the scan plan with actual stats
                        let plan_str = format!("{}", scan_plan);
                        for (i, line) in plan_str.lines().enumerate() {
                            if i == 0 {
                                lines.push(format!(
                                    "{}-> {} (actual rows={})",
                                    prefix, line, row_count
                                ));
                            } else {
                                lines.push(format!("{}   {}", prefix, line));
                            }
                        }

                        // Add alias if present
                        if let Some(ref alias) = simple.alias {
                            lines.push(format!("{}   Alias: {}", prefix, alias));
                        }

                        return;
                    }
                }

                // Fallback if table not found
                let mut table_info = format!(
                    "{}-> Seq Scan on {} (actual rows={})",
                    prefix, simple.name, row_count
                );
                if let Some(ref alias) = simple.alias {
                    table_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(table_info);
                if let Some(ref where_expr) = where_clause {
                    lines.push(format!("{}   Filter: {}", prefix, where_expr));
                }
            }
            Expression::SubquerySource(subquery) => {
                let mut sub_info =
                    format!("{}-> Subquery Scan (actual rows={})", prefix, row_count);
                if let Some(ref alias) = subquery.alias {
                    sub_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(sub_info);
                self.explain_select(&subquery.subquery, lines, indent + 1);
            }
            Expression::JoinSource(join) => {
                // Determine join algorithm based on condition
                let join_algorithm = if join.condition.is_none() && join.using_columns.is_empty() {
                    "Nested Loop"
                } else if let Some(ref cond) = join.condition {
                    if Self::is_equality_condition(cond) {
                        "Hash Join"
                    } else {
                        "Nested Loop"
                    }
                } else {
                    "Hash Join" // USING clause implies equality
                };

                lines.push(format!(
                    "{}-> {} ({} Join) (actual rows={})",
                    prefix, join_algorithm, join.join_type, row_count
                ));
                if let Some(ref condition) = join.condition {
                    lines.push(format!("{}   Join Cond: {}", prefix, condition));
                }
                if !join.using_columns.is_empty() {
                    let cols: Vec<String> =
                        join.using_columns.iter().map(|c| c.to_string()).collect();
                    lines.push(format!("{}   Using: ({})", prefix, cols.join(", ")));
                }
                // Left side gets the WHERE clause for potential pushdown
                self.explain_table_expr_with_where(&join.left, where_clause, lines, indent + 1);
                // Right side typically doesn't get the outer WHERE
                self.explain_table_expr_with_where(&join.right, None, lines, indent + 1);
            }
            Expression::CteReference(cte_ref) => {
                let mut cte_info = format!(
                    "{}-> CTE Scan on {} (actual rows={})",
                    prefix, cte_ref.name, row_count
                );
                if let Some(ref alias) = cte_ref.alias {
                    cte_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(cte_info);
            }
            _ => {
                lines.push(format!(
                    "{}-> Scan: {} (actual rows={})",
                    prefix, expr, row_count
                ));
            }
        }
    }

    /// Generate EXPLAIN output for a statement
    fn explain_statement(&self, stmt: &Statement, lines: &mut Vec<String>, indent: usize) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                self.explain_select(select, lines, indent);
            }
            Statement::Insert(insert) => {
                lines.push(format!("{}INSERT INTO {}", prefix, insert.table_name));
                if let Some(ref select) = insert.select {
                    lines.push(format!("{}  Source:", prefix));
                    self.explain_select(select, lines, indent + 2);
                } else {
                    lines.push(format!(
                        "{}  Values: {} row(s)",
                        prefix,
                        insert.values.len()
                    ));
                }
            }
            Statement::Update(update) => {
                lines.push(format!("{}UPDATE {}", prefix, update.table_name));
                lines.push(format!(
                    "{}  Set: {} column(s)",
                    prefix,
                    update.updates.len()
                ));
                if let Some(ref where_clause) = update.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            Statement::Delete(delete) => {
                lines.push(format!("{}DELETE FROM {}", prefix, delete.table_name));
                if let Some(ref where_clause) = delete.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            _ => {
                lines.push(format!("{}Statement: {}", prefix, stmt));
            }
        }
    }

    /// Generate EXPLAIN output for a SELECT statement
    fn explain_select(&self, select: &SelectStatement, lines: &mut Vec<String>, indent: usize) {
        let prefix = "  ".repeat(indent);

        // CTE info
        if let Some(ref with) = select.with {
            lines.push(format!("{}WITH (CTEs: {})", prefix, with.ctes.len()));
            for cte in &with.ctes {
                lines.push(format!(
                    "{}  {} = ({})",
                    prefix,
                    cte.name,
                    if cte.is_recursive {
                        "RECURSIVE"
                    } else {
                        "non-recursive"
                    }
                ));
            }
        }

        // Main operation
        if select.distinct {
            lines.push(format!("{}SELECT DISTINCT", prefix));
        } else {
            lines.push(format!("{}SELECT", prefix));
        }

        // Columns
        let col_count = select.columns.len();
        if col_count <= 3 {
            let cols: Vec<String> = select.columns.iter().map(|c| format!("{}", c)).collect();
            lines.push(format!("{}  Columns: {}", prefix, cols.join(", ")));
        } else {
            lines.push(format!("{}  Columns: {} column(s)", prefix, col_count));
        }

        // FROM clause with access plan
        if let Some(ref table_expr) = select.table_expr {
            self.explain_table_expr_with_where(
                table_expr,
                select.where_clause.as_deref(),
                lines,
                indent + 1,
            );
        }

        // GROUP BY
        if !select.group_by.columns.is_empty() {
            let groups: Vec<String> = select
                .group_by
                .columns
                .iter()
                .map(|g| format!("{}", g))
                .collect();
            lines.push(format!("{}  Group By: {}", prefix, groups.join(", ")));
        }

        // HAVING
        if let Some(ref having) = select.having {
            lines.push(format!("{}  Having: {}", prefix, having));
        }

        // ORDER BY
        if !select.order_by.is_empty() {
            let orders: Vec<String> = select.order_by.iter().map(|o| format!("{}", o)).collect();
            lines.push(format!("{}  Order By: {}", prefix, orders.join(", ")));
        }

        // LIMIT/OFFSET
        if let Some(ref limit) = select.limit {
            lines.push(format!("{}  Limit: {}", prefix, limit));
        }
        if let Some(ref offset) = select.offset {
            lines.push(format!("{}  Offset: {}", prefix, offset));
        }

        // Set operations
        if !select.set_operations.is_empty() {
            for set_op in &select.set_operations {
                lines.push(format!("{}  {}", prefix, set_op.operation));
                self.explain_select(&set_op.right, lines, indent + 2);
            }
        }
    }

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis
    fn explain_table_expr_with_where(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.engine.begin_transaction() {
                    if let Ok(table) = tx.get_table(&simple.name.value) {
                        // Build storage expression from WHERE clause for analysis
                        let storage_expr = if let Some(where_expr) = where_clause {
                            let schema = table.schema();
                            self.build_storage_expression_with_ctx(where_expr, schema, None)
                                .ok()
                        } else {
                            None
                        };

                        // Get the scan plan
                        let scan_plan = table.explain_scan(storage_expr.as_deref());

                        // For SeqScan, use the AST expression's Display format instead of storage expr Debug
                        let scan_plan = match scan_plan {
                            ScanPlan::SeqScan { table, filter: _ } if where_clause.is_some() => {
                                ScanPlan::SeqScan {
                                    table,
                                    filter: Some(format!("{}", where_clause.unwrap())),
                                }
                            }
                            other => other,
                        };

                        // Format the scan plan with indentation
                        let plan_str = format!("{}", scan_plan);
                        for (i, line) in plan_str.lines().enumerate() {
                            if i == 0 {
                                lines.push(format!("{}-> {}", prefix, line));
                            } else {
                                lines.push(format!("{}   {}", prefix, line));
                            }
                        }

                        // Add alias if present
                        if let Some(ref alias) = simple.alias {
                            lines.push(format!("{}   Alias: {}", prefix, alias));
                        }

                        return;
                    }
                }

                // Fallback if table not found
                let mut table_info = format!("{}-> Seq Scan on {}", prefix, simple.name);
                if let Some(ref alias) = simple.alias {
                    table_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(table_info);
                if let Some(ref where_expr) = where_clause {
                    lines.push(format!("{}   Filter: {}", prefix, where_expr));
                }
            }
            Expression::SubquerySource(subquery) => {
                let mut sub_info = format!("{}-> Subquery Scan", prefix);
                if let Some(ref alias) = subquery.alias {
                    sub_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(sub_info);
                self.explain_select(&subquery.subquery, lines, indent + 1);
            }
            Expression::JoinSource(join) => {
                // Determine join algorithm based on condition
                let join_algorithm = if join.condition.is_none() && join.using_columns.is_empty() {
                    // CROSS JOIN or no condition -> Nested Loop
                    "Nested Loop"
                } else if let Some(ref cond) = join.condition {
                    // Check if it's an equality join (a.col = b.col)
                    if Self::is_equality_condition(cond) {
                        "Hash Join"
                    } else {
                        // Range condition or complex join
                        "Nested Loop"
                    }
                } else {
                    // USING clause implies equality join
                    "Hash Join"
                };

                // Get cost estimate from query planner
                let planner = self.get_query_planner();
                let left_table_name = Self::extract_table_name(&join.left);
                let right_table_name = Self::extract_table_name(&join.right);

                // Get table statistics for cost estimation
                let left_stats = left_table_name
                    .as_ref()
                    .and_then(|name| planner.get_table_stats(name));
                let right_stats = right_table_name
                    .as_ref()
                    .and_then(|name| planner.get_table_stats(name));

                // Calculate estimated rows and cost
                let (estimated_rows, estimated_cost) = match (left_stats, right_stats) {
                    (Some(ls), Some(rs)) => {
                        // Hash join cost estimation
                        let left_rows = ls.row_count.max(1);
                        let right_rows = rs.row_count.max(1);
                        // Simplified join cardinality estimate
                        let rows = if join_algorithm == "Nested Loop" && join.condition.is_none() {
                            // Cross join: left * right
                            left_rows * right_rows
                        } else {
                            // Equality join: estimate as smaller side (pessimistic)
                            left_rows.min(right_rows)
                        };
                        // Cost = build cost + probe cost
                        let cost = if join_algorithm == "Hash Join" {
                            (left_rows.min(right_rows) as f64)
                                + (left_rows.max(right_rows) as f64 * 0.1)
                        } else {
                            // Nested loop: O(n*m) but with early termination
                            (left_rows as f64) * (right_rows as f64).sqrt()
                        };
                        (rows, cost)
                    }
                    (Some(ls), None) => {
                        // Only left stats available
                        (ls.row_count, ls.row_count as f64 * 10.0)
                    }
                    (None, Some(rs)) => {
                        // Only right stats available
                        (rs.row_count, rs.row_count as f64 * 10.0)
                    }
                    (None, None) => {
                        // No stats - use default estimate
                        (1000, 10000.0)
                    }
                };

                // Show join algorithm, type, cost and rows
                lines.push(format!(
                    "{}-> {} ({} Join) (cost={:.2} rows={})",
                    prefix, join_algorithm, join.join_type, estimated_cost, estimated_rows
                ));
                if let Some(ref condition) = join.condition {
                    lines.push(format!("{}   Join Cond: {}", prefix, condition));
                }
                if !join.using_columns.is_empty() {
                    let cols: Vec<String> =
                        join.using_columns.iter().map(|c| c.to_string()).collect();
                    lines.push(format!("{}   Using: ({})", prefix, cols.join(", ")));
                }
                // Left side gets the WHERE clause for potential pushdown
                self.explain_table_expr_with_where(&join.left, where_clause, lines, indent + 1);
                // Right side typically doesn't get the outer WHERE
                self.explain_table_expr_with_where(&join.right, None, lines, indent + 1);
            }
            Expression::CteReference(cte_ref) => {
                let mut cte_info = format!("{}-> CTE Scan on {}", prefix, cte_ref.name);
                if let Some(ref alias) = cte_ref.alias {
                    cte_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(cte_info);
            }
            _ => {
                lines.push(format!("{}-> Scan: {}", prefix, expr));
            }
        }
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
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);

        let value = evaluator.evaluate(&stmt.expression)?;
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
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
        let limit = if let Some(ref limit_expr) = stmt.limit {
            match evaluator.evaluate(limit_expr) {
                Ok(Value::Integer(l)) => l as usize,
                Ok(Value::Float(f)) => f as usize,
                _ => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        let offset = if let Some(ref offset_expr) = stmt.offset {
            match evaluator.evaluate(offset_expr) {
                Ok(Value::Integer(o)) => o as usize,
                Ok(Value::Float(f)) => f as usize,
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
        // Note: .ok() converts Result to Option, discarding errors for complex expressions
        let storage_expr = stmt.where_clause.as_ref().and_then(|where_expr| {
            self.build_storage_expression_with_ctx(where_expr, &schema, Some(ctx))
                .ok()
        });

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

        // Apply in-memory WHERE filter if storage expression couldn't handle it
        if stmt.where_clause.is_some() && storage_expr.is_none() {
            let mut evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
            evaluator.init_columns(&all_columns);

            let where_expr = stmt.where_clause.as_ref().unwrap();
            rows.retain(|row| {
                evaluator.set_row_array(row);
                evaluator.evaluate_bool(where_expr).unwrap_or(false)
            });
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
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);

        match as_of.as_of_type.to_uppercase().as_str() {
            "TRANSACTION" => {
                // Expect integer transaction ID
                match evaluator.evaluate(&as_of.value)? {
                    Value::Integer(txn_id) => Ok(txn_id),
                    _ => Err(Error::invalid_argument(
                        "AS OF TRANSACTION requires integer value",
                    )),
                }
            }
            "TIMESTAMP" => {
                // Expect timestamp string or timestamp value
                match evaluator.evaluate(&as_of.value)? {
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

    // ========================================================================
    // Hash Join Implementation - O(N + M) instead of O(N * M)
    // ========================================================================

    /// Extract equality join keys and residual conditions
    /// Returns (left_indices, right_indices, residual_conditions) where residual
    /// contains non-equality conditions that must be applied after the hash join
    fn extract_join_keys_and_residual(
        condition: &Expression,
        left_columns: &[String],
        right_columns: &[String],
    ) -> (Vec<usize>, Vec<usize>, Vec<Expression>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        let mut residual = Vec::new();

        Self::extract_join_keys_recursive(
            condition,
            left_columns,
            right_columns,
            &mut left_indices,
            &mut right_indices,
            &mut residual,
        );

        (left_indices, right_indices, residual)
    }

    /// Recursively extract equality join keys and residual conditions from AND expressions
    fn extract_join_keys_recursive(
        condition: &Expression,
        left_columns: &[String],
        right_columns: &[String],
        left_indices: &mut Vec<usize>,
        right_indices: &mut Vec<usize>,
        residual: &mut Vec<Expression>,
    ) {
        match condition {
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Recurse into AND branches
                Self::extract_join_keys_recursive(
                    &infix.left,
                    left_columns,
                    right_columns,
                    left_indices,
                    right_indices,
                    residual,
                );
                Self::extract_join_keys_recursive(
                    &infix.right,
                    left_columns,
                    right_columns,
                    left_indices,
                    right_indices,
                    residual,
                );
            }
            Expression::Infix(infix) if infix.op_type == InfixOperator::Equal => {
                // Extract equality condition
                if let (Some(left_col), Some(right_col)) = (
                    Self::extract_join_column_name(&infix.left),
                    Self::extract_join_column_name(&infix.right),
                ) {
                    // Case 1: left.col = right.col
                    if let (Some(left_idx), Some(right_idx)) = (
                        Self::find_column_index(&left_col, left_columns),
                        Self::find_column_index(&right_col, right_columns),
                    ) {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                        return;
                    }

                    // Case 2: right.col = left.col (swapped)
                    if let (Some(left_idx), Some(right_idx)) = (
                        Self::find_column_index(&right_col, left_columns),
                        Self::find_column_index(&left_col, right_columns),
                    ) {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                        return;
                    }
                }
                // Non-join equality (e.g., a.x = 5) - add to residual
                residual.push(condition.clone());
            }
            _ => {
                // Non-equality condition - add to residual filters
                residual.push(condition.clone());
            }
        }
    }

    /// Extract column name from identifier expression (for hash join key extraction)
    /// Returns (qualifier, column_name) where qualifier is Some for qualified identifiers
    fn extract_join_column_name(expr: &Expression) -> Option<(Option<String>, String)> {
        match expr {
            Expression::Identifier(id) => Some((None, id.value_lower.clone())),
            Expression::QualifiedIdentifier(qid) => Some((
                Some(qid.qualifier.value_lower.clone()),
                qid.name.value_lower.clone(),
            )),
            _ => None,
        }
    }

    /// Find column index by name (case-insensitive)
    /// Takes (qualifier, column_name) tuple to properly match qualified columns
    /// OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocations per column
    fn find_column_index(col_info: &(Option<String>, String), columns: &[String]) -> Option<usize> {
        let (qualifier, col_name) = col_info;

        columns.iter().position(|c| {
            if let Some(ref qual) = qualifier {
                // We have a qualified name like "t2.id" - match the full qualified name
                let expected = format!("{}.{}", qual, col_name);
                c.eq_ignore_ascii_case(&expected)
            } else {
                // Unqualified name - match either full name or just the column part
                if c.eq_ignore_ascii_case(col_name) {
                    return true;
                }
                // Match qualified name like "table.column" -> match "column" part
                if let Some(dot_pos) = c.rfind('.') {
                    c[dot_pos + 1..].eq_ignore_ascii_case(col_name)
                } else {
                    false
                }
            }
        })
    }

    /// Apply residual conditions to join result rows
    ///
    /// For OUTER JOINs, we need special handling: rows with NULLs from the
    /// padded side should be preserved even if residual conditions fail.
    #[allow(clippy::too_many_arguments)]
    fn apply_residual_conditions(
        &self,
        rows: &mut Vec<Row>,
        residual: &[Expression],
        all_columns: &[String],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
        ctx: &ExecutionContext,
    ) {
        let mut residual_eval = Evaluator::new(&self.function_registry);
        residual_eval = residual_eval.with_context(ctx);
        residual_eval.init_columns(all_columns);

        let is_outer_join =
            join_type.contains("LEFT") || join_type.contains("RIGHT") || join_type.contains("FULL");

        if is_outer_join {
            // For outer joins, we need to identify which rows are "padded"
            // (have NULLs from the non-matching side) and preserve them
            rows.retain(|row| {
                // Check if this is a NULL-padded row from an outer join
                let left_all_null =
                    (0..left_col_count).all(|i| row.get(i).map(|v| v.is_null()).unwrap_or(true));
                let right_all_null = (left_col_count..left_col_count + right_col_count)
                    .all(|i| row.get(i).map(|v| v.is_null()).unwrap_or(true));

                // Preserve NULL-padded outer join rows unconditionally
                // (they already represent "no match" which is correct)
                if (join_type.contains("LEFT") && right_all_null)
                    || (join_type.contains("RIGHT") && left_all_null)
                    || (join_type.contains("FULL") && (left_all_null || right_all_null))
                {
                    return true;
                }

                // For matched rows, apply residual conditions normally
                residual_eval.set_row_array(row);
                residual
                    .iter()
                    .all(|cond| residual_eval.evaluate_bool(cond).unwrap_or(false))
            });
        } else {
            // For INNER JOIN, simple filter is correct
            rows.retain(|row| {
                residual_eval.set_row_array(row);
                residual
                    .iter()
                    .all(|cond| residual_eval.evaluate_bool(cond).unwrap_or(false))
            });
        }
    }

    /// Execute hash join - O(N + M) complexity
    /// Supports multiple join keys for conditions like: a.x = b.x AND a.y = b.y
    /// Build phase: Create hash table from smaller side (optimization)
    /// Probe phase: Scan larger side and probe hash table
    ///
    /// Uses parallel execution when either side has >= 5000 rows.
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        left_key_indices: &[usize],
        right_key_indices: &[usize],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
    ) -> Result<Vec<Row>> {
        // Optimization: Build hash table on smaller side
        // For LEFT/FULL joins, we must build on right side to track unmatched rows correctly
        // For RIGHT joins, we must build on left side
        let build_on_left = !join_type.contains("LEFT")
            && !join_type.contains("FULL")
            && (join_type.contains("RIGHT") || left_rows.len() < right_rows.len());

        // Check if parallel execution would be beneficial
        let parallel_config = ParallelConfig::default();
        let use_parallel = parallel_config.should_parallel_join(left_rows.len())
            || parallel_config.should_parallel_join(right_rows.len());

        if use_parallel {
            // Use parallel hash join for large datasets
            let parallel_join_type = parallel::JoinType::from_str(join_type);

            let result = if build_on_left {
                parallel::parallel_hash_join(
                    right_rows,
                    left_rows,
                    right_key_indices,
                    left_key_indices,
                    parallel_join_type,
                    right_col_count,
                    left_col_count,
                    true, // swapped
                    &parallel_config,
                )
            } else {
                parallel::parallel_hash_join(
                    left_rows,
                    right_rows,
                    left_key_indices,
                    right_key_indices,
                    parallel_join_type,
                    left_col_count,
                    right_col_count,
                    false, // not swapped
                    &parallel_config,
                )
            };

            return Ok(result.rows);
        }

        // Fall back to sequential execution for small datasets
        if build_on_left {
            // Build on left, probe with right (swap roles)
            self.execute_hash_join_impl(
                right_rows,
                left_rows,
                right_key_indices,
                left_key_indices,
                join_type,
                right_col_count,
                left_col_count,
                true, // swapped = true
            )
        } else {
            // Build on right, probe with left (normal)
            self.execute_hash_join_impl(
                left_rows,
                right_rows,
                left_key_indices,
                right_key_indices,
                join_type,
                left_col_count,
                right_col_count,
                false, // swapped = false
            )
        }
    }

    /// Core hash join implementation
    /// `swapped` indicates if left/right were swapped for build side optimization
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join_impl(
        &self,
        probe_rows: &[Row],
        build_rows: &[Row],
        probe_key_indices: &[usize],
        build_key_indices: &[usize],
        join_type: &str,
        probe_col_count: usize,
        build_col_count: usize,
        swapped: bool,
    ) -> Result<Vec<Row>> {
        use rustc_hash::FxHashMap;

        // Build phase: Create hash table from build side
        // Key: composite hash of all join key values, Value: list of row indices
        let mut hash_table: FxHashMap<u64, Vec<usize>> = FxHashMap::default();

        // Build bloom filter for faster probe-side filtering when build side is large enough
        // Use adaptive FP rate based on observed bloom filter effectiveness
        use crate::optimizer::bloom::BloomEffectivenessTracker;
        let bloom_stats = BloomEffectivenessTracker::global();
        let use_bloom_filter = build_rows.len() >= BLOOM_FILTER_MIN_BUILD_SIZE;
        let mut bloom_filter = if use_bloom_filter {
            // Use adaptively tuned false positive rate based on historical effectiveness
            let fp_rate = bloom_stats.recommend_false_positive_rate();
            Some(BloomFilter::new(build_rows.len(), fp_rate))
        } else {
            None
        };

        for (idx, row) in build_rows.iter().enumerate() {
            let hash = Self::hash_composite_key(row, build_key_indices);
            hash_table.entry(hash).or_default().push(idx);

            // Insert into bloom filter if enabled
            if let Some(ref mut bf) = bloom_filter {
                bf.insert_raw_hash(hash);
            }
        }

        let mut result_rows = Vec::new();
        let mut build_matched = vec![false; build_rows.len()];

        // Probe phase: Scan probe rows and lookup in hash table
        for probe_row in probe_rows {
            let mut matched = false;
            let hash = Self::hash_composite_key(probe_row, probe_key_indices);

            // Bloom filter early rejection: if definitely not in build side, skip
            if let Some(ref bf) = bloom_filter {
                if !bf.might_contain_raw_hash(hash) {
                    // Record true negative for adaptive optimization
                    bloom_stats.record_true_negative();
                    // Definitely not in build side - handle OUTER join case and continue
                    let needs_null_row = if swapped {
                        join_type.contains("RIGHT") || join_type.contains("FULL")
                    } else {
                        join_type.contains("LEFT") || join_type.contains("FULL")
                    };

                    if needs_null_row {
                        let values = if swapped {
                            Self::combine_rows_with_nulls(
                                probe_row,
                                probe_col_count,
                                build_col_count,
                                false,
                            )
                        } else {
                            Self::combine_rows_with_nulls(
                                probe_row,
                                probe_col_count,
                                build_col_count,
                                true,
                            )
                        };
                        result_rows.push(Row::from_values(values));
                    }
                    continue;
                }
                // Bloom filter passed - record for stats (will be true positive or false positive)
                bloom_stats.record_filter_passed();
            }

            if let Some(build_indices) = hash_table.get(&hash) {
                for &build_idx in build_indices {
                    let build_row = &build_rows[build_idx];

                    // Verify actual equality for all keys (handle hash collisions)
                    if Self::verify_composite_key_equality(
                        probe_row,
                        build_row,
                        probe_key_indices,
                        build_key_indices,
                    ) {
                        matched = true;
                        build_matched[build_idx] = true;

                        // Build combined row (respect original left/right order)
                        let combined = if swapped {
                            Self::combine_rows(
                                build_row,
                                probe_row,
                                build_col_count,
                                probe_col_count,
                            )
                        } else {
                            Self::combine_rows(
                                probe_row,
                                build_row,
                                probe_col_count,
                                build_col_count,
                            )
                        };
                        result_rows.push(Row::from_values(combined));
                    }
                }
            }

            // Handle unmatched probe rows for OUTER joins
            if !matched {
                let needs_null_row = if swapped {
                    // Swapped: probe=right, build=left
                    // RIGHT JOIN needs unmatched right rows (probe side)
                    join_type.contains("RIGHT") || join_type.contains("FULL")
                } else {
                    // Normal: probe=left, build=right
                    // LEFT JOIN needs unmatched left rows (probe side)
                    join_type.contains("LEFT") || join_type.contains("FULL")
                };

                if needs_null_row {
                    let values = if swapped {
                        // Left side (build) is NULL, right side (probe) has values
                        Self::combine_rows_with_nulls(
                            probe_row,
                            probe_col_count,
                            build_col_count,
                            false, // right side has values
                        )
                    } else {
                        // Right side (build) is NULL, left side (probe) has values
                        Self::combine_rows_with_nulls(
                            probe_row,
                            probe_col_count,
                            build_col_count,
                            true, // left side has values
                        )
                    };
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        // Handle unmatched build rows for OUTER joins
        let needs_unmatched_build = if swapped {
            // Swapped: probe=right, build=left
            // LEFT JOIN needs unmatched left rows (build side)
            join_type.contains("LEFT") || join_type.contains("FULL")
        } else {
            // Normal: probe=left, build=right
            // RIGHT JOIN needs unmatched right rows (build side)
            join_type.contains("RIGHT") || join_type.contains("FULL")
        };

        if needs_unmatched_build {
            for (build_idx, was_matched) in build_matched.iter().enumerate() {
                if !was_matched {
                    let values = if swapped {
                        // Left side (build) has values, right side (probe) is NULL
                        Self::combine_rows_with_nulls(
                            &build_rows[build_idx],
                            build_col_count,
                            probe_col_count,
                            true, // left side has values
                        )
                    } else {
                        // Right side (build) has values, left side (probe) is NULL
                        Self::combine_rows_with_nulls(
                            &build_rows[build_idx],
                            build_col_count,
                            probe_col_count,
                            false, // right side has values
                        )
                    };
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Combine two rows into one
    /// OPTIMIZATION: Use extend_from_slice instead of iter().cloned() for efficiency
    fn combine_rows(left: &Row, right: &Row, left_count: usize, right_count: usize) -> Vec<Value> {
        let mut combined = Vec::with_capacity(left_count + right_count);
        combined.extend_from_slice(left.as_slice());
        combined.extend_from_slice(right.as_slice());
        combined
    }

    /// Combine a row with NULLs for the other side
    /// OPTIMIZATION: Use extend_from_slice and resize for efficiency
    fn combine_rows_with_nulls(
        row: &Row,
        row_count: usize,
        null_count: usize,
        row_is_left: bool,
    ) -> Vec<Value> {
        let mut values = Vec::with_capacity(row_count + null_count);
        if row_is_left {
            values.extend_from_slice(row.as_slice());
            values.resize(row_count + null_count, Value::null_unknown());
        } else {
            values.resize(null_count, Value::null_unknown());
            values.extend_from_slice(row.as_slice());
        }
        values
    }

    /// Hash multiple key columns into a single hash
    fn hash_composite_key(row: &Row, key_indices: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();

        for &idx in key_indices {
            if let Some(value) = row.get(idx) {
                Self::hash_value_into(value, &mut hasher);
            } else {
                // NULL marker
                0xDEADBEEFu64.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Hash a single value into an existing hasher
    fn hash_value_into<H: std::hash::Hasher>(value: &Value, hasher: &mut H) {
        use std::hash::Hash;
        match value {
            Value::Integer(i) => {
                1u8.hash(hasher);
                i.hash(hasher);
            }
            Value::Float(f) => {
                2u8.hash(hasher);
                f.to_bits().hash(hasher);
            }
            Value::Text(s) => {
                3u8.hash(hasher);
                s.hash(hasher);
            }
            Value::Boolean(b) => {
                4u8.hash(hasher);
                b.hash(hasher);
            }
            Value::Null(_) => {
                5u8.hash(hasher);
            }
            Value::Timestamp(ts) => {
                6u8.hash(hasher);
                ts.timestamp_nanos_opt().hash(hasher);
            }
            Value::Json(j) => {
                7u8.hash(hasher);
                j.hash(hasher);
            }
        }
    }

    /// Verify that all composite key columns match (handles hash collisions)
    fn verify_composite_key_equality(
        row1: &Row,
        row2: &Row,
        indices1: &[usize],
        indices2: &[usize],
    ) -> bool {
        debug_assert_eq!(indices1.len(), indices2.len());

        for (&idx1, &idx2) in indices1.iter().zip(indices2.iter()) {
            match (row1.get(idx1), row2.get(idx2)) {
                (Some(v1), Some(v2)) => {
                    if !Self::values_equal(v1, v2) {
                        return false;
                    }
                }
                (None, None) => {
                    // Both NULL - in SQL, NULL != NULL, so no match
                    return false;
                }
                _ => {
                    // One NULL, one not - no match
                    return false;
                }
            }
        }
        true
    }
    /// Compare two Values for equality
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
            (Value::Text(x), Value::Text(y)) => x == y,
            (Value::Boolean(x), Value::Boolean(y)) => x == y,
            (Value::Null(_), Value::Null(_)) => false, // NULL != NULL in SQL
            (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
            (Value::Json(x), Value::Json(y)) => x == y,
            // Cross-type comparisons - try numeric
            (Value::Integer(x), Value::Float(y)) | (Value::Float(y), Value::Integer(x)) => {
                (*x as f64 - y).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    /// Nested loop join fallback for complex conditions
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_nested_loop_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        condition: Option<&Expression>,
        all_columns: &[String],
        left_columns: &[String],
        right_columns: &[String],
        join_type: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Row>> {
        // Create evaluator once and reuse for all join condition checks
        let mut eval = if condition.is_some() {
            let mut e = Evaluator::new(&self.function_registry);
            e = e.with_context(ctx);
            e.init_columns(all_columns);
            Some(e)
        } else {
            None
        };

        let mut result_rows = Vec::new();
        let mut right_matched = vec![false; right_rows.len()];

        // Calculate total columns for result rows
        let left_col_count = left_columns.len();
        let right_col_count = right_columns.len();
        let total_cols = left_col_count + right_col_count;

        for left_row in left_rows {
            let mut matched = false;
            let left_slice = left_row.as_slice();

            for (right_idx, right_row) in right_rows.iter().enumerate() {
                let right_slice = right_row.as_slice();

                // Check join condition using ZERO-COPY evaluation
                // OPTIMIZATION: Use set_join_rows to avoid cloning values for condition check
                let matches = if let Some(cond) = condition {
                    if let Some(ref mut evaluator) = eval {
                        // Zero-copy: set both row pointers directly
                        evaluator.set_join_rows(left_row, right_row);
                        evaluator.evaluate_bool(cond)?
                    } else {
                        true
                    }
                } else {
                    true // CROSS JOIN
                };

                if matches {
                    matched = true;
                    right_matched[right_idx] = true;
                    // Only allocate result row when condition matches
                    // OPTIMIZATION: Use extend_from_slice instead of iter().cloned()
                    let mut values = Vec::with_capacity(total_cols);
                    values.extend_from_slice(left_slice);
                    values.extend_from_slice(right_slice);
                    result_rows.push(Row::from_values(values));
                }
            }

            // Handle LEFT OUTER JOIN
            if !matched && (join_type.contains("LEFT") || join_type.contains("FULL")) {
                // OPTIMIZATION: Use extend_from_slice and resize for efficiency
                let mut values = Vec::with_capacity(total_cols);
                values.extend_from_slice(left_slice);
                values.resize(total_cols, Value::null_unknown());
                result_rows.push(Row::from_values(values));
            }
        }

        // Handle RIGHT OUTER JOIN - now O(n) instead of O(n*m)
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            let left_col_count = left_columns.len();
            for (right_idx, was_matched) in right_matched.iter().enumerate() {
                if !was_matched {
                    // OPTIMIZATION: Use resize and extend_from_slice for efficiency
                    let mut values = Vec::with_capacity(total_cols);
                    values.resize(left_col_count, Value::null_unknown());
                    values.extend_from_slice(right_rows[right_idx].as_slice());
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Merge join implementation for pre-sorted inputs
    ///
    /// Merge join is optimal when both inputs are sorted on the join keys.
    /// Time complexity: O(N + M) where N and M are the input sizes.
    ///
    /// This implementation handles:
    /// - INNER, LEFT, RIGHT, and FULL OUTER joins
    /// - Duplicate key handling (many-to-many joins)
    /// - Pre-sorted inputs (no sorting needed)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_merge_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        left_key_indices: &[usize],
        right_key_indices: &[usize],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
    ) -> Result<Vec<Row>> {
        let total_cols = left_col_count + right_col_count;
        let mut result_rows = Vec::new();

        // Track matched rows for outer joins
        let mut left_matched = vec![false; left_rows.len()];
        let mut right_matched = vec![false; right_rows.len()];

        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left_rows.len() && right_idx < right_rows.len() {
            let left_row = &left_rows[left_idx];
            let right_row = &right_rows[right_idx];

            // Compare keys using the first key column (primary sort)
            let cmp = Self::compare_composite_keys(
                left_row,
                right_row,
                left_key_indices,
                right_key_indices,
            );

            match cmp {
                Ordering::Less => {
                    // Left key is smaller - advance left
                    left_idx += 1;
                }
                Ordering::Greater => {
                    // Right key is smaller - advance right
                    right_idx += 1;
                }
                Ordering::Equal => {
                    // Keys match - find all matching rows on both sides
                    // This handles duplicate keys (many-to-many joins)

                    // Find the range of left rows with the same key
                    let left_start = left_idx;
                    while left_idx < left_rows.len()
                        && Self::compare_composite_keys(
                            &left_rows[left_start],
                            &left_rows[left_idx],
                            left_key_indices,
                            left_key_indices,
                        ) == Ordering::Equal
                    {
                        left_idx += 1;
                    }

                    // Find the range of right rows with the same key
                    let right_start = right_idx;
                    while right_idx < right_rows.len()
                        && Self::compare_composite_keys(
                            &right_rows[right_start],
                            &right_rows[right_idx],
                            right_key_indices,
                            right_key_indices,
                        ) == Ordering::Equal
                    {
                        right_idx += 1;
                    }

                    // Cartesian product of matching groups
                    for l_idx in left_start..left_idx {
                        left_matched[l_idx] = true;
                        for r_idx in right_start..right_idx {
                            right_matched[r_idx] = true;
                            let mut values = Vec::with_capacity(total_cols);
                            values.extend_from_slice(left_rows[l_idx].as_slice());
                            values.extend_from_slice(right_rows[r_idx].as_slice());
                            result_rows.push(Row::from_values(values));
                        }
                    }
                }
            }
        }

        // Handle LEFT OUTER JOIN - add unmatched left rows
        if join_type.contains("LEFT") || join_type.contains("FULL") {
            for (idx, was_matched) in left_matched.iter().enumerate() {
                if !was_matched {
                    let mut values = Vec::with_capacity(total_cols);
                    values.extend_from_slice(left_rows[idx].as_slice());
                    values.resize(total_cols, Value::null_unknown());
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        // Handle RIGHT OUTER JOIN - add unmatched right rows
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            for (idx, was_matched) in right_matched.iter().enumerate() {
                if !was_matched {
                    let mut values = Vec::with_capacity(total_cols);
                    values.resize(left_col_count, Value::null_unknown());
                    values.extend_from_slice(right_rows[idx].as_slice());
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Compare composite keys for merge join ordering
    ///
    /// Returns Ordering::Less if left < right, Greater if left > right, Equal otherwise.
    /// NULLs are considered equal to each other but are sorted last.
    fn compare_composite_keys(
        row1: &Row,
        row2: &Row,
        indices1: &[usize],
        indices2: &[usize],
    ) -> Ordering {
        debug_assert_eq!(indices1.len(), indices2.len());

        for (&idx1, &idx2) in indices1.iter().zip(indices2.iter()) {
            let cmp = match (row1.get(idx1), row2.get(idx2)) {
                (Some(v1), Some(v2)) => Self::compare_values(v1, v2),
                (None, None) => Ordering::Equal,      // Both NULL
                (None, Some(_)) => Ordering::Greater, // NULL sorts last
                (Some(_), None) => Ordering::Less,    // NULL sorts last
            };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Compare two Values for ordering
    fn compare_values(a: &Value, b: &Value) -> Ordering {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Text(x), Value::Text(y)) => x.cmp(y),
            (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
            (Value::Null(_), Value::Null(_)) => Ordering::Equal,
            (Value::Timestamp(x), Value::Timestamp(y)) => x.cmp(y),
            (Value::Json(x), Value::Json(y)) => x.cmp(y),
            // Cross-type comparisons - try numeric
            (Value::Integer(x), Value::Float(y)) => {
                (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)
            }
            (Value::Float(x), Value::Integer(y)) => {
                x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)
            }
            // NULL sorts last
            (Value::Null(_), _) => Ordering::Greater,
            (_, Value::Null(_)) => Ordering::Less,
            // Different types - use a stable ordering based on type
            _ => {
                // Convert to a numeric type code for ordering
                fn type_code(v: &Value) -> u8 {
                    match v {
                        Value::Null(_) => 0,
                        Value::Boolean(_) => 1,
                        Value::Integer(_) => 2,
                        Value::Float(_) => 3,
                        Value::Text(_) => 4,
                        Value::Timestamp(_) => 5,
                        Value::Json(_) => 6,
                    }
                }
                type_code(a).cmp(&type_code(b))
            }
        }
    }

    /// Check if rows are sorted in ascending order on the specified key indices
    ///
    /// This is used to detect when merge join can be used efficiently.
    /// Returns true if the rows are sorted (ascending) on all key columns.
    /// For efficiency, only checks up to a sample of rows for large inputs.
    fn is_sorted_on_keys(rows: &[Row], key_indices: &[usize]) -> bool {
        if rows.len() <= 1 || key_indices.is_empty() {
            return true; // Trivially sorted
        }

        // For large inputs, sample check is sufficient (merge join will work
        // correctly even if not perfectly sorted, just less efficiently)
        // But for correctness, we check all rows for small inputs
        let check_limit = if rows.len() > 10000 {
            // Sample: check first 1000, last 1000, and 1000 random middle
            // For simplicity, just check first and last portions
            1000
        } else {
            rows.len()
        };

        // Check if rows are sorted by comparing consecutive pairs
        for i in 1..check_limit.min(rows.len()) {
            let prev = &rows[i - 1];
            let curr = &rows[i];

            for &idx in key_indices {
                match (prev.get(idx), curr.get(idx)) {
                    (Some(v1), Some(v2)) => {
                        let cmp = Self::compare_values(v1, v2);
                        match cmp {
                            Ordering::Less => break,           // prev < curr, sorted so far
                            Ordering::Greater => return false, // prev > curr, not sorted
                            Ordering::Equal => continue,       // Check next key
                        }
                    }
                    (None, Some(_)) => break, // NULL < value (NULL first)
                    (Some(_), None) => return false, // value > NULL
                    (None, None) => continue, // Both NULL, check next key
                }
            }
        }

        // Also check last portion for large inputs
        if rows.len() > 10000 {
            let start = rows.len() - check_limit;
            for i in (start + 1)..rows.len() {
                let prev = &rows[i - 1];
                let curr = &rows[i];

                for &idx in key_indices {
                    match (prev.get(idx), curr.get(idx)) {
                        (Some(v1), Some(v2)) => {
                            let cmp = Self::compare_values(v1, v2);
                            match cmp {
                                Ordering::Less => break,
                                Ordering::Greater => return false,
                                Ordering::Equal => continue,
                            }
                        }
                        (None, Some(_)) => break,
                        (Some(_), None) => return false,
                        (None, None) => continue,
                    }
                }
            }
        }

        true
    }

    /// Get estimated cardinalities for join inputs using table statistics
    ///
    /// This is used by AQE to detect significant estimation errors.
    /// Returns (estimated_left, estimated_right) based on statistics if available,
    /// or falls back to actual counts if stats are not available.
    fn get_join_estimates(
        &self,
        left_expr: &Expression,
        right_expr: &Expression,
        actual_left: u64,
        actual_right: u64,
    ) -> (u64, u64) {
        // Try to get estimated cardinality from table statistics
        let estimated_left = self
            .estimate_table_expression_rows(left_expr)
            .unwrap_or(actual_left);
        let estimated_right = self
            .estimate_table_expression_rows(right_expr)
            .unwrap_or(actual_right);

        (estimated_left, estimated_right)
    }

    /// Estimate row count for a table expression using statistics
    fn estimate_table_expression_rows(&self, expr: &Expression) -> Option<u64> {
        match expr {
            Expression::TableSource(table_source) => {
                // For simple table source, use table stats
                let table_name = &table_source.name.value_lower;
                self.get_query_planner()
                    .get_table_stats(table_name)
                    .map(|stats| stats.row_count)
            }
            Expression::SubquerySource(_) => {
                // For subqueries, we don't have estimates (would need to plan the subquery)
                None
            }
            Expression::JoinSource(_) => {
                // For nested joins, we don't estimate (complex to compute)
                None
            }
            _ => None,
        }
    }

    /// Check if an expression is an equality condition (for join algorithm selection)
    fn is_equality_condition(expr: &Expression) -> bool {
        match expr {
            Expression::Infix(infix) => {
                // Check for equality operator
                if infix.operator == "=" {
                    // Check that both sides are column references (not literals)
                    let left_is_col = matches!(
                        infix.left.as_ref(),
                        Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                    );
                    let right_is_col = matches!(
                        infix.right.as_ref(),
                        Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                    );
                    left_is_col && right_is_col
                } else if infix.operator.eq_ignore_ascii_case("AND") {
                    // AND condition - check if any part is an equality join
                    Self::is_equality_condition(&infix.left)
                        || Self::is_equality_condition(&infix.right)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Extract table name from a table expression (for statistics lookup)
    fn extract_table_name(expr: &Expression) -> Option<String> {
        match expr {
            Expression::TableSource(simple) => Some(simple.name.value.clone()),
            Expression::JoinSource(join) => Self::extract_table_name(&join.left),
            Expression::SubquerySource(_) => None, // Can't get stats for subquery
            _ => None,
        }
    }

    /// Build a map of column aliases to their underlying expressions from SELECT columns
    fn build_alias_map(columns: &[Expression]) -> rustc_hash::FxHashMap<String, Expression> {
        let mut alias_map = rustc_hash::FxHashMap::default();
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
        alias_map: &rustc_hash::FxHashMap<String, Expression>,
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
