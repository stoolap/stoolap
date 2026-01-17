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

//! Subquery Execution
//!
//! This module handles execution of subqueries including:
//! - EXISTS subqueries
//! - Scalar subqueries
//! - IN subqueries

use std::sync::Arc;

use crate::common::CompactArc;
use crate::common::SmartString;
use ahash::{AHashMap, AHashSet};

use crate::core::{Error, Result, Value};
use crate::parser::ast::*;
use crate::parser::token::TokenType;
use crate::storage::traits::Engine;

use super::context::{
    cache_batch_aggregate, cache_batch_aggregate_info, cache_count_counter,
    cache_exists_correlation, cache_exists_fetcher, cache_exists_index, cache_exists_pred_key,
    cache_exists_predicate, cache_exists_schema, cache_in_subquery, cache_scalar_subquery,
    cache_semi_join_arc, compute_semi_join_cache_key, extract_table_names_for_cache,
    get_cached_batch_aggregate, get_cached_batch_aggregate_info, get_cached_count_counter,
    get_cached_exists_correlation, get_cached_exists_fetcher, get_cached_exists_index,
    get_cached_exists_pred_key, get_cached_exists_predicate, get_cached_exists_schema,
    get_cached_in_subquery, get_cached_scalar_subquery, get_cached_semi_join,
    BatchAggregateLookupInfo, ExecutionContext, ExistsCorrelationInfo,
};
use super::expr_converter::convert_ast_to_storage_expr;
use super::expression::compute_expression_hash;
use super::operator::{ColumnInfo, MaterializedOperator, Operator};
use super::operators::hash_join::{HashJoinOperator, JoinSide, JoinType};
use super::utils::{dummy_token, dummy_token_clone, value_to_expression};
use super::Executor;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of row IDs to check when verifying visibility for EXISTS/COUNT.
/// We batch up to this many to balance between:
/// - Wasted work if first row is visible (checking extra rows)
/// - Overhead if most row IDs point to deleted rows (need multiple round trips)
///
/// A value of 10 provides reasonable tradeoff for typical workloads.
const VISIBILITY_CHECK_BATCH_SIZE: usize = 10;

// ============================================================================
// Semi-Join Optimization for EXISTS Subqueries
// ============================================================================

// Result type for correlation extraction: (outer_col, outer_table, inner_col, remaining_predicate)
type CorrelationExtraction = (String, Option<String>, String, Option<Arc<Expression>>);

/// Information extracted from an EXISTS subquery for semi-join optimization.
/// Example: EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500)
/// - outer_column: "u.id" (or "id" with outer table "u")
/// - inner_column: "o.user_id" (or "user_id")
/// - inner_table: "orders"
/// - inner_alias: Some("o")
/// - non_correlated_where: Some("o.amount > 500")
#[derive(Debug)]
pub struct SemiJoinInfo {
    /// The outer column referenced in the correlation (e.g., "id" from "u.id")
    pub outer_column: String,
    /// The outer table alias if qualified (e.g., "u" from "u.id")
    pub outer_table: Option<String>,
    /// The inner column used in the correlation (e.g., "user_id" from "o.user_id")
    pub inner_column: String,
    /// The inner table name
    pub inner_table: String,
    /// The inner table alias if present
    pub inner_alias: Option<String>,
    /// Non-correlated part of the WHERE clause (filters only on inner table)
    /// Uses Arc to avoid cloning expression trees during semi-join optimization
    pub non_correlated_where: Option<Arc<Expression>>,
    /// Whether this is NOT EXISTS
    pub is_negated: bool,
}

/// Information needed for index-nested-loop EXISTS execution.
///
/// This is used for direct index probing instead of running a full subquery.
#[derive(Debug, Clone)]
struct IndexNestedLoopInfo {
    outer_column: String,
    outer_table: Option<String>,
    inner_column: String,
    inner_table: String,
    #[allow(dead_code)]
    additional_predicate: Option<Expression>,
}

impl Executor {
    /// Process subqueries in WHERE clause, replacing EXISTS with boolean literals
    ///
    /// This function walks the expression tree and executes any EXISTS subqueries,
    /// replacing them with boolean literal values.
    pub(crate) fn process_where_subqueries(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<Expression> {
        match expr {
            Expression::Exists(exists) => {
                // Execute the EXISTS subquery
                let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                Ok(Expression::BooleanLiteral(BooleanLiteral {
                    token: dummy_token(
                        if exists_result { "TRUE" } else { "FALSE" },
                        TokenType::Keyword,
                    ),
                    value: exists_result,
                }))
            }

            Expression::AllAny(all_any) => {
                // Execute the subquery to get all values
                let values = self.execute_in_subquery(&all_any.subquery, ctx)?;

                // Convert ALL/ANY to an equivalent expression that the evaluator can handle
                self.convert_all_any_to_expression(all_any, values)
            }

            Expression::Prefix(prefix) => {
                // Handle NOT EXISTS
                if prefix.operator.eq_ignore_ascii_case("NOT") {
                    if let Expression::Exists(exists) = prefix.right.as_ref() {
                        // Execute the EXISTS subquery and negate the result
                        let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                        return Ok(Expression::BooleanLiteral(BooleanLiteral {
                            token: dummy_token(
                                if !exists_result { "TRUE" } else { "FALSE" },
                                TokenType::Keyword,
                            ),
                            value: !exists_result,
                        }));
                    }
                }

                // Process the inner expression recursively
                let processed_right = self.process_where_subqueries(&prefix.right, ctx)?;
                Ok(Expression::Prefix(PrefixExpression {
                    token: prefix.token.clone(),
                    operator: prefix.operator.clone(),
                    op_type: prefix.op_type,
                    right: Box::new(processed_right),
                }))
            }

            Expression::Infix(infix) => {
                // Process both sides recursively
                let processed_left = self.process_where_subqueries(&infix.left, ctx)?;
                let processed_right = self.process_where_subqueries(&infix.right, ctx)?;

                Ok(Expression::Infix(InfixExpression {
                    token: infix.token.clone(),
                    left: Box::new(processed_left),
                    operator: infix.operator.clone(),
                    op_type: infix.op_type,
                    right: Box::new(processed_right),
                }))
            }

            Expression::In(in_expr) => {
                // Process the left expression
                let processed_left = self.process_where_subqueries(&in_expr.left, ctx)?;

                // Check if the right side is a scalar subquery
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    // Check if left side is a tuple (multi-column IN)
                    let is_multi_column = matches!(processed_left, Expression::ExpressionList(_));

                    if is_multi_column {
                        // Multi-column IN: (a, b) IN (SELECT x, y FROM t)
                        let rows = self.execute_in_subquery_rows(&subquery.subquery, ctx)?;

                        // Pre-allocate with known capacity for better performance
                        let mut expressions = Vec::with_capacity(rows.len());
                        let paren_token = dummy_token("(", TokenType::Punctuator);

                        // Convert each row to an ExpressionList (tuple)
                        for row in rows {
                            let col_count = row.len();
                            let mut tuple_exprs = Vec::with_capacity(col_count);
                            for value in &row {
                                tuple_exprs.push(value_to_expression(value));
                            }
                            expressions.push(Expression::ExpressionList(Box::new(
                                ExpressionList {
                                    token: paren_token.clone(),
                                    expressions: tuple_exprs,
                                },
                            )));
                        }

                        return Ok(Expression::In(InExpression {
                            token: in_expr.token.clone(),
                            left: Box::new(processed_left),
                            right: Box::new(Expression::ExpressionList(Box::new(ExpressionList {
                                token: paren_token,
                                expressions,
                            }))),
                            not: in_expr.not,
                        }));
                    } else {
                        // Single-column IN - use InHashSet for O(1) lookups
                        let values = self.execute_in_subquery(&subquery.subquery, ctx)?;

                        // Collect into AHashSet for O(1) membership testing (better for Value types)
                        let hash_set: AHashSet<Value> = values.into_iter().collect();

                        // Use InHashSet with Arc for fast O(1) lookup per row
                        return Ok(Expression::InHashSet(InHashSetExpression {
                            token: in_expr.token.clone(),
                            column: Box::new(processed_left),
                            values: CompactArc::new(hash_set),
                            not: in_expr.not,
                        }));
                    }
                }

                // If not a subquery, just return the original with processed left
                Ok(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(processed_left),
                    right: in_expr.right.clone(),
                    not: in_expr.not,
                }))
            }

            Expression::Between(between) => {
                let processed_expr = self.process_where_subqueries(&between.expr, ctx)?;
                let processed_lower = self.process_where_subqueries(&between.lower, ctx)?;
                let processed_upper = self.process_where_subqueries(&between.upper, ctx)?;

                Ok(Expression::Between(BetweenExpression {
                    token: between.token.clone(),
                    expr: Box::new(processed_expr),
                    not: between.not,
                    lower: Box::new(processed_lower),
                    upper: Box::new(processed_upper),
                }))
            }

            Expression::ScalarSubquery(subquery) => {
                // Execute scalar subquery and replace with literal value
                let value = self.execute_scalar_subquery(&subquery.subquery, ctx)?;

                match value {
                    crate::core::Value::Integer(i) => {
                        Ok(Expression::IntegerLiteral(IntegerLiteral {
                            token: dummy_token(&i.to_string(), TokenType::Integer),
                            value: i,
                        }))
                    }
                    crate::core::Value::Float(f) => Ok(Expression::FloatLiteral(FloatLiteral {
                        token: dummy_token(&f.to_string(), TokenType::Float),
                        value: f,
                    })),
                    crate::core::Value::Text(s) => Ok(Expression::StringLiteral(StringLiteral {
                        token: dummy_token(&format!("'{}'", s), TokenType::String),
                        value: s.as_str().into(),
                        type_hint: None,
                    })),
                    crate::core::Value::Boolean(b) => {
                        Ok(Expression::BooleanLiteral(BooleanLiteral {
                            token: dummy_token(
                                if b { "TRUE" } else { "FALSE" },
                                TokenType::Keyword,
                            ),
                            value: b,
                        }))
                    }
                    crate::core::Value::Null(_) => Ok(Expression::NullLiteral(NullLiteral {
                        token: dummy_token("NULL", TokenType::Keyword),
                    })),
                    _ => Ok(Expression::StringLiteral(StringLiteral {
                        token: dummy_token(&format!("'{}'", value), TokenType::String),
                        value: value.to_string().into(),
                        type_hint: None,
                    })),
                }
            }

            Expression::Case(case) => {
                // Process the operand (if present)
                let processed_value = if let Some(ref value) = case.value {
                    Some(Box::new(self.process_where_subqueries(value, ctx)?))
                } else {
                    None
                };

                // Process each WHEN clause
                let processed_whens: Result<Vec<WhenClause>> = case
                    .when_clauses
                    .iter()
                    .map(|when| {
                        Ok(WhenClause {
                            token: when.token.clone(),
                            condition: self.process_where_subqueries(&when.condition, ctx)?,
                            then_result: self.process_where_subqueries(&when.then_result, ctx)?,
                        })
                    })
                    .collect();

                // Process the ELSE clause (if present)
                let processed_else = if let Some(ref else_val) = case.else_value {
                    Some(Box::new(self.process_where_subqueries(else_val, ctx)?))
                } else {
                    None
                };

                Ok(Expression::Case(Box::new(CaseExpression {
                    token: case.token.clone(),
                    value: processed_value,
                    when_clauses: processed_whens?,
                    else_value: processed_else,
                })))
            }

            Expression::Cast(cast) => {
                let processed_expr = self.process_where_subqueries(&cast.expr, ctx)?;
                Ok(Expression::Cast(CastExpression {
                    token: cast.token.clone(),
                    expr: Box::new(processed_expr),
                    type_name: cast.type_name.clone(),
                }))
            }

            Expression::FunctionCall(func) => {
                // Process function arguments to handle any nested subqueries
                let processed_args: Result<Vec<Expression>> = func
                    .arguments
                    .iter()
                    .map(|arg| self.process_where_subqueries(arg, ctx))
                    .collect();

                Ok(Expression::FunctionCall(Box::new(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: processed_args?,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                })))
            }

            // For all other expression types, return as-is
            _ => Ok(expr.clone()),
        }
    }

    /// Execute an EXISTS subquery and return true if any rows exist
    pub(crate) fn execute_exists_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<bool> {
        // Try index-nested-loop optimization for correlated EXISTS
        if let Some(exists) = self.try_execute_exists_with_index_probe(subquery, ctx)? {
            return Ok(exists);
        }

        // Fall back to full subquery execution
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Check if there's at least one row
        let exists = result.next();

        Ok(exists)
    }

    /// Try to execute EXISTS using index-nested-loop optimization.
    ///
    /// This optimization is used when:
    /// 1. The subquery has a simple correlation: inner.col = outer.col
    /// 2. The inner table has an index on the correlation column
    /// 3. There's an outer row value available in the context
    ///
    /// Instead of running a full query, we probe the index directly for O(log n) or O(1) lookup.
    fn try_execute_exists_with_index_probe(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Option<bool>> {
        // Need outer row context for correlated subquery
        let outer_row = match ctx.outer_row() {
            Some(row) => row,
            None => return Ok(None), // Not a correlated context
        };

        // OPTIMIZATION: Cache correlation info to avoid per-row extraction
        // The subquery pointer is stable within a query execution, so we use it as cache key.
        // Using pointer address directly as usize avoids format! allocation entirely.
        let subquery_ptr = subquery as *const SelectStatement as usize;

        let correlation = match get_cached_exists_correlation(subquery_ptr) {
            Some(Some(info)) => info,
            Some(None) => return Ok(None), // Previously determined not extractable
            None => {
                // First probe for this subquery - extract and cache
                let info = Self::extract_index_nested_loop_info(subquery);
                let cached_info = info.map(|i| {
                    // Pre-compute index cache key once to avoid per-probe format! allocation
                    let index_cache_key = format!("{}:{}", i.inner_table, i.inner_column);
                    ExistsCorrelationInfo {
                        outer_column: i.outer_column.clone(),
                        outer_table: i.outer_table.clone(),
                        inner_column: i.inner_column.clone(),
                        inner_table: i.inner_table.clone(),
                        outer_column_lower: i.outer_column.to_lowercase(),
                        outer_qualified_lower: i.outer_table.as_ref().map(|tbl| {
                            format!("{}.{}", tbl.to_lowercase(), i.outer_column.to_lowercase())
                        }),
                        additional_predicate: i.additional_predicate.clone(),
                        index_cache_key,
                    }
                });
                // cache_exists_correlation returns the Arc-wrapped version
                match cache_exists_correlation(subquery_ptr, cached_info) {
                    Some(arc) => arc,
                    None => return Ok(None),
                }
            }
        };

        // Get the outer value from the outer row hashmap using pre-computed lowercase keys
        // This avoids per-row to_lowercase() calls
        // Use .as_str() for lookups since map now uses Arc<str> keys
        let outer_value = if let Some(ref qualified) = correlation.outer_qualified_lower {
            outer_row
                .get(qualified.as_str())
                .or_else(|| outer_row.get(correlation.outer_column_lower.as_str()))
        } else {
            outer_row.get(correlation.outer_column_lower.as_str())
        };

        let outer_value = match outer_value {
            Some(v) if !v.is_null() => v.clone(),
            Some(_) => return Ok(Some(false)), // NULL never matches in EXISTS
            None => return Ok(None),           // Column not found, fall back
        };

        // OPTIMIZATION: Cache index reference to avoid repeated lookups
        // This reduces the ~2-5μs overhead per EXISTS probe to nearly zero for subsequent probes
        // Uses pre-computed index_cache_key from ExistsCorrelationInfo to avoid per-probe format!
        let index = match get_cached_exists_index(&correlation.index_cache_key) {
            Some(idx) => idx,
            None => {
                // First time: get index from engine and cache it
                let indexes = match self.engine.get_all_indexes(&correlation.inner_table) {
                    Ok(idxs) => idxs,
                    Err(_) => return Ok(None), // Table not found, fall back
                };

                // Find the index on the correlation column
                let idx = indexes
                    .into_iter()
                    .find(|idx| idx.column_names().contains(&correlation.inner_column));

                match idx {
                    Some(idx) => {
                        cache_exists_index(correlation.index_cache_key.clone(), idx.clone());
                        idx
                    }
                    None => return Ok(None), // No index, fall back to full query
                }
            }
        };

        // Probe the index for matching row IDs
        let row_ids = index.get_row_ids_equal(std::slice::from_ref(&outer_value));

        if row_ids.is_empty() {
            return Ok(Some(false)); // No matches from index
        }

        // If there's no additional predicate, check if at least one row is visible
        // Note: Index may contain row_ids for deleted rows, so we must verify visibility
        if correlation.additional_predicate.is_none() {
            let row_fetcher = match self.get_or_create_row_fetcher(&correlation.inner_table) {
                Some(f) => f,
                None => return Ok(None), // Fall back if fetcher creation fails
            };
            // Check batches until we find a visible row or exhaust all row_ids
            // We can't stop after first batch because deleted rows may precede visible ones
            for chunk in row_ids.chunks(VISIBILITY_CHECK_BATCH_SIZE) {
                let visible = row_fetcher(chunk);
                if !visible.is_empty() {
                    return Ok(Some(true)); // Found at least one visible row
                }
            }
            return Ok(Some(false)); // No visible rows in any batch
        }

        // With additional predicate, we need to check each matching row
        // OPTIMIZATION: Directly fetch rows by row_ids and evaluate the predicate
        // This avoids the overhead of building and executing a full SELECT query
        let additional_pred = correlation.additional_predicate.as_ref().unwrap();

        // OPTIMIZATION: Cache schema column names to avoid repeated get_table_schema() calls
        // This reduces the ~1μs overhead per EXISTS probe
        let columns = match get_cached_exists_schema(&correlation.inner_table) {
            Some(cols) => cols,
            None => {
                let schema = match self.engine.get_table_schema(&correlation.inner_table) {
                    Ok(s) => s,
                    Err(_) => return Ok(None), // Fall back if schema not found
                };
                // Use schema's cached column names - O(1) Arc clone
                let cols = schema.column_names_arc();
                cache_exists_schema(correlation.inner_table.clone(), CompactArc::clone(&cols));
                cols
            }
        };

        // Try to get cached predicate filter using the cached predicate cache key
        // (reuse subquery_ptr from correlation cache above)
        let predicate_filter = match get_cached_exists_pred_key(subquery_ptr) {
            Some(cache_key) => {
                // Fast path: we have a cached predicate cache key
                match get_cached_exists_predicate(&cache_key) {
                    Some(filter) => filter,
                    None => {
                        // Cache key exists but filter was evicted - this shouldn't happen normally
                        // but handle it by recompiling
                        let stripped_pred = Self::strip_table_alias_from_expr(additional_pred);
                        match super::expression::RowFilter::new(&stripped_pred, &columns) {
                            Ok(filter) => {
                                cache_exists_predicate(cache_key, filter.clone());
                                filter
                            }
                            Err(_) => return Ok(None),
                        }
                    }
                }
            }
            None => {
                // First probe for this subquery - compute and cache the predicate cache key
                let stripped_pred = Self::strip_table_alias_from_expr(additional_pred);
                // Use expression hash instead of Debug formatting to avoid expensive string allocation
                let pred_hash = compute_expression_hash(&stripped_pred);
                let cache_key = format!("{}:{}", correlation.inner_table, pred_hash);

                // Cache the predicate cache key for subsequent probes
                cache_exists_pred_key(subquery_ptr, cache_key.clone());

                match get_cached_exists_predicate(&cache_key) {
                    Some(filter) => filter,
                    None => match super::expression::RowFilter::new(&stripped_pred, &columns) {
                        Ok(filter) => {
                            cache_exists_predicate(cache_key, filter.clone());
                            filter
                        }
                        Err(_) => return Ok(None),
                    },
                }
            }
        };

        // Get or create a cached row fetcher for this table
        let row_fetcher = match self.get_or_create_row_fetcher(&correlation.inner_table) {
            Some(f) => f,
            None => return Ok(None), // Fall back if fetcher creation fails
        };

        // Fetch rows by their IDs using the cached row fetcher
        const BATCH_SIZE: usize = 100;
        for batch in row_ids.chunks(BATCH_SIZE) {
            let fetched = row_fetcher(batch);

            // Check each row against the predicate
            for (_row_id, row) in fetched {
                if predicate_filter.matches(&row) {
                    return Ok(Some(true));
                }
            }
        }

        // No rows matched the predicate
        Ok(Some(false))
    }

    /// Extract index-nested-loop correlation info from a subquery.
    ///
    /// Looks for patterns like:
    /// SELECT 1 FROM orders WHERE orders.user_id = u.id [AND additional_predicates]
    fn extract_index_nested_loop_info(subquery: &SelectStatement) -> Option<IndexNestedLoopInfo> {
        // Must have a simple table source
        let (inner_table, inner_alias) = match subquery.table_expr.as_ref().map(|b| b.as_ref()) {
            Some(Expression::TableSource(ts)) => {
                let alias = ts.alias.as_ref().map(|a| a.value.clone());
                (ts.name.value.clone(), alias)
            }
            _ => return None,
        };

        // Must have a WHERE clause
        let where_clause = subquery.where_clause.as_ref()?;

        // Extract correlation condition
        let inner_table_lower: String = inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase())
            .to_lowercase()
            .into();
        let inner_tables = vec![inner_table_lower];

        Self::extract_correlation_for_index(where_clause, &inner_tables, &inner_table)
    }

    /// Extract correlation info suitable for index-nested-loop from a WHERE clause.
    fn extract_correlation_for_index(
        expr: &Expression,
        inner_tables: &[String],
        inner_table_name: &str,
    ) -> Option<IndexNestedLoopInfo> {
        match expr {
            Expression::Infix(infix) if infix.operator == "=" => {
                // Try to match: inner.col = outer.col or outer.col = inner.col
                if let Some((inner_col, outer_col, outer_tbl)) =
                    Self::extract_correlation_pair(&infix.left, &infix.right, inner_tables)
                {
                    return Some(IndexNestedLoopInfo {
                        outer_column: outer_col,
                        outer_table: outer_tbl,
                        inner_column: inner_col,
                        inner_table: inner_table_name.to_string(),
                        additional_predicate: None,
                    });
                }
                if let Some((inner_col, outer_col, outer_tbl)) =
                    Self::extract_correlation_pair(&infix.right, &infix.left, inner_tables)
                {
                    return Some(IndexNestedLoopInfo {
                        outer_column: outer_col,
                        outer_table: outer_tbl,
                        inner_column: inner_col,
                        inner_table: inner_table_name.to_string(),
                        additional_predicate: None,
                    });
                }
                None
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try left side for correlation
                if let Some(mut info) =
                    Self::extract_correlation_for_index(&infix.left, inner_tables, inner_table_name)
                {
                    // Right side becomes additional predicate
                    info.additional_predicate = Some((*infix.right).clone());
                    return Some(info);
                }
                // Try right side for correlation
                if let Some(mut info) = Self::extract_correlation_for_index(
                    &infix.right,
                    inner_tables,
                    inner_table_name,
                ) {
                    // Left side becomes additional predicate
                    info.additional_predicate = Some((*infix.left).clone());
                    return Some(info);
                }
                None
            }

            _ => None,
        }
    }

    /// Try to execute a correlated scalar COUNT subquery using index probe.
    ///
    /// This optimization handles patterns like:
    /// `(SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id)`
    ///
    /// Instead of running a full query for each outer row, we:
    /// 1. Extract the correlation info (user_id = u.id)
    /// 2. Probe the index on user_id with the outer value
    /// 3. Return the count of matching row IDs directly
    ///
    /// This is O(1) index lookup per row vs O(n) query execution.
    fn try_execute_scalar_count_with_index(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Option<i64>> {
        // Need outer row context for correlated subquery
        let outer_row = match ctx.outer_row() {
            Some(row) => row,
            None => return Ok(None), // Not a correlated context
        };

        // Check if this is a COUNT(*) or COUNT(col) aggregate
        if subquery.columns.len() != 1 {
            return Ok(None);
        }

        let is_count = match &subquery.columns[0] {
            Expression::Aliased(a) => Self::is_count_expression(&a.expression),
            expr => Self::is_count_expression(expr),
        };

        if !is_count {
            return Ok(None);
        }

        // Must not have GROUP BY (scalar COUNT must return single value)
        if !subquery.group_by.columns.is_empty() {
            return Ok(None);
        }

        // OPTIMIZATION: Use the same ExistsCorrelationInfo caching as EXISTS optimization
        // This avoids per-probe format! allocations for qualified names and index cache keys
        let subquery_ptr = subquery as *const SelectStatement as usize;

        let correlation = match get_cached_exists_correlation(subquery_ptr) {
            Some(Some(info)) => info,
            Some(None) => return Ok(None), // Previously determined not extractable
            None => {
                // First probe for this subquery - extract and cache
                let info = Self::extract_index_nested_loop_info(subquery);
                let cached_info = info.map(|i| {
                    // Pre-compute index cache key once to avoid per-probe format! allocation
                    let index_cache_key = format!("{}:{}", i.inner_table, i.inner_column);
                    ExistsCorrelationInfo {
                        outer_column: i.outer_column.clone(),
                        outer_table: i.outer_table.clone(),
                        inner_column: i.inner_column.clone(),
                        inner_table: i.inner_table.clone(),
                        outer_column_lower: i.outer_column.to_lowercase(),
                        outer_qualified_lower: i.outer_table.as_ref().map(|tbl| {
                            format!("{}.{}", tbl.to_lowercase(), i.outer_column.to_lowercase())
                        }),
                        additional_predicate: i.additional_predicate.clone(),
                        index_cache_key,
                    }
                });
                // cache_exists_correlation returns the Arc-wrapped version
                match cache_exists_correlation(subquery_ptr, cached_info) {
                    Some(arc) => arc,
                    None => return Ok(None),
                }
            }
        };

        // Get the outer value from the outer row hashmap using pre-computed lowercase keys
        // This avoids per-probe to_lowercase() calls
        // Use .as_str() for lookups since map now uses Arc<str> keys
        let outer_value = if let Some(ref qualified) = correlation.outer_qualified_lower {
            outer_row
                .get(qualified.as_str())
                .or_else(|| outer_row.get(correlation.outer_column_lower.as_str()))
        } else {
            outer_row.get(correlation.outer_column_lower.as_str())
        };

        let outer_value = match outer_value {
            Some(v) if !v.is_null() => v.clone(),
            Some(_) => return Ok(Some(0)), // NULL never matches, count is 0
            None => return Ok(None),       // Column not found, fall back
        };

        // Use cached index lookup with pre-computed index_cache_key
        let index = match get_cached_exists_index(&correlation.index_cache_key) {
            Some(idx) => idx,
            None => {
                // First time: get index from engine and cache it
                let indexes = match self.engine.get_all_indexes(&correlation.inner_table) {
                    Ok(idxs) => idxs,
                    Err(_) => return Ok(None), // Table not found, fall back
                };

                // Find the index on the correlation column
                let idx = indexes
                    .into_iter()
                    .find(|idx| idx.column_names().contains(&correlation.inner_column));

                match idx {
                    Some(idx) => {
                        cache_exists_index(correlation.index_cache_key.clone(), idx.clone());
                        idx
                    }
                    None => return Ok(None), // No index, fall back to full query
                }
            }
        };

        // Probe the index for matching row IDs
        let row_ids = index.get_row_ids_equal(std::slice::from_ref(&outer_value));

        // If there's no additional predicate, count only visible rows
        // Note: Index may contain row_ids for deleted rows, so we must verify visibility
        if correlation.additional_predicate.is_none() {
            if row_ids.is_empty() {
                return Ok(Some(0));
            }
            // Use row_counter for COUNT (avoids cloning row data)
            let row_counter = match get_cached_count_counter(&correlation.inner_table) {
                Some(c) => c,
                None => match self.get_or_create_row_counter(&correlation.inner_table) {
                    Some(c) => c,
                    None => {
                        // Fall back to row_fetcher if counter not available
                        let row_fetcher = match get_cached_exists_fetcher(&correlation.inner_table)
                        {
                            Some(f) => f,
                            None => {
                                match self.get_or_create_row_fetcher(&correlation.inner_table) {
                                    Some(f) => f,
                                    None => return Ok(None),
                                }
                            }
                        };
                        let visible_rows = row_fetcher(&row_ids);
                        return Ok(Some(visible_rows.len() as i64));
                    }
                },
            };
            // Count visible rows without cloning
            let count = row_counter(&row_ids);
            return Ok(Some(count as i64));
        }

        // With additional predicate, we need to filter the matching rows
        // Fall back to full query execution for complex cases
        Ok(None)
    }

    /// Check if an expression is a COUNT aggregate function.
    fn is_count_expression(expr: &Expression) -> bool {
        match expr {
            Expression::FunctionCall(func) => func.function.eq_ignore_ascii_case("COUNT"),
            Expression::Aliased(a) => Self::is_count_expression(&a.expression),
            _ => false,
        }
    }

    /// Extract the aggregate function name from an expression.
    /// Returns Some(func_name) for COUNT, SUM, AVG, MIN, MAX.
    fn extract_aggregate_function(expr: &Expression) -> Option<String> {
        match expr {
            Expression::FunctionCall(func) => {
                let name = func.function.to_uppercase();
                if matches!(name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX") {
                    Some(name.into())
                } else {
                    None
                }
            }
            Expression::Aliased(a) => Self::extract_aggregate_function(&a.expression),
            _ => None,
        }
    }

    /// Try to look up a batch aggregate result from cache.
    /// Returns Some(value) if found in cache, None otherwise.
    /// Uses cached lookup info to avoid per-row allocations.
    fn try_lookup_batch_aggregate(
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<crate::core::Value> {
        // Need outer row context for correlated subquery
        let outer_row = ctx.outer_row()?;

        // Use pointer address as cache key (O(1) vs O(n) for to_string())
        let subquery_ptr = subquery as *const _ as usize;
        let lookup_info = match get_cached_batch_aggregate_info(subquery_ptr) {
            Some(cached) => cached?,
            None => {
                // First time: compute and cache the lookup info
                let info = Self::compute_batch_aggregate_info(subquery);
                cache_batch_aggregate_info(subquery_ptr, info)?
            }
        };

        // Get the outer value from the outer row hashmap (no allocation)
        // Use .as_str() for lookups since map now uses Arc<str> keys
        let outer_value = if let Some(ref qualified) = lookup_info.outer_qualified_lower {
            outer_row
                .get(qualified.as_str())
                .or_else(|| outer_row.get(lookup_info.outer_column_lower.as_str()))
        } else {
            outer_row.get(lookup_info.outer_column_lower.as_str())
        };

        let outer_value = match outer_value {
            Some(v) if !v.is_null() => v.clone(),
            // NULL never matches - for COUNT return 0, for other aggregates return NULL
            Some(_) => {
                return if lookup_info.is_count {
                    Some(Value::Integer(0))
                } else {
                    Some(Value::null_unknown())
                }
            }
            None => return None, // Column not found, can't optimize
        };

        // Look up in cache
        let cache = get_cached_batch_aggregate(&lookup_info.cache_key)?;
        let result = cache.get(&outer_value).cloned();

        // Return 0 for COUNT if key not found (no matching rows)
        if result.is_none() && lookup_info.is_count {
            return Some(Value::Integer(0));
        }

        result
    }

    /// Compute batch aggregate lookup info for a subquery.
    /// Returns None if the subquery is not batchable.
    fn compute_batch_aggregate_info(
        subquery: &SelectStatement,
    ) -> Option<BatchAggregateLookupInfo> {
        // Build cache key - returns None if not a batchable aggregate
        let cache_key = Self::build_batch_aggregate_key(subquery)?;

        // Extract correlation info
        let correlation = Self::extract_index_nested_loop_info(subquery)?;

        // Pre-compute lowercase column names
        let outer_column_lower = correlation.outer_column.to_lowercase();
        let outer_qualified_lower = correlation
            .outer_table
            .as_ref()
            .map(|tbl| format!("{}.{}", tbl.to_lowercase(), outer_column_lower));

        let is_count = Self::is_count_expression(&subquery.columns[0]);

        Some(BatchAggregateLookupInfo {
            cache_key,
            outer_column_lower,
            outer_qualified_lower,
            is_count,
        })
    }

    /// Build a cache key for batch aggregate based on subquery structure.
    /// The key identifies the subquery pattern (table, correlation column, aggregate function).
    fn build_batch_aggregate_key(subquery: &SelectStatement) -> Option<String> {
        // Extract table name
        let table_name = match subquery.table_expr.as_ref().map(|b| b.as_ref()) {
            Some(Expression::TableSource(ts)) => ts.name.value.clone(),
            _ => return None,
        };

        // Extract aggregate function
        if subquery.columns.len() != 1 {
            return None;
        }
        let agg_func = Self::extract_aggregate_function(&subquery.columns[0])?;

        // Extract correlation column
        let correlation = Self::extract_index_nested_loop_info(subquery)?;

        Some(format!(
            "batch_agg:{}:{}:{}",
            table_name.to_lowercase(),
            correlation.inner_column.to_lowercase(),
            agg_func
        ))
    }

    /// Try to execute and cache a batch aggregate query.
    /// This executes `SELECT correlation_col, AGG() FROM table GROUP BY correlation_col`
    /// and caches the results for O(1) lookup per outer row.
    fn try_execute_and_cache_batch_aggregate(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Option<CompactArc<AHashMap<Value, Value>>>> {
        // Must have single aggregate column
        if subquery.columns.len() != 1 {
            return Ok(None);
        }

        // Verify this is an aggregate function (COUNT, SUM, AVG, MIN, MAX)
        if Self::extract_aggregate_function(&subquery.columns[0]).is_none() {
            return Ok(None);
        }

        // Must not have GROUP BY (we'll add our own)
        if !subquery.group_by.columns.is_empty() {
            return Ok(None);
        }

        // Must not have HAVING, LIMIT, OFFSET (these would change results)
        if subquery.having.is_some() || subquery.limit.is_some() || subquery.offset.is_some() {
            return Ok(None);
        }

        // Extract correlation info
        let correlation = match Self::extract_index_nested_loop_info(subquery) {
            Some(c) => c,
            None => return Ok(None),
        };

        // Build cache key (includes additional predicate in key for uniqueness)
        let cache_key = match Self::build_batch_aggregate_key(subquery) {
            Some(k) => k,
            None => return Ok(None),
        };

        // Check if already cached
        if let Some(cached) = get_cached_batch_aggregate(&cache_key) {
            return Ok(Some(cached));
        }

        // Build the batch aggregate query:
        // SELECT correlation_column, AGG(...) FROM table GROUP BY correlation_column
        let inner_col_val: SmartString = correlation.inner_column.clone().into();
        let inner_col_expr = Expression::Identifier(Identifier {
            token: dummy_token(&correlation.inner_column, TokenType::Identifier),
            value: inner_col_val.clone(),
            value_lower: inner_col_val.to_lowercase(),
        });

        // Clone the aggregate expression from the original subquery
        let agg_expr = subquery.columns[0].clone();

        // Build list of inner table names for outer reference detection
        let inner_tables = vec![
            correlation.inner_table.to_lowercase(),
            correlation.inner_table.clone(), // Also check original case
        ];

        // Include additional predicate in WHERE clause ONLY if it doesn't reference outer columns.
        // If the predicate contains outer references (like "o.amount > c.id * 50"),
        // we can't use batch caching because those references can't be resolved.
        let where_clause = correlation.additional_predicate.as_ref().and_then(|pred| {
            if Self::expression_has_outer_reference(pred, &inner_tables) {
                None // Can't use batch caching with outer references in predicate
            } else {
                Some(Box::new(pred.clone()))
            }
        });

        // If additional predicate has outer references, we can't use batch caching
        if correlation.additional_predicate.is_some() && where_clause.is_none() {
            return Ok(None);
        }

        let batch_query = SelectStatement {
            token: dummy_token("SELECT", TokenType::Keyword),
            columns: vec![inner_col_expr.clone(), agg_expr],
            table_expr: subquery.table_expr.clone(),
            where_clause, // Include additional predicate for filtering
            group_by: GroupByClause {
                columns: vec![inner_col_expr],
                modifier: GroupByModifier::None,
            },
            having: None,
            order_by: vec![],
            limit: None,
            offset: None,
            distinct: false,
            with: None,
            window_defs: vec![],
            set_operations: vec![],
        };

        // Execute the batch query
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(&batch_query, &subquery_ctx)?;

        // Build the result map - use take_row() to avoid cloning
        let mut result_map: AHashMap<Value, Value> = AHashMap::new();
        while result.next() {
            let row = result.take_row();
            if row.len() >= 2 {
                // into_values() uses Arc::try_unwrap() to move without cloning when sole owner
                let mut values = row.into_values();
                if values.len() >= 2 {
                    let value = values.pop().unwrap();
                    let key = values.swap_remove(0);
                    if !key.is_null() {
                        result_map.insert(key, value);
                    }
                }
            }
        }

        // Cache the results
        cache_batch_aggregate(cache_key.clone(), result_map);

        // Return the cached Arc
        Ok(get_cached_batch_aggregate(&cache_key))
    }

    /// Get or create a cached row fetcher for the given table.
    ///
    /// This helper reduces code duplication for the pattern of:
    /// 1. Check cache for existing fetcher
    /// 2. If not found, create from engine and cache it
    /// 3. Return the fetcher or None if creation fails
    fn get_or_create_row_fetcher(
        &self,
        table_name: &str,
    ) -> Option<std::sync::Arc<super::context::RowFetcher>> {
        if let Some(f) = get_cached_exists_fetcher(table_name) {
            return Some(f);
        }

        let fetcher = match self.engine.get_row_fetcher(table_name) {
            Ok(f) => f,
            Err(_) => return None, // Fall back if fetcher creation fails
        };
        cache_exists_fetcher(table_name.to_string(), fetcher);
        get_cached_exists_fetcher(table_name)
    }

    /// Get or create a cached row counter for a table.
    ///
    /// This is similar to get_or_create_row_fetcher but for COUNT operations.
    /// It returns a function that counts visible rows without cloning row data.
    fn get_or_create_row_counter(
        &self,
        table_name: &str,
    ) -> Option<std::sync::Arc<super::context::RowCounter>> {
        if let Some(c) = get_cached_count_counter(table_name) {
            return Some(c);
        }

        let counter = match self.engine.get_row_counter(table_name) {
            Ok(c) => c,
            Err(_e) => {
                // Fall back to slower path if counter creation fails
                #[cfg(debug_assertions)]
                eprintln!(
                    "[WARN] get_row_counter failed for '{}': {:?}",
                    table_name, _e
                );
                return None;
            }
        };
        cache_count_counter(table_name.to_string(), counter);
        get_cached_count_counter(table_name)
    }

    /// Extract correlation pair from two expressions.
    /// Returns (inner_column, outer_column, outer_table) if one side is inner ref and other is outer ref.
    fn extract_correlation_pair(
        left: &Expression,
        right: &Expression,
        inner_tables: &[String],
    ) -> Option<(String, String, Option<String>)> {
        // left should be inner column, right should be outer column
        let inner_col = Self::get_inner_column_name(left, inner_tables)?;
        let (outer_col, outer_tbl) = Self::get_outer_column_name(right, inner_tables)?;
        Some((inner_col, outer_col, outer_tbl))
    }

    /// Get column name if expression is an inner table column reference.
    fn get_inner_column_name(expr: &Expression, inner_tables: &[String]) -> Option<String> {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                let table = qid.qualifier.value_lower.as_str();
                if inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table)) {
                    Some(qid.name.value.to_string())
                } else {
                    None
                }
            }
            Expression::Identifier(id) => {
                // Unqualified identifier assumed to be inner if in context
                Some(id.value.to_string())
            }
            _ => None,
        }
    }

    /// Get column name if expression is an outer table column reference.
    /// Returns (column_name, table_alias).
    fn get_outer_column_name(
        expr: &Expression,
        inner_tables: &[String],
    ) -> Option<(String, Option<String>)> {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                let table = qid.qualifier.value_lower.as_str();
                // If NOT in inner_tables, it's an outer reference
                if !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table)) {
                    Some((
                        qid.name.value.to_string(),
                        Some(qid.qualifier.value.to_string()),
                    ))
                } else {
                    None
                }
            }
            // Unqualified identifiers could be outer, but we can't be sure without schema
            _ => None,
        }
    }

    /// Check if an expression contains any outer column references.
    /// Returns true if the expression references columns from tables not in inner_tables.
    fn expression_has_outer_reference(expr: &Expression, inner_tables: &[String]) -> bool {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                let table = qid.qualifier.value_lower.as_str();
                // If NOT in inner_tables, it's an outer reference
                !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table))
            }
            Expression::Infix(infix) => {
                Self::expression_has_outer_reference(&infix.left, inner_tables)
                    || Self::expression_has_outer_reference(&infix.right, inner_tables)
            }
            Expression::Prefix(prefix) => {
                Self::expression_has_outer_reference(&prefix.right, inner_tables)
            }
            Expression::FunctionCall(func) => func
                .arguments
                .iter()
                .any(|arg| Self::expression_has_outer_reference(arg, inner_tables)),
            Expression::In(in_expr) => {
                Self::expression_has_outer_reference(&in_expr.left, inner_tables)
                    || Self::expression_has_outer_reference(&in_expr.right, inner_tables)
            }
            Expression::Between(between) => {
                Self::expression_has_outer_reference(&between.expr, inner_tables)
                    || Self::expression_has_outer_reference(&between.lower, inner_tables)
                    || Self::expression_has_outer_reference(&between.upper, inner_tables)
            }
            Expression::Case(case) => {
                case.value
                    .as_ref()
                    .map(|op| Self::expression_has_outer_reference(op, inner_tables))
                    .unwrap_or(false)
                    || case.when_clauses.iter().any(|wc| {
                        Self::expression_has_outer_reference(&wc.condition, inner_tables)
                            || Self::expression_has_outer_reference(&wc.then_result, inner_tables)
                    })
                    || case
                        .else_value
                        .as_ref()
                        .map(|el| Self::expression_has_outer_reference(el, inner_tables))
                        .unwrap_or(false)
            }
            // Cast and Aliased have inner expressions
            Expression::Cast(cast) => {
                Self::expression_has_outer_reference(&cast.expr, inner_tables)
            }
            Expression::Aliased(aliased) => {
                Self::expression_has_outer_reference(&aliased.expression, inner_tables)
            }
            // Like has left expression (pattern is usually a literal)
            Expression::Like(like) => {
                Self::expression_has_outer_reference(&like.left, inner_tables)
                    || Self::expression_has_outer_reference(&like.pattern, inner_tables)
            }
            // Expression lists can contain outer references
            Expression::ExpressionList(list) => list
                .expressions
                .iter()
                .any(|e| Self::expression_has_outer_reference(e, inner_tables)),
            Expression::List(list) => list
                .elements
                .iter()
                .any(|e| Self::expression_has_outer_reference(e, inner_tables)),
            // InHashSet references a column (which could be qualified)
            Expression::InHashSet(_) => {
                // InHashSet.column is a String (column name), not an Expression
                // The column reference is already resolved, so no outer ref possible here
                false
            }
            // Window functions can have outer refs in partition/order expressions
            Expression::Window(window) => {
                window
                    .partition_by
                    .iter()
                    .any(|e| Self::expression_has_outer_reference(e, inner_tables))
                    || window.order_by.iter().any(|ob| {
                        Self::expression_has_outer_reference(&ob.expression, inner_tables)
                    })
            }
            // Literals and unqualified identifiers don't count as outer references
            Expression::Identifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_) => false,
            // Subqueries and other complex expressions - conservatively return true
            // (ScalarSubquery, Exists, AllAny, etc. could have correlated outer refs)
            _ => true,
        }
    }

    /// Strip table alias from column references in an expression.
    /// Converts "o.amount" to "amount", "t.name" to "name", etc.
    /// This is needed when evaluating predicates against rows from fetch_rows_by_ids,
    /// which uses unqualified column names from the table schema.
    fn strip_table_alias_from_expr(expr: &Expression) -> Expression {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                // Convert qualified identifier to unqualified
                Expression::Identifier(Identifier::new(
                    qid.name.token.clone(),
                    qid.name.value.clone(),
                ))
            }
            Expression::Infix(infix) => {
                // Recursively strip aliases from both sides
                Expression::Infix(InfixExpression::new(
                    infix.token.clone(),
                    Box::new(Self::strip_table_alias_from_expr(&infix.left)),
                    infix.operator.clone(),
                    Box::new(Self::strip_table_alias_from_expr(&infix.right)),
                ))
            }
            Expression::Prefix(prefix) => {
                // Recursively strip aliases from the inner expression
                Expression::Prefix(PrefixExpression {
                    token: prefix.token.clone(),
                    operator: prefix.operator.clone(),
                    op_type: prefix.op_type,
                    right: Box::new(Self::strip_table_alias_from_expr(&prefix.right)),
                })
            }
            Expression::FunctionCall(func) => {
                // Recursively strip aliases from function arguments
                let new_args: Vec<Expression> = func
                    .arguments
                    .iter()
                    .map(Self::strip_table_alias_from_expr)
                    .collect();
                Expression::FunctionCall(Box::new(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: new_args,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                }))
            }
            // For other expressions, return as-is
            _ => expr.clone(),
        }
    }

    /// Convert ALL/ANY expression to an equivalent expression that the evaluator can handle.
    ///
    /// This executes the subquery once and converts:
    /// - `x = ANY (values)` → `x IN (values)`
    /// - `x <> ALL (values)` → `x NOT IN (values)`
    /// - `x op ANY (values)` → `x op v1 OR x op v2 OR ...` (or optimized MIN/MAX)
    /// - `x op ALL (values)` → `x op v1 AND x op v2 AND ...` (or optimized MIN/MAX)
    fn convert_all_any_to_expression(
        &self,
        all_any: &AllAnyExpression,
        values: Vec<crate::core::Value>,
    ) -> Result<Expression> {
        use crate::parser::ast::AllAnyType;

        let op = all_any.operator.as_str();

        // Handle empty result set
        if values.is_empty() {
            return match all_any.all_any_type {
                AllAnyType::All => {
                    // ALL with empty set is vacuously TRUE
                    Ok(Expression::BooleanLiteral(BooleanLiteral {
                        token: dummy_token("TRUE", TokenType::Keyword),
                        value: true,
                    }))
                }
                AllAnyType::Any => {
                    // ANY with empty set is FALSE (no value satisfies the condition)
                    Ok(Expression::BooleanLiteral(BooleanLiteral {
                        token: dummy_token("FALSE", TokenType::Keyword),
                        value: false,
                    }))
                }
            };
        }

        // Convert values to expressions
        let value_exprs: Vec<Expression> = values.iter().map(value_to_expression).collect();

        // Special case: = ANY is equivalent to IN
        if op == "=" && matches!(all_any.all_any_type, AllAnyType::Any) {
            return Ok(Expression::In(InExpression {
                token: all_any.token.clone(),
                left: all_any.left.clone(),
                right: Box::new(Expression::ExpressionList(Box::new(ExpressionList {
                    token: dummy_token("(", TokenType::Punctuator),
                    expressions: value_exprs,
                }))),
                not: false,
            }));
        }

        // Special case: <> ALL is equivalent to NOT IN
        if (op == "<>" || op == "!=") && matches!(all_any.all_any_type, AllAnyType::All) {
            return Ok(Expression::In(InExpression {
                token: all_any.token.clone(),
                left: all_any.left.clone(),
                right: Box::new(Expression::ExpressionList(Box::new(ExpressionList {
                    token: dummy_token("(", TokenType::Punctuator),
                    expressions: value_exprs,
                }))),
                not: true,
            }));
        }

        // For comparison operators, we can optimize using MIN/MAX:
        // - x > ALL (values) → x > MAX(values)
        // - x >= ALL (values) → x >= MAX(values)
        // - x < ALL (values) → x < MIN(values)
        // - x <= ALL (values) → x <= MIN(values)
        // - x > ANY (values) → x > MIN(values)
        // - x >= ANY (values) → x >= MIN(values)
        // - x < ANY (values) → x < MAX(values)
        // - x <= ANY (values) → x <= MAX(values)

        // Filter out NULL values for comparison
        let non_null_values: Vec<&crate::core::Value> =
            values.iter().filter(|v| !v.is_null()).collect();

        // If all values are NULL, result depends on semantics
        if non_null_values.is_empty() {
            // Comparison with all NULLs is UNKNOWN, which filters as FALSE
            return Ok(Expression::BooleanLiteral(BooleanLiteral {
                token: dummy_token("FALSE", TokenType::Keyword),
                value: false,
            }));
        }

        // Find min and max
        let min_val = non_null_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();
        let max_val = non_null_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        let comparison_value = match (all_any.all_any_type, op) {
            // ALL with > or >= needs MAX (must be greater than all = greater than max)
            (AllAnyType::All, ">") | (AllAnyType::All, ">=") => max_val,
            // ALL with < or <= needs MIN (must be less than all = less than min)
            (AllAnyType::All, "<") | (AllAnyType::All, "<=") => min_val,
            // ANY with > or >= needs MIN (greater than at least one = greater than min)
            (AllAnyType::Any, ">") | (AllAnyType::Any, ">=") => min_val,
            // ANY with < or <= needs MAX (less than at least one = less than max)
            (AllAnyType::Any, "<") | (AllAnyType::Any, "<=") => max_val,
            // For other cases, fall back to building compound expression
            _ => None,
        };

        if let Some(cmp_val) = comparison_value {
            // Build simple comparison: left op value
            return Ok(Expression::Infix(InfixExpression::new(
                all_any.token.clone(),
                all_any.left.clone(),
                op.to_string(),
                Box::new(value_to_expression(cmp_val)),
            )));
        }

        // Fallback: build compound expression with AND/OR
        // This handles cases like = ALL (must equal all values, which is only possible if all same)
        let logical_op = match all_any.all_any_type {
            AllAnyType::All => "AND",
            AllAnyType::Any => "OR",
        };

        // Build: (left op v1) AND/OR (left op v2) AND/OR ...
        let mut result_expr: Option<Expression> = None;

        for value_expr in value_exprs {
            let comparison = Expression::Infix(InfixExpression::new(
                all_any.token.clone(),
                all_any.left.clone(),
                op.to_string(),
                Box::new(value_expr),
            ));

            result_expr = Some(match result_expr {
                None => comparison,
                Some(prev) => Expression::Infix(InfixExpression::new(
                    all_any.token.clone(),
                    Box::new(prev),
                    logical_op.to_string(),
                    Box::new(comparison),
                )),
            });
        }

        Ok(result_expr.unwrap_or_else(|| {
            Expression::BooleanLiteral(BooleanLiteral {
                token: dummy_token("TRUE", TokenType::Keyword),
                value: true,
            })
        }))
    }

    /// Execute a scalar subquery and return its single value.
    /// For non-correlated subqueries (no outer row context), results are cached
    /// to avoid re-execution when the same subquery appears multiple times.
    fn execute_scalar_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<crate::core::Value> {
        // Check if this is a non-correlated subquery (no outer row context)
        // Non-correlated subqueries can be cached since they return the same result
        let is_non_correlated =
            ctx.outer_row().is_none() && !Self::is_subquery_correlated(subquery);

        // For non-correlated subqueries, check cache first using SQL string as key
        let cache_key = if is_non_correlated {
            let key = subquery.to_string();
            if let Some(cached_value) = get_cached_scalar_subquery(&key) {
                return Ok(cached_value);
            }
            Some(key)
        } else {
            None
        };

        // OPTIMIZATION: For correlated scalar subqueries with LIMIT, index-based is faster
        // because it only checks rows for the limited outer rows.
        // Batch aggregate is faster for large outer result sets (no LIMIT or large LIMIT).
        if !is_non_correlated {
            // First check if batch aggregate cache already exists (O(1) lookup)
            if let Some(value) = Self::try_lookup_batch_aggregate(subquery, ctx) {
                return Ok(value);
            }

            // Try index-based COUNT first (faster for LIMIT queries)
            if let Some(count) = self.try_execute_scalar_count_with_index(subquery, ctx)? {
                return Ok(crate::core::Value::Integer(count));
            }

            // Fall back to batch aggregate for cases index doesn't handle
            if let Some(batch_cache) = self.try_execute_and_cache_batch_aggregate(subquery, ctx)? {
                if let Some(value) = Self::try_lookup_batch_aggregate(subquery, ctx) {
                    return Ok(value);
                }
                if Self::is_count_expression(&subquery.columns[0]) {
                    return Ok(Value::Integer(0));
                }
                drop(batch_cache);
            }
        }

        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Get the first row
        if !result.next() {
            let null_value = crate::core::Value::null_unknown();
            // Cache the result for non-correlated subqueries
            if let Some(key) = cache_key {
                cache_scalar_subquery(
                    key,
                    extract_table_names_for_cache(subquery),
                    null_value.clone(),
                );
            }
            return Ok(null_value);
        }

        let row = result.take_row();
        // take_first_value() is more efficient than get(0).cloned()
        let first_value = match row.take_first_value() {
            Some(v) => v,
            None => {
                let null_value = crate::core::Value::null_unknown();
                if let Some(key) = cache_key {
                    cache_scalar_subquery(
                        key,
                        extract_table_names_for_cache(subquery),
                        null_value.clone(),
                    );
                }
                return Ok(null_value);
            }
        };

        // Check that there's only one row (scalar subquery should return single value)
        if result.next() {
            return Err(Error::Internal {
                message: "scalar subquery returned more than one row".to_string(),
            });
        }

        // Cache the result for non-correlated subqueries
        if let Some(key) = cache_key {
            cache_scalar_subquery(
                key,
                extract_table_names_for_cache(subquery),
                first_value.clone(),
            );
        }

        Ok(first_value)
    }

    /// Execute an IN subquery and return its values.
    /// For non-correlated subqueries (no outer row context), results are cached
    /// to avoid re-execution when the same subquery appears multiple times.
    fn execute_in_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Vec<crate::core::Value>> {
        // Check if this is a non-correlated subquery (no outer row context)
        // Non-correlated subqueries can be cached since they return the same result
        let is_non_correlated =
            ctx.outer_row().is_none() && !Self::is_subquery_correlated(subquery);

        // For non-correlated subqueries, check cache first using SQL string as key
        let cache_key = if is_non_correlated {
            let key = subquery.to_string();
            if let Some(cached_values) = get_cached_in_subquery(&key) {
                return Ok(cached_values);
            }
            Some(key)
        } else {
            None
        };

        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Collect all values from the first column - use take_row() to avoid cloning
        let mut values = Vec::new();
        while result.next() {
            let row = result.take_row();
            // take_first_value() is more efficient than into_values().swap_remove(0)
            if let Some(value) = row.take_first_value() {
                values.push(value);
            }
        }

        // Cache the result for non-correlated subqueries
        if let Some(key) = cache_key {
            cache_in_subquery(key, extract_table_names_for_cache(subquery), values.clone());
        }

        Ok(values)
    }

    /// Execute an IN subquery and return all rows (for multi-column IN)
    fn execute_in_subquery_rows(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Vec<crate::core::Value>>> {
        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Collect all values from all columns - use take_row() to avoid cloning
        let mut rows = Vec::new();
        while result.next() {
            let row = result.take_row();
            if !row.is_empty() {
                // into_values() uses Arc::try_unwrap() to move without cloning when sole owner
                rows.push(row.into_values());
            }
        }

        Ok(rows)
    }

    /// Check if an expression contains EXISTS or other subqueries that need processing
    pub(crate) fn has_subqueries(expr: &Expression) -> bool {
        match expr {
            Expression::Exists(_) => true,
            Expression::ScalarSubquery(_) => true,
            Expression::AllAny(_) => true,
            Expression::Prefix(prefix) => Self::has_subqueries(&prefix.right),
            Expression::Infix(infix) => {
                Self::has_subqueries(&infix.left) || Self::has_subqueries(&infix.right)
            }
            Expression::In(in_expr) => {
                Self::has_subqueries(&in_expr.left)
                    || matches!(in_expr.right.as_ref(), Expression::ScalarSubquery(_))
            }
            Expression::Between(between) => {
                Self::has_subqueries(&between.expr)
                    || Self::has_subqueries(&between.lower)
                    || Self::has_subqueries(&between.upper)
            }
            Expression::Aliased(aliased) => Self::has_subqueries(&aliased.expression),
            Expression::FunctionCall(func) => func.arguments.iter().any(Self::has_subqueries),
            _ => false,
        }
    }

    /// Process subqueries in SELECT column expressions (single-pass optimization)
    ///
    /// Returns `None` if no subqueries were found (caller should use original columns).
    /// Returns `Some(processed)` if any subqueries were found and processed.
    ///
    /// This combines the check and processing into a single traversal to avoid
    /// walking the expression tree twice.
    pub(crate) fn try_process_select_subqueries(
        &self,
        columns: &[Expression],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<Expression>>> {
        let mut result: Option<Vec<Expression>> = None;

        for (i, col) in columns.iter().enumerate() {
            if let Some(processed) = self.try_process_expression_subqueries(col, ctx)? {
                // Lazily initialize result vec, copying prior columns
                let vec = result.get_or_insert_with(|| columns[..i].to_vec());
                vec.push(processed);
            } else if let Some(ref mut vec) = result {
                // No subquery in this column, but we're already building a new vec
                vec.push(col.clone());
            }
            // If result is None and no subquery found, do nothing (use original)
        }

        Ok(result)
    }

    /// Try to process subqueries in an expression (single-pass optimization)
    ///
    /// Returns `None` if no subqueries were found (expression unchanged).
    /// Returns `Some(processed)` if any subqueries were found and processed.
    fn try_process_expression_subqueries(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<Option<Expression>> {
        match expr {
            Expression::ScalarSubquery(subquery) => {
                // Execute scalar subquery and replace with literal value
                let value = self.execute_scalar_subquery(&subquery.subquery, ctx)?;
                Ok(Some(value_to_expression(&value)))
            }

            Expression::Exists(exists) => {
                // Execute EXISTS subquery and replace with boolean literal
                let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                Ok(Some(Expression::BooleanLiteral(BooleanLiteral {
                    token: dummy_token(
                        if exists_result { "TRUE" } else { "FALSE" },
                        TokenType::Keyword,
                    ),
                    value: exists_result,
                })))
            }

            Expression::Aliased(aliased) => {
                // Only create new expression if inner has subqueries
                if let Some(processed) =
                    self.try_process_expression_subqueries(&aliased.expression, ctx)?
                {
                    Ok(Some(Expression::Aliased(AliasedExpression {
                        token: aliased.token.clone(),
                        expression: Box::new(processed),
                        alias: aliased.alias.clone(),
                    })))
                } else {
                    Ok(None)
                }
            }

            Expression::Infix(infix) => {
                // Process both sides, only create new expression if either changed
                let left = self.try_process_expression_subqueries(&infix.left, ctx)?;
                let right = self.try_process_expression_subqueries(&infix.right, ctx)?;

                if left.is_some() || right.is_some() {
                    Ok(Some(Expression::Infix(InfixExpression {
                        token: infix.token.clone(),
                        left: Box::new(left.unwrap_or_else(|| (*infix.left).clone())),
                        operator: infix.operator.clone(),
                        op_type: infix.op_type,
                        right: Box::new(right.unwrap_or_else(|| (*infix.right).clone())),
                    })))
                } else {
                    Ok(None)
                }
            }

            Expression::Prefix(prefix) => {
                if let Some(processed) =
                    self.try_process_expression_subqueries(&prefix.right, ctx)?
                {
                    Ok(Some(Expression::Prefix(PrefixExpression {
                        token: prefix.token.clone(),
                        operator: prefix.operator.clone(),
                        op_type: prefix.op_type,
                        right: Box::new(processed),
                    })))
                } else {
                    Ok(None)
                }
            }

            Expression::FunctionCall(func) => {
                // Process arguments, only create new expression if any changed
                let mut any_changed = false;
                let mut processed_args: Vec<Option<Expression>> =
                    Vec::with_capacity(func.arguments.len());

                for arg in &func.arguments {
                    let processed = self.try_process_expression_subqueries(arg, ctx)?;
                    if processed.is_some() {
                        any_changed = true;
                    }
                    processed_args.push(processed);
                }

                if any_changed {
                    let final_args: Vec<Expression> = func
                        .arguments
                        .iter()
                        .zip(processed_args)
                        .map(|(orig, processed)| processed.unwrap_or_else(|| orig.clone()))
                        .collect();

                    Ok(Some(Expression::FunctionCall(Box::new(FunctionCall {
                        token: func.token.clone(),
                        function: func.function.clone(),
                        arguments: final_args,
                        is_distinct: func.is_distinct,
                        order_by: func.order_by.clone(),
                        filter: func.filter.clone(),
                    }))))
                } else {
                    Ok(None)
                }
            }

            Expression::Case(case) => {
                // Process CASE expression to handle subqueries in any part
                let mut any_changed = false;

                // Process the operand (if present)
                let processed_value = if let Some(ref value) = case.value {
                    let processed = self.try_process_expression_subqueries(value, ctx)?;
                    if processed.is_some() {
                        any_changed = true;
                    }
                    processed.map(Box::new)
                } else {
                    None
                };

                // Process each WHEN clause
                let mut processed_whens: Vec<(Option<Expression>, Option<Expression>)> =
                    Vec::with_capacity(case.when_clauses.len());
                for when in &case.when_clauses {
                    let cond = self.try_process_expression_subqueries(&when.condition, ctx)?;
                    let then = self.try_process_expression_subqueries(&when.then_result, ctx)?;
                    if cond.is_some() || then.is_some() {
                        any_changed = true;
                    }
                    processed_whens.push((cond, then));
                }

                // Process the ELSE clause (if present)
                let processed_else = if let Some(ref else_val) = case.else_value {
                    let processed = self.try_process_expression_subqueries(else_val, ctx)?;
                    if processed.is_some() {
                        any_changed = true;
                    }
                    processed.map(Box::new)
                } else {
                    None
                };

                if any_changed {
                    let final_whens: Vec<WhenClause> = case
                        .when_clauses
                        .iter()
                        .zip(processed_whens)
                        .map(|(orig, (cond, then))| WhenClause {
                            token: orig.token.clone(),
                            condition: cond.unwrap_or_else(|| orig.condition.clone()),
                            then_result: then.unwrap_or_else(|| orig.then_result.clone()),
                        })
                        .collect();

                    Ok(Some(Expression::Case(Box::new(CaseExpression {
                        token: case.token.clone(),
                        value: processed_value.or_else(|| case.value.clone()),
                        when_clauses: final_whens,
                        else_value: processed_else.or_else(|| case.else_value.clone()),
                    }))))
                } else {
                    Ok(None)
                }
            }

            Expression::Cast(cast) => {
                // Process inner expression for subqueries
                if let Some(processed) = self.try_process_expression_subqueries(&cast.expr, ctx)? {
                    Ok(Some(Expression::Cast(CastExpression {
                        token: cast.token.clone(),
                        expr: Box::new(processed),
                        type_name: cast.type_name.clone(),
                    })))
                } else {
                    Ok(None)
                }
            }

            Expression::AllAny(all_any) => {
                // Execute the subquery to get all values
                let values = self.execute_in_subquery(&all_any.subquery, ctx)?;

                // Convert ALL/ANY to an equivalent expression that the evaluator can handle
                Ok(Some(self.convert_all_any_to_expression(all_any, values)?))
            }

            // No subqueries possible in other expression types
            _ => Ok(None),
        }
    }

    // ============================================================================
    // Correlated Subquery Support
    // ============================================================================

    /// Check if an expression contains correlated subqueries that reference outer columns.
    /// A correlated subquery references columns from outer tables that are not defined
    /// in the subquery's own FROM clause.
    pub(crate) fn has_correlated_subqueries(expr: &Expression) -> bool {
        match expr {
            Expression::Exists(exists) => Self::is_subquery_correlated(&exists.subquery),
            Expression::ScalarSubquery(subquery) => {
                Self::is_subquery_correlated(&subquery.subquery)
            }
            Expression::Prefix(prefix) => {
                // Handle NOT EXISTS
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    return Self::is_subquery_correlated(&exists.subquery);
                }
                Self::has_correlated_subqueries(&prefix.right)
            }
            Expression::Infix(infix) => {
                Self::has_correlated_subqueries(&infix.left)
                    || Self::has_correlated_subqueries(&infix.right)
            }
            Expression::In(in_expr) => {
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    return Self::is_subquery_correlated(&subquery.subquery);
                }
                Self::has_correlated_subqueries(&in_expr.left)
            }
            Expression::Between(between) => {
                Self::has_correlated_subqueries(&between.expr)
                    || Self::has_correlated_subqueries(&between.lower)
                    || Self::has_correlated_subqueries(&between.upper)
            }
            Expression::Aliased(aliased) => Self::has_correlated_subqueries(&aliased.expression),
            Expression::FunctionCall(func) => {
                func.arguments.iter().any(Self::has_correlated_subqueries)
            }
            _ => false,
        }
    }

    /// Check if any SELECT column expressions contain correlated subqueries
    pub(crate) fn has_correlated_select_subqueries(columns: &[Expression]) -> bool {
        columns.iter().any(Self::has_correlated_subqueries)
    }

    /// Process a single expression with correlated subqueries, replacing scalar subqueries
    /// with their evaluated values using the provided outer row context.
    pub(crate) fn process_correlated_expression(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<Expression> {
        match expr {
            Expression::ScalarSubquery(subquery) => {
                // Execute scalar subquery with outer row context
                let value = self.execute_scalar_subquery(&subquery.subquery, ctx)?;
                Ok(value_to_expression(&value))
            }

            Expression::Exists(exists) => {
                // Execute EXISTS subquery with outer row context
                let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                Ok(Expression::BooleanLiteral(BooleanLiteral {
                    token: dummy_token(
                        if exists_result { "TRUE" } else { "FALSE" },
                        TokenType::Keyword,
                    ),
                    value: exists_result,
                }))
            }

            Expression::Aliased(aliased) => {
                let processed = self.process_correlated_expression(&aliased.expression, ctx)?;
                Ok(Expression::Aliased(AliasedExpression {
                    token: aliased.token.clone(),
                    expression: Box::new(processed),
                    alias: aliased.alias.clone(),
                }))
            }

            Expression::Infix(infix) => {
                let left = self.process_correlated_expression(&infix.left, ctx)?;
                let right = self.process_correlated_expression(&infix.right, ctx)?;
                Ok(Expression::Infix(InfixExpression {
                    token: infix.token.clone(),
                    left: Box::new(left),
                    operator: infix.operator.clone(),
                    op_type: infix.op_type,
                    right: Box::new(right),
                }))
            }

            Expression::Prefix(prefix) => {
                let right = self.process_correlated_expression(&prefix.right, ctx)?;
                Ok(Expression::Prefix(PrefixExpression {
                    token: prefix.token.clone(),
                    operator: prefix.operator.clone(),
                    op_type: prefix.op_type,
                    right: Box::new(right),
                }))
            }

            Expression::FunctionCall(func) => {
                let processed_args: Result<Vec<Expression>> = func
                    .arguments
                    .iter()
                    .map(|arg| self.process_correlated_expression(arg, ctx))
                    .collect();

                Ok(Expression::FunctionCall(Box::new(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: processed_args?,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                })))
            }

            Expression::In(in_expr) => {
                let processed_left = self.process_correlated_expression(&in_expr.left, ctx)?;

                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    // Use InHashSet for O(1) lookups with FxHash (optimized for trusted keys)
                    let values = self.execute_in_subquery(&subquery.subquery, ctx)?;
                    let hash_set: AHashSet<Value> = values.into_iter().collect();

                    return Ok(Expression::InHashSet(InHashSetExpression {
                        token: in_expr.token.clone(),
                        column: Box::new(processed_left),
                        values: CompactArc::new(hash_set),
                        not: in_expr.not,
                    }));
                }

                let processed_right = self.process_correlated_expression(&in_expr.right, ctx)?;
                Ok(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(processed_left),
                    right: Box::new(processed_right),
                    not: in_expr.not,
                }))
            }

            Expression::Between(between) => {
                let processed_expr = self.process_correlated_expression(&between.expr, ctx)?;
                let processed_lower = self.process_correlated_expression(&between.lower, ctx)?;
                let processed_upper = self.process_correlated_expression(&between.upper, ctx)?;

                Ok(Expression::Between(BetweenExpression {
                    token: between.token.clone(),
                    expr: Box::new(processed_expr),
                    not: between.not,
                    lower: Box::new(processed_lower),
                    upper: Box::new(processed_upper),
                }))
            }

            Expression::Case(case) => {
                let processed_value = if let Some(ref value) = case.value {
                    Some(Box::new(self.process_correlated_expression(value, ctx)?))
                } else {
                    None
                };

                let processed_whens: Result<Vec<WhenClause>> = case
                    .when_clauses
                    .iter()
                    .map(|when| {
                        Ok(WhenClause {
                            token: when.token.clone(),
                            condition: self.process_correlated_expression(&when.condition, ctx)?,
                            then_result: self
                                .process_correlated_expression(&when.then_result, ctx)?,
                        })
                    })
                    .collect();

                let processed_else = if let Some(ref else_val) = case.else_value {
                    Some(Box::new(self.process_correlated_expression(else_val, ctx)?))
                } else {
                    None
                };

                Ok(Expression::Case(Box::new(CaseExpression {
                    token: case.token.clone(),
                    value: processed_value,
                    when_clauses: processed_whens?,
                    else_value: processed_else,
                })))
            }

            Expression::Cast(cast) => {
                let processed_expr = self.process_correlated_expression(&cast.expr, ctx)?;
                Ok(Expression::Cast(CastExpression {
                    token: cast.token.clone(),
                    expr: Box::new(processed_expr),
                    type_name: cast.type_name.clone(),
                }))
            }

            // For all other expression types, return as-is
            _ => Ok(expr.clone()),
        }
    }

    /// Check if a subquery is correlated (references outer columns)
    pub(crate) fn is_subquery_correlated(subquery: &SelectStatement) -> bool {
        // Get table/alias names defined in the subquery's FROM clause
        let subquery_tables = Self::collect_subquery_table_columns(subquery);

        // Check if the WHERE clause references any outer columns
        if let Some(ref where_clause) = subquery.where_clause {
            if Self::references_outer_columns(where_clause, &subquery_tables) {
                return true;
            }
        }

        // Also check SELECT expressions (for scalar subqueries)
        for col in &subquery.columns {
            if Self::references_outer_columns(col, &subquery_tables) {
                return true;
            }
        }

        false
    }

    /// Collect table/alias names from a subquery's FROM clause
    fn collect_subquery_table_columns(subquery: &SelectStatement) -> Vec<String> {
        let mut tables = Vec::new();

        if let Some(ref table_expr) = subquery.table_expr {
            Self::collect_table_names_from_source(table_expr, &mut tables);
        }

        tables
    }

    /// Recursively collect table names from a table source expression
    fn collect_table_names_from_source(source: &Expression, tables: &mut Vec<String>) {
        match source {
            Expression::TableSource(ts) => {
                // When a table has an alias, SQL semantics require using the alias,
                // not the original table name. So for `FROM t t2`, only `t2` is valid.
                // If there's no alias, use the table name.
                if let Some(ref alias) = ts.alias {
                    tables.push(alias.value_lower.to_string());
                } else {
                    tables.push(ts.name.value_lower.to_string());
                }
            }
            Expression::JoinSource(js) => {
                Self::collect_table_names_from_source(&js.left, tables);
                Self::collect_table_names_from_source(&js.right, tables);
            }
            Expression::SubquerySource(ss) => {
                // Subquery source has an optional alias
                if let Some(ref alias) = ss.alias {
                    tables.push(alias.value_lower.to_string());
                }
            }
            _ => {}
        }
    }

    /// Check if an expression references columns from outer scope.
    ///
    /// For simple identifiers, we cannot reliably determine if they reference outer columns
    /// since the same column name might exist in both inner and outer scopes. The inner
    /// scope takes precedence per SQL semantics, so simple identifiers are NOT considered
    /// outer references (they will resolve to inner scope if available).
    ///
    /// For qualified identifiers (e.g., c.id), we check if the qualifier (table/alias)
    /// is NOT defined in the subquery's FROM clause - if so, it must be an outer reference.
    fn references_outer_columns(expr: &Expression, subquery_tables: &[String]) -> bool {
        match expr {
            Expression::Identifier(_id) => {
                // Simple identifiers are ambiguous - they resolve to inner scope first per SQL semantics.
                // We cannot determine if this is an outer reference without knowing the inner schema.
                // Conservative approach: don't mark as correlated based on simple identifiers alone.
                // Users should use qualified names (e.g., c.id) for outer references in correlated subqueries.
                false
            }
            Expression::QualifiedIdentifier(qid) => {
                // Qualified identifier like "c.id" or "outer_table.column"
                let table_name = &qid.qualifier.value_lower;

                // If the table/alias is NOT in subquery tables, it's an outer reference
                // This is the key check: if "c" is not in ["orders", "o"], then c.id is outer
                !subquery_tables
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(table_name))
            }
            Expression::Infix(infix) => {
                Self::references_outer_columns(&infix.left, subquery_tables)
                    || Self::references_outer_columns(&infix.right, subquery_tables)
            }
            Expression::Prefix(prefix) => {
                Self::references_outer_columns(&prefix.right, subquery_tables)
            }
            Expression::FunctionCall(func) => func
                .arguments
                .iter()
                .any(|arg| Self::references_outer_columns(arg, subquery_tables)),
            Expression::In(in_expr) => {
                Self::references_outer_columns(&in_expr.left, subquery_tables)
                    || Self::references_outer_columns(&in_expr.right, subquery_tables)
            }
            Expression::Between(between) => {
                Self::references_outer_columns(&between.expr, subquery_tables)
                    || Self::references_outer_columns(&between.lower, subquery_tables)
                    || Self::references_outer_columns(&between.upper, subquery_tables)
            }
            Expression::Case(case) => {
                if let Some(ref value) = case.value {
                    if Self::references_outer_columns(value, subquery_tables) {
                        return true;
                    }
                }
                for when in &case.when_clauses {
                    if Self::references_outer_columns(&when.condition, subquery_tables)
                        || Self::references_outer_columns(&when.then_result, subquery_tables)
                    {
                        return true;
                    }
                }
                if let Some(ref else_val) = case.else_value {
                    if Self::references_outer_columns(else_val, subquery_tables) {
                        return true;
                    }
                }
                false
            }
            Expression::Aliased(aliased) => {
                Self::references_outer_columns(&aliased.expression, subquery_tables)
            }
            Expression::Cast(cast) => Self::references_outer_columns(&cast.expr, subquery_tables),
            _ => false,
        }
    }

    /// Process WHERE clause with correlated subqueries for a specific outer row.
    /// This evaluates correlated subqueries using the outer row context.
    pub(crate) fn process_correlated_where(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<Expression> {
        match expr {
            Expression::Exists(exists) => {
                // Execute EXISTS with outer row context
                let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                Ok(Expression::BooleanLiteral(BooleanLiteral {
                    token: dummy_token(
                        if exists_result { "TRUE" } else { "FALSE" },
                        TokenType::Keyword,
                    ),
                    value: exists_result,
                }))
            }

            Expression::Prefix(prefix) => {
                // Handle NOT EXISTS
                if prefix.operator.eq_ignore_ascii_case("NOT") {
                    if let Expression::Exists(exists) = prefix.right.as_ref() {
                        let exists_result = self.execute_exists_subquery(&exists.subquery, ctx)?;
                        return Ok(Expression::BooleanLiteral(BooleanLiteral {
                            token: dummy_token(
                                if !exists_result { "TRUE" } else { "FALSE" },
                                TokenType::Keyword,
                            ),
                            value: !exists_result,
                        }));
                    }
                }

                let processed_right = self.process_correlated_where(&prefix.right, ctx)?;
                Ok(Expression::Prefix(PrefixExpression {
                    token: prefix.token.clone(),
                    operator: prefix.operator.clone(),
                    op_type: prefix.op_type,
                    right: Box::new(processed_right),
                }))
            }

            Expression::Infix(infix) => {
                let processed_left = self.process_correlated_where(&infix.left, ctx)?;
                let processed_right = self.process_correlated_where(&infix.right, ctx)?;

                Ok(Expression::Infix(InfixExpression {
                    token: infix.token.clone(),
                    left: Box::new(processed_left),
                    operator: infix.operator.clone(),
                    op_type: infix.op_type,
                    right: Box::new(processed_right),
                }))
            }

            Expression::ScalarSubquery(subquery) => {
                // Execute scalar subquery with outer row context
                let value = self.execute_scalar_subquery(&subquery.subquery, ctx)?;
                Ok(value_to_expression(&value))
            }

            Expression::In(in_expr) => {
                let processed_left = self.process_correlated_where(&in_expr.left, ctx)?;

                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    // Use InHashSet for O(1) lookups with FxHash (optimized for trusted keys)
                    let values = self.execute_in_subquery(&subquery.subquery, ctx)?;
                    let hash_set: AHashSet<Value> = values.into_iter().collect();

                    return Ok(Expression::InHashSet(InHashSetExpression {
                        token: in_expr.token.clone(),
                        column: Box::new(processed_left),
                        values: CompactArc::new(hash_set),
                        not: in_expr.not,
                    }));
                }

                Ok(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(processed_left),
                    right: in_expr.right.clone(),
                    not: in_expr.not,
                }))
            }

            Expression::Between(between) => {
                let processed_expr = self.process_correlated_where(&between.expr, ctx)?;
                let processed_lower = self.process_correlated_where(&between.lower, ctx)?;
                let processed_upper = self.process_correlated_where(&between.upper, ctx)?;

                Ok(Expression::Between(BetweenExpression {
                    token: between.token.clone(),
                    expr: Box::new(processed_expr),
                    not: between.not,
                    lower: Box::new(processed_lower),
                    upper: Box::new(processed_upper),
                }))
            }

            // For all other expression types, return as-is
            _ => Ok(expr.clone()),
        }
    }

    // ============================================================================
    // Semi-Join Optimization Methods
    // ============================================================================

    /// Try to extract semi-join information from a correlated EXISTS subquery.
    ///
    /// For semi-join optimization, we need:
    /// 1. A simple table source (no joins in subquery)
    /// 2. A WHERE clause with `inner.col = outer.col` equality
    /// 3. Optional additional non-correlated predicates
    ///
    /// Returns None if the subquery cannot be optimized as a semi-join.
    pub fn try_extract_semi_join_info(
        exists: &ExistsExpression,
        is_negated: bool,
        outer_tables: &[String],
    ) -> Option<SemiJoinInfo> {
        let subquery = &exists.subquery;

        // 1. Check for simple table source (not a join)
        let (inner_table, inner_alias): (String, Option<String>) =
            match subquery.table_expr.as_ref().map(|b| b.as_ref()) {
                Some(Expression::TableSource(ts)) => {
                    let alias = ts.alias.as_ref().map(|a| a.value.to_string());
                    (ts.name.value.to_string(), alias)
                }
                _ => return None, // Can't optimize subquery joins or derived tables
            };

        // 2. Parse WHERE clause to find correlation condition
        let where_clause = subquery.where_clause.as_ref()?;

        // Get inner table identifiers for distinguishing inner vs outer references
        let inner_table_lower: String = inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase());
        let inner_tables = vec![inner_table_lower.to_lowercase()];

        // Try to extract: outer.col = inner.col (or inner.col = outer.col)
        let extraction =
            Self::extract_equality_correlation(where_clause, outer_tables, &inner_tables);
        let (outer_col, outer_tbl, inner_col, remaining) = extraction?;

        // IMPORTANT: Check if the remaining predicates reference outer tables.
        // If they do, we cannot use semi-join optimization because those predicates
        // cannot be evaluated on the inner table alone.
        // Example: WHERE o.customer_id = c.id AND c.country = 'USA'
        // The "c.country = 'USA'" references outer table and can't be pushed to inner query.
        if let Some(ref rem) = remaining {
            if Self::expression_references_outer_tables(rem.as_ref(), outer_tables, &inner_tables) {
                return None;
            }
        }

        Some(SemiJoinInfo {
            outer_column: outer_col,
            outer_table: outer_tbl,
            inner_column: inner_col,
            inner_table,
            inner_alias,
            non_correlated_where: remaining,
            is_negated,
        })
    }

    /// Extract an equality correlation from a WHERE clause.
    ///
    /// Looks for patterns like:
    /// - `o.user_id = u.id` → inner_col="user_id", outer_col="id", outer_table="u"
    /// - `o.user_id = u.id AND o.amount > 500` → same, with remaining predicate
    ///
    /// Returns: (outer_column, outer_table, inner_column, remaining_predicates)
    /// Uses Arc<Expression> for remaining predicates to avoid cloning expression trees.
    fn extract_equality_correlation(
        expr: &Expression,
        outer_tables: &[String],
        inner_tables: &[String],
    ) -> Option<CorrelationExtraction> {
        match expr {
            // Direct equality: inner.col = outer.col
            Expression::Infix(infix) if infix.operator == "=" => Self::try_extract_equality_pair(
                &infix.left,
                &infix.right,
                outer_tables,
                inner_tables,
            )
            .map(|(outer_col, outer_tbl, inner_col)| (outer_col, outer_tbl, inner_col, None)),

            // AND expression: look for equality in one branch
            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try left side first
                if let Some((outer_col, outer_tbl, inner_col, left_remaining)) =
                    Self::extract_equality_correlation(&infix.left, outer_tables, inner_tables)
                {
                    // Combine remaining from left with right (Arc avoids later clones)
                    let remaining = Self::combine_and_predicates_arc(left_remaining, &infix.right);
                    return Some((outer_col, outer_tbl, inner_col, remaining));
                }

                // Try right side
                if let Some((outer_col, outer_tbl, inner_col, right_remaining)) =
                    Self::extract_equality_correlation(&infix.right, outer_tables, inner_tables)
                {
                    // Combine left with remaining from right (Arc avoids later clones)
                    let remaining = Self::combine_and_predicates_arc(right_remaining, &infix.left);
                    return Some((outer_col, outer_tbl, inner_col, remaining));
                }

                None
            }

            _ => None,
        }
    }

    /// Try to extract an equality pair from two expressions.
    /// One should reference outer table, one should reference inner table.
    fn try_extract_equality_pair(
        left: &Expression,
        right: &Expression,
        outer_tables: &[String],
        inner_tables: &[String],
    ) -> Option<(String, Option<String>, String)> {
        // Try left=outer, right=inner
        if let (Some((outer_col, outer_tbl)), Some(inner_col)) = (
            Self::extract_outer_column(left, outer_tables, inner_tables),
            Self::extract_inner_column(right, inner_tables),
        ) {
            return Some((outer_col, outer_tbl, inner_col));
        }

        // Try left=inner, right=outer
        if let (Some(inner_col), Some((outer_col, outer_tbl))) = (
            Self::extract_inner_column(left, inner_tables),
            Self::extract_outer_column(right, outer_tables, inner_tables),
        ) {
            return Some((outer_col, outer_tbl, inner_col));
        }

        None
    }

    /// Extract column name if expression references outer table.
    /// Returns (column_name, table_alias) where table_alias may be None.
    fn extract_outer_column(
        expr: &Expression,
        outer_tables: &[String],
        inner_tables: &[String],
    ) -> Option<(String, Option<String>)> {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                // Use pre-computed value_lower to avoid allocation
                let table = qid.qualifier.value_lower.as_str();
                // Must be in outer tables and NOT in inner tables
                if outer_tables.iter().any(|t| t.eq_ignore_ascii_case(table))
                    && !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table))
                {
                    Some((
                        qid.name.value.to_string(),
                        Some(qid.qualifier.value.to_string()),
                    ))
                } else {
                    None
                }
            }
            // Simple identifier could be outer if not inner table column
            // But we can't reliably determine this without schema info
            _ => None,
        }
    }

    /// Extract column name if expression references inner table.
    fn extract_inner_column(expr: &Expression, inner_tables: &[String]) -> Option<String> {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                // Use pre-computed value_lower to avoid allocation
                let table = qid.qualifier.value_lower.as_str();
                if inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table)) {
                    Some(qid.name.value.to_string())
                } else {
                    None
                }
            }
            Expression::Identifier(id) => {
                // Unqualified identifier - assume it's inner table column
                // This is safe because outer refs should be qualified in correlated subqueries
                Some(id.value.to_string())
            }
            _ => None,
        }
    }

    /// Check if an expression references any outer tables.
    /// Used to determine if a predicate can be pushed to the inner query in semi-join optimization.
    fn expression_references_outer_tables(
        expr: &Expression,
        outer_tables: &[String],
        inner_tables: &[String],
    ) -> bool {
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                // Use pre-computed value_lower to avoid allocation
                let table = &qid.qualifier.value_lower;
                // References outer if it's in outer_tables and NOT in inner_tables
                outer_tables.iter().any(|t| t.eq_ignore_ascii_case(table))
                    && !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table))
            }
            Expression::Infix(infix) => {
                Self::expression_references_outer_tables(&infix.left, outer_tables, inner_tables)
                    || Self::expression_references_outer_tables(
                        &infix.right,
                        outer_tables,
                        inner_tables,
                    )
            }
            Expression::Prefix(prefix) => {
                Self::expression_references_outer_tables(&prefix.right, outer_tables, inner_tables)
            }
            Expression::FunctionCall(func) => func.arguments.iter().any(|arg| {
                Self::expression_references_outer_tables(arg, outer_tables, inner_tables)
            }),
            Expression::In(in_expr) => {
                Self::expression_references_outer_tables(&in_expr.left, outer_tables, inner_tables)
                    || Self::expression_references_outer_tables(
                        &in_expr.right,
                        outer_tables,
                        inner_tables,
                    )
            }
            Expression::Between(between) => {
                Self::expression_references_outer_tables(&between.expr, outer_tables, inner_tables)
                    || Self::expression_references_outer_tables(
                        &between.lower,
                        outer_tables,
                        inner_tables,
                    )
                    || Self::expression_references_outer_tables(
                        &between.upper,
                        outer_tables,
                        inner_tables,
                    )
            }
            Expression::Case(case) => {
                case.value.as_ref().is_some_and(|op| {
                    Self::expression_references_outer_tables(
                        op.as_ref(),
                        outer_tables,
                        inner_tables,
                    )
                }) || case.when_clauses.iter().any(|wc| {
                    Self::expression_references_outer_tables(
                        &wc.condition,
                        outer_tables,
                        inner_tables,
                    ) || Self::expression_references_outer_tables(
                        &wc.then_result,
                        outer_tables,
                        inner_tables,
                    )
                }) || case.else_value.as_ref().is_some_and(|el| {
                    Self::expression_references_outer_tables(
                        el.as_ref(),
                        outer_tables,
                        inner_tables,
                    )
                })
            }
            // Literals and other expressions don't reference tables
            _ => false,
        }
    }

    /// Combine two optional predicates with AND.
    /// Returns Arc<Expression> to avoid cloning when the result is used multiple times.
    fn combine_and_predicates_arc(
        left: Option<Arc<Expression>>,
        right: &Expression,
    ) -> Option<Arc<Expression>> {
        match left {
            None => Some(Arc::new(right.clone())),
            Some(l) => {
                // Unwrap Arc if we're the only owner, otherwise clone
                let left_expr = Arc::try_unwrap(l).unwrap_or_else(|arc| (*arc).clone());
                Some(Arc::new(Expression::Infix(InfixExpression {
                    token: dummy_token_clone(),
                    left: Box::new(left_expr),
                    operator: "AND".into(),
                    op_type: InfixOperator::And,
                    right: Box::new(right.clone()),
                })))
            }
        }
    }

    /// Check if index-nested-loop would be more efficient than semi-join for EXISTS.
    ///
    /// Returns true if:
    /// 1. There's a small LIMIT (< 500)
    /// 2. Inner table has index on correlation column
    ///
    /// With small LIMIT and early termination at the outer level, per-row EXISTS
    /// evaluation is faster because:
    /// - O(LIMIT × log(inner_size)) for index probe vs O(inner_size) for hash build
    /// - Example: LIMIT 100, inner=30K → 100×15=1500 ops vs 30000 ops
    ///
    /// For EXISTS with additional predicate (e.g., EXISTS ... WHERE o.user_id = u.id AND o.amount > 500):
    /// - Index lookup gets candidate row_ids for correlation
    /// - Rows are fetched in batches and predicate is evaluated with early exit
    /// - This is O(LIMIT × avg_rows_per_key × predicate_selectivity) which is still efficient
    fn should_use_index_nested_loop(&self, info: &SemiJoinInfo, outer_limit: Option<i64>) -> bool {
        // Use index-nested-loop for small LIMIT queries WITHOUT additional predicates.
        //
        // IMPORTANT: If there's a non-correlated predicate (e.g., status = 'cancelled'),
        // semi-join is FASTER because:
        // 1. Semi-join executes the filtered inner query ONCE, builds a hash set
        // 2. Index NL would probe the index for EACH outer row, then filter
        //
        // Benchmark shows semi-join is 3x faster when additional predicates exist:
        // - Semi-join: ~290μs (execute filtered query once, O(1) hash lookups)
        // - Index NL:  ~940μs (per-row index probe + filter)
        //
        // Only use Index NL when:
        // 1. Small LIMIT (early termination benefit)
        // 2. NO additional predicates (pure correlation only)
        // 3. Index exists on correlation column
        const SMALL_LIMIT_THRESHOLD: i64 = 500;

        // If there's a non-correlated predicate, always use semi-join
        // The semi-join can efficiently filter by predicate in bulk
        if info.non_correlated_where.is_some() {
            return false;
        }

        // For pure correlation (no additional predicate), check if index NL is worth it
        if let Some(limit) = outer_limit {
            if limit > 0 && limit <= SMALL_LIMIT_THRESHOLD {
                // Check if inner table has an index on correlation column
                // Without index, per-row evaluation would be slow
                let txn = match self.engine.begin_transaction() {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                let inner_table = match txn.get_table(&info.inner_table) {
                    Ok(t) => t,
                    Err(_) => return false,
                };

                // Check for index on correlation column
                if inner_table
                    .get_index_on_column(&info.inner_column)
                    .is_some()
                {
                    return true;
                }
            }
        }

        // For larger queries or no index, use semi-join
        false
    }

    /// Check if index-nested-loop should be preferred over anti-join for NOT EXISTS.
    ///
    /// For NOT EXISTS, anti-join using HashJoinOperator is almost always more efficient
    /// than both index-nested-loop and InHashSet because:
    /// 1. HashJoinOperator does bulk hash table build/probe (cache-efficient)
    /// 2. No per-row expression evaluation overhead
    /// 3. Even with LIMIT, the bulk operation is faster than per-row checking
    ///
    /// The only case where we might prefer index-nested-loop is for VERY small LIMIT
    /// (e.g., LIMIT 10) with a highly selective index, but benchmarks show hash join
    /// is still faster in most cases.
    pub fn should_use_index_nested_loop_for_anti_join(
        &self,
        _info: &SemiJoinInfo,
        outer_limit: Option<i64>,
    ) -> bool {
        // For very small LIMIT (<= 10), index-nested-loop might be faster
        // because it can terminate very early
        if let Some(limit) = outer_limit {
            if limit <= 10 {
                return true;
            }
        }
        // For all other cases, prefer anti-join for NOT EXISTS
        false
    }

    /// Execute the semi-join optimization for an EXISTS subquery.
    ///
    /// Instead of executing the subquery for each outer row, we:
    /// 1. Execute the inner query once with non-correlated predicates
    /// 2. Collect all distinct values of the inner correlation column
    /// 3. Return an AHashSet for fast O(1) lookups
    ///
    /// Results are cached to avoid re-execution for the same query within a single
    /// top-level query execution.
    pub fn execute_semi_join_optimization(
        &self,
        info: &SemiJoinInfo,
        ctx: &ExecutionContext,
    ) -> Result<CompactArc<AHashSet<crate::core::Value>>> {
        // Build cache key hash from inner table, column, and WHERE predicate hash
        // Uses u64 hash to avoid any string allocation
        let pred_hash = info
            .non_correlated_where
            .as_ref()
            .map(|arc| compute_expression_hash(arc.as_ref()))
            .unwrap_or(0);
        let cache_key =
            compute_semi_join_cache_key(&info.inner_table, &info.inner_column, pred_hash);

        // Check cache first - return Arc directly (no clone needed)
        if let Some(cached) = get_cached_semi_join(cache_key) {
            return Ok(cached);
        }

        // Build SELECT inner_column FROM inner_table WHERE non_correlated_predicates
        // Use dummy_token_clone() to avoid allocations - token literal is not used during execution
        let inner_col_expr = Expression::Identifier(Identifier::new(
            dummy_token_clone(),
            info.inner_column.clone(),
        ));

        let table_source = Expression::TableSource(Box::new(SimpleTableSource {
            token: dummy_token_clone(),
            name: Identifier::new(dummy_token_clone(), info.inner_table.clone()),
            alias: info
                .inner_alias
                .as_ref()
                .map(|a| Identifier::new(dummy_token_clone(), a.clone())),
            as_of: None,
        }));

        let select_stmt = SelectStatement {
            token: dummy_token_clone(),
            // Don't use DISTINCT here - it's slower in Stoolap because it requires
            // additional hashing/sorting overhead. Instead, we collect into HashSet
            // which deduplicates more efficiently for this use case.
            distinct: false,
            columns: vec![inner_col_expr],
            with: None,
            table_expr: Some(Box::new(table_source)),
            where_clause: info
                .non_correlated_where
                .as_ref()
                .map(|arc| Box::new(arc.as_ref().clone())),
            group_by: GroupByClause {
                columns: vec![],
                modifier: GroupByModifier::None,
            },
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
        };

        // Execute the query with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(&select_stmt, &subquery_ctx)?;

        // Collect values into Vec first (faster than direct AHashSet insertion),
        // then convert to AHashSet for deduplication and O(1) lookups
        let mut values_vec = Vec::with_capacity(10_000);
        while result.next() {
            let row = result.row();
            if let Some(value) = row.get(0) {
                if !value.is_null() {
                    values_vec.push(value.clone());
                }
            }
        }
        // Build AHashSet from Vec - this deduplicates automatically
        let hash_set: AHashSet<crate::core::Value> = values_vec.into_iter().collect();

        // Wrap in CompactArc once - no cloning needed
        let hash_set_arc = CompactArc::new(hash_set);

        // Cache for subsequent calls within this query (CompactArc clone is cheap)
        cache_semi_join_arc(
            cache_key,
            &info.inner_table,
            CompactArc::clone(&hash_set_arc),
        );

        Ok(hash_set_arc)
    }

    /// Execute NOT EXISTS as a true anti-join using HashJoinOperator.
    ///
    /// This is more efficient than the InHashSet approach because:
    /// 1. HashJoinOperator builds hash table once and probes in bulk
    /// 2. No per-row expression evaluation overhead
    /// 3. Better cache efficiency due to batch processing
    /// 4. Direct table access without going through full query pipeline
    ///
    /// # Arguments
    /// * `info` - SemiJoinInfo extracted from the NOT EXISTS subquery
    /// * `outer_rows` - Pre-materialized outer table rows
    /// * `outer_columns` - Column names for outer table
    /// * `_ctx` - Execution context (not used but kept for API consistency)
    ///
    /// # Returns
    /// Rows from outer table that have NO match in inner table (anti-join result)
    pub fn execute_anti_join(
        &self,
        info: &SemiJoinInfo,
        outer_rows: CompactArc<Vec<crate::core::Row>>,
        outer_columns: &[String],
        _ctx: &ExecutionContext,
    ) -> Result<crate::core::RowVec> {
        // Direct table access - much faster than going through execute_select
        let txn = self.engine.begin_transaction()?;
        let inner_table = txn.get_table(&info.inner_table)?;

        // Convert non-correlated WHERE to storage expression for pushdown
        let storage_expr = info
            .non_correlated_where
            .as_ref()
            .and_then(|arc| convert_ast_to_storage_expr(arc.as_ref()));

        // Get inner table rows with filter pushed down to storage layer
        let inner_all_rows =
            inner_table.collect_all_rows(storage_expr.as_ref().map(|e| e.as_ref()))?;

        // Find the inner column index for join key extraction
        // Use schema's cached lowercase column names to avoid computing to_lowercase()
        let inner_schema = inner_table.schema();
        let inner_columns = inner_schema.column_names_arc();
        let inner_columns_lower = inner_schema.column_names_lower_arc();

        let inner_key_source_idx = {
            let search_col = info.inner_column.to_lowercase();
            inner_columns_lower
                .iter()
                .position(|c| c == &search_col)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Anti-join inner key column '{}' not found in table columns: {:?}",
                        info.inner_column, inner_columns
                    ))
                })?
        };

        // Extract only the join key values (deduplicated) and convert to single-column rows
        // Use a HashSet for deduplication to minimize the build side
        // Cap initial capacity to avoid over-allocation when many rows have few unique keys
        let estimated_unique = inner_all_rows.len().min(10000);
        let mut seen: AHashSet<crate::core::Value> = AHashSet::with_capacity(estimated_unique);
        let mut inner_rows: Vec<crate::core::Row> = Vec::with_capacity(estimated_unique);

        for (_, row) in &inner_all_rows {
            if let Some(value) = row.get(inner_key_source_idx) {
                if !value.is_null() {
                    // Clone once and reuse for both HashSet and Row to avoid double allocation
                    let cloned = value.clone();
                    if seen.insert(cloned.clone()) {
                        inner_rows.push(crate::core::Row::from_values(vec![cloned]));
                    }
                }
            }
        }

        // Find the outer column index for join key
        // OPTIMIZATION: Pre-compute lowercase column names once to avoid per-column to_lowercase()
        let outer_columns_lower: Vec<String> =
            outer_columns.iter().map(|c| c.to_lowercase()).collect();

        let outer_key_idx = {
            let search_col = info.outer_column.to_lowercase();
            let search_suffix = format!(".{}", search_col); // Pre-compute once outside loop
            outer_columns_lower
                .iter()
                .position(|c| {
                    c == &search_col
                        || c.ends_with(&search_suffix)
                        || c.split('.').next_back() == Some(search_col.as_str())
                })
                .ok_or_else(|| {
                    Error::internal(format!(
                        "Anti-join outer key column '{}' not found in columns: {:?}",
                        info.outer_column, outer_columns
                    ))
                })?
        };

        // Inner column is always index 0 (we extracted only the join key)
        let inner_key_idx = 0;

        // Create schemas for operators
        let outer_schema: Vec<ColumnInfo> = outer_columns.iter().map(ColumnInfo::new).collect();
        let inner_schema = vec![ColumnInfo::new(&info.inner_column)];

        // Create MaterializedOperators
        let outer_op = Box::new(MaterializedOperator::from_arc(
            outer_rows,
            outer_schema.clone(),
        ));
        let inner_op = Box::new(MaterializedOperator::new(inner_rows, inner_schema));

        // Create anti-join operator
        // Anti-join: return outer rows that have NO match in inner
        let mut join_op = HashJoinOperator::new(
            outer_op,
            inner_op,
            JoinType::Anti,
            vec![outer_key_idx],
            vec![inner_key_idx],
            JoinSide::Right, // Build on smaller (inner) side
        );

        // Execute the join with synthetic row IDs
        join_op.open()?;
        let mut result_rows = crate::core::RowVec::new();
        let mut row_id = 0i64;
        while let Some(row_ref) = join_op.next()? {
            result_rows.push((row_id, row_ref.into_owned()));
            row_id += 1;
        }
        join_op.close()?;

        Ok(result_rows)
    }

    /// Try to extract SemiJoinInfo from a NOT EXISTS expression.
    /// Returns None if the expression is not a valid NOT EXISTS pattern.
    pub fn try_extract_not_exists_info(
        expr: &Expression,
        outer_tables: &[String],
    ) -> Option<SemiJoinInfo> {
        if let Expression::Prefix(prefix) = expr {
            if prefix.operator.eq_ignore_ascii_case("NOT") {
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    return Self::try_extract_semi_join_info(exists, true, outer_tables);
                }
            }
        }
        None
    }

    /// Transform a WHERE clause with EXISTS into one using a pre-computed hash set.
    ///
    /// Replaces: EXISTS (SELECT ...) with: outer_col IN (hash_set_values)
    pub fn transform_exists_to_in_list(
        info: &SemiJoinInfo,
        hash_set: CompactArc<AHashSet<crate::core::Value>>,
    ) -> Expression {
        // For empty hash set, return FALSE (no matches exist)
        // For NOT EXISTS with empty set, return TRUE (nothing exists to negate)
        if hash_set.is_empty() {
            return Expression::BooleanLiteral(BooleanLiteral {
                token: dummy_token_clone(),
                value: info.is_negated,
            });
        }

        // Build the outer column expression using dummy_token_clone() to avoid allocations
        let outer_col_expr = if let Some(ref tbl) = info.outer_table {
            Expression::QualifiedIdentifier(QualifiedIdentifier {
                token: dummy_token_clone(),
                qualifier: Box::new(Identifier::new(dummy_token_clone(), tbl.clone())),
                name: Box::new(Identifier::new(
                    dummy_token_clone(),
                    info.outer_column.clone(),
                )),
            })
        } else {
            Expression::Identifier(Identifier::new(
                dummy_token_clone(),
                info.outer_column.clone(),
            ))
        };

        // Use InHashSet with Arc for O(1) lookup and cheap cloning in parallel execution
        Expression::InHashSet(InHashSetExpression {
            token: dummy_token_clone(),
            column: Box::new(outer_col_expr),
            values: hash_set, // Already Arc, no wrapping needed
            not: info.is_negated,
        })
    }

    /// Try to optimize correlated EXISTS subqueries to semi-join.
    /// Returns Some(optimized_expression) if successful, None if not applicable.
    ///
    /// Note: This function now checks if index-nested-loop would be more efficient
    /// and skips the semi-join transformation in that case, allowing per-row index probing.
    ///
    /// The `outer_limit` parameter helps decide between strategies:
    /// - With small LIMIT + index: prefer index-nested-loop (per-row probing with early termination)
    /// - Without LIMIT: prefer semi-join (scan inner once, hash lookup per outer row)
    pub fn try_optimize_exists_to_semi_join(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        outer_tables: &[String],
        outer_limit: Option<i64>,
    ) -> Result<Option<Expression>> {
        match expr {
            Expression::Exists(exists) => {
                if let Some(info) = Self::try_extract_semi_join_info(exists, false, outer_tables) {
                    // Check if index-nested-loop would be more efficient
                    // (index exists + no additional predicates, OR index exists + small LIMIT)
                    if self.should_use_index_nested_loop(&info, outer_limit) {
                        return Ok(None); // Skip semi-join, use index probing per row
                    }
                    // Semi-join optimization: execute inner query once, collect into hash set
                    // This enables InHashSet optimization to probe outer table's PK directly
                    let hash_set = self.execute_semi_join_optimization(&info, ctx)?;
                    return Ok(Some(Self::transform_exists_to_in_list(&info, hash_set)));
                }
                Ok(None)
            }

            Expression::Prefix(prefix) if prefix.operator.eq_ignore_ascii_case("NOT") => {
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    if let Some(info) = Self::try_extract_semi_join_info(exists, true, outer_tables)
                    {
                        // Check if index-nested-loop would be more efficient
                        if self.should_use_index_nested_loop(&info, outer_limit) {
                            return Ok(None); // Skip semi-join, use index probing per row
                        }
                        let hash_set = self.execute_semi_join_optimization(&info, ctx)?;
                        return Ok(Some(Self::transform_exists_to_in_list(&info, hash_set)));
                    }
                }
                Ok(None)
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try to optimize EXISTS in either branch of AND
                let left_opt = self.try_optimize_exists_to_semi_join(
                    &infix.left,
                    ctx,
                    outer_tables,
                    outer_limit,
                )?;
                let right_opt = self.try_optimize_exists_to_semi_join(
                    &infix.right,
                    ctx,
                    outer_tables,
                    outer_limit,
                )?;

                match (left_opt, right_opt) {
                    (Some(new_left), Some(new_right)) => {
                        Ok(Some(Expression::Infix(InfixExpression {
                            token: dummy_token_clone(),
                            left: Box::new(new_left),
                            operator: "AND".into(),
                            op_type: InfixOperator::And,
                            right: Box::new(new_right),
                        })))
                    }
                    (Some(new_left), None) => Ok(Some(Expression::Infix(InfixExpression {
                        token: dummy_token_clone(),
                        left: Box::new(new_left),
                        operator: "AND".into(),
                        op_type: InfixOperator::And,
                        right: infix.right.clone(),
                    }))),
                    (None, Some(new_right)) => Ok(Some(Expression::Infix(InfixExpression {
                        token: dummy_token_clone(),
                        left: infix.left.clone(),
                        operator: "AND".into(),
                        op_type: InfixOperator::And,
                        right: Box::new(new_right),
                    }))),
                    (None, None) => Ok(None),
                }
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("OR") => {
                // For OR, both branches must be optimizable for benefit
                // But we can still optimize individual EXISTS clauses
                let left_opt = self.try_optimize_exists_to_semi_join(
                    &infix.left,
                    ctx,
                    outer_tables,
                    outer_limit,
                )?;
                let right_opt = self.try_optimize_exists_to_semi_join(
                    &infix.right,
                    ctx,
                    outer_tables,
                    outer_limit,
                )?;

                match (left_opt, right_opt) {
                    (Some(new_left), Some(new_right)) => {
                        Ok(Some(Expression::Infix(InfixExpression {
                            token: dummy_token_clone(),
                            left: Box::new(new_left),
                            operator: "OR".into(),
                            op_type: InfixOperator::Or,
                            right: Box::new(new_right),
                        })))
                    }
                    (Some(new_left), None) => Ok(Some(Expression::Infix(InfixExpression {
                        token: dummy_token_clone(),
                        left: Box::new(new_left),
                        operator: "OR".into(),
                        op_type: InfixOperator::Or,
                        right: infix.right.clone(),
                    }))),
                    (None, Some(new_right)) => Ok(Some(Expression::Infix(InfixExpression {
                        token: dummy_token_clone(),
                        left: infix.left.clone(),
                        operator: "OR".into(),
                        op_type: InfixOperator::Or,
                        right: Box::new(new_right),
                    }))),
                    (None, None) => Ok(None),
                }
            }

            _ => Ok(None),
        }
    }

    // ============================================================================
    // IN Subquery Semi-Join Optimization
    // ============================================================================

    /// Try to optimize IN subqueries to semi-join (execute once, hash lookup per row).
    ///
    /// This transforms:
    /// ```sql
    /// WHERE outer.col IN (SELECT inner_col FROM t WHERE non_correlated_pred)
    /// ```
    /// Into:
    /// ```sql
    /// WHERE outer.col IN (hash_set_of_inner_col_values)
    /// ```
    ///
    /// # Optimization Criteria
    ///
    /// 1. IN right side must be a scalar subquery
    /// 2. Subquery must SELECT exactly one column
    /// 3. Subquery must have a simple table source (no joins)
    /// 4. Subquery WHERE clause must NOT reference outer tables (non-correlated)
    ///
    /// # Performance Impact
    ///
    /// - **Before**: O(N×M) - executes subquery for each outer row
    /// - **After**: O(N+M) - executes subquery once, O(1) hash lookup per row
    pub fn try_optimize_in_to_semi_join(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        outer_tables: &[String],
    ) -> Result<Option<Expression>> {
        match expr {
            Expression::In(in_expr) => {
                // Check if right side is a scalar subquery
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    if let Some(info) = Self::try_extract_in_semi_join_info(
                        in_expr,
                        &subquery.subquery,
                        outer_tables,
                    ) {
                        // Execute subquery once and build hash set
                        let hash_set = self.execute_semi_join_optimization(&info, ctx)?;
                        return Ok(Some(Self::transform_exists_to_in_list(&info, hash_set)));
                    }
                }
                Ok(None)
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try to optimize IN in either branch of AND
                let left_opt = self.try_optimize_in_to_semi_join(&infix.left, ctx, outer_tables)?;
                let right_opt =
                    self.try_optimize_in_to_semi_join(&infix.right, ctx, outer_tables)?;

                match (left_opt, right_opt) {
                    (Some(new_left), Some(new_right)) => {
                        Ok(Some(Expression::Infix(InfixExpression {
                            token: infix.token.clone(),
                            left: Box::new(new_left),
                            operator: infix.operator.clone(),
                            op_type: infix.op_type,
                            right: Box::new(new_right),
                        })))
                    }
                    (Some(new_left), None) => Ok(Some(Expression::Infix(InfixExpression {
                        token: infix.token.clone(),
                        left: Box::new(new_left),
                        operator: infix.operator.clone(),
                        op_type: infix.op_type,
                        right: infix.right.clone(),
                    }))),
                    (None, Some(new_right)) => Ok(Some(Expression::Infix(InfixExpression {
                        token: infix.token.clone(),
                        left: infix.left.clone(),
                        operator: infix.operator.clone(),
                        op_type: infix.op_type,
                        right: Box::new(new_right),
                    }))),
                    (None, None) => Ok(None),
                }
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("OR") => {
                // Try to optimize IN in either branch of OR
                let left_opt = self.try_optimize_in_to_semi_join(&infix.left, ctx, outer_tables)?;
                let right_opt =
                    self.try_optimize_in_to_semi_join(&infix.right, ctx, outer_tables)?;

                match (left_opt, right_opt) {
                    (Some(new_left), Some(new_right)) => {
                        Ok(Some(Expression::Infix(InfixExpression {
                            token: infix.token.clone(),
                            left: Box::new(new_left),
                            operator: infix.operator.clone(),
                            op_type: infix.op_type,
                            right: Box::new(new_right),
                        })))
                    }
                    (Some(new_left), None) => Ok(Some(Expression::Infix(InfixExpression {
                        token: infix.token.clone(),
                        left: Box::new(new_left),
                        operator: infix.operator.clone(),
                        op_type: infix.op_type,
                        right: infix.right.clone(),
                    }))),
                    (None, Some(new_right)) => Ok(Some(Expression::Infix(InfixExpression {
                        token: infix.token.clone(),
                        left: infix.left.clone(),
                        operator: infix.operator.clone(),
                        op_type: infix.op_type,
                        right: Box::new(new_right),
                    }))),
                    (None, None) => Ok(None),
                }
            }

            _ => Ok(None),
        }
    }

    /// Extract semi-join info from an IN expression with subquery.
    ///
    /// Pattern: `outer.col IN (SELECT inner_col FROM t WHERE pred)`
    ///
    /// Returns None if:
    /// - Subquery has more than one SELECT column
    /// - Subquery has joins or derived tables
    /// - WHERE clause references outer tables (correlated)
    fn try_extract_in_semi_join_info(
        in_expr: &InExpression,
        subquery: &SelectStatement,
        outer_tables: &[String],
    ) -> Option<SemiJoinInfo> {
        // 1. Extract outer column from left side of IN
        let (outer_column, outer_table): (String, Option<String>) = match in_expr.left.as_ref() {
            Expression::QualifiedIdentifier(qid) => (
                qid.name.value.to_string(),
                Some(qid.qualifier.value.to_string()),
            ),
            Expression::Identifier(id) => (id.value.to_string(), None),
            _ => return None, // Complex expression on left side, can't optimize
        };

        // 2. Subquery must SELECT exactly one column (not *)
        if subquery.columns.len() != 1 {
            return None;
        }

        // Extract inner column name from SELECT
        let inner_column: String = match &subquery.columns[0] {
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
            Expression::Aliased(a) => match a.expression.as_ref() {
                Expression::Identifier(id) => id.value.to_string(),
                Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
                _ => return None,
            },
            _ => return None, // Can't handle expressions in SELECT
        };

        // 3. Check for simple table source (not a join)
        let (inner_table, inner_alias): (String, Option<String>) =
            match subquery.table_expr.as_ref().map(|b| b.as_ref()) {
                Some(Expression::TableSource(ts)) => {
                    let alias = ts.alias.as_ref().map(|a| a.value.to_string());
                    (ts.name.value.to_string(), alias)
                }
                _ => return None, // Can't optimize subquery joins or derived tables
            };

        // 4. Get inner table identifiers
        let inner_table_lower: String = inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase());
        let inner_tables = vec![inner_table_lower.to_lowercase()];

        // 5. Check if WHERE clause references outer tables
        if let Some(ref where_clause) = subquery.where_clause {
            if Self::expression_references_outer_tables(where_clause, outer_tables, &inner_tables) {
                return None; // Correlated WHERE, can't optimize
            }
        }

        Some(SemiJoinInfo {
            outer_column,
            outer_table,
            inner_column,
            inner_table,
            inner_alias,
            non_correlated_where: subquery
                .where_clause
                .as_ref()
                .map(|b| Arc::new(b.as_ref().clone())),
            is_negated: in_expr.not,
        })
    }

    /// Get outer table names from a table expression (for semi-join optimization).
    pub fn collect_outer_table_names(table_expr: &Option<Box<Expression>>) -> Vec<String> {
        let mut tables = Vec::new();
        if let Some(ref expr) = table_expr {
            Self::collect_table_names_from_source(expr.as_ref(), &mut tables);
        }
        tables
    }
}
