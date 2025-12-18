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

use ahash::AHashSet;

use crate::core::{Error, Result, Value};
use crate::parser::ast::*;
use crate::parser::token::TokenType;
use crate::storage::traits::Engine;

use super::context::{
    cache_exists_fetcher, cache_exists_index, cache_exists_pred_key, cache_exists_predicate,
    cache_exists_schema, cache_in_subquery, cache_scalar_subquery, cache_semi_join,
    get_cached_exists_fetcher, get_cached_exists_index, get_cached_exists_pred_key,
    get_cached_exists_predicate, get_cached_exists_schema, get_cached_in_subquery,
    get_cached_scalar_subquery, get_cached_semi_join, ExecutionContext,
};
use super::utils::{dummy_token, value_to_expression};
use super::Executor;

// ============================================================================
// Semi-Join Optimization for EXISTS Subqueries
// ============================================================================

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
    pub non_correlated_where: Option<Expression>,
    /// Whether this is NOT EXISTS
    pub is_negated: bool,
}

/// Information needed for index-nested-loop EXISTS execution.
///
/// This is used for direct index probing instead of running a full subquery.
#[derive(Debug)]
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
                            expressions.push(Expression::ExpressionList(ExpressionList {
                                token: paren_token.clone(),
                                expressions: tuple_exprs,
                            }));
                        }

                        return Ok(Expression::In(InExpression {
                            token: in_expr.token.clone(),
                            left: Box::new(processed_left),
                            right: Box::new(Expression::ExpressionList(ExpressionList {
                                token: paren_token,
                                expressions,
                            })),
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
                            values: Arc::new(hash_set),
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
                        value: s.to_string(),
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
                        value: value.to_string(),
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

                Ok(Expression::Case(CaseExpression {
                    token: case.token.clone(),
                    value: processed_value,
                    when_clauses: processed_whens?,
                    else_value: processed_else,
                }))
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

                Ok(Expression::FunctionCall(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: processed_args?,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                }))
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

        // Extract correlation info from subquery
        let correlation = match Self::extract_index_nested_loop_info(subquery) {
            Some(info) => info,
            None => return Ok(None), // Can't use index-nested-loop
        };

        // Get the outer value from the outer row hashmap
        // Try qualified name first (e.g., "u.id"), then just column name
        let outer_value = if let Some(tbl) = &correlation.outer_table {
            let qualified = format!("{}.{}", tbl, &correlation.outer_column);
            outer_row
                .get(&qualified.to_lowercase())
                .or_else(|| outer_row.get(&correlation.outer_column.to_lowercase()))
        } else {
            outer_row.get(&correlation.outer_column.to_lowercase())
        };

        let outer_value = match outer_value {
            Some(v) if !v.is_null() => v.clone(),
            Some(_) => return Ok(Some(false)), // NULL never matches in EXISTS
            None => return Ok(None),           // Column not found, fall back
        };

        // OPTIMIZATION: Cache index reference to avoid repeated lookups
        // This reduces the ~2-5μs overhead per EXISTS probe to nearly zero for subsequent probes
        let index_cache_key = format!("{}:{}", correlation.inner_table, correlation.inner_column);

        let index = match get_cached_exists_index(&index_cache_key) {
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
                        cache_exists_index(index_cache_key, idx.clone());
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

        // If there's no additional predicate, any match means EXISTS is true
        // (we found at least one matching row_id in the index)
        if correlation.additional_predicate.is_none() {
            // The index returned row_ids, which means rows exist
            // For EXISTS we just need to know at least one exists
            return Ok(Some(true));
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
                let cols: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
                cache_exists_schema(correlation.inner_table.clone(), cols.clone());
                Arc::new(cols)
            }
        };

        // OPTIMIZATION: Use a cheap subquery identifier to look up the cached predicate cache key.
        // This avoids expensive alias stripping and Debug formatting on every probe.
        // The subquery pointer (as string) is stable within a query execution.
        let subquery_id = format!("{:p}", subquery as *const SelectStatement);

        // Try to get cached predicate filter using the cached predicate cache key
        let predicate_filter = match get_cached_exists_pred_key(&subquery_id) {
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
                let cache_key = format!("{}:{:?}", correlation.inner_table, stripped_pred);

                // Cache the predicate cache key for subsequent probes
                cache_exists_pred_key(subquery_id, cache_key.clone());

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
        // This avoids repeated version store lookups per EXISTS probe
        let row_fetcher = match get_cached_exists_fetcher(&correlation.inner_table) {
            Some(f) => f,
            None => {
                let fetcher = match self.engine.get_row_fetcher(&correlation.inner_table) {
                    Ok(f) => f,
                    Err(_) => return Ok(None), // Fall back if fetcher creation fails
                };
                cache_exists_fetcher(correlation.inner_table.clone(), fetcher);
                match get_cached_exists_fetcher(&correlation.inner_table) {
                    Some(f) => f,
                    None => return Ok(None), // Should not happen, but be safe
                }
            }
        };

        // Fetch rows by their IDs using the cached row fetcher
        const BATCH_SIZE: usize = 10;
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
        let inner_tables = vec![inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase())
            .to_lowercase()];

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
                let table = qid.qualifier.value.to_lowercase();
                if inner_tables.iter().any(|t| t.eq_ignore_ascii_case(&table)) {
                    Some(qid.name.value.clone())
                } else {
                    None
                }
            }
            Expression::Identifier(id) => {
                // Unqualified identifier assumed to be inner if in context
                Some(id.value.clone())
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
                let table = qid.qualifier.value.to_lowercase();
                // If NOT in inner_tables, it's an outer reference
                if !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(&table)) {
                    Some((qid.name.value.clone(), Some(qid.qualifier.value.clone())))
                } else {
                    None
                }
            }
            // Unqualified identifiers could be outer, but we can't be sure without schema
            _ => None,
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
                Expression::FunctionCall(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: new_args,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                })
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
                right: Box::new(Expression::ExpressionList(ExpressionList {
                    token: dummy_token("(", TokenType::Punctuator),
                    expressions: value_exprs,
                })),
                not: false,
            }));
        }

        // Special case: <> ALL is equivalent to NOT IN
        if (op == "<>" || op == "!=") && matches!(all_any.all_any_type, AllAnyType::All) {
            return Ok(Expression::In(InExpression {
                token: all_any.token.clone(),
                left: all_any.left.clone(),
                right: Box::new(Expression::ExpressionList(ExpressionList {
                    token: dummy_token("(", TokenType::Punctuator),
                    expressions: value_exprs,
                })),
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

        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Get the first row
        if !result.next() {
            let null_value = crate::core::Value::null_unknown();
            // Cache the result for non-correlated subqueries
            if let Some(key) = cache_key {
                cache_scalar_subquery(key, null_value.clone());
            }
            return Ok(null_value);
        }

        let row = result.take_row();
        if row.is_empty() {
            let null_value = crate::core::Value::null_unknown();
            if let Some(key) = cache_key {
                cache_scalar_subquery(key, null_value.clone());
            }
            return Ok(null_value);
        }

        // Get the first value
        let first_value = row
            .get(0)
            .cloned()
            .unwrap_or_else(crate::core::Value::null_unknown);

        // Check that there's only one row (scalar subquery should return single value)
        if result.next() {
            return Err(Error::Internal {
                message: "scalar subquery returned more than one row".to_string(),
            });
        }

        // Cache the result for non-correlated subqueries
        if let Some(key) = cache_key {
            cache_scalar_subquery(key, first_value.clone());
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

        // Cache the result for non-correlated subqueries
        if let Some(key) = cache_key {
            cache_in_subquery(key, values.clone());
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

        // Collect all values from all columns
        let mut rows = Vec::new();
        while result.next() {
            let row = result.row();
            if !row.is_empty() {
                // Convert Row to Vec<Value>
                rows.push(row.iter().cloned().collect());
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

                    Ok(Some(Expression::FunctionCall(FunctionCall {
                        token: func.token.clone(),
                        function: func.function.clone(),
                        arguments: final_args,
                        is_distinct: func.is_distinct,
                        order_by: func.order_by.clone(),
                        filter: func.filter.clone(),
                    })))
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

                    Ok(Some(Expression::Case(CaseExpression {
                        token: case.token.clone(),
                        value: processed_value.or_else(|| case.value.clone()),
                        when_clauses: final_whens,
                        else_value: processed_else.or_else(|| case.else_value.clone()),
                    })))
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

                Ok(Expression::FunctionCall(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: processed_args?,
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: func.filter.clone(),
                }))
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
                        values: Arc::new(hash_set),
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

                Ok(Expression::Case(CaseExpression {
                    token: case.token.clone(),
                    value: processed_value,
                    when_clauses: processed_whens?,
                    else_value: processed_else,
                }))
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
                    tables.push(alias.value.to_lowercase());
                } else {
                    tables.push(ts.name.value.to_lowercase());
                }
            }
            Expression::JoinSource(js) => {
                Self::collect_table_names_from_source(&js.left, tables);
                Self::collect_table_names_from_source(&js.right, tables);
            }
            Expression::SubquerySource(ss) => {
                // Subquery source has an optional alias
                if let Some(ref alias) = ss.alias {
                    tables.push(alias.value.to_lowercase());
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
                        values: Arc::new(hash_set),
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
        let (inner_table, inner_alias) = match subquery.table_expr.as_ref().map(|b| b.as_ref()) {
            Some(Expression::TableSource(ts)) => {
                let alias = ts.alias.as_ref().map(|a| a.value.clone());
                (ts.name.value.clone(), alias)
            }
            _ => return None, // Can't optimize subquery joins or derived tables
        };

        // 2. Parse WHERE clause to find correlation condition
        let where_clause = subquery.where_clause.as_ref()?;

        // Get inner table identifiers for distinguishing inner vs outer references
        let inner_tables = vec![inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase())
            .to_lowercase()];

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
            if Self::expression_references_outer_tables(rem, outer_tables, &inner_tables) {
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
    fn extract_equality_correlation(
        expr: &Expression,
        outer_tables: &[String],
        inner_tables: &[String],
    ) -> Option<(String, Option<String>, String, Option<Expression>)> {
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
                    // Combine remaining from left with right
                    let remaining =
                        Self::combine_and_predicates(left_remaining, Some((*infix.right).clone()));
                    return Some((outer_col, outer_tbl, inner_col, remaining));
                }

                // Try right side
                if let Some((outer_col, outer_tbl, inner_col, right_remaining)) =
                    Self::extract_equality_correlation(&infix.right, outer_tables, inner_tables)
                {
                    // Combine left with remaining from right
                    let remaining =
                        Self::combine_and_predicates(Some((*infix.left).clone()), right_remaining);
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
                let table = qid.qualifier.value.to_lowercase();
                // Must be in outer tables and NOT in inner tables
                if outer_tables.iter().any(|t| t.eq_ignore_ascii_case(&table))
                    && !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(&table))
                {
                    Some((qid.name.value.clone(), Some(qid.qualifier.value.clone())))
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
                let table = qid.qualifier.value.to_lowercase();
                if inner_tables.iter().any(|t| t.eq_ignore_ascii_case(&table)) {
                    Some(qid.name.value.clone())
                } else {
                    None
                }
            }
            Expression::Identifier(id) => {
                // Unqualified identifier - assume it's inner table column
                // This is safe because outer refs should be qualified in correlated subqueries
                Some(id.value.clone())
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
                let table = qid.qualifier.value.to_lowercase();
                // References outer if it's in outer_tables and NOT in inner_tables
                outer_tables.iter().any(|t| t.eq_ignore_ascii_case(&table))
                    && !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(&table))
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
    fn combine_and_predicates(
        left: Option<Expression>,
        right: Option<Expression>,
    ) -> Option<Expression> {
        match (left, right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some(l), Some(r)) => Some(Expression::Infix(InfixExpression::new(
                dummy_token("AND", TokenType::Keyword),
                Box::new(l),
                "AND".to_string(),
                Box::new(r),
            ))),
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
        // Use index-nested-loop for small LIMIT queries
        // The LIMIT early termination in the sequential path makes this efficient
        const SMALL_LIMIT_THRESHOLD: i64 = 500;

        // For small LIMIT, per-row EXISTS with index probe is faster
        // Complexity: O(LIMIT × log(inner)) vs O(inner) for semi-join hash build
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

        // For larger queries, additional predicates, or no index, use semi-join
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
    ) -> Result<AHashSet<crate::core::Value>> {
        // Build cache key from inner table, column, and WHERE predicate
        let cache_key = format!(
            "SEMI:{}:{}:{}",
            info.inner_table,
            info.inner_column,
            info.non_correlated_where
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or_default()
        );

        // Check cache first
        if let Some(cached) = get_cached_semi_join(&cache_key) {
            // Return owned copy from Arc (Arc::unwrap_or_clone equivalent)
            return Ok((*cached).clone());
        }

        // Build SELECT inner_column FROM inner_table WHERE non_correlated_predicates
        let inner_col_expr = Expression::Identifier(Identifier {
            token: dummy_token(&info.inner_column, TokenType::Identifier),
            value: info.inner_column.clone(),
            value_lower: info.inner_column.to_lowercase(),
        });

        let table_source = Expression::TableSource(SimpleTableSource {
            token: dummy_token(&info.inner_table, TokenType::Identifier),
            name: Identifier {
                token: dummy_token(&info.inner_table, TokenType::Identifier),
                value: info.inner_table.clone(),
                value_lower: info.inner_table.to_lowercase(),
            },
            alias: info.inner_alias.as_ref().map(|a| Identifier {
                token: dummy_token(a, TokenType::Identifier),
                value: a.clone(),
                value_lower: a.to_lowercase(),
            }),
            as_of: None,
        });

        let select_stmt = SelectStatement {
            token: dummy_token("SELECT", TokenType::Keyword),
            // Don't use DISTINCT here - it's slower in Stoolap because it requires
            // additional hashing/sorting overhead. Instead, we collect into HashSet
            // which deduplicates more efficiently for this use case.
            distinct: false,
            columns: vec![inner_col_expr],
            with: None,
            table_expr: Some(Box::new(table_source)),
            where_clause: info.non_correlated_where.clone().map(Box::new),
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

        // Cache for subsequent calls within this query
        cache_semi_join(cache_key, hash_set.clone());

        Ok(hash_set)
    }

    /// Transform a WHERE clause with EXISTS into one using a pre-computed hash set.
    ///
    /// Replaces: EXISTS (SELECT ...) with: outer_col IN (hash_set_values)
    pub fn transform_exists_to_in_list(
        info: &SemiJoinInfo,
        hash_set: AHashSet<crate::core::Value>,
    ) -> Expression {
        // For empty hash set, return FALSE (no matches exist)
        // For NOT EXISTS with empty set, return TRUE (nothing exists to negate)
        if hash_set.is_empty() {
            return Expression::BooleanLiteral(BooleanLiteral {
                token: dummy_token(
                    if info.is_negated { "TRUE" } else { "FALSE" },
                    TokenType::Keyword,
                ),
                value: info.is_negated,
            });
        }

        // Build the outer column expression
        let outer_col_expr = if let Some(ref tbl) = info.outer_table {
            Expression::QualifiedIdentifier(QualifiedIdentifier {
                token: dummy_token(&info.outer_column, TokenType::Identifier),
                qualifier: Box::new(Identifier {
                    token: dummy_token(tbl, TokenType::Identifier),
                    value: tbl.clone(),
                    value_lower: tbl.to_lowercase(),
                }),
                name: Box::new(Identifier {
                    token: dummy_token(&info.outer_column, TokenType::Identifier),
                    value: info.outer_column.clone(),
                    value_lower: info.outer_column.to_lowercase(),
                }),
            })
        } else {
            Expression::Identifier(Identifier {
                token: dummy_token(&info.outer_column, TokenType::Identifier),
                value: info.outer_column.clone(),
                value_lower: info.outer_column.to_lowercase(),
            })
        };

        // Use InHashSet with Arc for O(1) lookup and cheap cloning in parallel execution
        Expression::InHashSet(InHashSetExpression {
            token: dummy_token("IN", TokenType::Keyword),
            column: Box::new(outer_col_expr),
            values: std::sync::Arc::new(hash_set),
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

    /// Get outer table names from a table expression (for semi-join optimization).
    pub fn collect_outer_table_names(table_expr: &Option<Box<Expression>>) -> Vec<String> {
        let mut tables = Vec::new();
        if let Some(ref expr) = table_expr {
            Self::collect_table_names_from_source(expr.as_ref(), &mut tables);
        }
        tables
    }
}
