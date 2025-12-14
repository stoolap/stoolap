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

use std::collections::HashSet;
use std::sync::Arc;

use crate::core::{Error, Result, Value};
use crate::parser::ast::*;
use crate::parser::token::TokenType;

use super::context::ExecutionContext;
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

                        // Collect into HashSet for O(1) membership testing
                        let hash_set: HashSet<Value> = values.into_iter().collect();

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
    fn execute_exists_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<bool> {
        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Check if there's at least one row
        let exists = result.next();

        Ok(exists)
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

    /// Execute a scalar subquery and return its single value
    fn execute_scalar_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<crate::core::Value> {
        // Execute the subquery with incremented depth to avoid creating new TimeoutGuard
        let subquery_ctx = ctx.with_incremented_query_depth();
        let mut result = self.execute_select(subquery, &subquery_ctx)?;

        // Get the first row
        if !result.next() {
            return Ok(crate::core::Value::null_unknown());
        }

        let row = result.take_row();
        if row.is_empty() {
            return Ok(crate::core::Value::null_unknown());
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

        Ok(first_value)
    }

    /// Execute an IN subquery and return its values
    fn execute_in_subquery(
        &self,
        subquery: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Vec<crate::core::Value>> {
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
                    // Use InHashSet for O(1) lookups
                    let values = self.execute_in_subquery(&subquery.subquery, ctx)?;
                    let hash_set: HashSet<Value> = values.into_iter().collect();

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
    fn is_subquery_correlated(subquery: &SelectStatement) -> bool {
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
                    // Use InHashSet for O(1) lookups
                    let values = self.execute_in_subquery(&subquery.subquery, ctx)?;
                    let hash_set: HashSet<Value> = values.into_iter().collect();

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
            _ => {
                return None; // Can't optimize subquery joins or derived tables
            }
        };

        // 2. Parse WHERE clause to find correlation condition
        let where_clause = subquery.where_clause.as_ref()?;

        // Get inner table identifiers for distinguishing inner vs outer references
        let inner_tables = vec![inner_alias
            .clone()
            .unwrap_or_else(|| inner_table.to_lowercase())
            .to_lowercase()];

        // Try to extract: outer.col = inner.col (or inner.col = outer.col)
        let (outer_col, outer_tbl, inner_col, remaining) =
            Self::extract_equality_correlation(where_clause, outer_tables, &inner_tables)?;

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

    /// Execute the semi-join optimization for an EXISTS subquery.
    ///
    /// Instead of executing the subquery for each outer row, we:
    /// 1. Execute the inner query once with non-correlated predicates
    /// 2. Collect all distinct values of the inner correlation column
    /// 3. Return a HashSet for fast O(1) lookups
    pub fn execute_semi_join_optimization(
        &self,
        info: &SemiJoinInfo,
        ctx: &ExecutionContext,
    ) -> Result<std::collections::HashSet<crate::core::Value>> {
        // Build SELECT DISTINCT inner_column FROM inner_table WHERE non_correlated_predicates
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
            distinct: false, // No DISTINCT - we collect into HashSet which deduplicates
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

        // Collect values into HashSet (deduplicates automatically)
        let mut hash_set = std::collections::HashSet::new();
        while result.next() {
            let row = result.row();
            if !row.is_empty() {
                if let Some(value) = row.get(0) {
                    if !value.is_null() {
                        hash_set.insert(value.clone());
                    }
                }
            }
        }

        Ok(hash_set)
    }

    /// Transform a WHERE clause with EXISTS into one using a pre-computed hash set.
    ///
    /// Replaces: EXISTS (SELECT ...) with: outer_col IN (hash_set_values)
    pub fn transform_exists_to_in_list(
        info: &SemiJoinInfo,
        hash_set: std::collections::HashSet<crate::core::Value>,
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
    pub fn try_optimize_exists_to_semi_join(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        outer_tables: &[String],
    ) -> Result<Option<Expression>> {
        match expr {
            Expression::Exists(exists) => {
                if let Some(info) = Self::try_extract_semi_join_info(exists, false, outer_tables) {
                    let hash_set = self.execute_semi_join_optimization(&info, ctx)?;
                    return Ok(Some(Self::transform_exists_to_in_list(&info, hash_set)));
                }
                Ok(None)
            }

            Expression::Prefix(prefix) if prefix.operator.eq_ignore_ascii_case("NOT") => {
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    if let Some(info) = Self::try_extract_semi_join_info(exists, true, outer_tables)
                    {
                        let hash_set = self.execute_semi_join_optimization(&info, ctx)?;
                        return Ok(Some(Self::transform_exists_to_in_list(&info, hash_set)));
                    }
                }
                Ok(None)
            }

            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try to optimize EXISTS in either branch of AND
                let left_opt =
                    self.try_optimize_exists_to_semi_join(&infix.left, ctx, outer_tables)?;
                let right_opt =
                    self.try_optimize_exists_to_semi_join(&infix.right, ctx, outer_tables)?;

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
                let left_opt =
                    self.try_optimize_exists_to_semi_join(&infix.left, ctx, outer_tables)?;
                let right_opt =
                    self.try_optimize_exists_to_semi_join(&infix.right, ctx, outer_tables)?;

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
