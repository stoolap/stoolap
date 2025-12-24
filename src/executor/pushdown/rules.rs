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

//! Pushdown Rule Implementations
//!
//! Each rule handles a specific type of expression and converts it to
//! a storage-layer expression when possible.

use crate::core::Value;
use crate::functions::global_registry;
use crate::parser::ast::{self as ast, InfixOperator, PrefixOperator};
use crate::storage::expression::{
    AndExpr, BetweenExpr, ComparisonExpr, ConstBoolExpr, Expression as StorageExpr, FunctionArg,
    FunctionExpr, InListExpr, LikeExpr, NotExpr, NullCheckExpr, OrExpr,
};
use std::sync::Arc;

use super::{registry, PushdownContext, PushdownResult, PushdownRule};
use crate::executor::utils::{
    extract_column_name, extract_literal_value, flip_operator, infix_to_operator,
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract a literal value with context (handles parameters)
fn extract_literal_with_ctx(expr: &ast::Expression, ctx: &PushdownContext<'_>) -> Option<Value> {
    // First try simple literal
    if let Some(value) = extract_literal_value(expr) {
        return Some(value);
    }

    // Try parameter reference
    if let ast::Expression::Parameter(param) = expr {
        if let Some(exec_ctx) = ctx.exec_ctx {
            // Check if it's a named parameter (starts with :)
            if param.name.starts_with(':') {
                let name = &param.name[1..];
                return exec_ctx.get_named_param(name).cloned();
            } else {
                // Positional parameter (1-indexed)
                let params = exec_ctx.params();
                if param.index > 0 && param.index <= params.len() {
                    return Some(params[param.index - 1].clone());
                }
            }
        }
    }

    None
}

/// Extract column and value from a comparison (handles both col op val and val op col)
fn extract_comparison_parts(
    left: &ast::Expression,
    right: &ast::Expression,
    ctx: &PushdownContext<'_>,
) -> Option<(String, Value, bool)> {
    // Try column on left, value on right
    if let Some(column) = extract_column_name(left) {
        if let Some(value) = extract_literal_with_ctx(right, ctx) {
            let coerced = ctx.coerce_to_column_type(&column, value);
            return Some((column, coerced, false)); // false = not flipped
        }
    }

    // Try value on left, column on right (will need to flip operator)
    if let Some(column) = extract_column_name(right) {
        if let Some(value) = extract_literal_with_ctx(left, ctx) {
            let coerced = ctx.coerce_to_column_type(&column, value);
            return Some((column, coerced, true)); // true = flipped
        }
    }

    None
}

/// Extract values from an IN list
fn extract_in_list_values(expr: &ast::Expression, ctx: &PushdownContext<'_>) -> Option<Vec<Value>> {
    match expr {
        ast::Expression::ExpressionList(list) => list
            .expressions
            .iter()
            .map(|e| extract_literal_with_ctx(e, ctx))
            .collect(),
        ast::Expression::List(list) => list
            .elements
            .iter()
            .map(|e| extract_literal_with_ctx(e, ctx))
            .collect(),
        // Subquery - cannot push down
        ast::Expression::ScalarSubquery(_) => None,
        // Single value
        _ => extract_literal_with_ctx(expr, ctx).map(|v| vec![v]),
    }
}

// =============================================================================
// Comparison Rule: col = val, col > val, etc.
// =============================================================================

pub struct ComparisonRule;

impl PushdownRule for ComparisonRule {
    fn name(&self) -> &'static str {
        "comparison"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) => i,
            _ => return PushdownResult::NotApplicable,
        };

        // Check if this is a comparison operator
        let base_op = match infix_to_operator(infix.op_type) {
            Some(op) => op,
            None => return PushdownResult::NotApplicable,
        };

        // Extract column and value
        // If extraction fails, return NotApplicable so FunctionRule can try
        let (column, value, flipped) =
            match extract_comparison_parts(&infix.left, &infix.right, ctx) {
                Some(parts) => parts,
                None => return PushdownResult::NotApplicable,
            };

        let operator = if flipped {
            flip_operator(base_op)
        } else {
            base_op
        };

        let mut expr = ComparisonExpr::new(column, operator, value);
        expr.prepare_for_schema(ctx.schema);
        PushdownResult::Converted(Box::new(expr))
    }
}

// =============================================================================
// Logical AND Rule: a AND b
// =============================================================================

pub struct LogicalAndRule;

impl PushdownRule for LogicalAndRule {
    fn name(&self) -> &'static str {
        "logical_and"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) if i.op_type == InfixOperator::And => i,
            _ => return PushdownResult::NotApplicable,
        };

        // For AND: try both sides, combine what we can push
        let (left_pushable, left_needs_mem) = registry().try_pushdown_with_ctx(&infix.left, ctx);
        let (right_pushable, right_needs_mem) = registry().try_pushdown_with_ctx(&infix.right, ctx);

        let needs_memory_filter = left_needs_mem || right_needs_mem;

        match (left_pushable, right_pushable) {
            (Some(mut left), Some(mut right)) => {
                // Both sides pushable - combine with AND
                left.prepare_for_schema(ctx.schema);
                right.prepare_for_schema(ctx.schema);
                let and_expr = AndExpr::new(vec![left, right]);
                if needs_memory_filter {
                    // We pushed both sides but one or both had partial pushdown
                    PushdownResult::Partial(Box::new(and_expr))
                } else {
                    // Fully pushed
                    PushdownResult::Converted(Box::new(and_expr))
                }
            }
            (Some(expr), None) | (None, Some(expr)) => {
                // Only one side pushable - partial pushdown (definitely needs memory filter)
                PushdownResult::Partial(expr)
            }
            (None, None) => {
                // Neither side pushable - need full memory filter
                PushdownResult::CannotPush
            }
        }
    }
}

// =============================================================================
// Logical OR Rule: a OR b
// =============================================================================

pub struct LogicalOrRule;

impl PushdownRule for LogicalOrRule {
    fn name(&self) -> &'static str {
        "logical_or"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) if i.op_type == InfixOperator::Or => i,
            _ => return PushdownResult::NotApplicable,
        };

        // For OR: can only push if BOTH sides are fully pushable
        // (partial pushdown of OR would change semantics)
        let left = registry().convert_expr(&infix.left, ctx);
        let right = registry().convert_expr(&infix.right, ctx);

        match (left, right) {
            (Some(mut l), Some(mut r)) => {
                l.prepare_for_schema(ctx.schema);
                r.prepare_for_schema(ctx.schema);
                PushdownResult::Converted(Box::new(OrExpr::new(vec![l, r])))
            }
            _ => PushdownResult::CannotPush,
        }
    }
}

// =============================================================================
// Logical NOT Rule: NOT a
// =============================================================================

pub struct LogicalNotRule;

impl PushdownRule for LogicalNotRule {
    fn name(&self) -> &'static str {
        "logical_not"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let prefix = match expr {
            ast::Expression::Prefix(p) if p.op_type == PrefixOperator::Not => p,
            _ => return PushdownResult::NotApplicable,
        };

        // NOT requires the inner expression to be fully pushable
        match registry().convert_expr(&prefix.right, ctx) {
            Some(mut inner) => {
                inner.prepare_for_schema(ctx.schema);
                PushdownResult::Converted(Box::new(NotExpr::new(inner)))
            }
            None => PushdownResult::CannotPush,
        }
    }
}

// =============================================================================
// Logical XOR Rule: a XOR b
// =============================================================================

pub struct LogicalXorRule;

impl PushdownRule for LogicalXorRule {
    fn name(&self) -> &'static str {
        "logical_xor"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) if i.op_type == InfixOperator::Xor => i,
            _ => return PushdownResult::NotApplicable,
        };

        // XOR is equivalent to: (A AND NOT B) OR (NOT A AND B)
        let left1 = registry().convert_expr(&infix.left, ctx);
        let right1 = registry().convert_expr(&infix.right, ctx);
        let left2 = registry().convert_expr(&infix.left, ctx);
        let right2 = registry().convert_expr(&infix.right, ctx);

        match (left1, right1, left2, right2) {
            (Some(mut l1), Some(mut r1), Some(mut l2), Some(mut r2)) => {
                l1.prepare_for_schema(ctx.schema);
                r1.prepare_for_schema(ctx.schema);
                l2.prepare_for_schema(ctx.schema);
                r2.prepare_for_schema(ctx.schema);

                // (left AND NOT right) OR (NOT left AND right)
                let left_and_not_right = AndExpr::new(vec![l1, Box::new(NotExpr::new(r1))]);
                let not_left_and_right = AndExpr::new(vec![Box::new(NotExpr::new(l2)), r2]);
                PushdownResult::Converted(Box::new(OrExpr::new(vec![
                    Box::new(left_and_not_right),
                    Box::new(not_left_and_right),
                ])))
            }
            _ => PushdownResult::CannotPush,
        }
    }
}

// =============================================================================
// BETWEEN Rule: col BETWEEN a AND b
// =============================================================================

pub struct BetweenRule;

impl PushdownRule for BetweenRule {
    fn name(&self) -> &'static str {
        "between"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let between = match expr {
            ast::Expression::Between(b) => b,
            _ => return PushdownResult::NotApplicable,
        };

        let column = match extract_column_name(&between.expr) {
            Some(c) => c,
            None => return PushdownResult::CannotPush,
        };

        let lower = match extract_literal_with_ctx(&between.lower, ctx) {
            Some(v) => ctx.coerce_to_column_type(&column, v),
            None => return PushdownResult::CannotPush,
        };

        let upper = match extract_literal_with_ctx(&between.upper, ctx) {
            Some(v) => ctx.coerce_to_column_type(&column, v),
            None => return PushdownResult::CannotPush,
        };

        let mut expr = if between.not {
            BetweenExpr::not_between(column, lower, upper)
        } else {
            BetweenExpr::new(column, lower, upper)
        };
        expr.prepare_for_schema(ctx.schema);
        PushdownResult::Converted(Box::new(expr))
    }
}

// =============================================================================
// IN List Rule: col IN (a, b, c)
// =============================================================================

pub struct InListRule;

impl PushdownRule for InListRule {
    fn name(&self) -> &'static str {
        "in_list"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let in_expr = match expr {
            ast::Expression::In(i) => i,
            _ => return PushdownResult::NotApplicable,
        };

        let column = match extract_column_name(&in_expr.left) {
            Some(c) => c,
            None => return PushdownResult::CannotPush,
        };

        let values = match extract_in_list_values(&in_expr.right, ctx) {
            Some(v) if !v.is_empty() => v,
            _ => return PushdownResult::CannotPush,
        };

        // Coerce values to column type
        let coerced_values: Vec<Value> = values
            .into_iter()
            .map(|v| ctx.coerce_to_column_type(&column, v))
            .collect();

        let mut expr = if in_expr.not {
            InListExpr::not_in(column, coerced_values)
        } else {
            InListExpr::new(column, coerced_values)
        };
        expr.prepare_for_schema(ctx.schema);
        PushdownResult::Converted(Box::new(expr))
    }
}

// =============================================================================
// LIKE Rule: col LIKE pattern
// =============================================================================

pub struct LikeRule;

impl PushdownRule for LikeRule {
    fn name(&self) -> &'static str {
        "like"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        // Handle both ast::Like and Infix LIKE/ILIKE
        match expr {
            ast::Expression::Like(like) => self.convert_like_expr(like, ctx),
            ast::Expression::Infix(infix) => self.convert_infix_like(infix, ctx),
            _ => PushdownResult::NotApplicable,
        }
    }
}

impl LikeRule {
    fn convert_like_expr(
        &self,
        like: &ast::LikeExpression,
        ctx: &PushdownContext<'_>,
    ) -> PushdownResult {
        // Parse operator to determine type
        let op_upper = like.operator.to_uppercase();

        // GLOB and REGEXP use different pattern matching semantics than LIKE
        // They must be evaluated in memory by the expression VM
        if op_upper.contains("GLOB") || op_upper.contains("REGEXP") || op_upper.contains("RLIKE") {
            return PushdownResult::CannotPush;
        }

        // Check for UPPER(col) or LOWER(col) LIKE pattern -> case-insensitive LIKE
        let (column, force_ilike) = if let Some(col) = extract_column_name(&like.left) {
            (col, false)
        } else if let Some(col) = self.extract_upper_lower_column(&like.left) {
            (col, true) // UPPER/LOWER(col) LIKE -> ILIKE
        } else {
            return PushdownResult::CannotPush;
        };

        let pattern = match &*like.pattern {
            ast::Expression::StringLiteral(s) => s.value.clone(),
            _ => return PushdownResult::CannotPush,
        };

        let is_not = op_upper.contains("NOT");
        let is_ilike = force_ilike || op_upper.contains("ILIKE");

        let mut expr = match (is_not, is_ilike) {
            (true, true) => LikeExpr::not_ilike(column, pattern),
            (true, false) => LikeExpr::not_like(column, pattern),
            (false, true) => LikeExpr::new_ilike(column, pattern),
            (false, false) => LikeExpr::new(column, pattern),
        };
        expr.prepare_for_schema(ctx.schema);
        PushdownResult::Converted(Box::new(expr))
    }

    /// Extract column name from UPPER(col) or LOWER(col) function call
    fn extract_upper_lower_column(&self, expr: &ast::Expression) -> Option<String> {
        if let ast::Expression::FunctionCall(fc) = expr {
            let func_name = fc.function.to_uppercase();
            if (func_name == "UPPER" || func_name == "LOWER") && fc.arguments.len() == 1 {
                return extract_column_name(&fc.arguments[0]);
            }
        }
        None
    }

    fn convert_infix_like(
        &self,
        infix: &ast::InfixExpression,
        ctx: &PushdownContext<'_>,
    ) -> PushdownResult {
        // Check operator type FIRST - must be a LIKE operator
        // This is critical: we must return NotApplicable for non-LIKE operators
        // so that ComparisonRule gets a chance to handle them
        let is_like_op = matches!(
            infix.op_type,
            InfixOperator::Like
                | InfixOperator::ILike
                | InfixOperator::NotLike
                | InfixOperator::NotILike
        );
        if !is_like_op {
            return PushdownResult::NotApplicable;
        }

        // Extract column, checking for UPPER/LOWER wrapper
        let (column, force_ilike) = if let Some(col) = extract_column_name(&infix.left) {
            (col, false)
        } else if let Some(col) = self.extract_upper_lower_column(&infix.left) {
            (col, true) // UPPER/LOWER(col) -> force case-insensitive
        } else {
            return PushdownResult::CannotPush;
        };

        let pattern = match extract_pattern(&infix.right) {
            Some(p) => p,
            None => return PushdownResult::CannotPush,
        };

        let (is_ilike, is_not) = match infix.op_type {
            InfixOperator::Like => (force_ilike, false),
            InfixOperator::ILike => (true, false),
            InfixOperator::NotLike => (force_ilike, true),
            InfixOperator::NotILike => (true, true),
            _ => return PushdownResult::NotApplicable, // Should never reach here due to check above
        };

        let mut like_expr = if is_ilike {
            LikeExpr::new_ilike(column, pattern)
        } else {
            LikeExpr::new(column, pattern)
        };
        like_expr.prepare_for_schema(ctx.schema);

        if is_not {
            PushdownResult::Converted(Box::new(NotExpr::new(Box::new(like_expr))))
        } else {
            PushdownResult::Converted(Box::new(like_expr))
        }
    }
}

fn extract_pattern(expr: &ast::Expression) -> Option<String> {
    match expr {
        ast::Expression::StringLiteral(s) => Some(s.value.clone()),
        _ => None,
    }
}

// =============================================================================
// NULL Check Rule: col IS NULL, col IS NOT NULL
// =============================================================================

pub struct NullCheckRule;

impl PushdownRule for NullCheckRule {
    fn name(&self) -> &'static str {
        "null_check"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) => i,
            _ => return PushdownResult::NotApplicable,
        };

        match infix.op_type {
            InfixOperator::Is | InfixOperator::IsNot => {
                // Check if right side is NULL
                if !matches!(&*infix.right, ast::Expression::NullLiteral(_)) {
                    return PushdownResult::NotApplicable;
                }

                let column = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };

                let mut expr = if infix.op_type == InfixOperator::IsNot {
                    NullCheckExpr::is_not_null(column)
                } else {
                    NullCheckExpr::is_null(column)
                };
                expr.prepare_for_schema(ctx.schema);
                PushdownResult::Converted(Box::new(expr))
            }
            _ => PushdownResult::NotApplicable,
        }
    }
}

// =============================================================================
// Boolean Check Rule: col IS TRUE, col IS FALSE, col IS NOT TRUE, etc.
// =============================================================================

pub struct BooleanCheckRule;

impl PushdownRule for BooleanCheckRule {
    fn name(&self) -> &'static str {
        "boolean_check"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        let infix = match expr {
            ast::Expression::Infix(i) => i,
            _ => return PushdownResult::NotApplicable,
        };

        match infix.op_type {
            InfixOperator::Is => {
                // IS TRUE or IS FALSE
                let bool_val = match &*infix.right {
                    ast::Expression::BooleanLiteral(b) => b.value,
                    _ => return PushdownResult::NotApplicable,
                };

                let column = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };

                let mut expr = ComparisonExpr::eq(column, Value::Boolean(bool_val));
                expr.prepare_for_schema(ctx.schema);
                PushdownResult::Converted(Box::new(expr))
            }
            InfixOperator::IsNot => {
                // IS NOT TRUE or IS NOT FALSE
                // This is equivalent to (col <> TRUE/FALSE OR col IS NULL)
                let bool_val = match &*infix.right {
                    ast::Expression::BooleanLiteral(b) => b.value,
                    _ => return PushdownResult::NotApplicable,
                };

                let column = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };

                let mut ne_expr = ComparisonExpr::ne(&column, Value::Boolean(bool_val));
                ne_expr.prepare_for_schema(ctx.schema);
                let mut null_expr = NullCheckExpr::is_null(&column);
                null_expr.prepare_for_schema(ctx.schema);

                PushdownResult::Converted(Box::new(OrExpr::new(vec![
                    Box::new(ne_expr),
                    Box::new(null_expr),
                ])))
            }
            _ => PushdownResult::NotApplicable,
        }
    }
}

// =============================================================================
// Boolean Literal Rule: TRUE, FALSE
// =============================================================================

pub struct BooleanLiteralRule;

impl PushdownRule for BooleanLiteralRule {
    fn name(&self) -> &'static str {
        "boolean_literal"
    }

    fn try_convert(&self, expr: &ast::Expression, _ctx: &PushdownContext<'_>) -> PushdownResult {
        match expr {
            ast::Expression::BooleanLiteral(b) => {
                PushdownResult::Converted(Box::new(ConstBoolExpr::new(b.value)))
            }
            _ => PushdownResult::NotApplicable,
        }
    }
}

// =============================================================================
// Function Rule: LENGTH(col) > 5, UPPER(col) = 'X', etc.
// =============================================================================

pub struct FunctionRule;

impl PushdownRule for FunctionRule {
    fn name(&self) -> &'static str {
        "function"
    }

    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult {
        // Handle: FUNC(col) op value or value op FUNC(col)
        let infix = match expr {
            ast::Expression::Infix(i) => i,
            _ => return PushdownResult::NotApplicable,
        };

        // Check if this is a comparison operator
        let base_op = match infix_to_operator(infix.op_type) {
            Some(op) => op,
            None => return PushdownResult::NotApplicable,
        };

        // Try: FUNC(args) op value
        if let Some((func_expr, flipped)) =
            self.try_extract_function_comparison(&infix.left, &infix.right, base_op, ctx)
        {
            let _ = flipped; // We handle flipping in try_extract_function_comparison
            return PushdownResult::Converted(func_expr);
        }

        // Try: value op FUNC(args) - need to flip operator since operands are reversed
        // e.g., "5 < LENGTH(name)" becomes "LENGTH(name) > 5"
        if let Some((func_expr, _)) = self.try_extract_function_comparison(
            &infix.right,
            &infix.left,
            flip_operator(base_op),
            ctx,
        ) {
            return PushdownResult::Converted(func_expr);
        }

        PushdownResult::NotApplicable
    }
}

impl FunctionRule {
    /// Try to extract a function comparison expression
    /// Returns (FunctionExpr, flipped) if successful
    fn try_extract_function_comparison(
        &self,
        func_side: &ast::Expression,
        value_side: &ast::Expression,
        operator: crate::core::Operator,
        ctx: &PushdownContext<'_>,
    ) -> Option<(Box<dyn StorageExpr>, bool)> {
        // Check if func_side is a function call
        let func_call = match func_side {
            ast::Expression::FunctionCall(fc) => fc,
            _ => return None,
        };

        // Get the scalar function from registry
        let func_name = func_call.function.to_uppercase();
        let scalar_func = global_registry().get_scalar(&func_name)?;

        // Convert function arguments to FunctionArg
        let mut args = Vec::with_capacity(func_call.arguments.len());
        for arg in &func_call.arguments {
            match self.convert_func_arg(arg, ctx) {
                Some(fa) => args.push(fa),
                None => return None, // Can't convert argument, bail out
            }
        }

        // Get the comparison value
        let compare_value = extract_literal_with_ctx(value_side, ctx)?;

        // Create FunctionExpr
        let mut func_expr =
            FunctionExpr::new(Arc::from(scalar_func), args, operator, compare_value);
        func_expr.prepare_for_schema(ctx.schema);

        Some((Box::new(func_expr), false))
    }

    /// Convert AST expression to FunctionArg
    fn convert_func_arg(
        &self,
        expr: &ast::Expression,
        ctx: &PushdownContext<'_>,
    ) -> Option<FunctionArg> {
        // Column reference
        if let Some(col_name) = extract_column_name(expr) {
            return Some(FunctionArg::Column(col_name));
        }

        // Literal value
        if let Some(value) = extract_literal_with_ctx(expr, ctx) {
            return Some(FunctionArg::Literal(value));
        }

        // Nested function call
        if let ast::Expression::FunctionCall(fc) = expr {
            let func_name = fc.function.to_uppercase();
            if let Some(scalar_func) = global_registry().get_scalar(&func_name) {
                let mut nested_args = Vec::with_capacity(fc.arguments.len());
                for arg in &fc.arguments {
                    nested_args.push(self.convert_func_arg(arg, ctx)?);
                }
                return Some(FunctionArg::Function {
                    function: Arc::from(scalar_func),
                    arguments: nested_args,
                });
            }
        }

        None
    }
}
