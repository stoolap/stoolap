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
use crate::parser::ast::{self as ast, InfixOperator, PrefixOperator};
use crate::storage::expression::{
    AndExpr, BetweenExpr, ComparisonExpr, ConstBoolExpr, Expression as StorageExpr, InListExpr,
    LikeExpr, NotExpr, NullCheckExpr, OrExpr,
};

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
        let (column, value, flipped) =
            match extract_comparison_parts(&infix.left, &infix.right, ctx) {
                Some(parts) => parts,
                None => return PushdownResult::CannotPush,
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

        let column = match extract_column_name(&like.left) {
            Some(c) => c,
            None => return PushdownResult::CannotPush,
        };

        let pattern = match &*like.pattern {
            ast::Expression::StringLiteral(s) => s.value.clone(),
            _ => return PushdownResult::CannotPush,
        };

        let is_not = op_upper.contains("NOT");
        let is_ilike = op_upper.contains("ILIKE");

        let mut expr = match (is_not, is_ilike) {
            (true, true) => LikeExpr::not_ilike(column, pattern),
            (true, false) => LikeExpr::not_like(column, pattern),
            (false, true) => LikeExpr::new_ilike(column, pattern),
            (false, false) => LikeExpr::new(column, pattern),
        };
        expr.prepare_for_schema(ctx.schema);
        PushdownResult::Converted(Box::new(expr))
    }

    fn convert_infix_like(
        &self,
        infix: &ast::InfixExpression,
        ctx: &PushdownContext<'_>,
    ) -> PushdownResult {
        let (column, pattern, is_ilike, is_not) = match infix.op_type {
            InfixOperator::Like => {
                let col = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };
                let pattern = match extract_pattern(&infix.right) {
                    Some(p) => p,
                    None => return PushdownResult::CannotPush,
                };
                (col, pattern, false, false)
            }
            InfixOperator::ILike => {
                let col = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };
                let pattern = match extract_pattern(&infix.right) {
                    Some(p) => p,
                    None => return PushdownResult::CannotPush,
                };
                (col, pattern, true, false)
            }
            InfixOperator::NotLike => {
                let col = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };
                let pattern = match extract_pattern(&infix.right) {
                    Some(p) => p,
                    None => return PushdownResult::CannotPush,
                };
                (col, pattern, false, true)
            }
            InfixOperator::NotILike => {
                let col = match extract_column_name(&infix.left) {
                    Some(c) => c,
                    None => return PushdownResult::CannotPush,
                };
                let pattern = match extract_pattern(&infix.right) {
                    Some(p) => p,
                    None => return PushdownResult::CannotPush,
                };
                (col, pattern, true, true)
            }
            _ => return PushdownResult::NotApplicable,
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
