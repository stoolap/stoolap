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

//! Expression simplification pass for query optimization
//!
//! This module provides expression simplification that runs before planning.
//! It handles:
//! - Constant folding (1 + 1 → 2)
//! - Boolean simplification (TRUE AND x → x, FALSE OR x → x)
//! - Tautology elimination (1 = 1 → TRUE, x = x → TRUE for NOT NULL cols)
//! - Contradiction detection (1 = 2 → FALSE)
//! - Range predicate merging (a > 5 AND a > 3 → a > 5)
//! - De Morgan's law application where beneficial
//!
//! IMPORTANT: This module uses Option<Expression> returns to avoid cloning
//! expressions when no simplification is performed. This is critical for
//! performance with heavy expressions like EXISTS subqueries.

#![allow(clippy::only_used_in_recursion)]

use crate::parser::ast::{
    BooleanLiteral, Expression, InfixExpression, InfixOperator, IntegerLiteral, PrefixExpression,
    PrefixOperator,
};
use crate::parser::token::{Position, Token, TokenType};

/// Expression simplifier that applies optimization rules
pub struct ExpressionSimplifier {
    /// Track if any simplifications were made
    simplified: bool,
}

impl Default for ExpressionSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressionSimplifier {
    /// Create a new expression simplifier
    pub fn new() -> Self {
        Self { simplified: false }
    }

    /// Check if any simplifications were made in the last run
    pub fn was_simplified(&self) -> bool {
        self.simplified
    }

    /// Simplify an expression, applying all optimization rules.
    /// Returns Some(simplified) if changes were made, None if expression is unchanged.
    /// This avoids cloning expressions that can't be simplified (like EXISTS subqueries).
    pub fn try_simplify(&mut self, expr: &Expression) -> Option<Expression> {
        self.simplified = false;
        let result = self.simplify_recursive(expr);
        if self.simplified {
            Some(result.unwrap_or_else(|| expr.clone()))
        } else {
            None
        }
    }

    /// Simplify an expression, always returning an Expression.
    /// Use try_simplify() when you want to avoid cloning unchanged expressions.
    pub fn simplify(&mut self, expr: &Expression) -> Expression {
        self.simplified = false;
        self.simplify_recursive(expr)
            .unwrap_or_else(|| expr.clone())
    }

    /// Recursively simplify an expression.
    /// Returns Some(new_expr) if simplified, None if unchanged.
    fn simplify_recursive(&mut self, expr: &Expression) -> Option<Expression> {
        match expr {
            Expression::Infix(infix) => self.simplify_infix(infix),
            Expression::Prefix(prefix) => self.simplify_prefix(prefix),
            Expression::Between(between) => self.simplify_between(between),
            Expression::In(in_expr) => self.simplify_in(in_expr),
            // EXISTS and ScalarSubquery contain heavy SelectStatement data.
            // They can't be simplified to constants, so return None (no clone needed).
            Expression::Exists(_) | Expression::ScalarSubquery(_) => None,
            // InHashSet, literals, identifiers, etc. can't be simplified
            _ => None,
        }
    }

    /// Simplify BETWEEN expression
    fn simplify_between(
        &mut self,
        between: &crate::parser::ast::BetweenExpression,
    ) -> Option<Expression> {
        let value_simplified = self.simplify_recursive(&between.expr);
        let lower_simplified = self.simplify_recursive(&between.lower);
        let upper_simplified = self.simplify_recursive(&between.upper);

        let value = value_simplified.as_ref().unwrap_or(&between.expr);
        let lower = lower_simplified.as_ref().unwrap_or(&between.lower);
        let upper = upper_simplified.as_ref().unwrap_or(&between.upper);

        // Check for constant BETWEEN
        if let (
            Expression::IntegerLiteral(v),
            Expression::IntegerLiteral(l),
            Expression::IntegerLiteral(h),
        ) = (value, lower, upper)
        {
            self.simplified = true;
            let result = v.value >= l.value && v.value <= h.value;
            return Some(if between.not {
                self.make_bool(!result)
            } else {
                self.make_bool(result)
            });
        }

        // If any child was simplified, rebuild the expression
        if value_simplified.is_some() || lower_simplified.is_some() || upper_simplified.is_some() {
            Some(Expression::Between(crate::parser::ast::BetweenExpression {
                token: between.token.clone(),
                expr: Box::new(value_simplified.unwrap_or_else(|| (*between.expr).clone())),
                lower: Box::new(lower_simplified.unwrap_or_else(|| (*between.lower).clone())),
                upper: Box::new(upper_simplified.unwrap_or_else(|| (*between.upper).clone())),
                not: between.not,
            }))
        } else {
            None
        }
    }

    /// Simplify IN expression
    fn simplify_in(&mut self, in_expr: &crate::parser::ast::InExpression) -> Option<Expression> {
        let value_simplified = self.simplify_recursive(&in_expr.left);
        let list_simplified = self.simplify_recursive(&in_expr.right);

        // If any child was simplified, rebuild the expression
        if value_simplified.is_some() || list_simplified.is_some() {
            Some(Expression::In(crate::parser::ast::InExpression {
                token: in_expr.token.clone(),
                left: Box::new(value_simplified.unwrap_or_else(|| (*in_expr.left).clone())),
                right: Box::new(list_simplified.unwrap_or_else(|| (*in_expr.right).clone())),
                not: in_expr.not,
            }))
        } else {
            None
        }
    }

    /// Simplify infix (binary) expressions
    fn simplify_infix(&mut self, infix: &InfixExpression) -> Option<Expression> {
        // First, recursively simplify children
        let left_simplified = self.simplify_recursive(&infix.left);
        let right_simplified = self.simplify_recursive(&infix.right);

        let left = left_simplified.as_ref().unwrap_or(&infix.left);
        let right = right_simplified.as_ref().unwrap_or(&infix.right);

        // Try operator-specific simplifications
        let simplified_result = match infix.op_type {
            InfixOperator::And => self.simplify_and(left, right),
            InfixOperator::Or => self.simplify_or(left, right),
            InfixOperator::Equal => self.simplify_equal(left, right),
            InfixOperator::NotEqual => self.simplify_not_equal(left, right),
            InfixOperator::LessThan => self.simplify_less_than(left, right),
            InfixOperator::LessEqual => self.simplify_less_equal(left, right),
            InfixOperator::GreaterThan => self.simplify_greater_than(left, right),
            InfixOperator::GreaterEqual => self.simplify_greater_equal(left, right),
            InfixOperator::Add => self.simplify_add(left, right),
            InfixOperator::Subtract => self.simplify_subtract(left, right),
            InfixOperator::Multiply => self.simplify_multiply(left, right),
            InfixOperator::Divide => self.simplify_divide(left, right),
            _ => None,
        };

        if let Some(result) = simplified_result {
            self.simplified = true;
            return Some(result);
        }

        // If children were simplified but operator wasn't, rebuild with simplified children
        if left_simplified.is_some() || right_simplified.is_some() {
            Some(Expression::Infix(InfixExpression {
                token: infix.token.clone(),
                left: Box::new(left_simplified.unwrap_or_else(|| (*infix.left).clone())),
                operator: infix.operator.clone(),
                op_type: infix.op_type,
                right: Box::new(right_simplified.unwrap_or_else(|| (*infix.right).clone())),
            }))
        } else {
            None
        }
    }

    /// Simplify AND expressions
    fn simplify_and(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // TRUE AND x → x
        if self.is_always_true(left) {
            return Some(right.clone());
        }
        // x AND TRUE → x
        if self.is_always_true(right) {
            return Some(left.clone());
        }
        // FALSE AND x → FALSE
        if self.is_always_false(left) {
            return Some(left.clone());
        }
        // x AND FALSE → FALSE
        if self.is_always_false(right) {
            return Some(right.clone());
        }
        // x AND x → x
        if self.expr_equals(left, right) {
            return Some(left.clone());
        }
        // Try to merge range predicates: a > 5 AND a > 3 → a > 5
        self.try_merge_range_predicates(left, right)
    }

    /// Simplify OR expressions
    fn simplify_or(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // FALSE OR x → x
        if self.is_always_false(left) {
            return Some(right.clone());
        }
        // x OR FALSE → x
        if self.is_always_false(right) {
            return Some(left.clone());
        }
        // TRUE OR x → TRUE
        if self.is_always_true(left) {
            return Some(left.clone());
        }
        // x OR TRUE → TRUE
        if self.is_always_true(right) {
            return Some(right.clone());
        }
        // x OR x → x
        if self.expr_equals(left, right) {
            return Some(left.clone());
        }
        None
    }

    /// Simplify equality expressions
    fn simplify_equal(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // Constant comparison: 1 = 1 → TRUE, 1 = 2 → FALSE
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::Equal) {
            return Some(self.make_bool(result));
        }
        // x = x → TRUE (for deterministic expressions)
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(true));
        }
        None
    }

    /// Simplify not-equal expressions
    fn simplify_not_equal(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::NotEqual) {
            return Some(self.make_bool(result));
        }
        // x <> x → FALSE (for deterministic expressions)
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(false));
        }
        None
    }

    /// Simplify less-than expressions
    fn simplify_less_than(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::LessThan) {
            return Some(self.make_bool(result));
        }
        // x < x → FALSE
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(false));
        }
        None
    }

    /// Simplify less-equal expressions
    fn simplify_less_equal(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::LessEqual) {
            return Some(self.make_bool(result));
        }
        // x <= x → TRUE
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(true));
        }
        None
    }

    /// Simplify greater-than expressions
    fn simplify_greater_than(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::GreaterThan) {
            return Some(self.make_bool(result));
        }
        // x > x → FALSE
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(false));
        }
        None
    }

    /// Simplify greater-equal expressions
    fn simplify_greater_equal(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        if let Some(result) = self.try_eval_comparison(left, right, InfixOperator::GreaterEqual) {
            return Some(self.make_bool(result));
        }
        // x >= x → TRUE
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_bool(true));
        }
        None
    }

    /// Simplify addition expressions
    fn simplify_add(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // x + 0 → x
        if self.is_zero(right) {
            return Some(left.clone());
        }
        // 0 + x → x
        if self.is_zero(left) {
            return Some(right.clone());
        }
        // Constant folding
        self.try_eval_arithmetic(left, right, InfixOperator::Add)
    }

    /// Simplify subtraction expressions
    fn simplify_subtract(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // x - 0 → x
        if self.is_zero(right) {
            return Some(left.clone());
        }
        // x - x → 0
        if self.is_deterministic(left) && self.expr_equals(left, right) {
            return Some(self.make_int(0));
        }
        // Constant folding
        self.try_eval_arithmetic(left, right, InfixOperator::Subtract)
    }

    /// Simplify multiplication expressions
    fn simplify_multiply(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // x * 1 → x
        if self.is_one(right) {
            return Some(left.clone());
        }
        // 1 * x → x
        if self.is_one(left) {
            return Some(right.clone());
        }
        // x * 0 → 0
        if self.is_zero(right) {
            return Some(self.make_int(0));
        }
        // 0 * x → 0
        if self.is_zero(left) {
            return Some(self.make_int(0));
        }
        // Constant folding
        self.try_eval_arithmetic(left, right, InfixOperator::Multiply)
    }

    /// Simplify division expressions
    fn simplify_divide(&self, left: &Expression, right: &Expression) -> Option<Expression> {
        // x / 1 → x
        if self.is_one(right) {
            return Some(left.clone());
        }
        // 0 / x → 0 (if x is non-zero constant)
        if self.is_zero(left) && self.is_nonzero_constant(right) {
            return Some(self.make_int(0));
        }
        None
    }

    /// Simplify prefix (unary) expressions
    fn simplify_prefix(&mut self, prefix: &PrefixExpression) -> Option<Expression> {
        let operand_simplified = self.simplify_recursive(&prefix.right);
        let operand = operand_simplified.as_ref().unwrap_or(&prefix.right);

        let simplified_result = match prefix.op_type {
            PrefixOperator::Not => {
                // NOT TRUE → FALSE
                if self.is_always_true(operand) {
                    Some(self.make_bool(false))
                }
                // NOT FALSE → TRUE
                else if self.is_always_false(operand) {
                    Some(self.make_bool(true))
                }
                // NOT NOT x → x
                else if let Expression::Prefix(inner) = operand {
                    if inner.op_type == PrefixOperator::Not {
                        Some((*inner.right).clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            PrefixOperator::Negate => {
                // -(-x) → x
                if let Expression::Prefix(inner) = operand {
                    if inner.op_type == PrefixOperator::Negate {
                        Some((*inner.right).clone())
                    } else {
                        None
                    }
                }
                // -(constant) → constant negated
                else if let Expression::IntegerLiteral(lit) = operand {
                    Some(self.make_int(-lit.value))
                } else {
                    None
                }
            }
            PrefixOperator::Plus => {
                // +x → x
                Some(operand.clone())
            }
            _ => None,
        };

        if let Some(result) = simplified_result {
            self.simplified = true;
            return Some(result);
        }

        // If operand was simplified, rebuild the expression
        operand_simplified.map(|simplified_operand| {
            Expression::Prefix(PrefixExpression {
                token: prefix.token.clone(),
                operator: prefix.operator.clone(),
                op_type: prefix.op_type,
                right: Box::new(simplified_operand),
            })
        })
    }

    /// Check if expression is always TRUE
    fn is_always_true(&self, expr: &Expression) -> bool {
        match expr {
            Expression::BooleanLiteral(b) => b.value,
            Expression::Infix(infix) => {
                // 1 = 1 is always true
                if infix.op_type == InfixOperator::Equal
                    && self.expr_equals(&infix.left, &infix.right)
                    && self.is_deterministic(&infix.left)
                {
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    /// Check if expression is always FALSE
    fn is_always_false(&self, expr: &Expression) -> bool {
        match expr {
            Expression::BooleanLiteral(b) => !b.value,
            Expression::Infix(infix) => {
                // 1 = 2 is always false
                if infix.op_type == InfixOperator::Equal {
                    if let (Expression::IntegerLiteral(l), Expression::IntegerLiteral(r)) =
                        (infix.left.as_ref(), infix.right.as_ref())
                    {
                        return l.value != r.value;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Check if expression is deterministic (same input → same output)
    fn is_deterministic(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_) => true,
            Expression::Infix(infix) => {
                self.is_deterministic(&infix.left) && self.is_deterministic(&infix.right)
            }
            Expression::Prefix(prefix) => self.is_deterministic(&prefix.right),
            Expression::FunctionCall(func) => {
                // Most functions are deterministic, but not RANDOM(), NOW(), etc.
                let name = func.function.to_uppercase();
                if matches!(
                    name.as_str(),
                    "RANDOM" | "NOW" | "CURRENT_TIMESTAMP" | "UUID"
                ) {
                    return false;
                }
                func.arguments.iter().all(|a| self.is_deterministic(a))
            }
            _ => false,
        }
    }

    /// Check if two expressions are structurally equal
    fn expr_equals(&self, a: &Expression, b: &Expression) -> bool {
        match (a, b) {
            (Expression::Identifier(a), Expression::Identifier(b)) => a.value == b.value,
            (Expression::QualifiedIdentifier(a), Expression::QualifiedIdentifier(b)) => {
                a.qualifier.value == b.qualifier.value && a.name.value == b.name.value
            }
            (Expression::IntegerLiteral(a), Expression::IntegerLiteral(b)) => a.value == b.value,
            (Expression::FloatLiteral(a), Expression::FloatLiteral(b)) => a.value == b.value,
            (Expression::StringLiteral(a), Expression::StringLiteral(b)) => a.value == b.value,
            (Expression::BooleanLiteral(a), Expression::BooleanLiteral(b)) => a.value == b.value,
            (Expression::NullLiteral(_), Expression::NullLiteral(_)) => true,
            (Expression::Infix(a), Expression::Infix(b)) => {
                a.op_type == b.op_type
                    && self.expr_equals(&a.left, &b.left)
                    && self.expr_equals(&a.right, &b.right)
            }
            (Expression::Prefix(a), Expression::Prefix(b)) => {
                a.op_type == b.op_type && self.expr_equals(&a.right, &b.right)
            }
            _ => false,
        }
    }

    /// Check if expression is integer literal 0
    fn is_zero(&self, expr: &Expression) -> bool {
        matches!(expr, Expression::IntegerLiteral(lit) if lit.value == 0)
    }

    /// Check if expression is integer literal 1
    fn is_one(&self, expr: &Expression) -> bool {
        matches!(expr, Expression::IntegerLiteral(lit) if lit.value == 1)
    }

    /// Check if expression is a non-zero constant
    fn is_nonzero_constant(&self, expr: &Expression) -> bool {
        match expr {
            Expression::IntegerLiteral(lit) => lit.value != 0,
            Expression::FloatLiteral(lit) => lit.value != 0.0,
            _ => false,
        }
    }

    /// Try to evaluate a comparison between constants
    fn try_eval_comparison(
        &self,
        left: &Expression,
        right: &Expression,
        op: InfixOperator,
    ) -> Option<bool> {
        match (left, right) {
            (Expression::IntegerLiteral(l), Expression::IntegerLiteral(r)) => Some(match op {
                InfixOperator::Equal => l.value == r.value,
                InfixOperator::NotEqual => l.value != r.value,
                InfixOperator::LessThan => l.value < r.value,
                InfixOperator::LessEqual => l.value <= r.value,
                InfixOperator::GreaterThan => l.value > r.value,
                InfixOperator::GreaterEqual => l.value >= r.value,
                _ => return None,
            }),
            (Expression::FloatLiteral(l), Expression::FloatLiteral(r)) => Some(match op {
                InfixOperator::Equal => l.value == r.value,
                InfixOperator::NotEqual => l.value != r.value,
                InfixOperator::LessThan => l.value < r.value,
                InfixOperator::LessEqual => l.value <= r.value,
                InfixOperator::GreaterThan => l.value > r.value,
                InfixOperator::GreaterEqual => l.value >= r.value,
                _ => return None,
            }),
            (Expression::StringLiteral(l), Expression::StringLiteral(r)) => Some(match op {
                InfixOperator::Equal => l.value == r.value,
                InfixOperator::NotEqual => l.value != r.value,
                InfixOperator::LessThan => l.value < r.value,
                InfixOperator::LessEqual => l.value <= r.value,
                InfixOperator::GreaterThan => l.value > r.value,
                InfixOperator::GreaterEqual => l.value >= r.value,
                _ => return None,
            }),
            (Expression::BooleanLiteral(l), Expression::BooleanLiteral(r)) => Some(match op {
                InfixOperator::Equal => l.value == r.value,
                InfixOperator::NotEqual => l.value != r.value,
                _ => return None,
            }),
            _ => None,
        }
    }

    /// Try to evaluate arithmetic on constants
    fn try_eval_arithmetic(
        &self,
        left: &Expression,
        right: &Expression,
        op: InfixOperator,
    ) -> Option<Expression> {
        match (left, right) {
            (Expression::IntegerLiteral(l), Expression::IntegerLiteral(r)) => {
                let result = match op {
                    InfixOperator::Add => l.value.checked_add(r.value)?,
                    InfixOperator::Subtract => l.value.checked_sub(r.value)?,
                    InfixOperator::Multiply => l.value.checked_mul(r.value)?,
                    InfixOperator::Divide => {
                        if r.value == 0 {
                            return None;
                        }
                        l.value.checked_div(r.value)?
                    }
                    InfixOperator::Modulo => {
                        if r.value == 0 {
                            return None;
                        }
                        l.value.checked_rem(r.value)?
                    }
                    _ => return None,
                };
                Some(self.make_int(result))
            }
            _ => None,
        }
    }

    /// Try to merge overlapping range predicates
    /// a > 5 AND a > 3 → a > 5
    /// a < 5 AND a < 10 → a < 5
    fn try_merge_range_predicates(
        &self,
        left: &Expression,
        right: &Expression,
    ) -> Option<Expression> {
        let (left_infix, right_infix) = match (left, right) {
            (Expression::Infix(l), Expression::Infix(r)) => (l, r),
            _ => return None,
        };

        // Check if both are comparisons on the same column
        let left_col = self.extract_column(&left_infix.left)?;
        let right_col = self.extract_column(&right_infix.left)?;

        if left_col != right_col {
            return None;
        }

        // Check for constant right-hand sides
        let left_val = self.extract_int_literal(&left_infix.right)?;
        let right_val = self.extract_int_literal(&right_infix.right)?;

        // Merge based on operator types
        match (left_infix.op_type, right_infix.op_type) {
            // a > 5 AND a > 3 → a > 5 (keep larger)
            (InfixOperator::GreaterThan, InfixOperator::GreaterThan)
            | (InfixOperator::GreaterEqual, InfixOperator::GreaterEqual) => {
                if left_val >= right_val {
                    Some(left.clone())
                } else {
                    Some(right.clone())
                }
            }
            // a < 5 AND a < 10 → a < 5 (keep smaller)
            (InfixOperator::LessThan, InfixOperator::LessThan)
            | (InfixOperator::LessEqual, InfixOperator::LessEqual) => {
                if left_val <= right_val {
                    Some(left.clone())
                } else {
                    Some(right.clone())
                }
            }
            // a > 5 AND a >= 5 → a > 5 (stricter)
            (InfixOperator::GreaterThan, InfixOperator::GreaterEqual) if left_val == right_val => {
                Some(left.clone())
            }
            (InfixOperator::GreaterEqual, InfixOperator::GreaterThan) if left_val == right_val => {
                Some(right.clone())
            }
            // a < 5 AND a <= 5 → a < 5 (stricter)
            (InfixOperator::LessThan, InfixOperator::LessEqual) if left_val == right_val => {
                Some(left.clone())
            }
            (InfixOperator::LessEqual, InfixOperator::LessThan) if left_val == right_val => {
                Some(right.clone())
            }
            _ => None,
        }
    }

    /// Extract column name from identifier expression
    fn extract_column(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value.clone()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value.clone()),
            _ => None,
        }
    }

    /// Extract integer literal value
    fn extract_int_literal(&self, expr: &Expression) -> Option<i64> {
        match expr {
            Expression::IntegerLiteral(lit) => Some(lit.value),
            _ => None,
        }
    }

    /// Create a boolean literal expression
    fn make_bool(&self, value: bool) -> Expression {
        Expression::BooleanLiteral(BooleanLiteral {
            token: Token::new(
                TokenType::Keyword,
                if value { "TRUE" } else { "FALSE" },
                Position::default(),
            ),
            value,
        })
    }

    /// Create an integer literal expression
    fn make_int(&self, value: i64) -> Expression {
        Expression::IntegerLiteral(IntegerLiteral {
            token: Token::new(TokenType::Integer, value.to_string(), Position::default()),
            value,
        })
    }
}

/// Convenience function to simplify an expression
pub fn simplify_expression(expr: &Expression) -> Expression {
    let mut simplifier = ExpressionSimplifier::new();
    simplifier.simplify(expr)
}

/// Repeatedly simplify until no more changes
pub fn simplify_expression_fixed_point(expr: &Expression) -> Expression {
    let mut simplifier = ExpressionSimplifier::new();
    let mut current = expr.clone();
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10;

    loop {
        let simplified = simplifier.try_simplify(&current);
        if simplified.is_none() || iterations >= MAX_ITERATIONS {
            break;
        }
        current = simplified.unwrap();
        iterations += 1;
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Identifier;

    fn make_int_lit(value: i64) -> Expression {
        Expression::IntegerLiteral(IntegerLiteral {
            token: Token::new(TokenType::Integer, value.to_string(), Position::default()),
            value,
        })
    }

    fn make_bool_lit(value: bool) -> Expression {
        Expression::BooleanLiteral(BooleanLiteral {
            token: Token::new(
                TokenType::Keyword,
                if value { "TRUE" } else { "FALSE" },
                Position::default(),
            ),
            value,
        })
    }

    fn make_identifier(name: &str) -> Expression {
        Expression::Identifier(Identifier::new(
            Token::new(TokenType::Identifier, name, Position::default()),
            name.to_string(),
        ))
    }

    fn make_infix(left: Expression, op: InfixOperator, right: Expression) -> Expression {
        let op_str = match op {
            InfixOperator::And => "AND",
            InfixOperator::Or => "OR",
            InfixOperator::Equal => "=",
            InfixOperator::NotEqual => "<>",
            InfixOperator::LessThan => "<",
            InfixOperator::GreaterThan => ">",
            InfixOperator::Add => "+",
            InfixOperator::Subtract => "-",
            InfixOperator::Multiply => "*",
            _ => "?",
        };
        Expression::Infix(InfixExpression {
            token: Token::new(TokenType::Operator, op_str, Position::default()),
            left: Box::new(left),
            operator: op_str.to_string(),
            op_type: op,
            right: Box::new(right),
        })
    }

    #[test]
    fn test_constant_folding_arithmetic() {
        let expr = make_infix(make_int_lit(2), InfixOperator::Add, make_int_lit(3));
        let result = simplify_expression(&expr);

        if let Expression::IntegerLiteral(lit) = result {
            assert_eq!(lit.value, 5);
        } else {
            panic!("Expected IntegerLiteral");
        }
    }

    #[test]
    fn test_constant_folding_comparison() {
        // 1 = 1 → TRUE
        let expr = make_infix(make_int_lit(1), InfixOperator::Equal, make_int_lit(1));
        let result = simplify_expression(&expr);

        if let Expression::BooleanLiteral(lit) = result {
            assert!(lit.value);
        } else {
            panic!("Expected BooleanLiteral(true)");
        }

        // 1 = 2 → FALSE
        let expr = make_infix(make_int_lit(1), InfixOperator::Equal, make_int_lit(2));
        let result = simplify_expression(&expr);

        if let Expression::BooleanLiteral(lit) = result {
            assert!(!lit.value);
        } else {
            panic!("Expected BooleanLiteral(false)");
        }
    }

    #[test]
    fn test_boolean_and_simplification() {
        // TRUE AND x → x
        let x = make_identifier("x");
        let expr = make_infix(make_bool_lit(true), InfixOperator::And, x.clone());
        let result = simplify_expression(&expr);

        if let Expression::Identifier(id) = result {
            assert_eq!(id.value, "x");
        } else {
            panic!("Expected Identifier 'x'");
        }

        // FALSE AND x → FALSE
        let expr = make_infix(make_bool_lit(false), InfixOperator::And, x.clone());
        let result = simplify_expression(&expr);

        if let Expression::BooleanLiteral(lit) = result {
            assert!(!lit.value);
        } else {
            panic!("Expected BooleanLiteral(false)");
        }
    }

    #[test]
    fn test_boolean_or_simplification() {
        // FALSE OR x → x
        let x = make_identifier("x");
        let expr = make_infix(make_bool_lit(false), InfixOperator::Or, x.clone());
        let result = simplify_expression(&expr);

        if let Expression::Identifier(id) = result {
            assert_eq!(id.value, "x");
        } else {
            panic!("Expected Identifier 'x'");
        }

        // TRUE OR x → TRUE
        let expr = make_infix(make_bool_lit(true), InfixOperator::Or, x.clone());
        let result = simplify_expression(&expr);

        if let Expression::BooleanLiteral(lit) = result {
            assert!(lit.value);
        } else {
            panic!("Expected BooleanLiteral(true)");
        }
    }

    #[test]
    fn test_arithmetic_identity() {
        let x = make_identifier("x");

        // x + 0 → x
        let expr = make_infix(x.clone(), InfixOperator::Add, make_int_lit(0));
        let result = simplify_expression(&expr);
        assert!(matches!(result, Expression::Identifier(_)));

        // x * 1 → x
        let expr = make_infix(x.clone(), InfixOperator::Multiply, make_int_lit(1));
        let result = simplify_expression(&expr);
        assert!(matches!(result, Expression::Identifier(_)));

        // x * 0 → 0
        let expr = make_infix(x.clone(), InfixOperator::Multiply, make_int_lit(0));
        let result = simplify_expression(&expr);
        if let Expression::IntegerLiteral(lit) = result {
            assert_eq!(lit.value, 0);
        } else {
            panic!("Expected IntegerLiteral(0)");
        }
    }

    #[test]
    fn test_range_predicate_merge() {
        let a = make_identifier("a");

        // a > 5 AND a > 3 → a > 5
        let left = make_infix(a.clone(), InfixOperator::GreaterThan, make_int_lit(5));
        let right = make_infix(a.clone(), InfixOperator::GreaterThan, make_int_lit(3));
        let expr = make_infix(left, InfixOperator::And, right);
        let result = simplify_expression(&expr);

        // Should keep the stricter condition (a > 5)
        if let Expression::Infix(infix) = result {
            if let Expression::IntegerLiteral(lit) = infix.right.as_ref() {
                assert_eq!(lit.value, 5);
            } else {
                panic!("Expected IntegerLiteral(5)");
            }
        } else {
            panic!("Expected Infix expression");
        }
    }

    #[test]
    fn test_idempotent_and() {
        let x = make_identifier("x");

        // x AND x → x
        let expr = make_infix(x.clone(), InfixOperator::And, x.clone());
        let result = simplify_expression(&expr);
        assert!(matches!(result, Expression::Identifier(_)));
    }

    #[test]
    fn test_idempotent_or() {
        let x = make_identifier("x");

        // x OR x → x
        let expr = make_infix(x.clone(), InfixOperator::Or, x.clone());
        let result = simplify_expression(&expr);
        assert!(matches!(result, Expression::Identifier(_)));
    }

    #[test]
    fn test_self_comparison() {
        let x = make_identifier("x");

        // x = x → TRUE (for deterministic x)
        let expr = make_infix(x.clone(), InfixOperator::Equal, x.clone());
        let result = simplify_expression(&expr);
        if let Expression::BooleanLiteral(lit) = result {
            assert!(lit.value);
        } else {
            panic!("Expected BooleanLiteral(true)");
        }

        // x < x → FALSE
        let expr = make_infix(x.clone(), InfixOperator::LessThan, x.clone());
        let result = simplify_expression(&expr);
        if let Expression::BooleanLiteral(lit) = result {
            assert!(!lit.value);
        } else {
            panic!("Expected BooleanLiteral(false)");
        }
    }

    #[test]
    fn test_subtraction_identity() {
        let x = make_identifier("x");

        // x - x → 0
        let expr = make_infix(x.clone(), InfixOperator::Subtract, x.clone());
        let result = simplify_expression(&expr);
        if let Expression::IntegerLiteral(lit) = result {
            assert_eq!(lit.value, 0);
        } else {
            panic!("Expected IntegerLiteral(0)");
        }
    }

    #[test]
    fn test_no_clone_for_unchanged() {
        // Test that try_simplify returns None for expressions that can't be simplified
        let mut simplifier = ExpressionSimplifier::new();
        let x = make_identifier("x");

        // Simple identifier can't be simplified
        let result = simplifier.try_simplify(&x);
        assert!(result.is_none());
        assert!(!simplifier.was_simplified());
    }
}
