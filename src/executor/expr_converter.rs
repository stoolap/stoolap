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

//! Expression Converter for Predicate Pushdown
//!
//! This module converts parser AST expressions to storage layer expressions
//! for predicate pushdown optimization. Only simple, pushable predicates
//! are converted - complex expressions that can't be pushed down return None.
//!
//! ## Supported Conversions
//!
//! | AST Expression          | Storage Expression |
//! |-------------------------|-------------------|
//! | `column = value`        | `ComparisonExpr`  |
//! | `column > value`        | `ComparisonExpr`  |
//! | `a AND b`               | `AndExpr`         |
//! | `a OR b`                | `OrExpr`          |
//! | `NOT a`                 | `NotExpr`         |
//! | `column IN (v1, v2)`    | `InListExpr`      |
//! | `column BETWEEN a AND b`| `BetweenExpr`     |
//! | `column LIKE pattern`   | `LikeExpr`        |
//! | `column IS NULL`        | `NullCheckExpr`   |
//!
//! ## Not Pushable (returns None)
//!
//! - Function calls (UPPER(col) = 'X')
//! - Subqueries
//! - CASE expressions
//! - Arithmetic expressions
//! - Column-to-column comparisons

use crate::core::{Operator, Value};
use crate::executor::utils::{dummy_token, extract_column_name, extract_literal_value, flip_operator, infix_to_operator};
use crate::parser::ast::{self as ast, InfixOperator, PrefixOperator};
use crate::parser::token::TokenType;
use crate::storage::expression::{
    AndExpr, BetweenExpr, ComparisonExpr, Expression, InListExpr, LikeExpr, NotExpr, NullCheckExpr,
    OrExpr,
};

/// Convert an AST expression to a storage expression for predicate pushdown
///
/// Returns `Some(expression)` if the expression can be pushed down to storage,
/// or `None` if it must be evaluated in memory.
pub fn convert_ast_to_storage_expr(expr: &ast::Expression) -> Option<Box<dyn Expression>> {
    match expr {
        // Infix expressions: comparisons and logical operators
        ast::Expression::Infix(infix) => convert_infix(infix),

        // Prefix expressions: NOT
        ast::Expression::Prefix(prefix) => convert_prefix(prefix),

        // IN expression
        ast::Expression::In(in_expr) => convert_in(in_expr),

        // BETWEEN expression
        ast::Expression::Between(between) => convert_between(between),

        // LIKE expression
        ast::Expression::Like(like) => convert_like(like),

        // Boolean literal - can be used as constant expression
        ast::Expression::BooleanLiteral(b) => {
            // Create a constant true/false expression
            Some(Box::new(crate::storage::expression::ConstBoolExpr::new(
                b.value,
            )))
        }

        // These expressions cannot be pushed down
        ast::Expression::FunctionCall(_)
        | ast::Expression::ScalarSubquery(_)
        | ast::Expression::Exists(_)
        | ast::Expression::AllAny(_)
        | ast::Expression::Case(_)
        | ast::Expression::Cast(_)
        | ast::Expression::Window(_)
        | ast::Expression::Parameter(_)
        | ast::Expression::IntervalLiteral(_) => None,

        // Simple expressions that don't make sense as filters
        ast::Expression::Identifier(_)
        | ast::Expression::QualifiedIdentifier(_)
        | ast::Expression::IntegerLiteral(_)
        | ast::Expression::FloatLiteral(_)
        | ast::Expression::StringLiteral(_)
        | ast::Expression::NullLiteral(_)
        | ast::Expression::List(_)
        | ast::Expression::Distinct(_)
        | ast::Expression::ExpressionList(_)
        | ast::Expression::Aliased(_) => None,

        // Table sources and other non-predicate expressions
        ast::Expression::TableSource(_)
        | ast::Expression::JoinSource(_)
        | ast::Expression::SubquerySource(_)
        | ast::Expression::ValuesSource(_)
        | ast::Expression::CteReference(_)
        | ast::Expression::Star(_)
        | ast::Expression::QualifiedStar(_)
        | ast::Expression::Default(_) => None,
    }
}

/// Convert an infix expression (comparison or logical)
fn convert_infix(infix: &ast::InfixExpression) -> Option<Box<dyn Expression>> {
    match infix.op_type {
        // Comparison operators - need column on one side, value on other
        InfixOperator::Equal
        | InfixOperator::NotEqual
        | InfixOperator::LessThan
        | InfixOperator::LessEqual
        | InfixOperator::GreaterThan
        | InfixOperator::GreaterEqual => convert_comparison(infix),

        // Logical operators
        InfixOperator::And => {
            let left = convert_ast_to_storage_expr(&infix.left)?;
            let right = convert_ast_to_storage_expr(&infix.right)?;
            Some(Box::new(AndExpr::and(left, right)))
        }

        InfixOperator::Or => {
            let left = convert_ast_to_storage_expr(&infix.left)?;
            let right = convert_ast_to_storage_expr(&infix.right)?;
            Some(Box::new(OrExpr::or(left, right)))
        }

        // IS NULL / IS NOT NULL
        InfixOperator::Is | InfixOperator::IsNot => convert_is_null(infix),

        // All other operators can't be pushed down directly
        _ => None,
    }
}

/// Convert a comparison expression (column op value or value op column)
fn convert_comparison(infix: &ast::InfixExpression) -> Option<Box<dyn Expression>> {
    // Try column on left, value on right
    if let Some((column, value)) = try_extract_column_value(&infix.left, &infix.right) {
        let operator = infix_to_operator(infix.op_type)?;
        return Some(Box::new(ComparisonExpr::new(column, operator, value)));
    }

    // Try value on left, column on right (flip the operator)
    if let Some((column, value)) = try_extract_column_value(&infix.right, &infix.left) {
        let operator = flip_operator(infix_to_operator(infix.op_type)?);
        return Some(Box::new(ComparisonExpr::new(column, operator, value)));
    }

    // Neither pattern matched - can't push down
    None
}

/// Try to extract a column name and literal value from two expressions
fn try_extract_column_value(
    maybe_column: &ast::Expression,
    maybe_value: &ast::Expression,
) -> Option<(String, Value)> {
    let column = extract_column_name(maybe_column)?;
    let value = extract_literal_value(maybe_value)?;
    Some((column, value))
}


/// Convert IS NULL / IS NOT NULL expression
fn convert_is_null(infix: &ast::InfixExpression) -> Option<Box<dyn Expression>> {
    let column = extract_column_name(&infix.left)?;

    // Check if it's IS or IS NOT
    let is_not = infix.op_type == InfixOperator::IsNot;

    // Check if right side is NULL literal
    match &*infix.right {
        ast::Expression::NullLiteral(_) => {
            if is_not {
                Some(Box::new(NullCheckExpr::is_not_null(column)))
            } else {
                Some(Box::new(NullCheckExpr::is_null(column)))
            }
        }
        _ => None,
    }
}

/// Convert a prefix expression (NOT)
fn convert_prefix(prefix: &ast::PrefixExpression) -> Option<Box<dyn Expression>> {
    match prefix.op_type {
        PrefixOperator::Not => {
            let inner = convert_ast_to_storage_expr(&prefix.right)?;
            Some(Box::new(NotExpr::new(inner)))
        }
        // Arithmetic negation and other operators can't be pushed down
        _ => None,
    }
}

/// Convert an IN expression
fn convert_in(in_expr: &ast::InExpression) -> Option<Box<dyn Expression>> {
    let column = extract_column_name(&in_expr.left)?;

    // Extract values from the IN list
    let values: Option<Vec<Value>> = match &*in_expr.right {
        ast::Expression::ExpressionList(list) => {
            list.expressions.iter().map(extract_literal_value).collect()
        }
        ast::Expression::List(list) => list.elements.iter().map(extract_literal_value).collect(),
        _ => return None, // Subquery IN - can't push down
    };

    let values = values?;
    if values.is_empty() {
        return None;
    }

    let expr = if in_expr.not {
        InListExpr::not_in(column, values)
    } else {
        InListExpr::new(column, values)
    };
    Some(Box::new(expr))
}

/// Convert a BETWEEN expression
fn convert_between(between: &ast::BetweenExpression) -> Option<Box<dyn Expression>> {
    let column = extract_column_name(&between.expr)?;
    let low = extract_literal_value(&between.lower)?;
    let high = extract_literal_value(&between.upper)?;

    let expr = if between.not {
        BetweenExpr::not_between(column, low, high)
    } else {
        BetweenExpr::new(column, low, high)
    };
    Some(Box::new(expr))
}

/// Convert a LIKE expression
fn convert_like(like: &ast::LikeExpression) -> Option<Box<dyn Expression>> {
    let column = extract_column_name(&like.left)?;

    // Extract pattern string
    let pattern = match &*like.pattern {
        ast::Expression::StringLiteral(s) => s.value.clone(),
        _ => return None, // Pattern must be a string literal
    };

    // Parse operator to determine case-sensitivity and negation
    let op_upper = like.operator.to_uppercase();
    let is_not = op_upper.contains("NOT");
    let is_ilike = op_upper.contains("ILIKE");

    // Note: escape character is not directly supported in our LikeExpr yet
    // but could be added if needed

    let expr = match (is_not, is_ilike) {
        (true, true) => LikeExpr::not_ilike(column, pattern),
        (true, false) => LikeExpr::not_like(column, pattern),
        (false, true) => LikeExpr::new_ilike(column, pattern),
        (false, false) => LikeExpr::new(column, pattern),
    };
    Some(Box::new(expr))
}

/// Split an expression into pushable and non-pushable parts
///
/// For AND expressions, extracts the parts that can be pushed down.
/// Returns (pushable, remaining) where remaining is None if everything was pushed.
pub fn split_pushable_predicates(
    expr: &ast::Expression,
) -> (Option<Box<dyn Expression>>, Option<ast::Expression>) {
    match expr {
        ast::Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
            // For AND, we can push down individual parts even if others can't be pushed
            let (left_push, left_remain) = split_pushable_predicates(&infix.left);
            let (right_push, right_remain) = split_pushable_predicates(&infix.right);

            // Combine pushable parts
            let pushable = match (left_push, right_push) {
                (Some(l), Some(r)) => Some(Box::new(AndExpr::and(l, r)) as Box<dyn Expression>),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            };

            // Combine remaining parts
            let remaining = match (left_remain, right_remain) {
                (Some(l), Some(r)) => Some(ast::Expression::Infix(ast::InfixExpression::new(
                    dummy_token("AND", TokenType::Keyword),
                    Box::new(l),
                    "AND".to_string(),
                    Box::new(r),
                ))),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            };

            (pushable, remaining)
        }

        _ => {
            // For non-AND expressions, it's all-or-nothing
            if let Some(storage_expr) = convert_ast_to_storage_expr(expr) {
                (Some(storage_expr), None)
            } else {
                (None, Some(expr.clone()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, Schema, SchemaBuilder};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add("age", DataType::Integer)
            .add_nullable("email", DataType::Text)
            .build()
    }

    fn test_row() -> Row {
        Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::integer(30),
            Value::text("alice@example.com"),
        ])
    }

    // Helper to create AST identifier
    fn make_ident(name: &str) -> ast::Expression {
        ast::Expression::Identifier(ast::Identifier::new(
            dummy_token(name, TokenType::Identifier),
            name.to_string(),
        ))
    }

    // Helper to create AST integer literal
    fn make_int(value: i64) -> ast::Expression {
        ast::Expression::IntegerLiteral(ast::IntegerLiteral {
            token: dummy_token(&value.to_string(), TokenType::Integer),
            value,
        })
    }

    // Helper to create AST string literal
    fn make_str(value: &str) -> ast::Expression {
        ast::Expression::StringLiteral(ast::StringLiteral {
            token: dummy_token(value, TokenType::String),
            value: value.to_string(),
            type_hint: None,
        })
    }

    // Helper to create AST infix expression
    fn make_infix(left: ast::Expression, op: &str, right: ast::Expression) -> ast::InfixExpression {
        ast::InfixExpression::new(
            dummy_token(op, TokenType::Operator),
            Box::new(left),
            op.to_string(),
            Box::new(right),
        )
    }

    #[test]
    fn test_convert_simple_equality() {
        let ast_expr = ast::Expression::Infix(make_infix(make_ident("id"), "=", make_int(1)));

        let storage_expr = convert_ast_to_storage_expr(&ast_expr);
        assert!(storage_expr.is_some());

        let mut expr = storage_expr.unwrap();
        let schema = test_schema();
        expr.prepare_for_schema(&schema);

        let row = test_row();
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_convert_and_expression() {
        // id = 1 AND age > 20
        let left_cond = ast::Expression::Infix(make_infix(make_ident("id"), "=", make_int(1)));
        let right_cond = ast::Expression::Infix(make_infix(make_ident("age"), ">", make_int(20)));
        let ast_expr = ast::Expression::Infix(make_infix(left_cond, "AND", right_cond));

        let storage_expr = convert_ast_to_storage_expr(&ast_expr);
        assert!(storage_expr.is_some());

        let mut expr = storage_expr.unwrap();
        let schema = test_schema();
        expr.prepare_for_schema(&schema);

        let row = test_row(); // id=1, age=30
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_convert_value_on_left() {
        // 0 < id (should become id > 0)
        let ast_expr = ast::Expression::Infix(make_infix(make_int(0), "<", make_ident("id")));

        let storage_expr = convert_ast_to_storage_expr(&ast_expr);
        assert!(storage_expr.is_some());

        let mut expr = storage_expr.unwrap();
        let schema = test_schema();
        expr.prepare_for_schema(&schema);

        let row = test_row(); // id=1
        assert!(expr.evaluate(&row).unwrap()); // 0 < 1
    }

    #[test]
    fn test_cannot_convert_function_call() {
        // Function calls can't be pushed down
        let ast_expr = ast::Expression::FunctionCall(ast::FunctionCall {
            token: dummy_token("UPPER", TokenType::Identifier),
            function: "UPPER".to_string(),
            arguments: vec![make_ident("name")],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        });

        let storage_expr = convert_ast_to_storage_expr(&ast_expr);
        assert!(storage_expr.is_none());
    }

    #[test]
    fn test_convert_string_comparison() {
        // name = 'Alice'
        let ast_expr =
            ast::Expression::Infix(make_infix(make_ident("name"), "=", make_str("Alice")));

        let storage_expr = convert_ast_to_storage_expr(&ast_expr);
        assert!(storage_expr.is_some());

        let mut expr = storage_expr.unwrap();
        let schema = test_schema();
        expr.prepare_for_schema(&schema);

        let row = test_row();
        assert!(expr.evaluate(&row).unwrap());
    }
}
