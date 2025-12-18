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

//! Shared utility functions for the executor module.
//!
//! This module provides common utilities used across the executor:
//! - Token creation for internal AST construction
//! - Value-to-Expression conversion
//! - Row combination for JOIN operations
//! - Value hashing and comparison
//! - Column index map building

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::core::{DataType, Operator, Row, Value};
use crate::parser::ast::{
    BetweenExpression, BooleanLiteral, Expression, FloatLiteral, FunctionCall, Identifier,
    InExpression, InfixExpression, InfixOperator, IntegerLiteral, LikeExpression, ListExpression,
    NullLiteral, PrefixExpression, QualifiedIdentifier, StringLiteral,
};
use crate::parser::token::{Position, Token, TokenType};

// ============================================================================
// Token Creation Utilities
// ============================================================================

/// Helper to create a dummy token for internal AST construction.
///
/// This is useful when programmatically building AST nodes that require tokens.
#[inline]
pub fn dummy_token(literal: &str, token_type: TokenType) -> Token {
    Token::new(token_type, literal, Position::new(0, 1, 1))
}

// ============================================================================
// Value-to-Expression Conversion
// ============================================================================

/// Convert a Value to an Expression for use in subquery result replacement
/// and other internal AST manipulation.
pub fn value_to_expression(v: &Value) -> Expression {
    match v {
        Value::Integer(i) => Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token(&i.to_string(), TokenType::Integer),
            value: *i,
        }),
        Value::Float(f) => Expression::FloatLiteral(FloatLiteral {
            token: dummy_token(&f.to_string(), TokenType::Float),
            value: *f,
        }),
        Value::Text(s) => Expression::StringLiteral(StringLiteral {
            token: dummy_token(&format!("'{}'", s), TokenType::String),
            value: s.to_string(),
            type_hint: None,
        }),
        Value::Boolean(b) => Expression::BooleanLiteral(BooleanLiteral {
            token: dummy_token(if *b { "TRUE" } else { "FALSE" }, TokenType::Keyword),
            value: *b,
        }),
        Value::Null(_) => Expression::NullLiteral(NullLiteral {
            token: dummy_token("NULL", TokenType::Keyword),
        }),
        _ => Expression::StringLiteral(StringLiteral {
            token: dummy_token(&format!("'{}'", v), TokenType::String),
            value: v.to_string(),
            type_hint: None,
        }),
    }
}

// ============================================================================
// Column Index Utilities
// ============================================================================

/// Build a column name to index map for fast column lookups.
/// Column names are lowercased for case-insensitive matching.
#[inline]
pub fn build_column_index_map(columns: &[String]) -> FxHashMap<String, usize> {
    columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_lowercase(), i))
        .collect()
}

// ============================================================================
// Row Combination Utilities
// ============================================================================

/// Combine two rows into one for join output.
#[inline]
pub fn combine_rows(left: &Row, right: &Row, left_count: usize, right_count: usize) -> Vec<Value> {
    let mut combined = Vec::with_capacity(left_count + right_count);
    combined.extend_from_slice(left.as_slice());
    combined.extend_from_slice(right.as_slice());
    combined
}

/// Combine a row with NULLs for the other side (used in OUTER JOINs).
#[inline]
pub fn combine_rows_with_nulls(
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

// ============================================================================
// Hashing Utilities
// ============================================================================

/// Hash multiple key columns into a single hash value.
/// Used heavily in hash joins - called on every row during build and probe phases.
/// Uses FxHasher which is optimized for trusted keys in embedded database context.
#[inline]
pub fn hash_composite_key(row: &Row, key_indices: &[usize]) -> u64 {
    let mut hasher = FxHasher::default();

    for &idx in key_indices {
        if let Some(value) = row.get(idx) {
            hash_value_into(value, &mut hasher);
        } else {
            // NULL marker
            0xDEADBEEFu64.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Hash a single value into an existing hasher.
#[inline]
pub fn hash_value_into<H: Hasher>(value: &Value, hasher: &mut H) {
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

// ============================================================================
// Value Comparison Utilities
// ============================================================================

/// Compare two Values for equality.
#[inline]
pub fn values_equal(a: &Value, b: &Value) -> bool {
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

/// Compare two Values for ordering.
pub fn compare_values(a: &Value, b: &Value) -> Ordering {
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

/// Verify that all composite key columns match (handles hash collisions).
#[inline]
pub fn verify_composite_key_equality(
    row1: &Row,
    row2: &Row,
    indices1: &[usize],
    indices2: &[usize],
) -> bool {
    debug_assert_eq!(indices1.len(), indices2.len());

    for (&idx1, &idx2) in indices1.iter().zip(indices2.iter()) {
        match (row1.get(idx1), row2.get(idx2)) {
            (Some(v1), Some(v2)) => {
                if !values_equal(v1, v2) {
                    return false;
                }
            }
            (None, None) => {
                // Both NULL - considered not equal in SQL join semantics
                return false;
            }
            _ => {
                // One NULL, one not - not equal
                return false;
            }
        }
    }
    true
}

// ============================================================================
// Row Utilities
// ============================================================================

/// Hash all values in a row into a single hash value.
/// Used for DISTINCT operations and set operations (UNION, INTERSECT, EXCEPT).
/// Uses FxHasher which is optimized for trusted keys in embedded database context.
#[inline]
pub fn hash_row(row: &Row) -> u64 {
    let mut hasher = FxHasher::default();
    for value in row.iter() {
        value.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compare two rows for equality.
/// Returns true if both rows have the same length and all values are equal.
#[inline]
pub fn rows_equal(a: &Row, b: &Row) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        match (a.get(i), b.get(i)) {
            (Some(va), Some(vb)) if va == vb => continue,
            (None, None) => continue,
            _ => return false,
        }
    }
    true
}

// ============================================================================
// Expression Extraction Utilities
// ============================================================================

/// Extract the column name from an Identifier or QualifiedIdentifier expression.
/// Returns the column name without table qualifier.
#[inline]
pub fn extract_column_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::Identifier(Identifier { value, .. }) => Some(value.clone()),
        Expression::QualifiedIdentifier(QualifiedIdentifier { name, .. }) => {
            Some(name.value.clone())
        }
        _ => None,
    }
}

/// Extract a literal value from an expression.
/// Converts AST literal expressions to runtime Values.
#[inline]
pub fn extract_literal_value(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(i) => Some(Value::Integer(i.value)),
        Expression::FloatLiteral(f) => Some(Value::Float(f.value)),
        Expression::StringLiteral(s) => Some(Value::Text(s.value.clone().into())),
        Expression::BooleanLiteral(b) => Some(Value::Boolean(b.value)),
        Expression::NullLiteral(_) => Some(Value::Null(DataType::Text)),
        _ => None,
    }
}

/// Flip a comparison operator for when column and value are swapped.
/// E.g., `5 > col` becomes `col < 5`.
#[inline]
pub fn flip_operator(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::Lte => Operator::Gte,
        Operator::Gt => Operator::Lt,
        Operator::Gte => Operator::Lte,
        other => other, // Eq, Ne are symmetric
    }
}

/// Convert AST InfixOperator to core Operator.
/// Returns None for operators that don't map to comparison operators.
#[inline]
pub fn infix_to_operator(op: InfixOperator) -> Option<Operator> {
    match op {
        InfixOperator::Equal => Some(Operator::Eq),
        InfixOperator::NotEqual => Some(Operator::Ne),
        InfixOperator::LessThan => Some(Operator::Lt),
        InfixOperator::LessEqual => Some(Operator::Lte),
        InfixOperator::GreaterThan => Some(Operator::Gt),
        InfixOperator::GreaterEqual => Some(Operator::Gte),
        _ => None,
    }
}

// ============================================================================
// Column Name Utilities
// ============================================================================

/// Extract the base (unqualified) column name from a potentially qualified column name.
/// For "table.column" returns "column", for "column" returns "column".
/// The result is always lowercase for case-insensitive comparisons.
#[inline]
pub fn extract_base_column_name(col_name: &str) -> String {
    if let Some(dot_idx) = col_name.rfind('.') {
        col_name[dot_idx + 1..].to_lowercase()
    } else {
        col_name.to_lowercase()
    }
}

// ============================================================================
// Expression Analysis Utilities
// ============================================================================

/// Check if an expression contains any Parameter nodes ($1, $2, etc.)
/// Parameterized queries cannot be semantically cached because the cache
/// stores results tied to specific parameter values, but the AST only
/// contains parameter indices, not values.
pub fn expression_has_parameters(expr: &Expression) -> bool {
    match expr {
        Expression::Parameter(_) => true,
        Expression::Prefix(prefix) => expression_has_parameters(&prefix.right),
        Expression::Infix(infix) => {
            expression_has_parameters(&infix.left) || expression_has_parameters(&infix.right)
        }
        Expression::In(in_expr) => {
            expression_has_parameters(&in_expr.left)
                || match in_expr.right.as_ref() {
                    Expression::List(list) => list.elements.iter().any(expression_has_parameters),
                    Expression::ExpressionList(list) => {
                        list.expressions.iter().any(expression_has_parameters)
                    }
                    other => expression_has_parameters(other),
                }
        }
        Expression::Between(between) => {
            expression_has_parameters(&between.expr)
                || expression_has_parameters(&between.lower)
                || expression_has_parameters(&between.upper)
        }
        Expression::Like(like) => {
            expression_has_parameters(&like.left) || expression_has_parameters(&like.pattern)
        }
        Expression::Case(case) => {
            case.value
                .as_ref()
                .is_some_and(|e| expression_has_parameters(e))
                || case.when_clauses.iter().any(|wc| {
                    expression_has_parameters(&wc.condition)
                        || expression_has_parameters(&wc.then_result)
                })
                || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| expression_has_parameters(e))
        }
        Expression::FunctionCall(func) => func.arguments.iter().any(expression_has_parameters),
        Expression::Aliased(aliased) => expression_has_parameters(&aliased.expression),
        Expression::Cast(cast) => expression_has_parameters(&cast.expr),
        _ => false,
    }
}

/// Check if two expressions are structurally equivalent.
/// Used for semantic matching and predicate comparison.
pub fn expressions_equivalent(a: &Expression, b: &Expression) -> bool {
    match (a, b) {
        (Expression::Identifier(ia), Expression::Identifier(ib)) => {
            ia.value_lower == ib.value_lower
        }
        (Expression::QualifiedIdentifier(qa), Expression::QualifiedIdentifier(qb)) => {
            qa.qualifier.value_lower == qb.qualifier.value_lower
                && qa.name.value_lower == qb.name.value_lower
        }
        (Expression::IntegerLiteral(la), Expression::IntegerLiteral(lb)) => la.value == lb.value,
        (Expression::FloatLiteral(la), Expression::FloatLiteral(lb)) => {
            (la.value - lb.value).abs() < f64::EPSILON
        }
        (Expression::StringLiteral(la), Expression::StringLiteral(lb)) => la.value == lb.value,
        (Expression::BooleanLiteral(la), Expression::BooleanLiteral(lb)) => la.value == lb.value,
        (Expression::NullLiteral(_), Expression::NullLiteral(_)) => true,
        (Expression::Infix(ia), Expression::Infix(ib)) => {
            ia.op_type == ib.op_type
                && expressions_equivalent(&ia.left, &ib.left)
                && expressions_equivalent(&ia.right, &ib.right)
        }
        (Expression::Prefix(pa), Expression::Prefix(pb)) => {
            pa.operator == pb.operator && expressions_equivalent(&pa.right, &pb.right)
        }
        (Expression::Between(ba), Expression::Between(bb)) => {
            expressions_equivalent(&ba.expr, &bb.expr)
                && expressions_equivalent(&ba.lower, &bb.lower)
                && expressions_equivalent(&ba.upper, &bb.upper)
        }
        _ => false,
    }
}

// ============================================================================
// Predicate Manipulation Utilities
// ============================================================================

/// Flatten AND predicates into a list of individual predicates.
/// E.g., `a AND b AND c` becomes `[a, b, c]`.
pub fn flatten_and_predicates(expr: &Expression) -> Vec<Expression> {
    match expr {
        Expression::Infix(infix) if infix.operator.to_uppercase() == "AND" => {
            let mut result = flatten_and_predicates(&infix.left);
            result.extend(flatten_and_predicates(&infix.right));
            result
        }
        _ => vec![expr.clone()],
    }
}

/// Combine predicates with AND operator.
/// Returns None if the input is empty.
pub fn combine_predicates_with_and(preds: Vec<Expression>) -> Option<Expression> {
    if preds.is_empty() {
        return None;
    }

    let mut result = preds.into_iter();
    let first = result.next().unwrap();

    Some(result.fold(first, |acc, pred| {
        Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Keyword, "AND", Position::default()),
            Box::new(acc),
            "AND".to_string(),
            Box::new(pred),
        ))
    }))
}

/// Extract all AND-ed conditions from an expression as references.
/// Similar to flatten_and_predicates but returns references.
pub fn extract_and_conditions(expr: &Expression) -> Vec<&Expression> {
    let mut conditions = Vec::new();

    fn collect<'a>(expr: &'a Expression, out: &mut Vec<&'a Expression>) {
        if let Expression::Infix(infix) = expr {
            if matches!(infix.op_type, InfixOperator::And) {
                collect(&infix.left, out);
                collect(&infix.right, out);
                return;
            }
        }
        out.push(expr);
    }

    collect(expr, &mut conditions);
    conditions
}

// ============================================================================
// Table Qualifier Utilities
// ============================================================================

/// Extract all table qualifiers (aliases) referenced in an expression.
/// Returns a set of lowercase table names/aliases.
pub fn collect_table_qualifiers(expr: &Expression) -> FxHashSet<String> {
    let mut qualifiers = FxHashSet::default();
    collect_table_qualifiers_impl(expr, &mut qualifiers);
    qualifiers
}

fn collect_table_qualifiers_impl(expr: &Expression, qualifiers: &mut FxHashSet<String>) {
    match expr {
        Expression::QualifiedIdentifier(qi) => {
            qualifiers.insert(qi.qualifier.value.to_lowercase());
        }
        Expression::Infix(infix) => {
            collect_table_qualifiers_impl(&infix.left, qualifiers);
            collect_table_qualifiers_impl(&infix.right, qualifiers);
        }
        Expression::Prefix(prefix) => {
            collect_table_qualifiers_impl(&prefix.right, qualifiers);
        }
        Expression::In(in_expr) => {
            collect_table_qualifiers_impl(&in_expr.left, qualifiers);
            if let Expression::List(list) = in_expr.right.as_ref() {
                for elem in &list.elements {
                    collect_table_qualifiers_impl(elem, qualifiers);
                }
            }
        }
        Expression::Between(between) => {
            collect_table_qualifiers_impl(&between.expr, qualifiers);
            collect_table_qualifiers_impl(&between.lower, qualifiers);
            collect_table_qualifiers_impl(&between.upper, qualifiers);
        }
        Expression::Like(like) => {
            collect_table_qualifiers_impl(&like.left, qualifiers);
            collect_table_qualifiers_impl(&like.pattern, qualifiers);
        }
        Expression::FunctionCall(func) => {
            for arg in &func.arguments {
                collect_table_qualifiers_impl(arg, qualifiers);
            }
        }
        Expression::Aliased(aliased) => {
            collect_table_qualifiers_impl(&aliased.expression, qualifiers);
        }
        Expression::Cast(cast) => {
            collect_table_qualifiers_impl(&cast.expr, qualifiers);
        }
        _ => {}
    }
}

/// Get table alias from a table expression.
/// Returns the alias if specified, otherwise the table name.
pub fn get_table_alias_from_expr(expr: &Expression) -> Option<String> {
    match expr {
        Expression::TableSource(ts) => Some(
            ts.alias
                .as_ref()
                .map(|a| a.value.clone())
                .unwrap_or_else(|| ts.name.value.clone()),
        ),
        Expression::SubquerySource(ss) => ss.alias.as_ref().map(|a| a.value.clone()),
        _ => None,
    }
}

/// Strip table qualifier from an expression, replacing qualified identifiers
/// with unqualified ones. Used when pushing filters to individual table scans.
pub fn strip_table_qualifier(expr: &Expression, table_alias: &str) -> Expression {
    let alias_lower = table_alias.to_lowercase();

    match expr {
        Expression::QualifiedIdentifier(qi) if qi.qualifier.value.to_lowercase() == alias_lower => {
            // Convert to simple identifier
            Expression::Identifier(Identifier::new(
                qi.name.token.clone(),
                qi.name.value.clone(),
            ))
        }
        Expression::Infix(infix) => Expression::Infix(InfixExpression::new(
            infix.token.clone(),
            Box::new(strip_table_qualifier(&infix.left, table_alias)),
            infix.operator.clone(),
            Box::new(strip_table_qualifier(&infix.right, table_alias)),
        )),
        Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression::new(
            prefix.token.clone(),
            prefix.operator.clone(),
            Box::new(strip_table_qualifier(&prefix.right, table_alias)),
        )),
        Expression::In(in_expr) => {
            let new_left = strip_table_qualifier(&in_expr.left, table_alias);
            let new_right = match in_expr.right.as_ref() {
                Expression::List(list) => Expression::List(ListExpression {
                    token: list.token.clone(),
                    elements: list
                        .elements
                        .iter()
                        .map(|e| strip_table_qualifier(e, table_alias))
                        .collect(),
                }),
                other => strip_table_qualifier(other, table_alias),
            };
            Expression::In(InExpression {
                token: in_expr.token.clone(),
                left: Box::new(new_left),
                right: Box::new(new_right),
                not: in_expr.not,
            })
        }
        Expression::Between(between) => Expression::Between(BetweenExpression {
            token: between.token.clone(),
            expr: Box::new(strip_table_qualifier(&between.expr, table_alias)),
            lower: Box::new(strip_table_qualifier(&between.lower, table_alias)),
            upper: Box::new(strip_table_qualifier(&between.upper, table_alias)),
            not: between.not,
        }),
        Expression::Like(like) => Expression::Like(LikeExpression {
            token: like.token.clone(),
            left: Box::new(strip_table_qualifier(&like.left, table_alias)),
            pattern: Box::new(strip_table_qualifier(&like.pattern, table_alias)),
            operator: like.operator.clone(),
            escape: like
                .escape
                .as_ref()
                .map(|e| Box::new(strip_table_qualifier(e, table_alias))),
        }),
        Expression::FunctionCall(func) => Expression::FunctionCall(FunctionCall {
            token: func.token.clone(),
            function: func.function.clone(),
            arguments: func
                .arguments
                .iter()
                .map(|a| strip_table_qualifier(a, table_alias))
                .collect(),
            is_distinct: func.is_distinct,
            order_by: func.order_by.clone(),
            filter: func
                .filter
                .as_ref()
                .map(|f| Box::new(strip_table_qualifier(f, table_alias))),
        }),
        // Return unchanged for other expression types
        other => other.clone(),
    }
}

/// Add table qualifier to an expression, converting simple identifiers
/// to qualified ones. Used when applying filters post-join that were
/// originally stripped for pushdown.
pub fn add_table_qualifier(expr: &Expression, table_alias: &str) -> Expression {
    match expr {
        Expression::Identifier(id) => {
            // Convert to qualified identifier
            Expression::QualifiedIdentifier(QualifiedIdentifier {
                token: Token::new(TokenType::Identifier, table_alias, Position::default()),
                qualifier: Box::new(Identifier::new(
                    Token::new(TokenType::Identifier, table_alias, Position::default()),
                    table_alias.to_string(),
                )),
                name: Box::new(id.clone()),
            })
        }
        Expression::Infix(infix) => Expression::Infix(InfixExpression::new(
            infix.token.clone(),
            Box::new(add_table_qualifier(&infix.left, table_alias)),
            infix.operator.clone(),
            Box::new(add_table_qualifier(&infix.right, table_alias)),
        )),
        Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression::new(
            prefix.token.clone(),
            prefix.operator.clone(),
            Box::new(add_table_qualifier(&prefix.right, table_alias)),
        )),
        Expression::In(in_expr) => {
            let new_left = add_table_qualifier(&in_expr.left, table_alias);
            let new_right = match in_expr.right.as_ref() {
                Expression::List(list) => Expression::List(ListExpression {
                    token: list.token.clone(),
                    elements: list
                        .elements
                        .iter()
                        .map(|e| add_table_qualifier(e, table_alias))
                        .collect(),
                }),
                other => add_table_qualifier(other, table_alias),
            };
            Expression::In(InExpression {
                token: in_expr.token.clone(),
                left: Box::new(new_left),
                right: Box::new(new_right),
                not: in_expr.not,
            })
        }
        Expression::Between(between) => Expression::Between(BetweenExpression {
            token: between.token.clone(),
            expr: Box::new(add_table_qualifier(&between.expr, table_alias)),
            lower: Box::new(add_table_qualifier(&between.lower, table_alias)),
            upper: Box::new(add_table_qualifier(&between.upper, table_alias)),
            not: between.not,
        }),
        Expression::Like(like) => Expression::Like(LikeExpression {
            token: like.token.clone(),
            left: Box::new(add_table_qualifier(&like.left, table_alias)),
            pattern: Box::new(add_table_qualifier(&like.pattern, table_alias)),
            operator: like.operator.clone(),
            escape: like
                .escape
                .as_ref()
                .map(|e| Box::new(add_table_qualifier(e, table_alias))),
        }),
        Expression::FunctionCall(func) => Expression::FunctionCall(FunctionCall {
            token: func.token.clone(),
            function: func.function.clone(),
            arguments: func
                .arguments
                .iter()
                .map(|a| add_table_qualifier(a, table_alias))
                .collect(),
            is_distinct: func.is_distinct,
            order_by: func.order_by.clone(),
            filter: func
                .filter
                .as_ref()
                .map(|f| Box::new(add_table_qualifier(f, table_alias))),
        }),
        // Return unchanged for other expression types (literals, qualified identifiers, etc.)
        other => other.clone(),
    }
}

// ============================================================================
// Aggregate Function Utilities
// ============================================================================

/// Check if a function name is an aggregate function.
/// Uses the function registry to determine this.
#[inline]
pub fn is_aggregate_function(name: &str) -> bool {
    crate::functions::registry::global_registry().is_aggregate(name)
}

/// Check if an expression contains an aggregate function.
/// Used for detecting nested aggregates and determining query structure.
pub fn expression_contains_aggregate(expr: &Expression) -> bool {
    match expr {
        Expression::FunctionCall(func) => {
            if is_aggregate_function(&func.function) {
                return true;
            }
            // Check arguments recursively
            func.arguments.iter().any(expression_contains_aggregate)
        }
        Expression::Aliased(aliased) => expression_contains_aggregate(&aliased.expression),
        Expression::Infix(infix) => {
            expression_contains_aggregate(&infix.left)
                || expression_contains_aggregate(&infix.right)
        }
        Expression::Prefix(prefix) => expression_contains_aggregate(&prefix.right),
        Expression::Cast(cast) => expression_contains_aggregate(&cast.expr),
        Expression::Case(case) => {
            for when_clause in &case.when_clauses {
                if expression_contains_aggregate(&when_clause.condition)
                    || expression_contains_aggregate(&when_clause.then_result)
                {
                    return true;
                }
            }
            if let Some(ref else_val) = case.else_value {
                if expression_contains_aggregate(else_val) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// ============================================================================
// Column Index Utilities
// ============================================================================

/// Extract column name from identifier expression with optional qualifier.
/// Returns (qualifier, column_name) where qualifier is Some for qualified identifiers.
/// Column names are returned in lowercase for case-insensitive matching.
#[inline]
pub fn extract_column_name_with_qualifier(expr: &Expression) -> Option<(Option<String>, String)> {
    match expr {
        Expression::Identifier(id) => Some((None, id.value_lower.clone())),
        Expression::QualifiedIdentifier(qid) => Some((
            Some(qid.qualifier.value_lower.clone()),
            qid.name.value_lower.clone(),
        )),
        _ => None,
    }
}

/// Find column index in column list, handling qualified names.
/// Supports exact match, qualified match (table.column), and suffix match.
pub fn find_column_index(col_info: &(Option<String>, String), columns: &[String]) -> Option<usize> {
    let (qualifier, col_name) = col_info;

    for (idx, column) in columns.iter().enumerate() {
        let col_lower = column.to_lowercase();

        // Try exact match first
        if col_lower == *col_name {
            return Some(idx);
        }

        // Try qualified match (table.column)
        if let Some(q) = qualifier {
            let qualified = format!("{}.{}", q, col_name);
            if col_lower == qualified {
                return Some(idx);
            }
        }

        // Try suffix match (column might be stored as table.column)
        if col_lower.ends_with(&format!(".{}", col_name)) {
            return Some(idx);
        }
    }

    None
}

// ============================================================================
// Row Sorting Utilities
// ============================================================================

/// Check if rows are sorted in ascending order on the specified key indices.
///
/// This is used to detect when merge join can be used efficiently.
/// Returns true if the rows are sorted (ascending) on all key columns.
/// For large inputs (>10000 rows), uses sampling for efficiency.
pub fn is_sorted_on_keys(rows: &[Row], key_indices: &[usize]) -> bool {
    if rows.len() <= 1 || key_indices.is_empty() {
        return true; // Trivially sorted
    }

    // For large inputs, sample check is sufficient
    let check_limit = if rows.len() > 10000 { 1000 } else { rows.len() };

    // Check if rows are sorted by comparing consecutive pairs
    for i in 1..check_limit.min(rows.len()) {
        let prev = &rows[i - 1];
        let curr = &rows[i];

        for &idx in key_indices {
            match (prev.get(idx), curr.get(idx)) {
                (Some(v1), Some(v2)) => {
                    let cmp = compare_values(v1, v2);
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
                        let cmp = compare_values(v1, v2);
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

// ============================================================================
// Type Conversion Utilities
// ============================================================================

/// Convert type string to DataType.
/// Handles common SQL type names like INTEGER, VARCHAR, BOOLEAN, etc.
pub fn string_to_datatype(type_str: &str) -> DataType {
    match type_str.to_uppercase().as_str() {
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => DataType::Integer,
        "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" | "NUMERIC" => DataType::Float,
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => DataType::Text,
        "BOOLEAN" | "BOOL" => DataType::Boolean,
        "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => DataType::Timestamp,
        "JSON" | "JSONB" => DataType::Json,
        _ => DataType::Text,
    }
}

// ============================================================================
// Expression Display Utilities
// ============================================================================

/// Convert expression to string representation.
/// Used for expression alias matching and display purposes.
pub fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::Identifier(id) => id.value.clone(),
        Expression::QualifiedIdentifier(qid) => {
            format!("{}.{}", qid.qualifier.value, qid.name.value)
        }
        Expression::IntegerLiteral(lit) => lit.value.to_string(),
        Expression::FloatLiteral(lit) => lit.value.to_string(),
        Expression::StringLiteral(lit) => format!("'{}'", lit.value),
        Expression::BooleanLiteral(lit) => lit.value.to_string(),
        Expression::FunctionCall(func) => {
            let args: Vec<String> = func.arguments.iter().map(expression_to_string).collect();
            format!("{}({})", func.function, args.join(", "))
        }
        Expression::Infix(infix) => {
            format!(
                "{} {} {}",
                expression_to_string(&infix.left),
                infix.operator,
                expression_to_string(&infix.right)
            )
        }
        _ => format!("{}", expr),
    }
}
