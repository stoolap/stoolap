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
use std::sync::LazyLock;

use crate::common::CompactArc;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::common::StringMap;

use crate::core::value::NULL_VALUE;
use crate::core::{DataType, Operator, Row, Schema, Value};
use crate::executor::operators::index_nested_loop::ColumnSource;
use crate::parser::ast::{
    BetweenExpression, BooleanLiteral, CaseExpression, CastExpression, DistinctExpression,
    Expression, ExpressionList, FloatLiteral, FunctionCall, Identifier, InExpression,
    InHashSetExpression, InfixExpression, InfixOperator, IntegerLiteral, LikeExpression,
    ListExpression, NullLiteral, PrefixExpression, QualifiedIdentifier, StringLiteral, WhenClause,
    WindowFrameBound,
};
use crate::parser::token::{Position, Token, TokenType};

// ============================================================================
// Token Creation Utilities
// ============================================================================

/// Static dummy token for internal AST construction - avoids allocation.
/// Token literal is not used during execution, only for display/errors.
static DUMMY_TOKEN: LazyLock<Token> =
    LazyLock::new(|| Token::new(TokenType::Identifier, String::new(), Position::default()));

/// Get a reference to a pre-allocated dummy token (zero allocation).
#[inline]
pub fn dummy_token_ref() -> &'static Token {
    &DUMMY_TOKEN
}

/// Clone the static dummy token (single String allocation for empty string).
#[inline]
pub fn dummy_token_clone() -> Token {
    DUMMY_TOKEN.clone()
}

/// Helper to create a dummy token for internal AST construction.
/// Use `dummy_token_clone()` when literal doesn't matter to avoid allocation.
#[inline]
pub fn dummy_token(literal: &str, token_type: TokenType) -> Token {
    Token::new(token_type, literal, Position::default())
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
            value: s.as_str().into(),
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
            value: v.to_string().into(),
            type_hint: None,
        }),
    }
}

/// Substitute outer references in an expression with their actual values.
///
/// This is used for correlated subqueries to enable predicate pushdown.
/// When we have a WHERE clause like `o.user_id = u.id` where `u.id` is an outer
/// reference, this function replaces `u.id` with its actual value (e.g., 42),
/// allowing the expression `o.user_id = 42` to be pushed down to storage
/// for index usage.
///
/// # Arguments
/// * `expr` - The expression to transform
/// * `outer_row` - Map of outer column names to their values
///
/// # Returns
/// A new expression with outer references replaced by literal values.
/// Uses copy-on-write semantics: only clones when substitution is actually needed.
pub fn substitute_outer_references(
    expr: &Expression,
    outer_row: &FxHashMap<CompactArc<str>, Value>,
) -> Expression {
    // Use the internal function that returns Option for copy-on-write semantics
    substitute_outer_references_inner(expr, outer_row, None).unwrap_or_else(|| expr.clone())
}

/// The table a correlated subquery scans. References into it are left
/// alone: an inner column may share its name with an outer one, and SQL
/// binds the nearer scope first.
pub struct InnerScope<'a> {
    /// The names the subquery's own FROM defines. With an alias only the
    /// alias names the table; the table's own name may then be an alias in
    /// the outer query.
    pub tables: &'a [&'a str],
    /// The scanned table's schema, where there is one table to have it.
    /// Without it a bare name is left alone: it can only mean the inner one
    pub schema: Option<&'a Schema>,
}

impl InnerScope<'_> {
    fn owns_qualifier(&self, qualifier: &str) -> bool {
        self.tables.contains(&qualifier)
    }

    fn owns_column(&self, column: &str) -> bool {
        match self.schema {
            Some(schema) => schema.column_index_map().contains_key(column),
            None => true,
        }
    }
}

/// Like [`substitute_outer_references`], but leaves references to the
/// subquery's own table untouched so the outer values it does substitute
/// can drive index and primary key lookups.
pub fn substitute_outer_references_in_scope(
    expr: &Expression,
    outer_row: &FxHashMap<CompactArc<str>, Value>,
    scope: &InnerScope<'_>,
) -> Expression {
    substitute_outer_references_inner(expr, outer_row, Some(scope)).unwrap_or_else(|| expr.clone())
}

/// Internal helper that returns None if no substitution was made (avoids cloning).
/// Returns Some(new_expr) only when a substitution occurred.
fn substitute_outer_references_inner(
    expr: &Expression,
    outer_row: &FxHashMap<CompactArc<str>, Value>,
    scope: Option<&InnerScope<'_>>,
) -> Option<Expression> {
    match expr {
        // Check if this is an outer reference
        Expression::QualifiedIdentifier(qid) => {
            if scope.is_some_and(|s| s.owns_qualifier(&qid.qualifier.value_lower)) {
                return None;
            }
            // Try qualified name: "alias.column"
            // Use .as_str() for lookups since map now uses CompactArc<str> keys
            let qualified_name = format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
            if let Some(value) = outer_row.get(qualified_name.as_str()) {
                return Some(value_to_expression(value));
            }
            // Try just the column name as fallback
            if let Some(value) = outer_row.get(qid.name.value_lower.as_str()) {
                return Some(value_to_expression(value));
            }
            None
        }

        // Check unqualified identifiers too
        Expression::Identifier(id) => {
            if scope.is_some_and(|s| s.owns_column(&id.value_lower)) {
                return None;
            }
            if let Some(value) = outer_row.get(id.value_lower.as_str()) {
                return Some(value_to_expression(value));
            }
            None
        }

        // Recursively handle infix expressions (AND, OR, comparisons)
        Expression::Infix(infix) => {
            let new_left = substitute_outer_references_inner(&infix.left, outer_row, scope);
            let new_right = substitute_outer_references_inner(&infix.right, outer_row, scope);

            // Only create new expression if something changed
            if new_left.is_some() || new_right.is_some() {
                Some(Expression::Infix(InfixExpression {
                    token: infix.token.clone(),
                    left: Box::new(new_left.unwrap_or_else(|| (*infix.left).clone())),
                    operator: infix.operator.clone(),
                    op_type: infix.op_type,
                    right: Box::new(new_right.unwrap_or_else(|| (*infix.right).clone())),
                }))
            } else {
                None
            }
        }

        // Recursively handle prefix expressions (NOT)
        Expression::Prefix(prefix) => {
            substitute_outer_references_inner(&prefix.right, outer_row, scope).map(|new_right| {
                Expression::Prefix(PrefixExpression {
                    token: prefix.token.clone(),
                    operator: prefix.operator.clone(),
                    op_type: prefix.op_type,
                    right: Box::new(new_right),
                })
            })
        }

        // Handle IN expressions
        Expression::In(in_expr) => {
            let new_left = substitute_outer_references_inner(&in_expr.left, outer_row, scope);
            let new_right = match &*in_expr.right {
                Expression::List(list) => {
                    // Check if any element changed
                    let mut any_changed = false;
                    let new_elements: Vec<Option<Expression>> = list
                        .elements
                        .iter()
                        .map(|e| {
                            let result = substitute_outer_references_inner(e, outer_row, scope);
                            if result.is_some() {
                                any_changed = true;
                            }
                            result
                        })
                        .collect();

                    if any_changed {
                        Some(Expression::List(Box::new(ListExpression {
                            token: list.token.clone(),
                            elements: new_elements
                                .into_iter()
                                .zip(list.elements.iter())
                                .map(|(new, old)| new.unwrap_or_else(|| old.clone()))
                                .collect(),
                        })))
                    } else {
                        None
                    }
                }
                other => substitute_outer_references_inner(other, outer_row, scope),
            };

            if new_left.is_some() || new_right.is_some() {
                Some(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(new_left.unwrap_or_else(|| (*in_expr.left).clone())),
                    not: in_expr.not,
                    right: Box::new(new_right.unwrap_or_else(|| (*in_expr.right).clone())),
                }))
            } else {
                None
            }
        }

        // Handle BETWEEN expressions
        Expression::Between(between) => {
            let new_expr = substitute_outer_references_inner(&between.expr, outer_row, scope);
            let new_lower = substitute_outer_references_inner(&between.lower, outer_row, scope);
            let new_upper = substitute_outer_references_inner(&between.upper, outer_row, scope);

            if new_expr.is_some() || new_lower.is_some() || new_upper.is_some() {
                Some(Expression::Between(BetweenExpression {
                    token: between.token.clone(),
                    expr: Box::new(new_expr.unwrap_or_else(|| (*between.expr).clone())),
                    not: between.not,
                    lower: Box::new(new_lower.unwrap_or_else(|| (*between.lower).clone())),
                    upper: Box::new(new_upper.unwrap_or_else(|| (*between.upper).clone())),
                }))
            } else {
                None
            }
        }

        // Handle LIKE expressions
        Expression::Like(like) => {
            let new_left = substitute_outer_references_inner(&like.left, outer_row, scope);
            let new_pattern = substitute_outer_references_inner(&like.pattern, outer_row, scope);
            let new_escape = like
                .escape
                .as_ref()
                .and_then(|e| substitute_outer_references_inner(e, outer_row, scope));

            if new_left.is_some() || new_pattern.is_some() || new_escape.is_some() {
                Some(Expression::Like(LikeExpression {
                    token: like.token.clone(),
                    left: Box::new(new_left.unwrap_or_else(|| (*like.left).clone())),
                    operator: like.operator.clone(),
                    pattern: Box::new(new_pattern.unwrap_or_else(|| (*like.pattern).clone())),
                    escape: if new_escape.is_some() {
                        new_escape.map(Box::new)
                    } else {
                        like.escape.clone()
                    },
                }))
            } else {
                None
            }
        }

        // Handle function calls
        Expression::FunctionCall(func) => {
            // Check if any argument changed
            let mut any_changed = false;
            let new_args: Vec<Option<Expression>> = func
                .arguments
                .iter()
                .map(|arg| {
                    let result = substitute_outer_references_inner(arg, outer_row, scope);
                    if result.is_some() {
                        any_changed = true;
                    }
                    result
                })
                .collect();

            let new_filter = func
                .filter
                .as_ref()
                .and_then(|f| substitute_outer_references_inner(f, outer_row, scope));
            if new_filter.is_some() {
                any_changed = true;
            }

            if any_changed {
                Some(Expression::FunctionCall(Box::new(FunctionCall {
                    token: func.token.clone(),
                    function: func.function.clone(),
                    arguments: new_args
                        .into_iter()
                        .zip(func.arguments.iter())
                        .map(|(new, old)| new.unwrap_or_else(|| old.clone()))
                        .collect(),
                    is_distinct: func.is_distinct,
                    order_by: func.order_by.clone(),
                    filter: if new_filter.is_some() {
                        new_filter.map(Box::new)
                    } else {
                        func.filter.clone()
                    },
                })))
            } else {
                None
            }
        }

        Expression::InHashSet(in_set) => {
            substitute_outer_references_inner(&in_set.column, outer_row, scope).map(|column| {
                Expression::InHashSet(InHashSetExpression {
                    token: in_set.token.clone(),
                    column: Box::new(column),
                    values: in_set.values.clone(),
                    not: in_set.not,
                })
            })
        }
        Expression::Cast(cast) => substitute_outer_references_inner(&cast.expr, outer_row, scope)
            .map(|expr| {
                Expression::Cast(CastExpression {
                    token: cast.token.clone(),
                    expr: Box::new(expr),
                    type_name: cast.type_name.clone(),
                })
            }),
        Expression::Distinct(distinct) => {
            substitute_outer_references_inner(&distinct.expr, outer_row, scope).map(|expr| {
                Expression::Distinct(DistinctExpression {
                    token: distinct.token.clone(),
                    expr: Box::new(expr),
                })
            })
        }
        Expression::ExpressionList(list) => {
            let new_items: Vec<Option<Expression>> = list
                .expressions
                .iter()
                .map(|item| substitute_outer_references_inner(item, outer_row, scope))
                .collect();
            if new_items.iter().any(Option::is_some) {
                Some(Expression::ExpressionList(Box::new(ExpressionList {
                    token: list.token.clone(),
                    expressions: new_items
                        .into_iter()
                        .zip(list.expressions.iter())
                        .map(|(new, old)| new.unwrap_or_else(|| old.clone()))
                        .collect(),
                })))
            } else {
                None
            }
        }
        Expression::Case(case) => {
            let new_value = case
                .value
                .as_ref()
                .and_then(|value| substitute_outer_references_inner(value, outer_row, scope));
            let new_whens: Vec<(Option<Expression>, Option<Expression>)> = case
                .when_clauses
                .iter()
                .map(|when| {
                    (
                        substitute_outer_references_inner(&when.condition, outer_row, scope),
                        substitute_outer_references_inner(&when.then_result, outer_row, scope),
                    )
                })
                .collect();
            let new_else = case
                .else_value
                .as_ref()
                .and_then(|value| substitute_outer_references_inner(value, outer_row, scope));
            let changed = new_value.is_some()
                || new_else.is_some()
                || new_whens.iter().any(|(c, t)| c.is_some() || t.is_some());
            if !changed {
                return None;
            }
            Some(Expression::Case(Box::new(CaseExpression {
                token: case.token.clone(),
                value: match new_value {
                    Some(value) => Some(Box::new(value)),
                    None => case.value.clone(),
                },
                when_clauses: new_whens
                    .into_iter()
                    .zip(case.when_clauses.iter())
                    .map(|((condition, then_result), when)| WhenClause {
                        token: when.token.clone(),
                        condition: condition.unwrap_or_else(|| when.condition.clone()),
                        then_result: then_result.unwrap_or_else(|| when.then_result.clone()),
                    })
                    .collect(),
                else_value: match new_else {
                    Some(value) => Some(Box::new(value)),
                    None => case.else_value.clone(),
                },
            })))
        }

        // Literals and other expressions that don't need substitution
        _ => None,
    }
}

// ============================================================================
// Column Index Utilities
// ============================================================================

/// Build a column name to index map for fast column lookups.
/// Column names are lowercased for case-insensitive matching.
/// Also adds unqualified base names as fallbacks for qualified columns
/// (e.g., "t.val" also registers "val") when the base name is unambiguous.
pub fn build_column_index_map(columns: &[String]) -> StringMap<usize> {
    let mut map: StringMap<usize> = StringMap::with_capacity(columns.len());
    for (i, c) in columns.iter().enumerate() {
        map.insert(c.to_lowercase(), i);
    }
    // Add unqualified fallbacks for qualified column names (e.g., "t.val" → "val")
    // Only add when the base name is unambiguous (appears in exactly one table)
    // and doesn't conflict with an existing entry (e.g., an unqualified column)
    let mut base_count: StringMap<u8> = StringMap::default();
    for c in columns {
        let lower = c.to_lowercase();
        if let Some(dot_pos) = lower.rfind('.') {
            let base = &lower[dot_pos + 1..];
            let entry = base_count.entry(base.to_string()).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }
    for (i, c) in columns.iter().enumerate() {
        let lower = c.to_lowercase();
        if let Some(dot_pos) = lower.rfind('.') {
            let base = &lower[dot_pos + 1..];
            if base_count.get(base).copied().unwrap_or(0) == 1 && !map.contains_key(base) {
                map.insert(base.to_string(), i);
            }
        }
    }
    map
}

// ============================================================================
// Row Combination Utilities
// ============================================================================

/// Combine two rows into one for join output.
#[inline]
pub fn combine_rows(left: &Row, right: &Row, left_count: usize, right_count: usize) -> Vec<Value> {
    let mut combined = Vec::with_capacity(left_count + right_count);
    combined.extend(left.iter().cloned());
    combined.extend(right.iter().cloned());
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
        values.extend(row.iter().cloned());
        values.resize(row_count + null_count, NULL_VALUE);
    } else {
        values.resize(null_count, NULL_VALUE);
        values.extend(row.iter().cloned());
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
            // Join keys hash via Value's Hash impl so cross-type numeric
            // keys (Integer 5, Float 5.0) share a bucket; values_equal
            // decides the match exactly. hash_value_into is NOT used
            // here: it keys GROUP BY, whose semantics stay unchanged
            value.hash(&mut hasher);
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
        Value::Extension(data) => {
            10u8.hash(hasher);
            data.hash(hasher);
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
        // Exact IEEE equality plus the NaN-equals-NaN convention, matching
        // Value's PartialEq and the streaming join verifier (an epsilon
        // here dropped Infinity self-joins and diverged from the
        // streaming path on 0.0 vs -0.0)
        (Value::Float(x), Value::Float(y)) => x == y || (x.is_nan() && y.is_nan()),
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Boolean(x), Value::Boolean(y)) => x == y,
        (Value::Null(_), Value::Null(_)) => false, // NULL != NULL in SQL
        (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        (Value::Extension(x), Value::Extension(y)) => x == y,
        // Cross-type comparisons - exact numeric equality (epsilon-based
        // equality matched near-values, and `i as f64` rounds above 2^53)
        (Value::Integer(x), Value::Float(y)) | (Value::Float(y), Value::Integer(x)) => {
            crate::core::value::i64_eq_f64(*x, *y)
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
        (Value::Extension(x), Value::Extension(y)) => x.cmp(y),
        // Cross-type comparisons - exact numeric ordering, consistent
        // with Value::compare (merge join consumes ORDER BY output)
        (Value::Integer(x), Value::Float(y)) => {
            crate::core::value::cmp_i64_f64(*x, *y).unwrap_or(Ordering::Less)
        }
        (Value::Float(x), Value::Integer(y)) => crate::core::value::cmp_i64_f64(*y, *x)
            .map(Ordering::reverse)
            .unwrap_or(Ordering::Greater),
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
                    Value::Extension(_) => 6,
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
        Expression::Identifier(Identifier { value, .. }) => Some(value.to_string()),
        Expression::QualifiedIdentifier(QualifiedIdentifier { name, .. }) => {
            Some(name.value.to_string())
        }
        _ => None,
    }
}

/// Extract a literal value from an expression.
/// Converts AST literal expressions to runtime Values.
/// Note: double-quoted identifiers are NOT treated as literals here.
/// They may refer to actual column names, so pushdown should not assume
/// they are string constants. The VM/compiler handles them correctly via
/// column-resolution-first-then-string-fallback.
#[inline]
pub fn extract_literal_value(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(i) => Some(Value::Integer(i.value)),
        Expression::FloatLiteral(f) => Some(Value::Float(f.value)),
        Expression::StringLiteral(s) => Some(Value::Text(s.value.clone())),
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
            ba.not == bb.not
                && expressions_equivalent(&ba.expr, &bb.expr)
                && expressions_equivalent(&ba.lower, &bb.lower)
                && expressions_equivalent(&ba.upper, &bb.upper)
        }
        (Expression::In(ia), Expression::In(ib)) => {
            ia.not == ib.not
                && expressions_equivalent(&ia.left, &ib.left)
                && expressions_equivalent(&ia.right, &ib.right)
        }
        (Expression::ExpressionList(la), Expression::ExpressionList(lb)) => {
            la.expressions.len() == lb.expressions.len()
                && la
                    .expressions
                    .iter()
                    .zip(lb.expressions.iter())
                    .all(|(ae, be)| expressions_equivalent(ae, be))
        }
        (Expression::Like(la), Expression::Like(lb)) => {
            la.operator == lb.operator
                && expressions_equivalent(&la.left, &lb.left)
                && expressions_equivalent(&la.pattern, &lb.pattern)
                && match (&la.escape, &lb.escape) {
                    (None, None) => true,
                    (Some(ea), Some(eb)) => expressions_equivalent(ea, eb),
                    _ => false,
                }
        }
        (Expression::FunctionCall(fa), Expression::FunctionCall(fb)) => {
            function_calls_equivalent(fa, fb)
        }
        (Expression::Window(wa), Expression::Window(wb)) => {
            function_calls_equivalent(&wa.function, &wb.function)
                && wa.window_ref == wb.window_ref
                && wa.partition_by.len() == wb.partition_by.len()
                && wa
                    .partition_by
                    .iter()
                    .zip(wb.partition_by.iter())
                    .all(|(ae, be)| expressions_equivalent(ae, be))
                && wa.order_by.len() == wb.order_by.len()
                && wa.order_by.iter().zip(wb.order_by.iter()).all(|(oa, ob)| {
                    oa.ascending == ob.ascending
                        && oa.nulls_first == ob.nulls_first
                        && expressions_equivalent(&oa.expression, &ob.expression)
                })
                && match (&wa.frame, &wb.frame) {
                    (None, None) => true,
                    (Some(fa), Some(fb)) => {
                        fa.unit == fb.unit
                            && window_bounds_equivalent(&fa.start, &fb.start)
                            && match (&fa.end, &fb.end) {
                                (None, None) => true,
                                (Some(ea), Some(eb)) => window_bounds_equivalent(ea, eb),
                                _ => false,
                            }
                    }
                    _ => false,
                }
        }
        _ => false,
    }
}

/// Compare two FunctionCall structs for structural equivalence.
fn function_calls_equivalent(fa: &FunctionCall, fb: &FunctionCall) -> bool {
    fa.function.eq_ignore_ascii_case(&fb.function)
        && fa.is_distinct == fb.is_distinct
        && fa.arguments.len() == fb.arguments.len()
        && fa
            .arguments
            .iter()
            .zip(fb.arguments.iter())
            .all(|(ae, be)| expressions_equivalent(ae, be))
        && fa.order_by.len() == fb.order_by.len()
        && fa.order_by.iter().zip(fb.order_by.iter()).all(|(oa, ob)| {
            oa.ascending == ob.ascending
                && oa.nulls_first == ob.nulls_first
                && expressions_equivalent(&oa.expression, &ob.expression)
        })
        && match (&fa.filter, &fb.filter) {
            (None, None) => true,
            (Some(ea), Some(eb)) => expressions_equivalent(ea, eb),
            _ => false,
        }
}

/// Compare two WindowFrameBound values for structural equivalence.
fn window_bounds_equivalent(a: &WindowFrameBound, b: &WindowFrameBound) -> bool {
    match (a, b) {
        (WindowFrameBound::CurrentRow, WindowFrameBound::CurrentRow)
        | (WindowFrameBound::UnboundedPreceding, WindowFrameBound::UnboundedPreceding)
        | (WindowFrameBound::UnboundedFollowing, WindowFrameBound::UnboundedFollowing) => true,
        (WindowFrameBound::Preceding(ea), WindowFrameBound::Preceding(eb))
        | (WindowFrameBound::Following(ea), WindowFrameBound::Following(eb)) => {
            expressions_equivalent(ea, eb)
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
            qualifiers.insert(qi.qualifier.value_lower.to_string());
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
            match in_expr.right.as_ref() {
                Expression::ExpressionList(el) => {
                    for elem in &el.expressions {
                        collect_table_qualifiers_impl(elem, qualifiers);
                    }
                }
                Expression::List(list) => {
                    for elem in &list.elements {
                        collect_table_qualifiers_impl(elem, qualifiers);
                    }
                }
                other => {
                    collect_table_qualifiers_impl(other, qualifiers);
                }
            }
        }
        Expression::InHashSet(in_set) => {
            collect_table_qualifiers_impl(&in_set.column, qualifiers);
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
        Expression::Case(case) => {
            if let Some(ref val) = case.value {
                collect_table_qualifiers_impl(val, qualifiers);
            }
            for when in &case.when_clauses {
                collect_table_qualifiers_impl(&when.condition, qualifiers);
                collect_table_qualifiers_impl(&when.then_result, qualifiers);
            }
            if let Some(ref else_val) = case.else_value {
                collect_table_qualifiers_impl(else_val, qualifiers);
            }
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
                .map(|a| a.value.to_string())
                .unwrap_or_else(|| ts.name.value.to_string()),
        ),
        Expression::SubquerySource(ss) => ss.alias.as_ref().map(|a| a.value.to_string()),
        Expression::FunctionTableSource(fs) => Some(
            fs.alias
                .as_ref()
                .map(|a| a.value.to_string())
                .unwrap_or_else(|| fs.function.value.to_string()),
        ),
        Expression::ValuesSource(vs) => vs.alias.as_ref().map(|a| a.value.to_string()),
        Expression::CteReference(cr) => Some(
            cr.alias
                .as_ref()
                .map(|a| a.value.to_string())
                .unwrap_or_else(|| cr.name.value.to_string()),
        ),
        _ => None,
    }
}

/// Strip table qualifier from an expression, replacing qualified identifiers
/// with unqualified ones. Used when pushing filters to individual table scans.
pub fn strip_table_qualifier(expr: &Expression, table_alias: &str) -> Expression {
    let alias_lower = table_alias.to_lowercase();

    match expr {
        Expression::QualifiedIdentifier(qi) if qi.qualifier.value_lower.as_str() == alias_lower => {
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
                Expression::List(list) => Expression::List(Box::new(ListExpression {
                    token: list.token.clone(),
                    elements: list
                        .elements
                        .iter()
                        .map(|e| strip_table_qualifier(e, table_alias))
                        .collect(),
                })),
                other => strip_table_qualifier(other, table_alias),
            };
            Expression::In(InExpression {
                token: in_expr.token.clone(),
                left: Box::new(new_left),
                right: Box::new(new_right),
                not: in_expr.not,
            })
        }
        Expression::InHashSet(in_set) => Expression::InHashSet(InHashSetExpression {
            token: in_set.token.clone(),
            column: Box::new(strip_table_qualifier(&in_set.column, table_alias)),
            values: in_set.values.clone(),
            not: in_set.not,
        }),
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
        Expression::FunctionCall(func) => Expression::FunctionCall(Box::new(FunctionCall {
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
        })),
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
                Expression::List(list) => Expression::List(Box::new(ListExpression {
                    token: list.token.clone(),
                    elements: list
                        .elements
                        .iter()
                        .map(|e| add_table_qualifier(e, table_alias))
                        .collect(),
                })),
                other => add_table_qualifier(other, table_alias),
            };
            Expression::In(InExpression {
                token: in_expr.token.clone(),
                left: Box::new(new_left),
                right: Box::new(new_right),
                not: in_expr.not,
            })
        }
        Expression::InHashSet(in_set) => Expression::InHashSet(InHashSetExpression {
            token: in_set.token.clone(),
            column: Box::new(add_table_qualifier(&in_set.column, table_alias)),
            values: in_set.values.clone(),
            not: in_set.not,
        }),
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
        Expression::FunctionCall(func) => Expression::FunctionCall(Box::new(FunctionCall {
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
        })),
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
        Expression::Identifier(id) => Some((None, id.value_lower.to_string())),
        Expression::QualifiedIdentifier(qid) => Some((
            Some(qid.qualifier.value_lower.to_string()),
            qid.name.value_lower.to_string(),
        )),
        _ => None,
    }
}

/// Find column index in column list, handling qualified names.
/// Supports exact match and qualified match (table.column).
///
/// IMPORTANT: When a qualifier is provided (e.g., "t2" for "t2.id"),
/// we ONLY match columns that have that exact qualifier. This prevents
/// incorrectly matching "t1.id" when looking for "t2.id".
pub fn find_column_index(col_info: &(Option<String>, String), columns: &[String]) -> Option<usize> {
    let (qualifier, col_name) = col_info;

    // Pre-compute qualified name if qualifier exists (avoid format! in loop)
    let qualified = qualifier.as_ref().map(|q| format!("{}.{}", q, col_name));

    // First pass: try exact or qualified match
    for (idx, column) in columns.iter().enumerate() {
        let col_lower = column.to_lowercase();

        // Try exact match (unqualified column name)
        if col_lower == *col_name {
            return Some(idx);
        }

        // Try qualified match (table.column)
        if let Some(ref q) = qualified {
            if col_lower == *q {
                return Some(idx);
            }
        }
    }

    // Second pass: ONLY if no qualifier was provided, try suffix match
    // This allows matching "id" against "t1.id" when the column ref is just "id"
    if qualifier.is_none() {
        // Pre-compute suffix pattern once (avoid format! in loop)
        let suffix_pattern = format!(".{}", col_name);
        for (idx, column) in columns.iter().enumerate() {
            let col_lower = column.to_lowercase();
            if col_lower.ends_with(&suffix_pattern) {
                return Some(idx);
            }
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
///
/// # Strategy
///
/// - For small datasets (<= 50,000 rows): Check all consecutive pairs (O(n))
/// - For larger datasets: Use strategic sampling with boundary checks
///
/// The sampling approach checks:
/// 1. First and last 100 pairs (catch boundary issues)
/// 2. 100 random pairs in the middle (catch random discontinuities)
///
/// This catches most unsorted data while keeping the check O(1) for large datasets.
/// False positives (thinking unsorted data is sorted) are rare but possible;
/// however, merge join will still produce correct results - it will just be slower
/// than hash join for truly unsorted data.
pub fn is_sorted_on_keys(rows: &[Row], key_indices: &[usize]) -> bool {
    if rows.len() <= 1 || key_indices.is_empty() {
        return true; // Trivially sorted
    }

    // For small datasets, check all pairs - O(n) is acceptable
    if rows.len() <= 50_000 {
        return is_sorted_all_pairs(rows, key_indices);
    }

    // For larger datasets, use sampling to avoid O(n) overhead
    is_sorted_sampled(rows, key_indices)
}

/// Check all consecutive pairs - O(n) but complete coverage.
fn is_sorted_all_pairs(rows: &[Row], key_indices: &[usize]) -> bool {
    for i in 1..rows.len() {
        if !is_pair_sorted(&rows[i - 1], &rows[i], key_indices) {
            return false;
        }
    }
    true
}

/// Sampling-based sorted check - O(1) for large datasets.
/// Checks first/last boundaries plus random samples in the middle.
fn is_sorted_sampled(rows: &[Row], key_indices: &[usize]) -> bool {
    let n = rows.len();
    const BOUNDARY_CHECK: usize = 100;
    const MIDDLE_SAMPLES: usize = 100;

    // Check first BOUNDARY_CHECK pairs
    let first_check = BOUNDARY_CHECK.min(n - 1);
    for i in 1..=first_check {
        if !is_pair_sorted(&rows[i - 1], &rows[i], key_indices) {
            return false;
        }
    }

    // Check last BOUNDARY_CHECK pairs
    let last_start = n.saturating_sub(BOUNDARY_CHECK);
    for i in last_start.max(first_check + 1)..n {
        if !is_pair_sorted(&rows[i - 1], &rows[i], key_indices) {
            return false;
        }
    }

    // Sample middle section with deterministic stride
    // (Using stride instead of random to ensure reproducibility)
    let middle_start = first_check + 1;
    let middle_end = last_start.saturating_sub(1);
    if middle_end > middle_start {
        let middle_len = middle_end - middle_start;
        let stride = (middle_len / MIDDLE_SAMPLES).max(1);
        for i in (middle_start..middle_end).step_by(stride) {
            if !is_pair_sorted(&rows[i - 1], &rows[i], key_indices) {
                return false;
            }
        }
    }

    true
}

/// Check if two consecutive rows are in sorted order on the given keys.
#[inline]
fn is_pair_sorted(prev: &Row, curr: &Row, key_indices: &[usize]) -> bool {
    for &idx in key_indices {
        match (prev.get(idx), curr.get(idx)) {
            (Some(v1), Some(v2)) => {
                let cmp = compare_values(v1, v2);
                match cmp {
                    Ordering::Less => return true,     // prev < curr, sorted
                    Ordering::Greater => return false, // prev > curr, not sorted
                    Ordering::Equal => continue,       // Check next key
                }
            }
            (None, Some(_)) => return true, // NULL < value (NULL first)
            (Some(_), None) => return false, // value > NULL
            (None, None) => continue,       // Both NULL, check next key
        }
    }
    true // All keys equal - considered sorted
}

// ============================================================================
// Type Conversion Utilities
// ============================================================================

/// Convert type string to DataType.
/// Handles common SQL type names like INTEGER, VARCHAR, BOOLEAN, etc.
pub fn string_to_datatype(type_str: &str) -> DataType {
    let upper = type_str.to_uppercase();
    match upper.as_str() {
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => DataType::Integer,
        "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" | "NUMERIC" => DataType::Float,
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => DataType::Text,
        "BOOLEAN" | "BOOL" => DataType::Boolean,
        "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => DataType::Timestamp,
        "JSON" | "JSONB" => DataType::Json,
        _ => {
            // Handle VECTOR or VECTOR(N) pattern
            if upper.starts_with("VECTOR") {
                DataType::Vector
            } else {
                DataType::Text
            }
        }
    }
}

/// Parse vector dimension from a type string like "VECTOR(768)".
/// Returns 0 if no dimension is specified.
pub fn parse_vector_dimension(type_str: &str) -> u16 {
    let upper = type_str.to_uppercase();
    if let Some(inner) = upper
        .strip_prefix("VECTOR(")
        .and_then(|s| s.strip_suffix(')'))
    {
        inner.trim().parse::<u16>().unwrap_or(0)
    } else {
        0
    }
}

// ============================================================================
// Expression Display Utilities
// ============================================================================

/// Convert expression to string representation.
/// Used for expression alias matching and display purposes.
pub fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::Identifier(id) => id.value.to_string(),
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

// ============================================================================
// Join Key Extraction
// ============================================================================

/// Extract equality join keys and residual conditions from a join condition.
///
/// Returns (left_indices, right_indices, residual_conditions) where residual
/// contains non-equality conditions that must be applied after the hash join.
pub fn extract_join_keys_and_residual(
    condition: &Expression,
    left_columns: &[String],
    right_columns: &[String],
) -> (Vec<usize>, Vec<usize>, Vec<Expression>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let mut residual = Vec::new();

    extract_join_keys_recursive(
        condition,
        left_columns,
        right_columns,
        &mut left_indices,
        &mut right_indices,
        &mut residual,
    );

    (left_indices, right_indices, residual)
}

/// Recursively extract equality join keys from AND expressions.
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
            extract_join_keys_recursive(
                &infix.left,
                left_columns,
                right_columns,
                left_indices,
                right_indices,
                residual,
            );
            extract_join_keys_recursive(
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
                extract_column_name_with_qualifier(&infix.left),
                extract_column_name_with_qualifier(&infix.right),
            ) {
                // Case 1: left.col = right.col
                if let (Some(left_idx), Some(right_idx)) = (
                    find_column_index(&left_col, left_columns),
                    find_column_index(&right_col, right_columns),
                ) {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                    return;
                }

                // Case 2: right.col = left.col (swapped)
                if let (Some(left_idx), Some(right_idx)) = (
                    find_column_index(&right_col, left_columns),
                    find_column_index(&left_col, right_columns),
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

// ============================================================================
// Join Key Equivalence - Column Substitution
// ============================================================================

/// Recursively check if an expression contains a reference to a specific column.
/// This handles nested expressions including function calls, AND/OR, prefix, etc.
fn expression_contains_column(expr: &Expression, target_lower: &str) -> bool {
    match expr {
        // Direct column reference
        Expression::Identifier(ident) => {
            ident.value_lower.as_str() == target_lower
                || extract_base_column_name(&ident.value) == target_lower
        }
        Expression::QualifiedIdentifier(qi) => {
            qi.name.value_lower.as_str() == target_lower
                || extract_base_column_name(&qi.name.value) == target_lower
        }

        // Function calls - check all arguments (e.g., LOWER(col), COALESCE(col, 0))
        Expression::FunctionCall(fc) => fc
            .arguments
            .iter()
            .any(|arg| expression_contains_column(arg, target_lower)),

        // Infix expressions - check both sides (e.g., col + 1, col = value)
        Expression::Infix(infix) => {
            expression_contains_column(&infix.left, target_lower)
                || expression_contains_column(&infix.right, target_lower)
        }

        // Prefix expressions - check inner (e.g., NOT col, -col)
        Expression::Prefix(prefix) => expression_contains_column(&prefix.right, target_lower),

        // IN expression - check the left side
        Expression::In(in_expr) => expression_contains_column(&in_expr.left, target_lower),

        // BETWEEN expression - check the main expression
        Expression::Between(between) => expression_contains_column(&between.expr, target_lower),

        // LIKE expression - check the left side
        Expression::Like(like) => expression_contains_column(&like.left, target_lower),

        // CASE expression - check condition and all branches
        Expression::Case(case) => {
            let in_value = case
                .value
                .as_ref()
                .map(|e| expression_contains_column(e, target_lower))
                .unwrap_or(false);
            let in_branches = case.when_clauses.iter().any(|clause| {
                expression_contains_column(&clause.condition, target_lower)
                    || expression_contains_column(&clause.then_result, target_lower)
            });
            let in_else = case
                .else_value
                .as_ref()
                .map(|e| expression_contains_column(e, target_lower))
                .unwrap_or(false);
            in_value || in_branches || in_else
        }

        // Cast expression - check inner expression
        Expression::Cast(cast) => expression_contains_column(&cast.expr, target_lower),

        // Subqueries - don't recurse into subqueries for this optimization
        Expression::ScalarSubquery(_) | Expression::SubquerySource(_) => false,

        // Literals and other terminals - no column reference
        _ => false,
    }
}

/// Check if a filter expression references a specific column (the join key).
/// Returns true if the filter's main column matches the target column name.
/// Handles IN, comparison, BETWEEN, LIKE, function calls, and nested expressions.
///
/// This is used for join key equivalence optimization: when a filter on the
/// inner table's join key can be pushed to the outer table.
pub fn filter_references_column(expr: &Expression, target_col: &str) -> bool {
    let target_lower = target_col.to_lowercase();

    match expr {
        Expression::In(in_expr) => {
            // Check if IN expression references the target column (direct or nested)
            expression_contains_column(&in_expr.left, &target_lower)
        }
        Expression::Infix(infix) => {
            // Handle AND/OR by checking both sides recursively
            if infix.operator == "AND" || infix.operator == "OR" {
                return filter_references_column(&infix.left, target_col)
                    || filter_references_column(&infix.right, target_col);
            }

            // Check comparison expressions: col = value, LOWER(col) = 'x', etc.
            expression_contains_column(&infix.left, &target_lower)
                || expression_contains_column(&infix.right, &target_lower)
        }
        Expression::Between(between) => {
            // Check if BETWEEN expression references the target column
            expression_contains_column(&between.expr, &target_lower)
        }
        Expression::Like(like) => {
            // Check if LIKE expression references the target column
            expression_contains_column(&like.left, &target_lower)
        }
        Expression::Prefix(prefix) => {
            // Handle NOT expression by checking inner (e.g., NOT col IS NULL)
            filter_references_column(&prefix.right, target_col)
        }
        Expression::FunctionCall(fc) => {
            // Function call at top level (rare, but handle it)
            fc.arguments
                .iter()
                .any(|arg| expression_contains_column(arg, &target_lower))
        }
        _ => false,
    }
}

/// Substitute a column reference in a filter expression with a new column name.
/// This is used for join key equivalence: when filter `o.user_id IN (1,2,3)`
/// can be transformed to `u.id IN (1,2,3)` based on join condition `u.id = o.user_id`.
///
/// Only handles simple cases where the column is directly referenced.
/// Returns None if substitution is not possible.
pub fn substitute_filter_column(
    expr: &Expression,
    from_col: &str,
    to_col: &str,
) -> Option<Expression> {
    let from_lower = from_col.to_lowercase();
    let from_base = extract_base_column_name(from_col);

    match expr {
        Expression::In(in_expr) => {
            // Substitute column in IN expression
            if let Some(col_name) = extract_column_name(&in_expr.left) {
                let col_lower = col_name.to_lowercase();
                let col_base = extract_base_column_name(&col_name);

                if col_lower == from_lower || col_base == from_base {
                    // Create new identifier with the target column name
                    let new_left = create_column_identifier(to_col);
                    return Some(Expression::In(InExpression {
                        token: in_expr.token.clone(),
                        left: Box::new(new_left),
                        right: in_expr.right.clone(),
                        not: in_expr.not,
                    }));
                }
            }
        }
        Expression::Infix(infix) => {
            // Substitute column in comparison expression
            let left_col = extract_column_name(&infix.left);
            let right_col = extract_column_name(&infix.right);

            // Check if left side is the target column
            if let Some(col_name) = &left_col {
                let col_lower = col_name.to_lowercase();
                let col_base = extract_base_column_name(col_name);

                if col_lower == from_lower || col_base == from_base {
                    let new_left = create_column_identifier(to_col);
                    return Some(Expression::Infix(InfixExpression::new(
                        infix.token.clone(),
                        Box::new(new_left),
                        infix.operator.clone(),
                        infix.right.clone(),
                    )));
                }
            }

            // Check if right side is the target column (for value = col cases)
            if let Some(col_name) = &right_col {
                let col_lower = col_name.to_lowercase();
                let col_base = extract_base_column_name(col_name);

                if col_lower == from_lower || col_base == from_base {
                    let new_right = create_column_identifier(to_col);
                    return Some(Expression::Infix(InfixExpression::new(
                        infix.token.clone(),
                        infix.left.clone(),
                        infix.operator.clone(),
                        Box::new(new_right),
                    )));
                }
            }
        }
        Expression::Between(between) => {
            if let Some(col_name) = extract_column_name(&between.expr) {
                let col_lower = col_name.to_lowercase();
                let col_base = extract_base_column_name(&col_name);

                if col_lower == from_lower || col_base == from_base {
                    let new_expr = create_column_identifier(to_col);
                    return Some(Expression::Between(BetweenExpression {
                        token: between.token.clone(),
                        expr: Box::new(new_expr),
                        lower: between.lower.clone(),
                        upper: between.upper.clone(),
                        not: between.not,
                    }));
                }
            }
        }
        Expression::Like(like) => {
            if let Some(col_name) = extract_column_name(&like.left) {
                let col_lower = col_name.to_lowercase();
                let col_base = extract_base_column_name(&col_name);

                if col_lower == from_lower || col_base == from_base {
                    let new_left = create_column_identifier(to_col);
                    return Some(Expression::Like(LikeExpression {
                        token: like.token.clone(),
                        left: Box::new(new_left),
                        pattern: like.pattern.clone(),
                        operator: like.operator.clone(),
                        escape: like.escape.clone(),
                    }));
                }
            }
        }
        _ => {}
    }
    None
}

/// Create a column identifier expression from a column name.
/// Handles qualified names (table.column) and unqualified names (column).
fn create_column_identifier(col_name: &str) -> Expression {
    if let Some(dot_idx) = col_name.find('.') {
        let qualifier = &col_name[..dot_idx];
        let name = &col_name[dot_idx + 1..];
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token(col_name, TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token(qualifier, TokenType::Identifier),
                qualifier.to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token(name, TokenType::Identifier),
                name.to_string(),
            )),
        })
    } else {
        Expression::Identifier(Identifier::new(
            dummy_token(col_name, TokenType::Identifier),
            col_name.to_string(),
        ))
    }
}

// ============================================================================
// Join Projection Utilities
// ============================================================================

/// Result of computing join projection indices.
/// Contains the column sources in SELECT order to satisfy the SELECT expressions.
pub struct JoinProjectionIndices {
    /// Column sources in SELECT order (preserves original column ordering)
    pub columns: Vec<ColumnSource>,
    /// Output column names for the projected result
    pub output_columns: Vec<String>,
}

/// Compute projection indices for a join operator.
///
/// Analyzes SELECT expressions and determines which columns from the outer and inner
/// sides are needed. Returns columns in SELECT order (not outer-first/inner-second).
///
/// # Arguments
/// * `select_exprs` - The SELECT expressions to analyze
/// * `outer_columns` - Column names from the outer (left) side of the join
/// * `inner_columns` - Column names from the inner (right) side of the join
///
/// # Returns
/// Some(JoinProjectionIndices) if all expressions are simple column references,
/// None otherwise.
pub fn compute_join_projection(
    select_exprs: &[Expression],
    outer_columns: &[String],
    inner_columns: &[String],
) -> Option<JoinProjectionIndices> {
    let outer_col_count = outer_columns.len();
    // Resolved by scanning the two short column lists instead of building a
    // map with an owned copy of every name on every execution.
    let columns_all = || outer_columns.iter().chain(inner_columns.iter());
    // Duplicate full names resolve to the later column, as the index map
    // built from these lists always did.
    let resolve = |name: &str| -> Option<usize> {
        if let Some((i, _)) = columns_all()
            .enumerate()
            .filter(|(_, c)| eq_folded(c, name))
            .last()
        {
            return Some(i);
        }
        // "t.val" answers to "val" when no other column has that base name
        let mut found = None;
        for (i, c) in columns_all().enumerate() {
            if let Some(dot) = c.rfind('.') {
                if eq_folded(&c[dot + 1..], name) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(i);
                }
            }
        }
        found
    };
    let resolve_qualified = |qualifier: &str, name: &str| -> Option<usize> {
        columns_all()
            .enumerate()
            .filter(|(_, c)| eq_folded_qualified(c, qualifier, name))
            .last()
            .map(|(i, _)| i)
            .or_else(|| resolve(name))
    };
    let source = |idx: usize| {
        if idx < outer_col_count {
            ColumnSource::Outer(idx)
        } else {
            ColumnSource::Inner(idx - outer_col_count)
        }
    };

    let mut columns = Vec::with_capacity(select_exprs.len());
    let mut output_columns = Vec::with_capacity(select_exprs.len());
    for expr in select_exprs {
        let (idx, name) = match expr {
            Expression::Identifier(id) => (resolve(&id.value_lower)?, id.value.to_string()),
            Expression::QualifiedIdentifier(qid) => (
                resolve_qualified(&qid.qualifier.value_lower, &qid.name.value_lower)?,
                qid.name.value.to_string(),
            ),
            Expression::Aliased(aliased) => {
                let alias = aliased.alias.value.to_string();
                match &*aliased.expression {
                    Expression::Identifier(id) => (resolve(&id.value_lower)?, alias),
                    Expression::QualifiedIdentifier(qid) => (
                        resolve_qualified(&qid.qualifier.value_lower, &qid.name.value_lower)?,
                        alias,
                    ),
                    _ => return None, // Complex expression - cannot push down
                }
            }
            _ => return None,
        };
        columns.push(source(idx));
        output_columns.push(name);
    }

    Some(JoinProjectionIndices {
        columns,
        output_columns,
    })
}

/// Case-insensitive comparison of a column name against an already
/// lowercased name, folding the Unicode way without allocating.
#[inline]
fn eq_folded(column: &str, lower: &str) -> bool {
    if column.is_ascii() && lower.is_ascii() {
        column.eq_ignore_ascii_case(lower)
    } else {
        column
            .chars()
            .flat_map(char::to_lowercase)
            .eq(lower.chars())
    }
}

/// The same for `qualifier.name`, compared as one string so that a dot
/// inside a quoted qualifier or name is not taken for the separator.
#[inline]
fn eq_folded_qualified(column: &str, qualifier: &str, name: &str) -> bool {
    if column.is_ascii() && qualifier.is_ascii() && name.is_ascii() {
        let q = qualifier.len();
        column.len() == q + 1 + name.len()
            && column.as_bytes()[q] == b'.'
            && column[..q].eq_ignore_ascii_case(qualifier)
            && column[q + 1..].eq_ignore_ascii_case(name)
    } else {
        let expected = qualifier
            .chars()
            .chain(std::iter::once('.'))
            .chain(name.chars());
        column.chars().flat_map(char::to_lowercase).eq(expected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(name: &str) -> Expression {
        Expression::Identifier(Identifier {
            token: dummy_token_clone(),
            value: name.into(),
            value_lower: name.to_lowercase().into(),
        })
    }

    fn qualified(qualifier: &str, name: &str) -> Expression {
        let (Expression::Identifier(q), Expression::Identifier(n)) =
            (ident(qualifier), ident(name))
        else {
            unreachable!()
        };
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token_clone(),
            qualifier: Box::new(q),
            name: Box::new(n),
        })
    }

    /// Column names fold case the Unicode way, and when two columns share
    /// a name the later one wins, as the column index map always did.
    #[test]
    fn test_join_projection_folds_unicode_and_prefers_the_later_duplicate() {
        let outer = vec!["u.ÄNAME".to_string(), "u.x".to_string()];
        let inner = vec!["o.x".to_string(), "x".to_string()];

        let proj = compute_join_projection(&[qualified("u", "äname")], &outer, &inner)
            .expect("Unicode-folded name must resolve");
        assert!(matches!(proj.columns[0], ColumnSource::Outer(0)));

        // bare `x`: exact matches only the unqualified inner column
        let proj = compute_join_projection(&[ident("x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));

        // `s.x` is not a full name; the bare fallback finds the exact `x`
        let proj = compute_join_projection(&[qualified("s", "x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));
        // without an exact `x`, two columns ending in `.x` are ambiguous
        let two = vec!["u.x".to_string(), "o.x".to_string()];
        assert!(compute_join_projection(&[ident("x")], &two, &[]).is_none());

        // duplicate full names: the later one wins
        let dup = vec!["v".to_string(), "v".to_string()];
        let proj = compute_join_projection(&[ident("v")], &dup, &[]).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Outer(1)));
    }

    /// A quoted alias may contain a dot; `"a.b".x` must match the column
    /// `a.b.x` as a whole and not fall back to a bare `x` elsewhere.
    #[test]
    fn test_join_projection_keeps_dots_inside_a_qualifier() {
        let outer = vec!["x".to_string()];
        let inner = vec!["a.b.id".to_string(), "a.b.x".to_string()];
        let proj = compute_join_projection(&[qualified("a.b", "x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));

        let unicode = vec!["Ä.b.x".to_string()];
        let proj = compute_join_projection(&[qualified("ä.b", "x")], &unicode, &[]).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Outer(0)));
    }

    /// Outer values reach the subquery's WHERE, but names that belong to
    /// the inner table are left alone even when the outer row has a column
    /// of the same name.
    #[test]
    fn test_substitute_outer_references_respects_inner_scope() {
        use crate::core::SchemaBuilder;

        let schema = SchemaBuilder::new("parent")
            .add_primary_key("id", DataType::Integer)
            .add("Name", DataType::Text)
            .build();
        let scope = InnerScope {
            tables: &["p"],
            schema: Some(&schema),
        };
        let mut outer: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
        for (key, value) in [
            ("c.id", 3),
            ("id", 3),
            ("c.pid", 7),
            ("pid", 7),
            ("c.name", 9),
            ("name", 9),
        ] {
            outer.insert(CompactArc::from(key), Value::Integer(value));
        }

        let is_literal = |expr: &Expression, expected: i64| matches!(expr, Expression::IntegerLiteral(l) if l.value == expected);

        // inner references, qualified by the alias or bare
        for expr in [qualified("p", "id"), ident("id"), ident("name")] {
            let out = substitute_outer_references_in_scope(&expr, &outer, &scope);
            assert!(
                matches!(
                    out,
                    Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                ),
                "{expr} must not be substituted"
            );
        }

        // outer references, qualified or bare, still become literals
        assert!(is_literal(
            &substitute_outer_references_in_scope(&qualified("c", "pid"), &outer, &scope),
            7
        ));
        assert!(is_literal(
            &substitute_outer_references_in_scope(&qualified("c", "id"), &outer, &scope),
            3
        ));
        assert!(is_literal(
            &substitute_outer_references_in_scope(&ident("pid"), &outer, &scope),
            7
        ));

        // with an alias, the table name is not inner: it may be an outer alias
        outer.insert(CompactArc::from("parent.id"), Value::Integer(5));
        outer.insert(CompactArc::from("parent.pid"), Value::Integer(7));
        assert!(is_literal(
            &substitute_outer_references_in_scope(&qualified("parent", "id"), &outer, &scope),
            5
        ));
        assert!(is_literal(
            &substitute_outer_references_in_scope(&qualified("parent", "pid"), &outer, &scope),
            7
        ));

        // without an alias, the table name is the inner qualifier
        let bare = InnerScope {
            tables: &["parent"],
            schema: Some(&schema),
        };
        assert!(matches!(
            substitute_outer_references_in_scope(&qualified("parent", "id"), &outer, &bare),
            Expression::QualifiedIdentifier(_)
        ));
    }

    // ============================================================================
    // Token Creation Tests
    // ============================================================================

    #[test]
    fn test_dummy_token_integer() {
        let token = dummy_token("42", TokenType::Integer);
        assert_eq!(token.literal, "42");
        assert_eq!(token.token_type, TokenType::Integer);
    }

    #[test]
    fn test_dummy_token_string() {
        let token = dummy_token("hello", TokenType::String);
        assert_eq!(token.literal, "hello");
        assert_eq!(token.token_type, TokenType::String);
    }

    // ============================================================================
    // Value to Expression Tests
    // ============================================================================

    #[test]
    fn test_value_to_expression_integer() {
        let expr = value_to_expression(&Value::Integer(42));
        match expr {
            Expression::IntegerLiteral(lit) => assert_eq!(lit.value, 42),
            _ => panic!("Expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_value_to_expression_float() {
        let expr = value_to_expression(&Value::Float(3.5));
        match expr {
            Expression::FloatLiteral(lit) => assert!((lit.value - 3.5).abs() < f64::EPSILON),
            _ => panic!("Expected FloatLiteral"),
        }
    }

    #[test]
    fn test_value_to_expression_text() {
        let expr = value_to_expression(&Value::Text("hello".into()));
        match expr {
            Expression::StringLiteral(lit) => assert_eq!(lit.value, "hello"),
            _ => panic!("Expected StringLiteral"),
        }
    }

    #[test]
    fn test_value_to_expression_boolean() {
        let expr_true = value_to_expression(&Value::Boolean(true));
        match expr_true {
            Expression::BooleanLiteral(lit) => assert!(lit.value),
            _ => panic!("Expected BooleanLiteral"),
        }

        let expr_false = value_to_expression(&Value::Boolean(false));
        match expr_false {
            Expression::BooleanLiteral(lit) => assert!(!lit.value),
            _ => panic!("Expected BooleanLiteral"),
        }
    }

    #[test]
    fn test_value_to_expression_null() {
        let expr = value_to_expression(&Value::Null(DataType::Integer));
        match expr {
            Expression::NullLiteral(_) => {}
            _ => panic!("Expected NullLiteral"),
        }
    }

    // ============================================================================
    // Column Index Map Tests
    // ============================================================================

    #[test]
    fn test_build_column_index_map() {
        let columns = vec!["ID".to_string(), "Name".to_string(), "Age".to_string()];
        let map = build_column_index_map(&columns);

        assert_eq!(map.get("id"), Some(&0));
        assert_eq!(map.get("name"), Some(&1));
        assert_eq!(map.get("age"), Some(&2));
        assert_eq!(map.get("unknown"), None);
    }

    #[test]
    fn test_build_column_index_map_empty() {
        let columns: Vec<String> = vec![];
        let map = build_column_index_map(&columns);
        assert!(map.is_empty());
    }

    // ============================================================================
    // Row Combination Tests
    // ============================================================================

    #[test]
    fn test_combine_rows() {
        let left = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let right = Row::from(vec![Value::Integer(2), Value::Text("b".into())]);

        let combined = combine_rows(&left, &right, 2, 2);
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], Value::Integer(1));
        assert_eq!(combined[1], Value::Text("a".into()));
        assert_eq!(combined[2], Value::Integer(2));
        assert_eq!(combined[3], Value::Text("b".into()));
    }

    #[test]
    fn test_combine_rows_with_nulls_left() {
        let row = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let combined = combine_rows_with_nulls(&row, 2, 2, true);

        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], Value::Integer(1));
        assert_eq!(combined[1], Value::Text("a".into()));
        assert!(combined[2].is_null());
        assert!(combined[3].is_null());
    }

    #[test]
    fn test_combine_rows_with_nulls_right() {
        let row = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let combined = combine_rows_with_nulls(&row, 2, 2, false);

        assert_eq!(combined.len(), 4);
        assert!(combined[0].is_null());
        assert!(combined[1].is_null());
        assert_eq!(combined[2], Value::Integer(1));
        assert_eq!(combined[3], Value::Text("a".into()));
    }

    // ============================================================================
    // Hashing Tests
    // ============================================================================

    fn verify_pair(a: f64, b: f64) -> bool {
        let r1 = Row::from_values(vec![Value::Float(a)]);
        let r2 = Row::from_values(vec![Value::Float(b)]);
        verify_composite_key_equality(&r1, &r2, &[0], &[0])
    }

    #[test]
    fn test_float_key_equality_ieee_semantics() {
        assert!(verify_pair(0.0, -0.0), "0.0 = -0.0 must join");
        assert!(
            verify_pair(f64::INFINITY, f64::INFINITY),
            "inf = inf must join"
        );
        assert!(
            verify_pair(f64::NAN, f64::NAN),
            "NaN convention: all NaNs join"
        );
        assert!(
            !verify_pair(0.5, 0.5 + f64::EPSILON),
            "epsilon-close is not equal"
        );
        assert!(!verify_pair(1.0, 1.5));
    }

    #[test]
    fn test_hash_composite_key_single() {
        let row = Row::from(vec![Value::Integer(42)]);
        let hash1 = hash_composite_key(&row, &[0]);
        let hash2 = hash_composite_key(&row, &[0]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_composite_key_multiple() {
        let row = Row::from(vec![
            Value::Integer(1),
            Value::Text("test".into()),
            Value::Integer(3),
        ]);
        let hash = hash_composite_key(&row, &[0, 1]);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_hash_composite_key_different_values() {
        let row1 = Row::from(vec![Value::Integer(1)]);
        let row2 = Row::from(vec![Value::Integer(2)]);

        let hash1 = hash_composite_key(&row1, &[0]);
        let hash2 = hash_composite_key(&row2, &[0]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_row() {
        let row1 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row3 = Row::from(vec![Value::Integer(2), Value::Text("a".into())]);

        assert_eq!(hash_row(&row1), hash_row(&row2));
        assert_ne!(hash_row(&row1), hash_row(&row3));
    }

    // ============================================================================
    // Value Comparison Tests
    // ============================================================================

    #[test]
    fn test_values_equal_integers() {
        assert!(values_equal(&Value::Integer(42), &Value::Integer(42)));
        assert!(!values_equal(&Value::Integer(42), &Value::Integer(43)));
    }

    #[test]
    fn test_values_equal_floats() {
        assert!(values_equal(&Value::Float(3.5), &Value::Float(3.5)));
        assert!(!values_equal(&Value::Float(3.5), &Value::Float(3.6)));
    }

    #[test]
    fn test_values_equal_text() {
        assert!(values_equal(
            &Value::Text("hello".into()),
            &Value::Text("hello".into())
        ));
        assert!(!values_equal(
            &Value::Text("hello".into()),
            &Value::Text("world".into())
        ));
    }

    #[test]
    fn test_values_equal_null() {
        // NULL != NULL in SQL
        assert!(!values_equal(
            &Value::Null(DataType::Integer),
            &Value::Null(DataType::Integer)
        ));
    }

    #[test]
    fn test_values_equal_cross_type_numeric() {
        assert!(values_equal(&Value::Integer(42), &Value::Float(42.0)));
        assert!(values_equal(&Value::Float(42.0), &Value::Integer(42)));
    }

    #[test]
    fn test_compare_values_integers() {
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Integer(2)),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&Value::Integer(2), &Value::Integer(1)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Integer(1)),
            Ordering::Equal
        );
    }

    #[test]
    fn test_compare_values_text() {
        assert_eq!(
            compare_values(&Value::Text("a".into()), &Value::Text("b".into())),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&Value::Text("b".into()), &Value::Text("a".into())),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_values_null_last() {
        // NULL sorts last
        assert_eq!(
            compare_values(&Value::Null(DataType::Integer), &Value::Integer(1)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Null(DataType::Integer)),
            Ordering::Less
        );
    }

    // ============================================================================
    // Row Equality Tests
    // ============================================================================

    #[test]
    fn test_rows_equal() {
        let row1 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row3 = Row::from(vec![Value::Integer(2), Value::Text("a".into())]);

        assert!(rows_equal(&row1, &row2));
        assert!(!rows_equal(&row1, &row3));
    }

    #[test]
    fn test_rows_equal_different_lengths() {
        let row1 = Row::from(vec![Value::Integer(1)]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Integer(2)]);

        assert!(!rows_equal(&row1, &row2));
    }

    // ============================================================================
    // Column Name Extraction Tests
    // ============================================================================

    #[test]
    fn test_extract_column_name_identifier() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("name", TokenType::Identifier),
            "name".to_string(),
        ));
        assert_eq!(extract_column_name(&expr), Some("name".to_string()));
    }

    #[test]
    fn test_extract_column_name_qualified() {
        let expr = Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token("t.col", TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token("t", TokenType::Identifier),
                "t".to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token("col", TokenType::Identifier),
                "col".to_string(),
            )),
        });
        assert_eq!(extract_column_name(&expr), Some("col".to_string()));
    }

    #[test]
    fn test_extract_column_name_literal() {
        let expr = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        assert_eq!(extract_column_name(&expr), None);
    }

    // ============================================================================
    // Literal Value Extraction Tests
    // ============================================================================

    #[test]
    fn test_extract_literal_value_integer() {
        let expr = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        assert_eq!(extract_literal_value(&expr), Some(Value::Integer(42)));
    }

    #[test]
    fn test_extract_literal_value_float() {
        let expr = Expression::FloatLiteral(FloatLiteral {
            token: dummy_token("3.5", TokenType::Float),
            value: 3.5,
        });
        match extract_literal_value(&expr) {
            Some(Value::Float(f)) => assert!((f - 3.5).abs() < f64::EPSILON),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_extract_literal_value_string() {
        let expr = Expression::StringLiteral(StringLiteral {
            token: dummy_token("'hello'", TokenType::String),
            value: "hello".into(),
            type_hint: None,
        });
        assert_eq!(
            extract_literal_value(&expr),
            Some(Value::Text("hello".into()))
        );
    }

    #[test]
    fn test_extract_literal_value_boolean() {
        let expr = Expression::BooleanLiteral(BooleanLiteral {
            token: dummy_token("TRUE", TokenType::Keyword),
            value: true,
        });
        assert_eq!(extract_literal_value(&expr), Some(Value::Boolean(true)));
    }

    #[test]
    fn test_extract_literal_value_null() {
        let expr = Expression::NullLiteral(NullLiteral {
            token: dummy_token("NULL", TokenType::Keyword),
        });
        match extract_literal_value(&expr) {
            Some(Value::Null(_)) => {}
            _ => panic!("Expected Null"),
        }
    }

    // ============================================================================
    // Operator Tests
    // ============================================================================

    #[test]
    fn test_flip_operator() {
        assert_eq!(flip_operator(Operator::Lt), Operator::Gt);
        assert_eq!(flip_operator(Operator::Lte), Operator::Gte);
        assert_eq!(flip_operator(Operator::Gt), Operator::Lt);
        assert_eq!(flip_operator(Operator::Gte), Operator::Lte);
        assert_eq!(flip_operator(Operator::Eq), Operator::Eq);
        assert_eq!(flip_operator(Operator::Ne), Operator::Ne);
    }

    #[test]
    fn test_infix_to_operator() {
        assert_eq!(infix_to_operator(InfixOperator::Equal), Some(Operator::Eq));
        assert_eq!(
            infix_to_operator(InfixOperator::NotEqual),
            Some(Operator::Ne)
        );
        assert_eq!(
            infix_to_operator(InfixOperator::LessThan),
            Some(Operator::Lt)
        );
        assert_eq!(
            infix_to_operator(InfixOperator::LessEqual),
            Some(Operator::Lte)
        );
        assert_eq!(
            infix_to_operator(InfixOperator::GreaterThan),
            Some(Operator::Gt)
        );
        assert_eq!(
            infix_to_operator(InfixOperator::GreaterEqual),
            Some(Operator::Gte)
        );
        assert_eq!(infix_to_operator(InfixOperator::Add), None);
    }

    // ============================================================================
    // Base Column Name Tests
    // ============================================================================

    #[test]
    fn test_extract_base_column_name_simple() {
        assert_eq!(extract_base_column_name("column"), "column");
        assert_eq!(extract_base_column_name("COLUMN"), "column");
    }

    #[test]
    fn test_extract_base_column_name_qualified() {
        assert_eq!(extract_base_column_name("table.column"), "column");
        assert_eq!(extract_base_column_name("TABLE.COLUMN"), "column");
    }

    #[test]
    fn test_extract_base_column_name_multiple_dots() {
        // Uses rfind so finds last dot
        assert_eq!(extract_base_column_name("a.b.c"), "c");
    }

    // ============================================================================
    // Sorted Rows Tests
    // ============================================================================

    #[test]
    fn test_is_sorted_on_keys_empty() {
        let rows: Vec<Row> = vec![];
        assert!(is_sorted_on_keys(&rows, &[0]));
    }

    #[test]
    fn test_is_sorted_on_keys_single() {
        let rows = vec![Row::from(vec![Value::Integer(1)])];
        assert!(is_sorted_on_keys(&rows, &[0]));
    }

    #[test]
    fn test_is_sorted_on_keys_sorted() {
        let rows = vec![
            Row::from(vec![Value::Integer(1)]),
            Row::from(vec![Value::Integer(2)]),
            Row::from(vec![Value::Integer(3)]),
        ];
        assert!(is_sorted_on_keys(&rows, &[0]));
    }

    #[test]
    fn test_is_sorted_on_keys_unsorted() {
        let rows = vec![
            Row::from(vec![Value::Integer(3)]),
            Row::from(vec![Value::Integer(1)]),
            Row::from(vec![Value::Integer(2)]),
        ];
        assert!(!is_sorted_on_keys(&rows, &[0]));
    }

    #[test]
    fn test_is_sorted_on_keys_equal_values() {
        let rows = vec![
            Row::from(vec![Value::Integer(1)]),
            Row::from(vec![Value::Integer(1)]),
            Row::from(vec![Value::Integer(1)]),
        ];
        assert!(is_sorted_on_keys(&rows, &[0]));
    }

    #[test]
    fn test_is_sorted_on_keys_empty_indices() {
        let rows = vec![
            Row::from(vec![Value::Integer(3)]),
            Row::from(vec![Value::Integer(1)]),
        ];
        // With no key indices, trivially sorted
        assert!(is_sorted_on_keys(&rows, &[]));
    }

    // ============================================================================
    // Find Column Index Tests
    // ============================================================================

    #[test]
    fn test_find_column_index_exact_match() {
        let columns = vec!["id".to_string(), "name".to_string(), "age".to_string()];
        assert_eq!(
            find_column_index(&(None, "id".to_string()), &columns),
            Some(0)
        );
        assert_eq!(
            find_column_index(&(None, "name".to_string()), &columns),
            Some(1)
        );
    }

    #[test]
    fn test_find_column_index_qualified_match() {
        let columns = vec!["t.id".to_string(), "t.name".to_string()];
        assert_eq!(
            find_column_index(&(Some("t".to_string()), "id".to_string()), &columns),
            Some(0)
        );
    }

    #[test]
    fn test_find_column_index_suffix_match() {
        let columns = vec!["t1.id".to_string(), "t1.name".to_string()];
        // Without qualifier, can match suffix
        assert_eq!(
            find_column_index(&(None, "id".to_string()), &columns),
            Some(0)
        );
    }

    #[test]
    fn test_find_column_index_not_found() {
        let columns = vec!["id".to_string(), "name".to_string()];
        assert_eq!(
            find_column_index(&(None, "unknown".to_string()), &columns),
            None
        );
    }

    // ============================================================================
    // Verify Composite Key Equality Tests
    // ============================================================================

    #[test]
    fn test_verify_composite_key_equality_equal() {
        let row1 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);

        assert!(verify_composite_key_equality(
            &row1,
            &row2,
            &[0, 1],
            &[0, 1]
        ));
    }

    #[test]
    fn test_verify_composite_key_equality_not_equal() {
        let row1 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Text("b".into())]);

        assert!(!verify_composite_key_equality(
            &row1,
            &row2,
            &[0, 1],
            &[0, 1]
        ));
    }

    #[test]
    fn test_verify_composite_key_equality_partial() {
        let row1 = Row::from(vec![Value::Integer(1), Value::Text("a".into())]);
        let row2 = Row::from(vec![Value::Integer(1), Value::Text("b".into())]);

        // Only compare first column
        assert!(verify_composite_key_equality(&row1, &row2, &[0], &[0]));
    }

    // ============================================================================
    // Expression Contains Aggregate Tests
    // ============================================================================

    #[test]
    fn test_expression_contains_aggregate_count() {
        let expr = Expression::FunctionCall(Box::new(FunctionCall {
            token: dummy_token("COUNT", TokenType::Identifier),
            function: "COUNT".into(),
            arguments: vec![Expression::Identifier(Identifier::new(
                dummy_token("*", TokenType::Operator),
                "*".to_string(),
            ))],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        }));
        assert!(expression_contains_aggregate(&expr));
    }

    #[test]
    fn test_expression_contains_aggregate_non_aggregate() {
        let expr = Expression::FunctionCall(Box::new(FunctionCall {
            token: dummy_token("UPPER", TokenType::Identifier),
            function: "UPPER".into(),
            arguments: vec![Expression::Identifier(Identifier::new(
                dummy_token("name", TokenType::Identifier),
                "name".to_string(),
            ))],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        }));
        assert!(!expression_contains_aggregate(&expr));
    }

    #[test]
    fn test_expression_contains_aggregate_identifier() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        assert!(!expression_contains_aggregate(&expr));
    }

    // ============================================================================
    // Extract Column Name With Qualifier Tests
    // ============================================================================

    #[test]
    fn test_extract_column_name_with_qualifier_simple() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        assert_eq!(
            extract_column_name_with_qualifier(&expr),
            Some((None, "col".to_string()))
        );
    }

    #[test]
    fn test_extract_column_name_with_qualifier_qualified() {
        let expr = Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token("t.col", TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token("t", TokenType::Identifier),
                "t".to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token("col", TokenType::Identifier),
                "col".to_string(),
            )),
        });
        let result = extract_column_name_with_qualifier(&expr);
        assert!(result.is_some());
        let (qual, name) = result.unwrap();
        assert_eq!(qual, Some("t".to_string()));
        assert_eq!(name, "col");
    }

    // ============================================================================
    // String to DataType Tests
    // ============================================================================

    #[test]
    fn test_string_to_datatype() {
        assert_eq!(string_to_datatype("INTEGER"), DataType::Integer);
        assert_eq!(string_to_datatype("int"), DataType::Integer);
        assert_eq!(string_to_datatype("BIGINT"), DataType::Integer);
        assert_eq!(string_to_datatype("FLOAT"), DataType::Float);
        assert_eq!(string_to_datatype("DOUBLE"), DataType::Float);
        assert_eq!(string_to_datatype("TEXT"), DataType::Text);
        assert_eq!(string_to_datatype("VARCHAR"), DataType::Text);
        assert_eq!(string_to_datatype("BOOLEAN"), DataType::Boolean);
        assert_eq!(string_to_datatype("TIMESTAMP"), DataType::Timestamp);
        assert_eq!(string_to_datatype("JSON"), DataType::Json);
        assert_eq!(string_to_datatype("unknown"), DataType::Text);
    }

    // ============================================================================
    // Create Column Identifier Tests
    // ============================================================================

    #[test]
    fn test_create_column_identifier_simple() {
        let expr = create_column_identifier("col");
        match expr {
            Expression::Identifier(id) => assert_eq!(id.value, "col"),
            _ => panic!("Expected Identifier"),
        }
    }

    #[test]
    fn test_create_column_identifier_qualified() {
        let expr = create_column_identifier("t.col");
        match expr {
            Expression::QualifiedIdentifier(qid) => {
                assert_eq!(qid.qualifier.value, "t");
                assert_eq!(qid.name.value, "col");
            }
            _ => panic!("Expected QualifiedIdentifier"),
        }
    }

    // ============================================================================
    // Expression Has Parameters Tests
    // ============================================================================

    #[test]
    fn test_expression_has_parameters_true() {
        use crate::parser::ast::Parameter;
        let expr = Expression::Parameter(Parameter {
            token: dummy_token("$1", TokenType::Parameter),
            name: "$1".into(),
            index: 1,
        });
        assert!(expression_has_parameters(&expr));
    }

    #[test]
    fn test_expression_has_parameters_false() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        assert!(!expression_has_parameters(&expr));
    }

    #[test]
    fn test_expression_has_parameters_infix() {
        use crate::parser::ast::Parameter;
        let left = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let right = Expression::Parameter(Parameter {
            token: dummy_token("$1", TokenType::Parameter),
            name: "$1".into(),
            index: 1,
        });
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("=", TokenType::Operator),
            Box::new(left),
            "=",
            Box::new(right),
        ));
        assert!(expression_has_parameters(&expr));
    }

    // ============================================================================
    // Expressions Equivalent Tests
    // ============================================================================

    #[test]
    fn test_expressions_equivalent_identifiers() {
        let a = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let b = Expression::Identifier(Identifier::new(
            dummy_token("COL", TokenType::Identifier),
            "COL".to_string(),
        ));
        assert!(expressions_equivalent(&a, &b));
    }

    #[test]
    fn test_expressions_equivalent_integers() {
        let a = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        let b = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        assert!(expressions_equivalent(&a, &b));
    }

    #[test]
    fn test_expressions_equivalent_different() {
        let a = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        let b = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("43", TokenType::Integer),
            value: 43,
        });
        assert!(!expressions_equivalent(&a, &b));
    }

    // ============================================================================
    // Flatten AND Predicates Tests
    // ============================================================================

    #[test]
    fn test_flatten_and_predicates_single() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = flatten_and_predicates(&expr);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_flatten_and_predicates_multiple() {
        let a = Expression::Identifier(Identifier::new(
            dummy_token("a", TokenType::Identifier),
            "a".to_string(),
        ));
        let b = Expression::Identifier(Identifier::new(
            dummy_token("b", TokenType::Identifier),
            "b".to_string(),
        ));
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("AND", TokenType::Keyword),
            Box::new(a),
            "AND".to_string(),
            Box::new(b),
        ));
        let result = flatten_and_predicates(&expr);
        assert_eq!(result.len(), 2);
    }

    // ============================================================================
    // Combine Predicates With AND Tests
    // ============================================================================

    #[test]
    fn test_combine_predicates_empty() {
        let result = combine_predicates_with_and(vec![]);
        assert!(result.is_none());
    }

    #[test]
    fn test_combine_predicates_single() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = combine_predicates_with_and(vec![expr.clone()]);
        assert!(result.is_some());
    }

    #[test]
    fn test_combine_predicates_multiple() {
        let a = Expression::Identifier(Identifier::new(
            dummy_token("a", TokenType::Identifier),
            "a".to_string(),
        ));
        let b = Expression::Identifier(Identifier::new(
            dummy_token("b", TokenType::Identifier),
            "b".to_string(),
        ));
        let result = combine_predicates_with_and(vec![a, b]);
        assert!(result.is_some());
        if let Some(Expression::Infix(infix)) = result {
            assert_eq!(infix.operator.to_uppercase(), "AND");
        } else {
            panic!("Expected Infix expression");
        }
    }

    // ============================================================================
    // Extract AND Conditions Tests
    // ============================================================================

    #[test]
    fn test_extract_and_conditions_single() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = extract_and_conditions(&expr);
        assert_eq!(result.len(), 1);
    }

    // ============================================================================
    // Collect Table Qualifiers Tests
    // ============================================================================

    #[test]
    fn test_collect_table_qualifiers_empty() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = collect_table_qualifiers(&expr);
        assert!(result.is_empty());
    }

    #[test]
    fn test_collect_table_qualifiers_qualified() {
        let expr = Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token("t.col", TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token("t", TokenType::Identifier),
                "t".to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token("col", TokenType::Identifier),
                "col".to_string(),
            )),
        });
        let result = collect_table_qualifiers(&expr);
        assert_eq!(result.len(), 1);
        assert!(result.contains("t"));
    }

    // ============================================================================
    // Strip Table Qualifier Tests
    // ============================================================================

    #[test]
    fn test_strip_table_qualifier_qualified() {
        let expr = Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token("t.col", TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token("t", TokenType::Identifier),
                "t".to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token("col", TokenType::Identifier),
                "col".to_string(),
            )),
        });
        let result = strip_table_qualifier(&expr, "t");
        match result {
            Expression::Identifier(id) => assert_eq!(id.value, "col"),
            _ => panic!("Expected Identifier"),
        }
    }

    #[test]
    fn test_strip_table_qualifier_unqualified() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = strip_table_qualifier(&expr, "t");
        match result {
            Expression::Identifier(id) => assert_eq!(id.value, "col"),
            _ => panic!("Expected Identifier"),
        }
    }

    // ============================================================================
    // Add Table Qualifier Tests
    // ============================================================================

    #[test]
    fn test_add_table_qualifier_simple() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = add_table_qualifier(&expr, "t");
        match result {
            Expression::QualifiedIdentifier(qi) => {
                assert_eq!(qi.qualifier.value, "t");
                assert_eq!(qi.name.value, "col");
            }
            _ => panic!("Expected QualifiedIdentifier"),
        }
    }

    // ============================================================================
    // Expression To String Tests
    // ============================================================================

    #[test]
    fn test_expression_to_string_identifier() {
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("col", TokenType::Identifier),
            "col".to_string(),
        ));
        let result = expression_to_string(&expr);
        assert_eq!(result, "col");
    }

    #[test]
    fn test_expression_to_string_qualified() {
        let expr = Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: dummy_token("t.col", TokenType::Identifier),
            qualifier: Box::new(Identifier::new(
                dummy_token("t", TokenType::Identifier),
                "t".to_string(),
            )),
            name: Box::new(Identifier::new(
                dummy_token("col", TokenType::Identifier),
                "col".to_string(),
            )),
        });
        let result = expression_to_string(&expr);
        assert_eq!(result, "t.col");
    }

    #[test]
    fn test_expression_to_string_literal() {
        let expr = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42", TokenType::Integer),
            value: 42,
        });
        let result = expression_to_string(&expr);
        assert_eq!(result, "42");
    }

    // ============================================================================
    // Filter References Column Tests
    // ============================================================================

    #[test]
    fn test_filter_references_column_infix() {
        // Test with infix expression: target = 1
        let left = Expression::Identifier(Identifier::new(
            dummy_token("target", TokenType::Identifier),
            "target".to_string(),
        ));
        let right = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("1", TokenType::Integer),
            value: 1,
        });
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("=", TokenType::Operator),
            Box::new(left),
            "=".to_string(),
            Box::new(right),
        ));
        assert!(filter_references_column(&expr, "target"));
    }

    #[test]
    fn test_filter_references_column_no_match() {
        // Test with infix expression that doesn't reference target
        let left = Expression::Identifier(Identifier::new(
            dummy_token("other", TokenType::Identifier),
            "other".to_string(),
        ));
        let right = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("1", TokenType::Integer),
            value: 1,
        });
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("=", TokenType::Operator),
            Box::new(left),
            "=".to_string(),
            Box::new(right),
        ));
        assert!(!filter_references_column(&expr, "target"));
    }

    #[test]
    fn test_filter_references_column_simple_identifier() {
        // Simple identifiers without a comparison operator return false
        let expr = Expression::Identifier(Identifier::new(
            dummy_token("target", TokenType::Identifier),
            "target".to_string(),
        ));
        // The function is for filter expressions, not simple identifiers
        assert!(!filter_references_column(&expr, "target"));
    }

    // ============================================================================
    // Hash Value Into Tests (detailed)
    // ============================================================================

    #[test]
    fn test_hash_value_into_null() {
        use crate::core::types::DataType;
        use std::hash::DefaultHasher;
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        hash_value_into(&Value::Null(DataType::Integer), &mut hasher1);
        hash_value_into(&Value::Null(DataType::Integer), &mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_value_into_different_types() {
        use std::hash::DefaultHasher;
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        hash_value_into(&Value::Integer(42), &mut hasher1);
        hash_value_into(&Value::Text("42".into()), &mut hasher2);
        // Different types should produce different hashes
        assert_ne!(hasher1.finish(), hasher2.finish());
    }
}
