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

pub(crate) use crate::common::names::{eq_folded, eq_names};
use crate::common::SmartString;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::common::StringMap;

use crate::core::predicate::kind;
use crate::core::{DataType, Error, Operator, Row, Schema, Value};
use crate::executor::operators::index_nested_loop::ColumnSource;
use crate::parser::ast::{
    AliasedExpression, BetweenExpression, BooleanLiteral, CastExpression, Expression, FloatLiteral,
    FunctionCall, GroupByModifier, Identifier, InExpression, InHashSetExpression, InfixExpression,
    InfixOperator, IntegerLiteral, LikeExpression, ListExpression, NullLiteral, PrefixExpression,
    QualifiedIdentifier, SelectStatement, StringLiteral, WindowFrameBound,
};
use crate::parser::token::{Position, Token, TokenType};

/// Whether a table that keeps rows outside its index answers an equality
/// probe of `column` key by key: a B-tree on it, or a hash on a TEXT column
pub(crate) fn probes_by_equality(table: &dyn crate::storage::traits::Table, column: &str) -> bool {
    use crate::core::IndexType;
    table
        .get_index_on_column(column)
        .is_some_and(|index| match index.index_type() {
            IndexType::BTree => true,
            IndexType::Hash => table
                .schema()
                .columns
                .iter()
                .any(|c| c.data_type == DataType::Text && eq_names(&c.name_lower, column)),
            _ => false,
        })
}

// ============================================================================
// Token Creation Utilities
// ============================================================================

/// Helper to create a dummy token for internal AST construction.
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
        // A timestamp keeps its type through the literal's hint, the way
        // TIMESTAMP '...' does when written
        Value::Timestamp(_) => Expression::StringLiteral(StringLiteral {
            token: dummy_token(&format!("TIMESTAMP '{}'", v), TokenType::String),
            value: v.to_string().into(),
            type_hint: Some("TIMESTAMP".into()),
        }),
        // JSON and a vector keep theirs through an explicit CAST; a hint
        // alone would read as the loose one the parser puts on any text
        // shaped like an object
        Value::Extension(_) => {
            let type_name = match v.data_type() {
                crate::core::DataType::Vector => "VECTOR",
                _ => "JSON",
            };
            Expression::Cast(CastExpression {
                token: dummy_token("CAST", TokenType::Keyword),
                expr: Box::new(Expression::StringLiteral(StringLiteral {
                    token: dummy_token(&format!("'{}'", v), TokenType::String),
                    value: v.to_string().into(),
                    type_hint: None,
                })),
                type_name: type_name.into(),
            })
        }
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
    let mut bases: Vec<(String, usize)> = Vec::new();
    for (i, c) in columns.iter().enumerate() {
        let lower = c.to_lowercase();
        if let Some((_, base)) = split_qualified(&lower) {
            bases.push((base.to_string(), i));
        }
        map.insert(lower, i);
    }
    // Add unqualified fallbacks for qualified column names (e.g., "t.val" → "val")
    // Only add when the base name is unambiguous (appears in exactly one table)
    // and doesn't conflict with an existing entry (e.g., an unqualified column)
    let mut base_count: FxHashMap<&str, u8> = FxHashMap::default();
    for (base, _) in &bases {
        let entry = base_count.entry(base.as_str()).or_insert(0);
        *entry = entry.saturating_add(1);
    }
    let unique: Vec<bool> = bases
        .iter()
        .map(|(base, _)| base_count.get(base.as_str()) == Some(&1))
        .collect();
    for ((base, i), unique) in bases.into_iter().zip(unique) {
        if unique && !map.contains_key(&base) {
            map.insert(base, i);
        }
    }
    map
}

/// The column `qualifier.name` reads, lowercase, `own` the column so
/// qualified, `find` finding a column by its name, `qualifies` telling
/// whether a column is qualified by a qualifier and `bare` whether a column
/// is named bare: `own`, else the bare name when no column is qualified by
/// the qualifier, as over one source named bare, or when that column is
/// named bare, as one a USING or NATURAL join merges; None for another
/// source's
pub(crate) fn by_qualifier(
    own: Option<usize>,
    qualifier: &str,
    name: &str,
    find: impl FnOnce(&str) -> Option<usize>,
    qualifies: impl FnOnce(&str) -> bool,
    bare: impl FnOnce(usize) -> bool,
) -> Option<usize> {
    if own.is_some() {
        return own;
    }
    let found = find(name)?;
    (!qualifies(qualifier) || bare(found)).then_some(found)
}

/// [`by_qualifier`] over `map`, built by [`build_column_index_map`]
pub(crate) fn qualified_key_index(map: &StringMap<usize>, key: &str) -> Option<usize> {
    let own = map.get(key).copied();
    let Some((qualifier, name)) = split_qualified(key) else {
        return own;
    };
    by_qualifier(
        own,
        qualifier,
        name,
        |name| map.get(name).copied(),
        |qualifier| {
            map.keys()
                .any(|key| split_qualified(key).is_some_and(|(q, _)| q == qualifier))
        },
        |index| {
            !map.iter()
                .any(|(key, &at)| at == index && split_qualified(key).is_some())
        },
    )
}

/// [`qualified_key_index`] of `qualifier.name`, both lowercase
pub(crate) fn qualified_index(
    map: &StringMap<usize>,
    qualifier: &str,
    name: &str,
) -> Option<usize> {
    qualified_key_index(map, &qualified(qualifier, name))
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
            // decides the match exactly
            value.hash(&mut hasher);
        } else {
            // NULL marker
            0xDEADBEEFu64.hash(&mut hasher);
        }
    }

    hasher.finish()
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

/// Compare two Values for ordering, ascending with NULL last
pub fn compare_values(a: &Value, b: &Value) -> Ordering {
    crate::core::value::compare_order_key(Some(a), Some(b), true, None)
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

/// The columns an expression reads, by lower-case name; false when the
/// expression reads something a name cannot stand for (a star outside a
/// function's argument, a subquery, a window), so the caller reads every
/// column instead
pub(crate) fn collect_column_names(
    expr: &Expression,
    out: &mut Vec<crate::common::SmartString>,
) -> bool {
    visit_column_refs(expr, &mut |column| push_column_name(column, out))
}

/// The bare name of `column`, a column reference, added to `out`
fn push_column_name(column: &Expression, out: &mut Vec<crate::common::SmartString>) {
    match column {
        Expression::Identifier(id) => out.push(id.value_lower.clone()),
        Expression::QualifiedIdentifier(q) => out.push(q.name.value_lower.clone()),
        _ => {}
    }
}

/// Each column reference an expression reads, a bare or a qualified name,
/// passed to `visit`, the parts past one it cannot read included; false as
/// `collect_column_names` says
pub(crate) fn visit_column_refs<'e>(
    expr: &'e Expression,
    visit: &mut dyn FnMut(&'e Expression),
) -> bool {
    visit_column_refs_outside(expr, &|_| false, visit)
}

/// `visit_column_refs` past every part `skip` takes, whose references are not visited
pub(crate) fn visit_column_refs_outside<'e>(
    expr: &'e Expression,
    skip: &dyn Fn(&Expression) -> bool,
    visit: &mut dyn FnMut(&'e Expression),
) -> bool {
    fn all<'e>(
        exprs: impl IntoIterator<Item = &'e Expression>,
        skip: &dyn Fn(&Expression) -> bool,
        visit: &mut dyn FnMut(&'e Expression),
    ) -> bool {
        exprs.into_iter().fold(true, |read, e| {
            visit_column_refs_outside(e, skip, visit) & read
        })
    }
    if skip(expr) {
        return true;
    }
    let one = |e: &'e Expression, visit: &mut dyn FnMut(&'e Expression)| {
        visit_column_refs_outside(e, skip, visit)
    };
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
            visit(expr);
            true
        }
        Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::IntervalLiteral(_)
        | Expression::Parameter(_) => true,
        Expression::Prefix(p) => one(&p.right, visit),
        Expression::Infix(i) => all([&*i.left, &*i.right], skip, visit),
        Expression::List(l) => all(&l.elements, skip, visit),
        Expression::ExpressionList(l) => all(&l.expressions, skip, visit),
        Expression::Distinct(d) => one(&d.expr, visit),
        Expression::In(i) => all([&*i.left, &*i.right], skip, visit),
        Expression::InHashSet(h) => one(&h.column, visit),
        Expression::QuantifiedSet(q) => one(&q.left, visit),
        Expression::Between(b) => all([&*b.expr, &*b.lower, &*b.upper], skip, visit),
        Expression::Like(l) => all(
            [&*l.left, &*l.pattern]
                .into_iter()
                .chain(l.escape.as_deref()),
            skip,
            visit,
        ),
        Expression::Case(c) => all(
            c.value
                .as_deref()
                .into_iter()
                .chain(
                    c.when_clauses
                        .iter()
                        .flat_map(|w| [&w.condition, &w.then_result]),
                )
                .chain(c.else_value.as_deref()),
            skip,
            visit,
        ),
        Expression::Cast(c) => one(&c.expr, visit),
        Expression::FunctionCall(f) => all(
            f.arguments
                .iter()
                .filter(|a| !matches!(a, Expression::Star(_)))
                .chain(f.filter.as_deref())
                .chain(f.order_by.iter().map(|o| &o.expression)),
            skip,
            visit,
        ),
        Expression::Aliased(a) => one(&a.expression, visit),
        _ => false,
    }
}

/// The columns a statement reads, as a mask over the columns of its row,
/// each name marking every column it spells, from its SELECT list,
/// `filters`, GROUP BY, HAVING and ORDER BY; None when any of them reads
/// something a name cannot stand for, or a name that is neither a column
/// nor a SELECT alias
pub(crate) fn needed_columns<'a, 'c>(
    stmt: &'a SelectStatement,
    filters: impl IntoIterator<Item = &'a Expression>,
    columns_lower: impl Iterator<Item = &'c str> + Clone,
) -> Option<Vec<bool>> {
    let mut names = Vec::new();
    let grouping_sets = match &stmt.group_by.modifier {
        GroupByModifier::GroupingSets(sets) => sets.as_slice(),
        _ => &[],
    };
    let exprs = stmt
        .columns
        .iter()
        .chain(stmt.distinct_on.iter())
        .chain(stmt.group_by.columns.iter())
        .chain(grouping_sets.iter().flatten())
        .chain(stmt.having.as_deref())
        .chain(stmt.order_by.iter().map(|o| &o.expression))
        .chain(filters);
    // A window's columns are its call's, and its partition's, its order's and
    // its frame's as it runs with them, a named window's from its definition
    let window_names = std::cell::RefCell::new(Vec::new());
    let windows_read = std::cell::Cell::new(true);
    let skip = |expr: &Expression| {
        let Expression::Window(window) = expr else {
            return false;
        };
        let Some((partition_by, order_by, frame)) = window.resolved(&stmt.window_defs) else {
            windows_read.set(false);
            return true;
        };
        let call = &window.function;
        let parts = call
            .arguments
            .iter()
            .filter(|a| !matches!(a, Expression::Star(_)))
            .chain(call.filter.as_deref())
            .chain(call.order_by.iter().map(|o| &o.expression))
            .chain(partition_by)
            .chain(order_by.iter().map(|o| &o.expression))
            .chain(frame.into_iter().flat_map(|frame| frame.offsets()));
        let mut window_names = window_names.borrow_mut();
        let read = parts.fold(true, |read, part| {
            collect_column_names(part, &mut window_names) & read
        });
        windows_read.set(windows_read.get() && read);
        true
    };
    for expr in exprs {
        let read = visit_column_refs_outside(expr, &skip, &mut |c| push_column_name(c, &mut names));
        if !read {
            return None;
        }
    }
    if !windows_read.get() {
        return None;
    }
    names.extend(window_names.into_inner());
    let mut mask = vec![false; columns_lower.clone().count()];
    for name in names {
        let mut spelled = false;
        for (read, column) in mask.iter_mut().zip(columns_lower.clone()) {
            if column == name.as_str() {
                *read = true;
                spelled = true;
            }
        }
        let is_alias = || {
            stmt.columns
                .iter()
                .any(|c| matches!(c, Expression::Aliased(a) if a.alias.value_lower == name))
        };
        if !spelled && !is_alias() {
            return None;
        }
    }
    Some(mask)
}

/// Extract the column name from an Identifier or QualifiedIdentifier expression.
/// Returns the column name without table qualifier.
#[inline]
pub fn extract_column_name(expr: &Expression) -> Option<String> {
    column_name_of(expr).map(SmartString::to_string)
}

/// The column name `extract_column_name` returns, as the expression holds it
#[inline]
pub fn column_name_of(expr: &Expression) -> Option<&SmartString> {
    match expr {
        Expression::Identifier(Identifier { value, .. }) => Some(value),
        Expression::QualifiedIdentifier(QualifiedIdentifier { name, .. }) => Some(&name.value),
        // A column the prepared builder bound in the table's own scope
        Expression::Column(column) => Some(&column.name.value),
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
        Expression::StringLiteral(s) => Some(string_literal_value(s)),
        Expression::BooleanLiteral(b) => Some(Value::Boolean(b.value)),
        Expression::NullLiteral(_) => Some(Value::Null(DataType::Text)),
        _ => None,
    }
}

/// The value a string literal stands for: a TIMESTAMP for `TIMESTAMP '...'`,
/// `DATETIME '...'` and `DATE '...'` when the text parses, a TEXT otherwise
pub fn string_literal_value(lit: &StringLiteral) -> Value {
    lit.type_hint
        .as_deref()
        .and_then(|hint| crate::core::value::typed_literal_timestamp(hint, &lit.value))
        .unwrap_or_else(|| Value::Text(lit.value.clone()))
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
    bare_name(col_name).to_lowercase()
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
            crate::core::value::f64_eq(la.value, lb.value)
        }
        (Expression::StringLiteral(la), Expression::StringLiteral(lb)) => {
            la.value == lb.value
                && match (&la.type_hint, &lb.type_hint) {
                    (Some(ha), Some(hb)) => ha.eq_ignore_ascii_case(hb),
                    (ha, hb) => ha.is_none() && hb.is_none(),
                }
        }
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
    let mut preds = preds.into_iter();
    let first = preds.next()?;
    let and = |pred| {
        let token = Token::new(TokenType::Keyword, "AND", Position::default());
        (token, SmartString::const_new("AND"), pred)
    };
    Some(InfixExpression::balanced_chain(first, preds.map(and)))
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
        Expression::QuantifiedSet(quantified) => {
            collect_table_qualifiers_impl(&quantified.left, qualifiers);
        }
        Expression::AllAny(all_any) => {
            collect_table_qualifiers_impl(&all_any.left, qualifiers);
        }
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
        Expression::Between(between) => {
            collect_table_qualifiers_impl(&between.expr, qualifiers);
            collect_table_qualifiers_impl(&between.lower, qualifiers);
            collect_table_qualifiers_impl(&between.upper, qualifiers);
        }
        Expression::Like(like) => {
            collect_table_qualifiers_impl(&like.left, qualifiers);
            collect_table_qualifiers_impl(&like.pattern, qualifiers);
            if let Some(escape) = &like.escape {
                collect_table_qualifiers_impl(escape, qualifiers);
            }
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

/// Whether an ORDER BY `expr` of `stmt` sorts by a SELECT alias rather than
/// the input column of its name: a bare name the list gives to anything but
/// that column, which only the same bare name proves, or over a single
/// table also that name qualified. A qualified name or an expression always
/// reads the input, so a path that orders input rows can answer every item
/// this is false for
pub fn order_by_names_an_alias(expr: &Expression, stmt: &SelectStatement) -> bool {
    let Expression::Identifier(id) = expr else {
        return false;
    };
    let one_table = matches!(stmt.table_expr.as_deref(), Some(Expression::TableSource(_)));
    stmt.columns.iter().any(|item| match item {
        Expression::Aliased(aliased) if aliased.alias.value_lower == id.value_lower => {
            match &*aliased.expression {
                Expression::Identifier(column) => column.value_lower != id.value_lower,
                Expression::QualifiedIdentifier(column) => {
                    !one_table || column.name.value_lower != id.value_lower
                }
                _ => true,
            }
        }
        _ => false,
    })
}

/// Collect the unqualified (bare) column names of an expression, lowercase,
/// through the same expressions `collect_table_qualifiers` reads
pub fn collect_unqualified_columns(expr: &Expression, columns: &mut FxHashSet<String>) {
    match expr {
        Expression::Identifier(id) => {
            columns.insert(id.value_lower.to_string());
        }
        Expression::Infix(infix) => {
            collect_unqualified_columns(&infix.left, columns);
            collect_unqualified_columns(&infix.right, columns);
        }
        Expression::Prefix(prefix) => {
            collect_unqualified_columns(&prefix.right, columns);
        }
        Expression::In(in_expr) => {
            collect_unqualified_columns(&in_expr.left, columns);
            match in_expr.right.as_ref() {
                Expression::ExpressionList(el) => {
                    for elem in &el.expressions {
                        collect_unqualified_columns(elem, columns);
                    }
                }
                Expression::List(list) => {
                    for elem in &list.elements {
                        collect_unqualified_columns(elem, columns);
                    }
                }
                other => {
                    collect_unqualified_columns(other, columns);
                }
            }
        }
        Expression::InHashSet(in_set) => {
            collect_unqualified_columns(&in_set.column, columns);
        }
        Expression::QuantifiedSet(quantified) => {
            collect_unqualified_columns(&quantified.left, columns);
        }
        Expression::AllAny(all_any) => {
            collect_unqualified_columns(&all_any.left, columns);
        }
        Expression::ExpressionList(el) => {
            for elem in &el.expressions {
                collect_unqualified_columns(elem, columns);
            }
        }
        Expression::List(list) => {
            for elem in &list.elements {
                collect_unqualified_columns(elem, columns);
            }
        }
        Expression::Between(between) => {
            collect_unqualified_columns(&between.expr, columns);
            collect_unqualified_columns(&between.lower, columns);
            collect_unqualified_columns(&between.upper, columns);
        }
        Expression::Like(like) => {
            collect_unqualified_columns(&like.left, columns);
            collect_unqualified_columns(&like.pattern, columns);
            if let Some(escape) = &like.escape {
                collect_unqualified_columns(escape, columns);
            }
        }
        Expression::FunctionCall(func) => {
            for arg in &func.arguments {
                collect_unqualified_columns(arg, columns);
            }
        }
        Expression::Aliased(aliased) => {
            collect_unqualified_columns(&aliased.expression, columns);
        }
        Expression::Cast(cast) => {
            collect_unqualified_columns(&cast.expr, columns);
        }
        Expression::Case(case) => {
            if let Some(ref val) = case.value {
                collect_unqualified_columns(val, columns);
            }
            for when in &case.when_clauses {
                collect_unqualified_columns(&when.condition, columns);
                collect_unqualified_columns(&when.then_result, columns);
            }
            if let Some(ref else_val) = case.else_value {
                collect_unqualified_columns(else_val, columns);
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

/// The lowercased table names and aliases of every leaf of a FROM
/// expression, so a nested join lists all of its tables
pub fn collect_table_aliases(expr: &Expression, out: &mut Vec<String>) {
    match expr {
        Expression::JoinSource(join) => {
            collect_table_aliases(&join.left, out);
            collect_table_aliases(&join.right, out);
        }
        Expression::Aliased(aliased) => out.push(aliased.alias.value_lower.to_string()),
        _ => {
            if let Some(alias) = get_table_alias_from_expr(expr) {
                out.push(alias.to_lowercase());
            }
        }
    }
}

/// Strip table qualifier from an expression, replacing qualified identifiers
/// with unqualified ones. Used when pushing filters to individual table scans.
pub fn strip_table_qualifier(expr: &Expression, table_alias: &str) -> Expression {
    let alias_lower = table_alias.to_lowercase();

    match expr {
        Expression::QualifiedIdentifier(qi) if qi.qualifier.value_lower.as_str() == alias_lower => {
            // Convert to simple identifier
            Expression::Identifier(qi.name.clone())
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
        Expression::QuantifiedSet(quantified) => {
            quantified.with_left(strip_table_qualifier(&quantified.left, table_alias))
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
                qualifier: Identifier::new(
                    Token::new(TokenType::Identifier, table_alias, Position::default()),
                    table_alias.to_string(),
                ),
                name: id.clone(),
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
        Expression::QuantifiedSet(quantified) => {
            quantified.with_left(add_table_qualifier(&quantified.left, table_alias))
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

/// The error of `call` written with DISTINCT, ORDER BY or FILTER to a
/// function `registry` holds no aggregate of: those modifiers shape the
/// rows an aggregate reads, and no other call reads rows
pub(crate) fn modifier_outside_aggregate(
    call: &FunctionCall,
    registry: &crate::functions::FunctionRegistry,
) -> Option<String> {
    let modifier = call.aggregate_modifier()?;
    (!registry.is_aggregate(&call.function)).then(|| {
        format!(
            "{modifier} is allowed only in an aggregate call, and {} is not an aggregate function",
            call.function.to_uppercase()
        )
    })
}

/// The first form in `statement`, at any depth, that no path answers: a call
/// [`modifier_outside_aggregate`] refuses, a window function outside a SELECT
/// list, DISTINCT ON and ORDER BY or inside another call's input, ANY or ALL
/// with an operator that does not compare. Checked before any path runs; a
/// table's CHECK and DEFAULT take the modifier rule where they compile
pub(crate) fn statement_form_error(
    statement: &crate::parser::ast::Statement,
    registry: &crate::functions::FunctionRegistry,
) -> crate::core::Result<()> {
    FormWalk { registry }
        .statement(statement)
        .map_err(crate::core::Error::invalid_argument)
}

/// Where an expression stands, for the window functions it may hold
#[derive(Clone, Copy)]
enum At<'e> {
    /// A SELECT list, DISTINCT ON or ORDER BY item
    Output,
    /// Any other place a statement reads
    Elsewhere,
    /// The input of the aggregate or window call named
    Inside(&'e str),
}

struct FormWalk<'a> {
    registry: &'a crate::functions::FunctionRegistry,
}

impl FormWalk<'_> {
    fn statement(&self, statement: &crate::parser::ast::Statement) -> Result<(), String> {
        use crate::parser::ast::Statement;
        let at = At::Elsewhere;
        match statement {
            Statement::Select(stmt) => self.select(stmt),
            Statement::Insert(stmt) => {
                stmt.values.iter().try_for_each(|row| self.all(row, at))?;
                stmt.select.as_deref().map_or(Ok(()), |s| self.select(s))?;
                self.all(&stmt.update_expressions, at)?;
                self.optional(stmt.update_where.as_deref(), at)?;
                self.all(&stmt.returning, at)
            }
            Statement::Update(stmt) => {
                self.with(stmt.with.as_ref())?;
                stmt.updates
                    .values()
                    .try_for_each(|e| self.expression(e, at))?;
                self.optional(stmt.where_clause.as_deref(), at)?;
                self.all(&stmt.returning, at)
            }
            Statement::Delete(stmt) => {
                self.with(stmt.with.as_ref())?;
                self.optional(stmt.where_clause.as_deref(), at)?;
                self.all(&stmt.returning, at)
            }
            Statement::CreateView(stmt) => self.select(&stmt.query),
            Statement::CreateTable(stmt) => {
                stmt.as_select.as_deref().map_or(Ok(()), |s| self.select(s))
            }
            Statement::Explain(stmt) => self.statement(&stmt.statement),
            Statement::Expression(stmt) => self.expression(&stmt.expression, at),
            Statement::Set(stmt) => self.expression(&stmt.value, at),
            Statement::Pragma(stmt) => self.optional(stmt.value.as_ref(), at),
            Statement::Truncate(_)
            | Statement::DropTable(_)
            | Statement::AlterTable(_)
            | Statement::CreateIndex(_)
            | Statement::DropIndex(_)
            | Statement::DropView(_)
            | Statement::Begin(_)
            | Statement::Commit(_)
            | Statement::Rollback(_)
            | Statement::Savepoint(_)
            | Statement::ReleaseSavepoint(_)
            | Statement::ShowTables(_)
            | Statement::ShowViews(_)
            | Statement::ShowCreateTable(_)
            | Statement::ShowCreateView(_)
            | Statement::ShowIndexes(_)
            | Statement::Describe(_)
            | Statement::Analyze(_)
            | Statement::Vacuum(_)
            | Statement::Copy(_) => Ok(()),
        }
    }

    fn select(&self, stmt: &SelectStatement) -> Result<(), String> {
        let SelectStatement {
            token: _,
            param_count: _,
            distinct: _,
            distinct_on,
            columns,
            with,
            table_expr,
            where_clause,
            group_by,
            having,
            window_defs: _,
            order_by,
            limit,
            offset,
            set_operations,
            prepared: _,
        } = stmt;
        let at = At::Elsewhere;
        self.all(distinct_on, At::Output)?;
        self.all(columns, At::Output)?;
        self.with(with.as_ref())?;
        self.optional(table_expr.as_deref(), at)?;
        self.optional(where_clause.as_deref(), at)?;
        self.all(&group_by.columns, at)?;
        if let GroupByModifier::GroupingSets(sets) = &group_by.modifier {
            sets.iter().try_for_each(|set| self.all(set, at))?;
        }
        self.optional(having.as_deref(), at)?;
        // Only the parts of a named window that a window function reads are read
        super::Executor::try_for_each_used_window_value(stmt, &mut |e| self.expression(e, at))?;
        self.ordered(order_by, At::Output)?;
        self.optional(limit.as_deref(), at)?;
        self.optional(offset.as_deref(), at)?;
        set_operations
            .iter()
            .try_for_each(|op| self.select(&op.right))
    }

    fn with(&self, with: Option<&crate::parser::ast::WithClause>) -> Result<(), String> {
        with.map_or(Ok(()), |with| {
            with.ctes.iter().try_for_each(|cte| self.select(&cte.query))
        })
    }

    fn expression<'e>(&self, expr: &'e Expression, at: At<'e>) -> Result<(), String> {
        match expr {
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::Column(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_)
            | Expression::CteReference(_) => Ok(()),
            Expression::Prefix(prefix) => self.expression(&prefix.right, at),
            Expression::Infix(infix) => {
                self.expression(&infix.left, at)?;
                self.expression(&infix.right, at)
            }
            Expression::List(list) => self.all(&list.elements, at),
            Expression::ExpressionList(list) => self.all(&list.expressions, at),
            Expression::Distinct(distinct) => self.expression(&distinct.expr, at),
            Expression::Exists(exists) => self.select(&exists.subquery),
            Expression::AllAny(all_any) => {
                if super::expression::QuantifiedMembers::operator(&all_any.operator).is_none() {
                    return Err(format!(
                        "{} takes a comparison operator (=, <>, <, <=, >, >=), not {}",
                        all_any.all_any_type, all_any.operator
                    ));
                }
                self.expression(&all_any.left, at)?;
                self.select(&all_any.subquery)
            }
            Expression::In(in_expr) => {
                self.expression(&in_expr.left, at)?;
                self.expression(&in_expr.right, at)
            }
            Expression::InHashSet(in_set) => self.expression(&in_set.column, at),
            Expression::QuantifiedSet(set) => self.expression(&set.left, at),
            Expression::Between(between) => {
                self.expression(&between.expr, at)?;
                self.expression(&between.lower, at)?;
                self.expression(&between.upper, at)
            }
            Expression::Like(like) => {
                self.expression(&like.left, at)?;
                self.expression(&like.pattern, at)?;
                self.optional(like.escape.as_deref(), at)
            }
            Expression::ScalarSubquery(subquery) => self.select(&subquery.subquery),
            Expression::Case(case) => {
                self.optional(case.value.as_deref(), at)?;
                for when in &case.when_clauses {
                    self.expression(&when.condition, at)?;
                    self.expression(&when.then_result, at)?;
                }
                self.optional(case.else_value.as_deref(), at)
            }
            Expression::Cast(cast) => self.expression(&cast.expr, at),
            // Which calls aggregate matters only to a window function under one
            Expression::FunctionCall(call)
                if !matches!(at, At::Inside(_))
                    && expr.holds_window()
                    && self.registry.is_aggregate(&call.function) =>
            {
                self.call(call, At::Inside(&call.function))
            }
            Expression::FunctionCall(call) => self.call(call, at),
            Expression::Aliased(aliased) => self.expression(&aliased.expression, at),
            Expression::Window(window) => {
                let name = || window.function.function.to_uppercase();
                match at {
                    At::Output => {}
                    At::Elsewhere => return Err(format!(
                        "the window function {} is allowed only in a SELECT list, DISTINCT ON or ORDER BY",
                        name()
                    )),
                    At::Inside(outer) => {
                        return Err(format!(
                            "the window function {} cannot be read inside a call of {}",
                            name(),
                            outer.to_uppercase()
                        ))
                    }
                }
                let inner = At::Inside(&window.function.function);
                self.call(&window.function, inner)?;
                self.all(&window.partition_by, inner)?;
                self.ordered(&window.order_by, inner)?;
                window
                    .frame_offsets()
                    .try_for_each(|e| self.expression(e, inner))
            }
            Expression::TableSource(table) => {
                self.optional(table.as_of.as_ref().map(|as_of| &*as_of.value), at)
            }
            Expression::JoinSource(join) => {
                self.expression(&join.left, at)?;
                self.expression(&join.right, at)?;
                self.optional(join.condition.as_deref(), at)
            }
            Expression::SubquerySource(source) => self.select(&source.subquery),
            Expression::ValuesSource(values) => {
                values.rows.iter().try_for_each(|row| self.all(row, at))
            }
            Expression::FunctionTableSource(source) => self.all(&source.arguments, at),
        }
    }

    fn call<'e>(&self, call: &'e FunctionCall, at: At<'e>) -> Result<(), String> {
        if let Some(error) = modifier_outside_aggregate(call, self.registry) {
            return Err(error);
        }
        self.all(&call.arguments, at)?;
        self.ordered(&call.order_by, at)?;
        self.optional(call.filter.as_deref(), at)
    }

    fn all<'e>(&self, exprs: &'e [Expression], at: At<'e>) -> Result<(), String> {
        exprs.iter().try_for_each(|e| self.expression(e, at))
    }

    fn optional<'e>(&self, expr: Option<&'e Expression>, at: At<'e>) -> Result<(), String> {
        expr.map_or(Ok(()), |e| self.expression(e, at))
    }

    fn ordered<'e>(
        &self,
        order: &'e [crate::parser::ast::OrderByExpression],
        at: At<'e>,
    ) -> Result<(), String> {
        order
            .iter()
            .try_for_each(|o| self.expression(&o.expression, at))
    }
}

/// An error when `where_clause`, a statement's WHERE, holds an aggregate
pub(crate) fn where_without_aggregates(
    where_clause: Option<&Expression>,
) -> crate::core::Result<()> {
    match where_clause.is_some_and(expression_contains_aggregate) {
        true => Err(Error::invalid_argument(
            "aggregate functions are not allowed in WHERE clause (use HAVING instead)",
        )),
        false => Ok(()),
    }
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
        Expression::QuantifiedSet(set) => expression_contains_aggregate(&set.left),
        Expression::In(in_expr) => {
            expression_contains_aggregate(&in_expr.left)
                || expression_contains_aggregate(&in_expr.right)
        }
        Expression::List(list) => list.elements.iter().any(expression_contains_aggregate),
        Expression::ExpressionList(list) => {
            list.expressions.iter().any(expression_contains_aggregate)
        }
        Expression::Between(between) => {
            expression_contains_aggregate(&between.expr)
                || expression_contains_aggregate(&between.lower)
                || expression_contains_aggregate(&between.upper)
        }
        Expression::Like(like) => {
            expression_contains_aggregate(&like.left)
                || expression_contains_aggregate(&like.pattern)
                || like
                    .escape
                    .as_deref()
                    .is_some_and(expression_contains_aggregate)
        }
        Expression::AllAny(all_any) => expression_contains_aggregate(&all_any.left),
        Expression::Distinct(distinct) => expression_contains_aggregate(&distinct.expr),
        Expression::Case(case) => {
            if case
                .value
                .as_deref()
                .is_some_and(expression_contains_aggregate)
            {
                return true;
            }
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
    let qualified = qualifier.as_ref().map(|q| qualified(q, col_name));

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
        for (idx, column) in columns.iter().enumerate() {
            if split_qualified(column).is_some_and(|(_, name)| name.to_lowercase() == *col_name) {
                return Some(idx);
            }
        }
    }

    None
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

/// The name a derived table gives an aggregate column it leaves unaliased,
/// unless a TEXT literal in it would match a column of the other case
pub fn derived_aggregate_column_name(expr: &Expression) -> Option<String> {
    let Expression::FunctionCall(func) = expr else {
        return None;
    };
    if !is_aggregate_function(&func.function) || holds_text_literal(expr) {
        return None;
    }
    Some(expression_to_string(expr))
}

/// Whether a TEXT literal may sit anywhere in `expr`; a kind not walked here counts as holding one
fn holds_text_literal(expr: &Expression) -> bool {
    match expr {
        Expression::StringLiteral(_) => true,
        Expression::Identifier(_)
        | Expression::QualifiedIdentifier(_)
        | Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::IntervalLiteral(_)
        | Expression::Parameter(_)
        | Expression::Star(_) => false,
        Expression::FunctionCall(func) => {
            func.arguments.iter().any(holds_text_literal)
                || func.filter.as_deref().is_some_and(holds_text_literal)
                || func
                    .order_by
                    .iter()
                    .any(|o| holds_text_literal(&o.expression))
        }
        Expression::Infix(infix) => {
            holds_text_literal(&infix.left) || holds_text_literal(&infix.right)
        }
        Expression::Prefix(prefix) => holds_text_literal(&prefix.right),
        Expression::Cast(cast) => holds_text_literal(&cast.expr),
        Expression::Distinct(distinct) => holds_text_literal(&distinct.expr),
        Expression::Aliased(aliased) => holds_text_literal(&aliased.expression),
        Expression::Case(case) => {
            case.value.as_deref().is_some_and(holds_text_literal)
                || case
                    .when_clauses
                    .iter()
                    .any(|w| holds_text_literal(&w.condition) || holds_text_literal(&w.then_result))
                || case.else_value.as_deref().is_some_and(holds_text_literal)
        }
        Expression::Between(between) => {
            holds_text_literal(&between.expr)
                || holds_text_literal(&between.lower)
                || holds_text_literal(&between.upper)
        }
        Expression::In(in_expr) => {
            holds_text_literal(&in_expr.left) || holds_text_literal(&in_expr.right)
        }
        Expression::List(list) => list.elements.iter().any(holds_text_literal),
        Expression::ExpressionList(list) => list.expressions.iter().any(holds_text_literal),
        Expression::Like(like) => {
            holds_text_literal(&like.left)
                || holds_text_literal(&like.pattern)
                || like.escape.as_deref().is_some_and(holds_text_literal)
        }
        _ => true,
    }
}

/// A function call as a result column names it: `COUNT(DISTINCT g)`
pub fn function_call_to_string(func: &crate::parser::ast::FunctionCall) -> String {
    name_text(written_call(func))
}

/// `function_call_to_string` before `name_text`, as `written_text` nests it
fn written_call(func: &crate::parser::ast::FunctionCall) -> String {
    let args: Vec<String> = func.arguments.iter().map(written_text).collect();
    let distinct = if func.is_distinct { "DISTINCT " } else { "" };
    format!("{}({}{})", func.function, distinct, args.join(", "))
}

/// The name a result column takes from its SELECT item on every path: its
/// alias, a column's own name, any other item as written
pub fn item_name(expr: &Expression) -> String {
    match item_spelled_name(expr) {
        Some(name) => name.to_string(),
        None => expression_to_string(expr),
    }
}

/// Whether `expr`, or a statement inside it, names an unquoted rowid, or the
/// row id slot a spelled one reads
pub(crate) fn expr_names_rowid(expr: &Expression) -> bool {
    let names = expr_names_rowid;
    let rowid = |name: &str| crate::core::is_rowid_name(name) || name == crate::core::ROWID_SLOT;
    match expr {
        Expression::Identifier(id) => !id.quoted && rowid(&id.value_lower),
        Expression::QualifiedIdentifier(q) => !q.name.quoted && rowid(&q.name.value_lower),
        Expression::ScalarSubquery(sq) => names_rowid(&sq.subquery),
        Expression::Exists(exists) => names_rowid(&exists.subquery),
        Expression::AllAny(all_any) => names(&all_any.left) || names_rowid(&all_any.subquery),
        Expression::SubquerySource(source) => names_rowid(&source.subquery),
        Expression::JoinSource(join) => {
            names(&join.left) || names(&join.right) || join.condition.as_deref().is_some_and(names)
        }
        Expression::ValuesSource(values) => values.rows.iter().flatten().any(names),
        Expression::FunctionTableSource(function) => function.arguments.iter().any(names),
        other => other
            .try_for_each_child(&mut |child| match names(child) {
                true => Err(()),
                false => Ok(()),
            })
            .is_err(),
    }
}

/// Whether `stmt`, or a statement inside it, names an unquoted rowid
pub(crate) fn names_rowid(stmt: &SelectStatement) -> bool {
    let names = expr_names_rowid;
    stmt.columns
        .iter()
        .chain(&stmt.distinct_on)
        .chain(stmt.table_expr.as_deref())
        .chain(stmt.where_clause.as_deref())
        .chain(&stmt.group_by.columns)
        .chain(stmt.having.as_deref())
        .chain(stmt.order_by.iter().map(|o| &o.expression))
        .chain(stmt.window_defs.iter().flat_map(|w| w.expressions()))
        .any(names)
        || stmt
            .with
            .iter()
            .flat_map(|w| &w.ctes)
            .any(|cte| names_rowid(&cte.query))
        || stmt.set_operations.iter().any(|op| names_rowid(&op.right))
}

/// `expr` with each unquoted rowid name outside its subqueries replaced by
/// what `spelled` gives for it
pub(crate) fn rowids_respelled(
    expr: &Expression,
    spelled: &mut dyn FnMut(&Expression) -> crate::core::Result<Option<Expression>>,
) -> crate::core::Result<Expression> {
    let name = match expr {
        Expression::Identifier(id) if !id.quoted => &id.value_lower,
        Expression::QualifiedIdentifier(q) if !q.name.quoted => &q.name.value_lower,
        other => return other.try_map_children(&mut |child| rowids_respelled(child, spelled)),
    };
    match crate::core::is_rowid_name(name) {
        true => Ok(spelled(expr)?.unwrap_or_else(|| expr.clone())),
        false => Ok(expr.clone()),
    }
}

/// `column` of the FROM entry named `table`, as a qualified reference, which
/// no SELECT alias and no literal can stand for
pub(crate) fn qualified_reference(table: &str, column: &str) -> Expression {
    let named = |name: &str| Identifier::new(dummy_token(name, TokenType::Identifier), name);
    Expression::QualifiedIdentifier(QualifiedIdentifier {
        qualifier: named(table),
        name: named(column),
    })
}

/// `item`, standing for the SELECT item `written`, under the name `written`
/// gives when `item`'s own would differ
pub fn named_as_written(item: Expression, written: &Expression) -> Expression {
    match written {
        Expression::Aliased(_) => item,
        _ if item == *written => item,
        _ => {
            let name = item_name(written);
            let token = dummy_token(&name, TokenType::Identifier);
            Expression::Aliased(AliasedExpression {
                token: token.clone(),
                expression: Box::new(item),
                alias: Identifier::new(token, name),
            })
        }
    }
}

/// The name `item_name` gives an alias or a column, as the item spells it;
/// None for an item named as written
pub fn item_spelled_name(expr: &Expression) -> Option<&str> {
    match expr {
        Expression::Aliased(aliased) => Some(aliased.alias.value.as_str()),
        Expression::Identifier(id) => Some(id.value.as_str()),
        Expression::QualifiedIdentifier(qid) => Some(qid.name.value.as_str()),
        _ => None,
    }
}

/// Convert expression to string representation.
/// Used for expression alias matching and display purposes.
pub fn expression_to_string(expr: &Expression) -> String {
    name_text(written_text(expr))
}

/// `text` as a column name: no qualifier separator, which a text value may
/// hold, and still apart from every other text. U+FFFD escapes, so the
/// separator shows as U+FFFD then `1`, and U+FFFD itself shows twice
pub(crate) fn name_text(text: String) -> String {
    const ESCAPE: char = '\u{FFFD}';
    if !text.contains([QUALIFIER_SEP, ESCAPE]) {
        return text;
    }
    let mut name = String::with_capacity(text.len() + 4);
    for c in text.chars() {
        match c {
            QUALIFIER_SEP => name.extend([ESCAPE, '1']),
            ESCAPE => name.extend([ESCAPE, ESCAPE]),
            c => name.push(c),
        }
    }
    name
}

fn written_text(expr: &Expression) -> String {
    match expr {
        // A name as the statement's own text spells it, quoted where it must be
        Expression::Identifier(id) => id.to_string(),
        Expression::QualifiedIdentifier(qid) => qid.to_string(),
        Expression::IntegerLiteral(lit) => lit.value.to_string(),
        Expression::FloatLiteral(lit) => lit.value.to_string(),
        Expression::StringLiteral(lit) => lit.to_string(),
        Expression::BooleanLiteral(lit) => lit.value.to_string(),
        Expression::FunctionCall(func) => written_call(func),
        // An operand bound less tightly than its operator keeps its
        // parentheses, or (a + b) * 2 and a + b * 2 would read the same; a
        // right operand of the same precedence keeps them too unless it is
        // the same associative operator, or a * (b % 3) and a * b % 3 would
        Expression::Infix(infix) => {
            let precedence = infix_precedence(&infix.op_type);
            let side = |operand: &Expression, on_the_right: bool| -> String {
                let text = written_text(operand);
                match operand {
                    Expression::Infix(inner) => {
                        let inner_precedence = infix_precedence(&inner.op_type);
                        // Arithmetic is not associative in floats or at the
                        // limits of an integer, so only these fold
                        let same_associative = inner.op_type == infix.op_type
                            && matches!(
                                infix.op_type,
                                InfixOperator::Or
                                    | InfixOperator::And
                                    | InfixOperator::Xor
                                    | InfixOperator::Concat
                            );
                        if inner_precedence < precedence
                            || (on_the_right && inner_precedence == precedence && !same_associative)
                        {
                            format!("({text})")
                        } else {
                            text
                        }
                    }
                    _ => text,
                }
            };
            format!(
                "{} {} {}",
                side(&infix.left, false),
                infix.operator,
                side(&infix.right, true)
            )
        }
        Expression::Prefix(prefix) => {
            let operand = written_text(&prefix.right);
            let operand = match prefix.right.as_ref() {
                Expression::Infix(_) => format!("({operand})"),
                _ => operand,
            };
            if prefix.operator == "-" || prefix.operator == "+" {
                format!("{}{}", prefix.operator, operand)
            } else {
                format!("{} {}", prefix.operator, operand)
            }
        }
        Expression::Cast(cast) => {
            format!("CAST({} AS {})", written_text(&cast.expr), cast.type_name)
        }
        // The call as an aggregate names it, then OVER as the query wrote it
        Expression::Window(window) => {
            let mut parts: Vec<String> = Vec::new();
            if !window.partition_by.is_empty() {
                let keys: Vec<String> = window.partition_by.iter().map(written_text).collect();
                parts.push(format!("PARTITION BY {}", keys.join(", ")));
            }
            if !window.order_by.is_empty() {
                let keys: Vec<String> = window.order_by.iter().map(order_key_to_string).collect();
                parts.push(format!("ORDER BY {}", keys.join(", ")));
            }
            if let Some(frame) = &window.frame {
                parts.push(frame.to_string());
            }
            let over = match (&window.window_ref, parts.is_empty()) {
                (Some(name), true) => name.to_string(),
                (name, _) => {
                    let name = name.iter().map(|n| n.to_string());
                    format!("({})", name.chain(parts).collect::<Vec<_>>().join(" "))
                }
            };
            format!("{} OVER {}", written_call(&window.function), over)
        }
        _ => format!("{}", expr),
    }
}

/// An ORDER BY key as written: ASC is the default and is left out
fn order_key_to_string(key: &crate::parser::ast::OrderByExpression) -> String {
    let mut text = written_text(&key.expression);
    if !key.ascending {
        text.push_str(" DESC");
    }
    match key.nulls_first {
        Some(true) => text.push_str(" NULLS FIRST"),
        Some(false) => text.push_str(" NULLS LAST"),
        None => {}
    }
    text
}

/// How tightly an infix operator binds, higher first
fn infix_precedence(op: &InfixOperator) -> u8 {
    match op {
        InfixOperator::Or => 1,
        InfixOperator::And | InfixOperator::Xor => 2,
        InfixOperator::Multiply | InfixOperator::Divide | InfixOperator::Modulo => 5,
        InfixOperator::Add | InfixOperator::Subtract | InfixOperator::Concat => 4,
        _ => 3,
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
    extract_join_keys_and_residual_keeping(condition, left_columns, right_columns, &|_, _| true)
}

/// What an index join asks of each pair besides its lookup key: every other
/// conjunct of the ON clause. The sides come in the statement's order, as a
/// bound column reads them, and `key` is the looked-up pair in that order
pub(crate) fn index_join_residual(
    on: &Expression,
    left_columns: &[String],
    right_columns: &[String],
    key: (Option<usize>, Option<usize>),
) -> Option<Expression> {
    let (_, _, residual) =
        extract_join_keys_and_residual_keeping(on, left_columns, right_columns, &|l, r| {
            (Some(l), Some(r)) == key
        });
    combine_predicates_with_and(residual)
}

/// The same extraction, where a column pair that `keep` refuses stays in the
/// residual as an ordinary equality.
pub fn extract_join_keys_and_residual_keeping(
    condition: &Expression,
    left_columns: &[String],
    right_columns: &[String],
    keep: &dyn Fn(usize, usize) -> bool,
) -> (Vec<usize>, Vec<usize>, Vec<Expression>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let mut residual = Vec::new();

    extract_join_keys_recursive(
        condition,
        left_columns,
        right_columns,
        keep,
        &mut left_indices,
        &mut right_indices,
        &mut residual,
    );

    (left_indices, right_indices, residual)
}

/// Whether a join key pair can meet, in any rows, as a TEXT the comparison
/// reads as the other side's type, which no key equality follows, so the
/// pair cannot be a key.
pub fn crossing_key_pair(
    left_rows: &[Row],
    left_idx: usize,
    right_rows: &[Row],
    right_idx: usize,
) -> bool {
    let left = key_kinds(left_rows, left_idx);
    left & (kind::TEXT | kind::READ_FROM_TEXT) != 0
        && kind::crosses(left, key_kinds(right_rows, right_idx))
}

/// The kinds a key column holds: by the schema of a side that reads a base
/// table, else by its rows
pub fn column_kinds(rows: &[Row], columns: &[String], schema: Option<&Schema>, idx: usize) -> u8 {
    let name = columns.get(idx).map(|column| bare_name(column));
    match schema
        .zip(name)
        .and_then(|(schema, name)| schema.get_column_type(name))
    {
        Some(data_type) => kind::of_type(data_type),
        None => key_kinds(rows, idx),
    }
}

/// The kinds a key column holds in any rows, read until a TEXT and a kind a
/// TEXT is read as are both found, which settles every crossing
pub fn key_kinds(rows: &[Row], idx: usize) -> u8 {
    let mut kinds = 0;
    for row in rows {
        kinds |= row.get(idx).map_or(0, kind::of);
        if kinds & kind::TEXT != 0 && kinds & kind::READ_FROM_TEXT != 0 {
            break;
        }
    }
    kinds
}

/// Recursively extract equality join keys from AND expressions.
fn extract_join_keys_recursive(
    condition: &Expression,
    left_columns: &[String],
    right_columns: &[String],
    keep: &dyn Fn(usize, usize) -> bool,
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
                keep,
                left_indices,
                right_indices,
                residual,
            );
            extract_join_keys_recursive(
                &infix.right,
                left_columns,
                right_columns,
                keep,
                left_indices,
                right_indices,
                residual,
            );
        }
        Expression::Infix(infix) if infix.op_type == InfixOperator::Equal => {
            // Columns bound over the joined row, the right side's after the left's
            if let (Expression::Column(a), Expression::Column(b)) = (&*infix.left, &*infix.right) {
                let split = left_columns.len();
                let (a, b) = (a.index as usize, b.index as usize);
                let pair = match (a < split, b < split) {
                    (true, false) => Some((a, b - split)),
                    (false, true) => Some((b, a - split)),
                    _ => None,
                };
                if let Some((left_idx, right_idx)) = pair.filter(|(l, r)| keep(*l, *r)) {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                    return;
                }
            }
            // Extract equality condition
            if let (Some(left_col), Some(right_col)) = (
                extract_column_name_with_qualifier(&infix.left),
                extract_column_name_with_qualifier(&infix.right),
            ) {
                // Case 1: left.col = right.col
                if let (Some(left_idx), Some(right_idx)) = (
                    find_join_side_index(&left_col, left_columns, right_columns),
                    find_join_side_index(&right_col, right_columns, left_columns),
                ) {
                    if keep(left_idx, right_idx) {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                        return;
                    }
                }

                // Case 2: right.col = left.col (swapped)
                if let (Some(left_idx), Some(right_idx)) = (
                    find_join_side_index(&right_col, left_columns, right_columns),
                    find_join_side_index(&left_col, right_columns, left_columns),
                ) {
                    if keep(left_idx, right_idx) {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                        return;
                    }
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

/// A qualified column the other side names exactly belongs to that side, even
/// when this side holds a bare column of the same name, such as a USING column.
fn find_join_side_index(
    col_info: &(Option<String>, String),
    columns: &[String],
    other_columns: &[String],
) -> Option<usize> {
    if let (Some(qualifier), col_name) = col_info {
        let qualified = qualified(qualifier, col_name);
        if other_columns
            .iter()
            .any(|column| column.to_lowercase() == qualified)
        {
            return None;
        }
    }
    find_column_index(col_info, columns)
}

// ============================================================================
// Join Key Equivalence - Column Substitution
// ============================================================================

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
    if let Some((qualifier, name)) = split_qualified(col_name) {
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            qualifier: Identifier::new(
                dummy_token(qualifier, TokenType::Identifier),
                qualifier.to_string(),
            ),
            name: Identifier::new(dummy_token(name, TokenType::Identifier), name.to_string()),
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
            if let Some((_, bare)) = split_qualified(c) {
                if eq_folded(bare, name) {
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
        by_qualifier(
            columns_all()
                .enumerate()
                .filter(|(_, c)| eq_folded_qualified(c, qualifier, name))
                .last()
                .map(|(i, _)| i),
            qualifier,
            name,
            resolve,
            |qualifier| {
                columns_all()
                    .any(|c| split_qualified(c).is_some_and(|(q, _)| eq_folded(q, qualifier)))
            },
            |index| {
                columns_all()
                    .nth(index)
                    .is_some_and(|c| split_qualified(c).is_none())
            },
        )
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

/// What separates a column's qualifier from its own name in the column
/// text the executor binds by. No identifier holds it, so a dot inside a
/// quoted qualifier or name never splits them otherwise. A caller handing
/// column names to the compiler names a qualified column with `qualified`
pub const QUALIFIER_SEP: char = '\u{1F}';

/// The column text of `name` read through the FROM entry `qualifier`
pub fn qualified(qualifier: &str, name: &str) -> String {
    let mut column = String::with_capacity(qualifier.len() + 1 + name.len());
    column.push_str(qualifier);
    column.push(QUALIFIER_SEP);
    column.push_str(name);
    column
}

/// A column text's qualifier and own name; None for a column no FROM
/// entry qualifies, whatever dots its name holds
#[inline]
pub fn split_qualified(column: &str) -> Option<(&str, &str)> {
    column.split_once(QUALIFIER_SEP)
}

/// A column's own name, without the qualifier its text may carry
#[inline]
pub fn bare_name(column: &str) -> &str {
    split_qualified(column).map_or(column, |(_, name)| name)
}

/// The name a column text shows: its qualifier and name joined by a dot
pub fn shown_name(column: &str) -> std::borrow::Cow<'_, str> {
    match column.contains(QUALIFIER_SEP) {
        true => std::borrow::Cow::Owned(column.replace(QUALIFIER_SEP, ".")),
        false => std::borrow::Cow::Borrowed(column),
    }
}

/// The same for a column `qualifier` qualifies, named `name`
#[inline]
pub(crate) fn eq_folded_qualified(column: &str, qualifier: &str, name: &str) -> bool {
    split_qualified(column).is_some_and(|(q, n)| eq_folded(q, qualifier) && eq_folded(n, name))
}

/// Each column a qualified star over `qualifier` takes from `columns`, as
/// its index and where its bare name starts: the columns named
/// `qualifier.name`, the part before the dot lowercased on its own, or,
/// when none is and `table_alias` is the qualifier, every column. The one
/// rule every projection expands `qualifier.*` by
pub(crate) fn for_each_qualified_star_column(
    columns: &[String],
    qualifier: &str,
    table_alias: Option<&str>,
    mut take: impl FnMut(usize, usize),
) {
    let mut found = false;
    for (idx, column) in columns.iter().enumerate() {
        if let Some(start) = qualified_name_start(column, qualifier) {
            take(idx, start);
            found = true;
        }
    }
    if !found && table_alias.is_some_and(|alias| eq_names(alias, qualifier)) {
        (0..columns.len()).for_each(|idx| take(idx, 0));
    }
}

/// Where the bare name of `column` starts when its qualifier is `qualifier`
fn qualified_name_start(column: &str, qualifier: &str) -> Option<usize> {
    let (q, _) = split_qualified(column)?;
    eq_names(q, qualifier).then_some(q.len() + QUALIFIER_SEP.len_utf8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(name: &str) -> Expression {
        Expression::Identifier(Identifier {
            value: name.into(),
            value_lower: name.to_lowercase().into(),
            position: Position::default(),
            quoted: false,
        })
    }

    fn qualified(qualifier: &str, name: &str) -> Expression {
        let (Expression::Identifier(q), Expression::Identifier(n)) =
            (ident(qualifier), ident(name))
        else {
            unreachable!()
        };
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            qualifier: q,
            name: n,
        })
    }

    /// Column names fold case the Unicode way, and when two columns share
    /// a name the later one wins, as the column index map always did.
    #[test]
    fn test_join_projection_folds_unicode_and_prefers_the_later_duplicate() {
        let outer = vec![super::qualified("u", "ÄNAME"), super::qualified("u", "x")];
        let inner = vec![super::qualified("o", "x"), "x".to_string()];

        let proj = compute_join_projection(&[qualified("u", "äname")], &outer, &inner)
            .expect("Unicode-folded name must resolve");
        assert!(matches!(proj.columns[0], ColumnSource::Outer(0)));

        // bare `x`: exact matches only the unqualified inner column
        let proj = compute_join_projection(&[ident("x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));

        // `s.x` is not a full name; the bare fallback finds the exact `x`
        let proj = compute_join_projection(&[qualified("s", "x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));
        // without an exact `x`, two columns qualifying `x` are ambiguous
        let two = vec![super::qualified("u", "x"), super::qualified("o", "x")];
        assert!(compute_join_projection(&[ident("x")], &two, &[]).is_none());

        // duplicate full names: the later one wins
        let dup = vec!["v".to_string(), "v".to_string()];
        let proj = compute_join_projection(&[ident("v")], &dup, &[]).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Outer(1)));
    }

    /// A quoted alias may contain a dot; `"a.b".x` must match the column
    /// `x` of `a.b` and not fall back to a bare `x` elsewhere.
    #[test]
    fn test_join_projection_keeps_dots_inside_a_qualifier() {
        let outer = vec!["x".to_string()];
        let inner = vec![super::qualified("a.b", "id"), super::qualified("a.b", "x")];
        let proj = compute_join_projection(&[qualified("a.b", "x")], &outer, &inner).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Inner(1)));

        let unicode = vec![super::qualified("Ä.b", "x")];
        let proj = compute_join_projection(&[qualified("ä.b", "x")], &unicode, &[]).unwrap();
        assert!(matches!(proj.columns[0], ColumnSource::Outer(0)));
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
    fn test_build_column_index_map_base_names() {
        let column = super::qualified;
        let columns = vec![
            column("T", "Val"),
            column("t", "ID"),
            column("u", "id"),
            column("U", "Name"),
            "name".to_string(),
        ];
        let map = build_column_index_map(&columns);

        assert_eq!(map.get(column("t", "val").as_str()), Some(&0));
        assert_eq!(map.get("val"), Some(&0), "a base one table has");
        assert_eq!(map.get("id"), None, "a base two tables have");
        assert_eq!(map.get("name"), Some(&4), "a column's own name wins");
    }

    #[test]
    fn test_build_column_index_map_empty() {
        let columns: Vec<String> = vec![];
        let map = build_column_index_map(&columns);
        assert!(map.is_empty());
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
            qualifier: Identifier::new(dummy_token("t", TokenType::Identifier), "t".to_string()),
            name: Identifier::new(dummy_token("col", TokenType::Identifier), "col".to_string()),
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
        assert_eq!(
            extract_base_column_name(&super::qualified("table", "column")),
            "column"
        );
        assert_eq!(
            extract_base_column_name(&super::qualified("TABLE", "COLUMN")),
            "column"
        );
    }

    #[test]
    fn test_extract_base_column_name_keeps_the_dots_of_a_name() {
        assert_eq!(extract_base_column_name("a.b.c"), "a.b.c");
        assert_eq!(
            extract_base_column_name(&super::qualified("a.b", "c.d")),
            "c.d"
        );
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
        let columns = vec![super::qualified("t", "id"), super::qualified("t", "name")];
        assert_eq!(
            find_column_index(&(Some("t".to_string()), "id".to_string()), &columns),
            Some(0)
        );
    }

    #[test]
    fn test_find_column_index_suffix_match() {
        let columns = vec![super::qualified("t1", "id"), super::qualified("t1", "name")];
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
            qualifier: Identifier::new(dummy_token("t", TokenType::Identifier), "t".to_string()),
            name: Identifier::new(dummy_token("col", TokenType::Identifier), "col".to_string()),
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
        let expr = create_column_identifier(&super::qualified("t", "col"));
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
            qualifier: Identifier::new(dummy_token("t", TokenType::Identifier), "t".to_string()),
            name: Identifier::new(dummy_token("col", TokenType::Identifier), "col".to_string()),
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
            qualifier: Identifier::new(dummy_token("t", TokenType::Identifier), "t".to_string()),
            name: Identifier::new(dummy_token("col", TokenType::Identifier), "col".to_string()),
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
            qualifier: Identifier::new(dummy_token("t", TokenType::Identifier), "t".to_string()),
            name: Identifier::new(dummy_token("col", TokenType::Identifier), "col".to_string()),
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
}
