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

//! The identity of an expression, hashed from its tree: names match in any
//! case, literals only as written, and tokens and positions are ignored.

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use rustc_hash::FxHasher;

use crate::core::Value;
use crate::parser::ast::{AllAnyType, Expression, FunctionCall, OrderByExpression};

/// Two expressions share a key when they are the same computation. The key
/// is two independent 64-bit hashes of the tree, so two different trees
/// meet only with a chance near 2^-128
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExprKey(u64, u64);

impl ExprKey {
    pub(crate) fn of(expr: &Expression) -> Self {
        let mut hasher = KeyHasher::new();
        write_expr(expr, &mut hasher);
        hasher.finish()
    }

    pub(crate) fn of_call(call: &FunctionCall) -> Self {
        let mut hasher = KeyHasher::new();
        write_call(call, &mut hasher);
        hasher.finish()
    }

    /// The key of a column reference by `name`, dotted when qualified
    pub(crate) fn column(name: &str) -> Self {
        let mut hasher = KeyHasher::new();
        hasher.tag(b'I');
        hasher.name(name);
        hasher.finish()
    }

    /// The name of a column carried only for ORDER BY and trimmed after the sort
    pub(crate) fn hidden_column_name(&self) -> String {
        format!("\u{0}{:016x}{:016x}", self.0, self.1)
    }

    /// The key a hidden column was named after, if `name` is one
    pub(crate) fn from_hidden_column_name(name: &str) -> Option<Self> {
        let hex = name.strip_prefix('\u{0}')?;
        if hex.len() != 32 {
            return None;
        }
        let first = u64::from_str_radix(&hex[..16], 16).ok()?;
        let second = u64::from_str_radix(&hex[16..], 16).ok()?;
        Some(ExprKey(first, second))
    }
}

/// Feeds every write to two hashers of different families
struct KeyHasher {
    fx: FxHasher,
    a: AHasher,
}

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.fx.write(bytes);
        self.a.write(bytes);
    }

    fn finish(&self) -> u64 {
        self.fx.finish()
    }
}

impl KeyHasher {
    fn new() -> Self {
        KeyHasher {
            fx: FxHasher::default(),
            a: AHasher::default(),
        }
    }

    fn finish(self) -> ExprKey {
        ExprKey(Hasher::finish(&self.fx), Hasher::finish(&self.a))
    }

    /// The node kind, so no two kinds read alike
    fn tag(&mut self, tag: u8) {
        self.write_u8(tag);
    }

    /// Text compared as written, its length first so nothing runs into it
    fn text(&mut self, text: &str) {
        self.write_usize(text.len());
        self.write(text.as_bytes());
    }

    /// A name, whatever its case
    fn name(&mut self, name: &str) {
        for c in name.chars().flat_map(char::to_lowercase) {
            self.write_u32(c as u32);
        }
        self.write_u32(u32::MAX);
    }

    fn flag(&mut self, flag: bool) {
        self.write_u8(flag as u8);
    }
}

fn write_list(items: &[Expression], hasher: &mut KeyHasher) {
    hasher.write_usize(items.len());
    for item in items {
        write_expr(item, hasher);
    }
}

fn write_order_by(items: &[OrderByExpression], hasher: &mut KeyHasher) {
    hasher.write_usize(items.len());
    for item in items {
        write_expr(&item.expression, hasher);
        hasher.flag(item.ascending);
        item.nulls_first.hash(hasher);
    }
}

fn write_optional(expr: Option<&Expression>, hasher: &mut KeyHasher) {
    match expr {
        Some(expr) => {
            hasher.flag(true);
            write_expr(expr, hasher);
        }
        None => hasher.flag(false),
    }
}

fn write_call(call: &FunctionCall, hasher: &mut KeyHasher) {
    hasher.tag(b'G');
    hasher.name(&call.function);
    hasher.flag(call.is_distinct);
    write_list(&call.arguments, hasher);
    write_order_by(&call.order_by, hasher);
    write_optional(call.filter.as_deref(), hasher);
}

fn write_value(value: &Value, hasher: &mut KeyHasher) {
    match value {
        Value::Null(_) => hasher.tag(b'Z'),
        Value::Integer(i) => {
            hasher.tag(b'N');
            hasher.write_i64(*i);
        }
        Value::Float(f) => {
            hasher.tag(b'R');
            hasher.write_u64(f.to_bits());
        }
        Value::Text(s) => {
            hasher.tag(b'S');
            hasher.text(s);
        }
        Value::Boolean(b) => {
            hasher.tag(b'B');
            hasher.flag(*b);
        }
        Value::Timestamp(ts) => {
            hasher.tag(b'D');
            ts.hash(hasher);
        }
        Value::Extension(data) => {
            hasher.tag(b'E');
            hasher.write_usize(data.len());
            hasher.write(data);
        }
    }
}

fn write_expr(expr: &Expression, hasher: &mut KeyHasher) {
    match expr {
        Expression::Identifier(id) => {
            hasher.tag(b'I');
            hasher.name(&id.value_lower);
        }
        // A GROUP BY keeps a qualified column as its dotted name
        Expression::QualifiedIdentifier(qid) => {
            hasher.tag(b'I');
            for c in qid
                .qualifier
                .value_lower
                .chars()
                .chain(std::iter::once('.'))
                .chain(qid.name.value_lower.chars())
                .flat_map(char::to_lowercase)
            {
                hasher.write_u32(c as u32);
            }
            hasher.write_u32(u32::MAX);
        }
        Expression::IntegerLiteral(lit) => {
            hasher.tag(b'N');
            hasher.write_i64(lit.value);
        }
        Expression::FloatLiteral(lit) => {
            hasher.tag(b'R');
            hasher.write_u64(lit.value.to_bits());
        }
        Expression::StringLiteral(lit) => {
            hasher.tag(b'S');
            hasher.text(&lit.value);
            match &lit.type_hint {
                Some(hint) => {
                    hasher.flag(true);
                    hasher.name(hint);
                }
                None => hasher.flag(false),
            }
        }
        Expression::BooleanLiteral(lit) => {
            hasher.tag(b'B');
            hasher.flag(lit.value);
        }
        Expression::NullLiteral(_) => hasher.tag(b'Z'),
        Expression::IntervalLiteral(lit) => {
            hasher.tag(b'V');
            hasher.write_i64(lit.quantity);
            hasher.name(&lit.unit);
        }
        // By name: every `?` reads alike, as a bound value is the same where it repeats
        Expression::Parameter(param) => {
            hasher.tag(b'P');
            hasher.text(&param.name);
        }
        Expression::Prefix(prefix) => {
            hasher.tag(b'u');
            hasher.name(&prefix.operator);
            write_expr(&prefix.right, hasher);
        }
        Expression::Infix(infix) => {
            hasher.tag(b'x');
            hasher.name(&infix.operator);
            write_expr(&infix.left, hasher);
            write_expr(&infix.right, hasher);
        }
        Expression::List(list) => {
            hasher.tag(b'L');
            write_list(&list.elements, hasher);
        }
        Expression::ExpressionList(list) => {
            hasher.tag(b'L');
            write_list(&list.expressions, hasher);
        }
        Expression::Distinct(distinct) => {
            hasher.tag(b'd');
            write_expr(&distinct.expr, hasher);
        }
        Expression::In(in_expr) => {
            hasher.tag(b'i');
            hasher.flag(in_expr.not);
            write_expr(&in_expr.left, hasher);
            write_expr(&in_expr.right, hasher);
        }
        // A set built at run time is itself only: two builds never share a key
        Expression::InHashSet(set) => {
            hasher.tag(b'h');
            hasher.flag(set.not);
            hasher.write_usize(crate::common::CompactArc::as_ptr(&set.values) as usize);
            write_expr(&set.column, hasher);
        }
        Expression::QuantifiedSet(set) => {
            hasher.tag(b'A');
            hasher.text(&set.operator);
            hasher.flag(set.all);
            hasher.flag(set.has_null);
            write_expr(&set.left, hasher);
            hasher.write_usize(set.members.len());
            for member in set.members.iter() {
                write_value(member, hasher);
            }
        }
        Expression::Between(between) => {
            hasher.tag(b'b');
            hasher.flag(between.not);
            write_expr(&between.expr, hasher);
            write_expr(&between.lower, hasher);
            write_expr(&between.upper, hasher);
        }
        Expression::Like(like) => {
            hasher.tag(b'K');
            hasher.name(&like.operator);
            write_expr(&like.left, hasher);
            write_expr(&like.pattern, hasher);
            write_optional(like.escape.as_deref(), hasher);
        }
        Expression::Case(case) => {
            hasher.tag(b'C');
            write_optional(case.value.as_deref(), hasher);
            hasher.write_usize(case.when_clauses.len());
            for when in &case.when_clauses {
                write_expr(&when.condition, hasher);
                write_expr(&when.then_result, hasher);
            }
            write_optional(case.else_value.as_deref(), hasher);
        }
        Expression::Cast(cast) => {
            hasher.tag(b'c');
            hasher.name(&cast.type_name);
            write_expr(&cast.expr, hasher);
        }
        Expression::FunctionCall(call) => write_call(call, hasher),
        Expression::Aliased(aliased) => write_expr(&aliased.expression, hasher),
        Expression::Window(window) => {
            hasher.tag(b'O');
            write_call(&window.function, hasher);
            write_list(&window.partition_by, hasher);
            write_order_by(&window.order_by, hasher);
            match &window.frame {
                Some(frame) => {
                    hasher.flag(true);
                    hasher.text(&frame.to_string());
                }
                None => hasher.flag(false),
            }
            match &window.window_ref {
                Some(name) => {
                    hasher.flag(true);
                    hasher.name(name);
                }
                None => hasher.flag(false),
            }
        }
        Expression::Star(_) => hasher.tag(b'*'),
        Expression::QualifiedStar(star) => {
            hasher.tag(b'q');
            hasher.name(&star.qualifier);
        }
        // A prepared block is its id, as every mark prints alike
        Expression::ScalarSubquery(sq) if sq.subquery.prepared.is_some() => {
            hasher.tag(b'k');
            sq.subquery.prepared.hash(hasher);
        }
        Expression::Exists(exists) if exists.subquery.prepared.is_some() => {
            hasher.tag(b'e');
            exists.subquery.prepared.hash(hasher);
        }
        Expression::AllAny(all_any) if all_any.subquery.prepared.is_some() => {
            hasher.tag(b'y');
            hasher.name(&all_any.operator);
            hasher.flag(matches!(all_any.all_any_type, AllAnyType::All));
            write_expr(&all_any.left, hasher);
            all_any.subquery.prepared.hash(hasher);
        }
        // Subqueries and table sources read the same when they print the same
        other => {
            hasher.tag(b'?');
            hasher.text(&other.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Statement;
    use crate::parser::parse_sql;

    fn select_key(sql: &str) -> ExprKey {
        let statements = parse_sql(&format!("SELECT {sql}")).unwrap();
        match &statements[0] {
            Statement::Select(select) => ExprKey::of(&select.columns[0]),
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    #[test]
    fn test_names_match_in_any_case_and_literals_only_as_written() {
        for (a, b) in [
            ("SUM(V)", "sum(v)"),
            ("t.A + 1", "T.a + 1"),
            ("\"a'B\" + 1", "\"a'b\" + 1"),
            ("v || 'a' AS x", "v || 'a'"),
            (
                "COUNT(*) FILTER (WHERE a > 1)",
                "count(*) filter (where A > 1)",
            ),
        ] {
            assert_eq!(
                select_key(a),
                select_key(b),
                "{a} and {b} are one expression"
            );
        }
        for (a, b) in [
            ("v || 'a'", "v || 'A'"),
            ("STRING_AGG(v, ',')", "STRING_AGG(v, ';')"),
            ("SUM(a + 1)", "SUM(b * 2)"),
            ("SUM(DISTINCT a)", "SUM(a)"),
            ("COUNT(*) FILTER (WHERE a > 1)", "COUNT(*)"),
            ("1", "1.0"),
            ("'1'", "1"),
            ("a - b", "a + b"),
            ("f(a, b)", "f(ab)"),
        ] {
            assert_ne!(
                select_key(a),
                select_key(b),
                "{a} and {b} are two expressions"
            );
        }
    }

    #[test]
    fn test_a_column_key_is_the_identifier_key() {
        assert_eq!(ExprKey::column("A'B"), select_key("\"a'b\""));
        assert_eq!(ExprKey::column("T.v"), select_key("t.V"));
    }

    #[test]
    fn test_a_prepared_block_reads_as_its_id() {
        let marked = |sql: &str, id: u16| {
            let statements = parse_sql(&format!("SELECT {sql}")).unwrap();
            let Statement::Select(select) = &statements[0] else {
                panic!("not a SELECT");
            };
            let mut expr = select.columns[0].clone();
            match &mut expr {
                Expression::ScalarSubquery(sq) => sq.subquery.prepared = Some(id),
                Expression::Exists(exists) => exists.subquery.prepared = Some(id),
                Expression::AllAny(all_any) => all_any.subquery.prepared = Some(id),
                other => panic!("not a subquery: {other:?}"),
            }
            ExprKey::of(&expr)
        };
        for sql in [
            "(SELECT 1 FROM u)",
            "EXISTS (SELECT 1 FROM u)",
            "a > ANY (SELECT x FROM u)",
        ] {
            assert_ne!(marked(sql, 0), marked(sql, 1), "{sql}: two blocks");
            assert_eq!(marked(sql, 1), marked(sql, 1), "{sql}: one block");
        }
        assert_ne!(
            marked("a > ANY (SELECT x FROM u)", 1),
            marked("b > ANY (SELECT x FROM u)", 1),
            "the left side reads too"
        );
    }
}
