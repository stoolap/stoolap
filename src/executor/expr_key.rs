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
use crate::parser::ast::{
    AllAnyType, Expression, FunctionCall, GroupByModifier, Identifier, OrderByExpression,
    SelectStatement,
};

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

    /// The key of `expr` as one answer is of: as [`Self::of`], but a quoted
    /// name as written and a qualified column apart from a dotted name,
    /// which a column lookup or a name read as text tells apart
    pub(crate) fn exact(expr: &Expression) -> Self {
        let mut hasher = KeyHasher::new();
        hasher.exact = true;
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
    /// Names keep their quoting and qualification, see [`ExprKey::exact`]
    exact: bool,
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
            exact: false,
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

/// One part of a qualified name under [`ExprKey::exact`]
fn write_identifier(id: &Identifier, hasher: &mut KeyHasher) {
    match id.quoted {
        true => hasher.text(&id.value),
        false => hasher.name(&id.value_lower),
    }
    hasher.flag(id.quoted);
}

fn write_expr(expr: &Expression, hasher: &mut KeyHasher) {
    match expr {
        Expression::Identifier(id) if hasher.exact && id.quoted => {
            hasher.tag(b'q');
            hasher.text(&id.value);
        }
        Expression::Identifier(id) => {
            hasher.tag(b'I');
            hasher.name(&id.value_lower);
        }
        Expression::QualifiedIdentifier(qid) if hasher.exact => {
            hasher.tag(b'Q');
            write_identifier(&qid.qualifier, hasher);
            write_identifier(&qid.name, hasher);
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
        // By slot: two `?` are two values, `$1` and a repeated name one
        Expression::Parameter(param) => {
            hasher.tag(b'P');
            hasher.write_usize(param.index);
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
        Expression::ScalarSubquery(sq) => {
            hasher.tag(b'k');
            write_statement(&sq.subquery, hasher);
        }
        Expression::Exists(exists) => {
            hasher.tag(b'e');
            write_statement(&exists.subquery, hasher);
        }
        Expression::AllAny(all_any) => {
            hasher.tag(b'y');
            hasher.name(&all_any.operator);
            hasher.flag(matches!(all_any.all_any_type, AllAnyType::All));
            write_expr(&all_any.left, hasher);
            write_statement(&all_any.subquery, hasher);
        }
        // Table sources read the same when they print the same
        other => {
            hasher.tag(b'?');
            hasher.text(&other.to_string());
        }
    }
}

/// A subquery: a prepared block by its id, as every mark prints alike, a
/// written one by its parts, so each parameter in it reads by its slot
fn write_statement(stmt: &SelectStatement, hasher: &mut KeyHasher) {
    hasher.flag(stmt.prepared.is_some());
    if let Some(id) = stmt.prepared {
        id.hash(hasher);
        return;
    }
    hasher.flag(stmt.distinct);
    write_list(&stmt.distinct_on, hasher);
    // A column's alias names what the statement around it reads
    hasher.write_usize(stmt.columns.len());
    for column in &stmt.columns {
        match column {
            Expression::Aliased(aliased) => {
                hasher.tag(b'a');
                // A name as the resolver reads it, which per letter lowering keeps
                hasher.name(&aliased.alias.value_lower);
                write_expr(&aliased.expression, hasher);
            }
            column => write_expr(column, hasher),
        }
    }
    hasher.flag(stmt.with.is_some());
    if let Some(with) = &stmt.with {
        hasher.flag(with.is_recursive);
        hasher.write_usize(with.ctes.len());
        for cte in &with.ctes {
            hasher.name(&cte.name.value_lower);
            hasher.write_usize(cte.column_names.len());
            for column in &cte.column_names {
                hasher.name(&column.value_lower);
            }
            write_statement(&cte.query, hasher);
        }
    }
    hasher.flag(stmt.table_expr.is_some());
    if let Some(source) = stmt.table_expr.as_deref() {
        write_source(source, hasher);
    }
    write_optional(stmt.where_clause.as_deref(), hasher);
    write_list(&stmt.group_by.columns, hasher);
    match &stmt.group_by.modifier {
        GroupByModifier::None => hasher.tag(b'n'),
        GroupByModifier::Rollup => hasher.tag(b'r'),
        GroupByModifier::Cube => hasher.tag(b'c'),
        GroupByModifier::GroupingSets(sets) => {
            hasher.tag(b'g');
            hasher.write_usize(sets.len());
            for set in sets {
                write_list(set, hasher);
            }
        }
    }
    write_optional(stmt.having.as_deref(), hasher);
    hasher.write_usize(stmt.window_defs.len());
    for window in &stmt.window_defs {
        hasher.name(&window.name);
        write_list(&window.partition_by, hasher);
        write_order_by(&window.order_by, hasher);
        hasher.flag(window.frame.is_some());
        if let Some(frame) = &window.frame {
            hasher.text(&frame.to_string());
        }
    }
    write_order_by(&stmt.order_by, hasher);
    write_optional(stmt.limit.as_deref(), hasher);
    write_optional(stmt.offset.as_deref(), hasher);
    hasher.write_usize(stmt.set_operations.len());
    for operation in &stmt.set_operations {
        hasher.text(&operation.operation.to_string());
        write_statement(&operation.right, hasher);
    }
}

/// A source's alias and the names of its columns
fn write_names(alias: Option<&Identifier>, columns: &[Identifier], hasher: &mut KeyHasher) {
    hasher.flag(alias.is_some());
    if let Some(alias) = alias {
        hasher.name(&alias.value_lower);
    }
    hasher.write_usize(columns.len());
    for column in columns {
        hasher.name(&column.value_lower);
    }
}

/// A FROM source of a subquery, each expression in it by its parts
fn write_source(source: &Expression, hasher: &mut KeyHasher) {
    match source {
        Expression::SubquerySource(sub) => {
            hasher.tag(b's');
            hasher.flag(sub.alias.is_some());
            if let Some(alias) = &sub.alias {
                hasher.name(&alias.value_lower);
            }
            write_statement(&sub.subquery, hasher);
        }
        Expression::JoinSource(join) => {
            hasher.tag(b'j');
            write_source(&join.left, hasher);
            hasher.name(&join.join_type);
            write_source(&join.right, hasher);
            write_optional(join.condition.as_deref(), hasher);
            hasher.write_usize(join.using_columns.len());
            for column in &join.using_columns {
                hasher.name(&column.value_lower);
            }
        }
        Expression::ValuesSource(values) => {
            hasher.tag(b'v');
            hasher.write_usize(values.rows.len());
            for row in &values.rows {
                write_list(row, hasher);
            }
            write_names(values.alias.as_ref(), &values.column_aliases, hasher);
        }
        Expression::FunctionTableSource(function) => {
            hasher.tag(b'f');
            hasher.name(&function.function.value);
            write_list(&function.arguments, hasher);
            write_names(function.alias.as_ref(), &function.column_aliases, hasher);
        }
        Expression::TableSource(table) => {
            hasher.tag(b't');
            write_source_name(&table.name, table.alias.as_ref(), hasher);
            hasher.flag(table.as_of.is_some());
            if let Some(as_of) = &table.as_of {
                hasher.name(&as_of.as_of_type);
                write_expr(&as_of.value, hasher);
            }
        }
        Expression::CteReference(cte) => {
            hasher.tag(b'r');
            write_source_name(&cte.name, cte.alias.as_ref(), hasher);
        }
        other => write_expr(other, hasher),
    }
}

/// A source's name and its alias, each a name of its own
fn write_source_name(name: &Identifier, alias: Option<&Identifier>, hasher: &mut KeyHasher) {
    let write = |id: &Identifier, hasher: &mut KeyHasher| match hasher.exact {
        true => write_identifier(id, hasher),
        false => hasher.name(&id.value_lower),
    };
    write(name, hasher);
    hasher.flag(alias.is_some());
    if let Some(alias) = alias {
        write(alias, hasher);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Statement;
    use crate::parser::parse_sql;

    fn select_key(sql: &str) -> ExprKey {
        let statements = parse_sql(&format!("SELECT {sql} FROM t, a")).unwrap();
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
        assert_eq!(ExprKey::column("a.Σ"), select_key("a.\"Σ\""));
        // A name as the resolver lowers it, a final sigma to its own letter
        assert_ne!(select_key("\"ΟΣ\""), select_key("\"οσ\""));
        assert_ne!(
            select_key("(SELECT v AS \"ΟΣ\" FROM u)"),
            select_key("(SELECT v AS \"οσ\" FROM u)")
        );
        // A qualified column's parts lowered each as the resolver lowers them
        assert_ne!(
            select_key("SUM(a.\"Σ\" * 2)"),
            select_key("SUM(a.\"ς\" * 2)")
        );
    }

    #[test]
    fn test_an_exact_key_keeps_quoting_and_qualification() {
        let exact = |sql: &str| {
            let statements = parse_sql(&format!("SELECT {sql} FROM t, a")).unwrap();
            match &statements[0] {
                Statement::Select(select) => ExprKey::exact(&select.columns[0]),
                other => panic!("not a SELECT: {other:?}"),
            }
        };
        for (a, b) in [
            ("\"a.b\"", "a.b"),
            ("\"UPPER\"", "\"upper\""),
            ("\"v\"", "v"),
        ] {
            assert_eq!(select_key(a), select_key(b), "{a} and {b} name one column");
            assert_ne!(exact(a), exact(b), "{a} and {b} may read two values");
        }
        for (a, b) in [("t.V + 1", "T.v + 1"), ("UPPER(v)", "upper(V)")] {
            assert_eq!(exact(a), exact(b), "{a} and {b} are one expression");
        }
        // A source's name and alias are parts of their own, not its text
        let (quoted, aliased) = ("(SELECT v FROM \"u AS a\")", "(SELECT v FROM u AS a)");
        assert_ne!(select_key(quoted), select_key(aliased));
        assert_ne!(exact(quoted), exact(aliased));
        let (upper, lower) = ("(SELECT v FROM U AS A)", "(SELECT v FROM u AS a)");
        assert_eq!(select_key(upper), select_key(lower), "one table");
        assert_eq!(exact(upper), exact(lower), "one table");
    }

    #[test]
    fn test_a_prepared_block_reads_as_its_id() {
        let marked = |sql: &str, id: u16| {
            let statements = parse_sql(&format!("SELECT {sql} FROM t, a")).unwrap();
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

    #[test]
    fn test_parameters_and_subqueries_read_by_what_they_hold() {
        let pair = |sql: &str| {
            let statements = parse_sql(&format!("SELECT {sql} FROM t, a")).unwrap();
            let Statement::Select(select) = &statements[0] else {
                panic!("not a SELECT");
            };
            (
                ExprKey::of(&select.columns[0]),
                ExprKey::of(&select.columns[1]),
            )
        };
        for sql in [
            "a + $1, a + $1",
            "a + :n, a + :n",
            "(SELECT p FROM (SELECT x AS p FROM u) s), (SELECT p FROM (SELECT x AS p FROM u) s)",
        ] {
            let (a, b) = pair(sql);
            assert_eq!(a, b, "{sql}: one expression");
        }
        for sql in [
            "a + ?, a + ?",
            "a + $1, a + $2",
            "(SELECT ?), (SELECT ?)",
            "(SELECT 1 FROM u WHERE v = 1), (SELECT 1 FROM u WHERE v = 2)",
            "(SELECT p FROM (SELECT x AS p, y AS q FROM u) s), \
             (SELECT p FROM (SELECT y AS p, x AS q FROM u) s)",
        ] {
            let (a, b) = pair(sql);
            assert_ne!(a, b, "{sql}: two expressions");
        }
    }
}
