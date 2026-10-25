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

//! The rule a qualifier is read by: `q.column` and `q.*` name a FROM entry
//! of the query they are written in or of an enclosing one

use smallvec::SmallVec;

use super::ast::*;
use super::token::Position;

/// The message for a qualifier no FROM entry anywhere names
pub(crate) fn unknown_qualifier(qualifier: &str) -> String {
    format!("unknown table or alias '{qualifier}'")
}

/// A qualifier the statement names where no FROM entry in scope does
pub(crate) struct ScopeError {
    pub message: String,
    pub position: Position,
}

type Check = Result<(), ScopeError>;

/// The names one query level makes visible, and the names it holds but
/// does not show there: a table its alias hides, or every name of the
/// query a FROM entry stands `beside`
#[derive(Default)]
struct Scope<'a, 'p> {
    visible: SmallVec<[&'a str; 4]>,
    hidden: SmallVec<[&'a str; 4]>,
    beside: Option<&'p Scope<'a, 'p>>,
    parent: Option<&'p Scope<'a, 'p>>,
}

impl<'a, 'p> Scope<'a, 'p> {
    fn under(parent: Option<&'p Scope<'a, 'p>>) -> Self {
        Self {
            parent,
            ..Self::default()
        }
    }

    /// A level that shows none of `query`'s names: what a derived table,
    /// a VALUES list or a table function's arguments see beside it
    fn beside(query: &'p Scope<'a, 'p>, parent: Option<&'p Scope<'a, 'p>>) -> Self {
        Self {
            beside: Some(query),
            parent,
            ..Self::default()
        }
    }

    fn holds(&self, name: &str) -> bool {
        self.visible.contains(&name) || self.hidden.contains(&name)
    }
}

/// Every qualifier `stmt` writes names a FROM entry visible where it is written
pub(crate) fn check(stmt: &Statement) -> Check {
    match stmt {
        Statement::Select(select) => check_select(select, None),
        Statement::Insert(insert) => check_insert(insert),
        Statement::Update(update) => check_update(update),
        Statement::Delete(delete) => check_delete(delete),
        Statement::CreateTable(create) => check_create_table(create),
        Statement::AlterTable(alter) => match &alter.column_def {
            Some(def) => check_column(def, &alter.table_name),
            None => Ok(()),
        },
        Statement::CreateView(view) => check_select(&view.query, None),
        Statement::Explain(explain) => check(&explain.statement),
        Statement::Expression(expr) => check_expr(&expr.expression, None),
        Statement::Set(set) => check_expr(&set.value, None),
        Statement::Pragma(pragma) => check_opt(pragma.value.as_ref(), None),
        _ => Ok(()),
    }
}

fn check_select<'a>(select: &'a SelectStatement, outer: Option<&Scope<'a, '_>>) -> Check {
    if let Some(with) = &select.with {
        for cte in &with.ctes {
            check_select(&cte.query, outer)?;
        }
    }
    let mut own = Scope::under(outer);
    if let Some(from) = &select.table_expr {
        from_names(from, &mut own);
        check_source(from, &own, outer)?;
    }
    let own = Some(&own);
    for expr in select.distinct_on.iter().chain(&select.columns) {
        check_expr(expr, own)?;
    }
    check_opt(select.where_clause.as_deref(), own)?;
    for expr in &select.group_by.columns {
        check_expr(expr, own)?;
    }
    if let GroupByModifier::GroupingSets(sets) = &select.group_by.modifier {
        for expr in sets.iter().flatten() {
            check_expr(expr, own)?;
        }
    }
    check_opt(select.having.as_deref(), own)?;
    for def in &select.window_defs {
        check_window(&def.partition_by, &def.order_by, def.frame.as_ref(), own)?;
    }
    // A set operation's ORDER BY reads the result's columns, not a branch's FROM
    let ordered = if select.set_operations.is_empty() {
        own
    } else {
        outer
    };
    for order in &select.order_by {
        check_expr(&order.expression, ordered)?;
    }
    check_opt(select.limit.as_deref(), outer)?;
    check_opt(select.offset.as_deref(), outer)?;
    for op in &select.set_operations {
        check_select(&op.right, outer)?;
    }
    Ok(())
}

/// The names a FROM tree shows: each entry's alias, else its own name
fn from_names<'a>(source: &'a Expression, scope: &mut Scope<'a, '_>) {
    visit_from_names(source, &mut |shown, hidden| {
        scope.visible.push(shown);
        scope.hidden.extend(hidden);
    });
}

/// Whether a FROM tree shows `name`, lowercased, as one of its entries: a
/// qualifier it does not show names an enclosing query's entry
pub(crate) fn from_shows(source: &Expression, name: &str) -> bool {
    let mut shows = false;
    visit_from_names(source, &mut |shown, _| shows |= shown == name);
    shows
}

/// Each name a FROM tree shows, lowercased, with the table name its alias hides
fn visit_from_names<'a>(source: &'a Expression, visit: &mut dyn FnMut(&'a str, Option<&'a str>)) {
    let (name, alias) = match source {
        Expression::TableSource(t) => (Some(&t.name), t.alias.as_ref()),
        Expression::CteReference(c) => (Some(&c.name), c.alias.as_ref()),
        Expression::FunctionTableSource(f) => (Some(&f.function), f.alias.as_ref()),
        Expression::SubquerySource(s) => (None, s.alias.as_ref()),
        Expression::ValuesSource(v) => (None, v.alias.as_ref()),
        Expression::JoinSource(j) => {
            visit_from_names(&j.left, visit);
            visit_from_names(&j.right, visit);
            return;
        }
        _ => return,
    };
    match (alias, name) {
        (Some(alias), name) => visit(&alias.value_lower, name.map(|n| n.value_lower.as_str())),
        (None, Some(name)) => visit(&name.value_lower, None),
        (None, None) => {}
    }
}

/// The expressions inside a FROM tree: a join's ON sees that join's entries,
/// a derived table, VALUES list or table function sees only enclosing levels
fn check_source<'a>(
    source: &'a Expression,
    query: &Scope<'a, '_>,
    outer: Option<&Scope<'a, '_>>,
) -> Check {
    match source {
        Expression::TableSource(t) => match &t.as_of {
            Some(as_of) => check_expr(&as_of.value, outer),
            None => Ok(()),
        },
        Expression::JoinSource(j) => {
            check_source(&j.left, query, outer)?;
            check_source(&j.right, query, outer)?;
            match &j.condition {
                Some(condition) => {
                    let mut joined = Scope::under(None);
                    from_names(source, &mut joined);
                    let mut on = Scope::beside(query, outer);
                    on.visible = joined.visible;
                    check_expr(condition, Some(&on))
                }
                None => Ok(()),
            }
        }
        Expression::SubquerySource(s) => {
            check_select(&s.subquery, Some(&Scope::beside(query, outer)))
        }
        Expression::ValuesSource(v) => {
            let beside = Scope::beside(query, outer);
            for expr in v.rows.iter().flatten() {
                check_expr(expr, Some(&beside))?;
            }
            Ok(())
        }
        Expression::FunctionTableSource(f) => {
            let beside = Scope::beside(query, outer);
            for expr in &f.arguments {
                check_expr(expr, Some(&beside))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_insert(insert: &InsertStatement) -> Check {
    // The target is not a FROM entry of the rows an INSERT writes
    let mut target = Scope::under(None);
    target.hidden.push(&insert.table_name.value_lower);
    let target = Some(&target);
    for expr in insert.values.iter().flatten() {
        check_expr(expr, target)?;
    }
    if let Some(select) = &insert.select {
        check_select(select, target)?;
    }
    let mut conflict = Scope::under(None);
    conflict.visible.push(&insert.table_name.value_lower);
    conflict.visible.push("excluded");
    for expr in &insert.update_expressions {
        check_expr(expr, Some(&conflict))?;
    }
    check_opt(insert.update_where.as_deref(), Some(&conflict))?;
    let mut returning = Scope::under(None);
    returning.visible.push(&insert.table_name.value_lower);
    returning.hidden.push("excluded");
    for expr in &insert.returning {
        check_expr(expr, Some(&returning))?;
    }
    Ok(())
}

fn check_update(update: &UpdateStatement) -> Check {
    check_with(update.with.as_ref())?;
    let mut target = Scope::under(None);
    target.visible.push(&update.table_name.value_lower);
    let target = Some(&target);
    for expr in update.updates.values() {
        check_expr(expr, target)?;
    }
    check_opt(update.where_clause.as_deref(), target)?;
    for expr in &update.returning {
        check_expr(expr, target)?;
    }
    Ok(())
}

fn check_delete(delete: &DeleteStatement) -> Check {
    check_with(delete.with.as_ref())?;
    let mut target = Scope::under(None);
    match &delete.alias {
        Some(alias) => {
            target.visible.push(&alias.value_lower);
            target.hidden.push(&delete.table_name.value_lower);
        }
        None => target.visible.push(&delete.table_name.value_lower),
    }
    let target = Some(&target);
    check_opt(delete.where_clause.as_deref(), target)?;
    for expr in &delete.returning {
        check_expr(expr, target)?;
    }
    Ok(())
}

fn check_with(with: Option<&WithClause>) -> Check {
    for cte in with.iter().flat_map(|w| &w.ctes) {
        check_select(&cte.query, None)?;
    }
    Ok(())
}

fn check_create_table(create: &CreateTableStatement) -> Check {
    for def in &create.columns {
        check_column(def, &create.table_name)?;
    }
    let mut table = Scope::under(None);
    table.visible.push(&create.table_name.value_lower);
    for constraint in &create.table_constraints {
        if let TableConstraint::Check(expr) = constraint {
            check_expr(expr, Some(&table))?;
        }
    }
    match &create.as_select {
        Some(select) => check_select(select, None),
        None => Ok(()),
    }
}

/// A CHECK reads the table's row; a DEFAULT reads no row
fn check_column(def: &ColumnDefinition, table_name: &Identifier) -> Check {
    let mut table = Scope::under(None);
    table.visible.push(&table_name.value_lower);
    for constraint in &def.constraints {
        match constraint {
            ColumnConstraint::Check(expr) => check_expr(expr, Some(&table))?,
            ColumnConstraint::Default(expr) => check_expr(expr, None)?,
            _ => {}
        }
    }
    Ok(())
}

fn check_opt<'a>(expr: Option<&'a Expression>, scope: Option<&Scope<'a, '_>>) -> Check {
    match expr {
        Some(expr) => check_expr(expr, scope),
        None => Ok(()),
    }
}

fn check_window<'a>(
    partition_by: &'a [Expression],
    order_by: &'a [OrderByExpression],
    frame: Option<&'a WindowFrame>,
    scope: Option<&Scope<'a, '_>>,
) -> Check {
    for expr in partition_by {
        check_expr(expr, scope)?;
    }
    for order in order_by {
        check_expr(&order.expression, scope)?;
    }
    if let Some(frame) = frame {
        for bound in std::iter::once(&frame.start).chain(frame.end.as_ref()) {
            if let WindowFrameBound::Preceding(expr) | WindowFrameBound::Following(expr) = bound {
                check_expr(expr, scope)?;
            }
        }
    }
    Ok(())
}

fn check_function<'a>(call: &'a FunctionCall, scope: Option<&Scope<'a, '_>>) -> Check {
    for expr in &call.arguments {
        check_expr(expr, scope)?;
    }
    for order in &call.order_by {
        check_expr(&order.expression, scope)?;
    }
    check_opt(call.filter.as_deref(), scope)
}

fn check_expr<'a>(expr: &'a Expression, scope: Option<&Scope<'a, '_>>) -> Check {
    match expr {
        Expression::QualifiedIdentifier(q) => qualifier(
            &q.qualifier.value,
            &q.qualifier.value_lower,
            q.qualifier.position,
            scope,
        ),
        Expression::QualifiedStar(star) => {
            let lower = star.qualifier.to_lowercase();
            qualifier(&star.qualifier, &lower, star.token.position, scope)
        }
        Expression::Prefix(p) => check_expr(&p.right, scope),
        Expression::Infix(i) => {
            check_expr(&i.left, scope)?;
            check_expr(&i.right, scope)
        }
        Expression::List(list) => list.elements.iter().try_for_each(|e| check_expr(e, scope)),
        Expression::ExpressionList(list) => list
            .expressions
            .iter()
            .try_for_each(|e| check_expr(e, scope)),
        Expression::Distinct(d) => check_expr(&d.expr, scope),
        Expression::Exists(e) => check_select(&e.subquery, scope),
        Expression::ScalarSubquery(s) => check_select(&s.subquery, scope),
        Expression::AllAny(a) => {
            check_expr(&a.left, scope)?;
            check_select(&a.subquery, scope)
        }
        Expression::In(i) => {
            check_expr(&i.left, scope)?;
            check_expr(&i.right, scope)
        }
        Expression::InHashSet(i) => check_expr(&i.column, scope),
        Expression::QuantifiedSet(q) => check_expr(&q.left, scope),
        Expression::Between(b) => {
            check_expr(&b.expr, scope)?;
            check_expr(&b.lower, scope)?;
            check_expr(&b.upper, scope)
        }
        Expression::Like(l) => {
            check_expr(&l.left, scope)?;
            check_expr(&l.pattern, scope)?;
            check_opt(l.escape.as_deref(), scope)
        }
        Expression::Case(c) => {
            check_opt(c.value.as_deref(), scope)?;
            for when in &c.when_clauses {
                check_expr(&when.condition, scope)?;
                check_expr(&when.then_result, scope)?;
            }
            check_opt(c.else_value.as_deref(), scope)
        }
        Expression::Cast(c) => check_expr(&c.expr, scope),
        Expression::FunctionCall(f) => check_function(f, scope),
        Expression::Aliased(a) => check_expr(&a.expression, scope),
        Expression::Window(w) => {
            check_function(&w.function, scope)?;
            check_window(&w.partition_by, &w.order_by, w.frame.as_ref(), scope)
        }
        _ => Ok(()),
    }
}

/// `lower` names a FROM entry some level of `scope` shows
fn qualifier(
    spelled: &str,
    lower: &str,
    position: Position,
    scope: Option<&Scope<'_, '_>>,
) -> Check {
    let mut held = false;
    let mut level = scope;
    while let Some(scope) = level {
        if scope.visible.contains(&lower) {
            return Ok(());
        }
        held |= scope.hidden.contains(&lower) || scope.beside.is_some_and(|q| q.holds(lower));
        level = scope.parent;
    }
    let message = if held {
        format!("table or alias '{spelled}' is not visible here")
    } else {
        unknown_qualifier(spelled)
    };
    Err(ScopeError { message, position })
}
