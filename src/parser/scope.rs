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
//! of the query they are written in or of an enclosing one; and under
//! SELECT DISTINCT, every ORDER BY key is a SELECT item

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
    let level: SmallVec<[&str; 4]> = own.map_or_else(SmallVec::new, |own| own.visible.clone());
    if let Some(key) = distinct_key_outside(select, &level) {
        let otherwise = match crate::executor::utils::expression_contains_aggregate(key) {
            true => String::from(
                "an aggregate is one value per distinct row once every GROUP BY item is \
                 selected",
            ),
            false => format!(
                "to sort by something else, group the rows and sort by an aggregate of it, \
                 such as MAX({key})"
            ),
        };
        return Err(ScopeError {
            message: format!(
                "ORDER BY {key} is not in the SELECT DISTINCT list: rows merged into one \
                 distinct row may hold different values of it. Sort by a selected item, its \
                 alias or position, or an expression of selected columns; {otherwise}"
            ),
            position: key.position(),
        });
    }
    check_opt(select.limit.as_deref(), outer)?;
    check_opt(select.offset.as_deref(), outer)?;
    for op in &select.set_operations {
        check_select(&op.right, outer)?;
    }
    Ok(())
}

/// Under SELECT DISTINCT, the first ORDER BY key the SELECT items do not
/// determine: one distinct row stands for several rows, which such a key
/// may tell apart. `level` holds the names this query's FROM entries show
fn distinct_key_outside<'s>(select: &'s SelectStatement, level: &[&str]) -> Option<&'s Expression> {
    if !select.distinct || !select.distinct_on.is_empty() || !select.set_operations.is_empty() {
        return None;
    }
    // Each group is then one distinct row, so DISTINCT merges none
    let grouping = &select.group_by;
    if matches!(grouping.modifier, GroupByModifier::None)
        && !grouping.columns.is_empty()
        && grouping
            .columns
            .iter()
            .all(|key| determined(key, &select.columns, level))
    {
        return None;
    }
    select
        .order_by
        .iter()
        .map(|order| &order.expression)
        .find(|key| !determined(key, &select.columns, level))
}

/// Whether ORDER BY `key` has one value across the rows a distinct row of
/// `items` stands for: a position in the list or an alias, an item's own
/// expression, or a scalar expression of columns the items determine
fn determined(key: &Expression, items: &[Expression], level: &[&str]) -> bool {
    let star = items
        .iter()
        .any(|item| matches!(item, Expression::Star(_) | Expression::QualifiedStar(_)));
    match key {
        // A star's width is not known here, only that it reaches past its place
        Expression::IntegerLiteral(position) => {
            return position.value >= 1 && (star || position.value as usize <= items.len())
        }
        Expression::Identifier(name)
            if items.iter().any(|item| {
                matches!(item, Expression::Aliased(a) if a.alias.value_lower == name.value_lower)
            }) =>
        {
            return true
        }
        _ => {}
    }
    items
        .iter()
        .any(|item| same_value(key, unaliased(item), level))
        || scalar_of(key, &mut |column| {
            column_determined(column, items.iter().map(unaliased), level)
        })
}

/// An item without its alias
fn unaliased(item: &Expression) -> &Expression {
    match item {
        Expression::Aliased(aliased) => &aliased.expression,
        item => item,
    }
}

/// Whether a column the key reads has one value across those rows: a
/// selected column, one a star gives, or an enclosing query's, which is
/// constant here
fn column_determined<'i>(
    column: &Expression,
    items: impl Iterator<Item = &'i Expression>,
    level: &[&str],
) -> bool {
    let outer = matches!(column, Expression::QualifiedIdentifier(c)
        if !level.contains(&c.qualifier.value_lower.as_str()));
    outer
        || items.into_iter().any(|item| match item {
            Expression::Star(_) => true,
            // A bare name is the starred table's when it is the one FROM entry
            Expression::QualifiedStar(star) => match column {
                Expression::QualifiedIdentifier(c) => {
                    c.qualifier.value_lower == star.qualifier.to_lowercase()
                }
                _ => level == [star.qualifier.to_lowercase().as_str()],
            },
            item => same_column(column, item, level),
        })
}

/// Whether `expr` is computed from its columns alone, each passing `column`:
/// no aggregate, window or subquery, which read rows beyond the one, and no
/// function the global registry says may answer anew per call
fn scalar_of(expr: &Expression, column: &mut dyn FnMut(&Expression) -> bool) -> bool {
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => column(expr),
        Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::IntervalLiteral(_)
        | Expression::Parameter(_) => true,
        Expression::Prefix(p) => scalar_of(&p.right, column),
        Expression::Infix(i) => scalar_of(&i.left, column) && scalar_of(&i.right, column),
        Expression::Cast(c) => scalar_of(&c.expr, column),
        Expression::Between(b) => {
            scalar_of(&b.expr, column) && scalar_of(&b.lower, column) && scalar_of(&b.upper, column)
        }
        Expression::Like(l) => {
            scalar_of(&l.left, column)
                && scalar_of(&l.pattern, column)
                && l.escape.as_deref().is_none_or(|e| scalar_of(e, column))
        }
        Expression::In(i) => scalar_of(&i.left, column) && scalar_of(&i.right, column),
        Expression::List(l) => l.elements.iter().all(|e| scalar_of(e, column)),
        Expression::ExpressionList(l) => l.expressions.iter().all(|e| scalar_of(e, column)),
        Expression::Case(c) => {
            c.value.as_deref().is_none_or(|v| scalar_of(v, column))
                && c.when_clauses
                    .iter()
                    .all(|w| scalar_of(&w.condition, column) && scalar_of(&w.then_result, column))
                && c.else_value.as_deref().is_none_or(|e| scalar_of(e, column))
        }
        Expression::FunctionCall(f) => {
            let registry = crate::functions::registry::global_registry();
            !registry.is_aggregate(&f.function)
                && registry.volatility(&f.function) != crate::functions::Volatility::PerCall
                && f.filter.is_none()
                && f.order_by.is_empty()
                && f.arguments.iter().all(|a| scalar_of(a, column))
        }
        _ => false,
    }
}

/// Whether two column references name the same column: the same name, and
/// a qualifier on one side only when it is an entry of this query's FROM,
/// where a bare name of a valid statement binds
fn same_column(a: &Expression, b: &Expression, level: &[&str]) -> bool {
    let at_level = |q: &Identifier| level.contains(&q.value_lower.as_str());
    match (a, b) {
        (Expression::Identifier(x), Expression::Identifier(y)) => x.value_lower == y.value_lower,
        (Expression::QualifiedIdentifier(x), Expression::QualifiedIdentifier(y)) => {
            x.qualifier.value_lower == y.qualifier.value_lower
                && x.name.value_lower == y.name.value_lower
        }
        (Expression::QualifiedIdentifier(q), Expression::Identifier(n))
        | (Expression::Identifier(n), Expression::QualifiedIdentifier(q)) => {
            at_level(&q.qualifier) && q.name.value_lower == n.value_lower
        }
        _ => false,
    }
}

/// Whether two expressions compute the same value: the same tree, each pair
/// of columns the same column. A tree reading more than its names, as a
/// subquery or a window does, must be written alike
fn same_value(a: &Expression, b: &Expression, level: &[&str]) -> bool {
    use crate::executor::expr_key::ExprKey;
    use crate::executor::utils::visit_column_refs;
    if ExprKey::by_column_name(a) != ExprKey::by_column_name(b) {
        return false;
    }
    let (mut left, mut right) = (Vec::new(), Vec::new());
    let plain =
        visit_column_refs(a, &mut |c| left.push(c)) & visit_column_refs(b, &mut |c| right.push(c));
    if !plain {
        return ExprKey::of(a) == ExprKey::of(b);
    }
    left.len() == right.len()
        && left
            .iter()
            .zip(&right)
            .all(|(x, y)| same_column(x, y, level))
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
