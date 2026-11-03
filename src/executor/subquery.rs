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

//! Which subqueries an expression or a statement holds, and whether they
//! read an enclosing row: the rules the prepared families decide by

use crate::functions::{global_registry, Volatility};
use crate::parser::ast::*;

use super::context::ExecutionContext;
use super::volatility::expression_volatility_in;
use super::Executor;

impl Executor {
    /// Check if an expression contains EXISTS or other subqueries that need processing
    pub(crate) fn has_subqueries(expr: &Expression) -> bool {
        match expr {
            Expression::Exists(_) => true,
            Expression::ScalarSubquery(_) => true,
            Expression::AllAny(_) => true,
            Expression::Prefix(prefix) => Self::has_subqueries(&prefix.right),
            Expression::Infix(infix) => {
                Self::has_subqueries(&infix.left) || Self::has_subqueries(&infix.right)
            }
            Expression::In(in_expr) => {
                Self::has_subqueries(&in_expr.left) || Self::has_subqueries(&in_expr.right)
            }
            Expression::Between(between) => {
                Self::has_subqueries(&between.expr)
                    || Self::has_subqueries(&between.lower)
                    || Self::has_subqueries(&between.upper)
            }
            Expression::Aliased(aliased) => Self::has_subqueries(&aliased.expression),
            Expression::FunctionCall(func) => {
                func.arguments.iter().any(Self::has_subqueries)
                    || func.filter.as_deref().is_some_and(Self::has_subqueries)
                    || func
                        .order_by
                        .iter()
                        .any(|order| Self::has_subqueries(&order.expression))
            }
            Expression::Case(case) => {
                case.value.as_deref().is_some_and(Self::has_subqueries)
                    || case.when_clauses.iter().any(|w| {
                        Self::has_subqueries(&w.condition) || Self::has_subqueries(&w.then_result)
                    })
                    || case.else_value.as_deref().is_some_and(Self::has_subqueries)
            }
            Expression::Cast(cast) => Self::has_subqueries(&cast.expr),
            Expression::Like(like) => {
                Self::has_subqueries(&like.left)
                    || Self::has_subqueries(&like.pattern)
                    || like.escape.as_deref().is_some_and(Self::has_subqueries)
            }
            Expression::List(list) => list.elements.iter().any(Self::has_subqueries),
            Expression::ExpressionList(list) => list.expressions.iter().any(Self::has_subqueries),
            Expression::Distinct(distinct) => Self::has_subqueries(&distinct.expr),
            Expression::Window(window) => window
                .function
                .arguments
                .iter()
                .chain(window.function.filter.as_deref())
                .chain(window.function.order_by.iter().map(|o| &o.expression))
                .chain(window.partition_by.iter())
                .chain(window.order_by.iter().map(|o| &o.expression))
                .chain(window.frame_offsets())
                .any(Self::has_subqueries),
            _ => false,
        }
    }

    /// Check if an expression contains correlated subqueries that reference outer columns.
    /// A correlated subquery references columns from outer tables that are not defined
    /// in the subquery's own FROM clause.
    pub(crate) fn has_correlated_subqueries(expr: &Expression) -> bool {
        match expr {
            Expression::Exists(exists) => Self::is_subquery_correlated(&exists.subquery),
            Expression::ScalarSubquery(subquery) => {
                Self::is_subquery_correlated(&subquery.subquery)
            }
            Expression::AllAny(all_any) => {
                Self::is_subquery_correlated(&all_any.subquery)
                    || Self::has_correlated_subqueries(&all_any.left)
            }
            Expression::Prefix(prefix) => {
                // Handle NOT EXISTS
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    return Self::is_subquery_correlated(&exists.subquery);
                }
                Self::has_correlated_subqueries(&prefix.right)
            }
            Expression::Infix(infix) => {
                Self::has_correlated_subqueries(&infix.left)
                    || Self::has_correlated_subqueries(&infix.right)
            }
            Expression::In(in_expr) => {
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    return Self::is_subquery_correlated(&subquery.subquery);
                }
                Self::has_correlated_subqueries(&in_expr.left)
                    || Self::has_correlated_subqueries(&in_expr.right)
            }
            Expression::Between(between) => {
                Self::has_correlated_subqueries(&between.expr)
                    || Self::has_correlated_subqueries(&between.lower)
                    || Self::has_correlated_subqueries(&between.upper)
            }
            Expression::Aliased(aliased) => Self::has_correlated_subqueries(&aliased.expression),
            Expression::FunctionCall(func) => {
                func.arguments.iter().any(Self::has_correlated_subqueries)
                    || func
                        .filter
                        .as_deref()
                        .is_some_and(Self::has_correlated_subqueries)
                    || func
                        .order_by
                        .iter()
                        .any(|order| Self::has_correlated_subqueries(&order.expression))
            }
            Expression::Case(case) => {
                case.value
                    .as_deref()
                    .is_some_and(Self::has_correlated_subqueries)
                    || case.when_clauses.iter().any(|w| {
                        Self::has_correlated_subqueries(&w.condition)
                            || Self::has_correlated_subqueries(&w.then_result)
                    })
                    || case
                        .else_value
                        .as_deref()
                        .is_some_and(Self::has_correlated_subqueries)
            }
            Expression::Cast(cast) => Self::has_correlated_subqueries(&cast.expr),
            Expression::Like(like) => {
                Self::has_correlated_subqueries(&like.left)
                    || Self::has_correlated_subqueries(&like.pattern)
                    || like
                        .escape
                        .as_deref()
                        .is_some_and(Self::has_correlated_subqueries)
            }
            Expression::List(list) => list.elements.iter().any(Self::has_correlated_subqueries),
            Expression::ExpressionList(list) => {
                list.expressions.iter().any(Self::has_correlated_subqueries)
            }
            Expression::Distinct(distinct) => Self::has_correlated_subqueries(&distinct.expr),
            Expression::Window(window) => window
                .function
                .arguments
                .iter()
                .chain(window.function.filter.as_deref())
                .chain(window.function.order_by.iter().map(|o| &o.expression))
                .chain(window.partition_by.iter())
                .chain(window.order_by.iter().map(|o| &o.expression))
                .any(Self::has_correlated_subqueries),
            _ => false,
        }
    }

    /// True when the SELECT list, HAVING or ORDER BY holds a subquery, which
    /// an aggregation path that reads them raw cannot run
    pub(crate) fn aggregates_hold_subquery(stmt: &SelectStatement) -> bool {
        stmt.columns.iter().any(Self::has_subqueries)
            || stmt.having.as_deref().is_some_and(Self::has_subqueries)
            || stmt
                .order_by
                .iter()
                .any(|order| Self::has_subqueries(&order.expression))
    }

    /// Whether a subquery under `ctx` reads committed rows no older than a
    /// stamp taken before its read: no explicit transaction and no statement
    /// snapshot that keeps the rows of its start or is written through. A
    /// statement view answers only through a stamp it sees whole
    pub(crate) fn reads_cacheable_rows(ctx: &ExecutionContext) -> bool {
        ctx.transaction_id().is_none()
            && ctx.statement_snapshot().is_none_or(|snapshot| {
                snapshot.writes_nothing()
                    && (snapshot.reads_latest()
                        || snapshot.view().is_some() && ctx.statement_view().is_some())
            })
    }

    /// Whether the value of `expr` may change from one statement to the next
    /// with the same text: it calls a function that is not fixed, or reads a
    /// CTE of the statement it runs in, by the built-in registry's rules
    pub(crate) fn expression_varies_by_statement(
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> bool {
        expression_volatility_in(expr, global_registry(), &|name| {
            ctx.get_cte_by_lower(name).is_some()
        }) != Volatility::Fixed
    }

    /// Whether a window function, its OVER clause or a named window holds a subquery
    pub(crate) fn windows_hold_subquery(stmt: &SelectStatement) -> bool {
        stmt.columns.iter().any(Self::has_subqueries)
            || stmt.window_defs.iter().any(|window| {
                window.partition_by.iter().any(Self::has_subqueries)
                    || window
                        .order_by
                        .iter()
                        .any(|order| Self::has_subqueries(&order.expression))
            })
    }

    /// Check if a subquery is correlated (references outer columns)
    pub(crate) fn is_subquery_correlated(subquery: &SelectStatement) -> bool {
        // Get table/alias names defined in the subquery's FROM clause
        let subquery_tables = Self::collect_subquery_table_columns(subquery);

        // Check if the WHERE clause references any outer columns
        if let Some(ref where_clause) = subquery.where_clause {
            if Self::references_outer_columns(where_clause, &subquery_tables) {
                return true;
            }
        }

        // Also check SELECT expressions (for scalar subqueries)
        for col in &subquery.columns {
            if Self::references_outer_columns(col, &subquery_tables) {
                return true;
            }
        }

        // A HAVING reads the parent row the same way a WHERE does
        if let Some(ref having) = subquery.having {
            if Self::references_outer_columns(having, &subquery_tables) {
                return true;
            }
        }

        // So does an ORDER BY, which decides what a LIMIT keeps, and a
        // DISTINCT ON, which decides which rows stay
        if subquery
            .order_by
            .iter()
            .map(|o| &o.expression)
            .chain(&subquery.distinct_on)
            .any(|e| Self::references_outer_columns(e, &subquery_tables))
        {
            return true;
        }

        // A derived table in the FROM may read the parent row too
        subquery
            .table_expr
            .as_deref()
            .is_some_and(|from| Self::from_reads_outer_columns(from, &subquery_tables))
    }

    /// Whether a derived table in a FROM, or a join condition there, reads a
    /// column that neither the query nor the derived table defines
    fn from_reads_outer_columns<S: AsRef<str>>(from: &Expression, subquery_tables: &[S]) -> bool {
        match from {
            Expression::SubquerySource(source) => {
                Self::nested_reads_outer_columns(&source.subquery, subquery_tables)
            }
            Expression::JoinSource(join) => {
                Self::from_reads_outer_columns(&join.left, subquery_tables)
                    || Self::from_reads_outer_columns(&join.right, subquery_tables)
                    || join
                        .condition
                        .as_deref()
                        .is_some_and(|c| Self::references_outer_columns(c, subquery_tables))
            }
            Expression::Aliased(aliased) => {
                Self::from_reads_outer_columns(&aliased.expression, subquery_tables)
            }
            _ => false,
        }
    }

    /// Collect table/alias names from a subquery's FROM clause
    fn collect_subquery_table_columns(subquery: &SelectStatement) -> smallvec::SmallVec<[&str; 4]> {
        let mut tables = smallvec::SmallVec::new();

        if let Some(ref table_expr) = subquery.table_expr {
            Self::collect_table_names_from_source(table_expr, &mut tables);
        }

        tables
    }

    /// Recursively collect table names from a table source expression
    pub(super) fn collect_table_names_from_source<'a>(
        source: &'a Expression,
        tables: &mut smallvec::SmallVec<[&'a str; 4]>,
    ) {
        match source {
            Expression::TableSource(ts) => {
                // When a table has an alias, SQL semantics require using the alias,
                // not the original table name. So for `FROM t t2`, only `t2` is valid.
                // If there's no alias, use the table name.
                if let Some(ref alias) = ts.alias {
                    tables.push(alias.value_lower.as_str());
                } else {
                    tables.push(ts.name.value_lower.as_str());
                }
            }
            Expression::JoinSource(js) => {
                Self::collect_table_names_from_source(&js.left, tables);
                Self::collect_table_names_from_source(&js.right, tables);
            }
            Expression::SubquerySource(ss) => {
                // Subquery source has an optional alias
                if let Some(ref alias) = ss.alias {
                    tables.push(alias.value_lower.as_str());
                }
            }
            Expression::ValuesSource(vs) => {
                if let Some(ref alias) = vs.alias {
                    tables.push(alias.value_lower.as_str());
                }
            }
            Expression::FunctionTableSource(fs) => {
                if let Some(ref alias) = fs.alias {
                    tables.push(alias.value_lower.as_str());
                } else {
                    tables.push(fs.function.value_lower.as_str());
                }
            }
            _ => {}
        }
    }

    /// Check if an expression references columns from outer scope.
    ///
    /// For simple identifiers, we cannot reliably determine if they reference outer columns
    /// since the same column name might exist in both inner and outer scopes. The inner
    /// scope takes precedence per SQL semantics, so simple identifiers are NOT considered
    /// outer references (they will resolve to inner scope if available).
    ///
    /// For qualified identifiers (e.g., c.id), we check if the qualifier (table/alias)
    /// is NOT defined in the subquery's FROM clause - if so, it must be an outer reference.
    pub(super) fn references_outer_columns<S: AsRef<str>>(
        expr: &Expression,
        subquery_tables: &[S],
    ) -> bool {
        match expr {
            Expression::Identifier(_id) => {
                // Simple identifiers are ambiguous - they resolve to inner scope first per SQL semantics.
                // We cannot determine if this is an outer reference without knowing the inner schema.
                // Conservative approach: don't mark as correlated based on simple identifiers alone.
                // Users should use qualified names (e.g., c.id) for outer references in correlated subqueries.
                false
            }
            Expression::QualifiedIdentifier(qid) => {
                // Qualified identifier like "c.id" or "outer_table.column"
                let table_name = &qid.qualifier.value_lower;

                // If the table/alias is NOT in subquery tables, it's an outer reference
                // This is the key check: if "c" is not in ["orders", "o"], then c.id is outer
                !subquery_tables
                    .iter()
                    .any(|t| t.as_ref().eq_ignore_ascii_case(table_name))
            }
            Expression::Infix(infix) => {
                Self::references_outer_columns(&infix.left, subquery_tables)
                    || Self::references_outer_columns(&infix.right, subquery_tables)
            }
            Expression::Prefix(prefix) => {
                Self::references_outer_columns(&prefix.right, subquery_tables)
            }
            Expression::FunctionCall(func) => {
                func.arguments
                    .iter()
                    .any(|arg| Self::references_outer_columns(arg, subquery_tables))
                    || func
                        .filter
                        .as_deref()
                        .is_some_and(|f| Self::references_outer_columns(f, subquery_tables))
                    || func
                        .order_by
                        .iter()
                        .any(|o| Self::references_outer_columns(&o.expression, subquery_tables))
            }
            Expression::In(in_expr) => {
                Self::references_outer_columns(&in_expr.left, subquery_tables)
                    || Self::references_outer_columns(&in_expr.right, subquery_tables)
            }
            Expression::Between(between) => {
                Self::references_outer_columns(&between.expr, subquery_tables)
                    || Self::references_outer_columns(&between.lower, subquery_tables)
                    || Self::references_outer_columns(&between.upper, subquery_tables)
            }
            Expression::Case(case) => {
                if let Some(ref value) = case.value {
                    if Self::references_outer_columns(value, subquery_tables) {
                        return true;
                    }
                }
                for when in &case.when_clauses {
                    if Self::references_outer_columns(&when.condition, subquery_tables)
                        || Self::references_outer_columns(&when.then_result, subquery_tables)
                    {
                        return true;
                    }
                }
                if let Some(ref else_val) = case.else_value {
                    if Self::references_outer_columns(else_val, subquery_tables) {
                        return true;
                    }
                }
                false
            }
            Expression::Aliased(aliased) => {
                Self::references_outer_columns(&aliased.expression, subquery_tables)
            }
            Expression::AllAny(all_any) => {
                Self::references_outer_columns(&all_any.left, subquery_tables)
                    || Self::nested_reads_outer_columns(&all_any.subquery, subquery_tables)
            }
            Expression::Cast(cast) => Self::references_outer_columns(&cast.expr, subquery_tables),
            Expression::Like(like) => {
                Self::references_outer_columns(&like.left, subquery_tables)
                    || Self::references_outer_columns(&like.pattern, subquery_tables)
                    || like
                        .escape
                        .as_deref()
                        .is_some_and(|e| Self::references_outer_columns(e, subquery_tables))
            }
            Expression::Distinct(distinct) => {
                Self::references_outer_columns(&distinct.expr, subquery_tables)
            }
            Expression::List(list) => list
                .elements
                .iter()
                .any(|e| Self::references_outer_columns(e, subquery_tables)),
            Expression::ExpressionList(list) => list
                .expressions
                .iter()
                .any(|e| Self::references_outer_columns(e, subquery_tables)),
            // A subquery folded to its members keeps the side it is tested on
            Expression::InHashSet(set) => {
                Self::references_outer_columns(&set.column, subquery_tables)
            }
            Expression::QuantifiedSet(set) => {
                Self::references_outer_columns(&set.left, subquery_tables)
            }
            Expression::Window(window) => window
                .function
                .arguments
                .iter()
                .chain(window.function.filter.as_deref())
                .chain(window.function.order_by.iter().map(|o| &o.expression))
                .chain(window.partition_by.iter())
                .chain(window.order_by.iter().map(|o| &o.expression))
                .any(|e| Self::references_outer_columns(e, subquery_tables)),
            // A subquery nested in this one may read a column from further
            // out than either of them, which this one has to carry in
            Expression::Exists(exists) => {
                Self::nested_reads_outer_columns(&exists.subquery, subquery_tables)
            }
            Expression::ScalarSubquery(subquery) => {
                Self::nested_reads_outer_columns(&subquery.subquery, subquery_tables)
            }
            _ => false,
        }
    }

    /// Whether a subquery nested inside another reads a column that neither
    /// of them defines
    fn nested_reads_outer_columns<S: AsRef<str>>(
        nested: &SelectStatement,
        subquery_tables: &[S],
    ) -> bool {
        let mut tables: smallvec::SmallVec<[&str; 4]> =
            subquery_tables.iter().map(AsRef::as_ref).collect();
        if let Some(from) = &nested.table_expr {
            Self::collect_table_names_from_source(from, &mut tables);
        }
        nested
            .where_clause
            .as_deref()
            .is_some_and(|where_clause| Self::references_outer_columns(where_clause, &tables))
            || nested
                .columns
                .iter()
                .any(|column| Self::references_outer_columns(column, &tables))
            || nested
                .having
                .as_deref()
                .is_some_and(|having| Self::references_outer_columns(having, &tables))
            || nested
                .order_by
                .iter()
                .any(|o| Self::references_outer_columns(&o.expression, &tables))
            || nested
                .table_expr
                .as_deref()
                .is_some_and(|from| Self::from_reads_outer_columns(from, &tables))
    }
}
