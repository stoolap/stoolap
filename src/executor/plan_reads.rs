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

//! The tables and views a statement names, which a plan made for it holds
//! while none of them changes; a view's query runs under its own plan

use crate::common::SmartString;
use crate::parser::ast::*;
use crate::storage::traits::Engine;

use super::query_cache::PlanValidity;
use super::Executor;

impl Executor {
    /// The validity of a plan for the SELECT `stmt` made from the schemas as
    /// they were at `epoch`, read before the plan read any of them: it knows
    /// what it read when no change came since
    pub(crate) fn select_validity(&self, stmt: &SelectStatement, epoch: u64) -> PlanValidity {
        self.validity(epoch, |reads| reads.select(stmt))
    }

    /// `select_validity` for the INSERT `stmt`
    pub(crate) fn insert_validity(&self, stmt: &InsertStatement, epoch: u64) -> PlanValidity {
        self.validity(epoch, |reads| reads.insert(stmt))
    }

    /// `select_validity` for the UPDATE `stmt`
    pub(crate) fn update_validity(&self, stmt: &UpdateStatement, epoch: u64) -> PlanValidity {
        self.validity(epoch, |reads| reads.update(stmt))
    }

    /// `select_validity` for the DELETE `stmt`
    pub(crate) fn delete_validity(&self, stmt: &DeleteStatement, epoch: u64) -> PlanValidity {
        self.validity(epoch, |reads| reads.delete(stmt))
    }

    /// `select_validity` for a plan over the tables `targets` that runs `exprs`
    pub(crate) fn parts_validity<'e>(
        &self,
        epoch: u64,
        targets: &[&str],
        exprs: impl IntoIterator<Item = &'e Expression>,
    ) -> PlanValidity {
        self.validity(epoch, |reads| {
            for target in targets {
                reads.name(&SmartString::from(target.to_lowercase()));
            }
            exprs.into_iter().for_each(|e| reads.expr(e));
        })
    }

    fn validity(&self, epoch: u64, walk: impl FnOnce(&mut Reads<'_>)) -> PlanValidity {
        let mut reads = Reads {
            executor: self,
            names: Vec::new(),
        };
        walk(&mut reads);
        let unchanged = self.engine.schema_epoch() == epoch;
        PlanValidity::new(epoch, unchanged.then(|| reads.names.into()))
    }
}

/// The names a walk has read, each with the epoch of its last change
struct Reads<'a> {
    executor: &'a Executor,
    names: Vec<(SmartString, u64)>,
}

impl Reads<'_> {
    /// A table or view `name`, lower case
    fn name(&mut self, name: &SmartString) {
        if self.names.iter().any(|(read, _)| read == name) {
            return;
        }
        let at = self.executor.engine.name_epoch(name);
        self.names.push((name.clone(), at));
    }

    fn insert(&mut self, insert: &InsertStatement) {
        self.name(&insert.table_name.value_lower);
        insert.values.iter().flatten().for_each(|e| self.expr(e));
        if let Some(select) = &insert.select {
            self.select(select);
        }
        insert.update_expressions.iter().for_each(|e| self.expr(e));
        self.exprs(insert.update_where.as_deref());
        insert.returning.iter().for_each(|e| self.expr(e));
    }

    fn update(&mut self, update: &UpdateStatement) {
        self.with(update.with.as_ref());
        self.name(&update.table_name.value_lower);
        update.updates.values().for_each(|e| self.expr(e));
        self.exprs(update.where_clause.as_deref());
        update.returning.iter().for_each(|e| self.expr(e));
    }

    fn delete(&mut self, delete: &DeleteStatement) {
        self.with(delete.with.as_ref());
        self.name(&delete.table_name.value_lower);
        self.exprs(delete.where_clause.as_deref());
        delete.returning.iter().for_each(|e| self.expr(e));
    }

    fn with(&mut self, with: Option<&WithClause>) {
        for cte in with.iter().flat_map(|with| &with.ctes) {
            self.select(&cte.query);
        }
    }

    fn select(&mut self, select: &SelectStatement) {
        self.with(select.with.as_ref());
        select.distinct_on.iter().for_each(|e| self.expr(e));
        select.columns.iter().for_each(|e| self.expr(e));
        self.exprs(select.table_expr.as_deref());
        self.exprs(select.where_clause.as_deref());
        select.group_by.columns.iter().for_each(|e| self.expr(e));
        self.exprs(select.having.as_deref());
        for window in &select.window_defs {
            window.partition_by.iter().for_each(|e| self.expr(e));
            window
                .order_by
                .iter()
                .for_each(|o| self.expr(&o.expression));
            for bound in window
                .frame
                .iter()
                .flat_map(|frame| std::iter::once(&frame.start).chain(&frame.end))
            {
                if let WindowFrameBound::Preceding(offset) | WindowFrameBound::Following(offset) =
                    bound
                {
                    self.expr(offset);
                }
            }
        }
        select
            .order_by
            .iter()
            .for_each(|o| self.expr(&o.expression));
        self.exprs(select.limit.as_deref());
        self.exprs(select.offset.as_deref());
        for set_op in &select.set_operations {
            self.select(&set_op.right);
        }
    }

    fn exprs(&mut self, expr: Option<&Expression>) {
        if let Some(expr) = expr {
            self.expr(expr);
        }
    }

    fn expr(&mut self, expr: &Expression) {
        match expr {
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::Column(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => {}
            Expression::Prefix(prefix) => self.expr(&prefix.right),
            Expression::Infix(infix) => {
                self.expr(&infix.left);
                self.expr(&infix.right);
            }
            Expression::List(list) => list.elements.iter().for_each(|e| self.expr(e)),
            Expression::Distinct(distinct) => self.expr(&distinct.expr),
            Expression::Exists(exists) => self.select(&exists.subquery),
            Expression::AllAny(all_any) => {
                self.expr(&all_any.left);
                self.select(&all_any.subquery);
            }
            Expression::In(in_expr) => {
                self.expr(&in_expr.left);
                self.expr(&in_expr.right);
            }
            Expression::InHashSet(in_set) => self.expr(&in_set.column),
            Expression::QuantifiedSet(set) => self.expr(&set.left),
            Expression::Between(between) => {
                self.expr(&between.expr);
                self.expr(&between.lower);
                self.expr(&between.upper);
            }
            Expression::Like(like) => {
                self.expr(&like.left);
                self.expr(&like.pattern);
                self.exprs(like.escape.as_deref());
            }
            Expression::ScalarSubquery(subquery) => self.select(&subquery.subquery),
            Expression::ExpressionList(list) => list.expressions.iter().for_each(|e| self.expr(e)),
            Expression::Case(case) => {
                self.exprs(case.value.as_deref());
                for when in &case.when_clauses {
                    self.expr(&when.condition);
                    self.expr(&when.then_result);
                }
                self.exprs(case.else_value.as_deref());
            }
            Expression::Cast(cast) => self.expr(&cast.expr),
            Expression::FunctionCall(call) => self.call(call),
            Expression::Aliased(aliased) => self.expr(&aliased.expression),
            Expression::Window(window) => {
                self.call(&window.function);
                window.partition_by.iter().for_each(|e| self.expr(e));
                window
                    .order_by
                    .iter()
                    .for_each(|o| self.expr(&o.expression));
                window.frame_offsets().for_each(|e| self.expr(e));
            }
            Expression::TableSource(source) => self.name(&source.name.value_lower),
            Expression::JoinSource(join) => {
                self.expr(&join.left);
                self.expr(&join.right);
                self.exprs(join.condition.as_deref());
            }
            Expression::SubquerySource(source) => self.select(&source.subquery),
            Expression::ValuesSource(values) => {
                values.rows.iter().flatten().for_each(|e| self.expr(e));
            }
            Expression::CteReference(cte) => self.name(&cte.name.value_lower),
            Expression::FunctionTableSource(source) => {
                source.arguments.iter().for_each(|e| self.expr(e));
            }
        }
    }

    fn call(&mut self, call: &FunctionCall) {
        call.arguments.iter().for_each(|e| self.expr(e));
        self.exprs(call.filter.as_deref());
        call.order_by.iter().for_each(|o| self.expr(&o.expression));
    }
}
