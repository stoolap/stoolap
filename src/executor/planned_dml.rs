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

//! An UPDATE or DELETE whose WHERE is its table's primary key equality: the
//! plan that reads the row the key names and writes it, decided once

use std::sync::{Arc, RwLock};

use crate::common::{CompactArc, SmartString};
use crate::core::{DataType, Result, Row, Schema, Value};
use crate::parser::ast::{DeleteStatement, Expression, UpdateStatement};
use crate::planner::ScanChoice;
use crate::storage::expression::{ComparisonExpr, Expression as StorageExpression};
use crate::storage::mvcc::index_plan::IndexPlan;
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::transaction::StatementTable;
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};

use super::context::ExecutionContext;
use super::dml::StatementRollback;
use super::expression::SharedProgram;
use super::query_cache::{
    check_compiled_token, or_normal_path, CompiledExecution, CompiledUpdateColumn, PkValueSource,
    PlanValidity, UpdateValueSource,
};
use super::result::ExecResult;
use super::Executor;

/// The table a planned write goes through, the autocommit transaction to
/// commit after it if any, and the rollback of the statement on failure
type PkWriteTable<'a> = (
    Box<dyn Table>,
    Option<Box<dyn Transaction>>,
    StatementRollback<'a>,
);

/// A row expression an UPDATE sets a column to: the column, its program,
/// type and vector dimensions
type SetProgram = (usize, SharedProgram, DataType, u16);

/// An UPDATE or DELETE, as the plan reads it
#[derive(Clone, Copy)]
pub(crate) enum DmlStatement<'a> {
    Update(&'a UpdateStatement),
    Delete(&'a DeleteStatement),
}

impl DmlStatement<'_> {
    fn table(&self) -> &str {
        match self {
            Self::Update(stmt) => &stmt.table_name.value_lower,
            Self::Delete(stmt) => &stmt.table_name.value_lower,
        }
    }

    fn where_clause(&self) -> Option<&Expression> {
        match self {
            Self::Update(stmt) => stmt.where_clause.as_deref(),
            Self::Delete(stmt) => stmt.where_clause.as_deref(),
        }
    }

    /// Whether it has a RETURNING or WITH, which the general path runs
    fn returns_or_reads_ctes(&self) -> bool {
        match self {
            Self::Update(stmt) => !stmt.returning.is_empty() || stmt.with.is_some(),
            Self::Delete(stmt) => !stmt.returning.is_empty() || stmt.with.is_some(),
        }
    }
}

/// What a planned statement writes to the row its key names
#[derive(Debug)]
enum DmlWrite {
    Update(Vec<CompiledUpdateColumn>),
    Delete,
}

/// An UPDATE or DELETE that reads the row its primary key equality names,
/// planned over the table it was planned for
#[derive(Debug)]
pub struct PlannedDml {
    table: SmartString,
    schema: CompactArc<Schema>,
    /// The table instance and layout the column positions describe
    token: LayoutToken,
    pub(crate) validity: PlanValidity,
    pk_column: SmartString,
    key: PkValueSource,
    write: DmlWrite,
    /// Whether a value it sets is a row expression, which a run reads its
    /// parameters for from its context
    sets_rows: bool,
}

impl Executor {
    /// `stmt` run by the plan `compiled` holds, planned now in `ctx` when it
    /// holds none; None for the general path to answer. `params` are the
    /// statement's values; a run without `ctx` sets no row expression
    pub(crate) fn run_dml_plan(
        &self,
        stmt: DmlStatement<'_>,
        compiled: &RwLock<CompiledExecution>,
        ctx: Option<&ExecutionContext>,
        params: &[Value],
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        let epoch = self.engine.schema_epoch();
        {
            let held = compiled.read().ok()?;
            match &*held {
                CompiledExecution::PlannedDml(plan)
                    if plan.validity.holds(&*self.engine, epoch) =>
                {
                    return self.run_planned_dml(plan, params, ctx, in_txn);
                }
                CompiledExecution::NotOptimizable(validity)
                    if validity.holds(&*self.engine, epoch) =>
                {
                    return None
                }
                CompiledExecution::PreparedDml(prepared)
                    if prepared.validity.holds(&*self.engine, epoch) =>
                {
                    return None
                }
                // One planned for tables since changed is planned again
                CompiledExecution::Unknown
                | CompiledExecution::NotOptimizable(_)
                | CompiledExecution::PlannedDml(_)
                | CompiledExecution::PreparedDml(_) => {}
                _ => return None,
            }
        }
        // Only a run with its context plans
        let ctx = ctx?;
        let mut slot = compiled.write().ok()?;
        match &*slot {
            CompiledExecution::PlannedDml(plan) if plan.validity.holds(&*self.engine, epoch) => {
                return self.run_planned_dml(plan, params, Some(ctx), in_txn);
            }
            CompiledExecution::NotOptimizable(validity) if validity.holds(&*self.engine, epoch) => {
                return None
            }
            CompiledExecution::PreparedDml(prepared)
                if prepared.validity.holds(&*self.engine, epoch) =>
            {
                return None
            }
            CompiledExecution::Unknown
            | CompiledExecution::NotOptimizable(_)
            | CompiledExecution::PlannedDml(_)
            | CompiledExecution::PreparedDml(_) => {}
            _ => return None,
        }
        // Read before the schema: a change in between leaves this behind
        let validity = match stmt {
            DmlStatement::Update(update) => self.update_validity(update, epoch),
            DmlStatement::Delete(delete) => self.delete_validity(delete, epoch),
        };
        match self.dml_plan(stmt, validity.clone()) {
            Err(error) => Some(Err(error)),
            Ok(None) => {
                *slot = CompiledExecution::NotOptimizable(validity);
                None
            }
            Ok(Some(plan)) => {
                let plan = Arc::new(plan);
                *slot = CompiledExecution::PlannedDml(Arc::clone(&plan));
                drop(slot);
                self.run_planned_dml(&plan, params, Some(ctx), in_txn)
            }
        }
    }

    /// The plan of `stmt` when its WHERE is its table's single-column primary
    /// key equality and its write needs nothing the general path runs: no
    /// RETURNING or WITH, no foreign key to keep, no CHECK reading a column
    /// it sets. An UPDATE of the primary key is refused
    fn dml_plan(
        &self,
        stmt: DmlStatement<'_>,
        validity: PlanValidity,
    ) -> Result<Option<PlannedDml>> {
        let Some(where_clause) = stmt.where_clause() else {
            return Ok(None);
        };
        if stmt.returns_or_reads_ctes() {
            return Ok(None);
        }
        let table = stmt.table();
        let Ok(store) = self.engine.get_version_store(table) else {
            return Ok(None);
        };
        let (schema, token) = store.schema_and_token();
        let &[pk] = schema.primary_key_indices() else {
            return Ok(None);
        };
        let pk_column = &schema.columns[pk].name;
        let write = match stmt {
            DmlStatement::Update(update) => {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::compile_schema_read();
                let Some(columns) = self.planned_sets(update, &schema, (table, pk))? else {
                    return Ok(None);
                };
                DmlWrite::Update(columns)
            }
            // A parent table's delete keeps its children's keys
            DmlStatement::Delete(_) => {
                if !super::foreign_key::find_referencing_fks(&self.engine, table).is_empty() {
                    return Ok(None);
                }
                DmlWrite::Delete
            }
        };
        let Some(key) = Self::key_source(where_clause, pk_column) else {
            return Ok(None);
        };
        let sets_rows = match &write {
            DmlWrite::Update(columns) => columns
                .iter()
                .any(|c| matches!(c.value_source, UpdateValueSource::Expression(_))),
            DmlWrite::Delete => false,
        };
        Ok(Some(PlannedDml {
            sets_rows,
            table: SmartString::new(table),
            pk_column: SmartString::new(pk_column),
            schema,
            token,
            validity,
            key,
            write,
        }))
    }

    /// The columns `update` sets on `schema`'s table, whose primary key is
    /// at `pk`; None when a foreign key, a CHECK reading a set column, or a
    /// value the plan does not take leaves it to the general path
    fn planned_sets(
        &self,
        update: &UpdateStatement,
        schema: &Schema,
        (table, pk): (&str, usize),
    ) -> Result<Option<Vec<CompiledUpdateColumn>>> {
        let col_map = schema.column_index_map();
        let mut set = Vec::with_capacity(update.updates.len());
        for name in update.updates.keys() {
            let column = col_map.get(name.to_lowercase().as_str()).copied();
            // The row id is the primary key's value
            if column == Some(pk) {
                return Err(crate::core::Error::invalid_argument(format!(
                    "cannot UPDATE primary key column '{}'. Use DELETE + INSERT instead",
                    schema.columns[pk].name
                )));
            }
            set.extend(column);
        }
        if !schema.foreign_keys.is_empty()
            || !super::foreign_key::find_referencing_fks(&self.engine, table).is_empty()
        {
            return Ok(None);
        }
        let checked = match super::dml::compile_check_programs(schema) {
            Ok(checks) if checks.is_empty() => false,
            Ok(checks) => !super::dml::checks_reading(checks, |c| set.contains(&c)).is_empty(),
            Err(_) => true,
        };
        if checked {
            return Ok(None);
        }
        let mut columns = Vec::with_capacity(update.updates.len());
        for (name, expr) in &update.updates {
            let Some(&column_idx) = col_map.get(name.to_lowercase().as_str()) else {
                return Ok(None);
            };
            let Some(value_source) = Self::value_source(expr, schema.column_names_owned()) else {
                return Ok(None);
            };
            columns.push(CompiledUpdateColumn {
                column_idx,
                column_type: schema.columns[column_idx].data_type,
                value_source,
                column_vector_dims: schema.columns[column_idx].vector_dimensions,
            });
        }
        Ok(Some(columns))
    }

    /// Where the key of a WHERE `pk_column = value` comes from: a literal
    /// that is an integer, or a parameter whose value each run checks
    fn key_source(where_clause: &Expression, pk_column: &str) -> Option<PkValueSource> {
        let Expression::Infix(infix) = where_clause else {
            return None;
        };
        if infix.operator != "=" {
            return None;
        }
        let names_key = |expr: &Expression| match expr {
            Expression::Identifier(id) => id.value.eq_ignore_ascii_case(pk_column),
            Expression::QualifiedIdentifier(q) => q.name.value.eq_ignore_ascii_case(pk_column),
            _ => false,
        };
        let value = match (&*infix.left, &*infix.right) {
            (column, value) if names_key(column) => value,
            (value, column) if names_key(column) => value,
            _ => return None,
        };
        match value {
            Expression::IntegerLiteral(lit) => Some(PkValueSource::Literal(lit.value)),
            // Truncating 5.5 to 5 would write row 5 where the WHERE matches none
            Expression::FloatLiteral(lit) => {
                Self::lossless_float_key(lit.value).map(PkValueSource::Literal)
            }
            Expression::Parameter(param) => {
                Some(PkValueSource::Parameter(param.index.checked_sub(1)?))
            }
            _ => None,
        }
    }

    /// How a SET value is read: a literal, a parameter, or a row expression
    /// compiled against the table's columns; None for one holding a
    /// subquery, which only the general path resolves
    fn value_source(expr: &Expression, columns: &[String]) -> Option<UpdateValueSource> {
        Some(match expr {
            Expression::IntegerLiteral(lit) => {
                UpdateValueSource::Literal(Value::Integer(lit.value))
            }
            Expression::FloatLiteral(lit) => UpdateValueSource::Literal(Value::Float(lit.value)),
            Expression::StringLiteral(lit) => {
                UpdateValueSource::Literal(super::utils::string_literal_value(lit))
            }
            Expression::BooleanLiteral(lit) => {
                UpdateValueSource::Literal(Value::Boolean(lit.value))
            }
            Expression::NullLiteral(_) => UpdateValueSource::Literal(Value::null_unknown()),
            Expression::Prefix(prefix) if prefix.operator == "-" => match prefix.right.as_ref() {
                // An overflow is left to the VM, which reports it
                Expression::IntegerLiteral(lit) => UpdateValueSource::Literal(Value::Integer(
                    crate::core::integer::integer_neg(lit.value).ok()?,
                )),
                Expression::FloatLiteral(lit) => {
                    UpdateValueSource::Literal(Value::Float(-lit.value))
                }
                _ => return None,
            },
            Expression::Parameter(param) => {
                UpdateValueSource::Parameter(param.index.checked_sub(1)?)
            }
            _ if Self::expr_contains_subquery(expr) => return None,
            _ => UpdateValueSource::Expression(
                super::expression::compile_expression(expr, columns).ok()?,
            ),
        })
    }

    /// `plan` run for `params`: None when the key is no row id or a value
    /// does not convert to its column, which the general path reports only
    /// for a row the WHERE finds
    fn run_planned_dml(
        &self,
        plan: &PlannedDml,
        params: &[Value],
        ctx: Option<&ExecutionContext>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        if plan.sets_rows && ctx.is_none() {
            return None;
        }
        let key = match &plan.key {
            PkValueSource::Literal(key) => *key,
            PkValueSource::Parameter(at) => match params.get(*at)? {
                Value::Integer(key) => *key,
                Value::Float(key) => Self::lossless_float_key(*key)?,
                _ => return None,
            },
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::compiled_epoch_checked();
        let DmlWrite::Update(columns) = &plan.write else {
            return or_normal_path(self.write_planned_dml(
                plan,
                key,
                Vec::new(),
                &[],
                None,
                in_txn,
            ));
        };
        // Row expressions run per row inside the setter, over its old values
        let mut values = Vec::with_capacity(columns.len());
        let mut programs: Vec<SetProgram> = Vec::new();
        for column in columns {
            let value = match &column.value_source {
                UpdateValueSource::Literal(value) => value.clone(),
                // A parameter the call did not give is the general path's error
                UpdateValueSource::Parameter(at) => params.get(*at)?.clone(),
                UpdateValueSource::Expression(program) => {
                    programs.push((
                        column.column_idx,
                        program.clone(),
                        column.column_type,
                        column.column_vector_dims,
                    ));
                    continue;
                }
            };
            let coerced = value.coerce_to_type(column.column_type);
            super::dml::validate_coercion(
                &value,
                &coerced,
                &plan.schema.columns[column.column_idx].name,
                column.column_type,
                column.column_vector_dims,
            )
            .ok()?;
            values.push((column.column_idx, coerced));
        }
        or_normal_path(self.write_planned_dml(plan, key, values, &programs, ctx, in_txn))
    }

    /// The table a planned write goes through: the open transaction's, kept
    /// for its commit or rollback, when `in_txn`; otherwise a new autocommit
    /// transaction's, handed back to commit after the write
    fn pk_write_table(&self, table_name: &str, in_txn: bool) -> Result<PkWriteTable<'_>> {
        if in_txn {
            let mut active_tx = self
                .active_transaction
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let rollback = StatementRollback::new(&self.engine, active_tx.as_ref());
            if let Some(tx_state) = active_tx.as_mut() {
                let table = tx_state.transaction.get_statement_table(table_name)?;
                if !tx_state.tables.contains_key(table_name) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_statement_table(table_name)?,
                    );
                }
                return Ok((table, None, rollback));
            }
        }
        let tx = self.begin_transaction()?;
        let table = tx.get_statement_table(table_name)?;
        Ok((table, Some(tx), StatementRollback::new(&self.engine, None)))
    }

    /// The row `key` names written as `plan` says: deleted, or set to
    /// `values` and the row expressions `programs` compute from its old
    /// values, whose parameters `ctx` holds
    fn write_planned_dml(
        &self,
        plan: &PlannedDml,
        key: i64,
        values: Vec<(usize, Value)>,
        programs: &[SetProgram],
        ctx: Option<&ExecutionContext>,
        in_txn: bool,
    ) -> Result<Box<dyn QueryResult>> {
        let (mut table, autocommit, rollback) = self.pk_write_table(&plan.table, in_txn)?;
        check_compiled_token(table.as_ref(), plan.token)?;
        let mut pk_expr = ComparisonExpr::new(
            plan.pk_column.as_str(),
            crate::core::Operator::Eq,
            Value::Integer(key),
        );
        pk_expr.prepare_for_schema(&plan.schema);
        let written = match plan.write {
            DmlWrite::Delete => table.delete(Some(&pk_expr)),
            DmlWrite::Update(_) => {
                let mut vm = super::expression::ExprVM::new();
                let mut setter = |mut row: Row| -> Result<(Row, bool)> {
                    if !programs.is_empty() {
                        let mut computed = Vec::with_capacity(programs.len());
                        {
                            let mut exec_ctx = super::expression::ExecuteContext::new(&row);
                            if let Some(ctx) = ctx {
                                exec_ctx = exec_ctx.with_transaction_id(ctx.transaction_id());
                                if !ctx.params().is_empty() {
                                    exec_ctx = exec_ctx.with_params(ctx.params());
                                }
                            }
                            for (idx, program, column_type, vector_dims) in programs {
                                let value = vm.execute(program, &exec_ctx)?;
                                let coerced = value.coerce_to_type(*column_type);
                                super::dml::validate_coercion(
                                    &value,
                                    &coerced,
                                    &plan.schema.columns[*idx].name,
                                    *column_type,
                                    *vector_dims,
                                )?;
                                computed.push((*idx, coerced));
                            }
                        }
                        for (idx, value) in computed {
                            let _ = row.set(idx, value);
                        }
                    }
                    for (idx, value) in &values {
                        let _ = row.set(*idx, value.clone());
                    }
                    Ok((row, true))
                };
                table.update(Some(&pk_expr), &mut setter)
            }
        };
        let rows_affected = rollback.finish(written)?;
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(&plan.table);
        }
        drop(table);
        if let Some(mut tx) = autocommit {
            tx.commit()?;
        }
        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// The plan of `stmt` as EXPLAIN shows it when it reads the row its
    /// primary key names, with the read a run whose key names no row id
    /// takes; None for any other statement
    pub(crate) fn explain_dml(&self, stmt: DmlStatement<'_>) -> Option<Vec<String>> {
        let epoch = self.engine.schema_epoch();
        let validity = match stmt {
            DmlStatement::Update(update) => self.update_validity(update, epoch),
            DmlStatement::Delete(delete) => self.delete_validity(delete, epoch),
        };
        let plan = self.dml_plan(stmt, validity).ok()??;
        let where_clause = stmt.where_clause()?;
        let table = &plan.table;
        let head = match stmt {
            DmlStatement::Update(update) => {
                let col_map = plan.schema.column_index_map();
                let mut sets: Vec<(usize, String)> = update
                    .updates
                    .iter()
                    .filter_map(|(name, expr)| {
                        let at = *col_map.get(name.to_lowercase().as_str())?;
                        Some((at, format!("{} = {expr}", plan.schema.columns[at].name)))
                    })
                    .collect();
                sets.sort_unstable_by_key(|(at, _)| *at);
                let sets: Vec<String> = sets.into_iter().map(|(_, set)| set).collect();
                format!("Update {table} SET {}", sets.join(", "))
            }
            DmlStatement::Delete(_) => format!("Delete {table}"),
        };
        let key = match &plan.key {
            PkValueSource::Literal(key) => key.to_string(),
            PkValueSource::Parameter(at) => format!("${}", at + 1),
        };
        let mut lines = vec![head, format!("  PkGet {table} {} = {key}", plan.pk_column)];
        // A key that names no row id leaves the WHERE to the table's write,
        // whose index rule reads a key equality by the key
        let scan = ScanChoice::Index(IndexPlan::Key);
        for (at, line) in scan
            .explain(table, Some(where_clause))
            .into_iter()
            .enumerate()
        {
            lines.push(match at {
                0 => format!("  else {line}"),
                _ => format!("  {line}"),
            });
        }
        Some(lines)
    }

    /// Conservative subquery detection for SET expressions; anything
    /// unrecognized that can carry one is treated as containing one
    pub(crate) fn expr_contains_subquery(expr: &Expression) -> bool {
        match expr {
            Expression::ScalarSubquery(_)
            | Expression::Exists(_)
            | Expression::AllAny(_)
            | Expression::SubquerySource(_)
            | Expression::In(_)
            | Expression::InHashSet(_)
            | Expression::QuantifiedSet(_)
            | Expression::CteReference(_) => true,
            Expression::Infix(i) => {
                Self::expr_contains_subquery(&i.left) || Self::expr_contains_subquery(&i.right)
            }
            Expression::Prefix(p) => Self::expr_contains_subquery(&p.right),
            Expression::FunctionCall(f) => {
                f.arguments.iter().any(Self::expr_contains_subquery)
                    || f.filter
                        .as_ref()
                        .is_some_and(|e| Self::expr_contains_subquery(e))
                    || f.order_by
                        .iter()
                        .any(|ob| Self::expr_contains_subquery(&ob.expression))
            }
            Expression::Cast(c) => Self::expr_contains_subquery(&c.expr),
            Expression::Aliased(a) => Self::expr_contains_subquery(&a.expression),
            Expression::Case(c) => {
                c.value
                    .as_ref()
                    .is_some_and(|v| Self::expr_contains_subquery(v))
                    || c.when_clauses.iter().any(|w| {
                        Self::expr_contains_subquery(&w.condition)
                            || Self::expr_contains_subquery(&w.then_result)
                    })
                    || c.else_value
                        .as_ref()
                        .is_some_and(|v| Self::expr_contains_subquery(v))
            }
            Expression::Between(b) => {
                Self::expr_contains_subquery(&b.expr)
                    || Self::expr_contains_subquery(&b.lower)
                    || Self::expr_contains_subquery(&b.upper)
            }
            Expression::Like(l) => {
                Self::expr_contains_subquery(&l.left)
                    || Self::expr_contains_subquery(&l.pattern)
                    || l.escape
                        .as_ref()
                        .is_some_and(|e| Self::expr_contains_subquery(e))
            }
            Expression::List(l) => l.elements.iter().any(Self::expr_contains_subquery),
            Expression::ExpressionList(l) => l.expressions.iter().any(Self::expr_contains_subquery),
            // Simple leaves cannot carry a subquery
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_) => false,
            // Unknown container: be conservative
            _ => true,
        }
    }
}
