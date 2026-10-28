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

//! Fast-path execution for simple PK-based UPDATE and DELETE operations
//!
//! This module provides optimized execution paths for simple DML like:
//! - `UPDATE table SET col = val WHERE pk = $1`
//! - `DELETE FROM table WHERE pk = $1`
//!
//! By detecting these patterns early and bypassing the full executor overhead
//! (subquery checking, memory filter setup, expression compilation), we can
//! reduce per-operation overhead significantly.

use std::sync::RwLock;

use crate::common::{CompactArc, SmartString};
use crate::core::{Result, Row, Schema, Value};
use crate::parser::ast::{DeleteStatement, Expression, UpdateStatement};
use crate::storage::expression::{ComparisonExpr, Expression as StorageExpression};
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::transaction::StatementTable;
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};

use super::context::ExecutionContext;
use super::dml::StatementRollback;
use super::query_cache::{
    check_compiled_token, or_normal_path, CompiledExecution, CompiledPkDelete, CompiledPkUpdate,
    CompiledUpdateColumn, PkValueSource, UpdateValueSource,
};
use super::result::ExecResult;
use super::Executor;

/// The table a compiled PK write goes through, the autocommit transaction
/// to commit after it if any, and the rollback of the statement on failure
type PkWriteTable<'a> = (
    Box<dyn Table>,
    Option<Box<dyn Transaction>>,
    StatementRollback<'a>,
);

impl Executor {
    /// Try to execute an UPDATE using pre-compiled state, inside the open
    /// transaction when `in_txn`
    pub(crate) fn try_fast_pk_update_compiled(
        &self,
        stmt: &UpdateStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable(validity)
                    if validity.holds(&*self.engine, self.engine.schema_epoch()) =>
                {
                    return None
                }
                CompiledExecution::PkUpdate(update) => {
                    // Fast validation using schema epoch (~1ns vs ~7ns for HashMap lookup)
                    if update
                        .validity
                        .holds(&*self.engine, self.engine.schema_epoch())
                    {
                        // Fast path: extract PK value and execute
                        let pk_value =
                            self.extract_pk_value_from_source(&update.pk_value_source, ctx)?;
                        #[cfg(feature = "test-failpoints")]
                        crate::test_failpoints::compiled_epoch_checked();
                        return self.execute_compiled_pk_update(update, pk_value, ctx, in_txn);
                    }
                    // Epoch changed - fall through to recompile
                }
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                _ => return None, // Different type of compiled execution
            }
        }

        // First execution or schema changed - compile and cache
        self.compile_and_execute_pk_update(stmt, ctx, compiled, in_txn)
    }

    /// Try to execute a DELETE using pre-compiled state, inside the open
    /// transaction when `in_txn`
    pub(crate) fn try_fast_pk_delete_compiled(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable(validity)
                    if validity.holds(&*self.engine, self.engine.schema_epoch()) =>
                {
                    return None
                }
                CompiledExecution::PkDelete(delete) => {
                    // Fast validation using schema epoch (~1ns vs ~7ns for HashMap lookup)
                    if delete
                        .validity
                        .holds(&*self.engine, self.engine.schema_epoch())
                    {
                        // Fast path: extract PK value and execute
                        let pk_value =
                            self.extract_pk_value_from_source(&delete.pk_value_source, ctx)?;
                        #[cfg(feature = "test-failpoints")]
                        crate::test_failpoints::compiled_epoch_checked();
                        return or_normal_path(
                            self.execute_compiled_pk_delete(delete, pk_value, in_txn),
                        );
                    }
                    // Epoch changed - fall through to recompile
                }
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                _ => return None, // Different type of compiled execution
            }
        }

        // First execution or schema changed - compile and cache
        self.compile_and_execute_pk_delete(stmt, ctx, compiled, in_txn)
    }

    // ============================================================================
    // HELPER METHODS
    // ============================================================================

    /// Extract PK equality value from WHERE clause
    /// Returns (pk_value, pk_source) if WHERE is `pk_col = literal` or `pk_col = $param`
    fn extract_pk_equality_value(
        &self,
        expr: &Expression,
        pk_column: &str,
        ctx: &ExecutionContext,
    ) -> Option<(i64, PkValueSource)> {
        match expr {
            Expression::Infix(infix) => {
                if infix.operator != "=" {
                    return None;
                }

                // Try column = value pattern
                if let Some((col, val, source)) =
                    self.extract_col_eq_val_dml(&infix.left, &infix.right, ctx)
                {
                    if col.eq_ignore_ascii_case(pk_column) {
                        return Some((val, source));
                    }
                }

                // Try value = column pattern
                if let Some((col, val, source)) =
                    self.extract_col_eq_val_dml(&infix.right, &infix.left, ctx)
                {
                    if col.eq_ignore_ascii_case(pk_column) {
                        return Some((val, source));
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Extract column name, integer value, and value source from col = val pattern
    fn extract_col_eq_val_dml(
        &self,
        col_expr: &Expression,
        val_expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<(String, i64, PkValueSource)> {
        // Get column name
        let col_name = match col_expr {
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(q) => q.name.value.to_string(),
            _ => return None,
        };

        // Get integer value and source
        let (pk_value, pk_value_source) = match val_expr {
            Expression::IntegerLiteral(lit) => (lit.value, PkValueSource::Literal(lit.value)),
            Expression::FloatLiteral(lit) => {
                // Only lossless float keys qualify: truncating 5.5 to 5
                // would mutate row 5 where correct execution matches nothing.
                let v = Self::lossless_float_key(lit.value)?;
                (v, PkValueSource::Literal(v))
            }
            Expression::Parameter(param) => {
                // Slots are 1-indexed, the array is 0-indexed
                let params = ctx.params();
                let param_idx = param.index.checked_sub(1)?;
                let pk_value = match params.get(param_idx)? {
                    Value::Integer(i) => *i,
                    Value::Float(f) => Self::lossless_float_key(*f)?,
                    _ => return None,
                };
                (pk_value, PkValueSource::Parameter(param_idx))
            }
            _ => return None,
        };

        Some((col_name, pk_value, pk_value_source))
    }

    /// Extract PK value from pre-compiled source
    fn extract_pk_value_from_source(
        &self,
        source: &PkValueSource,
        ctx: &ExecutionContext,
    ) -> Option<i64> {
        Self::extract_pk_value_from_params(source, ctx.params())
    }

    /// Extract PK value from params slice directly (avoids ExecutionContext overhead)
    #[inline]
    fn extract_pk_value_from_params(source: &PkValueSource, params: &[Value]) -> Option<i64> {
        match source {
            PkValueSource::Literal(v) => Some(*v),
            PkValueSource::Parameter(idx) => {
                if *idx >= params.len() {
                    return None;
                }
                match &params[*idx] {
                    Value::Integer(i) => Some(*i),
                    Value::Float(f) => Self::lossless_float_key(*f),
                    _ => None,
                }
            }
        }
    }

    /// Extract update value from params slice directly
    #[inline]
    fn extract_update_value_from_slice(
        source: &UpdateValueSource,
        params: &[Value],
    ) -> Option<Value> {
        match source {
            UpdateValueSource::Literal(v) => Some(v.clone()),
            UpdateValueSource::Parameter(idx) => params.get(*idx).cloned(),
            UpdateValueSource::Expression(_) => None, // Needs a row context; normal path
        }
    }

    /// Try fast PK update with borrowed params slice (avoids Arc allocation)
    pub(crate) fn try_fast_pk_update_with_params(
        &self,
        _stmt: &UpdateStatement,
        params: &[Value],
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Try read lock first - check if already compiled
        let compiled_guard = compiled.read().ok()?;
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(_) => None,
            CompiledExecution::PkUpdate(update) => {
                // Fast validation using schema epoch
                if update
                    .validity
                    .holds(&*self.engine, self.engine.schema_epoch())
                {
                    let pk_value =
                        Self::extract_pk_value_from_params(&update.pk_value_source, params)?;
                    // Extract update values
                    let mut updates = Vec::with_capacity(update.updates.len());
                    for u in &update.updates {
                        let value = Self::extract_update_value_from_slice(&u.value_source, params)?;
                        let coerced = value.coerce_to_type(u.column_type);
                        if super::dml::validate_coercion(
                            &value,
                            &coerced,
                            &update.schema.columns[u.column_idx].name,
                            u.column_type,
                            u.column_vector_dims,
                        )
                        .is_err()
                        {
                            // Let the normal path produce the full error
                            return None;
                        }
                        updates.push((u.column_idx, coerced));
                    }
                    // Clone only what we need (cheap: SmartString + Arc)
                    let table_name = update.table_name.clone();
                    let pk_column_name = update.pk_column_name.clone();
                    let schema = update.schema.clone();
                    let token = update.token;
                    drop(compiled_guard);
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::compiled_epoch_checked();
                    // Expression sources bail above, so no row programs
                    // and no parameters are needed here
                    return or_normal_path(self.execute_pk_update_minimal(
                        &table_name,
                        &pk_column_name,
                        &schema,
                        token,
                        pk_value,
                        updates,
                        &[],
                        None,
                        false,
                    ));
                }
                None // Epoch changed, use normal path
            }
            CompiledExecution::Unknown => None,
            _ => None,
        }
    }

    /// Try fast PK delete with borrowed params slice (avoids Arc allocation)
    pub(crate) fn try_fast_pk_delete_with_params(
        &self,
        _stmt: &DeleteStatement,
        params: &[Value],
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Try read lock first - check if already compiled
        let compiled_guard = compiled.read().ok()?;
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(_) => None,
            CompiledExecution::PkDelete(delete) => {
                // Fast validation using schema epoch
                if delete
                    .validity
                    .holds(&*self.engine, self.engine.schema_epoch())
                {
                    let pk_value =
                        Self::extract_pk_value_from_params(&delete.pk_value_source, params)?;
                    // Clone only what we need (cheap: SmartString + Arc)
                    let table_name = delete.table_name.clone();
                    let pk_column_name = delete.pk_column_name.clone();
                    let schema = delete.schema.clone();
                    let token = delete.token;
                    drop(compiled_guard);
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::compiled_epoch_checked();
                    return or_normal_path(self.execute_pk_delete_minimal(
                        &table_name,
                        &pk_column_name,
                        &schema,
                        token,
                        pk_value,
                        false,
                    ));
                }
                None // Epoch changed, use normal path
            }
            CompiledExecution::Unknown => None,
            _ => None,
        }
    }

    /// The table a compiled PK write goes through: the open transaction's,
    /// kept for its commit or rollback, when `in_txn`; otherwise a new
    /// autocommit transaction's, handed back to commit after the write
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

    /// Execute PK update with minimal data (avoids cloning CompiledPkUpdate).
    /// `ctx` carries the parameters row expressions read; None without any
    #[allow(clippy::too_many_arguments)]
    fn execute_pk_update_minimal(
        &self,
        table_name: &str,
        pk_column_name: &str,
        schema: &CompactArc<Schema>,
        token: LayoutToken,
        pk_value: i64,
        updates: Vec<(usize, Value)>,
        expr_updates: &[(
            usize,
            super::expression::SharedProgram,
            crate::core::DataType,
            u16,
        )],
        ctx: Option<&ExecutionContext>,
        in_txn: bool,
    ) -> Result<Box<dyn QueryResult>> {
        let (mut table, autocommit, rollback) = self.pk_write_table(table_name, in_txn)?;
        check_compiled_token(table.as_ref(), token)?;

        // Build WHERE expression for PK lookup
        let mut pk_expr = ComparisonExpr::new(
            pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(schema);

        // Execute update with simple setter. Row expressions all read the
        // ORIGINAL row (SET a = b, b = a swaps), then everything applies.
        let mut vm = super::expression::ExprVM::new();
        let mut setter = |mut row: Row| -> Result<(Row, bool)> {
            if !expr_updates.is_empty() {
                let mut computed = Vec::with_capacity(expr_updates.len());
                {
                    let mut exec_ctx = super::expression::ExecuteContext::new(&row);
                    if let Some(ctx) = ctx {
                        exec_ctx = exec_ctx.with_transaction_id(ctx.transaction_id());
                        if !ctx.params().is_empty() {
                            exec_ctx = exec_ctx.with_params(ctx.params());
                        }
                    }
                    for (idx, program, col_type, vector_dims) in expr_updates {
                        let value = vm.execute_cow(program, &exec_ctx)?;
                        let coerced = value.coerce_to_type(*col_type);
                        super::dml::validate_coercion(
                            &value,
                            &coerced,
                            &schema.columns[*idx].name,
                            *col_type,
                            *vector_dims,
                        )?;
                        computed.push((*idx, coerced));
                    }
                }
                for (idx, value) in computed {
                    let _ = row.set(idx, value);
                }
            }
            for (idx, new_value) in &updates {
                let _ = row.set(*idx, new_value.clone());
            }
            Ok((row, true))
        };

        let rows_affected = rollback.finish(table.update(Some(&pk_expr), &mut setter))?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        drop(table);
        if let Some(mut tx) = autocommit {
            tx.commit()?;
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Execute PK delete with minimal data (avoids cloning CompiledPkDelete)
    fn execute_pk_delete_minimal(
        &self,
        table_name: &str,
        pk_column_name: &str,
        schema: &CompactArc<Schema>,
        token: LayoutToken,
        pk_value: i64,
        in_txn: bool,
    ) -> Result<Box<dyn QueryResult>> {
        let (mut table, autocommit, rollback) = self.pk_write_table(table_name, in_txn)?;
        check_compiled_token(table.as_ref(), token)?;

        // Build WHERE expression for PK lookup
        let mut pk_expr = ComparisonExpr::new(
            pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(schema);

        let rows_affected = rollback.finish(table.delete(Some(&pk_expr)))?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        drop(table);
        if let Some(mut tx) = autocommit {
            tx.commit()?;
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    // ============================================================================
    // EXECUTION METHODS
    // ============================================================================

    /// Execute a compiled PK update (extracts values from ctx then delegates
    /// to core impl). None when a value does not convert to its column: the
    /// normal path reports that only for a row the WHERE finds
    fn execute_compiled_pk_update(
        &self,
        compiled: &CompiledPkUpdate,
        pk_value: i64,
        ctx: &ExecutionContext,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Extract constant values; row expressions run per update inside
        // the setter (they read the current row, e.g. SET n = n + 1)
        let mut updates = Vec::with_capacity(compiled.updates.len());
        let mut expr_updates: Vec<(
            usize,
            super::expression::SharedProgram,
            crate::core::DataType,
            u16,
        )> = Vec::new();
        for u in &compiled.updates {
            let value = match &u.value_source {
                UpdateValueSource::Literal(v) => v.clone(),
                // A parameter the call did not give is the general path's error
                UpdateValueSource::Parameter(idx) => ctx.params().get(*idx)?.clone(),
                UpdateValueSource::Expression(program) => {
                    expr_updates.push((
                        u.column_idx,
                        program.clone(),
                        u.column_type,
                        u.column_vector_dims,
                    ));
                    continue;
                }
            };
            let coerced = value.coerce_to_type(u.column_type);
            super::dml::validate_coercion(
                &value,
                &coerced,
                &compiled.schema.columns[u.column_idx].name,
                u.column_type,
                u.column_vector_dims,
            )
            .ok()?;
            updates.push((u.column_idx, coerced));
        }

        or_normal_path(self.execute_pk_update_minimal(
            &compiled.table_name,
            &compiled.pk_column_name,
            &compiled.schema,
            compiled.token,
            pk_value,
            updates,
            &expr_updates,
            Some(ctx),
            in_txn,
        ))
    }

    /// Execute a compiled PK delete (delegates to core impl)
    fn execute_compiled_pk_delete(
        &self,
        compiled: &CompiledPkDelete,
        pk_value: i64,
        in_txn: bool,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_pk_delete_minimal(
            &compiled.table_name,
            &compiled.pk_column_name,
            &compiled.schema,
            compiled.token,
            pk_value,
            in_txn,
        )
    }

    // ============================================================================
    // COMPILE AND EXECUTE METHODS
    // ============================================================================

    /// Compile and execute a PK update, caching the compiled state
    fn compile_and_execute_pk_update(
        &self,
        stmt: &UpdateStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check after acquiring lock
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(validity)
                if validity.holds(&*self.engine, self.engine.schema_epoch()) =>
            {
                return None
            }
            CompiledExecution::PkUpdate(update)
                if update
                    .validity
                    .holds(&*self.engine, self.engine.schema_epoch()) =>
            {
                let pk_value = self.extract_pk_value_from_source(&update.pk_value_source, ctx)?;
                return self.execute_compiled_pk_update(update, pk_value, ctx, in_txn);
            }
            CompiledExecution::PkUpdate(_)
            | CompiledExecution::NotOptimizable(_)
            | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Read before the schema: a change in between leaves this behind
        let validity = self.update_validity(stmt, self.engine.schema_epoch());

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() || stmt.with.is_some() {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let (schema, token) = match self.engine.get_version_store(table_name) {
            Ok(store) => store.schema_and_token(),
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(validity);
                return None;
            }
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::compile_schema_read();

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Reject UPDATE on primary key column (row_id == pk_value invariant)
        {
            let col_map = schema.column_index_map();
            for col_name in stmt.updates.keys() {
                let col_lower = col_name.to_lowercase();
                if col_map.get(col_lower.as_str()).copied() == Some(pk_idx) {
                    return Some(Err(crate::core::Error::invalid_argument(format!(
                        "cannot UPDATE primary key column '{}'. Use DELETE + INSERT instead",
                        pk_column
                    ))));
                }
            }
        }

        // Bail if table has FK constraints (child table) or is referenced by other tables (parent table)
        // FK enforcement requires cross-table lookups — fall back to normal path
        if !schema.foreign_keys.is_empty()
            || !super::foreign_key::find_referencing_fks(&self.engine, table_name).is_empty()
        {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }

        // Bail if this update must run a CHECK, a column's or the table's:
        // it runs on the general path
        {
            let checked = match super::dml::compile_check_programs(&schema) {
                Ok(checks) if checks.is_empty() => false,
                Ok(checks) => {
                    let col_map = schema.column_index_map();
                    let updated: Vec<usize> = stmt
                        .updates
                        .keys()
                        .filter_map(|name| col_map.get(name.to_lowercase().as_str()).copied())
                        .collect();
                    !super::dml::checks_reading(checks, |c| updated.contains(&c)).is_empty()
                }
                Err(_) => true,
            };
            if checked {
                *compiled_guard = CompiledExecution::NotOptimizable(validity);
                return None;
            }
        }

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(validity);
                    return None;
                }
            };

        // Extract update value sources
        let col_map = schema.column_index_map();
        let mut compiled_updates = Vec::with_capacity(stmt.updates.len());
        for (col_name, expr) in &stmt.updates {
            let col_lower = col_name.to_lowercase();
            let col_idx = match col_map.get(col_lower.as_str()) {
                Some(&idx) => idx,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(validity);
                    return None;
                }
            };
            let col_type = schema.columns[col_idx].data_type;

            let value_source = match self.extract_value_source(expr, schema.column_names_owned()) {
                Some(s) => s,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(validity);
                    return None;
                }
            };

            compiled_updates.push(CompiledUpdateColumn {
                column_idx: col_idx,
                column_type: col_type,
                value_source,
                column_vector_dims: schema.columns[col_idx].vector_dimensions,
            });
        }

        // Build compiled state
        let compiled_update = CompiledPkUpdate {
            table_name: SmartString::new(table_name),
            schema: CompactArc::new((*schema).clone()),
            pk_column_name: SmartString::new(pk_column),
            pk_value_source: pk_source,
            updates: compiled_updates,
            validity,
            token,
        };

        *compiled_guard = CompiledExecution::PkUpdate(compiled_update.clone());
        drop(compiled_guard);

        // Execute
        self.execute_compiled_pk_update(&compiled_update, pk_value, ctx, in_txn)
    }

    /// Compile and execute a PK delete, caching the compiled state
    fn compile_and_execute_pk_delete(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check after acquiring lock
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(validity)
                if validity.holds(&*self.engine, self.engine.schema_epoch()) =>
            {
                return None
            }
            CompiledExecution::PkDelete(delete)
                if delete
                    .validity
                    .holds(&*self.engine, self.engine.schema_epoch()) =>
            {
                let pk_value = self.extract_pk_value_from_source(&delete.pk_value_source, ctx)?;
                return or_normal_path(self.execute_compiled_pk_delete(delete, pk_value, in_txn));
            }
            CompiledExecution::PkDelete(_)
            | CompiledExecution::NotOptimizable(_)
            | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Read before the schema: a change in between leaves this behind
        let validity = self.delete_validity(stmt, self.engine.schema_epoch());

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() || stmt.with.is_some() {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let (schema, token) = match self.engine.get_version_store(table_name) {
            Ok(store) => store.schema_and_token(),
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(validity);
                return None;
            }
        };

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Bail if this table is referenced by child tables (FK enforcement needed)
        if !super::foreign_key::find_referencing_fks(&self.engine, table_name).is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable(validity);
            return None;
        }

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(validity);
                    return None;
                }
            };

        // Build compiled state
        let compiled_delete = CompiledPkDelete {
            table_name: SmartString::new(table_name),
            schema: CompactArc::new((*schema).clone()),
            pk_column_name: SmartString::new(pk_column),
            pk_value_source: pk_source,
            validity,
            token,
        };

        *compiled_guard = CompiledExecution::PkDelete(compiled_delete.clone());
        drop(compiled_guard);

        // Execute
        or_normal_path(self.execute_compiled_pk_delete(&compiled_delete, pk_value, in_txn))
    }

    /// Extract value source (literal or parameter) from expression
    fn extract_value_source(
        &self,
        expr: &Expression,
        schema_columns: &[String],
    ) -> Option<UpdateValueSource> {
        match expr {
            Expression::IntegerLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Integer(lit.value)))
            }
            Expression::FloatLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Float(lit.value)))
            }
            Expression::StringLiteral(lit) => Some(UpdateValueSource::Literal(
                super::utils::string_literal_value(lit),
            )),
            Expression::BooleanLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Boolean(lit.value)))
            }
            Expression::NullLiteral(_) => Some(UpdateValueSource::Literal(Value::null_unknown())),
            Expression::Prefix(prefix) if prefix.operator == "-" => match prefix.right.as_ref() {
                // An overflow is left to the VM, which reports it
                Expression::IntegerLiteral(lit) => crate::core::integer::integer_neg(lit.value)
                    .ok()
                    .map(|neg| UpdateValueSource::Literal(Value::Integer(neg))),
                Expression::FloatLiteral(lit) => {
                    Some(UpdateValueSource::Literal(Value::Float(-lit.value)))
                }
                _ => None,
            },
            Expression::Parameter(param) => {
                Some(UpdateValueSource::Parameter(param.index.checked_sub(1)?))
            }
            // Row expressions (e.g. SET n = n + 1) compile once per plan
            // against the schema columns. Subquery-bearing expressions must
            // take the normal path: the compiler emits placeholder ops for
            // them that only the full planner can resolve.
            _ => {
                if Self::expr_contains_subquery(expr) {
                    return None;
                }
                use super::expression::compile_expression;
                compile_expression(expr, schema_columns)
                    .ok()
                    .map(UpdateValueSource::Expression)
            }
        }
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
