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

use std::sync::{Arc, RwLock};

use crate::core::{Result, Row, Value};
use crate::parser::ast::{DeleteStatement, Expression, UpdateStatement};
use crate::storage::expression::{ComparisonExpr, Expression as StorageExpression};
use crate::storage::traits::{Engine, QueryResult};

use super::context::{invalidate_semi_join_cache_for_table, ExecutionContext};
use super::query_cache::{
    CompiledExecution, CompiledPkDelete, CompiledPkUpdate, CompiledUpdateColumn, PkValueSource,
    UpdateValueSource,
};
use super::result::ExecResult;
use super::Executor;

impl Executor {
    /// Try to execute an UPDATE using pre-compiled state
    pub(crate) fn try_fast_pk_update_compiled(
        &self,
        stmt: &UpdateStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Quick reject: explicit transaction (use try_lock for fast rejection)
        {
            let active_tx = match self.active_transaction.try_lock() {
                Ok(guard) => guard,
                Err(_) => return None, // Lock contention - fall back to normal path
            };
            if active_tx.is_some() {
                return None;
            }
        }

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable => return None,
                CompiledExecution::PkUpdate(update) => {
                    // Fast validation using schema epoch (~1ns vs ~7ns for HashMap lookup)
                    if self.engine.schema_epoch() == update.cached_epoch {
                        // Fast path: extract PK value and execute
                        let pk_value =
                            self.extract_pk_value_from_source(&update.pk_value_source, ctx)?;
                        return Some(self.execute_compiled_pk_update(update, pk_value, ctx));
                    }
                    // Epoch changed - fall through to recompile
                }
                CompiledExecution::Unknown => {} // Fall through to compile
                _ => return None,                // Different type of compiled execution
            }
        }

        // First execution or schema changed - compile and cache
        self.compile_and_execute_pk_update(stmt, ctx, compiled)
    }

    /// Try to execute a DELETE using pre-compiled state
    pub(crate) fn try_fast_pk_delete_compiled(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Quick reject: explicit transaction (use try_lock for fast rejection)
        {
            let active_tx = match self.active_transaction.try_lock() {
                Ok(guard) => guard,
                Err(_) => return None, // Lock contention - fall back to normal path
            };
            if active_tx.is_some() {
                return None;
            }
        }

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable => return None,
                CompiledExecution::PkDelete(delete) => {
                    // Fast validation using schema epoch (~1ns vs ~7ns for HashMap lookup)
                    if self.engine.schema_epoch() == delete.cached_epoch {
                        // Fast path: extract PK value and execute
                        let pk_value =
                            self.extract_pk_value_from_source(&delete.pk_value_source, ctx)?;
                        return Some(self.execute_compiled_pk_delete(delete, pk_value));
                    }
                    // Epoch changed - fall through to recompile
                }
                CompiledExecution::Unknown => {} // Fall through to compile
                _ => return None,                // Different type of compiled execution
            }
        }

        // First execution or schema changed - compile and cache
        self.compile_and_execute_pk_delete(stmt, ctx, compiled)
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
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(q) => q.name.value.clone(),
            _ => return None,
        };

        // Get integer value and source
        let (pk_value, pk_value_source) = match val_expr {
            Expression::IntegerLiteral(lit) => (lit.value, PkValueSource::Literal(lit.value)),
            Expression::FloatLiteral(lit) => {
                let v = lit.value as i64;
                (v, PkValueSource::Literal(v))
            }
            Expression::Parameter(param) => {
                let params = ctx.params();
                let param_idx = if param.index > 0 {
                    param.index - 1
                } else {
                    param.index
                };
                if param_idx >= params.len() {
                    return None;
                }
                let pk_value = match &params[param_idx] {
                    Value::Integer(i) => *i,
                    Value::Float(f) => *f as i64,
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
                    Value::Float(f) => Some(*f as i64),
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
            CompiledExecution::NotOptimizable => None,
            CompiledExecution::PkUpdate(update) => {
                // Fast validation using schema epoch
                if self.engine.schema_epoch() == update.cached_epoch {
                    let pk_value =
                        Self::extract_pk_value_from_params(&update.pk_value_source, params)?;
                    // Extract update values
                    let mut updates = Vec::with_capacity(update.updates.len());
                    for u in &update.updates {
                        let value = Self::extract_update_value_from_slice(&u.value_source, params)?;
                        updates.push((u.column_idx, value.coerce_to_type(u.column_type)));
                    }
                    // Clone what we need before dropping guard
                    let update_clone = update.clone();
                    drop(compiled_guard);
                    return Some(self.execute_pk_update_direct(&update_clone, pk_value, updates));
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
            CompiledExecution::NotOptimizable => None,
            CompiledExecution::PkDelete(delete) => {
                // Fast validation using schema epoch
                if self.engine.schema_epoch() == delete.cached_epoch {
                    let pk_value =
                        Self::extract_pk_value_from_params(&delete.pk_value_source, params)?;
                    // Clone what we need before dropping guard
                    let delete_clone = delete.clone();
                    drop(compiled_guard);
                    return Some(self.execute_compiled_pk_delete(&delete_clone, pk_value));
                }
                None // Epoch changed, use normal path
            }
            CompiledExecution::Unknown => None,
            _ => None,
        }
    }

    /// Execute PK update directly with pre-extracted values (no ExecutionContext needed)
    fn execute_pk_update_direct(
        &self,
        compiled: &CompiledPkUpdate,
        pk_value: i64,
        updates: Vec<(usize, Value)>,
    ) -> Result<Box<dyn QueryResult>> {
        // Create auto-commit transaction
        let tx = self.engine.begin_transaction()?;
        let mut table = tx.get_table(&compiled.table_name)?;

        // Build WHERE expression for PK lookup
        let mut pk_expr = ComparisonExpr::new(
            &compiled.pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(&compiled.schema);

        // Execute update with simple setter
        let mut setter = |mut row: Row| -> (Row, bool) {
            for (idx, new_value) in &updates {
                let _ = row.set(*idx, new_value.clone());
            }
            (row, true)
        };

        let rows_affected = table.update(Some(&pk_expr), &mut setter)?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(&compiled.table_name);
            invalidate_semi_join_cache_for_table(&compiled.table_name);
        }

        // Commit
        drop(table);
        let mut tx = tx;
        tx.commit()?;

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    // ============================================================================
    // EXECUTION METHODS
    // ============================================================================

    /// Execute a compiled PK update
    fn execute_compiled_pk_update(
        &self,
        compiled: &CompiledPkUpdate,
        pk_value: i64,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Create auto-commit transaction
        let tx = self.engine.begin_transaction()?;
        let mut table = tx.get_table(&compiled.table_name)?;

        // Build WHERE expression for PK lookup using cached column name
        let mut pk_expr = ComparisonExpr::new(
            &compiled.pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(&compiled.schema);

        // Extract values from compiled sources (pre-allocate for typical 1-3 column updates)
        let mut updates = Vec::with_capacity(compiled.updates.len());
        for u in &compiled.updates {
            let value = match &u.value_source {
                UpdateValueSource::Literal(v) => v.clone(),
                UpdateValueSource::Parameter(idx) => {
                    let params = ctx.params();
                    match params.get(*idx) {
                        Some(v) => v.clone(),
                        None => continue,
                    }
                }
            };
            updates.push((u.column_idx, value.coerce_to_type(u.column_type)));
        }

        // Execute update with simple setter
        let mut setter = |mut row: Row| -> (Row, bool) {
            for (idx, new_value) in &updates {
                let _ = row.set(*idx, new_value.clone());
            }
            (row, true)
        };

        let rows_affected = table.update(Some(&pk_expr), &mut setter)?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(&compiled.table_name);
            invalidate_semi_join_cache_for_table(&compiled.table_name);
        }

        // Commit
        drop(table);
        let mut tx = tx;
        tx.commit()?;

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Execute a compiled PK delete
    fn execute_compiled_pk_delete(
        &self,
        compiled: &CompiledPkDelete,
        pk_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        // Create auto-commit transaction
        let tx = self.engine.begin_transaction()?;
        let mut table = tx.get_table(&compiled.table_name)?;

        // Build WHERE expression for PK lookup using cached schema and column name
        let mut pk_expr = ComparisonExpr::new(
            &compiled.pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(&compiled.schema);

        // Execute delete
        let rows_affected = table.delete(Some(&pk_expr))?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(&compiled.table_name);
            invalidate_semi_join_cache_for_table(&compiled.table_name);
        }

        // Commit
        drop(table);
        let mut tx = tx;
        tx.commit()?;

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
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
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check after acquiring lock
        match &*compiled_guard {
            CompiledExecution::NotOptimizable => return None,
            CompiledExecution::PkUpdate(update) => {
                let pk_value = self.extract_pk_value_from_source(&update.pk_value_source, ctx)?;
                return Some(self.execute_compiled_pk_update(update, pk_value, ctx));
            }
            CompiledExecution::Unknown => {} // Continue with compilation
            _ => return None,
        }

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let schema = match self.engine.get_table_schema(table_name) {
            Ok(s) => s,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable;
                return None;
            }
        };

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable;
                    return None;
                }
            };

        // Extract update value sources
        let mut compiled_updates = Vec::with_capacity(stmt.updates.len());
        for (col_name, expr) in &stmt.updates {
            let col_idx = match schema
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(col_name))
            {
                Some(idx) => idx,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable;
                    return None;
                }
            };
            let col_type = schema.columns[col_idx].data_type;

            let value_source = match self.extract_value_source(expr) {
                Some(s) => s,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable;
                    return None;
                }
            };

            compiled_updates.push(CompiledUpdateColumn {
                column_idx: col_idx,
                column_type: col_type,
                value_source,
            });
        }

        // Build compiled state
        let compiled_update = CompiledPkUpdate {
            table_name: table_name.clone(),
            schema: Arc::new((*schema).clone()),
            pk_column_name: pk_column.clone(),
            pk_value_source: pk_source,
            updates: compiled_updates,
            cached_epoch: self.engine.schema_epoch(),
        };

        *compiled_guard = CompiledExecution::PkUpdate(compiled_update.clone());
        drop(compiled_guard);

        // Execute
        Some(self.execute_compiled_pk_update(&compiled_update, pk_value, ctx))
    }

    /// Compile and execute a PK delete, caching the compiled state
    fn compile_and_execute_pk_delete(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check after acquiring lock
        match &*compiled_guard {
            CompiledExecution::NotOptimizable => return None,
            CompiledExecution::PkDelete(delete) => {
                let pk_value = self.extract_pk_value_from_source(&delete.pk_value_source, ctx)?;
                return Some(self.execute_compiled_pk_delete(delete, pk_value));
            }
            CompiledExecution::Unknown => {} // Continue with compilation
            _ => return None,
        }

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let schema = match self.engine.get_table_schema(table_name) {
            Ok(s) => s,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable;
                return None;
            }
        };

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable;
                    return None;
                }
            };

        // Build compiled state
        let compiled_delete = CompiledPkDelete {
            table_name: table_name.clone(),
            schema: Arc::new((*schema).clone()),
            pk_column_name: pk_column.clone(),
            pk_value_source: pk_source,
            cached_epoch: self.engine.schema_epoch(),
        };

        *compiled_guard = CompiledExecution::PkDelete(compiled_delete.clone());
        drop(compiled_guard);

        // Execute
        Some(self.execute_compiled_pk_delete(&compiled_delete, pk_value))
    }

    /// Extract value source (literal or parameter) from expression
    fn extract_value_source(&self, expr: &Expression) -> Option<UpdateValueSource> {
        match expr {
            Expression::IntegerLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Integer(lit.value)))
            }
            Expression::FloatLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Float(lit.value)))
            }
            Expression::StringLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::text(&lit.value)))
            }
            Expression::BooleanLiteral(lit) => {
                Some(UpdateValueSource::Literal(Value::Boolean(lit.value)))
            }
            Expression::NullLiteral(_) => Some(UpdateValueSource::Literal(Value::null_unknown())),
            Expression::Prefix(prefix) if prefix.operator == "-" => match prefix.right.as_ref() {
                Expression::IntegerLiteral(lit) => {
                    Some(UpdateValueSource::Literal(Value::Integer(-lit.value)))
                }
                Expression::FloatLiteral(lit) => {
                    Some(UpdateValueSource::Literal(Value::Float(-lit.value)))
                }
                _ => None,
            },
            Expression::Parameter(param) => {
                let param_idx = if param.index > 0 {
                    param.index - 1
                } else {
                    param.index
                };
                Some(UpdateValueSource::Parameter(param_idx))
            }
            _ => None, // Complex expression
        }
    }
}
