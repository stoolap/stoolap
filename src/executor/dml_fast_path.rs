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
use crate::storage::traits::QueryResult;

use super::context::{
    invalidate_in_subquery_cache_for_table, invalidate_scalar_subquery_cache_for_table,
    invalidate_semi_join_cache_for_table, ExecutionContext,
};
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
                CompiledExecution::NotOptimizable(epoch)
                    if self.engine.schema_epoch() == *epoch =>
                {
                    return None
                }
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
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                _ => return None, // Different type of compiled execution
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
                CompiledExecution::NotOptimizable(epoch)
                    if self.engine.schema_epoch() == *epoch =>
                {
                    return None
                }
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
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                _ => return None, // Different type of compiled execution
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
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(q) => q.name.value.to_string(),
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
                // Named parameters (e.g., :name) resolve via get_named_param() at execution time
                // Positional parameters ($1, $2, ...) are 1-indexed, array is 0-indexed
                if param.name.starts_with(':') {
                    let name = &param.name[1..];
                    let value = ctx.get_named_param(name)?;
                    let pk_value = match value {
                        Value::Integer(i) => *i,
                        Value::Float(f) => *f as i64,
                        _ => return None,
                    };
                    (
                        pk_value,
                        PkValueSource::NamedParameter(SmartString::new(name)),
                    )
                } else {
                    let params = ctx.params();
                    let param_idx = if param.index > 0 {
                        param.index - 1
                    } else {
                        return None;
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
        match source {
            PkValueSource::NamedParameter(name) => match ctx.get_named_param(name)? {
                Value::Integer(i) => Some(*i),
                Value::Float(f) => Some(*f as i64),
                _ => None,
            },
            _ => Self::extract_pk_value_from_params(source, ctx.params()),
        }
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
            PkValueSource::NamedParameter(_) => None, // No ctx available in slice path
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
            UpdateValueSource::NamedParameter(_) => None, // No ctx available in slice path
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
                if self.engine.schema_epoch() == update.cached_epoch {
                    let pk_value =
                        Self::extract_pk_value_from_params(&update.pk_value_source, params)?;
                    // Extract update values
                    let mut updates = Vec::with_capacity(update.updates.len());
                    for u in &update.updates {
                        let value = Self::extract_update_value_from_slice(&u.value_source, params)?;
                        updates.push((u.column_idx, value.coerce_to_type(u.column_type)));
                    }
                    // Clone only what we need (cheap: SmartString + Arc)
                    let table_name = update.table_name.clone();
                    let pk_column_name = update.pk_column_name.clone();
                    let schema = update.schema.clone();
                    drop(compiled_guard);
                    return Some(self.execute_pk_update_minimal(
                        &table_name,
                        &pk_column_name,
                        &schema,
                        pk_value,
                        updates,
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
                if self.engine.schema_epoch() == delete.cached_epoch {
                    let pk_value =
                        Self::extract_pk_value_from_params(&delete.pk_value_source, params)?;
                    // Clone only what we need (cheap: SmartString + Arc)
                    let table_name = delete.table_name.clone();
                    let pk_column_name = delete.pk_column_name.clone();
                    let schema = delete.schema.clone();
                    drop(compiled_guard);
                    return Some(self.execute_pk_delete_minimal(
                        &table_name,
                        &pk_column_name,
                        &schema,
                        pk_value,
                    ));
                }
                None // Epoch changed, use normal path
            }
            CompiledExecution::Unknown => None,
            _ => None,
        }
    }

    /// Execute PK update with minimal data (avoids cloning CompiledPkUpdate)
    fn execute_pk_update_minimal(
        &self,
        table_name: &str,
        pk_column_name: &str,
        schema: &CompactArc<Schema>,
        pk_value: i64,
        updates: Vec<(usize, Value)>,
    ) -> Result<Box<dyn QueryResult>> {
        // Create auto-commit transaction
        let tx = self.engine.begin_transaction()?;
        let mut table = tx.get_table(table_name)?;

        // Build WHERE expression for PK lookup
        let mut pk_expr = ComparisonExpr::new(
            pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(schema);

        // Execute update with simple setter
        let mut setter = |mut row: Row| -> Result<(Row, bool)> {
            for (idx, new_value) in &updates {
                let _ = row.set(*idx, new_value.clone());
            }
            Ok((row, true))
        };

        let rows_affected = table.update(Some(&pk_expr), &mut setter)?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
            invalidate_semi_join_cache_for_table(table_name);
            invalidate_scalar_subquery_cache_for_table(table_name);
            invalidate_in_subquery_cache_for_table(table_name);
        }

        // Commit
        drop(table);
        let mut tx = tx;
        tx.commit()?;

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
        pk_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        // Create auto-commit transaction
        let tx = self.engine.begin_transaction()?;
        let mut table = tx.get_table(table_name)?;

        // Build WHERE expression for PK lookup
        let mut pk_expr = ComparisonExpr::new(
            pk_column_name,
            crate::core::Operator::Eq,
            Value::Integer(pk_value),
        );
        pk_expr.prepare_for_schema(schema);

        // Execute delete
        let rows_affected = table.delete(Some(&pk_expr))?;

        // Invalidate caches
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
            invalidate_semi_join_cache_for_table(table_name);
            invalidate_scalar_subquery_cache_for_table(table_name);
            invalidate_in_subquery_cache_for_table(table_name);
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

    /// Execute a compiled PK update (extracts values from ctx then delegates to core impl)
    fn execute_compiled_pk_update(
        &self,
        compiled: &CompiledPkUpdate,
        pk_value: i64,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Extract values from compiled sources
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
                UpdateValueSource::NamedParameter(name) => match ctx.get_named_param(name) {
                    Some(v) => v.clone(),
                    None => continue,
                },
            };
            updates.push((u.column_idx, value.coerce_to_type(u.column_type)));
        }

        self.execute_pk_update_minimal(
            &compiled.table_name,
            &compiled.pk_column_name,
            &compiled.schema,
            pk_value,
            updates,
        )
    }

    /// Execute a compiled PK delete (delegates to core impl)
    fn execute_compiled_pk_delete(
        &self,
        compiled: &CompiledPkDelete,
        pk_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_pk_delete_minimal(
            &compiled.table_name,
            &compiled.pk_column_name,
            &compiled.schema,
            pk_value,
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
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check after acquiring lock
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(epoch) if self.engine.schema_epoch() == *epoch => {
                return None
            }
            CompiledExecution::PkUpdate(update) => {
                let pk_value = self.extract_pk_value_from_source(&update.pk_value_source, ctx)?;
                return Some(self.execute_compiled_pk_update(update, pk_value, ctx));
            }
            CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let schema = match self.engine.get_table_schema(table_name) {
            Ok(s) => s,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Bail if table has FK constraints (child table) or is referenced by other tables (parent table)
        // FK enforcement requires cross-table lookups — fall back to normal path
        if !schema.foreign_keys.is_empty()
            || !super::foreign_key::find_referencing_fks(self.engine.as_ref(), table_name)
                .is_empty()
        {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
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
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
            };
            let col_type = schema.columns[col_idx].data_type;

            let value_source = match self.extract_value_source(expr) {
                Some(s) => s,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
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
            table_name: SmartString::new(table_name),
            schema: CompactArc::new((*schema).clone()),
            pk_column_name: SmartString::new(pk_column),
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
            CompiledExecution::NotOptimizable(epoch) if self.engine.schema_epoch() == *epoch => {
                return None
            }
            CompiledExecution::PkDelete(delete) => {
                let pk_value = self.extract_pk_value_from_source(&delete.pk_value_source, ctx)?;
                return Some(self.execute_compiled_pk_delete(delete, pk_value));
            }
            CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Validate pattern
        let where_clause = stmt.where_clause.as_ref()?;
        if !stmt.returning.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        let table_name = &stmt.table_name.value_lower;
        let schema = match self.engine.get_table_schema(table_name) {
            Ok(s) => s,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Bail if this table is referenced by child tables (FK enforcement needed)
        if !super::foreign_key::find_referencing_fks(self.engine.as_ref(), table_name).is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Extract PK value source
        let (pk_value, pk_source) =
            match self.extract_pk_equality_value(where_clause, pk_column, ctx) {
                Some(v) => v,
                None => {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
            };

        // Build compiled state
        let compiled_delete = CompiledPkDelete {
            table_name: SmartString::new(table_name),
            schema: CompactArc::new((*schema).clone()),
            pk_column_name: SmartString::new(pk_column),
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
                Some(UpdateValueSource::Literal(Value::text(lit.value.as_str())))
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
                if param.name.starts_with(':') {
                    let name = &param.name[1..];
                    Some(UpdateValueSource::NamedParameter(SmartString::new(name)))
                } else {
                    let param_idx = if param.index > 0 {
                        param.index - 1
                    } else {
                        return None;
                    };
                    Some(UpdateValueSource::Parameter(param_idx))
                }
            }
            _ => None, // Complex expression
        }
    }
}
