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

//! DML Statement Execution
//!
//! This module implements execution of Data Manipulation Language (DML) statements:
//! - INSERT
//! - UPDATE
//! - DELETE

use crate::core::{DataType, Error, Result, Row, Value};
use crate::parser::ast::*;
use crate::storage::expression::{ComparisonExpr, Expression as StorageExpr};
use crate::storage::traits::{Engine, QueryResult, Table};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::context::ExecutionContext;
use super::evaluator::Evaluator;
use super::result::ExecResult;
use super::Executor;

/// Validate type coercion didn't silently fail.
/// Returns an error if a non-null value became null during coercion.
fn validate_coercion(
    original: &Value,
    coerced: &Value,
    column_name: &str,
    target_type: DataType,
) -> Result<()> {
    // If original was non-null but coerced is null, the conversion failed
    if !original.is_null() && coerced.is_null() {
        return Err(Error::Type(format!(
            "cannot convert value '{}' to {:?} for column '{}'",
            original, target_type, column_name
        )));
    }
    Ok(())
}

impl Executor {
    /// Execute an INSERT statement
    pub(crate) fn execute_insert(
        &self,
        stmt: &InsertStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);

        // Pre-compute schema information to avoid repeated borrows during insert
        let schema_column_count: usize;
        let column_indices: Vec<usize>;
        // Pre-compute column types for type coercion
        let column_types: Vec<crate::core::DataType>;
        // Pre-compute column names for error messages
        let column_names: Vec<String>;
        // Pre-compute ALL column types for default values and check constraints
        let all_column_types: Vec<crate::core::DataType>;
        // Pre-compute default values and check expressions for all columns
        let default_exprs: Vec<Option<String>>;
        let check_exprs: Vec<(String, Option<String>)>; // (column_name, check_expr)
        {
            let schema = table.schema();
            schema_column_count = schema.columns.len();

            // Extract default and check expressions from schema
            default_exprs = schema
                .columns
                .iter()
                .map(|c| c.default_expr.clone())
                .collect();
            check_exprs = schema
                .columns
                .iter()
                .map(|c| (c.name.clone(), c.check_expr.clone()))
                .collect();
            all_column_types = schema.columns.iter().map(|c| c.data_type).collect();

            // OPTIMIZATION: When no columns specified, insert into all columns in order
            // Skip all column name lookups - just use sequential indices
            if stmt.columns.is_empty() {
                column_indices = (0..schema_column_count).collect();
                column_types = all_column_types.clone();
                column_names = schema.columns.iter().map(|c| c.name.clone()).collect();
            } else {
                // Validate columns exist and pre-compute their indices
                column_indices = stmt
                    .columns
                    .iter()
                    .map(|id| {
                        // Use pre-computed lowercase value from AST
                        let col_lower = &id.value_lower;
                        schema
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(col_lower))
                            .ok_or_else(|| Error::ColumnNotFoundNamed(id.value.clone()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                // Get column types for the specified columns
                column_types = column_indices
                    .iter()
                    .map(|&idx| schema.columns[idx].data_type)
                    .collect();
                // Get column names for error messages
                column_names = column_indices
                    .iter()
                    .map(|&idx| schema.columns[idx].name.clone())
                    .collect();
            }
        }

        // Create evaluator for expressions
        let evaluator = Evaluator::new(&self.function_registry).with_context(ctx);

        let mut rows_affected = 0i64;

        // RETURNING clause support - collect inserted rows if RETURNING is specified
        let has_returning = !stmt.returning.is_empty();
        let mut returning_rows: Vec<Row> = Vec::new();
        let schema_column_names: Vec<String> = table.schema().column_names_owned().to_vec();

        // Check if this is INSERT ... SELECT
        if let Some(ref select_stmt) = stmt.select {
            // Execute the SELECT query
            let mut select_result = self.execute_select(select_stmt, ctx)?;

            // Process each row from the SELECT result
            while select_result.next() {
                let select_row = select_result.row();
                let select_values = select_row.as_slice();

                if select_values.len() != column_indices.len() {
                    return Err(Error::InvalidArgumentMessage(format!(
                        "INSERT has {} columns but SELECT returns {} columns",
                        column_indices.len(),
                        select_values.len()
                    )));
                }

                // Build row values - initialize with DEFAULT values for missing columns
                // This matches the behavior of regular INSERT
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    if let Some(ref default_expr) = default_exprs[i] {
                        let default_type = all_column_types[i];
                        match self.evaluate_default_expr(default_expr, default_type) {
                            Ok(val) => row_values.push(val),
                            Err(_) => row_values.push(Value::null_unknown()),
                        }
                    } else {
                        row_values.push(Value::null_unknown());
                    }
                }

                // Fill in values from SELECT using pre-computed indices with type coercion
                for (i, value) in select_values.iter().enumerate() {
                    // Coerce value to target column type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(value, &coerced, &column_names[i], column_types[i])?;
                    row_values[column_indices[i]] = coerced;
                }

                // Create row and insert (returns row with AUTO_INCREMENT applied)
                let row = Row::from_values(row_values);
                let inserted_row = table.insert(row)?;
                rows_affected += 1;

                // Collect inserted row for RETURNING if specified
                if has_returning {
                    returning_rows.push(inserted_row);
                }
            }

            // Invalidate semantic cache for this table BEFORE commit
            // CRITICAL: Must invalidate before commit to prevent stale data window
            // where concurrent queries could see new data in storage but get old cached results
            if rows_affected > 0 {
                self.semantic_cache.invalidate_table(table_name);
            }

            // Commit if this is a standalone (auto-commit) transaction
            if should_auto_commit {
                // Just commit the transaction - it will commit all tables via commit_all_tables()
                if let Some(mut tx) = standalone_tx {
                    tx.commit()?;
                }
            }

            // Handle RETURNING clause for INSERT...SELECT
            if has_returning {
                return self.build_returning_result(
                    &stmt.returning,
                    returning_rows,
                    &schema_column_names,
                    ctx,
                );
            }

            return Ok(Box::new(ExecResult::with_rows_affected(rows_affected)));
        }

        // Process each row of values - use fast path for normal INSERT, slow path for ON DUPLICATE KEY
        if stmt.on_duplicate {
            // ON DUPLICATE KEY UPDATE requires schema clone for potential updates
            let schema = table.schema().clone();

            for value_row in &stmt.values {
                if value_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgumentMessage(format!(
                        "INSERT has {} columns but {} values",
                        column_indices.len(),
                        value_row.len()
                    )));
                }

                // Build row values - initialize with DEFAULT values for missing columns
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    if let Some(ref default_expr) = default_exprs[i] {
                        let default_type = all_column_types[i];
                        match self.evaluate_default_expr(default_expr, default_type) {
                            Ok(val) => row_values.push(val),
                            Err(_) => row_values.push(Value::null_unknown()),
                        }
                    } else {
                        row_values.push(Value::null_unknown());
                    }
                }
                // Fill in provided values using pre-computed indices with type coercion
                for (i, expr) in value_row.iter().enumerate() {
                    // Handle DEFAULT keyword - skip this column to use pre-initialized default
                    if matches!(expr, Expression::Default(_)) {
                        continue;
                    }
                    let value = evaluator.evaluate(expr)?;
                    // Coerce to target type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(&value, &coerced, &column_names[i], column_types[i])?;
                    row_values[column_indices[i]] = coerced;
                }

                // Need to clone for potential update
                let row = Row::from_values(row_values.clone());
                match table.insert(row) {
                    Ok(_inserted_row) => {
                        rows_affected += 1;
                    }
                    Err(Error::PrimaryKeyConstraint { row_id }) => {
                        self.apply_on_duplicate_update(
                            &mut table,
                            &schema,
                            row_id,
                            &row_values,
                            stmt,
                            ctx,
                        )?;
                        rows_affected += 1;
                    }
                    Err(Error::UniqueConstraint {
                        index,
                        column,
                        value: _,
                    }) => {
                        if let Some(row_id) = self.find_row_by_unique_index(
                            &*table,
                            &schema,
                            &index,
                            &column,
                            &row_values,
                        )? {
                            self.apply_on_duplicate_update(
                                &mut table,
                                &schema,
                                row_id,
                                &row_values,
                                stmt,
                                ctx,
                            )?;
                            rows_affected += 1;
                        } else {
                            return Err(Error::UniqueConstraint {
                                index,
                                column,
                                value: "unknown".to_string(),
                            });
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        } else {
            // Fast path: normal INSERT without clones
            for value_row in &stmt.values {
                if value_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgumentMessage(format!(
                        "INSERT has {} columns but {} values",
                        column_indices.len(),
                        value_row.len()
                    )));
                }

                // Build row values - initialize with DEFAULT values for missing columns
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    if let Some(ref default_expr) = default_exprs[i] {
                        // Evaluate the default expression using the actual column type
                        let default_type = all_column_types[i];
                        match self.evaluate_default_expr(default_expr, default_type) {
                            Ok(val) => row_values.push(val),
                            Err(_) => row_values.push(Value::null_unknown()),
                        }
                    } else {
                        row_values.push(Value::null_unknown());
                    }
                }

                // Fill in provided values using pre-computed indices with type coercion
                for (i, expr) in value_row.iter().enumerate() {
                    // Handle DEFAULT keyword - skip this column to use pre-initialized default
                    if matches!(expr, Expression::Default(_)) {
                        continue;
                    }
                    let value = evaluator.evaluate(expr)?;
                    // Coerce to target type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(&value, &coerced, &column_names[i], column_types[i])?;
                    row_values[column_indices[i]] = coerced;
                }

                // Validate CHECK constraints
                for (col_idx, (col_name, check_expr_opt)) in check_exprs.iter().enumerate() {
                    if let Some(ref check_expr) = check_expr_opt {
                        let col_type = all_column_types[col_idx];
                        self.validate_check_constraint(
                            check_expr,
                            col_name,
                            &row_values[col_idx],
                            col_type,
                        )?;
                    }
                }

                // Insert row (returns row with AUTO_INCREMENT applied)
                let row = Row::from_values(row_values);
                let inserted_row = table.insert(row)?;
                rows_affected += 1;

                // Collect inserted row for RETURNING if specified
                if has_returning {
                    returning_rows.push(inserted_row);
                }
            }
        }

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        // Handle RETURNING clause
        if has_returning {
            return self.build_returning_result(
                &stmt.returning,
                returning_rows,
                &schema_column_names,
                ctx,
            );
        }

        Ok(Box::new(ExecResult::with_rows_affected(rows_affected)))
    }

    /// Execute an UPDATE statement
    pub(crate) fn execute_update(
        &self,
        stmt: &UpdateStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);

        // Check for RETURNING clause
        let has_returning = !stmt.returning.is_empty();

        // Pre-compute column names and indices to avoid schema borrow conflicts
        let schema = table.schema();
        // OPTIMIZATION: Use reference directly, avoid cloning all column names
        let column_names = schema.column_names_owned();

        // Check if any update expressions contain subqueries
        let has_update_subqueries = stmt
            .updates
            .iter()
            .any(|(_, expr)| Self::has_subqueries(expr));

        // Check if any update expressions have correlated subqueries
        let has_correlated_updates = stmt
            .updates
            .iter()
            .any(|(_, expr)| Self::has_subqueries(expr) && Self::has_correlated_subqueries(expr));

        // Pre-process update expressions if they contain NON-correlated subqueries
        // Correlated subqueries must be processed per-row with outer row context
        let processed_updates: Option<Vec<(String, Expression)>> =
            if has_update_subqueries && !has_correlated_updates {
                let processed: Result<Vec<_>> = stmt
                    .updates
                    .iter()
                    .map(|(col_name, expr)| {
                        let processed_expr = self.process_where_subqueries(expr, ctx)?;
                        Ok((col_name.clone(), processed_expr))
                    })
                    .collect();
                Some(processed?)
            } else {
                None
            };

        // Pre-compute column indices and types for updates (avoids string comparison per row)
        // We store the index, type, expression, and whether it has correlated subqueries
        let update_indices: Vec<(usize, crate::core::DataType, Expression, bool)> =
            if let Some(ref processed) = processed_updates {
                processed
                    .iter()
                    .filter_map(|(col_name, expr)| {
                        schema
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(col_name))
                            .map(|idx| (idx, schema.columns[idx].data_type, expr.clone(), false))
                    })
                    .collect()
            } else {
                stmt.updates
                    .iter()
                    .filter_map(|(col_name, expr)| {
                        let is_correlated =
                            Self::has_subqueries(expr) && Self::has_correlated_subqueries(expr);
                        schema
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(col_name))
                            .map(|idx| {
                                (
                                    idx,
                                    schema.columns[idx].data_type,
                                    expr.clone(),
                                    is_correlated,
                                )
                            })
                    })
                    .collect()
            };

        // Build WHERE expression for storage layer
        // Try to convert to storage expression, fall back to in-memory filtering if not possible
        let (where_expr, needs_memory_filter, memory_where_clause): (
            Option<Box<dyn StorageExpr>>,
            bool,
            Option<Expression>,
        ) = if let Some(ref where_clause) = stmt.where_clause {
            let processed_where = if Self::has_subqueries(where_clause) {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Try to build storage expression, fall back to memory filter if it fails
            match self.build_storage_expression_with_ctx(&processed_where, schema, Some(ctx)) {
                Ok(expr) => (Some(expr), false, None),
                Err(_) => {
                    // Complex expression (like a + b > 100) - use in-memory filtering
                    (None, true, Some(processed_where))
                }
            }
        } else {
            (None, false, None)
        };

        let function_registry = &self.function_registry;

        // Create evaluator once and reuse for all rows (optimization)
        let mut evaluator = Evaluator::new(function_registry);
        evaluator = evaluator.with_context(ctx);
        evaluator.init_columns(column_names);

        // For correlated subqueries, we need to process per-row with outer row context
        // Build column name mappings for outer row context
        let column_names_vec: Vec<String> = column_names.to_vec();

        // Use RefCell to collect updated rows for RETURNING clause
        use std::cell::RefCell;
        let returning_rows: RefCell<Vec<Row>> = RefCell::new(Vec::new());

        // Create a setter function that applies updates using pre-computed indices
        // If we need memory filtering, include the WHERE check in the setter
        // For correlated subqueries, we need special handling
        let rows_affected = if has_correlated_updates {
            // Path for correlated subqueries: we need to pre-compute all values
            // because process_correlated_expression calls self methods and can't be
            // used inside the closure. Strategy:
            // 1. Scan table to find all rows (matching WHERE if applicable)
            // 2. For each row, build outer_row context and evaluate correlated expressions
            // 3. Store computed values keyed by PK
            // 4. Call table.update with a setter that looks up pre-computed values

            // Get primary key column index
            let pk_indices = schema.primary_key_indices();
            let pk_idx = pk_indices.first().copied().unwrap_or(0);

            // Pre-compute values for all rows
            // Map: pk_value -> Vec<(col_idx, new_value)>
            let mut precomputed: FxHashMap<Value, Vec<(usize, Value)>> =
                FxHashMap::with_capacity_and_hasher(64, Default::default());

            // Build column indices for scanning (all columns)
            let all_col_indices: Vec<usize> = (0..column_names_vec.len()).collect();
            let column_names_arc = Arc::new(column_names_vec.clone());

            // Scan all rows (WHERE filtering happens in the setter)
            let mut scanner = table.scan(&all_col_indices, None)?;
            while scanner.next() {
                let row = scanner.row();

                // Check WHERE condition if needed
                evaluator.set_row_array(row);
                if needs_memory_filter {
                    if let Some(ref where_clause) = memory_where_clause {
                        match evaluator.evaluate_bool(where_clause) {
                            Ok(true) => {}
                            _ => continue,
                        }
                    }
                }

                // Get PK value for this row
                let pk_value = row.get(pk_idx).cloned().unwrap_or(Value::null_unknown());

                // Build outer row context from current row values
                let mut outer_row_map: FxHashMap<String, Value> =
                    FxHashMap::with_capacity_and_hasher(
                        column_names_vec.len() * 2,
                        Default::default(),
                    );
                for (i, col_name) in column_names_vec.iter().enumerate() {
                    if let Some(value) = row.get(i) {
                        let col_lower = col_name.to_lowercase();
                        outer_row_map.insert(col_lower.clone(), value.clone());
                        outer_row_map
                            .insert(format!("{}.{}", table_name, col_lower), value.clone());
                    }
                }

                // Create context with outer row for correlated subquery evaluation
                let correlated_ctx = ctx.with_outer_row(outer_row_map, column_names_arc.clone());

                // Evaluate all update expressions
                let mut new_values: Vec<(usize, Value)> = Vec::with_capacity(update_indices.len());
                for (idx, col_type, expr, is_correlated) in update_indices.iter() {
                    let evaluated = if *is_correlated {
                        // Process correlated expression - this executes the subquery
                        match self.process_correlated_expression(expr, &correlated_ctx) {
                            Ok(processed_expr) => {
                                // Now evaluate the processed expression (subquery replaced with value)
                                let mut eval = Evaluator::new(function_registry);
                                eval = eval.with_context(&correlated_ctx);
                                eval.init_columns(column_names);
                                eval.set_row_array(row);
                                eval.evaluate(&processed_expr).ok()
                            }
                            Err(_) => None,
                        }
                    } else {
                        evaluator.evaluate(expr).ok()
                    };

                    if let Some(new_value) = evaluated {
                        new_values.push((*idx, new_value.into_coerce_to_type(*col_type)));
                    }
                }

                if !new_values.is_empty() {
                    precomputed.insert(pk_value, new_values);
                }
            }
            drop(scanner);

            // Now update using precomputed values
            let mut setter = |mut row: Row| -> (Row, bool) {
                let pk_value = row.get(pk_idx).cloned().unwrap_or(Value::null_unknown());

                if let Some(updates) = precomputed.get(&pk_value) {
                    for (idx, new_value) in updates {
                        let _ = row.set(*idx, new_value.clone());
                    }
                    // Collect row for RETURNING clause
                    if has_returning {
                        returning_rows.borrow_mut().push(row.clone());
                    }
                    (row, true)
                } else {
                    (row, false)
                }
            };

            table.update(where_expr.as_deref(), &mut setter)?
        } else {
            // Optimized path for non-correlated subqueries
            let mut setter = |mut row: Row| -> (Row, bool) {
                evaluator.set_row_array(&row);

                // If we need in-memory WHERE filtering, check the condition first
                if needs_memory_filter {
                    if let Some(ref where_expr) = memory_where_clause {
                        match evaluator.evaluate_bool(where_expr) {
                            Ok(true) => {}
                            _ => return (row, false),
                        }
                    }
                }

                // Evaluate ALL expressions FIRST using original row values
                let new_values: Vec<(usize, crate::core::Value)> = update_indices
                    .iter()
                    .filter_map(|(idx, col_type, expr, _)| {
                        evaluator
                            .evaluate(expr)
                            .ok()
                            .map(|new_value| (*idx, new_value.into_coerce_to_type(*col_type)))
                    })
                    .collect();

                // Now apply all the computed values to the row
                let changed = !new_values.is_empty();
                for (idx, new_value) in new_values {
                    let _ = row.set(idx, new_value);
                }

                // Collect row for RETURNING clause
                if changed && has_returning {
                    returning_rows.borrow_mut().push(row.clone());
                }

                (row, changed)
            };

            table.update(where_expr.as_deref(), &mut setter)?
        };

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        // Handle RETURNING clause
        if has_returning {
            let rows = returning_rows.into_inner();
            return self.build_returning_result(&stmt.returning, rows, &column_names_vec, ctx);
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Execute a DELETE statement
    pub(crate) fn execute_delete(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;
        // Use alias if provided, otherwise use table name
        let effective_name = stmt
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str())
            .unwrap_or(table_name.as_str());

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);

        // Check for RETURNING clause
        let has_returning = !stmt.returning.is_empty();
        let mut returning_rows: Vec<Row> = Vec::new();

        // Build WHERE expression - try to convert to storage expression
        // If that fails (complex expression like a + b > 100), fall back to in-memory filtering
        let schema = table.schema();

        // Check if WHERE has correlated subqueries (needs per-row evaluation)
        let has_correlated = if let Some(ref where_clause) = stmt.where_clause {
            Self::has_subqueries(where_clause) && Self::has_correlated_subqueries(where_clause)
        } else {
            false
        };

        let (where_expr, needs_memory_filter, memory_where_clause): (
            Option<Box<dyn StorageExpr>>,
            bool,
            Option<Expression>,
        ) = if let Some(ref where_clause) = stmt.where_clause {
            if has_correlated {
                // For correlated subqueries, keep original and process per-row
                (None, true, Some((**where_clause).clone()))
            } else {
                let processed_where = if Self::has_subqueries(where_clause) {
                    self.process_where_subqueries(where_clause, ctx)?
                } else {
                    (**where_clause).clone()
                };

                // Try to build storage expression, fall back to memory filter if it fails
                match self.build_storage_expression_with_ctx(&processed_where, schema, Some(ctx)) {
                    Ok(expr) => (Some(expr), false, None),
                    Err(_) => {
                        // Complex expression (like a + b > 100) - use in-memory filtering
                        (None, true, Some(processed_where))
                    }
                }
            }
        } else {
            (None, false, None)
        };

        // Get schema info for RETURNING clause processing
        let column_names_owned = schema.column_names_owned().to_vec();
        let column_count = schema.columns.len();
        let pk_col_idx = schema.columns.iter().position(|c| c.primary_key);
        let pk_col_name = pk_col_idx.map(|idx| schema.columns[idx].name.clone());

        // Build column names with effective prefix (alias or table name)
        // This allows WHERE clauses to reference columns using the alias
        let column_names_with_prefix: Vec<String> = column_names_owned
            .iter()
            .map(|c| format!("{}.{}", effective_name, c))
            .collect();

        // Delete rows
        let rows_affected = if needs_memory_filter || has_returning {
            // Complex WHERE expression OR RETURNING - need to scan rows first
            // Scan all rows, filter with evaluator, collect for RETURNING, delete matching ones by primary key
            // Clone schema for later use to avoid borrow conflict
            let schema_clone = schema.clone();

            // Create evaluator for WHERE filtering
            let mut evaluator = Evaluator::new(&self.function_registry);
            evaluator = evaluator.with_context(ctx);
            // Initialize with prefixed column names to support alias.column syntax
            evaluator.init_columns(&column_names_with_prefix);

            // Scan all rows and collect IDs of rows to delete
            let column_indices: Vec<usize> = (0..column_count).collect();
            let mut scanner = table.scan(&column_indices, where_expr.as_deref())?;
            let mut rows_to_delete: Vec<(Value, Option<Row>)> = Vec::new();

            // Pre-compute column name mappings for correlated subqueries
            let column_names_arc = if has_correlated {
                Some(std::sync::Arc::new(column_names_owned.clone()))
            } else {
                None
            };

            while scanner.next() {
                let row = scanner.row();

                // Check memory filter if needed
                let matches = if needs_memory_filter {
                    evaluator.set_row_array(row);
                    if let Some(ref where_expr) = memory_where_clause {
                        if has_correlated {
                            // For correlated subqueries, build outer row context and process per-row
                            let mut outer_row_map: rustc_hash::FxHashMap<String, Value> =
                                rustc_hash::FxHashMap::default();

                            // Build outer row context with column values
                            for (i, col_name) in column_names_owned.iter().enumerate() {
                                if let Some(value) = row.get(i) {
                                    let col_lower = col_name.to_lowercase();
                                    outer_row_map.insert(col_lower.clone(), value.clone());
                                    // Add with effective name prefix (alias if present, otherwise table name)
                                    outer_row_map.insert(
                                        format!("{}.{}", effective_name, col_lower),
                                        value.clone(),
                                    );
                                    // Also add with table name prefix for cases where both are needed
                                    if effective_name != table_name {
                                        outer_row_map.insert(
                                            format!("{}.{}", table_name, col_lower),
                                            value.clone(),
                                        );
                                    }
                                }
                            }

                            // Create context with outer row
                            let correlated_ctx = ctx
                                .with_outer_row(outer_row_map, column_names_arc.clone().unwrap());

                            // Process correlated subquery with outer context
                            match self.process_correlated_where(where_expr, &correlated_ctx) {
                                Ok(processed) => {
                                    evaluator.set_outer_row(correlated_ctx.outer_row());
                                    let result =
                                        evaluator.evaluate_bool(&processed).unwrap_or(false);
                                    evaluator.clear_outer_row();
                                    result
                                }
                                Err(_) => false,
                            }
                        } else {
                            matches!(evaluator.evaluate_bool(where_expr), Ok(true))
                        }
                    } else {
                        true
                    }
                } else {
                    true // Storage layer already filtered
                };

                if matches {
                    // Row matches - get primary key value for deletion
                    if let Some(pk_idx) = pk_col_idx {
                        if let Some(pk_value) = row.get(pk_idx) {
                            let row_data = if has_returning {
                                Some(row.clone())
                            } else {
                                None
                            };
                            rows_to_delete.push((pk_value.clone(), row_data));
                        }
                    }
                }
            }
            // Drop scanner to release borrow
            drop(scanner);

            // Delete matching rows by primary key
            let mut delete_count = 0;
            if let Some(ref pk_name) = pk_col_name {
                for (pk_value, row_data) in rows_to_delete {
                    let mut pk_expr =
                        ComparisonExpr::new(pk_name, crate::core::Operator::Eq, pk_value);
                    pk_expr.prepare_for_schema(&schema_clone);
                    let deleted = table.delete(Some(&pk_expr))?;
                    if deleted > 0 {
                        if let Some(row) = row_data {
                            returning_rows.push(row);
                        }
                        delete_count += deleted;
                    }
                }
            }
            delete_count
        } else {
            // Simple WHERE expression without RETURNING - use storage layer directly
            table.delete(where_expr.as_deref())?
        };

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        // Handle RETURNING clause
        if has_returning {
            return self.build_returning_result(
                &stmt.returning,
                returning_rows,
                &column_names_owned,
                ctx,
            );
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Execute a TRUNCATE statement
    /// TRUNCATE is equivalent to DELETE without WHERE clause, but more efficient
    pub(crate) fn execute_truncate(
        &self,
        stmt: &TruncateStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                let table = tx_state.transaction.get_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);

        // Delete all rows (no WHERE clause)
        let rows_affected = table.delete(None)?;

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        // (TRUNCATE always invalidates, regardless of rows_affected, for safety)
        self.semantic_cache.invalidate_table(table_name);

        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Build a storage-layer expression from a parser AST expression
    #[allow(dead_code)]
    pub(crate) fn build_storage_expression(
        &self,
        expr: &Expression,
        schema: &crate::core::Schema,
    ) -> Result<Box<dyn StorageExpr>> {
        self.build_storage_expression_with_ctx(expr, schema, None)
    }

    pub(crate) fn build_storage_expression_with_ctx(
        &self,
        expr: &Expression,
        schema: &crate::core::Schema,
        ctx: Option<&ExecutionContext>,
    ) -> Result<Box<dyn StorageExpr>> {
        use crate::parser::ast::InfixOperator;

        match expr {
            Expression::Infix(infix) => {
                // OPTIMIZATION: Use pre-computed op_type enum instead of string comparison
                // This avoids to_uppercase() allocation on every query
                match infix.op_type {
                    InfixOperator::And => {
                        let left =
                            self.build_storage_expression_with_ctx(&infix.left, schema, ctx)?;
                        let right =
                            self.build_storage_expression_with_ctx(&infix.right, schema, ctx)?;
                        Ok(Box::new(crate::storage::expression::AndExpr::new(vec![
                            left, right,
                        ])))
                    }
                    InfixOperator::Or => {
                        let left =
                            self.build_storage_expression_with_ctx(&infix.left, schema, ctx)?;
                        let right =
                            self.build_storage_expression_with_ctx(&infix.right, schema, ctx)?;
                        Ok(Box::new(crate::storage::expression::OrExpr::new(vec![
                            left, right,
                        ])))
                    }
                    InfixOperator::Xor => {
                        // XOR is equivalent to: (A OR B) AND NOT (A AND B)
                        // Or more simply: (A AND NOT B) OR (NOT A AND B)
                        let left =
                            self.build_storage_expression_with_ctx(&infix.left, schema, ctx)?;
                        let right =
                            self.build_storage_expression_with_ctx(&infix.right, schema, ctx)?;
                        // Clone the expressions for the second part
                        let left2 =
                            self.build_storage_expression_with_ctx(&infix.left, schema, ctx)?;
                        let right2 =
                            self.build_storage_expression_with_ctx(&infix.right, schema, ctx)?;
                        // Build: (left AND NOT right) OR (NOT left AND right)
                        let left_and_not_right = crate::storage::expression::AndExpr::new(vec![
                            left,
                            Box::new(crate::storage::expression::NotExpr::new(right)),
                        ]);
                        let not_left_and_right = crate::storage::expression::AndExpr::new(vec![
                            Box::new(crate::storage::expression::NotExpr::new(left2)),
                            right2,
                        ]);
                        Ok(Box::new(crate::storage::expression::OrExpr::new(vec![
                            Box::new(left_and_not_right),
                            Box::new(not_left_and_right),
                        ])))
                    }
                    InfixOperator::Equal
                    | InfixOperator::NotEqual
                    | InfixOperator::LessThan
                    | InfixOperator::LessEqual
                    | InfixOperator::GreaterThan
                    | InfixOperator::GreaterEqual => {
                        // Simple comparison - use pre-computed operator
                        let (column, value) =
                            self.extract_comparison_with_ctx(&infix.left, &infix.right, ctx)?;
                        let operator = self.infix_op_to_core_op(infix.op_type);
                        // Coerce value to column type for proper comparison
                        // (e.g., WHERE float_col = 5 should work with integer literal)
                        // OPTIMIZATION: Use into_coerce_to_type to avoid clone when types match
                        let coerced_value = if let Some(col_type) = schema
                            .column_index_map()
                            .get(column)
                            .and_then(|&idx| schema.columns.get(idx).map(|c| c.data_type))
                        {
                            value.into_coerce_to_type(col_type)
                        } else {
                            value
                        };
                        let mut expr = ComparisonExpr::new(column, operator, coerced_value);
                        expr.prepare_for_schema(schema);
                        Ok(Box::new(expr))
                    }
                    InfixOperator::Like => {
                        let (column, pattern) =
                            self.extract_comparison_with_ctx(&infix.left, &infix.right, ctx)?;
                        let pattern_str = match pattern {
                            Value::Text(s) => s.to_string(),
                            _ => {
                                return Err(Error::Type(
                                    "LIKE pattern must be a string".to_string(),
                                ))
                            }
                        };
                        let mut expr =
                            crate::storage::expression::LikeExpr::new(column, pattern_str);
                        expr.prepare_for_schema(schema);
                        Ok(Box::new(expr))
                    }
                    InfixOperator::ILike => {
                        let (column, pattern) =
                            self.extract_comparison_with_ctx(&infix.left, &infix.right, ctx)?;
                        let pattern_str = match pattern {
                            Value::Text(s) => s.to_string(),
                            _ => {
                                return Err(Error::Type(
                                    "ILIKE pattern must be a string".to_string(),
                                ))
                            }
                        };
                        let mut expr =
                            crate::storage::expression::LikeExpr::new_ilike(column, pattern_str);
                        expr.prepare_for_schema(schema);
                        Ok(Box::new(expr))
                    }
                    InfixOperator::NotLike => {
                        let (column, pattern) =
                            self.extract_comparison_with_ctx(&infix.left, &infix.right, ctx)?;
                        let pattern_str = match pattern {
                            Value::Text(s) => s.to_string(),
                            _ => {
                                return Err(Error::Type(
                                    "NOT LIKE pattern must be a string".to_string(),
                                ))
                            }
                        };
                        let mut like_expr =
                            crate::storage::expression::LikeExpr::new(column, pattern_str);
                        like_expr.prepare_for_schema(schema);
                        // Wrap in NOT expression
                        Ok(Box::new(crate::storage::expression::NotExpr::new(
                            Box::new(like_expr),
                        )))
                    }
                    InfixOperator::NotILike => {
                        let (column, pattern) =
                            self.extract_comparison_with_ctx(&infix.left, &infix.right, ctx)?;
                        let pattern_str = match pattern {
                            Value::Text(s) => s.to_string(),
                            _ => {
                                return Err(Error::Type(
                                    "NOT ILIKE pattern must be a string".to_string(),
                                ))
                            }
                        };
                        let mut like_expr =
                            crate::storage::expression::LikeExpr::new_ilike(column, pattern_str);
                        like_expr.prepare_for_schema(schema);
                        // Wrap in NOT expression
                        Ok(Box::new(crate::storage::expression::NotExpr::new(
                            Box::new(like_expr),
                        )))
                    }
                    InfixOperator::Is => {
                        // Handle IS NULL, IS TRUE, IS FALSE
                        let column = self.extract_column_name(&infix.left)?;
                        match infix.right.as_ref() {
                            Expression::NullLiteral(_) => {
                                // IS NULL
                                let mut expr =
                                    crate::storage::expression::NullCheckExpr::is_null(column);
                                expr.prepare_for_schema(schema);
                                Ok(Box::new(expr))
                            }
                            Expression::BooleanLiteral(b) => {
                                // IS TRUE or IS FALSE
                                let mut expr = ComparisonExpr::eq(column, Value::Boolean(b.value));
                                expr.prepare_for_schema(schema);
                                Ok(Box::new(expr))
                            }
                            _ => Err(Error::NotSupportedMessage(
                                "IS requires NULL, TRUE, or FALSE".to_string(),
                            )),
                        }
                    }
                    InfixOperator::IsNot => {
                        // Handle IS NOT NULL, IS NOT TRUE, IS NOT FALSE
                        let column = self.extract_column_name(&infix.left)?;
                        match infix.right.as_ref() {
                            Expression::NullLiteral(_) => {
                                // IS NOT NULL
                                let mut expr =
                                    crate::storage::expression::NullCheckExpr::is_not_null(column);
                                expr.prepare_for_schema(schema);
                                Ok(Box::new(expr))
                            }
                            Expression::BooleanLiteral(b) => {
                                // IS NOT TRUE or IS NOT FALSE
                                // This is equivalent to (col <> TRUE/FALSE OR col IS NULL)
                                let mut ne_expr =
                                    ComparisonExpr::ne(column, Value::Boolean(b.value));
                                ne_expr.prepare_for_schema(schema);
                                let mut null_expr =
                                    crate::storage::expression::NullCheckExpr::is_null(column);
                                null_expr.prepare_for_schema(schema);
                                Ok(Box::new(crate::storage::expression::OrExpr::new(vec![
                                    Box::new(ne_expr),
                                    Box::new(null_expr),
                                ])))
                            }
                            _ => Err(Error::NotSupportedMessage(
                                "IS NOT requires NULL, TRUE, or FALSE".to_string(),
                            )),
                        }
                    }
                    _ => Err(Error::NotSupportedMessage(format!(
                        "Operator {} in WHERE clause",
                        infix.operator
                    ))),
                }
            }
            Expression::Prefix(prefix) => {
                // OPTIMIZATION: Use pre-computed op_type enum instead of string comparison
                use crate::parser::ast::PrefixOperator;
                if prefix.op_type == PrefixOperator::Not {
                    let inner =
                        self.build_storage_expression_with_ctx(&prefix.right, schema, ctx)?;
                    Ok(Box::new(crate::storage::expression::NotExpr::new(inner)))
                } else {
                    Err(Error::NotSupportedMessage(format!(
                        "Prefix operator {} in WHERE clause",
                        prefix.operator
                    )))
                }
            }
            Expression::Between(between) => {
                let column = self.extract_column_name(&between.expr)?;
                let lower = self.evaluate_literal_with_ctx(&between.lower, ctx)?;
                let upper = self.evaluate_literal_with_ctx(&between.upper, ctx)?;
                // Coerce values to column type
                // OPTIMIZATION: Use into_coerce_to_type to avoid clone when types match
                let (lower, upper) = if let Some(col_type) = schema
                    .column_index_map()
                    .get(column)
                    .and_then(|&idx| schema.columns.get(idx).map(|c| c.data_type))
                {
                    (
                        lower.into_coerce_to_type(col_type),
                        upper.into_coerce_to_type(col_type),
                    )
                } else {
                    (lower, upper)
                };
                // Use not_between for NOT BETWEEN to handle NULL correctly
                // (NOT(NULL) = NULL = false in WHERE context)
                let mut expr = if between.not {
                    crate::storage::expression::BetweenExpr::not_between(column, lower, upper)
                } else {
                    crate::storage::expression::BetweenExpr::new(column, lower, upper)
                };
                expr.prepare_for_schema(schema);
                Ok(Box::new(expr))
            }
            Expression::In(in_expr) => {
                let column = self.extract_column_name(&in_expr.left)?;
                let values = self.extract_in_list_values_with_ctx(&in_expr.right, ctx)?;
                // Coerce values to column type
                // OPTIMIZATION: Use into_coerce_to_type to avoid clone when types match
                let values = if let Some(col_type) = schema
                    .column_index_map()
                    .get(column)
                    .and_then(|&idx| schema.columns.get(idx).map(|c| c.data_type))
                {
                    values
                        .into_iter()
                        .map(|v| v.into_coerce_to_type(col_type))
                        .collect()
                } else {
                    values
                };
                let mut expr = if in_expr.not {
                    crate::storage::expression::InListExpr::not_in(column, values)
                } else {
                    crate::storage::expression::InListExpr::new(column, values)
                };
                expr.prepare_for_schema(schema);
                Ok(Box::new(expr))
            }
            Expression::BooleanLiteral(bool_lit) => {
                // Handle boolean literals (e.g., from EXISTS subquery evaluation)
                Ok(Box::new(crate::storage::expression::ConstBoolExpr::new(
                    bool_lit.value,
                )))
            }
            _ => Err(Error::NotSupportedMessage(
                "Expression type in WHERE clause not supported".to_string(),
            )),
        }
    }

    /// Try to extract pushable conjuncts from a WHERE clause for partial pushdown.
    ///
    /// For AND expressions, if one side can be pushed to storage and the other cannot,
    /// returns the pushable portion. This enables using indexes for simple predicates
    /// while filtering complex predicates in memory.
    ///
    /// Returns: (Option<StorageExpr>, needs_memory_filter)
    /// - Some(expr) means we have a storage expression to push down
    /// - needs_memory_filter=true means there are predicates that couldn't be pushed
    ///
    /// Example: `WHERE indexed_col = 5 AND complex_func(x) > 0`
    /// Returns: (Some(indexed_col = 5), true) - push indexed_col, memory filter the rest
    pub(crate) fn try_extract_pushable_conjuncts(
        &self,
        expr: &Expression,
        schema: &crate::core::Schema,
        ctx: Option<&ExecutionContext>,
    ) -> (Option<Box<dyn StorageExpr>>, bool) {
        use crate::parser::ast::InfixOperator;

        match expr {
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // For AND: try both sides, combine what we can push
                let (left_pushable, left_needs_mem) =
                    self.try_extract_pushable_conjuncts(&infix.left, schema, ctx);
                let (right_pushable, right_needs_mem) =
                    self.try_extract_pushable_conjuncts(&infix.right, schema, ctx);

                let needs_memory_filter = left_needs_mem || right_needs_mem;

                match (left_pushable, right_pushable) {
                    (Some(left), Some(right)) => {
                        // Both sides pushable - combine with AND
                        (
                            Some(Box::new(crate::storage::expression::AndExpr::new(vec![
                                left, right,
                            ]))),
                            needs_memory_filter,
                        )
                    }
                    (Some(expr), None) | (None, Some(expr)) => {
                        // Only one side pushable - push that side, memory filter the rest
                        (Some(expr), true)
                    }
                    (None, None) => {
                        // Neither side pushable - need full memory filter
                        (None, true)
                    }
                }
            }
            Expression::Infix(infix) if infix.op_type == InfixOperator::Or => {
                // For OR: can only push if BOTH sides are fully pushable
                // (partial pushdown of OR would change semantics)
                match self.build_storage_expression_with_ctx(expr, schema, ctx) {
                    Ok(storage_expr) => (Some(storage_expr), false),
                    Err(_) => (None, true),
                }
            }
            _ => {
                // Other expressions: try to convert entirely
                match self.build_storage_expression_with_ctx(expr, schema, ctx) {
                    Ok(storage_expr) => (Some(storage_expr), false),
                    Err(_) => (None, true),
                }
            }
        }
    }

    /// Extract values from an IN list expression
    #[allow(dead_code)]
    fn extract_in_list_values(&self, expr: &Expression) -> Result<Vec<Value>> {
        match expr {
            Expression::List(list) => list
                .elements
                .iter()
                .map(|e| self.evaluate_literal(e))
                .collect(),
            Expression::ExpressionList(list) => list
                .expressions
                .iter()
                .map(|e| self.evaluate_literal(e))
                .collect(),
            _ => {
                // Single value
                Ok(vec![self.evaluate_literal(expr)?])
            }
        }
    }

    /// Extract column name and value from comparison operands
    #[allow(dead_code)]
    fn extract_comparison<'a>(
        &self,
        left: &'a Expression,
        right: &'a Expression,
    ) -> Result<(&'a str, Value)> {
        // Try left as column, right as value
        if let Ok(column) = self.extract_column_name(left) {
            if let Ok(value) = self.evaluate_literal(right) {
                return Ok((column, value));
            }
        }

        // Try right as column, left as value
        if let Ok(column) = self.extract_column_name(right) {
            if let Ok(value) = self.evaluate_literal(left) {
                return Ok((column, value));
            }
        }

        Err(Error::InvalidArgumentMessage(
            "Comparison must have a column and a literal value".to_string(),
        ))
    }

    /// Extract column name from an expression
    /// OPTIMIZATION: Return reference to pre-computed lowercase value to avoid allocation
    fn extract_column_name<'a>(&self, expr: &'a Expression) -> Result<&'a str> {
        match expr {
            Expression::Identifier(id) => Ok(&id.value_lower),
            Expression::QualifiedIdentifier(qid) => Ok(&qid.name.value_lower),
            _ => Err(Error::InvalidArgumentMessage(
                "Expected column reference".to_string(),
            )),
        }
    }

    /// Evaluate a literal expression to a Value
    #[allow(dead_code)]
    fn evaluate_literal(&self, expr: &Expression) -> Result<Value> {
        self.evaluate_literal_with_ctx(expr, None)
    }

    /// Evaluate a literal expression to a Value, with context for parameter binding
    fn evaluate_literal_with_ctx(
        &self,
        expr: &Expression,
        ctx: Option<&ExecutionContext>,
    ) -> Result<Value> {
        match expr {
            Expression::IntegerLiteral(lit) => Ok(Value::Integer(lit.value)),
            Expression::FloatLiteral(lit) => Ok(Value::Float(lit.value)),
            Expression::StringLiteral(lit) => {
                // Handle typed literals (TIMESTAMP '...', DATE '...', TIME '...')
                if let Some(ref type_hint) = lit.type_hint {
                    match type_hint.to_uppercase().as_str() {
                        "TIMESTAMP" => {
                            crate::core::value::parse_timestamp(&lit.value).map(Value::Timestamp)
                        }
                        "DATE" => {
                            // Parse date and convert to timestamp at midnight
                            crate::core::value::parse_timestamp(&lit.value).map(Value::Timestamp)
                        }
                        "TIME" => {
                            // Parse time and convert to timestamp
                            crate::core::value::parse_timestamp(&lit.value).map(Value::Timestamp)
                        }
                        _ => Ok(Value::Text(std::sync::Arc::from(lit.value.as_str()))),
                    }
                } else {
                    Ok(Value::Text(std::sync::Arc::from(lit.value.as_str())))
                }
            }
            Expression::BooleanLiteral(lit) => Ok(Value::Boolean(lit.value)),
            Expression::NullLiteral(_) => Ok(Value::null_unknown()),
            Expression::Parameter(param) => {
                if let Some(ctx) = ctx {
                    // Check if it's a named parameter (starts with :)
                    if param.name.starts_with(':') {
                        let name = &param.name[1..];
                        if let Some(value) = ctx.get_named_param(name) {
                            Ok(value.clone())
                        } else {
                            Err(Error::InvalidArgumentMessage(format!(
                                "Named parameter '{}' not found",
                                param.name
                            )))
                        }
                    } else {
                        // Positional parameter
                        let params = ctx.params();
                        if param.index > 0 && param.index <= params.len() {
                            Ok(params[param.index - 1].clone())
                        } else {
                            Err(Error::InvalidArgumentMessage(format!(
                                "Parameter index {} out of range (have {} parameters)",
                                param.index,
                                params.len()
                            )))
                        }
                    }
                } else {
                    Err(Error::InvalidArgumentMessage(
                        "Parameters require execution context".to_string(),
                    ))
                }
            }
            // Handle identifier references from outer query context (for correlated subqueries)
            Expression::Identifier(id) => {
                if let Some(ctx) = ctx {
                    if let Some(outer_row) = ctx.outer_row() {
                        let name = id.value.to_lowercase();
                        if let Some(value) = outer_row.get(&name) {
                            return Ok(value.clone());
                        }
                    }
                }
                Err(Error::InvalidArgumentMessage(
                    "Expected literal value".to_string(),
                ))
            }
            // Handle qualified identifier references from outer query context (e.g., c.id)
            Expression::QualifiedIdentifier(qid) => {
                if let Some(ctx) = ctx {
                    if let Some(outer_row) = ctx.outer_row() {
                        // Try qualified name first (e.g., "c.id")
                        let qualified_name = format!(
                            "{}.{}",
                            qid.qualifier.value.to_lowercase(),
                            qid.name.value.to_lowercase()
                        );
                        if let Some(value) = outer_row.get(&qualified_name) {
                            return Ok(value.clone());
                        }
                        // Try just the column name (e.g., "id")
                        let name = qid.name.value.to_lowercase();
                        if let Some(value) = outer_row.get(&name) {
                            return Ok(value.clone());
                        }
                    }
                }
                Err(Error::InvalidArgumentMessage(
                    "Expected literal value".to_string(),
                ))
            }
            // Handle arithmetic expressions (e.g., 5 + 10, 100 * 2)
            // These are constant expressions that can be evaluated without row context
            // BUT only if they don't contain column references
            Expression::Infix(_) | Expression::Prefix(_) | Expression::FunctionCall(_) => {
                // Check if the expression contains any column references
                // If so, we can't evaluate it as a constant (needs row context)
                if Self::contains_column_reference(expr) {
                    return Err(Error::InvalidArgumentMessage(
                        "Expression contains column references and cannot be evaluated as a constant".to_string(),
                    ));
                }
                // Use the evaluator to evaluate constant expressions
                let evaluator = Evaluator::new(&self.function_registry);
                evaluator.evaluate(expr)
            }
            _ => Err(Error::InvalidArgumentMessage(
                "Expected literal value".to_string(),
            )),
        }
    }

    /// Extract values from an IN list expression with context
    fn extract_in_list_values_with_ctx(
        &self,
        expr: &Expression,
        ctx: Option<&ExecutionContext>,
    ) -> Result<Vec<Value>> {
        match expr {
            Expression::List(list) => list
                .elements
                .iter()
                .map(|e| self.evaluate_literal_with_ctx(e, ctx))
                .collect(),
            Expression::ExpressionList(list) => list
                .expressions
                .iter()
                .map(|e| self.evaluate_literal_with_ctx(e, ctx))
                .collect(),
            _ => {
                // Single value
                Ok(vec![self.evaluate_literal_with_ctx(expr, ctx)?])
            }
        }
    }

    /// Extract column name and value from comparison operands with context
    /// OPTIMIZATION: Returns &str reference to avoid String allocation
    fn extract_comparison_with_ctx<'a>(
        &self,
        left: &'a Expression,
        right: &'a Expression,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(&'a str, Value)> {
        // Try left as column, right as value
        if let Ok(column) = self.extract_column_name(left) {
            if let Ok(value) = self.evaluate_literal_with_ctx(right, ctx) {
                return Ok((column, value));
            }
        }

        // Try right as column, left as value
        if let Ok(column) = self.extract_column_name(right) {
            if let Ok(value) = self.evaluate_literal_with_ctx(left, ctx) {
                return Ok((column, value));
            }
        }

        Err(Error::InvalidArgumentMessage(
            "Comparison must have a column and a literal value".to_string(),
        ))
    }

    /// Convert InfixOperator enum to core Operator enum (no string allocation)
    #[inline]
    fn infix_op_to_core_op(&self, op: crate::parser::ast::InfixOperator) -> crate::core::Operator {
        use crate::parser::ast::InfixOperator;
        match op {
            InfixOperator::Equal => crate::core::Operator::Eq,
            InfixOperator::NotEqual => crate::core::Operator::Ne,
            InfixOperator::LessThan => crate::core::Operator::Lt,
            InfixOperator::GreaterThan => crate::core::Operator::Gt,
            InfixOperator::LessEqual => crate::core::Operator::Lte,
            InfixOperator::GreaterEqual => crate::core::Operator::Gte,
            // These shouldn't be called but map to Eq as fallback
            _ => crate::core::Operator::Eq,
        }
    }

    /// Check if an expression contains any column references (Identifier or QualifiedIdentifier)
    /// This is used to determine if an expression can be evaluated as a constant.
    fn contains_column_reference(expr: &Expression) -> bool {
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => true,
            Expression::Infix(infix) => {
                Self::contains_column_reference(&infix.left)
                    || Self::contains_column_reference(&infix.right)
            }
            Expression::Prefix(prefix) => Self::contains_column_reference(&prefix.right),
            Expression::FunctionCall(func) => {
                func.arguments.iter().any(Self::contains_column_reference)
            }
            Expression::Case(case) => {
                case.value
                    .as_ref()
                    .is_some_and(|e| Self::contains_column_reference(e))
                    || case.when_clauses.iter().any(|w| {
                        Self::contains_column_reference(&w.condition)
                            || Self::contains_column_reference(&w.then_result)
                    })
                    || case
                        .else_value
                        .as_ref()
                        .is_some_and(|e| Self::contains_column_reference(e))
            }
            Expression::Cast(cast) => Self::contains_column_reference(&cast.expr),
            Expression::List(list) => list.elements.iter().any(Self::contains_column_reference),
            Expression::ExpressionList(list) => {
                list.expressions.iter().any(Self::contains_column_reference)
            }
            Expression::In(in_expr) => {
                Self::contains_column_reference(&in_expr.left)
                    || Self::contains_column_reference(&in_expr.right)
            }
            Expression::Between(between) => {
                Self::contains_column_reference(&between.expr)
                    || Self::contains_column_reference(&between.lower)
                    || Self::contains_column_reference(&between.upper)
            }
            Expression::Aliased(aliased) => Self::contains_column_reference(&aliased.expression),
            // Literals and other non-column expressions
            _ => false,
        }
    }

    /// Apply ON DUPLICATE KEY UPDATE to an existing row
    fn apply_on_duplicate_update(
        &self,
        table: &mut Box<dyn Table>,
        schema: &crate::core::Schema,
        row_id: i64,
        _insert_values: &[Value],
        stmt: &InsertStatement,
        ctx: &ExecutionContext,
    ) -> Result<()> {
        // Build a WHERE clause to find the specific row by primary key
        let pk_col = schema
            .columns
            .iter()
            .find(|c| c.primary_key)
            .map(|c| c.name.clone());

        let where_expr: Option<Box<dyn StorageExpr>> = if let Some(pk_name) = pk_col {
            let mut expr =
                ComparisonExpr::new(pk_name, crate::core::Operator::Eq, Value::Integer(row_id));
            expr.prepare_for_schema(schema);
            Some(Box::new(expr))
        } else {
            None
        };

        // OPTIMIZATION: Pre-compute column indices and types to avoid per-row linear search
        let update_specs: Vec<(usize, crate::core::DataType, &Expression)> = stmt
            .update_columns
            .iter()
            .zip(stmt.update_expressions.iter())
            .filter_map(|(col, expr)| {
                schema
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(&col.value))
                    .map(|idx| (idx, schema.columns[idx].data_type, expr))
            })
            .collect();

        let function_registry = &self.function_registry;
        let ctx_clone = ctx.clone();
        let column_names: Vec<String> = schema.column_names_owned().to_vec();

        // Create evaluator once and reuse for all rows
        let mut evaluator = Evaluator::new(function_registry);
        evaluator = evaluator.with_context(&ctx_clone);
        evaluator.init_columns(&column_names);

        // Create a setter function that applies the ON DUPLICATE KEY UPDATE
        let mut setter = |mut row: Row| -> (Row, bool) {
            evaluator.set_row_array(&row);

            let mut changed = false;

            // Use pre-computed indices instead of per-row position() lookup
            for (idx, col_type, expr) in &update_specs {
                if let Ok(new_value) = evaluator.evaluate(expr) {
                    // OPTIMIZATION: Use into_coerce_to_type to avoid clone when types match
                    let _ = row.set(*idx, new_value.into_coerce_to_type(*col_type));
                    changed = true;
                }
            }

            (row, changed)
        };

        // Update the row
        table.update(where_expr.as_deref(), &mut setter)?;

        Ok(())
    }

    /// Find a row by unique index value
    fn find_row_by_unique_index(
        &self,
        table: &dyn Table,
        schema: &crate::core::Schema,
        _index_name: &str,
        column_name: &str,
        row_values: &[Value],
    ) -> Result<Option<i64>> {
        // Find the column index
        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(column_name));

        if col_idx.is_none() {
            return Ok(None);
        }

        let col_idx = col_idx.unwrap();
        let value = row_values
            .get(col_idx)
            .cloned()
            .unwrap_or(Value::null_unknown());

        // Create a search expression for this value
        let mut expr =
            ComparisonExpr::new(column_name.to_string(), crate::core::Operator::Eq, value);
        expr.prepare_for_schema(schema);

        // Scan for the row
        let column_indices: Vec<usize> = (0..schema.columns.len()).collect();
        let mut scanner = table.scan(&column_indices, Some(&expr))?;

        // Get the first matching row's ID
        let result = if scanner.next() {
            let row = scanner.take_row();
            // Find the primary key column to get the row_id
            let mut found_id = None;
            for (i, col) in schema.columns.iter().enumerate() {
                if col.primary_key {
                    if let Some(Value::Integer(id)) = row.get(i) {
                        found_id = Some(*id);
                        break;
                    }
                }
            }
            found_id
        } else {
            None
        };

        scanner.close()?;
        Ok(result)
    }

    /// Evaluate a default expression string and return the resulting Value
    pub(crate) fn evaluate_default_expr(
        &self,
        default_expr: &str,
        target_type: crate::core::DataType,
    ) -> Result<Value> {
        use crate::parser::parse_sql;

        // Parse the default expression as a SELECT expression
        let sql = format!("SELECT {}", default_expr);
        let stmts = match parse_sql(&sql) {
            Ok(s) => s,
            Err(_) => return Ok(Value::null_unknown()),
        };
        if stmts.is_empty() {
            return Ok(Value::null_unknown());
        }

        // Extract the expression from the SELECT statement
        if let crate::parser::ast::Statement::Select(select) = &stmts[0] {
            if let Some(expr) = select.columns.first() {
                let evaluator = Evaluator::new(&self.function_registry);
                let value = evaluator.evaluate(expr)?;
                return Ok(value.into_coerce_to_type(target_type));
            }
        }

        Ok(Value::null_unknown())
    }

    /// Validate a CHECK constraint against row values
    /// Returns Ok(()) if the constraint passes, Err if it fails
    pub(crate) fn validate_check_constraint(
        &self,
        check_expr: &str,
        col_name: &str,
        col_value: &Value,
        _col_type: crate::core::DataType,
    ) -> Result<()> {
        use crate::parser::parse_sql;

        // NULL values pass CHECK constraints (SQL standard)
        if col_value.is_null() {
            return Ok(());
        }

        // Parse the check expression
        let sql = format!("SELECT {}", check_expr);
        let stmts = match parse_sql(&sql) {
            Ok(s) => s,
            Err(_) => return Ok(()), // If we can't parse, skip validation
        };
        if stmts.is_empty() {
            return Ok(());
        }

        // Create an evaluator with the column value in context
        if let crate::parser::ast::Statement::Select(select) = &stmts[0] {
            if let Some(expr) = select.columns.first() {
                // Create evaluator and evaluate with row context
                let columns = vec![col_name.to_string()];
                let row = crate::core::Row::from_values(vec![col_value.clone()]);
                let evaluator = Evaluator::new(&self.function_registry).with_row(row, &columns);

                let result = evaluator.evaluate(expr)?;

                // Check if the result is truthy
                match result {
                    Value::Boolean(true) => Ok(()),
                    Value::Boolean(false) => Err(Error::CheckConstraintViolation {
                        column: col_name.to_string(),
                        expression: check_expr.to_string(),
                    }),
                    Value::Null(_) => {
                        // NULL passes CHECK constraint (SQL standard)
                        Ok(())
                    }
                    _ => {
                        // Non-boolean result - treat non-zero/non-empty as true
                        let is_truthy = match &result {
                            Value::Integer(i) => *i != 0,
                            Value::Float(f) => *f != 0.0,
                            Value::Text(s) => !s.is_empty(),
                            _ => false,
                        };
                        if is_truthy {
                            Ok(())
                        } else {
                            Err(Error::CheckConstraintViolation {
                                column: col_name.to_string(),
                                expression: check_expr.to_string(),
                            })
                        }
                    }
                }
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Build a result from RETURNING clause expressions
    ///
    /// Evaluates the RETURNING expressions for each affected row and returns
    /// the results as a QueryResult.
    fn build_returning_result(
        &self,
        returning: &[Expression],
        source_rows: Vec<Row>,
        column_names: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use super::result::ExecutorMemoryResult;
        use crate::parser::{Identifier, Position, Token, TokenType};

        // Expand Star expressions to all columns
        let mut expanded_exprs: Vec<Expression> = Vec::new();
        let mut result_columns: Vec<String> = Vec::new();

        for (i, expr) in returning.iter().enumerate() {
            match expr {
                Expression::Star(_) => {
                    // Expand * to all columns
                    for col_name in column_names {
                        result_columns.push(col_name.clone());
                        let token = Token::new(
                            TokenType::Identifier,
                            col_name.clone(),
                            Position::new(0, 0, 0),
                        );
                        expanded_exprs.push(Expression::Identifier(Identifier::new(
                            token,
                            col_name.clone(),
                        )));
                    }
                }
                _ => {
                    result_columns.push(Self::get_returning_column_name(expr, i));
                    expanded_exprs.push(expr.clone());
                }
            }
        }

        // If no source rows, return empty result
        if source_rows.is_empty() {
            return Ok(Box::new(ExecutorMemoryResult::new(
                result_columns,
                Vec::new(),
            )));
        }

        // Create evaluator for RETURNING expressions
        let mut evaluator = Evaluator::new(&self.function_registry).with_context(ctx);
        evaluator.init_columns(column_names);

        // Evaluate RETURNING expressions for each row
        let mut result_rows = Vec::with_capacity(source_rows.len());
        for row in source_rows {
            evaluator.set_row_array(&row);

            let mut row_values = Vec::with_capacity(expanded_exprs.len());
            for expr in &expanded_exprs {
                let value = evaluator
                    .evaluate(expr)
                    .unwrap_or_else(|_| Value::null_unknown());
                row_values.push(value);
            }
            result_rows.push(Row::from_values(row_values));
        }

        Ok(Box::new(ExecutorMemoryResult::new(
            result_columns,
            result_rows,
        )))
    }

    /// Get a column name for a RETURNING expression
    fn get_returning_column_name(expr: &Expression, index: usize) -> String {
        match expr {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            Expression::Star(_) => "*".to_string(),
            Expression::Aliased(aliased) => aliased.alias.value.clone(),
            Expression::FunctionCall(func) => {
                let args: Vec<String> = func
                    .arguments
                    .iter()
                    .map(|a| Self::get_returning_column_name(a, 0))
                    .collect();
                format!("{}({})", func.function, args.join(", "))
            }
            _ => format!("column_{}", index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mvcc::engine::MVCCEngine;
    use std::sync::Arc;

    fn create_test_executor() -> Executor {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        Executor::new(Arc::new(engine))
    }

    fn setup_test_table(executor: &Executor) {
        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();
    }

    #[test]
    fn test_insert_single_row() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        let result = executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);
    }

    #[test]
    fn test_insert_multiple_rows() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        let result = executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)")
            .unwrap();
        assert_eq!(result.rows_affected(), 2);
    }

    #[test]
    fn test_insert_and_select() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();

        let mut result = executor.execute("SELECT * FROM users").unwrap();
        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("Alice")));
        assert_eq!(row.get(2), Some(&Value::Integer(30)));
    }

    #[test]
    fn test_type_coercion_insert_int_to_float() {
        let executor = create_test_executor();
        // Create table with FLOAT column
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();

        // Insert integer into float column - should coerce 5 -> 5.0
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5)")
            .unwrap();

        let mut result = executor.execute("SELECT price FROM products").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Float(5.0), not Integer(5)
        assert_eq!(row.get(0), Some(&Value::Float(5.0)));
    }

    #[test]
    fn test_type_coercion_insert_float_to_int() {
        let executor = create_test_executor();
        // Create table with INTEGER column
        executor
            .execute("CREATE TABLE counts (id INTEGER PRIMARY KEY, amount INTEGER)")
            .unwrap();

        // Insert float into integer column - should coerce 5.9 -> 5
        executor
            .execute("INSERT INTO counts (id, amount) VALUES (1, 5.9)")
            .unwrap();

        let mut result = executor.execute("SELECT amount FROM counts").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Integer(5), not Float(5.9)
        assert_eq!(row.get(0), Some(&Value::Integer(5)));
    }

    #[test]
    fn test_type_coercion_where_int_literal_on_float_column() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // Query with integer literal against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price = 5")
            .unwrap();
        assert!(result.next(), "Should find row with WHERE price = 5");
    }

    #[test]
    fn test_type_coercion_where_float_literal_on_int_column() {
        let executor = create_test_executor();
        setup_test_table(&executor);
        executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();

        // Query with float literal against integer column
        let mut result = executor
            .execute("SELECT * FROM users WHERE age = 30.0")
            .unwrap();
        assert!(result.next(), "Should find row with WHERE age = 30.0");
    }

    #[test]
    fn test_type_coercion_between() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // BETWEEN with integer literals against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price BETWEEN 4 AND 6")
            .unwrap();
        assert!(result.next(), "Should find row with BETWEEN 4 AND 6");
    }

    #[test]
    fn test_type_coercion_in_list() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // IN with integer literals against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price IN (4, 5, 6)")
            .unwrap();
        assert!(result.next(), "Should find row with IN (4, 5, 6)");
    }

    #[test]
    fn test_type_coercion_insert_text_to_timestamp() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();

        // Insert text string into timestamp column - should parse to timestamp
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        let mut result = executor.execute("SELECT created_at FROM events").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Timestamp, not Text
        match row.get(0) {
            Some(Value::Timestamp(_)) => {} // Success
            other => panic!("Expected Timestamp, got {:?}", other),
        }
    }

    #[test]
    fn test_type_coercion_where_text_on_timestamp_column() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        // Query with text literal against timestamp column
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at > '2024-01-01'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with WHERE created_at > '2024-01-01'"
        );

        // Query with exact match
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at = '2024-01-15 10:30:00'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with WHERE created_at = '2024-01-15 10:30:00'"
        );
    }

    #[test]
    fn test_type_coercion_timestamp_between() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        // BETWEEN with text literals against timestamp column
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-02-01'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with BETWEEN '2024-01-01' AND '2024-02-01'"
        );
    }
}
