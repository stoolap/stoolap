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
use ahash::AHashMap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::context::ExecutionContext;
use super::expression::CompiledEvaluator;
use super::pushdown;
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

/// Try to extract a literal value directly from an expression without VM compilation.
/// Returns Some(value) for simple literals, None for complex expressions that need VM.
#[inline]
fn try_extract_literal(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::StringLiteral(lit) => Some(Value::text(&lit.value)),
        Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
        Expression::NullLiteral(_) => Some(Value::null_unknown()),
        // Negative numbers: -5, -3.14
        Expression::Prefix(prefix) if prefix.operator == "-" => match prefix.right.as_ref() {
            Expression::IntegerLiteral(lit) => Some(Value::Integer(-lit.value)),
            Expression::FloatLiteral(lit) => Some(Value::Float(-lit.value)),
            _ => None,
        },
        _ => None, // Complex expression - needs VM
    }
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

        // Create VM for constant expression evaluation (reused for all INSERT values)
        use super::expression::{compile_expression, ExecuteContext, ExprVM};
        let mut vm = ExprVM::new();
        let params = ctx.params();
        let named_params = ctx.named_params();
        let empty_row: &[Value] = &[];

        // OPTIMIZATION: Pre-build ExecuteContext once (reused for all expressions)
        let mut base_exec_ctx = ExecuteContext::new(empty_row);
        if !params.is_empty() {
            base_exec_ctx = base_exec_ctx.with_params(params);
        }
        if !named_params.is_empty() {
            base_exec_ctx = base_exec_ctx.with_named_params(named_params);
        }

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
                    // OPTIMIZATION: Try to extract literal value directly without VM compilation
                    // This avoids ~1-2μs per expression for simple literals (INTEGER, TEXT, etc.)
                    let value = if let Some(lit_value) = try_extract_literal(expr) {
                        lit_value
                    } else {
                        // Fall back to VM for complex expressions (Parameters, functions, etc.)
                        let program = compile_expression(expr, &[])?;
                        vm.execute(&program, &base_exec_ctx)?
                    };
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
                    // OPTIMIZATION: Try to extract literal value directly without VM compilation
                    // This avoids ~1-2μs per expression for simple literals (INTEGER, TEXT, etc.)
                    let value = if let Some(lit_value) = try_extract_literal(expr) {
                        lit_value
                    } else {
                        // Fall back to VM for complex expressions (Parameters, functions, etc.)
                        let program = compile_expression(expr, &[])?;
                        vm.execute(&program, &base_exec_ctx)?
                    };
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

            // Try to push down predicate to storage layer
            let (storage_expr, needs_mem) =
                pushdown::try_pushdown(&processed_where, schema, Some(ctx));
            if needs_mem {
                // Complex expression (like a + b > 100) - use in-memory filtering
                (storage_expr, true, Some(processed_where))
            } else {
                (storage_expr, false, None)
            }
        } else {
            (None, false, None)
        };

        let function_registry = &self.function_registry;

        // Create evaluator once and reuse for all rows (optimization)
        let mut evaluator = CompiledEvaluator::new(function_registry).with_context(ctx);
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
            // OPTIMIZATION: Use AHashMap for Value keys (better hash distribution)
            let mut precomputed: AHashMap<Value, Vec<(usize, Value)>> = AHashMap::with_capacity(64);

            // Build column indices for scanning (all columns)
            let all_col_indices: Vec<usize> = (0..column_names_vec.len()).collect();
            let column_names_arc = Arc::new(column_names_vec.clone());

            // OPTIMIZATION: Pre-compute lowercase and qualified column names once
            let col_name_pairs: Vec<(String, String)> = column_names_vec
                .iter()
                .map(|col_name| {
                    let col_lower = col_name.to_lowercase();
                    let qualified = format!("{}.{}", table_name, col_lower);
                    (col_lower, qualified)
                })
                .collect();

            // Reusable outer_row_map - cleared and reused each iteration
            let mut outer_row_map: FxHashMap<String, Value> =
                FxHashMap::with_capacity_and_hasher(col_name_pairs.len() * 2, Default::default());

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

                // Build outer row context from current row values using pre-computed names
                outer_row_map.clear();
                for (i, (col_lower, qualified)) in col_name_pairs.iter().enumerate() {
                    if let Some(value) = row.get(i) {
                        outer_row_map.insert(col_lower.clone(), value.clone());
                        outer_row_map.insert(qualified.clone(), value.clone());
                    }
                }

                // Create context with outer row for correlated subquery evaluation
                // Move map into context, we'll take it back after
                let mut correlated_ctx = ctx
                    .with_outer_row(std::mem::take(&mut outer_row_map), column_names_arc.clone());

                // Evaluate all update expressions
                let mut new_values: Vec<(usize, Value)> = Vec::with_capacity(update_indices.len());
                for (idx, col_type, expr, is_correlated) in update_indices.iter() {
                    let evaluated = if *is_correlated {
                        // Process correlated expression - this executes the subquery
                        match self.process_correlated_expression(expr, &correlated_ctx) {
                            Ok(processed_expr) => {
                                // Now evaluate the processed expression (subquery replaced with value)
                                let mut eval = CompiledEvaluator::new(function_registry)
                                    .with_context(&correlated_ctx);
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

                // Take back the map for reuse (zero-copy transfer)
                outer_row_map = correlated_ctx.outer_row.take().unwrap_or_default();

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

                // Try to push down predicate to storage layer
                let (storage_expr, needs_mem) =
                    pushdown::try_pushdown(&processed_where, schema, Some(ctx));
                if needs_mem {
                    // Complex expression (like a + b > 100) - use in-memory filtering
                    (storage_expr, true, Some(processed_where))
                } else {
                    (storage_expr, false, None)
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
            let mut evaluator = CompiledEvaluator::new(&self.function_registry).with_context(ctx);
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

            // OPTIMIZATION: Pre-compute lowercase and qualified column names once
            // Each entry: (col_lower, effective_qualified, optional_table_qualified)
            let col_name_triples: Vec<(String, String, Option<String>)> = column_names_owned
                .iter()
                .map(|col_name| {
                    let col_lower = col_name.to_lowercase();
                    let effective_qualified = format!("{}.{}", effective_name, col_lower);
                    let table_qualified = if effective_name != table_name {
                        Some(format!("{}.{}", table_name, col_lower))
                    } else {
                        None
                    };
                    (col_lower, effective_qualified, table_qualified)
                })
                .collect();

            // Reusable outer_row_map for correlated subqueries
            let estimated_entries = col_name_triples.len() * 3; // up to 3 entries per column
            let mut outer_row_map: FxHashMap<String, Value> =
                FxHashMap::with_capacity_and_hasher(estimated_entries, Default::default());

            while scanner.next() {
                let row = scanner.row();

                // Check memory filter if needed
                let matches = if needs_memory_filter {
                    evaluator.set_row_array(row);
                    if let Some(ref where_expr) = memory_where_clause {
                        if has_correlated {
                            // Build outer row context using pre-computed names
                            outer_row_map.clear();
                            for (i, (col_lower, effective_qualified, table_qualified)) in
                                col_name_triples.iter().enumerate()
                            {
                                if let Some(value) = row.get(i) {
                                    outer_row_map.insert(col_lower.clone(), value.clone());
                                    outer_row_map
                                        .insert(effective_qualified.clone(), value.clone());
                                    if let Some(tq) = table_qualified {
                                        outer_row_map.insert(tq.clone(), value.clone());
                                    }
                                }
                            }

                            // Create context with outer row (move map, take it back later)
                            let mut correlated_ctx = ctx.with_outer_row(
                                std::mem::take(&mut outer_row_map),
                                column_names_arc.clone().unwrap(),
                            );

                            // Process correlated subquery with outer context
                            match self.process_correlated_where(where_expr, &correlated_ctx) {
                                Ok(processed) => {
                                    // OPTIMIZATION: Take ownership instead of cloning
                                    evaluator.set_outer_row_owned(
                                        correlated_ctx.outer_row.take().unwrap_or_default(),
                                    );
                                    let result =
                                        evaluator.evaluate_bool(&processed).unwrap_or(false);
                                    // Take back map for reuse instead of clearing
                                    outer_row_map = evaluator.take_outer_row();
                                    result
                                }
                                Err(_) => {
                                    // Take back map from context even on error
                                    outer_row_map =
                                        correlated_ctx.outer_row.take().unwrap_or_default();
                                    false
                                }
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

    /// Apply ON DUPLICATE KEY UPDATE to an existing row
    fn apply_on_duplicate_update(
        &self,
        table: &mut Box<dyn Table>,
        schema: &crate::core::Schema,
        row_id: i64,
        _insert_values: &[Value],
        stmt: &InsertStatement,
        _ctx: &ExecutionContext,
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

        let column_names: Vec<String> = schema.column_names_owned().to_vec();

        // Pre-compile update expressions for efficient evaluation
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};
        let compiled_updates: Vec<(usize, crate::core::DataType, SharedProgram)> = update_specs
            .iter()
            .filter_map(|(idx, col_type, expr)| {
                compile_expression(expr, &column_names)
                    .ok()
                    .map(|program| (*idx, *col_type, program))
            })
            .collect();

        // Create VM once and reuse for all rows
        let mut vm = ExprVM::new();

        // Create a setter function that applies the ON DUPLICATE KEY UPDATE
        let mut setter = |mut row: Row| -> (Row, bool) {
            // Collect all updates first to avoid borrow conflicts
            let updates_to_apply: Vec<(usize, Value)> = {
                let row_data = row.as_slice();
                let exec_ctx = ExecuteContext::new(row_data);

                compiled_updates
                    .iter()
                    .filter_map(|(idx, col_type, program)| {
                        vm.execute(program, &exec_ctx)
                            .ok()
                            .map(|v| (*idx, v.into_coerce_to_type(*col_type)))
                    })
                    .collect()
            };

            // Now apply updates
            let changed = !updates_to_apply.is_empty();
            for (idx, new_value) in updates_to_apply {
                let _ = row.set(idx, new_value);
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
        use super::expression::ExpressionEval;
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
                // Constant expression - no row context needed
                let value = ExpressionEval::compile(expr, &[])?.eval_slice(&[])?;
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
                use super::expression::ExpressionEval;

                // Create evaluator and evaluate with row context
                let columns = vec![col_name.to_string()];
                let row = crate::core::Row::from_values(vec![col_value.clone()]);

                // Compile and evaluate expression with single-column row context
                let result = ExpressionEval::compile(expr, &columns)?.eval(&row)?;

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

        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};

        // Pre-compile all RETURNING expressions
        let compiled_exprs: Vec<SharedProgram> = expanded_exprs
            .iter()
            .map(|expr| compile_expression(expr, column_names))
            .collect::<Result<Vec<_>>>()?;

        // Create VM for execution (reused for all rows)
        let mut vm = ExprVM::new();

        // Evaluate RETURNING expressions for each row
        let mut result_rows = Vec::with_capacity(source_rows.len());
        for row in source_rows {
            let row_data = row.as_slice();
            // CRITICAL: Include params from context for parameterized queries
            let exec_ctx = ExecuteContext::new(row_data)
                .with_params(ctx.params())
                .with_named_params(ctx.named_params());

            let mut row_values = Vec::with_capacity(compiled_exprs.len());
            for program in &compiled_exprs {
                // CRITICAL: Propagate errors instead of silently returning NULL
                let value = vm.execute(program, &exec_ctx)?;
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
