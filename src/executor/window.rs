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

//! Window Function Execution
//!
//! This module implements window function execution for SQL queries.
//!
//! Supports:
//! - ROW_NUMBER() - Sequential row numbering
//! - RANK() - Ranking with gaps
//! - DENSE_RANK() - Ranking without gaps
//! - NTILE(n) - Divides rows into n groups
//! - LEAD(col, offset, default) - Access next row's value
//! - LAG(col, offset, default) - Access previous row's value
//!
//! Window clauses:
//! - OVER () - Entire result set as one partition
//! - OVER (PARTITION BY col) - Partition by column values
//! - OVER (ORDER BY col) - Order within partition

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;

use crate::core::{Error, Result, Row, Value};

/// Type alias for partition keys - stack-allocated for common case (up to 4 columns)
type PartitionKey = SmallVec<[Value; 4]>;

use crate::functions::WindowFunction;
use crate::parser::ast::*;
use crate::storage::traits::{QueryResult, Table};

use super::context::ExecutionContext;
use super::expression::{ExpressionEval, MultiExpressionEval};
use super::join::build_column_index_map;
use super::result::ExecutorMemoryResult;
use super::Executor;

/// Information about a window function call in a SELECT list
#[derive(Clone, Debug)]
pub struct WindowFunctionInfo {
    /// The window function name (ROW_NUMBER, RANK, etc.)
    pub name: String,
    /// Arguments to the function (for LEAD, LAG, NTILE)
    pub arguments: Vec<Expression>,
    /// Partition by columns
    pub partition_by: Vec<String>,
    /// Order by expressions
    pub order_by: Vec<OrderByExpression>,
    /// Window frame specification (e.g., ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
    pub frame: Option<WindowFrame>,
    /// Result column name (may include alias)
    pub column_name: String,
    /// Index in the result columns
    #[allow(dead_code)]
    pub column_index: usize,
    /// Whether DISTINCT was specified (for COUNT(DISTINCT col) OVER())
    pub is_distinct: bool,
}

/// Information about a SELECT list item for window function processing
pub struct SelectItem {
    pub output_name: String,
    pub source: SelectItemSource,
}

/// Source of a SELECT item value
#[allow(clippy::large_enum_variant)]
pub enum SelectItemSource {
    BaseColumn(usize),
    /// Window function name - stored in lowercase for O(1) lookup in window_value_map
    WindowFunction(String),
    Expression(Expression),
    /// Expression containing a window function - needs special handling
    /// Stores (expression, synthetic_column_name_for_window_func)
    ExpressionWithWindow(Expression, String),
}

/// Pre-sorted state for window function optimization
/// When rows are pre-sorted by an indexed column, we can skip sorting in window functions
#[derive(Clone, Debug)]
pub struct WindowPreSortedState {
    /// Column name that rows are sorted by (lowercase)
    pub column: String,
    /// Whether sorted in ascending order
    pub ascending: bool,
}

/// Pre-grouped state for window function PARTITION BY optimization
/// When rows are fetched grouped by an indexed partition column, we can skip hash-based grouping
#[derive(Clone)]
pub struct WindowPreGroupedState {
    /// Column name that rows are partitioned by (lowercase)
    #[allow(dead_code)]
    pub column: String,
    /// Pre-built partition map: partition key -> row indices
    pub partition_map: FxHashMap<PartitionKey, Vec<usize>>,
}

impl Executor {
    /// Execute SELECT with window functions
    pub(crate) fn execute_select_with_window_functions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            None,
            None,
        )
    }

    /// Execute SELECT with window functions, with optional pre-sorted state
    /// When pre_sorted is Some, rows are already sorted by the specified column,
    /// allowing us to skip sorting for window functions that ORDER BY the same column
    pub(crate) fn execute_select_with_window_functions_presorted(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
        pre_sorted: Option<WindowPreSortedState>,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            pre_sorted,
            None,
        )
    }

    /// Execute SELECT with window functions, with pre-grouped partitions
    /// When pre_grouped is provided, rows are already grouped by partition column,
    /// allowing us to skip hash-based grouping for window functions
    pub(crate) fn execute_select_with_window_functions_pregrouped(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
        pre_grouped: WindowPreGroupedState,
    ) -> Result<Box<dyn QueryResult>> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            None,
            Some(pre_grouped),
        )
    }

    /// Internal implementation with pre-sorted and pre-grouped state parameters
    fn execute_select_with_window_functions_internal(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
        pre_sorted: Option<WindowPreSortedState>,
        pre_grouped: Option<WindowPreGroupedState>,
    ) -> Result<Box<dyn QueryResult>> {
        // Parse window functions from the SELECT list
        let window_functions = self.parse_window_functions(stmt, base_columns)?;

        if window_functions.is_empty() {
            // No window functions found, return base result
            return Ok(Box::new(ExecutorMemoryResult::new(
                base_columns.to_vec(),
                base_rows,
            )));
        }

        // OPTIMIZATION: LIMIT pushdown for PARTITION BY queries
        // If there's a LIMIT and PARTITION BY, we can process partitions one at a time
        // and stop early once we have enough rows (like SQLite does)
        if let Some(limit_expr) = &stmt.limit {
            // Check if we have PARTITION BY (not just ORDER BY)
            let has_partition_by = window_functions
                .iter()
                .any(|wf| !wf.partition_by.is_empty());

            if has_partition_by {
                // Try to evaluate the LIMIT expression
                if let Expression::IntegerLiteral(lit) = limit_expr.as_ref() {
                    let limit_val = lit.value;
                    if limit_val > 0 {
                        return self.execute_select_with_window_functions_streaming(
                            stmt,
                            ctx,
                            base_rows,
                            base_columns,
                            &window_functions,
                            limit_val as usize,
                        );
                    }
                }
            }
        }

        // Build column index map for base columns
        let mut col_index_map = build_column_index_map(base_columns);

        // Build a mapping from aggregate expression patterns to their column names
        // This handles cases like:
        // - SUM(val) AS grp_sum -> maps "sum(val)" to column index of "grp_sum"
        // - COALESCE(SUM(val), 0) AS total -> maps "sum(val)" to column index of "total"
        for col_expr in stmt.columns.iter() {
            if let Expression::Aliased(aliased) = col_expr {
                let alias_lower = aliased.alias.value.to_lowercase();
                if let Some(&idx) = col_index_map.get(&alias_lower) {
                    // Extract all aggregate patterns from this expression (including nested ones)
                    let patterns =
                        Self::extract_aggregate_patterns(aliased.expression.as_ref(), self);
                    for pattern in patterns {
                        col_index_map.insert(pattern.to_lowercase(), idx);
                    }
                }
            }
        }

        // Step 1: Compute all window function values upfront
        // OPTIMIZATION: Use FxHashMap for fastest lookups with trusted keys
        let mut window_value_map: FxHashMap<String, Vec<Value>> = FxHashMap::default();
        for wf in &window_functions {
            let window_values = self.compute_window_function(
                wf,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
                pre_sorted.as_ref(),
                pre_grouped.as_ref(),
            )?;
            window_value_map.insert(wf.column_name.to_lowercase(), window_values);
        }

        // Step 2: Build output columns and rows based on the SELECT list
        // The result should respect the SELECT list order, not just append window functions
        let mut result_columns = Vec::new();

        // Parse the SELECT list to determine output column order
        let select_items = self.parse_select_list_for_window(stmt, base_columns, &window_functions);

        for item in &select_items {
            result_columns.push(item.output_name.clone());
        }

        // Step 3: Build result rows based on SELECT list
        let num_rows = base_rows.len();
        let mut result_rows: Vec<Row> = Vec::with_capacity(num_rows);

        // Build aliases from col_index_map for expression evaluation
        let agg_aliases: Vec<(String, usize)> =
            col_index_map.iter().map(|(k, v)| (k.clone(), *v)).collect();

        // Pre-transform expressions with window functions by replacing Window expr with Identifier
        // This is done once, not per row
        let transformed_items: Vec<_> = select_items
            .iter()
            .map(|item| match &item.source {
                SelectItemSource::ExpressionWithWindow(expr, wf_name) => {
                    let transformed = Self::replace_window_with_identifier(expr, wf_name);
                    (item, Some(transformed), Some(wf_name.clone()))
                }
                _ => (item, None, None),
            })
            .collect();

        // Build extended columns (base_columns + synthetic window columns)
        let mut extended_columns = base_columns.to_vec();
        let mut added_wf_names: Vec<String> = Vec::new();
        for (_, _, wf_name_opt) in &transformed_items {
            if let Some(wf_name) = wf_name_opt {
                if !added_wf_names.contains(wf_name) {
                    extended_columns.push(wf_name.clone());
                    added_wf_names.push(wf_name.clone());
                }
            }
        }

        // Collect Expression items (with base_columns) and their indices
        let base_expr_items: Vec<(usize, &Expression)> = transformed_items
            .iter()
            .enumerate()
            .filter_map(|(i, (item, _, _))| {
                if let SelectItemSource::Expression(expr) = &item.source {
                    Some((i, expr))
                } else {
                    None
                }
            })
            .collect();

        // Collect ExpressionWithWindow transformed expressions and their indices
        let ext_expr_items: Vec<(usize, &Expression)> = transformed_items
            .iter()
            .enumerate()
            .filter_map(|(i, (_, transformed_opt, _))| transformed_opt.as_ref().map(|t| (i, t)))
            .collect();

        // Pre-compile base expressions (Expression items with base_columns)
        // CRITICAL: Propagate compilation errors instead of silently producing NULLs
        let base_exprs: Vec<Expression> =
            base_expr_items.iter().map(|(_, e)| (*e).clone()).collect();
        let mut base_eval = if !base_exprs.is_empty() {
            Some(
                MultiExpressionEval::compile_with_aliases(&base_exprs, base_columns, &agg_aliases)?
                    .with_context(ctx),
            )
        } else {
            None
        };

        // Pre-compile extended expressions (ExpressionWithWindow with extended_columns)
        // CRITICAL: Propagate compilation errors instead of silently producing NULLs
        let ext_exprs: Vec<Expression> = ext_expr_items.iter().map(|(_, e)| (*e).clone()).collect();
        let mut ext_eval = if !ext_exprs.is_empty() {
            Some(
                MultiExpressionEval::compile_with_aliases(
                    &ext_exprs,
                    &extended_columns,
                    &agg_aliases,
                )?
                .with_context(ctx),
            )
        } else {
            None
        };

        for (row_idx, base_row) in base_rows.iter().enumerate() {
            // Initialize result values with null
            let mut values: Vec<Value> = vec![Value::null_unknown(); select_items.len()];

            // Fill BaseColumn and WindowFunction items directly
            for (item_idx, (item, _, _)) in transformed_items.iter().enumerate() {
                match &item.source {
                    SelectItemSource::BaseColumn(col_idx) => {
                        values[item_idx] = base_row
                            .get(*col_idx)
                            .cloned()
                            .unwrap_or_else(Value::null_unknown);
                    }
                    SelectItemSource::WindowFunction(wf_name_lower) => {
                        values[item_idx] = window_value_map
                            .get(wf_name_lower)
                            .and_then(|vals| vals.get(row_idx).cloned())
                            .unwrap_or_else(Value::null_unknown);
                    }
                    _ => {} // Handled below
                }
            }

            // Evaluate base expressions and fill their values
            if let Some(ref mut eval) = base_eval {
                if let Ok(base_values) = eval.eval_all(base_row) {
                    for (eval_idx, (item_idx, _)) in base_expr_items.iter().enumerate() {
                        if eval_idx < base_values.len() {
                            values[*item_idx] = base_values[eval_idx].clone();
                        }
                    }
                }
            }

            // Evaluate extended expressions (if any) and fill their values
            if !ext_expr_items.is_empty() {
                if let Some(ref mut eval) = ext_eval {
                    // Build extended row: base_row + window function values
                    let mut ext_values = Vec::with_capacity(base_row.len() + added_wf_names.len());
                    ext_values.extend(base_row.iter().cloned());
                    for wf_name in &added_wf_names {
                        let wf_value = window_value_map
                            .get(wf_name)
                            .and_then(|vals| vals.get(row_idx).cloned())
                            .unwrap_or_else(Value::null_unknown);
                        ext_values.push(wf_value);
                    }
                    let ext_row = Row::from_values(ext_values);

                    if let Ok(ext_result_values) = eval.eval_all(&ext_row) {
                        for (eval_idx, (item_idx, _)) in ext_expr_items.iter().enumerate() {
                            if eval_idx < ext_result_values.len() {
                                values[*item_idx] = ext_result_values[eval_idx].clone();
                            }
                        }
                    }
                }
            }

            result_rows.push(Row::from_values(values));
        }

        Ok(Box::new(ExecutorMemoryResult::new(
            result_columns,
            result_rows,
        )))
    }

    /// Streaming execution for window functions with LIMIT pushdown
    /// Processes partitions one at a time and stops early when LIMIT is reached
    fn execute_select_with_window_functions_streaming(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
        window_functions: &[WindowFunctionInfo],
        limit: usize,
    ) -> Result<Box<dyn QueryResult>> {
        // Get the first window function with PARTITION BY (for partitioning logic)
        let wf_info = window_functions
            .iter()
            .find(|wf| !wf.partition_by.is_empty())
            .unwrap(); // Safe: we checked has_partition_by before calling this

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);

        // Get window function from registry
        let window_func = self
            .function_registry
            .get_window(&wf_info.name)
            .ok_or_else(|| {
                Error::NotSupportedMessage(format!("Unknown window function: {}", wf_info.name))
            })?;

        // Build partition map: group rows by partition key
        let partition_indices: SmallVec<[Option<usize>; 4]> = wf_info
            .partition_by
            .iter()
            .map(|part_col| {
                let lower = part_col.to_lowercase();
                col_index_map.get(&lower).copied().or_else(|| {
                    if let Some(dot_pos) = lower.rfind('.') {
                        col_index_map.get(&lower[dot_pos + 1..]).copied()
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut partitions: FxHashMap<PartitionKey, Vec<usize>> = FxHashMap::default();
        for (i, row) in base_rows.iter().enumerate() {
            let mut key: PartitionKey = SmallVec::with_capacity(partition_indices.len());
            for idx_opt in &partition_indices {
                let value = if let Some(&idx) = idx_opt.as_ref() {
                    row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
                } else {
                    Value::null_unknown()
                };
                key.push(value);
            }
            partitions.entry(key).or_default().push(i);
        }

        // Build result columns from SELECT list
        let select_items = self.parse_select_list_for_window(stmt, base_columns, window_functions);
        let result_columns: Vec<String> =
            select_items.iter().map(|i| i.output_name.clone()).collect();

        // Precompute ORDER BY values once for all rows
        let precomputed_order_by = if !wf_info.order_by.is_empty() {
            Some(self.precompute_order_by_values(
                &wf_info.order_by,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
            ))
        } else {
            None
        };

        // Process partitions one at a time, stopping when we have enough rows
        let mut result_rows: Vec<Row> = Vec::with_capacity(limit);

        // Convert to vec for sequential processing
        let partitions_vec: Vec<_> = partitions.into_iter().collect();

        for (_partition_key, row_indices) in partitions_vec {
            if result_rows.len() >= limit {
                break; // Early exit: we have enough rows
            }

            // Compute window function for this partition
            let (partition_results, sorted_indices) = self.compute_window_for_partition(
                &*window_func,
                wf_info,
                &base_rows,
                row_indices,
                precomputed_order_by.as_ref(),
                base_columns,
                &col_index_map,
                ctx,
                false,
            )?;

            // Build result rows in sorted order (within this partition)
            for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break; // Early exit within partition
                }

                let base_row = &base_rows[orig_idx];
                let window_value = &partition_results[sorted_pos];

                // Build output row
                let mut values: Vec<Value> = Vec::with_capacity(select_items.len());
                for item in &select_items {
                    match &item.source {
                        SelectItemSource::BaseColumn(col_idx) => {
                            values.push(
                                base_row
                                    .get(*col_idx)
                                    .cloned()
                                    .unwrap_or_else(Value::null_unknown),
                            );
                        }
                        SelectItemSource::WindowFunction(_) => {
                            values.push(window_value.clone());
                        }
                        SelectItemSource::Expression(expr) => {
                            // Evaluate expression
                            if let Ok(mut eval) = ExpressionEval::compile(expr, base_columns) {
                                let val = eval
                                    .eval(base_row)
                                    .unwrap_or_else(|_| Value::null_unknown());
                                values.push(val);
                            } else {
                                values.push(Value::null_unknown());
                            }
                        }
                        SelectItemSource::ExpressionWithWindow(expr, _wf_name) => {
                            // For expressions with window functions, we need to substitute
                            // the window function value
                            let transformed =
                                Self::replace_window_with_identifier(expr, &item.output_name);
                            let mut ext_columns = base_columns.to_vec();
                            ext_columns.push(item.output_name.clone());
                            let mut ext_values: Vec<Value> = base_row.iter().cloned().collect();
                            ext_values.push(window_value.clone());
                            let ext_row = Row::from_values(ext_values);
                            if let Ok(mut eval) =
                                ExpressionEval::compile(&transformed, &ext_columns)
                            {
                                let val = eval
                                    .eval(&ext_row)
                                    .unwrap_or_else(|_| Value::null_unknown());
                                values.push(val);
                            } else {
                                values.push(Value::null_unknown());
                            }
                        }
                    }
                }
                result_rows.push(Row::from_values(values));
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(
            result_columns,
            result_rows,
        )))
    }

    /// Lazy partition fetching for window functions with LIMIT pushdown
    /// Fetches partitions one at a time from the index and stops when LIMIT is reached
    /// This is the key optimization for PARTITION BY + LIMIT queries
    pub fn execute_select_with_window_functions_lazy_partition(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        table: &dyn Table,
        base_columns: &[String],
        partition_col: &str,
        limit: usize,
    ) -> Result<Box<dyn QueryResult>> {
        // Parse window functions from SELECT list
        let window_functions = self.parse_window_functions(stmt, base_columns)?;
        if window_functions.is_empty() {
            return Err(Error::internal(
                "No window functions found for lazy partition fetch",
            ));
        }

        // Get the first window function with PARTITION BY
        let wf_info = window_functions
            .iter()
            .find(|wf| !wf.partition_by.is_empty())
            .ok_or_else(|| Error::internal("No PARTITION BY window function found"))?;

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);

        // Get window function from registry
        let window_func = self
            .function_registry
            .get_window(&wf_info.name)
            .ok_or_else(|| {
                Error::NotSupportedMessage(format!("Unknown window function: {}", wf_info.name))
            })?;

        // Build result columns from SELECT list
        let select_items = self.parse_select_list_for_window(stmt, base_columns, &window_functions);
        let result_columns: Vec<String> =
            select_items.iter().map(|i| i.output_name.clone()).collect();

        // Get partition values from the index (lazy iteration key!)
        let partition_values = match table.get_partition_values(partition_col) {
            Some(values) => values,
            None => return Err(Error::internal("Failed to get partition values from index")),
        };

        // Process partitions one at a time, stopping when we have enough rows
        let mut result_rows: Vec<Row> = Vec::with_capacity(limit);

        for partition_value in partition_values {
            if result_rows.len() >= limit {
                break; // Early exit: we have enough rows
            }

            // Fetch rows for this partition only (KEY OPTIMIZATION!)
            let partition_rows =
                match table.get_rows_for_partition_value(partition_col, &partition_value) {
                    Some(rows) => rows,
                    None => continue,
                };

            if partition_rows.is_empty() {
                continue;
            }

            // Precompute ORDER BY values for this partition
            let precomputed_order_by = if !wf_info.order_by.is_empty() {
                Some(self.precompute_order_by_values(
                    &wf_info.order_by,
                    &partition_rows,
                    base_columns,
                    &col_index_map,
                    ctx,
                ))
            } else {
                None
            };

            // Compute window function for this partition
            let row_indices: Vec<usize> = (0..partition_rows.len()).collect();
            let (partition_results, sorted_indices) = self.compute_window_for_partition(
                &*window_func,
                wf_info,
                &partition_rows,
                row_indices,
                precomputed_order_by.as_ref(),
                base_columns,
                &col_index_map,
                ctx,
                false,
            )?;

            // Build result rows in sorted order (within this partition)
            for (sorted_pos, &row_idx) in sorted_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break; // Early exit within partition
                }

                let base_row = &partition_rows[row_idx];
                let window_value = &partition_results[sorted_pos];

                // Build output row
                let mut values: Vec<Value> = Vec::with_capacity(select_items.len());
                for item in &select_items {
                    match &item.source {
                        SelectItemSource::BaseColumn(col_idx) => {
                            values.push(
                                base_row
                                    .get(*col_idx)
                                    .cloned()
                                    .unwrap_or_else(Value::null_unknown),
                            );
                        }
                        SelectItemSource::WindowFunction(_) => {
                            values.push(window_value.clone());
                        }
                        SelectItemSource::Expression(expr) => {
                            if let Ok(mut eval) = ExpressionEval::compile(expr, base_columns) {
                                let val = eval
                                    .eval(base_row)
                                    .unwrap_or_else(|_| Value::null_unknown());
                                values.push(val);
                            } else {
                                values.push(Value::null_unknown());
                            }
                        }
                        SelectItemSource::ExpressionWithWindow(expr, _wf_name) => {
                            let transformed =
                                Self::replace_window_with_identifier(expr, &item.output_name);
                            let mut ext_columns = base_columns.to_vec();
                            ext_columns.push(item.output_name.clone());
                            let mut ext_values: Vec<Value> = base_row.iter().cloned().collect();
                            ext_values.push(window_value.clone());
                            let ext_row = Row::from_values(ext_values);
                            if let Ok(mut eval) =
                                ExpressionEval::compile(&transformed, &ext_columns)
                            {
                                let val = eval
                                    .eval(&ext_row)
                                    .unwrap_or_else(|_| Value::null_unknown());
                                values.push(val);
                            } else {
                                values.push(Value::null_unknown());
                            }
                        }
                    }
                }
                result_rows.push(Row::from_values(values));
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(
            result_columns,
            result_rows,
        )))
    }

    /// Parse SELECT list to determine output order and sources
    fn parse_select_list_for_window(
        &self,
        stmt: &SelectStatement,
        base_columns: &[String],
        window_functions: &[WindowFunctionInfo],
    ) -> Vec<SelectItem> {
        let col_index_map = build_column_index_map(base_columns);

        let mut items = Vec::new();
        let mut wf_idx = 0;

        for col_expr in &stmt.columns {
            match col_expr {
                Expression::Window(window_expr) => {
                    // Window function - use pre-computed values
                    let wf_name = if wf_idx < window_functions.len() {
                        window_functions[wf_idx].column_name.clone()
                    } else {
                        format!("{}()", window_expr.function.function)
                    };
                    items.push(SelectItem {
                        output_name: wf_name.clone(),
                        // OPTIMIZATION: Store lowercase for O(1) lookup in window_value_map
                        source: SelectItemSource::WindowFunction(wf_name.to_lowercase()),
                    });
                    wf_idx += 1;
                }
                Expression::Aliased(aliased) => {
                    if let Expression::Window(_) = aliased.expression.as_ref() {
                        // Aliased window function
                        let alias = aliased.alias.value.clone();
                        items.push(SelectItem {
                            output_name: alias.clone(),
                            // OPTIMIZATION: Store lowercase for O(1) lookup in window_value_map
                            source: SelectItemSource::WindowFunction(alias.to_lowercase()),
                        });
                        wf_idx += 1;
                    } else if let Expression::Identifier(id) = aliased.expression.as_ref() {
                        // Aliased column reference - use pre-computed lowercase
                        if let Some(&idx) = col_index_map.get(&id.value_lower) {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.clone(),
                                source: SelectItemSource::BaseColumn(idx),
                            });
                        } else {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.clone(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    } else if Self::find_window_in_expression(aliased.expression.as_ref()).is_some()
                    {
                        // Expression containing a window function
                        // Use the column_name from window_functions for consistency with window_value_map key
                        let wf_name = if wf_idx < window_functions.len() {
                            window_functions[wf_idx].column_name.clone()
                        } else {
                            format!("__wf_{}", wf_idx)
                        };
                        items.push(SelectItem {
                            output_name: aliased.alias.value.clone(),
                            source: SelectItemSource::ExpressionWithWindow(
                                aliased.expression.as_ref().clone(),
                                wf_name.to_lowercase(),
                            ),
                        });
                        wf_idx += 1;
                    } else if let Expression::FunctionCall(_) = aliased.expression.as_ref() {
                        // Aliased function call - check if it's an aggregate that's already in base_columns
                        // This happens when window functions are combined with GROUP BY aggregation
                        // The aggregate result is stored under the alias name (e.g., "total" for SUM(val) as total)
                        let alias_lower = aliased.alias.value.to_lowercase();
                        if let Some(&idx) = col_index_map.get(&alias_lower) {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.clone(),
                                source: SelectItemSource::BaseColumn(idx),
                            });
                        } else {
                            // Not found by alias, try expression evaluation
                            items.push(SelectItem {
                                output_name: aliased.alias.value.clone(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    } else {
                        // Other aliased expression
                        items.push(SelectItem {
                            output_name: aliased.alias.value.clone(),
                            source: SelectItemSource::Expression(
                                aliased.expression.as_ref().clone(),
                            ),
                        });
                    }
                }
                Expression::Identifier(id) => {
                    // Simple column reference - use pre-computed lowercase
                    if let Some(&idx) = col_index_map.get(&id.value_lower) {
                        items.push(SelectItem {
                            output_name: id.value.clone(),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else {
                        // Column not found directly - try to match against qualified columns (e.g., "name" matches "e.name")
                        // This handles JOIN cases where columns have table prefixes
                        let suffix = format!(".{}", id.value_lower);
                        let mut found = false;
                        for (col_name, &idx) in &col_index_map {
                            if col_name.ends_with(&suffix) {
                                items.push(SelectItem {
                                    output_name: id.value.clone(),
                                    source: SelectItemSource::BaseColumn(idx),
                                });
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            // Still not found - treat as expression and let evaluator handle it
                            items.push(SelectItem {
                                output_name: id.value.clone(),
                                source: SelectItemSource::Expression(col_expr.clone()),
                            });
                        }
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    // Qualified column reference (table.column)
                    // Try full qualified name first (e.g., "c.name" for JOINs)
                    let full_name =
                        format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    if let Some(&idx) = col_index_map.get(&full_name) {
                        items.push(SelectItem {
                            output_name: format!("{}.{}", qid.qualifier.value, qid.name.value),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else if let Some(&idx) = col_index_map.get(&qid.name.value_lower) {
                        // Fall back to unqualified name
                        items.push(SelectItem {
                            output_name: qid.name.value.clone(),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else {
                        // Column not found - treat as expression and let evaluator handle it
                        items.push(SelectItem {
                            output_name: format!("{}.{}", qid.qualifier.value, qid.name.value),
                            source: SelectItemSource::Expression(col_expr.clone()),
                        });
                    }
                }
                Expression::Star(_) | Expression::QualifiedStar(_) => {
                    // SELECT * or t.* - include all base columns
                    for (idx, col) in base_columns.iter().enumerate() {
                        items.push(SelectItem {
                            output_name: col.clone(),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    }
                }
                Expression::FunctionCall(func) => {
                    // Check if this is an aggregate function that's already in base columns
                    // This happens when window functions are combined with GROUP BY aggregation
                    // OPTIMIZATION: func.function is already uppercase from parsing
                    let func_col_name = if func.arguments.is_empty()
                        || matches!(func.arguments.first(), Some(Expression::Star(_)))
                    {
                        format!("{}(*)", func.function)
                    } else if func.arguments.len() == 1 {
                        if let Expression::Identifier(id) = &func.arguments[0] {
                            format!("{}({})", func.function, id.value)
                        } else {
                            format!("{}(expr)", func.function)
                        }
                    } else {
                        format!("{}(...)", func.function)
                    };

                    // Try to find the function result in base columns
                    if let Some(&idx) = col_index_map.get(&func_col_name.to_lowercase()) {
                        items.push(SelectItem {
                            output_name: func_col_name,
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else {
                        // Fallback to expression evaluation
                        items.push(SelectItem {
                            output_name: format!("expr_{}", items.len()),
                            source: SelectItemSource::Expression(col_expr.clone()),
                        });
                    }
                }
                _ => {
                    // Other expressions - check for embedded window functions
                    if Self::find_window_in_expression(col_expr).is_some() {
                        // Use the column_name from window_functions for consistency with window_value_map key
                        let wf_name = if wf_idx < window_functions.len() {
                            window_functions[wf_idx].column_name.clone()
                        } else {
                            format!("__wf_{}", wf_idx)
                        };
                        items.push(SelectItem {
                            output_name: format!("expr_{}", items.len()),
                            source: SelectItemSource::ExpressionWithWindow(
                                col_expr.clone(),
                                wf_name.to_lowercase(),
                            ),
                        });
                        wf_idx += 1;
                    } else {
                        items.push(SelectItem {
                            output_name: format!("expr_{}", items.len()),
                            source: SelectItemSource::Expression(col_expr.clone()),
                        });
                    }
                }
            }
        }

        items
    }

    /// Parse window functions from SELECT list
    /// This also detects window functions nested inside expressions (like val - LAG(...))
    fn parse_window_functions(
        &self,
        stmt: &SelectStatement,
        _base_columns: &[String],
    ) -> Result<Vec<WindowFunctionInfo>> {
        let mut window_functions = Vec::new();
        let mut col_idx = 0;

        for col_expr in &stmt.columns {
            match col_expr {
                Expression::Window(window_expr) => {
                    let wf_info =
                        self.extract_window_function_info(window_expr, col_idx, &stmt.window_defs)?;
                    window_functions.push(wf_info);
                    col_idx += 1;
                }
                Expression::Aliased(aliased) => {
                    if let Expression::Window(window_expr) = aliased.expression.as_ref() {
                        let mut wf_info = self.extract_window_function_info(
                            window_expr,
                            col_idx,
                            &stmt.window_defs,
                        )?;
                        wf_info.column_name = aliased.alias.value.clone();
                        window_functions.push(wf_info);
                    } else if let Some(window_expr) =
                        Self::find_window_in_expression(aliased.expression.as_ref())
                    {
                        // Expression containing window function (e.g., val - LAG(...) AS diff)
                        // Generate synthetic name: __wf_<col_idx>
                        let mut wf_info = self.extract_window_function_info(
                            window_expr,
                            col_idx,
                            &stmt.window_defs,
                        )?;
                        wf_info.column_name = format!("__wf_{}", col_idx);
                        window_functions.push(wf_info);
                    }
                    col_idx += 1;
                }
                _ => {
                    // Check for window function inside other expressions
                    if let Some(window_expr) = Self::find_window_in_expression(col_expr) {
                        let mut wf_info = self.extract_window_function_info(
                            window_expr,
                            col_idx,
                            &stmt.window_defs,
                        )?;
                        wf_info.column_name = format!("__wf_{}", col_idx);
                        window_functions.push(wf_info);
                    }
                    col_idx += 1;
                }
            }
        }

        Ok(window_functions)
    }

    /// Recursively find a window expression inside an expression
    /// Returns the first Window expression found, if any
    fn find_window_in_expression(expr: &Expression) -> Option<&WindowExpression> {
        match expr {
            Expression::Window(w) => Some(w),
            Expression::Infix(infix) => Self::find_window_in_expression(&infix.left)
                .or_else(|| Self::find_window_in_expression(&infix.right)),
            Expression::Prefix(prefix) => Self::find_window_in_expression(&prefix.right),
            Expression::FunctionCall(f) => {
                for arg in &f.arguments {
                    if let Some(w) = Self::find_window_in_expression(arg) {
                        return Some(w);
                    }
                }
                None
            }
            Expression::Aliased(a) => Self::find_window_in_expression(&a.expression),
            Expression::Case(c) => {
                for clause in &c.when_clauses {
                    if let Some(w) = Self::find_window_in_expression(&clause.condition) {
                        return Some(w);
                    }
                    if let Some(w) = Self::find_window_in_expression(&clause.then_result) {
                        return Some(w);
                    }
                }
                if let Some(else_val) = &c.else_value {
                    if let Some(w) = Self::find_window_in_expression(else_val) {
                        return Some(w);
                    }
                }
                None
            }
            Expression::Cast(cast) => Self::find_window_in_expression(&cast.expr),
            _ => None,
        }
    }

    /// Replace window expressions in an expression tree with identifier references
    /// This transforms `val - LAG(...) OVER (...)` into `val - __wf_N` where __wf_N
    /// contains the pre-computed window function value
    fn replace_window_with_identifier(expr: &Expression, wf_name: &str) -> Expression {
        use crate::parser::token::{Position, Token, TokenType};

        match expr {
            Expression::Window(_) => {
                // Replace window expression with a simple identifier
                let dummy_token =
                    Token::new(TokenType::Identifier, wf_name, Position::new(0, 0, 0));
                Expression::Identifier(Identifier::new(dummy_token, wf_name.to_string()))
            }
            Expression::Infix(infix) => Expression::Infix(InfixExpression {
                token: infix.token.clone(),
                left: Box::new(Self::replace_window_with_identifier(&infix.left, wf_name)),
                operator: infix.operator.clone(),
                op_type: infix.op_type,
                right: Box::new(Self::replace_window_with_identifier(&infix.right, wf_name)),
            }),
            Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression {
                token: prefix.token.clone(),
                operator: prefix.operator.clone(),
                op_type: prefix.op_type,
                right: Box::new(Self::replace_window_with_identifier(&prefix.right, wf_name)),
            }),
            Expression::FunctionCall(f) => Expression::FunctionCall(FunctionCall {
                token: f.token.clone(),
                function: f.function.clone(),
                arguments: f
                    .arguments
                    .iter()
                    .map(|a| Self::replace_window_with_identifier(a, wf_name))
                    .collect(),
                is_distinct: f.is_distinct,
                order_by: f.order_by.clone(),
                filter: f.filter.clone(),
            }),
            Expression::Aliased(a) => Expression::Aliased(AliasedExpression {
                token: a.token.clone(),
                expression: Box::new(Self::replace_window_with_identifier(&a.expression, wf_name)),
                alias: a.alias.clone(),
            }),
            Expression::Cast(cast) => Expression::Cast(CastExpression {
                token: cast.token.clone(),
                expr: Box::new(Self::replace_window_with_identifier(&cast.expr, wf_name)),
                type_name: cast.type_name.clone(),
            }),
            Expression::Case(case) => {
                // Process window functions inside CASE expressions
                let new_value = case
                    .value
                    .as_ref()
                    .map(|v| Box::new(Self::replace_window_with_identifier(v, wf_name)));
                let new_whens: Vec<WhenClause> = case
                    .when_clauses
                    .iter()
                    .map(|w| WhenClause {
                        token: w.token.clone(),
                        condition: Self::replace_window_with_identifier(&w.condition, wf_name),
                        then_result: Self::replace_window_with_identifier(&w.then_result, wf_name),
                    })
                    .collect();
                let new_else = case
                    .else_value
                    .as_ref()
                    .map(|e| Box::new(Self::replace_window_with_identifier(e, wf_name)));
                Expression::Case(CaseExpression {
                    token: case.token.clone(),
                    value: new_value,
                    when_clauses: new_whens,
                    else_value: new_else,
                })
            }
            _ => expr.clone(),
        }
    }

    /// Extract window function info from WindowExpression
    fn extract_window_function_info(
        &self,
        window_expr: &WindowExpression,
        col_idx: usize,
        window_defs: &[WindowDefinition],
    ) -> Result<WindowFunctionInfo> {
        let func = &window_expr.function;

        // Resolve named window reference if present
        let (partition_by_exprs, order_by, frame) =
            if let Some(ref win_ref) = window_expr.window_ref {
                // Look up the named window definition
                let win_def = window_defs
                    .iter()
                    .find(|wd| wd.name.eq_ignore_ascii_case(win_ref))
                    .ok_or_else(|| {
                        Error::NotSupportedMessage(format!("Unknown window name: {}", win_ref))
                    })?;
                (
                    win_def.partition_by.clone(),
                    win_def.order_by.clone(),
                    win_def.frame.clone(),
                )
            } else {
                (
                    window_expr.partition_by.clone(),
                    window_expr.order_by.clone(),
                    window_expr.frame.clone(),
                )
            };

        // Extract partition by column names
        // For qualified identifiers (e.g., l.grp), keep the full qualified name
        // to properly match columns from JOINs
        let partition_by: Vec<String> = partition_by_exprs
            .iter()
            .filter_map(|e| match e {
                Expression::Identifier(id) => Some(id.value.clone()),
                Expression::QualifiedIdentifier(qid) => {
                    // Keep the full qualified name for JOIN cases
                    Some(format!("{}.{}", qid.qualifier.value, qid.name.value))
                }
                _ => None,
            })
            .collect();

        // Build column name
        let column_name = format!("{}()", func.function);

        // OPTIMIZATION: func.function is already uppercase from parsing
        Ok(WindowFunctionInfo {
            name: func.function.clone(),
            arguments: func.arguments.clone(),
            partition_by,
            order_by,
            frame,
            column_name,
            column_index: col_idx,
            is_distinct: func.is_distinct,
        })
    }

    /// Compute window function values for all rows
    /// pre_sorted: Optional info about whether rows are already sorted by an indexed column
    /// pre_grouped: Optional pre-grouped partitions from index (avoids hash-based grouping)
    #[allow(clippy::too_many_arguments)]
    fn compute_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
        pre_grouped: Option<&WindowPreGroupedState>,
    ) -> Result<Vec<Value>> {
        // Check if this is an aggregate function used as window function
        let is_aggregate = self.function_registry.is_aggregate(&wf_info.name);

        if is_aggregate {
            // Handle aggregate functions as window functions (SUM, COUNT, AVG, etc.)
            return self.compute_aggregate_window_function(
                wf_info,
                rows,
                columns,
                col_index_map,
                ctx,
                pre_sorted,
            );
        }

        // Get the window function from registry
        let window_func = self
            .function_registry
            .get_window(&wf_info.name)
            .ok_or_else(|| {
                Error::NotSupportedMessage(format!("Unknown window function: {}", wf_info.name))
            })?;

        // If there's no partitioning, treat all rows as one partition
        if wf_info.partition_by.is_empty() {
            // OPTIMIZATION: Check if rows are already sorted by the window ORDER BY column
            // This allows us to skip sorting when we've pre-sorted using an index
            let rows_already_sorted = self.check_rows_presorted(wf_info, pre_sorted);

            // Precompute ORDER BY values once (needed for tie detection even if pre-sorted)
            let precomputed_order_by = if !wf_info.order_by.is_empty() {
                Some(self.precompute_order_by_values(
                    &wf_info.order_by,
                    rows,
                    columns,
                    col_index_map,
                    ctx,
                ))
            } else {
                None
            };

            let row_indices: Vec<usize> = (0..rows.len()).collect();
            let (partition_results, sorted_indices) = self.compute_window_for_partition(
                &*window_func,
                wf_info,
                rows,
                row_indices,
                precomputed_order_by.as_ref(),
                columns,
                col_index_map,
                ctx,
                rows_already_sorted, // Skip sorting if rows are pre-sorted by index
            )?;

            // Map results back to original row order
            // partition_results are in ORDER BY sorted order, we need to map back
            if wf_info.order_by.is_empty() {
                // No ORDER BY means row_indices weren't sorted, results are in original order
                return Ok(partition_results);
            }

            // Map partition_results (in sorted order) back to original row order
            let mut results = vec![Value::null_unknown(); rows.len()];
            for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
                results[orig_idx] = partition_results[sorted_pos].clone();
            }
            return Ok(results);
        }

        // OPTIMIZATION: Precompute ORDER BY values ONCE for all rows (not per-partition!)
        // This is critical - precompute_order_by_values was being called for ALL rows
        // inside each partition, causing O(n × p) work instead of O(n).
        let precomputed_order_by = if !wf_info.order_by.is_empty() {
            Some(self.precompute_order_by_values(
                &wf_info.order_by,
                rows,
                columns,
                col_index_map,
                ctx,
            ))
        } else {
            None
        };

        // Group rows by partition key
        // OPTIMIZATION: Use pre-grouped partitions from index if available (avoids O(n) hashing)
        // OPTIMIZATION: Use SmallVec for partition keys to avoid heap allocation
        // for common cases (up to 4 partition columns)
        // OPTIMIZATION: Use FxHashMap for fastest hash table operations with trusted keys
        let partitions: FxHashMap<PartitionKey, Vec<usize>> = if let Some(pg) = pre_grouped {
            // Use pre-grouped partitions from index - no hashing needed!
            pg.partition_map.clone()
        } else {
            // Build partition map by hashing (default path)
            let mut partitions: FxHashMap<PartitionKey, Vec<usize>> = FxHashMap::default();

            // OPTIMIZATION: Pre-compute partition column indices to avoid to_lowercase() per row
            // Try both qualified (e.g., "l.grp") and unqualified (e.g., "grp") names
            let partition_indices: SmallVec<[Option<usize>; 4]> = wf_info
                .partition_by
                .iter()
                .map(|part_col| {
                    let lower = part_col.to_lowercase();
                    col_index_map.get(&lower).copied().or_else(|| {
                        // If qualified name not found, try unqualified (last part after dot)
                        if let Some(dot_pos) = lower.rfind('.') {
                            col_index_map.get(&lower[dot_pos + 1..]).copied()
                        } else {
                            None
                        }
                    })
                })
                .collect();

            for (i, row) in rows.iter().enumerate() {
                let mut key: PartitionKey = SmallVec::with_capacity(partition_indices.len());
                for idx_opt in &partition_indices {
                    let value = if let Some(&idx) = idx_opt.as_ref() {
                        row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
                    } else {
                        Value::null_unknown()
                    };
                    key.push(value);
                }
                partitions.entry(key).or_default().push(i);
            }
            partitions
        };

        // Compute window function for each partition
        // Use parallel execution for large number of partitions
        let partition_count = partitions.len();
        let use_parallel = partition_count >= 10 && rows.len() >= 1000;

        if use_parallel {
            // Parallel execution: process partitions concurrently
            let results_values: std::sync::RwLock<Vec<Value>> =
                std::sync::RwLock::new(vec![Value::null_unknown(); rows.len()]);

            let partitions_vec: Vec<_> = partitions.into_iter().collect();
            partitions_vec
                .par_iter()
                .try_for_each(|(_key, row_indices)| -> Result<()> {
                    let (partition_results, sorted_indices) = self.compute_window_for_partition(
                        &*window_func,
                        wf_info,
                        rows,
                        row_indices.clone(),
                        precomputed_order_by.as_ref(),
                        columns,
                        col_index_map,
                        ctx,
                        false, // Partitioned case: sorting is needed per-partition
                    )?;

                    // Map results from sorted order back to original row indices
                    let mut results_guard = results_values.write().unwrap();
                    for (i, &orig_idx) in sorted_indices.iter().enumerate() {
                        results_guard[orig_idx] = partition_results[i].clone();
                    }
                    Ok(())
                })?;

            Ok(results_values.into_inner().unwrap())
        } else {
            // Sequential execution for small partition counts
            let mut results = vec![Value::null_unknown(); rows.len()];

            for (_key, row_indices) in partitions {
                let (partition_results, sorted_indices) = self.compute_window_for_partition(
                    &*window_func,
                    wf_info,
                    rows,
                    row_indices,
                    precomputed_order_by.as_ref(),
                    columns,
                    col_index_map,
                    ctx,
                    false, // Partitioned case: sorting is needed per-partition
                )?;

                // Map results from sorted order back to original row indices
                for (i, &orig_idx) in sorted_indices.iter().enumerate() {
                    results[orig_idx] = partition_results[i].clone();
                }
            }

            Ok(results)
        }
    }

    /// Compute window function for a single partition
    /// Returns (results in sorted order, sorted row indices) to avoid re-sorting in the caller
    /// precomputed_order_by: Optional precomputed ORDER BY values for ALL rows (avoids recomputation)
    /// skip_sorting: If true, skip sorting (rows are already pre-sorted by index)
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[Row],
        mut row_indices: Vec<usize>,
        precomputed_order_by: Option<&Vec<Vec<(Value, bool)>>>,
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        ctx: &ExecutionContext,
        skip_sorting: bool,
    ) -> Result<(Vec<Value>, Vec<usize>)> {
        // Suppress unused variable warnings - these are needed for compute_lead_lag, compute_ntile, etc.
        let _ = columns;

        // Use precomputed ORDER BY values if available (avoids O(n × p) recomputation)
        // The precomputed values are indexed by row index in the original rows array
        let order_by_values: &[Vec<(Value, bool)>] =
            precomputed_order_by.map(|v| v.as_slice()).unwrap_or(&[]);

        // Sort partition by ORDER BY if specified (skip if already pre-sorted by index)
        if !skip_sorting && !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        // OPTIMIZATION: Pre-compute column index for function argument
        let arg_col_idx: Option<usize> = if !wf_info.arguments.is_empty() {
            self.extract_column_from_arg(&wf_info.arguments[0])
                .and_then(|col_name| col_index_map.get(&col_name.to_lowercase()).copied())
        } else {
            None
        };

        // Build partition values for the window function
        let partition_values: Vec<Value> = row_indices
            .iter()
            .map(|&idx| {
                // For functions like LAG/LEAD, we need the column value
                // For ROW_NUMBER/RANK, we don't need it
                if let Some(col_idx) = arg_col_idx {
                    return all_rows[idx]
                        .get(col_idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                }
                Value::null_unknown()
            })
            .collect();

        // Build order by values for ranking functions
        // Use the precomputed order_by_values which handles complex expressions like COALESCE(SUM(val), 0)
        let order_values: Vec<Value> = if !order_by_values.is_empty() {
            // Extract the first ORDER BY value for each row in the partition (in sorted order)
            row_indices
                .iter()
                .map(|&idx| {
                    order_by_values
                        .get(idx)
                        .and_then(|vals| vals.first())
                        .map(|(v, _)| v.clone())
                        .unwrap_or_else(Value::null_unknown)
                })
                .collect()
        } else {
            vec![]
        };

        // Compute window function for each row in the partition
        let mut results = Vec::with_capacity(row_indices.len());
        let partition_len = row_indices.len();

        for (i, &row_idx) in row_indices.iter().enumerate() {
            // Handle special functions
            let value = match wf_info.name.as_str() {
                "LEAD" | "LAG" => self.compute_lead_lag(
                    wf_info,
                    &partition_values,
                    i,
                    &all_rows[row_idx],
                    columns,
                    ctx,
                )?,
                "NTILE" => self.compute_ntile(wf_info, row_indices.len(), i, ctx)?,
                "RANK" | "DENSE_RANK" => self.compute_rank(wf_info, &order_values, i)?,
                "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE" => {
                    // Compute frame bounds for navigation functions
                    let (frame_start, frame_end) =
                        self.compute_simple_frame_bounds(wf_info, i, partition_len);

                    // Get values within the frame
                    let frame_values: Vec<Value> =
                        if frame_start < frame_end && frame_end <= partition_len {
                            partition_values[frame_start..frame_end].to_vec()
                        } else {
                            vec![]
                        };

                    match wf_info.name.as_str() {
                        "FIRST_VALUE" => self.compute_first_value(&frame_values)?,
                        "LAST_VALUE" => self.compute_last_value(&frame_values)?,
                        "NTH_VALUE" => self.compute_nth_value(wf_info, &frame_values, ctx)?,
                        _ => unreachable!(),
                    }
                }
                "PERCENT_RANK" => self.compute_percent_rank(&order_values, i, row_indices.len())?,
                "CUME_DIST" => self.compute_cume_dist(&order_values, i, row_indices.len())?,
                _ => {
                    // ROW_NUMBER and other simple functions
                    window_func.process(&partition_values, &order_values, i)?
                }
            };
            results.push(value);
        }

        Ok((results, row_indices))
    }

    /// Compute LEAD or LAG function
    fn compute_lead_lag(
        &self,
        wf_info: &WindowFunctionInfo,
        partition_values: &[Value],
        current_row: usize,
        current_row_data: &Row,
        columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        // Get offset (default 1)
        let offset = if wf_info.arguments.len() > 1 {
            let mut eval = ExpressionEval::compile(&wf_info.arguments[1], &[])?.with_context(ctx);
            match eval.eval_slice(&[])? {
                Value::Integer(n) => n as usize,
                _ => 1,
            }
        } else {
            1
        };

        // Get default value - use current row context for column references
        let default_value = if wf_info.arguments.len() > 2 {
            let mut eval =
                ExpressionEval::compile(&wf_info.arguments[2], columns)?.with_context(ctx);
            eval.eval(current_row_data)?
        } else {
            Value::null_unknown()
        };

        // Calculate target index
        let target_idx = if wf_info.name == "LEAD" {
            current_row.checked_add(offset)
        } else {
            // LAG
            current_row.checked_sub(offset)
        };

        match target_idx {
            Some(idx) if idx < partition_values.len() => Ok(partition_values[idx].clone()),
            _ => Ok(default_value),
        }
    }

    /// Compute NTILE function
    fn compute_ntile(
        &self,
        wf_info: &WindowFunctionInfo,
        partition_size: usize,
        current_row: usize,
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        // Get n (number of buckets)
        let n = if !wf_info.arguments.is_empty() {
            let mut eval = ExpressionEval::compile(&wf_info.arguments[0], &[])?.with_context(ctx);
            match eval.eval_slice(&[])? {
                Value::Integer(n) if n > 0 => n as usize,
                _ => 1,
            }
        } else {
            1
        };

        // NTILE divides rows into n buckets as evenly as possible.
        // If partition_size doesn't divide evenly by n, the first (partition_size % n)
        // buckets get one extra row.
        //
        // For example, NTILE(3) with 7 rows:
        // - base_size = 7 / 3 = 2 (rows per bucket)
        // - remainder = 7 % 3 = 1 (1 bucket gets an extra row)
        // - Bucket 1: rows 0, 1, 2 (3 rows - gets extra)
        // - Bucket 2: rows 3, 4 (2 rows)
        // - Bucket 3: rows 5, 6 (2 rows)

        if n >= partition_size {
            // More buckets than rows - each row gets its own bucket
            return Ok(Value::Integer((current_row + 1).min(n) as i64));
        }

        let base_size = partition_size / n;
        let remainder = partition_size % n;

        // Calculate which bucket this row belongs to
        // First 'remainder' buckets have (base_size + 1) rows
        // Remaining buckets have base_size rows
        let bucket = if current_row < remainder * (base_size + 1) {
            // Row is in one of the larger buckets
            current_row / (base_size + 1) + 1
        } else {
            // Row is in one of the smaller buckets
            let rows_in_larger_buckets = remainder * (base_size + 1);
            let row_in_smaller_section = current_row - rows_in_larger_buckets;
            remainder + row_in_smaller_section / base_size + 1
        };

        Ok(Value::Integer(bucket as i64))
    }

    /// Compute RANK or DENSE_RANK function
    fn compute_rank(
        &self,
        wf_info: &WindowFunctionInfo,
        order_values: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        if order_values.is_empty() || current_row == 0 {
            return Ok(Value::Integer(1));
        }

        let current_value = &order_values[current_row];

        // Check if current value equals previous - same rank as previous row
        if order_values[current_row - 1] == *current_value {
            // Find the first row with this value and compute its rank
            let mut first_same = current_row;
            while first_same > 0 && order_values[first_same - 1] == *current_value {
                first_same -= 1;
            }

            if first_same == 0 {
                return Ok(Value::Integer(1));
            }

            // For RANK: rank = first_same + 1 (position-based)
            // For DENSE_RANK: count distinct values up to and including first_same
            if wf_info.name == "RANK" {
                return Ok(Value::Integer((first_same + 1) as i64));
            } else {
                // DENSE_RANK: count distinct values up to and including first_same
                let mut distinct_count = 1;
                for i in 1..=first_same {
                    if order_values[i] != order_values[i - 1] {
                        distinct_count += 1;
                    }
                }
                return Ok(Value::Integer(distinct_count));
            }
        }

        // Current value differs from previous
        if wf_info.name == "RANK" {
            // RANK: position-based (1-indexed)
            Ok(Value::Integer((current_row + 1) as i64))
        } else {
            // DENSE_RANK: count distinct values up to and including current
            let mut distinct_count = 1;
            for i in 1..=current_row {
                if order_values[i] != order_values[i - 1] {
                    distinct_count += 1;
                }
            }
            Ok(Value::Integer(distinct_count))
        }
    }

    /// Compute FIRST_VALUE function
    fn compute_first_value(&self, partition_values: &[Value]) -> Result<Value> {
        if partition_values.is_empty() {
            return Ok(Value::null_unknown());
        }
        Ok(partition_values[0].clone())
    }

    /// Compute LAST_VALUE function
    fn compute_last_value(&self, partition_values: &[Value]) -> Result<Value> {
        if partition_values.is_empty() {
            return Ok(Value::null_unknown());
        }
        Ok(partition_values[partition_values.len() - 1].clone())
    }

    /// Compute NTH_VALUE function
    fn compute_nth_value(
        &self,
        wf_info: &WindowFunctionInfo,
        partition_values: &[Value],
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        if partition_values.is_empty() {
            return Ok(Value::null_unknown());
        }

        // Get n (1-indexed position) from second argument
        let n = if wf_info.arguments.len() > 1 {
            let mut eval = ExpressionEval::compile(&wf_info.arguments[1], &[])?.with_context(ctx);
            match eval.eval_slice(&[])? {
                Value::Integer(n) if n > 0 => n as usize,
                _ => return Ok(Value::null_unknown()),
            }
        } else {
            return Ok(Value::null_unknown());
        };

        // n is 1-indexed, so subtract 1 for array access
        let index = n - 1;
        if index >= partition_values.len() {
            return Ok(Value::null_unknown());
        }

        Ok(partition_values[index].clone())
    }

    /// Compute simple ROWS-based frame bounds for navigation functions
    /// Returns (start, end) where end is exclusive
    fn compute_simple_frame_bounds(
        &self,
        wf_info: &WindowFunctionInfo,
        current_row: usize,
        partition_len: usize,
    ) -> (usize, usize) {
        if let Some(ref frame) = wf_info.frame {
            // Only handle ROWS frames for now (not RANGE)
            // Calculate start bound
            let start = match &frame.start {
                WindowFrameBound::UnboundedPreceding => 0,
                WindowFrameBound::CurrentRow => current_row,
                WindowFrameBound::Preceding(expr) => {
                    if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                        current_row.saturating_sub(lit.value as usize)
                    } else {
                        0
                    }
                }
                WindowFrameBound::Following(expr) => {
                    if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                        (current_row + lit.value as usize).min(partition_len)
                    } else {
                        current_row
                    }
                }
                WindowFrameBound::UnboundedFollowing => partition_len,
            };

            // Calculate end bound (exclusive)
            let end = match &frame.end {
                Some(WindowFrameBound::UnboundedFollowing) => partition_len,
                Some(WindowFrameBound::CurrentRow) => current_row + 1,
                Some(WindowFrameBound::Following(expr)) => {
                    if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                        (current_row + lit.value as usize + 1).min(partition_len)
                    } else {
                        partition_len
                    }
                }
                Some(WindowFrameBound::Preceding(expr)) => {
                    if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                        if lit.value as usize <= current_row {
                            current_row - lit.value as usize + 1
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                }
                Some(WindowFrameBound::UnboundedPreceding) => 0,
                None => {
                    // If no end is specified, default to CURRENT ROW
                    current_row + 1
                }
            };

            (start, end)
        } else {
            // No explicit frame specified
            // SQL standard:
            // - With ORDER BY: default is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            // - Without ORDER BY: default is the entire partition
            if wf_info.order_by.is_empty() {
                // No ORDER BY - entire partition
                (0, partition_len)
            } else {
                // Has ORDER BY - default frame is UNBOUNDED PRECEDING to CURRENT ROW
                (0, current_row + 1)
            }
        }
    }

    /// Compute PERCENT_RANK function
    /// PERCENT_RANK = (rank - 1) / (total_rows - 1)
    /// The partition is already sorted by ORDER BY, so we calculate rank based on position
    fn compute_percent_rank(
        &self,
        order_values: &[Value],
        current_row: usize,
        partition_size: usize,
    ) -> Result<Value> {
        // Use partition_size for the denominator when order_values is empty
        let n = if order_values.is_empty() {
            partition_size
        } else {
            order_values.len()
        };

        if n <= 1 {
            return Ok(Value::Float(0.0));
        }

        // Calculate rank for current row
        // Since partition is already sorted, rank is position + 1, with ties handled
        let rank = if order_values.is_empty() || current_row == 0 {
            1
        } else {
            let current_value = &order_values[current_row];
            // Find the first row with the same value (this determines the rank for ties)
            let mut first_same = current_row;
            while first_same > 0 && order_values[first_same - 1] == *current_value {
                first_same -= 1;
            }
            first_same + 1
        };

        let percent_rank = (rank - 1) as f64 / (n - 1) as f64;
        Ok(Value::Float(percent_rank))
    }

    /// Compute CUME_DIST function
    /// CUME_DIST = number of rows with value <= current / total rows
    /// The partition is already sorted by ORDER BY
    fn compute_cume_dist(
        &self,
        order_values: &[Value],
        current_row: usize,
        partition_size: usize,
    ) -> Result<Value> {
        // Use partition_size for the denominator when order_values is empty
        let n = if order_values.is_empty() {
            partition_size
        } else {
            order_values.len()
        };

        if n == 0 {
            return Ok(Value::Float(1.0));
        }

        if order_values.is_empty() {
            // No ordering - all rows are equal, so CUME_DIST is 1.0 for all
            return Ok(Value::Float(1.0));
        }

        let current_value = &order_values[current_row];

        // Find the last row with value <= current value (including ties)
        // Since sorted, find the last position with the same value
        let mut last_same = current_row;
        while last_same + 1 < order_values.len() && order_values[last_same + 1] == *current_value {
            last_same += 1;
        }

        // CUME_DIST = (position of last row with same value + 1) / total rows
        let cume_dist = (last_same + 1) as f64 / n as f64;
        Ok(Value::Float(cume_dist))
    }

    /// Extract column name from expression
    fn extract_column_from_expr(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value.clone()),
            Expression::QualifiedIdentifier(qid) => {
                // Return fully qualified name "qualifier.column" for JOIN results
                Some(format!("{}.{}", qid.qualifier.value, qid.name.value))
            }
            _ => None,
        }
    }

    /// Extract column name from function argument
    fn extract_column_from_arg(&self, arg: &Expression) -> Option<String> {
        self.extract_column_from_expr(arg)
    }

    /// Resolve column index from expression, trying both qualified and unqualified names
    fn resolve_column_index(
        &self,
        expr: &Expression,
        col_index_map: &FxHashMap<String, usize>,
    ) -> Option<usize> {
        match expr {
            Expression::Identifier(id) => col_index_map.get(&id.value_lower).copied(),
            Expression::QualifiedIdentifier(qid) => {
                // Try fully qualified name first (e.g., "s.qty")
                let qualified =
                    format!("{}.{}", qid.qualifier.value, qid.name.value).to_lowercase();
                col_index_map
                    .get(&qualified)
                    // Then try unqualified (e.g., "qty") for CTE/subquery cases
                    .or_else(|| col_index_map.get(&qid.name.value_lower))
                    .copied()
            }
            Expression::FunctionCall(func) => {
                // Handle aggregate functions that have been computed in GROUP BY
                // e.g., SUM(val) -> look for column named "SUM(val)"
                let func_col_name = if func.arguments.is_empty()
                    || matches!(func.arguments.first(), Some(Expression::Star(_)))
                {
                    format!("{}(*)", func.function)
                } else if func.arguments.len() == 1 {
                    match &func.arguments[0] {
                        Expression::Identifier(id) => {
                            format!("{}({})", func.function, id.value)
                        }
                        Expression::QualifiedIdentifier(qid) => {
                            format!("{}({})", func.function, qid.name.value)
                        }
                        _ => format!("{}(expr)", func.function),
                    }
                } else {
                    format!("{}(...)", func.function)
                };

                // First try exact match (e.g., "sum(val)")
                if let Some(idx) = col_index_map.get(&func_col_name.to_lowercase()).copied() {
                    return Some(idx);
                }

                // If not found and this is an aggregate function, the result might be aliased
                // In that case, we need the caller to use expression evaluation instead
                // Return None to trigger the expression evaluation path
                None
            }
            _ => None,
        }
    }

    /// Precompute ORDER BY values for all rows
    /// Returns a vec where each element is a vec of (value, ascending) pairs for one row
    fn precompute_order_by_values(
        &self,
        order_by: &[OrderByExpression],
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        ctx: &ExecutionContext,
    ) -> Vec<Vec<(Value, bool)>> {
        // Check if any ORDER BY expression is complex (not a simple column reference)
        let has_complex_expr = order_by.iter().any(|ob| {
            !matches!(
                &ob.expression,
                Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
            )
        });

        if has_complex_expr {
            // Build aliases from col_index_map (e.g., "sum(val)" -> column_index)
            // This allows ORDER BY SUM(val) to resolve to the correct column
            let agg_aliases: Vec<(String, usize)> =
                col_index_map.iter().map(|(k, v)| (k.clone(), *v)).collect();

            // Extract order_by expressions and ascending flags
            let order_exprs: Vec<Expression> =
                order_by.iter().map(|ob| ob.expression.clone()).collect();
            let ascending_flags: Vec<bool> = order_by.iter().map(|ob| ob.ascending).collect();

            // Compile all expressions with aliases
            match MultiExpressionEval::compile_with_aliases(&order_exprs, columns, &agg_aliases) {
                Ok(eval) => {
                    let mut eval = eval.with_context(ctx);
                    rows.iter()
                        .map(|row| match eval.eval_all(row) {
                            Ok(values) => values
                                .into_iter()
                                .zip(ascending_flags.iter())
                                .map(|(value, &asc)| (value, asc))
                                .collect(),
                            Err(_) => ascending_flags
                                .iter()
                                .map(|&asc| (Value::null_unknown(), asc))
                                .collect(),
                        })
                        .collect()
                }
                Err(_) => {
                    // Compilation failed - return all nulls
                    rows.iter()
                        .map(|_| {
                            ascending_flags
                                .iter()
                                .map(|&asc| (Value::null_unknown(), asc))
                                .collect()
                        })
                        .collect()
                }
            }
        } else {
            // Fast path: simple column references
            let order_by_indices: Vec<(Option<usize>, bool)> = order_by
                .iter()
                .map(|ob| {
                    let idx = match &ob.expression {
                        Expression::Identifier(id) => col_index_map.get(&id.value_lower).copied(),
                        Expression::QualifiedIdentifier(qid) => {
                            let qualified = format!("{}.{}", qid.qualifier.value, qid.name.value)
                                .to_lowercase();
                            col_index_map
                                .get(&qualified)
                                .or_else(|| col_index_map.get(&qid.name.value_lower))
                                .copied()
                        }
                        _ => None,
                    };
                    (idx, ob.ascending)
                })
                .collect();

            rows.iter()
                .map(|row| {
                    order_by_indices
                        .iter()
                        .map(|(idx_opt, ascending)| {
                            let value = idx_opt
                                .and_then(|idx| row.get(idx).cloned())
                                .unwrap_or_else(Value::null_unknown);
                            (value, *ascending)
                        })
                        .collect()
                })
                .collect()
        }
    }

    /// Check if rows are already pre-sorted by the window ORDER BY column
    /// Returns true if we can skip sorting
    fn check_rows_presorted(
        &self,
        wf_info: &WindowFunctionInfo,
        pre_sorted: Option<&WindowPreSortedState>,
    ) -> bool {
        let pre_sorted = match pre_sorted {
            Some(ps) => ps,
            None => return false,
        };

        // Only optimize if there's exactly one ORDER BY column (simple case)
        if wf_info.order_by.len() != 1 {
            return false;
        }

        let order_by = &wf_info.order_by[0];

        // Extract column name from ORDER BY expression
        let order_col = match &order_by.expression {
            Expression::Identifier(id) => id.value_lower.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
            _ => return false, // Complex expressions can't be pre-sorted
        };

        // Check if pre-sorted column matches and direction matches
        order_col == pre_sorted.column && order_by.ascending == pre_sorted.ascending
    }

    /// Sort row indices using precomputed ORDER BY values
    fn sort_by_order_values(row_indices: &mut [usize], order_by_values: &[Vec<(Value, bool)>]) {
        row_indices.sort_by(|&a, &b| {
            for (idx, ((a_val, ascending), (b_val, _))) in order_by_values[a]
                .iter()
                .zip(order_by_values[b].iter())
                .enumerate()
            {
                let _ = idx; // Suppress warning
                let cmp = a_val.partial_cmp(b_val).unwrap_or(Ordering::Equal);
                let cmp = if !ascending { cmp.reverse() } else { cmp };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        });
    }

    /// Compute aggregate function as window function (SUM, COUNT, AVG, MIN, MAX)
    fn compute_aggregate_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
    ) -> Result<Vec<Value>> {
        // Check if this is COUNT(*) - Star expression means count all rows
        let is_count_star =
            !wf_info.arguments.is_empty() && matches!(wf_info.arguments[0], Expression::Star(_));

        // Get the column index for the aggregate argument
        let arg_col_idx: Option<usize> = if !wf_info.arguments.is_empty() && !is_count_star {
            self.resolve_column_index(&wf_info.arguments[0], col_index_map)
        } else {
            // COUNT(*) has no arguments or Star expression
            None
        };

        // Check if the argument is an expression that needs evaluation
        // This handles cases like SUM(val * 2) or SUM(SUM(val)) in grouped results
        // But NOT for COUNT(*) which should just count rows
        let has_expression_arg =
            !wf_info.arguments.is_empty() && !is_count_star && arg_col_idx.is_none();

        // Pre-compute expression values for all rows if needed
        let expression_values: Vec<Value> = if has_expression_arg {
            let mut eval =
                ExpressionEval::compile(&wf_info.arguments[0], columns)?.with_context(ctx);
            rows.iter()
                .map(|row| eval.eval(row).unwrap_or_else(|_| Value::null_unknown()))
                .collect()
        } else {
            vec![]
        };

        // Pre-compute partition column indices
        // OPTIMIZATION: Use SmallVec to avoid heap allocation for common cases
        // Try both qualified (e.g., "l.grp") and unqualified (e.g., "grp") names
        let partition_indices: SmallVec<[Option<usize>; 4]> = wf_info
            .partition_by
            .iter()
            .map(|part_col| {
                let lower = part_col.to_lowercase();
                col_index_map.get(&lower).copied().or_else(|| {
                    // If qualified name not found, try unqualified (last part after dot)
                    if let Some(dot_pos) = lower.rfind('.') {
                        col_index_map.get(&lower[dot_pos + 1..]).copied()
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Precompute ORDER BY values (supports complex expressions like -score)
        let order_by_values = if !wf_info.order_by.is_empty() {
            self.precompute_order_by_values(&wf_info.order_by, rows, columns, col_index_map, ctx)
        } else {
            vec![]
        };

        // Group rows by partition key
        // OPTIMIZATION: Use SmallVec for partition keys to avoid heap allocation
        // OPTIMIZATION: Use FxHashMap for fastest hash table operations with trusted keys
        let mut partitions: FxHashMap<PartitionKey, Vec<usize>> = FxHashMap::default();

        for (i, row) in rows.iter().enumerate() {
            let mut key: PartitionKey = SmallVec::with_capacity(partition_indices.len());
            for idx_opt in &partition_indices {
                let value = if let Some(&idx) = idx_opt.as_ref() {
                    row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
                } else {
                    Value::null_unknown()
                };
                key.push(value);
            }
            partitions.entry(key).or_default().push(i);
        }

        // Check if we can skip sorting (index optimization)
        // Only applies when there's no PARTITION BY (single partition)
        let skip_sorting =
            wf_info.partition_by.is_empty() && self.check_rows_presorted(wf_info, pre_sorted);

        // Compute aggregate for each partition
        let mut results = vec![Value::null_unknown(); rows.len()];

        for (_key, mut row_indices) in partitions {
            // Sort partition by ORDER BY if specified (skip if pre-sorted)
            if !wf_info.order_by.is_empty() {
                // Only sort if not already pre-sorted by index
                if !skip_sorting {
                    Self::sort_by_order_values(&mut row_indices, &order_by_values);
                }

                // With ORDER BY, compute aggregate with frame specification
                // Default frame is UNBOUNDED PRECEDING to CURRENT ROW if no explicit frame
                let partition_len = row_indices.len();

                // OPTIMIZATION: Precompute peer group boundaries in O(n) instead of O(n²)
                // For RANGE frames, rows with the same ORDER BY value are "peers"
                // After sorting, peers are adjacent, so we can find boundaries in one pass
                // peer_groups[i] = (start_idx, end_idx) where end_idx is exclusive
                let peer_groups: Vec<(usize, usize)> = {
                    let mut groups = Vec::with_capacity(partition_len);
                    if partition_len == 0 {
                        groups
                    } else {
                        let mut group_start = 0;
                        // Use precomputed ORDER BY values (supports complex expressions)
                        let mut prev_values: Vec<Value> = order_by_values[row_indices[0]]
                            .iter()
                            .map(|(v, _)| v.clone())
                            .collect();

                        for i in 1..partition_len {
                            let current_values: Vec<Value> = order_by_values[row_indices[i]]
                                .iter()
                                .map(|(v, _)| v.clone())
                                .collect();

                            if current_values != prev_values {
                                // New peer group starts - fill in previous group for all its members
                                for _ in group_start..i {
                                    groups.push((group_start, i));
                                }
                                group_start = i;
                                prev_values = current_values;
                            }
                        }
                        // Fill in the last group
                        for _ in group_start..partition_len {
                            groups.push((group_start, partition_len));
                        }
                        groups
                    }
                };

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    let mut agg_func = self
                        .function_registry
                        .get_aggregate(&wf_info.name)
                        .ok_or_else(|| {
                            Error::NotSupportedMessage(format!(
                                "Unknown aggregate function: {}",
                                wf_info.name
                            ))
                        })?;

                    // Compute frame bounds based on frame specification
                    let (frame_start, frame_end) = if let Some(ref frame) = wf_info.frame {
                        let is_range = matches!(frame.unit, WindowFrameUnit::Range);

                        // For RANGE frames with numeric offsets, we need value-based comparison
                        // Get the current row's ORDER BY value for RANGE calculations
                        let current_order_value = if is_range && !order_by_values.is_empty() {
                            order_by_values[row_idx].first().map(|(v, _)| v.clone())
                        } else {
                            None
                        };

                        // Helper to convert value to f64 for range comparisons
                        let value_to_f64 = |v: &Value| -> Option<f64> {
                            match v {
                                Value::Integer(i) => Some(*i as f64),
                                Value::Float(f) => Some(*f),
                                _ => None,
                            }
                        };

                        // Calculate start bound
                        let start = match &frame.start {
                            WindowFrameBound::UnboundedPreceding => 0,
                            WindowFrameBound::CurrentRow => {
                                if is_range {
                                    // For RANGE, start of peer group (O(1) lookup)
                                    peer_groups[i].0
                                } else {
                                    i
                                }
                            }
                            WindowFrameBound::Preceding(expr) => {
                                if is_range {
                                    // RANGE PRECEDING: find first row where value >= current - offset
                                    if let (Some(curr_val), Expression::IntegerLiteral(lit)) =
                                        (&current_order_value, expr.as_ref())
                                    {
                                        if let Some(curr_f64) = value_to_f64(curr_val) {
                                            let lower_bound = curr_f64 - lit.value as f64;
                                            // Linear scan from start to find first row in range
                                            let mut start_idx = 0;
                                            for (j, &idx) in row_indices.iter().enumerate() {
                                                if let Some(row_val) = order_by_values[idx]
                                                    .first()
                                                    .and_then(|(v, _)| value_to_f64(v))
                                                {
                                                    if row_val >= lower_bound {
                                                        start_idx = j;
                                                        break;
                                                    }
                                                }
                                            }
                                            start_idx
                                        } else {
                                            0
                                        }
                                    } else {
                                        0
                                    }
                                } else {
                                    // ROWS PRECEDING: simple row offset
                                    if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                                        i.saturating_sub(lit.value as usize)
                                    } else {
                                        0
                                    }
                                }
                            }
                            WindowFrameBound::Following(expr) => {
                                if is_range {
                                    // RANGE FOLLOWING as start: find first row where value >= current + offset
                                    if let (Some(curr_val), Expression::IntegerLiteral(lit)) =
                                        (&current_order_value, expr.as_ref())
                                    {
                                        if let Some(curr_f64) = value_to_f64(curr_val) {
                                            let lower_bound = curr_f64 + lit.value as f64;
                                            let mut start_idx = partition_len;
                                            for (j, &idx) in row_indices.iter().enumerate() {
                                                if let Some(row_val) = order_by_values[idx]
                                                    .first()
                                                    .and_then(|(v, _)| value_to_f64(v))
                                                {
                                                    if row_val >= lower_bound {
                                                        start_idx = j;
                                                        break;
                                                    }
                                                }
                                            }
                                            start_idx
                                        } else {
                                            i
                                        }
                                    } else {
                                        i
                                    }
                                } else if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                                    (i + lit.value as usize).min(partition_len - 1)
                                } else {
                                    i
                                }
                            }
                            WindowFrameBound::UnboundedFollowing => partition_len - 1,
                        };

                        // Calculate end bound
                        let end = if let Some(ref end_bound) = frame.end {
                            match end_bound {
                                WindowFrameBound::UnboundedFollowing => partition_len,
                                WindowFrameBound::CurrentRow => {
                                    if is_range {
                                        // For RANGE, end of peer group (O(1) lookup)
                                        peer_groups[i].1
                                    } else {
                                        i + 1
                                    }
                                }
                                WindowFrameBound::Following(expr) => {
                                    if is_range {
                                        // RANGE FOLLOWING: find last row where value <= current + offset
                                        if let (Some(curr_val), Expression::IntegerLiteral(lit)) =
                                            (&current_order_value, expr.as_ref())
                                        {
                                            if let Some(curr_f64) = value_to_f64(curr_val) {
                                                let upper_bound = curr_f64 + lit.value as f64;
                                                // Scan from end backwards to find last row in range
                                                let mut end_idx = 0;
                                                for (j, &idx) in row_indices.iter().enumerate() {
                                                    if let Some(row_val) = order_by_values[idx]
                                                        .first()
                                                        .and_then(|(v, _)| value_to_f64(v))
                                                    {
                                                        if row_val <= upper_bound {
                                                            end_idx = j + 1; // exclusive end
                                                        }
                                                    }
                                                }
                                                end_idx
                                            } else {
                                                partition_len
                                            }
                                        } else {
                                            partition_len
                                        }
                                    } else if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                                        (i + lit.value as usize + 1).min(partition_len)
                                    } else {
                                        partition_len
                                    }
                                }
                                WindowFrameBound::Preceding(expr) => {
                                    if is_range {
                                        // RANGE PRECEDING as end: find last row where value <= current - offset
                                        if let (Some(curr_val), Expression::IntegerLiteral(lit)) =
                                            (&current_order_value, expr.as_ref())
                                        {
                                            if let Some(curr_f64) = value_to_f64(curr_val) {
                                                let upper_bound = curr_f64 - lit.value as f64;
                                                let mut end_idx = 0;
                                                for (j, &idx) in row_indices.iter().enumerate() {
                                                    if let Some(row_val) = order_by_values[idx]
                                                        .first()
                                                        .and_then(|(v, _)| value_to_f64(v))
                                                    {
                                                        if row_val <= upper_bound {
                                                            end_idx = j + 1;
                                                        }
                                                    }
                                                }
                                                end_idx
                                            } else {
                                                i + 1
                                            }
                                        } else {
                                            i + 1
                                        }
                                    } else if let Expression::IntegerLiteral(lit) = expr.as_ref() {
                                        (i + 1).saturating_sub(lit.value as usize)
                                    } else {
                                        i + 1
                                    }
                                }
                                WindowFrameBound::UnboundedPreceding => 0,
                            }
                        } else {
                            // No end bound specified, SQL standard says implicit end is CURRENT ROW
                            // For ROWS: current row index + 1 (exclusive)
                            // For RANGE: end of current peer group
                            if is_range {
                                peer_groups[i].1
                            } else {
                                i + 1
                            }
                        };

                        (start, end)
                    } else {
                        // Default frame: UNBOUNDED PRECEDING to CURRENT ROW
                        (0, i + 1)
                    };

                    // Accumulate values within the frame
                    for &idx in &row_indices[frame_start..frame_end] {
                        let value = if let Some(col_idx) = arg_col_idx {
                            rows[idx]
                                .get(col_idx)
                                .cloned()
                                .unwrap_or_else(Value::null_unknown)
                        } else if has_expression_arg {
                            // Expression argument (e.g., val * 2) - use pre-computed value
                            expression_values[idx].clone()
                        } else {
                            // COUNT(*) counts all rows
                            Value::Integer(1)
                        };
                        agg_func.accumulate(&value, wf_info.is_distinct);
                    }
                    results[row_idx] = agg_func.result();
                }
            } else {
                // Without ORDER BY, compute aggregate over entire partition
                let mut agg_func = self
                    .function_registry
                    .get_aggregate(&wf_info.name)
                    .ok_or_else(|| {
                        Error::NotSupportedMessage(format!(
                            "Unknown aggregate function: {}",
                            wf_info.name
                        ))
                    })?;

                // Accumulate all values in the partition
                for &row_idx in &row_indices {
                    let value = if let Some(col_idx) = arg_col_idx {
                        rows[row_idx]
                            .get(col_idx)
                            .cloned()
                            .unwrap_or_else(Value::null_unknown)
                    } else if has_expression_arg {
                        // Expression argument (e.g., val * 2) - use pre-computed value
                        expression_values[row_idx].clone()
                    } else {
                        // COUNT(*) counts all rows
                        Value::Integer(1)
                    };
                    agg_func.accumulate(&value, wf_info.is_distinct);
                }
                let aggregate_result = agg_func.result();

                // Assign the same aggregate result to all rows in the partition
                for &row_idx in &row_indices {
                    results[row_idx] = aggregate_result.clone();
                }
            }
        }

        Ok(results)
    }

    /// Check if a SELECT statement has window functions
    pub(crate) fn has_window_functions(&self, stmt: &SelectStatement) -> bool {
        for col_expr in &stmt.columns {
            if self.expression_has_window_function(col_expr) {
                return true;
            }
        }
        false
    }

    /// Check if an expression contains window functions
    /// Uses find_window_in_expression for recursive checking
    fn expression_has_window_function(&self, expr: &Expression) -> bool {
        Self::find_window_in_expression(expr).is_some()
    }

    /// Extract aggregate function patterns from an expression (including nested ones)
    /// This handles cases like COALESCE(SUM(val), 0) where SUM(val) is nested
    fn extract_aggregate_patterns(expr: &Expression, executor: &Executor) -> Vec<String> {
        let mut patterns = Vec::new();
        Self::collect_aggregate_patterns(expr, executor, &mut patterns);
        patterns
    }

    /// Helper to recursively collect aggregate patterns from an expression
    fn collect_aggregate_patterns(
        expr: &Expression,
        executor: &Executor,
        patterns: &mut Vec<String>,
    ) {
        match expr {
            Expression::FunctionCall(func) => {
                if executor.function_registry.is_aggregate(&func.function) {
                    // This is an aggregate function - generate its pattern
                    let pattern = if func.arguments.is_empty()
                        || matches!(func.arguments.first(), Some(Expression::Star(_)))
                    {
                        format!("{}(*)", func.function)
                    } else if func.arguments.len() == 1 {
                        match &func.arguments[0] {
                            Expression::Identifier(id) => {
                                format!("{}({})", func.function, id.value)
                            }
                            Expression::QualifiedIdentifier(qid) => {
                                // Generate BOTH qualified and unqualified patterns
                                // e.g., for SUM(o.amount), add "SUM(amount)" first
                                let unqualified = format!("{}({})", func.function, qid.name.value);
                                patterns.push(unqualified);
                                // Then add qualified pattern "SUM(o.amount)"
                                format!(
                                    "{}({}.{})",
                                    func.function, qid.qualifier.value, qid.name.value
                                )
                            }
                            Expression::Distinct(d) => {
                                // Handle DISTINCT, e.g., COUNT(DISTINCT val)
                                match d.expr.as_ref() {
                                    Expression::Identifier(id) => {
                                        format!("{}(DISTINCT {})", func.function, id.value)
                                    }
                                    Expression::QualifiedIdentifier(qid) => {
                                        // Generate both qualified and unqualified patterns
                                        let unqualified = format!(
                                            "{}(DISTINCT {})",
                                            func.function, qid.name.value
                                        );
                                        patterns.push(unqualified);
                                        format!(
                                            "{}(DISTINCT {}.{})",
                                            func.function, qid.qualifier.value, qid.name.value
                                        )
                                    }
                                    _ => return,
                                }
                            }
                            _ => return,
                        }
                    } else {
                        return;
                    };
                    patterns.push(pattern);
                } else {
                    // Non-aggregate function - check its arguments for nested aggregates
                    for arg in &func.arguments {
                        Self::collect_aggregate_patterns(arg, executor, patterns);
                    }
                }
            }
            Expression::Infix(infix) => {
                Self::collect_aggregate_patterns(&infix.left, executor, patterns);
                Self::collect_aggregate_patterns(&infix.right, executor, patterns);
            }
            Expression::Prefix(prefix) => {
                Self::collect_aggregate_patterns(&prefix.right, executor, patterns);
            }
            Expression::Case(case) => {
                if let Some(ref value) = case.value {
                    Self::collect_aggregate_patterns(value, executor, patterns);
                }
                for when_clause in &case.when_clauses {
                    Self::collect_aggregate_patterns(&when_clause.condition, executor, patterns);
                    Self::collect_aggregate_patterns(&when_clause.then_result, executor, patterns);
                }
                if let Some(ref else_value) = case.else_value {
                    Self::collect_aggregate_patterns(else_value, executor, patterns);
                }
            }
            Expression::Cast(cast) => {
                Self::collect_aggregate_patterns(&cast.expr, executor, patterns);
            }
            Expression::Aliased(aliased) => {
                Self::collect_aggregate_patterns(&aliased.expression, executor, patterns);
            }
            Expression::List(list) => {
                for e in &list.elements {
                    Self::collect_aggregate_patterns(e, executor, patterns);
                }
            }
            _ => {}
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

    fn setup_test_data(executor: &Executor) {
        executor
            .execute(
                "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER)",
            )
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 100000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 90000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (3, 'Carol', 'Sales', 80000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (4, 'Dave', 'Sales', 85000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 95000)")
            .unwrap();
    }

    #[test]
    fn test_row_number_basic() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, ROW_NUMBER() OVER () FROM employees")
            .unwrap();

        let columns = result.columns();
        assert!(columns.len() >= 2);

        let mut count = 0;
        while result.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_row_number_with_order() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) FROM employees")
            .unwrap();

        let mut found_rows = false;
        while result.next() {
            found_rows = true;
        }
        assert!(found_rows);
    }

    #[test]
    fn test_row_number_with_partition() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, dept, ROW_NUMBER() OVER (PARTITION BY dept) FROM employees")
            .unwrap();

        let mut count = 0;
        while result.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_has_window_functions() {
        let executor = create_test_executor();

        // Test with window function
        let mut parser = crate::parser::Parser::new("SELECT ROW_NUMBER() OVER () FROM test");
        if let Ok(program) = parser.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                assert!(executor.has_window_functions(stmt));
            }
        }

        // Test without window function
        let mut parser2 = crate::parser::Parser::new("SELECT * FROM test");
        if let Ok(program) = parser2.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                assert!(!executor.has_window_functions(stmt));
            }
        }
    }

    #[test]
    fn test_window_function_info() {
        let info = WindowFunctionInfo {
            name: "ROW_NUMBER".to_string(),
            arguments: vec![],
            partition_by: vec!["dept".to_string()],
            order_by: vec![],
            frame: None,
            column_name: "rn".to_string(),
            column_index: 0,
            is_distinct: false,
        };

        assert_eq!(info.name, "ROW_NUMBER");
        assert_eq!(info.partition_by.len(), 1);
        assert_eq!(info.column_name, "rn");
    }

    #[test]
    fn test_percent_rank_with_order() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT salary, PERCENT_RANK() OVER (ORDER BY salary) AS pct FROM employees ORDER BY salary")
            .unwrap();

        let mut pct_ranks = Vec::new();
        let mut row_count = 0;
        while result.next() {
            let row = result.row();
            if let Some(pct) = row.get(1) {
                match pct {
                    crate::core::Value::Float(f) => pct_ranks.push(*f),
                    crate::core::Value::Integer(i) => pct_ranks.push(*i as f64),
                    _ => {}
                }
            }
            row_count += 1;
        }

        eprintln!("DEBUG: pct_ranks = {:?}", pct_ranks);
        eprintln!("DEBUG: row_count = {}", row_count);
        eprintln!("DEBUG: columns = {:?}", result.columns());

        // First row should have pct_rank = 0.0
        assert!(
            (pct_ranks[0] - 0.0).abs() < 0.001,
            "First pct_rank should be 0.0, got {}",
            pct_ranks[0]
        );

        // Verify monotonically non-decreasing
        for i in 1..pct_ranks.len() {
            assert!(
                pct_ranks[i] >= pct_ranks[i - 1],
                "pct_ranks should be non-decreasing"
            );
        }
    }
}
