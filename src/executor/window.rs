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

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;

use crate::common::{CompactVec, StringMap};
use crate::core::row_vec::RowVec;
use crate::core::value::NULL_VALUE;
use crate::core::{Error, Result, Row, Value};

/// Type alias for partition keys - stack-allocated for common case (up to 4 columns)
type PartitionKey = SmallVec<[Value; 4]>;

use crate::functions::WindowFunction;
use crate::parser::ast::*;
use crate::storage::traits::{QueryResult, Table};

use super::context::ExecutionContext;
use super::expression::{ExpressionEval, MultiExpressionEval};
use super::result::{ColumnarResult, ExecutorResult};
use super::utils::build_column_index_map;
use super::Executor;

/// Wrapper for parallel writes to disjoint indices in a Vec<Value>.
///
/// SAFETY INVARIANT: Each index must be written by at most one thread.
/// In window function computation, each partition writes to its own row indices,
/// which are guaranteed to be disjoint (each row belongs to exactly one partition).
struct ParallelVec {
    ptr: *mut Value,
    len: usize,
}

// SAFETY: ParallelVec is Send because we only write to disjoint indices
// and Value is Send. Each partition writes to its own indices.
unsafe impl Send for ParallelVec {}

// SAFETY: ParallelVec is Sync because parallel writes to disjoint indices
// don't race with each other. Each row index belongs to exactly one partition.
unsafe impl Sync for ParallelVec {}

impl ParallelVec {
    /// Create from a mutable Vec reference.
    /// The Vec must not be accessed through other references while ParallelVec exists.
    #[inline]
    fn new(vec: &mut Vec<Value>) -> Self {
        Self {
            ptr: vec.as_mut_ptr(),
            len: vec.len(),
        }
    }

    /// Write a value at the given index.
    /// SAFETY: No two threads may write to the same index concurrently.
    #[inline]
    fn write(&self, idx: usize, value: Value) {
        debug_assert!(idx < self.len, "index out of bounds");
        // SAFETY: Caller guarantees no concurrent writes to the same index.
        // In window functions, each partition writes to disjoint row indices.
        unsafe {
            *self.ptr.add(idx) = value;
        }
    }
}

/// Information about a window function call in a SELECT list
#[derive(Clone, Debug)]
pub struct WindowFunctionInfo {
    /// The window function name (ROW_NUMBER, RANK, etc.)
    pub name: String,
    /// Arguments to the function (for LEAD, LAG, NTILE)
    pub arguments: Vec<Expression>,
    /// Partition by column names (simple identifiers only, for fast-path lookups)
    pub partition_by: Vec<String>,
    /// Original PARTITION BY expressions (includes function calls, complex exprs)
    pub partition_by_exprs: Vec<Expression>,
    /// Order by expressions
    pub order_by: Vec<OrderByExpression>,
    /// Window frame specification (e.g., ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
    pub frame: Option<WindowFrame>,
    /// Result column name (may include alias)
    pub column_name: String,
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
    /// Expression containing one or more window functions.
    /// Stores (expression, list_of_window_function_names_lowercase).
    /// Each Window node in the expression is replaced with a placeholder identifier
    /// referencing the corresponding name at the same index.
    ExpressionWithWindow(Expression, Vec<String>),
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

/// Columnar layout for ORDER BY values - optimized for sorting performance
///
/// Instead of `Vec<Vec<(Value, bool)>>` (row-oriented, N allocations for N rows),
/// this uses `Vec<Vec<Value>>` (column-oriented, K allocations for K ORDER BY columns).
///
/// Benefits:
/// - Reduces allocations from O(N) to O(K) where K = number of ORDER BY columns
/// - Stores ascending flags once per column instead of once per value
/// - Better cache locality when accessing sort keys across rows
#[derive(Clone, Debug)]
pub struct ColumnarOrderByValues {
    /// Column values: columns[col_idx][row_idx] = value
    columns: Vec<Vec<Value>>,
    /// Ascending flags: one per ORDER BY column
    ascending: Vec<bool>,
    /// Number of rows
    num_rows: usize,
}

impl ColumnarOrderByValues {
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0 || self.columns.is_empty()
    }

    /// Get number of columns
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get value at (row, column)
    #[inline]
    pub fn get(&self, row_idx: usize, col_idx: usize) -> Option<&Value> {
        self.columns.get(col_idx).and_then(|col| col.get(row_idx))
    }

    /// Get first ORDER BY value for a row
    #[inline]
    pub fn get_first(&self, row_idx: usize) -> Option<&Value> {
        self.get(row_idx, 0)
    }

    /// Get ascending flag for column
    #[inline]
    pub fn is_ascending(&self, col_idx: usize) -> bool {
        self.ascending.get(col_idx).copied().unwrap_or(true)
    }

    /// Compare ORDER BY values of two rows for equality (without cloning)
    /// Returns true if all ORDER BY column values are equal
    #[inline]
    pub fn rows_equal(&self, row_a: usize, row_b: usize) -> bool {
        for col in &self.columns {
            let val_a = col.get(row_a);
            let val_b = col.get(row_b);
            match (val_a, val_b) {
                (Some(a), Some(b)) if a == b => continue,
                (None, None) => continue,
                _ => return false,
            }
        }
        true
    }
}

/// Pre-grouped state for window function PARTITION BY optimization
/// When rows are fetched grouped by an indexed partition column, we can skip hash-based grouping
#[derive(Clone)]
pub struct WindowPreGroupedState {
    /// Pre-built partition map: partition key -> row indices
    pub partition_map: FxHashMap<PartitionKey, Vec<usize>>,
    /// The column name (lowercase) that the partition map was built from
    pub partition_column: String,
}

impl Executor {
    /// Execute SELECT with window functions
    /// Accepts &[(i64, Row)] to allow RowVec to be passed directly via deref
    pub(crate) fn execute_select_with_window_functions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
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
        base_rows: &[(i64, Row)],
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
        base_rows: &[(i64, Row)],
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
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        pre_sorted: Option<WindowPreSortedState>,
        pre_grouped: Option<WindowPreGroupedState>,
    ) -> Result<Box<dyn QueryResult>> {
        // Parse window functions from the SELECT list
        let window_functions = self.parse_window_functions(stmt, base_columns)?;

        if window_functions.is_empty() {
            // No window functions found, return base result
            let mut rows = RowVec::with_capacity(base_rows.len());
            for (id, row) in base_rows.iter() {
                rows.push((*id, row.clone()));
            }
            return Ok(Box::new(ExecutorResult::new(base_columns.to_vec(), rows)));
        }

        // OPTIMIZATION: LIMIT pushdown for PARTITION BY queries
        // Only safe when there is no top-level ORDER BY (otherwise the sort must see all
        // rows before LIMIT can be applied) and when all window functions share the same
        // PARTITION BY so one partition map can serve them all.
        if stmt.order_by.is_empty() && stmt.offset.is_none() {
            if let Some(limit_expr) = &stmt.limit {
                let has_partition_by = window_functions
                    .iter()
                    .any(|wf| !wf.partition_by_exprs.is_empty());

                if has_partition_by && Self::all_partitions_match(&window_functions) {
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
        }

        // Build column index map for base columns
        let mut col_index_map = build_column_index_map(base_columns);

        // Build a mapping from aggregate expression patterns to their column names
        // This handles cases like:
        // - SUM(val) AS grp_sum -> maps "sum(val)" to column index of "grp_sum"
        // - COALESCE(SUM(val), 0) AS total -> maps "sum(val)" to column index of "total"
        for col_expr in stmt.columns.iter() {
            if let Expression::Aliased(aliased) = col_expr {
                let alias_lower = aliased.alias.value_lower.as_str();
                if let Some(&idx) = col_index_map.get(alias_lower) {
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

        // OPTIMIZATION: Precompute ORDER BY values ONCE for each unique ORDER BY clause
        // This avoids redundant computation when multiple window functions share the same ORDER BY
        // We use a Vec for the cache since the number of unique ORDER BY clauses is typically small
        // NOTE: We use string representation for semantic comparison because PartialEq on expressions
        // compares token positions, making structurally identical expressions from different window
        // functions appear different.
        let mut order_by_cache: Vec<(String, ColumnarOrderByValues)> = Vec::new();
        for wf in &window_functions {
            if !wf.order_by.is_empty() {
                // Create a semantic key from the ORDER BY expressions (ignores token positions)
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                // Check if this ORDER BY clause is already in the cache
                let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                if !already_cached {
                    let precomputed = self.precompute_order_by_values(
                        &wf.order_by,
                        base_rows,
                        base_columns,
                        &col_index_map,
                        ctx,
                    );
                    order_by_cache.push((cache_key, precomputed));
                }
            }
        }

        let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();
        for wf in &window_functions {
            let window_values = self.compute_window_function(
                wf,
                base_rows,
                base_columns,
                &col_index_map,
                ctx,
                pre_sorted.as_ref(),
                pre_grouped.as_ref(),
                &order_by_cache,
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

        // Step 3: Build result using COLUMNAR storage
        // OPTIMIZATION: Instead of allocating one Row per result row, we store data column-major
        // and use ColumnarResult which materializes rows lazily with a single reused buffer.
        // This reduces allocations from O(num_rows) to O(num_columns).
        let num_rows = base_rows.len();

        // Build aliases from col_index_map for expression evaluation
        let agg_aliases: Vec<(String, usize)> =
            col_index_map.iter().map(|(k, v)| (k.clone(), *v)).collect();

        // Pre-transform expressions with window functions by replacing Window expr with Identifier
        // This is done once, not per row
        let transformed_items: Vec<_> = select_items
            .iter()
            .map(|item| match &item.source {
                SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                    let mut counter = 0;
                    let transformed =
                        Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                    (item, Some(transformed), Some(wf_names.clone()))
                }
                _ => (item, None, None),
            })
            .collect();

        // Build extended columns (base_columns + synthetic window columns)
        let mut extended_columns = base_columns.to_vec();
        let mut added_wf_names: Vec<String> = Vec::new();
        for (_, _, wf_names_opt) in &transformed_items {
            if let Some(wf_names) = wf_names_opt {
                for wf_name in wf_names {
                    if !added_wf_names.contains(wf_name) {
                        extended_columns.push(wf_name.clone());
                        added_wf_names.push(wf_name.clone());
                    }
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

        // OPTIMIZATION: Pre-allocate ext_values buffer for extended expressions
        let ext_values_capacity = if !ext_expr_items.is_empty() {
            base_rows.first().map_or(0, |r| r.1.len()) + added_wf_names.len()
        } else {
            0
        };
        let mut ext_values: CompactVec<Value> = CompactVec::with_capacity(ext_values_capacity);

        // Number of output columns
        let num_items = select_items.len();

        // COLUMNAR STORAGE OPTIMIZATION: Build columns in column-major order
        // This is more cache-efficient than row-by-row iteration and enables
        // moving window function Vecs directly (zero-copy for window results).
        //
        // Phase 1: Build non-expression columns (WindowFunction, BaseColumn)
        // Phase 2: Fill expression columns row-by-row (requires evaluation)
        let mut column_data: Vec<Vec<Value>> = Vec::with_capacity(num_items);

        // Track which window functions are used in expressions (need to keep them)
        let wf_names_in_exprs: std::collections::HashSet<&str> =
            added_wf_names.iter().map(|s| s.as_str()).collect();

        // Phase 1: Build columns for WindowFunction and BaseColumn items
        // Expression columns get placeholder Vecs (filled in Phase 2)
        for (item, _, _) in &transformed_items {
            match &item.source {
                SelectItemSource::WindowFunction(wf_name_lower) => {
                    // OPTIMIZATION: Move or clone the entire Vec at once
                    // If this window function is also used in an expression, we need to keep it
                    if wf_names_in_exprs.contains(wf_name_lower.as_str()) {
                        // Clone the entire Vec (more cache-efficient than element-by-element)
                        let values = window_value_map
                            .get(wf_name_lower)
                            .cloned()
                            .unwrap_or_else(|| vec![NULL_VALUE; num_rows]);
                        column_data.push(values);
                    } else {
                        // Move the Vec directly (zero-copy)
                        let values = window_value_map
                            .remove(wf_name_lower)
                            .unwrap_or_else(|| vec![NULL_VALUE; num_rows]);
                        column_data.push(values);
                    }
                }
                SelectItemSource::BaseColumn(base_col_idx) => {
                    // OPTIMIZATION: Build base column in one pass (column-wise)
                    let mut values = Vec::with_capacity(num_rows);
                    for (_, base_row) in base_rows {
                        values.push(base_row.get(*base_col_idx).cloned().unwrap_or(NULL_VALUE));
                    }
                    column_data.push(values);
                }
                SelectItemSource::Expression(_) | SelectItemSource::ExpressionWithWindow(_, _) => {
                    // Placeholder - will be filled in Phase 2
                    column_data.push(vec![NULL_VALUE; num_rows]);
                }
            }
        }

        // Phase 2: Fill expression columns row-by-row (requires evaluation)
        let has_expressions = !base_expr_items.is_empty() || !ext_expr_items.is_empty();
        if has_expressions {
            for (row_idx, (_, base_row)) in base_rows.iter().enumerate() {
                // Evaluate base expressions and update their column values
                if let Some(ref mut eval) = base_eval {
                    if let Ok(base_results) = eval.eval_all(base_row) {
                        for (eval_idx, (item_idx, _)) in base_expr_items.iter().enumerate() {
                            if eval_idx < base_results.len() {
                                column_data[*item_idx][row_idx] = base_results[eval_idx].clone();
                            }
                        }
                    }
                }

                // Evaluate extended expressions (if any) and update their column values
                if !ext_expr_items.is_empty() {
                    if let Some(ref mut eval) = ext_eval {
                        // Reuse ext_values buffer: clear and refill
                        ext_values.clear();
                        ext_values.extend(base_row.iter().cloned());
                        for wf_name in &added_wf_names {
                            let wf_value = window_value_map
                                .get(wf_name)
                                .and_then(|vals| vals.get(row_idx).cloned())
                                .unwrap_or(NULL_VALUE);
                            ext_values.push(wf_value);
                        }
                        let ext_row = Row::from_compact_vec(ext_values.clone());

                        if let Ok(ext_result_values) = eval.eval_all(&ext_row) {
                            for (eval_idx, (item_idx, _)) in ext_expr_items.iter().enumerate() {
                                if eval_idx < ext_result_values.len() {
                                    column_data[*item_idx][row_idx] =
                                        ext_result_values[eval_idx].clone();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return ColumnarResult which materializes rows lazily with zero per-row allocation
        Ok(Box::new(ColumnarResult::new(result_columns, column_data)))
    }

    /// Streaming execution for window functions with LIMIT pushdown
    /// Processes partitions one at a time and stops early when LIMIT is reached
    fn execute_select_with_window_functions_streaming(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        window_functions: &[WindowFunctionInfo],
        limit: usize,
    ) -> Result<Box<dyn QueryResult>> {
        // Use the first window function with PARTITION BY for partitioning
        let primary_wf = window_functions
            .iter()
            .find(|wf| !wf.partition_by_exprs.is_empty())
            .unwrap(); // Safe: we checked has_partition_by before calling this

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);

        // Build partition map from the primary window function
        let partitions =
            Self::build_partition_map(primary_wf, base_rows, base_columns, &col_index_map, ctx)?;

        // Build result columns from SELECT list
        let select_items = self.parse_select_list_for_window(stmt, base_columns, window_functions);
        let result_columns: Vec<String> =
            select_items.iter().map(|i| i.output_name.clone()).collect();

        // Precompute ORDER BY values once for each unique ORDER BY clause
        let mut order_by_cache: Vec<(String, ColumnarOrderByValues)> = Vec::new();
        for wf in window_functions {
            if !wf.order_by.is_empty() {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                if !already_cached {
                    let precomputed = self.precompute_order_by_values(
                        &wf.order_by,
                        base_rows,
                        base_columns,
                        &col_index_map,
                        ctx,
                    );
                    order_by_cache.push((cache_key, precomputed));
                }
            }
        }

        // Resolve window functions from registry
        let resolved_wfs: Vec<_> = window_functions
            .iter()
            .map(|wf| {
                let is_agg = self.function_registry.is_aggregate(&wf.name);
                let func = if is_agg {
                    None
                } else {
                    self.function_registry.get_window(&wf.name)
                };
                (wf, func, is_agg)
            })
            .collect();

        // Precompute aggregate window functions once over all rows (not per partition)
        let mut precomputed_agg: StringMap<Vec<Value>> = StringMap::new();
        for (wf, _, is_agg) in &resolved_wfs {
            if *is_agg {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let precomputed_order_by = order_by_cache
                    .iter()
                    .find(|(key, _)| key == &cache_key)
                    .map(|(_, v)| v);
                let agg_results = self.compute_aggregate_window_function(
                    wf,
                    base_rows,
                    base_columns,
                    &col_index_map,
                    ctx,
                    None,
                    precomputed_order_by,
                )?;
                precomputed_agg.insert(wf.column_name.to_lowercase(), agg_results);
            }
        }

        // Process partitions one at a time, stopping when we have enough rows
        let mut result_rows = RowVec::with_capacity(limit);
        let mut result_row_id = 0i64;

        let partitions_vec: Vec<_> = partitions.into_iter().collect();

        for (_partition_key, row_indices) in partitions_vec {
            if result_rows.len() >= limit {
                break;
            }

            // Compute window functions for this partition.
            let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();

            for (wf, win_func_opt, is_agg) in &resolved_wfs {
                if *is_agg {
                    // Slice precomputed aggregate results for this partition
                    let key = wf.column_name.to_lowercase();
                    if let Some(all_results) = precomputed_agg.get(&key) {
                        let partition_vals: Vec<Value> = row_indices
                            .iter()
                            .map(|&i| all_results[i].clone())
                            .collect();
                        window_value_map.insert(key, partition_vals);
                    }
                } else if let Some(win_func) = win_func_opt {
                    let cache_key = Self::order_by_cache_key(&wf.order_by);
                    let precomputed_order_by = order_by_cache
                        .iter()
                        .find(|(key, _)| key == &cache_key)
                        .map(|(_, v)| v);
                    let (sorted_values, sorted_indices) = self.compute_window_for_partition(
                        win_func.as_ref(),
                        wf,
                        base_rows,
                        row_indices.clone(),
                        precomputed_order_by,
                        base_columns,
                        &col_index_map,
                        ctx,
                        false,
                    )?;
                    // Remap: sorted_values[pos] corresponds to row sorted_indices[pos].
                    // Build a vec indexed by position within row_indices using an O(n) index map.
                    let orig_to_local: FxHashMap<usize, usize> = row_indices
                        .iter()
                        .enumerate()
                        .map(|(local, &orig)| (orig, local))
                        .collect();
                    let mut by_orig = vec![NULL_VALUE; row_indices.len()];
                    for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                        if let Some(&local) = orig_to_local.get(&orig_idx) {
                            by_orig[local] = sorted_values[pos].clone();
                        }
                    }
                    window_value_map.insert(wf.column_name.to_lowercase(), by_orig);
                }
            }

            // Precompile expression evaluators once per partition (not per row)
            enum CompiledItem<'a> {
                BaseColumn(usize),
                WindowFunction(&'a str),
                Expression(ExpressionEval),
                ExpressionWithWindow(ExpressionEval, Vec<&'a str>),
            }
            let num_items = select_items.len();
            let mut compiled_items: Vec<CompiledItem<'_>> = Vec::with_capacity(num_items);
            for item in &select_items {
                compiled_items.push(match &item.source {
                    SelectItemSource::BaseColumn(idx) => CompiledItem::BaseColumn(*idx),
                    SelectItemSource::WindowFunction(name) => {
                        CompiledItem::WindowFunction(name.as_str())
                    }
                    SelectItemSource::Expression(expr) => {
                        CompiledItem::Expression(ExpressionEval::compile(expr, base_columns)?)
                    }
                    SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                        let mut counter = 0;
                        let transformed =
                            Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                        let mut ext_columns = base_columns.to_vec();
                        for wf_name in wf_names {
                            ext_columns.push(wf_name.clone());
                        }
                        let eval = ExpressionEval::compile(&transformed, &ext_columns)?;
                        let name_refs: Vec<&str> = wf_names.iter().map(|s| s.as_str()).collect();
                        CompiledItem::ExpressionWithWindow(eval, name_refs)
                    }
                });
            }

            let ext_capacity = base_rows.first().map_or(0, |r| r.1.len()) + 1;
            let mut ext_values: CompactVec<Value> = CompactVec::with_capacity(ext_capacity);

            // Output rows in partition order (row_indices order)
            for (local_pos, &orig_idx) in row_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break;
                }

                let base_row = &base_rows[orig_idx].1;
                let mut values: CompactVec<Value> = CompactVec::with_capacity(num_items);

                for ci in &mut compiled_items {
                    let value = match ci {
                        CompiledItem::BaseColumn(col_idx) => {
                            base_row.get(*col_idx).cloned().unwrap_or(NULL_VALUE)
                        }
                        CompiledItem::WindowFunction(wf_name_lower) => window_value_map
                            .get(*wf_name_lower)
                            .and_then(|vals| vals.get(local_pos).cloned())
                            .unwrap_or(NULL_VALUE),
                        CompiledItem::Expression(eval) => eval.eval(base_row)?,
                        CompiledItem::ExpressionWithWindow(eval, wf_name_refs) => {
                            ext_values.clear();
                            ext_values.extend(base_row.iter().cloned());
                            for wf_name in wf_name_refs.iter() {
                                let wf_value = window_value_map
                                    .get(*wf_name)
                                    .and_then(|vals| vals.get(local_pos).cloned())
                                    .unwrap_or(NULL_VALUE);
                                ext_values.push(wf_value);
                            }
                            let ext_row = Row::from_compact_vec(ext_values.clone());
                            eval.eval(&ext_row)?
                        }
                    };
                    values.push(value);
                }
                result_rows.push((result_row_id, Row::from_compact_vec(values)));
                result_row_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(result_columns, result_rows)))
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

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);

        // Resolve window functions from registry
        let resolved_wfs: Vec<_> = window_functions
            .iter()
            .map(|wf| {
                let is_agg = self.function_registry.is_aggregate(&wf.name);
                let func = if is_agg {
                    None
                } else {
                    self.function_registry.get_window(&wf.name)
                };
                (wf, func, is_agg)
            })
            .collect();

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
        let mut result_rows = RowVec::with_capacity(limit);
        let mut result_row_id = 0i64;

        for partition_value in partition_values {
            if result_rows.len() >= limit {
                break;
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

            // Precompute ORDER BY values for each unique ORDER BY clause
            let mut order_by_cache: Vec<(String, ColumnarOrderByValues)> = Vec::new();
            for wf in &window_functions {
                if !wf.order_by.is_empty() {
                    let cache_key = Self::order_by_cache_key(&wf.order_by);
                    let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                    if !already_cached {
                        let precomputed = self.precompute_order_by_values(
                            &wf.order_by,
                            &partition_rows,
                            base_columns,
                            &col_index_map,
                            ctx,
                        );
                        order_by_cache.push((cache_key, precomputed));
                    }
                }
            }

            // Compute ALL window functions for this partition.
            // Values are stored keyed by local position within row_indices.
            let row_indices: Vec<usize> = (0..partition_rows.len()).collect();
            let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();

            for (wf, win_func_opt, is_agg) in &resolved_wfs {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let precomputed_order_by = order_by_cache
                    .iter()
                    .find(|(key, _)| key == &cache_key)
                    .map(|(_, v)| v);

                if *is_agg {
                    let agg_results = self.compute_aggregate_window_function(
                        wf,
                        &partition_rows,
                        base_columns,
                        &col_index_map,
                        ctx,
                        None,
                        precomputed_order_by,
                    )?;
                    // agg_results is already indexed by local row index
                    window_value_map.insert(wf.column_name.to_lowercase(), agg_results);
                } else if let Some(win_func) = win_func_opt {
                    let (sorted_values, sorted_indices) = self.compute_window_for_partition(
                        win_func.as_ref(),
                        wf,
                        &partition_rows,
                        row_indices.clone(),
                        precomputed_order_by,
                        base_columns,
                        &col_index_map,
                        ctx,
                        false,
                    )?;
                    // Remap from sorted order to local row order
                    let mut by_local = vec![NULL_VALUE; row_indices.len()];
                    for (pos, &local_idx) in sorted_indices.iter().enumerate() {
                        by_local[local_idx] = sorted_values[pos].clone();
                    }
                    window_value_map.insert(wf.column_name.to_lowercase(), by_local);
                }
            }

            // Precompile expression evaluators once per partition (not per row)
            enum CompiledItemLazy<'a> {
                BaseColumn(usize),
                WindowFunction(&'a str),
                Expression(ExpressionEval),
                ExpressionWithWindow(ExpressionEval, Vec<&'a str>),
            }
            let num_items = select_items.len();
            let mut compiled_items: Vec<CompiledItemLazy<'_>> = Vec::with_capacity(num_items);
            for item in &select_items {
                compiled_items.push(match &item.source {
                    SelectItemSource::BaseColumn(idx) => CompiledItemLazy::BaseColumn(*idx),
                    SelectItemSource::WindowFunction(name) => {
                        CompiledItemLazy::WindowFunction(name.as_str())
                    }
                    SelectItemSource::Expression(expr) => {
                        CompiledItemLazy::Expression(ExpressionEval::compile(expr, base_columns)?)
                    }
                    SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                        let mut counter = 0;
                        let transformed =
                            Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                        let mut ext_columns = base_columns.to_vec();
                        for wf_name in wf_names {
                            ext_columns.push(wf_name.clone());
                        }
                        let eval = ExpressionEval::compile(&transformed, &ext_columns)?;
                        let name_refs: Vec<&str> = wf_names.iter().map(|s| s.as_str()).collect();
                        CompiledItemLazy::ExpressionWithWindow(eval, name_refs)
                    }
                });
            }

            let mut ext_values: CompactVec<Value> =
                CompactVec::with_capacity(partition_rows.first().map_or(0, |r| r.1.len()) + 1);

            // Output rows in partition order
            for (local_pos, &row_idx) in row_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break;
                }

                let (_, base_row) = &partition_rows[row_idx];
                let mut values: CompactVec<Value> = CompactVec::with_capacity(num_items);

                for ci in &mut compiled_items {
                    let val = match ci {
                        CompiledItemLazy::BaseColumn(col_idx) => {
                            base_row.get(*col_idx).cloned().unwrap_or(NULL_VALUE)
                        }
                        CompiledItemLazy::WindowFunction(wf_name_lower) => window_value_map
                            .get(*wf_name_lower)
                            .and_then(|vals| vals.get(local_pos).cloned())
                            .unwrap_or(NULL_VALUE),
                        CompiledItemLazy::Expression(eval) => eval.eval(base_row)?,
                        CompiledItemLazy::ExpressionWithWindow(eval, wf_name_refs) => {
                            ext_values.clear();
                            ext_values.extend(base_row.iter().cloned());
                            for wf_name in wf_name_refs.iter() {
                                let wf_value = window_value_map
                                    .get(*wf_name)
                                    .and_then(|vals| vals.get(local_pos).cloned())
                                    .unwrap_or(NULL_VALUE);
                                ext_values.push(wf_value);
                            }
                            let ext_row = Row::from_compact_vec(ext_values.clone());
                            eval.eval(&ext_row)?
                        }
                    };
                    values.push(val);
                }
                result_rows.push((result_row_id, Row::from_compact_vec(values)));
                result_row_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(result_columns, result_rows)))
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
                        let alias = aliased.alias.value.to_string();
                        items.push(SelectItem {
                            output_name: alias.clone(),
                            // OPTIMIZATION: Store lowercase for O(1) lookup in window_value_map
                            source: SelectItemSource::WindowFunction(alias.to_lowercase()),
                        });
                        wf_idx += 1;
                    } else if let Expression::Identifier(id) = aliased.expression.as_ref() {
                        // Aliased column reference - use pre-computed lowercase
                        if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::BaseColumn(idx),
                            });
                        } else {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    } else {
                        // Collect all embedded window functions
                        let mut embedded_windows = Vec::new();
                        Self::collect_all_windows_in_expression(
                            aliased.expression.as_ref(),
                            &mut embedded_windows,
                        );
                        if !embedded_windows.is_empty() {
                            let mut wf_names = Vec::with_capacity(embedded_windows.len());
                            for _ in &embedded_windows {
                                let name = if wf_idx < window_functions.len() {
                                    window_functions[wf_idx].column_name.to_lowercase()
                                } else {
                                    format!("__wf_{}", wf_idx)
                                };
                                wf_names.push(name);
                                wf_idx += 1;
                            }
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::ExpressionWithWindow(
                                    aliased.expression.as_ref().clone(),
                                    wf_names,
                                ),
                            });
                        } else if let Expression::FunctionCall(_) = aliased.expression.as_ref() {
                            // Aliased function call - check if it's an aggregate that's already in base_columns
                            let alias_lower = aliased.alias.value_lower.as_str();
                            if let Some(&idx) = col_index_map.get(alias_lower) {
                                items.push(SelectItem {
                                    output_name: aliased.alias.value.to_string(),
                                    source: SelectItemSource::BaseColumn(idx),
                                });
                            } else {
                                items.push(SelectItem {
                                    output_name: aliased.alias.value.to_string(),
                                    source: SelectItemSource::Expression(
                                        aliased.expression.as_ref().clone(),
                                    ),
                                });
                            }
                        } else {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    }
                }
                Expression::Identifier(id) => {
                    // Simple column reference - use pre-computed lowercase
                    if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                        items.push(SelectItem {
                            output_name: id.value.to_string(),
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
                                    output_name: id.value.to_string(),
                                    source: SelectItemSource::BaseColumn(idx),
                                });
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            // Still not found - treat as expression and let evaluator handle it
                            items.push(SelectItem {
                                output_name: id.value.to_string(),
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
                    } else if let Some(&idx) = col_index_map.get(qid.name.value_lower.as_str()) {
                        // Fall back to unqualified name
                        items.push(SelectItem {
                            output_name: qid.name.value.to_string(),
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
                    // First check for embedded window functions (e.g., ABS(ROW_NUMBER() OVER (...)))
                    let mut embedded_windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut embedded_windows);
                    if !embedded_windows.is_empty() {
                        let mut wf_names = Vec::with_capacity(embedded_windows.len());
                        for _ in &embedded_windows {
                            let name = if wf_idx < window_functions.len() {
                                window_functions[wf_idx].column_name.to_lowercase()
                            } else {
                                format!("__wf_{}", wf_idx)
                            };
                            wf_names.push(name);
                            wf_idx += 1;
                        }
                        items.push(SelectItem {
                            output_name: format!("expr_{}", items.len()),
                            source: SelectItemSource::ExpressionWithWindow(
                                col_expr.clone(),
                                wf_names,
                            ),
                        });
                    } else {
                        // Check if this is an aggregate function already in base columns
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

                        if let Some(&idx) = col_index_map.get(&func_col_name.to_lowercase()) {
                            items.push(SelectItem {
                                output_name: func_col_name,
                                source: SelectItemSource::BaseColumn(idx),
                            });
                        } else {
                            items.push(SelectItem {
                                output_name: format!("expr_{}", items.len()),
                                source: SelectItemSource::Expression(col_expr.clone()),
                            });
                        }
                    }
                }
                _ => {
                    // Other expressions - collect all embedded window functions
                    let mut embedded_windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut embedded_windows);
                    if !embedded_windows.is_empty() {
                        let mut wf_names = Vec::with_capacity(embedded_windows.len());
                        for _ in &embedded_windows {
                            let name = if wf_idx < window_functions.len() {
                                window_functions[wf_idx].column_name.to_lowercase()
                            } else {
                                format!("__wf_{}", wf_idx)
                            };
                            wf_names.push(name);
                            wf_idx += 1;
                        }
                        items.push(SelectItem {
                            output_name: format!("expr_{}", items.len()),
                            source: SelectItemSource::ExpressionWithWindow(
                                col_expr.clone(),
                                wf_names,
                            ),
                        });
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

    /// Parse window functions from SELECT list.
    /// Collects ALL window functions, including multiple ones nested in a single expression.
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
                        self.extract_window_function_info(window_expr, &stmt.window_defs)?;
                    window_functions.push(wf_info);
                    col_idx += 1;
                }
                Expression::Aliased(aliased) => {
                    if let Expression::Window(window_expr) = aliased.expression.as_ref() {
                        let mut wf_info =
                            self.extract_window_function_info(window_expr, &stmt.window_defs)?;
                        wf_info.column_name = aliased.alias.value.to_string();
                        window_functions.push(wf_info);
                    } else {
                        // Collect ALL window functions in the expression
                        let mut windows = Vec::new();
                        Self::collect_all_windows_in_expression(
                            aliased.expression.as_ref(),
                            &mut windows,
                        );
                        for (i, window_expr) in windows.into_iter().enumerate() {
                            let mut wf_info =
                                self.extract_window_function_info(window_expr, &stmt.window_defs)?;
                            wf_info.column_name = format!("__wf_{}_{}", col_idx, i);
                            window_functions.push(wf_info);
                        }
                    }
                    col_idx += 1;
                }
                _ => {
                    let mut windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut windows);
                    for (i, window_expr) in windows.into_iter().enumerate() {
                        let mut wf_info =
                            self.extract_window_function_info(window_expr, &stmt.window_defs)?;
                        wf_info.column_name = format!("__wf_{}_{}", col_idx, i);
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
    /// Collect all window expressions found anywhere in an expression tree (depth-first order).
    fn collect_all_windows_in_expression<'a>(
        expr: &'a Expression,
        out: &mut Vec<&'a WindowExpression>,
    ) {
        match expr {
            Expression::Window(w) => out.push(w),
            Expression::Infix(infix) => {
                Self::collect_all_windows_in_expression(&infix.left, out);
                Self::collect_all_windows_in_expression(&infix.right, out);
            }
            Expression::Prefix(prefix) => {
                Self::collect_all_windows_in_expression(&prefix.right, out);
            }
            Expression::FunctionCall(f) => {
                for arg in &f.arguments {
                    Self::collect_all_windows_in_expression(arg, out);
                }
            }
            Expression::Aliased(a) => {
                Self::collect_all_windows_in_expression(&a.expression, out);
            }
            Expression::Cast(cast) => {
                Self::collect_all_windows_in_expression(&cast.expr, out);
            }
            Expression::Distinct(d) => {
                Self::collect_all_windows_in_expression(&d.expr, out);
            }
            Expression::Case(c) => {
                if let Some(v) = &c.value {
                    Self::collect_all_windows_in_expression(v, out);
                }
                for clause in &c.when_clauses {
                    Self::collect_all_windows_in_expression(&clause.condition, out);
                    Self::collect_all_windows_in_expression(&clause.then_result, out);
                }
                if let Some(else_val) = &c.else_value {
                    Self::collect_all_windows_in_expression(else_val, out);
                }
            }
            Expression::Between(b) => {
                Self::collect_all_windows_in_expression(&b.expr, out);
                Self::collect_all_windows_in_expression(&b.lower, out);
                Self::collect_all_windows_in_expression(&b.upper, out);
            }
            Expression::In(i) => {
                Self::collect_all_windows_in_expression(&i.left, out);
                Self::collect_all_windows_in_expression(&i.right, out);
            }
            Expression::Like(l) => {
                Self::collect_all_windows_in_expression(&l.left, out);
                Self::collect_all_windows_in_expression(&l.pattern, out);
                if let Some(esc) = &l.escape {
                    Self::collect_all_windows_in_expression(esc, out);
                }
            }
            Expression::List(l) => {
                for elem in &l.elements {
                    Self::collect_all_windows_in_expression(elem, out);
                }
            }
            Expression::ExpressionList(l) => {
                for elem in &l.expressions {
                    Self::collect_all_windows_in_expression(elem, out);
                }
            }
            _ => {}
        }
    }

    /// Replace window expressions in an expression tree with identifier references.
    /// Each Window node is replaced with a unique placeholder from `wf_names` in
    /// depth-first order (matching `collect_all_windows_in_expression`).
    fn replace_windows_with_identifiers(
        expr: &Expression,
        wf_names: &[String],
        counter: &mut usize,
    ) -> Expression {
        use crate::parser::token::{Position, Token, TokenType};

        match expr {
            Expression::Window(_) => {
                let name = wf_names
                    .get(*counter)
                    .map_or("__wf_unknown", |s| s.as_str());
                *counter += 1;
                let dummy_token = Token::new(TokenType::Identifier, name, Position::new(0, 0, 0));
                Expression::Identifier(Identifier::new(dummy_token, name.to_string()))
            }
            Expression::Infix(infix) => Expression::Infix(InfixExpression {
                token: infix.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &infix.left,
                    wf_names,
                    counter,
                )),
                operator: infix.operator.clone(),
                op_type: infix.op_type,
                right: Box::new(Self::replace_windows_with_identifiers(
                    &infix.right,
                    wf_names,
                    counter,
                )),
            }),
            Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression {
                token: prefix.token.clone(),
                operator: prefix.operator.clone(),
                op_type: prefix.op_type,
                right: Box::new(Self::replace_windows_with_identifiers(
                    &prefix.right,
                    wf_names,
                    counter,
                )),
            }),
            Expression::FunctionCall(f) => Expression::FunctionCall(Box::new(FunctionCall {
                token: f.token.clone(),
                function: f.function.clone(),
                arguments: f
                    .arguments
                    .iter()
                    .map(|a| Self::replace_windows_with_identifiers(a, wf_names, counter))
                    .collect(),
                is_distinct: f.is_distinct,
                order_by: f.order_by.clone(),
                filter: f.filter.clone(),
            })),
            Expression::Aliased(a) => Expression::Aliased(AliasedExpression {
                token: a.token.clone(),
                expression: Box::new(Self::replace_windows_with_identifiers(
                    &a.expression,
                    wf_names,
                    counter,
                )),
                alias: a.alias.clone(),
            }),
            Expression::Cast(cast) => Expression::Cast(CastExpression {
                token: cast.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &cast.expr, wf_names, counter,
                )),
                type_name: cast.type_name.clone(),
            }),
            Expression::Distinct(d) => Expression::Distinct(DistinctExpression {
                token: d.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &d.expr, wf_names, counter,
                )),
            }),
            Expression::Case(case) => {
                let new_value = case.value.as_ref().map(|v| {
                    Box::new(Self::replace_windows_with_identifiers(v, wf_names, counter))
                });
                let new_whens: Vec<WhenClause> = case
                    .when_clauses
                    .iter()
                    .map(|w| WhenClause {
                        token: w.token.clone(),
                        condition: Self::replace_windows_with_identifiers(
                            &w.condition,
                            wf_names,
                            counter,
                        ),
                        then_result: Self::replace_windows_with_identifiers(
                            &w.then_result,
                            wf_names,
                            counter,
                        ),
                    })
                    .collect();
                let new_else = case.else_value.as_ref().map(|e| {
                    Box::new(Self::replace_windows_with_identifiers(e, wf_names, counter))
                });
                Expression::Case(Box::new(CaseExpression {
                    token: case.token.clone(),
                    value: new_value,
                    when_clauses: new_whens,
                    else_value: new_else,
                }))
            }
            Expression::Between(b) => Expression::Between(BetweenExpression {
                token: b.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &b.expr, wf_names, counter,
                )),
                lower: Box::new(Self::replace_windows_with_identifiers(
                    &b.lower, wf_names, counter,
                )),
                upper: Box::new(Self::replace_windows_with_identifiers(
                    &b.upper, wf_names, counter,
                )),
                not: b.not,
            }),
            Expression::In(i) => Expression::In(InExpression {
                token: i.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &i.left, wf_names, counter,
                )),
                right: Box::new(Self::replace_windows_with_identifiers(
                    &i.right, wf_names, counter,
                )),
                not: i.not,
            }),
            Expression::Like(l) => Expression::Like(LikeExpression {
                token: l.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &l.left, wf_names, counter,
                )),
                pattern: Box::new(Self::replace_windows_with_identifiers(
                    &l.pattern, wf_names, counter,
                )),
                operator: l.operator.clone(),
                escape: l.escape.as_ref().map(|e| {
                    Box::new(Self::replace_windows_with_identifiers(e, wf_names, counter))
                }),
            }),
            Expression::List(l) => Expression::List(Box::new(ListExpression {
                token: l.token.clone(),
                elements: l
                    .elements
                    .iter()
                    .map(|e| Self::replace_windows_with_identifiers(e, wf_names, counter))
                    .collect(),
            })),
            Expression::ExpressionList(l) => Expression::ExpressionList(Box::new(ExpressionList {
                token: l.token.clone(),
                expressions: l
                    .expressions
                    .iter()
                    .map(|e| Self::replace_windows_with_identifiers(e, wf_names, counter))
                    .collect(),
            })),
            _ => expr.clone(),
        }
    }

    /// Extract window function info from WindowExpression
    fn extract_window_function_info(
        &self,
        window_expr: &WindowExpression,
        window_defs: &[WindowDefinition],
    ) -> Result<WindowFunctionInfo> {
        let func = &window_expr.function;

        // Resolve named window reference if present
        let (partition_by_exprs, order_by, frame) = if let Some(ref win_ref) =
            window_expr.window_ref
        {
            // Look up the named window definition
            let win_def = window_defs
                .iter()
                .find(|wd| wd.name.eq_ignore_ascii_case(win_ref))
                .ok_or_else(|| Error::NotSupported(format!("Unknown window name: {}", win_ref)))?;
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

        // Extract partition by column names (identifiers only, for fast-path)
        // For qualified identifiers (e.g., l.grp), keep the full qualified name
        // to properly match columns from JOINs
        let partition_by: Vec<String> = partition_by_exprs
            .iter()
            .filter_map(|e| match e {
                Expression::Identifier(id) => Some(id.value.to_string()),
                Expression::QualifiedIdentifier(qid) => {
                    Some(format!("{}.{}", qid.qualifier.value, qid.name.value))
                }
                _ => None,
            })
            .collect();

        // Build column name
        let column_name = format!("{}()", func.function);

        // OPTIMIZATION: func.function is already uppercase from parsing
        Ok(WindowFunctionInfo {
            name: func.function.to_string(),
            arguments: func.arguments.clone(),
            partition_by,
            partition_by_exprs: partition_by_exprs.to_vec(),
            order_by,
            frame,
            column_name,
            is_distinct: func.is_distinct,
        })
    }

    /// Create a cache key for ORDER BY expressions using their string representation.
    /// This enables semantic comparison (ignoring token positions) for ORDER BY clause deduplication.
    #[inline]
    fn order_by_cache_key(order_by: &[OrderByExpression]) -> String {
        use std::fmt::Write;
        let mut key = String::with_capacity(64);
        for (i, ob) in order_by.iter().enumerate() {
            if i > 0 {
                key.push(',');
            }
            // Use Display trait to get semantic string representation
            let _ = write!(key, "{}", ob);
        }
        key
    }

    /// Check that ALL window functions share the exact same PARTITION BY clause.
    /// Returns false if any window function is unpartitioned (global) or uses a
    /// different partition scheme, because the streaming path builds only one
    /// partition map and computes per-partition subsets.
    fn all_partitions_match(window_functions: &[WindowFunctionInfo]) -> bool {
        use std::fmt::Write;
        let mut canonical: Option<String> = None;
        for wf in window_functions {
            // A global (unpartitioned) window must see all rows, so
            // per-partition streaming is unsafe.
            if wf.partition_by_exprs.is_empty() {
                return false;
            }
            let mut key = String::with_capacity(64);
            for (i, expr) in wf.partition_by_exprs.iter().enumerate() {
                if i > 0 {
                    key.push(',');
                }
                let _ = write!(key, "{}", expr);
            }
            match &canonical {
                None => canonical = Some(key),
                Some(c) if c != &key => return false,
                _ => {}
            }
        }
        canonical.is_some()
    }

    /// Build partition map from PARTITION BY expressions.
    /// Handles both simple column references (fast path) and complex expressions like
    /// function calls (e.g., TIME_TRUNC('30m', timestamp)) via MultiExpressionEval.
    fn build_partition_map(
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
    ) -> Result<FxHashMap<PartitionKey, Vec<usize>>> {
        let mut partitions: FxHashMap<PartitionKey, Vec<usize>> = FxHashMap::default();

        // Check if any PARTITION BY expression is complex (not a simple column reference)
        let has_complex_expr = wf_info.partition_by_exprs.iter().any(|e| {
            !matches!(
                e,
                Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
            )
        });

        if has_complex_expr && !wf_info.partition_by_exprs.is_empty() {
            // Complex path: use MultiExpressionEval to evaluate expressions per row
            let agg_aliases: Vec<(String, usize)> =
                col_index_map.iter().map(|(k, v)| (k.clone(), *v)).collect();

            let eval = MultiExpressionEval::compile_with_aliases(
                &wf_info.partition_by_exprs,
                columns,
                &agg_aliases,
            )
            .map_err(|e| {
                Error::internal(format!("Failed to compile PARTITION BY expression: {}", e))
            })?;
            let mut eval = eval.with_context(ctx);
            for (i, (_, row)) in rows.iter().enumerate() {
                let values = eval.eval_all(row).map_err(|e| {
                    Error::internal(format!("Failed to evaluate PARTITION BY expression: {}", e))
                })?;
                let key: PartitionKey = SmallVec::from_vec(values);
                partitions.entry(key).or_default().push(i);
            }
        } else {
            // Fast path: simple column references
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

            for (i, (_, row)) in rows.iter().enumerate() {
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
        }

        Ok(partitions)
    }

    /// Compute window function values for all rows
    /// pre_sorted: Optional info about whether rows are already sorted by an indexed column
    /// pre_grouped: Optional pre-grouped partitions from index (avoids hash-based grouping)
    /// order_by_cache: Precomputed ORDER BY values cache (keyed by semantic string representation)
    #[allow(clippy::too_many_arguments)]
    fn compute_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
        pre_grouped: Option<&WindowPreGroupedState>,
        order_by_cache: &[(String, ColumnarOrderByValues)],
    ) -> Result<Vec<Value>> {
        // Check if this is an aggregate function used as window function
        let is_aggregate = self.function_registry.is_aggregate(&wf_info.name);

        if is_aggregate {
            // Handle aggregate functions as window functions (SUM, COUNT, AVG, etc.)
            // Look up precomputed ORDER BY values from cache using semantic key
            let cache_key = Self::order_by_cache_key(&wf_info.order_by);
            let cached_order_by = order_by_cache
                .iter()
                .find(|(key, _)| key == &cache_key)
                .map(|(_, v)| v);
            return self.compute_aggregate_window_function(
                wf_info,
                rows,
                columns,
                col_index_map,
                ctx,
                pre_sorted,
                cached_order_by,
            );
        }

        // Get the window function from registry
        let window_func = self
            .function_registry
            .get_window(&wf_info.name)
            .ok_or_else(|| {
                Error::NotSupported(format!("Unknown window function: {}", wf_info.name))
            })?;

        // Look up precomputed ORDER BY values from cache using semantic key
        let precomputed_order_by: Option<&ColumnarOrderByValues> = if !wf_info.order_by.is_empty() {
            let cache_key = Self::order_by_cache_key(&wf_info.order_by);
            order_by_cache
                .iter()
                .find(|(key, _)| key == &cache_key)
                .map(|(_, v)| v)
        } else {
            None
        };

        // If there's no partitioning, treat all rows as one partition
        if wf_info.partition_by_exprs.is_empty() {
            // Pre-allocate results array and use direct-writing variant
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];
            let row_indices: Vec<usize> = (0..rows.len()).collect();

            self.compute_window_for_partition_direct(
                &*window_func,
                wf_info,
                rows,
                row_indices,
                precomputed_order_by,
                columns,
                col_index_map,
                ctx,
                &mut results,
            )?;

            return Ok(results);
        }

        // Group rows by partition key
        // OPTIMIZATION: Use pre-grouped partitions from index if available (avoids O(n) hashing)
        // Only valid when this WF partitions by the exact same single simple column that
        // the planner used to build the pre-grouped map.
        let partitions: FxHashMap<PartitionKey, Vec<usize>> = if let Some(pg) =
            pre_grouped.filter(|pg| {
                wf_info.partition_by.len() == 1
                    && wf_info.partition_by.len() == wf_info.partition_by_exprs.len()
                    && wf_info.partition_by[0].to_lowercase() == pg.partition_column
            }) {
            pg.partition_map.clone()
        } else {
            Self::build_partition_map(wf_info, rows, columns, col_index_map, ctx)?
        };

        // Compute window function for each partition
        // Use parallel execution for large number of partitions
        let partition_count = partitions.len();
        let use_parallel = partition_count >= 10 && rows.len() >= 1000;

        if use_parallel {
            // Parallel execution: process partitions concurrently
            // MEMORY OPTIMIZATION: Use ParallelVec for direct writes to disjoint indices.
            // This eliminates:
            //   1. Per-partition Vec<Value> allocation from compute_window_for_partition
            //   2. The final .collect() mapping step
            // Each partition writes to its own row indices (guaranteed disjoint).
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];
            let parallel_results = ParallelVec::new(&mut results);

            let partitions_vec: Vec<_> = partitions.into_iter().collect();
            #[cfg(feature = "parallel")]
            let iter_result =
                partitions_vec
                    .par_iter()
                    .try_for_each(|(_key, row_indices)| -> Result<()> {
                        // Direct writing: each partition writes to its own indices
                        self.compute_window_for_partition_parallel(
                            &*window_func,
                            wf_info,
                            rows,
                            row_indices.clone(),
                            precomputed_order_by,
                            columns,
                            col_index_map,
                            ctx,
                            &parallel_results,
                        )
                    });
            #[cfg(not(feature = "parallel"))]
            let iter_result =
                partitions_vec
                    .iter()
                    .try_for_each(|(_key, row_indices)| -> Result<()> {
                        // Direct writing: each partition writes to its own indices
                        self.compute_window_for_partition_parallel(
                            &*window_func,
                            wf_info,
                            rows,
                            row_indices.clone(),
                            precomputed_order_by,
                            columns,
                            col_index_map,
                            ctx,
                            &parallel_results,
                        )
                    });
            iter_result?;

            // ParallelVec is done, results are written in place
            Ok(results)
        } else {
            // Sequential execution for small partition counts
            // MEMORY OPTIMIZATION: Write directly to final results array instead of
            // creating per-partition Vec<Value> and then mapping back
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];

            for (_key, row_indices) in partitions {
                // MEMORY OPTIMIZATION: Direct writing variant - writes to results in-place
                self.compute_window_for_partition_direct(
                    &*window_func,
                    wf_info,
                    rows,
                    row_indices,
                    precomputed_order_by,
                    columns,
                    col_index_map,
                    ctx,
                    &mut results,
                )?;
            }

            Ok(results)
        }
    }

    /// Compute window function for a single partition
    /// Returns (results in sorted order, sorted row indices) to avoid re-sorting in the caller
    /// precomputed_order_by: Optional precomputed ORDER BY values for ALL rows (avoids recomputation)
    /// skip_sorting: If true, skip sorting (rows are already pre-sorted by index)
    ///
    /// MEMORY OPTIMIZATION: This function uses index-based value access instead of cloning
    /// values into intermediate Vec<Value> collections. For LEAD/LAG/FIRST_VALUE/LAST_VALUE,
    /// values are accessed directly from all_rows via indices. For RANK/DENSE_RANK, we use
    /// ColumnarOrderByValues::rows_equal() to compare ORDER BY values without cloning.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        skip_sorting: bool,
    ) -> Result<(Vec<Value>, Vec<usize>)> {
        // Suppress unused variable warnings - these are needed for compute_lead_lag, compute_ntile, etc.
        let _ = columns;

        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

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

        // MEMORY OPTIMIZATION: Compute rank info directly from ColumnarOrderByValues
        // instead of cloning ORDER BY values into a Vec<Value>
        // This uses rows_equal() for O(n) comparison without allocating order_values
        let is_rank_function = matches!(wf_info.name.as_str(), "RANK" | "DENSE_RANK");
        let is_rank = wf_info.name == "RANK";
        let rank_info = if is_rank_function && !order_by_values.is_empty() {
            Self::precompute_rank_info_columnar(&row_indices, order_by_values)
        } else {
            vec![]
        };

        // Compute window function for each row in the partition
        let mut results = Vec::with_capacity(row_indices.len());
        let partition_len = row_indices.len();

        // Precompute peer group ends for RANGE frame semantics
        let peer_group_ends = if !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::precompute_peer_group_ends(&row_indices, order_by_values)
        } else {
            vec![partition_len; partition_len]
        };

        for (i, &row_idx) in row_indices.iter().enumerate() {
            // Handle special functions
            let value = match wf_info.name.as_str() {
                // MEMORY OPTIMIZATION: Access values directly from all_rows via indices
                // instead of cloning into partition_values Vec
                "LEAD" | "LAG" => self.compute_lead_lag_indexed(
                    wf_info,
                    all_rows,
                    &row_indices,
                    arg_col_idx,
                    i,
                    &all_rows[row_idx].1,
                    columns,
                    ctx,
                )?,
                "NTILE" => self.compute_ntile(wf_info, row_indices.len(), i, ctx)?,
                "RANK" | "DENSE_RANK" => Self::compute_rank_fast(is_rank, &rank_info, i),
                "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE" => {
                    // Compute frame bounds for navigation functions
                    let peer_end = if i < peer_group_ends.len() {
                        peer_group_ends[i]
                    } else {
                        partition_len
                    };
                    let (frame_start, frame_end) =
                        self.compute_simple_frame_bounds(wf_info, i, partition_len, peer_end);

                    // MEMORY OPTIMIZATION: Access values directly via indices
                    match wf_info.name.as_str() {
                        "FIRST_VALUE" => self.compute_first_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "LAST_VALUE" => self.compute_last_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "NTH_VALUE" => self.compute_nth_value_indexed(
                            wf_info,
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                            ctx,
                        )?,
                        _ => unreachable!(),
                    }
                }
                "PERCENT_RANK" => {
                    self.compute_percent_rank_columnar(order_by_values, &row_indices, i)?
                }
                "CUME_DIST" => self.compute_cume_dist_columnar(order_by_values, &row_indices, i)?,
                _ => {
                    // ROW_NUMBER and other simple functions - no values needed
                    window_func.process(&[], &[], i)?
                }
            };
            results.push(value);
        }

        Ok((results, row_indices))
    }

    /// Direct-writing variant of compute_window_for_partition
    /// Writes results directly to the provided results array, avoiding per-partition Vec allocation
    ///
    /// MEMORY OPTIMIZATION: Instead of returning (Vec<Value>, Vec<usize>) and mapping back,
    /// this writes directly to results[orig_idx] for each computed value.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition_direct(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        results: &mut [Value],
    ) -> Result<()> {
        let _ = columns;

        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

        // Sort partition by ORDER BY if specified
        if !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        // Pre-compute column index for function argument
        let arg_col_idx: Option<usize> = if !wf_info.arguments.is_empty() {
            self.extract_column_from_arg(&wf_info.arguments[0])
                .and_then(|col_name| col_index_map.get(&col_name.to_lowercase()).copied())
        } else {
            None
        };

        // Compute rank info for RANK/DENSE_RANK
        let is_rank_function = matches!(wf_info.name.as_str(), "RANK" | "DENSE_RANK");
        let is_rank = wf_info.name == "RANK";
        let rank_info = if is_rank_function && !order_by_values.is_empty() {
            Self::precompute_rank_info_columnar(&row_indices, order_by_values)
        } else {
            vec![]
        };

        let partition_len = row_indices.len();

        // Precompute peer group ends for RANGE frame semantics
        let peer_group_ends = if !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::precompute_peer_group_ends(&row_indices, order_by_values)
        } else {
            vec![partition_len; partition_len]
        };

        // Compute and write directly to results array
        for (i, &row_idx) in row_indices.iter().enumerate() {
            let value = match wf_info.name.as_str() {
                "LEAD" | "LAG" => self.compute_lead_lag_indexed(
                    wf_info,
                    all_rows,
                    &row_indices,
                    arg_col_idx,
                    i,
                    &all_rows[row_idx].1,
                    columns,
                    ctx,
                )?,
                "NTILE" => self.compute_ntile(wf_info, partition_len, i, ctx)?,
                "RANK" | "DENSE_RANK" => Self::compute_rank_fast(is_rank, &rank_info, i),
                "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE" => {
                    let peer_end = if i < peer_group_ends.len() {
                        peer_group_ends[i]
                    } else {
                        partition_len
                    };
                    let (frame_start, frame_end) =
                        self.compute_simple_frame_bounds(wf_info, i, partition_len, peer_end);
                    match wf_info.name.as_str() {
                        "FIRST_VALUE" => self.compute_first_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "LAST_VALUE" => self.compute_last_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "NTH_VALUE" => self.compute_nth_value_indexed(
                            wf_info,
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                            ctx,
                        )?,
                        _ => unreachable!(),
                    }
                }
                "PERCENT_RANK" => {
                    self.compute_percent_rank_columnar(order_by_values, &row_indices, i)?
                }
                "CUME_DIST" => self.compute_cume_dist_columnar(order_by_values, &row_indices, i)?,
                _ => {
                    // ROW_NUMBER and other simple functions - no values needed
                    window_func.process(&[], &[], i)?
                }
            };
            // DIRECT WRITE: Write to original row position, avoiding the mapping step
            results[row_idx] = value;
        }

        Ok(())
    }

    /// Parallel-safe variant of compute_window_for_partition_direct.
    /// Uses ParallelVec for safe writes to disjoint indices from multiple threads.
    ///
    /// MEMORY OPTIMIZATION: Eliminates per-partition Vec<Value> allocation and the
    /// final mapping/collect step in parallel execution path.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition_parallel(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        results: &ParallelVec,
    ) -> Result<()> {
        let _ = columns;

        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

        // Sort partition by ORDER BY if specified
        if !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        // Pre-compute column index for function argument
        let arg_col_idx: Option<usize> = if !wf_info.arguments.is_empty() {
            self.extract_column_from_arg(&wf_info.arguments[0])
                .and_then(|col_name| col_index_map.get(&col_name.to_lowercase()).copied())
        } else {
            None
        };

        // Compute rank info for RANK/DENSE_RANK
        let is_rank_function = matches!(wf_info.name.as_str(), "RANK" | "DENSE_RANK");
        let is_rank = wf_info.name == "RANK";
        let rank_info = if is_rank_function && !order_by_values.is_empty() {
            Self::precompute_rank_info_columnar(&row_indices, order_by_values)
        } else {
            vec![]
        };

        let partition_len = row_indices.len();

        // Precompute peer group ends for RANGE frame semantics
        let peer_group_ends = if !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::precompute_peer_group_ends(&row_indices, order_by_values)
        } else {
            vec![partition_len; partition_len]
        };

        // Compute and write directly using ParallelVec
        for (i, &row_idx) in row_indices.iter().enumerate() {
            let value = match wf_info.name.as_str() {
                "LEAD" | "LAG" => self.compute_lead_lag_indexed(
                    wf_info,
                    all_rows,
                    &row_indices,
                    arg_col_idx,
                    i,
                    &all_rows[row_idx].1,
                    columns,
                    ctx,
                )?,
                "NTILE" => self.compute_ntile(wf_info, partition_len, i, ctx)?,
                "RANK" | "DENSE_RANK" => Self::compute_rank_fast(is_rank, &rank_info, i),
                "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE" => {
                    let peer_end = if i < peer_group_ends.len() {
                        peer_group_ends[i]
                    } else {
                        partition_len
                    };
                    let (frame_start, frame_end) =
                        self.compute_simple_frame_bounds(wf_info, i, partition_len, peer_end);
                    match wf_info.name.as_str() {
                        "FIRST_VALUE" => self.compute_first_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "LAST_VALUE" => self.compute_last_value_indexed(
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                        )?,
                        "NTH_VALUE" => self.compute_nth_value_indexed(
                            wf_info,
                            all_rows,
                            &row_indices,
                            arg_col_idx,
                            frame_start,
                            frame_end,
                            ctx,
                        )?,
                        _ => unreachable!(),
                    }
                }
                "PERCENT_RANK" => {
                    self.compute_percent_rank_columnar(order_by_values, &row_indices, i)?
                }
                "CUME_DIST" => self.compute_cume_dist_columnar(order_by_values, &row_indices, i)?,
                _ => {
                    // ROW_NUMBER and other simple functions - no values needed
                    window_func.process(&[], &[], i)?
                }
            };
            // DIRECT WRITE: Write to original row position using ParallelVec
            results.write(row_idx, value);
        }

        Ok(())
    }

    /// Precompute rank information using ColumnarOrderByValues directly (no cloning)
    ///
    /// Returns a vector of (group_start, dense_rank) for each position in sorted_indices.
    /// Uses rows_equal() for O(n) comparison without allocating intermediate Vec<Value>.
    #[inline]
    fn precompute_rank_info_columnar(
        sorted_indices: &[usize],
        order_by: &ColumnarOrderByValues,
    ) -> Vec<(usize, i64)> {
        let n = sorted_indices.len();
        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        // First row: group starts at 0, dense_rank = 1
        result.push((0, 1));

        let mut current_group_start = 0;
        let mut current_dense_rank: i64 = 1;

        for i in 1..n {
            // Compare ORDER BY values of adjacent sorted rows without cloning
            if !order_by.rows_equal(sorted_indices[i - 1], sorted_indices[i]) {
                // New group starts
                current_group_start = i;
                current_dense_rank += 1;
            }
            result.push((current_group_start, current_dense_rank));
        }

        result
    }

    /// Precompute peer group end positions for RANGE frame semantics.
    /// Returns a vec where `result[i]` is the exclusive end of position i's peer group.
    fn precompute_peer_group_ends(
        sorted_indices: &[usize],
        order_by: &ColumnarOrderByValues,
    ) -> Vec<usize> {
        let n = sorted_indices.len();
        if n == 0 {
            return vec![];
        }
        let mut ends = vec![n; n];
        let mut group_start = 0;
        for i in 1..n {
            if !order_by.rows_equal(sorted_indices[i - 1], sorted_indices[i]) {
                for end in ends.iter_mut().take(i).skip(group_start) {
                    *end = i;
                }
                group_start = i;
            }
        }
        // Last group already has ends[j] = n from initialization
        ends
    }

    /// Compute LEAD or LAG using index-based access (no cloning)
    #[allow(clippy::too_many_arguments)]
    fn compute_lead_lag_indexed(
        &self,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        arg_col_idx: Option<usize>,
        current_pos: usize,
        current_row_data: &Row,
        columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        // Get offset (default 1)
        let offset = if wf_info.arguments.len() > 1 {
            let mut eval = ExpressionEval::compile(&wf_info.arguments[1], &[])?.with_context(ctx);
            match eval.eval_slice(&Row::new())? {
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

        // Calculate target position in sorted order
        let target_pos = if wf_info.name == "LEAD" {
            current_pos.checked_add(offset)
        } else {
            // LAG
            current_pos.checked_sub(offset)
        };

        match target_pos {
            Some(pos) if pos < sorted_indices.len() => {
                // Access value directly from all_rows via index
                let target_row_idx = sorted_indices[pos];
                if let Some(col_idx) = arg_col_idx {
                    Ok(all_rows[target_row_idx]
                        .1
                        .get(col_idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown))
                } else {
                    Ok(default_value)
                }
            }
            _ => Ok(default_value),
        }
    }

    /// Compute FIRST_VALUE using index-based access (no cloning)
    fn compute_first_value_indexed(
        &self,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        arg_col_idx: Option<usize>,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<Value> {
        if frame_start >= frame_end || frame_start >= sorted_indices.len() {
            return Ok(Value::null_unknown());
        }
        let row_idx = sorted_indices[frame_start];
        if let Some(col_idx) = arg_col_idx {
            Ok(all_rows[row_idx]
                .1
                .get(col_idx)
                .cloned()
                .unwrap_or_else(Value::null_unknown))
        } else {
            Ok(Value::null_unknown())
        }
    }

    /// Compute LAST_VALUE using index-based access (no cloning)
    fn compute_last_value_indexed(
        &self,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        arg_col_idx: Option<usize>,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<Value> {
        if frame_start >= frame_end || frame_end > sorted_indices.len() {
            return Ok(Value::null_unknown());
        }
        let row_idx = sorted_indices[frame_end - 1];
        if let Some(col_idx) = arg_col_idx {
            Ok(all_rows[row_idx]
                .1
                .get(col_idx)
                .cloned()
                .unwrap_or_else(Value::null_unknown))
        } else {
            Ok(Value::null_unknown())
        }
    }

    /// Compute NTH_VALUE using index-based access (no cloning)
    #[allow(clippy::too_many_arguments)]
    fn compute_nth_value_indexed(
        &self,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        arg_col_idx: Option<usize>,
        frame_start: usize,
        frame_end: usize,
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        if frame_start >= frame_end {
            return Ok(Value::null_unknown());
        }

        // Get n (1-indexed position) from second argument
        let n = if wf_info.arguments.len() > 1 {
            let mut eval = ExpressionEval::compile(&wf_info.arguments[1], &[])?.with_context(ctx);
            match eval.eval_slice(&Row::new())? {
                Value::Integer(n) if n > 0 => n as usize,
                _ => return Ok(Value::null_unknown()),
            }
        } else {
            return Ok(Value::null_unknown());
        };

        // n is 1-indexed within the frame
        let frame_len = frame_end - frame_start;
        if n > frame_len {
            return Ok(Value::null_unknown());
        }

        let pos_in_frame = n - 1;
        let row_idx = sorted_indices[frame_start + pos_in_frame];
        if let Some(col_idx) = arg_col_idx {
            Ok(all_rows[row_idx]
                .1
                .get(col_idx)
                .cloned()
                .unwrap_or_else(Value::null_unknown))
        } else {
            Ok(Value::null_unknown())
        }
    }

    /// Compute PERCENT_RANK using ColumnarOrderByValues (no cloning)
    /// PERCENT_RANK = (rank - 1) / (total_rows - 1)
    fn compute_percent_rank_columnar(
        &self,
        order_by: &ColumnarOrderByValues,
        sorted_indices: &[usize],
        current_pos: usize,
    ) -> Result<Value> {
        let n = sorted_indices.len();
        if n <= 1 {
            return Ok(Value::Float(0.0));
        }

        // Find rank by counting how many preceding rows have different ORDER BY value
        let mut rank = 1usize;
        for i in 0..current_pos {
            if i == 0 || !order_by.rows_equal(sorted_indices[i - 1], sorted_indices[i]) {
                // New group - this row gets a new rank
                if i < current_pos
                    && !order_by.rows_equal(sorted_indices[i], sorted_indices[current_pos])
                {
                    rank = i + 1;
                }
            }
        }
        // Find actual rank of current row (position of first row with same ORDER BY value)
        for i in 0..=current_pos {
            if order_by.rows_equal(sorted_indices[i], sorted_indices[current_pos]) {
                rank = i + 1;
                break;
            }
        }

        Ok(Value::Float((rank - 1) as f64 / (n - 1) as f64))
    }

    /// Compute CUME_DIST using ColumnarOrderByValues (no cloning)
    /// CUME_DIST = (number of rows with value <= current) / total_rows
    fn compute_cume_dist_columnar(
        &self,
        order_by: &ColumnarOrderByValues,
        sorted_indices: &[usize],
        current_pos: usize,
    ) -> Result<Value> {
        let n = sorted_indices.len();
        if n == 0 {
            return Ok(Value::Float(1.0));
        }

        // Find how many rows have ORDER BY value <= current (i.e., last row with same value)
        let mut count = current_pos + 1;
        for i in (current_pos + 1)..n {
            if order_by.rows_equal(sorted_indices[current_pos], sorted_indices[i]) {
                count = i + 1;
            } else {
                break;
            }
        }

        Ok(Value::Float(count as f64 / n as f64))
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
            match eval.eval_slice(&Row::new())? {
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

    /// Compute RANK or DENSE_RANK function using precomputed rank info.
    ///
    /// This is an O(1) lookup using the precomputed (group_start, dense_rank) tuple.
    #[inline]
    fn compute_rank_fast(is_rank: bool, rank_info: &[(usize, i64)], current_row: usize) -> Value {
        if rank_info.is_empty() || current_row >= rank_info.len() {
            return Value::Integer(1);
        }

        let (group_start, dense_rank) = rank_info[current_row];

        if is_rank {
            // RANK: 1-indexed position of first row in group
            Value::Integer((group_start + 1) as i64)
        } else {
            // DENSE_RANK: sequential group number
            Value::Integer(dense_rank)
        }
    }

    /// Compute frame bounds for navigation functions (FIRST_VALUE, LAST_VALUE, NTH_VALUE).
    /// `peer_group_end` is the exclusive end of the current row's peer group (for RANGE semantics).
    /// Returns (start, end) where end is exclusive.
    fn compute_simple_frame_bounds(
        &self,
        wf_info: &WindowFunctionInfo,
        current_row: usize,
        partition_len: usize,
        peer_group_end: usize,
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
                // Has ORDER BY - default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                // SQL standard: peer rows (same ORDER BY value) share the same frame end
                (0, peer_group_end)
            }
        }
    }

    /// Extract column name from expression
    fn extract_column_from_expr(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value.to_string()),
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
        col_index_map: &StringMap<usize>,
    ) -> Option<usize> {
        match expr {
            Expression::Identifier(id) => col_index_map.get(id.value_lower.as_str()).copied(),
            Expression::QualifiedIdentifier(qid) => {
                // Try fully qualified name first (e.g., "s.qty")
                let qualified =
                    format!("{}.{}", qid.qualifier.value, qid.name.value).to_lowercase();
                col_index_map
                    .get(&qualified)
                    // Then try unqualified (e.g., "qty") for CTE/subquery cases
                    .or_else(|| col_index_map.get(qid.name.value_lower.as_str()))
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

    /// Precompute ORDER BY values for all rows using columnar layout
    /// Returns a ColumnarOrderByValues structure for cache-efficient sorting
    fn precompute_order_by_values(
        &self,
        order_by: &[OrderByExpression],
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
    ) -> ColumnarOrderByValues {
        let num_rows = rows.len();
        let num_cols = order_by.len();

        if num_cols == 0 || num_rows == 0 {
            return ColumnarOrderByValues {
                columns: vec![],
                ascending: vec![],
                num_rows: 0,
            };
        }

        // Extract ascending flags once (not per row!)
        let ascending_flags: Vec<bool> = order_by.iter().map(|ob| ob.ascending).collect();

        // Check if any ORDER BY expression is complex (not a simple column reference)
        let has_complex_expr = order_by.iter().any(|ob| {
            !matches!(
                &ob.expression,
                Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
            )
        });

        // Pre-allocate columns with exact capacity
        let mut result_columns: Vec<Vec<Value>> = (0..num_cols)
            .map(|_| Vec::with_capacity(num_rows))
            .collect();

        if has_complex_expr {
            // Build aliases from col_index_map (e.g., "sum(val)" -> column_index)
            // This allows ORDER BY SUM(val) to resolve to the correct column
            let agg_aliases: Vec<(String, usize)> =
                col_index_map.iter().map(|(k, v)| (k.clone(), *v)).collect();

            // Extract order_by expressions
            let order_exprs: Vec<Expression> =
                order_by.iter().map(|ob| ob.expression.clone()).collect();

            // Compile all expressions with aliases
            match MultiExpressionEval::compile_with_aliases(&order_exprs, columns, &agg_aliases) {
                Ok(eval) => {
                    let mut eval = eval.with_context(ctx);
                    for (_, row) in rows {
                        match eval.eval_all(row) {
                            Ok(values) => {
                                for (col_idx, value) in values.into_iter().enumerate() {
                                    if col_idx < result_columns.len() {
                                        result_columns[col_idx].push(value);
                                    }
                                }
                            }
                            Err(_) => {
                                for col in result_columns.iter_mut() {
                                    col.push(Value::null_unknown());
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Compilation failed - fill with nulls
                    for _ in 0..num_rows {
                        for col in result_columns.iter_mut() {
                            col.push(Value::null_unknown());
                        }
                    }
                }
            }
        } else {
            // Fast path: simple column references
            let order_by_indices: Vec<Option<usize>> = order_by
                .iter()
                .map(|ob| match &ob.expression {
                    Expression::Identifier(id) => {
                        col_index_map.get(id.value_lower.as_str()).copied()
                    }
                    Expression::QualifiedIdentifier(qid) => {
                        let qualified =
                            format!("{}.{}", qid.qualifier.value, qid.name.value).to_lowercase();
                        col_index_map
                            .get(&qualified)
                            .or_else(|| col_index_map.get(qid.name.value_lower.as_str()))
                            .copied()
                    }
                    _ => None,
                })
                .collect();

            // Extract values column by column for better cache locality
            for (col_idx, idx_opt) in order_by_indices.iter().enumerate() {
                let col = &mut result_columns[col_idx];
                match idx_opt {
                    Some(src_idx) => {
                        for (_, row) in rows {
                            col.push(
                                row.get(*src_idx)
                                    .cloned()
                                    .unwrap_or_else(Value::null_unknown),
                            );
                        }
                    }
                    None => {
                        for _ in 0..num_rows {
                            col.push(Value::null_unknown());
                        }
                    }
                }
            }
        }

        ColumnarOrderByValues {
            columns: result_columns,
            ascending: ascending_flags,
            num_rows,
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

    /// Sort row indices using precomputed ORDER BY values (columnar layout)
    fn sort_by_order_values(row_indices: &mut [usize], order_by_values: &ColumnarOrderByValues) {
        if row_indices.len() < 2 || order_by_values.is_empty() {
            return;
        }

        // Check if single ORDER BY column (most common case) - use fast path
        let is_single_order_by = order_by_values.num_columns() == 1;

        if is_single_order_by {
            // Fast path: single ORDER BY column with type-specific comparison
            let ascending = order_by_values.is_ascending(0);

            // Get direct reference to the column for cache-efficient access
            let col = &order_by_values.columns[0];

            // Detect column type from first non-null value (single pass)
            #[derive(Clone, Copy, PartialEq)]
            enum DetectedType {
                Unknown,
                Integer,
                Float,
                Mixed,
            }

            let mut detected = DetectedType::Unknown;
            for &idx in row_indices.iter().take(100) {
                if let Some(val) = col.get(idx) {
                    match val {
                        Value::Integer(_) => {
                            if detected == DetectedType::Unknown {
                                detected = DetectedType::Integer;
                            } else if detected != DetectedType::Integer {
                                detected = DetectedType::Mixed;
                                break;
                            }
                        }
                        Value::Float(_) => {
                            if detected == DetectedType::Unknown {
                                detected = DetectedType::Float;
                            } else if detected != DetectedType::Float {
                                detected = DetectedType::Mixed;
                                break;
                            }
                        }
                        Value::Null(_) => {
                            // Skip nulls, they're handled by unwrap_or in sort functions
                        }
                        _ => {
                            detected = DetectedType::Mixed;
                            break;
                        }
                    }
                }
            }

            match detected {
                DetectedType::Integer => {
                    Self::sort_by_integer_key_columnar(row_indices, col, ascending);
                    return;
                }
                DetectedType::Float => {
                    Self::sort_by_float_key_columnar(row_indices, col, ascending);
                    return;
                }
                _ => {
                    // Mixed types or unknown - use generic single column sort
                    Self::sort_single_column_columnar(row_indices, col, ascending);
                    return;
                }
            }
        }

        // Multi-column ORDER BY: use parallel sort for large partitions
        const PARALLEL_THRESHOLD: usize = 10_000;

        if row_indices.len() >= PARALLEL_THRESHOLD {
            #[cfg(feature = "parallel")]
            row_indices.par_sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
            #[cfg(not(feature = "parallel"))]
            row_indices.sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
        } else {
            row_indices.sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
        }
    }

    /// Compare two rows by their ORDER BY values (columnar layout)
    #[inline]
    fn compare_order_values_columnar(
        order_by_values: &ColumnarOrderByValues,
        a: usize,
        b: usize,
    ) -> Ordering {
        for col_idx in 0..order_by_values.num_columns() {
            let a_val = order_by_values.get(a, col_idx);
            let b_val = order_by_values.get(b, col_idx);

            let cmp = match (a_val, b_val) {
                (Some(av), Some(bv)) => Self::compare_values_fast(av, bv),
                (None, None) => Ordering::Equal,
                (None, _) => Ordering::Greater,
                (_, None) => Ordering::Less,
            };

            let cmp = if !order_by_values.is_ascending(col_idx) {
                cmp.reverse()
            } else {
                cmp
            };

            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Fast value comparison with type-specific paths
    #[inline]
    fn compare_values_fast(a: &Value, b: &Value) -> Ordering {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Integer(x), Value::Float(y)) => {
                (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)
            }
            (Value::Float(x), Value::Integer(y)) => {
                x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)
            }
            (Value::Null(_), Value::Null(_)) => Ordering::Equal,
            (Value::Null(_), _) => Ordering::Greater, // NULLs last
            (_, Value::Null(_)) => Ordering::Less,
            _ => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        }
    }

    /// Ultra-fast sort for integer ORDER BY column (columnar layout)
    fn sort_by_integer_key_columnar(row_indices: &mut [usize], col: &[Value], ascending: bool) {
        const PARALLEL_THRESHOLD: usize = 50_000;

        // OPTIMIZATION: Pre-extract sort keys to avoid bounds checking in the comparison hot path.
        // The sort comparator is called O(n log n) times, and each .get() does bounds checking.
        // By pre-extracting into a Vec<i64>, we pay O(k) allocation cost but get O(1) unchecked
        // access during sorting, which is a significant win for large partitions.
        //
        // We build a parallel array of (original_position, sort_key) and sort by key,
        // then extract the sorted positions back to row_indices.
        let n = row_indices.len();
        if n < 2 {
            return;
        }

        // Pre-extract sort keys: O(k) where k = partition size
        let mut keyed: Vec<(usize, i64)> = Vec::with_capacity(n);
        for &idx in row_indices.iter() {
            let key = match col.get(idx) {
                Some(Value::Integer(i)) => *i,
                _ => i64::MAX, // NULLs sort last
            };
            keyed.push((idx, key));
        }

        // Sort by pre-extracted key - no bounds checking in comparison!
        if n >= PARALLEL_THRESHOLD {
            if ascending {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| key);
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| key);
            } else {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
            }
        } else if ascending {
            keyed.sort_unstable_by_key(|&(_, key)| key);
        } else {
            keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
        }

        // Write sorted indices back
        for (i, (idx, _)) in keyed.into_iter().enumerate() {
            row_indices[i] = idx;
        }
    }

    /// Fast sort for float ORDER BY column (columnar layout)
    fn sort_by_float_key_columnar(row_indices: &mut [usize], col: &[Value], ascending: bool) {
        const PARALLEL_THRESHOLD: usize = 50_000;

        // OPTIMIZATION: Pre-extract sort keys to avoid bounds checking in the comparison hot path.
        // The sort comparator is called O(n log n) times, and each .get() does bounds checking.
        // By pre-extracting into a Vec<f64>, we pay O(k) allocation cost but get O(1) unchecked
        // access during sorting. For floats, we also use to_bits() to enable faster integer
        // comparison via sort_by_key instead of sort_by with a closure.
        let n = row_indices.len();
        if n < 2 {
            return;
        }

        // Pre-extract sort keys as OrderedFloat bits for fast integer comparison
        // Using to_bits() allows sort_by_key which is faster than sort_by with closure
        let mut keyed: Vec<(usize, u64)> = Vec::with_capacity(n);
        for &idx in row_indices.iter() {
            let f = match col.get(idx) {
                Some(Value::Float(f)) => *f,
                Some(Value::Integer(i)) => *i as f64,
                _ => f64::MAX, // NULLs sort last
            };
            // Convert to sortable integer representation (handles NaN, -0.0, negative numbers)
            let bits = if f.is_nan() {
                u64::MAX // NaN sorts last
            } else {
                let b = f.to_bits();
                // Flip sign bit and conditionally flip all bits for proper ordering
                if (b & (1u64 << 63)) != 0 {
                    !b // Negative: flip all bits
                } else {
                    b | (1u64 << 63) // Positive: set sign bit
                }
            };
            keyed.push((idx, bits));
        }

        // Sort by pre-extracted key - no bounds checking, pure integer comparison!
        if n >= PARALLEL_THRESHOLD {
            if ascending {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| key);
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| key);
            } else {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
            }
        } else if ascending {
            keyed.sort_unstable_by_key(|&(_, key)| key);
        } else {
            keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
        }

        // Write sorted indices back
        for (i, (idx, _)) in keyed.into_iter().enumerate() {
            row_indices[i] = idx;
        }
    }

    /// Single column generic sort (columnar layout)
    fn sort_single_column_columnar(row_indices: &mut [usize], col: &[Value], ascending: bool) {
        const PARALLEL_THRESHOLD: usize = 10_000;

        let compare = |&a: &usize, &b: &usize| -> Ordering {
            let a_val = col.get(a);
            let b_val = col.get(b);

            let cmp = match (a_val, b_val) {
                (Some(a), Some(b)) => Self::compare_values_fast(a, b),
                (None, None) => Ordering::Equal,
                (None, _) => Ordering::Greater,
                (_, None) => Ordering::Less,
            };
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        };

        if row_indices.len() >= PARALLEL_THRESHOLD {
            #[cfg(feature = "parallel")]
            row_indices.par_sort_unstable_by(compare);
            #[cfg(not(feature = "parallel"))]
            row_indices.sort_unstable_by(compare);
        } else {
            row_indices.sort_unstable_by(compare);
        }
    }

    /// Compute aggregate function as window function (SUM, COUNT, AVG, MIN, MAX)
    /// cached_order_by: Optional precomputed ORDER BY values from cache to avoid redundant computation
    #[allow(clippy::too_many_arguments)]
    fn compute_aggregate_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
        cached_order_by: Option<&ColumnarOrderByValues>,
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
                .map(|(_, row)| eval.eval(row).unwrap_or_else(|_| Value::null_unknown()))
                .collect()
        } else {
            vec![]
        };

        // Use precomputed ORDER BY values from cache if available
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            num_rows: 0,
        };
        let order_by_values: &ColumnarOrderByValues = cached_order_by.unwrap_or(&empty_order_by);

        // Group rows by partition key
        let partitions = Self::build_partition_map(wf_info, rows, columns, col_index_map, ctx)?;

        // Check if we can skip sorting (index optimization)
        // Only applies when there's no PARTITION BY (single partition)
        let skip_sorting =
            wf_info.partition_by_exprs.is_empty() && self.check_rows_presorted(wf_info, pre_sorted);

        // Compute aggregate for each partition
        // OPTIMIZATION: Use Option<Value> with vec![None; n] - None is just a discriminant
        // (no data to clone), much faster than vec![NULL_VALUE; n] which clones ~32 byte Values
        let mut results: Vec<Option<Value>> = vec![None; rows.len()];

        for (_key, mut row_indices) in partitions {
            // Sort partition by ORDER BY if specified (skip if pre-sorted)
            if !wf_info.order_by.is_empty() {
                // Only sort if not already pre-sorted by index
                if !skip_sorting {
                    Self::sort_by_order_values(&mut row_indices, order_by_values);
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
                        // Compare adjacent rows by reference to avoid O(n) cloning
                        let mut prev_row_idx = row_indices[0];

                        for (i, &row_idx) in row_indices.iter().enumerate().skip(1) {
                            // Compare ORDER BY values without cloning
                            if !order_by_values.rows_equal(prev_row_idx, row_idx) {
                                // New peer group starts - fill in previous group for all its members
                                for _ in group_start..i {
                                    groups.push((group_start, i));
                                }
                                group_start = i;
                            }
                            prev_row_idx = row_idx;
                        }
                        // Fill in the last group
                        for _ in group_start..partition_len {
                            groups.push((group_start, partition_len));
                        }
                        groups
                    }
                };

                // Get aggregate function ONCE, reuse with reset() for each row
                let mut agg_func = self
                    .function_registry
                    .get_aggregate(&wf_info.name)
                    .ok_or_else(|| {
                        Error::NotSupported(format!("Unknown aggregate function: {}", wf_info.name))
                    })?;

                // Check if we can use O(n) incremental accumulation instead of O(n²) reset.
                // Safe when: frame start is UNBOUNDED PRECEDING (frame only grows),
                // frame end is not PRECEDING (which would make end < current row),
                // and no DISTINCT (distinct requires full set tracking).
                let can_incremental = wf_info.frame.as_ref().is_none_or(|f| {
                    let start_ok = matches!(f.start, WindowFrameBound::UnboundedPreceding);
                    let end_ok = f.end.as_ref().is_none_or(|e| {
                        !matches!(
                            e,
                            WindowFrameBound::Preceding(_) | WindowFrameBound::UnboundedPreceding
                        )
                    });
                    start_ok && end_ok
                });

                if can_incremental && !wf_info.is_distinct {
                    // O(n) incremental path: frame start is always 0, only end grows.
                    // Accumulate new values as frame_end advances, never reset.
                    let mut prev_end = 0usize;
                    for (i, &row_idx) in row_indices.iter().enumerate() {
                        let frame_end = if let Some(ref frame) = wf_info.frame {
                            let is_range = matches!(frame.unit, WindowFrameUnit::Range);
                            if let Some(ref end_bound) = frame.end {
                                match end_bound {
                                    WindowFrameBound::UnboundedFollowing => partition_len,
                                    WindowFrameBound::CurrentRow => {
                                        if is_range {
                                            peer_groups[i].1
                                        } else {
                                            i + 1
                                        }
                                    }
                                    WindowFrameBound::Following(expr) => {
                                        if is_range {
                                            if let (
                                                Some(curr_f64),
                                                Expression::IntegerLiteral(lit),
                                            ) = (
                                                order_by_values.get_first(row_idx).and_then(|v| {
                                                    match v {
                                                        Value::Integer(i) => Some(*i as f64),
                                                        Value::Float(f) => Some(*f),
                                                        _ => None,
                                                    }
                                                }),
                                                expr.as_ref(),
                                            ) {
                                                let upper = curr_f64 + lit.value as f64;
                                                let mut end_idx = 0;
                                                for (j, &idx) in row_indices.iter().enumerate() {
                                                    if let Some(rv) = order_by_values
                                                        .get_first(idx)
                                                        .and_then(|v| match v {
                                                            Value::Integer(i) => Some(*i as f64),
                                                            Value::Float(f) => Some(*f),
                                                            _ => None,
                                                        })
                                                    {
                                                        if rv <= upper {
                                                            end_idx = j + 1;
                                                        }
                                                    }
                                                }
                                                end_idx
                                            } else {
                                                partition_len
                                            }
                                        } else if let Expression::IntegerLiteral(lit) =
                                            expr.as_ref()
                                        {
                                            (i + lit.value as usize + 1).min(partition_len)
                                        } else {
                                            partition_len
                                        }
                                    }
                                    _ => {
                                        // PRECEDING / UNBOUNDED PRECEDING end bounds are excluded
                                        // by the can_incremental guard above. This branch is
                                        // unreachable but kept as a safe fallback.
                                        partition_len
                                    }
                                }
                            } else {
                                // No explicit end: default CURRENT ROW
                                if is_range {
                                    peer_groups[i].1
                                } else {
                                    i + 1
                                }
                            }
                        } else {
                            // Default frame: RANGE UNBOUNDED PRECEDING TO CURRENT ROW
                            peer_groups[i].1
                        };

                        // Accumulate only the new values since prev_end
                        for j in prev_end..frame_end {
                            let value = if let Some(col_idx) = arg_col_idx {
                                rows[row_indices[j]]
                                    .1
                                    .get(col_idx)
                                    .cloned()
                                    .unwrap_or_else(Value::null_unknown)
                            } else if has_expression_arg {
                                expression_values[row_indices[j]].clone()
                            } else {
                                Value::Integer(1)
                            };
                            agg_func.accumulate(&value, false);
                        }
                        if frame_end > prev_end {
                            prev_end = frame_end;
                        }
                        results[row_idx] = Some(agg_func.result());
                    }
                } else {
                    for (i, &row_idx) in row_indices.iter().enumerate() {
                        // Reset aggregate state for new frame computation
                        agg_func.reset();

                        // Compute frame bounds based on frame specification
                        let (frame_start, frame_end) = if let Some(ref frame) = wf_info.frame {
                            let is_range = matches!(frame.unit, WindowFrameUnit::Range);

                            // For RANGE frames with numeric offsets, we need value-based comparison
                            // Get the current row's ORDER BY value for RANGE calculations
                            let current_order_value = if is_range && !order_by_values.is_empty() {
                                order_by_values.get_first(row_idx).cloned()
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
                                                    if let Some(row_val) = order_by_values
                                                        .get_first(idx)
                                                        .and_then(value_to_f64)
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
                                                    if let Some(row_val) = order_by_values
                                                        .get_first(idx)
                                                        .and_then(value_to_f64)
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
                                            if let (
                                                Some(curr_val),
                                                Expression::IntegerLiteral(lit),
                                            ) = (&current_order_value, expr.as_ref())
                                            {
                                                if let Some(curr_f64) = value_to_f64(curr_val) {
                                                    let upper_bound = curr_f64 + lit.value as f64;
                                                    // Scan from end backwards to find last row in range
                                                    let mut end_idx = 0;
                                                    for (j, &idx) in row_indices.iter().enumerate()
                                                    {
                                                        if let Some(row_val) = order_by_values
                                                            .get_first(idx)
                                                            .and_then(value_to_f64)
                                                        {
                                                            if row_val <= upper_bound {
                                                                end_idx = j + 1;
                                                                // exclusive end
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
                                        } else if let Expression::IntegerLiteral(lit) =
                                            expr.as_ref()
                                        {
                                            (i + lit.value as usize + 1).min(partition_len)
                                        } else {
                                            partition_len
                                        }
                                    }
                                    WindowFrameBound::Preceding(expr) => {
                                        if is_range {
                                            // RANGE PRECEDING as end: find last row where value <= current - offset
                                            if let (
                                                Some(curr_val),
                                                Expression::IntegerLiteral(lit),
                                            ) = (&current_order_value, expr.as_ref())
                                            {
                                                if let Some(curr_f64) = value_to_f64(curr_val) {
                                                    let upper_bound = curr_f64 - lit.value as f64;
                                                    let mut end_idx = 0;
                                                    for (j, &idx) in row_indices.iter().enumerate()
                                                    {
                                                        if let Some(row_val) = order_by_values
                                                            .get_first(idx)
                                                            .and_then(value_to_f64)
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
                                        } else if let Expression::IntegerLiteral(lit) =
                                            expr.as_ref()
                                        {
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
                            // Default frame: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            // SQL standard: peer rows (same ORDER BY value) share the same frame end
                            (0, peer_groups[i].1)
                        };

                        // Accumulate values within the frame
                        for &idx in &row_indices[frame_start..frame_end] {
                            let value = if let Some(col_idx) = arg_col_idx {
                                rows[idx]
                                    .1
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
                        results[row_idx] = Some(agg_func.result());
                    }
                } // end of else (non-incremental path)
            } else {
                // Without ORDER BY, compute aggregate over entire partition
                let mut agg_func = self
                    .function_registry
                    .get_aggregate(&wf_info.name)
                    .ok_or_else(|| {
                        Error::NotSupported(format!("Unknown aggregate function: {}", wf_info.name))
                    })?;

                // Accumulate all values in the partition
                for &row_idx in &row_indices {
                    let value = if let Some(col_idx) = arg_col_idx {
                        rows[row_idx]
                            .1
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
                    results[row_idx] = Some(aggregate_result.clone());
                }
            }
        }

        // Unwrap all values (all indices should have been written)
        Ok(results
            .into_iter()
            .map(|opt| opt.unwrap_or(NULL_VALUE))
            .collect())
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
        use crate::executor::query_classification::get_classification;

        // Test with window function
        let mut parser = crate::parser::Parser::new("SELECT ROW_NUMBER() OVER () FROM test");
        if let Ok(program) = parser.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                let classification = get_classification(stmt);
                assert!(classification.has_window_functions);
            }
        }

        // Test without window function
        let mut parser2 = crate::parser::Parser::new("SELECT * FROM test");
        if let Ok(program) = parser2.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                let classification = get_classification(stmt);
                assert!(!classification.has_window_functions);
            }
        }
    }

    #[test]
    fn test_window_function_info() {
        let info = WindowFunctionInfo {
            name: "ROW_NUMBER".to_string(),
            arguments: vec![],
            partition_by: vec!["dept".to_string()],
            partition_by_exprs: vec![],
            order_by: vec![],
            frame: None,
            column_name: "rn".to_string(),
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
