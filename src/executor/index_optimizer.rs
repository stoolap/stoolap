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

//! Index-based Query Optimizations
//!
//! This module contains optimizations that leverage indexes to speed up queries:
//! - MIN/MAX on indexed columns (O(1) instead of O(n))
//! - COUNT(*) using row_count() (O(1) instead of O(n))
//! - ORDER BY + LIMIT using index-ordered scan
//! - IN list/subquery/hashset using index probe
//! - Window function pre-sorting detection

use std::sync::Arc;

use ahash::AHashSet;
use rustc_hash::FxHashSet;

use crate::core::{Result, Row, Value};
use crate::parser::ast::*;
use crate::storage::traits::{QueryResult, Table};

use super::context::{cache_in_subquery, get_cached_in_subquery, ExecutionContext};
use super::expression::{ExpressionEval, RowFilter};
use super::query_classification::QueryClassification;
use super::result::ExecutorMemoryResult;
use super::Executor;

impl Executor {
    /// Try to optimize simple MIN/MAX aggregates using index
    ///
    /// For queries like `SELECT MIN(col) FROM table` or `SELECT MAX(col) FROM table`
    /// without WHERE or GROUP BY, use the index's O(1) min/max lookup instead of O(n) scan.
    #[allow(clippy::type_complexity)]
    pub(crate) fn try_min_max_index_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn Table,
        _all_columns: &[String],
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Only optimize single MIN or MAX without DISTINCT
        if stmt.columns.len() != 1 {
            return Ok(None);
        }

        let col_expr = &stmt.columns[0];

        // Extract function info (handle aliased case too)
        let (func, alias) = match col_expr {
            Expression::FunctionCall(func) => (func, None),
            Expression::Aliased(aliased) => {
                if let Expression::FunctionCall(func) = aliased.expression.as_ref() {
                    (func, Some(aliased.alias.value.clone()))
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Check if it's MIN or MAX
        // OPTIMIZATION: func.function is already uppercase from parsing
        if func.function != "MIN" && func.function != "MAX" {
            return Ok(None);
        }

        // Don't optimize DISTINCT
        if func.is_distinct {
            return Ok(None);
        }

        // Extract column name
        if func.arguments.is_empty() {
            return Ok(None);
        }

        let column_name = match &func.arguments[0] {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            _ => return Ok(None),
        };

        // Try to get value from index
        let value = if func.function == "MIN" {
            table.get_index_min_value(&column_name)
        } else {
            table.get_index_max_value(&column_name)
        };

        if let Some(val) = value {
            // Build result
            let col_name = alias.unwrap_or_else(|| format!("{}({})", func.function, column_name));
            let columns = vec![col_name.clone()];
            let rows = vec![Row::from_values(vec![val])];
            let result: Box<dyn QueryResult> =
                Box::new(ExecutorMemoryResult::new(columns.clone(), rows));
            return Ok(Some((result, columns)));
        }

        Ok(None)
    }

    /// Try to optimize simple COUNT(*) queries using table row_count
    ///
    /// For queries like `SELECT COUNT(*) FROM table` without WHERE or GROUP BY,
    /// use the table's O(1) row_count() method instead of O(n) scan.
    #[allow(clippy::type_complexity)]
    pub(crate) fn try_count_star_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn Table,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Only optimize single COUNT(*) without DISTINCT
        if stmt.columns.len() != 1 {
            return Ok(None);
        }

        let col_expr = &stmt.columns[0];

        // Extract function info (handle aliased case too)
        let (func, alias) = match col_expr {
            Expression::FunctionCall(func) => (func, None),
            Expression::Aliased(aliased) => {
                if let Expression::FunctionCall(func) = aliased.expression.as_ref() {
                    (func, Some(aliased.alias.value.clone()))
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        // Check if it's COUNT
        // OPTIMIZATION: func.function is already uppercase from parsing
        if func.function != "COUNT" {
            return Ok(None);
        }

        // Don't optimize DISTINCT
        if func.is_distinct {
            return Ok(None);
        }

        // Don't optimize if FILTER clause is present - requires row-by-row evaluation
        if func.filter.is_some() {
            return Ok(None);
        }

        // Must be COUNT(*) - either empty args or Star expression
        let is_count_star = func.arguments.is_empty()
            || (func.arguments.len() == 1 && matches!(func.arguments[0], Expression::Star(_)));

        if !is_count_star {
            return Ok(None);
        }

        // Use table's row_count method (O(1) instead of O(n))
        let count = table.row_count();

        // Build result
        let col_name = alias.unwrap_or_else(|| "COUNT(*)".to_string());
        let columns = vec![col_name.clone()];
        let rows = vec![Row::from_values(vec![Value::Integer(count as i64)])];
        let result: Box<dyn QueryResult> =
            Box::new(ExecutorMemoryResult::new(columns.clone(), rows));
        Ok(Some((result, columns)))
    }

    /// Try to optimize ORDER BY + LIMIT using index-ordered scan
    ///
    /// For queries like `SELECT * FROM table ORDER BY col LIMIT 10`,
    /// use the index to get rows in sorted order directly (O(limit) instead of O(n log n)).
    #[allow(clippy::type_complexity)]
    pub(crate) fn try_order_by_index_optimization(
        &self,
        stmt: &SelectStatement,
        table: &dyn Table,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Get the ORDER BY column name
        let order_by = &stmt.order_by[0];
        let column_name = match &order_by.expression {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
            _ => return Ok(None), // Can't optimize complex ORDER BY expressions
        };

        // Determine sort order
        let ascending = order_by.ascending;

        // Evaluate limit and offset
        let limit = if let Some(ref limit_expr) = stmt.limit {
            match ExpressionEval::compile(limit_expr, &[])
                .ok()
                .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
            {
                Some(Value::Integer(l)) => l as usize,
                Some(Value::Float(f)) => f as usize,
                _ => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        let offset = if let Some(ref offset_expr) = stmt.offset {
            match ExpressionEval::compile(offset_expr, &[])
                .ok()
                .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
            {
                Some(Value::Integer(o)) => o as usize,
                Some(Value::Float(f)) => f as usize,
                _ => 0,
            }
        } else {
            0
        };

        // Try to use index-ordered scan
        if let Some(rows) =
            table.collect_rows_ordered_by_index(&column_name, ascending, limit, offset)
        {
            // Project rows according to SELECT expressions
            let projected_rows = self.project_rows(&stmt.columns, rows, all_columns, ctx)?;

            // Note: This optimization path doesn't have table_alias available,
            // so we pass None. The prefix-based matching will still work for JOINs.
            let output_columns = self.get_output_column_names(&stmt.columns, all_columns, None);

            let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
            return Ok(Some((Box::new(result), output_columns)));
        }

        Ok(None)
    }

    /// Extract window ORDER BY information for optimization
    /// Returns (column_name, ascending) if a simple optimizable case is found
    pub(crate) fn extract_window_order_info(stmt: &SelectStatement) -> Option<(String, bool)> {
        // Look for window functions in SELECT columns
        for col_expr in &stmt.columns {
            if let Some(info) = Self::find_window_order_in_expr(col_expr) {
                return Some(info);
            }
        }
        None
    }

    /// Find window ORDER BY info in an expression
    fn find_window_order_in_expr(expr: &Expression) -> Option<(String, bool)> {
        match expr {
            Expression::Window(window_expr) => {
                // Check for no PARTITION BY (single partition case)
                if !window_expr.partition_by.is_empty() {
                    return None; // Pre-sorting doesn't help with partitions
                }

                // Check if using a window reference (can't analyze those)
                if window_expr.window_ref.is_some() {
                    return None;
                }

                // Get ORDER BY info
                let order_by = &window_expr.order_by;

                // Only optimize if exactly one simple ORDER BY column
                if order_by.len() != 1 {
                    return None;
                }

                let order = &order_by[0];
                let column_name = match &order.expression {
                    Expression::Identifier(id) => id.value.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
                    _ => return None, // Complex expression, can't optimize
                };

                Some((column_name, order.ascending))
            }
            Expression::Aliased(aliased) => Self::find_window_order_in_expr(&aliased.expression),
            _ => None,
        }
    }

    /// Extract window PARTITION BY information for optimization
    /// Returns column_name if a simple single-column PARTITION BY is found
    pub(crate) fn extract_window_partition_info(stmt: &SelectStatement) -> Option<String> {
        // Look for window functions in SELECT columns
        for col_expr in &stmt.columns {
            if let Some(info) = Self::find_window_partition_in_expr(col_expr) {
                return Some(info);
            }
        }
        None
    }

    /// Find window PARTITION BY info in an expression
    fn find_window_partition_in_expr(expr: &Expression) -> Option<String> {
        match expr {
            Expression::Window(window_expr) => {
                // Only optimize single-column PARTITION BY
                if window_expr.partition_by.len() != 1 {
                    return None;
                }

                // Check if using a window reference (can't analyze those)
                if window_expr.window_ref.is_some() {
                    return None;
                }

                // Get PARTITION BY column
                let partition_col = &window_expr.partition_by[0];
                let column_name = match partition_col {
                    Expression::Identifier(id) => id.value.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value.clone(),
                    _ => return None, // Complex expression, can't optimize
                };

                Some(column_name)
            }
            Expression::Aliased(aliased) => {
                Self::find_window_partition_in_expr(&aliased.expression)
            }
            _ => None,
        }
    }

    /// IN subquery index optimization
    ///
    /// For queries like `SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE ...)`
    /// where `id` has an index, probe the index for each subquery value instead of scanning all rows.
    /// This is O(k log n) where k = subquery result size, vs O(n) for full table scan.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_in_subquery_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
        classification: &Arc<QueryClassification>,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract IN subquery info: (column_name, subquery, is_negated, remaining_predicate)
        let (column_name, subquery, is_negated, remaining_predicate) =
            match Self::extract_in_subquery_info(where_expr) {
                Some(info) => info,
                None => return Ok(None),
            };

        // Skip correlated subqueries - they can't be pre-evaluated
        if Self::is_subquery_correlated(&subquery.subquery) {
            return Ok(None);
        }

        // Skip if SELECT columns have correlated subqueries (need per-row context)
        // classification is passed from caller to avoid redundant cache lookups
        if classification.select_has_correlated_subqueries {
            return Ok(None);
        }

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            schema.columns[pk_col_idx].name.to_lowercase() == column_name
        };

        // If not PK, check for index
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        // Execute the subquery to get values (with caching for non-correlated subqueries)
        let cache_key = subquery.subquery.to_string();
        let values = if let Some(cached) = get_cached_in_subquery(&cache_key) {
            cached
        } else {
            let subquery_ctx = ctx.with_incremented_query_depth();
            let mut result = self.execute_select(&subquery.subquery, &subquery_ctx)?;

            // Collect all values from the first column
            let mut values = Vec::new();
            while result.next() {
                let row = result.row();
                if !row.is_empty() {
                    values.push(
                        row.get(0)
                            .cloned()
                            .unwrap_or_else(crate::core::Value::null_unknown),
                    );
                }
            }
            // Cache for future use
            cache_in_subquery(cache_key, values.clone());
            values
        };

        if values.is_empty() {
            // Empty subquery result
            if is_negated {
                // NOT IN empty set = all rows match (fall through to normal scan)
                return Ok(None);
            } else {
                // IN empty set = no rows match
                let output_columns =
                    self.get_output_column_names(&stmt.columns, all_columns, table_alias);
                let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
                return Ok(Some((Box::new(result), output_columns)));
            }
        }

        // Collect row_ids: either from PK (direct) or from index probe
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            if is_negated {
                // NOT IN optimization for INTEGER PRIMARY KEY:
                // Instead of scanning the table, iterate through row_ids and exclude
                // This is O(row_count) but with O(1) hashset lookup, which is faster
                // than table scan because we only touch row_ids, not full rows
                let exclusion_set: FxHashSet<i64> = values
                    .iter()
                    .filter_map(|v| {
                        if let Value::Integer(id) = v {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Calculate LIMIT + OFFSET to know how many row_ids we need
                let offset = stmt
                    .offset
                    .as_ref()
                    .and_then(|e| {
                        ExpressionEval::compile(e, &[])
                            .ok()
                            .and_then(|eval| eval.with_context(ctx).eval_slice(&[]).ok())
                            .and_then(|v| {
                                if let Value::Integer(o) = v {
                                    Some(o.max(0) as usize)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(0);

                let limit = stmt
                    .limit
                    .as_ref()
                    .and_then(|e| {
                        ExpressionEval::compile(e, &[])
                            .ok()
                            .and_then(|eval| eval.with_context(ctx).eval_slice(&[]).ok())
                            .and_then(|v| {
                                if let Value::Integer(l) = v {
                                    Some(l.max(0) as usize)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or(usize::MAX);

                let target = if limit == usize::MAX {
                    usize::MAX
                } else {
                    offset.saturating_add(limit)
                };

                // Iterate through row_ids and collect non-excluded ones
                // Row IDs are typically 1-based and sequential
                let row_count = table.row_count();
                for row_id in 1..=(row_count as i64) {
                    if !exclusion_set.contains(&row_id) {
                        all_row_ids.push(row_id);
                        if all_row_ids.len() >= target {
                            break;
                        }
                    }
                }

                // Apply offset
                if offset > 0 && offset < all_row_ids.len() {
                    all_row_ids = all_row_ids.split_off(offset);
                } else if offset >= all_row_ids.len() {
                    all_row_ids.clear();
                }

                // Apply limit
                if limit < all_row_ids.len() {
                    all_row_ids.truncate(limit);
                }
            } else {
                // IN: PRIMARY KEY - the value IS the row_id (for INTEGER PK)
                for value in &values {
                    if let Value::Integer(id) = value {
                        all_row_ids.push(*id);
                    }
                }
            }
        } else if is_negated {
            // NOT IN with non-PK index: fall back to normal scan
            return Ok(None);
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in &values {
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // Remove duplicates and sort for efficient batch fetch (only for IN, NOT IN is already ordered)
        if !is_negated {
            all_row_ids.sort_unstable();
            all_row_ids.dedup();
        }

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Process any subqueries in the remaining predicate
            let processed_remaining = if Self::has_subqueries(remaining) {
                self.process_where_subqueries(remaining, ctx)?
            } else {
                remaining.clone()
            };

            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter = RowFilter::new(&processed_remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// IN list literal index optimization
    ///
    /// For queries like `SELECT * FROM users WHERE id IN (1, 2, 3, 5, 8)`
    /// where `id` has an index or is PRIMARY KEY, probe the index directly for each value
    /// instead of scanning all rows. This is O(k log n) where k = list size.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_in_list_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
        classification: &Arc<QueryClassification>,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract IN list info: (column_name, values, is_negated, remaining_predicate)
        let (column_name, values, is_negated, remaining_predicate) =
            match Self::extract_in_list_info(where_expr, ctx) {
                Some(info) => info,
                None => return Ok(None),
            };

        // Skip if SELECT columns have correlated subqueries (need per-row context)
        // classification is passed from caller to avoid redundant cache lookups
        if classification.select_has_correlated_subqueries {
            return Ok(None);
        }

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            schema.columns[pk_col_idx].name.to_lowercase() == column_name
        };

        // If not PK, check for index
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        if values.is_empty() {
            // Empty IN list
            if is_negated {
                // NOT IN empty set = all rows match (fall through to normal scan)
                return Ok(None);
            } else {
                // IN empty set = no rows match
                let output_columns =
                    self.get_output_column_names(&stmt.columns, all_columns, table_alias);
                let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
                return Ok(Some((Box::new(result), output_columns)));
            }
        }

        // For NOT IN, we can't easily use the index (would need to exclude rows)
        // Fall back to normal scan
        if is_negated {
            return Ok(None);
        }

        // Collect row_ids: either from PK (direct) or from index probe
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            // PRIMARY KEY: the value IS the row_id (for INTEGER PK)
            for value in &values {
                if let Value::Integer(id) = value {
                    all_row_ids.push(*id);
                }
            }
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in &values {
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // Remove duplicates and sort for efficient batch fetch
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Process any subqueries in the remaining predicate
            let processed_remaining = if Self::has_subqueries(remaining) {
                self.process_where_subqueries(remaining, ctx)?
            } else {
                remaining.clone()
            };

            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter = RowFilter::new(&processed_remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// Extract IN list literal information from a WHERE clause.
    /// Returns (column_name, values, is_negated, remaining_predicate)
    pub(crate) fn extract_in_list_info(
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<(String, Vec<Value>, bool, Option<Expression>)> {
        match expr {
            // Direct IN list: column IN (v1, v2, ...)
            Expression::In(in_expr) => {
                // Get the column name from the left side (lowercase for case-insensitive match)
                let column_name = match in_expr.left.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex left expressions
                };

                // Get the values from the right side (must be a literal list, not subquery)
                let values = match in_expr.right.as_ref() {
                    Expression::List(list) => {
                        // ListExpression has Vec<Expression>
                        Self::extract_literal_values(&list.elements, ctx)
                    }
                    Expression::ExpressionList(list) => {
                        // ExpressionList has Vec<Expression>
                        Self::extract_literal_values(&list.expressions, ctx)
                    }
                    _ => return None, // Not a literal list (might be subquery)
                };

                values.map(|v| (column_name, v, in_expr.not, None))
            }

            // IN list with AND: column IN (...) AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as IN list
                if let Some((col, vals, neg, _)) = Self::extract_in_list_info(&infix.left, ctx) {
                    return Some((col, vals, neg, Some((*infix.right).clone())));
                }
                // Try right side as IN list
                if let Some((col, vals, neg, _)) = Self::extract_in_list_info(&infix.right, ctx) {
                    return Some((col, vals, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
    }

    /// Extract literal values from a list of expressions.
    /// Returns None if any expression is not a literal (e.g., column reference, subquery).
    pub(crate) fn extract_literal_values(
        exprs: &[Expression],
        ctx: &ExecutionContext,
    ) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(exprs.len());
        for expr in exprs {
            // Try to evaluate as a constant expression
            match ExpressionEval::compile(expr, &[]) {
                Ok(compiled) => match compiled.with_context(ctx).eval_slice(&[]) {
                    Ok(val) => values.push(val),
                    Err(_) => return None, // Can't evaluate as constant
                },
                Err(_) => return None, // Can't compile (e.g., column reference)
            }
        }
        Some(values)
    }

    /// Extract IN subquery information from a WHERE clause.
    /// Returns (column_name, subquery, is_negated, remaining_predicate)
    pub(crate) fn extract_in_subquery_info(
        expr: &Expression,
    ) -> Option<(String, &ScalarSubquery, bool, Option<Expression>)> {
        match expr {
            // Direct IN subquery: column IN (SELECT ...)
            Expression::In(in_expr) => {
                // Get the column name from the left side (lowercase for case-insensitive match)
                let column_name = match in_expr.left.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex left expressions
                };

                // Get the subquery from the right side
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    return Some((column_name, subquery, in_expr.not, None));
                }
                None
            }

            // IN subquery with AND: column IN (SELECT ...) AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as IN subquery
                if let Some((col, sq, neg, _)) = Self::extract_in_subquery_info(&infix.left) {
                    return Some((col, sq, neg, Some((*infix.right).clone())));
                }
                // Try right side as IN subquery
                if let Some((col, sq, neg, _)) = Self::extract_in_subquery_info(&infix.right) {
                    return Some((col, sq, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
    }

    /// Try to optimize InHashSet expressions (from EXISTS → semi-join transformation).
    ///
    /// When EXISTS is transformed to InHashSet via semi-join, we can further optimize
    /// by probing the PRIMARY KEY or index directly instead of scanning all rows.
    ///
    /// For example: `SELECT * FROM users WHERE users.id IN {1, 2, 3} LIMIT 100`
    /// Instead of scanning all 10,000 users, probe the PK for ids 1, 2, 3 directly.
    #[allow(clippy::type_complexity)]
    pub(crate) fn try_in_hashset_index_optimization(
        &self,
        stmt: &SelectStatement,
        where_expr: &Expression,
        table: &dyn Table,
        all_columns: &[String],
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Option<(Box<dyn QueryResult>, Vec<String>)>> {
        // Extract InHashSet info: (column_name, values, is_negated, remaining_predicate)
        let (column_name, values, is_negated, remaining_predicate) =
            match Self::extract_in_hashset_info(where_expr) {
                Some(info) => info,
                None => return Ok(None),
            };

        // Check if this is a PRIMARY KEY column (O(1) lookup) or has an index
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let is_pk_column = pk_indices.len() == 1 && {
            let pk_col_idx = pk_indices[0];
            let pk_col_name = schema.columns[pk_col_idx].name.to_lowercase();
            pk_col_name == column_name
        };

        // For NOT IN (negated), only optimize if it's a PK column
        // Non-PK NOT IN would require full index scan which isn't much better than table scan
        if is_negated && !is_pk_column {
            return Ok(None);
        }

        // If not PK, check for index (only for non-negated IN)
        let index = if !is_pk_column {
            match table.get_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None => return Ok(None), // No PK, no index, can't optimize
            }
        } else {
            None
        };

        // EARLY LIMIT CHECK: Compute limit+offset before collecting row_ids
        // This allows us to stop collection early when there's no ORDER BY
        let early_termination_target = if stmt.order_by.is_empty() && remaining_predicate.is_none()
        {
            let offset = stmt
                .offset
                .as_ref()
                .and_then(|offset_expr| {
                    ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                        .and_then(|v| {
                            if let Value::Integer(o) = v {
                                Some(o.max(0) as usize)
                            } else {
                                None
                            }
                        })
                })
                .unwrap_or(0);

            let limit = stmt.limit.as_ref().and_then(|limit_expr| {
                ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    .and_then(|v| {
                        if let Value::Integer(l) = v {
                            Some(l.max(0) as usize)
                        } else {
                            None
                        }
                    })
            });

            limit.map(|l| offset + l)
        } else {
            None // Can't use early termination with ORDER BY or remaining predicate
        };

        // Collect row_ids: either from PK (direct) or from index probe
        // With early termination target, stop once we have enough
        let mut all_row_ids = Vec::new();
        if is_pk_column {
            if is_negated {
                // NOT IN optimization for INTEGER PRIMARY KEY (from NOT EXISTS semi-join):
                // Iterate through row_ids and exclude those in the hash set
                let exclusion_set: FxHashSet<i64> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Integer(id) => Some(*id),
                        Value::Float(f) if f.fract() == 0.0 => Some(*f as i64),
                        _ => None,
                    })
                    .collect();

                let target = early_termination_target.unwrap_or(usize::MAX);
                let row_count = table.row_count();

                for row_id in 1..=(row_count as i64) {
                    if !exclusion_set.contains(&row_id) {
                        all_row_ids.push(row_id);
                        if all_row_ids.len() >= target {
                            break;
                        }
                    }
                }
            } else {
                // IN: PRIMARY KEY - the value IS the row_id (for INTEGER PK)
                for value in values.iter() {
                    // Early termination check
                    if let Some(target) = early_termination_target {
                        if all_row_ids.len() >= target {
                            break;
                        }
                    }
                    match value {
                        Value::Integer(id) => all_row_ids.push(*id),
                        Value::Float(f) => {
                            // Handle case where integer was stored as float
                            if f.fract() == 0.0 {
                                all_row_ids.push(*f as i64);
                            }
                        }
                        _ => {}
                    }
                }
            }
        } else if let Some(ref idx) = index {
            // Index probe for each value
            for value in values.iter() {
                // Early termination check
                if let Some(target) = early_termination_target {
                    if all_row_ids.len() >= target {
                        break;
                    }
                }
                let row_ids = idx.get_row_ids_equal(std::slice::from_ref(value));
                all_row_ids.extend(row_ids);
            }
        }

        // If no row_ids found, return empty result
        if all_row_ids.is_empty() {
            let output_columns =
                self.get_output_column_names(&stmt.columns, all_columns, table_alias);
            let result = ExecutorMemoryResult::new(output_columns.clone(), vec![]);
            return Ok(Some((Box::new(result), output_columns)));
        }

        // Remove duplicates and sort for efficient batch fetch
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        let (early_limit_applied, early_limit, early_offset) =
            if stmt.order_by.is_empty() && remaining_predicate.is_none() {
                let offset = if let Some(ref offset_expr) = stmt.offset {
                    match ExpressionEval::compile(offset_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(o)) if o >= 0 => o as usize,
                        _ => 0,
                    }
                } else {
                    0
                };

                let limit = if let Some(ref limit_expr) = stmt.limit {
                    match ExpressionEval::compile(limit_expr, &[])
                        .ok()
                        .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                    {
                        Some(Value::Integer(l)) if l >= 0 => l as usize,
                        _ => usize::MAX,
                    }
                } else {
                    usize::MAX
                };

                // Truncate row_ids to avoid fetching unnecessary rows
                if limit < usize::MAX {
                    let take_count = (offset + limit).min(all_row_ids.len());
                    all_row_ids.truncate(take_count);
                }
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids
        let fetched_rows = table.fetch_rows_by_ids(&all_row_ids, filter.as_ref());

        // Convert (row_id, Row) to just Row
        let mut rows: Vec<crate::core::Row> =
            fetched_rows.into_iter().map(|(_, row)| row).collect();

        // Apply remaining predicate if any
        if let Some(ref remaining) = remaining_predicate {
            // Compile the filter using RowFilter
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter = RowFilter::new(remaining, &columns_slice)?;

            // Filter rows
            rows.retain(|row| row_filter.matches(row));
        }

        // Apply LIMIT/OFFSET if present (and no ORDER BY)
        // Skip if we already applied early limit optimization
        if early_limit_applied {
            // Early optimization already truncated row_ids, but we still need to apply offset
            if early_offset > 0 {
                rows = rows
                    .into_iter()
                    .skip(early_offset)
                    .take(early_limit)
                    .collect();
            } else if early_limit < rows.len() {
                rows.truncate(early_limit);
            }
        } else if stmt.order_by.is_empty() {
            let offset = if let Some(ref offset_expr) = stmt.offset {
                match ExpressionEval::compile(offset_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(o)) if o >= 0 => o as usize,
                    _ => 0,
                }
            } else {
                0
            };

            let limit = if let Some(ref limit_expr) = stmt.limit {
                match ExpressionEval::compile(limit_expr, &[])
                    .ok()
                    .and_then(|e| e.with_context(ctx).eval_slice(&[]).ok())
                {
                    Some(Value::Integer(l)) if l >= 0 => l as usize,
                    _ => usize::MAX,
                }
            } else {
                usize::MAX
            };

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        let projected_rows =
            self.project_rows_with_alias(&stmt.columns, rows, all_columns, ctx, table_alias)?;
        let output_columns = self.get_output_column_names(&stmt.columns, all_columns, table_alias);
        let result = ExecutorMemoryResult::new(output_columns.clone(), projected_rows);
        Ok(Some((Box::new(result), output_columns)))
    }

    /// Extract InHashSet information from a WHERE clause.
    /// Returns (column_name, hash_set_values, is_negated, remaining_predicate)
    #[allow(clippy::type_complexity)]
    pub(crate) fn extract_in_hashset_info(
        expr: &Expression,
    ) -> Option<(String, Arc<AHashSet<Value>>, bool, Option<Expression>)> {
        match expr {
            // Direct InHashSet: column IN {hash_set}
            Expression::InHashSet(in_hash) => {
                // Get the column name from the column expression (lowercase for case-insensitive match)
                let column_name = match in_hash.column.as_ref() {
                    Expression::Identifier(id) => id.value_lower.clone(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
                    _ => return None, // Can't optimize complex column expressions
                };

                Some((column_name, in_hash.values.clone(), in_hash.not, None))
            }

            // InHashSet with AND: column IN {hash_set} AND other_condition
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
                // Try left side as InHashSet
                if let Some((col, vals, neg, _)) = Self::extract_in_hashset_info(&infix.left) {
                    return Some((col, vals, neg, Some((*infix.right).clone())));
                }
                // Try right side as InHashSet
                if let Some((col, vals, neg, _)) = Self::extract_in_hashset_info(&infix.right) {
                    return Some((col, vals, neg, Some((*infix.left).clone())));
                }
                None
            }

            _ => None,
        }
    }
}
