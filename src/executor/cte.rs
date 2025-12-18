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

//! Common Table Expression (CTE) Execution
//!
//! This module implements WITH clause execution for SQL queries.
//!
//! Supports:
//! - Basic CTEs: `WITH x AS (SELECT ...) SELECT * FROM x`
//! - Multiple CTEs: `WITH a AS (...), b AS (...) SELECT ...`
//! - CTE with column aliases: `WITH x(col1, col2) AS (SELECT ...) SELECT ...`
//! - CTEs referencing other CTEs: `WITH a AS (...), b AS (SELECT * FROM a) ...`
//!
//! Optimizations:
//! - CTE Inlining: Single-use, non-recursive CTEs are converted to subqueries
//!   to preserve index access and benefit from LIMIT pushdown
//!
//! Note: Recursive CTEs are parsed but not yet executed.

use ahash::AHashSet;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::core::{Error, Result, Row, Value};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::traits::QueryResult;

use super::context::ExecutionContext;
use super::expression::ExpressionEval;
use super::join::{self, build_column_index_map};
use super::result::ExecutorMemoryResult;
use super::Executor;

/// Type alias for CTE data map
pub type CteDataMap = FxHashMap<String, (Vec<String>, Vec<Row>)>;

/// Type alias for join data result (left_columns, left_rows, right_columns, right_rows)
type JoinDataResult = (Vec<String>, Vec<Row>, Vec<String>, Vec<Row>);

/// Registry for CTE results during query execution
#[derive(Default)]
pub struct CteRegistry {
    /// Materialized CTE results (name -> (columns, rows))
    /// Wrapped in Arc for cheap sharing with ExecutionContext
    materialized: CteDataMap,
    /// Cached Arc for sharing - created lazily when data() is called
    shared: Option<Arc<CteDataMap>>,
}

impl CteRegistry {
    /// Create a new CTE registry
    pub fn new() -> Self {
        Self {
            materialized: FxHashMap::default(),
            shared: None,
        }
    }

    /// Store a materialized CTE result
    pub fn store(&mut self, name: &str, columns: Vec<String>, rows: Vec<Row>) {
        // Use stack-allocated buffer for lowercase conversion when possible
        let name_lower = name.to_lowercase();
        self.materialized.insert(name_lower, (columns, rows));
        // Invalidate cached Arc since data changed
        self.shared = None;
    }

    /// Get a CTE result by name
    pub fn get(&self, name: &str) -> Option<(&Vec<String>, &Vec<Row>)> {
        let name_lower = name.to_lowercase();
        self.materialized
            .get(&name_lower)
            .map(|(cols, rows)| (cols, rows))
    }

    /// Check if a CTE exists
    #[allow(dead_code)]
    pub fn exists(&self, name: &str) -> bool {
        self.materialized.contains_key(&name.to_lowercase())
    }

    /// Get a shared Arc reference to the internal data map for context transfer
    /// This is cheap after the first call (returns cached Arc clone)
    pub fn data(&mut self) -> Arc<CteDataMap> {
        if let Some(ref arc) = self.shared {
            arc.clone()
        } else {
            // Clone the data into an Arc only once
            let arc = Arc::new(self.materialized.clone());
            self.shared = Some(arc.clone());
            arc
        }
    }

    /// Iterate over all stored CTEs (for copying to temp registries)
    pub fn iter(&self) -> impl Iterator<Item = (&String, &(Vec<String>, Vec<Row>))> {
        self.materialized.iter()
    }
}

impl Executor {
    /// Execute a SELECT statement with WITH clause (CTEs)
    pub(crate) fn execute_select_with_ctes(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Get the WITH clause
        let with_clause = match &stmt.with {
            Some(with) => with,
            None => return self.execute_select(stmt, ctx),
        };

        // CTE INLINING OPTIMIZATION:
        // For single-use, non-recursive CTEs, convert to subqueries to:
        // 1. Preserve index access (CTEs lose indexes when materialized)
        // 2. Enable LIMIT pushdown through subqueries
        // This is similar to PostgreSQL 12+'s CTE inlining behavior
        if let Some(inlined_stmt) = self.try_inline_ctes(stmt, with_clause) {
            // Execute the rewritten query (without WITH clause or with fewer CTEs)
            return self.execute_select(&inlined_stmt, ctx);
        }

        // Create CTE registry
        let mut cte_registry = CteRegistry::new();

        // Execute each CTE in order
        for cte in &with_clause.ctes {
            // Execute the CTE query (handles recursive CTEs)
            let (columns, rows) = if cte.is_recursive {
                // Pass column aliases to recursive CTE execution so they're available during iteration
                let aliases = if cte.column_names.is_empty() {
                    None
                } else {
                    Some(cte.column_names.as_slice())
                };
                self.execute_recursive_cte_with_columns(
                    &cte.name.value,
                    &cte.query,
                    ctx,
                    &mut cte_registry,
                    aliases,
                )?
            } else {
                self.execute_cte_query(&cte.query, ctx, &mut cte_registry)?
            };

            // Apply column aliases if specified
            let columns = if !cte.column_names.is_empty() {
                cte.column_names
                    .iter()
                    .enumerate()
                    .map(|(i, alias)| {
                        if i < columns.len() {
                            alias.value.clone()
                        } else {
                            columns
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("col{}", i))
                        }
                    })
                    .collect()
            } else {
                columns
            };

            // Store the materialized result
            cte_registry.store(&cte.name.value, columns, rows);
        }

        // Execute the main query with CTE registry
        self.execute_main_query_with_ctes(stmt, ctx, &mut cte_registry)
    }

    /// Execute a single CTE query
    fn execute_cte_query(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Check if the CTE references another CTE
        if let Some(ref table_expr) = stmt.table_expr {
            // First check for simple CTE reference
            // BUT: if there are set operations (UNION, etc.), we need to fall through
            // to execute_select which handles them properly
            if stmt.set_operations.is_empty() {
                if let Some(cte_name) = self.extract_cte_name_for_lookup(table_expr) {
                    if let Some((columns, rows)) = cte_registry.get(&cte_name) {
                        // Execute query against CTE result
                        return self.execute_query_on_cte_result(
                            stmt,
                            ctx,
                            columns.clone(),
                            rows.clone(),
                        );
                    }
                }
            }

            // Check for JOIN expression that might involve CTEs
            if let Expression::JoinSource(join_source) = &**table_expr {
                // Check if either side is a CTE
                let left_is_cte = self
                    .extract_cte_name_for_lookup(&join_source.left)
                    .map(|n| cte_registry.get(&n).is_some())
                    .unwrap_or(false);
                let right_is_cte = self
                    .extract_cte_name_for_lookup(&join_source.right)
                    .map(|n| cte_registry.get(&n).is_some())
                    .unwrap_or(false);

                if left_is_cte || right_is_cte {
                    // Execute the JOIN with CTE awareness
                    if let Some(result) =
                        self.try_execute_join_with_ctes(stmt, ctx, join_source, cte_registry)?
                    {
                        let columns = result.columns().to_vec();
                        let rows = Self::materialize_result(result)?;
                        return Ok((columns, rows));
                    }
                }
            }
        }

        // Execute as a normal SELECT and materialize
        // Create context with CTE data so tables can resolve CTE references
        let ctx_with_ctes = ctx.with_cte_data(cte_registry.data());
        let result = self.execute_select(stmt, &ctx_with_ctes)?;
        let columns = result.columns().to_vec();
        let rows = Self::materialize_result(result)?;

        Ok((columns, rows))
    }

    /// Execute a recursive CTE
    fn execute_recursive_cte_with_columns(
        &self,
        cte_name: &str,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
        column_aliases: Option<&[Identifier]>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        use crate::parser::ast::SetOperationType;

        // Maximum iterations to prevent infinite loops
        const MAX_ITERATIONS: usize = 10000;

        // The recursive CTE query should have UNION ALL structure
        if stmt.set_operations.is_empty() {
            return Err(Error::InvalidArgumentMessage(
                "Recursive CTE must have UNION ALL between anchor and recursive members"
                    .to_string(),
            ));
        }

        // Check that all set operations are UNION ALL
        for set_op in &stmt.set_operations {
            if !matches!(set_op.operation, SetOperationType::UnionAll) {
                return Err(Error::InvalidArgumentMessage(
                    "Recursive CTE only supports UNION ALL (not UNION)".to_string(),
                ));
            }
        }

        // Execute the anchor member (the first SELECT before UNION ALL)
        let anchor_stmt = SelectStatement {
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            columns: stmt.columns.clone(),
            with: None,
            table_expr: stmt.table_expr.clone(),
            where_clause: stmt.where_clause.clone(),
            group_by: stmt.group_by.clone(),
            having: stmt.having.clone(),
            window_defs: vec![],
            order_by: vec![], // No ORDER BY for anchor
            limit: None,
            offset: None,
            set_operations: vec![],
        };

        let result = self.execute_select(&anchor_stmt, ctx)?;
        let anchor_columns = result.columns().to_vec();

        // Apply column aliases if provided (for recursive CTE column naming)
        let columns: Vec<String> = if let Some(aliases) = column_aliases {
            aliases
                .iter()
                .enumerate()
                .map(|(i, alias)| {
                    if i < anchor_columns.len() {
                        alias.value.clone()
                    } else {
                        anchor_columns
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("col{}", i))
                    }
                })
                .collect()
        } else {
            anchor_columns
        };

        let mut all_rows = Self::materialize_result(result)?;

        // If no anchor rows, return empty result
        if all_rows.is_empty() {
            return Ok((columns, all_rows));
        }

        // Current working set (rows from previous iteration)
        let mut working_rows = all_rows.clone();

        // Iterate until no new rows or max iterations
        for _iteration in 0..MAX_ITERATIONS {
            if working_rows.is_empty() {
                break;
            }

            // Create a temporary CTE registry with current working set
            let mut temp_registry = CteRegistry::new();
            // Copy existing CTEs
            for (name, (cols, rows)) in cte_registry.iter() {
                temp_registry.store(name, cols.clone(), rows.clone());
            }
            // Add current CTE with working rows
            temp_registry.store(cte_name, columns.clone(), working_rows.clone());

            // Execute each recursive member
            let mut new_rows: Vec<Row> = Vec::new();
            for set_op in &stmt.set_operations {
                // The recursive member is in set_op.right
                let recursive_result =
                    self.execute_cte_query(&set_op.right, ctx, &mut temp_registry)?;

                new_rows.extend(recursive_result.1);
            }

            if new_rows.is_empty() {
                break;
            }

            // Add new rows to total result
            all_rows.extend(new_rows.clone());

            // New rows become the working set for next iteration
            working_rows = new_rows;
        }

        Ok((columns, all_rows))
    }

    /// Execute the main query with CTEs available
    fn execute_main_query_with_ctes(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
    ) -> Result<Box<dyn QueryResult>> {
        // Create a context with CTE data so subqueries can access CTEs
        let ctx_with_ctes = ctx.with_cte_data(cte_registry.data());

        // Check if the main query references a CTE
        // NOTE: Skip the CTE fast path if there are set operations (UNION/INTERSECT/EXCEPT)
        // because execute_query_on_cte_result doesn't handle set_operations
        let has_set_operations = !stmt.set_operations.is_empty();

        if let Some(ref table_expr) = stmt.table_expr {
            // First check for simple CTE reference (only when no set operations)
            // Use the base CTE name (not alias) for registry lookup
            if !has_set_operations {
                if let Some(cte_name) = self.extract_cte_name_for_lookup(table_expr) {
                    if let Some((columns, rows)) = cte_registry.get(&cte_name) {
                        // Execute query against CTE result
                        let (result_cols, result_rows) = self.execute_query_on_cte_result(
                            stmt,
                            &ctx_with_ctes,
                            columns.clone(),
                            rows.clone(),
                        )?;
                        return Ok(Box::new(ExecutorMemoryResult::new(
                            result_cols,
                            result_rows,
                        )));
                    }
                }
            }

            // Check for JOIN expression that might involve CTEs
            if let Expression::JoinSource(join_source) = &**table_expr {
                // Try to execute JOIN with CTE awareness
                if let Some(result) = self.try_execute_join_with_ctes(
                    stmt,
                    &ctx_with_ctes,
                    join_source,
                    cte_registry,
                )? {
                    return Ok(result);
                }
            }
        }

        // Execute as a normal SELECT (strip the WITH clause)
        let stmt_without_with = SelectStatement {
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            columns: stmt.columns.clone(),
            with: None,
            table_expr: stmt.table_expr.clone(),
            where_clause: stmt.where_clause.clone(),
            group_by: stmt.group_by.clone(),
            having: stmt.having.clone(),
            window_defs: stmt.window_defs.clone(),
            order_by: stmt.order_by.clone(),
            limit: stmt.limit.clone(),
            offset: stmt.offset.clone(),
            set_operations: stmt.set_operations.clone(),
        };

        self.execute_select(&stmt_without_with, &ctx_with_ctes)
    }

    /// Try to execute a JOIN where one or both sides might be CTEs
    fn try_execute_join_with_ctes(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        join_source: &JoinTableSource,
        cte_registry: &CteRegistry,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        // Try to resolve left side - either from CTE registry or regular table
        let left_data = self.resolve_table_or_cte(&join_source.left, ctx, cte_registry)?;
        let right_data = self.resolve_table_or_cte(&join_source.right, ctx, cte_registry)?;

        // If neither side is a CTE, let normal processing handle it
        if left_data.is_none() && right_data.is_none() {
            return Ok(None);
        }

        // Extract table/CTE names for qualified column names
        let left_name = self.extract_cte_reference(&join_source.left);
        let right_name = self.extract_cte_reference(&join_source.right);

        // Track whether each side is from a CTE (needs prefixing) or regular table (already prefixed)
        // Must check before the match moves the data
        let left_is_cte = left_data.is_some();
        let right_is_cte = right_data.is_some();

        // Determine join type for semi-join reduction eligibility
        let join_type = join_source.join_type.to_uppercase();
        let is_inner_join = join_type == "INNER" || join_type.is_empty() || join_type == "JOIN";

        // SEMI-JOIN REDUCTION: For INNER JOIN where one side is CTE and other is regular table,
        // extract join keys from CTE to filter the regular table scan (uses index!)
        // This avoids materializing the entire regular table.
        let (left_columns, left_rows, right_columns, right_rows) = if is_inner_join
            && left_is_cte != right_is_cte
        {
            // One side is CTE, other is regular table - try semi-join reduction
            self.execute_cte_join_with_semijoin_reduction(
                &join_source.left,
                &join_source.right,
                join_source.condition.as_deref(),
                left_data,
                right_data,
                left_is_cte,
                ctx,
            )?
        } else {
            // Both CTEs or both regular tables - use standard path
            let (left_columns, left_rows) = match left_data {
                Some(data) => data,
                None => {
                    let (result, cols) = self.execute_table_expression(&join_source.left, ctx)?;
                    let rows = Self::materialize_result(result)?;
                    (cols, rows)
                }
            };

            let (right_columns, right_rows) = match right_data {
                Some(data) => data,
                None => {
                    let (result, cols) = self.execute_table_expression(&join_source.right, ctx)?;
                    let rows = Self::materialize_result(result)?;
                    (cols, rows)
                }
            };
            (left_columns, left_rows, right_columns, right_rows)
        };

        // Combine columns with qualified names (table.column) for proper resolution
        // This enables JOIN ON cte1.id = cte2.id to work correctly
        // CTE columns need prefixing; regular table columns are already prefixed by execute_table_expression
        let mut all_columns = Vec::new();

        // Add left columns with table prefix (only if from CTE, not already prefixed)
        let left_qualified: Vec<String> = if left_is_cte {
            if let Some(ref name) = left_name {
                left_columns
                    .iter()
                    .map(|col| format!("{}.{}", name, col))
                    .collect()
            } else {
                left_columns.clone()
            }
        } else {
            // Already prefixed by execute_table_expression
            left_columns.clone()
        };
        all_columns.extend(left_qualified.clone());

        // Add right columns with table prefix (only if from CTE, not already prefixed)
        let right_qualified: Vec<String> = if right_is_cte {
            if let Some(ref name) = right_name {
                right_columns
                    .iter()
                    .map(|col| format!("{}.{}", name, col))
                    .collect()
            } else {
                right_columns.clone()
            }
        } else {
            // Already prefixed by execute_table_expression
            right_columns.clone()
        };
        all_columns.extend(right_qualified.clone());

        // Execute the JOIN
        let join_type = join_source.join_type.to_uppercase();

        // Determine if we can safely push LIMIT down to the JOIN
        // Safe conditions: INNER/CROSS JOIN, no ORDER BY, no aggregation, no window functions
        let can_push_limit = (join_type == "INNER" || join_type == "CROSS" || join_type.is_empty())
            && stmt.order_by.is_empty()
            && !self.has_aggregation(stmt)
            && !self.has_window_functions(stmt)
            && stmt.where_clause.is_none(); // WHERE could filter rows, so we need all rows first

        // Calculate effective limit (LIMIT + OFFSET since we need offset rows too)
        let effective_limit = if can_push_limit {
            match (&stmt.limit, &stmt.offset) {
                (Some(lim), Some(off)) => {
                    let limit_val = self.evaluate_limit_offset(lim).unwrap_or(u64::MAX);
                    let offset_val = self.evaluate_limit_offset(off).unwrap_or(0);
                    Some(limit_val.saturating_add(offset_val))
                }
                (Some(lim), None) => self.evaluate_limit_offset(lim),
                _ => None,
            }
        } else {
            None
        };

        // Use hash join for better performance (O(n+m) instead of O(n*m))
        let result_rows = if let Some(condition) = join_source.condition.as_deref() {
            // Extract equality keys for hash join
            let (left_key_indices, right_key_indices, residual) =
                join::extract_join_keys_and_residual(condition, &left_qualified, &right_qualified);

            if !left_key_indices.is_empty() {
                // Disable LIMIT pushdown if there are residual conditions (they filter after join)
                let join_limit = if residual.is_empty() {
                    effective_limit
                } else {
                    None
                };

                // Use hash join with equality keys
                let mut rows = self.execute_hash_join(
                    &left_rows,
                    &right_rows,
                    &left_key_indices,
                    &right_key_indices,
                    &join_type,
                    left_qualified.len(),
                    right_qualified.len(),
                    join_limit,
                )?;

                // Apply residual conditions (non-equality parts of ON clause)
                if !residual.is_empty() {
                    self.apply_residual_conditions(
                        &mut rows,
                        &residual,
                        &all_columns,
                        &join_type,
                        left_qualified.len(),
                        right_qualified.len(),
                        ctx,
                    )?;
                }
                rows
            } else {
                // Fall back to nested loop for non-equality joins
                self.execute_nested_loop_join(
                    &left_rows,
                    &right_rows,
                    Some(condition),
                    &all_columns,
                    &left_qualified,
                    &right_qualified,
                    &join_type,
                    ctx,
                    effective_limit,
                )?
            }
        } else {
            // CROSS JOIN (no condition)
            self.execute_nested_loop_join(
                &left_rows,
                &right_rows,
                None,
                &all_columns,
                &left_qualified,
                &right_qualified,
                &join_type,
                ctx,
                effective_limit,
            )?
        };

        // Apply WHERE clause if present
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            let mut where_eval =
                ExpressionEval::compile(where_clause, &all_columns)?.with_context(ctx);

            let mut rows = Vec::new();
            for row in result_rows {
                if where_eval.eval_bool(&row) {
                    rows.push(row);
                }
            }
            rows
        } else {
            result_rows
        };

        // Check for window functions (must handle before projection)
        if self.has_window_functions(stmt) {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, filtered_rows, &all_columns)?;
            return Ok(Some(result));
        }

        // Check for aggregation (must handle before projection)
        if self.has_aggregation(stmt) {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &all_columns)?;

            // Apply ORDER BY if present (aggregation doesn't handle it for CTE JOINs)
            if !stmt.order_by.is_empty() {
                let output_columns = result.columns().to_vec();
                let rows = Self::materialize_result(result)?;
                let sorted_rows =
                    self.apply_order_by_to_rows(rows, &stmt.order_by, &output_columns)?;
                return Ok(Some(Box::new(ExecutorMemoryResult::new(
                    output_columns,
                    sorted_rows,
                ))));
            }
            return Ok(Some(result));
        }

        // Project rows according to SELECT expressions
        let projected_rows = self.project_rows(&stmt.columns, filtered_rows, &all_columns, ctx)?;

        // Determine output column names
        // Note: For CTE JOINs, columns are already qualified (e.g., "cte1.id", "cte2.id"),
        // so we pass None for table_alias - the prefix matching will work
        let output_columns = self.get_output_column_names(&stmt.columns, &all_columns, None);

        // Apply ORDER BY if present
        let final_rows = if !stmt.order_by.is_empty() {
            self.apply_order_by_to_rows(projected_rows, &stmt.order_by, &output_columns)?
        } else {
            projected_rows
        };

        Ok(Some(Box::new(ExecutorMemoryResult::new(
            output_columns,
            final_rows,
        ))))
    }

    /// Resolve a table expression - returns Some if it's a CTE, None if it's a regular table
    fn resolve_table_or_cte(
        &self,
        expr: &Expression,
        _ctx: &ExecutionContext,
        cte_registry: &CteRegistry,
    ) -> Result<Option<(Vec<String>, Vec<Row>)>> {
        // Use the base CTE name (not alias) for registry lookup
        if let Some(cte_name) = self.extract_cte_name_for_lookup(expr) {
            if let Some((columns, rows)) = cte_registry.get(&cte_name) {
                return Ok(Some((columns.clone(), rows.clone())));
            }
        }
        Ok(None)
    }

    /// Execute CTE JOIN with semi-join reduction optimization.
    /// When one side is a CTE and the other is a regular table, extract join keys from the CTE
    /// and use them to filter the regular table scan via IN clause (which can use indexes).
    #[allow(clippy::too_many_arguments)]
    fn execute_cte_join_with_semijoin_reduction(
        &self,
        left_expr: &Expression,
        right_expr: &Expression,
        join_condition: Option<&Expression>,
        left_data: Option<(Vec<String>, Vec<Row>)>,
        right_data: Option<(Vec<String>, Vec<Row>)>,
        left_is_cte: bool,
        ctx: &ExecutionContext,
    ) -> Result<JoinDataResult> {
        // Determine which side is CTE and which is regular table
        let (cte_expr, table_expr, cte_data, cte_on_left) = if left_is_cte {
            (left_expr, right_expr, left_data.unwrap(), true)
        } else {
            (right_expr, left_expr, right_data.unwrap(), false)
        };

        let (cte_columns, cte_rows) = cte_data;

        // Extract aliases for proper qualifier matching in join conditions
        let cte_alias = self.extract_cte_reference(cte_expr);
        let table_alias = self.extract_cte_reference(table_expr);

        // Semi-join reduction: only for small CTEs where IN clause is efficient
        // For large CTEs, hash join on full table is faster than multiple IN queries
        const MAX_SEMIJOIN_SIZE: usize = 500;

        // Try to extract join key column from condition (e.g., "u.id = h.user_id" -> id, user_id)
        // Uses qualifier matching to correctly identify which column belongs to which side
        let (table_join_col, cte_join_col) = if cte_rows.len() <= MAX_SEMIJOIN_SIZE {
            if let Some(condition) = join_condition {
                self.extract_join_key_columns_with_qualifiers(
                    condition,
                    cte_alias.as_deref(),
                    table_alias.as_deref(),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // If we can identify the CTE join column, extract values for IN filter
        let in_filter = if let Some(cte_col) = cte_join_col {
            if let Some(table_col) = table_join_col.clone() {
                // Find CTE column index
                let cte_col_lower = cte_col.to_lowercase();
                let cte_col_idx = cte_columns
                    .iter()
                    .position(|c| c.to_lowercase() == cte_col_lower);

                if let Some(idx) = cte_col_idx {
                    // Extract distinct non-NULL values from CTE
                    let mut seen: AHashSet<Value> = AHashSet::with_capacity(cte_rows.len());

                    for row in &cte_rows {
                        if let Some(val) = row.get(idx) {
                            if !val.is_null() {
                                seen.insert(val.clone());
                            }
                        }
                    }

                    let key_values: Vec<Value> = seen.into_iter().collect();

                    // Build IN filter: table_col IN (v1, v2, ...)
                    if !key_values.is_empty() {
                        self.build_in_filter_expression(&table_col, &key_values)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Execute regular table with or without IN filter
        let (table_result, table_cols) = if in_filter.is_some() {
            self.execute_table_expression_with_filter(table_expr, ctx, in_filter.as_ref())?
        } else {
            self.execute_table_expression(table_expr, ctx)?
        };
        let table_rows = Self::materialize_result(table_result)?;

        // Return in correct order (left, right)
        if cte_on_left {
            Ok((cte_columns, cte_rows, table_cols, table_rows))
        } else {
            Ok((table_cols, table_rows, cte_columns, cte_rows))
        }
    }

    /// Extract join key column names from a condition like "a.id = b.user_id".
    /// Returns (table_column, cte_column) by matching qualifiers against known aliases.
    /// This correctly handles both "cte.col = table.col" and "table.col = cte.col".
    fn extract_join_key_columns_with_qualifiers(
        &self,
        condition: &Expression,
        cte_alias: Option<&str>,
        table_alias: Option<&str>,
    ) -> (Option<String>, Option<String>) {
        match condition {
            Expression::Infix(infix) if infix.operator == "=" => {
                let left_info = self.extract_qualified_column(&infix.left);
                let right_info = self.extract_qualified_column(&infix.right);

                if let (Some((left_qualifier, left_col)), Some((right_qualifier, right_col))) =
                    (left_info, right_info)
                {
                    // Match qualifiers to aliases (case-insensitive)
                    let left_is_cte = self.qualifier_matches_alias(&left_qualifier, cte_alias);
                    let left_is_table = self.qualifier_matches_alias(&left_qualifier, table_alias);
                    let right_is_cte = self.qualifier_matches_alias(&right_qualifier, cte_alias);
                    let right_is_table =
                        self.qualifier_matches_alias(&right_qualifier, table_alias);

                    // Determine which column belongs to which side based on qualifiers
                    if left_is_cte && right_is_table {
                        (Some(right_col), Some(left_col))
                    } else if left_is_table && right_is_cte {
                        (Some(left_col), Some(right_col))
                    } else {
                        // Qualifiers don't match known aliases - skip optimization
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            Expression::Infix(infix) if infix.operator.eq_ignore_ascii_case("AND") => {
                // Try left side first
                let (table_col, cte_col) = self.extract_join_key_columns_with_qualifiers(
                    &infix.left,
                    cte_alias,
                    table_alias,
                );
                if table_col.is_some() && cte_col.is_some() {
                    return (table_col, cte_col);
                }
                // Try right side
                self.extract_join_key_columns_with_qualifiers(&infix.right, cte_alias, table_alias)
            }
            _ => (None, None),
        }
    }

    /// Check if a qualifier matches an alias (case-insensitive).
    fn qualifier_matches_alias(&self, qualifier: &Option<String>, alias: Option<&str>) -> bool {
        match (qualifier, alias) {
            (Some(q), Some(a)) => q.eq_ignore_ascii_case(a),
            _ => false,
        }
    }

    /// Extract qualifier and column name from an expression.
    /// Returns (Some(qualifier), column_name) for qualified identifiers,
    /// or (None, column_name) for simple identifiers.
    fn extract_qualified_column(&self, expr: &Expression) -> Option<(Option<String>, String)> {
        match expr {
            Expression::Identifier(id) => Some((None, id.value.clone())),
            Expression::QualifiedIdentifier(qi) => {
                Some((Some(qi.qualifier.value.clone()), qi.name.value.clone()))
            }
            _ => None,
        }
    }

    /// Evaluate a LIMIT or OFFSET expression to a u64 value
    fn evaluate_limit_offset(&self, expr: &Expression) -> Option<u64> {
        match expr {
            Expression::IntegerLiteral(lit) => {
                if lit.value >= 0 {
                    Some(lit.value as u64)
                } else {
                    None
                }
            }
            _ => None, // For complex expressions, return None (don't push limit)
        }
    }

    /// Execute a query on CTE result data
    pub(crate) fn execute_query_on_cte_result(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_columns: Vec<String>,
        cte_rows: Vec<Row>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Apply WHERE clause filter
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            // Process subqueries in WHERE clause (e.g., IN subqueries on CTEs)
            let processed_where = if Self::has_subqueries(where_clause) {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Compile filter once and reuse for all rows
            let mut eval =
                ExpressionEval::compile(&processed_where, &cte_columns)?.with_context(ctx);

            let mut result = Vec::new();
            for row in cte_rows {
                if eval.eval_bool(&row) {
                    result.push(row);
                }
            }
            result
        } else {
            cte_rows
        };

        // Check for aggregation
        if self.has_aggregation(stmt) {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &cte_columns)?;
            let columns = result.columns().to_vec();
            let rows = Self::materialize_result(result)?;

            return Ok((columns, rows));
        }

        // Check for window functions
        if self.has_window_functions(stmt) {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, filtered_rows, &cte_columns)?;
            let columns = result.columns().to_vec();
            let rows = Self::materialize_result(result)?;

            return Ok((columns, rows));
        }

        // Process scalar subqueries in SELECT columns before projection
        let processed_columns = self.try_process_select_subqueries(&stmt.columns, ctx)?;
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        // Determine output columns
        let output_columns =
            self.resolve_cte_output_columns_from_exprs(columns_to_use, &cte_columns)?;

        // Project rows if needed
        let mut result_rows = if self.needs_projection_for_columns(columns_to_use) {
            self.project_cte_rows_from_columns(columns_to_use, &filtered_rows, &cte_columns, ctx)?
        } else {
            filtered_rows
        };

        // Apply ORDER BY sorting
        if !stmt.order_by.is_empty() {
            result_rows =
                self.apply_order_by_to_rows(result_rows, &stmt.order_by, &output_columns)?;
        }

        // Apply LIMIT and OFFSET
        if let Some(ref offset_expr) = stmt.offset {
            let offset = match offset_expr.as_ref() {
                Expression::IntegerLiteral(lit) => lit.value as usize,
                Expression::FloatLiteral(lit) => lit.value as usize,
                _ => 0,
            };
            // Use drain to avoid extra allocation from skip().collect()
            if offset > 0 && offset < result_rows.len() {
                result_rows.drain(..offset);
            } else if offset >= result_rows.len() {
                result_rows.clear();
            }
        }

        if let Some(ref limit_expr) = stmt.limit {
            let limit = match limit_expr.as_ref() {
                Expression::IntegerLiteral(lit) => lit.value as usize,
                Expression::FloatLiteral(lit) => lit.value as usize,
                _ => usize::MAX,
            };
            if limit < result_rows.len() {
                result_rows.truncate(limit);
            }
        }

        Ok((output_columns, result_rows))
    }

    /// Extract the base CTE/table name for registry lookup (ignores aliases)
    fn extract_cte_name_for_lookup(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::CteReference(cte_ref) => Some(cte_ref.name.value.clone()),
            Expression::TableSource(simple_table_source) => {
                Some(simple_table_source.name.value.clone())
            }
            Expression::Identifier(id) => Some(id.value.clone()),
            _ => None,
        }
    }

    /// Extract CTE reference from table expression for column prefixing
    /// Returns the alias if present, otherwise the table/CTE name
    fn extract_cte_reference(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::CteReference(cte_ref) => {
                // Use alias if present, otherwise use the CTE name
                if let Some(ref alias) = cte_ref.alias {
                    Some(alias.value.clone())
                } else {
                    Some(cte_ref.name.value.clone())
                }
            }
            Expression::TableSource(simple_table_source) => {
                // Use alias if present, otherwise use the table name
                if let Some(ref alias) = simple_table_source.alias {
                    Some(alias.value.clone())
                } else {
                    Some(simple_table_source.name.value.clone())
                }
            }
            Expression::Identifier(id) => Some(id.value.clone()),
            _ => None,
        }
    }

    /// Resolve output column names from expression list (for processed subqueries)
    fn resolve_cte_output_columns_from_exprs(
        &self,
        columns: &[Expression],
        cte_columns: &[String],
    ) -> Result<Vec<String>> {
        let mut output_columns = Vec::new();

        for (i, col_expr) in columns.iter().enumerate() {
            match col_expr {
                Expression::Star(_) | Expression::QualifiedStar(_) => {
                    output_columns.extend(cte_columns.iter().cloned());
                }
                Expression::Identifier(id) => {
                    output_columns.push(id.value.clone());
                }
                Expression::Aliased(aliased) => {
                    output_columns.push(aliased.alias.value.clone());
                }
                _ => {
                    output_columns.push(format!("expr{}", i + 1));
                }
            }
        }

        if output_columns.is_empty() {
            output_columns = cte_columns.to_vec();
        }

        Ok(output_columns)
    }

    /// Check if columns need projection
    fn needs_projection_for_columns(&self, columns: &[Expression]) -> bool {
        if columns.is_empty() {
            return false;
        }

        // Check if it's just SELECT *
        if columns.len() == 1 {
            if let Expression::Star(_) = &columns[0] {
                return false;
            }
        }

        true
    }

    /// Project rows based on provided column expressions (for processed subqueries)
    fn project_cte_rows_from_columns(
        &self,
        columns: &[Expression],
        rows: &[Row],
        cte_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Row>> {
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};

        let col_index_map = build_column_index_map(cte_columns);

        // Pre-compile expressions that need evaluation
        // Store: Star -> None, Identifier -> column index, Complex -> compiled program
        enum CompiledColumn {
            Star,
            Identifier(Option<usize>),
            Compiled(SharedProgram),
        }

        let compiled_columns: Vec<CompiledColumn> = columns
            .iter()
            .map(|col_expr| match col_expr {
                Expression::Star(_) => Ok(CompiledColumn::Star),
                Expression::Identifier(id) => {
                    let idx = col_index_map.get(&id.value_lower).copied();
                    Ok(CompiledColumn::Identifier(idx))
                }
                Expression::Aliased(aliased) => {
                    let program = compile_expression(&aliased.expression, cte_columns)?;
                    Ok(CompiledColumn::Compiled(program))
                }
                _ => {
                    let program = compile_expression(col_expr, cte_columns)?;
                    Ok(CompiledColumn::Compiled(program))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Create VM for expression execution (reused for all rows)
        let mut vm = ExprVM::new();
        let mut result_rows = Vec::with_capacity(rows.len());

        for row in rows {
            // OPTIMIZATION: Pre-allocate Vec with estimated capacity
            let mut values = Vec::with_capacity(columns.len().max(row.len()));
            let row_data = row.as_slice();
            // CRITICAL: Include params from context for parameterized queries
            let exec_ctx = ExecuteContext::new(row_data)
                .with_params(ctx.params())
                .with_named_params(ctx.named_params());

            for compiled in &compiled_columns {
                match compiled {
                    CompiledColumn::Star => {
                        // OPTIMIZATION: Use extend_from_slice instead of to_vec()
                        values.extend_from_slice(row_data);
                    }
                    CompiledColumn::Identifier(Some(idx)) => {
                        values.push(row.get(*idx).cloned().unwrap_or_else(Value::null_unknown));
                    }
                    CompiledColumn::Identifier(None) => {
                        values.push(Value::null_unknown());
                    }
                    CompiledColumn::Compiled(program) => {
                        values.push(vm.execute(program, &exec_ctx)?);
                    }
                }
            }

            result_rows.push(Row::from_values(values));
        }

        Ok(result_rows)
    }

    /// Check if a SELECT statement has a WITH clause
    pub(crate) fn has_cte(&self, stmt: &SelectStatement) -> bool {
        stmt.with.is_some()
    }

    /// Apply ORDER BY to rows
    fn apply_order_by_to_rows(
        &self,
        mut rows: Vec<Row>,
        order_by: &[crate::parser::ast::OrderByExpression],
        columns: &[String],
    ) -> Result<Vec<Row>> {
        if order_by.is_empty() || rows.is_empty() {
            return Ok(rows);
        }

        // Build column index map
        let col_index_map = build_column_index_map(columns);

        // Build order specs: (column_index, ascending, nulls_first)
        let order_specs: Vec<(Option<usize>, bool, Option<bool>)> = order_by
            .iter()
            .map(|ob| {
                let col_idx = match &ob.expression {
                    Expression::Identifier(id) => col_index_map.get(&id.value_lower).copied(),
                    Expression::QualifiedIdentifier(qi) => {
                        // Try both qualified and unqualified names
                        let full_name =
                            format!("{}.{}", qi.qualifier, qi.name.value).to_lowercase();
                        col_index_map
                            .get(&full_name)
                            .or_else(|| col_index_map.get(&qi.name.value_lower))
                            .copied()
                    }
                    Expression::IntegerLiteral(lit) => {
                        // ORDER BY 1, 2, etc. - 1-based column position
                        let pos = lit.value as usize;
                        if pos > 0 && pos <= columns.len() {
                            Some(pos - 1)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                (col_idx, ob.ascending, ob.nulls_first)
            })
            .collect();

        // Sort using the same comparison function as the main query executor
        rows.sort_by(|a, b| {
            for (col_idx, ascending, nulls_first) in &order_specs {
                if let Some(idx) = col_idx {
                    let a_val = a.get(*idx);
                    let b_val = b.get(*idx);

                    // Check if either value is NULL
                    let a_is_null = a_val.is_none() || a_val.map(|v| v.is_null()).unwrap_or(true);
                    let b_is_null = b_val.is_none() || b_val.map(|v| v.is_null()).unwrap_or(true);

                    // Handle NULL comparison
                    if a_is_null || b_is_null {
                        if a_is_null && b_is_null {
                            continue; // Both NULL, move to next column
                        }
                        // Default: NULLS LAST for ASC, NULLS FIRST for DESC
                        let nulls_come_first = nulls_first.unwrap_or(!*ascending);
                        let cmp = if a_is_null {
                            if nulls_come_first {
                                std::cmp::Ordering::Less
                            } else {
                                std::cmp::Ordering::Greater
                            }
                        } else if nulls_come_first {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        };
                        return cmp;
                    }

                    // Both non-NULL - normal comparison
                    let cmp = match (a_val, b_val) {
                        (Some(av), Some(bv)) => {
                            av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => std::cmp::Ordering::Equal,
                    };

                    let cmp = if !*ascending { cmp.reverse() } else { cmp };

                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(rows)
    }

    // =========================================================================
    // CTE INLINING OPTIMIZATION
    // =========================================================================

    /// Try to inline single-use, non-recursive CTEs as subqueries.
    /// Returns Some(rewritten_stmt) if all CTEs can be inlined, None otherwise.
    ///
    /// Benefits of inlining:
    /// - Preserves index access (materialized CTEs lose all indexes)
    /// - Enables LIMIT pushdown through subqueries
    /// - Avoids memory overhead of full CTE materialization
    fn try_inline_ctes(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
    ) -> Option<SelectStatement> {
        // Build map of CTE definitions
        let mut cte_defs: FxHashMap<String, &CommonTableExpression> = FxHashMap::default();
        for cte in &with_clause.ctes {
            // Skip recursive CTEs - they must be materialized
            if cte.is_recursive {
                return None;
            }
            // Skip CTEs with column aliases - these rename the output columns
            // and the inlined subquery wouldn't preserve the aliases
            if !cte.column_names.is_empty() {
                return None;
            }
            cte_defs.insert(cte.name.value.to_lowercase(), cte);
        }

        // Check if any CTE references another CTE (CTE chaining)
        // These cannot be simply inlined as they have data dependencies
        for cte in &with_clause.ctes {
            for other_cte_name in cte_defs.keys() {
                if self.query_references_cte(&cte.query, other_cte_name) {
                    // CTE references another CTE - can't inline
                    return None;
                }
            }
        }

        // Count CTE references in the main query
        // Separate counts for table expressions (JOIN targets) vs WHERE clause subqueries
        let mut table_ref_counts: FxHashMap<String, usize> = FxHashMap::default();
        let mut where_ref_counts: FxHashMap<String, usize> = FxHashMap::default();
        for name in cte_defs.keys() {
            table_ref_counts.insert(name.clone(), 0);
            where_ref_counts.insert(name.clone(), 0);
        }

        // Count references in table expression (FROM/JOIN)
        if let Some(ref table_expr) = stmt.table_expr {
            self.count_cte_references_in_expr(table_expr, &mut table_ref_counts);
        }

        // Count references in WHERE clause (including IN/EXISTS subqueries)
        if let Some(ref where_clause) = stmt.where_clause {
            self.count_cte_references_in_expr(where_clause, &mut where_ref_counts);
        }

        // Only inline CTEs that:
        // 1. Are used exactly once in table expressions (FROM/JOIN)
        // 2. Are NOT used in WHERE clause subqueries (these need special handling)
        for name in cte_defs.keys() {
            let table_refs = table_ref_counts.get(name).copied().unwrap_or(0);
            let where_refs = where_ref_counts.get(name).copied().unwrap_or(0);

            // Skip if used in WHERE clause - subquery handling is different
            if where_refs > 0 {
                return None;
            }

            // Skip if used more than once - multi-use benefits from materialization
            if table_refs > 1 {
                return None;
            }
            // table_refs == 0 (unused) or table_refs == 1 (single-use) can be inlined
        }

        // All CTEs are single-use - perform inlining
        let mut new_stmt = stmt.clone();
        new_stmt.with = None; // Remove WITH clause

        // Replace CTE references with subqueries in the main query
        if let Some(ref table_expr) = new_stmt.table_expr {
            let inlined_expr = self.inline_cte_references(table_expr, &cte_defs);
            new_stmt.table_expr = Some(Box::new(inlined_expr));
        }

        Some(new_stmt)
    }

    /// Check if a query references a specific CTE by name
    fn query_references_cte(&self, stmt: &SelectStatement, cte_name: &str) -> bool {
        // Check table expression
        if let Some(ref table_expr) = stmt.table_expr {
            if self.expr_references_cte(table_expr, cte_name) {
                return true;
            }
        }

        // Check WHERE clause
        if let Some(ref where_clause) = stmt.where_clause {
            if self.expr_references_cte(where_clause, cte_name) {
                return true;
            }
        }

        false
    }

    /// Check if an expression references a CTE
    fn expr_references_cte(&self, expr: &Expression, cte_name: &str) -> bool {
        match expr {
            Expression::CteReference(cte_ref) => cte_ref.name.value.eq_ignore_ascii_case(cte_name),
            Expression::TableSource(ts) => ts.name.value.eq_ignore_ascii_case(cte_name),
            Expression::Identifier(id) => id.value.eq_ignore_ascii_case(cte_name),
            Expression::JoinSource(js) => {
                self.expr_references_cte(&js.left, cte_name)
                    || self.expr_references_cte(&js.right, cte_name)
            }
            Expression::SubquerySource(sq) => self.query_references_cte(&sq.subquery, cte_name),
            Expression::ScalarSubquery(sq) => self.query_references_cte(&sq.subquery, cte_name),
            Expression::In(in_expr) => {
                // Check if right side is a ScalarSubquery
                if let Expression::ScalarSubquery(sq) = &*in_expr.right {
                    self.query_references_cte(&sq.subquery, cte_name)
                } else {
                    false
                }
            }
            Expression::Exists(ex) => self.query_references_cte(&ex.subquery, cte_name),
            Expression::Infix(infix) => {
                self.expr_references_cte(&infix.left, cte_name)
                    || self.expr_references_cte(&infix.right, cte_name)
            }
            _ => false,
        }
    }

    /// Count CTE references in a SELECT statement
    fn count_cte_references_in_stmt(
        &self,
        stmt: &SelectStatement,
        ref_counts: &mut FxHashMap<String, usize>,
    ) {
        // Check table expression
        if let Some(ref table_expr) = stmt.table_expr {
            self.count_cte_references_in_expr(table_expr, ref_counts);
        }

        // Check WHERE clause
        if let Some(ref where_clause) = stmt.where_clause {
            self.count_cte_references_in_expr(where_clause, ref_counts);
        }

        // Check SELECT columns for subqueries
        for col in &stmt.columns {
            self.count_cte_references_in_expr(col, ref_counts);
        }
    }

    /// Count CTE references in an expression
    fn count_cte_references_in_expr(
        &self,
        expr: &Expression,
        ref_counts: &mut FxHashMap<String, usize>,
    ) {
        match expr {
            Expression::CteReference(cte_ref) => {
                let name = cte_ref.name.value.to_lowercase();
                if let Some(count) = ref_counts.get_mut(&name) {
                    *count += 1;
                }
            }
            Expression::TableSource(ts) => {
                let name = ts.name.value.to_lowercase();
                if let Some(count) = ref_counts.get_mut(&name) {
                    *count += 1;
                }
            }
            Expression::Identifier(id) => {
                let name = id.value.to_lowercase();
                if let Some(count) = ref_counts.get_mut(&name) {
                    *count += 1;
                }
            }
            Expression::JoinSource(js) => {
                self.count_cte_references_in_expr(&js.left, ref_counts);
                self.count_cte_references_in_expr(&js.right, ref_counts);
            }
            Expression::SubquerySource(sq) => {
                self.count_cte_references_in_stmt(&sq.subquery, ref_counts);
            }
            Expression::ScalarSubquery(sq) => {
                self.count_cte_references_in_stmt(&sq.subquery, ref_counts);
            }
            Expression::In(in_expr) => {
                self.count_cte_references_in_expr(&in_expr.left, ref_counts);
                // Check if right side is a ScalarSubquery
                if let Expression::ScalarSubquery(sq) = &*in_expr.right {
                    self.count_cte_references_in_stmt(&sq.subquery, ref_counts);
                }
            }
            Expression::Exists(ex) => {
                self.count_cte_references_in_stmt(&ex.subquery, ref_counts);
            }
            Expression::Infix(infix) => {
                self.count_cte_references_in_expr(&infix.left, ref_counts);
                self.count_cte_references_in_expr(&infix.right, ref_counts);
            }
            Expression::Aliased(aliased) => {
                self.count_cte_references_in_expr(&aliased.expression, ref_counts);
            }
            _ => {}
        }
    }

    /// Replace CTE references with subqueries in an expression
    fn inline_cte_references(
        &self,
        expr: &Expression,
        cte_defs: &FxHashMap<String, &CommonTableExpression>,
    ) -> Expression {
        match expr {
            Expression::CteReference(cte_ref) => {
                let name = cte_ref.name.value.to_lowercase();
                if let Some(cte) = cte_defs.get(&name) {
                    // Convert CTE to SubquerySource
                    let alias = cte_ref
                        .alias
                        .clone()
                        .unwrap_or_else(|| cte_ref.name.clone());
                    Expression::SubquerySource(SubqueryTableSource {
                        token: Token::new(TokenType::Punctuator, "(", Position::new(0, 0, 0)),
                        subquery: cte.query.clone(),
                        alias: Some(alias),
                    })
                } else {
                    expr.clone()
                }
            }
            Expression::TableSource(ts) => {
                let name = ts.name.value.to_lowercase();
                if let Some(cte) = cte_defs.get(&name) {
                    // Convert to SubquerySource preserving alias
                    let alias = ts.alias.clone().unwrap_or_else(|| ts.name.clone());
                    Expression::SubquerySource(SubqueryTableSource {
                        token: Token::new(TokenType::Punctuator, "(", Position::new(0, 0, 0)),
                        subquery: cte.query.clone(),
                        alias: Some(alias),
                    })
                } else {
                    expr.clone()
                }
            }
            Expression::JoinSource(js) => {
                let left = self.inline_cte_references(&js.left, cte_defs);
                let right = self.inline_cte_references(&js.right, cte_defs);
                Expression::JoinSource(Box::new(JoinTableSource {
                    token: js.token.clone(),
                    left: Box::new(left),
                    right: Box::new(right),
                    join_type: js.join_type.clone(),
                    condition: js.condition.clone(),
                    using_columns: js.using_columns.clone(),
                }))
            }
            Expression::SubquerySource(sq) => {
                // Recursively inline CTEs in nested subqueries
                let mut new_subquery = (*sq.subquery).clone();
                if let Some(ref table_expr) = new_subquery.table_expr {
                    let inlined = self.inline_cte_references(table_expr, cte_defs);
                    new_subquery.table_expr = Some(Box::new(inlined));
                }
                Expression::SubquerySource(SubqueryTableSource {
                    token: sq.token.clone(),
                    subquery: Box::new(new_subquery),
                    alias: sq.alias.clone(),
                })
            }
            _ => expr.clone(),
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
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER)")
            .unwrap();
        executor
            .execute("INSERT INTO products VALUES (1, 'Widget', 10)")
            .unwrap();
        executor
            .execute("INSERT INTO products VALUES (2, 'Gadget', 20)")
            .unwrap();
        executor
            .execute("INSERT INTO products VALUES (3, 'Doodad', 15)")
            .unwrap();
    }

    #[test]
    fn test_cte_registry() {
        let mut registry = CteRegistry::new();

        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![Row::from_values(vec![
            Value::Integer(1),
            Value::text("test"),
        ])];

        registry.store("my_cte", columns.clone(), rows.clone());

        assert!(registry.exists("my_cte"));
        assert!(registry.exists("MY_CTE")); // case-insensitive

        let (cols, retrieved_rows) = registry.get("my_cte").unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(retrieved_rows.len(), 1);
    }

    #[test]
    fn test_simple_cte() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("WITH expensive AS (SELECT * FROM products WHERE price > 12) SELECT * FROM expensive")
            .unwrap();

        let columns = result.columns();
        assert_eq!(columns.len(), 3);

        let mut count = 0;
        while result.next() {
            count += 1;
        }
        // Should have 2 products with price > 12 (Gadget=20, Doodad=15)
        assert_eq!(count, 2);
    }

    #[test]
    fn test_cte_with_aggregation() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute(
                "WITH all_products AS (SELECT * FROM products) SELECT COUNT(*) FROM all_products",
            )
            .unwrap();

        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(3)));
    }

    #[test]
    fn test_has_cte() {
        let executor = create_test_executor();

        // With CTE
        let mut parser = crate::parser::Parser::new("WITH x AS (SELECT 1) SELECT * FROM x");
        if let Ok(program) = parser.parse_program() {
            if let Statement::Select(stmt) = &program.statements[0] {
                assert!(executor.has_cte(stmt));
            }
        }

        // Without CTE
        let mut parser2 = crate::parser::Parser::new("SELECT * FROM test");
        if let Ok(program) = parser2.parse_program() {
            if let Statement::Select(stmt) = &program.statements[0] {
                assert!(!executor.has_cte(stmt));
            }
        }
    }
}
