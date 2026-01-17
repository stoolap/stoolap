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

use crate::common::SmartString;
use ahash::AHashSet;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::common::{CompactArc, CompactVec};
use crate::core::{Error, Result, Row, RowVec, Value};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::traits::{Engine, QueryResult, Table, Transaction};

use super::context::ExecutionContext;
use super::expression::ExpressionEval;
use super::join_executor::{JoinExecutor, JoinRequest};
use super::operator::{ColumnInfo, Operator, QueryResultOperator};
use super::operators::hash_join::JoinType as OperatorJoinType;
use super::operators::index_nested_loop::{IndexLookupStrategy, IndexNestedLoopJoinOperator};
use super::query_classification::get_classification;
use super::result::ExecutorResult;
use super::utils::{build_column_index_map, extract_join_keys_and_residual, is_sorted_on_keys};
use super::Executor;

/// Type alias for CTE data: (columns, rows) with Arc for zero-copy sharing
/// Uses Vec<(i64, Row)> for rows - same structure as RowVec but Arc-shareable
pub type CteData = (CompactArc<Vec<String>>, CompactArc<Vec<(i64, Row)>>);

/// Type alias for CTE data map
/// Uses CompactArc<Vec<String>> for columns and CompactArc<Vec<(i64, Row)>> for rows
/// to enable zero-copy sharing of CTE results with joins
pub type CteDataMap = FxHashMap<String, CteData>;

/// Type alias for join data result (left_columns, left_rows, right_columns, right_rows)
/// Uses Arc for zero-copy sharing of CTE results with join operators
type JoinDataResult = (
    CompactArc<Vec<String>>,
    CompactArc<Vec<(i64, Row)>>,
    CompactArc<Vec<String>>,
    CompactArc<Vec<(i64, Row)>>,
);

/// Convert an expression to a normalized lowercase string for ORDER BY matching.
/// Handles nested function calls, multiple arguments, and various expression types.
fn expr_to_normalized_string(expr: &Expression) -> String {
    match expr {
        Expression::Identifier(id) => id.value_lower.to_string(),
        Expression::QualifiedIdentifier(qi) => {
            format!("{}.{}", qi.qualifier.value_lower, qi.name.value_lower)
        }
        Expression::FunctionCall(fc) => {
            let func_name = fc.function.to_lowercase();
            if fc.arguments.is_empty() {
                // Handle COUNT(*) style
                format!("{}(*)", func_name)
            } else {
                // Handle all arguments recursively
                let args: Vec<String> =
                    fc.arguments.iter().map(expr_to_normalized_string).collect();
                format!("{}({})", func_name, args.join(", "))
            }
        }
        Expression::IntegerLiteral(lit) => lit.value.to_string(),
        Expression::FloatLiteral(lit) => lit.value.to_string(),
        Expression::StringLiteral(lit) => format!("'{}'", lit.value),
        Expression::Infix(infix) => {
            // Handle expressions like col + 1, a * b
            format!(
                "{} {} {}",
                expr_to_normalized_string(&infix.left),
                infix.operator.to_lowercase(),
                expr_to_normalized_string(&infix.right)
            )
        }
        Expression::Prefix(prefix) => {
            format!(
                "{}{}",
                prefix.operator.to_lowercase(),
                expr_to_normalized_string(&prefix.right)
            )
        }
        Expression::Star(_) => "*".to_string(),
        // Default: convert to debug string and lowercase
        _ => format!("{:?}", expr).to_lowercase(),
    }
}

/// Hint for CTE optimization including LIMIT and optional ORDER BY pushdown
#[derive(Clone)]
struct CtePushdownHint {
    limit: u64,
    order_by: Vec<OrderByExpression>,
}

/// Registry for CTE results during query execution
///
/// Uses Arc with COW (copy-on-write) semantics to avoid cloning:
/// - During building: `Arc::make_mut` gives mutable access without cloning (single owner)
/// - During sharing: `data()` returns cheap Arc clone (O(1), no data copy)
#[derive(Clone)]
pub struct CteRegistry {
    /// Materialized CTE results (name -> (columns, rows))
    /// Arc provides cheap sharing; make_mut provides COW for modifications
    data: Arc<CteDataMap>,
}

impl Default for CteRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CteRegistry {
    /// Create a new CTE registry
    pub fn new() -> Self {
        Self {
            data: Arc::new(FxHashMap::default()),
        }
    }

    /// Store a materialized CTE result
    ///
    /// Uses Arc::make_mut for COW semantics:
    /// - If we're the only owner, mutates in place (no clone)
    /// - If shared, clones first then mutates (preserves other references)
    ///
    /// Both columns and rows are wrapped in Arc to enable zero-copy sharing.
    /// Accepts RowVec and converts to CompactArc<Vec<(i64, Row)>> for sharing.
    pub fn store(&mut self, name: &str, columns: Vec<String>, rows: RowVec) {
        let name_lower = name.to_lowercase();
        // Convert RowVec to Vec<(i64, Row)> for Arc sharing
        let rows_vec: Vec<(i64, Row)> = rows.into_iter().collect();
        Arc::make_mut(&mut self.data).insert(
            name_lower,
            (CompactArc::new(columns), CompactArc::new(rows_vec)),
        );
    }

    /// Store a materialized CTE result with pre-wrapped Arcs
    ///
    /// Use this when you already have Arc-wrapped data to avoid cloning.
    /// This enables zero-copy sharing of CTE results between queries.
    pub fn store_arc(
        &mut self,
        name: &str,
        columns: CompactArc<Vec<String>>,
        rows: CompactArc<Vec<(i64, Row)>>,
    ) {
        let name_lower = name.to_lowercase();
        Arc::make_mut(&mut self.data).insert(name_lower, (columns, rows));
    }

    /// Get a CTE result by name
    /// Returns Arc references to enable zero-copy sharing with joins
    pub fn get(&self, name: &str) -> Option<&CteData> {
        let name_lower = name.to_lowercase();
        self.data.get(&name_lower)
    }

    /// Check if a CTE exists
    #[allow(dead_code)]
    pub fn exists(&self, name: &str) -> bool {
        self.data.contains_key(&name.to_lowercase())
    }

    /// Get a shared Arc reference to the internal data map for context transfer
    ///
    /// This is always O(1) - just an Arc reference count increment.
    /// No data cloning ever happens here.
    pub fn data(&self) -> Arc<CteDataMap> {
        self.data.clone()
    }

    /// Iterate over all stored CTEs (for copying to temp registries)
    pub fn iter(&self) -> impl Iterator<Item = (&String, &CteData)> {
        self.data.iter()
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

        // LAZY CTE OPTIMIZATION:
        // For single-use CTEs in INNER JOIN with LIMIT, push LIMIT into CTE
        // This enables streaming GROUP BY early termination
        let cte_limit_hints = self.compute_cte_limit_hints(stmt, with_clause);

        // Create CTE registry
        let mut cte_registry = CteRegistry::new();

        // Execute each CTE in order
        for cte in &with_clause.ctes {
            // Check if we have a pushdown hint for this CTE
            let cte_name_lower: String = cte.name.value_lower.to_string();
            let pushdown_hint = cte_limit_hints.get(&cte_name_lower);

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
            } else if let Some(hint) = pushdown_hint {
                // Execute CTE with pushed-down LIMIT and optional ORDER BY
                self.execute_cte_query_with_hint(&cte.query, ctx, &mut cte_registry, hint)?
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
                            alias.value.to_string()
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
    ) -> Result<(Vec<String>, RowVec)> {
        // Check if the CTE references another CTE
        if let Some(ref table_expr) = stmt.table_expr {
            // First check for simple CTE reference
            // BUT: if there are set operations (UNION, etc.), we need to fall through
            // to execute_select which handles them properly
            if stmt.set_operations.is_empty() {
                if let Some(cte_name) = self.extract_cte_name_for_lookup(table_expr) {
                    if let Some((columns, rows)) = cte_registry.get(&cte_name) {
                        // Execute query against CTE result
                        // Clone Arc data into RowVec for processing
                        return self.execute_query_on_cte_result(
                            stmt,
                            ctx,
                            columns.to_vec(),
                            RowVec::from_vec((**rows).clone()),
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
    ) -> Result<(Vec<String>, RowVec)> {
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
                        alias.value.to_string()
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
            // Copy existing CTEs - share Arc to avoid cloning row data
            for (name, (cols, rows)) in cte_registry.iter() {
                temp_registry.store_arc(name, cols.clone(), CompactArc::clone(rows));
            }
            // Add current CTE with working rows
            temp_registry.store(cte_name, columns.clone(), working_rows.clone());

            // Execute each recursive member
            let mut new_rows = RowVec::new();
            for set_op in &stmt.set_operations {
                // The recursive member is in set_op.right
                let recursive_result =
                    self.execute_cte_query(&set_op.right, ctx, &mut temp_registry)?;

                // Extend with rows from recursive result, renumbering row IDs
                let base_id = new_rows.len() as i64;
                for (i, (_, row)) in recursive_result.1.into_iter().enumerate() {
                    new_rows.push((base_id + i as i64, row));
                }
            }

            if new_rows.is_empty() {
                break;
            }

            // Add new rows to total result, renumbering row IDs
            let base_id = all_rows.len() as i64;
            for (i, (_, row)) in new_rows.clone().into_iter().enumerate() {
                all_rows.push((base_id + i as i64, row));
            }

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
                        // Clone Arc data into RowVec for processing
                        let (result_cols, result_rows) = self.execute_query_on_cte_result(
                            stmt,
                            &ctx_with_ctes,
                            columns.to_vec(),
                            RowVec::from_vec((**rows).clone()),
                        )?;
                        return Ok(Box::new(ExecutorResult::new(result_cols, result_rows)));
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
        // OPTIMIZATION: Get cached query classification to avoid repeated AST traversals
        let classification = get_classification(stmt);

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

        // INDEX NESTED LOOP: Try Index Nested Loop first when one side is CTE and other is table.
        // This is much faster than hash join when the table has PK/index on join key because
        // we can do O(N * log M) lookups instead of O(N + M) hash build + probe.
        if is_inner_join && left_is_cte != right_is_cte {
            // Determine which side is CTE and which is table
            let (cte_data, table_expr, cte_on_left) = if left_is_cte {
                (left_data.clone().unwrap(), &join_source.right, true)
            } else {
                (right_data.clone().unwrap(), &join_source.left, false)
            };

            // Calculate effective limit for Index Nested Loop (same logic as later)
            let can_push_limit = !classification.has_order_by
                && !classification.has_aggregation
                && !classification.has_window_functions
                && stmt.where_clause.is_none();

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

            // Get aliases for qualifier matching
            let cte_alias = if cte_on_left {
                left_name.clone()
            } else {
                right_name.clone()
            };
            let table_alias = if cte_on_left {
                right_name.clone()
            } else {
                left_name.clone()
            };

            // Try Index Nested Loop - returns early if successful
            if let Some(result) = self.try_cte_index_nested_loop_join(
                stmt,
                ctx,
                join_source,
                cte_data,
                table_expr,
                cte_on_left,
                cte_alias,
                table_alias,
                effective_limit,
            )? {
                return Ok(Some(result));
            }
        }

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
            let (left_columns, left_rows): CteData = match left_data {
                Some(data) => data,
                None => {
                    let (result, cols) = self.execute_table_expression(&join_source.left, ctx)?;
                    let rows = Self::materialize_result(result)?;
                    // Convert RowVec to Vec<(i64, Row)> for Arc sharing
                    let rows_vec: Vec<(i64, Row)> = rows.into_iter().collect();
                    (CompactArc::new(cols), CompactArc::new(rows_vec))
                }
            };

            let (right_columns, right_rows): CteData = match right_data {
                Some(data) => data,
                None => {
                    let (result, cols) = self.execute_table_expression(&join_source.right, ctx)?;
                    let rows = Self::materialize_result(result)?;
                    // Convert RowVec to Vec<(i64, Row)> for Arc sharing
                    let rows_vec: Vec<(i64, Row)> = rows.into_iter().collect();
                    (CompactArc::new(cols), CompactArc::new(rows_vec))
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
                left_columns.to_vec()
            }
        } else {
            // Already prefixed by execute_table_expression
            left_columns.to_vec()
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
                right_columns.to_vec()
            }
        } else {
            // Already prefixed by execute_table_expression
            right_columns.to_vec()
        };
        all_columns.extend(right_qualified.clone());

        // Execute the JOIN
        let join_type = join_source.join_type.to_uppercase();

        // Determine if we can safely push LIMIT down to the JOIN
        // Safe conditions: INNER/CROSS JOIN, no ORDER BY, no aggregation, no window functions
        let can_push_limit = (join_type == "INNER" || join_type == "CROSS" || join_type.is_empty())
            && !classification.has_order_by
            && !classification.has_aggregation
            && !classification.has_window_functions
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

        // Get join algorithm decision from QueryPlanner (same as query.rs)
        let (left_key_indices, right_key_indices, _) =
            if let Some(cond) = join_source.condition.as_ref() {
                extract_join_keys_and_residual(cond, &left_qualified, &right_qualified)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
        let has_equality_keys = !left_key_indices.is_empty();

        // Extract just the Row part for sort checking (JoinExecutor works with Row, not (i64, Row))
        let left_rows_only: Vec<Row> = left_rows.iter().map(|(_, row)| row.clone()).collect();
        let right_rows_only: Vec<Row> = right_rows.iter().map(|(_, row)| row.clone()).collect();

        // Check if inputs are sorted on join keys
        let left_sorted =
            has_equality_keys && is_sorted_on_keys(&left_rows_only, &left_key_indices);
        let right_sorted =
            has_equality_keys && is_sorted_on_keys(&right_rows_only, &right_key_indices);

        // Get algorithm decision from QueryPlanner
        let algorithm_decision = self.get_query_planner().plan_runtime_join_with_sort_info(
            left_rows_only.len(),
            right_rows_only.len(),
            has_equality_keys,
            left_sorted,
            right_sorted,
        );

        // Use JoinExecutor for consistent join execution
        // Passes CompactArc<Vec<Row>> (without row IDs) to JoinExecutor
        let join_executor = JoinExecutor::new();
        let join_result = join_executor.execute(JoinRequest {
            left_rows: CompactArc::new(left_rows_only),
            right_rows: CompactArc::new(right_rows_only),
            left_columns: &left_qualified,
            right_columns: &right_qualified,
            condition: join_source.condition.as_deref(),
            join_type: &join_type,
            limit: effective_limit,
            ctx,
            algorithm_hint: Some(&algorithm_decision),
        })?;
        let result_rows = join_result.rows;

        // Apply WHERE clause if present (result_rows is RowVec with (row_id, Row) tuples)
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            let mut where_eval =
                ExpressionEval::compile(where_clause, &all_columns)?.with_context(ctx);

            let mut rows = RowVec::with_capacity(result_rows.len());
            for (row_id, row) in result_rows {
                if where_eval.eval_bool(&row) {
                    rows.push((row_id, row));
                }
            }
            rows
        } else {
            result_rows
        };

        // Check for window functions (must handle before projection)
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &filtered_rows, &all_columns)?;
            return Ok(Some(result));
        }

        // Check for aggregation (must handle before projection)
        if classification.has_aggregation {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &all_columns)?;

            // Apply ORDER BY if present (aggregation doesn't handle it for CTE JOINs)
            if !stmt.order_by.is_empty() {
                let output_columns = result.columns().to_vec();
                let rows = Self::materialize_result(result)?;
                let sorted_rows =
                    self.apply_order_by_to_rows(rows, &stmt.order_by, &output_columns)?;
                return Ok(Some(Box::new(ExecutorResult::new(
                    output_columns,
                    sorted_rows,
                ))));
            }
            return Ok(Some(result));
        }

        // Project rows according to SELECT expressions
        let projected_rows = self.project_rows_with_alias(
            &stmt.columns,
            filtered_rows,
            &all_columns,
            None,
            ctx,
            None,
        )?;

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

        Ok(Some(Box::new(ExecutorResult::new(
            output_columns,
            final_rows,
        ))))
    }

    /// Resolve a table expression - returns Some if it's a CTE, None if it's a regular table
    /// Returns Arc-wrapped columns and rows to enable zero-copy sharing with join operators
    fn resolve_table_or_cte(
        &self,
        expr: &Expression,
        _ctx: &ExecutionContext,
        cte_registry: &CteRegistry,
    ) -> Result<Option<CteData>> {
        // Use the base CTE name (not alias) for registry lookup
        if let Some(cte_name) = self.extract_cte_name_for_lookup(expr) {
            if let Some((columns, rows)) = cte_registry.get(&cte_name) {
                // Return CompactArc::clone/Arc::clone for zero-copy sharing
                return Ok(Some((CompactArc::clone(columns), CompactArc::clone(rows))));
            }
        }
        Ok(None)
    }

    /// Execute CTE JOIN with semi-join reduction optimization.
    /// When one side is a CTE and the other is a regular table, extract join keys from the CTE
    /// and use them to filter the regular table scan via IN clause (which can use indexes).
    /// Accepts CompactArc<Vec<Row>> to enable zero-copy sharing of CTE results.
    #[allow(clippy::too_many_arguments)]
    fn execute_cte_join_with_semijoin_reduction(
        &self,
        left_expr: &Expression,
        right_expr: &Expression,
        join_condition: Option<&Expression>,
        left_data: Option<CteData>,
        right_data: Option<CteData>,
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
                // Find CTE column index using case-insensitive comparison (no allocation)
                let cte_col_lower = cte_col.to_lowercase();
                let cte_col_idx = cte_columns
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case(&cte_col_lower));

                if let Some(idx) = cte_col_idx {
                    // Extract distinct non-NULL values from CTE
                    let mut seen: AHashSet<Value> = AHashSet::with_capacity(cte_rows.len());

                    // Iterate over CompactArc<Vec<(i64, Row)>> by dereferencing
                    for (_, row) in cte_rows.iter() {
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
        // Wrap materialized result in Arc for consistency with CTE data
        let table_rows_vec: Vec<(i64, Row)> = Self::materialize_result(table_result)?
            .into_iter()
            .collect();
        let table_rows = CompactArc::new(table_rows_vec);

        // Return in correct order (left, right)
        let table_cols = CompactArc::new(table_cols);
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
            Expression::Identifier(id) => Some((None, id.value.to_string())),
            Expression::QualifiedIdentifier(qi) => Some((
                Some(qi.qualifier.value.to_string()),
                qi.name.value.to_string(),
            )),
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
        cte_rows: RowVec,
    ) -> Result<(Vec<String>, RowVec)> {
        // OPTIMIZATION: Get cached query classification to avoid repeated AST traversals
        let classification = get_classification(stmt);

        // Apply WHERE clause filter
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            // Process subqueries in WHERE clause (e.g., IN subqueries on CTEs)
            // Use cached classification to avoid AST traversal
            let processed_where = if classification.where_has_subqueries {
                self.process_where_subqueries(where_clause, ctx)?
            } else {
                (**where_clause).clone()
            };

            // Compile filter once and reuse for all rows
            let mut eval =
                ExpressionEval::compile(&processed_where, &cte_columns)?.with_context(ctx);

            let mut result = RowVec::new();
            let mut row_id = 0i64;
            for (_, row) in cte_rows {
                if eval.eval_bool(&row) {
                    result.push((row_id, row));
                    row_id += 1;
                }
            }
            result
        } else {
            cte_rows
        };

        // Check for aggregation
        if classification.has_aggregation {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &cte_columns)?;
            let columns = result.columns().to_vec();
            let rows = Self::materialize_result(result)?;

            return Ok((columns, rows));
        }

        // Check for window functions
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &filtered_rows, &cte_columns)?;
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
        if classification.has_order_by {
            result_rows =
                self.apply_order_by_to_rows(result_rows, &stmt.order_by, &output_columns)?;
        }

        // Apply LIMIT and OFFSET (using classification for quick check)
        if classification.has_offset {
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
        }

        if classification.has_limit {
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
        }

        Ok((output_columns, result_rows))
    }

    /// Extract the base CTE/table name for registry lookup (ignores aliases)
    fn extract_cte_name_for_lookup(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::CteReference(cte_ref) => Some(cte_ref.name.value.to_string()),
            Expression::TableSource(simple_table_source) => {
                Some(simple_table_source.name.value.to_string())
            }
            Expression::Identifier(id) => Some(id.value.to_string()),
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
                    Some(alias.value.to_string())
                } else {
                    Some(cte_ref.name.value.to_string())
                }
            }
            Expression::TableSource(simple_table_source) => {
                // Use alias if present, otherwise use the table name
                if let Some(ref alias) = simple_table_source.alias {
                    Some(alias.value.to_string())
                } else {
                    Some(simple_table_source.name.value.to_string())
                }
            }
            Expression::Identifier(id) => Some(id.value.to_string()),
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
                    output_columns.push(id.value.to_string());
                }
                Expression::Aliased(aliased) => {
                    output_columns.push(aliased.alias.value.to_string());
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
        rows: &RowVec,
        cte_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
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
                    let idx = col_index_map.get(id.value_lower.as_str()).copied();
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
        let mut result_rows = RowVec::with_capacity(rows.len());

        for (row_id, (_, row)) in rows.iter().enumerate() {
            // OPTIMIZATION: Pre-allocate CompactVec with estimated capacity
            let mut values: CompactVec<Value> =
                CompactVec::with_capacity(columns.len().max(row.len()));
            // CRITICAL: Include params from context for parameterized queries
            let exec_ctx = ExecuteContext::new(row)
                .with_params(ctx.params())
                .with_named_params(ctx.named_params());

            for compiled in &compiled_columns {
                match compiled {
                    CompiledColumn::Star => {
                        // OPTIMIZATION: Extend with row values
                        values.extend(row.iter().cloned());
                    }
                    CompiledColumn::Identifier(Some(idx)) => {
                        values.push(row.get(*idx).cloned().unwrap_or_else(Value::null_unknown));
                    }
                    CompiledColumn::Identifier(None) => {
                        values.push(Value::null_unknown());
                    }
                    CompiledColumn::Compiled(program) => {
                        values.push(vm.execute_cow(program, &exec_ctx)?);
                    }
                }
            }

            result_rows.push((row_id as i64, Row::from_compact_vec(values)));
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
        mut rows: RowVec,
        order_by: &[crate::parser::ast::OrderByExpression],
        columns: &[String],
    ) -> Result<RowVec> {
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
                    Expression::Identifier(id) => {
                        col_index_map.get(id.value_lower.as_str()).copied()
                    }
                    Expression::QualifiedIdentifier(qi) => {
                        // Try both qualified and unqualified names
                        let full_name =
                            format!("{}.{}", qi.qualifier, qi.name.value).to_lowercase();
                        col_index_map
                            .get(&full_name)
                            .or_else(|| col_index_map.get(qi.name.value_lower.as_str()))
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
        // RowVec derefs to Vec<(i64, Row)>, so we sort by the Row part
        rows.sort_by(|(_, a), (_, b)| {
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

    /// Check if LIMIT pushdown would be more beneficial than CTE inlining.
    ///
    /// Returns true when streaming aggregation with limit pushdown will be faster
    /// than inlining as a subquery.
    fn should_use_limit_pushdown_instead(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
    ) -> bool {
        // Must have LIMIT without ORDER BY
        if stmt.limit.is_none() || !stmt.order_by.is_empty() {
            return false;
        }

        // Must have a JOIN
        let join_source = match &stmt.table_expr {
            Some(expr) => match expr.as_ref() {
                Expression::JoinSource(js) => js,
                _ => return false,
            },
            None => return false,
        };

        let join_type = join_source.join_type.to_uppercase();
        let is_inner_join = join_type == "INNER" || join_type.is_empty() || join_type == "JOIN";
        let is_left_join = join_type == "LEFT" || join_type == "LEFT OUTER";
        let is_right_join = join_type == "RIGHT" || join_type == "RIGHT OUTER";

        if !is_inner_join && !is_left_join && !is_right_join {
            return false;
        }

        // Check if exactly one side is a CTE with GROUP BY
        let cte_names: AHashSet<String> = with_clause
            .ctes
            .iter()
            .filter(|c| !c.is_recursive && !c.query.group_by.columns.is_empty())
            .map(|c| c.name.value_lower.to_string())
            .collect();

        if cte_names.is_empty() {
            return false;
        }

        let left_cte = self
            .extract_cte_name_for_lookup(&join_source.left)
            .filter(|n| cte_names.contains(&n.to_lowercase()));
        let right_cte = self
            .extract_cte_name_for_lookup(&join_source.right)
            .filter(|n| cte_names.contains(&n.to_lowercase()));

        // For INNER JOIN: either side can be CTE
        // For LEFT JOIN: CTE must be on the RIGHT (each CTE row produces at most one result)
        // For RIGHT JOIN: CTE must be on the LEFT (each CTE row produces at most one result)
        match (&left_cte, &right_cte) {
            (Some(_), None) if is_inner_join || is_right_join => true,
            (None, Some(_)) if is_inner_join || is_left_join => true,
            _ => false,
        }
    }

    /// Try to inline single-use, non-recursive CTEs as subqueries.
    /// Returns Some(rewritten_stmt) if all CTEs can be inlined, None otherwise.
    ///
    /// Benefits of inlining:
    /// - Preserves index access (materialized CTEs lose all indexes)
    /// - Enables LIMIT pushdown through subqueries
    /// - Avoids memory overhead of full CTE materialization
    pub(crate) fn try_inline_ctes(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
    ) -> Option<SelectStatement> {
        // Early exit: no table expression means nothing to inline
        let table_expr = stmt.table_expr.as_ref()?;

        // Skip inlining if LIMIT pushdown with streaming would be more beneficial.
        // This happens when:
        // 1. Main query has LIMIT (no ORDER BY)
        // 2. CTE has GROUP BY (can use streaming aggregation)
        // 3. CTE is in INNER JOIN (limit pushdown is safe)
        if self.should_use_limit_pushdown_instead(stmt, with_clause) {
            return None;
        }

        // Pre-compute lowercase CTE names once to avoid repeated to_lowercase() calls
        // Store (lowercase_name, original_cte) pairs
        let cte_names_lower: Vec<(String, &CommonTableExpression)> = with_clause
            .ctes
            .iter()
            .map(|cte| {
                // Early exit for conditions that prevent inlining
                if cte.is_recursive || !cte.column_names.is_empty() {
                    return Err(());
                }
                Ok((cte.name.value_lower.to_string(), cte))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?;

        // Build map from pre-computed lowercase names
        let cte_defs: FxHashMap<String, &CommonTableExpression> =
            cte_names_lower.iter().cloned().collect();
        let cte_name_set: AHashSet<&str> = cte_defs.keys().map(|s| s.as_str()).collect();

        // Check if any CTE references another CTE (CTE chaining)
        // These cannot be simply inlined as they have data dependencies
        for (_, cte) in &cte_names_lower {
            for other_cte_name in &cte_name_set {
                if self.query_references_cte(&cte.query, other_cte_name) {
                    // CTE references another CTE - can't inline
                    return None;
                }
            }
        }

        // Count CTE references in the main query using pre-computed names
        // Separate counts for table expressions (JOIN targets) vs WHERE clause subqueries
        let mut table_ref_counts: FxHashMap<String, usize> =
            cte_defs.keys().map(|name| (name.clone(), 0)).collect();
        let mut where_ref_counts: FxHashMap<String, usize> = table_ref_counts.clone();

        // Count references in table expression (FROM/JOIN)
        self.count_cte_references_in_expr(table_expr, &mut table_ref_counts);

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
        // OPTIMIZATION: Only clone if something actually changes
        // First check if any CTE references exist that we can inline
        let any_refs = table_ref_counts.values().any(|&count| count > 0);
        if !any_refs {
            // No CTE references in table expression - no inlining needed
            return None;
        }

        // Try to inline - if nothing changes, skip the expensive cloning
        let inlined_expr = self.try_inline_cte_references(table_expr, &cte_defs)?;

        Some(SelectStatement {
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            columns: stmt.columns.clone(),
            with: None, // Remove WITH clause
            table_expr: Some(Box::new(inlined_expr)),
            where_clause: stmt.where_clause.clone(),
            group_by: stmt.group_by.clone(),
            having: stmt.having.clone(),
            window_defs: stmt.window_defs.clone(),
            order_by: stmt.order_by.clone(),
            limit: stmt.limit.clone(),
            offset: stmt.offset.clone(),
            set_operations: stmt.set_operations.clone(),
        })
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
                let name: &str = cte_ref.name.value_lower.as_str();
                if let Some(count) = ref_counts.get_mut(name) {
                    *count += 1;
                }
            }
            Expression::TableSource(ts) => {
                let name: &str = ts.name.value_lower.as_str();
                if let Some(count) = ref_counts.get_mut(name) {
                    *count += 1;
                }
            }
            Expression::Identifier(id) => {
                let name: &str = id.value_lower.as_str();
                if let Some(count) = ref_counts.get_mut(name) {
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

    /// Replace CTE references with subqueries in an expression.
    /// Returns Some(new_expr) if any replacement was made, None if no changes needed.
    fn try_inline_cte_references(
        &self,
        expr: &Expression,
        cte_defs: &FxHashMap<String, &CommonTableExpression>,
    ) -> Option<Expression> {
        match expr {
            Expression::CteReference(cte_ref) => {
                // Use pre-computed lowercase from value_lower if available
                let name = &cte_ref.name.value_lower;
                cte_defs.get(name.as_str()).map(|cte| {
                    // Convert CTE to SubquerySource
                    let alias = cte_ref
                        .alias
                        .clone()
                        .unwrap_or_else(|| cte_ref.name.clone());
                    Expression::SubquerySource(Box::new(SubqueryTableSource {
                        token: Token::new(TokenType::Punctuator, "(", Position::new(0, 0, 0)),
                        subquery: cte.query.clone(),
                        alias: Some(alias),
                    }))
                })
            }
            Expression::TableSource(ts) => {
                // Use pre-computed lowercase from value_lower
                let name = &ts.name.value_lower;
                cte_defs.get(name.as_str()).map(|cte| {
                    // Convert to SubquerySource preserving alias
                    let alias = ts.alias.clone().unwrap_or_else(|| ts.name.clone());
                    Expression::SubquerySource(Box::new(SubqueryTableSource {
                        token: Token::new(TokenType::Punctuator, "(", Position::new(0, 0, 0)),
                        subquery: cte.query.clone(),
                        alias: Some(alias),
                    }))
                })
            }
            Expression::JoinSource(js) => {
                let left_changed = self.try_inline_cte_references(&js.left, cte_defs);
                let right_changed = self.try_inline_cte_references(&js.right, cte_defs);

                // Only create new JoinSource if something changed
                if left_changed.is_some() || right_changed.is_some() {
                    let left = left_changed.unwrap_or_else(|| (*js.left).clone());
                    let right = right_changed.unwrap_or_else(|| (*js.right).clone());
                    Some(Expression::JoinSource(Box::new(JoinTableSource {
                        token: js.token.clone(),
                        left: Box::new(left),
                        right: Box::new(right),
                        join_type: js.join_type.clone(),
                        condition: js.condition.clone(),
                        using_columns: js.using_columns.clone(),
                    })))
                } else {
                    None
                }
            }
            Expression::SubquerySource(sq) => {
                // Only recurse if there's a table_expr
                if let Some(ref table_expr) = sq.subquery.table_expr {
                    if let Some(inlined) = self.try_inline_cte_references(table_expr, cte_defs) {
                        let mut new_subquery = (*sq.subquery).clone();
                        new_subquery.table_expr = Some(Box::new(inlined));
                        return Some(Expression::SubquerySource(Box::new(SubqueryTableSource {
                            token: sq.token.clone(),
                            subquery: Box::new(new_subquery),
                            alias: sq.alias.clone(),
                        })));
                    }
                }
                None
            }
            _ => None, // No change needed
        }
    }

    /// Try to use Index Nested Loop for CTE + Table joins.
    ///
    /// When one side is a CTE (materialized) and the other is a regular table with
    /// a PK or index on the join key, Index Nested Loop is often faster than hash join
    /// because we can do direct lookups instead of building a hash table.
    ///
    /// Returns Some(result) if Index Nested Loop was used, None if not applicable.
    /// Accepts CompactArc<Vec<Row>> to enable zero-copy sharing of CTE results.
    #[allow(clippy::too_many_arguments)]
    fn try_cte_index_nested_loop_join(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        join_source: &JoinTableSource,
        cte_data: CteData,
        table_expr: &Expression,
        cte_on_left: bool,
        cte_alias: Option<String>,
        table_alias: Option<String>,
        effective_limit: Option<u64>,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        // OPTIMIZATION: Get cached query classification to avoid repeated AST traversals
        let classification = get_classification(stmt);

        let join_type = join_source.join_type.to_uppercase();

        // Only for INNER or LEFT joins (not RIGHT or FULL)
        // RIGHT/FULL would need to track all unmatched inner rows
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            return Ok(None);
        }

        // Need a join condition
        let condition = match join_source.condition.as_deref() {
            Some(c) => c,
            None => return Ok(None),
        };

        // Extract table name from table expression
        let table_name: String = match table_expr {
            Expression::TableSource(ts) => ts.name.value_lower.to_string(),
            Expression::Aliased(aliased) => match aliased.expression.as_ref() {
                Expression::TableSource(ts) => ts.name.value_lower.to_string(),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };

        // Extract join key columns using qualifier matching
        let (table_col, cte_col) = self.extract_join_key_columns_with_qualifiers(
            condition,
            cte_alias.as_deref(),
            table_alias.as_deref(),
        );

        let table_col = match table_col {
            Some(c) => c,
            None => return Ok(None),
        };
        let cte_col = match cte_col {
            Some(c) => c,
            None => return Ok(None),
        };

        // Get the table column (unqualified)
        let table_col_unqualified = if let Some(dot_pos) = table_col.rfind('.') {
            table_col[dot_pos + 1..].to_string()
        } else {
            table_col.clone()
        };

        // Try to get the table and check for index or PK
        let txn: Box<dyn Transaction> = self.engine.begin_transaction()?;
        let table: Box<dyn Table> = match txn.get_table(&table_name) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let schema = table.schema();

        // Check if table column is the PRIMARY KEY (direct row_id lookup)
        let table_col_lower = table_col_unqualified.to_lowercase();
        let lookup_strategy = if let Some(pk_idx) = schema.pk_column_index() {
            if schema.columns[pk_idx].name_lower == table_col_lower {
                // It's a primary key lookup - most efficient!
                IndexLookupStrategy::PrimaryKey
            } else if let Some(index) = table.get_index_on_column(&table_col_unqualified) {
                IndexLookupStrategy::SecondaryIndex(index)
            } else {
                return Ok(None);
            }
        } else if let Some(index) = table.get_index_on_column(&table_col_unqualified) {
            IndexLookupStrategy::SecondaryIndex(index)
        } else {
            return Ok(None);
        };

        // Get CTE data
        let (cte_columns, cte_rows) = cte_data;

        // Find CTE column index for the join key
        // Pre-compute the suffix pattern to avoid repeated allocations
        let cte_col_lower = cte_col.to_lowercase();
        let suffix_pattern = format!(".{}", cte_col_lower);
        let cte_key_idx = cte_columns
            .iter()
            .position(|c| {
                // Use case-insensitive comparison without allocation
                c.eq_ignore_ascii_case(&cte_col_lower)
                    || (c.len() > suffix_pattern.len()
                        && c[c.len() - suffix_pattern.len()..]
                            .eq_ignore_ascii_case(&suffix_pattern))
            })
            .ok_or_else(|| Error::internal(format!("CTE column not found: {}", cte_col)))?;

        // Build qualified column names for CTE (they're not prefixed yet)
        let cte_qualified: Vec<String> = if let Some(ref alias) = cte_alias {
            cte_columns
                .iter()
                .map(|col| format!("{}.{}", alias, col))
                .collect()
        } else {
            cte_columns.to_vec()
        };

        // Get table columns with qualified names
        let table_qualified: Vec<String> = {
            let alias = table_alias.as_ref().unwrap_or(&table_name);
            schema
                .columns
                .iter()
                .map(|col| format!("{}.{}", alias, col.name))
                .collect()
        };

        // Determine outer/inner based on which side is CTE
        // CTE is the outer (driver), Table is the inner (lookup)
        let (outer_cols, outer_rows, outer_key_idx, inner_cols) = if cte_on_left {
            // CTE on left = CTE is outer, Table is inner
            (
                cte_qualified.clone(),
                &cte_rows,
                cte_key_idx,
                table_qualified.clone(),
            )
        } else {
            // CTE on right = CTE is outer, Table is inner
            // Note: We swap the semantics but the result order must match SQL expectations
            (
                cte_qualified.clone(),
                &cte_rows,
                cte_key_idx,
                table_qualified.clone(),
            )
        };

        // Execute Index Nested Loop Join using operator
        // Extract just the Row part from the (i64, Row) tuples for the operator
        let rows_only: Vec<Row> = outer_rows.iter().map(|(_, row)| row.clone()).collect();
        let outer_result: Box<dyn QueryResult> = Box::new(ExecutorResult::with_shared_rows(
            outer_cols.clone(),
            CompactArc::new(rows_only),
        ));
        let outer_op: Box<dyn Operator> =
            Box::new(QueryResultOperator::new(outer_result, outer_cols.clone()));

        // Build inner schema info
        let inner_schema_info: Vec<ColumnInfo> = inner_cols.iter().map(ColumnInfo::new).collect();

        // Parse join type
        let op_join_type = OperatorJoinType::parse(&join_type);

        // Create the Index Nested Loop Join operator
        let mut join_op: Box<dyn Operator> = Box::new(IndexNestedLoopJoinOperator::new(
            outer_op,
            table,
            inner_schema_info,
            op_join_type,
            outer_key_idx,
            lookup_strategy,
            None, // No residual filter for now (handled later if needed)
        ));

        // Execute and collect results with synthetic row IDs
        join_op.open()?;
        let mut row_id = 0i64;
        let mut result_rows = RowVec::with_capacity(effective_limit.unwrap_or(1000) as usize);
        while let Some(row_ref) = join_op.next()? {
            result_rows.push((row_id, row_ref.into_owned()));
            row_id += 1;
            if let Some(lim) = effective_limit {
                if result_rows.len() >= lim as usize {
                    break;
                }
            }
        }
        join_op.close()?;

        // Build all_columns to match physical row order from Index NL (outer=CTE, inner=table)
        // This avoids expensive per-row rotation - projections find columns by name
        let all_columns: Vec<String> = {
            let mut cols = cte_qualified;
            cols.extend(table_qualified);
            cols
        };

        // Apply WHERE clause if present (result_rows is RowVec with (row_id, Row) tuples)
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            let mut where_eval =
                ExpressionEval::compile(where_clause, &all_columns)?.with_context(ctx);

            let mut rows = RowVec::with_capacity(result_rows.len());
            for (row_id, row) in result_rows {
                if where_eval.eval_bool(&row) {
                    rows.push((row_id, row));
                }
            }
            rows
        } else {
            result_rows
        };

        // Check for window functions
        if classification.has_window_functions {
            let result =
                self.execute_select_with_window_functions(stmt, ctx, &filtered_rows, &all_columns)?;
            return Ok(Some(result));
        }

        // Check for aggregation
        if classification.has_aggregation {
            let result =
                self.execute_select_with_aggregation(stmt, ctx, filtered_rows, &all_columns)?;

            // Apply ORDER BY if present (aggregation doesn't handle it for CTE JOINs)
            if !stmt.order_by.is_empty() {
                let output_columns = result.columns().to_vec();
                let rows = Self::materialize_result(result)?;
                let sorted_rows =
                    self.apply_order_by_to_rows(rows, &stmt.order_by, &output_columns)?;
                return Ok(Some(Box::new(ExecutorResult::new(
                    output_columns,
                    sorted_rows,
                ))));
            }
            return Ok(Some(result));
        }

        // Project rows according to SELECT expressions
        let projected_rows = self.project_rows_with_alias(
            &stmt.columns,
            filtered_rows,
            &all_columns,
            None,
            ctx,
            None,
        )?;

        // Determine output column names
        let output_columns = self.get_output_column_names(&stmt.columns, &all_columns, None);

        // Apply ORDER BY if present
        let final_rows = if !stmt.order_by.is_empty() {
            self.apply_order_by_to_rows(projected_rows, &stmt.order_by, &output_columns)?
        } else {
            projected_rows
        };

        Ok(Some(Box::new(ExecutorResult::new(
            output_columns,
            final_rows,
        ))))
    }

    // =========================================================================
    // LAZY CTE LIMIT PUSHDOWN
    // =========================================================================

    /// Compute LIMIT hints for CTEs that can benefit from early termination.
    ///
    /// Returns a map from CTE name to the hint (LIMIT + optional ORDER BY).
    /// This is safe when:
    /// 1. CTE is used exactly once in INNER/LEFT/RIGHT JOIN
    /// 2. Main query has LIMIT
    /// 3. ORDER BY (if present) references only CTE columns
    fn compute_cte_limit_hints(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
    ) -> FxHashMap<String, CtePushdownHint> {
        let mut hints = FxHashMap::default();

        // Must have a LIMIT on the main query
        let main_limit = match &stmt.limit {
            Some(lim) => match lim.as_ref() {
                Expression::IntegerLiteral(lit) if lit.value > 0 => lit.value as u64,
                _ => return hints,
            },
            None => return hints,
        };

        // Add OFFSET to limit if present
        let main_limit = match &stmt.offset {
            Some(off) => match off.as_ref() {
                Expression::IntegerLiteral(lit) => main_limit.saturating_add(lit.value as u64),
                _ => main_limit,
            },
            None => main_limit,
        };

        // Check for JOIN pattern in main query
        let join_source = match &stmt.table_expr {
            Some(expr) => match expr.as_ref() {
                Expression::JoinSource(js) => js,
                _ => return hints,
            },
            None => return hints,
        };

        let join_type = join_source.join_type.to_uppercase();
        let is_inner_join = join_type == "INNER" || join_type.is_empty() || join_type == "JOIN";
        let is_left_join = join_type == "LEFT" || join_type == "LEFT OUTER";
        let is_right_join = join_type == "RIGHT" || join_type == "RIGHT OUTER";

        if !is_inner_join && !is_left_join && !is_right_join {
            return hints;
        }

        // Build set of CTE names
        let cte_names: AHashSet<String> = with_clause
            .ctes
            .iter()
            .filter(|c| !c.is_recursive) // Skip recursive CTEs
            .map(|c| c.name.value_lower.to_string())
            .collect();

        // Check if one side of the join is a CTE
        let left_cte = self
            .extract_cte_name_for_lookup(&join_source.left)
            .filter(|n| cte_names.contains(&n.to_lowercase()));
        let right_cte = self
            .extract_cte_name_for_lookup(&join_source.right)
            .filter(|n| cte_names.contains(&n.to_lowercase()));

        // For INNER JOIN: either side can be CTE
        // For LEFT JOIN: CTE must be on the RIGHT
        // For RIGHT JOIN: CTE must be on the LEFT
        let cte_name = match (&left_cte, &right_cte) {
            (Some(name), None) if is_inner_join || is_right_join => name.to_lowercase(),
            (None, Some(name)) if is_inner_join || is_left_join => name.to_lowercase(),
            _ => return hints,
        };

        // Check that the CTE has GROUP BY (the pattern we're optimizing)
        let cte = with_clause
            .ctes
            .iter()
            .find(|c| c.name.value_lower.as_str() == cte_name);

        if let Some(cte) = cte {
            // CTE must have GROUP BY for streaming optimization to help
            if !cte.query.group_by.columns.is_empty() {
                // Check if ORDER BY can be pushed down (all columns from CTE)
                let cte_alias = match (&left_cte, &right_cte) {
                    (Some(name), None) => name.clone(),
                    (None, Some(name)) => name.clone(),
                    _ => cte_name.clone(),
                };

                let order_by_for_cte = self.extract_pushable_order_by(stmt, &cte_alias, cte);

                // CORRECTNESS FIX: Don't push LIMIT if ORDER BY exists but can't be pushed.
                // If the main query has ORDER BY on non-CTE columns, pushing LIMIT to CTE
                // would incorrectly limit results before the final ordering.
                // Example: WITH stats AS (...) SELECT * FROM t JOIN stats ORDER BY t.name LIMIT 10
                // The CTE needs ALL matching rows, not just the first N, because the final
                // ORDER BY t.name determines which rows appear in the result.
                if !stmt.order_by.is_empty() && order_by_for_cte.is_empty() {
                    return hints; // Can't safely push LIMIT without ORDER BY
                }

                hints.insert(
                    cte_name,
                    CtePushdownHint {
                        limit: main_limit,
                        order_by: order_by_for_cte,
                    },
                );
            }
        }

        hints
    }

    /// Extract ORDER BY items that can be pushed to a CTE.
    /// Returns empty vec if ORDER BY cannot be pushed (references non-CTE columns).
    fn extract_pushable_order_by(
        &self,
        stmt: &SelectStatement,
        cte_alias: &str,
        cte: &CommonTableExpression,
    ) -> Vec<OrderByExpression> {
        if stmt.order_by.is_empty() {
            return vec![];
        }

        // Get CTE output column names
        let cte_columns: Vec<String> = if !cte.column_names.is_empty() {
            cte.column_names
                .iter()
                .map(|n| n.value_lower.to_string())
                .collect()
        } else {
            // Extract from SELECT columns
            cte.query
                .columns
                .iter()
                .filter_map(|col| {
                    // Handle Aliased expressions (e.g., SUM(amount) AS total)
                    if let Expression::Aliased(aliased) = col {
                        return Some(aliased.alias.value_lower.to_string());
                    }
                    // Handle plain identifiers
                    if let Expression::Identifier(id) = col {
                        return Some(id.value_lower.to_string());
                    }
                    // Handle function calls (e.g., SUM(amount), ROUND(SUM(x), 2))
                    // Use expr_to_normalized_string for robust handling of nested functions,
                    // multiple arguments, and proper case normalization
                    if matches!(col, Expression::FunctionCall(_)) {
                        return Some(expr_to_normalized_string(col));
                    }
                    None
                })
                .collect()
        };

        let cte_alias_lower = cte_alias.to_lowercase();
        let mut result = Vec::new();

        for order_item in &stmt.order_by {
            // Check if this ORDER BY item references a CTE column
            let col_name: Option<String> = match &order_item.expression {
                Expression::Identifier(id) => Some(id.value_lower.to_string()),
                Expression::QualifiedIdentifier(qi) => {
                    // table.column format
                    if qi.qualifier.value_lower.as_str() == cte_alias_lower {
                        Some(qi.name.value_lower.to_string())
                    } else {
                        // References different table - can't push
                        return vec![];
                    }
                }
                Expression::FunctionCall(_) => {
                    // Handle ORDER BY SUM(amount), ROUND(AVG(x), 2), etc.
                    Some(expr_to_normalized_string(&order_item.expression))
                }
                _ => None,
            };

            if let Some(col) = col_name {
                // Check if column exists in CTE (exact match only)
                if cte_columns.iter().any(|c| c == &col) {
                    // Create ORDER BY item without table qualifier
                    let col_compact: SmartString = col.clone().into();
                    let new_expr = Expression::Identifier(Identifier {
                        token: Token::new(
                            TokenType::Identifier,
                            col_compact.clone(),
                            Position::new(0, 0, 0),
                        ),
                        value: col_compact.clone(),
                        value_lower: col_compact,
                    });
                    result.push(OrderByExpression {
                        expression: new_expr,
                        ascending: order_item.ascending,
                        nulls_first: order_item.nulls_first,
                    });
                } else {
                    // Column not in CTE - can't push ORDER BY
                    return vec![];
                }
            } else {
                // Complex expression - can't push
                return vec![];
            }
        }

        result
    }

    /// Execute a CTE query with pushed-down LIMIT and optional ORDER BY.
    fn execute_cte_query_with_hint(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
        hint: &CtePushdownHint,
    ) -> Result<(Vec<String>, RowVec)> {
        // Create a modified statement with LIMIT and ORDER BY pushed down
        let mut modified_stmt = stmt.clone();

        // Push ORDER BY if provided and CTE doesn't have one
        if !hint.order_by.is_empty() && modified_stmt.order_by.is_empty() {
            modified_stmt.order_by = hint.order_by.clone();
        }

        // Push LIMIT if CTE doesn't already have one
        if modified_stmt.limit.is_none() {
            modified_stmt.limit = Some(Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: Token::new(
                    TokenType::Integer,
                    hint.limit.to_string(),
                    Position::new(0, 0, 0),
                ),
                value: hint.limit as i64,
            })));
        }

        // Execute with the pushed optimizations
        self.execute_cte_query(&modified_stmt, ctx, cte_registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::row_vec::RowVec;
    use crate::storage::mvcc::engine::MVCCEngine;
    use std::sync::Arc;

    fn make_rows(rows: Vec<Row>) -> RowVec {
        let mut rv = RowVec::with_capacity(rows.len());
        for (i, row) in rows.into_iter().enumerate() {
            rv.push((i as i64, row));
        }
        rv
    }

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
        let rows = make_rows(vec![Row::from_values(vec![
            Value::Integer(1),
            Value::text("test"),
        ])]);

        registry.store("my_cte", columns.clone(), rows);

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
