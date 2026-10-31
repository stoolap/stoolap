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
use std::sync::Arc;

use crate::common::{CompactArc, CompactVec, StringMap};
use crate::core::{Error, Result, Row, RowVec, Value};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::traits::QueryResult;

use super::context::ExecutionContext;
use super::expression::ExpressionEval;
use super::query::{order_by_inputs, ColumnSource, OutputLayout};
use super::query_cache::NestedPlan;
use super::query_classification::get_classification;
use super::result::ExecutorResult;
use super::row_count::{row_counts, static_row_counts};
use super::utils::{
    build_column_index_map, for_each_qualified_star_column, item_name, order_by_names_an_alias,
};
use super::Executor;

/// Type alias for CTE data: (columns, rows) with Arc for zero-copy sharing
/// Uses Vec<(i64, Row)> for rows - same structure as RowVec but Arc-shareable
pub type CteData = (CompactArc<Vec<String>>, CompactArc<Vec<(i64, Row)>>);

/// Type alias for CTE data map
/// Uses CompactArc<Vec<String>> for columns and CompactArc<Vec<(i64, Row)>> for rows
/// to enable zero-copy sharing of CTE results with joins
pub type CteDataMap = StringMap<CteData>;

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
        Expression::StringLiteral(lit) => lit.to_string(),
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
pub(crate) struct CtePushdownHint {
    limit: u64,
    order_by: Vec<OrderByExpression>,
    /// Under an INNER join, the rows the statement must still return from
    /// the cut CTE, or it runs again under a larger cut; None for a kept side
    must_fill: Option<u64>,
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
            data: Arc::new(StringMap::new()),
        }
    }

    /// A registry holding the CTEs `ctx` brings: a statement's own CTEs
    /// join those of the WITH clauses around it, hiding one of their name
    pub(crate) fn within(ctx: &ExecutionContext) -> Self {
        Self {
            data: ctx.cte_data().cloned().unwrap_or_default(),
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
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        // Get the WITH clause
        let with_clause = match &stmt.with {
            Some(with) => with,
            None => return self.execute_select_with_plan(stmt, ctx, None, plan, None),
        };

        // CTE INLINING OPTIMIZATION:
        // For single-use, non-recursive CTEs, convert to subqueries to:
        // 1. Preserve index access (CTEs lose indexes when materialized)
        // 2. Enable LIMIT pushdown through subqueries
        // This is similar to PostgreSQL 12+'s CTE inlining behavior
        if let Some(inlined_stmt) = self.try_inline_ctes(stmt, with_clause) {
            // Execute the rewritten query (without WITH clause or with fewer CTEs)
            return self.execute_select_with_plan(&inlined_stmt, ctx, None, plan, None);
        }

        self.run_over_cut_ctes(stmt, with_clause, plan, ctx, |cte_registry| {
            self.execute_main_query_with_ctes(stmt, ctx, cte_registry, plan)
        })
    }

    /// `main` run over the CTEs of `with_clause`, a single-use CTE joined
    /// under a LIMIT read only as far as the LIMIT needs, so a streaming
    /// GROUP BY stops early
    pub(crate) fn run_over_cut_ctes(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
        plan: Option<&NestedPlan>,
        ctx: &ExecutionContext,
        mut main: impl FnMut(&mut CteRegistry) -> Result<Box<dyn QueryResult>>,
    ) -> Result<Box<dyn QueryResult>> {
        let mut cte_limit_hints = self.compute_cte_limit_hints(stmt, with_clause);
        let plans = plan.map_or(&[][..], |plan| plan.ctes(stmt));

        loop {
            let mut cte_registry =
                self.materialize_ctes(with_clause, plans, ctx, Some(&cte_limit_hints))?;

            let result = main(&mut cte_registry)?;

            // A CTE cut under an INNER join may match fewer rows than the LIMIT
            // asks; then its cut grows until the rows fill it or nothing is cut
            let short = cte_limit_hints.iter_mut().find_map(|(name, hint)| {
                let wanted = hint.must_fill?;
                let cut = cte_registry
                    .get(name)
                    .is_some_and(|(_, rows)| rows.len() as u64 >= hint.limit);
                cut.then_some((wanted, hint))
            });
            let Some((wanted, hint)) = short else {
                return Ok(result);
            };
            let columns = result.columns().to_vec();
            let rows = Self::materialize_result(result)?;
            if rows.len() as u64 >= wanted {
                return Ok(Box::new(ExecutorResult::new(columns, rows)));
            }
            // Grow by the share of cut rows that matched, with an eighth to spare
            let got = rows.len() as u128;
            let grown = match got {
                0 => hint.limit as u128 * 2,
                _ => hint.limit as u128 * wanted as u128 * 9 / (got * 8) + 1,
            };
            hint.limit = grown.clamp(hint.limit as u128 + 1, i64::MAX as u128) as u64;
        }
    }

    /// Run the CTEs of a WITH clause in order and store each result, so
    /// the statement that follows, a SELECT or a DML, can read them; each
    /// body runs under its own plan in `plans`, when it holds one
    pub(crate) fn materialize_ctes(
        &self,
        with_clause: &WithClause,
        plans: &[NestedPlan],
        ctx: &ExecutionContext,
        cte_limit_hints: Option<&StringMap<CtePushdownHint>>,
    ) -> Result<CteRegistry> {
        let mut cte_registry = CteRegistry::within(ctx);
        // Each body counts its own rows, not the statement's around it
        let scoped = ctx.without_read_counts();
        let ctx = scoped.as_ref().unwrap_or(ctx);

        // Execute each CTE in order
        for (index, cte) in with_clause.ctes.iter().enumerate() {
            // Check if we have a pushdown hint for this CTE
            let cte_name_lower: String = cte.name.value_lower.to_string();
            let pushdown_hint = cte_limit_hints.and_then(|hints| hints.get(&cte_name_lower));
            let plan = plans.get(index);

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
                    plan,
                    ctx,
                    &mut cte_registry,
                    aliases,
                )?
            } else if let Some(hint) = pushdown_hint {
                // Execute CTE with pushed-down LIMIT and optional ORDER BY
                self.execute_cte_query_with_hint(&cte.query, plan, ctx, &mut cte_registry, hint)?
            } else {
                self.execute_cte_query(&cte.query, plan, ctx, &mut cte_registry)?
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

        Ok(cte_registry)
    }

    /// The CTE of `registry` the CTE body `stmt` reads when the CTE path can
    /// finish it on its own; None for a body execute_select runs, which
    /// handles set operations and the rest
    pub(super) fn cte_result_source<'r>(
        &self,
        stmt: &SelectStatement,
        registry: &'r CteRegistry,
    ) -> Option<&'r CteData> {
        if !Self::cte_result_path_finishes(stmt) {
            return None;
        }
        let name = self.extract_cte_name_for_lookup(stmt.table_expr.as_deref()?)?;
        registry.get(&name)
    }

    /// Execute a single CTE query, under `plan` when it holds one
    fn execute_cte_query(
        &self,
        stmt: &SelectStatement,
        plan: Option<&NestedPlan>,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
    ) -> Result<(Vec<String>, RowVec)> {
        if let Some((columns, rows)) = self.cte_result_source(stmt, cte_registry) {
            // Execute query against CTE result
            // Clone Arc data into RowVec for processing
            // A subquery in the statement reads the earlier
            // CTEs through the context
            let counted = self.read_volatile_counts(stmt, ctx, None)?;
            let ctx = counted.as_ref().unwrap_or(ctx);
            return self.execute_query_on_cte_result(
                stmt,
                &ctx.with_cte_data(cte_registry.data()),
                columns.to_vec(),
                RowVec::from_vec((**rows).clone()),
            );
        }

        // Execute as a normal SELECT and materialize
        // Create context with CTE data so tables can resolve CTE references
        let ctx_with_ctes = ctx.with_cte_data(cte_registry.data());
        let result = self.execute_select_with_plan(stmt, &ctx_with_ctes, None, plan, None)?;
        let columns = result.columns().to_vec();
        let rows = Self::materialize_result(result)?;

        Ok((columns, rows))
    }

    /// Execute a recursive CTE; with `plan`, its body's, the first member
    /// runs under it and each later member under its branch's
    fn execute_recursive_cte_with_columns(
        &self,
        cte_name: &str,
        stmt: &SelectStatement,
        plan: Option<&NestedPlan>,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
        column_aliases: Option<&[Identifier]>,
    ) -> Result<(Vec<String>, RowVec)> {
        use crate::parser::ast::SetOperationType;

        // Maximum iterations to prevent infinite loops
        const MAX_ITERATIONS: usize = 10000;

        // The recursive CTE query should have UNION ALL structure
        if stmt.set_operations.is_empty() {
            return Err(Error::InvalidArgument(
                "Recursive CTE must have UNION ALL between anchor and recursive members"
                    .to_string(),
            ));
        }

        // Check that all set operations are UNION ALL
        for set_op in &stmt.set_operations {
            if !matches!(set_op.operation, SetOperationType::UnionAll) {
                return Err(Error::InvalidArgument(
                    "Recursive CTE only supports UNION ALL (not UNION)".to_string(),
                ));
            }
        }

        // Execute the anchor member (the first SELECT before UNION ALL)
        let anchor_stmt = SelectStatement {
            param_count: 0,
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            distinct_on: stmt.distinct_on.clone(),
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
            prepared: None,
        };

        // Taken first, so the body's plans are laid out for the whole body
        let branches = plan.map_or(&[][..], |plan| plan.branches(stmt));
        let result = self.execute_select_with_plan(&anchor_stmt, ctx, None, plan, None)?;
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

        // A branch that does not read the CTE belongs to the first: it runs
        // once, and its rows start the recursion with the first's
        let mut recursive_members = Vec::with_capacity(stmt.set_operations.len());
        for (set_op, index) in stmt.set_operations.iter().zip(0..) {
            let branch = branches.get(index);
            if self.statement_reads_cte(&set_op.right, cte_name) {
                recursive_members.push((&set_op.right, branch));
                continue;
            }
            let (_, rows) = self.execute_cte_query(&set_op.right, branch, ctx, cte_registry)?;
            let base_id = all_rows.len() as i64;
            for (i, (_, row)) in rows.into_iter().enumerate() {
                all_rows.push((base_id + i as i64, row));
            }
        }

        // If no anchor rows, return empty result
        if all_rows.is_empty() {
            return Ok((columns, all_rows));
        }

        // Shared rows make every accumulator/working-set/registry copy
        // an Arc bump instead of a deep row clone
        let mut all_rows: RowVec = all_rows
            .into_iter()
            .map(|(id, row)| (id, row.into_shared()))
            .collect();

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
            // Add current CTE with working rows; the working set is
            // rebuilt from new_rows below, so move it instead of cloning
            // mem::replace with a pool-free empty vec: RowVec::default
            // would check a buffer out of the thread-local pool just to
            // serve as a throwaway placeholder
            temp_registry.store(
                cte_name,
                columns.clone(),
                std::mem::replace(&mut working_rows, RowVec::from_vec(Vec::new())),
            );

            // Execute each recursive member
            let mut new_rows = RowVec::new();
            for &(member, branch) in &recursive_members {
                let recursive_result =
                    self.execute_cte_query(member, branch, ctx, &mut temp_registry)?;

                // Extend with rows from recursive result, renumbering row IDs
                let base_id = new_rows.len() as i64;
                for (i, (_, row)) in recursive_result.1.into_iter().enumerate() {
                    new_rows.push((base_id + i as i64, row));
                }
            }

            if new_rows.is_empty() {
                break;
            }

            // Convert to Shared once: the accumulator extension and the
            // next working set then share the same row data
            let new_rows: RowVec = new_rows
                .into_iter()
                .map(|(id, row)| (id, row.into_shared()))
                .collect();

            // Add new rows to total result, renumbering row IDs
            let base_id = all_rows.len() as i64;
            for (i, (_, row)) in new_rows.iter().enumerate() {
                all_rows.push((base_id + i as i64, row.clone()));
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
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        // Create a context with CTE data so subqueries can access CTEs
        let ctx_with_ctes = ctx.with_cte_data(cte_registry.data());

        // Check if the main query references a CTE
        if let Some(ref table_expr) = stmt.table_expr {
            // First check for simple CTE reference, when the CTE path can
            // finish the statement on its own
            // Use the base CTE name (not alias) for registry lookup
            if Self::cte_result_path_finishes(stmt) {
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
        }

        // Execute as a normal SELECT (strip the WITH clause)
        let stmt_without_with = SelectStatement {
            param_count: 0,
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            distinct_on: stmt.distinct_on.clone(),
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
            prepared: None,
        };

        self.execute_select_with_plan(&stmt_without_with, &ctx_with_ctes, None, plan, None)
    }

    /// Execute a query on CTE result data
    pub(crate) fn execute_query_on_cte_result(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_columns: Vec<String>,
        cte_rows: RowVec,
    ) -> Result<(Vec<String>, RowVec)> {
        let (cols, rows, _applied, _layout) =
            self.execute_query_on_cte_result_inner(stmt, ctx, cte_columns, cte_rows, false)?;
        Ok((cols, rows))
    }

    /// Inner implementation that optionally skips ORDER BY/LIMIT processing.
    /// Returns (columns, rows, order_limit_applied, layout of the columns
    /// for the ORDER BY the caller applies).
    /// When `skip_order_limit` is true, ORDER BY and LIMIT/OFFSET are NOT applied,
    /// allowing the caller to delegate to a more capable ORDER BY handler.
    pub(crate) fn execute_query_on_cte_result_inner(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        cte_columns: Vec<String>,
        cte_rows: RowVec,
        skip_order_limit: bool,
    ) -> Result<(Vec<String>, RowVec, bool, Option<OutputLayout>)> {
        debug_assert!(skip_order_limit || stmt.order_by.is_empty());
        // OPTIMIZATION: Get cached query classification to avoid repeated AST traversals
        let classification = get_classification(stmt);

        let cte_alias = stmt
            .table_expr
            .as_deref()
            .and_then(super::utils::get_table_alias_from_expr)
            .map(|name| name.to_lowercase());

        // Apply WHERE clause filter
        let filtered_rows = if let Some(ref where_clause) = stmt.where_clause {
            // Process subqueries in WHERE clause (e.g., IN subqueries on CTEs);
            // a correlated one is evaluated per row below
            let processed_where = self.answered_where(where_clause, ctx)?.into_owned();
            // A parent row's column named through its own table is bound
            // first, or the bare name it would fall back to could be the
            // CTE's own
            let processed_where = match ctx.outer_row() {
                Some(outer) => {
                    let inner: Vec<&str> = cte_alias.iter().map(String::as_str).collect();
                    let scope = super::utils::InnerScope {
                        tables: &inner,
                        schema: None,
                    };
                    super::utils::substitute_outer_references_in_scope(
                        &processed_where,
                        outer,
                        &scope,
                    )
                }
                None => processed_where,
            };

            if Self::has_correlated_subqueries(&processed_where) {
                self.filter_rows_by_correlated_where(
                    &processed_where,
                    cte_rows,
                    &cte_columns,
                    cte_alias.as_deref(),
                    ctx,
                )?
            } else {
                // Compile filter once and reuse for all rows
                let mut eval =
                    ExpressionEval::compile(&processed_where, &cte_columns)?.with_context(ctx);

                let mut result = RowVec::new();
                let mut row_id = 0i64;
                for (_, row) in cte_rows {
                    if eval.eval_bool_checked(&row)? {
                        result.push((row_id, row));
                        row_id += 1;
                    }
                }
                result
            }
        } else {
            cte_rows
        };

        if classification.has_aggregation || classification.has_window_functions {
            let (result, layout) = self.group_and_window_rows(
                stmt,
                ctx,
                &classification,
                filtered_rows,
                &cte_columns,
                None,
                |out| Self::result_layout(stmt, &cte_columns, cte_alias.as_deref(), out),
            )?;
            let columns = result.columns().to_vec();
            let mut rows = Self::materialize_result(result)?;

            if !skip_order_limit {
                row_counts(stmt, ctx)?.apply(&mut rows);
            }
            return Ok((columns, rows, !skip_order_limit, layout));
        }

        // Process scalar subqueries in SELECT columns before projection; a
        // correlated one is evaluated per row by the projection
        let processed_columns = if Self::has_correlated_select_subqueries(&stmt.columns) {
            None
        } else {
            self.try_process_select_subqueries(&stmt.columns, ctx)?
        };
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        // Determine output columns
        let output_columns = self.resolve_cte_output_columns_from_exprs(
            &stmt.columns,
            &cte_columns,
            cte_alias.as_deref(),
        )?;

        let needs_projection = self.needs_projection_for_columns(columns_to_use);

        if skip_order_limit {
            // The caller sorts by the layout and trims after: every input
            // column ORDER BY reads that no output column copies rides along
            let mut layout = Self::output_layout(stmt, &cte_columns, cte_alias.as_deref());
            let extra: Vec<usize> = match &layout {
                Some(layout) if needs_projection => order_by_inputs(stmt, &cte_columns)
                    .into_iter()
                    .filter(|&idx| !layout.sources.iter().any(|s| s.input == Some(idx)))
                    .collect(),
                _ => Vec::new(),
            };
            let result_rows = if needs_projection {
                let projected = self.project_cte_rows_from_columns(
                    columns_to_use,
                    &filtered_rows,
                    &cte_columns,
                    cte_alias.as_deref(),
                    ctx,
                )?;
                if extra.is_empty() {
                    projected
                } else {
                    projected
                        .into_iter()
                        .zip(filtered_rows.iter())
                        .map(|((id, proj_row), (_, src_row))| {
                            let mut vals = proj_row.into_values();
                            vals.extend(extra.iter().map(|&idx| {
                                src_row
                                    .get(idx)
                                    .cloned()
                                    .unwrap_or_else(Value::null_unknown)
                            }));
                            (id, Row::from_values(vals))
                        })
                        .collect()
                }
            } else {
                filtered_rows
            };
            let mut columns = output_columns;
            if let Some(layout) = layout.as_mut() {
                for &idx in &extra {
                    columns.push(cte_columns[idx].clone());
                    layout.sources.push(ColumnSource {
                        item: None,
                        input: Some(idx),
                    });
                }
            }
            return Ok((columns, result_rows, false, layout));
        }

        let mut result_rows = if needs_projection {
            self.project_cte_rows_from_columns(
                columns_to_use,
                &filtered_rows,
                &cte_columns,
                cte_alias.as_deref(),
                ctx,
            )?
        } else {
            filtered_rows
        };

        row_counts(stmt, ctx)?.apply(&mut result_rows);

        Ok((output_columns, result_rows, true, None))
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

    /// The output column names of the SELECT items `columns` as written
    fn resolve_cte_output_columns_from_exprs(
        &self,
        columns: &[Expression],
        cte_columns: &[String],
        table_alias: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut output_columns = Vec::new();

        for col_expr in columns {
            match col_expr {
                Expression::Star(_) => {
                    output_columns.extend(cte_columns.iter().cloned());
                }
                Expression::QualifiedStar(qs) => {
                    for_each_qualified_star_column(
                        cte_columns,
                        &qs.qualifier,
                        table_alias,
                        |idx, start| output_columns.push(cte_columns[idx][start..].to_string()),
                    );
                }
                other => output_columns.push(item_name(other)),
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
        table_alias: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};

        // A correlated subquery reads each row as its outer row
        if Self::has_correlated_select_subqueries(columns) {
            return self.project_rows_with_alias(
                columns,
                RowVec::from_vec((**rows).clone()),
                cte_columns,
                None,
                ctx,
                table_alias,
            );
        }

        let col_index_map = build_column_index_map(cte_columns);

        // Pre-compile expressions that need evaluation
        // Store: Star -> None, Identifier -> column index, Complex -> compiled program
        enum CompiledColumn {
            Star,
            Columns(Vec<usize>),
            Identifier(usize),
            Compiled(SharedProgram),
        }

        let compiled_columns: Vec<CompiledColumn> = columns
            .iter()
            .map(|col_expr| match col_expr {
                Expression::Star(_) => Ok(CompiledColumn::Star),
                Expression::QualifiedStar(qs) => {
                    let mut taken = Vec::new();
                    for_each_qualified_star_column(
                        cte_columns,
                        &qs.qualifier,
                        table_alias,
                        |idx, _| taken.push(idx),
                    );
                    Ok(CompiledColumn::Columns(taken))
                }
                // A name no column has reads as the compiler says: the enclosing
                // row's column, a quoted name's text, or an error
                Expression::Identifier(id) => match col_index_map.get(id.value_lower.as_str()) {
                    Some(&idx) => Ok(CompiledColumn::Identifier(idx)),
                    None => compile_expression(col_expr, cte_columns).map(CompiledColumn::Compiled),
                },
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
            let exec_ctx = ExecuteContext::new(row).with_params(ctx.params());

            for compiled in &compiled_columns {
                match compiled {
                    CompiledColumn::Star => {
                        // OPTIMIZATION: Extend with row values
                        values.extend(row.iter().cloned());
                    }
                    CompiledColumn::Columns(taken) => {
                        values.extend(
                            taken.iter().map(|&idx| {
                                row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
                            }),
                        );
                    }
                    CompiledColumn::Identifier(idx) => {
                        values.push(row.get(*idx).cloned().unwrap_or_else(Value::null_unknown));
                    }
                    CompiledColumn::Compiled(program) => {
                        values.push(vm.execute(program, &exec_ctx)?);
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

    /// Whether the query-on-CTE-result path can finish a statement by
    /// itself: it leaves set operations, DISTINCT and ORDER BY, which the
    /// standard sort binds by the output's layout, and a LIMIT or OFFSET
    /// subquery, which execute_select reads first, to execute_select
    fn cte_result_path_finishes(stmt: &SelectStatement) -> bool {
        stmt.set_operations.is_empty()
            && !stmt.distinct
            && stmt.order_by.is_empty()
            && ![stmt.limit.as_deref(), stmt.offset.as_deref()]
                .into_iter()
                .flatten()
                .any(Self::has_subqueries)
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

        Self::limit_keeps_cte_rows(stmt)
            && match (&left_cte, &right_cte) {
                (Some(_), None) => is_left_join || is_inner_join,
                (None, Some(_)) => is_right_join || is_inner_join,
                _ => false,
            }
    }

    /// Whether nothing after the join drops or merges rows, so the rows a
    /// cut CTE joins are rows the statement may return
    fn limit_keeps_cte_rows(stmt: &SelectStatement) -> bool {
        let classification = get_classification(stmt);
        stmt.where_clause.is_none()
            && stmt.group_by.columns.is_empty()
            && stmt.having.is_none()
            && !stmt.distinct
            && stmt.distinct_on.is_empty()
            && stmt.set_operations.is_empty()
            && !classification.has_aggregation
            && !classification.has_window_functions
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
        // 3. CTE is the kept side of an outer join (limit pushdown is safe)
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
        let cte_defs: StringMap<&CommonTableExpression> = cte_names_lower.iter().cloned().collect();
        let cte_name_set: AHashSet<&str> = cte_defs.keys().map(|s| s.as_str()).collect();

        // Inlining reads names by their text: a WITH inside that names one
        // again hides it there, so the CTEs run where their scopes hold
        if super::prepared_select::redefines_cte(stmt, &|name| cte_name_set.contains(name)) {
            return None;
        }

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
        let mut table_ref_counts: StringMap<usize> =
            cte_defs.keys().map(|name| (name.clone(), 0)).collect();
        let mut where_ref_counts: StringMap<usize> = table_ref_counts.clone();

        // Count references in table expression (FROM/JOIN)
        self.count_cte_references_in_expr(table_expr, &mut table_ref_counts);

        // Count references in WHERE clause (including IN/EXISTS subqueries)
        if let Some(ref where_clause) = stmt.where_clause {
            self.count_cte_references_in_expr(where_clause, &mut where_ref_counts);
        }

        // A reference from any other clause reads the CTE by name as well
        self.count_cte_references_outside_from(stmt, &mut where_ref_counts);

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
            param_count: 0,
            token: stmt.token.clone(),
            distinct: stmt.distinct,
            distinct_on: stmt.distinct_on.clone(),
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
            prepared: None,
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

    /// Whether `stmt` reads the CTE named `name`, in any of its clauses
    fn statement_reads_cte(&self, stmt: &SelectStatement, name: &str) -> bool {
        let name = name.to_lowercase();
        let mut counts: StringMap<usize> = StringMap::new();
        counts.insert(name.clone(), 0);
        self.count_cte_references_in_stmt(stmt, &mut counts);
        counts.get(&name).is_some_and(|&count| count > 0)
    }

    /// Count CTE references in a SELECT statement
    fn count_cte_references_in_stmt(
        &self,
        stmt: &SelectStatement,
        ref_counts: &mut StringMap<usize>,
    ) {
        // Check table expression
        if let Some(ref table_expr) = stmt.table_expr {
            self.count_cte_references_in_expr(table_expr, ref_counts);
        }

        // Check WHERE clause
        if let Some(ref where_clause) = stmt.where_clause {
            self.count_cte_references_in_expr(where_clause, ref_counts);
        }

        self.count_cte_references_outside_from(stmt, ref_counts);
    }

    /// Count CTE references in every clause other than FROM and WHERE,
    /// the nested WITH bodies and the set-operation branches included
    fn count_cte_references_outside_from(
        &self,
        stmt: &SelectStatement,
        ref_counts: &mut StringMap<usize>,
    ) {
        let windows = stmt.window_defs.iter().flat_map(|w| {
            w.partition_by
                .iter()
                .chain(w.order_by.iter().map(|o| &o.expression))
                .chain(Self::frame_bounds(w.frame.as_ref()))
        });
        let grouping_sets = match &stmt.group_by.modifier {
            GroupByModifier::GroupingSets(sets) => sets.as_slice(),
            _ => &[],
        };
        for expr in stmt
            .columns
            .iter()
            .chain(stmt.distinct_on.iter())
            .chain(stmt.group_by.columns.iter())
            .chain(grouping_sets.iter().flatten())
            .chain(stmt.having.as_deref())
            .chain(stmt.order_by.iter().map(|o| &o.expression))
            .chain(stmt.limit.as_deref())
            .chain(stmt.offset.as_deref())
            .chain(windows)
        {
            self.count_cte_references_in_expr(expr, ref_counts);
        }

        for cte in stmt.with.iter().flat_map(|w| &w.ctes) {
            self.count_cte_references_in_stmt(&cte.query, ref_counts);
        }
        for set_op in &stmt.set_operations {
            self.count_cte_references_in_stmt(&set_op.right, ref_counts);
        }
    }

    /// The offset expressions of a window frame
    fn frame_bounds(frame: Option<&WindowFrame>) -> impl Iterator<Item = &Expression> {
        frame
            .into_iter()
            .flat_map(|f| std::iter::once(&f.start).chain(f.end.as_ref()))
            .filter_map(|bound| match bound {
                WindowFrameBound::Preceding(expr) | WindowFrameBound::Following(expr) => {
                    Some(expr.as_ref())
                }
                _ => None,
            })
    }

    /// Count the CTE references in a function call's arguments, ORDER BY and FILTER
    fn count_cte_references_in_call(&self, func: &FunctionCall, ref_counts: &mut StringMap<usize>) {
        for arg in func
            .arguments
            .iter()
            .chain(func.order_by.iter().map(|o| &o.expression))
            .chain(func.filter.as_deref())
        {
            self.count_cte_references_in_expr(arg, ref_counts);
        }
    }

    /// Count CTE references in an expression
    fn count_cte_references_in_expr(&self, expr: &Expression, ref_counts: &mut StringMap<usize>) {
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
            // A bare name is a column: a source is parsed as a TableSource
            Expression::Identifier(_) | Expression::Column(_) => {}
            Expression::JoinSource(js) => {
                self.count_cte_references_in_expr(&js.left, ref_counts);
                self.count_cte_references_in_expr(&js.right, ref_counts);
                if let Some(condition) = js.condition.as_deref() {
                    self.count_cte_references_in_expr(condition, ref_counts);
                }
            }
            Expression::SubquerySource(sq) => {
                self.count_cte_references_in_stmt(&sq.subquery, ref_counts);
            }
            Expression::List(list) => {
                for element in &list.elements {
                    self.count_cte_references_in_expr(element, ref_counts);
                }
            }
            Expression::ExpressionList(list) => {
                for element in &list.expressions {
                    self.count_cte_references_in_expr(element, ref_counts);
                }
            }
            Expression::Distinct(distinct) => {
                self.count_cte_references_in_expr(&distinct.expr, ref_counts);
            }
            Expression::InHashSet(in_set) => {
                self.count_cte_references_in_expr(&in_set.column, ref_counts);
            }
            Expression::QuantifiedSet(quantified) => {
                self.count_cte_references_in_expr(&quantified.left, ref_counts);
            }
            Expression::ValuesSource(values) => {
                for value in values.rows.iter().flatten() {
                    self.count_cte_references_in_expr(value, ref_counts);
                }
            }
            Expression::FunctionTableSource(source) => {
                for arg in &source.arguments {
                    self.count_cte_references_in_expr(arg, ref_counts);
                }
            }
            Expression::ScalarSubquery(sq) => {
                self.count_cte_references_in_stmt(&sq.subquery, ref_counts);
            }
            Expression::In(in_expr) => {
                self.count_cte_references_in_expr(&in_expr.left, ref_counts);
                self.count_cte_references_in_expr(&in_expr.right, ref_counts);
            }
            Expression::Exists(ex) => {
                self.count_cte_references_in_stmt(&ex.subquery, ref_counts);
            }
            Expression::AllAny(all_any) => {
                self.count_cte_references_in_expr(&all_any.left, ref_counts);
                self.count_cte_references_in_stmt(&all_any.subquery, ref_counts);
            }
            Expression::Infix(infix) => {
                self.count_cte_references_in_expr(&infix.left, ref_counts);
                self.count_cte_references_in_expr(&infix.right, ref_counts);
            }
            Expression::Prefix(prefix) => {
                self.count_cte_references_in_expr(&prefix.right, ref_counts);
            }
            Expression::Between(between) => {
                self.count_cte_references_in_expr(&between.expr, ref_counts);
                self.count_cte_references_in_expr(&between.lower, ref_counts);
                self.count_cte_references_in_expr(&between.upper, ref_counts);
            }
            Expression::Cast(cast) => {
                self.count_cte_references_in_expr(&cast.expr, ref_counts);
            }
            Expression::Like(like) => {
                self.count_cte_references_in_expr(&like.left, ref_counts);
                self.count_cte_references_in_expr(&like.pattern, ref_counts);
                if let Some(escape) = like.escape.as_deref() {
                    self.count_cte_references_in_expr(escape, ref_counts);
                }
            }
            Expression::FunctionCall(func) => self.count_cte_references_in_call(func, ref_counts),
            Expression::Case(case) => {
                for expr in case
                    .value
                    .as_deref()
                    .into_iter()
                    .chain(
                        case.when_clauses
                            .iter()
                            .flat_map(|w| [&w.condition, &w.then_result]),
                    )
                    .chain(case.else_value.as_deref())
                {
                    self.count_cte_references_in_expr(expr, ref_counts);
                }
            }
            Expression::Aliased(aliased) => {
                self.count_cte_references_in_expr(&aliased.expression, ref_counts);
            }
            Expression::Window(window) => {
                self.count_cte_references_in_call(&window.function, ref_counts);
                for expr in window
                    .partition_by
                    .iter()
                    .chain(window.order_by.iter().map(|o| &o.expression))
                    .chain(Self::frame_bounds(window.frame.as_ref()))
                {
                    self.count_cte_references_in_expr(expr, ref_counts);
                }
            }
            Expression::QualifiedIdentifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => {}
        }
    }

    /// Replace CTE references with subqueries in an expression.
    /// Returns Some(new_expr) if any replacement was made, None if no changes needed.
    fn try_inline_cte_references(
        &self,
        expr: &Expression,
        cte_defs: &StringMap<&CommonTableExpression>,
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

    // =========================================================================
    // LAZY CTE LIMIT PUSHDOWN
    // =========================================================================

    /// Whether the statement reads the CTE once and no other CTE reads it
    fn cte_read_once(&self, stmt: &SelectStatement, with_clause: &WithClause, name: &str) -> bool {
        let mut counts: StringMap<usize> = StringMap::new();
        counts.insert(name.to_string(), 0);
        self.count_cte_references_in_stmt(stmt, &mut counts);
        for cte in &with_clause.ctes {
            if cte.name.value_lower.as_str() != name {
                self.count_cte_references_in_stmt(&cte.query, &mut counts);
            }
        }
        counts.get(name).copied() == Some(1)
    }

    /// Compute LIMIT hints for CTEs that can benefit from early termination.
    ///
    /// Returns a map from CTE name to the hint (LIMIT + optional ORDER BY).
    /// This is safe when:
    /// 1. CTE is used exactly once, as the kept side of a LEFT/RIGHT JOIN or
    ///    either side of an INNER JOIN, whose result must then fill the LIMIT
    /// 2. Main query has LIMIT
    /// 3. ORDER BY (if present) references only CTE columns
    pub(crate) fn compute_cte_limit_hints(
        &self,
        stmt: &SelectStatement,
        with_clause: &WithClause,
    ) -> StringMap<CtePushdownHint> {
        let mut hints = StringMap::new();

        // A positive literal LIMIT on the main query, plus its OFFSET; any
        // other count, or a sum past what the pushed LIMIT literal can hold,
        // gives no hint
        let Some((wanted, main_limit)) = static_row_counts(stmt).and_then(|counts| {
            let wanted = counts.limit.filter(|limit| *limit > 0)?;
            let sum = wanted.checked_add(counts.offset)?;
            i64::try_from(sum).ok()?;
            Some((wanted as u64, sum as u64))
        }) else {
            return hints;
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

        if (!is_inner_join && !is_left_join && !is_right_join) || !Self::limit_keeps_cte_rows(stmt)
        {
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

        // The CTE must be the side an outer join keeps, or either side of an
        // INNER join, whose result is checked to fill the LIMIT
        let cte_name = match (&left_cte, &right_cte) {
            (Some(name), None) if is_left_join || is_inner_join => name.to_lowercase(),
            (None, Some(name)) if is_right_join || is_inner_join => name.to_lowercase(),
            _ => return hints,
        };
        let must_fill = is_inner_join.then_some(wanted);

        // Check that the CTE has GROUP BY (the pattern we're optimizing)
        let cte = with_clause
            .ctes
            .iter()
            .find(|c| c.name.value_lower.as_str() == cte_name);

        if let Some(cte) = cte {
            // CTE must have GROUP BY for streaming optimization to help, no
            // order or cut of its own, and no reader but this join
            if !cte.query.group_by.columns.is_empty()
                && cte.query.order_by.is_empty()
                && cte.query.limit.is_none()
                && cte.query.offset.is_none()
                && cte.query.set_operations.is_empty()
                && self.cte_read_once(stmt, with_clause, &cte_name)
            {
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
                        must_fill,
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
            if order_by_names_an_alias(&order_item.expression, stmt) {
                return vec![];
            }
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
                if let Some(at) = cte_columns.iter().position(|c| c == &col) {
                    // A column list names the query's items only after it
                    // runs, so the query is ordered by the item's position
                    let new_expr = if cte.column_names.is_empty() {
                        let col_compact: SmartString = col.clone().into();
                        Expression::Identifier(Identifier {
                            value: col_compact.clone(),
                            value_lower: col_compact,
                            position: Position::new(0, 0, 0),
                            quoted: false,
                        })
                    } else {
                        Expression::IntegerLiteral(IntegerLiteral {
                            token: Token::new(
                                TokenType::Integer,
                                (at + 1).to_string(),
                                Position::new(0, 0, 0),
                            ),
                            value: at as i64 + 1,
                        })
                    };
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

    /// Execute a CTE query with pushed-down LIMIT and optional ORDER BY,
    /// under `plan` when it holds one and the query runs blocks with it
    fn execute_cte_query_with_hint(
        &self,
        stmt: &SelectStatement,
        plan: Option<&NestedPlan>,
        ctx: &ExecutionContext,
        cte_registry: &mut CteRegistry,
        hint: &CtePushdownHint,
    ) -> Result<(Vec<String>, RowVec)> {
        // Create a modified statement with LIMIT and ORDER BY pushed down
        let mut modified_stmt = stmt.clone();

        // Push ORDER BY if provided and CTE doesn't have one
        if !hint.order_by.is_empty() && modified_stmt.order_by.is_empty() {
            super::result::OrderLimitPath::CteOrderLimitPushdown.taken();
            modified_stmt.order_by = hint.order_by.clone();
        }

        // Push LIMIT if CTE doesn't already have one: under a plan as a
        // parameter past the statement's own, so every cut runs one shape
        let plan = plan.filter(|_| Self::body_runs_blocks(stmt));
        let mut cut_ctx = None;
        if modified_stmt.limit.is_none() {
            let limit = match plan {
                Some(_) => {
                    let cut = cut_ctx.insert(ctx.clone());
                    cut.add_param(Value::Integer(hint.limit as i64));
                    super::dml::answer_parameter(cut.params().len())
                }
                None => Expression::IntegerLiteral(IntegerLiteral {
                    token: Token::new(
                        TokenType::Integer,
                        hint.limit.to_string(),
                        Position::new(0, 0, 0),
                    ),
                    value: hint.limit as i64,
                }),
            };
            modified_stmt.limit = Some(Box::new(limit));
        }

        // Execute with the pushed optimizations
        let ctx = cut_ctx.as_ref().unwrap_or(ctx);
        self.execute_cte_query(&modified_stmt, plan, ctx, cte_registry)
    }

    /// Whether the CTE body `stmt` runs blocks under its plan: a subquery,
    /// a set operation branch's and a derived FROM's included, or a join
    fn body_runs_blocks(stmt: &SelectStatement) -> bool {
        let classification = get_classification(stmt);
        classification.where_has_subqueries
            || classification.output_has_subqueries
            || classification.grouped_has_subqueries
            || stmt
                .table_expr
                .as_deref()
                .is_some_and(Self::source_runs_blocks)
            || stmt
                .set_operations
                .iter()
                .any(|op| Self::body_runs_blocks(&op.right))
    }

    /// Whether a FROM source runs blocks under its statement's plan: a join,
    /// which reads its sides through them, or a derived table's statement
    /// holding a subquery
    fn source_runs_blocks(source: &Expression) -> bool {
        match source {
            Expression::JoinSource(_) => true,
            Expression::SubquerySource(derived) => Self::body_runs_blocks(&derived.subquery),
            _ => false,
        }
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

        assert!(registry.get("my_cte").is_some());
        assert!(registry.get("MY_CTE").is_some()); // case-insensitive

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
