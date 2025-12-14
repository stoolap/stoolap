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

//! EXPLAIN statement execution
//!
//! This module handles EXPLAIN and EXPLAIN ANALYZE query plan output,
//! showing the execution strategy and cost estimates for SQL statements.

use std::sync::Arc;

use crate::core::{Result, Row, Value};
use crate::optimizer::feedback::{fingerprint_predicate, global_feedback_cache};
use crate::parser::ast::*;
use crate::storage::traits::{Engine, QueryResult, ScanPlan};

use super::context::ExecutionContext;
use super::parallel;
use super::pushdown;
use super::result::ExecutorMemoryResult;
use super::Executor;

impl Executor {
    /// Execute EXPLAIN statement - shows query plan
    pub(crate) fn execute_explain(
        &self,
        stmt: &ExplainStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut plan_lines: Vec<String> = Vec::new();

        if stmt.analyze {
            // EXPLAIN ANALYZE: Execute the query and collect statistics
            let start = std::time::Instant::now();
            let mut result = self.execute_statement(&stmt.statement, ctx)?;

            // Count rows by iterating through the result
            let mut row_count = 0usize;
            while result.next() {
                row_count += 1;
            }
            let duration = start.elapsed();

            // Format duration nicely
            let time_str = if duration.as_secs() > 0 {
                format!("{:.2}s", duration.as_secs_f64())
            } else if duration.as_millis() > 0 {
                format!(
                    "{:.2}ms",
                    duration.as_millis() as f64 + (duration.as_micros() % 1000) as f64 / 1000.0
                )
            } else {
                format!("{:.2}µs", duration.as_micros() as f64)
            };

            // Record cardinality feedback for SELECT statements with WHERE
            if let Statement::Select(select) = &*stmt.statement {
                if let Some(ref where_clause) = select.where_clause {
                    // Try to get the table name and record feedback
                    if let Some(ref table_expr) = select.table_expr {
                        if let Some(table_name) = extract_table_name(table_expr) {
                            // Compute predicate fingerprint
                            let predicate_hash = fingerprint_predicate(&table_name, where_clause);

                            // Get estimated row count from planner
                            let tx = self.engine.begin_transaction().ok();
                            let estimated_rows = tx
                                .as_ref()
                                .and_then(|tx| tx.get_table(&table_name).ok())
                                .map(|table| {
                                    let stats = self
                                        .get_query_planner()
                                        .get_table_stats_with_fallback(&*table);
                                    stats.row_count as usize
                                })
                                .unwrap_or(row_count);

                            // Record feedback to global cache
                            global_feedback_cache().record_feedback(
                                &table_name,
                                predicate_hash,
                                None, // column_name for more granular tracking
                                estimated_rows as u64,
                                row_count as u64,
                            );
                        }
                    }
                }
            }

            // Generate plan with actual statistics
            self.explain_statement_with_stats(
                &stmt.statement,
                &mut plan_lines,
                0,
                row_count,
                &time_str,
            );

            // Return the plan as a result
            let columns = vec!["plan".to_string()];
            let rows: Vec<Row> = plan_lines
                .into_iter()
                .map(|line| Row::from_values(vec![Value::Text(Arc::from(line.as_str()))]))
                .collect();

            Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
        } else {
            // Regular EXPLAIN: Just show the plan without executing
            self.explain_statement(&stmt.statement, &mut plan_lines, 0);

            // Return as a single-column result
            let columns = vec!["plan".to_string()];
            let rows: Vec<Row> = plan_lines
                .into_iter()
                .map(|line| Row::from_values(vec![Value::Text(Arc::from(line.as_str()))]))
                .collect();

            Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
        }
    }

    /// Generate EXPLAIN output with actual execution statistics
    fn explain_statement_with_stats(
        &self,
        stmt: &Statement,
        lines: &mut Vec<String>,
        indent: usize,
        row_count: usize,
        time_str: &str,
    ) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                lines.push(format!(
                    "{}SELECT (actual time={}, rows={})",
                    prefix, time_str, row_count
                ));
                self.explain_select_columns(select, lines, indent);

                // FROM clause with access plan
                if let Some(ref table_expr) = select.table_expr {
                    self.explain_table_expr_with_where_and_stats(
                        table_expr,
                        select.where_clause.as_deref(),
                        lines,
                        indent + 1,
                        row_count,
                    );
                }

                // GROUP BY
                if !select.group_by.columns.is_empty() {
                    let groups: Vec<String> = select
                        .group_by
                        .columns
                        .iter()
                        .map(|g| format!("{}", g))
                        .collect();
                    lines.push(format!("{}  Group: {}", prefix, groups.join(", ")));
                }

                // HAVING
                if let Some(ref having) = select.having {
                    lines.push(format!("{}  Having: {}", prefix, having));
                }

                // ORDER BY
                if !select.order_by.is_empty() {
                    let orders: Vec<String> = select
                        .order_by
                        .iter()
                        .map(|o| {
                            let dir = if !o.ascending { " DESC" } else { "" };
                            format!("{}{}", o.expression, dir)
                        })
                        .collect();
                    lines.push(format!("{}  Order: {}", prefix, orders.join(", ")));
                }

                // LIMIT/OFFSET
                if let Some(ref limit) = select.limit {
                    lines.push(format!("{}  Limit: {}", prefix, limit));
                }
                if let Some(ref offset) = select.offset {
                    lines.push(format!("{}  Offset: {}", prefix, offset));
                }
            }
            Statement::Insert(insert) => {
                lines.push(format!(
                    "{}INSERT INTO {} (actual time={}, rows={})",
                    prefix, insert.table_name, time_str, row_count
                ));
                if let Some(ref select) = insert.select {
                    lines.push(format!("{}  Source:", prefix));
                    self.explain_select(select, lines, indent + 2);
                } else {
                    lines.push(format!(
                        "{}  Values: {} row(s)",
                        prefix,
                        insert.values.len()
                    ));
                }
            }
            Statement::Update(update) => {
                lines.push(format!(
                    "{}UPDATE {} (actual time={}, rows={})",
                    prefix, update.table_name, time_str, row_count
                ));
                lines.push(format!(
                    "{}  Set: {} column(s)",
                    prefix,
                    update.updates.len()
                ));
                if let Some(ref where_clause) = update.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            Statement::Delete(delete) => {
                lines.push(format!(
                    "{}DELETE FROM {} (actual time={}, rows={})",
                    prefix, delete.table_name, time_str, row_count
                ));
                if let Some(ref where_clause) = delete.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            _ => {
                lines.push(format!(
                    "{}Statement: {} (actual time={}, rows={})",
                    prefix, stmt, time_str, row_count
                ));
            }
        }
    }

    /// Helper to show just the SELECT columns
    fn explain_select_columns(
        &self,
        select: &SelectStatement,
        lines: &mut Vec<String>,
        indent: usize,
    ) {
        let prefix = "  ".repeat(indent);

        // Show columns
        let col_count = select.columns.len();
        if col_count <= 5 {
            let cols: Vec<String> = select.columns.iter().map(|c| format!("{}", c)).collect();
            lines.push(format!("{}  Columns: {}", prefix, cols.join(", ")));
        } else {
            lines.push(format!("{}  Columns: {} column(s)", prefix, col_count));
        }
    }

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis and stats
    fn explain_table_expr_with_where_and_stats(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        row_count: usize,
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.engine.begin_transaction() {
                    if let Ok(table) = tx.get_table(&simple.name.value) {
                        // Build storage expression from WHERE clause for analysis
                        let storage_expr = if let Some(where_expr) = where_clause {
                            let schema = table.schema();
                            let (expr, _) = pushdown::try_pushdown(where_expr, schema, None);
                            expr
                        } else {
                            None
                        };

                        // Get the scan plan
                        let scan_plan = table.explain_scan(storage_expr.as_deref());

                        // For SeqScan, use the AST expression's Display format instead of storage expr Debug
                        // Check if parallel execution would be used based on TABLE's row count (not output rows)
                        // Parallel decision is based on input size, not filtered output
                        let parallel_config = parallel::ParallelConfig::default();
                        let table_row_count = table.row_count();
                        let would_use_parallel = where_clause.is_some()
                            && parallel_config.should_parallel_filter(table_row_count);

                        let scan_plan = match scan_plan {
                            ScanPlan::SeqScan {
                                table: tbl,
                                filter: _,
                            } if where_clause.is_some() => {
                                let filter_str = Some(format!("{}", where_clause.unwrap()));
                                if would_use_parallel {
                                    ScanPlan::ParallelSeqScan {
                                        table: tbl,
                                        filter: filter_str,
                                        workers: rayon::current_num_threads(),
                                    }
                                } else {
                                    ScanPlan::SeqScan {
                                        table: tbl,
                                        filter: filter_str,
                                    }
                                }
                            }
                            other => other,
                        };

                        // Format the scan plan with actual stats
                        let plan_str = format!("{}", scan_plan);
                        for (i, line) in plan_str.lines().enumerate() {
                            if i == 0 {
                                lines.push(format!(
                                    "{}-> {} (actual rows={})",
                                    prefix, line, row_count
                                ));
                            } else {
                                lines.push(format!("{}   {}", prefix, line));
                            }
                        }

                        // Add alias if present
                        if let Some(ref alias) = simple.alias {
                            lines.push(format!("{}   Alias: {}", prefix, alias));
                        }

                        return;
                    }
                }

                // Fallback if table not found
                let mut table_info = format!(
                    "{}-> Seq Scan on {} (actual rows={})",
                    prefix, simple.name, row_count
                );
                if let Some(ref alias) = simple.alias {
                    table_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(table_info);
                if let Some(ref where_expr) = where_clause {
                    lines.push(format!("{}   Filter: {}", prefix, where_expr));
                }
            }
            Expression::SubquerySource(subquery) => {
                let mut sub_info =
                    format!("{}-> Subquery Scan (actual rows={})", prefix, row_count);
                if let Some(ref alias) = subquery.alias {
                    sub_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(sub_info);
                self.explain_select(&subquery.subquery, lines, indent + 1);
            }
            Expression::JoinSource(join) => {
                // Determine join algorithm based on condition
                let join_algorithm = if join.condition.is_none() && join.using_columns.is_empty() {
                    "Nested Loop"
                } else if let Some(ref cond) = join.condition {
                    if is_equality_condition(cond) {
                        "Hash Join"
                    } else {
                        "Nested Loop"
                    }
                } else {
                    "Hash Join" // USING clause implies equality
                };

                lines.push(format!(
                    "{}-> {} ({} Join) (actual rows={})",
                    prefix, join_algorithm, join.join_type, row_count
                ));
                if let Some(ref condition) = join.condition {
                    lines.push(format!("{}   Join Cond: {}", prefix, condition));
                }
                if !join.using_columns.is_empty() {
                    let cols: Vec<String> =
                        join.using_columns.iter().map(|c| c.to_string()).collect();
                    lines.push(format!("{}   Using: ({})", prefix, cols.join(", ")));
                }
                // Left side gets the WHERE clause for potential pushdown
                self.explain_table_expr_with_where(&join.left, where_clause, lines, indent + 1);
                // Right side typically doesn't get the outer WHERE
                self.explain_table_expr_with_where(&join.right, None, lines, indent + 1);
            }
            Expression::CteReference(cte_ref) => {
                let mut cte_info = format!(
                    "{}-> CTE Scan on {} (actual rows={})",
                    prefix, cte_ref.name, row_count
                );
                if let Some(ref alias) = cte_ref.alias {
                    cte_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(cte_info);
            }
            _ => {
                lines.push(format!(
                    "{}-> Scan: {} (actual rows={})",
                    prefix, expr, row_count
                ));
            }
        }
    }

    /// Generate EXPLAIN output for a statement
    fn explain_statement(&self, stmt: &Statement, lines: &mut Vec<String>, indent: usize) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                self.explain_select(select, lines, indent);
            }
            Statement::Insert(insert) => {
                lines.push(format!("{}INSERT INTO {}", prefix, insert.table_name));
                if let Some(ref select) = insert.select {
                    lines.push(format!("{}  Source:", prefix));
                    self.explain_select(select, lines, indent + 2);
                } else {
                    lines.push(format!(
                        "{}  Values: {} row(s)",
                        prefix,
                        insert.values.len()
                    ));
                }
            }
            Statement::Update(update) => {
                lines.push(format!("{}UPDATE {}", prefix, update.table_name));
                lines.push(format!(
                    "{}  Set: {} column(s)",
                    prefix,
                    update.updates.len()
                ));
                if let Some(ref where_clause) = update.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            Statement::Delete(delete) => {
                lines.push(format!("{}DELETE FROM {}", prefix, delete.table_name));
                if let Some(ref where_clause) = delete.where_clause {
                    lines.push(format!("{}  Filter: {}", prefix, where_clause));
                }
            }
            _ => {
                lines.push(format!("{}Statement: {}", prefix, stmt));
            }
        }
    }

    /// Generate EXPLAIN output for a SELECT statement
    fn explain_select(&self, select: &SelectStatement, lines: &mut Vec<String>, indent: usize) {
        let prefix = "  ".repeat(indent);

        // CTE info
        if let Some(ref with) = select.with {
            lines.push(format!("{}WITH (CTEs: {})", prefix, with.ctes.len()));
            for cte in &with.ctes {
                lines.push(format!(
                    "{}  {} = ({})",
                    prefix,
                    cte.name,
                    if cte.is_recursive {
                        "RECURSIVE"
                    } else {
                        "non-recursive"
                    }
                ));
            }
        }

        // Main operation
        if select.distinct {
            lines.push(format!("{}SELECT DISTINCT", prefix));
        } else {
            lines.push(format!("{}SELECT", prefix));
        }

        // Columns
        let col_count = select.columns.len();
        if col_count <= 3 {
            let cols: Vec<String> = select.columns.iter().map(|c| format!("{}", c)).collect();
            lines.push(format!("{}  Columns: {}", prefix, cols.join(", ")));
        } else {
            lines.push(format!("{}  Columns: {} column(s)", prefix, col_count));
        }

        // FROM clause with access plan
        if let Some(ref table_expr) = select.table_expr {
            self.explain_table_expr_with_where(
                table_expr,
                select.where_clause.as_deref(),
                lines,
                indent + 1,
            );
        }

        // GROUP BY
        if !select.group_by.columns.is_empty() {
            let groups: Vec<String> = select
                .group_by
                .columns
                .iter()
                .map(|g| format!("{}", g))
                .collect();
            lines.push(format!("{}  Group By: {}", prefix, groups.join(", ")));
        }

        // HAVING
        if let Some(ref having) = select.having {
            lines.push(format!("{}  Having: {}", prefix, having));
        }

        // ORDER BY
        if !select.order_by.is_empty() {
            let orders: Vec<String> = select.order_by.iter().map(|o| format!("{}", o)).collect();
            lines.push(format!("{}  Order By: {}", prefix, orders.join(", ")));
        }

        // LIMIT/OFFSET
        if let Some(ref limit) = select.limit {
            lines.push(format!("{}  Limit: {}", prefix, limit));
        }
        if let Some(ref offset) = select.offset {
            lines.push(format!("{}  Offset: {}", prefix, offset));
        }

        // Set operations
        if !select.set_operations.is_empty() {
            for set_op in &select.set_operations {
                lines.push(format!("{}  {}", prefix, set_op.operation));
                self.explain_select(&set_op.right, lines, indent + 2);
            }
        }
    }

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis
    fn explain_table_expr_with_where(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.engine.begin_transaction() {
                    if let Ok(table) = tx.get_table(&simple.name.value) {
                        // Build storage expression from WHERE clause for analysis
                        let storage_expr = if let Some(where_expr) = where_clause {
                            let schema = table.schema();
                            let (expr, _) = pushdown::try_pushdown(where_expr, schema, None);
                            expr
                        } else {
                            None
                        };

                        // Get the scan plan
                        let scan_plan = table.explain_scan(storage_expr.as_deref());

                        // For SeqScan, use the AST expression's Display format instead of storage expr Debug
                        let scan_plan = match scan_plan {
                            ScanPlan::SeqScan { table, filter: _ } if where_clause.is_some() => {
                                ScanPlan::SeqScan {
                                    table,
                                    filter: Some(format!("{}", where_clause.unwrap())),
                                }
                            }
                            other => other,
                        };

                        // Format the scan plan with indentation
                        let plan_str = format!("{}", scan_plan);
                        for (i, line) in plan_str.lines().enumerate() {
                            if i == 0 {
                                lines.push(format!("{}-> {}", prefix, line));
                            } else {
                                lines.push(format!("{}   {}", prefix, line));
                            }
                        }

                        // Add alias if present
                        if let Some(ref alias) = simple.alias {
                            lines.push(format!("{}   Alias: {}", prefix, alias));
                        }

                        return;
                    }
                }

                // Fallback if table not found
                let mut table_info = format!("{}-> Seq Scan on {}", prefix, simple.name);
                if let Some(ref alias) = simple.alias {
                    table_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(table_info);
                if let Some(ref where_expr) = where_clause {
                    lines.push(format!("{}   Filter: {}", prefix, where_expr));
                }
            }
            Expression::SubquerySource(subquery) => {
                let mut sub_info = format!("{}-> Subquery Scan", prefix);
                if let Some(ref alias) = subquery.alias {
                    sub_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(sub_info);
                self.explain_select(&subquery.subquery, lines, indent + 1);
            }
            Expression::JoinSource(join) => {
                // Determine join algorithm based on condition
                let join_algorithm = if join.condition.is_none() && join.using_columns.is_empty() {
                    // CROSS JOIN or no condition -> Nested Loop
                    "Nested Loop"
                } else if let Some(ref cond) = join.condition {
                    // Check if it's an equality join (a.col = b.col)
                    if is_equality_condition(cond) {
                        "Hash Join"
                    } else {
                        // Range condition or complex join
                        "Nested Loop"
                    }
                } else {
                    // USING clause implies equality join
                    "Hash Join"
                };

                // Get cost estimate from query planner
                let planner = self.get_query_planner();
                let left_table_name = extract_table_name(&join.left);
                let right_table_name = extract_table_name(&join.right);

                // Get table statistics for cost estimation
                let left_stats = left_table_name
                    .as_ref()
                    .and_then(|name| planner.get_table_stats(name));
                let right_stats = right_table_name
                    .as_ref()
                    .and_then(|name| planner.get_table_stats(name));

                // Calculate estimated rows and cost
                let (estimated_rows, estimated_cost) = match (left_stats, right_stats) {
                    (Some(ls), Some(rs)) => {
                        // Hash join cost estimation
                        let left_rows = ls.row_count.max(1);
                        let right_rows = rs.row_count.max(1);
                        // Simplified join cardinality estimate
                        let rows = if join_algorithm == "Nested Loop" && join.condition.is_none() {
                            // Cross join: left * right
                            left_rows * right_rows
                        } else {
                            // Equality join: estimate as smaller side (pessimistic)
                            left_rows.min(right_rows)
                        };
                        // Cost = build cost + probe cost
                        let cost = if join_algorithm == "Hash Join" {
                            (left_rows.min(right_rows) as f64)
                                + (left_rows.max(right_rows) as f64 * 0.1)
                        } else {
                            // Nested loop: O(n*m) but with early termination
                            (left_rows as f64) * (right_rows as f64).sqrt()
                        };
                        (rows, cost)
                    }
                    (Some(ls), None) => {
                        // Only left stats available
                        (ls.row_count, ls.row_count as f64 * 10.0)
                    }
                    (None, Some(rs)) => {
                        // Only right stats available
                        (rs.row_count, rs.row_count as f64 * 10.0)
                    }
                    (None, None) => {
                        // No stats - use default estimate
                        (1000, 10000.0)
                    }
                };

                // Show join algorithm, type, cost and rows
                lines.push(format!(
                    "{}-> {} ({} Join) (cost={:.2} rows={})",
                    prefix, join_algorithm, join.join_type, estimated_cost, estimated_rows
                ));
                if let Some(ref condition) = join.condition {
                    lines.push(format!("{}   Join Cond: {}", prefix, condition));
                }
                if !join.using_columns.is_empty() {
                    let cols: Vec<String> =
                        join.using_columns.iter().map(|c| c.to_string()).collect();
                    lines.push(format!("{}   Using: ({})", prefix, cols.join(", ")));
                }
                // Left side gets the WHERE clause for potential pushdown
                self.explain_table_expr_with_where(&join.left, where_clause, lines, indent + 1);
                // Right side typically doesn't get the outer WHERE
                self.explain_table_expr_with_where(&join.right, None, lines, indent + 1);
            }
            Expression::CteReference(cte_ref) => {
                let mut cte_info = format!("{}-> CTE Scan on {}", prefix, cte_ref.name);
                if let Some(ref alias) = cte_ref.alias {
                    cte_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(cte_info);
            }
            _ => {
                lines.push(format!("{}-> Scan: {}", prefix, expr));
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if an expression is an equality condition (for EXPLAIN join algorithm display)
pub(crate) fn is_equality_condition(expr: &Expression) -> bool {
    match expr {
        Expression::Infix(infix) => {
            // Check for equality operator
            if infix.operator == "=" {
                // Check that both sides are column references (not literals)
                let left_is_col = matches!(
                    infix.left.as_ref(),
                    Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                );
                let right_is_col = matches!(
                    infix.right.as_ref(),
                    Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
                );
                left_is_col && right_is_col
            } else if infix.operator.eq_ignore_ascii_case("AND") {
                // AND condition - check if any part is an equality join
                is_equality_condition(&infix.left) || is_equality_condition(&infix.right)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Extract table name from a table expression (for statistics lookup)
pub(crate) fn extract_table_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::TableSource(simple) => Some(simple.name.value.clone()),
        Expression::JoinSource(join) => extract_table_name(&join.left),
        Expression::SubquerySource(_) => None, // Can't get stats for subquery
        _ => None,
    }
}
