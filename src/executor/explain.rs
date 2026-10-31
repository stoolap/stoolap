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

use crate::common::SmartString;
use crate::core::{Result, Row, RowVec, Value};
use crate::optimizer::feedback::{fingerprint_predicate, global_feedback_cache};
use crate::parser::ast::*;
use crate::planner::join::{pair_method, PairMethod};
use crate::storage::traits::transaction::{StatementTable, TableUse};
use crate::storage::traits::{QueryResult, ScanPlan};

use super::context::ExecutionContext;
use super::operators::index_nested_loop::IndexLookupStrategy;
use super::parallel;
use super::planned_dml::DmlStatement;
use super::planned_join::pick_index_join_side;
use super::prepared_join::JoinSides;
use super::prepared_select::OutOfFamily;
use super::pushdown;
use super::query::{split_join_where, CteNames, JoinIndexUse, SplitSide};
use super::query_cache::NestedPlan;
use super::query_classification::{get_classification, QueryClassification};
use super::result::ExecutorResult;
use super::utils::get_table_alias_from_expr;
use super::Executor;

impl Executor {
    /// Execute EXPLAIN statement - shows query plan; EXPLAIN ANALYZE runs
    /// its statement under the plan `plan` holds for it
    pub(crate) fn execute_explain(
        &self,
        stmt: &ExplainStatement,
        ctx: &ExecutionContext,
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        let mut plan_lines: Vec<String> = Vec::new();

        // For SELECT statements with CTEs, try to get the inlined version for accurate EXPLAIN
        let inlined_statement: Option<Box<Statement>> =
            if let Statement::Select(select) = &*stmt.statement {
                if let Some(ref with_clause) = select.with {
                    self.try_inline_ctes(select, with_clause)
                        .map(|inlined| Box::new(Statement::Select(inlined)))
                } else {
                    None
                }
            } else {
                None
            };

        // Use inlined statement if available, otherwise original
        let explain_stmt = inlined_statement.as_ref().unwrap_or(&stmt.statement);

        if stmt.analyze {
            // EXPLAIN ANALYZE: Execute the query and collect statistics
            let start = crate::common::time_compat::Instant::now();
            let holds = super::statement_holds_as_of(&stmt.statement);
            let ctx =
                self.with_statement_cut(&stmt.statement, std::borrow::Cow::Borrowed(ctx), holds)?;
            let mut result = self.execute_statement_inner(
                &stmt.statement,
                ctx,
                self.active_txn_id(),
                None,
                plan.and_then(NestedPlan::source),
                None,
            )?;

            // Count rows by iterating through the result
            let mut row_count = 0usize;
            while result.next() {
                row_count += 1;
            }
            if let Some(err) = result.last_error() {
                return Err(err);
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
                            let tx = self.begin_transaction().ok();
                            let estimated_rows = tx
                                .as_ref()
                                .and_then(|tx| {
                                    tx.get_read_table_at(&table_name, None, TableUse::Read).ok()
                                })
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

            // A SELECT run as a parsed tree held the plan its structure keys
            let held = plan.is_some()
                || matches!(&*stmt.statement, Statement::Select(select)
                    if self.query_cache.statement_plan(select).is_some());
            // Generate plan with actual statistics (using inlined version if available)
            self.explain_statement_with_stats(
                explain_stmt,
                &mut plan_lines,
                0,
                (row_count, &time_str),
                held,
            );

            // Return the plan as a result
            let columns = vec!["plan".to_string()];
            let rows: RowVec = plan_lines
                .into_iter()
                .enumerate()
                .map(|(i, line)| {
                    (
                        i as i64,
                        Row::from_values(vec![Value::Text(SmartString::from_string(line))]),
                    )
                })
                .collect();

            Ok(Box::new(ExecutorResult::new(columns, rows)))
        } else {
            // Regular EXPLAIN: Just show the plan without executing (using inlined version if available)
            let planned = match &**explain_stmt {
                Statement::Select(select) => self.explain_plan(select, ctx),
                Statement::Update(update) => self.explain_dml(DmlStatement::Update(update)),
                Statement::Delete(delete) => self.explain_dml(DmlStatement::Delete(delete)),
                _ => None,
            };
            match planned {
                Some(lines) => plan_lines = lines,
                None => self.explain_statement(explain_stmt, &mut plan_lines, 0),
            }

            // Return as a single-column result
            let columns = vec!["plan".to_string()];
            let rows: RowVec = plan_lines
                .into_iter()
                .enumerate()
                .map(|(i, line)| {
                    (
                        i as i64,
                        Row::from_values(vec![Value::Text(SmartString::from_string(line))]),
                    )
                })
                .collect();

            Ok(Box::new(ExecutorResult::new(columns, rows)))
        }
    }

    /// Generate EXPLAIN output with actual execution statistics, `held`
    /// when `stmt` ran under a plan
    fn explain_statement_with_stats(
        &self,
        stmt: &Statement,
        lines: &mut Vec<String>,
        indent: usize,
        (row_count, time_str): (usize, &str),
        held: bool,
    ) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                let distinct_str = if select.distinct { " DISTINCT" } else { "" };
                lines.push(format!(
                    "{}SELECT{} (actual time={}, rows={})",
                    prefix, distinct_str, time_str, row_count
                ));
                self.explain_select_columns(select, lines, indent);

                // FROM clause with access plan
                // Check for vector search fast path first
                let vector_plan = if let Some(ref table_expr) = select.table_expr {
                    if let Some(table_name) = extract_table_name(table_expr) {
                        self.detect_vector_search_plan(select, &table_name)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(ref vplan) = vector_plan {
                    let plan_str = format!("{}", vplan);
                    let inner_prefix = "  ".repeat(indent + 1);
                    for (i, line) in plan_str.lines().enumerate() {
                        if i == 0 {
                            lines.push(format!(
                                "{}-> {} (actual rows={})",
                                inner_prefix, line, row_count
                            ));
                        } else {
                            lines.push(format!("{}   {}", inner_prefix, line));
                        }
                    }
                } else if let Some(ref table_expr) = select.table_expr {
                    let classification = get_classification(select);
                    let grouped = Self::grouped_family_reach(select, &classification)
                        .then(|| self.grouped_family(select, None));
                    let from_plan =
                        Self::sources_hold_plan(select, &classification, grouped.as_ref(), held);
                    self.explain_table_expr_with_where_and_stats(
                        table_expr,
                        select.where_clause.as_deref(),
                        lines,
                        indent + 1,
                        row_count,
                        self.join_index_use(select, self.join_runs_blocks(select, held)),
                        &cte_names(select),
                        (from_plan, select.with.as_ref()),
                    );
                }

                // GROUP BY (including ROLLUP, CUBE, GROUPING SETS)
                {
                    let gb_str = format!("{}", select.group_by);
                    if !gb_str.is_empty() {
                        lines.push(format!("{}  Group By: {}", prefix, gb_str));
                    }
                }

                // HAVING
                if let Some(ref having) = select.having {
                    lines.push(format!("{}  Having: {}", prefix, having));
                }

                // ORDER BY
                if !select.order_by.is_empty() {
                    let orders: Vec<String> =
                        select.order_by.iter().map(|o| format!("{}", o)).collect();
                    lines.push(format!("{}  Order By: {}", prefix, orders.join(", ")));
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
                    self.explain_select(select, lines, indent + 2, held);
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

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis and stats,
    /// `plans` as `explain_table_expr_with_where` takes them
    #[allow(clippy::too_many_arguments)]
    fn explain_table_expr_with_where_and_stats(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        row_count: usize,
        index_use: JoinIndexUse,
        ctes: &[String],
        plans: (bool, Option<&WithClause>),
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.begin_transaction() {
                    if let Ok(table) =
                        tx.get_read_table_at(&simple.name.value, None, TableUse::Read)
                    {
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
                        let table_row_count = table.row_count_hint();
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
                                        workers: {
                                            #[cfg(feature = "parallel")]
                                            {
                                                rayon::current_num_threads()
                                            }
                                            #[cfg(not(feature = "parallel"))]
                                            {
                                                1
                                            }
                                        },
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
                if let Some(wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
                self.explain_select(&subquery.subquery, lines, indent + 1, plans.0);
            }
            Expression::JoinSource(join) => {
                let sides = self.unrun_join_sides(join, plans.1);
                // Determine join algorithm including INLJ check
                let (join_algorithm, _) = self.determine_join_algorithm_for_explain(
                    &join.left,
                    &join.right,
                    join.condition.as_deref(),
                    join.join_type.as_ref(),
                    &join.using_columns,
                    where_clause,
                    index_use,
                    ctes,
                    sides.as_ref(),
                );

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
                // Partition WHERE predicates by which table they reference
                let (left_where, right_where, join_filter) = match where_clause {
                    Some(wc) => self.explain_where_split(wc, join, sides.as_ref(), ctes),
                    None => (None, None, None),
                };
                if let Some(ref jf) = join_filter {
                    lines.push(format!("{}   Join Filter: {}", prefix, jf));
                }
                self.explain_table_expr_plan_only(
                    &join.left,
                    left_where.as_ref(),
                    lines,
                    indent + 1,
                    index_use,
                    ctes,
                    plans,
                );
                self.explain_table_expr_plan_only(
                    &join.right,
                    right_where.as_ref(),
                    lines,
                    indent + 1,
                    index_use,
                    ctes,
                    plans,
                );
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
            Expression::FunctionTableSource(fs) => {
                let mut info = format!(
                    "{}-> Function Scan on {} (actual rows={})",
                    prefix, fs.function.value, row_count
                );
                if let Some(ref alias) = fs.alias {
                    info.push_str(&format!(" AS {}", alias.value));
                }
                lines.push(info);
                if let Some(wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
            }
            Expression::ValuesSource(vs) => {
                let mut info = format!("{}-> Values Scan (actual rows={})", prefix, row_count);
                if let Some(ref alias) = vs.alias {
                    info.push_str(&format!(" AS {}", alias));
                }
                lines.push(info);
                if let Some(wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
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
    pub(super) fn explain_statement(
        &self,
        stmt: &Statement,
        lines: &mut Vec<String>,
        indent: usize,
    ) {
        let prefix = "  ".repeat(indent);

        match stmt {
            Statement::Select(select) => {
                self.explain_select(select, lines, indent, true);
            }
            Statement::Insert(insert) => {
                lines.push(format!("{}INSERT INTO {}", prefix, insert.table_name));
                if let Some(ref select) = insert.select {
                    lines.push(format!("{}  Source:", prefix));
                    self.explain_select(select, lines, indent + 2, true);
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

    /// Whether the join `select` reads from runs its subqueries as prepared
    /// blocks, as the statement run alone decides it, or why not; `own`
    /// when `select` holds a plan, `with` the clause naming its CTEs
    fn join_prepared(
        &self,
        select: &SelectStatement,
        join: &JoinTableSource,
        own: bool,
        with: Option<&WithClause>,
    ) -> std::result::Result<(), OutOfFamily> {
        if !own {
            return Err("a join in a SELECT that holds no plan");
        }
        // A join with CTEs keeps its plan as execution keeps it, its CTE
        // sides named by their bodies
        let ctx = match with {
            Some(with) if self.join_keeps_plan(select, with, false) => {
                self.named_ctes(with, false).map_err(|_| "a WITH clause")?
            }
            Some(_) => return Err("a WITH clause"),
            None => ExecutionContext::new(),
        };
        if Self::folds_select_subqueries(select, &get_classification(select)) {
            return Err("a grouped SELECT list whose subqueries are read first");
        }
        // EXPLAIN runs no view to name it
        let sides = self.join_side_schemas(join, &ctx, false);
        self.join_blocks_family(
            select,
            join,
            &join.join_type.to_uppercase(),
            (
                get_table_alias_from_expr(&join.left).as_deref(),
                get_table_alias_from_expr(&join.right).as_deref(),
            ),
            sides.as_ref(),
        )
    }

    /// Whether the join `select` reads from runs its subqueries as prepared
    /// blocks, `own` when `select` holds a plan
    fn join_runs_blocks(&self, select: &SelectStatement, own: bool) -> bool {
        match select.table_expr.as_deref() {
            Some(Expression::JoinSource(join)) => self
                .join_prepared(select, join, own, select.with.as_ref())
                .is_ok(),
            _ => false,
        }
    }

    /// Whether a derived FROM or join side of `select` holds a plan in its
    /// statement's, `own` when `select` holds one: a statement reading its
    /// SELECT list's subqueries first outside the grouped family keeps none
    fn sources_hold_plan(
        select: &SelectStatement,
        classification: &QueryClassification,
        grouped: Option<&std::result::Result<(), OutOfFamily>>,
        own: bool,
    ) -> bool {
        own && !(Self::folds_select_subqueries(select, classification)
            && !matches!(grouped, Some(Ok(()))))
    }

    /// Generate EXPLAIN output for a SELECT statement; `own` when it holds a
    /// plan: the statement's own, an INSERT's source, or a branch, a derived
    /// FROM or a join side of one that does
    fn explain_select(
        &self,
        select: &SelectStatement,
        lines: &mut Vec<String>,
        indent: usize,
        own: bool,
    ) {
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

        // Columns - use same threshold as EXPLAIN ANALYZE (5 columns)
        let col_count = select.columns.len();
        if col_count <= 5 {
            let cols: Vec<String> = select.columns.iter().map(|c| format!("{}", c)).collect();
            lines.push(format!("{}  Columns: {}", prefix, cols.join(", ")));
        } else {
            lines.push(format!("{}  Columns: {} column(s)", prefix, col_count));
        }

        let join = match select.table_expr.as_deref() {
            Some(Expression::JoinSource(join)) => Some(&**join),
            _ => None,
        };
        let on_subqueries = join.is_some_and(Self::ons_hold_subqueries);
        let classification = get_classification(select);
        let grouped_output = Self::grouped_family_reach(select, &classification);
        let grouped = grouped_output.then(|| self.grouped_family(select, None));
        let from_plan = Self::sources_hold_plan(select, &classification, grouped.as_ref(), own);
        if on_subqueries
            || classification.output_has_subqueries
            || grouped_output
            || select
                .where_clause
                .as_deref()
                .is_some_and(Self::has_subqueries)
        {
            let prepared = match (grouped, join) {
                (Some(grouped), _) => grouped,
                (None, Some(join)) => self.join_prepared(select, join, own, select.with.as_ref()),
                (None, None) => self.prepared_family(select, None),
            };
            match prepared {
                Ok(()) => lines.push(format!("{}  Prepared: yes", prefix)),
                Err(reason) => lines.push(format!("{}  Prepared: no ({})", prefix, reason)),
            }
        }

        // FROM clause with access plan
        // Check for vector search fast path first
        let vector_plan = if let Some(ref table_expr) = select.table_expr {
            if let Some(table_name) = extract_table_name(table_expr) {
                self.detect_vector_search_plan(select, &table_name)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref vplan) = vector_plan {
            // Vector search detected: show vector-specific scan plan
            let plan_str = format!("{}", vplan);
            let inner_prefix = "  ".repeat(indent + 1);
            for (i, line) in plan_str.lines().enumerate() {
                if i == 0 {
                    lines.push(format!("{}-> {}", inner_prefix, line));
                } else {
                    lines.push(format!("{}   {}", inner_prefix, line));
                }
            }
        } else if let Some(ref table_expr) = select.table_expr {
            self.explain_table_expr_with_where(
                table_expr,
                select.where_clause.as_deref(),
                lines,
                indent + 1,
                self.join_index_use(select, self.join_runs_blocks(select, own)),
                &cte_names(select),
                (from_plan, select.with.as_ref()),
            );
        }

        // GROUP BY (including ROLLUP, CUBE, GROUPING SETS)
        {
            let gb_str = format!("{}", select.group_by);
            if !gb_str.is_empty() {
                lines.push(format!("{}  Group By: {}", prefix, gb_str));
            }
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
                self.explain_select(&set_op.right, lines, indent + 2, own);
            }
        }
    }

    /// Generate EXPLAIN output for a table expression with WHERE clause analysis
    /// Show plan structure without join cost estimates (for EXPLAIN ANALYZE child nodes)
    #[allow(clippy::too_many_arguments)]
    fn explain_table_expr_plan_only(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        index_use: JoinIndexUse,
        ctes: &[String],
        plans: (bool, Option<&WithClause>),
    ) {
        self.explain_table_expr_inner(
            expr,
            where_clause,
            lines,
            indent,
            false,
            index_use,
            ctes,
            plans,
        )
    }

    /// `plans`: whether `expr`, a derived table or a join of them, holds a
    /// plan for each, and the WITH clause naming the statement's CTEs
    #[allow(clippy::too_many_arguments)]
    fn explain_table_expr_with_where(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        index_use: JoinIndexUse,
        ctes: &[String],
        plans: (bool, Option<&WithClause>),
    ) {
        self.explain_table_expr_inner(
            expr,
            where_clause,
            lines,
            indent,
            true,
            index_use,
            ctes,
            plans,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn explain_table_expr_inner(
        &self,
        expr: &Expression,
        where_clause: Option<&Expression>,
        lines: &mut Vec<String>,
        indent: usize,
        show_join_cost: bool,
        index_use: JoinIndexUse,
        ctes: &[String],
        (from_plan, with): (bool, Option<&WithClause>),
    ) {
        let prefix = "  ".repeat(indent);

        match expr {
            Expression::TableSource(simple) => {
                // Try to get the table and analyze access plan
                if let Ok(tx) = self.begin_transaction() {
                    if let Ok(table) =
                        tx.get_read_table_at(&simple.name.value, None, TableUse::Read)
                    {
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
                if let Some(wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
                self.explain_select(&subquery.subquery, lines, indent + 1, from_plan);
            }
            Expression::JoinSource(join) => {
                let sides = self.unrun_join_sides(join, with);
                // Determine join algorithm including INLJ check
                let (join_algorithm, _) = self.determine_join_algorithm_for_explain(
                    &join.left,
                    &join.right,
                    join.condition.as_deref(),
                    join.join_type.as_ref(),
                    &join.using_columns,
                    where_clause,
                    index_use,
                    ctes,
                    sides.as_ref(),
                );

                if show_join_cost {
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
                            let rows =
                                if join_algorithm == "Nested Loop" && join.condition.is_none() {
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
                } else {
                    // Plan-only mode (EXPLAIN ANALYZE child nodes): no cost estimates
                    lines.push(format!(
                        "{}-> {} ({} Join)",
                        prefix, join_algorithm, join.join_type
                    ));
                }
                if let Some(ref condition) = join.condition {
                    lines.push(format!("{}   Join Cond: {}", prefix, condition));
                }
                if !join.using_columns.is_empty() {
                    let cols: Vec<String> =
                        join.using_columns.iter().map(|c| c.to_string()).collect();
                    lines.push(format!("{}   Using: ({})", prefix, cols.join(", ")));
                }
                // Partition WHERE predicates by which table they reference
                let (left_where, right_where, join_filter) = match where_clause {
                    Some(wc) => self.explain_where_split(wc, join, sides.as_ref(), ctes),
                    None => (None, None, None),
                };
                if let Some(ref jf) = join_filter {
                    lines.push(format!("{}   Join Filter: {}", prefix, jf));
                }
                self.explain_table_expr_inner(
                    &join.left,
                    left_where.as_ref(),
                    lines,
                    indent + 1,
                    show_join_cost,
                    index_use,
                    ctes,
                    (from_plan, with),
                );
                self.explain_table_expr_inner(
                    &join.right,
                    right_where.as_ref(),
                    lines,
                    indent + 1,
                    show_join_cost,
                    index_use,
                    ctes,
                    (from_plan, with),
                );
            }
            Expression::CteReference(cte_ref) => {
                let mut cte_info = format!("{}-> CTE Scan on {}", prefix, cte_ref.name);
                if let Some(ref alias) = cte_ref.alias {
                    cte_info.push_str(&format!(" AS {}", alias));
                }
                lines.push(cte_info);
            }
            Expression::FunctionTableSource(fs) => {
                let mut info = format!("{}-> Function Scan on {}", prefix, fs.function.value);
                if let Some(ref alias) = fs.alias {
                    info.push_str(&format!(" AS {}", alias.value));
                }
                lines.push(info);
                if let Some(ref wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
            }
            Expression::ValuesSource(vs) => {
                let mut info = format!("{}-> Values Scan", prefix);
                if let Some(ref alias) = vs.alias {
                    info.push_str(&format!(" AS {}", alias));
                }
                lines.push(info);
                if let Some(ref wc) = where_clause {
                    lines.push(format!("{}     Filter: {}", prefix, wc));
                }
            }
            _ => {
                lines.push(format!("{}-> Scan: {}", prefix, expr));
            }
        }
    }

    /// The WHERE of a join over `join` split as its run splits it, over
    /// `sides` as its blocks read them when they are known
    fn explain_where_split(
        &self,
        where_clause: &Expression,
        join: &JoinTableSource,
        sides: Option<&JoinSides>,
        ctes: &[String],
    ) -> (Option<Expression>, Option<Expression>, Option<Expression>) {
        let schemas = [&join.left, &join.right]
            .map(|side| self.base_table_side_schema(side, CteNames::Listed(ctes)));
        let aliases = [&join.left, &join.right].map(|side| get_table_alias_from_expr(side));
        split_join_where(
            where_clause,
            SplitSide::both(
                sides,
                [aliases[0].as_deref(), aliases[1].as_deref()],
                [schemas[0].as_deref(), schemas[1].as_deref()],
            ),
            &join.join_type.to_uppercase(),
        )
    }

    /// Determine the join algorithm for EXPLAIN output
    /// This uses the same rules as actual execution for consistency
    #[allow(clippy::too_many_arguments)]
    fn determine_join_algorithm_for_explain(
        &self,
        join_left: &Expression,
        join_right: &Expression,
        join_condition: Option<&Expression>,
        join_type: &str,
        using_columns: &[Identifier],
        where_clause: Option<&Expression>,
        index_use: JoinIndexUse,
        ctes: &[String],
        sides: Option<&JoinSides>,
    ) -> (String, Option<IndexLookupStrategy>) {
        // The index join is reported only where execution takes it: never
        // for a shape the reduction plan takes without the grouped operator,
        // and for the grouped operator on the right side only
        let (bounded, rows) = match index_use {
            JoinIndexUse::None => (None, [false; 2]),
            JoinIndexUse::Streaming { bounded, rows } => (Some(bounded), rows),
            JoinIndexUse::Grouped => (Some(true), [false; 2]),
        };
        // Check for INLJ opportunity on right side (default check)
        if let (Some(cond), Some(bounded)) = (join_condition, bounded) {
            // Get aliases for the check
            let left_alias = extract_table_alias(join_left);
            let right_alias = extract_table_alias(join_right);
            let join_type_upper = join_type.to_uppercase();
            // An INNER join without a limit narrows the inner side by a
            // table's candidates only when its outer side turns out small,
            // which EXPLAIN cannot run
            let small_outer = !bounded && join_type_upper == "INNER";
            let name = |strategy: &IndexLookupStrategy| match strategy {
                IndexLookupStrategy::PrimaryKey => "Index Nested Loop (PK)".to_string(),
                IndexLookupStrategy::TableEquality { .. } if small_outer => {
                    "Standard Join (conditional: a small outer narrows the inner side by its keys)"
                        .to_string()
                }
                IndexLookupStrategy::SecondaryIndex(_)
                | IndexLookupStrategy::TableEquality { .. } => "Index Nested Loop".to_string(),
            };

            let left_schema = self.base_table_side_schema(join_left, CteNames::Listed(ctes));
            let right_schema = self.base_table_side_schema(join_right, CteNames::Listed(ctes));
            let left_is_table = left_schema.is_some();
            let right_is_table = right_schema.is_some();
            let [right_probe, left_probe] = self.join_probes(
                CteNames::Listed(ctes),
                [join_left, join_right],
                Some(cond),
                &join_type_upper,
                [left_alias.as_deref(), right_alias.as_deref()],
                rows,
            );
            let inlj_info = right_probe
                .as_ref()
                .and_then(|probe| self.probe_strategy(probe, bounded, small_outer, !left_is_table));

            // The side execution probes, by the rule it picks it with. The
            // executor swaps a small outer in only from a right side that is
            // not a base table
            let picked = if index_use == JoinIndexUse::Grouped {
                inlj_info.map(|info| (info, false))
            } else {
                let (left_filtered, right_filtered) = match where_clause {
                    Some(where_clause) => {
                        let aliases = [join_left, join_right].map(get_table_alias_from_expr);
                        let (left, right, _) = split_join_where(
                            where_clause,
                            SplitSide::both(
                                sides,
                                [aliases[0].as_deref(), aliases[1].as_deref()],
                                [left_schema.as_deref(), right_schema.as_deref()],
                            ),
                            &join_type_upper,
                        );
                        (left.is_some(), right.is_some())
                    }
                    None => (false, false),
                };
                pick_index_join_side(
                    &join_type_upper,
                    right_is_table,
                    right_filtered,
                    left_filtered,
                    inlj_info,
                    || {
                        left_probe.as_ref().and_then(|probe| {
                            self.probe_strategy(probe, bounded, small_outer, !right_is_table)
                        })
                    },
                    |(_, strategy, _, _)| matches!(strategy, IndexLookupStrategy::PrimaryKey),
                )
            };

            if let Some(((_, strategy, _, _), _)) = picked {
                return (name(&strategy), Some(strategy));
            }
        }

        // The rule a run picks the method by, over the tables' statistics
        let planner = self.get_query_planner();
        let left_table_name = extract_table_name(join_left);
        let right_table_name = extract_table_name(join_right);

        // Get row counts from statistics (or use defaults)
        let left_rows = left_table_name
            .as_ref()
            .and_then(|name| planner.get_table_stats(name))
            .map(|s| s.row_count as usize)
            .unwrap_or(1000);
        let right_rows = right_table_name
            .as_ref()
            .and_then(|name| planner.get_table_stats(name))
            .map(|s| s.row_count as usize)
            .unwrap_or(1000);

        // Determine if we have equality keys
        let has_equality_keys = if let Some(cond) = join_condition {
            is_equality_condition(cond)
        } else {
            !using_columns.is_empty()
        };

        let algo_name = match pair_method(
            left_rows,
            right_rows,
            has_equality_keys,
            &join_type.to_uppercase(),
        ) {
            PairMethod::Hash { build_left: true } => "Hash Join (build: left)".to_string(),
            PairMethod::Hash { build_left: false } => "Hash Join (build: right)".to_string(),
            PairMethod::NestedLoop => "Nested Loop".to_string(),
        };

        (algo_name, None)
    }

    /// Detect if a SELECT statement would use the vector search fast path.
    /// Returns the appropriate ScanPlan (VectorSearch or VectorBruteForce) if detected.
    fn detect_vector_search_plan(
        &self,
        select: &SelectStatement,
        table_name: &str,
    ) -> Option<ScanPlan> {
        // Same conditions as the execution fast path in query.rs
        if select.limit.is_none()
            || select.order_by.len() != 1
            || !select.group_by.columns.is_empty()
            || select.having.is_some()
            || select.distinct
        {
            return None;
        }

        let order_by = &select.order_by[0];
        if !order_by.ascending {
            return None;
        }

        // Find VEC_DISTANCE function: direct call, alias, or <=> operator
        let (fn_name_upper, vec_col_name) = match &order_by.expression {
            Expression::FunctionCall(fc) => {
                let name = fc.function.to_uppercase();
                if !matches!(
                    name.as_str(),
                    "VEC_DISTANCE_L2" | "VEC_DISTANCE_COSINE" | "VEC_DISTANCE_IP"
                ) {
                    return None;
                }
                if fc.arguments.len() != 2 {
                    return None;
                }
                let col = Self::extract_vec_col_name(&fc.arguments[0])?;
                (name.to_string(), col)
            }
            Expression::Identifier(id) => {
                // Alias lookup in SELECT columns
                if let Some(fc) =
                    Self::find_vec_distance_alias_for_explain(&id.value_lower, &select.columns)
                {
                    let name = fc.function.to_uppercase();
                    let col = Self::extract_vec_col_name(&fc.arguments[0])?;
                    (name.to_string(), col)
                } else {
                    let infix = Self::find_vec_distance_infix_alias_for_explain(
                        &id.value_lower,
                        &select.columns,
                    )?;
                    // <=> operator is equivalent to VEC_DISTANCE_L2
                    let col = Self::extract_vec_col_name(&infix.left)?;
                    ("VEC_DISTANCE_L2".to_string(), col)
                }
            }
            Expression::Infix(infix) if infix.op_type == InfixOperator::VectorDistance => {
                let col = Self::extract_vec_col_name(&infix.left)?;
                ("VEC_DISTANCE_L2".to_string(), col)
            }
            _ => return None,
        };

        // Map function name to metric string and numeric code
        let (metric, expected_metric): (&str, u8) = match fn_name_upper.as_str() {
            "VEC_DISTANCE_L2" => ("L2", 0),
            "VEC_DISTANCE_COSINE" => ("Cosine", 1),
            "VEC_DISTANCE_IP" => ("InnerProduct", 2),
            _ => return None,
        };

        // Extract k from LIMIT
        let k = Self::extract_limit_value(select)?;

        // Check for WHERE clause filter text
        let filter = select.where_clause.as_ref().map(|w| format!("{}", w));

        if let Ok(tx) = self.begin_transaction() {
            if let Ok(table) = tx.get_read_table_at(table_name, None, TableUse::Read) {
                let hnsw_index = table.get_index_on_column(&vec_col_name).filter(|idx| {
                    idx.index_type() == crate::core::IndexType::Hnsw
                        && idx.hnsw_distance_metric() == Some(expected_metric)
                });

                if let Some(idx) = hnsw_index {
                    let ef_search = idx.default_ef_search().unwrap_or(64);
                    return Some(ScanPlan::VectorSearch {
                        table: table_name.to_string(),
                        index_name: idx.name().to_string(),
                        vector_column: vec_col_name,
                        metric: metric.to_string(),
                        k,
                        ef_search,
                        filter,
                    });
                }

                return Some(ScanPlan::VectorBruteForce {
                    table: table_name.to_string(),
                    vector_column: vec_col_name,
                    metric: metric.to_string(),
                    k,
                    filter,
                });
            }
        }

        None
    }

    /// Extract vector column name from a function argument expression
    fn extract_vec_col_name(expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier(id) => Some(id.value.to_string()),
            Expression::QualifiedIdentifier(qid) => Some(qid.name.value.to_string()),
            _ => None,
        }
    }

    /// Find a VEC_DISTANCE function call aliased in SELECT columns (for EXPLAIN)
    fn find_vec_distance_alias_for_explain<'a>(
        alias_lower: &str,
        columns: &'a [Expression],
    ) -> Option<&'a FunctionCall> {
        for col_expr in columns {
            if let Expression::Aliased(aliased) = col_expr {
                if aliased.alias.value_lower == alias_lower {
                    if let Expression::FunctionCall(fc) = &*aliased.expression {
                        let fn_upper = fc.function.to_uppercase();
                        if matches!(
                            fn_upper.as_str(),
                            "VEC_DISTANCE_L2" | "VEC_DISTANCE_COSINE" | "VEC_DISTANCE_IP"
                        ) && fc.arguments.len() == 2
                        {
                            return Some(fc.as_ref());
                        }
                    }
                }
            }
        }
        None
    }

    /// Find a <=> infix operator aliased in SELECT columns (for EXPLAIN)
    fn find_vec_distance_infix_alias_for_explain<'a>(
        alias_lower: &str,
        columns: &'a [Expression],
    ) -> Option<&'a InfixExpression> {
        for col_expr in columns {
            if let Expression::Aliased(aliased) = col_expr {
                if aliased.alias.value_lower == alias_lower {
                    if let Expression::Infix(infix) = &*aliased.expression {
                        if infix.op_type == InfixOperator::VectorDistance {
                            return Some(infix);
                        }
                    }
                }
            }
        }
        None
    }

    /// LIMIT + OFFSET of a SELECT, when both are literals: EXPLAIN runs nothing
    fn extract_limit_value(select: &SelectStatement) -> Option<usize> {
        let counts = super::row_count::static_row_counts(select)?;
        Some(counts.limit?.saturating_add(counts.offset))
    }
}

/// Extract table alias from a table expression
/// The names the WITH of `select` gives its CTEs, which hide tables of the
/// same name in its FROM
pub(crate) fn cte_names(select: &SelectStatement) -> Vec<String> {
    select
        .with
        .iter()
        .flat_map(|with| with.ctes.iter().map(|cte| cte.name.value_lower.to_string()))
        .collect()
}

fn extract_table_alias(expr: &Expression) -> Option<String> {
    match expr {
        Expression::TableSource(simple) => simple
            .alias
            .as_ref()
            .map(|a| a.value.to_string())
            .or_else(|| Some(simple.name.value.to_string())),
        Expression::Aliased(aliased) => Some(aliased.alias.value.to_string()),
        _ => None,
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
        Expression::TableSource(simple) => Some(simple.name.value.to_string()),
        Expression::JoinSource(join) => extract_table_name(&join.left),
        Expression::SubquerySource(_) => None, // Can't get stats for subquery
        _ => None,
    }
}
