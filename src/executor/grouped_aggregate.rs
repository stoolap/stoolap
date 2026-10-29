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

//! A statement that groups one table by one column without WHERE: the one
//! rule that says where its groups come from, and the run of that source

use crate::core::{Result, Row, RowVec, Value};
use crate::parser::ast::{Expression, GroupByModifier, SelectStatement};
use crate::storage::index::id_list::GroupIds;
use crate::storage::traits::{QueryResult, Table};

use super::aggregation::{AggregatePath, AggregateSlots, Modifiers, SqlAggregateFunction};
use super::context::ExecutionContext;
use super::expression::ExprVM;
use super::query_classification::QueryClassification;
use super::row_count::rows_needed;
use super::Executor;

/// Where a statement that groups one table by one column, without WHERE,
/// reads its groups from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupedSource {
    /// Every group computed by the table over its rows; a run the table
    /// declines walks the column's index when it may
    Storage,
    /// The groups in the order of the column's index, each read by its row
    /// ids, stopping once the LIMIT's groups passed HAVING; a table whose
    /// index keeps no order, or cannot answer for the run, declines it
    SortedStream,
    /// The rows read and grouped by their key
    Hash,
}

impl Executor {
    /// Where `stmt`'s groups come from, a statement grouping one table by one
    /// column without WHERE: the rule every run decides it by. `plain` when
    /// its SELECT list reads the group column and `aggregations` only
    pub(crate) fn grouped_source(
        stmt: &SelectStatement,
        classification: &QueryClassification,
        aggregations: &[SqlAggregateFunction],
        plain: bool,
    ) -> GroupedSource {
        if !plain
            || aggregations.is_empty()
            || Self::aggregates_hold_subquery(stmt)
            || !aggregations
                .iter()
                .all(|agg| agg.answered_with(Modifiers::NONE))
        {
            return GroupedSource::Hash;
        }
        // The table computes counts, sums, averages and bounds of columns
        let stored = stmt.having.is_none()
            && aggregations
                .iter()
                .all(|agg| matches!(agg.name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"));
        if stored {
            return GroupedSource::Storage;
        }
        match Self::sorts_groups(stmt, classification, aggregations) {
            true => GroupedSource::SortedStream,
            false => GroupedSource::Hash,
        }
    }

    /// Whether the groups may be read in the order of the column's index:
    /// with no ORDER BY to sort them again, a HAVING without subqueries, and
    /// COUNT(*), or under a LIMIT the stop pays for reading each group's
    /// rows, SUM, AVG, MIN and MAX of a column
    fn sorts_groups(
        stmt: &SelectStatement,
        classification: &QueryClassification,
        aggregations: &[SqlAggregateFunction],
    ) -> bool {
        let counts = aggregations
            .iter()
            .all(|agg| agg.name == "COUNT" && agg.column == "*");
        !classification.has_order_by
            && stmt
                .having
                .as_deref()
                .is_none_or(|h| !Self::has_subqueries(h))
            && aggregations.iter().all(|agg| match agg.name.as_str() {
                "COUNT" => agg.column == "*",
                "SUM" | "AVG" | "MIN" | "MAX" => true,
                _ => false,
            })
            && (counts || stmt.limit.is_some())
    }

    /// `stmt`'s groups over `table` when it groups the table by one column
    /// without WHERE, from the source the rule picks; None for the hash
    /// grouping, which the scan does. Inlined into the table scan, it costs
    /// that scan's loop more per row
    #[inline(never)]
    pub(crate) fn grouped_aggregate(
        &self,
        table: &dyn Table,
        stmt: &SelectStatement,
        classification: &QueryClassification,
        ctx: &ExecutionContext,
        all_columns: &[String],
    ) -> Result<Option<Box<dyn QueryResult>>> {
        if !classification.has_group_by
            || classification.has_where
            || classification.has_window_functions
            || stmt.group_by.columns.len() != 1
            || !matches!(stmt.group_by.modifier, GroupByModifier::None)
        {
            return Ok(None);
        }
        let Some((group, false)) = self.single_group_column(stmt)? else {
            return Ok(None);
        };
        let Some(group_index) = all_columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(&group))
        else {
            return Ok(None);
        };
        let (aggregations, non_aggregated) = self.parse_aggregations(stmt, Some(ctx))?;
        // The SELECT list reads the group column and aggregate calls only
        let group_item = |expr: &Expression| {
            let inner = match expr {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                other => other,
            };
            match inner {
                Expression::Identifier(id) => id.value.eq_ignore_ascii_case(&group),
                Expression::FunctionCall(call) => {
                    super::utils::is_aggregate_function(&call.function)
                }
                _ => false,
            }
        };
        let plain = stmt.columns.iter().all(group_item)
            && non_aggregated
                .iter()
                .all(|column| column.eq_ignore_ascii_case(&group));
        let source = Self::grouped_source(stmt, classification, &aggregations, plain);
        if source == GroupedSource::Hash {
            return Ok(None);
        }
        let columns = Self::call_columns(&aggregations, all_columns);
        let stored = match source {
            GroupedSource::Storage => {
                Self::stored_groups(table, group_index, &aggregations, &columns)?
            }
            GroupedSource::SortedStream | GroupedSource::Hash => None,
        };
        // A table that declines its own grouping is walked by its index when
        // the walk may read the groups
        let sorted = || {
            source == GroupedSource::SortedStream
                || Self::sorts_groups(stmt, classification, &aggregations)
        };
        let rows = match stored {
            Some(rows) => (AggregatePath::StorageGrouped, rows),
            None if sorted() => {
                let group = (group.as_str(), group_index);
                match self.sorted_groups(
                    table,
                    stmt,
                    group,
                    &aggregations,
                    &columns,
                    all_columns,
                    ctx,
                )? {
                    Some(rows) => (AggregatePath::StreamingGroupBy, rows),
                    None => return Ok(None),
                }
            }
            None => return Ok(None),
        };
        let (path, rows) = rows;
        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(all_columns[group_index].clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 1,
            source_columns: all_columns,
            blocks: None,
            alone: None,
        };
        self.finish_aggregation(path, slots, ctx, slot_columns, rows)
            .map(Some)
    }

    /// The groups the table computes over its rows, each call reading its
    /// column of `columns`; None when the table declines
    fn stored_groups(
        table: &dyn Table,
        group_index: usize,
        aggregations: &[SqlAggregateFunction],
        columns: &[Option<usize>],
    ) -> Result<Option<RowVec>> {
        use crate::storage::mvcc::version_store::AggregateOp;
        let mut ops = Vec::with_capacity(aggregations.len());
        for (agg, &column) in aggregations.iter().zip(columns) {
            let op = match (agg.name.as_str(), agg.column.as_str(), column) {
                ("COUNT", "*", _) => (AggregateOp::CountStar, 0),
                ("COUNT", _, Some(idx)) => (AggregateOp::Count, idx),
                ("SUM", _, Some(idx)) => (AggregateOp::Sum, idx),
                ("AVG", _, Some(idx)) => (AggregateOp::Avg, idx),
                ("MIN", _, Some(idx)) => (AggregateOp::Min, idx),
                ("MAX", _, Some(idx)) => (AggregateOp::Max, idx),
                _ => return Ok(None),
            };
            ops.push(op);
        }
        match table.compute_grouped_aggregates(&[group_index], &ops)? {
            Some(groups) => groups.into_rows().map(Some),
            None => Ok(None),
        }
    }

    /// The groups in the order of `group`'s index, the column named and its
    /// place, each read by its row ids and passed through HAVING, up to the
    /// ones LIMIT and OFFSET need; None when the table's index cannot answer
    /// for the statement's rows
    #[allow(clippy::too_many_arguments)]
    fn sorted_groups(
        &self,
        table: &dyn Table,
        stmt: &SelectStatement,
        (group, group_index): (&str, usize),
        aggregations: &[SqlAggregateFunction],
        columns: &[Option<usize>],
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Option<RowVec>> {
        // Rows a table that holds volumes may capture under the index lock,
        // and the bytes its keys and ids may take. Either bound stops the
        // capture, and a capture that stopped before the end of the index
        // answers only when the caller stopped on its own count.
        const CAPTURE_ROWS: usize = 4096;
        const CAPTURE_BYTES: usize = 1024 * 1024;

        // COUNT(*) reads no column; the others read theirs
        let agg_columns = columns;
        if aggregations
            .iter()
            .zip(agg_columns)
            .any(|(agg, column)| agg.column != "*" && column.is_none())
        {
            return Ok(None);
        }
        let needs_row_fetch = agg_columns.iter().any(Option::is_some);
        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(all_columns[group_index].clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));

        // HAVING over the slot row, compiled as every aggregate path compiles
        // it; a memory table walks under its index lock, so a HAVING that
        // calls a scalar function, which may sleep or write, runs elsewhere
        let having_filter = match stmt.having.as_deref() {
            Some(having) => {
                let slots = super::aggregation::aggregate_slot_keys(
                    &self.parse_group_by(stmt, all_columns)?,
                    aggregations,
                    &stmt.columns,
                );
                let filter = super::aggregation::compile_over_slots(
                    having,
                    &slot_columns,
                    crate::functions::registry::global_registry(),
                    Some(&slots),
                    None,
                    false,
                    Some(ctx.params()),
                )?;
                use super::expression::Op;
                if filter
                    .ops()
                    .iter()
                    .any(|op| matches!(op, Op::CallScalar { .. } | Op::NativeFn1(_)))
                {
                    return Ok(None);
                }
                Some(filter)
            }
            None => None,
        };
        let mut having_vm = ExprVM::new();
        let mut result_rows = RowVec::new();
        let mut result_row_id = 0i64;

        // With streaming aggregation we can stop once LIMIT + OFFSET groups pass
        // HAVING; OFFSET is applied by the caller and must find its groups here
        let limit_for_early_exit = rows_needed(stmt, ctx);

        // Pre-allocate reusable buffer for row fetching (avoids alloc/dealloc per group)
        let mut row_buffer = RowVec::with_capacity(256);
        use crate::storage::expression::logical::ConstBoolExpr;
        let true_expr = ConstBoolExpr::true_expr();

        // One run of each call by the implementation every aggregate path
        // runs, reset for each group
        let mut runs = aggregations
            .iter()
            .map(|agg| Ok(agg.template(&self.function_registry)?.instance()))
            .collect::<Result<Vec<_>>>()?;
        let count_star = Value::Integer(1);

        let mut aggregate = |group_value: &Value, row_ids: GroupIds<'_>| -> Result<bool> {
            for run in runs.iter_mut() {
                run.reset();
            }

            if needs_row_fetch {
                // A page of ids at a time through the one row buffer; the
                // aggregate state carries across the group's pages
                for page in row_ids.pages() {
                    row_buffer.clear();
                    table.fetch_rows_by_ids_into(page, &true_expr, &mut row_buffer)?;
                    for (_row_id, row) in &row_buffer {
                        for (run, column) in runs.iter_mut().zip(agg_columns) {
                            match column {
                                Some(idx) => {
                                    if let Some(value) = row.get(*idx) {
                                        run.accumulate(value);
                                    }
                                }
                                None => run.accumulate(&count_star),
                            }
                        }
                    }
                }
            } else {
                // COUNT(*) alone: the group's row count, no row read
                let rows = crate::functions::aggregate::CompiledAggregate::CountStar {
                    count: row_ids.len() as i64,
                };
                for run in runs.iter_mut() {
                    run.merge(&rows);
                }
            }

            // The slot row, then HAVING over it
            let mut values = Vec::with_capacity(1 + runs.len());
            values.push(group_value.clone());
            for run in &runs {
                values.push(run.result()?);
            }
            let row = Row::from_values(values);
            if let Some(filter) = &having_filter {
                if !having_vm
                    .execute_bool_checked(filter, &super::aggregation::slot_context(&row, ctx))?
                {
                    return Ok(true); // Continue to next group
                }
            }
            result_rows.push((result_row_id, row));
            result_row_id += 1;

            // Early termination: stop once we have LIMIT groups that passed HAVING
            if let Some(limit) = limit_for_early_exit {
                if result_rows.len() >= limit {
                    return Ok(false); // Stop iteration
                }
            }

            Ok(true) // Continue to next group
        };

        // The table owns whether its index can answer for this statement's
        // rows, and for how long it holds the index lock: a memory table
        // walks the index under that lock, a table holding volumes captures
        // a prefix of it first and releases the lock before the fetch above
        let walked = table.walk_btree_groups(group, CAPTURE_ROWS, CAPTURE_BYTES, &mut aggregate)?;
        if walked.is_none() {
            return Ok(None);
        }
        if let Some(keep) = limit_for_early_exit {
            result_rows.truncate(keep);
        }
        Ok(Some(result_rows))
    }
}
