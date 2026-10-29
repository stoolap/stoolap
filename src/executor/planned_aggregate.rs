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

//! A statement that aggregates one table into one row: the one rule that
//! says where its values come from, and the run of that source

use std::sync::{Arc, RwLock};

use smallvec::SmallVec;

use crate::common::{CompactArc, CompactVec, SmartString};
use crate::core::{IndexType, Result, Row, RowVec, Schema, Value};
use crate::parser::ast::{Expression, GroupByModifier, SelectStatement};
use crate::planner::{steps_over, ScanChoice, Step};
use crate::storage::expression::Expression as StorageExpr;
use crate::storage::mvcc::index_plan;
use crate::storage::traits::transaction::{StatementTable, TableUse};
use crate::storage::traits::{Engine, Index, QueryResult, Table};

use super::aggregation::{AggregatePath, AggregateSlots, Modifiers, SqlAggregateFunction};
use super::context::ExecutionContext;
use super::expression::SubqueryExecutor;
use super::grouped_aggregate::GroupedSource;
use super::query_cache::{CompiledExecution, PlanValidity};
use super::query_classification::{get_classification, QueryClassification};
use super::Executor;

/// Where a statement that aggregates one table without grouping reads its
/// one row from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalSource {
    /// Every call from what the table keeps of its rows: the row count, an
    /// index's distinct count, a column's sum and bounds
    Stats,
    /// The one MIN or MAX of a column with an ordered index, from its first
    /// or last key; a run the index cannot answer takes the stats
    IndexBound,
    /// Every call computed by the table over the rows its WHERE passes
    StorageFiltered,
    /// The rows read and fed to the calls
    Scan,
}

/// The column a call reads, without the table naming it
fn bare_column(agg: &SqlAggregateFunction) -> &str {
    super::utils::bare_name(&agg.column_lower)
}

/// Whether `index` keeps its values in order, which gives their bounds
pub(super) fn bounds(index: Option<Arc<dyn Index>>) -> bool {
    index
        .is_some_and(|index| matches!(index.index_type(), IndexType::BTree | IndexType::PrimaryKey))
}

impl GlobalSource {
    /// The path that answered from this source
    fn path(self) -> AggregatePath {
        match self {
            Self::Stats => AggregatePath::Stats,
            Self::IndexBound => AggregatePath::IndexBound,
            Self::StorageFiltered => AggregatePath::StorageFiltered,
            Self::Scan => AggregatePath::GlobalScan,
        }
    }
}

/// The calls of a statement that aggregates one table and where its row or
/// groups come from, as the rules decide them
pub(crate) struct AggregatePlan {
    pub(crate) aggregations: Vec<SqlAggregateFunction>,
    /// Whether the SELECT list reads nothing but the calls
    pub(super) plain: bool,
    pub(crate) read: AggregateRead,
}

/// Where an aggregate over one table reads from
pub(crate) enum AggregateRead {
    /// Its one row from `source`, `pushed` the WHERE the table runs whole
    Global {
        source: GlobalSource,
        pushed: Option<Box<dyn StorageExpr>>,
    },
    /// Its groups by one column, named and placed as `group`, from `source`;
    /// `walks` whether the index walk reads them when the table declines
    Grouped {
        source: GroupedSource,
        group: (String, usize),
        walks: bool,
    },
}

/// A statement without WHERE whose one row is one count the table keeps or
/// an index's bound, planned once and run before the statement is classified
#[derive(Debug)]
pub struct PlannedAggregate {
    source: GlobalSource,
    aggregations: Vec<SqlAggregateFunction>,
    table: SmartString,
    /// The table's columns, which the SELECT list reads through the calls
    source_columns: CompactArc<Vec<String>>,
    /// The output names of a SELECT list of nothing but the calls
    alone: Option<CompactArc<Vec<String>>>,
    pub(crate) validity: PlanValidity,
}

impl Executor {
    /// The plan of `stmt`, which aggregates one table of `schema` whose
    /// ordered indexes `indexed` names: its calls and the source the rules
    /// pick; None for a statement the aggregation of its read rows answers.
    /// Inlined into the table scan, it costs that scan's loop more per row
    #[inline(never)]
    pub(crate) fn aggregate_read_plan(
        &self,
        stmt: &SelectStatement,
        classification: &QueryClassification,
        ctx: &ExecutionContext,
        schema: &Schema,
        indexed: impl Fn(&str) -> bool,
    ) -> Result<Option<AggregatePlan>> {
        if classification.has_window_functions {
            return Ok(None);
        }
        if classification.has_group_by {
            return self.grouped_read_plan(stmt, classification, ctx, schema);
        }
        if classification.has_having {
            return Ok(None);
        }
        // The table runs no WHERE over a parameter or a subquery: its rows are
        // the scan's to read
        if classification.has_where
            && (classification.where_has_parameters || classification.where_has_subqueries)
        {
            return Ok(None);
        }
        let (aggregations, non_aggregated) = self.parse_aggregations(stmt, Some(ctx))?;
        let pushed = match stmt.where_clause.as_deref() {
            Some(where_clause) => {
                match super::pushdown::try_pushdown(where_clause, schema, Some(ctx)) {
                    (Some(pushed), false) => Some(pushed),
                    _ => None,
                }
            }
            None => None,
        };
        let plain = non_aggregated.is_empty();
        let source = Self::global_source(stmt, &aggregations, plain, pushed.is_some(), indexed);
        Ok(Some(AggregatePlan {
            aggregations,
            plain,
            read: AggregateRead::Global { source, pushed },
        }))
    }

    /// Where `stmt`, which aggregates one table into one row, reads its row
    /// from: the rule every caller decides it by. `plain` when its SELECT
    /// list reads nothing but `aggregations`, `indexed` whether a column's
    /// index keeps order, `pushed` whether the table runs its WHERE whole
    pub(crate) fn global_source(
        stmt: &SelectStatement,
        aggregations: &[SqlAggregateFunction],
        plain: bool,
        pushed: bool,
        indexed: impl Fn(&str) -> bool,
    ) -> GlobalSource {
        if !plain || aggregations.is_empty() || Self::aggregates_hold_subquery(stmt) {
            return GlobalSource::Scan;
        }
        let plain_call = |agg: &SqlAggregateFunction| {
            matches!(agg.name.as_str(), "SUM" | "MIN" | "MAX" | "AVG")
                && agg.answered_with(Modifiers::NONE)
        };
        if stmt.where_clause.is_none() {
            // The stats count rows and an index's distinct values, not a column's
            let stats = aggregations.iter().all(|agg| {
                plain_call(agg)
                    || agg.counts_rows()
                    || agg.name == "COUNT" && agg.distinct && agg.answered_with(Modifiers::DISTINCT)
            });
            if !stats {
                return GlobalSource::Scan;
            }
            return match aggregations {
                [agg]
                    if matches!(agg.name.as_str(), "MIN" | "MAX")
                        && agg.expression.is_none()
                        && indexed(bare_column(agg)) =>
                {
                    GlobalSource::IndexBound
                }
                _ => GlobalSource::Stats,
            };
        }
        // The table computes the calls over the rows its WHERE passes: a
        // column's count among them
        let computed = aggregations.iter().all(|agg| {
            plain_call(agg)
                || agg.counts_rows()
                || agg.name == "COUNT" && agg.answered_with(Modifiers::NONE)
        });
        match pushed && computed {
            true => GlobalSource::StorageFiltered,
            false => GlobalSource::Scan,
        }
    }

    /// Whether every row of the table feeds the calls one by one, faster
    /// than the scan that collects them: plain counts, sums, averages and
    /// bounds, under no ORDER BY or LIMIT
    fn streams_calls(
        classification: &QueryClassification,
        aggregations: &[SqlAggregateFunction],
        plain: bool,
    ) -> bool {
        plain
            && !classification.has_order_by
            && !classification.has_limit
            && aggregations.iter().all(|agg| match agg.name.as_str() {
                "COUNT" => agg.counts_rows() || agg.answered_with(Modifiers::DISTINCT),
                "SUM" | "MIN" | "MAX" | "AVG" => agg.answered_with(Modifiers::NONE),
                _ => false,
            })
    }

    /// The one row of `aggregations` over `table` from `source`, each call
    /// reading its column of `columns`; None when the table cannot answer
    /// from there and the rows are read instead. A `quiet` read, through no
    /// view, takes only a count that vouches for itself; `pushed` is the
    /// WHERE the table runs
    fn global_values(
        source: GlobalSource,
        table: &dyn Table,
        aggregations: &[SqlAggregateFunction],
        columns: &[Option<usize>],
        quiet: bool,
        pushed: Option<&dyn StorageExpr>,
    ) -> Result<Option<(GlobalSource, CompactVec<Value>)>> {
        match source {
            GlobalSource::IndexBound => {
                if let Some(value) = Self::index_bound(table, &aggregations[0])? {
                    let mut values = CompactVec::with_capacity(1);
                    values.push(value);
                    return Ok(Some((source, values)));
                }
                // A column's bound read through no view does not vouch for itself
                if quiet {
                    return Ok(None);
                }
                Ok(Self::stats_values(table, aggregations, columns, quiet)?
                    .map(|values| (GlobalSource::Stats, values)))
            }
            GlobalSource::Stats => Ok(Self::stats_values(table, aggregations, columns, quiet)?
                .map(|values| (source, values))),
            GlobalSource::StorageFiltered => match pushed {
                Some(pushed) => Ok(Self::filtered_values(table, pushed, aggregations, columns)?
                    .map(|values| (source, values))),
                None => Ok(None),
            },
            GlobalSource::Scan => Ok(None),
        }
    }

    /// MIN or MAX of the call's column from its index; None when the table
    /// has none, or holds changes or a view the index does not see
    fn index_bound(table: &dyn Table, agg: &SqlAggregateFunction) -> Result<Option<Value>> {
        // The index knows nothing of this transaction's own writes, and holds
        // what the statement sees only while the table vouches for it
        if table.has_local_changes() {
            return Ok(None);
        }
        let Some(epoch) = table.index_view_epoch() else {
            return Ok(None);
        };
        let value = match agg.name.as_str() {
            "MIN" => table.get_index_min_value(bare_column(agg)),
            _ => table.get_index_max_value(bare_column(agg)),
        };
        if table.index_view_epoch() != Some(epoch) {
            return Ok(None);
        }
        #[cfg(feature = "test-failpoints")]
        if value.is_some() {
            crate::test_failpoints::min_max_index_answered();
        }
        Ok(value)
    }

    /// Every call answered from what the table keeps without reading its
    /// rows; None when one call cannot be
    fn stats_values(
        table: &dyn Table,
        aggregations: &[SqlAggregateFunction],
        columns: &[Option<usize>],
        quiet: bool,
    ) -> Result<Option<CompactVec<Value>>> {
        let mut values = CompactVec::with_capacity(aggregations.len());
        for (agg, &column) in aggregations.iter().zip(columns) {
            let null = || Value::null(crate::core::DataType::Integer);
            let value = match (agg.name.as_str(), column) {
                ("COUNT", _) if agg.distinct => table
                    .get_partition_count(bare_column(agg))?
                    .map(|count| Value::Integer(count as i64)),
                // Through no view only the count that vouches for itself; through
                // one, the count of what the view sees
                ("COUNT", _) if agg.counts_rows() => match quiet {
                    true => table.fast_row_count()?,
                    false => Some(table.row_count()?),
                }
                .map(|count| Value::Integer(count as i64)),
                ("SUM", Some(idx)) => table.sum_column(idx)?.map(|sum| sum.sum()).transpose()?,
                ("AVG", Some(idx)) => table.avg_column(idx)?.map(|sum| sum.avg()),
                ("MIN", Some(idx)) => table.min_column(idx)?.map(|min| min.unwrap_or_else(null)),
                ("MAX", Some(idx)) => table.max_column(idx)?.map(|max| max.unwrap_or_else(null)),
                _ => None,
            };
            match value {
                Some(value) => values.push(value),
                None => return Ok(None),
            }
        }
        Ok(Some(values))
    }

    /// Every call computed by the table over the rows the WHERE passes;
    /// None when the table cannot run the WHERE or the calls
    fn filtered_values(
        table: &dyn Table,
        pushed: &dyn StorageExpr,
        aggregations: &[SqlAggregateFunction],
        columns: &[Option<usize>],
    ) -> Result<Option<CompactVec<Value>>> {
        use crate::storage::mvcc::version_store::AggregateOp;
        let mut ops = Vec::with_capacity(aggregations.len());
        for (agg, &column) in aggregations.iter().zip(columns) {
            let op = match (agg.name.as_str(), column) {
                ("COUNT", _) if agg.counts_rows() => (AggregateOp::CountStar, 0),
                ("COUNT", Some(idx)) => (AggregateOp::Count, idx),
                ("SUM", Some(idx)) => (AggregateOp::Sum, idx),
                ("AVG", Some(idx)) => (AggregateOp::Avg, idx),
                ("MIN", Some(idx)) => (AggregateOp::Min, idx),
                ("MAX", Some(idx)) => (AggregateOp::Max, idx),
                _ => return Ok(None),
            };
            ops.push(op);
        }
        Ok(table
            .compute_filtered_aggregates(&ops, pushed)?
            .map(|values| values.into_iter().collect()))
    }

    /// Each call's column in `columns`, the one table's, whatever qualifier
    /// names it
    pub(super) fn call_columns(
        aggregations: &[SqlAggregateFunction],
        columns: &[String],
    ) -> Vec<Option<usize>> {
        aggregations
            .iter()
            .map(|agg| {
                let column = bare_column(agg);
                let named = || columns.iter().position(|c| c.eq_ignore_ascii_case(column));
                agg.expression.is_none().then(named).flatten()
            })
            .collect()
    }

    /// The result of the one row `rows` holds for `aggregations`, the end
    /// of every global aggregate path; `alone` the output names a plan found
    /// for a SELECT list of nothing but the calls
    #[allow(clippy::too_many_arguments)]
    fn finish_global(
        &self,
        path: AggregatePath,
        stmt: &SelectStatement,
        aggregations: &[SqlAggregateFunction],
        source_columns: &[String],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
        alone: Option<&CompactArc<Vec<String>>>,
        rows: RowVec,
    ) -> Result<Box<dyn QueryResult>> {
        // The slots' names serve a SELECT list that computes over them
        let names = match alone {
            Some(_) => Vec::new(),
            None => aggregations
                .iter()
                .map(|agg| agg.get_column_name())
                .collect(),
        };
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations,
            group_by_count: 0,
            source_columns,
            blocks,
            alone,
        };
        self.finish_aggregation(path, slots, ctx, names, rows)
    }

    /// `stmt`'s one row over `table` when it aggregates the table without
    /// grouping, from the source the rule picks; None when its rows are read
    /// past a WHERE, which the scan does. Inlined into the table scan, it
    /// costs that scan's loop more per row
    #[inline(never)]
    pub(crate) fn global_aggregate(
        &self,
        table: &dyn Table,
        stmt: &SelectStatement,
        (classification, plan): (&QueryClassification, &AggregatePlan),
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        let AggregateRead::Global { source, pushed } = &plan.read else {
            return Ok(None);
        };
        let aggregations = &plan.aggregations;
        let source_columns = table.schema().column_names_arc();
        let columns = Self::call_columns(aggregations, &source_columns);
        let row = Self::global_values(
            *source,
            table,
            aggregations,
            &columns,
            false,
            pushed.as_deref(),
        )?;
        let (path, rows) = match row {
            Some((answered, values)) => {
                let mut rows = RowVec::with_capacity(1);
                rows.push((0, Row::from_compact_vec(values)));
                (answered.path(), rows)
            }
            // A WHERE the table declined is the scan's to read, and so are
            // calls the feed does not stream
            None if stmt.where_clause.is_some() => return Ok(None),
            None if !Self::streams_calls(classification, aggregations, plan.plain) => {
                return Ok(None)
            }
            None => (
                AggregatePath::GlobalScan,
                self.global_scan_rows(table, aggregations, ctx, blocks)?,
            ),
        };
        self.finish_global(
            path,
            stmt,
            aggregations,
            &source_columns,
            ctx,
            blocks,
            None,
            rows,
        )
        .map(Some)
    }

    /// `stmt` run by the aggregate plan `compiled` holds: one count the table
    /// keeps or an index's bound, read through committed state. None for the
    /// regular path to answer, as it does the run that plans
    pub(crate) fn run_planned_aggregate(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
        ctx: &ExecutionContext,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        let epoch = self.engine.schema_epoch();
        let held = match &*compiled.read().ok()? {
            CompiledExecution::GlobalAggregate(p) if p.validity.holds(&*self.engine, epoch) => {
                Some(Arc::clone(p))
            }
            CompiledExecution::NotOptimizable(validity) if validity.holds(&*self.engine, epoch) => {
                return None
            }
            CompiledExecution::Unknown
            | CompiledExecution::NotOptimizable(_)
            | CompiledExecution::GlobalAggregate(_) => None,
            _ => return None,
        };
        // The run that plans is the regular path's, through its view
        let Some(planned) = held else {
            self.plan_aggregate(stmt, compiled, ctx, epoch);
            return None;
        };
        if let Err(error) = ctx.check_cancelled() {
            return Some(Err(error));
        }
        let run = || -> Result<Option<Box<dyn QueryResult>>> {
            // Under read committed only a count that vouches for itself, the
            // latest committed one; under snapshot isolation the counts of a
            // snapshot of its own
            let quiet = self.isolation_level() == crate::core::IsolationLevel::ReadCommitted;
            let tx = self.begin_transaction()?;
            let table = tx.get_read_table_at(&planned.table, None, TableUse::Read)?;
            let row = Self::global_values(
                planned.source,
                &*table,
                &planned.aggregations,
                // A count reads no column, a bound its index: no column place
                // the table's layout may move
                &[None],
                quiet,
                None,
            )?;
            let Some((answered, values)) = row else {
                return Ok(None);
            };
            let mut rows = RowVec::with_capacity(1);
            rows.push((0, Row::from_compact_vec(values)));
            self.finish_global(
                answered.path(),
                stmt,
                &planned.aggregations,
                &planned.source_columns,
                ctx,
                None,
                planned.alone.as_ref(),
                rows,
            )
            .map(Some)
        };
        run().transpose()
    }

    /// The aggregate plan of `stmt`, kept in `compiled`; a statement it does
    /// not take is kept as one the plan declines while its tables hold
    fn plan_aggregate(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
        ctx: &ExecutionContext,
        epoch: u64,
    ) {
        let validity = self.select_validity(stmt, epoch);
        let planned = self.aggregate_plan(stmt, ctx, validity.clone());
        let Ok(mut slot) = compiled.write() else {
            return;
        };
        if matches!(
            &*slot,
            CompiledExecution::Unknown
                | CompiledExecution::NotOptimizable(_)
                | CompiledExecution::GlobalAggregate(_)
        ) {
            *slot = match planned {
                Some(planned) => CompiledExecution::GlobalAggregate(planned),
                None => CompiledExecution::NotOptimizable(validity),
            };
        }
    }

    /// The plan of `stmt` as EXPLAIN shows it when it aggregates one table
    /// from the source a rule picks, with the reads a run the table declines
    /// takes; None for a statement the aggregation of its read rows answers,
    /// which the general format shows
    pub(crate) fn explain_aggregate(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<Vec<String>> {
        let shape = get_classification(stmt);
        if !shape.has_aggregation
            || shape.has_set_operations
            || shape.has_cte
            || shape.has_distinct_on
            || shape.has_joins
            || shape.output_has_subqueries
            || shape.where_has_subqueries
        {
            return None;
        }
        let Some(Expression::TableSource(ts)) = stmt.table_expr.as_deref() else {
            return None;
        };
        let table = &ts.name.value_lower;
        if ts.as_of.is_some() || ctx.get_cte_by_lower(table).is_some() {
            return None;
        }
        let store = self.engine.get_version_store(table).ok()?;
        let schema = store.schema_and_token().0;
        let indexed = |column: &str| bounds(store.get_index_by_column(column));
        let plan = self
            .aggregate_read_plan(stmt, &shape, ctx, &schema, indexed)
            .ok()??;
        let mut reads = Vec::new();
        let mut by = None;
        match &plan.read {
            AggregateRead::Global { source, pushed } => match source {
                GlobalSource::IndexBound => {
                    let column = bare_column(&plan.aggregations[0]);
                    reads.push(format!("IndexBound {table} ({column})"));
                    reads.push(format!("else Stats {table}"));
                    reads.push(format!("else SeqScan {table}"));
                }
                GlobalSource::Stats => {
                    reads.push(format!("Stats {table}"));
                    reads.push(format!("else SeqScan {table}"));
                }
                // A WHERE the table declines is read by the scan its index rule picks
                GlobalSource::StorageFiltered => {
                    let where_clause = stmt.where_clause.as_deref()?;
                    reads.push(format!("StorageFiltered {table} {where_clause}"));
                    let scan = pushed
                        .as_deref()
                        .and_then(|pushed| index_plan::decide(&store, &schema, pushed))
                        .map_or(ScanChoice::Seq, ScanChoice::Index);
                    let lines = scan.explain(table, Some(where_clause));
                    for (at, line) in lines.into_iter().enumerate() {
                        reads.push(if at == 0 {
                            format!("else {line}")
                        } else {
                            line
                        });
                    }
                }
                GlobalSource::Scan
                    if stmt.where_clause.is_none()
                        && Self::streams_calls(&shape, &plan.aggregations, plan.plain) =>
                {
                    reads.push(format!("SeqScan {table}"));
                }
                GlobalSource::Scan => return None,
            },
            AggregateRead::Grouped {
                source,
                group: (group, _),
                walks,
            } => {
                match source {
                    GroupedSource::Storage => {
                        reads.push(format!("StorageGrouped {table} ({group})"));
                        if *walks {
                            reads.push(format!("else SortedStream {table} ({group})"));
                        }
                    }
                    GroupedSource::SortedStream | GroupedSource::Hash => {
                        reads.push(format!("SortedStream {table} ({group})"));
                    }
                }
                reads.push(format!("else SeqScan {table}"));
                by = Some(group.clone());
            }
        }
        let items: Vec<String> = stmt.columns.iter().map(super::utils::item_name).collect();
        let head = match by {
            Some(group) => format!("Aggregate {} by {group}", items.join(", ")),
            None => format!("Aggregate {}", items.join(", ")),
        };
        let below = std::iter::once(head)
            .chain(reads.into_iter().map(|line| format!("  {line}")))
            .collect();
        // The DISTINCT over the aggregate's rows, and the HAVING over its groups
        let mut inner = SmallVec::<[Step; 2]>::new();
        if stmt.distinct {
            inner.push(Step::Distinct);
        }
        if stmt.having.is_some() {
            inner.push(Step::Having);
        }
        let steps = Self::finishing_steps(stmt, &inner);
        Some(steps_over(
            &steps,
            &Self::finishing_text(stmt, table),
            below,
        ))
    }

    /// The plan of a statement over one table, without WHERE, grouping,
    /// ORDER BY, LIMIT or OFFSET, whose one row is one count the table keeps
    /// or an index's bound
    fn aggregate_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        validity: PlanValidity,
    ) -> Option<Arc<PlannedAggregate>> {
        let (table, schema, aggregations, source) = self.global_inputs(stmt, ctx)?;
        // One count or an index's bound vouches for itself read through no
        // view. Several calls each vouch only for their own read, and a sum
        // or a column's bound may cross a seal: the regular path reads those
        // through the statement's snapshot
        let [agg] = aggregations.as_slice() else {
            return None;
        };
        let count = agg.counts_rows() || agg.name == "COUNT" && agg.distinct;
        match source {
            GlobalSource::Stats if count => {}
            GlobalSource::IndexBound => {}
            _ => return None,
        }
        let source_columns = schema.column_names_arc();
        Some(Arc::new(PlannedAggregate {
            alone: Self::calls_alone(stmt, &aggregations).map(CompactArc::new),
            source,
            aggregations,
            table,
            source_columns,
            validity,
        }))
    }

    /// The table, its schema, the calls and the source of a statement of a
    /// SELECT list over one table and no other clause, that aggregates it
    /// into one row; None for any other statement
    fn global_inputs(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<(
        SmartString,
        CompactArc<Schema>,
        Vec<SqlAggregateFunction>,
        GlobalSource,
    )> {
        // The plan runs a SELECT list over one table and nothing else: a
        // statement with any other clause has the regular path run it whole
        let list_alone = !stmt.distinct
            && stmt.distinct_on.is_empty()
            && stmt.with.is_none()
            && stmt.where_clause.is_none()
            && stmt.group_by.columns.is_empty()
            && matches!(stmt.group_by.modifier, GroupByModifier::None)
            && stmt.having.is_none()
            && stmt.window_defs.is_empty()
            && stmt.order_by.is_empty()
            && stmt.limit.is_none()
            && stmt.offset.is_none()
            && stmt.set_operations.is_empty();
        let shape = get_classification(stmt);
        if !list_alone
            || !shape.has_aggregation
            || shape.has_window_functions
            || shape.output_has_subqueries
        {
            return None;
        }
        let Some(Expression::TableSource(ts)) = stmt.table_expr.as_deref() else {
            return None;
        };
        let table = &ts.name.value_lower;
        if ts.as_of.is_some() || ctx.get_cte_by_lower(table).is_some() {
            return None;
        }
        let store = self.engine.get_version_store(table).ok()?;
        let schema = store.schema_and_token().0;
        let indexed = |column: &str| bounds(store.get_index_by_column(column));
        let plan = self
            .aggregate_read_plan(stmt, &shape, ctx, &schema, indexed)
            .ok()??;
        let AggregateRead::Global { source, .. } = plan.read else {
            return None;
        };
        Some((table.clone(), schema, plan.aggregations, source))
    }
}
