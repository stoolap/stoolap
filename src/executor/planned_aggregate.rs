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

use std::sync::Arc;

use crate::common::deadline::Deadline;
use crate::common::names::eq_names;
use crate::common::{CompactArc, CompactVec, SmartString};
use crate::core::{IndexType, Result, Row, RowVec, Schema, Value};
use crate::parser::ast::{Expression, GroupByModifier, SelectStatement};
use crate::storage::expression::Expression as StorageExpr;
use crate::storage::mvcc::version_store::{AggregateInput, GroupBy};
use crate::storage::traits::transaction::{StatementTable, TableUse};
use crate::storage::traits::{Index, QueryResult, Table};

use super::aggregation::{AggregatePath, AggregateSlots, Modifiers, SqlAggregateFunction};
use super::context::ExecutionContext;
use super::expression::SubqueryExecutor;
use super::grouped_aggregate::GroupedSource;
use super::prepared_select::PreparedCalls;
use super::query_classification::{get_classification, QueryClassification};
use super::Executor;

/// Where a statement that aggregates one table without grouping reads its
/// one row from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalSource {
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

/// The places in `names`, a table's columns, of the columns `expr` names,
/// in the order it first names them; None for a name the table lacks, one
/// a table qualifies, or a call of a function an application registered,
/// which may call back into the database while the table reads its rows
fn named_columns(expr: &Expression, names: &[String]) -> Option<Vec<usize>> {
    fn walk(expr: &Expression, names: &[String], columns: &mut Vec<usize>) -> Option<()> {
        let name = match expr {
            Expression::Identifier(id) => Some(&id.value),
            Expression::Column(bound) => Some(&bound.name.value),
            Expression::QualifiedIdentifier(_) => return None,
            Expression::FunctionCall(call)
                if !crate::functions::registry::global_registry().is_built_in(&call.function) =>
            {
                return None
            }
            _ => None,
        };
        if let Some(name) = name {
            let at = names.iter().position(|c| eq_names(c, name))?;
            if !columns.contains(&at) {
                columns.push(at);
            }
        }
        expr.try_for_each_child(&mut |child| walk(child, names, columns).ok_or(()))
            .ok()
    }
    let mut columns = Vec::new();
    walk(expr, names, &mut columns)?;
    Some(columns)
}

/// The column each call reads alone, and each call's expression compiled
/// for the table to compute when a prepared statement holds them
pub(crate) type CallReads<'a> = (&'a [Option<usize>], Option<&'a [Option<CompiledArgument>]>);

/// A call's expression compiled for the table to compute: its program,
/// its parameters not yet bound, and the table columns it reads
#[derive(Debug)]
pub(crate) struct CompiledArgument {
    program: super::expression::SharedProgram,
    columns: Vec<usize>,
}

/// `agg`'s expression compiled over `names`, the table's columns, as the
/// statement's own path compiles it; None when the table cannot compute it
pub(crate) fn compiled_argument(
    agg: &SqlAggregateFunction,
    names: &[String],
) -> Option<CompiledArgument> {
    use super::expression::{CompileContext, ExprCompiler, Op};
    if !expression_call(agg) {
        return None;
    }
    let expr = agg.expression.as_ref()?;
    let columns = named_columns(expr, names)?;
    // The program reads column `k` of a row holding the named columns alone,
    // so a name it reads beyond them fails to compile
    let named: Vec<String> = columns.iter().map(|&at| names[at].clone()).collect();
    let compile = CompileContext::new(&named, crate::functions::registry::global_registry());
    let program = ExprCompiler::new(&compile).compile(expr).ok()?;
    let row_only = !program.has_subqueries()
        && !program
            .ops()
            .iter()
            .any(|op| matches!(op, Op::LoadTransactionId));
    row_only.then(|| CompiledArgument {
        program: CompactArc::new(program),
        columns,
    })
}

/// Whether a COUNT, SUM, AVG, MIN or MAX reads an expression the table may
/// compute: one that answers the same for the same row, with no other
/// modifier
pub(crate) fn expression_call(agg: &SqlAggregateFunction) -> bool {
    matches!(agg.name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
        && agg.answered_with(Modifiers::EXPRESSION)
        && agg.expression.as_ref().is_some_and(|expr| {
            super::volatility::expression_volatility(
                expr,
                crate::functions::registry::global_registry(),
            ) == crate::functions::Volatility::Fixed
        })
}

/// What a call reads for the table to compute it: its column of `columns`,
/// or its expression over columns of `schema`, `held` when a prepared
/// statement compiled it before, bound to `ctx`'s parameters; None for
/// anything else
pub(crate) fn call_input(
    agg: &SqlAggregateFunction,
    (column, held): (Option<usize>, Option<&CompiledArgument>),
    schema: &Schema,
    ctx: Option<&ExecutionContext>,
) -> Option<AggregateInput> {
    use super::expression::{ArgumentProgram, Program};
    if let Some(column) = column {
        return Some(AggregateInput::Column(column));
    }
    let compiled;
    let argument = match held {
        Some(held) => held,
        None => {
            compiled = compiled_argument(agg, &schema.column_names_arc())?;
            &compiled
        }
    };
    let params = ctx.map(|ctx| ctx.params().to_vec()).unwrap_or_default();
    let program = Program::bound(CompactArc::clone(&argument.program), &params);
    Some(AggregateInput::Program(Arc::new(ArgumentProgram::new(
        program,
        argument.columns.clone(),
        params,
        (agg.name.clone(), agg.column.clone()),
    ))))
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
    pub(crate) read: AggregateRead,
}

/// Where an aggregate over one table reads from
pub(crate) enum AggregateRead {
    /// Its one row from `source`
    Global { source: GlobalSource },
    /// Its groups by the keys of `groups`, each named, from `source`;
    /// `walks` whether the index walk reads them when the table declines,
    /// which only one column's groups may
    Grouped {
        source: GroupedSource,
        groups: Vec<(String, GroupBy)>,
        walks: bool,
    },
}

/// A statement without WHERE whose one row is one count the table keeps or
/// an index's bound, held with its plan and answered before the run opens
/// the statement's snapshot
#[derive(Debug)]
pub struct PlannedAggregate {
    source: GlobalSource,
    aggregations: Vec<SqlAggregateFunction>,
    table: SmartString,
    /// The table's columns, which the SELECT list reads through the calls
    source_columns: CompactArc<Vec<String>>,
    /// The output names of a SELECT list of nothing but the calls
    alone: Option<CompactArc<Vec<String>>>,
    /// The statement planned, whose SELECT list a run computes
    stmt: SelectStatement,
}

impl Executor {
    /// The plan of `stmt`, which aggregates one table of `schema` whose
    /// ordered indexes `indexed` names: its calls and the source the rules
    /// pick; None for a statement the aggregation of its read rows answers.
    /// The calls' extra arguments are read only for a run, `runs`. Inlined
    /// into the table scan, it costs that scan's loop more per row
    #[inline(never)]
    pub(crate) fn aggregate_read_plan(
        &self,
        stmt: &SelectStatement,
        classification: &QueryClassification,
        (ctx, runs): (&ExecutionContext, bool),
        schema: &Schema,
        indexed: impl Fn(&str) -> bool,
    ) -> Result<Option<AggregatePlan>> {
        if classification.has_window_functions {
            return Ok(None);
        }
        if classification.has_group_by {
            return self.grouped_read_plan(stmt, classification, (ctx, runs), schema);
        }
        if classification.has_having {
            return Ok(None);
        }
        // The table runs no WHERE over a subquery: its rows are the scan's to
        // read
        if classification.has_where && classification.where_has_subqueries {
            return Ok(None);
        }
        let (aggregations, non_aggregated) = match runs {
            true => self.parse_aggregations(stmt, Some(ctx))?,
            false => self.parse_aggregation_shape(stmt)?,
        };
        let pushed = Self::pushed_where(stmt, schema, ctx);
        let plain = non_aggregated.is_empty();
        let source = Self::global_source(stmt, &aggregations, plain, pushed.is_some(), indexed);
        Ok(Some(AggregatePlan {
            aggregations,
            read: AggregateRead::Global { source },
        }))
    }

    /// `stmt`'s WHERE as the table of `schema` runs it, when it runs the
    /// whole of it; None without a WHERE
    pub(crate) fn pushed_where(
        stmt: &SelectStatement,
        schema: &Schema,
        ctx: &ExecutionContext,
    ) -> Option<Box<dyn StorageExpr>> {
        match super::pushdown::try_pushdown(stmt.where_clause.as_deref()?, schema, Some(ctx)) {
            (Some(pushed), false) => Some(pushed),
            _ => None,
        }
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
        // The table computes the calls over the rows its WHERE passes: a
        // column's count among them, and an expression over columns
        let computed = aggregations.iter().all(|agg| {
            plain_call(agg)
                || agg.counts_rows()
                || agg.name == "COUNT" && agg.answered_with(Modifiers::NONE)
                || expression_call(agg)
        });
        if stmt.where_clause.is_none() {
            // The stats count rows and an index's distinct values, not a column's
            let stats = aggregations.iter().all(|agg| {
                plain_call(agg)
                    || agg.counts_rows()
                    || agg.name == "COUNT" && agg.distinct && agg.answered_with(Modifiers::DISTINCT)
            });
            if !stats {
                // An expression the stats do not keep is computed over every row
                return match computed && aggregations.iter().any(expression_call) {
                    true => GlobalSource::StorageFiltered,
                    false => GlobalSource::Scan,
                };
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
        match pushed && computed {
            true => GlobalSource::StorageFiltered,
            false => GlobalSource::Scan,
        }
    }

    /// The one row of `aggregations` over `table` from `source`, each call
    /// reading its column of `columns` or its expression `held` compiled;
    /// None when the table cannot answer from there and the rows are read
    /// instead. A `quiet` read, through no view, takes only a count that
    /// vouches for itself; `pushed` is the WHERE the table runs, `filtered`
    /// whether the statement has a WHERE, and `ctx` holds the parameters a
    /// call's expression reads
    fn global_values(
        source: GlobalSource,
        table: &dyn Table,
        aggregations: &[SqlAggregateFunction],
        (columns, held): CallReads<'_>,
        quiet: bool,
        (pushed, filtered, ctx): (Option<&dyn StorageExpr>, bool, Option<&ExecutionContext>),
        deadline: Deadline,
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
                Ok(
                    Self::stats_values(table, aggregations, columns, quiet, deadline)?
                        .map(|values| (GlobalSource::Stats, values)),
                )
            }
            GlobalSource::Stats => {
                Ok(
                    Self::stats_values(table, aggregations, columns, quiet, deadline)?
                        .map(|values| (source, values)),
                )
            }
            // A WHERE the table does not run leaves the rows to the scan
            GlobalSource::StorageFiltered if quiet || (filtered && pushed.is_none()) => Ok(None),
            GlobalSource::StorageFiltered => {
                let read = (aggregations, (columns, held), ctx);
                Ok(Self::filtered_values(table, pushed, read, deadline)?
                    .map(|values| (source, values)))
            }
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
        deadline: Deadline,
    ) -> Result<Option<CompactVec<Value>>> {
        let mut values = CompactVec::with_capacity(aggregations.len());
        for (agg, &column) in aggregations.iter().zip(columns) {
            let null = || Value::null(crate::core::DataType::Integer);
            let value = match (agg.name.as_str(), column) {
                ("COUNT", _) if agg.distinct => table
                    .get_partition_count(bare_column(agg), deadline)?
                    .map(|count| Value::Integer(count as i64)),
                // Through no view only the count that vouches for itself; through
                // one, the count of what the view sees
                ("COUNT", _) if agg.counts_rows() => match quiet {
                    true => table.fast_row_count()?,
                    false => Some(table.row_count(deadline)?),
                }
                .map(|count| Value::Integer(count as i64)),
                ("SUM", Some(idx)) => table
                    .sum_column(idx, deadline)?
                    .map(|sum| sum.sum())
                    .transpose()?,
                ("AVG", Some(idx)) => table
                    .avg_column(idx, deadline)?
                    .map(|sum| sum.avg())
                    .transpose()?,
                ("MIN", Some(idx)) => table
                    .min_column(idx, deadline)?
                    .map(|min| min.unwrap_or_else(null)),
                ("MAX", Some(idx)) => table
                    .max_column(idx, deadline)?
                    .map(|max| max.unwrap_or_else(null)),
                _ => None,
            };
            match value {
                Some(value) => values.push(value),
                None => return Ok(None),
            }
        }
        // A read whose last charge read no clock is checked once more
        deadline.check()?;
        Ok(Some(values))
    }

    /// Every call computed by the table over the rows the WHERE passes;
    /// None when the table cannot run the WHERE or the calls
    fn filtered_values(
        table: &dyn Table,
        pushed: Option<&dyn StorageExpr>,
        (aggregations, (columns, held), ctx): (
            &[SqlAggregateFunction],
            CallReads<'_>,
            Option<&ExecutionContext>,
        ),
        deadline: Deadline,
    ) -> Result<Option<CompactVec<Value>>> {
        use crate::storage::mvcc::version_store::AggregateOp;
        let mut ops = Vec::with_capacity(aggregations.len());
        for (at, (agg, &column)) in aggregations.iter().zip(columns).enumerate() {
            if agg.name == "COUNT" && agg.counts_rows() {
                ops.push((AggregateOp::CountStar, AggregateInput::Column(0)));
                continue;
            }
            let op = match agg.name.as_str() {
                "COUNT" => AggregateOp::Count,
                "SUM" => AggregateOp::Sum,
                "AVG" => AggregateOp::Avg,
                "MIN" => AggregateOp::Min,
                "MAX" => AggregateOp::Max,
                _ => return Ok(None),
            };
            let argument = held.and_then(|held| held[at].as_ref());
            let Some(input) = call_input(agg, (column, argument), table.schema(), ctx) else {
                return Ok(None);
            };
            ops.push((op, input));
        }
        let Some(groups) = table.compute_grouped_aggregates(&[], &ops, pushed, deadline)? else {
            return Ok(None);
        };
        // The one group of an aggregation without GROUP BY, there with no rows
        Ok(groups
            .into_rows()?
            .into_iter()
            .next()
            .map(|(_, row)| row.into_values().into_iter().collect()))
    }

    /// Each call's column in `columns`, the one table's, whatever qualifier
    /// names it
    pub(crate) fn call_columns(
        aggregations: &[SqlAggregateFunction],
        columns: &[String],
    ) -> Vec<Option<usize>> {
        aggregations
            .iter()
            .map(|agg| {
                let column = bare_column(agg);
                let named = || columns.iter().position(|c| eq_names(c, column));
                agg.expression.is_none().then(named).flatten()
            })
            .collect()
    }

    /// Each call's expression compiled over `columns`, the one table's, for
    /// the table to compute; None for a call it cannot
    pub(crate) fn call_arguments(
        aggregations: &[SqlAggregateFunction],
        columns: &[String],
    ) -> Vec<Option<CompiledArgument>> {
        aggregations
            .iter()
            .map(|agg| compiled_argument(agg, columns))
            .collect()
    }

    /// The result of the one row `rows` holds for `aggregations`, the end
    /// of every global aggregate path; `alone` the output names a plan found
    /// for a SELECT list of nothing but the calls, `held` the prepared calls
    /// that hold its projection
    #[allow(clippy::too_many_arguments)]
    fn finish_global(
        &self,
        path: AggregatePath,
        stmt: &SelectStatement,
        aggregations: &[SqlAggregateFunction],
        source_columns: &[String],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
        (alone, held): (Option<&CompactArc<Vec<String>>>, Option<&PreparedCalls>),
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
            held,
        };
        self.finish_aggregation(path, slots, ctx, names, rows)
    }

    /// `stmt`'s one row of `aggregations` over `table`, from `source` with
    /// the WHERE the table runs, or else from the group `scan` feeds the
    /// rows it reads; None when `scan` reads none
    pub(super) fn global_answer(
        &self,
        table: &dyn Table,
        stmt: &SelectStatement,
        (aggregations, (source, pushed)): (
            &[SqlAggregateFunction],
            (GlobalSource, Option<&dyn StorageExpr>),
        ),
        ctx: &ExecutionContext,
        (blocks, prepared): (Option<&dyn SubqueryExecutor>, Option<&PreparedCalls>),
        scan: impl FnOnce() -> Result<Option<RowVec>>,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        let source_columns = table.schema().column_names_arc();
        let places = prepared.map(|calls| calls.places(&source_columns));
        let found;
        let reads: CallReads<'_> = match places.as_deref() {
            Some(places) => (&places.calls, Some(&places.arguments)),
            None => {
                found = Self::call_columns(aggregations, &source_columns);
                (&found, None)
            }
        };
        let alone = prepared.and_then(|calls| calls.alone.as_ref());
        let deadline = ctx.deadline_of_call();
        let row = Self::global_values(
            source,
            table,
            aggregations,
            reads,
            false,
            (pushed, stmt.where_clause.is_some(), Some(ctx)),
            deadline,
        )?;
        let (path, rows) = match row {
            Some((answered, values)) => {
                let mut rows = RowVec::with_capacity(1);
                rows.push((0, Row::from_compact_vec(values)));
                (answered.path(), rows)
            }
            None => match scan()? {
                Some(rows) => (AggregatePath::GlobalScan, rows),
                None => return Ok(None),
            },
        };
        self.finish_global(
            path,
            stmt,
            aggregations,
            &source_columns,
            ctx,
            blocks,
            (alone, prepared),
            rows,
        )
        .map(Some)
    }

    /// The statement `planned` was planned for, answered by it, held with
    /// its plan: one count the table keeps or an index's bound, read through
    /// committed state. None for the plan's rows to answer, when the table
    /// cannot answer so in this run
    pub(crate) fn run_quiet_aggregate(
        &self,
        planned: &PlannedAggregate,
        ctx: &ExecutionContext,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        if let Err(error) = ctx.check_deadline() {
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
                (&[None], None),
                quiet,
                (None, true, None),
                ctx.deadline_of_call(),
            )?;
            let Some((answered, values)) = row else {
                return Ok(None);
            };
            let mut rows = RowVec::with_capacity(1);
            rows.push((0, Row::from_compact_vec(values)));
            self.finish_global(
                answered.path(),
                &planned.stmt,
                &planned.aggregations,
                &planned.source_columns,
                ctx,
                None,
                (planned.alone.as_ref(), None),
                rows,
            )
            .map(Some)
        };
        run().transpose()
    }

    /// The plan of a statement over one table, without WHERE, grouping,
    /// ORDER BY, LIMIT or OFFSET, whose one row is one count the table keeps
    /// or an index's bound
    pub(crate) fn aggregate_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<PlannedAggregate> {
        let (table, schema, aggregations, source) = self.held_aggregate(stmt, ctx)?;
        let source_columns = schema.column_names_arc();
        Some(PlannedAggregate {
            alone: Self::calls_alone(stmt, &aggregations).map(CompactArc::new),
            source,
            aggregations,
            table,
            source_columns,
            stmt: stmt.clone(),
        })
    }

    /// What [`Self::global_inputs`] gives for a statement the aggregate plan
    /// takes: one count or an index's bound, which vouches for itself read
    /// through no view. Several calls each vouch only for their own read,
    /// and a sum or a column's bound may cross a seal: the statement's
    /// snapshot reads those
    fn held_aggregate(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<(
        SmartString,
        CompactArc<Schema>,
        Vec<SqlAggregateFunction>,
        GlobalSource,
    )> {
        let inputs = self.global_inputs(stmt, ctx)?;
        let [agg] = inputs.2.as_slice() else {
            return None;
        };
        let count = agg.counts_rows() || agg.name == "COUNT" && agg.distinct;
        match inputs.3 {
            GlobalSource::Stats if count => Some(inputs),
            GlobalSource::IndexBound => Some(inputs),
            _ => None,
        }
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
            .aggregate_read_plan(stmt, &shape, (ctx, false), &schema, indexed)
            .ok()??;
        let AggregateRead::Global { source, .. } = plan.read else {
            return None;
        };
        Some((table.clone(), schema, plan.aggregations, source))
    }
}
