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

//! Runs a statement the planner planned: its access, then its steps

use std::sync::{Arc, OnceLock, RwLock, Weak};

use smallvec::SmallVec;

use crate::core::{Error, KeyRows, Result, Row, RowVec, Value};
use crate::parser::ast::{Expression, SelectStatement};
use crate::planner::{steps_over, KeyRead, PlanText, Step};
use crate::storage::expression::ConstBoolExpr;
use crate::storage::mvcc::registry::StatementView;
use crate::storage::mvcc::table::normalize_row_to_schema;
use crate::storage::mvcc::version_store::{LayoutToken, VersionStore};
use crate::storage::traits::transaction::{StatementTable, TableUse};
use crate::storage::traits::{Engine, QueryResult};

use super::context::ExecutionContext;
use super::expression::{ExecuteContext, ExprVM};
use super::prepared_select::{first_branch_of, PreparedSelect};
use super::query_cache::{check_compiled_token, or_normal_path, CompiledExecution};
use super::query_classification::{get_classification, QueryClassification};
use super::result::ExecutorResult;
use super::row_count::static_row_counts;
use super::Executor;

impl Executor {
    /// `run` over the plan `compiled` holds for `stmt` when it reads a key,
    /// planned now in `ctx` when it holds none yet; None for a statement
    /// whose plan reads no key, or that is not planned and `ctx` is not
    /// given for. `classification` is the plan's slot for the statement's
    /// classification, which planning fills
    pub(crate) fn with_planned<R>(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
        classification: Option<&OnceLock<Arc<QueryClassification>>>,
        ctx: Option<&ExecutionContext>,
        run: impl FnOnce(&PreparedSelect) -> R,
    ) -> Option<R> {
        let epoch = self.engine.schema_epoch();
        {
            let held = compiled.read().ok()?;
            match &*held {
                CompiledExecution::PreparedSelect(p) if p.validity.holds(&*self.engine, epoch) => {
                    // The runner reads a statement prepared for the tables as they are
                    p.key_read()?;
                    // A context of other CTEs than it was planned in may hide its table
                    if ctx.is_none_or(|ctx| p.cte_scope.holds(ctx)) {
                        #[cfg(feature = "test-failpoints")]
                        crate::test_failpoints::compiled_epoch_checked();
                        return Some(run(p));
                    }
                }
                CompiledExecution::NotOptimizable(validity)
                    if validity.holds(&*self.engine, epoch) =>
                {
                    return None
                }
                // One prepared for tables since changed is planned again
                CompiledExecution::Unknown
                | CompiledExecution::NotOptimizable(_)
                | CompiledExecution::PreparedSelect(_) => {}
                _ => return None,
            }
        }
        let planned = self.plan_into(stmt, compiled, classification, ctx?, epoch)?;
        planned.key_read()?;
        Some(run(&planned))
    }

    /// The plan of `stmt`, kept in `compiled`; a statement prepared but not
    /// planned is kept as one the planner declines while its tables hold
    fn plan_into(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
        classification: Option<&OnceLock<Arc<QueryClassification>>>,
        ctx: &ExecutionContext,
        epoch: u64,
    ) -> Option<Arc<PreparedSelect>> {
        let shape = match classification {
            Some(slot) => Arc::clone(slot.get_or_init(|| get_classification(stmt))),
            None => get_classification(stmt),
        };
        if shape.has_aggregation
            || shape.has_window_functions
            || shape.has_group_by
            || shape.has_having
            || shape.has_cte
            || shape.output_has_subqueries
            || shape.where_has_subqueries
            || shape.has_joins
            || Self::vector_distance_order(stmt).is_some()
        {
            return None;
        }
        let Expression::TableSource(ts) = stmt.table_expr.as_deref()? else {
            return None;
        };
        let table = &ts.name.value_lower;
        if ts.as_of.is_some() || ctx.get_cte_by_lower(table).is_some() {
            return None;
        }
        // DISTINCT, DISTINCT ON and set operations as the runner's family takes them
        if shape.has_distinct || shape.has_distinct_on || shape.has_set_operations {
            self.planned_family(stmt, Some(ctx)).ok()?;
        }
        // Read before the schema: a change in between leaves this behind
        let validity = self.select_validity(stmt, epoch);
        let store = self.engine.get_version_store(table).ok()?;
        let (schema, token) = store.schema_and_token();
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::compile_schema_read();
        let scope = ts.alias.as_ref().map_or(table, |alias| &alias.value_lower);
        let param_count = ctx.params().len();
        let branch = first_branch_of(stmt);
        let prepared =
            PreparedSelect::prepare(self, ctx, &branch, &schema, scope, param_count, epoch)
                .ok()
                .map(|mut prepared| {
                    // A LIMIT or OFFSET read per run leaves the statement to a
                    // scan, and so do a DISTINCT, which the key read does not
                    // take, and a set operation, whose rows are its branches' too
                    let counts = static_row_counts(stmt)
                        .filter(|_| stmt.set_operations.is_empty() && !stmt.distinct);
                    if let Some(counts) = counts {
                        prepared.read_by_key(
                            (table.clone(), &store),
                            (schema, token),
                            (counts.offset, counts.limit),
                        );
                    }
                    Arc::new(prepared)
                });
        let mut slot = compiled.write().ok()?;
        if matches!(
            &*slot,
            CompiledExecution::Unknown
                | CompiledExecution::NotOptimizable(_)
                | CompiledExecution::PreparedSelect(_)
        ) {
            *slot = match &prepared {
                Some(prepared) => CompiledExecution::PreparedSelect(Arc::clone(prepared)),
                None => CompiledExecution::NotOptimizable(validity),
            };
        }
        prepared
    }

    /// The plan the planner makes of `stmt` in `ctx`, as EXPLAIN shows it;
    /// None for a statement it does not plan
    pub(crate) fn explain_plan(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<Vec<String>> {
        let slot = RwLock::new(CompiledExecution::Unknown);
        self.with_planned(stmt, &slot, None, Some(ctx), |_| ());
        if let CompiledExecution::PreparedSelect(prepared) = &*slot.read().ok()? {
            let Some(Expression::TableSource(source)) = stmt.table_expr.as_deref() else {
                return None;
            };
            let table = &source.name.value_lower;
            let first = prepared.explain(&first_branch_of(stmt), table, Vec::new())?;
            return Some(self.explain_set_operations(stmt, first, ctx));
        }
        self.explain_over_source(stmt, ctx)
            .or_else(|| self.explain_aggregate(stmt, ctx))
    }

    /// `stmt`'s plan over `first`, its first branch's: each set operation
    /// over the rows before it and its right branch as EXPLAIN shows that
    /// statement, under the whole statement's ORDER BY and LIMIT
    fn explain_set_operations(
        &self,
        stmt: &SelectStatement,
        first: Vec<String>,
        ctx: &ExecutionContext,
    ) -> Vec<String> {
        if stmt.set_operations.is_empty() {
            return first;
        }
        let branch_ctx = ctx.with_incremented_query_depth();
        let mut lines = first;
        for op in &stmt.set_operations {
            let right = self.explain_lines(&op.right, &branch_ctx);
            lines = std::iter::once(op.operation.to_string())
                .chain(
                    lines
                        .into_iter()
                        .chain(right)
                        .map(|line| format!("  {line}")),
                )
                .collect();
        }
        let steps = Self::finishing_steps(stmt, &[]);
        steps_over(&steps, &Self::finishing_text(stmt, ""), lines)
    }

    /// The steps the statement's own clauses take over its executor's rows,
    /// outermost first: its LIMIT, its ORDER BY, cut by the LIMIT or whole,
    /// then `inner`
    pub(super) fn finishing_steps(stmt: &SelectStatement, inner: &[Step]) -> SmallVec<[Step; 4]> {
        let mut steps = SmallVec::new();
        if stmt.limit.is_some() || stmt.offset.is_some() {
            steps.push(Step::Limit);
        }
        match (stmt.order_by.is_empty(), stmt.limit.is_some()) {
            (true, _) => {}
            (false, true) => steps.push(Step::TopN),
            (false, false) => steps.push(Step::Sort),
        }
        steps.extend_from_slice(inner);
        steps
    }

    /// What EXPLAIN names [`Self::finishing_steps`] by, over `table`
    pub(super) fn finishing_text<'a>(stmt: &SelectStatement, table: &'a str) -> PlanText<'a> {
        let keys: Vec<String> = stmt.order_by.iter().map(|o| o.to_string()).collect();
        PlanText {
            table,
            limit: stmt.limit.as_deref().map(Expression::to_string),
            offset: stmt
                .offset
                .as_deref()
                .filter(|o| !matches!(o, Expression::IntegerLiteral(lit) if lit.value == 0))
                .map(Expression::to_string),
            keys: Some(keys.join(", ")),
            distinct_on: None,
            having: stmt.having.as_deref().map(Expression::to_string),
            names: &[],
            pushed: None,
            ordered: None,
            empty_unless: None,
            body: Vec::new(),
        }
    }

    /// `stmt` run by the plan `compiled` holds, planned now in `ctx` when it
    /// holds none; None for the regular path to answer
    pub(crate) fn run_select_plan(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
        classification: Option<&OnceLock<Arc<QueryClassification>>>,
        ctx: Option<&ExecutionContext>,
        params: &[Value],
        in_txn: bool,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        self.with_planned(stmt, compiled, classification, ctx, |planned| {
            if let Some(ctx) = ctx {
                // In a transaction the normal path arms a timeout; every
                // statement refuses a cancel
                if in_txn && ctx.has_timeout() {
                    return None;
                }
                if let Err(e) = ctx.check_cancelled() {
                    return Some(Err(e));
                }
            }
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::pk_lookup_resolved();
            // A constant term that is false keeps every row out unread
            if planned.has_constant_where() {
                let held;
                let terms_ctx = match ctx {
                    Some(ctx) => ctx,
                    None => {
                        held = ExecutionContext::with_params(params.into());
                        &held
                    }
                };
                if planned.keeps_no_row(stmt, terms_ctx) {
                    let (plan, _) = planned.key_read()?;
                    let empty = ExecutorResult::with_arc_columns(plan.names.clone(), RowVec::new());
                    return Some(Ok(Box::new(empty) as Box<dyn QueryResult>));
                }
            }
            let view = ctx.and_then(ExecutionContext::statement_view);
            // CURRENT_TRANSACTION_ID() reads the open transaction's id
            let transaction = match ctx.and_then(ExecutionContext::transaction_id) {
                Some(id) => Some(id),
                None if in_txn => self.active_txn_id().map(|id| id as u64),
                None => None,
            };
            self.run_planned(planned, params, (in_txn, transaction), view)
        })
        .flatten()
    }

    /// The rows of `planned`'s key read for `params`, in the open
    /// transaction `in_txn` says and with the transaction id its programs
    /// read; None when this execution's key names no row, or a unique read
    /// declines, which the plan's other read answers
    pub(crate) fn run_planned(
        &self,
        planned: &PreparedSelect,
        params: &[Value],
        (in_txn, transaction): (bool, Option<u64>),
        view: Option<&StatementView>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Read through the statement, and `read_planned` inlined: a key read
        // borrowed as an argument has its fields loaded ahead of every row
        let (plan, unique) = planned.key_read()?;
        let rows = match unique {
            false => {
                let key = plan.key.integer(params)?;
                self.fetch_pk_row(&plan.table, key, plan.token, in_txn, view, &plan.store)
            }
            true => self
                .fetch_unique_row(plan, params, in_txn, view)
                .transpose()?,
        };
        or_normal_path(rows.and_then(|rows| self.read_planned(plan, rows, params, transaction)))
    }

    /// The row `plan`'s unique index holds for its key's value, read as the
    /// latest committed state; None outside an autocommit read committed
    /// statement without a view, or when the index or the read declines
    #[inline(never)]
    fn fetch_unique_row(
        &self,
        plan: &KeyRead,
        params: &[Value],
        in_txn: bool,
        view: Option<&StatementView>,
    ) -> Result<Option<KeyRows>> {
        if in_txn
            || view.is_some()
            || self.isolation_level() != crate::core::IsolationLevel::ReadCommitted
        {
            return Ok(None);
        }
        let Some(value) = plan.key.value(params).filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let Some(store) = plan.store.upgrade().filter(|store| !store.is_closed()) else {
            return Ok(None);
        };
        let rows = self.engine.fetch_latest_unique_in(
            &plan.table,
            &store,
            (plan.key.column(), &value),
            plan.token,
        )?;
        #[cfg(feature = "test-failpoints")]
        if rows.is_some() {
            crate::test_failpoints::note_read("exec.unique_get");
        }
        Ok(rows)
    }

    #[inline(always)]
    fn read_planned(
        &self,
        plan: &KeyRead,
        rows: KeyRows,
        params: &[Value],
        transaction: Option<u64>,
    ) -> Result<Box<dyn QueryResult>> {
        let mut vm: Option<ExprVM> = None;
        let mut out = RowVec::with_capacity(1);
        let mut passed = 0;
        for (id, row) in rows {
            let row = normalize_row_to_schema(row, &plan.schema);
            if let Some(filter) = &plan.filter {
                let exec = ExecuteContext::with_common_params(&row, params, transaction);
                let vm = vm.get_or_insert_with(ExprVM::new);
                if !vm.execute_bool_checked(filter, &exec)? {
                    continue;
                }
            }
            passed += 1;
            if passed <= plan.offset {
                continue;
            }
            if plan.limit.is_some_and(|limit| out.len() >= limit) {
                break;
            }
            if plan.whole_row {
                out.push((id, row));
                continue;
            }
            if let Some(columns) = plan.projection.column_indices() {
                let values = columns
                    .iter()
                    .map(|&i| row.get(i).cloned().unwrap_or_else(Value::null_unknown));
                out.push((id, Row::from_values(values.collect())));
                continue;
            }
            let exec = ExecuteContext::with_common_params(&row, params, transaction);
            let mut projected = Row::with_capacity(plan.projection.width());
            let vm = vm.get_or_insert_with(ExprVM::new);
            plan.projection.project_into(vm, &exec, &mut projected)?;
            out.push((id, projected));
        }
        Ok(Box::new(ExecutorResult::with_arc_columns(
            plan.names.clone(),
            out,
        )))
    }

    /// The row with id `pk_value`: committed state outside a transaction,
    /// else what the active transaction sees through a handle taken for this
    /// statement, refused as SchemaChanged unless it holds layout `token`.
    /// A view the statement already took, `taken`, holds the read to it.
    /// `held` is the table's store as a plan keeps it, read while it is open
    fn fetch_pk_row(
        &self,
        table_name: &str,
        pk_value: i64,
        token: LayoutToken,
        in_txn: bool,
        taken: Option<&StatementView>,
        held: &Weak<VersionStore>,
    ) -> Result<KeyRows> {
        let read_committed = |level| level == crate::core::IsolationLevel::ReadCommitted;
        if !in_txn {
            if taken.is_none() {
                // Under read committed a committed, live head is this row's
                // latest committed state whatever else publishes to the table
                let latest = if read_committed(self.isolation_level()) {
                    match held.upgrade() {
                        Some(store) if !store.is_closed() => self
                            .engine
                            .fetch_latest_row_in(table_name, &store, pk_value, token)?,
                        _ => self.engine.fetch_latest_row(table_name, pk_value, token)?,
                    }
                } else {
                    self.engine
                        .fetch_rows_by_ids_settled(table_name, &[pk_value], token, true)?
                        .map(KeyRows::Many)
                };
                if let Some(rows) = latest {
                    return Ok(rows);
                }
            }
            // A commit is publishing to the table: read the commits complete
            // now through a view, or at a snapshot's own point
            let view = taken.cloned().or_else(|| self.autocommit_view());
            if let Some(view) = view.as_ref() {
                if let Some(rows) =
                    self.engine
                        .fetch_hot_rows_at(table_name, &[pk_value], token, view)?
                {
                    view.held()?;
                    return Ok(KeyRows::Many(rows));
                }
            }
            let tx = self.begin_transaction()?;
            let table = tx.get_read_table_at(table_name, view.as_ref(), TableUse::Read)?;
            check_compiled_token(&*table, token)?;
            let rows = table.fetch_rows_by_ids(&[pk_value], &ConstBoolExpr::true_expr())?;
            view.as_ref().map_or(Ok(()), StatementView::held)?;
            return Ok(KeyRows::Many(rows));
        }
        let (table, view) = {
            let active = self
                .active_transaction
                .lock()
                .map_err(|_| Error::internal("active transaction lock poisoned"))?;
            // A missing transaction or table is the normal path's to report
            let found = active.as_ref().and_then(|tx| {
                let level = self
                    .engine
                    .registry()
                    .get_isolation_level(tx.transaction.id());
                let view = match taken {
                    Some(view) => Some(view.clone()),
                    None => read_committed(level).then(|| self.capture_view()),
                };
                // The lookup is a SELECT: it writes no table
                tx.transaction
                    .get_read_table_at(table_name, view.as_ref(), TableUse::Read)
                    .ok()
                    .map(|table| (table, view))
            });
            found.ok_or_else(|| Error::SchemaChanged {
                table: table_name.to_string(),
            })?
        };
        check_compiled_token(&*table, token)?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::note_read("exec.pk_lookup_in_transaction");
        let rows = table.fetch_rows_by_ids(&[pk_value], &ConstBoolExpr::true_expr())?;
        view.as_ref().map_or(Ok(()), StatementView::held)?;
        Ok(KeyRows::Many(rows))
    }

    /// A float key qualifies only when it converts to i64 without loss:
    /// truncating 5.5 to 5 would serve row 5 where correct execution
    /// matches nothing. None falls back to the standard path.
    ///
    /// The range check must happen BEFORE the cast: 2^63 saturates to
    /// i64::MAX, and i64::MAX rounds back to 2^63 as f64, so a naive
    /// round-trip equality check passes at the boundary.
    #[inline]
    pub(crate) fn lossless_float_key(f: f64) -> Option<i64> {
        const I64_MIN_F: f64 = -9_223_372_036_854_775_808.0; // -2^63, exact
        const I64_MAX_PLUS_1_F: f64 = 9_223_372_036_854_775_808.0; // 2^63, not representable as i64
        if !(I64_MIN_F..I64_MAX_PLUS_1_F).contains(&f) {
            return None; // Also rejects NaN and infinities
        }
        let v = f as i64;
        (v as f64 == f).then_some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::Executor;

    #[test]
    fn a_cte_naming_the_table_hides_it_from_a_kept_plan() {
        use crate::core::{Row, RowVec, Value};
        use crate::executor::context::ExecutionContext;
        use crate::storage::mvcc::engine::MVCCEngine;
        use std::sync::Arc;
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        let executor = Executor::new(Arc::new(engine));
        let run = |sql: &str, ctx: &ExecutionContext| {
            let statement = crate::parser::parse_sql(sql).unwrap().remove(0);
            executor.execute_statement(&statement, ctx).unwrap()
        };
        let plain = ExecutionContext::new();
        run("CREATE TABLE t (id INTEGER PRIMARY KEY)", &plain);
        run("INSERT INTO t VALUES (1)", &plain);
        let statement = crate::parser::parse_sql("SELECT id + 0 FROM t WHERE id = 1")
            .unwrap()
            .remove(0);
        let count = |ctx: &ExecutionContext| {
            let mut rows = executor.execute_statement(&statement, ctx).unwrap();
            let mut count = 0;
            while rows.next() {
                count += 1;
            }
            count
        };
        assert_eq!(count(&plain), 1, "the table's row");
        let mut ctes = crate::executor::cte::CteRegistry::new();
        let mut rows = RowVec::new();
        rows.push((0, Row::from_values(vec![Value::Integer(2)])));
        ctes.store("t", vec!["id".to_string()], rows);
        let shadowed = ExecutionContext::new().with_cte_data(ctes.data());
        assert_eq!(count(&shadowed), 0, "the plan read the table the CTE hides");
    }

    #[test]
    fn lossless_float_key_boundaries() {
        // Exact conversions pass
        assert_eq!(Executor::lossless_float_key(5.0), Some(5));
        assert_eq!(Executor::lossless_float_key(-5.0), Some(-5));
        assert_eq!(Executor::lossless_float_key(0.0), Some(0));
        // i64::MIN is exactly representable as f64
        assert_eq!(
            Executor::lossless_float_key(-9_223_372_036_854_775_808.0),
            Some(i64::MIN)
        );
        // Fractional values reject
        assert_eq!(Executor::lossless_float_key(5.5), None);
        assert_eq!(Executor::lossless_float_key(-0.5), None);
        // 2^63 saturates to i64::MAX under `as i64` and i64::MAX rounds
        // back to 2^63, so a naive round-trip check passes; the range
        // check must reject it
        assert_eq!(
            Executor::lossless_float_key(9_223_372_036_854_775_808.0),
            None
        );
        assert_eq!(Executor::lossless_float_key(1e19), None);
        assert_eq!(Executor::lossless_float_key(-1e19), None);
        // Non-finite rejects
        assert_eq!(Executor::lossless_float_key(f64::NAN), None);
        assert_eq!(Executor::lossless_float_key(f64::INFINITY), None);
        assert_eq!(Executor::lossless_float_key(f64::NEG_INFINITY), None);
    }
}
