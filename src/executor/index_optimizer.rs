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

use crate::common::{CompactArc, I64Set};
use crate::core::predicate::{kind, PredicateSet};
use crate::core::{DataType, Result, Row, RowVec, Value, ValueSet};
use crate::parser::ast::*;
use crate::storage::traits::{QueryResult, ScanAccess, Table};

use super::context::ExecutionContext;
use super::expression::RowFilter;
use super::query_classification::QueryClassification;
use super::result::ExecutorResult;
use super::row_count::row_counts;
use super::utils::order_by_names_an_alias;
use super::Executor;

/// The most rows a LIMIT's fetch by primary key reads in one batch
const FETCH_BATCH_ROWS: usize = 1024;

/// The most candidate ids an IN list's members may name in all, 512 KiB
/// of ids, before the members are read by a scan instead
const IN_CANDIDATE_CAP: usize = 65_536;

/// The members of a set as the comparison reads them against a numeric
/// column: the numbers, a TEXT read as one; and whether a member was UNKNOWN
/// to every number, a NULL or a TEXT that is no number
fn numeric_members<'a>(members: impl Iterator<Item = &'a Value>) -> (Vec<Value>, bool) {
    let mut numbers = Vec::new();
    let mut unknown = false;
    for member in members {
        match member {
            Value::Integer(_) | Value::Float(_) => numbers.push(member.clone()),
            Value::Text(text) => match crate::core::predicate::parse_sql_number(text) {
                Some(number) => numbers.push(number),
                None => unknown = true,
            },
            Value::Null(_) => unknown = true,
            _ => {}
        }
    }
    (numbers, unknown)
}

/// Whether a set holds a number, which a TEXT column cannot be probed for
fn has_number<'a>(mut members: impl Iterator<Item = &'a Value>) -> bool {
    members.any(|v| matches!(v, Value::Integer(_) | Value::Float(_)))
}

impl Executor {
    /// The one table column `stmt`'s ORDER BY sorts by, as written, with its
    /// direction and NULLS clause: what an index-ordered or a top-k read
    /// answers; None for any other ORDER BY
    pub(crate) fn ordered_read_column(
        stmt: &SelectStatement,
    ) -> Option<(&str, bool, Option<bool>)> {
        let [order_by] = stmt.order_by.as_slice() else {
            return None;
        };
        let column_name = match &order_by.expression {
            Expression::Identifier(id) => id.value.as_str(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.as_str(),
            _ => return None,
        };
        if order_by_names_an_alias(&order_by.expression, stmt) {
            return None;
        }
        Some((column_name, order_by.ascending, order_by.nulls_first))
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

    /// Whether every window function in the statement can run on a fetch cut at
    /// LIMIT + OFFSET rows in the given index order: each window must follow
    /// that order exactly and must not look past the current row.
    pub(crate) fn is_window_safe_for_limit_pushdown(
        stmt: &SelectStatement,
        order_col: &str,
        ascending: bool,
    ) -> bool {
        for col_expr in &stmt.columns {
            if !Self::is_expr_window_safe(col_expr, order_col, ascending) {
                return false;
            }
        }
        true
    }

    /// Whether every window function of `expr` outside its subqueries is safe
    /// for LIMIT pushdown
    fn is_expr_window_safe(expr: &Expression, order_col: &str, ascending: bool) -> bool {
        expr.try_for_each_window(&mut |window| match Self::is_window_safe(
            window, order_col, ascending,
        ) {
            true => Ok(()),
            false => Err(()),
        })
        .is_ok()
    }

    /// Whether `window_expr` is safe for LIMIT pushdown
    fn is_window_safe(
        window_expr: &crate::parser::ast::WindowExpression,
        order_col: &str,
        ascending: bool,
    ) -> bool {
        use crate::parser::ast::{WindowFrameBound, WindowFrameUnit};
        // The fetch follows one window's ORDER BY; a window that is
        // partitioned, ordered by anything else or by more than that
        // key would rank a cut subset
        let follows_fetch_order = window_expr.partition_by.is_empty()
            && window_expr.order_by.len() == 1
            && window_expr.order_by[0].ascending == ascending
            && match &window_expr.order_by[0].expression {
                Expression::Identifier(id) => id.value.eq_ignore_ascii_case(order_col),
                Expression::QualifiedIdentifier(qid) => {
                    qid.name.value.eq_ignore_ascii_case(order_col)
                }
                _ => false,
            };
        if !follows_fetch_order {
            return false;
        }
        let func_name = window_expr.function.function.to_uppercase();
        // A fetch cut at LIMIT rows is wrong for anything that looks past
        // the current row: total-count functions, LEAD, a frame that
        // reaches forward, or a named window whose frame is not visible here
        let forward = |b: &WindowFrameBound| {
            matches!(
                b,
                WindowFrameBound::Following(_) | WindowFrameBound::UnboundedFollowing
            )
        };
        let frame = window_expr.frame.as_ref();
        let reaches_forward = window_expr.window_ref.is_some()
            || frame.is_some_and(|f| forward(&f.start) || f.end.as_ref().is_some_and(forward));
        if reaches_forward
            || matches!(
                func_name.as_str(),
                "NTILE" | "PERCENT_RANK" | "CUME_DIST" | "LEAD"
            )
        {
            return false;
        }
        // A ROWS frame ending at the current row never reads past it. Any
        // other frame, the implicit RANGE one included, also covers the
        // current row's later peers, which the cut fetch may not hold, so
        // only functions that ignore the frame end stay safe there
        frame.is_some_and(|f| f.unit == WindowFrameUnit::Rows)
            || matches!(
                func_name.as_str(),
                "ROW_NUMBER" | "RANK" | "DENSE_RANK" | "LAG" | "FIRST_VALUE"
            )
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
                    Expression::Identifier(id) => id.value.to_string(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
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
        // All window functions (including nested ones) must share the same single
        // simple partition column. If any window is unpartitioned (global), uses
        // complex expressions, or partitions by a different column, bail out.
        let mut canonical: Option<String> = None;
        for col_expr in &stmt.columns {
            if !Self::collect_window_partitions(col_expr, &mut canonical) {
                return None;
            }
        }
        canonical
    }

    /// Whether every window function of `expr` outside its subqueries is
    /// partitioned by the one column `canonical` holds, or takes as the first
    fn collect_window_partitions(expr: &Expression, canonical: &mut Option<String>) -> bool {
        expr.try_for_each_window(&mut |window_expr| {
            if window_expr.partition_by.len() != 1 || window_expr.window_ref.is_some() {
                return Err(());
            }
            let col_name = match &window_expr.partition_by[0] {
                Expression::Identifier(id) => id.value.to_string(),
                Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
                _ => return Err(()),
            };
            match canonical {
                None => *canonical = Some(col_name),
                Some(c) if !c.eq_ignore_ascii_case(&col_name) => return Err(()),
                _ => {}
            }
            Ok(())
        })
        .is_ok()
    }

    /// Whether an IN on `column` may be read by `in_member_rows`: a table
    /// that keeps rows outside its index, an index probed key by key on the
    /// column, and no change of the transaction's own the index would miss
    pub(crate) fn in_members_eligible(table: &dyn Table, column: &str) -> bool {
        table.lookup_index_on_column(column).is_none()
            && !table.has_local_changes()
            && crate::executor::utils::probes_by_equality(table, column)
    }

    /// The filter of the conjuncts beside an IN
    fn remaining_row_filter(
        &self,
        remaining: Option<&Expression>,
        all_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Option<RowFilter>> {
        let Some(remaining) = remaining else {
            return Ok(None);
        };
        Ok(Some(
            RowFilter::new(remaining, all_columns)?.with_context(ctx),
        ))
    }

    /// Whether a LIMIT applies to the rows read here: nothing that needs
    /// every row runs between this path and the result, and the LIMIT
    /// evaluates to a count
    fn limit_applies(stmt: &SelectStatement, ctx: &ExecutionContext) -> bool {
        let limit_here = stmt.order_by.is_empty() && !stmt.distinct && stmt.distinct_on.is_empty();
        limit_here && row_counts(stmt, ctx).is_ok_and(|counts| counts.limit.is_some())
    }

    /// The rows whose `column` is one of `members`, with no LIMIT applying
    /// here: on a table `in_members_eligible` admits, the B-tree's
    /// candidates under one capture, each row read and checked against the
    /// members; else with `scan` a scan with the same check, else None. The
    /// filter `remaining` gives is asked for only when the rows are read here
    pub(crate) fn in_member_rows(
        table: &dyn Table,
        column: &str,
        members: Vec<Value>,
        remaining: impl FnOnce() -> Result<Option<RowFilter>>,
        scan: bool,
    ) -> Result<Option<RowVec>> {
        use crate::storage::expression::{Expression as _, InListExpr};
        use crate::storage::traits::{CappedEqual, ProbeScratch};
        let mut membership = InListExpr::new(column, members);
        membership.prepare_for_schema(table.schema());
        let mut ids = Vec::new();
        // A table `in_members_eligible` refuses is not asked, only scanned
        let found = if Self::in_members_eligible(table, column) {
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::in_members_probed();
            table.equality_candidates_in(
                column,
                membership.values(),
                IN_CANDIDATE_CAP,
                &mut ids,
                &mut ProbeScratch::default(),
            )?
        } else {
            None
        };
        if found == Some(CappedEqual::Copied) {
            #[cfg(any(test, feature = "test-failpoints"))]
            crate::test_failpoints::in_members_read(false);
            let remaining = remaining()?;
            let mut rows = table.fetch_rows_by_ids(&ids, &membership)?;
            if let Some(filter) = remaining {
                filter.retain_checked(&mut rows)?;
            }
            return Ok(Some(rows));
        }
        if !scan {
            return Ok(None);
        }
        let remaining = remaining()?;
        let keep = |row: &Row| {
            remaining
                .as_ref()
                .map_or(Ok(true), |filter| filter.matches_checked(row))
        };
        let columns: Vec<usize> = (0..table.schema().columns.len()).collect();
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::in_members_read(true);
        let plan = table.index_plan(&membership);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        let mut scanner = table.scan_access(access, &columns, None, Some(&membership))?;
        let mut rows = RowVec::new();
        while scanner.next() {
            let row_id = scanner.current_row_id();
            let row = scanner.take_row();
            if keep(&row)? {
                rows.push((row_id, row));
            }
        }
        if let Some(error) = scanner.err() {
            return Err(error.clone());
        }
        scanner.close()?;
        Ok(Some(rows))
    }

    /// Fetches the rows of `row_ids` a batch at a time until `needed`
    /// rows are found or the ids are spent: a member with no row, or with
    /// a row not visible, costs its own fetch and nothing more. A batch
    /// grows from the rows still needed and is bounded whatever the list
    /// holds, so a run of members without a row never makes the fetch
    /// after it read every remaining row
    /// The row ids `members` name on an INTEGER PRIMARY KEY: a member is
    /// the row id, a float read as the comparison reads it
    fn primary_key_members<'v, I: Iterator<Item = &'v Value> + 'v>(
        members: I,
    ) -> impl Iterator<Item = i64> + 'v {
        members.filter_map(|value| match value {
            Value::Integer(id) => Some(*id),
            Value::Float(f) => Self::lossless_float_key(*f),
            _ => None,
        })
    }

    /// Each row id `index` holds for one of `members`, until `out` holds
    /// `target`, checked after every member; false when the index does not
    /// answer for the view the table reads: it holds neither the
    /// transaction's own writes nor, once a commit the statement misses
    /// landed, its keys
    fn index_members<'v>(
        table: &dyn Table,
        index: &dyn crate::storage::traits::Index,
        members: impl Iterator<Item = &'v Value>,
        target: Option<usize>,
        out: &mut Vec<i64>,
    ) -> bool {
        let epoch = table.index_view_epoch();
        // A NULL member equals nothing, so it is not looked up
        let mut members = members.filter(|value| !value.is_null());
        match target {
            None => index.get_row_ids_in_into(&mut members, out),
            Some(target) => {
                while out.len() < target {
                    let Some(value) = members.next() else {
                        break;
                    };
                    index.get_row_ids_in_into(&mut std::iter::once(value), out);
                }
            }
        }
        epoch.is_some() && table.index_view_epoch() == epoch
    }

    /// A set's members as the comparison reads them against a column of
    /// `column_type`, for a probe of the column to look up, and whether one
    /// is UNKNOWN to every number. None where a probe's equality is not the
    /// comparison's: a member outside the column's domain (a BOOLEAN against
    /// a number, a number against TEXT), or a column of another type
    pub(crate) fn members_for_column(
        column_type: Option<DataType>,
        set: &PredicateSet,
    ) -> Option<(CompactArc<ValueSet>, bool)> {
        // Read from the set's own summary: no member is visited here
        let domain = match column_type {
            // TEXT members a number column reads as numbers
            Some(DataType::Integer | DataType::Float) => kind::INTEGER | kind::FLOAT | kind::TEXT,
            Some(DataType::Text) => kind::TEXT,
            Some(DataType::Boolean) => kind::BOOLEAN,
            Some(DataType::Timestamp) => kind::TIMESTAMP,
            _ => 0,
        };
        if set.kinds() & !domain != 0 {
            return None;
        }
        match column_type {
            Some(DataType::Integer | DataType::Float) if set.has_text() => {
                let (numbers, unknown) = numeric_members(set.members().iter());
                Some((CompactArc::new(numbers.into_iter().collect()), unknown))
            }
            _ => Some((CompactArc::clone(set.members()), false)),
        }
    }

    /// Whether `column` is the table's one-column primary key
    fn is_primary_key(table: &dyn Table, column: &str) -> bool {
        let schema = table.schema();
        let pk = schema.primary_key_indices();
        pk.len() == 1 && schema.columns[pk[0]].name_lower == column
    }

    /// The rows whose `column` is one of `members`, each of the column's
    /// type, read by the primary key, the column's index, or the table's
    /// candidates and else a scan that keeps the members' rows; None when
    /// none of them answers. Under a LIMIT the last two do not, and
    /// `target` rows, when given, are enough
    pub(crate) fn rows_of_members<'v>(
        table: &dyn Table,
        column: &str,
        members: impl Iterator<Item = &'v Value> + 'v,
        limited: bool,
        target: Option<usize>,
    ) -> Result<Option<RowVec>> {
        use crate::storage::expression::logical::ConstBoolExpr;
        let known = members.size_hint().0;
        let mut ids = Vec::with_capacity(target.map_or(known, |target| target.min(known)));
        if Self::is_primary_key(table, column) {
            let mut keys = Self::primary_key_members(members);
            if let Some(target) = target {
                return Self::fetch_members_up_to(table, &mut keys, target, &mut ids).map(Some);
            }
            ids.extend(keys);
        } else if let Some(index) = table.lookup_index_on_column(column) {
            if !Self::index_members(table, index.as_ref(), members, target, &mut ids) {
                return Ok(None);
            }
        } else if !limited {
            // The scan's check reads a member as the column's type, so only
            // members of that type keep the comparison's equality
            let column_type = table.schema().get_column_type(column);
            let members: Vec<Value> = members.cloned().collect();
            if members.iter().any(|v| Some(v.data_type()) != column_type) {
                return Ok(None);
            }
            return Self::in_member_rows(table, column, members, || Ok(None), true);
        } else {
            return Ok(None);
        }
        ids.sort_unstable();
        ids.dedup();
        let needed = target.unwrap_or(usize::MAX);
        Self::fetch_rows_up_to(table, &ids, &ConstBoolExpr::true_expr(), needed).map(Some)
    }

    /// The rows whose `column`, the table's primary key, is none of
    /// `members`, in key order and up to `target`; None for another column
    pub(crate) fn rows_without_members<'v>(
        table: &dyn Table,
        column: &str,
        members: impl Iterator<Item = &'v Value> + 'v,
        target: Option<usize>,
    ) -> Result<Option<RowVec>> {
        use crate::storage::expression::logical::ConstBoolExpr;
        if !Self::is_primary_key(table, column) {
            return Ok(None);
        }
        let excluded: I64Set = Self::primary_key_members(members).collect();
        let ids = table.visible_row_ids_excluding(&excluded, target.unwrap_or(usize::MAX))?;
        Self::fetch_rows_up_to(table, &ids, &ConstBoolExpr::true_expr(), usize::MAX).map(Some)
    }

    fn fetch_rows_up_to(
        table: &dyn Table,
        row_ids: &[i64],
        filter: &dyn crate::storage::expression::Expression,
        needed: usize,
    ) -> Result<RowVec> {
        if needed == usize::MAX || row_ids.len() <= needed {
            return table.fetch_rows_by_ids(row_ids, filter);
        }
        let mut rows = RowVec::with_capacity(needed);
        let mut at = 0;
        let mut batch = needed.clamp(16, FETCH_BATCH_ROWS);
        while at < row_ids.len() && rows.len() < needed {
            let end = at.saturating_add(batch).min(row_ids.len());
            rows.extend(table.fetch_rows_by_ids(&row_ids[at..end], filter)?);
            at = end;
            batch = batch.saturating_mul(2).min(FETCH_BATCH_ROWS);
        }
        Ok(rows)
    }

    /// Takes a set's members, distinct keys, a batch at a time, each batch
    /// sorted and fetched, until `needed` rows are found or the members are
    /// spent: a member with no row counts for nothing, and a small LIMIT
    /// takes a few members of a large set. The ids taken go to `taken`
    fn fetch_members_up_to(
        table: &dyn Table,
        members: &mut impl Iterator<Item = i64>,
        needed: usize,
        taken: &mut Vec<i64>,
    ) -> Result<RowVec> {
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter = ConstBoolExpr::true_expr();
        let mut rows = RowVec::with_capacity(needed.min(FETCH_BATCH_ROWS));
        let mut batch = needed.clamp(16, FETCH_BATCH_ROWS);
        while rows.len() < needed {
            let start = taken.len();
            taken.extend(members.by_ref().take(batch));
            if taken.len() == start {
                break;
            }
            let chunk = &mut taken[start..];
            chunk.sort_unstable();
            rows.extend(table.fetch_rows_by_ids(chunk, &filter)?);
            batch = batch.saturating_mul(2).min(FETCH_BATCH_ROWS);
        }
        Ok(rows)
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
        probe_secondary_index: bool,
    ) -> Result<Option<(Box<dyn QueryResult>, CompactArc<Vec<String>>, bool)>> {
        // Extract IN list info: (column_name, values, is_negated, remaining_predicate)
        // Evaluating the list's literals is the expensive part of the
        // extraction, so decide on the column first: a secondary-index list
        // that will not be probed must not pay for values it never uses.
        let schema = table.schema();
        let pk_indices = schema.primary_key_indices();
        let pk_name = if pk_indices.len() == 1 {
            Some(schema.columns[pk_indices[0]].name_lower.as_str())
        } else {
            None
        };
        // A list under a LIMIT is left to the scan, which stops at its first
        // rows, before its literals are evaluated; the LIMIT is read once,
        // and only for a column whose members could be read here
        let limited = std::cell::OnceCell::new();
        let members_here = |column: &str| {
            Self::in_members_eligible(table, column)
                && !*limited.get_or_init(|| Self::limit_applies(stmt, ctx))
        };
        let (column_name, mut values, is_negated, remaining_predicate) =
            match Self::extract_in_list_info(where_expr, ctx, &|column| {
                probe_secondary_index || pk_name == Some(column) || members_here(column)
            }) {
                Some(info) => info,
                None => return Ok(None),
            };
        let is_pk_column = pk_name == Some(column_name.as_str());

        // Skip if SELECT columns have correlated subqueries (need per-row context)
        // classification is passed from caller to avoid redundant cache lookups
        if classification.select_has_correlated_subqueries {
            return Ok(None);
        }
        // A correlated conjunct beside the IN reads each row's outer columns
        if remaining_predicate
            .as_ref()
            .is_some_and(Self::has_correlated_subqueries)
        {
            return Ok(None);
        }

        // The members as an equality against the column reads each, so the
        // index is asked for the column's own keys; a number against a TEXT
        // column, or a member with no reading, leaves the list to the scan,
        // as the pushed-down IN does. Other column types keep their members
        if !is_negated {
            let column_type = schema.get_column_type(&column_name);
            match column_type {
                Some(DataType::Text) if has_number(values.iter()) => return Ok(None),
                Some(
                    DataType::Integer
                    | DataType::Float
                    | DataType::Text
                    | DataType::Timestamp
                    | DataType::Boolean,
                ) => {
                    let coerced: Option<Vec<Value>> = values
                        .into_iter()
                        .map(|member| {
                            crate::executor::pushdown::coerce_constant(column_type, member)
                        })
                        .collect();
                    let Some(coerced) = coerced else {
                        return Ok(None);
                    };
                    values = coerced;
                }
                _ => {}
            }
        }

        // A secondary index probe collects every matching row id for every
        // value before fetching, while the pushed-down range scan streams
        // and stops at LIMIT; so it is only taken when a memory filter is
        // needed anyway. Primary key values are the row ids themselves. A
        // table that keeps rows outside its index reads the members by its
        // candidates when no LIMIT applies here
        let index = if !is_pk_column {
            match table.lookup_index_on_column(&column_name) {
                Some(idx) => Some(idx),
                None if !is_negated && members_here(&column_name) => None,
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
                let output_columns = CompactArc::new(self.get_output_column_names(
                    &stmt.columns,
                    all_columns,
                    table_alias,
                ));
                let result = ExecutorResult::with_arc_columns(
                    CompactArc::clone(&output_columns),
                    RowVec::new(),
                );
                return Ok(Some((Box::new(result), output_columns, false)));
            }
        }

        // For NOT IN, we can't easily use the index (would need to exclude rows)
        // Fall back to normal scan
        if is_negated {
            return Ok(None);
        }

        // A table that keeps rows outside its index: the members' rows by
        // its candidates, else the normal scan of the literals
        let member_rows = if !is_pk_column && index.is_none() {
            match Self::in_member_rows(
                table,
                &column_name,
                std::mem::take(&mut values),
                || self.remaining_row_filter(remaining_predicate.as_ref(), all_columns, ctx),
                false,
            )? {
                Some(rows) => Some(rows),
                None => return Ok(None),
            }
        } else {
            None
        };

        // Collect row_ids: either from PK (direct) or from index probe
        // Pre-allocate based on expected size to avoid reallocations
        let mut all_row_ids = Vec::with_capacity(values.len());
        if is_pk_column {
            // A member that is no key matches no row of an IN
            all_row_ids.extend(Self::primary_key_members(values.iter()));
        } else if let Some(ref idx) = index {
            if !Self::index_members(table, idx.as_ref(), values.iter(), None, &mut all_row_ids) {
                return Ok(None);
            }
        }

        // Sort row_ids for better cache locality during version lookup
        // and deduplicate. More efficient than HashSet when sorted output is needed anyway.
        all_row_ids.sort_unstable();
        all_row_ids.dedup();

        // EARLY LIMIT OPTIMIZATION: When there's no ORDER BY and no remaining predicate,
        // we can apply LIMIT early to avoid fetching unnecessary rows
        // LIMIT and OFFSET can be applied here only when nothing runs between
        // this path and the result: no ORDER BY, and no DISTINCT that would
        // otherwise see already-truncated rows.
        let limit_here = stmt.order_by.is_empty() && !stmt.distinct && stmt.distinct_on.is_empty();
        let (early_limit_applied, early_limit, early_offset) =
            if limit_here && remaining_predicate.is_none() && member_rows.is_none() {
                let counts = row_counts(stmt, ctx)?;
                let (offset, limit) = (counts.offset, counts.limit.unwrap_or(usize::MAX));

                // The rows are trimmed after the fetch: a member with no
                // row, or with a row not visible, counts for nothing
                (true, limit, offset)
            } else {
                (false, usize::MAX, 0)
            };

        // Create a filter expression for remaining predicate + visibility
        use crate::storage::expression::logical::ConstBoolExpr;
        let filter: Box<dyn crate::storage::expression::Expression> =
            Box::new(ConstBoolExpr::true_expr());

        // Fetch rows by row_ids - returns RowVec directly
        let needed = if early_limit_applied {
            early_offset.saturating_add(early_limit)
        } else {
            usize::MAX
        };
        let member_read = member_rows.is_some();
        let mut rows = match member_rows {
            Some(rows) => rows,
            None => Self::fetch_rows_up_to(table, &all_row_ids, filter.as_ref(), needed)?,
        };

        // Apply remaining predicate if any; the members' read applied it
        if let Some(remaining) = remaining_predicate.as_ref().filter(|_| !member_read) {
            // Compile the filter using RowFilter (with params for $1 etc.)
            let columns_slice: Vec<String> = all_columns.to_vec();
            let row_filter = RowFilter::new(remaining, &columns_slice)?.with_context(ctx);

            // Filter rows (retain works on (i64, Row) tuples via Deref)
            row_filter.retain_checked(&mut rows)?;
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
        } else if limit_here {
            let counts = row_counts(stmt, ctx)?;
            let (offset, limit) = (counts.offset, counts.limit.unwrap_or(usize::MAX));

            if offset > 0 || limit < usize::MAX {
                rows = rows.into_iter().skip(offset).take(limit).collect();
            }
        }

        // Project rows according to SELECT expressions
        // Use schema cache for lowercase column names
        let all_columns_lower = schema.column_names_lower_arc();
        let projected_rows = self.project_rows_with_alias(
            &stmt.columns,
            rows,
            all_columns,
            Some(&all_columns_lower),
            ctx,
            table_alias,
        )?;
        let output_columns =
            CompactArc::new(self.get_output_column_names(&stmt.columns, all_columns, table_alias));
        let result =
            ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), projected_rows);
        // Without ORDER BY the LIMIT and OFFSET were applied above; the
        // caller must not apply them a second time.
        let limit_applied = limit_here && (stmt.limit.is_some() || stmt.offset.is_some());
        Ok(Some((Box::new(result), output_columns, limit_applied)))
    }

    /// Extract IN list literal information from a WHERE clause.
    /// Returns (column_name, values, is_negated, remaining_predicate)
    pub(crate) fn extract_in_list_info(
        expr: &Expression,
        ctx: &ExecutionContext,
        accept: &dyn Fn(&str) -> bool,
    ) -> Option<(String, Vec<Value>, bool, Option<Expression>)> {
        match expr {
            // Direct IN list: column IN (v1, v2, ...)
            Expression::In(in_expr) => {
                // Get the column name from the left side (lowercase for case-insensitive match)
                let column_name = match in_expr.left.as_ref() {
                    Expression::Identifier(id) => id.value_lower.to_string(),
                    Expression::QualifiedIdentifier(qid) => qid.name.value_lower.to_string(),
                    _ => return None, // Can't optimize complex left expressions
                };
                // Decided before the values are evaluated, so a list the
                // caller would not probe costs nothing here.
                if !accept(&column_name) {
                    return None;
                }

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
                // The IN list may sit under further ANDs; whatever those
                // left over is kept together with this level's sibling.
                let keep = |rest: Option<Expression>, sibling: &Expression| -> Expression {
                    match rest {
                        Some(rest) => Expression::Infix(InfixExpression {
                            token: infix.token.clone(),
                            left: Box::new(rest),
                            operator: infix.operator.clone(),
                            op_type: infix.op_type,
                            right: Box::new(sibling.clone()),
                        }),
                        None => sibling.clone(),
                    }
                };
                if let Some((col, vals, neg, rest)) =
                    Self::extract_in_list_info(&infix.left, ctx, accept)
                {
                    return Some((col, vals, neg, Some(keep(rest, &infix.right))));
                }
                if let Some((col, vals, neg, rest)) =
                    Self::extract_in_list_info(&infix.right, ctx, accept)
                {
                    return Some((col, vals, neg, Some(keep(rest, &infix.left))));
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
            // A column reference does not compile over no columns
            values.push(super::expression::evaluate_once(expr, Some(ctx)).ok()?);
        }
        Some(values)
    }

    /// The VEC_DISTANCE call a nearest-vectors search answers: the one
    /// ascending ORDER BY key of a statement with a LIMIT, as the ORDER BY
    /// binding chose it (the SELECT item at `item`, else the key as
    /// written), with `<=>` read as VEC_DISTANCE_L2
    pub(crate) fn vector_distance_order(
        stmt: &SelectStatement,
        item: Option<usize>,
    ) -> Option<std::borrow::Cow<'_, FunctionCall>> {
        use std::borrow::Cow;
        // Must have exactly 1 ORDER BY + LIMIT, ascending order
        let [order_by] = stmt.order_by.as_slice() else {
            return None;
        };
        stmt.limit.as_ref()?;
        if !order_by.ascending || crate::core::value::nulls_first(true, order_by.nulls_first) {
            return None; // Vector search returns closest first, NULLs last
        }
        // <=> is VEC_DISTANCE_L2
        let l2 = |infix: &InfixExpression| FunctionCall {
            token: infix.token.clone(),
            function: "VEC_DISTANCE_L2".into(),
            arguments: vec![(*infix.left).clone(), (*infix.right).clone()],
            is_distinct: false,
            order_by: Vec::new(),
            filter: None,
        };
        let key = match item {
            Some(item) => match stmt.columns.get(item)? {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                expr => expr,
            },
            None => &order_by.expression,
        };
        let func_call = match key {
            Expression::FunctionCall(fc) => Cow::Borrowed(fc.as_ref()),
            Expression::Infix(infix) if infix.op_type == InfixOperator::VectorDistance => {
                Cow::Owned(l2(infix))
            }
            _ => return None,
        };
        let is_vec_distance = ["VEC_DISTANCE_L2", "VEC_DISTANCE_COSINE", "VEC_DISTANCE_IP"]
            .iter()
            .any(|name| func_call.function.eq_ignore_ascii_case(name));
        (is_vec_distance && func_call.arguments.len() == 2).then_some(func_call)
    }
}
