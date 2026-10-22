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

//! Index Nested Loop Join Operator.
//!
//! This operator implements index nested loop join with O(N * log M) complexity
//! by using indexes on the inner table for lookups. It's optimal when:
//! - The inner (right) table has an index on the join key column
//! - The outer (left) table is small or has good selectivity
//!
//! For each row in the outer table, we use the index/PK to find matching rows
//! in the inner table, avoiding a full scan of the inner table.

use std::sync::Arc;

use crate::common::{CompactVec, I64Map, I64Set};
use crate::core::predicate::{kind, sql_eq, KeySide, TextReading};
use crate::core::value::NULL_VALUE;
use crate::core::ValueMap;
use crate::core::{Result, Row, RowVec, Value};
use crate::executor::expression::{JoinFilter, SubqueryExecutor};
use crate::executor::operator::{ColumnInfo, Operator, RowRef};
use crate::storage::expression::{ConstBoolExpr, Expression};
use crate::storage::traits::{CappedEqual, Index, ProbeScratch, Table};
use smallvec::SmallVec;

use super::hash_join::JoinType;

/// Err with the first error the residual filter read as false, once
/// `matches` set `failed`
#[inline]
fn residual_failed(filter: &Option<JoinFilter>, failed: bool) -> Result<()> {
    match filter {
        Some(filter) if failed => Err(filter.failure()),
        _ => Ok(()),
    }
}

/// The residual over the pair in the statement's order: inner first when
/// the outer side is the right one
#[inline]
fn pair_matches(
    filter: &JoinFilter,
    outer_on_right: bool,
    outer: &Row,
    inner: &Row,
    failed: &mut bool,
    blocks: Option<&dyn SubqueryExecutor>,
) -> bool {
    if outer_on_right {
        filter.matches_with(inner, outer, failed, blocks)
    } else {
        filter.matches_with(outer, inner, failed, blocks)
    }
}

/// Index Nested Loop Join lookup strategy.
/// Determines how to find matching rows in the inner (right) table.
#[derive(Clone)]
pub enum IndexLookupStrategy {
    /// Use a secondary index for lookups (index.get_row_ids_equal)
    SecondaryIndex(Arc<dyn Index>),
    /// Use primary key lookup (direct row_id = value)
    /// In stoolap, PRIMARY KEY INTEGER values ARE the row_ids
    PrimaryKey,
    /// Ask the inner table for the ids of one key, at most
    /// `EQUALITY_CANDIDATE_CAP` of them, when it keeps rows its index does
    /// not cover: the table decides per probe, and a probe it cannot answer
    /// ends the join, which the executor then runs another way
    TableEquality { column: String },
}

/// The most row ids one probe takes from the inner table's index
pub const EQUALITY_CANDIDATE_CAP: usize = 1024;

/// The side of a cross the inner key column is on; None for a type no key
/// meets across types
fn inner_key_side(
    table: &dyn Table,
    strategy: &IndexLookupStrategy,
    idx: Option<usize>,
) -> Option<KeySide> {
    if matches!(strategy, IndexLookupStrategy::PrimaryKey) {
        return Some(KeySide::Read(TextReading::Number));
    }
    KeySide::of_kinds(kind::of_type(table.schema().columns.get(idx?)?.data_type))
}

/// An outer key the comparison reads against the inner column across types:
/// a TEXT read as the column's type, or a key that reads the column's TEXT
#[inline]
pub(crate) fn crosses_inner_key(inner: Option<KeySide>, key: &Value) -> bool {
    inner.is_some_and(|side| side.crosses(key))
}

/// A number as the readings of a TEXT column hold it: an integral FLOAT that fits i64 as
/// that INTEGER, so equal numbers meet; None for NaN and any non-number
pub(crate) fn number_key(number: &Value) -> Option<Value> {
    match number {
        Value::Integer(_) => Some(number.clone()),
        Value::Float(f) if f.is_nan() => None,
        Value::Float(f) => Some(match crate::executor::Executor::lossless_float_key(*f) {
            Some(i) => Value::Integer(i),
            None => number.clone(),
        }),
        _ => None,
    }
}

/// A key as the readings of a TEXT column hold it: a number by
/// `number_key`, a BOOLEAN or TIMESTAMP as it is; None for any other value
pub(crate) fn reading_key(key: &Value) -> Option<Value> {
    match key {
        Value::Boolean(_) | Value::Timestamp(_) => Some(key.clone()),
        _ => number_key(key),
    }
}

/// The inner rows of a TEXT key column by the value each TEXT reads as, one
/// map per reading, each taken by one scan when a key of that type first
/// meets the column
#[derive(Default)]
struct TextReads([Option<ValueMap<SmallVec<[i64; 2]>>>; 3]);

impl TextReads {
    fn scan(
        table: &dyn Table,
        column: usize,
        reading: TextReading,
    ) -> Result<ValueMap<SmallVec<[i64; 2]>>> {
        let width = table.schema().columns.len();
        let columns: Vec<usize> = (0..width).collect();
        let mut needed = vec![false; width];
        needed[column] = true;
        let mut scanner = table.scan_needed(&columns, &needed, None)?;
        let mut by_read: ValueMap<SmallVec<[i64; 2]>> = ValueMap::default();
        while scanner.next() {
            let row_id = scanner.current_row_id();
            if let Some(Value::Text(text)) = scanner.row().get(column) {
                if let Some(read) = reading.read(text).as_ref().and_then(reading_key) {
                    by_read.entry(read).or_default().push(row_id);
                }
            }
        }
        if let Some(error) = scanner.err() {
            return Err(error.clone());
        }
        scanner.close()?;
        Ok(by_read)
    }

    /// The ids of the inner rows whose TEXT key reads as `key`
    fn row_ids(&mut self, table: &dyn Table, column: usize, key: &Value) -> Result<&[i64]> {
        let Some(reading) = TextReading::of(key) else {
            return Ok(&[]);
        };
        let slot = &mut self.0[reading as usize];
        if slot.is_none() {
            *slot = Some(Self::scan(table, column, reading)?);
        }
        Ok(reading_key(key)
            .and_then(|read| slot.as_ref()?.get(&read))
            .map_or(&[], |ids| ids.as_slice()))
    }
}

/// The inner rows' ids by key, read through the table as the statement
/// sees it: the probes' answer from the first probe of the shared index
/// that could not be checked against what the statement sees
struct ViewKeys(ValueMap<SmallVec<[i64; 2]>>);

impl ViewKeys {
    fn scan(table: &dyn Table, column: usize) -> Result<Self> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::join_view_keys_scanned();
        let width = table.schema().columns.len();
        let columns: Vec<usize> = (0..width).collect();
        let mut needed = vec![false; width];
        needed[column] = true;
        let mut scanner = table.scan_needed(&columns, &needed, None)?;
        let mut by_key: ValueMap<SmallVec<[i64; 2]>> = ValueMap::default();
        while scanner.next() {
            let row_id = scanner.current_row_id();
            if let Some(key) = scanner.row().get(column).filter(|key| !key.is_null()) {
                by_key.entry(key.clone()).or_default().push(row_id);
            }
        }
        if let Some(error) = scanner.err() {
            return Err(error.clone());
        }
        scanner.close()?;
        Ok(Self(by_key))
    }

    fn row_ids(&self, key: &Value) -> &[i64] {
        self.0.get(key).map_or(&[], |ids| ids.as_slice())
    }
}

/// Specifies which side (outer or inner) a projected column comes from.
/// A projected value out of a row that is being consumed. A source used
/// more than once must be cloned every time; moving it would hand the
/// second use a NULL.
#[inline]
fn take_or_clone(row: &mut Row, idx: usize, repeats: bool) -> Value {
    if repeats {
        row.as_slice()[idx].clone()
    } else {
        row.take_or_clone(idx)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ColumnSource {
    /// Column from outer row at given index
    Outer(usize),
    /// Column from inner row at given index
    Inner(usize),
}

/// Projection configuration for fused projection during row combination.
/// When set, the operator creates projected rows directly instead of full combined rows.
/// Columns are stored in SELECT order (not outer-first/inner-second).
#[derive(Clone)]
pub struct JoinProjection {
    /// Columns to extract, in SELECT order (preserves original column ordering)
    pub columns: Vec<ColumnSource>,
    /// A source appears more than once, so values are cloned, not moved.
    pub repeats_source: bool,
}

/// Index Nested Loop Join Operator.
///
/// For each row in the outer input, looks up matching rows in the inner table
/// using an index. This avoids full table scans of the inner table.
pub struct IndexNestedLoopJoinOperator<'a> {
    // Outer input operator
    outer: Box<dyn Operator>,

    // Inner table (accessed via index)
    inner_table: Box<dyn Table>,

    // Join configuration
    join_type: JoinType,
    outer_key_idx: usize,
    lookup_strategy: IndexLookupStrategy,
    residual_filter: Option<JoinFilter>,
    // The inner key column, checked on each fetched row: inside a
    // transaction an index entry may be older than the row it points at
    inner_key_idx: Option<usize>,
    // The side of a cross that column is on, for outer keys no probe answers
    inner_side: Option<KeySide>,

    // Output schema
    schema: Vec<ColumnInfo>,
    inner_col_count: usize,

    // Optional projection pushdown: create projected rows during combine
    projection: Option<JoinProjection>,
    // The outer side is the statement's right side: rows and the residual
    // pair read inner first
    outer_on_right: bool,
    // The statement's subquery blocks, as the residual's site calls them
    blocks: Option<&'a dyn SubqueryExecutor>,
    // Set when the residual filter read an error as false
    condition_failed: bool,

    // Current state
    current_outer_row: Option<Row>,
    // Optimization: Store (id, row) to verify specific inner rows if needed
    current_inner_rows: RowVec,
    current_inner_idx: usize,
    outer_had_match: bool,

    // Optimization: Reusable buffer for row IDs to avoid allocation per outer row
    row_id_buffer: Vec<i64>,

    // Optimization: Reusable row buffer to avoid allocation per join output
    row_buffer: Row,

    // Expression for fetching rows (always true - we apply residual separately)
    true_expr: ConstBoolExpr,
    // A predicate on inner columns the fetch applies instead of the residual
    inner_filter: Option<Box<dyn Expression>>,

    // State tracking
    opened: bool,
    outer_exhausted: bool,
    // Outer rows pulled so far; the executor uses it to see a full chunk
    outer_rows_seen: usize,
    // Set when the inner table could not answer a probe; the rows joined
    // so far are then no answer either
    needs_fallback: bool,
    // The outer row the chunk ended on when it left with its last match,
    // kept for the executor's continuation
    ended_on: Option<Row>,
    // What a table probe keeps between probes
    probe_scratch: ProbeScratch,
    // This transaction's uncommitted inner rows, whose keys the shared
    // index may not hold yet; taken once at open
    local_rows: Option<LocalRows>,
    // The inner TEXT keys read as other types, taken when such a key first meets them
    text_reads: TextReads,
    // The inner keys as the statement sees them, once a probe was not
    view_keys: Option<ViewKeys>,
}

/// The transaction's uncommitted rows of the inner table: every row it
/// wrote, and the live ones by their value in the key column. A probe
/// drops the written rows from what the shared index names, since the
/// index may hold them under a key the transaction changed or another
/// transaction moved, and adds the live ones under the probed key.
struct LocalRows {
    written: I64Set,
    by_key: ValueMap<SmallVec<[i64; 2]>>,
}

impl LocalRows {
    /// Taken when the table has local changes and the strategy probes a
    /// shared index; a primary key names its row directly and the fetch
    /// sees the local version
    fn of(
        table: &dyn Table,
        strategy: &IndexLookupStrategy,
        column: Option<usize>,
    ) -> Option<Self> {
        let column = column?;
        if !matches!(strategy, IndexLookupStrategy::SecondaryIndex(_)) || !table.has_local_changes()
        {
            return None;
        }
        let mut pairs = Vec::new();
        let mut written = I64Set::new();
        table.local_column_values(column, &mut pairs, &mut written);
        let mut by_key: ValueMap<SmallVec<[i64; 2]>> = ValueMap::default();
        for (value, row_id) in pairs {
            by_key.entry(value).or_default().push(row_id);
        }
        Some(Self { written, by_key })
    }

    /// Replaces the index's answer for the transaction's rows with the
    /// transaction's own
    fn apply(&self, key: &Value, buffer: &mut Vec<i64>) {
        buffer.retain(|id| !self.written.contains(*id));
        if let Some(ids) = self.by_key.get(key) {
            buffer.extend_from_slice(ids);
        }
    }
}

impl<'a> IndexNestedLoopJoinOperator<'a> {
    /// Create a new index nested loop join operator.
    ///
    /// # Arguments
    /// * `outer` - Outer input operator
    /// * `inner_table` - Inner table to lookup from
    /// * `inner_schema` - Schema of the inner table
    /// * `join_type` - Type of join (INNER or LEFT)
    /// * `outer_key_idx` - Column index of the join key in outer rows
    /// * `lookup_strategy` - How to find matching inner rows
    /// * `residual_filter` - Optional additional filter after key match
    pub fn new(
        outer: Box<dyn Operator>,
        inner_table: Box<dyn Table>,
        inner_schema: Vec<ColumnInfo>,
        join_type: JoinType,
        outer_key_idx: usize,
        lookup_strategy: IndexLookupStrategy,
        residual_filter: Option<JoinFilter>,
    ) -> Self {
        // Build combined schema
        let mut schema = Vec::new();
        schema.extend(outer.schema().iter().cloned());
        schema.extend(inner_schema.iter().cloned());

        let inner_col_count = inner_schema.len();

        // Pre-allocate row buffer for typical join output size
        let outer_col_count = outer.schema().len();
        let total_cols = outer_col_count + inner_col_count;
        // A table probe copies up to the cap under the index lock, so the
        // buffer holds the cap before the first probe and grows on none
        let id_capacity = match lookup_strategy {
            IndexLookupStrategy::TableEquality { .. } => EQUALITY_CANDIDATE_CAP,
            _ => 16,
        };
        let inner_side = inner_key_side(&*inner_table, &lookup_strategy, None);

        Self {
            outer,
            inner_table,
            join_type,
            outer_key_idx,
            lookup_strategy,
            residual_filter,
            inner_key_idx: None,
            inner_side,
            schema,
            inner_col_count,
            projection: None,
            outer_on_right: false,
            blocks: None,
            condition_failed: false,
            current_outer_row: None,
            current_inner_rows: RowVec::new(),
            current_inner_idx: 0,
            outer_had_match: false,
            row_id_buffer: Vec::with_capacity(id_capacity),
            // Pre-allocate row buffer to avoid per-row allocation
            row_buffer: Row::with_capacity(total_cols),
            true_expr: ConstBoolExpr::true_expr(),
            inner_filter: None,
            opened: false,
            outer_exhausted: false,
            outer_rows_seen: 0,
            needs_fallback: false,
            ended_on: None,
            probe_scratch: ProbeScratch::default(),
            local_rows: None,
            text_reads: TextReads::default(),
            view_keys: None,
        }
    }

    /// Set projection pushdown configuration.
    ///
    /// When set, the operator creates projected rows directly during the combine step
    /// instead of creating full combined rows. This avoids creating large intermediate
    /// rows that would be immediately projected down to fewer columns.
    ///
    /// # Arguments
    /// * `columns` - Column sources in SELECT order (preserves original column ordering)
    /// * `projected_schema` - Schema for the projected output
    pub fn with_projection(
        mut self,
        columns: Vec<ColumnSource>,
        projected_schema: Vec<ColumnInfo>,
    ) -> Self {
        let repeats_source = columns
            .iter()
            .enumerate()
            .any(|(i, c)| columns[..i].contains(c));
        self.projection = Some(JoinProjection {
            columns,
            repeats_source,
        });
        self.schema = projected_schema;
        self
    }

    /// The outer side is the statement's right side, so each joined row and
    /// the residual's pair read the inner side first
    pub fn with_outer_on_right(mut self) -> Self {
        if self.projection.is_none() {
            let outer_len = self.schema.len() - self.inner_col_count;
            self.schema.rotate_left(outer_len);
        }
        self.outer_on_right = true;
        self
    }

    /// The statement's subquery blocks the residual calls
    pub fn with_blocks(mut self, blocks: Option<&'a dyn SubqueryExecutor>) -> Self {
        self.blocks = blocks;
        self
    }

    /// A filter the inner fetch applies to each candidate row, for a
    /// predicate on inner columns only
    pub fn with_inner_filter(mut self, filter: Box<dyn Expression>) -> Self {
        self.inner_filter = Some(filter);
        self
    }

    /// The inner key column, so a row an index entry points at is kept
    /// only when its own key equals the outer one
    pub fn with_inner_key(mut self, inner_key_idx: Option<usize>) -> Self {
        self.inner_key_idx = inner_key_idx;
        self.inner_side = inner_key_side(&*self.inner_table, &self.lookup_strategy, inner_key_idx);
        self
    }

    /// Create a NULL row for the inner side.
    /// Always creates full-width row so combine() can use consistent indexing.
    #[inline]
    fn null_inner_row(&self) -> Row {
        let null_values: Vec<Value> = (0..self.inner_col_count).map(|_| NULL_VALUE).collect();
        Row::from_values(null_values)
    }

    /// Combine outer and inner rows into reusable buffer (owns both).
    /// OPTIMIZATION: Moves both outer and inner values without cloning.
    /// Use when outer row is no longer needed (last match for this outer).
    ///
    /// # Safety
    /// Uses `get_unchecked` for performance - projection indices are validated
    /// at query planning time against the table schemas.
    #[inline]
    fn combine_owned_into_buffer(&mut self, mut outer: Row, mut inner: Row) {
        match &self.projection {
            Some(proj) => {
                // Fused projection into buffer - columns are in SELECT order
                self.row_buffer.clear();
                self.row_buffer.reserve(proj.columns.len());

                // Only the projected values leave the two rows; converting
                // both rows to Vec<Value> first cloned every column of a
                // shared row for the one or two that were kept.
                let repeats = proj.repeats_source;
                for col_source in &proj.columns {
                    match col_source {
                        ColumnSource::Outer(idx) => {
                            self.row_buffer
                                .push(take_or_clone(&mut outer, *idx, repeats));
                        }
                        ColumnSource::Inner(idx) => {
                            self.row_buffer
                                .push(take_or_clone(&mut inner, *idx, repeats));
                        }
                    }
                }
            }
            None if self.outer_on_right => self.row_buffer.combine_into_owned(inner, outer),
            None => {
                // Combine both rows into buffer
                self.row_buffer.combine_into_owned(outer, inner);
            }
        }
    }

    /// Take the combined row from the buffer, preserving buffer capacity.
    /// OPTIMIZATION: Uses take_and_clear to keep buffer capacity for next iteration,
    /// avoiding reallocation in combine_into_owned.
    #[inline]
    fn take_from_buffer(&mut self) -> Row {
        self.row_buffer.take_and_clear()
    }

    /// Create combined row directly (for cases where buffer can't be used).
    /// When projection is set, creates projected row directly.
    ///
    /// # Safety
    /// Uses `get_unchecked` for performance - projection indices are validated
    /// at query planning time against the table schemas.
    #[inline]
    fn create_combined_row(&self, outer: &Row, mut inner: Row) -> Row {
        match &self.projection {
            Some(proj) => {
                // Fused projection: create only the columns we need, in SELECT order
                let mut values: CompactVec<Value> = CompactVec::with_capacity(proj.columns.len());

                // We need to move from inner but clone from outer (we don't own outer)
                let outer_slice = outer.as_slice();

                for col_source in &proj.columns {
                    match col_source {
                        ColumnSource::Outer(idx) => {
                            // SAFETY: indices validated at planning time against outer table schema.
                            values.push(unsafe { outer_slice.get_unchecked(*idx) }.clone());
                        }
                        ColumnSource::Inner(idx) => {
                            values.push(take_or_clone(&mut inner, *idx, proj.repeats_source));
                        }
                    }
                }

                Row::from_compact_vec(values)
            }
            None if self.outer_on_right => Row::from_combined_move_clone(inner, outer),
            None => Row::from_combined_clone_move(outer, inner),
        }
    }

    pub fn outer_rows_seen(&self) -> usize {
        self.outer_rows_seen
    }

    /// The outer row the join is on, or ended on
    pub fn last_outer_row(&self) -> Option<&Row> {
        self.current_outer_row.as_ref().or(self.ended_on.as_ref())
    }

    /// Whether the inner table stopped answering probes, so the join ended
    /// early and what it produced is to be discarded
    pub fn needs_fallback(&self) -> bool {
        self.needs_fallback
    }

    /// Look up matching inner rows for the current outer row.
    /// Uses internal buffers to avoid allocations.
    fn lookup_inner_rows(&mut self, key_value: &Value) -> Result<()> {
        // Clear buffers for reuse
        self.row_id_buffer.clear();
        self.current_inner_rows.clear();
        if crosses_inner_key(self.inner_side, key_value) {
            return self.lookup_across(key_value);
        }

        // Get row IDs based on lookup strategy into buffer
        let checked = match &self.lookup_strategy {
            IndexLookupStrategy::SecondaryIndex(_) | IndexLookupStrategy::TableEquality { .. }
                if self.view_keys.is_some() =>
            {
                false
            }
            IndexLookupStrategy::SecondaryIndex(index) => {
                let epoch = self.inner_table.committed_candidate_epoch();
                index.get_row_ids_equal_into(
                    std::slice::from_ref(key_value),
                    &mut self.row_id_buffer,
                );
                epoch.is_some()
                    && self.inner_table.add_removed_equal(
                        &**index,
                        key_value,
                        &mut self.row_id_buffer,
                    )
                    && self.inner_table.committed_candidate_epoch() == epoch
            }
            IndexLookupStrategy::TableEquality { column } => {
                let found = self.inner_table.equality_candidates(
                    column,
                    key_value,
                    EQUALITY_CANDIDATE_CAP,
                    &mut self.row_id_buffer,
                    &mut self.probe_scratch,
                )?;
                found == Some(CappedEqual::Copied)
            }
            IndexLookupStrategy::PrimaryKey => {
                match key_value {
                    Value::Integer(id) => self.row_id_buffer.push(*id),
                    // Only a losslessly integral float can equal a PK;
                    // truncation would join 5.5 against row 5
                    Value::Float(f) => {
                        if let Some(id) = crate::executor::Executor::lossless_float_key(*f) {
                            self.row_id_buffer.push(id);
                        }
                    }
                    _ => {} // Non-numeric PK values can't match
                }
                true
            }
        };

        // A probe not checked against what the statement sees is answered by
        // the keys read through the table, for this outer row and the rest
        if !checked {
            self.row_id_buffer.clear();
            let Some(idx) = self.inner_key_idx else {
                self.needs_fallback = true;
                return Ok(());
            };
            if self.view_keys.is_none() {
                self.view_keys = Some(ViewKeys::scan(&*self.inner_table, idx)?);
            }
            if let Some(keys) = &self.view_keys {
                self.row_id_buffer
                    .extend_from_slice(keys.row_ids(key_value));
            }
        }

        // The transaction's own rows answer for themselves
        if let Some(local) = &self.local_rows {
            local.apply(key_value, &mut self.row_id_buffer);
        }

        if self.row_id_buffer.is_empty() {
            return Ok(());
        }

        // Fetch matching rows directly into inner_rows buffer
        let filter: &dyn Expression = match &self.inner_filter {
            Some(filter) => filter.as_ref(),
            None => &self.true_expr,
        };
        self.inner_table.fetch_rows_by_ids_into(
            &self.row_id_buffer,
            filter,
            &mut self.current_inner_rows,
        )?;

        // An index entry may point at a row whose key has since changed
        // inside the transaction, so the row's own key decides
        if let (
            IndexLookupStrategy::SecondaryIndex(_) | IndexLookupStrategy::TableEquality { .. },
            Some(idx),
        ) = (&self.lookup_strategy, self.inner_key_idx)
        {
            self.current_inner_rows
                .retain(|(_, row)| row.get(idx) == Some(key_value));
        }
        Ok(())
    }

    /// The inner rows for a key that meets the inner column across types: a
    /// TEXT key probes as the value it reads as, any other key takes the
    /// TEXT rows that read as it
    fn lookup_across(&mut self, key_value: &Value) -> Result<()> {
        if let (Some(KeySide::Read(reading)), Value::Text(text)) = (self.inner_side, key_value) {
            return match reading.read(text) {
                Some(read) => self.lookup_inner_rows(&read),
                None => Ok(()),
            };
        }
        let Some(idx) = self.inner_key_idx else {
            return Ok(());
        };
        let ids = self
            .text_reads
            .row_ids(&*self.inner_table, idx, key_value)?;
        self.row_id_buffer.extend_from_slice(ids);
        if self.row_id_buffer.is_empty() {
            return Ok(());
        }
        let filter: &dyn Expression = match &self.inner_filter {
            Some(filter) => filter.as_ref(),
            None => &self.true_expr,
        };
        self.inner_table.fetch_rows_by_ids_into(
            &self.row_id_buffer,
            filter,
            &mut self.current_inner_rows,
        )?;
        self.current_inner_rows
            .retain(|(_, row)| row.get(idx).and_then(|key| sql_eq(key, key_value)) == Some(true));
        Ok(())
    }

    /// Advance to the next outer row and lookup matching inner rows.
    fn advance_outer(&mut self) -> Result<bool> {
        match self.outer.next()? {
            Some(row_ref) => {
                let outer_row = row_ref.into_owned();
                self.outer_rows_seen += 1;
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::index_join_outer_row();

                // Get the join key value from the outer row
                let key_value = match outer_row.get(self.outer_key_idx) {
                    Some(v) if !v.is_null() => v.clone(),
                    _ => {
                        // NULL key - no match possible (NULL != NULL in SQL)
                        self.current_outer_row = Some(outer_row);
                        self.current_inner_rows.clear();
                        self.current_inner_idx = 0;
                        self.outer_had_match = false;
                        return Ok(true);
                    }
                };

                // Lookup matching inner rows
                self.lookup_inner_rows(&key_value)?;
                if self.needs_fallback {
                    self.current_outer_row = Some(outer_row);
                    self.outer_exhausted = true;
                    return Ok(false);
                }

                self.current_outer_row = Some(outer_row);
                // self.current_inner_rows is already populated by lookup_inner_rows
                self.current_inner_idx = 0;
                self.outer_had_match = false;
                Ok(true)
            }
            None => {
                self.outer_exhausted = true;
                Ok(false)
            }
        }
    }
}

impl Operator for IndexNestedLoopJoinOperator<'_> {
    fn open(&mut self) -> Result<()> {
        self.outer.open()?;
        self.local_rows = LocalRows::of(
            &*self.inner_table,
            &self.lookup_strategy,
            self.inner_key_idx,
        );
        // An error a join on this thread left behind is not this one's
        if let Some(filter) = &self.residual_filter {
            let _ = filter.take_error();
        }

        // Get first outer row
        self.advance_outer()?;

        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Err(crate::core::Error::internal(
                "IndexNestedLoopJoinOperator::next called before open",
            ));
        }

        let is_left_outer = matches!(self.join_type, JoinType::Left | JoinType::Full);

        loop {
            // Check if outer is exhausted
            if self.outer_exhausted {
                return Ok(None);
            }

            // Ensure we have an outer row
            if self.current_outer_row.is_none() && !self.advance_outer()? {
                return Ok(None);
            }

            // Try to find a match in current inner rows
            let inner_len = self.current_inner_rows.len();
            while self.current_inner_idx < inner_len {
                let inner_idx = self.current_inner_idx;
                self.current_inner_idx += 1;

                // Check filter with borrowed references first
                let passes_filter = {
                    let outer_row = self.current_outer_row.as_ref().unwrap();
                    // SAFETY: inner_idx < inner_len checked by while condition
                    let inner_entry = unsafe { self.current_inner_rows.get_unchecked(inner_idx) };
                    if let Some(ref filter) = self.residual_filter {
                        pair_matches(
                            filter,
                            self.outer_on_right,
                            outer_row,
                            &inner_entry.1,
                            &mut self.condition_failed,
                            self.blocks,
                        )
                    } else {
                        true
                    }
                };

                if passes_filter {
                    residual_failed(&self.residual_filter, self.condition_failed)?;
                    self.outer_had_match = true;
                    // SAFETY: inner_idx < inner_len, bounds already checked
                    let inner_row = std::mem::take(unsafe {
                        &mut self.current_inner_rows.get_unchecked_mut(inner_idx).1
                    });

                    // OPTIMIZATION: Check if this is the last inner row to check
                    // If so, we can take ownership of outer_row and move its values
                    let is_last_inner = self.current_inner_idx >= inner_len;
                    if is_last_inner {
                        // No more inner rows - take ownership of outer and move values
                        let outer_row = self.current_outer_row.take().unwrap();
                        // Pre-advance to next outer for next call; a chunk that
                        // ends here keeps the row it ended on
                        if !self.advance_outer()? {
                            self.ended_on = Some(outer_row.clone());
                        }
                        self.combine_owned_into_buffer(outer_row, inner_row);
                        return Ok(Some(RowRef::Owned(self.take_from_buffer())));
                    } else {
                        // More inner rows to check - create combined row directly
                        // (Can't use buffer here due to borrow of current_outer_row)
                        let outer_row = self.current_outer_row.as_ref().unwrap();
                        let combined = self.create_combined_row(outer_row, inner_row);
                        return Ok(Some(RowRef::Owned(combined)));
                    }
                }
            }

            // Exhausted inner rows for current outer row
            residual_failed(&self.residual_filter, self.condition_failed)?;
            // Handle LEFT OUTER: emit outer row with NULLs if no match
            if is_left_outer && !self.outer_had_match {
                let outer_row = self.current_outer_row.take().unwrap();
                if !self.advance_outer()? {
                    self.ended_on = Some(outer_row.clone());
                }
                let null_inner = self.null_inner_row();
                // Use buffer-based combine since we own outer_row
                self.combine_owned_into_buffer(outer_row, null_inner);
                return Ok(Some(RowRef::Owned(self.take_from_buffer())));
            }

            // Move to next outer row
            if !self.advance_outer()? {
                return Ok(None);
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.outer.close()
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        // Rough estimate based on outer side
        let outer_est = self.outer.estimated_rows()?;
        Some(match self.join_type {
            JoinType::Inner => outer_est, // Assume most outer rows match
            JoinType::Left | JoinType::Full => outer_est,
            _ => outer_est,
        })
    }

    fn name(&self) -> &str {
        match self.join_type {
            JoinType::Inner => "IndexNL (INNER)",
            JoinType::Left => "IndexNL (LEFT)",
            _ => "IndexNL",
        }
    }
}

/// Batch Index Nested Loop Join Operator.
///
/// Optimized for queries without LIMIT. Instead of looking up one row at a time,
/// this collects all outer rows first, batches all row ID lookups into a single
/// call, and builds all results at once. This reduces lock overhead from O(N)
/// to O(1) for the inner table access.
pub struct BatchIndexNestedLoopJoinOperator<'a> {
    // Outer input operator
    outer: Box<dyn Operator>,

    // Inner table
    inner_table: Box<dyn Table>,

    // Join configuration
    join_type: JoinType,
    outer_key_idx: usize,
    lookup_strategy: IndexLookupStrategy,
    residual_filter: Option<JoinFilter>,
    // A filter the inner fetch applies to each candidate row
    inner_filter: Option<Box<dyn Expression>>,
    // The inner key column, checked on each fetched row: inside a
    // transaction an index entry may be older than the row it points at
    inner_key_idx: Option<usize>,
    // The side of a cross that column is on, for outer keys no probe answers
    inner_side: Option<KeySide>,

    // Output schema
    schema: Vec<ColumnInfo>,
    inner_col_count: usize,

    // Optional projection pushdown: create projected rows during combine
    projection: Option<JoinProjection>,
    // The outer side is the statement's right side: rows and the residual
    // pair read inner first
    outer_on_right: bool,
    // The statement's subquery blocks, as the residual's site calls them
    blocks: Option<&'a dyn SubqueryExecutor>,

    // Pre-computed results (built in open())
    results: Vec<Row>,
    result_idx: usize,

    // Optimization: Reusable buffer for row IDs to avoid allocation per outer row
    row_id_buffer: Vec<i64>,

    // State
    opened: bool,
}

impl<'a> BatchIndexNestedLoopJoinOperator<'a> {
    /// Create a new batch index nested loop join operator.
    pub fn new(
        outer: Box<dyn Operator>,
        inner_table: Box<dyn Table>,
        inner_schema: Vec<ColumnInfo>,
        join_type: JoinType,
        outer_key_idx: usize,
        lookup_strategy: IndexLookupStrategy,
        residual_filter: Option<JoinFilter>,
    ) -> Self {
        let mut schema = Vec::new();
        schema.extend(outer.schema().iter().cloned());
        schema.extend(inner_schema.iter().cloned());

        let inner_col_count = inner_schema.len();
        let inner_side = inner_key_side(&*inner_table, &lookup_strategy, None);

        Self {
            outer,
            inner_table,
            join_type,
            outer_key_idx,
            lookup_strategy,
            residual_filter,
            inner_filter: None,
            inner_key_idx: None,
            inner_side,
            schema,
            inner_col_count,
            projection: None,
            outer_on_right: false,
            blocks: None,
            results: Vec::new(),
            result_idx: 0,
            // Pre-allocate buffer for typical number of matches
            row_id_buffer: Vec::with_capacity(16),
            opened: false,
        }
    }

    /// The inner key column, so a row an index entry points at is kept
    /// only when its own key equals the outer one
    pub fn with_inner_key(mut self, inner_key_idx: Option<usize>) -> Self {
        self.inner_key_idx = inner_key_idx;
        self.inner_side = inner_key_side(&*self.inner_table, &self.lookup_strategy, inner_key_idx);
        self
    }

    /// The outer side is the statement's right side, so each joined row and
    /// the residual's pair read the inner side first
    pub fn with_outer_on_right(mut self) -> Self {
        if self.projection.is_none() {
            let outer_len = self.schema.len() - self.inner_col_count;
            self.schema.rotate_left(outer_len);
        }
        self.outer_on_right = true;
        self
    }

    /// The statement's subquery blocks the residual calls
    pub fn with_blocks(mut self, blocks: Option<&'a dyn SubqueryExecutor>) -> Self {
        self.blocks = blocks;
        self
    }

    /// A filter the inner fetch applies to each candidate row, for a
    /// predicate on inner columns only
    pub fn with_inner_filter(mut self, filter: Box<dyn Expression>) -> Self {
        self.inner_filter = Some(filter);
        self
    }

    /// Set projection pushdown configuration.
    ///
    /// When set, the operator creates projected rows directly during the combine step
    /// instead of creating full combined rows. This avoids creating large intermediate
    /// rows that would be immediately projected down to fewer columns.
    ///
    /// # Arguments
    /// * `columns` - Column sources in SELECT order (preserves original column ordering)
    /// * `projected_schema` - Schema for the projected output
    pub fn with_projection(
        mut self,
        columns: Vec<ColumnSource>,
        projected_schema: Vec<ColumnInfo>,
    ) -> Self {
        let repeats_source = columns
            .iter()
            .enumerate()
            .any(|(i, c)| columns[..i].contains(c));
        self.projection = Some(JoinProjection {
            columns,
            repeats_source,
        });
        self.schema = projected_schema;
        self
    }

    /// Combine outer and inner rows into output row.
    /// When projection is set, creates projected row directly (avoids full combine).
    ///
    /// # Safety
    /// Uses `get_unchecked` for performance - projection indices are validated
    /// at query planning time against the table schemas.
    #[inline]
    fn combine(&self, outer: &Row, inner: &Row) -> Row {
        match &self.projection {
            Some(proj) => {
                // Fused projection: create only the columns we need, in SELECT order
                let mut values: CompactVec<Value> = CompactVec::with_capacity(proj.columns.len());

                let outer_slice = outer.as_slice();
                let inner_slice = inner.as_slice();

                for col_source in &proj.columns {
                    match col_source {
                        ColumnSource::Outer(idx) => {
                            // SAFETY: indices validated at planning time against outer table schema.
                            values.push(unsafe { outer_slice.get_unchecked(*idx) }.clone());
                        }
                        ColumnSource::Inner(idx) => {
                            // SAFETY: indices validated at planning time against inner table schema.
                            values.push(unsafe { inner_slice.get_unchecked(*idx) }.clone());
                        }
                    }
                }

                Row::from_compact_vec(values)
            }
            None if self.outer_on_right => Row::from_combined(inner, outer),
            None => Row::from_combined(outer, inner),
        }
    }

    /// Create a NULL row for the inner side.
    /// Always creates full-width row so combine() can use consistent indexing.
    #[inline]
    fn null_inner_row(&self) -> Row {
        let null_values: Vec<Value> = (0..self.inner_col_count).map(|_| NULL_VALUE).collect();
        Row::from_values(null_values)
    }
}

impl Operator for BatchIndexNestedLoopJoinOperator<'_> {
    fn open(&mut self) -> Result<()> {
        use crate::common::I64Set;

        self.outer.open()?;

        let is_left_join = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let true_expr = ConstBoolExpr::true_expr();
        let local_rows = LocalRows::of(
            &*self.inner_table,
            &self.lookup_strategy,
            self.inner_key_idx,
        );

        // Step 1: Collect all outer rows and their join keys
        let mut outer_rows: Vec<Row> = Vec::new();
        let mut row_id_set: I64Set = I64Set::new();
        // SmallVec keeps the common unique-key case inline (no heap Vec
        // per distinct row id)
        let mut key_to_outer_indices: I64Map<SmallVec<[usize; 2]>> = I64Map::new();
        let mut text_reads = TextReads::default();
        let mut view_keys: Option<ViewKeys> = None;

        while let Some(row_ref) = self.outer.next()? {
            let outer_row = row_ref.into_owned();
            let outer_idx = outer_rows.len();

            // Get the join key value
            let key_value = match outer_row.get(self.outer_key_idx) {
                Some(v) if !v.is_null() => v.clone(),
                _ => {
                    // NULL key - no match possible
                    outer_rows.push(outer_row);
                    continue;
                }
            };
            let key_value = if crosses_inner_key(self.inner_side, &key_value) {
                match (self.inner_side, &key_value) {
                    // A TEXT key probes as the value it reads as
                    (Some(KeySide::Read(reading)), Value::Text(text)) => match reading.read(text) {
                        Some(read) => read,
                        None => {
                            outer_rows.push(outer_row);
                            continue;
                        }
                    },
                    // Any other key takes the TEXT rows that read as it
                    _ => {
                        if let Some(idx) = self.inner_key_idx {
                            for &row_id in
                                text_reads.row_ids(&*self.inner_table, idx, &key_value)?
                            {
                                key_to_outer_indices
                                    .entry(row_id)
                                    .or_default()
                                    .push(outer_idx);
                                row_id_set.insert(row_id);
                            }
                        }
                        outer_rows.push(outer_row);
                        continue;
                    }
                }
            } else {
                key_value
            };

            // Get row IDs based on lookup strategy into reusable buffer
            self.row_id_buffer.clear();
            match &self.lookup_strategy {
                IndexLookupStrategy::SecondaryIndex(index) => {
                    let checked = view_keys.is_none() && {
                        let epoch = self.inner_table.committed_candidate_epoch();
                        index.get_row_ids_equal_into(
                            std::slice::from_ref(&key_value),
                            &mut self.row_id_buffer,
                        );
                        epoch.is_some()
                            && self.inner_table.add_removed_equal(
                                &**index,
                                &key_value,
                                &mut self.row_id_buffer,
                            )
                            && self.inner_table.committed_candidate_epoch() == epoch
                    };
                    // One probe not checked against what the statement sees
                    // leaves this one and the rest to the keys read through it
                    if !checked {
                        self.row_id_buffer.clear();
                        let idx = self.inner_key_idx.ok_or_else(|| {
                            crate::core::Error::internal("an index join without its inner key")
                        })?;
                        if view_keys.is_none() {
                            view_keys = Some(ViewKeys::scan(&*self.inner_table, idx)?);
                        }
                        if let Some(keys) = &view_keys {
                            self.row_id_buffer
                                .extend_from_slice(keys.row_ids(&key_value));
                        }
                    }
                }
                // Only the streaming join can end early and be run again
                IndexLookupStrategy::TableEquality { .. } => {
                    return Err(crate::core::Error::internal(
                        "table equality probes need the streaming index join",
                    ));
                }
                IndexLookupStrategy::PrimaryKey => match &key_value {
                    Value::Integer(id) => self.row_id_buffer.push(*id),
                    // Only a losslessly integral float can equal a PK;
                    // truncation would join 5.5 against row 5
                    Value::Float(f) => {
                        if let Some(id) = crate::executor::Executor::lossless_float_key(*f) {
                            self.row_id_buffer.push(id);
                        }
                    }
                    _ => {}
                },
            }

            // The transaction's own rows answer for themselves
            if let Some(local) = &local_rows {
                local.apply(&key_value, &mut self.row_id_buffer);
            }

            // Map row IDs to outer row indices
            for &row_id in &self.row_id_buffer {
                key_to_outer_indices
                    .entry(row_id)
                    .or_default()
                    .push(outer_idx);
                row_id_set.insert(row_id);
            }

            outer_rows.push(outer_row);
        }

        if outer_rows.is_empty() {
            self.opened = true;
            return Ok(());
        }

        // Step 2: Single batch fetch of ALL inner rows at once
        let all_row_ids: Vec<i64> = row_id_set.into_iter().collect();
        let filter: &dyn Expression = match &self.inner_filter {
            Some(filter) => filter.as_ref(),
            None => &true_expr,
        };
        let inner_rows_batch = self.inner_table.fetch_rows_by_ids(&all_row_ids, filter)?;

        // Step 3: Build results by matching outer rows with inner rows.
        // Row ids are already deduplicated, so the fetched batch is
        // iterated directly (the old I64Map was built only to iterate)
        let mut matched_outers: Vec<bool> = vec![false; outer_rows.len()];

        // An error a join on this thread left behind is not this one's
        if let Some(filter) = &self.residual_filter {
            let _ = filter.take_error();
        }
        let mut condition_failed = false;
        for (row_id, inner_row) in inner_rows_batch.iter() {
            if let Some(outer_indices) = key_to_outer_indices.get(*row_id) {
                for &outer_idx in outer_indices {
                    let outer_row = &outer_rows[outer_idx];

                    // An index entry may point at a row whose key has since
                    // changed inside the transaction, so the row's own key decides
                    if let (IndexLookupStrategy::SecondaryIndex(_), Some(idx)) =
                        (&self.lookup_strategy, self.inner_key_idx)
                    {
                        let equal = inner_row
                            .get(idx)
                            .zip(outer_row.get(self.outer_key_idx))
                            .and_then(|(inner, outer)| sql_eq(inner, outer));
                        if equal != Some(true) {
                            continue;
                        }
                    }

                    // Apply residual filter if present
                    let passes_filter = if let Some(ref filter) = self.residual_filter {
                        pair_matches(
                            filter,
                            self.outer_on_right,
                            outer_row,
                            inner_row,
                            &mut condition_failed,
                            self.blocks,
                        )
                    } else {
                        true
                    };

                    if passes_filter {
                        matched_outers[outer_idx] = true;
                        self.results.push(self.combine(outer_row, inner_row));
                    }
                }
            }
        }
        residual_failed(&self.residual_filter, condition_failed)?;

        // Handle LEFT OUTER JOIN - emit unmatched outer rows with NULLs
        if is_left_join {
            let null_inner = self.null_inner_row();
            for (idx, outer_row) in outer_rows.iter().enumerate() {
                if !matched_outers[idx] {
                    self.results.push(self.combine(outer_row, &null_inner));
                }
            }
        }

        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Err(crate::core::Error::internal(
                "BatchIndexNestedLoopJoinOperator::next called before open",
            ));
        }

        if self.result_idx < self.results.len() {
            // Use mem::take to avoid cloning - each result is only returned once
            let row = std::mem::take(&mut self.results[self.result_idx]);
            self.result_idx += 1;
            Ok(Some(RowRef::Owned(row)))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.outer.close()
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        self.outer.estimated_rows()
    }

    fn name(&self) -> &str {
        "BatchIndexNL"
    }
}
