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

//! Set operations execution (UNION, INTERSECT, EXCEPT)
//!
//! This module handles SQL set operations that combine results from multiple queries:
//! - UNION / UNION ALL
//! - INTERSECT / INTERSECT ALL
//! - EXCEPT / EXCEPT ALL

use crate::core::row_vec::RowVec;
use crate::core::{Result, Row};
use crate::parser::ast::{SelectStatement, SetOperationType};
use crate::storage::traits::QueryResult;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::context::ExecutionContext;
use super::query_cache::NestedPlan;
use super::result::ExecutorResult;
use super::utils::{hash_row, rows_equal};
use super::Executor;

impl Executor {
    /// Execute set operations (UNION, INTERSECT, EXCEPT)
    /// The limit parameter enables early termination for UNION ALL
    /// Each later branch runs under its own plan in `plan`, `stmt`'s
    pub(crate) fn execute_set_operations(
        &self,
        left_result: Box<dyn QueryResult>,
        stmt: &SelectStatement,
        plan: Option<&NestedPlan>,
        ctx: &ExecutionContext,
        limit: Option<usize>,
    ) -> Result<Box<dyn QueryResult>> {
        let set_ops = &stmt.set_operations;
        // Materialize the left result (with limit for UNION ALL optimization)
        let columns = left_result.columns().to_vec();

        // For UNION ALL with limit, we can take advantage of early termination
        // Check if all operations are UNION ALL
        let all_union_all = set_ops
            .iter()
            .all(|op| matches!(op.operation, SetOperationType::UnionAll));

        let mut running = if let (true, Some(lim)) = (all_union_all, limit) {
            // Only materialize up to limit rows from left side
            let mut rows = RowVec::with_capacity(lim.min(1024));
            let mut left_result = left_result;
            let mut row_id = 0i64;
            while left_result.next() {
                rows.push((row_id, left_result.take_row()));
                row_id += 1;
                if rows.len() >= lim {
                    return Ok(Box::new(ExecutorResult::new(columns, rows)));
                }
            }
            if let Some(err) = left_result.last_error() {
                return Err(err);
            }
            Running::Rows(rows)
        } else if set_ops
            .first()
            .is_some_and(|op| removes_duplicates(&op.operation))
        {
            // The first operation keeps one row of each value, so the left side does
            let mut left = RowSet::default();
            left.read(left_result)?;
            Running::Set(left)
        } else {
            Running::Rows(Self::materialize_result(left_result)?)
        };

        // Process each set operation in sequence
        for (index, set_op) in set_ops.iter().enumerate() {
            // For UNION ALL with limit, check if we already have enough rows
            if matches!(set_op.operation, SetOperationType::UnionAll) {
                if let Some(lim) = limit {
                    if running.len() >= lim {
                        // Already have enough rows, skip remaining set operations
                        break;
                    }
                }
            }

            // Execute the right side query with incremented depth (part of same logical query)
            let set_ctx = ctx.with_incremented_query_depth();
            let branch = plan.and_then(|plan| plan.branch(stmt, index));
            let right_result =
                self.execute_select_with_plan(&set_op.right, &set_ctx, None, branch)?;

            // Validate column count matches (SQL standard requirement)
            let right_col_count = right_result.columns().len();
            let left_col_count = columns.len();
            if left_col_count != right_col_count {
                return Err(crate::Error::internal(format!(
                    "each {} query must have the same number of columns: left has {}, right has {}",
                    match &set_op.operation {
                        SetOperationType::Union | SetOperationType::UnionAll => "UNION",
                        SetOperationType::Intersect | SetOperationType::IntersectAll => "INTERSECT",
                        SetOperationType::Except | SetOperationType::ExceptAll => "EXCEPT",
                    },
                    left_col_count,
                    right_col_count
                )));
            }

            // Apply the set operation
            running = match &set_op.operation {
                SetOperationType::Union => {
                    let mut union = running.into_set();
                    union.read(right_result)?;
                    Running::Set(union)
                }
                SetOperationType::Intersect | SetOperationType::Except => {
                    let mut right = RowSet::default();
                    right.read(right_result)?;
                    let keep_matches = set_op.operation == SetOperationType::Intersect;
                    running.retain(|row| right.contains(row) == keep_matches)
                }
                SetOperationType::UnionAll => {
                    let mut result_rows = running.into_rows();
                    // UNION ALL: just concatenate (keep all duplicates)
                    // With limit optimization, only take as many rows as needed
                    if let Some(lim) = limit {
                        let needed = lim.saturating_sub(result_rows.len());
                        if needed > 0 {
                            let mut right_result = right_result;
                            let mut count = 0;
                            let mut row_id = result_rows.len() as i64;
                            while right_result.next() && count < needed {
                                result_rows.push((row_id, right_result.take_row()));
                                row_id += 1;
                                count += 1;
                            }
                            if count < needed {
                                if let Some(err) = right_result.last_error() {
                                    return Err(err);
                                }
                            }
                        }
                    } else {
                        let right_rows = Self::materialize_result(right_result)?;
                        for (row_id, (_, row)) in (result_rows.len() as i64..).zip(right_rows) {
                            result_rows.push((row_id, row));
                        }
                    }
                    Running::Rows(result_rows)
                }
                SetOperationType::IntersectAll => {
                    let result_rows = running.into_rows();
                    // INTERSECT ALL: keep matching rows with multiplicity with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of (row_index, remaining_count)
                    // Each unique row in right side gets its own counter
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, (_, row)) in right_rows.iter().enumerate() {
                        let hash = hash_row(row);
                        right_hash_map.entry(hash).or_default().push(idx);
                    }

                    // For each unique right row, count how many times it appears
                    let mut right_row_counts: FxHashMap<u64, Vec<(usize, usize)>> =
                        FxHashMap::default();
                    for (hash, indices) in right_hash_map {
                        let mut unique_rows_in_bucket: Vec<(usize, usize)> = Vec::new();
                        for &idx in &indices {
                            // Find if this row already exists in unique_rows_in_bucket
                            if let Some(entry) =
                                unique_rows_in_bucket.iter_mut().find(|(rep_idx, _)| {
                                    rows_equal(&right_rows[*rep_idx].1, &right_rows[idx].1)
                                })
                            {
                                entry.1 += 1; // Increment count
                            } else {
                                unique_rows_in_bucket.push((idx, 1)); // New unique row
                            }
                        }
                        right_row_counts.insert(hash, unique_rows_in_bucket);
                    }

                    let mut intersected_rows = RowVec::new();
                    let mut row_id = 0i64;
                    for (_, left_row) in result_rows {
                        let hash = hash_row(&left_row);

                        // Find matching right row and decrement its count
                        if let Some(bucket) = right_row_counts.get_mut(&hash) {
                            // Find the first matching row with count > 0
                            if let Some(entry) = bucket.iter_mut().find(|(rep_idx, count)| {
                                *count > 0 && rows_equal(&left_row, &right_rows[*rep_idx].1)
                            }) {
                                entry.1 -= 1; // Decrement count
                                intersected_rows.push((row_id, left_row));
                                row_id += 1;
                            }
                        }
                    }

                    Running::Rows(intersected_rows)
                }
                SetOperationType::ExceptAll => {
                    let result_rows = running.into_rows();
                    // EXCEPT ALL: remove matching rows with multiplicity with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of row indices
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, (_, row)) in right_rows.iter().enumerate() {
                        let hash = hash_row(row);
                        right_hash_map.entry(hash).or_default().push(idx);
                    }

                    // For each unique right row, count how many times it appears
                    let mut right_row_counts: FxHashMap<u64, Vec<(usize, usize)>> =
                        FxHashMap::default();
                    for (hash, indices) in right_hash_map {
                        let mut unique_rows_in_bucket: Vec<(usize, usize)> = Vec::new();
                        for &idx in &indices {
                            // Find if this row already exists in unique_rows_in_bucket
                            if let Some(entry) =
                                unique_rows_in_bucket.iter_mut().find(|(rep_idx, _)| {
                                    rows_equal(&right_rows[*rep_idx].1, &right_rows[idx].1)
                                })
                            {
                                entry.1 += 1; // Increment count
                            } else {
                                unique_rows_in_bucket.push((idx, 1)); // New unique row
                            }
                        }
                        right_row_counts.insert(hash, unique_rows_in_bucket);
                    }

                    let mut excepted_rows = RowVec::new();
                    let mut row_id = 0i64;
                    for (_, left_row) in result_rows {
                        let hash = hash_row(&left_row);

                        // Check if this row should be removed (exists in right with count > 0)
                        let mut should_remove = false;
                        if let Some(bucket) = right_row_counts.get_mut(&hash) {
                            // Find the first matching row with count > 0
                            if let Some(entry) = bucket.iter_mut().find(|(rep_idx, count)| {
                                *count > 0 && rows_equal(&left_row, &right_rows[*rep_idx].1)
                            }) {
                                entry.1 -= 1; // Decrement count
                                should_remove = true;
                            }
                        }

                        if !should_remove {
                            excepted_rows.push((row_id, left_row));
                            row_id += 1;
                        }
                    }

                    Running::Rows(excepted_rows)
                }
            };
        }

        Ok(Box::new(ExecutorResult::new(columns, running.into_rows())))
    }
}

/// Whether the operation answers one row of each value
fn removes_duplicates(operation: &SetOperationType) -> bool {
    matches!(
        operation,
        SetOperationType::Union | SetOperationType::Intersect | SetOperationType::Except
    )
}

/// The rows the operations so far answer, and whether each value is in them once
enum Running {
    Rows(RowVec),
    Distinct(RowVec),
    Set(RowSet),
}

impl Running {
    fn len(&self) -> usize {
        match self {
            Running::Rows(rows) | Running::Distinct(rows) => rows.len(),
            Running::Set(set) => set.rows.len(),
        }
    }

    fn into_rows(self) -> RowVec {
        match self {
            Running::Rows(rows) | Running::Distinct(rows) => rows,
            Running::Set(set) => set.rows,
        }
    }

    fn into_set(self) -> RowSet {
        match self {
            Running::Set(set) => set,
            Running::Rows(rows) | Running::Distinct(rows) => {
                let mut set = RowSet::default();
                for (_, row) in rows {
                    set.insert(row);
                }
                set
            }
        }
    }

    /// One row of each value that `keep` keeps
    fn retain(self, keep: impl Fn(&Row) -> bool) -> Running {
        match self {
            Running::Rows(rows) => {
                let mut kept = RowSet::default();
                for (_, row) in rows {
                    if keep(&row) {
                        kept.insert(row);
                    }
                }
                Running::Set(kept)
            }
            distinct => {
                let mut rows = distinct.into_rows();
                rows.retain(|(_, row)| keep(row));
                for (id, (row_id, _)) in (0i64..).zip(rows.iter_mut()) {
                    *row_id = id;
                }
                Running::Distinct(rows)
            }
        }
    }
}

/// One row of each value, in the order first met, by the equality the set
/// operations compare rows by
#[derive(Default)]
struct RowSet {
    rows: RowVec,
    by_hash: FxHashMap<u64, SmallVec<[usize; 1]>>,
}

impl RowSet {
    fn holds(&self, hash: u64, row: &Row) -> bool {
        self.by_hash
            .get(&hash)
            .is_some_and(|at| at.iter().any(|&i| rows_equal(&self.rows[i].1, row)))
    }

    fn contains(&self, row: &Row) -> bool {
        self.holds(hash_row(row), row)
    }

    /// The place a row of `row`'s value takes, taken for it; None when a
    /// row of that value is kept
    fn place(&mut self, row: &Row) -> Option<usize> {
        let hash = hash_row(row);
        if self.holds(hash, row) {
            return None;
        }
        let at = self.rows.len();
        self.by_hash.entry(hash).or_default().push(at);
        Some(at)
    }

    /// Keeps `row` unless a row of its value is kept
    fn insert(&mut self, row: Row) {
        if let Some(at) = self.place(&row) {
            self.rows.push((at as i64, row));
        }
    }

    /// Takes each row of `result` whose value it does not hold; a row it
    /// holds is read in place and never taken
    fn read(&mut self, mut result: Box<dyn QueryResult>) -> Result<()> {
        while result.next() {
            if let Some(at) = self.place(result.row()) {
                self.rows.push((at as i64, result.take_row()));
            }
        }
        match result.last_error() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}
