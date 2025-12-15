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

use crate::core::Result;
use crate::parser::ast::{SetOperation, SetOperationType};
use crate::storage::traits::QueryResult;
use rustc_hash::FxHashMap;

use super::context::ExecutionContext;
use super::result::ExecutorMemoryResult;
use super::utils::{hash_row, rows_equal};
use super::Executor;

impl Executor {
    /// Execute set operations (UNION, INTERSECT, EXCEPT)
    /// The limit parameter enables early termination for UNION ALL
    pub(crate) fn execute_set_operations(
        &self,
        left_result: Box<dyn QueryResult>,
        set_ops: &[SetOperation],
        ctx: &ExecutionContext,
        limit: Option<usize>,
    ) -> Result<Box<dyn QueryResult>> {
        // Materialize the left result (with limit for UNION ALL optimization)
        let columns = left_result.columns().to_vec();

        // For UNION ALL with limit, we can take advantage of early termination
        // Check if all operations are UNION ALL
        let all_union_all = set_ops
            .iter()
            .all(|op| matches!(op.operation, SetOperationType::UnionAll));

        let mut result_rows = if let (true, Some(lim)) = (all_union_all, limit) {
            // Only materialize up to limit rows from left side
            let mut rows = Vec::with_capacity(lim.min(1024));
            let mut left_result = left_result;
            while left_result.next() {
                rows.push(left_result.take_row());
                if rows.len() >= lim {
                    return Ok(Box::new(ExecutorMemoryResult::new(columns, rows)));
                }
            }
            rows
        } else {
            Self::materialize_result(left_result)?
        };

        // Process each set operation in sequence
        for set_op in set_ops {
            // For UNION ALL with limit, check if we already have enough rows
            if matches!(set_op.operation, SetOperationType::UnionAll) {
                if let Some(lim) = limit {
                    if result_rows.len() >= lim {
                        // Already have enough rows, skip remaining set operations
                        break;
                    }
                }
            }

            // Execute the right side query with incremented depth (part of same logical query)
            let set_ctx = ctx.with_incremented_query_depth();
            let right_result = self.execute_select(&set_op.right, &set_ctx)?;

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
            match &set_op.operation {
                SetOperationType::Union => {
                    // UNION: combine rows and remove duplicates with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;
                    // Use hash map: hash -> list of indices to detect duplicates with collision handling
                    let mut hash_to_indices: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    let mut unique_rows = Vec::new();

                    // Add left rows (dedup)
                    for row in result_rows {
                        let hash = hash_row(&row);
                        let indices = hash_to_indices.entry(hash).or_default();

                        // Check if this exact row already exists (handle hash collisions)
                        let is_duplicate = indices
                            .iter()
                            .any(|&idx| rows_equal(&unique_rows[idx], &row));

                        if !is_duplicate {
                            indices.push(unique_rows.len());
                            unique_rows.push(row);
                        }
                    }

                    // Add right rows (dedup)
                    for row in right_rows {
                        let hash = hash_row(&row);
                        let indices = hash_to_indices.entry(hash).or_default();

                        // Check if this exact row already exists (handle hash collisions)
                        let is_duplicate = indices
                            .iter()
                            .any(|&idx| rows_equal(&unique_rows[idx], &row));

                        if !is_duplicate {
                            indices.push(unique_rows.len());
                            unique_rows.push(row);
                        }
                    }

                    result_rows = unique_rows;
                }
                SetOperationType::UnionAll => {
                    // UNION ALL: just concatenate (keep all duplicates)
                    // With limit optimization, only take as many rows as needed
                    if let Some(lim) = limit {
                        let needed = lim.saturating_sub(result_rows.len());
                        if needed > 0 {
                            let mut right_result = right_result;
                            let mut count = 0;
                            while right_result.next() && count < needed {
                                result_rows.push(right_result.take_row());
                                count += 1;
                            }
                        }
                    } else {
                        let right_rows = Self::materialize_result(right_result)?;
                        result_rows.extend(right_rows);
                    }
                }
                SetOperationType::Intersect => {
                    // INTERSECT: keep only rows that exist in both (dedup) with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of right rows with that hash
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, row) in right_rows.iter().enumerate() {
                        let hash = hash_row(row);
                        right_hash_map.entry(hash).or_default().push(idx);
                    }

                    // Track which left rows we've already added (for deduplication)
                    let mut left_seen: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    let mut intersected_rows = Vec::new();

                    for left_row in result_rows {
                        let hash = hash_row(&left_row);

                        // Check if this hash exists in right side
                        if let Some(right_indices) = right_hash_map.get(&hash) {
                            // Check if any right row with this hash actually equals this left row
                            let has_match = right_indices
                                .iter()
                                .any(|&idx| rows_equal(&left_row, &right_rows[idx]));

                            if has_match {
                                // Check if we've already added this left row (dedup)
                                let left_indices = left_seen.entry(hash).or_default();
                                let is_duplicate = left_indices
                                    .iter()
                                    .any(|&idx| rows_equal(&intersected_rows[idx], &left_row));

                                if !is_duplicate {
                                    left_indices.push(intersected_rows.len());
                                    intersected_rows.push(left_row);
                                }
                            }
                        }
                    }

                    result_rows = intersected_rows;
                }
                SetOperationType::IntersectAll => {
                    // INTERSECT ALL: keep matching rows with multiplicity with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of (row_index, remaining_count)
                    // Each unique row in right side gets its own counter
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, row) in right_rows.iter().enumerate() {
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
                                    rows_equal(&right_rows[*rep_idx], &right_rows[idx])
                                })
                            {
                                entry.1 += 1; // Increment count
                            } else {
                                unique_rows_in_bucket.push((idx, 1)); // New unique row
                            }
                        }
                        right_row_counts.insert(hash, unique_rows_in_bucket);
                    }

                    let mut intersected_rows = Vec::new();
                    for left_row in result_rows {
                        let hash = hash_row(&left_row);

                        // Find matching right row and decrement its count
                        if let Some(bucket) = right_row_counts.get_mut(&hash) {
                            // Find the first matching row with count > 0
                            if let Some(entry) = bucket.iter_mut().find(|(rep_idx, count)| {
                                *count > 0 && rows_equal(&left_row, &right_rows[*rep_idx])
                            }) {
                                entry.1 -= 1; // Decrement count
                                intersected_rows.push(left_row);
                            }
                        }
                    }

                    result_rows = intersected_rows;
                }
                SetOperationType::Except => {
                    // EXCEPT: keep left rows not in right (dedup) with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of right row indices
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, row) in right_rows.iter().enumerate() {
                        let hash = hash_row(row);
                        right_hash_map.entry(hash).or_default().push(idx);
                    }

                    // Track which left rows we've already added (for deduplication)
                    let mut left_seen: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    let mut excepted_rows = Vec::new();

                    for left_row in result_rows {
                        let hash = hash_row(&left_row);

                        // Check if this row exists in right side
                        let exists_in_right = if let Some(right_indices) = right_hash_map.get(&hash)
                        {
                            right_indices
                                .iter()
                                .any(|&idx| rows_equal(&left_row, &right_rows[idx]))
                        } else {
                            false
                        };

                        if !exists_in_right {
                            // Check if we've already added this left row (dedup)
                            let left_indices = left_seen.entry(hash).or_default();
                            let is_duplicate = left_indices
                                .iter()
                                .any(|&idx| rows_equal(&excepted_rows[idx], &left_row));

                            if !is_duplicate {
                                left_indices.push(excepted_rows.len());
                                excepted_rows.push(left_row);
                            }
                        }
                    }

                    result_rows = excepted_rows;
                }
                SetOperationType::ExceptAll => {
                    // EXCEPT ALL: remove matching rows with multiplicity with proper collision handling
                    let right_rows = Self::materialize_result(right_result)?;

                    // Build hash map: hash -> list of row indices
                    let mut right_hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
                    for (idx, row) in right_rows.iter().enumerate() {
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
                                    rows_equal(&right_rows[*rep_idx], &right_rows[idx])
                                })
                            {
                                entry.1 += 1; // Increment count
                            } else {
                                unique_rows_in_bucket.push((idx, 1)); // New unique row
                            }
                        }
                        right_row_counts.insert(hash, unique_rows_in_bucket);
                    }

                    let mut excepted_rows = Vec::new();
                    for left_row in result_rows {
                        let hash = hash_row(&left_row);

                        // Check if this row should be removed (exists in right with count > 0)
                        let mut should_remove = false;
                        if let Some(bucket) = right_row_counts.get_mut(&hash) {
                            // Find the first matching row with count > 0
                            if let Some(entry) = bucket.iter_mut().find(|(rep_idx, count)| {
                                *count > 0 && rows_equal(&left_row, &right_rows[*rep_idx])
                            }) {
                                entry.1 -= 1; // Decrement count
                                should_remove = true;
                            }
                        }

                        if !should_remove {
                            excepted_rows.push(left_row);
                        }
                    }

                    result_rows = excepted_rows;
                }
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(columns, result_rows)))
    }
}
