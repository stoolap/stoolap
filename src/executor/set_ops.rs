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

use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::Result;
use crate::parser::ast::{SetOperation, SetOperationType};
use crate::storage::traits::QueryResult;

use super::context::ExecutionContext;
use super::result::ExecutorMemoryResult;
use super::utils::hash_row;
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
                    // UNION: combine rows and remove duplicates
                    let right_rows = Self::materialize_result(right_result)?;
                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    let mut unique_rows = Vec::new();

                    // Add left rows (dedup)
                    for row in result_rows {
                        let hash = hash_row(&row);
                        if seen.insert(hash) {
                            unique_rows.push(row);
                        }
                    }

                    // Add right rows (dedup)
                    for row in right_rows {
                        let hash = hash_row(&row);
                        if seen.insert(hash) {
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
                    // INTERSECT: keep only rows that exist in both (dedup)
                    let right_rows = Self::materialize_result(right_result)?;
                    let right_hashes: FxHashSet<u64> = right_rows.iter().map(hash_row).collect();

                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    result_rows.retain(|row| {
                        let hash = hash_row(row);
                        right_hashes.contains(&hash) && seen.insert(hash)
                    });
                }
                SetOperationType::IntersectAll => {
                    // INTERSECT ALL: keep matching rows with multiplicity
                    let right_rows = Self::materialize_result(right_result)?;
                    let mut right_counts: FxHashMap<u64, usize> = FxHashMap::default();
                    for row in &right_rows {
                        *right_counts.entry(hash_row(row)).or_insert(0) += 1;
                    }

                    result_rows.retain(|row| {
                        let hash = hash_row(row);
                        if let Some(count) = right_counts.get_mut(&hash) {
                            if *count > 0 {
                                *count -= 1;
                                return true;
                            }
                        }
                        false
                    });
                }
                SetOperationType::Except => {
                    // EXCEPT: keep left rows not in right (dedup)
                    let right_rows = Self::materialize_result(right_result)?;
                    let right_hashes: FxHashSet<u64> = right_rows.iter().map(hash_row).collect();

                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    result_rows.retain(|row| {
                        let hash = hash_row(row);
                        !right_hashes.contains(&hash) && seen.insert(hash)
                    });
                }
                SetOperationType::ExceptAll => {
                    // EXCEPT ALL: remove matching rows with multiplicity
                    let right_rows = Self::materialize_result(right_result)?;
                    let mut right_counts: FxHashMap<u64, usize> = FxHashMap::default();
                    for row in &right_rows {
                        *right_counts.entry(hash_row(row)).or_insert(0) += 1;
                    }

                    result_rows.retain(|row| {
                        let hash = hash_row(row);
                        if let Some(count) = right_counts.get_mut(&hash) {
                            if *count > 0 {
                                *count -= 1;
                                return false; // Remove this row
                            }
                        }
                        true // Keep this row
                    });
                }
            }
        }

        Ok(Box::new(ExecutorMemoryResult::new(columns, result_rows)))
    }
}
