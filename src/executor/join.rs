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

//! JOIN Execution Module
//!
//! This module implements various join algorithms:
//! - Hash Join: O(N + M) with build/probe phases and bloom filter optimization
//! - Merge Join: O(N + M) for pre-sorted inputs
//! - Nested Loop Join: O(N * M) fallback for complex conditions
//!
//! Also provides utilities for:
//! - Join key extraction from ON conditions
//! - Row combination for join output
//! - Value hashing and comparison

use std::cmp::Ordering;

use rustc_hash::FxHashMap;

#[cfg(test)]
use crate::core::Value;
use crate::core::{Result, Row};
use crate::optimizer::bloom::BloomFilter;
use crate::parser::ast::{Expression, InfixOperator};

use super::context::ExecutionContext;
use super::expression::{JoinFilter, RowFilter};
use super::parallel::{self, ParallelConfig};
use super::Executor;

/// Index Nested Loop Join lookup strategy.
/// Determines how to find matching rows in the inner (right) table.
#[derive(Clone)]
pub enum IndexLookupStrategy {
    /// Use a secondary index for lookups (index.get_row_ids_equal)
    SecondaryIndex(std::sync::Arc<dyn crate::storage::traits::Index>),
    /// Use primary key lookup (direct row_id = value)
    /// In stoolap, PRIMARY KEY INTEGER values ARE the row_ids
    PrimaryKey,
}

// Re-export utilities from utils.rs for backward compatibility
#[allow(unused_imports)]
pub use super::utils::{
    build_column_index_map, combine_rows, combine_rows_with_nulls, compare_values,
    extract_column_name_with_qualifier, find_column_index, hash_composite_key, hash_value_into,
    is_sorted_on_keys, values_equal, verify_composite_key_equality,
};

/// Minimum build side size to use bloom filter optimization.
/// For small build sides, the bloom filter overhead isn't worth it.
const BLOOM_FILTER_MIN_BUILD_SIZE: usize = 100;

/// Compare composite keys for merge join ordering.
pub fn compare_composite_keys(
    row1: &Row,
    row2: &Row,
    indices1: &[usize],
    indices2: &[usize],
) -> Ordering {
    debug_assert_eq!(indices1.len(), indices2.len());

    for (&idx1, &idx2) in indices1.iter().zip(indices2.iter()) {
        let cmp = match (row1.get(idx1), row2.get(idx2)) {
            (Some(v1), Some(v2)) => compare_values(v1, v2),
            (None, None) => Ordering::Equal,      // Both NULL
            (None, Some(_)) => Ordering::Greater, // NULL sorts last
            (Some(_), None) => Ordering::Less,    // NULL sorts last
        };
        if cmp != Ordering::Equal {
            return cmp;
        }
    }
    Ordering::Equal
}

// ============================================================================
// Join Key Extraction
// ============================================================================

/// Extract equality join keys and residual conditions from a join condition.
///
/// Returns (left_indices, right_indices, residual_conditions) where residual
/// contains non-equality conditions that must be applied after the hash join.
pub fn extract_join_keys_and_residual(
    condition: &Expression,
    left_columns: &[String],
    right_columns: &[String],
) -> (Vec<usize>, Vec<usize>, Vec<Expression>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    let mut residual = Vec::new();

    extract_join_keys_recursive(
        condition,
        left_columns,
        right_columns,
        &mut left_indices,
        &mut right_indices,
        &mut residual,
    );

    (left_indices, right_indices, residual)
}

/// Recursively extract equality join keys from AND expressions.
fn extract_join_keys_recursive(
    condition: &Expression,
    left_columns: &[String],
    right_columns: &[String],
    left_indices: &mut Vec<usize>,
    right_indices: &mut Vec<usize>,
    residual: &mut Vec<Expression>,
) {
    match condition {
        Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
            // Recurse into AND branches
            extract_join_keys_recursive(
                &infix.left,
                left_columns,
                right_columns,
                left_indices,
                right_indices,
                residual,
            );
            extract_join_keys_recursive(
                &infix.right,
                left_columns,
                right_columns,
                left_indices,
                right_indices,
                residual,
            );
        }
        Expression::Infix(infix) if infix.op_type == InfixOperator::Equal => {
            // Extract equality condition
            if let (Some(left_col), Some(right_col)) = (
                extract_column_name_with_qualifier(&infix.left),
                extract_column_name_with_qualifier(&infix.right),
            ) {
                // Case 1: left.col = right.col
                if let (Some(left_idx), Some(right_idx)) = (
                    find_column_index(&left_col, left_columns),
                    find_column_index(&right_col, right_columns),
                ) {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                    return;
                }

                // Case 2: right.col = left.col (swapped)
                if let (Some(left_idx), Some(right_idx)) = (
                    find_column_index(&right_col, left_columns),
                    find_column_index(&left_col, right_columns),
                ) {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                    return;
                }
            }
            // Non-join equality (e.g., a.x = 5) - add to residual
            residual.push(condition.clone());
        }
        _ => {
            // Non-equality condition - add to residual filters
            residual.push(condition.clone());
        }
    }
}

// Note: extract_column_name_with_qualifier, find_column_index, and is_sorted_on_keys
// are now provided by utils.rs and re-exported above

// ============================================================================
// Join Algorithm Implementations (impl Executor)
// ============================================================================

impl Executor {
    /// Apply residual conditions (non-equality parts of ON clause) after join.
    ///
    /// For OUTER JOINs, preserves NULL-padded rows that represent "no match".
    ///
    /// CRITICAL: This function now returns Result to properly propagate compilation errors.
    /// Previously, compilation failures were silently ignored which could cause incorrect results.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_residual_conditions(
        &self,
        rows: &mut Vec<Row>,
        residual: &[Expression],
        all_columns: &[String],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
        ctx: &ExecutionContext,
    ) -> Result<()> {
        // Pre-compile all residual conditions into filters
        // CRITICAL: Propagate compilation errors instead of silently ignoring them
        let all_columns_vec: Vec<String> = all_columns.to_vec();
        let filters: Vec<RowFilter> = residual
            .iter()
            .map(|cond| RowFilter::new(cond, &all_columns_vec).map(|f| f.with_context(ctx)))
            .collect::<Result<Vec<_>>>()?;

        // If no filters, nothing to do
        if filters.is_empty() {
            return Ok(());
        }

        let is_outer_join =
            join_type.contains("LEFT") || join_type.contains("RIGHT") || join_type.contains("FULL");

        if is_outer_join {
            rows.retain(|row| {
                // Check if this is a NULL-padded row from an outer join
                let left_all_null =
                    (0..left_col_count).all(|i| row.get(i).map(|v| v.is_null()).unwrap_or(true));
                let right_all_null = (left_col_count..left_col_count + right_col_count)
                    .all(|i| row.get(i).map(|v| v.is_null()).unwrap_or(true));

                // Preserve NULL-padded outer join rows unconditionally
                if (join_type.contains("LEFT") && right_all_null)
                    || (join_type.contains("RIGHT") && left_all_null)
                    || (join_type.contains("FULL") && (left_all_null || right_all_null))
                {
                    return true;
                }

                // For matched rows, apply residual conditions normally
                filters.iter().all(|f| f.matches(row))
            });
        } else {
            // For INNER JOIN, simple filter is correct
            rows.retain(|row| filters.iter().all(|f| f.matches(row)));
        }

        Ok(())
    }

    /// Execute hash join - O(N + M) complexity.
    ///
    /// Supports multiple join keys for conditions like: a.x = b.x AND a.y = b.y
    /// Uses parallel execution when either side has >= 5000 rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_hash_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        left_key_indices: &[usize],
        right_key_indices: &[usize],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        // Optimization: Build hash table on smaller side
        // For LEFT/FULL joins, we must build on right side to track unmatched rows correctly
        // For RIGHT joins, we must build on left side
        let build_on_left = !join_type.contains("LEFT")
            && !join_type.contains("FULL")
            && (join_type.contains("RIGHT") || left_rows.len() < right_rows.len());

        // Check if parallel execution would be beneficial
        let parallel_config = ParallelConfig::default();
        let use_parallel = limit.is_none()
            && (parallel_config.should_parallel_join(left_rows.len())
                || parallel_config.should_parallel_join(right_rows.len()));

        if use_parallel {
            let parallel_join_type = parallel::JoinType::from_str(join_type);

            let result = if build_on_left {
                parallel::parallel_hash_join(
                    right_rows,
                    left_rows,
                    right_key_indices,
                    left_key_indices,
                    parallel_join_type,
                    right_col_count,
                    left_col_count,
                    true, // swapped
                    &parallel_config,
                )
            } else {
                parallel::parallel_hash_join(
                    left_rows,
                    right_rows,
                    left_key_indices,
                    right_key_indices,
                    parallel_join_type,
                    left_col_count,
                    right_col_count,
                    false, // not swapped
                    &parallel_config,
                )
            };

            return Ok(result.rows);
        }

        // Fall back to sequential execution
        if build_on_left {
            self.execute_hash_join_impl(
                right_rows,
                left_rows,
                right_key_indices,
                left_key_indices,
                join_type,
                right_col_count,
                left_col_count,
                true,
                limit,
            )
        } else {
            self.execute_hash_join_impl(
                left_rows,
                right_rows,
                left_key_indices,
                right_key_indices,
                join_type,
                left_col_count,
                right_col_count,
                false,
                limit,
            )
        }
    }

    /// Core hash join implementation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_hash_join_impl(
        &self,
        probe_rows: &[Row],
        build_rows: &[Row],
        probe_key_indices: &[usize],
        build_key_indices: &[usize],
        join_type: &str,
        probe_col_count: usize,
        build_col_count: usize,
        swapped: bool,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        use crate::optimizer::bloom::BloomEffectivenessTracker;

        // Build phase: Create hash table from build side with FxHash (optimized for trusted keys)
        let mut hash_table: FxHashMap<u64, Vec<usize>> = FxHashMap::default();

        // Build bloom filter for faster probe-side filtering
        let bloom_stats = BloomEffectivenessTracker::global();
        let use_bloom_filter = build_rows.len() >= BLOOM_FILTER_MIN_BUILD_SIZE;
        let mut bloom_filter = if use_bloom_filter {
            let fp_rate = bloom_stats.recommend_false_positive_rate();
            Some(BloomFilter::new(build_rows.len(), fp_rate))
        } else {
            None
        };

        for (idx, row) in build_rows.iter().enumerate() {
            let hash = hash_composite_key(row, build_key_indices);
            hash_table.entry(hash).or_default().push(idx);

            if let Some(ref mut bf) = bloom_filter {
                bf.insert_raw_hash(hash);
            }
        }

        let mut result_rows = Vec::new();
        let mut build_matched = vec![false; build_rows.len()];

        // Early termination conditions
        let is_inner_join = !join_type.contains("LEFT")
            && !join_type.contains("RIGHT")
            && !join_type.contains("FULL");
        let is_left_not_swapped =
            join_type.contains("LEFT") && !swapped && !join_type.contains("FULL");
        let is_right_swapped =
            join_type.contains("RIGHT") && swapped && !join_type.contains("FULL");
        let can_early_terminate = is_inner_join || is_left_not_swapped || is_right_swapped;
        let effective_limit = if can_early_terminate { limit } else { None };

        // Probe phase
        'probe: for probe_row in probe_rows {
            let mut matched = false;
            let hash = hash_composite_key(probe_row, probe_key_indices);

            // Bloom filter early rejection
            if let Some(ref bf) = bloom_filter {
                if !bf.might_contain_raw_hash(hash) {
                    bloom_stats.record_true_negative();
                    let needs_null_row = if swapped {
                        join_type.contains("RIGHT") || join_type.contains("FULL")
                    } else {
                        join_type.contains("LEFT") || join_type.contains("FULL")
                    };

                    if needs_null_row {
                        let values = if swapped {
                            combine_rows_with_nulls(
                                probe_row,
                                probe_col_count,
                                build_col_count,
                                false,
                            )
                        } else {
                            combine_rows_with_nulls(
                                probe_row,
                                probe_col_count,
                                build_col_count,
                                true,
                            )
                        };
                        result_rows.push(Row::from_values(values));
                        if let Some(lim) = effective_limit {
                            if result_rows.len() >= lim as usize {
                                break 'probe;
                            }
                        }
                    }
                    continue;
                }
            }

            // Hash table lookup
            if let Some(build_indices) = hash_table.get(&hash) {
                for &build_idx in build_indices {
                    let build_row = &build_rows[build_idx];

                    // Verify actual key equality (handle hash collisions)
                    if verify_composite_key_equality(
                        probe_row,
                        build_row,
                        probe_key_indices,
                        build_key_indices,
                    ) {
                        matched = true;
                        build_matched[build_idx] = true;

                        let values = if swapped {
                            combine_rows(build_row, probe_row, build_col_count, probe_col_count)
                        } else {
                            combine_rows(probe_row, build_row, probe_col_count, build_col_count)
                        };
                        result_rows.push(Row::from_values(values));

                        if let Some(lim) = effective_limit {
                            if result_rows.len() >= lim as usize {
                                break 'probe;
                            }
                        }
                    }
                }
            }

            // Handle OUTER join null padding for unmatched probe rows
            if !matched {
                let needs_null_row = if swapped {
                    join_type.contains("RIGHT") || join_type.contains("FULL")
                } else {
                    join_type.contains("LEFT") || join_type.contains("FULL")
                };

                if needs_null_row {
                    let values = if swapped {
                        combine_rows_with_nulls(probe_row, probe_col_count, build_col_count, false)
                    } else {
                        combine_rows_with_nulls(probe_row, probe_col_count, build_col_count, true)
                    };
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        // Handle unmatched build rows for FULL OUTER or RIGHT/LEFT (when swapped)
        let needs_unmatched_build = if swapped {
            join_type.contains("LEFT") || join_type.contains("FULL")
        } else {
            join_type.contains("RIGHT") || join_type.contains("FULL")
        };

        if needs_unmatched_build {
            for (build_idx, &was_matched) in build_matched.iter().enumerate() {
                if !was_matched {
                    let build_row = &build_rows[build_idx];
                    let values = if swapped {
                        combine_rows_with_nulls(build_row, build_col_count, probe_col_count, true)
                    } else {
                        combine_rows_with_nulls(build_row, build_col_count, probe_col_count, false)
                    };
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Nested loop join fallback for complex conditions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_nested_loop_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        condition: Option<&Expression>,
        _all_columns: &[String],
        left_columns: &[String],
        right_columns: &[String],
        join_type: &str,
        _ctx: &ExecutionContext,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        let is_inner_or_cross = !join_type.contains("LEFT")
            && !join_type.contains("RIGHT")
            && !join_type.contains("FULL");
        let effective_limit = if is_inner_or_cross { limit } else { None };

        // Pre-compile join filter if condition exists
        let join_filter = if let Some(cond) = condition {
            Some(JoinFilter::new(
                cond,
                left_columns,
                right_columns,
                &self.function_registry,
            )?)
        } else {
            None
        };

        let mut result_rows = Vec::new();
        let mut right_matched = vec![false; right_rows.len()];

        let left_col_count = left_columns.len();
        let right_col_count = right_columns.len();

        'outer: for left_row in left_rows {
            let mut matched = false;

            for (right_idx, right_row) in right_rows.iter().enumerate() {
                let matches = if let Some(ref filter) = join_filter {
                    filter.matches(left_row, right_row)
                } else {
                    true // CROSS JOIN
                };

                if matches {
                    matched = true;
                    right_matched[right_idx] = true;
                    let values = combine_rows(left_row, right_row, left_col_count, right_col_count);
                    result_rows.push(Row::from_values(values));

                    if let Some(lim) = effective_limit {
                        if result_rows.len() >= lim as usize {
                            break 'outer;
                        }
                    }
                }
            }

            // Handle LEFT OUTER JOIN
            if !matched && (join_type.contains("LEFT") || join_type.contains("FULL")) {
                let values =
                    combine_rows_with_nulls(left_row, left_col_count, right_col_count, true);
                result_rows.push(Row::from_values(values));
            }
        }

        // Handle RIGHT OUTER JOIN
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            for (right_idx, was_matched) in right_matched.iter().enumerate() {
                if !was_matched {
                    let values = combine_rows_with_nulls(
                        &right_rows[right_idx],
                        right_col_count,
                        left_col_count,
                        false,
                    );
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Merge join implementation for pre-sorted inputs - O(N + M).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_merge_join(
        &self,
        left_rows: &[Row],
        right_rows: &[Row],
        left_key_indices: &[usize],
        right_key_indices: &[usize],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
    ) -> Result<Vec<Row>> {
        let mut result_rows = Vec::new();

        let mut left_matched = vec![false; left_rows.len()];
        let mut right_matched = vec![false; right_rows.len()];

        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left_rows.len() && right_idx < right_rows.len() {
            let left_row = &left_rows[left_idx];
            let right_row = &right_rows[right_idx];

            let cmp =
                compare_composite_keys(left_row, right_row, left_key_indices, right_key_indices);

            match cmp {
                Ordering::Less => {
                    left_idx += 1;
                }
                Ordering::Greater => {
                    right_idx += 1;
                }
                Ordering::Equal => {
                    // Find range of left rows with same key
                    let left_start = left_idx;
                    while left_idx < left_rows.len()
                        && compare_composite_keys(
                            &left_rows[left_start],
                            &left_rows[left_idx],
                            left_key_indices,
                            left_key_indices,
                        ) == Ordering::Equal
                    {
                        left_idx += 1;
                    }

                    // Find range of right rows with same key
                    let right_start = right_idx;
                    while right_idx < right_rows.len()
                        && compare_composite_keys(
                            &right_rows[right_start],
                            &right_rows[right_idx],
                            right_key_indices,
                            right_key_indices,
                        ) == Ordering::Equal
                    {
                        right_idx += 1;
                    }

                    // Cartesian product of matching groups
                    for l_idx in left_start..left_idx {
                        left_matched[l_idx] = true;
                        for r_idx in right_start..right_idx {
                            right_matched[r_idx] = true;
                            let values = combine_rows(
                                &left_rows[l_idx],
                                &right_rows[r_idx],
                                left_col_count,
                                right_col_count,
                            );
                            result_rows.push(Row::from_values(values));
                        }
                    }
                }
            }
        }

        // Handle LEFT OUTER JOIN
        if join_type.contains("LEFT") || join_type.contains("FULL") {
            for (idx, was_matched) in left_matched.iter().enumerate() {
                if !was_matched {
                    let values = combine_rows_with_nulls(
                        &left_rows[idx],
                        left_col_count,
                        right_col_count,
                        true,
                    );
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        // Handle RIGHT OUTER JOIN
        if join_type.contains("RIGHT") || join_type.contains("FULL") {
            for (idx, was_matched) in right_matched.iter().enumerate() {
                if !was_matched {
                    let values = combine_rows_with_nulls(
                        &right_rows[idx],
                        right_col_count,
                        left_col_count,
                        false,
                    );
                    result_rows.push(Row::from_values(values));
                }
            }
        }

        Ok(result_rows)
    }

    /// Index Nested Loop Join - uses index on inner table for O(N * log M) lookups.
    ///
    /// This is optimal when:
    /// - The inner (right) table has an index on the join key column
    /// - The outer (left) table is small or has good selectivity
    ///
    /// For each row in the outer table, we use the index/PK to find matching rows
    /// in the inner table, avoiding a full scan of the inner table.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_index_nested_loop_join(
        &self,
        outer_rows: &[Row],
        inner_table: &dyn crate::storage::traits::Table,
        lookup_strategy: &IndexLookupStrategy,
        outer_key_idx: usize,
        residual_filter: Option<&super::expression::JoinFilter>,
        join_type: &str,
        outer_col_count: usize,
        inner_col_count: usize,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        use crate::core::Value;
        use crate::storage::expression::ConstBoolExpr;

        let is_inner_join = !join_type.contains("LEFT")
            && !join_type.contains("RIGHT")
            && !join_type.contains("FULL");
        let effective_limit = if is_inner_join { limit } else { None };

        let mut result_rows = Vec::new();
        let true_expr = ConstBoolExpr::true_expr();

        'outer: for outer_row in outer_rows {
            let mut matched = false;

            // Get the join key value from the outer row
            let key_value = match outer_row.get(outer_key_idx) {
                Some(v) if !v.is_null() => v,
                _ => {
                    // NULL key - no match possible (NULL != NULL in SQL)
                    if join_type.contains("LEFT") || join_type.contains("FULL") {
                        let values = combine_rows_with_nulls(
                            outer_row,
                            outer_col_count,
                            inner_col_count,
                            true,
                        );
                        result_rows.push(Row::from_values(values));
                    }
                    continue;
                }
            };

            // Find matching row IDs based on lookup strategy
            let row_ids: Vec<i64> = match lookup_strategy {
                IndexLookupStrategy::SecondaryIndex(index) => {
                    // Use secondary index to find matching row IDs
                    index.get_row_ids_equal(std::slice::from_ref(key_value))
                }
                IndexLookupStrategy::PrimaryKey => {
                    // For PK lookup, the key_value IS the row_id
                    match key_value {
                        Value::Integer(id) => vec![*id],
                        Value::Float(f) => vec![*f as i64],
                        _ => vec![], // Non-numeric PK values can't match
                    }
                }
            };

            if !row_ids.is_empty() {
                // Fetch matching rows from inner table
                let inner_rows = inner_table.fetch_rows_by_ids(&row_ids, &true_expr);

                for (_row_id, inner_row) in inner_rows {
                    // Apply residual filter if present
                    let passes_filter = if let Some(filter) = residual_filter {
                        filter.matches(outer_row, &inner_row)
                    } else {
                        true
                    };

                    if passes_filter {
                        matched = true;
                        let values =
                            combine_rows(outer_row, &inner_row, outer_col_count, inner_col_count);
                        result_rows.push(Row::from_values(values));

                        if let Some(lim) = effective_limit {
                            if result_rows.len() >= lim as usize {
                                break 'outer;
                            }
                        }
                    }
                }
            }

            // Handle LEFT OUTER JOIN - emit outer row with NULLs if no match
            if !matched && (join_type.contains("LEFT") || join_type.contains("FULL")) {
                let values =
                    combine_rows_with_nulls(outer_row, outer_col_count, inner_col_count, true);
                result_rows.push(Row::from_values(values));
            }
        }

        // Note: RIGHT/FULL OUTER JOIN would require tracking all matched inner rows,
        // which defeats the purpose of index nested loop. For those cases, fall back
        // to hash join or nested loop join.

        Ok(result_rows)
    }

    /// Get estimated cardinalities for join inputs using table statistics.
    pub(crate) fn get_join_estimates(
        &self,
        left_expr: &Expression,
        right_expr: &Expression,
        actual_left: u64,
        actual_right: u64,
    ) -> (u64, u64) {
        let estimated_left = self
            .estimate_table_expression_rows(left_expr)
            .unwrap_or(actual_left);
        let estimated_right = self
            .estimate_table_expression_rows(right_expr)
            .unwrap_or(actual_right);

        (estimated_left, estimated_right)
    }

    /// Estimate row count for a table expression using statistics.
    pub(crate) fn estimate_table_expression_rows(&self, expr: &Expression) -> Option<u64> {
        match expr {
            Expression::TableSource(table_source) => {
                let table_name = &table_source.name.value_lower;
                self.get_query_planner()
                    .get_table_stats(table_name)
                    .map(|stats| stats.row_count)
            }
            Expression::SubquerySource(_) => None,
            Expression::JoinSource(_) => None,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_rows() {
        let left = Row::from_values(vec![Value::Integer(1), Value::Text("a".into())]);
        let right = Row::from_values(vec![Value::Integer(2)]);

        let combined = combine_rows(&left, &right, 2, 1);
        assert_eq!(combined.len(), 3);
        assert_eq!(combined[0], Value::Integer(1));
        assert_eq!(combined[2], Value::Integer(2));
    }

    #[test]
    fn test_combine_rows_with_nulls_left() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let values = combine_rows_with_nulls(&row, 1, 2, true);

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], Value::Integer(1));
        assert!(values[1].is_null());
        assert!(values[2].is_null());
    }

    #[test]
    fn test_combine_rows_with_nulls_right() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let values = combine_rows_with_nulls(&row, 1, 2, false);

        assert_eq!(values.len(), 3);
        assert!(values[0].is_null());
        assert!(values[1].is_null());
        assert_eq!(values[2], Value::Integer(1));
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Integer(5), &Value::Integer(5)));
        assert!(!values_equal(&Value::Integer(5), &Value::Integer(6)));
        assert!(values_equal(&Value::Float(1.0), &Value::Integer(1)));
        assert!(!values_equal(
            &Value::null_unknown(),
            &Value::null_unknown()
        ));
    }

    #[test]
    fn test_compare_values() {
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Integer(2)),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&Value::Integer(2), &Value::Integer(1)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Integer(1)),
            Ordering::Equal
        );
    }

    #[test]
    fn test_is_sorted_on_keys() {
        let rows = vec![
            Row::from_values(vec![Value::Integer(1)]),
            Row::from_values(vec![Value::Integer(2)]),
            Row::from_values(vec![Value::Integer(3)]),
        ];
        assert!(is_sorted_on_keys(&rows, &[0]));

        let unsorted = vec![
            Row::from_values(vec![Value::Integer(3)]),
            Row::from_values(vec![Value::Integer(1)]),
            Row::from_values(vec![Value::Integer(2)]),
        ];
        assert!(!is_sorted_on_keys(&unsorted, &[0]));
    }
}
