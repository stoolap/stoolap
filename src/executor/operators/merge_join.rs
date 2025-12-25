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

//! Merge Join Operator for pre-sorted inputs.
//!
//! This operator implements merge join with O(N+M) complexity when both
//! inputs are already sorted on the join keys. It's optimal for:
//! - Joining tables that are physically sorted (e.g., clustered index)
//! - Joining results of ORDER BY queries
//! - Self-joins on sorted columns

use std::cmp::Ordering;

use crate::core::{Result, Row, Value};
use crate::executor::operator::{ColumnInfo, Operator, RowRef};
use crate::executor::utils::{combine_rows, compare_values};

use super::hash_join::JoinType;

/// Merge Join Operator for pre-sorted inputs.
///
/// Both inputs must be sorted on their respective join keys.
/// The operator performs a single pass through both inputs,
/// producing matches as they're found.
pub struct MergeJoinOperator {
    // Input operators
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,

    // Join configuration
    join_type: JoinType,
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,

    // Output schema
    schema: Vec<ColumnInfo>,
    left_col_count: usize,
    right_col_count: usize,

    // Materialized inputs (needed for merge join to handle duplicates)
    left_rows: Vec<Row>,
    right_rows: Vec<Row>,

    // Current position in merge
    left_idx: usize,
    right_idx: usize,

    // For handling duplicate key groups
    current_left_group_start: usize,
    current_right_group_start: usize,
    current_right_group_end: usize,
    current_left_in_group: usize,
    current_right_in_group: usize,
    in_cartesian_product: bool,

    // Track matched rows for OUTER joins
    right_matched: Vec<bool>,

    // Returning unmatched right rows phase
    returning_unmatched_right: bool,
    unmatched_right_idx: usize,

    // State
    opened: bool,
}

impl MergeJoinOperator {
    /// Create a new merge join operator.
    ///
    /// # Arguments
    /// * `left` - Left input operator (must be sorted on left_key_indices)
    /// * `right` - Right input operator (must be sorted on right_key_indices)
    /// * `join_type` - Type of join (INNER, LEFT, RIGHT, FULL - NOT Cross)
    /// * `left_key_indices` - Column indices for left join keys
    /// * `right_key_indices` - Column indices for right join keys
    ///
    /// # Panics
    /// Debug builds will panic if `join_type` is `JoinType::Cross`.
    /// Cross joins should use NestedLoopJoinOperator instead.
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
    ) -> Self {
        // Cross joins should use NestedLoop, not MergeJoin (no keys to merge on)
        debug_assert!(
            !matches!(join_type, JoinType::Cross),
            "MergeJoin cannot be used for CROSS JOIN - use NestedLoopJoin instead"
        );

        // Build combined schema
        let mut schema = Vec::new();
        schema.extend(left.schema().iter().cloned());
        schema.extend(right.schema().iter().cloned());

        let left_col_count = left.schema().len();
        let right_col_count = right.schema().len();

        Self {
            left,
            right,
            join_type,
            left_key_indices,
            right_key_indices,
            schema,
            left_col_count,
            right_col_count,
            left_rows: Vec::new(),
            right_rows: Vec::new(),
            left_idx: 0,
            right_idx: 0,
            current_left_group_start: 0,
            current_right_group_start: 0,
            current_right_group_end: 0,
            current_left_in_group: 0,
            current_right_in_group: 0,
            in_cartesian_product: false,
            right_matched: Vec::new(),
            returning_unmatched_right: false,
            unmatched_right_idx: 0,
            opened: false,
        }
    }

    /// Compare two rows on their respective join keys.
    fn compare_on_keys(&self, left: &Row, right: &Row) -> Ordering {
        for (li, ri) in self
            .left_key_indices
            .iter()
            .zip(self.right_key_indices.iter())
        {
            let lv = left.get(*li).cloned().unwrap_or(Value::null_unknown());
            let rv = right.get(*ri).cloned().unwrap_or(Value::null_unknown());

            // NULL comparison: NULLs sort last
            if lv.is_null() && rv.is_null() {
                continue;
            }
            if lv.is_null() {
                return Ordering::Greater;
            }
            if rv.is_null() {
                return Ordering::Less;
            }

            let cmp = compare_values(&lv, &rv);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Compare two rows from the same side on their keys.
    fn compare_same_side(&self, row1: &Row, row2: &Row, key_indices: &[usize]) -> Ordering {
        for &idx in key_indices {
            let v1 = row1.get(idx).cloned().unwrap_or(Value::null_unknown());
            let v2 = row2.get(idx).cloned().unwrap_or(Value::null_unknown());

            if v1.is_null() && v2.is_null() {
                continue;
            }
            if v1.is_null() {
                return Ordering::Greater;
            }
            if v2.is_null() {
                return Ordering::Less;
            }

            let cmp = compare_values(&v1, &v2);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Create a NULL row for the left side.
    fn null_left_row(&self) -> Row {
        Row::from_values(vec![Value::null_unknown(); self.left_col_count])
    }

    /// Create a NULL row for the right side.
    fn null_right_row(&self) -> Row {
        Row::from_values(vec![Value::null_unknown(); self.right_col_count])
    }

    /// Combine left and right rows into output row.
    fn combine(&self, left: &Row, right: &Row) -> Row {
        Row::from_values(combine_rows(
            left,
            right,
            self.left_col_count,
            self.right_col_count,
        ))
    }
}

impl Operator for MergeJoinOperator {
    fn open(&mut self) -> Result<()> {
        // Open and materialize both inputs
        self.left.open()?;
        self.right.open()?;

        // Materialize left
        while let Some(row_ref) = self.left.next()? {
            self.left_rows.push(row_ref.into_owned());
        }

        // Materialize right
        while let Some(row_ref) = self.right.next()? {
            self.right_rows.push(row_ref.into_owned());
        }

        // Initialize matched tracking for OUTER joins
        if matches!(self.join_type, JoinType::Right | JoinType::Full) {
            self.right_matched = vec![false; self.right_rows.len()];
        }

        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Err(crate::core::Error::internal(
                "MergeJoinOperator::next called before open",
            ));
        }

        let is_left_outer = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let is_right_outer = matches!(self.join_type, JoinType::Right | JoinType::Full);

        // Phase 2: Return unmatched right rows (for RIGHT/FULL OUTER)
        if self.returning_unmatched_right {
            while self.unmatched_right_idx < self.right_rows.len() {
                let idx = self.unmatched_right_idx;
                self.unmatched_right_idx += 1;

                if !self.right_matched[idx] {
                    let null_left = self.null_left_row();
                    let right_row = &self.right_rows[idx];
                    let combined = self.combine(&null_left, right_row);
                    return Ok(Some(RowRef::Owned(combined)));
                }
            }
            return Ok(None);
        }

        // Phase 1: Merge join with cartesian product for duplicate key groups
        loop {
            // If we're in a cartesian product of matching groups
            if self.in_cartesian_product {
                // Try to emit next pair from current groups
                if self.current_left_in_group < self.left_idx
                    && self.current_right_in_group < self.current_right_group_end
                {
                    let left_row = &self.left_rows[self.current_left_in_group];
                    let right_row = &self.right_rows[self.current_right_in_group];

                    // Mark right row as matched
                    if is_right_outer {
                        self.right_matched[self.current_right_in_group] = true;
                    }

                    self.current_right_in_group += 1;

                    // If we've exhausted right group, move to next left row
                    if self.current_right_in_group >= self.current_right_group_end {
                        self.current_left_in_group += 1;
                        self.current_right_in_group = self.current_right_group_start;

                        // If we've exhausted left group, exit cartesian product mode
                        if self.current_left_in_group >= self.left_idx {
                            self.in_cartesian_product = false;
                            self.right_idx = self.current_right_group_end;
                        }
                    }

                    let combined = self.combine(left_row, right_row);
                    return Ok(Some(RowRef::Owned(combined)));
                }

                // Shouldn't reach here, but safety exit
                self.in_cartesian_product = false;
            }

            // Check if either side is exhausted
            if self.left_idx >= self.left_rows.len() {
                // Left exhausted - handle remaining right for RIGHT/FULL OUTER
                if is_right_outer && !self.returning_unmatched_right {
                    self.returning_unmatched_right = true;
                    self.unmatched_right_idx = self.right_idx;
                    return self.next();
                }
                return Ok(None);
            }

            if self.right_idx >= self.right_rows.len() {
                // Right exhausted - handle remaining left for LEFT/FULL OUTER
                if is_left_outer {
                    let left_row = &self.left_rows[self.left_idx];
                    self.left_idx += 1;
                    let null_right = self.null_right_row();
                    let combined = self.combine(left_row, &null_right);
                    return Ok(Some(RowRef::Owned(combined)));
                }

                // If RIGHT/FULL OUTER, switch to returning unmatched right
                if is_right_outer {
                    self.returning_unmatched_right = true;
                    self.unmatched_right_idx = 0;
                    return self.next();
                }

                return Ok(None);
            }

            let left_row = &self.left_rows[self.left_idx];
            let right_row = &self.right_rows[self.right_idx];

            match self.compare_on_keys(left_row, right_row) {
                Ordering::Less => {
                    // Left row has no match
                    self.left_idx += 1;
                    if is_left_outer {
                        let null_right = self.null_right_row();
                        let combined = self.combine(left_row, &null_right);
                        return Ok(Some(RowRef::Owned(combined)));
                    }
                }
                Ordering::Greater => {
                    // Right row has no match (mark as unmatched for later)
                    self.right_idx += 1;
                }
                Ordering::Equal => {
                    // Found match - find extent of matching groups
                    let left_start = self.left_idx;
                    let right_start = self.right_idx;

                    // Find all left rows with same key
                    while self.left_idx < self.left_rows.len()
                        && self.compare_same_side(
                            &self.left_rows[self.left_idx],
                            &self.left_rows[left_start],
                            &self.left_key_indices,
                        ) == Ordering::Equal
                    {
                        self.left_idx += 1;
                    }

                    // Find all right rows with same key
                    let mut right_end = self.right_idx;
                    while right_end < self.right_rows.len()
                        && self.compare_same_side(
                            &self.right_rows[right_end],
                            &self.right_rows[right_start],
                            &self.right_key_indices,
                        ) == Ordering::Equal
                    {
                        right_end += 1;
                    }

                    // Set up cartesian product iteration
                    self.current_left_group_start = left_start;
                    self.current_right_group_start = right_start;
                    self.current_right_group_end = right_end;
                    self.current_left_in_group = left_start;
                    self.current_right_in_group = right_start;
                    self.in_cartesian_product = true;

                    // Continue loop to emit first match
                }
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.left.close()?;
        self.right.close()?;
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        let left_est = self.left.estimated_rows()?;
        let right_est = self.right.estimated_rows()?;

        Some(match self.join_type {
            JoinType::Inner => left_est.min(right_est),
            JoinType::Left => left_est,
            JoinType::Right => right_est,
            JoinType::Full => left_est + right_est,
            JoinType::Cross => left_est * right_est,
            JoinType::Semi => left_est.min(right_est),
            JoinType::Anti => left_est,
        })
    }

    fn name(&self) -> &str {
        match self.join_type {
            JoinType::Inner => "MergeJoin (INNER)",
            JoinType::Left => "MergeJoin (LEFT)",
            JoinType::Right => "MergeJoin (RIGHT)",
            JoinType::Full => "MergeJoin (FULL)",
            JoinType::Cross => "MergeJoin (CROSS)",
            JoinType::Semi => "MergeJoin (SEMI)",
            JoinType::Anti => "MergeJoin (ANTI)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::operator::MaterializedOperator;

    fn make_rows(data: Vec<Vec<i64>>) -> Vec<Row> {
        data.into_iter()
            .map(|vals| Row::from_values(vals.into_iter().map(Value::integer).collect()))
            .collect()
    }

    fn make_operator(data: Vec<Vec<i64>>, cols: Vec<&str>) -> Box<dyn Operator> {
        let rows = make_rows(data);
        let schema = cols.into_iter().map(ColumnInfo::new).collect();
        Box::new(MaterializedOperator::new(rows, schema))
    }

    fn collect_results(op: &mut dyn Operator) -> Result<Vec<Row>> {
        let mut results = Vec::new();
        op.open()?;
        while let Some(row_ref) = op.next()? {
            results.push(row_ref.into_owned());
        }
        op.close()?;
        Ok(results)
    }

    #[test]
    fn test_inner_merge_join() {
        // Both sides sorted on id
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["id", "value"],
        );
        let right = make_operator(vec![vec![1, 100], vec![3, 300]], vec!["id", "data"]);

        let mut join = MergeJoinOperator::new(
            left,
            right,
            JoinType::Inner,
            vec![0], // left key: id
            vec![0], // right key: id
        );

        let results = collect_results(&mut join).unwrap();

        // Should have 2 matches: id=1 and id=3
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_left_merge_join() {
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["id", "value"],
        );
        let right = make_operator(vec![vec![1, 100]], vec!["id", "data"]);

        let mut join = MergeJoinOperator::new(left, right, JoinType::Left, vec![0], vec![0]);

        let results = collect_results(&mut join).unwrap();

        // All 3 left rows should be preserved
        assert_eq!(results.len(), 3);

        // Check that id=2 and id=3 have NULLs on right side
        let row2 = results
            .iter()
            .find(|r| r.get(0) == Some(&Value::integer(2)))
            .unwrap();
        assert!(row2.get(2).unwrap().is_null());
    }

    #[test]
    fn test_merge_join_with_duplicates() {
        // Both sides have duplicate keys
        let left = make_operator(
            vec![vec![1, 10], vec![1, 11], vec![2, 20]],
            vec!["id", "value"],
        );
        let right = make_operator(
            vec![vec![1, 100], vec![1, 101], vec![2, 200]],
            vec!["id", "data"],
        );

        let mut join = MergeJoinOperator::new(left, right, JoinType::Inner, vec![0], vec![0]);

        let results = collect_results(&mut join).unwrap();

        // id=1: 2 left x 2 right = 4 matches
        // id=2: 1 left x 1 right = 1 match
        // Total = 5
        assert_eq!(results.len(), 5);
    }
}
