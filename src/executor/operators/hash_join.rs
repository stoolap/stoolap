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

//! Streaming hash join operator.
//!
//! This operator implements hash join with the following key optimizations:
//!
//! 1. **Streaming Probe Side**: Only the build side is materialized.
//!    The probe side streams through without full materialization.
//!
//! 2. **Pre-allocated Hash Table**: The hash table is sized upfront
//!    based on build side cardinality, avoiding resizing.
//!
//! 3. **Zero-Copy Output**: Uses CompositeRow to combine rows without
//!    cloning values until final materialization is needed.
//!
//! # Join Types
//!
//! - INNER: Only matching rows
//! - LEFT OUTER: All left rows, matched right or NULLs
//! - RIGHT OUTER: All right rows, matched left or NULLs
//! - FULL OUTER: All rows from both sides

use std::sync::Arc;

use crate::core::{Result, Row, Value};
use crate::executor::hash_table::{hash_row_keys, verify_key_equality, JoinHashTable};
use crate::executor::operator::{ColumnInfo, Operator, RowRef};

/// Which side of the join to use as the build side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSide {
    /// Use left side as build (right as probe)
    Left,
    /// Use right side as build (left as probe)
    Right,
}

/// Type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// INNER JOIN - only matching rows
    Inner,
    /// LEFT OUTER JOIN - all left rows
    Left,
    /// RIGHT OUTER JOIN - all right rows
    Right,
    /// FULL OUTER JOIN - all rows from both sides
    Full,
    /// CROSS JOIN - cartesian product
    Cross,
    /// SEMI JOIN - return left rows that have at least one match (for EXISTS)
    Semi,
    /// ANTI JOIN - return left rows that have NO matches (for NOT EXISTS)
    Anti,
}

impl JoinType {
    /// Parse join type from string (as used in parser AST).
    pub fn parse(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        if s_lower.contains("left") {
            JoinType::Left
        } else if s_lower.contains("right") {
            JoinType::Right
        } else if s_lower.contains("full") {
            JoinType::Full
        } else if s_lower.contains("cross") {
            JoinType::Cross
        } else if s_lower.contains("semi") {
            JoinType::Semi
        } else if s_lower.contains("anti") {
            JoinType::Anti
        } else {
            JoinType::Inner
        }
    }

    /// Alias for parse() - used by parallel.rs for compatibility.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self::parse(s)
    }

    /// Check if this join needs unmatched probe rows (NULL-extended).
    /// Used by parallel hash join.
    pub fn needs_unmatched_probe(&self, swapped: bool) -> bool {
        match self {
            JoinType::Inner | JoinType::Cross | JoinType::Semi => false,
            JoinType::Anti => !swapped, // ANTI: unmatched probe rows (when not swapped)
            JoinType::Left => !swapped, // LEFT JOIN: unmatched left (probe when not swapped)
            JoinType::Right => swapped, // RIGHT JOIN: unmatched right (probe when swapped)
            JoinType::Full => true,     // FULL JOIN: always needs unmatched rows
        }
    }

    /// Check if this join needs unmatched build rows (NULL-extended).
    /// Used by parallel hash join.
    pub fn needs_unmatched_build(&self, swapped: bool) -> bool {
        match self {
            JoinType::Inner | JoinType::Cross | JoinType::Semi | JoinType::Anti => false,
            JoinType::Left => swapped, // LEFT JOIN: unmatched left (build when swapped)
            JoinType::Right => !swapped, // RIGHT JOIN: unmatched right (build when not swapped)
            JoinType::Full => true,    // FULL JOIN: always needs unmatched rows
        }
    }

    /// Check if this is a semi-join (EXISTS semantics).
    pub fn is_semi(&self) -> bool {
        matches!(self, JoinType::Semi)
    }

    /// Check if this is an anti-join (NOT EXISTS semantics).
    pub fn is_anti(&self) -> bool {
        matches!(self, JoinType::Anti)
    }
}

/// Streaming hash join operator.
///
/// The join proceeds in two phases:
///
/// 1. **Build Phase** (in `open()`):
///    - Materialize the build side (smaller side)
///    - Build hash table on join keys
///
/// 2. **Probe Phase** (in `next()`):
///    - Stream through probe side one row at a time
///    - Lookup matches in hash table
///    - Return combined rows
///
/// For OUTER joins, additional tracking is used to ensure unmatched
/// rows are returned with NULL padding.
pub struct HashJoinOperator {
    // Input operators
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,

    // Join configuration
    join_type: JoinType,
    build_side: JoinSide,
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,

    // Build phase state (populated in open())
    // Uses Arc<Vec<Row>> to enable zero-copy sharing with CTE results.
    // When dropped, only decrements refcount (O(1)) instead of deallocating rows.
    build_rows: Arc<Vec<Row>>,
    hash_table: Option<JoinHashTable>,

    // Output schema
    schema: Vec<ColumnInfo>,
    left_col_count: usize,
    right_col_count: usize,

    // Probe phase state
    current_probe_row: Option<Row>,
    current_probe_hash: u64,
    current_match_idx: usize,
    current_matches: Vec<usize>,
    probe_had_match: bool,

    // For OUTER joins: track which build rows were matched
    build_matched: Vec<bool>,
    returning_unmatched_build: bool,
    unmatched_build_idx: usize,

    // For self-join optimization
    is_self_join: bool,
    self_join_probe_idx: usize,

    // State tracking
    opened: bool,
    probe_exhausted: bool,
}

impl HashJoinOperator {
    /// Create a new hash join operator.
    ///
    /// # Arguments
    /// * `left` - Left input operator
    /// * `right` - Right input operator
    /// * `join_type` - Type of join (INNER, LEFT, RIGHT, FULL)
    /// * `left_key_indices` - Column indices for left join keys
    /// * `right_key_indices` - Column indices for right join keys
    /// * `build_side` - Which side to use as build (typically smaller)
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
        build_side: JoinSide,
    ) -> Self {
        // Build schema
        // For Semi/Anti joins, only return left (probe) columns
        let mut schema = Vec::new();
        if join_type.is_semi() || join_type.is_anti() {
            schema.extend(left.schema().iter().cloned());
        } else {
            schema.extend(left.schema().iter().cloned());
            schema.extend(right.schema().iter().cloned());
        }

        let left_col_count = left.schema().len();
        let right_col_count = right.schema().len();

        Self {
            left,
            right,
            join_type,
            build_side,
            left_key_indices,
            right_key_indices,
            build_rows: Arc::new(Vec::new()),
            hash_table: None,
            schema,
            left_col_count,
            right_col_count,
            current_probe_row: None,
            current_probe_hash: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            probe_had_match: false,
            build_matched: Vec::new(),
            returning_unmatched_build: false,
            unmatched_build_idx: 0,
            is_self_join: false,
            self_join_probe_idx: 0,
            opened: false,
            probe_exhausted: false,
        }
    }

    /// Create a hash join operator with pre-built hash table and rows.
    ///
    /// This avoids the build phase in `open()` since the hash table is already
    /// constructed. Used by streaming joins where hash table and bloom filter
    /// are built together in a single pass for efficiency.
    ///
    /// # Arguments
    /// * `probe` - Probe side operator (will be iterated during join)
    /// * `build_rows` - Pre-materialized build side rows (Arc for zero-copy sharing)
    /// * `hash_table` - Pre-built hash table for build side
    /// * `join_type` - Type of join
    /// * `probe_key_indices` - Key indices for probe side
    /// * `build_key_indices` - Key indices for build side
    /// * `build_is_left` - Whether build side is left (for schema ordering)
    pub fn with_prebuilt(
        probe: Box<dyn Operator>,
        build_rows: Arc<Vec<Row>>,
        hash_table: crate::executor::hash_table::JoinHashTable,
        join_type: JoinType,
        probe_key_indices: Vec<usize>,
        build_key_indices: Vec<usize>,
        build_is_left: bool,
    ) -> Self {
        let probe_col_count = probe.schema().len();
        let build_col_count = if build_rows.is_empty() {
            0
        } else {
            build_rows[0].len()
        };

        // Build schema based on build side position
        let mut schema = Vec::new();
        let (left_col_count, right_col_count) = if build_is_left {
            // Build is left: [build_cols, probe_cols]
            for i in 0..build_col_count {
                schema.push(ColumnInfo::new(format!("build_{}", i)));
            }
            schema.extend(probe.schema().iter().cloned());
            (build_col_count, probe_col_count)
        } else {
            // Build is right: [probe_cols, build_cols]
            schema.extend(probe.schema().iter().cloned());
            for i in 0..build_col_count {
                schema.push(ColumnInfo::new(format!("build_{}", i)));
            }
            (probe_col_count, build_col_count)
        };

        let (left_key_indices, right_key_indices, build_side) = if build_is_left {
            (build_key_indices, probe_key_indices, JoinSide::Left)
        } else {
            (probe_key_indices, build_key_indices, JoinSide::Right)
        };

        // Track matched builds for OUTER joins
        let build_matched = if matches!(join_type, JoinType::Full)
            || (matches!(join_type, JoinType::Left) && build_is_left)
            || (matches!(join_type, JoinType::Right) && !build_is_left)
        {
            vec![false; build_rows.len()]
        } else {
            Vec::new()
        };

        // Store probe operator in the non-build side slot
        let (left, right) = if build_is_left {
            // Build is left, probe is right
            (
                Box::new(crate::executor::operator::EmptyOperator::new()) as Box<dyn Operator>,
                probe,
            )
        } else {
            // Build is right, probe is left
            (
                probe,
                Box::new(crate::executor::operator::EmptyOperator::new()) as Box<dyn Operator>,
            )
        };

        Self {
            left,
            right,
            join_type,
            build_side,
            left_key_indices,
            right_key_indices,
            build_rows,
            hash_table: Some(hash_table),
            schema,
            left_col_count,
            right_col_count,
            current_probe_row: None,
            current_probe_hash: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            probe_had_match: false,
            build_matched,
            returning_unmatched_build: false,
            unmatched_build_idx: 0,
            is_self_join: false,
            self_join_probe_idx: 0,
            opened: false,
            probe_exhausted: false,
        }
    }

    /// Create an optimized self-join operator.
    ///
    /// For self-joins (t1 JOIN t1), this avoids scanning the table twice
    /// by reusing the same materialized data for both build and probe.
    pub fn self_join(
        input: Box<dyn Operator>,
        join_type: JoinType,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
    ) -> Self {
        // For self-join, schema is input schema duplicated
        let mut schema = Vec::new();
        schema.extend(input.schema().iter().cloned());
        schema.extend(input.schema().iter().cloned());

        let col_count = input.schema().len();

        // We'll use left as the input, right will be unused
        Self {
            left: input,
            right: Box::new(crate::executor::operator::EmptyOperator::new()),
            join_type,
            build_side: JoinSide::Left, // Build from the single input
            left_key_indices,
            right_key_indices,
            build_rows: Arc::new(Vec::new()),
            hash_table: None,
            schema,
            left_col_count: col_count,
            right_col_count: col_count,
            current_probe_row: None,
            current_probe_hash: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            probe_had_match: false,
            build_matched: Vec::new(),
            returning_unmatched_build: false,
            unmatched_build_idx: 0,
            is_self_join: true,
            self_join_probe_idx: 0,
            opened: false,
            probe_exhausted: false,
        }
    }

    /// Get the key indices based on which side is build vs probe.
    fn build_key_indices(&self) -> &[usize] {
        match self.build_side {
            JoinSide::Left => &self.left_key_indices,
            JoinSide::Right => &self.right_key_indices,
        }
    }

    fn probe_key_indices(&self) -> &[usize] {
        match self.build_side {
            JoinSide::Left => &self.right_key_indices,
            JoinSide::Right => &self.left_key_indices,
        }
    }

    /// Create a NULL row for the build side (used in OUTER joins).
    #[inline]
    fn null_build_row(&self) -> Row {
        let count = match self.build_side {
            JoinSide::Left => self.left_col_count,
            JoinSide::Right => self.right_col_count,
        };
        Row::from_values(vec![Value::null_unknown(); count])
    }

    /// Create a NULL row for the probe side (used in OUTER joins).
    #[inline]
    fn null_probe_row(&self) -> Row {
        let count = match self.build_side {
            JoinSide::Left => self.right_col_count,
            JoinSide::Right => self.left_col_count,
        };
        Row::from_values(vec![Value::null_unknown(); count])
    }

    /// Combine probe row with a build row by index, avoiding clone.
    /// Uses SharedBuildCompositeRow to reference the build row via Arc.
    #[inline]
    fn combine_rows_shared(&self, probe_row: Row, build_idx: usize) -> RowRef {
        // probe_is_left determines output column order:
        // - true: output = [probe, build]
        // - false: output = [build, probe]
        let probe_is_left = self.build_side == JoinSide::Right;

        RowRef::shared_build_composite(
            probe_row,
            Arc::clone(&self.build_rows),
            build_idx,
            probe_is_left,
        )
    }

    /// Combine probe and build rows into a RowRef without allocation.
    /// Uses CompositeRow to defer materialization until needed.
    /// Used for OUTER join unmatched rows where we need an actual null row.
    #[inline]
    fn combine_rows_ref(&self, probe_row: Row, build_row: Row) -> RowRef {
        match self.build_side {
            JoinSide::Left => {
                // Build is left, probe is right
                // Output: [build_row, probe_row] = [left, right]
                RowRef::Composite(crate::executor::operator::CompositeRow::new(
                    build_row, probe_row,
                ))
            }
            JoinSide::Right => {
                // Build is right, probe is left
                // Output: [probe_row, build_row] = [left, right]
                RowRef::Composite(crate::executor::operator::CompositeRow::new(
                    probe_row, build_row,
                ))
            }
        }
    }

    /// Get the next probe row (from probe operator or self-join iteration).
    fn next_probe_row(&mut self) -> Result<Option<Row>> {
        if self.is_self_join {
            // For self-join, iterate over the materialized build rows
            if self.self_join_probe_idx >= self.build_rows.len() {
                return Ok(None);
            }
            let row = self.build_rows[self.self_join_probe_idx].clone();
            self.self_join_probe_idx += 1;
            Ok(Some(row))
        } else {
            // Normal case: get from probe operator
            let probe_op = match self.build_side {
                JoinSide::Left => &mut self.right,
                JoinSide::Right => &mut self.left,
            };

            match probe_op.next()? {
                Some(row_ref) => Ok(Some(row_ref.into_owned())),
                None => Ok(None),
            }
        }
    }
}

impl Operator for HashJoinOperator {
    fn open(&mut self) -> Result<()> {
        // Check if hash table was pre-built (via with_prebuilt constructor)
        if self.hash_table.is_some() {
            // Pre-built case: only need to open the probe side
            // Build side is already materialized
            let probe_op = match self.build_side {
                JoinSide::Left => &mut self.right,
                JoinSide::Right => &mut self.left,
            };
            probe_op.open()?;
            self.opened = true;
            return Ok(());
        }

        // Standard case: open both inputs and build hash table
        self.left.open()?;
        if !self.is_self_join {
            self.right.open()?;
        }

        // Materialize build side
        let build_op = match self.build_side {
            JoinSide::Left => &mut self.left,
            JoinSide::Right => &mut self.right,
        };

        // Collect all build rows
        let mut build_rows = Vec::new();
        while let Some(row_ref) = build_op.next()? {
            build_rows.push(row_ref.into_owned());
        }

        // Build hash table
        let build_key_indices = self.build_key_indices().to_vec();
        let hash_table = JoinHashTable::build(&build_rows, &build_key_indices);

        // Track which build rows match for OUTER joins that need unmatched BUILD rows:
        // - FULL: always need unmatched rows from both sides
        // - LEFT with build_side=Left: unmatched LEFT (build) rows need NULLs
        // - RIGHT with build_side=Right: unmatched RIGHT (build) rows need NULLs
        let needs_build_tracking = matches!(self.join_type, JoinType::Full)
            || (matches!(self.join_type, JoinType::Left) && self.build_side == JoinSide::Left)
            || (matches!(self.join_type, JoinType::Right) && self.build_side == JoinSide::Right)
            || (self.is_self_join && !matches!(self.join_type, JoinType::Inner));
        if needs_build_tracking {
            self.build_matched = vec![false; build_rows.len()];
        }

        // Wrap in Arc for zero-copy drop (only refcount decrement, not deallocation)
        self.build_rows = Arc::new(build_rows);
        self.hash_table = Some(hash_table);
        self.opened = true;

        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Err(crate::core::Error::internal(
                "HashJoinOperator::next called before open",
            ));
        }

        // If we're returning unmatched build rows (for FULL/RIGHT OUTER)
        if self.returning_unmatched_build {
            while self.unmatched_build_idx < self.build_rows.len() {
                let idx = self.unmatched_build_idx;
                self.unmatched_build_idx += 1;

                if !self.build_matched[idx] {
                    let build_row = self.build_rows[idx].clone();
                    let null_probe = self.null_probe_row();
                    return Ok(Some(self.combine_rows_ref(null_probe, build_row)));
                }
            }
            return Ok(None);
        }

        loop {
            // Try to return next match for current probe row
            while self.current_match_idx < self.current_matches.len() {
                let build_idx = self.current_matches[self.current_match_idx];
                self.current_match_idx += 1;

                let build_row = &self.build_rows[build_idx];

                // Verify actual key equality (handle hash collisions)
                if verify_key_equality(
                    self.current_probe_row.as_ref().unwrap(),
                    build_row,
                    self.probe_key_indices(),
                    self.build_key_indices(),
                ) {
                    self.probe_had_match = true;

                    // SEMI JOIN: Return probe row only (no build columns), then skip remaining matches
                    if self.join_type.is_semi() {
                        let probe_row = self.current_probe_row.take().unwrap();
                        // Skip remaining matches for this probe row
                        self.current_match_idx = self.current_matches.len();
                        return Ok(Some(RowRef::Owned(probe_row)));
                    }

                    // ANTI JOIN: Found a match, so this probe row should NOT be returned
                    // Just skip remaining matches and move to next probe row
                    if self.join_type.is_anti() {
                        self.current_match_idx = self.current_matches.len();
                        continue;
                    }

                    // Mark build row as matched (for OUTER joins)
                    if !self.build_matched.is_empty() {
                        self.build_matched[build_idx] = true;
                    }

                    // Use SharedBuildComposite to avoid cloning build_row entirely.
                    // The Arc<Vec<Row>> is shared, only refcount is incremented.
                    //
                    // For probe row: if this is the last match, take ownership instead of cloning.
                    // This optimizes the common 1:1 join case where each probe has exactly one match.
                    let probe_row = if self.current_match_idx >= self.current_matches.len() {
                        // No more matches - take ownership
                        self.current_probe_row.take().unwrap()
                    } else {
                        // More potential matches - clone
                        self.current_probe_row.as_ref().unwrap().clone()
                    };
                    return Ok(Some(self.combine_rows_shared(probe_row, build_idx)));
                }
            }

            // Handle unmatched probe row
            // - ANTI JOIN: Return probe row when NO match found
            // - OUTER JOINs: Return probe row with NULL build columns
            if self.join_type.is_anti() && !self.probe_had_match {
                if let Some(probe_row) = self.current_probe_row.take() {
                    return Ok(Some(RowRef::Owned(probe_row)));
                }
            }

            // Handle unmatched probe row for OUTER joins
            // Output unmatched probe rows only when probe side needs "all rows":
            // - FULL: all rows from both sides
            // - RIGHT with build_side=Left: probe=right, need all right rows
            // - LEFT with build_side=Right: probe=left, need all left rows
            let needs_unmatched_probe = matches!(self.join_type, JoinType::Full)
                || (matches!(self.join_type, JoinType::Right) && self.build_side == JoinSide::Left)
                || (matches!(self.join_type, JoinType::Left) && self.build_side == JoinSide::Right);

            if needs_unmatched_probe && !self.probe_had_match {
                if let Some(probe_row) = self.current_probe_row.take() {
                    let null_build = self.null_build_row();
                    return Ok(Some(self.combine_rows_ref(probe_row, null_build)));
                }
            }

            // Get next probe row (must be done before borrowing hash_table)
            let next_probe = self.next_probe_row()?;
            match next_probe {
                Some(probe_row) => {
                    // Compute hash for probe row (no allocation - use slice directly)
                    let probe_key_indices = self.probe_key_indices();
                    let hash = hash_row_keys(&probe_row, probe_key_indices);

                    // Find matching build rows - reuse Vec to avoid allocation
                    let hash_table = self.hash_table.as_ref().unwrap();
                    self.current_matches.clear();
                    self.current_matches.extend(hash_table.probe(hash));

                    self.current_probe_row = Some(probe_row);
                    self.current_probe_hash = hash;
                    self.current_match_idx = 0;
                    self.probe_had_match = false;
                }
                None => {
                    // Probe side exhausted
                    self.probe_exhausted = true;

                    // Return unmatched build rows for OUTER joins where build side
                    // corresponds to the "all rows" side of the join:
                    // - FULL: all rows from both sides
                    // - LEFT with build_side=Left: all left (build) rows
                    // - RIGHT with build_side=Right: all right (build) rows
                    if !self.build_matched.is_empty() {
                        self.returning_unmatched_build = true;
                        self.unmatched_build_idx = 0;
                        // Recursive call to handle unmatched build rows
                        return self.next();
                    }

                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.left.close()?;
        if !self.is_self_join {
            self.right.close()?;
        }
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        // Rough estimate: min of both sides (for INNER)
        // Could be refined with statistics
        let left_est = self.left.estimated_rows()?;
        let right_est = self.right.estimated_rows()?;

        Some(match self.join_type {
            JoinType::Inner => left_est.min(right_est),
            JoinType::Left => left_est,
            JoinType::Right => right_est,
            JoinType::Full => left_est + right_est,
            JoinType::Cross => left_est * right_est,
            JoinType::Semi => left_est.min(right_est), // At most all left rows
            JoinType::Anti => left_est,                // At most all left rows
        })
    }

    fn name(&self) -> &str {
        match self.join_type {
            JoinType::Inner => "HashJoin (INNER)",
            JoinType::Left => "HashJoin (LEFT)",
            JoinType::Right => "HashJoin (RIGHT)",
            JoinType::Full => "HashJoin (FULL)",
            JoinType::Cross => "HashJoin (CROSS)",
            JoinType::Semi => "HashJoin (SEMI)",
            JoinType::Anti => "HashJoin (ANTI)",
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
    fn test_inner_join() {
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["id", "value"],
        );
        let right = make_operator(vec![vec![1, 100], vec![3, 300]], vec!["id", "data"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Inner,
            vec![0], // left key: id
            vec![0], // right key: id
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Should have 2 matches: id=1 and id=3
        assert_eq!(results.len(), 2);

        // Verify first match (id=1)
        let row1 = &results[0];
        assert_eq!(row1.get(0), Some(&Value::integer(1)));
        assert_eq!(row1.get(1), Some(&Value::integer(10)));
        assert_eq!(row1.get(2), Some(&Value::integer(1)));
        assert_eq!(row1.get(3), Some(&Value::integer(100)));
    }

    #[test]
    fn test_left_join() {
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["id", "value"],
        );
        let right = make_operator(vec![vec![1, 100]], vec!["id", "data"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Left,
            vec![0],
            vec![0],
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Should have 3 rows: id=1 matched, id=2 and id=3 with NULLs
        assert_eq!(results.len(), 3);

        // Check that id=2 has NULLs on right side
        let row2 = results
            .iter()
            .find(|r| r.get(0) == Some(&Value::integer(2)))
            .unwrap();
        assert!(row2.get(2).unwrap().is_null());
        assert!(row2.get(3).unwrap().is_null());
    }

    #[test]
    fn test_self_join() {
        let input = make_operator(
            vec![vec![1, 10], vec![2, 10], vec![3, 20]],
            vec!["id", "age"],
        );

        // Self-join on age (find pairs with same age)
        let mut join = HashJoinOperator::self_join(
            input,
            JoinType::Inner,
            vec![1], // left key: age
            vec![1], // right key: age
        );

        let results = collect_results(&mut join).unwrap();

        // id=1 and id=2 both have age=10, so we get:
        // (1,10) x (1,10), (1,10) x (2,10), (2,10) x (1,10), (2,10) x (2,10)
        // = 4 matches for age=10
        // id=3 has age=20, matches only itself = 1 match
        // Total = 5
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_empty_build() {
        let left = make_operator(vec![vec![1, 10], vec![2, 20]], vec!["id", "value"]);
        let right = make_operator(vec![], vec!["id", "data"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Inner,
            vec![0],
            vec![0],
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_multi_key_join() {
        let left = make_operator(
            vec![vec![1, 10, 100], vec![1, 20, 200], vec![2, 10, 300]],
            vec!["a", "b", "val"],
        );
        let right = make_operator(
            vec![vec![1, 10, 1000], vec![1, 20, 2000]],
            vec!["a", "b", "data"],
        );

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Inner,
            vec![0, 1], // left keys: a, b
            vec![0, 1], // right keys: a, b
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Should match (1,10) and (1,20)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_semi_join() {
        // Left: users with id 1, 2, 3
        let left = make_operator(
            vec![vec![1, 100], vec![2, 200], vec![3, 300]],
            vec!["id", "value"],
        );
        // Right: orders for users 1 and 3 (user 1 has 2 orders)
        let right = make_operator(
            vec![vec![1, 10], vec![1, 20], vec![3, 30]],
            vec!["user_id", "order_id"],
        );

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Semi,
            vec![0], // left key: id
            vec![0], // right key: user_id
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Semi join: return users who have at least one order
        // User 1 has 2 orders but should only appear once
        // User 2 has no orders - should NOT appear
        // User 3 has 1 order - should appear
        assert_eq!(results.len(), 2);

        // Schema should only have left columns
        assert_eq!(join.schema().len(), 2);

        // Verify we got users 1 and 3
        let ids: Vec<i64> = results
            .iter()
            .map(|r| r.get(0).unwrap().as_int64().unwrap())
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_anti_join() {
        // Left: users with id 1, 2, 3
        let left = make_operator(
            vec![vec![1, 100], vec![2, 200], vec![3, 300]],
            vec!["id", "value"],
        );
        // Right: orders for users 1 and 3
        let right = make_operator(vec![vec![1, 10], vec![3, 30]], vec!["user_id", "order_id"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Anti,
            vec![0], // left key: id
            vec![0], // right key: user_id
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Anti join: return users who have NO orders
        // User 1 has orders - should NOT appear
        // User 2 has no orders - should appear
        // User 3 has orders - should NOT appear
        assert_eq!(results.len(), 1);

        // Schema should only have left columns
        assert_eq!(join.schema().len(), 2);

        // Verify we only got user 2
        let row = &results[0];
        assert_eq!(row.get(0), Some(&Value::integer(2)));
        assert_eq!(row.get(1), Some(&Value::integer(200)));
    }

    #[test]
    fn test_anti_join_empty_right() {
        // Left: users with id 1, 2, 3
        let left = make_operator(
            vec![vec![1, 100], vec![2, 200], vec![3, 300]],
            vec!["id", "value"],
        );
        // Right: no orders
        let right = make_operator(vec![], vec!["user_id", "order_id"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Anti,
            vec![0],
            vec![0],
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Anti join with empty right: all left rows should be returned
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_semi_join_empty_right() {
        // Left: users with id 1, 2, 3
        let left = make_operator(
            vec![vec![1, 100], vec![2, 200], vec![3, 300]],
            vec!["id", "value"],
        );
        // Right: no orders
        let right = make_operator(vec![], vec!["user_id", "order_id"]);

        let mut join = HashJoinOperator::new(
            left,
            right,
            JoinType::Semi,
            vec![0],
            vec![0],
            JoinSide::Right,
        );

        let results = collect_results(&mut join).unwrap();

        // Semi join with empty right: no left rows should be returned
        assert_eq!(results.len(), 0);
    }
}
