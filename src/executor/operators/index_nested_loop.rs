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

use crate::core::{Result, Row, Value};
use crate::executor::expression::JoinFilter;
use crate::executor::operator::{ColumnInfo, Operator, RowRef};
use crate::executor::utils::combine_rows_with_nulls;
use crate::storage::expression::ConstBoolExpr;
use crate::storage::traits::{Index, Table};

use super::hash_join::JoinType;

/// Index Nested Loop Join lookup strategy.
/// Determines how to find matching rows in the inner (right) table.
#[derive(Clone)]
pub enum IndexLookupStrategy {
    /// Use a secondary index for lookups (index.get_row_ids_equal)
    SecondaryIndex(Arc<dyn Index>),
    /// Use primary key lookup (direct row_id = value)
    /// In stoolap, PRIMARY KEY INTEGER values ARE the row_ids
    PrimaryKey,
}

/// Index Nested Loop Join Operator.
///
/// For each row in the outer input, looks up matching rows in the inner table
/// using an index. This avoids full table scans of the inner table.
pub struct IndexNestedLoopJoinOperator {
    // Outer input operator
    outer: Box<dyn Operator>,

    // Inner table (accessed via index)
    inner_table: Box<dyn Table>,

    // Join configuration
    join_type: JoinType,
    outer_key_idx: usize,
    lookup_strategy: IndexLookupStrategy,
    residual_filter: Option<JoinFilter>,

    // Output schema
    schema: Vec<ColumnInfo>,
    inner_col_count: usize,

    // Current state
    current_outer_row: Option<Row>,
    // Optimization: Store (id, row) to verify specific inner rows if needed, and match fetch_rows_by_ids_into signature
    current_inner_rows: Vec<(i64, Row)>,
    current_inner_idx: usize,
    outer_had_match: bool,

    // Optimization: Reusable buffer for row IDs to avoid allocation per outer row
    row_id_buffer: Vec<i64>,

    // Expression for fetching rows (always true - we apply residual separately)
    true_expr: ConstBoolExpr,

    // State tracking
    opened: bool,
    outer_exhausted: bool,
}

impl IndexNestedLoopJoinOperator {
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

        Self {
            outer,
            inner_table,
            join_type,
            outer_key_idx,
            lookup_strategy,
            residual_filter,
            schema,
            inner_col_count,
            current_outer_row: None,
            current_inner_rows: Vec::new(),
            current_inner_idx: 0,
            outer_had_match: false,
            // Pre-allocate buffer for typical number of matches (small)
            row_id_buffer: Vec::with_capacity(16),
            true_expr: ConstBoolExpr::true_expr(),
            opened: false,
            outer_exhausted: false,
        }
    }

    /// Create a NULL row for the inner side.
    fn null_inner_row(&self) -> Row {
        Row::from_values(vec![Value::null_unknown(); self.inner_col_count])
    }

    /// Combine outer and inner rows into output row.
    #[inline]
    fn combine(&self, outer: &Row, inner: &Row) -> Row {
        Row::from_combined(outer, inner)
    }

    /// Look up matching inner rows for the current outer row.
    /// Uses internal buffers to avoid allocations.
    fn lookup_inner_rows(&mut self, key_value: &Value) {
        // Clear buffers for reuse
        self.row_id_buffer.clear();
        self.current_inner_rows.clear();

        // Get row IDs based on lookup strategy into buffer
        match &self.lookup_strategy {
            IndexLookupStrategy::SecondaryIndex(index) => {
                index.get_row_ids_equal_into(
                    std::slice::from_ref(key_value),
                    &mut self.row_id_buffer,
                );
            }
            IndexLookupStrategy::PrimaryKey => {
                match key_value {
                    Value::Integer(id) => self.row_id_buffer.push(*id),
                    Value::Float(f) => self.row_id_buffer.push(*f as i64),
                    _ => {} // Non-numeric PK values can't match
                }
            }
        }

        if self.row_id_buffer.is_empty() {
            return;
        }

        // Fetch matching rows directly into inner_rows buffer
        self.inner_table.fetch_rows_by_ids_into(
            &self.row_id_buffer,
            &self.true_expr,
            &mut self.current_inner_rows,
        );
    }

    /// Advance to the next outer row and lookup matching inner rows.
    fn advance_outer(&mut self) -> Result<bool> {
        match self.outer.next()? {
            Some(row_ref) => {
                let outer_row = row_ref.into_owned();

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
                self.lookup_inner_rows(&key_value);

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

impl Operator for IndexNestedLoopJoinOperator {
    fn open(&mut self) -> Result<()> {
        self.outer.open()?;

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

            let outer_row = match &self.current_outer_row {
                Some(row) => row,
                None => {
                    if !self.advance_outer()? {
                        return Ok(None);
                    }
                    self.current_outer_row.as_ref().unwrap()
                }
            };

            // Try to find a match in current inner rows
            while self.current_inner_idx < self.current_inner_rows.len() {
                let inner_idx = self.current_inner_idx;
                self.current_inner_idx += 1;

                // Extract just the row from the (id, row) tuple
                let inner_row = &self.current_inner_rows[inner_idx].1;

                // Apply residual filter if present
                let passes_filter = if let Some(ref filter) = self.residual_filter {
                    filter.matches(outer_row, inner_row)
                } else {
                    true
                };

                if passes_filter {
                    self.outer_had_match = true;
                    let combined = self.combine(outer_row, inner_row);
                    return Ok(Some(RowRef::Owned(combined)));
                }
            }

            // Exhausted inner rows for current outer row
            // Handle LEFT OUTER: emit outer row with NULLs if no match
            if is_left_outer && !self.outer_had_match {
                let outer_row = self.current_outer_row.take().unwrap();
                self.advance_outer()?;
                let null_inner = self.null_inner_row();
                // Use owned variant - both rows are owned and won't be used again
                let combined = Row::from_combined_owned(outer_row, null_inner);
                return Ok(Some(RowRef::Owned(combined)));
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
pub struct BatchIndexNestedLoopJoinOperator {
    // Outer input operator
    outer: Box<dyn Operator>,

    // Inner table
    inner_table: Box<dyn Table>,

    // Join configuration
    join_type: JoinType,
    outer_key_idx: usize,
    lookup_strategy: IndexLookupStrategy,
    residual_filter: Option<JoinFilter>,

    // Output schema
    schema: Vec<ColumnInfo>,
    outer_col_count: usize,
    inner_col_count: usize,

    // Pre-computed results (built in open())
    results: Vec<Row>,
    result_idx: usize,

    // Optimization: Reusable buffer for row IDs to avoid allocation per outer row
    row_id_buffer: Vec<i64>,

    // State
    opened: bool,
}

impl BatchIndexNestedLoopJoinOperator {
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

        let outer_col_count = outer.schema().len();
        let inner_col_count = inner_schema.len();

        Self {
            outer,
            inner_table,
            join_type,
            outer_key_idx,
            lookup_strategy,
            residual_filter,
            schema,
            outer_col_count,
            inner_col_count,
            results: Vec::new(),
            result_idx: 0,
            // Pre-allocate buffer for typical number of matches
            row_id_buffer: Vec::with_capacity(16),
            opened: false,
        }
    }
}

impl Operator for BatchIndexNestedLoopJoinOperator {
    fn open(&mut self) -> Result<()> {
        use rustc_hash::{FxHashMap, FxHashSet};

        self.outer.open()?;

        let is_left_join = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let true_expr = ConstBoolExpr::true_expr();

        // Step 1: Collect all outer rows and their join keys
        let mut outer_rows: Vec<Row> = Vec::new();
        let mut row_id_set: FxHashSet<i64> = FxHashSet::default();
        let mut key_to_outer_indices: FxHashMap<i64, Vec<usize>> = FxHashMap::default();

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

            // Get row IDs based on lookup strategy into reusable buffer
            self.row_id_buffer.clear();
            match &self.lookup_strategy {
                IndexLookupStrategy::SecondaryIndex(index) => {
                    index.get_row_ids_equal_into(
                        std::slice::from_ref(&key_value),
                        &mut self.row_id_buffer,
                    );
                }
                IndexLookupStrategy::PrimaryKey => match &key_value {
                    Value::Integer(id) => self.row_id_buffer.push(*id),
                    Value::Float(f) => self.row_id_buffer.push(*f as i64),
                    _ => {}
                },
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
        let inner_rows_batch = self.inner_table.fetch_rows_by_ids(&all_row_ids, &true_expr);

        // Build inner row map by row_id
        let mut inner_by_id: FxHashMap<i64, Row> = FxHashMap::default();
        for (row_id, row) in inner_rows_batch {
            inner_by_id.insert(row_id, row);
        }

        // Step 3: Build results by matching outer rows with inner rows
        let mut matched_outers: Vec<bool> = vec![false; outer_rows.len()];

        for (row_id, inner_row) in &inner_by_id {
            if let Some(outer_indices) = key_to_outer_indices.get(row_id) {
                for &outer_idx in outer_indices {
                    let outer_row = &outer_rows[outer_idx];

                    // Apply residual filter if present
                    let passes_filter = if let Some(ref filter) = self.residual_filter {
                        filter.matches(outer_row, inner_row)
                    } else {
                        true
                    };

                    if passes_filter {
                        matched_outers[outer_idx] = true;
                        self.results.push(Row::from_combined(outer_row, inner_row));
                    }
                }
            }
        }

        // Handle LEFT OUTER JOIN - emit unmatched outer rows with NULLs
        if is_left_join {
            for (idx, outer_row) in outer_rows.iter().enumerate() {
                if !matched_outers[idx] {
                    let values = combine_rows_with_nulls(
                        outer_row,
                        self.outer_col_count,
                        self.inner_col_count,
                        true,
                    );
                    self.results.push(Row::from_values(values));
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
