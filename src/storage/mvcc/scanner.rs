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

//! MVCC Scanner implementations
//!
//! Provides scanner implementations for MVCC query results.
//!

use crate::common::CompactArc;
use crate::core::{Result, Row, RowVec, Schema};
use crate::storage::expression::Expression;
use crate::storage::traits::Scanner;

/// MVCC Scanner for iterating over versioned rows
pub struct MVCCScanner {
    /// Source rows with their IDs (cached - returns to pool on drop)
    rows: RowVec,
    /// Current index in the rows vector
    current_index: isize,
    /// Column indices to include in projection
    column_indices: Vec<usize>,
    /// Any error that occurred
    error: Option<crate::core::Error>,
    /// Whether the scanner has been closed
    closed: bool,
}

impl MVCCScanner {
    /// Creates a new MVCC scanner with filtering (legacy method)
    pub fn new(
        rows: RowVec,
        schema: CompactArc<Schema>,
        column_indices: Vec<usize>,
        filter: Option<Box<dyn Expression>>,
    ) -> Self {
        // Filter rows if needed
        let filtered_rows: RowVec = if let Some(ref expr) = filter {
            rows.into_iter()
                .filter(|(_, row)| expr.evaluate(row).unwrap_or_default())
                .collect()
        } else {
            rows
        };

        Self::from_rows(filtered_rows, schema, column_indices)
    }

    /// Creates scanner from RowVec
    #[inline]
    pub fn from_rows(rows: RowVec, schema: CompactArc<Schema>, column_indices: Vec<usize>) -> Self {
        let num_schema_cols = schema.columns.len();

        // Check if we need projection (column_indices is a proper subset)
        let needs_projection = !column_indices.is_empty()
            && column_indices.len() < num_schema_cols
            && !column_indices.iter().enumerate().all(|(i, &idx)| i == idx);

        // Project rows upfront if needed
        let projected_rows: RowVec = if needs_projection {
            rows.into_iter()
                .map(|(id, row)| {
                    let projected_values: Vec<crate::core::Value> = column_indices
                        .iter()
                        .map(|&idx| {
                            row.get(idx)
                                .cloned()
                                .unwrap_or_else(crate::core::Value::null_unknown)
                        })
                        .collect();
                    (id, Row::from_values(projected_values))
                })
                .collect()
        } else {
            rows
        };

        Self {
            rows: projected_rows,
            current_index: -1,
            column_indices: if needs_projection {
                vec![]
            } else {
                column_indices
            }, // Clear if already projected
            error: None,
            closed: false,
        }
    }

    /// Creates an empty scanner
    #[inline]
    pub fn empty(column_indices: Vec<usize>) -> Self {
        Self {
            rows: RowVec::new(),
            current_index: -1,
            column_indices,
            error: None,
            closed: false,
        }
    }

    /// Creates a scanner with a single row
    pub fn single(row: Row, column_indices: Vec<usize>) -> Self {
        let mut rows = RowVec::new();
        rows.push((0, row));
        Self {
            rows,
            current_index: -1,
            column_indices,
            error: None,
            closed: false,
        }
    }

    /// Returns the number of rows in the scanner
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns true if the scanner has no rows
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl Scanner for MVCCScanner {
    fn next(&mut self) -> bool {
        if self.closed || self.error.is_some() {
            return false;
        }

        self.current_index += 1;

        (self.current_index as usize) < self.rows.len()
    }

    fn row(&self) -> &Row {
        if self.current_index < 0 || (self.current_index as usize) >= self.rows.len() {
            // Return a static empty row for safety
            static EMPTY_ROW: std::sync::OnceLock<Row> = std::sync::OnceLock::new();
            return EMPTY_ROW.get_or_init(|| Row::from_values(vec![]));
        }

        let (_, ref source_row) = self.rows[self.current_index as usize];

        // If no column projection, return the row directly
        if self.column_indices.is_empty() {
            return source_row;
        }

        // We need to return a reference to a projected row
        // This is tricky because we need to modify projected_row
        // For now, return the source row if indices match all columns
        if self.column_indices.len() == source_row.len() {
            let all_match = self
                .column_indices
                .iter()
                .enumerate()
                .all(|(i, &idx)| i == idx);
            if all_match {
                return source_row;
            }
        }

        // Otherwise return source row - projection handled in caller
        source_row
    }

    fn err(&self) -> Option<&crate::core::Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.rows.clear();
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        if self.current_index < 0 || (self.current_index as usize) >= self.rows.len() {
            return Row::new();
        }

        // Swap out the row with an empty one to avoid cloning
        let idx = self.current_index as usize;
        std::mem::take(&mut self.rows[idx].1)
    }

    fn current_row_id(&self) -> i64 {
        if self.current_index < 0 || (self.current_index as usize) >= self.rows.len() {
            return 0;
        }
        self.rows[self.current_index as usize].0
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.rows.len())
    }
}

/// Empty scanner that returns no rows
pub struct EmptyScanner;

impl EmptyScanner {
    /// Creates a new empty scanner
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmptyScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Scanner for EmptyScanner {
    fn next(&mut self) -> bool {
        false
    }

    fn row(&self) -> &Row {
        static EMPTY_ROW: std::sync::OnceLock<Row> = std::sync::OnceLock::new();
        EMPTY_ROW.get_or_init(|| Row::from_values(vec![]))
    }

    fn err(&self) -> Option<&crate::core::Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, SchemaBuilder, Value};

    fn test_schema() -> CompactArc<Schema> {
        CompactArc::new(
            SchemaBuilder::new("test")
                .column("id", DataType::Integer, false, false)
                .build(),
        )
    }

    #[test]
    fn test_mvcc_scanner_empty() {
        let mut scanner = MVCCScanner::empty(vec![0]);

        assert!(!scanner.next());
        assert!(scanner.is_empty());
    }

    #[test]
    fn test_mvcc_scanner_single() {
        let row = Row::from_values(vec![Value::Integer(42)]);
        let mut scanner = MVCCScanner::single(row, vec![0]);

        assert_eq!(scanner.len(), 1);
        assert!(!scanner.is_empty());

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(42)));

        assert!(!scanner.next());
    }

    #[test]
    fn test_mvcc_scanner_multiple_rows() {
        let schema = test_schema();

        let mut rows = RowVec::new();
        rows.push((1, Row::from_values(vec![Value::Integer(1)])));
        rows.push((2, Row::from_values(vec![Value::Integer(2)])));
        rows.push((3, Row::from_values(vec![Value::Integer(3)])));

        let mut scanner = MVCCScanner::from_rows(rows, schema, vec![0]);

        assert_eq!(scanner.len(), 3);

        // Check all rows
        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(1)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(2)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(3)));

        assert!(!scanner.next());
    }

    #[test]
    fn test_mvcc_scanner_close() {
        let schema = test_schema();

        let mut rows = RowVec::new();
        rows.push((1, Row::from_values(vec![Value::Integer(1)])));

        let mut scanner = MVCCScanner::from_rows(rows, schema, vec![0]);

        assert!(scanner.next());
        assert!(scanner.close().is_ok());

        // After close, next should return false
        assert!(!scanner.next());
    }

    #[test]
    fn test_empty_scanner() {
        let mut scanner = EmptyScanner::new();

        assert!(!scanner.next());
        assert!(scanner.err().is_none());
        assert!(scanner.close().is_ok());
    }
}
