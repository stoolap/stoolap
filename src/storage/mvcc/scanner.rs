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

use crate::core::{Result, Row, Schema};
use crate::storage::expression::Expression;
use crate::storage::traits::Scanner;

/// MVCC Scanner for iterating over versioned rows
pub struct MVCCScanner {
    /// Source rows with their IDs
    rows: Vec<(i64, Row)>,
    /// Current index in the rows vector
    current_index: isize,
    /// Column indices to include in projection
    column_indices: Vec<usize>,
    /// Table schema (kept for future projection improvements)
    #[allow(dead_code)]
    schema: Schema,
    /// Filter expression (optional, kept for streaming filter support)
    #[allow(dead_code)]
    filter: Option<Box<dyn Expression>>,
    /// Any error that occurred
    error: Option<crate::core::Error>,
    /// Pre-allocated buffer for projected row (kept for future optimization)
    #[allow(dead_code)]
    projected_row: Row,
    /// Whether the scanner has been closed
    closed: bool,
}

impl MVCCScanner {
    /// Creates a new MVCC scanner with filtering (legacy method)
    pub fn new(
        rows: Vec<(i64, Row)>,
        schema: Schema,
        column_indices: Vec<usize>,
        filter: Option<Box<dyn Expression>>,
    ) -> Self {
        // Filter rows if needed
        let filtered_rows: Vec<(i64, Row)> = if let Some(ref expr) = filter {
            rows.into_iter()
                .filter(|(_, row)| expr.evaluate(row).unwrap_or_default())
                .collect()
        } else {
            rows
        };

        Self::from_rows(filtered_rows, schema, column_indices)
    }

    /// Creates scanner from pre-filtered rows with optional projection
    ///
    /// If column_indices is a proper subset of columns, projects rows upfront
    /// to avoid repeated projection on each row() call.
    #[inline]
    pub fn from_rows(rows: Vec<(i64, Row)>, schema: Schema, column_indices: Vec<usize>) -> Self {
        let num_schema_cols = schema.columns.len();

        // Check if we need projection (column_indices is a proper subset)
        let needs_projection = !column_indices.is_empty()
            && column_indices.len() < num_schema_cols
            && !column_indices.iter().enumerate().all(|(i, &idx)| i == idx);

        // Project rows upfront if needed
        let projected_rows = if needs_projection {
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
            schema,
            filter: None,
            error: None,
            projected_row: Row::default(),
            closed: false,
        }
    }

    /// Creates an empty scanner
    #[inline]
    pub fn empty(schema: Schema, column_indices: Vec<usize>) -> Self {
        Self {
            rows: Vec::new(),
            current_index: -1,
            column_indices,
            schema,
            filter: None,
            error: None,
            projected_row: Row::default(),
            closed: false,
        }
    }

    /// Creates a scanner with a single row
    pub fn single(row: Row, schema: Schema, column_indices: Vec<usize>) -> Self {
        Self {
            rows: vec![(0, row)],
            current_index: -1,
            column_indices: column_indices.clone(),
            schema,
            filter: None,
            error: None,
            projected_row: Row::from_values(vec![
                crate::core::Value::null_unknown();
                column_indices.len()
            ]),
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

    /// Consumes the scanner and returns all rows without cloning
    ///
    /// This is more efficient than iterating and cloning each row.
    /// Use this when you need all rows and won't use the scanner afterwards.
    #[inline]
    pub fn into_rows(self) -> Vec<Row> {
        self.rows.into_iter().map(|(_, row)| row).collect()
    }

    /// Takes ownership of the rows, leaving the scanner empty
    ///
    /// This is more efficient than iterating and cloning each row.
    #[inline]
    pub fn take_rows(&mut self) -> Vec<Row> {
        std::mem::take(&mut self.rows)
            .into_iter()
            .map(|(_, row)| row)
            .collect()
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
}

/// Range scanner optimized for consecutive ID range queries
pub struct RangeScanner {
    /// Transaction ID for visibility checks
    txn_id: i64,
    /// Current ID in range (kept for range iteration improvements)
    #[allow(dead_code)]
    current_id: i64,
    /// End ID in range (kept for range iteration improvements)
    #[allow(dead_code)]
    end_id: i64,
    /// Whether end_id is inclusive (kept for range iteration improvements)
    #[allow(dead_code)]
    inclusive: bool,
    /// Column indices to include (kept for future projection improvements)
    #[allow(dead_code)]
    column_indices: Vec<usize>,
    /// Table schema (kept for future projection improvements)
    #[allow(dead_code)]
    schema: Schema,
    /// Any scanning error
    error: Option<crate::core::Error>,
    /// Pre-allocated projected row buffer (kept for future optimization)
    #[allow(dead_code)]
    projected_row: Row,
    /// Row iterator (stores pre-fetched rows)
    rows: Vec<(i64, Row)>,
    /// Current position in rows
    row_index: isize,
}

impl RangeScanner {
    /// Creates a new range scanner
    pub fn new(
        start_id: i64,
        end_id: i64,
        inclusive: bool,
        txn_id: i64,
        schema: Schema,
        column_indices: Vec<usize>,
        rows: Vec<(i64, Row)>,
    ) -> Self {
        // Filter rows to only include those in range
        let actual_end = if inclusive { end_id } else { end_id - 1 };
        let filtered_rows: Vec<(i64, Row)> = rows
            .into_iter()
            .filter(|(id, _)| *id >= start_id && *id <= actual_end)
            .collect();

        Self {
            txn_id,
            current_id: start_id,
            end_id,
            inclusive,
            column_indices: column_indices.clone(),
            schema,
            error: None,
            projected_row: Row::from_values(vec![
                crate::core::Value::null_unknown();
                column_indices.len()
            ]),
            rows: filtered_rows,
            row_index: -1,
        }
    }

    /// Returns the transaction ID
    pub fn txn_id(&self) -> i64 {
        self.txn_id
    }
}

impl Scanner for RangeScanner {
    fn next(&mut self) -> bool {
        if self.error.is_some() {
            return false;
        }

        self.row_index += 1;

        // OPTIMIZATION: Don't clone the row here - just track the index
        // We can return a reference directly in row() since rows are stored in self.rows
        (self.row_index as usize) < self.rows.len()
    }

    fn row(&self) -> &Row {
        static EMPTY_ROW: std::sync::OnceLock<Row> = std::sync::OnceLock::new();

        // OPTIMIZATION: Return reference directly from self.rows instead of cloning
        if self.row_index >= 0 && (self.row_index as usize) < self.rows.len() {
            let (_, ref row) = self.rows[self.row_index as usize];
            row
        } else {
            EMPTY_ROW.get_or_init(|| Row::from_values(vec![]))
        }
    }

    fn err(&self) -> Option<&crate::core::Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.rows.clear();
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        if self.row_index >= 0 && (self.row_index as usize) < self.rows.len() {
            let idx = self.row_index as usize;
            std::mem::take(&mut self.rows[idx].1)
        } else {
            Row::new()
        }
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

/// Single row scanner that returns exactly one row
pub struct SingleRowScanner {
    /// The single row to return
    row: Row,
    /// Column indices for projection (kept for future projection support)
    #[allow(dead_code)]
    column_indices: Vec<usize>,
    /// Whether next() has been called
    done: bool,
}

impl SingleRowScanner {
    /// Creates a new single row scanner
    pub fn new(row: Row, column_indices: Vec<usize>) -> Self {
        Self {
            row,
            column_indices,
            done: false,
        }
    }
}

impl Scanner for SingleRowScanner {
    fn next(&mut self) -> bool {
        if self.done {
            false
        } else {
            self.done = true;
            true
        }
    }

    fn row(&self) -> &Row {
        if self.done {
            &self.row
        } else {
            static EMPTY_ROW: std::sync::OnceLock<Row> = std::sync::OnceLock::new();
            EMPTY_ROW.get_or_init(|| Row::from_values(vec![]))
        }
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

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, false)
            .build()
    }

    #[test]
    fn test_mvcc_scanner_empty() {
        let schema = test_schema();

        let mut scanner = MVCCScanner::empty(schema, vec![0]);

        assert!(!scanner.next());
        assert!(scanner.is_empty());
    }

    #[test]
    fn test_mvcc_scanner_single() {
        let schema = test_schema();

        let row = Row::from_values(vec![Value::Integer(42)]);
        let mut scanner = MVCCScanner::single(row, schema, vec![0]);

        assert_eq!(scanner.len(), 1);
        assert!(!scanner.is_empty());

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(42)));

        assert!(!scanner.next());
    }

    #[test]
    fn test_mvcc_scanner_multiple_rows() {
        let schema = test_schema();

        let rows = vec![
            (1, Row::from_values(vec![Value::Integer(1)])),
            (2, Row::from_values(vec![Value::Integer(2)])),
            (3, Row::from_values(vec![Value::Integer(3)])),
        ];

        let mut scanner = MVCCScanner::new(rows, schema, vec![0], None);

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

        let rows = vec![(1, Row::from_values(vec![Value::Integer(1)]))];

        let mut scanner = MVCCScanner::new(rows, schema, vec![0], None);

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

    #[test]
    fn test_single_row_scanner() {
        let row = Row::from_values(vec![Value::Integer(42), Value::text("test")]);

        let mut scanner = SingleRowScanner::new(row, vec![0, 1]);

        // First call to next should succeed
        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(42)));
        assert_eq!(scanner.row().get(1), Some(&Value::text("test")));

        // Second call should fail
        assert!(!scanner.next());
    }

    #[test]
    fn test_range_scanner() {
        let schema = test_schema();

        let rows = vec![
            (1, Row::from_values(vec![Value::Integer(1)])),
            (2, Row::from_values(vec![Value::Integer(2)])),
            (3, Row::from_values(vec![Value::Integer(3)])),
            (5, Row::from_values(vec![Value::Integer(5)])),
        ];

        // Inclusive range 1-3
        let mut scanner = RangeScanner::new(1, 3, true, 1, schema, vec![0], rows);

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(1)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(2)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(3)));

        assert!(!scanner.next()); // Row 5 is outside range
    }
}
