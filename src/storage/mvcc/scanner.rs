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

use crate::common::cow_btree::MAX_KEYS;
use crate::common::CompactArc;
use crate::core::{Result, Row, RowVec, Schema};
use crate::storage::expression::Expression;
use crate::storage::mvcc::registry::StatementView;
use crate::storage::mvcc::table::normalize_row_to_schema;
use crate::storage::mvcc::version_store::{HotRows, Reader, ScanFilter};
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
        let cut = projection(&schema, &column_indices);
        let needs_projection = !cut.is_empty();

        // Project rows upfront if needed
        let projected_rows: RowVec = if needs_projection {
            rows.into_iter()
                .map(|(id, row)| (id, projected(&row, &cut)))
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

/// The rows the first step of a hot scan reads at least; each step reads
/// twice as many as the one before, up to `MAX_STEP_ROWS`
const FIRST_STEP_ROWS: usize = 64;
const MAX_STEP_ROWS: usize = 4096;

/// A scan of the hot rows a statement view sees, read in steps from the
/// rows taken when the scan began, so a reader that stops early reads no
/// further
pub(crate) struct HotScanner {
    /// The rows taken and the view that reads them, let go on close
    source: Option<(HotRows, StatementView)>,
    txn_id: i64,
    filter: Option<ScanFilter>,
    schema: CompactArc<Schema>,
    /// The columns each row is cut to, empty for the whole row
    projection: Vec<usize>,
    /// Whether a read without a filter normalizes its rows, which its
    /// first row decides, as the store's whole read decides it
    normalize: Option<bool>,
    rows: RowVec,
    /// One past the current row of `rows`
    at: usize,
    step: usize,
    /// The source ran out, or a step failed
    finished: bool,
    error: Option<crate::core::Error>,
}

impl HotScanner {
    /// A scan of `source` through `view`, its filter read as `schema`, the
    /// schema of the rows taken
    pub(crate) fn new(
        source: HotRows,
        reader: Reader<'_>,
        view: StatementView,
        filter: Option<&dyn Expression>,
        schema: CompactArc<Schema>,
        column_indices: &[usize],
    ) -> Self {
        // A step stops at the end of a leaf, so it may read a leaf more
        let rows = RowVec::with_capacity((MAX_STEP_ROWS + MAX_KEYS).min(source.held()));
        Self {
            source: Some((source, view)),
            txn_id: reader.txn_id,
            filter: filter.map(|filter| ScanFilter::new(&schema, filter)),
            projection: projection(&schema, column_indices),
            schema,
            normalize: None,
            rows,
            at: 0,
            step: FIRST_STEP_ROWS,
            finished: false,
            error: None,
        }
    }

    /// Reads the next step into `rows`, as the scan hands them on: a
    /// filtered read's as the store holds them
    fn read_step(&mut self) -> Result<()> {
        self.rows.clear();
        self.at = 0;
        let Some((source, view)) = &mut self.source else {
            self.finished = true;
            return Ok(());
        };
        let reader = Reader {
            txn_id: self.txn_id,
            view: Some(view),
        };
        let more = match &self.filter {
            None => source.fill(reader, &mut self.rows, self.step, |_| {
                stepped();
                Ok(true)
            })?,
            Some(filter) => source.fill(reader, &mut self.rows, self.step, |row| {
                stepped();
                filter.passes(row)
            })?,
        };
        self.finished = !more;
        self.step = (self.step * 2).min(MAX_STEP_ROWS);
        let columns = self.schema.columns.len();
        if let Some((_, first)) = self.rows.first() {
            if self.filter.is_none() && *self.normalize.get_or_insert(first.len() != columns) {
                for (_, row) in self.rows.iter_mut() {
                    *row = normalize_row_to_schema(std::mem::take(row), &self.schema);
                }
            }
        }
        if !self.projection.is_empty() {
            for (_, row) in self.rows.iter_mut() {
                *row = projected(row, &self.projection);
            }
        }
        Ok(())
    }

    /// Reads steps until one holds a row: false once the rows run out or
    /// a step failed
    #[inline(never)]
    fn refill(&mut self) -> bool {
        while !self.finished {
            if let Err(error) = self.read_step() {
                self.error = Some(error);
                self.finished = true;
                return false;
            }
            if !self.rows.is_empty() {
                self.at = 1;
                return true;
            }
        }
        false
    }
}

#[inline(always)]
fn stepped() {
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::hot_scan_stepped();
}

impl Scanner for HotScanner {
    fn next(&mut self) -> bool {
        self.at += 1;
        self.at <= self.rows.len() || self.refill()
    }

    fn row(&self) -> &Row {
        &self.rows[self.at - 1].1
    }

    fn err(&self) -> Option<&crate::core::Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.finished = true;
        self.source = None;
        self.rows.clear();
        self.at = 0;
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.rows[self.at - 1].1)
    }

    fn current_row_id(&self) -> i64 {
        self.rows[self.at - 1].0
    }

    fn estimated_count(&self) -> Option<usize> {
        match (&self.filter, &self.source) {
            (None, Some((source, _))) => Some(source.held()),
            _ => None,
        }
    }
}

/// The columns `column_indices` cuts a row of `schema` to, empty when it
/// keeps the whole row in order
fn projection(schema: &Schema, column_indices: &[usize]) -> Vec<usize> {
    let cuts = !column_indices.is_empty()
        && column_indices.len() < schema.columns.len()
        && !column_indices.iter().enumerate().all(|(i, &idx)| i == idx);
    if cuts {
        column_indices.to_vec()
    } else {
        Vec::new()
    }
}

/// `row` cut to the columns `projection` names, NULL for one it lacks
fn projected(row: &Row, projection: &[usize]) -> Row {
    Row::from_values(
        projection
            .iter()
            .map(|&idx| {
                row.get(idx)
                    .cloned()
                    .unwrap_or_else(crate::core::Value::null_unknown)
            })
            .collect(),
    )
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
