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

//! Result trait for query results
//!

use crate::common::CompactArc;
use crate::core::{Result, Row};

/// QueryResult represents the result of a SQL query
///
/// This trait provides iteration over result rows with both cursor-style
/// and direct row access patterns. It handles column metadata and aliasing.
///
/// # Example
///
/// ```ignore
/// let result = transaction.select("users", &["id", "name"], None)?;
/// println!("Columns: {:?}", result.columns());
/// while result.next() {
///     let row = result.row();
///     // Process row...
/// }
/// result.close()?;
/// ```
pub trait QueryResult: Send {
    /// Returns the column names in the result
    ///
    /// If aliases are set, this returns the aliased column names.
    fn columns(&self) -> &[String];

    /// Returns column names as Arc for zero-copy sharing
    ///
    /// This is an optimization for the API layer to avoid cloning column names.
    /// Default implementation returns None, meaning caller should fall back to
    /// cloning from columns().
    fn columns_arc(&self) -> Option<CompactArc<Vec<String>>> {
        None
    }

    /// Moves the cursor to the next row
    ///
    /// Returns `true` if there is another row available, `false` otherwise.
    fn next(&mut self) -> bool;

    /// Returns the current row directly without copying
    ///
    /// This is a high-performance method to access raw column values.
    /// The returned row is valid until the next call to `next()` or `close()`.
    fn row(&self) -> &Row;

    /// Takes ownership of the current row (avoids clone)
    ///
    /// This is a high-performance method that moves the row data out of the result.
    /// After calling this, `row()` will return an empty row until `next()` is called.
    /// The default implementation clones the row for backward compatibility.
    fn take_row(&mut self) -> Row {
        self.row().clone()
    }

    /// Closes the result set and releases resources
    ///
    /// Default implementation does nothing. Override if cleanup is needed.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Returns the number of rows affected by an INSERT, UPDATE, or DELETE
    ///
    /// Default implementation returns 0. Override for DML results.
    fn rows_affected(&self) -> i64 {
        0
    }

    /// The rows an UPDATE changed, those its SET gave a value different
    /// from the one they held; the rows affected for any other statement,
    /// 0 for one with RETURNING
    fn rows_changed(&self) -> i64 {
        self.rows_affected()
    }

    /// Try to extract all rows as CompactArc<Vec<Row>> for zero-copy joins
    ///
    /// Returns None if the result cannot provide Arc-wrapped rows.
    /// This consumes the result - after calling, iteration will yield no more rows.
    /// Default implementation returns None.
    fn try_into_arc_rows(&mut self) -> Option<CompactArc<Vec<Row>>> {
        None
    }

    /// Returns an estimate of the total number of rows in the result.
    ///
    /// This is used for pre-allocating vectors to avoid reallocations.
    /// Returns None if the count is unknown. Default implementation returns None.
    fn estimated_count(&self) -> Option<usize> {
        None
    }

    /// The exact number of rows a fresh result will yield, when it is known
    /// without iterating. Default implementation returns None.
    fn exact_len(&self) -> Option<usize> {
        None
    }

    /// Returns a pending error from the last `next()` call, if any.
    ///
    /// When `next()` returns false due to a runtime error (e.g. invalid REGEXP
    /// pattern), this method returns the error so callers can surface it.
    /// Default returns None (no error).
    fn last_error(&mut self) -> Option<crate::core::Error> {
        None
    }
}

/// A simple in-memory query result (useful for testing and simple results)
pub struct MemoryResult {
    columns: Vec<String>,
    rows: Vec<Row>,
    current_index: Option<usize>,
    rows_affected: i64,
    closed: bool,
}

impl MemoryResult {
    /// Creates a result with columns and rows
    pub fn with_rows(columns: Vec<String>, rows: Vec<Row>) -> Self {
        Self {
            columns,
            rows,
            current_index: None,
            rows_affected: 0,
            closed: false,
        }
    }
}

impl QueryResult for MemoryResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.rows.len())
    }

    fn next(&mut self) -> bool {
        if self.closed {
            return false;
        }

        let next_index = match self.current_index {
            None => 0,
            Some(i) => i + 1,
        };

        if next_index < self.rows.len() {
            self.current_index = Some(next_index);
            true
        } else {
            false
        }
    }

    fn row(&self) -> &Row {
        match self.current_index {
            Some(i) if i < self.rows.len() => &self.rows[i],
            _ => panic!("row() called without successful next()"),
        }
    }

    /// Optimized take_row that swaps out the row instead of cloning
    fn take_row(&mut self) -> Row {
        match self.current_index {
            Some(i) if i < self.rows.len() => std::mem::take(&mut self.rows[i]),
            _ => panic!("take_row() called without successful next()"),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }

    fn rows_affected(&self) -> i64 {
        self.rows_affected
    }
}

/// An empty result that returns no rows
pub struct EmptyResult {
    columns: Vec<String>,
    rows_affected: i64,
}

impl EmptyResult {
    /// Creates a new empty result
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows_affected: 0,
        }
    }
}

impl Default for EmptyResult {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryResult for EmptyResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        false
    }

    fn row(&self) -> &Row {
        panic!("row() called on empty result")
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn rows_affected(&self) -> i64 {
        self.rows_affected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_memory_result_with_rows() {
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::text("Alice")]),
            Row::from_values(vec![Value::Integer(2), Value::text("Bob")]),
        ];

        let mut result = MemoryResult::with_rows(vec!["id".to_string(), "name".to_string()], rows);

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));

        assert!(!result.next());
    }

    #[test]
    fn test_memory_result_close() {
        let rows = vec![Row::from_values(vec![Value::Integer(1)])];
        let mut result = MemoryResult::with_rows(vec!["id".to_string()], rows);

        assert!(result.next());
        assert!(result.close().is_ok());
        assert!(!result.next()); // After close, next returns false
    }

    #[test]
    fn test_empty_result() {
        let mut result = EmptyResult::new();

        assert!(result.columns().is_empty());
        assert!(!result.next());
        assert_eq!(result.rows_affected(), 0);
        assert!(result.close().is_ok());
    }
}
