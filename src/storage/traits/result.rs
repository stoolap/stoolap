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

use rustc_hash::FxHashMap;

use crate::core::{Result, Row, Value};

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

    /// Moves the cursor to the next row
    ///
    /// Returns `true` if there is another row available, `false` otherwise.
    fn next(&mut self) -> bool;

    /// Scans the current row into the provided values
    ///
    /// The number of destination values must match the number of columns.
    /// Values are converted to the destination types where possible.
    fn scan(&self, dest: &mut [Value]) -> Result<()>;

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
    fn close(&mut self) -> Result<()>;

    /// Returns the number of rows affected by an INSERT, UPDATE, or DELETE
    fn rows_affected(&self) -> i64;

    /// Returns the last inserted ID for an INSERT operation
    fn last_insert_id(&self) -> i64;

    /// Sets column aliases for this result
    ///
    /// The map keys are alias names, values are original column names.
    /// Returns a new result with the aliases applied.
    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult>;
}

/// A simple in-memory query result (useful for testing and simple results)
pub struct MemoryResult {
    columns: Vec<String>,
    rows: Vec<Row>,
    current_index: Option<usize>,
    rows_affected: i64,
    last_insert_id: i64,
    closed: bool,
}

impl MemoryResult {
    /// Creates a new empty result with the given columns
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            current_index: None,
            rows_affected: 0,
            last_insert_id: 0,
            closed: false,
        }
    }

    /// Creates a result with columns and rows
    pub fn with_rows(columns: Vec<String>, rows: Vec<Row>) -> Self {
        Self {
            columns,
            rows,
            current_index: None,
            rows_affected: 0,
            last_insert_id: 0,
            closed: false,
        }
    }

    /// Creates a result for a modification operation (INSERT/UPDATE/DELETE)
    pub fn for_modification(rows_affected: i64, last_insert_id: i64) -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            current_index: None,
            rows_affected,
            last_insert_id,
            closed: false,
        }
    }

    /// Adds a row to the result
    pub fn add_row(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// Sets the rows affected count
    pub fn set_rows_affected(&mut self, count: i64) {
        self.rows_affected = count;
    }

    /// Sets the last insert ID
    pub fn set_last_insert_id(&mut self, id: i64) {
        self.last_insert_id = id;
    }
}

impl QueryResult for MemoryResult {
    fn columns(&self) -> &[String] {
        &self.columns
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

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        let row = self.row();

        if dest.len() != row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                row.len()
            )));
        }

        for (i, value) in row.iter().enumerate() {
            dest[i] = value.clone();
        }

        Ok(())
    }

    fn row(&self) -> &Row {
        match self.current_index {
            Some(i) if i < self.rows.len() => &self.rows[i],
            _ => panic!("row() called without successful next()"),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }

    fn rows_affected(&self) -> i64 {
        self.rows_affected
    }

    fn last_insert_id(&self) -> i64 {
        self.last_insert_id
    }

    fn with_aliases(
        mut self: Box<Self>,
        aliases: FxHashMap<String, String>,
    ) -> Box<dyn QueryResult> {
        // Apply aliases to column names
        for col in &mut self.columns {
            // Find if this column has an alias (reverse lookup)
            for (alias, original) in &aliases {
                if col == original {
                    *col = alias.clone();
                    break;
                }
            }
        }
        self
    }
}

/// An empty result that returns no rows
pub struct EmptyResult {
    columns: Vec<String>,
    rows_affected: i64,
    last_insert_id: i64,
}

impl EmptyResult {
    /// Creates a new empty result
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows_affected: 0,
            last_insert_id: 0,
        }
    }

    /// Creates an empty result for a modification operation
    pub fn for_modification(rows_affected: i64, last_insert_id: i64) -> Self {
        Self {
            columns: Vec::new(),
            rows_affected,
            last_insert_id,
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

    fn scan(&self, _dest: &mut [Value]) -> Result<()> {
        Err(crate::core::Error::internal(
            "scan() called on empty result",
        ))
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

    fn last_insert_id(&self) -> i64 {
        self.last_insert_id
    }

    fn with_aliases(self: Box<Self>, _aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_result_empty() {
        let mut result = MemoryResult::new(vec!["id".to_string(), "name".to_string()]);

        assert_eq!(result.columns(), &["id", "name"]);
        assert!(!result.next());
        assert_eq!(result.rows_affected(), 0);
        assert_eq!(result.last_insert_id(), 0);
    }

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
    fn test_memory_result_scan() {
        let rows = vec![Row::from_values(vec![
            Value::Integer(42),
            Value::text("test"),
        ])];

        let mut result = MemoryResult::with_rows(vec!["id".to_string(), "name".to_string()], rows);

        assert!(result.next());

        let mut dest = vec![Value::null_unknown(), Value::null_unknown()];
        result.scan(&mut dest).unwrap();

        assert_eq!(dest[0], Value::Integer(42));
        assert_eq!(dest[1], Value::text("test"));
    }

    #[test]
    fn test_memory_result_for_modification() {
        let result = MemoryResult::for_modification(5, 100);

        assert_eq!(result.rows_affected(), 5);
        assert_eq!(result.last_insert_id(), 100);
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
    fn test_memory_result_with_aliases() {
        let rows = vec![Row::from_values(vec![Value::Integer(1)])];
        let result = Box::new(MemoryResult::with_rows(vec!["user_id".to_string()], rows));

        let mut aliases = FxHashMap::default();
        aliases.insert("id".to_string(), "user_id".to_string());

        let aliased = result.with_aliases(aliases);
        assert_eq!(aliased.columns(), &["id"]);
    }

    #[test]
    fn test_empty_result() {
        let mut result = EmptyResult::new();

        assert!(result.columns().is_empty());
        assert!(!result.next());
        assert_eq!(result.rows_affected(), 0);
        assert!(result.close().is_ok());
    }

    #[test]
    fn test_empty_result_for_modification() {
        let result = EmptyResult::for_modification(10, 0);

        assert_eq!(result.rows_affected(), 10);
        assert_eq!(result.last_insert_id(), 0);
    }
}
