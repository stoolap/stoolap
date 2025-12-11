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

//! Scanner trait for iterating over table rows
//!

use crate::core::{Result, Row};

/// Scanner provides an iterator over rows in a table
///
/// This trait is the primary way to read rows from a table. It follows
/// an iterator pattern where `next()` advances to the next row and
/// `row()` returns the current row.
///
/// # Example
///
/// ```ignore
/// let scanner = table.scan(&[0, 1], None)?;
/// while scanner.next() {
///     let row = scanner.row();
///     // Process row...
/// }
/// if let Some(err) = scanner.err() {
///     // Handle error...
/// }
/// scanner.close()?;
/// ```
pub trait Scanner: Send {
    /// Advances the scanner to the next row
    ///
    /// Returns `true` if there is another row available, `false` otherwise.
    /// After returning `false`, the caller should check `err()` to see if
    /// iteration stopped due to an error.
    fn next(&mut self) -> bool;

    /// Returns the current row
    ///
    /// The returned row is valid until the next call to `next()` or `close()`.
    /// Calling this before `next()` or after `next()` returns `false` is undefined.
    fn row(&self) -> &Row;

    /// Returns any error that occurred during scanning
    ///
    /// Should be called after `next()` returns `false` to check if iteration
    /// stopped due to an error or simply because there are no more rows.
    fn err(&self) -> Option<&crate::core::Error>;

    /// Closes the scanner and releases any resources
    ///
    /// This should be called when done with the scanner, even if `next()`
    /// returned `false` due to reaching the end of the data.
    fn close(&mut self) -> Result<()>;

    /// Takes ownership of the current row (avoids clone)
    ///
    /// This is more efficient than `row().clone()` when you need to move
    /// the row data out of the scanner. After calling this, the internal
    /// row buffer may be empty until `next()` is called again.
    /// The default implementation clones the row for backward compatibility.
    fn take_row(&mut self) -> Row {
        self.row().clone()
    }
}

/// An empty scanner that immediately returns no rows
pub struct EmptyScanner {
    empty_row: Row,
    closed: bool,
}

impl EmptyScanner {
    /// Creates a new empty scanner
    pub fn new() -> Self {
        Self {
            empty_row: Row::new(),
            closed: false,
        }
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
        // This should never be called since next() always returns false
        &self.empty_row
    }

    fn err(&self) -> Option<&crate::core::Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}

/// A scanner over a vector of rows (useful for testing)
pub struct VecScanner {
    rows: Vec<Row>,
    current_index: Option<usize>,
    error: Option<crate::core::Error>,
    closed: bool,
}

impl VecScanner {
    /// Creates a new scanner over the given rows
    pub fn new(rows: Vec<Row>) -> Self {
        Self {
            rows,
            current_index: None,
            error: None,
            closed: false,
        }
    }

    /// Creates a scanner that will return an error
    pub fn with_error(error: crate::core::Error) -> Self {
        Self {
            rows: Vec::new(),
            current_index: None,
            error: Some(error),
            closed: false,
        }
    }
}

impl Scanner for VecScanner {
    fn next(&mut self) -> bool {
        if self.closed || self.error.is_some() {
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
            _ => {
                // Panic in debug mode, return first row or panic in release
                // This is a programming error - row() called without next()
                panic!("row() called without successful next()")
            }
        }
    }

    fn err(&self) -> Option<&crate::core::Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_empty_scanner() {
        let mut scanner = EmptyScanner::new();
        assert!(!scanner.next());
        assert!(scanner.err().is_none());
        assert!(scanner.close().is_ok());
    }

    #[test]
    fn test_vec_scanner_empty() {
        let mut scanner = VecScanner::new(vec![]);
        assert!(!scanner.next());
        assert!(scanner.err().is_none());
    }

    #[test]
    fn test_vec_scanner_with_rows() {
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::text("a")]),
            Row::from_values(vec![Value::Integer(2), Value::text("b")]),
            Row::from_values(vec![Value::Integer(3), Value::text("c")]),
        ];

        let mut scanner = VecScanner::new(rows);

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(1)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(2)));

        assert!(scanner.next());
        assert_eq!(scanner.row().get(0), Some(&Value::Integer(3)));

        assert!(!scanner.next());
        assert!(scanner.err().is_none());
    }

    #[test]
    fn test_vec_scanner_with_error() {
        let mut scanner = VecScanner::with_error(crate::core::Error::internal("test error"));
        assert!(!scanner.next());
        assert!(scanner.err().is_some());
    }

    #[test]
    fn test_vec_scanner_close() {
        let rows = vec![Row::from_values(vec![Value::Integer(1)])];
        let mut scanner = VecScanner::new(rows);

        assert!(scanner.next());
        assert!(scanner.close().is_ok());

        // After close, next should return false
        assert!(!scanner.next());
    }
}
