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
/// let scanner = table.scan(&[0, 1], None, Deadline::NONE)?;
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

    /// Returns an estimate of the total number of rows in the scan.
    ///
    /// This is used for pre-allocating vectors to avoid reallocations.
    /// Returns None if the count is unknown. Default implementation returns None.
    fn estimated_count(&self) -> Option<usize> {
        None
    }

    /// Takes ownership of the current row along with its row ID
    ///
    /// This is more efficient than `row().clone()` when you need to move
    /// the row data out of the scanner with its associated ID. After calling this,
    /// the internal row buffer may be empty until `next()` is called again.
    /// The default implementation returns the current position as the ID.
    fn take_row_with_id(&mut self) -> (i64, Row) {
        (0, self.take_row())
    }

    /// Returns the current row ID
    ///
    /// Returns the internal row ID for the current row. This is used for
    /// row-level operations that need to track row identity.
    /// Default implementation returns 0.
    fn current_row_id(&self) -> i64 {
        0
    }
}
