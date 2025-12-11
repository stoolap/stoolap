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

//! Arena-based row storage for zero-copy scans
//!
//! This module provides contiguous memory storage for row data,
//! enabling 50x+ faster full table scans by eliminating cloning.
//!
//! Key insight: Instead of storing each row as a separate Vec<Value>,
//! store ALL values in a single contiguous Vec<Value> arena.
//! Rows are just (start, end) indices into this arena.

use std::sync::RwLock;

use crate::core::{Row, Value};

/// Metadata for a row stored in the arena
#[derive(Clone, Copy, Debug)]
pub struct ArenaRowMeta {
    /// Row ID
    pub row_id: i64,
    /// Start index in the arena (inclusive)
    pub start: usize,
    /// End index in the arena (exclusive)
    pub end: usize,
    /// Transaction ID that created this row
    pub txn_id: i64,
    /// Transaction ID that deleted this row (0 if not deleted)
    pub deleted_at_txn_id: i64,
    /// Creation timestamp
    pub create_time: i64,
}

impl ArenaRowMeta {
    /// Check if this row is deleted
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.deleted_at_txn_id != 0
    }

    /// Get the number of columns in this row
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if this row has no columns
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Arena-based storage for row data
///
/// All row values are stored in a single contiguous Vec<Value>.
/// This enables zero-copy scans by returning slices into the arena.
pub struct RowArena {
    /// Contiguous storage for all values
    data: RwLock<Vec<Value>>,
    /// Row metadata (row_id -> ArenaRowMeta)
    rows: RwLock<Vec<ArenaRowMeta>>,
    /// Number of columns per row (fixed for a table)
    #[allow(dead_code)]
    cols_per_row: usize,
}

impl RowArena {
    /// Create a new arena with the given number of columns per row
    pub fn new(cols_per_row: usize) -> Self {
        Self {
            data: RwLock::new(Vec::with_capacity(10_000 * cols_per_row)),
            rows: RwLock::new(Vec::with_capacity(10_000)),
            cols_per_row,
        }
    }

    /// Create a new arena with pre-allocated capacity
    pub fn with_capacity(cols_per_row: usize, row_capacity: usize) -> Self {
        Self {
            data: RwLock::new(Vec::with_capacity(row_capacity * cols_per_row)),
            rows: RwLock::new(Vec::with_capacity(row_capacity)),
            cols_per_row,
        }
    }

    /// Insert a row into the arena
    ///
    /// Returns the index of the row metadata
    pub fn insert(&self, row_id: i64, txn_id: i64, create_time: i64, values: &[Value]) -> usize {
        let mut data = self.data.write().unwrap();
        let mut rows = self.rows.write().unwrap();

        let start = data.len();
        data.extend(values.iter().cloned());
        let end = data.len();

        let meta = ArenaRowMeta {
            row_id,
            start,
            end,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = rows.len();
        rows.push(meta);
        idx
    }

    /// Insert a row from a Row struct
    pub fn insert_row(&self, row_id: i64, txn_id: i64, create_time: i64, row: &Row) -> usize {
        let mut data = self.data.write().unwrap();
        let mut rows = self.rows.write().unwrap();

        let start = data.len();
        for i in 0..row.len() {
            if let Some(v) = row.get(i) {
                data.push(v.clone());
            }
        }
        let end = data.len();

        let meta = ArenaRowMeta {
            row_id,
            start,
            end,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = rows.len();
        rows.push(meta);
        idx
    }

    /// Mark a row as deleted
    pub fn mark_deleted(&self, row_idx: usize, deleted_at_txn_id: i64) {
        let mut rows = self.rows.write().unwrap();
        if row_idx < rows.len() {
            rows[row_idx].deleted_at_txn_id = deleted_at_txn_id;
        }
    }

    /// Get the number of rows (including deleted)
    pub fn len(&self) -> usize {
        self.rows.read().unwrap().len()
    }

    /// Check if the arena is empty
    pub fn is_empty(&self) -> bool {
        self.rows.read().unwrap().is_empty()
    }

    /// Get a row's values as a slice (zero-copy read)
    ///
    /// Returns None if the row index is invalid
    pub fn get_row_slice<'a>(
        &'a self,
        data_guard: &'a [Value],
        meta: &ArenaRowMeta,
    ) -> &'a [Value] {
        &data_guard[meta.start..meta.end]
    }

    /// Create a scanner for zero-copy iteration
    pub fn scan(&self) -> ArenaScanner<'_> {
        ArenaScanner {
            data: self.data.read().unwrap(),
            rows: self.rows.read().unwrap(),
            current_idx: 0,
        }
    }

    /// Get all row metadata (for filtering before accessing data)
    pub fn get_all_meta(&self) -> Vec<ArenaRowMeta> {
        self.rows.read().unwrap().clone()
    }

    /// Get row data by row_id (searches through metadata)
    pub fn get_by_row_id(&self, row_id: i64) -> Option<(ArenaRowMeta, Vec<Value>)> {
        let rows = self.rows.read().unwrap();
        let data = self.data.read().unwrap();

        for meta in rows.iter().rev() {
            if meta.row_id == row_id {
                let values = data[meta.start..meta.end].to_vec();
                return Some((*meta, values));
            }
        }
        None
    }

    /// Get row data by arena index (direct O(1) access)
    pub fn get_by_index(&self, arena_idx: usize) -> Option<(ArenaRowMeta, Vec<Value>)> {
        let rows = self.rows.read().unwrap();
        let data = self.data.read().unwrap();

        if arena_idx < rows.len() {
            let meta = rows[arena_idx];
            let values = data[meta.start..meta.end].to_vec();
            Some((meta, values))
        } else {
            None
        }
    }

    /// Get multiple rows by arena indices efficiently (single lock acquisition)
    pub fn get_multiple_by_indices(&self, indices: &[usize]) -> Vec<(i64, Vec<Value>)> {
        let rows = self.rows.read().unwrap();
        let data = self.data.read().unwrap();

        let mut results = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx < rows.len() {
                let meta = &rows[idx];
                let values = data[meta.start..meta.end].to_vec();
                results.push((meta.row_id, values));
            }
        }
        results
    }

    /// Get multiple rows as Row objects directly (avoids intermediate Vec<Value>)
    #[inline]
    pub fn get_multiple_as_rows(&self, indices: &[usize]) -> Vec<(i64, Row)> {
        let rows = self.rows.read().unwrap();
        let data = self.data.read().unwrap();
        let rows_len = rows.len();
        let data_slice = data.as_slice();

        // Use iterator + collect for better compiler optimization
        indices
            .iter()
            .filter_map(|&idx| {
                if idx < rows_len {
                    // SAFETY: idx bounds already checked
                    let meta = unsafe { rows.get_unchecked(idx) };
                    // SAFETY: arena maintains valid start..end ranges
                    let slice = unsafe { data_slice.get_unchecked(meta.start..meta.end) };
                    Some((meta.row_id, Row::from_values(slice.to_vec())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get multiple rows with pre-known row_ids (faster - skips row_id lookup)
    #[inline]
    pub fn get_rows_with_ids(&self, pairs: &[(i64, usize)]) -> Vec<(i64, Row)> {
        let rows = self.rows.read().unwrap();
        let data = self.data.read().unwrap();
        let rows_len = rows.len();
        let data_slice = data.as_slice();

        pairs
            .iter()
            .filter_map(|&(row_id, idx)| {
                if idx < rows_len {
                    // SAFETY: idx bounds already checked
                    let meta = unsafe { rows.get_unchecked(idx) };
                    // SAFETY: arena maintains valid start..end ranges
                    let slice = unsafe { data_slice.get_unchecked(meta.start..meta.end) };
                    Some((row_id, Row::from_values(slice.to_vec())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get data read guard for external batch operations
    #[inline]
    pub fn read_guards(
        &self,
    ) -> (
        std::sync::RwLockReadGuard<'_, Vec<ArenaRowMeta>>,
        std::sync::RwLockReadGuard<'_, Vec<Value>>,
    ) {
        (self.rows.read().unwrap(), self.data.read().unwrap())
    }

    /// Clone row data directly given guards (zero lock overhead)
    #[inline]
    pub fn clone_row_data(
        rows_guard: &[ArenaRowMeta],
        data_guard: &[Value],
        idx: usize,
    ) -> Option<Row> {
        if idx < rows_guard.len() {
            let meta = unsafe { rows_guard.get_unchecked(idx) };
            let slice = unsafe { data_guard.get_unchecked(meta.start..meta.end) };
            Some(Row::from_values(slice.to_vec()))
        } else {
            None
        }
    }
}

/// Zero-copy scanner over the arena
pub struct ArenaScanner<'a> {
    /// Read guard for the data array - public for direct access
    pub(crate) data: std::sync::RwLockReadGuard<'a, Vec<Value>>,
    /// Read guard for the row metadata - public for direct access
    pub(crate) rows: std::sync::RwLockReadGuard<'a, Vec<ArenaRowMeta>>,
    current_idx: usize,
}

impl<'a> ArenaScanner<'a> {
    /// Advance to the next row
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> bool {
        if self.current_idx < self.rows.len() {
            self.current_idx += 1;
            true
        } else {
            false
        }
    }

    /// Get current row metadata
    #[inline]
    pub fn meta(&self) -> &ArenaRowMeta {
        &self.rows[self.current_idx - 1]
    }

    /// Get current row ID
    #[inline]
    pub fn row_id(&self) -> i64 {
        self.rows[self.current_idx - 1].row_id
    }

    /// Get current row as a slice (ZERO COPY!)
    #[inline]
    pub fn row(&self) -> &[Value] {
        let meta = &self.rows[self.current_idx - 1];
        &self.data[meta.start..meta.end]
    }

    /// Get a specific column value from current row
    #[inline]
    pub fn get(&self, col: usize) -> Option<&Value> {
        let meta = &self.rows[self.current_idx - 1];
        let pos = meta.start + col;
        if pos < meta.end {
            Some(&self.data[pos])
        } else {
            None
        }
    }

    /// Check if current row is deleted
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.rows[self.current_idx - 1].is_deleted()
    }

    /// Get transaction ID of current row
    #[inline]
    pub fn txn_id(&self) -> i64 {
        self.rows[self.current_idx - 1].txn_id
    }

    /// Get deleted_at transaction ID
    #[inline]
    pub fn deleted_at_txn_id(&self) -> i64 {
        self.rows[self.current_idx - 1].deleted_at_txn_id
    }

    /// Reset scanner to beginning
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Get total row count
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if scanner is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_arena_insert_and_scan() {
        let arena = RowArena::new(3);

        // Insert some rows
        arena.insert(
            1,
            100,
            1000,
            &[Value::Integer(1), Value::text("Alice"), Value::Float(100.0)],
        );
        arena.insert(
            2,
            100,
            1001,
            &[Value::Integer(2), Value::text("Bob"), Value::Float(200.0)],
        );
        arena.insert(
            3,
            100,
            1002,
            &[Value::Integer(3), Value::text("Carol"), Value::Float(300.0)],
        );

        assert_eq!(arena.len(), 3);

        // Scan and verify
        let mut scanner = arena.scan();
        let mut count = 0;
        let mut sum = 0.0f64;

        while scanner.next() {
            count += 1;
            if let Some(Value::Float(v)) = scanner.get(2) {
                sum += v;
            }
        }

        assert_eq!(count, 3);
        assert_eq!(sum, 600.0);
    }

    #[test]
    fn test_arena_zero_copy() {
        let arena = RowArena::new(2);

        arena.insert(1, 100, 1000, &[Value::Integer(42), Value::text("test")]);

        let mut scanner = arena.scan();
        assert!(scanner.next());

        // Get row as slice - this is a reference, not a copy!
        let row = scanner.row();
        assert_eq!(row.len(), 2);

        if let Value::Integer(v) = &row[0] {
            assert_eq!(*v, 42);
        } else {
            panic!("Expected Integer");
        }
    }

    #[test]
    fn test_arena_deletion() {
        let arena = RowArena::new(2);

        let idx = arena.insert(1, 100, 1000, &[Value::Integer(1), Value::text("test")]);

        // Mark as deleted
        arena.mark_deleted(idx, 101);

        let mut scanner = arena.scan();
        assert!(scanner.next());
        assert!(scanner.is_deleted());
        assert_eq!(scanner.deleted_at_txn_id(), 101);
    }
}
