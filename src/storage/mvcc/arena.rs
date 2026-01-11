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

//! Arena-based row storage for O(1) clone reads
//!
//! This module provides Arc-based storage for row data,
//! enabling O(1) row cloning on read (just Arc::clone).
//!
//! Key insight: Store row data as Arc<[CompactArc<Value>]> for O(1) clone on read.
//! Single clone on insert, zero clone on read.
//!
//! # Lock Design
//!
//! Uses a single RwLock for both data and metadata to:
//! - Ensure consistent lock ordering (eliminates deadlock risk)
//! - Reduce lock acquisition overhead (one lock vs two)
//! - Guarantee atomic insert operations

use std::sync::Arc;

use parking_lot::RwLock;

use crate::common::CompactArc;
use crate::core::{Row, Value};

/// Metadata for a row stored in the arena
#[derive(Clone, Copy, Debug)]
pub struct ArenaRowMeta {
    /// Row ID
    pub row_id: i64,
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
}

/// Inner arena data protected by single lock
pub struct ArenaInner {
    /// Arc storage per row (for O(1) clone on read)
    /// Stores Arc<[CompactArc<Value>]> matching the new Row storage format
    pub data: Vec<Arc<[CompactArc<Value>]>>,
    /// Row metadata
    pub meta: Vec<ArenaRowMeta>,
}

/// Arena-based storage for row data
///
/// Row data is stored as Arc<[Value]> for O(1) clone on read.
/// Single clone on insert, zero clone on read.
///
/// Uses single RwLock for atomicity and deadlock prevention.
pub struct RowArena {
    /// Combined data and metadata under single lock
    inner: RwLock<ArenaInner>,
}

impl RowArena {
    /// Create a new arena
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(ArenaInner {
                data: Vec::with_capacity(10_000),
                meta: Vec::with_capacity(10_000),
            }),
        }
    }

    /// Create a new arena with pre-allocated capacity
    pub fn with_capacity(row_capacity: usize) -> Self {
        Self {
            inner: RwLock::new(ArenaInner {
                data: Vec::with_capacity(row_capacity),
                meta: Vec::with_capacity(row_capacity),
            }),
        }
    }

    /// Insert a row into the arena
    ///
    /// Returns the index of the row metadata
    #[inline]
    pub fn insert(&self, row_id: i64, txn_id: i64, create_time: i64, values: &[Value]) -> usize {
        let mut inner = self.inner.write();

        // Convert values to CompactArc<Value> and store
        let arc_values: Vec<CompactArc<Value>> =
            values.iter().map(|v| CompactArc::new(v.clone())).collect();
        inner.data.push(Arc::from(arc_values));

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = inner.meta.len();
        inner.meta.push(meta);
        idx
    }

    /// Insert a row from a Row struct
    /// Handles all storage types: Inline values are wrapped in Arc, Owned/Shared are cloned.
    #[inline]
    pub fn insert_row(&self, row_id: i64, txn_id: i64, create_time: i64, row: &Row) -> usize {
        let mut inner = self.inner.write();

        // Store Arc for O(1) clone on read - handle all storage types
        let arc_data = match row.try_as_arc_slice() {
            Some(slice) => Arc::from(slice),
            None => {
                // Inline storage: wrap values in Arc
                let arc_vec: Vec<CompactArc<Value>> =
                    row.iter().map(|v| CompactArc::new(v.clone())).collect();
                Arc::from(arc_vec.into_boxed_slice())
            }
        };
        inner.data.push(arc_data);

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = inner.meta.len();
        inner.meta.push(meta);
        idx
    }

    /// Insert a row and return both the index AND the Arc
    /// This allows the caller to reuse the Arc for O(1) clones
    /// Handles all storage types: Inline values are wrapped in Arc, Owned/Shared are cloned.
    #[inline]
    pub fn insert_row_get_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        create_time: i64,
        row: &Row,
    ) -> (usize, Arc<[CompactArc<Value>]>) {
        let mut inner = self.inner.write();

        // Create Arc once, clone for storage and return - handle all storage types
        let arc_data = match row.try_as_arc_slice() {
            Some(slice) => Arc::from(slice),
            None => {
                // Inline storage: wrap values in Arc
                let arc_vec: Vec<CompactArc<Value>> =
                    row.iter().map(|v| CompactArc::new(v.clone())).collect();
                Arc::from(arc_vec.into_boxed_slice())
            }
        };
        inner.data.push(Arc::clone(&arc_data));

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = inner.meta.len();
        inner.meta.push(meta);
        (idx, arc_data)
    }

    /// Insert an already-created Arc directly - avoids copy when caller has Arc
    /// Returns the index where it was stored
    #[inline]
    pub fn insert_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        create_time: i64,
        arc_data: Arc<[CompactArc<Value>]>,
    ) -> usize {
        let mut inner = self.inner.write();

        inner.data.push(arc_data);

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        let idx = inner.meta.len();
        inner.meta.push(meta);
        idx
    }

    /// Mark a row as deleted
    #[inline]
    pub fn mark_deleted(&self, row_idx: usize, deleted_at_txn_id: i64) {
        let mut inner = self.inner.write();
        if row_idx < inner.meta.len() {
            inner.meta[row_idx].deleted_at_txn_id = deleted_at_txn_id;
        }
    }

    /// Update data at an existing arena index (for slot reuse during UPDATEs)
    ///
    /// This replaces both data and metadata at the given index, avoiding
    /// unbounded arena growth during update-heavy workloads.
    /// Returns true if the update was successful, false if index out of bounds.
    #[inline]
    pub fn update_at(
        &self,
        arena_idx: usize,
        row_id: i64,
        txn_id: i64,
        create_time: i64,
        arc_data: Arc<[CompactArc<Value>]>,
    ) -> bool {
        let mut inner = self.inner.write();
        if arena_idx < inner.meta.len() {
            inner.data[arena_idx] = arc_data;
            inner.meta[arena_idx] = ArenaRowMeta {
                row_id,
                txn_id,
                deleted_at_txn_id: 0,
                create_time,
            };
            true
        } else {
            false
        }
    }

    /// Get the number of rows (including deleted)
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.read().meta.len()
    }

    /// Check if the arena is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.read().meta.is_empty()
    }

    /// Get read guard for arena data
    ///
    /// Returns a guard that provides access to both metadata and data slices.
    /// Single lock acquisition for both.
    #[inline]
    pub fn read_guard(&self) -> ArenaReadGuard<'_> {
        ArenaReadGuard {
            inner: self.inner.read(),
        }
    }

    /// Get Arc for a row by arena index - O(1) clone
    #[inline]
    pub fn get_arc(&self, arena_idx: usize) -> Option<Arc<[CompactArc<Value>]>> {
        let inner = self.inner.read();
        inner.data.get(arena_idx).cloned()
    }

    /// Get both metadata and Arc for a row by arena index - O(1) with single lock
    ///
    /// This is optimized for the visibility fast path where we need to check
    /// txn_id and deleted_at_txn_id before returning the data.
    #[inline]
    pub fn get_meta_and_arc(
        &self,
        arena_idx: usize,
    ) -> Option<(ArenaRowMeta, Arc<[CompactArc<Value>]>)> {
        let inner = self.inner.read();
        if arena_idx < inner.meta.len() {
            let meta = inner.meta[arena_idx];
            let arc = Arc::clone(&inner.data[arena_idx]);
            Some((meta, arc))
        } else {
            None
        }
    }
}

impl Default for RowArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Read guard for arena providing access to both data and metadata
pub struct ArenaReadGuard<'a> {
    inner: parking_lot::RwLockReadGuard<'a, ArenaInner>,
}

impl<'a> ArenaReadGuard<'a> {
    /// Get data slice
    #[inline]
    pub fn data(&self) -> &[Arc<[CompactArc<Value>]>] {
        &self.inner.data
    }

    /// Get metadata slice
    #[inline]
    pub fn meta(&self) -> &[ArenaRowMeta] {
        &self.inner.meta
    }

    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.meta.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.meta.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_arena_insert_and_iterate() {
        let arena = RowArena::new();

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

        // Iterate using read_guard
        let guard = arena.read_guard();
        let mut count = 0;
        let mut sum = 0.0f64;

        for (idx, _meta) in guard.meta().iter().enumerate() {
            count += 1;
            if let Some(arc_val) = guard.data()[idx].get(2) {
                if let Value::Float(v) = &**arc_val {
                    sum += v;
                }
            }
        }

        assert_eq!(count, 3);
        assert_eq!(sum, 600.0);
    }

    #[test]
    fn test_arena_arc_clone() {
        let arena = RowArena::new();

        arena.insert(1, 100, 1000, &[Value::Integer(42), Value::text("test")]);

        let guard = arena.read_guard();
        assert_eq!(guard.len(), 1);

        // Get row as Arc - O(1) clone
        let row_arc = Arc::clone(&guard.data()[0]);
        assert_eq!(row_arc.len(), 2);

        if let Value::Integer(v) = &*row_arc[0] {
            assert_eq!(*v, 42);
        } else {
            panic!("Expected Integer");
        }
    }

    #[test]
    fn test_arena_deletion() {
        let arena = RowArena::new();

        let idx = arena.insert(1, 100, 1000, &[Value::Integer(1), Value::text("test")]);

        // Mark as deleted
        arena.mark_deleted(idx, 101);

        let guard = arena.read_guard();
        assert!(guard.meta()[0].is_deleted());
        assert_eq!(guard.meta()[0].deleted_at_txn_id, 101);
    }

    #[test]
    fn test_arena_get_column_value() {
        let arena = RowArena::new();

        arena.insert(
            1,
            100,
            1000,
            &[Value::Integer(42), Value::text("hello"), Value::Float(3.15)],
        );

        let guard = arena.read_guard();

        // Get column 0
        let val = guard.data()[0].first().map(|arc| (**arc).clone());
        assert_eq!(val, Some(Value::Integer(42)));

        // Get column 1
        let val = guard.data()[0].get(1).map(|arc| (**arc).clone());
        assert_eq!(val, Some(Value::text("hello")));

        // Get column 2
        let val = guard.data()[0].get(2).map(|arc| (**arc).clone());
        assert_eq!(val, Some(Value::Float(3.15)));

        // Out of bounds column - returns None
        let val: Option<Value> = guard.data()[0].get(3).map(|arc| (**arc).clone());
        assert_eq!(val, None);
    }

    #[test]
    fn test_arena_read_guard() {
        let arena = RowArena::new();

        arena.insert(1, 100, 1000, &[Value::Integer(1), Value::text("a")]);
        arena.insert(2, 100, 1001, &[Value::Integer(2), Value::text("b")]);

        let guard = arena.read_guard();
        assert_eq!(guard.len(), 2);
        assert_eq!(guard.meta()[0].row_id, 1);
        assert_eq!(guard.meta()[1].row_id, 2);
    }
}
