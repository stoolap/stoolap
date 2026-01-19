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
//! Key insight: Store row data as Arc<[Value]> for O(1) clone on read.
//! Single clone on insert, zero clone on read.
//!
//! # Lock Design
//!
//! Uses a single RwLock for both data and metadata to:
//! - Ensure consistent lock ordering (eliminates deadlock risk)
//! - Reduce lock acquisition overhead (one lock vs two)
//! - Guarantee atomic insert operations

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

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
///
/// NOTE: This struct is on the hot read path. Keep it minimal.
/// The free_list is stored separately in RowArena to avoid bloating this struct.
pub struct ArenaInner {
    /// Arc storage per row (for O(1) clone on read)
    /// Stores Arc<[Value]> for efficient row sharing
    pub data: Vec<Arc<[Value]>>,
    /// Row metadata
    pub meta: Vec<ArenaRowMeta>,
}

/// Arena-based storage for row data
///
/// Row data is stored as Arc<[Value]> for O(1) clone on read.
/// Single clone on insert, zero clone on read.
///
/// Uses single RwLock for atomicity and deadlock prevention.
/// Free list is stored separately to avoid bloating the hot read path.
pub struct RowArena {
    /// Combined data and metadata under single lock
    inner: RwLock<ArenaInner>,
    /// Free list of cleared slot indices for reuse (separate lock, write-path only)
    /// This prevents unbounded arena growth during insert/delete cycles
    free_list: Mutex<Vec<usize>>,
}

impl RowArena {
    /// Create a new arena
    /// Note: Pre-allocate 10,000 slots to avoid reallocation during bulk inserts.
    /// Vec doubling during growth can cause peak memory spikes.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(ArenaInner {
                data: Vec::with_capacity(10_000),
                meta: Vec::with_capacity(10_000),
            }),
            free_list: Mutex::new(Vec::new()),
        }
    }

    /// Create a new arena with pre-allocated capacity
    pub fn with_capacity(row_capacity: usize) -> Self {
        Self {
            inner: RwLock::new(ArenaInner {
                data: Vec::with_capacity(row_capacity),
                meta: Vec::with_capacity(row_capacity),
            }),
            free_list: Mutex::new(Vec::new()),
        }
    }

    /// Insert a row into the arena
    ///
    /// Returns the index of the row metadata.
    /// Reuses cleared slots from the free list to prevent unbounded growth.
    #[inline]
    pub fn insert(&self, row_id: i64, txn_id: i64, create_time: i64, values: &[Value]) -> usize {
        // Check free list first (separate lock, doesn't affect read path)
        let reuse_idx = self.free_list.lock().pop();

        let mut inner = self.inner.write();

        // Convert values to Arc<[Value]>
        let arc_data: Arc<[Value]> = Arc::from(values.to_vec().into_boxed_slice());

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        // Reuse a slot from the free list if available
        if let Some(idx) = reuse_idx {
            inner.data[idx] = arc_data;
            inner.meta[idx] = meta;
            idx
        } else {
            inner.data.push(arc_data);
            let idx = inner.meta.len();
            inner.meta.push(meta);
            idx
        }
    }

    /// Insert a row from a Row struct
    /// Handles all storage types: Shared Arc is cloned O(1), Owned values create new Arc.
    /// Reuses cleared slots from the free list to prevent unbounded growth.
    #[inline]
    pub fn insert_row(&self, row_id: i64, txn_id: i64, create_time: i64, row: &Row) -> usize {
        // Check free list first (separate lock, doesn't affect read path)
        let reuse_idx = self.free_list.lock().pop();

        let mut inner = self.inner.write();

        // Store Arc for O(1) clone on read
        // If row is Shared, clone the Arc (O(1))
        // If row is Owned, create new Arc from values
        let arc_data = match row.as_arc() {
            Some(arc) => Arc::clone(arc),
            None => {
                // Owned storage: create new Arc from values
                let values: Vec<Value> = row.iter().cloned().collect();
                Arc::from(values.into_boxed_slice())
            }
        };

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        // Reuse a slot from the free list if available
        if let Some(idx) = reuse_idx {
            inner.data[idx] = arc_data;
            inner.meta[idx] = meta;
            idx
        } else {
            inner.data.push(arc_data);
            let idx = inner.meta.len();
            inner.meta.push(meta);
            idx
        }
    }

    /// Insert a row and return both the index AND the Arc
    /// This allows the caller to reuse the Arc for O(1) clones
    /// Handles all storage types: Shared Arc is cloned O(1), Owned values create new Arc.
    /// Reuses cleared slots from the free list to prevent unbounded growth.
    #[inline]
    pub fn insert_row_get_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        create_time: i64,
        row: &Row,
    ) -> (usize, Arc<[Value]>) {
        // Check free list first (separate lock, doesn't affect read path)
        let reuse_idx = self.free_list.lock().pop();

        let mut inner = self.inner.write();

        // Create Arc once, clone for storage and return
        let arc_data = match row.as_arc() {
            Some(arc) => Arc::clone(arc),
            None => {
                // Owned storage: create new Arc from values
                let values: Vec<Value> = row.iter().cloned().collect();
                Arc::from(values.into_boxed_slice())
            }
        };

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        // Reuse a slot from the free list if available
        let idx = if let Some(idx) = reuse_idx {
            inner.data[idx] = Arc::clone(&arc_data);
            inner.meta[idx] = meta;
            idx
        } else {
            inner.data.push(Arc::clone(&arc_data));
            let idx = inner.meta.len();
            inner.meta.push(meta);
            idx
        };
        (idx, arc_data)
    }

    /// Insert an already-created Arc directly - avoids copy when caller has Arc
    /// Returns the index where it was stored
    /// Reuses cleared slots from the free list to prevent unbounded growth.
    #[inline]
    pub fn insert_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        create_time: i64,
        arc_data: Arc<[Value]>,
    ) -> usize {
        // Check free list first (separate lock, doesn't affect read path)
        let reuse_idx = self.free_list.lock().pop();

        let mut inner = self.inner.write();

        let meta = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            create_time,
        };

        // Reuse a slot from the free list if available
        if let Some(idx) = reuse_idx {
            inner.data[idx] = arc_data;
            inner.meta[idx] = meta;
            idx
        } else {
            inner.data.push(arc_data);
            let idx = inner.meta.len();
            inner.meta.push(meta);
            idx
        }
    }

    /// Mark a row as deleted
    #[inline]
    pub fn mark_deleted(&self, row_idx: usize, deleted_at_txn_id: i64) {
        let mut inner = self.inner.write();
        if row_idx < inner.meta.len() {
            inner.meta[row_idx].deleted_at_txn_id = deleted_at_txn_id;
        }
    }

    /// Clear a slot in the arena to release memory
    ///
    /// This replaces the data with an empty Arc and marks the metadata as cleared
    /// (row_id = 0). The slot index is added to the free list for reuse.
    /// This is used during cleanup of deleted rows.
    #[inline]
    pub fn clear_at(&self, arena_idx: usize) -> bool {
        let cleared = {
            let mut inner = self.inner.write();
            if arena_idx < inner.meta.len() {
                // Replace data with empty Arc to release memory
                inner.data[arena_idx] = Arc::from(Vec::<Value>::new().into_boxed_slice());
                // Mark metadata as cleared (row_id = 0 indicates cleared slot)
                inner.meta[arena_idx] = ArenaRowMeta {
                    row_id: 0,
                    txn_id: 0,
                    deleted_at_txn_id: 0,
                    create_time: 0,
                };
                true
            } else {
                false
            }
        };

        if cleared {
            // Add to free list for reuse (separate lock)
            self.free_list.lock().push(arena_idx);
        }
        cleared
    }

    /// Clear multiple slots in the arena efficiently (single lock acquisition)
    ///
    /// Returns the number of slots actually cleared.
    /// Cleared indices are added to the free list for reuse.
    #[inline]
    pub fn clear_batch(&self, arena_indices: &[usize]) -> usize {
        let mut cleared_indices = Vec::with_capacity(arena_indices.len());

        {
            let mut inner = self.inner.write();
            let empty_data: Arc<[Value]> = Arc::from(Vec::<Value>::new().into_boxed_slice());
            let cleared_meta = ArenaRowMeta {
                row_id: 0,
                txn_id: 0,
                deleted_at_txn_id: 0,
                create_time: 0,
            };

            for &arena_idx in arena_indices {
                if arena_idx < inner.meta.len() {
                    inner.data[arena_idx] = Arc::clone(&empty_data);
                    inner.meta[arena_idx] = cleared_meta;
                    cleared_indices.push(arena_idx);
                }
            }
        }

        // Add to free list for reuse (separate lock, after releasing inner lock)
        let count = cleared_indices.len();
        if count > 0 {
            let mut free_list = self.free_list.lock();
            free_list.reserve(count);
            free_list.extend(cleared_indices);
        }
        count
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
        arc_data: Arc<[Value]>,
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
    pub fn get_arc(&self, arena_idx: usize) -> Option<Arc<[Value]>> {
        let inner = self.inner.read();
        inner.data.get(arena_idx).cloned()
    }

    /// Get both metadata and Arc for a row by arena index - O(1) with single lock
    ///
    /// This is optimized for the visibility fast path where we need to check
    /// txn_id and deleted_at_txn_id before returning the data.
    #[inline]
    pub fn get_meta_and_arc(&self, arena_idx: usize) -> Option<(ArenaRowMeta, Arc<[Value]>)> {
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
    pub fn data(&self) -> &[Arc<[Value]>] {
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
            if let Some(Value::Float(v)) = guard.data()[idx].get(2) {
                sum += v;
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

        if let Value::Integer(v) = &row_arc[0] {
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
        let val = guard.data()[0].first().cloned();
        assert_eq!(val, Some(Value::Integer(42)));

        // Get column 1
        let val = guard.data()[0].get(1).cloned();
        assert_eq!(val, Some(Value::text("hello")));

        // Get column 2
        let val = guard.data()[0].get(2).cloned();
        assert_eq!(val, Some(Value::Float(3.15)));

        // Out of bounds column - returns None
        let val: Option<Value> = guard.data()[0].get(3).cloned();
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

    #[test]
    fn test_arena_free_list_reuse() {
        let arena = RowArena::new();

        // Insert 5 rows
        for i in 1..=5 {
            arena.insert(
                i,
                100,
                1000 + i,
                &[Value::Integer(i), Value::text(format!("row{}", i))],
            );
        }
        assert_eq!(arena.len(), 5);

        // Clear slots 1 and 3 (0-indexed)
        arena.clear_at(1);
        arena.clear_at(3);

        // Arena len is still 5 (slots are cleared but not removed)
        assert_eq!(arena.len(), 5);

        // Verify cleared slots have row_id = 0
        {
            let guard = arena.read_guard();
            assert_eq!(guard.meta()[1].row_id, 0);
            assert_eq!(guard.meta()[3].row_id, 0);
        }

        // Insert new rows - should reuse cleared slots (3 then 1, LIFO order)
        let idx1 = arena.insert(10, 200, 2000, &[Value::Integer(10), Value::text("new1")]);
        let idx2 = arena.insert(11, 200, 2001, &[Value::Integer(11), Value::text("new2")]);

        // Slots should be reused, not appended
        assert_eq!(arena.len(), 5); // Still 5, slots were reused

        // Verify indices are the cleared slots (LIFO: 3 was pushed last, so popped first)
        assert_eq!(idx1, 3);
        assert_eq!(idx2, 1);

        // Verify the new data is in the reused slots
        {
            let guard = arena.read_guard();
            assert_eq!(guard.meta()[3].row_id, 10);
            assert_eq!(guard.meta()[1].row_id, 11);
        }

        // Insert one more - should append since free list is empty
        let idx3 = arena.insert(12, 200, 2002, &[Value::Integer(12), Value::text("new3")]);
        assert_eq!(idx3, 5); // New slot at end
        assert_eq!(arena.len(), 6);
    }

    #[test]
    fn test_arena_clear_batch_free_list() {
        let arena = RowArena::new();

        // Insert 10 rows
        for i in 0..10 {
            arena.insert(i, 100, 1000 + i, &[Value::Integer(i)]);
        }
        assert_eq!(arena.len(), 10);

        // Clear slots 2, 4, 6, 8 in batch
        let cleared = arena.clear_batch(&[2, 4, 6, 8]);
        assert_eq!(cleared, 4);
        assert_eq!(arena.len(), 10); // Still 10, slots cleared but not removed

        // Insert 4 new rows - should reuse all 4 cleared slots
        let mut new_indices = Vec::new();
        for i in 100..104 {
            new_indices.push(arena.insert(i, 200, 2000 + i, &[Value::Integer(i)]));
        }

        // Should still be 10 (all slots reused)
        assert_eq!(arena.len(), 10);

        // Verify all new indices are from cleared slots (8, 6, 4, 2 in LIFO order)
        assert!(new_indices.iter().all(|&idx| [2, 4, 6, 8].contains(&idx)));
    }
}
