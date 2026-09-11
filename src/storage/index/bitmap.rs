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

//! Bitmap Index implementation for low-cardinality columns
//!
//! This module provides a bitmap-based index optimized for columns with
//! few distinct values (< 1000). Uses RoaringBitmap for compression.
//!
//! ## Performance characteristics:
//! - INSERT: O(1)
//! - DELETE: O(1)
//! - FIND exact: O(1) bitmap lookup
//! - AND/OR operations: O(n/64) bitwise operations
//!
//! ## When to use BitmapIndex:
//! - BOOLEAN columns (only 2 values)
//! - Status/state columns (pending, active, completed, etc.)
//! - Category columns with few values
//! - Any column with < 1000 distinct values
//!
//! ## When NOT to use BitmapIndex:
//! - High-cardinality columns (use Hash or BTree)
//! - Range queries on numeric data (use BTree)
//! - Unique columns (use Hash or BTree)
//!
//! ## Key advantage:
//! Multi-predicate queries (WHERE a = 1 AND b = 2 AND c = 3) can be
//! answered with O(n/64) bitwise AND operations instead of O(n) scans.
//!
//! ## Implementation:
//! Uses `roaring` crate (RoaringBitmap) - same as Lucene, Druid, Spark.
//! Automatic compression: array for sparse, bitmap for dense, RLE for runs.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

use ahash::AHashMap;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::common::{CompactArc, I64Map};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;
use crate::storage::index::memory::{
    btree_node_bytes, hash_table_bytes, value_bytes, IndexMemory, IndexMemoryOwner, IndexValueMap,
};
use crate::storage::traits::Index;

/// Warning threshold for cardinality
const HIGH_CARDINALITY_WARNING_THRESHOLD: usize = 1000;

/// Bitmap Index for low-cardinality columns
///
/// Optimized for BOOLEAN, status, and category columns where the number
/// of distinct values is small (< 1000).
///
/// ## Key features:
/// - One RoaringBitmap per distinct value
/// - O(n/64) AND/OR/NOT operations for multi-predicate queries
/// - Automatic compression (array, bitmap, or RLE per 8KB chunk)
/// - Thread-safe with RwLock
///
/// ## Memory efficiency:
/// For 1M rows with 10 distinct values:
/// - B-tree: ~80 MB
/// - Bitmap: ~500KB - 2.5MB (30-160x savings)
///
/// ## Row ID Support:
/// Uses RoaringTreemap which supports full i64 row IDs (up to u64::MAX).
/// This is implemented as a BTreeMap of RoaringBitmaps for efficient
/// storage while supporting the full 64-bit range.
pub struct BitmapIndex {
    name: String,
    table_name: String,
    column_names: Vec<String>,
    column_ids: Vec<i32>,
    data_types: Vec<DataType>,
    is_unique: bool,
    closed: AtomicBool,

    /// One bitmap per distinct value, using AHash for user-controlled keys.
    bitmaps: RwLock<BitmapStorage>,

    /// Reverse mapping: row_id -> CompactArc<Value> for efficient removal
    /// Uses I64Map for fast O(1) lookups and CompactArc<Value> (8 bytes per entry)
    row_to_value: RwLock<IndexValueMap>,

    /// Track cardinality for warnings
    distinct_count: AtomicUsize,
    memory: IndexMemoryOwner,
}

struct BitmapRows {
    rows: RoaringTreemap,
    containers: usize,
    submaps: usize,
    peak_containers: usize,
    present_row: u64,
    payload_bound: u64,
}

impl Default for BitmapRows {
    fn default() -> Self {
        Self {
            rows: RoaringTreemap::new(),
            containers: 0,
            submaps: 0,
            peak_containers: 0,
            present_row: u64::MAX,
            payload_bound: 0,
        }
    }
}

impl BitmapRows {
    fn occupancy(&self, row_id: u64) -> (bool, bool) {
        if self.present_row >> 16 == row_id >> 16 {
            return (true, true);
        }
        let mut rows = self.rows.iter();
        rows.advance_to(row_id & !0xffff);
        let next = rows.next();
        if next.is_some_and(|found| found >> 16 == row_id >> 16) {
            return (true, true);
        }
        if self.present_row >> 32 == row_id >> 32
            || next.is_some_and(|found| found >> 32 == row_id >> 32)
        {
            return (false, true);
        }
        let mut rows = self.rows.iter();
        rows.advance_to(row_id & !0xffff_ffff);
        (
            false,
            rows.next().is_some_and(|found| found >> 32 == row_id >> 32),
        )
    }

    fn insert(&mut self, row_id: u64) {
        debug_assert!(row_id <= i64::MAX as u64);
        let (container_exists, submap_exists) = self.occupancy(row_id);
        if self.rows.insert(row_id) {
            if container_exists {
                self.payload_bound += 4;
            } else {
                self.containers += 1;
                self.peak_containers = self.peak_containers.max(self.containers);
                self.submaps += usize::from(!submap_exists);
                self.payload_bound += 8;
            }
            self.payload_bound = self.payload_bound.min(8192 * self.containers as u64);
        }
        self.present_row = row_id;
    }

    fn remove(&mut self, row_id: u64) {
        if !self.rows.remove(row_id) {
            return;
        }
        if self.present_row == row_id {
            self.present_row = u64::MAX;
        }
        let (container_exists, submap_exists) = self.occupancy(row_id);
        if !container_exists {
            self.containers -= 1;
            self.submaps -= usize::from(!submap_exists);
            self.payload_bound -= 8;
            self.payload_bound = self.payload_bound.min(8192 * self.containers as u64);
        }
    }

    fn estimated_bytes(&self) -> u128 {
        if self.containers == 0 {
            return 0;
        }
        // Roaring point mutations retain container-directory capacity on deletion.
        let directory =
            64 * self.submaps as u128 * (2 * self.peak_containers.min(65536)).max(4) as u128;
        self.payload_bound as u128
            + directory
            + btree_node_bytes::<u32, RoaringBitmap>(self.submaps)
    }
}

impl std::ops::Deref for BitmapRows {
    type Target = RoaringTreemap;

    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

#[derive(Default)]
struct BitmapStorage {
    map: AHashMap<CompactArc<Value>, BitmapRows>,
    key_bytes: u128,
    bitmap_bytes: u128,
    capacity_high_water: usize,
}

impl BitmapStorage {
    fn estimated_bytes(&self) -> u128 {
        self.bitmap_bytes
            + hash_table_bytes::<CompactArc<Value>, BitmapRows>(self.capacity_high_water)
    }

    fn add(&mut self, value: &CompactArc<Value>, row_id: u64) -> bool {
        let mut new_key = false;
        let bitmap = self.map.entry(CompactArc::clone(value)).or_insert_with(|| {
            new_key = true;
            self.key_bytes += value_bytes(value);
            BitmapRows::default()
        });
        let before = bitmap.estimated_bytes();
        bitmap.insert(row_id);
        self.bitmap_bytes = self.bitmap_bytes - before + bitmap.estimated_bytes();
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
        new_key
    }

    fn remove(&mut self, value: &CompactArc<Value>, row_id: u64) -> bool {
        let Some(bitmap) = self.map.get_mut(value) else {
            return false;
        };
        let before = bitmap.estimated_bytes();
        bitmap.remove(row_id);
        self.bitmap_bytes = self.bitmap_bytes - before + bitmap.estimated_bytes();
        if bitmap.is_empty() {
            if let Some((stored, _)) = self.map.remove_entry(value) {
                self.key_bytes -= value_bytes(&stored);
            }
            true
        } else {
            false
        }
    }
}

impl std::ops::Deref for BitmapStorage {
    type Target = AHashMap<CompactArc<Value>, BitmapRows>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl std::fmt::Debug for BitmapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapIndex")
            .field("name", &self.name)
            .field("table_name", &self.table_name)
            .field("column_names", &self.column_names)
            .field("column_ids", &self.column_ids)
            .field("is_unique", &self.is_unique)
            .field(
                "distinct_count",
                &self.distinct_count.load(AtomicOrdering::Relaxed),
            )
            .field("closed", &self.closed.load(AtomicOrdering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl BitmapIndex {
    /// Create a new BitmapIndex
    ///
    /// # Arguments
    /// * `expected_rows` - Hint for initial capacity of row_to_value map.
    ///   Pass 0 if unknown; the map will grow automatically.
    pub fn new(
        name: String,
        table_name: String,
        column_names: Vec<String>,
        column_ids: Vec<i32>,
        data_types: Vec<DataType>,
        is_unique: bool,
        expected_rows: usize,
    ) -> Self {
        let map = if expected_rows > 0 {
            I64Map::with_capacity(expected_rows)
        } else {
            I64Map::new()
        };
        let requested = name.capacity() as u128
            + table_name.capacity() as u128
            + (column_names.capacity() * std::mem::size_of::<String>()) as u128
            + column_names
                .iter()
                .map(|name| name.capacity() as u128)
                .sum::<u128>()
            + (column_ids.capacity() * std::mem::size_of::<i32>()) as u128
            + (data_types.capacity() * std::mem::size_of::<DataType>()) as u128
            + map.allocation_bytes() as u128;
        Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            closed: AtomicBool::new(false),
            bitmaps: RwLock::new(BitmapStorage::default()),
            row_to_value: RwLock::new(IndexValueMap {
                map,
                payload_bytes: 0,
            }),
            distinct_count: AtomicUsize::new(0),
            memory: IndexMemoryOwner::new::<Self>(requested, 0),
        }
    }

    fn mutate(
        &self,
        mutation: impl FnOnce(&mut BitmapStorage, &mut IndexValueMap) -> Result<()>,
    ) -> Result<()> {
        let mut bitmaps = self.bitmaps.write();
        let mut reverse = self.row_to_value.write();
        let before = bitmaps.key_bytes + reverse.requested_bytes();
        let estimated_before = bitmaps.estimated_bytes();
        let result = mutation(&mut bitmaps, &mut reverse);
        self.memory
            .account
            .resize(before, bitmaps.key_bytes + reverse.requested_bytes());
        self.memory
            .account
            .resize_estimate(estimated_before, bitmaps.estimated_bytes());
        result
    }

    /// Get the current cardinality (number of distinct values)
    pub fn cardinality(&self) -> usize {
        self.distinct_count.load(AtomicOrdering::Relaxed)
    }

    /// Check if cardinality is too high for efficient bitmap operations
    pub fn is_high_cardinality(&self) -> bool {
        self.cardinality() > HIGH_CARDINALITY_WARNING_THRESHOLD
    }

    /// Get the bitmap for a specific value (for AND/OR operations)
    pub fn get_bitmap(&self, value: &Value) -> Option<RoaringTreemap> {
        // Intern value to get Arc for lookup
        let arc_key = self.value_to_arc_key(std::slice::from_ref(value));
        let bitmaps = self.bitmaps.read();
        bitmaps.get(&arc_key).map(|bitmap| bitmap.rows.clone())
    }

    /// Perform AND operation on multiple values (for multi-predicate queries)
    /// Returns row IDs that match ALL values
    pub fn and_values(&self, values: &[Value]) -> RoaringTreemap {
        let bitmaps = self.bitmaps.read();
        let mut result: Option<RoaringTreemap> = None;

        for value in values {
            let arc_key = self.value_to_arc_key(std::slice::from_ref(value));
            if let Some(bitmap) = bitmaps.get(&arc_key) {
                result = Some(match result {
                    Some(r) => r & &bitmap.rows,
                    None => bitmap.rows.clone(),
                });
            } else {
                // Value not found - result is empty
                return RoaringTreemap::new();
            }
        }

        result.unwrap_or_default()
    }

    /// Perform OR operation on multiple values
    /// Returns row IDs that match ANY value
    pub fn or_values(&self, values: &[Value]) -> RoaringTreemap {
        let bitmaps = self.bitmaps.read();
        let mut result = RoaringTreemap::new();

        for value in values {
            let arc_key = self.value_to_arc_key(std::slice::from_ref(value));
            if let Some(bitmap) = bitmaps.get(&arc_key) {
                result |= &bitmap.rows;
            }
        }

        result
    }

    /// Perform NOT operation on a value
    /// Returns row IDs that do NOT match the value
    /// Note: Requires knowing all row IDs in the table
    pub fn not_value(&self, value: &Value) -> RoaringTreemap {
        let arc_key = self.value_to_arc_key(std::slice::from_ref(value));
        let bitmaps = self.bitmaps.read();

        // Get all row IDs (union of all bitmaps)
        let mut all_rows = RoaringTreemap::new();
        for bitmap in bitmaps.values() {
            all_rows |= &bitmap.rows;
        }

        // Subtract the matching bitmap
        if let Some(bitmap) = bitmaps.get(&arc_key) {
            all_rows - &bitmap.rows
        } else {
            all_rows
        }
    }

    /// Convert values to an CompactArc<Value> key
    /// For single-column indexes, wraps the value in Arc
    /// For multi-column indexes, creates a composite key
    fn value_to_arc_key(&self, values: &[Value]) -> CompactArc<Value> {
        if values.len() == 1 {
            CompactArc::new(values[0].clone())
        } else {
            // For multi-column bitmap index, create a composite key
            // This is less common but supported
            let composite = Value::Text(
                values
                    .iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join("||")
                    .into(),
            );
            CompactArc::new(composite)
        }
    }

    /// Slow path for multi-column indexes (rare case)
    /// Creates composite key and handles add operation
    #[cold]
    fn add_multi_column_slow(
        &self,
        values: &[Value],
        row_id: i64,
        row_id_u64: u64,
        bitmaps: &mut BitmapStorage,
        row_to_value: &mut IndexValueMap,
    ) -> Result<()> {
        // Create composite key for multi-column lookup
        let arc_key = self.value_to_arc_key(values);

        // Check uniqueness constraint
        if self.is_unique {
            let has_null = values.iter().any(|v| v.is_null());
            if !has_null {
                if let Some(bitmap) = bitmaps.get(&arc_key) {
                    let existing_count = if bitmap.contains(row_id_u64) {
                        bitmap.len() - 1
                    } else {
                        bitmap.len()
                    };
                    if existing_count > 0 {
                        let values_str: Vec<String> =
                            values.iter().map(|v| format!("{:?}", v)).collect();
                        return Err(Error::unique_constraint(
                            &self.name,
                            self.column_names.join(", "),
                            format!("[{}]", values_str.join(", ")),
                        ));
                    }
                }
            }
        }

        // Check if row already exists with a different value
        if let Some(old_arc_key) = row_to_value.get(row_id).cloned() {
            if !CompactArc::ptr_eq(&old_arc_key, &arc_key)
                && bitmaps.remove(&old_arc_key, row_id_u64)
            {
                self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
            }
        }

        // Add to bitmap
        let is_new_value = bitmaps.add(&arc_key, row_id_u64);

        // Update reverse mapping
        row_to_value.insert(row_id, arc_key);

        if is_new_value {
            self.distinct_count.fetch_add(1, AtomicOrdering::Relaxed);
        }

        Ok(())
    }
}

impl Index for BitmapIndex {
    fn memory_account(&self) -> Option<&Arc<IndexMemory>> {
        Some(&self.memory.account)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn table_name(&self) -> &str {
        &self.table_name
    }

    fn build(&mut self) -> Result<()> {
        Ok(())
    }

    fn add(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        // Validate row_id is non-negative (can be safely converted to u64)
        if row_id < 0 {
            return Err(Error::internal(format!(
                "bitmap index: row_id must be non-negative, got {}",
                row_id
            )));
        }
        let row_id_u64 = row_id as u64;

        let num_cols = self.column_ids.len();
        if values.len() != num_cols {
            return Err(Error::internal(format!(
                "expected {} values, got {}",
                num_cols,
                values.len()
            )));
        }

        self.mutate(|bitmaps, row_to_value| {
            // Build lookup key (for single-column, just the first value)
            let lookup_value = if values.len() == 1 {
                &values[0]
            } else {
                // For multi-column, we need to create a composite key for lookup
                // This is less common but supported
                return self.add_multi_column_slow(
                    values,
                    row_id,
                    row_id_u64,
                    bitmaps,
                    row_to_value,
                );
            };

            // Check uniqueness constraint (using Borrow trait for &Value lookup)
            if self.is_unique && !lookup_value.is_null() {
                if let Some(bitmap) = bitmaps.get(lookup_value) {
                    // Check if there's already a row with this value (excluding current row)
                    let existing_count = if bitmap.contains(row_id_u64) {
                        bitmap.len() - 1
                    } else {
                        bitmap.len()
                    };
                    if existing_count > 0 {
                        return Err(Error::unique_constraint(
                            &self.name,
                            self.column_names.join(", "),
                            format!("{:?}", lookup_value),
                        ));
                    }
                }
            }

            // Try to reuse existing Arc if value already exists (O(1) clone)
            // Only create new Arc if this is a new unique value
            let arc_key = if let Some((existing_arc, _)) = bitmaps.get_key_value(lookup_value) {
                // Value exists - reuse the existing Arc (O(1) atomic refcount bump)
                CompactArc::clone(existing_arc)
            } else {
                // New unique value - create Arc once
                CompactArc::new(lookup_value.clone())
            };

            // Check if row already exists with a different value (for updates)
            if let Some(old_arc_key) = row_to_value.get(row_id).cloned() {
                if !CompactArc::ptr_eq(&old_arc_key, &arc_key)
                    && bitmaps.remove(&old_arc_key, row_id_u64)
                {
                    self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
                }
            }

            // Add to bitmap
            let is_new_value = bitmaps.add(&arc_key, row_id_u64);

            // Update reverse mapping with Arc reference
            row_to_value.insert(row_id, arc_key);

            // Update cardinality if this is a new distinct value
            if is_new_value {
                self.distinct_count.fetch_add(1, AtomicOrdering::Relaxed);
            }

            Ok(())
        })
    }

    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        for (row_id, values) in entries.iter() {
            self.add(values, row_id, 0)?;
        }
        Ok(())
    }

    fn remove(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        // Validate row_id is non-negative
        if row_id < 0 {
            return Err(Error::internal(format!(
                "bitmap index: row_id must be non-negative, got {}",
                row_id
            )));
        }
        let row_id_u64 = row_id as u64;

        // Intern value to get Arc key for lookup
        let arc_key = self.value_to_arc_key(values);

        self.mutate(|bitmaps, row_to_value| {
            if bitmaps.remove(&arc_key, row_id_u64) {
                self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
            }

            row_to_value.remove(row_id);

            Ok(())
        })
    }

    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        for (row_id, values) in entries.iter() {
            self.remove(values, row_id, 0)?;
        }
        Ok(())
    }

    /// Optimized batch add with single lock acquisition
    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        let num_cols = self.column_ids.len();

        self.mutate(|bitmaps, row_to_value| {
            // Reserve capacity
            row_to_value.map.reserve(entries.len());

            // PRE-CHECK PHASE: Validate all unique constraints BEFORE modifying anything
            // This prevents partial batch execution on failure
            if self.is_unique {
                // Track keys seen in this batch for intra-batch duplicate detection
                let mut batch_keys: AHashMap<CompactArc<Value>, i64> =
                    AHashMap::with_capacity(entries.len());

                for &(row_id, values) in entries {
                    if row_id < 0 || values.len() != num_cols {
                        continue;
                    }

                    // NULL values don't violate uniqueness
                    if values.iter().any(|v| v.is_null()) {
                        continue;
                    }

                    let row_id_u64 = row_id as u64;
                    let arc_key = self.value_to_arc_key(values);

                    // Check intra-batch duplicates
                    if let Some(&existing_row_id) = batch_keys.get(&arc_key) {
                        if existing_row_id != row_id {
                            return Err(Error::unique_constraint(
                                &self.name,
                                self.column_names.join(", "),
                                format!("{:?}", values),
                            ));
                        }
                    }

                    // Check against existing index
                    if let Some(bitmap) = bitmaps.get(&arc_key) {
                        let existing_count = if bitmap.contains(row_id_u64) {
                            bitmap.len() - 1
                        } else {
                            bitmap.len()
                        };
                        if existing_count > 0 {
                            return Err(Error::unique_constraint(
                                &self.name,
                                self.column_names.join(", "),
                                format!("{:?}", values),
                            ));
                        }
                    }

                    batch_keys.insert(arc_key, row_id);
                }
            }

            // MODIFICATION PHASE: All constraints checked, now safe to modify
            for &(row_id, values) in entries {
                if row_id < 0 {
                    continue; // Skip invalid row IDs
                }
                let row_id_u64 = row_id as u64;

                if values.len() != num_cols {
                    continue;
                }

                let arc_key = self.value_to_arc_key(values);

                // Try to reuse existing Arc
                let final_arc_key = if let Some((existing_arc, _)) = bitmaps.get_key_value(&arc_key)
                {
                    CompactArc::clone(existing_arc)
                } else {
                    arc_key
                };

                // Handle update case - remove from old bitmap
                if let Some(old_arc_key) = row_to_value.get(row_id).cloned() {
                    if !CompactArc::ptr_eq(&old_arc_key, &final_arc_key)
                        && bitmaps.remove(&old_arc_key, row_id_u64)
                    {
                        self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }

                // Add to bitmap
                let is_new_value = bitmaps.add(&final_arc_key, row_id_u64);

                // Update reverse mapping
                row_to_value.insert(row_id, final_arc_key);

                if is_new_value {
                    self.distinct_count.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }

            Ok(())
        })
    }

    /// Optimized batch remove with single lock acquisition
    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        self.mutate(|bitmaps, row_to_value| {
            for &(row_id, values) in entries {
                if row_id < 0 {
                    continue;
                }
                let row_id_u64 = row_id as u64;

                let arc_key = self.value_to_arc_key(values);

                if bitmaps.remove(&arc_key, row_id_u64) {
                    self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
                }

                // Remove from reverse mapping
                row_to_value.remove(row_id);
            }

            Ok(())
        })
    }

    fn remove_batch_ids(&self, row_ids: &[i64]) -> Option<Result<()>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Some(Err(Error::IndexClosed));
        }
        Some(self.mutate(|bitmaps, row_to_value| {
            for &row_id in row_ids {
                if row_id < 0 {
                    continue;
                }
                if let Some(arc_key) = row_to_value.remove(row_id) {
                    if bitmaps.remove(&arc_key, row_id as u64) {
                        self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
            }
            Ok(())
        }))
    }

    fn column_ids(&self) -> &[i32] {
        &self.column_ids
    }

    fn column_names(&self) -> &[String] {
        &self.column_names
    }

    fn data_types(&self) -> &[DataType] {
        &self.data_types
    }

    fn index_type(&self) -> IndexType {
        IndexType::Bitmap
    }

    fn is_unique(&self) -> bool {
        self.is_unique
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        if values.len() != self.column_ids.len() {
            return Err(Error::internal(
                "bitmap index requires exact match on all columns",
            ));
        }

        // Intern value to get Arc key for lookup
        let arc_key = self.value_to_arc_key(values);
        let bitmaps = self.bitmaps.read();

        if let Some(bitmap) = bitmaps.get(&arc_key) {
            Ok(bitmap
                .iter()
                .map(|row_id| IndexEntry {
                    row_id: row_id as i64,
                    ref_id: 0,
                })
                .collect())
        } else {
            Ok(vec![])
        }
    }

    fn find_range(
        &self,
        _min: &[Value],
        _max: &[Value],
        _min_inclusive: bool,
        _max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        // Bitmap index doesn't efficiently support range queries
        // For ordered values, we could iterate through bitmaps, but it's not optimal
        Err(Error::internal(
            "bitmap index does not efficiently support range queries; use btree index instead",
        ))
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        match op {
            Operator::Eq => self.find(values),
            Operator::Ne => {
                // Use NOT operation
                if values.len() != self.column_ids.len() {
                    return Err(Error::internal(
                        "bitmap index requires exact match on all columns",
                    ));
                }
                // not_value() takes &Value, not the key directly
                // For single column, pass the value; for multi, create composite
                let result = if values.len() == 1 {
                    self.not_value(&values[0])
                } else {
                    let composite = Value::Text(
                        values
                            .iter()
                            .map(|v| format!("{:?}", v))
                            .collect::<Vec<_>>()
                            .join("||")
                            .into(),
                    );
                    self.not_value(&composite)
                };
                Ok(result
                    .iter()
                    .map(|row_id| IndexEntry {
                        row_id: row_id as i64,
                        ref_id: 0,
                    })
                    .collect())
            }
            _ => Err(Error::internal(format!(
                "bitmap index only supports = and != operators, not {:?}",
                op
            ))),
        }
    }

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        if values.len() != self.column_ids.len() {
            return;
        }

        // Intern value to get Arc key for lookup
        let arc_key = self.value_to_arc_key(values);
        let bitmaps = self.bitmaps.read();

        if let Some(bitmap) = bitmaps.get(&arc_key) {
            // RoaringTreemap iteration is efficient
            buffer.extend(bitmap.iter().map(|row_id| row_id as i64));
        }
    }

    fn get_row_ids_in_range_into(
        &self,
        _min_value: &[Value],
        _max_value: &[Value],
        _include_min: bool,
        _include_max: bool,
        _buffer: &mut Vec<i64>,
    ) {
        // Bitmap index doesn't support range queries efficiently - do nothing
    }

    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec {
        // For complex expressions, return all row IDs and let caller filter
        let bitmaps = self.bitmaps.read();
        let mut all_rows = RoaringTreemap::new();
        for bitmap in bitmaps.values() {
            all_rows |= &bitmap.rows;
        }
        let _ = expr;
        let collected: Vec<i64> = all_rows.iter().map(|id| id as i64).collect();
        RowIdVec::from_vec(collected)
    }

    fn get_all_values(&self) -> Vec<Value> {
        let bitmaps = self.bitmaps.read();
        let mut exports = crate::storage::mvcc::read_memory::ExportBatch::new();
        // Dereference CompactArc<Value> to clone inner Value
        bitmaps
            .keys()
            .map(|arc| exports.capture_value(arc))
            .collect()
    }

    fn clear(&self) -> Result<()> {
        self.mutate(|bitmaps, reverse| {
            bitmaps.map.clear();
            bitmaps.key_bytes = 0;
            bitmaps.bitmap_bytes = 0;
            reverse.map.clear();
            reverse.payload_bytes = 0;
            self.distinct_count.store(0, AtomicOrdering::Relaxed);
            Ok(())
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);
        self.clear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bitmap_bound(bitmap: &BitmapRows) {
        let mut containers = 0;
        let mut submaps = 0;
        let mut payload = 0;
        for (_, rows) in bitmap.rows.bitmaps() {
            let stats = rows.statistics();
            containers += stats.n_containers as usize;
            submaps += 1;
            assert_eq!(stats.n_run_containers, 0);
            // Roaring 0.11.3 reports array capacity in u32 units and bitsets in bits.
            payload += stats.n_bytes_array_containers / 2 + stats.n_bytes_bitset_containers / 8;
        }
        assert_eq!(bitmap.containers, containers);
        assert_eq!(bitmap.submaps, submaps);
        assert!(bitmap.peak_containers >= containers);
        assert!(
            bitmap.payload_bound >= payload,
            "payload {payload} exceeds {}",
            bitmap.payload_bound
        );
        assert!(bitmap.payload_bound <= 8192 * containers as u64);
        assert!(bitmap.present_row == u64::MAX || bitmap.rows.contains(bitmap.present_row));
    }

    #[test]
    fn bitmap_accounting_tracks_sparse_containers_and_submap_lifetimes() {
        let mut bitmap = BitmapRows::default();
        let ids = [
            0,
            1 << 16,
            5 << 16,
            1 << 32,
            (1 << 32) + (7 << 16),
            i64::MAX as u64,
        ];
        for id in ids {
            bitmap.insert(id);
            assert_bitmap_bound(&bitmap);
            let before = bitmap.payload_bound;
            bitmap.insert(id);
            assert_eq!(bitmap.payload_bound, before);
        }
        assert_eq!(bitmap.payload_bound, 8 * ids.len() as u64);
        for id in [ids[4], ids[2], ids[1], ids[3], ids[5], ids[0]] {
            bitmap.remove(id);
            assert_bitmap_bound(&bitmap);
        }
        assert_eq!(bitmap.estimated_bytes(), 0);

        for id in 0..1024 {
            bitmap.insert(id << 16);
        }
        assert_bitmap_bound(&bitmap);
        assert_eq!(bitmap.payload_bound, 8192);
        let peak = bitmap.peak_containers;
        for id in 1..1024 {
            bitmap.remove(id << 16);
        }
        assert_bitmap_bound(&bitmap);
        assert_eq!(bitmap.peak_containers, peak);
        assert!(bitmap.estimated_bytes() >= 64 * 2 * peak as u128);
    }

    #[test]
    fn bitmap_accounting_bounds_dense_conversion_and_retained_array_churn() {
        let mut bitmap = BitmapRows::default();
        for id in 0..5000 {
            bitmap.insert(id);
            if id % 64 == 0 || (4095..=4097).contains(&id) {
                assert_bitmap_bound(&bitmap);
            }
        }
        assert_bitmap_bound(&bitmap);
        for id in 1..5000 {
            bitmap.remove(id);
            if id % 64 == 0 || (902..=905).contains(&id) {
                assert_bitmap_bound(&bitmap);
            }
        }
        assert_bitmap_bound(&bitmap);
        assert_eq!(bitmap.payload_bound, 8192);
        bitmap.remove(0);
        assert_bitmap_bound(&bitmap);
        assert_eq!(bitmap.payload_bound, 0);

        bitmap.insert(0);
        for _ in 0..3000 {
            bitmap.insert(1);
            bitmap.remove(1);
        }
        assert_bitmap_bound(&bitmap);
        assert_eq!(bitmap.payload_bound, 8192);
    }

    fn memory_index(unique: bool, columns: usize) -> BitmapIndex {
        BitmapIndex::new(
            "idx_label".into(),
            "items".into(),
            (0..columns).map(|id| format!("column_{id}")).collect(),
            (0..columns as i32).collect(),
            vec![DataType::Text; columns],
            unique,
            0,
        )
    }

    fn assert_requested_allocations(index: &BitmapIndex) {
        let bitmaps = index.bitmaps.read();
        let reverse = index.row_to_value.read();
        let payload = |value: &CompactArc<Value>| {
            (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Value>()) as u128
                + value.heap_bytes() as u128
        };
        let keys: u128 = bitmaps.keys().map(payload).sum();
        let reverse_keys: u128 = reverse.values().map(payload).sum();
        assert_eq!(bitmaps.key_bytes, keys);
        assert_eq!(reverse.payload_bytes, reverse_keys);
        let expected = crate::storage::mvcc::memory::arc_allocation_bytes::<BitmapIndex>() as u128
            + keys
            + reverse_keys
            + reverse.map.allocation_bytes() as u128
            + index.name.capacity() as u128
            + index.table_name.capacity() as u128
            + (index.column_names.capacity() * std::mem::size_of::<String>()) as u128
            + index
                .column_names
                .iter()
                .map(|name| name.capacity() as u128)
                .sum::<u128>()
            + (index.column_ids.capacity() * std::mem::size_of::<i32>()) as u128
            + (index.data_types.capacity() * std::mem::size_of::<DataType>()) as u128;
        assert_eq!(index.memory.account.requested_bytes() as u128, expected);
        for bitmap in bitmaps.values() {
            assert_bitmap_bound(bitmap);
        }
        assert_eq!(
            index.memory.account.estimated_bytes() as u128,
            bitmaps.estimated_bytes()
        );
    }

    #[test]
    fn bitmap_accounting_covers_failed_reserve_and_scalar_replacement() {
        let index = memory_index(true, 1);
        assert_requested_allocations(&index);
        let initial = index.memory.account.requested_bytes();
        let key = [Value::text("the same long key in a rejected batch")];
        let entries: Vec<_> = (0..1024).map(|id| (id, &key[..])).collect();
        assert!(index.add_batch_slice(&entries).is_err());
        assert!(index.memory.account.requested_bytes() > initial);
        assert_requested_allocations(&index);
        assert_eq!(index.memory.account.estimated_bytes(), 0);

        for columns in [1, 2] {
            let index = memory_index(false, columns);
            let key =
                vec![Value::text("a retained key shared by bitmap and reverse owners"); columns];
            let next = vec![Value::text("a replacement key"); columns];
            index.add(&key, 1, 0).unwrap();
            index.add(&key, 2, 0).unwrap();
            assert_requested_allocations(&index);
            index.add(&next, 1, 0).unwrap();
            assert_requested_allocations(&index);
            index.remove(&key, 2, 0).unwrap();
            assert_requested_allocations(&index);
            index.remove_batch_ids(&[1]).unwrap().unwrap();
            assert_requested_allocations(&index);
            assert_eq!(index.cardinality(), 0);
            assert!(index.memory.account.estimated_bytes() > 0);
            let entries = [(3, key.as_slice()), (4, key.as_slice())];
            index.add_batch_slice(&entries).unwrap();
            assert_requested_allocations(&index);
            index.remove_batch_slice(&entries[..1]).unwrap();
            assert_requested_allocations(&index);
            index.remove_batch_slice(&entries[1..]).unwrap();
            assert_requested_allocations(&index);
            index.clear().unwrap();
            assert_requested_allocations(&index);
            let account = Arc::clone(index.memory_account().unwrap());
            drop(index);
            assert_eq!(account.requested_bytes(), 0);
            assert_eq!(account.estimated_bytes(), 0);
        }
    }

    #[test]
    fn test_bitmap_index_basic() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Add entries
        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 2, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 3, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 4, 0).unwrap();
        index.add(&[Value::Text("delivered".into())], 5, 0).unwrap();

        // Check cardinality
        assert_eq!(index.cardinality(), 3);

        // Find by status
        let results = index.find(&[Value::Text("pending".into())]).unwrap();
        assert_eq!(results.len(), 2);

        let results = index.find(&[Value::Text("shipped".into())]).unwrap();
        assert_eq!(results.len(), 2);

        let results = index.find(&[Value::Text("delivered".into())]).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bitmap_index_boolean() {
        let index = BitmapIndex::new(
            "idx_active".to_string(),
            "users".to_string(),
            vec!["active".to_string()],
            vec![1],
            vec![DataType::Boolean],
            false,
            0,
        );

        // Add boolean values
        index.add(&[Value::Boolean(true)], 1, 0).unwrap();
        index.add(&[Value::Boolean(true)], 2, 0).unwrap();
        index.add(&[Value::Boolean(true)], 3, 0).unwrap();
        index.add(&[Value::Boolean(false)], 4, 0).unwrap();
        index.add(&[Value::Boolean(false)], 5, 0).unwrap();

        // Only 2 distinct values
        assert_eq!(index.cardinality(), 2);

        let active = index.find(&[Value::Boolean(true)]).unwrap();
        assert_eq!(active.len(), 3);

        let inactive = index.find(&[Value::Boolean(false)]).unwrap();
        assert_eq!(inactive.len(), 2);
    }

    #[test]
    fn test_bitmap_index_and_operation() {
        // Note: AND operation is for combining multiple bitmap indexes
        // Here we test the single-index AND which isn't as useful
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 2, 0).unwrap();

        // AND of different values in same column = empty (row can't have two values)
        let result =
            index.and_values(&[Value::Text("pending".into()), Value::Text("shipped".into())]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bitmap_index_or_operation() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 2, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 3, 0).unwrap();
        index.add(&[Value::Text("delivered".into())], 4, 0).unwrap();

        // OR of pending and shipped
        let result =
            index.or_values(&[Value::Text("pending".into()), Value::Text("shipped".into())]);
        assert_eq!(result.len(), 3); // rows 1, 2, 3
    }

    #[test]
    fn test_bitmap_index_not_operation() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 2, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 3, 0).unwrap();
        index.add(&[Value::Text("delivered".into())], 4, 0).unwrap();

        // NOT pending = shipped + delivered
        let result = index.not_value(&Value::Text("pending".into()));
        assert_eq!(result.len(), 2); // rows 3, 4
    }

    #[test]
    fn test_bitmap_index_remove() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 2, 0).unwrap();

        assert_eq!(index.cardinality(), 1);

        // Remove one
        index
            .remove(&[Value::Text("pending".into())], 1, 0)
            .unwrap();

        let results = index.find(&[Value::Text("pending".into())]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 2);

        // Remove last one - cardinality should decrease
        index
            .remove(&[Value::Text("pending".into())], 2, 0)
            .unwrap();
        assert_eq!(index.cardinality(), 0);
    }

    #[test]
    fn test_bitmap_index_update() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Initial value
        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        assert_eq!(index.cardinality(), 1);

        // Update to new value
        index.add(&[Value::Text("shipped".into())], 1, 0).unwrap();
        assert_eq!(index.cardinality(), 1); // Old empty bitmap removed

        // Old value should be gone
        let pending = index.find(&[Value::Text("pending".into())]).unwrap();
        assert!(pending.is_empty());

        // New value should be found
        let shipped = index.find(&[Value::Text("shipped".into())]).unwrap();
        assert_eq!(shipped.len(), 1);
        assert_eq!(shipped[0].row_id, 1);
    }

    #[test]
    fn test_bitmap_index_not_equal() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 2, 0).unwrap();
        index.add(&[Value::Text("delivered".into())], 3, 0).unwrap();

        // != pending should return shipped and delivered
        let results = index
            .find_with_operator(Operator::Ne, &[Value::Text("pending".into())])
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bitmap_index_high_cardinality_check() {
        let index = BitmapIndex::new(
            "idx_id".to_string(),
            "items".to_string(),
            vec!["id".to_string()],
            vec![1],
            vec![DataType::Integer],
            false,
            0,
        );

        // Add many distinct values
        for i in 0..HIGH_CARDINALITY_WARNING_THRESHOLD + 100 {
            index.add(&[Value::Integer(i as i64)], i as i64, 0).unwrap();
        }

        assert!(index.is_high_cardinality());
    }

    #[test]
    fn test_bitmap_index_null_handling() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index.add(&[Value::Null(DataType::Text)], 1, 0).unwrap();
        index.add(&[Value::Null(DataType::Text)], 2, 0).unwrap();
        index.add(&[Value::Text("active".into())], 3, 0).unwrap();

        let nulls = index.find(&[Value::Null(DataType::Text)]).unwrap();
        assert_eq!(nulls.len(), 2);
    }

    #[test]
    fn test_bitmap_index_unique_constraint() {
        let index = BitmapIndex::new(
            "idx_status_unique".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            true, // unique
            0,
        );

        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();

        // Try to add duplicate - should fail
        let result = index.add(&[Value::Text("pending".into())], 2, 0);
        assert!(result.is_err());

        // NULL doesn't violate uniqueness
        index.add(&[Value::Null(DataType::Text)], 3, 0).unwrap();
        index.add(&[Value::Null(DataType::Text)], 4, 0).unwrap();
    }

    #[test]
    fn test_bitmap_index_large_row_ids() {
        // Test that bitmap index correctly handles row IDs > u32::MAX
        // This uses RoaringTreemap (64-bit) instead of RoaringBitmap (32-bit)
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Use row IDs beyond u32::MAX
        let large_row_id_1: i64 = (u32::MAX as i64) + 1; // 4,294,967,296
        let large_row_id_2: i64 = (u32::MAX as i64) + 1000; // 4,294,968,295
        let large_row_id_3: i64 = i64::MAX / 2; // Very large value

        index
            .add(&[Value::Text("active".into())], large_row_id_1, 0)
            .unwrap();
        index
            .add(&[Value::Text("active".into())], large_row_id_2, 0)
            .unwrap();
        index
            .add(&[Value::Text("inactive".into())], large_row_id_3, 0)
            .unwrap();

        // Verify we can find them
        let results = index.find(&[Value::Text("active".into())]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|e| e.row_id == large_row_id_1));
        assert!(results.iter().any(|e| e.row_id == large_row_id_2));

        let results = index.find(&[Value::Text("inactive".into())]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, large_row_id_3);

        // Verify removal works
        index
            .remove(&[Value::Text("active".into())], large_row_id_1, 0)
            .unwrap();
        let results = index.find(&[Value::Text("active".into())]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, large_row_id_2);
    }

    #[test]
    fn test_bitmap_index_rejects_negative_row_ids() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Negative row IDs should be rejected
        let result = index.add(&[Value::Text("pending".into())], -1, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-negative"));

        let result = index.add(&[Value::Text("pending".into())], i64::MIN, 0);
        assert!(result.is_err());

        // Removal of negative row ID should also be rejected
        let result = index.remove(&[Value::Text("pending".into())], -1, 0);
        assert!(result.is_err());
    }
}
