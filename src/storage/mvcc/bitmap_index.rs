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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::RwLock;

use ahash::AHashMap;
use roaring::RoaringTreemap;

use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, Value};
use crate::storage::expression::Expression;
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

    /// One bitmap per distinct value
    /// Maps Value -> RoaringTreemap of row IDs (supports full u64 range)
    bitmaps: RwLock<AHashMap<Value, RoaringTreemap>>,

    /// Reverse mapping: row_id -> value for efficient removal
    row_to_value: RwLock<AHashMap<i64, Value>>,

    /// Track cardinality for warnings
    distinct_count: AtomicUsize,
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
    pub fn new(
        name: String,
        table_name: String,
        column_names: Vec<String>,
        column_ids: Vec<i32>,
        data_types: Vec<DataType>,
        is_unique: bool,
    ) -> Self {
        Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            closed: AtomicBool::new(false),
            bitmaps: RwLock::new(AHashMap::new()),
            row_to_value: RwLock::new(AHashMap::new()),
            distinct_count: AtomicUsize::new(0),
        }
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
        let bitmaps = self.bitmaps.read().unwrap();
        bitmaps.get(value).cloned()
    }

    /// Perform AND operation on multiple values (for multi-predicate queries)
    /// Returns row IDs that match ALL values
    pub fn and_values(&self, values: &[Value]) -> RoaringTreemap {
        let bitmaps = self.bitmaps.read().unwrap();
        let mut result: Option<RoaringTreemap> = None;

        for value in values {
            if let Some(bitmap) = bitmaps.get(value) {
                result = Some(match result {
                    Some(r) => r & bitmap,
                    None => bitmap.clone(),
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
        let bitmaps = self.bitmaps.read().unwrap();
        let mut result = RoaringTreemap::new();

        for value in values {
            if let Some(bitmap) = bitmaps.get(value) {
                result |= bitmap;
            }
        }

        result
    }

    /// Perform NOT operation on a value
    /// Returns row IDs that do NOT match the value
    /// Note: Requires knowing all row IDs in the table
    pub fn not_value(&self, value: &Value) -> RoaringTreemap {
        let bitmaps = self.bitmaps.read().unwrap();

        // Get all row IDs (union of all bitmaps)
        let mut all_rows = RoaringTreemap::new();
        for bitmap in bitmaps.values() {
            all_rows |= bitmap;
        }

        // Subtract the matching bitmap
        if let Some(bitmap) = bitmaps.get(value) {
            all_rows - bitmap
        } else {
            all_rows
        }
    }

    /// Convert a single value to the key used for lookup
    /// For single-column indexes, we use the value directly
    /// For multi-column indexes, we would need to combine values
    fn value_to_key(&self, values: &[Value]) -> Value {
        if values.len() == 1 {
            values[0].clone()
        } else {
            // For multi-column bitmap index, create a composite key
            // This is less common but supported
            Value::Text(
                values
                    .iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join("||")
                    .into(),
            )
        }
    }
}

impl Index for BitmapIndex {
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

        let key = self.value_to_key(values);

        // Acquire write locks
        let mut bitmaps = self.bitmaps.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        // Check uniqueness constraint
        if self.is_unique {
            // NULL values don't violate uniqueness
            let has_null = values.iter().any(|v| v.is_null());
            if !has_null {
                if let Some(bitmap) = bitmaps.get(&key) {
                    // Check if there's already a row with this value (excluding current row)
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

        // Check if row already exists with a different value (for updates)
        if let Some(old_key) = row_to_value.get(&row_id).cloned() {
            if old_key != key {
                // Remove from old bitmap
                if let Some(old_bitmap) = bitmaps.get_mut(&old_key) {
                    old_bitmap.remove(row_id_u64);
                    if old_bitmap.is_empty() {
                        bitmaps.remove(&old_key);
                        self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }

        // Add to bitmap
        let is_new_value = !bitmaps.contains_key(&key);
        let bitmap = bitmaps.entry(key.clone()).or_default();
        bitmap.insert(row_id_u64);

        // Update reverse mapping
        row_to_value.insert(row_id, key);

        // Update cardinality if this is a new distinct value
        if is_new_value {
            self.distinct_count.fetch_add(1, AtomicOrdering::Relaxed);
        }

        Ok(())
    }

    fn add_batch(&self, entries: &HashMap<i64, Vec<Value>>) -> Result<()> {
        for (&row_id, values) in entries {
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

        let key = self.value_to_key(values);

        let mut bitmaps = self.bitmaps.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        // Remove from bitmap
        if let Some(bitmap) = bitmaps.get_mut(&key) {
            bitmap.remove(row_id_u64);
            if bitmap.is_empty() {
                bitmaps.remove(&key);
                self.distinct_count.fetch_sub(1, AtomicOrdering::Relaxed);
            }
        }

        // Remove from reverse mapping
        row_to_value.remove(&row_id);

        Ok(())
    }

    fn remove_batch(&self, entries: &HashMap<i64, Vec<Value>>) -> Result<()> {
        for (&row_id, values) in entries {
            self.remove(values, row_id, 0)?;
        }
        Ok(())
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

        let key = self.value_to_key(values);
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(&key) {
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
                let key = self.value_to_key(values);
                let result = self.not_value(&key);
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

    fn get_row_ids_equal(&self, values: &[Value]) -> Vec<i64> {
        match self.find(values) {
            Ok(entries) => entries.into_iter().map(|e| e.row_id).collect(),
            Err(_) => vec![],
        }
    }

    fn get_row_ids_in_range(
        &self,
        _min_value: &[Value],
        _max_value: &[Value],
        _include_min: bool,
        _include_max: bool,
    ) -> Vec<i64> {
        // Bitmap index doesn't support range queries efficiently
        vec![]
    }

    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> Vec<i64> {
        // For complex expressions, return all row IDs and let caller filter
        let bitmaps = self.bitmaps.read().unwrap();
        let mut all_rows = RoaringTreemap::new();
        for bitmap in bitmaps.values() {
            all_rows |= bitmap;
        }
        let _ = expr;
        all_rows.iter().map(|id| id as i64).collect()
    }

    fn get_all_values(&self) -> Vec<Value> {
        let bitmaps = self.bitmaps.read().unwrap();
        bitmaps.keys().cloned().collect()
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_index_basic() {
        let index = BitmapIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
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
