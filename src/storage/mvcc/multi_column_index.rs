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

//! Multi-Column Index implementation for composite key queries
//!
//! This module provides an optimized multi-column index that combines:
//! - Hash index for O(1) exact lookups
//! - Lazy BTree for range queries (built on first range query)
//! - Lazy prefix indexes for partial key queries (built on demand)
//!
//! ## Performance characteristics (100K rows benchmark):
//! - INSERT: ~134ms (lazy - only updates hash index)
//! - DELETE: ~243ms (lazy - only updates built indexes)
//! - FIND exact: ~1.1ms (O(1) hash lookup) - BEST
//! - FIND partial: ~47ms (includes one-time prefix index build)
//! - RANGE: ~30ms (includes one-time BTree build)
//!
//! ## Design
//!
//! The index uses lazy building strategy:
//! - `value_to_rows` hash is always maintained (for exact lookups)
//! - `row_to_key` reverse mapping is always maintained (for removal)
//! - `sorted_values` BTree is built on first range query
//! - `prefix_indexes` are built on first partial query per prefix length

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::RwLock;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::Index;

// ============================================================================
// CompositeKey - Ordered key for BTreeMap
// ============================================================================

/// Composite key for BTreeMap ordering
/// Wraps Vec<Value> with proper Ord implementation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeKey(pub Vec<Value>);

impl PartialOrd for CompositeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompositeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare element by element
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            match a.cmp(b) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        // If all compared elements are equal, shorter is less
        self.0.len().cmp(&other.0.len())
    }
}

impl std::hash::Hash for CompositeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for v in &self.0 {
            v.hash(state);
        }
    }
}

// ============================================================================
// MultiColumnIndex - Optimized multi-column index with lazy building
// ============================================================================

/// Multi-Column Index with lazy BTree and prefix index building
///
/// Combines hash index (always maintained) with lazy BTree (for range queries)
/// and lazy prefix indexes (for partial key queries).
///
/// ## Key features:
/// - O(1) exact lookups via hash index
/// - Lazy BTree building on first range query
/// - Lazy prefix index building on first partial query
/// - Fast INSERT/DELETE (only updates hash index until queries trigger builds)
///
/// ## Implementation:
/// - `value_to_rows`: FxHashMap<CompositeKey, SmallVec> for exact lookups (always maintained)
/// - `row_to_key`: FxHashMap for reverse mapping (always maintained)
/// - `sorted_values`: BTreeMap for RANGE queries (lazy built)
/// - `prefix_indexes`: Vec of FxHashMaps for partial queries (lazy built)
pub struct MultiColumnIndex {
    name: String,
    table_name: String,
    column_names: Vec<String>,
    column_ids: Vec<i32>,
    data_types: Vec<DataType>,
    is_unique: bool,
    closed: AtomicBool,

    /// Main BTree index for range queries - LAZY built on first range query
    sorted_values: RwLock<BTreeMap<CompositeKey, SmallVec<[i64; 4]>>>,
    btree_built: AtomicBool,

    /// Hash index for exact lookups (full key) - always maintained
    value_to_rows: RwLock<FxHashMap<CompositeKey, SmallVec<[i64; 4]>>>,

    /// Prefix indexes: LAZY built on first partial query
    /// Index 0 = first column, Index 1 = first two columns, etc.
    prefix_indexes: Vec<RwLock<FxHashMap<CompositeKey, SmallVec<[i64; 4]>>>>,
    prefix_built: Vec<AtomicBool>,

    /// Reverse mapping for removal
    row_to_key: RwLock<FxHashMap<i64, CompositeKey>>,
}

impl std::fmt::Debug for MultiColumnIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiColumnIndex")
            .field("name", &self.name)
            .field("table_name", &self.table_name)
            .field("column_names", &self.column_names)
            .field("column_ids", &self.column_ids)
            .field("is_unique", &self.is_unique)
            .field(
                "btree_built",
                &self.btree_built.load(AtomicOrdering::Relaxed),
            )
            .field("closed", &self.closed.load(AtomicOrdering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl MultiColumnIndex {
    pub fn new(
        name: String,
        table_name: String,
        column_names: Vec<String>,
        column_ids: Vec<i32>,
        data_types: Vec<DataType>,
        is_unique: bool,
    ) -> Self {
        let num_cols = column_names.len();
        let mut prefix_indexes = Vec::with_capacity(num_cols);
        let mut prefix_built = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            prefix_indexes.push(RwLock::new(FxHashMap::default()));
            prefix_built.push(AtomicBool::new(false));
        }

        Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            closed: AtomicBool::new(false),
            sorted_values: RwLock::new(BTreeMap::new()),
            btree_built: AtomicBool::new(false),
            value_to_rows: RwLock::new(FxHashMap::default()),
            prefix_indexes,
            prefix_built,
            row_to_key: RwLock::new(FxHashMap::default()),
        }
    }

    /// Build BTree index lazily from hash index (on first range query)
    fn ensure_btree_built(&self) {
        if self.btree_built.load(AtomicOrdering::Acquire) {
            return;
        }

        // Acquire both locks to prevent race condition:
        // We need to ensure no inserts happen between reading hash and setting btree_built
        let value_to_rows = self.value_to_rows.read().unwrap();
        let mut sorted_values = self.sorted_values.write().unwrap();

        // Double-check after acquiring write lock
        if self.btree_built.load(AtomicOrdering::Acquire) {
            return;
        }

        // Build BTree from hash index (holding read lock prevents concurrent inserts)
        for (key, rows) in value_to_rows.iter() {
            sorted_values.insert(key.clone(), rows.clone());
        }

        // Set flag before releasing locks - subsequent inserts will see btree_built=true
        // and will add to the BTree themselves
        self.btree_built.store(true, AtomicOrdering::Release);
    }

    /// Build prefix index lazily from main hash index (on first partial query)
    fn ensure_prefix_built(&self, prefix_len: usize) {
        if prefix_len == 0 || prefix_len > self.prefix_built.len() {
            return;
        }

        let idx = prefix_len - 1;
        if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
            return;
        }

        // Acquire both locks to prevent race condition:
        // Holding row_to_key read lock blocks concurrent inserts (which need write lock)
        let row_to_key = self.row_to_key.read().unwrap();
        let mut prefix_index = self.prefix_indexes[idx].write().unwrap();

        // Double-check after acquiring write lock
        if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
            return;
        }

        // Build prefix index from row_to_key (read lock prevents concurrent inserts)
        for (&row_id, key) in row_to_key.iter() {
            if key.0.len() >= prefix_len {
                let prefix_key = CompositeKey(key.0[..prefix_len].to_vec());
                prefix_index.entry(prefix_key).or_default().push(row_id);
            }
        }

        // Set flag before releasing locks - subsequent inserts will see prefix_built=true
        self.prefix_built[idx].store(true, AtomicOrdering::Release);
    }

    /// Check uniqueness constraint (must be called while holding write lock on value_to_rows)
    fn check_unique_constraint_locked(
        &self,
        key: &CompositeKey,
        row_id: i64,
        value_to_rows: &FxHashMap<CompositeKey, SmallVec<[i64; 4]>>,
    ) -> Result<()> {
        if !self.is_unique {
            return Ok(());
        }
        // NULL values don't violate uniqueness
        for v in &key.0 {
            if v.is_null() {
                return Ok(());
            }
        }

        if let Some(rows) = value_to_rows.get(key) {
            if !rows.is_empty() && !rows.contains(&row_id) {
                // Format all values in the key for a clear error message
                let values_str: Vec<String> = key.0.iter().map(|v| format!("{:?}", v)).collect();
                return Err(Error::unique_constraint(
                    &self.name,
                    self.column_names.join(", "),
                    format!("[{}]", values_str.join(", ")),
                ));
            }
        }
        Ok(())
    }
}

impl Index for MultiColumnIndex {
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

        let num_cols = self.column_ids.len();
        if values.len() != num_cols {
            return Err(Error::internal(format!(
                "expected {} values, got {}",
                num_cols,
                values.len()
            )));
        }

        let key = CompositeKey(values.to_vec());

        // Acquire BOTH write locks FIRST to prevent race conditions during updates
        // Lock order: value_to_rows → row_to_key (same as remove())
        let mut value_to_rows = self.value_to_rows.write().unwrap();
        let mut row_to_key = self.row_to_key.write().unwrap();

        // Check if row already exists with different key (for updates)
        // Now safe to do atomically since we hold both write locks
        if let Some(existing_key) = row_to_key.get(&row_id) {
            if existing_key == &key {
                // Same key, nothing to do
                return Ok(());
            }
            // Different key - remove old entry from value_to_rows
            if let Some(rows) = value_to_rows.get_mut(existing_key) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    value_to_rows.remove(existing_key);
                }
            }
            // Note: row_to_key will be overwritten below, no need to remove here

            // Also update BTree if built
            if self.btree_built.load(AtomicOrdering::Acquire) {
                let mut sorted_values = self.sorted_values.write().unwrap();
                if let Some(rows) = sorted_values.get_mut(existing_key) {
                    rows.retain(|id| *id != row_id);
                    if rows.is_empty() {
                        sorted_values.remove(existing_key);
                    }
                }
            }

            // Update prefix indexes if built
            let old_values = &existing_key.0;
            for prefix_len in 1..old_values.len() {
                let idx = prefix_len - 1;
                if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                    let prefix_key = CompositeKey(old_values[..prefix_len].to_vec());
                    let mut prefix_index = self.prefix_indexes[idx].write().unwrap();
                    if let Some(rows) = prefix_index.get_mut(&prefix_key) {
                        rows.retain(|id| *id != row_id);
                        if rows.is_empty() {
                            prefix_index.remove(&prefix_key);
                        }
                    }
                }
            }
        }

        // Check uniqueness while holding write lock (atomic check + insert)
        self.check_unique_constraint_locked(&key, row_id, &value_to_rows)?;

        // Check if BTree needs update before we move key
        let btree_needs_update = self.btree_built.load(AtomicOrdering::Acquire);

        // Add to hash index (for exact lookups) - ALWAYS maintained
        // Clone key once for hash index, keep original for row_to_key or BTree
        let key_for_hash = key.clone();
        value_to_rows.entry(key_for_hash).or_default().push(row_id);

        // Update BTree only if it was already built (consumes key, avoiding extra clone)
        if btree_needs_update {
            // Clone for row_to_key since BTree will consume original
            let key_for_reverse = key.clone();
            row_to_key.insert(row_id, key_for_reverse);
            drop(value_to_rows);
            drop(row_to_key);
            let mut sorted_values = self.sorted_values.write().unwrap();
            sorted_values.entry(key).or_default().push(row_id);
        } else {
            // No BTree update needed - move key directly to row_to_key (no clone needed)
            row_to_key.insert(row_id, key);
        }

        // Update prefix indexes only if they were already built
        for prefix_len in 1..num_cols {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                let mut prefix_index = self.prefix_indexes[idx].write().unwrap();
                prefix_index.entry(prefix_key).or_default().push(row_id);
            }
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

        let key = CompositeKey(values.to_vec());

        // LAZY: Only update hash index and reverse mapping
        {
            let mut value_to_rows = self.value_to_rows.write().unwrap();
            let mut row_to_key = self.row_to_key.write().unwrap();

            // Remove from hash index - ALWAYS maintained
            if let Some(rows) = value_to_rows.get_mut(&key) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    value_to_rows.remove(&key);
                }
            }

            // Remove reverse mapping - ALWAYS maintained
            row_to_key.remove(&row_id);
        }

        // Only update BTree if it was built
        if self.btree_built.load(AtomicOrdering::Acquire) {
            let mut sorted_values = self.sorted_values.write().unwrap();
            if let Some(rows) = sorted_values.get_mut(&key) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    sorted_values.remove(&key);
                }
            }
        }

        // Only update prefix indexes if they were built
        for prefix_len in 1..values.len() {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                let mut prefix_index = self.prefix_indexes[idx].write().unwrap();
                if let Some(rows) = prefix_index.get_mut(&prefix_key) {
                    rows.retain(|id| *id != row_id);
                    if rows.is_empty() {
                        prefix_index.remove(&prefix_key);
                    }
                }
            }
        }

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
        IndexType::MultiColumn
    }

    fn is_unique(&self) -> bool {
        self.is_unique
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        if values.is_empty() || values.len() > self.column_ids.len() {
            return Err(Error::internal("invalid value count"));
        }

        let key = CompositeKey(values.to_vec());

        if values.len() == self.column_ids.len() {
            // Exact match - use full key hash index (O(1))
            let value_to_rows = self.value_to_rows.read().unwrap();
            if let Some(row_ids) = value_to_rows.get(&key) {
                return Ok(row_ids
                    .iter()
                    .map(|&row_id| IndexEntry { row_id, ref_id: 0 })
                    .collect());
            }
            return Ok(vec![]);
        }

        // Partial match - LAZY build prefix index if needed
        self.ensure_prefix_built(values.len());

        let prefix_index = self.prefix_indexes[values.len() - 1].read().unwrap();
        if let Some(row_ids) = prefix_index.get(&key) {
            return Ok(row_ids
                .iter()
                .map(|&row_id| IndexEntry { row_id, ref_id: 0 })
                .collect());
        }
        Ok(vec![])
    }

    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        // LAZY build BTree if needed
        self.ensure_btree_built();

        let sorted_values = self.sorted_values.read().unwrap();
        let mut results = Vec::new();
        let min_key = CompositeKey(min.to_vec());
        let max_key = CompositeKey(max.to_vec());

        // Determine range bounds
        let start = if min.is_empty() {
            Bound::Unbounded
        } else if min_inclusive {
            Bound::Included(min_key.clone())
        } else {
            Bound::Excluded(min_key.clone())
        };

        let end = if max.is_empty() {
            Bound::Unbounded
        } else if max_inclusive {
            Bound::Included(max_key.clone())
        } else {
            Bound::Excluded(max_key.clone())
        };

        for (key, row_ids) in sorted_values.range((start, end)) {
            // Verify bounds for partial keys
            let mut matches = true;

            // Check min bounds
            for (i, min_val) in min.iter().enumerate() {
                if i >= key.0.len() {
                    matches = false;
                    break;
                }
                let cmp = key.0[i].cmp(min_val);
                if min_inclusive {
                    if cmp == std::cmp::Ordering::Less {
                        matches = false;
                        break;
                    }
                } else if cmp != std::cmp::Ordering::Greater {
                    matches = false;
                    break;
                }
            }

            // Check max bounds
            if matches {
                for (i, max_val) in max.iter().enumerate() {
                    if i >= key.0.len() {
                        matches = false;
                        break;
                    }
                    let cmp = key.0[i].cmp(max_val);
                    if max_inclusive {
                        if cmp == std::cmp::Ordering::Greater {
                            matches = false;
                            break;
                        }
                    } else if cmp != std::cmp::Ordering::Less {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                for &row_id in row_ids {
                    results.push(IndexEntry { row_id, ref_id: 0 });
                }
            }
        }

        Ok(results)
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        match op {
            Operator::Eq => self.find(values),
            Operator::Lt => self.find_range(&[], values, false, false),
            Operator::Lte => self.find_range(&[], values, false, true),
            Operator::Gt => self.find_range(values, &[], false, false),
            Operator::Gte => self.find_range(values, &[], true, false),
            _ => Err(Error::internal(format!("unsupported operator {:?}", op))),
        }
    }

    fn get_row_ids_equal(&self, values: &[Value]) -> Vec<i64> {
        self.find(values)
            .map(|entries| entries.into_iter().map(|e| e.row_id).collect())
            .unwrap_or_default()
    }

    fn get_row_ids_in_range(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
    ) -> Vec<i64> {
        self.find_range(min_value, max_value, include_min, include_max)
            .map(|entries| entries.into_iter().map(|e| e.row_id).collect())
            .unwrap_or_default()
    }

    fn get_filtered_row_ids(&self, _expr: &dyn Expression) -> Vec<i64> {
        Vec::new()
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);

        let mut sorted_values = self.sorted_values.write().unwrap();
        sorted_values.clear();

        let mut value_to_rows = self.value_to_rows.write().unwrap();
        value_to_rows.clear();

        for prefix_index in &self.prefix_indexes {
            let mut pi = prefix_index.write().unwrap();
            pi.clear();
        }

        let mut row_to_key = self.row_to_key.write().unwrap();
        row_to_key.clear();

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_column_index_basic() {
        let index = MultiColumnIndex::new(
            "test_idx".to_string(),
            "test_table".to_string(),
            vec!["col1".to_string(), "col2".to_string()],
            vec![0, 1],
            vec![DataType::Integer, DataType::Text],
            false,
        );

        // Add some values
        index
            .add(&[Value::Integer(1), Value::Text("a".into())], 100, 0)
            .unwrap();
        index
            .add(&[Value::Integer(1), Value::Text("b".into())], 101, 0)
            .unwrap();
        index
            .add(&[Value::Integer(2), Value::Text("a".into())], 102, 0)
            .unwrap();

        // Exact match
        let results = index
            .find(&[Value::Integer(1), Value::Text("a".into())])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 100);

        // Partial match (first column only)
        let results = index.find(&[Value::Integer(1)]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_multi_column_index_range() {
        let index = MultiColumnIndex::new(
            "test_idx".to_string(),
            "test_table".to_string(),
            vec!["amount".to_string()],
            vec![0],
            vec![DataType::Integer],
            false,
        );

        // Add values
        for i in 0..100 {
            index.add(&[Value::Integer(i)], i, 0).unwrap();
        }

        // Range query
        let results = index
            .find_range(&[Value::Integer(10)], &[Value::Integer(20)], true, true)
            .unwrap();
        assert_eq!(results.len(), 11); // 10 to 20 inclusive
    }

    #[test]
    fn test_multi_column_index_unique() {
        let index = MultiColumnIndex::new(
            "test_idx".to_string(),
            "test_table".to_string(),
            vec!["id".to_string()],
            vec![0],
            vec![DataType::Integer],
            true, // unique
        );

        // First insert should succeed
        index.add(&[Value::Integer(1)], 100, 0).unwrap();

        // Duplicate should fail
        let result = index.add(&[Value::Integer(1)], 101, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_column_index_remove() {
        let index = MultiColumnIndex::new(
            "test_idx".to_string(),
            "test_table".to_string(),
            vec!["col1".to_string()],
            vec![0],
            vec![DataType::Integer],
            false,
        );

        // Add and remove
        index.add(&[Value::Integer(1)], 100, 0).unwrap();
        let results = index.find(&[Value::Integer(1)]).unwrap();
        assert_eq!(results.len(), 1);

        index.remove(&[Value::Integer(1)], 100, 0).unwrap();
        let results = index.find(&[Value::Integer(1)]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_composite_key_ordering() {
        let k1 = CompositeKey(vec![Value::Integer(1), Value::Integer(2)]);
        let k2 = CompositeKey(vec![Value::Integer(1), Value::Integer(3)]);
        let k3 = CompositeKey(vec![Value::Integer(2), Value::Integer(1)]);

        assert!(k1 < k2);
        assert!(k2 < k3);
        assert!(k1 < k3);
    }
}
