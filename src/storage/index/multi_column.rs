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

use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use rustc_hash::FxHashMap;

use crate::common::{CompactArc, CompactVec, I64Map};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
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
/// - `value_to_rows`: FxHashMap<CompositeKey, CompactVec> for exact lookups (always maintained)
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
    sorted_values: RwLock<BTreeMap<CompositeKey, CompactVec<i64>>>,
    btree_built: AtomicBool,

    /// Hash index for exact lookups (full key) - always maintained
    value_to_rows: RwLock<FxHashMap<CompositeKey, CompactVec<i64>>>,

    /// Prefix indexes: LAZY built on first partial query
    /// Index 0 = first column, Index 1 = first two columns, etc.
    prefix_indexes: Vec<RwLock<FxHashMap<CompositeKey, CompactVec<i64>>>>,
    prefix_built: Vec<AtomicBool>,

    /// Reverse mapping for removal - uses Vec<CompactArc<Value>> for memory efficiency
    /// Arc references are shared with ValueArena (8 bytes per value)
    row_to_key: RwLock<I64Map<Vec<CompactArc<Value>>>>,
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
        expected_rows: usize,
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
            value_to_rows: RwLock::new(if expected_rows > 0 {
                FxHashMap::with_capacity_and_hasher(expected_rows, Default::default())
            } else {
                FxHashMap::default()
            }),
            prefix_indexes,
            prefix_built,
            row_to_key: RwLock::new(if expected_rows > 0 {
                I64Map::with_capacity(expected_rows)
            } else {
                I64Map::new()
            }),
        }
    }

    /// Helper to compare stored CompactArc<Value> with input &[Value]
    /// Specialized unrolling for common cases (1-4 columns): ~5% faster than zip().all()
    #[inline]
    fn values_match(stored: &[CompactArc<Value>], input: &[Value]) -> bool {
        if stored.len() != input.len() {
            return false;
        }
        match stored.len() {
            0 => true,
            1 => stored[0].as_ref() == &input[0],
            2 => stored[0].as_ref() == &input[0] && stored[1].as_ref() == &input[1],
            3 => {
                stored[0].as_ref() == &input[0]
                    && stored[1].as_ref() == &input[1]
                    && stored[2].as_ref() == &input[2]
            }
            4 => {
                stored[0].as_ref() == &input[0]
                    && stored[1].as_ref() == &input[1]
                    && stored[2].as_ref() == &input[2]
                    && stored[3].as_ref() == &input[3]
            }
            _ => stored
                .iter()
                .zip(input.iter())
                .all(|(s, i)| s.as_ref() == i),
        }
    }

    /// Build BTree index lazily from hash index (on first range query)
    fn ensure_btree_built(&self) {
        if self.btree_built.load(AtomicOrdering::Acquire) {
            return;
        }

        // Acquire both locks to prevent race condition:
        // We need to ensure no inserts happen between reading hash and setting btree_built
        let value_to_rows = self.value_to_rows.read();
        let mut sorted_values = self.sorted_values.write();

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
        let row_to_key = self.row_to_key.read();
        let mut prefix_index = self.prefix_indexes[idx].write();

        // Double-check after acquiring write lock
        if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
            return;
        }

        // Build prefix index from row_to_key (read lock prevents concurrent inserts)
        // Insert in sorted order for O(N+M) merge operations
        for (row_id, arc_values) in row_to_key.iter() {
            if arc_values.len() >= prefix_len {
                // Dereference CompactArc<Value> to create CompositeKey for prefix
                let prefix_key = CompositeKey(
                    arc_values[..prefix_len]
                        .iter()
                        .map(|a| (**a).clone())
                        .collect(),
                );
                let rows = prefix_index.entry(prefix_key).or_default();
                if let Err(pos) = rows.binary_search(&row_id) {
                    rows.insert(pos, row_id);
                }
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
        value_to_rows: &FxHashMap<CompositeKey, CompactVec<i64>>,
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

        // Track old key if this is an update (for BTree/prefix cleanup after releasing locks)
        let mut old_key_for_cleanup: Option<Vec<CompactArc<Value>>> = None;

        // Acquire BOTH write locks FIRST to prevent race conditions during updates
        // Lock order: value_to_rows → row_to_key (same as remove())
        let mut value_to_rows = self.value_to_rows.write();
        let mut row_to_key = self.row_to_key.write();

        // Check if row already exists with different key (for updates)
        // Now safe to do atomically since we hold both write locks
        if let Some(existing_arc_values) = row_to_key.get(row_id) {
            // Compare CompactArc<Value> contents with new values using optimized helper
            if Self::values_match(existing_arc_values, values) {
                // Same key, nothing to do
                return Ok(());
            }

            // Different key - save for BTree/prefix cleanup AFTER releasing locks
            old_key_for_cleanup = Some(existing_arc_values.clone());

            // Create CompositeKey from existing Arc values for removal
            let existing_key =
                CompositeKey(existing_arc_values.iter().map(|a| (**a).clone()).collect());

            // Remove old entry from value_to_rows
            if let Some(rows) = value_to_rows.get_mut(&existing_key) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    value_to_rows.remove(&existing_key);
                }
            }
            // Note: row_to_key will be overwritten below, no need to remove here
        }

        // Check uniqueness while holding write lock (atomic check + insert)
        self.check_unique_constraint_locked(&key, row_id, &value_to_rows)?;

        // Check if BTree needs update before we move key
        let btree_needs_update = self.btree_built.load(AtomicOrdering::Acquire);

        // Wrap values in Arc for O(1) cloning in row_to_key
        let arc_values: Vec<CompactArc<Value>> =
            values.iter().map(|v| CompactArc::new(v.clone())).collect();

        // Add to hash index (for exact lookups) - ALWAYS maintained
        // Clone key once for hash index, keep original for BTree
        // Insert in sorted order for O(N+M) merge operations
        let key_for_hash = key.clone();
        let hash_rows = value_to_rows.entry(key_for_hash).or_default();
        if let Err(pos) = hash_rows.binary_search(&row_id) {
            hash_rows.insert(pos, row_id);
        }

        // Store CompactArc<Value> references in row_to_key (memory efficient)
        row_to_key.insert(row_id, arc_values);

        // Release main locks BEFORE acquiring BTree/prefix locks
        // This ensures consistent lock ordering across all code paths
        drop(value_to_rows);
        drop(row_to_key);

        // Update BTree only if it was already built
        if btree_needs_update {
            let mut sorted_values = self.sorted_values.write();

            // First remove old entry if this was an update
            if let Some(ref old_arc_values) = old_key_for_cleanup {
                let existing_key =
                    CompositeKey(old_arc_values.iter().map(|a| (**a).clone()).collect());
                if let Some(rows) = sorted_values.get_mut(&existing_key) {
                    rows.retain(|id| *id != row_id);
                    if rows.is_empty() {
                        sorted_values.remove(&existing_key);
                    }
                }
            }

            // Then add new entry
            let btree_rows = sorted_values.entry(key).or_default();
            if let Err(pos) = btree_rows.binary_search(&row_id) {
                btree_rows.insert(pos, row_id);
            }
        }

        // Update prefix indexes only if they were already built
        // Insert in sorted order for O(N+M) merge operations
        for prefix_len in 1..num_cols {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let mut prefix_index = self.prefix_indexes[idx].write();

                // First remove old entry if this was an update
                if let Some(ref old_arc_values) = old_key_for_cleanup {
                    if old_arc_values.len() >= prefix_len {
                        let old_prefix_key = CompositeKey(
                            old_arc_values[..prefix_len]
                                .iter()
                                .map(|a| (**a).clone())
                                .collect(),
                        );
                        if let Some(rows) = prefix_index.get_mut(&old_prefix_key) {
                            rows.retain(|id| *id != row_id);
                            if rows.is_empty() {
                                prefix_index.remove(&old_prefix_key);
                            }
                        }
                    }
                }

                // Then add new entry
                let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                let prefix_rows = prefix_index.entry(prefix_key).or_default();
                if let Err(pos) = prefix_rows.binary_search(&row_id) {
                    prefix_rows.insert(pos, row_id);
                }
            }
        }

        Ok(())
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

        let key = CompositeKey(values.to_vec());

        // LAZY: Only update hash index and reverse mapping
        {
            let mut value_to_rows = self.value_to_rows.write();
            let mut row_to_key = self.row_to_key.write();

            // Remove from hash index (row_ids are sorted, use binary search)
            if let Some(rows) = value_to_rows.get_mut(&key) {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
                if rows.is_empty() {
                    value_to_rows.remove(&key);
                }
            }

            // Remove reverse mapping - ALWAYS maintained
            row_to_key.remove(row_id);
        }

        // Only update BTree if it was built (row_ids are sorted, use binary search)
        if self.btree_built.load(AtomicOrdering::Acquire) {
            let mut sorted_values = self.sorted_values.write();
            if let Some(rows) = sorted_values.get_mut(&key) {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
                if rows.is_empty() {
                    sorted_values.remove(&key);
                }
            }
        }

        // Only update prefix indexes if they were built (row_ids are sorted, use binary search)
        for prefix_len in 1..values.len() {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                let mut prefix_index = self.prefix_indexes[idx].write();
                if let Some(rows) = prefix_index.get_mut(&prefix_key) {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                    }
                    if rows.is_empty() {
                        prefix_index.remove(&prefix_key);
                    }
                }
            }
        }

        Ok(())
    }

    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        for (row_id, values) in entries.iter() {
            self.remove(values, row_id, 0)?;
        }
        Ok(())
    }

    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        let num_cols = self.column_ids.len();

        // Validate all entries first
        for &(_row_id, values) in entries {
            if values.len() != num_cols {
                return Err(Error::internal(format!(
                    "expected {} values, got {}",
                    num_cols,
                    values.len()
                )));
            }
        }

        // Acquire ALL write locks ONCE for the entire batch
        let mut value_to_rows = self.value_to_rows.write();
        let mut row_to_key = self.row_to_key.write();

        // Reserve capacity to reduce reallocations
        value_to_rows.reserve(entries.len());
        row_to_key.reserve(entries.len());

        // Check if BTree needs update
        let btree_needs_update = self.btree_built.load(AtomicOrdering::Acquire);

        // Pre-check unique constraints if this is a unique index
        if self.is_unique {
            // Build a set of keys in this batch for intra-batch duplicate detection
            let mut batch_keys: FxHashMap<CompositeKey, i64> =
                FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());

            for &(row_id, values) in entries {
                // Skip if any value is NULL (NULL doesn't violate uniqueness)
                let has_null = values.iter().any(|v| v.is_null());
                if has_null {
                    continue;
                }

                let key = CompositeKey(values.to_vec());

                // Check intra-batch duplicates
                if let Some(&existing_row_id) = batch_keys.get(&key) {
                    if existing_row_id != row_id {
                        let values_str: Vec<String> =
                            values.iter().map(|v| format!("{:?}", v)).collect();
                        return Err(Error::unique_constraint(
                            &self.name,
                            self.column_names.join(", "),
                            format!("[{}]", values_str.join(", ")),
                        ));
                    }
                }

                // Check against existing index
                if let Some(existing_rows) = value_to_rows.get(&key) {
                    if !existing_rows.is_empty() && !existing_rows.contains(&row_id) {
                        let values_str: Vec<String> =
                            values.iter().map(|v| format!("{:?}", v)).collect();
                        return Err(Error::unique_constraint(
                            &self.name,
                            self.column_names.join(", "),
                            format!("[{}]", values_str.join(", ")),
                        ));
                    }
                }

                batch_keys.insert(key, row_id);
            }
        }

        // Collect old keys for rows that need updating (to remove from BTree/prefix later)
        // We need to collect these before modifying row_to_key
        let mut updates_to_old_key: Vec<(i64, Vec<CompactArc<Value>>)> = Vec::new();

        // Now add all entries
        for &(row_id, values) in entries {
            let key = CompositeKey(values.to_vec());

            // Wrap values in Arc for O(1) cloning in row_to_key
            let arc_values: Vec<CompactArc<Value>> =
                values.iter().map(|v| CompactArc::new(v.clone())).collect();

            // Check if row already exists with different key (for updates)
            if let Some(existing_arc_values) = row_to_key.get(row_id) {
                if !Self::values_match(existing_arc_values, values) {
                    // Different key - collect for BTree/prefix removal later
                    updates_to_old_key.push((row_id, existing_arc_values.clone()));

                    // Remove old entry from value_to_rows
                    let existing_key =
                        CompositeKey(existing_arc_values.iter().map(|a| (**a).clone()).collect());

                    if let Some(rows) = value_to_rows.get_mut(&existing_key) {
                        rows.retain(|id| *id != row_id);
                        if rows.is_empty() {
                            value_to_rows.remove(&existing_key);
                        }
                    }
                }
            }

            // Add to hash index - insert in sorted order
            let hash_rows = value_to_rows.entry(key).or_default();
            if let Err(pos) = hash_rows.binary_search(&row_id) {
                hash_rows.insert(pos, row_id);
            }

            // Store Arc references in row_to_key
            row_to_key.insert(row_id, arc_values);
        }

        // Release main locks before updating BTree/prefix indexes
        drop(value_to_rows);
        drop(row_to_key);

        // Update BTree only if it was already built
        if btree_needs_update {
            let mut sorted_values = self.sorted_values.write();

            // First remove old entries for updated rows
            for (row_id, existing_arc_values) in &updates_to_old_key {
                let existing_key =
                    CompositeKey(existing_arc_values.iter().map(|a| (**a).clone()).collect());
                if let Some(rows) = sorted_values.get_mut(&existing_key) {
                    rows.retain(|id| *id != *row_id);
                    if rows.is_empty() {
                        sorted_values.remove(&existing_key);
                    }
                }
            }

            // Then add new entries
            for &(row_id, values) in entries {
                let key = CompositeKey(values.to_vec());
                let btree_rows = sorted_values.entry(key).or_default();
                if let Err(pos) = btree_rows.binary_search(&row_id) {
                    btree_rows.insert(pos, row_id);
                }
            }
        }

        // Update prefix indexes only if they were already built
        for prefix_len in 1..num_cols {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let mut prefix_index = self.prefix_indexes[idx].write();

                // First remove old entries for updated rows
                for (row_id, existing_arc_values) in &updates_to_old_key {
                    if existing_arc_values.len() >= prefix_len {
                        let prefix_key = CompositeKey(
                            existing_arc_values[..prefix_len]
                                .iter()
                                .map(|a| (**a).clone())
                                .collect(),
                        );
                        if let Some(rows) = prefix_index.get_mut(&prefix_key) {
                            rows.retain(|id| *id != *row_id);
                            if rows.is_empty() {
                                prefix_index.remove(&prefix_key);
                            }
                        }
                    }
                }

                // Then add new entries
                for &(row_id, values) in entries {
                    let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                    let prefix_rows = prefix_index.entry(prefix_key).or_default();
                    if let Err(pos) = prefix_rows.binary_search(&row_id) {
                        prefix_rows.insert(pos, row_id);
                    }
                }
            }
        }

        Ok(())
    }

    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        // Acquire BOTH write locks ONCE for entire batch
        let mut value_to_rows = self.value_to_rows.write();
        let mut row_to_key = self.row_to_key.write();

        // Remove all entries from hash index and reverse mapping
        for &(row_id, values) in entries {
            let key = CompositeKey(values.to_vec());

            // Remove from hash index
            if let Some(rows) = value_to_rows.get_mut(&key) {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
                if rows.is_empty() {
                    value_to_rows.remove(&key);
                }
            }

            // Remove reverse mapping
            row_to_key.remove(row_id);
        }

        // Release main locks before updating BTree/prefix indexes
        drop(value_to_rows);
        drop(row_to_key);

        // Only update BTree if it was built
        if self.btree_built.load(AtomicOrdering::Acquire) {
            let mut sorted_values = self.sorted_values.write();
            for &(row_id, values) in entries {
                let key = CompositeKey(values.to_vec());
                if let Some(rows) = sorted_values.get_mut(&key) {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                    }
                    if rows.is_empty() {
                        sorted_values.remove(&key);
                    }
                }
            }
        }

        // Only update prefix indexes if they were built
        for prefix_len in 1..self.column_ids.len() {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let mut prefix_index = self.prefix_indexes[idx].write();
                for &(row_id, values) in entries {
                    if values.len() >= prefix_len {
                        let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                        if let Some(rows) = prefix_index.get_mut(&prefix_key) {
                            if let Ok(pos) = rows.binary_search(&row_id) {
                                rows.remove(pos);
                            }
                            if rows.is_empty() {
                                prefix_index.remove(&prefix_key);
                            }
                        }
                    }
                }
            }
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
            let value_to_rows = self.value_to_rows.read();
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

        let prefix_index = self.prefix_indexes[values.len() - 1].read();
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

        let sorted_values = self.sorted_values.read();
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
            let mut matches = true;

            // Post-check for PARTIAL key bounds only.
            // When bounds have fewer elements than the stored composite key,
            // the BTree range over-includes keys that share the prefix but
            // don't satisfy column-wise semantics. E.g., Excluded([1]) includes
            // [1,5] (composite [1,5] > [1]), but for `col1 > 1` we need col1 strictly > 1.
            //
            // When bounds have the same number of elements as the key (full key),
            // the BTree's CompositeKey::Ord already gives correct results and
            // no post-filtering is needed.

            // Check min bounds (only for partial key prefix)
            if !min.is_empty() && min.len() < key.0.len() {
                for (i, min_val) in min.iter().enumerate() {
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
            }

            // Check max bounds (only for partial key prefix)
            if matches && !max.is_empty() && max.len() < key.0.len() {
                for (i, max_val) in max.iter().enumerate() {
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

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        if values.is_empty() || values.len() > self.column_ids.len() {
            return;
        }

        let key = CompositeKey(values.to_vec());

        if values.len() == self.column_ids.len() {
            // Exact match - use full key hash index (O(1))
            let value_to_rows = self.value_to_rows.read();
            if let Some(row_ids) = value_to_rows.get(&key) {
                // extend_from_slice uses memcpy for efficient bulk copy
                buffer.extend_from_slice(row_ids.as_slice());
            }
            return;
        }

        // Partial match - LAZY build prefix index if needed
        self.ensure_prefix_built(values.len());

        let prefix_index = self.prefix_indexes[values.len() - 1].read();
        if let Some(row_ids) = prefix_index.get(&key) {
            buffer.extend_from_slice(row_ids.as_slice());
        }
    }

    // Uses default trait implementation: get_row_ids_in_range delegates to get_row_ids_in_range_into

    fn get_filtered_row_ids(&self, _expr: &dyn Expression) -> RowIdVec {
        RowIdVec::new()
    }

    fn clear(&self) -> Result<()> {
        self.sorted_values.write().clear();
        self.btree_built.store(false, AtomicOrdering::Release);
        self.value_to_rows.write().clear();
        for (prefix_index, built_flag) in self.prefix_indexes.iter().zip(self.prefix_built.iter()) {
            prefix_index.write().clear();
            built_flag.store(false, AtomicOrdering::Release);
        }
        self.row_to_key.write().clear();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);
        self.clear()
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
            0,
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
            0,
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
            0,
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
            0,
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
