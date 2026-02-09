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

//! B-tree Index implementation for range queries and ordered access
//!
//! Provides a B-tree based index optimized for:
//! - Range queries (O(log n + k) with BTreeMap)
//! - Ordered access (ORDER BY, MIN, MAX)
//! - Efficient column scans
//! - Aggregation operations (cached min/max)
//!
//! ## Performance Optimizations
//!
//! 1. **BTreeMap for sorted value storage**: Enables O(log n + k) range queries
//!    instead of O(n) full scans
//! 2. **Cached min/max**: O(1) MIN/MAX aggregate queries
//! 3. **SmallVec for row IDs**: Reduces memory for values with few rows
//! 4. **Parallel filtering**: Uses Rayon for large datasets (>10K values)
//!
//! B-tree index implementation for single-column indexing

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::RwLock;

use rayon::prelude::*;

use crate::common::{CompactArc, CompactVec, I64Map};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::Index;

/// Threshold for parallel filtering (number of unique values)
const PARALLEL_FILTER_THRESHOLD: usize = 10_000;

/// CompactVec for row IDs per value (16 bytes vs SmallVec's 48 bytes)
type RowIdSet = CompactVec<i64>;

/// B-tree index for efficient range queries and ordered access
///
/// This index stores column values in sorted order using BTreeMap for:
/// - O(log n + k) range queries (k = number of matching values)
/// - O(log n) point lookups via BTreeMap
/// - O(1) MIN/MAX queries via cached values
/// - Efficient IN list queries
/// - NULL handling
///
/// ## Memory Optimization
///
/// - Uses `CompactArc<Value>` for value deduplication. Each unique value is wrapped
///   in Arc for O(1) cloning (8 bytes per reference).
/// - Uses SmallVec<[i64; 4]> for row IDs per value. Since most values have
///   few duplicate rows, this avoids heap allocation for the common case.
///
/// ## Lock Ordering (Deadlock Prevention)
///
/// When acquiring multiple write locks simultaneously, always use this order:
/// 1. `sorted_values` (B-tree index)
/// 2. `row_to_value` (reverse mapping)
///
/// When locks are acquired in separate scopes (not held simultaneously), the order
/// doesn't affect deadlock safety, but this ordering should still be preferred
/// for consistency. See `add()` for the canonical pattern.
///
/// B-tree index for single-column ordered access and range queries
pub struct BTreeIndex {
    /// Index name
    name: String,
    /// Table name
    table_name: String,
    /// Column ID
    column_id: i32,
    /// Column name
    column_name: String,
    /// Data type
    data_type: DataType,
    /// Whether this is a unique index
    unique: bool,
    /// Whether the index is closed
    closed: AtomicBool,

    /// Sorted value to row IDs mapping (main index for range and equality queries)
    /// BTreeMap provides O(log n) lookups and efficient range iteration
    /// Uses CompactArc<Value> to share references with ValueArena (8 bytes per entry)
    sorted_values: RwLock<BTreeMap<CompactArc<Value>, RowIdSet>>,

    /// Row ID to value mapping (for removal operations)
    /// Uses I64Map for fast O(1) lookups with CompactArc<Value> (8 bytes per entry)
    row_to_value: RwLock<I64Map<CompactArc<Value>>>,

    /// Cached minimum value (excluding NULLs)
    cached_min: RwLock<Option<CompactArc<Value>>>,

    /// Cached maximum value (excluding NULLs)
    cached_max: RwLock<Option<CompactArc<Value>>>,

    /// Whether the min/max cache is valid
    cache_valid: AtomicBool,

    /// Reference ID counter
    next_ref_id: RwLock<i64>,
}

impl BTreeIndex {
    /// Creates a new B-tree index
    ///
    /// # Arguments
    /// * `expected_rows` - Hint for initial capacity of row_to_value map.
    ///   Pass 0 if unknown; the map will grow automatically.
    pub fn new(
        name: String,
        table_name: String,
        column_id: i32,
        column_name: String,
        data_type: DataType,
        unique: bool,
        expected_rows: usize,
    ) -> Self {
        Self {
            name,
            table_name,
            column_id,
            column_name,
            data_type,
            unique,
            closed: AtomicBool::new(false),
            sorted_values: RwLock::new(BTreeMap::new()),
            row_to_value: RwLock::new(if expected_rows > 0 {
                I64Map::with_capacity(expected_rows)
            } else {
                I64Map::new()
            }),
            cached_min: RwLock::new(None),
            cached_max: RwLock::new(None),
            cache_valid: AtomicBool::new(true),
            next_ref_id: RwLock::new(0),
        }
    }

    /// Creates a B-tree index with a custom name
    pub fn with_custom_name(
        table_name: String,
        column_id: i32,
        column_name: String,
        data_type: DataType,
        unique: bool,
        custom_name: Option<&str>,
        expected_rows: usize,
    ) -> Self {
        let name = custom_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("idx_{}_{}_btree", table_name, column_name));

        Self::new(
            name,
            table_name,
            column_id,
            column_name,
            data_type,
            unique,
            expected_rows,
        )
    }

    /// Check if the index is closed
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }
        Ok(())
    }

    /// Gets the next reference ID
    fn next_ref(&self) -> i64 {
        let mut ref_id = self.next_ref_id.write().unwrap();
        *ref_id += 1;
        *ref_id
    }

    /// Returns the number of unique values in the index
    pub fn unique_value_count(&self) -> usize {
        let sorted_values = self.sorted_values.read().unwrap();
        sorted_values.len()
    }

    /// Returns the total number of entries (row IDs) in the index
    pub fn entry_count(&self) -> usize {
        let row_to_value = self.row_to_value.read().unwrap();
        row_to_value.len()
    }

    /// Gets all values (sorted by BTreeMap ordering)
    pub fn get_all_values(&self) -> Vec<Value> {
        let sorted_values = self.sorted_values.read().unwrap();
        sorted_values.keys().map(|arc| (**arc).clone()).collect()
    }

    /// Gets all row IDs for a specific value
    /// Uses the BTreeMap for O(log n) lookup
    pub fn get_row_ids_for_value(&self, value: &Value) -> Vec<i64> {
        let sorted_values = self.sorted_values.read().unwrap();
        // Use Borrow trait - BTreeMap accepts &Value since CompactArc<Value>: Borrow<Value>
        sorted_values
            .get(value)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Checks if a value exists in the index
    /// Uses BTreeMap for O(log n) lookup
    pub fn contains_value(&self, value: &Value) -> bool {
        let sorted_values = self.sorted_values.read().unwrap();
        // Use Borrow trait - no allocation needed
        sorted_values.contains_key(value)
    }

    /// Checks if a row ID exists in the index
    pub fn contains_row(&self, row_id: i64) -> bool {
        let row_to_value = self.row_to_value.read().unwrap();
        row_to_value.contains_key(row_id)
    }

    /// Invalidate the min/max cache (called on mutations)
    #[inline]
    fn invalidate_cache(&self) {
        self.cache_valid.store(false, AtomicOrdering::Release);
    }

    /// Update the min/max cache if needed
    ///
    /// Thread-safety: Writes cache values BEFORE setting cache_valid=true.
    /// This ensures readers never see cache_valid=true with stale values.
    /// If a concurrent mutation invalidates the cache during our update,
    /// the next reader will see cache_valid=false and recompute.
    fn update_cache_if_needed(&self) {
        // Quick check without acquiring locks
        if self.cache_valid.load(AtomicOrdering::Acquire) {
            return;
        }

        let sorted_values = self.sorted_values.read().unwrap();

        // Find first non-null min
        let min = sorted_values
            .iter()
            .find(|(v, _)| !v.is_null())
            .map(|(v, _)| CompactArc::clone(v));

        // Find last non-null max
        let max = sorted_values
            .iter()
            .rev()
            .find(|(v, _)| !v.is_null())
            .map(|(v, _)| CompactArc::clone(v));

        drop(sorted_values);

        // Write values FIRST, then mark as valid
        // This ordering is critical: readers check cache_valid before reading values
        // If we set cache_valid first, readers could see stale values
        {
            let mut cached_min = self.cached_min.write().unwrap();
            let mut cached_max = self.cached_max.write().unwrap();
            *cached_min = min;
            *cached_max = max;
        }

        // Only now mark cache as valid
        // If invalidate_cache() was called between our computation and here,
        // cache_valid is already false and this store is a no-op (benign race)
        // The next reader will recompute with fresh data
        self.cache_valid.store(true, AtomicOrdering::Release);
    }

    /// Internal method to find row IDs matching an operator
    /// Uses BTreeMap for O(log n + k) range queries instead of O(n) full scans
    fn find_with_op(&self, op: Operator, value: &Value) -> Vec<i64> {
        let sorted_values = self.sorted_values.read().unwrap();

        match op {
            // Equality uses BTreeMap for O(log n) lookup via Borrow trait
            Operator::Eq | Operator::In => sorted_values
                .get(value)
                .map(|rows| rows.iter().copied().collect())
                .unwrap_or_default(),

            // Range queries need owned keys for bounds (BTreeMap limitation)
            // We create CompactArc only when needed for range operations
            Operator::Lt => {
                // range(..value) gives us all values < value
                let lookup_key = CompactArc::new(value.clone());
                let capacity = sorted_values.len() / 4; // Estimate
                let mut results = Vec::with_capacity(capacity);
                for (_, rows) in sorted_values.range(..lookup_key) {
                    results.extend_from_slice(rows.as_slice());
                }
                results
            }

            Operator::Lte => {
                // range(..=value) gives us all values <= value
                let lookup_key = CompactArc::new(value.clone());
                let capacity = sorted_values.len() / 4;
                let mut results = Vec::with_capacity(capacity);
                for (_, rows) in sorted_values.range(..=lookup_key) {
                    results.extend_from_slice(rows.as_slice());
                }
                results
            }

            Operator::Gt => {
                // range((Excluded(value), Unbounded)) gives us all values > value
                let lookup_key = CompactArc::new(value.clone());
                let capacity = sorted_values.len() / 4;
                let mut results = Vec::with_capacity(capacity);
                for (_, rows) in
                    sorted_values.range((Bound::Excluded(lookup_key), Bound::Unbounded))
                {
                    results.extend_from_slice(rows.as_slice());
                }
                results
            }

            Operator::Gte => {
                // range(value..) gives us all values >= value
                let lookup_key = CompactArc::new(value.clone());
                let capacity = sorted_values.len() / 4;
                let mut results = Vec::with_capacity(capacity);
                for (_, rows) in sorted_values.range(lookup_key..) {
                    results.extend_from_slice(rows.as_slice());
                }
                results
            }

            // Not equal and NotIn require full scan
            Operator::Ne | Operator::NotIn => {
                let capacity = sorted_values.len();
                let mut results = Vec::with_capacity(capacity);
                for (v, rows) in sorted_values.iter() {
                    if v.as_ref() != value {
                        results.extend_from_slice(rows.as_slice());
                    }
                }
                results
            }

            // Like operator needs string pattern matching - return empty
            Operator::Like => Vec::new(),

            // NULL operators need to check all values
            Operator::IsNull => {
                let mut results = Vec::new();
                for (v, rows) in sorted_values.iter() {
                    if v.is_null() {
                        results.extend_from_slice(rows.as_slice());
                    }
                }
                results
            }

            Operator::IsNotNull => {
                let capacity = sorted_values.len();
                let mut results = Vec::with_capacity(capacity);
                for (v, rows) in sorted_values.iter() {
                    if !v.is_null() {
                        results.extend_from_slice(rows.as_slice());
                    }
                }
                results
            }
        }
    }
}

impl Index for BTreeIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn table_name(&self) -> &str {
        &self.table_name
    }

    fn build(&mut self) -> Result<()> {
        self.check_closed()?;
        // Index is built incrementally via add() calls
        Ok(())
    }

    fn add(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(());
        }

        // BTree index only uses the first value (single column)
        let value = &values[0];

        // Hold write locks for entire operation to prevent race conditions
        // (read-then-write pattern is unsafe with concurrent modifications)
        let mut sorted_values = self.sorted_values.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        // Check if this row_id already exists with the same value (no-op)
        // or different value (need to remove old entry first)
        if let Some(old_arc) = row_to_value.get(row_id) {
            if old_arc.as_ref() == value {
                // Same value, nothing to do
                return Ok(());
            }
            // Different value - remove old entry from sorted index
            let old_arc = old_arc.clone();
            if let Some(rows) = sorted_values.get_mut(&old_arc) {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
                if rows.is_empty() {
                    sorted_values.remove(&old_arc);
                }
            }
        }

        // Check uniqueness constraint using BTreeMap O(log n) lookup
        if self.unique && !value.is_null() {
            if let Some(rows) = sorted_values.get(value) {
                // Check if any OTHER row has this value (allow updating same row)
                for existing_row_id in rows.iter() {
                    if *existing_row_id != row_id {
                        return Err(Error::unique_constraint(
                            &self.name,
                            &self.column_name,
                            format!("{:?}", value),
                        ));
                    }
                }
            }
        }

        // Try to reuse existing Arc if value already exists (O(1) clone)
        // Only create new Arc if this is a new unique value
        let arc_value = if let Some((existing_arc, _)) = sorted_values.get_key_value(value) {
            // Value exists - reuse the existing Arc (O(1) atomic refcount bump)
            CompactArc::clone(existing_arc)
        } else {
            // New unique value - create Arc once
            CompactArc::new(value.clone())
        };

        // Add to sorted index (for O(log n) range and equality queries)
        // Insert in sorted order for O(N+M) intersection/union without re-sorting
        let btree_rows = sorted_values
            .entry(CompactArc::clone(&arc_value))
            .or_default();
        if let Err(pos) = btree_rows.binary_search(&row_id) {
            btree_rows.insert(pos, row_id);
        }

        // Add to row -> value mapping (stores Arc reference)
        row_to_value.insert(row_id, arc_value);

        // Drop locks before invalidating cache
        drop(sorted_values);
        drop(row_to_value);

        // Invalidate min/max cache
        self.invalidate_cache();

        Ok(())
    }

    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        self.check_closed()?;

        for (row_id, values) in entries.iter() {
            self.add(values, row_id, self.next_ref())?;
        }
        Ok(())
    }

    fn remove(&self, _values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        self.check_closed()?;

        // Hold write locks for entire operation to prevent race conditions
        let mut sorted_values = self.sorted_values.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        // Check if the row exists and remove atomically
        if let Some(arc_value) = row_to_value.remove(row_id) {
            // Remove from sorted index (row_ids are sorted, use binary search)
            if let Some(rows) = sorted_values.get_mut(&arc_value) {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
                if rows.is_empty() {
                    sorted_values.remove(&arc_value);
                }
            }

            // Drop locks before invalidating cache
            drop(sorted_values);
            drop(row_to_value);

            // Invalidate min/max cache
            self.invalidate_cache();
        }

        Ok(())
    }

    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        self.check_closed()?;

        for (row_id, values) in entries.iter() {
            self.remove(values, row_id, 0)?;
        }
        Ok(())
    }

    /// Optimized batch add with single lock acquisition
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.check_closed()?;

        // Acquire write locks ONCE for entire batch
        let mut sorted_values = self.sorted_values.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        // Reserve capacity
        row_to_value.reserve(entries.len());

        // For unique indexes: pre-check all entries before modifying
        if self.unique {
            for &(row_id, values) in entries {
                if values.is_empty() {
                    continue;
                }
                let value = &values[0];
                if value.is_null() {
                    continue;
                }

                // Check uniqueness against existing index
                if let Some(rows) = sorted_values.get(value) {
                    for &existing_row_id in rows.iter() {
                        if existing_row_id != row_id {
                            return Err(Error::unique_constraint(
                                &self.name,
                                &self.column_name,
                                format!("{:?}", value),
                            ));
                        }
                    }
                }
            }

            // Check for intra-batch duplicates
            // Use AHashMap per CLAUDE.md guidelines for Value keys (HashDoS resistance)
            let mut seen: ahash::AHashMap<&Value, i64> =
                ahash::AHashMap::with_capacity(entries.len());
            for &(row_id, values) in entries {
                if values.is_empty() {
                    continue;
                }
                let value = &values[0];
                if value.is_null() {
                    continue;
                }
                if let Some(&existing_row_id) = seen.get(value) {
                    if existing_row_id != row_id {
                        return Err(Error::unique_constraint(
                            &self.name,
                            &self.column_name,
                            format!("{:?}", value),
                        ));
                    }
                }
                seen.insert(value, row_id);
            }
        }

        // All checks passed - now add all entries
        for &(row_id, values) in entries {
            if values.is_empty() {
                continue;
            }

            let value = &values[0];

            // Handle existing row (update case)
            if let Some(old_arc) = row_to_value.get(row_id) {
                if old_arc.as_ref() == value {
                    continue; // Same value, skip
                }
                // Different value - remove old entry
                let old_arc = old_arc.clone();
                if let Some(rows) = sorted_values.get_mut(&old_arc) {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                    }
                    if rows.is_empty() {
                        sorted_values.remove(&old_arc);
                    }
                }
            }

            // Try to reuse existing Arc if value exists (memory deduplication)
            let arc_value = if let Some((existing_arc, _)) = sorted_values.get_key_value(value) {
                CompactArc::clone(existing_arc)
            } else {
                CompactArc::new(value.clone())
            };

            // Add to sorted index (sorted insertion)
            let btree_rows = sorted_values
                .entry(CompactArc::clone(&arc_value))
                .or_default();
            if let Err(pos) = btree_rows.binary_search(&row_id) {
                btree_rows.insert(pos, row_id);
            }

            // Add to row_to_value
            row_to_value.insert(row_id, arc_value);
        }

        // Drop locks before invalidating cache
        drop(sorted_values);
        drop(row_to_value);

        self.invalidate_cache();
        Ok(())
    }

    /// Optimized batch remove with single lock acquisition
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.check_closed()?;

        // Acquire write locks ONCE for entire batch
        let mut sorted_values = self.sorted_values.write().unwrap();
        let mut row_to_value = self.row_to_value.write().unwrap();

        let mut any_removed = false;

        for &(row_id, _values) in entries {
            // BTreeIndex looks up by row_id, values aren't used
            if let Some(arc_value) = row_to_value.remove(row_id) {
                if let Some(rows) = sorted_values.get_mut(&arc_value) {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                        any_removed = true;
                    }
                    if rows.is_empty() {
                        sorted_values.remove(&arc_value);
                    }
                }
            }
        }

        // Drop locks before invalidating cache
        drop(sorted_values);
        drop(row_to_value);

        if any_removed {
            self.invalidate_cache();
        }

        Ok(())
    }

    fn column_ids(&self) -> &[i32] {
        std::slice::from_ref(&self.column_id)
    }

    fn column_names(&self) -> &[String] {
        std::slice::from_ref(&self.column_name)
    }

    fn data_types(&self) -> &[DataType] {
        std::slice::from_ref(&self.data_type)
    }

    fn index_type(&self) -> IndexType {
        IndexType::BTree
    }

    fn is_unique(&self) -> bool {
        self.unique
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(Vec::new());
        }

        let value = &values[0];
        let sorted_values = self.sorted_values.read().unwrap();
        // Use Borrow trait - no allocation needed for equality lookup
        let entries = sorted_values
            .get(value)
            .map(|rows| {
                rows.iter()
                    .map(|&row_id| IndexEntry {
                        row_id,
                        ref_id: row_id, // Use row_id as ref_id for simplicity
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(entries)
    }

    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if min.is_empty() || max.is_empty() {
            return Ok(Vec::new());
        }

        let min_val = &min[0];
        let max_val = &max[0];

        let sorted_values = self.sorted_values.read().unwrap();

        // Build the range bounds for BTreeMap using CompactArc<Value>
        let min_arc = CompactArc::new(min_val.clone());
        let max_arc = CompactArc::new(max_val.clone());

        let min_bound = if min_inclusive {
            Bound::Included(min_arc)
        } else {
            Bound::Excluded(min_arc)
        };

        let max_bound = if max_inclusive {
            Bound::Included(max_arc)
        } else {
            Bound::Excluded(max_arc)
        };

        // Estimate capacity based on range
        let capacity = sorted_values.len() / 4;
        let mut entries = Vec::with_capacity(capacity);

        // Use BTreeMap range for O(log n + k) instead of O(n)
        for (_, rows) in sorted_values.range((min_bound, max_bound)) {
            for &row_id in rows.iter() {
                entries.push(IndexEntry {
                    row_id,
                    ref_id: row_id,
                });
            }
        }

        Ok(entries)
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(Vec::new());
        }

        let value = &values[0];
        let row_ids = self.find_with_op(op, value);

        Ok(row_ids
            .into_iter()
            .map(|row_id| IndexEntry {
                row_id,
                ref_id: row_id,
            })
            .collect())
    }

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        if values.is_empty() {
            return;
        }

        let value = &values[0];
        let sorted_values = self.sorted_values.read().unwrap();
        // Use Borrow trait - no allocation needed for equality lookup
        if let Some(rows) = sorted_values.get(value) {
            // Optimization: extend_from_slice uses memcpy for efficient bulk copy
            buffer.extend_from_slice(rows.as_slice());
        }
    }

    fn get_row_ids_in_range_into(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
        buffer: &mut Vec<i64>,
    ) {
        if min_value.is_empty() || max_value.is_empty() {
            return;
        }

        // OPTIMIZATION: Collect row_ids directly without intermediate IndexEntry allocation
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        let min_val = &min_value[0];
        let max_val = &max_value[0];

        let sorted_values = self.sorted_values.read().unwrap();

        // Build the range bounds for BTreeMap using CompactArc<Value>
        let min_arc = CompactArc::new(min_val.clone());
        let max_arc = CompactArc::new(max_val.clone());

        let min_bound = if include_min {
            Bound::Included(min_arc)
        } else {
            Bound::Excluded(min_arc)
        };

        let max_bound = if include_max {
            Bound::Included(max_arc)
        } else {
            Bound::Excluded(max_arc)
        };

        for (_, rows) in sorted_values.range((min_bound, max_bound)) {
            buffer.extend_from_slice(rows.as_slice());
        }
    }

    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec {
        if self.closed.load(AtomicOrdering::Acquire) {
            return RowIdVec::new();
        }

        // Try to get comparison info from expression
        if let Some((col_name, operator, value)) = expr.get_comparison_info() {
            // Only handle expressions for this column
            if col_name == self.column_name {
                match operator {
                    Operator::Eq => {
                        // OPTIMIZATION: Use from_ref to avoid clone
                        return self.get_row_ids_equal(std::slice::from_ref(value));
                    }
                    Operator::Gt | Operator::Gte | Operator::Lt | Operator::Lte => {
                        // OPTIMIZATION: Use from_ref to avoid clone
                        let value_slice = std::slice::from_ref(value);
                        let empty_slice: &[Value] = &[];
                        let (min_vals, max_vals, include_min, include_max) = match operator {
                            Operator::Gt => (value_slice, empty_slice, false, false),
                            Operator::Gte => (value_slice, empty_slice, true, false),
                            Operator::Lt => (empty_slice, value_slice, false, false),
                            Operator::Lte => (empty_slice, value_slice, false, true),
                            _ => return RowIdVec::new(),
                        };

                        return self.get_row_ids_in_range(
                            min_vals,
                            max_vals,
                            include_min,
                            include_max,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Try to extract range from collect_comparisons (for AND expressions)
        let comparisons = expr.collect_comparisons();
        if !comparisons.is_empty() {
            // OPTIMIZATION: Use references instead of cloning values
            let mut min_val: Option<&Value> = None;
            let mut max_val: Option<&Value> = None;
            let mut include_min = false;
            let mut include_max = false;
            let mut eq_val: Option<&Value> = None;

            for (col_name, op, val) in &comparisons {
                if *col_name != self.column_name {
                    continue;
                }
                match op {
                    Operator::Eq => eq_val = Some(*val),
                    Operator::Gt => {
                        min_val = Some(*val);
                        include_min = false;
                    }
                    Operator::Gte => {
                        min_val = Some(*val);
                        include_min = true;
                    }
                    Operator::Lt => {
                        max_val = Some(*val);
                        include_max = false;
                    }
                    Operator::Lte => {
                        max_val = Some(*val);
                        include_max = true;
                    }
                    _ => {}
                }
            }

            // Equality takes precedence
            if let Some(val) = eq_val {
                return self.get_row_ids_equal(std::slice::from_ref(val));
            }

            // Range query - use from_ref to avoid cloning
            let empty_slice: &[Value] = &[];
            if min_val.is_some() || max_val.is_some() {
                let min_vals = min_val.map(std::slice::from_ref).unwrap_or(empty_slice);
                let max_vals = max_val.map(std::slice::from_ref).unwrap_or(empty_slice);
                return self.get_row_ids_in_range(min_vals, max_vals, include_min, include_max);
            }
        }

        // Fallback: evaluate expression on each value
        // Use parallel processing for large datasets
        let row_to_value = self.row_to_value.read().unwrap();

        if row_to_value.len() >= PARALLEL_FILTER_THRESHOLD {
            // Parallel filtering with Rayon
            let pairs: Vec<_> = row_to_value.iter().collect();
            let collected: Vec<i64> = pairs
                .par_iter()
                .filter_map(|&(row_id, arc_value)| {
                    // Dereference Arc to get the Value
                    let row = crate::core::Row::from_values(vec![(**arc_value).clone()]);
                    if expr.evaluate(&row).unwrap_or(false) {
                        Some(row_id)
                    } else {
                        None
                    }
                })
                .collect();
            RowIdVec::from_vec(collected)
        } else {
            // Sequential for small datasets
            let mut results = RowIdVec::with_capacity(row_to_value.len() / 4);
            for (row_id, arc_value) in row_to_value.iter() {
                // Dereference Arc to get the Value
                let row = crate::core::Row::from_values(vec![(**arc_value).clone()]);
                if expr.evaluate(&row).unwrap_or(false) {
                    results.push(row_id);
                }
            }
            results
        }
    }

    /// Returns the minimum value in the index
    ///
    /// OPTIMIZATION: O(1) using cached min value from BTreeMap.
    /// The cache is lazily updated when accessed after mutations.
    fn get_min_value(&self) -> Option<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        // Update cache if needed
        self.update_cache_if_needed();

        // Return cached min (clone the inner Value from Arc)
        let cached_min = self.cached_min.read().unwrap();
        cached_min.as_ref().map(|arc| (**arc).clone())
    }

    /// Returns the maximum value in the index
    ///
    /// OPTIMIZATION: O(1) using cached max value from BTreeMap.
    /// The cache is lazily updated when accessed after mutations.
    fn get_max_value(&self) -> Option<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        // Update cache if needed
        self.update_cache_if_needed();

        // Return cached max (clone the inner Value from Arc)
        let cached_max = self.cached_max.read().unwrap();
        cached_max.as_ref().map(|arc| (**arc).clone())
    }

    fn get_all_values(&self) -> Vec<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }
        // Use sorted_values for deterministic ordering
        let sorted_values = self.sorted_values.read().unwrap();
        sorted_values.keys().map(|arc| (**arc).clone()).collect()
    }

    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let sorted_values = self.sorted_values.read().unwrap();
        // Count non-null values without cloning
        let count = sorted_values.keys().filter(|v| !v.is_null()).count();
        Some(count)
    }

    fn get_row_ids_ordered(
        &self,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<i64>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        let sorted_values = self.sorted_values.read().unwrap();

        // Pre-allocate with expected capacity
        let mut result = Vec::with_capacity(limit.min(128));
        let mut skipped = 0;

        // Iterate in the requested order using BTreeMap's natural ordering
        if ascending {
            // Forward iteration (ascending order)
            'outer: for row_ids in sorted_values.values() {
                for &row_id in row_ids {
                    // Handle offset
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }

                    result.push(row_id);

                    // Check limit
                    if result.len() >= limit {
                        break 'outer;
                    }
                }
            }
        } else {
            // Reverse iteration (descending order)
            'outer: for row_ids in sorted_values.values().rev() {
                for &row_id in row_ids {
                    // Handle offset
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }

                    result.push(row_id);

                    // Check limit
                    if result.len() >= limit {
                        break 'outer;
                    }
                }
            }
        }

        Some(result)
    }

    fn get_grouped_row_ids(&self) -> Option<Vec<(Value, Vec<i64>)>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        let sorted_values = self.sorted_values.read().unwrap();

        // Convert BTreeMap entries to (Value, Vec<i64>) pairs in sorted order
        let result: Vec<(Value, Vec<i64>)> = sorted_values
            .iter()
            .map(|(arc_value, row_ids)| ((**arc_value).clone(), row_ids.to_vec()))
            .collect();

        Some(result)
    }

    fn for_each_group(
        &self,
        callback: &mut dyn FnMut(&Value, &[i64]) -> Result<bool>,
    ) -> Option<Result<()>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        let sorted_values = self.sorted_values.read().unwrap();

        // Iterate through groups in sorted order without collecting
        for (arc_value, row_ids) in sorted_values.iter() {
            // Pass slice reference directly - no allocation (Arc dereferences to Value)
            match callback(arc_value.as_ref(), row_ids.as_slice()) {
                Ok(true) => continue,          // Continue to next group
                Ok(false) => break,            // Early termination requested
                Err(e) => return Some(Err(e)), // Propagate error
            }
        }

        Some(Ok(()))
    }

    fn clear(&self) -> Result<()> {
        let mut sorted_values = self.sorted_values.write().unwrap();
        sorted_values.clear();
        drop(sorted_values);

        let mut row_to_value = self.row_to_value.write().unwrap();
        row_to_value.clear();
        drop(row_to_value);

        *self.cached_min.write().unwrap() = None;
        *self.cached_max.write().unwrap() = None;

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);
        self.clear()
    }
}

/// Intersects two sorted slices of i64 and returns the common elements
///
/// This is an O(n) algorithm when both slices are sorted.
/// Returns a pooled RowIdVec for efficient memory reuse.
pub fn intersect_sorted_ids(a: &[i64], b: &[i64]) -> RowIdVec {
    if a.is_empty() || b.is_empty() {
        return RowIdVec::new();
    }

    // Fast path: check if ranges don't overlap at all
    let (a_min, a_max) = (a[0], a[a.len() - 1]);
    let (b_min, b_max) = (b[0], b[b.len() - 1]);
    if a_max < b_min || b_max < a_min {
        return RowIdVec::new();
    }

    // Use the smaller slice for the outer loop (binary search approach)
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    let mut result = RowIdVec::with_capacity(smaller.len());

    for &val in smaller {
        if larger.binary_search(&val).is_ok() {
            result.push(val);
        }
    }

    result
}

/// Intersects multiple sorted slices and returns common elements
/// Returns a pooled RowIdVec for efficient memory reuse.
pub fn intersect_multiple_sorted_ids(slices: &[&[i64]]) -> RowIdVec {
    if slices.is_empty() {
        return RowIdVec::new();
    }
    if slices.len() == 1 {
        let mut result = RowIdVec::with_capacity(slices[0].len());
        result.extend_from_slice(slices[0]);
        return result;
    }

    let mut result = intersect_sorted_ids(slices[0], slices[1]);
    for slice in &slices[2..] {
        if result.is_empty() {
            break;
        }
        result = intersect_sorted_ids(&result, slice);
    }
    result
}

/// Union two sorted slices of row IDs
///
/// Returns a sorted slice containing all unique elements from both input slices.
/// This is an O(n+m) algorithm when both slices are sorted.
/// Used for OR expression index optimization.
/// Returns a pooled RowIdVec for efficient memory reuse.
pub fn union_sorted_ids(a: &[i64], b: &[i64]) -> RowIdVec {
    if a.is_empty() {
        let mut result = RowIdVec::with_capacity(b.len());
        result.extend_from_slice(b);
        return result;
    }
    if b.is_empty() {
        let mut result = RowIdVec::with_capacity(a.len());
        result.extend_from_slice(a);
        return result;
    }

    let mut result = RowIdVec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    // Append remaining elements
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    result
}

/// Union multiple sorted slices and returns all unique elements
/// Returns a pooled RowIdVec for efficient memory reuse.
pub fn union_multiple_sorted_ids(slices: &[&[i64]]) -> RowIdVec {
    if slices.is_empty() {
        return RowIdVec::new();
    }
    if slices.len() == 1 {
        let mut result = RowIdVec::with_capacity(slices[0].len());
        result.extend_from_slice(slices[0]);
        return result;
    }

    let mut result = union_sorted_ids(slices[0], slices[1]);
    for slice in &slices[2..] {
        result = union_sorted_ids(&result, slice);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> BTreeIndex {
        BTreeIndex::new(
            "idx_test".to_string(),
            "test_table".to_string(),
            0,
            "id".to_string(),
            DataType::Integer,
            false,
            0, // expected_rows: 0 for tests (will grow as needed)
        )
    }

    #[test]
    fn test_btree_index_creation() {
        let index = create_test_index();
        assert_eq!(index.name(), "idx_test");
        assert_eq!(index.table_name(), "test_table");
        assert_eq!(index.column_names()[0], "id");
        assert_eq!(index.index_type(), IndexType::BTree);
        assert!(!index.is_unique());
    }

    #[test]
    fn test_btree_index_with_custom_name() {
        let index = BTreeIndex::with_custom_name(
            "users".to_string(),
            1,
            "email".to_string(),
            DataType::Text,
            true,
            Some("custom_email_idx"),
            0,
        );
        assert_eq!(index.name(), "custom_email_idx");
        assert!(index.is_unique());
    }

    #[test]
    fn test_btree_index_add_and_find() {
        let index = create_test_index();

        // Add some values
        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(200)], 2, 2).unwrap();
        index.add(&[Value::Integer(100)], 3, 3).unwrap(); // Duplicate value

        assert_eq!(index.unique_value_count(), 2);
        assert_eq!(index.entry_count(), 3);

        // Find exact match
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 2);

        // Find non-existent
        let entries = index.find(&[Value::Integer(999)]).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_btree_index_unique_constraint() {
        let index = BTreeIndex::new(
            "idx_unique".to_string(),
            "test".to_string(),
            0,
            "id".to_string(),
            DataType::Integer,
            true, // unique
            0,
        );

        // First insert should succeed
        assert!(index.add(&[Value::Integer(1)], 1, 1).is_ok());

        // Duplicate should fail
        let result = index.add(&[Value::Integer(1)], 2, 2);
        assert!(result.is_err());

        // NULL values should be allowed multiple times
        assert!(index.add(&[Value::null_unknown()], 3, 3).is_ok());
        assert!(index.add(&[Value::null_unknown()], 4, 4).is_ok());
    }

    #[test]
    fn test_btree_index_remove() {
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::Integer(200)], 3, 3).unwrap();

        assert_eq!(index.entry_count(), 3);

        // Remove one entry
        index.remove(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 2);

        // Value 100 should still exist (for row 2)
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 2);

        // Remove the last entry with value 100
        index.remove(&[Value::Integer(100)], 2, 2).unwrap();
        assert!(!index.contains_value(&Value::Integer(100)));
    }

    #[test]
    fn test_btree_index_range_query() {
        let index = create_test_index();

        for i in 1..=10 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Range [30, 70] inclusive
        let entries = index
            .find_range(&[Value::Integer(30)], &[Value::Integer(70)], true, true)
            .unwrap();
        assert_eq!(entries.len(), 5); // 30, 40, 50, 60, 70

        // Range (30, 70) exclusive
        let entries = index
            .find_range(&[Value::Integer(30)], &[Value::Integer(70)], false, false)
            .unwrap();
        assert_eq!(entries.len(), 3); // 40, 50, 60
    }

    #[test]
    fn test_btree_index_operators() {
        let index = create_test_index();

        for i in 1..=5 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Less than 30
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Integer(30)])
            .unwrap();
        assert_eq!(entries.len(), 2); // 10, 20

        // Greater than or equal to 40
        let entries = index
            .find_with_operator(Operator::Gte, &[Value::Integer(40)])
            .unwrap();
        assert_eq!(entries.len(), 2); // 40, 50

        // Not equal to 30
        let entries = index
            .find_with_operator(Operator::Ne, &[Value::Integer(30)])
            .unwrap();
        assert_eq!(entries.len(), 4); // 10, 20, 40, 50
    }

    #[test]
    fn test_btree_index_batch_operations() {
        let index = create_test_index();

        let mut entries = I64Map::new();
        entries.insert(1, vec![Value::Integer(100)]);
        entries.insert(2, vec![Value::Integer(200)]);
        entries.insert(3, vec![Value::Integer(100)]);

        index.add_batch(&entries).unwrap();
        assert_eq!(index.entry_count(), 3);

        // Remove batch
        let mut to_remove = I64Map::new();
        to_remove.insert(1, vec![Value::Integer(100)]);
        to_remove.insert(2, vec![Value::Integer(200)]);

        index.remove_batch(&to_remove).unwrap();
        assert_eq!(index.entry_count(), 1);
    }

    #[test]
    fn test_btree_index_null_handling() {
        let index = create_test_index();

        index.add(&[Value::null_unknown()], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::null_unknown()], 3, 3).unwrap();

        // IS NULL - value parameter is ignored for IsNull operator
        let entries = index
            .find_with_operator(Operator::IsNull, &[Value::null_unknown()])
            .unwrap();
        assert_eq!(entries.len(), 2);

        // IS NOT NULL - value parameter is ignored for IsNotNull operator
        let entries = index
            .find_with_operator(Operator::IsNotNull, &[Value::null_unknown()])
            .unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_btree_index_text_values() {
        let index = BTreeIndex::new(
            "idx_name".to_string(),
            "test".to_string(),
            0,
            "name".to_string(),
            DataType::Text,
            false,
            0,
        );

        index.add(&[Value::text("Alice")], 1, 1).unwrap();
        index.add(&[Value::text("Bob")], 2, 2).unwrap();
        index.add(&[Value::text("Charlie")], 3, 3).unwrap();

        // Range query on text (alphabetical)
        let entries = index
            .find_range(&[Value::text("Alice")], &[Value::text("Bob")], true, true)
            .unwrap();
        assert_eq!(entries.len(), 2); // Alice, Bob
    }

    #[test]
    fn test_btree_index_close() {
        let mut index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 1);

        index.close().unwrap();

        // Operations should fail after close
        assert!(index.add(&[Value::Integer(200)], 2, 2).is_err());
        assert!(index.find(&[Value::Integer(100)]).is_err());
    }

    #[test]
    fn test_btree_index_get_all_values() {
        let index = create_test_index();

        index.add(&[Value::Integer(30)], 1, 1).unwrap();
        index.add(&[Value::Integer(10)], 2, 2).unwrap();
        index.add(&[Value::Integer(20)], 3, 3).unwrap();

        let values = index.get_all_values();
        assert_eq!(values.len(), 3);
        // Values may not be sorted (HashMap)
        assert!(values.contains(&Value::Integer(10)));
        assert!(values.contains(&Value::Integer(20)));
        assert!(values.contains(&Value::Integer(30)));
    }

    #[test]
    fn test_btree_index_row_id_helpers() {
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::Integer(200)], 3, 3).unwrap();

        // get_row_ids_equal
        let row_ids = index.get_row_ids_equal(&[Value::Integer(100)]);
        assert_eq!(row_ids.len(), 2);

        // get_row_ids_in_range
        let row_ids =
            index.get_row_ids_in_range(&[Value::Integer(100)], &[Value::Integer(200)], true, true);
        assert_eq!(row_ids.len(), 3);
    }

    #[test]
    fn test_btree_index_lte_gte_operators() {
        let index = create_test_index();

        for i in 1..=5 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Less than or equal to 30
        let entries = index
            .find_with_operator(Operator::Lte, &[Value::Integer(30)])
            .unwrap();
        assert_eq!(entries.len(), 3); // 10, 20, 30

        // Greater than 20
        let entries = index
            .find_with_operator(Operator::Gt, &[Value::Integer(20)])
            .unwrap();
        assert_eq!(entries.len(), 3); // 30, 40, 50
    }

    #[test]
    fn test_mixed_integer_float_ordering() {
        // Test that Integer and Float values are properly ordered together
        // This validates the Ord implementation consistency with PartialEq
        let index = BTreeIndex::new(
            "idx_mixed".to_string(),
            "test".to_string(),
            0,
            "value".to_string(),
            DataType::Float, // Use Float type to allow mixed values
            false,
            0,
        );

        // Add mixed Integer and Float values
        index.add(&[Value::Integer(1)], 1, 1).unwrap();
        index.add(&[Value::Float(2.5)], 2, 2).unwrap();
        index.add(&[Value::Integer(3)], 3, 3).unwrap();
        index.add(&[Value::Float(1.5)], 4, 4).unwrap();
        index.add(&[Value::Float(3.0)], 5, 5).unwrap(); // Same as Integer(3)

        // Integer(3) == Float(3.0), so they should be in the same bucket
        // The add() for Float(3.0) should be a no-op since row 5 doesn't exist yet
        // Actually, it should add normally since row 5 is new
        assert_eq!(index.entry_count(), 5);

        // Range query: all values >= 2
        let entries = index
            .find_with_operator(Operator::Gte, &[Value::Integer(2)])
            .unwrap();
        // Should find: 2.5, 3, 3.0 (row_ids: 2, 3, 5)
        assert_eq!(entries.len(), 3);

        // Range query: all values < 2
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Float(2.0)])
            .unwrap();
        // Should find: 1, 1.5 (row_ids: 1, 4)
        assert_eq!(entries.len(), 2);

        // Equality lookup with cross-type comparison
        // Integer(3) should find Float(3.0) and vice versa if they're in the same bucket
        let entries = index.find(&[Value::Integer(3)]).unwrap();
        // Due to how BTreeMap works with Ord, Integer(3) and Float(3.0) are Equal
        // so they should be in the same entry
        assert!(!entries.is_empty()); // At least row 3
    }

    #[test]
    fn test_nan_handling_in_range_queries() {
        // Test that NaN values are handled correctly (ordered last)
        let index = BTreeIndex::new(
            "idx_nan".to_string(),
            "test".to_string(),
            0,
            "value".to_string(),
            DataType::Float,
            false,
            0,
        );

        index.add(&[Value::Float(1.0)], 1, 1).unwrap();
        index.add(&[Value::Float(f64::NAN)], 2, 2).unwrap();
        index.add(&[Value::Float(2.0)], 3, 3).unwrap();
        index.add(&[Value::Float(f64::INFINITY)], 4, 4).unwrap();
        index.add(&[Value::Float(f64::NEG_INFINITY)], 5, 5).unwrap();

        assert_eq!(index.entry_count(), 5);

        // NaN should be ordered last, so Gt(any number) should NOT include NaN
        // Actually, per our Ord impl, NaN > any number, so Gt(100) would include NaN
        let entries = index
            .find_with_operator(Operator::Gt, &[Value::Float(100.0)])
            .unwrap();
        // Should find: INFINITY, NAN (row_ids: 4, 2)
        assert_eq!(entries.len(), 2);

        // Lt(NaN) - everything except NaN should be less than NaN
        // But NaN comparisons are tricky - partial_cmp returns None for NaN
        // Our Ord impl makes NaN > everything, so Lt(NaN) should return all non-NaN values
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Float(f64::NAN)])
            .unwrap();
        // Should find: NEG_INFINITY, 1.0, 2.0, INFINITY (row_ids: 5, 1, 3, 4)
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn test_high_cardinality_duplicates() {
        // Test performance and correctness with many row_ids per value
        let index = BTreeIndex::new(
            "idx_highcard".to_string(),
            "test".to_string(),
            0,
            "status".to_string(),
            DataType::Text,
            false,
            0,
        );

        // Add 100 rows with just 3 distinct values (low cardinality column)
        for i in 0..100 {
            let status = match i % 3 {
                0 => "active",
                1 => "pending",
                _ => "completed",
            };
            index
                .add(&[Value::text(status)], i as i64, i as i64)
                .unwrap();
        }

        assert_eq!(index.unique_value_count(), 3);
        assert_eq!(index.entry_count(), 100);

        // Find all "active" rows (should be ~33)
        let entries = index.find(&[Value::text("active")]).unwrap();
        assert_eq!(entries.len(), 34); // 0, 3, 6, 9, ... 99

        // Find all "pending" rows
        let entries = index.find(&[Value::text("pending")]).unwrap();
        assert_eq!(entries.len(), 33); // 1, 4, 7, ... 97

        // Update a row's value (tests the optimization that avoids O(n) contains)
        index.add(&[Value::text("archived")], 0, 0).unwrap();
        assert_eq!(index.unique_value_count(), 4);

        // "active" should now have one less row
        let entries = index.find(&[Value::text("active")]).unwrap();
        assert_eq!(entries.len(), 33);

        // "archived" should have the moved row
        let entries = index.find(&[Value::text("archived")]).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_update_same_value_is_noop() {
        // Test that updating a row with the same value is a no-op
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 1);

        // Add same row_id with same value - should be no-op
        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 1);

        // Verify the value is still there
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 1);
    }

    #[test]
    fn test_update_different_value_moves_row() {
        // Test that updating a row with a different value moves it
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        assert_eq!(index.entry_count(), 2);

        // Update row 1 to have value 200
        index.add(&[Value::Integer(200)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 2);
        assert_eq!(index.unique_value_count(), 2);

        // Value 100 should only have row 2 now
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 2);

        // Value 200 should have row 1
        let entries = index.find(&[Value::Integer(200)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 1);
    }
}
