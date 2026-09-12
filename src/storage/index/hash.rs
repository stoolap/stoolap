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

//! Hash Index implementation for O(1) equality lookups
//!
//! This module provides a hash-based index optimized for equality queries on
//! TEXT, VARCHAR, and UUID columns where string comparisons are expensive.
//!
//! ## Performance characteristics:
//! - INSERT: O(1) amortized
//! - DELETE: O(1) amortized
//! - FIND exact: O(1) - hash lookup
//! - RANGE queries: NOT SUPPORTED (use BTree index instead)
//!
//! ## When to use HashIndex:
//! - TEXT/VARCHAR columns with equality queries (email, username, token)
//! - UUID columns (always equality lookups)
//! - High-cardinality string columns
//!
//! ## When NOT to use HashIndex:
//! - Range queries (<, >, BETWEEN) - use BTree
//! - Low-cardinality columns - use Bitmap
//! - ORDER BY optimization - use BTree
//!
//! ## Implementation:
//! Uses `ahash` for fast hashing, avoiding O(strlen) comparisons
//! that would occur with each B-tree node comparison.
//!
//! ## Known Limitations:
//!
//! ### Paired Lock Pattern for Write Operations
//! The `add` and `remove` methods acquire both RwLock write locks simultaneously:
//! - `row_to_hash`: Maps row_id -> hash for efficient removal
//! - `hash_to_values`: Maps hash -> values for collision handling
//!
//! This pattern ensures atomicity but can cause write contention under high
//! concurrent write load. Read operations (`find`) only acquire a single read lock.
//!
//! **Impact**: Under heavy concurrent INSERTs/UPDATEs to the same indexed column,
//! writers will serialize. This is acceptable for most workloads where writes are
//! less frequent than reads.
//!
//! **Alternative considered**: Using a single RwLock<struct> would have similar
//! contention characteristics. A concurrent map could help for truly concurrent writes
//! but would complicate atomicity across the two maps.

use parking_lot::RwLock;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::common::{CompactArc, CompactVec, I64Map};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::{ComparisonExpr, Expression, InListExpr};
use crate::storage::index::memory::{hash_table_bytes, value_bytes, IndexMemory, IndexMemoryOwner};
use crate::storage::traits::Index;

/// Fixed seeds for deterministic hashing across add/find operations
const HASH_SEEDS: [u64; 4] = [
    0x517cc1b727220a95,
    0x8a36afbc28b36e9c,
    0x2f24bc8d75cd8b0a,
    0xe9a5e3f10d13d6f7,
];

/// Compute hash for a slice of Values using ahash
/// Uses fixed seeds for deterministic hashing across add/find operations
fn hash_values(values: &[Value]) -> u64 {
    let hasher_builder =
        ahash::RandomState::with_seeds(HASH_SEEDS[0], HASH_SEEDS[1], HASH_SEEDS[2], HASH_SEEDS[3]);
    let mut hasher = hasher_builder.build_hasher();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// Hash Index for O(1) equality lookups
///
/// Optimized for TEXT, VARCHAR, and UUID columns where equality queries
/// dominate and string comparison costs are significant.
///
/// ## Key features:
/// - O(1) exact lookups via hash
/// - Uses ahash - faster than SipHash
/// - SmallVec optimization for single-match case (common for unique indexes)
/// - Thread-safe with RwLock
/// - Memory efficient with CompactArc<Value> for O(1) cloning
///
/// ## Limitations:
/// - Does NOT support range queries (returns error)
/// - Does NOT support ORDER BY optimization
/// - Does NOT support partial key matches (all columns required)
pub struct HashIndex {
    name: String,
    table_name: String,
    column_names: Vec<String>,
    column_ids: Vec<i32>,
    data_types: Vec<DataType>,
    is_unique: bool,
    closed: AtomicBool,

    /// Row ID -> hash mapping for efficient removal
    /// Uses I64Map for fast O(1) lookups (optimized for i64 keys)
    row_to_hash: RwLock<I64Map<u64>>,

    /// Full values storage for hash collision handling and unique constraint checking
    /// Maps hash -> (values as CompactArc<Value>, row_ids) for collision resolution
    /// Uses CompactArc<Value> to share references with ValueArena (8 bytes per value)
    hash_to_values: RwLock<HashStorage>,
    memory: IndexMemoryOwner,
}

type HashGroups = Vec<(Vec<CompactArc<Value>>, CompactVec<i64>)>;

struct HashStorage {
    map: FxHashMap<u64, HashGroups>,
    nested_bytes: u128,
    capacity_high_water: usize,
}

impl HashStorage {
    fn estimated_bytes(&self) -> u128 {
        hash_table_bytes::<u64, HashGroups>(self.capacity_high_water)
    }

    fn group_bytes(values: &Vec<CompactArc<Value>>, rows: &CompactVec<i64>) -> u128 {
        (values.capacity() * std::mem::size_of::<CompactArc<Value>>()) as u128
            + (rows.capacity() * std::mem::size_of::<i64>()) as u128
            + values.iter().map(value_bytes).sum::<u128>()
    }

    fn prune_empty(groups: &mut HashGroups) -> u128 {
        let mut removed = 0;
        groups.retain(|(values, rows)| {
            if rows.is_empty() {
                removed += Self::group_bytes(values, rows);
                false
            } else {
                true
            }
        });
        if groups.is_empty() {
            removed += (groups.capacity()
                * std::mem::size_of::<(Vec<CompactArc<Value>>, CompactVec<i64>)>())
                as u128;
        }
        removed
    }

    fn add(&mut self, hash: u64, values: &[Value], row_id: i64) {
        let groups = self.map.entry(hash).or_default();
        for (stored, rows) in groups.iter_mut() {
            if HashIndex::values_match(stored, values) {
                if let Err(pos) = rows.binary_search(&row_id) {
                    let before = rows.capacity();
                    rows.insert(pos, row_id);
                    self.nested_bytes +=
                        ((rows.capacity() - before) * std::mem::size_of::<i64>()) as u128;
                }
                return;
            }
        }
        let stored: Vec<_> = values
            .iter()
            .map(|value| CompactArc::new(value.clone()))
            .collect();
        let mut rows = CompactVec::new();
        rows.push(row_id);
        self.nested_bytes += Self::group_bytes(&stored, &rows);
        let before = groups.capacity();
        groups.push((stored, rows));
        self.nested_bytes += ((groups.capacity() - before)
            * std::mem::size_of::<(Vec<CompactArc<Value>>, CompactVec<i64>)>())
            as u128;
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
    }

    fn remove_old(&mut self, hash: u64, row_id: i64) {
        if let Some(groups) = self.map.get_mut(&hash) {
            for (_, rows) in groups.iter_mut() {
                rows.retain(|id| *id != row_id);
            }
            self.nested_bytes -= Self::prune_empty(groups);
            if groups.is_empty() {
                self.map.remove(&hash);
            }
        }
    }

    fn remove(&mut self, hash: u64, values: &[Value], row_id: i64) {
        if let Some(groups) = self.map.get_mut(&hash) {
            for (stored, rows) in groups.iter_mut() {
                if HashIndex::values_match(stored, values) {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                    }
                    break;
                }
            }
            self.nested_bytes -= Self::prune_empty(groups);
            if groups.is_empty() {
                self.map.remove(&hash);
            }
        }
    }
}

impl std::ops::Deref for HashStorage {
    type Target = FxHashMap<u64, HashGroups>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl std::fmt::Debug for HashIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HashIndex")
            .field("name", &self.name)
            .field("table_name", &self.table_name)
            .field("column_names", &self.column_names)
            .field("column_ids", &self.column_ids)
            .field("is_unique", &self.is_unique)
            .field("closed", &self.closed.load(AtomicOrdering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl HashIndex {
    /// Create a new HashIndex
    ///
    /// # Arguments
    /// * `expected_rows` - Hint for initial capacity of row_to_hash map.
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
        let row_to_hash = if expected_rows > 0 {
            I64Map::with_capacity(expected_rows)
        } else {
            I64Map::new()
        };
        let map = if expected_rows > 0 {
            FxHashMap::with_capacity_and_hasher(expected_rows, Default::default())
        } else {
            FxHashMap::default()
        };
        let hash_to_values = HashStorage {
            capacity_high_water: map.capacity(),
            map,
            nested_bytes: 0,
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
            + row_to_hash.allocation_bytes() as u128;
        let memory = IndexMemoryOwner::new::<Self>(requested, hash_to_values.estimated_bytes());
        Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            closed: AtomicBool::new(false),
            row_to_hash: RwLock::new(row_to_hash),
            hash_to_values: RwLock::new(hash_to_values),
            memory,
        }
    }

    fn mutate(
        &self,
        mutation: impl FnOnce(&mut I64Map<u64>, &mut HashStorage) -> Result<()>,
    ) -> Result<()> {
        let mut rows = self.row_to_hash.write();
        let mut storage = self.hash_to_values.write();
        let before = rows.allocation_bytes() as u128 + storage.nested_bytes;
        let estimated_before = storage.estimated_bytes();
        let result = mutation(&mut rows, &mut storage);
        storage.capacity_high_water = storage.capacity_high_water.max(storage.map.capacity());
        self.memory.account.resize(
            before,
            rows.allocation_bytes() as u128 + storage.nested_bytes,
        );
        self.memory
            .account
            .resize_estimate(estimated_before, storage.estimated_bytes());
        result
    }

    /// Check uniqueness constraint
    #[allow(clippy::type_complexity)]
    fn check_unique_constraint(
        &self,
        values: &[Value],
        row_id: i64,
        hash: u64,
        hash_to_values: &FxHashMap<u64, Vec<(Vec<CompactArc<Value>>, CompactVec<i64>)>>,
    ) -> Result<()> {
        if !self.is_unique {
            return Ok(());
        }

        // NULL values don't violate uniqueness
        for v in values {
            if v.is_null() {
                return Ok(());
            }
        }

        if let Some(entries) = hash_to_values.get(&hash) {
            for (stored_values, row_ids) in entries {
                // Compare CompactArc<Value> contents with input values using optimized helper
                if Self::values_match(stored_values, values) && !row_ids.is_empty() {
                    // Check if any row_id is different (would be a duplicate)
                    if row_ids.iter().any(|&id| id != row_id) {
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
        Ok(())
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
}

impl Index for HashIndex {
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

        let num_cols = self.column_ids.len();
        if values.len() != num_cols {
            return Err(Error::internal(format!(
                "expected {} values, got {}",
                num_cols,
                values.len()
            )));
        }

        let hash = hash_values(values);

        self.mutate(|row_to_hash, storage| {
            if let Some(&old_hash) = row_to_hash.get(row_id) {
                if old_hash == hash {
                    if let Some(entries) = storage.get(&old_hash) {
                        let same = entries.iter().any(|(stored, rows)| {
                            rows.contains(&row_id) && Self::values_match(stored, values)
                        });
                        if same {
                            return Ok(());
                        }
                    }
                }
                storage.remove_old(old_hash, row_id);
            }
            self.check_unique_constraint(values, row_id, hash, storage)?;
            row_to_hash.insert(row_id, hash);
            storage.add(hash, values, row_id);
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

        let hash = hash_values(values);

        self.mutate(|row_to_hash, storage| {
            row_to_hash.remove(row_id);
            storage.remove(hash, values, row_id);
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
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        let num_cols = self.column_ids.len();

        self.mutate(|row_to_hash, hash_to_values| {
            row_to_hash.reserve(entries.len());
            hash_to_values.map.reserve(entries.len());
            hash_to_values.capacity_high_water = hash_to_values
                .capacity_high_water
                .max(hash_to_values.map.capacity());

            // For unique indexes: pre-check all entries before modifying
            // This detects both existing duplicates AND intra-batch duplicates
            if self.is_unique {
                // Track actual values seen in this batch for intra-batch duplicate detection
                // Use AHashMap<&[Value], i64> to properly handle hash collisions with different values
                // (using hash as key would lose track of entries when collisions occur)
                let mut batch_values: ahash::AHashMap<&[Value], i64> =
                    ahash::AHashMap::with_capacity(entries.len());

                for &(row_id, values) in entries {
                    if values.len() != num_cols {
                        return Err(Error::internal(format!(
                            "expected {} values, got {}",
                            num_cols,
                            values.len()
                        )));
                    }

                    // Skip NULL values (don't violate uniqueness)
                    if values.iter().any(|v| v.is_null()) {
                        continue;
                    }

                    let hash = hash_values(values);

                    // Check against existing index entries
                    self.check_unique_constraint(values, row_id, hash, hash_to_values)?;

                    // Check against other entries in this batch using actual values as key
                    if let Some(&existing_row_id) = batch_values.get(values) {
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
                    batch_values.insert(values, row_id);
                }
            }

            // All uniqueness checks passed - now add all entries
            for &(row_id, values) in entries {
                if values.len() != num_cols {
                    continue; // Already validated above for unique indexes
                }

                let hash = hash_values(values);

                // Handle existing row (update case)
                if let Some(&old_hash) = row_to_hash.get(row_id) {
                    if old_hash == hash {
                        // Same hash - but could be hash collision with different values!
                        // Must compare actual values to determine if this is a no-op
                        let old_values_match = if let Some(entries) = hash_to_values.get(&old_hash)
                        {
                            entries.iter().any(|(stored_values, row_ids)| {
                                row_ids.contains(&row_id)
                                    && Self::values_match(stored_values, values)
                            })
                        } else {
                            false
                        };
                        if old_values_match {
                            continue; // Truly the same values - no update needed
                        }
                        // Hash collision: different values with same hash - fall through to update
                    }

                    // Different hash (or same hash with different values) - remove old entry
                    hash_to_values.remove_old(old_hash, row_id);
                }

                // Add to row_to_hash
                row_to_hash.insert(row_id, hash);

                hash_to_values.add(hash, values, row_id);
            }

            Ok(())
        })
    }

    /// Optimized batch remove with single lock acquisition
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        self.mutate(|row_to_hash, hash_to_values| {
            for &(row_id, values) in entries {
                let hash = hash_values(values);

                // Remove from row_to_hash
                row_to_hash.remove(row_id);

                hash_to_values.remove(hash, values, row_id);
            }

            Ok(())
        })
    }

    fn remove_batch_ids(&self, row_ids: &[i64]) -> Option<Result<()>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Some(Err(Error::IndexClosed));
        }
        Some(self.mutate(|row_to_hash, hash_to_values| {
            // Group the rows by hash, then subtract each group from its bucket
            // in one pass per value
            let mut by_hash: FxHashMap<u64, Vec<i64>> = FxHashMap::default();
            for &row_id in row_ids {
                if let Some(hash) = row_to_hash.remove(row_id) {
                    by_hash.entry(hash).or_default().push(row_id);
                }
            }
            for (hash, mut ids) in by_hash {
                ids.sort_unstable();
                if let Some(val_entries) = hash_to_values.map.get_mut(&hash) {
                    for (_, bucket) in val_entries.iter_mut() {
                        super::subtract_sorted(bucket, &ids);
                    }
                    hash_to_values.nested_bytes -= HashStorage::prune_empty(val_entries);
                    if val_entries.is_empty() {
                        hash_to_values.map.remove(&hash);
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
        IndexType::Hash
    }

    fn is_unique(&self) -> bool {
        self.is_unique
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        // Hash index only supports exact matches on all columns
        if values.len() != self.column_ids.len() {
            return Err(Error::internal(
                "hash index requires exact match on all columns",
            ));
        }

        let hash = hash_values(values);

        let hash_to_values = self.hash_to_values.read();

        if let Some(entries) = hash_to_values.get(&hash) {
            // Handle hash collisions by checking actual values (CompactArc<Value>)
            for (stored_values, row_ids) in entries {
                if Self::values_match(stored_values, values) {
                    return Ok(row_ids
                        .iter()
                        .map(|&row_id| IndexEntry { row_id, ref_id: 0 })
                        .collect());
                }
            }
        }

        Ok(vec![])
    }

    fn find_range(
        &self,
        _min: &[Value],
        _max: &[Value],
        _min_inclusive: bool,
        _max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        // Hash index does NOT support range queries
        Err(Error::internal(
            "hash index does not support range queries; use btree index instead",
        ))
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        // Hash index only supports equality
        match op {
            Operator::Eq => self.find(values),
            _ => Err(Error::internal(format!(
                "hash index only supports equality operator, not {:?}",
                op
            ))),
        }
    }

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        // Hash index only supports exact matches on all columns
        if values.len() != self.column_ids.len() {
            return;
        }

        let hash = hash_values(values);
        let hash_to_values = self.hash_to_values.read();

        if let Some(entries) = hash_to_values.get(&hash) {
            // Handle hash collisions by checking actual values (CompactArc<Value>)
            for (stored_values, row_ids) in entries {
                if Self::values_match(stored_values, values) {
                    // Use extend_from_slice for memcpy instead of iterator
                    buffer.extend_from_slice(row_ids.as_slice());
                    return;
                }
            }
        }
    }

    /// Optimized IN list lookup - acquires lock once for all values
    fn get_row_ids_in_into(&self, value_list: &[Value], buffer: &mut Vec<i64>) {
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        // Single lock acquisition for all lookups
        let hash_to_values = self.hash_to_values.read();

        for value in value_list {
            // Hash without cloning - pass reference to slice
            let hash = hash_values(std::slice::from_ref(value));
            if let Some(entries) = hash_to_values.get(&hash) {
                for (stored_values, row_ids) in entries {
                    // Compare CompactArc<Value> with input value
                    if stored_values.len() == 1 && stored_values[0].as_ref() == value {
                        buffer.extend_from_slice(row_ids.as_slice());
                        break;
                    }
                }
            }
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
        // Hash index does not support range queries - do nothing
    }

    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec {
        // Try to optimize for IN list expressions
        if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
            // Check if this IN list is on our indexed column
            if let Some(col_name) = in_list.get_column_name() {
                if self.column_names.len() == 1 && self.column_names[0] == col_name {
                    // Use efficient get_row_ids_in
                    let values = in_list.get_values();
                    return self.get_row_ids_in(values);
                }
            }
        }

        // Try to optimize for equality expressions
        if let Some(comparison) = expr.as_any().downcast_ref::<ComparisonExpr>() {
            if comparison.operator() == Operator::Eq {
                if let Some(col_name) = comparison.get_column_name() {
                    if self.column_names.len() == 1 && self.column_names[0] == col_name {
                        // Use efficient equality lookup
                        return self.get_row_ids_equal(&[comparison.value().to_value()]);
                    }
                }
            }
        }

        // Fallback: return all row IDs and let caller filter
        // This is inefficient but necessary for correctness for complex expressions
        let hash_to_values = self.hash_to_values.read();
        let mut results = RowIdVec::new();

        for entries in hash_to_values.values() {
            for (_values, row_ids) in entries {
                results.extend_from_slice(row_ids.as_slice());
            }
        }

        results
    }

    fn get_all_values(&self) -> Vec<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }
        let hash_to_values = self.hash_to_values.read();
        let mut exports = crate::storage::mvcc::read_memory::ExportBatch::new();
        let mut result = Vec::with_capacity(hash_to_values.len());
        for entries in hash_to_values.values() {
            for (values, _row_ids) in entries {
                // For single-column index, return the value directly
                if values.len() == 1 {
                    result.push(exports.capture_value(&values[0]));
                }
            }
        }
        result
    }

    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let hash_to_values = self.hash_to_values.read();
        // Count unique value combinations excluding null
        let mut count = 0;
        for entries in hash_to_values.values() {
            for (values, _row_ids) in entries {
                // For single-column index, check if value is null
                if values.len() == 1 {
                    if !values[0].is_null() {
                        count += 1;
                    }
                } else {
                    // For multi-column index, count if not all values are null
                    if !values.iter().all(|v| v.is_null()) {
                        count += 1;
                    }
                }
            }
        }
        Some(count)
    }

    fn clear(&self) -> Result<()> {
        self.mutate(|rows, storage| {
            rows.clear();
            storage.map.clear();
            storage.nested_bytes = 0;
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

    fn memory_index(unique: bool, expected_rows: usize) -> HashIndex {
        HashIndex::new(
            "idx_key".to_string(),
            "items".to_string(),
            vec!["key".to_string()],
            vec![1],
            vec![DataType::Text],
            unique,
            expected_rows,
        )
    }

    fn assert_requested_allocations(index: &HashIndex) {
        let rows = index.row_to_hash.read();
        let storage = index.hash_to_values.read();
        let mut nested = 0u128;
        for groups in storage.map.values() {
            nested += (groups.capacity()
                * std::mem::size_of::<(Vec<CompactArc<Value>>, CompactVec<i64>)>())
                as u128;
            for (values, row_ids) in groups {
                nested += (values.capacity() * std::mem::size_of::<CompactArc<Value>>()) as u128;
                nested += (row_ids.capacity() * std::mem::size_of::<i64>()) as u128;
                for value in values {
                    nested += (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Value>())
                        as u128
                        + value.heap_bytes() as u128;
                }
            }
        }
        assert_eq!(storage.nested_bytes, nested);
        let expected = crate::storage::mvcc::memory::arc_allocation_bytes::<HashIndex>() as u128
            + nested
            + rows.allocation_bytes() as u128
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
        assert!(
            index.memory.account.estimated_bytes() as u128
                >= (storage.map.capacity() * std::mem::size_of::<(u64, HashGroups)>()) as u128
        );
    }

    #[test]
    fn hash_accounting_keeps_capacity_after_failed_batch_and_clear() {
        for expected_rows in [0, 2048] {
            let mut index = memory_index(true, expected_rows);
            assert_requested_allocations(&index);
            let initial = index.memory.account.requested_bytes();
            let invalid: Vec<_> = (1..=1024).map(|id| (id, &[][..])).collect();
            assert!(index.add_batch_slice(&invalid).is_err());
            assert_requested_allocations(&index);
            if expected_rows == 0 {
                assert!(index.memory.account.requested_bytes() > initial);
            }
            let keys: Vec<_> = (1..=512)
                .map(|id| {
                    [Value::text(format!(
                        "a retained text key with heap storage {id}"
                    ))]
                })
                .collect();
            let entries: Vec<_> = keys
                .iter()
                .enumerate()
                .map(|(id, values)| (id as i64, &values[..]))
                .collect();
            index.add_batch_slice(&entries).unwrap();
            assert_requested_allocations(&index);
            let estimate = index.memory.account.estimated_bytes();
            index
                .remove_batch_ids(&(0..256).collect::<Vec<_>>())
                .unwrap()
                .unwrap();
            assert_requested_allocations(&index);
            index.remove_batch_slice(&entries[256..]).unwrap();
            assert_requested_allocations(&index);
            assert_eq!(index.hash_to_values.read().nested_bytes, 0);
            index.clear().unwrap();
            assert_requested_allocations(&index);
            assert_eq!(index.memory.account.estimated_bytes(), estimate);
            let account = Arc::clone(index.memory_account().unwrap());
            index.close().unwrap();
            assert_requested_allocations(&index);
            drop(index);
            assert_eq!(account.requested_bytes(), 0);
            assert_eq!(account.estimated_bytes(), 0);
        }
    }

    #[test]
    fn hash_accounting_publishes_removed_key_on_uniqueness_error() {
        let index = memory_index(true, 0);
        let first = [Value::text(
            "a heap key removed before a uniqueness failure",
        )];
        let second = [Value::text("an existing key that rejects the replacement")];
        index.add(&first, 1, 0).unwrap();
        index.add(&second, 2, 0).unwrap();
        let before = index.memory.account.requested_bytes();
        assert!(index.add(&second, 1, 0).is_err());
        assert!(index.memory.account.requested_bytes() < before);
        assert_requested_allocations(&index);
    }

    #[test]
    fn hash_collision_group_removal_keeps_surviving_capacity_charged() {
        let index = memory_index(false, 0);
        let keys: Vec<_> = (0..8)
            .map(|id| {
                [Value::text(format!(
                    "a collision key retained on the heap {id}"
                ))]
            })
            .collect();
        index
            .mutate(|rows, storage| {
                for (id, key) in keys.iter().enumerate() {
                    rows.insert(id as i64, 7);
                    storage.add(7, key, id as i64);
                }
                Ok(())
            })
            .unwrap();
        assert_requested_allocations(&index);
        let bucket_capacity = index.hash_to_values.read().map[&7].capacity();
        index
            .mutate(|rows, storage| {
                rows.remove(0);
                storage.remove(7, &keys[0], 0);
                Ok(())
            })
            .unwrap();
        assert_requested_allocations(&index);
        assert_eq!(
            index.hash_to_values.read().map[&7].capacity(),
            bucket_capacity
        );
        index
            .remove_batch_ids(&(1..8).collect::<Vec<_>>())
            .unwrap()
            .unwrap();
        assert_requested_allocations(&index);
        assert_eq!(index.hash_to_values.read().nested_bytes, 0);
    }

    #[test]
    fn test_hash_index_basic() {
        let index = HashIndex::new(
            "idx_email".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Add some entries
        index
            .add(&[Value::Text("alice@example.com".into())], 1, 0)
            .unwrap();
        index
            .add(&[Value::Text("bob@example.com".into())], 2, 0)
            .unwrap();
        index
            .add(&[Value::Text("charlie@example.com".into())], 3, 0)
            .unwrap();

        // Find by exact match
        let results = index
            .find(&[Value::Text("alice@example.com".into())])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 1);

        let results = index
            .find(&[Value::Text("bob@example.com".into())])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 2);

        // Not found
        let results = index
            .find(&[Value::Text("nobody@example.com".into())])
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hash_index_unique_constraint() {
        let index = HashIndex::new(
            "idx_email_unique".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            true, // unique
            0,
        );

        // Add first entry
        index
            .add(&[Value::Text("alice@example.com".into())], 1, 0)
            .unwrap();

        // Try to add duplicate - should fail
        let result = index.add(&[Value::Text("alice@example.com".into())], 2, 0);
        assert!(result.is_err());

        // Same row_id update should succeed
        index
            .add(&[Value::Text("alice@example.com".into())], 1, 0)
            .unwrap();
    }

    #[test]
    fn test_hash_index_null_values() {
        let index = HashIndex::new(
            "idx_email_unique".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            true, // unique
            0,
        );

        // NULL values don't violate uniqueness
        index.add(&[Value::Null(DataType::Text)], 1, 0).unwrap();
        index.add(&[Value::Null(DataType::Text)], 2, 0).unwrap(); // Should succeed

        // Find NULL
        let results = index.find(&[Value::Null(DataType::Text)]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hash_index_remove() {
        let index = HashIndex::new(
            "idx_email".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        index
            .add(&[Value::Text("alice@example.com".into())], 1, 0)
            .unwrap();
        index
            .add(&[Value::Text("bob@example.com".into())], 2, 0)
            .unwrap();

        // Remove alice
        index
            .remove(&[Value::Text("alice@example.com".into())], 1, 0)
            .unwrap();

        // Alice should be gone
        let results = index
            .find(&[Value::Text("alice@example.com".into())])
            .unwrap();
        assert!(results.is_empty());

        // Bob should still be there
        let results = index
            .find(&[Value::Text("bob@example.com".into())])
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hash_index_update() {
        let index = HashIndex::new(
            "idx_email".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Add initial value
        index
            .add(&[Value::Text("old@example.com".into())], 1, 0)
            .unwrap();

        // Update to new value (same row_id, different value)
        index
            .add(&[Value::Text("new@example.com".into())], 1, 0)
            .unwrap();

        // Old value should be gone
        let results = index
            .find(&[Value::Text("old@example.com".into())])
            .unwrap();
        assert!(results.is_empty());

        // New value should be found
        let results = index
            .find(&[Value::Text("new@example.com".into())])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 1);
    }

    #[test]
    fn test_hash_index_range_not_supported() {
        let index = HashIndex::new(
            "idx_email".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
            0,
        );

        // Range query should fail
        let result = index.find_range(
            &[Value::Text("a".into())],
            &[Value::Text("z".into())],
            true,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_index_multi_column() {
        let index = HashIndex::new(
            "idx_name_email".to_string(),
            "users".to_string(),
            vec!["name".to_string(), "email".to_string()],
            vec![1, 2],
            vec![DataType::Text, DataType::Text],
            false,
            0,
        );

        index
            .add(
                &[
                    Value::Text("Alice".into()),
                    Value::Text("alice@example.com".into()),
                ],
                1,
                0,
            )
            .unwrap();
        index
            .add(
                &[
                    Value::Text("Bob".into()),
                    Value::Text("bob@example.com".into()),
                ],
                2,
                0,
            )
            .unwrap();

        // Find by both columns
        let results = index
            .find(&[
                Value::Text("Alice".into()),
                Value::Text("alice@example.com".into()),
            ])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 1);

        // Partial match not supported
        let result = index.find(&[Value::Text("Alice".into())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_index_duplicate_values() {
        let index = HashIndex::new(
            "idx_status".to_string(),
            "orders".to_string(),
            vec!["status".to_string()],
            vec![1],
            vec![DataType::Text],
            false, // not unique
            0,
        );

        // Multiple rows with same value
        index.add(&[Value::Text("pending".into())], 1, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 2, 0).unwrap();
        index.add(&[Value::Text("pending".into())], 3, 0).unwrap();
        index.add(&[Value::Text("shipped".into())], 4, 0).unwrap();

        let results = index.find(&[Value::Text("pending".into())]).unwrap();
        assert_eq!(results.len(), 3);

        let results = index.find(&[Value::Text("shipped".into())]).unwrap();
        assert_eq!(results.len(), 1);
    }
}
