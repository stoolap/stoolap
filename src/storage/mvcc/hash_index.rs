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
//! ### Triple Lock Pattern for Write Operations
//! The `add` and `remove` methods acquire three RwLock write locks simultaneously:
//! - `hash_to_rows`: Maps hash -> row IDs for quick lookup
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
//! contention characteristics. DashMap could help for truly concurrent writes but
//! would complicate atomicity across the three maps.

use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::RwLock;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::common::CompactArc;
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::{ComparisonExpr, Expression, InListExpr};
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

/// Compute hash for a slice of CompactArc<Value> using ahash
/// Uses same fixed seeds as hash_values for consistency
#[inline]
fn hash_arc_values(values: &[CompactArc<Value>]) -> u64 {
    let hasher_builder =
        ahash::RandomState::with_seeds(HASH_SEEDS[0], HASH_SEEDS[1], HASH_SEEDS[2], HASH_SEEDS[3]);
    let mut hasher = hasher_builder.build_hasher();
    for v in values {
        v.as_ref().hash(&mut hasher);
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

    /// Hash -> row IDs mapping
    /// Uses u64 hash key to avoid storing full values (memory efficient)
    /// SmallVec<[i64; 4]> avoids heap allocation for 1-4 matches (common case)
    hash_to_rows: RwLock<FxHashMap<u64, SmallVec<[i64; 4]>>>,

    /// Row ID -> hash mapping for efficient removal
    /// Uses FxHashMap for O(1) lookups (optimized for integer keys)
    row_to_hash: RwLock<FxHashMap<i64, u64>>,

    /// Full values storage for hash collision handling and unique constraint checking
    /// Maps hash -> (values as CompactArc<Value>, row_ids) for collision resolution
    /// Uses CompactArc<Value> to share references with ValueArena (8 bytes per value)
    #[allow(clippy::type_complexity)]
    hash_to_values: RwLock<FxHashMap<u64, Vec<(Vec<CompactArc<Value>>, SmallVec<[i64; 4]>)>>>,
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
            hash_to_rows: RwLock::new(FxHashMap::default()),
            row_to_hash: RwLock::new(FxHashMap::default()),
            hash_to_values: RwLock::new(FxHashMap::default()),
        }
    }

    /// Check uniqueness constraint
    #[allow(clippy::type_complexity)]
    fn check_unique_constraint(
        &self,
        values: &[Value],
        row_id: i64,
        hash: u64,
        hash_to_values: &FxHashMap<u64, Vec<(Vec<CompactArc<Value>>, SmallVec<[i64; 4]>)>>,
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

    /// Helper to compare stored CompactArc<Value> with input &[CompactArc<Value>]
    /// Specialized unrolling for common cases (1-4 columns): ~5% faster than zip().all()
    #[inline]
    fn arc_values_match(stored: &[CompactArc<Value>], input: &[CompactArc<Value>]) -> bool {
        #[inline]
        fn eq(s: &CompactArc<Value>, i: &CompactArc<Value>) -> bool {
            CompactArc::ptr_eq(s, i) || s.as_ref() == i.as_ref()
        }
        if stored.len() != input.len() {
            return false;
        }
        match stored.len() {
            0 => true,
            1 => eq(&stored[0], &input[0]),
            2 => eq(&stored[0], &input[0]) && eq(&stored[1], &input[1]),
            3 => {
                eq(&stored[0], &input[0]) && eq(&stored[1], &input[1]) && eq(&stored[2], &input[2])
            }
            4 => {
                eq(&stored[0], &input[0])
                    && eq(&stored[1], &input[1])
                    && eq(&stored[2], &input[2])
                    && eq(&stored[3], &input[3])
            }
            _ => stored
                .iter()
                .zip(input.iter())
                .all(|(s, i)| CompactArc::ptr_eq(s, i) || s.as_ref() == i.as_ref()),
        }
    }

    /// Check uniqueness constraint for Arc values
    #[allow(clippy::type_complexity)]
    fn check_unique_constraint_arc(
        &self,
        values: &[CompactArc<Value>],
        row_id: i64,
        hash: u64,
        hash_to_values: &FxHashMap<u64, Vec<(Vec<CompactArc<Value>>, SmallVec<[i64; 4]>)>>,
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
                // Compare CompactArc<Value> contents
                if Self::arc_values_match(stored_values, values) && !row_ids.is_empty() {
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
}

impl Index for HashIndex {
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

        // Acquire all write locks to ensure atomic operation
        let mut hash_to_rows = self.hash_to_rows.write().unwrap();
        let mut row_to_hash = self.row_to_hash.write().unwrap();
        let mut hash_to_values = self.hash_to_values.write().unwrap();

        // Check if row already exists (for updates)
        if let Some(&old_hash) = row_to_hash.get(&row_id) {
            if old_hash == hash {
                // Same hash - might be same values, check if we need to update
                // (could be hash collision with different values)
                return Ok(());
            }

            // Different hash - remove old entry
            if let Some(rows) = hash_to_rows.get_mut(&old_hash) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    hash_to_rows.remove(&old_hash);
                }
            }

            // Remove from values storage
            if let Some(entries) = hash_to_values.get_mut(&old_hash) {
                for (_, row_ids) in entries.iter_mut() {
                    row_ids.retain(|id| *id != row_id);
                }
                entries.retain(|(_, row_ids)| !row_ids.is_empty());
                if entries.is_empty() {
                    hash_to_values.remove(&old_hash);
                }
            }
        }

        // Check uniqueness constraint
        self.check_unique_constraint(values, row_id, hash, &hash_to_values)?;

        // Add to hash_to_rows (insert sorted for O(N+M) merge operations)
        let rows = hash_to_rows.entry(hash).or_default();
        if let Err(pos) = rows.binary_search(&row_id) {
            rows.insert(pos, row_id);
        }

        // Add to row_to_hash
        row_to_hash.insert(row_id, hash);

        // Add to hash_to_values for collision handling
        // Use CompactArc<Value> via arena for memory efficiency
        let entries = hash_to_values.entry(hash).or_default();
        let mut found = false;
        for (stored_values, row_ids) in entries.iter_mut() {
            // Compare CompactArc<Value> contents with input values
            if Self::values_match(stored_values, values) {
                // Insert sorted for O(N+M) merge operations
                if let Err(pos) = row_ids.binary_search(&row_id) {
                    row_ids.insert(pos, row_id);
                }
                found = true;
                break;
            }
        }
        if !found {
            // Wrap values in Arc for O(1) cloning
            let arc_values: Vec<CompactArc<Value>> =
                values.iter().map(|v| CompactArc::new(v.clone())).collect();
            let mut row_ids = SmallVec::new();
            row_ids.push(row_id); // First element, already sorted
            entries.push((arc_values, row_ids));
        }

        Ok(())
    }

    fn add_arc(&self, values: &[CompactArc<Value>], row_id: i64, _ref_id: i64) -> Result<()> {
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

        let hash = hash_arc_values(values);

        // Acquire all write locks to ensure atomic operation
        let mut hash_to_rows = self.hash_to_rows.write().unwrap();
        let mut row_to_hash = self.row_to_hash.write().unwrap();
        let mut hash_to_values = self.hash_to_values.write().unwrap();

        // Check if row already exists (for updates)
        if let Some(&old_hash) = row_to_hash.get(&row_id) {
            if old_hash == hash {
                // Same hash - might be same values, check if we need to update
                // (could be hash collision with different values)
                return Ok(());
            }

            // Different hash - remove old entry
            if let Some(rows) = hash_to_rows.get_mut(&old_hash) {
                rows.retain(|id| *id != row_id);
                if rows.is_empty() {
                    hash_to_rows.remove(&old_hash);
                }
            }

            // Remove from values storage
            if let Some(entries) = hash_to_values.get_mut(&old_hash) {
                for (_, row_ids) in entries.iter_mut() {
                    row_ids.retain(|id| *id != row_id);
                }
                entries.retain(|(_, row_ids)| !row_ids.is_empty());
                if entries.is_empty() {
                    hash_to_values.remove(&old_hash);
                }
            }
        }

        // Check uniqueness constraint
        self.check_unique_constraint_arc(values, row_id, hash, &hash_to_values)?;

        // Add to hash_to_rows (insert sorted for O(N+M) merge operations)
        let rows = hash_to_rows.entry(hash).or_default();
        if let Err(pos) = rows.binary_search(&row_id) {
            rows.insert(pos, row_id);
        }

        // Add to row_to_hash
        row_to_hash.insert(row_id, hash);

        // Add to hash_to_values for collision handling
        // Use Arc::clone - O(1), no value cloning!
        let entries = hash_to_values.entry(hash).or_default();
        let mut found = false;
        for (stored_values, row_ids) in entries.iter_mut() {
            // Compare Arc values
            if Self::arc_values_match(stored_values, values) {
                // Insert sorted for O(N+M) merge operations
                if let Err(pos) = row_ids.binary_search(&row_id) {
                    row_ids.insert(pos, row_id);
                }
                found = true;
                break;
            }
        }
        if !found {
            // Clone Arc references - O(1) per value, no value cloning!
            let arc_values: Vec<CompactArc<Value>> = values.iter().map(CompactArc::clone).collect();
            let mut row_ids = SmallVec::new();
            row_ids.push(row_id); // First element, already sorted
            entries.push((arc_values, row_ids));
        }

        Ok(())
    }

    fn add_batch(&self, entries: &FxHashMap<i64, Vec<Value>>) -> Result<()> {
        for (&row_id, values) in entries {
            self.add(values, row_id, 0)?;
        }
        Ok(())
    }

    fn remove(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        let hash = hash_values(values);

        let mut hash_to_rows = self.hash_to_rows.write().unwrap();
        let mut row_to_hash = self.row_to_hash.write().unwrap();
        let mut hash_to_values = self.hash_to_values.write().unwrap();

        // Remove from hash_to_rows (row_ids are sorted, use binary search)
        if let Some(rows) = hash_to_rows.get_mut(&hash) {
            if let Ok(pos) = rows.binary_search(&row_id) {
                rows.remove(pos);
            }
            if rows.is_empty() {
                hash_to_rows.remove(&hash);
            }
        }

        // Remove from row_to_hash
        row_to_hash.remove(&row_id);

        // Remove from hash_to_values (row_ids are sorted, use binary search)
        if let Some(entries) = hash_to_values.get_mut(&hash) {
            for (stored_values, row_ids) in entries.iter_mut() {
                // Compare CompactArc<Value> contents with input values
                if Self::values_match(stored_values, values) {
                    if let Ok(pos) = row_ids.binary_search(&row_id) {
                        row_ids.remove(pos);
                    }
                    break;
                }
            }
            entries.retain(|(_, row_ids)| !row_ids.is_empty());
            if entries.is_empty() {
                hash_to_values.remove(&hash);
            }
        }

        Ok(())
    }

    fn remove_arc(&self, values: &[CompactArc<Value>], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }

        let hash = hash_arc_values(values);

        let mut hash_to_rows = self.hash_to_rows.write().unwrap();
        let mut row_to_hash = self.row_to_hash.write().unwrap();
        let mut hash_to_values = self.hash_to_values.write().unwrap();

        // Remove from hash_to_rows (row_ids are sorted, use binary search)
        if let Some(rows) = hash_to_rows.get_mut(&hash) {
            if let Ok(pos) = rows.binary_search(&row_id) {
                rows.remove(pos);
            }
            if rows.is_empty() {
                hash_to_rows.remove(&hash);
            }
        }

        // Remove from row_to_hash
        row_to_hash.remove(&row_id);

        // Remove from hash_to_values (row_ids are sorted, use binary search)
        if let Some(entries) = hash_to_values.get_mut(&hash) {
            for (stored_values, row_ids) in entries.iter_mut() {
                // Compare CompactArc<Value> contents with input Arc values
                if Self::arc_values_match(stored_values, values) {
                    if let Ok(pos) = row_ids.binary_search(&row_id) {
                        row_ids.remove(pos);
                    }
                    break;
                }
            }
            entries.retain(|(_, row_ids)| !row_ids.is_empty());
            if entries.is_empty() {
                hash_to_values.remove(&hash);
            }
        }

        Ok(())
    }

    fn remove_batch(&self, entries: &FxHashMap<i64, Vec<Value>>) -> Result<()> {
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

        // First check hash_to_rows for quick path (no collisions)
        let hash_to_values = self.hash_to_values.read().unwrap();

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
        let hash_to_values = self.hash_to_values.read().unwrap();

        if let Some(entries) = hash_to_values.get(&hash) {
            // Handle hash collisions by checking actual values (CompactArc<Value>)
            for (stored_values, row_ids) in entries {
                if Self::values_match(stored_values, values) {
                    // SmallVec can be iterated efficiently
                    buffer.extend(row_ids.iter().copied());
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
        let hash_to_values = self.hash_to_values.read().unwrap();

        for value in value_list {
            // Hash without cloning - pass reference to slice
            let hash = hash_values(std::slice::from_ref(value));
            if let Some(entries) = hash_to_values.get(&hash) {
                for (stored_values, row_ids) in entries {
                    // Compare CompactArc<Value> with input value
                    if stored_values.len() == 1 && stored_values[0].as_ref() == value {
                        buffer.extend(row_ids.iter().copied());
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
        let hash_to_values = self.hash_to_values.read().unwrap();
        let mut results = RowIdVec::new();

        for entries in hash_to_values.values() {
            for (_values, row_ids) in entries {
                results.extend(row_ids.iter().copied());
            }
        }

        results
    }

    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let hash_to_values = self.hash_to_values.read().unwrap();
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

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, AtomicOrdering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_index_basic() {
        let index = HashIndex::new(
            "idx_email".to_string(),
            "users".to_string(),
            vec!["email".to_string()],
            vec![1],
            vec![DataType::Text],
            false,
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
