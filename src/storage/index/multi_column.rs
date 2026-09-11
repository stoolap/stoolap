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
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::common::{CompactArc, CompactVec, I64Map};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;
use crate::storage::index::memory::{
    btree_node_bytes, hash_table_bytes, value_bytes, IndexMemory, IndexMemoryOwner,
};
use crate::storage::mvcc::read_memory::PayloadCharge;
use crate::storage::traits::Index;

// ============================================================================
// CompositeKey - Ordered key for BTreeMap
// ============================================================================

/// Composite key for BTreeMap ordering
/// Wraps Vec<Value> with proper Ord implementation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeKey(pub Vec<Value>);

impl CompositeKey {
    fn heap_bytes(&self) -> u128 {
        (self.0.capacity() * std::mem::size_of::<Value>()) as u128
            + self
                .0
                .iter()
                .map(|value| value.heap_bytes() as u128)
                .sum::<u128>()
    }
}

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
    sorted_values: RwLock<CompositeTree>,
    btree_built: AtomicBool,

    /// Hash index for exact lookups (full key) - always maintained
    value_to_rows: RwLock<CompositeHash>,

    /// Prefix indexes: LAZY built on first partial query
    /// Index 0 = first column, Index 1 = first two columns, etc.
    prefix_indexes: Vec<RwLock<CompositeHash>>,
    prefix_built: Vec<AtomicBool>,

    /// Reverse mapping for removal - uses Vec<CompactArc<Value>> for memory efficiency
    /// Arc references are shared with ValueArena (8 bytes per value)
    row_to_key: RwLock<CompositeReverse>,

    /// Lazily built prefix-group row orders. Writers touch these last;
    /// a build holds the write lock while reading its prefix/reverse inputs.
    walk_orders: RwLock<WalkOrders>,
    memory: IndexMemoryOwner,
}

fn row_ids_bytes(rows: &CompactVec<i64>) -> u128 {
    (rows.capacity() * std::mem::size_of::<i64>()) as u128
}

#[derive(Default)]
struct CompositeHash {
    map: FxHashMap<CompositeKey, CompactVec<i64>>,
    nested_bytes: u128,
    capacity_high_water: usize,
}

impl CompositeHash {
    fn estimated_bytes(&self) -> u128 {
        hash_table_bytes::<CompositeKey, CompactVec<i64>>(self.capacity_high_water)
    }

    fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
    }

    fn add(&mut self, key: CompositeKey, row_id: i64) {
        let rows = match self.map.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.nested_bytes += entry.key().heap_bytes();
                entry.insert(CompactVec::new())
            }
        };
        if let Err(pos) = rows.binary_search(&row_id) {
            let before = row_ids_bytes(rows);
            rows.insert(pos, row_id);
            self.nested_bytes = self.nested_bytes - before + row_ids_bytes(rows);
        }
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
    }

    fn insert(&mut self, key: CompositeKey, rows: CompactVec<i64>) {
        self.nested_bytes += row_ids_bytes(&rows);
        match self.map.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                self.nested_bytes -= row_ids_bytes(entry.get());
                entry.insert(rows);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.nested_bytes += entry.key().heap_bytes();
                entry.insert(rows);
            }
        }
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
    }

    fn update_rows(&mut self, key: &CompositeKey, update: impl FnOnce(&mut CompactVec<i64>)) {
        let Some(rows) = self.map.get_mut(key) else {
            return;
        };
        let before = row_ids_bytes(rows);
        update(rows);
        self.nested_bytes = self.nested_bytes - before + row_ids_bytes(rows);
        if rows.is_empty() {
            if let Some((stored, rows)) = self.map.remove_entry(key) {
                self.nested_bytes -= stored.heap_bytes() + row_ids_bytes(&rows);
            }
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.nested_bytes = 0;
    }
}

impl std::ops::Deref for CompositeHash {
    type Target = FxHashMap<CompositeKey, CompactVec<i64>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

#[derive(Default)]
struct CompositeTree {
    map: BTreeMap<CompositeKey, CompactVec<i64>>,
    nested_bytes: u128,
    has_nodes: bool,
}

impl CompositeTree {
    fn estimated_bytes(&self) -> u128 {
        if self.has_nodes {
            btree_node_bytes::<CompositeKey, CompactVec<i64>>(self.map.len())
        } else {
            0
        }
    }

    fn add(&mut self, key: CompositeKey, row_id: i64) {
        let rows = match self.map.entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.nested_bytes += entry.key().heap_bytes();
                self.has_nodes = true;
                entry.insert(CompactVec::new())
            }
        };
        if let Err(pos) = rows.binary_search(&row_id) {
            let before = row_ids_bytes(rows);
            rows.insert(pos, row_id);
            self.nested_bytes = self.nested_bytes - before + row_ids_bytes(rows);
        }
    }

    fn insert(&mut self, key: CompositeKey, rows: CompactVec<i64>) {
        self.nested_bytes += row_ids_bytes(&rows);
        match self.map.entry(key) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                self.nested_bytes -= row_ids_bytes(entry.get());
                entry.insert(rows);
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.nested_bytes += entry.key().heap_bytes();
                self.has_nodes = true;
                entry.insert(rows);
            }
        }
    }

    fn update_rows(&mut self, key: &CompositeKey, update: impl FnOnce(&mut CompactVec<i64>)) {
        let Some(rows) = self.map.get_mut(key) else {
            return;
        };
        let before = row_ids_bytes(rows);
        update(rows);
        self.nested_bytes = self.nested_bytes - before + row_ids_bytes(rows);
        if rows.is_empty() {
            if let Some((stored, rows)) = self.map.remove_entry(key) {
                self.nested_bytes -= stored.heap_bytes() + row_ids_bytes(&rows);
            }
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.nested_bytes = 0;
        self.has_nodes = false;
    }
}

impl std::ops::Deref for CompositeTree {
    type Target = BTreeMap<CompositeKey, CompactVec<i64>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

struct CompositeReverse {
    map: I64Map<Vec<CompactArc<Value>>>,
    payload_bytes: u128,
}

impl CompositeReverse {
    fn key_bytes(values: &Vec<CompactArc<Value>>) -> u128 {
        (values.capacity() * std::mem::size_of::<CompactArc<Value>>()) as u128
            + values.iter().map(value_bytes).sum::<u128>()
    }

    fn requested_bytes(&self) -> u128 {
        self.map.allocation_bytes() as u128 + self.payload_bytes
    }

    fn insert(&mut self, row_id: i64, values: Vec<CompactArc<Value>>) {
        self.payload_bytes += Self::key_bytes(&values);
        if let Some(previous) = self.map.insert(row_id, values) {
            self.payload_bytes -= Self::key_bytes(&previous);
        }
    }

    fn remove(&mut self, row_id: i64) {
        if let Some(previous) = self.map.remove(row_id) {
            self.payload_bytes -= Self::key_bytes(&previous);
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.payload_bytes = 0;
    }
}

impl std::ops::Deref for CompositeReverse {
    type Target = I64Map<Vec<CompactArc<Value>>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

#[derive(Default)]
struct WalkOrders {
    map: FxHashMap<CompositeKey, BTreeSet<(Value, i64)>>,
    nested_bytes: u128,
    node_units: u128,
    capacity_high_water: usize,
}

impl WalkOrders {
    fn estimated_bytes(&self) -> u128 {
        hash_table_bytes::<CompositeKey, BTreeSet<(Value, i64)>>(self.capacity_high_water)
            + self.node_units * btree_node_bytes::<(Value, i64), ()>(0)
    }

    fn update(&mut self, key: &CompositeKey, value: &Value, row_id: i64, insert: bool) {
        let Some(order) = self.map.get_mut(key) else {
            return;
        };
        let before = 1 + order.len() / 5;
        if insert {
            let value = value.clone();
            let bytes = value.heap_bytes() as u128;
            if order.insert((value, row_id)) {
                self.nested_bytes += bytes;
            }
        } else if let Some((stored, _)) = order.take(&(value.clone(), row_id)) {
            self.nested_bytes -= stored.heap_bytes() as u128;
        }
        self.node_units = self.node_units - before as u128 + (1 + order.len() / 5) as u128;
    }

    fn insert(&mut self, key: CompositeKey, order: BTreeSet<(Value, i64)>, value_bytes: u128) {
        debug_assert!(!self.map.contains_key(&key));
        self.nested_bytes += key.heap_bytes() + value_bytes;
        self.node_units += (1 + order.len() / 5) as u128;
        self.map.insert(key, order);
        self.capacity_high_water = self.capacity_high_water.max(self.map.capacity());
    }

    fn clear(&mut self) {
        self.map.clear();
        self.nested_bytes = 0;
        self.node_units = 0;
    }
}

impl std::ops::Deref for WalkOrders {
    type Target = FxHashMap<CompositeKey, BTreeSet<(Value, i64)>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
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
            prefix_indexes.push(RwLock::new(CompositeHash::default()));
            prefix_built.push(AtomicBool::new(false));
        }

        let map = if expected_rows > 0 {
            FxHashMap::with_capacity_and_hasher(expected_rows, Default::default())
        } else {
            FxHashMap::default()
        };
        let value_to_rows = CompositeHash {
            capacity_high_water: map.capacity(),
            map,
            nested_bytes: 0,
        };
        let row_to_key = CompositeReverse {
            map: if expected_rows > 0 {
                I64Map::with_capacity(expected_rows)
            } else {
                I64Map::new()
            },
            payload_bytes: 0,
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
            + (prefix_indexes.capacity() * std::mem::size_of::<RwLock<CompositeHash>>()) as u128
            + (prefix_built.capacity() * std::mem::size_of::<AtomicBool>()) as u128
            + row_to_key.requested_bytes();
        let memory = IndexMemoryOwner::new::<Self>(requested, value_to_rows.estimated_bytes());
        Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            closed: AtomicBool::new(false),
            sorted_values: RwLock::new(CompositeTree::default()),
            btree_built: AtomicBool::new(false),
            value_to_rows: RwLock::new(value_to_rows),
            prefix_indexes,
            prefix_built,
            row_to_key: RwLock::new(row_to_key),
            walk_orders: RwLock::new(WalkOrders::default()),
            memory,
        }
    }

    fn mutate_main<T>(
        &self,
        mutate: impl FnOnce(&mut CompositeHash, &mut CompositeReverse) -> Result<T>,
    ) -> Result<T> {
        let mut values = self.value_to_rows.write();
        let mut reverse = self.row_to_key.write();
        let before = values.nested_bytes + reverse.requested_bytes();
        let estimated_before = values.estimated_bytes();
        let result = mutate(&mut values, &mut reverse);
        self.memory
            .account
            .resize(before, values.nested_bytes + reverse.requested_bytes());
        self.memory
            .account
            .resize_estimate(estimated_before, values.estimated_bytes());
        result
    }

    fn mutate_tree(&self, mutate: impl FnOnce(&mut CompositeTree)) {
        let mut tree = self.sorted_values.write();
        let before = tree.nested_bytes;
        let estimated_before = tree.estimated_bytes();
        mutate(&mut tree);
        self.memory.account.resize(before, tree.nested_bytes);
        self.memory
            .account
            .resize_estimate(estimated_before, tree.estimated_bytes());
    }

    fn mutate_prefix(&self, index: usize, mutate: impl FnOnce(&mut CompositeHash)) {
        let mut prefix = self.prefix_indexes[index].write();
        let before = prefix.nested_bytes;
        let estimated_before = prefix.estimated_bytes();
        mutate(&mut prefix);
        self.memory.account.resize(before, prefix.nested_bytes);
        self.memory
            .account
            .resize_estimate(estimated_before, prefix.estimated_bytes());
    }

    fn mutate_orders(&self, mutate: impl FnOnce(&mut WalkOrders)) {
        let mut orders = self.walk_orders.write();
        let before = orders.nested_bytes;
        let estimated_before = orders.estimated_bytes();
        mutate(&mut orders);
        self.memory.account.resize(before, orders.nested_bytes);
        self.memory
            .account
            .resize_estimate(estimated_before, orders.estimated_bytes());
    }

    /// A batch this large drops the built walk orders; they come back on the
    /// next walk, cheaper than updating them row by row
    const WALK_ORDER_BATCH_DROP: usize = 1024;

    /// Rows removed per lock acquisition in a batch removal
    const REMOVE_CHUNK_ROWS: usize = 8192;

    /// Enter `row_id` into the built order of every prefix group of `values`
    fn order_insert(&self, values: &[Value], row_id: i64) {
        self.mutate_orders(|orders| {
            for prefix_len in 1..values.len() {
                orders.update(
                    &CompositeKey(values[..prefix_len].to_vec()),
                    &values[prefix_len],
                    row_id,
                    true,
                );
            }
        });
    }

    fn order_remove(&self, values: &[Value], row_id: i64) {
        self.mutate_orders(|orders| {
            for prefix_len in 1..values.len() {
                orders.update(
                    &CompositeKey(values[..prefix_len].to_vec()),
                    &values[prefix_len],
                    row_id,
                    false,
                );
            }
        });
    }

    fn order_remove_arcs(&self, values: &[CompactArc<Value>], row_id: i64) {
        let owned: Vec<Value> = values.iter().map(|v| (**v).clone()).collect();
        self.order_remove(&owned, row_id);
    }

    fn orders_built(&self) -> bool {
        !self.walk_orders.read().is_empty()
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
        self.mutate_tree(|sorted_values| {
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
        });
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
        self.mutate_prefix(idx, |prefix_index| {
            // Double-check after acquiring write lock
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                return;
            }

            // Build prefix index from row_to_key (read lock prevents concurrent inserts).
            // Gather each key's row ids unsorted, then sort once: a sorted insert per
            // row is quadratic in the rows per key.
            let mut gathered: FxHashMap<CompositeKey, Vec<i64>> = FxHashMap::default();
            for (row_id, arc_values) in row_to_key.iter() {
                if arc_values.len() >= prefix_len {
                    // Dereference CompactArc<Value> to create CompositeKey for prefix
                    let prefix_key = CompositeKey(
                        arc_values[..prefix_len]
                            .iter()
                            .map(|a| (**a).clone())
                            .collect(),
                    );
                    gathered.entry(prefix_key).or_default().push(row_id);
                }
            }
            for (prefix_key, mut rows) in gathered {
                rows.sort_unstable();
                rows.dedup();
                prefix_index.insert(prefix_key, CompactVec::from_vec(rows));
            }

            // Set flag before releasing locks - subsequent inserts will see prefix_built=true
            self.prefix_built[idx].store(true, AtomicOrdering::Release);
        });
    }

    /// The row ids of a removal batch grouped by the first `key_len` values,
    /// each group sorted for binary search.
    fn group_removed_by_key(
        entries: &[(i64, &[Value])],
        key_len: usize,
    ) -> FxHashMap<CompositeKey, Vec<i64>> {
        let mut grouped: FxHashMap<CompositeKey, Vec<i64>> = FxHashMap::default();
        for &(row_id, values) in entries {
            if values.len() >= key_len {
                grouped
                    .entry(CompositeKey(values[..key_len].to_vec()))
                    .or_default()
                    .push(row_id);
            }
        }
        for ids in grouped.values_mut() {
            ids.sort_unstable();
        }
        grouped
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

        let key = CompositeKey(values.to_vec());

        // Track old key if this is an update (for BTree/prefix cleanup after releasing locks)
        let mut cleanup_charge = PayloadCharge::unshared(0);
        let mut old_key_for_cleanup: Option<Vec<CompactArc<Value>>> = None;

        let Some(btree_needs_update) = self.mutate_main(|value_to_rows, row_to_key| {
            // Check if row already exists with different key (for updates)
            // Now safe to do atomically since we hold both write locks
            if let Some(existing_arc_values) = row_to_key.get(row_id) {
                // Compare CompactArc<Value> contents with new values using optimized helper
                if Self::values_match(existing_arc_values, values) {
                    // Same key, nothing to do
                    return Ok(None);
                }

                // Different key - save for BTree/prefix cleanup AFTER releasing locks
                let old_key = existing_arc_values.clone();
                cleanup_charge.add(CompositeReverse::key_bytes(&old_key));
                old_key_for_cleanup = Some(old_key);

                // Create CompositeKey from existing Arc values for removal
                let existing_key =
                    CompositeKey(existing_arc_values.iter().map(|a| (**a).clone()).collect());

                // Remove old entry from value_to_rows
                value_to_rows.update_rows(&existing_key, |rows| {
                    rows.retain(|id| *id != row_id);
                });
                // Note: row_to_key will be overwritten below, no need to remove here
            }

            // Check uniqueness while holding write lock (atomic check + insert)
            self.check_unique_constraint_locked(&key, row_id, value_to_rows)?;

            // Check if BTree needs update before we move key
            let btree_needs_update = self.btree_built.load(AtomicOrdering::Acquire);

            // Wrap values in Arc for O(1) cloning in row_to_key
            let arc_values: Vec<CompactArc<Value>> =
                values.iter().map(|v| CompactArc::new(v.clone())).collect();

            // Add to hash index (for exact lookups) - ALWAYS maintained
            // Clone key once for hash index, keep original for BTree
            // Insert in sorted order for O(N+M) merge operations
            let key_for_hash = key.clone();
            value_to_rows.add(key_for_hash, row_id);

            // Store CompactArc<Value> references in row_to_key (memory efficient)
            row_to_key.insert(row_id, arc_values);

            Ok(Some(btree_needs_update))
        })?
        else {
            return Ok(());
        };

        // Update BTree only if it was already built
        if btree_needs_update {
            self.mutate_tree(|sorted_values| {
                // First remove old entry if this was an update
                if let Some(ref old_arc_values) = old_key_for_cleanup {
                    let existing_key =
                        CompositeKey(old_arc_values.iter().map(|a| (**a).clone()).collect());
                    sorted_values.update_rows(&existing_key, |rows| {
                        rows.retain(|id| *id != row_id);
                    });
                }

                // Then add new entry
                sorted_values.add(key, row_id);
            });
        }

        // Update prefix indexes only if they were already built
        // Insert in sorted order for O(N+M) merge operations
        for prefix_len in 1..num_cols {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                self.mutate_prefix(idx, |prefix_index| {
                    // First remove old entry if this was an update
                    if let Some(ref old_arc_values) = old_key_for_cleanup {
                        if old_arc_values.len() >= prefix_len {
                            let old_prefix_key = CompositeKey(
                                old_arc_values[..prefix_len]
                                    .iter()
                                    .map(|a| (**a).clone())
                                    .collect(),
                            );
                            prefix_index.update_rows(&old_prefix_key, |rows| {
                                rows.retain(|id| *id != row_id);
                            });
                        }
                    }

                    // Then add new entry
                    let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                    prefix_index.add(prefix_key, row_id);
                });
            }
        }

        if self.orders_built() {
            if let Some(ref old_arc_values) = old_key_for_cleanup {
                self.order_remove_arcs(old_arc_values, row_id);
            }
            self.order_insert(values, row_id);
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
        self.mutate_main(|value_to_rows, row_to_key| {
            // Remove from hash index (row_ids are sorted, use binary search)
            value_to_rows.update_rows(&key, |rows| {
                if let Ok(pos) = rows.binary_search(&row_id) {
                    rows.remove(pos);
                }
            });

            // Remove reverse mapping - ALWAYS maintained
            row_to_key.remove(row_id);
            Ok(())
        })?;

        // Only update BTree if it was built (row_ids are sorted, use binary search)
        if self.btree_built.load(AtomicOrdering::Acquire) {
            self.mutate_tree(|sorted_values| {
                sorted_values.update_rows(&key, |rows| {
                    if let Ok(pos) = rows.binary_search(&row_id) {
                        rows.remove(pos);
                    }
                });
            });
        }

        // Only update prefix indexes if they were built (row_ids are sorted, use binary search)
        for prefix_len in 1..values.len() {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                self.mutate_prefix(idx, |prefix_index| {
                    prefix_index.update_rows(&prefix_key, |rows| {
                        if let Ok(pos) = rows.binary_search(&row_id) {
                            rows.remove(pos);
                        }
                    });
                });
            }
        }

        if self.orders_built() {
            self.order_remove(values, row_id);
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

        let mut cleanup_charge = PayloadCharge::unshared(0);
        let mut updates_to_old_key: Vec<(i64, Vec<CompactArc<Value>>)> = Vec::new();
        let btree_needs_update = self.mutate_main(|value_to_rows, row_to_key| {
            // Reserve capacity to reduce reallocations
            value_to_rows.reserve(entries.len());
            row_to_key.map.reserve(entries.len());

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

            // Now add all entries
            let mut cleanup_bytes = 0u128;
            for &(row_id, values) in entries {
                let key = CompositeKey(values.to_vec());

                // Wrap values in Arc for O(1) cloning in row_to_key
                let arc_values: Vec<CompactArc<Value>> =
                    values.iter().map(|v| CompactArc::new(v.clone())).collect();

                // Check if row already exists with different key (for updates)
                if let Some(existing_arc_values) = row_to_key.get(row_id) {
                    if !Self::values_match(existing_arc_values, values) {
                        // Different key - collect for BTree/prefix removal later
                        let old_key = existing_arc_values.clone();
                        cleanup_bytes += CompositeReverse::key_bytes(&old_key);
                        updates_to_old_key.push((row_id, old_key));

                        // Remove old entry from value_to_rows
                        let existing_key = CompositeKey(
                            existing_arc_values.iter().map(|a| (**a).clone()).collect(),
                        );

                        value_to_rows.update_rows(&existing_key, |rows| {
                            rows.retain(|id| *id != row_id);
                        });
                    }
                }

                // Add to hash index - insert in sorted order
                value_to_rows.add(key, row_id);

                // Store Arc references in row_to_key
                row_to_key.insert(row_id, arc_values);
            }

            cleanup_charge.add(
                cleanup_bytes
                    + (updates_to_old_key.capacity()
                        * std::mem::size_of::<(i64, Vec<CompactArc<Value>>)>())
                        as u128,
            );
            Ok(btree_needs_update)
        })?;

        // Update BTree only if it was already built
        if btree_needs_update {
            self.mutate_tree(|sorted_values| {
                // First remove old entries for updated rows
                for (row_id, existing_arc_values) in &updates_to_old_key {
                    let existing_key =
                        CompositeKey(existing_arc_values.iter().map(|a| (**a).clone()).collect());
                    sorted_values.update_rows(&existing_key, |rows| {
                        rows.retain(|id| *id != *row_id);
                    });
                }

                // Then add new entries
                for &(row_id, values) in entries {
                    let key = CompositeKey(values.to_vec());
                    sorted_values.add(key, row_id);
                }
            });
        }

        // Update prefix indexes only if they were already built
        for prefix_len in 1..num_cols {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                self.mutate_prefix(idx, |prefix_index| {
                    // First remove old entries for updated rows
                    for (row_id, existing_arc_values) in &updates_to_old_key {
                        if existing_arc_values.len() >= prefix_len {
                            let prefix_key = CompositeKey(
                                existing_arc_values[..prefix_len]
                                    .iter()
                                    .map(|a| (**a).clone())
                                    .collect(),
                            );
                            prefix_index.update_rows(&prefix_key, |rows| {
                                rows.retain(|id| *id != *row_id);
                            });
                        }
                    }

                    // Then add new entries
                    for &(row_id, values) in entries {
                        let prefix_key = CompositeKey(values[..prefix_len].to_vec());
                        prefix_index.add(prefix_key, row_id);
                    }
                });
            }
        }

        if self.orders_built() {
            if entries.len() >= Self::WALK_ORDER_BATCH_DROP {
                self.mutate_orders(WalkOrders::clear);
            } else {
                for (row_id, existing_arc_values) in &updates_to_old_key {
                    self.order_remove_arcs(existing_arc_values, *row_id);
                }
                for &(row_id, values) in entries {
                    self.order_insert(values, row_id);
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

        // The locks are taken per chunk so a reader waits for one chunk of
        // a seal's removal, not for all of it
        for chunk in entries.chunks(Self::REMOVE_CHUNK_ROWS) {
            // Grouped by key: one subtraction per key instead of a shift per row
            let removed = Self::group_removed_by_key(chunk, self.column_ids.len());
            self.mutate_main(|value_to_rows, row_to_key| {
                for (key, ids) in removed {
                    value_to_rows.update_rows(&key, |rows| {
                        super::subtract_sorted(rows, &ids);
                    });
                }
                for &(row_id, _) in chunk {
                    row_to_key.remove(row_id);
                }
                Ok(())
            })?;
        }

        // The sorted and prefix structures are subtracted one key at a time:
        // a removal per row is quadratic in the rows per key.
        if self.btree_built.load(AtomicOrdering::Acquire) {
            let removed = Self::group_removed_by_key(entries, self.column_ids.len());
            self.mutate_tree(|sorted_values| {
                for (key, ids) in removed {
                    sorted_values.update_rows(&key, |rows| {
                        super::subtract_sorted(rows, &ids);
                    });
                }
            });
        }

        for prefix_len in 1..self.column_ids.len() {
            let idx = prefix_len - 1;
            if self.prefix_built[idx].load(AtomicOrdering::Acquire) {
                let removed = Self::group_removed_by_key(entries, prefix_len);
                self.mutate_prefix(idx, |prefix_index| {
                    for (key, ids) in removed {
                        prefix_index.update_rows(&key, |rows| {
                            super::subtract_sorted(rows, &ids);
                        });
                    }
                });
            }
        }

        if self.orders_built() {
            if entries.len() >= Self::WALK_ORDER_BATCH_DROP {
                self.mutate_orders(WalkOrders::clear);
            } else {
                for &(row_id, values) in entries {
                    self.order_remove(values, row_id);
                }
            }
        }

        Ok(())
    }

    fn remove_batch_ids(&self, row_ids: &[i64]) -> Option<Result<()>> {
        let mut cleanup_charge = PayloadCharge::unshared(0);
        // The keys come from the row map; the batch path then removes them
        let owned: Vec<(i64, Vec<Value>)> = {
            let row_to_key = self.row_to_key.read();
            let mut cleanup_bytes = 0u128;
            let owned: Vec<_> = row_ids
                .iter()
                .filter_map(|&row_id| {
                    row_to_key.get(row_id).map(|key| {
                        let values: Vec<_> = key
                            .iter()
                            .map(|v| {
                                cleanup_bytes += v.heap_bytes() as u128;
                                (**v).clone()
                            })
                            .collect();
                        cleanup_bytes += (values.capacity() * std::mem::size_of::<Value>()) as u128;
                        (row_id, values)
                    })
                })
                .collect();
            cleanup_charge.add(
                cleanup_bytes
                    + (owned.capacity() * std::mem::size_of::<(i64, Vec<Value>)>()) as u128,
            );
            owned
        };
        if owned.is_empty() {
            return Some(Ok(()));
        }
        let borrowed: Vec<(i64, &[Value])> = owned
            .iter()
            .map(|(row_id, values)| (*row_id, values.as_slice()))
            .collect();
        cleanup_charge.add((borrowed.capacity() * std::mem::size_of::<(i64, &[Value])>()) as u128);
        Some(self.remove_batch_slice(&borrowed))
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

    fn walk_prefix_ordered(
        &self,
        prefix: &[Value],
        lower: Option<(&Value, bool)>,
        upper: Option<(&Value, bool)>,
        ascending: bool,
        visit: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> bool {
        let walked = prefix.len();
        if self.closed.load(AtomicOrdering::Acquire)
            || walked == 0
            || walked >= self.column_ids.len()
        {
            return false;
        }
        let group = CompositeKey(prefix.to_vec());
        // The prefix index is built outside the row map's lock: its build
        // takes that lock too
        self.ensure_prefix_built(walked);
        // A built group is read under the shared lock; a missing one is built
        // under the exclusive lock, checked again once that is held
        let orders = {
            let orders = self.walk_orders.read();
            if orders.contains_key(&group) {
                orders
            } else {
                drop(orders);
                let mut orders = self.walk_orders.write();
                if !orders.contains_key(&group) {
                    let before = orders.nested_bytes;
                    let estimated_before = orders.estimated_bytes();
                    let ids = self.get_row_ids_equal(prefix);
                    let row_to_key = self.row_to_key.read();
                    let mut entries: Vec<(Value, i64)> = Vec::with_capacity(ids.len());
                    let mut value_bytes = 0;
                    for row_id in ids.iter() {
                        if let Some(key) = row_to_key.get(*row_id) {
                            if let Some(value) = key.get(walked) {
                                value_bytes += value.heap_bytes() as u128;
                                entries.push(((**value).clone(), *row_id));
                            }
                        }
                    }
                    entries.sort_unstable();
                    orders.insert(group.clone(), entries.into_iter().collect(), value_bytes);
                    self.memory.account.resize(before, orders.nested_bytes);
                    self.memory
                        .account
                        .resize_estimate(estimated_before, orders.estimated_bytes());
                }
                parking_lot::RwLockWriteGuard::downgrade(orders)
            }
        };
        let Some(order) = orders.get(&group) else {
            return true;
        };
        let start = match lower {
            Some((value, true)) => Bound::Included((value.clone(), i64::MIN)),
            Some((value, false)) => Bound::Excluded((value.clone(), i64::MAX)),
            None => Bound::Unbounded,
        };
        let end = match upper {
            Some((value, true)) => Bound::Included((value.clone(), i64::MAX)),
            Some((value, false)) => Bound::Excluded((value.clone(), i64::MIN)),
            None => Bound::Unbounded,
        };
        // Bounds that cross, or meet with one side open, hold nothing
        if let (
            Bound::Included(low) | Bound::Excluded(low),
            Bound::Included(high) | Bound::Excluded(high),
        ) = (&start, &end)
        {
            let open = matches!(start, Bound::Excluded(_)) || matches!(end, Bound::Excluded(_));
            if low > high || (low == high && open) {
                return true;
            }
        }
        let range = order.range((start, end));
        if ascending {
            for (value, row_id) in range {
                if !visit(*row_id, value) {
                    break;
                }
            }
        } else {
            for (value, row_id) in range.rev() {
                if !visit(*row_id, value) {
                    break;
                }
            }
        }
        true
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
        self.mutate_orders(WalkOrders::clear);
        self.mutate_tree(CompositeTree::clear);
        self.btree_built.store(false, AtomicOrdering::Release);
        {
            let mut values = self.value_to_rows.write();
            let before = values.nested_bytes;
            values.clear();
            self.memory.account.resize(before, 0);
        }
        for (idx, built_flag) in self.prefix_built.iter().enumerate() {
            self.mutate_prefix(idx, CompositeHash::clear);
            built_flag.store(false, AtomicOrdering::Release);
        }
        {
            let mut reverse = self.row_to_key.write();
            let before = reverse.requested_bytes();
            reverse.clear();
            self.memory
                .account
                .resize(before, reverse.requested_bytes());
        }
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

    fn memory_index(unique: bool) -> MultiColumnIndex {
        MultiColumnIndex::new(
            "idx_group_order".into(),
            "items".into(),
            vec!["group".into(), "order".into()],
            vec![0, 1],
            vec![DataType::Text; 2],
            unique,
            32,
        )
    }

    fn assert_requested_allocations(index: &MultiColumnIndex) {
        let key_bytes = |key: &CompositeKey| {
            (key.0.capacity() * std::mem::size_of::<Value>()) as u128
                + key
                    .0
                    .iter()
                    .map(|value| value.heap_bytes() as u128)
                    .sum::<u128>()
        };
        let entry_bytes = |(key, rows): (&CompositeKey, &CompactVec<i64>)| {
            key_bytes(key) + (rows.capacity() * std::mem::size_of::<i64>()) as u128
        };
        let main = index.value_to_rows.read();
        let reverse = index.row_to_key.read();
        let tree = index.sorted_values.read();
        let orders = index.walk_orders.read();
        let main_bytes = main.iter().map(entry_bytes).sum::<u128>();
        let tree_bytes = tree.iter().map(entry_bytes).sum::<u128>();
        let reverse_bytes = reverse
            .values()
            .map(|values| {
                (values.capacity() * std::mem::size_of::<CompactArc<Value>>()) as u128
                    + values
                        .iter()
                        .map(|value| {
                            (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Value>())
                                as u128
                                + value.heap_bytes() as u128
                        })
                        .sum::<u128>()
            })
            .sum::<u128>();
        let order_bytes = orders
            .iter()
            .map(|(key, rows)| {
                key_bytes(key)
                    + rows
                        .iter()
                        .map(|(value, _)| value.heap_bytes() as u128)
                        .sum::<u128>()
            })
            .sum::<u128>();
        assert_eq!(main.nested_bytes, main_bytes);
        assert_eq!(tree.nested_bytes, tree_bytes);
        assert_eq!(reverse.payload_bytes, reverse_bytes);
        assert_eq!(orders.nested_bytes, order_bytes);
        assert_eq!(
            orders.node_units,
            orders
                .values()
                .map(|rows| (1 + rows.len() / 5) as u128)
                .sum::<u128>()
        );
        let mut requested = crate::storage::mvcc::memory::arc_allocation_bytes::<MultiColumnIndex>()
            as u128
            + main_bytes
            + tree_bytes
            + reverse_bytes
            + order_bytes
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
            + (index.data_types.capacity() * std::mem::size_of::<DataType>()) as u128
            + (index.prefix_indexes.capacity() * std::mem::size_of::<RwLock<CompositeHash>>())
                as u128
            + (index.prefix_built.capacity() * std::mem::size_of::<AtomicBool>()) as u128;
        let mut estimated =
            main.estimated_bytes() + tree.estimated_bytes() + orders.estimated_bytes();
        for prefix in &index.prefix_indexes {
            let prefix = prefix.read();
            let bytes = prefix.iter().map(entry_bytes).sum::<u128>();
            assert_eq!(prefix.nested_bytes, bytes);
            requested += bytes;
            estimated += prefix.estimated_bytes();
        }
        assert_eq!(index.memory.account.requested_bytes() as u128, requested);
        assert_eq!(index.memory.account.estimated_bytes() as u128, estimated);
    }

    #[test]
    fn composite_accounting_follows_lazy_build_and_retained_capacity() {
        let index = memory_index(false);
        assert_requested_allocations(&index);
        let key = [
            Value::text("a heap allocated group key"),
            Value::text("a heap allocated order key"),
        ];
        let entries: Vec<_> = (0..32).map(|id| (id, &key[..])).collect();
        index.add_batch_slice(&entries).unwrap();
        assert_requested_allocations(&index);
        let unbuilt = index.memory.account.requested_bytes();
        index.ensure_btree_built();
        index.ensure_prefix_built(1);
        assert!(index.walk_prefix_ordered(&key[..1], None, None, true, &mut |_, _| true));
        assert_requested_allocations(&index);
        let built = index.memory.account.requested_bytes();
        assert!(built > unbuilt);
        assert!(index.walk_prefix_ordered(&key[..1], None, None, true, &mut |_, _| true));
        assert_eq!(index.memory.account.requested_bytes(), built);
        index.remove_batch_slice(&entries[..16]).unwrap();
        assert_requested_allocations(&index);
        let next = [
            key[0].clone(),
            Value::text("a different order key on the same group"),
        ];
        index.add(&next, 16, 0).unwrap();
        assert_requested_allocations(&index);
        index
            .remove_batch_ids(&(16..32).collect::<Vec<_>>())
            .unwrap()
            .unwrap();
        assert_requested_allocations(&index);
        assert!(index.walk_orders.read().values().all(BTreeSet::is_empty));
        assert!(index.memory.account.estimated_bytes() > 0);
        index.clear().unwrap();
        assert_requested_allocations(&index);
        assert!(!index.sorted_values.read().has_nodes);
        let account = Arc::clone(index.memory_account().unwrap());
        drop(index);
        assert_eq!(account.requested_bytes(), 0);
        assert_eq!(account.estimated_bytes(), 0);
    }

    #[test]
    fn composite_accounting_finalizes_failed_mutations() {
        let index = memory_index(true);
        let initial = index.memory.account.requested_bytes();
        let key = [
            Value::text("a repeated group key"),
            Value::text("a repeated order key"),
        ];
        let entries: Vec<_> = (0..1024).map(|id| (id, &key[..])).collect();
        assert!(index.add_batch_slice(&entries).is_err());
        assert!(index.memory.account.requested_bytes() > initial);
        assert_requested_allocations(&index);
        index.add(&key, 1, 0).unwrap();
        let other = [Value::text("a separate group key"), key[1].clone()];
        index.add(&other, 2, 0).unwrap();
        index.ensure_btree_built();
        index.ensure_prefix_built(1);
        assert!(index.walk_prefix_ordered(&other[..1], None, None, true, &mut |_, _| true));
        assert!(index.add(&key, 2, 0).is_err());
        assert_requested_allocations(&index);
    }

    #[test]
    fn composite_large_batch_releases_walk_orders() {
        let index = memory_index(false);
        let key = [
            Value::text("a shared long group key"),
            Value::text("a shared long order key"),
        ];
        let entries: Vec<_> = (0..1024).map(|id| (id, &key[..])).collect();
        index.add_batch_slice(&entries).unwrap();
        assert!(index.walk_prefix_ordered(&key[..1], None, None, true, &mut |_, _| true));
        assert_requested_allocations(&index);
        index.remove_batch_slice(&entries).unwrap();
        assert_requested_allocations(&index);
        let orders = index.walk_orders.read();
        assert!(orders.is_empty());
        assert_eq!(orders.nested_bytes, 0);
        assert_eq!(orders.node_units, 0);
        assert!(orders.capacity_high_water > 0);
    }

    #[test]
    fn composite_removal_debits_the_stored_walk_value() {
        let index = memory_index(false);
        let mut large = String::with_capacity(4096);
        large.push_str("equal text with different retained capacity");
        let original = [Value::Integer(1), Value::Text(large.into())];
        let replacement = [
            Value::Integer(1),
            Value::text("equal text with different retained capacity"),
        ];
        assert!(original[1].heap_bytes() > replacement[1].heap_bytes());
        index.add(&original, 1, 0).unwrap();
        assert!(index.walk_prefix_ordered(&original[..1], None, None, true, &mut |_, _| true));
        index.add_batch_slice(&[(1, &replacement)]).unwrap();
        assert_requested_allocations(&index);
        index.remove(&replacement, 1, 0).unwrap();
        assert_requested_allocations(&index);
        assert_eq!(
            index.walk_orders.read().nested_bytes,
            CompositeKey(original[..1].to_vec()).heap_bytes()
        );
    }

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

    /// Building the prefix index and removing a batch stay linear in the rows
    /// per key. Left out of the default nextest profile as a timing test.
    #[test]
    fn test_prefix_build_and_batch_removal_scale() {
        let index = MultiColumnIndex::new(
            "scale_idx".to_string(),
            "scale_table".to_string(),
            vec!["k".to_string(), "t".to_string()],
            vec![0, 1],
            vec![DataType::Integer, DataType::Integer],
            true,
            0,
        );
        let per_key = 400_000i64;
        let mut all: Vec<(i64, Vec<Value>)> = Vec::new();
        for k in 0..1 {
            for t in 0..per_key {
                let row_id = k * per_key + t;
                let values = vec![Value::Integer(k), Value::Integer(t)];
                index.add(&values, row_id, 0).unwrap();
                all.push((row_id, values));
            }
        }

        let start = std::time::Instant::now();
        let hits = index.find(&[Value::Integer(0)]).unwrap();
        assert!(
            start.elapsed().as_secs() < 15,
            "prefix build took {:?}",
            start.elapsed()
        );
        assert_eq!(hits.len(), per_key as usize);
        assert!(hits.windows(2).all(|w| w[0].row_id < w[1].row_id));

        // Taking the newest two rows off the group leaves the rest untouched
        let newest: Vec<(i64, &[Value])> = all[all.len() - 2..]
            .iter()
            .map(|(row_id, values)| (*row_id, values.as_slice()))
            .collect();
        index.remove_batch_slice(&newest).unwrap();
        let hits = index.find(&[Value::Integer(0)]).unwrap();
        assert_eq!(hits.len(), per_key as usize - 2);
        assert_eq!(hits.last().map(|h| h.row_id), Some(per_key - 3));
        all.truncate(all.len() - 2);

        // Range use builds the sorted structure too
        assert_eq!(
            index
                .find_range(
                    &[Value::Integer(0), Value::Integer(0)],
                    &[Value::Integer(0), Value::Integer(9)],
                    true,
                    true
                )
                .unwrap()
                .len(),
            10
        );

        let half: Vec<(i64, &[Value])> = all
            .iter()
            .filter(|(row_id, _)| row_id % 2 == 0)
            .map(|(row_id, values)| (*row_id, values.as_slice()))
            .collect();
        let start = std::time::Instant::now();
        index.remove_batch_slice(&half).unwrap();
        assert!(
            start.elapsed().as_secs() < 15,
            "batch removal took {:?}",
            start.elapsed()
        );
        let hits = index.find(&[Value::Integer(0)]).unwrap();
        assert_eq!(hits.len(), all.len() - half.len());
        assert!(hits.iter().all(|h| h.row_id % 2 == 1));

        let rest: Vec<(i64, &[Value])> = all
            .iter()
            .filter(|(row_id, _)| row_id % 2 == 1)
            .map(|(row_id, values)| (*row_id, values.as_slice()))
            .collect();
        index.remove_batch_slice(&rest).unwrap();
        assert!(index.find(&[Value::Integer(0)]).unwrap().is_empty());
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

    /// A group built while an add is between the row map and the prefix
    /// index must still end up holding the added row: the add enters it
    /// into the group once it gets to the orders, which come last
    #[test]
    fn test_walk_order_built_during_an_add_keeps_the_added_row() {
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        let index = Arc::new(MultiColumnIndex::new(
            "idx".into(),
            "c".into(),
            vec!["g".into(), "k".into(), "t".into()],
            vec![0, 1, 2],
            vec![DataType::Integer; 3],
            false,
            0,
        ));
        let row = |t: i64| vec![Value::Integer(1), Value::Integer(2), Value::Integer(t)];
        index.add(&row(10), 1, 0).unwrap();
        index.ensure_btree_built();
        index.ensure_prefix_built(2);

        // The writer stops before the prefix update while this range reader
        // holds the sorted structure it updates first
        let range_reader = index.sorted_values.read();
        let writer_index = Arc::clone(&index);
        let writer = std::thread::spawn(move || {
            writer_index.add(&row(20), 2, 0).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while index.row_to_key.read().get(2).is_none() {
            if Instant::now() > deadline {
                drop(range_reader);
                writer.join().unwrap();
                panic!("the writer did not reach the row map");
            }
            std::thread::yield_now();
        }
        let prefix = [Value::Integer(1), Value::Integer(2)];
        assert!(index.walk_prefix_ordered(&prefix, None, None, true, &mut |_, _| true));
        drop(range_reader);
        writer.join().unwrap();

        let mut ordered = Vec::new();
        assert!(
            index.walk_prefix_ordered(&prefix, None, None, true, &mut |id, _| {
                ordered.push(id);
                true
            })
        );
        assert_eq!(ordered, vec![1, 2]);
    }
}
