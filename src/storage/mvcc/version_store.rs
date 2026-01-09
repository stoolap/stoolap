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

//! Version store for MVCC row versioning
//!
//! This module provides the core version storage for MVCC, including:
//! - [`RowVersion`] - Represents a specific version of a row
//! - [`VersionStore`] - Tracks latest committed versions for a table
//! - [`TransactionVersionStore`] - Transaction-local changes before commit
//!
//! # Performance
//!
//! The version store uses arena-based storage for zero-copy full table scans.
//! Row data is stored contiguously in memory, enabling 50x+ faster scans
//! compared to traditional per-row cloning.
//!

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::common::{
    new_btree_int64_map, new_int64_map, new_int64_map_with_capacity, BTreeInt64Map, Int64Map,
};
use crate::core::types::DataType;
use crate::core::{Error, Row, RowVec, Schema, Value};
use crate::storage::expression::CompiledFilter;
use crate::storage::mvcc::arena::RowArena;
use crate::storage::mvcc::get_fast_timestamp;
#[cfg(not(test))]
use crate::storage::mvcc::registry::TransactionRegistry;
use crate::storage::mvcc::streaming_result::{StreamingResult, VisibleRowInfo};
use crate::storage::Index;
// radsort removed - BTreeMap iteration is already ordered
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};

/// Type alias for version lists - uses SmallVec to avoid heap allocation
/// for the common case of a single version per row within a transaction.
type VersionList = SmallVec<[RowVersion; 1]>;

/// Group key using Arc<Value> to avoid cloning during aggregation.
/// Uses Arc::clone (O(1) atomic increment) instead of Value::clone (deep copy).
#[derive(Clone, Debug)]
pub enum GroupKey {
    Single(Arc<Value>),
    Multi(Vec<Arc<Value>>),
}

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GroupKey::Single(a), GroupKey::Single(b)) => **a == **b,
            (GroupKey::Multi(a), GroupKey::Multi(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| **x == **y)
            }
            _ => false,
        }
    }
}

impl Eq for GroupKey {}

impl std::hash::Hash for GroupKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            GroupKey::Single(v) => (**v).hash(state),
            GroupKey::Multi(vs) => {
                for v in vs {
                    (**v).hash(state);
                }
            }
        }
    }
}

/// Result of storage-level grouped aggregation
#[derive(Debug, Clone)]
pub struct GroupedAggregateResult {
    /// Group key values
    pub group_values: Vec<Value>,
    /// Aggregate results in order of requested aggregates
    pub aggregate_values: Vec<Value>,
}

/// Represents a specific version of a row with complete data
///

#[derive(Clone)]
pub struct RowVersion {
    /// Transaction that created this version
    pub txn_id: i64,
    /// Transaction that deleted this version (0 if not deleted)
    pub deleted_at_txn_id: i64,
    /// Complete row data
    pub data: Row,
    /// Row identifier
    pub row_id: i64,
    /// Timestamp when this version was created
    pub create_time: i64,
}

impl RowVersion {
    /// Creates a new row version
    pub fn new(txn_id: i64, row_id: i64, data: Row) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: 0,
            data,
            row_id,
            create_time: get_fast_timestamp(),
        }
    }

    /// Creates a new row version with a pre-computed timestamp
    /// This avoids calling SystemTime::now() for each row in bulk operations
    #[inline]
    pub fn new_with_timestamp(txn_id: i64, row_id: i64, data: Row, create_time: i64) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: 0,
            data,
            row_id,
            create_time,
        }
    }

    /// Creates a new deleted version
    pub fn new_deleted(txn_id: i64, row_id: i64, data: Row) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: txn_id,
            data,
            row_id,
            create_time: get_fast_timestamp(),
        }
    }

    /// Creates a new deleted version with a pre-computed timestamp
    #[inline]
    pub fn new_deleted_with_timestamp(
        txn_id: i64,
        row_id: i64,
        data: Row,
        create_time: i64,
    ) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: txn_id,
            data,
            row_id,
            create_time,
        }
    }

    /// Returns true if this version has been marked as deleted
    pub fn is_deleted(&self) -> bool {
        self.deleted_at_txn_id != 0
    }
}

impl fmt::Debug for RowVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowVersion")
            .field("txn_id", &self.txn_id)
            .field("deleted_at_txn_id", &self.deleted_at_txn_id)
            .field("row_id", &self.row_id)
            .field("create_time", &self.create_time)
            .finish()
    }
}

impl fmt::Display for RowVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RowVersion{{TxnID: {}, DeletedAtTxnID: {}, RowID: {}, CreateTime: {}}}",
            self.txn_id, self.deleted_at_txn_id, self.row_id, self.create_time
        )
    }
}

/// Entry in the version chain (linked list of versions)
/// Uses Arc for the prev pointer to enable O(1) cloning of the chain
struct VersionChainEntry {
    /// Current version
    version: RowVersion,
    /// Previous version in the chain (Arc allows cheap cloning)
    prev: Option<Arc<VersionChainEntry>>,
    /// Index into the arena for zero-copy access (None if data not in arena)
    arena_idx: Option<usize>,
    /// Depth of this chain (number of previous versions including this one)
    /// Used for amortized O(1) truncation - only truncate when depth exceeds limit
    chain_depth: usize,
}

/// Tracks write operations with the version read for conflict detection
///

#[derive(Clone)]
pub struct WriteSetEntry {
    /// Version when first read (None if row didn't exist)
    pub read_version: Option<RowVersion>,
    /// Sequence number when read
    pub read_version_seq: i64,
}

// ============================================================================
// HashMap Pool for TransactionVersionStore
// ============================================================================
//
// Pools recycled HashMaps to avoid allocation overhead when creating many
// short-lived transactions. Each auto-commit INSERT creates a transaction,
// and without pooling, this causes ~5.5KB allocation per INSERT.
//
// With pooling, maps are returned to the pool on Drop and reused by the next
// transaction, reducing allocation churn by ~99% for bulk insert workloads.

/// Maximum number of maps to keep in each pool.
/// Prevents unbounded memory growth while allowing reasonable reuse.
const MAP_POOL_MAX_SIZE: usize = 64;

/// Global pool for VersionList maps (local_versions in TransactionVersionStore)
static VERSION_LIST_MAP_POOL: Mutex<Vec<Int64Map<VersionList>>> = Mutex::new(Vec::new());

/// Global pool for WriteSetEntry maps (write_set in TransactionVersionStore)
static WRITE_SET_MAP_POOL: Mutex<Vec<Int64Map<WriteSetEntry>>> = Mutex::new(Vec::new());

/// Get a VersionList map from pool or create a new one
#[inline]
fn get_version_list_map() -> Int64Map<VersionList> {
    if let Some(map) = VERSION_LIST_MAP_POOL.lock().pop() {
        map
    } else {
        new_int64_map_with_capacity(TX_VERSION_MAP_INITIAL_CAPACITY)
    }
}

/// Get a WriteSetEntry map from pool or create a new one
#[inline]
fn get_write_set_map() -> Int64Map<WriteSetEntry> {
    if let Some(map) = WRITE_SET_MAP_POOL.lock().pop() {
        map
    } else {
        new_int64_map_with_capacity(TX_VERSION_MAP_INITIAL_CAPACITY)
    }
}

/// Return a VersionList map to the pool for reuse
#[inline]
fn return_version_list_map(mut map: Int64Map<VersionList>) {
    map.clear();
    let mut pool = VERSION_LIST_MAP_POOL.lock();
    if pool.len() < MAP_POOL_MAX_SIZE {
        pool.push(map);
    }
    // If pool is full, map is dropped (deallocated)
}

/// Return a WriteSetEntry map to the pool for reuse
#[inline]
fn return_write_set_map(mut map: Int64Map<WriteSetEntry>) {
    map.clear();
    let mut pool = WRITE_SET_MAP_POOL.lock();
    if pool.len() < MAP_POOL_MAX_SIZE {
        pool.push(map);
    }
    // If pool is full, map is dropped (deallocated)
}

/// Capacity hint for transaction version maps - used by pool functions
const TX_VERSION_MAP_INITIAL_CAPACITY: usize = 16;

/// Lightweight row index for deferred materialization
///
/// Instead of cloning row data during scans, we return indices that can be
/// materialized later. This enables zero-copy filtering and limiting.
///
/// # Performance
/// For `SELECT * FROM t WHERE x > 100 LIMIT 10` on 100K rows:
/// - Old: Clone 100K rows, filter to 50K, limit to 10 (100K allocations)
/// - New: Get 100K indices, filter to 50K, limit to 10, clone 10 (10 allocations)
#[derive(Clone, Copy, Debug)]
pub struct RowIndex {
    /// Row ID
    pub row_id: i64,
    /// Arena index (None if row data is not in arena, must clone from version)
    pub arena_idx: Option<usize>,
}

/// Aggregate operation type for deferred aggregation
///
/// Used with `compute_aggregates()` to perform multiple aggregations in a single pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateOp {
    Count,
    CountStar,
    Sum,
    Min,
    Max,
    Avg,
}

/// Result of an aggregate operation
#[derive(Clone, Debug)]
pub enum AggregateResult {
    /// Count result
    Count(usize),
    /// Sum result (sum, count of non-null values)
    Sum(f64, usize),
    /// Min result
    Min(Option<Value>),
    /// Max result
    Max(Option<Value>),
    /// Avg result (sum, count of non-null values) - caller computes sum/count
    Avg(f64, usize),
}

/// Internal accumulator for aggregations
enum AggregateAccumulator {
    Count(usize),
    Sum(f64, usize),
    Min(Option<Value>),
    Max(Option<Value>),
    Avg(f64, usize),
}

/// Visibility checker trait - will be implemented by TransactionRegistry
///
/// This allows VersionStore to check visibility without circular dependencies
pub trait VisibilityChecker: Send + Sync {
    /// Check if a version created by `version_txn_id` is visible to `viewing_txn_id`
    fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool;

    /// Get the current global sequence number
    fn get_current_sequence(&self) -> i64;

    /// Get all active transaction IDs (for cleanup operations)
    fn get_active_transaction_ids(&self) -> Vec<i64>;

    /// Check if a transaction was committed before a given commit sequence cutoff.
    ///
    /// Returns true if the transaction is committed AND its commit sequence
    /// is less than the cutoff. Used for consistent snapshot iteration to ensure
    /// only transactions committed before the snapshot point are included.
    ///
    /// Default implementation returns true for all committed transactions.
    fn is_committed_before(&self, _txn_id: i64, _cutoff_commit_seq: i64) -> bool {
        true // Default: no cutoff filtering
    }
}

/// VersionStore tracks the latest committed version of each row for a table
///
/// Uses OrderedInt64Map (RwLock<BTreeMap>) for the version store because:
/// - Ordered iteration is free (BTreeMap is sorted by key)
/// - MVCC has single-writer semantics per transaction, so DashMap's sharding is overhead
/// - Point lookups are O(log n) which is fast enough for typical row counts
/// - Eliminates the ~350μs sort overhead during full scans
///
/// Arena-based storage provides 50x+ faster full table scans by:
/// - Storing all row data contiguously in memory
/// - Returning slices instead of clones during iteration
/// - Eliminating per-row allocation overhead
pub struct VersionStore {
    /// Row versions indexed by row ID (BTreeMap for ordered iteration)
    versions: BTreeInt64Map<VersionChainEntry>,
    /// The name of the table this store belongs to
    table_name: String,
    /// Table schema (Arc for zero-cost cloning on read)
    schema: RwLock<Arc<Schema>>,
    /// Indexes on this table (FxHashMap for fast string key lookups)
    indexes: RwLock<FxHashMap<String, Arc<dyn Index>>>,
    /// Whether this store has been closed
    closed: AtomicBool,
    /// Auto-increment counter for tables without explicit PK
    auto_increment_counter: AtomicI64,
    /// Track which transaction has uncommitted changes to each row
    uncommitted_writes: RwLock<Int64Map<i64>>,
    /// Visibility checker (registry reference)
    /// Production: concrete type for zero-cost inlining in hot paths
    /// Test: dyn trait for TestVisibilityChecker flexibility
    #[cfg(not(test))]
    visibility_checker: Option<Arc<TransactionRegistry>>,
    #[cfg(test)]
    visibility_checker: Option<Arc<dyn VisibilityChecker>>,
    /// Arena-based storage for zero-copy full table scans
    arena: RowArena,
    /// Maps row_id to the latest arena index for that row (Int64Map for fast i64 key lookups)
    row_arena_index: RwLock<Int64Map<usize>>,
    /// Zone maps for segment pruning (set by ANALYZE)
    /// Uses Arc to avoid cloning on every read - critical for high QPS workloads
    zone_maps: RwLock<Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>>>,
    /// Maximum number of previous versions to keep per row (0 = unlimited)
    /// This limits memory growth during write-heavy operations.
    /// Default is 10 - enough for most concurrent transaction scenarios.
    max_version_history: usize,
}

impl VersionStore {
    /// Creates a new version store
    pub fn new(table_name: String, schema: Schema) -> Self {
        Self::with_capacity(table_name, schema, None, 0)
    }

    /// Creates a new version store with pre-allocated capacity
    ///
    /// Pre-allocating capacity avoids hash map resizing during bulk inserts.
    /// Use this when the expected row count is known (e.g., during recovery).
    #[cfg(not(test))]
    pub fn with_capacity(
        table_name: String,
        schema: Schema,
        checker: Option<Arc<TransactionRegistry>>,
        expected_rows: usize,
    ) -> Self {
        // BTreeMap doesn't support capacity hints, but row_arena_index (HashMap) does
        let row_arena_index = if expected_rows > 0 {
            RwLock::new(new_int64_map_with_capacity(expected_rows))
        } else {
            RwLock::new(new_int64_map())
        };
        let versions = new_btree_int64_map();

        Self {
            versions,
            table_name,
            schema: RwLock::new(Arc::new(schema)),
            indexes: RwLock::new(FxHashMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: RwLock::new(new_int64_map()),
            visibility_checker: checker,
            arena: RowArena::new(),
            row_arena_index,
            zone_maps: RwLock::new(None),
            max_version_history: 10, // Default: keep up to 10 previous versions
        }
    }

    /// Creates a new version store with pre-allocated capacity (test version)
    #[cfg(test)]
    pub fn with_capacity(
        table_name: String,
        schema: Schema,
        checker: Option<Arc<dyn VisibilityChecker>>,
        expected_rows: usize,
    ) -> Self {
        let row_arena_index = if expected_rows > 0 {
            RwLock::new(new_int64_map_with_capacity(expected_rows))
        } else {
            RwLock::new(new_int64_map())
        };
        let versions = new_btree_int64_map();

        Self {
            versions,
            table_name,
            schema: RwLock::new(Arc::new(schema)),
            indexes: RwLock::new(FxHashMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: RwLock::new(new_int64_map()),
            visibility_checker: checker,
            arena: RowArena::new(),
            row_arena_index,
            zone_maps: RwLock::new(None),
            max_version_history: 10,
        }
    }

    /// Sets the maximum version history limit per row
    ///
    /// This controls how many previous versions are kept for each row.
    /// Lower values reduce memory usage but limit time-travel query range.
    /// - 0 = unlimited (not recommended for write-heavy workloads)
    /// - 1 = only keep immediate previous (minimal memory, limited AS OF range)
    /// - 10 = default, good balance for most workloads
    pub fn set_max_version_history(&mut self, limit: usize) {
        self.max_version_history = limit;
    }

    /// Gets the current max version history limit
    pub fn max_version_history(&self) -> usize {
        self.max_version_history
    }

    /// Creates a new version store with a visibility checker (production)
    #[cfg(not(test))]
    pub fn with_visibility_checker(
        table_name: String,
        schema: Schema,
        checker: Arc<TransactionRegistry>,
    ) -> Self {
        Self::with_capacity(table_name, schema, Some(checker), 0)
    }

    /// Creates a new version store with a visibility checker (test)
    #[cfg(test)]
    pub fn with_visibility_checker(
        table_name: String,
        schema: Schema,
        checker: Arc<dyn VisibilityChecker>,
    ) -> Self {
        Self::with_capacity(table_name, schema, Some(checker), 0)
    }

    /// Sets the visibility checker (production)
    #[cfg(not(test))]
    pub fn set_visibility_checker(&mut self, checker: Arc<TransactionRegistry>) {
        self.visibility_checker = Some(checker);
    }

    /// Sets the visibility checker (test)
    #[cfg(test)]
    pub fn set_visibility_checker(&mut self, checker: Arc<dyn VisibilityChecker>) {
        self.visibility_checker = Some(checker);
    }

    /// Returns the table name
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Returns the schema (cheap Arc clone)
    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&*self.schema.read())
    }

    /// Returns a mutable reference to the schema (for modifications)
    /// Callers must use Arc::make_mut() to get &mut Schema
    pub fn schema_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Arc<Schema>> {
        self.schema.write()
    }

    /// Returns the current auto-increment counter value
    pub fn get_auto_increment_counter(&self) -> i64 {
        self.auto_increment_counter.load(Ordering::Acquire)
    }

    /// Returns the next available auto-increment ID
    pub fn get_next_auto_increment_id(&self) -> i64 {
        self.auto_increment_counter.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Sets the auto-increment counter to a specific value (only if current is lower)
    ///
    /// Returns true if the value was updated, false if no update was needed
    pub fn set_auto_increment_counter(&self, value: i64) -> bool {
        loop {
            let current = self.auto_increment_counter.load(Ordering::Acquire);
            if current >= value {
                return false;
            }

            if self
                .auto_increment_counter
                .compare_exchange(current, value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Returns the current auto-increment value without incrementing
    pub fn get_current_auto_increment_value(&self) -> i64 {
        self.auto_increment_counter.load(Ordering::Acquire)
    }

    /// Adds a new version for a row
    pub fn add_version(&self, row_id: i64, version: RowVersion) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }

        // Use write lock for the entire operation (MVCC single-writer semantics)
        let mut versions = self.versions.write();

        // Use entry API to avoid double BTreeMap traversal (get + insert -> single entry)
        match versions.entry(row_id) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                // Extract existing data from the entry
                let existing = occupied.get();
                let existing_arena_idx = existing.arena_idx;
                let existing_depth = existing.chain_depth;

                // O(1) chain management with depth tracking
                // When limit exceeded: drop old chain AND reuse arena slot
                let new_depth = existing_depth + 1;
                let can_reuse_arena =
                    self.max_version_history > 0 && new_depth > self.max_version_history;

                // Only clone existing version data when needed:
                // 1. For delete operations that need to preserve data
                // 2. When keeping version history (not pruning)
                let mut new_version = version;
                if new_version.deleted_at_txn_id != 0 && new_version.data.is_empty() {
                    // For deletes, preserve data from current version
                    new_version.data = existing.version.data.clone();
                }

                // Store in arena (only for non-deleted versions)
                // OPTIMIZATION: Always reuse arena slot - historical data is in prev_chain
                let arena_idx = if new_version.deleted_at_txn_id == 0 {
                    // Convert Row to Arc once (takes ownership, no copy if already Arc)
                    let arc_data = std::mem::take(&mut new_version.data).into_arc();

                    // Always reuse existing arena slot if available
                    // Historical versions are stored in prev_chain.version.data
                    let idx = if let Some(old_idx) = existing_arena_idx {
                        // Reuse the arena slot - prevents unbounded growth
                        self.arena.update_at(
                            old_idx,
                            row_id,
                            new_version.txn_id,
                            new_version.create_time,
                            Arc::clone(&arc_data),
                        );
                        old_idx
                    } else {
                        // No existing slot (shouldn't happen for updates), append
                        self.arena.insert_arc(
                            row_id,
                            new_version.txn_id,
                            new_version.create_time,
                            Arc::clone(&arc_data),
                        )
                    };

                    // Reuse the Arc for the version's data - enables O(1) clone on read
                    new_version.data = Row::from_arc_slice(arc_data);
                    // Update row -> arena index mapping
                    self.row_arena_index.write().insert(row_id, idx);

                    Some(idx)
                } else {
                    // Deleted version - mark arena as deleted for visibility
                    if let Some(old_arena_idx) = existing_arena_idx {
                        self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                    }
                    existing_arena_idx
                };

                // Build version chain entry
                // When limit exceeded: drop entire history (no prev_chain allocation)
                // When under limit: create prev_chain with existing version
                let (final_prev, final_depth) = if can_reuse_arena {
                    // Exceeded limit - drop all history, no allocation
                    (None, 1)
                } else {
                    // Under limit - clone existing version and create chain
                    let existing_version = existing.version.clone();
                    let existing_prev = existing.prev.clone();
                    let prev_chain = Arc::new(VersionChainEntry {
                        version: existing_version,
                        prev: existing_prev,
                        arena_idx: existing_arena_idx,
                        chain_depth: existing_depth, // Preserve existing chain depth
                    });
                    (Some(prev_chain), existing_depth + 1) // New entry is one deeper
                };

                let new_entry = VersionChainEntry {
                    version: new_version,
                    prev: final_prev,
                    arena_idx,
                    chain_depth: final_depth,
                };

                // Replace entry in-place (no additional tree traversal)
                occupied.insert(new_entry);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                // First version for this row - store in arena
                // OPTIMIZATION: Convert Row to Arc once, then just clone Arc (no data copy)
                let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                    let mut v = version;
                    // Convert Row to Arc once (takes ownership, no copy if already Arc)
                    let arc_data = std::mem::take(&mut v.data).into_arc();
                    // Insert Arc into arena (just Arc::clone, no data copy)
                    let idx = self.arena.insert_arc(
                        row_id,
                        v.txn_id,
                        v.create_time,
                        Arc::clone(&arc_data),
                    );
                    // Update row -> arena index mapping
                    self.row_arena_index.write().insert(row_id, idx);
                    // Create version with Arc-backed data for O(1) clone
                    v.data = Row::from_arc_slice(arc_data);
                    (Some(idx), v)
                } else {
                    (None, version)
                };

                let new_entry = VersionChainEntry {
                    version: final_version,
                    prev: None,
                    arena_idx,
                    chain_depth: 1, // First version
                };

                // Insert into vacant slot (no additional traversal)
                vacant.insert(new_entry);
            }
        }
    }

    /// Adds multiple versions in batch - used by commit
    ///
    /// Updates both the version store and the arena atomically per row.
    /// Arena updates are O(1) per row (just an insert), so this is efficient.
    /// Row arena index updates are batched under a single lock acquisition.
    /// Version data uses Arc-backed storage for O(1) clones on read.
    #[inline]
    pub fn add_versions_batch(&self, batch: Vec<(i64, RowVersion)>) {
        if self.closed.load(Ordering::Acquire) || batch.is_empty() {
            return;
        }

        // Use write lock for the entire batch operation (MVCC single-writer semantics)
        let mut versions = self.versions.write();

        // Collect arena index updates to batch them under a single lock
        let mut arena_index_updates: Vec<(i64, usize)> = Vec::with_capacity(batch.len());

        for (row_id, version) in batch {
            // Use entry API to avoid double BTreeMap traversal (get + insert -> single entry)
            match versions.entry(row_id) {
                std::collections::btree_map::Entry::Occupied(mut occupied) => {
                    // Extract existing data from the entry
                    let existing = occupied.get();
                    let existing_arena_idx = existing.arena_idx;
                    let existing_depth = existing.chain_depth;

                    // O(1) chain management with depth tracking
                    // When limit exceeded: drop old chain AND reuse arena slot
                    let new_depth = existing_depth + 1;
                    let can_reuse_arena =
                        self.max_version_history > 0 && new_depth > self.max_version_history;

                    // Only clone existing version data when needed:
                    // 1. For delete operations that need to preserve data
                    // 2. When keeping version history (not pruning)
                    let mut new_version = version;
                    if new_version.deleted_at_txn_id != 0 && new_version.data.is_empty() {
                        // For deletes, preserve data from current version
                        new_version.data = existing.version.data.clone();
                    }

                    // Update arena with Arc reuse for O(1) clones on read
                    // OPTIMIZATION: Always reuse arena slot - historical data is in prev_chain
                    let arena_idx = if new_version.deleted_at_txn_id == 0 {
                        // Convert Row to Arc once (takes ownership, no copy if already Arc)
                        let arc_data = std::mem::take(&mut new_version.data).into_arc();

                        // Always reuse existing arena slot if available
                        // Historical versions are stored in prev_chain.version.data
                        let idx = if let Some(old_idx) = existing_arena_idx {
                            // Reuse the arena slot - prevents unbounded growth
                            self.arena.update_at(
                                old_idx,
                                row_id,
                                new_version.txn_id,
                                new_version.create_time,
                                Arc::clone(&arc_data),
                            );
                            old_idx
                        } else {
                            // No existing slot (shouldn't happen for updates), append
                            self.arena.insert_arc(
                                row_id,
                                new_version.txn_id,
                                new_version.create_time,
                                Arc::clone(&arc_data),
                            )
                        };

                        // Reuse the Arc for the version's data
                        new_version.data = Row::from_arc_slice(arc_data);
                        arena_index_updates.push((row_id, idx));

                        Some(idx)
                    } else {
                        // Deleted version - mark arena as deleted for visibility
                        if let Some(old_arena_idx) = existing_arena_idx {
                            self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                        }
                        existing_arena_idx
                    };

                    // Build version chain entry
                    // When limit exceeded: drop entire history (no prev_chain allocation)
                    // When under limit: create prev_chain with existing version
                    let (final_prev, final_depth) = if can_reuse_arena {
                        // Exceeded limit - drop all history, no allocation
                        (None, 1)
                    } else {
                        // Under limit - clone existing version and create chain
                        let existing_version = existing.version.clone();
                        let existing_prev = existing.prev.clone();
                        let prev_chain = Arc::new(VersionChainEntry {
                            version: existing_version,
                            prev: existing_prev,
                            arena_idx: existing_arena_idx,
                            chain_depth: existing_depth, // Preserve existing chain depth
                        });
                        (Some(prev_chain), existing_depth + 1) // New entry is one deeper
                    };

                    let new_entry = VersionChainEntry {
                        version: new_version,
                        prev: final_prev,
                        arena_idx,
                        chain_depth: final_depth,
                    };

                    // Replace entry in-place (no additional tree traversal)
                    occupied.insert(new_entry);
                }
                std::collections::btree_map::Entry::Vacant(vacant) => {
                    // First version for this row - store in arena with Arc reuse
                    // OPTIMIZATION: Convert Row to Arc once, then just clone Arc (no data copy)
                    let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                        let mut v = version;
                        // Convert Row to Arc once (takes ownership, no copy if already Arc)
                        let arc_data = std::mem::take(&mut v.data).into_arc();
                        // Insert Arc into arena (just Arc::clone, no data copy)
                        let idx = self.arena.insert_arc(
                            row_id,
                            v.txn_id,
                            v.create_time,
                            Arc::clone(&arc_data),
                        );
                        arena_index_updates.push((row_id, idx));
                        // Create version with Arc-backed data for O(1) clone
                        v.data = Row::from_arc_slice(arc_data);
                        (Some(idx), v)
                    } else {
                        (None, version)
                    };

                    let new_entry = VersionChainEntry {
                        version: final_version,
                        prev: None,
                        arena_idx,
                        chain_depth: 1,
                    };

                    // Insert into vacant slot (no additional traversal)
                    vacant.insert(new_entry);
                }
            }
        }

        // Batch update row_arena_index under a single lock acquisition
        if !arena_index_updates.is_empty() {
            let mut index = self.row_arena_index.write();
            for (row_id, idx) in arena_index_updates {
                index.insert(row_id, idx);
            }
        }
    }

    /// Add a single version to the store (optimized for auto-commit single-row inserts)
    ///
    /// This avoids Vec allocation for the common single-row commit case.
    #[inline]
    pub fn add_version_single(&self, row_id: i64, version: RowVersion) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }

        let mut versions = self.versions.write();

        match versions.entry(row_id) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                // Extract existing data from the entry
                let existing = occupied.get();
                let existing_arena_idx = existing.arena_idx;
                let existing_depth = existing.chain_depth;

                // O(1) chain management with depth tracking
                // When limit exceeded: drop old chain AND reuse arena slot
                let new_depth = existing_depth + 1;
                let can_reuse_arena =
                    self.max_version_history > 0 && new_depth > self.max_version_history;

                // Only clone existing version data when needed:
                // 1. For delete operations that need to preserve data
                // 2. When keeping version history (not pruning)
                let mut new_version = version;
                if new_version.deleted_at_txn_id != 0 && new_version.data.is_empty() {
                    // For deletes, preserve data from current version
                    new_version.data = existing.version.data.clone();
                }

                // OPTIMIZATION: Always reuse arena slot - historical data is in prev_chain
                let arena_idx = if new_version.deleted_at_txn_id == 0 {
                    let arc_data = std::mem::take(&mut new_version.data).into_arc();

                    // Always reuse existing arena slot if available
                    // Historical versions are stored in prev_chain.version.data
                    let idx = if let Some(old_idx) = existing_arena_idx {
                        // Reuse the arena slot - prevents unbounded growth
                        self.arena.update_at(
                            old_idx,
                            row_id,
                            new_version.txn_id,
                            new_version.create_time,
                            Arc::clone(&arc_data),
                        );
                        old_idx
                    } else {
                        // No existing slot (shouldn't happen for updates), append
                        self.arena.insert_arc(
                            row_id,
                            new_version.txn_id,
                            new_version.create_time,
                            Arc::clone(&arc_data),
                        )
                    };

                    new_version.data = Row::from_arc_slice(arc_data);
                    // Update arena index immediately (single row, no batching needed)
                    self.row_arena_index.write().insert(row_id, idx);

                    Some(idx)
                } else {
                    // Deleted version - mark arena as deleted for visibility
                    if let Some(old_arena_idx) = existing_arena_idx {
                        self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                    }
                    existing_arena_idx
                };

                // Build version chain entry
                // When limit exceeded: drop entire history (no prev_chain allocation)
                // When under limit: create prev_chain with existing version
                let (final_prev, final_depth) = if can_reuse_arena {
                    // Exceeded limit - drop all history, no allocation
                    (None, 1)
                } else {
                    // Under limit - clone existing version and create chain
                    let existing_version = existing.version.clone();
                    let existing_prev = existing.prev.clone();
                    let prev_chain = Arc::new(VersionChainEntry {
                        version: existing_version,
                        prev: existing_prev,
                        arena_idx: existing_arena_idx,
                        chain_depth: existing_depth, // Preserve existing chain depth
                    });
                    (Some(prev_chain), existing_depth + 1) // New entry is one deeper
                };

                let new_entry = VersionChainEntry {
                    version: new_version,
                    prev: final_prev,
                    arena_idx,
                    chain_depth: final_depth,
                };

                occupied.insert(new_entry);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                    let mut v = version;
                    let arc_data = std::mem::take(&mut v.data).into_arc();
                    let idx = self.arena.insert_arc(
                        row_id,
                        v.txn_id,
                        v.create_time,
                        Arc::clone(&arc_data),
                    );
                    // Update arena index immediately
                    self.row_arena_index.write().insert(row_id, idx);
                    v.data = Row::from_arc_slice(arc_data);
                    (Some(idx), v)
                } else {
                    (None, version)
                };

                let new_entry = VersionChainEntry {
                    version: final_version,
                    prev: None,
                    arena_idx,
                    chain_depth: 1,
                };

                vacant.insert(new_entry);
            }
        }
    }

    /// Quick check if a row might exist
    pub fn quick_check_row_existence(&self, row_id: i64) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }

        self.versions.read().contains_key(&row_id)
    }

    /// Gets the latest visible version of a row
    pub fn get_visible_version(&self, row_id: i64, txn_id: i64) -> Option<RowVersion> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let checker = self.visibility_checker.as_ref()?;

        // ARENA FAST PATH: O(1) HashMap lookup instead of O(log n) BTreeMap
        // This bypasses BTreeMap entirely when HEAD is visible (common case).
        // row_arena_index maps row_id → arena_idx for O(1) access.
        if let Some(&arena_idx) = self.row_arena_index.read().get(&row_id) {
            if let Some((meta, arc_data)) = self.arena.get_meta_and_arc(arena_idx) {
                // Check HEAD visibility using arena metadata
                if checker.is_visible(meta.txn_id, txn_id) {
                    // HEAD is visible - check if deleted
                    if meta.deleted_at_txn_id != 0
                        && checker.is_visible(meta.deleted_at_txn_id, txn_id)
                    {
                        return None; // Row is deleted
                    }
                    // Return from arena - O(1) total!
                    return Some(RowVersion {
                        txn_id: meta.txn_id,
                        deleted_at_txn_id: meta.deleted_at_txn_id,
                        data: Row::from_arc_slice(arc_data),
                        row_id: meta.row_id,
                        create_time: meta.create_time,
                    });
                }
                // HEAD not visible - fall through to BTreeMap for version chain
            }
        }

        // BTREE SLOW PATH: HEAD not visible or row not in arena index
        // Need BTreeMap for version chain traversal (Snapshot Isolation, AS OF, etc.)
        let versions = self.versions.read();
        let chain = versions.get(&row_id)?;

        // Check HEAD in case arena path was skipped
        let head_txn_id = chain.version.txn_id;
        let head_deleted_at = chain.version.deleted_at_txn_id;

        if checker.is_visible(head_txn_id, txn_id) {
            if head_deleted_at != 0 && checker.is_visible(head_deleted_at, txn_id) {
                return None;
            }
            return Some(chain.version.clone());
        }

        // Traverse version chain for older visible versions
        let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());

        while let Some(e) = current {
            let version_txn_id = e.version.txn_id;
            let deleted_at_txn_id = e.version.deleted_at_txn_id;

            if checker.is_visible(version_txn_id, txn_id) {
                if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                    return None;
                }
                return Some(e.version.clone());
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
        }

        None
    }

    /// Check if any of the given row_ids have a visible version
    /// Returns the first row_id that has a visible version, or None if none exist
    /// OPTIMIZATION: Used for conflict detection - stops at first hit, no data fetch
    #[inline]
    pub fn has_any_visible_version(&self, row_ids: &[i64], txn_id: i64) -> Option<i64> {
        if self.closed.load(Ordering::Acquire) || row_ids.is_empty() {
            return None;
        }

        let checker = self.visibility_checker.as_ref()?;
        let versions = self.versions.read();

        for &row_id in row_ids {
            if let Some(chain) = versions.get(&row_id) {
                // Check HEAD visibility (most common case)
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        return Some(row_id); // Found visible version - conflict!
                    }
                    continue; // Deleted, check next
                }

                // Check chain for older visible versions
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());

                while let Some(e) = current {
                    if checker.is_visible(e.version.txn_id, txn_id) {
                        if e.version.deleted_at_txn_id == 0
                            || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
                        {
                            return Some(row_id); // Found visible version - conflict!
                        }
                        break; // Deleted, move to next row
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }
        None
    }

    /// Gets multiple visible versions in a single batch operation
    ///
    /// Optimized to use arena-based storage for zero-copy reading,
    /// with pre-acquired locks for minimal overhead.
    ///
    /// Uses parallel processing for large batches (>1000 row_ids) to leverage
    /// multiple CPU cores for visibility checking and row materialization.
    ///
    /// OPTIMIZATION: Uses arena fast path (O(1) HashMap) before falling back to
    /// BTreeMap (O(log n)) for version chain traversal.
    pub fn get_visible_versions_batch(&self, row_ids: &[i64], txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire all locks ONCE for all lookups
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();
        let arena_meta = arena_guard.meta();
        let row_index = self.row_arena_index.read();
        // BTreeMap for fallback when HEAD not visible (Snapshot Isolation, AS OF)
        let versions = self.versions.read();

        // Sequential processing - avoids Rayon allocation overhead
        let mut results = RowVec::with_capacity(row_ids.len());
        let arena_len = arena_meta.len();

        for &row_id in row_ids {
            // ARENA FAST PATH: O(1) HashMap + direct array access
            if let Some(&arena_idx) = row_index.get(&row_id) {
                if arena_idx < arena_len {
                    // SAFETY: bounds checked above, unchecked access for speed
                    let meta = &arena_meta[arena_idx];

                    // Check HEAD visibility
                    if checker.is_visible(meta.txn_id, txn_id) {
                        // HEAD visible - check deletion
                        if meta.deleted_at_txn_id == 0
                            || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                        {
                            // O(1) success! Return from arena
                            results.push((
                                row_id,
                                Row::from_arc_slice(Arc::clone(&arena_data[arena_idx])),
                            ));
                        }
                        // Else: deleted, skip
                        continue;
                    }
                    // HEAD not visible - fall through to BTreeMap
                }
            }

            // BTREE FALLBACK: Version chain traversal for Snapshot Isolation
            if let Some(chain) = versions.get(&row_id) {
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        // Get from arena if available, else clone
                        let row = if let Some(idx) = chain.arena_idx {
                            if let Some(arc_row) = arena_data.get(idx) {
                                Row::from_arc_slice(Arc::clone(arc_row))
                            } else {
                                chain.version.data.clone()
                            }
                        } else {
                            chain.version.data.clone()
                        };
                        results.push((row_id, row));
                    }
                    continue;
                }

                // Traverse version chain
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());

                while let Some(e) = current {
                    if checker.is_visible(e.version.txn_id, txn_id) {
                        if e.version.deleted_at_txn_id == 0
                            || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
                        {
                            let row = if let Some(idx) = e.arena_idx {
                                if let Some(arc_row) = arena_data.get(idx) {
                                    Row::from_arc_slice(Arc::clone(arc_row))
                                } else {
                                    e.version.data.clone()
                                }
                            } else {
                                e.version.data.clone()
                            };
                            results.push((row_id, row));
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }
        results
    }

    /// Counts visible versions for batch operations (COUNT optimization)
    ///
    /// This is an optimized version of get_visible_versions_batch that only counts
    /// visible rows without cloning their data. Used for COUNT(*) subqueries.
    ///
    /// Uses parallel processing for large batches (>1000 row_ids) to leverage
    /// multiple CPU cores for visibility checking.
    pub fn count_visible_versions_batch(&self, row_ids: &[i64], txn_id: i64) -> usize {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0,
        };

        // Parallel processing constants
        /// Minimum batch size before enabling parallel processing (avoid overhead for small batches)
        const PARALLEL_THRESHOLD: usize = 1000;
        /// Chunk size for parallel processing (balances parallelism vs. overhead)
        const PARALLEL_CHUNK_SIZE: usize = 512;

        // BTreeMap with RwLock - hold read lock during parallel processing
        // Read locks are shared, so concurrent readers can proceed
        let versions = self.versions.read();

        if row_ids.len() >= PARALLEL_THRESHOLD {
            // Parallel path: process chunks while holding read lock (zero-copy)
            // RwLock read locks allow concurrent readers, so this is safe and fast
            row_ids
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .map(|chunk| {
                    let mut chunk_count = 0;
                    for &row_id in chunk {
                        if let Some(chain) = versions.get(&row_id) {
                            // FAST PATH: Check HEAD version first - O(1) for common case
                            let head_txn_id = chain.version.txn_id;
                            let head_deleted_at = chain.version.deleted_at_txn_id;

                            if checker.is_visible(head_txn_id, txn_id) {
                                // HEAD is visible - check if deleted
                                if head_deleted_at == 0
                                    || !checker.is_visible(head_deleted_at, txn_id)
                                {
                                    chunk_count += 1;
                                }
                                continue;
                            }

                            // SLOW PATH: HEAD not visible - traverse chain for older versions
                            let mut current: Option<&VersionChainEntry> =
                                chain.prev.as_ref().map(|b| b.as_ref());

                            while let Some(e) = current {
                                let version_txn_id = e.version.txn_id;
                                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                                if checker.is_visible(version_txn_id, txn_id) {
                                    if deleted_at_txn_id == 0
                                        || !checker.is_visible(deleted_at_txn_id, txn_id)
                                    {
                                        chunk_count += 1;
                                    }
                                    break;
                                }
                                current = e.prev.as_ref().map(|b| b.as_ref());
                            }
                        }
                    }
                    chunk_count
                })
                .sum()
        } else {
            // Sequential path for small batches
            let mut count = 0;
            for &row_id in row_ids {
                if let Some(chain) = versions.get(&row_id) {
                    // FAST PATH: Check HEAD version first - O(1) for common case
                    let head_txn_id = chain.version.txn_id;
                    let head_deleted_at = chain.version.deleted_at_txn_id;

                    if checker.is_visible(head_txn_id, txn_id) {
                        // HEAD is visible - check if deleted
                        if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                            count += 1;
                        }
                        continue;
                    }

                    // SLOW PATH: HEAD not visible - traverse chain for older versions
                    let mut current: Option<&VersionChainEntry> =
                        chain.prev.as_ref().map(|b| b.as_ref());

                    while let Some(e) = current {
                        let version_txn_id = e.version.txn_id;
                        let deleted_at_txn_id = e.version.deleted_at_txn_id;

                        if checker.is_visible(version_txn_id, txn_id) {
                            if deleted_at_txn_id == 0
                                || !checker.is_visible(deleted_at_txn_id, txn_id)
                            {
                                count += 1;
                            }
                            break;
                        }
                        current = e.prev.as_ref().map(|b| b.as_ref());
                    }
                }
            }
            count
        }
    }

    /// Gets visible versions for batch update operations
    ///
    /// Returns (row_id, row_data, original_version) for each visible row.
    /// The original_version is used for write-set tracking to avoid redundant lookups.
    ///
    /// This is optimized for UPDATE operations where we need to:
    /// 1. Read the current row data
    /// 2. Track the original version for conflict detection
    /// 3. Skip redundant lookups during put
    pub fn get_visible_versions_for_update(
        &self,
        row_ids: &[i64],
        txn_id: i64,
    ) -> Vec<(i64, Row, RowVersion)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Pre-acquire arena lock for O(1) Arc clones
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper: get row from arena (O(1)) or version (O(n) clone)
        let get_row = |entry: &VersionChainEntry| -> Row {
            if let Some(idx) = entry.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            entry.version.data.clone()
        };

        let current_seq = checker.get_current_sequence();
        let mut results = Vec::with_capacity(row_ids.len());

        let versions = self.versions.read();
        for &row_id in row_ids {
            if let Some(chain) = versions.get(&row_id) {
                // FAST PATH: Check HEAD version first - O(1) for common case
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    // HEAD is visible - check if deleted
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        let mut version_copy = chain.version.clone();
                        version_copy.create_time = current_seq;
                        results.push((row_id, get_row(chain), version_copy));
                    }
                    continue;
                }

                // SLOW PATH: HEAD not visible - traverse chain for older versions
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            // Create a RowVersion with the correct read_version_seq for tracking
                            let mut version_copy = e.version.clone();
                            // Store the current sequence in create_time for later retrieval
                            // (This is a bit of a hack, but avoids changing the struct)
                            version_copy.create_time = current_seq;
                            results.push((row_id, get_row(e), version_copy));
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }

        results
    }

    /// Gets the current sequence number for write-set tracking
    pub fn get_current_sequence(&self) -> i64 {
        self.visibility_checker
            .as_ref()
            .map(|c| c.get_current_sequence())
            .unwrap_or(0)
    }

    /// Gets the visible version as of a specific transaction
    pub fn get_visible_version_as_of_transaction(
        &self,
        row_id: i64,
        as_of_txn_id: i64,
    ) -> Option<RowVersion> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let versions = self.versions.read();
        let chain = versions.get(&row_id)?;

        // Traverse version chain from newest to oldest
        let mut current: Option<&VersionChainEntry> = Some(chain);
        while let Some(e) = current {
            // Check if this version was created before or at the asOf transaction
            if e.version.txn_id <= as_of_txn_id {
                // Check if deleted before or at asOfTxnID
                if e.version.deleted_at_txn_id != 0 && e.version.deleted_at_txn_id <= as_of_txn_id {
                    return None;
                }
                return Some(e.version.clone());
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
        }

        None
    }

    /// Gets the visible version as of a specific timestamp
    pub fn get_visible_version_as_of_timestamp(
        &self,
        row_id: i64,
        as_of_timestamp: i64,
    ) -> Option<RowVersion> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let versions = self.versions.read();
        let chain = versions.get(&row_id)?;

        // Traverse version chain from newest to oldest
        let mut current: Option<&VersionChainEntry> = Some(chain);
        while let Some(e) = current {
            // Check if this version was created before or at the asOf timestamp
            if e.version.create_time <= as_of_timestamp {
                // Check if deleted (we can't easily determine timestamp of deletion in this model)
                // For now, check if DeletedAtTxnID is set
                if e.version.deleted_at_txn_id != 0 {
                    return None;
                }
                return Some(e.version.clone());
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
        }

        None
    }

    /// Returns all row IDs in the version store (already sorted by BTreeMap)
    pub fn get_all_row_ids(&self) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        self.versions.read().keys().copied().collect()
    }

    /// Returns all row IDs that are visible to the given transaction
    ///
    /// OPTIMIZATION: Single-pass iteration with O(1) lock acquisition instead of O(N).
    /// Uses the same HEAD-first visibility pattern as count_visible_rows.
    pub fn get_all_visible_row_ids(&self, txn_id: i64) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Single pass through all versions (holding read lock once)
        let versions = self.versions.read();
        let mut visible_row_ids = Vec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    visible_row_ids.push(row_id);
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        visible_row_ids.push(row_id);
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        visible_row_ids
    }

    /// Count visible non-deleted rows in a single pass (optimized for row_count)
    ///
    /// OPTIMIZATION: This method counts rows in O(1) lock acquisition instead of O(N)
    /// by iterating through versions once without cloning any row data.
    pub fn count_visible_rows(&self, txn_id: i64) -> usize {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0,
        };

        let mut count = 0;

        // Single pass through all versions (holding read lock)
        let versions = self.versions.read();
        for chain in versions.values() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    count += 1;
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        count += 1;
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        count
    }

    /// Returns all visible rows for a transaction (optimized batch operation)
    ///
    /// This is more efficient than calling get_visible_version for each row
    /// because it batches the visibility checks and avoids repeated map lookups.
    /// Results are already sorted by row_id (BTreeMap iteration order).
    #[inline]
    pub fn get_all_visible_rows(&self, txn_id: i64) -> RowVec {
        // BTreeMap is already sorted, no additional sorting needed!
        self.get_all_visible_rows_internal(txn_id)
    }

    /// Internal implementation for getting all visible rows
    #[inline]
    fn get_all_visible_rows_internal(&self, txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire arena lock for O(1) Arc clones
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper: get row from arena (O(1)) or version (O(n) clone)
        let get_row = |entry: &VersionChainEntry| -> Row {
            if let Some(idx) = entry.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            entry.version.data.clone()
        };

        // Collect all versions in one pass (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut results = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    results.push((row_id, get_row(chain)));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        results.push((row_id, get_row(e)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // BTreeMap already iterates in sorted order, no need to sort
        results
    }

    /// Returns all visible rows using arena for zero-copy scanning
    ///
    /// This method provides 50x+ faster full table scans by:
    /// 1. Pre-acquiring arena locks once
    /// 2. Reading directly during visibility iteration (single pass)
    /// 3. Using contiguous arena memory for cache locality
    #[inline]
    pub fn get_all_visible_rows_arena(&self, txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    result.push((row_id, get_row_data(chain)));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        result.push((row_id, get_row_data(e)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // BTreeMap already iterates in sorted order, no need to sort
        result
    }

    /// Returns all visible rows using RowVec for zero-allocation reuse.
    ///
    /// Same as `get_all_visible_rows_arena` but uses cached RowVec.
    /// The returned `RowVec` auto-returns to cache on drop.
    #[inline]
    pub fn get_all_visible_rows_cached(&self, txn_id: i64) -> RowVec {
        let mut result = RowVec::new();

        if self.closed.load(Ordering::Acquire) {
            return result;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return result,
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check
        let versions = self.versions.read();

        // Ensure capacity
        let current_capacity = result.capacity();
        let needed = versions.len();
        if current_capacity < needed {
            result.reserve(needed - current_capacity);
        }

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    result.push((row_id, get_row_data(chain)));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        result.push((row_id, get_row_data(e)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        result
    }

    /// Returns all visible rows with their original RowVersions for UPDATE operations.
    ///
    /// This is optimized for UPDATE operations that need to track the original version
    /// for conflict detection. By returning the RowVersion along with the row data,
    /// callers can use `put_batch_with_originals()` to avoid redundant `get_visible_version()`
    /// calls during the put phase.
    ///
    /// Returns: Vec of (row_id, row_data, original_version) tuples
    /// The original_version.create_time contains the read_version_seq for conflict detection.
    #[inline]
    pub fn get_all_visible_rows_for_update(&self, txn_id: i64) -> Vec<(i64, Row, RowVersion)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        let current_seq = checker.get_current_sequence();

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check (BTreeMap)
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row, RowVersion)> = Vec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    let mut version_copy = chain.version.clone();
                    version_copy.create_time = current_seq;
                    result.push((row_id, get_row_data(chain), version_copy));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        let mut version_copy = e.version.clone();
                        version_copy.create_time = current_seq;
                        result.push((row_id, get_row_data(e), version_copy));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        result
    }

    /// Get all visible rows for UPDATE with filter applied BEFORE cloning.
    ///
    /// This is a performance optimization for UPDATE operations with WHERE clauses.
    /// Instead of fetching all rows and then filtering, we filter during the scan
    /// to avoid allocating Row objects for non-matching rows.
    ///
    /// Returns (row_id, Row, RowVersion) tuples - the RowVersion is needed for
    /// MVCC conflict detection during the update.
    pub fn get_all_visible_rows_for_update_filtered(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
    ) -> Vec<(i64, Row, RowVersion)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        let current_seq = checker.get_current_sequence();

        // Compile the filter once at the start for speedup in hot loop
        let schema = self.schema.read();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema); // Release lock early

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Single-pass: read, filter, and collect in one loop (BTreeMap)
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row, RowVersion)> = Vec::with_capacity(versions.len() / 4);

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    // Try arena path first (zero-copy filter check using matches_arc_slice)
                    if let Some(idx) = chain.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter directly on Arc slice - no Row allocation for non-matching rows
                            if compiled_filter.matches_arc_slice(arc_row.as_ref()) {
                                let mut version_copy = chain.version.clone();
                                version_copy.create_time = current_seq;
                                result.push((
                                    row_id,
                                    Row::from_arc_slice(Arc::clone(arc_row)),
                                    version_copy,
                                ));
                            }
                            continue;
                        }
                    }
                    // Fallback: filter on version data
                    if compiled_filter.matches(&chain.version.data) {
                        let mut version_copy = chain.version.clone();
                        version_copy.create_time = current_seq;
                        result.push((row_id, chain.version.data.clone(), version_copy));
                    }
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        // Try arena path first (zero-copy filter check using matches_arc_slice)
                        if let Some(idx) = e.arena_idx {
                            if let Some(arc_row) = arena_data.get(idx) {
                                // Filter directly on Arc slice - no Row allocation for non-matching rows
                                if compiled_filter.matches_arc_slice(arc_row.as_ref()) {
                                    let mut version_copy = e.version.clone();
                                    version_copy.create_time = current_seq;
                                    result.push((
                                        row_id,
                                        Row::from_arc_slice(Arc::clone(arc_row)),
                                        version_copy,
                                    ));
                                }
                                break;
                            }
                        }
                        // Fallback: filter on version data
                        if compiled_filter.matches(&e.version.data) {
                            let mut version_copy = e.version.clone();
                            version_copy.create_time = current_seq;
                            result.push((row_id, e.version.data.clone(), version_copy));
                        }
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        result
    }

    /// Returns all visible rows.
    ///
    /// Note: With BTreeMap, this is functionally identical to get_all_visible_rows_arena
    /// since BTreeMap iteration is inherently ordered.
    #[inline]
    pub fn get_all_visible_rows_unsorted(&self, txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    result.push((row_id, get_row_data(chain)));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        result.push((row_id, get_row_data(e)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        result
    }

    /// Get visible rows with limit and offset applied at the storage layer.
    ///
    /// # True Early Termination
    /// With BTreeMap, iteration is already ordered by row_id. We can now do true
    /// early termination: skip `offset` visible rows, then collect `limit` rows
    /// and stop iterating.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility check
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of rows to skip before collecting
    pub fn get_visible_rows_with_limit(&self, txn_id: i64, limit: usize, offset: usize) -> RowVec {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap is sorted - collect with early termination
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(limit);
        let mut skipped = 0usize;

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            // Track the actual visible entry, not just a boolean
            let visible_entry: Option<&VersionChainEntry> = if checker
                .is_visible(head_txn_id, txn_id)
            {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    Some(chain)
                } else {
                    None
                }
            } else {
                // SLOW PATH: HEAD not visible - traverse chain for older versions
                let mut found_entry: Option<&VersionChainEntry> = None;
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            found_entry = Some(e);
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
                found_entry
            };

            if let Some(entry) = visible_entry {
                if skipped < offset {
                    skipped += 1;
                } else {
                    result.push((row_id, get_row_data(entry)));
                    if result.len() >= limit {
                        break; // Early termination!
                    }
                }
            }
        }

        result
    }

    /// Get visible rows with LIMIT (with early termination).
    ///
    /// Note: With BTreeMap, this is functionally identical to get_visible_rows_with_limit
    /// since BTreeMap iteration is inherently ordered by row_id. Kept for API compatibility.
    #[inline]
    pub fn get_visible_rows_with_limit_unordered(
        &self,
        txn_id: i64,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        // Delegate to the sorted version - BTreeMap is always sorted
        self.get_visible_rows_with_limit(txn_id, limit, offset)
    }

    /// Get a batch of visible rows starting after a given row_id (cursor-based pagination).
    ///
    /// This is designed for lazy/streaming scanners that fetch rows in batches.
    /// Uses BTreeMap's range() for efficient cursor-based iteration.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility check
    /// * `after_row_id` - Start fetching rows AFTER this row_id (use i64::MIN to start from beginning)
    /// * `batch_size` - Maximum number of rows to return in this batch
    ///
    /// # Returns
    /// A tuple of (rows, has_more) where has_more indicates if there are more rows to fetch
    pub fn get_visible_rows_batch(
        &self,
        txn_id: i64,
        after_row_id: i64,
        batch_size: usize,
    ) -> (RowVec, bool) {
        if self.closed.load(Ordering::Acquire) || batch_size == 0 {
            return (RowVec::new(), false);
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return (RowVec::new(), false),
        };

        // Pre-acquire arena lock ONCE for this batch
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap is sorted - use range for efficient cursor-based iteration
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(batch_size);
        let mut has_more = false;

        // Use range to start after the cursor row_id
        for (&row_id, chain) in versions.range((
            std::ops::Bound::Excluded(after_row_id),
            std::ops::Bound::Unbounded::<i64>,
        )) {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            // Track the actual visible entry, not just a boolean
            let visible_entry: Option<&VersionChainEntry> = if checker
                .is_visible(head_txn_id, txn_id)
            {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    Some(chain)
                } else {
                    None
                }
            } else {
                // SLOW PATH: HEAD not visible - traverse chain for older versions
                let mut found_entry: Option<&VersionChainEntry> = None;
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            found_entry = Some(e);
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
                found_entry
            };

            if let Some(entry) = visible_entry {
                if result.len() >= batch_size {
                    has_more = true;
                    break; // Early termination - found one more than needed
                }
                result.push((row_id, get_row_data(entry)));
            }
        }

        (result, has_more)
    }

    /// Fetch visible rows into an existing buffer (avoids allocation)
    ///
    /// This is the same as `get_visible_rows_batch` but reuses the provided buffer
    /// instead of allocating a new Vec. The buffer is cleared before filling.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility checks
    /// * `after_row_id` - Cursor position (exclusive lower bound)
    /// * `batch_size` - Maximum number of rows to fetch
    /// * `buffer` - Existing buffer to fill (will be cleared first)
    ///
    /// # Returns
    /// `has_more` - true if there are more rows to fetch after this batch
    pub fn get_visible_rows_batch_into(
        &self,
        txn_id: i64,
        after_row_id: i64,
        batch_size: usize,
        buffer: &mut RowVec,
    ) -> bool {
        buffer.clear();

        if self.closed.load(Ordering::Acquire) || batch_size == 0 {
            return false;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return false,
        };

        // Pre-acquire arena lock ONCE for this batch
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap is sorted - use range for efficient cursor-based iteration
        let versions = self.versions.read();
        buffer.reserve(batch_size);
        let mut has_more = false;

        // Use range to start after the cursor row_id
        for (&row_id, chain) in versions.range((
            std::ops::Bound::Excluded(after_row_id),
            std::ops::Bound::Unbounded::<i64>,
        )) {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            // Track the actual visible entry, not just a boolean
            let visible_entry: Option<&VersionChainEntry> = if checker
                .is_visible(head_txn_id, txn_id)
            {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    Some(chain)
                } else {
                    None
                }
            } else {
                // SLOW PATH: HEAD not visible - traverse chain for older versions
                let mut found_entry: Option<&VersionChainEntry> = None;
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            found_entry = Some(e);
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
                found_entry
            };

            if let Some(entry) = visible_entry {
                if buffer.len() >= batch_size {
                    has_more = true;
                    break; // Early termination - found one more than needed
                }
                buffer.push((row_id, get_row_data(entry)));
            }
        }

        has_more
    }

    /// Collect visible rows ordered by row_id (PRIMARY KEY) with efficient OFFSET/LIMIT
    ///
    /// This method uses BTreeMap range iteration to efficiently skip OFFSET rows
    /// without cloning them, providing O(offset + limit) complexity instead of
    /// O(n) for full materialization.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility checks
    /// * `ascending` - If true, iterate from smallest to largest row_id
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of visible rows to skip before collecting
    ///
    /// # Returns
    /// Vector of rows in row_id order, or None if iteration fails
    pub fn collect_rows_pk_ordered(
        &self,
        txn_id: i64,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<RowVec> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Some(RowVec::new());
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Some(RowVec::new()),
        };

        // Pre-acquire arena lock ONCE for this entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper to get row data from an entry
        let get_row_from_entry = |entry: &VersionChainEntry| -> Row {
            if let Some(idx) = entry.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc_slice(Arc::clone(arc_row));
                }
            }
            entry.version.data.clone()
        };

        // BTreeMap is already sorted ascending - collect with offset/limit
        let versions = self.versions.read();
        // Cap capacity to avoid overflow when limit is usize::MAX
        let capacity = limit.min(versions.len()).min(10_000);
        let mut result = RowVec::with_capacity(capacity);
        let mut skipped = 0usize;

        if ascending {
            // Forward iteration with early termination
            for (row_id, chain) in versions.iter() {
                // Inline visibility check
                let mut current: Option<&VersionChainEntry> = Some(chain);
                while let Some(entry) = current {
                    if checker.is_visible(entry.version.txn_id, txn_id) {
                        if entry.version.deleted_at_txn_id == 0
                            || !checker.is_visible(entry.version.deleted_at_txn_id, txn_id)
                        {
                            // Found visible entry
                            if skipped < offset {
                                skipped += 1;
                            } else {
                                result.push((*row_id, get_row_from_entry(entry)));
                                if result.len() >= limit {
                                    return Some(result);
                                }
                            }
                        }
                        break;
                    }
                    current = entry.prev.as_ref().map(|b| b.as_ref());
                }
            }
        } else {
            // For descending order, we need to collect all, then reverse
            let mut all_visible = RowVec::with_capacity(versions.len());
            for (row_id, chain) in versions.iter() {
                // Inline visibility check
                let mut current: Option<&VersionChainEntry> = Some(chain);
                while let Some(entry) = current {
                    if checker.is_visible(entry.version.txn_id, txn_id) {
                        if entry.version.deleted_at_txn_id == 0
                            || !checker.is_visible(entry.version.deleted_at_txn_id, txn_id)
                        {
                            all_visible.push((*row_id, get_row_from_entry(entry)));
                        }
                        break;
                    }
                    current = entry.prev.as_ref().map(|b| b.as_ref());
                }
            }
            // Reverse and apply offset/limit
            for item in all_visible.into_iter().rev().skip(offset).take(limit) {
                result.push(item);
            }
        }

        Some(result)
    }

    /// Collect visible rows using keyset pagination (WHERE id > X ORDER BY id LIMIT Y)
    ///
    /// This method uses BTreeMap range iteration starting from a specific row_id,
    /// providing O(limit) complexity instead of O(n) for full table scans.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility checks
    /// * `start_after_row_id` - Start iteration after this row_id (exclusive, for id > X)
    /// * `start_from_row_id` - Start iteration from this row_id (inclusive, for id >= X)
    /// * `ascending` - If true, iterate from smallest to largest row_id
    /// * `limit` - Maximum number of rows to return
    ///
    /// # Returns
    /// RowVec of (row_id, row) pairs in row_id order
    pub fn collect_rows_keyset(
        &self,
        txn_id: i64,
        start_after_row_id: Option<i64>,
        start_from_row_id: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> RowVec {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Pre-acquire arena lock ONCE for this entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper to find visible version and get row data
        let find_visible_row = |chain: &VersionChainEntry| -> Option<Row> {
            let mut current: Option<&VersionChainEntry> = Some(chain);
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Read row data from arena or version
                    let row_data = if let Some(idx) = e.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            Row::from_arc_slice(Arc::clone(arc_row))
                        } else {
                            e.version.data.clone()
                        }
                    } else {
                        e.version.data.clone()
                    };
                    return Some(row_data);
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
            None
        };

        // Determine range bounds
        let start_bound = if let Some(after_id) = start_after_row_id {
            std::ops::Bound::Excluded(after_id)
        } else if let Some(from_id) = start_from_row_id {
            std::ops::Bound::Included(from_id)
        } else {
            std::ops::Bound::Unbounded
        };

        // BTreeMap is already sorted - collect with early termination
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(limit);

        if ascending {
            for (&row_id, chain) in versions.range((start_bound, std::ops::Bound::Unbounded::<i64>))
            {
                if let Some(row_data) = find_visible_row(chain) {
                    result.push((row_id, row_data));
                    if result.len() >= limit {
                        break;
                    }
                }
            }
        } else {
            // For descending, we need to collect all in range then reverse
            let mut all_visible = RowVec::new();
            for (&row_id, chain) in versions.range((start_bound, std::ops::Bound::Unbounded::<i64>))
            {
                if let Some(row_data) = find_visible_row(chain) {
                    all_visible.push((row_id, row_data));
                }
            }
            // Reverse iteration with limit
            for item in all_visible.into_iter().rev().take(limit) {
                result.push(item);
            }
        }

        result
    }

    /// Get all visible rows with filter applied during collection
    /// This saves memory by not allocating space for non-matching rows
    ///
    /// # Performance
    ///
    /// The filter expression is compiled into a `CompiledFilter` at the start
    /// to eliminate virtual dispatch overhead in the hot loop. This provides
    /// ~3-5x speedup for filter-heavy queries.
    pub fn get_all_visible_rows_filtered(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
    ) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Compile the filter once at the start for ~3-5x speedup in hot loop
        // CompiledFilter eliminates virtual dispatch via enum-based specialization
        let schema = self.schema.read();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema); // Release lock early

        // Pre-acquire arena lock ONCE
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Single-pass: read, filter, and collect in one loop (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // OPTIMIZATION: Filter BEFORE cloning to avoid allocation for non-matching rows
                    // Try arena path first (zero-copy filter check using matches_arc_slice)
                    if let Some(idx) = e.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter directly on Arc slice - no Row allocation for non-matching rows
                            if compiled_filter.matches_arc_slice(arc_row.as_ref()) {
                                result.push((row_id, Row::from_arc_slice(Arc::clone(arc_row))));
                            }
                            break;
                        }
                    }
                    // Fallback: filter on version data (already allocated)
                    if compiled_filter.matches(&e.version.data) {
                        result.push((row_id, e.version.data.clone()));
                    }
                    break;
                }
                current = e.prev.as_deref();
            }
        }

        // BTreeMap already iterates in sorted order
        result
    }

    /// Get visible rows with filter, limit and offset applied at the storage layer.
    ///
    /// # True Early Termination
    /// With BTreeMap, we get ordered iteration. After collecting `limit` matching rows
    /// (after skipping `offset`), we can stop iterating.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility check
    /// * `filter` - Expression filter to apply to rows
    /// * `limit` - Maximum number of matching rows to return
    /// * `offset` - Number of matching rows to skip before collecting
    pub fn get_visible_rows_filtered_with_limit(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Compile the filter once at the start
        let schema = self.schema.read();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema);

        // Pre-acquire arena lock ONCE
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // BTreeMap is already sorted - collect with offset/limit and early termination
        let versions = self.versions.read();
        let mut result = RowVec::with_capacity(limit);
        let mut skipped = 0usize;

        for (&row_id, chain) in versions.iter() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // OPTIMIZATION: Filter BEFORE cloning to avoid allocation for non-matching rows
                    // Try arena path first (zero-copy filter check using matches_arc_slice)
                    if let Some(idx) = e.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter directly on Arc slice - no Row allocation for non-matching rows
                            if compiled_filter.matches_arc_slice(arc_row.as_ref()) {
                                if skipped < offset {
                                    skipped += 1;
                                } else {
                                    result.push((row_id, Row::from_arc_slice(Arc::clone(arc_row))));
                                    if result.len() >= limit {
                                        return result; // Early termination!
                                    }
                                }
                            }
                            break;
                        }
                    }
                    // Fallback: filter on version data (already allocated)
                    if compiled_filter.matches(&e.version.data) {
                        if skipped < offset {
                            skipped += 1;
                        } else {
                            result.push((row_id, e.version.data.clone()));
                            if result.len() >= limit {
                                return result; // Early termination!
                            }
                        }
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        result
    }

    /// Get visible rows with filter and LIMIT (with early termination).
    ///
    /// Note: With BTreeMap, this is functionally identical to get_visible_rows_filtered_with_limit
    /// since BTreeMap iteration is inherently ordered by row_id. Kept for API compatibility.
    #[inline]
    pub fn get_visible_rows_filtered_with_limit_unordered(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        // Delegate to the sorted version - BTreeMap is always sorted
        self.get_visible_rows_filtered_with_limit(txn_id, filter, limit, offset)
    }

    /// Get visible row indices without materializing row data (ZERO ALLOCATION SCAN!)
    ///
    /// This method returns lightweight `RowIndex` structs instead of cloning row data.
    /// Callers can then filter/sort/limit these indices and only materialize the final
    /// set of rows needed using `materialize_rows()`.
    ///
    /// # Performance
    /// For `SELECT * FROM t WHERE x > 100 LIMIT 10` on 100K rows:
    /// - Old approach: Clone 100K rows, filter, limit (100K allocations)
    /// - New approach: Get 100K indices (0 allocations), filter, limit, clone 10 rows
    ///
    /// # Returns
    /// Vector of `RowIndex` structs containing row_id and arena location
    pub fn get_visible_row_indices(&self, txn_id: i64) -> Vec<RowIndex> {
        let checker = self.visibility_checker.as_ref();

        // Get arena metadata for fast path detection
        let arena_guard = self.arena.read_guard();
        let arena_meta = arena_guard.meta();
        let arena_len = arena_guard.len();

        // FAST PATH: If uncommitted_writes is empty, scan arena directly
        if let Some(checker) = checker {
            let uncommitted_empty = self.uncommitted_writes.read().is_empty();

            if uncommitted_empty && arena_len > 0 {
                let mut indices: Vec<RowIndex> = Vec::with_capacity(arena_len);

                for (idx, meta) in arena_meta.iter().enumerate() {
                    if meta.deleted_at_txn_id == 0 && checker.is_visible(meta.txn_id, txn_id) {
                        indices.push(RowIndex {
                            row_id: meta.row_id,
                            arena_idx: Some(idx),
                        });
                    }
                }

                // Sort by row_id for consistent ordering
                indices.sort_unstable_by_key(|idx| idx.row_id);
                return indices;
            }
        }

        // Drop arena guard before slow path to avoid holding locks
        drop(arena_guard);

        // SLOW PATH: Full iteration (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut indices: Vec<RowIndex> = Vec::with_capacity(versions.len());

        if let Some(checker) = checker {
            for (&row_id, chain) in versions.iter() {
                let mut current: Option<&VersionChainEntry> = Some(chain);

                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            // Found visible, non-deleted version
                            indices.push(RowIndex {
                                row_id,
                                arena_idx: e.arena_idx,
                            });
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }

        // BTreeMap already iterates in sorted order
        indices
    }

    /// Get visible row indices without guaranteed ordering.
    ///
    /// This is optimized for aggregation operations (SUM, MIN, MAX, COUNT) that
    /// don't need row ordering. It skips the expensive sort in the arena fast path.
    #[inline]
    pub fn get_visible_row_indices_unordered(&self, txn_id: i64) -> Vec<RowIndex> {
        let checker = self.visibility_checker.as_ref();

        // Get arena metadata for fast path detection
        let arena_guard = self.arena.read_guard();
        let arena_meta = arena_guard.meta();
        let arena_len = arena_guard.len();

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (NO SORT!)
        if let Some(checker) = checker {
            let uncommitted_empty = self.uncommitted_writes.read().is_empty();

            if uncommitted_empty && arena_len > 0 {
                let mut indices: Vec<RowIndex> = Vec::with_capacity(arena_len);

                for (idx, meta) in arena_meta.iter().enumerate() {
                    if meta.deleted_at_txn_id == 0 && checker.is_visible(meta.txn_id, txn_id) {
                        indices.push(RowIndex {
                            row_id: meta.row_id,
                            arena_idx: Some(idx),
                        });
                    }
                }

                // Skip sorting - aggregations don't need ordering
                return indices;
            }
        }

        // Drop arena guard before slow path to avoid holding locks
        drop(arena_guard);

        // SLOW PATH: Full iteration (BTreeMap is sorted but we don't need it)
        let versions = self.versions.read();
        let mut indices: Vec<RowIndex> = Vec::with_capacity(versions.len());

        if let Some(checker) = checker {
            for (&row_id, chain) in versions.iter() {
                let mut current: Option<&VersionChainEntry> = Some(chain);

                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            indices.push(RowIndex {
                                row_id,
                                arena_idx: e.arena_idx,
                            });
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }

        indices
    }

    /// Materialize selected row indices into actual Row data
    ///
    /// This is the second step of deferred materialization. After filtering/limiting
    /// `RowIndex` values, call this to get the actual row data.
    ///
    /// # Performance
    /// - Only clones the rows you actually need
    /// - Falls back to version chain for non-arena rows
    pub fn materialize_rows(&self, indices: &[RowIndex]) -> RowVec {
        if indices.is_empty() {
            return RowVec::new();
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        let mut result = RowVec::with_capacity(indices.len());

        for idx in indices {
            if let Some(arena_idx) = idx.arena_idx {
                // Fast path: get from arena
                if arena_idx < arena_len {
                    if let Some(arc_row) = arena_data.get(arena_idx) {
                        result.push((idx.row_id, Row::from_arc_slice(Arc::clone(arc_row))));
                        continue;
                    }
                }
            }

            // Slow path: look up in version chain (lock already held)
            if let Some(entry) = versions.get(&idx.row_id) {
                result.push((idx.row_id, entry.version.data.clone()));
            }
        }

        result
    }

    /// Materialize a single row by index (for use in iterators)
    #[inline]
    pub fn materialize_row(&self, idx: &RowIndex) -> Option<(i64, Row)> {
        if let Some(arena_idx) = idx.arena_idx {
            let arena_guard = self.arena.read_guard();
            let arena_data = arena_guard.data();
            if let Some(arc_row) = arena_data.get(arena_idx) {
                return Some((idx.row_id, Row::from_arc_slice(Arc::clone(arc_row))));
            }
        }

        // Slow path: look up in version chain
        let versions = self.versions.read();
        versions
            .get(&idx.row_id)
            .map(|chain| (idx.row_id, chain.version.data.clone()))
    }

    /// Get a single column value from a row index WITHOUT full row materialization
    ///
    /// This is the key optimization for ORDER BY + LIMIT queries:
    /// - Load only the sort column, not all columns
    /// - Enables sorting indices by column value before materializing
    ///
    /// # Performance
    /// For `SELECT * FROM t ORDER BY col LIMIT 10` on 100K rows with 20 columns:
    /// - Old: Clone 100K rows (2M values), sort, take 10
    /// - New: Load 100K single values, sort indices, clone 10 rows (200 values)
    #[inline]
    pub fn get_column_value(&self, idx: &RowIndex, col_idx: usize) -> Option<Value> {
        if let Some(arena_idx) = idx.arena_idx {
            let arena_guard = self.arena.read_guard();
            let arena_data = arena_guard.data();
            if let Some(arc_row) = arena_data.get(arena_idx) {
                return arc_row.get(col_idx).map(|arc| arc.as_ref().clone());
            }
        }

        // Slow path: look up in version chain
        let versions = self.versions.read();
        versions
            .get(&idx.row_id)
            .and_then(|entry| entry.version.data.get(col_idx).cloned())
    }

    /// Batch get column values for multiple indices
    ///
    /// Optimized for ORDER BY: loads sort key values for all indices at once.
    #[inline]
    pub fn get_column_values_batch(
        &self,
        indices: &[RowIndex],
        col_idx: usize,
    ) -> Vec<Option<Value>> {
        if indices.is_empty() {
            return Vec::new();
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        let versions = self.versions.read();
        indices
            .iter()
            .map(|idx| {
                if let Some(arena_idx) = idx.arena_idx {
                    if arena_idx < arena_len {
                        if let Some(arc_row) = arena_data.get(arena_idx) {
                            return arc_row.get(col_idx).map(|arc| (**arc).clone());
                        }
                    }
                }
                // Slow path: version chain lookup
                versions
                    .get(&idx.row_id)
                    .and_then(|entry| entry.version.data.get(col_idx).cloned())
            })
            .collect()
    }

    /// Optimized ORDER BY + LIMIT scan using deferred materialization
    ///
    /// This is the FAST PATH for queries like `SELECT * FROM t ORDER BY col LIMIT 10`:
    /// 1. Get row indices (no cloning)
    /// 2. Load only sort column values (not full rows)
    /// 3. Sort indices by sort values
    /// 4. Take top N indices
    /// 5. Materialize only N rows
    ///
    /// # Performance
    /// For 100K rows with 20 columns, LIMIT 10:
    /// - Old: Clone 2M values, sort 100K rows, take 10 → ~100ms
    /// - New: Load 100K values, sort indices, clone 200 values → ~10ms
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility
    /// * `sort_col_idx` - Column index to sort by
    /// * `ascending` - Sort direction
    /// * `limit` - Maximum rows to return
    /// * `offset` - Rows to skip before collecting
    pub fn get_visible_rows_sorted_limit(
        &self,
        txn_id: i64,
        sort_col_idx: usize,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        if limit == 0 {
            return RowVec::new();
        }

        // Step 1: Get all visible row indices (no cloning!)
        let indices = self.get_visible_row_indices_unordered(txn_id);

        if indices.is_empty() {
            return RowVec::new();
        }

        // Step 2: Load only the sort column values (partial materialization)
        let sort_values = self.get_column_values_batch(&indices, sort_col_idx);

        // Step 3: Sort indices by sort values
        let mut paired: Vec<(RowIndex, Option<Value>)> =
            indices.into_iter().zip(sort_values).collect();

        paired.sort_by(
            |(_, a): &(RowIndex, Option<Value>), (_, b): &(RowIndex, Option<Value>)| {
                let cmp = match (a, b) {
                    (None, None) => std::cmp::Ordering::Equal,
                    (None, Some(_)) => std::cmp::Ordering::Less, // NULLs first
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (Some(va), Some(vb)) => va.compare(vb).unwrap_or(std::cmp::Ordering::Equal),
                };
                if ascending {
                    cmp
                } else {
                    cmp.reverse()
                }
            },
        );

        // Step 4: Take limit+offset indices, skip offset
        let selected: Vec<RowIndex> = paired
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(idx, _)| idx)
            .collect();

        // Step 5: Materialize only the selected rows
        self.materialize_rows(&selected)
    }

    /// Compute COUNT(*) without materializing any rows
    ///
    /// This is the FASTEST path for `SELECT COUNT(*) FROM table`:
    /// - No row data is loaded
    /// - Only visibility checks are performed
    /// - O(n) time, O(1) memory
    #[inline]
    pub fn count_visible(&self, txn_id: i64) -> usize {
        self.count_visible_rows(txn_id)
    }

    /// Compute SUM(column) without materializing full rows
    ///
    /// OPTIMIZATION: Single-pass approach that combines visibility checking with summing.
    /// Avoids Vec allocation and second iteration over indices.
    /// Returns (sum, count_non_null) for proper NULL handling.
    pub fn sum_column(&self, txn_id: i64, col_idx: usize) -> (f64, usize) {
        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return (0.0, 0),
        };

        let arena_guard = self.arena.read_guard();
        let arena_meta = arena_guard.meta();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        // OPTIMIZATION: Separate accumulators to avoid i64->f64 conversion per row
        let mut int_sum = 0i64;
        let mut float_sum = 0.0f64;
        let mut count = 0usize;

        // Helper to accumulate numeric value
        #[inline(always)]
        fn accumulate_sum(int_sum: &mut i64, float_sum: &mut f64, count: &mut usize, val: &Value) {
            match val {
                Value::Integer(i) => {
                    *int_sum = int_sum.wrapping_add(*i);
                    *count += 1;
                }
                Value::Float(f) => {
                    *float_sum += *f;
                    *count += 1;
                }
                _ => {} // NULL or non-numeric
            }
        }

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass)
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        if uncommitted_empty && arena_len > 0 {
            // OPTIMIZATION: Cache visibility result for repeated txn_ids
            // When rows are inserted in batches, consecutive rows often have the same txn_id.
            // Caching avoids repeated thread-local access overhead (~27ms per 100K rows).
            let mut last_txn_id: i64 = -1;
            let mut last_visible: bool = false;

            let mut idx = 0;
            while idx < arena_len {
                let meta = &arena_meta[idx];
                if meta.deleted_at_txn_id == 0 {
                    // Check visibility with cache
                    let version_txn_id = meta.txn_id;
                    let is_vis = if version_txn_id == last_txn_id {
                        last_visible
                    } else {
                        let vis = checker.is_visible(version_txn_id, txn_id);
                        last_txn_id = version_txn_id;
                        last_visible = vis;
                        vis
                    };

                    if is_vis {
                        if let Some(val) = arena_data[idx].get(col_idx) {
                            accumulate_sum(&mut int_sum, &mut float_sum, &mut count, val);
                        }
                    }
                }
                idx += 1;
            }
            return (int_sum as f64 + float_sum, count);
        }

        // Drop arena guard before slow path to avoid holding locks
        drop(arena_guard);

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.versions.read();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = -1;
        let mut last_visible: bool = false;

        for chain in versions.values() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                let is_vis = if version_txn_id == last_txn_id {
                    last_visible
                } else {
                    let vis = checker.is_visible(version_txn_id, txn_id);
                    last_txn_id = version_txn_id;
                    last_visible = vis;
                    vis
                };

                if is_vis
                    && (deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id))
                {
                    // This version is visible - accumulate its value
                    if let Some(val) = e.version.data.get(col_idx) {
                        accumulate_sum(&mut int_sum, &mut float_sum, &mut count, val);
                    }
                    break; // Found visible version, move to next row
                }

                current = e.prev.as_deref();
            }
        }

        (int_sum as f64 + float_sum, count)
    }

    /// Compute MIN(column) without materializing full rows
    ///
    /// OPTIMIZATION: Single-pass approach that combines visibility checking with min computation.
    pub fn min_column(&self, txn_id: i64, col_idx: usize) -> Option<Value> {
        let checker = self.visibility_checker.as_ref()?;

        let arena_guard = self.arena.read_guard();
        let arena_meta = arena_guard.meta();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        let mut min_val: Option<Value> = None;

        // Helper to update min value
        #[inline(always)]
        fn update_min(min_val: &mut Option<Value>, val: &Value) {
            if !val.is_null() {
                match min_val {
                    None => *min_val = Some(val.clone()),
                    Some(ref current) => {
                        if let Ok(std::cmp::Ordering::Less) = val.compare(current) {
                            *min_val = Some(val.clone());
                        }
                    }
                }
            }
        }

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass)
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        if uncommitted_empty && arena_len > 0 {
            // OPTIMIZATION: Cache visibility result for repeated txn_ids
            let mut last_txn_id: i64 = -1;
            let mut last_visible: bool = false;

            let mut idx = 0;
            while idx < arena_len {
                let meta = &arena_meta[idx];
                if meta.deleted_at_txn_id == 0 {
                    let version_txn_id = meta.txn_id;
                    let is_vis = if version_txn_id == last_txn_id {
                        last_visible
                    } else {
                        let vis = checker.is_visible(version_txn_id, txn_id);
                        last_txn_id = version_txn_id;
                        last_visible = vis;
                        vis
                    };

                    if is_vis {
                        if let Some(val) = arena_data[idx].get(col_idx) {
                            update_min(&mut min_val, val);
                        }
                    }
                }
                idx += 1;
            }
            return min_val;
        }

        // Drop arena guard before slow path
        drop(arena_guard);

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.versions.read();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = -1;
        let mut last_visible: bool = false;

        for chain in versions.values() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                let is_vis = if version_txn_id == last_txn_id {
                    last_visible
                } else {
                    let vis = checker.is_visible(version_txn_id, txn_id);
                    last_txn_id = version_txn_id;
                    last_visible = vis;
                    vis
                };

                if is_vis
                    && (deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id))
                {
                    if let Some(val) = e.version.data.get(col_idx) {
                        update_min(&mut min_val, val);
                    }
                    break;
                }

                current = e.prev.as_deref();
            }
        }

        min_val
    }

    /// Compute MAX(column) without materializing full rows
    ///
    /// OPTIMIZATION: Single-pass approach that combines visibility checking with max computation.
    pub fn max_column(&self, txn_id: i64, col_idx: usize) -> Option<Value> {
        let checker = self.visibility_checker.as_ref()?;

        let arena_guard = self.arena.read_guard();
        let arena_meta = arena_guard.meta();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        let mut max_val: Option<Value> = None;

        // Helper to update max value
        #[inline(always)]
        fn update_max(max_val: &mut Option<Value>, val: &Value) {
            if !val.is_null() {
                match max_val {
                    None => *max_val = Some(val.clone()),
                    Some(ref current) => {
                        if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                            *max_val = Some(val.clone());
                        }
                    }
                }
            }
        }

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass)
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        if uncommitted_empty && arena_len > 0 {
            // OPTIMIZATION: Cache visibility result for repeated txn_ids
            let mut last_txn_id: i64 = -1;
            let mut last_visible: bool = false;

            let mut idx = 0;
            while idx < arena_len {
                let meta = &arena_meta[idx];
                if meta.deleted_at_txn_id == 0 {
                    let version_txn_id = meta.txn_id;
                    let is_vis = if version_txn_id == last_txn_id {
                        last_visible
                    } else {
                        let vis = checker.is_visible(version_txn_id, txn_id);
                        last_txn_id = version_txn_id;
                        last_visible = vis;
                        vis
                    };

                    if is_vis {
                        if let Some(val) = arena_data[idx].get(col_idx) {
                            update_max(&mut max_val, val);
                        }
                    }
                }
                idx += 1;
            }
            return max_val;
        }

        // Drop arena guard before slow path
        drop(arena_guard);

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.versions.read();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = -1;
        let mut last_visible: bool = false;

        for chain in versions.values() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                let is_vis = if version_txn_id == last_txn_id {
                    last_visible
                } else {
                    let vis = checker.is_visible(version_txn_id, txn_id);
                    last_txn_id = version_txn_id;
                    last_visible = vis;
                    vis
                };

                if is_vis
                    && (deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id))
                {
                    if let Some(val) = e.version.data.get(col_idx) {
                        update_max(&mut max_val, val);
                    }
                    break;
                }

                current = e.prev.as_deref();
            }
        }

        max_val
    }

    /// Compute multiple column aggregates in a single pass
    ///
    /// This is the most efficient path for queries like:
    /// `SELECT SUM(a), AVG(b), MIN(c), MAX(d) FROM table`
    ///
    /// Returns aggregates in the order requested.
    pub fn compute_aggregates(
        &self,
        txn_id: i64,
        aggregates: &[(AggregateOp, usize)], // (operation, column_index)
    ) -> Vec<AggregateResult> {
        let indices = self.get_visible_row_indices_unordered(txn_id);
        if indices.is_empty() {
            return aggregates
                .iter()
                .map(|(op, _)| match op {
                    AggregateOp::Count | AggregateOp::CountStar => AggregateResult::Count(0),
                    AggregateOp::Sum => AggregateResult::Sum(0.0, 0),
                    AggregateOp::Min => AggregateResult::Min(None),
                    AggregateOp::Max => AggregateResult::Max(None),
                    AggregateOp::Avg => AggregateResult::Avg(0.0, 0),
                })
                .collect();
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        // Initialize accumulators
        let mut results: Vec<AggregateAccumulator> = aggregates
            .iter()
            .map(|(op, _)| match op {
                AggregateOp::Count | AggregateOp::CountStar => AggregateAccumulator::Count(0),
                AggregateOp::Sum => AggregateAccumulator::Sum(0.0, 0),
                AggregateOp::Min => AggregateAccumulator::Min(None),
                AggregateOp::Max => AggregateAccumulator::Max(None),
                AggregateOp::Avg => AggregateAccumulator::Avg(0.0, 0),
            })
            .collect();

        // Helper to update accumulator with a value
        fn update_accumulator(acc: &mut AggregateAccumulator, op: &AggregateOp, val: &Value) {
            match (acc, op) {
                (AggregateAccumulator::Count(c), AggregateOp::Count) => {
                    if !val.is_null() {
                        *c += 1;
                    }
                }
                (AggregateAccumulator::Count(c), AggregateOp::CountStar) => {
                    // COUNT(*) counts all rows including NULL
                    *c += 1;
                }
                (AggregateAccumulator::Sum(sum, cnt), AggregateOp::Sum) => match val {
                    Value::Integer(i) => {
                        *sum += *i as f64;
                        *cnt += 1;
                    }
                    Value::Float(f) => {
                        *sum += *f;
                        *cnt += 1;
                    }
                    _ => {}
                },
                (AggregateAccumulator::Min(min), AggregateOp::Min) => {
                    if !val.is_null() {
                        match min {
                            None => *min = Some(val.clone()),
                            Some(current) => {
                                if let Ok(std::cmp::Ordering::Less) = val.compare(current) {
                                    *min = Some(val.clone());
                                }
                            }
                        }
                    }
                }
                (AggregateAccumulator::Max(max), AggregateOp::Max) => {
                    if !val.is_null() {
                        match max {
                            None => *max = Some(val.clone()),
                            Some(current) => {
                                if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                                    *max = Some(val.clone());
                                }
                            }
                        }
                    }
                }
                (AggregateAccumulator::Avg(sum, cnt), AggregateOp::Avg) => match val {
                    Value::Integer(i) => {
                        *sum += *i as f64;
                        *cnt += 1;
                    }
                    Value::Float(f) => {
                        *sum += *f;
                        *cnt += 1;
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Single pass through all rows
        for idx in &indices {
            // Try arena path first (fast)
            if let Some(arena_idx) = idx.arena_idx {
                if let Some(arc_row) = arena_data.get(arena_idx) {
                    for (i, (op, col_idx)) in aggregates.iter().enumerate() {
                        if let Some(val) = arc_row.get(*col_idx) {
                            update_accumulator(&mut results[i], op, val);
                        }
                    }
                    continue;
                }
            }
            // Slow path: version chain lookup (lock already held)
            if let Some(entry) = versions.get(&idx.row_id) {
                for (i, (op, col_idx)) in aggregates.iter().enumerate() {
                    if let Some(val) = entry.version.data.get(*col_idx) {
                        update_accumulator(&mut results[i], op, val);
                    }
                }
            }
        }

        // Convert accumulators to results
        results
            .into_iter()
            .map(|acc| match acc {
                AggregateAccumulator::Count(c) => AggregateResult::Count(c),
                AggregateAccumulator::Sum(s, c) => AggregateResult::Sum(s, c),
                AggregateAccumulator::Min(v) => AggregateResult::Min(v),
                AggregateAccumulator::Max(v) => AggregateResult::Max(v),
                AggregateAccumulator::Avg(s, c) => AggregateResult::Avg(s, c),
            })
            .collect()
    }

    /// Returns a zero-copy streaming iterator over visible rows
    ///
    /// This is the TRUE zero-copy approach that leverages Rust's borrowing system.
    /// Instead of cloning row data, it:
    /// 1. Pre-computes visible row indices in a single pass
    /// 2. Holds arena locks for the duration of iteration
    /// 3. Yields &[Value] slices directly from arena memory
    ///
    /// # Performance
    /// - Eliminates ALL cloning during iteration
    /// - Single lock acquisition for entire scan
    /// - Cache-friendly contiguous memory access
    /// - Fast path: O(n) arena scan when all rows are visible (skips DashMap)
    ///
    /// # Usage
    /// ```ignore
    /// let mut stream = version_store.stream_visible_rows(txn_id);
    /// while stream.next() {
    ///     let values: &[Value] = stream.row_slice(); // Zero-copy!
    ///     let row_id = stream.row_id();
    ///     // Process values...
    /// }
    /// ```
    pub fn stream_visible_rows(&self, txn_id: i64) -> StreamingResult<'_> {
        let checker = self.visibility_checker.as_ref();

        // Pre-acquire arena lock
        let arena_guard = self.arena.read_guard();
        let arena_len = arena_guard.len();

        // Get column names from schema
        let columns: Vec<String> = self
            .schema
            .read()
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // FAST PATH: If we can iterate arena directly without DashMap
        // This is possible when:
        // 1. No active uncommitted writes
        // 2. All arena rows are visible to this transaction
        // Check if we can use the fast path by scanning arena directly
        if let Some(checker) = checker {
            // Check if uncommitted_writes is empty (no dirty reads to worry about)
            let uncommitted_empty = self.uncommitted_writes.read().is_empty();

            if uncommitted_empty && arena_len > 0 {
                // Fast path: scan arena directly, skip DashMap iteration
                let mut visible_indices: Vec<VisibleRowInfo> = Vec::with_capacity(arena_len);

                for (idx, meta) in arena_guard.meta().iter().enumerate() {
                    // Check visibility and deleted status
                    if meta.deleted_at_txn_id == 0 && checker.is_visible(meta.txn_id, txn_id) {
                        visible_indices.push(VisibleRowInfo {
                            row_id: meta.row_id,
                            arena_idx: idx,
                        });
                    }
                }

                // Sort by row_id for consistent ordering
                visible_indices.sort_unstable_by_key(|info| info.row_id);

                return StreamingResult::new(arena_guard, visible_indices, columns);
            }
        }

        // SLOW PATH: Full iteration (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut visible_indices: Vec<VisibleRowInfo> = Vec::with_capacity(versions.len());

        if let Some(checker) = checker {
            for (&row_id, chain) in versions.iter() {
                let mut current: Option<&VersionChainEntry> = Some(chain);

                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            // Found visible, non-deleted version
                            if let Some(idx) = e.arena_idx {
                                if idx < arena_len {
                                    visible_indices.push(VisibleRowInfo {
                                        row_id,
                                        arena_idx: idx,
                                    });
                                }
                            }
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }
        drop(versions); // Release lock before returning

        // BTreeMap already iterates in sorted order
        StreamingResult::new(arena_guard, visible_indices, columns)
    }

    /// Returns the count of rows
    pub fn row_count(&self) -> usize {
        self.versions.read().len()
    }
    /// Tries to claim a row for update (dirty write prevention)
    pub fn try_claim_row(&self, row_id: i64, txn_id: i64) -> Result<(), Error> {
        use std::collections::hash_map::Entry;

        let mut map = self.uncommitted_writes.write();
        match map.entry(row_id) {
            Entry::Occupied(e) => {
                let existing_txn = *e.get();
                if existing_txn != txn_id {
                    return Err(Error::internal(format!(
                        "row {} has uncommitted changes from transaction {}",
                        row_id, existing_txn
                    )));
                }
                // Same transaction already owns it
                Ok(())
            }
            Entry::Vacant(e) => {
                e.insert(txn_id);
                Ok(())
            }
        }
    }

    /// Releases a row claim
    pub fn release_row_claim(&self, row_id: i64, txn_id: i64) {
        let mut map = self.uncommitted_writes.write();
        if let Some(&v) = map.get(&row_id) {
            if v == txn_id {
                map.remove(&row_id);
            }
        }
    }

    /// Releases multiple row claims in batch
    /// OPTIMIZATION: Single lock acquisition for all removals
    #[inline]
    pub fn release_row_claims_batch(&self, row_ids: &[i64], txn_id: i64) {
        let mut map = self.uncommitted_writes.write();
        for &row_id in row_ids {
            if let Some(&v) = map.get(&row_id) {
                if v == txn_id {
                    map.remove(&row_id);
                }
            }
        }
    }

    /// Check if an index exists
    pub fn index_exists(&self, index_name: &str) -> bool {
        let indexes = self.indexes.read();
        indexes.contains_key(index_name)
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<String> {
        let indexes = self.indexes.read();
        indexes.keys().cloned().collect()
    }

    /// Iterate over unique indexes only, calling the provided function for each
    /// OPTIMIZATION: Avoids collecting index names and allows early exit on error
    pub fn for_each_unique_index<F>(&self, mut f: F) -> crate::core::Result<()>
    where
        F: FnMut(&str, &Arc<dyn Index>) -> crate::core::Result<()>,
    {
        let indexes = self.indexes.read();
        for (name, index) in indexes.iter() {
            if index.is_unique() {
                f(name, index)?;
            }
        }
        Ok(())
    }

    /// Add an index
    pub fn add_index(&self, name: String, index: Arc<dyn Index>) {
        let mut indexes = self.indexes.write();
        indexes.insert(name, index);
    }

    /// Remove an index
    pub fn remove_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        let mut indexes = self.indexes.write();
        indexes.remove(name)
    }

    /// Get an index by name
    pub fn get_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        let indexes = self.indexes.read();
        indexes.get(name).cloned()
    }

    /// Get an index by column name (single-column indexes only)
    pub fn get_index_by_column(&self, column_name: &str) -> Option<Arc<dyn Index>> {
        let indexes = self.indexes.read();
        for index in indexes.values() {
            let column_names = index.column_names();
            if column_names.len() == 1 && column_names[0] == column_name {
                return Some(index.clone());
            }
        }
        None
    }

    /// Find the best multi-column index that matches a set of predicate columns.
    /// Returns the index if predicate columns cover a prefix of the index columns (leftmost prefix rule).
    /// For example, an index on (a, b, c) can be used for queries that include (a), (a, b), or (a, b, c).
    /// The predicate columns don't need to be in the same order as the index columns.
    pub fn get_multi_column_index(
        &self,
        predicate_columns: &[&str],
    ) -> Option<(Arc<dyn Index>, usize)> {
        if predicate_columns.is_empty() {
            return None;
        }

        let indexes = self.indexes.read();
        let mut best_match: Option<(Arc<dyn Index>, usize)> = None;

        // Create a set of predicate columns for O(1) lookup
        let pred_set: FxHashSet<&str> = predicate_columns.iter().copied().collect();

        for index in indexes.values() {
            let index_columns = index.column_names();
            if index_columns.len() < 2 {
                continue; // Skip single-column indexes
            }

            // Count how many of the leading index columns are in the predicate set.
            // This implements the leftmost prefix rule: we can only use the index
            // if we have predicates on a contiguous prefix of the index columns.
            let mut matched = 0;
            for idx_col in index_columns.iter() {
                if pred_set.contains(idx_col.as_str()) {
                    matched += 1;
                } else {
                    // Stop at the first index column not in predicates
                    break;
                }
            }

            // We need at least 2 columns to match to prefer multi-col over single-col
            if matched >= 2 {
                // Prefer index with more matching columns
                if best_match.is_none() || matched > best_match.as_ref().unwrap().1 {
                    best_match = Some((index.clone(), matched));
                }
            }
        }

        best_match
    }

    /// Get all indexes (for optimizer to inspect)
    pub fn get_all_indexes(&self) -> Vec<Arc<dyn Index>> {
        let indexes = self.indexes.read();
        indexes.values().cloned().collect()
    }

    // =========================================================================
    // Zone Map Operations (Statistics for Segment Pruning)
    // =========================================================================

    /// Sets the zone maps for this table
    ///
    /// Zone maps contain min/max statistics per segment, enabling the query
    /// executor to skip entire segments when predicates fall outside the range.
    pub fn set_zone_maps(&self, zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {
        let mut guard = self.zone_maps.write();
        *guard = Some(Arc::new(zone_maps));
    }

    /// Gets the zone maps for this table
    ///
    /// Returns None if zone maps have not been built (ANALYZE not run)
    /// Uses Arc to avoid expensive cloning on high QPS workloads
    pub fn get_zone_maps(&self) -> Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        let guard = self.zone_maps.read();
        guard.clone()
    }

    /// Gets the segments that need to be scanned for a given predicate
    ///
    /// Uses zone maps to determine which segments can be pruned (skipped)
    pub fn get_segments_to_scan(
        &self,
        column: &str,
        operator: crate::core::Operator,
        value: &crate::core::Value,
    ) -> Option<Vec<u32>> {
        let guard = self.zone_maps.read();
        guard
            .as_ref()
            .and_then(|zm| zm.get_segments_to_scan(column, operator, value))
    }

    /// Gets prune statistics for a single-column predicate
    pub fn get_prune_stats(
        &self,
        column: &str,
        operator: crate::core::Operator,
        value: &crate::core::Value,
    ) -> Option<crate::storage::mvcc::zonemap::PruneStats> {
        let guard = self.zone_maps.read();
        guard
            .as_ref()
            .and_then(|zm| zm.get_prune_stats(column, operator, value))
    }

    /// Marks zone maps as stale (needing rebuild after data changes)
    pub fn mark_zone_maps_stale(&self) {
        let guard = self.zone_maps.read();
        if let Some(ref zm) = *guard {
            zm.mark_stale();
        }
    }

    /// Close the version store
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    /// Check if the version store is closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    // =========================================================================
    // Recovery Functions (for WAL replay)
    // =========================================================================

    /// Apply a recovered row version during WAL replay
    ///
    /// This is used during database recovery to apply row versions from the WAL.
    /// Unlike normal operations, this directly adds the version without visibility checks.
    /// Also updates any existing indexes with the new row data.
    /// Also updates the auto_increment counter if row_id is higher than current.
    ///
    /// Duplicate Detection: If the row already exists with identical data (same values),
    /// the version is skipped to avoid duplicate entries in the version chain. This can
    /// occur when snapshot and WAL both contain the same committed data due to race
    /// conditions during snapshot creation.
    pub fn apply_recovered_version(&self, version: RowVersion) {
        let row_id = version.row_id;
        let is_deleted = version.is_deleted();
        let row_data = version.data.clone();

        // Check for duplicate: if row already exists with identical data, skip adding
        // This prevents duplicate version chain entries when both snapshot and WAL
        // contain the same row data (can happen due to race conditions during snapshot)
        {
            let versions = self.versions.read();
            if let Some(existing_entry) = versions.get(&row_id) {
                let existing = &existing_entry.version;
                // Check if data is identical (both deleted status and row data)
                if existing.is_deleted() == is_deleted && existing.data == row_data {
                    // Identical data already exists, skip to avoid duplicate
                    // Still update auto_increment counter
                    if row_id > 0 {
                        self.set_auto_increment_counter(row_id);
                    }
                    return;
                }
            }
        }

        // Add the version to the store
        self.add_version(row_id, version);

        // Update auto_increment counter if this row_id is higher
        // This ensures the counter is restored to at least the max seen row_id
        if row_id > 0 {
            self.set_auto_increment_counter(row_id);
        }

        // Update indexes with the new row data (if not deleted)
        // Use get_arc + add_arc for zero-copy Arc sharing
        if !is_deleted {
            let indexes = self.indexes.read();
            for index in indexes.values() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }
                let col_id = column_ids[0] as usize;
                if let Some(arc_value) = row_data.get_arc(col_id) {
                    // Ignore errors during recovery - index might already have this entry
                    let _ = index.add_arc(std::slice::from_ref(&arc_value), row_id, row_id);
                }
            }
        }
    }

    /// Mark a row as deleted during WAL replay
    ///
    /// This creates a deleted version for the row during recovery.
    /// Also removes the row from any existing indexes.
    pub fn mark_deleted(&self, row_id: i64, txn_id: i64) {
        // Get the old row data for index removal BEFORE creating the deleted version
        let old_row = self
            .get_visible_version(row_id, txn_id)
            .map(|v| v.data.clone());

        // Create a deleted version (empty data with deleted flag)
        let deleted_version = RowVersion {
            txn_id,
            deleted_at_txn_id: txn_id,
            data: Row::new(),
            row_id,
            create_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0),
        };
        self.add_version(row_id, deleted_version);

        // Remove from indexes using old row data
        if let Some(old_data) = old_row {
            let indexes = self.indexes.read();
            for index in indexes.values() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }
                let col_id = column_ids[0] as usize;
                if let Some(value) = old_data.get(col_id) {
                    let _ = index.remove(std::slice::from_ref(value), row_id, row_id);
                }
            }
        }
    }

    /// Drop an index by name (alias for remove_index)
    pub fn drop_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        self.remove_index(name)
    }

    /// Create an index from persistence metadata during WAL replay
    ///
    /// This recreates an index from its persisted metadata.
    ///
    /// # Arguments
    /// * `meta` - Index metadata from WAL
    /// * `skip_population` - If true, creates the index structure without populating it.
    ///   This is used during batch recovery to defer population to a single-pass scan.
    pub fn create_index_from_metadata(
        &self,
        meta: &crate::storage::mvcc::persistence::IndexMetadata,
        skip_population: bool,
    ) -> crate::core::Result<()> {
        use crate::core::IndexType;
        use crate::storage::mvcc::{BitmapIndex, HashIndex};

        // Check if we have the required column information
        if meta.column_names.is_empty() {
            return Err(crate::core::Error::internal(
                "index metadata must have at least one column",
            ));
        }

        // Check if index already exists
        if self.index_exists(&meta.name) {
            return Ok(()); // Already recovered, skip
        }

        if meta.column_names.len() == 1 {
            // Single-column index
            let column_name = &meta.column_names[0];
            let column_id = meta.column_ids.first().copied().unwrap_or(0);
            let data_type = meta
                .data_types
                .first()
                .copied()
                .unwrap_or(crate::core::DataType::Null);

            // Create index based on stored index_type
            let index: Arc<dyn crate::storage::Index> = match meta.index_type {
                IndexType::Hash => {
                    let idx = HashIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        vec![column_name.clone()],
                        vec![column_id],
                        vec![data_type],
                        meta.is_unique,
                    );
                    Arc::new(idx)
                }
                IndexType::Bitmap => {
                    let idx = BitmapIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        vec![column_name.clone()],
                        vec![column_id],
                        vec![data_type],
                        meta.is_unique,
                    );
                    Arc::new(idx)
                }
                IndexType::BTree => {
                    // BTree uses BTreeIndex implementation
                    let idx = crate::storage::mvcc::BTreeIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        column_id,
                        column_name.clone(),
                        data_type,
                        meta.is_unique,
                    );
                    Arc::new(idx)
                }
                IndexType::MultiColumn => {
                    // MultiColumn uses MultiColumnIndex implementation
                    let idx = crate::storage::mvcc::MultiColumnIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        meta.column_names.clone(),
                        meta.column_ids.clone(),
                        meta.data_types.clone(),
                        meta.is_unique,
                    );
                    Arc::new(idx)
                }
            };

            // Populate the index with existing data unless deferred
            if !skip_population {
                let col_idx = column_id as usize;
                let versions = self.versions.read();
                for (&row_id, version_chain) in versions.iter() {
                    let version = &version_chain.version;
                    if !version.is_deleted() {
                        // Use get_arc + add_arc for zero-copy Arc sharing
                        if let Some(arc_value) = version.data.get_arc(col_idx) {
                            let _ = index.add_arc(std::slice::from_ref(&arc_value), row_id, row_id);
                        }
                    }
                }
            }

            self.add_index(meta.name.clone(), index);
        } else {
            // Multi-column index: use MultiColumnIndex
            let index = crate::storage::mvcc::MultiColumnIndex::new(
                meta.name.clone(),
                meta.table_name.clone(),
                meta.column_names.clone(),
                meta.column_ids.clone(),
                meta.data_types.clone(),
                meta.is_unique,
            );

            let index = Arc::new(index);

            // Populate the index with existing data unless deferred
            if !skip_population {
                let col_indices: Vec<usize> =
                    meta.column_ids.iter().map(|&id| id as usize).collect();
                let versions = self.versions.read();
                for (&row_id, version_chain) in versions.iter() {
                    let version = &version_chain.version;
                    if !version.is_deleted() {
                        // Use get_arc + add_arc for zero-copy Arc sharing
                        let arc_values: Vec<std::sync::Arc<crate::core::Value>> = col_indices
                            .iter()
                            .map(|&idx| {
                                version.data.get_arc(idx).unwrap_or_else(|| {
                                    std::sync::Arc::new(crate::core::Value::Null(
                                        crate::core::DataType::Null,
                                    ))
                                })
                            })
                            .collect();
                        let _ = index.add_arc(&arc_values, row_id, row_id);
                    }
                }
            }

            self.add_index(meta.name.clone(), index);
        }

        Ok(())
    }

    /// Populate all indexes in a single pass over the version store
    ///
    /// This is O(N + M) where N = number of rows and M = number of indexes,
    /// compared to O(N * M) when populating each index separately.
    ///
    /// Call this after WAL replay completes with skip_population=true.
    pub fn populate_all_indexes(&self) {
        let indexes = self.indexes.read();
        if indexes.is_empty() {
            return;
        }

        // Collect index info: (column_ids as Vec<usize>, index_arc)
        // Supports both single-column and multi-column indexes
        let index_infos: Vec<(Vec<usize>, Arc<dyn Index>)> = indexes
            .values()
            .filter_map(|idx| {
                let col_ids = idx.column_ids();
                if col_ids.is_empty() {
                    None
                } else {
                    let col_indices: Vec<usize> = col_ids.iter().map(|&id| id as usize).collect();
                    Some((col_indices, Arc::clone(idx)))
                }
            })
            .collect();

        drop(indexes); // Release lock before iteration

        if index_infos.is_empty() {
            return;
        }

        // Single pass over all rows (BTreeMap)
        let versions = self.versions.read();
        for (&row_id, version_chain) in versions.iter() {
            let version = &version_chain.version;

            if version.is_deleted() {
                continue;
            }

            // Add to each index - use get_arc + add_arc for zero-copy Arc sharing
            for (col_indices, index) in &index_infos {
                if col_indices.len() == 1 {
                    // Single-column index
                    if let Some(arc_value) = version.data.get_arc(col_indices[0]) {
                        let _ = index.add_arc(std::slice::from_ref(&arc_value), row_id, row_id);
                    }
                } else {
                    // Multi-column index
                    let arc_values: Vec<std::sync::Arc<crate::core::Value>> = col_indices
                        .iter()
                        .map(|&idx| {
                            version.data.get_arc(idx).unwrap_or_else(|| {
                                std::sync::Arc::new(crate::core::Value::Null(
                                    crate::core::DataType::Null,
                                ))
                            })
                        })
                        .collect();
                    let _ = index.add_arc(&arc_values, row_id, row_id);
                }
            }
        }
    }

    // =========================================================================
    // Cleanup Functions
    // =========================================================================

    /// Cleanup deleted rows that are older than the retention period
    ///
    /// This removes soft-deleted rows that are no longer visible to any active
    /// transaction and are older than the specified retention period.
    pub fn cleanup_deleted_rows(&self, retention_period: std::time::Duration) -> i32 {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let now = get_fast_timestamp();
        let cutoff_time = now - retention_period.as_nanos() as i64;

        let mut rows_to_delete = Vec::new();

        // First pass: identify deleted rows older than retention period
        {
            let versions = self.versions.read();
            for (&row_id, chain) in versions.iter() {
                let version = &chain.version;
                // Only process rows that are actually deleted and old enough
                if version.is_deleted() && version.create_time < cutoff_time {
                    // Check if safe to remove (no active transaction can see it)
                    if self.can_safely_remove(version) {
                        rows_to_delete.push(row_id);
                    }
                }
            }
        }

        // Second pass: remove the identified rows
        for row_id in &rows_to_delete {
            // Remove from indexes first
            {
                let versions = self.versions.read();
                if let Some(entry) = versions.get(row_id) {
                    let version = &entry.version;
                    let indexes = self.indexes.read();
                    for index in indexes.values() {
                        let column_ids = index.column_ids();
                        if !column_ids.is_empty() {
                            if column_ids.len() == 1 {
                                // Single-column index
                                let col_id = column_ids[0] as usize;
                                if let Some(value) = version.data.get(col_id) {
                                    let _ =
                                        index.remove(std::slice::from_ref(value), *row_id, *row_id);
                                }
                            } else {
                                // Multi-column index
                                let values: Vec<crate::core::Value> = column_ids
                                    .iter()
                                    .map(|&col_id| {
                                        version.data.get(col_id as usize).cloned().unwrap_or(
                                            crate::core::Value::Null(crate::core::DataType::Null),
                                        )
                                    })
                                    .collect();
                                let _ = index.remove(&values, *row_id, *row_id);
                            }
                        }
                    }
                }
            }

            // Now remove from version store
            self.versions.write().remove(row_id);
        }

        rows_to_delete.len() as i32
    }

    /// Check if a version can be safely removed (not visible to any active transaction)
    fn can_safely_remove(&self, version: &RowVersion) -> bool {
        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return true, // No checker, assume safe
        };

        // Get all active transaction IDs
        let active_txns = checker.get_active_transaction_ids();

        // If no active transactions, safe to remove
        if active_txns.is_empty() {
            return true;
        }

        // Check if any active transaction can still see this version
        for txn_id in active_txns {
            if checker.is_visible(version.txn_id, txn_id) {
                return false; // Still visible to at least one transaction
            }
        }

        true
    }

    /// Cleanup old previous versions that are no longer needed
    ///
    /// This prunes old version chains, keeping only versions that are:
    /// 1. Needed by active transactions
    /// 2. Within the retention period (for AS OF TIMESTAMP queries)
    pub fn cleanup_old_previous_versions(&self) -> i32 {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0, // Need visibility checker for cleanup
        };

        // Default 24-hour retention for historical versions
        let retention_period = std::time::Duration::from_secs(24 * 60 * 60);
        let now = get_fast_timestamp();
        let retention_cutoff = now - retention_period.as_nanos() as i64;

        // Get active transaction IDs
        let active_txns = checker.get_active_transaction_ids();

        let mut cleaned = 0;

        // Collect entries that need cleanup (BTreeMap doesn't support in-place mutation while iterating)
        let mut updates: Vec<(i64, VersionChainEntry)> = Vec::new();

        // First pass: identify entries that need pruning
        let versions = self.versions.read();
        for (&row_id, chain_entry) in versions.iter() {
            // Find the oldest version we need to keep
            let mut current = chain_entry.prev.as_ref();
            let mut versions_to_check = Vec::new();

            // Collect all versions in the chain
            while let Some(prev_entry) = current {
                versions_to_check.push(prev_entry.clone());
                current = prev_entry.prev.as_ref();
            }

            if versions_to_check.is_empty() {
                continue;
            }

            // Find cutoff point - first version we can discard
            let mut keep_count = 0;
            for prev_entry in &versions_to_check {
                let mut keep = false;

                // Rule 1: Keep if needed by any active transaction
                for &txn_id in &active_txns {
                    if checker.is_visible(prev_entry.version.txn_id, txn_id) {
                        keep = true;
                        break;
                    }
                }

                // Rule 2: Keep if within retention period
                if !keep && prev_entry.version.create_time >= retention_cutoff {
                    keep = true;
                }

                if keep {
                    keep_count += 1;
                } else {
                    // Found the cutoff point
                    break;
                }
            }

            // If we need to prune some versions, prepare an update
            if keep_count < versions_to_check.len() {
                let to_remove = versions_to_check.len() - keep_count;
                cleaned += to_remove as i32;

                // Clone and modify the chain entry
                let mut modified_entry = chain_entry.clone();

                if keep_count == 0 {
                    // Remove all previous versions
                    modified_entry.prev = None;
                } else {
                    // Rebuild chain with only kept versions
                    let kept_versions: Vec<_> =
                        versions_to_check.into_iter().take(keep_count).collect();

                    // Build chain from oldest to newest (reversed)
                    let mut new_prev: Option<Arc<VersionChainEntry>> = None;
                    for entry in kept_versions.into_iter().rev() {
                        let mut cloned = (*entry).clone();
                        cloned.prev = new_prev;
                        new_prev = Some(Arc::new(cloned));
                    }
                    modified_entry.prev = new_prev;
                }

                updates.push((row_id, modified_entry));
            }
        }
        drop(versions); // Release read lock before writing

        // Second pass: apply updates
        if !updates.is_empty() {
            let mut versions = self.versions.write();
            for (row_id, modified_entry) in updates {
                versions.insert(row_id, modified_entry);
            }
        }

        cleaned
    }

    /// Iterate over all committed (non-deleted) versions for snapshot creation
    ///
    /// This method iterates over all rows that are visible to a snapshot transaction
    /// (i.e., all committed, non-deleted rows). The callback receives the row_id and
    /// a reference to the RowVersion. Return false from the callback to stop iteration.
    ///
    /// This is designed for creating point-in-time snapshots to disk.
    pub fn for_each_committed_version<F>(&self, callback: F)
    where
        F: FnMut(i64, &RowVersion) -> bool,
    {
        // Delegate to the cutoff version with no cutoff (0 means no filtering)
        self.for_each_committed_version_with_cutoff(callback, 0);
    }

    /// Iterate over committed versions with a commit sequence cutoff for consistent snapshots
    ///
    /// This is the same as `for_each_committed_version` but only includes transactions
    /// that were committed before the given `commit_seq_cutoff`. This ensures consistent
    /// point-in-time snapshots even when new transactions commit during iteration.
    ///
    /// # Arguments
    /// * `callback` - Called for each visible, non-deleted version
    /// * `commit_seq_cutoff` - Only include transactions with commit_seq < cutoff (0 = no filter)
    pub fn for_each_committed_version_with_cutoff<F>(&self, mut callback: F, commit_seq_cutoff: i64)
    where
        F: FnMut(i64, &RowVersion) -> bool,
    {
        if self.closed.load(Ordering::Acquire) {
            return;
        }

        // Get visibility checker for determining committed status
        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return,
        };

        // Use a very high txn_id to see all committed rows
        let snapshot_txn_id = i64::MAX;
        let use_cutoff = commit_seq_cutoff > 0;

        // Iterate all versions (BTreeMap maintains sorted order)
        let versions = self.versions.read();
        for (&row_id, chain_entry) in versions.iter() {
            // Walk the version chain to find the visible version
            let mut current: Option<&VersionChainEntry> = Some(chain_entry);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                // Check if this version is visible
                if checker.is_visible(version_txn_id, snapshot_txn_id) {
                    // When cutoff is specified, only include versions from transactions
                    // that committed before the cutoff to ensure snapshot consistency
                    if use_cutoff && !checker.is_committed_before(version_txn_id, commit_seq_cutoff)
                    {
                        // This transaction committed after our snapshot point, try older version
                        current = e.prev.as_ref().map(|arc| arc.as_ref());
                        continue;
                    }

                    // Skip if deleted and deletion is visible (and within cutoff if specified)
                    if deleted_at_txn_id != 0
                        && checker.is_visible(deleted_at_txn_id, snapshot_txn_id)
                        && (!use_cutoff
                            || checker.is_committed_before(deleted_at_txn_id, commit_seq_cutoff))
                    {
                        break; // Row is deleted, skip
                    }

                    // Found visible, non-deleted version
                    if !callback(row_id, &e.version) {
                        return; // Callback wants to stop
                    }
                    break;
                }

                // Try older version
                current = e.prev.as_ref().map(|arc| arc.as_ref());
            }
        }
    }

    /// Get the count of committed (non-deleted) versions for statistics
    pub fn count_committed_versions(&self) -> usize {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0,
        };

        let snapshot_txn_id = i64::MAX;
        let mut count = 0;

        let versions = self.versions.read();
        for (_, chain_entry) in versions.iter() {
            let mut current: Option<&VersionChainEntry> = Some(chain_entry);

            while let Some(e) = current {
                if checker.is_visible(e.version.txn_id, snapshot_txn_id) {
                    if e.version.deleted_at_txn_id == 0
                        || !checker.is_visible(e.version.deleted_at_txn_id, snapshot_txn_id)
                    {
                        count += 1;
                    }
                    break;
                }
                current = e.prev.as_ref().map(|arc| arc.as_ref());
            }
        }

        count
    }

    /// Compute grouped aggregates directly from arena storage.
    ///
    /// This method performs GROUP BY aggregation at the storage level without
    /// materializing Row objects. It uses Arc::clone for group keys (O(1))
    /// instead of Value::clone (deep copy), significantly reducing allocations.
    ///
    /// # Arguments
    /// * `txn_id` - Transaction ID for visibility checks
    /// * `group_by_indices` - Column indices to group by
    /// * `aggregates` - List of (operation, column_index) pairs
    ///
    /// # Returns
    /// Vector of grouped aggregate results, or empty if optimization not possible
    pub fn compute_grouped_aggregates(
        &self,
        txn_id: i64,
        group_by_indices: &[usize],
        aggregates: &[(AggregateOp, usize)],
    ) -> Vec<GroupedAggregateResult> {
        use ahash::AHashMap;

        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Accumulator for each group: (count, sum, min, max) per aggregate
        #[derive(Clone)]
        struct Accum {
            count: i64,
            sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }

        impl Default for Accum {
            fn default() -> Self {
                Self {
                    count: 0,
                    sum: 0.0,
                    min: None,
                    max: None,
                }
            }
        }

        let mut groups: AHashMap<GroupKey, Vec<Accum>> = AHashMap::new();

        // Pre-acquire arena lock ONCE
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();
        let arena_meta = arena_guard.meta();

        // Process arena rows directly (fast path - no Row allocation)
        for (idx, meta) in arena_meta.iter().enumerate() {
            // Visibility check
            if meta.deleted_at_txn_id != 0 && checker.is_visible(meta.deleted_at_txn_id, txn_id) {
                continue;
            }
            if !checker.is_visible(meta.txn_id, txn_id) {
                continue;
            }

            // Get row data from arena
            let row_data = match arena_data.get(idx) {
                Some(data) => data,
                None => continue,
            };

            // Build group key using Arc::clone (O(1)) instead of Value::clone
            let group_key = if group_by_indices.len() == 1 {
                let col_idx = group_by_indices[0];
                if col_idx < row_data.len() {
                    GroupKey::Single(Arc::clone(&row_data[col_idx]))
                } else {
                    GroupKey::Single(Arc::new(Value::Null(DataType::Null)))
                }
            } else {
                let key_values: Vec<Arc<Value>> = group_by_indices
                    .iter()
                    .map(|&col_idx| {
                        if col_idx < row_data.len() {
                            Arc::clone(&row_data[col_idx])
                        } else {
                            Arc::new(Value::Null(DataType::Null))
                        }
                    })
                    .collect();
                GroupKey::Multi(key_values)
            };

            // Get or create accumulator for this group
            let accums = groups
                .entry(group_key)
                .or_insert_with(|| vec![Accum::default(); aggregates.len()]);

            // Update each aggregate
            for (agg_idx, (op, col_idx)) in aggregates.iter().enumerate() {
                let accum = &mut accums[agg_idx];

                match op {
                    AggregateOp::CountStar => {
                        accum.count += 1;
                    }
                    AggregateOp::Count => {
                        if *col_idx < row_data.len() && !row_data[*col_idx].is_null() {
                            accum.count += 1;
                        }
                    }
                    AggregateOp::Sum | AggregateOp::Avg => {
                        if *col_idx < row_data.len() {
                            let val = &*row_data[*col_idx];
                            let f_opt = match val {
                                Value::Integer(i) => Some(*i as f64),
                                Value::Float(f) => Some(*f),
                                _ => None,
                            };
                            if let Some(v) = f_opt {
                                accum.sum += v;
                                accum.count += 1;
                            }
                        }
                    }
                    AggregateOp::Min => {
                        if *col_idx < row_data.len() {
                            let val = &row_data[*col_idx];
                            if !val.is_null() {
                                match &accum.min {
                                    None => accum.min = Some((**val).clone()),
                                    Some(current) => {
                                        if **val < *current {
                                            accum.min = Some((**val).clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AggregateOp::Max => {
                        if *col_idx < row_data.len() {
                            let val = &row_data[*col_idx];
                            if !val.is_null() {
                                match &accum.max {
                                    None => accum.max = Some((**val).clone()),
                                    Some(current) => {
                                        if **val > *current {
                                            accum.max = Some((**val).clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to results
        let mut results: Vec<GroupedAggregateResult> = Vec::with_capacity(groups.len());

        for (group_key, accums) in groups {
            // Extract group values
            let group_values = match group_key {
                GroupKey::Single(v) => vec![(*v).clone()],
                GroupKey::Multi(vs) => vs.iter().map(|v| (**v).clone()).collect(),
            };

            // Compute final aggregate values
            let aggregate_values: Vec<Value> = aggregates
                .iter()
                .zip(accums.iter())
                .map(|((op, _), accum)| match op {
                    AggregateOp::Count | AggregateOp::CountStar => Value::Integer(accum.count),
                    AggregateOp::Sum => {
                        if accum.count > 0 {
                            Value::Float(accum.sum)
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Avg => {
                        if accum.count > 0 {
                            Value::Float(accum.sum / accum.count as f64)
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Min => accum.min.clone().unwrap_or(Value::Null(DataType::Null)),
                    AggregateOp::Max => accum.max.clone().unwrap_or(Value::Null(DataType::Null)),
                })
                .collect();

            results.push(GroupedAggregateResult {
                group_values,
                aggregate_values,
            });
        }

        results
    }
}

impl Clone for VersionChainEntry {
    fn clone(&self) -> Self {
        Self {
            version: self.version.clone(),
            prev: self.prev.clone(), // Arc clone is O(1)
            arena_idx: self.arena_idx,
            chain_depth: self.chain_depth,
        }
    }
}

impl fmt::Debug for VersionStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VersionStore")
            .field("table_name", &self.table_name)
            .field("row_count", &self.row_count())
            .field("closed", &self.closed.load(Ordering::Acquire))
            .finish()
    }
}

/// Transaction-local version store for uncommitted changes
pub struct TransactionVersionStore {
    /// Local versions for this transaction - stores version history per row for savepoint support
    /// The list is ordered by create_time (oldest first, newest last)
    /// Lazily allocated on first write to avoid allocation overhead for read-only queries.
    /// Uses SmallVec<[RowVersion; 1]> to avoid heap allocation for single-version rows.
    local_versions: Option<Int64Map<VersionList>>,
    /// Parent (shared) version store
    parent_store: Arc<VersionStore>,
    /// This transaction's ID
    txn_id: i64,
    /// Write set for conflict detection
    /// Lazily allocated on first write to avoid allocation overhead for read-only queries
    write_set: Option<Int64Map<WriteSetEntry>>,
}

impl TransactionVersionStore {
    /// Creates a new transaction-local version store
    ///
    /// Uses lazy allocation for local_versions and write_set maps to avoid
    /// allocation overhead for read-only queries. These are only allocated
    /// when the first write operation occurs.
    pub fn new(parent_store: Arc<VersionStore>, txn_id: i64) -> Self {
        Self {
            // Lazy allocation - maps are created on first write
            local_versions: None,
            parent_store,
            txn_id,
            write_set: None,
        }
    }

    /// Returns the transaction ID
    pub fn txn_id(&self) -> i64 {
        self.txn_id
    }

    /// Ensures local_versions map is allocated, returning a mutable reference.
    /// Uses pooled maps when available to reduce allocation overhead.
    #[inline]
    fn ensure_local_versions(&mut self) -> &mut Int64Map<VersionList> {
        self.local_versions.get_or_insert_with(get_version_list_map)
    }

    /// Ensures write_set map is allocated, returning a mutable reference.
    /// Uses pooled maps when available to reduce allocation overhead.
    #[inline]
    fn ensure_write_set(&mut self) -> &mut Int64Map<WriteSetEntry> {
        self.write_set.get_or_insert_with(get_write_set_map)
    }

    /// Put adds or updates a row in the transaction's local store
    pub fn put(&mut self, row_id: i64, data: Row, is_delete: bool) -> Result<(), Error> {
        // Create the row version
        let mut rv = RowVersion::new(self.txn_id, row_id, data);
        if is_delete {
            rv.deleted_at_txn_id = self.txn_id;
        }

        // Check if we already have a local version for this row
        let has_local = self
            .local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(&row_id));

        if has_local {
            // Already have local version - just append
            self.ensure_local_versions()
                .get_mut(&row_id)
                .unwrap()
                .push(rv);
        } else {
            // New row - need to check write-set and parent store
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(&row_id));

            if needs_write_set_entry {
                let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                let row_exists = read_version.is_some();

                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.ensure_write_set().insert(
                    row_id,
                    WriteSetEntry {
                        read_version,
                        read_version_seq,
                    },
                );

                // For existing rows, try to claim them
                if row_exists {
                    self.parent_store.try_claim_row(row_id, self.txn_id)?;
                }
            }
            self.ensure_local_versions().insert(row_id, smallvec![rv]);
        }
        Ok(())
    }

    /// Batch put for UPDATE operations where we already have the row data
    ///
    /// This is used for rows that are already tracked in local_versions (updates within same txn)
    /// or when we don't have pre-fetched original versions.
    pub fn put_batch_for_update(&mut self, rows: RowVec) -> Result<(), Error> {
        for (row_id, data) in rows {
            self.put(row_id, data, false)?;
        }
        Ok(())
    }

    /// Optimized single-row put with pre-fetched original version
    ///
    /// This avoids redundant get_visible_version() calls by accepting the original
    /// version that was already fetched during the read phase.
    /// Used for PK-based UPDATE operations.
    #[inline]
    pub fn put_with_original(
        &mut self,
        row_id: i64,
        data: Row,
        original_version: RowVersion,
        is_delete: bool,
    ) -> Result<(), Error> {
        // Create the new row version
        let mut rv = RowVersion::new(self.txn_id, row_id, data);
        if is_delete {
            rv.deleted_at_txn_id = self.txn_id;
        }

        // Check if we already have a local version for this row
        let has_local = self
            .local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(&row_id));

        if has_local {
            // Already have local version - just append
            self.ensure_local_versions()
                .get_mut(&row_id)
                .unwrap()
                .push(rv);
        } else {
            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(&row_id));

            if needs_write_set_entry {
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.ensure_write_set().insert(
                    row_id,
                    WriteSetEntry {
                        read_version: Some(original_version),
                        read_version_seq,
                    },
                );

                // Claim the row for update
                self.parent_store.try_claim_row(row_id, self.txn_id)?;
            }
            self.ensure_local_versions().insert(row_id, smallvec![rv]);
        }
        Ok(())
    }

    /// Optimized batch put for UPDATE operations with pre-fetched original versions
    ///
    /// This avoids redundant get_visible_version() calls by accepting the original
    /// versions that were already fetched during the batch read.
    ///
    /// Parameters:
    /// - rows: Vec of (row_id, new_row_data, original_version)
    ///   where original_version.create_time contains the read_version_seq
    pub fn put_batch_with_originals(
        &mut self,
        rows: Vec<(i64, Row, RowVersion)>,
    ) -> Result<(), Error> {
        let now = get_fast_timestamp();

        for (row_id, data, original_version) in rows {
            // Create the new row version
            let mut rv = RowVersion::new(self.txn_id, row_id, data);
            rv.create_time = now;

            // Check if already in local versions (already processed in this transaction)
            if let Some(versions) = self.ensure_local_versions().get_mut(&row_id) {
                // Append new version to history
                versions.push(rv);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(&row_id));

            if needs_write_set_entry {
                // Get current sequence for conflict detection
                // Note: We get a fresh sequence instead of relying on create_time because
                // callers may use get_visible_version() which doesn't set create_time to the sequence
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);
                self.ensure_write_set().insert(
                    row_id,
                    WriteSetEntry {
                        read_version: Some(original_version),
                        read_version_seq,
                    },
                );

                // Claim the row for update
                self.parent_store.try_claim_row(row_id, self.txn_id)?;
            }

            // Insert new version history for this row
            self.ensure_local_versions().insert(row_id, smallvec![rv]);
        }
        Ok(())
    }

    /// Optimized batch delete for DELETE operations
    ///
    /// This marks multiple rows as deleted in a single operation, avoiding
    /// the overhead of individual put() calls with lock acquisitions per row.
    ///
    /// Parameters:
    /// - rows: RowVec of (row_id, row_data) to mark as deleted
    pub fn put_batch_deleted(&mut self, rows: RowVec) -> Result<(), Error> {
        for (row_id, data) in rows {
            // Check if we already have a local version
            let has_local = self
                .local_versions
                .as_ref()
                .is_some_and(|lv| lv.contains_key(&row_id));

            if !has_local {
                // Check if this row exists in parent store and track in write-set
                let needs_write_set_entry = self
                    .write_set
                    .as_ref()
                    .is_none_or(|ws| !ws.contains_key(&row_id));

                if needs_write_set_entry {
                    let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                    let row_exists = read_version.is_some();

                    let read_version_seq = self
                        .parent_store
                        .visibility_checker
                        .as_ref()
                        .map(|c| c.get_current_sequence())
                        .unwrap_or(0);

                    self.ensure_write_set().insert(
                        row_id,
                        WriteSetEntry {
                            read_version,
                            read_version_seq,
                        },
                    );

                    // For existing rows, try to claim them
                    if row_exists {
                        self.parent_store.try_claim_row(row_id, self.txn_id)?;
                    }
                }
            }

            // Create deleted row version
            let mut rv = RowVersion::new(self.txn_id, row_id, data);
            rv.deleted_at_txn_id = self.txn_id;

            // Append to version history for this row
            let local_versions = self.ensure_local_versions();
            if let Some(versions) = local_versions.get_mut(&row_id) {
                versions.push(rv);
            } else {
                local_versions.insert(row_id, smallvec![rv]);
            }
        }
        Ok(())
    }

    /// Optimized batch delete with pre-fetched original versions
    ///
    /// This avoids redundant get_visible_version() calls by accepting the original
    /// versions that were already fetched during the read phase.
    /// Used for PK range DELETE operations.
    pub fn put_batch_deleted_with_originals(
        &mut self,
        rows: Vec<(i64, Row, RowVersion)>,
    ) -> Result<(), Error> {
        for (row_id, data, original_version) in rows {
            // Create deleted row version
            let mut rv = RowVersion::new(self.txn_id, row_id, data);
            rv.deleted_at_txn_id = self.txn_id;

            // Check if already in local versions (already processed in this transaction)
            if let Some(versions) = self.ensure_local_versions().get_mut(&row_id) {
                versions.push(rv);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(&row_id));

            if needs_write_set_entry {
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.ensure_write_set().insert(
                    row_id,
                    WriteSetEntry {
                        read_version: Some(original_version),
                        read_version_seq,
                    },
                );

                // Claim the row for delete
                self.parent_store.try_claim_row(row_id, self.txn_id)?;
            }

            // Insert deleted version for this row
            self.ensure_local_versions().insert(row_id, smallvec![rv]);
        }
        Ok(())
    }

    /// Check if we have local changes for a row
    pub fn has_locally_seen(&self, row_id: i64) -> bool {
        self.local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(&row_id))
    }

    /// Returns true if this transaction has any uncommitted local changes
    pub fn has_local_changes(&self) -> bool {
        self.local_versions
            .as_ref()
            .is_some_and(|lv| !lv.is_empty())
    }

    /// Iterate over local versions (returns most recent version per row)
    pub fn iter_local(&self) -> impl Iterator<Item = (i64, &RowVersion)> {
        self.local_versions
            .iter()
            .flat_map(|lv| lv.iter())
            .filter_map(|(k, versions)| versions.last().map(|v| (*k, v)))
    }

    /// Iterate over local versions with their original (old) versions for index updates
    /// Returns (row_id, new_version, old_row_option)
    pub fn iter_local_with_old(&self) -> impl Iterator<Item = (i64, &RowVersion, Option<&Row>)> {
        let write_set_ref = self.write_set.as_ref();
        self.local_versions
            .iter()
            .flat_map(|lv| lv.iter())
            .filter_map(move |(row_id, versions)| {
                versions.last().map(|version| {
                    let old_row = write_set_ref
                        .and_then(|ws| ws.get(row_id))
                        .and_then(|entry| entry.read_version.as_ref())
                        .filter(|v| !v.is_deleted())
                        .map(|v| &v.data);
                    (*row_id, version, old_row)
                })
            })
    }

    /// Get the local version for a row (without checking parent)
    /// Returns the most recent version in the transaction's history
    pub fn get_local_version(&self, row_id: i64) -> Option<&RowVersion> {
        self.local_versions
            .as_ref()
            .and_then(|lv| lv.get(&row_id))
            .and_then(|versions| versions.last())
    }

    /// Get a row, checking local versions first then parent store
    pub fn get(&self, row_id: i64) -> Option<Row> {
        // Check local versions first (get most recent)
        if let Some(lv) = self.local_versions.as_ref() {
            if let Some(versions) = lv.get(&row_id) {
                if let Some(local_version) = versions.last() {
                    if local_version.is_deleted() {
                        return None;
                    }
                    return Some(local_version.data.clone());
                }
            }
        }

        // Check parent store
        self.parent_store
            .get_visible_version(row_id, self.txn_id)
            .map(|v| v.data.clone())
    }

    /// Detect conflicts before commit
    ///
    /// Optimized to skip redundant checks for rows we've successfully claimed.
    /// Since try_claim_row() prevents other transactions from modifying claimed rows,
    /// we only need to verify conflicts for:
    /// 1. New inserts (read_version is None) - check if row was inserted by another txn
    /// 2. Unclaimed rows (shouldn't happen with current code paths)
    pub fn detect_conflicts(&self) -> Result<(), Error> {
        // Fast path: no write set means no writes, no conflicts possible
        let Some(write_set) = self.write_set.as_ref() else {
            return Ok(());
        };

        // OPTIMIZATION: For single-row auto-commit (common case), avoid Vec allocation
        // by checking directly without collecting
        if write_set.len() == 1 {
            // Single entry - check directly
            if let Some((row_id, write_entry)) = write_set.iter().next() {
                if write_entry.read_version.is_none() {
                    // Single insert - check conflict directly without Vec
                    if self
                        .parent_store
                        .has_any_visible_version(&[*row_id], self.txn_id)
                        .is_some()
                    {
                        return Err(Error::internal(format!(
                            "write conflict: row {} was inserted by another transaction",
                            row_id
                        )));
                    }
                }
            }
            return Ok(());
        }

        // Multi-row path: collect row_ids for inserts that need conflict checking
        let insert_row_ids: Vec<i64> = write_set
            .iter()
            .filter_map(|(row_id, write_entry)| {
                if write_entry.read_version.is_none() {
                    Some(*row_id)
                } else {
                    None
                }
            })
            .collect();

        // Fast path: no inserts, no conflicts possible
        if insert_row_ids.is_empty() {
            return Ok(());
        }

        // Batch check: see if any insert row_id has a visible version (conflict)
        // OPTIMIZATION: Single lock acquisition, early exit on first conflict
        if let Some(conflicting_row_id) = self
            .parent_store
            .has_any_visible_version(&insert_row_ids, self.txn_id)
        {
            return Err(Error::internal(format!(
                "write conflict: row {} was inserted by another transaction",
                conflicting_row_id
            )));
        }

        Ok(())
    }

    /// Prepare commit - returns list of versions to commit (most recent per row)
    pub fn prepare_commit(&self) -> Vec<(i64, RowVersion)> {
        let Some(local_versions) = self.local_versions.as_ref() else {
            return Vec::new();
        };

        let mut versions = Vec::new();
        for (row_id, version_history) in local_versions.iter() {
            // Only commit the most recent version per row
            if let Some(version) = version_history.last() {
                versions.push((*row_id, version.clone()));
            }
        }
        versions
    }

    /// Commit local changes to parent store
    ///
    /// Performance: This method drains local_versions to take ownership of
    /// RowVersion values, avoiding expensive clones. The transaction is
    /// consumed after commit anyway, so this is safe.
    pub fn commit(&mut self) -> Result<(), Error> {
        // Detect conflicts first
        self.detect_conflicts()?;

        // OPTIMIZATION: Single-row fast path (common auto-commit INSERT case)
        // Avoids Vec allocation for both local_versions and write_set
        if let Some(local_versions) = self.local_versions.as_mut() {
            if local_versions.len() == 1 {
                // Single-row commit - no Vec allocation needed
                if let Some((row_id, mut versions)) = local_versions.drain().next() {
                    if let Some(version) = versions.pop() {
                        self.parent_store.add_version_single(row_id, version);
                    }
                }
                // Release single claim directly
                if let Some(write_set) = self.write_set.as_mut() {
                    if let Some((row_id, _)) = write_set.drain().next() {
                        self.parent_store.release_row_claim(row_id, self.txn_id);
                    }
                }
                return Ok(());
            }

            // Multi-row path: collect into Vec
            let batch: Vec<(i64, RowVersion)> = local_versions
                .drain()
                .filter_map(|(row_id, mut versions)| versions.pop().map(|v| (row_id, v)))
                .collect();

            self.parent_store.add_versions_batch(batch);
        }

        // Release all claims - drain write_set to get row_ids
        if let Some(write_set) = self.write_set.as_mut() {
            let row_ids: Vec<i64> = write_set.drain().map(|(row_id, _)| row_id).collect();
            self.parent_store
                .release_row_claims_batch(&row_ids, self.txn_id);
        }

        Ok(())
    }

    /// Rollback - discard local changes and release claims
    pub fn rollback(&self) {
        self.release_all_claims();
    }

    /// Rollback to a specific timestamp (for savepoint support)
    ///
    /// Discards all local changes that were made after the given timestamp.
    /// For rows with version history, keeps versions at or before the timestamp.
    /// Row claims are released only if all versions for that row are discarded.
    pub fn rollback_to_timestamp(&mut self, timestamp: i64) {
        let Some(local_versions) = self.local_versions.as_mut() else {
            return;
        };

        let mut rows_to_remove_completely: Vec<i64> = Vec::new();

        // For each row, remove versions with create_time > timestamp
        for (row_id, versions) in local_versions.iter_mut() {
            // Keep only versions at or before the timestamp
            versions.retain(|v| v.create_time <= timestamp);

            // If all versions are removed, mark for complete removal
            if versions.is_empty() {
                rows_to_remove_completely.push(*row_id);
            }
        }

        // Remove rows with no remaining versions and release their claims
        for row_id in &rows_to_remove_completely {
            local_versions.remove(row_id);
            self.parent_store.release_row_claim(*row_id, self.txn_id);
            if let Some(write_set) = self.write_set.as_mut() {
                write_set.remove(row_id);
            }
        }
    }

    /// Release all row claims held by this transaction
    fn release_all_claims(&self) {
        let Some(write_set) = self.write_set.as_ref() else {
            return;
        };
        // OPTIMIZATION: Collect row_ids first, then batch release
        // Avoids holding write_set iterator while accessing parent_store
        let row_ids: Vec<i64> = write_set.keys().copied().collect();
        self.parent_store
            .release_row_claims_batch(&row_ids, self.txn_id);
    }
}

impl Drop for TransactionVersionStore {
    fn drop(&mut self) {
        // Return maps to the pool for reuse by future transactions.
        // This reduces allocation overhead from ~5.5KB per transaction to near zero
        // for bulk insert workloads where many short-lived transactions are created.
        if let Some(map) = self.local_versions.take() {
            return_version_list_map(map);
        }
        if let Some(map) = self.write_set.take() {
            return_write_set_map(map);
        }
    }
}

impl fmt::Debug for TransactionVersionStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransactionVersionStore")
            .field("txn_id", &self.txn_id)
            .field(
                "local_version_count",
                &self.local_versions.as_ref().map_or(0, |lv| lv.len()),
            )
            .field(
                "write_set_count",
                &self.write_set.as_ref().map_or(0, |ws| ws.len()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use std::sync::atomic::AtomicI64;

    /// Simple visibility checker for testing
    struct TestVisibilityChecker {
        current_seq: AtomicI64,
    }

    impl TestVisibilityChecker {
        fn new() -> Self {
            Self {
                current_seq: AtomicI64::new(0),
            }
        }
    }

    impl VisibilityChecker for TestVisibilityChecker {
        fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
            // Simple rule: a version is visible if it was created by a transaction
            // with a lower or equal ID (simplified for testing)
            version_txn_id <= viewing_txn_id
        }

        fn get_current_sequence(&self) -> i64 {
            self.current_seq.fetch_add(1, Ordering::AcqRel)
        }

        fn get_active_transaction_ids(&self) -> Vec<i64> {
            // No active transactions in test
            Vec::new()
        }
    }

    #[test]
    fn test_row_version_creation() {
        let row = Row::from(vec![Value::from(1), Value::from("test")]);
        let version = RowVersion::new(1, 100, row);

        assert_eq!(version.txn_id, 1);
        assert_eq!(version.row_id, 100);
        assert!(!version.is_deleted());
        assert!(version.create_time > 0);
    }

    #[test]
    fn test_row_version_deleted() {
        let row = Row::from(vec![Value::from(1)]);
        let version = RowVersion::new_deleted(1, 100, row);

        assert!(version.is_deleted());
        assert_eq!(version.deleted_at_txn_id, 1);
    }

    use crate::core::SchemaBuilder;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test_table").build()
    }

    #[test]
    fn test_version_store_auto_increment() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        assert_eq!(store.get_current_auto_increment_value(), 0);
        assert_eq!(store.get_next_auto_increment_id(), 1);
        assert_eq!(store.get_next_auto_increment_id(), 2);
        assert_eq!(store.get_current_auto_increment_value(), 2);
    }

    #[test]
    fn test_version_store_set_auto_increment() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        assert!(store.set_auto_increment_counter(10));
        assert_eq!(store.get_current_auto_increment_value(), 10);

        // Should not go backwards
        assert!(!store.set_auto_increment_counter(5));
        assert_eq!(store.get_current_auto_increment_value(), 10);

        // Should update if higher
        assert!(store.set_auto_increment_counter(20));
        assert_eq!(store.get_current_auto_increment_value(), 20);
    }

    #[test]
    fn test_version_store_add_and_get() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);

        store.add_version(100, version);

        // Transaction 2 should see version from transaction 1
        let visible = store.get_visible_version(100, 2);
        assert!(visible.is_some());
        assert_eq!(visible.unwrap().txn_id, 1);
    }

    #[test]
    fn test_version_store_visibility() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add version from transaction 5
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(5, 100, row);
        store.add_version(100, version);

        // Transaction 3 should NOT see version from transaction 5
        let visible = store.get_visible_version(100, 3);
        assert!(visible.is_none());

        // Transaction 5 should see its own version
        let visible = store.get_visible_version(100, 5);
        assert!(visible.is_some());

        // Transaction 10 should see version from transaction 5
        let visible = store.get_visible_version(100, 10);
        assert!(visible.is_some());
    }

    #[test]
    fn test_version_store_deleted_row() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add version from transaction 1
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row.clone());
        store.add_version(100, version);

        // Delete in transaction 2
        let deleted_version = RowVersion::new_deleted(2, 100, row);
        store.add_version(100, deleted_version);

        // Transaction 1 should still see the row (delete not visible)
        let visible = store.get_visible_version(100, 1);
        assert!(visible.is_some());

        // Transaction 3 should NOT see the deleted row
        let visible = store.get_visible_version(100, 3);
        assert!(visible.is_none());
    }

    #[test]
    fn test_version_store_row_ids() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        let row = Row::from(vec![Value::from(1)]);
        store.add_version(100, RowVersion::new(1, 100, row.clone()));
        store.add_version(200, RowVersion::new(1, 200, row.clone()));
        store.add_version(300, RowVersion::new(1, 300, row));

        let row_ids = store.get_all_row_ids();
        assert_eq!(row_ids.len(), 3);
        assert!(row_ids.contains(&100));
        assert!(row_ids.contains(&200));
        assert!(row_ids.contains(&300));
    }

    #[test]
    fn test_version_store_close() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        assert!(!store.is_closed());
        store.close();
        assert!(store.is_closed());

        // Operations should be no-ops when closed
        let row = Row::from(vec![Value::from(1)]);
        store.add_version(100, RowVersion::new(1, 100, row));
        assert_eq!(store.row_count(), 0);
    }

    #[test]
    fn test_transaction_version_store_basic() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        let mut tvs = TransactionVersionStore::new(store, 1);

        // Put a new row
        let row = Row::from(vec![Value::from(42)]);
        tvs.put(100, row, false).unwrap();

        // Should see it locally
        assert!(tvs.has_locally_seen(100));
        let got = tvs.get(100);
        assert!(got.is_some());
    }

    #[test]
    fn test_transaction_version_store_commit() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        let mut tvs = TransactionVersionStore::new(Arc::clone(&store), 1);

        // Put a new row
        let row = Row::from(vec![Value::from(42)]);
        tvs.put(100, row, false).unwrap();

        // Commit
        tvs.commit().unwrap();

        // Should be visible in parent store now
        let visible = store.get_visible_version(100, 2);
        assert!(visible.is_some());
    }

    #[test]
    fn test_transaction_version_store_rollback() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        let mut tvs = TransactionVersionStore::new(Arc::clone(&store), 1);

        // Put a new row
        let row = Row::from(vec![Value::from(42)]);
        tvs.put(100, row, false).unwrap();

        // Rollback
        tvs.rollback();

        // Should NOT be visible in parent store
        let visible = store.get_visible_version(100, 2);
        assert!(visible.is_none());
    }

    #[test]
    fn test_version_history_limit_default() {
        let store = VersionStore::new("test_table".to_string(), test_schema());
        // Default limit is 10
        assert_eq!(store.max_version_history(), 10);
    }

    #[test]
    fn test_version_history_limit_drop() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let mut store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Set a small limit for testing
        store.set_max_version_history(3);

        let row_id = 100;

        // Add 5 versions to the same row
        // v1: depth=1, v2: depth=2, v3: depth=3
        // v4: depth=4 > 3, triggers drop -> depth=2
        // v5: depth=3
        for txn_id in 1..=5 {
            let row = Row::from(vec![Value::from(txn_id)]);
            let version = RowVersion::new(txn_id, row_id, row);
            store.add_version(row_id, version);
        }

        // After 5 versions with limit 3:
        // v4 triggered drop (4 > 3), so chain was: v4 -> v3 -> None (depth=2)
        // v5 added: v5 -> v4 -> v3 -> None (depth=3)
        let versions = store.versions.read();
        let entry = versions.get(&row_id).expect("Row should exist");

        // Chain depth should be at most limit + 1 (oscillates between 2 and limit+1)
        assert!(
            entry.chain_depth <= 4,
            "Chain depth {} exceeds limit+1",
            entry.chain_depth
        );

        // Count actual chain length
        let mut count = 1;
        let mut current = entry.prev.as_ref();
        while let Some(prev) = current {
            count += 1;
            current = prev.prev.as_ref();
        }

        // Should be <= limit + 1
        assert!(count <= 4, "Actual chain length {} exceeds limit+1", count);
    }

    #[test]
    fn test_version_history_drop_cycles() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let mut store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Set limit to 5
        store.set_max_version_history(5);

        let row_id = 100;

        // Add 20 versions - drop should happen multiple times
        // Pattern: 1,2,3,4,5,6(drop->2),3,4,5,6(drop->2),...
        for txn_id in 1..=20 {
            let row = Row::from(vec![Value::from(txn_id)]);
            let version = RowVersion::new(txn_id, row_id, row);
            store.add_version(row_id, version);
        }

        // Verify chain is bounded (between 2 and limit+1)
        let versions = store.versions.read();
        let entry = versions.get(&row_id).expect("Row should exist");

        let mut count = 1;
        let mut current = entry.prev.as_ref();
        while let Some(prev) = current {
            count += 1;
            current = prev.prev.as_ref();
        }

        assert!(
            count <= 6,
            "Chain length {} exceeds limit+1 (6) after 20 updates",
            count
        );
        assert!(count >= 2, "Chain length {} should be at least 2", count);
    }

    #[test]
    fn test_version_history_unlimited() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let mut store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Set to unlimited (0)
        store.set_max_version_history(0);

        let row_id = 100;

        // Add 15 versions
        for txn_id in 1..=15 {
            let row = Row::from(vec![Value::from(txn_id)]);
            let version = RowVersion::new(txn_id, row_id, row);
            store.add_version(row_id, version);
        }

        // Count chain length - should be 15 (unlimited)
        let versions = store.versions.read();
        let entry = versions.get(&row_id).expect("Row should exist");

        let mut count = 1;
        let mut current = entry.prev.as_ref();
        while let Some(prev) = current {
            count += 1;
            current = prev.prev.as_ref();
        }

        assert_eq!(count, 15, "Unlimited mode should keep all 15 versions");
    }

    #[test]
    fn test_version_history_batch_drop() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let mut store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Set limit to 3
        store.set_max_version_history(3);

        let row_id = 100;

        // First add some versions individually (depth reaches 3)
        for txn_id in 1..=3 {
            let row = Row::from(vec![Value::from(txn_id)]);
            let version = RowVersion::new(txn_id, row_id, row);
            store.add_version(row_id, version);
        }

        // Now add more via batch - each will trigger drop when exceeding limit
        let batch: Vec<(i64, RowVersion)> = (4..=7)
            .map(|txn_id| {
                let row = Row::from(vec![Value::from(txn_id)]);
                (row_id, RowVersion::new(txn_id, row_id, row))
            })
            .collect();

        store.add_versions_batch(batch);

        // Verify chain is bounded (at most limit+1)
        // Chain can be as short as 1 right after pruning (when new_depth > limit)
        let versions = store.versions.read();
        let entry = versions.get(&row_id).expect("Row should exist");

        let mut count = 1;
        let mut current = entry.prev.as_ref();
        while let Some(prev) = current {
            count += 1;
            current = prev.prev.as_ref();
        }

        assert!(
            count <= 4,
            "Chain length {} exceeds limit+1 (4) after batch",
            count
        );
        // With consistent logic: after 7 versions with limit 3,
        // the 7th version triggers pruning (depth 4 > 3), resulting in depth = 1
        assert!(count >= 1, "Chain length {} should be at least 1", count);
    }

    #[test]
    fn test_row_version_with_timestamp() {
        let row = Row::from(vec![Value::from(1)]);
        let timestamp = 12345678;
        let version = RowVersion::new_with_timestamp(1, 100, row, timestamp);

        assert_eq!(version.txn_id, 1);
        assert_eq!(version.row_id, 100);
        assert_eq!(version.create_time, timestamp);
        assert!(!version.is_deleted());
    }

    #[test]
    fn test_row_version_deleted_with_timestamp() {
        let row = Row::from(vec![Value::from(1)]);
        let timestamp = 87654321;
        let version = RowVersion::new_deleted_with_timestamp(1, 100, row, timestamp);

        assert_eq!(version.txn_id, 1);
        assert_eq!(version.deleted_at_txn_id, 1);
        assert_eq!(version.create_time, timestamp);
        assert!(version.is_deleted());
    }

    #[test]
    fn test_row_version_debug_display() {
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);

        let debug = format!("{:?}", version);
        assert!(debug.contains("RowVersion"));
        assert!(debug.contains("txn_id: 1"));
        assert!(debug.contains("row_id: 100"));

        let display = format!("{}", version);
        assert!(display.contains("TxnID: 1"));
        assert!(display.contains("RowID: 100"));
    }

    #[test]
    fn test_write_set_entry_clone() {
        let row = Row::from(vec![Value::from(1)]);
        let version = RowVersion::new(1, 100, row);

        let entry = WriteSetEntry {
            read_version: Some(version),
            read_version_seq: 42,
        };

        let cloned = entry.clone();
        assert!(cloned.read_version.is_some());
        assert_eq!(cloned.read_version_seq, 42);

        // Test with None
        let empty_entry = WriteSetEntry {
            read_version: None,
            read_version_seq: 0,
        };
        let cloned_empty = empty_entry.clone();
        assert!(cloned_empty.read_version.is_none());
    }

    #[test]
    fn test_row_index() {
        let idx = RowIndex {
            row_id: 100,
            arena_idx: Some(5),
        };

        // Test Copy trait
        let copied = idx;
        assert_eq!(copied.row_id, 100);
        assert_eq!(copied.arena_idx, Some(5));

        // Test Clone trait (use Clone::clone to avoid clone_on_copy warning)
        let cloned = Clone::clone(&idx);
        assert_eq!(cloned.row_id, 100);

        // Test with None arena_idx
        let idx_none = RowIndex {
            row_id: 200,
            arena_idx: None,
        };
        assert!(idx_none.arena_idx.is_none());

        // Test Debug
        let debug = format!("{:?}", idx);
        assert!(debug.contains("RowIndex"));
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_aggregate_op() {
        // Test equality
        assert_eq!(AggregateOp::Count, AggregateOp::Count);
        assert_ne!(AggregateOp::Count, AggregateOp::Sum);

        // Test all variants
        let ops = [
            AggregateOp::Count,
            AggregateOp::Sum,
            AggregateOp::Min,
            AggregateOp::Max,
            AggregateOp::Avg,
        ];

        for op in ops {
            let debug = format!("{:?}", op);
            assert!(!debug.is_empty());

            // Test Clone and Copy (use Clone::clone to avoid clone_on_copy warning)
            let copied = op;
            let cloned = Clone::clone(&op);
            assert_eq!(copied, cloned);
        }
    }

    #[test]
    fn test_version_store_with_capacity() {
        let store = VersionStore::with_capacity("test_table".to_string(), test_schema(), None, 100);

        assert_eq!(store.table_name(), "test_table");
        assert_eq!(store.row_count(), 0);
    }

    #[test]
    fn test_version_store_quick_check_row_existence() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        // Row doesn't exist
        assert!(!store.quick_check_row_existence(100));

        // Add a row
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        // Row exists
        assert!(store.quick_check_row_existence(100));
        assert!(!store.quick_check_row_existence(200));
    }

    #[test]
    fn test_version_store_get_visible_versions_batch() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        // Batch query
        let row_ids = vec![1, 3, 5, 99]; // 99 doesn't exist
        let results = store.get_visible_versions_batch(&row_ids, 2);

        assert_eq!(results.len(), 3); // Only 1, 3, 5 exist
    }

    #[test]
    fn test_version_store_count_visible_versions_batch() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let count = store.count_visible_versions_batch(&[1, 2, 3, 99, 100], 2);
        assert_eq!(count, 3); // Only 1, 2, 3 exist
    }

    #[test]
    fn test_version_store_count_visible_rows() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        assert_eq!(store.count_visible_rows(1), 0);

        // Add rows
        for i in 1..=10 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        assert_eq!(store.count_visible_rows(2), 10);
    }

    #[test]
    fn test_version_store_mark_deleted() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add a row
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        // Mark it deleted
        store.mark_deleted(100, 2);

        // Transaction 1 should still see it
        assert!(store.get_visible_version(100, 1).is_some());

        // Transaction 3 should not see deleted row
        assert!(store.get_visible_version(100, 3).is_none());

        // Mark non-existent row as deleted (no-op)
        store.mark_deleted(999, 2);
    }

    #[test]
    fn test_version_store_get_visible_rows_with_limit() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add 20 rows
        for i in 1..=20 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        // Get with limit (txn_id, limit, offset)
        let results = store.get_visible_rows_with_limit(2, 5, 0);
        assert_eq!(results.len(), 5);

        // Test with offset
        let results_offset = store.get_visible_rows_with_limit(2, 5, 10);
        assert_eq!(results_offset.len(), 5);
    }

    #[test]
    fn test_version_store_as_of_transaction() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add version from transaction 5
        let row = Row::from(vec![Value::from(100)]);
        let version = RowVersion::new(5, 1, row);
        store.add_version(1, version);

        // Add updated version from transaction 10
        let row2 = Row::from(vec![Value::from(200)]);
        let version2 = RowVersion::new(10, 1, row2);
        store.add_version(1, version2);

        // AS OF transaction 7 should see the first version
        let result = store.get_visible_version_as_of_transaction(1, 7);
        assert!(result.is_some());
        let rv = result.unwrap();
        assert_eq!(rv.txn_id, 5);

        // AS OF transaction 15 should see the second version
        let result = store.get_visible_version_as_of_transaction(1, 15);
        assert!(result.is_some());
        let rv = result.unwrap();
        assert_eq!(rv.txn_id, 10);

        // AS OF transaction 3 should see nothing
        let result = store.get_visible_version_as_of_transaction(1, 3);
        assert!(result.is_none());

        // Non-existent row
        let result = store.get_visible_version_as_of_transaction(999, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_version_store_as_of_timestamp() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add version with specific timestamp
        let row = Row::from(vec![Value::from(100)]);
        let version = RowVersion::new_with_timestamp(1, 1, row, 1000);
        store.add_version(1, version);

        // Add version with later timestamp
        let row2 = Row::from(vec![Value::from(200)]);
        let version2 = RowVersion::new_with_timestamp(2, 1, row2, 2000);
        store.add_version(1, version2);

        // AS OF timestamp 1500 should see first version
        let result = store.get_visible_version_as_of_timestamp(1, 1500);
        assert!(result.is_some());
        assert_eq!(result.unwrap().create_time, 1000);

        // AS OF timestamp 2500 should see second version
        let result = store.get_visible_version_as_of_timestamp(1, 2500);
        assert!(result.is_some());
        assert_eq!(result.unwrap().create_time, 2000);

        // AS OF timestamp 500 should see nothing
        let result = store.get_visible_version_as_of_timestamp(1, 500);
        assert!(result.is_none());
    }

    #[test]
    fn test_version_store_sum_column() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows with integer values
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let (sum, count) = store.sum_column(2, 0);
        assert_eq!(sum, 150.0); // 10 + 20 + 30 + 40 + 50
        assert_eq!(count, 5);
    }

    #[test]
    fn test_version_store_sum_column_with_floats() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows with float values
        let values = [1.5, 2.5, 3.5];
        for (i, v) in values.iter().enumerate() {
            let row = Row::from(vec![Value::from(*v)]);
            let version = RowVersion::new(1, (i + 1) as i64, row);
            store.add_version((i + 1) as i64, version);
        }

        let (sum, count) = store.sum_column(2, 0);
        assert!((sum - 7.5).abs() < 0.001);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_version_store_min_max_column() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in [30, 10, 50, 20, 40] {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let min = store.min_column(2, 0);
        assert_eq!(min, Some(Value::from(10)));

        let max = store.max_column(2, 0);
        assert_eq!(max, Some(Value::from(50)));
    }

    #[test]
    fn test_version_store_min_max_empty() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        let min = store.min_column(1, 0);
        assert!(min.is_none());

        let max = store.max_column(1, 0);
        assert!(max.is_none());
    }

    #[test]
    fn test_version_store_compute_aggregates() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        // Compute multiple aggregates at once
        let ops = vec![
            (AggregateOp::Count, 0),
            (AggregateOp::Sum, 0),
            (AggregateOp::Min, 0),
            (AggregateOp::Max, 0),
            (AggregateOp::Avg, 0),
        ];

        let results = store.compute_aggregates(2, &ops);
        assert_eq!(results.len(), 5);

        // Check count
        match &results[0] {
            AggregateResult::Count(c) => assert_eq!(*c, 5),
            _ => panic!("Expected Count"),
        }

        // Check sum
        match &results[1] {
            AggregateResult::Sum(s, _) => assert_eq!(*s, 150.0),
            _ => panic!("Expected Sum"),
        }

        // Check min
        match &results[2] {
            AggregateResult::Min(Some(v)) => assert_eq!(*v, Value::from(10)),
            _ => panic!("Expected Min"),
        }

        // Check max
        match &results[3] {
            AggregateResult::Max(Some(v)) => assert_eq!(*v, Value::from(50)),
            _ => panic!("Expected Max"),
        }

        // Check avg (returns sum, count - caller computes sum/count)
        match &results[4] {
            AggregateResult::Avg(sum, count) => {
                let avg = sum / *count as f64;
                assert!((avg - 30.0).abs() < 0.001);
            }
            _ => panic!("Expected Avg"),
        }
    }

    #[test]
    fn test_version_store_stream_visible_rows() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=3 {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let mut stream = store.stream_visible_rows(2);
        let mut count = 0;

        while stream.next() {
            count += 1;
            let slice = stream.row_arc_slice().unwrap();
            assert!(!slice.is_empty());
        }

        assert_eq!(count, 3);
    }

    #[test]
    fn test_version_store_row_claim() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        // Claim a row
        assert!(store.try_claim_row(100, 1).is_ok());

        // Same transaction can claim again
        assert!(store.try_claim_row(100, 1).is_ok());

        // Different transaction should fail
        assert!(store.try_claim_row(100, 2).is_err());

        // Release and another transaction can claim
        store.release_row_claim(100, 1);
        assert!(store.try_claim_row(100, 2).is_ok());
    }

    #[test]
    fn test_version_store_index_operations() {
        use crate::core::types::DataType;
        use crate::storage::mvcc::hash_index::HashIndex;

        let store = VersionStore::new("test_table".to_string(), test_schema());

        // No indexes initially
        assert!(!store.index_exists("idx_test"));
        assert!(store.list_indexes().is_empty());

        // Add an index with all required parameters
        let index = Arc::new(HashIndex::new(
            "idx_test".to_string(),
            "test_table".to_string(),
            vec!["test_col".to_string()],
            vec![0],
            vec![DataType::Integer],
            false,
        ));
        store.add_index("idx_test".to_string(), index);

        assert!(store.index_exists("idx_test"));
        assert_eq!(store.list_indexes().len(), 1);

        // Get index
        assert!(store.get_index("idx_test").is_some());
        assert!(store.get_index("nonexistent").is_none());

        // Get by column
        assert!(store.get_index_by_column("test_col").is_some());

        // Remove index
        let removed = store.remove_index("idx_test");
        assert!(removed.is_some());
        assert!(!store.index_exists("idx_test"));
    }

    #[test]
    fn test_version_store_get_current_sequence() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Sequence should increment
        let seq1 = store.get_current_sequence();
        let seq2 = store.get_current_sequence();
        assert!(seq2 > seq1);
    }

    #[test]
    fn test_version_store_get_visible_row_indices() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let indices = store.get_visible_row_indices(2);
        assert_eq!(indices.len(), 5);

        // Verify indices can be materialized
        let materialized = store.materialize_rows(&indices);
        assert_eq!(materialized.len(), 5);
    }

    #[test]
    fn test_version_store_get_column_value() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add a row with multiple columns
        let row = Row::from(vec![Value::from(42), Value::from("test")]);
        let version = RowVersion::new(1, 1, row);
        store.add_version(1, version);

        let indices = store.get_visible_row_indices(2);
        assert_eq!(indices.len(), 1);

        // Get column values
        let val = store.get_column_value(&indices[0], 0);
        assert_eq!(val, Some(Value::from(42)));

        let val = store.get_column_value(&indices[0], 1);
        assert_eq!(val, Some(Value::from("test")));

        // Out of bounds column
        let val = store.get_column_value(&indices[0], 99);
        assert!(val.is_none());
    }

    #[test]
    fn test_version_store_apply_recovered_version() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        // Apply a recovered version
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.apply_recovered_version(version);

        assert_eq!(store.row_count(), 1);
        assert!(store.quick_check_row_existence(100));
    }

    #[test]
    fn test_version_store_schema_operations() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        // Get schema
        let schema = store.schema();
        assert_eq!(schema.table_name, "test_table");

        // Modify schema through mutable reference
        {
            let schema_guard = store.schema_mut();
            // Just verify we can get mutable access
            assert_eq!(schema_guard.table_name, "test_table");
        }
    }

    #[test]
    fn test_version_store_visibility_checker_setter() {
        let mut store = VersionStore::new("test_table".to_string(), test_schema());

        // Set a new visibility checker
        let checker = Arc::new(TestVisibilityChecker::new());
        store.set_visibility_checker(checker);

        // Verify it works with the new checker
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        let visible = store.get_visible_version(100, 2);
        assert!(visible.is_some());
    }

    #[test]
    fn test_transaction_version_store_update() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        // Add a row first
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        // Start a new transaction and update
        let mut tvs = TransactionVersionStore::new(Arc::clone(&store), 2);

        // Update the row (is_delete = false for updates)
        let new_row = Row::from(vec![Value::from(99)]);
        tvs.put(100, new_row, false).unwrap();

        // Should see updated value locally
        let got = tvs.get(100);
        assert!(got.is_some());
        let data = got.unwrap();
        assert_eq!(data.get(0), Some(&Value::from(99)));

        // Commit
        tvs.commit().unwrap();

        // Updated value should be visible
        let visible = store.get_visible_version(100, 3);
        assert!(visible.is_some());
        assert_eq!(visible.unwrap().data.get(0), Some(&Value::from(99)));
    }

    #[test]
    fn test_transaction_version_store_delete() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        // Add a row first
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        // Start a new transaction and delete
        let mut tvs = TransactionVersionStore::new(Arc::clone(&store), 2);

        // Delete the row by using put with is_delete=true
        let delete_row = Row::from(vec![Value::from(42)]);
        tvs.put(100, delete_row, true).unwrap(); // is_delete = true

        // Should see it as deleted locally
        let got = tvs.get(100);
        assert!(got.is_none());

        // Commit
        tvs.commit().unwrap();

        // Should not be visible after commit
        let visible = store.get_visible_version(100, 3);
        assert!(visible.is_none());
    }

    #[test]
    fn test_get_all_visible_rows() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let rows = store.get_all_visible_rows(2);
        assert_eq!(rows.len(), 5);

        // Verify values
        let values: Vec<i64> = rows
            .iter()
            .map(|(_, row)| match row.get(0) {
                Some(Value::Integer(i)) => *i,
                _ => panic!("Expected integer"),
            })
            .collect();

        assert!(values.contains(&10));
        assert!(values.contains(&50));
    }

    #[test]
    fn test_get_all_visible_row_ids() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        let row_ids = store.get_all_visible_row_ids(2);
        assert_eq!(row_ids.len(), 5);
        assert!(row_ids.contains(&1));
        assert!(row_ids.contains(&5));
    }

    #[test]
    fn test_collect_rows_pk_ordered() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows in non-sequential order
        for i in [5, 3, 1, 4, 2] {
            let row = Row::from(vec![Value::from(i * 10)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        // Get rows in ascending PK order
        let rows = store.collect_rows_pk_ordered(2, true, 3, 0);
        assert!(rows.is_some());
        let rows = rows.unwrap();
        assert_eq!(rows.len(), 3);

        // First 3 rows in ascending PK order should be row_ids 1, 2, 3
        // With values 10, 20, 30
        assert_eq!(rows[0].1.get(0), Some(&Value::from(10)));
        assert_eq!(rows[1].1.get(0), Some(&Value::from(20)));
        assert_eq!(rows[2].1.get(0), Some(&Value::from(30)));

        // Test descending order
        let rows_desc = store.collect_rows_pk_ordered(2, false, 3, 0);
        assert!(rows_desc.is_some());
        let rows_desc = rows_desc.unwrap();
        assert_eq!(rows_desc.len(), 3);

        // First 3 rows in descending PK order should be row_ids 5, 4, 3
        // With values 50, 40, 30
        assert_eq!(rows_desc[0].1.get(0), Some(&Value::from(50)));
        assert_eq!(rows_desc[1].1.get(0), Some(&Value::from(40)));
        assert_eq!(rows_desc[2].1.get(0), Some(&Value::from(30)));

        // Test with offset
        let rows_offset = store.collect_rows_pk_ordered(2, true, 2, 2);
        assert!(rows_offset.is_some());
        let rows_offset = rows_offset.unwrap();
        assert_eq!(rows_offset.len(), 2);
        // Skip first 2 (values 10, 20), get next 2 (values 30, 40)
        assert_eq!(rows_offset[0].1.get(0), Some(&Value::from(30)));
        assert_eq!(rows_offset[1].1.get(0), Some(&Value::from(40)));
    }

    #[test]
    fn test_count_visible() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        assert_eq!(store.count_visible(1), 0);

        // Add rows
        for i in 1..=10 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, i, row);
            store.add_version(i, version);
        }

        assert_eq!(store.count_visible(2), 10);
    }

    #[test]
    fn test_materialize_single_row() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add a row
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, 100, row);
        store.add_version(100, version);

        let indices = store.get_visible_row_indices(2);
        assert_eq!(indices.len(), 1);

        // Materialize single row
        let result = store.materialize_row(&indices[0]);
        assert!(result.is_some());
        let (row_id, row) = result.unwrap();
        assert_eq!(row_id, 100);
        assert_eq!(row.get(0), Some(&Value::from(42)));

        // Invalid index
        let invalid_idx = RowIndex {
            row_id: 999,
            arena_idx: None,
        };
        let result = store.materialize_row(&invalid_idx);
        assert!(result.is_none());
    }
}
