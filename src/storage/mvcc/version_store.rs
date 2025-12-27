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

use parking_lot::RwLock;

use crate::common::{
    new_btree_int64_map, new_concurrent_int64_map, new_int64_map, new_int64_map_with_capacity,
    BTreeInt64Map, ConcurrentInt64Map, Int64Map,
};
use crate::core::{Error, Row, Schema, Value};
use crate::storage::expression::CompiledFilter;
use crate::storage::mvcc::arena::RowArena;
use crate::storage::mvcc::get_fast_timestamp;
use crate::storage::mvcc::streaming_result::{StreamingResult, VisibleRowInfo};
use crate::storage::Index;
// radsort removed - BTreeMap iteration is already ordered
use rayon::prelude::*;
use rustc_hash::FxHashMap;

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
    uncommitted_writes: ConcurrentInt64Map<i64>,
    /// Visibility checker (registry reference)
    visibility_checker: Option<Arc<dyn VisibilityChecker>>,
    /// Arena-based storage for zero-copy full table scans
    arena: RowArena,
    /// Maps row_id to the latest arena index for that row (Int64Map for fast i64 key lookups)
    row_arena_index: RwLock<Int64Map<usize>>,
    /// Zone maps for segment pruning (set by ANALYZE)
    /// Uses Arc to avoid cloning on every read - critical for high QPS workloads
    zone_maps: RwLock<Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>>>,
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
    pub fn with_capacity(
        table_name: String,
        schema: Schema,
        checker: Option<Arc<dyn VisibilityChecker>>,
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
            uncommitted_writes: new_concurrent_int64_map(),
            visibility_checker: checker,
            arena: RowArena::new(),
            row_arena_index,
            zone_maps: RwLock::new(None),
        }
    }

    /// Creates a new version store with a visibility checker
    pub fn with_visibility_checker(
        table_name: String,
        schema: Schema,
        checker: Arc<dyn VisibilityChecker>,
    ) -> Self {
        Self::with_capacity(table_name, schema, Some(checker), 0)
    }

    /// Sets the visibility checker
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

        // Check if existing version exists first
        let existing_data = versions
            .get(&row_id)
            .map(|e| (e.version.clone(), e.prev.clone(), e.arena_idx));

        let entry =
            if let Some((existing_version, existing_prev, existing_arena_idx)) = existing_data {
                // Create new entry with previous version
                let mut new_version = version;
                // For deletes, if no data provided, preserve data from current version
                if new_version.deleted_at_txn_id != 0 && new_version.data.is_empty() {
                    new_version.data = existing_version.data.clone();
                }

                // Mark old arena entry as deleted to avoid double-counting
                if let Some(old_arena_idx) = existing_arena_idx {
                    self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                }

                // Store in arena (only for non-deleted versions)
                // Use insert_row_get_arc to get the Arc back and reuse it for O(1) clones
                let arena_idx = if new_version.deleted_at_txn_id == 0 {
                    let (idx, arc_data) = self.arena.insert_row_get_arc(
                        row_id,
                        new_version.txn_id,
                        new_version.create_time,
                        &new_version.data,
                    );
                    // Reuse the Arc for the version's data - enables O(1) clone on read
                    new_version.data = Row::from_arc(arc_data);
                    // Update row -> arena index mapping
                    self.row_arena_index.write().insert(row_id, idx);
                    Some(idx)
                } else {
                    None
                };

                // Use Arc to share the previous chain - O(1) instead of deep clone
                let prev_chain = Arc::new(VersionChainEntry {
                    version: existing_version,
                    prev: existing_prev,
                    arena_idx: existing_arena_idx,
                });

                VersionChainEntry {
                    version: new_version,
                    prev: Some(prev_chain),
                    arena_idx,
                }
            } else {
                // First version for this row - store in arena
                // Use insert_row_get_arc to get the Arc back and reuse it for O(1) clones
                let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                    let (idx, arc_data) = self.arena.insert_row_get_arc(
                        row_id,
                        version.txn_id,
                        version.create_time,
                        &version.data,
                    );
                    // Update row -> arena index mapping
                    self.row_arena_index.write().insert(row_id, idx);
                    // Create version with Arc-backed data for O(1) clone
                    let mut v = version;
                    v.data = Row::from_arc(arc_data);
                    (Some(idx), v)
                } else {
                    (None, version)
                };

                VersionChainEntry {
                    version: final_version,
                    prev: None,
                    arena_idx,
                }
            };

        versions.insert(row_id, entry);
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
            // Get existing data first (if any)
            let existing_data = versions
                .get(&row_id)
                .map(|e| (e.version.clone(), e.prev.clone(), e.arena_idx));

            let entry = if let Some((existing_version, existing_prev, existing_arena_idx)) =
                existing_data
            {
                // Create new entry with previous version
                let mut new_version = version;
                // For deletes, if no data provided, preserve data from current version
                if new_version.deleted_at_txn_id != 0 && new_version.data.is_empty() {
                    new_version.data = existing_version.data.clone();
                }

                // Mark old arena entry as deleted to avoid double-counting
                if let Some(old_arena_idx) = existing_arena_idx {
                    self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                }

                // Update arena with Arc reuse for O(1) clones on read
                let arena_idx = if new_version.deleted_at_txn_id == 0 {
                    let (idx, arc_data) = self.arena.insert_row_get_arc(
                        row_id,
                        new_version.txn_id,
                        new_version.create_time,
                        &new_version.data,
                    );
                    // Reuse the Arc for the version's data
                    new_version.data = Row::from_arc(arc_data);
                    arena_index_updates.push((row_id, idx));
                    Some(idx)
                } else {
                    None
                };

                // Use Arc to share the previous chain - O(1) instead of deep clone
                let prev_chain = Arc::new(VersionChainEntry {
                    version: existing_version,
                    prev: existing_prev,
                    arena_idx: existing_arena_idx,
                });

                VersionChainEntry {
                    version: new_version,
                    prev: Some(prev_chain),
                    arena_idx,
                }
            } else {
                // First version for this row - store in arena with Arc reuse
                let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                    let (idx, arc_data) = self.arena.insert_row_get_arc(
                        row_id,
                        version.txn_id,
                        version.create_time,
                        &version.data,
                    );
                    arena_index_updates.push((row_id, idx));
                    // Create version with Arc-backed data for O(1) clone
                    let mut v = version;
                    v.data = Row::from_arc(arc_data);
                    (Some(idx), v)
                } else {
                    (None, version)
                };

                VersionChainEntry {
                    version: final_version,
                    prev: None,
                    arena_idx,
                }
            };

            versions.insert(row_id, entry);
        }

        // Batch update row_arena_index under a single lock acquisition
        if !arena_index_updates.is_empty() {
            let mut index = self.row_arena_index.write();
            for (row_id, idx) in arena_index_updates {
                index.insert(row_id, idx);
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

        // BTreeMap with read lock
        let versions = self.versions.read();
        let chain = versions.get(&row_id)?;

        // FAST PATH: Check HEAD version first - O(1) for common case
        // Most readers want latest committed data. If HEAD is visible, return immediately.
        // This avoids chain traversal even for multi-version rows.
        let head_txn_id = chain.version.txn_id;
        let head_deleted_at = chain.version.deleted_at_txn_id;

        if checker.is_visible(head_txn_id, txn_id) {
            // HEAD is visible - check if deleted
            if head_deleted_at != 0 && checker.is_visible(head_deleted_at, txn_id) {
                return None; // Row is deleted
            }
            return Some(chain.version.clone());
        }

        // SLOW PATH: HEAD not visible - need to traverse chain for older versions
        // This only happens for transactions that need to see historical data
        // (e.g., Snapshot Isolation reading versions from before they started)
        let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());

        while let Some(e) = current {
            let version_txn_id = e.version.txn_id;
            let deleted_at_txn_id = e.version.deleted_at_txn_id;

            // Check visibility using only txn_ids (no clone needed yet)
            if checker.is_visible(version_txn_id, txn_id) {
                // Check if deleted and deletion is visible
                if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                    return None;
                }
                // Only clone the ONE version we actually need
                return Some(e.version.clone());
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
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
    pub fn get_visible_versions_batch(&self, row_ids: &[i64], txn_id: i64) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Pre-acquire arena lock ONCE for all lookups (same optimization as full scan)
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Parallel processing constants
        /// Minimum batch size before enabling parallel processing (avoid overhead for small batches)
        const PARALLEL_THRESHOLD: usize = 1000;
        /// Chunk size for parallel processing (balances parallelism vs. overhead)
        const PARALLEL_CHUNK_SIZE: usize = 512;

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap with RwLock - hold read lock during parallel processing
        // Read locks are shared, so concurrent readers can proceed
        // This avoids expensive chain cloning while maintaining MVCC consistency
        let versions = self.versions.read();

        if row_ids.len() >= PARALLEL_THRESHOLD {
            // Parallel path: process chunks while holding read lock (zero-copy)
            // RwLock read locks allow concurrent readers, so this is safe and fast
            row_ids
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .flat_map(|chunk| {
                    let mut chunk_results = Vec::with_capacity(chunk.len());
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
                                    chunk_results.push((row_id, get_row_data(chain)));
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
                                        chunk_results.push((row_id, get_row_data(e)));
                                    }
                                    break;
                                }
                                current = e.prev.as_ref().map(|b| b.as_ref());
                            }
                        }
                    }
                    chunk_results
                })
                .collect()
        } else {
            // Sequential path for small batches
            let mut results = Vec::with_capacity(row_ids.len());
            for &row_id in row_ids {
                if let Some(chain) = versions.get(&row_id) {
                    // FAST PATH: Check HEAD version first - O(1) for common case
                    let head_txn_id = chain.version.txn_id;
                    let head_deleted_at = chain.version.deleted_at_txn_id;

                    if checker.is_visible(head_txn_id, txn_id) {
                        // HEAD is visible - check if deleted
                        if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                            results.push((row_id, get_row_data(chain)));
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
                                results.push((row_id, get_row_data(e)));
                            }
                            break;
                        }
                        current = e.prev.as_ref().map(|b| b.as_ref());
                    }
                }
            }
            results
        }
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
                    return Row::from_arc(Arc::clone(arc_row));
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
    pub fn get_all_visible_rows(&self, txn_id: i64) -> Vec<(i64, Row)> {
        // BTreeMap is already sorted, no additional sorting needed!
        self.get_all_visible_rows_internal(txn_id)
    }

    /// Internal implementation for getting all visible rows
    #[inline]
    fn get_all_visible_rows_internal(&self, txn_id: i64) -> Vec<(i64, Row)> {
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
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            entry.version.data.clone()
        };

        // Collect all versions in one pass (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut results = Vec::with_capacity(versions.len());

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
    pub fn get_all_visible_rows_arena(&self, txn_id: i64) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(versions.len());

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
                    return Row::from_arc(Arc::clone(arc_row));
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
                    // Try arena path first (zero-copy filter check)
                    if let Some(idx) = chain.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter on slice - no allocation!
                            if compiled_filter.matches_slice(arc_row) {
                                let mut version_copy = chain.version.clone();
                                version_copy.create_time = current_seq;
                                result.push((
                                    row_id,
                                    Row::from_arc(Arc::clone(arc_row)),
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
                        // Try arena path first
                        if let Some(idx) = e.arena_idx {
                            if let Some(arc_row) = arena_data.get(idx) {
                                if compiled_filter.matches_slice(arc_row) {
                                    let mut version_copy = e.version.clone();
                                    version_copy.create_time = current_seq;
                                    result.push((
                                        row_id,
                                        Row::from_arc(Arc::clone(arc_row)),
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
    pub fn get_all_visible_rows_unsorted(&self, txn_id: i64) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // Single-pass: read directly from arena during visibility check (BTreeMap is sorted)
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(versions.len());

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
    pub fn get_visible_rows_with_limit(
        &self,
        txn_id: i64,
        limit: usize,
        offset: usize,
    ) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Pre-acquire arena lock ONCE for the entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap is sorted - collect with early termination
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(limit);
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
    ) -> Vec<(i64, Row)> {
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
    ) -> (Vec<(i64, Row)>, bool) {
        if self.closed.load(Ordering::Acquire) || batch_size == 0 {
            return (Vec::new(), false);
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return (Vec::new(), false),
        };

        // Pre-acquire arena lock ONCE for this batch
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper closure to get row data from arena or version
        let get_row_data = |e: &VersionChainEntry| -> Row {
            if let Some(idx) = e.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            e.version.data.clone()
        };

        // BTreeMap is sorted - use range for efficient cursor-based iteration
        let versions = self.versions.read();
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(batch_size);
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
    ) -> Option<Vec<Row>> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Some(Vec::new());
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Some(Vec::new()),
        };

        // Pre-acquire arena lock ONCE for this entire operation
        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Helper to get row data from an entry
        let get_row_from_entry = |entry: &VersionChainEntry| -> Row {
            if let Some(idx) = entry.arena_idx {
                if let Some(arc_row) = arena_data.get(idx) {
                    return Row::from_arc(Arc::clone(arc_row));
                }
            }
            entry.version.data.clone()
        };

        // BTreeMap is already sorted ascending - collect with offset/limit
        let versions = self.versions.read();
        // Cap capacity to avoid overflow when limit is usize::MAX
        let capacity = limit.min(versions.len()).min(10_000);
        let mut result: Vec<Row> = Vec::with_capacity(capacity);
        let mut skipped = 0usize;

        if ascending {
            // Forward iteration with early termination
            for chain in versions.values() {
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
                                result.push(get_row_from_entry(entry));
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
            let mut all_visible: Vec<Row> = Vec::with_capacity(versions.len());
            for chain in versions.values() {
                // Inline visibility check
                let mut current: Option<&VersionChainEntry> = Some(chain);
                while let Some(entry) = current {
                    if checker.is_visible(entry.version.txn_id, txn_id) {
                        if entry.version.deleted_at_txn_id == 0
                            || !checker.is_visible(entry.version.deleted_at_txn_id, txn_id)
                        {
                            all_visible.push(get_row_from_entry(entry));
                        }
                        break;
                    }
                    current = entry.prev.as_ref().map(|b| b.as_ref());
                }
            }
            // Reverse and apply offset/limit
            result = all_visible
                .into_iter()
                .rev()
                .skip(offset)
                .take(limit)
                .collect();
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
    /// Vector of (row_id, row) pairs in row_id order
    pub fn collect_rows_keyset(
        &self,
        txn_id: i64,
        start_after_row_id: Option<i64>,
        start_from_row_id: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
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
                            Row::from_arc(Arc::clone(arc_row))
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
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(limit);

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
            let mut all_visible: Vec<(i64, Row)> = Vec::new();
            for (&row_id, chain) in versions.range((start_bound, std::ops::Bound::Unbounded::<i64>))
            {
                if let Some(row_data) = find_visible_row(chain) {
                    all_visible.push((row_id, row_data));
                }
            }
            result = all_visible.into_iter().rev().take(limit).collect();
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
    ) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
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
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(versions.len());

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
                    // Try arena path first (zero-copy filter check)
                    if let Some(idx) = e.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter on slice - no allocation!
                            if compiled_filter.matches_slice(arc_row) {
                                // Only clone if filter matched
                                result.push((row_id, Row::from_arc(Arc::clone(arc_row))));
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
    ) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
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
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(limit);
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
                    // Try arena path first (zero-copy filter check)
                    if let Some(idx) = e.arena_idx {
                        if let Some(arc_row) = arena_data.get(idx) {
                            // Filter on slice - no allocation!
                            if compiled_filter.matches_slice(arc_row) {
                                if skipped < offset {
                                    skipped += 1;
                                } else {
                                    result.push((row_id, Row::from_arc(Arc::clone(arc_row))));
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
    ) -> Vec<(i64, Row)> {
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
            let uncommitted_empty = self.uncommitted_writes.is_empty();

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

    /// Get visible row indices (BTreeMap iteration is already sorted).
    ///
    /// Note: With BTreeMap, this is functionally identical to get_visible_row_indices
    /// since BTreeMap iteration is inherently ordered by row_id. Kept for API compatibility.
    #[inline]
    pub fn get_visible_row_indices_unordered(&self, txn_id: i64) -> Vec<RowIndex> {
        // Delegate to the sorted version - BTreeMap is always sorted
        self.get_visible_row_indices(txn_id)
    }

    /// Materialize selected row indices into actual Row data
    ///
    /// This is the second step of deferred materialization. After filtering/limiting
    /// `RowIndex` values, call this to get the actual row data.
    ///
    /// # Performance
    /// - Only clones the rows you actually need
    /// - Falls back to version chain for non-arena rows
    pub fn materialize_rows(&self, indices: &[RowIndex]) -> Vec<(i64, Row)> {
        if indices.is_empty() {
            return Vec::new();
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();
        let arena_len = arena_guard.len();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        let mut result: Vec<(i64, Row)> = Vec::with_capacity(indices.len());

        for idx in indices {
            if let Some(arena_idx) = idx.arena_idx {
                // Fast path: get from arena
                if arena_idx < arena_len {
                    if let Some(arc_row) = arena_data.get(arena_idx) {
                        result.push((idx.row_id, Row::from_arc(Arc::clone(arc_row))));
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
                return Some((idx.row_id, Row::from_arc(Arc::clone(arc_row))));
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
                return arc_row.get(col_idx).cloned();
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
                            return arc_row.get(col_idx).cloned();
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
    ) -> Vec<(i64, Row)> {
        if limit == 0 {
            return Vec::new();
        }

        // Step 1: Get all visible row indices (no cloning!)
        let indices = self.get_visible_row_indices_unordered(txn_id);

        if indices.is_empty() {
            return Vec::new();
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
    /// Only loads the specified column values, not entire rows.
    /// Returns (sum, count_non_null) for proper NULL handling.
    pub fn sum_column(&self, txn_id: i64, col_idx: usize) -> (f64, usize) {
        let indices = self.get_visible_row_indices_unordered(txn_id);
        if indices.is_empty() {
            return (0.0, 0);
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        let mut sum = 0.0f64;
        let mut count = 0usize;

        for idx in &indices {
            // Try arena path first (fast)
            if let Some(arena_idx) = idx.arena_idx {
                if let Some(arc_row) = arena_data.get(arena_idx) {
                    if let Some(val) = arc_row.get(col_idx) {
                        match val {
                            Value::Integer(i) => {
                                sum += *i as f64;
                                count += 1;
                            }
                            Value::Float(f) => {
                                sum += *f;
                                count += 1;
                            }
                            _ => {} // NULL or non-numeric
                        }
                        continue;
                    }
                }
            }
            // Slow path: version chain lookup (lock already held)
            if let Some(entry) = versions.get(&idx.row_id) {
                if let Some(val) = entry.version.data.get(col_idx) {
                    match val {
                        Value::Integer(i) => {
                            sum += *i as f64;
                            count += 1;
                        }
                        Value::Float(f) => {
                            sum += *f;
                            count += 1;
                        }
                        _ => {} // NULL or non-numeric
                    }
                }
            }
        }

        (sum, count)
    }

    /// Compute MIN(column) without materializing full rows
    pub fn min_column(&self, txn_id: i64, col_idx: usize) -> Option<Value> {
        let indices = self.get_visible_row_indices_unordered(txn_id);
        if indices.is_empty() {
            return None;
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        let mut min_val: Option<Value> = None;

        // Helper to update min value
        let update_min = |min_val: &mut Option<Value>, val: &Value| {
            if !val.is_null() {
                match min_val {
                    None => *min_val = Some(val.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Less) = val.compare(current) {
                            *min_val = Some(val.clone());
                        }
                    }
                }
            }
        };

        for idx in &indices {
            // Try arena path first (fast)
            if let Some(arena_idx) = idx.arena_idx {
                if let Some(arc_row) = arena_data.get(arena_idx) {
                    if let Some(val) = arc_row.get(col_idx) {
                        update_min(&mut min_val, val);
                        continue;
                    }
                }
            }
            // Slow path: version chain lookup (lock already held)
            if let Some(entry) = versions.get(&idx.row_id) {
                if let Some(val) = entry.version.data.get(col_idx) {
                    update_min(&mut min_val, val);
                }
            }
        }

        min_val
    }

    /// Compute MAX(column) without materializing full rows
    pub fn max_column(&self, txn_id: i64, col_idx: usize) -> Option<Value> {
        let indices = self.get_visible_row_indices_unordered(txn_id);
        if indices.is_empty() {
            return None;
        }

        let arena_guard = self.arena.read_guard();
        let arena_data = arena_guard.data();

        // Pre-acquire versions lock ONCE for all slow path lookups
        let versions = self.versions.read();

        let mut max_val: Option<Value> = None;

        // Helper to update max value
        let update_max = |max_val: &mut Option<Value>, val: &Value| {
            if !val.is_null() {
                match max_val {
                    None => *max_val = Some(val.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                            *max_val = Some(val.clone());
                        }
                    }
                }
            }
        };

        for idx in &indices {
            // Try arena path first (fast)
            if let Some(arena_idx) = idx.arena_idx {
                if let Some(arc_row) = arena_data.get(arena_idx) {
                    if let Some(val) = arc_row.get(col_idx) {
                        update_max(&mut max_val, val);
                        continue;
                    }
                }
            }
            // Slow path: version chain lookup (lock already held)
            if let Some(entry) = versions.get(&idx.row_id) {
                if let Some(val) = entry.version.data.get(col_idx) {
                    update_max(&mut max_val, val);
                }
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
                    AggregateOp::Count => AggregateResult::Count(0),
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
                AggregateOp::Count => AggregateAccumulator::Count(0),
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
            let uncommitted_empty = self.uncommitted_writes.is_empty();

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
        // Use DashMap's entry API for atomic check-and-insert
        use dashmap::mapref::entry::Entry;

        match self.uncommitted_writes.entry(row_id) {
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
        // Use remove_if for atomic check-and-remove
        self.uncommitted_writes
            .remove_if(&row_id, |_, &v| v == txn_id);
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
        let pred_set: std::collections::HashSet<&str> = predicate_columns.iter().copied().collect();

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
        if !is_deleted {
            let indexes = self.indexes.read();
            for index in indexes.values() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }
                let col_id = column_ids[0] as usize;
                if let Some(value) = row_data.get(col_id) {
                    // Ignore errors during recovery - index might already have this entry
                    let _ = index.add(std::slice::from_ref(value), row_id, row_id);
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
                        if let Some(value) = version.data.get(col_idx) {
                            let _ = index.add(std::slice::from_ref(value), row_id, row_id);
                        }
                    }
                }
            }

            self.add_index(meta.name.clone(), index);
        } else {
            // Multi-column index: use MultiColumnIndex (always, regardless of index_type)
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
                        let values: Vec<crate::core::Value> =
                            col_indices
                                .iter()
                                .map(|&idx| {
                                    version.data.get(idx).cloned().unwrap_or(
                                        crate::core::Value::Null(crate::core::DataType::Null),
                                    )
                                })
                                .collect();
                        let _ = index.add(&values, row_id, row_id);
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

            // Add to each index
            for (col_indices, index) in &index_infos {
                if col_indices.len() == 1 {
                    // Single-column index
                    if let Some(value) = version.data.get(col_indices[0]) {
                        let _ = index.add(std::slice::from_ref(value), row_id, row_id);
                    }
                } else {
                    // Multi-column index
                    let values: Vec<crate::core::Value> = col_indices
                        .iter()
                        .map(|&idx| {
                            version
                                .data
                                .get(idx)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    let _ = index.add(&values, row_id, row_id);
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
}

impl Clone for VersionChainEntry {
    fn clone(&self) -> Self {
        Self {
            version: self.version.clone(),
            prev: self.prev.clone(), // Arc clone is O(1)
            arena_idx: self.arena_idx,
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
    /// The Vec is ordered by create_time (oldest first, newest last)
    local_versions: Int64Map<Vec<RowVersion>>,
    /// Parent (shared) version store
    parent_store: Arc<VersionStore>,
    /// This transaction's ID
    txn_id: i64,
    /// Write set for conflict detection
    write_set: Int64Map<WriteSetEntry>,
}

impl TransactionVersionStore {
    /// Creates a new transaction-local version store
    pub fn new(parent_store: Arc<VersionStore>, txn_id: i64) -> Self {
        Self {
            local_versions: new_int64_map(),
            parent_store,
            txn_id,
            write_set: new_int64_map(),
        }
    }

    /// Returns the transaction ID
    pub fn txn_id(&self) -> i64 {
        self.txn_id
    }

    /// Put adds or updates a row in the transaction's local store
    pub fn put(&mut self, row_id: i64, data: Row, is_delete: bool) -> Result<(), Error> {
        // Check if we already have a local version
        if !self.local_versions.contains_key(&row_id) {
            // Check if this row exists in parent store and track in write-set
            // Single lookup instead of two separate calls
            if !self.write_set.contains_key(&row_id) {
                let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                let row_exists = read_version.is_some();

                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.write_set.insert(
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

        // Create the row version
        let mut rv = RowVersion::new(self.txn_id, row_id, data);

        if is_delete {
            rv.deleted_at_txn_id = self.txn_id;
        }

        // Append to version history for this row (for savepoint support)
        if let Some(versions) = self.local_versions.get_mut(&row_id) {
            versions.push(rv);
        } else {
            self.local_versions.insert(row_id, vec![rv]);
        }
        Ok(())
    }

    /// Batch put for UPDATE operations where we already have the row data
    ///
    /// This is used for rows that are already tracked in local_versions (updates within same txn)
    /// or when we don't have pre-fetched original versions.
    pub fn put_batch_for_update(&mut self, rows: Vec<(i64, Row)>) -> Result<(), Error> {
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

        // Check if already in local versions (already processed in this transaction)
        if let Some(versions) = self.local_versions.get_mut(&row_id) {
            versions.push(rv);
            return Ok(());
        }

        // Track in write-set using the pre-fetched original version
        if !self.write_set.contains_key(&row_id) {
            let read_version_seq = self
                .parent_store
                .visibility_checker
                .as_ref()
                .map(|c| c.get_current_sequence())
                .unwrap_or(0);

            self.write_set.insert(
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
        self.local_versions.insert(row_id, vec![rv]);
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
            if let Some(versions) = self.local_versions.get_mut(&row_id) {
                // Append new version to history
                versions.push(rv);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            if let std::collections::hash_map::Entry::Vacant(e) = self.write_set.entry(row_id) {
                // Get current sequence for conflict detection
                // Note: We get a fresh sequence instead of relying on create_time because
                // callers may use get_visible_version() which doesn't set create_time to the sequence
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);
                e.insert(WriteSetEntry {
                    read_version: Some(original_version),
                    read_version_seq,
                });

                // Claim the row for update
                self.parent_store.try_claim_row(row_id, self.txn_id)?;
            }

            // Insert new version history for this row
            self.local_versions.insert(row_id, vec![rv]);
        }
        Ok(())
    }

    /// Optimized batch delete for DELETE operations
    ///
    /// This marks multiple rows as deleted in a single operation, avoiding
    /// the overhead of individual put() calls with lock acquisitions per row.
    ///
    /// Parameters:
    /// - rows: Vec of (row_id, row_data) to mark as deleted
    pub fn put_batch_deleted(&mut self, rows: Vec<(i64, Row)>) -> Result<(), Error> {
        for (row_id, data) in rows {
            // Check if we already have a local version
            if !self.local_versions.contains_key(&row_id) {
                // Check if this row exists in parent store and track in write-set
                if !self.write_set.contains_key(&row_id) {
                    let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                    let row_exists = read_version.is_some();

                    let read_version_seq = self
                        .parent_store
                        .visibility_checker
                        .as_ref()
                        .map(|c| c.get_current_sequence())
                        .unwrap_or(0);

                    self.write_set.insert(
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
            if let Some(versions) = self.local_versions.get_mut(&row_id) {
                versions.push(rv);
            } else {
                self.local_versions.insert(row_id, vec![rv]);
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
            if let Some(versions) = self.local_versions.get_mut(&row_id) {
                versions.push(rv);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            if !self.write_set.contains_key(&row_id) {
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.write_set.insert(
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
            self.local_versions.insert(row_id, vec![rv]);
        }
        Ok(())
    }

    /// Check if we have local changes for a row
    pub fn has_locally_seen(&self, row_id: i64) -> bool {
        self.local_versions.contains_key(&row_id)
    }

    /// Returns true if this transaction has any uncommitted local changes
    pub fn has_local_changes(&self) -> bool {
        !self.local_versions.is_empty()
    }

    /// Iterate over local versions (returns most recent version per row)
    pub fn iter_local(&self) -> impl Iterator<Item = (i64, &RowVersion)> {
        self.local_versions
            .iter()
            .filter_map(|(k, versions)| versions.last().map(|v| (*k, v)))
    }

    /// Iterate over local versions with their original (old) versions for index updates
    /// Returns (row_id, new_version, old_row_option)
    pub fn iter_local_with_old(&self) -> impl Iterator<Item = (i64, &RowVersion, Option<&Row>)> {
        self.local_versions
            .iter()
            .filter_map(move |(row_id, versions)| {
                versions.last().map(|version| {
                    let old_row = self
                        .write_set
                        .get(row_id)
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
            .get(&row_id)
            .and_then(|versions| versions.last())
    }

    /// Get a row, checking local versions first then parent store
    pub fn get(&self, row_id: i64) -> Option<Row> {
        // Check local versions first (get most recent)
        if let Some(versions) = self.local_versions.get(&row_id) {
            if let Some(local_version) = versions.last() {
                if local_version.is_deleted() {
                    return None;
                }
                return Some(local_version.data.clone());
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
        // Fast path: if we have no write set entries without read_version,
        // and all rows were claimed, there can be no conflicts
        let mut needs_insert_check = false;
        for (_, write_entry) in self.write_set.iter() {
            if write_entry.read_version.is_none() {
                needs_insert_check = true;
                break;
            }
        }

        // For UPDATE/DELETE operations where we claimed existing rows,
        // the claim mechanism already prevents conflicts. Skip the expensive
        // get_visible_version() calls for these rows.
        if !needs_insert_check {
            return Ok(());
        }

        // Only check rows where we're inserting (read_version was None)
        // These are the only rows where another transaction could have
        // inserted a conflicting row
        for (row_id, write_entry) in self.write_set.iter() {
            if write_entry.read_version.is_none() {
                // Check if row was inserted by another transaction
                let current_version = self.parent_store.get_visible_version(*row_id, self.txn_id);
                if current_version.is_some() {
                    return Err(Error::internal(format!(
                        "write conflict: row {} was inserted by another transaction",
                        row_id
                    )));
                }
            }
            // Skip rows with read_version - they were claimed and can't conflict
        }
        Ok(())
    }

    /// Prepare commit - returns list of versions to commit (most recent per row)
    pub fn prepare_commit(&self) -> Vec<(i64, RowVersion)> {
        let mut versions = Vec::new();
        for (row_id, version_history) in self.local_versions.iter() {
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

        // Batch apply all local versions to parent store (single lock acquisition)
        // Only commit the most recent version per row
        // Use drain() to take ownership instead of cloning
        let batch: Vec<(i64, RowVersion)> = self
            .local_versions
            .drain()
            .filter_map(|(row_id, mut versions)| versions.pop().map(|v| (row_id, v)))
            .collect();

        self.parent_store.add_versions_batch(batch);

        // Release all claims
        self.release_all_claims();

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
        let mut rows_to_remove_completely: Vec<i64> = Vec::new();

        // For each row, remove versions with create_time > timestamp
        for (row_id, versions) in self.local_versions.iter_mut() {
            // Keep only versions at or before the timestamp
            versions.retain(|v| v.create_time <= timestamp);

            // If all versions are removed, mark for complete removal
            if versions.is_empty() {
                rows_to_remove_completely.push(*row_id);
            }
        }

        // Remove rows with no remaining versions and release their claims
        for row_id in &rows_to_remove_completely {
            self.local_versions.remove(row_id);
            self.parent_store.release_row_claim(*row_id, self.txn_id);
            self.write_set.remove(row_id);
        }
    }

    /// Release all row claims held by this transaction
    fn release_all_claims(&self) {
        for (row_id, _) in self.write_set.iter() {
            self.parent_store.release_row_claim(*row_id, self.txn_id);
        }
    }
}

impl fmt::Debug for TransactionVersionStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransactionVersionStore")
            .field("txn_id", &self.txn_id)
            .field("local_version_count", &self.local_versions.len())
            .field("write_set_count", &self.write_set.len())
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
}
