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
use std::sync::{Arc, RwLock};

use crate::common::{new_concurrent_int64_map, new_int64_map, ConcurrentInt64Map, Int64Map};
use crate::core::{Error, Row, Schema};
use crate::storage::expression::CompiledFilter;
use crate::storage::mvcc::arena::RowArena;
use crate::storage::mvcc::get_fast_timestamp;
use crate::storage::mvcc::streaming_result::{StreamingResult, VisibleRowInfo};
use crate::storage::Index;
use radsort::sort_by_key;
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
    /// Row versions indexed by row ID (concurrent hash map for fast lookups)
    versions: ConcurrentInt64Map<VersionChainEntry>,
    /// The name of the table this store belongs to
    table_name: String,
    /// Table schema
    schema: RwLock<Schema>,
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
        let cols = schema.column_count();
        Self {
            versions: new_concurrent_int64_map(),
            table_name,
            schema: RwLock::new(schema),
            indexes: RwLock::new(FxHashMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: new_concurrent_int64_map(),
            visibility_checker: None,
            arena: RowArena::new(cols),
            row_arena_index: RwLock::new(new_int64_map()),
            zone_maps: RwLock::new(None),
        }
    }

    /// Creates a new version store with a visibility checker
    pub fn with_visibility_checker(
        table_name: String,
        schema: Schema,
        checker: Arc<dyn VisibilityChecker>,
    ) -> Self {
        let cols = schema.column_count();
        Self {
            versions: new_concurrent_int64_map(),
            table_name,
            schema: RwLock::new(schema),
            indexes: RwLock::new(FxHashMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: new_concurrent_int64_map(),
            visibility_checker: Some(checker),
            arena: RowArena::new(cols),
            row_arena_index: RwLock::new(new_int64_map()),
            zone_maps: RwLock::new(None),
        }
    }

    /// Sets the visibility checker
    pub fn set_visibility_checker(&mut self, checker: Arc<dyn VisibilityChecker>) {
        self.visibility_checker = Some(checker);
    }

    /// Returns the table name
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Returns a reference to the schema
    pub fn schema(&self) -> Schema {
        self.schema.read().unwrap().clone()
    }

    /// Returns a mutable reference to the schema (for modifications)
    pub fn schema_mut(&self) -> std::sync::RwLockWriteGuard<'_, Schema> {
        self.schema.write().unwrap()
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

        // Check if existing version exists first
        let existing_data = self
            .versions
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

                // Store in arena and get index (only for non-deleted versions)
                let arena_idx = if new_version.deleted_at_txn_id == 0 {
                    let idx = self.arena.insert_row(
                        row_id,
                        new_version.txn_id,
                        new_version.create_time,
                        &new_version.data,
                    );
                    // Update row -> arena index mapping
                    self.row_arena_index.write().unwrap().insert(row_id, idx);
                    Some(idx)
                } else {
                    // Mark the previous arena entry as deleted
                    if let Some(&prev_idx) = self.row_arena_index.read().unwrap().get(&row_id) {
                        self.arena.mark_deleted(prev_idx, new_version.txn_id);
                    }
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
                let arena_idx = if version.deleted_at_txn_id == 0 {
                    let idx = self.arena.insert_row(
                        row_id,
                        version.txn_id,
                        version.create_time,
                        &version.data,
                    );
                    // Update row -> arena index mapping
                    self.row_arena_index.write().unwrap().insert(row_id, idx);
                    Some(idx)
                } else {
                    None
                };

                VersionChainEntry {
                    version,
                    prev: None,
                    arena_idx,
                }
            };

        self.versions.insert(row_id, entry);
    }

    /// Adds multiple versions in batch - optimized for commit
    ///
    /// This is faster than calling add_version() in a loop because:
    /// 1. Uses DashMap's sharded locking for concurrent access
    /// 2. Skips arena updates (arena is rebuilt lazily for scans)
    #[inline]
    pub fn add_versions_batch(&self, batch: Vec<(i64, RowVersion)>) {
        if self.closed.load(Ordering::Acquire) || batch.is_empty() {
            return;
        }

        for (row_id, version) in batch {
            // Get existing data first (if any)
            let existing_data = self
                .versions
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

                // Skip arena updates for batch operations (arena rebuilt lazily)
                // Use Arc to share the previous chain - O(1) instead of deep clone
                let prev_chain = Arc::new(VersionChainEntry {
                    version: existing_version,
                    prev: existing_prev,
                    arena_idx: existing_arena_idx,
                });

                VersionChainEntry {
                    version: new_version,
                    prev: Some(prev_chain),
                    arena_idx: None, // Skip arena for batch
                }
            } else {
                // First version for this row
                VersionChainEntry {
                    version,
                    prev: None,
                    arena_idx: None, // Skip arena for batch
                }
            };

            self.versions.insert(row_id, entry);
        }
    }

    /// Quick check if a row might exist
    pub fn quick_check_row_existence(&self, row_id: i64) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }

        self.versions.contains_key(&row_id)
    }

    /// Gets the latest visible version of a row
    pub fn get_visible_version(&self, row_id: i64, txn_id: i64) -> Option<RowVersion> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let checker = self.visibility_checker.as_ref()?;

        // First pass: find which version is visible (only need txn_ids, no cloning)
        // Then clone only that one version
        let entry_ref = self.versions.get(&row_id)?;
        let mut current: Option<&VersionChainEntry> = Some(&*entry_ref);
        let mut result = None;

        while let Some(e) = current {
            let version_txn_id = e.version.txn_id;
            let deleted_at_txn_id = e.version.deleted_at_txn_id;

            // Check visibility using only txn_ids (no clone needed yet)
            if checker.is_visible(version_txn_id, txn_id) {
                // Check if deleted and deletion is visible
                if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                    result = None;
                } else {
                    // Only clone the ONE version we actually need
                    result = Some(e.version.clone());
                }
                break;
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
        }

        result
    }

    /// Gets multiple visible versions in a single batch operation
    ///
    /// This is much faster than calling get_visible_version for each row_id
    /// because it uses DashMap's sharded locking for concurrent access.
    pub fn get_visible_versions_batch(&self, row_ids: &[i64], txn_id: i64) -> Vec<(i64, Row)> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut results = Vec::with_capacity(row_ids.len());

        for &row_id in row_ids {
            if let Some(entry_ref) = self.versions.get(&row_id) {
                let mut current: Option<&VersionChainEntry> = Some(&*entry_ref);

                while let Some(e) = current {
                    let version_txn_id = e.version.txn_id;
                    let deleted_at_txn_id = e.version.deleted_at_txn_id;

                    if checker.is_visible(version_txn_id, txn_id) {
                        if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id)
                        {
                            results.push((row_id, e.version.data.clone()));
                        }
                        break;
                    }
                    current = e.prev.as_ref().map(|b| b.as_ref());
                }
            }
        }

        results
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

        let current_seq = checker.get_current_sequence();
        let mut results = Vec::with_capacity(row_ids.len());

        for &row_id in row_ids {
            if let Some(entry_ref) = self.versions.get(&row_id) {
                let mut current: Option<&VersionChainEntry> = Some(&*entry_ref);

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
                            results.push((row_id, e.version.data.clone(), version_copy));
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

        let entry_ref = self.versions.get(&row_id)?;

        // Traverse version chain from newest to oldest
        let mut current: Option<&VersionChainEntry> = Some(&*entry_ref);
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

        let entry_ref = self.versions.get(&row_id)?;

        // Traverse version chain from newest to oldest
        let mut current: Option<&VersionChainEntry> = Some(&*entry_ref);
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

    /// Returns all row IDs in the version store
    pub fn get_all_row_ids(&self) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        self.versions.iter().map(|e| *e.key()).collect()
    }

    /// Returns all row IDs that are visible to the given transaction
    pub fn get_all_visible_row_ids(&self, txn_id: i64) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        // First, collect all row IDs
        let all_row_ids: Vec<i64> = self.versions.iter().map(|e| *e.key()).collect();

        // Now check visibility for each row ID without holding any version store locks
        let mut visible_row_ids = Vec::with_capacity(all_row_ids.len());
        for row_id in all_row_ids {
            if self.get_visible_version(row_id, txn_id).is_some() {
                visible_row_ids.push(row_id);
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

        // Single pass through all versions - no per-row lock acquisition
        for entry in self.versions.iter() {
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                // Check visibility
                if checker.is_visible(version_txn_id, txn_id) {
                    // Check if deleted
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        count += 1; // Visible and not deleted
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

        // Collect all versions in one pass using DashMap's iteration
        let mut results = Vec::with_capacity(self.versions.len());

        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            // Find the first visible version in the chain
            let mut current: Option<&VersionChainEntry> = Some(chain);
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                // Check visibility
                if checker.is_visible(version_txn_id, txn_id) {
                    // Check if deleted
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }
                    results.push((row_id, e.version.data.clone()));
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // DashMap iteration order is not guaranteed, sort by row_id
        // Use radix sort - O(n) for integer keys vs O(n log n) for comparison sort
        sort_by_key(&mut results, |(row_id, _)| *row_id);
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

        // Pre-acquire arena locks ONCE for the entire operation
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        // Single-pass: read directly from arena during visibility check
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(self.versions.len());

        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Read directly from arena (no intermediate collection)
                    if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            // SAFETY: bounds checked above
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            result.push((row_id, Row::from_values(slice.to_vec())));
                        }
                    } else {
                        // No arena entry (batch-committed row) - clone from version
                        result.push((row_id, e.version.data.clone()));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // Drop arena locks before sorting
        drop(arena_rows);
        drop(arena_data);

        // Sort by row_id to maintain insertion order
        // Critical for FIRST/LAST aggregate functions
        sort_by_key(&mut result, |(row_id, _)| *row_id);

        result
    }

    /// Get visible rows with limit and offset applied at the storage layer.
    ///
    /// # Current Limitations (NOT True Early Termination)
    /// Due to DashMap's concurrent design, iteration order is non-deterministic.
    /// To maintain consistent results (e.g., LIMIT 1 returns first inserted row),
    /// this function:
    /// 1. Iterates ALL visible rows (O(n) scan)
    /// 2. Sorts by row_id (O(n log n))
    /// 3. Applies offset/limit (O(limit))
    ///
    /// This is NOT early termination - it still scans the full table. The benefit
    /// is reduced data transfer to the executor (only `limit` rows returned vs all).
    /// True early termination would require ordered storage (B-tree on row_id).
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

        // Pre-acquire arena locks ONCE for the entire operation
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        // Collect all visible rows (we need to sort to maintain insertion order)
        // For tables smaller than 2x the needed rows, this is efficient
        let needed = limit + offset;
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(self.versions.len().min(needed * 2));

        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Read row data
                    if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            result.push((row_id, Row::from_values(slice.to_vec())));
                        }
                    } else {
                        result.push((row_id, e.version.data.clone()));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // Drop arena locks before sorting
        drop(arena_rows);
        drop(arena_data);

        // Sort by row_id to maintain insertion order (critical for deterministic LIMIT)
        sort_by_key(&mut result, |(row_id, _)| *row_id);

        // Apply offset and limit
        result.into_iter().skip(offset).take(limit).collect()
    }

    /// Get visible rows with LIMIT but without sorting (for LIMIT without ORDER BY).
    ///
    /// This is an optimized version that enables true early termination by skipping
    /// the sort step. Since SQL doesn't guarantee order for LIMIT without ORDER BY,
    /// returning rows in arbitrary order is correct and much faster.
    ///
    /// # Performance
    /// - For LIMIT 100 on 10K rows: ~50x speedup vs sorted version
    /// - Stops iterating once limit+offset rows are collected
    pub fn get_visible_rows_with_limit_unordered(
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

        // Pre-acquire arena locks ONCE for the entire operation
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        let needed = limit + offset;
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(needed.min(1024));
        let mut count = 0usize;

        // Early termination: stop once we have enough rows
        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Handle offset: skip first `offset` rows
                    if count < offset {
                        count += 1;
                        break;
                    }

                    // Read row data
                    if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            result.push((row_id, Row::from_values(slice.to_vec())));
                        }
                    } else {
                        result.push((row_id, e.version.data.clone()));
                    }

                    // Early termination: we have enough rows
                    if result.len() >= limit {
                        return result;
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
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
        let schema = self.schema.read().unwrap();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema); // Release lock early

        // Pre-acquire arena locks ONCE
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        // Single-pass: read, filter, and collect in one loop
        let mut result: Vec<(i64, Row)> = Vec::with_capacity(self.versions.len());

        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            let row = Row::from_values(slice.to_vec());
                            // Filter using compiled filter (eliminates virtual dispatch)
                            if compiled_filter.matches(&row) {
                                result.push((row_id, row));
                            }
                        }
                    } else {
                        // Non-arena row - filter using compiled filter
                        if compiled_filter.matches(&e.version.data) {
                            result.push((row_id, e.version.data.clone()));
                        }
                    }
                    break;
                }
                current = e.prev.as_deref();
            }
        }

        // Drop locks before sorting
        drop(arena_rows);
        drop(arena_data);

        sort_by_key(&mut result, |(row_id, _)| *row_id);
        result
    }

    /// Get visible rows with filter, limit and offset applied at the storage layer.
    ///
    /// # Current Limitations (NOT True Early Termination)
    /// Like `get_visible_rows_with_limit`, this scans all rows due to DashMap's
    /// non-deterministic iteration order. The filter is applied during iteration,
    /// but all matching rows are collected before sorting and applying limit.
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
        let schema = self.schema.read().unwrap();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema);

        // Pre-acquire arena locks ONCE
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        // Collect all matching rows (need to sort for deterministic order)
        let mut result: Vec<(i64, Row)> = Vec::new();

        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Read row data
                    let row = if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            Row::from_values(slice.to_vec())
                        } else {
                            e.version.data.clone()
                        }
                    } else {
                        e.version.data.clone()
                    };

                    // Apply filter - only collect matching rows
                    if compiled_filter.matches(&row) {
                        result.push((row_id, row));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // Drop arena locks before sorting
        drop(arena_rows);
        drop(arena_data);

        // Sort by row_id to maintain insertion order
        sort_by_key(&mut result, |(row_id, _)| *row_id);

        // Apply offset and limit
        result.into_iter().skip(offset).take(limit).collect()
    }

    /// Get visible rows with filter and LIMIT but without sorting (for LIMIT without ORDER BY).
    ///
    /// This is an optimized version that enables true early termination by skipping
    /// the sort step. Since SQL doesn't guarantee order for LIMIT without ORDER BY,
    /// returning rows in arbitrary order is correct and much faster.
    pub fn get_visible_rows_filtered_with_limit_unordered(
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
        let schema = self.schema.read().unwrap();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema);

        // Pre-acquire arena locks ONCE
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_rows_slice = arena_rows.as_slice();
        let arena_data_slice = arena_data.as_slice();
        let arena_len = arena_rows_slice.len();

        let mut result: Vec<(i64, Row)> = Vec::with_capacity((limit + offset).min(1024));
        let mut count = 0usize;

        // Early termination: stop once we have enough matching rows
        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain = entry.value();
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    // Read row data
                    let row = if let Some(idx) = e.arena_idx {
                        if idx < arena_len {
                            let meta = unsafe { arena_rows_slice.get_unchecked(idx) };
                            let slice =
                                unsafe { arena_data_slice.get_unchecked(meta.start..meta.end) };
                            Row::from_values(slice.to_vec())
                        } else {
                            e.version.data.clone()
                        }
                    } else {
                        e.version.data.clone()
                    };

                    // Apply filter - only count matching rows
                    if compiled_filter.matches(&row) {
                        // Handle offset: skip first `offset` matching rows
                        if count < offset {
                            count += 1;
                        } else {
                            result.push((row_id, row));
                            // Early termination: we have enough rows
                            if result.len() >= limit {
                                return result;
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

        // Pre-acquire arena locks
        let (arena_rows, arena_data) = self.arena.read_guards();
        let arena_len = arena_rows.len();

        // Get column names from schema
        let columns: Vec<String> = self
            .schema
            .read()
            .unwrap()
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

                for (idx, meta) in arena_rows.iter().enumerate() {
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

                return StreamingResult::new(arena_rows, arena_data, visible_indices, columns);
            }
        }

        // SLOW PATH: Full DashMap iteration for complex visibility scenarios
        let mut visible_indices: Vec<VisibleRowInfo> = Vec::with_capacity(self.versions.len());

        if let Some(checker) = checker {
            for entry in self.versions.iter() {
                let row_id = *entry.key();
                let chain = entry.value();
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

        // Sort by row_id for consistent ordering
        visible_indices.sort_unstable_by_key(|info| info.row_id);

        StreamingResult::new(arena_rows, arena_data, visible_indices, columns)
    }

    /// Returns the count of rows
    pub fn row_count(&self) -> usize {
        self.versions.len()
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
        let indexes = self.indexes.read().unwrap();
        indexes.contains_key(index_name)
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<String> {
        let indexes = self.indexes.read().unwrap();
        indexes.keys().cloned().collect()
    }

    /// Iterate over unique indexes only, calling the provided function for each
    /// OPTIMIZATION: Avoids collecting index names and allows early exit on error
    pub fn for_each_unique_index<F>(&self, mut f: F) -> crate::core::Result<()>
    where
        F: FnMut(&str, &Arc<dyn Index>) -> crate::core::Result<()>,
    {
        let indexes = self.indexes.read().unwrap();
        for (name, index) in indexes.iter() {
            if index.is_unique() {
                f(name, index)?;
            }
        }
        Ok(())
    }

    /// Add an index
    pub fn add_index(&self, name: String, index: Arc<dyn Index>) {
        let mut indexes = self.indexes.write().unwrap();
        indexes.insert(name, index);
    }

    /// Remove an index
    pub fn remove_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        let mut indexes = self.indexes.write().unwrap();
        indexes.remove(name)
    }

    /// Get an index by name
    pub fn get_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        let indexes = self.indexes.read().unwrap();
        indexes.get(name).cloned()
    }

    /// Get an index by column name (single-column indexes only)
    pub fn get_index_by_column(&self, column_name: &str) -> Option<Arc<dyn Index>> {
        let indexes = self.indexes.read().unwrap();
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

        let indexes = self.indexes.read().unwrap();
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
        let indexes = self.indexes.read().unwrap();
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
        let mut guard = self.zone_maps.write().unwrap();
        *guard = Some(Arc::new(zone_maps));
    }

    /// Gets the zone maps for this table
    ///
    /// Returns None if zone maps have not been built (ANALYZE not run)
    /// Uses Arc to avoid expensive cloning on high QPS workloads
    pub fn get_zone_maps(&self) -> Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        let guard = self.zone_maps.read().unwrap();
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
        let guard = self.zone_maps.read().unwrap();
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
        let guard = self.zone_maps.read().unwrap();
        guard
            .as_ref()
            .and_then(|zm| zm.get_prune_stats(column, operator, value))
    }

    /// Marks zone maps as stale (needing rebuild after data changes)
    pub fn mark_zone_maps_stale(&self) {
        let guard = self.zone_maps.read().unwrap();
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
        if let Some(existing_entry) = self.versions.get(&row_id) {
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

        // Add the version to the store
        self.add_version(row_id, version);

        // Update auto_increment counter if this row_id is higher
        // This ensures the counter is restored to at least the max seen row_id
        if row_id > 0 {
            self.set_auto_increment_counter(row_id);
        }

        // Update indexes with the new row data (if not deleted)
        if !is_deleted {
            let indexes = self.indexes.read().unwrap();
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
            let indexes = self.indexes.read().unwrap();
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
                for entry in self.versions.iter() {
                    let row_id = *entry.key();
                    let version_chain = entry.value();
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
                for entry in self.versions.iter() {
                    let row_id = *entry.key();
                    let version_chain = entry.value();
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
        let indexes = self.indexes.read().unwrap();
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

        // Single pass over all rows
        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let version_chain = entry.value();
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
        self.versions.iter().for_each(|entry| {
            let row_id = *entry.key();
            let version = &entry.value().version;

            // Only process rows that are actually deleted and old enough
            if version.is_deleted() && version.create_time < cutoff_time {
                // Check if safe to remove (no active transaction can see it)
                if self.can_safely_remove(version) {
                    rows_to_delete.push(row_id);
                }
            }
        });

        // Second pass: remove the identified rows
        for row_id in &rows_to_delete {
            // Remove from indexes first
            if let Some(entry) = self.versions.get(row_id) {
                let version = &entry.version;
                let indexes = self.indexes.read().unwrap();
                for index in indexes.values() {
                    let column_ids = index.column_ids();
                    if !column_ids.is_empty() {
                        if column_ids.len() == 1 {
                            // Single-column index
                            let col_id = column_ids[0] as usize;
                            if let Some(value) = version.data.get(col_id) {
                                let _ = index.remove(std::slice::from_ref(value), *row_id, *row_id);
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

            // Now remove from version store
            self.versions.remove(row_id);
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

        // Iterate through all rows
        self.versions.iter_mut().for_each(|mut entry| {
            let chain_entry = entry.value_mut();

            // Find the oldest version we need to keep
            let mut current = chain_entry.prev.as_ref();
            let mut versions_to_check = Vec::new();

            // Collect all versions in the chain
            while let Some(prev_entry) = current {
                versions_to_check.push(prev_entry.clone());
                current = prev_entry.prev.as_ref();
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

            // Prune versions beyond keep_count
            if keep_count < versions_to_check.len() {
                let to_remove = versions_to_check.len() - keep_count;
                cleaned += to_remove as i32;

                // Disconnect the chain at the cutoff point
                if keep_count == 0 {
                    // Remove all previous versions
                    chain_entry.prev = None;
                } else {
                    // Navigate to the keep_count-th version and cut there
                    let mut nav = &mut chain_entry.prev;
                    for _ in 0..(keep_count - 1) {
                        if let Some(ref mut entry) = nav {
                            nav = &mut Arc::make_mut(entry).prev;
                        }
                    }
                    if let Some(ref mut entry) = nav {
                        Arc::make_mut(entry).prev = None;
                    }
                }
            }
        });

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

        // Iterate all versions
        for entry in self.versions.iter() {
            let row_id = *entry.key();
            let chain_entry = entry.value();

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

        for entry in self.versions.iter() {
            let chain_entry = entry.value();
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
                // The read_version_seq was stored in create_time during get_visible_versions_for_update
                let read_version_seq = original_version.create_time;
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
    pub fn commit(&self) -> Result<(), Error> {
        // Detect conflicts first
        self.detect_conflicts()?;

        // Batch apply all local versions to parent store (single lock acquisition)
        // Only commit the most recent version per row
        let batch: Vec<(i64, RowVersion)> = self
            .local_versions
            .iter()
            .filter_map(|(row_id, versions)| versions.last().map(|v| (*row_id, v.clone())))
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
