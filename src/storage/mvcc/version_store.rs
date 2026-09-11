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
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::common::SmartString;

use crate::common::i64_map::{f64_from_key, key_from_f64};
use crate::common::{new_i64_map, new_i64_map_with_capacity, CompactArc, CowBTree, I64Map};
use crate::core::types::DataType;
use crate::core::{Error, Row, RowVec, Schema, Value};
use crate::storage::expression::CompiledFilter;
use crate::storage::mvcc::arena::{ArenaReservation, ArenaRetirement, ArenaSlot, RowArena};
use crate::storage::mvcc::get_fast_timestamp;
use crate::storage::mvcc::memory::{
    arc_allocation_bytes, name_bytes, smallvec_bytes, HotMetadataCharge, HotObjectCharge, NamedMap,
    RetainedBytes, TableMemory,
};
use crate::storage::mvcc::read_memory::{charge_export, charge_value_export, ExportBatch};
#[cfg(not(test))]
use crate::storage::mvcc::registry::TransactionRegistry;
use crate::storage::Index;
use ahash::AHashMap;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Type alias for version lists - uses SmallVec to avoid heap allocation
/// for the common case of a single version per row within a transaction.
type VersionList = SmallVec<[RowVersion; 1]>;

/// Group key using CompactArc<Value> to avoid cloning during aggregation.
/// Uses Arc::clone (O(1) atomic increment) instead of Value::clone (deep copy).
#[derive(Clone, Debug)]
pub enum GroupKey {
    Single(CompactArc<Value>),
    Multi(Vec<CompactArc<Value>>),
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

/// Hash map for GroupKey with randomized hashing (HashDoS resistant)
type GroupKeyMap<V> = AHashMap<GroupKey, V>;

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
    /// Timestamp when this version was created
    pub create_time: i64,
}

impl RowVersion {
    /// Creates a new row version
    pub fn new(txn_id: i64, data: Row) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: 0,
            data,
            create_time: get_fast_timestamp(),
        }
    }

    /// Creates a new row version with a pre-computed timestamp
    /// This avoids calling SystemTime::now() for each row in bulk operations
    #[inline]
    pub fn new_with_timestamp(txn_id: i64, data: Row, create_time: i64) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: 0,
            data,
            create_time,
        }
    }

    /// Creates a new deleted version
    pub fn new_deleted(txn_id: i64, data: Row) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: txn_id,
            data,
            create_time: get_fast_timestamp(),
        }
    }

    /// Creates a new deleted version with a pre-computed timestamp
    #[inline]
    pub fn new_deleted_with_timestamp(txn_id: i64, data: Row, create_time: i64) -> Self {
        Self {
            txn_id,
            deleted_at_txn_id: txn_id,
            data,
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
            .field("create_time", &self.create_time)
            .finish()
    }
}

impl fmt::Display for RowVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RowVersion{{TxnID: {}, DeletedAtTxnID: {}, CreateTime: {}}}",
            self.txn_id, self.deleted_at_txn_id, self.create_time
        )
    }
}

/// Version payload and history, with an optional compact arena address for the head.
struct VersionChainEntry {
    /// Current version
    version: RowVersion,
    /// Previous version in the chain (Arc allows cheap cloning)
    prev: Option<Arc<VersionChainEntry>>,
    arena_idx: Option<ArenaSlot>,
}

impl VersionChainEntry {
    fn payloads(&self) -> VersionPayloads {
        let mut payloads = VersionPayloads::default();
        payloads.add_row(&self.version.data);
        let mut current = self.prev.as_deref();
        while let Some(entry) = current {
            payloads.add_row(&entry.version.data);
            current = entry.prev.as_deref();
        }
        payloads
    }
}

#[derive(Clone, Copy, Default)]
struct VersionPayloads {
    shared_and_children: u128,
    versions: usize,
    owned_rows: usize,
    // COW clones may shrink vectors. This maximum bounds every remaining owner.
    owned_capacity: usize,
}

impl VersionPayloads {
    fn add_row(&mut self, row: &Row) {
        self.versions += 1;
        let mut bytes = row.heap_bytes();
        if let Some(capacity) = row.owned_capacity() {
            bytes -= (capacity * std::mem::size_of::<Value>()) as u128;
            self.owned_rows += 1;
            self.owned_capacity = self.owned_capacity.max(capacity);
        }
        self.shared_and_children += bytes;
    }

    fn remove(&mut self, removed: Self) {
        self.shared_and_children -= removed.shared_and_children;
        self.versions -= removed.versions;
        self.owned_rows -= removed.owned_rows;
        if self.owned_rows == 0 {
            self.owned_capacity = 0;
        }
    }

    fn bytes(&self) -> u128 {
        self.shared_and_children
            + self.owned_rows as u128
                * self.owned_capacity as u128
                * std::mem::size_of::<Value>() as u128
    }
}

#[derive(Default)]
struct VersionTree {
    entries: CowBTree<VersionChainEntry>,
    payloads: VersionPayloads,
}

impl VersionTree {
    fn tree_bytes(&self) -> u128 {
        let links = self.payloads.versions - self.entries.len();
        self.entries.node_bytes() as u128
            + links as u128
                * (2 * std::mem::size_of::<usize>() + std::mem::size_of::<VersionChainEntry>())
                    as u128
    }

    fn publish_memory(&self, account: &TableMemory) {
        account.version_payloads.store(
            self.payloads.bytes().min(usize::MAX as u128) as usize,
            Ordering::Release,
        );
        account.version_tree.store(
            self.tree_bytes().min(usize::MAX as u128) as usize,
            Ordering::Release,
        );
    }
}

impl std::ops::Deref for VersionTree {
    type Target = CowBTree<VersionChainEntry>;

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl std::ops::DerefMut for VersionTree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries
    }
}

/// Count the depth of a version chain by traversing prev pointers.
/// O(k) where k is the chain length (typically <= max_version_history).
#[inline]
fn count_chain_depth(entry: &VersionChainEntry) -> usize {
    let mut depth = 1;
    let mut current = &entry.prev;
    while let Some(prev) = current {
        depth += 1;
        current = &prev.prev;
    }
    depth
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
static VERSION_LIST_MAP_POOL: Mutex<Vec<I64Map<VersionList>>> = Mutex::new(Vec::new());

/// Global pool for WriteSetEntry maps (write_set in TransactionVersionStore)
static WRITE_SET_MAP_POOL: Mutex<Vec<I64Map<WriteSetEntry>>> = Mutex::new(Vec::new());

static TRANSACTION_MAP_BYTES: RetainedBytes = RetainedBytes::new();

pub(crate) fn transaction_map_bytes() -> usize {
    TRANSACTION_MAP_BYTES.get()
}

fn account_map_capacity<V>(before: usize, map: &I64Map<V>) {
    let after = map.allocation_bytes();
    if after > before {
        TRANSACTION_MAP_BYTES.add((after - before) as u128);
    } else if before > after {
        TRANSACTION_MAP_BYTES.remove((before - after) as u128);
    }
}

/// Get a VersionList map from pool or create a new one
#[inline]
fn get_version_list_map() -> I64Map<VersionList> {
    if let Some(map) = VERSION_LIST_MAP_POOL.lock().pop() {
        map
    } else {
        let map = new_i64_map_with_capacity(TX_VERSION_MAP_INITIAL_CAPACITY);
        account_map_capacity(0, &map);
        map
    }
}

/// Get a WriteSetEntry map from pool or create a new one
#[inline]
fn get_write_set_map() -> I64Map<WriteSetEntry> {
    if let Some(map) = WRITE_SET_MAP_POOL.lock().pop() {
        map
    } else {
        let map = new_i64_map_with_capacity(TX_VERSION_MAP_INITIAL_CAPACITY);
        account_map_capacity(0, &map);
        map
    }
}

/// Return a VersionList map to the pool for reuse
#[inline]
fn return_version_list_map(mut map: I64Map<VersionList>) {
    map.clear();
    let mut pool = VERSION_LIST_MAP_POOL.lock();
    if pool.len() < MAP_POOL_MAX_SIZE {
        let before = pool.capacity();
        pool.push(map);
        let added = (pool.capacity() - before) * std::mem::size_of::<I64Map<VersionList>>();
        if added != 0 {
            TRANSACTION_MAP_BYTES.add(added as u128);
        }
    } else {
        drop(pool);
        let removed = map.allocation_bytes();
        drop(map);
        TRANSACTION_MAP_BYTES.remove(removed as u128);
    }
}

/// Return a WriteSetEntry map to the pool for reuse
#[inline]
fn return_write_set_map(mut map: I64Map<WriteSetEntry>) {
    map.clear();
    let mut pool = WRITE_SET_MAP_POOL.lock();
    if pool.len() < MAP_POOL_MAX_SIZE {
        let before = pool.capacity();
        pool.push(map);
        let added = (pool.capacity() - before) * std::mem::size_of::<I64Map<WriteSetEntry>>();
        if added != 0 {
            TRANSACTION_MAP_BYTES.add(added as u128);
        }
    } else {
        drop(pool);
        let removed = map.allocation_bytes();
        drop(map);
        TRANSACTION_MAP_BYTES.remove(removed as u128);
    }
}

fn clear_map_pool<V>(pool: &Mutex<Vec<I64Map<V>>>) {
    let mut pool = pool.lock();
    let removed: usize = pool.iter().map(I64Map::allocation_bytes).sum();
    pool.clear();
    if removed != 0 {
        TRANSACTION_MAP_BYTES.remove(removed as u128);
    }
}

/// Clear the transaction version map pools.
/// Call this when dropping the database to release pooled memory.
pub fn clear_version_map_pools() {
    clear_map_pool(&VERSION_LIST_MAP_POOL);
    clear_map_pool(&WRITE_SET_MAP_POOL);
}

/// Capacity hint for transaction version maps - used by pool functions
const TX_VERSION_MAP_INITIAL_CAPACITY: usize = 16;

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

/// Internal accumulator for aggregations.
/// Sum/Avg use split i128 + f64 accumulators: i128 for integers (no overflow),
/// f64 for floats. Combined as `int as f64 + float` at finalization.
enum AggregateAccumulator {
    Count(usize),
    /// (int_sum, float_sum, count)
    Sum(i128, f64, usize),
    Min(Option<Value>),
    Max(Option<Value>),
    /// (int_sum, float_sum, count)
    Avg(i128, f64, usize),
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

    /// Check if a transaction uses snapshot isolation.
    ///
    /// Under snapshot isolation, the arena-only fast path is unsafe because
    /// HEAD versions committed after the viewer's snapshot are not visible,
    /// and the correct behavior requires walking version chains to find older
    /// visible versions.
    fn needs_snapshot_isolation(&self, _txn_id: i64) -> bool {
        false // Default: ReadCommitted (arena fast path is safe)
    }
}

/// Opaque snapshot of the version store at extraction time.
/// Used by `remove_sealed_rows` to detect concurrent commits.
pub struct ExtractionSnapshot {
    inner: VersionSnapshot,
}

struct VersionSnapshot {
    inner: crate::common::CowBTree<VersionChainEntry>,
    // Fields drop in order: release the tree before its charge.
    _charge: VersionSnapshotCharge,
}

impl std::ops::Deref for VersionSnapshot {
    type Target = crate::common::CowBTree<VersionChainEntry>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

struct VersionSnapshotCharge {
    account: Arc<TableMemory>,
    bytes: u128,
    tree_bytes: u128,
}

impl VersionSnapshotCharge {
    fn new(account: &Arc<TableMemory>, versions: &VersionTree) -> Self {
        let bytes = versions.payloads.bytes();
        let tree_bytes = versions.tree_bytes();
        {
            let mut pinned = account.pinned_versions.lock();
            pinned.payloads += bytes;
            pinned.tree += tree_bytes;
        }
        Self {
            account: Arc::clone(account),
            bytes,
            tree_bytes,
        }
    }
}

impl Drop for VersionSnapshotCharge {
    fn drop(&mut self) {
        let mut pinned = self.account.pinned_versions.lock();
        pinned.payloads -= self.bytes;
        pinned.tree -= self.tree_bytes;
    }
}

/// Token holding pre-removal snapshot data needed for deferred index cleanup.
/// Created by `remove_sealed_rows`, consumed by `remove_sealed_index_entries`.
#[derive(Default)]
pub struct SealedIndexCleanup {
    /// Row IDs that were removed from the version store.
    pub removed_ids: Vec<i64>,
}

/// Truncated storage is released when this result leaves the caller's fence.
pub struct TruncateResult {
    pub rows_affected: i32,
    _versions: Option<VersionSnapshot>,
    _arena: ArenaRetirement,
}

impl From<i32> for TruncateResult {
    fn from(rows_affected: i32) -> Self {
        Self {
            rows_affected,
            _versions: None,
            _arena: ArenaRetirement::default(),
        }
    }
}

/// Held for a table from the first index update of a commit until the
/// commit is visible or undone; see VersionStore::begin_publish
pub struct PublishGuard {
    store: Arc<VersionStore>,
}

impl Drop for PublishGuard {
    fn drop(&mut self) {
        self.store.publish_epoch.fetch_add(1, Ordering::SeqCst);
        self.store.publishing.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What a committing transaction holds while it publishes: a guard per
/// table it writes, and the tables' stores so their index updates can be
/// undone if the commit fails after they were applied
#[derive(Default)]
pub struct PublishHold {
    guards: SmallVec<[PublishGuard; 4]>,
    stores: SmallVec<[Arc<std::sync::RwLock<TransactionVersionStore>>; 4]>,
    metadata: HotMetadataCharge,
}

/// Keeps prepared slots and table heads owned until publication guards are gone.
#[derive(Default)]
pub struct PreparedCommit {
    pub(crate) tables: SmallVec<[PreparedTable; 4]>,
    _metadata: HotMetadataCharge,
}

pub(crate) struct PreparedTable {
    pub name: crate::common::SmartString,
    pub txn_store: Arc<std::sync::RwLock<TransactionVersionStore>>,
    pub version_store: Option<Arc<VersionStore>>,
}

impl PreparedCommit {
    pub(crate) fn new(tables: SmallVec<[PreparedTable; 4]>) -> Self {
        let bytes = smallvec_bytes(&tables) as u128
            + tables
                .iter()
                .map(|table| name_bytes(&table.name))
                .sum::<u128>();
        Self {
            tables,
            _metadata: HotMetadataCharge::new(bytes),
        }
    }

    pub(crate) fn reserve(mut self) -> Result<Self, Error> {
        for table in &mut self.tables {
            let mut store = table
                .txn_store
                .write()
                .map_err(|_| Error::internal("transaction version lock poisoned"))?;
            if !store.has_local_changes() {
                table.version_store = None;
            }
            if table.version_store.is_some() {
                store.reserve_commit_capacity()?;
            }
        }
        Ok(self)
    }
}

impl Drop for PreparedCommit {
    fn drop(&mut self) {
        for table in &self.tables {
            let reservation = table
                .txn_store
                .write()
                .unwrap_or_else(|error| error.into_inner())
                .arena_reservation
                .take();
            drop(reservation);
        }
    }
}

impl PublishHold {
    pub fn add(
        &mut self,
        version_store: &Arc<VersionStore>,
        txn_store: Arc<std::sync::RwLock<TransactionVersionStore>>,
    ) {
        self.guards.push(version_store.begin_publish());
        self.stores.push(txn_store);
        self.metadata
            .resize((smallvec_bytes(&self.guards) + smallvec_bytes(&self.stores)) as u128);
    }

    /// True when a table this commit wrote holds `max_rows` committed hot
    /// rows or `max_bytes` hot bytes or more (0 = no limit); asked once the
    /// commit is visible, so the seal it requests can extract the rows
    pub fn any_table_at(&self, max_rows: usize, max_bytes: usize) -> bool {
        self.stores.iter().any(|store| {
            store.read().is_ok_and(|s| {
                let parent = &s.parent_store;
                (max_rows > 0 && parent.committed_row_count() >= max_rows)
                    || (max_bytes > 0 && parent.hot_bytes() >= max_bytes)
            })
        })
    }

    /// Takes back the index updates of a commit that failed after applying
    /// them, so the indexes describe the rows that stayed visible
    pub fn undo_index_updates(&self) {
        for store in &self.stores {
            if let Ok(store) = store.read() {
                store.undo_index_updates();
            }
        }
    }
}

/// The index entries a commit added and removed for one index, kept until
/// the commit is visible or undone
struct IndexUndo {
    index: Arc<dyn Index>,
    added: Vec<(i64, Vec<Value>)>,
    removed: Vec<(i64, Vec<Value>)>,
}

impl IndexUndo {
    fn heap_bytes(&self) -> u128 {
        [&self.added, &self.removed]
            .into_iter()
            .map(|rows| {
                rows.capacity() as u128 * std::mem::size_of::<(i64, Vec<Value>)>() as u128
                    + rows
                        .iter()
                        .map(|(_, values)| {
                            values.capacity() as u128 * std::mem::size_of::<Value>() as u128
                                + values
                                    .iter()
                                    .map(|value| value.heap_bytes() as u128)
                                    .sum::<u128>()
                        })
                        .sum::<u128>()
            })
            .sum()
    }
}

#[derive(Default)]
struct IndexUndoLog {
    entries: Vec<IndexUndo>,
    // Entries and their vector storage drop before the charge.
    charge: IndexUndoCharge,
}

impl IndexUndoLog {
    fn push(&mut self, entry: IndexUndo, account: &Arc<TableMemory>) {
        let bytes = entry.heap_bytes();
        let before = self.entries.capacity();
        self.entries.push(entry);
        let bytes =
            bytes + ((self.entries.capacity() - before) * std::mem::size_of::<IndexUndo>()) as u128;
        if self.charge.account.is_none() {
            self.charge.account = Some(Arc::clone(account));
        }
        account.transaction_undo.add(bytes);
        self.charge.bytes += bytes;
    }
}

#[derive(Default)]
struct IndexUndoCharge {
    account: Option<Arc<TableMemory>>,
    bytes: u128,
}

impl Drop for IndexUndoCharge {
    fn drop(&mut self) {
        if let Some(account) = &self.account {
            account.transaction_undo.remove(self.bytes);
        }
    }
}

/// Refreshes retained schema metadata before releasing the write lock.
pub struct SchemaWriteGuard<'a> {
    inner: parking_lot::RwLockWriteGuard<'a, CompactArc<Schema>>,
}

impl std::ops::Deref for SchemaWriteGuard<'_> {
    type Target = Schema;

    fn deref(&self) -> &Schema {
        &self.inner
    }
}

impl std::ops::DerefMut for SchemaWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Schema {
        CompactArc::make_mut(&mut self.inner)
    }
}

impl Drop for SchemaWriteGuard<'_> {
    fn drop(&mut self) {
        if let Some(schema) = CompactArc::get_mut(&mut self.inner) {
            schema.refresh_memory_charge();
        }
    }
}

/// VersionStore tracks the latest committed version of each row for a table
///
/// Uses CowBTreeMap (RwLock<CowBTree>) for the version store because:
/// - O(1) snapshot cloning for lock-free reads (critical for MVCC)
/// - Ordered iteration is free (B+ tree is sorted by key)
/// - MVCC has single-writer semantics per transaction, so concurrent map sharding is overhead
/// - Point lookups are O(log n) which is fast enough for typical row counts
/// - Eliminates the ~350μs sort overhead during full scans
///
/// Arena-based storage provides 50x+ faster full table scans by:
/// - Storing all row data contiguously in memory
/// - Returning slices instead of clones during iteration
/// - Eliminating per-row allocation overhead
pub struct VersionStore {
    /// Row versions indexed by row ID (CowBTree for O(1) snapshot cloning)
    versions: RwLock<VersionTree>,
    /// Independent of table registration and retained through payload destruction.
    memory: Arc<TableMemory>,
    /// The name of the table this store belongs to (SmartString inlines up to 15 bytes)
    table_name: SmartString,
    /// Table schema (Arc for zero-cost cloning on read)
    schema: RwLock<CompactArc<Schema>>,
    /// Indexes on this table (FxHashMap for fast string key lookups)
    indexes: RwLock<NamedMap<Arc<dyn Index>>>,
    /// Whether this store has been closed
    closed: AtomicBool,
    /// Auto-increment counter for tables without explicit PK
    auto_increment_counter: AtomicI64,
    /// Track which transaction has uncommitted changes to each row
    uncommitted_writes: RwLock<I64Map<i64>>,
    /// Visibility checker (registry reference)
    /// Production: concrete type for zero-cost inlining in hot paths
    /// Test: dyn trait for TestVisibilityChecker flexibility
    #[cfg(not(test))]
    visibility_checker: Option<Arc<TransactionRegistry>>,
    #[cfg(test)]
    visibility_checker: Option<Arc<dyn VisibilityChecker>>,
    /// Arena-based storage for zero-copy full table scans
    arena: RowArena,
    /// Zone maps for segment pruning (set by ANALYZE)
    /// Uses Arc to avoid cloning on every read - critical for high QPS workloads
    zone_maps: RwLock<Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>>>,
    /// Maximum number of previous versions to keep per row (0 = unlimited)
    /// This limits memory growth during write-heavy operations.
    /// Default is 10 - enough for most concurrent transaction scenarios.
    max_version_history: usize,
    /// Count of committed non-deleted rows for O(1) COUNT(*) queries.
    /// Updated on commit: +1 for INSERT, -1 for DELETE.
    /// This is an optimization for the common case of autocommit queries.
    committed_row_count: AtomicUsize,
    /// Per-table mutex for ON CONFLICT INSERT serialization.
    /// Only acquired for upsert statements to prevent TOCTOU races.
    /// Plain INSERTs (no ON CONFLICT) proceed lock-free.
    upsert_mutex: Arc<parking_lot::Mutex<()>>,
    /// Commits between their index updates and their versions being visible
    publishing: AtomicUsize,
    /// Publishes completed
    publish_epoch: AtomicU64,
    claim_memory: ClaimMemory,
    _metadata: HotMetadataCharge,
}

struct ClaimMemory(Arc<TableMemory>);

impl ClaimMemory {
    fn new(memory: &Arc<TableMemory>, claims: &I64Map<i64>) -> Self {
        memory
            .row_claims
            .store(claims.allocation_bytes(), Ordering::Release);
        Self(Arc::clone(memory))
    }

    fn resize(&self, before: usize, claims: &I64Map<i64>) {
        let after = claims.allocation_bytes();
        if before != after {
            self.0.row_claims.store(after, Ordering::Release);
        }
    }
}

impl Drop for ClaimMemory {
    fn drop(&mut self) {
        self.0.row_claims.store(0, Ordering::Release);
    }
}

impl VersionStore {
    /// Creates a new version store
    pub fn new(table_name: impl Into<SmartString>, schema: Schema) -> Self {
        Self::with_capacity(table_name, schema, None, 0)
    }

    /// Creates a new version store with pre-allocated capacity
    ///
    /// Pre-allocating capacity avoids hash map resizing during bulk inserts.
    /// Use this when the expected row count is known (e.g., during recovery).
    #[cfg(not(test))]
    pub fn with_capacity(
        table_name: impl Into<SmartString>,
        mut schema: Schema,
        checker: Option<Arc<TransactionRegistry>>,
        expected_rows: usize,
    ) -> Self {
        schema.refresh_memory_charge();
        let table_name = table_name.into();
        let metadata = HotMetadataCharge::new(
            arc_allocation_bytes::<Self>() as u128
                + arc_allocation_bytes::<parking_lot::Mutex<()>>() as u128
                + name_bytes(&table_name),
        );
        let memory = Arc::new(TableMemory::default());
        let claims = new_i64_map();
        let claim_memory = ClaimMemory::new(&memory, &claims);
        Self {
            versions: RwLock::new(VersionTree::default()),
            table_name,
            schema: RwLock::new(CompactArc::new(schema)),
            indexes: RwLock::new(NamedMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: RwLock::new(claims),
            visibility_checker: checker,
            arena: RowArena::with_account(expected_rows, Arc::clone(&memory)),
            memory,
            zone_maps: RwLock::new(None),
            max_version_history: 10, // Default: keep up to 10 previous versions
            committed_row_count: AtomicUsize::new(0),
            upsert_mutex: Arc::new(parking_lot::Mutex::new(())),
            publishing: AtomicUsize::new(0),
            publish_epoch: AtomicU64::new(0),
            claim_memory,
            _metadata: metadata,
        }
    }

    /// Creates a new version store with pre-allocated capacity (test version)
    #[cfg(test)]
    pub fn with_capacity(
        table_name: impl Into<SmartString>,
        mut schema: Schema,
        checker: Option<Arc<dyn VisibilityChecker>>,
        expected_rows: usize,
    ) -> Self {
        schema.refresh_memory_charge();
        let table_name = table_name.into();
        let metadata = HotMetadataCharge::new(
            arc_allocation_bytes::<Self>() as u128
                + arc_allocation_bytes::<parking_lot::Mutex<()>>() as u128
                + name_bytes(&table_name),
        );
        let memory = Arc::new(TableMemory::default());
        let claims = new_i64_map();
        let claim_memory = ClaimMemory::new(&memory, &claims);
        Self {
            versions: RwLock::new(VersionTree::default()),
            table_name,
            schema: RwLock::new(CompactArc::new(schema)),
            indexes: RwLock::new(NamedMap::default()),
            closed: AtomicBool::new(false),
            auto_increment_counter: AtomicI64::new(0),
            uncommitted_writes: RwLock::new(claims),
            visibility_checker: checker,
            arena: RowArena::with_account(expected_rows, Arc::clone(&memory)),
            memory,
            zone_maps: RwLock::new(None),
            max_version_history: 10,
            committed_row_count: AtomicUsize::new(0),
            upsert_mutex: Arc::new(parking_lot::Mutex::new(())),
            publishing: AtomicUsize::new(0),
            publish_epoch: AtomicU64::new(0),
            claim_memory,
            _metadata: metadata,
        }
    }

    /// Sets the maximum version history limit per row
    ///
    /// This controls how many previous versions are kept for each row.
    /// Lower values reduce memory usage but limit time-travel query range.
    /// - 0 = unlimited (not recommended for write-heavy workloads)
    /// - 1 = only keep immediate previous (minimal memory, limited AS OF range)
    /// - 10 = default, good balance for most workloads
    ///
    /// Note: Values above 254 are capped to 254 for practical reasons.
    pub fn set_max_version_history(&mut self, limit: usize) {
        // Cap at 254 for practical memory management
        self.max_version_history = limit.min(254);
    }

    /// Gets the current max version history limit
    pub fn max_version_history(&self) -> usize {
        self.max_version_history
    }

    /// Creates a new version store with a visibility checker (production)
    #[cfg(not(test))]
    pub fn with_visibility_checker(
        table_name: impl Into<SmartString>,
        schema: Schema,
        checker: Arc<TransactionRegistry>,
    ) -> Self {
        Self::with_capacity(table_name, schema, Some(checker), 0)
    }

    /// Creates a new version store with a visibility checker (test)
    #[cfg(test)]
    pub fn with_visibility_checker(
        table_name: impl Into<SmartString>,
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

    #[inline]
    fn capture_versions(&self) -> VersionSnapshot {
        let versions = self.versions.read();
        VersionSnapshot {
            inner: versions.entries.clone(),
            _charge: VersionSnapshotCharge::new(&self.memory, &versions),
        }
    }

    #[inline]
    fn snapshot_versions(&self) -> VersionSnapshot {
        let versions = self.capture_versions();
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::version_root_captured();
        versions
    }

    /// Returns the schema (cheap CompactArc clone)
    pub fn schema(&self) -> CompactArc<Schema> {
        self.schema.read().clone()
    }

    /// Returns copy-on-write schema access with retained-memory accounting.
    pub fn schema_mut(&self) -> SchemaWriteGuard<'_> {
        SchemaWriteGuard {
            inner: self.schema.write(),
        }
    }

    pub(crate) fn replace_schema(&self, schema: CompactArc<Schema>) {
        *self.schema.write() = schema;
    }

    /// Returns the current auto-increment counter value
    pub fn get_auto_increment_counter(&self) -> i64 {
        self.auto_increment_counter.load(Ordering::Acquire)
    }

    /// Returns the next available auto-increment ID, or None when the
    /// counter is exhausted. A plain fetch_add would overflow past
    /// i64::MAX: a panic in debug, and in release a wrap to i64::MIN that
    /// poisons every later id.
    pub fn try_next_auto_increment_id(&self) -> Option<i64> {
        let mut current = self.auto_increment_counter.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(1)?;
            match self.auto_increment_counter.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(next),
                Err(observed) => current = observed,
            }
        }
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
    pub fn add_version(&self, row_id: i64, version: RowVersion) -> Result<(), Error> {
        let reuse = {
            let versions = self.versions.read();
            if versions
                .get(row_id)
                .is_some_and(|head| head.arena_idx.is_some())
            {
                Some(self.arena.reserve_existing())
            } else {
                None
            }
        };
        let mut reservation = match reuse {
            Some(reservation) => reservation,
            None => self.arena.reserve(usize::from(!version.is_deleted()))?,
        };
        let mut versions = self.versions.write();
        let VersionTree { entries, payloads } = &mut *versions;
        let delta = self.install_version(entries, &mut reservation, payloads, row_id, version);
        versions.publish_memory(&self.memory);
        if delta != 0 {
            self.committed_row_count
                .fetch_add(delta as usize, Ordering::Relaxed);
        }
        Ok(())
    }

    fn install_version(
        &self,
        versions: &mut crate::common::CowBTree<VersionChainEntry>,
        reservation: &mut ArenaReservation,
        payloads: &mut VersionPayloads,
        row_id: i64,
        version: RowVersion,
    ) -> isize {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let is_new_version_deleted = version.deleted_at_txn_id != 0;
        let delta;

        // Use entry API to avoid double traversal
        match versions.entry(row_id) {
            crate::common::cow_btree::Entry::Occupied(mut occupied) => {
                // Extract existing data from the entry
                let existing = occupied.get();
                let existing_arena_idx = existing.arena_idx;
                let was_deleted = existing.version.deleted_at_txn_id != 0;

                delta = isize::from(was_deleted) - isize::from(is_new_version_deleted);

                // O(k) chain management - depth computed by traversal
                // When limit exceeded: drop old chain AND reuse arena slot
                let existing_depth = count_chain_depth(existing);
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

                    let (idx, bytes) = self.arena.install(
                        reservation,
                        existing_arena_idx,
                        row_id,
                        new_version.txn_id,
                        CompactArc::clone(&arc_data),
                    );

                    // Reuse the Arc for the version's data - enables O(1) clone on read
                    new_version.data = Row::from_arc(arc_data);

                    payloads.shared_and_children += bytes;
                    payloads.versions += 1;
                    Some(idx)
                } else {
                    // Deleted version - mark arena as deleted for visibility
                    if let Some(old_arena_idx) = existing_arena_idx {
                        self.arena.mark_deleted(old_arena_idx, new_version.txn_id);
                    }
                    payloads.add_row(&new_version.data);
                    existing_arena_idx
                };

                if can_reuse_arena {
                    payloads.remove(existing.payloads());
                }

                // Build version chain entry
                // When limit exceeded: drop entire history (no prev_chain allocation)
                // When under limit: create prev_chain with existing version
                let final_prev = if can_reuse_arena {
                    // Exceeded limit - drop all history, no allocation
                    None
                } else {
                    // Under limit - clone existing version and create chain
                    let existing_version = existing.version.clone();
                    let existing_prev = existing.prev.clone();
                    Some(Arc::new(VersionChainEntry {
                        version: existing_version,
                        prev: existing_prev,
                        // Historical versions don't use arena (slot reused by new HEAD)
                        arena_idx: None,
                    }))
                };

                let new_entry = VersionChainEntry {
                    version: new_version,
                    prev: final_prev,
                    arena_idx,
                };

                // Replace entry in-place (no additional tree traversal)
                occupied.insert(new_entry);
            }
            crate::common::cow_btree::Entry::Vacant(vacant) => {
                delta = isize::from(!is_new_version_deleted);
                // First version for this row - store in arena
                // OPTIMIZATION: Convert Row to Arc once, then just clone Arc (no data copy)
                let (arena_idx, final_version) = if version.deleted_at_txn_id == 0 {
                    let mut v = version;
                    // Convert Row to Arc once (takes ownership, no copy if already Arc)
                    let arc_data = std::mem::take(&mut v.data).into_arc();
                    // Insert Arc into arena (just Arc::clone, no data copy)
                    let (idx, bytes) = self.arena.install(
                        reservation,
                        None,
                        row_id,
                        v.txn_id,
                        CompactArc::clone(&arc_data),
                    );
                    // Create version with Arc-backed data for O(1) clone
                    v.data = Row::from_arc(arc_data);
                    payloads.shared_and_children += bytes;
                    payloads.versions += 1;
                    (Some(idx), v)
                } else {
                    payloads.add_row(&version.data);
                    (None, version)
                };

                let new_entry = VersionChainEntry {
                    version: final_version,
                    prev: None,
                    arena_idx,
                };

                // Insert into vacant slot (no additional traversal)
                vacant.insert(new_entry);
            }
        }
        delta
    }

    fn install_versions(
        &self,
        reservation: &mut ArenaReservation,
        batch: impl IntoIterator<Item = (i64, RowVersion)>,
    ) {
        let mut versions = self.versions.write();
        let mut delta = 0;
        let VersionTree { entries, payloads } = &mut *versions;
        for (row_id, version) in batch {
            delta += self.install_version(entries, reservation, payloads, row_id, version);
        }
        versions.publish_memory(&self.memory);
        if delta != 0 {
            self.committed_row_count
                .fetch_add(delta as usize, Ordering::Relaxed);
        }
    }

    /// Quick check if a row might exist
    pub fn quick_check_row_existence(&self, row_id: i64) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }

        self.versions.read().contains_key(row_id)
    }

    /// Gets the latest visible version of a row
    pub fn get_visible_version(&self, row_id: i64, txn_id: i64) -> Option<RowVersion> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let checker = self.visibility_checker.as_ref()?;

        // Drop the speculative arena guard before taking the versions lock.
        {
            let arena_guard = self.arena.read_guard();
            if let Some((meta, payload)) = arena_guard.probe(row_id) {
                if checker.is_visible(meta.txn_id, txn_id) {
                    if meta.deleted_at_txn_id != 0
                        && checker.is_visible(meta.deleted_at_txn_id, txn_id)
                    {
                        return None;
                    }
                    let data = Row::from_arc(CompactArc::clone(payload));
                    charge_export(&data);
                    return Some(RowVersion {
                        txn_id: meta.txn_id,
                        deleted_at_txn_id: meta.deleted_at_txn_id,
                        data,
                        // Arena metadata doesn't store create_time (saves 8 bytes/row).
                        // All callers of get_visible_version() only use .data, .is_deleted(),
                        // and .txn_id. AS OF queries use get_visible_version_as_of_timestamp()
                        // which goes through CowBTree and has the real create_time.
                        create_time: 0,
                    });
                }
            }
            // arena_guard dropped here before acquiring versions lock
        }

        // Phase 2: CowBTree fallback (correct lock ordering: versions first, then arena)
        let versions = self.versions.read();
        let chain = versions.get(row_id)?;

        // Check HEAD visibility (most common case)
        let head_txn_id = chain.version.txn_id;
        let head_deleted_at = chain.version.deleted_at_txn_id;

        if checker.is_visible(head_txn_id, txn_id) {
            if head_deleted_at != 0 && checker.is_visible(head_deleted_at, txn_id) {
                return None;
            }
            let row = chain.version.data.clone();
            charge_export(&row);
            return Some(RowVersion {
                txn_id: head_txn_id,
                deleted_at_txn_id: head_deleted_at,
                data: row,
                create_time: chain.version.create_time,
            });
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
                charge_export(&e.version.data);
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

        // Lock ordering: versions first, then arena (matches commit path)
        let versions = self.versions.read();
        let arena_guard = self.arena.read_guard();
        for &row_id in row_ids {
            if let Some((meta, _)) = arena_guard.probe(row_id) {
                if checker.is_visible(meta.txn_id, txn_id) {
                    if meta.deleted_at_txn_id == 0
                        || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                    {
                        return Some(row_id);
                    }
                    continue;
                }
            }

            // CowBTree lookup: O(log n)
            if let Some(chain) = versions.get(row_id) {
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
    /// Pre-acquires all locks once, then performs per-key CowBTree lookups
    /// with visibility checking and version chain traversal as needed.
    pub fn get_visible_versions_batch(&self, row_ids: &[i64], txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) || row_ids.is_empty() {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Lock ordering: versions first, then arena (matches commit path)
        let versions = self.versions.read();
        let arena_guard = self.arena.read_guard();
        let mut exports = ExportBatch::new();

        // Fast path: if both arena and version tree are empty, no rows can match.
        // This avoids iterating millions of phantom row_ids from volume-populated indexes.
        if arena_guard.is_empty() && versions.is_empty() {
            return RowVec::new();
        }

        // Don't over-allocate: most row_ids may be phantom (from volume indexes)
        let mut results = RowVec::with_capacity(row_ids.len().min(4096));

        for &row_id in row_ids {
            if let Some((meta, payload)) = arena_guard.probe(row_id) {
                if checker.is_visible(meta.txn_id, txn_id) {
                    if meta.deleted_at_txn_id == 0
                        || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                    {
                        let row = Row::from_arc(CompactArc::clone(payload));
                        exports.record(&row);
                        results.push((row_id, row));
                    }
                    continue;
                }
            }

            // CowBTree lookup: O(log n)
            if let Some(chain) = versions.get(row_id) {
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        let row = exports.capture(&chain.version.data);
                        results.push((row_id, row));
                    }
                    continue;
                }

                // Traverse version chain for older visible versions
                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    if checker.is_visible(e.version.txn_id, txn_id) {
                        if e.version.deleted_at_txn_id == 0
                            || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
                        {
                            let row = exports.capture(&e.version.data);
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

    /// Visits captured version payloads until the callback returns false.
    pub fn for_each_visible<F>(&self, row_ids: &[i64], txn_id: i64, mut callback: F)
    where
        F: FnMut(i64, Row) -> bool,
    {
        if self.closed.load(Ordering::Acquire) || row_ids.is_empty() {
            return;
        }

        if let [row_id] = row_ids {
            if let Some(version) = self.get_visible_version(*row_id, txn_id) {
                callback(*row_id, version.data);
            }
            return;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return,
        };

        let versions = self.capture_versions();
        let mut exports = ExportBatch::new();
        let mut payloads: [Option<CompactArc<[Value]>>; 128] = std::array::from_fn(|_| None);
        let mut remaining = row_ids;
        let mut batch_size = 1;
        while !remaining.is_empty() {
            let count = remaining.len().min(batch_size);
            {
                let current = self.versions.read();
                let live = versions.shares_root(&current).then_some(current);
                let arena = live.as_ref().map(|_| self.arena.read_guard());
                let mut last_visible_txn = 0;
                for (&row_id, slot) in remaining[..count].iter().zip(&mut payloads[..count]) {
                    if let Some((meta, payload)) = arena.as_ref().and_then(|a| a.probe(row_id)) {
                        if meta.txn_id == last_visible_txn
                            || checker.is_visible(meta.txn_id, txn_id)
                        {
                            last_visible_txn = meta.txn_id;
                            if meta.deleted_at_txn_id == 0
                                || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                            {
                                *slot = Some(CompactArc::clone(payload));
                            }
                            continue;
                        }
                    }
                    let mut current = versions.get(row_id);
                    while let Some(entry) = current {
                        if checker.is_visible(entry.version.txn_id, txn_id) {
                            if entry.version.deleted_at_txn_id == 0
                                || !checker.is_visible(entry.version.deleted_at_txn_id, txn_id)
                            {
                                *slot = Some(entry.version.data.clone().into_arc());
                            }
                            break;
                        }
                        current = entry.prev.as_deref();
                    }
                }
            }
            for (&row_id, slot) in remaining[..count].iter().zip(&mut payloads[..count]) {
                if let Some(payload) = slot.take() {
                    let row = Row::from_arc(payload);
                    exports.record(&row);
                    if !callback(row_id, row) {
                        return;
                    }
                }
            }
            remaining = &remaining[count..];
            batch_size = (batch_size * 2).min(128);
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
        if self.closed.load(Ordering::Acquire) || row_ids.is_empty() {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0,
        };

        /// Minimum batch size before enabling parallel processing
        #[cfg(feature = "parallel")]
        const PARALLEL_THRESHOLD: usize = 1000;
        /// Chunk size for parallel processing
        #[cfg(feature = "parallel")]
        const PARALLEL_CHUNK_SIZE: usize = 512;

        // Lock ordering: versions first, then arena (matches commit path)
        // Clone CowBTree for parallel path (O(1) Arc clone of root node)
        let versions = self.capture_versions();
        let arena_guard = self.arena.read_guard();

        #[cfg(feature = "parallel")]
        if row_ids.len() >= PARALLEL_THRESHOLD {
            return row_ids
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .map(|chunk| {
                    let mut chunk_count = 0;
                    for &row_id in chunk {
                        if let Some((meta, _)) = arena_guard.probe(row_id) {
                            if checker.is_visible(meta.txn_id, txn_id) {
                                if meta.deleted_at_txn_id == 0
                                    || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                                {
                                    chunk_count += 1;
                                }
                                continue;
                            }
                        }

                        // CowBTree fallback: O(log n)
                        if let Some(chain) = versions.get(row_id) {
                            let head_txn_id = chain.version.txn_id;
                            let head_deleted_at = chain.version.deleted_at_txn_id;

                            if checker.is_visible(head_txn_id, txn_id) {
                                if head_deleted_at == 0
                                    || !checker.is_visible(head_deleted_at, txn_id)
                                {
                                    chunk_count += 1;
                                }
                                continue;
                            }

                            let mut current: Option<&VersionChainEntry> =
                                chain.prev.as_ref().map(|b| b.as_ref());
                            while let Some(e) = current {
                                if checker.is_visible(e.version.txn_id, txn_id) {
                                    if e.version.deleted_at_txn_id == 0
                                        || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
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
                .sum();
        }

        // Sequential path for small batches (or when parallel feature is disabled)
        let mut count = 0;
        for &row_id in row_ids {
            if let Some((meta, _)) = arena_guard.probe(row_id) {
                if checker.is_visible(meta.txn_id, txn_id) {
                    if meta.deleted_at_txn_id == 0
                        || !checker.is_visible(meta.deleted_at_txn_id, txn_id)
                    {
                        count += 1;
                    }
                    continue;
                }
            }

            // CowBTree fallback: O(log n)
            if let Some(chain) = versions.get(row_id) {
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        count += 1;
                    }
                    continue;
                }

                let mut current: Option<&VersionChainEntry> =
                    chain.prev.as_ref().map(|b| b.as_ref());
                while let Some(e) = current {
                    if checker.is_visible(e.version.txn_id, txn_id) {
                        if e.version.deleted_at_txn_id == 0
                            || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        for &row_id in row_ids {
            if let Some(chain) = versions.get(row_id) {
                // FAST PATH: Check HEAD version first - O(1) for common case
                let head_txn_id = chain.version.txn_id;
                let head_deleted_at = chain.version.deleted_at_txn_id;

                if checker.is_visible(head_txn_id, txn_id) {
                    // HEAD is visible - check if deleted
                    if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                        let mut version_copy = chain.version.clone();
                        version_copy.create_time = current_seq;
                        results.push((row_id, exports.capture(&chain.version.data), version_copy));
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
                            results.push((row_id, exports.capture(&e.version.data), version_copy));
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

        // No need to clone tree for single-row lookup - just hold read guard
        let versions = self.versions.read();
        let chain = versions.get(row_id)?;

        // Traverse version chain from newest to oldest
        let mut current: Option<&VersionChainEntry> = Some(chain);
        while let Some(e) = current {
            // Check if this version was created before or at the asOf transaction
            if e.version.txn_id <= as_of_txn_id {
                // Check if deleted before or at asOfTxnID
                if e.version.deleted_at_txn_id != 0 && e.version.deleted_at_txn_id <= as_of_txn_id {
                    return None;
                }
                charge_export(&e.version.data);
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

        // No need to clone tree for single-row lookup - just hold read guard
        let versions = self.versions.read();
        let chain = versions.get(row_id)?;

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
                charge_export(&e.version.data);
                return Some(e.version.clone());
            }
            current = e.prev.as_ref().map(|b| b.as_ref());
        }

        None
    }

    /// Returns all row IDs in the version store (sorted)
    pub fn get_all_row_ids(&self) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) {
            return Vec::new();
        }

        // Clone tree to avoid holding lock for the duration of iteration
        let versions = self.capture_versions();
        versions.keys().collect()
    }

    /// Populate a FxHashSet with all hot row_ids. Iterates the B-tree under
    /// read lock without cloning — much cheaper than get_all_row_ids() + collect()
    /// for skip-set construction.
    pub fn collect_row_ids_into(&self, dest: &mut rustc_hash::FxHashSet<i64>) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let versions = self.versions.read();
        for key in versions.keys() {
            dest.insert(key);
        }
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

        // FAST PATH: Scan arena directly when no uncommitted writes.
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        {
            let uncommitted_empty = self.uncommitted_writes.read().is_empty();
            if uncommitted_empty && !checker.needs_snapshot_isolation(txn_id) {
                let arena_guard = self.arena.read_guard();
                if !arena_guard.is_empty() {
                    let mut visible_row_ids =
                        Vec::with_capacity(self.committed_row_count.load(Ordering::Relaxed));
                    for (meta, _) in arena_guard.rows() {
                        if meta.txn_id != 0
                            && meta.deleted_at_txn_id == 0
                            && checker.is_visible(meta.txn_id, txn_id)
                        {
                            visible_row_ids.push(meta.row_id);
                        }
                    }
                    return visible_row_ids;
                }
            }
        }

        // SLOW PATH: Full CowBTree iteration
        let versions = self.capture_versions();
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

        // FAST PATH: Scan arena directly when no uncommitted writes.
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        {
            let uncommitted_empty = self.uncommitted_writes.read().is_empty();
            if uncommitted_empty && !checker.needs_snapshot_isolation(txn_id) {
                let arena_guard = self.arena.read_guard();
                if !arena_guard.is_empty() {
                    let mut count = 0usize;
                    for (meta, _) in arena_guard.rows() {
                        if meta.txn_id != 0
                            && meta.deleted_at_txn_id == 0
                            && checker.is_visible(meta.txn_id, txn_id)
                        {
                            count += 1;
                        }
                    }
                    return count;
                }
            }
        }

        // SLOW PATH: Full CowBTree iteration
        let mut count = 0;
        let versions = self.capture_versions();
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

    /// Returns the count of committed non-deleted rows (O(1) operation)
    ///
    /// This is the fast path for COUNT(*) queries without WHERE clause.
    /// The counter is updated atomically on INSERT commit (+1) and DELETE commit (-1).
    ///
    /// # Important
    ///
    /// This count does NOT include uncommitted changes from the current transaction.
    /// Use this only for queries that see committed data (e.g., autocommit queries).
    #[inline]
    pub fn committed_row_count(&self) -> usize {
        self.committed_row_count.load(Ordering::Relaxed)
    }

    /// Live arena payload requests used by the existing hot-size trigger.
    #[inline]
    pub fn hot_bytes(&self) -> usize {
        self.arena.bytes()
    }

    /// Canonical, pinned-root and retired-arena payload bounds, with shared overcount.
    #[cfg(test)]
    pub(crate) fn version_payload_footprint(&self) -> (usize, usize, usize) {
        let usage = self.memory.usage();
        (
            usage.version_payloads,
            usage.pinned_version_payloads,
            usage.retired_arena_payloads,
        )
    }

    pub(crate) fn memory_account(&self) -> &Arc<TableMemory> {
        &self.memory
    }

    /// Arena slots in use (deleted and cleared ones included) and the bytes
    /// its slot vectors reserve
    pub fn arena_footprint(&self) -> (usize, usize) {
        (self.arena.slot_count(), self.arena.capacity_bytes())
    }

    pub(crate) fn prepare_arena_retirement(&self, rows: usize) -> ArenaRetirement {
        self.arena.prepare_clear(rows)
    }

    pub(crate) fn finish_arena_retirement(&self, retirement: &mut ArenaRetirement) {
        self.arena.finish_clear(retirement);
    }

    /// Previous versions kept alive by the version chains
    pub fn chain_entries(&self) -> usize {
        let versions = self.versions.read();
        versions.payloads.versions - versions.len()
    }

    /// Check if a row_id exists in the committed version store (B-tree).
    /// Used during WAL replay to distinguish sealed INSERTs from post-seal UPDATEs.
    pub fn has_committed_row(&self, row_id: i64) -> bool {
        self.versions.read().contains_key(row_id)
    }

    /// Atomically subtract sealed rows from committed_row_count.
    ///
    /// After seal batch-removes rows from the B-tree, committed_row_count must
    /// decrease by the sealed amount. Using fetch_sub (not store) preserves
    /// concurrent fetch_add/fetch_sub from other threads committing INSERTs or
    /// DELETEs during the seal window. A plain store() would race with those
    /// concurrent updates, causing the counter to drift by ~N (where N is the
    /// number of rows committed during the seal operation).
    pub fn subtract_committed_row_count(&self, sealed: usize) {
        // Use saturating arithmetic to avoid wrapping if concurrent deletes
        // already reduced the counter below the sealed amount.
        loop {
            let current = self.committed_row_count.load(Ordering::Relaxed);
            let new_val = current.saturating_sub(sealed);
            match self.committed_row_count.compare_exchange_weak(
                current,
                new_val,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Check if a transaction requires snapshot isolation visibility checks.
    /// When true, the O(1) committed_row_count is inaccurate because it includes
    /// rows committed after this transaction's snapshot point.
    #[inline]
    /// Marks a commit's publish, from its index updates until its versions
    /// are visible or undone, so a reader that trusts the index order can
    /// stand down
    pub fn begin_publish(self: &Arc<Self>) -> PublishGuard {
        self.publishing.fetch_add(1, Ordering::SeqCst);
        PublishGuard {
            store: Arc::clone(self),
        }
    }

    /// The publish epoch while no commit is publishing; None while one is.
    /// Equal values before and after a read mean no publish overlapped it.
    pub fn publish_epoch_if_quiet(&self) -> Option<u64> {
        if self.publishing.load(Ordering::SeqCst) != 0 {
            return None;
        }
        Some(self.publish_epoch.load(Ordering::SeqCst))
    }

    pub fn needs_snapshot_isolation(&self, txn_id: i64) -> bool {
        self.visibility_checker
            .as_ref()
            .is_some_and(|c| c.needs_snapshot_isolation(txn_id))
    }

    /// Clears all data in O(1) for TRUNCATE.
    /// Drops all versions, arena data, indexes, and resets row count.
    /// Returns the number of rows that were truncated.
    ///
    /// **Non-rollbackable**: Like MySQL/Oracle, data is physically destroyed
    /// immediately with no undo log. This is O(1) vs O(N) for DELETE.
    /// PostgreSQL/SQL Server support rollbackable TRUNCATE by deferring the
    /// physical destruction to commit time, but that adds complexity to every
    /// read path (13+ code locations). Use `DELETE FROM table` for rollback.
    ///
    /// Fails with `TableHasActiveTransactions` if another transaction holds
    /// uncommitted UPDATE/DELETE claims on this table.
    pub fn truncate_all(&self) -> crate::core::Result<TruncateResult> {
        // Hold uncommitted_writes(W) for the ENTIRE check-and-clear sequence
        // to prevent TOCTOU race: without this, a concurrent try_claim_row()
        // could add a claim between the check and the clear, and truncate would
        // silently destroy it — causing a ghost row when the UPDATE commits.
        //
        // Lock ordering: uncommitted_writes(W) → versions(W) → arena(W).
        // This is safe because all read methods that use uncommitted_writes
        // check it BEFORE acquiring arena(R), so there is no circular dependency:
        //   - Read paths: uncommitted_writes(R) [brief, dropped] → arena(R)
        //   - Commit path: versions(W) → arena(W) (never uncommitted_writes(W))
        //   - try_claim_row: uncommitted_writes(W) only — blocked by our lock
        let mut uncommitted = self.uncommitted_writes.write();
        if !uncommitted.is_empty() {
            return Err(crate::core::Error::TableHasActiveTransactions);
        }

        // Hold versions(W) while clearing BOTH versions AND arena atomically.
        // INSERT commits acquire versions(W) → arena(W) in the same order,
        // so this blocks them and prevents a race where a commit inserts a
        // version pointing to an arena slot that we then clear.
        // Reset committed_row_count inside the lock so COUNT(*) never sees 0
        // while data still exists.
        let result;
        {
            let mut versions = self.versions.write();
            let arena = self.arena.clear_all()?;
            let count = self.committed_row_count.swap(0, Ordering::SeqCst) as i32;
            let charge = VersionSnapshotCharge::new(&self.memory, &versions);
            let entries = std::mem::take(&mut versions.entries);
            versions.payloads = VersionPayloads::default();
            versions.publish_memory(&self.memory);
            result = TruncateResult {
                rows_affected: count,
                _versions: Some(VersionSnapshot {
                    inner: entries,
                    _charge: charge,
                }),
                _arena: arena,
            };
        }

        // Clear uncommitted_writes (already held as write lock)
        uncommitted.clear();
        drop(uncommitted);

        // 5. Clear all indexes
        let indexes = self.indexes.read();
        for index in indexes.values() {
            let _ = index.clear();
        }

        // 6. Reset auto-increment counter so new rows start at 1
        //    and the speculative arena probe ((row_id - 1) as usize) stays valid
        self.auto_increment_counter.store(0, Ordering::Release);

        // 7. Invalidate zone maps (stale after truncate)
        *self.zone_maps.write() = None;

        Ok(result)
    }

    /// Returns all visible rows for a transaction (optimized batch operation)
    ///
    /// This is more efficient than calling get_visible_version for each row
    /// because it batches the visibility checks and avoids repeated map lookups.
    /// Results are sorted by row_id.
    #[inline]
    pub fn get_all_visible_rows(&self, txn_id: i64) -> RowVec {
        self.get_all_visible_rows_internal(txn_id)
    }

    /// Extract visible rows AND the CowBTree snapshot at extraction time.
    /// Used by seal: the snapshot records each row's `txn_id` so that
    /// `remove_sealed_rows` can detect concurrent commits and skip them.
    pub fn extract_for_seal(&self, txn_id: i64) -> (RowVec, ExtractionSnapshot) {
        let snapshot = self.capture_versions();
        let rows = self.get_all_visible_rows_internal(txn_id);
        (rows, ExtractionSnapshot { inner: snapshot })
    }

    /// Extract committed rows for seal, filtered by commit_seq cutoff.
    /// Only includes rows committed before `commit_seq_cutoff`, ensuring
    /// snapshot isolation transactions that began before the cutoff can still
    /// see those rows after they move to cold storage.
    pub fn extract_for_seal_with_cutoff(
        &self,
        commit_seq_cutoff: i64,
    ) -> (RowVec, ExtractionSnapshot) {
        let snapshot = self.capture_versions();
        let mut rows = RowVec::with_capacity(self.committed_row_count());
        self.for_each_committed_version_with_cutoff(
            |row_id, version| {
                rows.push((row_id, version.data.clone()));
                true
            },
            commit_seq_cutoff,
        );
        (rows, ExtractionSnapshot { inner: snapshot })
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        let mut results = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    results.push((row_id, exports.capture(&chain.version.data)));
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
                        results.push((row_id, exports.capture(&e.version.data)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        results
    }

    /// Returns all visible rows from a captured version root.
    #[inline]
    pub fn get_all_visible_rows_arena(&self, txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        let mut result = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    result.push((row_id, exports.capture(&chain.version.data)));
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
                        result.push((row_id, exports.capture(&e.version.data)));
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

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
                    result.push((row_id, exports.capture(&chain.version.data)));
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
                        result.push((row_id, exports.capture(&e.version.data)));
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
    /// Returns: Vec of (row_id, row_data, original_version) tuples.
    /// Note: read_version_seq is obtained from visibility_checker, not from create_time.
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

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
                    result.push((row_id, exports.capture(&chain.version.data), version_copy));
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
                        result.push((row_id, exports.capture(&e.version.data), version_copy));
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
        let fully_compiled = compiled_filter.is_fully_compiled();
        let matches_row = |row: &Row| {
            if fully_compiled {
                compiled_filter.matches_arc_slice(row.as_slice())
            } else {
                compiled_filter.matches(row)
            }
        };

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Single-pass: read, filter, and collect in one loop
        let mut result: Vec<(i64, Row, RowVersion)> = Vec::with_capacity(versions.len() / 4);

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if (head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id))
                    && matches_row(&chain.version.data)
                {
                    let mut version_copy = chain.version.clone();
                    version_copy.create_time = current_seq;
                    result.push((row_id, exports.capture(&chain.version.data), version_copy));
                }
                continue;
            }

            // SLOW PATH: HEAD not visible - traverse chain for older versions
            let mut current: Option<&VersionChainEntry> = chain.prev.as_ref().map(|b| b.as_ref());
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if (deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id))
                        && matches_row(&e.version.data)
                    {
                        let mut version_copy = e.version.clone();
                        version_copy.create_time = current_seq;
                        result.push((row_id, exports.capture(&e.version.data), version_copy));
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
    /// Note: Functionally identical to get_all_visible_rows_arena (iteration is ordered).
    #[inline]
    pub fn get_all_visible_rows_unsorted(&self, txn_id: i64) -> RowVec {
        if self.closed.load(Ordering::Acquire) {
            return RowVec::new();
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return RowVec::new(),
        };

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        let mut result = RowVec::with_capacity(versions.len());

        for (&row_id, chain) in versions.iter() {
            // FAST PATH: Check HEAD version first - O(1) for common case
            let head_txn_id = chain.version.txn_id;
            let head_deleted_at = chain.version.deleted_at_txn_id;

            if checker.is_visible(head_txn_id, txn_id) {
                // HEAD is visible - check if deleted
                if head_deleted_at == 0 || !checker.is_visible(head_deleted_at, txn_id) {
                    result.push((row_id, exports.capture(&chain.version.data)));
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
                        result.push((row_id, exports.capture(&e.version.data)));
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
    /// Iteration is ordered by row_id, enabling true
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Collect with early termination
        // A user-supplied limit can be i64::MAX; only pre-allocate what
        // the store could actually return
        let mut result = RowVec::with_capacity(limit.min(4096));
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
                    result.push((row_id, exports.capture(&entry.version.data)));
                    if result.len() >= limit {
                        break; // Early termination!
                    }
                }
            }
        }

        result
    }

    /// The row ids visible to `txn_id`, in key order, leaving out the ones
    /// in `exclude` and stopping after `limit` of them
    pub fn visible_row_ids_excluding(
        &self,
        txn_id: i64,
        exclude: &crate::common::I64Set,
        limit: usize,
    ) -> Vec<i64> {
        if self.closed.load(Ordering::Acquire) || limit == 0 {
            return Vec::new();
        }
        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };
        // Read under the lock rather than cloning the tree: the walk stops
        // at the limit, and a full one only reads keys and heads
        let versions = self.versions.read();
        let mut result = Vec::with_capacity(limit.min(4096));
        for (&row_id, chain) in versions.iter() {
            if exclude.contains(row_id) {
                continue;
            }
            // The newest version this transaction sees decides whether
            // the row exists for it
            let mut current: Option<&VersionChainEntry> = Some(chain);
            let mut exists = false;
            while let Some(e) = current {
                if checker.is_visible(e.version.txn_id, txn_id) {
                    let deleted_at = e.version.deleted_at_txn_id;
                    exists = deleted_at == 0 || !checker.is_visible(deleted_at, txn_id);
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
            if exists {
                result.push(row_id);
                if result.len() >= limit {
                    break;
                }
            }
        }
        result
    }

    /// Get visible rows with LIMIT (with early termination).
    ///
    /// Note: Functionally identical to get_visible_rows_with_limit. Kept for API compatibility.
    #[inline]
    pub fn get_visible_rows_with_limit_unordered(
        &self,
        txn_id: i64,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        // Delegate to the sorted version
        self.get_visible_rows_with_limit(txn_id, limit, offset)
    }

    /// Get a batch of visible rows starting after a given row_id (cursor-based pagination).
    ///
    /// This is designed for lazy/streaming scanners that fetch rows in batches.
    /// Uses range() for efficient cursor-based iteration.
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Use range for efficient cursor-based iteration
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
                result.push((row_id, exports.capture(&entry.version.data)));
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Use range for efficient cursor-based iteration
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
                buffer.push((row_id, exports.capture(&entry.version.data)));
            }
        }

        has_more
    }

    /// Collect visible rows ordered by row_id (PRIMARY KEY) with efficient OFFSET/LIMIT
    ///
    /// This method uses range iteration to efficiently skip OFFSET rows
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Collect with offset/limit
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
                                result.push((*row_id, exports.capture(&entry.version.data)));
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
            // Reverse iteration with early termination: O(limit + offset) instead of O(n)
            for (&row_id, chain) in versions.iter_rev() {
                let mut current: Option<&VersionChainEntry> = Some(chain);
                while let Some(entry) = current {
                    if checker.is_visible(entry.version.txn_id, txn_id) {
                        if entry.version.deleted_at_txn_id == 0
                            || !checker.is_visible(entry.version.deleted_at_txn_id, txn_id)
                        {
                            if skipped < offset {
                                skipped += 1;
                            } else {
                                result.push((row_id, exports.capture(&entry.version.data)));
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
        }

        Some(result)
    }

    /// Collect visible rows using keyset pagination (WHERE id > X ORDER BY id LIMIT Y)
    ///
    /// This method uses range iteration starting from a specific row_id,
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

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Helper to find visible version and get row data
        let mut find_visible_row = |chain: &VersionChainEntry| -> Option<Row> {
            let mut current: Option<&VersionChainEntry> = Some(chain);
            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    return Some(exports.capture(&e.version.data));
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

        // Collect with early termination
        // A user-supplied limit can be i64::MAX; only pre-allocate what
        // the store could actually return
        let mut result = RowVec::with_capacity(limit.min(4096));

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
            // Reverse range iteration with early termination: O(limit) instead of O(n)
            for (&row_id, chain) in
                versions.range_rev((start_bound, std::ops::Bound::Unbounded::<i64>))
            {
                if let Some(row_data) = find_visible_row(chain) {
                    result.push((row_id, row_data));
                    if result.len() >= limit {
                        break;
                    }
                }
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
        let fully_compiled = compiled_filter.is_fully_compiled();
        let matches_row = |row: &Row| {
            if fully_compiled {
                compiled_filter.matches_arc_slice(row.as_slice())
            } else {
                compiled_filter.matches(row)
            }
        };

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Single-pass: read, filter, and collect in one loop
        let mut result = RowVec::with_capacity(versions.len() / 4);

        for (&row_id, chain) in versions.iter() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break; // Row is deleted
                    }

                    if matches_row(&e.version.data) {
                        result.push((row_id, exports.capture(&e.version.data)));
                    }
                    break;
                }
                current = e.prev.as_deref();
            }
        }

        result
    }

    /// Iterate visible rows matching a filter with early termination via callback.
    ///
    /// Calls `callback(row_id, row_data)` for each matching row. The callback
    /// returns `true` to continue or `false` to stop iteration.
    /// This avoids materializing all matching rows into a Vec.
    pub fn for_each_visible_filtered<F>(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
        mut callback: F,
    ) where
        F: FnMut(i64, Row) -> bool,
    {
        if self.closed.load(Ordering::Acquire) {
            return;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return,
        };

        let schema = self.schema.read();
        let compiled_filter = CompiledFilter::compile(filter, &schema);
        drop(schema);
        let fully_compiled = compiled_filter.is_fully_compiled();
        let matches_row = |row: &Row| {
            if fully_compiled {
                compiled_filter.matches_arc_slice(row.as_slice())
            } else {
                compiled_filter.matches(row)
            }
        };

        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        for (&row_id, chain) in versions.iter() {
            let mut current: Option<&VersionChainEntry> = Some(chain);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                if checker.is_visible(version_txn_id, txn_id) {
                    if deleted_at_txn_id != 0 && checker.is_visible(deleted_at_txn_id, txn_id) {
                        break;
                    }

                    if matches_row(&e.version.data)
                        && !callback(row_id, exports.capture(&e.version.data))
                    {
                        return;
                    }
                    break;
                }
                current = e.prev.as_deref();
            }
        }
    }

    /// Get visible rows with filter, limit and offset applied at the storage layer.
    ///
    /// # True Early Termination
    /// Iteration is ordered. After collecting `limit` matching rows
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
        let fully_compiled = compiled_filter.is_fully_compiled();
        let matches_row = |row: &Row| {
            if fully_compiled {
                compiled_filter.matches_arc_slice(row.as_slice())
            } else {
                compiled_filter.matches(row)
            }
        };

        // Clone CowBTree to release read lock early, allowing concurrent commits
        let versions = self.snapshot_versions();
        let mut exports = ExportBatch::new();

        // Collect with offset/limit and early termination
        // A user-supplied limit can be i64::MAX; only pre-allocate what
        // the store could actually return
        let mut result = RowVec::with_capacity(limit.min(4096));
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

                    if matches_row(&e.version.data) {
                        if skipped < offset {
                            skipped += 1;
                        } else {
                            result.push((row_id, exports.capture(&e.version.data)));
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
    /// Note: Functionally identical to get_visible_rows_filtered_with_limit. Kept for API compatibility.
    #[inline]
    pub fn get_visible_rows_filtered_with_limit_unordered(
        &self,
        txn_id: i64,
        filter: &dyn crate::storage::expression::Expression,
        limit: usize,
        offset: usize,
    ) -> RowVec {
        // Delegate to the sorted version
        self.get_visible_rows_filtered_with_limit(txn_id, filter, limit, offset)
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

        // OPTIMIZATION: Separate accumulators to avoid i64->f64 conversion per row.
        // Uses i128 for integer accumulation to prevent overflow (i128 holds sum of
        // 2^63 rows of i64::MAX without overflow).
        let mut int_sum = 0i128;
        let mut float_sum = 0.0f64;
        let mut count = 0usize;

        // Helper to accumulate numeric value
        #[inline(always)]
        fn accumulate_sum(int_sum: &mut i128, float_sum: &mut f64, count: &mut usize, val: &Value) {
            match val {
                Value::Integer(i) => {
                    *int_sum += *i as i128;
                    *count += 1;
                }
                Value::Float(f) => {
                    *float_sum += *f;
                    *count += 1;
                }
                // A boolean counts as one or nought, as SUM reads it
                Value::Boolean(b) => {
                    *int_sum += *b as i128;
                    *count += 1;
                }
                _ => {} // NULL or non-numeric
            }
        }

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass).
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        //
        // LOCK ORDERING: Check uncommitted_writes BEFORE acquiring arena to maintain
        // consistent ordering with truncate_all (uncommitted_writes → arena).
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        {
            let arena_guard = self.arena.read_guard();
            if uncommitted_empty
                && !arena_guard.is_empty()
                && !checker.needs_snapshot_isolation(txn_id)
            {
                // OPTIMIZATION: Cache visibility result for repeated txn_ids
                // When rows are inserted in batches, consecutive rows often have the same txn_id.
                // Caching avoids repeated thread-local access overhead (~27ms per 100K rows).
                let mut last_txn_id: i64 = 0;
                let mut last_visible: bool = false;

                for (meta, payload) in arena_guard.rows() {
                    if meta.txn_id != 0 && meta.deleted_at_txn_id == 0 {
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
                            if let Some(val) = payload.get(col_idx) {
                                accumulate_sum(&mut int_sum, &mut float_sum, &mut count, val);
                            }
                        }
                    }
                }
                return (int_sum as f64 + float_sum, count);
            }
            // arena_guard dropped here — no need to hold it for slow path
        }

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.capture_versions();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = 0;
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

                if is_vis {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        // This version is visible and not deleted - accumulate its value
                        if let Some(val) = e.version.data.get(col_idx) {
                            accumulate_sum(&mut int_sum, &mut float_sum, &mut count, val);
                        }
                    }
                    break; // Always break on first visible version
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

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass).
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        //
        // LOCK ORDERING: Check uncommitted_writes BEFORE acquiring arena to maintain
        // consistent ordering with truncate_all (uncommitted_writes → arena).
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        {
            let arena_guard = self.arena.read_guard();
            if uncommitted_empty
                && !arena_guard.is_empty()
                && !checker.needs_snapshot_isolation(txn_id)
            {
                // OPTIMIZATION: Cache visibility result for repeated txn_ids
                let mut last_txn_id: i64 = 0;
                let mut last_visible: bool = false;

                for (meta, payload) in arena_guard.rows() {
                    if meta.txn_id != 0 && meta.deleted_at_txn_id == 0 {
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
                            if let Some(val) = payload.get(col_idx) {
                                update_min(&mut min_val, val);
                            }
                        }
                    }
                }
                if let Some(value) = &min_val {
                    charge_value_export(value);
                }
                return min_val;
            }
            // arena_guard dropped here — no need to hold it for slow path
        }

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.capture_versions();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = 0;
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

                if is_vis {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        if let Some(val) = e.version.data.get(col_idx) {
                            update_min(&mut min_val, val);
                        }
                    }
                    break; // Always break on first visible version
                }

                current = e.prev.as_deref();
            }
        }

        if let Some(value) = &min_val {
            charge_value_export(value);
        }
        min_val
    }

    /// Compute MAX(column) without materializing full rows
    ///
    /// OPTIMIZATION: Single-pass approach that combines visibility checking with max computation.
    pub fn max_column(&self, txn_id: i64, col_idx: usize) -> Option<Value> {
        let checker = self.visibility_checker.as_ref()?;

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

        // FAST PATH: If uncommitted_writes is empty, scan arena directly (single pass).
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        //
        // LOCK ORDERING: Check uncommitted_writes BEFORE acquiring arena to maintain
        // consistent ordering with truncate_all (uncommitted_writes → arena).
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();

        {
            let arena_guard = self.arena.read_guard();
            if uncommitted_empty
                && !arena_guard.is_empty()
                && !checker.needs_snapshot_isolation(txn_id)
            {
                // OPTIMIZATION: Cache visibility result for repeated txn_ids
                let mut last_txn_id: i64 = 0;
                let mut last_visible: bool = false;

                for (meta, payload) in arena_guard.rows() {
                    if meta.txn_id != 0 && meta.deleted_at_txn_id == 0 {
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
                            if let Some(val) = payload.get(col_idx) {
                                update_max(&mut max_val, val);
                            }
                        }
                    }
                }
                if let Some(value) = &max_val {
                    charge_value_export(value);
                }
                return max_val;
            }
            // arena_guard dropped here — no need to hold it for slow path
        }

        // SLOW PATH: Full iteration over version chains (single pass)
        let versions = self.capture_versions();

        // Cache visibility for slow path too
        let mut last_txn_id: i64 = 0;
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

                if is_vis {
                    if deleted_at_txn_id == 0 || !checker.is_visible(deleted_at_txn_id, txn_id) {
                        if let Some(val) = e.version.data.get(col_idx) {
                            update_max(&mut max_val, val);
                        }
                    }
                    break; // Always break on first visible version
                }

                current = e.prev.as_deref();
            }
        }

        if let Some(value) = &max_val {
            charge_value_export(value);
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
        let empty_result = || {
            aggregates
                .iter()
                .map(|(op, _)| match op {
                    AggregateOp::Count | AggregateOp::CountStar => AggregateResult::Count(0),
                    AggregateOp::Sum => AggregateResult::Sum(0.0, 0),
                    AggregateOp::Min => AggregateResult::Min(None),
                    AggregateOp::Max => AggregateResult::Max(None),
                    AggregateOp::Avg => AggregateResult::Avg(0.0, 0),
                })
                .collect()
        };

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return empty_result(),
        };

        // Initialize accumulators
        let mut results: Vec<AggregateAccumulator> = aggregates
            .iter()
            .map(|(op, _)| match op {
                AggregateOp::Count | AggregateOp::CountStar => AggregateAccumulator::Count(0),
                AggregateOp::Sum => AggregateAccumulator::Sum(0, 0.0, 0),
                AggregateOp::Min => AggregateAccumulator::Min(None),
                AggregateOp::Max => AggregateAccumulator::Max(None),
                AggregateOp::Avg => AggregateAccumulator::Avg(0, 0.0, 0),
            })
            .collect();

        // Helper to update accumulator with a value
        fn update_accumulator(acc: &mut AggregateAccumulator, op: &AggregateOp, val: &Value) {
            match (acc, op) {
                (AggregateAccumulator::Count(c), AggregateOp::Count) if !val.is_null() => {
                    *c += 1;
                }
                (AggregateAccumulator::Count(c), AggregateOp::CountStar) => {
                    *c += 1;
                }
                (AggregateAccumulator::Sum(int_sum, float_sum, cnt), AggregateOp::Sum) => match val
                {
                    Value::Integer(i) => {
                        *int_sum += *i as i128;
                        *cnt += 1;
                    }
                    Value::Float(f) => {
                        *float_sum += *f;
                        *cnt += 1;
                    }
                    Value::Boolean(b) => {
                        *int_sum += *b as i128;
                        *cnt += 1;
                    }
                    _ => {}
                },
                (AggregateAccumulator::Min(min), AggregateOp::Min) if !val.is_null() => match min {
                    None => *min = Some(val.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Less) = val.compare(current) {
                            *min = Some(val.clone());
                        }
                    }
                },
                (AggregateAccumulator::Max(max), AggregateOp::Max) if !val.is_null() => match max {
                    None => *max = Some(val.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                            *max = Some(val.clone());
                        }
                    }
                },
                (AggregateAccumulator::Avg(int_sum, float_sum, cnt), AggregateOp::Avg) => match val
                {
                    Value::Integer(i) => {
                        *int_sum += *i as i128;
                        *cnt += 1;
                    }
                    Value::Float(f) => {
                        *float_sum += *f;
                        *cnt += 1;
                    }
                    Value::Boolean(b) => {
                        *int_sum += *b as i128;
                        *cnt += 1;
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Helper macro: accumulate values from a row-like source by column index
        macro_rules! accumulate_from {
            ($src:expr, $results:expr, $aggregates:expr) => {
                for (i, (op, col_idx)) in $aggregates.iter().enumerate() {
                    if let Some(val) = $src.get(*col_idx) {
                        update_accumulator(&mut $results[i], op, val);
                    }
                }
            };
        }

        // FAST PATH: Scan arena directly when no uncommitted writes.
        // SAFETY: Arena heads are used only under ReadCommitted isolation.
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();
        if uncommitted_empty && !checker.needs_snapshot_isolation(txn_id) {
            let arena_guard = self.arena.read_guard();
            if !arena_guard.is_empty() {
                for (meta, arc_row) in arena_guard.rows() {
                    if meta.txn_id != 0
                        && meta.deleted_at_txn_id == 0
                        && checker.is_visible(meta.txn_id, txn_id)
                    {
                        accumulate_from!(arc_row, results, aggregates);
                    }
                }

                return results
                    .into_iter()
                    .map(|acc| match acc {
                        AggregateAccumulator::Count(c) => AggregateResult::Count(c),
                        AggregateAccumulator::Sum(is, fs, c) => {
                            AggregateResult::Sum(is as f64 + fs, c)
                        }
                        AggregateAccumulator::Min(v) => {
                            if let Some(value) = &v {
                                charge_value_export(value);
                            }
                            AggregateResult::Min(v)
                        }
                        AggregateAccumulator::Max(v) => {
                            if let Some(value) = &v {
                                charge_value_export(value);
                            }
                            AggregateResult::Max(v)
                        }
                        AggregateAccumulator::Avg(is, fs, c) => {
                            AggregateResult::Avg(is as f64 + fs, c)
                        }
                    })
                    .collect();
            }
        }

        // SLOW PATH: Read the payload owned by each captured version.
        let versions = self.snapshot_versions();

        for chain in versions.values() {
            let mut current: Option<&VersionChainEntry> = Some(chain);
            while let Some(e) = current {
                if checker.is_visible(e.version.txn_id, txn_id) {
                    if e.version.deleted_at_txn_id == 0
                        || !checker.is_visible(e.version.deleted_at_txn_id, txn_id)
                    {
                        accumulate_from!(e.version.data, results, aggregates);
                    }
                    break;
                }
                current = e.prev.as_ref().map(|b| b.as_ref());
            }
        }

        // Convert accumulators to results
        results
            .into_iter()
            .map(|acc| match acc {
                AggregateAccumulator::Count(c) => AggregateResult::Count(c),
                AggregateAccumulator::Sum(is, fs, c) => AggregateResult::Sum(is as f64 + fs, c),
                AggregateAccumulator::Min(v) => {
                    if let Some(value) = &v {
                        charge_value_export(value);
                    }
                    AggregateResult::Min(v)
                }
                AggregateAccumulator::Max(v) => {
                    if let Some(value) = &v {
                        charge_value_export(value);
                    }
                    AggregateResult::Max(v)
                }
                AggregateAccumulator::Avg(is, fs, c) => AggregateResult::Avg(is as f64 + fs, c),
            })
            .collect()
    }

    /// Returns the count of rows
    pub fn row_count(&self) -> usize {
        self.versions.read().len()
    }
    /// Tries to claim a row for update (dirty write prevention)
    pub fn try_claim_row(&self, row_id: i64, txn_id: i64) -> Result<(), Error> {
        use crate::common::i64_map::Entry;

        let mut map = self.uncommitted_writes.write();
        let before = map.allocation_bytes();
        let result = match map.entry(row_id) {
            Entry::Occupied(e) => {
                let existing_txn = *e.get();
                if existing_txn != txn_id {
                    return Err(Error::internal(format!(
                        "row {} has uncommitted changes from transaction {}",
                        row_id, existing_txn
                    )));
                }
                Ok(())
            }
            Entry::Vacant(e) => {
                e.insert(txn_id);
                Ok(())
            }
        };
        self.claim_memory.resize(before, &map);
        result
    }

    /// Releases a row claim
    pub fn release_row_claim(&self, row_id: i64, txn_id: i64) {
        let mut map = self.uncommitted_writes.write();
        let before = map.allocation_bytes();
        if let Some(&v) = map.get(row_id) {
            if v == txn_id {
                map.remove(row_id);
            }
        }
        self.claim_memory.resize(before, &map);
    }

    /// Releases multiple row claims in batch
    /// OPTIMIZATION: Single lock acquisition for all removals
    #[inline]
    pub fn release_row_claims_batch(&self, row_ids: &[i64], txn_id: i64) {
        let mut map = self.uncommitted_writes.write();
        let before = map.allocation_bytes();
        for &row_id in row_ids {
            if let Some(&v) = map.get(row_id) {
                if v == txn_id {
                    map.remove(row_id);
                }
            }
        }
        self.claim_memory.resize(before, &map);
    }

    /// Check if an index exists
    pub fn index_exists(&self, index_name: &str) -> bool {
        let indexes = self.indexes.read();
        indexes.contains_key(index_name)
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<String> {
        let indexes = self.indexes.read();
        indexes
            .iter()
            .filter(|(_, idx)| idx.index_type() != crate::core::IndexType::PrimaryKey)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Check if there are any indexes
    #[inline]
    pub fn has_indexes(&self) -> bool {
        let indexes = self.indexes.read();
        !indexes.is_empty()
    }

    /// Iterate over all indexes, calling the provided function for each
    /// OPTIMIZATION: Avoids collecting index names and allows early exit on error
    pub fn for_each_index<F>(&self, mut f: F) -> crate::core::Result<()>
    where
        F: FnMut(&Arc<dyn Index>) -> crate::core::Result<()>,
    {
        let indexes = self.indexes.read();
        for index in indexes.values() {
            f(index)?;
        }
        Ok(())
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

    /// Acquire the per-table upsert mutex. Returns an owned guard.
    /// Only used for ON CONFLICT statements to serialize check+insert+commit.
    #[inline]
    pub fn acquire_upsert_lock(&self) -> parking_lot::ArcMutexGuard<parking_lot::RawMutex, ()> {
        parking_lot::Mutex::lock_arc(&self.upsert_mutex)
    }

    /// Add an index
    pub fn add_index(&self, name: String, index: Arc<dyn Index>) {
        if let Some(account) = index.memory_account() {
            account.register(&self.memory);
        }
        let mut indexes = self.indexes.write();
        let replaced = indexes.insert(name, index);
        drop(indexes);
        drop(replaced);
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

            // Use composite index if predicate covers a leftmost prefix
            if matched >= 1 {
                // Prefer index with more matching columns
                if best_match.is_none() || matched > best_match.as_ref().unwrap().1 {
                    best_match = Some((index.clone(), matched));
                }
            }
        }

        best_match
    }

    /// Get all indexes as Arc clones - avoids String allocation and repeated lookups
    /// OPTIMIZATION: Arc clones are cheap (atomic increment), uses SmallVec for ≤4 indexes
    #[inline]
    pub fn get_all_indexes(&self) -> SmallVec<[Arc<dyn Index>; 4]> {
        let indexes = self.indexes.read();
        indexes.values().cloned().collect()
    }

    /// Borrow the indexes map under a read lock — no Arc clones, no allocations.
    #[inline]
    pub fn indexes_read(
        &self,
    ) -> parking_lot::MappedRwLockReadGuard<'_, FxHashMap<String, Arc<dyn Index>>> {
        parking_lot::RwLockReadGuard::map(self.indexes.read(), |indexes| &**indexes)
    }

    /// Get column indices and names for all non-PK unique indexes.
    /// Used by commit-time cold revalidation.
    pub fn get_unique_non_pk_index_columns(&self) -> Vec<(Vec<usize>, Vec<String>)> {
        let schema = self.schema();
        let pk_col = schema
            .pk_column_index()
            .map(|i| &schema.columns[i].name_lower);
        let indexes = self.indexes.read();
        let mut result = Vec::new();
        for idx in indexes.values() {
            if !idx.is_unique() {
                continue;
            }
            let names = idx.column_names();
            if names.len() == 1 {
                if let Some(pk) = pk_col {
                    if names[0].eq_ignore_ascii_case(pk) {
                        continue;
                    }
                }
            }
            let col_indices: Vec<usize> = names
                .iter()
                .filter_map(|name| schema.columns.iter().position(|c| c.name_lower == *name))
                .collect();
            if col_indices.len() == names.len() {
                result.push((col_indices, names.to_vec()));
            }
        }
        result
    }

    // =========================================================================
    // Zone Map Operations (Statistics for Segment Pruning)
    // =========================================================================

    /// Sets the zone maps for this table
    ///
    /// Zone maps contain min/max statistics per segment, enabling the query
    /// executor to skip entire segments when predicates fall outside the range.
    pub fn set_zone_maps(&self, mut zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {
        zone_maps.refresh_memory();
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
    pub fn apply_recovered_version(&self, row_id: i64, version: RowVersion) -> Result<(), Error> {
        let is_deleted = version.is_deleted();
        let row_data = version.data.clone();

        // Check for duplicate: if row already exists with identical data, skip adding
        // This prevents duplicate version chain entries when both snapshot and WAL
        // contain the same row data (can happen due to race conditions during snapshot)
        // No need to clone tree for single-row lookup - just hold read guard
        {
            let versions = self.versions.read();
            if let Some(existing_entry) = versions.get(row_id) {
                let existing = &existing_entry.version;
                // Check if data is identical (both deleted status and row data)
                if existing.is_deleted() == is_deleted && existing.data == row_data {
                    // Identical data already exists, skip to avoid duplicate
                    // Still update auto_increment counter
                    if row_id > 0 {
                        self.set_auto_increment_counter(row_id);
                    }
                    return Ok(());
                }
            }
        }

        // Add the version to the store
        self.add_version(row_id, version)?;

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
                if column_ids.len() == 1 {
                    let col_id = column_ids[0] as usize;
                    if let Some(value) = row_data.get(col_id) {
                        let _ = index.add(std::slice::from_ref(value), row_id, row_id);
                    }
                } else {
                    let values: Vec<crate::core::Value> = column_ids
                        .iter()
                        .map(|&col_id| {
                            row_data
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    let _ = index.add(&values, row_id, row_id);
                }
            }
        }
        Ok(())
    }

    /// Mark a row as deleted during WAL replay
    ///
    /// This creates a deleted version for the row during recovery.
    /// Also removes the row from any existing indexes.
    pub fn mark_deleted(&self, row_id: i64, txn_id: i64) -> Result<(), Error> {
        // Get the old row data for index removal BEFORE creating the deleted version
        let old_row = self
            .get_visible_version(row_id, txn_id)
            .map(|v| v.data.clone());

        // Create a deleted version (empty data with deleted flag)
        let deleted_version = RowVersion {
            txn_id,
            deleted_at_txn_id: txn_id,
            data: Row::new(),
            create_time: crate::common::time_compat::SystemTime::now()
                .duration_since(crate::common::time_compat::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0),
        };
        self.add_version(row_id, deleted_version)?;

        // Remove from indexes using old row data
        if let Some(old_data) = old_row {
            let indexes = self.indexes.read();
            for index in indexes.values() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }
                if column_ids.len() == 1 {
                    let col_id = column_ids[0] as usize;
                    if let Some(value) = old_data.get(col_id) {
                        let _ = index.remove(std::slice::from_ref(value), row_id, row_id);
                    }
                } else {
                    let values: Vec<crate::core::Value> = column_ids
                        .iter()
                        .map(|&col_id| {
                            old_data
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    let _ = index.remove(&values, row_id, row_id);
                }
            }
        }
        Ok(())
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
        self.create_index_from_metadata_with_graph(meta, skip_population, None)
    }

    /// Create an index from persisted metadata, optionally loading HNSW graph from disk.
    /// `hnsw_graph_path` is the full path to the HNSW graph file (timestamped to match the
    /// loaded data snapshot, ensuring consistency during fallback recovery).
    pub fn create_index_from_metadata_with_graph(
        &self,
        meta: &crate::storage::mvcc::persistence::IndexMetadata,
        skip_population: bool,
        hnsw_graph_path: Option<&std::path::Path>,
    ) -> crate::core::Result<()> {
        use crate::core::IndexType;
        use crate::storage::index::{BitmapIndex, HashIndex};

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

        // Skip if a PkIndex already covers this single column
        if meta.column_names.len() == 1 {
            if let Some(existing) = self.get_index_by_column(&meta.column_names[0]) {
                if existing.index_type() == IndexType::PrimaryKey {
                    return Ok(()); // PK column covered by PkIndex
                }
            }
        }

        // Get row count for capacity hint
        // No need to clone tree just for len()
        let expected_rows = self.versions.read().len();

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
                        expected_rows,
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
                        expected_rows,
                    );
                    Arc::new(idx)
                }
                IndexType::BTree => {
                    // BTree uses BTreeIndex implementation
                    let idx = crate::storage::index::BTreeIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        column_id,
                        column_name.clone(),
                        data_type,
                        meta.is_unique,
                        expected_rows,
                    );
                    Arc::new(idx)
                }
                IndexType::MultiColumn => {
                    // MultiColumn uses MultiColumnIndex implementation
                    let idx = crate::storage::index::MultiColumnIndex::new(
                        meta.name.clone(),
                        meta.table_name.clone(),
                        meta.column_names.clone(),
                        meta.column_ids.clone(),
                        meta.data_types.clone(),
                        meta.is_unique,
                        expected_rows,
                    );
                    Arc::new(idx)
                }
                IndexType::PrimaryKey => {
                    // PrimaryKey indexes are auto-created, never persisted via CREATE INDEX
                    return Ok(());
                }
                IndexType::Hnsw => {
                    // Get vector dimensions from schema
                    let schema = self.schema();
                    let dims = schema
                        .find_column(column_name)
                        .map(|(_, col)| col.vector_dimensions as usize)
                        .unwrap_or(0);
                    if dims == 0 {
                        return Ok(()); // Cannot rebuild without dimension info
                    }
                    let m = meta
                        .hnsw_m
                        .map(|v| v as usize)
                        .unwrap_or_else(|| crate::storage::index::default_m_for_dims(dims));
                    let ef_construction = meta
                        .hnsw_ef_construction
                        .map(|v| v as usize)
                        .unwrap_or_else(|| crate::storage::index::default_ef_construction(m));
                    let ef_search = meta
                        .hnsw_ef_search
                        .map(|v| v as usize)
                        .unwrap_or_else(|| crate::storage::index::default_ef_search(m));

                    // Try loading saved HNSW graph from snapshot file
                    if let Some(graph_path) = hnsw_graph_path {
                        if let Ok(Some(mut loaded)) = crate::storage::index::HnswIndex::load_graph(
                            graph_path,
                            meta.name.clone(),
                            meta.table_name.clone(),
                            column_name.clone(),
                            column_id,
                            dims,
                            m,
                            ef_construction,
                            ef_search,
                        ) {
                            loaded.set_unique(meta.is_unique);
                            Arc::new(loaded)
                        } else {
                            let mut idx = crate::storage::index::HnswIndex::new(
                                meta.name.clone(),
                                meta.table_name.clone(),
                                column_name.clone(),
                                column_id,
                                dims,
                                m,
                                ef_construction,
                                ef_search,
                                crate::storage::index::HnswDistanceMetric::from_u8(
                                    meta.hnsw_distance_metric.unwrap_or(0),
                                )
                                .unwrap_or(crate::storage::index::HnswDistanceMetric::L2),
                            );
                            idx.set_unique(meta.is_unique);
                            Arc::new(idx)
                        }
                    } else {
                        let mut idx = crate::storage::index::HnswIndex::new(
                            meta.name.clone(),
                            meta.table_name.clone(),
                            column_name.clone(),
                            column_id,
                            dims,
                            m,
                            ef_construction,
                            ef_search,
                            crate::storage::index::HnswDistanceMetric::from_u8(
                                meta.hnsw_distance_metric.unwrap_or(0),
                            )
                            .unwrap_or(crate::storage::index::HnswDistanceMetric::L2),
                        );
                        idx.set_unique(meta.is_unique);
                        Arc::new(idx)
                    }
                }
            };

            // Populate the index with existing data unless deferred
            // Uses batch_slice for better performance
            if !skip_population {
                let col_idx = column_id as usize;
                let versions = self.capture_versions();
                let mut entries: Vec<(i64, Vec<crate::core::Value>)> = Vec::new();
                for (&row_id, version_chain) in versions.iter() {
                    let version = &version_chain.version;
                    if !version.is_deleted() {
                        if let Some(value) = version.data.get(col_idx) {
                            entries.push((row_id, vec![value.clone()]));
                        }
                    }
                }
                if !entries.is_empty() {
                    let entry_refs: Vec<(i64, &[crate::core::Value])> = entries
                        .iter()
                        .map(|(row_id, values)| (*row_id, values.as_slice()))
                        .collect();
                    let _ = index.add_batch_slice(&entry_refs);
                }
            }

            self.add_index(meta.name.clone(), index);
        } else {
            // Multi-column index: use MultiColumnIndex
            let index = crate::storage::index::MultiColumnIndex::new(
                meta.name.clone(),
                meta.table_name.clone(),
                meta.column_names.clone(),
                meta.column_ids.clone(),
                meta.data_types.clone(),
                meta.is_unique,
                expected_rows,
            );

            let index = Arc::new(index);

            // Populate the index with existing data unless deferred
            // Uses batch_slice for better performance
            if !skip_population {
                let col_indices: Vec<usize> =
                    meta.column_ids.iter().map(|&id| id as usize).collect();
                let versions = self.capture_versions();
                let mut entries: Vec<(i64, Vec<crate::core::Value>)> = Vec::new();
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
                        entries.push((row_id, values));
                    }
                }
                if !entries.is_empty() {
                    let entry_refs: Vec<(i64, &[crate::core::Value])> = entries
                        .iter()
                        .map(|(row_id, values)| (*row_id, values.as_slice()))
                        .collect();
                    let _ = index.add_batch_slice(&entry_refs);
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
    /// OPTIMIZATION: Uses batch_slice operations to reduce lock acquisitions
    /// from O(rows × indexes) to O(indexes).
    ///
    /// Call this after WAL replay completes with skip_population=true.
    pub fn populate_all_indexes(&self) {
        let indexes = self.indexes.read();
        if indexes.is_empty() {
            return;
        }

        // Collect index info: (column_ids as Vec<usize>, index_arc)
        // Supports both single-column and multi-column indexes
        // HNSW indexes loaded from graph are included — HnswInner::insert skips duplicate row_ids
        let index_infos: Vec<(Vec<usize>, Arc<dyn Index>)> = indexes
            .values()
            .filter_map(|idx| {
                let col_ids = idx.column_ids();
                if col_ids.is_empty() {
                    return None;
                }
                let col_indices: Vec<usize> = col_ids.iter().map(|&id| id as usize).collect();
                Some((col_indices, Arc::clone(idx)))
            })
            .collect();

        drop(indexes); // Release lock before iteration

        if index_infos.is_empty() {
            return;
        }

        // Pre-allocate per-index batch vectors
        let num_indexes = index_infos.len();
        let mut batches: Vec<Vec<(i64, Vec<crate::core::Value>)>> =
            (0..num_indexes).map(|_| Vec::new()).collect();

        // First pass: Collect all entries per index
        let versions = self.capture_versions();
        for (&row_id, version_chain) in versions.iter() {
            let version = &version_chain.version;

            if version.is_deleted() {
                continue;
            }

            // Collect entries for each index
            for (idx, (col_indices, _)) in index_infos.iter().enumerate() {
                if col_indices.len() == 1 {
                    // Single-column index
                    if let Some(value) = version.data.get(col_indices[0]) {
                        batches[idx].push((row_id, vec![value.clone()]));
                    }
                } else {
                    // Multi-column index
                    let values: Vec<crate::core::Value> =
                        col_indices
                            .iter()
                            .map(|&col_idx| {
                                version.data.get(col_idx).cloned().unwrap_or(
                                    crate::core::Value::Null(crate::core::DataType::Null),
                                )
                            })
                            .collect();
                    batches[idx].push((row_id, values));
                }
            }
        }

        // Second pass: Apply batch operations per index
        // This reduces lock acquisitions from O(rows × indexes) to O(indexes)
        for (idx, (_, index)) in index_infos.iter().enumerate() {
            if !batches[idx].is_empty() {
                let entry_refs: Vec<(i64, &[crate::core::Value])> = batches[idx]
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();
                let _ = index.add_batch_slice(&entry_refs);
            }
        }
    }

    /// Populate HNSW indexes from external rows (e.g., cold segment data).
    ///
    /// Regular indexes (B-tree, Hash, Bitmap) are hot-only by design and use
    /// zone maps/bloom filters for cold data. HNSW indexes are different:
    /// vector similarity search cannot fall back to zone maps, so HNSW must
    /// contain all rows (hot + cold) to return correct results.
    ///
    /// HnswInner::insert skips duplicate row_ids, so calling this after
    /// populate_all_indexes() is safe (hot rows already in the index).
    pub fn populate_hnsw_from_rows(&self, rows: &[(i64, crate::core::Row)]) {
        let indexes = self.indexes.read();
        if indexes.is_empty() || rows.is_empty() {
            return;
        }

        // Collect only HNSW indexes with their column indices
        let hnsw_infos: Vec<(Vec<usize>, Arc<dyn Index>)> = indexes
            .values()
            .filter(|idx| idx.index_type() == crate::core::IndexType::Hnsw)
            .filter_map(|idx| {
                let col_ids = idx.column_ids();
                if col_ids.is_empty() {
                    return None;
                }
                let col_indices: Vec<usize> = col_ids.iter().map(|&id| id as usize).collect();
                Some((col_indices, Arc::clone(idx)))
            })
            .collect();

        drop(indexes);

        if hnsw_infos.is_empty() {
            return;
        }

        // Pre-allocate per-index batch vectors
        let mut batches: Vec<Vec<(i64, Vec<crate::core::Value>)>> = (0..hnsw_infos.len())
            .map(|_| Vec::with_capacity(rows.len()))
            .collect();

        for &(row_id, ref row) in rows {
            for (idx, (col_indices, _)) in hnsw_infos.iter().enumerate() {
                if col_indices.len() == 1 {
                    if let Some(value) = row.get(col_indices[0]) {
                        batches[idx].push((row_id, vec![value.clone()]));
                    }
                } else {
                    let values: Vec<crate::core::Value> = col_indices
                        .iter()
                        .map(|&col_idx| {
                            row.get(col_idx)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    batches[idx].push((row_id, values));
                }
            }
        }

        for (idx, (_, index)) in hnsw_infos.iter().enumerate() {
            if !batches[idx].is_empty() {
                let entry_refs: Vec<(i64, &[crate::core::Value])> = batches[idx]
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();
                let _ = index.add_batch_slice(&entry_refs);
            }
        }
    }

    // =========================================================================
    // Cleanup Functions
    // =========================================================================

    /// Remove sealed rows from the hot version store (phase 1 of seal).
    /// Removes version data and arena slots but KEEPS hot index entries.
    /// The stale index entries act as a safety net: unique constraint checks
    /// still find them, preventing duplicate inserts during the seal window
    /// when the row has moved to cold but might not yet be visible to a
    /// cold check that took a snapshot before register_volume.
    ///
    /// Callers MUST also call `subtract_committed_row_count(n)` with the
    /// returned count to keep the committed row count accurate.
    ///
    /// `extraction_snapshot` is an O(1) CowBTree clone taken at the moment
    /// rows were extracted. For each row_id, the removal compares the
    /// current head version's `txn_id` against the extraction snapshot's
    /// `txn_id`. If they differ, a concurrent commit published a newer
    /// version after extraction — that row is skipped (stays in hot,
    /// sealed next cycle). This prevents discarding concurrent updates
    /// that the sealed volume doesn't contain.
    /// Returns `(removed_count, index_cleanup, skipped_row_ids)`.
    /// Skipped row_ids are rows that were modified after extraction — the
    /// caller must tombstone them so recovery and row_count are correct.
    pub fn remove_sealed_rows(
        &self,
        row_ids: &[i64],
        extraction_snapshot: &ExtractionSnapshot,
        retirement: &mut ArenaRetirement,
    ) -> (usize, SealedIndexCleanup, Vec<i64>) {
        if row_ids.is_empty() {
            return (0, SealedIndexCleanup::default(), Vec::new());
        }

        // Remove rows in small sub-batches to reduce write lock hold time.
        // Each sub-batch acquires versions.write() briefly, then releases it,
        // giving concurrent commits a chance to proceed between sub-batches.
        // Without this, a 50K-row seal batch holds the write lock for the
        // entire removal, blocking all commits on this table for ~100ms+.
        const SUB_BATCH_SIZE: usize = 2_000;
        let mut removed_ids: Vec<i64> = Vec::with_capacity(row_ids.len());
        let mut skipped_ids: Vec<i64> = Vec::new();
        let mut arena_indices_to_clear: Vec<ArenaSlot> = Vec::new();

        for chunk in row_ids.chunks(SUB_BATCH_SIZE) {
            let uncommitted = self.uncommitted_writes.read();
            let mut versions = self.versions.write();
            if self.arena.has_reserved_heads() {
                skipped_ids.extend_from_slice(chunk);
                continue;
            }
            for &row_id in chunk {
                if uncommitted.contains_key(row_id) {
                    skipped_ids.push(row_id);
                    continue;
                }
                if let Some(entry) = versions.get(row_id) {
                    // Compare current txn_id against extraction-time txn_id.
                    // If they differ, a concurrent commit changed this row
                    // after we extracted it — the sealed volume has stale data
                    // for this row. Keep the newer version in hot.
                    let extracted_txn_id = extraction_snapshot
                        .inner
                        .get(row_id)
                        .map(|e| e.version.txn_id)
                        .unwrap_or(0);
                    if entry.version.txn_id != extracted_txn_id {
                        skipped_ids.push(row_id);
                        continue;
                    }
                    if let Some(idx) = entry.arena_idx {
                        arena_indices_to_clear.push(idx);
                    }
                    let removed_payloads = entry.payloads();
                    versions.remove(row_id);
                    versions.payloads.remove(removed_payloads);
                    removed_ids.push(row_id);
                }
            }
            versions.publish_memory(&self.memory);
            // Locks released here — concurrent commits can proceed
        }

        // Invalidate sealed arena slots so speculative probes don't return
        // stale data. This sets row_id=0 in the meta, making the probe fail.
        if !arena_indices_to_clear.is_empty() {
            self.arena.clear_batch(&arena_indices_to_clear, retirement);
        }

        let count = removed_ids.len();
        (count, SealedIndexCleanup { removed_ids }, skipped_ids)
    }

    /// Remove stale hot index entries for sealed rows (phase 2 of seal).
    /// Called while the table's seal fence is still held so INSERT cannot race
    /// between cold constraint checks and hot-index cleanup.
    /// Indexes that keep a row-to-key map remove by row id; the others get
    /// their values from `rows`, the sealed rows.
    pub fn remove_sealed_index_entries(&self, cleanup: SealedIndexCleanup, rows: &RowVec) {
        if cleanup.removed_ids.is_empty() {
            return;
        }

        let indexes = self.indexes.read();
        let hot_only_indexes: Vec<_> = indexes
            .values()
            .filter(|idx| idx.index_type() != crate::core::IndexType::Hnsw)
            .cloned()
            .collect();
        drop(indexes);

        if hot_only_indexes.is_empty() {
            return;
        }

        let mut by_id: Option<I64Map<usize>> = None;
        for index in &hot_only_indexes {
            if index.remove_batch_ids(&cleanup.removed_ids).is_some() {
                continue;
            }
            let positions = by_id.get_or_insert_with(|| {
                let mut map = I64Map::with_capacity(rows.len());
                for (pos, (row_id, _)) in rows.iter().enumerate() {
                    map.insert(*row_id, pos);
                }
                map
            });
            let col_ids = index.column_ids();
            let owned_entries: Vec<(i64, Vec<crate::core::Value>)> = cleanup
                .removed_ids
                .iter()
                .filter_map(|&row_id| {
                    let pos = *positions.get(row_id)?;
                    let row = &rows[pos].1;
                    let values: Vec<crate::core::Value> = col_ids
                        .iter()
                        .map(|&col_id| {
                            row.get(col_id as usize)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    Some((row_id, values))
                })
                .collect();
            if owned_entries.is_empty() {
                continue;
            }
            let borrowed_entries: Vec<(i64, &[crate::core::Value])> = owned_entries
                .iter()
                .map(|(row_id, values)| (*row_id, values.as_slice()))
                .collect();
            let _ = index.remove_batch_slice(&borrowed_entries);
        }
    }

    /// Cleanup deleted rows that are older than the retention period.
    ///
    /// This removes soft-deleted rows that are no longer visible to any active
    /// transaction and are older than the specified retention period.
    /// Also clears the arena slots to release memory.
    pub fn cleanup_deleted_rows(&self, retention_period: std::time::Duration) -> i32 {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let now = get_fast_timestamp();
        let cutoff_time = now - retention_period.as_nanos() as i64;

        let mut rows_to_delete = Vec::new();

        // Clone CowBTree once, reuse for index cleanup (O(1) Arc clone)
        let versions = self.capture_versions();

        // First pass: identify deleted rows older than retention period
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

        if rows_to_delete.is_empty() {
            return 0;
        }

        // Second pass: acquire write lock and re-validate before removing.
        // Between the snapshot (first pass) and now, a concurrent transaction may have
        // committed a new version for a row_id we marked for deletion (e.g., re-INSERT
        // with the same PK value). We must re-check that each row is still deleted
        // before removing it to prevent data loss.
        // Arena indices are read directly from the live entry under the write lock
        // to avoid index misalignment with the rows_to_delete vector.
        let mut actually_deleted = Vec::with_capacity(rows_to_delete.len());
        let mut actual_arena_indices = Vec::with_capacity(rows_to_delete.len());
        {
            let mut versions = self.versions.write();
            if self.arena.has_reserved_heads() {
                return 0;
            }
            for &row_id in &rows_to_delete {
                // Re-check: the row must still exist AND still be deleted
                if let Some(entry) = versions.get(row_id) {
                    if entry.version.is_deleted() {
                        if let Some(idx) = entry.arena_idx {
                            actual_arena_indices.push(idx);
                        }
                        let removed_payloads = entry.payloads();
                        versions.remove(row_id);
                        versions.payloads.remove(removed_payloads);
                        actually_deleted.push(row_id);
                    }
                }
            }
            versions.publish_memory(&self.memory);
        }

        if actually_deleted.is_empty() {
            return 0;
        }

        // Third pass: remove from indexes using batch operations (single lock per index)
        // This runs AFTER the version store removal so we use the snapshot data
        // (which is still valid for the rows we confirmed were deleted).
        {
            let indexes = self.indexes.read();

            for index in indexes.values() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }

                // Collect all entries for this index
                let mut entries: Vec<(i64, Vec<crate::core::Value>)> =
                    Vec::with_capacity(actually_deleted.len());

                for &row_id in &actually_deleted {
                    if let Some(entry) = versions.get(row_id) {
                        let version = &entry.version;
                        if column_ids.len() == 1 {
                            // Single-column index
                            let col_id = column_ids[0] as usize;
                            if let Some(value) = version.data.get(col_id) {
                                entries.push((row_id, vec![value.clone()]));
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
                            entries.push((row_id, values));
                        }
                    }
                }

                if !entries.is_empty() {
                    // Convert to slice format for remove_batch_slice
                    let batch: Vec<(i64, &[crate::core::Value])> = entries
                        .iter()
                        .map(|(row_id, values)| (*row_id, values.as_slice()))
                        .collect();
                    let _ = index.remove_batch_slice(&batch);
                }

                // Let index-specific maintenance run (e.g., HNSW graph compaction)
                let _ = index.cleanup();
            }
        }

        // Clear arena slots only for rows we actually removed
        let mut retirement = self.arena.prepare_clear(actual_arena_indices.len());
        self.arena
            .clear_batch(&actual_arena_indices, &mut retirement);
        self.arena.finish_clear(&mut retirement);

        actually_deleted.len() as i32
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
        // Default 24-hour retention for background cleanup
        self.cleanup_old_previous_versions_with_retention(std::time::Duration::from_secs(
            24 * 60 * 60,
        ))
    }

    pub fn cleanup_old_previous_versions_with_retention(
        &self,
        retention_period: std::time::Duration,
    ) -> i32 {
        if self.closed.load(Ordering::Acquire) {
            return 0;
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return 0, // Need visibility checker for cleanup
        };

        let now = get_fast_timestamp();
        let retention_cutoff = now - retention_period.as_nanos() as i64;

        // Get active transaction IDs
        let active_txns = checker.get_active_transaction_ids();

        // First pass (read lock): identify row_ids that MAY need pruning
        let mut candidate_row_ids: Vec<i64> = Vec::new();
        {
            let versions = self.capture_versions();
            for (&row_id, chain_entry) in versions.iter() {
                // Quick check: does this entry have any prev versions?
                if chain_entry.prev.is_none() {
                    continue;
                }
                candidate_row_ids.push(row_id);
            }
        }

        if candidate_row_ids.is_empty() {
            return 0;
        }

        // Second pass (write lock): re-read each entry from live data and prune in-place.
        // This prevents the race where a concurrent commit adds a new HEAD between
        // the snapshot read and the write — we always work on the current live entry.
        let mut cleaned = 0;
        let mut versions = self.versions.write();

        for row_id in candidate_row_ids {
            let Some(chain_entry) = versions.get(row_id) else {
                continue; // Row was removed between passes
            };

            // Collect previous versions from the LIVE entry
            let mut prev_versions: Vec<Arc<VersionChainEntry>> = Vec::new();
            let mut current = chain_entry.prev.as_ref();
            while let Some(prev_entry) = current {
                prev_versions.push(prev_entry.clone());
                current = prev_entry.prev.as_ref();
            }

            if prev_versions.is_empty() {
                continue;
            }

            // Check each previous version independently — do NOT assume monotonic
            // visibility. With rapid updates, a newer prev version may be invisible
            // to an active txn while an older one IS visible (e.g., HEAD seq=120,
            // prev_0 seq=110, prev_1 seq=80, active txn snapshot at seq=100 needs prev_1).
            let mut keep_count = 0;
            for (i, prev_entry) in prev_versions.iter().enumerate() {
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
                    // Keep this version and all newer ones (indices 0..=i)
                    keep_count = i + 1;
                }
            }

            // If we need to prune some versions, modify the live entry
            if keep_count < prev_versions.len() {
                let to_remove = prev_versions.len() - keep_count;
                cleaned += to_remove as i32;
                let mut removed_payloads = VersionPayloads::default();
                for entry in &prev_versions[keep_count..] {
                    removed_payloads.add_row(&entry.version.data);
                }

                // Clone the LIVE entry (not stale snapshot) and modify
                let mut modified_entry = chain_entry.clone();

                if keep_count == 0 {
                    modified_entry.prev = None;
                } else {
                    // Rebuild chain with only kept versions
                    let kept_versions: Vec<_> =
                        prev_versions.into_iter().take(keep_count).collect();

                    // Build chain from oldest to newest (reversed)
                    let mut new_prev: Option<Arc<VersionChainEntry>> = None;
                    for entry in kept_versions.into_iter().rev() {
                        let mut cloned = (*entry).clone();
                        cloned.prev = new_prev;
                        new_prev = Some(Arc::new(cloned));
                    }
                    modified_entry.prev = new_prev;
                }

                versions.insert(row_id, modified_entry);
                versions.payloads.remove(removed_payloads);
            }
        }
        versions.publish_memory(&self.memory);

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
        let versions = self.capture_versions();
        let mut exports = ExportBatch::new();
        for (&row_id, chain_entry) in versions.iter() {
            // Walk the version chain to find the visible version
            let mut current: Option<&VersionChainEntry> = Some(chain_entry);

            while let Some(e) = current {
                let version_txn_id = e.version.txn_id;
                let deleted_at_txn_id = e.version.deleted_at_txn_id;

                // Check if this version is visible
                if checker.is_visible(version_txn_id, snapshot_txn_id) {
                    // When cutoff is specified, only include versions from transactions
                    // that committed before the cutoff to ensure snapshot consistency.
                    // CRITICAL: Do NOT walk the prev chain. The extraction snapshot
                    // captures the HEAD txn_id, and remove_sealed_rows compares against
                    // it. If we extract an older version but the HEAD txn_id matches,
                    // the HEAD is removed from hot, permanently losing the newer version.
                    // The row stays entirely in hot where MVCC handles visibility.
                    if use_cutoff && !checker.is_committed_before(version_txn_id, commit_seq_cutoff)
                    {
                        break;
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
                    exports.record(&e.version.data);
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

        let versions = self.capture_versions();
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

    /// Get the transaction ID of the latest committed version for a row.
    ///
    /// This retrieves the "head" of the version chain, effectively checking
    /// the most recently committed change. This is critical for conflict
    /// detection (First-Committer-Wins).
    ///
    /// Returns:
    /// - Some(txn_id) if the row exists
    /// - None if the row does not exist
    pub fn get_latest_version_id(&self, row_id: i64) -> Option<i64> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }

        let versions = self.versions.read();
        versions.get(row_id).map(|entry| entry.version.txn_id)
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
    ) -> Option<Vec<GroupedAggregateResult>> {
        if self.closed.load(Ordering::Acquire) {
            return Some(Vec::new());
        }

        let checker = match self.visibility_checker.as_ref() {
            Some(c) => c,
            None => return Some(Vec::new()),
        };

        // Guard: arena-only path requires ReadCommitted and no uncommitted writes.
        // Under snapshot isolation, HEAD versions in the arena may have been committed
        // after the viewer's snapshot. The correct behavior requires walking version
        // chains to find older visible versions, which the arena path doesn't support.
        // Return None to let the caller fall back to regular GROUP BY aggregation.
        let uncommitted_empty = self.uncommitted_writes.read().is_empty();
        if !uncommitted_empty || checker.needs_snapshot_isolation(txn_id) {
            return None;
        }

        // Accumulator for each group: (count, sum, min, max) per aggregate
        #[derive(Clone)]
        struct Accum {
            count: i64,
            int_sum: i128,
            float_sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }

        impl Default for Accum {
            fn default() -> Self {
                Self {
                    count: 0,
                    int_sum: 0,
                    float_sum: 0.0,
                    min: None,
                    max: None,
                }
            }
        }

        // Pre-acquire arena lock ONCE
        let arena_guard = self.arena.read_guard();
        let mut exports = ExportBatch::new();

        // Helper to update accumulators
        #[inline(always)]
        fn update_accums(
            accums: &mut [Accum],
            aggregates: &[(AggregateOp, usize)],
            row_data: &[Value],
        ) {
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
                            match &row_data[*col_idx] {
                                Value::Integer(i) => {
                                    accum.int_sum += *i as i128;
                                    accum.count += 1;
                                }
                                Value::Float(f) => {
                                    accum.float_sum += *f;
                                    accum.count += 1;
                                }
                                // A boolean counts as one or nought
                                Value::Boolean(b) => {
                                    accum.int_sum += *b as i128;
                                    accum.count += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                    AggregateOp::Min => {
                        if *col_idx < row_data.len() {
                            let val = &row_data[*col_idx];
                            if !val.is_null() {
                                match &accum.min {
                                    None => accum.min = Some(val.clone()),
                                    Some(current) => {
                                        if val < current {
                                            accum.min = Some(val.clone());
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
                                    None => accum.max = Some(val.clone()),
                                    Some(current) => {
                                        if val > current {
                                            accum.max = Some(val.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Helper to compute final aggregate values
        fn compute_aggregate_values(
            aggregates: &[(AggregateOp, usize)],
            accums: &[Accum],
            exports: &mut ExportBatch,
        ) -> Vec<Value> {
            aggregates
                .iter()
                .zip(accums.iter())
                .map(|((op, _), accum)| match op {
                    AggregateOp::Count | AggregateOp::CountStar => Value::Integer(accum.count),
                    AggregateOp::Sum => {
                        if accum.count > 0 {
                            Value::Float(accum.int_sum as f64 + accum.float_sum)
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Avg => {
                        if accum.count > 0 {
                            Value::Float(
                                (accum.int_sum as f64 + accum.float_sum) / accum.count as f64,
                            )
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Min => accum
                        .min
                        .as_ref()
                        .map(|v| exports.capture_value(v))
                        .unwrap_or(Value::Null(DataType::Null)),
                    AggregateOp::Max => accum
                        .max
                        .as_ref()
                        .map(|v| exports.capture_value(v))
                        .unwrap_or(Value::Null(DataType::Null)),
                })
                .collect()
        }

        // FAST PATH: Single-column GROUP BY with primitive types (Integer, Float, Boolean)
        // Uses I64Map directly for ~3x faster hashing and comparison
        // Avoids GroupKey allocation entirely for primitive types and NULL
        if group_by_indices.len() == 1 {
            let col_idx = group_by_indices[0];

            // Track key type: 0=Integer, 1=Float, 2=Boolean, 3=Mixed/Other
            let mut key_type: u8 = 255; // uninitialized
                                        // Accumulators live in one flat pool, `n_aggs` per group, and the
                                        // maps hold the group's ordinal into it. A Vec per group would be
                                        // one allocation per distinct group, which for a high-cardinality
                                        // GROUP BY is the dominant cost of the scan.
            let n_aggs = aggregates.len();
            let mut accums: Vec<Accum> = Vec::new();
            let mut group_count: u32 = 0;
            // The ordinal is counted, not derived from the pool length: an
            // empty aggregate list is a valid request and would divide by zero.
            fn new_group(accums: &mut Vec<Accum>, group_count: &mut u32, n_aggs: usize) -> u32 {
                let ordinal = *group_count;
                *group_count += 1;
                accums.resize_with(accums.len() + n_aggs, Accum::default);
                ordinal
            }
            use crate::common::i64_map::Entry;
            let mut int_groups: I64Map<u32> = I64Map::new();
            // Separate NULL accumulator - avoids creating GroupKey for NULL
            let mut null_group: Option<u32> = None;
            // Only used for String/other types that can't be mapped to i64
            let mut other_groups: GroupKeyMap<u32> = GroupKeyMap::default();

            for (meta, row_data) in arena_guard.rows() {
                // Visibility check (standard pattern: check creation, then deletion)
                if meta.txn_id == 0 || !checker.is_visible(meta.txn_id, txn_id) {
                    continue;
                }
                if meta.deleted_at_txn_id != 0 && checker.is_visible(meta.deleted_at_txn_id, txn_id)
                {
                    continue;
                }

                let row_slice = row_data.as_ref();
                let val = if col_idx < row_slice.len() {
                    &row_slice[col_idx]
                } else {
                    // NULL - track separately without GroupKey allocation
                    let ordinal = match null_group {
                        Some(ordinal) => ordinal,
                        None => {
                            *null_group.insert(new_group(&mut accums, &mut group_count, n_aggs))
                        }
                    };
                    let base = ordinal as usize * n_aggs;
                    update_accums(&mut accums[base..base + n_aggs], aggregates, row_slice);
                    continue;
                };

                // Try to extract i64 key for primitive types.
                // Only route to int_groups when the type matches key_type.
                // Mixed-type columns route mismatched types to other_groups
                // to avoid reconstructing Float bits as Integer or vice versa.
                let i64_key = match val {
                    Value::Integer(i) => {
                        if key_type == 255 {
                            key_type = 0;
                        }
                        if key_type == 0 {
                            Some(*i)
                        } else {
                            None // Type mismatch → other_groups
                        }
                    }
                    Value::Float(f) => {
                        if key_type == 255 {
                            key_type = 1;
                        }
                        if key_type == 1 {
                            Some(key_from_f64(*f))
                        } else {
                            None // Type mismatch → other_groups
                        }
                    }
                    Value::Boolean(b) => {
                        if key_type == 255 {
                            key_type = 2;
                        }
                        if key_type == 2 {
                            Some(if *b { 1 } else { 0 })
                        } else {
                            None // Type mismatch → other_groups
                        }
                    }
                    Value::Null(_) => {
                        // NULL value in the column - track separately
                        let ordinal = match null_group {
                            Some(ordinal) => ordinal,
                            None => {
                                *null_group.insert(new_group(&mut accums, &mut group_count, n_aggs))
                            }
                        };
                        let base = ordinal as usize * n_aggs;
                        update_accums(&mut accums[base..base + n_aggs], aggregates, row_slice);
                        continue;
                    }
                    _ => None,
                };

                let ordinal = if let Some(key) = i64_key {
                    match int_groups.entry(key) {
                        Entry::Occupied(existing) => *existing.into_mut(),
                        Entry::Vacant(slot) => {
                            *slot.insert(new_group(&mut accums, &mut group_count, n_aggs))
                        }
                    }
                } else {
                    match other_groups.entry(GroupKey::Single(CompactArc::new(val.clone()))) {
                        std::collections::hash_map::Entry::Occupied(existing) => *existing.get(),
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            *slot.insert(new_group(&mut accums, &mut group_count, n_aggs))
                        }
                    }
                };
                let base = ordinal as usize * n_aggs;
                update_accums(&mut accums[base..base + n_aggs], aggregates, row_slice);
            }

            // Convert to results
            let has_null = null_group.is_some();
            let mut results: Vec<GroupedAggregateResult> = Vec::with_capacity(
                int_groups.len() + other_groups.len() + if has_null { 1 } else { 0 },
            );

            // The caller appends the aggregate values onto the group values to
            // build the output row, so the key vector is sized for both here
            // and that append does not reallocate.
            let row_width = 1 + n_aggs;
            let slice_of = |ordinal: u32| {
                let base = ordinal as usize * n_aggs;
                &accums[base..base + n_aggs]
            };

            // Convert int_groups based on key_type
            for (key, &ordinal) in int_groups.iter() {
                let group_value = match key_type {
                    0 => Value::Integer(key),             // Integer
                    1 => Value::Float(f64_from_key(key)), // Float
                    2 => Value::Boolean(key != 0),        // Boolean
                    _ => Value::Integer(key),             // Fallback
                };
                let mut group_values = Vec::with_capacity(row_width);
                group_values.push(group_value);
                results.push(GroupedAggregateResult {
                    group_values,
                    aggregate_values: compute_aggregate_values(
                        aggregates,
                        slice_of(ordinal),
                        &mut exports,
                    ),
                });
            }

            // Add NULL group if present
            if let Some(ordinal) = null_group {
                let mut group_values = Vec::with_capacity(row_width);
                group_values.push(Value::Null(DataType::Null));
                results.push(GroupedAggregateResult {
                    group_values,
                    aggregate_values: compute_aggregate_values(
                        aggregates,
                        slice_of(ordinal),
                        &mut exports,
                    ),
                });
            }

            // Convert other_groups (strings, etc.), consuming the map so each
            // key is released as its values are moved out rather than being
            // held alongside its clone for the rest of the loop.
            for (group_key, ordinal) in other_groups {
                let mut group_values = Vec::with_capacity(row_width);
                match group_key {
                    GroupKey::Single(v) => group_values.push(exports.capture_value(&v)),
                    GroupKey::Multi(vs) => {
                        group_values.extend(vs.iter().map(|v| exports.capture_value(v)))
                    }
                }
                results.push(GroupedAggregateResult {
                    group_values,
                    aggregate_values: compute_aggregate_values(
                        aggregates,
                        slice_of(ordinal),
                        &mut exports,
                    ),
                });
            }

            return Some(results);
        }

        // SLOW PATH: Multi-column GROUP BY (currently not used from try_storage_aggregation)
        let mut groups: GroupKeyMap<Vec<Accum>> = GroupKeyMap::default();

        for (meta, row_data) in arena_guard.rows() {
            // Visibility check (standard pattern: check creation, then deletion)
            if meta.txn_id == 0 || !checker.is_visible(meta.txn_id, txn_id) {
                continue;
            }
            if meta.deleted_at_txn_id != 0 && checker.is_visible(meta.deleted_at_txn_id, txn_id) {
                continue;
            }

            let row_slice = row_data.as_ref();

            let key_values: Vec<CompactArc<Value>> = group_by_indices
                .iter()
                .map(|&col_idx| {
                    if col_idx < row_slice.len() {
                        CompactArc::new(row_slice[col_idx].clone())
                    } else {
                        CompactArc::new(Value::Null(DataType::Null))
                    }
                })
                .collect();
            let group_key = GroupKey::Multi(key_values);

            let accums = groups
                .entry(group_key)
                .or_insert_with(|| vec![Accum::default(); aggregates.len()]);
            update_accums(accums, aggregates, row_slice);
        }

        // Convert to results
        let mut results: Vec<GroupedAggregateResult> = Vec::with_capacity(groups.len());

        for (group_key, accums) in groups {
            let group_values = match group_key {
                GroupKey::Single(v) => vec![exports.capture_value(&v)],
                GroupKey::Multi(vs) => vs.iter().map(|v| exports.capture_value(v)).collect(),
            };
            results.push(GroupedAggregateResult {
                group_values,
                aggregate_values: compute_aggregate_values(aggregates, &accums, &mut exports),
            });
        }

        Some(results)
    }
}

impl Drop for VersionStore {
    fn drop(&mut self) {
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::hot_owner_dropping();
        drop(std::mem::take(&mut self.versions.get_mut().entries));
        self.memory.version_payloads.store(0, Ordering::Release);
        self.memory.version_tree.store(0, Ordering::Release);
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
    /// The list is ordered by create_time (oldest first, newest last)
    /// Lazily allocated on first write to avoid allocation overhead for read-only queries.
    /// Uses SmallVec<[RowVersion; 1]> to avoid heap allocation for single-version rows.
    local_versions: Option<I64Map<VersionList>>,
    /// Parent (shared) version store
    parent_store: Arc<VersionStore>,
    /// This transaction's ID
    txn_id: i64,
    /// Write set for conflict detection
    /// Lazily allocated on first write to avoid allocation overhead for read-only queries
    write_set: Option<I64Map<WriteSetEntry>>,
    /// Index updates applied by commit, until the commit is visible or undone
    index_undo: Mutex<IndexUndoLog>,
    arena_reservation: Option<ArenaReservation>,
    version_bytes: u128,
    _object_charge: HotObjectCharge<std::sync::RwLock<TransactionVersionStore>>,
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
            index_undo: Mutex::new(IndexUndoLog::default()),
            arena_reservation: None,
            version_bytes: 0,
            _object_charge: HotObjectCharge::new(),
        }
    }

    /// Takes back the index updates of this transaction's commit, in reverse
    pub fn undo_index_updates(&self) {
        let undo = std::mem::take(&mut *self.index_undo.lock());
        for entry in undo.entries.iter().rev() {
            if !entry.added.is_empty() {
                let batch: Vec<(i64, &[Value])> = entry
                    .added
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();
                let _ = entry.index.remove_batch_slice(&batch);
            }
            if !entry.removed.is_empty() {
                let batch: Vec<(i64, &[Value])> = entry
                    .removed
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();
                let _ = entry.index.add_batch_slice(&batch);
            }
        }
    }

    /// Returns the transaction ID
    pub fn txn_id(&self) -> i64 {
        self.txn_id
    }

    /// Read-only access to local versions for rollback cleanup.
    pub fn local_versions_ref(&self) -> Option<&I64Map<VersionList>> {
        self.local_versions.as_ref()
    }

    /// Read-only access to write set for commit-time cold revalidation.
    pub fn write_set_ref(&self) -> Option<&I64Map<WriteSetEntry>> {
        self.write_set.as_ref()
    }

    /// Ensures local_versions map is allocated, returning a mutable reference.
    /// Uses pooled maps when available to reduce allocation overhead.
    #[inline]
    fn ensure_local_versions(&mut self) -> &mut I64Map<VersionList> {
        self.local_versions.get_or_insert_with(get_version_list_map)
    }

    /// Ensures write_set map is allocated, returning a mutable reference.
    /// Uses pooled maps when available to reduce allocation overhead.
    #[inline]
    fn ensure_write_set(&mut self) -> &mut I64Map<WriteSetEntry> {
        self.write_set.get_or_insert_with(get_write_set_map)
    }

    fn charge_versions(&mut self, bytes: u128) {
        self.version_bytes += bytes;
        self.parent_store.memory.transaction_versions.add(bytes);
    }

    fn release_versions(&mut self, bytes: u128) {
        self.version_bytes -= bytes;
        self.parent_store.memory.transaction_versions.remove(bytes);
    }

    fn push_local_version(versions: &mut VersionList, version: RowVersion) -> u128 {
        let bytes = version.data.heap_bytes();
        let before = if versions.spilled() {
            versions.capacity()
        } else {
            0
        };
        versions.push(version);
        let after = if versions.spilled() {
            versions.capacity()
        } else {
            0
        };
        bytes + ((after - before) * std::mem::size_of::<RowVersion>()) as u128
    }

    fn append_local_version(&mut self, row_id: i64, version: RowVersion) {
        let map = self.ensure_local_versions();
        let before = map.allocation_bytes();
        let versions = map.entry(row_id).or_default();
        let bytes = Self::push_local_version(versions, version);
        account_map_capacity(before, map);
        self.charge_versions(bytes);
    }

    fn record_read_version(&mut self, row_id: i64, entry: WriteSetEntry) {
        let bytes = entry
            .read_version
            .as_ref()
            .map_or(0, |v| v.data.heap_bytes());
        let map = self.ensure_write_set();
        let before = map.allocation_bytes();
        let previous = map.insert(row_id, entry);
        account_map_capacity(before, map);
        debug_assert!(previous.is_none());
        self.charge_versions(bytes);
    }

    /// Put adds or updates a row in the transaction's local store
    pub fn put(&mut self, row_id: i64, data: Row, is_delete: bool) -> Result<(), Error> {
        // Convert to Shared (Arc) storage immediately for efficient Arc sharing:
        // - get_arc() will return cheap Arc clones (no value cloning)
        // - into_arc() at commit time returns the existing Arc (no clone)
        let data = Row::from_arc(data.into_arc());

        // Get timestamp once at the start (avoids calling SystemTime::now() inside RowVersion::new)
        let timestamp = get_fast_timestamp();

        // Create the row version with pre-computed timestamp
        let mut rv = RowVersion::new_with_timestamp(self.txn_id, data, timestamp);
        if is_delete {
            rv.deleted_at_txn_id = self.txn_id;
        }

        // Check if we already have a local version for this row
        let has_local = self
            .local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(row_id));

        if !has_local {
            // New row - need to check write-set and parent store
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(row_id));

            if needs_write_set_entry {
                let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                let row_exists = read_version.is_some();

                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.record_read_version(
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
        self.append_local_version(row_id, rv);
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
        // Convert to Shared (Arc) storage immediately for efficient Arc sharing
        let data = Row::from_arc(data.into_arc());

        // Get timestamp once at the start (avoids calling SystemTime::now() inside RowVersion::new)
        let timestamp = get_fast_timestamp();

        // Create the new row version with pre-computed timestamp
        let mut rv = RowVersion::new_with_timestamp(self.txn_id, data, timestamp);
        if is_delete {
            rv.deleted_at_txn_id = self.txn_id;
        }

        // Check if we already have a local version for this row
        let has_local = self
            .local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(row_id));

        if !has_local {
            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(row_id));

            if needs_write_set_entry {
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.record_read_version(
                    row_id,
                    WriteSetEntry {
                        read_version: Some(original_version),
                        read_version_seq,
                    },
                );

                // Claim the row for update
                self.parent_store.try_claim_row(row_id, self.txn_id)?;
            }
        }
        self.append_local_version(row_id, rv);
        Ok(())
    }

    /// Optimized batch put for UPDATE operations with pre-fetched original versions
    ///
    /// This avoids redundant get_visible_version() calls by accepting the original
    /// versions that were already fetched during the batch read.
    ///
    /// Parameters:
    /// - rows: Vec of (row_id, new_row_data, original_version)
    ///   Note: read_version_seq is obtained from visibility_checker, not from create_time
    pub fn put_batch_with_originals(
        &mut self,
        rows: Vec<(i64, Row, RowVersion)>,
    ) -> Result<(), Error> {
        let now = get_fast_timestamp();

        for (row_id, data, original_version) in rows {
            // Convert to Shared (Arc) storage immediately for efficient Arc sharing
            let data = Row::from_arc(data.into_arc());

            // Create the new row version with pre-computed timestamp (avoids wasteful
            // get_fast_timestamp() call inside RowVersion::new that would be overwritten)
            let rv = RowVersion::new_with_timestamp(self.txn_id, data, now);

            // Check if already in local versions (already processed in this transaction)
            if let Some(versions) = self.ensure_local_versions().get_mut(row_id) {
                // Append new version to history
                let bytes = Self::push_local_version(versions, rv);
                self.charge_versions(bytes);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(row_id));

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
                self.record_read_version(
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
            self.append_local_version(row_id, rv);
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
        // Get timestamp once for all rows in the batch
        let timestamp = get_fast_timestamp();

        for (row_id, data) in rows {
            // Check if we already have a local version
            let has_local = self
                .local_versions
                .as_ref()
                .is_some_and(|lv| lv.contains_key(row_id));

            if !has_local {
                // Check if this row exists in parent store and track in write-set
                let needs_write_set_entry = self
                    .write_set
                    .as_ref()
                    .is_none_or(|ws| !ws.contains_key(row_id));

                if needs_write_set_entry {
                    let read_version = self.parent_store.get_visible_version(row_id, self.txn_id);
                    let row_exists = read_version.is_some();

                    let read_version_seq = self
                        .parent_store
                        .visibility_checker
                        .as_ref()
                        .map(|c| c.get_current_sequence())
                        .unwrap_or(0);

                    self.record_read_version(
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

            // Create deleted row version with pre-computed timestamp
            let mut rv = RowVersion::new_with_timestamp(self.txn_id, data, timestamp);
            rv.deleted_at_txn_id = self.txn_id;

            // Append to version history for this row
            self.append_local_version(row_id, rv);
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
        // Get timestamp once for all rows in the batch
        let timestamp = get_fast_timestamp();

        for (row_id, data, original_version) in rows {
            // Create deleted row version with pre-computed timestamp
            let mut rv = RowVersion::new_with_timestamp(self.txn_id, data, timestamp);
            rv.deleted_at_txn_id = self.txn_id;

            // Check if already in local versions (already processed in this transaction)
            if let Some(versions) = self.ensure_local_versions().get_mut(row_id) {
                let bytes = Self::push_local_version(versions, rv);
                self.charge_versions(bytes);
                continue;
            }

            // Track in write-set using the pre-fetched original version
            let needs_write_set_entry = self
                .write_set
                .as_ref()
                .is_none_or(|ws| !ws.contains_key(row_id));

            if needs_write_set_entry {
                let read_version_seq = self
                    .parent_store
                    .visibility_checker
                    .as_ref()
                    .map(|c| c.get_current_sequence())
                    .unwrap_or(0);

                self.record_read_version(
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
            self.append_local_version(row_id, rv);
        }
        Ok(())
    }

    /// Check if we have local changes for a row
    pub fn has_locally_seen(&self, row_id: i64) -> bool {
        self.local_versions
            .as_ref()
            .is_some_and(|lv| lv.contains_key(row_id))
    }

    /// True when the transaction claimed rows that already exist in the hot
    /// store: a seal skips those, so waiting for it cannot free them
    pub fn holds_hot_rows(&self) -> bool {
        self.write_set
            .as_ref()
            .is_some_and(|ws| ws.values().any(|e| e.read_version.is_some()))
    }

    /// Returns true if this transaction has any uncommitted local changes
    pub fn has_local_changes(&self) -> bool {
        self.local_versions
            .as_ref()
            .is_some_and(|lv| !lv.is_empty())
    }

    /// Returns the number of local changes (distinct row IDs)
    pub fn local_count(&self) -> usize {
        self.local_versions.as_ref().map_or(0, |lv| lv.len())
    }

    /// Get the latest local version for a specific row, if any.
    #[inline]
    pub fn get_latest_local(&self, row_id: i64) -> Option<&RowVersion> {
        self.local_versions
            .as_ref()
            .and_then(|lv| lv.get(row_id))
            .and_then(|versions| versions.last())
    }

    /// Iterate over local versions (returns most recent version per row)
    pub fn iter_local(&self) -> impl Iterator<Item = (i64, &RowVersion)> {
        self.local_versions
            .iter()
            .flat_map(|lv| lv.iter())
            .filter_map(|(k, versions)| versions.last().map(|v| (k, v)))
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
                    (row_id, version, old_row)
                })
            })
    }

    /// Get the local version for a row (without checking parent)
    /// Returns the most recent version in the transaction's history
    pub fn get_local_version(&self, row_id: i64) -> Option<&RowVersion> {
        self.local_versions
            .as_ref()
            .and_then(|lv| lv.get(row_id))
            .and_then(|versions| versions.last())
    }

    /// Get a row, checking local versions first then parent store
    pub fn get(&self, row_id: i64) -> Option<Row> {
        // Check local versions first (get most recent)
        if let Some(lv) = self.local_versions.as_ref() {
            if let Some(versions) = lv.get(row_id) {
                if let Some(local_version) = versions.last() {
                    if local_version.is_deleted() {
                        return None;
                    }
                    charge_export(&local_version.data);
                    return Some(local_version.data.clone());
                }
            }
        }

        // Check parent store
        self.parent_store
            .get_visible_version(row_id, self.txn_id)
            .map(|v| v.data.clone())
    }

    /// OCC validation that tolerates seal-removed rows.
    ///
    /// If a row is missing from the hot B-tree, it was moved to cold by seal.
    /// This is not a conflict for READ COMMITTED transactions. The transaction
    /// read the row data into its local txn_versions before seal removed it.
    /// On commit, the updated version is re-inserted into hot, and the
    /// skip-set dedup mechanism ensures the stale cold version is shadowed.
    /// Seal also skips rows claimed by active transactions (uncommitted_writes).
    ///
    /// For INSERT paths (read_version is None), we still check whether another
    /// transaction concurrently inserted the same row_id to prevent lost writes
    /// when explicit PK values are used.
    pub fn detect_conflicts_safe(&self) -> Result<(), Error> {
        let Some(write_set) = self.write_set.as_ref() else {
            return Ok(());
        };

        for (row_id, write_entry) in write_set.iter() {
            if let Some(read_version) = &write_entry.read_version {
                // UPDATE path: check that the row hasn't been modified concurrently
                match self.parent_store.get_latest_version_id(row_id) {
                    Some(latest_txn_id) if latest_txn_id != read_version.txn_id => {
                        return Err(Error::internal(format!(
                            "write conflict: row {} was modified by another transaction",
                            row_id
                        )));
                    }
                    Some(_) => {}
                    None => {
                        // Row is absent from the hot B-tree. This happens when seal
                        // moved the row to cold storage between our read and commit.
                        // This is NOT a real conflict — the row data is unchanged,
                        // just relocated. Skip the conflict check for this row.
                        // The commit will add the updated version to hot, and the
                        // skip-set mechanism at scan time handles dedup with cold.
                    }
                }
            } else {
                // INSERT path: check that no other transaction concurrently inserted
                // the same row_id. This guards against TOCTOU races with explicit PK values.
                // For auto-increment IDs this is effectively a no-op (AtomicI64 guarantees
                // uniqueness), but it is the safety net for user-supplied PKs.
                if self
                    .parent_store
                    .has_any_visible_version(&[row_id], self.txn_id)
                    .is_some()
                {
                    return Err(Error::internal(format!(
                        "write conflict: row {} was concurrently inserted by another transaction",
                        row_id
                    )));
                }
            }
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
                versions.push((row_id, version.clone()));
            }
        }
        versions
    }

    /// Reserve every possible new slot before any table starts publication.
    pub fn reserve_commit_capacity(&mut self) -> Result<(), Error> {
        if self.arena_reservation.is_some() {
            return Ok(());
        }
        let Some(local) = self.local_versions.as_ref() else {
            return Ok(());
        };
        let writes = local
            .values()
            .filter(|history| history.last().is_some_and(|v| !v.is_deleted()))
            .count();
        if writes > 0 && self.holds_hot_rows() {
            let versions = self.parent_store.versions.read();
            let reuse = local.iter().all(|(id, history)| {
                history.last().is_none_or(|v| v.is_deleted())
                    || versions
                        .get(id)
                        .is_some_and(|head| head.arena_idx.is_some())
            });
            if reuse {
                self.arena_reservation = Some(self.parent_store.arena.reserve_existing());
                return Ok(());
            }
        }
        self.arena_reservation = Some(self.parent_store.arena.reserve(writes)?);
        Ok(())
    }

    /// Commit local changes to parent store
    ///
    /// Performance: This method drains local_versions to take ownership of
    /// RowVersion values, avoiding expensive clones. The transaction is
    /// consumed after commit anyway, so this is safe.
    pub fn commit(&mut self) -> Result<(), Error> {
        let externally_prepared = self.arena_reservation.is_some();
        self.reserve_commit_capacity()?;
        let result = self.commit_prepared();
        if !externally_prepared {
            self.arena_reservation.take();
        }
        result
    }

    fn commit_prepared(&mut self) -> Result<(), Error> {
        // OCC validation: detect concurrent write conflicts.
        // Rows removed by seal (missing from hot B-tree) are not conflicts —
        // they were moved to cold segments, not modified by another transaction.
        self.detect_conflicts_safe()?;

        // Update indexes BEFORE committing versions
        self.update_indexes_on_commit()?;

        // Commit local versions to parent store
        if let (Some(local_versions), Some(reservation)) = (
            self.local_versions.as_mut(),
            self.arena_reservation.as_mut(),
        ) {
            let before = local_versions.allocation_bytes();
            if local_versions.len() == 1 {
                // Single-row fast path: avoid Vec allocation
                if let Some((row_id, mut versions)) = local_versions.drain().next() {
                    if let Some(version) = versions.pop() {
                        self.parent_store
                            .install_versions(reservation, [(row_id, version)]);
                    }
                }
            } else {
                // Multi-row path: collect into Vec
                let mut batch: Vec<(i64, RowVersion)> = local_versions
                    .drain()
                    .filter_map(|(row_id, mut versions)| versions.pop().map(|v| (row_id, v)))
                    .collect();

                // Sort by row_id to ensure deterministic locking order
                batch.sort_by_key(|(row_id, _)| *row_id);

                self.parent_store.install_versions(reservation, batch);
            }
            account_map_capacity(before, local_versions);
        }

        // Release ALL claims from write_set (includes both local and external claims).
        // Must drain the entire write_set — external claims from track_external_claim()
        // have no corresponding local_versions entry.
        if let Some(write_set) = self.write_set.as_mut() {
            let before = write_set.allocation_bytes();
            if write_set.len() == 1 {
                // Single-claim fast path: avoid Vec allocation
                if let Some((row_id, _)) = write_set.drain().next() {
                    self.parent_store.release_row_claim(row_id, self.txn_id);
                }
            } else {
                let mut row_ids: Vec<i64> = write_set.drain().map(|(row_id, _)| row_id).collect();
                // Sort by row_id to ensure deterministic locking order
                row_ids.sort_unstable();
                self.parent_store
                    .release_row_claims_batch(&row_ids, self.txn_id);
            }
            account_map_capacity(before, write_set);
        }

        self.release_versions(self.version_bytes);
        Ok(())
    }

    /// Update indexes during commit
    ///
    /// This method updates all indexes with the changes from this transaction.
    /// For each row being committed:
    /// - If there's an old version (UPDATE/DELETE), remove the old indexed values
    /// - If the new version is not deleted (INSERT/UPDATE), add the new indexed values
    ///
    /// Uses two paths:
    /// - Single-row fast path: SmallVec + immediate apply (zero heap allocation for 1-2 column indexes)
    /// - Multi-row batch path: Collects changes, then applies in batch (reduces lock acquisitions)
    ///
    /// Returns an error if a unique constraint is violated.
    fn update_indexes_on_commit(&self) -> Result<(), Error> {
        // Early exit if no local changes
        let Some(local_versions) = self.local_versions.as_ref() else {
            return Ok(());
        };

        if local_versions.is_empty() {
            return Ok(());
        }

        // Get all indexes - early exit if none
        let indexes: SmallVec<[Arc<dyn Index>; 4]> =
            self.parent_store.get_all_indexes().into_iter().collect();
        if indexes.is_empty() {
            return Ok(());
        }

        // FAST PATH: Single-row commit (most common case for auto-commit INSERT/UPDATE/DELETE)
        // Uses SmallVec to avoid heap allocation for 1-2 column indexes
        if local_versions.len() == 1 {
            return self.update_indexes_single_row(&indexes);
        }

        // BATCH PATH: Multi-row commit
        // Sort indexes by name for deterministic lock ordering (prevents deadlocks)
        let mut indexes: Vec<_> = indexes.into_vec();
        indexes.sort_by(|a, b| a.name().cmp(b.name()));

        let num_indexes = indexes.len();

        // Pre-allocate per-index batch vectors
        let mut add_batches: Vec<Vec<(i64, Vec<crate::core::Value>)>> =
            (0..num_indexes).map(|_| Vec::new()).collect();
        let mut remove_batches: Vec<Vec<(i64, Vec<crate::core::Value>)>> =
            (0..num_indexes).map(|_| Vec::new()).collect();

        // Collect index updates for each row
        for (row_id, versions) in local_versions.iter() {
            // Get the latest version for this row (last in the list)
            let Some(new_version) = versions.last() else {
                continue;
            };

            let is_deleted = new_version.is_deleted();
            let new_row = &new_version.data;

            // Get old version from write_set (if exists)
            let old_row: Option<&crate::core::Row> = self
                .write_set
                .as_ref()
                .and_then(|ws| ws.get(row_id))
                .and_then(|entry| entry.read_version.as_ref())
                .map(|rv| &rv.data);

            for (idx, index) in indexes.iter().enumerate() {
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    continue;
                }

                // OPTIMIZATION: For UPDATEs, check if any indexed column changed
                // BEFORE allocating Vecs
                if let Some(old_r) = old_row {
                    if !is_deleted {
                        // UPDATE case: check if indexed columns differ
                        let any_changed = column_ids.iter().any(|&col_id| {
                            let col_idx = col_id as usize;
                            let old_val = old_r.get(col_idx);
                            let new_val = new_row.get(col_idx);
                            old_val != new_val
                        });

                        if !any_changed {
                            // Indexed columns unchanged - skip this index
                            continue;
                        }

                        // Values differ - collect old and new values
                        let old_values: Vec<crate::core::Value> = column_ids
                            .iter()
                            .map(|&col_id| {
                                old_r.get(col_id as usize).cloned().unwrap_or(
                                    crate::core::Value::Null(crate::core::DataType::Null),
                                )
                            })
                            .collect();

                        let new_values: Vec<crate::core::Value> = column_ids
                            .iter()
                            .map(|&col_id| {
                                new_row.get(col_id as usize).cloned().unwrap_or(
                                    crate::core::Value::Null(crate::core::DataType::Null),
                                )
                            })
                            .collect();

                        remove_batches[idx].push((row_id, old_values));
                        add_batches[idx].push((row_id, new_values));
                        continue;
                    }
                }

                if is_deleted {
                    // DELETE: collect values to remove from index
                    // Use old_row if available, otherwise fall back to new_row
                    let source_row = old_row.unwrap_or(new_row);
                    let values_to_remove: Vec<crate::core::Value> = column_ids
                        .iter()
                        .map(|&col_id| {
                            source_row
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    remove_batches[idx].push((row_id, values_to_remove));
                } else {
                    // INSERT: collect values to add to index
                    // (old_row.is_some() cases with !is_deleted are handled above with continue)
                    let new_values: Vec<crate::core::Value> = column_ids
                        .iter()
                        .map(|&col_id| {
                            new_row
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(crate::core::Value::Null(crate::core::DataType::Null))
                        })
                        .collect();
                    add_batches[idx].push((row_id, new_values));
                }
            }
        }

        // PHASE 1: Pre-validate ALL unique indexes before modifying ANY index
        // This prevents index pollution when a later index fails
        for (idx, index) in indexes.iter().enumerate() {
            if !index.is_unique() || add_batches[idx].is_empty() {
                continue;
            }

            // Build a set of row_ids being removed for O(1) conflict resolution
            // For unique indexes, if a row_id is being removed, its current value is being removed.
            // We don't need to track values - just knowing the row_id is sufficient.
            let removals_set: crate::common::I64Set = remove_batches[idx]
                .iter()
                .map(|(row_id, _)| *row_id)
                .collect();

            // Check for intra-batch duplicates first
            let mut seen: ahash::AHashMap<&[crate::core::Value], i64> =
                ahash::AHashMap::with_capacity(add_batches[idx].len());

            let is_hnsw = index.index_type() == crate::core::IndexType::Hnsw;

            for (row_id, values) in &add_batches[idx] {
                // Skip NULLs - they don't violate uniqueness
                if values.iter().any(|v| v.is_null()) {
                    continue;
                }

                // Check intra-batch duplicates (applies to all index types, including HNSW)
                if let Some(&existing_row_id) = seen.get(values.as_slice()) {
                    if existing_row_id != *row_id {
                        let values_str: Vec<String> =
                            values.iter().map(|v| format!("{:?}", v)).collect();
                        return Err(Error::unique_constraint(
                            index.name(),
                            index.column_names().join(", "),
                            format!("[{}]", values_str.join(", ")),
                        ));
                    }
                }
                seen.insert(values.as_slice(), *row_id);

                if is_hnsw {
                    // HNSW uniqueness: use exact vector-byte duplicate check.
                    // This is metric-independent and avoids threshold-based false negatives.
                    let Some(hnsw_index) = index
                        .as_any()
                        .downcast_ref::<crate::storage::index::HnswIndex>()
                    else {
                        return Err(Error::internal(format!(
                            "index '{}' advertised HNSW type but cannot be downcast",
                            index.name()
                        )));
                    };

                    if let Some(value) = values.first() {
                        if let Some(existing_row_id) =
                            hnsw_index.find_exact_duplicate(value, *row_id, Some(&removals_set))
                        {
                            let dims = value.as_vector_f32().map_or(0, |v| v.len());
                            return Err(Error::unique_constraint(
                                index.name(),
                                index.column_names().join(", "),
                                format!(
                                    "<vector({} dims)> conflicts with row_id {}",
                                    dims, existing_row_id
                                ),
                            ));
                        }
                    }
                } else {
                    // Non-HNSW: standard equality-based uniqueness check
                    // Check against existing index entries
                    let existing = index.get_row_ids_equal(values);

                    // Identify potential conflicts (exclude self and rows being removed)
                    // O(1) lookup using removals_set instead of O(N) scan
                    let has_real_conflict = existing.iter().any(|&conflict_id| {
                        conflict_id != *row_id && !removals_set.contains(conflict_id)
                    });

                    if has_real_conflict {
                        let values_str: Vec<String> =
                            values.iter().map(|v| format!("{:?}", v)).collect();
                        return Err(Error::unique_constraint(
                            index.name(),
                            index.column_names().join(", "),
                            format!("[{}]", values_str.join(", ")),
                        ));
                    }
                }
            }
        }

        // PHASE 2: All validations passed - now modify all indexes
        // Track modifications for rollback if a later index fails (race condition protection)
        // Between Phase 1 validation and Phase 2 modification, another transaction could commit,
        // causing a unique constraint violation that wasn't detected in Phase 1.
        let mut completed_removals: Vec<usize> = Vec::new();
        let mut completed_additions: Vec<usize> = Vec::new();

        for (idx, index) in indexes.iter().enumerate() {
            // Remove old entries first (for UPDATE correctness)
            if !remove_batches[idx].is_empty() {
                let batch: Vec<(i64, &[crate::core::Value])> = remove_batches[idx]
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();
                // Note: remove_batch_slice can fail (e.g., I/O error), but typically succeeds
                if index.remove_batch_slice(&batch).is_ok() {
                    completed_removals.push(idx);
                }
            }

            // Add new entries
            if !add_batches[idx].is_empty() {
                let batch: Vec<(i64, &[crate::core::Value])> = add_batches[idx]
                    .iter()
                    .map(|(row_id, values)| (*row_id, values.as_slice()))
                    .collect();

                if let Err(e) = index.add_batch_slice(&batch) {
                    // ROLLBACK: A later index failed - undo all previous modifications
                    // This prevents index corruption where indexes point to uncommitted rows

                    // 1. Remove additions we made to earlier indexes
                    for &rollback_idx in &completed_additions {
                        let rollback_batch: Vec<(i64, &[crate::core::Value])> = add_batches
                            [rollback_idx]
                            .iter()
                            .map(|(row_id, values)| (*row_id, values.as_slice()))
                            .collect();
                        let _ = indexes[rollback_idx].remove_batch_slice(&rollback_batch);
                    }

                    // 2. Re-add removals we made to earlier indexes
                    for &rollback_idx in &completed_removals {
                        let rollback_batch: Vec<(i64, &[crate::core::Value])> = remove_batches
                            [rollback_idx]
                            .iter()
                            .map(|(row_id, values)| (*row_id, values.as_slice()))
                            .collect();
                        // Rollback additions - ignore errors as we're in error recovery
                        let _ = indexes[rollback_idx].add_batch_slice(&rollback_batch);
                    }

                    return Err(e);
                }
                completed_additions.push(idx);
            }
        }

        let mut undo = self.index_undo.lock();
        for (idx, index) in indexes.iter().enumerate() {
            if add_batches[idx].is_empty() && remove_batches[idx].is_empty() {
                continue;
            }
            undo.push(
                IndexUndo {
                    index: Arc::clone(index),
                    added: std::mem::take(&mut add_batches[idx]),
                    removed: std::mem::take(&mut remove_batches[idx]),
                },
                &self.parent_store.memory,
            );
        }

        Ok(())
    }

    /// Fast path for single-row index updates
    ///
    /// Uses SmallVec to avoid heap allocation for indexes with 1-2 columns (most common case).
    /// Applies changes immediately without batching overhead.
    fn update_indexes_single_row(&self, indexes: &[Arc<dyn Index>]) -> Result<(), Error> {
        let local_versions = self.local_versions.as_ref().unwrap();

        // Get the single row entry
        let Some((row_id, versions)) = local_versions.iter().next() else {
            return Ok(());
        };

        let Some(new_version) = versions.last() else {
            return Ok(());
        };

        let is_deleted = new_version.is_deleted();
        let new_row = &new_version.data;

        // Get old version from write_set (if exists)
        let old_row: Option<&Row> = self
            .write_set
            .as_ref()
            .and_then(|ws| ws.get(row_id))
            .and_then(|entry| entry.read_version.as_ref())
            .map(|rv| &rv.data);

        // Track what we've done for rollback on error
        let mut completed_ops: SmallVec<[(usize, bool); 4]> = SmallVec::new(); // (index_idx, is_add)

        for (idx, index) in indexes.iter().enumerate() {
            let column_ids = index.column_ids();
            if column_ids.is_empty() {
                continue;
            }

            // OPTIMIZATION: For UPDATEs, check if any indexed column changed BEFORE allocating
            if let Some(old_r) = old_row {
                if !is_deleted {
                    // UPDATE case: check if indexed columns differ (no allocation)
                    let any_changed = column_ids.iter().any(|&col_id| {
                        let col_idx = col_id as usize;
                        old_r.get(col_idx) != new_row.get(col_idx)
                    });

                    if !any_changed {
                        // Indexed columns unchanged - skip this index entirely
                        continue;
                    }

                    // Values differ - collect and update using SmallVec (stack allocation for 1-2 cols)
                    let old_values: SmallVec<[Value; 2]> = column_ids
                        .iter()
                        .map(|&col_id| {
                            old_r
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(Value::Null(DataType::Null))
                        })
                        .collect();

                    let new_values: SmallVec<[Value; 2]> = column_ids
                        .iter()
                        .map(|&col_id| {
                            new_row
                                .get(col_id as usize)
                                .cloned()
                                .unwrap_or(Value::Null(DataType::Null))
                        })
                        .collect();

                    // Remove old, add new
                    let _ = index.remove(&old_values, row_id, row_id);
                    completed_ops.push((idx, false)); // false = removal

                    if let Err(e) = index.add(&new_values, row_id, row_id) {
                        // Rollback: re-add the old value we removed
                        let _ = index.add(&old_values, row_id, row_id);
                        // Rollback previous indexes
                        self.rollback_single_row_ops(
                            &completed_ops,
                            indexes,
                            row_id,
                            old_row,
                            new_row,
                        );
                        return Err(e);
                    }
                    completed_ops.push((idx, true)); // true = addition
                    continue;
                }
            }

            if is_deleted {
                // DELETE: remove from index
                let source_row = old_row.unwrap_or(new_row);
                let values_to_remove: SmallVec<[Value; 2]> = column_ids
                    .iter()
                    .map(|&col_id| {
                        source_row
                            .get(col_id as usize)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect();
                let _ = index.remove(&values_to_remove, row_id, row_id);
                completed_ops.push((idx, false));
            } else {
                // INSERT: add values to index
                let new_values: SmallVec<[Value; 2]> = column_ids
                    .iter()
                    .map(|&col_id| {
                        new_row
                            .get(col_id as usize)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect();

                if let Err(e) = index.add(&new_values, row_id, row_id) {
                    // Rollback previous indexes
                    self.rollback_single_row_ops(&completed_ops, indexes, row_id, old_row, new_row);
                    return Err(e);
                }
                completed_ops.push((idx, true));
            }
        }

        let mut undo = self.index_undo.lock();
        for &(idx, is_add) in completed_ops.iter() {
            let index = &indexes[idx];
            let source = if is_add {
                new_row
            } else {
                old_row.unwrap_or(new_row)
            };
            let values: Vec<Value> = index
                .column_ids()
                .iter()
                .map(|&col_id| {
                    source
                        .get(col_id as usize)
                        .cloned()
                        .unwrap_or(Value::Null(DataType::Null))
                })
                .collect();
            let entry = vec![(row_id, values)];
            undo.push(
                IndexUndo {
                    index: Arc::clone(index),
                    added: if is_add { entry.clone() } else { Vec::new() },
                    removed: if is_add { Vec::new() } else { entry },
                },
                &self.parent_store.memory,
            );
        }

        Ok(())
    }

    /// Rollback helper for single-row operations
    #[inline]
    fn rollback_single_row_ops(
        &self,
        completed_ops: &[(usize, bool)],
        indexes: &[Arc<dyn Index>],
        row_id: i64,
        old_row: Option<&Row>,
        new_row: &Row,
    ) {
        for &(idx, is_add) in completed_ops.iter().rev() {
            let index = &indexes[idx];
            let column_ids = index.column_ids();

            if is_add {
                // We added new values - remove them
                let values: SmallVec<[Value; 2]> = column_ids
                    .iter()
                    .map(|&col_id| {
                        new_row
                            .get(col_id as usize)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect();
                let _ = index.remove(&values, row_id, row_id);
            } else {
                // We removed old values - re-add them
                let source = old_row.unwrap_or(new_row);
                let values: SmallVec<[Value; 2]> = column_ids
                    .iter()
                    .map(|&col_id| {
                        source
                            .get(col_id as usize)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect();
                let _ = index.add(&values, row_id, row_id);
            }
        }
    }

    /// Rollback - discard local changes and release claims
    pub fn rollback(&self) {
        self.release_all_claims();
    }

    /// Discard later local versions and release claims with no surviving write.
    pub fn rollback_to_timestamp(&mut self, timestamp: i64) {
        self.rollback_to_timestamp_with_pending(timestamp, &[]);
    }

    /// Retain claims for surviving cold tombstones as well as local versions.
    /// `pending` contains sorted row IDs after the cold rollback.
    pub(crate) fn rollback_to_timestamp_with_pending(&mut self, timestamp: i64, pending: &[i64]) {
        debug_assert!(pending.windows(2).all(|pair| pair[0] < pair[1]));
        let mut removed_bytes = 0;
        if let Some(local_versions) = self.local_versions.as_mut() {
            let before = local_versions.allocation_bytes();
            local_versions.retain(|_, versions| {
                versions.retain(|v| {
                    let keep = v.create_time <= timestamp;
                    if !keep {
                        removed_bytes += v.data.heap_bytes();
                    }
                    keep
                });
                if versions.is_empty() && versions.spilled() {
                    removed_bytes +=
                        (versions.capacity() * std::mem::size_of::<RowVersion>()) as u128;
                }
                !versions.is_empty()
            });
            account_map_capacity(before, local_versions);
        }
        let Some(write_set) = self.write_set.as_mut() else {
            self.release_versions(removed_bytes);
            return;
        };
        let before = write_set.allocation_bytes();
        let mut released: Vec<i64> = write_set
            .keys()
            .filter(|&row_id| {
                !self
                    .local_versions
                    .as_ref()
                    .is_some_and(|versions| versions.contains_key(row_id))
                    && pending.binary_search(&row_id).is_err()
            })
            .collect();
        for &row_id in &released {
            if let Some(entry) = write_set.remove(row_id) {
                removed_bytes += entry
                    .read_version
                    .as_ref()
                    .map_or(0, |v| v.data.heap_bytes());
            }
        }
        account_map_capacity(before, write_set);
        if !released.is_empty() {
            released.sort_unstable();
            self.parent_store
                .release_row_claims_batch(&released, self.txn_id);
        }
        self.release_versions(removed_bytes);
    }

    /// Track a claim made directly on the parent VersionStore (not through put()).
    /// Used by SegmentedTable for cold row UPDATE/DELETE claims that bypass
    /// TransactionVersionStore's put methods. Without tracking, these claims
    /// leak because commit() only releases claims found in write_set.
    pub fn track_external_claim(&mut self, row_id: i64) {
        let write_set = self.ensure_write_set();
        let before = write_set.allocation_bytes();
        // Only add if not already tracked (idempotent).
        // Use empty read_version since this is a cold-only claim — the actual
        // row data lives in cold storage, not in the hot version store.
        use crate::common::i64_map::Entry;
        if let Entry::Vacant(e) = write_set.entry(row_id) {
            e.insert(WriteSetEntry {
                read_version: None,
                read_version_seq: 0,
            });
        }
        account_map_capacity(before, write_set);
    }

    /// Release all row claims held by this transaction
    fn release_all_claims(&self) {
        let Some(write_set) = self.write_set.as_ref() else {
            return;
        };
        // OPTIMIZATION: Collect row_ids first, then batch release
        // Avoids holding write_set iterator while accessing parent_store
        let mut row_ids: Vec<i64> = write_set.keys().collect();
        // Sort by row_id to ensure deterministic locking order
        row_ids.sort_unstable();
        self.parent_store
            .release_row_claims_batch(&row_ids, self.txn_id);
    }
}

impl Drop for TransactionVersionStore {
    fn drop(&mut self) {
        // Release any row claims still held by this transaction.
        // This is a safety net for cases where drop happens without explicit
        // commit/rollback (e.g., transaction panics, implicit drop on scope exit).
        // Without this, claims in uncommitted_writes would leak permanently,
        // blocking future UPDATE/DELETE on those rows.
        self.release_all_claims();

        // Return maps to the pool for reuse by future transactions.
        // This reduces allocation overhead from ~5.5KB per transaction to near zero
        // for bulk insert workloads where many short-lived transactions are created.
        if let Some(map) = self.local_versions.take() {
            return_version_list_map(map);
        }
        if let Some(map) = self.write_set.take() {
            return_write_set_map(map);
        }
        self.release_versions(self.version_bytes);
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

    /// A visibility checker that does not force snapshot isolation, so the
    /// arena-backed grouped aggregation path is reachable.
    struct ReadCommittedChecker;

    #[test]
    fn version_payload_charges_follow_history_and_each_captured_root() {
        let mut store = VersionStore::with_visibility_checker(
            "payload_history",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        );
        store.set_max_version_history(2);
        let add = |txn_id| {
            store
                .add_version(
                    1,
                    RowVersion::new_with_timestamp(
                        txn_id,
                        Row::from(vec![Value::Integer(txn_id)]),
                        0,
                    ),
                )
                .unwrap();
        };
        add(1);
        let node_bytes = store.versions.read().node_bytes();
        let link_bytes =
            2 * std::mem::size_of::<usize>() + std::mem::size_of::<VersionChainEntry>();
        let first = store.capture_versions();
        add(2);
        assert_eq!(store.memory.usage().version_tree, node_bytes + link_bytes);
        let second = store.capture_versions();
        add(3);
        assert_eq!(store.memory.usage().version_tree, node_bytes);
        assert_eq!(
            store.memory.usage().pinned_version_tree,
            node_bytes * 2 + link_bytes
        );
        assert_eq!(store.version_payload_footprint(), (32, 96, 0));
        assert_eq!(first.get(1).unwrap().version.data[0], Value::Integer(1));
        assert_eq!(second.get(1).unwrap().version.data[0], Value::Integer(2));
        drop(first);
        let copy = store.capture_versions();
        assert_eq!(store.version_payload_footprint(), (32, 96, 0));
        drop(second);
        assert_eq!(store.version_payload_footprint(), (32, 32, 0));
        assert_eq!(store.memory.usage().pinned_version_tree, node_bytes);
        drop(copy);
        add(4);
        assert_eq!(store.version_payload_footprint(), (64, 0, 0));
        assert_eq!(store.memory.usage().version_tree, node_bytes + link_bytes);
        assert_eq!(
            store.cleanup_old_previous_versions_with_retention(std::time::Duration::ZERO),
            1
        );
        assert_eq!(store.version_payload_footprint(), (32, 0, 0));
        assert_eq!(store.memory.usage().version_tree, node_bytes);
    }

    #[test]
    fn captured_payload_account_outlives_the_version_store() {
        let registry = crate::storage::mvcc::memory::HotMemoryRegistry::default();
        let store = VersionStore::new("captured_payload_owner", test_schema());
        registry.register(store.memory_account());
        store
            .add_version(1, RowVersion::new(1, Row::from(vec![Value::Integer(7)])))
            .unwrap();
        let tree_bytes = store.versions.read().node_bytes();
        assert!(tree_bytes > 0);
        assert_eq!(registry.total().version_tree, tree_bytes);
        let snapshot = store.capture_versions();
        drop(store);
        let usage = registry.total();
        assert_eq!(usage.version_payloads, 0);
        assert_eq!(usage.pinned_version_payloads, 32);
        assert_eq!(usage.version_tree, 0);
        assert_eq!(usage.pinned_version_tree, tree_bytes);
        assert_eq!(usage.arena_payloads, 0);
        assert_eq!(usage.arena_capacity, 0);
        assert_eq!(snapshot.get(1).unwrap().version.data[0], Value::Integer(7));
        drop(snapshot);
        assert_eq!(registry.total().pinned_version_payloads, 0);
        assert_eq!(registry.total().pinned_version_tree, 0);
    }

    #[test]
    fn owned_payload_bound_resets_after_cow_and_history_removal() {
        let mut store = VersionStore::new("owned_payloads", test_schema());
        store.set_max_version_history(2);
        let mut wide = Row::with_capacity(64);
        wide.push(Value::Integer(1));
        store
            .add_version(1, RowVersion::new_deleted_with_timestamp(1, wide, 0))
            .unwrap();
        let snapshot = store.capture_versions();
        store
            .add_version(1, RowVersion::new(2, Row::from(vec![Value::Integer(2)])))
            .unwrap();
        assert_eq!(store.version_payload_footprint(), (1056, 1024, 0));
        assert_eq!(
            store
                .versions
                .read()
                .get(1)
                .unwrap()
                .prev
                .as_ref()
                .unwrap()
                .version
                .data
                .owned_capacity(),
            Some(1),
            "the canonical clone shrinks while its recorded capacity bound stays conservative"
        );
        store
            .add_version(1, RowVersion::new(3, Row::from(vec![Value::Integer(3)])))
            .unwrap();
        assert_eq!(store.version_payload_footprint(), (32, 1024, 0));
        assert_eq!(store.versions.read().payloads.owned_capacity, 0);
        let mut small = Row::with_capacity(4);
        small.push(Value::Integer(4));
        store
            .add_version(2, RowVersion::new_deleted_with_timestamp(4, small, 0))
            .unwrap();
        assert_eq!(store.version_payload_footprint(), (96, 1024, 0));
        assert_eq!(store.cleanup_deleted_rows(std::time::Duration::ZERO), 1);
        assert_eq!(store.version_payload_footprint(), (32, 1024, 0));
        drop(snapshot);
        assert_eq!(store.version_payload_footprint(), (32, 0, 0));
    }

    #[test]
    fn reinsert_keeps_replaced_arena_payload_charged_until_drop() {
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let mut store = VersionStore::new("reinsert_payload_owner", test_schema());
        store.set_max_version_history(1);
        store
            .add_version(1, RowVersion::new(1, Row::from(vec![Value::Integer(7)])))
            .unwrap();
        store
            .add_version(
                1,
                RowVersion::new_deleted(2, Row::from(vec![Value::Integer(8)])),
            )
            .unwrap();
        assert_eq!(store.version_payload_footprint(), (16, 0, 0));
        let account = Arc::clone(store.memory_account());
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_drop = Arc::clone(&observed);
        crate::test_failpoints::before_hot_owner_drop(move || {
            let usage = account.usage();
            assert_eq!(usage.version_payloads, 16);
            assert_eq!(usage.arena_payloads, 32);
            assert_eq!(
                usage.retired_arena_payloads, 32,
                "the replaced arena payload is still retained"
            );
            observed_drop.store(true, Ordering::Relaxed);
        });
        store
            .add_version(1, RowVersion::new(3, Row::from(vec![Value::Integer(9)])))
            .unwrap();
        assert!(observed.load(Ordering::Relaxed));
        assert_eq!(store.version_payload_footprint(), (32, 0, 0));
    }

    #[test]
    fn different_deleted_payload_and_truncate_keep_each_owner_charged() {
        let mut store = VersionStore::new("deleted_payloads", test_schema());
        store.set_max_version_history(1);
        let row = Row::from(vec![Value::text("a heap payload retained by the arena")]);
        let arena_bytes = (row.heap_bytes() + 16) as usize;
        store.add_version(1, RowVersion::new(1, row)).unwrap();
        let mut deleted = Row::with_capacity(128);
        deleted.push(Value::Integer(2));
        store
            .add_version(1, RowVersion::new_deleted(2, deleted))
            .unwrap();
        assert_eq!(store.version_payload_footprint(), (2048, 0, 0));
        assert_eq!(store.hot_bytes(), arena_bytes);
        let tree_bytes = store.versions.read().node_bytes();
        assert!(tree_bytes > 0);
        let snapshot = store.capture_versions();
        let reservation = store.arena.reserve(1).unwrap();
        assert!(store.truncate_all().is_err());
        assert_eq!(store.version_payload_footprint(), (2048, 2048, 0));
        assert_eq!(store.memory.usage().version_tree, tree_bytes);
        assert_eq!(store.memory.usage().pinned_version_tree, tree_bytes);
        assert_eq!(store.hot_bytes(), arena_bytes);
        drop(reservation);
        let retired = store.truncate_all().unwrap();
        assert_eq!(store.hot_bytes(), 0);
        assert_eq!(store.version_payload_footprint(), (0, 4096, arena_bytes));
        assert_eq!(store.memory.usage().version_tree, 0);
        assert_eq!(store.memory.usage().pinned_version_tree, tree_bytes * 2);
        drop(retired);
        assert_eq!(store.arena.capacity_bytes(), 0);
        assert_eq!(store.version_payload_footprint(), (0, 2048, 0));
        assert_eq!(store.memory.usage().pinned_version_tree, tree_bytes);
        drop(snapshot);
        assert_eq!(store.version_payload_footprint(), (0, 0, 0));
        assert_eq!(store.memory.usage().pinned_version_tree, 0);
    }

    #[test]
    fn seal_payload_charges_keep_skipped_rows_and_retirement_owners() {
        let store = VersionStore::with_visibility_checker(
            "seal_payloads",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        );
        for id in 1..=3 {
            store
                .add_version(id, RowVersion::new(1, Row::from(vec![Value::Integer(id)])))
                .unwrap();
        }
        let snapshot = ExtractionSnapshot {
            inner: store.capture_versions(),
        };
        store
            .add_version(2, RowVersion::new(2, Row::from(vec![Value::Integer(20)])))
            .unwrap();
        store.try_claim_row(3, 99).unwrap();
        let mut retired = store.prepare_arena_retirement(3);
        let (removed, _, skipped) = store.remove_sealed_rows(&[1, 2, 3], &snapshot, &mut retired);
        assert_eq!(removed, 1);
        assert_eq!(skipped, vec![2, 3]);
        assert_eq!(store.hot_bytes(), 64);
        assert_eq!(store.version_payload_footprint(), (96, 96, 32));
        drop(snapshot);
        assert_eq!(store.version_payload_footprint(), (96, 0, 32));
        drop(retired);
        assert_eq!(store.version_payload_footprint(), (96, 0, 0));
        store.release_row_claim(3, 99);
    }

    impl VisibilityChecker for ReadCommittedChecker {
        fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
            version_txn_id <= viewing_txn_id
        }

        fn get_current_sequence(&self) -> i64 {
            0
        }

        fn get_active_transaction_ids(&self) -> Vec<i64> {
            Vec::new()
        }

        fn needs_snapshot_isolation(&self, _txn_id: i64) -> bool {
            false
        }
    }

    /// `compute_grouped_aggregates` is public and its signature allows an
    /// empty aggregate list. The executor never asks for one, but the contract
    /// does, and it used to answer with a group per distinct key and no
    /// aggregate values.
    #[test]
    fn test_grouped_aggregates_with_no_aggregates() {
        use crate::core::DataType;

        let schema = crate::core::SchemaBuilder::new("no_aggs")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Integer, true, false)
            .build();

        let store = Arc::new(VersionStore::with_visibility_checker(
            "no_aggs".to_string(),
            schema,
            Arc::new(ReadCommittedChecker),
        ));

        let mut txn = TransactionVersionStore::new(Arc::clone(&store), 1);
        for (id, k) in [(1i64, 10i64), (2, 10), (3, 20)] {
            txn.put(id, Row::from(vec![Value::from(id), Value::from(k)]), false)
                .unwrap();
        }
        txn.commit().unwrap();

        let results = store
            .compute_grouped_aggregates(2, &[1], &[])
            .expect("arena path should handle an empty aggregate list");

        let mut keys: Vec<i64> = results
            .iter()
            .map(|r| match r.group_values.first() {
                Some(Value::Integer(k)) => *k,
                other => panic!("unexpected group value {other:?}"),
            })
            .collect();
        keys.sort();

        assert_eq!(keys, vec![10, 20]);
        assert!(results.iter().all(|r| r.aggregate_values.is_empty()));
    }

    #[test]
    fn test_row_version_creation() {
        let row = Row::from(vec![Value::from(1), Value::from("test")]);
        let version = RowVersion::new(1, row);

        assert_eq!(version.txn_id, 1);
        assert!(!version.is_deleted());
        assert!(version.create_time > 0);
    }

    #[test]
    fn test_row_version_deleted() {
        let row = Row::from(vec![Value::from(1)]);
        let version = RowVersion::new_deleted(1, row);

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
        assert_eq!(store.try_next_auto_increment_id(), Some(1));
        assert_eq!(store.try_next_auto_increment_id(), Some(2));
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
        let version = RowVersion::new(1, row);

        store.add_version(100, version).unwrap();

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
        let version = RowVersion::new(5, row);
        store.add_version(100, version).unwrap();

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
        let version = RowVersion::new(1, row.clone());
        store.add_version(100, version).unwrap();

        // Delete in transaction 2
        let deleted_version = RowVersion::new_deleted(2, row);
        store.add_version(100, deleted_version).unwrap();

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
        store
            .add_version(100, RowVersion::new(1, row.clone()))
            .unwrap();
        store
            .add_version(200, RowVersion::new(1, row.clone()))
            .unwrap();
        store.add_version(300, RowVersion::new(1, row)).unwrap();

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
        store.add_version(100, RowVersion::new(1, row)).unwrap();
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

    fn retained_transaction_versions(store: &TransactionVersionStore) -> u128 {
        let local: u128 = store
            .local_versions
            .iter()
            .flat_map(|m| m.values())
            .map(|history| {
                let payloads: u128 = history.iter().map(|v| v.data.heap_bytes()).sum();
                payloads
                    + if history.spilled() {
                        (history.capacity() * std::mem::size_of::<RowVersion>()) as u128
                    } else {
                        0
                    }
            })
            .sum();
        let originals: u128 = store
            .write_set
            .iter()
            .flat_map(|m| m.values())
            .filter_map(|entry| entry.read_version.as_ref())
            .map(|v| v.data.heap_bytes())
            .sum();
        local + originals
    }

    #[test]
    fn transaction_version_charges_cover_each_put_path() {
        let store = Arc::new(VersionStore::with_visibility_checker(
            "transaction_versions",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        ));
        for id in 1..=5 {
            store
                .add_version(id, RowVersion::new(1, Row::from(vec![Value::Integer(id)])))
                .unwrap();
        }
        let row = || Row::from(vec![Value::text("a retained transaction payload")]);
        let original = |id| store.get_visible_version(id, 10).unwrap();
        let mut local = TransactionVersionStore::new(Arc::clone(&store), 10);
        local.put(1, row(), false).unwrap();
        local
            .put_with_original(2, row(), original(2), false)
            .unwrap();
        local
            .put_batch_with_originals(vec![(3, row(), original(3)), (3, row(), original(3))])
            .unwrap();
        let mut wide = Row::with_capacity(64);
        wide.push(Value::Integer(4));
        local
            .put_batch_deleted(RowVec::from_vec(vec![(4, wide), (4, row())]))
            .unwrap();
        local
            .put_batch_deleted_with_originals(vec![
                (5, row(), original(5)),
                (5, row(), original(5)),
            ])
            .unwrap();
        let bytes = retained_transaction_versions(&local);
        assert!(bytes > 64 * std::mem::size_of::<Value>() as u128);
        assert_eq!(local.version_bytes, bytes);
        assert_eq!(store.memory.usage().transaction_versions as u128, bytes);
        local.rollback();
        assert_eq!(store.memory.usage().transaction_versions as u128, bytes);
        drop(local);
        assert_eq!(store.memory.usage().transaction_versions, 0);
    }

    #[test]
    fn transaction_version_charges_keep_failed_claim_original() {
        let store = Arc::new(VersionStore::with_visibility_checker(
            "failed_claim_versions",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        ));
        store
            .add_version(1, RowVersion::new(1, Row::from(vec![Value::Integer(7)])))
            .unwrap();
        store.try_claim_row(1, 2).unwrap();
        let mut local = TransactionVersionStore::new(Arc::clone(&store), 3);
        assert!(local
            .put(1, Row::from(vec![Value::Integer(8)]), false)
            .is_err());
        assert!(!local.has_local_changes());
        assert_eq!(store.memory.usage().transaction_versions, 32);
        assert_eq!(retained_transaction_versions(&local), 32);
        local.rollback_to_timestamp(i64::MIN);
        assert_eq!(store.memory.usage().transaction_versions, 0);
        store.release_row_claim(1, 2);
    }

    #[test]
    fn transaction_version_charges_keep_spill_capacity_after_savepoint() {
        let store = Arc::new(VersionStore::with_visibility_checker(
            "savepoint_versions",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        ));
        let mut local = TransactionVersionStore::new(Arc::clone(&store), 1);
        local
            .put(1, Row::from(vec![Value::Integer(0)]), false)
            .unwrap();
        let timestamp = local.get_latest_local(1).unwrap().create_time;
        for value in 1..9 {
            local
                .put(1, Row::from(vec![Value::Integer(value)]), false)
                .unwrap();
        }
        let capacity = local
            .local_versions
            .as_ref()
            .unwrap()
            .get(1)
            .unwrap()
            .capacity();
        local.rollback_to_timestamp(timestamp);
        let history = local.local_versions.as_ref().unwrap().get(1).unwrap();
        assert_eq!(history.len(), 1);
        assert!(history.spilled());
        assert_eq!(history.capacity(), capacity);
        assert_eq!(local.version_bytes, retained_transaction_versions(&local));
        assert_eq!(
            store.memory.usage().transaction_versions,
            32 + capacity * std::mem::size_of::<RowVersion>()
        );
        local.rollback_to_timestamp(i64::MIN);
        assert_eq!(store.memory.usage().transaction_versions, 0);
    }

    #[test]
    fn transaction_version_charges_follow_commit_success_and_failure() {
        let store = Arc::new(VersionStore::with_visibility_checker(
            "commit_versions",
            test_schema(),
            Arc::new(TestVisibilityChecker::new()),
        ));
        let mut first = TransactionVersionStore::new(Arc::clone(&store), 1);
        first
            .put(1, Row::from(vec![Value::Integer(7)]), false)
            .unwrap();
        let mut second = TransactionVersionStore::new(Arc::clone(&store), 2);
        second
            .put(1, Row::from(vec![Value::Integer(8)]), false)
            .unwrap();
        assert_eq!(store.memory.usage().transaction_versions, 64);
        first.commit().unwrap();
        assert_eq!(store.memory.usage().transaction_versions, 32);
        assert!(second.commit().is_err());
        assert_eq!(store.memory.usage().transaction_versions, 32);
        assert_eq!(retained_transaction_versions(&second), 32);
        drop(second);
        assert_eq!(store.memory.usage().transaction_versions, 0);
        assert_eq!(store.memory.usage().version_payloads, 32);
    }

    #[test]
    fn index_undo_charges_survive_transaction_and_table_owners() {
        for row_count in [1, 2] {
            let registry = crate::storage::mvcc::memory::HotMemoryRegistry::default();
            let schema = crate::core::SchemaBuilder::new("undo_memory")
                .column("id", DataType::Integer, false, true)
                .column("key", DataType::Text, true, false)
                .build();
            let mut store = VersionStore::with_visibility_checker(
                "undo_memory",
                schema,
                Arc::new(TestVisibilityChecker::new()),
            );
            store.set_max_version_history(1);
            let store = Arc::new(store);
            registry.register(store.memory_account());
            store.add_index(
                "key".to_string(),
                Arc::new(crate::storage::index::HashIndex::new(
                    "key".to_string(),
                    "undo_memory".to_string(),
                    vec!["key".to_string()],
                    vec![1],
                    vec![DataType::Text],
                    false,
                    0,
                )),
            );
            let mut first = TransactionVersionStore::new(Arc::clone(&store), 1);
            for id in 1..=row_count {
                first
                    .put(
                        id,
                        Row::from(vec![
                            Value::Integer(id),
                            Value::text("an old heap key retained for index undo"),
                        ]),
                        false,
                    )
                    .unwrap();
            }
            first.commit().unwrap();
            drop(first);
            assert_eq!(registry.total().transaction_undo, 0);
            let mut second = TransactionVersionStore::new(Arc::clone(&store), 2);
            for id in 1..=row_count {
                second
                    .put(
                        id,
                        Row::from(vec![Value::Integer(id), Value::text("new")]),
                        false,
                    )
                    .unwrap();
            }
            second.commit().unwrap();
            assert_eq!(registry.total().transaction_versions, 0);
            let undo = std::mem::take(&mut *second.index_undo.lock());
            let removed_keys = undo
                .entries
                .iter()
                .flat_map(|entry| &entry.removed)
                .flat_map(|(_, values)| values)
                .map(|value| value.heap_bytes() as u128)
                .sum::<u128>();
            assert!(removed_keys > 0);
            let bytes = undo.entries.capacity() as u128 * std::mem::size_of::<IndexUndo>() as u128
                + undo.entries.iter().map(IndexUndo::heap_bytes).sum::<u128>();
            assert!(bytes > removed_keys);
            assert_eq!(registry.total().transaction_undo as u128, bytes);
            drop(second);
            drop(store);
            assert_eq!(registry.total().transaction_undo as u128, bytes);
            drop(undo);
            assert_eq!(registry.total().transaction_undo, 0);
        }
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
            let version = RowVersion::new(txn_id, row);
            store.add_version(row_id, version).unwrap();
        }

        // After 5 versions with limit 3:
        // v4 triggered drop (4 > 3), so chain was: v4 -> v3 -> None (depth=2)
        // v5 added: v5 -> v4 -> v3 -> None (depth=3)
        let versions = store.versions.read().clone();
        let entry = versions.get(row_id).expect("Row should exist");

        // Count actual chain length
        let chain_depth = count_chain_depth(entry);

        // Chain depth should be at most limit + 1 (oscillates between 2 and limit+1)
        assert!(
            chain_depth <= 4,
            "Chain depth {} exceeds limit+1",
            chain_depth
        );
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
            let version = RowVersion::new(txn_id, row);
            store.add_version(row_id, version).unwrap();
        }

        // Verify chain is bounded (between 2 and limit+1)
        let versions = store.versions.read().clone();
        let entry = versions.get(row_id).expect("Row should exist");

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
            let version = RowVersion::new(txn_id, row);
            store.add_version(row_id, version).unwrap();
        }

        // Count chain length - should be 15 (unlimited)
        let versions = store.versions.read().clone();
        let entry = versions.get(row_id).expect("Row should exist");

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
            let version = RowVersion::new(txn_id, row);
            store.add_version(row_id, version).unwrap();
        }

        // Now add more via batch - each will trigger drop when exceeding limit
        let batch: Vec<(i64, RowVersion)> = (4..=7)
            .map(|txn_id| {
                let row = Row::from(vec![Value::from(txn_id)]);
                (row_id, RowVersion::new(txn_id, row))
            })
            .collect();

        let mut reservation = store.arena.reserve_existing();
        store.install_versions(&mut reservation, batch);

        // Verify chain is bounded (at most limit+1)
        // Chain can be as short as 1 right after pruning (when new_depth > limit)
        let versions = store.versions.read().clone();
        let entry = versions.get(row_id).expect("Row should exist");

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
        let version = RowVersion::new_with_timestamp(1, row, timestamp);

        assert_eq!(version.txn_id, 1);
        assert_eq!(version.create_time, timestamp);
        assert!(!version.is_deleted());
    }

    #[test]
    fn test_row_version_deleted_with_timestamp() {
        let row = Row::from(vec![Value::from(1)]);
        let timestamp = 87654321;
        let version = RowVersion::new_deleted_with_timestamp(1, row, timestamp);

        assert_eq!(version.txn_id, 1);
        assert_eq!(version.deleted_at_txn_id, 1);
        assert_eq!(version.create_time, timestamp);
        assert!(version.is_deleted());
    }

    #[test]
    fn test_row_version_debug_display() {
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, row);

        let debug = format!("{:?}", version);
        assert!(debug.contains("RowVersion"));
        assert!(debug.contains("txn_id: 1"));
        assert!(debug.contains("create_time"));

        let display = format!("{}", version);
        assert!(display.contains("TxnID: 1"));
        assert!(display.contains("CreateTime"));
    }

    #[test]
    fn test_write_set_entry_clone() {
        let row = Row::from(vec![Value::from(1)]);
        let version = RowVersion::new(1, row);

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
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

        // Mark it deleted
        store.mark_deleted(100, 2).unwrap();

        // Transaction 1 should still see it
        assert!(store.get_visible_version(100, 1).is_some());

        // Transaction 3 should not see deleted row
        assert!(store.get_visible_version(100, 3).is_none());

        // Mark non-existent row as deleted (no-op)
        store.mark_deleted(999, 2).unwrap();
    }

    #[test]
    fn test_version_store_get_visible_rows_with_limit() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add 20 rows
        for i in 1..=20 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
        let version = RowVersion::new(5, row);
        store.add_version(1, version).unwrap();

        // Add updated version from transaction 10
        let row2 = Row::from(vec![Value::from(200)]);
        let version2 = RowVersion::new(10, row2);
        store.add_version(1, version2).unwrap();

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
        let version = RowVersion::new_with_timestamp(1, row, 1000);
        store.add_version(1, version).unwrap();

        // Add version with later timestamp
        let row2 = Row::from(vec![Value::from(200)]);
        let version2 = RowVersion::new_with_timestamp(2, row2, 2000);
        store.add_version(1, version2).unwrap();

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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version((i + 1) as i64, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
    fn claim_capacity_follows_growth_shrink_and_store_destruction() {
        let store = VersionStore::new("claims", test_schema());
        let account = Arc::clone(store.memory_account());
        let check = || {
            assert_eq!(
                account.usage().row_claims,
                store.uncommitted_writes.read().allocation_bytes()
            );
        };
        check();
        let initial = account.usage().row_claims;
        for id in 0..1024 {
            store.try_claim_row(id, 1).unwrap();
        }
        check();
        assert!(account.usage().row_claims > initial);
        assert!(store.try_claim_row(0, 2).is_err());
        store.release_row_claim(0, 2);
        check();
        store.release_row_claim(0, 1);
        check();
        store.release_row_claims_batch(&(1..1024).collect::<Vec<_>>(), 1);
        check();
        let empty_capacity = account.usage().row_claims;
        assert!(empty_capacity >= initial);
        drop(store.truncate_all().unwrap());
        check();
        assert_eq!(account.usage().row_claims, empty_capacity);
        drop(store);
        assert_eq!(account.usage().row_claims, 0);
    }

    #[test]
    fn test_version_store_index_operations() {
        use crate::core::types::DataType;
        use crate::storage::index::HashIndex;

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
            0,
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
    fn test_version_store_apply_recovered_version() {
        let store = VersionStore::new("test_table".to_string(), test_schema());

        // Apply a recovered version
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, row);
        store.apply_recovered_version(100, version).unwrap();

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
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

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
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

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
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
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
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
        }

        assert_eq!(store.count_visible(2), 10);
    }

    // =========================================================================
    // Cleanup Tests
    // =========================================================================

    #[test]
    fn test_cleanup_deleted_rows_basic() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add 10 rows from transaction 1
        for i in 1..=10 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
        }

        // Delete all rows in transaction 2
        for i in 1..=10 {
            let row = Row::from(vec![Value::from(i)]);
            let mut version = RowVersion::new(1, row);
            version.deleted_at_txn_id = 2;
            store.add_version(i, version).unwrap();
        }

        // Cleanup with 0 retention (immediate)
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(0));

        // All 10 rows should be cleaned
        assert_eq!(cleaned, 10, "Expected 10 rows to be cleaned");

        // Verify versions map is empty
        assert_eq!(
            store.versions.read().len(),
            0,
            "Versions map should be empty"
        );
    }

    #[test]
    fn test_cleanup_respects_retention_period() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add and delete a row
        let row = Row::from(vec![Value::from(1)]);
        let version = RowVersion::new(1, row.clone());
        store.add_version(1, version).unwrap();

        let mut deleted_version = RowVersion::new(1, row);
        deleted_version.deleted_at_txn_id = 2;
        store.add_version(1, deleted_version).unwrap();

        // Cleanup with very long retention - should not clean
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(3600));
        assert_eq!(cleaned, 0, "Should not clean rows within retention period");

        // Verify row still exists
        assert_eq!(
            store.versions.read().len(),
            1,
            "Row should still exist in versions"
        );
    }

    #[test]
    fn test_cleanup_only_deleted_rows() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add 5 rows, delete only 3
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i)]);
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
        }

        // Delete rows 1, 3, 5
        for i in [1, 3, 5] {
            let row = Row::from(vec![Value::from(i)]);
            let mut version = RowVersion::new(1, row);
            version.deleted_at_txn_id = 2;
            store.add_version(i, version).unwrap();
        }

        // Cleanup
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(0));

        assert_eq!(cleaned, 3, "Should clean only 3 deleted rows");
        assert_eq!(
            store.versions.read().len(),
            2,
            "2 non-deleted rows should remain"
        );
    }

    #[test]
    fn test_cleanup_arena_memory_released() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Add rows with data
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i), Value::text(format!("data_{}", i))]);
            let version = RowVersion::new(1, row);
            store.add_version(i, version).unwrap();
        }

        // Record initial arena length
        let initial_arena_len = store.arena.slot_count();
        assert_eq!(initial_arena_len, 5);

        // Delete all rows
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i), Value::text(format!("data_{}", i))]);
            let mut version = RowVersion::new(1, row);
            version.deleted_at_txn_id = 2;
            store.add_version(i, version).unwrap();
        }

        // Cleanup
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(0));
        assert_eq!(cleaned, 5);

        let guard = store.arena.read_guard();
        assert!(guard.is_empty());
        assert_eq!(guard.rows().count(), 0);
    }

    #[test]
    fn test_cleanup_does_not_remove_reinserted_row() {
        // Regression test for race condition: if a deleted row_id gets a new live
        // version committed between the cleanup snapshot and the removal pass,
        // cleanup must NOT remove the new live version.
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Tx 1: insert row_id=100
        let row = Row::from(vec![Value::from(42)]);
        let version = RowVersion::new(1, row);
        store.add_version(100, version).unwrap();

        // Tx 2: delete row_id=100
        let row = Row::from(vec![Value::from(42)]);
        let mut deleted = RowVersion::new(1, row);
        deleted.deleted_at_txn_id = 2;
        store.add_version(100, deleted).unwrap();

        // Simulate the cleanup's first pass: snapshot and identify row for deletion
        assert!(store.versions.read().get(100).unwrap().version.is_deleted());

        // Now, BEFORE cleanup's removal pass, tx 3 re-inserts a live version at row_id=100
        let row = Row::from(vec![Value::from(99)]);
        let live_version = RowVersion::new(3, row);
        store.add_version(100, live_version).unwrap();

        // The row should now be live (not deleted)
        assert!(!store.versions.read().get(100).unwrap().version.is_deleted());

        // Run cleanup — it should see the row is no longer deleted and skip it
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(0));
        assert_eq!(cleaned, 0, "Should not clean a row that was re-inserted");

        // Row must still exist with the new live value
        let versions = store.versions.read();
        let entry = versions.get(100);
        assert!(entry.is_some(), "Row 100 must still exist");
        let version = &entry.unwrap().version;
        assert!(!version.is_deleted(), "Row 100 must be live, not deleted");
        assert_eq!(
            version.data.get(0),
            Some(&Value::from(99)),
            "Row 100 must have the re-inserted value"
        );
    }

    #[test]
    fn test_cleanup_mixed_reinserted_and_deleted() {
        // Some rows are still deleted (should be cleaned), some were re-inserted (should be kept).
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Insert and delete rows 1..=4
        for i in 1..=4 {
            let row = Row::from(vec![Value::from(i)]);
            store.add_version(i, RowVersion::new(1, row)).unwrap();
        }
        for i in 1..=4 {
            let row = Row::from(vec![Value::from(i)]);
            let mut del = RowVersion::new(1, row);
            del.deleted_at_txn_id = 2;
            store.add_version(i, del).unwrap();
        }

        // Re-insert rows 2 and 4 with new values (simulating concurrent commit)
        for &i in &[2, 4] {
            let row = Row::from(vec![Value::from(i * 100)]);
            store.add_version(i, RowVersion::new(3, row)).unwrap();
        }

        // Cleanup should only remove rows 1 and 3 (still deleted)
        let cleaned = store.cleanup_deleted_rows(std::time::Duration::from_secs(0));
        assert_eq!(cleaned, 2, "Should clean only rows 1 and 3");

        let versions = store.versions.read();
        assert!(versions.get(1).is_none(), "Row 1 should be removed");
        assert!(versions.get(3).is_none(), "Row 3 should be removed");
        assert!(versions.get(2).is_some(), "Row 2 should still exist");
        assert!(versions.get(4).is_some(), "Row 4 should still exist");

        // Verify re-inserted values
        assert_eq!(
            versions.get(2).unwrap().version.data.get(0),
            Some(&Value::from(200))
        );
        assert_eq!(
            versions.get(4).unwrap().version.data.get(0),
            Some(&Value::from(400))
        );
    }

    #[test]
    fn test_descending_order_returns_correct_version_under_snapshot_isolation() {
        let checker = Arc::new(TestVisibilityChecker::new());
        let store =
            VersionStore::with_visibility_checker("test_table".to_string(), test_schema(), checker);

        // Tx 1: insert row_id=1 with value=100, row_id=2 with value=200
        let row1 = Row::from(vec![Value::from(100)]);
        store.add_version(1, RowVersion::new(1, row1)).unwrap();
        let row2 = Row::from(vec![Value::from(200)]);
        store.add_version(2, RowVersion::new(1, row2)).unwrap();

        // Tx 3: update row_id=1 with new value=999
        // This creates a version chain: HEAD(txn=3, val=999) -> prev(txn=1, val=100)
        let updated_row = Row::from(vec![Value::from(999)]);
        store
            .add_version(1, RowVersion::new(3, updated_row))
            .unwrap();

        // Viewer txn_id=2: sees txn<=2, so sees txn=1 but NOT txn=3
        // Expected: row_id=1 should have value=100, row_id=2 should have value=200

        // Ascending order — materializes during iteration (should be correct)
        let asc = store.collect_rows_pk_ordered(2, true, 100, 0).unwrap();
        assert_eq!(asc.len(), 2, "Should see 2 rows ascending");
        // row_id=1 should have the original value (100), NOT the update (999)
        let (rid1, data1) = &asc[0];
        assert_eq!(*rid1, 1);
        assert_eq!(
            data1.get(0),
            Some(&Value::from(100)),
            "Ascending: row_id=1 should have original value 100, not updated 999"
        );

        // Descending order
        let desc = store.collect_rows_pk_ordered(2, false, 100, 0).unwrap();
        assert_eq!(desc.len(), 2, "Should see 2 rows descending");
        // row_id=1 is at index 1 in descending order (row_id=2 comes first)
        let (rid1_desc, data1_desc) = &desc[1];
        assert_eq!(*rid1_desc, 1);
        assert_eq!(
            data1_desc.get(0),
            Some(&Value::from(100)),
            "Descending: row_id=1 should have original value 100, not updated 999"
        );
    }

    #[test]
    fn test_unique_constraint_swap() {
        use crate::core::DataType;
        use crate::storage::index::HashIndex;

        // Setup: Table with unique index on column 'u' (index 1)
        let schema = crate::core::SchemaBuilder::new("test_swap")
            .column("id", DataType::Integer, false, true) // nullable=false, pk=true
            .column("u", DataType::Integer, true, false) // nullable=true, pk=false
            .build();

        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_swap".to_string(),
            schema.clone(),
            checker,
        ));

        // Create Unique Hash Index on 'u'
        let index = Arc::new(HashIndex::new(
            "idx_u".to_string(),
            "test_swap".to_string(),
            vec!["u".to_string()],
            vec![1],
            vec![DataType::Integer],
            true, // is_unique
            0,
        ));
        store.add_index("idx_u".to_string(), index);

        // Initial data: (1, 10), (2, 20)
        let mut txn1 = TransactionVersionStore::new(Arc::clone(&store), 1);
        txn1.put(1, Row::from(vec![Value::from(1), Value::from(10)]), false)
            .unwrap();
        txn1.put(2, Row::from(vec![Value::from(2), Value::from(20)]), false)
            .unwrap();
        txn1.commit().unwrap();

        // Swap: (1, 20), (2, 10) in Single Transaction
        let mut txn2 = TransactionVersionStore::new(Arc::clone(&store), 2);

        // Update row 1: 10 -> 20 (conflict with row 2's old value)
        txn2.put(1, Row::from(vec![Value::from(1), Value::from(20)]), false)
            .unwrap();

        // Update row 2: 20 -> 10 (taking row 1's old value)
        txn2.put(2, Row::from(vec![Value::from(2), Value::from(10)]), false)
            .unwrap();

        // Commit should succeed
        let result = txn2.commit();
        assert!(result.is_ok(), "Commit failed: {:?}", result.err());

        // Verify values
        let v1_new = store.get_visible_version(1, 3).unwrap();
        assert_eq!(v1_new.data.get(1).unwrap(), &Value::from(20));

        let v2_new = store.get_visible_version(2, 3).unwrap();
        assert_eq!(v2_new.data.get(1).unwrap(), &Value::from(10));
    }

    #[test]
    fn test_unique_constraint_performance_bulk_update() {
        use crate::core::DataType;
        use crate::storage::index::HashIndex;
        use std::time::Instant;

        // Setup: Table with unique index on column 'u' (index 1)
        let schema = crate::core::SchemaBuilder::new("test_perf")
            .column("id", DataType::Integer, false, true) // nullable=false, pk=true
            .column("u", DataType::Integer, true, false) // nullable=true, pk=false
            .build();

        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_perf".to_string(),
            schema.clone(),
            checker,
        ));

        // Create Unique Hash Index on 'u'
        let index = Arc::new(HashIndex::new(
            "idx_u".to_string(),
            "test_perf".to_string(),
            vec!["u".to_string()],
            vec![1],
            vec![DataType::Integer],
            true, // is_unique
            0,
        ));
        store.add_index("idx_u".to_string(), index);

        let row_count = 30000;
        let mut txn1 = TransactionVersionStore::new(Arc::clone(&store), 1);
        for i in 0..row_count {
            txn1.put(
                i as i64,
                Row::from(vec![Value::from(i), Value::from(i)]),
                false,
            )
            .unwrap();
        }
        txn1.commit().unwrap();

        let mut txn2 = TransactionVersionStore::new(Arc::clone(&store), 2);
        // Bulk update: shift every value by 1 (e.g., 0->1, 1->2, ... 4999->5000)
        // This will cause every row to conflict with the next one's old value
        for i in 0..row_count {
            txn2.put(
                i as i64,
                Row::from(vec![Value::from(i), Value::from(i + 1)]),
                false,
            )
            .unwrap();
        }

        let start = Instant::now();
        txn2.commit().unwrap();
        let duration = start.elapsed();
        println!("Commit time for {} rows: {:?}", row_count, duration);

        // If it's O(N^2), 30000 rows would take several seconds.
        // If it's O(N), it should be < 200ms normally.
        // Coverage instrumentation (llvm-cov) adds ~2-3x overhead, so use 2000ms.
        assert!(
            duration.as_millis() < 2000,
            "Performance regression detected! Commit took {:?}",
            duration
        );
    }

    /// Verify that truncate_all() holds uncommitted_writes(W) during the
    /// entire check-and-clear sequence, preventing the TOCTOU race where
    /// a concurrent try_claim_row() could add a claim between the check
    /// and the clear.
    ///
    /// With the fix, truncate_all() holds uncommitted_writes(W) for the
    /// entire duration, so any concurrent try_claim_row() blocks until
    /// truncate completes — at which point truncate has already cleared
    /// the table and the UPDATE will operate on a clean state.
    #[test]
    fn test_truncate_blocks_concurrent_claims() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let checker = Arc::new(TestVisibilityChecker::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "test_table".to_string(),
            test_schema(),
            checker,
        ));

        // Setup: 10 committed rows
        for i in 1..=10 {
            let row = Row::from(vec![Value::from(i)]);
            store.add_version(i, RowVersion::new(1, row)).unwrap();
        }
        assert_eq!(store.committed_row_count.load(Ordering::Relaxed), 10);

        // Pre-claim a row to simulate an in-progress UPDATE
        store
            .try_claim_row(5, 99)
            .expect("claim should succeed on fresh store");

        // truncate_all must fail because uncommitted_writes is not empty
        let result = store.truncate_all();
        assert!(
            result.is_err(),
            "truncate must fail when uncommitted writes exist"
        );

        // Verify data was NOT destroyed
        assert_eq!(
            store.committed_row_count.load(Ordering::Relaxed),
            10,
            "row count must be unchanged after failed truncate"
        );
        assert_eq!(store.versions.read().len(), 10);

        // Release the claim, then truncate should succeed
        store.release_row_claim(5, 99);
        let count = store.truncate_all().expect("truncate should succeed now");
        assert_eq!(count.rows_affected, 10);
        assert!(store.versions.read().is_empty());
        assert!(store.uncommitted_writes.read().is_empty());

        // Concurrent test: truncate holds the lock, blocking try_claim_row
        // Insert fresh data for next round
        for i in 1..=5 {
            let row = Row::from(vec![Value::from(i)]);
            store.add_version(i, RowVersion::new(1, row)).unwrap();
        }
        // Force committed_row_count to reflect the 5 rows
        store.committed_row_count.store(5, Ordering::Relaxed);

        let barrier = Arc::new(Barrier::new(2));
        let store2 = Arc::clone(&store);
        let barrier2 = Arc::clone(&barrier);

        // Thread: try to claim a row concurrently with truncate
        let handle = thread::spawn(move || {
            barrier2.wait();
            // This will either:
            // a) Execute BEFORE truncate's lock → claim succeeds, truncate sees
            //    non-empty uncommitted_writes → truncate fails (correct!)
            // b) Execute AFTER truncate's lock → claim succeeds on empty store
            //    (correct, no data to corrupt)
            store2.try_claim_row(3, 200)
        });

        barrier.wait();
        let truncate_result = store.truncate_all();
        let claim_result = handle.join().unwrap();

        // Both operations complete without panic.
        // Either truncate succeeded (claim was after) or failed (claim was before).
        // In no case should a ghost row appear.
        if truncate_result.is_ok() {
            // Truncate completed first — table is empty, claim is on empty store
            assert!(store.versions.read().is_empty() || store.versions.read().len() <= 1);
        } else {
            // Claim was first — truncate correctly rejected
            assert!(claim_result.is_ok());
        }
    }
}
