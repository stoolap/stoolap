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

//! Transaction state, fixed read epochs, and retention leases for MVCC.
//!
//! Lifecycle transitions, sequence allocation, and retention registration share
//! one mutex. Completed transactions are implicit after their sequence mappings
//! become older than every retained view. A small registry-owned cache avoids
//! locking for repeated ReadCommitted visibility checks.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;

use crate::common::{CompactArc, I64Map};
use crate::core::IsolationLevel;
use crate::storage::VisibilityChecker;

pub const INVALID_TRANSACTION_ID: i64 = -999999999;
pub const RECOVERY_TRANSACTION_ID: i64 = -1;
const ABORTED_SENTINEL: i64 = -1;
const STATUS_SHIFT: u32 = 62;
const SEQ_MASK: i64 = (1i64 << STATUS_SHIFT) - 1;
const SNAPSHOT_FLAG: i64 = 1i64 << STATUS_SHIFT;
// Namespaces are never reused, including after an engine is dropped. Assigned
// once per registry; no global atomic access occurs during statement reads.
static NEXT_REGISTRY_NAMESPACE: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TxnStatus {
    Active = 0,
    Committing = 1,
    Aborted = 2,
}

/// Transaction state and its begin-view registration. The begin word records the
/// transaction's effective isolation level; changing the global default does
/// not change transactions which have already begun.
/// Active state_seq stores the begin registration; Committing stores its commit
/// sequence and moves the registration into the small in-flight commit list.
/// Aborted state_seq stores the rollback acknowledgment. Keep this state 16 bytes
/// because the main registry also retains completed abort markers until GC.
#[derive(Clone, Copy, Debug)]
pub struct TxnState {
    begin_seq: i64,
    state_seq: i64,
}

impl TxnState {
    #[inline]
    const fn new_active(begin_seq: i64, snapshot: bool, begin_registration: i64) -> Self {
        Self {
            begin_seq: begin_seq | if snapshot { SNAPSHOT_FLAG } else { 0 },
            state_seq: begin_registration,
        }
    }

    const fn new_aborted() -> Self {
        Self {
            begin_seq: ABORTED_SENTINEL,
            state_seq: (TxnStatus::Aborted as i64) << STATUS_SHIFT,
        }
    }

    #[inline(always)]
    pub const fn is_aborted(&self) -> bool {
        self.begin_seq == ABORTED_SENTINEL
    }

    #[inline(always)]
    pub const fn begin_seq(&self) -> i64 {
        if self.is_aborted() {
            0
        } else {
            self.begin_seq & SEQ_MASK
        }
    }

    #[inline(always)]
    pub const fn status(&self) -> TxnStatus {
        if self.is_aborted() {
            TxnStatus::Aborted
        } else if self.state_seq >> STATUS_SHIFT == 0 {
            TxnStatus::Active
        } else {
            TxnStatus::Committing
        }
    }

    #[inline(always)]
    pub const fn is_active_or_committing(&self) -> bool {
        !self.is_aborted()
    }

    #[inline]
    const fn is_snapshot(&self) -> bool {
        !self.is_aborted() && self.begin_seq & SNAPSHOT_FLAG != 0
    }

    fn set_snapshot(&mut self, snapshot: bool) {
        self.begin_seq = self.begin_seq() | if snapshot { SNAPSHOT_FLAG } else { 0 };
    }

    #[inline]
    const fn active_begin_registration(&self) -> Option<i64> {
        if !self.is_aborted() && self.state_seq >> STATUS_SHIFT == 0 {
            Some(self.state_seq & SEQ_MASK)
        } else {
            None
        }
    }

    #[inline]
    fn set_committing(&mut self, commit_seq: i64) {
        self.state_seq = commit_seq | (1i64 << STATUS_SHIFT);
    }

    fn rollback_acknowledgment(&self) -> Option<i64> {
        (self.is_aborted() && self.state_seq & SEQ_MASK != 0)
            .then(|| (self.state_seq & SEQ_MASK) - 1)
    }

    fn acknowledge_rollback(&mut self, registration: i64) {
        if self.rollback_acknowledgment().is_none() {
            self.state_seq = ((TxnStatus::Aborted as i64) << STATUS_SHIFT) | (registration + 1);
        }
    }

    #[inline(always)]
    pub const fn commit_seq(&self) -> i64 {
        if !self.is_aborted() && self.state_seq >> STATUS_SHIFT == 1 {
            self.state_seq & SEQ_MASK
        } else {
            0
        }
    }
}

/// No process TLS: equal transaction IDs in separate registries never alias.
const CACHE_SIZE: usize = 256;
struct CommittedCache {
    entries: [AtomicI64; CACHE_SIZE],
}

impl CommittedCache {
    fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| AtomicI64::new(0)),
        }
    }

    #[inline(always)]
    fn index(txn_id: i64) -> usize {
        let x = txn_id as u64;
        ((x ^ (x >> CACHE_SIZE.trailing_zeros())) as usize) & (CACHE_SIZE - 1)
    }

    #[inline(always)]
    fn contains(&self, txn_id: i64) -> bool {
        txn_id > 0 && self.entries[Self::index(txn_id)].load(Ordering::Relaxed) == txn_id
    }

    #[inline]
    fn insert(&self, txn_id: i64) {
        self.entries[Self::index(txn_id)].store(txn_id, Ordering::Relaxed);
    }

    fn invalidate(&self, txn_id: i64) {
        self.entries[Self::index(txn_id)].store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
struct EpochData {
    cutoff: i64,
    // Zero means this view cannot authorize shared result reuse. Historical
    // views must never inherit the generation current when they are reopened.
    cache_generation: u64,
    // None avoids an empty-slice allocation in the common case.
    excluded: Option<CompactArc<[i64]>>,
}

impl EpochData {
    fn exclusions(&self) -> &[i64] {
        self.excluded.as_deref().unwrap_or(&[])
    }

    fn horizon(&self) -> i64 {
        self.exclusions()
            .first()
            .map_or(self.cutoff, |seq| self.cutoff.min(seq - 1))
    }

    #[inline]
    fn admits(&self, sequence: i64) -> bool {
        // Zero is the prehistory marker used by recovered legacy tombstones.
        sequence >= 0
            && sequence <= self.cutoff
            && self.exclusions().binary_search(&sequence).is_err()
    }
}

/// Fixed committed view. Clones share one registered lease; the last owner
/// unregisters it. This handle may outlive its originating transaction.
#[derive(Clone)]
pub struct ReadEpoch {
    lease: Arc<EpochLease>,
}

/// Scalar proof of one engine's committed logical view. It retains no epoch,
/// row tree, file, or registry owner. Only a matching engine can issue it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheProvenance {
    namespace: usize,
    generation: std::num::NonZeroU64,
}

struct EpochLease {
    registry: Arc<RegistryShared>,
    id: i64,
    data: EpochData,
}

impl Drop for EpochLease {
    fn drop(&mut self) {
        let mut state = self.registry.state.lock();
        state.leases.remove(self.id);
        state.reset_retention_if_idle(self.registry.active_txn_count.load(Ordering::Relaxed));
    }
}

impl std::fmt::Debug for ReadEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadEpoch")
            .field("cutoff", &self.cutoff())
            .field("exclusions", &self.excluded_sequences())
            .finish_non_exhaustive()
    }
}

impl ReadEpoch {
    /// Used only while all claimed owners are exclusively borrowed by a
    /// destructive DDL operation. The caller must count actual private epoch
    /// values, and reject cached views with any externally shared Arc owner.
    pub(crate) fn has_exact_owners(&self, owners: usize) -> bool {
        owners != 0 && Arc::strong_count(&self.lease) == owners
    }

    pub(crate) fn cache_identity(&self) -> (usize, i64) {
        (self.lease.registry.namespace, self.lease.id)
    }

    /// True for handles sharing the same registered statement lease.
    #[inline]
    pub fn same_epoch(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.lease, &other.lease)
    }

    pub fn cutoff(&self) -> i64 {
        self.lease.data.cutoff
    }
    pub fn excluded_sequences(&self) -> &[i64] {
        self.lease.data.exclusions()
    }
    pub fn retention_horizon(&self) -> i64 {
        self.lease.data.horizon()
    }

    /// Apply the same fixed decision to versioned cold tombstones.
    pub fn admits_commit_sequence(&self, sequence: i64) -> bool {
        self.lease.data.admits(sequence)
    }

    /// Committed visibility only. The transaction's own writes belong to the
    /// caller's separate overlay and are not made committed by this method.
    pub fn is_visible(&self, txn_id: i64) -> bool {
        if txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        self.lease
            .registry
            .state
            .lock()
            .visible_at(txn_id, &self.lease.data)
    }
}

/// A small, allocation-free cache for one iterator over a fixed epoch. Answers
/// for versions reachable from a valid captured view are immutable, including
/// transactions that were Active or Committing at capture. After acknowledged
/// undo, newer captures cannot contain that aborted version; GC may retire its
/// marker, so arbitrary queries about that retired identity are not stable.
pub struct EpochReader<'a> {
    lease: &'a EpochLease,
    cache: [(i64, bool); 8],
}

impl ReadEpoch {
    pub fn reader(&self) -> EpochReader<'_> {
        EpochReader {
            lease: &self.lease,
            cache: [(0, false); 8],
        }
    }
}

impl EpochReader<'_> {
    #[inline]
    pub fn is_visible(&mut self, txn_id: i64) -> bool {
        if txn_id <= 0 {
            return txn_id == RECOVERY_TRANSACTION_ID;
        }
        let slot = CommittedCache::index(txn_id) & (self.cache.len() - 1);
        if self.cache[slot].0 == txn_id {
            return self.cache[slot].1;
        }
        let visible = self
            .lease
            .registry
            .state
            .lock()
            .visible_at(txn_id, &self.lease.data);
        self.cache[slot] = (txn_id, visible);
        visible
    }
}

/// A fixed seal/compaction eligibility bound registered before capture starts.
/// Dropping it releases retention, without changing any already issued epoch.
#[derive(Clone)]
pub struct BuildLease {
    lease: Arc<EpochLease>,
}

impl BuildLease {
    pub fn reader(&self) -> EpochReader<'_> {
        EpochReader {
            lease: &self.lease,
            cache: [(0, false); 8],
        }
    }

    pub fn cutoff(&self) -> i64 {
        self.lease.data.cutoff
    }

    pub fn is_eligible(&self, txn_id: i64) -> bool {
        if txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        self.lease
            .registry
            .state
            .lock()
            .visible_at(txn_id, &self.lease.data)
    }
}

#[derive(Clone, Copy)]
struct CommittingTxn {
    id: i64,
    commit_seq: i64,
    begin_registration: i64,
}

struct RegistryState {
    /// Prevent new read/build/transaction registrations while a prevalidated
    /// destructive DDL writes WAL. No lifecycle mutex is held during that IO.
    destructive_ddl: bool,
    logical_generation: u64,
    immediate_mutations: usize,
    transactions: I64Map<TxnState>,
    snapshot_seqs: I64Map<i64>,
    // Preserve begin-time exclusions for legacy callers that set isolation
    // after begin. Empty exclusions need no map entry/allocation.
    begin_exclusions: I64Map<CompactArc<[i64]>>,
    // Only committing transactions, normally inline. Avoid scanning all active
    // transactions whenever a statement or transaction captures an epoch.
    committing: SmallVec<[CommittingTxn; 4]>,
    committing_charge: Option<crate::common::memory::MemoryCharge>,
    leases: I64Map<i64>,
    // Lowered atomically with each registration; stale low floors only retain
    // extra history. Exact GC/pressure refresh raises it without per-row scans.
    cached_retention_horizon: Option<i64>,
    next_registration: i64,
    next_txn_id: i64,
    next_sequence: i64,
    // Every omitted committed mapping is known to be at or below this floor.
    // A bare historical cutoff older than it cannot safely infer visibility.
    discarded_through: i64,
}

impl RegistryState {
    fn advance_logical_generation(&mut self) {
        // Exhaustion disables reuse permanently. In particular, completion of
        // a durable COMMIT must neither fail nor reuse an earlier identity.
        if self.logical_generation != 0 {
            self.logical_generation = self.logical_generation.checked_add(1).unwrap_or(0);
        }
    }

    fn allocate_sequence(&mut self) -> i64 {
        let next = self
            .next_sequence
            .checked_add(1)
            .filter(|seq| *seq <= SEQ_MASK)
            .expect("transaction sequence exhausted");
        self.next_sequence = next;
        next
    }

    fn capture_data(&self) -> EpochData {
        let mut excluded: SmallVec<[i64; 4]> =
            self.committing.iter().map(|txn| txn.commit_seq).collect();
        excluded.sort_unstable();
        EpochData {
            cutoff: self.next_sequence,
            cache_generation: if self.immediate_mutations == 0 && !self.destructive_ddl {
                self.logical_generation
            } else {
                0
            },
            excluded: (!excluded.is_empty()).then(|| match self.transactions.memory_account() {
                Some(account) => CompactArc::from_slice_in(excluded.as_slice(), account),
                None => CompactArc::from_slice(excluded.as_slice()),
            }),
        }
    }

    fn transaction_data(&self, txn_id: i64) -> Option<EpochData> {
        let txn = self.transactions.get(txn_id)?;
        txn.is_active_or_committing().then(|| EpochData {
            cutoff: txn.begin_seq(),
            cache_generation: 0,
            excluded: self.begin_exclusions.get(txn_id).cloned(),
        })
    }

    fn allocate_registration(&mut self) -> i64 {
        self.next_registration = self
            .next_registration
            .checked_add(1)
            .filter(|id| *id < SEQ_MASK)
            .expect("read registration identity exhausted");
        self.next_registration
    }

    fn register_lease(&mut self, data: &EpochData) -> i64 {
        let id = self.allocate_registration();
        let horizon = data.horizon();
        self.lower_retention_horizon(horizon);
        self.leases.insert(id, horizon);
        id
    }

    fn lower_retention_horizon(&mut self, horizon: i64) {
        self.cached_retention_horizon = Some(
            self.cached_retention_horizon
                .map_or(horizon, |old| old.min(horizon)),
        );
    }

    fn reset_retention_if_idle(&mut self, active: usize) {
        if active == 0 && self.leases.is_empty() {
            self.cached_retention_horizon = None;
        }
    }

    fn refresh_retention_horizon(&mut self) -> Option<i64> {
        let horizon = self.retention_horizon();
        self.cached_retention_horizon = horizon;
        horizon
    }

    fn oldest_registration(&self) -> Option<i64> {
        self.transactions
            .values()
            .filter_map(TxnState::active_begin_registration)
            .chain(self.committing.iter().map(|txn| txn.begin_registration))
            .chain(self.leases.keys())
            .min()
    }

    fn safe_completed_cutoff(&self) -> i64 {
        self.committing
            .iter()
            .map(|txn| txn.commit_seq - 1)
            .min()
            .unwrap_or(self.next_sequence)
    }

    fn retention_horizon(&self) -> Option<i64> {
        // The legacy isolation setter can promote an active transaction to SI.
        // Preserve its begin view until that compatibility API is retired.
        let transactions = self
            .transactions
            .iter()
            .filter(|(_, txn)| txn.is_active_or_committing())
            .map(|(id, txn)| {
                self.begin_exclusions
                    .get(id)
                    .and_then(|seqs| seqs.first())
                    .map_or(txn.begin_seq(), |seq| txn.begin_seq().min(seq - 1))
            });
        transactions.chain(self.leases.values().copied()).min()
    }

    fn committed(&self, txn_id: i64) -> bool {
        txn_id > 0 && txn_id <= self.next_txn_id && !self.transactions.contains_key(txn_id)
    }

    fn visible_at(&self, txn_id: i64, epoch: &EpochData) -> bool {
        self.visible_at_parts(txn_id, epoch.cutoff, epoch.exclusions())
    }

    fn visible_at_parts(&self, txn_id: i64, cutoff: i64, excluded: &[i64]) -> bool {
        if !self.committed(txn_id) {
            return false;
        }
        match self.snapshot_seqs.get(txn_id) {
            Some(&sequence) => {
                sequence >= 0 && sequence <= cutoff && excluded.binary_search(&sequence).is_err()
            }
            None => {
                self.discarded_through <= excluded.first().map_or(cutoff, |seq| cutoff.min(seq - 1))
            }
        }
    }

    fn push_committing(&mut self, txn: CommittingTxn) {
        if let Some(charge) = self.committing_charge.as_mut() {
            if self.committing.len() == self.committing.capacity() {
                let capacity = self
                    .committing
                    .capacity()
                    .checked_mul(2)
                    .expect("committing capacity exhausted");
                let replacement = SmallVec::with_capacity(capacity);
                let new_bytes = capacity * std::mem::size_of::<CommittingTxn>();
                // Charge both live buffers before moving values and releasing
                // the old allocation. No counter update on the inline path.
                charge.resize(charge.bytes() + new_bytes);
                let previous = std::mem::replace(&mut self.committing, replacement);
                self.committing.extend(previous);
                charge.resize(new_bytes);
            }
        }
        self.committing.push(txn);
    }

    fn remove_committing(&mut self, txn_id: i64) {
        if let Some(index) = self.committing.iter().position(|txn| txn.id == txn_id) {
            self.committing.swap_remove(index);
        }
    }
}

struct RegistryShared {
    namespace: usize,
    state: Mutex<RegistryState>,
    ddl_finished: Condvar,
    global_isolation_level: AtomicU8,
    active_txn_count: AtomicUsize,
    snapshot_txn_count: AtomicUsize,
    accepting: AtomicBool,
    current_sequence: AtomicI64,
    committed_cache: CommittedCache,
    _object_charge: Option<crate::common::memory::MemoryCharge>,
}

/// Registry facade. Leases keep the state and retention metadata alive without
/// a cycle: registry state stores only lease IDs/horizons, never lease handles.
pub struct TransactionRegistry {
    shared: Arc<RegistryShared>,
    _object_charge: Option<crate::common::memory::MemoryCharge>,
}

/// Marks a logical mutation which does not wait for transaction visibility.
/// Owned guards may nest; they hold no mutex during the operation or its I/O.
#[must_use = "keep the guard alive through the complete logical mutation"]
pub struct LogicalMutationGuard {
    shared: Arc<RegistryShared>,
}

impl Drop for LogicalMutationGuard {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.advance_logical_generation();
        state.immediate_mutations -= 1;
    }
}

/// Admission reservation, not a held mutex: registered epochs cannot be
/// created between destructive DDL prevalidation and its completed mutation.
pub(crate) struct DestructiveDdlGuard<'a> {
    shared: &'a Arc<RegistryShared>,
}

impl DestructiveDdlGuard<'_> {
    /// Internal current-state validation is part of the reserved operation.
    /// It must not wait on its own admission flag or reuse an older SI epoch.
    pub(crate) fn capture_current_epoch(&self) -> ReadEpoch {
        let (data, id) = {
            let mut state = self.shared.state.lock();
            debug_assert!(state.destructive_ddl);
            let data = state.capture_data();
            let id = state.register_lease(&data);
            (data, id)
        };
        ReadEpoch {
            lease: Arc::new(EpochLease {
                registry: Arc::clone(self.shared),
                id,
                data,
            }),
        }
    }
}

impl Drop for DestructiveDdlGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.advance_logical_generation();
        state.destructive_ddl = false;
        self.shared.ddl_finished.notify_all();
    }
}

impl TransactionRegistry {
    pub fn begin_logical_mutation(&self) -> LogicalMutationGuard {
        let mut state = self.shared.state.lock();
        state.immediate_mutations = state
            .immediate_mutations
            .checked_add(1)
            .expect("logical mutation count exhausted");
        state.advance_logical_generation();
        LogicalMutationGuard {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Call after binding the table and schema. Never stamp a completed old
    /// result with a newly sampled generation; retain this original proof.
    pub fn cache_provenance(&self, epoch: &ReadEpoch) -> Option<CacheProvenance> {
        if !Arc::ptr_eq(&self.shared, &epoch.lease.registry) {
            return None;
        }
        let generation = std::num::NonZeroU64::new(epoch.lease.data.cache_generation)?;
        let state = self.shared.state.lock();
        (state.logical_generation == generation.get()
            && state.immediate_mutations == 0
            && !state.destructive_ddl)
            .then_some(CacheProvenance {
                namespace: self.shared.namespace,
                generation,
            })
    }

    fn wait_for_ddl(&self, state: &mut parking_lot::MutexGuard<'_, RegistryState>) {
        while state.destructive_ddl {
            self.shared.ddl_finished.wait(state);
        }
    }

    /// The caller exclusively borrows every counted epoch value until this
    /// guard drops. No additional lease is exempted, including another lease
    /// issued for the same transaction. This permits private table bindings,
    /// but rejects a delayed table capture/result sharing the caller's epoch.
    pub(crate) fn prepare_destructive_ddl(
        &self,
        txn_id: i64,
        private_epoch: Option<(&ReadEpoch, usize)>,
    ) -> crate::core::Result<DestructiveDdlGuard<'_>> {
        let mut state = self.shared.state.lock();
        if state.destructive_ddl
            || !self.shared.accepting.load(Ordering::Acquire)
            || state
                .transactions
                .get(txn_id)
                .is_none_or(|txn| txn.status() != TxnStatus::Active)
            || state
                .transactions
                .iter()
                .any(|(id, txn)| id != txn_id && txn.is_active_or_committing())
        {
            return Err(crate::core::Error::TableHasActiveTransactions);
        }
        let exempt = match private_epoch {
            Some((epoch, owners))
                if Arc::ptr_eq(&self.shared, &epoch.lease.registry)
                    && epoch.has_exact_owners(owners) =>
            {
                Some(epoch.lease.id)
            }
            Some(_) => return Err(crate::core::Error::TableHasActiveTransactions),
            None => None,
        };
        if state.leases.keys().any(|id| Some(id) != exempt) {
            return Err(crate::core::Error::TableHasActiveTransactions);
        }
        state.advance_logical_generation();
        state.destructive_ddl = true;
        Ok(DestructiveDdlGuard {
            shared: &self.shared,
        })
    }

    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_optional_account(capacity, None)
    }

    fn with_optional_account(
        capacity: usize,
        account: Option<&crate::common::memory::MemoryAccount>,
    ) -> Self {
        Self {
            shared: Arc::new(RegistryShared {
                namespace: NEXT_REGISTRY_NAMESPACE
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                        next.checked_add(1)
                    })
                    .expect("registry namespace exhausted"),
                state: Mutex::new(RegistryState {
                    destructive_ddl: false,
                    logical_generation: 1,
                    immediate_mutations: 0,
                    transactions: account.map_or_else(
                        || I64Map::with_capacity(capacity),
                        |origin| I64Map::with_capacity_in(capacity, origin),
                    ),
                    snapshot_seqs: account.map_or_else(I64Map::new, I64Map::new_in),
                    begin_exclusions: account.map_or_else(I64Map::new, I64Map::new_in),
                    committing: SmallVec::new(),
                    committing_charge: account
                        .map(|origin| crate::common::memory::MemoryCharge::new(origin, 0)),
                    leases: account.map_or_else(I64Map::new, I64Map::new_in),
                    cached_retention_horizon: None,
                    next_registration: 0,
                    next_txn_id: 0,
                    next_sequence: 0,
                    discarded_through: 0,
                }),
                ddl_finished: Condvar::new(),
                global_isolation_level: AtomicU8::new(0),
                active_txn_count: AtomicUsize::new(0),
                snapshot_txn_count: AtomicUsize::new(0),
                accepting: AtomicBool::new(true),
                current_sequence: AtomicI64::new(0),
                committed_cache: CommittedCache::new(),
                // std::Arc headers are opaque. Keep the allowance with this
                // physical owner: external epochs can outlive the facade.
                _object_charge: account.map(|origin| {
                    crate::common::memory::MemoryCharge::conservative(
                        origin,
                        std::mem::size_of::<RegistryShared>() + 4 * std::mem::size_of::<usize>(),
                    )
                }),
            }),
            _object_charge: account.map(|origin| {
                crate::common::memory::MemoryCharge::conservative(
                    origin,
                    std::mem::size_of::<Self>() + 4 * std::mem::size_of::<usize>(),
                )
            }),
        }
    }

    pub(crate) fn new_in(account: &crate::common::memory::MemoryAccount) -> Self {
        Self::with_optional_account(16, Some(account))
    }

    #[inline]
    const fn isolation_to_u8(level: IsolationLevel) -> u8 {
        match level {
            IsolationLevel::ReadCommitted => 0,
            IsolationLevel::SnapshotIsolation => 1,
        }
    }

    #[inline]
    const fn u8_to_isolation(level: u8) -> IsolationLevel {
        if level == 0 {
            IsolationLevel::ReadCommitted
        } else {
            IsolationLevel::SnapshotIsolation
        }
    }

    pub fn set_global_isolation_level(&self, level: IsolationLevel) {
        let _state = self.shared.state.lock();
        self.shared
            .global_isolation_level
            .store(Self::isolation_to_u8(level), Ordering::Release);
    }

    pub fn get_global_isolation_level(&self) -> IsolationLevel {
        Self::u8_to_isolation(self.shared.global_isolation_level.load(Ordering::Acquire))
    }

    pub fn set_transaction_isolation_level(&self, txn_id: i64, level: IsolationLevel) {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        if let Some(txn) = state.transactions.get_mut(txn_id) {
            if !txn.is_active_or_committing() {
                return;
            }
            let snapshot = level == IsolationLevel::SnapshotIsolation;
            if txn.is_snapshot() != snapshot {
                if snapshot {
                    self.shared
                        .snapshot_txn_count
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    self.shared
                        .snapshot_txn_count
                        .fetch_sub(1, Ordering::Relaxed);
                }
                txn.set_snapshot(snapshot);
            }
        }
    }

    pub fn remove_transaction_isolation_level(&self, txn_id: i64) {
        self.set_transaction_isolation_level(txn_id, self.get_global_isolation_level());
    }

    pub fn get_isolation_level(&self, txn_id: i64) -> IsolationLevel {
        if self.shared.snapshot_txn_count.load(Ordering::Relaxed) == 0
            && self.shared.global_isolation_level.load(Ordering::Relaxed) == 0
        {
            return IsolationLevel::ReadCommitted;
        }
        let state = self.shared.state.lock();
        match state
            .transactions
            .get(txn_id)
            .filter(|txn| txn.is_active_or_committing())
        {
            Some(txn) => Self::u8_to_isolation(txn.is_snapshot() as u8),
            None => self.get_global_isolation_level(),
        }
    }

    #[inline]
    fn needs_snapshot_isolation(&self, txn_id: i64) -> bool {
        self.shared.snapshot_txn_count.load(Ordering::Relaxed) > 0
            && self
                .shared
                .state
                .lock()
                .transactions
                .get(txn_id)
                .is_some_and(TxnState::is_snapshot)
    }

    pub fn has_active_snapshot_transactions(&self) -> bool {
        self.shared.snapshot_txn_count.load(Ordering::Relaxed) > 0
    }

    pub fn get_snapshot_transaction_ids(&self) -> Vec<i64> {
        self.shared
            .state
            .lock()
            .transactions
            .iter()
            .filter_map(|(id, txn)| txn.is_snapshot().then_some(id))
            .collect()
    }

    pub fn get_min_snapshot_begin_seq(&self) -> Option<i64> {
        self.shared
            .state
            .lock()
            .transactions
            .values()
            .filter(|txn| txn.is_snapshot())
            .map(TxnState::begin_seq)
            .min()
    }

    /// Capture and register before releasing the lifecycle mutex.
    pub fn capture_read_epoch(&self) -> ReadEpoch {
        let (data, id) = {
            let mut state = self.shared.state.lock();
            self.wait_for_ddl(&mut state);
            let data = state.capture_data();
            let id = state.register_lease(&data);
            (data, id)
        };
        ReadEpoch {
            lease: Arc::new(EpochLease {
                registry: Arc::clone(&self.shared),
                id,
                data,
            }),
        }
    }

    /// SI reuses its original begin epoch; RC captures this statement's epoch.
    /// Returns None after the transaction has reached a terminal state.
    pub fn read_epoch_for_transaction(&self, txn_id: i64) -> Option<ReadEpoch> {
        let (data, id) = {
            let mut state = self.shared.state.lock();
            self.wait_for_ddl(&mut state);
            let txn = state.transactions.get(txn_id)?;
            if !txn.is_active_or_committing() {
                return None;
            }
            let data = if txn.is_snapshot() {
                state.transaction_data(txn_id)?
            } else {
                state.capture_data()
            };
            let id = state.register_lease(&data);
            (data, id)
        };
        Some(ReadEpoch {
            lease: Arc::new(EpochLease {
                registry: Arc::clone(&self.shared),
                id,
                data,
            }),
        })
    }

    /// Choose one safe bound and register its retention in the same critical
    /// section. The bound never advances, even when an excluded commit finishes.
    pub fn register_build_lease(&self) -> BuildLease {
        let (data, id) = {
            let mut state = self.shared.state.lock();
            self.wait_for_ddl(&mut state);
            let safe = state.safe_completed_cutoff();
            let cutoff = state
                .refresh_retention_horizon()
                .map_or(safe, |horizon| safe.min(horizon));
            let data = EpochData {
                cutoff,
                cache_generation: 0,
                excluded: None,
            };
            let id = state.register_lease(&data);
            (data, id)
        };
        BuildLease {
            lease: Arc::new(EpochLease {
                registry: Arc::clone(&self.shared),
                id,
                data,
            }),
        }
    }

    pub fn oldest_retention_horizon(&self) -> Option<i64> {
        self.refresh_retention_horizon()
    }

    /// Conservative O(1) floor for publication. Registration lowers the floor
    /// before becoming observable. Last-obligation release resets it in O(1).
    /// Continuous overlap may leave a stale low floor until explicit refresh.
    pub fn cached_retention_horizon(&self) -> Option<i64> {
        self.shared.state.lock().cached_retention_horizon
    }

    /// Refresh before pressure-driven history cleanup declares a live reader
    /// pin. This and normal GC perform the full scan outside row mutation paths.
    pub fn refresh_retention_horizon(&self) -> Option<i64> {
        self.shared.state.lock().refresh_retention_horizon()
    }

    /// Check a bounded candidate prefix under one registry lock. The supplied
    /// identities belong to one row's newest-to-oldest version chain.
    pub fn history_retention(
        &self,
        creators: &[i64],
    ) -> crate::storage::mvcc::version_store::HistoryRetention {
        use crate::storage::mvcc::version_store::HistoryRetention;
        let state = self.shared.state.lock();
        let Some(cutoff) = state.cached_retention_horizon else {
            return HistoryRetention::Unprotected;
        };
        let epoch = EpochData {
            cutoff,
            cache_generation: 0,
            excluded: None,
        };
        creators
            .iter()
            .position(|&id| id == RECOVERY_TRANSACTION_ID || state.visible_at(id, &epoch))
            .map_or(HistoryRetention::Protected, HistoryRetention::KeepThrough)
    }

    /// Presence-only check for history-cap decisions. Avoid computing the
    /// minimum across every registration for each row in a publishing batch.
    pub fn has_retention_obligations(&self) -> bool {
        let state = self.shared.state.lock();
        self.shared.active_txn_count.load(Ordering::Relaxed) != 0 || !state.leases.is_empty()
    }

    pub fn is_visible_in_epoch(&self, txn_id: i64, epoch: &ReadEpoch) -> bool {
        Arc::ptr_eq(&self.shared, &epoch.lease.registry) && epoch.is_visible(txn_id)
    }

    pub fn begin_transaction(&self) -> (i64, i64) {
        self.begin_transaction_inner(None)
    }

    /// Unlike begin-then-set, the initial isolation and epoch are published
    /// atomically with the transaction's identity.
    pub fn begin_transaction_with_isolation(&self, isolation: IsolationLevel) -> (i64, i64) {
        self.begin_transaction_inner(Some(isolation))
    }

    fn begin_transaction_inner(&self, isolation: Option<IsolationLevel>) -> (i64, i64) {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        if !self.shared.accepting.load(Ordering::Acquire) {
            return (INVALID_TRANSACTION_ID, 0);
        }
        let txn_id = state
            .next_txn_id
            .checked_add(1)
            .expect("transaction identity exhausted");
        let data = state.capture_data();
        let begin_seq = state.allocate_sequence();
        let begin_registration = state.allocate_registration();
        let snapshot = isolation.unwrap_or_else(|| self.get_global_isolation_level())
            == IsolationLevel::SnapshotIsolation;
        state.next_txn_id = txn_id;
        let horizon = data
            .exclusions()
            .first()
            .map_or(begin_seq, |seq| begin_seq.min(seq - 1));
        state.lower_retention_horizon(horizon);
        state.transactions.insert(
            txn_id,
            TxnState::new_active(begin_seq, snapshot, begin_registration),
        );
        if let Some(excluded) = data.excluded {
            state.begin_exclusions.insert(txn_id, excluded);
        }
        self.shared.active_txn_count.fetch_add(1, Ordering::Relaxed);
        if snapshot {
            self.shared
                .snapshot_txn_count
                .fetch_add(1, Ordering::Relaxed);
        }
        self.shared
            .current_sequence
            .store(begin_seq, Ordering::Release);
        (txn_id, begin_seq)
    }

    #[inline]
    pub fn start_commit(&self, txn_id: i64) -> i64 {
        self.start_commit_after_sequence(txn_id, || {})
    }

    fn start_commit_after_sequence(&self, txn_id: i64, after_sequence: impl FnOnce()) -> i64 {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        match state.transactions.get(txn_id).map(TxnState::status) {
            Some(TxnStatus::Active) => {}
            Some(TxnStatus::Committing) => {
                return state.transactions.get(txn_id).unwrap().commit_seq()
            }
            _ => return 0,
        }
        let commit_seq = state.allocate_sequence();
        after_sequence();
        let begin_registration = state
            .transactions
            .get(txn_id)
            .unwrap()
            .active_begin_registration()
            .expect("validated active transaction");
        // Grow the bounded in-flight list before overwriting the Active word,
        // which is still the sole owner of this begin registration. If a spill
        // allocation unwinds, the transaction remains Active and fully tracked.
        state.push_committing(CommittingTxn {
            id: txn_id,
            commit_seq,
            begin_registration,
        });
        state
            .transactions
            .get_mut(txn_id)
            .unwrap()
            .set_committing(commit_seq);

        self.shared
            .current_sequence
            .store(commit_seq, Ordering::Release);
        commit_seq
    }

    fn finish_commit_locked(&self, state: &mut RegistryState, txn_id: i64, commit_seq: i64) {
        state.advance_logical_generation();
        let txn = state.transactions.remove(txn_id).unwrap();
        state.remove_committing(txn_id);
        state.begin_exclusions.remove(txn_id);
        self.shared.active_txn_count.fetch_sub(1, Ordering::Relaxed);
        if txn.is_snapshot() {
            self.shared
                .snapshot_txn_count
                .fetch_sub(1, Ordering::Relaxed);
        }
        if self.shared.active_txn_count.load(Ordering::Relaxed) > 0
            || !state.leases.is_empty()
            || self.shared.global_isolation_level.load(Ordering::Relaxed) == 1
        {
            state.snapshot_seqs.insert(txn_id, commit_seq);
        } else {
            state.discarded_through = state.discarded_through.max(commit_seq);
        }
        // Publish only while holding the state lock; recovery invalidation uses
        // the same lock and cannot race a stale cache insertion.
        self.shared.committed_cache.insert(txn_id);
        state.reset_retention_if_idle(self.shared.active_txn_count.load(Ordering::Relaxed));
    }

    #[inline]
    pub fn complete_commit(&self, txn_id: i64) {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        // Read-only transactions use this API without start_commit.
        let commit_seq = match state.transactions.get(txn_id).map(TxnState::status) {
            Some(TxnStatus::Active) => state.allocate_sequence(),
            Some(TxnStatus::Committing) => state.transactions.get(txn_id).unwrap().commit_seq(),
            _ => return,
        };
        self.finish_commit_locked(&mut state, txn_id, commit_seq);
        self.shared
            .current_sequence
            .store(state.next_sequence, Ordering::Release);
    }

    pub fn commit_transaction(&self, txn_id: i64) -> i64 {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        let commit_seq = match state.transactions.get(txn_id).map(TxnState::status) {
            Some(TxnStatus::Active) => state.allocate_sequence(),
            Some(TxnStatus::Committing) => state.transactions.get(txn_id).unwrap().commit_seq(),
            _ => return state.snapshot_seqs.get(txn_id).copied().unwrap_or(0),
        };
        self.finish_commit_locked(&mut state, txn_id, commit_seq);
        self.shared
            .current_sequence
            .store(state.next_sequence, Ordering::Release);
        commit_seq
    }

    #[inline]
    pub fn abort_transaction(&self, txn_id: i64) {
        let mut state = self.shared.state.lock();
        self.wait_for_ddl(&mut state);
        let registration = state.next_registration;
        if let Some(txn) = state.transactions.get_mut(txn_id) {
            if !txn.is_active_or_committing() {
                return;
            }
            let unpublished = txn.status() == TxnStatus::Active;
            if txn.is_snapshot() {
                self.shared
                    .snapshot_txn_count
                    .fetch_sub(1, Ordering::Relaxed);
            }
            *txn = TxnState::new_aborted();
            // Active transactions have not published global versions. A failed
            // Committing transaction needs explicit successful-undo acknowledgment.
            if unpublished {
                txn.acknowledge_rollback(registration);
            }
            state.begin_exclusions.remove(txn_id);
            state.remove_committing(txn_id);
            self.shared.active_txn_count.fetch_sub(1, Ordering::Relaxed);
            state.reset_retention_if_idle(self.shared.active_txn_count.load(Ordering::Relaxed));
        }
    }

    /// Call only after all published versions have been successfully undone and
    /// transaction-local state discarded. A pre-ack captured tree can still own
    /// an aborted head, so GC waits for every pre-ack read/begin registration.
    /// Repeated acknowledgments do not advance the retirement watermark.
    pub fn acknowledge_rollback(&self, txn_id: i64) {
        let mut state = self.shared.state.lock();
        let registration = state.next_registration;
        if let Some(txn) = state
            .transactions
            .get_mut(txn_id)
            .filter(|txn| txn.is_aborted())
        {
            txn.acknowledge_rollback(registration);
        }
    }

    /// Startup-only recovery keeps assignment watermarks and mappings atomic.
    /// WAL also contains negative system identities (recovery and DDL); these
    /// advance the sequence watermark without consuming user transaction IDs.
    pub fn recover_committed_transaction(&self, txn_id: i64, commit_seq: i64) {
        assert!(
            (0..=SEQ_MASK).contains(&commit_seq),
            "invalid recovered transaction: id={txn_id}, sequence={commit_seq}"
        );
        let mut state = self.shared.state.lock();
        state.advance_logical_generation();
        if let Some(txn) = state.transactions.remove(txn_id) {
            if txn.is_active_or_committing() {
                self.shared.active_txn_count.fetch_sub(1, Ordering::Relaxed);
                if txn.is_snapshot() {
                    self.shared
                        .snapshot_txn_count
                        .fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
        state.begin_exclusions.remove(txn_id);
        state.remove_committing(txn_id);
        state.snapshot_seqs.insert(txn_id, commit_seq);
        state.next_txn_id = state.next_txn_id.max(txn_id);
        state.next_sequence = state.next_sequence.max(commit_seq);
        self.shared
            .current_sequence
            .store(state.next_sequence, Ordering::Release);
        self.shared.committed_cache.invalidate(txn_id);
        state.reset_retention_if_idle(self.shared.active_txn_count.load(Ordering::Relaxed));
    }

    /// Startup-only recovery. Acknowledgment remains separate until recovery
    /// has confirmed that the transaction's published state is absent.
    pub fn recover_aborted_transaction(&self, txn_id: i64) {
        assert!(txn_id > 0, "invalid recovered transaction");
        let mut state = self.shared.state.lock();
        state.advance_logical_generation();
        if let Some(txn) = state.transactions.insert(txn_id, TxnState::new_aborted()) {
            if txn.is_active_or_committing() {
                self.shared.active_txn_count.fetch_sub(1, Ordering::Relaxed);
                if txn.is_snapshot() {
                    self.shared
                        .snapshot_txn_count
                        .fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
        state.begin_exclusions.remove(txn_id);
        state.remove_committing(txn_id);
        state.snapshot_seqs.remove(txn_id);
        state.next_txn_id = state.next_txn_id.max(txn_id);
        self.shared.committed_cache.invalidate(txn_id);
        state.reset_retention_if_idle(self.shared.active_txn_count.load(Ordering::Relaxed));
    }

    #[inline(always)]
    pub fn is_visible(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        if version_txn_id == viewer_txn_id || version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        if self.shared.snapshot_txn_count.load(Ordering::Relaxed) == 0 {
            return self.check_committed(version_txn_id);
        }
        let state = self.shared.state.lock();
        if let Some(txn) = state
            .transactions
            .get(viewer_txn_id)
            .filter(|txn| txn.is_snapshot())
        {
            let excluded = state
                .begin_exclusions
                .get(viewer_txn_id)
                .map_or(&[][..], |seqs| seqs.as_ref());
            state.visible_at_parts(version_txn_id, txn.begin_seq(), excluded)
        } else {
            state.committed(version_txn_id)
        }
    }

    #[inline(always)]
    fn check_committed(&self, txn_id: i64) -> bool {
        if self.shared.committed_cache.contains(txn_id) {
            return true;
        }
        let state = self.shared.state.lock();
        if state.committed(txn_id) {
            self.shared.committed_cache.insert(txn_id);
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn is_directly_visible(&self, version_txn_id: i64) -> bool {
        version_txn_id == RECOVERY_TRANSACTION_ID || self.check_committed(version_txn_id)
    }

    /// Legacy API: zero means committed before the retained mapping floor.
    /// Fixed epochs/build leases use visible_at and never treat zero as an
    /// exact sequence that can be compared with an arbitrary historical cutoff.
    pub fn get_commit_sequence(&self, txn_id: i64) -> Option<i64> {
        let state = self.shared.state.lock();
        state
            .committed(txn_id)
            .then(|| state.snapshot_seqs.get(txn_id).copied().unwrap_or(0))
    }

    pub fn get_transaction_begin_sequence(&self, txn_id: i64) -> i64 {
        self.shared
            .state
            .lock()
            .transactions
            .get(txn_id)
            .map_or(0, TxnState::begin_seq)
    }

    pub fn get_committing_sequence(&self, txn_id: i64) -> i64 {
        self.shared
            .state
            .lock()
            .transactions
            .get(txn_id)
            .filter(|txn| txn.status() == TxnStatus::Committing)
            .map_or(0, TxnState::commit_seq)
    }

    pub fn get_current_sequence(&self) -> i64 {
        self.shared.current_sequence.load(Ordering::Acquire)
    }
    pub fn current_commit_sequence(&self) -> i64 {
        self.get_current_sequence()
    }

    pub fn safe_snapshot_cutoff(&self) -> i64 {
        self.shared.state.lock().safe_completed_cutoff()
    }

    /// Registration, lifecycle changes, horizon selection, and pruning cannot
    /// interleave: all operate under this same mutex.
    pub fn run_gc(&self) -> usize {
        self.run_gc_after_horizon(|| {})
    }

    fn run_gc_after_horizon(&self, after_horizon: impl FnOnce()) -> usize {
        let mut state = self.shared.state.lock();
        let horizon = state
            .refresh_retention_horizon()
            .unwrap_or(state.next_sequence);
        let oldest_registration = state.oldest_registration();
        after_horizon();
        let old_transactions = state.transactions.len();
        state.transactions.retain(|_, txn| {
            !txn.rollback_acknowledgment()
                .is_some_and(|ack| oldest_registration.is_none_or(|oldest| oldest > ack))
        });
        let old_sequences = state.snapshot_seqs.len();
        let mut discarded_through = state.discarded_through;
        state.snapshot_seqs.retain(|_, seq| {
            if *seq <= horizon {
                discarded_through = discarded_through.max(*seq);
                false
            } else {
                true
            }
        });
        state.discarded_through = discarded_through;
        old_transactions - state.transactions.len() + old_sequences - state.snapshot_seqs.len()
    }

    pub fn cleanup_old_transactions(&self, _max_age: std::time::Duration) -> i32 {
        self.run_gc() as i32
    }

    pub fn wait_for_active_transactions(&self, timeout: std::time::Duration) -> i32 {
        let deadline = crate::common::time_compat::Instant::now() + timeout;
        loop {
            if crate::common::time_compat::Instant::now() > deadline {
                break;
            }
            if self.active_count() == 0 {
                return 0;
            }
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(10));
            #[cfg(target_arch = "wasm32")]
            break;
        }
        self.active_count() as i32
    }

    pub fn stop_accepting_transactions(&self) {
        let _state = self.shared.state.lock();
        self.shared.accepting.store(false, Ordering::Release);
    }
    pub fn start_accepting_transactions(&self) {
        let _state = self.shared.state.lock();
        self.shared.accepting.store(true, Ordering::Release);
    }
    pub fn shutdown(&self) {
        self.stop_accepting_transactions();
    }
    pub fn is_accepting(&self) -> bool {
        self.shared.accepting.load(Ordering::Acquire)
    }
    pub fn active_count(&self) -> usize {
        self.shared.active_txn_count.load(Ordering::Relaxed)
    }
    pub fn active_transaction_ids(&self) -> Vec<i64> {
        self.shared
            .state
            .lock()
            .transactions
            .iter()
            .filter(|(_, txn)| txn.is_active_or_committing())
            .map(|(id, _)| id)
            .collect()
    }
    pub fn committed_count(&self) -> usize {
        self.shared.state.lock().snapshot_seqs.len()
    }
    pub fn is_active(&self, txn_id: i64) -> bool {
        self.shared
            .state
            .lock()
            .transactions
            .get(txn_id)
            .is_some_and(|txn| txn.status() == TxnStatus::Active)
    }
    pub fn is_committed(&self, txn_id: i64) -> bool {
        self.check_committed(txn_id)
    }
    #[cfg(test)]
    pub fn is_committing(&self, txn_id: i64) -> bool {
        self.shared
            .state
            .lock()
            .transactions
            .get(txn_id)
            .is_some_and(|txn| txn.status() == TxnStatus::Committing)
    }
    pub fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        if txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        self.shared.state.lock().visible_at(
            txn_id,
            &EpochData {
                cutoff: cutoff_commit_seq,
                cache_generation: 0,
                excluded: None,
            },
        )
    }
}

impl Default for TransactionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VisibilityChecker for TransactionRegistry {
    fn begin_logical_mutation(&self) -> Option<LogicalMutationGuard> {
        Some(TransactionRegistry::begin_logical_mutation(self))
    }

    fn capture_current_read_epoch(&self) -> Option<ReadEpoch> {
        Some(self.capture_read_epoch())
    }

    fn has_retention_obligations(&self) -> bool {
        TransactionRegistry::has_retention_obligations(self)
    }

    fn oldest_retention_horizon(&self) -> Option<i64> {
        TransactionRegistry::oldest_retention_horizon(self)
    }
    fn cached_retention_horizon(&self) -> Option<i64> {
        TransactionRegistry::cached_retention_horizon(self)
    }
    fn history_retention(
        &self,
        creators: &[i64],
    ) -> crate::storage::mvcc::version_store::HistoryRetention {
        TransactionRegistry::history_retention(self, creators)
    }

    fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
        TransactionRegistry::is_visible(self, version_txn_id, viewing_txn_id)
    }
    fn get_current_sequence(&self) -> i64 {
        self.get_current_sequence()
    }
    fn get_active_transaction_ids(&self) -> Vec<i64> {
        self.active_transaction_ids()
    }
    fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        self.is_committed_before(txn_id, cutoff_commit_seq)
    }
    fn needs_snapshot_isolation(&self, txn_id: i64) -> bool {
        self.needs_snapshot_isolation(txn_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_retention_floor_lowers_on_registration_and_refreshes_after_overlap() {
        let registry = TransactionRegistry::new();
        assert_eq!(registry.cached_retention_horizon(), None);
        let (older, old_begin) = registry.begin_transaction();
        let (newer, new_begin) = registry.begin_transaction();
        assert_eq!(registry.cached_retention_horizon(), Some(old_begin));
        registry.abort_transaction(older);
        assert_eq!(registry.cached_retention_horizon(), Some(old_begin));
        assert_eq!(registry.refresh_retention_horizon(), Some(new_begin));
        assert_eq!(registry.cached_retention_horizon(), Some(new_begin));
        registry.start_commit(newer);
        let epoch = registry.capture_read_epoch();
        registry.complete_commit(newer);
        assert!(registry.cached_retention_horizon().unwrap() <= epoch.retention_horizon());
        assert!(!epoch.is_visible(newer));
        registry.run_gc();
        assert_eq!(
            registry.cached_retention_horizon(),
            Some(epoch.retention_horizon())
        );
        let held = epoch.clone();
        drop(epoch);
        assert!(registry.cached_retention_horizon().is_some());
        drop(held);
        assert_eq!(registry.cached_retention_horizon(), None);
        let build = registry.register_build_lease();
        assert!(registry.cached_retention_horizon().is_some());
        drop(build);
        assert_eq!(registry.cached_retention_horizon(), None);
    }

    #[test]
    fn cached_retention_floor_resets_for_serial_writers_and_recovery() {
        let registry = TransactionRegistry::new();
        for _ in 0..128 {
            let (txn, begin) = registry.begin_transaction();
            assert_eq!(registry.cached_retention_horizon(), Some(begin));
            registry.commit_transaction(txn);
            assert_eq!(registry.cached_retention_horizon(), None);
        }
        let (committed, _) = registry.begin_transaction();
        let commit_seq = registry.start_commit(committed);
        registry.recover_committed_transaction(committed, commit_seq);
        assert_eq!(registry.cached_retention_horizon(), None);
        let (aborted, _) = registry.begin_transaction();
        registry.recover_aborted_transaction(aborted);
        assert_eq!(registry.cached_retention_horizon(), None);
    }

    #[test]
    fn retention_presence_tracks_transactions_and_external_leases() {
        let registry = TransactionRegistry::new();
        assert!(!registry.has_retention_obligations());
        let (txn, _) = registry.begin_transaction();
        assert!(registry.has_retention_obligations());
        registry.start_commit(txn);
        assert!(registry.has_retention_obligations());
        let epoch = registry.capture_read_epoch();
        registry.abort_transaction(txn);
        assert!(registry.has_retention_obligations());
        drop(epoch);
        assert!(!registry.has_retention_obligations());
        let build = registry.register_build_lease();
        assert!(registry.has_retention_obligations());
        drop(build);
        assert!(!registry.has_retention_obligations());
    }

    #[test]
    fn test_begin_transaction() {
        let registry = TransactionRegistry::new();

        let (txn_id1, seq1) = registry.begin_transaction();
        assert!(txn_id1 > 0);
        assert!(seq1 > 0);

        let (txn_id2, seq2) = registry.begin_transaction();
        assert!(txn_id2 > txn_id1);
        assert!(seq2 > seq1);
    }

    #[test]
    fn test_commit_transaction() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        assert!(registry.is_active(txn_id));
        assert!(!registry.is_committed(txn_id));

        registry.commit_transaction(txn_id);
        assert!(!registry.is_active(txn_id));
        assert!(registry.is_committed(txn_id));
    }

    #[test]
    fn test_two_phase_commit() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();

        let commit_seq = registry.start_commit(txn_id);
        assert!(commit_seq > 0);
        assert!(!registry.is_active(txn_id));
        assert!(registry.is_committing(txn_id));

        registry.complete_commit(txn_id);
        assert!(!registry.is_committing(txn_id));
        assert!(registry.is_committed(txn_id));
    }

    #[test]
    fn test_abort_transaction() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        assert!(registry.is_active(txn_id));

        registry.abort_transaction(txn_id);
        assert!(!registry.is_active(txn_id));
        assert!(!registry.is_committed(txn_id));

        // Verify aborted status
        let state = registry
            .shared
            .state
            .lock()
            .transactions
            .get(txn_id)
            .copied();
        assert!(state.map(|s| s.is_aborted()).unwrap_or(false));
    }

    #[test]
    fn test_visibility_own_writes() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        assert!(registry.is_visible(txn_id, txn_id));
    }

    #[test]
    fn test_visibility_recovery_transaction() {
        let registry = TransactionRegistry::new();

        let (viewer_id, _) = registry.begin_transaction();
        assert!(registry.is_visible(RECOVERY_TRANSACTION_ID, viewer_id));
    }

    #[test]
    fn test_visibility_read_committed() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::ReadCommitted);

        let (txn1, _) = registry.begin_transaction();
        let (txn2, _) = registry.begin_transaction();

        // Active transaction not visible
        assert!(!registry.is_visible(txn1, txn2));

        // After commit, visible
        registry.commit_transaction(txn1);
        assert!(registry.is_visible(txn1, txn2));
    }

    #[test]
    fn test_visibility_snapshot_isolation() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn1, _) = registry.begin_transaction();
        registry.commit_transaction(txn1);

        let (txn2, _) = registry.begin_transaction();

        // txn1 committed before txn2 began - visible
        assert!(registry.is_visible(txn1, txn2));

        let (txn3, _) = registry.begin_transaction();
        registry.commit_transaction(txn3);

        // txn3 committed after txn2 began - NOT visible
        assert!(!registry.is_visible(txn3, txn2));
    }

    #[test]
    fn test_stop_accepting() {
        let registry = TransactionRegistry::new();
        assert!(registry.is_accepting());

        registry.stop_accepting_transactions();
        assert!(!registry.is_accepting());

        let (txn_id, _) = registry.begin_transaction();
        assert_eq!(txn_id, INVALID_TRANSACTION_ID);
    }

    #[test]
    fn test_isolation_level_override() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::ReadCommitted);

        let (txn_id, _) = registry.begin_transaction();

        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::ReadCommitted
        );

        registry.set_transaction_isolation_level(txn_id, IsolationLevel::SnapshotIsolation);
        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::SnapshotIsolation
        );

        registry.remove_transaction_isolation_level(txn_id);
        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::ReadCommitted
        );
    }

    #[test]
    fn test_get_commit_sequence() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn_id, _) = registry.begin_transaction();
        assert!(registry.get_commit_sequence(txn_id).is_none());

        let commit_seq = registry.commit_transaction(txn_id);
        assert_eq!(registry.get_commit_sequence(txn_id), Some(commit_seq));
    }

    #[test]
    fn test_recover_committed_transaction() {
        let registry = TransactionRegistry::new();

        registry.recover_committed_transaction(1000, 500);

        assert!(registry.is_committed(1000));
        assert_eq!(registry.get_commit_sequence(1000), Some(500));

        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 1000);
    }

    #[test]
    fn test_gc() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        for _ in 0..10 {
            let (txn_id, _) = registry.begin_transaction();
            registry.commit_transaction(txn_id);
        }

        assert_eq!(registry.shared.state.lock().snapshot_seqs.len(), 10);

        let (active_txn, _) = registry.begin_transaction();

        for _ in 0..5 {
            let (txn_id, _) = registry.begin_transaction();
            registry.commit_transaction(txn_id);
        }

        let removed = registry.run_gc();
        assert!(removed > 0);

        registry.commit_transaction(active_txn);
    }

    #[test]
    fn test_aborted_not_visible() {
        let registry = TransactionRegistry::new();

        let (txn1, _) = registry.begin_transaction();
        let (txn2, _) = registry.begin_transaction();

        registry.abort_transaction(txn1);

        assert!(!registry.is_visible(txn1, txn2));
        assert!(!registry.is_committed(txn1));
    }

    #[test]
    fn test_per_transaction_snapshot_isolation() {
        let registry = TransactionRegistry::new();
        // Global is READ COMMITTED
        registry.set_global_isolation_level(IsolationLevel::ReadCommitted);

        let (txn1, _) = registry.begin_transaction();
        registry.commit_transaction(txn1);

        // txn2 uses SNAPSHOT ISOLATION override
        let (txn2, _) = registry.begin_transaction();
        registry.set_transaction_isolation_level(txn2, IsolationLevel::SnapshotIsolation);

        // txn1 committed before txn2 began - visible
        assert!(registry.is_visible(txn1, txn2));

        let (txn3, _) = registry.begin_transaction();
        registry.commit_transaction(txn3);

        // txn3 committed after txn2 began - NOT visible (snapshot isolation)
        assert!(!registry.is_visible(txn3, txn2));

        // But txn4 (READ COMMITTED) should see txn3
        let (txn4, _) = registry.begin_transaction();
        assert!(registry.is_visible(txn3, txn4));
    }

    // === commit_transaction: line 371 override_count > 0 ===

    #[test]
    fn test_commit_read_committed_skips_snapshot_seqs() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::ReadCommitted);

        let (txn_id, _) = registry.begin_transaction();
        registry.commit_transaction(txn_id);

        // override_count == 0, global != snapshot => snapshot_seqs NOT populated
        assert!(registry.shared.state.lock().snapshot_seqs.is_empty());
    }

    // === recover_committed_transaction: lines 416, 432 ===

    #[test]
    fn test_recover_committed_advances_next_txn_id() {
        let registry = TransactionRegistry::new();

        // Recover txn 100 with commit_seq 50
        registry.recover_committed_transaction(100, 50);
        // next_txn_id must advance past 100
        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 100);
    }

    #[test]
    fn test_recover_committed_advances_next_sequence() {
        let registry = TransactionRegistry::new();

        registry.recover_committed_transaction(10, 200);
        // next_sequence must advance past 200
        let (_, begin_seq) = registry.begin_transaction();
        assert!(begin_seq > 200);
    }

    #[test]
    fn test_recover_committed_descending_order() {
        let registry = TransactionRegistry::new();

        // Recover in descending order — next_txn_id must still track the max
        registry.recover_committed_transaction(100, 50);
        registry.recover_committed_transaction(50, 30);

        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 100, "next_txn_id should be >= 100, got {}", new_id);
    }

    // === recover_aborted_transaction: lines 448, 455 ===

    #[test]
    fn test_recover_aborted_marks_aborted() {
        let registry = TransactionRegistry::new();

        registry.recover_aborted_transaction(42);

        // Must be aborted, not committed, not active
        assert!(!registry.is_committed(42));
        assert!(!registry.is_active(42));
        let state = registry.shared.state.lock().transactions.get(42).copied();
        assert!(state.is_some());
        assert!(state.unwrap().is_aborted());
    }

    #[test]
    fn test_recover_aborted_advances_next_txn_id() {
        let registry = TransactionRegistry::new();

        registry.recover_aborted_transaction(100);
        // next_txn_id must advance past 100
        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 100);
    }

    #[test]
    fn test_recover_aborted_descending_order() {
        let registry = TransactionRegistry::new();

        registry.recover_aborted_transaction(200);
        registry.recover_aborted_transaction(100);

        // next_txn_id must still be >= 200
        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 200, "next_txn_id should be > 200, got {}", new_id);
    }

    #[test]
    fn test_recover_aborted_not_visible() {
        let registry = TransactionRegistry::new();

        registry.recover_aborted_transaction(5);

        let (viewer, _) = registry.begin_transaction();
        // Aborted txn must never be visible
        assert!(!registry.is_visible(5, viewer));
    }

    // === check_committed: line 512 ===

    #[test]
    fn test_check_committed_negative_txn_id_not_committed() {
        let registry = TransactionRegistry::new();

        // Start a real transaction so next_txn_id > 0
        let (txn_id, _) = registry.begin_transaction();
        registry.commit_transaction(txn_id);

        // Negative txn_id must NOT be committed.
        // Catches && -> || mutation: (-5 > 0 || -5 <= next) would wrongly be true
        assert!(!registry.check_committed(-5));
        assert!(!registry.check_committed(-100));
    }

    #[test]
    fn test_check_committed_future_txn_id() {
        let registry = TransactionRegistry::new();

        // No transactions started yet, so txn_id 1 is beyond next_txn_id
        assert!(!registry.check_committed(1));
    }

    #[test]
    fn test_check_committed_valid_committed() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn_id, _) = registry.begin_transaction();
        registry.commit_transaction(txn_id);

        assert!(registry.check_committed(txn_id));
    }

    #[test]
    fn test_check_committed_boundary_next_txn_id() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        registry.commit_transaction(txn_id);

        // txn_id == next_txn_id should be committed (boundary: <= next)
        assert!(registry.check_committed(txn_id));
        // txn_id + 1 > next_txn_id should NOT be committed
        assert!(!registry.check_committed(txn_id + 1));
    }

    // === is_directly_visible: line 523 ===

    #[test]
    fn test_is_directly_visible_recovery() {
        let registry = TransactionRegistry::new();

        // RECOVERY_TRANSACTION_ID is always directly visible
        assert!(registry.is_directly_visible(RECOVERY_TRANSACTION_ID));
    }

    #[test]
    fn test_is_directly_visible_normal_txn() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn_id, _) = registry.begin_transaction();
        // Active txn is NOT directly visible
        assert!(!registry.is_directly_visible(txn_id));

        registry.commit_transaction(txn_id);
        // Committed txn IS directly visible
        assert!(registry.is_directly_visible(txn_id));
    }

    #[test]
    fn test_is_directly_visible_non_recovery_negative() {
        let registry = TransactionRegistry::new();

        // A negative txn_id that is NOT RECOVERY_TRANSACTION_ID should not be visible
        assert!(!registry.is_directly_visible(-99));
    }

    // === is_visible_snapshot: lines 541, 570 ===

    #[test]
    fn test_snapshot_committed_viewer_fallback() {
        // When the viewer has already committed, the match guard
        // `is_active_or_committing()` fails → falls back to check_committed.
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn1, _) = registry.begin_transaction();
        registry.commit_transaction(txn1);

        let (txn2, _) = registry.begin_transaction();
        registry.commit_transaction(txn2);

        // txn1 is committed → visible to committed viewer txn2
        assert!(registry.is_visible(txn1, txn2));
    }

    #[test]
    fn test_snapshot_invalid_version_txn_zero() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (viewer, _) = registry.begin_transaction();

        // version_txn_id 0 is invalid — not visible
        // Catches || → && mutation: (0 <= 0 && 0 > next) = (true && false) = false
        // but || → && would never return false for valid-looking IDs
        assert!(!registry.is_visible(0, viewer));
    }

    #[test]
    fn test_snapshot_invalid_version_txn_negative() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (viewer, _) = registry.begin_transaction();

        // Negative txn_id (not RECOVERY_TRANSACTION_ID) is invalid
        assert!(!registry.is_visible(-50, viewer));
    }

    #[test]
    fn test_snapshot_future_version_txn_not_visible() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (viewer, _) = registry.begin_transaction();

        // version_txn_id beyond next_txn_id is invalid
        // Catches > → == mutation: 9999 == next would be false (not caught),
        // but next+1 is immediately beyond
        let next = registry.shared.state.lock().next_txn_id;
        assert!(!registry.is_visible(next + 1, viewer));
        assert!(!registry.is_visible(next + 100, viewer));
    }

    #[test]
    fn test_snapshot_boundary_version_equals_next() {
        // version_txn_id == next_txn_id should be valid (it's an assigned ID)
        // Catches > → >= mutation at line 570
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn1, _) = registry.begin_transaction();
        registry.commit_transaction(txn1);

        // txn1 == next_txn_id at this point
        let (viewer, _) = registry.begin_transaction();
        // txn1 committed before viewer — should be visible
        assert!(registry.is_visible(txn1, viewer));
    }

    // === get_commit_sequence: line 608 ===

    #[test]
    fn test_get_commit_sequence_invalid_txn_id() {
        let registry = TransactionRegistry::new();

        // txn_id 0 is invalid
        assert_eq!(registry.get_commit_sequence(0), None);
        // Negative is invalid
        assert_eq!(registry.get_commit_sequence(-1), None);
    }

    #[test]
    fn test_get_commit_sequence_future_txn_id() {
        let registry = TransactionRegistry::new();

        // No transactions started, txn_id 1 is beyond next_txn_id
        assert_eq!(registry.get_commit_sequence(1), None);
    }

    #[test]
    fn test_get_commit_sequence_active() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        // Active transaction has no commit_seq
        assert_eq!(registry.get_commit_sequence(txn_id), None);
    }

    #[test]
    fn test_get_commit_sequence_aborted() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        registry.abort_transaction(txn_id);
        // Aborted transaction has no commit_seq
        assert_eq!(registry.get_commit_sequence(txn_id), None);
    }

    #[test]
    fn test_get_commit_sequence_committed_with_snapshot() {
        let registry = TransactionRegistry::new();
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        let (txn_id, _) = registry.begin_transaction();
        let commit_seq = registry.commit_transaction(txn_id);

        // With snapshot isolation, exact commit_seq is stored
        assert_eq!(registry.get_commit_sequence(txn_id), Some(commit_seq));
        assert!(commit_seq > 0);
    }
}

#[cfg(test)]
mod epoch_tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn committed_cache_never_aliases_independent_registries() {
        let first = TransactionRegistry::new();
        let second = TransactionRegistry::new();
        let (a, _) = first.begin_transaction();
        first.commit_transaction(a);
        assert!(first.is_directly_visible(a));
        let (b, _) = second.begin_transaction();
        assert_eq!(a, b);
        assert!(!second.is_directly_visible(b));
        second.start_commit(b);
        assert!(!second.is_directly_visible(b));
        second.abort_transaction(b);
        assert!(!second.is_directly_visible(b));
        assert!(first.is_directly_visible(a));
        // Recovery can replace a cached status before serving the recovered DB.
        first.recover_aborted_transaction(a);
        assert!(!first.is_directly_visible(a));
    }

    #[test]
    fn in_flight_exclusions_survive_completion_gc_and_reader_cache() {
        let registry = TransactionRegistry::new();
        let (a, _) = registry.begin_transaction();
        let a_seq = registry.start_commit(a);
        let (b, _) = registry.begin_transaction();
        let b_seq = registry.commit_transaction(b);
        let epoch = registry.capture_read_epoch();
        assert_eq!(epoch.cutoff(), b_seq);
        assert_eq!(epoch.excluded_sequences(), &[a_seq]);
        assert_eq!(epoch.retention_horizon(), a_seq - 1);
        let mut reader = epoch.reader();
        assert!(!reader.is_visible(a));
        assert!(reader.is_visible(b));
        registry.complete_commit(a);
        registry.run_gc();
        assert!(!epoch.is_visible(a));
        assert!(!reader.is_visible(a));
        assert!(reader.is_visible(b));
        assert!(!epoch.admits_commit_sequence(a_seq));
        assert!(epoch.admits_commit_sequence(b_seq));
        assert_eq!(registry.get_commit_sequence(a), Some(a_seq));
        let build = registry.register_build_lease();
        assert_eq!(build.cutoff(), a_seq - 1);
        assert!(!build.is_eligible(a));
        assert!(!build.is_eligible(b));
    }

    #[test]
    fn active_at_capture_remains_invisible_after_later_commit() {
        let registry = TransactionRegistry::new();
        let (writer, _) = registry.begin_transaction();
        let epoch = registry.capture_read_epoch();
        let mut reader = epoch.reader();
        assert!(!reader.is_visible(writer));
        let sequence = registry.commit_transaction(writer);
        assert!(sequence > epoch.cutoff());
        assert!(!epoch.is_visible(writer));
        assert!(!reader.is_visible(writer));
        assert!(registry.capture_read_epoch().is_visible(writer));
    }

    #[test]
    fn snapshot_begin_is_atomic_and_legacy_promotion_preserves_original_exclusions() {
        let registry = TransactionRegistry::new();
        let (writer, _) = registry.begin_transaction();
        let writer_seq = registry.start_commit(writer);
        let (snapshot, begin_seq) =
            registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let (legacy, _) = registry.begin_transaction();
        registry.complete_commit(writer);
        registry.run_gc();
        registry.set_transaction_isolation_level(legacy, IsolationLevel::SnapshotIsolation);
        assert!(!registry.is_visible(writer, snapshot));
        assert!(!registry.is_visible(writer, legacy));
        let epoch = registry.read_epoch_for_transaction(snapshot).unwrap();
        assert_eq!(epoch.cutoff(), begin_seq);
        assert_eq!(epoch.excluded_sequences(), &[writer_seq]);
        assert!(!epoch.is_visible(writer));
        // Changing the default affects only future transactions.
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);
        let (rc, _) = registry.begin_transaction_with_isolation(IsolationLevel::ReadCommitted);
        assert_eq!(
            registry.get_isolation_level(rc),
            IsolationLevel::ReadCommitted
        );
        assert!(registry.is_visible(writer, rc));
    }

    #[test]
    fn read_committed_statements_advance_but_snapshot_statements_do_not() {
        let registry = TransactionRegistry::new();
        let (rc, _) = registry.begin_transaction();
        let (si, _) = registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let rc_first = registry.read_epoch_for_transaction(rc).unwrap();
        let si_first = registry.read_epoch_for_transaction(si).unwrap();
        let (writer, _) = registry.begin_transaction();
        registry.commit_transaction(writer);
        assert!(!rc_first.is_visible(writer));
        assert!(!si_first.is_visible(writer));
        assert!(registry
            .read_epoch_for_transaction(rc)
            .unwrap()
            .is_visible(writer));
        assert!(!registry
            .read_epoch_for_transaction(si)
            .unwrap()
            .is_visible(writer));
        registry.abort_transaction(rc);
        assert!(registry.read_epoch_for_transaction(rc).is_none());
        assert!(!rc_first.is_visible(writer));
    }

    #[test]
    fn rc_read_lease_retains_commit_mappings_until_final_drop() {
        let registry = TransactionRegistry::new();
        let epoch = registry.capture_read_epoch();
        let clone = epoch.clone();
        let (writer, _) = registry.begin_transaction();
        let sequence = registry.commit_transaction(writer);
        registry.run_gc();
        assert_eq!(registry.get_commit_sequence(writer), Some(sequence));
        drop(epoch);
        registry.run_gc();
        assert_eq!(registry.get_commit_sequence(writer), Some(sequence));
        drop(clone);
        registry.run_gc();
        assert_eq!(registry.get_commit_sequence(writer), Some(0));
        // An unleased historical cutoff must not treat unknown commit time as zero.
        assert!(!registry.is_committed_before(writer, sequence - 1));
        assert!(registry.is_committed_before(writer, sequence));
    }

    #[test]
    fn leases_release_horizons_and_can_outlive_registry_facade() {
        let registry = TransactionRegistry::new();
        let epoch = registry.capture_read_epoch();
        let clone = epoch.clone();
        let build = registry.register_build_lease();
        assert_eq!(registry.oldest_retention_horizon(), Some(0));
        drop(epoch);
        drop(build);
        assert_eq!(registry.oldest_retention_horizon(), Some(0));
        let weak = Arc::downgrade(&registry.shared);
        drop(registry);
        assert!(weak.upgrade().is_some());
        assert!(!clone.is_visible(1));
        drop(clone);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn build_bound_is_fixed_and_uses_compatibility_begin_horizon() {
        let registry = TransactionRegistry::new();
        let (old, _) = registry.begin_transaction();
        registry.commit_transaction(old);
        let (idle_rc, begin) = registry.begin_transaction();
        let build = registry.register_build_lease();
        assert_eq!(build.cutoff(), begin);
        assert!(build.is_eligible(old));
        let (new, _) = registry.begin_transaction();
        registry.commit_transaction(new);
        registry.abort_transaction(idle_rc);
        registry.run_gc();
        assert_eq!(build.cutoff(), begin);
        assert!(!build.is_eligible(new));
        assert_eq!(registry.oldest_retention_horizon(), Some(begin));
        drop(build);
        assert_eq!(registry.oldest_retention_horizon(), None);
    }

    #[test]
    fn gc_horizon_pruning_is_atomic_with_new_snapshot_registration() {
        let registry = Arc::new(TransactionRegistry::new());
        registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);
        let (old, _) = registry.begin_transaction();
        registry.commit_transaction(old);
        let (horizon_tx, horizon_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (begun_tx, begun_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let gc_registry = Arc::clone(&registry);
            scope.spawn(move || {
                gc_registry.run_gc_after_horizon(|| {
                    horizon_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                })
            });
            horizon_rx.recv().unwrap();
            let reader_registry = Arc::clone(&registry);
            scope.spawn(move || {
                attempt_tx.send(()).unwrap();
                let (reader, _) = reader_registry
                    .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
                begun_tx.send(reader).unwrap();
            });
            attempt_rx.recv().unwrap();
            assert!(matches!(
                begun_rx.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            resume_tx.send(()).unwrap();
            let reader = begun_rx.recv().unwrap();
            let (writer, _) = registry.begin_transaction();
            let sequence = registry.commit_transaction(writer);
            registry.run_gc();
            assert_eq!(registry.get_commit_sequence(writer), Some(sequence));
            assert!(!registry.is_visible(writer, reader));
            registry.abort_transaction(reader);
        });
    }

    #[test]
    fn repeated_terminal_calls_keep_counts_and_commit_identity_stable() {
        let registry = TransactionRegistry::new();
        let (txn, _) = registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let sequence = registry.start_commit(txn);
        assert_eq!(registry.start_commit(txn), sequence);
        registry.complete_commit(txn);
        registry.complete_commit(txn);
        registry.abort_transaction(txn);
        assert_eq!(registry.active_count(), 0);
        assert!(!registry.has_active_snapshot_transactions());
        let (read_only, _) = registry.begin_transaction();
        registry.complete_commit(read_only);
        assert!(registry.is_committed(read_only));
        assert_eq!(registry.active_count(), 0);
        assert_eq!(registry.start_commit(read_only), 0);
    }

    #[test]
    fn rollback_ack_waits_for_pre_ack_captures_even_after_abort() {
        let registry = TransactionRegistry::new();
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        registry.abort_transaction(writer);
        // A root captured after the abort can still retain its not-yet-undone head.
        let captured = registry.capture_read_epoch();
        registry.acknowledge_rollback(writer);
        registry.run_gc();
        assert!(!registry.is_committed(writer));
        assert!(!captured.is_visible(writer));
        let new_capture = registry.capture_read_epoch();
        drop(captured);
        registry.run_gc();
        assert!(!registry
            .shared
            .state
            .lock()
            .transactions
            .contains_key(writer));
        // A post-ack capture does not prevent retirement of this marker.
        drop(new_capture);
    }

    #[test]
    fn rollback_ack_waits_for_legacy_begin_snapshot_and_unacked_markers_remain() {
        let registry = TransactionRegistry::new();
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        registry.abort_transaction(writer);
        let (reader, _) = registry.begin_transaction();
        registry.acknowledge_rollback(writer);
        registry.run_gc();
        assert!(!registry.is_visible(writer, reader));
        registry.abort_transaction(reader);
        registry.run_gc();
        assert!(!registry
            .shared
            .state
            .lock()
            .transactions
            .contains_key(writer));
        let (unacked, _) = registry.begin_transaction();
        registry.start_commit(unacked);
        registry.abort_transaction(unacked);
        registry.run_gc();
        assert!(!registry.is_committed(unacked));
        registry.acknowledge_rollback(unacked);
        registry.run_gc();
        assert!(!registry
            .shared
            .state
            .lock()
            .transactions
            .contains_key(unacked));
    }

    #[test]
    fn active_abort_is_unpublished_and_automatically_acknowledged() {
        let registry = TransactionRegistry::new();
        let (txn, _) = registry.begin_transaction();
        registry.abort_transaction(txn);
        registry.run_gc();
        assert!(!registry.shared.state.lock().transactions.contains_key(txn));
        assert_eq!(registry.active_count(), 0);
    }

    #[test]
    fn foreign_epoch_cannot_be_used_with_another_registry() {
        let first = TransactionRegistry::new();
        let second = TransactionRegistry::new();
        let epoch = first.capture_read_epoch();
        assert!(!second.is_visible_in_epoch(RECOVERY_TRANSACTION_ID, &epoch));
        assert!(first.is_visible_in_epoch(RECOVERY_TRANSACTION_ID, &epoch));
    }

    #[test]
    fn stop_accepting_is_serialized_with_begin_and_empty_exclusions_do_not_allocate() {
        let registry = TransactionRegistry::new();
        let epoch = registry.capture_read_epoch();
        assert!(epoch.lease.data.excluded.is_none());
        let (txn, _) = registry.begin_transaction();
        assert!(registry.shared.state.lock().begin_exclusions.is_empty());
        registry.stop_accepting_transactions();
        assert_eq!(registry.begin_transaction(), (INVALID_TRANSACTION_ID, 0));
        assert_eq!(registry.active_count(), 1);
        registry.abort_transaction(txn);
        registry.start_accepting_transactions();
        assert!(registry.begin_transaction().0 > txn);
    }

    #[test]
    fn sequence_assignment_and_committing_transition_exclude_epoch_capture_gap() {
        let registry = Arc::new(TransactionRegistry::new());
        let (writer, begin) = registry.begin_transaction();
        let (allocated_tx, allocated_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (captured_tx, captured_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let commit_registry = Arc::clone(&registry);
            scope.spawn(move || {
                commit_registry.start_commit_after_sequence(writer, || {
                    allocated_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                })
            });
            allocated_rx.recv().unwrap();
            assert_eq!(registry.get_current_sequence(), begin);
            let capture_registry = Arc::clone(&registry);
            scope.spawn(move || {
                attempt_tx.send(()).unwrap();
                captured_tx
                    .send(capture_registry.capture_read_epoch())
                    .unwrap();
            });
            attempt_rx.recv().unwrap();
            assert!(matches!(
                captured_rx.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            resume_tx.send(()).unwrap();
            let epoch = captured_rx.recv().unwrap();
            let sequence = registry.get_committing_sequence(writer);
            assert!(sequence > begin);
            assert_eq!(epoch.excluded_sequences(), &[sequence]);
            assert_eq!(registry.safe_snapshot_cutoff(), sequence - 1);
            registry.complete_commit(writer);
            assert!(!epoch.is_visible(writer));
        });
    }

    #[test]
    fn recovered_zero_sequence_is_prehistory_and_negative_tombstones_are_invalid() {
        let registry = TransactionRegistry::new();
        registry.recover_committed_transaction(5, 0);
        let epoch = registry.capture_read_epoch();
        assert!(epoch.is_visible(5));
        assert!(epoch.admits_commit_sequence(0));
        assert!(!epoch.admits_commit_sequence(-2));
        let build_reader = registry.register_build_lease();
        assert!(build_reader.reader().is_visible(5));
        registry.run_gc();
        assert!(epoch.is_visible(5));
    }

    #[test]
    fn snapshot_recovery_transaction_advances_sequence_without_consuming_user_identity() {
        let registry = TransactionRegistry::new();
        registry.recover_committed_transaction(RECOVERY_TRANSACTION_ID, 300);
        assert!(registry
            .capture_read_epoch()
            .is_visible(RECOVERY_TRANSACTION_ID));
        let (txn, sequence) = registry.begin_transaction();
        assert_eq!(txn, 1);
        assert!(sequence > 300);
    }

    #[test]
    fn ddl_recovery_identity_advances_watermark_without_becoming_a_visible_row_writer() {
        let registry = TransactionRegistry::new();
        // persistence::DDL_TXN_ID is -2; its COMMIT records also replay here.
        registry.recover_committed_transaction(-2, 400);
        assert!(!registry.capture_read_epoch().is_visible(-2));
        let (txn, sequence) = registry.begin_transaction();
        assert_eq!(txn, 1);
        assert!(sequence > 400);
    }
}

#[cfg(test)]
mod packing_tests {
    use super::*;

    #[test]
    fn packing_keeps_active_commit_and_abort_words_disjoint() {
        assert_eq!(std::mem::size_of::<TxnState>(), 16);
        #[repr(C)]
        struct Slot {
            key: i64,
            value: std::mem::MaybeUninit<TxnState>,
        }
        assert_eq!(std::mem::size_of::<Slot>(), 24);
        let mut active = TxnState::new_active(SEQ_MASK, true, SEQ_MASK - 1);
        assert_eq!(active.status(), TxnStatus::Active);
        assert_eq!(active.begin_seq(), SEQ_MASK);
        assert_eq!(active.commit_seq(), 0);
        assert_eq!(active.active_begin_registration(), Some(SEQ_MASK - 1));
        assert!(active.is_snapshot());
        active.set_snapshot(false);
        assert_eq!(active.active_begin_registration(), Some(SEQ_MASK - 1));
        active.set_snapshot(true);
        active.set_committing(SEQ_MASK);
        assert_eq!(active.status(), TxnStatus::Committing);
        assert_eq!(active.commit_seq(), SEQ_MASK);
        assert_eq!(active.active_begin_registration(), None);
        assert!(active.is_snapshot());
        let mut aborted = TxnState::new_aborted();
        assert_eq!(aborted.status(), TxnStatus::Aborted);
        assert_eq!(aborted.commit_seq(), 0);
        assert_eq!(aborted.active_begin_registration(), None);
        assert_eq!(aborted.rollback_acknowledgment(), None);
        aborted.acknowledge_rollback(SEQ_MASK - 1);
        aborted.acknowledge_rollback(2);
        assert_eq!(aborted.rollback_acknowledgment(), Some(SEQ_MASK - 1));
    }

    #[test]
    fn packing_preserves_begin_registration_through_inflight_spill_and_recovery() {
        let registry = TransactionRegistry::with_capacity(8);
        let txns: [(i64, i64); 6] = std::array::from_fn(|_| registry.begin_transaction());
        let oldest = registry.shared.state.lock().oldest_registration();
        let mut sequences = [0; 6];
        for i in 0..4 {
            sequences[i] = registry.start_commit(txns[i].0);
        }
        assert!(!registry.shared.state.lock().committing.spilled());
        assert_eq!(registry.shared.state.lock().oldest_registration(), oldest);
        sequences[4] = registry.start_commit(txns[4].0);
        assert!(registry.shared.state.lock().committing.spilled());
        sequences[5] = registry.start_commit(txns[5].0);
        assert_eq!(registry.start_commit(txns[4].0), sequences[4]);
        assert_eq!(registry.shared.state.lock().committing.len(), 6);
        assert_eq!(registry.shared.state.lock().oldest_registration(), oldest);
        let epoch = registry.capture_read_epoch();
        assert_eq!(epoch.excluded_sequences(), sequences);
        assert_eq!(registry.safe_snapshot_cutoff(), sequences[0] - 1);
        registry.recover_aborted_transaction(txns[0].0);
        assert_eq!(registry.shared.state.lock().committing.len(), 5);
        registry.acknowledge_rollback(txns[0].0);
        registry.recover_committed_transaction(txns[1].0, sequences[1]);
        assert_eq!(registry.shared.state.lock().committing.len(), 4);
        for (id, _) in &txns[2..] {
            registry.complete_commit(*id);
        }
        assert!(registry.shared.state.lock().committing.is_empty());
        assert_eq!(registry.active_count(), 0);
        for (id, _) in txns {
            assert!(!epoch.is_visible(id));
        }
    }

    #[test]
    fn packing_committing_old_begin_still_pins_acknowledged_abort() {
        let registry = TransactionRegistry::new();
        let (old_reader, _) = registry.begin_transaction();
        let oldest = registry.shared.state.lock().oldest_registration();
        let (victim, _) = registry.begin_transaction();
        registry.start_commit(victim);
        registry.abort_transaction(victim);
        registry.acknowledge_rollback(victim);
        // Its later commit sequence cannot replace its earlier registration.
        registry.start_commit(old_reader);
        assert_eq!(registry.shared.state.lock().oldest_registration(), oldest);
        registry.run_gc();
        assert!(!registry.is_committed(victim));
        registry.complete_commit(old_reader);
        registry.run_gc();
        assert!(!registry
            .shared
            .state
            .lock()
            .transactions
            .contains_key(victim));
    }

    #[test]
    fn packing_prepublication_unwind_keeps_active_registration() {
        let registry = TransactionRegistry::new();
        let (txn, _) = registry.begin_transaction();
        let oldest = registry.shared.state.lock().oldest_registration();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            registry.start_commit_after_sequence(txn, || panic!("before publication"));
        }));
        assert!(result.is_err());
        let state = registry.shared.state.lock();
        assert_eq!(
            state.transactions.get(txn).unwrap().status(),
            TxnStatus::Active
        );
        assert_eq!(state.transactions.get(txn).unwrap().commit_seq(), 0);
        assert_eq!(state.oldest_registration(), oldest);
        assert!(state.committing.is_empty());
        drop(state);
        assert!(registry.start_commit(txn) > 0);
        registry.complete_commit(txn);
    }
}

#[cfg(test)]
mod destructive_ddl_tests {
    use super::*;
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    #[test]
    fn ddl_rejects_other_transaction_and_delayed_read_or_build_capture() {
        let registry = TransactionRegistry::new();
        let (older, _) =
            registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let (owner, _) = registry.begin_transaction();
        assert!(registry.prepare_destructive_ddl(owner, None).is_err());
        registry.abort_transaction(older);
        let read = registry.capture_read_epoch();
        assert!(registry.prepare_destructive_ddl(owner, None).is_err());
        drop(read);
        let build = registry.register_build_lease();
        assert!(registry.prepare_destructive_ddl(owner, None).is_err());
        drop(build);
        assert!(registry.prepare_destructive_ddl(owner, None).is_ok());
    }

    #[test]
    fn ddl_private_epoch_exemption_rejects_external_alias_and_foreign_registry() {
        let registry = TransactionRegistry::new();
        let (owner, _) = registry.begin_transaction();
        let epoch = registry.read_epoch_for_transaction(owner).unwrap();
        let external = epoch.clone();
        assert!(registry
            .prepare_destructive_ddl(owner, Some((&epoch, 1)))
            .is_err());
        drop(external);
        let guard = registry
            .prepare_destructive_ddl(owner, Some((&epoch, 1)))
            .unwrap();
        assert!(
            registry.shared.state.try_lock().is_some(),
            "WAL runs without the lifecycle mutex"
        );
        drop(guard);
        let foreign = TransactionRegistry::new().capture_read_epoch();
        assert!(registry
            .prepare_destructive_ddl(owner, Some((&foreign, 1)))
            .is_err());
    }

    #[test]
    fn ddl_blocks_all_new_registration_and_isolation_paths_until_apply_finishes() {
        let registry = Arc::new(TransactionRegistry::new());
        let (owner, _) = registry.begin_transaction();
        let guard = registry.prepare_destructive_ddl(owner, None).unwrap();
        let (done, rx) = mpsc::channel();
        let start = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for mode in 0..4 {
            let registry = registry.clone();
            let done = done.clone();
            let start = start.clone();
            threads.push(std::thread::spawn(move || {
                start.wait();
                match mode {
                    0 => {
                        let _ = registry.begin_transaction();
                    }
                    1 => {
                        let _ = registry.capture_read_epoch();
                    }
                    2 => {
                        let _ = registry.register_build_lease();
                    }
                    _ => registry
                        .set_transaction_isolation_level(owner, IsolationLevel::SnapshotIsolation),
                }
                done.send(mode).unwrap();
            }));
        }
        start.wait();
        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        assert!(registry.shared.state.try_lock().is_some());
        drop(guard);
        for _ in 0..4 {
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn ddl_error_or_unwind_reopens_registration() {
        let registry = TransactionRegistry::new();
        let (owner, _) = registry.begin_transaction();
        let result: crate::core::Result<()> = (|| {
            let _guard = registry.prepare_destructive_ddl(owner, None)?;
            Err(crate::core::Error::internal("injected WAL failure"))
        })();
        assert!(result.is_err());
        assert!(!registry.shared.state.lock().destructive_ddl);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry.prepare_destructive_ddl(owner, None).unwrap();
            panic!("injected preparation panic");
        }));
        assert!(panic.is_err());
        assert!(!registry.shared.state.lock().destructive_ddl);
        let lease = registry.read_epoch_for_transaction(owner).unwrap();
        assert!(lease.cutoff() > 0);
    }
}

#[cfg(test)]
mod cache_provenance_tests {
    use super::*;

    #[test]
    fn ordinary_reads_reuse_proof_but_every_visible_outcome_changes_it() {
        let registry = TransactionRegistry::new();
        let first = registry.capture_read_epoch();
        let proof = registry.cache_provenance(&first).unwrap();
        for _ in 0..3 {
            let (reader, _) = registry.begin_transaction();
            registry.abort_transaction(reader);
            registry.acknowledge_rollback(reader);
            let next = registry.capture_read_epoch();
            assert_eq!(registry.cache_provenance(&next), Some(proof));
        }
        let (early, _) = registry.begin_transaction();
        let (late, _) = registry.begin_transaction();
        let early_seq = registry.start_commit(early);
        let late_seq = registry.start_commit(late);
        let excluded = registry.capture_read_epoch();
        assert_eq!(excluded.excluded_sequences(), [early_seq, late_seq]);
        assert_eq!(registry.cache_provenance(&excluded), Some(proof));
        registry.complete_commit(late);
        let after_late = registry.capture_read_epoch();
        let late_proof = registry.cache_provenance(&after_late).unwrap();
        assert_ne!(late_proof, proof);
        assert!(registry.cache_provenance(&excluded).is_none());
        registry.complete_commit(early);
        let after_both = registry.capture_read_epoch();
        assert_ne!(registry.cache_provenance(&after_both), Some(late_proof));
        assert!(!after_late.is_visible(early));
        assert!(after_both.is_visible(early));
        assert!(registry.cache_provenance(&first).is_none());
        let (direct, _) = registry.begin_transaction();
        registry.complete_commit(direct); // Public direct Active -> committed path.
        assert!(registry.cache_provenance(&after_both).is_none());
    }

    #[test]
    fn historical_foreign_and_in_mutation_views_never_gain_fresh_proof() {
        let registry = TransactionRegistry::new();
        let other = TransactionRegistry::new();
        let foreign = other.capture_read_epoch();
        assert!(registry.cache_provenance(&foreign).is_none());
        let (snapshot, _) =
            registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let historical = registry.read_epoch_for_transaction(snapshot).unwrap();
        assert!(registry.cache_provenance(&historical).is_none());
        let before = registry.capture_read_epoch();
        let outer = registry.begin_logical_mutation();
        let during = registry.capture_read_epoch();
        assert!(registry.cache_provenance(&before).is_none());
        assert!(registry.cache_provenance(&during).is_none());
        {
            let _inner = registry.begin_logical_mutation();
        }
        assert!(registry
            .cache_provenance(&registry.capture_read_epoch())
            .is_none());
        drop(outer);
        assert!(registry.cache_provenance(&during).is_none());
        assert!(registry.cache_provenance(&historical).is_none());
        assert!(registry
            .cache_provenance(&registry.capture_read_epoch())
            .is_some());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry.begin_logical_mutation();
            panic!("failed immediate operation");
        }));
        assert!(result.is_err());
        assert_eq!(registry.shared.state.lock().immediate_mutations, 0);
        registry.abort_transaction(snapshot);
    }

    #[test]
    fn failed_publication_guard_covers_abort_through_undo_acknowledgment() {
        let registry = TransactionRegistry::new();
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        let before_abort = registry.capture_read_epoch();
        assert!(registry.cache_provenance(&before_abort).is_some());
        let undo = registry.begin_logical_mutation();
        registry.abort_transaction(writer);
        let after_abort_before_undo = registry.capture_read_epoch();
        assert!(after_abort_before_undo.excluded_sequences().is_empty());
        assert!(registry
            .cache_provenance(&after_abort_before_undo)
            .is_none());
        registry.acknowledge_rollback(writer);
        drop(undo);
        assert!(registry
            .cache_provenance(&after_abort_before_undo)
            .is_none());
        assert!(registry.cache_provenance(&before_abort).is_none());
        assert!(registry
            .cache_provenance(&registry.capture_read_epoch())
            .is_some());
    }

    #[test]
    fn cached_scalar_does_not_keep_a_read_registration_alive() {
        let registry = TransactionRegistry::new();
        let epoch = registry.capture_read_epoch();
        let proof = registry.cache_provenance(&epoch).unwrap();
        let cache = crate::executor::semantic_cache::SemanticCache::new();
        cache.insert_with_provenance("t", vec!["id".into()], Vec::new(), None, proof);
        drop(epoch);
        let (owner, _) = registry.begin_transaction();
        assert!(registry.prepare_destructive_ddl(owner, None).is_ok());
        drop(cache);
        registry.abort_transaction(owner);
    }

    #[test]
    fn proof_exhaustion_disables_reuse_without_failing_commit_or_wrapping() {
        let registry = TransactionRegistry::new();
        registry.shared.state.lock().logical_generation = u64::MAX;
        let before = registry.capture_read_epoch();
        assert!(registry.cache_provenance(&before).is_some());
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        registry.complete_commit(writer);
        assert!(registry.is_committed(writer));
        assert_eq!(registry.shared.state.lock().logical_generation, 0);
        assert!(registry.cache_provenance(&before).is_none());
        for _ in 0..3 {
            drop(registry.begin_logical_mutation());
            assert!(registry
                .cache_provenance(&registry.capture_read_epoch())
                .is_none());
        }
    }
}

#[cfg(test)]
mod memory_ownership_tests {
    use super::*;
    use crate::common::memory::MemoryAccount;

    #[test]
    fn committing_spill_accounts_replacement_overlap_without_charging_inline_entries() {
        let account = MemoryAccount::new();
        let initial = account.snapshot().accounted_bytes;
        let registry = TransactionRegistry::new_in(&account);
        let txns: [i64; 9] = std::array::from_fn(|_| registry.begin_transaction().0);
        let before = account.snapshot();
        for &txn in &txns[..4] {
            registry.start_commit(txn);
        }
        assert_eq!(account.snapshot().accounted_bytes, before.accounted_bytes);
        registry.start_commit(txns[4]);
        let first_capacity = 8 * std::mem::size_of::<CommittingTxn>();
        assert_eq!(
            account.snapshot().retained_bytes,
            before.retained_bytes + first_capacity
        );
        for &txn in &txns[5..] {
            registry.start_commit(txn);
        }
        let second_capacity = 16 * std::mem::size_of::<CommittingTxn>();
        let after = account.snapshot();
        assert_eq!(
            after.retained_bytes,
            before.retained_bytes + second_capacity
        );
        assert!(
            after.peak_accounted_bytes >= before.accounted_bytes + first_capacity + second_capacity
        );
        for txn in txns {
            registry.complete_commit(txn);
        }
        let state = registry.shared.state.lock();
        assert!(state.committing.is_empty());
        assert_eq!(
            state.committing_charge.as_ref().unwrap().bytes(),
            second_capacity
        );
        drop(state);
        drop(registry);
        assert_eq!(account.snapshot().accounted_bytes, initial);
    }

    #[test]
    fn shared_exclusions_and_registry_state_keep_their_own_last_owner_charges() {
        let account = MemoryAccount::new();
        let initial = account.snapshot();
        let registry = TransactionRegistry::new_in(&account);
        let txns: [i64; 6] = std::array::from_fn(|_| registry.begin_transaction().0);
        for txn in txns {
            registry.start_commit(txn);
        }
        let (reader, _) =
            registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let epoch = registry.read_epoch_for_transaction(reader).unwrap();
        let exclusions = epoch.lease.data.excluded.as_ref().unwrap().clone();
        {
            let state = registry.shared.state.lock();
            assert!(CompactArc::ptr_eq(
                state.begin_exclusions.get(reader).unwrap(),
                &exclusions
            ));
            assert!(exclusions.belongs_to(&account));
        }
        for txn in txns {
            registry.complete_commit(txn);
            assert!(!epoch.is_visible(txn));
        }
        registry.abort_transaction(reader);
        registry.acknowledge_rollback(reader);
        drop(registry);
        // The read lease retains RegistryShared, but not its original facade.
        assert!(account.snapshot().retained_bytes > exclusions.allocation_size());
        drop(epoch);
        assert_eq!(
            account.snapshot().retained_bytes,
            exclusions.allocation_size()
        );
        assert_eq!(
            account.snapshot().conservative_bytes,
            initial.conservative_bytes
        );
        drop(exclusions);
        assert_eq!(account.snapshot().accounted_bytes, initial.accounted_bytes);
    }
}
