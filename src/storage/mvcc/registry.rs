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

//! Transaction registry for MVCC visibility.
//!
//! Optimized design with minimal memory footprint:
//! - Active/Committing/Aborted transactions: tracked in single map
//! - Committed transactions: implicit (not in map = committed)
//! - READ COMMITTED checks read a status table and a base, without a lock
//!
//! Memory: O(active_transactions + aborted_transactions)

use std::sync::atomic::{
    fence, AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::common::I64Map;
use crate::core::IsolationLevel;
use crate::storage::VisibilityChecker;

/// Invalid transaction ID returned when registry is not accepting new transactions.
pub const INVALID_TRANSACTION_ID: i64 = -999999999;

/// Special transaction ID for recovery transactions (always visible).
pub const RECOVERY_TRANSACTION_ID: i64 = -1;

/// Sentinel value for aborted transactions (negative begin_seq).
const ABORTED_SENTINEL: i64 = -1;

/// Transaction status.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TxnStatus {
    /// Transaction is in progress
    Active = 0,
    /// Two-phase commit in progress
    Committing = 1,
    /// Transaction was aborted
    Aborted = 2,
}

/// Packed transaction state (16 bytes).
///
/// - For Active/Committing: begin_seq > 0, state_seq encodes status + commit_seq
/// - For Aborted: begin_seq = ABORTED_SENTINEL (-1)
#[derive(Clone, Copy, Debug)]
pub struct TxnState {
    /// Sequence when transaction began (negative = aborted sentinel)
    begin_seq: i64,
    /// Packed: lower 62 bits = commit_seq, upper 2 bits = status
    state_seq: i64,
}

const STATUS_SHIFT: u32 = 62;
const SEQ_MASK: i64 = (1i64 << STATUS_SHIFT) - 1;

impl TxnState {
    /// Creates a new active transaction state.
    #[inline]
    const fn new_active(begin_seq: i64) -> Self {
        Self {
            begin_seq,
            state_seq: 0, // Active=0, commit_seq=0
        }
    }

    /// Creates an aborted transaction marker.
    #[inline]
    const fn new_aborted() -> Self {
        Self {
            begin_seq: ABORTED_SENTINEL,
            state_seq: (TxnStatus::Aborted as i64) << STATUS_SHIFT,
        }
    }

    /// Returns true if this is an aborted transaction.
    #[inline(always)]
    pub const fn is_aborted(&self) -> bool {
        self.begin_seq == ABORTED_SENTINEL
    }

    /// Returns the begin sequence (0 if aborted).
    #[inline(always)]
    pub const fn begin_seq(&self) -> i64 {
        if self.begin_seq == ABORTED_SENTINEL {
            0
        } else {
            self.begin_seq
        }
    }

    /// Returns the transaction status.
    #[inline(always)]
    pub const fn status(&self) -> TxnStatus {
        if self.begin_seq == ABORTED_SENTINEL {
            return TxnStatus::Aborted;
        }
        match (self.state_seq >> STATUS_SHIFT) as u8 {
            0 => TxnStatus::Active,
            1 => TxnStatus::Committing,
            _ => TxnStatus::Aborted,
        }
    }

    /// Returns true if Active or Committing (not aborted).
    #[inline(always)]
    pub const fn is_active_or_committing(&self) -> bool {
        self.begin_seq != ABORTED_SENTINEL
    }

    /// Sets status to Committing with given commit_seq.
    #[inline(always)]
    fn set_committing(&mut self, commit_seq: i64) {
        self.state_seq = commit_seq | (1i64 << STATUS_SHIFT);
    }

    /// Returns the commit sequence (0 if not committing).
    #[inline(always)]
    pub const fn commit_seq(&self) -> i64 {
        self.state_seq & SEQ_MASK
    }
}

/// Transaction ids per segment of the status table (128 KiB of slots)
const STATUS_SEGMENT_SHIFT: u32 = 14;
const STATUS_SEGMENT_LEN: usize = 1 << STATUS_SEGMENT_SHIFT;
/// Segments the table holds at once: a window of about four million ids
const STATUS_SEGMENTS: usize = 256;

/// A slot no transaction begun in this process wrote: the map decides
const SLOT_UNKNOWN: i64 = 0;
/// A slot whose transaction aborted
const SLOT_ABORTED: i64 = i64::MIN;
/// Ids and commit sequences stay below this, so the states below are disjoint
const SLOT_ID_LIMIT: i64 = 1 << 60;

/// The slot of active `txn_id`: carries the id, so a compare-and-swap never
/// matches another transaction that reuses the slot
#[inline(always)]
const fn slot_open(txn_id: i64) -> i64 {
    -(2 * SLOT_ID_LIMIT + txn_id)
}

/// The slot of active `txn_id` once it put versions in a shared store
#[inline(always)]
const fn slot_published(txn_id: i64) -> i64 {
    -(3 * SLOT_ID_LIMIT + txn_id)
}

/// Whether `slot` is a committing one, the negated commit sequence
#[inline(always)]
const fn slot_committing(slot: i64) -> bool {
    slot < 0 && slot > -SLOT_ID_LIMIT
}

/// The status of one run of transaction ids: a positive slot is the commit
/// sequence, a committing one its negation
struct StatusSegment {
    /// The run held, or -1 while the segment is being reused
    run: AtomicI64,
    slots: Box<[AtomicI64]>,
}

impl StatusSegment {
    fn new() -> Self {
        Self {
            run: AtomicI64::new(-1),
            slots: (0..STATUS_SEGMENT_LEN)
                .map(|_| AtomicI64::new(SLOT_UNKNOWN))
                .collect(),
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Map lookups and records visited by base advances on this thread
    static BASE_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Transaction status read without a lock. Segments are created on demand,
/// never freed, and reused once their run is below the base; a reader checks
/// the run before and after its read. Writers hold the transactions lock.
struct TxnStatusTable {
    /// Pool position + 1 of the segment holding run r, at r % STATUS_SEGMENTS
    directory: Box<[AtomicU32]>,
    pool: Box<[OnceLock<StatusSegment>]>,
}

impl TxnStatusTable {
    fn new() -> Self {
        Self {
            directory: (0..STATUS_SEGMENTS).map(|_| AtomicU32::new(0)).collect(),
            pool: (0..STATUS_SEGMENTS).map(|_| OnceLock::new()).collect(),
        }
    }

    #[inline(always)]
    fn run_of(txn_id: i64) -> i64 {
        txn_id >> STATUS_SEGMENT_SHIFT
    }

    #[inline(always)]
    fn segment_at(&self, run: i64, order: Ordering) -> Option<&StatusSegment> {
        let position = self.directory[run as usize % STATUS_SEGMENTS].load(order);
        self.pool.get((position as usize).checked_sub(1)?)?.get()
    }

    /// The slot of a positive `txn_id`, or None when no segment holds its run
    #[inline(always)]
    fn get(&self, txn_id: i64) -> Option<i64> {
        let run = Self::run_of(txn_id);
        let segment = self.segment_at(run, Ordering::Acquire)?;
        if segment.run.load(Ordering::Acquire) != run {
            return None;
        }
        // Acquire keeps the second run read after this one, and pairs with
        // the release fence a reuse puts before clearing the slots
        let slot =
            segment.slots[txn_id as usize & (STATUS_SEGMENT_LEN - 1)].load(Ordering::Acquire);
        (segment.run.load(Ordering::Relaxed) == run).then_some(slot)
    }

    /// Stores `slot` for `txn_id` when a segment holds its run; false when none does
    fn set(&self, txn_id: i64, slot: i64) -> bool {
        let run = Self::run_of(txn_id);
        if let Some(segment) = self.segment_at(run, Ordering::Relaxed) {
            if segment.run.load(Ordering::Relaxed) == run {
                segment.slots[txn_id as usize & (STATUS_SEGMENT_LEN - 1)]
                    .store(slot, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Moves the open slot of `txn_id` to `to` in one step; false when it is
    /// not open or no segment holds it. The segment may be reused between the
    /// run check and the swap; the expected value names `txn_id`, so the swap
    /// then fails instead of changing the transaction that took the slot
    fn leave_open(&self, txn_id: i64, to: i64) -> bool {
        let run = Self::run_of(txn_id);
        self.segment_at(run, Ordering::Acquire)
            .filter(|segment| segment.run.load(Ordering::Acquire) == run)
            .is_some_and(|segment| {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::status_run_checked();
                segment.slots[txn_id as usize & (STATUS_SEGMENT_LEN - 1)]
                    .compare_exchange(slot_open(txn_id), to, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            })
    }

    /// Gives `run` a segment, reusing one whose run is below `base_run`.
    /// False when another run still in use holds its directory entry or no
    /// segment is free; its ids are then decided by the map.
    fn hold(&self, run: i64, base_run: i64) -> bool {
        let index = run as usize % STATUS_SEGMENTS;
        if let Some(segment) = self.segment_at(run, Ordering::Relaxed) {
            let held = segment.run.load(Ordering::Relaxed);
            if held == run {
                return true;
            }
            if held >= base_run {
                return false;
            }
        }
        let reusable = |p: &usize| {
            self.pool[*p]
                .get()
                .is_some_and(|s| s.run.load(Ordering::Relaxed) < base_run)
        };
        let Some(position) = (0..STATUS_SEGMENTS)
            .find(reusable)
            .or_else(|| (0..STATUS_SEGMENTS).find(|p| self.pool[*p].get().is_none()))
        else {
            return false;
        };
        let segment = self.pool[position].get_or_init(StatusSegment::new);
        let old_run = segment.run.load(Ordering::Relaxed);
        if old_run >= 0 {
            let _ = self.directory[old_run as usize % STATUS_SEGMENTS].compare_exchange(
                position as u32 + 1,
                0,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        segment.run.store(-1, Ordering::Relaxed);
        fence(Ordering::Release);
        for slot in segment.slots.iter() {
            slot.store(SLOT_UNKNOWN, Ordering::Relaxed);
        }
        segment.run.store(run, Ordering::Release);
        self.directory[index].store(position as u32 + 1, Ordering::Release);
        true
    }
}

/// The transactions map and what is kept beside it under the same lock
struct TxnTable {
    /// All tracked transactions (Active, Committing, Aborted).
    /// Committed transactions are REMOVED from this map.
    map: I64Map<TxnState>,
    /// The commit sequences of transactions committing and not yet complete
    committing: SmallVec<[i64; 4]>,
    /// The statement views alive now
    views: SmallVec<[LiveView; 4]>,
    /// The id the next statement view takes
    next_view_id: u64,
    /// The commit sequences of committed ids the status table holds no slot
    /// for, kept until no live view or base can ask about them
    slotless: I64Map<i64>,
}

impl TxnTable {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            map: I64Map::with_capacity(capacity),
            committing: SmallVec::new(),
            views: SmallVec::new(),
            next_view_id: 0,
            slotless: I64Map::new(),
        }
    }

    /// Ends `seq`'s place among the commits not yet complete
    fn uncommitting(&mut self, seq: i64) {
        if let Some(position) = self.committing.iter().position(|&s| s == seq) {
            self.committing.swap_remove(position);
        }
    }

    /// The smallest base any live view was taken at, or None with no view
    fn oldest_view_base(&self) -> Option<i64> {
        self.views.iter().map(|view| view.base).min()
    }
}

impl std::ops::Deref for TxnTable {
    type Target = I64Map<TxnState>;
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl std::ops::DerefMut for TxnTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

/// What the registry keeps of a live statement view
#[derive(Clone, Copy)]
struct LiveView {
    id: u64,
    floor: i64,
    base: i64,
}

/// The memory of a statement view. Its leases are the [`StatementView`]
/// handles out; one memory is taken again only once no `Arc` of it is left
/// but the caller's, which the last lease keeps until it has unregistered
pub struct ViewState {
    registry: Arc<TransactionRegistry>,
    id: u64,
    /// The sequence the view was taken at
    seq: i64,
    /// Commit sequences at or below `seq` not complete at the start
    in_flight: SmallVec<[i64; 4]>,
    /// Every id below this finished before the view was taken
    base: i64,
    /// The registry's destruction epoch when the view was taken, and whether
    /// a truncate or drop was taking rows away then
    destructions: u64,
    destroying: bool,
    leases: AtomicUsize,
}

/// One statement's view of the commits: those complete at its start, read
/// through `is_visible_at`. Registered while a handle lives, so maintenance
/// keeps what it reads and the commit metadata it asks about
pub struct StatementView {
    state: Arc<ViewState>,
}

impl StatementView {
    /// The sequence the view was taken at
    pub fn seq(&self) -> i64 {
        self.state.seq
    }

    /// Whether a truncate or drop took rows away since the view was taken,
    /// or while it was: what the view reads may be gone
    #[inline]
    pub fn overtaken(&self) -> bool {
        self.state.destroying
            || self
                .state
                .registry
                .destruction_epoch
                .load(Ordering::Acquire)
                != self.state.destructions
    }

    /// Err once the view is overtaken: what was read through it may be gone
    pub fn held(&self) -> crate::core::Result<()> {
        if self.overtaken() {
            return Err(crate::core::Error::ViewOvertaken);
        }
        Ok(())
    }

    /// Every commit at or below this sequence is visible to the view
    pub fn floor(&self) -> i64 {
        self.state
            .in_flight
            .iter()
            .min()
            .map_or(self.state.seq, |&s| s - 1)
    }

    /// Whether a commit that completed with `commit_seq` is visible to the view
    #[inline]
    pub fn sees_commit(&self, commit_seq: i64) -> bool {
        commit_seq <= self.state.seq && !self.state.in_flight.contains(&commit_seq)
    }

    /// The memory under this view, for a later capture to take again once
    /// every handle is gone; it reads nothing through it
    pub(crate) fn spare(&self) -> Arc<ViewState> {
        Arc::clone(&self.state)
    }
}

impl Clone for StatementView {
    fn clone(&self) -> Self {
        self.state.leases.fetch_add(1, Ordering::Relaxed);
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl std::fmt::Debug for StatementView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatementView")
            .field("seq", &self.state.seq)
            .field("in_flight", &self.state.in_flight)
            .field("base", &self.state.base)
            .finish()
    }
}

impl Drop for StatementView {
    fn drop(&mut self) {
        // The last handle unregisters before its Arc goes, so no capture
        // takes this memory again while it is still listed
        if self.state.leases.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }
        fence(Ordering::Acquire);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::last_view_lease_released();
        let registry = &self.state.registry;
        let mut txns = registry.transactions.lock();
        if let Some(position) = txns.views.iter().position(|view| view.id == self.state.id) {
            txns.views.swap_remove(position);
            registry.live_views.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Transaction registry with minimal memory footprint.
///
/// Design:
/// - `transactions`: Single map for Active/Committing/Aborted
/// - If txn_id not in map and valid → committed (implicit)
/// - `snapshot_seqs`: Commit sequences for snapshot isolation
///
/// Memory: O(active + aborted) instead of O(total)
pub struct TransactionRegistry {
    /// All tracked transactions, and what is kept beside them under one lock
    transactions: Mutex<TxnTable>,

    /// For SNAPSHOT ISOLATION: txn_id -> commit_seq.
    /// GC removes old entries when commit_seq < min_active_begin_seq.
    snapshot_seqs: Mutex<I64Map<i64>>,

    /// Last assigned transaction ID (after begin_transaction, equals txn_id).
    next_txn_id: AtomicI64,

    /// Last assigned sequence number (after begin/commit, equals that seq).
    next_sequence: AtomicI64,

    /// Global isolation level (0 = ReadCommitted, 1 = SnapshotIsolation).
    global_isolation_level: AtomicU8,

    /// Per-transaction isolation level overrides.
    isolation_overrides: Mutex<I64Map<u8>>,

    /// Count of active isolation overrides (skip lookup when 0).
    override_count: AtomicUsize,

    /// Count of active transactions (O(1) lookup).
    active_txn_count: AtomicUsize,

    /// Whether new transactions are being accepted.
    accepting: AtomicBool,

    /// Per-transaction status for visibility checks without the lock.
    status: TxnStatusTable,

    /// Every id below this finished; written under the transactions lock.
    base: AtomicI64,

    /// The highest id below `base` the map still holds as aborted, or 0.
    /// Written before `base`, read after it.
    aborted_floor: AtomicI64,

    /// The number of live statement views; written under the transactions lock
    live_views: AtomicUsize,

    /// Moves as a truncate or drop begins and again as it ends; written
    /// under the transactions lock, read by a view checking it still holds
    destruction_epoch: AtomicU64,

    /// Truncates and drops taking rows away now; written under the
    /// transactions lock, read there by a capture and outside it by a wait
    destructions_active: AtomicU32,
}

/// A truncate or drop taking rows away, from `begin_destruction` until this
/// drops on any exit: a view taken inside it, or before it, is overtaken
pub struct ViewDestruction {
    registry: Arc<TransactionRegistry>,
}

impl Drop for ViewDestruction {
    fn drop(&mut self) {
        let _txns = self.registry.transactions.lock();
        self.registry
            .destructions_active
            .fetch_sub(1, Ordering::Release);
        self.registry
            .destruction_epoch
            .fetch_add(1, Ordering::Release);
    }
}

impl TransactionRegistry {
    /// Creates a new transaction registry.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Creates a new transaction registry with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            transactions: Mutex::new(TxnTable::with_capacity(capacity)),
            snapshot_seqs: Mutex::new(I64Map::new()),
            next_txn_id: AtomicI64::new(0),
            next_sequence: AtomicI64::new(0),
            global_isolation_level: AtomicU8::new(0),
            isolation_overrides: Mutex::new(I64Map::new()),
            override_count: AtomicUsize::new(0),
            active_txn_count: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
            status: TxnStatusTable::new(),
            base: AtomicI64::new(1),
            aborted_floor: AtomicI64::new(0),
            live_views: AtomicUsize::new(0),
            destruction_epoch: AtomicU64::new(0),
            destructions_active: AtomicU32::new(0),
        }
    }

    /// Moves the base past every finished id from the current one, noting
    /// aborted ones the map keeps. Called under the transactions lock; the
    /// work is the smaller of the ids passed and the records the map holds.
    fn advance_base(&self, txns: &I64Map<TxnState>) {
        let next = self.next_txn_id.load(Ordering::Acquire);
        let mut base = self.base.load(Ordering::Relaxed);
        let mut floor = self.aborted_floor.load(Ordering::Relaxed);
        if txns
            .get(base)
            .is_some_and(|state| state.is_active_or_committing())
        {
            return;
        }
        if next.saturating_sub(base) > txns.len() as i64 {
            let open = txns
                .iter()
                .filter(|(id, s)| *id >= base && s.is_active_or_committing())
                .map(|(id, _)| id)
                .min()
                .unwrap_or(next + 1);
            let aborted = txns
                .iter()
                .filter(|(id, s)| *id >= base && *id < open && s.is_aborted())
                .map(|(id, _)| id)
                .max();
            #[cfg(test)]
            BASE_WORK.with(|work| work.set(work.get() + 2 * txns.len()));
            floor = floor.max(aborted.unwrap_or(0));
            base = open;
        } else {
            while base <= next {
                #[cfg(test)]
                BASE_WORK.with(|work| work.set(work.get() + 1));
                match txns.get(base) {
                    Some(state) if state.is_active_or_committing() => break,
                    Some(_) => floor = base,
                    None => {}
                }
                base += 1;
            }
        }
        self.aborted_floor.store(floor, Ordering::Release);
        self.base.store(base, Ordering::Release);
    }

    /// Recomputes the base and the aborted floor from the whole map, for
    /// after recovery and GC. Called under the transactions lock.
    fn rebase(&self, txns: &I64Map<TxnState>) {
        let next = self.next_txn_id.load(Ordering::Acquire);
        let base = txns
            .iter()
            .filter(|(_, s)| s.is_active_or_committing())
            .map(|(id, _)| id)
            .min()
            .unwrap_or(next + 1)
            .max(self.base.load(Ordering::Relaxed));
        let floor = txns
            .iter()
            .filter(|(id, s)| s.is_aborted() && *id < base)
            .map(|(id, _)| id)
            .max()
            .unwrap_or(0);
        self.aborted_floor.store(floor, Ordering::Release);
        self.base.store(base, Ordering::Release);
    }

    #[inline(always)]
    const fn isolation_to_u8(level: IsolationLevel) -> u8 {
        match level {
            IsolationLevel::ReadCommitted => 0,
            IsolationLevel::SnapshotIsolation => 1,
        }
    }

    #[inline(always)]
    const fn u8_to_isolation(value: u8) -> IsolationLevel {
        match value {
            0 => IsolationLevel::ReadCommitted,
            _ => IsolationLevel::SnapshotIsolation,
        }
    }

    /// Sets the global isolation level.
    pub fn set_global_isolation_level(&self, level: IsolationLevel) {
        self.global_isolation_level
            .store(Self::isolation_to_u8(level), Ordering::Release);
    }

    /// Gets the current global isolation level.
    #[inline(always)]
    pub fn get_global_isolation_level(&self) -> IsolationLevel {
        Self::u8_to_isolation(self.global_isolation_level.load(Ordering::Acquire))
    }

    /// Sets the isolation level for a specific transaction.
    pub fn set_transaction_isolation_level(&self, txn_id: i64, level: IsolationLevel) {
        let mut map = self.isolation_overrides.lock();
        let is_new = !map.contains_key(txn_id);
        map.insert(txn_id, Self::isolation_to_u8(level));
        if is_new {
            self.override_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Removes the isolation level override for a transaction.
    pub fn remove_transaction_isolation_level(&self, txn_id: i64) {
        // No transaction holds a level of its own
        if self.override_count.load(Ordering::Relaxed) == 0 {
            return;
        }
        if self.isolation_overrides.lock().remove(txn_id).is_some() {
            self.override_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Gets the isolation level for a specific transaction.
    #[inline(always)]
    pub fn get_isolation_level(&self, txn_id: i64) -> IsolationLevel {
        if self.override_count.load(Ordering::Relaxed) > 0 {
            if let Some(&level) = self.isolation_overrides.lock().get(txn_id) {
                return Self::u8_to_isolation(level);
            }
        }
        self.get_global_isolation_level()
    }

    /// Checks if snapshot isolation is needed for a transaction.
    #[inline(always)]
    fn needs_snapshot_isolation(&self, txn_id: i64) -> bool {
        let global = self.global_isolation_level.load(Ordering::Relaxed);
        if global == 1 {
            return true;
        }
        if self.override_count.load(Ordering::Relaxed) > 0 {
            if let Some(&level) = self.isolation_overrides.lock().get(txn_id) {
                return level == 1;
            }
        }
        false
    }

    /// Get transaction IDs of all active snapshot-isolation transactions.
    pub fn get_snapshot_transaction_ids(&self) -> Vec<i64> {
        let global = self.global_isolation_level.load(Ordering::Relaxed);
        let transactions = self.transactions.lock();

        if global == 1 {
            // All active transactions use snapshot isolation
            return transactions.keys().collect();
        }

        if self.override_count.load(Ordering::Relaxed) > 0 {
            let overrides = self.isolation_overrides.lock();
            return transactions
                .keys()
                .filter(|txn_id| overrides.get(*txn_id) == Some(&1))
                .collect();
        }

        Vec::new()
    }

    /// The minimum begin_seq among active snapshot isolation transactions,
    /// None with none, for a caller holding the lock
    fn min_snapshot_begin(&self, transactions: &TxnTable) -> Option<i64> {
        let global = self.global_isolation_level.load(Ordering::Relaxed);
        if global == 1 {
            return transactions
                .values()
                .filter(|s| s.is_active_or_committing())
                .map(|s| s.begin_seq())
                .min();
        }

        if self.override_count.load(Ordering::Relaxed) > 0 {
            let overrides = self.isolation_overrides.lock();
            return transactions
                .iter()
                .filter(|(id, s)| s.is_active_or_committing() && overrides.get(*id) == Some(&1))
                .map(|(_, s)| s.begin_seq())
                .min();
        }

        None
    }

    /// The lowest id whose status slot must stay: the base, or a live view's
    /// base below it, since a view may ask about any id at or above its own
    fn status_kept_from(&self, txns: &TxnTable) -> i64 {
        let base = self.base.load(Ordering::Relaxed);
        txns.oldest_view_base()
            .map_or(base, |view_base| view_base.min(base))
    }

    /// Begins a new transaction.
    pub fn begin_transaction(&self) -> (i64, i64) {
        self.begin_transaction_at(IsolationLevel::ReadCommitted)
    }

    /// Begins a transaction at `level`. A snapshot is protected under the same
    /// lock its sequence is taken in, the lock the oldest-snapshot query reads
    /// under, so every publication after that sequence keeps what it reads.
    pub fn begin_transaction_at(&self, level: IsolationLevel) -> (i64, i64) {
        if !self.accepting.load(Ordering::Acquire) {
            return (INVALID_TRANSACTION_ID, 0);
        }

        let (txn_id, begin_seq) = {
            let mut txns = self.transactions.lock();
            // Taken under the lock, so no id is handed out the map does not hold
            let txn_id = self.next_txn_id.fetch_add(1, Ordering::AcqRel) + 1;
            if txns.is_empty() {
                self.aborted_floor.store(0, Ordering::Release);
                self.base.store(txn_id, Ordering::Release);
            }
            // A snapshot keeps its level whatever the default becomes later
            if level == IsolationLevel::SnapshotIsolation
                || level != self.get_global_isolation_level()
            {
                let mut map = self.isolation_overrides.lock();
                map.insert(txn_id, Self::isolation_to_u8(level));
                self.override_count.fetch_add(1, Ordering::AcqRel);
            }
            let begin_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;
            txns.insert(txn_id, TxnState::new_active(begin_seq));
            debug_assert!(txn_id < SLOT_ID_LIMIT && begin_seq < SLOT_ID_LIMIT);
            let base_run = TxnStatusTable::run_of(self.status_kept_from(&txns));
            if self.status.hold(TxnStatusTable::run_of(txn_id), base_run) {
                self.status.set(txn_id, slot_open(txn_id));
            }
            (txn_id, begin_seq)
        };
        self.active_txn_count.fetch_add(1, Ordering::Relaxed);

        (txn_id, begin_seq)
    }

    /// Starts the commit process (two-phase commit). The sequence is taken in
    /// the lock section that marks the transaction committing, so a view
    /// never sees a sequence it does not also see as in flight
    #[inline]
    pub fn start_commit(&self, txn_id: i64) -> i64 {
        let mut txns = self.transactions.lock();
        let commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::commit_seq_taken();

        if let Some(entry) = txns.get_mut(txn_id) {
            entry.set_committing(commit_seq);
            debug_assert!(commit_seq < SLOT_ID_LIMIT);
            self.status.set(txn_id, -commit_seq);
            txns.committing.push(commit_seq);
        }

        commit_seq
    }

    /// Completes the commit process (two-phase commit).
    #[inline]
    pub fn complete_commit(&self, txn_id: i64) {
        {
            let mut txns = self.transactions.lock();
            let seq = txns.get(txn_id).map(|e| e.commit_seq()).unwrap_or(0);
            if seq > 0 {
                txns.uncommitting(seq);
            }

            // Store commit_seq for snapshot isolation while holding transactions lock
            // This prevents a race where the transaction is removed from txns but not yet
            // in snapshot_seqs, which could cause concurrent readers to incorrectly
            // treat it as an "old committed" transaction (visible).
            if self.global_isolation_level.load(Ordering::Relaxed) == 1
                || self.override_count.load(Ordering::Relaxed) > 0
            {
                self.snapshot_seqs.lock().insert(txn_id, seq);
            }

            txns.remove(txn_id);
            // With no slot to hold it, a view that asks later learns the sequence here
            if !self.status.set(txn_id, seq) && seq > 0 {
                txns.slotless.insert(txn_id, seq);
            }
            self.advance_base(&txns);
        };
        self.active_txn_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Commits a transaction (single-phase). Its sequence is taken in the
    /// lock section that makes it visible, as `start_commit` does
    pub fn commit_transaction(&self, txn_id: i64) -> i64 {
        let commit_seq;
        {
            let mut txns = self.transactions.lock();
            commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

            // Store commit_seq for snapshot isolation while holding transactions lock
            if self.global_isolation_level.load(Ordering::Relaxed) == 1
                || self.override_count.load(Ordering::Relaxed) > 0
            {
                self.snapshot_seqs.lock().insert(txn_id, commit_seq);
            }

            txns.remove(txn_id);
            // With no slot to hold it, a view that asks later learns the sequence here
            if !self.status.set(txn_id, commit_seq) {
                txns.slotless.insert(txn_id, commit_seq);
            }
            self.advance_base(&txns);
        }
        self.active_txn_count.fetch_sub(1, Ordering::Relaxed);

        commit_seq
    }

    /// Aborts a transaction.
    #[inline]
    pub fn abort_transaction(&self, txn_id: i64) {
        // Replace with aborted marker (don't remove - need to track for visibility)
        // Only decrement counter if transaction was actually active/committing
        //
        // CRITICAL: We must check if the transaction exists before inserting the aborted marker.
        // If it's not in the map, it's already committed (or invalid), and we must NOT
        // resurrect it as an aborted transaction.
        let should_decrement = {
            let mut txns = self.transactions.lock();
            if let Some(state) = txns.get_mut(txn_id) {
                let was_active = state.is_active_or_committing();
                let committing =
                    (state.status() == TxnStatus::Committing).then(|| state.commit_seq());
                *state = TxnState::new_aborted();
                if let Some(seq) = committing {
                    txns.uncommitting(seq);
                }
                self.status.set(txn_id, SLOT_ABORTED);
                self.advance_base(&txns);
                was_active
            } else {
                false
            }
        };

        if should_decrement {
            self.active_txn_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Ends a transaction that wrote nothing: no version carries its id, so
    /// it leaves no aborted record behind. The slot closes in the same step
    /// a publication would take; one that published, or one the table does
    /// not cover, is aborted instead
    pub fn forget_transaction(&self, txn_id: i64) {
        if !self.status.leave_open(txn_id, SLOT_ABORTED) {
            self.abort_transaction(txn_id);
            return;
        }
        let was_active = {
            let mut txns = self.transactions.lock();
            let was_active = txns
                .remove(txn_id)
                .is_some_and(|state| state.is_active_or_committing());
            self.advance_base(&txns);
            was_active
        };
        if was_active {
            self.active_txn_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Makes the next transaction id greater than `txn_id`, without marking
    /// anything committed
    pub fn reserve_txn_ids_through(&self, txn_id: i64) {
        let txns = self.transactions.lock();
        self.next_txn_id.fetch_max(txn_id, Ordering::AcqRel);
        self.rebase(&txns);
    }

    /// Recovers a committed transaction during startup recovery.
    pub fn recover_committed_transaction(&self, txn_id: i64, commit_seq: i64) {
        // Store in snapshot_seqs for visibility checks
        self.snapshot_seqs.lock().insert(txn_id, commit_seq);

        // Update next_txn_id if necessary (next_txn_id = last assigned)
        loop {
            let current = self.next_txn_id.load(Ordering::Acquire);
            if txn_id > current {
                if self
                    .next_txn_id
                    .compare_exchange(current, txn_id, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }

        // Update next_sequence if necessary (next_sequence = last assigned)
        loop {
            let current = self.next_sequence.load(Ordering::Acquire);
            if commit_seq > current {
                if self
                    .next_sequence
                    .compare_exchange(current, commit_seq, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Records an aborted transaction during recovery.
    pub fn recover_aborted_transaction(&self, txn_id: i64) {
        {
            let mut txns = self.transactions.lock();
            txns.insert(txn_id, TxnState::new_aborted());
            self.status.set(txn_id, SLOT_ABORTED);
            if txn_id < self.base.load(Ordering::Relaxed) {
                self.aborted_floor.fetch_max(txn_id, Ordering::AcqRel);
            }
        }

        // Update next_txn_id if necessary (next_txn_id = last assigned)
        loop {
            let current = self.next_txn_id.load(Ordering::Acquire);
            if txn_id > current {
                if self
                    .next_txn_id
                    .compare_exchange(current, txn_id, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Checks if a version is visible (main entry point).
    ///
    /// Hot path - optimized with single lock acquisition.
    #[inline(always)]
    pub fn is_visible(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        // FAST PATH 1: Own writes always visible
        if version_txn_id == viewer_txn_id {
            return true;
        }

        // FAST PATH 2: Recovery transactions always visible
        if version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }

        // FAST PATH 3: Check isolation level
        let needs_snapshot = self.needs_snapshot_isolation(viewer_txn_id);

        if needs_snapshot {
            self.is_visible_snapshot(version_txn_id, viewer_txn_id)
        } else {
            self.check_committed(version_txn_id)
        }
    }

    /// Checks if a transaction is committed (READ COMMITTED visibility).
    ///
    /// Answered without the lock from the base or the status table; only an
    /// id the table does not cover takes the lock.
    #[inline(always)]
    fn check_committed(&self, txn_id: i64) -> bool {
        if txn_id <= 0 {
            return false;
        }
        if txn_id < self.base.load(Ordering::Acquire)
            && txn_id > self.aborted_floor.load(Ordering::Acquire)
        {
            return true;
        }
        match self.status.get(txn_id) {
            Some(SLOT_UNKNOWN) | None => {}
            Some(slot) => return slot > 0,
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::note_read("registry.check_committed.locked");

        // If in map = Active, Committing, or Aborted = not committed
        if self.transactions.lock().contains_key(txn_id) {
            return false;
        }

        // Not in map - committed if valid txn_id
        txn_id <= self.next_txn_id.load(Ordering::Acquire)
    }

    /// Checks if a version is directly visible (for READ COMMITTED).
    #[inline(always)]
    pub fn is_directly_visible(&self, version_txn_id: i64) -> bool {
        if version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        self.check_committed(version_txn_id)
    }

    /// Snapshot isolation visibility check.
    ///
    /// Avoids nested locking by releasing transactions lock before acquiring snapshot_seqs.
    #[cold]
    #[inline(never)]
    fn is_visible_snapshot(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        // First lock: get transaction states
        let (version_state, viewer_begin_seq) = {
            let txns = self.transactions.lock();

            // Get viewer's begin_seq (must be active to be viewing)
            let viewer_begin_seq = match txns.get(viewer_txn_id) {
                Some(state) if state.is_active_or_committing() => state.begin_seq(),
                _ => {
                    // Viewer not active - this is unusual but handle gracefully
                    drop(txns);
                    // For a committed viewer, use current sequence as begin_seq
                    // (they can see everything committed before they finished)
                    return self.check_committed(version_txn_id);
                }
            };

            // Get version's state
            let version_state = txns.get(version_txn_id).copied();

            (version_state, viewer_begin_seq)
        };
        // transactions lock released here

        // Check version state
        match version_state {
            Some(state) => {
                if state.is_aborted() {
                    return false; // Aborted = never visible
                }
                // Active or Committing = not visible to others
                false
            }
            None => {
                // Not in transactions map - either committed or invalid
                let next = self.next_txn_id.load(Ordering::Acquire);
                if version_txn_id <= 0 || version_txn_id > next {
                    return false; // Invalid txn_id
                }

                // Second lock: get commit_seq (only if needed)
                let commit_seq = self.snapshot_seqs.lock().get(version_txn_id).copied();

                // Committed - check commit_seq against viewer's begin_seq
                if let Some(commit_seq) = commit_seq {
                    return commit_seq <= viewer_begin_seq;
                }

                // Not in snapshot_seqs = old committed (GC'd)
                // GC only removes when commit_seq < min_active_begin_seq
                // Since viewer is active, viewer_begin_seq >= min_active_begin_seq
                // So if GC'd, commit_seq < min_active_begin_seq <= viewer_begin_seq
                // Therefore visible
                true
            }
        }
    }

    /// Gets the commit sequence for a transaction.
    pub fn get_commit_sequence(&self, txn_id: i64) -> Option<i64> {
        // Check snapshot_seqs first
        if let Some(&seq) = self.snapshot_seqs.lock().get(txn_id) {
            return Some(seq);
        }

        // Check if still active/aborted
        if let Some(state) = self.transactions.lock().get(txn_id) {
            if state.is_aborted() || state.is_active_or_committing() {
                return None;
            }
        }

        // Valid committed transaction (GC'd from snapshot_seqs)
        let next = self.next_txn_id.load(Ordering::Acquire);
        if txn_id > 0 && txn_id <= next {
            return Some(0); // Committed but commit_seq unknown
        }

        None
    }

    /// Gets the begin sequence for an active transaction.
    pub fn get_transaction_begin_sequence(&self, txn_id: i64) -> i64 {
        self.transactions
            .lock()
            .get(txn_id)
            .map(|e| e.begin_seq())
            .unwrap_or(0)
    }

    /// Gets the commit sequence for a transaction that is currently committing.
    /// Returns 0 if the transaction is not found or not in committing state.
    pub fn get_committing_sequence(&self, txn_id: i64) -> i64 {
        self.transactions
            .lock()
            .get(txn_id)
            .map(|e| e.commit_seq())
            .unwrap_or(0)
    }

    /// Gets the current sequence number.
    pub fn get_current_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Gets the current commit sequence number.
    pub fn current_commit_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Gets a snapshot-safe commit sequence cutoff.
    ///
    /// Returns the minimum commit_seq among all in-flight (committing)
    /// transactions minus 1, or `current_commit_sequence()` if none are
    /// in-flight. This guarantees all commits with seq ≤ the returned
    /// value have fully completed (versions applied, tombstones applied,
    /// removed from transactions map).
    pub fn safe_snapshot_cutoff(&self) -> i64 {
        let txns = self.transactions.lock();
        Self::completed_through(&txns, self.next_sequence.load(Ordering::Acquire))
    }

    /// Every commit at or below the returned sequence is complete. Read under
    /// the lock `start_commit` takes its sequence in, with `current` loaded
    /// under it
    fn completed_through(txns: &TxnTable, current: i64) -> i64 {
        txns.committing.iter().min().map_or(current, |&seq| seq - 1)
    }

    /// Takes a statement view: the commits complete now, plus the reader's
    /// own writes, and nothing that completes later. Registered under the
    /// lock commits take their sequence in, so maintenance and the status
    /// table keep what it reads until it drops
    pub fn capture_view(self: &Arc<Self>) -> StatementView {
        self.capture_view_reusing(None)
    }

    /// Whether a truncate or drop is taking rows away now: a view taken while
    /// one is would be overtaken from the start
    pub fn destruction_running(&self) -> bool {
        self.destructions_active.load(Ordering::Acquire) > 0
    }

    /// Marks a truncate or drop taking rows away until the guard drops. Taken
    /// before the rows go, under the lock views are taken in
    pub fn begin_destruction(self: &Arc<Self>) -> ViewDestruction {
        let txns = self.transactions.lock();
        self.destructions_active.fetch_add(1, Ordering::Release);
        self.destruction_epoch.fetch_add(1, Ordering::Release);
        drop(txns);
        ViewDestruction {
            registry: Arc::clone(self),
        }
    }

    /// [`Self::capture_view`] into the memory `spare` when it is this
    /// registry's and no other `Arc` of it is left, else into new memory.
    /// The capture and its registration are the same either way
    pub fn capture_view_reusing(self: &Arc<Self>, spare: Option<Arc<ViewState>>) -> StatementView {
        let mut spare = spare.filter(|state| Arc::ptr_eq(&state.registry, self));
        let mut txns = self.transactions.lock();
        let id = txns.next_view_id;
        txns.next_view_id += 1;
        let base = self.base.load(Ordering::Acquire);
        let seq = self.next_sequence.load(Ordering::Acquire);
        // Both move only under this lock, so the view orders with each truncate or drop
        let destructions = self.destruction_epoch.load(Ordering::Acquire);
        let destroying = self.destructions_active.load(Ordering::Acquire) > 0;
        let (floor, fresh) = match spare.as_mut().and_then(Arc::get_mut) {
            Some(state) => {
                state.id = id;
                state.seq = seq;
                state.base = base;
                state.destructions = destructions;
                state.destroying = destroying;
                state.in_flight.clear();
                state.in_flight.extend(txns.committing.iter().copied());
                *state.leases.get_mut() = 1;
                let floor = state.in_flight.iter().min().map_or(seq, |&s| s - 1);
                (floor, None)
            }
            None => {
                let in_flight: SmallVec<[i64; 4]> = txns.committing.iter().copied().collect();
                let floor = in_flight.iter().min().map_or(seq, |&s| s - 1);
                (floor, Some(in_flight))
            }
        };
        txns.views.push(LiveView { id, floor, base });
        self.live_views.fetch_add(1, Ordering::Release);
        drop(txns);
        let state = match (spare, fresh) {
            (Some(state), None) => state,
            (_, in_flight) => Arc::new(ViewState {
                registry: Arc::clone(self),
                id,
                seq,
                in_flight: in_flight.unwrap_or_default(),
                base,
                destructions,
                destroying,
                leases: AtomicUsize::new(1),
            }),
        };
        StatementView { state }
    }

    /// The sequence maintenance may act below: a commit with a lower sequence
    /// is complete and visible to every snapshot and every live view.
    /// Removals act only on what they classified before reading it
    pub fn maintenance_limit(&self) -> i64 {
        let txns = self.transactions.lock();
        let snapshot = self.min_snapshot_begin(&txns);
        let completed = Self::completed_through(&txns, self.next_sequence.load(Ordering::Acquire));
        let views = txns.views.iter().map(|view| view.floor).min();
        drop(txns);
        let limit = views.map_or(completed, |floor| floor.min(completed)) + 1;
        snapshot.map_or(limit, |begin| begin.min(limit))
    }

    /// Every commit at or below the returned sequence is seen by every open
    /// snapshot and live view; None with neither. It says nothing about
    /// commits still in flight: a caller classifies by state first
    pub fn min_reader_seq(&self) -> Option<i64> {
        let txns = self.transactions.lock();
        let snapshot = self.min_snapshot_begin(&txns);
        let views = txns.views.iter().map(|view| view.floor).min();
        match (snapshot, views) {
            (Some(begin), Some(floor)) => Some(begin.min(floor)),
            (begin, floor) => begin.or(floor),
        }
    }

    /// Whether `version_txn_id`'s version is visible to a statement of
    /// `viewer_txn_id` under `view`
    #[inline(always)]
    pub fn is_visible_at(
        &self,
        version_txn_id: i64,
        viewer_txn_id: i64,
        view: &StatementView,
    ) -> bool {
        if version_txn_id == viewer_txn_id || version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }
        if version_txn_id <= 0 {
            return false;
        }
        // Finished before the view was taken: committed or aborted for good
        if version_txn_id < view.state.base {
            return self.check_committed(version_txn_id);
        }
        match self.status.get(version_txn_id) {
            Some(slot) if slot > 0 => return view.sees_commit(slot),
            // Open, published, committing or aborted: a commit completing
            // after this read completed after the view was taken
            Some(slot) if slot < 0 => return false,
            _ => {}
        }
        self.is_visible_at_locked(version_txn_id, view)
    }

    /// [`Self::is_visible_at`] for an id the status table holds no slot for
    #[cold]
    #[inline(never)]
    fn is_visible_at_locked(&self, version_txn_id: i64, view: &StatementView) -> bool {
        let txns = self.transactions.lock();
        match self.status.get(version_txn_id) {
            Some(slot) if slot > 0 => return view.sees_commit(slot),
            Some(slot) if slot < 0 => return false,
            _ => {}
        }
        if txns.contains_key(version_txn_id) {
            return false;
        }
        match txns.slotless.get(version_txn_id) {
            Some(&seq) => view.sees_commit(seq),
            None => {
                debug_assert!(
                    false,
                    "transaction {version_txn_id} at or above a live view's base has no commit record"
                );
                false
            }
        }
    }

    /// Runs garbage collection.
    ///
    /// Removes:
    /// - Old aborted entries (txn_id < min_active_txn_id - buffer)
    /// - Old snapshot_seqs entries (commit_seq < min_active_begin_seq)
    /// - Isolation overrides for completed transactions
    pub fn run_gc(&self) -> usize {
        let mut removed = 0;

        // Step 1: Collect override keys FIRST if any exist (O(Overrides))
        // This avoids building O(ActiveTxns) set just to filter O(Overrides) entries
        let override_keys: Vec<i64> = if self.override_count.load(Ordering::Relaxed) > 0 {
            self.isolation_overrides.lock().keys().collect()
        } else {
            Vec::new()
        };

        // Step 2: Single lock on transactions to:
        // - Calculate min values
        // - Remove old aborted entries, slotless records and snapshot_seqs,
        //   so no view registers between a bound and the removal it allows
        // - Check which override keys are no longer active
        let invalid_overrides = {
            let mut txns = self.transactions.lock();
            let mut min_begin = i64::MAX;
            let mut min_id = i64::MAX;

            for (id, state) in txns.iter() {
                if state.is_active_or_committing() {
                    min_begin = min_begin.min(state.begin_seq());
                    min_id = min_id.min(id);
                }
            }

            if min_id == i64::MAX {
                min_id = self.next_txn_id.load(Ordering::Acquire);
            }

            // GC aborted entries while holding lock, keeping any a live view may ask about
            let view_base = txns.oldest_view_base().unwrap_or(i64::MAX);
            let aborted_cutoff = min_id.saturating_sub(10000).min(view_base);
            if aborted_cutoff > 0 {
                let len_before = txns.len();
                txns.retain(|id, state| !state.is_aborted() || id >= aborted_cutoff);
                removed += len_before - txns.len();
            }
            self.rebase(&txns);

            // A slotless record goes once its id is below the base and below every view's
            let slotless_cutoff = self.base.load(Ordering::Relaxed).min(view_base);
            let len_before = txns.slotless.len();
            txns.slotless.retain(|id, _| id >= slotless_cutoff);
            removed += len_before - txns.slotless.len();

            {
                let mut seqs = self.snapshot_seqs.lock();
                let len_before = seqs.len();
                seqs.retain(|_, &mut commit_seq| commit_seq >= min_begin);
                removed += len_before - seqs.len();
            }

            // Check which override keys are no longer active (O(Overrides) lookups)
            let invalid: Vec<i64> = override_keys
                .iter()
                .filter(|&&txn_id| {
                    // Invalid if not in transactions map (committed/unknown)
                    // or if aborted
                    match txns.get(txn_id) {
                        None => true, // Not active = committed or never existed
                        Some(state) => state.is_aborted(),
                    }
                })
                .copied()
                .collect();

            invalid
        };

        // Step 3: Remove invalid isolation overrides (O(InvalidOverrides))
        if !invalid_overrides.is_empty() {
            let mut overrides = self.isolation_overrides.lock();
            let mut actual_removals = 0usize;
            for txn_id in &invalid_overrides {
                if overrides.remove(*txn_id).is_some() {
                    removed += 1;
                    actual_removals += 1;
                }
            }
            if actual_removals > 0 {
                self.override_count
                    .fetch_sub(actual_removals, Ordering::Relaxed);
            }
        }

        removed
    }

    /// Cleans up old transactions (legacy API).
    pub fn cleanup_old_transactions(&self, _max_age: std::time::Duration) -> i32 {
        self.run_gc() as i32
    }

    /// Waits for active transactions to complete with timeout.
    pub fn wait_for_active_transactions(&self, timeout: std::time::Duration) -> i32 {
        let deadline = crate::common::time_compat::Instant::now() + timeout;

        loop {
            if crate::common::time_compat::Instant::now() > deadline {
                break;
            }

            let count = self.active_count();
            if count == 0 {
                return 0;
            }

            // Single-threaded: no other transaction can end while this one waits
            if cfg!(target_arch = "wasm32") {
                break;
            }
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        self.active_count() as i32
    }

    /// Stops accepting new transactions.
    pub fn stop_accepting_transactions(&self) {
        self.accepting.store(false, Ordering::Release);
    }

    /// Starts accepting new transactions.
    pub fn start_accepting_transactions(&self) {
        self.accepting.store(true, Ordering::Release);
    }

    /// Shuts down the registry.
    pub fn shutdown(&self) {
        self.stop_accepting_transactions();
    }

    /// Checks if the registry is accepting new transactions.
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    /// Gets the count of active transactions.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_txn_count.load(Ordering::Relaxed)
    }

    /// Checks if a transaction is active.
    pub fn is_active(&self, txn_id: i64) -> bool {
        self.transactions
            .lock()
            .get(txn_id)
            .map(|e| e.status() == TxnStatus::Active)
            .unwrap_or(false)
    }

    /// Whether a transaction is active or committing: its applied versions
    /// may still become visible, or be taken back
    pub fn is_open(&self, txn_id: i64) -> bool {
        self.transactions
            .lock()
            .get(txn_id)
            .is_some_and(|e| matches!(e.status(), TxnStatus::Active | TxnStatus::Committing))
    }

    /// Checks if a transaction is committed.
    pub fn is_committed(&self, txn_id: i64) -> bool {
        // Check if in transactions map
        if self.transactions.lock().contains_key(txn_id) {
            // If aborted or active/committing, not committed
            return false;
        }

        // Not in map - committed if valid ID
        let next = self.next_txn_id.load(Ordering::Acquire);
        txn_id > 0 && txn_id <= next
    }

    /// Checks if a transaction is in committing state.
    #[cfg(test)]
    pub fn is_committing(&self, txn_id: i64) -> bool {
        self.transactions
            .lock()
            .get(txn_id)
            .map(|e| e.status() == TxnStatus::Committing)
            .unwrap_or(false)
    }

    /// Checks if a transaction was committed before a given sequence.
    pub fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        // Special case: negative IDs are always "old"
        if txn_id < 0 {
            return true;
        }

        // A slot answers without a lock: the commit sequence, or not complete
        match self.status.get(txn_id) {
            Some(slot) if slot > 0 => return slot <= cutoff_commit_seq,
            Some(slot) if slot < 0 => return false,
            _ => {}
        }

        {
            let txns = self.transactions.lock();
            // If aborted or still active/committing, not committed
            if txns.contains_key(txn_id) {
                return false;
            }
            if let Some(&commit_seq) = txns.slotless.get(txn_id) {
                return commit_seq <= cutoff_commit_seq;
            }
        }

        // Check snapshot_seqs for exact commit_seq
        if let Some(&commit_seq) = self.snapshot_seqs.lock().get(txn_id) {
            return commit_seq <= cutoff_commit_seq;
        }

        // Not in snapshot_seqs = old committed (GC'd)
        // GC'd means commit_seq was < min_active_begin_seq
        // Therefore definitely < cutoff_commit_seq
        let next = self.next_txn_id.load(Ordering::Acquire);
        txn_id > 0 && txn_id <= next
    }
}

impl Default for TransactionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VisibilityChecker for TransactionRegistry {
    #[inline(always)]
    fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
        TransactionRegistry::is_visible(self, version_txn_id, viewing_txn_id)
    }

    #[inline(always)]
    fn is_visible_at(
        &self,
        version_txn_id: i64,
        viewing_txn_id: i64,
        view: &StatementView,
    ) -> bool {
        TransactionRegistry::is_visible_at(self, version_txn_id, viewing_txn_id, view)
    }

    fn get_current_sequence(&self) -> i64 {
        TransactionRegistry::get_current_sequence(self)
    }

    fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        TransactionRegistry::is_committed_before(self, txn_id, cutoff_commit_seq)
    }

    #[inline(always)]
    fn needs_snapshot_isolation(&self, txn_id: i64) -> bool {
        TransactionRegistry::needs_snapshot_isolation(self, txn_id)
    }

    fn admit_publication(&self, txn_id: i64) -> bool {
        if self.status.leave_open(txn_id, slot_published(txn_id)) {
            return true;
        }
        // A read the run check confirms; a slot reused since reads as None
        match self.status.get(txn_id) {
            // Published before, or committing: start_commit precedes publication
            Some(slot) if slot == slot_published(txn_id) || slot_committing(slot) => true,
            Some(slot) if slot == SLOT_ABORTED || slot > 0 => false,
            _ => self
                .transactions
                .lock()
                .get(txn_id)
                .is_some_and(|state| state.is_active_or_committing()),
        }
    }

    fn maintenance_limit(&self) -> i64 {
        TransactionRegistry::maintenance_limit(self)
    }

    fn history_limit(&self) -> Option<i64> {
        if self.global_isolation_level.load(Ordering::Relaxed) != 1
            && self.override_count.load(Ordering::Relaxed) == 0
            && self.live_views.load(Ordering::Acquire) == 0
        {
            return None;
        }
        Some(TransactionRegistry::maintenance_limit(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let state = registry.transactions.lock().get(txn_id).copied();
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

        assert_eq!(registry.snapshot_seqs.lock().len(), 10);

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
        assert!(registry.snapshot_seqs.lock().is_empty());
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
        let state = registry.transactions.lock().get(42).copied();
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
        let next = registry.next_txn_id.load(Ordering::Acquire);
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

    #[test]
    fn an_abort_the_base_passes_stays_invisible() {
        let registry = TransactionRegistry::new();
        let (older, _) = registry.begin_transaction();
        let (aborted, _) = registry.begin_transaction();
        let (committed, _) = registry.begin_transaction();
        registry.abort_transaction(aborted);
        registry.commit_transaction(committed);
        registry.commit_transaction(older);
        assert!(registry.base.load(Ordering::Relaxed) > committed);
        assert!(!registry.is_directly_visible(aborted));
        assert!(registry.is_directly_visible(committed));
        assert!(registry.is_directly_visible(older));
    }

    #[test]
    fn ids_whose_segment_was_reused_keep_their_status() {
        let registry = TransactionRegistry::new();
        let (aborted, _) = registry.begin_transaction();
        registry.abort_transaction(aborted);
        let (committed, _) = registry.begin_transaction();
        registry.commit_transaction(committed);
        for _ in 0..3 * STATUS_SEGMENT_LEN {
            let (id, _) = registry.begin_transaction();
            registry.forget_transaction(id);
        }
        assert_eq!(
            registry.status.get(committed),
            None,
            "the segment was reused"
        );
        assert!(registry.is_directly_visible(committed));
        assert!(!registry.is_directly_visible(aborted));
    }

    #[test]
    fn a_published_transaction_is_aborted_not_forgotten() {
        let registry = TransactionRegistry::new();
        let (id, _) = registry.begin_transaction();
        assert!(registry.admit_publication(id));
        registry.forget_transaction(id);
        assert!(!registry.is_directly_visible(id));
        assert!(registry
            .transactions
            .lock()
            .get(id)
            .is_some_and(|s| s.is_aborted()));
    }

    #[test]
    fn a_forgotten_transaction_cannot_publish() {
        let registry = TransactionRegistry::new();
        let (id, _) = registry.begin_transaction();
        registry.forget_transaction(id);
        assert!(!registry.admit_publication(id));
    }

    #[test]
    fn a_committing_transaction_may_publish() {
        let registry = TransactionRegistry::new();
        let (id, _) = registry.begin_transaction();
        registry.start_commit(id);
        assert!(registry.admit_publication(id));
        registry.complete_commit(id);
        assert!(
            !registry.admit_publication(id),
            "a committed transaction published again"
        );
    }

    #[test]
    fn closing_a_long_transaction_does_not_walk_every_later_id() {
        let registry = TransactionRegistry::new();
        let (long, _) = registry.begin_transaction();
        let mut last = long;
        for _ in 0..100_000 {
            let (id, _) = registry.begin_transaction();
            registry.commit_transaction(id);
            last = id;
        }
        let before = BASE_WORK.with(|work| work.get());
        registry.commit_transaction(long);
        let work = BASE_WORK.with(|work| work.get()) - before;
        assert!(work < 100, "closing it took {work} steps");
        assert!(registry.base.load(Ordering::Relaxed) > last);
        assert!(registry.is_directly_visible(last));
    }

    /// Drops the status segment holding `txn_id`'s run, as a reuse would
    fn drop_status_segment(registry: &TransactionRegistry, txn_id: i64) {
        let run = TxnStatusTable::run_of(txn_id);
        registry.status.directory[run as usize % STATUS_SEGMENTS].store(0, Ordering::Release);
    }

    #[test]
    fn a_view_is_overtaken_by_a_destruction_begun_after_it_or_running_when_taken() {
        let registry = Arc::new(TransactionRegistry::new());
        let before = registry.capture_view();
        assert!(!before.overtaken(), "nothing destroyed yet");
        let first = registry.begin_destruction();
        assert!(before.overtaken(), "a destruction begun after the view");
        let during = registry.capture_view();
        assert!(
            during.overtaken(),
            "taken while rows are going, before the epoch moves again"
        );
        let second = registry.begin_destruction();
        drop(first);
        let overlapped = registry.capture_view();
        drop(second);
        assert!(during.overtaken(), "taken while rows were going");
        assert!(overlapped.overtaken(), "taken while the second still ran");
        assert!(
            before.overtaken(),
            "stays overtaken after the destruction ends"
        );
        let after = registry.capture_view();
        assert!(!after.overtaken(), "taken once both ended");
        let failed = || -> Result<(), ()> {
            let _destruction = registry.begin_destruction();
            Err(())
        };
        assert!(failed().is_err());
        assert!(
            after.overtaken(),
            "a destruction that failed still moved the epoch"
        );
        assert!(
            !registry.capture_view().overtaken(),
            "a failed destruction ends its window on the error exit"
        );
    }

    #[test]
    fn a_view_memory_taken_again_keeps_its_room_for_more_commits_in_flight_than_fit_inline() {
        let registry = Arc::new(TransactionRegistry::new());
        for _ in 0..6 {
            let (txn, _) = registry.begin_transaction();
            registry.start_commit(txn);
        }
        let view = registry.capture_view();
        assert!(
            view.state.in_flight.spilled(),
            "six commits in flight spill"
        );
        assert_eq!(view.state.in_flight.len(), 6);
        let room = view.state.in_flight.as_ptr();
        let spare = view.spare();
        drop(view);
        let again = registry.capture_view_reusing(Some(spare));
        assert_eq!(again.state.in_flight.len(), 6);
        assert_eq!(
            again.state.in_flight.as_ptr(),
            room,
            "the memory taken again keeps its spilled room"
        );
    }

    #[test]
    fn a_view_copied_across_threads_unregisters_once_when_its_last_copy_goes() {
        let registry = Arc::new(TransactionRegistry::new());
        let other = registry.capture_view();
        let view = registry.capture_view();
        let mut copies: Vec<StatementView> = (0..8).map(|_| view.clone()).collect();
        drop(view);
        let last = copies.pop().unwrap();
        let start = Arc::new(std::sync::Barrier::new(copies.len()));
        let threads: Vec<_> = copies
            .into_iter()
            .map(|copy| {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    drop(copy);
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let listed = || registry.transactions.lock().views.len();
        assert_eq!(listed(), 2, "a view one copy still holds is registered");
        std::thread::spawn(move || drop(last)).join().unwrap();
        assert_eq!(listed(), 1, "the view went, the other stayed");
        assert_eq!(registry.live_views.load(Ordering::Acquire), 1);
        drop(other);
        assert_eq!(
            (listed(), registry.live_views.load(Ordering::Acquire)),
            (0, 0)
        );
    }

    #[test]
    fn a_view_hides_a_commit_in_flight_and_shows_one_completed_above_it() {
        let registry = Arc::new(TransactionRegistry::new());
        let (a, _) = registry.begin_transaction();
        let (b, _) = registry.begin_transaction();
        let (viewer, _) = registry.begin_transaction();
        let a_seq = registry.start_commit(a);
        let b_seq = registry.start_commit(b);
        registry.complete_commit(b);
        assert!(b_seq > a_seq);
        let view = registry.capture_view();
        assert!(
            registry.is_visible_at(b, viewer, &view),
            "b completed before the view"
        );
        assert!(
            !registry.is_visible_at(a, viewer, &view),
            "a was in flight at the view"
        );
        registry.complete_commit(a);
        assert!(
            !registry.is_visible_at(a, viewer, &view),
            "a completed after the view"
        );
        let (c, _) = registry.begin_transaction();
        registry.commit_transaction(c);
        assert!(
            !registry.is_visible_at(c, viewer, &view),
            "c began after the view"
        );
        let later = registry.capture_view();
        assert!(registry.is_visible_at(a, viewer, &later));
        assert!(registry.is_visible_at(c, viewer, &later));
        assert_eq!(view.floor(), a_seq - 1);
    }

    #[test]
    fn a_commit_sequence_is_taken_under_the_lock_views_are_taken_in() {
        let registry = Arc::new(TransactionRegistry::new());
        let (a, _) = registry.begin_transaction();
        let held = std::rc::Rc::new(std::cell::Cell::new(None));
        let (hook_registry, hook_held) = (Arc::clone(&registry), std::rc::Rc::clone(&held));
        crate::test_failpoints::after_commit_seq_taken(move || {
            hook_held.set(Some(hook_registry.transactions.try_lock().is_none()));
        });
        registry.start_commit(a);
        assert_eq!(
            held.get(),
            Some(true),
            "the sequence was taken with the lock free, so a view could be taken before the commit is marked"
        );
    }

    #[test]
    fn a_live_view_keeps_the_status_segment_a_new_run_would_reuse() {
        let registry = Arc::new(TransactionRegistry::new());
        let (old, _) = registry.begin_transaction();
        let (x, _) = registry.begin_transaction();
        let x_seq = registry.commit_transaction(x);
        let (viewer, _) = registry.begin_transaction();
        let view = registry.capture_view();
        registry.commit_transaction(old);
        registry.commit_transaction(viewer);
        // The next ids start a run past x's, whose hold may take x's segment
        registry.reserve_txn_ids_through(3 * STATUS_SEGMENT_LEN as i64);
        let x_run = TxnStatusTable::run_of(x);
        assert!(TxnStatusTable::run_of(registry.base.load(Ordering::Relaxed)) > x_run);
        let (z, _) = registry.begin_transaction();
        assert!(TxnStatusTable::run_of(z) > x_run);
        assert_eq!(
            registry.status.get(x),
            Some(x_seq),
            "a new run took x's segment while a view below it lived"
        );
        assert!(registry.is_visible_at(x, viewer, &view));

        // With no view left, the next new run reuses that segment as before
        drop(view);
        registry.commit_transaction(z);
        registry.reserve_txn_ids_through(5 * STATUS_SEGMENT_LEN as i64);
        registry.begin_transaction();
        assert_eq!(registry.status.get(x), None);
    }

    #[test]
    fn a_commit_without_a_slot_before_any_view_is_seen_by_a_later_one() {
        let registry = Arc::new(TransactionRegistry::new());
        let (old, _) = registry.begin_transaction();
        let (x, _) = registry.begin_transaction();
        drop_status_segment(&registry, x);
        let x_seq = registry.start_commit(x);
        registry.complete_commit(x);
        let (y, _) = registry.begin_transaction();
        drop_status_segment(&registry, y);
        let y_seq = registry.commit_transaction(y);
        let (viewer, _) = registry.begin_transaction();
        let view = registry.capture_view();
        assert!(
            view.state.base <= x,
            "the old transaction holds the base below x"
        );
        assert!(
            registry.is_visible_at(x, viewer, &view),
            "x completed at {x_seq}"
        );
        assert!(
            registry.is_visible_at(y, viewer, &view),
            "y committed at {y_seq}"
        );
        assert!(registry.is_committed_before(x, x_seq));
        assert!(!registry.is_committed_before(x, x_seq - 1));

        // The old transaction ends and the base passes x while the view lives
        registry.commit_transaction(old);
        registry.commit_transaction(viewer);
        assert!(registry.base.load(Ordering::Relaxed) > x);
        registry.run_gc();
        assert!(
            registry.is_visible_at(x, viewer, &view),
            "run_gc removed x's record while a view below it lived"
        );
        drop(view);
        registry.run_gc();
        assert!(registry.transactions.lock().slotless.is_empty());
    }

    #[test]
    fn a_view_reads_an_open_transaction_as_invisible_through_its_commit() {
        let registry = Arc::new(TransactionRegistry::new());
        let (viewer, _) = registry.begin_transaction();
        let (a, _) = registry.begin_transaction();
        let view = registry.capture_view();
        assert!(!registry.is_visible_at(a, viewer, &view));
        registry.start_commit(a);
        assert!(!registry.is_visible_at(a, viewer, &view));
        registry.complete_commit(a);
        assert!(!registry.is_visible_at(a, viewer, &view));
        assert!(registry.is_visible_at(viewer, viewer, &view), "own writes");
    }

    #[test]
    fn the_maintenance_limit_stays_below_commits_in_flight_and_view_floors() {
        let registry = Arc::new(TransactionRegistry::new());
        let (a, _) = registry.begin_transaction();
        let a_seq = registry.start_commit(a);
        let (b, _) = registry.begin_transaction();
        registry.commit_transaction(b);
        assert_eq!(registry.maintenance_limit(), a_seq, "a is still committing");
        let view = registry.capture_view();
        registry.complete_commit(a);
        assert_eq!(
            registry.maintenance_limit(),
            view.floor() + 1,
            "the view excludes a"
        );
        assert_eq!(view.floor(), a_seq - 1);
        drop(view);
        assert_eq!(
            registry.maintenance_limit(),
            registry.get_current_sequence() + 1,
            "nothing in flight and no view"
        );
        assert_eq!(
            registry.safe_snapshot_cutoff(),
            registry.get_current_sequence()
        );
    }

    #[test]
    fn a_failed_commit_leaves_no_sequence_in_flight() {
        let registry = Arc::new(TransactionRegistry::new());
        let (a, _) = registry.begin_transaction();
        registry.start_commit(a);
        registry.abort_transaction(a);
        assert!(registry.transactions.lock().committing.is_empty());
        assert_eq!(
            registry.safe_snapshot_cutoff(),
            registry.get_current_sequence()
        );
    }

    #[test]
    fn committed_before_compares_a_read_committed_commit_with_its_sequence() {
        let registry = TransactionRegistry::new();
        let (x, _) = registry.begin_transaction();
        let x_seq = registry.commit_transaction(x);
        assert!(registry.is_committed_before(x, x_seq));
        assert!(
            !registry.is_committed_before(x, x_seq - 1),
            "a commit at {x_seq} was taken as before {}",
            x_seq - 1
        );
    }
}
