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
//! - Single lock acquisition in hot path
//!
//! Memory: O(active_transactions + aborted_transactions)

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicUsize, Ordering};

use parking_lot::Mutex;

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

/// Thread-local cache size (512KB per thread).
const CACHE_SIZE: usize = 65536;

thread_local! {
    static COMMITTED_CACHE: RefCell<CommittedCache> = const { RefCell::new(CommittedCache::new()) };
}

/// Direct-mapped cache for committed transaction IDs.
struct CommittedCache {
    entries: [i64; CACHE_SIZE],
}

impl CommittedCache {
    #[inline]
    const fn new() -> Self {
        Self {
            entries: [0; CACHE_SIZE],
        }
    }

    #[inline(always)]
    fn contains(&self, txn_id: i64) -> bool {
        let idx = (txn_id as usize) & (CACHE_SIZE - 1);
        self.entries[idx] == txn_id
    }

    #[inline(always)]
    fn insert(&mut self, txn_id: i64) {
        let idx = (txn_id as usize) & (CACHE_SIZE - 1);
        self.entries[idx] = txn_id;
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
    /// All tracked transactions (Active, Committing, Aborted).
    /// Committed transactions are REMOVED from this map.
    transactions: Mutex<I64Map<TxnState>>,

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

    /// Whether new transactions are being accepted.
    accepting: AtomicBool,
}

impl TransactionRegistry {
    /// Creates a new transaction registry.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Creates a new transaction registry with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            transactions: Mutex::new(I64Map::with_capacity(capacity)),
            snapshot_seqs: Mutex::new(I64Map::new()),
            next_txn_id: AtomicI64::new(0),
            next_sequence: AtomicI64::new(0),
            global_isolation_level: AtomicU8::new(0),
            isolation_overrides: Mutex::new(I64Map::new()),
            override_count: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
        }
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

    /// Begins a new transaction.
    pub fn begin_transaction(&self) -> (i64, i64) {
        if !self.accepting.load(Ordering::Acquire) {
            return (INVALID_TRANSACTION_ID, 0);
        }

        let txn_id = self.next_txn_id.fetch_add(1, Ordering::AcqRel) + 1;
        let begin_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        self.transactions
            .lock()
            .insert(txn_id, TxnState::new_active(begin_seq));

        (txn_id, begin_seq)
    }

    /// Starts the commit process (two-phase commit).
    #[inline]
    pub fn start_commit(&self, txn_id: i64) -> i64 {
        let commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        if let Some(entry) = self.transactions.lock().get_mut(txn_id) {
            entry.set_committing(commit_seq);
        }

        commit_seq
    }

    /// Completes the commit process (two-phase commit).
    #[inline]
    pub fn complete_commit(&self, txn_id: i64) {
        let commit_seq = {
            let mut txns = self.transactions.lock();
            let seq = txns.get(txn_id).map(|e| e.commit_seq()).unwrap_or(0);
            txns.remove(txn_id);
            seq
        };

        // Store commit_seq for snapshot isolation if ANY transaction might need it
        if self.global_isolation_level.load(Ordering::Relaxed) == 1
            || self.override_count.load(Ordering::Relaxed) > 0
        {
            self.snapshot_seqs.lock().insert(txn_id, commit_seq);
        }
    }

    /// Commits a transaction (single-phase).
    pub fn commit_transaction(&self, txn_id: i64) -> i64 {
        let commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        self.transactions.lock().remove(txn_id);

        // Store commit_seq for snapshot isolation if ANY transaction might need it
        if self.global_isolation_level.load(Ordering::Relaxed) == 1
            || self.override_count.load(Ordering::Relaxed) > 0
        {
            self.snapshot_seqs.lock().insert(txn_id, commit_seq);
        }

        commit_seq
    }

    /// Aborts a transaction.
    #[inline]
    pub fn abort_transaction(&self, txn_id: i64) {
        // Replace with aborted marker (don't remove - need to track for visibility)
        self.transactions
            .lock()
            .insert(txn_id, TxnState::new_aborted());
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
        self.transactions
            .lock()
            .insert(txn_id, TxnState::new_aborted());

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
    /// Single lock acquisition for the hot path.
    #[inline(always)]
    fn check_committed(&self, txn_id: i64) -> bool {
        // Cache check first (no lock)
        if COMMITTED_CACHE.with(|c| c.borrow().contains(txn_id)) {
            return true;
        }

        // SINGLE lock acquisition - check if in transactions map
        // If in map = Active, Committing, or Aborted = not committed
        if self.transactions.lock().contains_key(txn_id) {
            return false;
        }

        // Not in map - committed if valid txn_id
        let next = self.next_txn_id.load(Ordering::Acquire);
        if txn_id > 0 && txn_id <= next {
            COMMITTED_CACHE.with(|c| c.borrow_mut().insert(txn_id));
            return true;
        }

        false
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
    /// Single lock acquisition for transactions map.
    #[cold]
    #[inline(never)]
    fn is_visible_snapshot(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        // Get both version and viewer state in SINGLE lock acquisition
        let (version_state, viewer_begin_seq) = {
            let txns = self.transactions.lock();

            // Get viewer's begin_seq (must be active to be viewing)
            let viewer_begin_seq = match txns.get(viewer_txn_id) {
                Some(state) if state.is_active_or_committing() => state.begin_seq(),
                _ => {
                    // Viewer not active - this is unusual but handle gracefully
                    // Try to get from snapshot_seqs as fallback
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

                // Committed - check commit_seq against viewer's begin_seq
                if let Some(&commit_seq) = self.snapshot_seqs.lock().get(version_txn_id) {
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

    /// Gets the current sequence number.
    pub fn get_current_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Gets the current commit sequence number.
    pub fn current_commit_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Runs garbage collection.
    ///
    /// Removes:
    /// - Old aborted entries (txn_id < min_active_txn_id - buffer)
    /// - Old snapshot_seqs entries (commit_seq < min_active_begin_seq)
    pub fn run_gc(&self) -> usize {
        // Get min values from active transactions
        let (min_begin_seq, min_txn_id) = {
            let txns = self.transactions.lock();
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

            (min_begin, min_id)
        };

        let mut removed = 0;

        // GC snapshot_seqs: remove old commit sequences
        {
            let mut seqs = self.snapshot_seqs.lock();
            let to_remove: Vec<i64> = seqs
                .iter()
                .filter(|(_, &commit_seq)| commit_seq < min_begin_seq)
                .map(|(txn_id, _)| txn_id)
                .collect();
            for txn_id in &to_remove {
                seqs.remove(*txn_id);
            }
            removed += to_remove.len();
        }

        // GC aborted: remove old aborted entries
        // Conservative buffer to handle in-flight queries
        let aborted_cutoff = min_txn_id.saturating_sub(10000);
        if aborted_cutoff > 0 {
            let mut txns = self.transactions.lock();
            let to_remove: Vec<i64> = txns
                .iter()
                .filter(|(id, state)| state.is_aborted() && *id < aborted_cutoff)
                .map(|(id, _)| id)
                .collect();
            for id in &to_remove {
                txns.remove(*id);
            }
            removed += to_remove.len();
        }

        // GC isolation overrides for completed transactions
        {
            let txns = self.transactions.lock();
            let mut overrides = self.isolation_overrides.lock();
            let to_remove: Vec<i64> = overrides
                .keys()
                .filter(|&txn_id| !txns.contains_key(txn_id))
                .collect();
            for txn_id in &to_remove {
                overrides.remove(*txn_id);
            }
            let removed_overrides = to_remove.len();
            if removed_overrides > 0 {
                self.override_count
                    .fetch_sub(removed_overrides, Ordering::Relaxed);
            }
            removed += removed_overrides;
        }

        removed
    }

    /// Cleans up old transactions (legacy API).
    pub fn cleanup_old_transactions(&self, _max_age: std::time::Duration) -> i32 {
        self.run_gc() as i32
    }

    /// Waits for active transactions to complete with timeout.
    pub fn wait_for_active_transactions(&self, timeout: std::time::Duration) -> i32 {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if std::time::Instant::now() > deadline {
                break;
            }

            let count = self.active_count();
            if count == 0 {
                return 0;
            }

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
    pub fn active_count(&self) -> usize {
        self.transactions
            .lock()
            .values()
            .filter(|s| s.status() == TxnStatus::Active)
            .count()
    }

    /// Gets the count of entries in snapshot_seqs.
    pub fn committed_count(&self) -> usize {
        self.snapshot_seqs.lock().len()
    }

    /// Checks if a transaction is active.
    pub fn is_active(&self, txn_id: i64) -> bool {
        self.transactions
            .lock()
            .get(txn_id)
            .map(|e| e.status() == TxnStatus::Active)
            .unwrap_or(false)
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

        // Check if still in transactions map
        if self.transactions.lock().contains_key(txn_id) {
            // If aborted or still active/committing, not committed
            return false;
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
    fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
        TransactionRegistry::is_visible(self, version_txn_id, viewing_txn_id)
    }

    fn get_current_sequence(&self) -> i64 {
        TransactionRegistry::get_current_sequence(self)
    }

    fn get_active_transaction_ids(&self) -> Vec<i64> {
        self.transactions
            .lock()
            .iter()
            .filter(|(_, s)| s.status() == TxnStatus::Active)
            .map(|(id, _)| id)
            .collect()
    }

    fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        TransactionRegistry::is_committed_before(self, txn_id, cutoff_commit_seq)
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
}
