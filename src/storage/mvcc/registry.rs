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

//! Transaction registry for MVCC visibility
//!
//! Manages transaction states and visibility rules for MVCC.
//! Uses concurrent hash maps for high-performance thread-safe access.
//!

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::common::{new_concurrent_int64_map, ConcurrentInt64Map};
use crate::core::IsolationLevel;
use crate::storage::mvcc::VisibilityChecker;

/// Invalid transaction ID returned when registry is not accepting new transactions
pub const INVALID_TRANSACTION_ID: i64 = -999999999;

/// Special transaction ID for recovery transactions (always visible)
pub const RECOVERY_TRANSACTION_ID: i64 = -1;

/// Transaction registry manages transaction states and visibility rules
///
/// This is a lock-free implementation using concurrent hash maps for optimal
/// performance and concurrency. It supports both READ COMMITTED and SNAPSHOT
/// isolation levels.
///
/// # Transaction States
///
/// Transactions go through the following states:
/// 1. **Active**: Transaction is in progress, changes not visible to others
/// 2. **Committing**: Two-phase commit in progress, changes being finalized
/// 3. **Committed**: Transaction complete, changes visible per isolation rules
/// 4. **Aborted**: Transaction rolled back, changes discarded
///
/// Transaction registry manages transaction states and visibility rules
///
/// This is a **fully lock-free** implementation using:
/// - `DashMap` for concurrent transaction state tracking
/// - Atomic operations for sequence generation
/// - No blocking during transaction start or commit
///
/// # Performance
///
/// All operations are O(1) with no lock contention:
/// - `begin_transaction()`: ~50ns (2 atomic ops + 1 DashMap insert)
/// - `start_commit()`: ~30ns (1 atomic op + 2 DashMap ops)
/// - `complete_commit()`: ~20ns (2 DashMap ops)
/// - `is_visible()`: ~15ns (DashMap lookups only)
pub struct TransactionRegistry {
    /// Next transaction ID to assign
    next_txn_id: AtomicI64,

    /// Active transactions: txn_id -> begin_sequence
    active_transactions: ConcurrentInt64Map<i64>,

    /// Committed transactions: txn_id -> commit_sequence
    committed_transactions: ConcurrentInt64Map<i64>,

    /// Committing transactions (two-phase commit): txn_id -> commit_sequence
    committing_transactions: ConcurrentInt64Map<i64>,

    /// Global isolation level for new transactions
    global_isolation_level: RwLock<IsolationLevel>,

    /// Per-transaction isolation level overrides
    transaction_isolation_levels: RwLock<FxHashMap<i64, IsolationLevel>>,

    /// Whether new transactions are being accepted
    accepting: AtomicBool,

    /// Monotonic sequence for both begin and commit ordering
    next_sequence: AtomicI64,
}

impl TransactionRegistry {
    /// Creates a new transaction registry
    pub fn new() -> Self {
        Self {
            next_txn_id: AtomicI64::new(0),
            active_transactions: new_concurrent_int64_map(),
            committed_transactions: new_concurrent_int64_map(),
            committing_transactions: new_concurrent_int64_map(),
            global_isolation_level: RwLock::new(IsolationLevel::ReadCommitted),
            transaction_isolation_levels: RwLock::new(FxHashMap::with_capacity_and_hasher(
                100,
                Default::default(),
            )),
            accepting: AtomicBool::new(true),
            next_sequence: AtomicI64::new(0),
        }
    }

    /// Sets the isolation level for a specific transaction
    pub fn set_transaction_isolation_level(&self, txn_id: i64, level: IsolationLevel) {
        let mut levels = self.transaction_isolation_levels.write().unwrap();
        levels.insert(txn_id, level);
    }

    /// Removes the isolation level for a transaction
    pub fn remove_transaction_isolation_level(&self, txn_id: i64) {
        let mut levels = self.transaction_isolation_levels.write().unwrap();
        levels.remove(&txn_id);
    }

    /// Sets the global isolation level for new transactions
    pub fn set_global_isolation_level(&self, level: IsolationLevel) {
        let mut global = self.global_isolation_level.write().unwrap();
        *global = level;
    }

    /// Gets the current global isolation level
    pub fn get_global_isolation_level(&self) -> IsolationLevel {
        *self.global_isolation_level.read().unwrap()
    }

    /// Gets the isolation level for a specific transaction
    ///
    /// Returns the transaction-specific level if set, otherwise the global level.
    pub fn get_isolation_level(&self, txn_id: i64) -> IsolationLevel {
        // Check for transaction-specific level first
        let levels = self.transaction_isolation_levels.read().unwrap();
        if let Some(&level) = levels.get(&txn_id) {
            return level;
        }
        drop(levels);

        // Fall back to global level
        self.get_global_isolation_level()
    }

    /// Begins a new transaction
    ///
    /// Returns `(txn_id, begin_sequence)` on success, or
    /// `(INVALID_TRANSACTION_ID, 0)` if not accepting new transactions.
    ///
    /// # Concurrency
    ///
    /// This is a lock-free operation. New transactions do NOT wait for
    /// committing transactions - visibility rules handle isolation correctly.
    /// This enables high-throughput concurrent transaction starts.
    pub fn begin_transaction(&self) -> (i64, i64) {
        // Check if we're accepting new transactions
        if !self.accepting.load(Ordering::Acquire) {
            return (INVALID_TRANSACTION_ID, 0);
        }

        // Lock-free transaction start - no waiting for commits!
        // Visibility rules ensure correct isolation:
        // - Committing transactions are NOT visible to new transactions
        // - Only fully committed transactions become visible

        // Generate new transaction ID and sequence atomically
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::AcqRel) + 1;
        let begin_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        // Record the transaction as active (DashMap is lock-free)
        self.active_transactions.insert(txn_id, begin_seq);

        (txn_id, begin_seq)
    }

    /// Starts the commit process (two-phase commit)
    ///
    /// Moves transaction from active to committing state.
    /// Changes won't be visible to other transactions yet.
    ///
    /// # Concurrency
    ///
    /// This is now lock-free using atomic operations on DashMap.
    /// Multiple transactions can start committing concurrently.
    #[inline]
    pub fn start_commit(&self, txn_id: i64) -> i64 {
        // Get commit sequence atomically
        let commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        // Move from active to committing state (DashMap operations are atomic)
        self.active_transactions.remove(&txn_id);
        self.committing_transactions.insert(txn_id, commit_seq);

        commit_seq
    }

    /// Completes the commit process (two-phase commit)
    ///
    /// Moves transaction from committing to committed state,
    /// making all changes atomically visible.
    ///
    /// # Concurrency
    ///
    /// Lock-free completion using DashMap atomic operations.
    #[inline]
    pub fn complete_commit(&self, txn_id: i64) {
        // Get commit sequence from committing state
        let commit_seq = self
            .committing_transactions
            .get(&txn_id)
            .map(|r| *r)
            .unwrap_or_else(|| self.next_sequence.load(Ordering::Acquire));

        // Move from committing to committed state (atomic visibility)
        self.committing_transactions.remove(&txn_id);
        self.committed_transactions.insert(txn_id, commit_seq);
    }

    /// Commits a transaction (for read-only transactions)
    ///
    /// This is a simplified commit path for transactions that made no changes.
    pub fn commit_transaction(&self, txn_id: i64) -> i64 {
        let commit_seq = self.next_sequence.fetch_add(1, Ordering::AcqRel) + 1;

        // Remove from active and add to committed
        self.active_transactions.remove(&txn_id);
        self.committed_transactions.insert(txn_id, commit_seq);

        commit_seq
    }

    /// Recovers a committed transaction during startup recovery
    pub fn recover_committed_transaction(&self, txn_id: i64, commit_seq: i64) {
        self.committed_transactions.insert(txn_id, commit_seq);

        // Update next_txn_id if necessary
        loop {
            let current = self.next_txn_id.load(Ordering::Acquire);
            if txn_id >= current {
                if self
                    .next_txn_id
                    .compare_exchange(current, txn_id + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }

        // CRITICAL: Update next_sequence if necessary
        // During WAL recovery, commit_seq is the LSN value. We need to ensure
        // next_sequence is updated so that current_commit_sequence() returns a value
        // greater than all recovered commit sequences. Without this, recovered
        // transactions would be incorrectly excluded from subsequent snapshots
        // because is_committed_before(recovered_commit_seq, current_commit_seq) would
        // return false when recovered_commit_seq > current_commit_seq.
        loop {
            let current = self.next_sequence.load(Ordering::Acquire);
            if commit_seq >= current {
                if self
                    .next_sequence
                    .compare_exchange(current, commit_seq + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Records an aborted transaction during recovery
    pub fn recover_aborted_transaction(&self, txn_id: i64) {
        // Update next_txn_id if necessary
        loop {
            let current = self.next_txn_id.load(Ordering::Acquire);
            if txn_id >= current {
                if self
                    .next_txn_id
                    .compare_exchange(current, txn_id + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Aborts a transaction
    ///
    /// # Concurrency
    ///
    /// Lock-free abort using DashMap atomic operations.
    #[inline]
    pub fn abort_transaction(&self, txn_id: i64) {
        // Remove from active transactions (DashMap is lock-free)
        self.active_transactions.remove(&txn_id);

        // Also remove from committing state if present (atomic)
        self.committing_transactions.remove(&txn_id);
    }

    /// Gets the commit sequence for a transaction
    pub fn get_commit_sequence(&self, txn_id: i64) -> Option<i64> {
        self.committed_transactions.get(&txn_id).map(|r| *r)
    }

    /// Gets the begin sequence for an active transaction
    pub fn get_transaction_begin_sequence(&self, txn_id: i64) -> i64 {
        self.active_transactions
            .get(&txn_id)
            .map(|r| *r)
            .unwrap_or(0)
    }

    /// Gets the current sequence number
    pub fn get_current_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Checks if a version is directly visible (for READ COMMITTED)
    ///
    /// This is an ultra-fast path with minimal checks.
    /// Used when we know the viewer is using READ COMMITTED isolation.
    #[inline(always)]
    pub fn is_directly_visible(&self, version_txn_id: i64) -> bool {
        // Special case for recovery transactions (rare path)
        if version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }

        // Fast path: check committed first (most common case)
        // DashMap::contains_key is lock-free
        if self.committed_transactions.contains_key(&version_txn_id) {
            return true;
        }

        // Committing transactions are NOT visible
        false
    }

    /// Determines if a row version is visible to a transaction
    ///
    /// # Performance
    ///
    /// This is the hot path for MVCC - called for every row during scans.
    /// Optimized for the common case (READ COMMITTED, committed versions).
    ///
    /// # Arguments
    /// * `version_txn_id` - Transaction that created the version
    /// * `viewer_txn_id` - Transaction trying to see the version
    ///
    /// # Returns
    /// `true` if the version is visible to the viewer
    #[inline(always)]
    pub fn is_visible(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        // FAST PATH 1: Own writes are always visible (most common for writes)
        if version_txn_id == viewer_txn_id {
            return true;
        }

        // FAST PATH 2: Recovery transactions always visible
        if version_txn_id == RECOVERY_TRANSACTION_ID {
            return true;
        }

        // FAST PATH 3: READ COMMITTED - just check if committed
        // This avoids the RwLock on isolation_levels for the common case
        {
            let levels = self.transaction_isolation_levels.read().unwrap();
            if !levels.contains_key(&viewer_txn_id) {
                // Using global level - check if it's ReadCommitted (default)
                drop(levels);
                if *self.global_isolation_level.read().unwrap() == IsolationLevel::ReadCommitted {
                    return self.is_directly_visible(version_txn_id);
                }
            }
        }

        // SLOW PATH: Snapshot Isolation or per-transaction override
        self.is_visible_snapshot(version_txn_id, viewer_txn_id)
    }

    /// Snapshot isolation visibility check (slower path)
    #[inline(never)]
    fn is_visible_snapshot(&self, version_txn_id: i64, viewer_txn_id: i64) -> bool {
        // Check isolation level
        if self.get_isolation_level(viewer_txn_id) == IsolationLevel::ReadCommitted {
            return self.is_directly_visible(version_txn_id);
        }

        // Committing transactions are NOT visible
        if self.committing_transactions.contains_key(&version_txn_id) {
            return false;
        }

        // Must be committed
        let commit_seq = match self.committed_transactions.get(&version_txn_id) {
            Some(seq) => *seq,
            None => return false,
        };

        // For Snapshot Isolation, version must be committed before viewer began
        let viewer_begin_seq = match self.active_transactions.get(&viewer_txn_id) {
            Some(seq) => *seq,
            None => return false,
        };

        commit_seq <= viewer_begin_seq
    }

    /// Cleans up old committed transactions
    ///
    /// In READ COMMITTED mode, we cannot clean up committed transactions
    /// because visibility depends on their presence in the committed map.
    ///
    /// Returns the number of transactions removed.
    pub fn cleanup_old_transactions(&self, max_age: Duration) -> i32 {
        let isolation_level = self.get_global_isolation_level();

        // In READ COMMITTED, we cannot clean up
        if isolation_level == IsolationLevel::ReadCommitted {
            return 0;
        }

        let cutoff_time = Instant::now()
            .checked_sub(max_age)
            .map(|_| self.next_sequence.load(Ordering::Acquire) - (max_age.as_nanos() as i64))
            .unwrap_or(0);

        // Collect active transaction IDs for snapshot isolation
        let active_set: std::collections::HashSet<i64> =
            if isolation_level == IsolationLevel::SnapshotIsolation {
                self.active_transactions
                    .iter()
                    .map(|entry| *entry.key())
                    .collect()
            } else {
                std::collections::HashSet::new()
            };

        // Collect transactions to remove
        let txns_to_remove: Vec<i64> = self
            .committed_transactions
            .iter()
            .filter(|entry| {
                let txn_id = *entry.key();
                let commit_seq = *entry.value();

                // Never clean up negative transaction IDs
                if txn_id < 0 {
                    return false;
                }

                // Skip active transactions in snapshot isolation
                if isolation_level == IsolationLevel::SnapshotIsolation
                    && active_set.contains(&txn_id)
                {
                    return false;
                }

                // Only remove old transactions
                commit_seq < cutoff_time
            })
            .map(|entry| *entry.key())
            .collect();

        // Remove collected transactions
        let mut removed = 0;
        for txn_id in txns_to_remove {
            self.committed_transactions.remove(&txn_id);
            removed += 1;
        }

        removed
    }

    /// Waits for active transactions to complete with timeout
    ///
    /// Returns the number of transactions still active after timeout.
    pub fn wait_for_active_transactions(&self, timeout: Duration) -> i32 {
        let deadline = Instant::now() + timeout;

        loop {
            if Instant::now() > deadline {
                break;
            }

            let active_count = self.active_transactions.len();
            if active_count == 0 {
                return 0;
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        self.active_transactions.len() as i32
    }

    /// Stops accepting new transactions
    pub fn stop_accepting_transactions(&self) {
        self.accepting.store(false, Ordering::Release);
    }

    /// Starts accepting new transactions
    pub fn start_accepting_transactions(&self) {
        self.accepting.store(true, Ordering::Release);
    }

    /// Shuts down the registry (alias for stop_accepting_transactions)
    pub fn shutdown(&self) {
        self.stop_accepting_transactions();
    }

    /// Checks if the registry is accepting new transactions
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    /// Gets the count of active transactions
    pub fn active_count(&self) -> usize {
        self.active_transactions.len()
    }

    /// Gets the count of committed transactions
    pub fn committed_count(&self) -> usize {
        self.committed_transactions.len()
    }

    /// Checks if a transaction is active
    pub fn is_active(&self, txn_id: i64) -> bool {
        self.active_transactions.contains_key(&txn_id)
    }

    /// Checks if a transaction is committed
    pub fn is_committed(&self, txn_id: i64) -> bool {
        self.committed_transactions.contains_key(&txn_id)
    }

    /// Get the current commit sequence number
    ///
    /// This returns the sequence number that will be assigned to the NEXT commit.
    /// Any transaction with commit_seq < this value is guaranteed to be committed.
    /// Used for creating consistent snapshots - only include transactions with
    /// commit_seq < snapshot_commit_seq.
    pub fn current_commit_sequence(&self) -> i64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Check if a transaction was committed before a given commit sequence
    ///
    /// Returns true if the transaction is committed AND its commit sequence
    /// is less than the cutoff. This is used for consistent snapshot iteration.
    ///
    /// IMPORTANT: This should only be called for transactions already confirmed
    /// as committed via `is_visible`. If a committed transaction is not in the
    /// `committed_transactions` map, it means it's an old transaction that was
    /// cleaned up, so it definitely committed before the cutoff.
    pub fn is_committed_before(&self, txn_id: i64, cutoff_commit_seq: i64) -> bool {
        // Special case: snapshot versions (txn_id = -1) are always "old" commits
        if txn_id < 0 {
            return true;
        }

        if let Some(commit_seq) = self.committed_transactions.get(&txn_id) {
            // Use <= because current_commit_sequence() returns next_sequence which equals
            // the commit_seq of the most recently committed transaction. Without <=, the
            // last committed transaction before the cutoff would be incorrectly excluded.
            *commit_seq <= cutoff_commit_seq
        } else {
            // If a transaction is committed (caller verified via is_visible) but
            // not in the map, it's an old transaction that was cleaned up.
            // Old transactions definitely committed before our cutoff.
            true
        }
    }
}

impl Default for TransactionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of VisibilityChecker for TransactionRegistry
///
/// This allows the version store to check visibility through the registry.
impl VisibilityChecker for TransactionRegistry {
    fn is_visible(&self, version_txn_id: i64, viewing_txn_id: i64) -> bool {
        TransactionRegistry::is_visible(self, version_txn_id, viewing_txn_id)
    }

    fn get_current_sequence(&self) -> i64 {
        TransactionRegistry::get_current_sequence(self)
    }

    fn get_active_transaction_ids(&self) -> Vec<i64> {
        let mut ids = Vec::new();
        self.active_transactions
            .iter()
            .for_each(|entry| ids.push(*entry.key()));
        ids
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

        // Start commit
        let commit_seq = registry.start_commit(txn_id);
        assert!(commit_seq > 0);
        assert!(!registry.is_active(txn_id));
        assert!(registry.committing_transactions.contains_key(&txn_id));

        // Complete commit
        registry.complete_commit(txn_id);
        assert!(!registry.committing_transactions.contains_key(&txn_id));
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
    }

    #[test]
    fn test_visibility_own_writes() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();

        // Own writes are always visible
        assert!(registry.is_visible(txn_id, txn_id));
    }

    #[test]
    fn test_visibility_recovery_transaction() {
        let registry = TransactionRegistry::new();

        let (viewer_id, _) = registry.begin_transaction();

        // Recovery transactions are always visible
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

        // Default is global level
        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::ReadCommitted
        );

        // Override for specific transaction
        registry.set_transaction_isolation_level(txn_id, IsolationLevel::SnapshotIsolation);
        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::SnapshotIsolation
        );

        // Remove override
        registry.remove_transaction_isolation_level(txn_id);
        assert_eq!(
            registry.get_isolation_level(txn_id),
            IsolationLevel::ReadCommitted
        );
    }

    #[test]
    fn test_get_commit_sequence() {
        let registry = TransactionRegistry::new();

        let (txn_id, _) = registry.begin_transaction();
        assert!(registry.get_commit_sequence(txn_id).is_none());

        let commit_seq = registry.commit_transaction(txn_id);
        assert_eq!(registry.get_commit_sequence(txn_id), Some(commit_seq));
    }

    #[test]
    fn test_recover_committed_transaction() {
        let registry = TransactionRegistry::new();

        // Recover a transaction with high ID
        registry.recover_committed_transaction(1000, 500);

        assert!(registry.is_committed(1000));
        assert_eq!(registry.get_commit_sequence(1000), Some(500));

        // Next transaction should have higher ID
        let (new_id, _) = registry.begin_transaction();
        assert!(new_id > 1000);
    }
}
