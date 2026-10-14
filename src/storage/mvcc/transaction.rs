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

//! MVCC Transaction implementation
//!
//! Provides transaction semantics with two-phase commit protocol.
//!

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::core::{Error, IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::mvcc::txn_gate::{GateRef, TxnGate};
use crate::storage::mvcc::{get_fast_timestamp, TransactionRegistry};
use crate::storage::traits::{QueryResult, Table, Transaction};
use crate::storage::Expression;

/// DDL state captured at savepoint creation time.
/// Used to rollback CREATE/DROP TABLE operations when rolling back to a savepoint.
#[derive(Debug, Clone, Copy)]
struct SavepointDdlState {
    /// Number of created_tables entries at savepoint time
    created_tables_len: usize,
    /// Number of dropped_tables entries at savepoint time
    dropped_tables_len: usize,
}

/// State captured when a savepoint is created.
#[derive(Debug, Clone, Copy)]
struct SavepointState {
    /// Timestamp for rolling back DML changes
    timestamp: i64,
    /// DDL state for rolling back CREATE/DROP TABLE operations
    ddl_state: SavepointDdlState,
}

/// MVCC Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active and can perform operations
    Active,
    /// Transaction is being committed (two-phase commit)
    Committing,
    /// Transaction has been committed
    Committed,
    /// Transaction has been rolled back
    RolledBack,
}

/// MVCC Transaction implementation
pub struct MvccTransaction {
    /// Transaction ID
    id: i64,
    /// Transaction state
    state: TransactionState,
    /// Tables accessed in this transaction
    tables: FxHashMap<String, Box<dyn Table>>,
    /// Transaction-specific isolation level (if different from engine default)
    isolation_level: Option<IsolationLevel>,
    /// Reference to the transaction registry
    registry: Arc<TransactionRegistry>,
    /// Begin sequence number (for snapshot isolation)
    begin_seq: i64,
    /// Fast path cache for single table operations
    last_table_name: Option<String>,
    /// Engine reference for table operations (will be set by Engine)
    engine_operations: Option<Arc<dyn TransactionEngineOperations>>,
    /// Savepoints in creation order; the same name may appear more than once
    savepoints: Vec<(String, SavepointState)>,
    /// Tables created in this transaction (for rollback)
    created_tables: Vec<String>,
    /// Tables dropped in this transaction (for rollback - stores name and schema)
    dropped_tables: Vec<(String, Schema)>,
    /// The write gate the handles handed out admit through, made by the
    /// first of them
    gate: OnceLock<GateRef>,
    /// Set once the gate is made, so an executor-only transaction checks
    /// one flag rather than the gate
    handed_out: AtomicBool,
}

/// The refusal of a finish or savepoint rollback inside one of the
/// transaction's own statements, whose end its wait would never see
pub(crate) fn inside_own_statement() -> Error {
    Error::InvalidArgument(
        "a transaction cannot finish or roll back to a savepoint inside one of its own statements"
            .to_string(),
    )
}

/// Operations that require engine access
///
/// This trait allows the transaction to call back into the engine
/// without creating circular dependencies.
pub trait TransactionEngineOperations: Send + Sync {
    /// Get a table by name, initializing transaction-local version store;
    /// a handle given a `gate` admits its statements through it
    fn get_table_for_transaction(
        &self,
        txn_id: i64,
        table_name: &str,
        gate: Option<&GateRef>,
    ) -> Result<Box<dyn Table>>;

    /// Create a new table
    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn Table>>;

    /// Drop a table
    fn drop_table(&self, name: &str) -> Result<()>;

    /// List all tables
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Rename a table
    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Commit table changes
    fn commit_table(&self, txn_id: i64, table: &dyn Table) -> Result<()>;

    /// Rollback table changes
    fn rollback_table(&self, txn_id: i64, table: &dyn Table);

    /// Record commit in WAL
    fn record_commit(&self, txn_id: i64) -> Result<()>;

    /// Record rollback in WAL
    fn record_rollback(&self, txn_id: i64) -> Result<()>;

    /// Check if transaction has any pending DML changes (without allocating)
    fn has_pending_dml_changes(&self, txn_id: i64) -> bool;

    /// Commit all tables for a transaction at once (includes WAL recording).
    ///
    /// Returns `(any_committed, optional_error)`:
    /// - `(false, None)`: no tables had changes, nothing to do
    /// - `(true, None)`: all tables committed successfully
    /// - `(true, Some(e))`: partial commit - some tables committed before error
    /// - `(false, Some(e))`: error before any table committed
    ///
    /// Callers MUST complete_commit if any_committed is true, even on error,
    /// to avoid orphaning already-committed rows.
    ///
    /// With `handed_out`, handles that may outlive the transaction keep
    /// its entry for the finish or rollback that follows.
    fn commit_all_tables(
        &self,
        txn_id: i64,
        handed_out: bool,
    ) -> (bool, Option<crate::core::Error>);

    /// Rollback all tables for a transaction at once
    /// This cleans up the transaction's entries in txn_version_stores;
    /// with `handed_out`, the rows and claims of its stores go now
    fn rollback_all_tables(&self, txn_id: i64, handed_out: bool);

    /// Lets the stores of a transaction with handles handed out go once its
    /// outcome is settled, with the claims of a table it did not write
    fn finish_all_tables(&self, txn_id: i64);

    /// Makes the transaction's pending cold tombstones visible under its
    /// commit sequence, once its commit marker is durable; a commit whose
    /// marker fails discards them instead, so the rows they name stay visible
    fn publish_pending_tombstones(&self, txn_id: i64, commit_seq: u64) {
        let _ = (txn_id, commit_seq);
    }

    /// Drops the transaction's pending cold tombstones after its commit
    /// marker failed
    fn discard_pending_tombstones(&self, txn_id: i64) {
        let _ = txn_id;
    }

    /// Marks every table the transaction writes as publishing, from before
    /// its index updates until the transaction is visible or undone
    fn begin_publish(&self, txn_id: i64) -> super::version_store::PublishHold {
        let _ = txn_id;
        super::version_store::PublishHold::default()
    }

    /// Waits while a table the transaction writes holds more hot bytes than
    /// its limit allows, so a seal can bring it back under. No fence is held.
    fn wait_for_hot_admission(&self, txn_id: i64) {
        let _ = txn_id;
    }

    /// Asks for a seal when a table the now visible commit wrote is at the
    /// hot row or byte limit
    fn request_seal_if_over(&self, hold: &super::version_store::PublishHold) {
        let _ = hold;
    }

    /// Discard later DML across all touched tables, including cold claims.
    fn rollback_dml_after(&self, txn_id: i64, timestamp: i64);

    /// Changes a column's definition as a DDL statement does: under the DDL
    /// guard, in every layer, with the table closed
    fn change_column_type(&self, change: &super::engine::ColumnTypeChange<'_>) -> Result<()> {
        let _ = change;
        Err(Error::NotSupported(
            "changing a column's definition needs the engine".to_string(),
        ))
    }

    /// Defer table cleanup to background thread (avoids synchronous deallocation)
    /// Default implementation drops synchronously
    fn defer_table_cleanup(&self, _tables: Vec<Box<dyn Table>>) {
        // Default: just drop synchronously (tables dropped when _tables goes out of scope)
    }

    /// Acquire the seal fence shared lock. Commits hold this to signal they
    /// are in-flight. The checkpoint micro-seal acquires the exclusive lock,
    /// waiting for all in-flight commits to complete before draining hot rows.
    /// Returns None for in-memory engines (no persistence, no seal fence needed).
    fn acquire_seal_fence(&self) -> Option<SealFenceGuard> {
        None
    }
}

/// RAII guard that holds a seal fence read lock. When dropped, the read lock
/// is released. The checkpoint micro-seal acquires the write lock, which
/// blocks until all SealFenceGuards are dropped.
pub struct SealFenceGuard {
    /// Keep the Arc alive so the lock outlives the guard.
    _lock: Arc<parking_lot::RwLock<()>>,
    /// Raw pointer to avoid lifetime issues with RwLockReadGuard.
    /// SAFETY: The Arc above keeps the RwLock alive.
    _raw: *const (),
}

// SAFETY: SealFenceGuard only holds an Arc (Send+Sync) and a raw read-lock
// that is released on drop. The guard is created and dropped on the same
// thread (the commit thread). The raw pointer is not dereferenced.
unsafe impl Send for SealFenceGuard {}
unsafe impl Sync for SealFenceGuard {}

impl SealFenceGuard {
    pub fn new(lock: Arc<parking_lot::RwLock<()>>) -> Self {
        // Acquire the read lock via the raw API so we can control the lifetime.
        // parking_lot::RawRwLock::lock_shared is balanced by unlock_shared in Drop.
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: lock_shared() is always safe to call. We balance it with
        // unlock_shared() in Drop. The Arc keeps the RwLock alive.
        unsafe { lock.raw().lock_shared() };
        Self {
            _raw: std::ptr::null(),
            _lock: lock,
        }
    }
}

impl Drop for SealFenceGuard {
    fn drop(&mut self) {
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: We acquired lock_shared() in new(), this is the balancing release.
        unsafe { self._lock.raw().unlock_shared() };
    }
}

impl MvccTransaction {
    /// Creates a new MVCC transaction
    pub fn new(id: i64, begin_seq: i64, registry: Arc<TransactionRegistry>) -> Self {
        Self {
            id,
            state: TransactionState::Active,
            tables: FxHashMap::default(),
            isolation_level: None,
            registry,
            begin_seq,
            last_table_name: None,
            engine_operations: None,
            savepoints: Vec::new(),
            created_tables: Vec::new(),
            dropped_tables: Vec::new(),
            gate: OnceLock::new(),
            handed_out: AtomicBool::new(false),
        }
    }

    /// Records the level the registry took for this transaction at its begin
    pub(crate) fn begun_at(&mut self, level: IsolationLevel) {
        self.isolation_level = Some(level);
    }

    /// Sets the engine operations callback
    pub fn set_engine_operations(&mut self, ops: Arc<dyn TransactionEngineOperations>) {
        self.engine_operations = Some(ops);
    }

    /// Returns the begin sequence number
    pub fn begin_seq(&self) -> i64 {
        self.begin_seq
    }

    /// Returns the current transaction state
    pub fn state(&self) -> TransactionState {
        self.state
    }

    /// Returns the isolation level for this transaction
    pub fn get_isolation_level(&self) -> IsolationLevel {
        self.isolation_level
            .unwrap_or_else(|| self.registry.get_global_isolation_level())
    }

    /// Check if transaction is active
    fn check_active(&self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(Error::TransactionClosed);
        }
        Ok(())
    }

    /// Get engine operations, returning error if not set
    fn get_engine_ops(&self) -> Result<&Arc<dyn TransactionEngineOperations>> {
        self.engine_operations
            .as_ref()
            .ok_or_else(|| Error::internal("engine operations not set"))
    }

    /// Clean up transaction resources
    fn cleanup(&mut self) {
        // Clear fast path cache
        self.last_table_name = None;

        // Clear tables
        self.tables.clear();

        // Clear DDL tracking
        self.created_tables.clear();
        self.dropped_tables.clear();

        // Remove transaction isolation level from registry
        self.registry.remove_transaction_isolation_level(self.id);
    }

    /// The transaction's gate when this thread is inside one of its
    /// statements
    fn own_statement_gate(&self) -> Option<GateRef> {
        if !self.handed_out() {
            return None;
        }
        self.gate.get().filter(|gate| gate.held_here()).cloned()
    }

    /// Whether a handle was handed out, which may outlive the transaction
    fn handed_out(&self) -> bool {
        self.handed_out.load(Ordering::Relaxed)
    }

    /// Refused inside one of the transaction's own statements
    fn refuse_inside_own_statement(&self) -> Result<()> {
        if self.own_statement_gate().is_some() {
            return Err(inside_own_statement());
        }
        Ok(())
    }

    /// Admits no statement of the transaction from now on, then waits for
    /// those in flight to finish; refused inside one of its own statements
    #[inline(always)]
    fn close_writes(&self) -> Result<()> {
        if !self.handed_out() {
            return Ok(());
        }
        match self.gate.get() {
            Some(gate) => Self::close_gate(gate),
            None => Ok(()),
        }
    }

    #[inline(never)]
    fn close_gate(gate: &GateRef) -> Result<()> {
        if gate.held_here() {
            return Err(inside_own_statement());
        }
        gate.close();
        gate.wait_idle();
        Ok(())
    }

    /// Holds the transaction's new statements off while those in flight
    /// finish; the gate, to hold on to while the hold lasts
    fn hold_writes(&self) -> Option<GateRef> {
        self.gate.get().cloned()
    }

    /// Roll back DDL operations (CREATE TABLE / DROP TABLE) in reverse order.
    /// Used by both explicit rollback() and implicit Drop.
    fn rollback_ddl(
        ops: &dyn TransactionEngineOperations,
        created_tables: &[String],
        dropped_tables: &[(String, Schema)],
    ) {
        // Drop tables that were created in this transaction
        for table_name in created_tables.iter().rev() {
            if let Err(e) = ops.drop_table(table_name) {
                eprintln!(
                    "Warning: Failed to drop table '{}' during DDL rollback: {}",
                    table_name, e
                );
            }
        }

        // Recreate tables that were dropped in this transaction
        for (table_name, schema) in dropped_tables.iter().rev() {
            if let Err(e) = ops.create_table(table_name, schema.clone()) {
                eprintln!(
                    "Warning: Failed to recreate table '{}' during DDL rollback: {}",
                    table_name, e
                );
            }
        }
    }

    /// Check if this is a read-only transaction
    fn is_read_only(&self) -> bool {
        // Check for DDL changes
        if !self.created_tables.is_empty() || !self.dropped_tables.is_empty() {
            return false;
        }
        // Check for DML changes via engine operations
        if let Some(ops) = &self.engine_operations {
            if ops.has_pending_dml_changes(self.id) {
                return false;
            }
        }
        true
    }

    /// Creates a savepoint with the given name
    ///
    /// Records the current timestamp and DDL state so we can rollback to this point later.
    /// A name already in use starts a new, more recent savepoint of that name.
    pub fn create_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        self.refuse_inside_own_statement()?;
        // Never in the middle of a statement, so a rollback to it keeps
        // or drops whole statements
        let gate = self.hold_writes();
        let _hold = gate.as_ref().map(|gate| gate.hold());
        let timestamp = get_fast_timestamp();
        let ddl_state = SavepointDdlState {
            created_tables_len: self.created_tables.len(),
            dropped_tables_len: self.dropped_tables.len(),
        };
        self.savepoints.push((
            name.to_string(),
            SavepointState {
                timestamp,
                ddl_state,
            },
        ));
        Ok(())
    }

    /// Position of the most recent savepoint with this name
    fn savepoint_position(&self, name: &str) -> Result<usize> {
        self.savepoints
            .iter()
            .rposition(|(sp_name, _)| sp_name == name)
            .ok_or_else(|| Error::invalid_argument(format!("savepoint '{}' does not exist", name)))
    }

    /// Releases a savepoint, and every savepoint created after it, without rolling back
    ///
    /// The changes made after the savepoint remain intact.
    /// Returns an error if the savepoint doesn't exist.
    pub fn release_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        let position = self.savepoint_position(name)?;
        self.savepoints.truncate(position);
        Ok(())
    }

    /// Rolls back to a savepoint, discarding all changes made after it
    ///
    /// All local DML changes with timestamps after the savepoint are discarded.
    /// DDL operations (CREATE/DROP TABLE) after the savepoint are also reversed.
    /// Savepoints created after it are removed; the savepoint itself stays.
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        self.refuse_inside_own_statement()?;
        let gate = self.hold_writes();
        let _hold = gate.as_ref().map(|gate| gate.hold());

        let position = self.savepoint_position(name)?;
        let sp_state = self.savepoints[position].1;

        if let Some(ops) = &self.engine_operations {
            ops.rollback_dml_after(self.id, sp_state.timestamp);
        }

        // Rollback DDL: undo CREATE TABLEs after savepoint
        if let Some(ops) = &self.engine_operations {
            // Tables created after savepoint need to be dropped
            while self.created_tables.len() > sp_state.ddl_state.created_tables_len {
                if let Some(table_name) = self.created_tables.pop() {
                    if let Err(e) = ops.drop_table(&table_name) {
                        eprintln!(
                            "Warning: Failed to drop table '{}' during savepoint rollback: {}",
                            table_name, e
                        );
                    }
                }
            }

            // Tables dropped after savepoint need to be recreated
            while self.dropped_tables.len() > sp_state.ddl_state.dropped_tables_len {
                if let Some((table_name, schema)) = self.dropped_tables.pop() {
                    if let Err(e) = ops.create_table(&table_name, schema) {
                        eprintln!(
                            "Warning: Failed to recreate table '{}' during savepoint rollback: {}",
                            table_name, e
                        );
                    }
                }
            }
        }

        self.savepoints.truncate(position + 1);

        Ok(())
    }

    /// Check if a savepoint exists
    pub fn has_savepoint(&self, name: &str) -> bool {
        self.savepoint_position(name).is_ok()
    }

    /// Gets the timestamp of the most recent savepoint with this name
    pub fn get_savepoint_ts(&self, name: &str) -> Option<i64> {
        let position = self.savepoint_position(name).ok()?;
        Some(self.savepoints[position].1.timestamp)
    }
}

impl Transaction for MvccTransaction {
    fn id(&self) -> i64 {
        self.id
    }

    fn begin(&mut self) -> Result<()> {
        // No-op for compatibility - transaction is initialized in new()
        self.check_active()
    }

    fn commit(&mut self) -> Result<()> {
        self.check_active()?;
        // Before anything reads the local changes or takes the seal fence,
        // which a statement in flight holds
        self.close_writes()?;

        // Update state to committing
        self.state = TransactionState::Committing;

        // Check if read-only: no DDL changes and no DML changes
        // Use has_pending_dml_changes() to avoid allocating Vec<Box<dyn Table>>
        let has_dml_changes = self
            .engine_operations
            .as_ref()
            .is_some_and(|ops| ops.has_pending_dml_changes(self.id));

        let is_read_only =
            self.created_tables.is_empty() && self.dropped_tables.is_empty() && !has_dml_changes;

        // Two-phase commit protocol
        if !is_read_only {
            if let Some(ops) = &self.engine_operations {
                ops.wait_for_hot_admission(self.id);
            }
            // Acquire seal fence shared lock. This signals to the checkpoint
            // micro-seal that a commit is in-flight. The micro-seal waits for
            // all in-flight commits to finish before draining hot rows.
            // Held until complete_commit (end of this block).
            let _seal_guard = self
                .engine_operations
                .as_ref()
                .and_then(|ops| ops.acquire_seal_fence());

            // Phase 1: Start commit - mark transaction as "committing"
            self.registry.start_commit(self.id);
            // Held until the transaction is visible or undone: readers that
            // trust an index's order stand down while it publishes
            let publish = self
                .engine_operations
                .as_ref()
                .map(|ops| ops.begin_publish(self.id));

            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::commit_counted();

            // Phase 2: Commit all tables - apply local changes to global store
            // This now includes WAL recording internally (before each table commit)
            if let Some(ops) = &self.engine_operations {
                let (_, error) = ops.commit_all_tables(self.id, self.handed_out());
                if let Some(e) = error {
                    // A table that failed leaves the transaction unfinished, and
                    // nothing of it is visible yet: the tables that applied
                    // their versions and index updates take them back, the
                    // cold tombstones are dropped, and the transaction aborts
                    // as if no table had committed, its row claims released.
                    // The WAL records written so far carry no marker, so
                    // recovery drops them too.
                    if let Some(hold) = &publish {
                        hold.undo_publication();
                    }
                    ops.discard_pending_tombstones(self.id);
                    self.registry.abort_transaction(self.id);
                    ops.rollback_all_tables(self.id, self.handed_out());
                    self.state = TransactionState::RolledBack;
                    self.cleanup();
                    return Err(e);
                }
            }

            // Phase 3: Record commit marker in WAL BEFORE making changes visible.
            // This ensures crash recovery sees the COMMIT marker even if we crash
            // before complete_commit(). WAL is only read during recovery, so writing
            // the marker before visibility doesn't affect normal operation.
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::commit_marker_next();
            if let Some(ops) = &self.engine_operations {
                if let Err(e) = ops.record_commit(self.id) {
                    // WAL commit marker failed. Phase 2 data is in the version store
                    // but not yet visible (complete_commit hasn't run). Abort so GC
                    // can reclaim the orphaned entries and active_txn_count is correct.
                    // On recovery, WAL has no COMMIT marker → entries are discarded.
                    // The store and its indexes already describe the aborted
                    // rows: take that back
                    if let Some(hold) = &publish {
                        hold.undo_publication();
                    }
                    // The cold rows the commit deleted or replaced stay: their
                    // tombstones were held back for this marker
                    ops.discard_pending_tombstones(self.id);
                    self.registry.abort_transaction(self.id);
                    if self.handed_out() {
                        ops.finish_all_tables(self.id);
                    }
                    self.state = TransactionState::RolledBack;
                    self.cleanup();
                    return Err(e);
                }
                // The marker is durable: the cold rows it deleted or replaced
                // go out of sight with the hot rows it made visible
                let commit_seq = self.registry.get_committing_sequence(self.id) as u64;
                ops.publish_pending_tombstones(self.id, commit_seq);
            }

            // Phase 4: Complete commit - make changes visible in registry
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::commit_becoming_visible();
            self.registry.complete_commit(self.id);
            if let (Some(ops), Some(hold)) = (&self.engine_operations, &publish) {
                // Nothing of the commit will be taken back now
                hold.release_publication();
                ops.request_seal_if_over(hold);
            }
            if let (Some(ops), true) = (&self.engine_operations, self.handed_out()) {
                ops.finish_all_tables(self.id);
            }
        } else {
            // Read-only transaction - just mark as committed in registry
            self.registry.complete_commit(self.id);
            // Nothing to publish; drop the stores its reads opened, as a
            // rollback does
            if let Some(ops) = &self.engine_operations {
                ops.rollback_all_tables(self.id, self.handed_out());
            }
        }

        // Mark as committed
        self.state = TransactionState::Committed;
        self.cleanup();

        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.check_active()?;
        self.close_writes()?;

        // Check if read-only before rolling back
        let is_read_only = self.is_read_only();

        // Mark transaction as aborted in registry
        if is_read_only {
            self.registry.forget_transaction(self.id);
        } else {
            self.registry.abort_transaction(self.id);
        }

        // Rollback DDL operations (CREATE TABLE / DROP TABLE) in reverse order
        if let Some(ops) = &self.engine_operations {
            Self::rollback_ddl(ops.as_ref(), &self.created_tables, &self.dropped_tables);
        }

        // Rollback all tables - discard local changes
        for table in self.tables.values_mut() {
            table.rollback();
        }

        // Notify engine of rollback (per-table callbacks)
        if let Some(ops) = &self.engine_operations {
            for table in self.tables.values() {
                ops.rollback_table(self.id, table.as_ref());
            }
            // Clean up txn_version_stores entry to prevent memory leak
            ops.rollback_all_tables(self.id, self.handed_out());
        }

        // Record in WAL if not read-only
        if !is_read_only {
            if let Some(ops) = &self.engine_operations {
                let _ = ops.record_rollback(self.id);
            }
        }

        // Mark as rolled back
        self.state = TransactionState::RolledBack;
        self.cleanup();
        Ok(())
    }

    fn create_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::create_savepoint(self, name)
    }

    fn release_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::release_savepoint(self, name)
    }

    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::rollback_to_savepoint(self, name)
    }

    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64> {
        // Delegate to the inherent method
        MvccTransaction::get_savepoint_ts(self, name)
    }

    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        self.check_active()?;

        // Set transaction-specific isolation level
        self.isolation_level = Some(level);

        // Also set in registry
        self.registry
            .set_transaction_isolation_level(self.id, level);

        Ok(())
    }

    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn Table>> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        let table = ops.create_table(name, schema)?;

        // Track for rollback - store the table name
        self.created_tables.push(name.to_lowercase());

        Ok(table)
    }

    /// Drop a table within this transaction.
    ///
    /// # Warning
    /// DROP TABLE is NOT fully transactional. While the table structure can be
    /// recreated on rollback, the data CANNOT be recovered. This is similar to
    /// PostgreSQL's behavior where certain DDL operations are destructive.
    ///
    /// If you need to safely remove all data with rollback support, use
    /// `DELETE FROM table_name` or `TRUNCATE TABLE table_name` instead.
    fn drop_table(&mut self, name: &str) -> Result<()> {
        self.check_active()?;

        // Before dropping, get the schema so we can recreate on rollback
        // We need to get the table to access its schema
        // Scope the borrow to allow later mutable operations
        let schema = {
            let ops = self.get_engine_ops()?;
            let table = ops.get_table_for_transaction(self.id, name, None)?;
            table.schema().clone()
        };

        // Save schema for potential rollback (note: data cannot be recovered)
        self.dropped_tables.push((name.to_lowercase(), schema));

        // Now drop the table
        let ops = self.get_engine_ops()?;
        ops.drop_table(name)?;

        // Remove from cache
        self.tables.remove(name);

        // Clear fast path cache if needed
        if let Some(last_name) = &self.last_table_name {
            if last_name == name {
                self.last_table_name = None;
            }
        }

        Ok(())
    }

    fn get_table(&self, name: &str) -> Result<Box<dyn Table>> {
        self.check_active()?;

        // Note: Cached tables would require Clone on Table trait, which isn't object-safe.
        // For now, always get from engine (engine will handle caching internally).
        // The tables HashMap is used for tracking which tables were accessed for commit/rollback.

        // Get from engine
        let ops = self.get_engine_ops()?;
        let gate = self.gate.get_or_init(|| {
            self.handed_out.store(true, Ordering::Relaxed);
            TxnGate::open()
        });
        ops.get_table_for_transaction(self.id, name, Some(gate))
    }

    fn get_statement_table(&self, name: &str) -> Result<Box<dyn Table>> {
        self.check_active()?;
        let ops = self.get_engine_ops()?;
        ops.get_table_for_transaction(self.id, name, None)
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        ops.list_tables()
    }

    fn rename_table(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        ops.rename_table(old_name, new_name)?;

        // Update cache if needed
        if let Some(table) = self.tables.remove(old_name) {
            self.tables.insert(new_name.to_string(), table);
        }

        // Update fast path cache
        if let Some(last_name) = &self.last_table_name {
            if last_name == old_name {
                self.last_table_name = Some(new_name.to_string());
            }
        }

        Ok(())
    }

    fn create_table_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
    ) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        table.create_index(index_name, &col_refs, is_unique)
    }

    fn drop_table_index(&mut self, table_name: &str, index_name: &str) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.drop_index(index_name)
    }

    fn create_table_btree_index(
        &mut self,
        table_name: &str,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.create_btree_index(column_name, is_unique, custom_name)
    }

    fn drop_table_btree_index(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.drop_btree_index(column_name)
    }

    fn add_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        self.check_active()?;

        let mut table = self.get_table(table_name)?;
        table.create_column(&column.name, column.data_type, column.nullable)
    }

    fn drop_table_column(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        self.check_active()?;

        let mut table = self.get_table(table_name)?;
        table.drop_column(column_name)
    }

    fn rename_table_column(
        &mut self,
        table_name: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        self.check_active()?;

        let mut table = self.get_table(table_name)?;
        table.rename_column(old_name, new_name)
    }

    fn modify_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        self.check_active()?;

        let ops = self.engine_operations.as_ref().ok_or_else(|| {
            Error::NotSupported("changing a column's definition needs the engine".to_string())
        })?;
        ops.change_column_type(&super::engine::ColumnTypeChange {
            table: table_name,
            column: &column.name,
            data_type: column.data_type,
            nullable: column.nullable,
            vector_dimensions: column.vector_dimensions,
        })
    }

    fn select(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select(&col_refs, expr)
    }

    fn select_with_aliases(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select_with_aliases(&col_refs, expr, aliases)
    }

    fn select_as_of(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select_as_of(&col_refs, expr, temporal_type, temporal_value)
    }
}

impl std::fmt::Debug for MvccTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MvccTransaction")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("begin_seq", &self.begin_seq)
            .finish()
    }
}

// Ensure transaction is rolled back on drop if still active
impl Drop for MvccTransaction {
    fn drop(&mut self) {
        if self.state == TransactionState::Active {
            // Dropped from inside one of its own statements, it cannot wait
            // for that statement: the last statement out cleans up
            let own_gate = self.own_statement_gate();
            if own_gate.is_none() {
                // Not inside its own statement, so the close is not refused
                let _ = self.close_writes();
            }

            // Silent rollback on drop
            if self.is_read_only() {
                self.registry.forget_transaction(self.id);
            } else {
                self.registry.abort_transaction(self.id);
            }

            if let Some(ops) = &self.engine_operations {
                if let Some(gate) = own_gate {
                    let (ops, id) = (Arc::clone(ops), self.id);
                    let created = std::mem::take(&mut self.created_tables);
                    let dropped = std::mem::take(&mut self.dropped_tables);
                    gate.finish_after_statements(Box::new(move || {
                        Self::rollback_ddl(ops.as_ref(), &created, &dropped);
                        ops.rollback_all_tables(id, true);
                    }));
                } else {
                    // Roll back DDL operations (CREATE TABLE / DROP TABLE)
                    Self::rollback_ddl(ops.as_ref(), &self.created_tables, &self.dropped_tables);

                    // Clean up txn_version_stores to prevent memory leak
                    // This is critical for read-only transactions that call get_table()
                    // but are dropped without explicit commit/rollback
                    ops.rollback_all_tables(self.id, self.handed_out());
                }
            }

            self.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.id(), txn_id);
        assert_eq!(txn.begin_seq(), begin_seq);
        assert_eq!(txn.state(), TransactionState::Active);
    }

    #[test]
    fn test_transaction_state_transitions() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.state(), TransactionState::Active);

        // Begin should be no-op
        txn.begin().unwrap();
        assert_eq!(txn.state(), TransactionState::Active);

        // Commit
        txn.commit().unwrap();
        assert_eq!(txn.state(), TransactionState::Committed);

        // Should fail to begin after commit
        assert!(txn.begin().is_err());
    }

    #[test]
    fn test_transaction_rollback() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.state(), TransactionState::Active);

        // Rollback
        txn.rollback().unwrap();
        assert_eq!(txn.state(), TransactionState::RolledBack);

        // Should fail to begin after rollback
        assert!(txn.begin().is_err());
    }

    #[test]
    fn test_transaction_isolation_level() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // Default isolation level
        let default_level = txn.get_isolation_level();
        assert_eq!(default_level, IsolationLevel::ReadCommitted);

        // Set transaction-specific level
        txn.set_isolation_level(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(txn.get_isolation_level(), IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_transaction_double_commit() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // First commit should succeed
        txn.commit().unwrap();

        // Second commit should fail
        assert!(txn.commit().is_err());
    }

    #[test]
    fn test_transaction_commit_after_rollback() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // Rollback first
        txn.rollback().unwrap();

        // Commit should fail
        assert!(txn.commit().is_err());
    }

    #[test]
    fn test_transaction_debug() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        let debug_str = format!("{:?}", txn);
        assert!(debug_str.contains("MvccTransaction"));
        assert!(debug_str.contains("Active"));
    }
}
