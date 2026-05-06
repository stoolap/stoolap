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
use std::sync::Arc;

use crate::core::{Error, IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::mvcc::{get_fast_timestamp, TransactionRegistry};
use crate::storage::traits::{
    QueryResult, ReadTable, ReadTransaction, WriteTable, WriteTransaction,
};
use crate::storage::Expression;

/// DDL state captured at savepoint creation time.
#[derive(Debug, Clone, Copy)]
struct SavepointDdlState {
    /// Length of `ddl_log` at savepoint time; rollback walks the suffix in reverse.
    ddl_log_len: usize,
}

/// Pre-drop snapshot used to undo a transactional DROP TABLE on rollback.
#[derive(Debug, Clone)]
pub struct DropSnapshot {
    /// Parent table schema, recreated on rollback.
    pub parent_schema: Schema,
    /// (child_table, pre-strip schema) for each child whose FK referenced the parent.
    pub child_schemas: Vec<(String, Schema)>,
    /// Serialized `IndexMetadata` for every non-PK index on the dropped table.
    pub indexes: Vec<Vec<u8>>,
}

/// DDL operation queued for durable WAL write at commit time, gated by the txn's commit marker.
#[derive(Debug, Clone)]
pub enum DeferredDdlOp {
    Create {
        name: String,
        schema_data: Vec<u8>,
    },
    Drop {
        name: String,
    },
    CreateIndex {
        table_name: String,
        metadata: Vec<u8>,
    },
}

/// One DDL op in an ordered log; rollback walks in reverse so CREATE/DROP sequences
/// against the same name resolve correctly without coalescing.
#[derive(Debug, Clone)]
enum DdlOp {
    Create(String, Schema),
    /// DROP with pre-drop snapshot. No durable compensation on rollback (durable DROP
    /// is only emitted from the commit path).
    Drop(String, DropSnapshot),
    /// Pre-serialized `IndexMetadata` for the deferred commit-time WAL flush.
    CreateIndex(String, Vec<u8>),
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
    tables: FxHashMap<String, Box<dyn WriteTable>>,
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
    /// Savepoints: maps savepoint name to state (timestamp + DDL snapshot)
    savepoints: FxHashMap<String, SavepointState>,
    /// Ordered DDL log; rollback walks in reverse to apply inverses.
    ddl_log: Vec<DdlOp>,
    /// Shared hold on the transactional-DDL fence, acquired lazily on the first
    /// CREATE/DROP and released on resolve. Blocks checkpoint's `rerecord_ddl_to_wal`
    /// from snapshotting a half-mutated catalog.
    transactional_ddl_guard: Option<TransactionalDdlFenceGuard>,
}

/// Operations that require engine access
///
/// This trait allows the transaction to call back into the engine
/// without creating circular dependencies.
pub trait TransactionEngineOperations: Send + Sync {
    /// Get a table by name, initializing transaction-local version store
    fn get_table_for_transaction(
        &self,
        txn_id: i64,
        table_name: &str,
    ) -> Result<Box<dyn WriteTable>>;

    /// Create a new table
    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn WriteTable>>;

    /// Drop a table within a transaction (in-memory only). Durable DropTable WAL is
    /// deferred to the commit path; physical file deletion is deferred to
    /// `finalize_committed_drops`. Returns the pre-drop snapshot needed for rollback.
    fn drop_table(&self, name: &str) -> Result<DropSnapshot>;

    /// Emit durable DDL WAL entries under `txn_id` (no auto-commit marker). Visibility
    /// is gated by the txn's commit marker; orphans without a marker are skipped on recovery.
    fn flush_transactional_ddl(&self, txn_id: i64, ops: &[DeferredDdlOp]) -> Result<()>;

    /// Run post-commit physical side effects of every committed DROP: clear segment
    /// state and delete on-disk volume files. Called after the commit marker is durable.
    fn finalize_committed_drops(&self, names: &[String]);

    /// Restore (child, schema) pairs in catalog and child VersionStores; inverse of
    /// `strip_fk_references` from `drop_table`.
    fn restore_child_fk_schemas(&self, schemas: &[(String, Schema)]) -> Result<()>;

    /// Drain `name` from `pending_drop_cleanups` so a rollback's inverse `create_table`
    /// isn't refused by the DROP-in-progress guard.
    fn release_pending_drop_cleanup(&self, name: &str);

    /// Rebuild secondary indexes on the re-inserted VersionStore from serialized metadata.
    fn restore_table_indexes(&self, table_name: &str, indexes: &[Vec<u8>]) -> Result<()>;

    /// Build the serialized `IndexMetadata` payload for staging on the txn's deferred
    /// commit-time WAL flush. Mirrors `MVCCEngine::record_create_index` on auto-commit.
    #[allow(clippy::too_many_arguments)]
    fn build_index_metadata(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
        hnsw_m: Option<u16>,
        hnsw_ef_construction: Option<u16>,
        hnsw_ef_search: Option<u16>,
        hnsw_distance_metric: Option<u8>,
    ) -> Result<Vec<u8>>;

    /// Release a pinned DDL marker LSN from `pending_marker_lsns` and publish the new
    /// safe-visible watermark. `lsn = 0` is a no-op; idempotent.
    fn release_pending_ddl_marker(&self, lsn: u64);

    /// List all tables
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Rename a table
    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Commit table changes
    fn commit_table(&self, txn_id: i64, table: &dyn WriteTable) -> Result<()>;

    /// Rollback table changes
    fn rollback_table(&self, txn_id: i64, table: &dyn WriteTable);

    /// Record a commit marker in the WAL. Returns marker LSN (0 when not written:
    /// recovery replay, in-memory engine, persistence disabled).
    fn record_commit(&self, txn_id: i64, commit_seq: i64) -> Result<u64>;

    /// Publish the marker LSN to `db.shm` header. Called AFTER `complete_commit` so
    /// readers see durable + in-process visible together. Also clears this txn from
    /// `active_txn_first_lsn` to keep the safe-visible watermark monotonic.
    fn publish_visible_commit_lsn(&self, txn_id: i64, lsn: u64);

    /// Record rollback in WAL
    fn record_rollback(&self, txn_id: i64) -> Result<()>;

    /// Get all tables with pending changes for a transaction
    fn get_tables_with_pending_changes(&self, txn_id: i64) -> Result<Vec<Box<dyn WriteTable>>>;

    /// Check if transaction has any pending DML changes (without allocating)
    fn has_pending_dml_changes(&self, txn_id: i64) -> bool;

    /// Commit all tables for a transaction at once (includes WAL recording).
    ///
    /// Returns `(any_committed, optional_error, tables_with_pending_tombstones)`:
    /// - `(false, None, [])`: no tables had changes, nothing to do
    /// - `(true, None, tables)`: all tables committed successfully
    /// - `(true, Some(e), tables)`: partial commit - some tables committed before error
    /// - `(false, Some(e), [])`: error before any table committed
    ///
    /// Callers MUST complete_commit if any_committed is true, even on error,
    /// to avoid orphaning already-committed rows.
    fn commit_all_tables(&self, txn_id: i64) -> (bool, Option<crate::core::Error>, Vec<String>);

    /// Latch the engine into catastrophic-failure state so seal/compaction/backup
    /// refuse until restart. Used when commit fails after parent VersionStores have
    /// been drained but no commit marker is durable.
    fn mark_engine_failed(&self);

    /// Move this txn's pending cold tombstones into committed state, stamped with
    /// `marker_lsn` as `visible_at_lsn` and `commit_seq` for snapshot-iso filtering.
    /// `commit_seq` is passed in (not re-read) so the partial-commit path can stamp
    /// after `complete_commit` has removed the txn from the registry.
    fn stamp_pending_tombstones(
        &self,
        txn_id: i64,
        commit_seq: u64,
        marker_lsn: u64,
        tables: &[String],
    );

    /// Rollback all tables for a transaction at once
    /// This cleans up the transaction's entries in txn_version_stores
    fn rollback_all_tables(&self, txn_id: i64);

    /// Defer table cleanup to background thread (avoids synchronous deallocation)
    /// Default implementation drops synchronously
    fn defer_table_cleanup(&self, _tables: Vec<Box<dyn WriteTable>>) {
        // Default: just drop synchronously (tables dropped when _tables goes out of scope)
    }

    /// Acquire the seal fence shared lock; held by in-flight commits to make the
    /// checkpoint micro-seal wait. None for in-memory engines.
    fn acquire_seal_fence(&self) -> Option<SealFenceGuard> {
        None
    }

    /// Acquire the transactional-DDL fence shared lock. None for non-engine impls.
    fn acquire_transactional_ddl_fence(&self) -> Option<TransactionalDdlFenceGuard> {
        None
    }
}

/// RAII shared read-lock on the seal fence. Checkpoint micro-seal takes the write
/// lock, blocking until all guards are dropped.
pub struct SealFenceGuard {
    _lock: Arc<parking_lot::RwLock<()>>,
    /// Raw pointer kept null; lock lifetime is managed via raw lock_shared/unlock_shared.
    _raw: *const (),
}

// SAFETY: holds an Arc and a balanced shared raw-lock; created/dropped on one thread.
unsafe impl Send for SealFenceGuard {}
unsafe impl Sync for SealFenceGuard {}

impl SealFenceGuard {
    pub fn new(lock: Arc<parking_lot::RwLock<()>>) -> Self {
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: lock_shared() balanced by unlock_shared() in Drop; Arc keeps lock alive.
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
        // SAFETY: balancing release of the lock_shared() in new().
        unsafe { self._lock.raw().unlock_shared() };
    }
}

/// RAII shared read-lock on the transactional-DDL fence. Checkpoint's
/// `rerecord_ddl_to_wal` takes the exclusive lock, blocking until all guards drop.
pub struct TransactionalDdlFenceGuard {
    _lock: Arc<parking_lot::RwLock<()>>,
    _raw: *const (),
}

// SAFETY: same reasoning as `SealFenceGuard`.
unsafe impl Send for TransactionalDdlFenceGuard {}
unsafe impl Sync for TransactionalDdlFenceGuard {}

impl TransactionalDdlFenceGuard {
    pub fn new(lock: Arc<parking_lot::RwLock<()>>) -> Self {
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: lock_shared() balanced by unlock_shared() in Drop.
        unsafe { lock.raw().lock_shared() };
        Self {
            _raw: std::ptr::null(),
            _lock: lock,
        }
    }
}

impl Drop for TransactionalDdlFenceGuard {
    fn drop(&mut self) {
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: balancing release of the lock_shared() in new().
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
            savepoints: FxHashMap::default(),
            ddl_log: Vec::new(),
            transactional_ddl_guard: None,
        }
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
        self.last_table_name = None;
        self.tables.clear();
        self.ddl_log.clear();
        // Released LAST so checkpoint observing the unlock sees fully converged catalog state.
        self.transactional_ddl_guard = None;
        self.registry.remove_transaction_isolation_level(self.id);
    }

    /// Walk `ddl_log` in reverse and apply each inverse. Continues across errors;
    /// returns the first one and latches the engine into the failure state so further
    /// durable writes refuse. Reverse-order undo handles CREATE/DROP interleavings
    /// against the same name correctly without coalescing.
    fn rollback_ddl(&self, ops: &dyn TransactionEngineOperations) -> Result<()> {
        let mut undo_err: Option<Error> = None;
        let mut latched = false;
        for op in self.ddl_log.iter().rev() {
            match op {
                DdlOp::Create(table_name, _schema) => {
                    // Inverse is in-memory drop only; original CREATE never wrote WAL.
                    match ops.drop_table(table_name) {
                        Ok(_snapshot) => {
                            // Release the pending-drop mark deposited by drop_table;
                            // otherwise it leaks for the process lifetime and blocks
                            // same-name CREATE forever on in-memory engines.
                            ops.release_pending_drop_cleanup(table_name);
                        }
                        Err(e) => {
                            eprintln!(
                                "Error: Failed to drop transaction-created table '{}' \
                                 during DDL rollback: {} - latching engine and \
                                 propagating; restart will reconverge via WAL \
                                 recovery.",
                                table_name, e
                            );
                            if undo_err.is_none() {
                                undo_err = Some(e);
                            }
                            if !latched {
                                ops.mark_engine_failed();
                                latched = true;
                            }
                        }
                    }
                }
                DdlOp::Drop(table_name, snapshot) => {
                    // Inverse is in-memory restore; the deferred-DDL DROP never wrote WAL.
                    // Release the DROP-in-progress mark before the inverse create_table.
                    ops.release_pending_drop_cleanup(table_name);
                    if let Err(e) = ops.create_table(table_name, snapshot.parent_schema.clone()) {
                        eprintln!(
                            "Error: Failed to recreate dropped table '{}' during \
                             DDL rollback: {} - latching engine and propagating.",
                            table_name, e
                        );
                        if undo_err.is_none() {
                            undo_err = Some(e);
                        }
                        if !latched {
                            ops.mark_engine_failed();
                            latched = true;
                        }
                        continue;
                    }
                    // Restore child FK constraints stripped by `strip_fk_references`.
                    if let Err(e) = ops.restore_child_fk_schemas(&snapshot.child_schemas) {
                        eprintln!(
                            "Error: Failed to restore child FK schemas while undoing \
                             drop of '{}': {} - latching engine and propagating.",
                            table_name, e
                        );
                        if undo_err.is_none() {
                            undo_err = Some(e);
                        }
                        if !latched {
                            ops.mark_engine_failed();
                            latched = true;
                        }
                    }
                    // Restore secondary indexes so live state matches recovery's view.
                    if let Err(e) = ops.restore_table_indexes(table_name, &snapshot.indexes) {
                        eprintln!(
                            "Error: Failed to restore secondary indexes while undoing \
                             drop of '{}': {} - latching engine and propagating.",
                            table_name, e
                        );
                        if undo_err.is_none() {
                            undo_err = Some(e);
                        }
                        if !latched {
                            ops.mark_engine_failed();
                            latched = true;
                        }
                    }
                }
                DdlOp::CreateIndex(_, _) => {
                    // No inverse: index lives on the parent VersionStore which the
                    // surrounding Create's rollback removes. Explicit tx.create_index
                    // on a pre-existing table would need its own inverse.
                }
            }
        }
        match undo_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Serialize `ddl_log` (in insertion order) into the deferred ops list for
    /// `flush_transactional_ddl`.
    fn build_deferred_ddl_ops(&self) -> Vec<DeferredDdlOp> {
        let mut out = Vec::with_capacity(self.ddl_log.len());
        for op in &self.ddl_log {
            match op {
                DdlOp::Create(name, schema) => {
                    let schema_data =
                        crate::storage::mvcc::engine::MVCCEngine::serialize_schema(schema);
                    out.push(DeferredDdlOp::Create {
                        name: name.clone(),
                        schema_data,
                    });
                }
                DdlOp::Drop(name, _snapshot) => {
                    out.push(DeferredDdlOp::Drop { name: name.clone() });
                }
                DdlOp::CreateIndex(table_name, metadata) => {
                    out.push(DeferredDdlOp::CreateIndex {
                        table_name: table_name.clone(),
                        metadata: metadata.clone(),
                    });
                }
            }
        }
        out
    }

    /// Collect DROP names from `ddl_log` for `finalize_committed_drops` after commit.
    fn collect_committed_drop_names(&self) -> Vec<String> {
        self.ddl_log
            .iter()
            .filter_map(|op| match op {
                DdlOp::Drop(name, _) => Some(name.clone()),
                DdlOp::Create(_, _) | DdlOp::CreateIndex(_, _) => None,
            })
            .collect()
    }

    /// Check if this is a read-only transaction
    fn is_read_only(&self) -> bool {
        if !self.ddl_log.is_empty() {
            return false;
        }
        if let Some(ops) = &self.engine_operations {
            if ops.has_pending_dml_changes(self.id) {
                return false;
            }
        }
        true
    }

    /// Creates a savepoint, overwriting any existing one with the same name.
    pub fn create_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        let timestamp = get_fast_timestamp();
        let ddl_state = SavepointDdlState {
            ddl_log_len: self.ddl_log.len(),
        };
        self.savepoints.insert(
            name.to_string(),
            SavepointState {
                timestamp,
                ddl_state,
            },
        );
        Ok(())
    }

    /// Releases a savepoint without rolling back. Errors if the savepoint doesn't exist.
    pub fn release_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        if self.savepoints.remove(name).is_none() {
            return Err(Error::invalid_argument(format!(
                "savepoint '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    /// Rolls back to a savepoint, discarding all DML and DDL changes made after it.
    /// The savepoint itself is removed (SQL standard).
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;

        let sp_state = self.savepoints.get(name).copied().ok_or_else(|| {
            Error::invalid_argument(format!("savepoint '{}' does not exist", name))
        })?;

        // Rollback DML changes via engine operations (not self.tables which is empty)
        if let Some(ops) = &self.engine_operations {
            if let Ok(tables) = ops.get_tables_with_pending_changes(self.id) {
                for table in &tables {
                    table.rollback_to_timestamp(sp_state.timestamp);
                }
            }
        }

        // Walk post-savepoint ddl_log suffix in reverse and apply inverses; mirrors
        // `rollback_ddl`. Truncate only after success so a partial failure leaves the
        // suffix intact for retry / hard rollback / Drop sweep.
        if let Some(ops) = &self.engine_operations {
            let after_save_lo = sp_state.ddl_state.ddl_log_len;
            let mut undo_err: Option<Error> = None;
            let mut latched = false;
            for op in self.ddl_log[after_save_lo..].iter().rev() {
                match op {
                    DdlOp::Create(table_name, _schema) => match ops.drop_table(table_name) {
                        Ok(_snapshot) => {
                            ops.release_pending_drop_cleanup(table_name);
                        }
                        Err(e) => {
                            eprintln!(
                                "Error: Failed to drop transaction-created table '{}' \
                                     during savepoint rollback: {} - latching engine and \
                                     propagating.",
                                table_name, e
                            );
                            if undo_err.is_none() {
                                undo_err = Some(e);
                            }
                            if !latched {
                                ops.mark_engine_failed();
                                latched = true;
                            }
                        }
                    },
                    DdlOp::Drop(table_name, snapshot) => {
                        ops.release_pending_drop_cleanup(table_name);
                        if let Err(e) = ops.create_table(table_name, snapshot.parent_schema.clone())
                        {
                            eprintln!(
                                "Error: Failed to recreate dropped table '{}' during \
                                 savepoint rollback: {} - latching engine and \
                                 propagating.",
                                table_name, e
                            );
                            if undo_err.is_none() {
                                undo_err = Some(e);
                            }
                            if !latched {
                                ops.mark_engine_failed();
                                latched = true;
                            }
                            continue;
                        }
                        if let Err(e) = ops.restore_child_fk_schemas(&snapshot.child_schemas) {
                            eprintln!(
                                "Error: Failed to restore child FK schemas while undoing \
                                 drop of '{}' during savepoint rollback: {} - latching \
                                 engine.",
                                table_name, e
                            );
                            if undo_err.is_none() {
                                undo_err = Some(e);
                            }
                            if !latched {
                                ops.mark_engine_failed();
                                latched = true;
                            }
                        }
                        if let Err(e) = ops.restore_table_indexes(table_name, &snapshot.indexes) {
                            eprintln!(
                                "Error: Failed to restore secondary indexes while undoing \
                                 drop of '{}' during savepoint rollback: {} - latching \
                                 engine.",
                                table_name, e
                            );
                            if undo_err.is_none() {
                                undo_err = Some(e);
                            }
                            if !latched {
                                ops.mark_engine_failed();
                                latched = true;
                            }
                        }
                    }
                    DdlOp::CreateIndex(_, _) => {}
                }
            }
            if let Some(e) = undo_err {
                // Leave suffix intact so a later sweep can re-apply inverses; still
                // scrub savepoints for internal consistency.
                self.savepoints
                    .retain(|_, sp| sp.timestamp <= sp_state.timestamp);
                return Err(e);
            }
            self.ddl_log.truncate(after_save_lo);
        }

        // Remove this savepoint and all savepoints created after it
        self.savepoints
            .retain(|_, sp| sp.timestamp <= sp_state.timestamp);

        Ok(())
    }

    /// Check if a savepoint exists
    pub fn has_savepoint(&self, name: &str) -> bool {
        self.savepoints.contains_key(name)
    }

    /// Gets the timestamp associated with a savepoint
    pub fn get_savepoint_ts(&self, name: &str) -> Option<i64> {
        self.savepoints.get(name).map(|sp| sp.timestamp)
    }
}

impl ReadTransaction for MvccTransaction {
    fn id(&self) -> i64 {
        self.id
    }

    fn begin(&mut self) -> Result<()> {
        // No-op for compatibility; transaction is initialized in new().
        self.check_active()
    }

    fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        // Update state to committing
        self.state = TransactionState::Committing;

        // has_pending_dml_changes() avoids allocating Vec<Box<dyn WriteTable>>.
        let has_dml_changes = self
            .engine_operations
            .as_ref()
            .is_some_and(|ops| ops.has_pending_dml_changes(self.id));

        let is_read_only = self.ddl_log.is_empty() && !has_dml_changes;

        if !is_read_only {
            // Held until complete_commit so checkpoint micro-seal waits for in-flight commits.
            let _seal_guard = self
                .engine_operations
                .as_ref()
                .and_then(|ops| ops.acquire_seal_fence());

            // Phase 1: allocate commit_seq embedded in the WAL marker for snapshot-iso filtering.
            let commit_seq = self.registry.start_commit(self.id);

            // Phase 1.5: flush deferred DDL WAL BEFORE any DML so recovery applies
            // CREATE TABLE before dependent INSERTs (LSN-strict ordering). Orphans
            // without a commit marker are skipped on recovery.
            let deferred_ddl_ops = self.build_deferred_ddl_ops();
            if !deferred_ddl_ops.is_empty() {
                if let Some(ops) = &self.engine_operations {
                    if let Err(e) = ops.flush_transactional_ddl(self.id, &deferred_ddl_ops) {
                        // No DML drained yet - safe abort.
                        self.registry.abort_transaction(self.id);
                        ops.rollback_all_tables(self.id);
                        let _ = ops.record_rollback(self.id);
                        let _ = self.rollback_ddl(ops.as_ref());
                        self.state = TransactionState::RolledBack;
                        self.cleanup();
                        return Err(e);
                    }
                }
            }

            // Phase 2: commit all tables (per-table WAL recording is internal).
            let mut pending_tombstone_tables = Vec::new();
            if let Some(ops) = &self.engine_operations {
                let (any_committed, error, tables_with_pending_tombstones) =
                    ops.commit_all_tables(self.id);
                pending_tombstone_tables = tables_with_pending_tombstones;
                if let Some(e) = error {
                    if any_committed {
                        // Partial commit ordering (critical):
                        //   1. record_commit -> marker_lsn
                        //   2. stamp_pending_tombstones (move pending -> committed,
                        //      keyed by commit_seq, visible_at_lsn = marker_lsn)
                        //   3. complete_commit (in-process visibility)
                        //   4. publish_visible_commit_lsn (cross-process visibility)
                        // Stamp BEFORE complete_commit so readers never see commit_seq
                        // published with tombstones still pending.
                        match ops.record_commit(self.id, commit_seq) {
                            Ok(lsn) => {
                                if !pending_tombstone_tables.is_empty() {
                                    ops.stamp_pending_tombstones(
                                        self.id,
                                        commit_seq as u64,
                                        lsn,
                                        &pending_tombstone_tables,
                                    );
                                }
                                self.registry.complete_commit(self.id);
                                ops.publish_visible_commit_lsn(self.id, lsn);
                                let drops_partial = self.collect_committed_drop_names();
                                if !drops_partial.is_empty() {
                                    ops.finalize_committed_drops(&drops_partial);
                                }
                                self.state = TransactionState::Committed;
                            }
                            Err(_) => {
                                // record_commit failed but parent VersionStores already
                                // hold drained data. No real undo (rollback_all_tables
                                // only clears pending tombstones; abort_transaction
                                // becomes a "ghost commit" once the GC removes the
                                // abort marker). Complete in-memory coherently and
                                // latch into the failure state.
                                //
                                // visible_at_lsn = u64::MAX makes these tombstones
                                // invisible to every capped cross-process reader (the
                                // 0 sentinel would mean "always visible" and could
                                // hide cold rows for a never-published commit).
                                if !pending_tombstone_tables.is_empty() {
                                    ops.stamp_pending_tombstones(
                                        self.id,
                                        commit_seq as u64,
                                        u64::MAX,
                                        &pending_tombstone_tables,
                                    );
                                }
                                // Latch BEFORE complete_commit so safe_snapshot_cutoff
                                // can't unblock and let backup/seal export this
                                // markerless commit's rows.
                                ops.mark_engine_failed();
                                self.registry.complete_commit(self.id);
                                let _ = ops.record_rollback(self.id);
                                self.state = TransactionState::Committed;
                            }
                        }
                        self.cleanup();
                        return Err(e);
                    } else {
                        // Nothing committed - safe to abort cleanly.
                        self.registry.abort_transaction(self.id);
                        // Revert in-memory DDL: failed commit otherwise leaves a Drop
                        // applied with no durable DROP, so restart would resurrect.
                        let _ = self.rollback_ddl(ops.as_ref());
                        ops.rollback_all_tables(self.id);
                        // Record a rollback marker so future shm publishes don't see
                        // a phantom low watermark from this txn's DML LSNs.
                        let _ = ops.record_rollback(self.id);
                        self.state = TransactionState::RolledBack;
                        self.cleanup();
                        return Err(e);
                    }
                }
            }

            // Phase 3: write commit marker BEFORE visibility so crash recovery sees
            // it. Capture marker LSN to publish to db.shm AFTER complete_commit.
            let commit_marker_lsn = if let Some(ops) = &self.engine_operations {
                match ops.record_commit(self.id, commit_seq) {
                    Ok(lsn) => {
                        // Stamp tombstones with marker_lsn BEFORE publish: closes the
                        // SWMR race where a reader sampling another concurrent commit's
                        // visible_commit_lsn could observe our tombstone before our
                        // marker is visible at that cap.
                        if !pending_tombstone_tables.is_empty() {
                            ops.stamp_pending_tombstones(
                                self.id,
                                commit_seq as u64,
                                lsn,
                                &pending_tombstone_tables,
                            );
                        }
                        lsn
                    }
                    Err(e) => {
                        // Same situation as the partial-commit Err branch: parent
                        // VersionStores hold drained data with no commit marker. No
                        // real undo. Stamp tombstones with u64::MAX (excluded by every
                        // capped reader), publish in-process visibility, write rollback
                        // marker so recovery discards, latch the engine.
                        if !pending_tombstone_tables.is_empty() {
                            ops.stamp_pending_tombstones(
                                self.id,
                                commit_seq as u64,
                                u64::MAX,
                                &pending_tombstone_tables,
                            );
                        }
                        // Latch BEFORE complete_commit (same ordering rule as partial-commit).
                        ops.mark_engine_failed();
                        self.registry.complete_commit(self.id);
                        let _ = ops.record_rollback(self.id);
                        self.state = TransactionState::Committed;
                        self.cleanup();
                        return Err(e);
                    }
                }
            } else {
                0
            };

            // Phase 4: in-process visibility.
            self.registry.complete_commit(self.id);

            // Phase 5: publish marker LSN to db.shm for cross-process readers. The
            // user marker LSN sits above every Phase 1.5 DDL entry LSN, so this one
            // publish advances safe_visible past all of them.
            if let Some(ops) = &self.engine_operations {
                ops.publish_visible_commit_lsn(self.id, commit_marker_lsn);
            }

            // Phase 6: physical DROP cleanup (segment state + on-disk volume files).
            // A crash between Phase 5 and here leaves orphan files for the next
            // checkpoint / compaction to reclaim - recoverable.
            let drops_to_finalize = self.collect_committed_drop_names();
            if !drops_to_finalize.is_empty() {
                if let Some(ops) = &self.engine_operations {
                    ops.finalize_committed_drops(&drops_to_finalize);
                }
            }
        } else {
            // Read-only - just mark committed in registry.
            self.registry.complete_commit(self.id);
        }

        // Mark as committed
        self.state = TransactionState::Committed;
        self.cleanup();

        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.check_active()?;

        let is_read_only = self.is_read_only();
        self.registry.abort_transaction(self.id);

        // Capture (don't propagate yet) so the rest of the rollback bookkeeping still
        // runs even when DDL compensation failed; in-memory state must be drained
        // either way. Err means the engine has been latched.
        let mut compensation_err: Option<Error> = None;
        if let Some(ops) = &self.engine_operations {
            if let Err(e) = self.rollback_ddl(ops.as_ref()) {
                compensation_err = Some(e);
            }
        }

        for (_, table) in self.tables.iter_mut() {
            table.rollback();
        }

        if let Some(ops) = &self.engine_operations {
            for (_, table) in self.tables.iter() {
                ops.rollback_table(self.id, table.as_ref());
            }
            // Clean up txn_version_stores entry to prevent memory leak.
            ops.rollback_all_tables(self.id);
        }

        if !is_read_only {
            if let Some(ops) = &self.engine_operations {
                let _ = ops.record_rollback(self.id);
            }
        }

        self.state = TransactionState::RolledBack;
        self.cleanup();
        match compensation_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn create_savepoint(&mut self, name: &str) -> Result<()> {
        MvccTransaction::create_savepoint(self, name)
    }

    fn release_savepoint(&mut self, name: &str) -> Result<()> {
        MvccTransaction::release_savepoint(self, name)
    }

    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        MvccTransaction::rollback_to_savepoint(self, name)
    }

    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64> {
        MvccTransaction::get_savepoint_ts(self, name)
    }

    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        self.check_active()?;
        self.isolation_level = Some(level);
        self.registry
            .set_transaction_isolation_level(self.id, level);
        Ok(())
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        ops.list_tables()
    }

    fn get_read_table(&self, name: &str) -> Result<Box<dyn ReadTable>> {
        let write: Box<dyn WriteTable> = self.get_table(name)?;
        Ok(write)
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

impl WriteTransaction for MvccTransaction {
    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn WriteTable>> {
        self.check_active()?;

        // Acquire transactional-DDL fence (idempotent) BEFORE mutating in-memory
        // catalog so checkpoint can't snapshot a half-mutated state.
        if self.transactional_ddl_guard.is_none() {
            let ops = self.get_engine_ops()?;
            self.transactional_ddl_guard = ops.acquire_transactional_ddl_fence();
        }
        let ops = self.get_engine_ops()?;
        let schema_for_log = schema.clone();
        let table = ops.create_table(name, schema)?;

        // Schema captured here for the commit-time deferred CreateTable WAL entry
        // (create_table only stages in-memory state).
        self.ddl_log
            .push(DdlOp::Create(name.to_lowercase(), schema_for_log));

        Ok(table)
    }

    /// Drop a table within this transaction.
    ///
    /// # Warning
    /// DROP TABLE is NOT fully transactional: structure can be recreated on rollback
    /// but data CANNOT (matches PostgreSQL behavior). Use DELETE or TRUNCATE for
    /// rollback-safe data removal.
    fn drop_table(&mut self, name: &str) -> Result<()> {
        self.check_active()?;

        if self.transactional_ddl_guard.is_none() {
            let ops = self.get_engine_ops()?;
            self.transactional_ddl_guard = ops.acquire_transactional_ddl_fence();
        }
        // Drop in-memory FIRST then log: logging a drop that didn't happen would let
        // rollback try to recreate a still-existing table.
        let ops = self.get_engine_ops()?;
        let snapshot = ops.drop_table(name)?;

        // Snapshot recorded for rollback (durable DROP WAL is deferred to commit;
        // physical volume deletion to `finalize_committed_drops`).
        self.ddl_log
            .push(DdlOp::Drop(name.to_lowercase(), snapshot));

        self.tables.remove(name);

        if let Some(last_name) = &self.last_table_name {
            if last_name == name {
                self.last_table_name = None;
            }
        }

        Ok(())
    }

    fn get_table(&self, name: &str) -> Result<Box<dyn WriteTable>> {
        self.check_active()?;

        // Caching here would require Clone on Table trait (not object-safe). Engine
        // handles caching internally. self.tables tracks accessed tables for commit/rollback.
        let ops = self.get_engine_ops()?;
        ops.get_table_for_transaction(self.id, name)
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

        let mut table = self.get_table(table_name)?;
        table.modify_column(&column.name, column.data_type, column.nullable)
    }

    fn stage_deferred_create_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
        hnsw_m: Option<u16>,
        hnsw_ef_construction: Option<u16>,
        hnsw_ef_search: Option<u16>,
        hnsw_distance_metric: Option<u8>,
    ) -> Result<()> {
        self.check_active()?;
        // Index staging is part of the same DDL window as the parent CREATE TABLE.
        if self.transactional_ddl_guard.is_none() {
            let ops = self.get_engine_ops()?;
            self.transactional_ddl_guard = ops.acquire_transactional_ddl_fence();
        }
        let ops = self.get_engine_ops()?;
        let metadata = ops.build_index_metadata(
            table_name,
            index_name,
            columns,
            is_unique,
            index_type,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_distance_metric,
        )?;
        // Empty payload = column lookup failed; matches auto-commit no-op behaviour.
        if metadata.is_empty() {
            return Ok(());
        }
        self.ddl_log
            .push(DdlOp::CreateIndex(table_name.to_lowercase(), metadata));
        Ok(())
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
            self.registry.abort_transaction(self.id);

            if let Some(ops) = &self.engine_operations {
                // Drop can't propagate; `rollback_ddl` latches the engine on failure
                // so further durable writes refuse.
                let _ = self.rollback_ddl(ops.as_ref());

                // Critical for read-only txns that called get_table() and were
                // dropped without explicit commit/rollback.
                ops.rollback_all_tables(self.id);
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
