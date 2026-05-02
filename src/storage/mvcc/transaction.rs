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
/// Used to rollback CREATE/DROP TABLE operations when rolling back to a savepoint.
#[derive(Debug, Clone, Copy)]
struct SavepointDdlState {
    /// Number of `ddl_log` entries at savepoint time. Rolling
    /// back to a savepoint walks `ddl_log[ddl_log_len..]` in
    /// reverse and applies the inverse of each entry, so the
    /// in-memory state and the durable WAL converge to what
    /// the txn looked like at savepoint time.
    ddl_log_len: usize,
}

/// Snapshot of pre-drop in-memory state captured by
/// `EngineOperations::drop_table`. Returned to
/// `MvccTransaction::drop_table` so the txn can store enough
/// context in `ddl_log` to fully restore on rollback —
/// including any child-table FK constraints that were
/// stripped during the drop AND the dropped table's own
/// secondary / unique / FK indexes.
#[derive(Debug, Clone)]
pub struct DropSnapshot {
    /// The parent table's pre-drop schema. Recreated by
    /// `ops.create_table` on rollback.
    pub parent_schema: Schema,
    /// Each child table whose FK constraints referenced the
    /// dropped parent, paired with its pre-strip schema. On
    /// rollback the txn restores the child's catalog schema
    /// AND the child VersionStore's schema (since
    /// `strip_fk_references` mutated both).
    pub child_schemas: Vec<(String, Schema)>,
    /// Serialized `IndexMetadata` for every secondary
    /// (non-PK) index on the dropped table. Recreated on the
    /// freshly re-inserted VersionStore during rollback so
    /// the live writer's in-memory state matches what
    /// recovery rebuilds from the deferred CreateTable +
    /// CreateIndex WAL entries. Without this, a savepoint
    /// case like `CREATE TABLE foo UNIQUE; SAVEPOINT s;
    /// DROP TABLE foo; ROLLBACK TO s; COMMIT` would commit
    /// the deferred CreateIndex WAL while the live writer
    /// still has no index, diverging live vs. recovered
    /// state.
    pub indexes: Vec<Vec<u8>>,
}

/// A DDL operation queued for durable write at commit time.
/// Each entry is emitted as a single WAL entry under the
/// user's transaction id (no auto-commit marker), gated for
/// recovery / SWMR-tail visibility by the txn's commit marker.
#[derive(Debug, Clone)]
pub enum DeferredDdlOp {
    /// `CREATE TABLE name` with serialized schema bytes.
    Create { name: String, schema_data: Vec<u8> },
    /// `DROP TABLE name`. No payload — recovery only needs
    /// the name to remove the table from the live catalog.
    Drop { name: String },
    /// `CREATE INDEX` carrying the serialized
    /// `IndexMetadata` payload. Recovery rebuilds the index
    /// after restoring the parent table's `CreateTable`
    /// entry — flush ordering inside a single txn matches
    /// `ddl_log` insertion order, so the table CREATE always
    /// precedes its generated indexes.
    CreateIndex {
        table_name: String,
        metadata: Vec<u8>,
    },
}

/// One DDL operation recorded during a writable transaction,
/// kept in an ordered log so rollback can apply inverses in
/// reverse and naturally handle any sequence of CREATE/DROP
/// against the same table name. Order matters: a set-based
/// coalesce can't tell `CREATE t; DROP t` (no-op rollback)
/// apart from `DROP t; CREATE t` (drop replacement, restore
/// original).
#[derive(Debug, Clone)]
enum DdlOp {
    /// `CREATE TABLE name` with the schema as captured at
    /// CREATE time. The schema is serialized into
    /// `DeferredDdlOp::Create` at commit time so recovery
    /// rebuilds the same table after restart. Inverse on
    /// rollback: drop the table in memory only (no durable
    /// record was written).
    Create(String, Schema),
    /// `DROP TABLE name` with a snapshot of the pre-drop
    /// catalog state. Inverse on rollback: recreate the
    /// parent (in memory only) AND restore every stripped
    /// child-table FK schema in BOTH the catalog and the
    /// child VersionStore. No durable compensation is needed
    /// — the durable DropTable record is only written from
    /// the commit path (`flush_transactional_ddl`), so a
    /// rollback simply never emits the drop and recovery /
    /// cross-process SWMR readers converge on "table still
    /// present."
    Drop(String, DropSnapshot),
    /// `CREATE INDEX` deferred for transactional WAL. The
    /// pre-serialized `IndexMetadata` payload is stored
    /// inline so the commit phase doesn't need to re-derive
    /// it from the live (potentially mutated) schema.
    /// Inverse on rollback: no-op — these only carry deferred
    /// WAL bytes; the in-memory index lives on its parent
    /// table's VersionStore which is removed by the parent's
    /// own CREATE / DROP rollback path.
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
    /// Ordered DDL log for rollback. Each entry is one
    /// CREATE or DROP applied in this transaction. Rollback
    /// walks this log in reverse and applies the inverse of
    /// each entry; that handles any interleaving of CREATE /
    /// DROP against the same table name correctly (which a
    /// set-based coalesce cannot — see `DdlOp` doc).
    ddl_log: Vec<DdlOp>,
    /// SHARED hold on the engine's transactional-DDL fence,
    /// acquired lazily on the first CREATE / DROP this txn
    /// performs and released when the txn resolves
    /// (commit / rollback / drop). Held for the entire
    /// duration that this txn has uncommitted DDL mutations
    /// in the engine's in-memory `schemas` /
    /// `version_stores`, so checkpoint's `rerecord_ddl_to_wal`
    /// (which takes the EXCLUSIVE lock) cannot snapshot the
    /// partially-mutated catalog and durably republish an
    /// uncommitted CREATE / omit a DROP that later rolls
    /// back. `None` until the first DDL touches the engine.
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

    /// Drop a table within a transaction.
    ///
    /// Performs the in-memory mutation (removes from schemas
    /// and version stores, strips FK references on child
    /// tables) BUT does NOT write the DropTable WAL record AND
    /// does NOT delete on-disk volume files. The durable
    /// record is emitted from the txn's commit phase via
    /// `flush_transactional_ddl` (under the user txn id, gated
    /// by the txn's commit marker), and physical file deletion
    /// is deferred to `finalize_committed_drops` after that
    /// marker is durable. Rollback restores in-memory state
    /// via `rollback_ddl` (using the snapshot returned here)
    /// and writes nothing durable.
    ///
    /// Returns a snapshot of the pre-drop state needed to undo
    /// the in-memory mutation: the parent schema and every
    /// child schema whose FK constraints reference the parent
    /// (those constraints are stripped during the drop and
    /// must be restored on rollback).
    fn drop_table(&self, name: &str) -> Result<DropSnapshot>;

    /// Emit a durable DDL WAL entry for each op in `ops` under
    /// `txn_id` (with NO auto-commit marker). Recovery /
    /// SWMR-tail visibility is gated by the user's commit
    /// marker (`record_commit`); a crash before the marker
    /// orphans these entries in WAL and recovery skips them.
    ///
    /// Called from `MvccTransaction::commit` AFTER
    /// `commit_all_tables` has drained DML to parent
    /// VersionStores and BEFORE `record_commit` writes the
    /// txn's commit marker. On Err the caller writes a
    /// rollback marker so recovery discards the orphaned
    /// partial DDL writes.
    fn flush_transactional_ddl(&self, txn_id: i64, ops: &[DeferredDdlOp]) -> Result<()>;

    /// Apply the post-commit physical side effects of every
    /// transactional DROP listed in `names` — clear segment
    /// manager state and delete on-disk volume files. Called
    /// from `MvccTransaction::commit` AFTER the user's commit
    /// marker is durable + visible to readers, so a crash
    /// between the marker write and these deletions leaves
    /// orphan files that the next checkpoint / compaction can
    /// reclaim (rather than the prior failure mode of a live
    /// catalog entry pointing at vanished files).
    fn finalize_committed_drops(&self, names: &[String]);

    /// Restore the supplied (child_table_name, schema) pairs
    /// in BOTH the catalog and each child VersionStore — the
    /// inverse of `strip_fk_references` performed inside
    /// `drop_table`. Called from `rollback_ddl` /
    /// `rollback_to_savepoint` so an undone DROP fully
    /// restores the child-table FK constraints that were
    /// stripped when the parent went away.
    fn restore_child_fk_schemas(&self, schemas: &[(String, Schema)]) -> Result<()>;

    /// Drain `name` from the engine's
    /// `pending_drop_cleanups` set so a rollback that's
    /// about to recreate the table via `create_table` isn't
    /// refused by the same-name DROP-in-progress guard.
    /// Called from `rollback_ddl` /
    /// `rollback_to_savepoint` BEFORE the inverse
    /// `create_table` for a `DdlOp::Drop` entry. No-op when
    /// the name isn't pending.
    fn release_pending_drop_cleanup(&self, name: &str);

    /// Recreate the supplied serialized `IndexMetadata`
    /// payloads on the freshly re-inserted VersionStore for
    /// `table_name`. Called from `rollback_ddl` /
    /// `rollback_to_savepoint` AFTER `ops.create_table` has
    /// recreated the empty parent — bringing the in-memory
    /// secondary / unique / FK indexes back so live writer
    /// state matches what recovery reconstructs from the
    /// deferred CreateTable + CreateIndex WAL entries.
    fn restore_table_indexes(&self, table_name: &str, indexes: &[Vec<u8>]) -> Result<()>;

    /// Build the serialized `IndexMetadata` payload for the
    /// named index on `table_name`. The transactional CREATE
    /// TABLE path uses this to capture the index payload at
    /// the SAME moment the in-memory `table.create_index`
    /// runs — and stages it on the txn for the deferred
    /// commit-time WAL flush. Mirrors the column-id /
    /// data-type derivation that `MVCCEngine::record_create_index`
    /// does on the auto-commit path.
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

    // `acquire_transactional_ddl_fence` is provided as a
    // default below (`None` for non-engine impls). Real
    // engines override it to return `Some(guard)` so
    // transactional DDL blocks checkpoint's
    // `rerecord_ddl_to_wal`.

    /// Release a previously-pinned DDL marker LSN from
    /// `pending_marker_lsns` and publish the new safe-visible
    /// watermark. `lsn = 0` is a no-op (no marker was pinned).
    /// Idempotent for an LSN already released. Retained for
    /// future pinning needs — the deferred-DDL drop path no
    /// longer pins anything.
    fn release_pending_ddl_marker(&self, lsn: u64);

    /// List all tables
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Rename a table
    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Commit table changes
    fn commit_table(&self, txn_id: i64, table: &dyn WriteTable) -> Result<()>;

    /// Rollback table changes
    fn rollback_table(&self, txn_id: i64, table: &dyn WriteTable);

    /// Record a commit marker in the WAL. Returns the LSN of the
    /// marker entry, which the caller publishes to `db.shm` via
    /// [`Self::publish_visible_commit_lsn`] after `complete_commit`.
    /// Returns `0` for cases where the marker was not written
    /// (recovery replay, in-memory engine, persistence disabled).
    fn record_commit(&self, txn_id: i64, commit_seq: i64) -> Result<u64>;

    /// Publish the WAL LSN of the most recent commit marker to the
    /// cross-process `db.shm` header (`visible_commit_lsn`). Called
    /// AFTER `complete_commit` so any reader that observes the new
    /// LSN finds the txn both durable on disk AND visible in the
    /// in-process registry. No-op when the engine has no shm
    /// attached (in-memory, read-only, or non-Unix).
    /// `txn_id` lets the engine clear this txn's entry from the
    /// WAL manager's `active_txn_first_lsn` map AFTER the
    /// safe-visible publish has fired. Without it,
    /// concurrent publishes could see the txn's first DML LSN
    /// disappear from `oldest_active_txn_lsn` before the txn's
    /// own publish exposed it, letting readers advance
    /// `next_entry_floor` past those DML records.
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

    /// Latch the engine into the catastrophic-failure state. Called
    /// from the partial-commit + record_commit-Err path: at that
    /// point parent VersionStores already hold the txn's data
    /// (drained by `commit_all_tables`) but the WAL has no commit
    /// marker. There is no real undo, so all subsequent durability
    /// paths (seal, compaction, backup) refuse to run until restart;
    /// recovery then converges by discarding the markerless txn.
    fn mark_engine_failed(&self);

    /// Stamp this txn's pending cold tombstones with `marker_lsn`
    /// as their `visible_at_lsn` and `commit_seq` as their
    /// snapshot-isolation commit sequence, moving them from
    /// pending to committed in each segment manager. Called AFTER
    /// `record_commit` returns the marker LSN so the visibility
    /// frontier matches what reader processes will see published
    /// in `db.shm`. No-op for tables with no pending tombstones
    /// for `txn_id`. Idempotent across tables (each table's call
    /// is keyed by `txn_id`).
    ///
    /// `commit_seq` is passed in (not re-read from the registry):
    /// the partial-commit path calls this AFTER `complete_commit`,
    /// which removes the txn from the registry — re-reading would
    /// return 0, and a `commit_seq = 0` tombstone is visible to
    /// every snapshot, exposing tombstones meant for the just-
    /// committed txn to all prior snapshots.
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

    /// Acquire the seal fence shared lock. Commits hold this to signal they
    /// are in-flight. The checkpoint micro-seal acquires the exclusive lock,
    /// waiting for all in-flight commits to complete before draining hot rows.
    /// Returns None for in-memory engines (no persistence, no seal fence needed).
    fn acquire_seal_fence(&self) -> Option<SealFenceGuard> {
        None
    }

    /// Default no-op so non-engine impls (tests, mocks) don't
    /// need to wire the fence. Real engines override.
    fn acquire_transactional_ddl_fence(&self) -> Option<TransactionalDdlFenceGuard> {
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

/// RAII guard that holds a SHARED read lock on the
/// transactional-DDL fence. Acquired by `MvccTransaction` on
/// the first CREATE / DROP in a txn, released when the txn
/// commits / rolls back. Checkpoint's `rerecord_ddl_to_wal`
/// takes the exclusive write lock, blocking until every
/// guard is dropped. Same raw-lock pattern as
/// `SealFenceGuard` so the borrow lives through arbitrary
/// txn-side state without lifetime gymnastics.
pub struct TransactionalDdlFenceGuard {
    _lock: Arc<parking_lot::RwLock<()>>,
    _raw: *const (),
}

// SAFETY: identical reasoning to `SealFenceGuard` — only an
// Arc plus a balanced shared raw-lock. Created and dropped on
// the same thread (the txn thread). The raw pointer is never
// dereferenced.
unsafe impl Send for TransactionalDdlFenceGuard {}
unsafe impl Sync for TransactionalDdlFenceGuard {}

impl TransactionalDdlFenceGuard {
    pub fn new(lock: Arc<parking_lot::RwLock<()>>) -> Self {
        use parking_lot::lock_api::RawRwLock;
        // SAFETY: lock_shared() is always safe; balanced by
        // unlock_shared() in Drop. The Arc keeps the RwLock
        // alive for the guard's lifetime.
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
        // Clear fast path cache
        self.last_table_name = None;

        // Clear tables
        self.tables.clear();

        // Clear DDL tracking
        self.ddl_log.clear();

        // Release the transactional-DDL fence (if held) so a
        // subsequent checkpoint that's been blocked waiting
        // for this txn can finally take the exclusive lock
        // and snapshot the now-converged catalog. Done LAST
        // among the catalog-related cleanup steps so any
        // concurrent checkpoint that observes the unlock has
        // already seen the in-memory state in its final form
        // (commit applied schemas, rollback restored them).
        self.transactional_ddl_guard = None;

        // Remove transaction isolation level from registry
        self.registry.remove_transaction_isolation_level(self.id);
    }

    /// Roll back DDL operations in reverse order.
    /// Used by both explicit rollback() and implicit Drop.
    ///
    /// Walks `ddl_log` in REVERSE and applies the inverse of
    /// each entry: undo a `Create` by calling `drop_table` (in
    /// memory + auto-publishing DropTable WAL marker), undo a
    /// `Drop` by recreating in-memory AND emitting a durable
    /// compensating CreateTable WAL record so restart and
    /// cross-process SWMR readers don't observe the original
    /// DropTable in isolation.
    ///
    /// Order matters because a set-based coalesce can't
    /// distinguish `CREATE t; DROP t` (both undo to a no-op
    /// for a table that didn't exist pre-txn) from
    /// `DROP t; CREATE t` (where pre-txn `t` existed: must
    /// drop the replacement AND restore the original schema).
    /// Reverse-order undo handles both cases correctly without
    /// any name-pair coalescing.
    ///
    /// Returns the FIRST error encountered. On any error the
    /// engine is also latched into the catastrophic-failure
    /// state so every subsequent durable path refuses (no
    /// further DDL/DML can land alongside the partially-undone
    /// log on disk). Callers that can propagate (`rollback()`)
    /// MUST surface this error so the caller doesn't see "Ok
    /// rollback" when durable state is actually inconsistent.
    /// Callers that can't propagate (`Drop::drop`) rely on the
    /// latch to block further durable writes.
    ///
    /// In-memory cleanup continues across errors — a partial
    /// failure on one undo entry shouldn't block undo of the
    /// next. The first captured error is returned.
    fn rollback_ddl(&self, ops: &dyn TransactionEngineOperations) -> Result<()> {
        let mut undo_err: Option<Error> = None;
        let mut latched = false;
        for op in self.ddl_log.iter().rev() {
            match op {
                DdlOp::Create(table_name, _schema) => {
                    // Inverse: drop the table in memory only.
                    // Deferred-DDL: the original CREATE never
                    // wrote WAL (it only staged in-memory
                    // schemas via `EngineOperations::create_table`),
                    // so undoing it doesn't need durable
                    // compensation — just remove the table
                    // from this process's catalog.
                    match ops.drop_table(table_name) {
                        Ok(_snapshot) => {
                            // The CREATE we're undoing never
                            // reached durability (no WAL
                            // CreateTable was ever written for
                            // it), so its inverse `drop_table`
                            // didn't make any committed catalog
                            // change either. The pending-drop
                            // mark `EngineOperations::drop_table`
                            // deposited would otherwise persist
                            // for the rest of the process — in
                            // memory-only / persistence-disabled
                            // engines there's no sweep that ever
                            // clears it, blocking same-name
                            // CREATE forever. Release here.
                            ops.release_pending_drop_cleanup(table_name);
                        }
                        Err(e) => {
                            eprintln!(
                                "Error: Failed to drop transaction-created table '{}' \
                                 during DDL rollback: {} — latching engine and \
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
                    // Inverse: restore the table in memory only.
                    // No durable compensation is needed — the
                    // transactional `drop_table` does not
                    // write WAL anymore; the durable record is
                    // emitted from the commit path. A rollback
                    // therefore never produces a durable drop
                    // for crash recovery / cross-process readers
                    // to converge on, so simply re-inserting
                    // the parent + child schemas brings this
                    // process back into alignment with disk.
                    //
                    // Clear the same-name DROP-in-progress
                    // mark `EngineOperations::drop_table`
                    // deposited so this rollback's inverse
                    // `create_table` isn't refused by the
                    // CREATE-while-pending guard.
                    ops.release_pending_drop_cleanup(table_name);
                    if let Err(e) = ops.create_table(table_name, snapshot.parent_schema.clone()) {
                        eprintln!(
                            "Error: Failed to recreate dropped table '{}' during \
                             DDL rollback: {} — latching engine and propagating.",
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
                    // Restore stripped FK constraints on every
                    // child table that referenced the parent.
                    // `strip_fk_references` mutated BOTH the
                    // catalog schema and each child
                    // VersionStore's schema, so the restore
                    // hits both via `restore_child_fk_schemas`.
                    if let Err(e) = ops.restore_child_fk_schemas(&snapshot.child_schemas) {
                        eprintln!(
                            "Error: Failed to restore child FK schemas while undoing \
                             drop of '{}': {} — latching engine and propagating.",
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
                    // Recreate the dropped table's secondary
                    // indexes on the freshly re-inserted
                    // VersionStore so live writer state
                    // matches what recovery rebuilds from the
                    // deferred CreateTable + CreateIndex WAL
                    // entries. Without this, a savepoint case
                    // like CREATE TABLE foo UNIQUE; SAVEPOINT
                    // s; DROP TABLE foo; ROLLBACK TO s;
                    // COMMIT would commit the deferred
                    // CreateIndex WAL while the live writer
                    // has no index, diverging the two views.
                    if let Err(e) = ops.restore_table_indexes(table_name, &snapshot.indexes) {
                        eprintln!(
                            "Error: Failed to restore secondary indexes while undoing \
                             drop of '{}': {} — latching engine and propagating.",
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
                    // No inverse needed at the txn level: the
                    // index was created on its parent table's
                    // VersionStore (in-memory). When the
                    // surrounding `Create` in this txn rolls
                    // back, `ops.drop_table` removes the
                    // VersionStore which carries the index
                    // away with it. When the index targets a
                    // pre-existing table, the index lives on
                    // that table's VersionStore and must be
                    // dropped explicitly — but the active-tx
                    // CREATE TABLE path is the only producer
                    // today, and its parent always rolls back
                    // alongside. Future explicit
                    // `tx.create_index` paths will need their
                    // own inverse.
                }
            }
        }
        match undo_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Build the deferred DDL ops list for the commit phase.
    /// Walks `ddl_log` in insertion order and serializes each
    /// entry into a `DeferredDdlOp` ready for
    /// `flush_transactional_ddl`. Both `Create` and `Drop`
    /// entries contribute — a successful BEGIN; CREATE TABLE;
    /// COMMIT must produce a durable CreateTable WAL record
    /// (otherwise recovery before the next checkpoint would
    /// lose the committed table), and DROP+CREATE replacement
    /// sequences need both ops in the WAL so recovery sees
    /// the replacement schema after the drop.
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

    /// Collect just the DROP names from the ddl_log — used by
    /// `MvccTransaction::commit` to drive
    /// `finalize_committed_drops` AFTER the commit marker is
    /// durable + visible.
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
        // Check for DDL changes
        if !self.ddl_log.is_empty() {
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
    /// If a savepoint with this name already exists, it is overwritten.
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

    /// Releases (removes) a savepoint without rolling back
    ///
    /// The changes made after the savepoint remain intact.
    /// Returns an error if the savepoint doesn't exist.
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

    /// Rolls back to a savepoint, discarding all changes made after it
    ///
    /// All local DML changes with timestamps after the savepoint are discarded.
    /// DDL operations (CREATE/DROP TABLE) after the savepoint are also reversed.
    /// The savepoint itself is also removed (SQL standard behavior).
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

        // Rollback DDL: walk `ddl_log[ddl_log_len..]` in
        // reverse and apply each inverse, mirroring
        // `rollback_ddl`. Same order-aware semantics — a
        // CREATE-then-DROP pair after the savepoint correctly
        // ends with no table; a DROP-then-CREATE pair against
        // a pre-savepoint table correctly drops the
        // replacement and restores the original. The
        // post-savepoint entries are then truncated from the
        // log so subsequent rollback / commit operates on the
        // pre-savepoint state.
        if let Some(ops) = &self.engine_operations {
            let after_save_lo = sp_state.ddl_state.ddl_log_len;
            // Iterate over a BORROWED view of the
            // post-savepoint suffix. Draining first would
            // remove the entries before the inverse runs, so
            // a partial-undo failure would leave the txn with
            // no record of what still needs to be undone — a
            // later full ROLLBACK couldn't drop a created
            // table or recreate a dropped one whose log entry
            // already vanished. Truncating ONLY after every
            // inverse succeeds keeps the log honest: on Err
            // the suffix stays intact so the caller can
            // retry, hard-rollback, or have `Drop::drop`
            // sweep it.
            let mut undo_err: Option<Error> = None;
            let mut latched = false;
            for op in self.ddl_log[after_save_lo..].iter().rev() {
                match op {
                    DdlOp::Create(table_name, _schema) => {
                        match ops.drop_table(table_name) {
                            Ok(_snapshot) => {
                                // Same release as
                                // `rollback_ddl`'s Create
                                // case: the CREATE we're
                                // undoing never reached
                                // durability, so the inverse
                                // `drop_table`'s pending mark
                                // would otherwise leak for
                                // the rest of the process.
                                ops.release_pending_drop_cleanup(table_name);
                            }
                            Err(e) => {
                                eprintln!(
                                    "Error: Failed to drop transaction-created table '{}' \
                                     during savepoint rollback: {} — latching engine and \
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
                        }
                    }
                    DdlOp::Drop(table_name, snapshot) => {
                        // Restore in-memory only — no durable
                        // compensation needed (deferred-DDL
                        // model: the original drop never wrote
                        // WAL, so rolling back simply skips
                        // the future flush).
                        //
                        // Same release as `rollback_ddl`'s
                        // Drop case — clear the DROP-in-
                        // progress mark before the inverse
                        // `create_table` so the same-name
                        // CREATE-while-pending guard doesn't
                        // refuse this rollback's restore.
                        ops.release_pending_drop_cleanup(table_name);
                        if let Err(e) = ops.create_table(table_name, snapshot.parent_schema.clone())
                        {
                            eprintln!(
                                "Error: Failed to recreate dropped table '{}' during \
                                 savepoint rollback: {} — latching engine and \
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
                        // Restore child FK schemas as the
                        // matching block in `rollback_ddl` does.
                        if let Err(e) = ops.restore_child_fk_schemas(&snapshot.child_schemas) {
                            eprintln!(
                                "Error: Failed to restore child FK schemas while undoing \
                                 drop of '{}' during savepoint rollback: {} — latching \
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
                        // Recreate the dropped table's secondary
                        // indexes — see matching block in
                        // `rollback_ddl` for the rationale.
                        if let Err(e) = ops.restore_table_indexes(table_name, &snapshot.indexes) {
                            eprintln!(
                                "Error: Failed to restore secondary indexes while undoing \
                                 drop of '{}' during savepoint rollback: {} — latching \
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
                    DdlOp::CreateIndex(_, _) => {
                        // No inverse needed — see matching
                        // block in `rollback_ddl` for the
                        // rationale (the index was created on
                        // a VersionStore that the parent
                        // CREATE/DROP rollback removes).
                    }
                }
            }
            if let Some(e) = undo_err {
                // Leave the post-savepoint `ddl_log` suffix
                // INTACT so subsequent rollback-attempt /
                // Drop sweep can re-apply the inverses (some
                // may have partially landed; the latched
                // engine refuses durable writes either way).
                // Still scrub savepoints so the txn's local
                // view is internally consistent before
                // returning Err.
                self.savepoints
                    .retain(|_, sp| sp.timestamp <= sp_state.timestamp);
                return Err(e);
            }
            // All inverses succeeded — truncate so commit /
            // subsequent rollback operates on the pre-savepoint
            // DDL view. No marker-release step is needed: the
            // deferred-DDL model never pinned anything during
            // the txn (writes are emitted from
            // `flush_transactional_ddl_drops` at commit time on
            // the FINAL `ddl_log` only).
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
        // No-op for compatibility - transaction is initialized in new()
        self.check_active()
    }

    fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        // Update state to committing
        self.state = TransactionState::Committing;

        // Check if read-only: no DDL changes and no DML changes
        // Use has_pending_dml_changes() to avoid allocating Vec<Box<dyn WriteTable>>
        let has_dml_changes = self
            .engine_operations
            .as_ref()
            .is_some_and(|ops| ops.has_pending_dml_changes(self.id));

        let is_read_only = self.ddl_log.is_empty() && !has_dml_changes;

        // Two-phase commit protocol
        if !is_read_only {
            // Acquire seal fence shared lock. This signals to the checkpoint
            // micro-seal that a commit is in-flight. The micro-seal waits for
            // all in-flight commits to finish before draining hot rows.
            // Held until complete_commit (end of this block).
            let _seal_guard = self
                .engine_operations
                .as_ref()
                .and_then(|ops| ops.acquire_seal_fence());

            // Phase 1: Start commit - mark transaction as "committing".
            // start_commit allocates the commit_seq we'll embed in the WAL
            // commit marker (SWMR v2: reader's WAL-tail uses it to tag
            // tombstones so snapshot_seq filtering matches the writer).
            let commit_seq = self.registry.start_commit(self.id);

            // Phase 1.5 (deferred-DDL): emit durable DDL WAL
            // entries BEFORE any DML so recovery applies
            // CREATE TABLE before the dependent INSERT/UPDATE/
            // DELETE entries that target it. WAL ordering is
            // LSN-strict, so writing DDL after DML would let
            // recovery hit a row entry for a table that does
            // not yet exist in the recovered catalog. Each
            // entry is written under the user txn id with NO
            // auto-commit marker — a crash before
            // `record_commit` (Phase 3) leaves these orphaned
            // and recovery skips them.
            let deferred_ddl_ops = self.build_deferred_ddl_ops();
            if !deferred_ddl_ops.is_empty() {
                if let Some(ops) = &self.engine_operations {
                    if let Err(e) = ops.flush_transactional_ddl(self.id, &deferred_ddl_ops) {
                        // No DML drained yet — safe abort
                        // path. The orphan DDL entries (if
                        // any landed before the failure) are
                        // skipped by recovery without a
                        // commit marker for self.id.
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

            // Phase 2: Commit all tables - apply local changes to global store
            // This now includes WAL recording internally (before each table commit)
            let mut pending_tombstone_tables = Vec::new();
            if let Some(ops) = &self.engine_operations {
                let (any_committed, error, tables_with_pending_tombstones) =
                    ops.commit_all_tables(self.id);
                pending_tombstone_tables = tables_with_pending_tombstones;
                if let Some(e) = error {
                    if any_committed {
                        // Partial commit: some tables already
                        // committed. Order matters:
                        //   1. record_commit (WAL marker → marker_lsn)
                        //   2. stamp_pending_tombstones (move cold
                        //      tombstones from pending into the
                        //      shared map, keyed by commit_seq, with
                        //      visible_at_lsn = marker_lsn)
                        //   3. complete_commit (publish commit_seq
                        //      → in-process visibility)
                        //   4. publish_visible_commit_lsn (publish
                        //      marker_lsn → cross-process visibility)
                        //
                        // The previous ordering (complete_commit →
                        // record_commit → stamp) opened a window
                        // where in-process readers could see the
                        // committed txn (commit_seq published) but
                        // its cold tombstones were still pending —
                        // the deleted row appeared not deleted. Now
                        // tombstones are in the shared map BEFORE
                        // commit_seq is visible to anyone.
                        // Deferred DDL was already flushed in
                        // Phase 1.5 (above). The orphan DDL
                        // entries are gated by `record_commit`
                        // below: if it succeeds, recovery
                        // applies CREATE TABLE before the
                        // partial DML; if it fails, no commit
                        // marker exists for `self.id` and
                        // recovery skips both the DDL and the
                        // partially-drained DML.
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
                                // Phase 6 equivalent for the
                                // partial-commit success path:
                                // physically reap dropped tables
                                // now that the marker is durable.
                                let drops_partial = self.collect_committed_drop_names();
                                if !drops_partial.is_empty() {
                                    ops.finalize_committed_drops(&drops_partial);
                                }
                                self.state = TransactionState::Committed;
                            }
                            Err(_) => {
                                // record_commit failed AFTER
                                // commit_all_tables already drained
                                // local versions into parent
                                // VersionStores and updated indexes.
                                // We CANNOT abort here:
                                //   - `rollback_all_tables` only
                                //     clears pending tombstones and
                                //     txn-local caches — it doesn't
                                //     undo parent VersionStore
                                //     writes or index updates. There
                                //     is no real undo path.
                                //   - Calling
                                //     `registry.abort_transaction`
                                //     installs an abort marker that
                                //     read-committed fast paths
                                //     consult to hide this txn's
                                //     versions. But registry GC
                                //     eventually removes the marker,
                                //     after which `check_committed`
                                //     treats the valid txn_id as
                                //     committed by default —
                                //     exposing the failed-marker
                                //     rows as a "ghost commit".
                                //
                                // Complete the in-memory commit
                                // coherently instead. Stamp pending
                                // tombstones BEFORE complete_commit
                                // (so in-process readers don't see
                                // commit_seq published with
                                // tombstones still pending). Use
                                // visible_at_lsn = u64::MAX so
                                // `retain_segments_visible_at_or_below(cap)`
                                // ALWAYS drops these tombstones for
                                // any realistic capped reader —
                                // visible_at_lsn = 0 would have been
                                // the "always visible" sentinel and
                                // a cross-process attach/refresh
                                // capped below this txn's notional
                                // marker would retain them, hiding
                                // cold rows for a commit that was
                                // never cross-process visible (no
                                // marker, no shm publish).
                                // u64::MAX keeps them invisible to
                                // every cross-process reader; in-
                                // process readers ignore
                                // visible_at_lsn and rely on the
                                // snapshot_seq filter, so they see
                                // the deletes as expected. Then
                                // complete_commit publishes
                                // commit_seq for in-process
                                // visibility, and record_rollback
                                // writes a WAL rollback marker
                                // (clears active_txn_first_lsn) so
                                // recovery consistently discards
                                // this txn from disk on the next
                                // process start. The accepted
                                // trade-off: live in-memory state
                                // shows committed; recovery would
                                // discard.
                                if !pending_tombstone_tables.is_empty() {
                                    ops.stamp_pending_tombstones(
                                        self.id,
                                        commit_seq as u64,
                                        u64::MAX,
                                        &pending_tombstone_tables,
                                    );
                                }
                                // Latch the engine into the
                                // catastrophic-failure state BEFORE
                                // `complete_commit` publishes the
                                // commit_seq. Order matters:
                                // complete_commit removes the txn
                                // from the committing set, which
                                // unblocks `safe_snapshot_cutoff` —
                                // a backup or seal already past its
                                // entry-time `is_failed()` check
                                // would then observe the cutoff
                                // including our markerless commit
                                // and proceed to export those parent-
                                // VersionStore rows. Setting the
                                // latch first means every later
                                // recheck (in create_backup_snapshot,
                                // seal_hot_buffers, compact_volumes,
                                // record_commit) sees the failed
                                // state at or before the moment
                                // commit_seq becomes visible to the
                                // cutoff.
                                ops.mark_engine_failed();
                                self.registry.complete_commit(self.id);
                                let _ = ops.record_rollback(self.id);
                                self.state = TransactionState::Committed;
                            }
                        }
                        self.cleanup();
                        return Err(e);
                    } else {
                        // Nothing committed yet - safe to abort cleanly.
                        // Release uncommitted_writes claims and remove from
                        // txn_version_stores to prevent permanent row blocking.
                        self.registry.abort_transaction(self.id);
                        // Roll back DDL state so any in-memory
                        // CREATE / DROP performed during this
                        // txn is reverted. Without this, the
                        // failed commit leaves a `Drop` removed
                        // from in-memory schemas / version
                        // stores even though no durable DROP
                        // was written — restart would resurrect
                        // the table while live state shows it
                        // gone.
                        let _ = self.rollback_ddl(ops.as_ref());
                        ops.rollback_all_tables(self.id);
                        // Phase 2 may have written DML entries before
                        // failing, installing this txn's
                        // active_txn_first_lsn in the WAL manager.
                        // `publish_visible_commit_lsn` is the only
                        // path that calls `clear_active_txn`, and we
                        // never reach it on this abort. Record a
                        // rollback marker (which `record_rollback`
                        // also clears the active record after) so
                        // future shm publishes don't see a phantom
                        // low watermark from this dead txn.
                        let _ = ops.record_rollback(self.id);
                        self.state = TransactionState::RolledBack;
                        self.cleanup();
                        return Err(e);
                    }
                }
            }

            // (Deferred DDL was already flushed in Phase 1.5
            // above, BEFORE commit_all_tables drained DML.
            // Recovery applies entries in LSN order, so DDL
            // before DML keeps `INSERT INTO foo` from racing
            // ahead of `CREATE TABLE foo` during replay.)

            // Phase 3: Record commit marker in WAL BEFORE making changes visible.
            // This ensures crash recovery sees the COMMIT marker even if we crash
            // before complete_commit(). WAL is only read during recovery, so writing
            // the marker before visibility doesn't affect normal operation.
            //
            // SWMR v2 Phase C: capture the marker LSN so we can publish it to
            // db.shm AFTER complete_commit, ensuring cross-process readers only
            // observe the LSN once the txn is also visible to in-process readers.
            let commit_marker_lsn = if let Some(ops) = &self.engine_operations {
                match ops.record_commit(self.id, commit_seq) {
                    Ok(lsn) => {
                        // Stamp pending cold tombstones with the
                        // marker LSN BEFORE complete_commit /
                        // publish. Closes the SWMR race where a
                        // reader sampling another concurrent
                        // commit's published visible_commit_lsn
                        // (between this txn's tombstone WAL entry
                        // and this txn's marker) would observe our
                        // tombstone via
                        // `retain_segments_visible_at_or_below(cap)`
                        // even though our marker isn't visible at
                        // that cap. Stamping with `lsn` ensures
                        // `visible_at_lsn = marker_lsn` so any cap
                        // below the marker excludes our tombstone.
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
                        // WAL commit marker failed AFTER
                        // commit_all_tables already drained local
                        // versions into parent VersionStores and
                        // updated indexes. Same situation as the
                        // partial-commit failure branch above —
                        // there is no real undo:
                        //   - `rollback_all_tables` only clears
                        //     pending tombstones / txn-local caches.
                        //   - Calling `abort_transaction` installs
                        //     an abort marker that fast paths
                        //     consult to hide this txn's versions,
                        //     but registry GC eventually removes the
                        //     marker — after which `check_committed`
                        //     defaults to "committed" and the
                        //     failed-marker rows resurface as a
                        //     "ghost commit".
                        //
                        // Complete the in-memory commit coherently
                        // and latch the engine into the
                        // catastrophic-failure state: stamp pending
                        // cold tombstones with `visible_at_lsn =
                        // u64::MAX` (ephemeral sentinel that
                        // serialize / compaction / backup all
                        // exclude), publish commit_seq via
                        // `complete_commit` for in-process
                        // visibility, write a WAL rollback marker
                        // (clears active_txn_first_lsn so recovery
                        // converges to "txn discarded"), then call
                        // `mark_engine_failed` so subsequent
                        // seal / compact / backup paths refuse to
                        // run. The user must restart; on next
                        // process start the WAL has no commit
                        // marker for this txn, recovery discards it,
                        // and durable state stays consistent.
                        if !pending_tombstone_tables.is_empty() {
                            ops.stamp_pending_tombstones(
                                self.id,
                                commit_seq as u64,
                                u64::MAX,
                                &pending_tombstone_tables,
                            );
                        }
                        // Latch BEFORE complete_commit publishes the
                        // commit_seq — same ordering rule as the
                        // partial-commit branch. Without this,
                        // safe_snapshot_cutoff unblocks at
                        // complete_commit and a backup / seal
                        // already past its entry-time is_failed()
                        // check would export this txn's markerless
                        // parent-store rows before the latch flips.
                        ops.mark_engine_failed();
                        // No DDL marker pins to release —
                        // deferred-DDL Phase 2.5 writes user-
                        // txn DDL entries with no auto-commit
                        // markers, so nothing is parked in
                        // `pending_marker_lsns`. The orphan
                        // DDL entries are safe in WAL: without
                        // a commit marker for self.id,
                        // recovery skips them.
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

            // Phase 4: Complete commit - make changes visible in registry
            self.registry.complete_commit(self.id);

            // Phase 5 (SWMR v2): publish the marker LSN to db.shm so reader
            // processes can advance their visible_commit_lsn watermark. Done
            // AFTER complete_commit so cross-process and in-process visibility
            // line up: any reader that observes the new LSN finds the txn
            // both durable on disk and live in the registry.
            //
            // The user marker LSN sits ABOVE every Phase 2.5
            // DDL entry LSN, so publishing the user marker
            // advances `safe_visible` past all of them in one
            // step. Cross-process readers tail this txn's
            // commit marker, see DDL + DML entries belonging
            // to a now-committed txn id, and apply them
            // together at the same visibility step.
            if let Some(ops) = &self.engine_operations {
                ops.publish_visible_commit_lsn(self.id, commit_marker_lsn);
            }

            // Phase 6 (deferred-DDL physical cleanup): now that
            // the commit marker is durable + visible, run the
            // physical post-commit side effects of every
            // transactional DROP — clear segment manager state
            // and delete on-disk volume files. A crash between
            // Phase 5 and here leaves orphan files / segment
            // state that the next checkpoint / compaction can
            // reclaim, which is recoverable; the inverse
            // (deleting before the marker is durable, as the
            // pre-deferred-DDL path did) was NOT recoverable.
            let drops_to_finalize = self.collect_committed_drop_names();
            if !drops_to_finalize.is_empty() {
                if let Some(ops) = &self.engine_operations {
                    ops.finalize_committed_drops(&drops_to_finalize);
                }
            }
        } else {
            // Read-only transaction - just mark as committed in registry
            self.registry.complete_commit(self.id);
        }

        // Mark as committed
        self.state = TransactionState::Committed;
        self.cleanup();

        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.check_active()?;

        // Check if read-only before rolling back
        let is_read_only = self.is_read_only();

        // Mark transaction as aborted in registry
        self.registry.abort_transaction(self.id);

        // Rollback DDL operations (CREATE TABLE / DROP TABLE) in reverse order.
        // Capture (don't propagate yet) so the rest of the
        // rollback bookkeeping (per-table rollback, txn_version_stores
        // cleanup, WAL rollback marker) still runs even when
        // compensation failed — the transaction's IN-MEMORY
        // state must be drained either way. The captured error
        // is surfaced AFTER cleanup so the caller sees a
        // truthful Result: a successful Ok means the durable
        // state is consistent with the in-memory rollback;
        // an Err means either the engine has been latched
        // (compensation gap was detected) or some other
        // failure happened.
        let mut compensation_err: Option<Error> = None;
        if let Some(ops) = &self.engine_operations {
            if let Err(e) = self.rollback_ddl(ops.as_ref()) {
                compensation_err = Some(e);
            }
            // No marker-release step: deferred-DDL means the
            // transactional `drop_table` did not pin any
            // marker LSN during the txn, and `rollback_ddl`
            // (which now only re-creates in-memory schemas)
            // doesn't write durable DDL either. So nothing to
            // release on rollback.
        }

        // Rollback all tables - discard local changes
        for (_, table) in self.tables.iter_mut() {
            table.rollback();
        }

        // Notify engine of rollback (per-table callbacks)
        if let Some(ops) = &self.engine_operations {
            for (_, table) in self.tables.iter() {
                ops.rollback_table(self.id, table.as_ref());
            }
            // Clean up txn_version_stores entry to prevent memory leak
            ops.rollback_all_tables(self.id);
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
        match compensation_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
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

        // Acquire the transactional-DDL fence (idempotent —
        // only the first DDL in the txn actually takes the
        // lock) BEFORE mutating the engine's in-memory
        // catalog. Held until commit / rollback so checkpoint
        // can't snapshot a half-mutated catalog mid-txn.
        if self.transactional_ddl_guard.is_none() {
            let ops = self.get_engine_ops()?;
            self.transactional_ddl_guard = ops.acquire_transactional_ddl_fence();
        }
        let ops = self.get_engine_ops()?;
        let schema_for_log = schema.clone();
        let table = ops.create_table(name, schema)?;

        // Append to the ordered DDL log for rollback. Order
        // matters: the log is replayed in reverse on rollback,
        // so the position of this Create relative to any later
        // Drop of the same name determines whether the Create
        // ends up undone (DROP-then-CREATE replacement scenario)
        // or coalesced-naturally to nothing (CREATE-then-DROP).
        // The schema is captured here so the commit phase can
        // serialize it into the deferred CreateTable WAL entry
        // — `EngineOperations::create_table` only stages the
        // in-memory state; durability comes from
        // `flush_transactional_ddl`.
        self.ddl_log
            .push(DdlOp::Create(name.to_lowercase(), schema_for_log));

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

        // Acquire the transactional-DDL fence (see
        // `create_table` for rationale) BEFORE mutating the
        // engine's in-memory catalog. Idempotent across
        // multiple DDLs in the same txn.
        if self.transactional_ddl_guard.is_none() {
            let ops = self.get_engine_ops()?;
            self.transactional_ddl_guard = ops.acquire_transactional_ddl_fence();
        }
        // Drop in-memory FIRST and capture the pre-drop
        // snapshot (parent + child FK schemas). Logging the
        // DROP into `ddl_log` BEFORE the trait call would
        // leave the log claiming a drop that didn't happen if
        // `ops.drop_table` returned Err on the catastrophic-
        // failure latch — a later rollback would then try to
        // recreate an existing table.
        let ops = self.get_engine_ops()?;
        let snapshot = ops.drop_table(name)?;

        // ops.drop_table succeeded (in-memory only — the
        // durable DropTable WAL record is deferred to the
        // commit phase, and physical volume deletion is
        // deferred to `finalize_committed_drops` after the
        // commit marker is durable). Record the snapshot in
        // the ordered DDL log so rollback can restore both
        // the parent schema and any stripped child FK
        // schemas. Position relative to any prior Create of
        // the same name controls rollback: an earlier Create
        // + this Drop coalesces naturally on reverse-walk;
        // a pre-txn table being replaced (no prior Create)
        // becomes a standalone Drop whose in-memory revert
        // restores the original table.
        self.ddl_log
            .push(DdlOp::Drop(name.to_lowercase(), snapshot));

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

    fn get_table(&self, name: &str) -> Result<Box<dyn WriteTable>> {
        self.check_active()?;

        // Note: Cached tables would require Clone on Table trait, which isn't object-safe.
        // For now, always get from engine (engine will handle caching internally).
        // The tables HashMap is used for tracking which tables were accessed for commit/rollback.

        // Get from engine
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
        // Hold the transactional-DDL fence across the
        // index-staging step too: this is part of the same
        // open-transaction DDL window as the parent CREATE
        // TABLE that triggered it. Idempotent.
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
        // Empty payload means the column lookup failed —
        // matches the auto-commit `record_create_index` no-op
        // behaviour for stale column names.
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
            // Silent rollback on drop
            self.registry.abort_transaction(self.id);

            if let Some(ops) = &self.engine_operations {
                // Roll back DDL operations (CREATE TABLE / DROP TABLE).
                // Drop can't propagate the Result, but
                // `rollback_ddl` already latches the engine on
                // compensation failure, so any subsequent
                // durable write refuses — the latch
                // protection is what keeps durable state from
                // diverging further. Deferred-DDL: no marker
                // release is needed because `drop_table` no
                // longer pins anything during the txn.

                let _ = self.rollback_ddl(ops.as_ref());

                // Clean up txn_version_stores to prevent memory leak
                // This is critical for read-only transactions that call get_table()
                // but are dropped without explicit commit/rollback
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
