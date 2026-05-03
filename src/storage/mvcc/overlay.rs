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

//! SWMR v2 Phase E: per-table reader overlay backed by the WAL tail.
//!
//! A read-only `ReadOnlyDatabase` sees the writer's state as of the last
//! checkpoint via cold volumes + manifest. Live SWMR additionally surfaces
//! committed-but-uncheckpointed rows by tailing the WAL between
//! `checkpoint_lsn` and the writer's `visible_commit_lsn` (published in
//! `db.shm`). Those rows live in a per-table in-memory overlay built
//! from `WALManager::tail_committed_entries`.
//!
//! This module owns the data structures and rebuild logic. Query-time
//! integration (merging the overlay into scan output with newest-wins
//! semantics) is wired up in a follow-up phase. Until then the overlay
//! is observable only via test/diagnostic accessors — refresh
//! correctness can be validated without touching every scanner.
//!
//! ## Visibility rules (single-source-of-truth)
//!
//! - **For row_id R**: the overlay's latest entry wins over any cold
//!   volume row. Newest WAL entry within the snapshot bound wins
//!   over older ones.
//! - **Tombstone (DELETE)**: marks the row as gone for this snapshot.
//!   Cold volumes and prior overlay entries are shadowed.
//! - **Snapshot bound**: only entries whose owning txn has a commit
//!   marker with `marker_lsn <= visible_commit_lsn` are applied. The
//!   `WALManager::tail_committed_entries` filter already enforces
//!   this.
//! - **Idempotency**: rebuilding from `from_lsn = 0, to_lsn = V` always
//!   produces the same overlay state. Re-applying entries the writer
//!   has already checkpointed is harmless because cold rows for the
//!   same row_id will be normalized to the same payload.
//!
//! ## Memory profile
//!
//! Bounded by the WAL gap between the last checkpoint and the writer's
//! current position. Default checkpoint cadence is 60s, so a continuous
//! 100K rows/sec writer produces ~6M overlay rows worst-case. Each
//! `OverlayRow::Live` holds an `Arc<RowVersion>`; tombstones are 0
//! bytes inline. The overlay is dropped on every full rebuild, so
//! steady-state memory tracks one snapshot's worth of WAL deltas.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::core::Result;
use crate::storage::mvcc::persistence::deserialize_row_version;
use crate::storage::mvcc::wal_manager::{WALManager, WALOperationType};
use crate::storage::mvcc::RowVersion;

/// One row's state in the overlay. `Live` shadows any cold version of
/// the same row_id; `Tombstone` makes the row invisible.
#[derive(Debug, Clone)]
pub enum OverlayRow {
    /// Live row from a committed Insert or Update WAL entry.
    Live(Arc<RowVersion>),
    /// Row was deleted by a committed Delete WAL entry. Shadows any
    /// cold row with the same row_id.
    Tombstone,
}

/// Per-table overlay: row_id -> latest committed state from the WAL
/// tail, bounded by the snapshot's `visible_commit_lsn`. Built fresh
/// on each `rebuild` call (no incremental updates yet).
#[derive(Debug, Default)]
pub struct TableOverlay {
    /// Row-id keyed map. We use `FxHashMap` (memory.md guidance for
    /// `i64` keys mentions `I64Map`, but `I64Map` reserves
    /// `i64::MIN` as a sentinel — overlay row_ids could in principle
    /// hit any value, and `FxHash` is fast enough for the rebuild
    /// frequency).
    rows: FxHashMap<i64, OverlayRow>,
}

impl TableOverlay {
    /// Empty overlay. Used as the starting state of every rebuild.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a single committed WAL entry to this table's overlay.
    /// Caller must have verified `entry.table_name` matches this
    /// overlay's table. Newest-wins: a later call for the same
    /// `row_id` overwrites an earlier one. Aborted/in-flight entries
    /// must be filtered upstream — this method assumes the entry is
    /// committed.
    fn apply(&mut self, entry_op: WALOperationType, row_id: i64, version: Arc<RowVersion>) {
        match entry_op {
            WALOperationType::Insert | WALOperationType::Update => {
                self.rows.insert(row_id, OverlayRow::Live(version));
            }
            WALOperationType::Delete => {
                self.rows.insert(row_id, OverlayRow::Tombstone);
            }
            // Other ops (DDL, etc.) shouldn't reach here — the WAL-tail
            // already filters DDL out. Defensive: ignore.
            _ => {}
        }
    }

    /// Number of overlay rows (live + tombstones). Used by tests and
    /// the diagnostic PRAGMA.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True if no rows are tracked. Helpful for the common
    /// "checkpoint just landed; overlay should be empty" case.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Inspect a single row_id's overlay state. `None` means no
    /// overlay entry — readers fall through to cold storage.
    pub fn get(&self, row_id: i64) -> Option<&OverlayRow> {
        self.rows.get(&row_id)
    }

    /// Iterate all overlay rows. Order is unspecified (FxHashMap).
    /// Used by test assertions and the diagnostic PRAGMA dump.
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &OverlayRow)> {
        self.rows.iter()
    }
}

/// SWMR v2 Phase H: one observed DDL event from the WAL tail. The
/// reader can't apply DDL to a live read-only handle (engine
/// metadata is structurally immutable), so these are surfaced via
/// `Error::SwmrPendingDdl` so the caller knows to reopen.
#[derive(Debug, Clone)]
pub struct DdlEvent {
    /// Lower-case target table name (or index/view name for non-table
    /// DDL). Empty when the DDL has no table affinity.
    pub table_name: String,
    /// WAL operation type — distinguishes CreateTable vs AlterTable,
    /// etc. Stored as a `WALOperationType` so the reader can render
    /// a precise diagnostic without re-parsing the entry.
    pub operation: WALOperationType,
    /// LSN of the DDL entry. Useful for ordering when multiple DDL
    /// events fall in the same refresh window.
    pub lsn: u64,
    /// Raw entry payload, populated only for CreateIndex events so
    /// `ReadOnlyDatabase::refresh` can decode the index name from
    /// `IndexMetadata` and suppress re-records of indexes the reader
    /// already knows about. `None` for all other DDL ops to avoid
    /// pinning serialized schema bytes for the lifetime of the event.
    pub payload: Option<Vec<u8>>,
}

/// Reader-side overlay store: one `TableOverlay` per table the reader
/// has seen WAL tail entries for. Rebuilt from scratch on each
/// `rebuild_from_wal` call so visibility semantics are obvious (no
/// incremental application bugs).
#[derive(Debug, Default)]
pub struct OverlayStore {
    /// Lower-cased table name -> overlay. RwLock so query-time reads
    /// can run concurrently with each other but not during rebuild.
    tables: RwLock<FxHashMap<String, Arc<TableOverlay>>>,
    /// Highest WAL LSN this overlay reflects. Reader compares against
    /// `db.shm.visible_commit_lsn` to decide whether a rebuild is
    /// needed (skip when already current).
    last_applied_lsn: AtomicU64,
    /// SWMR v2 P2 perf fix: snapshot of the writer's published
    /// `oldest_active_txn_lsn` taken at the END of the previous
    /// rebuild. Used as the Phase 2 entry-scan floor on the NEXT
    /// rebuild — anything below this LSN belongs to a transaction
    /// that committed (and was applied) before the previous
    /// rebuild, so the next rebuild can skip those entries
    /// entirely. `0` means "no snapshot yet, scan from 0
    /// (conservative)". Updated atomically after each successful
    /// rebuild.
    next_entry_floor: AtomicU64,
    /// SWMR v2 Phase H: DDL events observed in the most recent
    /// rebuild window. Cleared on each rebuild and re-populated. The
    /// reader's refresh path inspects this after rebuild and surfaces
    /// `Error::SwmrPendingDdl` if non-empty so callers can reopen
    /// rather than running queries against a stale schema view.
    pending_ddl: RwLock<Vec<DdlEvent>>,
    /// Per-reader CRC-validated WAL byte cursor. Records `(path,
    /// offset)` such that bytes in `[0, offset)` of `path` have
    /// been verified by THIS reader's prior tail. The next
    /// `tail_committed_entries` call passes this in to skip
    /// re-reading + re-CRC-validating those bytes — refresh
    /// cost on a polling read-only handle drops from
    /// O(WAL size) to O(delta).
    ///
    /// `None` after construction (cold start) and any time the
    /// cached file no longer matches the current scan target
    /// (file rotation). Updated under the same `parking_lot`
    /// mutex on every successful tail.
    wal_scan_cursor: parking_lot::Mutex<Option<crate::storage::mvcc::wal_manager::WalScanCursor>>,
}

impl OverlayStore {
    /// Empty store. The reader creates one at construction; it stays
    /// empty until the first `rebuild_from_wal` call.
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty store with `last_applied_lsn` initialized to a known
    /// engine replay frontier and `next_entry_floor` initialized
    /// to the writer's `oldest_active_txn_lsn` snapshot at attach
    /// (clamped to `<= visible_commit_lsn` like
    /// `rebuild_from_wal`'s steady-state advance).
    ///
    /// `last_applied_lsn = visible_commit_lsn`: the engine already
    /// replayed up to here via `open_engine`'s capped recovery,
    /// so the next tail's commit-marker scan starts after this LSN.
    /// Without seeding, the first tail would scan from 0 and trip
    /// `SwmrSnapshotExpired` if pre-attach WAL was truncated.
    ///
    /// `next_entry_floor` is INCLUSIVE: Phase 2 scans entries with
    /// LSN >= floor. `last_applied_lsn = visible_commit_lsn` is
    /// the already-consumed commit-marker frontier, so the next
    /// floor for the no-active-txn case must be
    /// `visible_commit_lsn + 1` (skip what we've already applied).
    /// Sentinel handling:
    ///   - `writer_oldest_active_txn_lsn = 0`: writer hasn't
    ///     published the watermark yet (e.g. fresh shm) — treat as
    ///     "scan from 0" so we don't accidentally skip pre-attach
    ///     DML for any transaction.
    ///   - `writer_oldest_active_txn_lsn = u64::MAX`: writer says
    ///     "no active user txns at attach time" → use
    ///     `visible_commit_lsn + 1`.
    ///   - any other value: use the writer's value (it's the lowest
    ///     LSN of any active txn's first DML, which is what Phase 2
    ///     needs to scan from).
    ///
    /// Clamped to `<= visible_commit_lsn + 1` so we never advance
    /// past what we've actually consumed.
    pub fn with_baseline(visible_commit_lsn: u64, writer_oldest_active_txn_lsn: u64) -> Self {
        let s = Self::default();
        s.last_applied_lsn
            .store(visible_commit_lsn, Ordering::Release);
        let already_consumed_ceiling = visible_commit_lsn.saturating_add(1);
        let entry_floor = match writer_oldest_active_txn_lsn {
            0 => 0,
            u64::MAX => already_consumed_ceiling,
            real => real.min(already_consumed_ceiling),
        };
        s.next_entry_floor.store(entry_floor, Ordering::Release);
        s
    }

    /// WAL pin LSN that protects exactly the entries this overlay's
    /// initial baseline needs to scan.
    ///
    /// Mirrors `with_baseline`'s `entry_floor` formula but folded
    /// into the lease pin's "1 = keep all WAL, 0 = released"
    /// semantics:
    ///   - `oldest_active = 0` (writer hasn't published the
    ///     watermark yet → conservative "scan from 0"): pin at
    ///     `1`. `0` would release the contribution.
    ///   - `oldest_active = u64::MAX` (writer says "no active
    ///     user txns at attach"): pin at `visible_commit_lsn + 1`.
    ///     Pre-attach commits are already replayed by the
    ///     engine; the next tail starts strictly above the
    ///     attached frontier.
    ///   - otherwise: pin at `min(oldest_active, visible_commit_lsn + 1)`.
    ///     A writer transaction that was active at attach may
    ///     have written DML at LSN < `visible_commit_lsn` and
    ///     commit LATER. The reader's overlay must scan from
    ///     that earlier DML's LSN, so the pin must protect WAL
    ///     from there. Clamped to `<= visible_commit_lsn + 1` so
    ///     we never pin past what we'd actually consume.
    ///
    /// This is the ONE source of truth for any pin install that
    /// represents a reader-side attach: `from_entry`,
    /// `DatabaseInner::new_with_entry`, and the bootstrap pin
    /// lowering in `Database::open` all use this so a writer can
    /// never truncate WAL the reader still needs.
    pub fn initial_pin_lsn(visible_commit_lsn: u64, writer_oldest_active_txn_lsn: u64) -> u64 {
        let already_consumed_ceiling = visible_commit_lsn.saturating_add(1);
        match writer_oldest_active_txn_lsn {
            0 => 1,
            u64::MAX => already_consumed_ceiling.max(1),
            real => real.min(already_consumed_ceiling).max(1),
        }
    }

    /// Highest LSN currently reflected in the overlay. Returns 0 when
    /// no rebuild has run yet. Compared by the reader's refresh path
    /// against `visible_commit_lsn` to skip no-op rebuilds.
    pub fn last_applied_lsn(&self) -> u64 {
        self.last_applied_lsn.load(Ordering::Acquire)
    }

    /// SWMR v2: lowest LSN the reader still needs to
    /// scan on its NEXT rebuild. The reader's WAL pin
    /// (`pinned_lsn`) should be set to this value AFTER a successful
    /// rebuild so the writer's truncate floor preserves exactly the
    /// WAL range the reader still consults. `0` before any rebuild
    /// has run (conservative: pin everything).
    pub fn next_entry_floor(&self) -> u64 {
        self.next_entry_floor.load(Ordering::Acquire)
    }

    /// Number of tables with at least one overlay row. Tests and the
    /// diagnostic PRAGMA expose this.
    pub fn table_count(&self) -> usize {
        self.tables.read().len()
    }

    /// Get a snapshot reference to one table's overlay. `Arc` clone is
    /// cheap; the returned overlay is whatever was current at call
    /// time — a concurrent rebuild may install a new one but won't
    /// invalidate this one.
    pub fn table(&self, table_name: &str) -> Option<Arc<TableOverlay>> {
        // Lower-case the lookup key; the overlay always stores
        // lower-case table names so casing in user SQL doesn't matter.
        let lower = table_name.to_lowercase();
        self.tables.read().get(&lower).cloned()
    }

    /// Drop every table's overlay. Used by tests and as a
    /// belt-and-braces step when the reader notices a writer
    /// reincarnation (writer_generation bumped) where the cached
    /// overlay state is no longer trustworthy.
    pub fn clear(&self) {
        self.tables.write().clear();
        self.last_applied_lsn.store(0, Ordering::Release);
        self.next_entry_floor.store(0, Ordering::Release);
        self.pending_ddl.write().clear();
    }

    /// Drop per-table row state and pending DDL but KEEP the LSN
    /// cursors (`last_applied_lsn`, `next_entry_floor`). Used by
    /// the reader's refresh path on manifest epoch advance: after
    /// the writer checkpoints, the per-table row state is stale
    /// (cold has the authoritative version) and would just leak
    /// memory unbounded. But resetting the LSN cursors would
    /// either:
    ///   - leave them at 0, causing the next tail to start from
    ///     the beginning of WAL — likely below the writer's
    ///     truncation floor (`SwmrSnapshotExpired`), AND
    ///   - skip any post-checkpoint DDL committed in the gap
    ///     between the prior `last_applied_lsn` and the new
    ///     visible frontier (lost SwmrPendingDdl / hidden new DDL).
    ///
    /// Keeping the cursors ensures the next tail picks up exactly
    /// where the prior one left off — the WAL pin (which tracks
    /// `next_entry_floor`) keeps that range alive.
    pub fn clear_rows_only(&self) {
        self.tables.write().clear();
        self.pending_ddl.write().clear();
    }

    /// SWMR v2 Phase H: snapshot of DDL events observed in the most
    /// recent rebuild window. Empty when no DDL fell in the window.
    /// Cleared on every `rebuild_from_wal` (and `clear`).
    pub fn pending_ddl(&self) -> Vec<DdlEvent> {
        self.pending_ddl.read().clone()
    }

    /// Apply WAL-tail delta from `(last_applied_lsn, to_lsn]` on top
    /// of the existing overlay. Atomic per-call: either every entry
    /// in the delta is applied or none (on tail/decode error, the
    /// overlay is left at its prior state and an error is returned).
    /// Idempotent across `to_lsn` values: calling twice with the
    /// same `to_lsn` is a no-op (the second call's delta is empty).
    ///
    /// `to_lsn` is normally the writer's published `visible_commit_lsn`
    /// from `db.shm`. Pass `0` to clear the overlay (no entries are
    /// in range).
    ///
    /// Correctness contract: `tail_committed_entries` scans commit
    /// markers in `(last_applied_lsn, to_lsn]` (incremental — small)
    /// but scans DML entries up to `to_lsn` from LSN 0 (full, so
    /// long-running explicit transactions that only just now
    /// committed still surface their pre-window DML). Without that
    /// asymmetric bound, a sequence like:
    ///   1. Txn A writes DML at LSN L1 (no commit marker yet)
    ///   2. Reader refreshes to LSN > L1 (Txn A still in-flight)
    ///   3. Txn B commits (overlay advances past L1)
    ///   4. Txn A commits at LSN > L1
    ///
    /// would silently lose Txn A's row from the overlay.
    ///
    /// Performance: the entry scan is O(live WAL size) per refresh.
    /// Future work: a writer-published "oldest-active-txn LSN"
    /// watermark would let the entry scan cap at that watermark
    /// instead of 0.
    ///
    /// Memory: the existing per-table overlays grow with newly
    /// applied entries; old entries are NOT pruned (they remain
    /// authoritative for row_ids the new delta doesn't touch).
    /// `clear()` resets fully.
    /// `dml_apply = false` runs in DDL-detection-only mode: the WAL
    /// tail scan still happens (so DDL events get collected into
    /// `pending_ddl` and `last_applied_lsn` / `next_entry_floor`
    /// advance), but per-row deserialization and per-table overlay
    /// merging are skipped. This is the default for read-only handles
    /// that haven't enabled query-time overlay materialization yet —
    /// they still need DDL detection to surface SwmrPendingDdl.
    pub fn rebuild_from_wal(
        &self,
        wal: &WALManager,
        to_lsn: u64,
        writer_oldest_active_txn_lsn: u64,
        dml_apply: bool,
    ) -> Result<()> {
        if to_lsn == 0 {
            self.clear();
            return Ok(());
        }
        let from_lsn = self.last_applied_lsn.load(Ordering::Acquire);
        if to_lsn <= from_lsn {
            // No-op: caller's snapshot is already at or beyond
            // to_lsn. We still have to clear pending_ddl from the
            // PRIOR rebuild — leaving it populated would cause
            // refresh() to re-raise SwmrPendingDdl on every call.
            // This matches Phase H semantics ("DDL events from the
            // most recent rebuild window") even on a no-op call.
            self.pending_ddl.write().clear();
            // Watermark-only advance: even with no new commits,
            // the writer may have re-published a higher
            // `oldest_active_txn_lsn` (e.g. a long-running txn
            // committed/rolled back without producing a new
            // visible commit). Advance `next_entry_floor` so the
            // reader's lease pin can move forward and stop
            // pinning WAL the writer wants to truncate. Same
            // formula as the post-tail update below; clamped to
            // `<= to_lsn` so we never advance past what we've
            // applied.
            let next_floor_advance = if writer_oldest_active_txn_lsn == u64::MAX {
                to_lsn.saturating_add(1)
            } else {
                writer_oldest_active_txn_lsn.min(to_lsn)
            };
            let current = self.next_entry_floor.load(Ordering::Acquire);
            if next_floor_advance > current {
                self.next_entry_floor
                    .store(next_floor_advance, Ordering::Release);
            }
            return Ok(());
        }
        // P2 perf fix: the watermark MUST come from the WRITER's
        // published `db.shm.oldest_active_txn_lsn` (the caller's
        // `writer_oldest_active_txn_lsn` parameter), NOT from
        // `wal.oldest_active_txn_lsn()` — the latter is a
        // reader-LOCAL cache that's always empty in cross-process
        // SWMR (the reader process doesn't write DML, so its WAL
        // manager has no active user txns to track). Using the
        // local cache would advance the floor to to_lsn even when
        // the writer has long-running in-flight txns whose DML
        // lives below to_lsn, silently losing those rows on the
        // next refresh.
        //
        // For THIS rebuild, use `next_entry_floor` — the snapshot
        // taken at the END of the PRIOR rebuild. On the very first
        // rebuild this is 0 (full DML scan).
        let entry_floor_now = self.next_entry_floor.load(Ordering::Acquire);
        let watermark_snapshot = writer_oldest_active_txn_lsn;
        // Incremental tail: only the (from_lsn, to_lsn] commit
        // window, and DML entries above `entry_floor_now`. Errors
        // propagate as SwmrOverlayApplyFailed.
        // Pass `dml_apply` as `include_dml` so the WAL tail skips
        // DML payload collection in DDL-only mode. Without this,
        // every refresh on a high-write-rate workload decodes +
        // allocates row deltas the overlay would just discard,
        // burning CPU and memory on default ReadOnlyDatabase
        // handles that haven't enabled query-time overlay
        // materialization.
        // Preserve the typed `SwmrSnapshotExpired` variant so the
        // caller (`ReadOnlyDatabase::maybe_auto_refresh`) can match
        // it as a must-reopen condition. Wrapping it as
        // `SwmrOverlayApplyFailed` would make that branch
        // unreachable and report a missing-WAL snapshot as a
        // retryable overlay failure. Only decode/CRC/IO failures
        // get wrapped — those are genuinely retryable from the
        // caller's perspective.
        //
        // Snapshot the per-reader CRC-validated WAL byte cursor
        // so `tail_committed_entries` can resume scanning past
        // already-validated bytes. The new cursor is captured
        // into `out_cursor` ONLY on successful tail; a failed
        // tail leaves the prior cursor untouched (the in-window
        // bytes were not validated this attempt).
        let prior_cursor = self.wal_scan_cursor.lock().clone();
        let mut new_cursor: Option<crate::storage::mvcc::wal_manager::WalScanCursor> = None;
        let entries = match wal.tail_committed_entries(
            from_lsn,
            entry_floor_now,
            to_lsn,
            dml_apply,
            prior_cursor.as_ref(),
            Some(&mut new_cursor),
        ) {
            Ok(e) => e,
            Err(e @ crate::core::Error::SwmrSnapshotExpired { .. }) => return Err(e),
            Err(e) => {
                return Err(crate::core::Error::SwmrOverlayApplyFailed(e.to_string()));
            }
        };
        if let Some(c) = new_cursor {
            *self.wal_scan_cursor.lock() = Some(c);
        }

        // Build the delta into a fresh per-table map first so a
        // mid-stream decode error can leave the existing overlay
        // unchanged (atomic semantics). DDL is collected separately.
        let mut delta_tables: FxHashMap<String, TableOverlay> = FxHashMap::default();
        let mut new_ddl: Vec<DdlEvent> = Vec::new();
        for entry in entries {
            if entry.operation.is_ddl() {
                let payload = if matches!(
                    entry.operation,
                    crate::storage::mvcc::wal_manager::WALOperationType::CreateIndex
                ) {
                    Some(entry.data.clone())
                } else {
                    None
                };
                new_ddl.push(DdlEvent {
                    table_name: entry.table_name.to_lowercase(),
                    operation: entry.operation,
                    lsn: entry.lsn,
                    payload,
                });
                continue;
            }
            let table_lower = entry.table_name.to_lowercase();
            // Skip per-row deserialize + per-table apply when in
            // DDL-only mode. We still need the WAL scan to advance
            // last_applied_lsn / next_entry_floor and to surface
            // DDL events; row materialization is the expensive part
            // (full Row deserialization + per-table Arc clone + map
            // insert) and is the part that's gated by
            // `swmr_overlay_enabled` until query execution actually
            // consumes overlay rows.
            if !dml_apply {
                continue;
            }
            let overlay = delta_tables.entry(table_lower).or_default();
            match deserialize_row_version(&entry.data) {
                Ok(rv) => {
                    overlay.apply(entry.operation, entry.row_id, Arc::new(rv));
                }
                Err(e) => {
                    return Err(crate::core::Error::SwmrOverlayApplyFailed(format!(
                        "decode row table={} row_id={} lsn={}: {}",
                        entry.table_name, entry.row_id, entry.lsn, e
                    )));
                }
            }
        }

        // Merge the delta into the live overlay map. For each
        // touched table, take the existing Arc<TableOverlay>, clone
        // its rows into a new TableOverlay, layer the delta on top
        // (newest-wins via the same `apply` semantics), and re-Arc
        // it. Untouched tables stay shared (no Arc churn). Empty
        // when `dml_apply == false`, so this whole block becomes a
        // no-op.
        if !delta_tables.is_empty() {
            let mut tables = self.tables.write();
            for (name, delta) in delta_tables {
                let merged = match tables.get(&name) {
                    Some(existing) => {
                        let mut merged = TableOverlay {
                            rows: existing.rows.clone(),
                        };
                        for (row_id, ov) in delta.rows {
                            merged.rows.insert(row_id, ov);
                        }
                        merged
                    }
                    None => delta,
                };
                tables.insert(name, Arc::new(merged));
            }
        }
        *self.pending_ddl.write() = new_ddl;
        self.last_applied_lsn.store(to_lsn, Ordering::Release);
        // P2 perf fix: persist the watermark snapshot taken at the
        // start of this rebuild as the entry-scan floor for the
        // NEXT rebuild. Clamp to `<= to_lsn` so we never advance
        // the floor past what we've actually applied. `u64::MAX`
        // from the writer means "no active user txns" — there's
        // nothing below `to_lsn` we still need to scan, so advance
        // the floor PAST `to_lsn` (to `to_lsn + 1`) instead of
        // landing on it. Otherwise the writer's WAL pin floor
        // (`min_pinned_lsn = next_entry_floor`) holds the last
        // consumed commit marker — and the entire WAL file
        // containing it — pinned until a later commit advances
        // the floor. Mirrors the `with_baseline` initial seed of
        // `visible_commit_lsn + 1` for the same reason.
        let next_floor = if watermark_snapshot == u64::MAX {
            to_lsn.saturating_add(1)
        } else {
            watermark_snapshot.min(to_lsn)
        };
        self.next_entry_floor.store(next_floor, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Row, Value};
    use crate::storage::config::SyncMode;
    use crate::storage::mvcc::{RowVersion, WALEntry, WALOperationType};

    fn make_version(value: i64) -> RowVersion {
        // RowVersion has no row_id field — that's on the WAL entry.
        // The version carries txn_id, deleted_at_txn_id, data, create_time.
        RowVersion::new(100, Row::from_values(vec![Value::Integer(value)]))
    }

    fn make_tombstone_version() -> RowVersion {
        let mut v = RowVersion::new(100, Row::new());
        v.deleted_at_txn_id = 200;
        v
    }

    fn tmp_wal() -> (tempfile::TempDir, WALManager) {
        let dir = tempfile::tempdir().expect("tempdir");
        // WALManager::new returns a running manager (running flag = true
        // on construction). No separate start() call needed.
        let wal = WALManager::new(dir.path(), SyncMode::Full).expect("wal manager");
        (dir, wal)
    }

    #[test]
    fn empty_store_has_zero_lsn_and_no_tables() {
        let s = OverlayStore::new();
        assert_eq!(s.last_applied_lsn(), 0);
        assert_eq!(s.table_count(), 0);
        assert!(s.table("anything").is_none());
    }

    #[test]
    fn rebuild_with_to_lsn_zero_clears() {
        let s = OverlayStore::new();
        // Pretend we had something. (We can't easily put one in
        // without rebuild, but clear() should also work.)
        s.last_applied_lsn.store(42, Ordering::Release);

        let (_dir, wal) = tmp_wal();
        s.rebuild_from_wal(&wal, 0, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(s.last_applied_lsn(), 0, "to_lsn=0 must reset");
        assert_eq!(s.table_count(), 0);
    }

    #[test]
    fn rebuild_picks_up_committed_inserts() {
        let (_dir, wal) = tmp_wal();
        // Append: txn 1 inserts row 10 into "t", then commits.
        let v = make_version(999);
        let e = WALEntry::new(
            1,
            "t".to_string(),
            10,
            WALOperationType::Insert,
            crate::storage::mvcc::serialize_row_version(&v).unwrap(),
        );
        let _ = wal.append_entry(e).unwrap();
        let commit_lsn = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();

        assert_eq!(s.last_applied_lsn(), commit_lsn);
        let overlay = s.table("t").expect("table 't' must be in overlay");
        assert_eq!(overlay.len(), 1);
        match overlay.get(10).unwrap() {
            OverlayRow::Live(rv) => {
                // RowVersion stores the row in `data`; row_id lived on
                // the WAL entry envelope, not the version. Verify the
                // row payload survived the round-trip.
                let v0 = rv.data.get(0).cloned();
                assert_eq!(v0, Some(Value::Integer(999)), "round-tripped row payload");
            }
            OverlayRow::Tombstone => panic!("expected Live, got Tombstone"),
        }
    }

    #[test]
    fn rebuild_skips_uncommitted_entries() {
        let (_dir, wal) = tmp_wal();
        // txn 1 inserts row 10 — but no commit marker for txn 1.
        let v = make_version(1);
        let e = WALEntry::new(
            1,
            "t".to_string(),
            10,
            WALOperationType::Insert,
            crate::storage::mvcc::serialize_row_version(&v).unwrap(),
        );
        let lsn = wal.append_entry(e).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        // Rebuild up to the entry's own LSN — but with no commit marker,
        // tail_committed_entries should not return it.
        s.rebuild_from_wal(&wal, lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();

        assert_eq!(s.last_applied_lsn(), lsn);
        assert!(
            s.table("t").is_none() || s.table("t").unwrap().is_empty(),
            "uncommitted txn must NOT appear in overlay"
        );
    }

    #[test]
    fn rebuild_applies_tombstones_for_committed_deletes() {
        let (_dir, wal) = tmp_wal();
        // Insert + commit, then Delete + commit (different txn).
        let v = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v).unwrap(),
            ))
            .unwrap();
        let _ = wal.write_commit_marker(1, 100).unwrap();

        let tomb = make_tombstone_version();
        let _ = wal
            .append_entry(WALEntry::new(
                2,
                "t".to_string(),
                10,
                WALOperationType::Delete,
                crate::storage::mvcc::serialize_row_version(&tomb).unwrap(),
            ))
            .unwrap();
        let final_lsn = wal.write_commit_marker(2, 200).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, final_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let overlay = s.table("t").expect("table must exist");
        assert!(matches!(overlay.get(10), Some(OverlayRow::Tombstone)));
    }

    #[test]
    fn rebuild_with_lsn_below_commit_marker_excludes_those_entries() {
        let (_dir, wal) = tmp_wal();
        let v = make_version(1);
        let entry_lsn = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v).unwrap(),
            ))
            .unwrap();
        // Commit marker comes AFTER entry. If the snapshot bound is
        // BELOW the commit marker LSN, the entry must NOT be applied.
        let _commit_lsn = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        // Rebuild at the entry LSN, which is < commit marker LSN.
        s.rebuild_from_wal(&wal, entry_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        // Without the commit marker visible at the snapshot bound,
        // the entry must be excluded.
        assert!(
            s.table("t").map(|t| t.is_empty()).unwrap_or(true),
            "entry must be excluded when commit marker LSN > snapshot bound"
        );
    }

    #[test]
    fn rebuild_is_idempotent_on_same_to_lsn() {
        let (_dir, wal) = tmp_wal();
        let v = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v).unwrap(),
            ))
            .unwrap();
        let commit_lsn = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let count_after_first = s.table("t").unwrap().len();
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let count_after_second = s.table("t").unwrap().len();
        assert_eq!(
            count_after_first, count_after_second,
            "rebuild with the same to_lsn must produce the same state"
        );
    }

    #[test]
    fn rebuild_separates_ddl_into_pending_ddl_list() {
        // SWMR v2 Phase H: a CREATE TABLE entry committed in the
        // tail window must NOT pollute the per-table DML overlay.
        // Instead it surfaces in pending_ddl so the reader can raise
        // SwmrPendingDdl. We synthesize a CreateTable WAL entry +
        // commit marker and verify the split.
        let (_dir, wal) = tmp_wal();

        // CreateTable for "new_table". Schema payload is opaque to
        // the overlay (it would normally be the serialized Schema).
        let _ = wal
            .append_entry(WALEntry::new(
                0, // DDL_TXN_ID
                "new_table".to_string(),
                0,
                WALOperationType::CreateTable,
                vec![1, 2, 3],
            ))
            .unwrap();
        let commit_lsn = wal.write_commit_marker(0, 0).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();

        // No DML overlay row for new_table.
        assert!(
            s.table("new_table").is_none() || s.table("new_table").unwrap().is_empty(),
            "DDL must NOT populate the per-table DML overlay"
        );
        // pending_ddl carries the event.
        let ddl = s.pending_ddl();
        assert_eq!(ddl.len(), 1, "exactly one DDL event");
        assert_eq!(ddl[0].table_name, "new_table");
        assert!(matches!(ddl[0].operation, WALOperationType::CreateTable));
        assert!(ddl[0].lsn > 0);
    }

    #[test]
    fn rebuild_clears_pending_ddl_on_subsequent_rebuild_with_no_new_ddl() {
        // Phase H: pending_ddl reflects ONLY events in the most
        // recent rebuild window. After the incremental P2 fix, a
        // second rebuild at the same to_lsn is a true no-op AND
        // explicitly clears pending_ddl from the prior call —
        // otherwise refresh() would re-raise SwmrPendingDdl on
        // every query without any new DDL.
        let (_dir, wal) = tmp_wal();
        let _ = wal
            .append_entry(WALEntry::new(
                0,
                "t".to_string(),
                0,
                WALOperationType::CreateTable,
                vec![],
            ))
            .unwrap();
        let _ = wal.write_commit_marker(0, 0).unwrap();
        let v = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v).unwrap(),
            ))
            .unwrap();
        let final_lsn = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, final_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(s.pending_ddl().len(), 1, "first rebuild sees the DDL");

        // Second rebuild at the SAME to_lsn: incremental tail has no
        // new entries, so no new DDL is observed. pending_ddl is
        // explicitly cleared so refresh() doesn't re-fire.
        s.rebuild_from_wal(&wal, final_lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert!(
            s.pending_ddl().is_empty(),
            "no-op rebuild must clear stale pending_ddl"
        );

        // The DML overlay survives the no-op rebuild.
        let overlay = s.table("t").expect("DML overlay must persist");
        assert_eq!(overlay.len(), 1);

        // clear() drops everything.
        s.clear();
        assert!(s.pending_ddl().is_empty(), "clear() drops pending_ddl");
        assert!(s.table("t").is_none(), "clear() drops tables");
    }

    #[test]
    fn incremental_rebuild_recovers_dml_for_late_commit_across_window() {
        // A transaction whose DML was written before
        // the previous tail bound but whose commit marker arrives
        // AFTER must still get its rows applied to the overlay on
        // the next refresh. Repro:
        //   1. Txn 1 writes DML at L1 (no commit marker yet)
        //   2. Reader refreshes to L_a (Txn 1 still in-flight)
        //   3. Txn 2 commits at L_b > L_a
        //   4. Reader refreshes incrementally — applies Txn 2's row
        //   5. Txn 1 commits at L_c > L_b
        //   6. Reader refreshes incrementally — MUST apply Txn 1's
        //      row (its DML LSN L1 < the previous tail bound L_b)
        let (_dir, wal) = tmp_wal();

        // Txn 1: writes DML for row 10 — NO commit marker yet.
        let v1 = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v1).unwrap(),
            ))
            .unwrap();
        wal.flush().unwrap();

        // Snapshot LSN BEFORE Txn 2's marker. We need a commit
        // marker at SOME LSN > Txn 1's DML so the reader has
        // somewhere to advance last_applied_lsn to.
        // Synthesize that via a no-op committed Txn (txn id 99,
        // no DML, just a commit marker).
        let lsn_a = wal.write_commit_marker(99, 99).unwrap();

        // Txn 2: commits a row 20.
        let v2 = make_version(2);
        let _ = wal
            .append_entry(WALEntry::new(
                2,
                "t".to_string(),
                20,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v2).unwrap(),
            ))
            .unwrap();
        let lsn_b = wal.write_commit_marker(2, 200).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        // Reader refreshes to lsn_a — Txn 1 is in-flight, Txn 99
        // committed (no rows). Overlay should be empty.
        s.rebuild_from_wal(&wal, lsn_a, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert!(
            s.table("t").map(|t| t.is_empty()).unwrap_or(true),
            "txn 1 in-flight, no rows expected"
        );
        // Reader refreshes to lsn_b — Txn 2 just committed.
        s.rebuild_from_wal(&wal, lsn_b, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(
            s.table("t").unwrap().len(),
            1,
            "only txn 2's row 20 should be visible"
        );
        assert!(matches!(
            s.table("t").unwrap().get(20),
            Some(OverlayRow::Live(_))
        ));

        // Now Txn 1 commits — its DML was at L1, BEFORE lsn_a (the
        // previous tail bound). The bug was that the entry scan
        // floored at last_applied = lsn_b, missing the Txn-1 DML.
        let lsn_c = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        s.rebuild_from_wal(&wal, lsn_c, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let overlay = s.table("t").unwrap();
        assert_eq!(
            overlay.len(),
            2,
            "after txn 1 commits, BOTH rows must be present (got {} entries)",
            overlay.len()
        );
        assert!(
            matches!(overlay.get(10), Some(OverlayRow::Live(_))),
            "row 10 from late-committing txn 1 must be applied"
        );
        assert!(matches!(overlay.get(20), Some(OverlayRow::Live(_))));
    }

    #[test]
    fn rebuild_advances_entry_floor_when_no_active_user_txns() {
        // P2 perf fix: when the writer has no in-flight user txns
        // at refresh time, the watermark is u64::MAX, and the
        // overlay's `next_entry_floor` advances to `to_lsn`. The
        // NEXT rebuild then floors its DML scan at `to_lsn`,
        // skipping all already-applied entries.
        let (_dir, wal) = tmp_wal();

        // Commit txn 1 (writes, marker, all clean).
        let v1 = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v1).unwrap(),
            ))
            .unwrap();
        let lsn_after_1 = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        // `write_commit_marker` no longer clears
        // active_txn_first_lsn — that's deferred to the engine's
        // publish_visible_commit_lsn so concurrent publishes see a
        // consistent snapshot. Simulate the engine's
        // post-publish clear here.
        wal.clear_active_txn(1);

        // After the clear, the writer's active map for user txns
        // is empty -> oldest_active_txn_lsn == u64::MAX.
        assert_eq!(wal.oldest_active_txn_lsn(), u64::MAX);

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_after_1, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(s.table("t").unwrap().len(), 1);
        // After rebuild with no active txns (watermark = u64::MAX),
        // next_entry_floor advances PAST the last consumed marker
        // to `to_lsn + 1`. Otherwise the writer's WAL pin floor
        // would hold the last consumed commit marker (and possibly
        // its WAL file) pinned until a later commit moved the
        // floor. Mirrors `with_baseline`'s `visible_commit_lsn + 1`
        // initial seed.
        assert_eq!(
            s.next_entry_floor.load(Ordering::Acquire),
            lsn_after_1.saturating_add(1),
            "next_entry_floor must advance past to_lsn when no active txns"
        );

        // Commit a second txn.
        let v2 = make_version(2);
        let _ = wal
            .append_entry(WALEntry::new(
                2,
                "t".to_string(),
                20,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v2).unwrap(),
            ))
            .unwrap();
        let lsn_after_2 = wal.write_commit_marker(2, 200).unwrap();
        wal.flush().unwrap();
        // Same deferred-clear simulation as above.
        wal.clear_active_txn(2);

        // Next rebuild: entry_floor = lsn_after_1 (from prior).
        // Phase 2 will skip txn 1's old entry at L1 < lsn_after_1
        // entirely (no header walk needed for it after file-skip,
        // and even if same file, the LSN check rejects it cheaply).
        s.rebuild_from_wal(&wal, lsn_after_2, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let overlay = s.table("t").unwrap();
        assert_eq!(
            overlay.len(),
            2,
            "both rows must be present (txn 1 from prior rebuild, txn 2 now)"
        );
        assert!(matches!(overlay.get(10), Some(OverlayRow::Live(_))));
        assert!(matches!(overlay.get(20), Some(OverlayRow::Live(_))));
    }

    #[test]
    fn rebuild_uses_explicit_watermark_not_local_wal_cache() {
        // In cross-process SWMR, the reader's WAL
        // manager has an EMPTY active-txn map (the reader doesn't
        // write DML), so `wal.oldest_active_txn_lsn()` returns
        // u64::MAX. The watermark must come from
        // `db.shm.oldest_active_txn_lsn` published by the WRITER
        // process. Simulate this by passing u64::MAX (what the
        // reader's local cache would return) AND the writer's
        // actual oldest-active LSN as the parameter.
        //
        // Sequence:
        //   1. Writer txn 1 writes DML at L1 (still active)
        //   2. Reader refresh at lsn_a (synthetic marker after L1)
        //      with watermark = L1 (writer's published value)
        //   3. Writer txn 2 commits at lsn_b
        //   4. Reader refresh at lsn_b — watermark still = L1
        //   5. Writer txn 1 commits at lsn_c
        //   6. Reader refresh at lsn_c — watermark = u64::MAX
        //   7. Row 10 from txn 1 must be present
        let (_dir, wal) = tmp_wal();

        let v1 = make_version(1);
        let l1 = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v1).unwrap(),
            ))
            .unwrap();
        let lsn_a = wal.write_commit_marker(99, 99).unwrap();

        let v2 = make_version(2);
        let _ = wal
            .append_entry(WALEntry::new(
                2,
                "t".to_string(),
                20,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v2).unwrap(),
            ))
            .unwrap();
        let lsn_b = wal.write_commit_marker(2, 200).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        // Step 2: Reader refresh — explicitly pass writer's
        // watermark = L1 (txn 1 still active in writer).
        s.rebuild_from_wal(&wal, lsn_a, l1, true).unwrap();
        // The next_entry_floor must be min(l1, lsn_a) = l1.
        assert_eq!(
            s.next_entry_floor.load(Ordering::Acquire),
            l1,
            "next floor must use the EXPLICIT watermark (l1), not the \
             local WAL cache (which would be u64::MAX in cross-process)"
        );

        // Step 4: Reader refresh again — writer still has txn 1
        // active so watermark is still L1. Floor stays low.
        s.rebuild_from_wal(&wal, lsn_b, l1, true).unwrap();
        assert_eq!(s.next_entry_floor.load(Ordering::Acquire), l1);

        // Step 5-7: txn 1 commits. Writer's watermark jumps to
        // u64::MAX. Reader refresh — must catch row 10.
        let lsn_c = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();
        s.rebuild_from_wal(&wal, lsn_c, u64::MAX, true).unwrap();
        assert!(
            matches!(s.table("t").unwrap().get(10), Some(OverlayRow::Live(_))),
            "row 10 from late-committing txn 1 must be applied; \
             reader-local watermark would have skipped it"
        );
        assert!(matches!(
            s.table("t").unwrap().get(20),
            Some(OverlayRow::Live(_))
        ));
    }

    #[test]
    fn rebuild_keeps_low_entry_floor_while_long_running_txn_active() {
        // P2 perf fix: while a long-running explicit txn is
        // in-flight, the watermark stays low (its first DML LSN),
        // so subsequent refreshes scan from that low floor — which
        // is exactly what's needed to catch the txn's pre-window
        // DML when it eventually commits.
        let (_dir, wal) = tmp_wal();

        // Long-running txn 1 writes DML at L1, no commit yet.
        let v1 = make_version(1);
        let l1 = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v1).unwrap(),
            ))
            .unwrap();
        wal.flush().unwrap();
        assert_eq!(wal.oldest_active_txn_lsn(), l1);

        // Commit a synthetic marker so the reader has somewhere to
        // advance last_applied_lsn to.
        let lsn_a = wal.write_commit_marker(99, 99).unwrap();
        wal.flush().unwrap();

        // Reader rebuilds — txn 1 still active, watermark = l1.
        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_a, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        // next_entry_floor must equal min(l1, lsn_a) = l1 because
        // l1 < lsn_a. This keeps the floor LOW so the next refresh
        // can still find txn 1's DML when it commits.
        assert_eq!(
            s.next_entry_floor.load(Ordering::Acquire),
            l1,
            "next_entry_floor must stay at the long-running txn's first DML LSN"
        );

        // Now txn 1 commits. The next rebuild MUST find row 10
        // even though its LSN is < lsn_a (the prior to_lsn).
        let lsn_b = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();
        s.rebuild_from_wal(&wal, lsn_b, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert!(
            matches!(s.table("t").unwrap().get(10), Some(OverlayRow::Live(_))),
            "row 10 must be applied after long-running txn finally commits"
        );
    }

    #[test]
    fn rebuild_does_not_replay_stale_ddl_when_new_ddl_commits() {
        // Every DDL entry uses the synthetic
        // DDL_TXN_ID = 0. A naive Phase 2 (scan from LSN 0,
        // emit any entry whose txn_id is in committed_txns) would
        // re-emit every prior DDL the moment a new DDL marker
        // arrived, bloating pending_ddl with history every refresh.
        //
        // Repro:
        //   1. Append CreateTable("t1") + commit marker
        //   2. Rebuild → pending_ddl = [CreateTable("t1")]
        //   3. Append CreateTable("t2") + commit marker
        //   4. Rebuild → pending_ddl MUST be [CreateTable("t2")] only
        let (_dir, wal) = tmp_wal();

        // First DDL.
        let _ = wal
            .append_entry(WALEntry::new(
                0,
                "t1".to_string(),
                0,
                WALOperationType::CreateTable,
                vec![],
            ))
            .unwrap();
        let lsn_after_t1 = wal.write_commit_marker(0, 0).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_after_t1, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(s.pending_ddl().len(), 1, "first refresh sees t1");
        assert_eq!(s.pending_ddl()[0].table_name, "t1");

        // Second DDL: a brand new table.
        let _ = wal
            .append_entry(WALEntry::new(
                0,
                "t2".to_string(),
                0,
                WALOperationType::CreateTable,
                vec![],
            ))
            .unwrap();
        let lsn_after_t2 = wal.write_commit_marker(0, 0).unwrap();
        wal.flush().unwrap();

        s.rebuild_from_wal(&wal, lsn_after_t2, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let ddl = s.pending_ddl();
        assert_eq!(
            ddl.len(),
            1,
            "second refresh must report ONLY t2, not t1+t2 (stale replay): {:?}",
            ddl.iter().map(|d| &d.table_name).collect::<Vec<_>>()
        );
        assert_eq!(ddl[0].table_name, "t2");
    }

    #[test]
    fn incremental_rebuild_preserves_prior_overlay_rows() {
        // P2 fix: the second rebuild at a higher LSN must layer new
        // entries on top of the existing overlay, not throw away
        // prior state.
        let (_dir, wal) = tmp_wal();
        // Txn 1: insert row 10.
        let v1 = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v1).unwrap(),
            ))
            .unwrap();
        let lsn_after_t1 = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_after_t1, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert_eq!(s.table("t").unwrap().len(), 1);

        // Txn 2: insert row 20. Incremental rebuild should add row
        // 20 WITHOUT losing row 10.
        let v2 = make_version(2);
        let _ = wal
            .append_entry(WALEntry::new(
                2,
                "t".to_string(),
                20,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v2).unwrap(),
            ))
            .unwrap();
        let lsn_after_t2 = wal.write_commit_marker(2, 200).unwrap();
        wal.flush().unwrap();

        s.rebuild_from_wal(&wal, lsn_after_t2, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        let overlay = s.table("t").unwrap();
        assert_eq!(
            overlay.len(),
            2,
            "both rows must be present after incremental"
        );
        assert!(matches!(overlay.get(10), Some(OverlayRow::Live(_))));
        assert!(matches!(overlay.get(20), Some(OverlayRow::Live(_))));
    }

    #[test]
    fn clear_resets_lsn_and_tables() {
        let (_dir, wal) = tmp_wal();
        let v = make_version(1);
        let _ = wal
            .append_entry(WALEntry::new(
                1,
                "t".to_string(),
                10,
                WALOperationType::Insert,
                crate::storage::mvcc::serialize_row_version(&v).unwrap(),
            ))
            .unwrap();
        let lsn = wal.write_commit_marker(1, 100).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn, wal.oldest_active_txn_lsn(), true)
            .unwrap();
        assert!(!s.table("t").unwrap().is_empty());

        s.clear();
        assert_eq!(s.last_applied_lsn(), 0);
        assert_eq!(s.table_count(), 0);
    }
}
