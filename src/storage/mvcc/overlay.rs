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

//! Reader-side WAL-tail cursor backed by `WALManager::tail_committed_entries`.
//!
//! A read-only `ReadOnlyDatabase` sees the writer's state as of the last
//! checkpoint via cold volumes + manifest. To pick up post-attach DDL the
//! reader can't apply live (engine metadata is structurally immutable),
//! we tail the WAL between `checkpoint_lsn` and the writer's
//! `visible_commit_lsn` (published in `db.shm`) and surface DDL events
//! as `Error::SwmrPendingDdl` so the caller knows to reopen.
//!
//! This module owns the WAL-scan cursor + DDL collection. Per-row DML
//! materialization is intentionally NOT done here: no scanner consumes
//! it. The WAL scan still advances `last_applied_lsn` and
//! `next_entry_floor` so the reader's lease pin tracks exactly the WAL
//! range the next refresh needs.
//!
//! ## Visibility rules
//!
//! - **Snapshot bound**: only entries whose owning txn has a commit
//!   marker with `marker_lsn <= visible_commit_lsn` are observed. The
//!   `WALManager::tail_committed_entries` filter already enforces this.
//! - **Idempotency**: rebuilding from `from_lsn = 0, to_lsn = V` always
//!   produces the same DDL set + cursor state.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::core::Result;
use crate::storage::mvcc::wal_manager::{WALManager, WALOperationType};

/// One observed DDL event from the WAL tail. The reader can't apply
/// DDL to a live read-only handle (engine metadata is structurally
/// immutable), so these are surfaced via `Error::SwmrPendingDdl` so
/// the caller knows to reopen.
#[derive(Debug, Clone)]
pub struct DdlEvent {
    /// Lower-case target table name (or index/view name for non-table
    /// DDL). Empty when the DDL has no table affinity.
    pub table_name: String,
    /// WAL operation type, distinguishes CreateTable vs AlterTable,
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

/// Reader-side WAL-tail state: DDL events observed in the most recent
/// refresh window plus LSN cursors that drive the WAL pin.
#[derive(Debug, Default)]
pub struct OverlayStore {
    /// Highest WAL LSN this cursor reflects. Reader compares against
    /// `db.shm.visible_commit_lsn` to decide whether a rebuild is
    /// needed (skip when already current).
    last_applied_lsn: AtomicU64,
    /// Snapshot of the writer's published `oldest_active_txn_lsn`
    /// taken at the END of the previous rebuild. Used as the Phase 2
    /// entry-scan floor on the NEXT rebuild, anything below this
    /// LSN belongs to a transaction that committed (and was applied)
    /// before the previous rebuild, so the next rebuild can skip
    /// those entries entirely. `0` means "no snapshot yet, scan
    /// from 0 (conservative)". Updated atomically after each
    /// successful rebuild.
    next_entry_floor: AtomicU64,
    /// DDL events observed in the most recent rebuild window. Cleared on each rebuild and re-populated. The
    /// reader's refresh path inspects this after rebuild and surfaces
    /// `Error::SwmrPendingDdl` if non-empty so callers can reopen
    /// rather than running queries against a stale schema view.
    pending_ddl: RwLock<Vec<DdlEvent>>,
    /// Per-reader CRC-validated WAL byte cursor. Records `(path,
    /// offset)` such that bytes in `[0, offset)` of `path` have
    /// been verified by THIS reader's prior tail. The next
    /// `tail_committed_entries` call passes this in to skip
    /// re-reading + re-CRC-validating those bytes, refresh
    /// cost on a polling read-only handle drops from
    /// O(WAL size) to O(delta).
    ///
    /// `None` after construction (cold start) and any time the
    /// cached file no longer matches the current scan target
    /// (file rotation). Updated under the same `parking_lot`
    /// mutex on every successful tail.
    wal_scan_cursor: parking_lot::Mutex<Option<crate::storage::mvcc::wal_manager::WalScanCursor>>,
    /// Test-observable count of WAL tail scans actually invoked.
    wal_scans: AtomicU64,
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
    ///     published the watermark yet (e.g. fresh shm), treat as
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

    /// Highest LSN currently reflected in the cursor. Returns 0 when
    /// no rebuild has run yet. Compared by the reader's refresh path
    /// against `visible_commit_lsn` to skip no-op rebuilds.
    pub fn last_applied_lsn(&self) -> u64 {
        self.last_applied_lsn.load(Ordering::Acquire)
    }

    /// Lowest LSN the reader still needs to scan on its NEXT
    /// rebuild. The reader's WAL pin
    /// (`pinned_lsn`) should be set to this value AFTER a successful
    /// rebuild so the writer's truncate floor preserves exactly the
    /// WAL range the reader still consults. `0` before any rebuild
    /// has run (conservative: pin everything).
    pub fn next_entry_floor(&self) -> u64 {
        self.next_entry_floor.load(Ordering::Acquire)
    }

    /// Snapshot of DDL events observed in the most recent rebuild
    /// window. Empty when no DDL fell in the window. Cleared on
    /// every `rebuild_from_wal`.
    pub fn pending_ddl(&self) -> Vec<DdlEvent> {
        self.pending_ddl.read().clone()
    }

    /// Total WAL tail scans invoked. Test-observable.
    pub fn wal_scan_count(&self) -> u64 {
        self.wal_scans.load(Ordering::Relaxed)
    }

    /// Watermark-only advance: clear pending_ddl and bump
    /// next_entry_floor without scanning the WAL. Shared by the two
    /// rebuild_from_wal fast paths (no new commits, or no new DDL).
    fn advance_cursors_only(&self, to_lsn: u64, writer_oldest_active_txn_lsn: u64) {
        self.pending_ddl.write().clear();
        let advance = if writer_oldest_active_txn_lsn == u64::MAX {
            to_lsn.saturating_add(1)
        } else {
            writer_oldest_active_txn_lsn.min(to_lsn)
        };
        let current = self.next_entry_floor.load(Ordering::Acquire);
        if advance > current {
            self.next_entry_floor.store(advance, Ordering::Release);
        }
        let last_applied = self.last_applied_lsn.load(Ordering::Acquire);
        if to_lsn > last_applied {
            self.last_applied_lsn.store(to_lsn, Ordering::Release);
        }
    }

    /// Tail the WAL in `(last_applied_lsn, to_lsn]`, collect DDL
    /// events into `pending_ddl`, and advance the LSN cursors.
    /// Idempotent across `to_lsn` values: calling twice with the
    /// same `to_lsn` is a no-op (the second call's delta is empty).
    ///
    /// `to_lsn` is normally the writer's published `visible_commit_lsn`
    /// from `db.shm`. Pass `0` to reset the cursors (no entries are
    /// in range).
    ///
    /// Correctness contract: `tail_committed_entries` scans commit
    /// markers in `(last_applied_lsn, to_lsn]` (incremental, small)
    /// but scans DML/DDL entries up to `to_lsn` from a watermark-
    /// derived floor (full enough that long-running explicit
    /// transactions that only just now committed still surface their
    /// pre-window entries).
    pub fn rebuild_from_wal(
        &self,
        wal: &WALManager,
        to_lsn: u64,
        writer_oldest_active_txn_lsn: u64,
        writer_last_ddl_lsn: u64,
    ) -> Result<()> {
        if to_lsn == 0 {
            self.last_applied_lsn.store(0, Ordering::Release);
            self.next_entry_floor.store(0, Ordering::Release);
            self.pending_ddl.write().clear();
            return Ok(());
        }
        let from_lsn = self.last_applied_lsn.load(Ordering::Acquire);
        // Fast path: no DDL since our last refresh. Advance the
        // cursors via the watermark; skip the tail scan entirely.
        if writer_last_ddl_lsn <= from_lsn {
            self.advance_cursors_only(to_lsn, writer_oldest_active_txn_lsn);
            return Ok(());
        }
        if to_lsn <= from_lsn {
            self.advance_cursors_only(to_lsn, writer_oldest_active_txn_lsn);
            return Ok(());
        }
        // The watermark MUST come from the WRITER's published
        // `db.shm.oldest_active_txn_lsn` (the caller's
        // `writer_oldest_active_txn_lsn` parameter), NOT from
        // `wal.oldest_active_txn_lsn()`, the latter is a
        // reader-LOCAL cache that's always empty in cross-process
        // SWMR (the reader process doesn't write DML, so its WAL
        // manager has no active user txns to track). Using the
        // local cache would advance the floor to to_lsn even when
        // the writer has long-running in-flight txns whose DML
        // lives below to_lsn, silently losing those rows on the
        // next refresh.
        //
        // For THIS rebuild, use `next_entry_floor`, the snapshot
        // taken at the END of the PRIOR rebuild. On the very first
        // rebuild this is 0 (full DML scan).
        let entry_floor_now = self.next_entry_floor.load(Ordering::Acquire);
        let watermark_snapshot = writer_oldest_active_txn_lsn;
        // Incremental tail: only the (from_lsn, to_lsn] commit
        // window, and entries above `entry_floor_now`. Errors
        // propagate as SwmrOverlayApplyFailed.
        // Pass `include_dml = false`: the cursor only needs DDL
        // events to surface SwmrPendingDdl. Skipping DML payload
        // collection avoids decoding + allocating row deltas no
        // scanner consumes.
        // Preserve the typed `SwmrSnapshotExpired` variant so the
        // caller (`ReadOnlyDatabase::maybe_auto_refresh`) can match
        // it as a must-reopen condition. Wrapping it as
        // `SwmrOverlayApplyFailed` would make that branch
        // unreachable and report a missing-WAL snapshot as a
        // retryable overlay failure. Only decode/CRC/IO failures
        // get wrapped, those are genuinely retryable from the
        // caller's perspective.
        //
        // Snapshot the per-reader CRC-validated WAL byte cursor
        // so `tail_committed_entries` can resume scanning past
        // already-validated bytes. The new cursor is captured
        // into `out_cursor` ONLY on successful tail; a failed
        // tail leaves the prior cursor untouched (the in-window
        // bytes were not validated this attempt).
        self.wal_scans.fetch_add(1, Ordering::Relaxed);
        let prior_cursor = self.wal_scan_cursor.lock().clone();
        let mut new_cursor: Option<crate::storage::mvcc::wal_manager::WalScanCursor> = None;
        let entries = match wal.tail_committed_entries(
            from_lsn,
            entry_floor_now,
            to_lsn,
            false,
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

        // Collect DDL events; DML entries are dropped (no scanner
        // consumes them). The WAL tail with `include_dml = false`
        // already filters most DML out, but defensively skip any
        // non-DDL that slipped through.
        let mut new_ddl: Vec<DdlEvent> = Vec::new();
        for entry in entries {
            if !entry.operation.is_ddl() {
                continue;
            }
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
        }

        *self.pending_ddl.write() = new_ddl;
        self.last_applied_lsn.store(to_lsn, Ordering::Release);
        // Persist the watermark snapshot taken at the start of this
        // rebuild as the entry-scan floor for the NEXT rebuild. Clamp to `<= to_lsn` so we never advance
        // the floor past what we've actually applied. `u64::MAX`
        // from the writer means "no active user txns", there's
        // nothing below `to_lsn` we still need to scan, so advance
        // the floor PAST `to_lsn` (to `to_lsn + 1`) instead of
        // landing on it. Otherwise the writer's WAL pin floor
        // (`min_pinned_lsn = next_entry_floor`) holds the last
        // consumed commit marker, and the entire WAL file
        // containing it, pinned until a later commit advances
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
        // RowVersion has no row_id field, that's on the WAL entry.
        // The version carries txn_id, deleted_at_txn_id, data, create_time.
        RowVersion::new(100, Row::from_values(vec![Value::Integer(value)]))
    }

    fn tmp_wal() -> (tempfile::TempDir, WALManager) {
        let dir = tempfile::tempdir().expect("tempdir");
        // WALManager::new returns a running manager (running flag = true
        // on construction). No separate start() call needed.
        let wal = WALManager::new(dir.path(), SyncMode::Full).expect("wal manager");
        (dir, wal)
    }

    #[test]
    fn empty_store_has_zero_lsn() {
        let s = OverlayStore::new();
        assert_eq!(s.last_applied_lsn(), 0);
        assert_eq!(s.next_entry_floor(), 0);
        assert!(s.pending_ddl().is_empty());
    }

    #[test]
    fn rebuild_with_to_lsn_zero_resets_cursors() {
        let s = OverlayStore::new();
        // Pretend we had advanced.
        s.last_applied_lsn.store(42, Ordering::Release);
        s.next_entry_floor.store(42, Ordering::Release);

        let (_dir, wal) = tmp_wal();
        s.rebuild_from_wal(&wal, 0, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        assert_eq!(s.last_applied_lsn(), 0, "to_lsn=0 must reset");
        assert_eq!(s.next_entry_floor(), 0);
    }

    #[test]
    fn rebuild_advances_lsn_for_committed_entries() {
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
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();

        assert_eq!(s.last_applied_lsn(), commit_lsn);
        // No DDL was committed, so pending_ddl stays empty.
        assert!(s.pending_ddl().is_empty());
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
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        let lsn_after_first = s.last_applied_lsn();
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        let lsn_after_second = s.last_applied_lsn();
        assert_eq!(
            lsn_after_first, lsn_after_second,
            "rebuild with the same to_lsn must produce the same cursor state"
        );
    }

    #[test]
    fn rebuild_collects_ddl_into_pending_ddl_list() {
        // A CREATE TABLE entry committed in the tail window must
        // surface in pending_ddl so the reader can raise
        // SwmrPendingDdl.
        let (_dir, wal) = tmp_wal();

        // CreateTable for "new_table". Schema payload is opaque to
        // the cursor (it would normally be the serialized Schema).
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
        s.rebuild_from_wal(&wal, commit_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();

        let ddl = s.pending_ddl();
        assert_eq!(ddl.len(), 1, "exactly one DDL event");
        assert_eq!(ddl[0].table_name, "new_table");
        assert!(matches!(ddl[0].operation, WALOperationType::CreateTable));
        assert!(ddl[0].lsn > 0);
    }

    #[test]
    fn rebuild_clears_pending_ddl_on_subsequent_rebuild_with_no_new_ddl() {
        // pending_ddl reflects ONLY events in the most recent
        // rebuild window. A second rebuild at the same to_lsn is a
        // true no-op AND explicitly clears pending_ddl from the
        // prior call, otherwise refresh() would re-raise
        // SwmrPendingDdl on every query without any new DDL.
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
        let final_lsn = wal.write_commit_marker(0, 0).unwrap();
        wal.flush().unwrap();

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, final_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        assert_eq!(s.pending_ddl().len(), 1, "first rebuild sees the DDL");

        // Second rebuild at the SAME to_lsn: incremental tail has no
        // new entries, so no new DDL is observed. pending_ddl is
        // explicitly cleared so refresh() doesn't re-fire.
        s.rebuild_from_wal(&wal, final_lsn, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        assert!(
            s.pending_ddl().is_empty(),
            "no-op rebuild must clear stale pending_ddl"
        );
    }

    #[test]
    fn rebuild_advances_entry_floor_when_no_active_user_txns() {
        // When the writer has no in-flight user txns at refresh
        // time, the watermark is u64::MAX, and the cursor's
        // `next_entry_floor` advances to `to_lsn + 1`. The NEXT
        // rebuild then floors its DDL scan at `to_lsn + 1`, skipping
        // already-applied entries.
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
        // active_txn_first_lsn, that's deferred to the engine's
        // publish_visible_commit_lsn so concurrent publishes see a
        // consistent snapshot. Simulate the engine's
        // post-publish clear here.
        wal.clear_active_txn(1);

        // After the clear, the writer's active map for user txns
        // is empty -> oldest_active_txn_lsn == u64::MAX.
        assert_eq!(wal.oldest_active_txn_lsn(), u64::MAX);

        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_after_1, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
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
    }

    #[test]
    fn rebuild_uses_explicit_watermark_not_local_wal_cache() {
        // In cross-process SWMR, the reader's WAL
        // manager has an EMPTY active-txn map (the reader doesn't
        // write DML), so `wal.oldest_active_txn_lsn()` returns
        // u64::MAX. The watermark must come from
        // `db.shm.oldest_active_txn_lsn` published by the WRITER
        // process. Simulate this by passing the writer's
        // actual oldest-active LSN as the parameter.
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
        wal.flush().unwrap();

        let s = OverlayStore::new();
        // Reader refresh, explicitly pass writer's
        // watermark = L1 (txn 1 still active in writer).
        s.rebuild_from_wal(&wal, lsn_a, l1, u64::MAX).unwrap();
        // The next_entry_floor must be min(l1, lsn_a) = l1.
        assert_eq!(
            s.next_entry_floor.load(Ordering::Acquire),
            l1,
            "next floor must use the EXPLICIT watermark (l1), not the \
             local WAL cache (which would be u64::MAX in cross-process)"
        );
    }

    #[test]
    fn rebuild_keeps_low_entry_floor_while_long_running_txn_active() {
        // While a long-running explicit txn is in-flight, the
        // watermark stays low (its first DML LSN), so subsequent
        // refreshes scan from that low floor, which is exactly
        // what's needed to catch the txn's pre-window DDL when it
        // eventually commits.
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

        // Reader rebuilds, txn 1 still active, watermark = l1.
        let s = OverlayStore::new();
        s.rebuild_from_wal(&wal, lsn_a, wal.oldest_active_txn_lsn(), u64::MAX)
            .unwrap();
        // next_entry_floor must equal min(l1, lsn_a) = l1 because
        // l1 < lsn_a. This keeps the floor LOW so the next refresh
        // can still find txn 1's DDL when it commits.
        assert_eq!(
            s.next_entry_floor.load(Ordering::Acquire),
            l1,
            "next_entry_floor must stay at the long-running txn's first DML LSN"
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
        s.rebuild_from_wal(&wal, lsn_after_t1, wal.oldest_active_txn_lsn(), u64::MAX)
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

        s.rebuild_from_wal(&wal, lsn_after_t2, wal.oldest_active_txn_lsn(), u64::MAX)
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
}
