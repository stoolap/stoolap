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

//! Test failpoints for I/O errors and deterministic reader/writer rendezvous.
//! Flags and hooks are enabled by unit tests or the `test-failpoints` feature.
//! Normal builds without that feature compile out the failpoints and their calls.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard};

/// Fail WAL `write_to_file()` with an I/O error
pub static WAL_WRITE_FAIL: AtomicBool = AtomicBool::new(false);

/// Write the WAL buffer but its last 5 bytes, then fail the write
pub static WAL_WRITE_PARTIAL: AtomicBool = AtomicBool::new(false);

/// Fail the cut of a WAL file back to a length after a failure
pub static WAL_CUT_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail WAL `sync_locked()` (fsync) with an I/O error
pub static WAL_SYNC_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail the search for the next WAL record past a damaged one with an I/O error
pub static WAL_SCAN_READ_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail snapshot `append_row()` write with an I/O error
pub static SNAPSHOT_WRITE_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail snapshot `finalize()` sync with an I/O error
pub static SNAPSHOT_SYNC_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail atomic rename in `create_snapshot()` phase
pub static SNAPSHOT_RENAME_FAIL: AtomicBool = AtomicBool::new(false);

/// Fail checkpoint metadata write
pub static CHECKPOINT_WRITE_FAIL: AtomicBool = AtomicBool::new(false);

/// The sync of a WAL file retired by a file start fails.
pub static RETIRED_WAL_SYNC_FAIL: AtomicBool = AtomicBool::new(false);

/// The next this many seal registrations find their preparation stale
pub static SEAL_REGISTRATION_STALE_ROUNDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// fetch_update is deprecated in 1.99; try_update is stable only from 1.95, above the 1.88 MSRV
#[allow(deprecated)]
pub(crate) fn seal_registration_forced_stale() -> bool {
    SEAL_REGISTRATION_STALE_ROUNDS
        .fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |left| left.checked_sub(1),
        )
        .is_ok()
}

/// Serializes failpoint tests so that only one can run at a time.
/// Global AtomicBool flags are process-wide; concurrent tests would
/// interfere with each other without this lock.
static FAILPOINT_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static VERSION_ROOT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread immediately after its next version root is copied.
pub fn after_version_root(hook: impl FnOnce() + 'static) {
    VERSION_ROOT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn version_root_captured() {
    let hook = VERSION_ROOT_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static TRIM_LOCK_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread after a history trim picks its rows, before it locks the versions.
pub fn before_trim_lock(hook: impl FnOnce() + 'static) {
    TRIM_LOCK_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn trim_lock_starting() {
    let hook = TRIM_LOCK_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static TRANSACTION_BEGUN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next transaction has its begin sequence
/// in the registry, before the transaction is built.
pub fn after_transaction_begun(hook: impl FnOnce() + 'static) {
    TRANSACTION_BEGUN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn transaction_begun() {
    let hook = TRANSACTION_BEGUN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COMMIT_VISIBLE_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next commit has published its versions,
/// before it makes them visible.
pub fn before_commit_visible(hook: impl FnOnce() + 'static) {
    COMMIT_VISIBLE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_becoming_visible() {
    let hook = COMMIT_VISIBLE_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static WAL_SYNC_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread right before its next WAL fsync takes the file lock.
pub fn before_wal_sync(hook: impl FnOnce() + 'static) {
    WAL_SYNC_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn wal_sync_starting() {
    let hook = WAL_SYNC_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static WAL_SWAP_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next WAL file start has the new file
/// ready, before the swap takes the locks.
pub fn before_wal_swap(hook: impl FnOnce() + 'static) {
    WAL_SWAP_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn wal_swap_starting() {
    let hook = WAL_SWAP_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static MAINTENANCE_SCHEMA_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once after seal or compaction captures a table schema on this thread.
pub fn after_maintenance_schema_taken(hook: impl FnOnce() + 'static) {
    MAINTENANCE_SCHEMA_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn maintenance_schema_taken() {
    let hook = MAINTENANCE_SCHEMA_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static SEAL_CUTOFF_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once after a seal on this thread read the oldest snapshot for its
/// cutoff, before it captures the version root.
pub fn after_seal_cutoff_read(hook: impl FnOnce() + 'static) {
    SEAL_CUTOFF_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_cutoff_read() {
    let hook = SEAL_CUTOFF_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

type FlagHook = Box<dyn FnOnce(bool)>;

thread_local! {
    static COMPACTION_TOMBSTONES_HOOK: RefCell<Option<FlagHook>> = RefCell::new(None);
}

/// Run once after a compaction on this thread took the tombstones it
/// applies, with whether the DDL guard was held at that point.
pub fn after_compaction_tombstones_taken(hook: impl FnOnce(bool) + 'static) {
    COMPACTION_TOMBSTONES_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compaction_tombstones_taken(ddl_held: bool) {
    let hook = COMPACTION_TOMBSTONES_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook(ddl_held);
    }
}

type CountHook = Box<dyn FnOnce(usize)>;

thread_local! {
    static COMPACTION_DEDUP_HOOK: RefCell<Option<CountHook>> = RefCell::new(None);
}

/// Run once after a compaction on this thread decided which rows it keeps,
/// before it writes any output, with the number of tombstone pairs it took.
pub fn after_compaction_dedup(hook: impl FnOnce(usize) + 'static) {
    COMPACTION_DEDUP_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compaction_deduped(applied_pairs: usize) {
    let hook = COMPACTION_DEDUP_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook(applied_pairs);
    }
}

thread_local! {
    static SIDE_FILES_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once after a seal or compaction on this thread built its side
/// files, before it takes the DDL guard to compare them with the catalog.
pub fn after_side_files_built(hook: impl FnOnce() + 'static) {
    SIDE_FILES_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn side_files_built() {
    let hook = SIDE_FILES_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static MANIFEST_CAPTURED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next manifest persist has captured the
/// manifest and tombstones, before it writes them.
pub fn after_manifest_captured(hook: impl FnOnce() + 'static) {
    MANIFEST_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn manifest_captured() {
    let hook = MANIFEST_CAPTURED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static SIDE_FILES_COMPARED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once after a seal or compaction on this thread compared its side
/// files' identities with the catalog under the DDL guard, before it
/// publishes: DDL run from the hook on another thread waits for the guard.
pub fn after_side_files_compared(hook: impl FnOnce() + 'static) {
    SIDE_FILES_COMPARED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn side_files_compared() {
    let hook = SIDE_FILES_COMPARED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static BACKFILL_VOLUME_LOADED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread after a backfill loaded a volume and before it
/// builds the side file, so a test can take the volume away first.
pub fn after_backfill_volume_loaded(hook: impl FnOnce() + 'static) {
    BACKFILL_VOLUME_LOADED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn backfill_volume_loaded() {
    let hook = BACKFILL_VOLUME_LOADED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static SIDE_BACKFILLED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread after a backfill built a volume's side file and
/// before it is published, so a test can change the index, compact the
/// volume, or abandon the process there.
pub fn after_side_backfilled(hook: impl FnOnce() + 'static) {
    SIDE_BACKFILLED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn side_backfilled() {
    let hook = SIDE_BACKFILLED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COLD_VOLUMES_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread immediately after its next cold volume list is
/// taken, so a test can compact or retire a segment inside that reader's own
/// window.
pub fn after_cold_volumes_taken(hook: impl FnOnce() + 'static) {
    COLD_VOLUMES_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_volumes_taken() {
    let hook = COLD_VOLUMES_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static HOT_ROWS_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next unordered LIMIT read has taken its
/// hot rows and not yet its cold volume list, so a test can seal inside that
/// window.
pub fn after_hot_rows_taken(hook: impl FnOnce() + 'static) {
    HOT_ROWS_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn hot_rows_taken() {
    let hook = HOT_ROWS_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static MERGED_READ_BEGAN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next read that merges the hot rows of
/// a sealed table with its volumes is about to read the hot rows, so a
/// test can hold a seal inside the window that follows
pub fn after_merged_read_began(hook: impl FnOnce() + 'static) {
    MERGED_READ_BEGAN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn merged_read_began() {
    let hook = MERGED_READ_BEGAN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static MERGED_READ_COLLECTED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next merged read has taken its hot rows
/// and its cold view and not yet returned them
pub fn after_merged_read_collected(hook: impl FnOnce() + 'static) {
    MERGED_READ_COLLECTED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn merged_read_collected() {
    let hook = MERGED_READ_COLLECTED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static SEAL_ROWS_REMOVED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next seal, under the table's fence, has
/// removed the hot rows it sealed and not yet tombstoned the ones it skipped
pub fn in_seal_after_rows_removed(hook: impl FnOnce() + 'static) {
    SEAL_ROWS_REMOVED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_rows_removed() {
    let hook = SEAL_ROWS_REMOVED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static SEAL_INDEXES_CLEANED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next seal, under the table's fence, has
/// cleaned the hot indexes and not yet cleared the tombstones of the rows
/// it sealed
pub fn in_seal_after_indexes_cleaned(hook: impl FnOnce() + 'static) {
    SEAL_INDEXES_CLEANED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_indexes_cleaned() {
    let hook = SEAL_INDEXES_CLEANED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static JOIN_PROBE_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next join probe has passed the checks
/// that admit it to the hot index, before the index is read: the window a
/// seal, a commit or a truncate must not slip into unnoticed
pub fn after_join_probe_admitted(hook: impl FnOnce() + 'static) {
    JOIN_PROBE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn join_probe_admitted() {
    let hook = JOIN_PROBE_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Process-wide: indexes filled from the sealed rows' volumes
static INDEXES_FILLED_FROM_COLD: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Indexes filled from the sealed rows' volumes, so far
pub fn indexes_filled_from_cold() -> usize {
    INDEXES_FILLED_FROM_COLD.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn index_filled_from_cold() {
    INDEXES_FILLED_FROM_COLD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide: IN subqueries run to take their members
static IN_SUBQUERY_RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// IN subqueries run to take their members, so far
pub fn in_subquery_runs() -> usize {
    IN_SUBQUERY_RUNS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn in_subquery_ran() {
    IN_SUBQUERY_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide: EXISTS subqueries run whole for one outer row
static EXISTS_SUBQUERY_RUNS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// EXISTS subqueries run whole for one outer row, so far
pub fn exists_subquery_runs() -> usize {
    EXISTS_SUBQUERY_RUNS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn exists_subquery_ran() {
    EXISTS_SUBQUERY_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide: IN members read by a table's candidates, and by a scan
static IN_MEMBER_READS: [std::sync::atomic::AtomicUsize; 2] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

/// IN member reads by candidates and by a scan of the members, so far
pub fn in_member_reads() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        IN_MEMBER_READS[0].load(Relaxed),
        IN_MEMBER_READS[1].load(Relaxed),
    )
}

pub(crate) fn in_members_read(by_scan: bool) {
    IN_MEMBER_READS[by_scan as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide: IN members asked of a table's candidates
static IN_MEMBER_PROBES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// IN members asked of a table's candidates, so far
pub fn in_member_probes() -> usize {
    IN_MEMBER_PROBES.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn in_members_probed() {
    IN_MEMBER_PROBES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    static EQUALITY_KEY_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next multi-key equality probe has
/// taken one key's ids and not yet the next key's
pub fn after_equality_key_probed(hook: impl FnOnce() + 'static) {
    EQUALITY_KEY_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn equality_key_probed() {
    let hook = EQUALITY_KEY_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static ROW_IDS_CLASSIFIED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next fetch of rows by id on a table
/// holding volumes has decided which ids are hot and has not yet read them:
/// the window a seal must not slip into unnoticed
pub fn after_row_ids_classified(hook: impl FnOnce() + 'static) {
    ROW_IDS_CLASSIFIED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn row_ids_classified() {
    let hook = ROW_IDS_CLASSIFIED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static DELETE_ROWS_SCANNED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next DELETE that scans its rows first
/// has them and has not yet deleted them: the window another transaction's
/// commit slips into
pub fn after_delete_rows_scanned(hook: impl FnOnce() + 'static) {
    DELETE_ROWS_SCANNED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn delete_rows_scanned() {
    let hook = DELETE_ROWS_SCANNED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static RENAME_MOVED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next table rename has moved the
/// directory, before the table takes its new name: the window in which the
/// old name and the new directory disagree
pub fn in_rename_after_the_move(hook: impl FnOnce() + 'static) {
    RENAME_MOVED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn rename_directory_moved() {
    let hook = RENAME_MOVED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static VOLUME_PATH_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once after resolving a volume path, before acquiring its file owner.
pub fn after_volume_file_path(hook: impl FnOnce() + 'static) {
    VOLUME_PATH_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn volume_file_path_resolved() {
    let hook = VOLUME_PATH_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static WAL_SWAPPED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

thread_local! {
    static FAILURE_CLEANUP_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next WAL failure cleanup has the file
/// lock and is about to take the retired file
pub fn before_wal_failure_cleanup(hook: impl FnOnce() + 'static) {
    FAILURE_CLEANUP_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn wal_failure_cleanup_starting() {
    let hook = FAILURE_CLEANUP_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static RECORD_BUFFERED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next WAL record that flushes is in the
/// buffer and the buffer lock is released, before its own flush
pub fn after_record_buffered(hook: impl FnOnce() + 'static) {
    RECORD_BUFFERED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn record_buffered() {
    let hook = RECORD_BUFFERED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread right after its next WAL file swap releases the
/// locks, before the retired file is settled.
pub fn after_wal_swap(hook: impl FnOnce() + 'static) {
    WAL_SWAPPED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn wal_swapped() {
    let hook = WAL_SWAPPED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static RETIRED_SETTLING_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static RETIRED_AWAITED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next settlement of a retired WAL file
/// has taken the debt, before its syncs.
pub fn before_retired_settle(hook: impl FnOnce() + 'static) {
    RETIRED_SETTLING_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn retired_settling() {
    let hook = RETIRED_SETTLING_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next WAL sync, under the file lock
/// with its records drained, is about to wait for a retired file.
pub fn before_retired_wait(hook: impl FnOnce() + 'static) {
    RETIRED_AWAITED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn retired_awaited() {
    let hook = RETIRED_AWAITED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static WAL_DIRECTORY_SYNC_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when it next syncs the WAL directory.
pub fn on_wal_directory_sync(hook: impl FnOnce() + 'static) {
    WAL_DIRECTORY_SYNC_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn wal_directory_syncing() {
    let hook = WAL_DIRECTORY_SYNC_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static UNIQUE_FOUND_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMMIT_CAPTURE_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next unique check has looked a value
/// up in a unique index and has not yet read the transaction's own rows
pub fn after_unique_index_found(hook: impl FnOnce() + 'static) {
    UNIQUE_FOUND_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn unique_index_found() {
    let hook = UNIQUE_FOUND_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next table commit holds the
/// transaction's local store and has not yet taken the index set
pub fn before_commit_index_capture(hook: impl FnOnce() + 'static) {
    COMMIT_CAPTURE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_index_capture_next() {
    let hook = COMMIT_CAPTURE_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COMMIT_MARKER_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next commit has published its tables
/// and has not yet written its commit marker
pub fn before_commit_marker(hook: impl FnOnce() + 'static) {
    COMMIT_MARKER_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_marker_next() {
    let hook = COMMIT_MARKER_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static DROP_HOT_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next DROP COLUMN has published the hot
/// rows without the column and has not yet moved the cold mappings
pub fn after_drop_hot_published(hook: impl FnOnce() + 'static) {
    DROP_HOT_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn drop_hot_published() {
    let hook = DROP_HOT_PUBLISHED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static DROP_RECORDED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next DROP COLUMN has recorded the drop
/// in the log and has not yet published the table without the column
pub fn after_drop_column_recorded(hook: impl FnOnce() + 'static) {
    DROP_RECORDED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn drop_column_recorded() {
    let hook = DROP_RECORDED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static INDEXES_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next commit has updated the shared
/// indexes and has not yet made its versions visible.
pub fn after_indexes_published(hook: impl FnOnce() + 'static) {
    INDEXES_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn indexes_published() {
    let hook = INDEXES_PUBLISHED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static TXN_STORE_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next table open has published a new
/// transaction-local store and has not yet taken its handle's schema
pub fn after_txn_store_published(hook: impl FnOnce() + 'static) {
    TXN_STORE_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn txn_store_published() {
    let hook = TXN_STORE_PUBLISHED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COMPILE_SCHEMA_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next PK fast path compile has taken
/// the schema it resolves positions from
pub fn after_compile_schema_read(hook: impl FnOnce() + 'static) {
    COMPILE_SCHEMA_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compile_schema_read() {
    let hook = COMPILE_SCHEMA_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static JOIN_INNER_OPENED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next index join has opened its inner
/// table and has not yet looked at its compiled residual program
pub fn after_join_inner_opened(hook: impl FnOnce() + 'static) {
    JOIN_INNER_OPENED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn join_inner_opened() {
    let hook = JOIN_INNER_OPENED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static DML_TABLE_OPENED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next INSERT, UPDATE or DELETE has
/// opened its table and has not yet resolved a column position
pub fn after_dml_table_opened(hook: impl FnOnce() + 'static) {
    DML_TABLE_OPENED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn dml_table_opened() {
    let hook = DML_TABLE_OPENED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static PK_HOT_FETCHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next engine fetch by row id has read
/// the hot rows and is about to read the rows it missed from cold
pub fn after_pk_hot_rows_fetched(hook: impl FnOnce() + 'static) {
    PK_HOT_FETCHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn pk_hot_rows_fetched() {
    let hook = PK_HOT_FETCHED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COMPILED_EPOCH_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next compiled PK statement has passed
/// its schema epoch check and has not yet opened its table
pub fn after_compiled_epoch_checked(hook: impl FnOnce() + 'static) {
    COMPILED_EPOCH_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compiled_epoch_checked() {
    let hook = COMPILED_EPOCH_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static VOLUME_LOAD_REQUESTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static VOLUME_FILE_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SEGMENT_MAP_CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ROW_ID_ORDER_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static COLD_MAP_CAPTURED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

thread_local! {
    static STATEMENT_CAPTURED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COLD_ROUND_PREPARED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// A statement takes over none of the cold volumes it read earlier, so
/// every read under the fence is deferred
pub static COLD_READS_FORGET: AtomicBool = AtomicBool::new(false);

/// Run once on this thread when its next statement snapshot is captured
/// and every guard of the capture is released
pub fn after_statement_captured(hook: impl FnOnce() + 'static) {
    STATEMENT_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn statement_captured() {
    let hook = STATEMENT_CAPTURED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when an UPDATE has prepared a round of cold
/// rows and not yet taken the seal fence
pub fn after_cold_round_prepared(hook: impl FnOnce() + 'static) {
    COLD_ROUND_PREPARED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_round_prepared() {
    let hook = COLD_ROUND_PREPARED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Segment maps this thread copied to publish reloaded volumes
pub fn segment_map_clones() -> usize {
    SEGMENT_MAP_CLONES.with(std::cell::Cell::get)
}

pub(crate) fn segment_map_cloned() {
    SEGMENT_MAP_CLONES.with(|n| n.set(n.get() + 1));
}

/// Process-wide: a cold read may run on another thread. Unique indexes
/// built, candidates a built unique index named for a read, blocks decoded
static UNIQUE_INDEX_BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static UNIQUE_READ_CANDIDATES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static BLOCK_DECODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Unique indexes built, read candidates named, blocks decoded, so far
pub fn unique_read_counts() -> (usize, usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        UNIQUE_INDEX_BUILDS.load(Relaxed),
        UNIQUE_READ_CANDIDATES.load(Relaxed),
        BLOCK_DECODES.load(Relaxed),
    )
}

pub(crate) fn unique_index_built() {
    UNIQUE_INDEX_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn unique_read_candidates(count: usize) {
    UNIQUE_READ_CANDIDATES.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn block_decoded() {
    BLOCK_DECODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    static UNIQUE_INDEX_TAKEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next read has taken a built unique
/// index for its candidates and not yet searched it
pub fn on_unique_index_taken(hook: impl FnOnce() + 'static) {
    UNIQUE_INDEX_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn unique_index_taken() {
    let hook = UNIQUE_INDEX_TAKEN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Volumes whose row ids this thread scanned to learn their order
pub fn row_id_order_scans() -> usize {
    ROW_ID_ORDER_SCANS.with(std::cell::Cell::get)
}

pub(crate) fn row_id_order_scanned() {
    ROW_ID_ORDER_SCANS.with(|n| n.set(n.get() + 1));
}

thread_local! {
    static SMALL_OUTER_JOINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static JOIN_SIDE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Joins this thread ran by a table's candidates for a small outer side
pub fn small_outer_joins() -> usize {
    SMALL_OUTER_JOINS.with(std::cell::Cell::get)
}

pub(crate) fn small_outer_joined() {
    SMALL_OUTER_JOINS.with(|n| n.set(n.get() + 1));
}

/// Sides this thread's two-table joins executed, so far
pub fn join_side_runs() -> usize {
    JOIN_SIDE_RUNS.with(std::cell::Cell::get)
}

pub(crate) fn join_side_ran() {
    JOIN_SIDE_RUNS.with(|n| n.set(n.get() + 1));
}

/// Run once on this thread when its next statement snapshot has read the
/// segment map and not yet the cold flag
pub fn on_cold_map_captured(hook: impl FnOnce() + 'static) {
    COLD_MAP_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_map_captured() {
    let hook = COLD_MAP_CAPTURED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Volume loads this thread asked for and the volume files it read for
/// them, since the last reset; a load the file owner's cache answers reads none
pub fn volume_loads() -> (usize, usize) {
    (
        VOLUME_LOAD_REQUESTS.with(std::cell::Cell::get),
        VOLUME_FILE_READS.with(std::cell::Cell::get),
    )
}

pub(crate) fn volume_load_requested() {
    VOLUME_LOAD_REQUESTS.with(|n| n.set(n.get() + 1));
}

pub(crate) fn volume_file_read() {
    VOLUME_FILE_READS.with(|n| n.set(n.get() + 1));
}

/// Reset all failpoints to disabled state
pub fn reset_all() {
    use std::sync::atomic::Ordering::Release;
    WAL_WRITE_FAIL.store(false, Release);
    WAL_WRITE_PARTIAL.store(false, Release);
    WAL_CUT_FAIL.store(false, Release);
    WAL_SYNC_FAIL.store(false, Release);
    WAL_SCAN_READ_FAIL.store(false, Release);
    SNAPSHOT_WRITE_FAIL.store(false, Release);
    SNAPSHOT_SYNC_FAIL.store(false, Release);
    SNAPSHOT_RENAME_FAIL.store(false, Release);
    CHECKPOINT_WRITE_FAIL.store(false, Release);
    RETIRED_WAL_SYNC_FAIL.store(false, Release);
    SEAL_REGISTRATION_STALE_ROUNDS.store(0, Release);
    VERSION_ROOT_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SYNC_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SWAP_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SWAPPED_HOOK.with(|slot| *slot.borrow_mut() = None);
    RECORD_BUFFERED_HOOK.with(|slot| *slot.borrow_mut() = None);
    FAILURE_CLEANUP_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETIRED_SETTLING_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETIRED_AWAITED_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_DIRECTORY_SYNC_HOOK.with(|slot| *slot.borrow_mut() = None);
    INDEXES_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
    VOLUME_LOAD_REQUESTS.with(|n| n.set(0));
    VOLUME_FILE_READS.with(|n| n.set(0));
    SEGMENT_MAP_CLONES.with(|n| n.set(0));
    COLD_MAP_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = None);
    UNIQUE_INDEX_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    STATEMENT_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_ROUND_PREPARED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_READS_FORGET.store(false, Release);
}

/// RAII guard that serializes failpoint tests and resets all failpoints on drop.
/// Acquires FAILPOINT_LOCK so only one test runs at a time, and ensures
/// cleanup even if a test panics after arming a failpoint.
pub struct FailpointGuard {
    _lock: MutexGuard<'static, ()>,
}

impl FailpointGuard {
    pub fn new() -> Self {
        // If a previous test panicked while holding the lock, the Mutex is
        // poisoned. Recover by accepting the poisoned guard; reset_all()
        // below will clean up the stale flags.
        let lock = FAILPOINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_all();
        FailpointGuard { _lock: lock }
    }
}

impl Default for FailpointGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FailpointGuard {
    fn drop(&mut self) {
        reset_all();
    }
}
