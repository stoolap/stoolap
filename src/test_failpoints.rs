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

/// Reset all failpoints to disabled state
pub fn reset_all() {
    use std::sync::atomic::Ordering::Release;
    WAL_WRITE_FAIL.store(false, Release);
    WAL_SYNC_FAIL.store(false, Release);
    WAL_SCAN_READ_FAIL.store(false, Release);
    SNAPSHOT_WRITE_FAIL.store(false, Release);
    SNAPSHOT_SYNC_FAIL.store(false, Release);
    SNAPSHOT_RENAME_FAIL.store(false, Release);
    CHECKPOINT_WRITE_FAIL.store(false, Release);
    RETIRED_WAL_SYNC_FAIL.store(false, Release);
    VERSION_ROOT_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SYNC_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SWAP_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_SWAPPED_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETIRED_SETTLING_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETIRED_AWAITED_HOOK.with(|slot| *slot.borrow_mut() = None);
    WAL_DIRECTORY_SYNC_HOOK.with(|slot| *slot.borrow_mut() = None);
    INDEXES_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
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
