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

/// A table manifest's write fails.
pub static MANIFEST_WRITE_FAIL: AtomicBool = AtomicBool::new(false);

/// The n-th volume a column type change reads fails to read; zero leaves
/// the reads alone
pub static RETYPE_READS_BEFORE_FAIL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// fetch_update is deprecated in 1.99; try_update is stable only from 1.95, above the 1.88 MSRV
#[allow(deprecated)]
pub(crate) fn retype_read_fails() -> bool {
    use std::sync::atomic::Ordering::AcqRel;
    let mut fails = false;
    let _ = RETYPE_READS_BEFORE_FAIL.fetch_update(
        AcqRel,
        std::sync::atomic::Ordering::Acquire,
        |left| match left {
            0 => None,
            1 => {
                fails = true;
                Some(0)
            }
            n => Some(n - 1),
        },
    );
    fails
}

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
    static SEAL_VOLUMES_BUILT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMPACTION_OUTPUTS_BUILT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMPACTION_PLANNED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMPACTION_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static MANIFESTS_PERSISTED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static BETWEEN_COUNT_CAPTURES_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static SEGMENTS_CLEARING_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static TRUNCATE_COLD_CLEARED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static DESTRUCTION_WAIT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static HOT_COLD_READS_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static UNSEALED_READ_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static CLEAR_GENERATION_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static TRUNCATE_MARKED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COUNT_CAPTURED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static GC_ROWS_REMOVED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COLUMN_CHANGE_WAITING_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static CHECKPOINT_CUT_TAKEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static RETYPE_OUTPUTS_WRITTEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static RETYPE_MANIFEST_WRITTEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static ANALYZE_ROWS_COLLECTED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static SEAL_PASSED_CHANGING_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMMIT_COUNTED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

static STALE_PUBLICATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Run once on this thread when its next seal built its volumes, before it
/// takes the DDL guard to publish them
pub fn after_seal_volumes_built(hook: impl FnOnce() + 'static) {
    SEAL_VOLUMES_BUILT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_volumes_built() {
    let hook = SEAL_VOLUMES_BUILT_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next compaction built its outputs,
/// before it takes the DDL guard to publish them
pub fn after_compaction_outputs_built(hook: impl FnOnce() + 'static) {
    COMPACTION_OUTPUTS_BUILT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compaction_outputs_built() {
    let hook = COMPACTION_OUTPUTS_BUILT_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next compaction settled its batch,
/// before it loads the members and reads the tombstones it applies
pub fn after_compaction_planned(hook: impl FnOnce() + 'static) {
    COMPACTION_PLANNED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compaction_planned() {
    let hook = COMPACTION_PLANNED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next checkpoint wrote the manifests,
/// before it truncates the log and compacts
pub fn after_checkpoint_manifests_persisted(hook: impl FnOnce() + 'static) {
    MANIFESTS_PERSISTED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn checkpoint_manifests_persisted() {
    let hook = MANIFESTS_PERSISTED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next compaction published its
/// outputs, before it removes the tombstones it applied
pub fn after_compaction_published(hook: impl FnOnce() + 'static) {
    COMPACTION_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn compaction_published() {
    let hook = COMPACTION_PUBLISHED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next live row recount captured both
/// maps it counts, before it counts them
pub fn after_count_captured(hook: impl FnOnce() + 'static) {
    COUNT_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn count_captured() {
    let hook = COUNT_CAPTURED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next segment set clearing emptied
/// everything and has not moved the generation yet
pub fn before_clear_generation_moves(hook: impl FnOnce() + 'static) {
    CLEAR_GENERATION_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn clear_before_generation() {
    let hook = CLEAR_GENERATION_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next hot-only read ran, before it
/// checks that no truncate moved under it
pub fn after_unsealed_read(hook: impl FnOnce() + 'static) {
    UNSEALED_READ_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn unsealed_read() {
    let hook = UNSEALED_READ_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next truncate marked rows as leaving,
/// before anything decides whether it may go ahead
pub fn after_truncate_marked(hook: impl FnOnce() + 'static) {
    TRUNCATE_MARKED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn truncate_marked() {
    let hook = TRUNCATE_MARKED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next read that takes the hot and the
/// sealed rows in two steps has taken the first of them
pub fn between_hot_and_cold_reads(hook: impl FnOnce() + 'static) {
    HOT_COLD_READS_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn hot_and_cold_reads_split() {
    let hook = HOT_COLD_READS_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next read waits for a truncate that
/// is taking rows away
pub fn on_destruction_wait(hook: impl FnOnce() + 'static) {
    DESTRUCTION_WAIT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn destruction_waited() {
    let hook = DESTRUCTION_WAIT_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next truncate cleared the volumes and
/// not yet the hot rows, with the hot store's claims still shut out
pub fn after_truncate_cold_cleared(hook: impl FnOnce() + 'static) {
    TRUNCATE_COLD_CLEARED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn truncate_cold_cleared() {
    let hook = TRUNCATE_COLD_CLEARED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next segment set clearing is halfway:
/// one of its segments and its tombstones is emptied, the other not yet
pub fn during_segments_clearing(hook: impl FnOnce() + 'static) {
    SEGMENTS_CLEARING_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn segments_clearing() {
    let hook = SEGMENTS_CLEARING_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next live row recount took the first of
/// the two maps it counts and not yet the second
pub fn between_count_captures(hook: impl FnOnce() + 'static) {
    BETWEEN_COUNT_CAPTURES_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn between_count_captures_reached() {
    let hook = BETWEEN_COUNT_CAPTURES_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next deleted-row cleanup let go of the
/// versions lock, before it cleans the indexes and the arena
pub fn after_gc_rows_removed(hook: impl FnOnce() + 'static) {
    GC_ROWS_REMOVED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn gc_rows_removed() {
    let hook = GC_ROWS_REMOVED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next checkpoint took the cut it will
/// publish, with the seal fence let go
pub fn after_checkpoint_cut_taken(hook: impl FnOnce() + 'static) {
    CHECKPOINT_CUT_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn checkpoint_cut_taken() {
    let hook = CHECKPOINT_CUT_TAKEN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next column type change wrote the
/// volumes it rewrote, before anything registers them
pub fn after_retype_outputs_written(hook: impl FnOnce() + 'static) {
    RETYPE_OUTPUTS_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn retype_outputs_written() {
    let hook = RETYPE_OUTPUTS_WRITTEN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next column type change wrote its
/// manifest, before the replaced volumes are retired
pub fn after_retype_manifest_written(hook: impl FnOnce() + 'static) {
    RETYPE_MANIFEST_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn retype_manifest_written() {
    let hook = RETYPE_MANIFEST_WRITTEN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next ANALYZE collected a table's rows,
/// before it builds the zone maps
pub fn after_analyze_rows_collected(hook: impl FnOnce() + 'static) {
    ANALYZE_ROWS_COLLECTED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn analyze_rows_collected() {
    let hook = ANALYZE_ROWS_COLLECTED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next column change closed the table and
/// starts waiting for the publications counted before
pub fn on_column_change_waiting(hook: impl FnOnce() + 'static) {
    COLUMN_CHANGE_WAITING_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn column_change_waiting() {
    let hook = COLUMN_CHANGE_WAITING_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next seal passes a table by because a
/// column change is in progress on it
pub fn on_seal_passed_changing_table(hook: impl FnOnce() + 'static) {
    SEAL_PASSED_CHANGING_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_passed_changing_table() {
    let hook = SEAL_PASSED_CHANGING_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next commit counted itself in the
/// tables it publishes into, before it applies any table's versions
pub fn after_commit_counted(hook: impl FnOnce() + 'static) {
    COMMIT_COUNTED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_counted() {
    let hook = COMMIT_COUNTED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Seal, compaction and side file outputs discarded at their publication
/// because a column change moved the schema since they were prepared
pub fn stale_publications() -> usize {
    STALE_PUBLICATIONS.load(std::sync::atomic::Ordering::SeqCst)
}

pub(crate) fn stale_publication_discarded() {
    STALE_PUBLICATIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

thread_local! {
    static KEY_ORDER_MERGES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ROW_ORDERS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static COUNT_IDS_WALKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static COUNT_BOUNDS_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Segment bounds compared on this thread to find a segment's id range
pub fn count_bounds_probes() -> usize {
    COUNT_BOUNDS_PROBES.with(std::cell::Cell::get)
}

pub(crate) fn count_bounds_probe() {
    COUNT_BOUNDS_PROBES.with(|c| c.set(c.get() + 1));
}

/// Cold row ids walked on this thread to recount a table's live rows
pub fn count_ids_walked() -> usize {
    COUNT_IDS_WALKED.with(std::cell::Cell::get)
}

pub(crate) fn count_ids_walk(n: usize) {
    COUNT_IDS_WALKED.with(|c| c.set(c.get() + n));
}

/// Permutations built on this thread for volumes whose ids do not ascend
pub fn row_orders_built() -> usize {
    ROW_ORDERS_BUILT.with(std::cell::Cell::get)
}

pub(crate) fn row_order_built() {
    ROW_ORDERS_BUILT.with(|n| n.set(n.get() + 1));
}

/// Compactions on this thread whose inputs merged in the clustering key's
/// order, without the sort
pub fn key_order_merges() -> usize {
    KEY_ORDER_MERGES.with(std::cell::Cell::get)
}

pub(crate) fn key_order_merged() {
    KEY_ORDER_MERGES.with(|n| n.set(n.get() + 1));
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

thread_local! {
    static IN_SUBQUERY_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// IN subqueries this thread ran to take their members, so far
pub fn in_subquery_runs() -> usize {
    IN_SUBQUERY_RUNS.with(std::cell::Cell::get)
}

pub(crate) fn in_subquery_ran() {
    IN_SUBQUERY_RUNS.with(|n| n.set(n.get() + 1));
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
    static SEGMENT_ROW_HOT_CHECKED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next lookup of a sealed row by id has
/// found no hot version deciding the row and has not yet read the volumes
pub fn after_segment_row_hot_checked(hook: impl FnOnce() + 'static) {
    SEGMENT_ROW_HOT_CHECKED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn segment_row_hot_checked() {
    run_hook(&SEGMENT_ROW_HOT_CHECKED_HOOK);
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
    static READ_PATHS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// Notes on this thread that the read entry `path` ran
pub(crate) fn note_read(path: &'static str) {
    READ_PATHS.with(|paths| {
        let mut paths = paths.borrow_mut();
        if !paths.contains(&path) {
            paths.push(path);
        }
    });
}

/// The read entries this thread ran since the last call, in first-run order
pub fn take_read_paths() -> Vec<&'static str> {
    READ_PATHS.with(|paths| std::mem::take(&mut *paths.borrow_mut()))
}

thread_local! {
    static FK_CHILD_PROBE_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static CORRELATED_FETCH_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static ADD_COLUMN_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static SNAPSHOT_SCHEMA_WRITTEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static JOIN_INDEX_CHOSEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static JOIN_NEXT_CHUNK_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static ALTER_DDL_GUARD_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static INDEX_BOUND_HOT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static CACHE_STAMP_PUBLISH_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static STATUS_RUN_CHECKED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static PROGRAM_STATEMENT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static SEMANTIC_CACHE_CLEARED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static STATEMENT_GATE_BLOCKED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next semantic cache clear has emptied
/// the cache and released its lock
pub fn after_semantic_cache_cleared(hook: impl FnOnce() + 'static) {
    SEMANTIC_CACHE_CLEARED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn semantic_cache_cleared() {
    run_hook(&SEMANTIC_CACHE_CLEARED_HOOK);
}

/// Run once on this thread when its next statement finds the statement gate
/// held against it and is about to wait for it
pub fn on_statement_gate_blocked(hook: impl FnOnce() + 'static) {
    STATEMENT_GATE_BLOCKED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn statement_gate_blocked() {
    run_hook(&STATEMENT_GATE_BLOCKED_HOOK);
}

/// Run once on this thread when its next swap of an open status slot has
/// checked the segment's run and not yet swapped
pub fn after_status_run_checked(hook: impl FnOnce() + 'static) {
    STATUS_RUN_CHECKED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn status_run_checked() {
    run_hook(&STATUS_RUN_CHECKED_HOOK);
}

/// Run once on this thread when a statement of its next multi-statement
/// program has finished and the next has not started
pub fn after_program_statement(hook: impl FnOnce() + 'static) {
    PROGRAM_STATEMENT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn program_statement_done() {
    run_hook(&PROGRAM_STATEMENT_HOOK);
}

/// Process-wide: scalar, IN and semi-join results served from their caches
static SUBQUERY_CACHE_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Subquery results served from the scalar, IN and semi-join caches, so far
pub fn subquery_cache_hits() -> usize {
    SUBQUERY_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn subquery_cache_hit() {
    SUBQUERY_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Run once on this thread when its next semantic or subquery cache stamp
/// has read the publish epochs and has not yet checked the schema epoch again
pub fn after_cache_stamp_publish_read(hook: impl FnOnce() + 'static) {
    CACHE_STAMP_PUBLISH_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cache_stamp_publish_read() {
    run_hook(&CACHE_STAMP_PUBLISH_HOOK);
}

/// Run once on this thread when its next index MIN or MAX over hot and
/// sealed rows has read the hot bound and is about to read the volumes
pub fn after_index_bound_hot(hook: impl FnOnce() + 'static) {
    INDEX_BOUND_HOT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn index_bound_hot_read() {
    run_hook(&INDEX_BOUND_HOT_HOOK);
}

/// Run once on this thread when its next ALTER TABLE is about to take the
/// DDL guard
pub fn before_alter_ddl_guard(hook: impl FnOnce() + 'static) {
    ALTER_DDL_GUARD_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn alter_ddl_guard_next() {
    run_hook(&ALTER_DDL_GUARD_HOOK);
}

type HookSlot = RefCell<Option<Box<dyn FnOnce()>>>;

fn run_hook(slot: &'static std::thread::LocalKey<HookSlot>) {
    let hook = slot.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next backup snapshot has written a
/// table's schema and has not yet taken its rows
pub fn after_snapshot_schema_written(hook: impl FnOnce() + 'static) {
    SNAPSHOT_SCHEMA_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn snapshot_schema_written() {
    run_hook(&SNAPSHOT_SCHEMA_WRITTEN_HOOK);
}

/// Run once on this thread when its next join plan has taken an inner
/// index from its planning handle, before the join opens a table to run on
pub fn after_join_index_chosen(hook: impl FnOnce() + 'static) {
    JOIN_INDEX_CHOSEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn join_index_chosen() {
    run_hook(&JOIN_INDEX_CHOSEN_HOOK);
}

/// Run once on this thread when its next streaming index join has joined
/// one outer chunk and is about to fetch the next one
pub fn before_join_next_chunk(hook: impl FnOnce() + 'static) {
    JOIN_NEXT_CHUNK_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn join_next_chunk() {
    run_hook(&JOIN_NEXT_CHUNK_HOOK);
}

/// Run once on this thread when its next foreign key check is about to
/// open a child table, its key positions already taken
pub fn before_fk_child_probe(hook: impl FnOnce() + 'static) {
    FK_CHILD_PROBE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn fk_child_probe_next() {
    run_hook(&FK_CHILD_PROBE_HOOK);
}

/// Run once on this thread when its next correlated EXISTS probe has
/// checked its token and has not yet fetched a row
pub fn before_correlated_fetch(hook: impl FnOnce() + 'static) {
    CORRELATED_FETCH_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn correlated_fetch_next() {
    run_hook(&CORRELATED_FETCH_HOOK);
}

/// Run once on this thread when its next ADD COLUMN has published the
/// schema and has not yet recorded the change or laid out the rows
pub fn after_add_column_published(hook: impl FnOnce() + 'static) {
    ADD_COLUMN_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn add_column_published() {
    run_hook(&ADD_COLUMN_PUBLISHED_HOOK);
}

thread_local! {
    static SELECT_TABLE_OPENED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next table scan SELECT has opened its
/// table and has not yet read a row
pub fn after_select_table_opened(hook: impl FnOnce() + 'static) {
    SELECT_TABLE_OPENED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn select_table_opened() {
    let hook = SELECT_TABLE_OPENED_HOOK.with(|slot| slot.borrow_mut().take());
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

thread_local! {
    static PK_LOOKUP_RESOLVED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next uncached PK lookup has resolved
/// positions from the schema and has not yet read the row
pub fn after_pk_lookup_resolved(hook: impl FnOnce() + 'static) {
    PK_LOOKUP_RESOLVED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn pk_lookup_resolved() {
    run_hook(&PK_LOOKUP_RESOLVED_HOOK);
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
    static TOP_K_COLD_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GROUP_CACHE_LOADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GROUP_COLUMN_DECODES: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
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
    COLD_ROUNDS.with(|n| n.set(n.get() + 1));
    let hook = COLD_ROUND_PREPARED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COLD_RECHECK_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its commit's cold recheck has read the
/// schema and the unique indexes, before its first lookup
pub fn on_cold_recheck(hook: impl FnOnce() + 'static) {
    COLD_RECHECK_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_recheck_started() {
    let hook = COLD_RECHECK_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// A hook and the number of batch rows written it runs at
type BatchRowHook = (usize, Box<dyn FnOnce()>);

thread_local! {
    static TABLE_WAL_RECORDED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static STATEMENT_ADMITTED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static BATCH_ROW_HOOK: RefCell<Option<BatchRowHook>> = RefCell::new(None);
    static HOT_PUBLISHED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static TXN_GATE_WAIT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its commit has recorded a table's versions
/// to the WAL, before the table's cold recheck and publication
pub fn after_table_wal_recorded(hook: impl FnOnce() + 'static) {
    TABLE_WAL_RECORDED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn table_wal_recorded() {
    run_hook(&TABLE_WAL_RECORDED_HOOK);
}

thread_local! {
    static INSERT_COLD_SNAPSHOT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next INSERT took the cold snapshot it
/// checks its keys against, before the check reads it
pub fn after_insert_cold_snapshot(hook: impl FnOnce() + 'static) {
    INSERT_COLD_SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn insert_cold_snapshot_taken() {
    run_hook(&INSERT_COLD_SNAPSHOT_HOOK);
}

pub(crate) fn table_wal_record_written() {
    TABLE_WAL_RECORDS.with(|n| n.set(n.get() + 1));
}

thread_local! {
    static TABLE_WAL_RECORDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LIVE_VOLUME_LOADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BLOCK_STORE_DROPS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
    static COMMIT_FENCE_HOOK: RefCell<Option<Box<dyn FnMut()>>> = RefCell::new(None);
    static SEAL_GENERATION_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COMMIT_ROUND_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static RECHECK_STORE_LOCKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static RECHECK_FENCE_SECTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static VOLUME_FILE_DROPS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// Tables this thread's commits recorded to the WAL
pub fn table_wal_records() -> usize {
    TABLE_WAL_RECORDS.with(std::cell::Cell::get)
}

/// Run on this thread each time its commit is about to take a table's seal
/// fence, until the failpoints are reset
pub fn on_commit_fence(hook: impl FnMut() + 'static) {
    COMMIT_FENCE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_fence_next() {
    let hook = COMMIT_FENCE_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(mut hook) = hook {
        hook();
        COMMIT_FENCE_HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                *slot = Some(hook);
            }
        });
    }
}

/// Run once on this thread when its commit dropped a cold recheck round
/// whose seal generation moved, before the next round begins
pub fn after_commit_round_moved(hook: impl FnOnce() + 'static) {
    COMMIT_ROUND_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn commit_round_moved() {
    run_hook(&COMMIT_ROUND_HOOK);
}

/// Run once on this thread when its seal has moved the seal generation and
/// not yet marked the table as having segments
pub fn after_seal_generation_moved(hook: impl FnOnce() + 'static) {
    SEAL_GENERATION_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn seal_generation_moved() {
    run_hook(&SEAL_GENERATION_HOOK);
}

/// Whether this thread's commit holds a transaction store for its cold
/// recheck; read by a test's counting allocator
pub fn in_recheck_store_lock() -> bool {
    RECHECK_STORE_LOCKED
        .try_with(std::cell::Cell::get)
        .unwrap_or(false)
}

/// Marks this thread as holding a transaction store for its cold recheck
pub(crate) struct RecheckStoreLock;

impl RecheckStoreLock {
    pub(crate) fn enter() -> Self {
        RECHECK_STORE_LOCKED.with(|l| l.set(true));
        Self
    }
}

impl Drop for RecheckStoreLock {
    fn drop(&mut self) {
        RECHECK_STORE_LOCKED.with(|l| l.set(false));
    }
}

/// Whether this thread is deciding a commit's cold recheck round or an
/// INSERT's cold check under the seal fence; read by a test's counting
/// allocator
pub fn in_recheck_fence_section() -> bool {
    RECHECK_FENCE_SECTION
        .try_with(std::cell::Cell::get)
        .unwrap_or(false)
}

/// Marks this thread as deciding a cold recheck or check round under the
/// seal fence
pub(crate) struct RecheckFenceSection;

impl RecheckFenceSection {
    pub(crate) fn enter() -> Self {
        RECHECK_FENCE_SECTION.with(|s| s.set(true));
        Self
    }
}

impl Drop for RecheckFenceSection {
    fn drop(&mut self) {
        RECHECK_FENCE_SECTION.with(|s| s.set(false));
    }
}

/// Volume file owners this thread let go last, (outside, under) the seal
/// fence: each takes the file registry lock and may remove a retired file
pub fn volume_file_drops_by_fence() -> (usize, usize) {
    VOLUME_FILE_DROPS.with(std::cell::Cell::get)
}

pub(crate) fn volume_file_dropped() {
    count_by_fence_depth(&VOLUME_FILE_DROPS);
}

/// Volumes this thread loaded through the live segment map, not through a
/// statement's captured owners
pub fn live_volume_loads() -> usize {
    LIVE_VOLUME_LOADS.with(std::cell::Cell::get)
}

pub(crate) fn live_volume_load() {
    LIVE_VOLUME_LOADS.with(|n| n.set(n.get() + 1));
}

/// Loaded volumes' block stores this thread let go last, (outside, under)
/// the seal fence: each takes the decoded group cache's lock
pub fn block_store_drops_by_fence() -> (usize, usize) {
    BLOCK_STORE_DROPS.with(std::cell::Cell::get)
}

pub(crate) fn block_store_dropped() {
    count_by_fence_depth(&BLOCK_STORE_DROPS);
}

/// Run once on this thread when its next write statement on a table was
/// admitted for its transaction, before it reads or writes anything
pub fn after_statement_admitted(hook: impl FnOnce() + 'static) {
    STATEMENT_ADMITTED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn statement_admitted() {
    run_hook(&STATEMENT_ADMITTED_HOOK);
}

/// Run once on this thread when a batch insert has written `rows` rows to
/// the hot store
pub fn after_batch_rows(rows: usize, hook: impl FnOnce() + 'static) {
    BATCH_ROW_HOOK.with(|slot| *slot.borrow_mut() = Some((rows, Box::new(hook))));
}

pub(crate) fn batch_rows_written(rows: usize) {
    let hook = BATCH_ROW_HOOK.with(|slot| {
        let mut slot = slot.borrow_mut();
        match slot.as_ref() {
            Some((at, _)) if *at == rows => slot.take().map(|(_, hook)| hook),
            _ => None,
        }
    });
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when an update statement has published its hot
/// versions, before its pending tombstones and its seal stamp
pub fn after_hot_publish(hook: impl FnOnce() + 'static) {
    HOT_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn hot_published() {
    run_hook(&HOT_PUBLISHED_HOOK);
}

/// Run once on this thread when its commit, rollback, drop or savepoint
/// starts waiting for the transaction's statements in flight
pub fn on_txn_gate_wait(hook: impl FnOnce() + 'static) {
    TXN_GATE_WAIT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn txn_gate_waiting() {
    run_hook(&TXN_GATE_WAIT_HOOK);
}

thread_local! {
    static TXN_GATE_HELD_OFF_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when a statement it starts finds the
/// transaction's new statements held off by a savepoint
pub fn on_txn_gate_held_off(hook: impl FnOnce() + 'static) {
    TXN_GATE_HELD_OFF_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn txn_gate_held_off() {
    run_hook(&TXN_GATE_HELD_OFF_HOOK);
}

thread_local! {
    static TXN_GATE_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TXN_GATE_ADMISSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The times this thread asked a transaction's gate to admit a statement
pub fn txn_gate_lookups() -> usize {
    TXN_GATE_LOOKUPS.with(std::cell::Cell::get)
}

pub(crate) fn txn_gate_looked_up() {
    TXN_GATE_LOOKUPS.with(|n| n.set(n.get() + 1));
}

/// The statements this thread was admitted for by a gate, nested ones not
/// counted
pub fn txn_gate_admissions() -> usize {
    TXN_GATE_ADMISSIONS.with(std::cell::Cell::get)
}

pub(crate) fn txn_gate_admitted() {
    TXN_GATE_ADMISSIONS.with(|n| n.set(n.get() + 1));
}

thread_local! {
    static COLD_ROUND_SETTLED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COLD_ROUND_SNAPSHOT_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when an UPDATE's cold round has read its
/// settled generation and not yet taken its snapshot
pub fn before_cold_round_snapshot(hook: impl FnOnce() + 'static) {
    COLD_ROUND_SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_round_snapshot_starting() {
    let hook = COLD_ROUND_SNAPSHOT_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when an UPDATE's cold round stood its checks
/// under the seal fence, before the rows are claimed
pub fn after_cold_round_settled(hook: impl FnOnce() + 'static) {
    COLD_ROUND_SETTLED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_round_settled() {
    let hook = COLD_ROUND_SETTLED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static COLD_ROUNDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SEAL_FENCE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BLOCK_FILE_READS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static VOLUME_FILE_READS_BY_DEPTH: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
    static SIDE_FILE_READS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// Cold rounds this thread's UPDATEs prepared outside the seal fence
pub fn cold_rounds() -> usize {
    COLD_ROUNDS.with(std::cell::Cell::get)
}

pub(crate) fn seal_fence_entered() {
    SEAL_FENCE_DEPTH.with(|d| d.set(d.get() + 1));
}

pub(crate) fn seal_fence_left() {
    SEAL_FENCE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
}

/// One more read in `counter`, as (outside, under) the seal fence
fn count_by_fence_depth(counter: &'static std::thread::LocalKey<std::cell::Cell<(usize, usize)>>) {
    let under = SEAL_FENCE_DEPTH.with(std::cell::Cell::get) > 0;
    counter.with(|c| {
        let (outside, inside) = c.get();
        c.set(if under {
            (outside, inside + 1)
        } else {
            (outside + 1, inside)
        });
    });
}

/// Row group blocks this thread read from volume files, (outside, under)
/// the seal fence
pub fn block_file_reads() -> (usize, usize) {
    BLOCK_FILE_READS.with(std::cell::Cell::get)
}

pub(crate) fn block_file_read() {
    count_by_fence_depth(&BLOCK_FILE_READS);
}

/// Volume files this thread read whole, (outside, under) the seal fence
pub fn volume_file_reads_by_fence() -> (usize, usize) {
    VOLUME_FILE_READS_BY_DEPTH.with(std::cell::Cell::get)
}

/// Side file reads this thread made, (outside, under) the seal fence
pub fn side_file_reads() -> (usize, usize) {
    SIDE_FILE_READS.with(std::cell::Cell::get)
}

pub(crate) fn side_file_read() {
    count_by_fence_depth(&SIDE_FILE_READS);
}

thread_local! {
    static COLD_CLAIM_TAKEN_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    static COLD_ONLY_CLAIMED_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

/// Run once on this thread when its next cold UPDATE has claimed a
/// prepared row and not yet claimed the next one
pub fn after_cold_claim_taken(hook: impl FnOnce() + 'static) {
    COLD_CLAIM_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_claim_taken() {
    let hook = COLD_CLAIM_TAKEN_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Run once on this thread when its next delete by id has claimed the
/// rows only a volume holds and not yet written their delete markers
pub fn after_cold_only_rows_claimed(hook: impl FnOnce() + 'static) {
    COLD_ONLY_CLAIMED_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn cold_only_rows_claimed() {
    let hook = COLD_ONLY_CLAIMED_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

thread_local! {
    static UNIQUE_PAIRS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Pairs of an UPDATE's planned rows this thread compared for one unique key
pub fn unique_pairs_compared() -> usize {
    UNIQUE_PAIRS.with(std::cell::Cell::get)
}

pub(crate) fn unique_pair_compared() {
    UNIQUE_PAIRS.with(|n| n.set(n.get() + 1));
}

thread_local! {
    static COLD_RECHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Commits on this thread that rechecked a table's keys against its volumes
pub fn cold_rechecks() -> usize {
    COLD_RECHECKS.with(std::cell::Cell::get)
}

pub(crate) fn cold_rechecked() {
    COLD_RECHECKS.with(|n| n.set(n.get() + 1));
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

/// Sealed rows this thread's ordered top-k scans built, so far
pub fn top_k_cold_rows() -> usize {
    TOP_K_COLD_ROWS.with(std::cell::Cell::get)
}

pub(crate) fn top_k_cold_row_built() {
    TOP_K_COLD_ROWS.with(|n| n.set(n.get() + 1));
}

/// Row groups whose needed columns this thread's volume scanners loaded
pub fn group_cache_loads() -> usize {
    GROUP_CACHE_LOADS.with(std::cell::Cell::get)
}

pub(crate) fn group_cache_loaded() {
    GROUP_CACHE_LOADS.with(|n| n.set(n.get() + 1));
}

/// The (column, row group) blocks this thread decoded since the last call,
/// in order; taking them clears the record
pub fn take_group_column_decodes() -> Vec<(usize, usize)> {
    GROUP_COLUMN_DECODES.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

pub(crate) fn group_column_decoded(col: usize, group: usize) {
    GROUP_COLUMN_DECODES.with(|d| d.borrow_mut().push((col, group)));
}

thread_local! {
    static SMALL_OUTER_JOINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static JOIN_SIDE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static IN_SUBQUERY_SETS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static STREAMING_JOINS_STREAMED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static INDEX_JOIN_OUTER_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Outer rows this thread's streaming index joins read, so far
pub fn index_join_outer_rows() -> usize {
    INDEX_JOIN_OUTER_ROWS.with(std::cell::Cell::get)
}

pub(crate) fn index_join_outer_row() {
    INDEX_JOIN_OUTER_ROWS.with(|n| n.set(n.get() + 1));
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

/// Shared sets this thread built from cached IN subquery lists, so far
pub fn in_subquery_sets_built() -> usize {
    IN_SUBQUERY_SETS_BUILT.with(std::cell::Cell::get)
}

pub(crate) fn in_subquery_set_built() {
    IN_SUBQUERY_SETS_BUILT.with(|n| n.set(n.get() + 1));
}

/// Streaming LIMIT joins this thread ran as they stream, without leaving
/// for the standard join, so far
pub fn streaming_joins_streamed() -> usize {
    STREAMING_JOINS_STREAMED.with(std::cell::Cell::get)
}

pub(crate) fn streaming_join_streamed() {
    STREAMING_JOINS_STREAMED.with(|n| n.set(n.get() + 1));
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
    count_by_fence_depth(&VOLUME_FILE_READS_BY_DEPTH);
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
    MANIFEST_WRITE_FAIL.store(false, Release);
    RETYPE_READS_BEFORE_FAIL.store(0, Release);
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
    KEY_ORDER_MERGES.with(|n| n.set(0));
    ROW_ORDERS_BUILT.with(|n| n.set(0));
    COUNT_IDS_WALKED.with(|n| n.set(0));
    COUNT_BOUNDS_PROBES.with(|n| n.set(0));
    VOLUME_FILE_READS.with(|n| n.set(0));
    SEGMENT_MAP_CLONES.with(|n| n.set(0));
    COLD_MAP_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = None);
    UNIQUE_INDEX_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    STATEMENT_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_ROUND_PREPARED_HOOK.with(|slot| *slot.borrow_mut() = None);
    SEAL_VOLUMES_BUILT_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMPACTION_OUTPUTS_BUILT_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMPACTION_PLANNED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMPACTION_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
    MANIFESTS_PERSISTED_HOOK.with(|slot| *slot.borrow_mut() = None);
    BETWEEN_COUNT_CAPTURES_HOOK.with(|slot| *slot.borrow_mut() = None);
    SEGMENTS_CLEARING_HOOK.with(|slot| *slot.borrow_mut() = None);
    TRUNCATE_COLD_CLEARED_HOOK.with(|slot| *slot.borrow_mut() = None);
    DESTRUCTION_WAIT_HOOK.with(|slot| *slot.borrow_mut() = None);
    HOT_COLD_READS_HOOK.with(|slot| *slot.borrow_mut() = None);
    UNSEALED_READ_HOOK.with(|slot| *slot.borrow_mut() = None);
    CLEAR_GENERATION_HOOK.with(|slot| *slot.borrow_mut() = None);
    TRUNCATE_MARKED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COUNT_CAPTURED_HOOK.with(|slot| *slot.borrow_mut() = None);
    GC_ROWS_REMOVED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLUMN_CHANGE_WAITING_HOOK.with(|slot| *slot.borrow_mut() = None);
    CHECKPOINT_CUT_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETYPE_OUTPUTS_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    RETYPE_MANIFEST_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    ANALYZE_ROWS_COLLECTED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMMIT_COUNTED_HOOK.with(|slot| *slot.borrow_mut() = None);
    SEAL_PASSED_CHANGING_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_RECHECK_HOOK.with(|slot| *slot.borrow_mut() = None);
    TABLE_WAL_RECORDED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMMIT_FENCE_HOOK.with(|slot| *slot.borrow_mut() = None);
    SEAL_GENERATION_HOOK.with(|slot| *slot.borrow_mut() = None);
    COMMIT_ROUND_HOOK.with(|slot| *slot.borrow_mut() = None);
    STATEMENT_ADMITTED_HOOK.with(|slot| *slot.borrow_mut() = None);
    BATCH_ROW_HOOK.with(|slot| *slot.borrow_mut() = None);
    HOT_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
    TXN_GATE_WAIT_HOOK.with(|slot| *slot.borrow_mut() = None);
    TXN_GATE_HELD_OFF_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_ROUND_SETTLED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_ROUND_SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = None);
    INSERT_COLD_SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_READS_FORGET.store(false, Release);
    SELECT_TABLE_OPENED_HOOK.with(|slot| *slot.borrow_mut() = None);
    READ_PATHS.with(|paths| paths.borrow_mut().clear());
    FK_CHILD_PROBE_HOOK.with(|slot| *slot.borrow_mut() = None);
    CORRELATED_FETCH_HOOK.with(|slot| *slot.borrow_mut() = None);
    ADD_COLUMN_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
    SNAPSHOT_SCHEMA_WRITTEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    JOIN_INDEX_CHOSEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    JOIN_NEXT_CHUNK_HOOK.with(|slot| *slot.borrow_mut() = None);
    ALTER_DDL_GUARD_HOOK.with(|slot| *slot.borrow_mut() = None);
    INDEX_BOUND_HOT_HOOK.with(|slot| *slot.borrow_mut() = None);
    JOIN_INNER_OPENED_HOOK.with(|slot| *slot.borrow_mut() = None);
    DROP_RECORDED_HOOK.with(|slot| *slot.borrow_mut() = None);
    DROP_HOT_PUBLISHED_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_CLAIM_TAKEN_HOOK.with(|slot| *slot.borrow_mut() = None);
    COLD_ONLY_CLAIMED_HOOK.with(|slot| *slot.borrow_mut() = None);
    UNIQUE_PAIRS.with(|n| n.set(0));
    COLD_RECHECKS.with(|n| n.set(0));
    TXN_GATE_LOOKUPS.with(|n| n.set(0));
    TXN_GATE_ADMISSIONS.with(|n| n.set(0));
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
