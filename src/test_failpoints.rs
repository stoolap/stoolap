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
