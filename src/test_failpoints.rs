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

/// Reset all failpoints to disabled state
pub fn reset_all() {
    use std::sync::atomic::Ordering::Release;
    WAL_WRITE_FAIL.store(false, Release);
    WAL_SYNC_FAIL.store(false, Release);
    SNAPSHOT_WRITE_FAIL.store(false, Release);
    SNAPSHOT_SYNC_FAIL.store(false, Release);
    SNAPSHOT_RENAME_FAIL.store(false, Release);
    CHECKPOINT_WRITE_FAIL.store(false, Release);
    VERSION_ROOT_HOOK.with(|slot| *slot.borrow_mut() = None);
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
