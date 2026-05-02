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

//! Per-handle reader attachment to a `Database` / `ReadOnlyDatabase`.
//!
//! `ReaderAttachment` is the explicit RAII object that owns this
//! handle's claim on the engine: the cross-process lease pin
//! contribution and the rate-limited heartbeat. One attachment per
//! user-visible read-only handle. Drop releases the pin; explicit
//! `detach()` does the same thing earlier (e.g. from `close()`)
//! and is idempotent.
//!
//! Out of scope: overlay rebuild, sticky must-reopen state,
//! per-table cache invalidation, manifest-epoch poll. Those are
//! per-surface refresh state owned by `ReadOnlyDatabaseInner`
//! (overlay-aware) or — in a follow-up slice — a thin
//! `PlainReadOnlyRefreshState` on `DatabaseInner` (no overlay).
//! Keeping the attachment lifecycle-only avoids leaking
//! refresh-policy concerns into the cross-process claim.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::api::database::EngineEntry;
use crate::storage::mvcc::overlay::OverlayStore;

/// One handle's reader attachment. Owns a single contribution to
/// the per-PID lease's `pinned_lsn` registry and the cached
/// heartbeat / pin-write state used to skip redundant syscalls.
pub(crate) struct ReaderAttachment {
    /// Keeps the engine alive for the duration of the attachment.
    /// Same Arc the owning `ReadOnlyDatabaseInner` /
    /// `DatabaseInner` holds — bumping the entry's strong count by
    /// 1, which the close-path baseline check accounts for
    /// (`baseline = if attachment.is_some() { 2 } else { 1 }`).
    pub(crate) entry: Arc<EngineEntry>,

    /// Per-attachment id assigned at construction; key into the
    /// process-wide `lease_pin_contributions` map so the on-disk
    /// pin stays at the MIN across every live in-process RO
    /// handle. Without this, two handles sharing the per-PID
    /// lease would race on `set_pinned_lsn` and the higher value
    /// would silently overwrite a lagging handle's lower one.
    handle_id: u64,

    /// Last `pinned_lsn` we wrote into the lease via
    /// `set_handle_pin`. `advance_pin` skips the open + write +
    /// set_modified syscalls when the value hasn't changed,
    /// making the no-change query path syscall-free. Initialized
    /// to the actual installed initial pin (or `u64::MAX` when
    /// pin install was skipped) so the first `advance_pin(initial)`
    /// is a no-op rather than a redundant rewrite.
    last_written_pin: AtomicU64,

    /// Epoch-millis of the last lease mtime touch. `touch_lease`
    /// skips the syscall when the prior touch is recent enough
    /// that the writer's lease reaper still considers us live (1
    /// second floor — far below the default
    /// `2 * checkpoint_interval = 120s` reaper window).
    last_touch_epoch_ms: AtomicU64,

    /// Writer's `visible_commit_lsn` snapshot at attach time. DDL
    /// events with `lsn <= attach_visible_commit_lsn` were already
    /// in effect when the reader opened (and the open's schema-
    /// replay path reflected them), so they are NOT surfaced as
    /// `SwmrPendingDdl`. Only DDL events with `lsn > this`
    /// represent post-attach schema changes.
    attach_visible_commit_lsn: u64,

    /// Writer generation observed at attach. On every refresh we
    /// compare this against the current shm value; if it
    /// advanced, the writer crashed and recovered (or
    /// closed+reopened) and our cached state (manifests, overlay,
    /// pinned_lsn) is no longer trustworthy — the surface
    /// surfaces `Error::SwmrWriterReincarnated`. `0` when no shm
    /// is attached. Read-only after attach.
    expected_writer_gen: u64,

    /// True iff this attachment installed a per-handle WAL pin
    /// (i.e. shm + lease both present AND `set_handle_pin`
    /// succeeded at attach). False handles fall through to v1
    /// mtime-only snapshot semantics. Decided at attach,
    /// immutable thereafter.
    swmr_pin_active: bool,

    /// Drop / detach idempotency guard. Set to `true` by the
    /// first `detach()` call (whether explicit or via `Drop`),
    /// short-circuiting any subsequent call. Prevents a double
    /// `remove_handle_pin` if `close()` runs `detach()` and then
    /// the value is dropped, or if multiple shared owners race
    /// on shutdown.
    detached: AtomicBool,
}

impl ReaderAttachment {
    /// Attach a new reader to `entry`: allocate a handle id,
    /// derive the attach snapshot from the entry's stored values,
    /// install a per-handle WAL pin (when shm + lease are both
    /// present), and return the wrapped `Arc`.
    ///
    /// `pin_active()` reflects whether the pin install succeeded —
    /// half-attached handles (shm acquired but lease registration
    /// failed, or vice versa) record `false` and silently skip
    /// the WAL-tail/pin-advance fast paths.
    pub(crate) fn attach(entry: Arc<EngineEntry>) -> Arc<Self> {
        let handle_id = crate::storage::mvcc::lease::next_handle_id();
        let expected_writer_gen = entry.attach_writer_gen;
        let attach_visible_commit_lsn = entry.attach_visible_commit_lsn;
        let fresh_oldest_active = entry.attach_oldest_active_txn_lsn;

        // Install the WAL pin at the OVERLAY BASELINE LSN, not at
        // `attach_visible_commit_lsn` directly. `OverlayStore::initial_pin_lsn`
        // is the one source of truth shared with the bootstrap pin
        // in `Database::open` and the per-handle pin in
        // `DatabaseInner::new_with_entry`.
        let initial_pin_lsn =
            OverlayStore::initial_pin_lsn(attach_visible_commit_lsn, fresh_oldest_active);
        let swmr_pin_active = if let (Some(_), Some(l)) = (entry.shm.as_ref(), entry.lease.as_ref())
        {
            l.set_handle_pin(handle_id, initial_pin_lsn).is_ok()
        } else {
            false
        };
        // Initialize `last_written_pin` to the value we actually
        // installed so the first `advance_pin(initial_pin_lsn)` is
        // a no-op rather than a redundant rewrite. `u64::MAX` is
        // the "no pin in place" sentinel for the half-attached path.
        let last_written_pin = if swmr_pin_active {
            initial_pin_lsn
        } else {
            u64::MAX
        };

        Arc::new(Self {
            entry,
            handle_id,
            last_written_pin: AtomicU64::new(last_written_pin),
            last_touch_epoch_ms: AtomicU64::new(0),
            attach_visible_commit_lsn,
            expected_writer_gen,
            swmr_pin_active,
            detached: AtomicBool::new(false),
        })
    }

    #[inline]
    pub(crate) fn attach_visible_commit_lsn(&self) -> u64 {
        self.attach_visible_commit_lsn
    }

    #[inline]
    pub(crate) fn expected_writer_gen(&self) -> u64 {
        self.expected_writer_gen
    }

    #[inline]
    pub(crate) fn pin_active(&self) -> bool {
        self.swmr_pin_active
    }

    /// Bump the lease's mtime if it hasn't been touched in the
    /// last second. Cheap (one atomic load on the no-op path).
    /// Called from the read-only handle's query / refresh entry
    /// points to keep the writer from reaping us as stale.
    #[inline]
    pub(crate) fn touch_lease(&self) {
        let Some(ref l) = self.entry.lease else {
            return;
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = self.last_touch_epoch_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < 1_000 {
            return;
        }
        if l.touch().is_ok() {
            self.last_touch_epoch_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Advance this handle's WAL pin to `next_floor` if it
    /// changed since the last write. The caller passes the value
    /// from its own refresh state (e.g. `overlay.next_entry_floor()`
    /// floored at 1) — the attachment does not know about
    /// overlay internals.
    ///
    /// No-op when the pin isn't active (no shm, no lease, or
    /// initial install failed). Updates the touch cache on
    /// success because `set_handle_pin` bumps mtime as a side
    /// effect, so the next `touch_lease` can skip its syscall.
    #[inline]
    pub(crate) fn advance_pin(&self, next_floor: u64) {
        let Some(ref l) = self.entry.lease else {
            return;
        };
        if !self.swmr_pin_active {
            return;
        }
        let last = self.last_written_pin.load(Ordering::Relaxed);
        if next_floor == last {
            return;
        }
        if l.set_handle_pin(self.handle_id, next_floor).is_ok() {
            self.last_written_pin.store(next_floor, Ordering::Relaxed);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.last_touch_epoch_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Release this handle's pin contribution. Idempotent: the
    /// first call removes the contribution; subsequent calls
    /// (including the implicit one in `Drop`) short-circuit.
    /// Called explicitly by `Database::close` /
    /// `ReadOnlyDatabase::close` to release the pin BEFORE the
    /// engine close path runs, so a closed handle held alive in a
    /// pool / FFI wrapper stops constraining writer WAL
    /// truncation immediately.
    pub(crate) fn detach(&self) {
        if self.detached.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(ref l) = self.entry.lease {
            l.remove_handle_pin(self.handle_id);
        }
    }
}

impl Drop for ReaderAttachment {
    fn drop(&mut self) {
        // `detach()` is idempotent so this is safe even after an
        // explicit `close()`-driven detach.
        self.detach();
    }
}

impl std::fmt::Debug for ReaderAttachment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderAttachment")
            .field("handle_id", &self.handle_id)
            .field("attach_visible_commit_lsn", &self.attach_visible_commit_lsn)
            .field("expected_writer_gen", &self.expected_writer_gen)
            .field("pin_active", &self.swmr_pin_active)
            .field(
                "detached",
                &self.detached.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}
