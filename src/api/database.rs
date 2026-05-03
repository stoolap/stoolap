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

//! Database struct and operations
//!
//! Provides a modern, ergonomic Rust API for database operations.
//!
//! # Examples
//!
//! ```ignore
//! use stoolap::{Database, params};
//!
//! let db = Database::open("memory://")?;
//!
//! // DDL - no params needed
//! db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;
//!
//! // Insert with params - using tuple syntax
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;
//!
//! // Insert with params! macro
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![2, "Bob", 25])?;
//!
//! // Query with iteration
//! for row in db.query("SELECT * FROM users WHERE age > $1", (20,))? {
//!     let row = row?;
//!     let name: String = row.get("name")?;
//!     println!("{}", name);
//! }
//!
//! // Query single value
//! let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
//! ```

use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::core::{DataType, Error, IsolationLevel, Result, Value};
use crate::executor::context::ExecutionContextBuilder;
use crate::executor::{CachedPlanRef, ExecutionContext, Executor};
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::{Config, SyncMode};

use super::params::{NamedParams, Params};
use super::rows::{FromRow, Rows};
use super::statement::Statement;
use super::transaction::Transaction;

/// Storage scheme constants
pub const MEMORY_SCHEME: &str = "memory";
pub const FILE_SCHEME: &str = "file";

/// Parse a `refresh_interval` value like `"30s"`, `"500ms"`, `"1m"`,
/// or `"0"` (special: no unit allowed for zero, means "disabled").
///
/// Used by `Database::dsn_refresh_interval_flag`. Free function (not
/// a method) so the parser stays close to its sole caller and is
/// directly unit-testable. Rejects unitless non-zero numbers,
/// unknown units, negatives (caught by `u64::parse`), and overflow.
pub(crate) fn parse_refresh_interval_value(value: &str) -> Result<Duration> {
    let trimmed = value.trim();
    if trimmed == "0" {
        return Ok(Duration::ZERO);
    }
    let (num_str, multiplier_ms): (&str, u64) = if let Some(rest) = trimmed.strip_suffix("ms") {
        (rest, 1)
    } else if let Some(rest) = trimmed.strip_suffix('s') {
        (rest, 1_000)
    } else if let Some(rest) = trimmed.strip_suffix('m') {
        (rest, 60_000)
    } else {
        return Err(Error::invalid_argument(format!(
            "invalid refresh_interval: '{}' (expected 'Nms', 'Ns', 'Nm', or '0')",
            value
        )));
    };
    let n: u64 = num_str.parse().map_err(|_| {
        Error::invalid_argument(format!(
            "invalid refresh_interval: '{}' (numeric portion not a non-negative integer)",
            value
        ))
    })?;
    let total_ms = n.checked_mul(multiplier_ms).ok_or_else(|| {
        Error::invalid_argument(format!(
            "invalid refresh_interval: '{}' overflows u64 milliseconds",
            value
        ))
    })?;
    Ok(Duration::from_millis(total_ms))
}

/// Global database registry to ensure single instance per DSN.
///
/// Stores `Weak<EngineEntry>` so the registry never keeps an engine alive
/// past its last user-visible handle. When the last `Database` /
/// `ReadOnlyDatabase` for a DSN drops, `Arc<EngineEntry>` count hits zero,
/// `EngineEntry::drop` closes the engine, and the registry's `Weak`
/// silently expires. The next `open(dsn)` finds the dead `Weak`, fails to
/// upgrade, and creates a fresh `EngineEntry`.
static DATABASE_REGISTRY: std::sync::LazyLock<
    RwLock<FxHashMap<String, std::sync::Weak<EngineEntry>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

/// Engine-level shared state, keyed by DSN in the registry.
///
/// Multiple user-visible handles (`Database` clones, sibling `Database::open`
/// calls, `ReadOnlyDatabase` views) all hold `Arc<EngineEntry>`. The Arc
/// count *is* the count of live user handles for this DSN — there is no
/// other path to an `Arc<EngineEntry>`, no internal clone leaks into other
/// subsystems (the executor and query planner hold `Arc<MVCCEngine>`, not
/// `Arc<EngineEntry>`).
///
/// `EngineEntry::drop` is the single point that closes the engine, so the
/// engine is closed iff every user handle has been dropped — independent of
/// which order they drop in.
pub(crate) struct EngineEntry {
    pub(crate) engine: Arc<MVCCEngine>,
    pub(crate) dsn: String,
    /// Semantic-cache shared across every per-handle `Executor` for this
    /// engine. Each `Database` clone / sibling `Database::open(dsn)` call
    /// gets its own `Executor` (for transaction-state isolation), but
    /// every executor holds an `Arc::clone` of this cache. That way a
    /// DML invalidation on one handle's executor reaches the cached
    /// SELECT results held by every sibling reader. Per-handle caches
    /// would silently serve stale rows after a peer's commit.
    pub(crate) semantic_cache: Arc<crate::executor::SemanticCache>,
    /// Query planner shared across every per-handle `Executor` for this
    /// engine. Same shape as `semantic_cache` and same reason: ANALYZE
    /// invalidates the planner's stats cache, and a per-handle planner
    /// would leave sibling handles on pre-ANALYZE estimates until the
    /// 5-minute TTL expires. Sharing keeps every reader's plan choices
    /// in sync with the writer's `ANALYZE`.
    pub(crate) query_planner: Arc<crate::executor::QueryPlanner>,
    /// Cross-process SWMR lease + shm live on the
    /// EngineEntry, not on `ReadOnlyDatabase`. They get registered
    /// for ANY engine opened in read-only mode against a file:// DSN —
    /// including the `Database::open("file://...?read_only=true")`
    /// path that returns a plain `Database` wrapper (not
    /// `ReadOnlyDatabase`). Without this, that documented entry
    /// point would skip the lease, leaving the writer's cleanup
    /// unable to defer for the reader's WAL/volume needs.
    ///
    /// `None` for in-memory or non-read-only engines.
    pub(crate) lease: Option<crate::storage::mvcc::lease::LeaseManager>,
    /// Read-only mmap of the writer's `db.shm`. Populated alongside
    /// `lease` (same conditions). Falls back to `None` when the
    /// writer hasn't created a shm yet — degrades to v1 mtime-only
    /// presence.
    pub(crate) shm: Option<Arc<crate::storage::mvcc::shm::ShmHandle>>,
    /// Snapshot of `writer_generation` at attach. ReadOnlyDatabase's
    /// refresh path compares this against the live shm value to
    /// detect writer reincarnation.
    pub(crate) attach_writer_gen: u64,
    /// Snapshot of `visible_commit_lsn` at attach. ReadOnlyDatabase's
    /// SwmrPendingDdl filter uses this to suppress DDL events that
    /// pre-date this attach.
    pub(crate) attach_visible_commit_lsn: u64,
    /// Snapshot of `oldest_active_txn_lsn` taken in
    /// `pre_acquire_swmr_for_read_only_path` BEFORE the
    /// visible_commit_lsn snapshot — so a writer txn that committed
    /// between attach and ReadOnlyDatabase construction can't move
    /// the on-disk floor above this value. ReadOnlyDatabase's
    /// overlay baseline uses `min(this, attach_visible_commit_lsn + 1)`
    /// as the entry-scan floor; sampling at attach (rather than at
    /// from_entry) ensures the floor covers any pre-attach DML for
    /// transactions whose commit markers will land post-attach.
    /// `u64::MAX` = no shm or no active user txns at attach time.
    pub(crate) attach_oldest_active_txn_lsn: u64,
    /// Manifest epoch read from `<db>/volumes/epoch` BEFORE this
    /// entry's `open_engine` ran. Used as the baseline that every
    /// `ReadOnlyDatabase::from_entry` seeds into `last_seen_epoch`,
    /// instead of re-reading the live on-disk epoch. Without this,
    /// a sibling ReadOnlyDatabase opened after a writer checkpoint
    /// (or after an entry-reuse via the DSN registry) would seed its
    /// cache from the NEW epoch, then `refresh()` would see
    /// `on_disk == cached` and skip `reload_manifests`, leaving cold
    /// rows the underlying engine never reloaded.
    pub(crate) loaded_epoch: u64,
    /// Epoch-millis of the last shared lease mtime
    /// touch. Both `Database` (when read-only) and
    /// `ReadOnlyDatabase` query paths call into the rate-limited
    /// `EngineEntry::heartbeat_swmr_lease`, which skips the
    /// underlying syscall when the prior touch is < 1 second ago.
    /// Keeps the writer's reaper from declaring the reader stale
    /// without paying a syscall on every query.
    last_lease_touch_ms: std::sync::atomic::AtomicU64,
    /// Long-lived shared file lock held by chmod-read-only
    /// fallback opens (lease registration failed because the
    /// readers/ dir isn't writable for THIS process, but the
    /// path isn't on a read-only mount). Without this, a
    /// concurrent writer running with elevated privileges
    /// could acquire `LOCK_EX` and reclaim WAL/volumes under
    /// the reader — the writer's GC/cleanup is lease-based
    /// and a chmod-RO reader has no lease. Holding `LOCK_SH`
    /// for the entry's lifetime blocks `LOCK_EX` acquisition,
    /// providing kernel-level reader presence as a fallback.
    /// `None` for the standard lease-backed path and for true
    /// read-only-mount opens (where no writer can ever exist).
    #[allow(dead_code)]
    pub(crate) chmod_ro_lock: Option<crate::storage::mvcc::file_lock::SharedLockGuard>,
    /// Temp directory for test-filedb feature. Deleted with the entry.
    #[cfg(feature = "test-filedb")]
    _temp_dir: Option<tempfile::TempDir>,
}

/// Probe whether `db_path` is writable for THIS process by
/// attempting to create a tiny throwaway file inside it. Used to
/// classify lease-registration failures: a writable directory ⇒
/// real failure (transient I/O, ENOSPC, etc.), an unwritable
/// directory ⇒ effectively-read-only (chmod-style RO DB on a
/// writable mount), so we accept lease=None.
///
/// The probe file is always cleaned up. Any I/O error is treated
/// as "not writable" — conservative for the SWMR attach decision.
fn is_directory_writable(db_path: &std::path::Path) -> bool {
    let pid = std::process::id();
    let probe = db_path.join(format!(".swmr-write-probe-{}", pid));
    match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

impl EngineEntry {
    /// Rate-limited mtime touch on the SWMR lease
    /// file. No-op when no lease is registered (non-read-only or
    /// memory engine). Skips the underlying syscall when the
    /// prior touch was < 1 second ago — far below the default
    /// `2 * checkpoint_interval = 120s` reaper floor, so net
    /// correctness is preserved and the common case is
    /// syscall-free.
    ///
    /// Called from every `Database` and `ReadOnlyDatabase` query /
    /// execute entry point. Without it, a documented
    /// `Database::open("file://...?read_only=true")` handle would
    /// leave its lease at the initial mtime and be reaped after
    /// `lease_max_age`, letting the writer proceed with cleanup
    /// while the handle was still active.
    pub(crate) fn heartbeat_swmr_lease(&self) {
        let Some(ref l) = self.lease else { return };
        use std::sync::atomic::Ordering;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = self.last_lease_touch_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < 1_000 {
            return;
        }
        if l.touch().is_ok() {
            self.last_lease_touch_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Register the SWMR lease + shm if the engine is
    /// opened in read-only mode against a file:// path. Returns
    /// `(Option<LeaseManager>, Option<Arc<ShmHandle>>, attach_gen,
    /// attach_lsn)`. `None`/0 for any non-read-only or memory engine.
    /// Pre-acquire SWMR lease + shm + attach snapshots BEFORE
    /// engine construction. The lease is registered first so the
    /// writer's GC sees us during the engine's WAL replay; the
    /// shm snapshot is taken next so the SAME `visible_commit_lsn`
    /// value drives both the engine's `replay_cap_lsn` and the
    /// EngineEntry's `attach_visible_commit_lsn`. This keeps
    /// SwmrPendingDdl pre-attach filtering aligned with what was
    /// actually replayed at attach time.
    ///
    /// Returns all-`None`/0 for non-file paths or non-read-only
    /// configs; the caller decides based on its own state whether
    /// to skip this call.
    #[allow(clippy::type_complexity)]
    pub(crate) fn pre_acquire_swmr_for_read_only_path(
        path: &std::path::Path,
    ) -> Result<(
        Option<crate::storage::mvcc::lease::LeaseManager>,
        Option<Arc<crate::storage::mvcc::shm::ShmHandle>>,
        u64,                                                      // attach_writer_gen
        u64,                                                      // attach_visible_commit_lsn
        u64,                                                      // attach_oldest_active_txn_lsn
        Option<crate::storage::mvcc::file_lock::SharedLockGuard>, // no-shm v1 fallback handshake guard
        u64, // pre_acquire_pin_handle_id (caller removes via remove_handle_pin)
    )> {
        // Lease registration failure is acceptable in TWO
        // "effectively read-only" cases — both proven no-writer
        // paths — and a hard failure otherwise:
        //
        //   1. Whole filesystem mounted read-only (`is_path_on_readonly_mount`).
        //   2. The DB directory itself is not writable for THIS
        //      process (chmod-style read-only DB on a writable
        //      mount). The reader has no lease to advertise
        //      presence; the caller must hold the no-shm
        //      handshake guard (`SharedLockGuard` returned at
        //      tuple position 5 below) for the entire entry
        //      lifetime so a concurrent writer with elevated
        //      privileges cannot acquire `LOCK_EX` and start
        //      cleanup. Without that, the writer's lease-based
        //      GC sees no live reader and can reclaim WAL /
        //      volumes under us.
        //
        // Otherwise (writable directory, transient I/O failure,
        // etc.), fail the SWMR attach loudly: silently dropping
        // lease presence on a writable mount would let the writer's
        // GC reclaim volumes/WAL while we're still attached.
        let lease = match crate::storage::mvcc::lease::LeaseManager::register(path) {
            Ok(l) => Some(l),
            Err(e) => {
                let ro_mount = crate::storage::mvcc::file_lock::is_path_on_readonly_mount_pub(path);
                let dir_unwritable = !ro_mount && !is_directory_writable(path);
                if ro_mount || dir_unwritable {
                    None
                } else {
                    return Err(Error::internal(format!(
                        "SWMR attach failed at '{}': could not register reader \
                         lease: {} (writable filesystem and directory; refusing \
                         to skip lease registration because the writer's GC \
                         depends on live lease presence)",
                        path.display(),
                        e
                    )));
                }
            }
        };
        // Pin WAL BEFORE sampling shm. `register` leaves the lease
        // at 0 bytes; the writer's `min_pinned_lsn` ignores
        // non-8-byte files and treats us as a v1 (no-WAL-pin)
        // reader. Pin `1` (= "keep all WAL") so a writer checkpoint
        // between this point and the engine's manifest load + capped
        // WAL replay can't truncate entries we're about to read.
        //
        // Shared opens do not take a kernel lock, so the writer can
        // start or checkpoint at any time — pinning conservatively
        // is the only safe option for the open window. This holds
        // regardless of shm availability:
        //   - shm present: ReadOnlyDatabase's first refresh advances
        //     this to `overlay.next_entry_floor` via `set_handle_pin`.
        //   - shm absent (no writer running yet, or writer not yet
        //     ready): the open path still needs WAL protection
        //     while it replays. Database::open's plain-Database
        //     release path releases the pin to `0` after
        //     `open_engine` completes (plain Database has no
        //     overlay/tail and doesn't need the pin afterward).
        //     ReadOnlyDatabase::from_entry's first refresh takes
        //     over via `set_handle_pin`.
        // Pin write must SUCCEED before we accept shm. The capped
        // WAL replay path uses `attach_visible_commit_lsn` (from
        // shm) as `replay_cap_lsn` BEFORE `open_engine` — so capped
        // replay would run while no reader pin protects the WAL
        // from concurrent writer checkpoint truncation.
        //
        // Use `set_handle_pin` with a fresh handle_id (NOT the
        // direct `set_pinned_lsn`) so the contribution lives in
        // the process-wide `lease_pin_contributions` registry.
        // The caller releases ONLY this contribution via
        // `remove_handle_pin(pre_acquire_pin_handle_id)`, leaving
        // any sibling ReadOnlyDatabase's contribution intact. A
        // direct `set_pinned_lsn(0)` to release would have
        // overwritten the on-disk MIN, dropping the sibling's
        // pin protection.
        //
        // RAII guard: every `?`/early-Err path between the
        // contribution install and the success return must
        // remove the contribution. `LeaseManager::Drop` only
        // unlinks the file — it does NOT clean the
        // `lease_pin_contributions` registry — so a leaked
        // contribution would let a later same-process open of
        // the same DB inherit a stale pin of `1` and block WAL
        // truncation. Defuse the guard right before the success
        // return so the contribution survives for the caller.
        struct PreAcquirePinGuard<'a> {
            lease: Option<&'a crate::storage::mvcc::lease::LeaseManager>,
            handle_id: u64,
            armed: bool,
        }
        impl Drop for PreAcquirePinGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    if let Some(l) = self.lease {
                        l.remove_handle_pin(self.handle_id);
                    }
                }
            }
        }
        let pre_acquire_pin_handle_id = crate::storage::mvcc::lease::next_handle_id();
        let pin_ok = lease
            .as_ref()
            .map(|l| l.set_handle_pin(pre_acquire_pin_handle_id, 1).is_ok())
            .unwrap_or(false);
        let mut pin_guard = PreAcquirePinGuard {
            lease: lease.as_ref(),
            handle_id: pre_acquire_pin_handle_id,
            armed: pin_ok,
        };
        // Probe whether the writer's `db.shm` file is present on
        // disk. If it IS present, a writer is (or was recently)
        // running and we MUST be able to coordinate with it via
        // shm + pin. Falling back to an uncapped WAL replay in
        // that case would let the reader apply commit markers the
        // writer has flushed but not yet published, observing data
        // the writer considers in-flight. When the file is absent,
        // no writer is running and uncapped recovery from the
        // on-disk WAL is safe (= v1 semantics) — but ONLY while
        // we keep `handshake_guard` alive across the engine open.
        let shm_path = path.join(crate::storage::mvcc::shm::SHM_FILENAME);

        // ALWAYS run the handshake, even when the shm file is on
        // disk. `close_engine` intentionally leaves db.shm behind
        // (and a crash can leave it half-initialized or stuck at
        // init_done = MAGIC_READY), so file presence alone does
        // not prove a live writer. Without this probe we'd cap
        // recovery at a stale visible_commit_lsn and miss durable
        // WAL commits, or fail forever on init_done = 0 until a
        // writer restarts.
        //
        // The handshake returns a [`HandshakeOutcome`] that
        // classifies the writer state:
        //   - NoWriter(SH guard): no LOCK_EX holder. Shm is
        //     stale; do uncapped recovery while the guard
        //     blocks new writers.
        //   - LiveWriter(startup-gate SH guard): a writer holds
        //     LOCK_EX and is past mark_ready right now. The
        //     gate guard MUST stay alive across opening + sampling
        //     db.shm, then a final liveness recheck via
        //     `recheck_writer_still_holds_lock`. If the recheck
        //     reveals the writer disappeared mid-sample, the
        //     sample is discarded and we fall back to uncapped
        //     recovery under the recheck's db.lock SH guard.
        //   - ReadOnlyMount: no writer can ever exist here.
        //     Trust whatever shm is on disk without further proof.
        //   - Err(...): transient flock failure; propagate to
        //     refuse silent downgrade to uncapped replay.
        let outcome =
            crate::storage::mvcc::file_lock::FileLock::await_writer_startup_quiescent(path)?;
        use crate::storage::mvcc::file_lock::HandshakeOutcome;
        let mut handshake_guard: Option<crate::storage::mvcc::file_lock::SharedLockGuard> = None;
        let mut startup_guard: Option<crate::storage::mvcc::file_lock::StartupLockGuard> = None;
        let mut shm_is_stale_leftover = false;
        match outcome {
            HandshakeOutcome::NoWriter(g) => {
                handshake_guard = Some(g);
                shm_is_stale_leftover = true;
            }
            HandshakeOutcome::LiveWriter(sg) => {
                startup_guard = Some(sg);
            }
            HandshakeOutcome::ReadOnlyMount => {}
        }

        let mut shm = if pin_ok && !shm_is_stale_leftover {
            crate::storage::mvcc::shm::ShmHandle::open_reader(path)
                .ok()
                .map(Arc::new)
        } else {
            None
        };
        // Best-effort coherence wait: poll briefly for the
        // writer's cleanup loop to barrier-publish a coherent
        // (visible, oldest) seqlock pair after observing our
        // freshly-written lease. NOT correctness-required —
        // the WAL → shm `fetch_min` mirror (wired in
        // `MVCCEngine::open_engine` via
        // `WALManager::set_shm_oldest_mirror`) already
        // guarantees `shm.oldest_active_txn_lsn` is a valid
        // lower bound at any sample point because every
        // active-set change `fetch_min`s into shm directly.
        // The wait just gives the writer's barrier a chance
        // to publish a fully-fresh pair (matching the writer's
        // ACTUAL oldest, not the running lower bound),
        // avoiding mild over-pinning on the first refresh.
        //
        // Poll up to 250ms (~2-3 cleanup ticks). Timing out
        // is fine: the reader proceeds with whatever shm
        // shows; the WAL→shm mirror's lower-bound guarantee
        // covers correctness.
        if let Some(h) = shm.as_ref() {
            let baseline_seq = h
                .header()
                .publish_seq
                .load(std::sync::atomic::Ordering::Acquire);
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
            while std::time::Instant::now() < deadline {
                let seq = h
                    .header()
                    .publish_seq
                    .load(std::sync::atomic::Ordering::Acquire);
                // Wait for an EVEN seq strictly greater than
                // baseline (any seqlock pair completion bumps
                // by 2; an in-flight publish leaves it odd —
                // keep polling).
                if seq > baseline_seq && seq.is_multiple_of(2) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        // Sample the shm header WHILE holding `startup_guard`
        // (when present). Sampling failures (mid-reincarnation)
        // are surfaced now; a successful sample is then liveness-
        // rechecked below to confirm the writer didn't exit
        // during the sample window.
        let mut attach_gen: u64 = 0;
        let mut attach_oldest_active: u64 = u64::MAX;
        let mut attach_lsn: u64 = 0;
        // Even on the no-shm path (no live writer at attach),
        // probe `db.shm` if it's on disk so we capture the
        // writer_generation as a baseline. A future writer that
        // starts AFTER this reader attached, commits, then
        // exits will leave `db.shm` with a HIGHER
        // writer_generation than this baseline; the reader's
        // refresh detects the increment and surfaces
        // `SwmrWriterReincarnated`. Without this baseline,
        // `attach_gen=0` and the refresh check `observed > 0`
        // would also fire on the test pattern of "writable
        // open, close, then read-only open" (where the prior
        // writable session left writer_generation > 0 with no
        // post-attach writer activity).
        if shm.is_none() {
            if let Ok(h) = crate::storage::mvcc::shm::ShmHandle::open_reader(path) {
                let gen = h
                    .header()
                    .writer_generation
                    .load(std::sync::atomic::Ordering::Acquire);
                attach_gen = gen;
            }
        }
        if let Some(h) = shm.as_ref() {
            match h.header().sample_attach_snapshot() {
                Some((gen, visible, oldest)) => {
                    attach_gen = gen;
                    attach_oldest_active = oldest;
                    attach_lsn = visible;
                }
                None => {
                    if let Some(ref l) = lease {
                        l.remove_handle_pin(pre_acquire_pin_handle_id);
                    }
                    return Err(Error::internal(format!(
                        "SWMR attach failed at '{}': writer is mid-reincarnation \
                         (db.shm header unstable across retries); retry the open",
                        path.display()
                    )));
                }
            }
        }
        // LiveWriter post-sample liveness recheck: if the writer
        // exited between handshake return and our shm sample, the
        // sampled visible_commit_lsn could be from an exited
        // writer's leftover READY shm. Re-acquire db.lock SH:
        //   - Success → writer exited mid-sample → discard the
        //     sample (treat as no-writer) and use the recheck's
        //     SH guard to block any new writer from racing the
        //     uncapped recovery that follows.
        //   - EWOULDBLOCK → writer still holds LOCK_EX, so it
        //     was alive throughout our sample window. The sample
        //     stands; drop the startup gate.
        if let Some(sg) = startup_guard.take() {
            match crate::storage::mvcc::file_lock::FileLock::recheck_writer_still_holds_lock(path)?
            {
                Some(db_lock_g) => {
                    // Writer disappeared mid-sample. Discard the
                    // shm Arc so we don't tail an exited
                    // writer's WAL frontier — uncapped recovery
                    // under the SH guard below replays the
                    // durable WAL the exited writer left behind.
                    //
                    // PRESERVE `attach_gen`: the leftover ready
                    // db.shm carries the just-exited writer's
                    // generation as the latest value any writer
                    // has set. The reader's no-shm refresh check
                    // (`observed > expected`) compares current
                    // db.shm gen against this baseline; without
                    // preserving it, expected=0 would cause every
                    // refresh to surface a spurious
                    // `SwmrWriterReincarnated` against the
                    // existing leftover gen value (typically >= 1)
                    // even though no writer ran post-attach.
                    handshake_guard = Some(db_lock_g);
                    shm_is_stale_leftover = true;
                    shm = None;
                    // attach_gen intentionally retained.
                    attach_oldest_active = u64::MAX;
                    attach_lsn = 0;
                }
                None => {
                    // Writer still alive. The sample stands ONLY
                    // when we actually attached to shm. If shm
                    // open failed (file unlinked between writer
                    // create and our open, EACCES on the inode,
                    // a torn read of the magic header, ...) we
                    // have a live writer with no coordination
                    // surface. Falling through to v1 uncapped
                    // recovery would race the writer's WAL
                    // appends. Hard fail; caller can retry.
                    if shm.is_none() {
                        if let Some(ref l) = lease {
                            l.remove_handle_pin(pre_acquire_pin_handle_id);
                        }
                        return Err(Error::internal(format!(
                            "SWMR attach failed at '{}': writer is live (db.lock \
                             still EX-held) but db.shm could not be attached \
                             (pin_ok={}, shm_path_exists={}); refusing to fall \
                             back to uncapped WAL replay against a live writer; \
                             retry the open",
                            path.display(),
                            pin_ok,
                            shm_path.exists()
                        )));
                    }
                    // Writer alive AND shm attached — release the gate.
                    drop(sg);
                }
            }
        }

        // Track whether shm is effectively available for the
        // post-attach error classification below. Stale shm is
        // treated as absent.
        let shm_file_exists = shm_path.exists() && !shm_is_stale_leftover;
        if shm_file_exists && shm.is_none() {
            // shm file exists but we couldn't open it / pin. If
            // the filesystem or directory is effectively read-only
            // (verified RO mount or unwritable dir on a writable
            // mount), no live writer can be running here — the
            // shm is just a leftover from a prior writer session.
            // Fall through to the no-shm v1 fallback in that case;
            // capped replay isn't required because no writer is
            // appending WAL.
            //
            // On a writable mount with normal perms, shm-exists +
            // can't-coordinate means a live writer is around and
            // we'd race its commits with uncapped replay — fail.
            let ro_mount = crate::storage::mvcc::file_lock::is_path_on_readonly_mount_pub(path);
            let dir_unwritable = !ro_mount && !is_directory_writable(path);
            if !ro_mount && !dir_unwritable {
                if let Some(ref l) = lease {
                    l.remove_handle_pin(pre_acquire_pin_handle_id);
                }
                return Err(Error::internal(format!(
                    "SWMR attach failed at '{}': writer's db.shm exists but \
                     reader could not acquire shm/pin (pin_ok={}, shm_open_ok={}); \
                     retry the open",
                    path.display(),
                    pin_ok,
                    shm.is_some()
                )));
            }
            // Effectively-RO path: drop any failed-pin contribution
            // (shm remains None for the rest of pre_acquire).
            if let Some(ref l) = lease {
                l.remove_handle_pin(pre_acquire_pin_handle_id);
            }
        }
        // Return the handshake guard ONLY for the no-shm v1
        // fallback. With shm attached, the engine's own
        // `open_engine` shared lock takes over (and our lease pin
        // already protects the WAL), so dropping the guard here is
        // safe. With shm absent, the caller MUST hold the guard
        // across `engine.open_engine()` to prevent a writer from
        // sneaking in, creating shm, and writing WAL during our
        // uncapped recovery.
        let returned_guard = if shm.is_none() { handshake_guard } else { None };
        // Defuse the RAII guard: success path returns the
        // contribution to the caller, who removes it later. Drop
        // the guard explicitly so its `&LeaseManager` borrow
        // releases before the success Ok moves `lease`.
        pin_guard.armed = false;
        drop(pin_guard);
        Ok((
            lease,
            shm,
            attach_gen,
            attach_lsn,
            attach_oldest_active,
            returned_guard,
            pre_acquire_pin_handle_id,
        ))
    }
}

impl Drop for EngineEntry {
    fn drop(&mut self) {
        // Clear all thread-local caches to release references to engine internals
        // (cached Arc<dyn Index>, closures). Done once per engine close.
        crate::executor::clear_all_thread_local_caches();
        let _ = self.engine.close_engine();

        // Reap our dead `Weak` from the registry. Without this, every
        // dropped DSN leaves a permanent (DSN string -> dead Weak) entry
        // behind, so a long-lived process opening many ephemeral DSNs
        // grows the registry monotonically and pays for the dead entries
        // on every `open()` lookup.
        //
        // We're inside `Drop` for the entry whose Arc count just hit 0,
        // so any `Weak` pointing at us is now dead. We only remove the
        // entry if the registry still has a *dead* weak for our DSN — if
        // a fresh entry was inserted concurrently between our drop and
        // this lock acquire, its weak is live and we leave it alone.
        //
        // `try_write` to avoid blocking on a held registry lock; the
        // entry will be reaped on the next `open()` of the same DSN
        // either way.
        if let Ok(mut registry) = DATABASE_REGISTRY.try_write() {
            if let Some(weak) = registry.get(&self.dsn) {
                if weak.strong_count() == 0 {
                    registry.remove(&self.dsn);
                }
            }
        }
    }
}

/// Per-handle database state.
///
/// Each user-visible handle (a `Database`, every `Database::clone`, every
/// sibling from `Database::open(dsn)`, every `ReadOnlyDatabase`) owns its
/// own `DatabaseInner` — primarily for executor isolation, so a `BEGIN` on
/// one handle doesn't leak into another. Engine-level shared state lives on
/// the `Arc<EngineEntry>` field, which is what the registry counts.
pub(crate) struct DatabaseInner {
    entry: Arc<EngineEntry>,
    /// Wrapped in `Arc<Mutex<...>>` so `Database::read_engine`'s
    /// guard can clone a reference and check for an active
    /// `BEGIN` on this handle. Without that share, a caller
    /// could open a transaction on `Database`, call
    /// `db.read_engine().begin_read_transaction()`, and have
    /// the guard refresh the shared read-only state
    /// mid-transaction (only matters when the trait object's
    /// `RefreshOwner::ReadOnly` variant is in play, e.g.
    /// `Database::as_read_only` paths).
    executor: Arc<Mutex<Executor>>,
}

/// Type alias for Statement to use (avoids exposing DatabaseInner directly)
pub(crate) type DatabaseInnerHandle = DatabaseInner;

impl DatabaseInner {
    /// Build a fresh per-handle inner around an existing engine entry.
    /// Picks a writable or read-only executor to match the engine mode,
    /// and shares the engine entry's semantic cache and query planner so
    /// DML invalidation and ANALYZE reach every sibling reader.
    fn new_with_entry(entry: Arc<EngineEntry>) -> Self {
        // `Database` is always writable — read-only DSNs are
        // rejected by `Database::open` and routed to
        // `Database::open_read_only`, which returns
        // `ReadOnlyDatabase` (a different type with its own
        // `ReaderAttachment` + refresh state). So this
        // constructor builds a writable executor unconditionally
        // and stores no SWMR claim on the inner.
        let engine = Arc::clone(&entry.engine);
        let semantic_cache = Arc::clone(&entry.semantic_cache);
        let query_planner = Arc::clone(&entry.query_planner);
        let executor = Executor::with_shared_semantic_cache(engine, semantic_cache, query_planner);
        Self {
            entry,
            executor: Arc::new(Mutex::new(executor)),
        }
    }
}

/// `ReadEngine` trait object returned by
/// [`Database::read_engine`] and [`crate::api::ReadOnlyDatabase::read_engine`].
///
/// Every `begin_read_transaction*` call goes through this
/// wrapper, which:
///   - Heartbeats the cross-process reader lease (rate-limited
///     internally by `EngineEntry::heartbeat_swmr_lease`), so a
///     long-lived caller polling only via the trait object
///     stays live and isn't reaped by the writer's lease GC.
///   - Dispatches refresh based on `RefreshOwner`: `ReadOnly`
///     runs overlay rebuild + DDL detection + pin advance via
///     the shared `Arc<ReadOnlyDatabaseInner>`; `None`
///     (writable / in-memory) is a no-op. Errors from refresh
///     propagate as `begin_read_transaction*` errors, surfacing
///     typed must-reopen sub-kinds (`SwmrPendingDdl`,
///     `SwmrWriterReincarnated`, ...) the same way the public
///     `query` paths do.
///
/// The wrapper is the SOLE ReadEngine surface for read-only
/// callers. The bare `Arc<MVCCEngine>` is no longer cast to
/// `Arc<dyn ReadEngine>` directly — that would bypass all of
/// the above and leave the documented "safe read surface"
/// silently stale.
pub(crate) struct SwmrReadEngineGuard {
    pub(crate) engine: Arc<MVCCEngine>,
    pub(crate) entry: Arc<EngineEntry>,
    /// Refresh dispatch. Two states: writable engines get
    /// `None`; `ReadOnlyDatabase` gets `ReadOnly` (which carries
    /// the shared `Arc<ReadOnlyDatabaseInner>` so the trait
    /// surface drives the same overlay/DDL/pin-advance state as
    /// the owning handle).
    pub(crate) refresh_owner: RefreshOwner,
}

/// Refresh dispatch for `SwmrReadEngineGuard`. Two discrete
/// states; the guard's `maintain` matches and dispatches.
pub(crate) enum RefreshOwner {
    /// `ReadOnlyDatabase`. The shared `Arc<ReadOnlyDatabaseInner>`
    /// drives overlay rebuild + DDL detection + pin advance.
    /// The guard delegates to the inner's `maybe_auto_refresh`.
    ReadOnly(Arc<crate::api::read_only_database::ReadOnlyDatabaseInner>),
    /// Writable engine or in-memory engine. No refresh needed.
    None,
}

impl SwmrReadEngineGuard {
    /// Run the per-call SWMR maintenance shared by every trait
    /// method: lease heartbeat, then refresh dispatch. Returns
    /// the typed error from refresh so the caller's
    /// `begin_read_transaction*` surfaces it instead of a stale
    /// Ok.
    #[inline]
    fn maintain(&self) -> Result<()> {
        self.entry.heartbeat_swmr_lease();
        match &self.refresh_owner {
            RefreshOwner::ReadOnly(inner) => {
                // ROD's `maybe_auto_refresh` does its own
                // active-txn check via `inner.executor`, which
                // is identical to the executor a SQL `BEGIN` on
                // the ROD touches.
                let rod = crate::api::ReadOnlyDatabase {
                    inner: Arc::clone(inner),
                };
                rod.maybe_auto_refresh()?;
            }
            RefreshOwner::None => {}
        }
        Ok(())
    }
}

impl crate::storage::traits::ReadEngine for SwmrReadEngineGuard {
    fn begin_read_transaction(&self) -> Result<Box<dyn crate::storage::traits::ReadTransaction>> {
        self.maintain()?;
        self.engine.begin_read_transaction()
    }

    fn begin_read_transaction_with_level(
        &self,
        level: crate::core::IsolationLevel,
    ) -> Result<Box<dyn crate::storage::traits::ReadTransaction>> {
        self.maintain()?;
        self.engine.begin_read_transaction_with_level(level)
    }
}

/// Database represents a Stoolap database connection
///
/// This is the main entry point for using Stoolap. It wraps the storage engine
/// and executor, providing a simple API for executing SQL queries.
///
/// # Thread Safety
///
/// Database is thread-safe and can be shared across threads via cloning.
/// Each clone shares the same underlying storage engine.
///
/// # Examples
///
/// ```ignore
/// use stoolap::{Database, params};
///
/// // Open in-memory database
/// let db = Database::open("memory://")?;
///
/// // Create table
/// db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())?;
///
/// // Insert with parameters
/// db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
///
/// // Query
/// for row in db.query("SELECT * FROM users", ())? {
///     let row = row?;
///     println!("{}: {}", row.get::<i64>("id")?, row.get::<String>("name")?);
/// }
/// ```
pub struct Database {
    inner: Arc<DatabaseInner>,
}

#[cfg(feature = "ffi")]
impl Database {
    /// Returns an Arc reference to the inner state, preventing the engine
    /// from being closed while any keepalive handle exists.
    ///
    /// Used by the FFI layer to ensure cloned handles keep the original
    /// engine-owning DatabaseInner alive.
    pub(crate) fn keepalive(&self) -> Arc<DatabaseInner> {
        Arc::clone(&self.inner)
    }

    /// Returns a borrow of the inner Arc (no clone, no count change).
    pub(crate) fn inner_arc(&self) -> &Arc<DatabaseInner> {
        &self.inner
    }
}

impl Database {
    /// Best-effort cleanup of a registry entry pointing to the same engine
    /// the caller holds.
    ///
    /// With the `Weak<EngineEntry>` registry the entry self-expires once the
    /// last user handle drops, so this method is no longer load-bearing for
    /// correctness. It is retained for the FFI's explicit `stoolap_close`
    /// flow to keep the registry tidy.
    ///
    /// Removal is only safe when the engine is about to die after the
    /// caller's `Arc<DatabaseInner>` is dropped, i.e. when:
    /// - `Arc::strong_count(inner) == 1`: nobody else holds *this*
    ///   `DatabaseInner` (FFI prepared-statement / transaction keepalives
    ///   clone the same `Arc<DatabaseInner>`, so they bump this count
    ///   without bumping `entry.strong_count` — checking only the entry
    ///   would orphan a still-live engine from the registry); AND
    /// - `Arc::strong_count(&inner.entry) == 1`: no sibling `DatabaseInner`
    ///   from a different `Database::open(dsn)` / clone holds the entry.
    ///
    /// If either count is greater than 1, the engine will outlive this
    /// caller — leave the registry alone so a subsequent `open(dsn)` can
    /// still find it. Otherwise the next `open(dsn)` would create a fresh
    /// engine (empty for `memory://`, file-lock conflict for `file://`)
    /// while the prior engine is still in use through a stale handle.
    #[cfg(feature = "ffi")]
    pub(crate) fn try_unregister_arc(inner: &Arc<DatabaseInner>) {
        if Arc::strong_count(inner) > 1 {
            // Other Arc<DatabaseInner> clones (FFI stmt/tx keepalive) keep
            // this exact DatabaseInner — and therefore its entry — alive.
            return;
        }
        // `Database` is always writable now, so the baseline is
        // simply 1 (this DatabaseInner's `Arc<EngineEntry>`).
        if Arc::strong_count(&inner.entry) > 1 {
            // Sibling DatabaseInners share the same engine entry.
            return;
        }
        if let Ok(mut registry) = DATABASE_REGISTRY.write() {
            if let Some(weak) = registry.get(&inner.entry.dsn) {
                match weak.upgrade() {
                    Some(reg_entry) if Arc::ptr_eq(&reg_entry, &inner.entry) => {
                        registry.remove(&inner.entry.dsn);
                    }
                    None => {
                        // Dead entry — clean it up.
                        registry.remove(&inner.entry.dsn);
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Database {
    /// Build a new `Database` handle that shares the engine entry of
    /// `existing` but has its own `DatabaseInner` and its own executor
    /// (independent transaction state).
    ///
    /// Used by both `Clone for Database` and the registry-hit fast path in
    /// `Database::open`. Each handle gets its own executor so a `BEGIN` on
    /// one handle does not leak into another, and each handle bumps the
    /// engine entry's strong count by one — so `Arc::strong_count(&entry)`
    /// is exactly the count of live user handles for the DSN, which is
    /// what `close()` and `try_unregister_arc` use to decide when to
    /// release engine resources.
    fn share_entry(entry: Arc<EngineEntry>) -> Database {
        Database {
            inner: Arc::new(DatabaseInner::new_with_entry(entry)),
        }
    }
}

impl Clone for Database {
    /// Clone the database handle.
    ///
    /// Each cloned handle has its own executor with independent transaction state,
    /// but shares the same underlying storage engine. This ensures proper transaction
    /// isolation - a BEGIN on one handle won't affect reads on another handle.
    fn clone(&self) -> Self {
        Database::share_entry(Arc::clone(&self.inner.entry))
    }
}

// `Database` has no `Drop` impl: dropping it drops `inner` which drops the
// per-handle `Arc<EngineEntry>`. When the *last* user handle for a DSN
// drops, the entry's strong count hits zero and `EngineEntry::drop` closes
// the engine. The registry's `Weak` then silently expires; the next
// `Database::open(dsn)` will see a dead Weak, fail the upgrade, and create
// a fresh entry. No registry-removal logic is needed in `Drop for
// Database` — relying on it was the source of the round-5 bug where a
// sibling's drop unregistered the engine while peers were still using it.

impl Database {
    /// Open a database connection
    ///
    /// The DSN (Data Source Name) specifies the database location:
    /// - `memory://` - In-memory database (data lost when closed)
    /// - `file:///path/to/db` - Persistent database at the specified path
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // In-memory database
    /// let db = Database::open("memory://")?;
    ///
    /// // Persistent database
    /// let db = Database::open("file:///tmp/mydb")?;
    /// ```
    ///
    /// # Engine Reuse
    ///
    /// Opening the same DSN multiple times returns the same engine instance.
    /// This ensures consistency and prevents data corruption.
    pub fn open(dsn: &str) -> Result<Self> {
        // Read-only DSNs go through `Database::open_read_only(dsn)`. The
        // type system distinguishes the writable surface (`Database`) from
        // the read-only one (`ReadOnlyDatabase`); routing read-only DSNs
        // through `open` would force a runtime check on every write
        // method instead of a compile-time gate. Refuse with a clear
        // pointer to the right entry point.
        if Self::dsn_read_only_flag(dsn)? == Some(true) {
            return Err(Error::invalid_argument(
                "read-only DSN flag (?read_only=true / ?readonly=true / \
                 ?mode=ro) passed to Database::open. Read-only handles \
                 must be opened via Database::open_read_only(dsn) so the \
                 returned ReadOnlyDatabase enforces the read-only \
                 contract at the type level. The DSN string itself \
                 (including the flag) can be passed unchanged to \
                 open_read_only.",
            ));
        }

        // Check if we already have an engine for this DSN.
        // Read-only-cached entries (from a prior
        // `Database::open_read_only(dsn)` call) cannot serve a
        // writable request; reject with the mode-mismatch error
        // so the caller drops the read-only handle first or
        // opens a fresh DSN.
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::LockAcquisitionFailed("registry read".to_string()))?;
            if let Some(weak) = registry.get(dsn) {
                if let Some(entry) = weak.upgrade() {
                    if entry.engine.is_read_only_mode() {
                        return Err(Error::read_only_mode_mismatch(dsn, true, false));
                    }
                    return Ok(Self::share_entry(entry));
                }
                // Dead Weak — fall through to create a fresh engine entry.
            }
        }

        // Need to create a new engine - acquire write lock
        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::LockAcquisitionFailed("registry write".to_string()))?;

        // Double-check after acquiring write lock
        if let Some(weak) = registry.get(dsn) {
            if let Some(entry) = weak.upgrade() {
                if entry.engine.is_read_only_mode() {
                    return Err(Error::read_only_mode_mismatch(dsn, true, false));
                }
                return Ok(Self::share_entry(entry));
            }
            // Dead Weak — will be overwritten by the insert below.
        }

        // Parse the DSN
        let (scheme, path) = Self::parse_dsn(dsn)?;

        // test-filedb: track temp dir so it lives as long as the engine
        #[cfg(feature = "test-filedb")]
        let mut _temp_dir_holder: Option<tempfile::TempDir> = None;

        // Create the engine. `Database::open` is writable-only (the
        // `?read_only` DSN flag was rejected above), so no SWMR
        // pre-acquire / lease / shm setup runs here.
        let engine = match scheme.as_str() {
            MEMORY_SCHEME => {
                #[cfg(feature = "test-filedb")]
                {
                    let tmp = tempfile::tempdir().map_err(|e| {
                        Error::internal(format!("failed to create temp dir: {}", e))
                    })?;
                    let file_dsn = format!("file://{}", tmp.path().display());
                    let (_clean_path, config) = Self::parse_file_config(&file_dsn[7..])?;
                    let engine = MVCCEngine::new(config);
                    engine.open_engine()?;
                    let engine = Arc::new(engine);
                    engine.start_cleanup();
                    _temp_dir_holder = Some(tmp);
                    engine
                }
                #[cfg(not(feature = "test-filedb"))]
                {
                    let engine = MVCCEngine::in_memory();
                    engine.open_engine()?;
                    let engine = Arc::new(engine);
                    engine.start_cleanup();
                    engine
                }
            }
            FILE_SCHEME => {
                let (_clean_path, config) = Self::parse_file_config(&path)?;
                let engine = MVCCEngine::new(config);
                engine.open_engine()?;
                let engine = Arc::new(engine);
                engine.start_cleanup();
                engine
            }
            _ => {
                return Err(Error::parse(format!(
                    "Unsupported scheme '{}'. Use 'memory://' or 'file://path'",
                    scheme
                )));
            }
        };

        // Build the engine entry. Writable-only: no lease, no shm,
        // no attach snapshot — those exist solely for cross-process
        // SWMR readers (`Database::open_read_only`).
        let semantic_cache = Arc::new(crate::executor::SemanticCache::default());
        let query_planner = Arc::new(crate::executor::QueryPlanner::new(Arc::clone(&engine)));
        let entry = Arc::new(EngineEntry {
            engine,
            dsn: dsn.to_string(),
            semantic_cache,
            query_planner,
            lease: None,
            shm: None,
            attach_writer_gen: 0,
            attach_visible_commit_lsn: 0,
            attach_oldest_active_txn_lsn: u64::MAX,
            loaded_epoch: 0,
            last_lease_touch_ms: std::sync::atomic::AtomicU64::new(0),
            chmod_ro_lock: None,
            #[cfg(feature = "test-filedb")]
            _temp_dir: _temp_dir_holder,
        });

        // Store a Weak in the registry so it self-expires when the last
        // user handle drops.
        registry.insert(dsn.to_string(), Arc::downgrade(&entry));

        Ok(Self::share_entry(entry))
    }

    /// Open an in-memory database
    ///
    /// This is a convenience method that creates a new in-memory database.
    /// Each call creates a unique instance (unlike `open("memory://")` which
    /// would share the same instance).
    pub fn open_in_memory() -> Result<Self> {
        Self::create_in_memory_engine()
    }

    /// Open a read-only handle over an existing database.
    ///
    /// Opens the database normally (or reuses an existing registry entry) and
    /// wraps the engine in a `ReadOnlyDatabase` that rejects all write SQL at
    /// query time.
    ///
    /// On a `file://` DSN this also registers a cross-process presence
    /// lease so a sibling writer process defers volume cleanup while this
    /// reader is attached. See [`crate::api::ReadOnlyDatabase`] doc for
    /// the lease-touch cadence requirement (issue at least one query
    /// per `2 * checkpoint_interval` to keep the lease fresh).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let rodb = Database::open_read_only("file:///tmp/mydb")?;
    /// for row in rodb.query("SELECT * FROM users", ())? {
    ///     let row = row?;
    ///     println!("{:?}", row);
    /// }
    /// ```
    pub fn open_read_only(dsn: &str) -> Result<crate::api::ReadOnlyDatabase> {
        // Read-only is meaningless on `memory://`: a fresh in-memory engine
        // has nothing to read. Reject early.
        if dsn.starts_with(MEMORY_SCHEME) {
            return Err(Error::invalid_argument(
                "open_read_only is not supported on memory:// (a fresh \
                 in-memory engine has no data to read); use file:// for \
                 read-only deployments",
            ));
        }

        // The function name already says read-only. DSN flags
        // requesting read-only (`?read_only=true` /
        // `?readonly=true` / `?mode=ro`) are accepted as
        // redundant — drivers that build DSN strings can keep
        // emitting them. A flag explicitly requesting writable
        // (`?read_only=false` / `?readonly=false` / `?mode=rw`)
        // contradicts the function name and is rejected so the
        // caller catches the mistake at the API surface.
        if Self::dsn_read_only_flag(dsn)? == Some(false) {
            return Err(Error::invalid_argument(
                "Database::open_read_only called with a DSN flag that \
                 explicitly requests writable mode (?read_only=false / \
                 ?readonly=false / ?mode=rw). The function name and the \
                 DSN flag disagree — drop the flag (it's redundant on \
                 open_read_only) or use Database::open(dsn) instead.",
            ));
        }

        // Parse `auto_refresh` and `refresh_interval` once, up front,
        // so the parse error surfaces before any registry / engine
        // work and the flags are uniformly applied to both the
        // cache-hit and fresh-create paths below. `Duration::ZERO`
        // means "explicitly disabled" (no background ticker); `None`
        // means "absent from DSN, leave the per-handle default".
        let dsn_auto_refresh = Self::dsn_auto_refresh_flag(dsn)?;
        let dsn_refresh_interval = Self::dsn_refresh_interval_flag(dsn)?;
        let apply_dsn_flags =
            |ro: crate::api::ReadOnlyDatabase| -> Result<crate::api::ReadOnlyDatabase> {
                if let Some(enabled) = dsn_auto_refresh {
                    ro.set_auto_refresh(enabled);
                }
                if let Some(d) = dsn_refresh_interval {
                    if !d.is_zero() {
                        ro.set_refresh_interval(Some(d))?;
                    }
                }
                Ok(ro)
            };

        // If the DSN is already open in this process (writable or
        // read-only), share the existing engine entry — but ONLY
        // when ALL of:
        //   - `entry.shm` is `Some` (no-shm v1 fallback entries
        //     are non-reusable: see the matching predicate in
        //     `Database::open` for why),
        //   - the writer's published frontier hasn't moved past
        //     the entry's attach, AND
        //   - the writer's `writer_generation` matches the
        //     entry's attach snapshot.
        //
        // The visible-LSN check protects against silently
        // skipping WAL: the shared engine was opened ONCE with
        // `attach_visible_commit_lsn` as its replay cap; any
        // commits the writer published after that point are not
        // in the engine's parent VersionStores. Older handles on
        // the same entry pick them up via their per-handle overlay
        // tail, but a new handle's overlay seeded at any
        // post-attach baseline would mark those WAL entries as
        // already applied and silently skip them.
        //
        // The generation check protects against writer
        // close/reopen or crash/restart that recovered to the
        // same `visible_commit_lsn`. In that case any old
        // handle correctly raises `SwmrWriterReincarnated` on
        // its next refresh, but reuse here would hand the
        // caller back the same stale `EngineEntry` (same
        // `attach_writer_gen`), and the next refresh would
        // immediately fail again. Refusing reuse on a generation
        // bump forces a fresh open path that re-samples the new
        // writer's identity into a new attach snapshot.
        //
        // The no-shm rejection protects against a cross-process
        // writer that started AFTER this process's no-writer
        // open. The original handshake (NoWriter outcome)
        // dropped the kernel SH right after `open_engine`, and
        // `acquire_shared` on Unix takes no kernel lock, so a
        // writer can have created `db.shm` and published commits
        // since attach. A reused entry would skip
        // `pre_acquire_swmr_for_read_only_path` and never see
        // the new writer's frontier or DDL.
        let frontier_static = |entry: &Arc<EngineEntry>| -> bool {
            match entry.shm.as_ref() {
                Some(h) => {
                    let observed_visible = h
                        .header()
                        .visible_commit_lsn
                        .load(std::sync::atomic::Ordering::Acquire);
                    let observed_gen = h
                        .header()
                        .writer_generation
                        .load(std::sync::atomic::Ordering::Acquire);
                    observed_visible == entry.attach_visible_commit_lsn
                        && observed_gen == entry.attach_writer_gen
                }
                None => false,
            }
        };
        // Cached writable entries are ALWAYS reusable: the writer
        // owns the engine, so wrapping it as a `ReadOnlyDatabase`
        // is the same operation as `Database::as_read_only()`.
        // The `frontier_static` check is gated on `entry.shm`,
        // which is always `None` for writable entries (shm/lease
        // snapshots are only stored on read-only pre-acquire), so
        // running the check would unconditionally fall through to
        // fresh-create, evicting the live writable weak from the
        // registry. A subsequent `Database::open(dsn)` would then
        // see the new read-only weak and return a mode-mismatch
        // error against the still-live writable handle.
        let cached_is_writable =
            |entry: &Arc<EngineEntry>| -> bool { !entry.engine.is_read_only_mode() };
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::LockAcquisitionFailed("registry read".to_string()))?;
            if let Some(weak) = registry.get(dsn) {
                if let Some(entry) = weak.upgrade() {
                    if cached_is_writable(&entry) || frontier_static(&entry) {
                        return apply_dsn_flags(crate::api::ReadOnlyDatabase::from_entry(entry));
                    }
                    // Frontier moved — fall through to fresh-create.
                    // Existing handles keep their Arc<EngineEntry>;
                    // the stale Weak<> in the registry will be
                    // overwritten by the new entry's insert below.
                }
            }
        }

        // Need to create a new engine in read-only mode (acquires LOCK_SH,
        // skips background cleanup). Acquire registry write lock.
        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::LockAcquisitionFailed("registry write".to_string()))?;

        // Double-check after acquiring write lock.
        if let Some(weak) = registry.get(dsn) {
            if let Some(entry) = weak.upgrade() {
                if cached_is_writable(&entry) || frontier_static(&entry) {
                    return apply_dsn_flags(crate::api::ReadOnlyDatabase::from_entry(entry));
                }
                // Frontier moved — fall through.
            }
        }

        let (scheme, path) = Self::parse_dsn(dsn)?;

        #[cfg(feature = "test-filedb")]
        let _temp_dir_holder: Option<tempfile::TempDir> = None;

        // memory:// was rejected at the top of this function. Only file://
        // reaches the engine-construction match. Each arm returns the engine
        // and the SWMR lease/shm/attach snapshots so the EngineEntry can
        // observe the writer's state from the moment recovery begins.
        let (
            engine,
            lease,
            shm,
            attach_writer_gen,
            attach_visible_commit_lsn,
            loaded_epoch,
            attach_oldest_active_txn_lsn,
            pre_acquire_pin_handle_id,
            chmod_ro_lock,
        ) = match scheme.as_str() {
            FILE_SCHEME => {
                let (clean_path, mut config) = Self::parse_file_config(&path)?;
                config.read_only = true;

                // Read-only opens must not create a new database. Refuse if
                // the directory doesn't already exist (or exists but lacks a
                // recognizable stoolap layout). Without this check,
                // PersistenceManager::new would `create_dir_all` and lay
                // down a fresh WAL, silently turning `open_read_only` into
                // a write that creates an empty DB.
                let path_obj = std::path::Path::new(&clean_path);
                if !path_obj.exists() {
                    return Err(Error::internal(format!(
                        "cannot open '{}' read-only: path does not exist",
                        clean_path
                    )));
                }
                if !path_obj.is_dir() {
                    return Err(Error::internal(format!(
                        "cannot open '{}' read-only: not a directory",
                        clean_path
                    )));
                }
                // A stoolap database always has a `wal` subdirectory once
                // it has been written to. If neither `wal/` nor `volumes/`
                // exists, the directory is not a stoolap database and we
                // refuse to materialize one in read-only mode.
                let has_wal = path_obj.join("wal").exists();
                let has_volumes = path_obj.join("volumes").exists();
                if !has_wal && !has_volumes {
                    return Err(Error::internal(format!(
                        "cannot open '{}' read-only: not a stoolap database \
                             (no wal/ or volumes/ directory)",
                        clean_path
                    )));
                }

                // Pre-acquire the SWMR lease + shm snapshot BEFORE engine
                // open. The lease must be live during WAL replay so the
                // writer's GC sees us; the attach snapshot drives the
                // engine's replay cap so unpublished commit markers are
                // not applied.
                let (
                    lease,
                    shm,
                    attach_writer_gen,
                    attach_visible_commit_lsn,
                    attach_oldest_active_txn_lsn,
                    handshake_guard,
                    pre_acquire_pin_handle_id,
                ) = EngineEntry::pre_acquire_swmr_for_read_only_path(std::path::Path::new(
                    &clean_path,
                ))?;
                // The pre-acquire pin contribution stays in the
                // registry until ReadOnlyDatabase::from_entry
                // installs its own per-handle pin (also in the
                // registry) and then we remove the pre-acquire
                // contribution below — registry MIN guarantees
                // the on-disk lease never drops below the
                // sibling-handles' floor at any point.
                // `handshake_guard` is `Some` only on no-shm
                // paths. Held across `engine.open_engine()` to
                // block writer-startup races; for the chmod-RO
                // case (lease=None on a writable mount) we
                // ALSO move it into `EngineEntry.chmod_ro_lock`
                // below so it persists for the entry's
                // lifetime, providing kernel-level reader
                // presence the writer's `LOCK_EX` acquisition
                // would otherwise bypass.
                // Snapshot the manifest epoch BEFORE open_engine
                // for the same reason as Database::open: a sibling
                // ReadOnlyDatabase from `from_entry` seeds its
                // `last_seen_epoch` from this baseline rather than
                // reading the live on-disk file.
                let loaded_epoch = crate::storage::mvcc::manifest_epoch::read_epoch(
                    std::path::Path::new(&clean_path),
                )
                .unwrap_or(0);
                let engine = MVCCEngine::new(config);
                if shm.is_some() {
                    engine.set_replay_cap_lsn(attach_visible_commit_lsn);
                }
                // open_engine failure must NOT leak the temp
                // pre-acquire pin contribution. See the
                // matching `Database::open` cleanup for the
                // full rationale.
                if let Err(e) = engine.open_engine() {
                    if let Some(ref l) = lease {
                        l.remove_handle_pin(pre_acquire_pin_handle_id);
                    }
                    return Err(e);
                }
                let engine = Arc::new(engine);
                // start_cleanup is a no-op when config.read_only is set,
                // but call it for symmetry with the writable path.
                engine.start_cleanup();
                // Do NOT release the pre-acquire pin here. There
                // is a gap between this point and
                // `ReadOnlyDatabase::from_entry` (which installs
                // the per-handle pin); a writer checkpoint in
                // that gap could truncate post-attach WAL the
                // first refresh needs. The pre-acquire pin of
                // `1` keeps WAL protected through the gap, and
                // `from_entry`'s `set_handle_pin(handle_id, 1)`
                // overwrites the on-disk value with the same `1`
                // (no behaviour change there). Drop's
                // `remove_handle_pin` then releases to `0` when
                // the last per-handle contribution drops.
                // Persist the handshake guard for the chmod-RO
                // / RO-mount fallback (lease=None paths). For
                // ro_mount no writer can ever exist so the
                // guard is harmless extra protection; for
                // chmod-RO it's the only thing keeping a
                // privileged writer from `LOCK_EX`-ing under
                // us. For the standard lease-backed path the
                // guard is dropped here — the writer
                // coordinates via the lease, not flock.
                let chmod_ro_lock = if lease.is_none() {
                    handshake_guard
                } else {
                    None
                };
                (
                    engine,
                    lease,
                    shm,
                    attach_writer_gen,
                    attach_visible_commit_lsn,
                    loaded_epoch,
                    attach_oldest_active_txn_lsn,
                    pre_acquire_pin_handle_id,
                    chmod_ro_lock,
                )
            }
            _ => {
                return Err(Error::parse(format!(
                    "Unsupported scheme '{}'. Use 'memory://' or 'file://path'",
                    scheme
                )));
            }
        };

        let semantic_cache = Arc::new(crate::executor::SemanticCache::default());
        let query_planner = Arc::new(crate::executor::QueryPlanner::new(Arc::clone(&engine)));
        // `open_read_only` does NOT install an entry-scoped pin.
        // The ReadOnlyDatabase built below via `from_entry`
        // installs its own per-handle pin at
        // `attach_visible_commit_lsn.max(1)`, which advances
        // forward via `advance_pin_after_refresh` as the reader
        // observes new commits. Holding an additional entry pin
        // at the original attach LSN would clamp the writer's
        // `min_pinned_lsn` to that fixed value for the entry's
        // full lifetime, blocking WAL truncation forever even
        // after the per-handle pin moved on.
        //
        // The plain `Database::open(...?read_only=true)` path
        // DOES install an entry pin because it has no per-handle
        // pin between open and a future `as_read_only()` call,
        // and the entry-pin floor is what makes that wrap safe.
        // Registry-share between the two paths is safe because
        // the `frontier_static` check in `Database::open` refuses
        // to reuse an entry whose `visible_commit_lsn` has moved
        // past attach: no new commits since attach implies the
        // writer's `compute_wal_truncate_floor` cannot move past
        // attach either.
        let entry = Arc::new(EngineEntry {
            engine,
            dsn: dsn.to_string(),
            semantic_cache,
            query_planner,
            lease,
            shm,
            attach_writer_gen,
            attach_visible_commit_lsn,
            attach_oldest_active_txn_lsn,
            loaded_epoch,
            last_lease_touch_ms: std::sync::atomic::AtomicU64::new(0),
            chmod_ro_lock,
            #[cfg(feature = "test-filedb")]
            _temp_dir: _temp_dir_holder,
        });

        registry.insert(dsn.to_string(), Arc::downgrade(&entry));

        let ro_db = crate::api::ReadOnlyDatabase::from_entry(entry.clone());
        // ReadOnlyDatabase::from_entry installed its own
        // per-handle pin at `attach_visible_commit_lsn.max(1)`.
        // That pin is now the protective floor; release the
        // pre-acquire contribution so the writer's truncate
        // cycle isn't held at `1` indefinitely.
        if let Some(ref l) = entry.lease {
            l.remove_handle_pin(pre_acquire_pin_handle_id);
        }
        apply_dsn_flags(ro_db)
    }

    /// Return a read-only view over this database.
    ///
    /// The returned handle shares the same underlying engine and sees the same
    /// committed data. Write SQL submitted through the `ReadOnlyDatabase`
    /// handle is rejected at query time.
    ///
    /// The returned handle holds an Arc to this Database's engine entry, so
    /// the engine stays open as long as either handle is alive.
    ///
    /// # Transaction visibility
    ///
    /// The returned `ReadOnlyDatabase` has its own executor with independent
    /// transaction state — it is a *view*, not a connection sharing this
    /// `Database`'s session. In particular:
    ///
    /// - An uncommitted `BEGIN` on this `Database` (e.g. via [`Self::begin`])
    ///   is **not** visible through the read-only view. Writes inside the
    ///   open transaction are not observed until they commit.
    /// - A `BEGIN` issued via SQL on the read-only view starts a separate
    ///   read-only transaction snapshot; it does not interact with any
    ///   transaction on this `Database`.
    /// - Default isolation level is independent: changing it on one handle
    ///   has no effect on the other.
    ///
    /// If you need a read-only handle that observes uncommitted writes from a
    /// specific transaction, do the read inside that same `Transaction`
    /// (which is gated by the parser at SQL time but allowed for read SQL).
    pub fn as_read_only(&self) -> crate::api::ReadOnlyDatabase {
        crate::api::ReadOnlyDatabase::from_entry(Arc::clone(&self.inner.entry))
    }

    #[cfg(feature = "test-filedb")]
    fn create_in_memory_engine() -> Result<Self> {
        let tmp = tempfile::tempdir()
            .map_err(|e| Error::internal(format!("failed to create temp dir: {}", e)))?;
        let file_dsn = format!("file://{}", tmp.path().display());
        let (_clean_path, config) = Self::parse_file_config(&file_dsn[7..])?;
        let engine = MVCCEngine::new(config);
        engine.open_engine()?;
        let engine = Arc::new(engine);
        engine.start_cleanup();
        let semantic_cache = Arc::new(crate::executor::SemanticCache::default());
        let query_planner = Arc::new(crate::executor::QueryPlanner::new(Arc::clone(&engine)));
        let entry = Arc::new(EngineEntry {
            engine,
            dsn: "memory://".to_string(),
            semantic_cache,
            query_planner,
            lease: None,
            shm: None,
            attach_writer_gen: 0,
            attach_visible_commit_lsn: 0,
            attach_oldest_active_txn_lsn: u64::MAX,
            loaded_epoch: 0,
            last_lease_touch_ms: std::sync::atomic::AtomicU64::new(0),
            chmod_ro_lock: None,
            _temp_dir: Some(tmp),
        });
        Ok(Self::share_entry(entry))
    }

    #[cfg(not(feature = "test-filedb"))]
    fn create_in_memory_engine() -> Result<Self> {
        let engine = MVCCEngine::in_memory();
        engine.open_engine()?;
        let engine = Arc::new(engine);
        engine.start_cleanup();
        let semantic_cache = Arc::new(crate::executor::SemanticCache::default());
        let query_planner = Arc::new(crate::executor::QueryPlanner::new(Arc::clone(&engine)));
        let entry = Arc::new(EngineEntry {
            engine,
            dsn: "memory://".to_string(),
            semantic_cache,
            query_planner,
            // memory:// engines are never read-only and have no
            // disk path, so no SWMR coordination applies.
            lease: None,
            shm: None,
            attach_writer_gen: 0,
            attach_visible_commit_lsn: 0,
            attach_oldest_active_txn_lsn: u64::MAX,
            loaded_epoch: 0,
            last_lease_touch_ms: std::sync::atomic::AtomicU64::new(0),
            chmod_ro_lock: None,
        });
        Ok(Self::share_entry(entry))
    }

    /// Parse a DSN into scheme and path
    /// Parse the DSN's query string for an explicit read-only
    /// flag.  Returns `Ok(None)` when no recognized flag is
    /// present, `Ok(Some(true))` for `?read_only=true` /
    /// `?readonly=true` / `?mode=ro`, and `Ok(Some(false))` for
    /// the writable spellings (`?read_only=false`, `?mode=rw`,
    /// ...).  `Err` when a recognized key has an invalid value
    /// (`?read_only=banana`).
    ///
    /// Used by `Database::open` (rejects `Some(true)` with the
    /// migration message) and `Database::open_read_only`
    /// (rejects `Some(false)` as a contradiction with the
    /// function name; treats `Some(true)` as a redundant no-op).
    ///
    /// Last-flag-wins semantics: the parser scans every param
    /// and records the LAST recognized value, regardless of
    /// which spelling it used. Without this, the same DSN
    /// could pass the API-surface check while
    /// `parse_file_config` interpreted it differently.
    ///
    /// Only the query-string portion (after `?`) is scanned —
    /// substring checks against the whole DSN would catch
    /// `mode=` / `read_only=` inside file paths or unrelated
    /// query values.
    /// Parse `auto_refresh=on/off/true/false/1/0/yes/no` from the DSN
    /// query string. Returns `Ok(None)` when the key is absent so
    /// callers can apply their own default. Same last-flag-wins
    /// semantics as `dsn_read_only_flag`.
    pub(crate) fn dsn_auto_refresh_flag(dsn: &str) -> Result<Option<bool>> {
        let query = match dsn.find('?') {
            Some(idx) => &dsn[idx + 1..],
            None => return Ok(None),
        };
        let mut last: Option<bool> = None;
        for param in query.split('&') {
            let mut parts = param.splitn(2, '=');
            let key = parts.next().unwrap_or("");
            let value = parts.next().unwrap_or("");
            if key == "auto_refresh" {
                last = Some(match value.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => true,
                    "false" | "0" | "no" | "off" => false,
                    _ => {
                        return Err(Error::invalid_argument(format!(
                            "invalid auto_refresh: '{}' (expected true/false/on/off)",
                            value
                        )))
                    }
                });
            }
        }
        Ok(last)
    }

    /// Parse `refresh_interval=Nms|Ns|Nm|0` from the DSN query
    /// string. Returns `Ok(None)` when the key is absent. `0` (with
    /// or without unit) means "no background ticker" and is
    /// returned as `Ok(Some(Duration::ZERO))` so callers can
    /// distinguish "explicitly disabled" from "unset". Last-flag-
    /// wins semantics; rejects unitless numbers and unknown units.
    pub(crate) fn dsn_refresh_interval_flag(dsn: &str) -> Result<Option<Duration>> {
        let query = match dsn.find('?') {
            Some(idx) => &dsn[idx + 1..],
            None => return Ok(None),
        };
        let mut last: Option<Duration> = None;
        for param in query.split('&') {
            let mut parts = param.splitn(2, '=');
            let key = parts.next().unwrap_or("");
            let value = parts.next().unwrap_or("");
            if key == "refresh_interval" {
                last = Some(parse_refresh_interval_value(value)?);
            }
        }
        Ok(last)
    }

    fn dsn_read_only_flag(dsn: &str) -> Result<Option<bool>> {
        let query = match dsn.find('?') {
            Some(idx) => &dsn[idx + 1..],
            None => return Ok(None),
        };
        let mut last: Option<bool> = None;
        for param in query.split('&') {
            let mut parts = param.splitn(2, '=');
            let key = parts.next().unwrap_or("");
            let value = parts.next().unwrap_or("");
            match key {
                "read_only" | "readonly" => {
                    last = Some(match value.to_lowercase().as_str() {
                        "true" | "1" | "yes" | "on" => true,
                        "false" | "0" | "no" | "off" => false,
                        _ => {
                            return Err(Error::invalid_argument(format!(
                                "invalid {}: '{}' (expected true/false)",
                                key, value
                            )))
                        }
                    });
                }
                "mode" => {
                    last = Some(match value.to_lowercase().as_str() {
                        "ro" => true,
                        "rw" => false,
                        _ => {
                            return Err(Error::invalid_argument(format!(
                                "invalid mode: '{}' (expected ro/rw)",
                                value
                            )))
                        }
                    });
                }
                _ => {}
            }
        }
        Ok(last)
    }

    fn parse_dsn(dsn: &str) -> Result<(String, String)> {
        let idx = dsn
            .find("://")
            .ok_or_else(|| Error::parse("Invalid DSN format: expected scheme://path"))?;

        let scheme = dsn[..idx].to_lowercase();
        let path = dsn[idx + 3..].to_string();

        // Validate scheme
        match scheme.as_str() {
            MEMORY_SCHEME | FILE_SCHEME => {}
            _ => {
                return Err(Error::parse(format!(
                    "Unsupported scheme '{}'. Use 'memory://' or 'file://path'",
                    scheme
                )));
            }
        }

        // Validate file path
        if scheme == FILE_SCHEME {
            let clean_path = if path.contains('?') {
                &path[..path.find('?').unwrap()]
            } else {
                &path
            };

            if clean_path.is_empty() {
                return Err(Error::parse("file:// scheme requires a non-empty path"));
            }
        }

        Ok((scheme, path))
    }

    /// Parse file:// config from query parameters
    fn parse_file_config(path: &str) -> Result<(String, Config)> {
        let (clean_path, query) = if let Some(idx) = path.find('?') {
            (path[..idx].to_string(), Some(&path[idx + 1..]))
        } else {
            (path.to_string(), None)
        };

        let mut config = Config::with_path(&clean_path);

        // Parse query parameters
        if let Some(query) = query {
            for param in query.split('&') {
                let mut parts = param.splitn(2, '=');
                let key = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");

                match key {
                    // Sync mode: sync=none|normal|full
                    "sync_mode" | "sync" => {
                        config.persistence.sync_mode = match value.to_lowercase().as_str() {
                            "none" | "off" | "0" => SyncMode::None,
                            "normal" | "1" => SyncMode::Normal,
                            "full" | "2" => SyncMode::Full,
                            _ => SyncMode::Normal,
                        };
                    }
                    // Checkpoint interval in seconds: checkpoint_interval=60
                    // Also accepts snapshot_interval for backward compatibility
                    "checkpoint_interval" | "snapshot_interval" => {
                        config.persistence.checkpoint_interval =
                            value.parse::<u32>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid checkpoint_interval: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Compaction threshold: compact_threshold=4
                    "compact_threshold" => {
                        config.persistence.compact_threshold =
                            value.parse::<u32>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid compact_threshold: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Number of backup snapshots to keep: keep_snapshots=3
                    "keep_snapshots" => {
                        config.persistence.keep_snapshots = value.parse::<u32>().map_err(|_| {
                            Error::invalid_argument(format!("invalid keep_snapshots: '{}'", value))
                        })?;
                    }
                    // WAL flush trigger in bytes: wal_flush_trigger=32768
                    "wal_flush_trigger" => {
                        config.persistence.wal_flush_trigger =
                            value.parse::<usize>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid wal_flush_trigger: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // WAL buffer size in bytes: wal_buffer_size=65536
                    "wal_buffer_size" => {
                        config.persistence.wal_buffer_size =
                            value.parse::<usize>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid wal_buffer_size: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // WAL max size in bytes: wal_max_size=67108864
                    "wal_max_size" => {
                        config.persistence.wal_max_size = value.parse::<usize>().map_err(|_| {
                            Error::invalid_argument(format!("invalid wal_max_size: '{}'", value))
                        })?;
                    }
                    // Commit batch size: commit_batch_size=100
                    "commit_batch_size" => {
                        config.persistence.commit_batch_size =
                            value.parse::<u32>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid commit_batch_size: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Reader lease max age in seconds: lease_max_age=2400
                    // (default 0 = engine-derived `max(120s, 2 * checkpoint_interval)`).
                    // Non-zero overrides for callers running long scans.
                    "lease_max_age" | "lease_max_age_secs" => {
                        config.persistence.lease_max_age_secs =
                            value.parse::<u32>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid lease_max_age: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Open in read-only mode: read_only=true
                    //
                    // When set, the engine acquires the file lock in shared
                    // mode (multiple readers coexist), skips the background
                    // checkpoint thread, and the executor refuses any write
                    // SQL via the parser-level gate plus the DML auto-commit
                    // guard. Equivalent to calling `Database::open_read_only`
                    // except the returned handle has the writable `Database`
                    // type — write attempts fail at runtime with
                    // `Error::ReadOnlyViolation`.
                    "read_only" | "readonly" | "mode" => {
                        // For "mode" the value is "ro" / "rw" (sqlite-style);
                        // for "read_only"/"readonly" it's "true"/"false"/"1"/"0".
                        config.read_only = match value.to_lowercase().as_str() {
                            "true" | "1" | "yes" | "on" | "ro" => true,
                            "false" | "0" | "no" | "off" | "rw" => false,
                            _ => {
                                return Err(Error::invalid_argument(format!(
                                    "invalid {}: '{}' (expected true/false or ro/rw)",
                                    key, value
                                )));
                            }
                        };
                    }
                    // Sync interval in ms: sync_interval_ms=10
                    "sync_interval_ms" | "sync_interval" => {
                        config.persistence.sync_interval_ms =
                            value.parse::<u32>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid sync_interval_ms: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // WAL compression: wal_compression=on|off
                    "wal_compression" => {
                        config.persistence.wal_compression =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    // Volume LZ4 compression: volume_compression=on|off
                    "volume_compression" => {
                        config.persistence.volume_compression =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    // All compressions (WAL + volume): compression=on|off
                    // Also accepts snapshot_compression for backward compatibility
                    "compression" | "snapshot_compression" => {
                        let enabled =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                        config.persistence.wal_compression = enabled;
                        config.persistence.volume_compression = enabled;
                    }
                    // Compression threshold in bytes: compression_threshold=64
                    "compression_threshold" => {
                        config.persistence.compression_threshold =
                            value.parse::<usize>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid compression_threshold: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Target rows per volume: target_volume_rows=1048576
                    "target_volume_rows" => {
                        let rows = value.parse::<usize>().map_err(|_| {
                            Error::invalid_argument(format!(
                                "invalid target_volume_rows: '{}'",
                                value
                            ))
                        })?;
                        config.persistence.target_volume_rows = rows.max(65_536);
                    }
                    // Checkpoint on close: checkpoint_on_close=off
                    // Set to off to simulate crashes in tests (WAL not truncated)
                    "checkpoint_on_close" => {
                        config.persistence.checkpoint_on_close =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    // Cleanup interval in seconds: cleanup_interval=60
                    "cleanup_interval" => {
                        config.cleanup.interval_secs = value.parse::<u64>().map_err(|_| {
                            Error::invalid_argument(format!(
                                "invalid cleanup_interval: '{}'",
                                value
                            ))
                        })?;
                    }
                    // Deleted row retention in seconds: deleted_row_retention=300
                    "deleted_row_retention" => {
                        config.cleanup.deleted_row_retention_secs =
                            value.parse::<u64>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid deleted_row_retention: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Transaction retention in seconds: transaction_retention=3600
                    "transaction_retention" => {
                        config.cleanup.transaction_retention_secs =
                            value.parse::<u64>().map_err(|_| {
                                Error::invalid_argument(format!(
                                    "invalid transaction_retention: '{}'",
                                    value
                                ))
                            })?;
                    }
                    // Disable cleanup: cleanup=off
                    "cleanup" => {
                        config.cleanup.enabled =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    _ => {} // Ignore unknown parameters
                }
            }
        }

        Ok((clean_path, config))
    }

    /// Execute a SQL statement
    ///
    /// Use this for DDL (CREATE, DROP, ALTER) and DML (INSERT, UPDATE, DELETE) statements.
    ///
    /// # Parameters
    ///
    /// Parameters can be passed using:
    /// - Empty tuple `()` for no parameters
    /// - Tuple syntax `(1, "Alice", 30)` for multiple parameters
    /// - `params!` macro `params![1, "Alice", 30]`
    ///
    /// # Returns
    ///
    /// Returns the number of rows affected for DML statements, or 0 for DDL.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // DDL - no parameters
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    ///
    /// // DML with tuple parameters
    /// db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
    ///
    /// // DML with params! macro
    /// db.execute("INSERT INTO users VALUES ($1, $2)", params![2, "Bob"])?;
    ///
    /// // Update with mixed types
    /// let affected = db.execute(
    ///     "UPDATE users SET name = $1 WHERE id = $2",
    ///     ("Charlie", 1)
    /// )?;
    /// ```
    /// Per-call SWMR maintenance for any query/execute entry
    /// point on this `Database`. Bumps the lease mtime so the
    /// writer's reaper doesn't declare us stale.
    ///
    /// `Database` is always writable. For writable engines the
    /// heartbeat is a no-op (no lease to bump). Cross-process
    /// read-only handles go through `ReadOnlyDatabase` (opened
    /// via `Database::open_read_only`), which has its own
    /// refresh + heartbeat path on `maybe_auto_refresh`.
    #[inline]
    pub(crate) fn heartbeat_and_maybe_refresh(&self) -> Result<()> {
        self.inner.entry.heartbeat_swmr_lease();
        Ok(())
    }

    pub fn execute<P: Params>(&self, sql: &str, params: P) -> Result<i64> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let param_values = params.into_params();
        let result = if param_values.is_empty() {
            executor.execute(sql)?
        } else if let Some(fast_result) = executor.try_fast_path_with_params(sql, &param_values) {
            fast_result?
        } else {
            executor.execute_with_params(sql, param_values)?
        };
        Ok(result.rows_affected())
    }

    /// Execute a query that returns rows
    ///
    /// # Parameters
    ///
    /// Parameters can be passed using:
    /// - Empty tuple `()` for no parameters
    /// - Tuple syntax `(value,)` for single parameter (note trailing comma)
    /// - Tuple syntax `(1, "Alice")` for multiple parameters
    /// - `params!` macro `params![1, "Alice"]`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Query all rows
    /// for row in db.query("SELECT * FROM users", ())? {
    ///     let row = row?;
    ///     let id: i64 = row.get(0)?;
    ///     let name: String = row.get("name")?;
    /// }
    ///
    /// // Query with parameters
    /// for row in db.query("SELECT * FROM users WHERE age > $1", (18,))? {
    ///     // ...
    /// }
    ///
    /// // Collect into Vec
    /// let users: Vec<_> = db.query("SELECT * FROM users", ())?
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// ```
    pub fn query<P: Params>(&self, sql: &str, params: P) -> Result<Rows> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let param_values = params.into_params();
        let result = if param_values.is_empty() {
            executor.execute(sql)?
        } else if let Some(fast_result) = executor.try_fast_path_with_params(sql, &param_values) {
            fast_result?
        } else {
            executor.execute_with_params(sql, param_values)?
        };
        Ok(Rows::new(result))
    }

    /// Execute a query and return a single value
    ///
    /// This is a convenience method for queries that return a single row with a single column.
    /// Returns an error if the query returns no rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
    /// let name: String = db.query_one("SELECT name FROM users WHERE id = $1", (1,))?;
    /// ```
    pub fn query_one<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<T> {
        let row = self
            .query(sql, params)?
            .next()
            .ok_or(Error::NoRowsReturned)??;
        row.get(0)
    }

    /// Execute a query and return an optional single value
    ///
    /// Like `query_one`, but returns `None` if no rows are returned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let name: Option<String> = db.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;
    /// assert!(name.is_none());
    /// ```
    pub fn query_opt<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<Option<T>> {
        match self.query(sql, params)?.next() {
            Some(row) => Ok(Some(row?.get(0)?)),
            None => Ok(None),
        }
    }

    /// Execute a write statement with a timeout
    ///
    /// Like `execute`, but cancels the query if it exceeds the timeout.
    /// Timeout is specified in milliseconds. Use 0 for no timeout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Execute with 5 second timeout
    /// db.execute_with_timeout("DELETE FROM large_table WHERE old = true", (), 5000)?;
    /// ```
    pub fn execute_with_timeout<P: Params>(
        &self,
        sql: &str,
        params: P,
        timeout_ms: u64,
    ) -> Result<i64> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let param_values = params.into_params();
        let ctx = ExecutionContextBuilder::new()
            .params(param_values)
            .timeout_ms(timeout_ms)
            .build();

        let result = executor.execute_with_context(sql, &ctx)?;
        Ok(result.rows_affected())
    }

    /// Execute a query with a timeout
    ///
    /// Like `query`, but cancels the query if it exceeds the timeout.
    /// Timeout is specified in milliseconds. Use 0 for no timeout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Query with 10 second timeout
    /// for row in db.query_with_timeout("SELECT * FROM large_table", (), 10000)? {
    ///     // process row
    /// }
    /// ```
    pub fn query_with_timeout<P: Params>(
        &self,
        sql: &str,
        params: P,
        timeout_ms: u64,
    ) -> Result<Rows> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let param_values = params.into_params();
        let ctx = ExecutionContextBuilder::new()
            .params(param_values)
            .timeout_ms(timeout_ms)
            .build();

        let result = executor.execute_with_context(sql, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Prepare a SQL statement for repeated execution
    ///
    /// Prepared statements are more efficient when executing the same query
    /// multiple times with different parameters.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;
    ///
    /// // Execute multiple times with different parameters
    /// for id in 1..=10 {
    ///     for row in stmt.query((id,))? {
    ///         // ...
    ///     }
    /// }
    /// ```
    pub fn prepare(&self, sql: &str) -> Result<Statement> {
        Statement::new(Arc::downgrade(&self.inner), sql.to_string(), self)
    }

    /// Create a Database from an existing Arc<DatabaseInner>.
    /// Used by Statement to upgrade weak references.
    pub(crate) fn from_inner(inner: Arc<DatabaseInner>) -> Self {
        Database { inner }
    }

    /// Execute a statement with named parameters
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;
    ///
    /// // Insert with named params
    /// db.execute_named(
    ///     "INSERT INTO users VALUES (:id, :name, :age)",
    ///     named_params!{ id: 1, name: "Alice", age: 30 }
    /// )?;
    ///
    /// // Update with named params
    /// db.execute_named(
    ///     "UPDATE users SET name = :name WHERE id = :id",
    ///     named_params!{ id: 1, name: "Alicia" }
    /// )?;
    /// ```
    pub fn execute_named(&self, sql: &str, params: NamedParams) -> Result<i64> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(result.rows_affected())
    }

    /// Execute a query with named parameters
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    /// db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())?;
    ///
    /// // Query with named params
    /// for row in db.query_named(
    ///     "SELECT * FROM users WHERE name = :name",
    ///     named_params!{ name: "Alice" }
    /// )? {
    ///     let row = row?;
    ///     println!("Found user: id={}", row.get::<i64>(0)?);
    /// }
    /// ```
    pub fn query_named(&self, sql: &str, params: NamedParams) -> Result<Rows> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(Rows::new(result))
    }

    /// Execute a query with named parameters and return a single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let count: i64 = db.query_one_named(
    ///     "SELECT COUNT(*) FROM users WHERE age > :min_age",
    ///     named_params!{ min_age: 18 }
    /// )?;
    /// ```
    pub fn query_one_named<T: FromValue>(&self, sql: &str, params: NamedParams) -> Result<T> {
        let mut rows = self.query_named(sql, params)?;
        match rows.next() {
            Some(Ok(row)) => row.get(0),
            Some(Err(e)) => Err(e),
            None => Err(Error::NoRowsReturned),
        }
    }

    /// Execute a query and map results to structs
    ///
    /// This method executes a query and converts each row to a struct
    /// that implements the `FromRow` trait.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, FromRow, ResultRow, Result};
    ///
    /// struct User {
    ///     id: i64,
    ///     name: String,
    /// }
    ///
    /// impl FromRow for User {
    ///     fn from_row(row: &ResultRow) -> Result<Self> {
    ///         Ok(User {
    ///             id: row.get(0)?,
    ///             name: row.get(1)?,
    ///         })
    ///     }
    /// }
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    /// db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())?;
    ///
    /// // Query and map to structs
    /// let users: Vec<User> = db.query_as("SELECT id, name FROM users", ())?;
    /// assert_eq!(users.len(), 2);
    /// assert_eq!(users[0].name, "Alice");
    /// ```
    pub fn query_as<T: FromRow, P: Params>(&self, sql: &str, params: P) -> Result<Vec<T>> {
        let rows = self.query(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Execute a query with named parameters and map results to structs
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, FromRow, ResultRow, Result, named_params};
    ///
    /// struct Product {
    ///     id: i64,
    ///     name: String,
    ///     price: f64,
    /// }
    ///
    /// impl FromRow for Product {
    ///     fn from_row(row: &ResultRow) -> Result<Self> {
    ///         Ok(Product {
    ///             id: row.get(0)?,
    ///             name: row.get(1)?,
    ///             price: row.get(2)?,
    ///         })
    ///     }
    /// }
    ///
    /// let products: Vec<Product> = db.query_as_named(
    ///     "SELECT id, name, price FROM products WHERE price > :min_price",
    ///     named_params!{ min_price: 10.0 }
    /// )?;
    /// ```
    pub fn query_as_named<T: FromRow>(&self, sql: &str, params: NamedParams) -> Result<Vec<T>> {
        let rows = self.query_named(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Begin a new transaction with default isolation level
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tx = db.begin()?;
    /// tx.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
    /// tx.commit()?;
    /// ```
    pub fn begin(&self) -> Result<Transaction> {
        self.begin_with_isolation(IsolationLevel::ReadCommitted)
    }

    /// Begin a new transaction with a specific isolation level
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::IsolationLevel;
    ///
    /// let tx = db.begin_with_isolation(IsolationLevel::Snapshot)?;
    /// // All reads in this transaction see a consistent snapshot
    /// tx.execute("UPDATE users SET balance = balance - 100 WHERE id = $1", (1,))?;
    /// tx.commit()?;
    /// ```
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<Transaction> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let tx = executor.begin_transaction_with_isolation(isolation)?;
        // Pass the engine entry (not just the engine Arc) so live
        // transactions count toward `Arc::strong_count(&entry)`. Without
        // this, `db.close()` could fire `engine.close_engine()` while a
        // transaction is alive — close() would see the entry's count as 1
        // (only db.inner.entry) and conclude no other peer needs the
        // engine. The transaction's `Arc<MVCCEngine>` clone wouldn't
        // affect that count, leaving the txn with a closed engine.
        let entry = Arc::clone(&self.inner.entry);
        Ok(Transaction::new(tx, entry))
    }

    /// Get the underlying storage engine
    ///
    /// This is primarily for advanced use cases and testing.
    ///
    /// # Read-only handles
    ///
    /// On a `Database` opened with `?read_only=true` / `?mode=ro`, every
    /// write-intent method on the returned `MVCCEngine` is gated and
    /// returns `Error::ReadOnlyViolation`:
    ///
    /// - `Engine::begin_transaction` / `begin_transaction_with_level`
    ///   (the trait methods reachable through `engine.begin_transaction()`).
    /// - `Engine::create_snapshot` / `restore_snapshot`.
    /// - `MVCCEngine::create_table`, `drop_table_internal`, `create_view`,
    ///   `drop_view`, `rename_table`, `create_column`,
    ///   `create_column_with_default`, `drop_column`, `rename_column`,
    ///   `modify_column`, `update_engine_config`, `vacuum`.
    /// - `MVCCEngine::cleanup_old_transactions`,
    ///   `cleanup_deleted_rows`, `cleanup_old_previous_versions` are
    ///   silent no-ops returning `0` on read-only engines.
    /// - `MVCCEngine::start_periodic_cleanup` returns a no-op
    ///   `CleanupHandle` (no thread is spawned).
    ///
    /// Other engine methods (`is_open`, `is_read_only_mode`, `path`,
    /// `volume_stats`, `config`, view lookup, `oldest_loaded_snapshot_timestamp`,
    /// the `ReadEngine::begin_read_transaction*` family) work normally on
    /// both writable and read-only handles.
    ///
    /// Internal-only methods like `propagate_column_*`,
    /// `refresh_schema_cache`, `modify_column_with_dimensions`,
    /// `get_table_for_txn`, `find_referencing_fks`, `get_version_store`
    /// are not part of the public surface — they are `pub(crate)` and
    /// not reachable through this accessor.
    pub fn engine(&self) -> &Arc<MVCCEngine> {
        &self.inner.entry.engine
    }

    /// Returns `true` if this `Database` was opened in read-only mode
    /// (`?read_only=true` / `?mode=ro`).
    ///
    /// Equivalent to `db.engine().is_read_only_mode()` — provided as a
    /// direct accessor so callers don't have to reach into the engine.
    /// Useful for branching in user code that wants to skip work it
    /// knows would be refused (e.g. issuing PRAGMA SNAPSHOT only when
    /// writable).
    pub fn is_read_only(&self) -> bool {
        self.inner.entry.engine.is_read_only_mode()
    }

    /// Get the engine as a read-only trait object.
    ///
    /// Returns `Arc<dyn ReadEngine>` instead of the concrete `Arc<MVCCEngine>`
    /// returned by [`Self::engine`]. The trait object exposes only
    /// `begin_read_transaction` / `begin_read_transaction_with_level`
    /// (plus `Engine::table_exists` via the supertrait). Callers holding
    /// the trait object cannot reach `Engine::begin_transaction` or any
    /// inherent write method on `MVCCEngine` — the read-only contract is
    /// enforced at the type level rather than at runtime.
    ///
    /// Works on writable Databases too: returning the read surface is
    /// always safe regardless of mode. Cheap (one Arc clone). Use this
    /// in libraries that want to accept "any database that can serve
    /// reads" without coupling to the writable surface.
    pub fn read_engine(&self) -> Arc<dyn crate::storage::traits::ReadEngine> {
        // Wrap the engine in a `SwmrReadEngineGuard` so every
        // `begin_read_transaction*` call through the trait
        // object first runs the same SWMR maintenance the
        // public `query` / `execute` paths do: lease heartbeat
        // (so a long-lived caller isn't reaped as stale) and —
        // for read-only handles — drive the per-handle refresh
        // (manifest reload + cache invalidation + writer-gen
        // reincarnation) so writer-side CREATE/DROP/checkpoint
        // is observed. Without the wrapper the trait object
        // went straight to `MVCCEngine::begin_read_transaction*`,
        // leaving a documented "safe read surface" that
        // silently kept returning stale state and let the
        // writer reap the lease.
        // `Database` is always writable, so the guard's refresh
        // owner is `None` (no overlay, no manifest poll, no
        // sticky DDL — those belong to `ReadOnlyDatabase`). The
        // guard still wraps the engine so a future `Database`
        // type with refresh needs has a stable extension point,
        // and so the heartbeat call still routes through here.
        Arc::new(SwmrReadEngineGuard {
            engine: Arc::clone(&self.inner.entry.engine),
            entry: Arc::clone(&self.inner.entry),
            refresh_owner: RefreshOwner::None,
        }) as Arc<dyn crate::storage::traits::ReadEngine>
    }

    /// Close the database connection
    ///
    /// When this handle is the last one for its DSN, closes the engine
    /// immediately so the file lock is released for other processes. If
    /// another `Database` clone, sibling `Database::open(dsn)` handle, or
    /// `ReadOnlyDatabase` view still references the same engine, the close
    /// is *deferred* until that last handle drops. This preserves the
    /// lifetime contract for `as_read_only()` / `open_read_only()` and
    /// makes `close()` safe to call on one of several handles without
    /// pulling the rug out from under in-flight queries on the others.
    ///
    /// Note: The engine is also closed automatically when the last handle
    /// is dropped.
    pub fn close(&self) -> Result<()> {
        // `Database` is always writable now: no `ReaderAttachment`,
        // no per-handle pin, no baseline math. The strong_count
        // check is the simple form — if any sibling
        // `Database::open(dsn)` clone, `Database::clone()`, or
        // shared `ReadOnlyDatabase` view still references the
        // entry, defer; otherwise close.
        //
        // The check runs under the registry write lock so a
        // concurrent `Database::open(dsn)` can't upgrade the
        // still-live Weak between our decision and our
        // `close_engine()` call.
        let mut registry = match DATABASE_REGISTRY.write() {
            Ok(g) => g,
            Err(_) => return Err(Error::LockAcquisitionFailed("registry write".to_string())),
        };
        if Arc::strong_count(&self.inner.entry) > 1 {
            // Sibling handle alive — defer. Registry stays intact.
            return Ok(());
        }
        // Commit path: clear the registry's dead-soon Weak so
        // the next `open(dsn)` doesn't have to upgrade-and-fail,
        // drop the lock, then close the engine. The
        // `is_open()` checks inside the transaction-begin paths
        // cover the residual race where `read_engine()` clones
        // the entry between this check and `close_engine` —
        // they surface `EngineNotOpen` as the soft contract
        // for that vanishing window.
        if let Some(weak) = registry.get(&self.inner.entry.dsn) {
            let same = weak
                .upgrade()
                .map(|reg| Arc::ptr_eq(&reg, &self.inner.entry))
                .unwrap_or(true);
            if same {
                registry.remove(&self.inner.entry.dsn);
            }
        }
        drop(registry);
        // Idempotent — safe to call multiple times.
        self.inner.entry.engine.close_engine()?;

        Ok(())
    }

    /// Get a cached plan for a SQL statement (parse once, execute many times).
    ///
    /// Returns a `CachedPlanRef` that can be stored and passed to
    /// `execute_plan()` / `query_plan()` for zero-lookup execution.
    pub fn cached_plan(&self, sql: &str) -> Result<CachedPlanRef> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        executor.get_or_create_plan(sql)
    }

    /// Execute a pre-cached plan with positional parameters (no parsing, no cache lookup).
    pub fn execute_plan<P: Params>(&self, plan: &CachedPlanRef, params: P) -> Result<i64> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(result.rows_affected())
    }

    /// Query using a pre-cached plan with positional parameters (no parsing, no cache lookup).
    pub fn query_plan<P: Params>(&self, plan: &CachedPlanRef, params: P) -> Result<Rows> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Execute a pre-cached plan with named parameters (no parsing, no cache lookup).
    pub fn execute_named_plan(&self, plan: &CachedPlanRef, params: NamedParams) -> Result<i64> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(result.rows_affected())
    }

    /// Query using a pre-cached plan with named parameters (no parsing, no cache lookup).
    pub fn query_named_plan(&self, plan: &CachedPlanRef, params: NamedParams) -> Result<Rows> {
        self.heartbeat_and_maybe_refresh()?;
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Check if a table exists
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        use crate::storage::traits::ReadEngine;
        // Run the same SWMR maintenance the query/execute paths
        // do: heartbeat the lease (so a `table_exists`-only
        // poller doesn't get reaped) and — for SWMR-eligible
        // read-only handles — refresh manifests/WAL so a
        // writer-side CREATE/DROP/checkpoint is observed instead
        // of returning the handle's stale schema cache. Surfaces
        // typed must-reopen errors (`SchemaChanged`,
        // `SwmrPendingDdl`, ...) too.
        self.heartbeat_and_maybe_refresh()?;
        // Read-only path: a `ReadTransaction` is enough for `get_read_table`,
        // and it works on both writable and read-only engines without any
        // gate bypass.
        let engine = &self.inner.entry.engine;
        let tx = ReadEngine::begin_read_transaction(engine.as_ref())?;
        Ok(tx.get_read_table(name).is_ok())
    }

    /// Returns the number of rows in `name` visible to this autocommit handle.
    ///
    /// Goes through the same read-transaction path as `SELECT COUNT(*)` so
    /// hot rows, sealed cold volumes, and pending tombstones are all
    /// accounted for. The fast path is O(1) (atomic load of the per-segment
    /// deduped counter plus the hot committed counter); the slow fallback
    /// fires only under snapshot-isolation autocommit, which is rare.
    ///
    /// For row counts visible to a specific in-flight transaction (including
    /// its own uncommitted INSERTs/DELETEs) use [`Transaction::table_count`].
    pub fn table_count(&self, name: &str) -> Result<u64> {
        use crate::storage::traits::ReadEngine;
        let engine = &self.inner.entry.engine;
        let tx = ReadEngine::begin_read_transaction(engine.as_ref())?;
        let table = tx.get_read_table(name)?;
        if let Some(c) = table.fast_row_count() {
            return Ok(c as u64);
        }
        Ok(table.row_count() as u64)
    }

    /// Get the DSN this database was opened with
    pub fn dsn(&self) -> &str {
        &self.inner.entry.dsn
    }

    /// Set the default isolation level for new transactions
    pub fn set_default_isolation_level(&self, level: IsolationLevel) -> Result<()> {
        let mut executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        executor.set_default_isolation_level(level);
        Ok(())
    }

    /// Create a backup snapshot of the database
    ///
    /// This creates snapshot files (.bin) for each table along with
    /// index/view definitions (ddl-{timestamp}.bin) for disaster recovery.
    /// Normal persistence uses the checkpoint cycle (seal to volumes + WAL).
    ///
    /// Note: This is a no-op for in-memory databases.
    ///
    /// Returns `Error::ReadOnlyViolation` when called on a read-only handle
    /// (`?read_only=true` / `?mode=ro`). The engine layer also refuses, but
    /// catching it here keeps the error message tied to the user-facing
    /// `Database::create_snapshot` rather than the lower-level
    /// `MVCCEngine::create_snapshot`.
    pub fn create_snapshot(&self) -> Result<()> {
        use crate::storage::Engine;
        if self.inner.entry.engine.is_read_only_mode() {
            return Err(Error::read_only_violation_at("database", "create_snapshot"));
        }
        self.inner.entry.engine.create_snapshot()
    }

    /// Restore the database from a backup snapshot.
    ///
    /// If no timestamp is provided, restores from the latest snapshot.
    /// If a timestamp is provided (format: "YYYYMMDD-HHMMSS.fff"),
    /// restores from that specific snapshot.
    ///
    /// This is a destructive operation that replaces all current data
    /// with the snapshot data. Indexes and views are restored from
    /// ddl-{timestamp}.bin or preserved from current in-memory state.
    ///
    /// Returns `Error::ReadOnlyViolation` when called on a read-only handle
    /// (`?read_only=true` / `?mode=ro`). Restore overwrites engine state
    /// in place, which is fundamentally incompatible with the read-only
    /// contract regardless of the on-disk write permissions.
    pub fn restore_snapshot(&self, timestamp: Option<&str>) -> Result<String> {
        use crate::storage::Engine;
        if self.inner.entry.engine.is_read_only_mode() {
            return Err(Error::read_only_violation_at(
                "database",
                "restore_snapshot",
            ));
        }
        let result = self.inner.entry.engine.restore_snapshot(timestamp)?;
        // Clear all query caches since all data has changed.
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        executor.clear_semantic_cache();
        crate::executor::context::clear_scalar_subquery_cache();
        crate::executor::context::clear_in_subquery_cache();
        crate::executor::context::clear_semi_join_cache();
        Ok(result)
    }

    /// Get the internal executor (for Statement use)
    pub(crate) fn executor(&self) -> &Arc<Mutex<Executor>> {
        &self.inner.executor
    }

    /// Get semantic cache statistics
    ///
    /// Returns statistics about the semantic query cache including hit rates,
    /// exact matches, and subsumption matches.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let db = Database::open("memory://")?;
    /// // ... execute some queries ...
    /// let stats = db.semantic_cache_stats()?;
    /// println!("Cache hits: {}", stats.hits);
    /// println!("Subsumption hits: {}", stats.subsumption_hits);
    /// ```
    pub fn semantic_cache_stats(&self) -> Result<crate::executor::SemanticCacheStatsSnapshot> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        Ok(executor.semantic_cache_stats())
    }

    /// Clear the semantic cache
    ///
    /// This clears all cached query results. Useful for testing or when
    /// you want to force queries to re-execute.
    pub fn clear_semantic_cache(&self) -> Result<()> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        executor.clear_semantic_cache();
        Ok(())
    }

    /// Get the oldest snapshot timestamp loaded during startup.
    /// Returns None if no snapshots were loaded.
    pub fn oldest_loaded_snapshot_timestamp(&self) -> Option<String> {
        self.inner.entry.engine.oldest_loaded_snapshot_timestamp()
    }
}

/// Trait for converting from Value to a Rust type
pub trait FromValue: Sized {
    /// Convert a Value to Self
    fn from_value(value: &Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Integer".to_string(),
            }),
        }
    }
}

impl FromValue for i32 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Integer(i) => Ok(*i as i32),
            Value::Float(f) => Ok(*f as i32),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Integer".to_string(),
            }),
        }
    }
}

impl FromValue for f64 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Integer(i) => Ok(*i as f64),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Float".to_string(),
            }),
        }
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Text(s) => Ok(s.to_string()),
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                Ok(std::str::from_utf8(&data[1..]).unwrap_or("").to_string())
            }
            // Convert other types to string representation
            Value::Integer(i) => Ok(i.to_string()),
            Value::Float(f) => Ok(f.to_string()),
            Value::Boolean(b) => Ok(if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }),
            Value::Timestamp(ts) => Ok(ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            Value::Extension(_) => value
                .as_string()
                .ok_or_else(|| Error::invalid_argument("Cannot convert extension to String")),
            Value::Null(_) => Ok(String::new()),
        }
    }
}

impl FromValue for bool {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Boolean(b) => Ok(*b),
            Value::Integer(i) => Ok(*i != 0),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Boolean".to_string(),
            }),
        }
    }
}

impl FromValue for Value {
    fn from_value(value: &Value) -> Result<Self> {
        Ok(value.clone())
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            Ok(None)
        } else {
            Ok(Some(T::from_value(value)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::named_params;

    #[test]
    fn test_open_memory() {
        let db = Database::open("memory://").unwrap();
        assert_eq!(db.dsn(), "memory://");
    }

    #[test]
    fn test_open_in_memory() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1)", (1,)).unwrap();

        for row in db.query("SELECT * FROM test", ()).unwrap() {
            let row = row.unwrap();
            let id: i64 = row.get(0).unwrap();
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn test_execute_and_query_new_api() {
        let db = Database::open_in_memory().unwrap();

        // Create table - no params
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            (),
        )
        .unwrap();

        // Insert with tuple params
        let affected = db
            .execute(
                "INSERT INTO users VALUES ($1, $2, $3), ($4, $5, $6)",
                (1, "Alice", 30, 2, "Bob", 25),
            )
            .unwrap();
        assert_eq!(affected, 2);

        // Query with tuple params
        let rows: Vec<_> = db
            .query("SELECT * FROM users ORDER BY id", ())
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<i64>(0).unwrap(), 1);
        assert_eq!(rows[0].get::<String>(1).unwrap(), "Alice");
        assert_eq!(rows[0].get::<i64>(2).unwrap(), 30);
    }

    #[test]
    fn test_query_one() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1), ($2), ($3)", (1, 2, 3))
            .unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_query_opt() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1)", (1,)).unwrap();

        // Found
        let result: Option<i64> = db
            .query_opt("SELECT id FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(result, Some(1));

        // Not found
        let result: Option<i64> = db
            .query_opt("SELECT id FROM test WHERE id = $1", (999,))
            .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_params_macro() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();

        // Use params! macro
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            crate::params![1, "Alice"],
        )
        .unwrap();

        let names: Vec<String> = db
            .query("SELECT name FROM users WHERE id = $1", crate::params![1])
            .unwrap()
            .map(|r| r.and_then(|row| row.get(0)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(names, vec!["Alice"]);
    }

    #[test]
    fn test_parse_dsn() {
        // Memory
        let (scheme, path) = Database::parse_dsn("memory://").unwrap();
        assert_eq!(scheme, "memory");
        assert_eq!(path, "");

        // File
        let (scheme, path) = Database::parse_dsn("file:///tmp/test.db").unwrap();
        assert_eq!(scheme, "file");
        assert_eq!(path, "/tmp/test.db");

        // File with params
        let (scheme, path) = Database::parse_dsn("file:///tmp/test.db?sync=full").unwrap();
        assert_eq!(scheme, "file");
        assert_eq!(path, "/tmp/test.db?sync=full");

        // Invalid
        assert!(Database::parse_dsn("invalid").is_err());
        assert!(Database::parse_dsn("unknown://test").is_err());
    }

    #[test]
    fn test_from_value_types() {
        assert_eq!(i64::from_value(&Value::Integer(42)).unwrap(), 42);
        assert_eq!(f64::from_value(&Value::Float(3.5)).unwrap(), 3.5);
        assert_eq!(
            String::from_value(&Value::Text("hello".into())).unwrap(),
            "hello"
        );
        assert!(bool::from_value(&Value::Boolean(true)).unwrap());

        // Optional
        assert_eq!(
            Option::<i64>::from_value(&Value::Integer(42)).unwrap(),
            Some(42)
        );
        assert_eq!(
            Option::<i64>::from_value(&Value::null_unknown()).unwrap(),
            None
        );
    }

    #[test]
    fn test_cached_plan_insert_and_query() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, score FLOAT)",
            (),
        )
        .unwrap();

        let insert_plan = db
            .cached_plan("INSERT INTO test VALUES ($1, $2, $3)")
            .unwrap();

        // Batch insert using cached plan
        db.execute_plan(&insert_plan, (1, "Alice", 95.5)).unwrap();
        db.execute_plan(&insert_plan, (2, "Bob", 82.0)).unwrap();
        db.execute_plan(&insert_plan, (3, "Charlie", 91.0)).unwrap();

        // Query using cached plan
        let query_plan = db
            .cached_plan("SELECT name FROM test WHERE id = $1")
            .unwrap();
        let mut rows = db.query_plan(&query_plan, (2,)).unwrap();
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<String>(0).unwrap(), "Bob");
    }

    #[test]
    fn test_cached_plan_reuse() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();

        // Get the same plan twice — second call should hit the cache
        let plan1 = db.cached_plan("INSERT INTO test VALUES ($1, $2)").unwrap();
        let plan2 = db.cached_plan("INSERT INTO test VALUES ($1, $2)").unwrap();

        // Both should work independently
        db.execute_plan(&plan1, (1, 100)).unwrap();
        db.execute_plan(&plan2, (2, 200)).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_cached_plan_update_delete() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES (1, 100)", ()).unwrap();
        db.execute("INSERT INTO test VALUES (2, 200)", ()).unwrap();

        // Update via cached plan
        let update_plan = db
            .cached_plan("UPDATE test SET value = $1 WHERE id = $2")
            .unwrap();
        let affected = db.execute_plan(&update_plan, (999, 1)).unwrap();
        assert_eq!(affected, 1);

        let val: i64 = db
            .query_one("SELECT value FROM test WHERE id = 1", ())
            .unwrap();
        assert_eq!(val, 999);

        // Delete via cached plan
        let delete_plan = db.cached_plan("DELETE FROM test WHERE id = $1").unwrap();
        let affected = db.execute_plan(&delete_plan, (2,)).unwrap();
        assert_eq!(affected, 1);

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_cached_plan_no_params() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES (1, 10)", ()).unwrap();
        db.execute("INSERT INTO test VALUES (2, 20)", ()).unwrap();

        let plan = db.cached_plan("SELECT COUNT(*) FROM test").unwrap();
        let mut rows = db.query_plan(&plan, ()).unwrap();
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(0).unwrap(), 2);
    }

    #[test]
    fn test_cached_plan_named_params() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();

        let plan = db
            .cached_plan("INSERT INTO test VALUES (:id, :name)")
            .unwrap();
        db.execute_named_plan(&plan, named_params! { id: 1, name: "Alice" })
            .unwrap();
        db.execute_named_plan(&plan, named_params! { id: 2, name: "Bob" })
            .unwrap();

        let query_plan = db
            .cached_plan("SELECT name FROM test WHERE id = :id")
            .unwrap();
        let mut rows = db
            .query_named_plan(&query_plan, named_params! { id: 1 })
            .unwrap();
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<String>(0).unwrap(), "Alice");
    }

    #[test]
    fn test_cached_plan_multi_statement_error() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();

        // Multiple statements should fail
        let result = db.cached_plan("INSERT INTO test VALUES (1); INSERT INTO test VALUES (2)");
        assert!(result.is_err());
    }
}
