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

//! Read-only database handle.
//!
//! Provides a read-only view over an existing database. Write SQL is
//! rejected at the parser-level gate.
//!
//! # SWMR visibility model
//!
//! Stoolap supports **single-writer, multi-reader** access across
//! processes:
//!
//! - **Coexistence**: a read-only attach via `Database::open_read_only`
//!   takes `LOCK_SH` on `db.lock`. Multiple `LOCK_SH` readers coexist
//!   with each other; a single `LOCK_EX` writer coexists with readers
//!   on platforms where the writer downgrades after init handshake.
//!
//! - **Lease + WAL pin**: each reader process registers its presence
//!   at `<db>/readers/<pid>.lease`. The 8-byte payload is the WAL LSN
//!   the reader still needs (`pinned_lsn`). The writer:
//!     - Defers destructive volume unlinks while any live lease exists.
//!     - Floors WAL truncation at `min_pinned_lsn - 1` across leases.
//!
//! - **Two visibility tiers**:
//!     1. **Manifest-epoch poll** (always): the reader auto-polls
//!        `<db>/volumes/epoch` on every query (one cheap `read()` of
//!        8 bytes) and reloads per-table manifests on advance. This
//!        picks up the writer's checkpoint output (cold rows, sealed
//!        segments). Visibility lag is bounded by the writer's
//!        checkpoint cadence (default 60s).
//!     2. **WAL-tail overlay** (sub-checkpoint, opt-in for query
//!        materialization but ALWAYS on for DDL detection): the
//!        reader tails the writer's WAL using `db.shm`'s published
//!        `visible_commit_lsn`. Post-attach DDL surfaces as
//!        `Error::SwmrPendingDdl` so the caller can reopen.
//!        Per-row DML materialization is gated on
//!        `set_swmr_overlay_enabled(true)` until the scanner
//!        integration ships.
//!
//! - **Writer reincarnation**: `db.shm` carries a writer generation
//!   counter. Every refresh compares the observed value against the
//!   one captured at attach; a mismatch (writer crashed and recovered,
//!   or closed and reopened) surfaces `Error::SwmrWriterReincarnated`
//!   so the caller hard-reopens against the new incarnation.
//!
//! ## What the user must do
//!
//! - **Touch cadence**: any query touches the lease (rate-limited to
//!   one syscall per second). Callers that go silent longer than
//!   `max(120s, 2 * checkpoint_interval)` risk being reaped as stale,
//!   after which the writer may unlink old compacted volumes the
//!   reader's cached manifest still points at and a subsequent query
//!   hits "volume not found". Mitigations: a longer
//!   `checkpoint_interval` (e.g. `?checkpoint_interval=300`), or a
//!   background ping (`SELECT 1`).
//!
//! - **Reopen on typed errors**: `SwmrPendingDdl` and
//!   `SwmrWriterReincarnated` are sticky once raised — keep returning
//!   the same error until the handle is dropped. Reopen the
//!   ReadOnlyDatabase to apply DDL or reattach to the new incarnation.
//!
//! - **Stable snapshot inside a transaction**: SQL `BEGIN` ... `COMMIT`
//!   on this handle (or `set_auto_refresh(false)`) skips the
//!   per-query refresh so all statements observe the same view. The
//!   writer's checkpoint is non-atomic across tables; without this
//!   skip, a multi-table JOIN that refreshes mid-checkpoint could
//!   briefly see table A at epoch N+1 and table B at epoch N.
//!
//! ## What plain `Database::open("file://...?read_only=true")` differs on
//!
//! A plain `Database` opened in read-only mode also gets the lease
//! pin lifecycle and manifest-epoch poll, but does NOT tail the WAL —
//! no overlay, no `SwmrPendingDdl` detection, no sub-checkpoint
//! visibility. Its initial WAL pin is kept (not released) because
//! `Database::as_read_only()` may later wrap the same engine into a
//! `ReadOnlyDatabase` whose overlay needs the WAL above attach to
//! still exist; releasing here would let the writer truncate that
//! range and surface `SwmrSnapshotExpired` on the new ROD's first
//! refresh. Callers that need sub-checkpoint visibility / DDL
//! detection should use `Database::open_read_only` to get a
//! `ReadOnlyDatabase` directly.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::{Error, Result};
use crate::executor::Executor;
use crate::storage::mvcc::manifest_epoch;
use crate::storage::mvcc::overlay::OverlayStore;
use crate::storage::traits::Engine;

use super::database::EngineEntry;
use super::params::Params;
use super::rows::Rows;

/// Read-only handle over a database.
///
/// Constructed via [`crate::api::database::Database::open_read_only`] or
/// [`crate::api::database::Database::as_read_only`]. Rejects all write
/// SQL (INSERT/UPDATE/DELETE/DDL/maintenance PRAGMA/SET TRANSACTION) at
/// query time. Read SQL (SELECT, SHOW, EXPLAIN, BEGIN/COMMIT/ROLLBACK,
/// SAVEPOINT, benign SET no-ops) is allowed.
///
/// Holds an `Arc<EngineEntry>` so the underlying engine cannot be closed
/// while this `ReadOnlyDatabase` handle is alive. The engine stays open
/// as long as any user-visible handle (`Database`, `Database` clone,
/// sibling `Database::open(dsn)`, or `ReadOnlyDatabase`) references it.
///
/// # Transaction visibility
///
/// A `ReadOnlyDatabase` is a *view*, not a connection sharing a session
/// with the writable handle that constructed it. Each handle owns its own
/// executor and therefore its own transaction state:
///
/// - An uncommitted `BEGIN` on the source `Database` is **not** observed
///   by queries through this `ReadOnlyDatabase`. Writes inside the open
///   transaction are not seen until they commit.
/// - A `BEGIN` issued via SQL on this `ReadOnlyDatabase` opens a
///   read-only snapshot transaction local to this handle; it does not
///   interact with any transaction running on the source `Database` or on
///   other `ReadOnlyDatabase` views.
/// - Default isolation level is independent across handles.
///
/// To observe uncommitted writes from a specific transaction, do the read
/// SQL inside that same `Transaction` (read SQL is allowed on transactions
/// regardless of mode).
///
/// # Cross-process coordination
///
/// When opened on a `file://` DSN, this handle registers a presence
/// lease at `<db>/readers/<pid>.lease` whose 8-byte payload is the
/// reader's `pinned_lsn`. Every `query` / `query_named` / `cached_plan`
/// / `query_plan` / `query_named_plan` call advances the WAL pin
/// (overlay-driven) and bumps the lease mtime (rate-limited). The
/// writer combines presence + pin to defer destructive volume unlinks
/// and to floor WAL truncation.
///
/// **Reader staleness contract**: callers that issue at least one
/// query per `max(120s, 2 * checkpoint_interval)` keep their lease
/// fresh. Callers that go silent for longer risk being reaped, after
/// which the writer may unlink old compacted volumes the reader's
/// cached manifest still points at — a subsequent query then hits
/// "volume not found". Mitigations for long-running scans: a longer
/// `checkpoint_interval` (e.g. `?checkpoint_interval=300`), or a
/// background ping (`SELECT 1`).
///
/// **Typed must-reopen errors**: `Error::SwmrPendingDdl` (post-attach
/// DDL) and `Error::SwmrWriterReincarnated` (writer crash + restart)
/// are sticky once raised. Reopen the handle to apply DDL or
/// reattach to the new writer incarnation.
/// Shared backing of a [`ReadOnlyDatabase`] handle.
///
/// Owns per-surface refresh state (overlay cursor, last-seen
/// epoch, per-table LSN cache, sticky must-reopen, executor)
/// plus the `Arc<ReaderAttachment>` that holds this handle's
/// cross-process claim (lease pin contribution + heartbeat).
///
/// Wrapped in `Arc` so [`ReadOnlyDatabase::read_engine`]'s
/// returned trait object can clone the same `Arc` and drive
/// refresh through `maybe_auto_refresh` against the SAME state
/// the owning handle observes — both surfaces advance the same
/// pin / overlay / epoch.
///
/// Lifecycle: this inner struct does NOT have a `Drop` impl.
/// Pin release runs via `ReaderAttachment::Drop` (when the last
/// `Arc<ReaderAttachment>` drops) OR via an explicit
/// `attachment.detach()` from `ReadOnlyDatabase::close` /
/// `Database::close` (idempotent). Engine close runs via
/// `EngineEntry::Drop` (when the last `Arc<EngineEntry>` drops).
///
/// `pub` (not `pub(crate)`) only because it appears in the
/// `Deref` Target on the public `ReadOnlyDatabase`; the type
/// itself is `#[doc(hidden)]` and not part of the stable
/// surface.
#[doc(hidden)]
pub struct ReadOnlyDatabaseInner {
    /// Keeps the engine alive (`EngineEntry::drop` closes the engine
    /// when the last Arc drops).
    entry: Arc<EngineEntry>,
    /// Cross-process reader claim: handle id, pin contribution,
    /// attach snapshot (`attach_visible_commit_lsn`,
    /// `expected_writer_gen`, `pin_active`), and rate-limited
    /// heartbeat. Drop releases the pin contribution; explicit
    /// `attachment.detach()` does the same earlier (idempotent).
    /// Bumps the `EngineEntry`'s strong count by 1 — the close
    /// path's baseline check accounts for that.
    attachment: Arc<crate::api::reader_attachment::ReaderAttachment>,
    /// Independent executor with its own transaction state — a BEGIN
    /// on the read-only handle does not affect the writable Database.
    executor: Mutex<Executor>,
    /// Sticky "must reopen this handle" signal. Set to a non-empty
    /// `String` (the original `SwmrPendingDdl` summary) the first time
    /// `maybe_rebuild_overlay` observes post-attach DDL the reader
    /// can't apply live. Subsequent `refresh()` and `maybe_auto_refresh`
    /// calls re-raise the same error UNTIL the handle is dropped.
    /// Without this, a caller that retries after the first DDL error
    /// would hit the no-op fast path (`to_lsn <= last_applied_lsn`),
    /// get `Ok`, and continue running queries against a stale schema
    /// while the WAL pin advances normally.
    swmr_must_reopen_summary: Mutex<Option<String>>,
    /// SWMR v2 Phase E: per-table overlay of committed-but-uncheckpointed
    /// rows from the WAL tail. Rebuilt on `refresh()` to reflect the
    /// writer's published `visible_commit_lsn`. Always non-None
    /// (defaults to an empty store), but only populated when both
    /// `lease` and `shm` are present (cross-process read-only opens).
    /// Query-time integration (merging into scanner output) is wired
    /// up in a follow-up phase; for now the overlay is observable
    /// via the `overlay()` accessor for tests + diagnostics.
    overlay: Arc<OverlayStore>,

    /// SWMR v2 Phase G: per-table `checkpoint_lsn` snapshot taken at
    /// the last successful `refresh()`. After `reload_manifests`,
    /// `refresh()` compares the engine's current per-table value
    /// against this map. Tables whose `checkpoint_lsn` advanced get
    /// per-table cache invalidation; unchanged tables keep their
    /// cached plans, semantic-cache rows, and stats. New tables
    /// (present on disk but absent here) are also invalidated since
    /// they may have been re-created since the last refresh.
    last_seen_table_lsns: Mutex<rustc_hash::FxHashMap<String, u64>>,
    /// Last manifest epoch this handle observed (from `<db>/volumes/epoch`).
    /// Compared to the on-disk value in `refresh()` to decide whether to
    /// reload manifests. Initialized at construction so the first
    /// `refresh()` no-ops if the writer hasn't checkpointed since open.
    /// Always 0 for in-process / memory engines (refresh is a no-op).
    last_seen_epoch: AtomicU64,
    /// When true (default), every public query path calls `refresh()`
    /// before executing — the cheap path costs one atomic load + one
    /// `read()` of the 8-byte epoch file, no manifest reload unless
    /// the writer has actually checkpointed. Disable via
    /// `set_auto_refresh(false)` for callers that want stable
    /// cross-query visibility (e.g. inside an explicit `BEGIN`/`COMMIT`
    /// block on the read-only handle).
    auto_refresh: AtomicBool,
    /// SWMR v2 P2 perf: gate the WAL-tail overlay rebuild.
    ///
    /// The overlay is the in-memory delta of committed WAL entries
    /// that haven't been checkpointed yet. Query-time scanner
    /// integration that merges overlay rows into scan output is a
    /// follow-up phase; until then, every refresh that builds the
    /// overlay does work no query consumes — and the per-table
    /// `TableOverlay::apply` clones the existing row map for every
    /// touched table, so frequent small commits create O(n) cloning
    /// and growing memory for nothing.
    ///
    /// Default `false`: refresh skips overlay rebuild and only
    /// reloads manifests. Tests and the upcoming `PRAGMA SWMR_OVERLAY`
    /// diagnostic call `set_swmr_overlay_enabled(true)` to opt in.
    swmr_overlay_enabled: AtomicBool,
    /// Coordination lock for `set_swmr_overlay_enabled` <-> overlay
    /// rebuild. Held by `set_swmr_overlay_enabled` for the entire
    /// flag flip + pin install/release sequence, and by
    /// `maybe_rebuild_overlay` for the duration of the tail scan.
    /// Prevents:
    ///   - enable: a concurrent refresh observing flag=true before
    ///     the WAL pin is in place (would tail unprotected).
    ///   - disable: removing the pin while an in-flight refresh
    ///     that already passed the flag check is still tailing.
    ///
    /// One mutex serializes refresh on a single handle, but the
    /// common case is one user per handle so contention is minimal.
    swmr_overlay_state_lock: Mutex<()>,
}

/// Read-only handle to a database opened via
/// [`crate::api::Database::open_read_only`] or
/// [`crate::api::Database::as_read_only`]. See module docs for
/// the SWMR semantics.
///
/// Internally a thin wrapper around `Arc<ReadOnlyDatabaseInner>`
/// so the per-handle WAL pin and refresh state can be shared
/// with `read_engine`'s returned trait object — both surfaces
/// then advance the same pin and overlay cursor on refresh.
pub struct ReadOnlyDatabase {
    pub(crate) inner: Arc<ReadOnlyDatabaseInner>,
}

impl std::ops::Deref for ReadOnlyDatabase {
    type Target = ReadOnlyDatabaseInner;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ReadOnlyDatabase {
    /// Construct a `ReadOnlyDatabase` from a shared `EngineEntry`.
    ///
    /// Crate-internal; `Database::open_read_only` and
    /// `Database::as_read_only` are the public entrypoints.
    pub(crate) fn from_entry(entry: Arc<EngineEntry>) -> Self {
        // Read-only executor: DML helper paths refuse to begin writable
        // auto-commit transactions even if the parser-level write gate
        // is bypassed. Shares the engine entry's semantic cache and
        // query planner so DML commits and ANALYZE on a sibling writable
        // handle invalidate this view's cached SELECT results and
        // planner stats.
        let engine = Arc::clone(&entry.engine);
        let semantic_cache = Arc::clone(&entry.semantic_cache);
        let query_planner = Arc::clone(&entry.query_planner);
        let executor =
            Executor::with_shared_semantic_cache_read_only(engine, semantic_cache, query_planner);

        // Lease + shm + attach snapshots live on the EngineEntry
        // (pre-acquired in `Database::open` / `Database::open_read_only`
        // before engine construction). This handle reads them from the
        // entry, so `Database::open("file://...?read_only=true")` (which
        // returns a plain Database, not a ReadOnlyDatabase) also
        // participates in SWMR coordination.
        // Seed `last_seen_epoch` from the EngineEntry's snapshot of
        // the on-disk epoch at engine-open time, NOT from the live
        // file. A writer checkpoint that lands between entry creation
        // and this constructor (or any subsequent `as_read_only` /
        // `open_read_only` call against a registry-cached entry)
        // would otherwise let `refresh()` see `on_disk == cached` and
        // skip `reload_manifests`, leaving cold rows the underlying
        // engine never reloaded. Seeding from the entry's load-time
        // epoch guarantees the next refresh detects any post-open
        // checkpoint.
        let last_seen_epoch_init = if entry.engine.is_read_only_mode() {
            entry.loaded_epoch
        } else {
            0
        };
        // Use the entry's saved attach values verbatim. The shared
        // engine was opened with `entry.attach_visible_commit_lsn`
        // as its WAL-replay cap; the overlay's `next_entry_floor`
        // must match that cap so it tails entries strictly AFTER
        // what the engine already replayed. Re-sampling shm here
        // would race a writer publish that landed between
        // pre_acquire and now (or between the registry's
        // frontier-static check and this constructor) — using the
        // newer value would seed `next_entry_floor` past WAL
        // entries the engine never applied, silently dropping that
        // DML and any post-attach DDL detection that depends on it.
        // Reuse safety is guaranteed at the registry level by
        // `Database::open_read_only` / `Database::open` refusing to
        // share an entry whose frontier has moved past attach.
        // The cross-process claim (handle id, pin contribution,
        // attach snapshot, heartbeat dedup) is owned by a
        // dedicated `ReaderAttachment`. `attach()` reads the
        // attach values from the entry's saved snapshot
        // (`attach_writer_gen`, `attach_visible_commit_lsn`,
        // `attach_oldest_active_txn_lsn`), allocates a fresh
        // handle_id, computes the overlay baseline LSN via
        // `OverlayStore::initial_pin_lsn`, and installs the per-
        // handle WAL pin (when shm + lease are both present).
        // Half-attached cases land at `pin_active() == false`, the
        // same v1-snapshot semantics as before.
        let attachment =
            crate::api::reader_attachment::ReaderAttachment::attach(Arc::clone(&entry));

        // Seed the overlay LSN cursor at the freshly-sampled
        // visible_commit_lsn + oldest_active_txn_lsn. For a fresh
        // open these are the same as pre_acquire's snapshot. For a
        // reused entry they capture the current writer state so the
        // tail starts at a baseline whose WAL the lease pin
        // (installed by `attach()` at the overlay baseline LSN)
        // protects against truncation by older handles' independent
        // pin advances.
        let overlay = Arc::new(if attachment.pin_active() {
            OverlayStore::with_baseline(
                entry.attach_visible_commit_lsn,
                entry.attach_oldest_active_txn_lsn,
            )
        } else {
            OverlayStore::new()
        });

        Self {
            inner: Arc::new(ReadOnlyDatabaseInner {
                entry,
                attachment,
                executor: Mutex::new(executor),
                last_seen_epoch: AtomicU64::new(last_seen_epoch_init),
                auto_refresh: AtomicBool::new(true),
                swmr_must_reopen_summary: Mutex::new(None),
                overlay,
                last_seen_table_lsns: Mutex::new(rustc_hash::FxHashMap::default()),
                // Off by default — overlay isn't query-integrated yet,
                // so building it on every refresh would be pure waste.
                // See `swmr_overlay_enabled` field doc.
                swmr_overlay_enabled: AtomicBool::new(false),
                swmr_overlay_state_lock: Mutex::new(()),
            }),
        }
    }

    /// Clone this read-only handle for multi-threaded use.
    ///
    /// Each clone shares the underlying engine (cold volumes,
    /// segment manager, semantic + plan caches, the writer's WAL
    /// being tailed) but has its own per-handle state:
    ///
    /// - independent `Executor` (a `BEGIN` on one clone does not
    ///   affect transactions on another),
    /// - its own `ReaderAttachment` (fresh handle id, fresh WAL
    ///   pin contribution to the writer's truncate floor),
    /// - its own per-handle `auto_refresh` flag,
    /// - its own per-handle `OverlayStore` so the WAL-tail
    ///   cursor advances independently per clone.
    ///
    /// Each clone must be closed independently. Engine resources
    /// are released only when the last clone (and any prepared
    /// statements / transactions still referencing it) drops.
    /// Mirrors [`crate::api::Database::clone`] for writable
    /// handles.
    pub fn try_clone(&self) -> Self {
        Self::from_entry(Arc::clone(&self.inner.entry))
    }
}

impl Clone for ReadOnlyDatabase {
    /// See [`Self::try_clone`] for the per-handle semantics. The
    /// clone is infallible at this layer; the `try_` prefix on
    /// the inherent method is preserved for symmetry with future
    /// fallible accessors.
    fn clone(&self) -> Self {
        self.try_clone()
    }
}

impl ReadOnlyDatabase {
    /// Enable WAL-tail overlay rebuild on every refresh. Off by
    /// default — the overlay isn't yet consumed by query execution,
    /// so building it on every query would be pure CPU + memory
    /// waste. Intended for tests and the `PRAGMA SWMR_OVERLAY`
    /// diagnostic. Will be on by default once the scanner integrates
    /// overlay rows.
    ///
    /// Toggles per-row DML overlay materialization. The WAL-tail
    /// scan + DDL detection always run regardless of this flag;
    /// disabling only skips the per-row deserialize + per-table
    /// merge (the part that grows memory and isn't yet consumed by
    /// query execution). Pin lifecycle is unaffected — DDL
    /// detection needs the same WAL protection.
    ///
    /// Held under `swmr_overlay_state_lock` so the flag flip
    /// happens atomically w.r.t. an in-flight rebuild — the
    /// rebuild would otherwise observe a torn flag mid-scan and
    /// fork into ddl-only-vs-full mode mid-stream.
    pub fn set_swmr_overlay_enabled(&self, enabled: bool) -> Result<()> {
        let _state_guard = self
            .swmr_overlay_state_lock
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let was_enabled = self.swmr_overlay_enabled.load(Ordering::Acquire);
        if enabled && !was_enabled {
            // Refuse to enable after the DDL-only cursor has
            // already advanced past attach: every refresh in
            // disabled mode passed `dml_apply = false` to
            // `rebuild_from_wal`, which advanced
            // `last_applied_lsn` past DML rows it discarded.
            // Flipping the flag now would only affect FUTURE
            // tails — the overlay is permanently missing those
            // already-consumed rows. Force the caller to reopen
            // the handle (which constructs a fresh OverlayStore
            // seeded at attach) so DML coverage starts from a
            // safe floor.
            if self.overlay.last_applied_lsn() > self.attachment.attach_visible_commit_lsn() {
                return Err(Error::internal(
                    "set_swmr_overlay_enabled(true) refused: the DDL-only \
                     cursor has already advanced past attach. Enabling now \
                     would leave the overlay permanently missing DML rows \
                     consumed in disabled mode. Reopen the ReadOnlyDatabase \
                     to enable overlay materialization from a safe floor.",
                ));
            }
        }
        self.swmr_overlay_enabled.store(enabled, Ordering::Release);
        Ok(())
    }

    /// Return whether WAL-tail overlay rebuild is currently enabled
    /// for this handle. See `set_swmr_overlay_enabled`.
    pub fn swmr_overlay_enabled(&self) -> bool {
        self.swmr_overlay_enabled.load(Ordering::Acquire)
    }

    /// SWMR v2 Phase E: borrow the per-table WAL-tail overlay. The
    /// returned `Arc` snapshots whatever overlay state existed at call
    /// time; a concurrent `refresh()` may install a new one but won't
    /// invalidate this reference. Stable across the caller's use.
    ///
    /// Currently exposed for tests and the upcoming PRAGMA
    /// `SWMR_OVERLAY` diagnostic. Query-time scanner integration that
    /// merges overlay rows into scan output is a follow-up phase.
    pub fn overlay(&self) -> Arc<OverlayStore> {
        Arc::clone(&self.overlay)
    }

    /// Returns the DSN this handle was opened with.
    pub fn dsn(&self) -> &str {
        &self.entry.dsn
    }

    /// Always returns `true`: a `ReadOnlyDatabase` is read-only by
    /// construction. Symmetric with [`crate::api::database::Database::is_read_only`]
    /// so generic code over both handle types can call the same accessor.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        true
    }

    /// Get the engine as a read-only trait object.
    ///
    /// Returns `Arc<dyn ReadEngine>` so callers get compile-time
    /// enforcement: the trait object exposes only read transactions
    /// (no `Engine::begin_transaction`, no inherent write methods).
    /// Symmetric with [`crate::api::database::Database::read_engine`].
    ///
    /// Every `begin_read_transaction*` call through the returned
    /// trait object runs the SAME SWMR maintenance the owning
    /// handle's `query` / `refresh` paths do: lease heartbeat,
    /// manifest reload, WAL pin advance, and typed must-reopen
    /// error surfacing. The trait object's refresh operates
    /// against the OWNING handle's per-handle pin and overlay
    /// cursor (shared via an internal `Arc<ReadOnlyDatabaseInner>`),
    /// so refresh on either surface advances the same state.
    ///
    /// Specifically:
    ///   - No NEW per-handle pin is installed when `read_engine`
    ///     is called. The trait object reuses `self`'s existing
    ///     pin contribution; refresh through the trait object
    ///     advances `self`'s pin in place.
    ///   - Holding the trait object alive ALSO holds `self`'s
    ///     `Arc<ReadOnlyDatabaseInner>` alive — the per-handle
    ///     pin is only released when BOTH the owner ROD and
    ///     every live trait-object clone have dropped.
    pub fn read_engine(&self) -> Arc<dyn crate::storage::traits::ReadEngine> {
        // Construct a `ReadOnlyDatabase` wrapper that SHARES
        // this handle's `Arc<ReadOnlyDatabaseInner>`. The
        // wrapper itself is a thin pointer; Drop-on-wrapper
        // is a no-op (the load-bearing Drop lives on Inner
        // and only fires when the LAST Arc drops). The guard's
        // `maybe_auto_refresh` then operates on `self`'s
        // pin/overlay/epoch in place rather than installing a
        // fresh proxy at a stale attach baseline.
        Arc::new(crate::api::database::SwmrReadEngineGuard {
            engine: Arc::clone(&self.entry.engine),
            entry: Arc::clone(&self.entry),
            // ReadOnly variant: dispatch to the shared
            // `Arc<ReadOnlyDatabaseInner>` so overlay rebuild,
            // sticky DDL, and pin advance are driven from
            // either surface. The shared inner already owns
            // the `ReaderAttachment` (lease pin contribution +
            // heartbeat), so cloning the inner Arc keeps the
            // pin alive for as long as the trait object is
            // alive. Active-txn skip happens inside
            // `inner.maybe_auto_refresh`, which gates on the
            // ROD's own executor — identical to what a SQL
            // `BEGIN` on this ROD touches.
            refresh_owner: crate::api::database::RefreshOwner::ReadOnly(Arc::clone(&self.inner)),
        }) as Arc<dyn crate::storage::traits::ReadEngine>
    }

    /// Returns `true` if a table with the given name exists.
    ///
    /// Runs the same SWMR maintenance the `query` paths do:
    /// touches the lease (so a `table_exists`-only poller stays
    /// fresh) and triggers `maybe_auto_refresh` so a writer-side
    /// CREATE/DROP/checkpoint is observed instead of returning
    /// this handle's stale schema cache. Typed must-reopen
    /// errors (`SchemaChanged`, `SwmrPendingDdl`,
    /// `SwmrSnapshotExpired`, `SwmrWriterReincarnated`,
    /// `SwmrPartialReload`) propagate to the caller, matching
    /// `query()` semantics.
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        Engine::table_exists(&*self.entry.engine, name)
    }

    /// Returns the row count of `name` visible to this read-only handle's
    /// snapshot, accounting for both hot rows and sealed cold volumes.
    /// Drives the same lease-touch + auto-refresh maintenance as
    /// [`Self::table_exists`] so cross-process visibility advances; the
    /// count itself is the SegmentedTable fast path (O(1) when no
    /// snapshot-iso fallback is required).
    pub fn table_count(&self, name: &str) -> Result<u64> {
        use crate::storage::traits::ReadEngine;
        self.touch_lease();
        self.maybe_auto_refresh()?;
        let tx = ReadEngine::begin_read_transaction(self.entry.engine.as_ref())?;
        let table = tx.get_read_table(name)?;
        if let Some(c) = table.fast_row_count() {
            return Ok(c as u64);
        }
        Ok(table.row_count() as u64)
    }

    /// Bump the cross-process presence lease's mtime (no pin
    /// advance). Touch failures are silently ignored: the worst case
    /// is the writer reaps our lease as stale and unlinks volumes we
    /// still reference. On Unix that's still safe because eager-fd-open
    /// keeps the inode alive — but the next manifest reload would miss
    /// the volume.
    ///
    /// this method NO LONGER advances `pinned_lsn`.
    /// It used to pin to the writer's current `visible_commit_lsn`,
    /// but `query()` calls touch BEFORE `maybe_auto_refresh()`, so
    /// the reader was advertising "I've consumed WAL up to V" before
    /// it had actually replayed the (last_applied..V] window — a
    /// concurrent writer checkpoint could then truncate exactly the
    /// range the reader was about to tail. Pin advancement now lives
    /// in `advance_pin_after_refresh()`, called only after a
    /// successful overlay rebuild has actually applied the entries.
    ///
    /// Rate-limited (1s floor) lease mtime touch — delegates to
    /// the attachment so the heartbeat-dedup state lives with the
    /// pin contribution it protects.
    #[inline]
    fn touch_lease(&self) {
        self.attachment.touch_lease();
    }

    /// Advance this handle's WAL pin AFTER refresh has replayed
    /// the window. Pinned LSN reflects the reader's
    /// `overlay.next_entry_floor` — the lowest LSN the reader still
    /// needs to scan on the NEXT rebuild. Writer's `truncate_wal`
    /// floor is `min(checkpoint_lsn, min_pinned_lsn - 1)`, so
    /// pinning at `next_entry_floor` preserves exactly the WAL
    /// range the reader's next refresh will consult.
    ///
    /// Floor the pin at 1: a refresh that observed nothing (e.g.
    /// `visible_commit_lsn == 0` at attach, no WAL tail yet)
    /// returns `next_entry_floor() = 0`. Writing 0 to the lease
    /// would release this reader from `min_pinned_lsn`'s scan
    /// (pin=0 is the "released" sentinel), so a writer checkpoint
    /// between this query's epoch read and its WAL tail scan
    /// could truncate the entries the next refresh needs.
    ///
    /// The attachment owns the syscall-dedup, half-attached gate,
    /// and `set_handle_pin` invocation. We just compute the floor
    /// from our overlay state and hand it over.
    #[inline]
    fn advance_pin_after_refresh(&self) {
        let pin = self.overlay.next_entry_floor().max(1);
        self.attachment.advance_pin(pin);
    }

    /// Toggle automatic refresh on every query.
    ///
    /// Default is `true`: each `query` / `query_named` / `query_plan` /
    /// `query_named_plan` call polls the on-disk manifest epoch (one
    /// `read()` of an 8-byte file) and, if advanced, reloads manifests
    /// before executing. The poll itself costs ~1µs when nothing has
    /// changed; the reload only happens after the writer checkpoints.
    ///
    /// Set to `false` when you need stable visibility across multiple
    /// queries — for example, inside a `BEGIN ... COMMIT` block on this
    /// read-only handle, or while iterating multiple `Rows` cursors that
    /// must agree on the same snapshot. Call `refresh()` manually when
    /// you want to advance.
    ///
    /// This is a per-handle setting; sibling `ReadOnlyDatabase` handles
    /// over the same DSN have independent flags.
    pub fn set_auto_refresh(&self, enabled: bool) {
        self.auto_refresh.store(enabled, Ordering::Release);
    }

    /// Returns whether auto-refresh is currently enabled. See
    /// [`set_auto_refresh`].
    pub fn auto_refresh_enabled(&self) -> bool {
        self.auto_refresh.load(Ordering::Acquire)
    }

    /// Cheap auto-refresh hook called from every query path. Skips when
    /// disabled. Most refresh errors (transient I/O, etc.) are silently
    /// ignored — the query proceeds against the current snapshot, since
    /// stale-snapshot is a better failure mode than refusing the query.
    ///
    /// **Skipped during an active transaction (v2 P0.3)**: when the
    /// executor has an open `BEGIN` (no `COMMIT`/`ROLLBACK` yet),
    /// auto-refresh would silently advance the snapshot mid-transaction
    /// and break the user's expected stable-reads-across-statements
    /// contract. The transaction's first statement pins; subsequent
    /// statements within it observe the same epoch. Call `refresh()`
    /// before `BEGIN` (or after `COMMIT`) for explicit control.
    ///
    /// **Exception**: `Error::SchemaChanged` is propagated. The reader's
    /// in-memory schema is older than the on-disk manifests; continuing
    /// to scan would interpret bytes against the wrong column layout.
    /// Surfacing the error here gives the caller a clear "reopen needed"
    /// signal instead of a silent wrong-result.
    #[inline]
    pub(crate) fn maybe_auto_refresh(&self) -> Result<()> {
        if !self.auto_refresh.load(Ordering::Acquire) {
            return Ok(());
        }
        // Skip if a BEGIN is open on this handle's executor. The lock
        // is brief — has_active_transaction is a single Mutex check —
        // so contention with concurrent queries on the same handle is
        // negligible.
        if let Ok(executor) = self.executor.lock() {
            if executor.has_active_transaction() {
                return Ok(());
            }
        }
        match self.refresh() {
            Ok(_) => {}
            Err(e @ Error::SchemaChanged(_)) => return Err(e),
            // SWMR v2 Phase H: surface typed sub-kinds so callers can
            // distinguish "must reopen" (SwmrPendingDdl,
            // SwmrWriterReincarnated, SwmrSnapshotExpired,
            // SwmrPartialReload) from "transient retry"
            // (SwmrOverlayApplyFailed). v1's silent-swallow behaviour
            // is preserved for any other refresh error since those
            // are typically transient I/O (lease touch failed,
            // manifest stat failed, etc.) that don't mutate state.
            Err(e @ Error::SwmrPendingDdl(_)) => return Err(e),
            Err(e @ Error::SwmrWriterReincarnated { .. }) => return Err(e),
            Err(e @ Error::SwmrSnapshotExpired { .. }) => return Err(e),
            Err(e @ Error::SwmrOverlayApplyFailed(_)) => return Err(e),
            // `reload_manifests` may have partially swapped per-table
            // state before failing on a later table — the snapshot is
            // now mixed-epoch. Propagate so the caller hard-reopens
            // instead of running queries against inconsistent
            // manifests.
            Err(e @ Error::SwmrPartialReload(_)) => return Err(e),
            Err(_) => {}
        }
        Ok(())
    }

    /// Refresh this handle's view of the database against the writer's
    /// latest checkpoint.
    ///
    /// Reads `<db>/volumes/epoch` (cheap, ~µs) and compares to the last
    /// epoch this handle observed. If unchanged, returns `Ok(false)`
    /// without doing any other work. If the writer has bumped the epoch
    /// since the last call (i.e. has produced a new checkpoint),
    /// reloads every per-table manifest from disk and returns `Ok(true)`.
    ///
    /// Use this when your application wants to deterministically pick
    /// up new committed state (e.g. on a timer, or before a snapshot
    /// query). For most callers, the auto-refresh on every `query` /
    /// `query_named` path is sufficient and you don't need to call this
    /// directly.
    ///
    /// **Visibility contract**: after a successful `refresh()` returns
    /// `true`, subsequent queries on this handle observe the writer's
    /// state as of the bumped checkpoint. Sub-checkpoint commits
    /// (rows still in the writer's hot buffer + WAL) become visible
    /// once `set_swmr_overlay_enabled(true)` is called and the
    /// scanner integration ships; until then, sub-checkpoint state
    /// is detected only for DDL surfacing as `SwmrPendingDdl`.
    ///
    /// **No-op for in-process and memory handles**: `as_read_only()` over
    /// a writable engine, and any `memory://` engine, share state with
    /// the writer in-process and don't need cross-process refresh. This
    /// method returns `Ok(false)` for them.
    ///
    /// **DDL surfaces as `SwmrPendingDdl`, not as auto-applied**:
    /// CREATE/DROP/ALTER from the writer that lands after attach is
    /// detected by the WAL-tail rebuild and surfaced as
    /// `Error::SwmrPendingDdl` (sticky — every subsequent call
    /// re-raises until the handle is reopened). Reopen to apply.
    pub fn refresh(&self) -> Result<bool> {
        // Sticky SwmrPendingDdl signal. Once the
        // overlay rebuild has observed post-attach DDL the reader
        // can't apply live, every subsequent refresh re-raises
        // the SAME error until the handle is reopened. Without
        // this, a caller that retries after the first error
        // would hit the no-op fast path below
        // (`to_lsn <= last_applied_lsn` inside
        // `maybe_rebuild_overlay`), get `Ok`, advance the WAL
        // pin, and silently run queries against a stale schema.
        if let Ok(guard) = self.swmr_must_reopen_summary.lock() {
            if let Some(ref summary) = *guard {
                return Err(Error::SwmrPendingDdl(summary.clone()));
            }
        }

        let path = self.entry.engine.get_path();
        if path.is_empty() || !self.entry.engine.is_read_only_mode() {
            return Ok(false);
        }

        // SWMR v2 Phase H: detect writer reincarnation BEFORE any
        // per-handle work. If db.shm.writer_generation has advanced
        // past what we observed at attach, the writer crashed and
        // recovered (or closed+reopened); none of our cached state
        // is trustworthy. Surface SwmrWriterReincarnated so the
        // caller hard-reopens.
        if let Some(handle) = self.entry.shm.as_ref() {
            let observed = handle.header().writer_generation.load(Ordering::Acquire);
            let expected = self.attachment.expected_writer_gen();
            // Any mismatch (not just `observed > expected`) is a
            // reincarnation. A crash during the new writer's
            // create_writer init window — between the in-place
            // zeroing of the mmap and the restore of `prior_gen` —
            // leaves writer_generation at 0 on disk. The next
            // writer reads `prior_gen=0`, so a reader attached
            // when the previous incarnation had `expected=N>1`
            // would observe `1 < N` and a `>` check would silently
            // miss the reincarnation, serving stale cached state.
            if expected > 0 && observed != expected {
                return Err(Error::SwmrWriterReincarnated {
                    observed_gen: observed,
                    expected_gen: expected,
                });
            }
        } else {
            // No-shm reader: this handle attached when no writer
            // was live (pre_acquire returned shm=None). The Unix
            // shared file lock isn't held as a kernel lock, so a
            // writer can start at any point post-attach. We must
            // detect that activity even if the writer has
            // since exited — a writer that ran, committed
            // (including empty/hot-only DDL like an empty
            // CREATE TABLE that never produces a manifest dir),
            // and closed leaves `db.shm` on disk with a
            // bumped `writer_generation`.
            //
            // Detection: probe `db.shm`. If `writer_generation`
            // observed now is greater than the baseline
            // captured at attach (`attachment.expected_writer_gen()`,
            // populated even on no-shm paths via
            // `pre_acquire_swmr_for_read_only_path`), a writer
            // ran post-attach and our cached engine state is
            // stale (engine was opened uncapped at attach with
            // no way to apply post-attach WAL). Surface
            // `SwmrWriterReincarnated` so the caller hard-
            // reopens — the new open goes through `pre_acquire`
            // and constructs a properly attached handle.
            //
            // A relying-on-LOCK_EX-presence check (we tried it)
            // is insufficient: it only catches writers that are
            // CURRENTLY alive, missing the
            // started/committed/closed sequence that the user
            // surfaced as P1.
            //
            // The registry already refuses to REUSE no-shm
            // entries on subsequent `open_read_only(dsn)` calls
            // for the same reason; this path covers EXISTING
            // handles that were already alive.
            let p = std::path::Path::new(path);
            if let Ok(handle) = crate::storage::mvcc::shm::ShmHandle::open_reader(p) {
                let observed = handle.header().writer_generation.load(Ordering::Acquire);
                let expected = self.attachment.expected_writer_gen();
                if observed > expected {
                    return Err(Error::SwmrWriterReincarnated {
                        observed_gen: observed,
                        expected_gen: expected,
                    });
                }
            }
        }

        let p = std::path::Path::new(path);
        let on_disk = manifest_epoch::read_epoch(p).unwrap_or(0);
        let cached = self.last_seen_epoch.load(Ordering::Acquire);
        let manifests_changed = on_disk > cached;
        if manifests_changed {
            // Reload manifests, then publish the new epoch. Order matters:
            // if we updated the epoch first and crashed before reload, a
            // future call would skip the reload because cached >= on_disk.
            if !self.entry.engine.reload_manifests()? {
                self.touch_lease();
                return Ok(false);
            }

            // SWMR v2 Phase G: per-table cache invalidation. Compare
            // each table's current checkpoint_lsn against the snapshot
            // taken at the last refresh. Tables whose checkpoint_lsn
            // advanced (or that newly appeared on disk) get per-table
            // invalidation across the semantic cache, planner stats,
            // and query plan cache. Unchanged tables keep their cached
            // entries — a writer commit on table A no longer evicts
            // every cached plan for unrelated tables B, C, ...
            let now_lsns = self.entry.engine.table_checkpoint_lsns();
            let mut prev = self
                .last_seen_table_lsns
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let executor_guard = self.executor.lock().ok();
            for (table, &now_lsn) in now_lsns.iter() {
                let prev_lsn = prev.get(table).copied().unwrap_or(0);
                // A table is considered "changed" when its
                // checkpoint_lsn advanced. Newly appeared tables
                // (prev_lsn = 0) also count, even if their current
                // checkpoint_lsn is 0, because their absence in `prev`
                // means they were created since the last refresh and
                // any cached entry referencing them would be stale.
                if now_lsn > prev_lsn || !prev.contains_key(table) {
                    self.entry.semantic_cache.invalidate_table(table);
                    self.entry.query_planner.invalidate_stats_cache(table);
                    if let Some(ref ex) = executor_guard {
                        ex.invalidate_query_cache_for_table(table);
                    }
                }
            }
            // Tables that disappeared from disk (DROP TABLE on the
            // writer side) also need invalidation. Walk `prev` and
            // invalidate any keys not in `now_lsns`.
            for table in prev.keys() {
                if !now_lsns.contains_key(table) {
                    self.entry.semantic_cache.invalidate_table(table);
                    self.entry.query_planner.invalidate_stats_cache(table);
                    if let Some(ref ex) = executor_guard {
                        ex.invalidate_query_cache_for_table(table);
                    }
                }
            }
            *prev = now_lsns;
            drop(prev);
            drop(executor_guard);

            // Bound overlay memory by the WAL gap.
            // After a writer checkpoint, all rows committed up to
            // the new checkpoint_lsn are now in cold storage. The
            // overlay's per-table row state is stale (cold has the
            // authoritative version) and would just leak memory
            // for long-lived readers.
            //
            // `clear_rows_only` drops the per-table maps and
            // pending DDL but KEEPS `last_applied_lsn` and
            // `next_entry_floor`. Resetting them would either
            // start the next tail below the writer's truncation
            // floor (`SwmrSnapshotExpired`) or skip post-checkpoint
            // DDL the reader hasn't seen yet (lost
            // `SwmrPendingDdl`). The WAL pin (tracking
            // `next_entry_floor`) keeps the needed range alive
            // across the checkpoint.
            self.overlay.clear_rows_only();

            self.last_seen_epoch.store(on_disk, Ordering::Release);
        }
        // SWMR v2 Phase E + H: rebuild the WAL-tail overlay if the
        // writer has advanced visible_commit_lsn beyond what's
        // currently applied. We rebuild even when manifests didn't
        // change, since hot-buffer commits between checkpoints don't
        // bump the manifest epoch but DO bump visible_commit_lsn.
        //
        // Errors propagate. The two SWMR sub-kinds the rebuild can
        // raise — SwmrPendingDdl and SwmrOverlayApplyFailed — are
        // both actionable signals to the caller (reopen, retry).
        //
        // ordering matters here. We:
        //   1. Try the rebuild.
        //   2. mtime-touch the lease so the writer doesn't reap us
        //      while we're still active.
        //   3. ON SUCCESSFUL TAIL, advance the WAL pin to
        //      `overlay.next_entry_floor()` — i.e., the lowest LSN
        //      we still need on the next rebuild. "Successful tail"
        //      includes the `SwmrPendingDdl` case: the tail
        //      completed and `rebuild_from_wal` already advanced
        //      `last_applied_lsn` / `next_entry_floor`; the error
        //      just signals the caller must reopen to apply DDL.
        //      Without advancing the pin in that case, a stale
        //      handle that keeps re-raising the sticky error would
        //      hold its pin at the initial `1` for its lifetime,
        //      blocking WAL truncation. Errors that indicate the
        //      tail itself failed (`SwmrOverlayApplyFailed`,
        //      `SwmrSnapshotExpired`) leave the cursors unchanged
        //      and we must NOT advance.
        let rebuild_result = self.maybe_rebuild_overlay();
        self.touch_lease();
        // Borrow `rebuild_result` for classification — `Error` is
        // Clone but not Copy, so a by-value `matches!` would move
        // the value out from under the `?` below.
        let tail_succeeded = matches!(&rebuild_result, Ok(()) | Err(Error::SwmrPendingDdl(_)));
        if tail_succeeded {
            self.advance_pin_after_refresh();
        }
        rebuild_result?;
        Ok(manifests_changed)
    }

    /// SWMR v2 Phase E + H: rebuild the per-table overlay from the
    /// WAL tail when the writer's published `visible_commit_lsn`
    /// exceeds the overlay's `last_applied_lsn`. Cheap when nothing
    /// has changed (one Acquire load comparison + one Acquire load
    /// on the shm header). When shm or wal is absent, no-op — the
    /// overlay stays empty and queries just use cold storage as in
    /// v1.
    ///
    /// Returns:
    /// - `Ok(())` on a clean rebuild (or no-op).
    /// - `Err(SwmrPendingDdl(_))` when DDL events fell in the
    ///   refresh window. The overlay is still updated with DML
    ///   deltas, but the caller should reopen to apply DDL safely.
    /// - `Err(SwmrOverlayApplyFailed(_))` on a tail/decode failure.
    #[inline]
    fn maybe_rebuild_overlay(&self) -> Result<()> {
        // Hold the overlay-state lock across the flag check AND the
        // tail scan: this is the read-side counterpart of
        // `set_swmr_overlay_enabled`'s flag-flip + pin-install
        // critical section. Without holding it, a `disable` call
        // could remove our WAL pin between our flag check and the
        // tail scan, letting the writer truncate WAL we still need.
        // The lock also blocks `disable` from finishing until our
        // tail completes (so the pin can't be removed mid-tail).
        let _state_guard = self
            .swmr_overlay_state_lock
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // The WAL-tail rebuild runs when this handle is fully SWMR-
        // attached (shm + active pin). It's the path that surfaces
        // post-attach DDL as `SwmrPendingDdl`, so it MUST run on
        // default ReadOnlyDatabase handles. The `swmr_overlay_enabled`
        // flag (passed as `dml_apply` to `rebuild_from_wal` below)
        // only controls whether per-row DML materialization happens
        // — the WAL scan + DDL collection + LSN watermark advance
        // happen in both modes.
        //
        // Gate on `swmr_pin_active` (NOT just `shm.is_some()`):
        // half-attached handles (shm acquired but lease/pin install
        // failed) have no protecting pin, so tailing WAL would let
        // a writer truncate entries we're reading.
        if !self.attachment.pin_active() {
            return Ok(());
        }
        let shm = match self.entry.shm.as_ref() {
            Some(h) => h,
            None => return Ok(()),
        };
        // Coherent (visible, oldest) snapshot via the shm
        // seqlock — see the matching
        // `pre_acquire_swmr_for_read_only_path` comment for the
        // full rationale. When `sample_visibility_pair` returns
        // None (writer kept publishing across all retries), fall
        // back to `oldest = 0` so `rebuild_from_wal` performs the
        // conservative full Phase-2 scan rather than risk
        // advancing `next_entry_floor` past DML for a transaction
        // that's active at the new target.
        let (target, writer_oldest_active) =
            shm.header().sample_visibility_pair().unwrap_or_else(|| {
                let v = shm.header().visible_commit_lsn.load(Ordering::Acquire);
                (v, 0)
            });
        // Post-visibility writer_generation recheck. Closes the
        // wipe race that the up-front gen check in `refresh()`
        // alone can't cover: an existing reader could load gen at
        // T0 (matches expected), the writer reincarnates at T1
        // (bumps gen, then wipes visibility), and the reader
        // loads visibility at T2 — getting 0 paired with the
        // pre-T1 expected gen. Without this recheck the reader
        // would treat target=0 as "no work" (correct fast path
        // for a fresh DB) and continue serving stale cached
        // state. Re-loading gen here turns that case into a
        // visible SwmrWriterReincarnated.
        let observed_after = shm.header().writer_generation.load(Ordering::Acquire);
        let expected = self.attachment.expected_writer_gen();
        if expected > 0 && observed_after != expected {
            return Err(Error::SwmrWriterReincarnated {
                observed_gen: observed_after,
                expected_gen: expected,
            });
        }
        if target == 0 {
            return Ok(());
        }
        // Note: we deliberately do NOT skip here when
        // `target <= overlay.last_applied_lsn()`. `rebuild_from_wal`
        // handles that no-op case internally AND uses the
        // sampled `writer_oldest_active` to advance
        // `next_entry_floor` if the writer has re-published a
        // higher watermark (e.g. after a long-running txn
        // commit/rollback without a new visible commit).
        // Skipping here would freeze `next_entry_floor` and
        // leave the lease pin stuck at the old low LSN until an
        // unrelated future commit moves visibility.
        let wal = match self.entry.engine.wal() {
            Some(w) => w,
            None => return Ok(()),
        };
        // rebuild_from_wal already wraps tail errors in
        // SwmrOverlayApplyFailed, so propagate directly.
        // `dml_apply` is gated by `swmr_overlay_enabled` — when
        // off, the WAL tail still runs (so DDL events surface
        // and `last_applied_lsn` / `next_entry_floor` advance to
        // keep the WAL pin moving forward), but per-row
        // deserialize + per-table merge are skipped.
        let dml_apply = self.swmr_overlay_enabled.load(Ordering::Acquire);
        self.overlay
            .rebuild_from_wal(wal, target, writer_oldest_active, dml_apply)?;

        // Phase H: surface DDL events that happened AFTER this
        // handle attached AND that aren't already-known schema
        // (idempotent re-records).
        //
        // Two filters needed:
        //
        // 1. `lsn > attach_visible_commit_lsn`: DDL applied before
        //    the reader opened was already absorbed by the engine's
        //    startup schema-replay.
        //
        // 2. CREATE TABLE for a table the reader already knows: the
        //    writer re-records DDL after every WAL truncation (so
        //    those entries survive the truncation). Each re-record
        //    gets a fresh LSN > the prior checkpoint, which would
        //    otherwise look like a brand-new CREATE TABLE to the
        //    reader. If the reader's segment_managers already has
        //    the table, the event is just a re-record — silently
        //    skip it. AlterTable/DropTable/etc. are NOT filtered:
        //    those represent real schema changes the reader's
        //    cached metadata can't apply live.
        // use the broader `known_catalog_objects`
        // (schemas + views) instead of `table_checkpoint_lsns`. The
        // latter only includes tables with segment managers, missing:
        //   - Empty tables (CREATE TABLE without inserts) → false
        //     SwmrPendingDdl after the next checkpoint re-records.
        //   - Views (CreateView's `entry.table_name` is the VIEW
        //     name, never present in segment managers).
        // The new set covers all catalog objects the reader's
        // engine has WAL-replayed, so re-records of pre-existing
        // tables AND views AND empty-but-declared tables are
        // suppressed correctly.
        let known_catalog = self.entry.engine.known_catalog_objects();
        let known_indexes = self.entry.engine.known_index_names();
        let ddl: Vec<_> = self
            .overlay
            .pending_ddl()
            .into_iter()
            .filter(|e| e.lsn > self.attachment.attach_visible_commit_lsn())
            .filter(|e| {
                // Skip Create* re-records for catalog objects we
                // already know about. The writer re-records all DDL
                // after WAL truncation so each entry survives, and
                // each re-record gets a fresh LSN > the prior
                // checkpoint. Without this filter every checkpoint
                // would surface SwmrPendingDdl on the next refresh
                // for any pre-existing table/index/view.
                //
                // The check is conservative: we ONLY look at
                // CreateTable/CreateView/CreateIndex, and only when
                // the object name is one the reader already tracks.
                // Drop/Alter/Rename/Truncate are NOT suppressed —
                // those are real schema changes the reader's cached
                // metadata can't apply live.
                //
                // CreateView is keyed on `entry.table_name` (which
                // IS the view name for CreateView entries) against
                // `known_catalog_objects`.
                //
                // CreateIndex is keyed on the INDEX name decoded
                // from the IndexMetadata payload. We can't use
                // `entry.table_name` here because for CreateIndex
                // it's the underlying TABLE name (would silently
                // absorb a brand-new index on a known table). The
                // `IndexMetadata` serialization starts with
                // `name_len: u16` then `name: bytes` (see
                // `IndexMetadata::serialize` in persistence.rs), so
                // we decode just the prefix without pulling in the
                // full deserializer. A decode failure is treated as
                // "not a known re-record" — better to surface a
                // false positive than silently swallow a real
                // CreateIndex.
                use crate::storage::mvcc::wal_manager::WALOperationType;
                let is_known_re_record = match e.operation {
                    WALOperationType::CreateTable | WALOperationType::CreateView => {
                        known_catalog.contains(&e.table_name)
                    }
                    WALOperationType::CreateIndex => e
                        .payload
                        .as_deref()
                        .and_then(decode_index_name_from_metadata)
                        .map(|name| known_indexes.contains(&name))
                        .unwrap_or(false),
                    _ => false,
                };
                !is_known_re_record
            })
            .collect();
        if !ddl.is_empty() {
            // Build a short summary: "createtable:t,altertable:orders,...".
            // Cap at 10 entries to keep the error string bounded.
            let mut parts: Vec<String> = ddl
                .iter()
                .take(10)
                .map(|e| {
                    if e.table_name.is_empty() {
                        format!("{:?}", e.operation).to_lowercase()
                    } else {
                        format!(
                            "{}:{}",
                            format!("{:?}", e.operation).to_lowercase(),
                            e.table_name
                        )
                    }
                })
                .collect();
            if ddl.len() > 10 {
                parts.push(format!("(+{} more)", ddl.len() - 10));
            }
            let summary = parts.join(",");
            // Store the summary so the next refresh
            // (which would otherwise hit the no-op fast path)
            // re-raises the same error. The signal stays sticky
            // until the handle is dropped — once a reader has
            // observed live-unappliable DDL it must reopen to
            // pick up the new schema, full stop.
            if let Ok(mut guard) = self.swmr_must_reopen_summary.lock() {
                if guard.is_none() {
                    *guard = Some(summary.clone());
                }
            }
            return Err(Error::SwmrPendingDdl(summary));
        }
        Ok(())
    }

    /// Execute a read-only SQL query.
    ///
    /// Rejects any statement that mutates persistent state (INSERT, UPDATE,
    /// DELETE, DDL, maintenance PRAGMA, SET TRANSACTION ISOLATION LEVEL).
    /// Read statements (SELECT, SHOW, EXPLAIN, BEGIN/COMMIT/ROLLBACK,
    /// SAVEPOINT, benign SET no-ops) are allowed.
    pub fn query<P: Params>(&self, sql: &str, params: P) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        // Write rejection happens inside the executor's parse/cache path
        // (Executor::read_only=true), so we don't pre-parse here. This
        // avoids paying for two full parses on every read-only query.
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;

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

    /// Execute a read-only query with named parameters.
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    pub fn query_named(&self, sql: &str, params: crate::api::NamedParams) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(Rows::new(result))
    }

    /// Cache a parsed plan for a read-only SQL statement.
    ///
    /// Same shape as [`crate::api::Database::cached_plan`]: parse once,
    /// reuse the [`CachedPlanRef`] across many `query_plan` /
    /// `query_named_plan` calls without re-parsing.
    ///
    /// Rejects write SQL at plan-creation time with `ReadOnlyViolation`.
    /// This is the prepared-statement equivalent on a `ReadOnlyDatabase`
    /// (the `prepare()` path on `Database` requires a `Weak<DatabaseInner>`
    /// that the read-only handle does not have; cached plans give the
    /// same parse-once / execute-many ergonomics without that coupling).
    pub fn cached_plan(&self, sql: &str) -> Result<crate::executor::query_cache::CachedPlanRef> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        executor.get_or_create_plan(sql)
    }

    /// Query using a pre-cached plan with positional parameters
    /// (no parsing, no cache lookup). Read-only equivalent of
    /// [`crate::api::Database::query_plan`].
    pub fn query_plan<P: Params>(
        &self,
        plan: &crate::executor::query_cache::CachedPlanRef,
        params: P,
    ) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        use crate::executor::context::ExecutionContext;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Query using a pre-cached plan with named parameters.
    /// Read-only equivalent of [`crate::api::Database::query_named_plan`].
    pub fn query_named_plan(
        &self,
        plan: &crate::executor::query_cache::CachedPlanRef,
        params: crate::api::NamedParams,
    ) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        use crate::executor::context::ExecutionContext;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }
}

// `ReadOnlyDatabase` has no `Drop` impl. Engine cleanup runs via
// `EngineEntry::Drop` when the last `Arc<EngineEntry>` drops (the
// registry's `Weak` silently expires). Pin contribution release
// runs via `ReaderAttachment::Drop` when the last
// `Arc<ReaderAttachment>` drops — that's idempotent with any
// explicit `attachment.detach()` call from `Database::close` /
// `ReadOnlyDatabase::close`.

/// Decode the index name from an `IndexMetadata` payload prefix
/// without pulling in the full deserializer. The serialization
/// format (see `IndexMetadata::serialize` in
/// `src/storage/mvcc/persistence.rs`) starts with `name_len: u16`
/// (little-endian) followed by `name: name_len bytes` (UTF-8).
/// Returns `None` on any decode failure — the caller treats that
/// as "not a known re-record" and surfaces SwmrPendingDdl, which
/// is the conservative choice (false positive over false negative).
fn decode_index_name_from_metadata(payload: &[u8]) -> Option<String> {
    if payload.len() < 2 {
        return None;
    }
    let name_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    if 2 + name_len > payload.len() {
        return None;
    }
    std::str::from_utf8(&payload[2..2 + name_len])
        .ok()
        .map(str::to_owned)
}
