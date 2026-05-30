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
//! Provides a read-only view over an existing database. SWMR semantics:
//! lease + WAL pin pair the reader with the writer's truncate floor; an
//! auto-poll of `<db>/volumes/epoch` reloads manifests on checkpoint;
//! WAL-tail tailing surfaces post-attach DDL as `Error::SwmrPendingDdl`
//! and writer reincarnation as `Error::SwmrWriterReincarnated`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{JoinHandle, ThreadId};
use std::time::Duration;

use crate::core::{Error, Result};
use crate::executor::Executor;
use crate::storage::mvcc::manifest_epoch;
use crate::storage::mvcc::overlay::OverlayStore;
use crate::storage::traits::Engine;

use super::database::EngineEntry;
use super::params::Params;
use super::rows::Rows;

/// Minimum allowed `refresh_interval`. Tighter cadences should drive
/// `refresh()` from a caller-owned loop.
const MIN_REFRESH_INTERVAL: Duration = Duration::from_millis(100);

struct RefreshTickerHandle {
    stop: Arc<(Mutex<bool>, Condvar)>,
    join: Option<JoinHandle<()>>,
    /// Captured at spawn so `shutdown` can detect the self-shutdown
    /// case (joining one's own thread deadlocks) and detach instead.
    thread_id: ThreadId,
}

impl RefreshTickerHandle {
    /// Signal the ticker to exit and reap it. Idempotent. Safe from
    /// any thread including the ticker itself (self-shutdown detaches).
    fn shutdown(&mut self) {
        {
            let mut want_stop = self.stop.0.lock().expect("ticker stop mutex poisoned");
            *want_stop = true;
            self.stop.1.notify_all();
        }
        if let Some(join) = self.join.take() {
            if std::thread::current().id() == self.thread_id {
                drop(join);
            } else {
                let _ = join.join();
            }
        }
    }
}

/// Body of the background refresh ticker thread. Skips the refresh
/// (without exiting) while `auto_refresh` is false or a BEGIN is open.
/// Exits on shutdown signal, dropped handle, or sticky must-reopen error.
fn run_refresh_ticker(
    weak: Weak<ReadOnlyDatabaseInner>,
    interval: Duration,
    stop: Arc<(Mutex<bool>, Condvar)>,
) {
    loop {
        let mut want_stop = match stop.0.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let result = stop.1.wait_timeout_while(want_stop, interval, |s| !*s);
        want_stop = match result {
            Ok((g, _)) => g,
            Err(_) => return,
        };
        if *want_stop {
            return;
        }
        drop(want_stop);

        let inner = match weak.upgrade() {
            Some(i) => i,
            None => return,
        };
        let temp = ReadOnlyDatabase { inner };
        // Hold refresh_fence across the iteration so
        // `set_auto_refresh(false)` becomes a synchronous fence.
        let _refresh_fence = match temp.refresh_fence.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if !temp.auto_refresh.load(Ordering::Acquire) {
            continue;
        }
        if let Ok(executor) = temp.executor.lock() {
            if executor.has_active_transaction() {
                continue;
            }
        }
        match temp.refresh() {
            Ok(_) => {}
            Err(Error::SwmrPendingDdl(_))
            | Err(Error::SwmrWriterReincarnated { .. })
            | Err(Error::SwmrSnapshotExpired { .. })
            | Err(Error::SwmrPartialReload(_))
            | Err(Error::SchemaChanged(_)) => return,
            Err(_) => {}
        }
    }
}

/// Shared backing of a [`ReadOnlyDatabase`] handle. `#[doc(hidden)]`,
/// only `pub` because it appears in the `Deref` target.
#[doc(hidden)]
pub struct ReadOnlyDatabaseInner {
    entry: Arc<EngineEntry>,
    /// Cross-process reader claim (handle id, pin contribution, attach
    /// snapshot, heartbeat). Drop releases the pin.
    attachment: Arc<crate::api::reader_attachment::ReaderAttachment>,
    /// Independent executor: a BEGIN here does not affect the writable Database.
    executor: Mutex<Executor>,
    /// Sticky `SwmrPendingDdl` summary. Once set, every subsequent
    /// refresh re-raises until the handle is dropped.
    swmr_must_reopen_summary: Mutex<Option<String>>,
    /// WAL-tail cursor; drives DDL detection and the WAL pin LSN.
    overlay: Arc<OverlayStore>,
    /// Per-table checkpoint_lsn snapshot from the last refresh, used
    /// to scope cache invalidation on reload.
    last_seen_table_lsns: Mutex<rustc_hash::FxHashMap<String, u64>>,
    /// Last manifest epoch this handle observed.
    last_seen_epoch: AtomicU64,
    auto_refresh: AtomicBool,
    refresh_ticker: Mutex<Option<RefreshTickerHandle>>,
    refresh_interval: Mutex<Option<Duration>>,
    /// Synchronous fence: `set_auto_refresh(false)` acquires it briefly
    /// so any in-flight ticker iteration completes before the call
    /// returns. Per-query auto-refresh runs on the caller's thread and
    /// does not need it.
    refresh_fence: Mutex<()>,
}

impl Drop for ReadOnlyDatabaseInner {
    fn drop(&mut self) {
        // Stop the background ticker so it doesn't leak the OS thread
        // sleeping until its next tick.
        if let Ok(mut guard) = self.refresh_ticker.lock() {
            if let Some(mut handle) = guard.take() {
                handle.shutdown();
            }
        }
    }
}

/// Read-only handle. See [`crate::api::Database::open_read_only`] /
/// [`crate::api::Database::as_read_only`]; see module docs for SWMR
/// semantics. Wraps `Arc<ReadOnlyDatabaseInner>` so the per-handle WAL
/// pin and refresh state are shared with `read_engine()`'s trait object.
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
    pub(crate) fn from_entry(entry: Arc<EngineEntry>) -> Result<Self> {
        // Read-only executor sharing the entry's semantic + planner
        // caches so writer-side DML/ANALYZE invalidates this view too.
        let engine = Arc::clone(&entry.engine);
        let semantic_cache = Arc::clone(&entry.semantic_cache);
        let query_planner = Arc::clone(&entry.query_planner);
        let query_cache = Arc::clone(&entry.query_cache);
        let executor = Executor::with_shared_semantic_cache_read_only(
            engine,
            semantic_cache,
            query_planner,
            query_cache,
        );

        // Seed from the entry's load-time epoch (not the live file): a
        // checkpoint that lands between entry creation and now would
        // otherwise look already-applied and skip `reload_manifests`.
        let last_seen_epoch_init = if entry.engine.is_read_only_mode() {
            entry.loaded_epoch
        } else {
            0
        };
        let attachment =
            crate::api::reader_attachment::ReaderAttachment::attach(Arc::clone(&entry))?;

        // Seed the overlay cursor at the entry's saved attach values
        // verbatim, re-sampling shm here would race a writer publish
        // and seed past WAL the engine never replayed.
        let overlay = Arc::new(if attachment.pin_active() {
            OverlayStore::with_baseline(
                entry.attach_visible_commit_lsn,
                entry.attach_oldest_active_txn_lsn,
            )
        } else {
            OverlayStore::new()
        });

        Ok(Self {
            inner: Arc::new(ReadOnlyDatabaseInner {
                entry,
                attachment,
                executor: Mutex::new(executor),
                last_seen_epoch: AtomicU64::new(last_seen_epoch_init),
                auto_refresh: AtomicBool::new(true),
                swmr_must_reopen_summary: Mutex::new(None),
                overlay,
                last_seen_table_lsns: Mutex::new(rustc_hash::FxHashMap::default()),
                refresh_ticker: Mutex::new(None),
                refresh_interval: Mutex::new(None),
                refresh_fence: Mutex::new(()),
            }),
        })
    }

    /// Clone for multi-threaded use. Each clone shares the engine and
    /// caches but gets its own executor, attachment (fresh WAL pin),
    /// auto-refresh flag, and overlay cursor. Mirrors
    /// [`crate::api::Database::clone`].
    pub fn try_clone(&self) -> Result<Self> {
        let cloned = Self::from_entry(Arc::clone(&self.inner.entry))?;
        cloned.set_auto_refresh(self.auto_refresh_enabled());
        if let Some(d) = self.refresh_interval() {
            cloned.set_refresh_interval(Some(d))?;
        }
        Ok(cloned)
    }
}

impl Clone for ReadOnlyDatabase {
    fn clone(&self) -> Self {
        self.try_clone().expect("ReadOnlyDatabase clone failed")
    }
}

impl ReadOnlyDatabase {
    /// Borrow the WAL-tail cursor. Exposed for tests and diagnostics.
    pub fn overlay(&self) -> Arc<OverlayStore> {
        Arc::clone(&self.overlay)
    }

    /// Returns the DSN this handle was opened with.
    pub fn dsn(&self) -> &str {
        &self.entry.dsn
    }

    /// Always returns `true`. Symmetric with
    /// [`crate::api::database::Database::is_read_only`].
    #[inline]
    pub fn is_read_only(&self) -> bool {
        true
    }

    /// Engine as a read-only trait object. The returned guard reuses
    /// this handle's existing pin and overlay (no new attachment), so
    /// refresh through the trait object advances `self`'s state in
    /// place. Holding the trait object keeps `self`'s inner alive.
    pub fn read_engine(&self) -> Arc<dyn crate::storage::traits::ReadEngine> {
        Arc::new(crate::api::database::SwmrReadEngineGuard {
            engine: Arc::clone(&self.entry.engine),
            entry: Arc::clone(&self.entry),
            refresh_owner: crate::api::database::RefreshOwner::ReadOnly(Arc::clone(&self.inner)),
        }) as Arc<dyn crate::storage::traits::ReadEngine>
    }

    /// Returns `true` if a table with the given name exists. Runs the
    /// same SWMR maintenance as `query()`.
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        Engine::table_exists(&*self.entry.engine, name)
    }

    /// Row count of `name` (hot + cold) visible to this handle's snapshot.
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

    /// Rate-limited lease mtime touch. Does NOT advance `pinned_lsn`,
    /// pin advance lives in `advance_pin_after_refresh()`, called only
    /// after the overlay actually applied the entries.
    #[inline]
    fn touch_lease(&self) {
        self.attachment.touch_lease();
    }

    /// Advance the WAL pin to `overlay.next_entry_floor` after a
    /// successful refresh. Floor at 1: pin=0 is the released sentinel.
    #[inline]
    fn advance_pin_after_refresh(&self) {
        let pin = self.overlay.next_entry_floor().max(1);
        self.attachment.advance_pin(pin);
    }

    /// Toggle implicit refresh. Default `true`: per-query polls the
    /// manifest epoch and the background ticker (if any) refreshes on
    /// each tick. Setting `false` pauses both paths until re-enabled
    /// or `refresh()` is called explicitly. Per-handle.
    pub fn set_auto_refresh(&self, enabled: bool) {
        // Disable: synchronously fence in-flight ticker iterations so
        // no implicit refresh can start after this call returns.
        let _fence = if !enabled {
            self.refresh_fence.lock().ok()
        } else {
            None
        };
        self.auto_refresh.store(enabled, Ordering::Release);
    }

    /// Whether auto-refresh is enabled. See [`Self::set_auto_refresh`].
    pub fn auto_refresh_enabled(&self) -> bool {
        self.auto_refresh.load(Ordering::Acquire)
    }

    #[doc(hidden)]
    pub fn last_seen_epoch_for_test(&self) -> u64 {
        self.last_seen_epoch.load(Ordering::Acquire)
    }

    /// Configure (or stop) the background refresh ticker. `Some(d)`
    /// spawns (or replaces) a thread calling [`Self::refresh`] every
    /// `d` (>= 100ms) so an idle handle keeps advancing its WAL pin.
    /// `None` stops the ticker. Paused while `auto_refresh == false`.
    /// Exits without respawn on a must-reopen error.
    pub fn set_refresh_interval(&self, interval: Option<Duration>) -> Result<()> {
        if let Some(d) = interval {
            if d < MIN_REFRESH_INTERVAL {
                return Err(Error::invalid_argument(format!(
                    "refresh_interval must be at least {}ms (got {}ms); for tighter \
                     cadence, drive refresh() from your own loop",
                    MIN_REFRESH_INTERVAL.as_millis(),
                    d.as_millis()
                )));
            }
        }

        let mut ticker_guard = self
            .refresh_ticker
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("refresh_ticker".to_string()))?;
        let mut interval_guard = self
            .refresh_interval
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("refresh_interval".to_string()))?;

        // Always join the existing ticker first to avoid a double-tick
        // window with the new one.
        if let Some(mut handle) = ticker_guard.take() {
            handle.shutdown();
        }
        *interval_guard = None;

        let Some(d) = interval else {
            return Ok(());
        };

        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let stop_for_thread = Arc::clone(&stop);
        let weak_inner = Arc::downgrade(&self.inner);
        let join = std::thread::Builder::new()
            .name("stoolap-ro-refresh".to_string())
            .spawn(move || run_refresh_ticker(weak_inner, d, stop_for_thread))
            .map_err(|e| {
                Error::internal(format!("failed to spawn refresh ticker thread: {}", e))
            })?;
        let thread_id = join.thread().id();

        *ticker_guard = Some(RefreshTickerHandle {
            stop,
            join: Some(join),
            thread_id,
        });
        *interval_guard = Some(d);
        Ok(())
    }

    /// Current refresh interval, or `None` if no ticker is running.
    pub fn refresh_interval(&self) -> Option<Duration> {
        self.refresh_interval.lock().ok().and_then(|g| *g)
    }

    /// Cheap auto-refresh hook called from every query path. Skipped
    /// when disabled or while a BEGIN is open. Transient errors are
    /// swallowed; typed must-reopen and `SchemaChanged` propagate.
    #[inline]
    pub(crate) fn maybe_auto_refresh(&self) -> Result<()> {
        if !self.auto_refresh.load(Ordering::Acquire) {
            return Ok(());
        }
        if let Ok(executor) = self.executor.lock() {
            if executor.has_active_transaction() {
                return Ok(());
            }
        }
        match self.refresh() {
            Ok(_) => {}
            Err(e @ Error::SchemaChanged(_)) => return Err(e),
            Err(e @ Error::SwmrPendingDdl(_)) => return Err(e),
            Err(e @ Error::SwmrWriterReincarnated { .. }) => return Err(e),
            Err(e @ Error::SwmrSnapshotExpired { .. }) => return Err(e),
            Err(e @ Error::SwmrOverlayApplyFailed(_)) => return Err(e),
            Err(e @ Error::SwmrPartialReload(_)) => return Err(e),
            Err(_) => {}
        }
        Ok(())
    }

    /// Refresh this handle against the writer's latest checkpoint.
    /// Polls `<db>/volumes/epoch`; reloads per-table manifests on
    /// advance and returns `Ok(true)`. Sub-checkpoint DML is not
    /// materialized; post-attach DDL surfaces as `SwmrPendingDdl`
    /// (sticky). No-op for in-process / memory handles.
    pub fn refresh(&self) -> Result<bool> {
        // Sticky SwmrPendingDdl re-raise: without this a retry would
        // hit the rebuild's no-op fast path and silently run against
        // a stale schema.
        if let Ok(guard) = self.swmr_must_reopen_summary.lock() {
            if let Some(ref summary) = *guard {
                return Err(Error::SwmrPendingDdl(summary.clone()));
            }
        }

        let path = self.entry.engine.get_path();
        if path.is_empty() || !self.entry.engine.is_read_only_mode() {
            return Ok(false);
        }

        // Writer reincarnation check before any per-handle work.
        if let Some(handle) = self.entry.shm.as_ref() {
            let observed = handle.header().writer_generation.load(Ordering::Acquire);
            let expected = self.attachment.expected_writer_gen();
            // ANY mismatch is a reincarnation: a crash inside
            // create_writer can leave gen=0 on disk, so `observed < expected`
            // also signals reincarnation and a `>` check would miss it.
            if expected > 0 && observed != expected {
                return Err(Error::SwmrWriterReincarnated {
                    observed_gen: observed,
                    expected_gen: expected,
                });
            }
        } else {
            // No-shm attach (no writer was live). A writer that ran +
            // closed post-attach leaves `db.shm` with a bumped gen on
            // disk; probe it so a started/committed/closed sequence is
            // detected even though no writer is currently alive.
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
            // Reload, THEN publish the new epoch. Reverse order would
            // make a crash between the two skip the reload forever.
            if !self.entry.engine.reload_manifests()? {
                self.touch_lease();
                return Ok(false);
            }

            // Per-table cache invalidation: only tables whose
            // checkpoint_lsn advanced (or are newly present) get
            // invalidated. Unchanged tables keep their cached entries.
            let now_lsns = self.entry.engine.table_checkpoint_lsns();
            let mut prev = self
                .last_seen_table_lsns
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let executor_guard = self.executor.lock().ok();
            for (table, &now_lsn) in now_lsns.iter() {
                let prev_lsn = prev.get(table).copied().unwrap_or(0);
                if now_lsn > prev_lsn || !prev.contains_key(table) {
                    self.entry.semantic_cache.invalidate_table(table);
                    self.entry.query_planner.invalidate_stats_cache(table);
                    if let Some(ref ex) = executor_guard {
                        ex.invalidate_query_cache_for_table(table);
                    }
                }
            }
            // DROP TABLE on the writer side: keys gone from now_lsns.
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

            self.last_seen_epoch.store(on_disk, Ordering::Release);
            // Publish to per-handle epoch file: writer's reaper uses
            // MIN across live handles to gate retire-sidecar unlinks.
            // Only fires after reload_manifests() succeeded.
            self.attachment.publish_loaded_epoch(on_disk);
        }
        // Rebuild WAL-tail overlay even when manifests didn't change:
        // hot-buffer commits bump visible_commit_lsn without bumping
        // the manifest epoch.
        //
        // Pin advance after a successful tail (incl. SwmrPendingDdl,
        // where the tail completed and only DDL surfacing failed).
        // SwmrOverlayApplyFailed / SwmrSnapshotExpired leave cursors
        // unchanged so we must NOT advance.
        let rebuild_result = self.maybe_rebuild_overlay();
        self.touch_lease();
        let tail_succeeded = matches!(&rebuild_result, Ok(()) | Err(Error::SwmrPendingDdl(_)));
        if tail_succeeded {
            self.advance_pin_after_refresh();
        }
        rebuild_result?;
        Ok(manifests_changed)
    }

    /// Rebuild the WAL-tail overlay if the writer's published
    /// `visible_commit_lsn` exceeds the overlay's `last_applied_lsn`.
    /// Returns `Err(SwmrPendingDdl)` for post-attach DDL the reader
    /// can't apply live, `Err(SwmrOverlayApplyFailed)` on tail failure.
    #[inline]
    fn maybe_rebuild_overlay(&self) -> Result<()> {
        // Gate on `pin_active`, not just `shm.is_some()`: half-attached
        // handles have no pin, so tailing WAL would race truncation.
        if !self.attachment.pin_active() {
            return Ok(());
        }
        let shm = match self.entry.shm.as_ref() {
            Some(h) => h,
            None => return Ok(()),
        };
        // Coherent (visible, oldest) snapshot via the shm seqlock;
        // fall back to oldest=0 (forces conservative full Phase-2 scan)
        // if the writer kept publishing across all retries.
        let (target, writer_oldest_active) =
            shm.header().sample_visibility_pair().unwrap_or_else(|| {
                let v = shm.header().visible_commit_lsn.load(Ordering::Acquire);
                (v, 0)
            });
        // Post-visibility gen recheck closes the wipe race the up-front
        // check in `refresh()` can't cover (reader loads gen at T0,
        // writer reincarnates + wipes visibility at T1, reader reads
        // visibility=0 at T2 with pre-T1 expected).
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
        // Do NOT skip when target <= last_applied_lsn:
        // `rebuild_from_wal` still uses `writer_oldest_active` to
        // advance `next_entry_floor` when the writer republished a
        // higher watermark without a new visible commit.
        let wal = match self.entry.engine.wal() {
            Some(w) => w,
            None => return Ok(()),
        };
        let writer_last_ddl_lsn = shm.header().catalog_epoch.load(Ordering::Acquire);
        self.overlay
            .rebuild_from_wal(wal, target, writer_oldest_active, writer_last_ddl_lsn)?;

        // Surface DDL post-attach (lsn > attach_visible_commit_lsn) and
        // not already known. The writer re-records DDL after every WAL
        // truncation; without the known-catalog filter every checkpoint
        // would falsely surface SwmrPendingDdl for pre-existing objects.
        // Drop/Alter/Rename/Truncate are NOT filtered, those are real
        // schema changes the cached metadata can't apply live.
        let known_catalog = self.entry.engine.known_catalog_objects();
        let known_indexes = self.entry.engine.known_index_names();
        let ddl: Vec<_> = self
            .overlay
            .pending_ddl()
            .into_iter()
            .filter(|e| e.lsn > self.attachment.attach_visible_commit_lsn())
            .filter(|e| {
                // CreateIndex keys on the INDEX name decoded from the
                // IndexMetadata payload (entry.table_name is the underlying
                // table for CreateIndex and would silently absorb a new
                // index on a known table).
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
            // Bounded summary: "createtable:t,altertable:orders,...".
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
            // Sticky: next refresh re-raises instead of hitting the
            // no-op fast path.
            if let Ok(mut guard) = self.swmr_must_reopen_summary.lock() {
                if guard.is_none() {
                    *guard = Some(summary.clone());
                }
            }
            return Err(Error::SwmrPendingDdl(summary));
        }
        Ok(())
    }

    /// Execute a read-only SQL query. Rejects writes (INSERT/UPDATE/
    /// DELETE/DDL/maintenance PRAGMA/SET TRANSACTION).
    pub fn query<P: Params>(&self, sql: &str, params: P) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        // Write rejection lives inside the executor's parse/cache path
        // so we don't pre-parse here.
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
        Ok(Rows::with_entry(result, Arc::clone(&self.entry)))
    }

    /// Execute with `:name` named parameters.
    pub fn query_named(&self, sql: &str, params: crate::api::NamedParams) -> Result<Rows> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(Rows::with_entry(result, Arc::clone(&self.entry)))
    }

    /// Cache a parsed plan. Read-only equivalent of
    /// [`crate::api::Database::cached_plan`]; rejects write SQL at plan
    /// creation.
    pub fn cached_plan(&self, sql: &str) -> Result<crate::executor::query_cache::CachedPlanRef> {
        self.touch_lease();
        self.maybe_auto_refresh()?;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        executor.get_or_create_plan(sql)
    }

    /// Read-only equivalent of [`crate::api::Database::query_plan`].
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
        Ok(Rows::with_entry(result, Arc::clone(&self.entry)))
    }

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
        Ok(Rows::with_entry(result, Arc::clone(&self.entry)))
    }
}

/// Decode the index name from an `IndexMetadata` payload prefix
/// (`name_len: u16 LE` then `name: name_len UTF-8 bytes`; see
/// `IndexMetadata::serialize` in `src/storage/mvcc/persistence.rs`).
/// `None` on decode failure: caller treats it as "not a known re-record"
/// (false positive over false negative).
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
