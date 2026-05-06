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

//! Database struct and operations.

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

/// Parse `"Nms"`, `"Ns"`, `"Nm"`, or `"0"` into a Duration.
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

/// Per-DSN registry. Weak refs so entries self-expire when the last user handle drops.
static DATABASE_REGISTRY: std::sync::LazyLock<
    RwLock<FxHashMap<String, std::sync::Weak<EngineEntry>>>,
> = std::sync::LazyLock::new(|| RwLock::new(FxHashMap::default()));

/// Engine-level shared state, keyed by DSN. Arc count = live user handles.
pub(crate) struct EngineEntry {
    pub(crate) engine: Arc<MVCCEngine>,
    pub(crate) dsn: String,
    /// Shared across per-handle executors so DML invalidation reaches all readers.
    pub(crate) semantic_cache: Arc<crate::executor::SemanticCache>,
    /// Shared so ANALYZE updates reach all readers' plans.
    pub(crate) query_planner: Arc<crate::executor::QueryPlanner>,
    /// SWMR lease for read-only file:// engines. `None` for in-memory or writable.
    pub(crate) lease: Option<crate::storage::mvcc::lease::LeaseManager>,
    /// Read-only mmap of writer's `db.shm`. `None` falls back to v1 mtime-only presence.
    pub(crate) shm: Option<Arc<crate::storage::mvcc::shm::ShmHandle>>,
    /// `writer_generation` snapshot at attach; refresh detects reincarnation against it.
    pub(crate) attach_writer_gen: u64,
    /// `visible_commit_lsn` snapshot at attach; SwmrPendingDdl filter suppresses pre-attach DDL.
    pub(crate) attach_visible_commit_lsn: u64,
    /// `oldest_active_txn_lsn` sampled before visible_commit_lsn, so post-attach commits
    /// can't move the floor above us. `u64::MAX` = no shm or no active txns at attach.
    pub(crate) attach_oldest_active_txn_lsn: u64,
    /// Manifest epoch read before `open_engine`; baseline for ReadOnlyDatabase refresh.
    pub(crate) loaded_epoch: u64,
    /// Epoch-millis of last lease touch; rate-limits the touch syscall to ~1Hz.
    last_lease_touch_ms: std::sync::atomic::AtomicU64,
    /// Long-lived SH lock for chmod-read-only fallback (lease registration failed on
    /// writable mount). Blocks LOCK_EX so a privileged writer can't reclaim under us.
    #[allow(dead_code)]
    pub(crate) chmod_ro_lock: Option<crate::storage::mvcc::file_lock::SharedLockGuard>,
    /// Temp directory for test-filedb feature. Deleted with the entry.
    #[cfg(feature = "test-filedb")]
    _temp_dir: Option<tempfile::TempDir>,
}

/// Probe `db_path` writability via a throwaway file. Used to classify lease failures.
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
    /// Rate-limited (~1Hz) mtime touch on the SWMR lease. No-op without a lease.
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

    /// Pre-acquire SWMR lease + shm + attach snapshots before engine construction.
    /// Lease registers first (writer's GC sees us during WAL replay); shm sample
    /// drives both engine `replay_cap_lsn` and EngineEntry's attach snapshot.
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
        // Lease registration failure: accept on RO mount or chmod-RO dir, fail otherwise.
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
        // Pin WAL=1 BEFORE shm sample so a concurrent checkpoint can't truncate our
        // replay range. Use `set_handle_pin` with a fresh id to coexist with sibling
        // contributions; RAII guard releases on early-Err so we never leak a pin of 1.
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
        let shm_path = path.join(crate::storage::mvcc::shm::SHM_FILENAME);

        // Always handshake: shm file presence doesn't prove a live writer (close_engine
        // leaves db.shm behind). Outcome classifies writer state and gates capped vs
        // uncapped WAL replay.
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
        // Best-effort 250ms poll for the writer to publish a fresh seqlock pair.
        // Not correctness-required (WAL→shm fetch_min mirror covers the lower bound).
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
                // Wait for an EVEN seq > baseline (in-flight publish leaves it odd).
                if seq > baseline_seq && seq.is_multiple_of(2) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        let mut attach_gen: u64 = 0;
        let mut attach_oldest_active: u64 = u64::MAX;
        let mut attach_lsn: u64 = 0;
        // No-shm path: still probe db.shm to capture writer_generation as baseline so
        // a future writer reincarnation is detected on refresh.
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
        // LiveWriter post-sample liveness recheck: writer may have exited mid-sample.
        if let Some(sg) = startup_guard.take() {
            match crate::storage::mvcc::file_lock::FileLock::recheck_writer_still_holds_lock(path)?
            {
                Some(db_lock_g) => {
                    // Writer exited mid-sample. Discard shm; preserve attach_gen
                    // (refresh check needs the leftover gen as baseline).
                    handshake_guard = Some(db_lock_g);
                    shm_is_stale_leftover = true;
                    shm = None;
                    attach_oldest_active = u64::MAX;
                    attach_lsn = 0;
                }
                None => {
                    // Writer still alive. Require shm — uncapped replay would race appends.
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
                    drop(sg);
                }
            }
        }

        let shm_file_exists = shm_path.exists() && !shm_is_stale_leftover;
        if shm_file_exists && shm.is_none() {
            // shm exists but unattachable: accept on RO/chmod-RO (leftover), fail otherwise.
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
            // Effectively-RO path: drop the failed-pin contribution.
            if let Some(ref l) = lease {
                l.remove_handle_pin(pre_acquire_pin_handle_id);
            }
        }
        // No-shm path: caller holds the guard across open_engine to fence new writers.
        let returned_guard = if shm.is_none() { handshake_guard } else { None };
        // Defuse the RAII guard so the contribution survives for the caller.
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
        // Release thread-local caches that hold engine-internal Arcs/closures.
        crate::executor::clear_all_thread_local_caches();
        let _ = self.engine.close_engine();

        // Reap our dead Weak from the registry; skip if a sibling re-inserted live.
        if let Ok(mut registry) = DATABASE_REGISTRY.try_write() {
            if let Some(weak) = registry.get(&self.dsn) {
                if weak.strong_count() == 0 {
                    registry.remove(&self.dsn);
                }
            }
        }
    }
}

/// Per-handle state. Each handle gets its own executor so BEGIN doesn't leak across.
pub(crate) struct DatabaseInner {
    entry: Arc<EngineEntry>,
    /// Shared so `read_engine`'s guard can check for an active BEGIN on this handle.
    executor: Arc<Mutex<Executor>>,
}

pub(crate) type DatabaseInnerHandle = DatabaseInner;

impl DatabaseInner {
    /// Build a fresh writable per-handle inner sharing the entry's caches.
    fn new_with_entry(entry: Arc<EngineEntry>) -> Self {
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

/// `ReadEngine` wrapper returned by `read_engine()`. Heartbeats lease and runs the
/// per-handle refresh on every `begin_read_transaction*` call.
pub(crate) struct SwmrReadEngineGuard {
    pub(crate) engine: Arc<MVCCEngine>,
    pub(crate) entry: Arc<EngineEntry>,
    pub(crate) refresh_owner: RefreshOwner,
}

pub(crate) enum RefreshOwner {
    ReadOnly(Arc<crate::api::read_only_database::ReadOnlyDatabaseInner>),
    None,
}

impl SwmrReadEngineGuard {
    #[inline]
    fn maintain(&self) -> Result<()> {
        self.entry.heartbeat_swmr_lease();
        match &self.refresh_owner {
            RefreshOwner::ReadOnly(inner) => {
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

/// Stoolap database connection. Thread-safe; clone to share across threads
/// (each clone gets independent transaction state but shares the engine).
pub struct Database {
    inner: Arc<DatabaseInner>,
}

#[cfg(feature = "ffi")]
impl Database {
    /// FFI keepalive: holds inner Arc to keep the engine alive across handles.
    pub(crate) fn keepalive(&self) -> Arc<DatabaseInner> {
        Arc::clone(&self.inner)
    }

    pub(crate) fn inner_arc(&self) -> &Arc<DatabaseInner> {
        &self.inner
    }
}

impl Database {
    /// FFI-only registry tidying for `stoolap_close`. No-op when other handles exist.
    #[cfg(feature = "ffi")]
    pub(crate) fn try_unregister_arc(inner: &Arc<DatabaseInner>) {
        if Arc::strong_count(inner) > 1 {
            return;
        }
        if Arc::strong_count(&inner.entry) > 1 {
            return;
        }
        if let Ok(mut registry) = DATABASE_REGISTRY.write() {
            if let Some(weak) = registry.get(&inner.entry.dsn) {
                match weak.upgrade() {
                    Some(reg_entry) if Arc::ptr_eq(&reg_entry, &inner.entry) => {
                        registry.remove(&inner.entry.dsn);
                    }
                    None => {
                        registry.remove(&inner.entry.dsn);
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Database {
    /// New handle sharing `entry` but with its own executor (independent BEGIN state).
    fn share_entry(entry: Arc<EngineEntry>) -> Database {
        Database {
            inner: Arc::new(DatabaseInner::new_with_entry(entry)),
        }
    }
}

impl Clone for Database {
    /// Clone gets its own executor (independent transaction state); shares the engine.
    fn clone(&self) -> Self {
        Database::share_entry(Arc::clone(&self.inner.entry))
    }
}

// No Drop impl: dropping `inner` drops the per-handle `Arc<EngineEntry>`; when the
// last handle drops, `EngineEntry::drop` closes the engine and reaps the Weak.

impl Database {
    /// Open a database connection.
    ///
    /// DSN forms:
    /// - `memory://` — in-memory (data lost on close).
    /// - `file:///path/to/db` — persistent.
    ///
    /// Opening the same DSN multiple times returns the same engine instance.
    /// Read-only DSNs (`?read_only=true` / `?mode=ro`) must use `open_read_only`.
    pub fn open(dsn: &str) -> Result<Self> {
        // Read-only DSNs go through `open_read_only` for compile-time type-level enforcement.
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

        // Reuse a writable cached entry; reject read-only-cached entries with mode mismatch.
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
            }
        }

        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::LockAcquisitionFailed("registry write".to_string()))?;

        // Re-check under write lock.
        if let Some(weak) = registry.get(dsn) {
            if let Some(entry) = weak.upgrade() {
                if entry.engine.is_read_only_mode() {
                    return Err(Error::read_only_mode_mismatch(dsn, true, false));
                }
                return Ok(Self::share_entry(entry));
            }
        }

        let (scheme, path) = Self::parse_dsn(dsn)?;

        #[cfg(feature = "test-filedb")]
        let mut _temp_dir_holder: Option<tempfile::TempDir> = None;

        // Writable-only path: no SWMR lease/shm setup.
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

        registry.insert(dsn.to_string(), Arc::downgrade(&entry));

        Ok(Self::share_entry(entry))
    }

    /// Open a fresh in-memory database. Unlike `open("memory://")`, each call is unique.
    pub fn open_in_memory() -> Result<Self> {
        Self::create_in_memory_engine()
    }

    /// Open a read-only handle over an existing `file://` database. The returned
    /// `ReadOnlyDatabase` rejects write SQL and registers a cross-process presence
    /// lease (issue at least one query per `2 * checkpoint_interval` to keep it fresh).
    pub fn open_read_only(dsn: &str) -> Result<crate::api::ReadOnlyDatabase> {
        if dsn.starts_with(MEMORY_SCHEME) {
            return Err(Error::invalid_argument(
                "open_read_only is not supported on memory:// (a fresh \
                 in-memory engine has no data to read); use file:// for \
                 read-only deployments",
            ));
        }

        // Reject DSN flag that explicitly requests writable; redundant RO flags accepted.
        if Self::dsn_read_only_flag(dsn)? == Some(false) {
            return Err(Error::invalid_argument(
                "Database::open_read_only called with a DSN flag that \
                 explicitly requests writable mode (?read_only=false / \
                 ?readonly=false / ?mode=rw). The function name and the \
                 DSN flag disagree — drop the flag (it's redundant on \
                 open_read_only) or use Database::open(dsn) instead.",
            ));
        }

        // Parse refresh flags up front so errors surface before any engine work.
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

        // Entry reuse requires shm present, frontier static, and writer_gen unchanged
        // since attach; otherwise fall through to a fresh open with new attach snapshot.
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
        // Writable cached entries are always reusable (equivalent to `as_read_only`).
        let cached_is_writable =
            |entry: &Arc<EngineEntry>| -> bool { !entry.engine.is_read_only_mode() };
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::LockAcquisitionFailed("registry read".to_string()))?;
            if let Some(weak) = registry.get(dsn) {
                if let Some(entry) = weak.upgrade() {
                    if cached_is_writable(&entry) || frontier_static(&entry) {
                        return apply_dsn_flags(crate::api::ReadOnlyDatabase::from_entry(entry)?);
                    }
                }
            }
        }

        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::LockAcquisitionFailed("registry write".to_string()))?;

        if let Some(weak) = registry.get(dsn) {
            if let Some(entry) = weak.upgrade() {
                if cached_is_writable(&entry) || frontier_static(&entry) {
                    return apply_dsn_flags(crate::api::ReadOnlyDatabase::from_entry(entry)?);
                }
            }
        }

        let (scheme, path) = Self::parse_dsn(dsn)?;

        #[cfg(feature = "test-filedb")]
        let _temp_dir_holder: Option<tempfile::TempDir> = None;

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

                // Refuse to materialize a fresh DB on a read-only open.
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
                // Require wal/ or volumes/ to confirm this is an existing stoolap DB.
                let has_wal = path_obj.join("wal").exists();
                let has_volumes = path_obj.join("volumes").exists();
                if !has_wal && !has_volumes {
                    return Err(Error::internal(format!(
                        "cannot open '{}' read-only: not a stoolap database \
                             (no wal/ or volumes/ directory)",
                        clean_path
                    )));
                }

                // Pre-acquire SWMR lease + shm + attach snapshots before engine open.
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
                // Snapshot manifest epoch before open_engine; from_entry seeds it as baseline.
                let loaded_epoch = crate::storage::mvcc::manifest_epoch::read_epoch(
                    std::path::Path::new(&clean_path),
                )
                .unwrap_or(0);
                let engine = MVCCEngine::new(config);
                if shm.is_some() {
                    engine.set_replay_cap_lsn(attach_visible_commit_lsn);
                }
                // Don't leak the pre-acquire pin contribution on open failure.
                if let Err(e) = engine.open_engine() {
                    if let Some(ref l) = lease {
                        l.remove_handle_pin(pre_acquire_pin_handle_id);
                    }
                    return Err(e);
                }
                let engine = Arc::new(engine);
                // No-op for read_only configs but call for symmetry.
                engine.start_cleanup();
                // Pre-acquire pin stays until `from_entry` installs the per-handle pin.
                // Persist handshake guard for chmod-RO / RO-mount lease=None paths.
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
        // No entry-scoped pin: `from_entry` installs an advancing per-handle pin instead.
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

        // Release pre-acquire pin on either path: success means
        // from_entry installed its own per-handle pin (the new floor);
        // failure means the in-process lease_pin_contributions entry
        // would otherwise leak past EngineEntry::drop and constrain
        // future readers' MIN computation.
        let ro_db = match crate::api::ReadOnlyDatabase::from_entry(entry.clone()) {
            Ok(db) => db,
            Err(e) => {
                if let Some(ref l) = entry.lease {
                    l.remove_handle_pin(pre_acquire_pin_handle_id);
                }
                return Err(e);
            }
        };
        if let Some(ref l) = entry.lease {
            l.remove_handle_pin(pre_acquire_pin_handle_id);
        }
        apply_dsn_flags(ro_db)
    }

    /// Return a read-only view sharing this database's engine. Writes through
    /// the view are rejected at query time. The view has its own executor:
    /// uncommitted writes from this `Database`'s transactions are not visible.
    pub fn as_read_only(&self) -> crate::api::ReadOnlyDatabase {
        // Writable entries have no lease, so attach() can't fail on the
        // epoch-file path. Any failure here is a hard internal error.
        crate::api::ReadOnlyDatabase::from_entry(Arc::clone(&self.inner.entry))
            .expect("as_read_only on a writable Database cannot fail")
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

    /// Parse `auto_refresh=on/off/...` from the DSN. `Ok(None)` if absent.
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

    /// Parse `refresh_interval=Nms|Ns|Nm|0` from the DSN. `0` = explicitly disabled.
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
                    // Read-only mode: read_only=true / readonly=true / mode=ro
                    "read_only" | "readonly" | "mode" => {
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

    /// Per-call SWMR maintenance: heartbeat the lease. No-op for writable engines.
    #[inline]
    pub(crate) fn heartbeat_and_maybe_refresh(&self) -> Result<()> {
        self.inner.entry.heartbeat_swmr_lease();
        Ok(())
    }

    /// Execute a DDL or DML statement. Returns rows affected (0 for DDL).
    /// Parameters: `()`, tuple `(1, "Alice")`, or `params![...]`.
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

    /// Execute a query and return rows. Single-param tuples need a trailing comma: `(v,)`.
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

    /// Query a single value from a single-row, single-column result.
    /// Errors if no rows returned.
    pub fn query_one<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<T> {
        let row = self
            .query(sql, params)?
            .next()
            .ok_or(Error::NoRowsReturned)??;
        row.get(0)
    }

    /// Like `query_one` but returns `None` instead of erroring on no rows.
    pub fn query_opt<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<Option<T>> {
        match self.query(sql, params)?.next() {
            Some(row) => Ok(Some(row?.get(0)?)),
            None => Ok(None),
        }
    }

    /// Like `execute` with a timeout in ms. `0` = no timeout.
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

    /// Like `query` with a timeout in ms. `0` = no timeout.
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

    /// Prepare a SQL statement for repeated execution with different params.
    pub fn prepare(&self, sql: &str) -> Result<Statement> {
        Statement::new(Arc::downgrade(&self.inner), sql.to_string(), self)
    }

    /// Used by Statement to upgrade weak references back to a Database handle.
    pub(crate) fn from_inner(inner: Arc<DatabaseInner>) -> Self {
        Database { inner }
    }

    /// Execute a statement with `:name`-style named parameters.
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

    /// Query with `:name`-style named parameters.
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

    /// Query a single value with named parameters.
    pub fn query_one_named<T: FromValue>(&self, sql: &str, params: NamedParams) -> Result<T> {
        let mut rows = self.query_named(sql, params)?;
        match rows.next() {
            Some(Ok(row)) => row.get(0),
            Some(Err(e)) => Err(e),
            None => Err(Error::NoRowsReturned),
        }
    }

    /// Query and map each row to `T` via the `FromRow` trait.
    pub fn query_as<T: FromRow, P: Params>(&self, sql: &str, params: P) -> Result<Vec<T>> {
        let rows = self.query(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Query with named parameters and map each row to `T` via `FromRow`.
    pub fn query_as_named<T: FromRow>(&self, sql: &str, params: NamedParams) -> Result<Vec<T>> {
        let rows = self.query_named(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Begin a transaction with the default isolation level (ReadCommitted).
    pub fn begin(&self) -> Result<Transaction> {
        self.begin_with_isolation(IsolationLevel::ReadCommitted)
    }

    /// Begin a transaction with a specific isolation level.
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<Transaction> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        let tx = executor.begin_transaction_with_isolation(isolation)?;
        // Pass the entry so live txns count in `Arc::strong_count(&entry)` and `close()` defers.
        let entry = Arc::clone(&self.inner.entry);
        Ok(Transaction::new(tx, entry))
    }

    /// Get the underlying storage engine. Advanced use only.
    /// On read-only handles, write-intent methods on the engine return `Error::ReadOnlyViolation`.
    pub fn engine(&self) -> &Arc<MVCCEngine> {
        &self.inner.entry.engine
    }

    /// True if opened with `?read_only=true` / `?mode=ro`.
    pub fn is_read_only(&self) -> bool {
        self.inner.entry.engine.is_read_only_mode()
    }

    /// Get the engine as a read-only trait object — type-level enforcement of the
    /// read-only contract. Works on writable Databases too.
    pub fn read_engine(&self) -> Arc<dyn crate::storage::traits::ReadEngine> {
        // Wrap so trait-object callers also get lease heartbeat + refresh maintenance.
        Arc::new(SwmrReadEngineGuard {
            engine: Arc::clone(&self.inner.entry.engine),
            entry: Arc::clone(&self.inner.entry),
            refresh_owner: RefreshOwner::None,
        }) as Arc<dyn crate::storage::traits::ReadEngine>
    }

    /// Close the database. If sibling handles exist, defer until the last drops.
    /// Engine also closes automatically when the last handle drops.
    pub fn close(&self) -> Result<()> {
        // Decision under the registry write lock so a concurrent open can't upgrade.
        let mut registry = match DATABASE_REGISTRY.write() {
            Ok(g) => g,
            Err(_) => return Err(Error::LockAcquisitionFailed("registry write".to_string())),
        };
        if Arc::strong_count(&self.inner.entry) > 1 {
            return Ok(());
        }
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
        self.inner.entry.engine.close_engine()?;

        Ok(())
    }

    /// Cache a plan for a SQL statement; pass the returned ref to `execute_plan` / `query_plan`.
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

    /// Check if a table exists.
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        use crate::storage::traits::ReadEngine;
        self.heartbeat_and_maybe_refresh()?;
        let engine = &self.inner.entry.engine;
        let tx = ReadEngine::begin_read_transaction(engine.as_ref())?;
        Ok(tx.get_read_table(name).is_ok())
    }

    /// Rows in `name` visible to this autocommit handle. For txn-visible counts
    /// (including uncommitted writes) use [`Transaction::table_count`].
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

    /// Create backup .bin snapshot files for each table plus ddl-*.bin.
    /// No-op for in-memory. `Error::ReadOnlyViolation` on read-only handles.
    pub fn create_snapshot(&self) -> Result<()> {
        use crate::storage::Engine;
        if self.inner.entry.engine.is_read_only_mode() {
            return Err(Error::read_only_violation_at("database", "create_snapshot"));
        }
        self.inner.entry.engine.create_snapshot()
    }

    /// Restore from a backup snapshot. `timestamp` format: `"YYYYMMDD-HHMMSS.fff"`,
    /// or `None` for the latest. Destructive. `Error::ReadOnlyViolation` on read-only handles.
    pub fn restore_snapshot(&self, timestamp: Option<&str>) -> Result<String> {
        use crate::storage::Engine;
        if self.inner.entry.engine.is_read_only_mode() {
            return Err(Error::read_only_violation_at(
                "database",
                "restore_snapshot",
            ));
        }
        let result = self.inner.entry.engine.restore_snapshot(timestamp)?;
        // Data wholly changed; flush all query caches.
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

    /// Semantic query cache statistics (hits, exact and subsumption matches).
    pub fn semantic_cache_stats(&self) -> Result<crate::executor::SemanticCacheStatsSnapshot> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        Ok(executor.semantic_cache_stats())
    }

    /// Clear all cached query results.
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
