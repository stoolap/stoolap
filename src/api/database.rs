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
    /// Temp directory for test-filedb feature. Deleted with the entry.
    #[cfg(feature = "test-filedb")]
    _temp_dir: Option<tempfile::TempDir>,
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
    executor: Mutex<Executor>,
}

/// Type alias for Statement to use (avoids exposing DatabaseInner directly)
pub(crate) type DatabaseInnerHandle = DatabaseInner;

impl DatabaseInner {
    /// Build a fresh per-handle inner around an existing engine entry.
    /// Picks a writable or read-only executor to match the engine mode,
    /// and shares the engine entry's semantic cache and query planner so
    /// DML invalidation and ANALYZE reach every sibling reader.
    fn new_with_entry(entry: Arc<EngineEntry>) -> Self {
        let engine = Arc::clone(&entry.engine);
        let semantic_cache = Arc::clone(&entry.semantic_cache);
        let query_planner = Arc::clone(&entry.query_planner);
        let executor = if entry.engine.is_read_only_mode() {
            Executor::with_shared_semantic_cache_read_only(engine, semantic_cache, query_planner)
        } else {
            Executor::with_shared_semantic_cache(engine, semantic_cache, query_planner)
        };
        Self {
            entry,
            executor: Mutex::new(executor),
        }
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
        // Parse the DSN's read_only flag upfront so registry sharing knows
        // whether the new request matches the cached engine's mode.
        let requested_ro = Self::dsn_requests_read_only(dsn)?;

        // Read-only is meaningless on `memory://`: a fresh in-memory engine
        // has nothing to read. Reject early with a clear diagnostic instead
        // of silently constructing an engine that can never serve a useful
        // query. Use `file://` for read-only deployments.
        if requested_ro && dsn.starts_with(MEMORY_SCHEME) {
            return Err(Error::invalid_argument(
                "read_only is not supported on memory:// (a fresh in-memory \
                 engine has no data to read); use file:// for read-only \
                 deployments",
            ));
        }

        // Check if we already have an engine for this DSN
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::LockAcquisitionFailed("registry read".to_string()))?;
            if let Some(weak) = registry.get(dsn) {
                if let Some(entry) = weak.upgrade() {
                    let cached_ro = entry.engine.is_read_only_mode();
                    // Mode mismatch is rejected: a read-only-cached engine
                    // cannot serve a writable request (would bypass the file
                    // lock and WAL guarantees), and a writable-cached engine
                    // serving a read-only request would hand out a
                    // write-capable executor in disguise.
                    if cached_ro != requested_ro {
                        return Err(Error::read_only_mode_mismatch(dsn, cached_ro, requested_ro));
                    }
                    // Independent per-handle state, shared engine entry.
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
                let cached_ro = entry.engine.is_read_only_mode();
                if cached_ro != requested_ro {
                    return Err(Error::read_only_mode_mismatch(dsn, cached_ro, requested_ro));
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

        // Create the engine based on scheme
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
                // Parse optional query parameters
                let (clean_path, config) = Self::parse_file_config(&path)?;

                // If the DSN requested read-only mode, refuse to materialize
                // a fresh database. Same guard as `Database::open_read_only`:
                // the path must already exist as a directory containing a
                // recognizable stoolap layout (`wal/` or `volumes/`).
                // Without this, `open("file://.../missing?read_only=true")`
                // would silently create an empty DB via PersistenceManager.
                if config.read_only {
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
                    let has_wal = path_obj.join("wal").exists();
                    let has_volumes = path_obj.join("volumes").exists();
                    if !has_wal && !has_volumes {
                        return Err(Error::internal(format!(
                            "cannot open '{}' read-only: not a stoolap database \
                             (no wal/ or volumes/ directory)",
                            clean_path
                        )));
                    }
                }

                let engine = MVCCEngine::new(config);
                engine.open_engine()?;
                let engine = Arc::new(engine);
                // Start background cleanup (uses config from engine)
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

        // Build the engine entry. The executor is created per-handle in
        // `share_entry` and inherits the engine's read-only mode (so a DSN
        // with `?read_only=true` mirrors the parser write gate plus the DML
        // auto-commit guard on every handle constructed from this entry).
        let semantic_cache = Arc::new(crate::executor::SemanticCache::default());
        let query_planner = Arc::new(crate::executor::QueryPlanner::new(Arc::clone(&engine)));
        let entry = Arc::new(EngineEntry {
            engine,
            dsn: dsn.to_string(),
            semantic_cache,
            query_planner,
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
        // has nothing to read. Reject early. (Mirrored on `Database::open`
        // for the `?read_only=true` query-param path.)
        if dsn.starts_with(MEMORY_SCHEME) {
            return Err(Error::invalid_argument(
                "open_read_only is not supported on memory:// (a fresh \
                 in-memory engine has no data to read); use file:// for \
                 read-only deployments",
            ));
        }

        // If the DSN is already open in this process (writable or read-only),
        // share the existing engine entry. The parser-level write gate on
        // ReadOnlyDatabase still rejects all write SQL, regardless of the
        // underlying engine's mode.
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::LockAcquisitionFailed("registry read".to_string()))?;
            if let Some(weak) = registry.get(dsn) {
                if let Some(entry) = weak.upgrade() {
                    return Ok(crate::api::ReadOnlyDatabase::from_entry(entry));
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
                return Ok(crate::api::ReadOnlyDatabase::from_entry(entry));
            }
        }

        let (scheme, path) = Self::parse_dsn(dsn)?;

        #[cfg(feature = "test-filedb")]
        let _temp_dir_holder: Option<tempfile::TempDir> = None;

        // memory:// was rejected at the top of this function. Only file://
        // reaches the engine-construction match.
        let engine = match scheme.as_str() {
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

                let engine = MVCCEngine::new(config);
                engine.open_engine()?;
                let engine = Arc::new(engine);
                // start_cleanup is a no-op when config.read_only is set,
                // but call it for symmetry with the writable path.
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
            #[cfg(feature = "test-filedb")]
            _temp_dir: _temp_dir_holder,
        });

        registry.insert(dsn.to_string(), Arc::downgrade(&entry));

        Ok(crate::api::ReadOnlyDatabase::from_entry(entry))
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
        });
        Ok(Self::share_entry(entry))
    }

    /// Parse a DSN into scheme and path
    /// Returns true if the DSN's query string requests read-only mode
    /// (`?read_only=true`, `?readonly=true`, or `?mode=ro`).
    ///
    /// Used by `Database::open` to make the registry-share decision before
    /// constructing the engine. For DSNs without a query string, returns
    /// `false`. For DSNs with conflicting / malformed values, returns the
    /// same parse error that `parse_file_config` would return.
    fn dsn_requests_read_only(dsn: &str) -> Result<bool> {
        // Find the query string portion; identical scan logic to parse_dsn
        // but without requiring a valid scheme (parse_dsn runs separately).
        //
        // Must mirror `parse_file_config`'s precedence: scan ALL params and
        // let the LAST recognized read-only flag win, regardless of which
        // key it used (`read_only`, `readonly`, `mode`). Returning on the
        // first match would disagree with the actual config — the same DSN
        // would then open with one mode while the registry's mode-mismatch
        // check at the next `open(dsn)` call computed the other, refusing
        // a perfectly idempotent reopen.
        let query = match dsn.find('?') {
            Some(idx) => &dsn[idx + 1..],
            None => return Ok(false),
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
        Ok(last.unwrap_or(false))
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
    pub fn execute<P: Params>(&self, sql: &str, params: P) -> Result<i64> {
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
        Arc::clone(&self.inner.entry.engine) as Arc<dyn crate::storage::traits::ReadEngine>
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
        // Last-handle detection uses the engine entry's strong count.
        // Each user-visible handle (Database, clone, sibling open, RO)
        // owns one Arc<EngineEntry>; nothing else holds an Arc to the
        // entry (the registry stores a Weak; the executor and lazy
        // QueryPlanner clone Arc<MVCCEngine>, not Arc<EngineEntry>). So
        // `strong_count == 1` means "this handle is the only one alive
        // for the DSN", and the engine can close now without disturbing
        // siblings.
        //
        // The strong_count check MUST happen under the registry write
        // lock. Otherwise a concurrent `Database::open(dsn)` could read
        // the registry, upgrade the still-live Weak, and return a fresh
        // handle to its caller — between our check and our `close_engine`
        // call — leaving that caller holding a Database whose engine is
        // closed under it.
        let mut registry = match DATABASE_REGISTRY.write() {
            Ok(g) => g,
            Err(_) => return Err(Error::LockAcquisitionFailed("registry write".to_string())),
        };
        if Arc::strong_count(&self.inner.entry) != 1 {
            return Ok(());
        }
        // Proactively clear the registry's dead-soon Weak so the next
        // `open(dsn)` doesn't have to upgrade-and-fail.
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
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
        executor.get_or_create_plan(sql)
    }

    /// Execute a pre-cached plan with positional parameters (no parsing, no cache lookup).
    pub fn execute_plan<P: Params>(&self, plan: &CachedPlanRef, params: P) -> Result<i64> {
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
        // Read-only path: a `ReadTransaction` is enough for `get_read_table`,
        // and it works on both writable and read-only engines without any
        // gate bypass.
        let engine = &self.inner.entry.engine;
        let tx = ReadEngine::begin_read_transaction(engine.as_ref())?;
        Ok(tx.get_read_table(name).is_ok())
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
    pub(crate) fn executor(&self) -> &Mutex<Executor> {
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
