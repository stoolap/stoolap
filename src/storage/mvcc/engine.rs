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

//! MVCC Storage Engine
//!
//! Provides the main MVCC storage engine implementation.
//!
//! # Lock taxonomy
//!
//! The engine has many synchronization primitives, each protecting a
//! different invariant. New code must respect the legal acquisition
//! order or risk deadlock; new readers must respect what each lock
//! actually guarantees.
//!
//! ## Catalog / per-table state
//!
//! - `schemas: RwLock<...>` — table schema map. Read on every query
//!   plan; written by DDL. SH on the read path; EX briefly under
//!   `transactional_ddl_fence` on the write path.
//! - `version_stores: RwLock<...>` — per-table hot VersionStore map.
//!   Same shape as `schemas`.
//! - `txn_version_stores: RwLock<TxnVersionStoreMap>` — per-txn
//!   pending writes. Accessed by transaction lifecycle methods
//!   (begin / DML / commit / rollback).
//! - `views: RwLock<...>` — view definitions; same shape as schemas.
//! - `segment_managers: RwLock<...>` — per-table cold segment
//!   managers. Accessed by checkpoint / compaction / scan paths.
//! - `snapshot_timestamps: RwLock<...>` — backup snapshot index.
//!
//! ## Cross-process / single-writer state
//!
//! - `file_lock: Mutex<Option<FileLock>>` — owns the kernel-level
//!   `db.lock` LOCK_EX (writers) or LOCK_SH (readers via `Database`
//!   instances using shared locks). One per engine; established at
//!   open, released at close.
//! - `shm: Mutex<Option<Arc<ShmHandle>>>` — writer's mmap-backed
//!   `db.shm` for cross-process visibility publication. `Some` only
//!   on writable `file://` engines.
//! - `pending_marker_lsns: Mutex<BTreeSet<u64>>` — set of WAL LSNs
//!   for commit markers that have been WRITTEN but whose
//!   `complete_commit` has NOT yet fired (or whose paired DDL entry
//!   has not yet been marker-paired). `safe_visible = (min_pending - 1)`
//!   is the cap on `visible_commit_lsn` advertised via shm.
//!   Insertion/removal MUST happen under this lock; safe-visible
//!   computation MUST also happen under this lock to be coherent.
//! - `shm_publish_lock: Mutex<()>` — serializes the seqlock
//!   publish-pair (publish_seq odd → field stores → publish_seq
//!   even). Without it, two concurrent publishes could interleave
//!   their odd/even bumps and tear the seqlock from a reader's
//!   perspective.
//! - `max_written_marker_lsn: AtomicU64` — high-watermark of any
//!   commit marker EVER written. Bumped under `pending_marker_lsns`.
//!   Read in safe-visible computation.
//!
//! ## Lifecycle latches
//!
//! - `transactional_ddl_fence: parking_lot::RwLock<()>` — SH held by
//!   transactional DDL mutators (CREATE INDEX, ALTER TABLE, ...);
//!   EX taken by checkpoint's `rerecord_ddl_to_wal`. Prevents the
//!   checkpoint from re-recording catalog state mid-DDL.
//! - `seal_fence: parking_lot::RwLock<()>` — SH held by hot-buffer
//!   readers; EX by seal/compaction. Prevents reads from racing the
//!   hot→cold transition.
//! - `checkpoint_mutex: Mutex<()>` — serializes checkpoint cycles;
//!   held for the entire seal+compact+truncate sequence.
//! - `pending_drop_cleanups: Mutex<FxHashSet<String>>` — table dirs
//!   whose `manifest.bin` is still on disk after a DROP. Non-empty
//!   forces `compute_wal_truncate_floor` to refuse truncation.
//! - `orphan_discovery_failed: AtomicBool` — set when
//!   `sweep_orphan_table_dirs` could not enumerate `volumes/`.
//!   Same effect as a non-empty `pending_drop_cleanups`.
//! - `failed_restore_attach_gate: Mutex<Option<StartupLockGuard>>` —
//!   holds the EX startup lock when a PRAGMA RESTORE failed past
//!   the destructive boundary; keeps cross-process readers out
//!   until process restart.
//! - `failed: AtomicBool` — catastrophic-failure latch. Once set,
//!   refuses every new write; recovery requires process restart.
//!
//! ## Acquisition order (MUST be respected to avoid deadlock)
//!
//! ```text
//!   transactional_ddl_fence  →  schemas / version_stores
//!   seal_fence               →  segment_managers
//!   checkpoint_mutex         →  shm_publish_lock
//!   pending_marker_lsns      →  shm_publish_lock        (publish path)
//!   shm.lock()               →  shm_publish_lock        (DDL publish path)
//!   pending_marker_lsns      →  WAL append              (record_ddl serialize)
//!   DATABASE_REGISTRY        →  pending_marker_lsns     (close path)
//! ```
//!
//! `pending_marker_lsns` is NEVER taken while holding `schemas` /
//! `version_stores` (commit path acquires them in the opposite
//! order; commit then publishes and the publish-side does not need
//! catalog locks).
//!
//! ## Common patterns
//!
//! - **Commit publish** (`record_commit` → `publish_visible_commit_lsn`):
//!   1. Lock `pending_marker_lsns`.
//!   2. Write commit marker (gets LSN).
//!   3. Insert LSN into pending; bump `max_written_marker_lsn`.
//!   4. Drop `pending_marker_lsns`.
//!   5. ... `complete_commit` runs ...
//!   6. Lock `pending_marker_lsns`; remove our LSN; compute
//!      safe_visible.
//!   7. Drop pending; lock `shm_publish_lock`; seqlock-publish.
//!
//! - **DDL publish** (`record_ddl`): hold `pending_marker_lsns`
//!   across BOTH the DDL entry append AND the DDL commit-marker
//!   append. Other publishes block on the lock during this
//!   window — no concurrent commit can advertise a `visible_commit_lsn`
//!   that includes the DDL entry without ALSO including its
//!   marker. Release before calling
//!   `publish_visible_commit_lsn_local` (which re-acquires the
//!   same lock).
//!
//! - **Rollback publish** (`record_rollback`): clear the txn from
//!   the WAL active set, then watermark-only re-publish under
//!   `shm_publish_lock`. `visible_commit_lsn` does NOT change;
//!   only `oldest_active_txn_lsn` advances.
//!
//! - **Close**: take `DATABASE_REGISTRY` write lock first, check
//!   `Arc::strong_count`, then drop the registry lock and call
//!   `close_engine`. The strong-count check + registry removal +
//!   `close_engine` is NOT a single critical section; the
//!   `is_open()` checks at every transaction-begin path provide
//!   the soft `EngineNotOpen` contract for the residual race.

use crate::common::{CompactArc, I64Map, SmartString, StringMap};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use std::path::Path;

use crate::common::time_compat::Instant;

/// Returns lowercase version of string, avoiding allocation if already lowercase.
/// This is a hot-path optimization - most table names are already lowercase.
#[inline]
fn to_lowercase_cow(s: &str) -> Cow<'_, str> {
    // Fast path: check if all bytes are already lowercase ASCII
    // This avoids allocation for the common case
    if s.bytes().all(|b| !b.is_ascii_uppercase()) {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.to_lowercase())
    }
}

use super::file_lock::{FileLock, StartupLockGuard};

use crate::core::{DataType, Error, ForeignKeyConstraint, IsolationLevel, Result, Schema, Value};
use crate::storage::config::Config;
use crate::storage::mvcc::wal_manager::WALOperationType;
#[cfg(test)]
use crate::storage::mvcc::VisibilityChecker;
use crate::storage::mvcc::{
    DeferredDdlOp, DropSnapshot, MVCCTable, MvccTransaction, PersistenceManager, PkIndex,
    RowVersion, SealFenceGuard, TransactionEngineOperations, TransactionRegistry,
    TransactionVersionStore, VersionStore, INVALID_TRANSACTION_ID,
};
use crate::storage::traits::{
    Engine, Index, ReadEngine, ReadTable, ReadTransaction, WriteTable, WriteTransaction,
};

/// Type alias for a single table entry in the transaction version store
type TxnTableEntry = (SmartString, Arc<RwLock<TransactionVersionStore>>);

/// Type alias for the transaction version store map
/// Structured as txn_id -> [(table_name, store)] for efficient lookup per transaction
/// Uses SmallVec<[T; 2]> since most transactions access 1-2 tables, avoiding heap allocation
type TxnVersionStoreMap = I64Map<SmallVec<[TxnTableEntry; 2]>>;

/// Helper to get registry as the visibility checker type expected by VersionStore.
/// In production: returns concrete Arc<TransactionRegistry> (zero-cost, inlined).
/// In tests: returns Arc<dyn VisibilityChecker> for TestVisibilityChecker flexibility.
#[cfg(not(test))]
#[inline]
fn registry_as_visibility_checker(registry: &Arc<TransactionRegistry>) -> Arc<TransactionRegistry> {
    Arc::clone(registry)
}

#[cfg(test)]
#[inline]
fn registry_as_visibility_checker(
    registry: &Arc<TransactionRegistry>,
) -> Arc<dyn VisibilityChecker> {
    Arc::clone(registry) as Arc<dyn VisibilityChecker>
}

/// Remove FK constraints referencing `parent_table_lower` from all child schemas.
/// Updates both the schemas map and each affected VersionStore's schema.
/// Caller must hold schemas as write-locked and version_stores as read-locked.
fn strip_fk_references(
    schemas: &mut FxHashMap<String, CompactArc<Schema>>,
    version_stores: &FxHashMap<String, Arc<VersionStore>>,
    parent_table_lower: &str,
) {
    let children_to_update: Vec<String> = schemas
        .iter()
        .filter(|(_, schema)| {
            schema
                .foreign_keys
                .iter()
                .any(|fk| fk.referenced_table == parent_table_lower)
        })
        .map(|(name, _)| name.clone())
        .collect();

    for child_name in &children_to_update {
        if let Some(old_schema_arc) = schemas.get(child_name) {
            let old_schema: &Schema = old_schema_arc;
            let mut new_fks: Vec<ForeignKeyConstraint> = old_schema
                .foreign_keys
                .iter()
                .filter(|fk| fk.referenced_table != parent_table_lower)
                .cloned()
                .collect();
            if new_fks.len() < old_schema.foreign_keys.len() {
                let mut new_schema = Schema::with_timestamps_and_foreign_keys(
                    old_schema.table_name.clone(),
                    old_schema.columns.clone(),
                    Vec::new(),
                    old_schema.created_at,
                    old_schema.updated_at,
                );
                std::mem::swap(&mut new_schema.foreign_keys, &mut new_fks);
                let new_arc = CompactArc::new(new_schema);

                if let Some(vs) = version_stores.get(child_name.as_str()) {
                    *vs.schema_mut() = new_arc.clone();
                }

                schemas.insert(child_name.clone(), new_arc);
            }
        }
    }
}

/// Try to parse a default expression as a simple literal value.
/// Handles integers, floats, booleans, strings, and NULL without requiring the full SQL parser.
/// Returns None for complex expressions (function calls, etc.) that can't be precomputed.
fn try_parse_default_literal(expr: &str, data_type: DataType) -> Option<Value> {
    let expr = expr.trim();

    // NULL
    if expr.eq_ignore_ascii_case("null") {
        return Some(Value::null(data_type));
    }

    // Boolean
    if expr.eq_ignore_ascii_case("true") {
        return Some(Value::Boolean(true));
    }
    if expr.eq_ignore_ascii_case("false") {
        return Some(Value::Boolean(false));
    }

    // String literal (single-quoted) — coerce to target data type so that
    // e.g. TIMESTAMP DEFAULT '2024-01-01' is restored as a Timestamp, not Text.
    if expr.len() >= 2 && expr.starts_with('\'') && expr.ends_with('\'') {
        let inner = &expr[1..expr.len() - 1];
        // Unescape doubled single quotes
        let unescaped = inner.replace("''", "'");
        let text_val = Value::from(unescaped.as_str());
        return Some(text_val.coerce_to_type(data_type));
    }

    // Integer — coerce to target type (e.g. FLOAT DEFAULT 42 should be Float(42.0))
    if let Ok(v) = expr.parse::<i64>() {
        return Some(Value::Integer(v).coerce_to_type(data_type));
    }

    // Float — coerce to target type
    if let Ok(v) = expr.parse::<f64>() {
        return Some(Value::Float(v).coerce_to_type(data_type));
    }

    // Complex expressions (NOW(), CURRENT_TIMESTAMP, etc.) cannot be recovered
    // from old WAL/snapshot formats — they require default_value persistence.
    None
}

/// Register a virtual PkIndex on an INTEGER PRIMARY KEY column.
/// This allows the optimizer to find the PK index via `get_index_by_column()`.
fn register_pk_index(schema: &Schema, version_store: &Arc<VersionStore>) {
    if let Some(pk_idx) = schema.pk_column_index() {
        let pk_col = &schema.columns[pk_idx];
        let index_name = format!("__pk_{}_{}", schema.table_name_lower, pk_col.name);
        let pk_index = Arc::new(PkIndex::new(
            index_name.clone(),
            schema.table_name.clone(),
            pk_col.id as i32,
            pk_col.name.clone(),
        ));
        version_store.add_index(index_name, pk_index);
    }
}

// ============================================================================
// Binary Snapshot Metadata Functions
// ============================================================================
// Format: MAGIC(4) | VERSION(4) | LSN(8) | TIMESTAMP(8) | CRC32(4) = 28 bytes
// Magic bytes: 0x534E4150 ("SNAP" in ASCII)

/// Magic bytes for snapshot metadata ("SNAP" in ASCII)
const SNAPSHOT_META_MAGIC: u32 = 0x50414E53; // "SNAP" in little-endian

/// Current version of the snapshot metadata format
const SNAPSHOT_META_VERSION: u32 = 1;

/// Read binary snapshot metadata, returns the LSN or 0 if invalid/not found
fn read_snapshot_metadata(path: &std::path::Path) -> u64 {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    // Minimum size: 28 bytes
    if data.len() < 28 {
        return 0;
    }

    // Verify magic
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != SNAPSHOT_META_MAGIC {
        return 0;
    }

    // Verify version (must be compatible)
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version > SNAPSHOT_META_VERSION {
        eprintln!(
            "Warning: Snapshot metadata version {} is newer than supported {}",
            version, SNAPSHOT_META_VERSION
        );
        return 0;
    }

    // Verify CRC32
    let stored_crc = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let computed_crc = crc32fast::hash(&data[0..24]);
    if stored_crc != computed_crc {
        eprintln!("Warning: Snapshot metadata checksum mismatch");
        return 0;
    }

    // Extract LSN
    u64::from_le_bytes(data[8..16].try_into().unwrap())
}

/// Write binary snapshot metadata (atomic via temp file + rename)
fn write_snapshot_metadata(path: &std::path::Path, lsn: u64) -> Result<()> {
    use std::io::Write;

    let mut buf = Vec::with_capacity(28);

    // Magic (4 bytes)
    buf.extend_from_slice(&SNAPSHOT_META_MAGIC.to_le_bytes());

    // Version (4 bytes)
    buf.extend_from_slice(&SNAPSHOT_META_VERSION.to_le_bytes());

    // LSN (8 bytes)
    buf.extend_from_slice(&lsn.to_le_bytes());

    // Timestamp in milliseconds since epoch (8 bytes)
    let timestamp = crate::common::time_compat::SystemTime::now()
        .duration_since(crate::common::time_compat::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    buf.extend_from_slice(&timestamp.to_le_bytes());

    // CRC32 of the data portion (magic + version + lsn + timestamp)
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    // Write atomically using temp file and rename
    let temp_path = path.with_extension("bin.tmp");

    let mut file = std::fs::File::create(&temp_path).map_err(|e| {
        Error::internal(format!(
            "failed to create snapshot metadata temp file: {}",
            e
        ))
    })?;

    file.write_all(&buf)
        .map_err(|e| Error::internal(format!("failed to write snapshot metadata: {}", e)))?;

    file.sync_all()
        .map_err(|e| Error::internal(format!("failed to sync snapshot metadata: {}", e)))?;

    // Atomic rename
    std::fs::rename(&temp_path, path)
        .map_err(|e| Error::internal(format!("failed to rename snapshot metadata: {}", e)))?;

    // Sync directory to ensure rename is durable.
    // Windows does not support opening directories for fsync.
    #[cfg(not(windows))]
    if let Some(parent) = path.parent() {
        if let Ok(dir_file) = std::fs::File::open(parent) {
            let _ = dir_file.sync_all();
        }
    }

    Ok(())
}

/// Read snapshot LSN from either binary or JSON format (backward compatibility)
fn read_snapshot_lsn(snapshot_dir: &std::path::Path) -> u64 {
    // First try new binary format
    let bin_path = snapshot_dir.join("snapshot_meta.bin");
    if bin_path.exists() {
        let lsn = read_snapshot_metadata(&bin_path);
        if lsn > 0 {
            return lsn;
        }
    }

    // Fall back to old JSON format for backward compatibility
    let json_path = snapshot_dir.join("snapshot_meta.json");
    if json_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&json_path) {
            return content
                .trim()
                .strip_prefix("{\"lsn\":")
                .and_then(|s| s.strip_suffix("}"))
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
        }
    }

    0
}

/// View definition storing the query that defines the view
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// View name (lowercase for case-insensitive lookup)
    pub name: String,
    /// Original view name (preserves case)
    pub original_name: String,
    /// The SQL query string that defines the view
    pub query: String,
}

impl ViewDefinition {
    /// Create a new view definition
    pub fn new(name: &str, query: String) -> Self {
        Self {
            name: name.to_lowercase(),
            original_name: name.to_string(),
            query,
        }
    }

    /// Serialize view definition to binary format for WAL
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Original name (length-prefixed)
        buf.extend_from_slice(&(self.original_name.len() as u16).to_le_bytes());
        buf.extend_from_slice(self.original_name.as_bytes());

        // Query (length-prefixed, using u32 for longer queries)
        buf.extend_from_slice(&(self.query.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.query.as_bytes());

        buf
    }

    /// Deserialize view definition from binary format
    pub fn deserialize(data: &[u8]) -> crate::core::Result<Self> {
        let mut pos = 0;

        // Original name
        if pos + 2 > data.len() {
            return Err(crate::core::Error::internal(
                "invalid view: missing name length",
            ));
        }
        let name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + name_len > data.len() {
            return Err(crate::core::Error::internal("invalid view: missing name"));
        }
        let original_name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|e| crate::core::Error::internal(format!("invalid view name: {}", e)))?;
        pos += name_len;

        // Query
        if pos + 4 > data.len() {
            return Err(crate::core::Error::internal(
                "invalid view: missing query length",
            ));
        }
        let query_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + query_len > data.len() {
            return Err(crate::core::Error::internal("invalid view: missing query"));
        }
        let query = String::from_utf8(data[pos..pos + query_len].to_vec())
            .map_err(|e| crate::core::Error::internal(format!("invalid view query: {}", e)))?;

        Ok(Self::new(&original_name, query))
    }
}

/// Cached reverse FK mapping: (schema_epoch, parent_table → Arc<Vec<(child_table, constraint)>>)
/// Arc-wrapped so lookups are a ref-count bump (no Vec clone on every FK check).
type FkReverseCache = (u64, StringMap<Arc<Vec<(String, ForeignKeyConstraint)>>>);

/// MVCC Storage Engine
///
/// Provides multi-version concurrency control with snapshot isolation.
pub struct MVCCEngine {
    /// Database path (empty for in-memory)
    path: String,
    /// Configuration
    config: RwLock<Config>,
    /// Table schemas (Arc-wrapped for safe sharing with transactions)
    /// Each schema is also Arc-wrapped to avoid cloning on lookup (critical for PK fast path)
    schemas: Arc<RwLock<FxHashMap<String, CompactArc<Schema>>>>,
    /// Version stores for each table (Arc-wrapped for safe sharing with transactions)
    version_stores: Arc<RwLock<FxHashMap<String, Arc<VersionStore>>>>,
    /// Transaction registry
    registry: Arc<TransactionRegistry>,
    /// Whether the engine is open
    open: AtomicBool,
    /// Catastrophic-failure latch. Set when a write that's already
    /// drained local versions into parent VersionStores hits an
    /// unrecoverable WAL marker error. Once set, all paths that
    /// could make markerless rows durable or cross-process visible
    /// (`seal_hot_buffers`, `compact_volumes`, `create_backup_snapshot`,
    /// new commits) refuse to run. There is no real undo for parent
    /// VersionStore writes / index updates, so the only safe move
    /// is to stop persisting and force a process restart, after
    /// which WAL recovery converges to "txn discarded" (no commit
    /// marker). Wrapped in `Arc` so `EngineOperations` (the trait
    /// shim used by `MvccTransaction`) can latch it without holding
    /// a back-reference to the engine.
    failed: Arc<AtomicBool>,
    /// Cache of transaction version stores per (txn_id, table_name) for proper commit/rollback
    /// (Arc-wrapped for safe sharing with transactions)
    txn_version_stores: Arc<RwLock<TxnVersionStoreMap>>,
    /// View definitions (Arc for cheap cloning on lookup)
    views: RwLock<FxHashMap<String, Arc<ViewDefinition>>>,
    /// Persistence manager for WAL and snapshot operations (Arc-wrapped for safe sharing)
    persistence: Arc<Option<PersistenceManager>>,
    /// Flag to indicate we're loading from disk to avoid triggering redundant WAL writes
    /// (Arc-wrapped for safe sharing with transactions)
    loading_from_disk: Arc<AtomicBool>,
    /// File lock to prevent multiple processes from accessing the same database
    file_lock: Mutex<Option<FileLock>>,
    /// Schema epoch counter - increments on any CREATE/ALTER/DROP TABLE
    /// Used for fast cache invalidation without HashMap lookup.
    /// Wrapped in `Arc` so `EngineOperations` can hold a clone and
    /// bump the same counter from transactional DDL paths
    /// (`create_table` / `drop_table` / `restore_child_fk_schemas` /
    /// `rename_table` etc.); without that, a transactional CREATE
    /// adding an FK child wouldn't invalidate `fk_reverse_cache`
    /// and a later parent UPDATE / DELETE could skip FK
    /// enforcement against the new constraint.
    schema_epoch: Arc<AtomicU64>,
    /// Handle for the background cleanup thread (None if not started)
    cleanup_handle: Mutex<Option<CleanupHandle>>,
    /// Cached reverse FK mapping: parent_table → Vec<(child_table, FK constraint)>
    /// Rebuilt lazily on schema_epoch change. Zero cost for non-FK databases.
    fk_reverse_cache: RwLock<FkReverseCache>,
    /// Snapshot timestamps loaded per table — used to pair HNSW graph files with their
    /// matching data snapshots during WAL replay.
    snapshot_timestamps: RwLock<FxHashMap<String, String>>,
    /// Per-table segment managers (owns segments, delete vectors, manifest).
    /// Key: lowercase table name. Replaces frozen_volumes + volume_tombstones.
    segment_managers:
        Arc<RwLock<FxHashMap<String, Arc<crate::storage::volume::manifest::SegmentManager>>>>,
    /// When true, seal_hot_buffers bypasses thresholds and seals all rows.
    /// Set during close_engine to ensure all data is in volumes before shutdown.
    force_seal_all: AtomicBool,
    /// Prevents concurrent checkpoint cycles (background thread vs PRAGMA SNAPSHOT).
    /// Without this, two concurrent seal+compact runs can each read the same old
    /// segments, produce overlapping compacted volumes, and delete each other's data.
    checkpoint_mutex: Mutex<()>,
    /// Seal fence: commits acquire READ (shared, ~5ns), micro-seal acquires WRITE
    /// (exclusive, brief ~100ms) to create a quiet moment where all_hot_empty can
    /// be true. This enables WAL truncation under continuous writes.
    seal_fence: Arc<parking_lot::RwLock<()>>,
    /// Transactional-DDL fence: an open transaction that has
    /// performed at least one CREATE / DROP holds READ
    /// (shared) until commit / rollback. Checkpoint's
    /// `rerecord_ddl_to_wal` acquires WRITE (exclusive) so it
    /// snapshots `schemas` / `version_stores` only when no
    /// uncommitted DDL is in flight. Without this fence,
    /// checkpoint can publish an uncommitted CREATE as a
    /// durable DDL_TXN_ID auto-commit (which a later rollback
    /// can't retract), or omit a table from an uncommitted
    /// DROP that subsequently rolls back (leaving WAL with no
    /// CREATE for the still-live table after WAL truncation).
    transactional_ddl_fence: Arc<parking_lot::RwLock<()>>,
    /// Lowercased names of tables whose post-commit volume
    /// directory cleanup (`finalize_committed_drops`) failed
    /// — the `manifest.bin` / `vol_*.vol` files survived on
    /// disk despite the catalog reflecting a DROP. While
    /// non-empty, `compute_wal_truncate_floor` refuses every
    /// WAL truncation so the still-on-disk DropTable record
    /// is preserved for crash recovery. Cleared by
    /// `sweep_orphan_table_dirs` once the directory is
    /// confirmed gone (the existing pass-1 sweep retries the
    /// removal). Shared with `EngineOperations` so the
    /// transactional-DDL commit path participates.
    pending_drop_cleanups: Arc<parking_lot::Mutex<rustc_hash::FxHashSet<String>>>,
    /// Set when `sweep_orphan_table_dirs` could not enumerate
    /// `volumes/` (or one of its entries) and a non-NotFound
    /// error prevented us from re-seeding `pending_drop_cleanups`
    /// with on-disk orphan table dirs. While true,
    /// `compute_wal_truncate_floor` refuses WAL truncation: a
    /// transient enumeration error could otherwise let a
    /// checkpoint truncate past `DropTable` while the leftover
    /// `volumes/<table>/manifest.bin` is still on disk, so a
    /// later reopen would resurface the dropped table from the
    /// stale manifest with no `DropTable` record left to replay.
    /// Cleared once a discovery pass succeeds end-to-end.
    orphan_discovery_failed: Arc<AtomicBool>,
    /// Effective lease max-age in nanoseconds — same value
    /// `defer_for_live_readers` derives from config at every
    /// call. Cached here in an `Arc<AtomicU64>` so
    /// `EngineOperations` (transactional-DDL DROP path) can
    /// share it without re-reading config and without
    /// hard-coding a possibly-shorter window than the engine
    /// uses (a shorter EngineOperations window would reap
    /// leases the engine still considers live, letting a
    /// transactional DROP unlink volumes a reader is about
    /// to lazy-load).
    lease_max_age_nanos: Arc<AtomicU64>,
    /// Stash for the EXCLUSIVE `db.startup.lock` guard
    /// acquired at the top of a PRAGMA RESTORE. On any
    /// post-destructive-boundary failure the restore moves
    /// its local guard into this slot instead of letting
    /// it Drop, keeping the gate held for the remainder of
    /// the writer process. Cross-process readers attempting
    /// to attach via `await_writer_startup_quiescent`
    /// block on the SH side of `db.startup.lock` while
    /// this is `Some`, refusing to load the partially-
    /// destroyed on-disk state. The gate is only released
    /// when the engine is dropped (writer process exits) —
    /// the operator is then expected to restart and either
    /// retry the restore or roll forward from the existing
    /// volume state.
    failed_restore_attach_gate: Mutex<Option<StartupLockGuard>>,
    /// True while a background compaction thread is running.
    /// Background checkpoint skips compaction when set. Forced compaction
    /// (PRAGMA CHECKPOINT, close, restore) waits for it to finish first.
    compaction_running: Arc<AtomicBool>,
    /// Recorded persistence-init failure on a read-only open. Surfaced
    /// from `open_engine()` as a hard error instead of silently coming
    /// up with an empty in-memory engine. Always `None` for writable
    /// opens (those preserve the historical warn-and-degrade behaviour).
    persistence_init_error: Mutex<Option<Error>>,
    /// Cross-process shared header (`<db>/db.shm`). Initialized in
    /// `open_engine()` for writable, disk-backed opens on Unix; `None`
    /// otherwise. Writer publishes `visible_commit_lsn` here after
    /// commits whose WAL bytes have reached the WAL file so reader
    /// processes can detect new commits without polling files. See
    /// `mvcc::shm`.
    shm: Mutex<Option<Arc<crate::storage::mvcc::shm::ShmHandle>>>,
    /// WAL commit-marker LSNs that have been
    /// written but whose `complete_commit` hasn't fired yet.
    /// `publish_visible_commit_lsn` advances db.shm to the
    /// HIGHEST LSN such that EVERY marker `<= lsn` has fully
    /// completed — i.e. `min(pending) - 1` if non-empty, else
    /// `max_written_marker_lsn`.
    ///
    /// Without this, txn A could write its marker at LSN_A and
    /// pause before complete_commit; txn B (LSN_B > LSN_A) then
    /// completes and publishes LSN_B; cross-process readers
    /// scan WAL up to LSN_B and treat A's marker as committed
    /// even though A's registry state is still `Committing`.
    /// Tracking pending + max-written closes the visibility
    /// race.
    pending_marker_lsns: Arc<parking_lot::Mutex<std::collections::BTreeSet<u64>>>,
    /// Highest LSN ever returned by `record_commit` for any txn.
    /// Used as the upper bound of `safe_visible_commit_lsn` once
    /// `pending_marker_lsns` drains.
    max_written_marker_lsn: Arc<std::sync::atomic::AtomicU64>,
    /// User transactions whose commit marker has completed in the
    /// in-process registry but has not necessarily been published to
    /// `db.shm.visible_commit_lsn` yet. Keyed by marker LSN. When
    /// `SyncMode::None` buffers commit markers, publication can lag
    /// completion; we must keep their first-DML LSN in the WAL
    /// active map until a later flush lets shm publish past the
    /// marker.
    completed_marker_txns: Arc<parking_lot::Mutex<std::collections::BTreeMap<u64, i64>>>,
    /// Cap for read-only WAL replay. When `Some`,
    /// `replay_wal` skips entries with `lsn > cap`. Set by the
    /// caller (Database::open / open_read_only) BEFORE
    /// `open_engine` runs, from the same snapshot that becomes
    /// `EngineEntry::attach_visible_commit_lsn` — so replay and
    /// the SwmrPendingDdl filter agree on what was published at
    /// attach time. `None` means "uncapped" (writer recovery).
    replay_cap_lsn: std::sync::atomic::AtomicU64,
    /// Serializes EVERY shm visibility publish path (commit
    /// publish, DDL publish, post-recovery publish) so the
    /// seqlock's odd→stores→even sequence is atomic w.r.t. other
    /// writers. Without this, two concurrent commits can interleave
    /// their bump-odd / bump-even calls and a reader's
    /// `sample_visibility_pair` can observe an even seq while the
    /// fields are mid-mutation. Shared via Arc with EngineOperations
    /// so the auto-commit path (which only has access to a cloned
    /// `Arc<ShmHandle>`, not `self.shm.lock()`) goes through the
    /// same gate.
    shm_publish_lock: Arc<parking_lot::Mutex<()>>,
    /// Cached "any cross-process reader currently holds a live
    /// lease". Set by `refresh_lease_present_cache` (called from
    /// the cleanup loop on each tick). When `false`, the
    /// commit publish paths skip the `shm_publish_lock` +
    /// `publish_seq` seqlock dance and just `fetch_max`
    /// `visible_commit_lsn` so the truncate clamp keeps
    /// advancing. The Release-Acquire ordering on
    /// `visible_commit_lsn` is sufficient for the two AtomicU64
    /// shm fields without `publish_seq` — the seqlock is
    /// paranoia for future multi-word fields.
    ///
    /// Default `true` (conservative — pay the publish cost
    /// until a lease scan proves no reader is present).
    /// Transition `false → true` triggers a barrier publish
    /// (full seqlock) so any reader that arrived between scans
    /// sees coherent shm state on its first refresh.
    lease_present: Arc<AtomicBool>,
}

/// RAII guard that clears an AtomicBool on drop. Used to release the
/// compaction_running flag even on early returns or panics.
struct AtomicBoolGuard<'a>(&'a AtomicBool);
impl Drop for AtomicBoolGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Parse a volume ID from a `.vol` filename (e.g., `vol_00065f1a2b3c4d5e.vol` -> `0x00065f1a2b3c4d5e`).
fn parse_volume_id(path: &std::path::Path) -> Option<u64> {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("vol_"))
        .and_then(|name| name.strip_suffix(".vol"))
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
}

/// Scan `<db>/volumes/` for subdirectories that contain a `manifest.bin`
/// — i.e. tables that have been checkpointed to disk. Returns an empty
/// set if the volumes dir is missing or unreadable. Used by SWMR's
/// `reload_manifests` to detect tables created or dropped by another
/// process. The returned names are the directory names verbatim
/// (already lowercased by checkpoint convention via `table_name_lower`).
fn scan_table_dirs(db_path: &std::path::Path) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    let vol_dir = db_path.join("volumes");
    let entries = match std::fs::read_dir(&vol_dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }
        let manifest_path = entry.path().join("manifest.bin");
        if !manifest_path.exists() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            out.insert(name.to_lowercase());
        }
    }
    out
}

fn cap_visible_lsn_by_flushed(
    persistence: &Arc<Option<PersistenceManager>>,
    safe_visible: u64,
) -> u64 {
    if safe_visible == 0 {
        return 0;
    }
    persistence
        .as_ref()
        .as_ref()
        .and_then(|pm| pm.wal())
        .map(|wal| safe_visible.min(wal.flushed_lsn()))
        .unwrap_or(safe_visible)
}

fn clear_published_completed_txns(
    completed_marker_txns: &parking_lot::Mutex<std::collections::BTreeMap<u64, i64>>,
    persistence: &Arc<Option<PersistenceManager>>,
    published_lsn: u64,
) -> bool {
    if published_lsn == 0 {
        return false;
    }
    let mut txn_ids = Vec::new();
    {
        let mut completed = completed_marker_txns.lock();
        loop {
            let Some((&marker_lsn, &txn_id)) = completed.iter().next() else {
                break;
            };
            if marker_lsn > published_lsn {
                break;
            }
            completed.remove(&marker_lsn);
            txn_ids.push(txn_id);
        }
    }
    if txn_ids.is_empty() {
        return false;
    }
    if let Some(wal) = persistence.as_ref().as_ref().and_then(|pm| pm.wal()) {
        for txn_id in txn_ids {
            wal.clear_active_txn(txn_id);
        }
        true
    } else {
        false
    }
}

impl MVCCEngine {
    /// Creates a new MVCC engine with the given configuration.
    ///
    /// Persistence init failures are recorded on the engine and surfaced
    /// from `open_engine()` when the engine was opened read-only. For
    /// writable opens the historical "warn and fall back to in-memory"
    /// behaviour is preserved (changing it would silently break callers
    /// that rely on degraded operation when the data dir is temporarily
    /// unavailable). For read-only opens silent degradation is
    /// catastrophic — a `Connected to database` followed by
    /// `table not found` against an unintentionally-empty engine — so
    /// the failure is fatal.
    pub fn new(config: Config) -> Self {
        let path = config.path.clone().unwrap_or_default();
        let read_only = config.read_only;

        // Initialize persistence manager if path is provided and persistence is enabled
        let mut persistence_init_error: Option<Error> = None;
        let persistence = if !path.is_empty() && config.persistence.enabled {
            match PersistenceManager::new(Some(Path::new(&path)), &config.persistence, read_only) {
                Ok(pm) => Some(pm),
                Err(e) => {
                    if read_only {
                        // Hold the error; surface it from open_engine so
                        // the caller sees a Result<()> failure rather than
                        // a silently-empty engine.
                        persistence_init_error = Some(e);
                        None
                    } else {
                        eprintln!("Warning: Failed to initialize persistence: {}", e);
                        None
                    }
                }
            }
        } else {
            None
        };

        Self {
            path: if path.is_empty() {
                "memory://".to_string()
            } else {
                path
            },
            config: RwLock::new(config),
            schemas: Arc::new(RwLock::new(FxHashMap::default())),
            version_stores: Arc::new(RwLock::new(FxHashMap::default())),
            registry: Arc::new(TransactionRegistry::new()),
            open: AtomicBool::new(false),
            failed: Arc::new(AtomicBool::new(false)),
            txn_version_stores: Arc::new(RwLock::new(I64Map::new())),
            views: RwLock::new(FxHashMap::default()),
            persistence: Arc::new(persistence),
            loading_from_disk: Arc::new(AtomicBool::new(false)),
            file_lock: Mutex::new(None),
            schema_epoch: Arc::new(AtomicU64::new(0)),
            cleanup_handle: Mutex::new(None),
            fk_reverse_cache: RwLock::new((u64::MAX, StringMap::default())),
            snapshot_timestamps: RwLock::new(FxHashMap::default()),
            segment_managers: Arc::new(RwLock::new(FxHashMap::default())),
            force_seal_all: AtomicBool::new(false),
            checkpoint_mutex: Mutex::new(()),
            seal_fence: Arc::new(parking_lot::RwLock::new(())),
            transactional_ddl_fence: Arc::new(parking_lot::RwLock::new(())),
            lease_max_age_nanos: Arc::new(AtomicU64::new(0)),
            failed_restore_attach_gate: Mutex::new(None),
            pending_drop_cleanups: Arc::new(parking_lot::Mutex::new(
                rustc_hash::FxHashSet::default(),
            )),
            orphan_discovery_failed: Arc::new(AtomicBool::new(false)),
            compaction_running: Arc::new(AtomicBool::new(false)),
            persistence_init_error: Mutex::new(persistence_init_error),
            shm: Mutex::new(None),
            pending_marker_lsns: Arc::new(parking_lot::Mutex::new(
                std::collections::BTreeSet::new(),
            )),
            max_written_marker_lsn: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            completed_marker_txns: Arc::new(parking_lot::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            shm_publish_lock: Arc::new(parking_lot::Mutex::new(())),
            // Default true (conservative): pay the publish cost
            // until the cleanup loop's first lease scan proves
            // no reader is present.
            lease_present: Arc::new(AtomicBool::new(true)),
            // u64::MAX = uncapped (writer recovery). Database::open
            // overrides this for read-only file:// opens before
            // calling open_engine.
            replay_cap_lsn: std::sync::atomic::AtomicU64::new(u64::MAX),
        }
    }

    /// Creates a new in-memory MVCC engine
    pub fn in_memory() -> Self {
        Self::new(Config::default())
    }

    /// Opens the engine (inherent method)
    pub fn open_engine(&self) -> Result<()> {
        // Use atomic swap to check and set open flag atomically
        if self.open.swap(true, Ordering::AcqRel) {
            return Ok(()); // Already open
        }

        // Publish the effective lease window so any
        // `EngineOperations` instance built from this engine
        // (transactional DDL DROP path) shares the SAME
        // window the engine uses for `defer_for_live_readers`
        // / `min_pinned_reader_lsn`. Without this priming the
        // first transactional DROP could see a 0-nanos value
        // and fall back to the 120s floor while the engine's
        // own checks use a different window.
        let _ = self.effective_lease_max_age();

        // Surface a deferred persistence-init failure recorded by `new()`.
        // For read-only opens, persistence init failure is fatal: silent
        // fallback to an in-memory engine would let the caller see a
        // "successful" open against a completely empty engine and only
        // discover the missing data later via `table not found`.
        if let Some(err) = self.persistence_init_error.lock().unwrap().take() {
            // Reset the open flag since we're failing.
            self.open.store(false, Ordering::Release);
            return Err(err);
        }

        // Acquire file lock for disk-based databases to prevent concurrent access.
        // Read-only engines acquire a SHARED lock so multiple readers can coexist
        // (and so we don't contend with another writer that holds the exclusive
        // lock). Writers always take exclusive.
        //
        // Writer-side startup gate: BEFORE taking `db.lock` EX, the
        // writer takes `db.startup.lock` EX. It releases the gate
        // only after `mark_ready` has flipped `init_done`. While the
        // gate is held, any reader that finds `db.lock` already
        // EX-locked must block on the gate's SH side and therefore
        // cannot trust a stale `db.shm` READY left by the prior
        // writer incarnation. See `await_writer_startup_quiescent`
        // for the reader half of the protocol.
        //
        // The gate is bound to a local variable (`_startup_gate`)
        // and explicitly dropped right after `mark_ready` below.
        // Any early-return path (lock failure, recovery error, …)
        // releases the gate via Drop.
        #[cfg(unix)]
        let mut _startup_gate: Option<crate::storage::mvcc::file_lock::StartupLockGuard> = None;
        if self.path != "memory://" {
            let read_only = self.config.read().unwrap().read_only;
            #[cfg(unix)]
            if !read_only {
                _startup_gate =
                    FileLock::acquire_startup_exclusive(std::path::Path::new(&self.path))?;
            }
            let lock = if read_only {
                FileLock::acquire_shared(&self.path)?
            } else {
                FileLock::acquire(&self.path)?
            };
            let mut file_lock = self.file_lock.lock().unwrap();
            *file_lock = Some(lock);
            drop(file_lock);

            // SWMR v2: writable Unix opens publish a `db.shm` so reader
            // processes can observe `visible_commit_lsn` updates without
            // polling. We hold the exclusive `writer.lock` above, so the
            // shm create is single-writer-safe. Failure here is logged
            // but non-fatal: the engine still works locally; only
            // cross-process live SWMR degrades.
            #[cfg(unix)]
            if !read_only {
                use crate::storage::mvcc::shm::ShmHandle;
                // shm creation is REQUIRED for writable opens.
                // Read-only readers detect "writer up" by checking
                // for `<path>/db.shm`; if a writer were allowed to
                // run without it, those readers would silently fall
                // back to uncapped WAL replay and could observe
                // commit markers the writer hasn't yet published
                // through `visible_commit_lsn`. Failing here keeps
                // the v1/v2 split clean: shm exists ⇔ writer is
                // running with full SWMR coordination.
                let handle =
                    ShmHandle::create_writer(std::path::Path::new(&self.path)).map_err(|e| {
                        Error::internal(format!(
                            "failed to create db.shm at '{}': {} \
                             (shm is required for writable SWMR opens; \
                             a missing shm would let read-only attaches \
                             silently fall back to uncapped WAL replay)",
                            self.path, e
                        ))
                    })?;
                // writer_generation is bumped to `prior_gen + 1`
                // INSIDE `create_writer` (single store, no
                // restore-then-bump window). Any prior reader's
                // attach snapshot is now guaranteed unequal,
                // surfacing SwmrWriterReincarnated on their next
                // refresh.
                let mut shm = self.shm.lock().unwrap();
                let arc = Arc::new(handle);
                *shm = Some(Arc::clone(&arc));
                drop(shm);
                // Wire the WAL → shm mirror so every active-set
                // change in the WAL `fetch_min`-mirrors into
                // `db.shm.oldest_active_txn_lsn`. This is the
                // deterministic guarantee a fresh reader's
                // pre_acquire relies on during a
                // `lease_present == false` window: any txn
                // that has appended its first DML before the
                // reader samples shm has its first-DML LSN
                // reflected (≤) in shm.oldest, regardless of
                // whether commits used the seqlock fast path or
                // the cleanup loop has had a chance to publish
                // a barrier.
                if let Some(ref pm) = *self.persistence {
                    if let Some(wal) = pm.wal() {
                        wal.set_shm_oldest_mirror(arc);
                    }
                }
            }
        }

        // Start accepting transactions
        self.registry.start_accepting_transactions();

        // If persistence is enabled, start it and replay WAL for recovery
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                pm.start()?;

                // Mark that we're loading from disk to prevent WAL writes during recovery
                self.loading_from_disk.store(true, Ordering::Release);

                // Determine recovery mode:
                // 1. If snapshots/ exists with snapshot files -> legacy snapshot recovery
                // 2. If volumes/ exists with manifests -> new checkpoint-to-volume recovery
                // 3. Otherwise -> pure WAL replay from beginning
                let snapshot_dir = pm.path().join("snapshots");
                let vol_dir = pm.path().join("volumes");

                // Check for volumes/ with manifests (new architecture).
                // This takes priority over snapshots/ because PRAGMA SNAPSHOT
                // now creates backup .bin files in snapshots/ alongside volumes/.
                let has_volumes = vol_dir.exists()
                    && std::fs::read_dir(&vol_dir)
                        .ok()
                        .map(|entries| {
                            entries
                                .flatten()
                                .any(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
                        })
                        .unwrap_or(false);

                // Legacy snapshots: only used when volumes/ does NOT exist.
                // This handles migration from pre-volume databases.
                let has_legacy_snapshots = !has_volumes
                    && snapshot_dir.exists()
                    && std::fs::read_dir(&snapshot_dir)
                        .ok()
                        .map(|entries| {
                            entries
                                .flatten()
                                .any(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
                        })
                        .unwrap_or(false);

                let replay_from_lsn = if has_volumes {
                    // New path: load manifests + volumes BEFORE WAL replay.
                    // Volumes must be loaded so is_row_id_in_volume() can check
                    // row_ids for idempotent INSERT during replay.
                    let lsn = self.load_manifests_from_volumes();
                    self.load_standalone_volumes_no_schema_check();
                    lsn
                } else if has_legacy_snapshots {
                    // Legacy path: load snapshots (creates schemas + version stores)
                    let snapshot_lsn = self.load_snapshots()?;
                    self.load_standalone_volumes();
                    snapshot_lsn
                } else {
                    0
                };

                // Clean up stale .dv files from previous versions
                self.cleanup_stale_dv_files();

                // Replay WAL entries after the checkpoint/snapshot LSN
                self.replay_wal(replay_from_lsn)?;

                // Populate pre-computed default_value from default_expr for schema evolution.
                // Neither snapshots nor WAL persist the cached default_value, so old rows
                // read via normalize_row_to_schema would get NULL instead of the column default.
                self.populate_schema_defaults();

                // Recompute cold volume column mappings now that schemas have
                // default_value populated. Volumes loaded from disk had identity
                // mappings; after ALTER TABLE ADD COLUMN, the schema differs.
                {
                    let schemas = self.schemas.read().unwrap();
                    let mgrs = self.segment_managers.read().unwrap();
                    for (table_name, schema) in schemas.iter() {
                        if let Some(mgr) = mgrs.get(table_name.as_str()) {
                            mgr.invalidate_mappings(schema);
                        }
                    }
                }

                // Sync auto-increment counters from segment data so the next
                // generated row_id doesn't collide with cold rows.
                self.sync_auto_increment_from_segments();

                // Sync schema_epoch from segment metadata. WAL replay for
                // CreateTable / AddColumn etc. inserts directly into
                // self.schemas without going through the public DDL methods
                // (line ~1596), so schema_epoch never bumps during recovery.
                // Without this, a fresh engine that has loaded N tables
                // from disk reports schema_epoch=0 — and the v2 SWMR
                // schema-drift check (`seg.schema_version > engine.schema_epoch`)
                // would then false-fire on every legitimately-loaded
                // segment with schema_version >= 1. Set the counter to
                // the highest schema_version observed across all loaded
                // manifests so the drift check correctly fires only for
                // segments produced by writer DDL AFTER this engine
                // opened.
                self.sync_schema_epoch_from_segments();

                // Migration and post-recovery seal both write to disk
                // (sealed volumes, manifest updates, snapshots/ removal).
                // Read-only engines must not perform any of this — a
                // shared-lock reader cannot be allowed to mutate the
                // on-disk layout. The hot rows from WAL replay stay in
                // memory; queries will pay O(hot_size) until the next
                // writer checkpoints, which is the correct trade-off
                // for a reader.
                let read_only = self.is_read_only_mode();

                // Migration: if we loaded from legacy snapshots, seal all data
                // into volumes and remove the old snapshots/ directory.
                if has_legacy_snapshots && !read_only {
                    // Clear loading flag temporarily so checkpoint can write WAL
                    self.loading_from_disk.store(false, Ordering::Release);

                    // Force seal ALL hot rows into volumes
                    if let Err(e) = self.checkpoint_cycle_inner(true) {
                        eprintln!("Warning: snapshot-to-volume conversion failed: {}", e);
                    }
                    self.compact_after_checkpoint_forced();

                    // Only remove snapshots/ if this is a real v0.3.7 migration
                    // (no ddl-*.bin). If DDL metadata exists, these are user-created
                    // backup snapshots from PRAGMA SNAPSHOT that should be kept.
                    let snapshot_dir = pm.path().join("snapshots");
                    let is_user_backup =
                        std::fs::read_dir(&snapshot_dir)
                            .ok()
                            .is_some_and(|entries| {
                                entries.filter_map(|e| e.ok()).any(|e| {
                                    e.file_name().to_str().is_some_and(|n| {
                                        (n.starts_with("ddl-") && n.ends_with(".bin"))
                                            || n == "ddl.bin"
                                    })
                                })
                            });
                    if !is_user_backup {
                        if let Err(e) = std::fs::remove_dir_all(&snapshot_dir) {
                            eprintln!("Warning: failed to remove snapshots/: {}", e);
                        }
                    }

                    // Set loading flag back so the code below clears it
                    self.loading_from_disk.store(true, Ordering::Release);
                }

                // After recovery, seal hot rows immediately before accepting
                // queries. Without this, a dirty shutdown that loads 1M+ rows
                // into hot via WAL replay makes ALL queries O(hot_size) until
                // the first background checkpoint fires (up to 60s later).
                // Sealing here drains the hot buffer while no concurrent reads
                // exist, so there's no seal_overlap performance impact.
                //
                // Skipped on read-only engines: sealing writes new volume
                // files and persists manifests, which a shared-lock reader
                // is not permitted to do.
                if !read_only {
                    self.loading_from_disk.store(false, Ordering::Release);
                    let has_hot_rows = {
                        let stores = self.version_stores.read().unwrap();
                        stores.values().any(|s| s.committed_row_count() > 0)
                    };
                    if has_hot_rows {
                        self.force_seal_all.store(true, Ordering::Release);
                        if let Err(e) = self.seal_hot_buffers() {
                            eprintln!("Warning: post-recovery seal failed: {}", e);
                        }
                        self.force_seal_all.store(false, Ordering::Release);

                        // Persist manifests so sealed volumes survive another
                        // crash. Don't advance checkpoint_lsn or truncate WAL
                        // here — the WAL may be corrupt (that's why recovery
                        // ran). The first clean checkpoint (background thread
                        // or close_engine) will advance and truncate safely.
                        {
                            let mgr_arcs: Vec<_> = {
                                let mgrs = self.segment_managers.read().unwrap();
                                mgrs.values().cloned().collect()
                            };
                            for mgr in &mgr_arcs {
                                let _ = mgr.persist_manifest_only();
                            }
                        }
                    }
                    self.loading_from_disk.store(true, Ordering::Release);
                }

                // Clear the loading flag
                self.loading_from_disk.store(false, Ordering::Release);

                // Bump the manifest epoch unconditionally on writer
                // recovery. v1 mtime/epoch readers attaching now (and
                // any reader that already observed a pre-recovery
                // snapshot) need an epoch advance to trigger
                // `reload_manifests` on their next refresh. Without
                // this, a DDL-only crash recovery (CREATE TABLE with
                // no rows, no post-recovery seal) leaves the epoch
                // file untouched and readers stay stuck on the
                // pre-crash snapshot until an unrelated future commit
                // happens to checkpoint. Even when no recovery work
                // ran, an epoch bump is harmless (readers just
                // re-read the same manifests).
                if !read_only && !self.path.is_empty() {
                    if let Err(e) = crate::storage::mvcc::manifest_epoch::bump_epoch(
                        std::path::Path::new(&self.path),
                    ) {
                        eprintln!("Warning: post-recovery manifest epoch bump failed: {}", e);
                    }
                }
            }
        }

        // SWMR v2: publish the recovered WAL frontier into shm BEFORE
        // marking it ready. `replay_two_phase_capped` advances the
        // WAL manager's `current_lsn` to the highest applied marker;
        // the writable open's recovery is authoritative, so that LSN
        // IS the visibility frontier any read-only attach should cap
        // at. Without this store, a reader attaching post-recovery
        // would read `visible_commit_lsn = 0` (still the
        // create_writer initial), cap its own replay to 0, and miss
        // every recovered transaction (especially DDL-only crashes
        // where no post-recovery seal can resync the manifest epoch
        // for it).
        //
        // Then publish `init_done = MAGIC_READY` so reader attaches
        // via `ShmHandle::open_reader` succeed. Until then, readers
        // fall back to v1 mtime-only snapshot mode (independent of
        // the writer's recovery state).
        #[cfg(unix)]
        {
            let recovered_frontier = self
                .persistence
                .as_ref()
                .as_ref()
                .map(|pm| pm.current_lsn())
                .unwrap_or(0);
            let shm_guard = self.shm.lock().unwrap();
            if let Some(ref handle) = *shm_guard {
                if recovered_frontier > 0 {
                    // Serialize against any concurrent publish
                    // (none expected during recovery, but the
                    // contract is "every shm publish goes through
                    // the lock").
                    let _publish_guard = self.shm_publish_lock.lock();
                    // Seqlock publish: bump-odd, store BOTH
                    // fields, bump-even. We publish
                    // `oldest_active_txn_lsn = u64::MAX` ("no
                    // active user txns") under the same seqlock
                    // sequence so the reader's coherent sample
                    // reads (visible=recovered_frontier,
                    // oldest=u64::MAX). With the previous
                    // 0-default left in place, readers attaching
                    // before any later commit would seed
                    // `next_entry_floor = 0` and pin WAL at 1
                    // for the handle's lifetime (the no-op fast
                    // path `target <= last_applied` would never
                    // advance the pin).
                    handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                    handle
                        .header()
                        .oldest_active_txn_lsn
                        .store(u64::MAX, Ordering::Release);
                    handle
                        .header()
                        .visible_commit_lsn
                        .fetch_max(recovered_frontier, Ordering::AcqRel);
                    handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                }
                handle.mark_ready();
            }
        }

        // Release the startup gate. From this point on, readers
        // probing `db.shm` are allowed to classify a READY shm as
        // "live writer authoritative" — see the matching reader
        // half in `await_writer_startup_quiescent`. Held above the
        // mark_ready boundary so a reader that gets through the
        // gate and finds db.lock still EX-locked has a guarantee
        // that init_done has been freshly stamped by THIS writer.
        #[cfg(unix)]
        {
            let _ = _startup_gate.take();
        }

        // Note: Cleanup is started separately via start_cleanup() after Arc wrapping
        // because start_periodic_cleanup requires Arc<Self>

        // Sweep any leftover `<volumes>/<dirname>/`
        // directories from prior runs where the writer
        // dropped a table while readers were live (and the
        // defer-on-live-readers path skipped the unlink) or
        // crashed mid-DROP. Inside `open_engine` we hold the
        // file lock — at most one writer process can be
        // running, but live RO readers may exist; the sweep
        // itself gates on `defer_for_live_readers` and skips
        // when any are.
        self.sweep_orphan_table_dirs();

        Ok(())
    }

    /// Starts the background cleanup thread
    ///
    /// This should be called after wrapping the engine in Arc.
    /// The cleanup thread periodically removes:
    /// - Deleted rows older than the retention period
    /// - Old previous versions no longer needed
    /// - Old transaction metadata (in Snapshot Isolation mode only)
    pub fn start_cleanup(self: &Arc<Self>) {
        let config = self.config.read().unwrap();
        if !config.cleanup.enabled {
            return;
        }
        // `cleanup_interval = 0` is meaningless for an "interval
        // between runs" value, but DSN parsing accepts it. Treat it
        // as cleanup-disabled (same semantics as
        // `cleanup.enabled = false`) instead of letting the loop spin
        // with `loop_interval = 0`, which would skip the inner sleep
        // and hammer eviction (read-only) or the cleanup sweeps
        // (writer) in a tight loop. The read-only path is especially
        // bad because the tight loop also fetch_add's
        // `GLOBAL_EVICTION_EPOCH` on every iteration.
        if config.cleanup.interval_secs == 0 {
            return;
        }
        // Read-only engines previously skipped this thread entirely.
        // That left warm-tier eviction unrun: every loaded volume sat
        // in warm forever (no `idle_cycles` advance, no demotion to
        // cold), so a long-lived ReadOnlyDatabase grew to hold every
        // touched volume's decompressed columns in RAM. The cleanup
        // loop now runs for read-only too; the read-only branch in
        // `start_periodic_cleanup_internal` does eviction only and
        // skips every writer-mutation step (cleanup_old_*,
        // checkpoint_cycle_inner, spawn_compaction,
        // flush_wal_for_visibility, refresh_lease_present_cache).

        let interval = std::time::Duration::from_secs(config.cleanup.interval_secs);
        let deleted_row_retention =
            std::time::Duration::from_secs(config.cleanup.deleted_row_retention_secs);
        let txn_retention =
            std::time::Duration::from_secs(config.cleanup.transaction_retention_secs);
        drop(config);

        let handle =
            self.start_periodic_cleanup_internal(interval, deleted_row_retention, txn_retention);

        let mut cleanup_handle = self.cleanup_handle.lock().unwrap();
        *cleanup_handle = Some(handle);
    }

    /// Internal method to start periodic cleanup with configurable parameters
    #[cfg(not(target_arch = "wasm32"))]
    fn start_periodic_cleanup_internal(
        self: &Arc<Self>,
        interval: std::time::Duration,
        deleted_row_retention: std::time::Duration,
        txn_retention: std::time::Duration,
    ) -> CleanupHandle {
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = Arc::clone(&stop_flag);
        let engine = Arc::clone(self);
        // Snapshot read-only mode for the loop. Read-only is fixed
        // for the engine's lifetime, so we never need to re-check it
        // per tick. Writer-only operations (lease scans, WAL
        // visibility flush, checkpoint, compaction, retention
        // sweeps) are skipped on read-only; eviction still runs.
        let read_only = engine.is_read_only_mode();

        let handle = thread::spawn(move || {
            let mut time_since_cleanup = std::time::Duration::ZERO;

            while !stop_flag_clone.load(Ordering::Acquire) {
                // Read config once per outer iteration (not per 100ms tick).
                // Checkpoint cadence drives the writer's loop interval;
                // read-only handles use the configured cleanup interval as-is.
                let current_checkpoint_interval = if read_only {
                    std::time::Duration::ZERO
                } else {
                    let cfg = engine.config.read().unwrap();
                    if cfg.persistence.checkpoint_interval > 0 {
                        std::time::Duration::from_secs(cfg.persistence.checkpoint_interval as u64)
                    } else {
                        std::time::Duration::ZERO
                    }
                };
                let loop_interval = if !current_checkpoint_interval.is_zero() {
                    interval.min(current_checkpoint_interval)
                } else {
                    interval
                };

                let check_interval = std::time::Duration::from_millis(100);
                let mut elapsed = std::time::Duration::ZERO;
                while elapsed < loop_interval && !stop_flag_clone.load(Ordering::Acquire) {
                    thread::sleep(check_interval);
                    elapsed += check_interval;
                    if !read_only {
                        // Writer side: refresh lease-presence cache on
                        // every 100ms inner tick so a freshly-attached
                        // reader sees the writer's barrier publish
                        // within a bounded window. SyncMode::None no
                        // longer flushes every commit marker, so we
                        // periodically write the WAL buffer and
                        // republish the flushed frontier here too.
                        if engine.flush_wal_for_visibility_if_due() {
                            engine.barrier_publish_full_state();
                        }
                        engine.refresh_lease_present_cache();
                    }
                }

                if stop_flag_clone.load(Ordering::Acquire) {
                    break;
                }

                // Perform cleanup at the original cleanup interval
                time_since_cleanup += loop_interval;
                if time_since_cleanup >= interval {
                    time_since_cleanup = std::time::Duration::ZERO;
                    if !read_only {
                        let _txn_count = engine.cleanup_old_transactions(txn_retention);
                        let _row_count = engine.cleanup_deleted_rows(deleted_row_retention);
                        let _prev_version_count = engine.cleanup_old_previous_versions();
                    }
                }

                // Auto-checkpoint using the cached interval. Writer
                // only: read-only zeroed `current_checkpoint_interval`
                // above, so this whole block is skipped.
                if !current_checkpoint_interval.is_zero() {
                    if let Some(ref pm) = *engine.persistence {
                        let last = pm.last_checkpoint_time();
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos() as i64)
                            .unwrap_or(0);
                        let elapsed_nanos = now.saturating_sub(last);
                        let interval_nanos = current_checkpoint_interval.as_nanos() as i64;
                        if elapsed_nanos >= interval_nanos {
                            // Call checkpoint_cycle_inner directly (not
                            // checkpoint_cycle) so compaction is spawned on
                            // a separate thread instead of running synchronously.
                            match engine.checkpoint_cycle_inner(false) {
                                Ok(()) => {
                                    // Eviction runs inside spawn_compaction, AFTER
                                    // compaction finishes. Running them in the same
                                    // cycle but concurrently causes thrashing:
                                    // eviction frees volumes that compaction
                                    // immediately reloads via segments_snapshot.
                                    engine.spawn_compaction();
                                }
                                Err(e) => {
                                    eprintln!("Warning: checkpoint cycle failed: {}", e);
                                }
                            }
                        }
                    }
                } else if read_only {
                    // Read-only handles never run compaction, so the
                    // writer's "eviction runs inside spawn_compaction"
                    // hook never fires. Drive eviction directly here on
                    // the cleanup cadence so warm volumes age out to
                    // cold instead of pinning every loaded volume's
                    // decompressed columns in RAM forever. Eviction is
                    // pure RAM management (drops decompressed/
                    // compressed columns; metadata stays via Arc) and
                    // does not touch disk, so it is safe for read-only.
                    engine.evict_idle_volumes();
                }
            }
        });

        CleanupHandle {
            stop_flag,
            thread: Some(handle),
        }
    }

    /// No-op cleanup on WASM (no background threads available)
    #[cfg(target_arch = "wasm32")]
    fn start_periodic_cleanup_internal(
        self: &Arc<Self>,
        _interval: std::time::Duration,
        _deleted_row_retention: std::time::Duration,
        _txn_retention: std::time::Duration,
    ) -> CleanupHandle {
        CleanupHandle {
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Load table snapshots from disk for faster recovery
    ///
    /// Returns the LSN of the snapshot (or 0 if no snapshots found)
    fn load_snapshots(&self) -> Result<u64> {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(0),
        };

        let snapshot_dir = pm.path().join("snapshots");
        if !snapshot_dir.exists() {
            return Ok(0); // No snapshots directory
        }

        // Read the snapshot LSN from metadata (supports both binary and JSON formats)
        let metadata_lsn = read_snapshot_lsn(&snapshot_dir);

        // Track min source_lsn from snapshot headers for crash-safe replay.
        // We use MIN (not max) because after a crash during snapshot rename,
        // some tables may have new snapshots (high LSN) while others still have
        // old snapshots (low LSN). Using max would skip WAL entries needed by
        // tables with old snapshots.
        let mut min_header_lsn: u64 = u64::MAX;
        let mut any_snapshot_loaded = false;

        // Find and load table snapshots
        let table_dirs = match std::fs::read_dir(&snapshot_dir) {
            Ok(entries) => entries,
            Err(_) => return Ok(0),
        };

        for entry in table_dirs.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }

            let table_name = entry.file_name().to_string_lossy().to_string();

            // Find snapshot files newest-first and try each until one loads.
            // This provides fallback: if the latest snapshot is corrupted,
            // we can load the previous one and replay WAL from its LSN.
            let snapshot_paths = Self::find_snapshots_newest_first(&entry.path());
            for snapshot_path in &snapshot_paths {
                let vol_path = snapshot_path.with_extension("vol");
                let has_vol = vol_path.exists();

                // Use volume loading for large snapshots (>16MB) or if a .vol file exists
                let file_size = std::fs::metadata(snapshot_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let use_volume = has_vol || file_size > 16 * 1024 * 1024;

                let load_result = if use_volume {
                    self.load_table_snapshot_as_volume(&table_name, snapshot_path)
                } else {
                    self.load_table_snapshot(&table_name, snapshot_path)
                };
                match load_result {
                    Ok(source_lsn) => {
                        any_snapshot_loaded = true;
                        if source_lsn < min_header_lsn {
                            min_header_lsn = source_lsn;
                        }
                        // Extract timestamp from snapshot filename for HNSW graph pairing.
                        // Format: "snapshot-{timestamp}.bin" → extract "{timestamp}"
                        if let Some(fname) = snapshot_path.file_name().and_then(|n| n.to_str()) {
                            if let Some(ts) = fname
                                .strip_prefix("snapshot-")
                                .and_then(|s| s.strip_suffix(".bin"))
                            {
                                self.snapshot_timestamps
                                    .write()
                                    .unwrap()
                                    .insert(table_name.clone(), ts.to_string());
                            }
                        }
                        break; // Successfully loaded, stop trying older snapshots
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load snapshot for {}: {}", table_name, e);
                        // Try the next (older) snapshot
                    }
                }
            }
        }

        // If no snapshot loaded successfully, replay WAL from the beginning.
        // This handles the case where snapshot files are corrupted — we must NOT trust
        // metadata_lsn because those entries are not in any loaded snapshot.
        // Also remove checkpoint.meta, which was created alongside the snapshot —
        // if the snapshot is gone, the checkpoint is invalid and would cause
        // replay_two_phase to skip entries that need to be replayed.
        if !any_snapshot_loaded {
            let checkpoint_path = pm.path().join("wal").join("checkpoint.meta");
            let _ = std::fs::remove_file(checkpoint_path);
            return Ok(0);
        }

        // Use the SMALLER of metadata LSN and min header LSN for safety.
        // - Normal case: metadata_lsn == header_lsn (consistent)
        // - Crash during snapshot rename: metadata_lsn may be old (or 0),
        //   header_lsn reflects the oldest snapshot — use the smaller to ensure
        //   WAL replay covers all tables
        // - If metadata is missing (0), header_lsn provides fallback
        let snapshot_lsn = if metadata_lsn == 0 {
            min_header_lsn
        } else if min_header_lsn == u64::MAX {
            metadata_lsn
        } else {
            std::cmp::min(metadata_lsn, min_header_lsn)
        };

        Ok(snapshot_lsn)
    }

    /// Find snapshot files in a directory, sorted newest-first.
    /// Returns all snapshot-*.bin files so the caller can try fallbacks if the latest is corrupt.
    fn find_snapshots_newest_first(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut snapshots: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("snapshot-") && n.ends_with(".bin"))
                        .unwrap_or(false)
                })
                .collect(),
            Err(_) => return Vec::new(),
        };

        // Sort by name (timestamp in filename) then reverse for newest-first
        snapshots.sort();
        snapshots.reverse();

        snapshots
    }

    /// Load a single table's snapshot from disk
    /// Returns the source_lsn from the snapshot header (0 if v2 format or not available)
    fn load_table_snapshot(
        &self,
        _table_name: &str,
        snapshot_path: &std::path::Path,
    ) -> Result<u64> {
        let mut reader = super::snapshot::SnapshotReader::open(snapshot_path)?;

        // Get the source LSN from the snapshot header (v3+ format)
        let source_lsn = reader.source_lsn();

        // Get the schema from the snapshot
        let schema = reader.schema().clone();
        let table_name_lower = schema.table_name_lower.clone();

        // Create the version store
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            registry_as_visibility_checker(&self.registry),
        ));

        // Register virtual PkIndex for INTEGER PRIMARY KEY
        register_pk_index(&schema, &version_store);

        // Load all rows from the snapshot
        reader.for_each(|row_id, mut version| {
            // Snapshot versions have txn_id = -1, we need to use the recovery txn_id
            version.txn_id = super::RECOVERY_TRANSACTION_ID;

            // Apply to version store
            version_store.apply_recovered_version(row_id, version);
            true
        })?;

        // Store the schema and version store
        {
            let mut schemas = self.schemas.write().unwrap();
            schemas.insert(table_name_lower.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name_lower, version_store);
        }

        Ok(source_lsn)
    }

    /// Load a table's snapshot as a frozen volume instead of into the arena.
    ///
    /// Fast-startup path:
    /// 1. Check for a pre-built `.vol` file next to the snapshot — load directly
    /// 2. If no `.vol` file, convert from snapshot and save the `.vol` for next time
    ///
    /// The VersionStore is created empty (only WAL-replayed rows go into the arena).
    /// Returns the source_lsn from the snapshot header.
    fn load_table_snapshot_as_volume(
        &self,
        _table_name: &str,
        snapshot_path: &std::path::Path,
    ) -> Result<u64> {
        // Derive the volume file path from the snapshot path:
        // snapshot-20260314-001020.947.bin → snapshot-20260314-001020.947.vol
        let vol_path = snapshot_path.with_extension("vol");

        // Try loading a pre-built volume file first (instant startup)
        if vol_path.exists() {
            let volume = crate::storage::volume::io::read_volume_from_disk(&vol_path)?;
            let vol_row_count = volume.meta.row_count;

            // We still need the schema from the snapshot header
            let reader = super::snapshot::SnapshotReader::open(snapshot_path)?;
            let source_lsn = reader.source_lsn();
            let schema = reader.schema().clone();
            let table_name_lower = schema.table_name_lower.clone();

            // Create empty version store (hot buffer for WAL data only)
            let version_store = Arc::new(VersionStore::with_visibility_checker(
                schema.table_name.clone(),
                schema.clone(),
                registry_as_visibility_checker(&self.registry),
            ));
            register_pk_index(&schema, &version_store);

            // Store schema and version store
            {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower.clone(), CompactArc::new(schema));
            }
            {
                let mut stores = self.version_stores.write().unwrap();
                stores.insert(table_name_lower.clone(), version_store);
            }

            // Store the frozen volume in the segment manager.
            // Promote to standalone FIRST to get the stable volume_id, then
            // register with that same ID so load_standalone_volumes skips it.
            if vol_row_count > 0 {
                let volume = Arc::new(volume);
                let stable_id =
                    self.promote_snapshot_vol_to_standalone(&table_name_lower, &vol_path);
                if let Some(id) = stable_id {
                    self.register_volume_with_id(&table_name_lower, volume, id);
                } else {
                    self.register_volume(&table_name_lower, volume);
                }
            }

            return Ok(source_lsn);
        }

        // No pre-built volume — convert from snapshot
        let mut reader = super::snapshot::SnapshotReader::open(snapshot_path)?;
        let source_lsn = reader.source_lsn();
        let schema = reader.schema().clone();
        let table_name_lower = schema.table_name_lower.clone();

        // Create an empty version store (hot buffer for WAL data only)
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            registry_as_visibility_checker(&self.registry),
        ));
        register_pk_index(&schema, &version_store);

        // Build a frozen volume from the snapshot rows
        let mut builder = crate::storage::volume::writer::VolumeBuilder::new(&schema);
        reader.for_each(|row_id, version| {
            if !version.is_deleted() {
                builder.add_row(row_id, &version.data);
            }
            true
        })?;
        let mut volume = builder.finish();
        let vol_row_count = volume.meta.row_count;

        // Write volume file to disk and retain CompressedBlockStore for
        // warm-tier eviction (single compress, no double work).
        if vol_row_count > 0 {
            match crate::storage::volume::io::serialize_v4_public(&volume) {
                Ok((data, store)) => {
                    volume.columns.attach_compressed_store(store);
                    if let Err(e) = (|| -> Result<()> {
                        let tmp_path = vol_path.with_extension("vol.tmp");
                        std::fs::write(&tmp_path, &data).map_err(|e| {
                            crate::core::Error::internal(format!("failed to write volume: {}", e))
                        })?;
                        std::fs::File::open(&tmp_path)
                            .and_then(|f| f.sync_all())
                            .map_err(|e| {
                                crate::core::Error::internal(format!(
                                    "failed to sync volume: {}",
                                    e
                                ))
                            })?;
                        std::fs::rename(&tmp_path, &vol_path).map_err(|e| {
                            crate::core::Error::internal(format!("failed to rename volume: {}", e))
                        })?;
                        Ok(())
                    })() {
                        eprintln!(
                            "Warning: Failed to save volume file for {}: {}",
                            table_name_lower, e
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to serialize volume for {}: {}",
                        table_name_lower, e
                    );
                }
            }
        }

        let volume = Arc::new(volume);

        // Store schema and empty version store
        {
            let mut schemas = self.schemas.write().unwrap();
            schemas.insert(table_name_lower.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name_lower.clone(), version_store);
        }

        // Promote to standalone FIRST, then register with the stable ID
        if vol_row_count > 0 {
            let stable_id = self.promote_snapshot_vol_to_standalone(&table_name_lower, &vol_path);
            if let Some(id) = stable_id {
                self.register_volume_with_id(&table_name_lower, volume, id);
            } else {
                self.register_volume(&table_name_lower, volume);
            }
        }

        Ok(source_lsn)
    }

    /// Copy a snapshot-adjacent .vol file into the standalone volumes directory.
    ///
    /// This makes the cold data durable across snapshot rotations: once a newer
    /// snapshot is created and old snapshots are cleaned up, the .vol that was
    /// written next to the old snapshot would be orphaned. By copying it into
    /// `volumes/<table>/`, `load_standalone_volumes` can find it on every future
    /// restart regardless of which snapshot is current.
    ///
    /// The copy is skipped if the standalone directory already contains any .vol
    /// files for this table, to avoid duplicating data on every restart.
    /// Promotes a snapshot .vol to standalone volumes/. Returns the volume_id
    /// used for the standalone file so the caller can register the segment with
    /// the same ID (preventing double-load by load_standalone_volumes).
    fn promote_snapshot_vol_to_standalone(
        &self,
        table_name: &str,
        vol_path: &std::path::Path,
    ) -> Option<u64> {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return None,
        };

        if !vol_path.exists() {
            return None;
        }

        // Marker file stores the volume_id from the previous promotion.
        // On subsequent opens, return that ID so the caller registers with
        // the same segment_id as the standalone copy — preventing double-load.
        let marker_path = vol_path.with_extension("vol.promoted");
        if marker_path.exists() {
            return std::fs::read_to_string(&marker_path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
        }

        let vol_dir = pm.path().join("volumes");
        let table_vol_dir = vol_dir.join(table_name);

        if let Err(e) = std::fs::create_dir_all(&table_vol_dir) {
            eprintln!(
                "Warning: Failed to create standalone volume dir for {}: {}",
                table_name, e
            );
            return None;
        }

        let volume_id = crate::storage::volume::io::next_volume_id();
        let dest_name = format!("vol_{:016x}.vol", volume_id);
        let dest_path = table_vol_dir.join(&dest_name);
        let tmp_path = table_vol_dir.join(format!("{}.tmp", dest_name));

        // Use fs::copy for streaming kernel-level copy (no userspace memory spike)
        if let Err(e) = std::fs::copy(vol_path, &tmp_path) {
            eprintln!(
                "Warning: Failed to copy snapshot vol for {}: {}",
                table_name, e
            );
            return None;
        }
        // Ensure data is durable before rename, preventing zero-length files after crash.
        // Open with write access — Windows requires it for sync_all().
        if let Err(e) = std::fs::OpenOptions::new()
            .write(true)
            .open(&tmp_path)
            .and_then(|f| f.sync_all())
        {
            eprintln!(
                "Warning: Failed to sync standalone volume for {}: {}",
                table_name, e
            );
            let _ = std::fs::remove_file(&tmp_path);
            return None;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &dest_path) {
            eprintln!(
                "Warning: Failed to rename standalone volume for {}: {}",
                table_name, e
            );
            let _ = std::fs::remove_file(&tmp_path);
            return None;
        }

        // Write marker with the volume_id so subsequent opens register
        // with the same segment_id as the standalone copy.
        // Fsync the marker to prevent re-promotion (and duplicate volumes) after crash.
        if std::fs::write(&marker_path, volume_id.to_string()).is_ok() {
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .open(&marker_path)
                .and_then(|f| f.sync_all());
        }
        Some(volume_id)
    }

    /// Replay WAL entries to recover database state starting from a specific LSN
    ///
    /// Uses two-phase recovery to ensure crash consistency:
    /// - Phase 1: Scan WAL to identify committed/aborted transactions
    /// - Phase 2: Apply only entries from committed transactions
    ///
    /// This guarantees that after a crash, only committed transactions are visible.
    fn replay_wal(&self, from_lsn: u64) -> Result<()> {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()),
        };

        // For read-only opens, bound replay by the
        // writer's published `db.shm.visible_commit_lsn`. The
        // writer may have written a commit marker but not yet
        // completed/published it; without the cap, the read-only
        // attach would silently apply that not-yet-visible
        // transaction during open, leaving the reader ahead of the
        // writer's in-process visibility state. SWMR refresh's
        // `visible_commit_lsn`-cap then can't bring the reader
        // back into sync.
        //
        // Writable opens (writer recovery) DON'T cap by
        // visible_commit_lsn — that value is from a prior
        // incarnation and may be stale. The writer's recovery is
        // authoritative.
        // Use the pre-acquired replay_cap_lsn set by
        // Database::open BEFORE open_engine, NOT a fresh shm
        // snapshot. The pre-acquired value is the SAME snapshot
        // EngineEntry uses for `attach_visible_commit_lsn`, so
        // replay-cap and SwmrPendingDdl pre-attach filtering
        // agree. Taking a fresh snapshot here would let DDL
        // published between this point and EngineEntry's snapshot
        // be treated as pre-attach (suppressed) even though we
        // never replayed it.
        use std::sync::atomic::Ordering;
        let max_lsn = self.replay_cap_lsn.load(Ordering::Acquire);

        // Use two-phase recovery for crash consistency.
        // This ensures uncommitted transactions are NOT applied after a crash.
        // For read-only opens, also caps at writer's safe-visible LSN.
        let result =
            pm.replay_two_phase_capped(from_lsn, max_lsn, |entry| self.apply_wal_entry(entry));

        match result {
            Ok(info) => {
                if info.skipped_entries > 0 {
                    eprintln!(
                        "Recovery: {} entries skipped (from aborted/uncommitted transactions)",
                        info.skipped_entries
                    );
                }

                // After WAL replay completes, populate all indexes in a single pass
                // This is O(N + M) instead of O(N * M) when populating each index separately
                self.populate_all_indexes();

                // HNSW indexes need cold segment data too — vector similarity search
                // cannot fall back to zone maps like other index types.
                self.populate_hnsw_from_segments();

                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Populate all indexes across all version stores in a single pass per table
    fn populate_all_indexes(&self) {
        let stores = self.version_stores.read().unwrap();
        for store in stores.values() {
            store.populate_all_indexes();
        }
    }

    /// Populate HNSW indexes from cold segment data.
    ///
    /// After WAL replay + populate_all_indexes(), HNSW indexes only contain hot rows.
    /// Cold segment rows must also be added because vector similarity search cannot
    /// fall back to zone maps like B-tree/Hash indexes can.
    ///
    /// Uses newest-first volume ordering with row_id dedup so that when the same
    /// row_id exists in multiple overlapping volumes, only the newest version is
    /// added to the HNSW graph. Also skips row_ids that are in the hot buffer
    /// (already indexed by populate_all_indexes).
    fn populate_hnsw_from_segments(&self) {
        let stores = self.version_stores.read().unwrap();
        let mgrs = self.segment_managers.read().unwrap();

        for (table_name, store) in stores.iter() {
            if let Some(mgr) = mgrs.get(table_name) {
                if !mgr.has_segments() {
                    continue;
                }

                // Collect HNSW index info before iterating volumes
                let indexes = store.get_all_indexes();
                let hnsw_infos: Vec<(Vec<usize>, std::sync::Arc<dyn Index>)> = indexes
                    .iter()
                    .filter(|idx| idx.index_type() == crate::core::IndexType::Hnsw)
                    .filter_map(|idx| {
                        let col_ids = idx.column_ids();
                        if col_ids.is_empty() {
                            return None;
                        }
                        let col_indices: Vec<usize> =
                            col_ids.iter().map(|&id| id as usize).collect();
                        Some((col_indices, std::sync::Arc::clone(idx)))
                    })
                    .collect();

                if hnsw_infos.is_empty() {
                    continue;
                }

                // Atomic capture of cold state. Newest-first ordering
                // makes overlapping row_ids resolve to the newest
                // version. Separate calls would race a concurrent
                // read-only refresh's `reload_from_disk`.
                let (volumes, tombstones) = mgr.volumes_and_tombstones_newest_first();

                // Seed seen set with hot row_ids (already indexed by populate_all_indexes)
                let mut seen: rustc_hash::FxHashSet<i64> = store
                    .get_all_visible_row_ids(INVALID_TRANSACTION_ID + 1)
                    .into_iter()
                    .collect();

                // Stream directly from volumes into per-index batches.
                // Pre-allocate a reusable buffer to avoid per-row Vec allocations.
                let max_cols = hnsw_infos
                    .iter()
                    .map(|(cols, _)| cols.len())
                    .max()
                    .unwrap_or(0);
                let mut batches: Vec<Vec<(i64, Vec<crate::core::Value>)>> =
                    (0..hnsw_infos.len()).map(|_| Vec::new()).collect();
                let mut values_buf: Vec<crate::core::Value> = Vec::with_capacity(max_cols);

                const HNSW_FLUSH_THRESHOLD: usize = 8192;

                for (_, cs) in volumes.iter() {
                    let vol = &cs.volume;
                    for i in 0..vol.meta.row_count {
                        let row_id = vol.meta.row_ids[i];
                        if tombstones.contains_key(&row_id) || !seen.insert(row_id) {
                            continue;
                        }
                        for (batch_idx, (col_indices, _)) in hnsw_infos.iter().enumerate() {
                            values_buf.clear();
                            let mut has_null = false;
                            for &ci in col_indices {
                                let v = if ci < vol.columns.len() {
                                    vol.columns[ci].get_value(i)
                                } else {
                                    crate::core::Value::Null(crate::core::DataType::Null)
                                };
                                if v.is_null() {
                                    has_null = true;
                                    break;
                                }
                                values_buf.push(v);
                            }
                            if !has_null {
                                // Move ownership instead of cloning to avoid per-row allocation
                                let owned = std::mem::replace(
                                    &mut values_buf,
                                    Vec::with_capacity(max_cols),
                                );
                                batches[batch_idx].push((row_id, owned));
                            }
                        }
                    }

                    // Flush large batches to limit peak memory
                    for (idx, (_, index)) in hnsw_infos.iter().enumerate() {
                        if batches[idx].len() >= HNSW_FLUSH_THRESHOLD {
                            let entry_refs: Vec<(i64, &[crate::core::Value])> = batches[idx]
                                .iter()
                                .map(|(row_id, values)| (*row_id, values.as_slice()))
                                .collect();
                            if let Err(e) = index.add_batch_slice(&entry_refs) {
                                eprintln!(
                                    "Warning: HNSW index population failed for {}: {}",
                                    table_name, e
                                );
                            }
                            batches[idx].clear();
                        }
                    }
                }

                // Flush remaining entries
                for (idx, (_, index)) in hnsw_infos.iter().enumerate() {
                    if !batches[idx].is_empty() {
                        let entry_refs: Vec<(i64, &[crate::core::Value])> = batches[idx]
                            .iter()
                            .map(|(row_id, values)| (*row_id, values.as_slice()))
                            .collect();
                        if let Err(e) = index.add_batch_slice(&entry_refs) {
                            eprintln!(
                                "Warning: HNSW index population failed for {}: {}",
                                table_name, e
                            );
                        }
                    }
                }
            }
        }
    }

    /// Populate pre-computed `default_value` from `default_expr` on all schema columns.
    ///
    /// Neither snapshot nor WAL serialization persists `default_value` (only the expression
    /// string `default_expr` is stored). After recovery, `normalize_row_to_schema` uses
    /// `default_value` for schema-evolved rows (ALTER TABLE ADD COLUMN). Without this,
    /// old rows would get NULL instead of the configured DEFAULT.
    fn populate_schema_defaults(&self) {
        let stores = self.version_stores.read().unwrap();
        for store in stores.values() {
            let mut schema_guard = store.schema_mut();
            let needs_update = schema_guard
                .columns
                .iter()
                .any(|col| col.default_value.is_none() && col.default_expr.is_some());
            if !needs_update {
                continue;
            }
            let schema = CompactArc::make_mut(&mut *schema_guard);
            for col in &mut schema.columns {
                if col.default_value.is_none() {
                    if let Some(ref expr) = col.default_expr {
                        // Backward-compat fallback for WAL/snapshots written before
                        // default_value persistence was added. Only handles simple
                        // literals (integers, floats, booleans, strings, NULL).
                        // Non-deterministic expressions (NOW(), CURRENT_TIMESTAMP)
                        // cannot be recovered here — they require the new format.
                        col.default_value = try_parse_default_literal(expr, col.data_type);
                    }
                }
            }
        }
        drop(stores);

        // Sync engine schema cache from version stores
        let stores = self.version_stores.read().unwrap();
        let mut schemas = self.schemas.write().unwrap();
        for (name, store) in stores.iter() {
            schemas.insert(name.clone(), store.schema().clone());
        }
    }

    /// Apply a single WAL entry during recovery
    fn apply_wal_entry(&self, entry: crate::storage::mvcc::wal_manager::WALEntry) -> Result<()> {
        use crate::storage::mvcc::persistence::{deserialize_row_version, IndexMetadata};
        use crate::storage::mvcc::wal_manager::WALOperationType;

        match entry.operation {
            WALOperationType::CreateTable => {
                // Deserialize schema from entry data
                if let Ok(schema) = self.deserialize_schema(&entry.data) {
                    // Create the table (version store)
                    let version_store = Arc::new(VersionStore::with_visibility_checker(
                        schema.table_name.clone(),
                        schema.clone(),
                        registry_as_visibility_checker(&self.registry),
                    ));

                    // Register virtual PkIndex for INTEGER PRIMARY KEY
                    register_pk_index(&schema, &version_store);

                    let table_name = schema.table_name_lower.clone();

                    {
                        let mut schemas = self.schemas.write().unwrap();
                        schemas.insert(table_name.clone(), CompactArc::new(schema));
                    }
                    {
                        let mut stores = self.version_stores.write().unwrap();
                        stores.insert(table_name, version_store);
                    }
                }
            }
            WALOperationType::DropTable => {
                let table_name = entry.table_name.to_lowercase();

                // Remove schema, strip FK references from child tables, and remove version store.
                // Must strip FK references before removing schema, same as drop_table_internal.
                {
                    let mut schemas = self.schemas.write().unwrap();
                    schemas.remove(&table_name);
                    let stores = self.version_stores.read().unwrap();
                    strip_fk_references(&mut schemas, &stores, &table_name);
                }
                {
                    let mut stores = self.version_stores.write().unwrap();
                    if let Some(store) = stores.remove(&table_name) {
                        store.close();
                    }
                }
                // Clear segments for dropped table
                {
                    let mut mgrs = self.segment_managers.write().unwrap();
                    if let Some(mgr) = mgrs.get(&table_name) {
                        mgr.clear();
                    }
                    mgrs.remove(&table_name);
                }
            }
            WALOperationType::CreateIndex => {
                // Deserialize index metadata
                // Use skip_population=true for deferred single-pass population
                if let Ok(index_meta) = IndexMetadata::deserialize(&entry.data) {
                    let table_name = entry.table_name.to_lowercase();
                    if let Ok(store) = self.get_version_store(&table_name) {
                        // For HNSW indexes, try loading saved graph from snapshot dir.
                        // Use the timestamped graph file matching the loaded data snapshot
                        // to avoid row_id mismatches on fallback recovery.
                        let hnsw_graph_path = if index_meta.index_type
                            == crate::core::IndexType::Hnsw
                        {
                            if let Some(ref pm) = *self.persistence {
                                let dir = pm.path().join("snapshots").join(&table_name);
                                if dir.exists() {
                                    // Look up the timestamp of the loaded snapshot
                                    let ts = self
                                        .snapshot_timestamps
                                        .read()
                                        .unwrap()
                                        .get(&table_name)
                                        .cloned();
                                    if let Some(ts) = ts {
                                        // Try timestamped file first (new format)
                                        let timestamped = dir
                                            .join(format!("hnsw_{}-{}.bin", index_meta.name, ts));
                                        if timestamped.exists() {
                                            Some(timestamped)
                                        } else {
                                            // Backwards compat: try old non-timestamped format
                                            let old =
                                                dir.join(format!("hnsw_{}.bin", index_meta.name));
                                            if old.exists() {
                                                Some(old)
                                            } else {
                                                None
                                            }
                                        }
                                    } else {
                                        // No snapshot loaded (pure WAL replay), try old format
                                        let old = dir.join(format!("hnsw_{}.bin", index_meta.name));
                                        if old.exists() {
                                            Some(old)
                                        } else {
                                            None
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let _ = store.create_index_from_metadata_with_graph(
                            &index_meta,
                            true,
                            hnsw_graph_path.as_deref(),
                        );
                    }
                }
            }
            WALOperationType::DropIndex => {
                // Index name is stored in entry.data as simple string
                if let Ok(index_name) = String::from_utf8(entry.data.clone()) {
                    let table_name = entry.table_name.to_lowercase();
                    if let Ok(store) = self.get_version_store(&table_name) {
                        let _ = store.drop_index(&index_name);
                    }
                }
            }
            WALOperationType::Insert | WALOperationType::Update => {
                // Deserialize row version and apply to version store.
                if let Ok(row_version) = deserialize_row_version(&entry.data) {
                    let table_name = entry.table_name.to_lowercase();
                    let mgr = self.get_or_create_segment_manager(&table_name);
                    let in_volume = mgr.is_row_id_in_volume(entry.row_id);

                    if in_volume {
                        // Row exists in a cold volume. Two cases:
                        // 1. Sealed INSERT: original data, already in volume → skip
                        // 2. Post-seal UPDATE: new data supersedes cold → apply + tombstone
                        //
                        // Distinguish by checking if the row_id is already tombstoned.
                        // If tombstoned, a previous WAL entry already marked it as
                        // superseded (UPDATE or DELETE), so this entry is a later
                        // version that should be applied. If not tombstoned, this is
                        // the first (original sealed) INSERT → skip.
                        //
                        // ORDERING INVARIANT: commit_all_tables writes tombstone
                        // DELETE entries BEFORE the corresponding INSERT entries.
                        // This guarantees `already_tombstoned` is true for post-seal
                        // UPDATEs (which are recorded as Insert in the WAL).
                        // The `WALOperationType::Update` arm is a safety net for
                        // any future code path that records with the Update op type.
                        let already_tombstoned = mgr.is_tombstoned(entry.row_id);

                        if already_tombstoned || entry.operation == WALOperationType::Update {
                            // Post-seal change: apply to hot. The hot version
                            // shadows the cold version via skip set (hot_row_ids
                            // in the cumulative skip set at scan time). No tombstone
                            // needed here — the hot version IS the dedup mechanism.
                            if let Ok(store) = self.get_version_store(&table_name) {
                                store.apply_recovered_version(entry.row_id, row_version);
                            }
                        }
                        // else: sealed INSERT, volume has authoritative data → skip
                    } else {
                        // Row not in any volume: standard hot insert
                        if let Ok(store) = self.get_version_store(&table_name) {
                            store.apply_recovered_version(entry.row_id, row_version);
                        }
                    }
                }
            }
            WALOperationType::Delete => {
                // For deletes, mark the row as deleted in the hot store.
                let table_name = entry.table_name.to_lowercase();
                if let Ok(store) = self.get_version_store(&table_name) {
                    store.mark_deleted(entry.row_id, entry.txn_id);
                }
                // If the deleted row_id lives in a cold segment, add a tombstone
                // so it is excluded from scans and point lookups.
                let mgr = self.get_or_create_segment_manager(&table_name);
                if mgr.is_row_id_in_volume(entry.row_id) {
                    // Recovery tombstones get commit_seq=0, which is always visible
                    // to all new snapshots (any begin_seq > 0). This is correct:
                    // these tombstones were committed before the restart.
                    // visible_at_lsn=0 marks "always visible" cross-process —
                    // recovery-rebuilt tombstones predate any current
                    // capped attach.
                    mgr.add_tombstones(&[entry.row_id], 0, 0);
                }
            }
            WALOperationType::Commit => {
                // Mark transaction as committed in registry for visibility
                // Use the LSN as the commit sequence number
                self.registry
                    .recover_committed_transaction(entry.txn_id, entry.lsn as i64);
            }
            WALOperationType::Rollback => {
                // Rolled back transactions don't need to be marked as committed
                // Their changes should not be visible
            }
            WALOperationType::AlterTable => {
                // Schema modification - replay the ALTER TABLE operation
                if let Err(e) = self.replay_alter_table(&entry.data) {
                    eprintln!("Warning: Failed to replay ALTER TABLE: {}", e);
                }
            }
            WALOperationType::CreateView => {
                // Deserialize view definition and recreate the view
                if let Ok(view_def) = ViewDefinition::deserialize(&entry.data) {
                    let name_lower = view_def.name.clone();
                    let mut views = self.views.write().unwrap();
                    views.insert(name_lower, Arc::new(view_def));
                }
            }
            WALOperationType::DropView => {
                // Remove the view
                if let Ok(view_name) = String::from_utf8(entry.data.clone()) {
                    let name_lower = view_name.to_lowercase();
                    let mut views = self.views.write().unwrap();
                    views.remove(&name_lower);
                }
            }
            WALOperationType::TruncateTable => {
                // Clear all data from the table
                // During WAL recovery there are no concurrent transactions,
                // so truncate_all() will always succeed.
                let table_name = entry.table_name.to_lowercase();
                if let Ok(store) = self.get_version_store(&table_name) {
                    let _ = store.truncate_all();
                }
                // Clear segments (truncate removes everything)
                {
                    let mgrs = self.segment_managers.read().unwrap();
                    if let Some(mgr) = mgrs.get(&table_name) {
                        mgr.clear();
                    }
                }
                // Delete standalone volume files. Same defer-
                // when-readers-live treatment as the live
                // TRUNCATE / DROP paths: a read-only process
                // can outlive a writer crash and start
                // lazy-loading these volumes via
                // `SegmentManager::ensure_volume` while the
                // newly-opened writer replays this WAL entry.
                // Unconditionally unlinking would silently
                // drop rows from the reader's view (the lazy
                // load returns None and the scanner / point
                // lookup treats it as "row not in cold").
                // The leftover directory is reaped by
                // `sweep_orphan_table_dirs` on the next
                // checkpoint or open once readers detach.
                if let Some(ref pm) = *self.persistence {
                    if pm.is_enabled() {
                        let vol_dir = pm.path().join("volumes");
                        let defer = self.defer_for_live_readers();
                        // Propagate the failure: leaving the
                        // dropped table's manifest behind would
                        // let a checkpoint re-record only the
                        // live tables and truncate WAL past the
                        // `TruncateTable` entry, breaking the
                        // recovery contract on the next open.
                        // The replay caller surfaces the error
                        // and aborts recovery instead of
                        // proceeding with a half-cleaned state.
                        crate::storage::volume::io::delete_table_volumes_when_safe(
                            &vol_dir,
                            &table_name,
                            defer,
                        )?;
                    }
                }
            }
        }

        // Advance schema_epoch on every DDL replay. Without this,
        // a CREATE TABLE replayed from WAL on a reader (or from
        // the SWMR overlay's WAL-tail) would leave schema_epoch
        // at 0; a subsequent writer checkpoint that produces the
        // table's first manifest with schema_version >= 1 would
        // then fail `peek_schema_drift` (segment.schema_version >
        // reader.schema_epoch) and surface SchemaChanged on the
        // very first refresh after CREATE TABLE.
        // `sync_schema_epoch_from_segments` (called once at
        // engine open) cannot help here: at the time CREATE TABLE
        // is replayed there are no segments to read a version
        // from. The DDL itself is the schema-change signal.
        if entry.operation.is_ddl() {
            self.schema_epoch
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }

        Ok(())
    }

    /// Deserialize a schema from binary format (WAL format)
    fn deserialize_schema(&self, data: &[u8]) -> Result<Schema> {
        use crate::core::SchemaColumn;

        if data.len() < 4 {
            return Err(Error::internal("schema data too short"));
        }

        let mut pos = 0;

        // Read table name length
        if pos + 2 > data.len() {
            return Err(Error::internal("invalid schema: missing table name length"));
        }
        let name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + name_len > data.len() {
            return Err(Error::internal("invalid schema: missing table name"));
        }
        let table_name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid table name: {}", e)))?;
        pos += name_len;

        // Read column count
        if pos + 2 > data.len() {
            return Err(Error::internal("invalid schema: missing column count"));
        }
        let column_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        // Read columns
        let mut columns = Vec::with_capacity(column_count);
        for i in 0..column_count {
            // Column name length
            if pos + 2 > data.len() {
                return Err(Error::internal(
                    "invalid schema: missing column name length",
                ));
            }
            let col_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;

            if pos + col_name_len > data.len() {
                return Err(Error::internal("invalid schema: missing column name"));
            }
            let col_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
                .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;
            pos += col_name_len;

            // Data type (1 byte, + 2 bytes dimension for Vector)
            if pos >= data.len() {
                return Err(Error::internal("invalid schema: missing data type"));
            }
            let dt_tag = data[pos];
            pos += 1;
            let data_type = DataType::from_u8(dt_tag).unwrap_or(DataType::Null);
            let mut vector_dimensions: u16 = 0;
            if dt_tag == 7 {
                // Vector type: read 2 bytes for dimension
                if pos + 2 <= data.len() {
                    vector_dimensions = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                    pos += 2;
                }
            }

            // Nullable (1 byte)
            if pos >= data.len() {
                return Err(Error::internal("invalid schema: missing nullable flag"));
            }
            let nullable = data[pos] != 0;
            pos += 1;

            // Primary key (1 byte)
            if pos >= data.len() {
                return Err(Error::internal("invalid schema: missing primary key flag"));
            }
            let primary_key = data[pos] != 0;
            pos += 1;

            // Auto-increment (1 byte) - optional for backwards compatibility
            let auto_increment = if pos < data.len() {
                let val = data[pos] != 0;
                pos += 1;
                val
            } else {
                false
            };

            // Default expression (length-prefixed string) - optional for backwards compatibility
            let default_expr = if pos + 2 <= data.len() {
                let expr_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if expr_len > 0 && pos + expr_len <= data.len() {
                    let expr = String::from_utf8(data[pos..pos + expr_len].to_vec())
                        .map_err(|e| Error::internal(format!("invalid default expr: {}", e)))?;
                    pos += expr_len;
                    Some(expr)
                } else {
                    None
                }
            } else {
                None
            };

            // Check expression (length-prefixed string) - optional for backwards compatibility
            let check_expr = if pos + 2 <= data.len() {
                let expr_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if expr_len > 0 && pos + expr_len <= data.len() {
                    let expr = String::from_utf8(data[pos..pos + expr_len].to_vec())
                        .map_err(|e| Error::internal(format!("invalid check expr: {}", e)))?;
                    pos += expr_len;
                    Some(expr)
                } else {
                    None
                }
            } else {
                None
            };

            let mut col = SchemaColumn::with_constraints(
                i,
                &col_name,
                data_type,
                nullable,
                primary_key,
                auto_increment,
                default_expr,
                check_expr,
            );
            col.vector_dimensions = vector_dimensions;
            columns.push(col);
        }

        // Foreign key constraints (optional for backwards compatibility)
        let mut foreign_keys = Vec::new();
        if pos + 2 <= data.len() {
            let fk_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            for _ in 0..fk_count {
                // Once fk_count is declared, truncation mid-constraint is corruption
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let col_idx = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let col_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if pos + col_name_len > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let col_name =
                    String::from_utf8(data[pos..pos + col_name_len].to_vec()).unwrap_or_default();
                pos += col_name_len;

                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let ref_table_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if pos + ref_table_len > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let ref_table =
                    String::from_utf8(data[pos..pos + ref_table_len].to_vec()).unwrap_or_default();
                pos += ref_table_len;

                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let ref_col_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if pos + ref_col_len > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let ref_col =
                    String::from_utf8(data[pos..pos + ref_col_len].to_vec()).unwrap_or_default();
                pos += ref_col_len;

                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "corrupted schema: truncated foreign key constraint data",
                    ));
                }
                let on_delete = crate::core::ForeignKeyAction::from_u8(data[pos])
                    .unwrap_or(crate::core::ForeignKeyAction::Restrict);
                pos += 1;
                let on_update = crate::core::ForeignKeyAction::from_u8(data[pos])
                    .unwrap_or(crate::core::ForeignKeyAction::Restrict);
                pos += 1;

                foreign_keys.push(crate::core::ForeignKeyConstraint {
                    column_index: col_idx,
                    column_name: col_name,
                    referenced_table: ref_table,
                    referenced_column: ref_col,
                    on_delete,
                    on_update,
                });
            }
        }

        // Default values section (after FK constraints, for backward compatibility).
        // Old WAL entries won't have this; populate_schema_defaults() fills from default_expr.
        if pos + 2 <= data.len() {
            let dv_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            for i in 0..dv_count.min(columns.len()) {
                if pos + 2 > data.len() {
                    break;
                }
                let val_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if val_len > 0 && pos + val_len <= data.len() {
                    use super::persistence::deserialize_value;
                    columns[i].default_value = deserialize_value(&data[pos..pos + val_len]).ok();
                    pos += val_len;
                }
            }
        }

        Ok(Schema::with_foreign_keys(
            &table_name,
            columns,
            foreign_keys,
        ))
    }

    /// Closes the engine (inherent method)
    pub fn close_engine(&self) -> Result<()> {
        // Use CAS to atomically check and set closed
        if self
            .open
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(()); // Already closed
        }

        // Stop the cleanup thread first (before stopping transactions)
        {
            let mut cleanup_handle = self.cleanup_handle.lock().unwrap();
            if let Some(mut handle) = cleanup_handle.take() {
                handle.stop();
            }
        }

        // Stop accepting new transactions
        self.registry.stop_accepting_transactions();

        // Run a final checkpoint to seal ALL remaining hot rows into volumes.
        // Use force_seal=true to bypass thresholds — on close, we want all data
        // in volumes so startup is fast and doesn't depend on WAL replay.
        // Skipped when checkpoint_on_close is false (crash simulation in tests),
        // and ALWAYS skipped on read-only engines (no DML to seal, no WAL to
        // truncate, and the shared file lock doesn't permit writes anyway).
        let checkpoint_on_close = self.config.read().unwrap().persistence.checkpoint_on_close
            && !self.is_read_only_mode();
        if checkpoint_on_close {
            if let Some(ref pm) = *self.persistence {
                if pm.is_enabled() {
                    // Retry checkpoint until all hot buffers are empty.
                    // After stop_accepting_transactions(), no new writes can start,
                    // but in-flight commits may still add rows between seal passes.
                    // Retry ensures WAL truncation happens and startup is fast.
                    for attempt in 0..5 {
                        if let Err(e) = self.checkpoint_cycle_inner(true) {
                            eprintln!("Warning: final checkpoint during close failed: {}", e);
                            break;
                        }
                        let all_empty = self
                            .version_stores
                            .read()
                            .unwrap()
                            .values()
                            .all(|s| s.committed_row_count() == 0);
                        if all_empty {
                            break;
                        }
                        if attempt < 4 {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                    self.compact_after_checkpoint_forced();
                }
            }
        } // checkpoint_on_close

        // Wait for any background compaction to finish before releasing
        // resources. Without this, a detached compaction thread could still
        // be rewriting manifests and deleting volume files after close returns.
        while self.compaction_running.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Close all version stores
        let stores = self.version_stores.read().unwrap();
        for store in stores.values() {
            store.close();
        }
        drop(stores);

        // Stop persistence manager
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                if let Err(e) = pm.stop() {
                    eprintln!("Warning: Error stopping persistence: {}", e);
                }
            }
        }

        // Release file lock (drops the lock, allowing other processes to access)
        {
            let mut file_lock = self.file_lock.lock().unwrap();
            *file_lock = None;
        }

        // Release the cross-process shm mapping. Drop unmaps; the file
        // stays on disk so the next writable open replaces it via
        // `create_writer` (truncate + re-init). Reader processes that
        // were attached observe the file truncation by their next
        // refresh seeing a different writer_generation when the new
        // writer reopens.
        {
            let mut shm = self.shm.lock().unwrap();
            *shm = None;
        }

        Ok(())
    }

    /// Returns whether the engine is open
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    /// True iff the engine is in the catastrophic-failure state (a
    /// post-`commit_all_tables` WAL marker write hit an
    /// unrecoverable error and parent VersionStores already hold
    /// markerless data). All durability paths consult this and
    /// refuse to run. The latch itself is set via the
    /// `EngineOperations::mark_engine_failed` trait method from the
    /// transaction commit path.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Stash the EXCLUSIVE `db.startup.lock` guard the
    /// caller holds (typically `restore_from_snapshot`'s
    /// `restore_attach_gate`) into the engine, where it
    /// remains held for the rest of the writer process's
    /// lifetime. Cross-process readers attempting to attach
    /// via `await_writer_startup_quiescent` block on the SH
    /// side until the engine drops. Used by post-destructive-
    /// boundary failures in restore so a partially destroyed
    /// on-disk state is never readable by a new attach.
    /// Idempotent (overwrites any previous stash, dropping
    /// the previous guard if any).
    pub(crate) fn latch_attach_gate_on_failure(&self, gate: Option<StartupLockGuard>) {
        if let Ok(mut slot) = self.failed_restore_attach_gate.lock() {
            *slot = gate;
        }
    }

    /// Manually trip the catastrophic-failure latch from outside
    /// the transaction trait. Used by paths like the auto-commit
    /// CREATE TABLE undo (`drop_table_internal` failure after a
    /// generated-index WAL write failed): if the undo itself
    /// can't run, durable state holds a half-formed table while
    /// this process keeps accepting writes — latching forces
    /// every subsequent durability path to refuse so the
    /// process must restart, where WAL recovery converges to a
    /// consistent state. Idempotent (Release store).
    pub fn enter_catastrophic_failure(&self) {
        self.failed.store(true, Ordering::Release);
    }

    /// Returns true if this engine was opened in read-only mode.
    ///
    /// Used by `Database::open` to refuse to share an existing read-only
    /// engine as a writable handle (which would bypass `ReadOnlyDatabase`'s
    /// gates), and by `MVCCEngine::begin_transaction` as defense-in-depth
    /// against any caller obtaining a `Box<dyn WriteTransaction>` from an
    /// engine that was never meant to write.
    /// Set the WAL replay cap used by `replay_wal` during
    /// `open_engine`. Caller (Database::open / open_read_only)
    /// passes the snapshot of `db.shm.visible_commit_lsn` taken
    /// BEFORE engine construction so replay and the EngineEntry's
    /// `attach_visible_commit_lsn` agree on what was published at
    /// attach time. `u64::MAX` means uncapped (writer recovery).
    pub fn set_replay_cap_lsn(&self, cap: u64) {
        use std::sync::atomic::Ordering;
        self.replay_cap_lsn.store(cap, Ordering::Release);
    }

    pub fn is_read_only_mode(&self) -> bool {
        self.config.read().unwrap().read_only
    }

    /// Defense-in-depth gate for write-intent inherent methods on
    /// `MVCCEngine` reachable through `Database::engine()`.
    ///
    /// Public methods like `create_table`, `drop_table_internal`,
    /// `create_view`, `update_engine_config`, `vacuum`, etc. would
    /// otherwise let an external caller mutate engine state on a
    /// `?read_only=true` `Database` even though the SQL surface and the
    /// `Engine::begin_transaction` trait method are gated. Calling this
    /// helper at the top of each write-intent method closes that back
    /// door without disrupting internal callers (which all go through the
    /// executor's read-only check at the SQL surface, so they never
    /// reach here on a read-only engine in the first place — the gate
    /// fires only when the call originates from outside the executor).
    ///
    /// `#[track_caller]` captures the call-site `file:line` automatically,
    /// so the error message identifies which method tripped the gate
    /// without callers having to pass a method-name string. Avoids the
    /// drift risk of hand-maintained per-method labels.
    #[track_caller]
    fn ensure_writable(&self) -> Result<()> {
        if self.is_read_only_mode() {
            let loc = std::panic::Location::caller();
            return Err(Error::read_only_violation_at(
                "engine",
                &format!("{}:{}", loc.file(), loc.line()),
            ));
        }
        // Refuse direct write helpers (CREATE TABLE / DROP TABLE /
        // ALTER / CREATE INDEX / ... — every path that goes through
        // self.engine.* without first calling
        // begin_writable_transaction_with_level_internal) once the
        // catastrophic-failure latch is set. Without this check, a
        // no-active-transaction DDL would mutate schemas /
        // version_stores AND publish a DDL LSN through record_ddl
        // even though a markerless commit has already poisoned the
        // engine — letting durable / cross-process state continue
        // to advance after recovery will discard the markerless
        // transaction.
        if self.is_failed() {
            return Err(Error::internal(
                "write refused: engine is in the catastrophic-failure state \
                 (a prior commit's WAL marker write failed after some tables \
                 were already committed). Restart the process; recovery will \
                 discard the markerless transaction.",
            ));
        }
        Ok(())
    }

    /// Per-table, per-volume statistics for PRAGMA VOLUME_STATS.
    /// Returns (table_name, segment_id, tier, row_count, memory_bytes, idle_cycles, tombstones).
    pub fn volume_stats(&self) -> Vec<(String, u64, &'static str, usize, usize, u64, usize)> {
        let mgrs = self.segment_managers.read().unwrap();
        let mut result = Vec::new();
        let mut table_names: Vec<&String> = mgrs.keys().collect();
        table_names.sort();
        for table_name in table_names {
            if let Some(mgr) = mgrs.get(table_name) {
                let tombstone_count = mgr.tombstone_count();
                for (seg_id, tier, row_count, mem, idle) in mgr.volume_stats() {
                    result.push((
                        table_name.clone(),
                        seg_id,
                        tier,
                        row_count,
                        mem,
                        idle,
                        tombstone_count,
                    ));
                }
            }
        }
        result
    }

    /// Returns the database path
    pub fn get_path(&self) -> &str {
        &self.path
    }

    /// Reload all per-table manifests from disk. Used by cross-process
    /// readers (v1 SWMR) to pick up writer-side checkpoints. Walks every
    /// `SegmentManager` currently registered with this engine and asks it
    /// to reconcile against `<db>/volumes/<table>/manifest.bin`.
    ///
    /// Schema drift handling (v2 P0.1): each per-table reload is gated on
    /// the segment's `schema_version` vs this engine's current
    /// `schema_epoch`. If any segment was written with a schema newer
    /// than the reader has WAL-replayed, that table's reload returns
    /// `Error::SchemaChanged` and we surface the FIRST such error to
    /// the caller. Other tables' reloads are skipped — a partial reload
    /// would yield a cross-table inconsistent view, which is worse than
    /// an explicit "reopen the database" error.
    ///
    /// DDL detection (v2 P0.2): before the per-table loop, walk the
    /// `<db>/volumes/` directory and compare against the engine's
    /// known segment managers. If the writer has created a NEW table
    /// (manifest.bin exists on disk for a table the reader doesn't
    /// know about) or dropped an EXISTING one (manifest.bin missing
    /// for a table the reader has loaded), surface
    /// `Error::SchemaChanged` so the caller knows to reopen. v1
    /// silently ignored both cases, leading to confusing
    /// "table not found" errors with no SWMR context.
    ///
    /// Returns `Ok(false)` when the on-disk table manifests look like
    /// an in-progress writer checkpoint: at least two tables advertise
    /// different `checkpoint_lsn` values. In that case no in-memory
    /// state is mutated; the caller should keep the prior epoch cached
    /// and retry on a later refresh.
    ///
    /// Crate-internal: called only from `ReadOnlyDatabase::refresh`.
    pub(crate) fn reload_manifests(&self) -> Result<bool> {
        if self.path.is_empty() {
            return Ok(true);
        }

        // ---- DDL detection: compare on-disk tables vs known
        // CATALOG (schemas + views), NOT just segment_managers. A
        // reader can have replayed a CREATE TABLE from WAL into
        // its `schemas` map without yet having a segment_manager
        // — managers are created lazily on first insert/seal. If
        // the writer later inserts and checkpoints, the new
        // manifest directory appears on disk; comparing against
        // segment_managers alone would falsely flag it as a
        // post-attach table creation and surface SchemaChanged.
        // For the "removed" side we still use segment_managers:
        // a manager existing without an on-disk dir is a real
        // anomaly (DROP after manifest persist).
        let known_catalog: rustc_hash::FxHashSet<String> = {
            let schemas = self.schemas.read().unwrap();
            schemas.keys().cloned().collect()
        };
        let known_managers: rustc_hash::FxHashSet<String> = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.keys().cloned().collect()
        };
        let on_disk_tables = scan_table_dirs(std::path::Path::new(&self.path));
        let added: Vec<String> = on_disk_tables
            .iter()
            .filter(|t| !known_catalog.contains(*t))
            .cloned()
            .collect();
        let removed: Vec<String> = known_managers
            .iter()
            .filter(|t| !on_disk_tables.contains(*t))
            .cloned()
            .collect();
        if !added.is_empty() || !removed.is_empty() {
            let mut parts = Vec::new();
            if !added.is_empty() {
                let mut a = added.clone();
                a.sort();
                parts.push(format!("tables added on disk: [{}]", a.join(", ")));
            }
            if !removed.is_empty() {
                let mut r = removed.clone();
                r.sort();
                parts.push(format!("tables dropped on disk: [{}]", r.join(", ")));
            }
            return Err(Error::SchemaChanged(format!(
                "{}; reopen the Database / ReadOnlyDatabase to pick up the new schema",
                parts.join("; ")
            )));
        }

        // ---- Per-table reconcile (with schema-drift gating per-segment) ----
        let max_known_schema = self.schema_epoch.load(std::sync::atomic::Ordering::Acquire);
        // For every table that's BOTH in the known catalog AND has
        // an on-disk volumes dir, ensure a SegmentManager exists.
        // A table replayed from WAL DDL but never inserted on the
        // reader yet has no manager, so a writer's first checkpoint
        // would otherwise leave its new manifest unloaded — the
        // reader would still see count 0 after refresh. `get_or_
        // create_segment_manager` is idempotent, so existing
        // managers are returned unchanged.
        let mgr_arcs: Vec<Arc<crate::storage::volume::manifest::SegmentManager>> = on_disk_tables
            .iter()
            .filter(|t| known_catalog.contains(*t))
            .map(|t| self.get_or_create_segment_manager(t))
            .collect();

        // Stage every table's manifest exactly once, then validate the
        // staged set before mutating any segment manager. This closes two
        // races:
        // - schema drift in a later table after an earlier table already
        //   swapped, and
        // - a writer checkpoint advancing one table between a preflight read
        //   and the actual per-table reload.
        let mut staged_manifests = Vec::with_capacity(mgr_arcs.len());
        let mut checkpoint_min = u64::MAX;
        let mut checkpoint_max = 0u64;
        for mgr in mgr_arcs {
            let Some(manifest) = mgr.read_manifest_from_disk()? else {
                continue;
            };
            mgr.validate_manifest_for_reload(&manifest, max_known_schema)?;
            let lsn = manifest.checkpoint_lsn;
            checkpoint_min = checkpoint_min.min(lsn);
            checkpoint_max = checkpoint_max.max(lsn);
            staged_manifests.push((mgr, manifest));
        }

        // Cross-table checkpoint publish preflight. Manifests are
        // written atomically per table, but the writer persists them
        // one at a time before bumping the global epoch. A reader
        // woken by epoch N can therefore race the writer's next
        // checkpoint and read table A at N+1 while table B is still
        // at N. Reject that staged view before mutating any manager;
        // the next refresh will retry once the writer has finished
        // the group publish.
        if checkpoint_min != u64::MAX && checkpoint_min != checkpoint_max {
            return Ok(false);
        }

        // Now safe to apply per-table reconciles. SchemaChanged
        // can't fire from reload_from_disk anymore (we just
        // verified every manifest is drift-free).
        //
        // Propagate reload errors instead of
        // log-and-continue. `reload_from_disk` returns Err when
        // it can't load a new manifest segment (without mutating
        // state), and the caller (`ReadOnlyDatabase::refresh`)
        // would otherwise still advance `last_seen_epoch` and
        // clear the overlay, leaving the failed table missing
        // from the reader's snapshot until ANOTHER unrelated
        // epoch bump triggers a retry. Surfacing the error keeps
        // the cached epoch unchanged so the very next refresh
        // re-attempts the reload.
        let mut failures: Vec<(String, Error)> = Vec::new();
        for (mgr, manifest) in staged_manifests {
            if let Err(e) = mgr.reload_from_manifest(manifest, max_known_schema) {
                eprintln!(
                    "Warning: Failed to reload manifest for {}: {}",
                    mgr.table_name(),
                    e
                );
                failures.push((mgr.table_name().to_string(), e));
            }
        }
        if !failures.is_empty() {
            // Mid-flight reload may have already swapped earlier
            // tables. Wrap as `SwmrPartialReload` so auto-refresh
            // propagates this distinct from transient I/O — callers
            // need to know the snapshot is mixed and reopen.
            let mut detail = String::new();
            for (i, (table, err)) in failures.iter().enumerate() {
                if i > 0 {
                    detail.push_str("; ");
                }
                detail.push_str(&format!("{}: {}", table, err));
            }
            return Err(Error::SwmrPartialReload(detail));
        }
        Ok(true)
    }

    /// Returns `true` if any live cross-process reader presence lease
    /// exists under `<db>/readers/`. Reaps stale leases (older than
    /// `2 * checkpoint_interval`) as a side effect.
    ///
    /// Used by destructive cleanup paths (volume unlink after compaction
    /// / DROP / TRUNCATE) to defer when readers are attached. v1 SWMR
    /// contract: writer never unlinks a volume while a reader might
    /// still hold its manifest pointer — the reader's lease is the
    /// signal.
    ///
    /// Returns `false` for memory engines (no path) and for paths with
    /// no `readers/` directory (no reader has ever attached).
    ///
    /// **Fail-closed on filesystem error**: if `live_leases()` fails
    /// (permissions, transient I/O, etc.), the writer cannot see the
    /// reader state — treating that as "no live readers" would let
    /// destructive cleanup proceed against unknown reader pins, which
    /// could break a reader's lazy `ensure_volume`. Return `true` so
    /// the caller defers the destructive op until the next attempt
    /// when the FS error may have cleared.
    ///
    /// Crate-internal: this is engine coordination, not user API.
    /// Verified by the in-module unit tests at the bottom of this file.
    /// Public accessor for the SHARED side of
    /// `transactional_ddl_fence`. Used by auto-commit DDL
    /// paths in the executor (CREATE / DROP INDEX, ALTER
    /// TABLE) that mutate catalog state across multiple
    /// engine calls and need to hold the fence across the
    /// whole mutation-to-WAL window — same coverage the
    /// engine's own DDL methods (`create_table`,
    /// `drop_table_internal`, etc.) and the transactional
    /// path's `MvccTransaction::transactional_ddl_guard`
    /// already provide. The returned `Arc` is cloned from
    /// the engine's field; callers acquire `.read()` on it
    /// and hold the guard for the duration of the mutation
    /// + WAL write.
    pub(crate) fn ddl_fence(&self) -> &Arc<parking_lot::RwLock<()>> {
        &self.transactional_ddl_fence
    }

    pub(crate) fn defer_for_live_readers(&self) -> bool {
        if self.path.is_empty() {
            return false;
        }
        let max_age = self.effective_lease_max_age();
        let dir = std::path::Path::new(&self.path).join(crate::storage::mvcc::lease::READERS_DIR);
        // Reap stale first so live_leases sees the post-reap state.
        let _ = crate::storage::mvcc::lease::reap_stale_leases(&dir, max_age);
        // The path exists and we're in a SWMR-eligible mode
        // (memory engines short-circuited above). If the
        // readers/ directory doesn't exist, no reader has
        // ever attached — that's a true negative, not an
        // error. `live_leases` returns Ok(empty) in that case.
        // A real Err means the FS state is unreadable; treat
        // it as "live readers may exist" (fail closed) so a
        // transient permission / I/O failure doesn't unlink
        // volumes a reader is about to lazy-load.
        match crate::storage::mvcc::lease::live_leases(&dir, max_age) {
            Ok(v) => !v.is_empty(),
            Err(_) => true,
        }
    }

    /// Refresh the cached `lease_present` flag and, on
    /// `false → true` transition, do a barrier publish so any
    /// reader that arrived between scans sees coherent shm
    /// state. Called from the cleanup loop on each tick.
    ///
    /// The cached flag drives the commit publish path: when
    /// `false`, commits skip the seqlock dance and just
    /// `fetch_max` `visible_commit_lsn` (preserving the
    /// truncate clamp's contract while saving the
    /// `shm_publish_lock` + `publish_seq` cost). The barrier
    /// publish on transition guarantees that even commits
    /// processed via the no-seqlock fast path produce a
    /// coherent shm snapshot for subsequent readers.
    pub(crate) fn refresh_lease_present_cache(&self) {
        // Memory / non-SWMR engines have no readers ever; keep
        // the flag at default (true) and don't pay the lease
        // scan cost (defer_for_live_readers short-circuits on
        // empty path).
        if self.path.is_empty() {
            return;
        }
        let observed = self.defer_for_live_readers();
        let prior = self.lease_present.swap(observed, Ordering::AcqRel);
        if !prior && observed {
            // false → true: re-sync shm under the full seqlock.
            // The fast-path commits during the no-readers window
            // advanced `visible_commit_lsn` but left
            // `oldest_active_txn_lsn` UNTOUCHED — possibly
            // stale-low from the last slow-path publish. We
            // can't use `publish_visible_commit_lsn_local`
            // here because it short-circuits when
            // `safe_visible <= current visible` (which is the
            // common case after a fast-path run that already
            // set visible to `max_written_marker_lsn`). Use
            // the dedicated barrier helper that always stores
            // a fresh oldest under the seqlock regardless of
            // whether visible advances.
            self.barrier_publish_full_state();
        }
    }

    fn flush_wal_for_visibility_if_due(&self) -> bool {
        let Some(pm) = self.persistence.as_ref().as_ref() else {
            return false;
        };
        let Some(wal) = pm.wal() else {
            return false;
        };
        match wal.flush_for_visibility_if_due() {
            Ok(advanced) => advanced,
            Err(e) => {
                eprintln!("Warning: periodic WAL visibility flush failed: {}", e);
                false
            }
        }
    }

    /// Force a seqlock-bracketed publish of BOTH
    /// `visible_commit_lsn` (via `fetch_max` against the current
    /// safe-visible watermark capped to the WAL file-write frontier)
    /// AND a freshly sampled
    /// `oldest_active_txn_lsn`. Unlike
    /// `publish_visible_commit_lsn_local`, this method runs
    /// the seqlock dance even when `visible_commit_lsn` would
    /// not advance — its job is to re-sync the shm header
    /// fields after a no-readers window during which fast-path
    /// commits left `oldest_active_txn_lsn` stale.
    ///
    /// Called from `refresh_lease_present_cache` on the
    /// `false → true` lease transition, before any reader can
    /// sample the just-flipped state.
    fn barrier_publish_full_state(&self) {
        // Compute safe_visible the same way the publish paths
        // do: bounded by the lowest pending marker (if any)
        // OR the highest written marker.
        let safe_visible = {
            let pending = self.pending_marker_lsns.lock();
            if let Some(&min_pending) = pending.iter().next() {
                min_pending.saturating_sub(1)
            } else {
                self.max_written_marker_lsn.load(Ordering::Acquire)
            }
        };
        let publish_lsn = cap_visible_lsn_by_flushed(&self.persistence, safe_visible);
        let shm = self.shm.lock().unwrap();
        let Some(handle) = shm.as_ref() else { return };
        let _publish_guard = self.shm_publish_lock.lock();
        // Always run the seqlock pair: bump-odd, store oldest,
        // store visible (fetch_max — may be no-op), bump-even.
        // Storing oldest unconditionally is the WHOLE POINT —
        // we're catching up after a no-readers window.
        handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
        if let Some(pm) = self.persistence.as_ref().as_ref() {
            if let Some(wal) = pm.wal() {
                let oldest = wal.oldest_active_txn_lsn();
                handle
                    .header()
                    .oldest_active_txn_lsn
                    .store(oldest, Ordering::Release);
            }
        }
        if publish_lsn > 0 {
            handle
                .header()
                .visible_commit_lsn
                .fetch_max(publish_lsn, Ordering::AcqRel);
        }
        handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
        let visible_after = handle.header().visible_commit_lsn.load(Ordering::Acquire);
        if clear_published_completed_txns(
            &self.completed_marker_txns,
            &self.persistence,
            visible_after,
        ) {
            if let Some(pm) = self.persistence.as_ref().as_ref() {
                if let Some(wal) = pm.wal() {
                    handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                    handle
                        .header()
                        .oldest_active_txn_lsn
                        .store(wal.oldest_active_txn_lsn(), Ordering::Release);
                    handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
    }

    /// SWMR v2 Phase E: borrow the engine's `WALManager`, if any.
    /// Reader-side overlay rebuild uses this to call
    /// `WALManager::tail_committed_entries`. Returns `None` for
    /// memory engines or when persistence is disabled.
    pub fn wal(&self) -> Option<&crate::storage::mvcc::wal_manager::WALManager> {
        self.persistence.as_ref().as_ref().and_then(|pm| pm.wal())
    }

    /// SWMR v2: snapshot of every catalog object the
    /// engine knows about, lower-cased. Includes:
    ///   - All tables in `self.schemas` (covers empty / hot-only
    ///     tables that don't yet have a segment manager).
    ///   - All views in `self.views`.
    ///
    /// Used by `ReadOnlyDatabase::maybe_rebuild_overlay` to suppress
    /// post-checkpoint DDL re-records (`CreateTable`,
    /// `CreateIndex`, `CreateView`) for objects the reader already
    /// knew about at attach time. Falls back to `table_checkpoint_lsns`
    /// would only cover tables WITH segments, missing
    /// empty-but-defined tables and all views — those would
    /// falsely surface as `SwmrPendingDdl` after every writer
    /// checkpoint.
    pub fn known_catalog_objects(&self) -> rustc_hash::FxHashSet<String> {
        let mut out = rustc_hash::FxHashSet::default();
        {
            let schemas = self.schemas.read().unwrap();
            for name in schemas.keys() {
                out.insert(name.clone());
            }
        }
        {
            let views = self.views.read().unwrap();
            for name in views.keys() {
                out.insert(name.clone());
            }
        }
        out
    }

    /// Snapshot of all known index names across every loaded version
    /// store. Used by `ReadOnlyDatabase::refresh` to suppress
    /// CreateIndex re-records for indexes the reader already knows
    /// about, while still surfacing brand-new indexes as
    /// `SwmrPendingDdl`. Names are stored case-sensitively (matching
    /// `IndexMetadata::name`).
    pub fn known_index_names(&self) -> rustc_hash::FxHashSet<String> {
        let mut out = rustc_hash::FxHashSet::default();
        let stores = self.version_stores.read().unwrap();
        for store in stores.values() {
            for idx in store.get_all_indexes() {
                out.insert(idx.name().to_string());
            }
        }
        out
    }

    /// SWMR v2 Phase G: snapshot of per-table `checkpoint_lsn` from
    /// every loaded segment manager. Returned map's keys are
    /// lower-case table names. Used by `ReadOnlyDatabase::refresh` to
    /// detect which tables actually changed since the last refresh,
    /// so cache invalidation can run per-table instead of bulk
    /// clearing every cached plan and stats entry.
    pub fn table_checkpoint_lsns(&self) -> rustc_hash::FxHashMap<String, u64> {
        let mgrs = self.segment_managers.read().unwrap();
        let mut out =
            rustc_hash::FxHashMap::with_capacity_and_hasher(mgrs.len(), Default::default());
        for (name, mgr) in mgrs.iter() {
            out.insert(name.clone(), mgr.manifest().checkpoint_lsn);
        }
        out
    }

    /// SWMR v2 Phase D: read the minimum pinned_lsn across live v2
    /// reader leases. Returns `None` when there is no v2 reader (no
    /// constraint on WAL truncation). Same `max_age` policy as
    /// `defer_for_live_readers` so the two helpers agree on which
    /// leases are alive.
    ///
    /// **Fail-closed on filesystem error**: if `min_pinned_lsn`
    /// returns Err, the writer cannot tell what readers have pinned.
    /// Returns `Some(1)` so `compute_wal_truncate_floor` cascades
    /// into its `Some(1) => None` arm and refuses every WAL
    /// truncation until lease state can be inspected again.
    pub(crate) fn min_pinned_reader_lsn(&self) -> Option<u64> {
        if self.path.is_empty() {
            return None;
        }
        let max_age = self.effective_lease_max_age();
        let dir = std::path::Path::new(&self.path).join(crate::storage::mvcc::lease::READERS_DIR);
        match crate::storage::mvcc::lease::min_pinned_lsn(&dir, max_age) {
            Ok(v) => v,
            Err(_) => Some(1),
        }
    }

    /// Effective lease max-age — same derivation
    /// `defer_for_live_readers` and `min_pinned_reader_lsn`
    /// both use, plus a write-through to
    /// `lease_max_age_nanos` so `EngineOperations` (which
    /// doesn't hold the config) shares the exact same window
    /// the engine uses. Anything shorter would let
    /// transactional DROP unlink volumes a reader the engine
    /// still considers live is about to lazy-load.
    fn effective_lease_max_age(&self) -> std::time::Duration {
        let cfg = self.config.read().unwrap();
        let max_age = if cfg.persistence.lease_max_age_secs > 0 {
            // User-configured override via `?lease_max_age=N` — trusted
            // verbatim. The user knows their workload; we don't
            // second-guess with a floor.
            std::time::Duration::from_secs(cfg.persistence.lease_max_age_secs as u64)
        } else {
            // Engine-derived default: 2x checkpoint_interval with a
            // 120s floor so very-aggressive checkpoint cadences (5s)
            // don't reap a reader that just touched its lease but
            // happens to be paused for GC.
            let interval = cfg.persistence.checkpoint_interval;
            std::time::Duration::from_secs(((interval * 2) as u64).max(120))
        };
        drop(cfg);
        // Cache the value so EngineOperations and any other
        // shared-Arc consumer picks up the same window
        // without rereading config (and without us paying
        // the config-lock cost in the hot WAL-truncate path).
        self.lease_max_age_nanos
            .store(max_age.as_nanos() as u64, Ordering::Release);
        max_age
    }

    /// SWMR v2 Phase D: cap the WAL truncate point at `min_pinned_lsn
    ///   - 1` so a v2 reader pinned at `min_pinned_lsn` keeps the
    ///     entries it still needs. Returns `None` to skip truncation
    ///     entirely (no safe floor: a reader is pinned at LSN 1, meaning
    ///     *every* entry is needed). Returns `Some(checkpoint_lsn)` in
    ///     the no-reader case (typical).
    ///
    /// Also clamped by the published `visible_commit_lsn`: WAL
    /// records past the visible frontier (e.g. transactional
    /// DDL whose publish was deferred to commit) MUST stay on
    /// disk so a fresh read-only attach pinning at
    /// `attach_visible_commit_lsn` doesn't observe a chain
    /// head past its pin and trip `SwmrSnapshotExpired`. This
    /// clamp turns the prior implicit invariant
    /// (`chain_head <= visible_commit_lsn`) — which was held
    /// because every WAL write also published — into an
    /// explicit constraint, allowing publish-deferred WAL
    /// writes to coexist safely with checkpoint truncation.
    fn compute_wal_truncate_floor(&self, checkpoint_lsn: u64) -> Option<u64> {
        // Refuse all WAL truncation while a prior
        // `finalize_committed_drops` has unfinished cleanup
        // (its `manifest.bin` survived on disk despite the
        // catalog reflecting a DROP). Truncating past the
        // associated `DropTable` record while the manifest
        // is still discoverable would leave a future open
        // unable to converge: recovery wouldn't replay the
        // drop, and the leftover manifest would let
        // `scan_table_dirs` resurface the dropped table.
        // Cleared by `sweep_orphan_table_dirs` once the
        // directory is confirmed gone.
        if !self.pending_drop_cleanups.lock().is_empty() {
            return None;
        }
        // Same DropTable / leftover-manifest concern as
        // `pending_drop_cleanups` above: if the most recent
        // `sweep_orphan_table_dirs` could not enumerate
        // `volumes/`, we have no proof there are no orphan
        // table dirs to gate truncation on. Refuse until a
        // discovery pass succeeds.
        if self.orphan_discovery_failed.load(Ordering::Acquire) {
            return None;
        }
        let visible_clamp = self.published_visible_commit_lsn();
        let bounded_checkpoint = if visible_clamp > 0 {
            checkpoint_lsn.min(visible_clamp)
        } else {
            checkpoint_lsn
        };
        match self.min_pinned_reader_lsn() {
            None => Some(bounded_checkpoint),
            Some(0) => Some(bounded_checkpoint), // 0 means no pin, ignore
            Some(1) => None,                     // can't safely truncate: every entry needed
            Some(pinned) => Some(bounded_checkpoint.min(pinned - 1)),
        }
    }

    /// Sample the published `visible_commit_lsn` from
    /// `db.shm` so `compute_wal_truncate_floor` can clamp
    /// against it. Falls back to 0 (no clamp) when no shm is
    /// attached (in-memory engine, non-Unix, or shm creation
    /// failed) — those paths don't have cross-process readers
    /// pinning at the visible frontier anyway.
    fn published_visible_commit_lsn(&self) -> u64 {
        let shm = self.shm.lock().unwrap();
        match shm.as_ref() {
            Some(handle) => handle
                .header()
                .visible_commit_lsn
                .load(std::sync::atomic::Ordering::Acquire),
            None => 0,
        }
    }

    /// SWMR v2 Phase D: publish the current `min_pinned_lsn` to db.shm
    /// so PRAGMA SWMR_STATUS and external monitors can observe what's
    /// holding back WAL truncation. `0` means "no v2 reader pinning"
    /// (default state). Called from checkpoint paths after the
    /// truncate decision so the published value reflects the same
    /// scan that the writer just used.
    fn publish_min_pinned_lsn(&self) {
        let value = self.min_pinned_reader_lsn().unwrap_or(0);
        let shm = self.shm.lock().unwrap();
        if let Some(handle) = shm.as_ref() {
            handle
                .header()
                .min_pinned_lsn
                .store(value, Ordering::Release);
        }
    }

    /// Returns a copy of the configuration
    pub fn config(&self) -> Config {
        self.config.read().unwrap().clone()
    }

    /// Updates the engine configuration
    pub fn update_engine_config(&self, config: Config) -> Result<()> {
        self.ensure_writable()?;
        let current = self.config.read().unwrap();
        if config.path != current.path {
            return Err(Error::internal("cannot change database path after opening"));
        }
        drop(current);

        *self.config.write().unwrap() = config;
        Ok(())
    }

    /// Returns the transaction registry
    pub fn registry(&self) -> Arc<TransactionRegistry> {
        Arc::clone(&self.registry)
    }

    /// Replay an ALTER TABLE operation from WAL
    fn replay_alter_table(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::internal("empty ALTER TABLE data"));
        }

        let op_type = data[0];
        let mut pos = 1;

        // Read table name
        if pos + 2 > data.len() {
            return Err(Error::internal(
                "invalid ALTER TABLE data: missing table name length",
            ));
        }
        let table_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + table_name_len > data.len() {
            return Err(Error::internal(
                "invalid ALTER TABLE data: missing table name",
            ));
        }
        let table_name = String::from_utf8(data[pos..pos + table_name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid table name: {}", e)))?;
        pos += table_name_len;

        match op_type {
            1 => {
                // AddColumn
                // Read column name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid AddColumn data: missing column name length",
                    ));
                }
                let col_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + col_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid AddColumn data: missing column name",
                    ));
                }
                let column_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;
                pos += col_name_len;

                // Read data type (1 byte, + 2 bytes dimension for Vector)
                if pos >= data.len() {
                    return Err(Error::internal("invalid AddColumn data: missing data type"));
                }
                let dt_tag = data[pos];
                let data_type = DataType::from_u8(dt_tag)
                    .ok_or_else(|| Error::internal("invalid data type byte"))?;
                pos += 1;
                let mut vector_dimensions: u16 = 0;
                if data_type == DataType::Vector && pos + 2 <= data.len() {
                    vector_dimensions = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                    pos += 2;
                }

                // Read nullable
                if pos >= data.len() {
                    return Err(Error::internal("invalid AddColumn data: missing nullable"));
                }
                let nullable = data[pos] != 0;
                pos += 1;

                // Read default expression
                let default_expr = if pos + 2 <= data.len() {
                    let expr_len =
                        u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                    pos += 2;
                    if expr_len > 0 && pos + expr_len <= data.len() {
                        let expr = String::from_utf8(data[pos..pos + expr_len].to_vec())
                            .map_err(|e| Error::internal(format!("invalid default expr: {}", e)))?;
                        Some(expr)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Apply the ADD COLUMN using engine method
                self.create_column_with_default(
                    &table_name,
                    &column_name,
                    data_type,
                    nullable,
                    default_expr,
                    vector_dimensions,
                )?;
            }
            2 => {
                // DropColumn
                // Read column name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid DropColumn data: missing column name length",
                    ));
                }
                let col_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + col_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid DropColumn data: missing column name",
                    ));
                }
                let column_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;

                // Apply the DROP COLUMN using engine method
                self.drop_column(&table_name, &column_name)?;
            }
            3 => {
                // RenameColumn
                // Read old column name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid RenameColumn data: missing old name length",
                    ));
                }
                let old_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + old_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid RenameColumn data: missing old name",
                    ));
                }
                let old_name = String::from_utf8(data[pos..pos + old_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid old column name: {}", e)))?;
                pos += old_name_len;

                // Read new column name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid RenameColumn data: missing new name length",
                    ));
                }
                let new_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + new_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid RenameColumn data: missing new name",
                    ));
                }
                let new_name = String::from_utf8(data[pos..pos + new_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid new column name: {}", e)))?;

                // Apply the RENAME COLUMN using engine method
                self.rename_column(&table_name, &old_name, &new_name)?;
            }
            4 => {
                // ModifyColumn
                // Read column name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing column name length",
                    ));
                }
                let col_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + col_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing column name",
                    ));
                }
                let column_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;
                pos += col_name_len;

                // Read data type (1 byte, + 2 bytes dimension for Vector)
                if pos >= data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing data type",
                    ));
                }
                let data_type = DataType::from_u8(data[pos])
                    .ok_or_else(|| Error::internal("invalid data type byte"))?;
                pos += 1;
                let mut vector_dimensions: u16 = 0;
                if data_type == DataType::Vector && pos + 2 <= data.len() {
                    vector_dimensions = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                    pos += 2;
                }

                // Read nullable
                if pos >= data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing nullable",
                    ));
                }
                let nullable = data[pos] != 0;

                // Apply the MODIFY COLUMN using engine method
                self.modify_column_with_dimensions(
                    &table_name,
                    &column_name,
                    data_type,
                    nullable,
                    vector_dimensions,
                )?;
            }
            5 => {
                // RenameTable - special handling since table name changes
                // The table_name read above is actually old_table_name for RenameTable
                // because we re-serialized in that format

                // Read new table name
                if pos + 2 > data.len() {
                    return Err(Error::internal(
                        "invalid RenameTable data: missing new name length",
                    ));
                }
                let new_name_len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;

                if pos + new_name_len > data.len() {
                    return Err(Error::internal(
                        "invalid RenameTable data: missing new name",
                    ));
                }
                let new_table_name = String::from_utf8(data[pos..pos + new_name_len].to_vec())
                    .map_err(|e| Error::internal(format!("invalid new table name: {}", e)))?;

                // Apply the RENAME TABLE using engine method
                self.rename_table(&table_name, &new_table_name)?;
            }
            _ => {
                return Err(Error::internal(format!(
                    "unknown ALTER TABLE operation type: {}",
                    op_type
                )));
            }
        }

        Ok(())
    }

    /// Check if we should skip WAL writes (during recovery replay)
    fn should_skip_wal(&self) -> bool {
        self.loading_from_disk.load(Ordering::Acquire)
    }

    /// Record a DDL operation to WAL. SWMR v2 Phase C: also publishes
    /// the marker LSN to `db.shm` so reader processes' WAL-tail can
    /// observe the new DDL via the same `visible_commit_lsn` watermark
    /// they already poll.
    fn record_ddl(&self, table_name: &str, op: WALOperationType, schema_data: &[u8]) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }
        // Defense in depth: ensure_writable() guards public DDL
        // helpers, but record_ddl is reached from internal callers
        // (rerecord_ddl_to_wal in the checkpoint cycle, the
        // record_create_index / alter / truncate helpers) and from
        // already-open transactions that bypassed the begin-time
        // check. Refuse here so no DDL LSN is ever published after
        // the catastrophic-failure latch — recovery will discard
        // the markerless transaction, and any DDL that landed after
        // it would diverge live state from on-disk state.
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "record_ddl refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart \
                 the process; recovery will discard markerless transactions.",
            ));
        }
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                // SERIALIZE the DDL entry+marker append against
                // concurrent publication of `visible_commit_lsn`.
                // `record_ddl_operation` does TWO appends — the
                // DDL record itself, then the DDL_TXN_ID commit
                // marker. Without this gate, a concurrent user
                // commit could publish a visible LSN BETWEEN the
                // two appends. A read-only SWMR refresh on that
                // window would tail past the DDL record and skip
                // it (no marker yet visible); a subsequent
                // refresh would only tail (last_applied, marker]
                // and the stale-DDL filter (`lsn <= from_lsn`)
                // would drop the older DDL entry. Reader keeps
                // serving the old schema.
                //
                // Holding `pending_marker_lsns` across the two
                // appends blocks every other publish path
                // (`publish_visible_commit_lsn` /
                // `publish_visible_commit_lsn_local`) for the
                // duration. Other commits can still APPEND to
                // WAL (different lock), but they can't advance
                // `visible_commit_lsn`. By the time they unblock,
                // both the DDL entry AND the DDL marker are in
                // WAL — any visible LSN they then advertise that
                // covers our entry also covers our marker.
                //
                // We MUST release the gate before calling
                // `publish_visible_commit_lsn_local` (it also
                // takes the same lock — would deadlock).
                let lsn = {
                    let _gate = self.pending_marker_lsns.lock();
                    pm.record_ddl_operation(table_name, op, schema_data)?
                };
                self.publish_visible_commit_lsn_local(lsn);
            }
        }
        Ok(())
    }

    /// SWMR v2 Phase C: publish a new
    /// `visible_commit_lsn` to db.shm. Same as the trait method on
    /// `EngineOperations` but for inherent paths (DDL) that don't
    /// go through the transaction commit pipeline.
    ///
    /// CRITICAL ordering: the writer's current oldest-active-txn
    /// LSN is stored BEFORE `visible_commit_lsn`. Release-Acquire
    /// pairing on `visible_commit_lsn` guarantees that any reader
    /// observing the new visible LSN also observes the matching
    /// (or lower) watermark. Without this ordering, a reader that
    /// Acquire-loads visible_commit_lsn between our two stores
    /// would see the new visible LSN but a STALE high watermark,
    /// causing it to advance `next_entry_floor` past in-flight DML
    /// LSNs and silently skip those rows on the next refresh.
    fn publish_visible_commit_lsn_local(&self, lsn: u64) {
        if lsn == 0 {
            return;
        }
        // DDL is auto-committed inline (no separate
        // complete_commit fires later), so it doesn't add to
        // pending_marker_lsns. But it DOES bump
        // max_written_marker_lsn so the safe-visible watermark can
        // advance to it once any pending non-DDL markers drain.
        //
        // Hold the pending lock across both the
        // bump AND the safe-visible computation so a concurrent
        // EngineOperations commit can't interleave between them.
        let safe_visible = {
            let pending = self.pending_marker_lsns.lock();
            self.max_written_marker_lsn.fetch_max(lsn, Ordering::AcqRel);
            if let Some(&min_pending) = pending.iter().next() {
                min_pending.saturating_sub(1)
            } else {
                self.max_written_marker_lsn.load(Ordering::Acquire)
            }
        };
        if safe_visible == 0 {
            return;
        }
        let publish_lsn = cap_visible_lsn_by_flushed(&self.persistence, safe_visible);
        if publish_lsn == 0 {
            return;
        }
        let shm = self.shm.lock().unwrap();
        if let Some(handle) = shm.as_ref() {
            // Serialize the entire seqlock publish (odd → stores
            // → even) under `shm_publish_lock`. Without this,
            // two concurrent publishes could interleave their
            // odd/even bumps so a reader observes an even seq
            // while a different writer's stores are mid-flight.
            let _publish_guard = self.shm_publish_lock.lock();
            // Skip the publish entirely when this flushed-capped visible is
            // <= what's already published. visible_commit_lsn uses
            // fetch_max so the visible store would be a no-op, but
            // an unconditional `oldest_active_txn_lsn.store(...)`
            // would still OVERWRITE the floor. A higher prior
            // publish may have already cleared the txn that made
            // visibility advance; storing a freshly-sampled
            // (potentially HIGHER) oldest now would pair the new
            // high visible with a high floor, letting readers skip
            // that txn's pre-window DML.
            if publish_lsn <= handle.header().visible_commit_lsn.load(Ordering::Acquire) {
                return;
            }
            // Seqlock publish: bump to ODD BEFORE the field
            // stores so a concurrent reader sample observes
            // "publish in progress" and retries. Then store both
            // fields. Then bump to EVEN AFTER both stores — the
            // pair is now coherent for reader sampling. See
            // `ShmHeader::publish_seq` doc for why "bump-after-
            // only" admits a torn read.
            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
            // Store watermark FIRST.
            if let Some(pm) = self.persistence.as_ref().as_ref() {
                if let Some(wal) = pm.wal() {
                    handle
                        .header()
                        .oldest_active_txn_lsn
                        .store(wal.oldest_active_txn_lsn(), Ordering::Release);
                }
            }
            // Then visible_commit_lsn — readers Acquire-loading
            // this also see the watermark store above.
            handle
                .header()
                .visible_commit_lsn
                .fetch_max(publish_lsn, Ordering::AcqRel);
            // Bump to EVEN: publish complete, pair is coherent.
            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Serialize a schema to binary format for WAL
    pub(crate) fn serialize_schema(schema: &Schema) -> Vec<u8> {
        let mut buf = Vec::new();

        // Table name
        buf.extend_from_slice(&(schema.table_name.len() as u16).to_le_bytes());
        buf.extend_from_slice(schema.table_name.as_bytes());

        // Column count
        buf.extend_from_slice(&(schema.columns.len() as u16).to_le_bytes());

        // Columns
        for col in &schema.columns {
            // Column name
            buf.extend_from_slice(&(col.name.len() as u16).to_le_bytes());
            buf.extend_from_slice(col.name.as_bytes());

            // Data type (1 byte, + 2 bytes dimension for Vector)
            buf.push(col.data_type.as_u8());
            if col.data_type == DataType::Vector {
                buf.extend_from_slice(&col.vector_dimensions.to_le_bytes());
            }

            // Nullable (1 byte)
            buf.push(if col.nullable { 1 } else { 0 });

            // Primary key (1 byte)
            buf.push(if col.primary_key { 1 } else { 0 });

            // Auto-increment (1 byte)
            buf.push(if col.auto_increment { 1 } else { 0 });

            // Default expression (length-prefixed string, 0 length if None)
            if let Some(ref default_expr) = col.default_expr {
                buf.extend_from_slice(&(default_expr.len() as u16).to_le_bytes());
                buf.extend_from_slice(default_expr.as_bytes());
            } else {
                buf.extend_from_slice(&0u16.to_le_bytes());
            }

            // Check expression (length-prefixed string, 0 length if None)
            if let Some(ref check_expr) = col.check_expr {
                buf.extend_from_slice(&(check_expr.len() as u16).to_le_bytes());
                buf.extend_from_slice(check_expr.as_bytes());
            } else {
                buf.extend_from_slice(&0u16.to_le_bytes());
            }
        }

        // Foreign key constraints
        buf.extend_from_slice(&(schema.foreign_keys.len() as u16).to_le_bytes());
        for fk in &schema.foreign_keys {
            buf.extend_from_slice(&(fk.column_index as u16).to_le_bytes());
            buf.extend_from_slice(&(fk.column_name.len() as u16).to_le_bytes());
            buf.extend_from_slice(fk.column_name.as_bytes());
            buf.extend_from_slice(&(fk.referenced_table.len() as u16).to_le_bytes());
            buf.extend_from_slice(fk.referenced_table.as_bytes());
            buf.extend_from_slice(&(fk.referenced_column.len() as u16).to_le_bytes());
            buf.extend_from_slice(fk.referenced_column.as_bytes());
            buf.push(fk.on_delete.as_u8());
            buf.push(fk.on_update.as_u8());
        }

        // Default values section (after FK constraints for backward compatibility).
        // Old WAL readers stop after FKs; new readers detect this section by
        // checking if data remains.
        {
            use super::persistence::serialize_value;
            buf.extend_from_slice(&(schema.columns.len() as u16).to_le_bytes());
            for col in &schema.columns {
                if let Some(ref default_value) = col.default_value {
                    if let Ok(val_bytes) = serialize_value(default_value) {
                        buf.extend_from_slice(&(val_bytes.len() as u16).to_le_bytes());
                        buf.extend_from_slice(&val_bytes);
                    } else {
                        buf.extend_from_slice(&0u16.to_le_bytes());
                    }
                } else {
                    buf.extend_from_slice(&0u16.to_le_bytes());
                }
            }
        }

        buf
    }

    /// Get a table handle for an existing transaction by txn_id.
    /// This allows FK enforcement to participate in the caller's transaction,
    /// ensuring CASCADE effects are atomic and uncommitted rows are visible.
    pub(crate) fn get_table_for_txn(
        &self,
        txn_id: i64,
        table_name: &str,
    ) -> Result<Box<dyn crate::storage::traits::WriteTable>> {
        EngineOperations::new(self).get_table_for_transaction(txn_id, table_name)
    }

    /// Find all FK constraints in other tables that reference the given parent table.
    /// Uses a cached reverse mapping that is rebuilt only when schema_epoch changes.
    /// Returns Arc-wrapped Vec for zero-copy sharing (ref-count bump only).
    /// Zero cost for databases without FK constraints.
    pub(crate) fn find_referencing_fks(
        &self,
        parent_table: &str,
    ) -> Arc<Vec<(String, ForeignKeyConstraint)>> {
        static EMPTY: std::sync::LazyLock<Arc<Vec<(String, ForeignKeyConstraint)>>> =
            std::sync::LazyLock::new(|| Arc::new(Vec::new()));

        let current_epoch = self.schema_epoch.load(Ordering::Acquire);

        // Fast path: check if cache is valid (read lock only)
        {
            let cache = self.fk_reverse_cache.read().unwrap();
            if cache.0 == current_epoch {
                return cache
                    .1
                    .get(parent_table)
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&EMPTY));
            }
        }

        // Cache miss: rebuild under write lock
        let mut cache = self.fk_reverse_cache.write().unwrap();
        // Double-check after acquiring write lock (another thread may have rebuilt)
        if cache.0 == current_epoch {
            return cache
                .1
                .get(parent_table)
                .cloned()
                .unwrap_or_else(|| Arc::clone(&EMPTY));
        }

        // Rebuild the full reverse mapping
        let schemas = self.schemas.read().unwrap();
        let mut map: StringMap<Vec<(String, ForeignKeyConstraint)>> = StringMap::default();
        for schema in schemas.values() {
            for fk in &schema.foreign_keys {
                map.entry(fk.referenced_table.clone())
                    .or_default()
                    .push((schema.table_name_lower.clone(), fk.clone()));
            }
        }
        // Wrap each Vec in Arc before storing in cache
        let arc_map: StringMap<Arc<Vec<(String, ForeignKeyConstraint)>>> =
            map.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
        *cache = (current_epoch, arc_map);

        cache
            .1
            .get(parent_table)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&EMPTY))
    }

    /// Creates a new table
    pub fn create_table(&self, schema: Schema) -> Result<Schema> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Hold the SHARED side of `transactional_ddl_fence`
        // across the entire mutation-to-WAL window. Checkpoint's
        // `rerecord_ddl_to_wal` takes EX, so this fence
        // blocks any concurrent checkpoint from snapshotting
        // the transient catalog (in-memory inserts done
        // BEFORE `record_ddl` writes the durable CreateTable
        // record). Without this guard a checkpoint that
        // landed mid-window would persist a `checkpoint_lsn`
        // / manifests reflecting the new table while
        // recovery sees no `CreateTable` WAL entry — a
        // phantom CREATE if `record_ddl` later fails, or a
        // missing-DDL CREATE that survives until the next
        // explicit re-record. The transactional DDL paths
        // already hold this fence via
        // `MvccTransaction::transactional_ddl_guard`.
        let _ddl_fence_guard = self.transactional_ddl_fence.read();

        let table_name = schema.table_name_lower.clone();

        // Validate schema before acquiring locks
        self.validate_schema(&schema)?;

        // Create version store for this table
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            registry_as_visibility_checker(&self.registry),
        ));

        // Register virtual PkIndex for INTEGER PRIMARY KEY
        register_pk_index(&schema, &version_store);

        // Prepare WAL data before locks (avoids holding lock during serialization)
        let schema_data = Self::serialize_schema(&schema);

        // Atomically check-and-insert under write lock to
        // prevent TOCTOU race. The pending-drop check happens
        // INSIDE the schemas write critical section using the
        // same `schemas → pending` lock order DROP follows
        // (see `drop_table_internal` /
        // `EngineOperations::drop_table`); without that, a
        // concurrent DROP that grabs schemas + inserts
        // pending + removes schema could land between this
        // CREATE's pre-check and its schemas acquire,
        // letting CREATE see "schemas missing AND pending
        // empty" even though a DROP is in progress.
        let return_schema = schema.clone();
        {
            let mut schemas = self.schemas.write().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists(table_name.to_string()));
            }
            // Recheck pending under the same critical
            // section. The leftover
            // `<volumes>/<table>/manifest.bin` from a still-
            // pending DROP would otherwise be reusable by
            // the new table — and since
            // `pending_drop_cleanups` is memory-only, a
            // checkpoint+restart in that state could
            // truncate WAL past the original DropTable, then
            // the next reopen would load the OLD cold rows
            // into the newly-created same-name table.
            if self.pending_drop_cleanups.lock().contains(&table_name) {
                return Err(Error::internal(format!(
                    "CREATE TABLE refused: a prior DROP/TRUNCATE for '{}' is still \
                     pending physical cleanup (live cross-process readers, or a \
                     cleanup I/O failure). Wait for the orphan sweep to drain \
                     before recreating the table.",
                    table_name
                )));
            }
            schemas.insert(table_name.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name.clone(), version_store);
        }

        // Record DDL to WAL only after successful insertion. Roll
        // back the in-memory insert on failure: `record_ddl` can
        // refuse the write (catastrophic-failure latch tripped
        // between our entry-time check and now) or fail on WAL
        // I/O. Without rollback the new table stays visible in
        // this process while WAL recovery on restart will not
        // replay it, leaving callers with a CREATE that
        // succeeded then "vanished".
        if let Err(e) = self.record_ddl(
            &return_schema.table_name,
            WALOperationType::CreateTable,
            &schema_data,
        ) {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(&table_name) {
                store.close();
            }
            drop(stores);
            let mut schemas = self.schemas.write().unwrap();
            schemas.remove(&table_name);
            drop(schemas);
            // Bump schema_epoch on the failure path too. The
            // transient table was visible in `schemas` /
            // `version_stores` between the inserts above and
            // this revert, so a concurrent reader may have
            // rebuilt `fk_reverse_cache` (or a compiled DML
            // fast path) against the transient catalog and
            // stored it under the old epoch. Without this
            // bump that stale cache survives the revert and
            // returns FK / column lookups against a table
            // this process no longer knows about.
            self.schema_epoch.fetch_add(1, Ordering::Release);
            return Err(e);
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(return_schema)
    }

    /// Drops a table
    pub fn drop_table_internal(&self, name: &str) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Hold the SHARED side of `transactional_ddl_fence`
        // across the mutation-to-WAL window — same rationale
        // as `MVCCEngine::create_table`. Without this guard
        // a checkpoint that landed mid-window could persist
        // manifests + truncate WAL between this DROP's
        // schemas-remove and its `record_ddl` call, leaving
        // recovery with a missing-DROP catalog (manifest /
        // checkpoint_lsn reflect the schema being gone, but
        // there's no `DropTable` WAL entry to apply on the
        // next open).
        let _ddl_fence_guard = self.transactional_ddl_fence.read();

        let table_name = name.to_lowercase();

        // Snapshot the pre-drop schema AND the FK-edit deltas to
        // child tables BEFORE mutating in-memory state, so a WAL
        // failure (catastrophic-failure latch tripped, I/O
        // failure, ...) below can fully restore this process's
        // view. Without restore, `record_ddl` failure leaves the
        // table gone in-memory while durable state still
        // contains it — restart will bring it back, leaving the
        // caller with a DROP that "succeeded" but reappears.
        let pre_schema_for_revert: Option<CompactArc<Schema>>;
        let mut pre_child_schemas: Vec<(String, CompactArc<Schema>)> = Vec::new();
        // Atomically remove schema AND strip FK references AND
        // mark the name as "drop in progress" in
        // `pending_drop_cleanups` under a single schemas write
        // lock. The pending insert under the same lock that
        // makes the schema absent is what closes the
        // CREATE-after-DROP race: a concurrent same-name
        // CREATE TABLE acquires the same schemas write lock
        // for its check-and-insert, sees the pending entry
        // we deposit here, and refuses (otherwise it would
        // see schemas-missing AND pending-empty in the brief
        // window between this remove and any later pending
        // insert in the cleanup branches below).
        //
        // Lock-ordering rule: schemas FIRST, pending SECOND
        // — same order every CREATE / DROP path observes.
        // Holding both across the remove + insert is brief
        // (no FS or WAL I/O inside the critical section).
        {
            let mut schemas = self.schemas.write().unwrap();
            if !schemas.contains_key(&table_name) {
                return Err(Error::TableNotFound(table_name.to_string()));
            }
            // Capture pre-drop state for the revert path.
            pre_schema_for_revert = schemas.get(&table_name).cloned();
            // Snapshot every child schema that might be modified
            // by `strip_fk_references`. Filter to those that
            // actually reference us so we don't snapshot the
            // whole table set.
            for (name, sch) in schemas.iter() {
                if sch
                    .foreign_keys
                    .iter()
                    .any(|fk| fk.referenced_table == table_name)
                {
                    pre_child_schemas.push((name.clone(), sch.clone()));
                }
            }
            // Mark dropping BEFORE removing the schema so
            // any concurrent same-name CREATE that's waiting
            // for the schemas write lock sees the pending
            // entry the moment it acquires.
            self.pending_drop_cleanups.lock().insert(table_name.clone());
            schemas.remove(&table_name);

            // Strip FK constraints from child tables that referenced the dropped table.
            // Done under the same schemas write lock for atomicity.
            let version_stores = self.version_stores.read().unwrap();
            strip_fk_references(&mut schemas, &version_stores, &table_name);
        }

        // Close and remove version store
        let removed_store: Option<Arc<VersionStore>> = {
            let mut stores = self.version_stores.write().unwrap();
            stores.remove(&table_name)
        };
        // Defer `store.close()` until AFTER the WAL succeeds — a
        // closed VersionStore can't serve reads if we have to
        // restore it on revert.

        // WAL FIRST: record the drop before deleting segment files.
        // If crash happens after WAL but before file deletion, WAL replay
        // will re-execute the drop. Orphan files are harmless.
        if let Err(e) = self.record_ddl(name, WALOperationType::DropTable, &[]) {
            // Revert in-memory state in reverse order of removal.
            if let Some(store) = removed_store {
                let mut stores = self.version_stores.write().unwrap();
                stores.insert(table_name.clone(), store);
            }
            let mut schemas = self.schemas.write().unwrap();
            if let Some(prior) = pre_schema_for_revert {
                schemas.insert(table_name.clone(), prior);
            }
            // Clear the "drop in progress" mark we
            // optimistically deposited above. The DROP
            // never reached durability, so a same-name
            // CREATE TABLE that arrives after this revert
            // must be allowed to proceed.
            self.pending_drop_cleanups.lock().remove(&table_name);
            // Restore each child schema in BOTH the schemas
            // catalog and the child's VersionStore. Without
            // restoring the VS schema, later table handles
            // observe a schema that no longer matches the
            // restored catalog (the FK constraint is back in
            // `schemas` but the VS still has it stripped),
            // causing FK enforcement and serialization to
            // disagree. Acquire version_stores read once
            // outside the loop to keep lock ordering stable.
            let stores_for_revert = self.version_stores.read().unwrap();
            for (cname, csch) in pre_child_schemas {
                if let Some(vs) = stores_for_revert.get(cname.as_str()) {
                    *vs.schema_mut() = csch.clone();
                }
                schemas.insert(cname, csch);
            }
            drop(stores_for_revert);
            drop(schemas);
            // Bump schema_epoch on the failure-revert path.
            // The catalog spent the WAL-write window in its
            // dropped state (parent removed, child FKs
            // stripped), so a concurrent reader may have
            // rebuilt `fk_reverse_cache` against THAT view
            // and stamped it with the still-current epoch.
            // Without this bump that stale cache survives
            // the revert and reports the parent as having no
            // referencing FKs.
            self.schema_epoch.fetch_add(1, Ordering::Release);
            return Err(e);
        }

        // WAL succeeded — close the now-orphaned VersionStore.
        if let Some(store) = removed_store {
            store.close();
        }

        // Clear in-memory segment state
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.get(&table_name) {
                mgr.clear();
            }
            mgrs.remove(&table_name);
        }
        // Delete volume files when no live cross-process
        // reader could still hold a stale manifest pointer
        // into them. While `defer_for_live_readers()` is
        // true, the directory and its `vol_NNNN.vol` files
        // stay UNTOUCHED at their original path so a live
        // reader's lazy `ensure_volume` continues to resolve
        // and `read_volume_from_disk` returns valid bytes.
        // `sweep_orphan_table_dirs` reaps the leftover
        // directory on a future checkpoint / open once
        // readers detach.
        //
        // Mark this table in `pending_drop_cleanups`
        // WHENEVER the immediate unlink doesn't run — that
        // is, on `defer=true` AND on Err. The leftover
        // `manifest.bin` is otherwise discoverable by
        // `scan_table_dirs`, so a checkpoint that runs
        // before the next sweep would re-record only the
        // live tables, bump the manifest epoch, and
        // truncate WAL past the `DropTable` record. Without
        // this gate, V1 readers (which don't pin WAL via
        // `min_pinned_reader_lsn`) wouldn't keep the
        // truncate from advancing, and after restart the
        // dropped table would resurface from the leftover
        // manifest.
        // `compute_wal_truncate_floor` refuses every
        // truncation while the set is non-empty;
        // `sweep_orphan_table_dirs` clears entries whose
        // directory is confirmed gone.
        // Bump schema_epoch BEFORE the fallible physical
        // cleanup. The catalog mutations above (schema
        // remove + version_store remove + child FK strip)
        // are durable as of the WAL DropTable record
        // already written, so any cached
        // `find_referencing_fks` result or compiled
        // schema-dependent fast path stamped under the old
        // epoch is now stale regardless of whether the
        // volume-file unlink succeeds. Bumping after the
        // unlink-failure return-path would leave those
        // caches valid against a no-longer-existent table.
        self.schema_epoch.fetch_add(1, Ordering::Release);

        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let defer = self.defer_for_live_readers();
                // The optimistic mark deposited under the
                // schemas write lock above stays in pending
                // when defer / Err keeps the directory on
                // disk. On non-deferred SUCCESS, clear it
                // — the directory is gone, so a same-name
                // CREATE TABLE arriving next is safe.
                if let Err(e) = crate::storage::volume::io::delete_table_volumes_when_safe(
                    &vol_dir,
                    &table_name,
                    defer,
                ) {
                    self.pending_drop_cleanups.lock().insert(table_name.clone());
                    return Err(e);
                }
                if !defer {
                    self.pending_drop_cleanups.lock().remove(&table_name);
                }
            } else {
                // Persistence disabled: there's no on-disk
                // directory to leak; clear the optimistic mark.
                self.pending_drop_cleanups.lock().remove(&table_name);
            }
        } else {
            // No persistence at all: same as above.
            self.pending_drop_cleanups.lock().remove(&table_name);
        }

        Ok(())
    }

    /// Gets a version store for a table
    pub(crate) fn get_version_store(&self, name: &str) -> Result<Arc<VersionStore>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name = to_lowercase_cow(name);

        let stores = self.version_stores.read().unwrap();
        stores
            .get(table_name.as_ref())
            .cloned()
            .ok_or_else(|| Error::TableNotFound(table_name.as_ref().to_string()))
    }

    /// Creates a column in a table
    pub fn create_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: DataType,
        nullable: bool,
    ) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if schema_arc.has_column(column_name) {
                return Err(Error::DuplicateColumn);
            }
        }

        // Update version store schema first (source of truth) — if this fails,
        // engine schema is still consistent
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                let vs_schema = CompactArc::make_mut(&mut *vs_schema_guard);
                let col = crate::core::SchemaColumn::new(
                    vs_schema.columns.len(),
                    column_name,
                    data_type,
                    nullable,
                    false,
                );
                vs_schema.add_column(col)?;
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower, schema);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Creates a column in a table with an optional default expression
    pub fn create_column_with_default(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
        vector_dimensions: u16,
    ) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if schema_arc.has_column(column_name) {
                return Err(Error::DuplicateColumn);
            }
        }

        // Update version store schema first (source of truth) — if this fails,
        // engine schema is still consistent
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                let vs_schema = CompactArc::make_mut(&mut *vs_schema_guard);
                let mut col = crate::core::SchemaColumn::new(
                    vs_schema.columns.len(),
                    column_name,
                    data_type,
                    nullable,
                    false,
                );
                col.default_expr = default_expr;
                col.vector_dimensions = vector_dimensions;
                vs_schema.add_column(col)?;
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower, schema);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        // Stamp the table's segment manager so a no-shm
        // reader's drift check sees this ADD COLUMN even when
        // no new segment is produced. See `propagate_schema_bump`.
        self.propagate_schema_bump(table_name);

        Ok(())
    }

    /// Refresh the engine's schema cache for a table from the version store
    /// This is used after DDL operations that modify the table's schema directly
    pub(crate) fn refresh_schema_cache(&self, table_name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Get the schema from the version store, then release the lock
        // LOCK ORDERING: Release version_stores(R) before acquiring schemas(W)
        // to maintain consistent ordering with column DDL (schemas(W) -> version_stores(R))
        let vs_schema = {
            let stores = self.version_stores.read().unwrap();
            let store = stores
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            store.schema().clone()
        };

        // Update the engine's schema cache
        let mut schemas = self.schemas.write().unwrap();
        schemas.insert(table_name_lower, vs_schema);
        drop(schemas);

        // Bump schema epoch so compiled fast paths detect the change
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Get or create a segment manager for a table.
    fn get_or_create_segment_manager(
        &self,
        table_name: &str,
    ) -> Arc<crate::storage::volume::manifest::SegmentManager> {
        // Fast path: read lock (common case during WAL replay — manager already exists)
        {
            let mgrs = self.segment_managers.read().unwrap();
            if let Some(mgr) = mgrs.get(table_name) {
                return Arc::clone(mgr);
            }
        }
        // Slow path: write lock (first access, creates new manager)
        let mut mgrs = self.segment_managers.write().unwrap();
        mgrs.entry(table_name.to_string())
            .or_insert_with(|| {
                let vol_dir = self
                    .persistence
                    .as_ref()
                    .as_ref()
                    .map(|pm| pm.path().join("volumes"));
                Arc::new(crate::storage::volume::manifest::SegmentManager::new(
                    table_name, vol_dir,
                ))
            })
            .clone()
    }

    /// Clean up stale .dv files from disk left over by previous versions.
    /// In the new design, delete vectors are no longer used. Hot versions
    /// shadow cold versions via skip sets at scan time.
    fn cleanup_stale_dv_files(&self) {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return,
        };

        let vol_dir = pm.path().join("volumes");

        // Collect table names under read lock, then process without holding it
        let table_names: Vec<String> = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.keys().cloned().collect()
        };

        for table_name in &table_names {
            let table_dir = vol_dir.join(table_name);
            if let Ok(entries) = std::fs::read_dir(&table_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("dv") {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
    }

    /// Register a frozen volume into a table's segment manager.
    /// Uses a fresh auto-assigned segment ID.
    fn register_volume(
        &self,
        table_name: &str,
        volume: Arc<crate::storage::volume::writer::FrozenVolume>,
    ) {
        let mgr = self.get_or_create_segment_manager(table_name);
        let seg_id = mgr.manifest_mut().allocate_segment_id();
        self.register_volume_with_id(table_name, volume, seg_id);
    }

    /// Register a frozen volume with a specific segment ID.
    /// Used during startup to restore stable IDs from volume filenames,
    /// ensuring .dv files match their segments across restarts. Recovery
    /// callers leave `visible_at_lsn` at 0 (= "visible to all readers"):
    /// the segment existed before this engine opened, so any reader
    /// attaching now should see it.
    fn register_volume_with_id(
        &self,
        table_name: &str,
        volume: Arc<crate::storage::volume::writer::FrozenVolume>,
        seg_id: u64,
    ) {
        self.register_volume_with_id_and_seal_seq(table_name, volume, seg_id, 0, 0);
    }

    fn register_volume_with_id_and_seal_seq(
        &self,
        table_name: &str,
        volume: Arc<crate::storage::volume::writer::FrozenVolume>,
        seg_id: u64,
        seal_seq: u64,
        visible_at_lsn: u64,
    ) {
        use crate::storage::volume::manifest::SegmentMeta;
        let mgr = self.get_or_create_segment_manager(table_name);
        let min_id = volume.meta.row_ids.first().copied().unwrap_or(0);
        let max_id = volume.meta.row_ids.last().copied().unwrap_or(0);
        let row_count = volume.meta.row_count;
        // None = identity mapping (volume was just built from current schema).
        // Load paths that may have schema mismatch pass Some(schema).
        mgr.register_segment(
            seg_id,
            volume,
            SegmentMeta {
                segment_id: seg_id,
                file_path: std::path::PathBuf::new(),
                row_count,
                min_row_id: min_id,
                max_row_id: max_id,
                creation_lsn: 0,
                seal_seq,
                schema_version: self.schema_epoch.load(Ordering::Acquire),
                // SWMR v2 Phase F: WAL LSN at which this segment
                // becomes visible. Capped read-only readers (attach
                // LSN P) hide segments with `visible_at_lsn > P`. A
                // value of 0 means "visible to all readers" — used
                // by recovery paths where the segment pre-existed
                // this engine open. Runtime seal/compact callers
                // pass the writer's current WAL LSN at registration
                // time so a reader sampling `visible_commit_lsn = P`
                // before a concurrent checkpoint won't load the
                // post-attach segments.
                visible_at_lsn,
            },
            None,
        );
    }

    /// Load manifests from the volumes/ directory for checkpoint-to-volume recovery.
    ///
    /// This is called during startup when no snapshot files exist but volumes/ has
    /// manifest.bin files from a previous checkpoint cycle. Loads manifest metadata
    /// (segment list, tombstones, checkpoint_lsn) into segment managers so that:
    /// - WAL replay can check `is_row_id_in_volume_range()` for tombstone creation
    /// - The minimum checkpoint_lsn determines where WAL replay starts
    ///
    /// Returns the minimum checkpoint_lsn across all loaded manifests (0 if none found).
    /// Actual .vol files are loaded later by `load_standalone_volumes()` after WAL replay
    /// creates the required schemas and version stores.
    fn load_manifests_from_volumes(&self) -> u64 {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return 0,
        };

        let vol_dir = pm.path().join("volumes");
        if !vol_dir.exists() {
            return 0;
        }

        let read_only_open = self.is_read_only_mode();
        let max_attempts = if read_only_open { 200 } else { 1 };

        for attempt in 0..max_attempts {
            let entries = match std::fs::read_dir(&vol_dir) {
                Ok(e) => e,
                Err(_) => return 0,
            };

            let mut min_checkpoint_lsn: u64 = u64::MAX;
            let mut observed_checkpoint_min: u64 = u64::MAX;
            let mut observed_checkpoint_max: u64 = 0;
            let mut any_loaded = false;
            let mut staged: Vec<(
                String,
                Arc<crate::storage::volume::manifest::SegmentManager>,
            )> = Vec::new();

            for entry in entries.flatten() {
                if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    continue;
                }

                let table_name = entry.file_name().to_string_lossy().to_lowercase();

                // Try to load manifest from this table directory.
                match crate::storage::volume::manifest::SegmentManager::load_from_disk(
                    &table_name,
                    &vol_dir,
                ) {
                    Ok(Some(mgr)) => {
                        let manifest_lsn = mgr.manifest().checkpoint_lsn;
                        observed_checkpoint_min = observed_checkpoint_min.min(manifest_lsn);
                        observed_checkpoint_max = observed_checkpoint_max.max(manifest_lsn);

                        // Capped read-only attach: hide segments the
                        // writer published AFTER our attach LSN. The shm
                        // sample → manifest load window can race a writer
                        // checkpoint; without this filter, cold rows from
                        // post-attach segments would be visible while WAL
                        // replay stays capped below them.
                        let cap = self
                            .replay_cap_lsn
                            .load(std::sync::atomic::Ordering::Acquire);
                        if cap != u64::MAX {
                            mgr.retain_segments_visible_at_or_below(cap);
                        }
                        // Effective replay floor:
                        //   - Uncapped (writable open): manifest's
                        //     `checkpoint_lsn` is authoritative — all rows
                        //     up to it are durable in cold.
                        //   - Capped (read-only attach): if the writer
                        //     checkpointed past our cap (`checkpoint_lsn
                        //     > cap`), `checkpoint_lsn` would push
                        //     `replay_two_phase_capped(floor, cap)` to an
                        //     empty range, silently dropping rows
                        //     committed in (max_kept_visible, cap]. Use
                        //     the maximum `visible_at_lsn` of the kept
                        //     segments instead — that's the LSN at which
                        //     OUR (filtered) cold view is complete.
                        //     Replay then covers (kept_max, cap], which
                        //     stays intact thanks to the WAL pin
                        //     `pre_acquire_swmr_for_read_only_path`
                        //     publishes BEFORE the shm sample.
                        if cap != u64::MAX && manifest_lsn > cap {
                            // Capped read-only attach: the writer
                            // checkpointed past our cap. Use the MAX
                            // `visible_at_lsn` of kept segments — that's
                            // the LSN at which OUR (filtered) cold view
                            // is complete. A value of 0 means EVERY
                            // segment for this table was filtered out;
                            // the table needs full WAL replay from the
                            // beginning. That dominates any positive
                            // floor another table contributes — using
                            // their higher floor would skip WAL ranges
                            // this table requires.
                            let kept_max = mgr
                                .manifest()
                                .segments
                                .iter()
                                .map(|s| s.visible_at_lsn)
                                .max()
                                .unwrap_or(0);
                            if kept_max == 0 {
                                min_checkpoint_lsn = 0;
                            } else if kept_max < min_checkpoint_lsn {
                                min_checkpoint_lsn = kept_max;
                            }
                        } else if manifest_lsn > 0 && manifest_lsn < min_checkpoint_lsn {
                            // Uncapped / writable open: keep the
                            // historical "skip 0" semantic to avoid
                            // regressing writable recovery — a
                            // manifest at `checkpoint_lsn = 0` here
                            // means "no prior checkpoint info" rather
                            // than "must replay from 0".
                            min_checkpoint_lsn = manifest_lsn;
                        }
                        any_loaded = true;
                        staged.push((table_name, Arc::new(mgr)));
                    }
                    Ok(None) => {
                        // No manifest.bin in this directory, skip.
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load manifest for {}: {}", table_name, e);
                    }
                }
            }

            if !any_loaded {
                // No manifests found, remove checkpoint.meta if present
                // to ensure full WAL replay.
                let checkpoint_path = pm.path().join("wal").join("checkpoint.meta");
                let _ = std::fs::remove_file(checkpoint_path);
                return 0;
            }

            if read_only_open
                && observed_checkpoint_min != u64::MAX
                && observed_checkpoint_min != observed_checkpoint_max
            {
                if attempt + 1 < max_attempts {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                }
                eprintln!(
                    "Warning: read-only open observed mixed checkpoint_lsn range {}..{}; \
                     falling back to WAL replay from 0",
                    observed_checkpoint_min, observed_checkpoint_max
                );
                let checkpoint_path = pm.path().join("wal").join("checkpoint.meta");
                let _ = std::fs::remove_file(checkpoint_path);
                return 0;
            }

            // Store the segment managers so WAL replay can use
            // is_row_id_in_volume_range() for tombstone creation.
            let mut mgrs = self.segment_managers.write().unwrap();
            for (table_name, mgr) in staged {
                mgrs.insert(table_name, mgr);
            }

            return if min_checkpoint_lsn == u64::MAX {
                0
            } else {
                min_checkpoint_lsn
            };
        }

        0
    }

    /// Load standalone volumes from the volumes/ directory.
    ///
    /// These are created by seal_hot_buffers and compact_volumes during runtime.
    /// They supplement the snapshot-adjacent .vol files loaded during load_snapshots.
    fn load_standalone_volumes(&self) {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return,
        };

        let vol_dir = pm.path().join("volumes");
        if !vol_dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(&vol_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }

            let table_name = entry.file_name().to_string_lossy().to_lowercase();

            {
                let schemas = self.schemas.read().unwrap();
                if !schemas.contains_key(&table_name) {
                    continue;
                }
            }

            let paths = crate::storage::volume::io::list_volumes(&vol_dir, &table_name);
            if paths.is_empty() {
                continue;
            }

            let mut standalone: Vec<(u64, Arc<crate::storage::volume::writer::FrozenVolume>)> =
                Vec::new();

            // Get column renames from manifest (if any) to merge into volumes.
            let mgr_for_renames = self.get_or_create_segment_manager(&table_name);
            let renames: Vec<(String, String)> = {
                let manifest = mgr_for_renames.manifest();
                manifest
                    .column_renames
                    .iter()
                    .map(|(old, new)| (old.to_string(), new.to_string()))
                    .collect()
            };

            for path in paths {
                let volume_id = parse_volume_id(&path);
                let volume = match crate::storage::volume::io::read_volume_from_disk(&path) {
                    Ok(mut volume) => {
                        // Merge renames into column_name_map BEFORE Arc wrapping.
                        // No RwLock needed — volume is still exclusively owned.
                        for (old_name, new_name) in &renames {
                            volume.merge_column_rename(new_name, old_name);
                        }
                        Arc::new(volume)
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to read volume {:?}: {}. Skipping file.",
                            path, e
                        );
                        continue;
                    }
                };

                let stable_id = volume_id.unwrap_or(0);
                standalone.push((stable_id, volume));
            }

            if standalone.is_empty() {
                continue;
            }

            // Load volumes that are listed in the manifest. Orphan files
            // (not in manifest) are cleaned up.
            let mgr = self.get_or_create_segment_manager(&table_name);
            for (stable_id, vol) in standalone {
                if stable_id > 0 {
                    if mgr.has_segment(stable_id) {
                        continue;
                    }
                    // Only load if manifest lists this segment_id.
                    if !mgr.load_volume_for_existing_segment(stable_id, Arc::clone(&vol)) {
                        // Orphan — not in manifest. Skip (file cleanup handled elsewhere).
                    }
                }
                // Volumes without parseable IDs are orphans — skip.
            }
            // Recompute visibility bitmaps after all volumes for this table are loaded.
            mgr.recompute_visibility();
        }
    }

    /// Load standalone volumes without requiring schemas to exist.
    ///
    /// Used during the new volume-only recovery path where volumes are loaded
    /// BEFORE WAL replay (which creates schemas). Volumes carry their own schema
    /// in the file format, so no external schema lookup is needed.
    fn load_standalone_volumes_no_schema_check(&self) {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return,
        };

        let vol_dir = pm.path().join("volumes");
        if !vol_dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(&vol_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        // Read-only opens must not unlink any .vol file: orphans may
        // belong to a sibling writer's in-flight checkpoint, and a
        // capped read-only attach legitimately filters out segments
        // the writer published after our attach LSN (their .vol
        // files are NOT orphans from the writer's perspective).
        let read_only = self.is_read_only_mode();
        // Writable startup ALSO defers orphan-volume unlinks
        // when live read-only processes are still attached.
        // The live DROP / TRUNCATE / compaction paths leave
        // unreferenced .vol files on disk specifically so a
        // reader's stale manifest can keep lazy-loading via
        // `SegmentManager::ensure_volume`. Unconditionally
        // unlinking here on writer restart would unlink the
        // exact files the surviving reader needs (the
        // writer's new manifest doesn't reference them, but
        // the reader's pre-restart manifest does). Same
        // fail-closed gating as the live cleanup paths.
        let defer = self.defer_for_live_readers();

        for entry in entries.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }

            let table_name = entry.file_name().to_string_lossy().to_lowercase();

            let paths = crate::storage::volume::io::list_volumes(&vol_dir, &table_name);
            if paths.is_empty() {
                continue;
            }

            let mgr = self.get_or_create_segment_manager(&table_name);
            for path in paths {
                let volume_id = parse_volume_id(&path);
                let stable_id = volume_id.unwrap_or(0);

                // Check manifest membership BEFORE expensive disk I/O.
                // Orphan .vol files (from compaction or crash) are cleaned up
                // without being deserialized.
                if stable_id == 0 {
                    if !read_only && !defer {
                        let _ = std::fs::remove_file(&path);
                    }
                    continue;
                }
                if mgr.has_segment(stable_id) {
                    continue;
                }
                if !mgr.manifest_has_segment(stable_id) {
                    if !read_only && !defer {
                        let _ = std::fs::remove_file(&path);
                    }
                    continue;
                }

                // Segment is in manifest but not yet loaded — read from disk.
                let renames: Vec<(String, String)> = {
                    let manifest = mgr.manifest();
                    manifest
                        .column_renames
                        .iter()
                        .map(|(old, new)| (old.to_string(), new.to_string()))
                        .collect()
                };
                let volume = match crate::storage::volume::io::read_volume_from_disk(&path) {
                    Ok(mut volume) => {
                        for (old_name, new_name) in &renames {
                            volume.merge_column_rename(new_name, old_name);
                        }
                        Arc::new(volume)
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to read volume {:?}: {}. Skipping file.",
                            path, e
                        );
                        continue;
                    }
                };
                mgr.load_volume_for_existing_segment(stable_id, volume);
            }
            // Recompute visibility bitmaps after all volumes for this table are loaded.
            mgr.recompute_visibility();
        }
    }

    /// Sync auto-increment counters from segment data.
    ///
    /// After WAL replay, set `schema_epoch` to the highest segment
    /// `schema_version` observed across all loaded manifests. WAL replay
    /// loads schemas via direct map insertion (recovery path) rather
    /// than the public DDL methods that bump `schema_epoch`, so without
    /// this sync the counter would stay at 0 after recovering N tables.
    /// The v2 SWMR drift check requires this baseline to distinguish
    /// "segment from a writer DDL the reader has WAL-replayed" (safe)
    /// from "segment from writer DDL since this reader opened" (must
    /// fail with `Error::SchemaChanged`).
    fn sync_schema_epoch_from_segments(&self) {
        let mgrs = self.segment_managers.read().unwrap();
        if mgrs.is_empty() {
            return;
        }
        let max_seen: u64 = mgrs
            .values()
            .flat_map(|mgr| {
                mgr.manifest()
                    .segments
                    .iter()
                    .map(|s| s.schema_version)
                    .collect::<Vec<u64>>()
            })
            .max()
            .unwrap_or(0);
        // fetch_max so we never go BACKWARDS (an existing engine that
        // already did DDL events past max_seen keeps its higher value).
        self.schema_epoch
            .fetch_max(max_seen, std::sync::atomic::Ordering::Release);
    }

    /// After WAL replay, ensure auto-increment counters account for
    /// the max row_id in segments (which may be higher than hot buffer).
    /// Normal secondary indexes are NOT populated from cold data.
    /// HNSW is the one cold index that needs explicit graph rebuild.
    fn sync_auto_increment_from_segments(&self) {
        // Collect segment data under segment_managers lock, then drop it
        // before acquiring version_stores lock to avoid multi-lock deadlock.
        let segment_data: Vec<(String, i64)> = {
            let mgrs = self.segment_managers.read().unwrap();
            if mgrs.is_empty() {
                return;
            }
            mgrs.iter()
                .filter_map(|(table_name, mgr)| {
                    let segments = mgr.get_segments_ordered_meta();
                    let mut max_vol_row_id: i64 = 0;
                    for vol in &segments {
                        // Row IDs are sorted (from B-tree iteration during seal).
                        // Use last() for O(1) instead of iterating all row_ids.
                        if let Some(&last_id) = vol.meta.row_ids.last() {
                            if last_id > max_vol_row_id {
                                max_vol_row_id = last_id;
                            }
                        }
                    }
                    if max_vol_row_id > 0 {
                        Some((table_name.clone(), max_vol_row_id))
                    } else {
                        None
                    }
                })
                .collect()
        };

        if segment_data.is_empty() {
            return;
        }

        let stores = self.version_stores.read().unwrap();
        for (table_name, max_vol_row_id) in &segment_data {
            if let Some(store) = stores.get(table_name) {
                store.set_auto_increment_counter(*max_vol_row_id);
            }
        }
    }

    /// Drops a column from a table
    pub fn drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if let Some((_, col)) = schema_arc.find_column(column_name) {
                if col.primary_key {
                    return Err(Error::CannotDropPrimaryKey);
                }
            } else {
                return Err(Error::ColumnNotFound(column_name.to_string()));
            }
        }

        // Update version store schema first (source of truth)
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                CompactArc::make_mut(&mut *vs_schema_guard).remove_column(column_name)?;
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower, schema);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        // Record the drop in the segment manifest so cold volume mappings
        // mask stale data. Same as the live DDL path in ddl.rs. Without this,
        // crash recovery (WAL replay) loses dropped_columns metadata.
        self.propagate_column_drop(table_name, column_name);

        Ok(())
    }

    /// Renames a column in a table
    pub fn rename_column(&self, table_name: &str, old_name: &str, new_name: &str) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if !schema_arc.has_column(old_name) {
                return Err(Error::ColumnNotFound(old_name.to_string()));
            }
            if schema_arc.has_column(new_name) {
                return Err(Error::DuplicateColumn);
            }
        }

        // Update version store schema first (source of truth)
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                CompactArc::make_mut(&mut *vs_schema_guard).rename_column(old_name, new_name)?;
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower.clone(), schema);
            }
        }

        // Propagate rename to cold volumes (persists in manifest) and recompute mappings
        {
            let schema = self.schemas.read().unwrap().get(&table_name_lower).cloned();
            if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
                mgr.record_column_rename(old_name, new_name);
                if let Some(ref s) = schema {
                    mgr.invalidate_mappings(s);
                }
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Record a column drop so old cold volumes don't leak stale data.
    pub(crate) fn propagate_column_drop(&self, table_name: &str, col_name: &str) {
        let table_name_lower = table_name.to_lowercase();
        let schema = self.schemas.read().unwrap().get(&table_name_lower).cloned();
        let current_epoch = self.schema_epoch.load(Ordering::Acquire);
        if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
            mgr.record_column_drop(col_name, current_epoch);
            mgr.record_table_schema_version(current_epoch);
            if let Some(ref s) = schema {
                mgr.invalidate_mappings(s);
            }
        }
    }

    /// Record a column rename and propagate alias to all cold volumes.
    /// Persists in the manifest so aliases survive restart.
    pub(crate) fn propagate_column_alias(&self, table_name: &str, new_name: &str, old_name: &str) {
        let table_name_lower = table_name.to_lowercase();
        let schema = self.schemas.read().unwrap().get(&table_name_lower).cloned();
        let current_epoch = self.schema_epoch.load(Ordering::Acquire);
        if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
            mgr.record_column_rename(old_name, new_name);
            mgr.record_table_schema_version(current_epoch);
            if let Some(ref s) = schema {
                mgr.invalidate_mappings(s);
            }
        }
    }

    /// Stamp the table's segment manager with the current
    /// `schema_epoch`. Called by ADD/MODIFY COLUMN paths that
    /// don't otherwise touch the segment manager. Without this,
    /// a no-shm reader's `peek_schema_drift` would not detect
    /// these DDLs (they don't produce a new segment, don't
    /// touch dropped_columns/column_renames, and don't reach
    /// `propagate_column_drop`/`propagate_column_alias`).
    pub(crate) fn propagate_schema_bump(&self, table_name: &str) {
        let table_name_lower = table_name.to_lowercase();
        let current_epoch = self.schema_epoch.load(Ordering::Acquire);
        if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
            mgr.record_table_schema_version(current_epoch);
        }
    }

    /// Modifies a column's type and nullable property in a table
    pub fn modify_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: DataType,
        nullable: bool,
    ) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if !schema_arc.has_column(column_name) {
                return Err(Error::ColumnNotFound(column_name.to_string()));
            }
        }

        // Update version store schema first (source of truth)
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                CompactArc::make_mut(&mut *vs_schema_guard).modify_column(
                    column_name,
                    Some(data_type),
                    Some(nullable),
                )?;
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower, schema);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        // Stamp the table's segment manager so a no-shm
        // reader's drift check sees this MODIFY COLUMN even when
        // no new segment is produced.
        self.propagate_schema_bump(table_name);

        Ok(())
    }

    /// Modifies a column's type, nullable, and vector dimensions
    /// Used by WAL replay to restore ALTER TABLE MODIFY COLUMN with full dimension info
    pub(crate) fn modify_column_with_dimensions(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: DataType,
        nullable: bool,
        vector_dimensions: u16,
    ) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Validate under read lock first
        {
            let schemas = self.schemas.read().unwrap();
            let schema_arc = schemas
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            if !schema_arc.has_column(column_name) {
                return Err(Error::ColumnNotFound(column_name.to_string()));
            }
        }

        // Update version store schema first (source of truth)
        {
            let stores = self.version_stores.read().unwrap();
            if let Some(store) = stores.get(&table_name_lower) {
                let mut vs_schema_guard = store.schema_mut();
                let vs_schema = CompactArc::make_mut(&mut *vs_schema_guard);
                vs_schema.modify_column(column_name, Some(data_type), Some(nullable))?;
                // Set vector_dimensions on the modified column
                if let Some(idx) = vs_schema.get_column_index(column_name) {
                    vs_schema.columns[idx].vector_dimensions = vector_dimensions;
                }
            }
        }

        // Sync engine schema cache from version store
        {
            let vs_schema = {
                let stores = self.version_stores.read().unwrap();
                stores
                    .get(&table_name_lower)
                    .map(|store| store.schema().clone())
            };
            if let Some(schema) = vs_schema {
                let mut schemas = self.schemas.write().unwrap();
                schemas.insert(table_name_lower, schema);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        // Stamp the table's segment manager so a no-shm
        // reader's drift check sees this MODIFY COLUMN even when
        // no new segment is produced.
        self.propagate_schema_bump(table_name);

        Ok(())
    }

    /// Renames a table
    /// Wholesale-restore a table's schema from a pre-mutation
    /// snapshot. Used by the executor's ALTER TABLE column-revert
    /// paths when `record_alter_table_*` failed (typically because
    /// the catastrophic-failure latch flipped between the begin-time
    /// check and the WAL write). Replaces the version_store's
    /// current schema with `pre_schema` verbatim, preserving every
    /// field — column ids, positions, primary-key / auto-increment /
    /// check / default metadata — that approximating the revert by
    /// running an inverse op (drop_column / re-create_column /
    /// rename back / modify back) would silently change. Bypasses
    /// the failed-latch gate (the latch is set BY the failure
    /// we're reverting from). After replacing, refreshes the
    /// engine's schema cache so reads see the restored shape.
    pub(crate) fn restore_table_schema(&self, table_name: &str, pre_schema: Schema) -> Result<()> {
        let table_name_lower = table_name.to_lowercase();
        {
            let stores = self.version_stores.read().unwrap();
            let store = stores
                .get(&table_name_lower)
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            let mut vs_schema_guard = store.schema_mut();
            *vs_schema_guard = crate::common::CompactArc::new(pre_schema);
        }
        // Refresh engine's schema cache so subsequent reads see
        // the restored schema. Don't go through the public
        // refresh_schema_cache (which checks `is_open` — fine, but
        // we'd also want to bump schema_epoch which it does).
        self.refresh_schema_cache(&table_name_lower)
    }

    /// Revert a rename done earlier in this same call site, when
    /// `record_alter_table_rename` failed because the catastrophic-
    /// failure latch flipped between the trait fn's check and the
    /// WAL write. Bypasses both `ensure_writable` (the rename
    /// already proved the engine was writable when it ran) and the
    /// failed-latch check (the latch is set BY the failure we're
    /// reverting from, and refusing here would leave the in-memory
    /// rename diverged from on-disk state with no path to undo).
    /// Same in-memory + on-disk work as `rename_table` but without
    /// the gating; intended ONLY for the executor's ALTER TABLE
    /// RENAME revert path.
    pub(crate) fn rename_table_revert(&self, current_name: &str, target_name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let old_name_lower = current_name.to_lowercase();
        let new_name_lower = target_name.to_lowercase();
        self.do_rename_table_unchecked(&old_name_lower, &new_name_lower, target_name)
    }

    /// Body of the rename: in-memory schema / version-store /
    /// segment-manager updates plus on-disk volume directory rename
    /// with revert-on-disk-failure. Extracted so the gated public
    /// `rename_table`, the gated trait `Engine::rename_table`, and
    /// the unchecked `rename_table_revert` all share the same
    /// logic.
    fn do_rename_table_unchecked(
        &self,
        old_name_lower: &str,
        new_name_lower: &str,
        new_name_display: &str,
    ) -> Result<()> {
        // Atomically check-and-rename under write lock to prevent TOCTOU race
        {
            let mut schemas = self.schemas.write().unwrap();
            if !schemas.contains_key(old_name_lower) {
                return Err(Error::TableNotFound(old_name_lower.to_string()));
            }
            if schemas.contains_key(new_name_lower) {
                return Err(Error::TableAlreadyExists(new_name_lower.to_string()));
            }
            if let Some(mut schema_arc) = schemas.remove(old_name_lower) {
                let schema = CompactArc::make_mut(&mut schema_arc);
                schema.table_name = new_name_display.to_string();
                schema.table_name_lower = new_name_lower.to_string();
                schemas.insert(new_name_lower.to_string(), schema_arc);
            }
        }

        // Update version_stores map
        {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(old_name_lower) {
                {
                    let mut vs_schema_guard = store.schema_mut();
                    let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                    schema.table_name = new_name_display.to_string();
                    schema.table_name_lower = new_name_lower.to_string();
                }
                stores.insert(new_name_lower.to_string(), store);
            }
        }

        // Move segment manager to new name and update its manifest table_name
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.remove(old_name_lower) {
                mgr.manifest_mut().table_name = crate::common::SmartString::from(new_name_lower);
                mgrs.insert(new_name_lower.to_string(), mgr);
            }
        }
        // Rename on-disk volume directory so it survives restart
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let old_dir = vol_dir.join(old_name_lower);
                let new_dir = vol_dir.join(new_name_lower);
                if old_dir.exists() {
                    if let Err(e) = std::fs::rename(&old_dir, &new_dir) {
                        // Revert in-memory renames on disk failure
                        let mut mgrs = self.segment_managers.write().unwrap();
                        if let Some(mgr) = mgrs.remove(new_name_lower) {
                            mgr.manifest_mut().table_name =
                                crate::common::SmartString::from(old_name_lower);
                            mgrs.insert(old_name_lower.to_string(), mgr);
                        }
                        drop(mgrs);
                        let mut stores = self.version_stores.write().unwrap();
                        if let Some(store) = stores.remove(new_name_lower) {
                            {
                                let mut vs_schema_guard = store.schema_mut();
                                let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                                schema.table_name = old_name_lower.to_string();
                                schema.table_name_lower = old_name_lower.to_string();
                            }
                            stores.insert(old_name_lower.to_string(), store);
                        }
                        drop(stores);
                        let mut schemas = self.schemas.write().unwrap();
                        if let Some(mut schema_arc) = schemas.remove(new_name_lower) {
                            let schema = CompactArc::make_mut(&mut schema_arc);
                            schema.table_name = old_name_lower.to_string();
                            schema.table_name_lower = old_name_lower.to_string();
                            schemas.insert(old_name_lower.to_string(), schema_arc);
                        }
                        return Err(Error::internal(format!(
                            "failed to rename volume directory: {}",
                            e
                        )));
                    }
                }
                // Move legacy snapshots/<old>/tombstones.dat to
                // snapshots/<new>/tombstones.dat. The original
                // pub `rename_table` does this; the unchecked
                // helper must match so a revert doesn't leave
                // tombstones under the failed name. Best-effort —
                // missing source is fine (no legacy tombstones).
                let snap_dir = pm.path().join("snapshots");
                let old_ts = snap_dir.join(old_name_lower).join("tombstones.dat");
                let new_ts_dir = snap_dir.join(new_name_lower);
                let new_ts = new_ts_dir.join("tombstones.dat");
                if old_ts.exists() {
                    let _ = std::fs::create_dir_all(&new_ts_dir);
                    let _ = std::fs::rename(&old_ts, &new_ts);
                }
            }
        }

        Ok(())
    }

    pub fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let old_name_lower = old_name.to_lowercase();
        let new_name_lower = new_name.to_lowercase();

        // Atomically check-and-rename under write lock to prevent TOCTOU race
        {
            let mut schemas = self.schemas.write().unwrap();
            if !schemas.contains_key(&old_name_lower) {
                return Err(Error::TableNotFound(old_name_lower.to_string()));
            }
            if schemas.contains_key(&new_name_lower) {
                return Err(Error::TableAlreadyExists(new_name_lower.to_string()));
            }
            if let Some(mut schema_arc) = schemas.remove(&old_name_lower) {
                let schema = CompactArc::make_mut(&mut schema_arc);
                schema.table_name = new_name.to_string();
                schema.table_name_lower = new_name_lower.clone();
                schemas.insert(new_name_lower.clone(), schema_arc);
            }
        }

        // Update version_stores map
        {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(&old_name_lower) {
                // Update the schema's table name within the store
                {
                    let mut vs_schema_guard = store.schema_mut();
                    let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                    schema.table_name = new_name.to_string();
                    schema.table_name_lower = new_name_lower.clone();
                }
                stores.insert(new_name_lower.clone(), store);
            }
        }

        // Move segment manager to new name and update its manifest table_name
        // (persist() uses manifest.table_name for the on-disk directory)
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.remove(&old_name_lower) {
                mgr.manifest_mut().table_name =
                    crate::common::SmartString::from(new_name_lower.as_str());
                mgrs.insert(new_name_lower.clone(), mgr);
            }
        }
        // Rename on-disk volume directory and tombstones so they survive restart
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let old_dir = vol_dir.join(&old_name_lower);
                let new_dir = vol_dir.join(&new_name_lower);
                if old_dir.exists() {
                    if let Err(e) = std::fs::rename(&old_dir, &new_dir) {
                        // Revert in-memory segment manager rename on disk failure
                        let mut mgrs = self.segment_managers.write().unwrap();
                        if let Some(mgr) = mgrs.remove(&new_name_lower) {
                            mgr.manifest_mut().table_name =
                                crate::common::SmartString::from(old_name_lower.as_str());
                            mgrs.insert(old_name_lower.clone(), mgr);
                        }
                        drop(mgrs);
                        // Revert version stores
                        let mut stores = self.version_stores.write().unwrap();
                        if let Some(store) = stores.remove(&new_name_lower) {
                            {
                                let mut vs_schema_guard = store.schema_mut();
                                let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                                schema.table_name = old_name.to_string();
                                schema.table_name_lower = old_name_lower.clone();
                            }
                            stores.insert(old_name_lower.clone(), store);
                        }
                        drop(stores);
                        // Revert schemas
                        let mut schemas = self.schemas.write().unwrap();
                        if let Some(mut schema_arc) = schemas.remove(&new_name_lower) {
                            let schema = CompactArc::make_mut(&mut schema_arc);
                            schema.table_name = old_name.to_string();
                            schema.table_name_lower = old_name_lower.clone();
                            schemas.insert(old_name_lower, schema_arc);
                        }
                        drop(schemas);
                        return Err(Error::Internal {
                            message: format!("Failed to rename volume directory: {}", e),
                        });
                    }
                }
                let snap_dir = pm.path().join("snapshots");
                let old_ts = snap_dir.join(&old_name_lower).join("tombstones.dat");
                let new_ts_dir = snap_dir.join(&new_name_lower);
                let new_ts = new_ts_dir.join("tombstones.dat");
                if old_ts.exists() {
                    let _ = std::fs::create_dir_all(&new_ts_dir);
                    let _ = std::fs::rename(&old_ts, &new_ts);
                }
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Validates a schema
    fn validate_schema(&self, schema: &Schema) -> Result<()> {
        if schema.table_name.is_empty() {
            return Err(Error::internal("schema missing table name"));
        }

        // Check for duplicate column names
        let mut seen_names = FxHashSet::default();
        for col in &schema.columns {
            if col.name.is_empty() {
                return Err(Error::internal("column name cannot be empty"));
            }

            if col.primary_key && col.data_type != DataType::Integer {
                return Err(Error::internal(format!(
                    "primary key column {} must be of type INTEGER",
                    col.name
                )));
            }

            if !seen_names.insert(col.name_lower.clone()) {
                return Err(Error::DuplicateColumn);
            }
        }

        Ok(())
    }

    /// Creates an engine operations wrapper for a transaction
    fn create_engine_operations(&self) -> Arc<dyn TransactionEngineOperations> {
        Arc::new(EngineOperations::new(self))
    }

    // --- View Management Methods ---

    /// Create a new view
    pub fn create_view(&self, name: &str, query: String, if_not_exists: bool) -> Result<()> {
        use crate::storage::mvcc::wal_manager::WALOperationType;

        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Hold SH on `transactional_ddl_fence` across the
        // mutation-to-WAL window — same rationale as
        // `create_table`. Without this guard a checkpoint
        // that landed mid-window could snapshot the
        // transient view in `rerecord_ddl_to_wal` and emit a
        // CreateView re-record that survives recovery even
        // if the original `record_ddl` later fails and the
        // rollback removes the view from memory.
        let _ddl_fence_guard = self.transactional_ddl_fence.read();

        let name_lower = name.to_lowercase();

        // Check if a table with the same name exists (acquire schemas before views
        // to maintain consistent lock ordering and prevent deadlock)
        {
            let schemas = self.schemas.read().unwrap();
            if schemas.contains_key(&name_lower) {
                return Err(Error::internal(format!(
                    "cannot create view '{}': a table with the same name exists",
                    name
                )));
            }
        }

        let mut views = self.views.write().unwrap();

        // Check if view already exists
        if views.contains_key(&name_lower) {
            if if_not_exists {
                return Ok(());
            }
            return Err(Error::ViewAlreadyExists(name.to_string()));
        }

        // Create the view definition wrapped in Arc for cheap cloning
        let view_def = Arc::new(ViewDefinition::new(name, query));
        views.insert(name_lower.clone(), Arc::clone(&view_def));

        // Release the lock before recording to WAL
        drop(views);

        // Record to WAL for persistence. Roll back the
        // in-memory insert if `record_ddl` fails (catastrophic-
        // failure latch tripped or WAL I/O failed). Without
        // rollback the view stays visible in this process while
        // restart's WAL recovery would not replay it — a CREATE
        // VIEW that succeeded then "vanishes" on next restart.
        let data = view_def.serialize();
        if let Err(e) = self.record_ddl(&name_lower, WALOperationType::CreateView, &data) {
            self.views.write().unwrap().remove(&name_lower);
            return Err(e);
        }

        Ok(())
    }

    /// Drop a view
    pub fn drop_view(&self, name: &str, if_exists: bool) -> Result<()> {
        use crate::storage::mvcc::wal_manager::WALOperationType;

        self.ensure_writable()?;
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Hold SH on `transactional_ddl_fence` across the
        // mutation-to-WAL window — same rationale as
        // `drop_table_internal`. Without this guard a
        // checkpoint that landed mid-window could omit the
        // view from `rerecord_ddl_to_wal` and truncate WAL
        // before the DropView record was durable, so
        // recovery would never see the drop and the next
        // reopen would resurface the dropped view from
        // checkpoint state.
        let _ddl_fence_guard = self.transactional_ddl_fence.read();

        let name_lower = name.to_lowercase();
        let mut views = self.views.write().unwrap();

        // Snapshot the removed view's `Arc<ViewDefinition>` so
        // we can reinsert it on WAL failure. Without restore, a
        // `record_ddl` failure leaves the view gone in memory
        // while durable state still contains it — restart
        // brings it back, leaving the caller with a DROP that
        // "succeeded" then reappears.
        let removed_view = match views.remove(&name_lower) {
            Some(v) => v,
            None => {
                if if_exists {
                    return Ok(());
                }
                return Err(Error::ViewNotFound(name.to_string()));
            }
        };

        // Release the lock before recording to WAL
        drop(views);

        // Record to WAL for persistence (just the view name)
        if let Err(e) = self.record_ddl(&name_lower, WALOperationType::DropView, name.as_bytes()) {
            self.views.write().unwrap().insert(name_lower, removed_view);
            return Err(e);
        }

        Ok(())
    }

    /// Check if a view exists
    pub fn view_exists(&self, name: &str) -> Result<bool> {
        self.view_exists_lowercase(&name.to_lowercase())
    }

    /// Check if a view exists (assumes name is already lowercase)
    /// Use this when you already have a lowercase name to avoid allocation
    #[inline]
    pub fn view_exists_lowercase(&self, name_lower: &str) -> Result<bool> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let views = self.views.read().unwrap();
        Ok(views.contains_key(name_lower))
    }

    /// Get a view definition
    pub fn get_view(&self, name: &str) -> Result<Option<Arc<ViewDefinition>>> {
        self.get_view_lowercase(&name.to_lowercase())
    }

    /// Get a view definition (assumes name is already lowercase)
    /// Use this when you already have a lowercase name to avoid allocation.
    /// Returns Arc clone (cheap pointer copy, no data clone).
    #[inline]
    pub fn get_view_lowercase(&self, name_lower: &str) -> Result<Option<Arc<ViewDefinition>>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let views = self.views.read().unwrap();
        Ok(views.get(name_lower).cloned()) // Arc::clone is cheap
    }

    /// List all view names
    pub fn list_views(&self) -> Result<Vec<String>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let views = self.views.read().unwrap();
        Ok(views.values().map(|v| v.original_name.clone()).collect())
    }

    /// No-op: in the new tombstone design, deduplication is handled at scan
    /// time via skip sets. Newer volumes shadow older volumes by row_id.
    /// Returns 0 (no explicit dedup performed).
    fn dedup_segments(&self) -> usize {
        0
    }

    /// Creates a full backup snapshot of all tables to .bin files.
    ///
    /// Writes all committed data (hot buffer + cold volumes) to snapshot .bin files
    /// in the snapshots/ directory. Each table gets its own subdirectory with
    /// timestamped snapshot files. The keep_snapshots config limits how many backup
    /// files are retained per table.
    fn create_backup_snapshot(&self) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }
        // See `seal_hot_buffers`: backup walks parent VersionStores
        // via `for_each_committed_version_with_cutoff` and would
        // export markerless committed rows into the snapshot .bin
        // files. A later restore would resurrect a commit the WAL
        // never has a marker for.
        if self.is_failed() {
            return Err(Error::internal(
                "backup snapshot refused: engine is in the catastrophic-failure \
                 state (a prior commit's WAL marker write failed after some tables \
                 were already committed). Restart the process; recovery will \
                 converge by discarding the markerless transaction.",
            ));
        }

        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()),
        };

        let snapshot_dir = pm.path().join("snapshots");
        if let Err(e) = std::fs::create_dir_all(&snapshot_dir) {
            return Err(Error::internal(format!(
                "failed to create snapshot directory: {}",
                e
            )));
        }

        // Block background compaction for the duration of the snapshot.
        // Compaction can swap segments and clear tombstones concurrently,
        // yielding a backup that mixes old tombstones with new volumes.
        while self
            .compaction_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let _compaction_guard = AtomicBoolGuard(&self.compaction_running);

        // Hold checkpoint_mutex for the entire snapshot operation.
        // This prevents concurrent seal from mutating cold state
        // (volumes, tombstones) while we read it, ensuring the backup is
        // a true point-in-time snapshot consistent with snapshot_commit_seq.
        let _checkpoint_guard = self.checkpoint_mutex.lock().unwrap();

        // Wait for all in-flight commits to complete before capturing state.
        // The commit path is: start_commit (alloc seq) → commit_all_tables
        // (apply versions + tombstones) → complete_commit (make visible).
        // Between start_commit and complete_commit, tombstones can be in the
        // shared set while hot versions are invisible. If we capture state in
        // that window, a cold row can be hidden by a tombstone whose txn is
        // excluded from the hot export. By waiting, we guarantee the tombstone
        // set and registry are fully consistent: every tombstone corresponds
        // to a visible commit, and the cutoff includes all of them.
        {
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if self.registry.safe_snapshot_cutoff() == self.registry.current_commit_sequence() {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(crate::core::Error::internal(
                        "PRAGMA SNAPSHOT timed out waiting for in-flight commits to complete",
                    ));
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }

        // Recheck the catastrophic-failure latch BEFORE capturing
        // and exporting any state. The wait above can sit blocked
        // while a failing commit is between commit_all_tables and
        // mark_engine_failed: it's already drained markerless rows
        // into parent VersionStores AND called complete_commit, so
        // safe_snapshot_cutoff includes its commit_seq. The entry-
        // time check from above passed (the latch wasn't set yet),
        // but proceeding now would export those markerless rows
        // via for_each_committed_version_with_cutoff.
        if self.is_failed() {
            return Err(Error::internal(
                "backup snapshot refused: engine entered the catastrophic-failure \
                 state during the in-flight-commit wait. Restart the process; \
                 recovery will discard the markerless transaction.",
            ));
        }

        // No in-flight commits at this point. Capture tombstones FIRST, then
        // cutoff. The commit path ordering is: start_commit (alloc seq) →
        // commit_all_tables (apply tombstones). So any tombstone in the shared
        // set was placed by a txn whose seq was already incremented. Our
        // subsequent current_commit_sequence() read will be ≥ that seq,
        // guaranteeing every frozen tombstone is within the cutoff.
        //
        // A new commit starting after our tombstone read but before our cutoff
        // read can only ADD to the cutoff (making it higher), never subtract.
        // Its tombstones won't be in our frozen set (captured before it started).
        let frozen_tombstones: ahash::AHashMap<String, Arc<FxHashMap<i64, u64>>> = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.iter()
                .map(|(name, mgr)| {
                    // Strip ephemeral (u64::MAX) tombstones from the
                    // backup snapshot's skip-set. These are failed-
                    // marker tombstones (record_commit IO failure
                    // after partial commit) that exist only in-memory
                    // — backing them up to a snapshot .bin file would
                    // let a later restore physically drop cold rows
                    // for a markerless commit.
                    let live = mgr.tombstone_set_arc();
                    let manifest = mgr.manifest();
                    let ephemeral: rustc_hash::FxHashSet<i64> = manifest
                        .tombstones
                        .iter()
                        .filter(|(_, _, vis)| *vis == u64::MAX)
                        .map(|(rid, _, _)| *rid)
                        .collect();
                    let value = if ephemeral.is_empty() {
                        live
                    } else {
                        let filtered: FxHashMap<i64, u64> = live
                            .iter()
                            .filter(|(rid, _)| !ephemeral.contains(rid))
                            .map(|(&k, &v)| (k, v))
                            .collect();
                        Arc::new(filtered)
                    };
                    (name.clone(), value)
                })
                .collect()
        };

        let snapshot_commit_seq = self.registry.current_commit_sequence();

        // Recheck the latch AFTER sampling cutoff and before
        // writing/finalizing snapshot files. The post-wait
        // recheck above only catches failures latched before the
        // capture begins; a brand-new commit can still slip in
        // here (the wait already returned), hit the marker-
        // failure path, latch the engine, complete_commit, and
        // its commit_seq lands at-or-below `snapshot_commit_seq`.
        // for_each_committed_version_with_cutoff would then
        // export those parent-store rows into the snapshot .bin
        // files, making markerless data restorable on future
        // engines.
        if self.is_failed() {
            return Err(Error::internal(
                "backup snapshot refused: engine entered the catastrophic-failure \
                 state after the in-flight-commit wait, before snapshot capture. \
                 Restart the process; recovery will discard the markerless \
                 transaction.",
            ));
        }

        let schemas = self.schemas.read().unwrap();
        let stores = self.version_stores.read().unwrap();

        // Generate consistent timestamp for all snapshots in this batch
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S%.3f").to_string();

        // Phase 1: Write all snapshots to temp files
        let mut pending_snapshots: Vec<(std::path::PathBuf, std::path::PathBuf, String)> =
            Vec::new();
        let mut all_succeeded = true;

        for (table_name, schema) in schemas.iter() {
            let table_snapshot_dir = snapshot_dir.join(table_name);
            if let Err(e) = std::fs::create_dir_all(&table_snapshot_dir) {
                eprintln!(
                    "Warning: Failed to create snapshot directory for {}: {}",
                    table_name, e
                );
                all_succeeded = false;
                break;
            }

            let final_path = table_snapshot_dir.join(format!("snapshot-{}.bin", timestamp));
            let temp_path = table_snapshot_dir.join(format!("snapshot-{}.bin.tmp", timestamp));

            let mut writer = match super::snapshot::SnapshotWriter::new(&temp_path) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to create snapshot writer for {}: {}",
                        table_name, e
                    );
                    all_succeeded = false;
                    break;
                }
            };

            if let Err(e) = writer.write_schema(schema) {
                eprintln!("Warning: Failed to write schema for {}: {}", table_name, e);
                writer.fail();
                all_succeeded = false;
                break;
            }

            let mut write_error = false;

            // Snapshot-consistent skip set: hot row_ids committed at or before
            // snapshot_commit_seq. Built during the hot-row write pass below to
            // avoid a redundant second B-tree traversal.
            let mut seen = FxHashSet::default();

            // Write hot committed rows (single pass: write + collect row_ids)
            if let Some(store) = stores.get(table_name) {
                seen.reserve(store.committed_row_count());
                store.for_each_committed_version_with_cutoff(
                    |row_id, version| {
                        seen.insert(row_id);
                        let mut snapshot_version = version.clone();
                        snapshot_version.txn_id = -1;
                        if let Err(e) = writer.append_row(row_id, &snapshot_version) {
                            eprintln!(
                                "Warning: Failed to write hot row {} to snapshot: {}",
                                row_id, e
                            );
                            write_error = true;
                            return false;
                        }
                        true
                    },
                    snapshot_commit_seq,
                );
            }

            if write_error {
                writer.fail();
                all_succeeded = false;
                break;
            }

            // Write cold volume rows (skip rows already written from hot buffer)
            let mgr = {
                let mgrs = self.segment_managers.read().unwrap();
                mgrs.get(table_name).cloned()
            };

            if let Some(mgr) = mgr {
                let volumes = mgr.get_volumes_newest_first();

                // Use the frozen tombstone snapshot captured immediately after
                // commit_seq. The commit path increments seq BEFORE applying
                // tombstones, so every tombstone in this set was committed at
                // seq ≤ snapshot_commit_seq. No post-cutoff tombstones can leak.
                let tombstones = frozen_tombstones
                    .get(table_name)
                    .cloned()
                    .unwrap_or_default();

                let schema_cols = schema.columns.len();

                for (seg_id, cs) in volumes.iter() {
                    let vol = &cs.volume;
                    let mapping = mgr.get_volume_mapping(*seg_id, schema);

                    for i in 0..vol.meta.row_count {
                        let row_id = vol.meta.row_ids[i];

                        // Skip tombstoned rows not already in hot snapshot.
                        // For int-PK tables, pre-cutoff deletes are in `seen`.
                        // For non-int-PK tables, tombstones are the only signal.
                        if tombstones.contains_key(&row_id) && !seen.contains(&row_id) {
                            continue;
                        }
                        // Skip rows already written from hot (cutoff-consistent)
                        // or already seen from a newer volume (dedup).
                        if !seen.insert(row_id) {
                            continue;
                        }

                        let mut row = if mapping.is_identity {
                            vol.get_row(i)
                        } else {
                            vol.get_row_mapped(i, &mapping)
                        };

                        if row.len() < schema_cols {
                            for ci in row.len()..schema_cols {
                                let col = &schema.columns[ci];
                                if let Some(ref default_val) = col.default_value {
                                    row.push(default_val.clone());
                                } else {
                                    row.push(crate::core::Value::null(col.data_type));
                                }
                            }
                        }

                        let snapshot_version = super::version_store::RowVersion {
                            txn_id: -1,
                            deleted_at_txn_id: 0,
                            data: row,
                            create_time: 0,
                        };

                        if let Err(e) = writer.append_row(row_id, &snapshot_version) {
                            eprintln!(
                                "Warning: Failed to write cold row {} to snapshot: {}",
                                row_id, e
                            );
                            write_error = true;
                            break;
                        }
                    }

                    if write_error {
                        break;
                    }
                }
            }

            if write_error {
                writer.fail();
                all_succeeded = false;
                break;
            }

            if let Err(e) = writer.finalize() {
                eprintln!(
                    "Warning: Failed to finalize snapshot for {}: {}",
                    table_name, e
                );
                writer.fail();
                all_succeeded = false;
                break;
            }

            pending_snapshots.push((temp_path, final_path, table_name.clone()));
        }

        // Final latch recheck before any rename. Phase 1 wrote
        // temp files; if a commit started after the cutoff sample
        // and hit the marker-failure path between then and now,
        // its parent-store rows landed in those temp files via
        // for_each_committed_version_with_cutoff (its commit_seq
        // was already in the cutoff at sample time). Renaming
        // would make markerless data restorable. Refuse and let
        // Phase 3 cleanup remove the temp files.
        if self.is_failed() {
            all_succeeded = false;
        }

        // Phase 2: Atomic rename of all temp files
        let mut renamed_successfully: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();

        if all_succeeded {
            let mut dirs_to_sync: FxHashSet<std::path::PathBuf> = FxHashSet::default();

            for (temp_path, final_path, table_name) in &pending_snapshots {
                if let Err(e) = std::fs::rename(temp_path, final_path) {
                    eprintln!(
                        "Warning: Failed to rename snapshot for {}: {}",
                        table_name, e
                    );
                    for (orig_temp, renamed_final) in renamed_successfully.iter().rev() {
                        if let Err(rollback_err) = std::fs::rename(renamed_final, orig_temp) {
                            eprintln!(
                                "Critical: Failed to rollback snapshot rename {:?} -> {:?}: {}",
                                renamed_final, orig_temp, rollback_err
                            );
                        }
                    }
                    all_succeeded = false;
                    break;
                }
                renamed_successfully.push((temp_path.clone(), final_path.clone()));
                if let Some(parent) = final_path.parent() {
                    dirs_to_sync.insert(parent.to_path_buf());
                }
            }

            #[cfg(not(windows))]
            if all_succeeded {
                for dir in &dirs_to_sync {
                    if let Ok(dir_file) = std::fs::File::open(dir) {
                        let _ = dir_file.sync_all();
                    }
                }
            }
        }

        // Phase 3: Cleanup on failure
        if !all_succeeded {
            for (temp_path, _, _) in &pending_snapshots {
                let _ = std::fs::remove_file(temp_path);
            }
            return Err(Error::internal(
                "snapshot creation failed, all temp files cleaned up",
            ));
        }

        // Phase 4: Write snapshot metadata
        let meta_path = snapshot_dir.join("snapshot_meta.bin");
        if let Err(e) = write_snapshot_metadata(&meta_path, 0) {
            eprintln!("Warning: Failed to write snapshot metadata: {}", e);
        }

        // Phase 4b: Save index and view definitions so --restore can recreate them.
        // Snapshot .bin files only contain row data + schema, not DDL objects.
        {
            let mut ddl_buf: Vec<u8> = Vec::new();
            // Magic + version
            ddl_buf.extend_from_slice(b"SDDL");
            ddl_buf.push(1); // version

            // Collect indexes
            let mut index_entries: Vec<Vec<u8>> = Vec::new();
            for (table_name, store) in stores.iter() {
                let _ = store.for_each_index(|index| {
                    if index.index_type() == crate::core::IndexType::PrimaryKey {
                        return Ok(());
                    }
                    let meta = super::persistence::IndexMetadata {
                        name: index.name().to_string(),
                        table_name: table_name.clone(),
                        column_names: index.column_names().to_vec(),
                        column_ids: index.column_ids().to_vec(),
                        data_types: index.data_types().to_vec(),
                        is_unique: index.is_unique(),
                        index_type: index.index_type(),
                        hnsw_m: index.hnsw_m(),
                        hnsw_ef_construction: index.hnsw_ef_construction(),
                        hnsw_ef_search: index.default_ef_search().map(|v| v as u16),
                        hnsw_distance_metric: index.hnsw_distance_metric(),
                    };
                    index_entries.push(meta.serialize());
                    Ok(())
                });
            }
            ddl_buf.extend_from_slice(&(index_entries.len() as u32).to_le_bytes());
            for entry in &index_entries {
                ddl_buf.extend_from_slice(&(entry.len() as u32).to_le_bytes());
                ddl_buf.extend_from_slice(entry);
            }

            // Collect views
            let view_entries: Vec<Vec<u8>> = {
                let views = self.views.read().unwrap();
                views.values().map(|v| v.serialize()).collect()
            };
            ddl_buf.extend_from_slice(&(view_entries.len() as u32).to_le_bytes());
            for entry in &view_entries {
                ddl_buf.extend_from_slice(&(entry.len() as u32).to_le_bytes());
                ddl_buf.extend_from_slice(entry);
            }

            // Append CRC32 checksum of the entire buffer
            let crc = crc32fast::hash(&ddl_buf);
            ddl_buf.extend_from_slice(&crc.to_le_bytes());

            // Write per-timestamp DDL file so timestamped restores get the correct
            // indexes/views that existed at that snapshot point.
            let ddl_path = snapshot_dir.join(format!("ddl-{}.bin", timestamp));
            let ddl_tmp = snapshot_dir.join(format!("ddl-{}.bin.tmp", timestamp));
            let ddl_result = (|| -> std::io::Result<()> {
                let f = std::fs::File::create(&ddl_tmp)?;
                std::io::Write::write_all(&mut &f, &ddl_buf)?;
                f.sync_all()?;
                std::fs::rename(&ddl_tmp, &ddl_path)?;
                #[cfg(not(windows))]
                if let Ok(dir) = std::fs::File::open(&snapshot_dir) {
                    let _ = dir.sync_all();
                }
                Ok(())
            })();
            // Write per-batch manifest listing all tables in this snapshot.
            // Used by CLI restore to select a complete, coherent snapshot set
            // instead of guessing from individual table directories.
            let manifest_path = snapshot_dir.join(format!("manifest-{}.json", timestamp));
            let manifest_tmp = snapshot_dir.join(format!("manifest-{}.json.tmp", timestamp));
            let table_list: Vec<&str> = pending_snapshots
                .iter()
                .map(|(_, _, name)| name.as_str())
                .collect();
            let manifest_json = format!(
                "{{\"timestamp\":\"{}\",\"tables\":[{}]}}",
                timestamp,
                table_list
                    .iter()
                    .map(|t| {
                        let escaped = t.replace('\\', "\\\\").replace('"', "\\\"");
                        format!("\"{}\"", escaped)
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let manifest_result = if ddl_result.is_ok() {
                (|| -> std::io::Result<()> {
                    let f = std::fs::File::create(&manifest_tmp)?;
                    std::io::Write::write_all(&mut &f, manifest_json.as_bytes())?;
                    f.sync_all()?;
                    std::fs::rename(&manifest_tmp, &manifest_path)?;
                    Ok(())
                })()
            } else {
                Ok(())
            };

            // DDL metadata + per-batch manifest are part of the atomic snapshot
            // batch. Restore depends on both: the manifest names the per-table
            // .bin files in the snapshot, and ddl-{ts}.bin reconstructs indexes
            // and views. If either write fails, treat the whole batch as failed
            // and roll back: delete the .bin files just renamed, delete partial
            // ddl/manifest files, and skip prune phases. Otherwise the latest
            // manifest could reference a missing ddl, or pruning could delete
            // .bin files the still-current previous manifest still names.
            if ddl_result.is_err() || manifest_result.is_err() {
                if let Err(e) = ddl_result {
                    eprintln!("Failed to write DDL metadata for snapshot: {}", e);
                }
                if let Err(e) = manifest_result {
                    eprintln!("Failed to write snapshot manifest: {}", e);
                }
                // Rollback the partial batch. These removes are
                // best-effort because we cannot fail any harder
                // than the original write that brought us here,
                // but log per-file failures distinctly so a user
                // inspecting the snapshot dir can correlate any
                // surviving stragglers (Phase 6 will eventually
                // prune by sort order, but that's many snapshot
                // cycles away under keep_snapshots=1).
                let to_remove = [
                    ("ddl tmp", ddl_tmp.clone()),
                    ("ddl final", ddl_path.clone()),
                    ("manifest tmp", manifest_tmp.clone()),
                    ("manifest final", manifest_path.clone()),
                ];
                for (label, path) in &to_remove {
                    if let Err(e) = std::fs::remove_file(path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            eprintln!(
                                "Snapshot rollback: could not remove {} {:?}: {} (orphan may persist until Phase 6 prune).",
                                label, path, e
                            );
                        }
                    }
                }
                for (_, final_path, table_name) in &pending_snapshots {
                    if let Err(e) = std::fs::remove_file(final_path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            eprintln!(
                                "Snapshot rollback: could not remove per-table snapshot for {} at {:?}: {} (orphan .bin may persist until cleanup_old_snapshots prunes).",
                                table_name, final_path, e
                            );
                        }
                    }
                }
                return Err(Error::internal(
                    "snapshot creation failed: DDL or manifest write failed; new batch rolled back, older snapshots preserved",
                ));
            }
        }

        // Phase 5: Cleanup old snapshots (read from live config, not immutable PM)
        let keep_snapshots = self
            .config
            .read()
            .unwrap()
            .persistence
            .keep_snapshots
            .max(1) as usize;
        for (_, _, table_name) in &pending_snapshots {
            if let Some(schema) = schemas.get(table_name) {
                let disk_store =
                    super::snapshot::DiskVersionStore::new(&snapshot_dir, table_name, schema);
                if let Ok(disk_store) = disk_store {
                    if let Err(e) = disk_store.cleanup_old_snapshots(keep_snapshots) {
                        eprintln!(
                            "Warning: Failed to cleanup old snapshots for {}: {}",
                            table_name, e
                        );
                    }
                }
            }
        }

        // Snapshot directories for dropped tables are intentionally preserved.
        // Deleting them would make point-in-time restore unable to reconstruct
        // the database state from before the table was dropped.

        // Phase 6: Cleanup old ddl-*.bin files to match keep_snapshots limit.
        // Each snapshot writes a ddl-{timestamp}.bin; without cleanup these grow unbounded.
        {
            let mut ddl_files: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshot_dir)
                .ok()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            p.file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|n| n.starts_with("ddl-") && n.ends_with(".bin"))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if ddl_files.len() > keep_snapshots {
                ddl_files.sort();
                for old_ddl in &ddl_files[..ddl_files.len() - keep_snapshots] {
                    let _ = std::fs::remove_file(old_ddl);
                }
            }
        }

        // Cleanup old manifest-*.json files to match keep_snapshots limit.
        {
            let mut manifest_files: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshot_dir)
                .ok()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            p.file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|n| n.starts_with("manifest-") && n.ends_with(".json"))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if manifest_files.len() > keep_snapshots {
                manifest_files.sort();
                for old_m in &manifest_files[..manifest_files.len() - keep_snapshots] {
                    let _ = std::fs::remove_file(old_m);
                }
            }
        }

        Ok(())
    }

    /// Restore the database state from backup snapshots.
    /// If timestamp is None, restores from the latest valid snapshot per table.
    /// If timestamp is Some("YYYYMMDD-HHMMSS.fff"), restores from that specific snapshot.
    ///
    /// This is a destructive operation: all current data (hot buffer + volumes) is
    /// Restore indexes and views from a ddl.bin file saved by PRAGMA SNAPSHOT.
    /// Format: "SDDL" (4) + version (1) + index_count (u32) + entries + view_count (u32) + entries
    fn restore_ddl_from_bin(&self, data: &[u8]) -> Result<()> {
        // Returns Err on ANY corruption (bad magic, CRC mismatch,
        // truncation, decode failure, per-entry create failure).
        // RESTORE depends on this output for correctness:
        // UNIQUE / FK enforcement is driven from the restored
        // indexes — silently skipping a corrupted entry would
        // make RESTORE report success while leaving the
        // database with missing constraints. Recovery cannot
        // recover from this without operator intervention.

        // Minimum: magic(4) + version(1) + crc(4)
        if data.len() < 9 {
            return Err(Error::internal(format!(
                "ddl.bin too short ({} bytes; minimum 9 for header + CRC)",
                data.len()
            )));
        }
        if &data[0..4] != b"SDDL" {
            return Err(Error::internal(format!(
                "ddl.bin bad magic (expected 'SDDL', got {:?})",
                &data[0..4.min(data.len())]
            )));
        }

        // Verify CRC32: last 4 bytes are checksum of everything before
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(Error::internal(format!(
                "ddl.bin CRC mismatch (stored=0x{:08x}, computed=0x{:08x})",
                stored_crc, computed_crc
            )));
        }

        let mut pos = 5; // skip magic + version

        // Read indexes
        if pos + 4 > data.len() {
            return Err(Error::internal(
                "ddl.bin truncated before index_count".to_string(),
            ));
        }
        let index_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let stores = self.version_stores.read().unwrap();
        for i in 0..index_count {
            if pos + 4 > data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin truncated before index {} length prefix",
                    i
                )));
            }
            let entry_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + entry_len > data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin truncated inside index {} payload (need {} bytes, \
                     have {})",
                    i,
                    entry_len,
                    data.len() - pos
                )));
            }
            let entry_data = &data[pos..pos + entry_len];
            pos += entry_len;

            let meta = super::persistence::IndexMetadata::deserialize(entry_data).map_err(|e| {
                Error::internal(format!("ddl.bin index {} decode failed: {}", i, e))
            })?;
            let table_lower = meta.table_name.to_lowercase();
            if let Some(store) = stores.get(&table_lower) {
                store
                    .create_index_from_metadata_with_graph(&meta, false, None)
                    .map_err(|e| {
                        Error::internal(format!(
                            "ddl.bin: failed to recreate index '{}' on '{}': {}",
                            meta.name, meta.table_name, e
                        ))
                    })?;
            } else {
                return Err(Error::internal(format!(
                    "ddl.bin index '{}' references unknown table '{}' (snapshot \
                     load may have skipped it)",
                    meta.name, meta.table_name
                )));
            }
        }
        drop(stores);

        // Read views
        if pos + 4 > data.len() {
            return Err(Error::internal(
                "ddl.bin truncated before view_count".to_string(),
            ));
        }
        let view_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        for i in 0..view_count {
            if pos + 4 > data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin truncated before view {} length prefix",
                    i
                )));
            }
            let entry_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + entry_len > data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin truncated inside view {} payload",
                    i
                )));
            }
            let entry_data = &data[pos..pos + entry_len];
            pos += entry_len;

            // ViewDefinition format: name_len(u16) + name + query_len(u32) + query
            let mut vpos = 0;
            if vpos + 2 > entry_data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin view {} truncated before name length",
                    i
                )));
            }
            let name_len =
                u16::from_le_bytes(entry_data[vpos..vpos + 2].try_into().unwrap()) as usize;
            vpos += 2;
            if vpos + name_len > entry_data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin view {} truncated inside name",
                    i
                )));
            }
            let view_name = String::from_utf8_lossy(&entry_data[vpos..vpos + name_len]).to_string();
            vpos += name_len;
            if vpos + 4 > entry_data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin view '{}' truncated before query length",
                    view_name
                )));
            }
            let query_len =
                u32::from_le_bytes(entry_data[vpos..vpos + 4].try_into().unwrap()) as usize;
            vpos += 4;
            if vpos + query_len > entry_data.len() {
                return Err(Error::internal(format!(
                    "ddl.bin view '{}' truncated inside query payload",
                    view_name
                )));
            }
            let query = String::from_utf8_lossy(&entry_data[vpos..vpos + query_len]).to_string();

            let view_def = Arc::new(ViewDefinition::new(&view_name, query));
            self.views
                .write()
                .unwrap()
                .insert(view_name.to_lowercase(), view_def);
        }
        Ok(())
    }

    /// Restore the database from a backup snapshot, replacing all current data.
    fn restore_from_snapshot(&self, timestamp: Option<&str>) -> Result<String> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => {
                return Err(Error::internal(
                    "PRAGMA RESTORE requires a persistent database",
                ))
            }
        };

        let snapshot_dir = pm.path().join("snapshots");
        if !snapshot_dir.exists() {
            return Err(Error::internal("No snapshots directory found"));
        }

        // Block any new cross-process reader attach for the
        // duration of restore by retaking `db.startup.lock`
        // EX. The writer's `open_engine` released this gate
        // after `mark_ready`; re-acquiring it here serializes
        // every subsequent reader's
        // `await_writer_startup_quiescent` call (it blocks on
        // SH while we hold EX). Held until the success path
        // below, where the freshly-restored manifests + WAL
        // are in place; on any early-Err the guard's Drop
        // releases the gate and queued readers proceed
        // against the OLD on-disk state, which is exactly
        // what they would have seen if restore had never run.
        //
        // Memory engines and read-only mounts return None
        // from `acquire_startup_exclusive`; restore on those
        // paths already short-circuits via the persistence
        // check above.
        let restore_attach_gate =
            match crate::storage::mvcc::file_lock::FileLock::acquire_startup_exclusive(
                std::path::Path::new(&self.path),
            ) {
                Ok(g) => g,
                Err(e) => {
                    return Err(Error::internal(format!(
                        "PRAGMA RESTORE refused: failed to acquire startup gate \
                     (another process may be attaching as read-only right now): {}",
                        e
                    )));
                }
            };
        // RAII helper: any panic / early-Err path that
        // doesn't explicitly stash the gate releases it on
        // unwind / function exit, restoring the pre-restore
        // attach behaviour. The `latch_attach_gate_on_failure`
        // method below moves the guard into
        // `MVCCEngine.failed_restore_attach_gate` for the
        // post-destructive-boundary failure paths.
        let mut restore_attach_gate = restore_attach_gate;
        // Refuse the entire restore while live cross-process
        // readers are attached. The destructive phase below
        // wholesale `remove_dir_all`s every `<volumes>/<table>/`
        // directory, including the .vol files a reader's
        // stale manifest may still lazy-load via
        // `SegmentManager::ensure_volume`. Unlike DROP /
        // TRUNCATE / compaction (which can defer the
        // unlink and leave files in place), restore by
        // construction needs a clean volumes/ tree before
        // it can lay down the snapshot's data — there is
        // no safe in-place defer.
        //
        // This check runs AFTER the startup-gate acquisition
        // above so a reader attempting to attach concurrently
        // is already blocked at its own
        // `await_writer_startup_quiescent` — its lease
        // (registered before the SH attempt) is visible to
        // `defer_for_live_readers`, OR it hasn't gotten as
        // far as registering and will block on SH while we
        // destroy and restore. Either outcome closes the
        // race the reviewer flagged.
        //
        // This check runs BEFORE
        // `registry.stop_accepting_transactions()` below, so
        // an early Err here doesn't need a matching
        // `start_accepting_transactions()`.
        if self.defer_for_live_readers() {
            return Err(Error::internal(
                "PRAGMA RESTORE refused: one or more cross-process read-only \
                 handles are attached. Detach all readers (or wait for their \
                 leases to expire) before retrying.",
            ));
        }

        // Claim the compaction slot so no background compaction can start
        // during restore. A detached compaction thread holds old
        // SegmentManager Arcs and could persist pre-restore segments
        // after we clear the segment manager map.
        while self
            .compaction_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Drop guard: release the compaction slot on any early return.
        let _compaction_guard = AtomicBoolGuard(&self.compaction_running);

        // Acquire checkpoint_mutex to prevent concurrent seal/compact/snapshot
        let _checkpoint_guard = self.checkpoint_mutex.lock().unwrap();

        // Stop accepting new transactions and wait for active ones to drain
        self.registry.stop_accepting_transactions();
        let remaining = self
            .registry
            .wait_for_active_transactions(std::time::Duration::from_secs(5));
        if remaining > 0 {
            self.registry.start_accepting_transactions();
            return Err(Error::internal(format!(
                "Cannot restore: {} active transactions still running",
                remaining
            )));
        }

        // Find matching snapshot files per table
        let mut snapshot_files: Vec<(String, std::path::PathBuf)> = Vec::new();
        let table_dirs = match std::fs::read_dir(&snapshot_dir) {
            Ok(entries) => entries,
            Err(e) => {
                self.registry.start_accepting_transactions();
                return Err(Error::internal(format!(
                    "Cannot read snapshots directory: {}",
                    e
                )));
            }
        };

        // Resolve the authoritative manifest for this restore (if any).
        // - explicit timestamp: look for the matching `manifest-{ts}.json`.
        // - latest restore: take the newest `manifest-*.json`.
        //
        // When a manifest exists it pins BOTH the table list AND the
        // exact `snapshot-{ts}.bin` filename to load for each listed
        // table (see the loop below). This prevents:
        //   - resurrecting tables dropped between snapshots,
        //   - mixing unmanifested orphan snapshot files (left behind
        //     when `create_snapshot`'s manifest write failed after the
        //     per-table writes) with the manifested batch.
        //
        // No-manifest is the legacy-backup fallback: per-table dir scan.
        let manifest_path: Option<std::path::PathBuf> = match timestamp {
            Some(ts) => {
                let p = snapshot_dir.join(format!("manifest-{}.json", ts));
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            }
            None => {
                // Fail closed on enumeration errors. Falling back to
                // None would let the caller scan every surviving
                // snapshot table directory, including ones
                // intentionally preserved for tables dropped after
                // older backups. We are still before the destructive
                // boundary, so a plain Err return after resuming the
                // registry is safe.
                let entries = std::fs::read_dir(&snapshot_dir).map_err(|e| {
                    self.registry.start_accepting_transactions();
                    Error::internal(format!(
                        "PRAGMA RESTORE: cannot enumerate snapshots directory {:?}: {} — \
                         refusing to fall back to all-directories scan (would resurrect \
                         tables that were dropped after older backups).",
                        snapshot_dir, e
                    ))
                })?;
                let mut manifests: Vec<std::path::PathBuf> = Vec::new();
                for entry in entries {
                    let entry = entry.map_err(|e| {
                        self.registry.start_accepting_transactions();
                        Error::internal(format!(
                            "PRAGMA RESTORE: cannot read entry in snapshots directory {:?}: {} — \
                             refusing to fall back to all-directories scan.",
                            snapshot_dir, e
                        ))
                    })?;
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json")
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("manifest-"))
                    {
                        manifests.push(p);
                    }
                }
                manifests.sort();
                manifests.pop()
            }
        };

        let effective_manifest: Option<(String, Vec<String>)> = if let Some(latest) = manifest_path
        {
            // manifest-*.json is {"timestamp":"...","tables":["t1","t2"]}.
            // Any read/parse failure is FATAL — falling back to a
            // dir-scan would resurrect dropped tables and/or mix in
            // unmanifested orphan snapshot files. We are still before
            // the destructive boundary, so plain Err is safe; the
            // caller resumes transactions on early Err paths.
            let data = std::fs::read_to_string(&latest).map_err(|e| {
                self.registry.start_accepting_transactions();
                Error::internal(format!(
                    "PRAGMA RESTORE: snapshot manifest {:?} exists but \
                         could not be read: {} — refusing to fall back to \
                         all-directories scan (would resurrect tables that \
                         were dropped after older backups).",
                    latest, e
                ))
            })?;
            let parsed: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
                self.registry.start_accepting_transactions();
                Error::internal(format!(
                    "PRAGMA RESTORE: snapshot manifest {:?} could not \
                         be parsed: {} — refusing to fall back to all-\
                         directories scan.",
                    latest, e
                ))
            })?;
            // Strict parse: any non-string entry is fatal.
            // Silently filtering would shrink the table set and the
            // destructive restore would omit a table that was actually
            // in the snapshot batch while reporting success.
            //
            // Empty `tables: []` is allowed: `create_backup_snapshot`
            // can legitimately publish a manifest for a database with
            // no tables (e.g. views-only / DDL-only point-in-time
            // state). The loop below skips per-table loading and the
            // restore reduces to "drop all current tables and apply
            // the manifest's DDL".
            let tables_array =
                parsed
                    .get("tables")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        self.registry.start_accepting_transactions();
                        Error::internal(format!(
                            "PRAGMA RESTORE: snapshot manifest {:?} missing or non-array \
                             `tables` field — refusing to fall back to all-directories scan \
                             (would resurrect dropped tables).",
                            latest
                        ))
                    })?;
            let mut tables: Vec<String> = Vec::with_capacity(tables_array.len());
            for (i, v) in tables_array.iter().enumerate() {
                match v.as_str() {
                    Some(s) => tables.push(s.to_string()),
                    None => {
                        self.registry.start_accepting_transactions();
                        return Err(Error::internal(format!(
                            "PRAGMA RESTORE: snapshot manifest {:?} has non-string \
                                 entry at `tables[{}]` ({}) — refusing to restore from a \
                                 corrupt manifest (would silently omit a table).",
                            latest, i, v
                        )));
                    }
                }
            }
            // Resolve manifest timestamp. The filename is
            // authoritative: it is what `manifests.sort()`
            // ordered on, what locates `snapshot-{ts}.bin` /
            // `ddl-{ts}.bin` on disk, and what an explicit
            // restore arg matched. If the JSON `timestamp`
            // field is present, it MUST equal the filename
            // suffix — disagreement is corrupt metadata, not
            // an excuse to silently load the wrong batch.
            // Trusting the JSON field over the filename would
            // let a corrupt `manifest-newer.json` (whose
            // `timestamp` points at an older batch) be picked
            // by latest-restore as "the latest", then load
            // older `snapshot-old.bin` / `ddl-old.bin` and
            // report success against the wrong point in time.
            // For an explicit empty-tables restore there is
            // also no per-table `snapshot-{ts}.bin` check
            // below to catch the mismatch.
            let ts_from_filename = latest
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("manifest-"))
                .and_then(|n| n.strip_suffix(".json"))
                .map(String::from);
            let ts_from_field = parsed
                .get("timestamp")
                .and_then(|v| v.as_str())
                .map(String::from);
            if let (Some(ref filename_ts), Some(ref field_ts)) =
                (ts_from_filename.as_ref(), ts_from_field.as_ref())
            {
                if filename_ts != field_ts {
                    self.registry.start_accepting_transactions();
                    return Err(Error::internal(format!(
                        "PRAGMA RESTORE: snapshot manifest {:?} has \
                         `timestamp` field '{}' that disagrees with the \
                         filename suffix '{}' — refusing to restore from \
                         corrupt metadata (would load the wrong batch).",
                        latest, field_ts, filename_ts
                    )));
                }
            }
            let manifest_ts = ts_from_filename.or(ts_from_field).ok_or_else(|| {
                self.registry.start_accepting_transactions();
                Error::internal(format!(
                    "PRAGMA RESTORE: snapshot manifest {:?} has no \
                         resolvable timestamp (neither `timestamp` field nor \
                         `manifest-{{ts}}.json` filename) — refusing to fall \
                         back to newest-per-table scan (could mix data from \
                         different backup batches).",
                    latest
                ))
            })?;
            Some((manifest_ts, tables))
        } else {
            None
        };

        for entry in table_dirs {
            // Per-entry Err is fatal here: when there's no
            // `effective_manifest` (legacy backup), this dir
            // scan IS the table list. Silently dropping a
            // failed entry would let restore proceed against a
            // shorter table set and report success while
            // missing one. Manifested restores are also
            // protected by the per-table verification at
            // 6810-6829, but applying the same explicit
            // handling here keeps both paths uniformly safe.
            // Pre-destructive boundary: resume registry, plain
            // Err.
            let entry = entry.map_err(|e| {
                self.registry.start_accepting_transactions();
                Error::internal(format!(
                    "PRAGMA RESTORE: cannot read entry in snapshots directory {:?}: {} — \
                     refusing to proceed (would silently omit a table from a legacy backup).",
                    snapshot_dir, e
                ))
            })?;
            // file_type() failure is also fatal — we cannot
            // tell if this entry is a table dir whose contents
            // we'd otherwise process.
            let ft = entry.file_type().map_err(|e| {
                self.registry.start_accepting_transactions();
                Error::internal(format!(
                    "PRAGMA RESTORE: cannot stat snapshot dir entry {:?}: {} — \
                     refusing to proceed (could silently omit a table).",
                    entry.path(),
                    e
                ))
            })?;
            if !ft.is_dir() {
                continue;
            }
            let table_name = entry.file_name().to_string_lossy().to_string();

            // Skip tables not in the manifest (prevents resurrecting dropped tables)
            if let Some((_, ref tables)) = effective_manifest {
                if !tables.iter().any(|t| t.eq_ignore_ascii_case(&table_name)) {
                    continue;
                }
            }

            let paths = Self::find_snapshots_newest_first(&entry.path());

            // Pick which snapshot file to load:
            //   - explicit timestamp arg → exact `snapshot-{ts}.bin`
            //   - latest restore with manifest → exact
            //     `snapshot-{manifest_ts}.bin` (same
            //     timestamp the manifest was written with),
            //     so unmanifested orphan files left behind
            //     by a snapshot batch whose later
            //     `manifest-*.json` write failed are
            //     IGNORED instead of being mixed into the
            //     restore.
            //   - latest restore with no manifest (legacy
            //     backup) → newest file per table.
            let target_ts: Option<&str> = if let Some(ts) = timestamp {
                Some(ts)
            } else {
                effective_manifest.as_ref().map(|(ts, _)| ts.as_str())
            };

            if let Some(ts) = target_ts {
                let target = format!("snapshot-{}.bin", ts);
                if let Some(path) = paths
                    .iter()
                    .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(target.as_str()))
                {
                    snapshot_files.push((table_name, path.clone()));
                }
            } else if let Some(path) = paths.first() {
                snapshot_files.push((table_name, path.clone()));
            }
        }

        if snapshot_files.is_empty() {
            // A manifest with `tables: []` is a valid empty
            // point-in-time state (no tables, possibly
            // views-only / DDL-only). Restore proceeds: skip
            // per-table loading and let DDL replay produce the
            // empty catalog the manifest pinned.
            let manifest_allows_empty = effective_manifest
                .as_ref()
                .is_some_and(|(_, tables)| tables.is_empty());
            if !manifest_allows_empty {
                self.registry.start_accepting_transactions();
                return Err(Error::internal("No matching snapshots found"));
            }
        }

        // When a manifest pinned the timestamp + table list,
        // require the exact `snapshot-{ts}.bin` for EVERY
        // table the manifest listed. A missing match here
        // means snapshot creation got partway then lost the
        // batch (older snapshot files exist but the
        // matching `snapshot-{ts}.bin` for this table was
        // never written, or was deleted). Mixing the
        // table's older snapshot in would be a partial
        // point-in-time restore — fail loud instead.
        if let Some((ref manifest_ts, ref tables)) = effective_manifest {
            let target = format!("snapshot-{}.bin", manifest_ts);
            for t in tables {
                let t_lower = t.to_ascii_lowercase();
                let have = snapshot_files.iter().any(|(name, path)| {
                    name.eq_ignore_ascii_case(&t_lower)
                        && path.file_name().and_then(|n| n.to_str()) == Some(target.as_str())
                });
                if !have {
                    self.registry.start_accepting_transactions();
                    return Err(Error::internal(format!(
                        "PRAGMA RESTORE: snapshot manifest pinned timestamp \
                         '{}' but table '{}' has no '{}' on disk — refusing \
                         to mix an older snapshot of this table with the \
                         rest of the batch (would be a partial point-in-time \
                         restore).",
                        manifest_ts, t, target
                    )));
                }
            }
        }

        // Pre-validate: open all readers AND companion .vol files to catch corruption
        // before the destructive step. Without this, a corrupt .vol would fail after
        // WAL/volumes are already deleted, causing total data loss.
        for (table_name, path) in &snapshot_files {
            if let Err(e) = super::snapshot::SnapshotReader::open(path) {
                self.registry.start_accepting_transactions();
                return Err(Error::internal(format!(
                    "Snapshot validation failed for table {}: {}",
                    table_name, e
                )));
            }
            // Also validate companion .vol if it exists (will be used by load_table_snapshot_as_volume)
            let vol_path = path.with_extension("vol");
            if vol_path.exists() {
                if let Err(e) = crate::storage::volume::io::read_volume_from_disk(&vol_path) {
                    self.registry.start_accepting_transactions();
                    return Err(Error::internal(format!(
                        "Snapshot volume validation failed for table {}: {}",
                        table_name, e
                    )));
                }
            }
        }

        // === Load DDL metadata (indexes + views) ===
        // Use per-timestamp DDL file (ddl-{ts}.bin) matching the snapshot data.
        // Preference order:
        //   1. Manifest-pinned timestamp (matches the batch the manifest published).
        //   2. Explicit `timestamp` arg (legacy timestamped restore with no manifest).
        //   3. Oldest timestamp across the selected snapshot files (latest restore
        //      with no manifest — conservative, never newer than any restored data).
        let ddl_data = {
            let effective_ts = if let Some((ref manifest_ts, _)) = effective_manifest {
                Some(manifest_ts.clone())
            } else if let Some(ts) = timestamp {
                Some(ts.to_string())
            } else {
                snapshot_files
                    .iter()
                    .filter_map(|(_, path)| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .and_then(|n| n.strip_prefix("snapshot-"))
                            .and_then(|n| n.strip_suffix(".bin"))
                            .map(|ts| ts.to_string())
                    })
                    .min()
            };
            let ddl_path = match &effective_ts {
                Some(ts) => snapshot_dir.join(format!("ddl-{}.bin", ts)),
                None => snapshot_dir.join("ddl.bin"),
            };
            if ddl_path.exists() {
                // The file exists — read failure is fatal.
                // Falling back to current in-memory indexes
                // / views would silently restore constraints
                // from the wrong point in time (timestamped
                // restore: the wrong snapshot; latest
                // restore: the live database's current
                // catalog). Either case can leave RESTORE
                // reporting success against a database
                // missing or carrying the wrong UNIQUE / FK
                // indexes. Runs BEFORE the destructive
                // boundary, so a plain Err return is safe —
                // no latch needed.
                Some(std::fs::read(&ddl_path).map_err(|e| {
                    // Resume the registry — restore changed
                    // nothing on disk yet (this read runs
                    // BEFORE the destructive boundary), so
                    // future user transactions must be
                    // unblocked before we return Err.
                    // Otherwise a transient permission /
                    // I/O failure here would leave the
                    // engine open but refusing transactions
                    // for the rest of the process.
                    self.registry.start_accepting_transactions();
                    Error::internal(format!(
                        "PRAGMA RESTORE: ddl metadata file {:?} exists but \
                         could not be read: {} — refusing to fall back to \
                         current in-memory indexes / views (would restore \
                         wrong constraints).",
                        ddl_path, e
                    ))
                })?)
            } else if timestamp.is_some() {
                // Timestamped restore with missing DDL: fail rather than silently
                // using current indexes/views which may not match the snapshot data.
                self.registry.start_accepting_transactions();
                return Err(Error::internal(format!(
                    "DDL metadata file not found for timestamp '{}'. Cannot restore indexes/views accurately.",
                    timestamp.unwrap()
                )));
            } else if effective_manifest.is_some() {
                // Manifest-selected latest restore with no
                // matching `ddl-{ts}.bin`: refuse rather
                // than fall through to the live in-memory
                // index/view fallback below. `create_snapshot`
                // can produce exactly this state — it
                // writes the snapshot batch + DDL metadata
                // BEFORE the manifest, but if the DDL
                // metadata write only logged a failure
                // before the manifest was attempted, the
                // manifest still names a timestamp whose
                // `ddl-{ts}.bin` is missing. Restoring
                // with current constraints / views would
                // report success for the wrong point in
                // time. The legacy no-manifest fallback
                // (no `manifest-*.json` on disk) still
                // proceeds with in-memory indexes —
                // covered by the `else { None }` arm
                // below.
                self.registry.start_accepting_transactions();
                return Err(Error::internal(format!(
                    "PRAGMA RESTORE: snapshot manifest pinned timestamp but \
                     DDL metadata {:?} is missing — refusing to fall back to \
                     current in-memory indexes / views (would restore wrong \
                     constraints).",
                    ddl_path
                )));
            } else {
                None
            }
        };

        // Fallback: if no DDL file (latest restore only), save current indexes from memory
        let saved_indexes: Vec<(String, super::persistence::IndexMetadata)> = if ddl_data.is_some()
        {
            Vec::new() // Will use DDL file instead
        } else {
            let stores = self.version_stores.read().unwrap();
            let mut indexes = Vec::new();
            for (table_name, store) in stores.iter() {
                let _ = store.for_each_index(|index| {
                    if index.index_type() == crate::core::IndexType::PrimaryKey {
                        return Ok(());
                    }
                    indexes.push((
                        table_name.clone(),
                        super::persistence::IndexMetadata {
                            name: index.name().to_string(),
                            table_name: table_name.clone(),
                            column_names: index.column_names().to_vec(),
                            column_ids: index.column_ids().to_vec(),
                            data_types: index.data_types().to_vec(),
                            is_unique: index.is_unique(),
                            index_type: index.index_type(),
                            hnsw_m: index.hnsw_m(),
                            hnsw_ef_construction: index.hnsw_ef_construction(),
                            hnsw_ef_search: index.default_ef_search().map(|v| v as u16),
                            hnsw_distance_metric: index.hnsw_distance_metric(),
                        },
                    ));
                    Ok(())
                });
            }
            indexes
        };

        // Save current views as fallback (if no ddl.bin)
        let saved_views: Vec<(String, String)> = if ddl_data.is_some() {
            Vec::new()
        } else {
            let views = self.views.read().unwrap();
            views
                .values()
                .map(|v| (v.original_name.clone(), v.query.clone()))
                .collect()
        };

        // === Destructive step: clear current state ===

        // Close all version stores
        {
            let stores = self.version_stores.read().unwrap();
            for store in stores.values() {
                store.close();
            }
        }

        // Clear segment managers and delete volume files on disk
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            for (_, mgr) in mgrs.iter() {
                mgr.clear();
            }
            mgrs.clear();
        }

        // Delete all volume directories on disk. Propagate
        // any failure BEFORE the WAL truncate below — with
        // volume-format manifests, correctness depends on
        // every old `<volumes>/<table>/manifest.bin` being
        // gone before WAL reset. A surviving stale manifest
        // would have NO `DropTable` record left in WAL to
        // remove it on the next recovery, so a same-name
        // table would resurface with the old cold rows.
        // Wrap the FS cleanup in a closure so any
        // `read_dir` / entry-decode / `remove_dir_all`
        // failure is funnelled through a single Err arm
        // that latches the catastrophic-failure flag
        // BEFORE returning. By this point all VersionStores
        // have been closed and segment managers cleared, so
        // the in-memory catalog is partially destroyed; the
        // registry's `stop_accepting_transactions` blocks
        // new user txns but auto-commit DDL paths bypass
        // the registry and only consult `ensure_writable` /
        // `record_ddl`'s catastrophic-failure check.
        // Without the latch a CREATE / DROP / TRUNCATE
        // issued before the operator restarts would mutate
        // / publish DDL against this half-destroyed state.
        let vol_dir = pm.path().join("volumes");
        let cleanup_result: Result<()> = (|| {
            if vol_dir.exists() {
                let entries = std::fs::read_dir(&vol_dir).map_err(|e| {
                    Error::internal(format!(
                        "PRAGMA RESTORE: failed to enumerate volume directories \
                         at {:?}: {}",
                        vol_dir, e
                    ))
                })?;
                for entry in entries {
                    let entry = entry.map_err(|e| {
                        Error::internal(format!(
                            "PRAGMA RESTORE: failed to read volumes directory entry: {}",
                            e
                        ))
                    })?;
                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        let path = entry.path();
                        std::fs::remove_dir_all(&path).map_err(|e| {
                            Error::internal(format!(
                                "PRAGMA RESTORE: failed to remove volume directory \
                                 {:?} (refusing to truncate WAL while a stale \
                                 table manifest survives on disk): {}",
                                path, e
                            ))
                        })?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = cleanup_result {
            self.enter_catastrophic_failure();
            self.latch_attach_gate_on_failure(restore_attach_gate.take());
            return Err(e);
        }

        // Destructive boundary cleared every old
        // `<volumes>/<table>/manifest.bin`; clear the
        // `pending_drop_cleanups` gate so a restored table
        // that happens to reuse a previously-pending name
        // doesn't keep the WAL-truncation gate held forever.
        // The directories that produced those entries are
        // gone (just verified above); any future DROP after
        // restore will repopulate the gate on its own.
        self.pending_drop_cleanups.lock().clear();

        // Reset WAL: old entries contain post-snapshot DML that would
        // overwrite restored data if replayed on next open.
        // Truncate with LSN=1 to create a fresh WAL starting from the beginning.
        // The post-restore checkpoint will re-record DDL with correct LSNs.
        //
        // Propagate truncate_wal failure: `WALManager::truncate_wal`
        // can return Err before / during the file swap and
        // restore the previous WAL on disk. Continuing past
        // that warning would let the next process open
        // replay pre-restore WAL entries OVER the restored
        // snapshot — exactly the case the comment above says
        // the reset prevents. The volume directories and
        // pending-drop set have already been wiped by the
        // earlier destructive boundary, so the database is
        // in a partially-restored state either way; failing
        // loudly forces the operator to retry rather than
        // proceeding to load the snapshot under a stale WAL.
        if let Err(e) = pm.truncate_wal(1) {
            // Catastrophic-failure latch: stores have been
            // closed, segment managers cleared, volume
            // directories removed, and `pending_drop_cleanups`
            // wiped — the registry's
            // `stop_accepting_transactions` blocks new user
            // txns, but auto-commit DDL paths bypass the
            // registry and only consult `ensure_writable` /
            // `record_ddl`'s catastrophic-failure check.
            // Without latching, an auto-commit CREATE / DROP
            // / TRUNCATE issued before the operator restarts
            // would mutate / publish DDL against this
            // half-destroyed restore state. Latch every
            // durability path off so the operator must
            // restart the process; recovery then converges
            // from whatever volume state did make it to disk
            // (i.e. effectively nothing — the operator
            // retries restore against a clean directory).
            // Also stash the EX startup-gate guard so
            // cross-process readers attempting to attach
            // block on its SH side until process exit
            // rather than loading the partially-destroyed
            // on-disk state.
            self.enter_catastrophic_failure();
            self.latch_attach_gate_on_failure(restore_attach_gate.take());
            return Err(Error::internal(format!(
                "PRAGMA RESTORE: WAL reset failed after volume directories were \
                 cleared: {} — engine latched into catastrophic-failure state. \
                 Restart the process and retry the restore.",
                e
            )));
        }

        // Clear all in-memory state
        self.schemas.write().unwrap().clear();
        self.version_stores.write().unwrap().clear();
        self.views.write().unwrap().clear();
        self.txn_version_stores.write().unwrap().clear();
        self.snapshot_timestamps.write().unwrap().clear();
        {
            let mut fk_cache = self.fk_reverse_cache.write().unwrap();
            *fk_cache = (u64::MAX, crate::common::StringMap::default());
        }

        // === Load snapshot data ===
        let mut total_tables = 0u32;
        let mut total_rows = 0u64;

        for (table_name, snapshot_path) in &snapshot_files {
            let file_size = std::fs::metadata(snapshot_path)
                .map(|m| m.len())
                .unwrap_or(0);
            let vol_path = snapshot_path.with_extension("vol");
            let use_volume = vol_path.exists() || file_size > 16 * 1024 * 1024;

            let load_result = if use_volume {
                self.load_table_snapshot_as_volume(table_name, snapshot_path)
            } else {
                self.load_table_snapshot(table_name, snapshot_path)
            };

            match load_result {
                Ok(_source_lsn) => {
                    total_tables += 1;
                    if let Ok(stores) = self.version_stores.read() {
                        if let Some(store) = stores.get(table_name.as_str()) {
                            total_rows += store.committed_row_count() as u64;
                        }
                    }
                    // Also count rows in segment managers (loaded as volume)
                    if let Ok(mgrs) = self.segment_managers.read() {
                        if let Some(mgr) = mgrs.get(table_name.as_str()) {
                            total_rows += mgr.total_row_count() as u64;
                        }
                    }
                }
                Err(e) => {
                    // Partial restore failure: some tables loaded, others not.
                    // The old volume tree is gone and the WAL was reset to LSN=1,
                    // so neither the original database nor the full requested
                    // snapshot can be reconstructed by recovery from this state.
                    // Persist what we have so a crash doesn't lose everything,
                    // then LATCH the engine — letting `start_accepting_transactions`
                    // run here would let new auto-commit DDL / DML append WAL
                    // on top of a partial restore that recovery can't reason
                    // about. Same fail-closed policy as the later DDL re-record
                    // and forced-checkpoint failure branches.
                    if total_tables > 0 {
                        eprintln!(
                            "Warning: partial restore ({} tables loaded), persisting before error return",
                            total_tables
                        );
                        // Drop checkpoint_mutex before calling checkpoint_cycle_inner
                        // to avoid deadlock (it acquires the same mutex internally).
                        drop(_checkpoint_guard);
                        let _ = self.checkpoint_cycle_inner(true);
                    } else {
                        drop(_checkpoint_guard);
                    }
                    self.enter_catastrophic_failure();
                    self.latch_attach_gate_on_failure(restore_attach_gate.take());
                    return Err(Error::internal(format!(
                        "PRAGMA RESTORE: failed to restore table '{}': {} — engine \
                         latched into catastrophic-failure state. Restart the \
                         process and retry the restore against the snapshot.",
                        table_name, e
                    )));
                }
            }
        }

        // === Post-restore ===

        // Recreate indexes and views from ddl.bin or fallback saved state.
        // Treat ddl.bin corruption as fatal: UNIQUE / FK
        // enforcement is driven from the restored indexes,
        // so silently skipping a corrupted entry would make
        // RESTORE report success against a database that
        // can't enforce its constraints. Latch the engine —
        // the destructive boundary is past, so reopening
        // writes against a partial DDL load could append WAL
        // mutating tables whose indexes are missing.
        if let Some(ref data) = ddl_data {
            if let Err(e) = self.restore_ddl_from_bin(data) {
                self.enter_catastrophic_failure();
                self.latch_attach_gate_on_failure(restore_attach_gate.take());
                return Err(Error::internal(format!(
                    "PRAGMA RESTORE: ddl.bin validation/restore failed: {} — \
                     engine latched into catastrophic-failure state. The \
                     restored snapshot is missing one or more indexes / views, \
                     so UNIQUE / FK enforcement would be silently broken. \
                     Restart the process and retry the restore against a \
                     valid snapshot.",
                    e
                )));
            }
        } else {
            // Fallback: recreate indexes from in-memory saved state
            let stores = self.version_stores.read().unwrap();
            for (table_name, index_meta) in &saved_indexes {
                if let Some(store) = stores.get(table_name.as_str()) {
                    if let Err(e) =
                        store.create_index_from_metadata_with_graph(index_meta, false, None)
                    {
                        eprintln!(
                            "Warning: Failed to recreate index '{}' on '{}': {}",
                            index_meta.name, table_name, e
                        );
                    }
                }
            }
            drop(stores);

            // Fallback: recreate views from in-memory saved state
            for (name, query) in &saved_views {
                let view_def = Arc::new(ViewDefinition::new(name, query.clone()));
                self.views
                    .write()
                    .unwrap()
                    .insert(name.to_lowercase(), view_def);
            }
        }

        // Populate default values from DEFAULT expressions
        self.populate_schema_defaults();

        // Sync auto-increment counters from segment managers.
        // Without this, tables loaded as volumes would have stale counters
        // and new INSERTs could collide with existing row IDs.
        self.sync_auto_increment_from_segments();

        // Increment schema epoch to invalidate all caches
        self.schema_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        // Re-record DDL to WAL so table schemas survive
        // WAL truncation. Treat failure as fatal for
        // restore: WAL has just been truncated to LSN=1, so
        // without these CreateTable / CreateIndex /
        // CreateView records the next process open finds an
        // empty WAL + manifests that may not yet reflect the
        // restored schemas (the forced checkpoint below has
        // not run yet, AND `checkpoint_cycle_inner` itself
        // logs-and-continues on its OWN `rerecord_ddl_to_wal`
        // failure). Returning success here would let the
        // caller believe the restore landed durably even
        // though restart could resurrect tables in their
        // pre-DDL-record-failure state.
        if let Err(e) = self.rerecord_ddl_to_wal() {
            // Catastrophic-failure latch: WAL was just
            // truncated to LSN=1, in-memory state holds the
            // restored snapshot, but `rerecord_ddl_to_wal`
            // may have written only PART of the schema DDL
            // before failing. Letting new writes proceed
            // would append WAL entries on top of a
            // half-recorded restore — recovery couldn't
            // reconstruct a consistent state. Latching the
            // engine forces every subsequent durability
            // path to refuse; the operator must restart,
            // and recovery converges from whichever DDL
            // entries did make it to disk + the
            // already-persisted volume manifests.
            //
            // Deliberately DO NOT call
            // `start_accepting_transactions` — the registry
            // stays in the stop state set above so even
            // non-durable in-memory operations refuse.
            self.enter_catastrophic_failure();
            self.latch_attach_gate_on_failure(restore_attach_gate.take());
            return Err(Error::internal(format!(
                "PRAGMA RESTORE: failed to re-record DDL to WAL after the WAL \
                 reset: {} — engine latched into catastrophic-failure state. \
                 Restart the process; recovery converges from the on-disk \
                 manifests + whatever DDL entries landed before the failure.",
                e
            )));
        }

        // Drop the checkpoint guard BEFORE calling checkpoint_cycle_inner
        // since it acquires the same mutex internally
        drop(_checkpoint_guard);

        // Release the compaction slot before the forced compaction call.
        // The destructive restore phase is complete — segment managers are
        // repopulated, so a compaction here is safe.
        drop(_compaction_guard);

        // Force checkpoint to persist restored data to volumes BEFORE accepting
        // transactions. WAL was truncated, so data is only in memory until this
        // checkpoint. If this fails, the restored data is non-durable — return
        // an error so the caller knows the restore did not complete safely.
        if let Err(e) = self.checkpoint_cycle_inner(true) {
            // Catastrophic-failure latch (same rationale as
            // the `rerecord_ddl_to_wal` branch above):
            // restored rows are only in memory because WAL
            // was truncated and this checkpoint failed to
            // persist them. Reopening writes here would
            // append new WAL on top of state recovery
            // cannot reconstruct. Restart converges from
            // whatever segments / manifests did land before
            // the failure.
            self.enter_catastrophic_failure();
            self.latch_attach_gate_on_failure(restore_attach_gate.take());
            return Err(Error::Internal {
                message: format!(
                    "PRAGMA RESTORE: forced checkpoint failed; restored data is \
                     non-durable: {} — engine latched into catastrophic-failure \
                     state. Restart the process to converge.",
                    e
                ),
            });
        }
        self.compact_after_checkpoint_forced();

        // Resume accepting transactions only after data is persisted
        self.registry.start_accepting_transactions();

        Ok(format!(
            "Restored {} tables ({} rows) from snapshot",
            total_tables, total_rows
        ))
    }

    /// Checkpoint-to-volume cycle: replaces the old snapshot-based persistence.
    ///
    /// Instead of serializing the entire hot buffer to a snapshot .bin file,
    /// this cycle seals committed hot rows into frozen volumes, re-records all DDL
    /// to WAL (so recovery can recreate schemas after WAL truncation), persists
    /// manifests, and optionally truncates the WAL.
    ///
    /// Recovery loads manifests + volumes from disk, replays WAL from 0 (skipping
    /// INSERT entries for rows already in volumes), and rebuilds hot state.
    ///
    /// WAL truncation is only safe when ALL committed hot rows have been sealed.
    /// Otherwise, unsealed rows' INSERT entries must survive in the WAL.
    fn checkpoint_cycle(&self) -> Result<()> {
        self.checkpoint_cycle_inner(false)?;
        // Run compaction synchronously so direct callers (Rust API,
        // trait users) get the full seal + compact behavior.
        // The background thread bypasses this by calling
        // checkpoint_cycle_inner + spawn_compaction directly.
        // Both paths reach `run_compaction_guarded`, which
        // drives `sweep_orphan_table_dirs` after compaction
        // — covers explicit AND background cycles.
        self.compact_after_checkpoint_forced();
        Ok(())
    }

    /// Reap orphan volume state in two passes, both gated on
    /// `!defer_for_live_readers()`:
    ///
    ///   1. Whole-table orphans: any `<volumes>/<dirname>/`
    ///      whose name is NOT in the engine's current
    ///      catalog (union of `schemas` and
    ///      `segment_managers`). Leftovers from DROP that
    ///      took the defer-on-live-readers path or from a
    ///      writer crash mid-DROP.
    ///   2. Per-file orphans for ACTIVE tables: any
    ///      `<volumes>/<table>/vol_<id>.vol` whose `<id>`
    ///      isn't in the table's current manifest. Leftovers
    ///      from TRUNCATE / compaction that took the
    ///      defer-on-live-readers path — the table is still
    ///      in the catalog so pass 1 wouldn't find it, but
    ///      its manifest no longer references the old
    ///      segment ids. Without this pass, a long-running
    ///      writer that truncates while readers are live
    ///      would accumulate dead `.vol` files indefinitely.
    fn sweep_orphan_table_dirs(&self) {
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return,
        };
        // Build the active-tables set + per-active-table
        // in-memory segment-id snapshot under read locks
        // BEFORE touching the filesystem so we don't hold any
        // lock while doing potentially slow FS work. The
        // active set is the union of `schemas` (covers empty
        // / hot-only tables) and `segment_managers` (covers
        // cold-only / loaded tables).
        let mut active: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        {
            let schemas = self.schemas.read().unwrap();
            for name in schemas.keys() {
                active.insert(name.clone());
            }
        }
        let mut per_table_in_memory: Vec<(String, rustc_hash::FxHashSet<u64>)> = Vec::new();
        {
            let mgrs = self.segment_managers.read().unwrap();
            for (name, mgr) in mgrs.iter() {
                active.insert(name.clone());
                let segs = mgr.segments_raw();
                let ids: rustc_hash::FxHashSet<u64> = segs.keys().copied().collect();
                per_table_in_memory.push((name.clone(), ids));
            }
        }
        let vol_dir = pm.path().join("volumes");

        // ALWAYS discover whole-table orphan dirs (dirs in
        // vol_dir whose lowercased name isn't in
        // `active`) and seed `pending_drop_cleanups` with
        // them — even when readers are live and we'll skip
        // the actual unlink below. The seed is what tells
        // `compute_wal_truncate_floor` to refuse WAL
        // truncation while the leftover `manifest.bin` is
        // still discoverable. Without this, a writer that
        // crashes mid-DROP and restarts with a still-live
        // reader would replay `DropTable` (clearing the
        // catalog), enter the deferred sweep below, return
        // immediately, and then the next checkpoint would
        // truncate WAL past the recovered DropTable while
        // the leftover manifest still resurfaces the
        // dropped table on the next reopen.
        // Enumerate `vol_dir` and re-seed `pending_drop_cleanups`
        // with on-disk orphan table dirs. NotFound (the volumes
        // dir doesn't exist on a fresh DB / no DDL yet) is a
        // legitimate empty result: no orphans to seed, clear
        // the gate flag and proceed. Any other error means we
        // could NOT prove the absence of orphans this cycle —
        // raise `orphan_discovery_failed` so
        // `compute_wal_truncate_floor` refuses truncation
        // until a future cycle succeeds. Per-entry errors are
        // similarly fatal for the cycle: a missed entry could
        // hide a leftover manifest.
        let read_result = std::fs::read_dir(&vol_dir);
        match read_result {
            Ok(entries) => {
                let mut pending = self.pending_drop_cleanups.lock();
                let mut entry_error = false;
                for entry in entries {
                    let entry = match entry {
                        Ok(e) => e,
                        Err(_) => {
                            entry_error = true;
                            break;
                        }
                    };
                    // file_type() failure is fatal for this
                    // cycle — same logic as entry errors above.
                    // If the un-stattable entry is actually a
                    // leftover `volumes/<table>/` directory with
                    // a `manifest.bin`, silently skipping it
                    // would clear `orphan_discovery_failed` and
                    // leave `pending_drop_cleanups` empty,
                    // letting a later checkpoint truncate past
                    // `DropTable` while the stale manifest is
                    // still discoverable.
                    let ft = match entry.file_type() {
                        Ok(ft) => ft,
                        Err(_) => {
                            entry_error = true;
                            break;
                        }
                    };
                    if !ft.is_dir() {
                        continue;
                    }
                    if let Some(name) = entry.file_name().to_str() {
                        let lower = name.to_lowercase();
                        if !active.contains(&lower) {
                            pending.insert(lower);
                        }
                    }
                }
                drop(pending);
                if entry_error {
                    self.orphan_discovery_failed.store(true, Ordering::Release);
                } else {
                    self.orphan_discovery_failed.store(false, Ordering::Release);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Fresh DB / no volumes yet: nothing to seed,
                // discovery is trivially complete.
                self.orphan_discovery_failed.store(false, Ordering::Release);
            }
            Err(_) => {
                self.orphan_discovery_failed.store(true, Ordering::Release);
            }
        }

        // While readers are live, we've recorded the
        // orphans in `pending_drop_cleanups` (so WAL
        // truncation refuses) but cannot actually unlink
        // anything — a reader's lazy `ensure_volume` may
        // still need the files. The next call (after
        // readers detach) will fall through to the actual
        // sweep below and clear the pending entries.
        if self.defer_for_live_readers() {
            return;
        }
        // The sweep returns a report with four independent
        // failure counters. Any non-zero error count raises
        // `orphan_discovery_failed`: the re-seed pass above
        // raised it for the SAME conditions, so this is just
        // belt-and-suspenders for the case where the sweep
        // sees an error the re-seed didn't (different
        // syscall, different timing). Without this, a
        // `remove_dir_all` failure on a known-orphan dir
        // would leave the dir on disk while the re-seed pass
        // had already cleared the flag.
        let report = crate::storage::volume::io::sweep_orphan_table_dirs(&vol_dir, &active);
        if report.read_dir_failed || report.entry_errors > 0 || report.remove_failures > 0 {
            self.orphan_discovery_failed.store(true, Ordering::Release);
        }
        // Drain `pending_drop_cleanups` entries whose
        // directory pass-1 just removed (or that some other
        // path cleaned in the meantime). The sweep already
        // unlinked them; this just clears the gate so
        // `compute_wal_truncate_floor` can advance.
        //
        // Use `try_exists()` rather than `exists()`: the latter
        // collapses metadata errors (permission, transient I/O)
        // to `false`, which would let an unreadable leftover
        // `volumes/<table>/` (manifest possibly still on disk)
        // clear the gate. On any stat error, KEEP the pending
        // mark AND raise `orphan_discovery_failed` so a later
        // sweep that can prove the directory is gone is the
        // only thing that lets WAL truncation advance.
        {
            let mut pending = self.pending_drop_cleanups.lock();
            let mut stat_failed = false;
            pending.retain(|name| match vol_dir.join(name).try_exists() {
                Ok(true) => true,   // still on disk → keep
                Ok(false) => false, // proven gone → drain
                Err(_) => {
                    stat_failed = true;
                    true // unknown → keep, gate stays held
                }
            });
            if stat_failed {
                self.orphan_discovery_failed.store(true, Ordering::Release);
            }
        }
        // Pass 2 — per-file sweep for active tables. The
        // keep-set is the UNION of (durable on-disk manifest
        // segments) AND (in-memory snapshot above):
        //   - Durable IDs guard against a compaction whose
        //     `mgr.replace_segments_atomic_*` ran but whose
        //     `persist_manifest_only` failed: in-memory has
        //     the new IDs only, but the durable manifest
        //     still references the OLD .vol files. Including
        //     the durable set in the keep-set prevents the
        //     sweep from deleting files the on-disk manifest
        //     still points at.
        //   - In-memory IDs guard the inverse: a successful
        //     compaction may have written new .vol files but
        //     not yet refreshed the on-disk manifest with
        //     them (the `persist` call is the next step). We
        //     keep those files so the next persist attempt
        //     finds them.
        // Files NOT in the union are truly orphan (compaction
        // leftovers, TRUNCATE leftovers from the
        // defer-on-live-readers path) and safe to delete.
        for (table, in_mem_ids) in &per_table_in_memory {
            let manifest_path = vol_dir.join(table).join("manifest.bin");
            // The per-file sweep needs BOTH in-memory and
            // durable segment IDs to be safe — see the
            // pass-2 comment above for the failure
            // scenarios. If `read_from_disk` returns Err
            // (transient I/O, corrupt header, partial
            // write) we cannot tell which old segment IDs
            // the on-disk manifest still references; with
            // only the in-memory IDs as the keep-set, a
            // post-`mgr.replace_segments_atomic_*` /
            // pre-`persist_manifest_only` compaction
            // would let this sweep delete the old `.vol`
            // files the still-durable manifest points at.
            // Skip the per-file sweep for this table —
            // wait for the next cycle when the manifest is
            // readable. Whole-table sweep (pass 1) is
            // unaffected because it doesn't depend on
            // per-table manifests.
            let mf = match crate::storage::volume::manifest::TableManifest::read_from_disk(
                &manifest_path,
            ) {
                Ok(mf) => mf,
                Err(_) => continue,
            };
            let mut keep = in_mem_ids.clone();
            for seg in &mf.segments {
                keep.insert(seg.segment_id);
            }
            // If pass 1 removed this dir (because the
            // segment manager just got cleared by an
            // in-flight DROP between our snapshot and the
            // pass-1 sweep), the per-file sweep harmlessly
            // returns 0.
            let _ =
                crate::storage::volume::io::sweep_orphan_volumes_in_table(&vol_dir, table, &keep);
        }
    }

    /// Inner checkpoint implementation. When `force` is true, seals ALL hot rows
    /// regardless of threshold (used by PRAGMA CHECKPOINT and close_engine).
    /// When false, respects the normal seal thresholds (used by background thread).
    fn checkpoint_cycle_inner(&self, force: bool) -> Result<()> {
        // Refuse the entire cycle once the engine is in the
        // catastrophic-failure state. seal_hot_buffers / compact_volumes
        // already early-return per their own latch checks, but the
        // surrounding cycle still re-records DDL, persists manifests,
        // bumps the manifest epoch, and truncates WAL — every one of
        // those is durable / cross-process state. The cycle is
        // reachable from the background checkpoint thread and from
        // close_engine; both must stop after the failed latch.
        if self.is_failed() {
            return Ok(());
        }
        if let Some(ref pm) = *self.persistence {
            if !pm.is_enabled() {
                return Ok(());
            }
        } else {
            return Ok(());
        }

        // Serialize the entire cycle: prevent concurrent seal+compact from
        // the background thread and explicit PRAGMA CHECKPOINT.
        let _checkpoint_guard = self.checkpoint_mutex.lock().unwrap();

        // Step 1: Seal hot rows into frozen volumes (the actual checkpoint).
        // Sealed rows are written to .vol files and removed from the hot buffer.
        // When force=true, bypass thresholds so ALL hot rows are sealed.
        if force {
            self.force_seal_all.store(true, Ordering::Release);
        }
        if let Err(e) = self.seal_hot_buffers() {
            eprintln!("Warning: seal_hot_buffers failed: {}", e);
        }
        if force {
            self.force_seal_all.store(false, Ordering::Release);
        }
        // Step 2: Force-seal any remaining small tables so all hot buffers
        // are empty. The first seal pass (Step 1) uses incremental thresholds
        // and may leave small tables (metrics, logs, etc.) unsealed. Without
        // draining them, all_hot_empty is never true and WAL never truncates.
        let all_hot_empty = {
            let stores = self.version_stores.read().unwrap();
            stores
                .values()
                .all(|store| store.committed_row_count() == 0)
        };

        if !all_hot_empty && !force {
            // Force-seal the stragglers (small tables below threshold)
            self.force_seal_all.store(true, Ordering::Release);
            if let Err(e) = self.seal_hot_buffers() {
                eprintln!("Warning: force-seal stragglers failed: {}", e);
            }
            self.force_seal_all.store(false, Ordering::Release);
        }

        // Step 3: Brief fence — block commits just long enough to check if all
        // hot buffers are empty and capture checkpoint_lsn. NO disk I/O inside
        // the fence. Previously this ran a full seal_hot_buffers() (with volume
        // building + disk writes) while blocking all commits, causing 1-2s INSERT
        // stalls. Now the fence is held for microseconds (atomic counter reads).
        // If hot buffers aren't empty after steps 1-2, we skip WAL truncation
        // this cycle and let the next cycle's bulk seal drain them.
        let checkpoint_lsn = match self
            .seal_fence
            .try_write_for(std::time::Duration::from_secs(15))
        {
            Some(_fence) => {
                // Recheck the latch AFTER the seal/fence wait. The
                // entry-time check passed; the wait above (seal_hot_buffers
                // + acquiring the seal_fence write lock) can block while
                // a marker-failure commit completes its drain, latches
                // the engine, and drops its own seal_fence read guard
                // — handing us the write lock. Without this recheck
                // we'd call create_checkpoint, persist manifests, bump
                // the manifest epoch, and truncate WAL after the
                // catastrophic-failure latch was set.
                if self.is_failed() {
                    return Ok(());
                }
                // Fence acquired — no new commits can start. In-flight commits
                // finished (they held the read lock, which is now released).
                // Just check if bulk seal (steps 1-2) drained everything.
                let all_hot_empty = {
                    let stores = self.version_stores.read().unwrap();
                    stores
                        .values()
                        .all(|store| store.committed_row_count() == 0)
                };

                if all_hot_empty {
                    // All data is in volumes. Safe to advance the WAL checkpoint.
                    if let Some(ref pm) = *self.persistence {
                        pm.create_checkpoint(vec![])?
                    } else {
                        0
                    }
                } else {
                    // Continuous writes kept hot buffers non-empty.
                    // Skip WAL truncation — next cycle's bulk seal will drain them.
                    0
                }
                // _fence dropped here — commits resume
            }
            None => {
                eprintln!("Warning: seal_fence timeout, skipping WAL truncation");
                0
            }
        };

        // Re-record DDL OUTSIDE the fence. DDL writes to WAL (append-only,
        // thread-safe) and doesn't need to block commits.
        if checkpoint_lsn > 0 {
            if let Err(e) = self.rerecord_ddl_to_wal() {
                eprintln!("Warning: Failed to re-record DDL to WAL: {}", e);
                return Ok(());
            }
        }

        // Step 4: Persist manifests (includes tombstones + checkpoint_lsn).
        // checkpoint_lsn is > 0 only when all hot buffers are empty (fully sealed).
        // Track success: WAL truncation is only safe if ALL manifests are durable.
        let mut all_manifests_persisted = true;
        {
            // Collect Arc clones under the read lock, then drop the lock
            // before doing file I/O to avoid blocking writers.
            let mgr_arcs: Vec<_> = {
                let mgrs = self.segment_managers.read().unwrap();
                mgrs.values().cloned().collect()
            };
            for mgr in &mgr_arcs {
                if checkpoint_lsn > 0 {
                    mgr.manifest_mut().checkpoint_lsn = checkpoint_lsn;
                }
                if let Err(e) = mgr.persist_manifest_only() {
                    eprintln!(
                        "Warning: Failed to persist manifest for {}: {}",
                        mgr.table_name(),
                        e
                    );
                    all_manifests_persisted = false;
                }
            }
        }

        // Step 4b: Bump the cross-process manifest epoch. Reader processes
        // poll `<db>/volumes/epoch` to detect new checkpoints and reload
        // their cached manifests on advance. We only bump when ALL
        // manifests are durable so a reader that observes epoch=N is
        // guaranteed every per-table manifest.bin is at the post-checkpoint
        // state. Skip on memory engines (no path) and when the fence
        // didn't acquire (checkpoint_lsn==0).
        if checkpoint_lsn > 0 && all_manifests_persisted && !self.path.is_empty() {
            if let Err(e) =
                crate::storage::mvcc::manifest_epoch::bump_epoch(std::path::Path::new(&self.path))
            {
                eprintln!("Warning: Failed to bump manifest epoch: {}", e);
            }
        }

        // Step 5: Truncate WAL BEFORE compaction.
        // WAL truncation only depends on seal + manifest persist, not compaction.
        // Running compaction first (30+ seconds) delays truncation and extends
        // the checkpoint window unnecessarily.
        // checkpoint_lsn > 0 already guarantees all_hot_empty was true inside
        // the fence. Don't re-check the outer all_hot_empty (stale, checked
        // before fence when continuous inserts make it always false).
        //
        // SWMR v2 Phase D: floor the truncate at `min_pinned_lsn - 1` so a
        // reader tailing the WAL never finds the entries it needs gone.
        // `min_pinned_lsn` returns None when no v2 reader is attached (v1
        // leases are zero-byte and don't constrain), so the typical case
        // is just `truncate_wal(checkpoint_lsn)`.
        if checkpoint_lsn > 0 && all_manifests_persisted {
            if let Some(ref pm) = *self.persistence {
                let truncate_floor = self.compute_wal_truncate_floor(checkpoint_lsn);
                if let Some(floor) = truncate_floor {
                    if let Err(e) = pm.truncate_wal(floor) {
                        eprintln!("Warning: Failed to truncate WAL: {}", e);
                    }
                }
                // Whether or not we actually truncated, publish the
                // current min_pinned_lsn to db.shm so monitoring (PRAGMA
                // SWMR_STATUS) reflects reality.
                self.publish_min_pinned_lsn();
            }
        }

        Ok(())
    }

    /// Run compaction synchronously under the compaction_running flag.
    /// The flag prevents concurrent compaction from background and forced
    /// callers. Clears the flag on exit (including panics via drop guard).
    ///
    /// Also drives the orphan-volume sweep AFTER compaction
    /// completes — covers BOTH the synchronous
    /// `compact_after_checkpoint_forced` path (explicit
    /// `PRAGMA CHECKPOINT` / close / restore) and the
    /// `spawn_compaction` background thread spawned from the
    /// auto-checkpoint loop. Without the sweep here, a
    /// long-running writer that only relies on background
    /// checkpoints never reaps DROP / TRUNCATE leftovers
    /// until an explicit checkpoint or process restart.
    /// `sweep_orphan_table_dirs` is itself gated on
    /// `!defer_for_live_readers()`, so the call is a cheap
    /// no-op when readers are still attached.
    fn run_compaction_guarded(&self) {
        let _guard = AtomicBoolGuard(&self.compaction_running);
        if let Err(e) = self.compact_volumes() {
            eprintln!("Warning: compact_volumes failed: {}", e);
        }
        self.sweep_orphan_table_dirs();
    }

    /// Evict idle volume data to save memory. Volumes not accessed since the
    /// last epoch transition: hot → warm (drop decompressed) → cold (drop compressed).
    #[cfg(not(target_arch = "wasm32"))]
    fn evict_idle_volumes(&self) {
        // Drive eviction directly from `GLOBAL_EVICTION_EPOCH` rather
        // than a per-engine counter. Volumes are stamped with
        // `last_access_epoch = GLOBAL_EVICTION_EPOCH.load(...)` on
        // load and on tier transitions (writer.rs:1570, 1961, 1977).
        // If we ticked a per-engine counter that started at 0, a
        // volume freshly loaded into warm could carry a global stamp
        // way ahead of our local epoch (any other live engine in the
        // process advances global), and `current - last_access`
        // saturating_sub would stay 0 for thousands of ticks until
        // local catches up — silently delaying warm→cold demotion
        // and pinning RAM. Sourcing from the global counter keeps
        // the comparison monotonic across all engines in the process.
        let epoch = crate::storage::volume::writer::GLOBAL_EVICTION_EPOCH
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        let mgrs = self.segment_managers.read().unwrap();
        for mgr in mgrs.values() {
            mgr.evict_idle_volumes(epoch);
        }
    }

    /// Spawn compaction on a background thread. If compaction is already
    /// running, this is a no-op (the next checkpoint cycle will retry).
    /// Called from the background cleanup thread which owns Arc<Self>.
    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_compaction(self: &Arc<Self>) {
        // Try to claim the compaction slot. If already running, skip
        // compaction but still run eviction on this thread.
        if self
            .compaction_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.evict_idle_volumes();
            return;
        }
        let engine = Arc::clone(self);
        std::thread::spawn(move || {
            engine.run_compaction_guarded();
            engine.evict_idle_volumes();
        });
    }

    /// Run compaction synchronously, waiting for any in-flight background
    /// compaction to finish first. Used by PRAGMA CHECKPOINT, close_engine,
    /// restore, and v0.3.7 migration.
    fn compact_after_checkpoint_forced(&self) {
        // Claim the compaction slot, waiting for any background compaction.
        // CAS loop: if background thread holds the flag, spin until it clears.
        // Once we claim it, no background thread can start a new compaction.
        while self
            .compaction_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.run_compaction_guarded();
    }

    /// Re-record all DDL entries to WAL after checkpoint.
    ///
    /// After WAL truncation, any CreateTable/CreateIndex/CreateView entries before
    /// the checkpoint LSN are lost. Re-recording them places fresh entries at
    /// the WAL head so they survive truncation and are replayed on recovery.
    ///
    /// Order matters: CreateTable must come before CreateIndex for the same table,
    /// because index replay needs the version store to exist.
    fn rerecord_ddl_to_wal(&self) -> Result<()> {
        // Take the EXCLUSIVE transactional-DDL fence so this
        // snapshot of `schemas` / `version_stores` only
        // observes COMMITTED catalog state. Any open
        // transaction that has performed a CREATE / DROP
        // holds the SHARED side until it commits / rolls
        // back, so this write call blocks until every such
        // txn resolves. Without this gate, an open
        // transactional CREATE would be republished as a
        // durable DDL_TXN_ID auto-commit (which a later
        // rollback can't retract), and an open transactional
        // DROP would be omitted from the re-record set
        // (leaving WAL with no CreateTable for the still-live
        // table after the next WAL truncation).
        let _ddl_fence = self.transactional_ddl_fence.write();

        // Collect CreateTable entries and table names in a single schemas lock
        let (table_entries, table_names_for_indexes): (Vec<(String, Vec<u8>)>, Vec<String>) = {
            let schemas = self.schemas.read().unwrap();
            let entries = schemas
                .values()
                .map(|schema| (schema.table_name.clone(), Self::serialize_schema(schema)))
                .collect();
            let names = schemas.keys().cloned().collect();
            (entries, names)
        };

        // Collect CreateIndex entries under version_stores lock, then drop
        let index_entries: Vec<(String, Vec<u8>)> = {
            let stores = self.version_stores.read().unwrap();
            let mut entries = Vec::new();
            for table_name in &table_names_for_indexes {
                if let Some(store) = stores.get(table_name) {
                    let _ = store.for_each_index(|index| {
                        // Skip PK indexes, they are auto-created from schema
                        if index.index_type() == crate::core::IndexType::PrimaryKey {
                            return Ok(());
                        }
                        let index_meta = super::persistence::IndexMetadata {
                            name: index.name().to_string(),
                            table_name: table_name.clone(),
                            column_names: index.column_names().to_vec(),
                            column_ids: index.column_ids().to_vec(),
                            data_types: index.data_types().to_vec(),
                            is_unique: index.is_unique(),
                            index_type: index.index_type(),
                            hnsw_m: index.hnsw_m(),
                            hnsw_ef_construction: index.hnsw_ef_construction(),
                            hnsw_ef_search: index.default_ef_search().map(|v| v as u16),
                            hnsw_distance_metric: index.hnsw_distance_metric(),
                        };
                        entries.push((table_name.clone(), index_meta.serialize()));
                        Ok(())
                    });
                }
            }
            entries
        };

        // Collect CreateView entries under views lock, then drop
        let view_entries: Vec<(String, Vec<u8>)> = {
            let views = self.views.read().unwrap();
            views
                .iter()
                .map(|(name, view_def)| (name.clone(), view_def.serialize()))
                .collect()
        };

        // Re-records use the publishing `record_ddl` path
        // because the writer's WAL truncate floor relies on
        // the invariant `chain_head <= db.shm.visible_commit_lsn`
        // — without publishing, writing rerecord bytes at LSNs
        // higher than the current published visible would let
        // a future checkpoint truncate WAL up to past the
        // visible LSN, and a fresh read-only attach (which
        // pins at `attach_visible_commit_lsn`) would
        // immediately trip `SwmrSnapshotExpired` because its
        // pin would be below `chain_head`.
        //
        // Spurious DDL detection on the reader side is
        // suppressed by the existing `known_catalog_objects` /
        // `known_index_names` filter in
        // `ReadOnlyDatabase::refresh`: a re-record of a
        // catalog object the reader already knows about is
        // not surfaced as `SwmrPendingDdl`, so the visibility
        // bump itself is harmless for steady-state schemas.
        // Write CreateTable entries first (schemas must exist before indexes)
        for (table_name, data) in &table_entries {
            self.record_ddl(table_name, WALOperationType::CreateTable, data)?;
        }

        for (table_name, data) in &index_entries {
            self.record_ddl(table_name, WALOperationType::CreateIndex, data)?;
        }

        for (view_name, data) in &view_entries {
            self.record_ddl(view_name, WALOperationType::CreateView, data)?;
        }

        Ok(())
    }

    fn compact_volumes(&self) -> Result<()> {
        // See `seal_hot_buffers` for rationale: compaction can
        // physically remove rows and rewrite manifests; running it
        // while a markerless commit's data sits in parent stores
        // would let those rows survive into the rewritten segments.
        if self.is_failed() {
            return Ok(());
        }
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()),
        };

        let (compact_threshold, target_volume_rows) = self
            .config
            .read()
            .map(|c| {
                (
                    c.persistence.compact_threshold as usize,
                    c.persistence.target_volume_rows,
                )
            })
            .unwrap_or((4, 1_048_576));
        let compact_threshold = compact_threshold.max(2);

        let tables_to_compact: Vec<String> = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.iter()
                .filter(|(_, mgr)| {
                    // Only count sub-target volumes toward threshold.
                    let sub_target = mgr.sub_target_segment_count(target_volume_rows);
                    if sub_target > compact_threshold {
                        return true;
                    }
                    // Sub-target volumes exist AND tombstones exist: the sub-target
                    // volumes likely contain new versions of tombstoned rows.
                    // Compact to merge new versions back into at-target volumes.
                    if sub_target >= 2 && !mgr.is_tombstone_set_empty() {
                        return true;
                    }
                    // Tombstones exist even without sub-target volumes (pure DELETE).
                    // At-target volumes with tombstoned rows need cleanup.
                    if sub_target == 0 && !mgr.is_tombstone_set_empty() && mgr.segment_count() >= 1
                    {
                        return true;
                    }
                    // Split oversized volumes that exceed 150% of target.
                    let oversized_threshold = target_volume_rows * 3 / 2;
                    if mgr.max_segment_row_count() > oversized_threshold {
                        return true;
                    }
                    false
                })
                .map(|(name, _)| name.clone())
                .collect()
        };

        if tables_to_compact.is_empty() {
            return Ok(());
        }

        for table_name in &tables_to_compact {
            // Recheck the catastrophic-failure latch per-table.
            // Candidate selection above only checked the latch at
            // function entry; a failing commit can complete its
            // in-memory drain and latch the engine while we're
            // iterating. Compaction reads tombstones and rewrites
            // segments, so proceeding could materialize markerless-
            // commit deletes (or merge in markerless-commit data via
            // hot-row visibility carryover paths) into a replacement
            // segment that survives recovery.
            if self.is_failed() {
                return Ok(());
            }
            let schema = {
                let schemas = self.schemas.read().unwrap();
                match schemas.get(table_name) {
                    Some(s) => s.clone(),
                    None => continue,
                }
            };

            let mgr = self.get_or_create_segment_manager(table_name);

            // Per-table snapshot gating: capture the current min snapshot begin_seq
            // for each table to close the TOCTOU window. A snapshot starting between
            // tables must not cause earlier compacted tables to lose visible rows.
            let compact_seal_seq_limit =
                self.registry.get_min_snapshot_begin_seq().map(|s| s as u64);

            // Targeted compaction: only rewrite volumes that need work.
            // At-target volumes are left untouched to minimize disk I/O.
            //
            // Categories:
            // - Sub-target (< target_volume_rows): small volumes to merge together
            // - Oversized (> target * 3/2): large volumes to split
            // - At-target: properly sized, never rewrite
            //
            // SWMR v2 visibility-frontier constraint: when live readers
            // exist, restrict compaction to segments whose
            // `visible_at_lsn <= min_pinned_reader_lsn`. The output
            // segment is stamped with `MAX(inputs.visible_at_lsn)`, so
            // a reader pinned at `L` either sees ALL merged inputs
            // (cap >= MAX) or NONE (cap < MIN). Mixing inputs from
            // different visibility frontiers across a live reader's
            // cap would let the compacted segment expose rows that
            // weren't visible to that reader before. The pin is a
            // conservative proxy for the reader's effective cap (pin
            // <= cap typically), so this rule is correctness-safe.
            //
            // No live readers (None or u64::MAX) → unrestricted.
            let visibility_cap = self.min_pinned_reader_lsn().unwrap_or(u64::MAX);
            let (old_ids, volumes, tombstones, old_visible_at_lsn_max, tombstone_visible_at) = {
                // Use segments_raw for planning — only metadata (row_ids,
                // row_count) is needed. Avoids reloading ALL cold volumes
                // for tables where only sub-target volumes need compaction.
                let segs = mgr.segments_raw();
                let manifest = mgr.manifest();
                let ts = mgr.tombstone_set_arc();

                let oversized_threshold = target_volume_rows * 3 / 2;

                let mut merge_indices: Vec<usize> = Vec::new();
                for (idx, seg) in manifest.segments.iter().enumerate() {
                    // Skip volumes sealed after the earliest snapshot began.
                    // seal_seq = cutoff used during extraction. A volume with
                    // seal_seq <= limit contains only pre-snapshot data (safe).
                    // seal_seq > limit means the volume may have post-snapshot data.
                    if let Some(limit) = compact_seal_seq_limit {
                        if seg.seal_seq > 0 && seg.seal_seq > limit {
                            continue;
                        }
                    }
                    // Cross-process SWMR: skip segments whose
                    // visibility frontier is above the lowest live
                    // reader's pin. Including them would force MAX
                    // of inputs to exceed `visibility_cap`, which
                    // would put a live reader strictly between the
                    // merged inputs and let the compacted segment
                    // expose rows the reader never had visibility to.
                    if seg.visible_at_lsn > visibility_cap {
                        continue;
                    }
                    if seg.row_count < target_volume_rows {
                        // Sub-target: merge together to reach target size
                        merge_indices.push(idx);
                    } else if seg.row_count > oversized_threshold {
                        // Oversized: needs splitting
                        merge_indices.push(idx);
                    } else if !ts.is_empty() {
                        // At-target: include only if it has tombstoned rows that
                        // compaction can actually apply. When snapshots are active,
                        // only count tombstones with commit_seq < limit (post-snapshot
                        // tombstones will be preserved, so rewriting is pointless).
                        if let Some(cs) = segs.get(&seg.segment_id) {
                            let tombstone_count = cs
                                .volume
                                .meta
                                .row_ids
                                .iter()
                                .filter(|rid| {
                                    if let Some(limit) = compact_seal_seq_limit {
                                        ts.get(rid).is_some_and(|&commit_seq| commit_seq < limit)
                                    } else {
                                        ts.contains_key(rid)
                                    }
                                })
                                .count();
                            if tombstone_count > 0 {
                                merge_indices.push(idx);
                            }
                        }
                    }
                    // At-target with no tombstones: frozen, don't touch
                }

                if merge_indices.is_empty() {
                    continue;
                }
                // Single sub-target volume (too small, not dirty): wait for
                // more to accumulate before merging.
                // A single at-target/oversized volume with tombstones should
                // NOT be skipped — it needs rewriting to remove dead rows.
                if merge_indices.len() == 1 {
                    let seg = &manifest.segments[merge_indices[0]];
                    if seg.row_count < target_volume_rows {
                        continue; // small volume, wait for more
                    }
                }

                // Load only the merge-candidate volumes (not the entire table).
                // Cold volumes are loaded on demand via ensure_volume.
                let old_ids: Vec<u64> = merge_indices
                    .iter()
                    .map(|&i| manifest.segments[i].segment_id)
                    .collect();
                // Capture the MAX visible_at_lsn across the segments
                // we're about to replace. The compacted output's
                // visible_at_lsn must equal this MAX: the merge gate
                // above guarantees `MAX(inputs) <= visibility_cap`,
                // so every live reader has a cap >= MAX and sees ALL
                // merged inputs (or none, if their cap is below the
                // MIN — but their cap also being above MAX would be
                // impossible only if their cap is <visibility_cap,
                // which contradicts pin <= cap).
                //
                // MIN would be wrong: it would put the replacement's
                // visibility frontier BELOW the highest input's
                // frontier, exposing rows from the higher input to a
                // reader whose cap was strictly between MIN and MAX
                // (which the merge gate has already ruled out for
                // live readers, but MIN would still misrepresent the
                // true visibility of the rows the replacement
                // contains).
                let old_visible_at_lsn_max: u64 = merge_indices
                    .iter()
                    .map(|&i| manifest.segments[i].visible_at_lsn)
                    .max()
                    .unwrap_or(0);
                let mut vols: Vec<(u64, Arc<crate::storage::volume::writer::FrozenVolume>)> =
                    merge_indices
                        .iter()
                        .filter_map(|&i| {
                            let seg = &manifest.segments[i];
                            let vol = segs.get(&seg.segment_id)?;
                            if vol.volume.is_cold() {
                                // Load only this merge candidate from disk.
                                let loaded = mgr.ensure_volume(seg.segment_id)?;
                                Some((seg.segment_id, loaded))
                            } else {
                                Some((seg.segment_id, Arc::clone(&vol.volume)))
                            }
                        })
                        .collect();
                drop(segs);

                // Every manifest entry must have a loaded volume.
                if vols.len() != old_ids.len() {
                    continue;
                }
                // Sort by segment_id descending (newest first) for correct dedup.
                vols.sort_by_key(|entry| std::cmp::Reverse(entry.0));
                let ts = mgr.tombstone_set_arc();
                // Capture per-tombstone visibility frontier from the
                // manifest (V8+: real visible_at_lsn; V7: synthesized
                // 0 = always visible). Used below to skip tombstones
                // newer than the lowest live reader's pin so
                // compaction can't physically remove a row that's
                // still visible to a capped reader.
                let ts_vis: rustc_hash::FxHashMap<i64, u64> = manifest
                    .tombstones
                    .iter()
                    .map(|&(rid, _, vis)| (rid, vis))
                    .collect();
                (old_ids, Arc::new(vols), ts, old_visible_at_lsn_max, ts_vis)
            };

            // Streaming compaction: iterate volumes newest-first, dedup by row_id,
            // collect only (row_id, volume_index, row_index) references, then sort
            // and stream into VolumeBuilder. This avoids materializing all Row objects
            // at once (which would be 2-3x the table size for large tables).
            let mut seen = FxHashSet::default();
            let mut live_refs: Vec<(i64, usize, usize)> = Vec::new(); // (row_id, vol_idx, row_idx)

            // When snapshots are active, build a filtered tombstone set containing
            // only tombstones that were actually applied (commit_seq < limit).
            // Post-snapshot tombstones are preserved so snapshot reads stay correct.
            //
            // Cross-process SWMR: ALSO filter out tombstones whose
            // `visible_at_lsn > visibility_cap`. Compaction physically
            // removes tombstoned rows, so applying a tombstone the
            // lowest live reader hasn't observed yet would silently
            // remove a row that's still visible at that reader's cap.
            // Tombstones with `visible_at_lsn = 0` (V7-loaded /
            // recovery-rebuilt) pass trivially since 0 <= any cap.
            let applied_tombstones: Arc<FxHashMap<i64, u64>> = {
                let filtered: FxHashMap<i64, u64> = tombstones
                    .iter()
                    .filter(|(_, &seq)| {
                        compact_seal_seq_limit
                            .map(|limit| seq < limit)
                            .unwrap_or(true)
                    })
                    .filter(|(rid, _)| {
                        let vis = tombstone_visible_at.get(rid).copied().unwrap_or(0);
                        // u64::MAX is the ephemeral sentinel used by
                        // partial-commit failure paths — never
                        // materialize these into a replacement segment,
                        // even when no live readers exist (visibility_cap
                        // == u64::MAX). Materializing would silently
                        // remove cold rows for a markerless commit.
                        vis != u64::MAX && vis <= visibility_cap
                    })
                    .map(|(&rid, &seq)| (rid, seq))
                    .collect();
                Arc::new(filtered)
            };

            for (vol_idx, (_seg_id, vol)) in volumes.iter().enumerate() {
                for i in 0..vol.meta.row_count {
                    let row_id = vol.meta.row_ids[i];
                    // Apply tombstone only if it (a) was committed before the
                    // earliest snapshot (safe to physically remove for snapshot
                    // isolation), AND (b) is visible to every live cross-process
                    // reader. Tombstones not meeting both conditions are
                    // preserved — the row stays in the merged volume so older
                    // snapshots / capped readers can still see it via versioned
                    // tombstone / segment filtering.
                    let snapshot_safe = compact_seal_seq_limit
                        .map(|limit| {
                            tombstones
                                .get(&row_id)
                                .is_some_and(|&commit_seq| commit_seq < limit)
                        })
                        .unwrap_or_else(|| tombstones.contains_key(&row_id));
                    let row_vis = tombstone_visible_at.get(&row_id).copied().unwrap_or(0);
                    // Same ephemeral sentinel guard as the
                    // applied_tombstones filter above: u64::MAX
                    // means a failed-marker tombstone that must
                    // never physically remove a cold row.
                    let visible_to_all_readers = row_vis != u64::MAX && row_vis <= visibility_cap;
                    let is_tombstoned = snapshot_safe && visible_to_all_readers;
                    if is_tombstoned && !seen.contains(&row_id) {
                        continue;
                    }
                    if seen.insert(row_id) {
                        live_refs.push((row_id, vol_idx, i));
                    }
                }
            }

            live_refs.sort_unstable_by_key(|(id, _, _)| *id);

            if live_refs.is_empty() {
                // All rows in merged volumes are tombstoned. Remove those
                // volumes and their tombstones, but keep unmerged volumes intact.
                let merged_row_ids: FxHashSet<i64> = volumes
                    .iter()
                    .flat_map(|(_, vol)| vol.meta.row_ids.iter().copied())
                    .collect();
                mgr.replace_segments_atomic_remove_only(&old_ids);
                // Only clear tombstones that were actually applied during compaction.
                // Post-snapshot tombstones are preserved for snapshot visibility.
                mgr.remove_tombstones_matching_snapshot(&applied_tombstones, &merged_row_ids);

                // Persist manifest BEFORE deleting files (same safety as non-empty path).
                if let Err(e) = mgr.persist_manifest_only() {
                    eprintln!(
                        "Warning: Failed to persist manifest after compaction for {}: {}",
                        table_name, e
                    );
                    continue;
                }

                // Defer unlink while cross-process readers are live —
                // mirrors the non-empty compaction branch below. The
                // new manifest is durable, but readers that haven't
                // yet refreshed still hold the OLD manifest in memory
                // and reference these volumes. Without this guard, an
                // all-tombstoned compaction could pull the .vol files
                // out from under a snapshot reader still scanning
                // them. Orphan files left behind here are harmless
                // (the new manifest doesn't reference them) and a
                // future compaction cycle reaps them once readers
                // detach.
                if self.defer_for_live_readers() {
                    continue;
                }

                let vol_dir = pm.path().join("volumes");
                let vol_table_dir = vol_dir.join(table_name);
                let old_filenames: FxHashSet<String> = old_ids
                    .iter()
                    .map(|id| format!("vol_{:016x}.vol", id))
                    .collect();
                if let Ok(entries) = std::fs::read_dir(&vol_table_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                            if old_filenames.contains(fname) {
                                let _ = std::fs::remove_file(&path);
                            }
                        }
                    }
                }
                continue;
            }

            // Build compacted volumes, splitting at target_volume_rows boundary.
            // Only one Row is materialized at a time, avoiding O(N) memory.
            let vol_dir = pm.path().join("volumes");
            let compress = self
                .config
                .read()
                .map(|c| c.persistence.volume_compression)
                .unwrap_or(true);

            // Precompute column mapping per volume (once each, not per row).
            let vol_mappings: Vec<crate::storage::volume::writer::ColumnMapping> = volumes
                .iter()
                .map(|(seg_id, _vol)| mgr.get_volume_mapping(*seg_id, &schema))
                .collect();

            // Split live_refs into target-sized chunks and build one volume per chunk.
            // Row-group aligned split: round to 64K boundary so every volume
            // has complete row groups. Optimal for LZ4 compression and zone maps.
            let row_group_size = 65_536usize;
            let chunk_size = (target_volume_rows / row_group_size).max(1) * row_group_size;
            let mut new_volumes: Vec<(
                u64,
                Arc<crate::storage::volume::writer::FrozenVolume>,
                crate::storage::volume::manifest::SegmentMeta,
            )> = Vec::new();
            let mut write_failed = false;

            // visible_at_lsn for compacted output: inherit the MIN
            // across the merged segments, NOT the writer's current
            // WAL LSN. The compacted rows were already visible at
            // each merged segment's `visible_at_lsn`, so the
            // compaction must not raise that floor — otherwise a
            // capped reader at P (where old_min <= P < current_lsn)
            // would drop the new segment even though the same rows
            // were visible to it through the merged-out segments.
            // After the old WAL range is truncated, the reader
            // could not reconstruct them.
            let visible_lsn = old_visible_at_lsn_max;

            for chunk in live_refs.chunks(chunk_size) {
                let mut builder = crate::storage::volume::writer::VolumeBuilder::with_capacity(
                    &schema,
                    chunk.len(),
                );
                for &(row_id, vol_idx, row_idx) in chunk {
                    let vol = &volumes[vol_idx].1;
                    let mapping = &vol_mappings[vol_idx];
                    let row = if mapping.is_identity {
                        vol.get_row(row_idx)
                    } else {
                        vol.get_row_mapped(row_idx, mapping)
                    };
                    builder.add_row(row_id, &row);
                }
                let mut compacted = builder.finish();

                let compact_vol_id = crate::storage::volume::io::next_volume_id();
                match crate::storage::volume::io::write_volume_to_disk_opts(
                    &vol_dir,
                    table_name,
                    compact_vol_id,
                    &compacted,
                    compress,
                ) {
                    Ok((_path, store)) => {
                        // Retain compressed store for hot→warm eviction.
                        compacted.columns.attach_compressed_store(store);
                        // Pre-build unique hash indices before registration.
                        {
                            let stores = self.version_stores.read().unwrap();
                            if let Some(store) = stores.get(table_name) {
                                for (col_indices, _) in store.get_unique_non_pk_index_columns() {
                                    compacted.prebuild_unique_index(&col_indices);
                                }
                            }
                        }
                        let min_id = chunk.first().map(|(id, _, _)| *id).unwrap_or(0);
                        let max_id = chunk.last().map(|(id, _, _)| *id).unwrap_or(0);
                        new_volumes.push((
                            compact_vol_id,
                            Arc::new(compacted),
                            crate::storage::volume::manifest::SegmentMeta {
                                segment_id: compact_vol_id,
                                file_path: std::path::PathBuf::new(),
                                row_count: chunk.len(),
                                min_row_id: min_id,
                                max_row_id: max_id,
                                creation_lsn: 0,
                                seal_seq: 0,
                                schema_version: self.schema_epoch.load(Ordering::Acquire),
                                visible_at_lsn: visible_lsn,
                            },
                        ));
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to write compacted volume for {}: {}",
                            table_name, e
                        );
                        write_failed = true;
                        break;
                    }
                }
            }

            if write_failed || new_volumes.is_empty() {
                // Clean up any volumes we did write before failure
                let vol_table_dir = vol_dir.join(table_name);
                for (vid, _, _) in &new_volumes {
                    let fname = format!("vol_{:016x}.vol", vid);
                    let _ = std::fs::remove_file(vol_table_dir.join(fname));
                }
                continue;
            }

            // Atomically register all new volumes and remove old segments.
            mgr.replace_segments_atomic_multi(new_volumes, &old_ids);

            // Clear only tombstones that existed at snapshot time for
            // row_ids in the merged volumes.
            {
                let merged_row_ids: FxHashSet<i64> = volumes
                    .iter()
                    .flat_map(|(_, vol)| vol.meta.row_ids.iter().copied())
                    .collect();
                mgr.remove_tombstones_matching_snapshot(&applied_tombstones, &merged_row_ids);
            }

            // CRITICAL: Persist manifest BEFORE deleting old files.
            if let Err(e) = mgr.persist_manifest_only() {
                eprintln!(
                    "Warning: Failed to persist manifest after compaction for {}: {}",
                    table_name, e
                );
                continue;
            }

            // Now safe to delete old volume files + stale .dv files —
            // UNLESS cross-process readers are attached (v1 SWMR). The
            // new manifest is durable, but readers that haven't yet
            // refreshed still hold the OLD manifest in memory and
            // reference these volumes. Defer unlink until the next
            // compaction cycle when readers may be gone.
            //
            // The cost of deferral is bounded: each compaction cycle
            // re-checks; once readers detach, the next cycle reaps the
            // accumulated old volumes. The orphan files are harmless on
            // disk — the new manifest doesn't reference them so the
            // engine never reads them.
            if self.defer_for_live_readers() {
                continue;
            }
            let vol_table_dir = vol_dir.join(table_name);
            let old_filenames: FxHashSet<String> = old_ids
                .iter()
                .map(|id| format!("vol_{:016x}.vol", id))
                .collect();
            if let Ok(entries) = std::fs::read_dir(&vol_table_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let ext = path.extension().and_then(|e| e.to_str());
                    if ext == Some("dv") {
                        let _ = std::fs::remove_file(&path);
                    } else if ext == Some("vol") {
                        if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                            if old_filenames.contains(fname) {
                                let _ = std::fs::remove_file(&path);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Seal hot buffer rows into frozen volumes and reclaim memory.
    ///
    /// Called automatically in the background. For each table with enough
    /// VISIBLE rows in the hot buffer, extracts them, writes a frozen volume
    /// to disk, then marks the sealed rows as deleted in the VersionStore.
    ///
    /// Uses actual visible row count (not committed_row_count which includes
    /// deleted rows and is never decremented).
    ///
    /// This is safe for concurrent queries because:
    /// - mark_deleted creates a new version (doesn't modify existing data)
    /// - In-flight scans that already read the row see the pre-delete version
    /// - New scans see the delete and skip the row (read from volume instead)
    ///
    /// Threshold: 100K rows (first seal) or 10K rows (subsequent seals).
    /// Output is split into target_volume_rows-sized volumes.
    fn seal_hot_buffers(&self) -> Result<()> {
        // Refuse to seal once the engine is in the catastrophic-failure
        // state. seal_hot_buffers exports visible committed versions
        // into volumes with a normal `visible_at_lsn`; a markerless
        // commit's parent-store rows would become durable + cross-
        // process visible, breaking recovery's "no commit marker →
        // discard" invariant. The user must restart.
        if self.is_failed() {
            return Ok(());
        }
        const SEAL_ROW_THRESHOLD: usize = 100_000;
        const SEAL_INCREMENTAL_THRESHOLD: usize = 10_000;
        let seal_row_threshold = SEAL_ROW_THRESHOLD;
        let seal_incremental_threshold = SEAL_INCREMENTAL_THRESHOLD;
        let target_volume_rows = self
            .config
            .read()
            .map(|c| c.persistence.target_volume_rows)
            .unwrap_or(1_048_576);

        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()),
        };

        // Snapshot-safe seal: cutoff is computed per-table (below) to minimize
        // the TOCTOU window between the check and extraction.

        // Collect table names that might need sealing.
        // Acquire each lock separately and drop before the next to avoid
        // holding multiple RwLocks simultaneously (deadlock prevention).
        let force_seal = self.force_seal_all.load(Ordering::Acquire);

        // Step 1: Collect candidates from version_stores (row count check)
        let candidates: Vec<(String, Arc<VersionStore>)> = {
            let stores = self.version_stores.read().unwrap();
            stores
                .iter()
                .filter_map(|(table_name, store)| {
                    let row_count = store.committed_row_count();
                    if force_seal {
                        if row_count == 0 {
                            return None;
                        }
                    } else if row_count == 0 {
                        return None;
                    }
                    Some((table_name.clone(), Arc::clone(store)))
                })
                .collect()
        };

        // Step 2: Filter by threshold using segment_managers.
        // Cache has_segments per table to avoid re-acquiring the lock later.
        let candidates: Vec<(String, Arc<VersionStore>, bool)> = if force_seal {
            let mgrs = self.segment_managers.read().unwrap();
            candidates
                .into_iter()
                .map(|(table_name, store)| {
                    let has_seg = mgrs
                        .get(&table_name)
                        .map(|m| m.has_segments())
                        .unwrap_or(false);
                    (table_name, store, has_seg)
                })
                .collect()
        } else {
            let mgrs = self.segment_managers.read().unwrap();
            candidates
                .into_iter()
                .filter_map(|(table_name, store)| {
                    let row_count = store.committed_row_count();
                    let has_seg = mgrs
                        .get(&table_name)
                        .map(|m| m.has_segments())
                        .unwrap_or(false);
                    let threshold = if has_seg {
                        seal_incremental_threshold
                    } else {
                        seal_row_threshold
                    };
                    if row_count >= threshold {
                        Some((table_name, store, has_seg))
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Step 3: Look up schemas (separate lock acquisition)
        let table_names: Vec<(String, CompactArc<Schema>, Arc<VersionStore>, bool)> = {
            let schemas = self.schemas.read().unwrap();
            candidates
                .into_iter()
                .filter_map(|(table_name, store, has_seg)| {
                    let schema = schemas.get(&table_name)?.clone();
                    Some((table_name, schema, store, has_seg))
                })
                .collect()
        };

        // Batch size for hot removal only. Volume is built once per table.
        // Smaller batches = shorter write lock hold time per batch.
        const REMOVE_BATCH_SIZE: usize = 50_000;

        for (table_name, schema, store, has_segments) in table_names {
            // Recheck the catastrophic-failure latch per-table.
            // Candidate selection above can take the segment_managers
            // / version_stores / config locks; while we hold those, a
            // failing commit may complete its in-memory drain and
            // latch the engine. Without this recheck we'd extract
            // markerless rows from `store` and seal them into a
            // volume with a normal `visible_at_lsn`, making the data
            // durable across restart even though WAL has no commit
            // marker for the txn.
            if self.is_failed() {
                return Ok(());
            }
            // Snapshot the pending-marker state BEFORE extracting.
            // Sealing while any commit marker is pending would let
            // `extract_for_seal` include rows whose marker LSN
            // exceeds the safe-visible frontier — a fresh reader
            // capped at safe-visible would then keep this segment
            // (visible_at_lsn ≤ cap) and observe rows committed
            // ABOVE its cap. Skip this table for now; the next
            // seal cycle picks it up once pending drains.
            //
            // `max_written` captured here is the visible_lsn we'll
            // stamp the segment with. The post-extract recheck
            // below (after the threshold/empty/latch checks)
            // verifies no commit landed during extract — if either
            // pending becomes non-empty OR `max_written` advances,
            // some commit's rows may have slipped into our
            // extraction and we discard the work.
            let (pre_pending_empty, visible_lsn_at_extract) = {
                let pending = self.pending_marker_lsns.lock();
                let empty = pending.is_empty();
                let mw = self.max_written_marker_lsn.load(Ordering::Acquire);
                (empty, mw)
            };
            if !pre_pending_empty {
                continue;
            }

            // Extract rows AND a CowBTree snapshot (O(1) Arc clone).
            // The snapshot records each row's txn_id at extraction time.
            // remove_sealed_rows compares against it to detect concurrent
            // commits that modified a row after extraction.
            // Re-check snapshot state per-table to close the TOCTOU window
            // between the top-of-function check and extraction. A snapshot that
            // starts between those points would otherwise cause phantom reads.
            let per_table_cutoff = self.registry.get_min_snapshot_begin_seq();
            let (mut all_rows, extraction_snapshot) = if let Some(cutoff) = per_table_cutoff {
                store.extract_for_seal_with_cutoff(cutoff)
            } else {
                let read_txn_id = INVALID_TRANSACTION_ID + 1;
                store.extract_for_seal(read_txn_id)
            };

            // Post-extract recheck: confirm no commit landed during
            // extraction. `record_commit` bumps `max_written` in
            // the same critical section as the pending insert, so
            // a max_written change OR pending becoming non-empty
            // both mean a commit is in flight and may have
            // contributed rows. Skip this table; next cycle
            // retries with a clean window.
            let (post_pending_empty, post_max_written) = {
                let pending = self.pending_marker_lsns.lock();
                let empty = pending.is_empty();
                let mw = self.max_written_marker_lsn.load(Ordering::Acquire);
                (empty, mw)
            };
            if !post_pending_empty || post_max_written != visible_lsn_at_extract {
                continue;
            }
            // On close (force_seal_all), seal ALL rows regardless of threshold.
            if !self.force_seal_all.load(Ordering::Acquire) {
                let threshold = if has_segments {
                    seal_incremental_threshold
                } else {
                    seal_row_threshold
                };
                if all_rows.len() < threshold {
                    continue;
                }
            }
            if all_rows.is_empty() {
                continue;
            }

            // Recheck the latch AFTER extraction. The pre-extraction
            // check above can pass just before a marker-failure
            // commit calls mark_engine_failed + complete_commit;
            // `extract_for_seal` then walks version chains via
            // `check_committed`, which sees the markerless txn as
            // committed and returns its rows. Sealing them into a
            // volume with a normal `visible_at_lsn` would survive
            // restart even though WAL recovery will discard the
            // txn. Drop the extracted rows on the floor — they're
            // still in the parent VersionStore and would have been
            // discarded on restart anyway, so no information is
            // lost.
            if self.is_failed() {
                return Ok(());
            }

            let total_rows = all_rows.len();

            // Normalize rows to current schema before sealing.
            // After ALTER TABLE ADD COLUMN ... DEFAULT ..., old rows have
            // fewer columns. Without normalization, the default is lost
            // permanently in the cold segment (stored as NULL).
            let schema_cols = schema.columns.len();
            for (_row_id, row) in &mut all_rows {
                if row.len() < schema_cols {
                    for i in row.len()..schema_cols {
                        let col = &schema.columns[i];
                        if let Some(ref default_val) = col.default_value {
                            row.push(default_val.clone());
                        } else {
                            row.push(crate::core::Value::null(col.data_type));
                        }
                    }
                } else if row.len() > schema_cols {
                    row.truncate(schema_cols);
                }
            }

            let vol_dir = pm.path().join("volumes");

            // Build volumes from rows, splitting at target_volume_rows boundary.
            let compress = self
                .config
                .read()
                .map(|c| c.persistence.volume_compression)
                .unwrap_or(true);
            match crate::storage::volume::seal::seal_and_persist_multi(
                &schema,
                &all_rows,
                &vol_dir,
                &table_name,
                compress,
                target_volume_rows,
            ) {
                Ok(sealed_volumes) => {
                    // Pre-build unique hash indices BEFORE registration so the
                    // first INSERT after seal doesn't pay a ~60ms stall scanning
                    // all rows. Safe: volumes are not yet visible to other threads.
                    {
                        let stores = self.version_stores.read().unwrap();
                        if let Some(store) = stores.get(&table_name) {
                            for (col_indices, _) in store.get_unique_non_pk_index_columns() {
                                for (vol, _, _) in &sealed_volumes {
                                    vol.prebuild_unique_index(&col_indices);
                                }
                            }
                        }
                    }
                    let mgr = self.get_or_create_segment_manager(&table_name);

                    // Seal critical section under exclusive fence: register cold
                    // segments + remove hot rows + remove hot index entries.
                    // DML operations hold the shared fence, so they cannot race
                    // between cold constraint checks and hot publication.
                    {
                        let _seal_guard = mgr.acquire_seal_write();

                        mgr.set_seal_overlap(total_rows);

                        // Stamp seal_seq to reflect what data the volume contains:
                        // - With cutoff: volume has rows committed before cutoff, so use cutoff
                        // - Without cutoff: all committed rows, use current sequence
                        // Compaction skips volumes with seal_seq >= min_snap_begin_seq.
                        let current_seal_seq = per_table_cutoff
                            .map(|s| s as u64)
                            .unwrap_or_else(|| self.registry.get_current_sequence() as u64);
                        // Use the SAFE-VISIBLE commit LSN captured
                        // at extract time (`visible_lsn_at_extract`).
                        // It equals `max_written_marker_lsn` AT THE
                        // MOMENT we extracted, while `pending` was
                        // empty — so every row in `all_rows` has a
                        // commit marker LSN <= this value. The
                        // pre/post pending+max_written recheck
                        // above guarantees no commit slipped into
                        // our extract.
                        //
                        // Re-sampling `safe_visible_commit_lsn`
                        // here would race a commit landing
                        // between extract and stamp: the new
                        // commit's record_commit bumps
                        // max_written, then if pending drains
                        // before our re-sample, safe_visible
                        // jumps to the new max_written, but our
                        // segment doesn't contain the new
                        // commit's rows. A capped reader at the
                        // new safe_visible would expect the
                        // segment to reflect that commit.
                        // Stamping at the captured pre-extract
                        // value keeps segment visibility tied to
                        // its actual contents.
                        let visible_lsn = visible_lsn_at_extract;
                        for (volume, _path, volume_id) in &sealed_volumes {
                            self.register_volume_with_id_and_seal_seq(
                                &table_name,
                                Arc::clone(volume),
                                *volume_id,
                                current_seal_seq,
                                visible_lsn,
                            );
                        }

                        let mut index_cleanups = Vec::new();
                        let mut all_skipped_inner: Vec<i64> = Vec::new();
                        let all_row_ids: Vec<i64> = all_rows.iter().map(|(id, _)| *id).collect();
                        for batch in all_row_ids.chunks(REMOVE_BATCH_SIZE) {
                            let (removed, cleanup, skipped) =
                                store.remove_sealed_rows(batch, &extraction_snapshot);
                            store.subtract_committed_row_count(removed);
                            index_cleanups.push(cleanup);
                            all_skipped_inner.extend(skipped);
                        }

                        if !all_skipped_inner.is_empty() {
                            let seal_seq = self.registry.get_current_sequence() as u64;
                            // Reuse the captured `visible_lsn_at_extract`
                            // for tombstone stamping (same source as
                            // the segment stamp above). Re-sampling
                            // `safe_visible_commit_lsn` here would
                            // race the same way: a commit landing
                            // between extract and stamp could push
                            // the live safe-visible past the
                            // tombstone's actual content cap, letting
                            // readers below the new safe-visible
                            // observe a deletion that wasn't yet
                            // visible to them.
                            mgr.add_tombstones(
                                &all_skipped_inner,
                                seal_seq,
                                visible_lsn_at_extract,
                            );
                        }

                        if let Some(&(max_id, _)) = all_rows.last() {
                            let current = store.get_auto_increment_counter();
                            if max_id > current {
                                store.set_auto_increment_counter(max_id);
                            }
                        }

                        mgr.clear_seal_overlap();

                        for cleanup in index_cleanups {
                            store.remove_sealed_index_entries(cleanup);
                        }

                        // Clear tombstones for sealed row_ids INSIDE the fence.
                        {
                            let skip_set: FxHashSet<i64> =
                                all_skipped_inner.iter().copied().collect();
                            let ts = mgr.tombstone_set_arc();
                            if !ts.is_empty() {
                                let mut sealed_ids: FxHashSet<i64> = FxHashSet::default();
                                for (vol, _, _) in &sealed_volumes {
                                    for &rid in &vol.meta.row_ids {
                                        if ts.contains_key(&rid) && !skip_set.contains(&rid) {
                                            sealed_ids.insert(rid);
                                        }
                                    }
                                }
                                if !sealed_ids.is_empty() {
                                    mgr.remove_tombstones_for_rows(&sealed_ids);
                                }
                            }
                        }

                        // _seal_guard dropped here — DML unblocked
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to seal hot buffer for {}: {}",
                        table_name, e
                    );
                }
            }
        }

        Ok(())
    }
}

impl ReadEngine for MVCCEngine {
    fn begin_read_transaction(&self) -> Result<Box<dyn ReadTransaction>> {
        // Writable transaction satisfies the read trait via the
        // WriteTransaction: ReadTransaction supertrait. The caller
        // sees only the read surface through the trait object.
        //
        // Read-only engines must still serve read transactions, but the
        // public Engine::begin_transaction is gated on `is_read_only_mode`
        // and would refuse here. Use the internal writable entry point;
        // the resulting Box<dyn WriteTransaction> is cast to the read
        // surface via the supertrait.
        let tx: Box<dyn WriteTransaction> = self.begin_writable_transaction_internal()?;
        Ok(tx)
    }

    fn begin_read_transaction_with_level(
        &self,
        level: IsolationLevel,
    ) -> Result<Box<dyn ReadTransaction>> {
        let tx: Box<dyn WriteTransaction> =
            self.begin_writable_transaction_with_level_internal(level)?;
        Ok(tx)
    }
}

impl MVCCEngine {
    /// Begin a writable transaction, bypassing the read-only mode gate.
    ///
    /// Internal-only entry point for trusted write-intent callers: DML
    /// (`INSERT`/`UPDATE`/`DELETE`), DDL, COPY, ANALYZE, and the executor's
    /// public `begin_transaction` (which has already checked the
    /// `Executor::read_only` flag at its API surface). The verbose name
    /// is deliberate: any new caller using this method should be
    /// challenged in code review to confirm the call site really needs
    /// to write — read-only access should go through
    /// [`ReadEngine::begin_read_transaction`] instead.
    ///
    /// External callers (anyone going through `Engine::begin_transaction`)
    /// hit the read-only gate in the trait impl below — no escape hatch.
    pub(crate) fn begin_writable_transaction_internal(&self) -> Result<Box<dyn WriteTransaction>> {
        self.begin_writable_transaction_with_level_internal(self.get_isolation_level())
    }

    /// `begin_writable_transaction_internal` with an explicit isolation level.
    pub(crate) fn begin_writable_transaction_with_level_internal(
        &self,
        level: IsolationLevel,
    ) -> Result<Box<dyn WriteTransaction>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }
        // Refuse new write transactions once the engine is in the
        // catastrophic-failure state. A prior commit already drained
        // markerless data into parent VersionStores and we can't
        // undo it; admitting another writer would let it read or
        // overwrite that markerless state and then publish its own
        // WAL marker, making the markerless data interleave with
        // legitimate durable state. The user must restart.
        if self.is_failed() {
            return Err(Error::internal(
                "engine is in the catastrophic-failure state (a prior commit's \
                 WAL marker write failed after some tables were already \
                 committed); no new write transactions can be started. Restart \
                 the process; recovery will discard the markerless transaction.",
            ));
        }

        // Begin transaction in registry
        let (txn_id, begin_seq) = self.registry.begin_transaction();
        if txn_id == INVALID_TRANSACTION_ID {
            return Err(Error::internal(
                "transaction registry is not accepting new transactions",
            ));
        }

        // Create transaction
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&self.registry));

        // Set isolation level if different from default
        if level != IsolationLevel::ReadCommitted {
            txn.set_isolation_level(level)?;
        }

        // Set engine operations
        let engine_ops = self.create_engine_operations();
        txn.set_engine_operations(engine_ops);

        Ok(Box::new(txn))
    }
}

impl Engine for MVCCEngine {
    fn open(&mut self) -> Result<()> {
        MVCCEngine::open_engine(self)
    }

    fn close(&mut self) -> Result<()> {
        MVCCEngine::close_engine(self)
    }

    fn begin_transaction(&self) -> Result<Box<dyn WriteTransaction>> {
        self.begin_transaction_with_level(self.get_isolation_level())
    }

    fn begin_transaction_with_level(
        &self,
        level: IsolationLevel,
    ) -> Result<Box<dyn WriteTransaction>> {
        // Read-only gate at the public Engine trait surface. Without this,
        // `Database::engine().begin_transaction()` on a `?read_only=true`
        // handle would still return a writable transaction (the engine
        // would happily mint one), bypassing every other gate. Trusted
        // internal callers go through
        // `begin_writable_transaction_with_level_internal` and are not
        // affected.
        if self.is_read_only_mode() {
            return Err(Error::read_only_violation_at("engine", "begin_transaction"));
        }
        self.begin_writable_transaction_with_level_internal(level)
    }

    fn path(&self) -> Option<&str> {
        if self.path == "memory://" {
            None
        } else {
            Some(&self.path)
        }
    }

    fn table_exists(&self, table_name: &str) -> Result<bool> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let name = to_lowercase_cow(table_name);
        let schemas = self.schemas.read().unwrap();
        Ok(schemas.contains_key(name.as_ref()))
    }

    fn index_exists(&self, index_name: &str, table_name: &str) -> Result<bool> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        Ok(store.index_exists(index_name))
    }

    fn get_index(&self, table_name: &str, index_name: &str) -> Result<Box<dyn Index>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        if store.index_exists(index_name) {
            // Index exists but we can't clone Arc<dyn Index> into Box<dyn Index>
            // This will be properly implemented in Phase 6.6
            return Err(Error::internal(format!(
                "index retrieval not yet implemented: {}.{}",
                table_name, index_name
            )));
        }

        Err(Error::internal(format!(
            "index not found: {}.{}",
            table_name, index_name
        )))
    }

    fn get_table_schema(&self, table_name: &str) -> Result<CompactArc<Schema>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let name = to_lowercase_cow(table_name);
        let schemas = self.schemas.read().unwrap();
        schemas
            .get(name.as_ref())
            .cloned()
            .ok_or_else(|| Error::TableNotFound(name.as_ref().to_string()))
    }

    #[inline]
    fn schema_epoch(&self) -> u64 {
        self.schema_epoch.load(Ordering::Acquire)
    }

    fn list_table_indexes(&self, table_name: &str) -> Result<FxHashMap<String, String>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        let mut result = FxHashMap::default();
        for index_name in store.list_indexes() {
            result.insert(index_name, "BTree".to_string());
        }
        Ok(result)
    }

    fn get_all_indexes(&self, table_name: &str) -> Result<Vec<std::sync::Arc<dyn Index>>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;

        // Get all indexes from the version store (convert SmallVec to Vec for trait compatibility)
        Ok(store.get_all_indexes().into_vec())
    }

    fn get_isolation_level(&self) -> IsolationLevel {
        self.registry.get_global_isolation_level()
    }

    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        self.registry.set_global_isolation_level(level);
        Ok(())
    }

    fn get_config(&self) -> Config {
        self.config.read().expect("config lock poisoned").clone()
    }

    fn update_config(&mut self, config: Config) -> Result<()> {
        self.update_engine_config(config)
    }

    fn dedup_segments(&self) -> usize {
        MVCCEngine::dedup_segments(self)
    }

    fn checkpoint_cycle(&self) -> Result<()> {
        self.ensure_writable()?;
        MVCCEngine::checkpoint_cycle(self)
    }

    fn force_checkpoint_cycle(&self) -> Result<()> {
        self.ensure_writable()?;
        MVCCEngine::checkpoint_cycle_inner(self, true)?;
        self.compact_after_checkpoint_forced();
        Ok(())
    }

    fn create_snapshot(&self) -> Result<()> {
        // Snapshot writes new files to disk. A read-only engine refuses
        // even though the I/O layer would also fail with EROFS / EACCES —
        // a clear `ReadOnlyViolation` here beats a confusing late error
        // and is consistent with how the SQL `PRAGMA SNAPSHOT` is gated
        // by the parser write-reason classifier.
        self.ensure_writable()?;
        MVCCEngine::create_backup_snapshot(self)
    }

    fn restore_snapshot(&self, timestamp: Option<&str>) -> Result<String> {
        // Restore is destructive: it replaces engine state in-place from
        // a backup. Refusing on read-only engines is mandatory, not
        // defense-in-depth — the on-disk replacement bypasses anything
        // the read-only file-lock contract was supposed to guarantee.
        self.ensure_writable()?;
        MVCCEngine::restore_from_snapshot(self, timestamp)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_create_index(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
        hnsw_m: Option<u16>,
        hnsw_ef_construction: Option<u16>,
        hnsw_ef_search: Option<u16>,
        hnsw_distance_metric: Option<u8>,
    ) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Get table schema to look up column IDs and data types
        let schema = self.get_table_schema(table_name)?;

        // Build column_ids and data_types from schema
        // Use schema's cached column_index_map for O(1) lookup instead of O(n) linear scan
        let col_index_map = schema.column_index_map();
        let mut column_ids = Vec::with_capacity(column_names.len());
        let mut data_types = Vec::with_capacity(column_names.len());

        for col_name in column_names {
            let col_name_lower = col_name.to_lowercase();
            if let Some(&idx) = col_index_map.get(&col_name_lower) {
                column_ids.push(idx as i32);
                data_types.push(schema.columns[idx].data_type);
            } else {
                return Ok(());
            }
        }

        // Create index metadata
        let index_meta = super::persistence::IndexMetadata {
            name: index_name.to_string(),
            table_name: table_name.to_string(),
            column_names: column_names.to_vec(),
            column_ids,
            data_types,
            is_unique,
            index_type,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_distance_metric,
        };

        // Serialize and record to WAL
        let data = index_meta.serialize();
        self.record_ddl(table_name, WALOperationType::CreateIndex, &data)
    }

    fn record_drop_index(&self, table_name: &str, index_name: &str) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // For drop index, the entry.data is simply the index name as bytes
        self.record_ddl(
            table_name,
            WALOperationType::DropIndex,
            index_name.as_bytes(),
        )
    }

    fn record_alter_table_add_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        default_expr: Option<&str>,
        vector_dimensions: u16,
    ) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name + column_name_len(2) + column_name
        //          + data_type(1) + [IF Vector: vec_dims(2)] + nullable(1) + default_expr_len(2) + default_expr
        let mut data = Vec::new();
        data.push(1u8); // Operation type: AddColumn = 1

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Column name
        data.extend_from_slice(&(column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(column_name.as_bytes());

        // Data type (1 byte, + 2 bytes dimension for Vector)
        data.push(data_type.as_u8());
        if data_type == DataType::Vector {
            data.extend_from_slice(&vector_dimensions.to_le_bytes());
        }

        // Nullable
        data.push(if nullable { 1 } else { 0 });

        // Default expression
        if let Some(expr) = default_expr {
            data.extend_from_slice(&(expr.len() as u16).to_le_bytes());
            data.extend_from_slice(expr.as_bytes());
        } else {
            data.extend_from_slice(&0u16.to_le_bytes());
        }

        // Record the WAL entry FIRST so the failed-latch guard
        // inside `record_ddl` can reject the call before we mutate
        // any segment-manager state. The previous order
        // (invalidate_mappings → record_ddl) left the cached
        // mapping pointing at the post-add schema even when the
        // WAL write failed and the caller reverted the schema —
        // cold reads then used a mapping that included a column
        // the schema no longer had.
        self.record_ddl(table_name, WALOperationType::AlterTable, &data)?;

        // Stamp the table-level manifest epoch so a no-shm reader's
        // drift check sees this ADD COLUMN even when the next
        // checkpoint produces no new segment. The SQL ALTER path
        // (executor/ddl.rs) doesn't go through `engine.create_column_with_default`
        // (which has its own propagate_schema_bump call) — it
        // mutates the version_store schema directly via
        // `table.create_column_with_default_value`, then arrives
        // here with no other manifest hook firing. Without this
        // bump, the reader's `peek_schema_drift` accepts the
        // unchanged on-disk schema_version and silently keeps
        // serving the stale column layout.
        self.propagate_schema_bump(table_name);

        // WAL succeeded — recompute cold mappings with the post-add
        // schema. dropped_columns stays permanent — old volumes
        // have stale data under the dropped name, new volumes get
        // the re-added column at a new position.
        {
            let schema = self.schemas.read().unwrap().get(table_name).cloned();
            let mgrs = self.segment_managers.read().unwrap();
            if let Some(mgr) = mgrs.get(table_name) {
                if let Some(ref s) = schema {
                    mgr.invalidate_mappings(s);
                }
            }
        }

        Ok(())
    }

    fn record_alter_table_drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name + column_name_len(2) + column_name
        let mut data = Vec::new();
        data.push(2u8); // Operation type: DropColumn = 2

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Column name
        data.extend_from_slice(&(column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(column_name.as_bytes());

        self.record_ddl(table_name, WALOperationType::AlterTable, &data)
    }

    fn record_alter_table_rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name
        //          + old_name_len(2) + old_name + new_name_len(2) + new_name
        let mut data = Vec::new();
        data.push(3u8); // Operation type: RenameColumn = 3

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Old column name
        data.extend_from_slice(&(old_column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(old_column_name.as_bytes());

        // New column name
        data.extend_from_slice(&(new_column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(new_column_name.as_bytes());

        self.record_ddl(table_name, WALOperationType::AlterTable, &data)
    }

    fn record_alter_table_modify_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        vector_dimensions: u16,
    ) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name
        //          + column_name_len(2) + column_name + data_type(1) + [IF Vector: vec_dims(2)] + nullable(1)
        let mut data = Vec::new();
        data.push(4u8); // Operation type: ModifyColumn = 4

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Column name
        data.extend_from_slice(&(column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(column_name.as_bytes());

        // Data type (1 byte, + 2 bytes dimension for Vector)
        data.push(data_type.as_u8());
        if data_type == DataType::Vector {
            data.extend_from_slice(&vector_dimensions.to_le_bytes());
        }

        // Nullable
        data.push(if nullable { 1 } else { 0 });

        self.record_ddl(table_name, WALOperationType::AlterTable, &data)?;
        // Stamp the table-level manifest epoch so a no-shm reader's
        // drift check sees this MODIFY COLUMN even when the next
        // checkpoint produces no new segment. See the matching
        // block in `record_alter_table_add_column` for the full
        // rationale — same SQL ALTER path bypass.
        self.propagate_schema_bump(table_name);
        Ok(())
    }

    fn record_alter_table_rename(&self, old_table_name: &str, new_table_name: &str) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }

        // Serialize: operation_type(1) + old_name_len(2) + old_name + new_name_len(2) + new_name
        let mut data = Vec::new();
        data.push(5u8); // Operation type: RenameTable = 5

        // Old table name
        data.extend_from_slice(&(old_table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(old_table_name.as_bytes());

        // New table name
        data.extend_from_slice(&(new_table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(new_table_name.as_bytes());

        self.record_ddl(old_table_name, WALOperationType::AlterTable, &data)
    }

    fn record_truncate_table(&self, table_name: &str) -> Result<()> {
        // The `transactional_ddl_fence` is held by the
        // CALLER (`Executor::execute_truncate`), spanning
        // both `table.truncate()` AND this WAL write. We do
        // NOT re-acquire SH here — parking_lot's RwLock can
        // deadlock on reentrant SH if a checkpoint EX is
        // waiting for the outer guard to drop.

        let table_lower = table_name.to_lowercase();

        // WAL FIRST: record the truncate before deleting segment files.
        // If crash happens after WAL but before file deletion, WAL replay
        // will re-execute the truncate. Orphan files are harmless.
        if !self.should_skip_wal() {
            self.record_ddl(table_name, WALOperationType::TruncateTable, &[])?;
        }

        // Clear in-memory segment state
        {
            let mgrs = self.segment_managers.read().unwrap();
            if let Some(mgr) = mgrs.get(&table_lower) {
                mgr.clear();
            }
        }

        // Delete volume files when no live cross-process
        // reader could still hold a stale manifest pointer.
        // Same defer-when-readers-live treatment as DROP
        // TABLE: while a reader is live the directory and
        // its `vol_NNNN.vol` files stay UNTOUCHED at their
        // original path, and `sweep_orphan_table_dirs`
        // reaps them on a future checkpoint / open.
        //
        // TRUNCATE differs from DROP in that the table
        // remains in the active catalog after the operation,
        // so the orphan-sweep does NOT pick this up by
        // itself. The deferred-delete window only matters
        // until the next time a writer DROPs the table or
        // crashes — both of which clear the catalog entry
        // and let the sweep run.
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let defer = self.defer_for_live_readers();
                // Propagate the failure: same recovery
                // hazard as DROP — leftover `manifest.bin`
                // can be re-recorded as live, then truncate
                // past the TruncateTable WAL entry.
                crate::storage::volume::io::delete_table_volumes_when_safe(
                    &vol_dir,
                    &table_lower,
                    defer,
                )?;
                let snapshot_dir = pm.path().join("snapshots");
                let ts_path = snapshot_dir.join(&table_lower).join("tombstones.dat");
                let _ = std::fs::remove_file(ts_path);
            }
        }

        Ok(())
    }

    fn fetch_rows_by_ids(&self, table_name: &str, row_ids: &[i64]) -> Result<crate::core::RowVec> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        let read_txn_id = INVALID_TRANSACTION_ID + 1;
        let mut result = store.get_visible_versions_batch(row_ids, read_txn_id);

        // Fall back to cold segments for rows not found in hot
        if result.len() < row_ids.len() {
            let mgr = self.get_or_create_segment_manager(table_name);
            if mgr.has_segments() {
                let schema = store.schema().clone();
                let found_ids: rustc_hash::FxHashSet<i64> =
                    result.iter().map(|(id, _)| *id).collect();
                for &rid in row_ids {
                    if !found_ids.contains(&rid) {
                        if let Some(row) = mgr.get_cold_row_normalized(rid, &schema) {
                            result.push((rid, row));
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    fn get_row_fetcher(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> crate::core::RowVec + Send + Sync>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        let mgr = self.get_or_create_segment_manager(table_name);
        let schema = store.schema().clone();
        let read_txn_id = INVALID_TRANSACTION_ID + 1;

        Ok(Box::new(move |row_ids: &[i64]| {
            let mut result = store.get_visible_versions_batch(row_ids, read_txn_id);
            if result.len() < row_ids.len() && mgr.has_segments() {
                let found_ids: rustc_hash::FxHashSet<i64> =
                    result.iter().map(|(id, _)| *id).collect();
                for &rid in row_ids {
                    if !found_ids.contains(&rid) {
                        if let Some(row) = mgr.get_cold_row_normalized(rid, &schema) {
                            result.push((rid, row));
                        }
                    }
                }
            }
            result
        }))
    }

    /// Get a count-only function for counting visible rows by their IDs.
    /// This is optimized for COUNT(*) subqueries where we don't need the actual row data.
    fn get_row_counter(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> usize + Send + Sync>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;
        let mgr = self.get_or_create_segment_manager(table_name);
        let read_txn_id = INVALID_TRANSACTION_ID + 1;

        Ok(Box::new(move |row_ids: &[i64]| {
            let mut count = store.count_visible_versions_batch(row_ids, read_txn_id);
            if count < row_ids.len() && mgr.has_segments() {
                for &rid in row_ids {
                    if !store.has_committed_row(rid) && mgr.row_exists(rid) {
                        count += 1;
                    }
                }
            }
            count
        }))
    }
}

// =============================================================================
// Cleanup Functions
// =============================================================================

impl MVCCEngine {
    /// Cleanup old transactions that have been idle for too long.
    ///
    /// Silent no-op on a read-only engine (returns 0). Read-only engines
    /// have no committed-transaction churn to clean and must not mutate
    /// the registry. The same `is_read_only_mode` short-circuit applies
    /// to every `cleanup_*` family method below.
    pub fn cleanup_old_transactions(&self, max_age: std::time::Duration) -> i32 {
        if !self.is_open() || self.is_read_only_mode() {
            return 0;
        }
        self.registry.cleanup_old_transactions(max_age)
    }

    /// Cleanup deleted rows older than retention period from all tables
    pub fn cleanup_deleted_rows(&self, max_age: std::time::Duration) -> i32 {
        if !self.is_open() || self.is_read_only_mode() {
            return 0;
        }

        let stores = self.version_stores.read().unwrap();
        let mut total_removed = 0;

        for store in stores.values() {
            total_removed += store.cleanup_deleted_rows(max_age);
        }

        total_removed
    }

    /// Cleanup old previous versions that are no longer needed from all tables
    pub fn cleanup_old_previous_versions(&self) -> i32 {
        if !self.is_open() || self.is_read_only_mode() {
            return 0;
        }

        let stores = self.version_stores.read().unwrap();
        let mut total_cleaned = 0;

        for store in stores.values() {
            total_cleaned += store.cleanup_old_previous_versions();
        }

        total_cleaned
    }

    /// No-op: in the new tombstone design, cold deletes are not tracked
    /// per-transaction. Kept for API compatibility. Always a no-op,
    /// regardless of read-only mode.
    pub fn cleanup_abandoned_cold_deletes(&self) {
        // Intentionally empty.
    }

    /// Manual VACUUM: cleanup deleted rows, old versions, and stale transactions.
    ///
    /// Get the oldest snapshot timestamp that was loaded during startup.
    /// Used by CLI restore to pick the correct DDL file after Database::open()
    /// has determined which snapshots were actually loadable.
    pub fn oldest_loaded_snapshot_timestamp(&self) -> Option<String> {
        let ts = self.snapshot_timestamps.read().unwrap();
        ts.values().min().cloned()
    }

    /// When `table_name` is `Some`, only that table is vacuumed.
    /// Returns `(deleted_rows_cleaned, old_versions_cleaned, transactions_cleaned)`.
    pub fn vacuum(
        &self,
        table_name: Option<&str>,
        retention: std::time::Duration,
    ) -> crate::core::Result<(i32, i32, i32)> {
        self.ensure_writable()?;
        if !self.is_open() {
            return Ok((0, 0, 0));
        }

        let txn_cleaned = self.cleanup_old_transactions(retention);

        let stores = self.version_stores.read().unwrap();

        let mut rows_cleaned = 0;
        let mut versions_cleaned = 0;

        if let Some(name) = table_name {
            if let Some(store) = stores.get(name) {
                rows_cleaned += store.cleanup_deleted_rows(retention);
                versions_cleaned += store.cleanup_old_previous_versions_with_retention(retention);
            } else {
                return Err(crate::core::Error::TableNotFound(name.to_string()));
            }
        } else {
            for store in stores.values() {
                rows_cleaned += store.cleanup_deleted_rows(retention);
                versions_cleaned += store.cleanup_old_previous_versions_with_retention(retention);
            }
        }

        drop(stores);

        Ok((rows_cleaned, versions_cleaned, txn_cleaned))
    }

    /// Start periodic cleanup of old transactions and deleted rows
    ///
    /// Returns a handle that can be used to stop the cleanup thread.
    /// On a read-only engine this is a silent no-op: the returned handle
    /// has no underlying thread, matching the behaviour of `start_cleanup`
    /// which also skips the background loop on read-only opens. Callers
    /// can drop or `stop()` the handle the same way; nothing else
    /// changes from their perspective.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_periodic_cleanup(
        self: &Arc<Self>,
        interval: std::time::Duration,
        max_age: std::time::Duration,
    ) -> CleanupHandle {
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let stop_flag = Arc::new(AtomicBool::new(false));

        if self.is_read_only_mode() {
            return CleanupHandle {
                stop_flag,
                thread: None,
            };
        }

        let stop_flag_clone = Arc::clone(&stop_flag);
        let engine = Arc::clone(self);

        let handle = thread::spawn(move || {
            while !stop_flag_clone.load(Ordering::Acquire) {
                // Sleep for the interval (check stop flag periodically)
                let check_interval = std::time::Duration::from_millis(100);
                let mut elapsed = std::time::Duration::ZERO;
                while elapsed < interval && !stop_flag_clone.load(Ordering::Acquire) {
                    thread::sleep(check_interval);
                    elapsed += check_interval;
                }

                if stop_flag_clone.load(Ordering::Acquire) {
                    break;
                }

                // Perform cleanup
                let _txn_count = engine.cleanup_old_transactions(max_age);
                let _row_count = engine.cleanup_deleted_rows(max_age);
                let _prev_version_count = engine.cleanup_old_previous_versions();
            }
        });

        CleanupHandle {
            stop_flag,
            thread: Some(handle),
        }
    }
}

/// Handle for stopping the cleanup thread
pub struct CleanupHandle {
    stop_flag: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CleanupHandle {
    /// Stop the cleanup thread
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CleanupHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Engine operations for transaction callbacks
///
/// Holds Arc references to shared engine state, allowing safe access
/// from transactions without raw pointers.
struct EngineOperations {
    /// Shared reference to schemas (each schema is Arc-wrapped to avoid cloning on lookup)
    schemas: Arc<RwLock<FxHashMap<String, CompactArc<Schema>>>>,
    /// Shared reference to version stores
    version_stores: Arc<RwLock<FxHashMap<String, Arc<VersionStore>>>>,
    /// Shared reference to registry
    registry: Arc<TransactionRegistry>,
    /// Shared reference to transaction version stores cache
    txn_version_stores: Arc<RwLock<TxnVersionStoreMap>>,
    /// Shared reference to persistence manager (optional)
    persistence: Arc<Option<PersistenceManager>>,
    /// Shared reference to loading_from_disk flag
    loading_from_disk: Arc<AtomicBool>,
    /// Shared reference to segment managers
    segment_managers:
        Arc<RwLock<FxHashMap<String, Arc<crate::storage::volume::manifest::SegmentManager>>>>,
    /// Seal fence for WAL truncation safety
    seal_fence: Arc<parking_lot::RwLock<()>>,
    /// Transactional-DDL fence (cloned from MVCCEngine).
    /// Held shared by `MvccTransaction` while it has any
    /// in-flight DDL changes; checkpoint takes exclusive in
    /// `rerecord_ddl_to_wal` to wait for them out.
    transactional_ddl_fence: Arc<parking_lot::RwLock<()>>,
    /// Shared clone of the engine's `schema_epoch` counter
    /// so the transactional DDL paths
    /// (`create_table` / `drop_table` /
    /// `restore_child_fk_schemas` / `rename_table`) can bump
    /// the same epoch the direct DDL paths bump. Required
    /// for cache invalidation: `find_referencing_fks`,
    /// compiled DML fast paths, and the SWMR schema-drift
    /// check all key on `schema_epoch`. Without this, a
    /// committed transactional CREATE adding an FK child
    /// would not invalidate `fk_reverse_cache` and a later
    /// parent UPDATE / DELETE could skip FK enforcement
    /// against the new constraint.
    schema_epoch: Arc<AtomicU64>,
    /// Engine root path (cloned from `MVCCEngine::path` at
    /// construction). Used by `finalize_committed_drops` to
    /// build the same `<root>/volumes/` and
    /// `<root>/readers/` paths that `MVCCEngine::defer_for_live_readers`
    /// inspects, so the defer-DROP-when-readers-live policy
    /// applies to transactional DROPs as well as direct
    /// `drop_table_internal` calls.
    engine_path: String,
    /// Shared clone of the engine's lease max-age (nanos)
    /// used by `MVCCEngine::defer_for_live_readers`. Stored
    /// in an `Arc<AtomicU64>` so a future PRAGMA / config
    /// reload that updates the engine's view propagates to
    /// both helpers without an EngineOperations rebuild.
    lease_max_age_nanos: Arc<AtomicU64>,
    /// Shared with `MVCCEngine::pending_drop_cleanups`.
    /// `EngineOperations::finalize_committed_drops` adds a
    /// table name here when the post-commit volume-directory
    /// cleanup fails (so `manifest.bin` survives despite the
    /// catalog reflecting a DROP). `compute_wal_truncate_floor`
    /// refuses every WAL truncation while this set is
    /// non-empty so the on-disk DropTable record stays
    /// available for crash recovery; `sweep_orphan_table_dirs`
    /// retries the removal and clears the set on success.
    pending_drop_cleanups: Arc<parking_lot::Mutex<rustc_hash::FxHashSet<String>>>,
    /// Snapshot of the engine's `shm` handle taken at txn-begin. Cloned
    /// out of the engine's `Mutex<Option<Arc<ShmHandle>>>` so commits
    /// don't need to re-acquire the engine lock to publish. `None`
    /// means no cross-process publishing (in-memory engine, read-only
    /// open, or non-Unix). Stable across the txn's lifetime: a
    /// concurrent close swaps the engine's slot but our Arc keeps the
    /// mapping alive until this struct drops.
    shm: Option<Arc<crate::storage::mvcc::shm::ShmHandle>>,
    /// Shared with `MVCCEngine`. Tracks WAL marker
    /// LSNs that have been written but not yet `complete_commit`-d.
    /// `publish_visible_commit_lsn` reads this to compute a
    /// safe `visible_commit_lsn` watermark instead of trusting
    /// the per-txn LSN.
    pending_marker_lsns: Arc<parking_lot::Mutex<std::collections::BTreeSet<u64>>>,
    /// Shared with `MVCCEngine::shm_publish_lock` — serializes
    /// every shm visibility publish so the seqlock sequence is
    /// atomic w.r.t. concurrent commits on this same engine.
    shm_publish_lock: Arc<parking_lot::Mutex<()>>,
    /// Shared with `MVCCEngine::lease_present`. When `false`,
    /// the txn-commit publish path skips the seqlock dance and
    /// just `fetch_max`-es `visible_commit_lsn` so the writer's
    /// own truncate clamp keeps advancing without paying the
    /// `shm_publish_lock` + `publish_seq` cost on every commit.
    lease_present: Arc<AtomicBool>,
    /// Highest marker LSN ever written by any commit on this
    /// engine. Used as the upper bound of `safe_visible_commit_lsn`
    /// when `pending_marker_lsns` drains.
    max_written_marker_lsn: Arc<std::sync::atomic::AtomicU64>,
    /// Shared with `MVCCEngine::completed_marker_txns`.
    completed_marker_txns: Arc<parking_lot::Mutex<std::collections::BTreeMap<u64, i64>>>,
    /// Shared catastrophic-failure latch (cloned from
    /// `MVCCEngine::failed`). Set by `mark_engine_failed` to block
    /// every durability path on the engine.
    failed: Arc<AtomicBool>,
    /// Snapshot of `MVCCEngine::is_read_only_mode()` taken at construction.
    /// Read-only engines need the SWMR-aware `count_via_scan` path in
    /// `SegmentedTable::row_count`; writers stay on the O(1) formula.
    /// Read-only mode is fixed for the engine's lifetime, so a
    /// snapshot at construction is sufficient.
    read_only: bool,
}

// EngineOperations is Send + Sync because all fields are Arc-wrapped thread-safe types

impl EngineOperations {
    fn new(engine: &MVCCEngine) -> Self {
        let shm = engine.shm.lock().unwrap().as_ref().map(Arc::clone);
        let read_only = engine.is_read_only_mode();
        Self {
            schemas: Arc::clone(&engine.schemas),
            version_stores: Arc::clone(&engine.version_stores),
            registry: Arc::clone(&engine.registry),
            txn_version_stores: Arc::clone(&engine.txn_version_stores),
            persistence: Arc::clone(&engine.persistence),
            loading_from_disk: Arc::clone(&engine.loading_from_disk),
            segment_managers: Arc::clone(&engine.segment_managers),
            seal_fence: Arc::clone(&engine.seal_fence),
            transactional_ddl_fence: Arc::clone(&engine.transactional_ddl_fence),
            schema_epoch: Arc::clone(&engine.schema_epoch),
            engine_path: engine.path.clone(),
            lease_max_age_nanos: Arc::clone(&engine.lease_max_age_nanos),
            pending_drop_cleanups: Arc::clone(&engine.pending_drop_cleanups),
            shm,
            pending_marker_lsns: Arc::clone(&engine.pending_marker_lsns),
            max_written_marker_lsn: Arc::clone(&engine.max_written_marker_lsn),
            completed_marker_txns: Arc::clone(&engine.completed_marker_txns),
            shm_publish_lock: Arc::clone(&engine.shm_publish_lock),
            lease_present: Arc::clone(&engine.lease_present),
            failed: Arc::clone(&engine.failed),
            read_only,
        }
    }

    fn schemas(&self) -> &RwLock<FxHashMap<String, CompactArc<Schema>>> {
        &self.schemas
    }

    fn version_stores(&self) -> &RwLock<FxHashMap<String, Arc<VersionStore>>> {
        &self.version_stores
    }

    fn txn_version_stores(&self) -> &RwLock<TxnVersionStoreMap> {
        &self.txn_version_stores
    }

    fn persistence(&self) -> &Option<PersistenceManager> {
        &self.persistence
    }

    fn should_skip_wal(&self) -> bool {
        self.loading_from_disk.load(Ordering::Acquire)
    }

    /// Validate pending INSERT/UPDATE rows against cold segments at commit time.
    /// Called when seal_generation changed since the transaction's INSERT,
    /// meaning a seal may have moved conflicting rows from hot to cold.
    /// The seal read fence is held by the caller.
    fn validate_pending_against_cold(
        &self,
        txn_id: i64,
        txn_store: &Arc<RwLock<TransactionVersionStore>>,
        version_store: &Arc<VersionStore>,
        mgr: &Arc<crate::storage::volume::manifest::SegmentManager>,
    ) -> Result<()> {
        let store = txn_store.read().unwrap();
        let Some(local) = store.local_versions_ref() else {
            return Ok(());
        };

        let schema_arc = version_store.schema();
        let schema = &*schema_arc;
        let pk_idx = schema.pk_column_index();

        // Collect unique index column info + precompute defaults once.
        let unique_indexes: Vec<(Vec<usize>, Vec<String>, Vec<crate::core::Value>)> = version_store
            .get_unique_non_pk_index_columns()
            .into_iter()
            .map(|(col_indices, col_names)| {
                let defaults: Vec<crate::core::Value> = col_indices
                    .iter()
                    .map(|&ci| {
                        schema.columns[ci]
                            .default_value
                            .clone()
                            .unwrap_or(crate::core::Value::null(schema.columns[ci].data_type))
                    })
                    .collect();
                (col_indices, col_names, defaults)
            })
            .collect();

        for (row_id, versions) in local.iter() {
            let Some(version) = versions.last() else {
                continue;
            };
            if version.is_deleted() {
                continue;
            }
            let row = &version.data;

            // Classify INSERT vs UPDATE:
            // - Hot UPDATE: write_set has read_version: Some(old_row)
            // - Cold UPDATE: pending tombstone exists for this row_id
            //   (added during cold-row claim in update/update_by_row_ids)
            // - Fresh INSERT: neither of the above
            // Note: put() creates write_set entries for all rows (including
            // INSERTs with read_version: None), so write_set presence alone
            // cannot distinguish INSERT from cold-row claim.
            let has_read_version = store
                .write_set_ref()
                .and_then(|ws| ws.get(row_id))
                .and_then(|entry| entry.read_version.as_ref())
                .is_some();
            let is_cold_claim = mgr.is_pending_tombstone(txn_id, row_id);
            let is_insert = !has_read_version && !is_cold_claim;

            if is_insert {
                // PK check against cold
                if let Some(pk_col) = pk_idx {
                    if let Some(pk_val) = row.get(pk_col) {
                        if !pk_val.is_null() {
                            if let Some(cold_rid) =
                                mgr.check_value_exists_in_segments(pk_col, pk_val)
                            {
                                if !mgr.is_pending_tombstone(txn_id, cold_rid) {
                                    return Err(crate::core::Error::PrimaryKeyConstraint {
                                        row_id: cold_rid,
                                    });
                                }
                            }
                        }
                    }
                }
                // UNIQUE constraint checks against cold
                for (col_indices, col_names, defaults) in &unique_indexes {
                    let coerced: Vec<crate::core::Value> = col_indices
                        .iter()
                        .filter_map(|&idx| {
                            let val = row.get(idx)?;
                            Some(val.coerce_to_type(schema.columns[idx].data_type))
                        })
                        .collect();
                    if coerced.len() != col_indices.len() || coerced.iter().any(|v| v.is_null()) {
                        continue;
                    }
                    let values: Vec<&crate::core::Value> = coerced.iter().collect();
                    if let Some(cold_rid) =
                        mgr.find_row_id_by_values(col_indices, &values, defaults)
                    {
                        if !mgr.is_pending_tombstone(txn_id, cold_rid) {
                            return Err(crate::core::Error::UniqueConstraint {
                                index: col_names.join("_"),
                                column: col_names.join(", "),
                                value: values
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                row_id: cold_rid,
                            });
                        }
                    }
                }
            } else {
                // UPDATE: check PK if it changed
                if let Some(pk_col) = pk_idx {
                    let new_pk = row.get(pk_col);
                    let old_pk = store
                        .write_set_ref()
                        .and_then(|ws| ws.get(row_id))
                        .and_then(|entry| entry.read_version.as_ref())
                        .and_then(|rv| rv.data.get(pk_col));
                    if new_pk != old_pk {
                        if let Some(pk_val) = new_pk {
                            if !pk_val.is_null() {
                                if let Some(cold_rid) =
                                    mgr.check_value_exists_in_segments(pk_col, pk_val)
                                {
                                    if cold_rid != row_id
                                        && !mgr.is_pending_tombstone(txn_id, cold_rid)
                                    {
                                        return Err(crate::core::Error::PrimaryKeyConstraint {
                                            row_id: cold_rid,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                // UPDATE: check unique constraints excluding the row being updated.
                // Skip when the indexed columns haven't changed.
                let old_row = store
                    .write_set_ref()
                    .and_then(|ws| ws.get(row_id))
                    .and_then(|entry| entry.read_version.as_ref())
                    .map(|rv| &rv.data);

                for (col_indices, col_names, defaults) in &unique_indexes {
                    // Skip if none of the indexed columns changed
                    if let Some(old_r) = old_row {
                        let any_changed = col_indices
                            .iter()
                            .any(|&idx| row.get(idx) != old_r.get(idx));
                        if !any_changed {
                            continue;
                        }
                    }

                    let coerced: Vec<crate::core::Value> = col_indices
                        .iter()
                        .filter_map(|&idx| {
                            let val = row.get(idx)?;
                            Some(val.coerce_to_type(schema.columns[idx].data_type))
                        })
                        .collect();
                    if coerced.len() != col_indices.len() || coerced.iter().any(|v| v.is_null()) {
                        continue;
                    }
                    let values: Vec<&crate::core::Value> = coerced.iter().collect();
                    if let Some(cold_rid) =
                        mgr.find_row_id_by_values(col_indices, &values, defaults)
                    {
                        if cold_rid != row_id && !mgr.is_pending_tombstone(txn_id, cold_rid) {
                            return Err(crate::core::Error::UniqueConstraint {
                                index: col_names.join("_"),
                                column: col_names.join(", "),
                                value: values
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                row_id: cold_rid,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Record pending versions for a table to WAL (helper for commit_all_tables)
    fn record_table_to_wal(&self, txn_id: i64, table: &MVCCTable) -> Result<()> {
        if let Some(ref pm) = self.persistence() {
            let table_name = table.name();
            let pending = table.get_pending_versions();

            for (row_id, row_data, is_deleted, version_txn_id) in pending {
                let version = RowVersion {
                    txn_id: version_txn_id,
                    deleted_at_txn_id: if is_deleted { version_txn_id } else { 0 },
                    data: row_data,
                    create_time: 0,
                };

                let op = if is_deleted {
                    WALOperationType::Delete
                } else {
                    WALOperationType::Insert
                };

                pm.record_dml_operation(txn_id, table_name, row_id, op, &version)?;
            }
        }
        Ok(())
    }
}

impl EngineOperations {
    /// Mirror of `MVCCEngine::defer_for_live_readers` for the
    /// transactional-DDL DROP path. Uses the SHARED
    /// `lease_max_age_nanos` so the effective window matches
    /// the engine's; falls back to the 120s engine floor only
    /// if the engine hasn't published a value yet (which
    /// `MVCCEngine::open_engine` primes). Same fail-closed
    /// semantics: an FS error reading reader leases returns
    /// `true`.
    fn defer_for_live_readers(&self) -> bool {
        if self.engine_path.is_empty() {
            return false;
        }
        let nanos = self.lease_max_age_nanos.load(Ordering::Acquire);
        let max_age = if nanos > 0 {
            std::time::Duration::from_nanos(nanos)
        } else {
            std::time::Duration::from_secs(120)
        };
        let dir =
            std::path::Path::new(&self.engine_path).join(crate::storage::mvcc::lease::READERS_DIR);
        let _ = crate::storage::mvcc::lease::reap_stale_leases(&dir, max_age);
        match crate::storage::mvcc::lease::live_leases(&dir, max_age) {
            Ok(v) => !v.is_empty(),
            Err(_) => true,
        }
    }
}

impl TransactionEngineOperations for EngineOperations {
    fn get_table_for_transaction(
        &self,
        txn_id: i64,
        table_name: &str,
    ) -> Result<Box<dyn WriteTable>> {
        // Use Cow to avoid allocation when table_name is already lowercase (common case)
        let table_name_lower = to_lowercase_cow(table_name);

        // Get version store
        let stores = self.version_stores().read().unwrap();
        let version_store = stores
            .get(&*table_name_lower)
            .cloned()
            .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
        drop(stores);

        // Check if we have a cached transaction version store for this (txn_id, table_name)
        let txn_versions = {
            let cache = self.txn_version_stores().read().unwrap();
            let found = if let Some(txn_tables) = cache.get(txn_id) {
                // Linear search on SmallVec (fast for 1-2 tables)
                txn_tables
                    .iter()
                    .find(|(name, _)| name == &*table_name_lower)
                    .map(|(_, cached)| Arc::clone(cached))
            } else {
                None
            };
            drop(cache);

            if let Some(cached) = found {
                cached
            } else {
                // Upgrade to write lock and re-check (another thread may have inserted)
                let mut cache = self.txn_version_stores().write().unwrap();
                let txn_tables = cache.entry(txn_id).or_default();
                if let Some((_, cached)) = txn_tables
                    .iter()
                    .find(|(name, _)| name == &*table_name_lower)
                {
                    Arc::clone(cached)
                } else {
                    let new_store = Arc::new(RwLock::new(TransactionVersionStore::new(
                        Arc::clone(&version_store),
                        txn_id,
                    )));
                    txn_tables.push((
                        table_name_lower.clone().into_owned().into(),
                        Arc::clone(&new_store),
                    ));
                    new_store
                }
            }
        };

        // Create MVCC table with shared transaction version store
        let table = MVCCTable::new_with_shared_store(txn_id, version_store, txn_versions);

        // If segments exist for this table, wrap in SegmentedTable.
        // For snapshot isolation transactions, pass the begin_seq so tombstone
        // filtering respects the snapshot's point-in-time view of cold data.
        let mgrs = self.segment_managers.read().unwrap();
        if let Some(mgr) = mgrs.get(&*table_name_lower) {
            if mgr.has_segments() {
                let mgr = Arc::clone(mgr);
                drop(mgrs);
                if self.registry.get_isolation_level(txn_id)
                    == crate::IsolationLevel::SnapshotIsolation
                {
                    let begin_seq = self.registry.get_transaction_begin_sequence(txn_id) as u64;
                    return Ok(Box::new(
                        crate::storage::volume::table::SegmentedTable::with_snapshot_seq(
                            Box::new(table),
                            mgr,
                            begin_seq,
                        ),
                    ));
                }
                // Read-only engines need the SWMR-aware row_count
                // path; writers stay on the O(1) formula. See the
                // `read_only` field doc on SegmentedTable.
                return Ok(Box::new(if self.read_only {
                    crate::storage::volume::table::SegmentedTable::new_read_only(
                        Box::new(table),
                        mgr,
                    )
                } else {
                    crate::storage::volume::table::SegmentedTable::new(Box::new(table), mgr)
                }));
            }
        }

        Ok(Box::new(table))
    }

    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn WriteTable>> {
        // Refuse if the engine is in the catastrophic-failure state.
        // Same shape as the drop_table / rename_table guards: this
        // trait method is reachable from inside an open transaction
        // (`MvccTransaction::create_table`) where `check_active()`
        // is the only gate. Without this guard, transactional
        // CREATE could insert the schema + VersionStore — a later
        // record_commit would then hit the failed latch and return
        // Err, but the new table would remain live in memory and
        // `rollback_ddl` couldn't remove it (drop_table now refuses
        // under the latch).
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "create_table refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart \
                 the process; recovery will discard the markerless transaction.",
            ));
        }
        let table_name = name.to_lowercase();

        // Create version store for this table (before acquiring locks)
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            registry_as_visibility_checker(&self.registry),
        ));

        // Register PkIndex if table has a primary key
        register_pk_index(&schema, &version_store);

        // Atomically check-and-insert under write lock to
        // prevent TOCTOU race. The pending-drop check happens
        // INSIDE the schemas write critical section using the
        // same `schemas → pending` lock order DROP follows
        // (see `EngineOperations::drop_table`); without that,
        // a concurrent DROP that grabs schemas + inserts
        // pending + removes schema could land between this
        // CREATE's pre-check and its schemas acquire,
        // letting CREATE see "schemas missing AND pending
        // empty" even though a DROP is in progress.
        {
            let mut schemas = self.schemas().write().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists(table_name.to_string()));
            }
            // See the matching guard in `MVCCEngine::create_table`
            // for the full rationale.
            if self.pending_drop_cleanups.lock().contains(&table_name) {
                return Err(Error::internal(format!(
                    "CREATE TABLE refused: a prior DROP/TRUNCATE for '{}' is still \
                     pending physical cleanup (live cross-process readers, or a \
                     cleanup I/O failure). Wait for the orphan sweep to drain \
                     before recreating the table.",
                    table_name
                )));
            }
            schemas.insert(table_name.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores().write().unwrap();
            stores.insert(table_name, Arc::clone(&version_store));
        }

        // Bump schema_epoch so cache consumers
        // (`fk_reverse_cache`, compiled DML fast paths,
        // SWMR schema-drift check) re-derive against the new
        // catalog. Done AFTER the schemas / version_stores
        // mutation so any concurrent reader observing the
        // higher epoch also sees the inserted entries. The
        // direct (auto-commit) `MVCCEngine::create_table`
        // bumps from the same point for identical reasons —
        // both transactional and direct paths need this for
        // FK enforcement to stay coherent across DDL.
        self.schema_epoch.fetch_add(1, Ordering::Release);

        // Create transaction version store with txn_id 0 (will be set by caller)
        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), 0);

        // Create MVCC table
        let table = MVCCTable::new(0, version_store, txn_versions);

        Ok(Box::new(table))
    }

    fn drop_table(&self, name: &str) -> Result<DropSnapshot> {
        // Refuse if the engine is in the catastrophic-failure
        // state. This trait method is reachable from inside an
        // already-open transaction (`MvccTransaction::drop_table`)
        // — `check_active()` is the only gate, so the begin-time
        // latch check from
        // `begin_writable_transaction_with_level_internal` doesn't
        // protect us. Without this guard a transactional DROP can
        // mutate schemas/version_stores even though the
        // surrounding txn never reaches the commit-phase deferred-
        // DDL flush.
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "drop_table refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart \
                 the process; recovery will discard the markerless transaction.",
            ));
        }
        let table_name_lower = name.to_lowercase();

        // Snapshot the parent schema AND every child schema
        // whose FK constraints reference the parent BEFORE
        // mutating, then perform the strip under the same
        // schemas write lock for atomicity. The snapshot is
        // returned so the surrounding txn can store it in
        // `ddl_log` for full rollback (parent + child FK
        // restoration).
        let parent_schema: Schema;
        let mut child_schemas: Vec<(String, Schema)> = Vec::new();
        {
            let mut schemas = self.schemas().write().unwrap();
            if !schemas.contains_key(&table_name_lower) {
                return Err(Error::TableNotFound(table_name_lower.to_string()));
            }
            // Clone the parent schema before removal.
            parent_schema = schemas
                .get(&table_name_lower)
                .map(|s| (**s).clone())
                .ok_or_else(|| Error::TableNotFound(table_name_lower.to_string()))?;
            // Capture every child whose FK constraints would be
            // stripped by `strip_fk_references`.
            for (cname, csch) in schemas.iter() {
                if csch
                    .foreign_keys
                    .iter()
                    .any(|fk| fk.referenced_table == table_name_lower)
                {
                    child_schemas.push((cname.clone(), (**csch).clone()));
                }
            }
            // Mark dropping under the same schemas write
            // lock that makes the table absent — same race
            // closure as the auto-commit
            // `MVCCEngine::drop_table_internal` path. A
            // concurrent same-name CREATE
            // (`MVCCEngine::create_table` /
            // `EngineOperations::create_table`) acquires the
            // same lock for its check-and-insert; with the
            // pending entry in place under the same critical
            // section, that CREATE refuses cleanly instead
            // of writing into the directory the dropped
            // table's manifest still occupies.
            self.pending_drop_cleanups
                .lock()
                .insert(table_name_lower.clone());
            schemas.remove(&table_name_lower);
            let version_stores = self.version_stores().read().unwrap();
            strip_fk_references(&mut schemas, &version_stores, &table_name_lower);
        }

        // Remove the version store. Before closing, capture
        // every secondary (non-PK) index so rollback can
        // recreate them on the fresh VersionStore that
        // `ops.create_table` will install. PK indexes are
        // auto-derived from the schema by `register_pk_index`
        // and don't need to be in the snapshot.
        let removed_store: Option<Arc<VersionStore>> = {
            let mut stores = self.version_stores().write().unwrap();
            stores.remove(&table_name_lower)
        };
        let mut captured_indexes: Vec<Vec<u8>> = Vec::new();
        if let Some(ref store) = removed_store {
            let _ = store.for_each_index(|index| {
                if index.index_type() == crate::core::IndexType::PrimaryKey {
                    return Ok(());
                }
                let meta = super::persistence::IndexMetadata {
                    name: index.name().to_string(),
                    table_name: table_name_lower.clone(),
                    column_names: index.column_names().to_vec(),
                    column_ids: index.column_ids().to_vec(),
                    data_types: index.data_types().to_vec(),
                    is_unique: index.is_unique(),
                    index_type: index.index_type(),
                    hnsw_m: index.hnsw_m(),
                    hnsw_ef_construction: index.hnsw_ef_construction(),
                    hnsw_ef_search: index.default_ef_search().map(|v| v as u16),
                    hnsw_distance_metric: index.hnsw_distance_metric(),
                };
                captured_indexes.push(meta.serialize());
                Ok(())
            });
        }
        if let Some(store) = removed_store {
            store.close();
        }

        // NEITHER the segment-manager clear NOR the volume-
        // file delete happens here. Both are POST-COMMIT
        // operations executed by `finalize_committed_drops`
        // after the user txn's commit marker is durable. The
        // pre-deferred-DDL path did them inline at drop_table
        // time, which left a window where a crash before the
        // commit marker durably landed produced a live
        // catalog entry pointing at vanished files. Deferring
        // them keeps in-memory and on-disk state aligned with
        // the WAL durability boundary: rollback discards the
        // in-memory mutation AND retains files / segments;
        // commit triggers the physical reap only after the
        // marker is durable.

        // Bump schema_epoch — same rationale as in
        // `create_table` above. The DROP also stripped FK
        // constraints from any child schemas via
        // `strip_fk_references`, so cached
        // `find_referencing_fks(parent)` results AND any
        // compiled DML against the affected children must be
        // invalidated.
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(DropSnapshot {
            parent_schema,
            child_schemas,
            indexes: captured_indexes,
        })
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        let schemas = self.schemas().read().unwrap();
        Ok(schemas.keys().cloned().collect())
    }

    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()> {
        // Refuse if the engine is in the catastrophic-failure state.
        // Same shape as `drop_table` above: this trait method is
        // reachable from inside an open transaction
        // (`MvccTransaction::rename_table`) where `check_active()`
        // is the only gate, so the begin-time latch check from
        // `begin_writable_transaction_with_level_internal` doesn't
        // protect us. Without this guard, transactional RENAME can
        // mutate schemas/version_stores AND rename the on-disk
        // volume + tombstone directories before the executor's
        // later `record_alter_table_rename` reaches the
        // `record_ddl` latch — leaving durable state diverged from
        // what recovery will rebuild.
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "rename_table refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart \
                 the process; recovery will discard the markerless transaction.",
            ));
        }
        let old_name_lower = old_name.to_lowercase();
        let new_name_lower = new_name.to_lowercase();

        // Atomically check-and-rename under write lock to prevent TOCTOU race
        {
            let mut schemas = self.schemas().write().unwrap();
            if !schemas.contains_key(&old_name_lower) {
                return Err(Error::TableNotFound(old_name_lower.to_string()));
            }
            if schemas.contains_key(&new_name_lower) {
                return Err(Error::TableAlreadyExists(new_name_lower.to_string()));
            }
            if let Some(mut schema_arc) = schemas.remove(&old_name_lower) {
                let schema = CompactArc::make_mut(&mut schema_arc);
                schema.table_name = new_name.to_string();
                schema.table_name_lower = new_name_lower.clone();
                schemas.insert(new_name_lower.clone(), schema_arc);
            }
        }

        // Rename in version stores
        {
            let mut stores = self.version_stores().write().unwrap();
            if let Some(store) = stores.remove(&old_name_lower) {
                {
                    let mut vs_schema_guard = store.schema_mut();
                    let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                    schema.table_name = new_name.to_string();
                    schema.table_name_lower = new_name_lower.clone();
                }
                stores.insert(new_name_lower.clone(), store);
            }
        }

        // Move segment manager to new name and update its manifest table_name
        // (persist() uses manifest.table_name for the on-disk directory)
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.remove(&old_name_lower) {
                mgr.manifest_mut().table_name =
                    crate::common::SmartString::from(new_name_lower.as_str());
                mgrs.insert(new_name_lower.clone(), mgr);
            }
        }
        // Rename on-disk volume directory and tombstones
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let old_dir = vol_dir.join(&old_name_lower);
                let new_dir = vol_dir.join(&new_name_lower);
                if old_dir.exists() {
                    if let Err(e) = std::fs::rename(&old_dir, &new_dir) {
                        // Revert ALL in-memory renames on disk failure
                        {
                            let mut schemas = self.schemas().write().unwrap();
                            if let Some(mut schema_arc) = schemas.remove(&new_name_lower) {
                                let schema = CompactArc::make_mut(&mut schema_arc);
                                schema.table_name = old_name.to_string();
                                schema.table_name_lower = old_name_lower.clone();
                                schemas.insert(old_name_lower.clone(), schema_arc);
                            }
                        }
                        {
                            let mut stores = self.version_stores().write().unwrap();
                            if let Some(store) = stores.remove(&new_name_lower) {
                                {
                                    let mut vs_schema_guard = store.schema_mut();
                                    let schema = CompactArc::make_mut(&mut *vs_schema_guard);
                                    schema.table_name = old_name.to_string();
                                    schema.table_name_lower = old_name_lower.clone();
                                }
                                stores.insert(old_name_lower.clone(), store);
                            }
                        }
                        {
                            let mut mgrs = self.segment_managers.write().unwrap();
                            if let Some(mgr) = mgrs.remove(&new_name_lower) {
                                mgr.manifest_mut().table_name =
                                    crate::common::SmartString::from(old_name_lower.as_str());
                                mgrs.insert(old_name_lower.clone(), mgr);
                            }
                        }
                        return Err(Error::Internal {
                            message: format!("Failed to rename volume directory: {}", e),
                        });
                    }
                }
                let snap_dir = pm.path().join("snapshots");
                let old_ts = snap_dir.join(&old_name_lower).join("tombstones.dat");
                let new_ts_dir = snap_dir.join(&new_name_lower);
                let new_ts = new_ts_dir.join("tombstones.dat");
                if old_ts.exists() {
                    let _ = std::fs::create_dir_all(&new_ts_dir);
                    let _ = std::fs::rename(&old_ts, &new_ts);
                }
            }
        }

        // Bump schema_epoch — same rationale as in
        // `create_table` / `drop_table`. A rename invalidates
        // any cached lookup keyed by the old or new name and
        // any compiled DML targeting either entry.
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    fn commit_table(&self, txn_id: i64, table: &dyn WriteTable) -> Result<()> {
        // Skip WAL writes during recovery replay
        if self.should_skip_wal() {
            return Ok(());
        }

        // Record DML operations to WAL before the table commits
        if let Some(ref pm) = self.persistence() {
            if pm.is_enabled() {
                let table_name = table.name();
                let pending = table.get_pending_versions();

                for (row_id, row_data, is_deleted, version_txn_id) in pending {
                    // Create a RowVersion for serialization
                    let version = RowVersion {
                        txn_id: version_txn_id,
                        deleted_at_txn_id: if is_deleted { version_txn_id } else { 0 },
                        data: row_data,
                        create_time: 0, // Not needed for persistence
                    };

                    // Determine operation type
                    let op = if is_deleted {
                        WALOperationType::Delete
                    } else {
                        // Check if this is an update vs insert by looking at global store
                        // For simplicity, we record everything as Insert
                        // (at replay time, the version store handles deduplication)
                        WALOperationType::Insert
                    };

                    pm.record_dml_operation(txn_id, table_name, row_id, op, &version)?;
                }
            }
        }

        Ok(())
    }

    fn rollback_table(&self, _txn_id: i64, table: &dyn WriteTable) {
        // The Table trait now has a rollback method.
        // This callback is for any engine-level rollback actions.
        let _ = table;
    }

    fn record_commit(&self, txn_id: i64, commit_seq: i64) -> Result<u64> {
        // Skip WAL writes during recovery replay
        if self.should_skip_wal() {
            return Ok(0);
        }
        // Defense in depth: a writer that started before the
        // catastrophic-failure latch was set would otherwise be
        // able to publish its marker AFTER the latch, making its
        // (legitimate) durable state coexist with the prior
        // markerless commit's parent-store rows in seal/backup
        // outputs once the engine restarts. Refuse the marker
        // write so the in-flight commit takes the same fatal
        // path as the original failure.
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "record_commit refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart the \
                 process; recovery will discard markerless transactions.",
            ));
        }

        // Record commit in WAL — propagate errors since missing commit records
        // means crash recovery won't replay this transaction's changes.
        // commit_seq is the value start_commit allocated; reader's WAL-tail
        // (SWMR v2) uses it for snapshot_seq-compatible tombstone tagging.
        // Returns the marker LSN so the caller can publish it to db.shm
        // (SWMR v2 Phase C). Returns 0 when persistence is disabled.
        if let Some(ref pm) = self.persistence() {
            if pm.is_enabled() {
                // Hold the pending lock ACROSS the
                // marker write. The prior version called
                // `pm.record_commit` (which writes the marker and
                // may flush it depending on SyncMode) BEFORE
                // inserting the LSN into pending. A concurrent
                // commit's publish in that gap would see no pending
                // entry for this txn, see max_written already
                // bumped (if the bump happened), compute safe =
                // max_written, and advertise visibility past this
                // marker before its `complete_commit` had fired.
                // Holding the mutex across both steps closes the
                // gap: by the time anyone else sees pending or
                // max_written, our LSN is in BOTH (so the
                // safe-visible computation correctly caps below us
                // until our publish removes us).
                use std::sync::atomic::Ordering;
                let mut pending = self.pending_marker_lsns.lock();
                let lsn = pm.record_commit(txn_id, commit_seq)?;
                if lsn > 0 {
                    pending.insert(lsn);
                    self.max_written_marker_lsn.fetch_max(lsn, Ordering::AcqRel);
                }
                drop(pending);
                return Ok(lsn);
            }
        }
        Ok(0)
    }

    fn publish_visible_commit_lsn(&self, txn_id: i64, lsn: u64) {
        // Skip if no marker was actually written (e.g., persistence
        // disabled, recovery replay, or in-memory engine). 0 means
        // "nothing to publish".
        if lsn == 0 {
            return;
        }
        // Compute the SAFE-VISIBLE watermark.
        // `lsn` is THIS txn's marker LSN, but other txns may have
        // marker LSNs lower OR higher than `lsn` that are still
        // pending complete_commit. We must NOT advertise visibility
        // past any pending marker — readers tailing the WAL would
        // otherwise see those still-Committing txns as committed
        // while in-process callers still treat them as invisible.
        //
        // Rule: safe_visible = (min(pending) - 1) if pending is
        // non-empty, else max_written_marker_lsn. Remove our own
        // LSN from pending FIRST, then compute.
        let safe_visible = {
            let mut pending = self.pending_marker_lsns.lock();
            pending.remove(&lsn);
            if let Some(&min_pending) = pending.iter().next() {
                // Some other txn's marker is still pending complete.
                // Cap visibility just below it.
                min_pending.saturating_sub(1)
            } else {
                // No pending markers — everything written so far has
                // both committed AND complete-committed. Safe to
                // advertise up to the highest marker ever written.
                use std::sync::atomic::Ordering;
                self.max_written_marker_lsn.load(Ordering::Acquire)
            }
        };
        if txn_id > 0 {
            self.completed_marker_txns.lock().insert(lsn, txn_id);
        }
        if safe_visible == 0 {
            // Nothing to publish (writer hasn't recorded any commit
            // marker yet, or this is a synthetic/zero LSN).
            return;
        }
        let publish_lsn = cap_visible_lsn_by_flushed(&self.persistence, safe_visible);
        if let Some(handle) = self.shm.as_ref() {
            // CRITICAL store ordering: watermark FIRST (Release),
            // then visible_commit_lsn (AcqRel). Release-Acquire
            // pairing on visible_commit_lsn guarantees readers
            // observing the new visible LSN also observe the
            // matching watermark.
            //
            // The watermark we publish for `safe_visible` MUST
            // include this committing txn's first DML LSN —
            // otherwise a reader observing the new
            // `visible_commit_lsn` would seed
            // `next_entry_floor` past this txn's DML and miss
            // its pre-window rows on the next refresh. We
            // achieve that by reading
            // `wal.oldest_active_txn_lsn()` BEFORE
            // `clear_active_txn(txn_id)` — at that point the
            // WAL manager's `active_txn_first_lsn` still
            // contains this txn's contribution.
            //
            // After the safe-visible publish we clear this
            // txn from the WAL manager AND issue a SECOND
            // watermark-only publish so a future reader (or
            // the existing reader's NEXT refresh) sees the
            // freshly-recomputed `oldest_active`, not the
            // stale value pinning this txn's DML LSN. Without
            // that follow-up, a workload that commits a long
            // txn and then quiesces would leave shm
            // advertising the long txn's first WAL LSN
            // forever, holding the writer's WAL truncate
            // floor at that LSN for the lifetime of every
            // live reader pin.
            //
            // Lease-presence fast path. When no cross-process
            // reader currently holds a live lease, skip the
            // `shm_publish_lock` + `publish_seq` seqlock dance
            // and just `fetch_max` the visible / oldest fields.
            // The Release-Acquire ordering on `visible_commit_lsn`
            // is sufficient for the two AtomicU64 fields without
            // `publish_seq` — the seqlock guards multi-word
            // updates, which the current ShmHeader doesn't have.
            // Truncate clamp uses `visible_commit_lsn` so the
            // store via `fetch_max` keeps it advancing.
            //
            // The cleanup loop refreshes `lease_present` on each
            // tick and triggers a barrier publish on
            // `false → true` transition, so any reader that
            // arrives between scans sees coherent shm state on
            // its first refresh.
            //
            // Default `lease_present = true` (conservative), so
            // the fast path activates only after a lease scan
            // has actively proved no reader is present.
            let do_full_publish = self.lease_present.load(Ordering::Acquire);
            if !do_full_publish {
                // Fast path: only advance `visible_commit_lsn`
                // so the truncate clamp moves forward. Do NOT
                // touch `oldest_active_txn_lsn` here — concurrent
                // fast-path commits can't keep it consistent
                // without serialization, and no readers are
                // sampling shm during the no-readers window
                // anyway. The next `false → true` transition
                // triggers a barrier publish that re-samples
                // and stores the correct oldest under the
                // seqlock so the first arriving reader sees a
                // coherent snapshot.
                //
                if publish_lsn > 0 {
                    handle
                        .header()
                        .visible_commit_lsn
                        .fetch_max(publish_lsn, Ordering::AcqRel);
                }
                let visible_after = handle.header().visible_commit_lsn.load(Ordering::Acquire);
                clear_published_completed_txns(
                    &self.completed_marker_txns,
                    &self.persistence,
                    visible_after,
                );
                return;
            }

            // Slow path (readers present): full seqlock publish.
            //
            // Serialize against concurrent publishes (commit and
            // DDL). EngineOperations holds a CLONE of the Arc<ShmHandle>
            // (not `MVCCEngine.shm.lock()`), so without this shared
            // mutex two auto-commit publishes could interleave their
            // odd/even bumps and tear the seqlock.
            let _publish_guard = self.shm_publish_lock.lock();
            // Skip the visible_commit_lsn publish when it wouldn't
            // advance — same rationale as in
            // `MVCCEngine::publish_visible_commit_lsn`: a stale
            // publish would overwrite the floor with a freshly-
            // sampled value paired with the existing higher
            // visible LSN, letting readers skip pre-window DML.
            // CRITICAL: only skip the SHM PUBLISH BLOCK, not the
            // outer function — `clear_active_txn(txn_id)` and
            // the watermark republish below must run for any
            // committed txn so future publishes / reader
            // refreshes observe a fresh `oldest_active_txn_lsn`.
            if publish_lsn > handle.header().visible_commit_lsn.load(Ordering::Acquire) {
                // Seqlock publish: bump-odd BEFORE field stores so
                // a concurrent reader sample retries.
                handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                if let Some(pm) = self.persistence.as_ref().as_ref() {
                    if let Some(wal) = pm.wal() {
                        let oldest = wal.oldest_active_txn_lsn();
                        handle
                            .header()
                            .oldest_active_txn_lsn
                            .store(oldest, Ordering::Release);
                    }
                }
                handle
                    .header()
                    .visible_commit_lsn
                    .fetch_max(publish_lsn, Ordering::AcqRel);
                // Bump-even AFTER both stores: pair is coherent.
                handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
            }

            // Clear every completed txn whose marker is now covered
            // by the published visible LSN, then re-publish the
            // now-recomputed watermark. The
            // visible_commit_lsn does NOT change in this second
            // publish — only `oldest_active_txn_lsn` may advance
            // (typically to a higher in-flight txn's LSN, or to
            // `u64::MAX` when no in-flight user txns remain).
            // Both stores happen under the same publish_guard so
            // the seqlock pair stays atomic.
            let visible_after = handle.header().visible_commit_lsn.load(Ordering::Acquire);
            let cleared = clear_published_completed_txns(
                &self.completed_marker_txns,
                &self.persistence,
                visible_after,
            );
            if cleared {
                if let Some(pm) = self.persistence.as_ref().as_ref() {
                    if let Some(wal) = pm.wal() {
                        let fresh_oldest = wal.oldest_active_txn_lsn();
                        let current_oldest = handle
                            .header()
                            .oldest_active_txn_lsn
                            .load(Ordering::Acquire);
                        if fresh_oldest != current_oldest {
                            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                            handle
                                .header()
                                .oldest_active_txn_lsn
                                .store(fresh_oldest, Ordering::Release);
                            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                }
            }
        } else {
            // No shm: still need to clear the WAL state so
            // future publishes (if shm appears) reflect the
            // cleared txn.
            self.completed_marker_txns.lock().remove(&lsn);
            if let Some(pm) = self.persistence.as_ref().as_ref() {
                if let Some(wal) = pm.wal() {
                    wal.clear_active_txn(txn_id);
                }
            }
        }
    }

    fn flush_transactional_ddl(&self, txn_id: i64, ops: &[DeferredDdlOp]) -> Result<()> {
        // Emit a durable WAL entry for each op under
        // `txn_id` (no auto-commit marker). Recovery /
        // SWMR-tail visibility is gated by the user txn's
        // commit marker (`record_commit`); a crash between
        // these writes and the marker leaves the entries
        // orphaned and recovery skips them.
        //
        // Refuse outright if the engine is in the
        // catastrophic-failure state. The transactional
        // commit path's pre-`record_commit` checks already
        // gate on this latch, but a flip between those checks
        // and now would otherwise emit user-txn DDL records
        // alongside a markerless commit.
        if self.should_skip_wal() {
            // Recovery replay or shutdown — emit nothing.
            return Ok(());
        }
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "flush_transactional_ddl refused: engine is in the \
                 catastrophic-failure state from a prior commit's marker write \
                 failure. Restart the process; recovery will discard the \
                 markerless transaction.",
            ));
        }
        let pm_guard = self.persistence();
        let pm = match pm_guard.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()),
        };
        for op in ops {
            // Recheck the latch before each write so a
            // concurrent flip doesn't slip a durable DDL past
            // the gate.
            if self.failed.load(Ordering::Acquire) {
                return Err(Error::internal(
                    "flush_transactional_ddl refused mid-batch: engine entered \
                     the catastrophic-failure state.",
                ));
            }
            match op {
                DeferredDdlOp::Create { name, schema_data } => {
                    pm.record_transactional_ddl(
                        txn_id,
                        name,
                        WALOperationType::CreateTable,
                        schema_data,
                    )?;
                }
                DeferredDdlOp::Drop { name } => {
                    pm.record_transactional_ddl(txn_id, name, WALOperationType::DropTable, &[])?;
                }
                DeferredDdlOp::CreateIndex {
                    table_name,
                    metadata,
                } => {
                    pm.record_transactional_ddl(
                        txn_id,
                        table_name,
                        WALOperationType::CreateIndex,
                        metadata,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn finalize_committed_drops(&self, names: &[String]) {
        // Post-commit physical reap. Called only AFTER the
        // user's commit marker is durable + visible, so a
        // crash between marker durability and these
        // operations leaves orphan files / segment state that
        // the next checkpoint or compaction can clean up.
        let defer = self.defer_for_live_readers();
        for name in names {
            let table_name_lower = name.to_lowercase();
            // Clear in-memory segment state first (prevents
            // phantom rows on a re-create using the same
            // name).
            {
                let mut mgrs = self.segment_managers.write().unwrap();
                if let Some(mgr) = mgrs.get(&table_name_lower) {
                    mgr.clear();
                }
                mgrs.remove(&table_name_lower);
            }
            // Then delete on-disk volume files when no live
            // cross-process reader could still hold a stale
            // manifest pointer. Same defer-when-readers-live
            // treatment as `MVCCEngine::drop_table_internal`:
            // while readers are live the directory and its
            // `vol_NNNN.vol` files stay UNTOUCHED at their
            // original path so a reader's lazy `ensure_volume`
            // continues to resolve. Future
            // `sweep_orphan_table_dirs` (run from checkpoint /
            // open) reaps the leftover directory once readers
            // detach.
            //
            // Mark this table in `pending_drop_cleanups`
            // WHENEVER the immediate unlink doesn't run —
            // that is, on `defer=true` AND on Err. The
            // user's COMMIT has already returned success
            // before this runs, so we can't propagate Err
            // up; tracking via the pending set is the only
            // signal `compute_wal_truncate_floor` has to
            // refuse WAL truncation while the leftover
            // `manifest.bin` is still on disk. Without this,
            // a checkpoint that runs before the next sweep
            // could re-record only the live tables and
            // truncate WAL past the `DropTable` record;
            // V1 readers (which don't pin WAL via
            // `min_pinned_reader_lsn`) wouldn't keep that
            // truncation from advancing, and after restart
            // the dropped table would resurface from the
            // leftover manifest. `sweep_orphan_table_dirs`
            // (run from checkpoint / open under
            // `!defer_for_live_readers`) retries the removal
            // and clears the set on success.
            if let Some(ref pm) = *self.persistence() {
                if pm.is_enabled() {
                    let vol_dir = pm.path().join("volumes");
                    // The transactional `drop_table` ALREADY
                    // inserted into `pending_drop_cleanups`
                    // under the schemas write lock that made
                    // the table absent — that's what blocks
                    // concurrent same-name CREATE during the
                    // open-transaction window. Here we only
                    // need to CLEAR on non-deferred SUCCESS;
                    // defer / Err leave the existing entry
                    // in place for the orphan sweep to drain.
                    match crate::storage::volume::io::delete_table_volumes_when_safe(
                        &vol_dir,
                        &table_name_lower,
                        defer,
                    ) {
                        Ok(()) => {
                            if !defer {
                                self.pending_drop_cleanups.lock().remove(&table_name_lower);
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: post-commit volume cleanup failed for '{}': {} \
                                 (deferring WAL truncation until cleanup succeeds)",
                                table_name_lower, e
                            );
                        }
                    }
                } else {
                    // Persistence disabled / no on-disk state
                    // exists for this table — clear the
                    // optimistic mark deposited by drop_table.
                    self.pending_drop_cleanups.lock().remove(&table_name_lower);
                }
            } else {
                // No persistence at all: same as above.
                self.pending_drop_cleanups.lock().remove(&table_name_lower);
            }
        }
    }

    fn build_index_metadata(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
        hnsw_m: Option<u16>,
        hnsw_ef_construction: Option<u16>,
        hnsw_ef_search: Option<u16>,
        hnsw_distance_metric: Option<u8>,
    ) -> Result<Vec<u8>> {
        // Mirrors the column-id / data-type derivation in
        // `MVCCEngine::record_create_index`. Returning the
        // serialized payload (rather than writing to WAL
        // here) lets the surrounding txn stage the entry on
        // its `ddl_log` for deferred commit-time flush so
        // the WAL ordering is CreateTable -> CreateIndex ->
        // commit marker — a recovery replay won't observe an
        // index whose parent table doesn't exist yet.
        let table_name_lower = table_name.to_lowercase();
        let schema = {
            let schemas = self.schemas().read().unwrap();
            schemas
                .get(&table_name_lower)
                .cloned()
                .ok_or_else(|| Error::TableNotFound(table_name_lower.clone()))?
        };
        let col_index_map = schema.column_index_map();
        let mut column_ids = Vec::with_capacity(column_names.len());
        let mut data_types = Vec::with_capacity(column_names.len());
        for col_name in column_names {
            let col_name_lower = col_name.to_lowercase();
            if let Some(&idx) = col_index_map.get(&col_name_lower) {
                column_ids.push(idx as i32);
                data_types.push(schema.columns[idx].data_type);
            } else {
                // Same fallback as `record_create_index`:
                // if a column doesn't exist (e.g., the index
                // refers to a stale name) skip the entry by
                // returning an empty payload. Caller treats
                // empty as "no flush needed."
                return Ok(Vec::new());
            }
        }
        let index_meta = super::persistence::IndexMetadata {
            name: index_name.to_string(),
            table_name: table_name.to_string(),
            column_names: column_names.to_vec(),
            column_ids,
            data_types,
            is_unique,
            index_type,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_distance_metric,
        };
        Ok(index_meta.serialize())
    }

    fn restore_table_indexes(&self, table_name: &str, indexes: &[Vec<u8>]) -> Result<()> {
        if indexes.is_empty() {
            return Ok(());
        }
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "restore_table_indexes refused: engine is in the \
                 catastrophic-failure state.",
            ));
        }
        let table_name_lower = table_name.to_lowercase();
        let store = {
            let stores = self.version_stores().read().unwrap();
            stores
                .get(&table_name_lower)
                .cloned()
                .ok_or_else(|| Error::TableNotFound(table_name_lower.clone()))?
        };
        // Find the segment manager — it survives the
        // transactional drop because `finalize_committed_drops`
        // only runs post-commit. On rollback the cold rows are
        // therefore still live, and HNSW indexes MUST include
        // them (vector search has no segment-scan fallback).
        let segment_mgr = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.get(&table_name_lower).cloned()
        };
        for serialized in indexes {
            let meta = crate::storage::mvcc::persistence::IndexMetadata::deserialize(serialized)?;
            // Step 1: recreate the index structure +
            // populate from the (currently empty) hot store.
            store.create_index_from_metadata(&meta, false)?;
            // Step 2: HNSW-only — populate from cold
            // segments. Other index types (BTree, Hash,
            // Bitmap, MultiColumn) intentionally cover only
            // hot rows; cold scans use zone maps + dictionary
            // pre-filters, so leaving cold rows out of the
            // in-memory index is correct. HNSW has no such
            // fallback path, so without this step a restored
            // HNSW index would miss every sealed vector and
            // unique-vector checks would ignore cold
            // duplicates.
            if meta.index_type == crate::core::IndexType::Hnsw {
                if let Some(ref mgr) = segment_mgr {
                    if mgr.has_segments() {
                        let cols: Vec<&str> =
                            meta.column_names.iter().map(|s| s.as_str()).collect();
                        if let Err(e) =
                            crate::storage::volume::table::populate_index_from_cold_segments(
                                &store,
                                mgr.as_ref(),
                                &meta.name,
                                &cols,
                            )
                        {
                            // Roll back the partially-built
                            // index so a retry sees a clean
                            // state. Mirrors the rollback in
                            // `SegmentedTable::create_index_with_type`.
                            let _ = store.remove_index(&meta.name);
                            return Err(e);
                        }
                    }
                }
            }
        }
        // Bump schema_epoch — restoring secondary indexes
        // changes the index set on the table, so cached
        // compiled DML / planner choices keyed on the prior
        // (no-index) state must rederive.
        self.schema_epoch.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn release_pending_drop_cleanup(&self, name: &str) {
        // The DROP that placed the entry never reached
        // durability (rollback) and the inverse
        // `create_table` is about to run; clearing here
        // lets that create_table acquire the schemas write
        // lock without tripping the same-name DROP-in-
        // progress guard. No-op when not present.
        self.pending_drop_cleanups.lock().remove(name);
    }

    fn restore_child_fk_schemas(&self, schemas: &[(String, Schema)]) -> Result<()> {
        if schemas.is_empty() {
            return Ok(());
        }
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::internal(
                "restore_child_fk_schemas refused: engine is in the \
                 catastrophic-failure state.",
            ));
        }
        // Acquire schemas write + version_stores read in the
        // same scope so the catalog and per-VS schema updates
        // are consistent. Lock-ordering rule: schemas FIRST,
        // then version_stores (matches `MVCCEngine::drop_table_internal`'s
        // revert path).
        {
            let mut catalog = self.schemas().write().unwrap();
            let stores = self.version_stores().read().unwrap();
            for (cname, csch) in schemas {
                if let Some(vs) = stores.get(cname.as_str()) {
                    *vs.schema_mut() = CompactArc::new(csch.clone());
                }
                catalog.insert(cname.clone(), CompactArc::new(csch.clone()));
            }
        }
        // Bump schema_epoch so any cached
        // `find_referencing_fks(parent)` result that was
        // computed against the FK-stripped catalog is
        // invalidated and recomputes against the restored
        // child constraints.
        self.schema_epoch.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn release_pending_ddl_marker(&self, lsn: u64) {
        // Mirrors the pending-drain shape of
        // `publish_visible_commit_lsn` but without the txn-side
        // bookkeeping (no `clear_active_txn`; DDL doesn't have
        // an active_txn record, only a marker LSN parked in
        // pending). `lsn = 0` means no marker was actually
        // pinned (in-memory engine, persistence disabled, or
        // `should_skip_wal()` true) — nothing to do.
        if lsn == 0 {
            return;
        }
        let safe_visible = {
            let mut pending = self.pending_marker_lsns.lock();
            pending.remove(&lsn);
            if let Some(&min_pending) = pending.iter().next() {
                min_pending.saturating_sub(1)
            } else {
                self.max_written_marker_lsn.load(Ordering::Acquire)
            }
        };
        if safe_visible == 0 {
            return;
        }
        let publish_lsn = cap_visible_lsn_by_flushed(&self.persistence, safe_visible);
        if publish_lsn == 0 {
            return;
        }
        if let Some(handle) = self.shm.as_ref() {
            // Same shm publish dance as
            // `publish_visible_commit_lsn` — see that method
            // for the seqlock + watermark ordering rationale.
            let _publish_guard = self.shm_publish_lock.lock();
            if publish_lsn > handle.header().visible_commit_lsn.load(Ordering::Acquire) {
                handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                if let Some(pm) = self.persistence.as_ref().as_ref() {
                    if let Some(wal) = pm.wal() {
                        let oldest = wal.oldest_active_txn_lsn();
                        handle
                            .header()
                            .oldest_active_txn_lsn
                            .store(oldest, Ordering::Release);
                    }
                }
                handle
                    .header()
                    .visible_commit_lsn
                    .fetch_max(publish_lsn, Ordering::AcqRel);
                handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    fn record_rollback(&self, txn_id: i64) -> Result<()> {
        // Skip WAL writes during recovery replay
        if self.should_skip_wal() {
            return Ok(());
        }

        // Record rollback in WAL
        if let Some(ref pm) = self.persistence() {
            if pm.is_enabled() {
                if let Err(e) = pm.record_rollback(txn_id) {
                    eprintln!("Warning: Failed to record rollback in WAL: {}", e);
                }
                // `write_abort_marker` no longer
                // clears active_txn_first_lsn (deferred so the
                // commit path's safe-visible publish reads a
                // consistent snapshot). Rollback has no
                // safe-visible publish, but the txn IS done — its
                // DML will never be applied (txn_id absent from
                // committed_txns), so we can clear immediately.
                //
                // After clearing, republish the recomputed
                // `oldest_active_txn_lsn` so readers' next
                // refresh advances `next_entry_floor` past this
                // rolled-back txn's first DML LSN. Without this
                // publish, if some earlier commit/DDL had
                // advertised THIS txn's first DML LSN as the
                // watermark (because it was active at that
                // publish), readers and fresh attaches would
                // keep that low watermark — pinning WAL — until
                // an unrelated future commit moves visibility.
                // The republish updates only `oldest_active_txn_lsn`
                // (visible_commit_lsn doesn't change for
                // rollback) under the seqlock so the pair stays
                // coherent.
                if let Some(wal) = pm.wal() {
                    wal.clear_active_txn(txn_id);
                    if let Some(handle) = self.shm.as_ref() {
                        let _publish_guard = self.shm_publish_lock.lock();
                        let fresh_oldest = wal.oldest_active_txn_lsn();
                        let current_oldest = handle
                            .header()
                            .oldest_active_txn_lsn
                            .load(Ordering::Acquire);
                        if fresh_oldest != current_oldest {
                            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                            handle
                                .header()
                                .oldest_active_txn_lsn
                                .store(fresh_oldest, Ordering::Release);
                            handle.header().publish_seq.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn get_tables_with_pending_changes(&self, txn_id: i64) -> Result<Vec<Box<dyn WriteTable>>> {
        let mut tables = Vec::new();

        // O(1) lookup for this transaction's tables (hot changes)
        let cache = self.txn_version_stores().read().unwrap();

        if let Some(txn_tables) = cache.get(txn_id) {
            for (table_name, txn_store) in txn_tables.iter() {
                let has_hot = {
                    let store = txn_store.read().unwrap();
                    store.has_local_changes()
                };

                if has_hot {
                    let stores = self.version_stores().read().unwrap();
                    if let Some(version_store) = stores.get(table_name.as_str()).cloned() {
                        drop(stores);

                        let table = MVCCTable::new_with_shared_store(
                            txn_id,
                            Arc::clone(&version_store),
                            Arc::clone(txn_store),
                        );

                        tables.push(Box::new(table) as Box<dyn WriteTable>);
                    }
                }
            }
        }

        Ok(tables)
    }

    fn has_pending_dml_changes(&self, txn_id: i64) -> bool {
        // Check hot-side DML changes
        let cache = self.txn_version_stores().read().unwrap();
        let txn_tables = match cache.get(txn_id) {
            Some(tables) if !tables.is_empty() => tables,
            _ => {
                // No txn_version_store entry means this txn never accessed any table
                // for DML, so no pending tombstones can exist either.
                return false;
            }
        };

        // Phase 1: Check hot mutations. Return immediately if any found.
        // Do NOT collect table names here — avoids SmartString clones
        // on the common early-return path.
        for (_, txn_store) in txn_tables.iter() {
            if txn_store.read().unwrap().has_local_changes() {
                return true;
            }
        }

        // Phase 2: No hot changes found. Now collect table names for cold check.
        let touched_tables: smallvec::SmallVec<[crate::common::SmartString; 4]> =
            txn_tables.iter().map(|(name, _)| name.clone()).collect();
        drop(cache);

        // Check cold-side pending tombstones only for tables this txn touched.
        // Cold DELETE/UPDATE on non-int-pk tables may create tombstones without
        // hot mutations, but they always go through get_table_for_transaction first.
        let mgrs = self.segment_managers.read().unwrap();
        for table_name in &touched_tables {
            if let Some(mgr) = mgrs.get(table_name.as_str()) {
                if mgr.has_pending_tombstones(txn_id) {
                    return true;
                }
            }
        }
        false
    }

    fn commit_all_tables(&self, txn_id: i64) -> (bool, Option<crate::core::Error>, Vec<String>) {
        // Get the commit_seq for this transaction. start_commit() was called before us,
        // so the commit_seq is in the registry. Used for versioned tombstones: snapshot
        // isolation transactions only see tombstones with commit_seq <= their begin_seq.
        let commit_seq = self.registry.get_committing_sequence(txn_id) as u64;

        // Collect table data under the read lock, then drop the lock before WAL I/O.
        // This prevents blocking concurrent get_table_for_transaction (which needs
        // a write lock on txn_version_stores) during potentially slow WAL writes.
        let tables_to_commit: Vec<(
            crate::common::SmartString,
            Arc<RwLock<TransactionVersionStore>>,
            Arc<VersionStore>,
        )>;
        let touched_table_names: SmallVec<[crate::common::SmartString; 4]>;
        {
            let cache = self.txn_version_stores().read().unwrap();
            if let Some(txn_tables) = cache.get(txn_id) {
                touched_table_names = txn_tables.iter().map(|(name, _)| name.clone()).collect();
                let stores = self.version_stores().read().unwrap();
                tables_to_commit = txn_tables
                    .iter()
                    .filter(|(_, txn_store)| {
                        let store = txn_store.read().unwrap();
                        store.has_local_changes()
                    })
                    .filter_map(|(table_name, txn_store)| {
                        stores
                            .get(table_name.as_str())
                            .cloned()
                            .map(|vs| (table_name.clone(), Arc::clone(txn_store), vs))
                    })
                    .collect();
            } else {
                touched_table_names = SmallVec::new();
                tables_to_commit = Vec::new();
            }
            // cache (read lock) and stores (read lock) dropped here
        }

        let mut commit_error: Option<crate::core::Error> = None;
        let mut any_committed = false;
        let mut pending_tombstone_tables = Vec::new();
        let mut tombstones_wal_recorded: rustc_hash::FxHashSet<String> =
            rustc_hash::FxHashSet::default();

        // Check if WAL recording is needed (persistence enabled and not in recovery)
        let should_record_wal = !self.should_skip_wal()
            && self
                .persistence()
                .as_ref()
                .is_some_and(|pm| pm.is_enabled());

        // Collect Arc clones for the managers this transaction touched, then
        // drop the read lock before WAL I/O. This keeps commit O(touched
        // tables) instead of O(all segment managers) for ordinary writes.
        let commit_mgrs: ahash::AHashMap<
            String,
            Arc<crate::storage::volume::manifest::SegmentManager>,
        > = {
            let mgrs = self.segment_managers.read().unwrap();
            touched_table_names
                .iter()
                .filter_map(|table_name| {
                    mgrs.get(table_name.as_str())
                        .map(|mgr| (table_name.to_string(), Arc::clone(mgr)))
                })
                .collect()
        };

        // WAL I/O and table commits happen here with NO lock on txn_version_stores
        for (table_name, txn_store, version_store) in &tables_to_commit {
            // Record cold tombstone DELETEs to WAL FIRST.
            // These must come before hot INSERTs so that on replay,
            // the tombstone marks the cold row as superseded BEFORE
            // the new hot version is encountered.
            if should_record_wal {
                if let Some(mgr) = commit_mgrs.get(table_name.as_str()) {
                    let pending = mgr.get_pending_tombstones(txn_id);
                    if !pending.is_empty() {
                        if let Some(ref pm) = *self.persistence() {
                            for &row_id in &pending {
                                let version = crate::storage::mvcc::version_store::RowVersion {
                                    txn_id,
                                    deleted_at_txn_id: txn_id,
                                    data: crate::core::Row::new(),
                                    create_time: 0,
                                };
                                if let Err(e) = pm.record_dml_operation(
                                    txn_id,
                                    table_name,
                                    row_id,
                                    crate::storage::mvcc::wal_manager::WALOperationType::Delete,
                                    &version,
                                ) {
                                    commit_error = Some(e);
                                    break;
                                }
                            }
                        }
                    }
                }
                if commit_error.is_some() {
                    break;
                }
            }

            // Create table and commit through it (updates indexes)
            let mut table = MVCCTable::new_with_shared_store(
                txn_id,
                Arc::clone(version_store),
                Arc::clone(txn_store),
            );

            // Record hot pending versions to WAL (after tombstone DELETEs)
            if should_record_wal {
                if let Err(e) = self.record_table_to_wal(txn_id, &table) {
                    commit_error = Some(e);
                    break;
                }
            }

            // Commit-time seal coordination:
            // 1. Always take fence for tables with segments (prevents concurrent
            //    seal from removing index entries during commit validation).
            // 2. If seal_generation changed since our INSERT, revalidate pending
            //    rows against cold (catches rows sealed before this commit).
            // Fall back to live lookup if the manager was created after our snapshot
            // (first seal on a previously hot-only table).
            let mgr_for_commit = commit_mgrs.get(table_name.as_str()).cloned().or_else(|| {
                self.segment_managers
                    .read()
                    .unwrap()
                    .get(table_name.as_str())
                    .cloned()
            });
            // Always take fence if manager exists — even before first seal
            // completes, the manager may exist without has_segments() being
            // true yet (created before register_volume publishes the flag).
            let _seal_guard = mgr_for_commit.as_ref().map(|mgr| mgr.acquire_seal_read());

            // Cold revalidation: only when a seal happened since our INSERT.
            if let Some(mgr) = mgr_for_commit.as_ref() {
                let recorded = mgr.get_txn_seal_generation(txn_id);
                let needs_recheck = match recorded {
                    Some(gen) => gen != mgr.seal_generation(),
                    // No recorded generation but segments exist: txn started
                    // before first seal or used hot-only path. Must recheck.
                    None => mgr.has_segments(),
                };
                if needs_recheck {
                    if let Err(e) =
                        self.validate_pending_against_cold(txn_id, txn_store, version_store, mgr)
                    {
                        commit_error = Some(e);
                        break;
                    }
                }
            }

            if let Err(e) = table.commit() {
                commit_error = Some(e);
                break;
            }

            // txn_seal_gens cleanup handled in the tail loop below

            // Mark tombstones as WAL-recorded ONLY after table.commit() succeeds.
            // Before this fix, the mark was set before commit, so a failed commit
            // would still apply tombstones in the tail loop — hiding cold rows
            // without the update version to replace them.
            if should_record_wal {
                tombstones_wal_recorded.insert(table_name.to_string());
            }

            any_committed = true;
        }

        // Always cleanup txn_version_stores to prevent memory leak,
        // even if commit failed partway through
        let mut cache = self.txn_version_stores().write().unwrap();
        cache.remove(txn_id);
        drop(cache);

        drop(tables_to_commit);

        // Commit or rollback pending tombstones on touched segment managers.
        // For tables with hot changes, tombstone WAL entries were already
        // recorded in the per-table loop above. For cold-only changes
        // (no hot), record tombstones here.
        //
        // On partial failure: tables whose hot changes were already committed
        // (tracked in tombstones_wal_recorded) must have their tombstones
        // committed too, not rolled back. Rolling back tombstones on an
        // already-committed table would leave stale cold rows visible.
        {
            let should_record_wal = commit_error.is_none()
                && !self.should_skip_wal()
                && self
                    .persistence()
                    .as_ref()
                    .is_some_and(|pm| pm.is_enabled());

            // Reuse the manager clones collected before the commit loop.
            //
            // Tombstones STAY PENDING here. `stamp_pending_tombstones`
            // (called from transaction.rs after `record_commit`
            // returns the marker LSN) does the actual commit with
            // `visible_at_lsn = marker_lsn`. Stamping with
            // `pm.current_lsn()` here would race another concurrent
            // commit's `publish_visible_commit_lsn`: a reader could
            // sample a cap between this txn's tombstone WAL entry
            // and this txn's marker, observe the new tombstone via
            // `retain_segments_visible_at_or_below(cap)`, but never
            // observe this txn's commit_seq via shm — hiding a row
            // from a transaction whose marker isn't visible at the
            // sampled cap.
            for (table_name, mgr) in commit_mgrs.iter() {
                let had_pending_tombstones = mgr.has_pending_tombstones(txn_id);
                // Decide whether to keep this table's pending
                // tombstones for `stamp_pending_tombstones` to commit
                // (with marker_lsn) OR roll them back now.
                //
                //   - No prior error AND cold-only tombstone WAL
                //     write succeeds → keep pending.
                //   - No prior error BUT cold-only tombstone WAL
                //     write fails → roll back.
                //   - Prior error from a LATER table AND this table
                //     was already in `tombstones_wal_recorded` (its
                //     hot changes committed in the per-table loop) →
                //     keep pending; rolling back would leave stale
                //     cold rows visible behind the new hot versions
                //     of the same row_ids.
                //   - Prior error AND this table never committed →
                //     roll back.
                let keep_pending = if commit_error.is_none() {
                    if should_record_wal && !tombstones_wal_recorded.contains(table_name.as_str()) {
                        let pending = mgr.get_pending_tombstones(txn_id);
                        if !pending.is_empty() {
                            if let Some(ref pm) = *self.persistence() {
                                for &row_id in &pending {
                                    let version = crate::storage::mvcc::version_store::RowVersion {
                                        txn_id,
                                        deleted_at_txn_id: txn_id,
                                        data: crate::core::Row::new(),
                                        create_time: 0,
                                    };
                                    if let Err(e) = pm.record_dml_operation(
                                        txn_id,
                                        table_name,
                                        row_id,
                                        crate::storage::mvcc::wal_manager::WALOperationType::Delete,
                                        &version,
                                    ) {
                                        commit_error = Some(e);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    commit_error.is_none()
                } else {
                    tombstones_wal_recorded.contains(table_name.as_str())
                };
                if !keep_pending {
                    mgr.rollback_pending_tombstones(txn_id);
                } else if had_pending_tombstones {
                    pending_tombstone_tables.push(table_name.clone());
                }
                mgr.clear_txn_seal_generation(txn_id);
            }
            // Suppress unused `commit_seq` warning when stamping
            // moved out of this loop. Kept in scope for clarity that
            // the seq is allocated here even though the actual
            // commit_pending_tombstones call lives in
            // `stamp_pending_tombstones`.
            let _ = commit_seq;
        }

        (any_committed, commit_error, pending_tombstone_tables)
    }

    fn mark_engine_failed(&self) {
        self.failed.store(true, Ordering::Release);
    }

    fn stamp_pending_tombstones(
        &self,
        txn_id: i64,
        commit_seq: u64,
        marker_lsn: u64,
        tables: &[String],
    ) {
        // `commit_seq` is the value `start_commit` allocated for
        // this txn; the caller passes it in directly because the
        // partial-commit path calls `complete_commit` BEFORE this
        // stamping (to publish partial state ASAP) — the registry
        // entry has already been removed, so a re-read here would
        // return 0 and downgrade every cold tombstone to "visible
        // to all snapshots".
        //
        // Commit only managers that `commit_all_tables` proved still
        // have pending tombstones for this txn. This keeps ordinary
        // INSERT/UPDATE commits from doing a second all-table scan.
        let mgrs = self.segment_managers.read().unwrap();
        for table_name in tables {
            if let Some(mgr) = mgrs.get(table_name.as_str()) {
                mgr.commit_pending_tombstones(txn_id, commit_seq, marker_lsn);
            }
        }
    }

    fn rollback_all_tables(&self, txn_id: i64) {
        // Collect touched table names BEFORE removing the cache entry,
        // so the common active-rollback path only iterates tables this
        // txn actually used (O(touched) instead of O(tables)).
        let mut cache = self.txn_version_stores().write().unwrap();
        let touched: smallvec::SmallVec<[crate::common::SmartString; 4]> = cache
            .get(txn_id)
            .map(|tables| tables.iter().map(|(name, _)| name.clone()).collect())
            .unwrap_or_default();
        cache.remove(txn_id);
        drop(cache);

        let mgrs = self.segment_managers.read().unwrap();
        if touched.is_empty() {
            // Cache was already drained — typically by `commit_all_tables`
            // which removes the txn entry before returning. The
            // partial-commit failure path in `MvccTransaction::commit`
            // calls us here to clear leftover pending tombstones for
            // tables that successfully committed (their tombstones were
            // kept pending by `commit_all_tables` so a subsequent
            // `stamp_pending_tombstones` could finalize them with the
            // marker LSN — but record_commit failed, so they need to
            // go away). We can't recover the touched list, so iterate
            // every manager. `rollback_pending_tombstones` is an O(1)
            // HashMap lookup keyed by `txn_id` and a no-op when nothing
            // is pending for the txn, so the cost is one lookup per
            // table.
            for mgr in mgrs.values() {
                mgr.rollback_pending_tombstones(txn_id);
                mgr.clear_txn_seal_generation(txn_id);
            }
        } else {
            for name in &touched {
                if let Some(mgr) = mgrs.get(name.as_str()) {
                    mgr.rollback_pending_tombstones(txn_id);
                    mgr.clear_txn_seal_generation(txn_id);
                }
            }
        }
    }

    fn acquire_seal_fence(&self) -> Option<SealFenceGuard> {
        Some(SealFenceGuard::new(Arc::clone(&self.seal_fence)))
    }

    fn acquire_transactional_ddl_fence(
        &self,
    ) -> Option<crate::storage::mvcc::transaction::TransactionalDdlFenceGuard> {
        Some(
            crate::storage::mvcc::transaction::TransactionalDdlFenceGuard::new(Arc::clone(
                &self.transactional_ddl_fence,
            )),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, IndexType, Row, SchemaBuilder, Value};

    #[test]
    fn test_engine_creation() {
        let engine = MVCCEngine::in_memory();
        assert!(!engine.is_open());
        assert_eq!(engine.get_path(), "memory://");
    }

    #[test]
    fn defer_for_live_readers_true_when_lease_present() {
        // SWMR v1 P1.3 (now V2.P1.5 unit-test): the writer's GC paths
        // consult `<db>/readers/` before unlinking compacted volumes.
        // The helper:
        //  - returns true while a fresh lease file exists (defer cleanup)
        //  - returns false once the lease is gone (safe to unlink)
        //  - reaps stale leases as a side effect
        use std::time::{Duration, SystemTime};

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("defer.db");
        let path_str = path.display().to_string();

        let engine = MVCCEngine::new(crate::storage::Config::with_path(path_str));
        engine.open_engine().unwrap();

        let readers_dir = path.join("readers");
        let lease = readers_dir.join("12345.lease");

        // No lease yet → no deferral.
        assert!(
            !engine.defer_for_live_readers(),
            "no readers/ dir → must not defer"
        );

        // Create a fresh lease file. Real RO opens use LeaseManager;
        // here we just need the file present with a recent mtime.
        std::fs::create_dir_all(&readers_dir).unwrap();
        std::fs::File::create(&lease).unwrap();
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&lease)
            .unwrap();
        f.set_modified(SystemTime::now()).unwrap();
        drop(f);
        assert!(
            engine.defer_for_live_readers(),
            "live lease must trigger deferral"
        );

        // Backdate the lease. Default max_age is `max(120s, 2 *
        // checkpoint_interval)`; 1h is unambiguously stale.
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&lease)
            .unwrap();
        f.set_modified(SystemTime::now() - Duration::from_secs(3600))
            .unwrap();
        drop(f);

        assert!(
            !engine.defer_for_live_readers(),
            "stale lease must NOT trigger deferral (and must be reaped)"
        );
        assert!(
            !lease.exists(),
            "stale lease must be reaped by defer_for_live_readers"
        );

        engine.close_engine().unwrap();
    }

    #[test]
    fn defer_for_live_readers_false_on_memory_engine() {
        // memory:// engines have no path; defer helper short-circuits to false.
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        assert!(
            !engine.defer_for_live_readers(),
            "memory engine must never defer (no readers/ dir possible)"
        );
        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_open_close() {
        let engine = MVCCEngine::in_memory();

        engine.open_engine().unwrap();
        assert!(engine.is_open());

        engine.close_engine().unwrap();
        assert!(!engine.is_open());
    }

    #[test]
    fn test_engine_create_table() {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("users")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();

        let created = engine.create_table(schema).unwrap();
        assert_eq!(created.table_name, "users");

        // Table should exist
        assert!(engine.table_exists("users").unwrap());
        assert!(engine.table_exists("USERS").unwrap()); // Case insensitive

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_drop_table() {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("temp")
            .column("id", DataType::Integer, false, true)
            .build();

        engine.create_table(schema).unwrap();
        assert!(engine.table_exists("temp").unwrap());

        engine.drop_table_internal("temp").unwrap();
        assert!(!engine.table_exists("temp").unwrap());

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_duplicate_table_error() {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("dup")
            .column("id", DataType::Integer, false, true)
            .build();

        engine.create_table(schema.clone()).unwrap();

        let result = engine.create_table(schema);
        assert!(result.is_err());

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_begin_transaction() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        let txn = engine.begin_transaction();
        assert!(txn.is_ok());

        let mut txn = txn.unwrap();
        assert!(txn.id() > 0);

        txn.rollback().unwrap();
        engine.close().unwrap();
    }

    #[test]
    fn test_engine_transaction_create_table() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        let mut txn = engine.begin_transaction().unwrap();

        // Create table through transaction
        let schema = SchemaBuilder::new("txn_table")
            .column("id", DataType::Integer, false, true)
            .column("value", DataType::Text, true, false)
            .build();

        let table = txn.create_table("txn_table", schema).unwrap();
        assert_eq!(table.name(), "txn_table");

        txn.commit().unwrap();
        engine.close().unwrap();
    }

    #[test]
    fn test_engine_transaction_insert_and_select() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        // Create table
        let schema = SchemaBuilder::new("data")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        engine.create_table(schema).unwrap();

        // Insert data in transaction
        let mut txn = engine.begin_transaction().unwrap();
        let mut table = txn.get_table("data").unwrap();

        table
            .insert(Row::from_values(vec![
                Value::Integer(1),
                Value::text("Alice"),
            ]))
            .unwrap();

        table
            .insert(Row::from_values(vec![
                Value::Integer(2),
                Value::text("Bob"),
            ]))
            .unwrap();

        // Scan to verify
        let mut scanner = table.scan(&[0, 1], None).unwrap();
        let mut count = 0;
        while scanner.next() {
            count += 1;
        }
        assert_eq!(count, 2);

        txn.commit().unwrap();
        engine.close().unwrap();
    }

    #[test]
    fn test_engine_isolation_level() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        // Default should be ReadCommitted
        assert_eq!(engine.get_isolation_level(), IsolationLevel::ReadCommitted);

        // Set to Snapshot
        engine
            .set_isolation_level(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(
            engine.get_isolation_level(),
            IsolationLevel::SnapshotIsolation
        );

        engine.close().unwrap();
    }

    #[test]
    fn test_engine_get_version_store() {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("versioned")
            .column("id", DataType::Integer, false, true)
            .build();
        engine.create_table(schema).unwrap();

        let store = engine.get_version_store("versioned");
        assert!(store.is_ok());

        let store = engine.get_version_store("nonexistent");
        assert!(store.is_err());

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_get_table_schema() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        let schema = SchemaBuilder::new("test_schema")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        engine.create_table(schema).unwrap();

        let retrieved = engine.get_table_schema("test_schema").unwrap();
        assert_eq!(retrieved.columns.len(), 2);
        assert_eq!(retrieved.columns[0].name, "id");

        // Non-existent table
        assert!(engine.get_table_schema("nonexistent").is_err());

        engine.close().unwrap();
    }

    #[test]
    fn test_engine_transaction_with_isolation_level() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        let txn = engine.begin_transaction_with_level(IsolationLevel::SnapshotIsolation);
        assert!(txn.is_ok());

        let mut txn = txn.unwrap();
        txn.rollback().unwrap();

        engine.close().unwrap();
    }

    #[test]
    fn test_engine_path() {
        let engine = MVCCEngine::in_memory();
        assert!(engine.path().is_none());

        let config = Config::with_path("/tmp/test.db");
        let engine = MVCCEngine::new(config);
        assert_eq!(engine.path(), Some("/tmp/test.db"));
    }

    #[test]
    fn test_engine_create_snapshot() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        // Should succeed (no-op for now)
        assert!(engine.create_snapshot().is_ok());

        engine.close().unwrap();
    }

    #[test]
    fn test_restart_merges_snapshot_and_newer_standalone_volume_without_duplicates() {
        fn count_rows(engine: &MVCCEngine, table_name: &str) -> i64 {
            let tx = engine.begin_transaction().unwrap();
            let table = tx.get_table(table_name).unwrap();
            let mut scanner = table.scan(&[0], None).unwrap();
            let mut count = 0i64;
            while scanner.next() {
                count += 1;
            }
            count
        }

        fn pseudo_random_payload(seed: i64, chunks: usize) -> String {
            let mut state = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
            let mut payload = String::with_capacity(chunks * 16);
            for _ in 0..chunks {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                payload.push_str(&format!("{:016x}", state));
            }
            payload
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("volume_restart_dedupe");
        let db_path_str = db_path.to_string_lossy().to_string();

        let config = Config::with_path(&db_path_str);
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("items")
            .column("id", DataType::Integer, false, true)
            .column("note", DataType::Text, false, false)
            .build();
        engine.create_table(schema).unwrap();

        {
            let mut tx = engine.begin_transaction().unwrap();
            let mut table = tx.get_table("items").unwrap();
            for i in 0..100_000i64 {
                table
                    .insert(Row::from_values(vec![
                        Value::Integer(i),
                        Value::text(pseudo_random_payload(i, 16)),
                    ]))
                    .unwrap();
            }
            tx.commit().unwrap();
        }

        // Checkpoint cycle: seals hot rows into frozen volumes and persists manifests.
        engine.checkpoint_cycle().unwrap();
        let vol_dir = db_path.join("volumes");
        assert!(
            !crate::storage::volume::io::list_volumes(&vol_dir, "items").is_empty(),
            "expected sealed standalone volume after checkpoint cycle"
        );

        // Verify manifest was persisted with checkpoint_lsn
        let manifest_path = vol_dir.join("items").join("manifest.bin");
        assert!(
            manifest_path.exists(),
            "expected manifest.bin after checkpoint cycle"
        );

        engine.close_engine().unwrap();

        let reopen_config = Config::with_path(&db_path_str);
        let reopened = MVCCEngine::new(reopen_config);
        reopened.open_engine().unwrap();
        assert_eq!(count_rows(&reopened, "items"), 100_000);
        reopened.close_engine().unwrap();
    }

    #[test]
    fn test_btree_index_on_volume_backed_table_covers_hot_rows_only() {
        // In the new architecture, hot indexes only cover hot rows.
        // Cold data is accessed via segment zone maps, binary search, and dictionary filters.
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("items")
            .column("id", DataType::Integer, false, true)
            .column("age", DataType::Integer, false, false)
            .build();
        engine.create_table(schema.clone()).unwrap();

        let mut builder = crate::storage::volume::writer::VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::Integer(30)]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::Integer(45)]),
        );
        engine.register_volume("items", Arc::new(builder.finish()));

        let tx = engine.begin_transaction().unwrap();
        let table = tx.get_table("items").unwrap();
        table.create_btree_index("age", false, None).unwrap();

        // Hot index should be empty (no hot rows)
        let index = table
            .get_index_on_column("age")
            .expect("btree index should exist");
        let row_ids = index.get_row_ids_equal(&[Value::Integer(30)]);
        assert!(
            row_ids.is_empty(),
            "hot index should not contain volume rows"
        );

        // But full table scan should still find volume rows
        let rows = table.collect_all_rows(None).unwrap();
        assert_eq!(rows.len(), 2, "scan should merge volume + hot rows");

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_unique_index_rejects_duplicate_cold_data() {
        // CREATE UNIQUE INDEX must validate cold volume data for duplicates.
        // If cold data already has duplicate values, the unique index must be rejected.
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("items")
            .column("id", DataType::Integer, false, true)
            .column("dup", DataType::Text, false, false)
            .build();
        engine.create_table(schema.clone()).unwrap();

        let mut builder = crate::storage::volume::writer::VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::text("dup")]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::text("dup")]),
        );
        engine.register_volume("items", Arc::new(builder.finish()));

        let tx = engine.begin_transaction().unwrap();
        let table = tx.get_table("items").unwrap();
        let result = table.create_index("idx_dup_unique", &["dup"], true);
        assert!(
            result.is_err(),
            "unique index creation should fail when cold data has duplicates"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unique constraint"));

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_unique_index_succeeds_on_distinct_cold_data() {
        // CREATE UNIQUE INDEX should succeed when cold data has no duplicates.
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("items")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();
        engine.create_table(schema.clone()).unwrap();

        let mut builder = crate::storage::volume::writer::VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::text("alice")]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::text("bob")]),
        );
        engine.register_volume("items", Arc::new(builder.finish()));

        let tx = engine.begin_transaction().unwrap();
        let table = tx.get_table("items").unwrap();
        let result = table.create_index("idx_name_unique", &["name"], true);
        assert!(
            result.is_ok(),
            "unique index creation should succeed when cold data is distinct"
        );

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_hnsw_index_on_volume_backed_table_populates_cold() {
        // HNSW indexes must include cold data because vector similarity search
        // cannot fall back to zone maps like B-tree/Hash indexes can.
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        let schema = SchemaBuilder::new("items")
            .column("id", DataType::Integer, false, true)
            .column("embedding", DataType::Vector, false, false)
            .set_last_vector_dimensions(2)
            .build();
        engine.create_table(schema.clone()).unwrap();

        let mut builder = crate::storage::volume::writer::VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::vector(vec![1.0, 0.0])]),
        );
        engine.register_volume("items", Arc::new(builder.finish()));

        let tx = engine.begin_transaction().unwrap();
        let table = tx.get_table("items").unwrap();
        table
            .create_index_with_type(
                "idx_embedding",
                &["embedding"],
                false,
                Some(IndexType::Hnsw),
            )
            .unwrap();

        // HNSW index should include cold data after creation
        let index = table
            .get_index_on_column("embedding")
            .expect("hnsw index should exist");
        let results = index
            .search_nearest(&Value::vector(vec![1.0, 0.0]), 1, 32)
            .unwrap_or_default();
        assert_eq!(
            results.len(),
            1,
            "HNSW index should include cold volume rows"
        );

        // Full table scan also returns volume data
        let rows = table.collect_all_rows(None).unwrap();
        assert_eq!(rows.len(), 1, "scan should find volume rows");

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_engine_list_table_indexes() {
        let mut engine = MVCCEngine::in_memory();
        engine.open().unwrap();

        let schema = SchemaBuilder::new("indexed")
            .column("id", DataType::Integer, false, true)
            .build();
        engine.create_table(schema).unwrap();

        // Should return empty map (no indexes yet)
        let indexes = engine.list_table_indexes("indexed").unwrap();
        assert!(indexes.is_empty());

        engine.close().unwrap();
    }

    #[test]
    fn test_cross_transaction_visibility() {
        // This test simulates the executor pattern: INSERT in one transaction, SELECT in another
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();

        // Create table
        let schema = SchemaBuilder::new("test_xact")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        engine.create_table(schema).unwrap();

        // Transaction 1: INSERT
        {
            let mut tx1 = engine.begin_transaction().unwrap();

            let mut table = tx1.get_table("test_xact").unwrap();
            table
                .insert(Row::from_values(vec![
                    Value::Integer(1),
                    Value::text("Alice"),
                ]))
                .unwrap();

            // Just commit the transaction - it commits all tables via commit_all_tables()
            tx1.commit().unwrap();
        }

        // Transaction 2: SELECT (different transaction)
        {
            let tx2 = engine.begin_transaction().unwrap();
            let table = tx2.get_table("test_xact").unwrap();
            let mut scanner = table.scan(&[0, 1], None).unwrap();

            let mut count = 0;
            while scanner.next() {
                count += 1;
            }
            // Should see the committed row from tx1
            assert_eq!(
                count, 1,
                "Transaction 2 should see 1 row committed by Transaction 1"
            );
        }

        engine.close_engine().unwrap();
    }

    // =========================================================================
    // Cleanup Mechanism Tests
    // =========================================================================

    use crate::storage::CleanupConfig;
    use std::time::Duration;

    #[test]
    fn test_cleanup_config_disabled() {
        let config = Config::in_memory().with_cleanup(CleanupConfig::disabled());
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();
        let engine = Arc::new(engine);

        // start_cleanup should be a no-op when disabled
        engine.start_cleanup();

        // Verify no cleanup handle was set
        let handle = engine.cleanup_handle.lock().unwrap();
        assert!(
            handle.is_none(),
            "Cleanup handle should be None when disabled"
        );
        drop(handle);

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_cleanup_config_custom_settings() {
        let config = Config::in_memory().with_cleanup(
            CleanupConfig::default()
                .with_interval_secs(30)
                .with_deleted_row_retention_secs(120)
                .with_transaction_retention_secs(600),
        );

        assert_eq!(config.cleanup.interval_secs, 30);
        assert_eq!(config.cleanup.deleted_row_retention_secs, 120);
        assert_eq!(config.cleanup.transaction_retention_secs, 600);
        assert!(config.cleanup.enabled);
    }

    #[test]
    fn test_cleanup_old_transactions_read_committed() {
        // In READ COMMITTED mode (default), transactions cannot be cleaned
        // because visibility depends on their presence in the registry
        let config = Config::in_memory().with_cleanup(CleanupConfig::disabled());
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();

        // Create and commit multiple transactions
        for _ in 0..10 {
            let mut txn = engine.begin_transaction().unwrap();
            txn.commit().unwrap();
        }

        // In READ COMMITTED mode, cleanup returns 0 (by design)
        let cleaned = engine.cleanup_old_transactions(Duration::from_secs(0));
        assert_eq!(
            cleaned, 0,
            "READ COMMITTED mode should not clean transactions"
        );

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_cleanup_old_transactions_snapshot_isolation() {
        use crate::core::IsolationLevel;

        // In SNAPSHOT ISOLATION mode, old transactions can be cleaned
        let config = Config::in_memory().with_cleanup(CleanupConfig::disabled());
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();

        // Set isolation level to Snapshot Isolation
        engine
            .registry
            .set_global_isolation_level(IsolationLevel::SnapshotIsolation);

        // Create and commit multiple transactions
        for _ in 0..10 {
            let mut txn = engine.begin_transaction().unwrap();
            txn.commit().unwrap();
        }

        // Wait a tiny bit for transactions to be "old"
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup with 0 retention should clean all committed transactions
        let cleaned = engine.cleanup_old_transactions(Duration::from_secs(0));
        assert!(
            cleaned > 0,
            "SNAPSHOT mode should clean up old committed transactions"
        );

        engine.close_engine().unwrap();
    }

    #[test]
    fn test_start_and_stop_cleanup() {
        let config = Config::in_memory().with_cleanup(
            CleanupConfig::default()
                .with_interval_secs(60) // Long interval so it doesn't run during test
                .with_deleted_row_retention_secs(0),
        );
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();
        let engine = Arc::new(engine);

        // Start cleanup
        engine.start_cleanup();

        // Verify cleanup handle was set
        {
            let handle = engine.cleanup_handle.lock().unwrap();
            assert!(
                handle.is_some(),
                "Cleanup handle should be set after start_cleanup"
            );
        }

        // Close engine should stop cleanup
        engine.close_engine().unwrap();

        // Verify cleanup handle was cleared
        {
            let handle = engine.cleanup_handle.lock().unwrap();
            assert!(
                handle.is_none(),
                "Cleanup handle should be cleared after close"
            );
        }
    }

    #[test]
    fn test_scheduled_cleanup_runs() {
        let config = Config::in_memory().with_cleanup(
            CleanupConfig::default()
                .with_interval_secs(1) // 1 second interval
                .with_deleted_row_retention_secs(0), // Immediate cleanup
        );
        let engine = MVCCEngine::new(config);
        engine.open_engine().unwrap();

        // Create table
        let schema = SchemaBuilder::new("cleanup_test")
            .column("id", DataType::Integer, false, true)
            .build();
        engine.create_table(schema).unwrap();

        // Insert rows
        {
            let mut tx = engine.begin_transaction().unwrap();
            let mut table = tx.get_table("cleanup_test").unwrap();
            for i in 1..=20 {
                table
                    .insert(Row::from_values(vec![Value::Integer(i)]))
                    .unwrap();
            }
            tx.commit().unwrap();
        }

        // Delete all rows using DELETE with no WHERE clause
        {
            let mut tx = engine.begin_transaction().unwrap();
            let mut table = tx.get_table("cleanup_test").unwrap();
            table.delete(None).unwrap();
            tx.commit().unwrap();
        }

        // Wrap in Arc for start_cleanup
        let engine = Arc::new(engine);

        // Start cleanup
        engine.start_cleanup();

        // Wait for scheduled cleanup to run
        std::thread::sleep(Duration::from_millis(1500));

        // Verify rows are cleaned
        {
            let tx = engine.begin_transaction().unwrap();
            let table = tx.get_table("cleanup_test").unwrap();
            let mut scanner = table.scan(&[0], None).unwrap();
            let mut count = 0;
            while scanner.next() {
                count += 1;
            }
            assert_eq!(
                count, 0,
                "All deleted rows should be cleaned by scheduled cleanup"
            );
        }

        engine.close_engine().unwrap();
    }
}
