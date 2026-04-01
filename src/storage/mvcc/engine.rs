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

use super::file_lock::FileLock;

use crate::core::{DataType, Error, ForeignKeyConstraint, IsolationLevel, Result, Schema, Value};
use crate::storage::config::Config;
use crate::storage::mvcc::wal_manager::WALOperationType;
#[cfg(test)]
use crate::storage::mvcc::VisibilityChecker;
use crate::storage::mvcc::{
    MVCCTable, MvccTransaction, PersistenceManager, PkIndex, RowVersion, SealFenceGuard,
    TransactionEngineOperations, TransactionRegistry, TransactionVersionStore, VersionStore,
    INVALID_TRANSACTION_ID,
};
use crate::storage::traits::{Engine, Index, Table, Transaction};

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
    /// Used for fast cache invalidation without HashMap lookup
    schema_epoch: AtomicU64,
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
    /// True while a background compaction thread is running.
    /// Background checkpoint skips compaction when set. Forced compaction
    /// (PRAGMA CHECKPOINT, close, restore) waits for it to finish first.
    compaction_running: Arc<AtomicBool>,
    /// Global epoch counter for volume eviction. Incremented each checkpoint
    /// cycle. Volumes whose last_access_epoch < eviction_epoch are idle.
    #[cfg(not(target_arch = "wasm32"))]
    eviction_epoch: AtomicU64,
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

impl MVCCEngine {
    /// Creates a new MVCC engine with the given configuration
    pub fn new(config: Config) -> Self {
        let path = config.path.clone().unwrap_or_default();

        // Initialize persistence manager if path is provided and persistence is enabled
        let persistence = if !path.is_empty() && config.persistence.enabled {
            match PersistenceManager::new(Some(Path::new(&path)), &config.persistence) {
                Ok(pm) => Some(pm),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize persistence: {}", e);
                    None
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
            txn_version_stores: Arc::new(RwLock::new(I64Map::new())),
            views: RwLock::new(FxHashMap::default()),
            persistence: Arc::new(persistence),
            loading_from_disk: Arc::new(AtomicBool::new(false)),
            file_lock: Mutex::new(None),
            schema_epoch: AtomicU64::new(0),
            cleanup_handle: Mutex::new(None),
            fk_reverse_cache: RwLock::new((u64::MAX, StringMap::default())),
            snapshot_timestamps: RwLock::new(FxHashMap::default()),
            segment_managers: Arc::new(RwLock::new(FxHashMap::default())),
            force_seal_all: AtomicBool::new(false),
            checkpoint_mutex: Mutex::new(()),
            seal_fence: Arc::new(parking_lot::RwLock::new(())),
            compaction_running: Arc::new(AtomicBool::new(false)),
            #[cfg(not(target_arch = "wasm32"))]
            eviction_epoch: AtomicU64::new(0),
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

        // Acquire file lock for disk-based databases to prevent concurrent access
        if self.path != "memory://" {
            let lock = FileLock::acquire(&self.path)?;
            let mut file_lock = self.file_lock.lock().unwrap();
            *file_lock = Some(lock);
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

                // Migration: if we loaded from legacy snapshots, seal all data
                // into volumes and remove the old snapshots/ directory.
                if has_legacy_snapshots {
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
                {
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
            }
        }

        // Note: Cleanup is started separately via start_cleanup() after Arc wrapping
        // because start_periodic_cleanup requires Arc<Self>

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

        let handle = thread::spawn(move || {
            let mut time_since_cleanup = std::time::Duration::ZERO;

            while !stop_flag_clone.load(Ordering::Acquire) {
                // Read config once per outer iteration (not per 100ms tick)
                let current_checkpoint_interval = {
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
                }

                if stop_flag_clone.load(Ordering::Acquire) {
                    break;
                }

                // Perform cleanup at the original cleanup interval
                time_since_cleanup += loop_interval;
                if time_since_cleanup >= interval {
                    time_since_cleanup = std::time::Duration::ZERO;
                    let _txn_count = engine.cleanup_old_transactions(txn_retention);
                    let _row_count = engine.cleanup_deleted_rows(deleted_row_retention);
                    let _prev_version_count = engine.cleanup_old_previous_versions();
                }

                // Auto-checkpoint using the cached interval
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

        // Use two-phase recovery for crash consistency
        // This ensures uncommitted transactions are NOT applied after a crash
        let result = pm.replay_two_phase(from_lsn, |entry| self.apply_wal_entry(entry));

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

                let tombstones = mgr.tombstone_set_arc();
                // Use newest-first ordering so overlapping row_ids resolve to newest version
                let volumes = mgr.get_volumes_newest_first();

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
                    mgr.add_tombstones(&[entry.row_id], 0);
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
                // Delete standalone volume files from disk
                if let Some(ref pm) = *self.persistence {
                    if pm.is_enabled() {
                        let vol_dir = pm.path().join("volumes");
                        let _ =
                            crate::storage::volume::io::delete_all_volumes(&vol_dir, &table_name);
                    }
                }
            }
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
        // Skipped when checkpoint_on_close is false (crash simulation in tests).
        let checkpoint_on_close = self.config.read().unwrap().persistence.checkpoint_on_close;
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

        Ok(())
    }

    /// Returns whether the engine is open
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
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

    /// Returns a copy of the configuration
    pub fn config(&self) -> Config {
        self.config.read().unwrap().clone()
    }

    /// Updates the engine configuration
    pub fn update_engine_config(&self, config: Config) -> Result<()> {
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

    /// Record a DDL operation to WAL
    fn record_ddl(&self, table_name: &str, op: WALOperationType, schema_data: &[u8]) -> Result<()> {
        if self.should_skip_wal() {
            return Ok(());
        }
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                pm.record_ddl_operation(table_name, op, schema_data)?;
            }
        }
        Ok(())
    }

    /// Serialize a schema to binary format for WAL
    pub fn serialize_schema(schema: &Schema) -> Vec<u8> {
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

    /// Returns all table names (lowercase) currently in the engine
    pub fn get_all_table_names(&self) -> Vec<String> {
        self.schemas.read().unwrap().keys().cloned().collect()
    }

    /// Returns all schemas currently in the engine (CompactArc ref-count bump only)
    pub fn get_all_schemas(&self) -> Vec<crate::common::CompactArc<Schema>> {
        self.schemas.read().unwrap().values().cloned().collect()
    }

    /// Get a table handle for an existing transaction by txn_id.
    /// This allows FK enforcement to participate in the caller's transaction,
    /// ensuring CASCADE effects are atomic and uncommitted rows are visible.
    pub fn get_table_for_txn(
        &self,
        txn_id: i64,
        table_name: &str,
    ) -> Result<Box<dyn crate::storage::traits::Table>> {
        EngineOperations::new(self).get_table_for_transaction(txn_id, table_name)
    }

    /// Find all FK constraints in other tables that reference the given parent table.
    /// Uses a cached reverse mapping that is rebuilt only when schema_epoch changes.
    /// Returns Arc-wrapped Vec for zero-copy sharing (ref-count bump only).
    /// Zero cost for databases without FK constraints.
    pub fn find_referencing_fks(
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
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

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

        // Atomically check-and-insert under write lock to prevent TOCTOU race
        let return_schema = schema.clone();
        {
            let mut schemas = self.schemas.write().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists(table_name.to_string()));
            }
            schemas.insert(table_name.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name, version_store);
        }

        // Record DDL to WAL only after successful insertion
        self.record_ddl(
            &return_schema.table_name,
            WALOperationType::CreateTable,
            &schema_data,
        )?;

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(return_schema)
    }

    /// Drops a table
    pub fn drop_table_internal(&self, name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name = name.to_lowercase();

        // Atomically remove schema AND strip FK references under single write lock.
        // This prevents a race where find_referencing_fks reads stale state between
        // schema removal and FK stripping.
        {
            let mut schemas = self.schemas.write().unwrap();
            if !schemas.contains_key(&table_name) {
                return Err(Error::TableNotFound(table_name.to_string()));
            }
            schemas.remove(&table_name);

            // Strip FK constraints from child tables that referenced the dropped table.
            // Done under the same schemas write lock for atomicity.
            let version_stores = self.version_stores.read().unwrap();
            strip_fk_references(&mut schemas, &version_stores, &table_name);
        }

        // Close and remove version store
        {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(&table_name) {
                store.close();
            }
        }

        // WAL FIRST: record the drop before deleting segment files.
        // If crash happens after WAL but before file deletion, WAL replay
        // will re-execute the drop. Orphan files are harmless.
        self.record_ddl(name, WALOperationType::DropTable, &[])?;

        // Clear in-memory segment state
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.get(&table_name) {
                mgr.clear();
            }
            mgrs.remove(&table_name);
        }
        // Delete volume files from disk
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let _ = crate::storage::volume::io::delete_all_volumes(&vol_dir, &table_name);
            }
        }

        // Increment schema epoch for cache invalidation
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// Gets a version store for a table
    pub fn get_version_store(&self, name: &str) -> Result<Arc<VersionStore>> {
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

        Ok(())
    }

    /// Refresh the engine's schema cache for a table from the version store
    /// This is used after DDL operations that modify the table's schema directly
    pub fn refresh_schema_cache(&self, table_name: &str) -> Result<()> {
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
    /// ensuring .dv files match their segments across restarts.
    fn register_volume_with_id(
        &self,
        table_name: &str,
        volume: Arc<crate::storage::volume::writer::FrozenVolume>,
        seg_id: u64,
    ) {
        self.register_volume_with_id_and_seal_seq(table_name, volume, seg_id, 0);
    }

    fn register_volume_with_id_and_seal_seq(
        &self,
        table_name: &str,
        volume: Arc<crate::storage::volume::writer::FrozenVolume>,
        seg_id: u64,
        seal_seq: u64,
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

        let entries = match std::fs::read_dir(&vol_dir) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut min_checkpoint_lsn: u64 = u64::MAX;
        let mut any_loaded = false;

        for entry in entries.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }

            let table_name = entry.file_name().to_string_lossy().to_lowercase();

            // Try to load manifest from this table directory
            match crate::storage::volume::manifest::SegmentManager::load_from_disk(
                &table_name,
                &vol_dir,
            ) {
                Ok(Some(mgr)) => {
                    let lsn = mgr.manifest().checkpoint_lsn;
                    if lsn > 0 && lsn < min_checkpoint_lsn {
                        min_checkpoint_lsn = lsn;
                    }
                    any_loaded = true;

                    // Store the segment manager so WAL replay can use
                    // is_row_id_in_volume_range() for tombstone creation.
                    let mut mgrs = self.segment_managers.write().unwrap();
                    mgrs.insert(table_name, Arc::new(mgr));
                }
                Ok(None) => {
                    // No manifest.bin in this directory, skip
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load manifest for {}: {}", table_name, e);
                }
            }
        }

        if !any_loaded {
            // No manifests found, remove checkpoint.meta if present
            // to ensure full WAL replay
            let checkpoint_path = pm.path().join("wal").join("checkpoint.meta");
            let _ = std::fs::remove_file(checkpoint_path);
            return 0;
        }

        if min_checkpoint_lsn == u64::MAX {
            0
        } else {
            min_checkpoint_lsn
        }
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
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                if mgr.has_segment(stable_id) {
                    continue;
                }
                if !mgr.manifest_has_segment(stable_id) {
                    let _ = std::fs::remove_file(&path);
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
    pub fn propagate_column_drop(&self, table_name: &str, col_name: &str) {
        let table_name_lower = table_name.to_lowercase();
        let schema = self.schemas.read().unwrap().get(&table_name_lower).cloned();
        let current_epoch = self.schema_epoch.load(Ordering::Acquire);
        if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
            mgr.record_column_drop(col_name, current_epoch);
            if let Some(ref s) = schema {
                mgr.invalidate_mappings(s);
            }
        }
    }

    /// Record a column rename and propagate alias to all cold volumes.
    /// Persists in the manifest so aliases survive restart.
    pub fn propagate_column_alias(&self, table_name: &str, new_name: &str, old_name: &str) {
        let table_name_lower = table_name.to_lowercase();
        let schema = self.schemas.read().unwrap().get(&table_name_lower).cloned();
        if let Some(mgr) = self.segment_managers.read().unwrap().get(&table_name_lower) {
            mgr.record_column_rename(old_name, new_name);
            if let Some(ref s) = schema {
                mgr.invalidate_mappings(s);
            }
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

        Ok(())
    }

    /// Modifies a column's type, nullable, and vector dimensions
    /// Used by WAL replay to restore ALTER TABLE MODIFY COLUMN with full dimension info
    pub fn modify_column_with_dimensions(
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

        Ok(())
    }

    /// Renames a table
    pub fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()> {
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

        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

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

        // Record to WAL for persistence
        let data = view_def.serialize();
        self.record_ddl(&name_lower, WALOperationType::CreateView, &data)?;

        Ok(())
    }

    /// Drop a view
    pub fn drop_view(&self, name: &str, if_exists: bool) -> Result<()> {
        use crate::storage::mvcc::wal_manager::WALOperationType;

        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let name_lower = name.to_lowercase();
        let mut views = self.views.write().unwrap();

        if views.remove(&name_lower).is_none() {
            if if_exists {
                return Ok(());
            }
            return Err(Error::ViewNotFound(name.to_string()));
        }

        // Release the lock before recording to WAL
        drop(views);

        // Record to WAL for persistence (just the view name)
        self.record_ddl(&name_lower, WALOperationType::DropView, name.as_bytes())?;

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
                .map(|(name, mgr)| (name.clone(), mgr.tombstone_set_arc()))
                .collect()
        };

        let snapshot_commit_seq = self.registry.current_commit_sequence();

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
            if let Err(e) = ddl_result {
                eprintln!("Warning: Failed to write DDL metadata: {}", e);
                let _ = std::fs::remove_file(&ddl_tmp);
            }

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
                        // Escape JSON-special characters in table names
                        let escaped = t.replace('\\', "\\\\").replace('"', "\\\"");
                        format!("\"{}\"", escaped)
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let manifest_result = (|| -> std::io::Result<()> {
                let f = std::fs::File::create(&manifest_tmp)?;
                std::io::Write::write_all(&mut &f, manifest_json.as_bytes())?;
                f.sync_all()?;
                std::fs::rename(&manifest_tmp, &manifest_path)?;
                Ok(())
            })();
            if let Err(e) = manifest_result {
                eprintln!("Warning: Failed to write snapshot manifest: {}", e);
                let _ = std::fs::remove_file(&manifest_tmp);
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
    fn restore_ddl_from_bin(&self, data: &[u8]) {
        // Minimum: magic(4) + version(1) + crc(4)
        if data.len() < 9 || &data[0..4] != b"SDDL" {
            return;
        }

        // Verify CRC32: last 4 bytes are checksum of everything before
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            eprintln!("Warning: ddl.bin CRC mismatch, skipping DDL restore");
            return;
        }

        let mut pos = 5; // skip magic + version

        // Read indexes
        if pos + 4 > data.len() {
            return;
        }
        let index_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let stores = self.version_stores.read().unwrap();
        for _ in 0..index_count {
            if pos + 4 > data.len() {
                break;
            }
            let entry_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + entry_len > data.len() {
                break;
            }
            let entry_data = &data[pos..pos + entry_len];
            pos += entry_len;

            if let Ok(meta) = super::persistence::IndexMetadata::deserialize(entry_data) {
                let table_lower = meta.table_name.to_lowercase();
                if let Some(store) = stores.get(&table_lower) {
                    if let Err(e) = store.create_index_from_metadata_with_graph(&meta, false, None)
                    {
                        eprintln!(
                            "Warning: Failed to recreate index '{}' on '{}': {}",
                            meta.name, meta.table_name, e
                        );
                    }
                }
            }
        }
        drop(stores);

        // Read views
        if pos + 4 > data.len() {
            return;
        }
        let view_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        for _ in 0..view_count {
            if pos + 4 > data.len() {
                break;
            }
            let entry_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + entry_len > data.len() {
                break;
            }
            let entry_data = &data[pos..pos + entry_len];
            pos += entry_len;

            // ViewDefinition format: name_len(u16) + name + query_len(u32) + query
            let mut vpos = 0;
            if vpos + 2 > entry_data.len() {
                continue;
            }
            let name_len =
                u16::from_le_bytes(entry_data[vpos..vpos + 2].try_into().unwrap()) as usize;
            vpos += 2;
            if vpos + name_len > entry_data.len() {
                continue;
            }
            let view_name = String::from_utf8_lossy(&entry_data[vpos..vpos + name_len]).to_string();
            vpos += name_len;
            if vpos + 4 > entry_data.len() {
                continue;
            }
            let query_len =
                u32::from_le_bytes(entry_data[vpos..vpos + 4].try_into().unwrap()) as usize;
            vpos += 4;
            if vpos + query_len > entry_data.len() {
                continue;
            }
            let query = String::from_utf8_lossy(&entry_data[vpos..vpos + query_len]).to_string();

            let view_def = Arc::new(ViewDefinition::new(&view_name, query));
            self.views
                .write()
                .unwrap()
                .insert(view_name.to_lowercase(), view_def);
        }
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

        // When no timestamp is given, find the latest manifest-*.json to get the
        // coherent table list from that snapshot batch. This prevents resurrecting
        // tables that were dropped between snapshots.
        let effective_tables: Option<Vec<String>> = if timestamp.is_none() {
            let mut manifests: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshot_dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json")
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("manifest-"))
                    {
                        Some(p)
                    } else {
                        None
                    }
                })
                .collect();
            manifests.sort();
            if let Some(latest) = manifests.last() {
                if let Ok(data) = std::fs::read_to_string(latest) {
                    // manifest-*.json is {"timestamp":"...","tables":["t1","t2"]}
                    let parsed: serde_json::Value = serde_json::from_str(&data).unwrap_or_default();
                    let tables: Vec<String> = parsed
                        .get("tables")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    if !tables.is_empty() {
                        Some(tables)
                    } else {
                        None
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

        for entry in table_dirs.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }
            let table_name = entry.file_name().to_string_lossy().to_string();

            // Skip tables not in the manifest (prevents resurrecting dropped tables)
            if let Some(ref tables) = effective_tables {
                if !tables.iter().any(|t| t.eq_ignore_ascii_case(&table_name)) {
                    continue;
                }
            }

            let paths = Self::find_snapshots_newest_first(&entry.path());

            if let Some(ts) = timestamp {
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
            self.registry.start_accepting_transactions();
            return Err(Error::internal("No matching snapshots found"));
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
        // For timestamped restore: use exact match. For latest restore: extract the
        // oldest timestamp from the selected snapshot files (conservative — ensures
        // DDL is never newer than any table data being restored).
        let ddl_data = {
            let effective_ts = if timestamp.is_some() {
                timestamp.map(|s| s.to_string())
            } else {
                // Extract oldest timestamp from selected snapshot filenames.
                // Format: "snapshot-YYYYMMDD-HHMMSS.fff.bin"
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
                std::fs::read(&ddl_path).ok()
            } else if timestamp.is_some() {
                // Timestamped restore with missing DDL: fail rather than silently
                // using current indexes/views which may not match the snapshot data.
                self.registry.start_accepting_transactions();
                return Err(Error::internal(format!(
                    "DDL metadata file not found for timestamp '{}'. Cannot restore indexes/views accurately.",
                    timestamp.unwrap()
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

        // Delete all volume directories on disk
        let vol_dir = pm.path().join("volumes");
        if vol_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&vol_dir) {
                for entry in entries.flatten() {
                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        let _ = std::fs::remove_dir_all(entry.path());
                    }
                }
            }
        }

        // Reset WAL: old entries contain post-snapshot DML that would
        // overwrite restored data if replayed on next open.
        // Truncate with LSN=1 to create a fresh WAL starting from the beginning.
        // The post-restore checkpoint will re-record DDL with correct LSNs.
        if let Err(e) = pm.truncate_wal(1) {
            eprintln!("Warning: WAL reset during restore: {}", e);
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
                    // Persist what we have so a crash doesn't lose everything.
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
                    self.registry.start_accepting_transactions();
                    return Err(Error::internal(format!(
                        "Failed to restore table {}: {}",
                        table_name, e
                    )));
                }
            }
        }

        // === Post-restore ===

        // Recreate indexes and views from ddl.bin or fallback saved state.
        if let Some(ref data) = ddl_data {
            self.restore_ddl_from_bin(data);
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

        // Re-record DDL to WAL so table schemas survive WAL truncation
        if let Err(e) = self.rerecord_ddl_to_wal() {
            eprintln!(
                "Warning: Failed to re-record DDL to WAL after restore: {}",
                e
            );
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
        self.checkpoint_cycle_inner(true).map_err(|e| {
            // Resume transactions so the database is usable even if non-durable
            self.registry.start_accepting_transactions();
            Error::Internal {
                message: format!("Restore data not persisted (crash may lose it): {}", e),
            }
        })?;
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
        self.compact_after_checkpoint_forced();
        Ok(())
    }

    /// Inner checkpoint implementation. When `force` is true, seals ALL hot rows
    /// regardless of threshold (used by PRAGMA CHECKPOINT and close_engine).
    /// When false, respects the normal seal thresholds (used by background thread).
    fn checkpoint_cycle_inner(&self, force: bool) -> Result<()> {
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

        // Step 5: Truncate WAL BEFORE compaction.
        // WAL truncation only depends on seal + manifest persist, not compaction.
        // Running compaction first (30+ seconds) delays truncation and extends
        // the checkpoint window unnecessarily.
        // checkpoint_lsn > 0 already guarantees all_hot_empty was true inside
        // the fence. Don't re-check the outer all_hot_empty (stale, checked
        // before fence when continuous inserts make it always false).
        if checkpoint_lsn > 0 && all_manifests_persisted {
            if let Some(ref pm) = *self.persistence {
                if let Err(e) = pm.truncate_wal(checkpoint_lsn) {
                    eprintln!("Warning: Failed to truncate WAL: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Run compaction synchronously under the compaction_running flag.
    /// The flag prevents concurrent compaction from background and forced
    /// callers. Clears the flag on exit (including panics via drop guard).
    fn run_compaction_guarded(&self) {
        let _guard = AtomicBoolGuard(&self.compaction_running);
        if let Err(e) = self.compact_volumes() {
            eprintln!("Warning: compact_volumes failed: {}", e);
        }
    }

    /// Evict idle volume data to save memory. Volumes not accessed since the
    /// last epoch transition: hot → warm (drop decompressed) → cold (drop compressed).
    #[cfg(not(target_arch = "wasm32"))]
    fn evict_idle_volumes(&self) {
        let epoch = self.eviction_epoch.fetch_add(1, Ordering::Relaxed) + 1;
        // Publish to global so scanners stamp volumes with the correct epoch.
        // Use fetch_max so multiple engines only move the global forward.
        crate::storage::volume::writer::GLOBAL_EVICTION_EPOCH.fetch_max(epoch, Ordering::Relaxed);
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
            let (old_ids, volumes, tombstones) = {
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
                vols.sort_by(|a, b| b.0.cmp(&a.0));
                let ts = mgr.tombstone_set_arc();
                (old_ids, Arc::new(vols), ts)
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
            let applied_tombstones: Arc<FxHashMap<i64, u64>> =
                if let Some(limit) = compact_seal_seq_limit {
                    let filtered: FxHashMap<i64, u64> = tombstones
                        .iter()
                        .filter(|(_, &seq)| seq < limit)
                        .map(|(&rid, &seq)| (rid, seq))
                        .collect();
                    Arc::new(filtered)
                } else {
                    Arc::clone(&tombstones)
                };

            for (vol_idx, (_seg_id, vol)) in volumes.iter().enumerate() {
                for i in 0..vol.meta.row_count {
                    let row_id = vol.meta.row_ids[i];
                    // Apply tombstone only if it was committed before the earliest
                    // snapshot (safe to physically remove). Tombstones created after
                    // are preserved — the row stays in the merged volume so snapshots
                    // can still see it via versioned tombstone filtering.
                    let is_tombstoned = if let Some(limit) = compact_seal_seq_limit {
                        tombstones
                            .get(&row_id)
                            .is_some_and(|&commit_seq| commit_seq < limit)
                    } else {
                        tombstones.contains_key(&row_id)
                    };
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

            // Now safe to delete old volume files + stale .dv files.
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
                        for (volume, _path, volume_id) in &sealed_volumes {
                            self.register_volume_with_id_and_seal_seq(
                                &table_name,
                                Arc::clone(volume),
                                *volume_id,
                                current_seal_seq,
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
                            mgr.add_tombstones(&all_skipped_inner, seal_seq);
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

impl Engine for MVCCEngine {
    fn open(&mut self) -> Result<()> {
        MVCCEngine::open_engine(self)
    }

    fn close(&mut self) -> Result<()> {
        MVCCEngine::close_engine(self)
    }

    fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.begin_transaction_with_level(self.get_isolation_level())
    }

    fn begin_transaction_with_level(&self, level: IsolationLevel) -> Result<Box<dyn Transaction>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
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
        MVCCEngine::checkpoint_cycle(self)
    }

    fn force_checkpoint_cycle(&self) -> Result<()> {
        MVCCEngine::checkpoint_cycle_inner(self, true)?;
        self.compact_after_checkpoint_forced();
        Ok(())
    }

    fn create_snapshot(&self) -> Result<()> {
        MVCCEngine::create_backup_snapshot(self)
    }

    fn restore_snapshot(&self, timestamp: Option<&str>) -> Result<String> {
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

        // If this column was previously dropped, clear it from dropped_columns.
        // Recompute cold mappings with the post-add schema.
        // dropped_columns stays permanent — old volumes have stale data under
        // the dropped name, new volumes get the re-added column at a new position.
        {
            let schema = self.schemas.read().unwrap().get(table_name).cloned();
            let mgrs = self.segment_managers.read().unwrap();
            if let Some(mgr) = mgrs.get(table_name) {
                if let Some(ref s) = schema {
                    mgr.invalidate_mappings(s);
                }
            }
        }

        self.record_ddl(table_name, WALOperationType::AlterTable, &data)
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

        self.record_ddl(table_name, WALOperationType::AlterTable, &data)
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

        // Delete volume files from disk (standalone volumes + legacy tombstones)
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let _ = crate::storage::volume::io::delete_all_volumes(&vol_dir, &table_lower);
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
    /// Cleanup old transactions that have been idle for too long
    pub fn cleanup_old_transactions(&self, max_age: std::time::Duration) -> i32 {
        if !self.is_open() {
            return 0;
        }
        self.registry.cleanup_old_transactions(max_age)
    }

    /// Cleanup deleted rows older than retention period from all tables
    pub fn cleanup_deleted_rows(&self, max_age: std::time::Duration) -> i32 {
        if !self.is_open() {
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
        if !self.is_open() {
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
    /// per-transaction. Kept for API compatibility.
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_periodic_cleanup(
        self: &Arc<Self>,
        interval: std::time::Duration,
        max_age: std::time::Duration,
    ) -> CleanupHandle {
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let stop_flag = Arc::new(AtomicBool::new(false));
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
}

// EngineOperations is Send + Sync because all fields are Arc-wrapped thread-safe types

impl EngineOperations {
    fn new(engine: &MVCCEngine) -> Self {
        Self {
            schemas: Arc::clone(&engine.schemas),
            version_stores: Arc::clone(&engine.version_stores),
            registry: Arc::clone(&engine.registry),
            txn_version_stores: Arc::clone(&engine.txn_version_stores),
            persistence: Arc::clone(&engine.persistence),
            loading_from_disk: Arc::clone(&engine.loading_from_disk),
            segment_managers: Arc::clone(&engine.segment_managers),
            seal_fence: Arc::clone(&engine.seal_fence),
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

impl TransactionEngineOperations for EngineOperations {
    fn get_table_for_transaction(&self, txn_id: i64, table_name: &str) -> Result<Box<dyn Table>> {
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
                return Ok(Box::new(
                    crate::storage::volume::table::SegmentedTable::new(Box::new(table), mgr),
                ));
            }
        }

        Ok(Box::new(table))
    }

    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn Table>> {
        let table_name = name.to_lowercase();

        // Create version store for this table (before acquiring locks)
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            registry_as_visibility_checker(&self.registry),
        ));

        // Register PkIndex if table has a primary key
        register_pk_index(&schema, &version_store);

        // Atomically check-and-insert under write lock to prevent TOCTOU race
        {
            let mut schemas = self.schemas().write().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists(table_name.to_string()));
            }
            schemas.insert(table_name.clone(), CompactArc::new(schema));
        }
        {
            let mut stores = self.version_stores().write().unwrap();
            stores.insert(table_name, Arc::clone(&version_store));
        }

        // Create transaction version store with txn_id 0 (will be set by caller)
        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), 0);

        // Create MVCC table
        let table = MVCCTable::new(0, version_store, txn_versions);

        Ok(Box::new(table))
    }

    fn drop_table(&self, name: &str) -> Result<()> {
        let table_name_lower = name.to_lowercase();

        // Remove schema and clean up FK references in child tables
        {
            let mut schemas = self.schemas().write().unwrap();
            if schemas.remove(&table_name_lower).is_none() {
                return Err(Error::TableNotFound(table_name_lower.to_string()));
            }
            let version_stores = self.version_stores().read().unwrap();
            strip_fk_references(&mut schemas, &version_stores, &table_name_lower);
        }
        // Close and remove version store
        {
            let mut stores = self.version_stores().write().unwrap();
            if let Some(store) = stores.remove(&table_name_lower) {
                store.close();
            }
        }

        // Record DDL to WAL so the drop survives crash recovery
        if !self.should_skip_wal() {
            if let Some(ref pm) = *self.persistence() {
                if pm.is_enabled() {
                    let _ = pm.record_ddl_operation(name, WALOperationType::DropTable, &[]);
                }
            }
        }

        // Clear in-memory segment state (prevents phantom rows on re-create)
        {
            let mut mgrs = self.segment_managers.write().unwrap();
            if let Some(mgr) = mgrs.get(&table_name_lower) {
                mgr.clear();
            }
            mgrs.remove(&table_name_lower);
        }

        // Delete volume files from disk
        if let Some(ref pm) = *self.persistence() {
            if pm.is_enabled() {
                let vol_dir = pm.path().join("volumes");
                let _ = crate::storage::volume::io::delete_all_volumes(&vol_dir, &table_name_lower);
            }
        }

        Ok(())
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        let schemas = self.schemas().read().unwrap();
        Ok(schemas.keys().cloned().collect())
    }

    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()> {
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

        Ok(())
    }

    fn commit_table(&self, txn_id: i64, table: &dyn Table) -> Result<()> {
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

    fn rollback_table(&self, _txn_id: i64, table: &dyn Table) {
        // The Table trait now has a rollback method.
        // This callback is for any engine-level rollback actions.
        let _ = table;
    }

    fn record_commit(&self, txn_id: i64) -> Result<()> {
        // Skip WAL writes during recovery replay
        if self.should_skip_wal() {
            return Ok(());
        }

        // Record commit in WAL — propagate errors since missing commit records
        // means crash recovery won't replay this transaction's changes
        if let Some(ref pm) = self.persistence() {
            if pm.is_enabled() {
                pm.record_commit(txn_id)?;
            }
        }
        Ok(())
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
            }
        }
        Ok(())
    }

    fn get_tables_with_pending_changes(&self, txn_id: i64) -> Result<Vec<Box<dyn Table>>> {
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

                        tables.push(Box::new(table) as Box<dyn Table>);
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

    fn commit_all_tables(&self, txn_id: i64) -> (bool, Option<crate::core::Error>) {
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
        {
            let cache = self.txn_version_stores().read().unwrap();
            tables_to_commit = if let Some(txn_tables) = cache.get(txn_id) {
                let stores = self.version_stores().read().unwrap();
                txn_tables
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
                    .collect()
            } else {
                Vec::new()
            };
            // cache (read lock) and stores (read lock) dropped here
        }

        let mut commit_error: Option<crate::core::Error> = None;
        let mut any_committed = false;
        let mut tombstones_wal_recorded: rustc_hash::FxHashSet<String> =
            rustc_hash::FxHashSet::default();

        // Check if WAL recording is needed (persistence enabled and not in recovery)
        let should_record_wal = !self.should_skip_wal()
            && self
                .persistence()
                .as_ref()
                .is_some_and(|pm| pm.is_enabled());

        // Collect Arc clones of segment managers, then drop the read lock
        // before WAL I/O to avoid blocking seal/compaction writes.
        let commit_mgrs: ahash::AHashMap<
            String,
            Arc<crate::storage::volume::manifest::SegmentManager>,
        > = {
            let mgrs = self.segment_managers.read().unwrap();
            mgrs.iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
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

        // Commit or rollback pending tombstones on all segment managers.
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

            // Reuse the commit_mgrs read lock acquired before the commit loop
            for (table_name, mgr) in commit_mgrs.iter() {
                if commit_error.is_none() {
                    // Record cold-only tombstones to WAL (tables not already handled above)
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
                    if commit_error.is_some() {
                        mgr.rollback_pending_tombstones(txn_id);
                    } else {
                        mgr.commit_pending_tombstones(txn_id, commit_seq);
                    }
                } else if tombstones_wal_recorded.contains(table_name.as_str()) {
                    mgr.commit_pending_tombstones(txn_id, commit_seq);
                } else {
                    mgr.rollback_pending_tombstones(txn_id);
                }
                mgr.clear_txn_seal_generation(txn_id);
            }
        }

        (any_committed, commit_error)
    }

    fn rollback_all_tables(&self, txn_id: i64) {
        // Collect touched table names BEFORE removing the cache entry,
        // so we only rollback tombstones on tables this txn actually used
        // instead of iterating every segment manager (O(tables) → O(touched)).
        let mut cache = self.txn_version_stores().write().unwrap();
        let touched: smallvec::SmallVec<[crate::common::SmartString; 4]> = cache
            .get(txn_id)
            .map(|tables| tables.iter().map(|(name, _)| name.clone()).collect())
            .unwrap_or_default();
        cache.remove(txn_id);
        drop(cache);

        // Rollback pending tombstones and clean up seal generation records
        // only on tables this transaction touched.
        if !touched.is_empty() {
            let mgrs = self.segment_managers.read().unwrap();
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
