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

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use std::path::Path;

use super::file_lock::FileLock;

use crate::core::{DataType, Error, IsolationLevel, Result, Schema};
use crate::storage::config::Config;
use crate::storage::mvcc::wal_manager::WALOperationType;
use crate::storage::mvcc::{
    MVCCTable, MvccTransaction, PersistenceManager, RowVersion, TransactionEngineOperations,
    TransactionRegistry, TransactionVersionStore, VersionStore, VisibilityChecker,
    INVALID_TRANSACTION_ID,
};
use crate::storage::traits::{Engine, Index, Table, Transaction};

/// Type alias for the transaction version store map
type TxnVersionStoreMap = FxHashMap<(i64, String), Arc<RwLock<TransactionVersionStore>>>;

// ============================================================================
// Binary Snapshot Metadata Functions
// ============================================================================
// Format: MAGIC(4) | VERSION(4) | LSN(8) | TIMESTAMP(8) | CRC32(4) = 28 bytes
// Magic bytes: 0x534E4150 ("SNAP" in ASCII)

/// Magic bytes for snapshot metadata ("SNAP" in ASCII)
const SNAPSHOT_META_MAGIC: u32 = 0x50414E53; // "SNAP" in little-endian

/// Current version of the snapshot metadata format
const SNAPSHOT_META_VERSION: u32 = 1;

/// Write binary snapshot metadata with magic number and checksum
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
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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

    // Sync directory to ensure rename is durable
    if let Some(parent) = path.parent() {
        if let Ok(dir_file) = std::fs::File::open(parent) {
            let _ = dir_file.sync_all();
        }
    }

    Ok(())
}

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

/// MVCC Storage Engine
///
/// Provides multi-version concurrency control with snapshot isolation.
pub struct MVCCEngine {
    /// Database path (empty for in-memory)
    path: String,
    /// Configuration
    config: RwLock<Config>,
    /// Table schemas (Arc-wrapped for safe sharing with transactions)
    schemas: Arc<RwLock<FxHashMap<String, Schema>>>,
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
            txn_version_stores: Arc::new(RwLock::new(FxHashMap::default())),
            views: RwLock::new(FxHashMap::default()),
            persistence: Arc::new(persistence),
            loading_from_disk: Arc::new(AtomicBool::new(false)),
            file_lock: Mutex::new(None),
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

                // Try to load from snapshots first (for faster recovery)
                let snapshot_lsn = self.load_snapshots()?;

                // Replay WAL entries after the snapshot LSN
                self.replay_wal(snapshot_lsn)?;

                // Clear the loading flag
                self.loading_from_disk.store(false, Ordering::Release);
            }
        }

        Ok(())
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

        // Track max source_lsn from snapshot headers for validation/fallback
        let mut max_header_lsn: u64 = 0;

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

            // Find the most recent snapshot file in this directory
            if let Some(snapshot_path) = self.find_latest_snapshot(&entry.path()) {
                match self.load_table_snapshot(&table_name, &snapshot_path) {
                    Ok(source_lsn) => {
                        // Track max source_lsn from snapshot headers (v3+ format)
                        if source_lsn > max_header_lsn {
                            max_header_lsn = source_lsn;
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load snapshot for {}: {}", table_name, e);
                    }
                }
            }
        }

        // Use the larger of metadata LSN and max header LSN
        // - If metadata file is missing, use header LSN (v3+ fallback)
        // - If metadata exists and matches header, use metadata (normal case)
        // - If metadata is smaller than header, prefer header (corruption recovery)
        let snapshot_lsn = std::cmp::max(metadata_lsn, max_header_lsn);

        Ok(snapshot_lsn)
    }

    /// Find the most recent snapshot file in a directory
    fn find_latest_snapshot(&self, dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let mut snapshots: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot-") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by name (timestamp in filename)
        snapshots.sort();

        // Return the latest one
        snapshots.pop()
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
            Arc::clone(&self.registry) as Arc<dyn VisibilityChecker>,
        ));

        // Load all rows from the snapshot
        reader.for_each(|_row_id, mut version| {
            // Snapshot versions have txn_id = -1, we need to use the recovery txn_id
            version.txn_id = super::RECOVERY_TRANSACTION_ID;

            // Apply to version store
            version_store.apply_recovered_version(version);
            true
        })?;

        // Store the schema and version store
        {
            let mut schemas = self.schemas.write().unwrap();
            schemas.insert(table_name_lower.clone(), schema);
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name_lower, version_store);
        }

        Ok(source_lsn)
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
                        Arc::clone(&self.registry) as Arc<dyn VisibilityChecker>,
                    ));

                    let table_name = schema.table_name_lower.clone();

                    {
                        let mut schemas = self.schemas.write().unwrap();
                        schemas.insert(table_name.clone(), schema);
                    }
                    {
                        let mut stores = self.version_stores.write().unwrap();
                        stores.insert(table_name, version_store);
                    }
                }
            }
            WALOperationType::DropTable => {
                let table_name = entry.table_name.to_lowercase();

                // Remove schema and version store
                {
                    let mut schemas = self.schemas.write().unwrap();
                    schemas.remove(&table_name);
                }
                {
                    let mut stores = self.version_stores.write().unwrap();
                    if let Some(store) = stores.remove(&table_name) {
                        store.close();
                    }
                }
            }
            WALOperationType::CreateIndex => {
                // Deserialize index metadata
                // Use skip_population=true for deferred single-pass population
                if let Ok(index_meta) = IndexMetadata::deserialize(&entry.data) {
                    let table_name = entry.table_name.to_lowercase();
                    if let Ok(store) = self.get_version_store(&table_name) {
                        let _ = store.create_index_from_metadata(&index_meta, true);
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
                // Deserialize row version and apply to version store
                if let Ok(row_version) = deserialize_row_version(&entry.data) {
                    let table_name = entry.table_name.to_lowercase();
                    if let Ok(store) = self.get_version_store(&table_name) {
                        // Apply the version to the store
                        store.apply_recovered_version(row_version);
                    }
                }
            }
            WALOperationType::Delete => {
                // For deletes, we need to mark the row as deleted
                let table_name = entry.table_name.to_lowercase();
                if let Ok(store) = self.get_version_store(&table_name) {
                    store.mark_deleted(entry.row_id, entry.txn_id);
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

            // Data type (1 byte)
            if pos >= data.len() {
                return Err(Error::internal("invalid schema: missing data type"));
            }
            let data_type = DataType::from_u8(data[pos]).unwrap_or(DataType::Null);
            pos += 1;

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

            columns.push(SchemaColumn::with_constraints(
                i,
                &col_name,
                data_type,
                nullable,
                primary_key,
                auto_increment,
                default_expr,
                check_expr,
            ));
        }

        Ok(Schema::new(&table_name, columns))
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

        // Stop accepting new transactions
        self.registry.stop_accepting_transactions();

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

                // Read data type
                if pos >= data.len() {
                    return Err(Error::internal("invalid AddColumn data: missing data type"));
                }
                let data_type = DataType::from_u8(data[pos])
                    .ok_or_else(|| Error::internal("invalid data type byte"))?;
                pos += 1;

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
                // Note: create_column doesn't support default_expr, so we need enhanced version
                self.create_column_with_default(
                    &table_name,
                    &column_name,
                    data_type,
                    nullable,
                    default_expr,
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

                // Read data type
                if pos >= data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing data type",
                    ));
                }
                let data_type = DataType::from_u8(data[pos])
                    .ok_or_else(|| Error::internal("invalid data type byte"))?;
                pos += 1;

                // Read nullable
                if pos >= data.len() {
                    return Err(Error::internal(
                        "invalid ModifyColumn data: missing nullable",
                    ));
                }
                let nullable = data[pos] != 0;

                // Apply the MODIFY COLUMN using engine method
                self.modify_column(&table_name, &column_name, data_type, nullable)?;
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
    fn record_ddl(&self, table_name: &str, op: WALOperationType, schema_data: &[u8]) {
        if self.should_skip_wal() {
            return;
        }
        if let Some(ref pm) = *self.persistence {
            if pm.is_enabled() {
                if let Err(e) = pm.record_ddl_operation(table_name, op, schema_data) {
                    eprintln!("Warning: Failed to record DDL operation in WAL: {}", e);
                }
            }
        }
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

            // Data type (1 byte)
            buf.push(col.data_type.as_u8());

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

        buf
    }

    /// Creates a new table
    pub fn create_table(&self, schema: Schema) -> Result<Schema> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name = schema.table_name_lower.clone();

        {
            let schemas = self.schemas.read().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists);
            }
        }

        // Validate schema
        self.validate_schema(&schema)?;

        // Create version store for this table
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            Arc::clone(&self.registry) as Arc<dyn VisibilityChecker>,
        ));

        // Store schema and version store
        {
            let mut schemas = self.schemas.write().unwrap();
            schemas.insert(table_name.clone(), schema.clone());
        }
        {
            let mut stores = self.version_stores.write().unwrap();
            stores.insert(table_name, version_store);
        }

        // Record DDL operation in WAL
        let schema_data = Self::serialize_schema(&schema);
        self.record_ddl(
            &schema.table_name,
            WALOperationType::CreateTable,
            &schema_data,
        );

        Ok(schema)
    }

    /// Drops a table
    pub fn drop_table_internal(&self, name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name = name.to_lowercase();

        // Check if table exists
        {
            let schemas = self.schemas.read().unwrap();
            if !schemas.contains_key(&table_name) {
                return Err(Error::TableNotFound);
            }
        }

        // Record DDL operation in WAL (before removing - use original name)
        self.record_ddl(name, WALOperationType::DropTable, &[]);

        // Close and remove version store
        {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(&table_name) {
                store.close();
            }
        }

        // Remove schema
        {
            let mut schemas = self.schemas.write().unwrap();
            schemas.remove(&table_name);
        }

        Ok(())
    }

    /// Gets a version store for a table
    pub fn get_version_store(&self, name: &str) -> Result<Arc<VersionStore>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name = name.to_lowercase();

        let stores = self.version_stores.read().unwrap();
        stores.get(&table_name).cloned().ok_or(Error::TableNotFound)
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

        // Get and modify schema
        let mut schemas = self.schemas.write().unwrap();
        let schema = schemas
            .get_mut(&table_name_lower)
            .ok_or(Error::TableNotFound)?;

        // Check if column already exists
        if schema.has_column(column_name) {
            return Err(Error::DuplicateColumn);
        }

        // Add column to schema
        let column = crate::core::SchemaColumn::new(
            schema.columns.len(),
            column_name,
            data_type,
            nullable,
            false,
        );
        schema.add_column(column)?;

        // Also update version store schema
        let stores = self.version_stores.read().unwrap();
        if let Some(store) = stores.get(&table_name_lower) {
            let mut vs_schema = store.schema_mut();
            let col = crate::core::SchemaColumn::new(
                vs_schema.columns.len(),
                column_name,
                data_type,
                nullable,
                false,
            );
            vs_schema.add_column(col)?;
        }

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
    ) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Get and modify schema
        let mut schemas = self.schemas.write().unwrap();
        let schema = schemas
            .get_mut(&table_name_lower)
            .ok_or(Error::TableNotFound)?;

        // Check if column already exists
        if schema.has_column(column_name) {
            return Err(Error::DuplicateColumn);
        }

        // Add column to schema with default expression
        let mut column = crate::core::SchemaColumn::new(
            schema.columns.len(),
            column_name,
            data_type,
            nullable,
            false,
        );
        column.default_expr = default_expr.clone();
        schema.add_column(column)?;

        // Also update version store schema
        let stores = self.version_stores.read().unwrap();
        if let Some(store) = stores.get(&table_name_lower) {
            let mut vs_schema = store.schema_mut();
            let mut col = crate::core::SchemaColumn::new(
                vs_schema.columns.len(),
                column_name,
                data_type,
                nullable,
                false,
            );
            col.default_expr = default_expr;
            vs_schema.add_column(col)?;
        }

        Ok(())
    }

    /// Drops a column from a table
    pub fn drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Get and modify schema
        let mut schemas = self.schemas.write().unwrap();
        let schema = schemas
            .get_mut(&table_name_lower)
            .ok_or(Error::TableNotFound)?;

        // Check if column is primary key
        if let Some((_, col)) = schema.find_column(column_name) {
            if col.primary_key {
                return Err(Error::CannotDropPrimaryKey);
            }
        } else {
            return Err(Error::ColumnNotFound);
        }

        // Remove column from schema
        schema.remove_column(column_name)?;

        // Also update version store schema
        let stores = self.version_stores.read().unwrap();
        if let Some(store) = stores.get(&table_name_lower) {
            let mut vs_schema = store.schema_mut();
            vs_schema.remove_column(column_name)?;
        }

        Ok(())
    }

    /// Renames a column in a table
    pub fn rename_column(&self, table_name: &str, old_name: &str, new_name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let table_name_lower = table_name.to_lowercase();

        // Get and modify schema
        let mut schemas = self.schemas.write().unwrap();
        let schema = schemas
            .get_mut(&table_name_lower)
            .ok_or(Error::TableNotFound)?;

        // Check if old column exists
        if !schema.has_column(old_name) {
            return Err(Error::ColumnNotFound);
        }

        // Check if new column name already exists
        if schema.has_column(new_name) {
            return Err(Error::DuplicateColumn);
        }

        // Rename column in schema
        schema.rename_column(old_name, new_name)?;

        // Also update version store schema
        let stores = self.version_stores.read().unwrap();
        if let Some(store) = stores.get(&table_name_lower) {
            let mut vs_schema = store.schema_mut();
            vs_schema.rename_column(old_name, new_name)?;
        }

        Ok(())
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

        // Get and modify schema
        let mut schemas = self.schemas.write().unwrap();
        let schema = schemas
            .get_mut(&table_name_lower)
            .ok_or(Error::TableNotFound)?;

        // Check if column exists
        if !schema.has_column(column_name) {
            return Err(Error::ColumnNotFound);
        }

        // Modify column in schema
        schema.modify_column(column_name, Some(data_type), Some(nullable))?;

        // Also update version store schema
        let stores = self.version_stores.read().unwrap();
        if let Some(store) = stores.get(&table_name_lower) {
            let mut vs_schema = store.schema_mut();
            vs_schema.modify_column(column_name, Some(data_type), Some(nullable))?;
        }

        Ok(())
    }

    /// Renames a table
    pub fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let old_name_lower = old_name.to_lowercase();
        let new_name_lower = new_name.to_lowercase();

        // Check if old table exists
        {
            let schemas = self.schemas.read().unwrap();
            if !schemas.contains_key(&old_name_lower) {
                return Err(Error::TableNotFound);
            }
            if schemas.contains_key(&new_name_lower) {
                return Err(Error::TableAlreadyExists);
            }
        }

        // Update schemas map
        {
            let mut schemas = self.schemas.write().unwrap();
            if let Some(mut schema) = schemas.remove(&old_name_lower) {
                schema.table_name = new_name.to_string();
                schemas.insert(new_name_lower.clone(), schema);
            }
        }

        // Update version_stores map
        {
            let mut stores = self.version_stores.write().unwrap();
            if let Some(store) = stores.remove(&old_name_lower) {
                // Update the schema's table name within the store
                {
                    let mut vs_schema = store.schema_mut();
                    vs_schema.table_name = new_name.to_string();
                }
                stores.insert(new_name_lower, store);
            }
        }

        Ok(())
    }

    /// Validates a schema
    fn validate_schema(&self, schema: &Schema) -> Result<()> {
        if schema.table_name.is_empty() {
            return Err(Error::internal("schema missing table name"));
        }

        // Check for duplicate column names
        let mut seen_names = std::collections::HashSet::new();
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

            if !seen_names.insert(col.name.to_lowercase()) {
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
        let mut views = self.views.write().unwrap();

        // Check if view already exists
        if views.contains_key(&name_lower) {
            if if_not_exists {
                return Ok(());
            }
            return Err(Error::ViewAlreadyExists(name.to_string()));
        }

        // Check if a table with the same name exists
        let schemas = self.schemas.read().unwrap();
        if schemas.contains_key(&name_lower) {
            return Err(Error::internal(format!(
                "cannot create view '{}': a table with the same name exists",
                name
            )));
        }
        drop(schemas);

        // Create the view definition wrapped in Arc for cheap cloning
        let view_def = Arc::new(ViewDefinition::new(name, query));
        views.insert(name_lower.clone(), Arc::clone(&view_def));

        // Release the lock before recording to WAL
        drop(views);

        // Record to WAL for persistence
        let data = view_def.serialize();
        self.record_ddl(&name_lower, WALOperationType::CreateView, &data);

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
        self.record_ddl(&name_lower, WALOperationType::DropView, name.as_bytes());

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

        let schemas = self.schemas.read().unwrap();
        Ok(schemas.contains_key(&table_name.to_lowercase()))
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

    fn get_table_schema(&self, table_name: &str) -> Result<Schema> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let schemas = self.schemas.read().unwrap();
        schemas
            .get(&table_name.to_lowercase())
            .cloned()
            .ok_or(Error::TableNotFound)
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

        // Get all indexes from the version store
        Ok(store.get_all_indexes())
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

    fn create_snapshot(&self) -> Result<()> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Check if persistence is enabled
        let pm = match self.persistence.as_ref() {
            Some(pm) if pm.is_enabled() => pm,
            _ => return Ok(()), // No persistence, nothing to snapshot
        };

        // CRITICAL: Create checkpoint and capture the LSN atomically.
        // create_checkpoint() flushes and syncs all pending WAL entries, then returns
        // the LSN at the exact checkpoint point. This prevents the race condition where:
        // 1. We read LSN separately after checkpoint
        // 2. New transaction commits between checkpoint and LSN read
        // 3. Snapshot might capture inconsistent data
        //
        // By using the LSN returned from create_checkpoint(), we guarantee the LSN
        // corresponds exactly to the data that was synced to disk.
        let snapshot_lsn = pm.create_checkpoint(vec![])?;

        // CRITICAL: Capture the commit sequence at the same point as the checkpoint.
        // This ensures we only include transactions that were committed at the time
        // of the checkpoint during the snapshot iteration. Without this, a transaction
        // that commits during iteration would be incorrectly included in the snapshot.
        let snapshot_commit_seq = self.registry.current_commit_sequence();

        // Create snapshot directory
        let snapshot_dir = pm.path().join("snapshots");
        if let Err(e) = std::fs::create_dir_all(&snapshot_dir) {
            return Err(Error::internal(format!(
                "failed to create snapshot directory: {}",
                e
            )));
        }

        // Get all table schemas and version stores
        let schemas = self.schemas.read().unwrap();
        let stores = self.version_stores.read().unwrap();

        // ATOMIC SNAPSHOT STRATEGY:
        // 1. Write all snapshots to .tmp files first
        // 2. After ALL succeed, rename all .tmp files to final names
        // 3. If any fails, cleanup all .tmp files
        // This ensures we never have a partially complete snapshot set

        // Collect (temp_path, final_path, table_name) for atomic rename
        let mut pending_snapshots: Vec<(std::path::PathBuf, std::path::PathBuf, String)> =
            Vec::new();
        let mut all_succeeded = true;

        // Generate a consistent timestamp for all snapshots in this batch
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S%.3f").to_string();

        // Phase 1: Write all snapshots to temp files
        for (table_name, schema) in schemas.iter() {
            if let Some(store) = stores.get(table_name) {
                // Create table-specific snapshot directory
                let table_snapshot_dir = snapshot_dir.join(table_name);
                if let Err(e) = std::fs::create_dir_all(&table_snapshot_dir) {
                    eprintln!(
                        "Warning: Failed to create snapshot directory for {}: {}",
                        table_name, e
                    );
                    all_succeeded = false;
                    break;
                }

                // Write to .tmp file first, will rename after all succeed
                let final_path = table_snapshot_dir.join(format!("snapshot-{}.bin", timestamp));
                let temp_path = table_snapshot_dir.join(format!("snapshot-{}.bin.tmp", timestamp));

                // Create snapshot writer for temp file with captured LSN
                let mut writer = match super::snapshot::SnapshotWriter::with_source_lsn(
                    &temp_path,
                    snapshot_lsn,
                ) {
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

                // Write schema
                if let Err(e) = writer.write_schema(schema) {
                    eprintln!("Warning: Failed to write schema for {}: {}", table_name, e);
                    writer.fail();
                    all_succeeded = false;
                    break;
                }

                // Write all committed versions using commit sequence cutoff for consistency.
                // This ensures that transactions that commit after the checkpoint but before
                // iteration completes are excluded from the snapshot, maintaining consistency
                // between the WAL checkpoint and the snapshot contents.
                let mut write_error = false;
                store.for_each_committed_version_with_cutoff(
                    |_row_id, version| {
                        // Clone version with snapshot TxnID marker
                        let mut snapshot_version = version.clone();
                        snapshot_version.txn_id = -1; // Mark as snapshot version

                        if let Err(e) = writer.append_row(&snapshot_version) {
                            eprintln!(
                                "Warning: Failed to write row {} to snapshot: {}",
                                _row_id, e
                            );
                            write_error = true;
                            return false; // Stop iteration
                        }
                        true
                    },
                    snapshot_commit_seq,
                );

                if write_error {
                    writer.fail();
                    all_succeeded = false;
                    break;
                }

                // Finalize the snapshot (writes CRC32, syncs to disk)
                if let Err(e) = writer.finalize() {
                    eprintln!(
                        "Warning: Failed to finalize snapshot for {}: {}",
                        table_name, e
                    );
                    writer.fail();
                    all_succeeded = false;
                    break;
                }

                // Track this temp file for later rename
                pending_snapshots.push((temp_path, final_path, table_name.clone()));
            }
        }

        // Phase 2: If all snapshots succeeded, rename all temp files atomically
        // Track successfully renamed files for rollback on failure
        let mut renamed_successfully: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();

        if all_succeeded {
            // Collect unique directories that need syncing
            let mut dirs_to_sync: std::collections::HashSet<std::path::PathBuf> =
                std::collections::HashSet::new();

            for (temp_path, final_path, table_name) in &pending_snapshots {
                if let Err(e) = std::fs::rename(temp_path, final_path) {
                    eprintln!(
                        "Warning: Failed to rename snapshot for {}: {}",
                        table_name, e
                    );
                    // CRITICAL: Rollback all previously successful renames to maintain consistency
                    // Without this, partial renames cause different tables to have snapshots
                    // from different points in time, leading to data loss on recovery.
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
                // Track successful rename for potential rollback
                renamed_successfully.push((temp_path.clone(), final_path.clone()));
                // Track the directory for syncing
                if let Some(parent) = final_path.parent() {
                    dirs_to_sync.insert(parent.to_path_buf());
                }
            }

            // Sync directories to ensure renames are durable
            // This is important on some file systems (e.g., ext4) where
            // rename durability requires directory sync
            if all_succeeded {
                for dir in &dirs_to_sync {
                    if let Ok(dir_file) = std::fs::File::open(dir) {
                        let _ = dir_file.sync_all();
                    }
                }
            }
        }

        // Phase 3: Cleanup - if anything failed, remove all temp files
        if !all_succeeded {
            for (temp_path, _, _) in &pending_snapshots {
                let _ = std::fs::remove_file(temp_path);
            }
            eprintln!("Warning: Snapshot creation failed, all temp files cleaned up");
            return Ok(());
        }

        // CRITICAL: Order of operations for crash safety:
        // 1. Write metadata FIRST (atomic via temp file + rename)
        // 2. Truncate WAL (safe because metadata is now durable)
        // 3. Cleanup old snapshots LAST (safe because we have new snapshots + metadata)
        //
        // This order ensures that if crash happens:
        // - After step 1: Metadata exists, WAL has all data, recovery is safe
        // - After step 2: Metadata exists, new snapshots have data, recovery uses snapshot
        // - After step 3: Complete, old snapshots cleaned up
        //
        // Note: Each snapshot file also embeds source_lsn in its header (v3 format),
        // providing a fallback if metadata file is corrupted. load_snapshots() uses
        // max(metadata_lsn, max_header_lsn) to handle this case.

        // Phase 4: Write snapshot metadata BEFORE cleanup and truncation
        // Using binary format with magic number and checksum for data integrity
        let meta_path = snapshot_dir.join("snapshot_meta.bin");
        if let Err(e) = write_snapshot_metadata(&meta_path, snapshot_lsn) {
            eprintln!("Warning: Failed to write snapshot metadata: {}", e);
            return Ok(()); // Don't truncate WAL or cleanup if metadata write failed
        }

        // Phase 5: Truncate WAL to remove entries up to the snapshot LSN
        // Safe because: metadata is durable, all data up to snapshot_lsn is in snapshot files
        if snapshot_lsn > 0 {
            if let Err(e) = pm.truncate_wal(snapshot_lsn) {
                eprintln!("Warning: Failed to truncate WAL after snapshot: {}", e);
                // Continue to cleanup even if truncation fails - data is safe in snapshots
            }
        }

        // Phase 6: Cleanup old snapshots LAST (only after metadata is durable)
        // This is safe because we now have:
        // - New snapshots with embedded source_lsn
        // - Metadata pointing to new snapshot_lsn
        // - WAL truncated (or still present if truncation failed)
        let keep_count = pm.keep_count();
        if keep_count > 0 {
            for (_, _, table_name) in &pending_snapshots {
                if let Some(schema) = schemas.get(table_name) {
                    let disk_store =
                        super::snapshot::DiskVersionStore::new(&snapshot_dir, table_name, schema);
                    if let Ok(disk_store) = disk_store {
                        if let Err(e) = disk_store.cleanup_old_snapshots(keep_count) {
                            eprintln!(
                                "Warning: Failed to cleanup old snapshots for {}: {}",
                                table_name, e
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn record_create_index(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
    ) {
        if self.should_skip_wal() {
            return;
        }

        // Get table schema to look up column IDs and data types
        let schema = match self.get_table_schema(table_name) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Build column_ids and data_types from schema
        let mut column_ids = Vec::with_capacity(column_names.len());
        let mut data_types = Vec::with_capacity(column_names.len());

        for col_name in column_names {
            let col_name_lower = col_name.to_lowercase();
            if let Some((idx, col)) = schema
                .columns
                .iter()
                .enumerate()
                .find(|(_, c)| c.name.to_lowercase() == col_name_lower)
            {
                column_ids.push(idx as i32);
                data_types.push(col.data_type);
            } else {
                // Column not found, skip recording
                return;
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
        };

        // Serialize and record to WAL
        let data = index_meta.serialize();
        self.record_ddl(table_name, WALOperationType::CreateIndex, &data);
    }

    fn record_drop_index(&self, table_name: &str, index_name: &str) {
        if self.should_skip_wal() {
            return;
        }

        // For drop index, the entry.data is simply the index name as bytes
        self.record_ddl(
            table_name,
            WALOperationType::DropIndex,
            index_name.as_bytes(),
        );
    }

    fn record_alter_table_add_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        default_expr: Option<&str>,
    ) {
        if self.should_skip_wal() {
            return;
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name + column_name_len(2) + column_name
        //          + data_type(1) + nullable(1) + default_expr_len(2) + default_expr
        let mut data = Vec::new();
        data.push(1u8); // Operation type: AddColumn = 1

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Column name
        data.extend_from_slice(&(column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(column_name.as_bytes());

        // Data type
        data.push(data_type as u8);

        // Nullable
        data.push(if nullable { 1 } else { 0 });

        // Default expression
        if let Some(expr) = default_expr {
            data.extend_from_slice(&(expr.len() as u16).to_le_bytes());
            data.extend_from_slice(expr.as_bytes());
        } else {
            data.extend_from_slice(&0u16.to_le_bytes());
        }

        self.record_ddl(table_name, WALOperationType::AlterTable, &data);
    }

    fn record_alter_table_drop_column(&self, table_name: &str, column_name: &str) {
        if self.should_skip_wal() {
            return;
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

        self.record_ddl(table_name, WALOperationType::AlterTable, &data);
    }

    fn record_alter_table_rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) {
        if self.should_skip_wal() {
            return;
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

        self.record_ddl(table_name, WALOperationType::AlterTable, &data);
    }

    fn record_alter_table_modify_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
    ) {
        if self.should_skip_wal() {
            return;
        }

        // Serialize: operation_type(1) + table_name_len(2) + table_name
        //          + column_name_len(2) + column_name + data_type(1) + nullable(1)
        let mut data = Vec::new();
        data.push(4u8); // Operation type: ModifyColumn = 4

        // Table name
        data.extend_from_slice(&(table_name.len() as u16).to_le_bytes());
        data.extend_from_slice(table_name.as_bytes());

        // Column name
        data.extend_from_slice(&(column_name.len() as u16).to_le_bytes());
        data.extend_from_slice(column_name.as_bytes());

        // Data type
        data.push(data_type as u8);

        // Nullable
        data.push(if nullable { 1 } else { 0 });

        self.record_ddl(table_name, WALOperationType::AlterTable, &data);
    }

    fn record_alter_table_rename(&self, old_table_name: &str, new_table_name: &str) {
        if self.should_skip_wal() {
            return;
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

        self.record_ddl(old_table_name, WALOperationType::AlterTable, &data);
    }

    fn fetch_rows_by_ids(
        &self,
        table_name: &str,
        row_ids: &[i64],
    ) -> Result<Vec<(i64, crate::core::Row)>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        let store = self.get_version_store(table_name)?;

        // Use a high txn_id to see the latest committed version
        // INVALID_TRANSACTION_ID + 1 will see all committed transactions
        let read_txn_id = INVALID_TRANSACTION_ID + 1;

        Ok(store.get_visible_versions_batch(row_ids, read_txn_id))
    }

    fn get_row_fetcher(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> Vec<(i64, crate::core::Row)> + Send + Sync>> {
        if !self.is_open() {
            return Err(Error::EngineNotOpen);
        }

        // Get the version store reference once
        let store = self.get_version_store(table_name)?;

        // Use a high txn_id to see the latest committed version
        let read_txn_id = INVALID_TRANSACTION_ID + 1;

        // Return a closure that captures the store and can be called repeatedly
        // without the overhead of looking up the version store each time
        Ok(Box::new(move |row_ids: &[i64]| {
            store.get_visible_versions_batch(row_ids, read_txn_id)
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

    /// Start periodic cleanup of old transactions and deleted rows
    ///
    /// Returns a handle that can be used to stop the cleanup thread.
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

                // Uncomment for debugging:
                // if txn_count > 0 || row_count > 0 || prev_version_count > 0 {
                //     eprintln!(
                //         "Cleanup: {} transactions, {} deleted rows, {} previous versions",
                //         txn_count, row_count, prev_version_count
                //     );
                // }
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
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CleanupHandle {
    /// Stop the cleanup thread
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
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
    /// Shared reference to schemas
    schemas: Arc<RwLock<FxHashMap<String, Schema>>>,
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
        }
    }

    fn schemas(&self) -> &RwLock<FxHashMap<String, Schema>> {
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
}

impl TransactionEngineOperations for EngineOperations {
    fn get_table_for_transaction(&self, txn_id: i64, table_name: &str) -> Result<Box<dyn Table>> {
        let table_name_lower = table_name.to_lowercase();

        // Get version store
        let stores = self.version_stores().read().unwrap();
        let version_store = stores
            .get(&table_name_lower)
            .cloned()
            .ok_or(Error::TableNotFound)?;
        drop(stores);

        // Check if we have a cached transaction version store for this (txn_id, table_name)
        let cache_key = (txn_id, table_name_lower.clone());
        let txn_versions = {
            let cache = self.txn_version_stores().read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                Arc::clone(cached)
            } else {
                drop(cache);
                // Create new transaction version store and cache it
                let new_store = Arc::new(RwLock::new(TransactionVersionStore::new(
                    Arc::clone(&version_store),
                    txn_id,
                )));
                let mut cache = self.txn_version_stores().write().unwrap();
                cache.insert(cache_key, Arc::clone(&new_store));
                new_store
            }
        };

        // Create MVCC table with shared transaction version store
        let table = MVCCTable::new_with_shared_store(txn_id, version_store, txn_versions);

        Ok(Box::new(table))
    }

    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn Table>> {
        let table_name = name.to_lowercase();

        // Check if table already exists
        {
            let schemas = self.schemas().read().unwrap();
            if schemas.contains_key(&table_name) {
                return Err(Error::TableAlreadyExists);
            }
        }

        // Create version store for this table
        let version_store = Arc::new(VersionStore::with_visibility_checker(
            schema.table_name.clone(),
            schema.clone(),
            Arc::clone(&self.registry) as Arc<dyn VisibilityChecker>,
        ));

        // Store schema and version store
        {
            let mut schemas = self.schemas().write().unwrap();
            schemas.insert(table_name.clone(), schema);
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

        // Remove schema and version store
        {
            let mut schemas = self.schemas().write().unwrap();
            if schemas.remove(&table_name_lower).is_none() {
                return Err(Error::TableNotFound);
            }
        }
        {
            let mut stores = self.version_stores().write().unwrap();
            if let Some(store) = stores.remove(&table_name_lower) {
                store.close();
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

        // Check if old table exists and new name doesn't exist
        {
            let schemas = self.schemas().read().unwrap();
            if !schemas.contains_key(&old_name_lower) {
                return Err(Error::TableNotFound);
            }
            if schemas.contains_key(&new_name_lower) {
                return Err(Error::TableAlreadyExists);
            }
        }

        // Rename in schemas
        {
            let mut schemas = self.schemas().write().unwrap();
            if let Some(mut schema) = schemas.remove(&old_name_lower) {
                schema.table_name = new_name.to_string();
                schemas.insert(new_name_lower.clone(), schema);
            }
        }

        // Rename in version stores
        {
            let mut stores = self.version_stores().write().unwrap();
            if let Some(store) = stores.remove(&old_name_lower) {
                stores.insert(new_name_lower, store);
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
                        row_id,
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

                    if let Err(e) =
                        pm.record_dml_operation(txn_id, table_name, row_id, op, &version)
                    {
                        eprintln!("Warning: Failed to record DML in WAL: {}", e);
                        // Continue with other operations
                    }
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

        // Record commit in WAL
        if let Some(ref pm) = self.persistence() {
            if pm.is_enabled() {
                if let Err(e) = pm.record_commit(txn_id) {
                    eprintln!("Warning: Failed to record commit in WAL: {}", e);
                }
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

        // Iterate over all cached transaction version stores for this txn_id
        let cache = self.txn_version_stores().read().unwrap();

        for ((cached_txn_id, table_name), txn_store) in cache.iter() {
            if *cached_txn_id == txn_id {
                // Check if this store has pending changes
                let store = txn_store.read().unwrap();
                if store.has_local_changes() {
                    drop(store);

                    // Get the version store for this table
                    let stores = self.version_stores().read().unwrap();
                    if let Some(version_store) = stores.get(table_name).cloned() {
                        drop(stores);

                        // Create a table instance with shared transaction store
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

    fn commit_all_tables(&self, txn_id: i64) -> Result<()> {
        // Iterate over all cached transaction version stores for this txn_id
        // and use MvccTable::commit() which properly updates indexes
        let cache = self.txn_version_stores().read().unwrap();

        for ((cached_txn_id, table_name), txn_store) in cache.iter() {
            if *cached_txn_id == txn_id {
                // Check if there are local changes before committing
                let has_changes = {
                    let store = txn_store.read().unwrap();
                    store.has_local_changes()
                };

                if has_changes {
                    // Get the version store for this table
                    let stores = self.version_stores().read().unwrap();
                    if let Some(version_store) = stores.get(table_name).cloned() {
                        drop(stores);

                        // Create table and commit through it (updates indexes)
                        let mut table = MVCCTable::new_with_shared_store(
                            txn_id,
                            Arc::clone(&version_store),
                            Arc::clone(txn_store),
                        );
                        table.commit()?;
                    }
                }
            }
        }

        // Clean up the transaction version store cache for this transaction
        drop(cache);
        let mut cache = self.txn_version_stores().write().unwrap();
        cache.retain(|(cached_txn_id, _), _| *cached_txn_id != txn_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};

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
}
