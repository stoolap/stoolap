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

//! Persistence Manager for MVCC Engine
//!
//! Coordinates all disk operations including:
//! - WAL (Write-Ahead Log) management
//! - Snapshot creation and loading
//! - Recovery from disk
//!

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::core::{DataType, Error, IndexType, Result, Row, Schema, Value};
use crate::storage::mvcc::version_store::RowVersion;
use crate::storage::mvcc::wal_manager::{WALEntry, WALManager, WALOperationType};
use crate::storage::PersistenceConfig;

/// Default snapshot interval (5 minutes)
pub const DEFAULT_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(300);

/// Default number of snapshots to keep
pub const DEFAULT_KEEP_SNAPSHOTS: usize = 3;

/// Special transaction ID for DDL operations
/// DDL operations (CREATE TABLE, DROP TABLE, etc.) are not part of a user transaction
/// and use this special marker to distinguish them from DML operations.
pub const DDL_TXN_ID: i64 = -2;

/// Index metadata for persistence
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// Index name
    pub name: String,
    /// Table the index belongs to
    pub table_name: String,
    /// Names of the columns this index is for
    pub column_names: Vec<String>,
    /// IDs of the columns in the table schema
    pub column_ids: Vec<i32>,
    /// Types of data in the index
    pub data_types: Vec<DataType>,
    /// Whether the index enforces uniqueness
    pub is_unique: bool,
    /// Type of index (BTree, Hash, Bitmap)
    pub index_type: IndexType,
}

impl IndexMetadata {
    /// Serialize to binary format
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Index name
        buf.extend_from_slice(&(self.name.len() as u16).to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());

        // Table name
        buf.extend_from_slice(&(self.table_name.len() as u16).to_le_bytes());
        buf.extend_from_slice(self.table_name.as_bytes());

        // Column count
        buf.extend_from_slice(&(self.column_names.len() as u16).to_le_bytes());

        // Column names
        for name in &self.column_names {
            buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
        }

        // Column IDs
        for id in &self.column_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }

        // Data types
        buf.extend_from_slice(&(self.data_types.len() as u16).to_le_bytes());
        for dt in &self.data_types {
            buf.push(dt.as_u8());
        }

        // Unique flag
        buf.push(if self.is_unique { 1 } else { 0 });

        // Index type (1 byte: 0=BTree, 1=Hash, 2=Bitmap, 3=MultiColumn)
        let index_type_byte = match self.index_type {
            IndexType::BTree => 0,
            IndexType::Hash => 1,
            IndexType::Bitmap => 2,
            IndexType::MultiColumn => 3,
        };
        buf.push(index_type_byte);

        buf
    }

    /// Deserialize from binary format
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::internal("empty metadata"));
        }

        let mut pos = 0;

        // Index name
        if pos + 2 > data.len() {
            return Err(Error::internal("invalid metadata: missing name length"));
        }
        let name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + name_len > data.len() {
            return Err(Error::internal("invalid metadata: missing name"));
        }
        let name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid name: {}", e)))?;
        pos += name_len;

        // Table name
        if pos + 2 > data.len() {
            return Err(Error::internal(
                "invalid metadata: missing table name length",
            ));
        }
        let table_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + table_name_len > data.len() {
            return Err(Error::internal("invalid metadata: missing table name"));
        }
        let table_name = String::from_utf8(data[pos..pos + table_name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid table name: {}", e)))?;
        pos += table_name_len;

        // Column count
        if pos + 2 > data.len() {
            return Err(Error::internal("invalid metadata: missing column count"));
        }
        let column_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        // Column names
        let mut column_names = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            if pos + 2 > data.len() {
                return Err(Error::internal(
                    "invalid metadata: missing column name length",
                ));
            }
            let col_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;

            if pos + col_name_len > data.len() {
                return Err(Error::internal("invalid metadata: missing column name"));
            }
            let col_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
                .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;
            pos += col_name_len;
            column_names.push(col_name);
        }

        // Column IDs
        let mut column_ids = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            if pos + 4 > data.len() {
                return Err(Error::internal("invalid metadata: missing column ID"));
            }
            column_ids.push(i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }

        // Data types
        if pos + 2 > data.len() {
            return Err(Error::internal("invalid metadata: missing data type count"));
        }
        let data_type_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let mut data_types = Vec::with_capacity(data_type_count);
        for _ in 0..data_type_count {
            if pos >= data.len() {
                return Err(Error::internal("invalid metadata: missing data type"));
            }
            let dt = DataType::from_u8(data[pos]).unwrap_or(DataType::Null);
            pos += 1;
            data_types.push(dt);
        }

        // Unique flag
        let is_unique = if pos < data.len() {
            let val = data[pos] != 0;
            pos += 1;
            val
        } else {
            false
        };

        // Index type (1 byte: 0=BTree, 1=Hash, 2=Bitmap, 3=MultiColumn)
        let index_type = if pos < data.len() {
            match data[pos] {
                0 => IndexType::BTree,
                1 => IndexType::Hash,
                2 => IndexType::Bitmap,
                3 => IndexType::MultiColumn,
                _ => IndexType::BTree,
            }
        } else {
            IndexType::BTree
        };

        Ok(Self {
            name,
            table_name,
            column_names,
            column_ids,
            data_types,
            is_unique,
            index_type,
        })
    }
}

/// Persistence metadata for tracking state
#[derive(Debug, Default)]
pub struct PersistenceMeta {
    /// Last snapshot time (Unix nanoseconds)
    pub last_snapshot_time: AtomicI64,
    /// LSN covered by the last snapshot
    pub last_snapshot_lsn: AtomicU64,
    /// Last WAL LSN (used during recovery)
    pub last_wal_lsn: AtomicU64,
}

/// Persistence Manager coordinates all disk operations
pub struct PersistenceManager {
    /// Base path for persistence files
    path: PathBuf,
    /// WAL manager
    wal: Option<WALManager>,
    /// Persistence metadata
    meta: PersistenceMeta,
    /// Whether persistence is enabled
    enabled: AtomicBool,
    /// Snapshot interval
    snapshot_interval: Duration,
    /// Number of snapshots to keep
    keep_count: usize,
    /// Running flag for background tasks
    running: AtomicBool,
    /// Table schemas cache
    schemas: RwLock<HashMap<String, Arc<Schema>>>,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(path: Option<&Path>, config: &PersistenceConfig) -> Result<Self> {
        // Memory-only mode if no path provided
        if path.is_none() || !config.enabled {
            return Ok(Self {
                path: PathBuf::new(),
                wal: None,
                meta: PersistenceMeta::default(),
                enabled: AtomicBool::new(false),
                snapshot_interval: DEFAULT_SNAPSHOT_INTERVAL,
                keep_count: DEFAULT_KEEP_SNAPSHOTS,
                running: AtomicBool::new(false),
                schemas: RwLock::new(HashMap::new()),
            });
        }

        let path = path.unwrap();

        // Create base directory
        fs::create_dir_all(path).map_err(|e| {
            Error::internal(format!("failed to create persistence directory: {}", e))
        })?;

        // Initialize WAL with config (including fast sync settings)
        let wal_path = path.join("wal");
        let wal = WALManager::with_config(&wal_path, config.sync_mode, Some(config))?;

        // Get initial LSN from WAL
        let initial_lsn = wal.current_lsn();

        // Configure intervals
        let snapshot_interval = if config.snapshot_interval > 0 {
            Duration::from_secs(config.snapshot_interval as u64)
        } else {
            DEFAULT_SNAPSHOT_INTERVAL
        };

        let keep_count = if config.keep_snapshots > 0 {
            config.keep_snapshots as usize
        } else {
            DEFAULT_KEEP_SNAPSHOTS
        };

        let pm = Self {
            path: path.to_path_buf(),
            wal: Some(wal),
            meta: PersistenceMeta::default(),
            enabled: AtomicBool::new(true),
            snapshot_interval,
            keep_count,
            running: AtomicBool::new(false),
            schemas: RwLock::new(HashMap::new()),
        };

        pm.meta.last_wal_lsn.store(initial_lsn, Ordering::Release);

        Ok(pm)
    }

    /// Check if persistence is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Start persistence operations
    pub fn start(&self) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        self.running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop persistence operations
    pub fn stop(&self) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        self.running.store(false, Ordering::Release);

        // Close WAL
        if let Some(ref wal) = self.wal {
            wal.close()?;
        }

        Ok(())
    }

    /// Record a DDL operation (CREATE TABLE, DROP TABLE, etc.)
    pub fn record_ddl_operation(
        &self,
        table_name: &str,
        op: WALOperationType,
        schema_data: &[u8],
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        let entry = WALEntry::new(
            DDL_TXN_ID, // DDL operations use special marker transaction ID
            table_name.to_string(),
            0,
            op,
            schema_data.to_vec(),
        );

        wal.append_entry(entry)?;

        // DDL operations are auto-committed (they don't participate in user transactions)
        // Write a commit marker so two-phase recovery will apply them
        wal.write_commit_marker(DDL_TXN_ID)?;

        Ok(())
    }

    /// Record an index operation (CREATE INDEX, DROP INDEX)
    pub fn record_index_operation(
        &self,
        table_name: &str,
        op: WALOperationType,
        index_data: &[u8],
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        let entry = WALEntry::new(
            DDL_TXN_ID,
            table_name.to_string(),
            0,
            op,
            index_data.to_vec(),
        );

        wal.append_entry(entry)?;

        // Index operations are auto-committed (like other DDL)
        // Write a commit marker so two-phase recovery will apply them
        wal.write_commit_marker(DDL_TXN_ID)?;

        Ok(())
    }

    /// Record a DML operation (INSERT, UPDATE, DELETE)
    pub fn record_dml_operation(
        &self,
        txn_id: i64,
        table_name: &str,
        row_id: i64,
        op: WALOperationType,
        version: &RowVersion,
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        // Serialize row data
        let data = serialize_row_version(version)?;

        let entry = WALEntry::new(txn_id, table_name.to_string(), row_id, op, data);

        wal.append_entry(entry)?;
        Ok(())
    }

    /// Record a transaction commit
    ///
    /// Uses commit_marker() which sets the COMMIT_MARKER flag for two-phase recovery
    pub fn record_commit(&self, txn_id: i64) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        // Use commit_marker to set COMMIT_MARKER flag for two-phase recovery
        wal.write_commit_marker(txn_id)?;

        Ok(())
    }

    /// Record a transaction rollback
    ///
    /// Uses abort_marker() which sets the ABORT_MARKER flag for two-phase recovery
    pub fn record_rollback(&self, txn_id: i64) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        // Use abort_marker to set ABORT_MARKER flag for two-phase recovery
        wal.write_abort_marker(txn_id)?;

        Ok(())
    }

    /// Replay WAL entries using two-phase recovery
    ///
    /// This method ensures crash consistency by:
    /// 1. Scanning to identify committed/aborted transactions
    /// 2. Only applying entries from committed transactions
    pub fn replay_two_phase<F>(
        &self,
        from_lsn: u64,
        callback: F,
    ) -> Result<super::wal_manager::TwoPhaseRecoveryInfo>
    where
        F: FnMut(super::wal_manager::WALEntry) -> Result<()>,
    {
        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        wal.replay_two_phase(from_lsn, callback)
    }

    /// Create a checkpoint and return the LSN at the checkpoint point
    ///
    /// Returns the LSN that represents the checkpoint point. All data up to
    /// this LSN is guaranteed to be durably written to disk when this returns.
    /// Returns 0 if persistence is not enabled.
    pub fn create_checkpoint(&self, active_transactions: Vec<i64>) -> Result<u64> {
        if !self.is_enabled() {
            return Ok(0);
        }

        let wal = self.wal.as_ref().ok_or(Error::WalNotInitialized)?;

        let checkpoint_lsn = wal.create_checkpoint(active_transactions)?;

        // Update last snapshot time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        self.meta.last_snapshot_time.store(now, Ordering::Release);

        Ok(checkpoint_lsn)
    }

    /// Get the persistence path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current WAL LSN
    pub fn current_lsn(&self) -> u64 {
        self.wal.as_ref().map(|w| w.current_lsn()).unwrap_or(0)
    }

    /// Truncate WAL to remove entries up to the given LSN
    ///
    /// This is used after a successful snapshot to reclaim disk space.
    /// Only entries with LSN > up_to_lsn are kept.
    pub fn truncate_wal(&self, up_to_lsn: u64) -> Result<()> {
        if let Some(wal) = &self.wal {
            wal.truncate_wal(up_to_lsn)
        } else {
            Ok(()) // No WAL, nothing to truncate
        }
    }

    /// Get the snapshot interval
    pub fn snapshot_interval(&self) -> Duration {
        self.snapshot_interval
    }

    /// Get the number of snapshots to keep
    pub fn keep_count(&self) -> usize {
        self.keep_count
    }

    /// Register a table schema
    pub fn register_schema(&self, name: &str, schema: Arc<Schema>) {
        let mut schemas = self
            .schemas
            .write()
            .expect("schemas lock poisoned in register_schema");
        schemas.insert(name.to_string(), schema);
    }

    /// Get a table schema
    pub fn get_schema(&self, name: &str) -> Option<Arc<Schema>> {
        let schemas = self
            .schemas
            .read()
            .expect("schemas lock poisoned in get_schema");
        schemas.get(name).cloned()
    }

    /// Remove a table schema
    pub fn remove_schema(&self, name: &str) {
        let mut schemas = self
            .schemas
            .write()
            .expect("schemas lock poisoned in remove_schema");
        schemas.remove(name);
    }
}

impl Drop for PersistenceManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Serialize a RowVersion to binary format
pub fn serialize_row_version(version: &RowVersion) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    // Transaction ID
    buf.extend_from_slice(&version.txn_id.to_le_bytes());

    // Deleted at transaction ID (0 if not deleted)
    buf.extend_from_slice(&version.deleted_at_txn_id.to_le_bytes());

    // Row ID
    buf.extend_from_slice(&version.row_id.to_le_bytes());

    // Create time
    buf.extend_from_slice(&version.create_time.to_le_bytes());

    // Data (Row - which is Vec<Value>)
    let values: Vec<&Value> = version.data.iter().collect();
    buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for value in values {
        let value_bytes = serialize_value(value)?;
        buf.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&value_bytes);
    }

    Ok(buf)
}

/// Deserialize a RowVersion from binary format
pub fn deserialize_row_version(data: &[u8]) -> Result<RowVersion> {
    if data.len() < 32 {
        return Err(Error::internal("data too short for RowVersion"));
    }

    let mut pos = 0;

    // Transaction ID
    let txn_id = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // Deleted at transaction ID
    let deleted_at_txn_id = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // Row ID
    let row_id = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // Create time
    let create_time = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // Data (values)
    if pos + 4 > data.len() {
        return Err(Error::internal("missing value count"));
    }
    let value_count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        if pos + 4 > data.len() {
            return Err(Error::internal("missing value length"));
        }
        let value_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + value_len > data.len() {
            return Err(Error::internal("missing value data"));
        }
        let value = deserialize_value(&data[pos..pos + value_len])?;
        pos += value_len;
        values.push(value);
    }

    Ok(RowVersion {
        txn_id,
        deleted_at_txn_id,
        data: Row::from_values(values),
        row_id,
        create_time,
    })
}

/// Serialize a Value to binary format
pub fn serialize_value(value: &Value) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    match value {
        Value::Null(dt) => {
            buf.push(0); // Type tag for Null
            buf.push(dt.as_u8()); // Store the DataType for typed nulls
        }
        Value::Boolean(b) => {
            buf.push(1);
            buf.push(if *b { 1 } else { 0 });
        }
        Value::Integer(i) => {
            buf.push(2);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(3);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::Text(s) => {
            buf.push(4);
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Timestamp(ts) => {
            // Use type tag 8 for binary timestamp format (seconds + subsec_nanos)
            // More efficient and preserves full nanosecond precision
            buf.push(8);
            buf.extend_from_slice(&ts.timestamp().to_le_bytes());
            buf.extend_from_slice(&ts.timestamp_subsec_nanos().to_le_bytes());
        }
        Value::Json(j) => {
            buf.push(6);
            buf.extend_from_slice(&(j.len() as u32).to_le_bytes());
            buf.extend_from_slice(j.as_bytes());
        }
    }

    Ok(buf)
}

/// Deserialize a Value from binary format
pub fn deserialize_value(data: &[u8]) -> Result<Value> {
    if data.is_empty() {
        return Err(Error::internal("empty value data"));
    }

    let type_tag = data[0];
    let rest = &data[1..];

    match type_tag {
        0 => {
            // Null with DataType
            if rest.is_empty() {
                Ok(Value::null_unknown())
            } else {
                let dt = DataType::from_u8(rest[0]).unwrap_or(DataType::Null);
                Ok(Value::Null(dt))
            }
        }
        1 => {
            // Boolean
            if rest.is_empty() {
                return Err(Error::internal("missing boolean value"));
            }
            Ok(Value::Boolean(rest[0] != 0))
        }
        2 => {
            // Integer
            if rest.len() < 8 {
                return Err(Error::internal("missing integer value"));
            }
            Ok(Value::Integer(i64::from_le_bytes(
                rest[..8].try_into().unwrap(),
            )))
        }
        3 => {
            // Float
            if rest.len() < 8 {
                return Err(Error::internal("missing float value"));
            }
            Ok(Value::Float(f64::from_le_bytes(
                rest[..8].try_into().unwrap(),
            )))
        }
        4 => {
            // Text
            if rest.len() < 4 {
                return Err(Error::internal("missing text length"));
            }
            let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
            if rest.len() < 4 + len {
                return Err(Error::internal("missing text data"));
            }
            let s = String::from_utf8(rest[4..4 + len].to_vec())
                .map_err(|e| Error::internal(format!("invalid text: {}", e)))?;
            Ok(Value::Text(Arc::from(s.as_str())))
        }
        5 => {
            // Legacy timestamp format (RFC3339 string) - for backward compatibility
            if rest.len() < 4 {
                return Err(Error::internal("missing timestamp length"));
            }
            let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
            if rest.len() < 4 + len {
                return Err(Error::internal("missing timestamp data"));
            }
            let s = String::from_utf8(rest[4..4 + len].to_vec())
                .map_err(|e| Error::internal(format!("invalid timestamp string: {}", e)))?;
            let ts = chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| Error::internal(format!("invalid timestamp: {}", e)))?;
            Ok(Value::Timestamp(ts))
        }
        8 => {
            // Binary timestamp format (seconds + subsec_nanos) - new efficient format
            if rest.len() < 12 {
                return Err(Error::internal("missing timestamp data"));
            }
            let secs = i64::from_le_bytes(rest[..8].try_into().unwrap());
            let nsecs = u32::from_le_bytes(rest[8..12].try_into().unwrap());
            let ts = chrono::DateTime::from_timestamp(secs, nsecs)
                .ok_or_else(|| Error::internal("invalid timestamp"))?;
            Ok(Value::Timestamp(ts))
        }
        6 => {
            // Json
            if rest.len() < 4 {
                return Err(Error::internal("missing json length"));
            }
            let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
            if rest.len() < 4 + len {
                return Err(Error::internal("missing json data"));
            }
            let s = String::from_utf8(rest[4..4 + len].to_vec())
                .map_err(|e| Error::internal(format!("invalid json: {}", e)))?;
            Ok(Value::Json(Arc::from(s.as_str())))
        }
        _ => Err(Error::internal(format!(
            "unknown value type tag: {}",
            type_tag
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::SyncMode;
    use chrono::Utc;
    use tempfile::tempdir;

    #[test]
    fn test_index_metadata_serialization() {
        let meta = IndexMetadata {
            name: "idx_test".to_string(),
            table_name: "test".to_string(),
            column_names: vec!["col1".to_string(), "col2".to_string()],
            column_ids: vec![0, 1],
            data_types: vec![DataType::Integer, DataType::Text],
            is_unique: true,
            index_type: IndexType::Hash,
        };

        let serialized = meta.serialize();
        let deserialized = IndexMetadata::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.name, "idx_test");
        assert_eq!(deserialized.table_name, "test");
        assert_eq!(deserialized.column_names, vec!["col1", "col2"]);
        assert_eq!(deserialized.column_ids, vec![0, 1]);
        assert!(deserialized.is_unique);
        assert_eq!(deserialized.index_type, IndexType::Hash);
    }

    #[test]
    fn test_index_metadata_all_types() {
        // Test all index types serialize/deserialize correctly
        for index_type in [
            IndexType::BTree,
            IndexType::Hash,
            IndexType::Bitmap,
            IndexType::MultiColumn,
        ] {
            let meta = IndexMetadata {
                name: "idx_test".to_string(),
                table_name: "test".to_string(),
                column_names: vec!["col1".to_string()],
                column_ids: vec![0],
                data_types: vec![DataType::Integer],
                is_unique: false,
                index_type,
            };

            let serialized = meta.serialize();
            let deserialized = IndexMetadata::deserialize(&serialized).unwrap();
            assert_eq!(deserialized.index_type, index_type);
        }
    }

    #[test]
    fn test_persistence_manager_disabled() {
        let config = PersistenceConfig::default();
        let pm = PersistenceManager::new(None, &config).unwrap();
        assert!(!pm.is_enabled());
    }

    #[test]
    fn test_persistence_manager_enabled() {
        let dir = tempdir().unwrap();
        let config = PersistenceConfig {
            enabled: true,
            ..Default::default()
        };
        let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
        assert!(pm.is_enabled());
        assert_eq!(pm.current_lsn(), 0);
    }

    #[test]
    fn test_persistence_manager_record_operations() {
        let dir = tempdir().unwrap();
        let config = PersistenceConfig {
            enabled: true,
            sync_mode: SyncMode::Full,
            ..Default::default()
        };
        let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();

        // Record DDL (auto-committed, so 2 entries: DDL + commit marker)
        pm.record_ddl_operation("test", WALOperationType::CreateTable, b"schema_data")
            .unwrap();
        assert_eq!(pm.current_lsn(), 2); // DDL entry + commit marker

        // Record DML
        let version = RowVersion::new(1, 100, Row::from_values(vec![Value::Integer(42)]));
        pm.record_dml_operation(1, "test", 100, WALOperationType::Insert, &version)
            .unwrap();
        assert_eq!(pm.current_lsn(), 3);

        // Record commit
        pm.record_commit(1).unwrap();
        assert_eq!(pm.current_lsn(), 4);
    }

    #[test]
    fn test_value_serialization() {
        // Test all value types
        let values = vec![
            Value::null_unknown(),
            Value::Boolean(true),
            Value::Integer(12345),
            Value::Float(3.54159),
            Value::text("hello world"),
            Value::Timestamp(Utc::now()),
            Value::json(r#"{"key": "value"}"#),
        ];

        for value in values {
            let serialized = serialize_value(&value).unwrap();
            let deserialized = deserialize_value(&serialized).unwrap();

            // Compare values - binary timestamp format preserves full nanosecond precision
            match (&value, &deserialized) {
                (Value::Timestamp(t1), Value::Timestamp(t2)) => {
                    // Full nanosecond precision comparison
                    assert_eq!(t1.timestamp(), t2.timestamp(), "Timestamp seconds mismatch");
                    assert_eq!(
                        t1.timestamp_subsec_nanos(),
                        t2.timestamp_subsec_nanos(),
                        "Timestamp nanoseconds mismatch"
                    );
                }
                _ => {
                    assert_eq!(value, deserialized, "Value mismatch for {:?}", value);
                }
            }
        }
    }

    #[test]
    fn test_row_version_serialization() {
        let version = RowVersion::new(
            123,
            100,
            Row::from_values(vec![
                Value::Integer(100),
                Value::text("test"),
                Value::Boolean(true),
            ]),
        );

        let serialized = serialize_row_version(&version).unwrap();
        let deserialized = deserialize_row_version(&serialized).unwrap();

        assert_eq!(deserialized.txn_id, 123);
        assert_eq!(deserialized.row_id, 100);
        assert_eq!(deserialized.deleted_at_txn_id, 0);
        assert_eq!(deserialized.data.len(), 3);
    }

    #[test]
    fn test_persistence_manager_replay() {
        let dir = tempdir().unwrap();
        let config = PersistenceConfig {
            enabled: true,
            sync_mode: SyncMode::Full,
            ..Default::default()
        };

        // Write some entries with commits
        {
            let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
            pm.start().unwrap();

            for i in 1..=5 {
                let version =
                    RowVersion::new(i, i * 100, Row::from_values(vec![Value::Integer(i * 10)]));
                pm.record_dml_operation(i, "test", i * 100, WALOperationType::Insert, &version)
                    .unwrap();
                // Commit each transaction
                pm.record_commit(i).unwrap();
            }

            pm.stop().unwrap();
        }

        // Replay entries using two-phase recovery
        {
            let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
            let mut data_count = 0;
            let mut commit_count = 0;

            pm.replay_two_phase(0, |entry| {
                assert!(entry.lsn > 0);
                if entry.is_commit_marker() {
                    commit_count += 1;
                } else {
                    data_count += 1;
                }
                Ok(())
            })
            .unwrap();

            assert_eq!(data_count, 5);
            assert_eq!(commit_count, 5); // 5 commit markers for 5 transactions
        }
    }

    #[test]
    fn test_persistence_manager_checkpoint() {
        let dir = tempdir().unwrap();
        let config = PersistenceConfig {
            enabled: true,
            sync_mode: SyncMode::Full,
            ..Default::default()
        };

        let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
        pm.start().unwrap();

        // Add some entries
        pm.record_ddl_operation("test", WALOperationType::CreateTable, b"schema")
            .unwrap();

        // Create checkpoint
        pm.create_checkpoint(vec![]).unwrap();

        // Verify checkpoint exists
        let checkpoint_path = dir.path().join("wal").join("checkpoint.meta");
        assert!(checkpoint_path.exists());

        pm.stop().unwrap();
    }
}
