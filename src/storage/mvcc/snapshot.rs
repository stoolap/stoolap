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

//! Snapshot system for MVCC engine
//!
//! This module provides disk-based snapshots for large database recovery optimization.
//! Instead of replaying the entire WAL, we can load snapshots and replay only
//! entries after the snapshot LSN.
//!

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::core::{DataType, Error, Result, Schema, SchemaColumn};
use crate::storage::mvcc::persistence::{deserialize_row_version, serialize_row_version};
use crate::storage::mvcc::version_store::RowVersion;

// ============================================================================
// Binary Format Constants
// ============================================================================

/// Magic bytes for snapshot file validation ("STSVSHD" - SToolaP VerSion Store HarD disk)
const MAGIC_BYTES: u64 = 0x5354534456534844;

/// Footer magic bytes ("STEND" - SToolaP END)
const FOOTER_MAGIC: u64 = 0x53544E445354454E;

/// File format version 3: New 64-byte header with source_lsn and header_size for extensibility
const FILE_FORMAT_VERSION: u32 = 3;

/// File header size in bytes (64-byte extensible header)
const FILE_HEADER_SIZE: usize = 64;

/// Index entry size in bytes (row_id: i64 + offset: u64)
const INDEX_ENTRY_SIZE: usize = 16;

/// Footer size in bytes (64-byte extensible footer)
const FOOTER_SIZE: usize = 64;

/// Default batch size for processing rows
const DEFAULT_BATCH_SIZE: usize = 1000;

/// Default block size for buffered I/O
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024; // 64KB

/// Minimum row data size to attempt compression (bytes)
const ROW_COMPRESSION_THRESHOLD: usize = 64;

/// Compression flag in the length field (MSB of u32)
/// If set, the row data is LZ4 compressed
const COMPRESSED_LENGTH_FLAG: u32 = 0x8000_0000;

// ============================================================================
// Feature Flags (4 bytes)
// ============================================================================

/// Feature flag: Data is compressed
#[allow(dead_code)]
const FEATURE_HAS_COMPRESSION: u32 = 1 << 0;
/// Feature flag: Has schema definition block
#[allow(dead_code)]
const FEATURE_HAS_SCHEMA_BLOCK: u32 = 1 << 1;
/// Feature flag: Has statistics extension
#[allow(dead_code)]
const FEATURE_HAS_STATISTICS: u32 = 1 << 2;
/// Feature flag: Has bloom filter extension
#[allow(dead_code)]
const FEATURE_HAS_BLOOM_FILTER: u32 = 1 << 3;
/// Feature flag: Has min/max index extension
#[allow(dead_code)]
const FEATURE_HAS_MIN_MAX_INDEX: u32 = 1 << 4;
/// Feature flag: This is an incremental snapshot
#[allow(dead_code)]
const FEATURE_INCREMENTAL_SNAPSHOT: u32 = 1 << 5;

// ============================================================================
// File Header and Footer structures
// ============================================================================

/// File header at the beginning of snapshot files (64 bytes)
///
/// Format (v3):
/// - Magic (8 bytes): 0x5354534456534844 "STSVSHD"
/// - Version (4 bytes): Format version (3)
/// - Feature Flags (4 bytes): Enabled features
/// - Header Size (4 bytes): Total header size (allows growth)
/// - Creation Time (8 bytes): Snapshot creation timestamp
/// - Source LSN (8 bytes): WAL LSN at snapshot time
/// - Prev Snap LSN (8 bytes): LSN of previous snapshot (for incremental)
/// - Schema Version (4 bytes): Schema version number
/// - Compression (1 byte): Compression algorithm (0=none)
/// - Reserved (15 bytes): For future use
#[derive(Debug, Clone, Copy)]
struct FileHeader {
    /// Magic bytes for validation
    magic: u64,
    /// File format version
    version: u32,
    /// Feature flags
    feature_flags: u32,
    /// Total header size (allows future growth)
    header_size: u32,
    /// Snapshot creation timestamp (Unix nanos)
    creation_time: i64,
    /// WAL LSN at snapshot time
    source_lsn: u64,
    /// LSN of previous snapshot (for incremental snapshots)
    prev_snap_lsn: u64,
    /// Schema version number
    schema_version: u32,
    /// Compression algorithm (0=none)
    compression: u8,
}

impl FileHeader {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let creation_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        Self {
            magic: MAGIC_BYTES,
            version: FILE_FORMAT_VERSION,
            feature_flags: 0,
            header_size: FILE_HEADER_SIZE as u32,
            creation_time,
            source_lsn: 0,
            prev_snap_lsn: 0,
            schema_version: 0,
            compression: 0,
        }
    }

    fn with_source_lsn(mut self, lsn: u64) -> Self {
        self.source_lsn = lsn;
        self
    }

    #[allow(dead_code)]
    fn with_prev_snap_lsn(mut self, lsn: u64) -> Self {
        self.prev_snap_lsn = lsn;
        self
    }

    fn to_bytes(self) -> [u8; FILE_HEADER_SIZE] {
        let mut buf = [0u8; FILE_HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic.to_le_bytes());
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.feature_flags.to_le_bytes());
        buf[16..20].copy_from_slice(&self.header_size.to_le_bytes());
        buf[20..28].copy_from_slice(&self.creation_time.to_le_bytes());
        buf[28..36].copy_from_slice(&self.source_lsn.to_le_bytes());
        buf[36..44].copy_from_slice(&self.prev_snap_lsn.to_le_bytes());
        buf[44..48].copy_from_slice(&self.schema_version.to_le_bytes());
        buf[48] = self.compression;
        // Bytes 49-63 are reserved (already zeroed)
        buf
    }

    fn from_bytes(data: &[u8]) -> Result<Self> {
        // Minimum header size for v3 is 64 bytes, but we support reading smaller v2 headers
        if data.len() < 16 {
            return Err(Error::internal("header data too short"));
        }

        let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
        if magic != MAGIC_BYTES {
            return Err(Error::internal(format!(
                "invalid snapshot file: magic mismatch (expected {:#x}, got {:#x})",
                MAGIC_BYTES, magic
            )));
        }

        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());

        // For v2 files (16-byte header), convert to v3 format
        if version <= 2 {
            let flags = u32::from_le_bytes(data[12..16].try_into().unwrap());
            return Ok(Self {
                magic,
                version,
                feature_flags: flags,
                header_size: 16, // Old header size
                creation_time: 0,
                source_lsn: 0,
                prev_snap_lsn: 0,
                schema_version: 0,
                compression: 0,
            });
        }

        // v3+ format (64 bytes)
        if data.len() < FILE_HEADER_SIZE {
            return Err(Error::internal("header data too short for v3 format"));
        }

        let feature_flags = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let header_size = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let creation_time = i64::from_le_bytes(data[20..28].try_into().unwrap());
        let source_lsn = u64::from_le_bytes(data[28..36].try_into().unwrap());
        let prev_snap_lsn = u64::from_le_bytes(data[36..44].try_into().unwrap());
        let schema_version = u32::from_le_bytes(data[44..48].try_into().unwrap());
        let compression = data[48];

        Ok(Self {
            magic,
            version,
            feature_flags,
            header_size,
            creation_time,
            source_lsn,
            prev_snap_lsn,
            schema_version,
            compression,
        })
    }

    /// Get the effective header size for reading (handles v2 compatibility)
    fn effective_header_size(&self) -> usize {
        if self.version <= 2 {
            16 // v2 header size
        } else {
            self.header_size as usize
        }
    }
}

/// Footer at the end of snapshot files (64 bytes)
///
/// Format (v3):
/// - Index Offset (8 bytes)
/// - Index Size (8 bytes)
/// - Row Count (8 bytes)
/// - TxnIDs Offset (8 bytes)
/// - TxnIDs Count (8 bytes)
/// - Data Checksum (4 bytes): CRC of row data only
/// - Reserved (12 bytes): For future use
/// - Footer Magic (8 bytes): "STEND"
#[derive(Debug, Clone, Copy)]
struct Footer {
    /// Offset to the index section
    index_offset: u64,
    /// Size of the index section
    index_size: u64,
    /// Total row count
    row_count: u64,
    /// Offset to committed transaction IDs section
    txn_ids_offset: u64,
    /// Number of committed transaction IDs
    txn_ids_count: u64,
    /// CRC32 checksum of row data section only
    data_checksum: u32,
    /// Footer magic bytes for validation
    magic: u64,
}

/// Legacy footer size for v2 files (48 bytes)
const LEGACY_FOOTER_SIZE: usize = 48;

impl Footer {
    fn to_bytes(self) -> [u8; FOOTER_SIZE] {
        let mut buf = [0u8; FOOTER_SIZE];
        buf[0..8].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.index_size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.row_count.to_le_bytes());
        buf[24..32].copy_from_slice(&self.txn_ids_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.txn_ids_count.to_le_bytes());
        buf[40..44].copy_from_slice(&self.data_checksum.to_le_bytes());
        // Bytes 44-55 are reserved (already zeroed)
        buf[56..64].copy_from_slice(&self.magic.to_le_bytes());
        buf
    }

    fn from_bytes(data: &[u8], version: u32) -> Result<Self> {
        // Handle v2 format (48 bytes with different layout)
        if version <= 2 {
            if data.len() < LEGACY_FOOTER_SIZE {
                return Err(Error::internal("footer data too short for v2 format"));
            }
            let magic = u64::from_le_bytes(data[40..48].try_into().unwrap());
            if magic != MAGIC_BYTES {
                return Err(Error::internal(format!(
                    "invalid snapshot file: footer magic mismatch (expected {:#x}, got {:#x})",
                    MAGIC_BYTES, magic
                )));
            }
            return Ok(Self {
                index_offset: u64::from_le_bytes(data[0..8].try_into().unwrap()),
                index_size: u64::from_le_bytes(data[8..16].try_into().unwrap()),
                row_count: u64::from_le_bytes(data[16..24].try_into().unwrap()),
                txn_ids_offset: u64::from_le_bytes(data[24..32].try_into().unwrap()),
                txn_ids_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
                data_checksum: 0,
                magic,
            });
        }

        // v3 format (64 bytes)
        if data.len() < FOOTER_SIZE {
            return Err(Error::internal("footer data too short for v3 format"));
        }

        let magic = u64::from_le_bytes(data[56..64].try_into().unwrap());
        if magic != FOOTER_MAGIC {
            return Err(Error::internal(format!(
                "invalid snapshot file: footer magic mismatch (expected {:#x}, got {:#x})",
                FOOTER_MAGIC, magic
            )));
        }

        Ok(Self {
            index_offset: u64::from_le_bytes(data[0..8].try_into().unwrap()),
            index_size: u64::from_le_bytes(data[8..16].try_into().unwrap()),
            row_count: u64::from_le_bytes(data[16..24].try_into().unwrap()),
            txn_ids_offset: u64::from_le_bytes(data[24..32].try_into().unwrap()),
            txn_ids_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
            data_checksum: u32::from_le_bytes(data[40..44].try_into().unwrap()),
            magic,
        })
    }

    /// Get the footer size based on version
    fn size_for_version(version: u32) -> usize {
        if version <= 2 {
            LEGACY_FOOTER_SIZE
        } else {
            FOOTER_SIZE
        }
    }
}

// ============================================================================
// Schema Serialization (for snapshots)
// ============================================================================

/// Serialize a schema to binary format for snapshot files
fn serialize_snapshot_schema(schema: &Schema) -> Vec<u8> {
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

        // Check constraint (length-prefixed string, 0 length if None)
        if let Some(ref check_expr) = col.check_expr {
            buf.extend_from_slice(&(check_expr.len() as u16).to_le_bytes());
            buf.extend_from_slice(check_expr.as_bytes());
        } else {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
    }

    // Created at timestamp (nanoseconds since Unix epoch)
    let created_nanos = schema.created_at.timestamp_nanos_opt().unwrap_or(0);
    buf.extend_from_slice(&created_nanos.to_le_bytes());

    // Updated at timestamp
    let updated_nanos = schema.updated_at.timestamp_nanos_opt().unwrap_or(0);
    buf.extend_from_slice(&updated_nanos.to_le_bytes());

    buf
}

/// Deserialize a schema from binary format
fn deserialize_snapshot_schema(data: &[u8]) -> Result<Schema> {
    if data.len() < 4 {
        return Err(Error::internal("schema data too short"));
    }

    let mut pos = 0;

    // Table name
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

    // Column count
    if pos + 2 > data.len() {
        return Err(Error::internal("invalid schema: missing column count"));
    }
    let column_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    // Columns
    let mut columns = Vec::with_capacity(column_count);
    for i in 0..column_count {
        // Column name
        if pos + 2 > data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} name length",
                i
            )));
        }
        let col_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if pos + col_name_len > data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} name",
                i
            )));
        }
        let col_name = String::from_utf8(data[pos..pos + col_name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid column name: {}", e)))?;
        pos += col_name_len;

        // Data type
        if pos >= data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} data type",
                i
            )));
        }
        let data_type = DataType::from_u8(data[pos]).unwrap_or(DataType::Null);
        pos += 1;

        // Nullable
        if pos >= data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} nullable",
                i
            )));
        }
        let nullable = data[pos] != 0;
        pos += 1;

        // Primary key
        if pos >= data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} primary_key",
                i
            )));
        }
        let primary_key = data[pos] != 0;
        pos += 1;

        // Auto-increment
        if pos >= data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} auto_increment",
                i
            )));
        }
        let auto_increment = data[pos] != 0;
        pos += 1;

        // Default expression
        if pos + 2 > data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} default expr length",
                i
            )));
        }
        let default_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let default_expr = if default_len > 0 {
            if pos + default_len > data.len() {
                return Err(Error::internal(format!(
                    "invalid schema: missing column {} default expr",
                    i
                )));
            }
            let expr = String::from_utf8(data[pos..pos + default_len].to_vec())
                .map_err(|e| Error::internal(format!("invalid default expr: {}", e)))?;
            pos += default_len;
            Some(expr)
        } else {
            None
        };

        // Check constraint
        if pos + 2 > data.len() {
            return Err(Error::internal(format!(
                "invalid schema: missing column {} check expr length",
                i
            )));
        }
        let check_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let check_expr = if check_len > 0 {
            if pos + check_len > data.len() {
                return Err(Error::internal(format!(
                    "invalid schema: missing column {} check expr",
                    i
                )));
            }
            let expr = String::from_utf8(data[pos..pos + check_len].to_vec())
                .map_err(|e| Error::internal(format!("invalid check expr: {}", e)))?;
            pos += check_len;
            Some(expr)
        } else {
            None
        };

        columns.push(SchemaColumn {
            id: i,
            name: col_name,
            data_type,
            nullable,
            primary_key,
            auto_increment,
            default_expr,
            default_value: None,
            check_expr,
        });
    }

    // Timestamps (optional for backward compatibility)
    let created_at = if pos + 8 <= data.len() {
        let nanos = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        if nanos > 0 {
            chrono::DateTime::from_timestamp_nanos(nanos)
        } else {
            chrono::Utc::now()
        }
    } else {
        chrono::Utc::now()
    };

    let updated_at = if pos + 8 <= data.len() {
        let nanos = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        if nanos > 0 {
            chrono::DateTime::from_timestamp_nanos(nanos)
        } else {
            chrono::Utc::now()
        }
    } else {
        chrono::Utc::now()
    };

    Ok(Schema::with_timestamps(
        table_name, columns, created_at, updated_at,
    ))
}

// ============================================================================
// SnapshotWriter - Writes table data to snapshot files
// ============================================================================

/// Writes row versions to a snapshot file
pub struct SnapshotWriter {
    /// Output file
    file: BufWriter<File>,
    /// File path for cleanup on failure
    file_path: PathBuf,
    /// Current write offset
    data_offset: u64,
    /// Row count
    row_count: u64,
    /// Index mapping row_id -> file offset
    row_index: BTreeMap<i64, u64>,
    /// Committed transaction IDs -> create_time
    committed_txn_ids: BTreeMap<i64, i64>,
    /// Whether the writer has failed
    failed: bool,
    /// Source LSN (WAL position at snapshot time)
    source_lsn: u64,
    /// Hasher for computing row data checksum (stored in footer)
    data_hasher: crc32fast::Hasher,
    /// Hasher for computing full-file CRC (incremental, avoids read-back)
    file_hasher: crc32fast::Hasher,
    /// Offset where row data starts (after header and schema)
    row_data_start: u64,
}

impl SnapshotWriter {
    /// Create a new snapshot writer
    pub fn new(file_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_source_lsn(file_path, 0)
    }

    /// Create a new snapshot writer with source LSN
    pub fn with_source_lsn(file_path: impl AsRef<Path>, source_lsn: u64) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Create parent directory if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::internal(format!("failed to create snapshot directory: {}", e))
            })?;
        }

        // Create the file
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&file_path)
            .map_err(|e| Error::internal(format!("failed to create snapshot file: {}", e)))?;

        let mut writer = BufWriter::with_capacity(DEFAULT_BLOCK_SIZE, file);

        // Write header with source_lsn
        let header = FileHeader::new().with_source_lsn(source_lsn);
        let header_bytes = header.to_bytes();
        writer
            .write_all(&header_bytes)
            .map_err(|e| Error::internal(format!("failed to write header: {}", e)))?;

        // Initialize file hasher with header bytes for incremental CRC computation
        let mut file_hasher = crc32fast::Hasher::new();
        file_hasher.update(&header_bytes);

        Ok(Self {
            file: writer,
            file_path,
            data_offset: FILE_HEADER_SIZE as u64,
            row_count: 0,
            row_index: BTreeMap::new(),
            committed_txn_ids: BTreeMap::new(),
            failed: false,
            source_lsn,
            data_hasher: crc32fast::Hasher::new(),
            file_hasher,
            row_data_start: 0, // Will be set after schema is written
        })
    }

    /// Mark the writer as failed (will delete file on close)
    pub fn fail(&mut self) {
        self.failed = true;
    }

    /// Get the source LSN
    pub fn source_lsn(&self) -> u64 {
        self.source_lsn
    }

    /// Write the table schema to the file
    pub fn write_schema(&mut self, schema: &Schema) -> Result<()> {
        let schema_bytes = serialize_snapshot_schema(schema);

        // Write length prefix
        let len_bytes = (schema_bytes.len() as u32).to_le_bytes();
        self.file
            .write_all(&len_bytes)
            .map_err(|e| Error::internal(format!("failed to write schema length: {}", e)))?;

        // Write schema data
        self.file
            .write_all(&schema_bytes)
            .map_err(|e| Error::internal(format!("failed to write schema: {}", e)))?;

        // Update file hasher for incremental CRC
        self.file_hasher.update(&len_bytes);
        self.file_hasher.update(&schema_bytes);

        self.data_offset += 4 + schema_bytes.len() as u64;

        // Mark where row data starts (after schema)
        self.row_data_start = self.data_offset;

        Ok(())
    }

    /// Append a single row version
    pub fn append_row(&mut self, version: &RowVersion) -> Result<()> {
        // Check for duplicate row_id
        if self.row_index.contains_key(&version.row_id) {
            return Err(Error::internal(format!(
                "duplicate row_id in snapshot: {}",
                version.row_id
            )));
        }

        // Serialize the row version
        let row_bytes = serialize_row_version(version)?;

        // Store the offset in our index
        self.row_index.insert(version.row_id, self.data_offset);

        // Track committed transaction
        self.committed_txn_ids
            .insert(version.txn_id, version.create_time);

        // Try compression for larger rows
        let (payload, is_compressed) = if row_bytes.len() >= ROW_COMPRESSION_THRESHOLD {
            let compressed = lz4_flex::compress_prepend_size(&row_bytes);
            if compressed.len() < row_bytes.len() {
                (compressed, true)
            } else {
                (row_bytes, false)
            }
        } else {
            (row_bytes, false)
        };

        // Write length prefix with compression flag in MSB
        let length_with_flag = if is_compressed {
            (payload.len() as u32) | COMPRESSED_LENGTH_FLAG
        } else {
            payload.len() as u32
        };
        let len_bytes = length_with_flag.to_le_bytes();
        self.file
            .write_all(&len_bytes)
            .map_err(|e| Error::internal(format!("failed to write row length: {}", e)))?;

        // Update data checksum with length prefix and row data
        self.data_hasher.update(&len_bytes);
        self.data_hasher.update(&payload);

        // Update file hasher for incremental CRC (same data)
        self.file_hasher.update(&len_bytes);
        self.file_hasher.update(&payload);

        // Write row data
        self.file
            .write_all(&payload)
            .map_err(|e| Error::internal(format!("failed to write row: {}", e)))?;

        self.data_offset += 4 + payload.len() as u64;
        self.row_count += 1;

        Ok(())
    }

    /// Append a batch of row versions
    pub fn append_batch(&mut self, versions: &[RowVersion]) -> Result<()> {
        for version in versions {
            self.append_row(version)?;
        }
        Ok(())
    }

    /// Finalize the snapshot file by writing the index and footer
    ///
    /// This method uses incremental CRC computation to avoid reading back
    /// the entire file, which significantly improves performance for large snapshots.
    pub fn finalize(&mut self) -> Result<()> {
        // Flush buffered data
        self.file
            .flush()
            .map_err(|e| Error::internal(format!("failed to flush: {}", e)))?;

        // Finalize data checksum (row data only)
        let data_checksum = std::mem::take(&mut self.data_hasher).finalize();

        // Build and write index section
        let index_offset = self.data_offset;
        let mut index_data = Vec::with_capacity(self.row_index.len() * INDEX_ENTRY_SIZE);

        for (&row_id, &offset) in &self.row_index {
            index_data.extend_from_slice(&row_id.to_le_bytes());
            index_data.extend_from_slice(&offset.to_le_bytes());
        }

        // Get the underlying file for direct writes
        let inner = self.file.get_mut();
        inner
            .seek(SeekFrom::Start(index_offset))
            .map_err(|e| Error::internal(format!("failed to seek to index: {}", e)))?;
        inner
            .write_all(&index_data)
            .map_err(|e| Error::internal(format!("failed to write index: {}", e)))?;

        // Update file hasher with index data (incremental CRC)
        self.file_hasher.update(&index_data);

        // Write transaction IDs section
        let txn_ids_offset = index_offset + index_data.len() as u64;
        let mut txn_data = Vec::with_capacity(self.committed_txn_ids.len() * 16);

        for (&txn_id, &timestamp) in &self.committed_txn_ids {
            txn_data.extend_from_slice(&txn_id.to_le_bytes());
            txn_data.extend_from_slice(&timestamp.to_le_bytes());
        }

        inner
            .write_all(&txn_data)
            .map_err(|e| Error::internal(format!("failed to write txn IDs: {}", e)))?;

        // Update file hasher with txn data (incremental CRC)
        self.file_hasher.update(&txn_data);

        // Write footer with data checksum and new footer magic
        let footer = Footer {
            index_offset,
            index_size: index_data.len() as u64,
            row_count: self.row_count,
            txn_ids_offset,
            txn_ids_count: self.committed_txn_ids.len() as u64,
            data_checksum,
            magic: FOOTER_MAGIC,
        };

        let footer_bytes = footer.to_bytes();
        inner
            .write_all(&footer_bytes)
            .map_err(|e| Error::internal(format!("failed to write footer: {}", e)))?;

        // Update file hasher with footer (incremental CRC)
        self.file_hasher.update(&footer_bytes);

        // Finalize incremental CRC - no file read-back needed!
        let crc = std::mem::take(&mut self.file_hasher).finalize();

        // Write CRC32 at end of file
        inner
            .write_all(&crc.to_le_bytes())
            .map_err(|e| Error::internal(format!("failed to write CRC: {}", e)))?;

        // Sync to disk
        inner
            .sync_all()
            .map_err(|e| Error::internal(format!("failed to sync: {}", e)))?;

        Ok(())
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        // Flush any remaining data
        let _ = self.file.flush();

        // Delete file if failed
        if self.failed {
            let _ = fs::remove_file(&self.file_path);
        }
    }
}

// ============================================================================
// SnapshotReader - Reads table data from snapshot files
// ============================================================================

/// Reads row versions from a snapshot file
pub struct SnapshotReader {
    /// Input file
    file: File,
    /// File path
    file_path: PathBuf,
    /// File header (kept for version info and future compatibility)
    #[allow(dead_code)]
    header: FileHeader,
    /// File footer
    footer: Footer,
    /// Table schema
    schema: Schema,
    /// Index mapping row_id -> file offset (BTreeMap for ordered access)
    index: BTreeMap<i64, u64>,
    /// Track which row_ids have been loaded to memory
    loaded_row_ids: RwLock<std::collections::HashSet<i64>>,
    /// Length buffer for reuse
    len_buffer: [u8; 4],
}

impl SnapshotReader {
    /// Open a snapshot file for reading
    pub fn open(file_path: impl AsRef<Path>) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        let mut file = File::open(&file_path)
            .map_err(|e| Error::internal(format!("failed to open snapshot: {}", e)))?;

        // Get file size
        let file_size = file
            .metadata()
            .map_err(|e| Error::internal(format!("failed to get file metadata: {}", e)))?
            .len();

        // First, read minimum header (16 bytes) to get version
        const MIN_HEADER_SIZE: usize = 16;
        if file_size < MIN_HEADER_SIZE as u64 {
            return Err(Error::internal("snapshot file too small for header"));
        }

        // Read first 16 bytes to determine version
        let mut min_header_data = [0u8; MIN_HEADER_SIZE];
        file.read_exact(&mut min_header_data)
            .map_err(|e| Error::internal(format!("failed to read header: {}", e)))?;

        // Get version from bytes 8-12
        let version = u32::from_le_bytes(min_header_data[8..12].try_into().unwrap());

        // Now read full header based on version
        let header = if version <= 2 {
            // v2 format: 16-byte header, parse directly
            FileHeader::from_bytes(&min_header_data)?
        } else {
            // v3+ format: 64-byte header, need to read more
            if file_size < FILE_HEADER_SIZE as u64 {
                return Err(Error::internal("snapshot file too small for v3 header"));
            }
            let mut header_data = [0u8; FILE_HEADER_SIZE];
            header_data[..MIN_HEADER_SIZE].copy_from_slice(&min_header_data);
            file.read_exact(&mut header_data[MIN_HEADER_SIZE..])
                .map_err(|e| Error::internal(format!("failed to read full header: {}", e)))?;
            FileHeader::from_bytes(&header_data)?
        };

        // Calculate footer size based on version
        let footer_size = Footer::size_for_version(header.version);

        // Check minimum file size
        let min_file_size = header.effective_header_size() + footer_size;
        if file_size < min_file_size as u64 {
            return Err(Error::internal("snapshot file too small"));
        }

        // For v2+ files, footer is followed by 4-byte CRC32
        let has_crc = header.version >= 2;
        let footer_offset = if has_crc {
            file_size - footer_size as u64 - 4
        } else {
            file_size - footer_size as u64
        };

        // Read footer
        file.seek(SeekFrom::Start(footer_offset))
            .map_err(|e| Error::internal(format!("failed to seek to footer: {}", e)))?;
        let mut footer_data = vec![0u8; footer_size];
        file.read_exact(&mut footer_data)
            .map_err(|e| Error::internal(format!("failed to read footer: {}", e)))?;
        let footer = Footer::from_bytes(&footer_data, header.version)?;

        // Verify CRC32 for v2 files
        if has_crc {
            // Read stored CRC32
            let mut crc_buf = [0u8; 4];
            file.read_exact(&mut crc_buf)
                .map_err(|e| Error::internal(format!("failed to read CRC32: {}", e)))?;
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Compute CRC32 of all data from header to footer (inclusive)
            file.seek(SeekFrom::Start(0)).map_err(|e| {
                Error::internal(format!("failed to seek for CRC verification: {}", e))
            })?;

            let mut hasher = crc32fast::Hasher::new();
            let mut buf = vec![0u8; 64 * 1024]; // 64KB buffer
            let data_len = footer_offset + footer_size as u64;
            let mut remaining = data_len;

            while remaining > 0 {
                let to_read = std::cmp::min(remaining as usize, buf.len());
                file.read_exact(&mut buf[..to_read]).map_err(|e| {
                    Error::internal(format!("failed to read for CRC verification: {}", e))
                })?;
                hasher.update(&buf[..to_read]);
                remaining -= to_read as u64;
            }

            let computed_crc = hasher.finalize();
            if stored_crc != computed_crc {
                return Err(Error::internal(format!(
                    "snapshot file CRC32 mismatch: stored={:#x}, computed={:#x}",
                    stored_crc, computed_crc
                )));
            }
        }

        // Read schema - use effective header size for v2/v3 compatibility
        file.seek(SeekFrom::Start(header.effective_header_size() as u64))
            .map_err(|e| Error::internal(format!("failed to seek to schema: {}", e)))?;

        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)
            .map_err(|e| Error::internal(format!("failed to read schema length: {}", e)))?;
        let schema_len = u32::from_le_bytes(len_buf) as usize;

        let mut schema_data = vec![0u8; schema_len];
        file.read_exact(&mut schema_data)
            .map_err(|e| Error::internal(format!("failed to read schema data: {}", e)))?;
        let schema = deserialize_snapshot_schema(&schema_data)?;

        // Load index
        file.seek(SeekFrom::Start(footer.index_offset))
            .map_err(|e| Error::internal(format!("failed to seek to index: {}", e)))?;

        let num_entries = footer.index_size as usize / INDEX_ENTRY_SIZE;
        let mut index_data = vec![0u8; footer.index_size as usize];
        file.read_exact(&mut index_data)
            .map_err(|e| Error::internal(format!("failed to read index: {}", e)))?;

        let mut index = BTreeMap::new();
        for i in 0..num_entries {
            let offset = i * INDEX_ENTRY_SIZE;
            let row_id = i64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
            let file_offset =
                u64::from_le_bytes(index_data[offset + 8..offset + 16].try_into().unwrap());
            index.insert(row_id, file_offset);
        }

        Ok(Self {
            file,
            file_path,
            header,
            footer,
            schema,
            index,
            loaded_row_ids: RwLock::new(std::collections::HashSet::new()),
            len_buffer: [0u8; 4],
        })
    }

    /// Get the table schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the file format version
    pub fn format_version(&self) -> u32 {
        self.header.version
    }

    /// Get the total row count
    pub fn row_count(&self) -> u64 {
        self.footer.row_count
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get the source WAL LSN at the time this snapshot was created
    /// Returns 0 for v2 files that don't have this field
    pub fn source_lsn(&self) -> u64 {
        self.header.source_lsn
    }

    /// Get the creation timestamp (Unix timestamp in milliseconds)
    /// Returns 0 for v2 files that don't have this field
    pub fn creation_time(&self) -> i64 {
        self.header.creation_time
    }

    /// Check if a row_id exists in this snapshot AND has not been loaded to memory yet.
    ///
    /// This is used to determine if a row should be fetched from disk during queries.
    /// Returns `false` if:
    /// - The row doesn't exist in this snapshot, OR
    /// - The row exists but has already been loaded to memory (marked via `mark_loaded`)
    ///
    /// For checking if a row exists in the snapshot regardless of load status,
    /// use `row_exists_in_index()` instead.
    pub fn has_unloaded_row(&self, row_id: i64) -> bool {
        // Check if already loaded to memory
        if self
            .loaded_row_ids
            .read()
            .map(|s| s.contains(&row_id))
            .unwrap_or(false)
        {
            return false; // Already in memory
        }
        self.index.contains_key(&row_id)
    }

    /// Check if a row_id exists in this snapshot's index (regardless of load status)
    ///
    /// This simply checks if the row is present in the snapshot file,
    /// without considering whether it has been loaded to memory.
    pub fn row_exists_in_index(&self, row_id: i64) -> bool {
        self.index.contains_key(&row_id)
    }

    /// Deprecated: Use `has_unloaded_row()` instead.
    /// This method is kept for backward compatibility.
    #[deprecated(since = "0.1.0", note = "Use has_unloaded_row() for clarity")]
    pub fn has_row(&self, row_id: i64) -> bool {
        self.has_unloaded_row(row_id)
    }

    /// Get a single row by row_id
    pub fn get_row(&mut self, row_id: i64) -> Option<RowVersion> {
        // Check if already loaded
        {
            let loaded = self.loaded_row_ids.read().ok()?;
            if loaded.contains(&row_id) {
                return None; // Already loaded
            }
        }

        // Look up offset
        let &offset = self.index.get(&row_id)?;

        // Read the row
        let version = self.read_row_at_offset(offset).ok()?;

        // Mark as loaded
        if let Ok(mut loaded) = self.loaded_row_ids.write() {
            loaded.insert(row_id);
        }

        Some(version)
    }

    /// Get multiple rows by row_ids
    pub fn get_rows(&mut self, row_ids: &[i64]) -> Vec<(i64, RowVersion)> {
        let mut results = Vec::new();

        for &row_id in row_ids {
            if let Some(version) = self.get_row(row_id) {
                results.push((row_id, version));
            }
        }

        results
    }

    /// Iterate over all rows (without loading tracking)
    pub fn for_each<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(i64, RowVersion) -> bool,
    {
        // Collect index entries (row_id, offset) to avoid borrow issues
        let entries: Vec<(i64, u64)> = self.index.iter().map(|(&k, &v)| (k, v)).collect();

        for (row_id, offset) in entries {
            match self.read_row_at_offset(offset) {
                Ok(version) => {
                    if !callback(row_id, version) {
                        break;
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(())
    }

    /// Get all rows (for full table scan)
    pub fn get_all_rows(&mut self) -> BTreeMap<i64, RowVersion> {
        let mut results = BTreeMap::new();

        // Collect entries first
        let entries: Vec<(i64, u64)> = self.index.iter().map(|(&k, &v)| (k, v)).collect();

        for (row_id, offset) in entries {
            if let Ok(version) = self.read_row_at_offset(offset) {
                results.insert(row_id, version);
            }
        }

        results
    }

    /// Read a row at a specific file offset
    fn read_row_at_offset(&mut self, offset: u64) -> Result<RowVersion> {
        use std::io::Read;

        // Seek to offset
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| Error::internal(format!("failed to seek: {}", e)))?;

        // Read length (with compression flag in MSB)
        self.file
            .read_exact(&mut self.len_buffer)
            .map_err(|e| Error::internal(format!("failed to read row length: {}", e)))?;
        let length_with_flag = u32::from_le_bytes(self.len_buffer);

        // Check compression flag and extract actual length
        let is_compressed = (length_with_flag & COMPRESSED_LENGTH_FLAG) != 0;
        let row_len = (length_with_flag & !COMPRESSED_LENGTH_FLAG) as usize;

        // Read row data
        let mut row_data = vec![0u8; row_len];
        self.file
            .read_exact(&mut row_data)
            .map_err(|e| Error::internal(format!("failed to read row data: {}", e)))?;

        // Decompress if needed
        let row_bytes = if is_compressed {
            lz4_flex::decompress_size_prepended(&row_data)
                .map_err(|e| Error::internal(format!("failed to decompress row data: {}", e)))?
        } else {
            row_data
        };

        // Deserialize
        deserialize_row_version(&row_bytes)
    }

    /// Mark a row_id as loaded (used during merge with memory)
    pub fn mark_loaded(&self, row_id: i64) {
        if let Ok(mut loaded) = self.loaded_row_ids.write() {
            loaded.insert(row_id);
        }
    }

    /// Check if a row_id has been loaded
    pub fn is_loaded(&self, row_id: i64) -> bool {
        self.loaded_row_ids
            .read()
            .map(|s| s.contains(&row_id))
            .unwrap_or(false)
    }

    /// Get the index for external iteration
    pub fn index(&self) -> &BTreeMap<i64, u64> {
        &self.index
    }
}

// ============================================================================
// DiskVersionStore - Manages on-disk snapshots for a table
// ============================================================================

/// Manages disk-based snapshots for a table
pub struct DiskVersionStore {
    /// Base directory for snapshots
    base_dir: PathBuf,
    /// Table name
    table_name: String,
    /// Active snapshot readers
    readers: RwLock<Vec<SnapshotReader>>,
    /// Schema hash for validation (kept for future schema migration support)
    #[allow(dead_code)]
    schema_hash: u64,
}

impl DiskVersionStore {
    /// Create a new disk version store
    pub fn new(base_dir: impl AsRef<Path>, table_name: &str, schema: &Schema) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let store_dir = base_dir.join(table_name);

        // Create directory
        fs::create_dir_all(&store_dir).map_err(|e| {
            Error::internal(format!(
                "failed to create snapshot directory for {}: {}",
                table_name, e
            ))
        })?;

        Ok(Self {
            base_dir,
            table_name: table_name.to_string(),
            readers: RwLock::new(Vec::new()),
            schema_hash: schema_hash(schema),
        })
    }

    /// Get the snapshot directory for this table
    pub fn snapshot_dir(&self) -> PathBuf {
        self.base_dir.join(&self.table_name)
    }

    /// Create a new snapshot from version data
    pub fn create_snapshot<F>(&self, row_iterator: F, schema: &Schema) -> Result<PathBuf>
    where
        F: FnMut(&mut dyn FnMut(i64, &RowVersion) -> bool),
    {
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S%.3f");
        let file_path = self
            .snapshot_dir()
            .join(format!("snapshot-{}.bin", timestamp));

        let mut writer = SnapshotWriter::new(&file_path)?;

        // Write schema
        writer.write_schema(schema)?;

        // Write rows from iterator
        let mut row_iterator = row_iterator;
        let mut batch: Vec<RowVersion> = Vec::with_capacity(DEFAULT_BATCH_SIZE);

        row_iterator(&mut |_row_id, version| {
            if !version.is_deleted() {
                // Clone with snapshot TxnID (-1)
                let mut snapshot_version = version.clone();
                snapshot_version.txn_id = -1;
                batch.push(snapshot_version);

                if batch.len() >= DEFAULT_BATCH_SIZE {
                    if writer.append_batch(&batch).is_err() {
                        return false;
                    }
                    batch.clear();
                }
            }
            true
        });

        // Write remaining rows
        if !batch.is_empty() {
            writer.append_batch(&batch)?;
        }

        // Finalize
        writer.finalize()?;

        Ok(file_path)
    }

    /// Load the most recent snapshot
    pub fn load_snapshots(&self) -> Result<()> {
        let snapshot_dir = self.snapshot_dir();

        if !snapshot_dir.exists() {
            return Ok(()); // No snapshots yet
        }

        // Find snapshot files
        let mut snapshot_files: Vec<PathBuf> = fs::read_dir(&snapshot_dir)
            .map_err(|e| Error::internal(format!("failed to read snapshot directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot-") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by name (which includes timestamp)
        snapshot_files.sort();

        if snapshot_files.is_empty() {
            return Ok(()); // No snapshots
        }

        // Try to load the newest snapshot
        let newest = snapshot_files.last().unwrap();
        match SnapshotReader::open(newest) {
            Ok(reader) => {
                let mut readers = self
                    .readers
                    .write()
                    .expect("snapshot readers lock poisoned in load_snapshots");
                readers.push(reader);
                Ok(())
            }
            Err(e) => {
                // Try older snapshots as fallback
                for path in snapshot_files.iter().rev().skip(1) {
                    if let Ok(reader) = SnapshotReader::open(path) {
                        let mut readers = self
                            .readers
                            .write()
                            .expect("snapshot readers lock poisoned in load_snapshots fallback");
                        readers.push(reader);
                        return Ok(());
                    }
                }
                Err(e)
            }
        }
    }

    /// Get a row from disk snapshots
    pub fn get_row(&self, row_id: i64) -> Option<RowVersion> {
        let mut readers = self.readers.write().ok()?;
        if readers.is_empty() {
            return None;
        }

        // Check newest reader
        let reader = readers.last_mut()?;
        reader.get_row(row_id)
    }

    /// Check if a row exists in disk snapshots AND has not been loaded to memory.
    ///
    /// This is used to determine if a row should be fetched from disk.
    /// Returns `false` if the row doesn't exist or has already been loaded.
    pub fn has_unloaded_row(&self, row_id: i64) -> bool {
        let readers = self.readers.read().ok();
        if let Some(readers) = readers {
            if let Some(reader) = readers.last() {
                return reader.has_unloaded_row(row_id);
            }
        }
        false
    }

    /// Deprecated: Use `has_unloaded_row()` instead.
    #[deprecated(since = "0.1.0", note = "Use has_unloaded_row() for clarity")]
    #[allow(deprecated)]
    pub fn has_row(&self, row_id: i64) -> bool {
        self.has_unloaded_row(row_id)
    }

    /// Mark a row_id as loaded to memory
    pub fn mark_row_loaded(&self, row_id: i64) {
        if let Ok(readers) = self.readers.read() {
            if let Some(reader) = readers.last() {
                reader.mark_loaded(row_id);
            }
        }
    }

    /// Cleanup old snapshots, keeping only the most recent `keep_count` files
    pub fn cleanup_old_snapshots(&self, keep_count: usize) -> Result<()> {
        let snapshot_dir = self.snapshot_dir();

        if !snapshot_dir.exists() {
            return Ok(());
        }

        // Find all snapshot files
        let mut snapshot_files: Vec<PathBuf> = fs::read_dir(&snapshot_dir)
            .map_err(|e| Error::internal(format!("failed to read snapshot directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot-") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        // If we have fewer snapshots than keep_count, nothing to do
        if snapshot_files.len() <= keep_count {
            return Ok(());
        }

        // Sort by name (which includes timestamp), newest first
        snapshot_files.sort();
        snapshot_files.reverse();

        // Keep the newest keep_count snapshots, delete the rest
        for file_to_delete in snapshot_files.iter().skip(keep_count) {
            if let Err(e) = fs::remove_file(file_to_delete) {
                // Log warning but don't fail - old snapshots may already be removed
                eprintln!(
                    "Warning: failed to delete old snapshot {:?}: {}",
                    file_to_delete, e
                );
            }
        }

        Ok(())
    }

    /// Close all readers
    pub fn close(&self) -> Result<()> {
        let mut readers = self
            .readers
            .write()
            .expect("snapshot readers lock poisoned in close");
        readers.clear();
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute a hash of the schema for validation
fn schema_hash(schema: &Schema) -> u64 {
    // FNV-1a hash
    let mut hash: u64 = 14695981039346656037;

    // Hash table name
    for c in schema.table_name.chars() {
        hash ^= c as u64;
        hash = hash.wrapping_mul(1099511628211);
    }

    // Hash columns
    for col in &schema.columns {
        // Hash column name
        for c in col.name.chars() {
            hash ^= c as u64;
            hash = hash.wrapping_mul(1099511628211);
        }

        // Hash column type
        hash ^= col.data_type.as_u8() as u64;
        hash = hash.wrapping_mul(1099511628211);

        // Hash nullable flag
        if col.nullable {
            hash ^= 1;
        }
        hash = hash.wrapping_mul(1099511628211);

        // Hash primary key flag
        if col.primary_key {
            hash ^= 1;
        }
        hash = hash.wrapping_mul(1099511628211);
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Row, Value};
    use tempfile::tempdir;

    fn create_test_schema() -> Schema {
        Schema::new(
            "test_table",
            vec![
                SchemaColumn::with_constraints(
                    0,
                    "id",
                    DataType::Integer,
                    false,
                    true,
                    true,
                    None,
                    None,
                ),
                SchemaColumn::with_constraints(
                    1,
                    "name",
                    DataType::Text,
                    true,
                    false,
                    false,
                    Some("'unnamed'".to_string()),
                    None,
                ),
                SchemaColumn::with_constraints(
                    2,
                    "value",
                    DataType::Float,
                    true,
                    false,
                    false,
                    None,
                    Some("value > 0".to_string()),
                ),
            ],
        )
    }

    #[test]
    fn test_schema_serialization() {
        let schema = create_test_schema();
        let data = serialize_snapshot_schema(&schema);
        let deserialized = deserialize_snapshot_schema(&data).unwrap();

        assert_eq!(deserialized.table_name, schema.table_name);
        assert_eq!(deserialized.columns.len(), schema.columns.len());

        for (orig, deser) in schema.columns.iter().zip(deserialized.columns.iter()) {
            assert_eq!(orig.name, deser.name);
            assert_eq!(orig.data_type, deser.data_type);
            assert_eq!(orig.nullable, deser.nullable);
            assert_eq!(orig.primary_key, deser.primary_key);
            assert_eq!(orig.auto_increment, deser.auto_increment);
            assert_eq!(orig.default_expr, deser.default_expr);
            assert_eq!(orig.check_expr, deser.check_expr);
        }
    }

    #[test]
    fn test_snapshot_writer_reader() {
        let dir = tempdir().unwrap();
        let snapshot_path = dir.path().join("test_snapshot.bin");

        let schema = create_test_schema();

        // Write snapshot
        {
            let mut writer = SnapshotWriter::new(&snapshot_path).unwrap();
            writer.write_schema(&schema).unwrap();

            for i in 1..=100 {
                let version = RowVersion::new(
                    1,
                    i,
                    Row::from_values(vec![
                        Value::Integer(i),
                        Value::text(format!("row_{}", i)),
                        Value::Float(i as f64 * 1.5),
                    ]),
                );
                writer.append_row(&version).unwrap();
            }

            writer.finalize().unwrap();
        }

        // Read snapshot
        {
            let mut reader = SnapshotReader::open(&snapshot_path).unwrap();

            assert_eq!(reader.row_count(), 100);
            assert_eq!(reader.schema().table_name, "test_table");

            // Get specific row
            let row = reader.get_row(50).unwrap();
            assert_eq!(row.row_id, 50);

            // Already loaded, should return None
            assert!(reader.get_row(50).is_none());

            // Get another row
            let row = reader.get_row(75).unwrap();
            assert_eq!(row.row_id, 75);
        }
    }

    #[test]
    fn test_snapshot_for_each() {
        let dir = tempdir().unwrap();
        let snapshot_path = dir.path().join("foreach_snapshot.bin");

        let schema = create_test_schema();

        // Write snapshot
        {
            let mut writer = SnapshotWriter::new(&snapshot_path).unwrap();
            writer.write_schema(&schema).unwrap();

            for i in 1..=50 {
                let version = RowVersion::new(
                    1,
                    i,
                    Row::from_values(vec![
                        Value::Integer(i),
                        Value::text(format!("item_{}", i)),
                        Value::Float(i as f64),
                    ]),
                );
                writer.append_row(&version).unwrap();
            }

            writer.finalize().unwrap();
        }

        // Iterate with for_each
        {
            let mut reader = SnapshotReader::open(&snapshot_path).unwrap();
            let mut count = 0;
            let mut sum = 0i64;

            reader
                .for_each(|row_id, version| {
                    count += 1;
                    sum += row_id;
                    assert_eq!(version.row_id, row_id);
                    true
                })
                .unwrap();

            assert_eq!(count, 50);
            assert_eq!(sum, (1..=50).sum::<i64>());
        }
    }

    #[test]
    fn test_disk_version_store() {
        let dir = tempdir().unwrap();
        let schema = create_test_schema();

        let dvs = DiskVersionStore::new(dir.path(), "test_table", &schema).unwrap();

        // Initially no snapshots
        assert!(dvs.load_snapshots().is_ok());
        assert!(dvs.get_row(1).is_none());
    }

    #[test]
    fn test_schema_hash() {
        let schema1 = create_test_schema();
        let schema2 = create_test_schema();

        assert_eq!(schema_hash(&schema1), schema_hash(&schema2));

        // Different schema should have different hash
        let mut schema3 = create_test_schema();
        schema3.table_name = "different_table".to_string();
        assert_ne!(schema_hash(&schema1), schema_hash(&schema3));
    }
}
