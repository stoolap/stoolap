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

//! Write-Ahead Log (WAL) Manager
//!
//! Provides durable logging of database operations for crash recovery.
//! Implements the WAL protocol with configurable sync modes.
//!

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::{Error, Result};
use crate::storage::{PersistenceConfig, SyncMode};

/// Magic bytes for WAL entry marker ("WALE" in ASCII)
/// Used to detect entry boundaries and partial writes
const WAL_ENTRY_MAGIC: u32 = 0x454C4157;

/// Special transaction ID for marker entries (used after WAL truncation)
pub const MARKER_TXN_ID: i64 = -1000;

/// Default maximum WAL file size before rotation (64MB)
pub const DEFAULT_WAL_MAX_SIZE: u64 = 64 * 1024 * 1024;

/// Default flush trigger size (32KB)
pub const DEFAULT_WAL_FLUSH_TRIGGER: u64 = 32 * 1024;

/// Default buffer size (64KB)
pub const DEFAULT_WAL_BUFFER_SIZE: usize = 64 * 1024;

/// Magic number for checkpoint files ("CHKP")
const CHECKPOINT_MAGIC: u32 = 0x43504F49;

// ============================================================================
// WAL Entry Header Format V2 (32 bytes)
// ============================================================================
// Provides extensible header with version field and reserved space for future growth.
//
// Layout:
// ┌─────────────────────────────────────────────────────────────────┐
// │ Magic          (4 bytes)  0x454C4157 "WALE"                     │
// │ Version        (1 byte)   Format version (currently 2)          │
// │ Flags          (1 byte)   Bit flags for entry properties        │
// │ Header Size    (2 bytes)  Total header size (allows growth)     │
// │ LSN            (8 bytes)  Log sequence number                   │
// │ Previous LSN   (8 bytes)  LSN of previous entry (chain link)    │
// │ Entry Size     (4 bytes)  Size of data payload                  │
// │ Reserved       (4 bytes)  Reserved for future use               │
// └─────────────────────────────────────────────────────────────────┘

/// Current WAL entry format version
const WAL_FORMAT_VERSION: u8 = 2;

/// WAL entry header size in bytes
const WAL_HEADER_SIZE: u16 = 32;

/// Checkpoint format version (v2 = section-based format)
const CHECKPOINT_FORMAT_VERSION: u8 = 2;

/// Checkpoint header size in bytes
const CHECKPOINT_HEADER_SIZE: u16 = 32;

/// Minimum data size to attempt compression (bytes)
/// Data smaller than this is unlikely to benefit from LZ4 compression
const COMPRESSION_THRESHOLD: usize = 64;

/// Section types for checkpoint data
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointSectionType {
    /// WAL file info (current + previous filenames)
    WalFileInfo = 0x0001,
    /// Active (in-progress) transactions at checkpoint time
    ActiveTransactions = 0x0002,
    /// Committed transactions since last checkpoint
    CommittedTransactions = 0x0003,
}

impl CheckpointSectionType {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::WalFileInfo),
            0x0002 => Some(Self::ActiveTransactions),
            0x0003 => Some(Self::CommittedTransactions),
            _ => None,
        }
    }
}

/// WAL entry flags (stored in 1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WalFlags(u8);

impl WalFlags {
    /// No flags set
    pub const NONE: WalFlags = WalFlags(0);
    /// Data is compressed (reserved for future use)
    pub const COMPRESSED: WalFlags = WalFlags(1 << 0);
    /// This is a commit record marker
    pub const COMMIT_MARKER: WalFlags = WalFlags(1 << 1);
    /// This is an abort record marker
    pub const ABORT_MARKER: WalFlags = WalFlags(1 << 2);
    /// This is a checkpoint record
    pub const CHECKPOINT_MARKER: WalFlags = WalFlags(1 << 3);
    /// This is a snapshot start marker
    pub const SNAPSHOT_START: WalFlags = WalFlags(1 << 4);
    /// This is a snapshot complete marker
    pub const SNAPSHOT_COMPLETE: WalFlags = WalFlags(1 << 5);
    /// This is a WAL rotation marker
    pub const ROTATION_MARKER: WalFlags = WalFlags(1 << 6);

    /// Create flags from raw byte
    pub fn from_byte(byte: u8) -> Self {
        WalFlags(byte)
    }

    /// Get raw byte value
    pub fn as_byte(&self) -> u8 {
        self.0
    }

    /// Check if a specific flag is set
    pub fn contains(&self, other: WalFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Set a flag
    pub fn set(&mut self, flag: WalFlags) {
        self.0 |= flag.0;
    }

    /// Clear a flag
    pub fn clear(&mut self, flag: WalFlags) {
        self.0 &= !flag.0;
    }

    /// Combine two flags
    pub fn union(self, other: WalFlags) -> WalFlags {
        WalFlags(self.0 | other.0)
    }
}

/// WAL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WALOperationType {
    Insert = 1,
    Update = 2,
    Delete = 3,
    Commit = 4,
    Rollback = 5,
    CreateTable = 6,
    DropTable = 7,
    AlterTable = 8,
    CreateIndex = 9,
    DropIndex = 10,
    CreateView = 11,
    DropView = 12,
}

impl WALOperationType {
    /// Convert from u8
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(WALOperationType::Insert),
            2 => Some(WALOperationType::Update),
            3 => Some(WALOperationType::Delete),
            4 => Some(WALOperationType::Commit),
            5 => Some(WALOperationType::Rollback),
            6 => Some(WALOperationType::CreateTable),
            7 => Some(WALOperationType::DropTable),
            8 => Some(WALOperationType::AlterTable),
            9 => Some(WALOperationType::CreateIndex),
            10 => Some(WALOperationType::DropIndex),
            11 => Some(WALOperationType::CreateView),
            12 => Some(WALOperationType::DropView),
            _ => None,
        }
    }

    /// Check if this is a DDL operation
    pub fn is_ddl(&self) -> bool {
        matches!(
            self,
            WALOperationType::CreateTable
                | WALOperationType::DropTable
                | WALOperationType::AlterTable
                | WALOperationType::CreateIndex
                | WALOperationType::DropIndex
                | WALOperationType::CreateView
                | WALOperationType::DropView
        )
    }

    /// Check if this is a commit or rollback
    pub fn is_transaction_end(&self) -> bool {
        matches!(self, WALOperationType::Commit | WALOperationType::Rollback)
    }
}

/// WAL entry representing a single operation
#[derive(Debug, Clone)]
pub struct WALEntry {
    /// Log Sequence Number
    pub lsn: u64,
    /// Previous Log Sequence Number (for backward chaining)
    pub previous_lsn: u64,
    /// Entry flags
    pub flags: WalFlags,
    /// Transaction ID
    pub txn_id: i64,
    /// Table name (empty for commits/rollbacks)
    pub table_name: String,
    /// Row ID (0 for commits/rollbacks)
    pub row_id: i64,
    /// Operation type
    pub operation: WALOperationType,
    /// Serialized row data (empty for commits/rollbacks)
    pub data: Vec<u8>,
    /// Operation timestamp (nanoseconds since epoch)
    pub timestamp: i64,
}

impl WALEntry {
    /// Create a new WAL entry
    pub fn new(
        txn_id: i64,
        table_name: String,
        row_id: i64,
        operation: WALOperationType,
        data: Vec<u8>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        Self {
            lsn: 0,          // Will be assigned by WALManager
            previous_lsn: 0, // Will be assigned by WALManager
            flags: WalFlags::NONE,
            txn_id,
            table_name,
            row_id,
            operation,
            data,
            timestamp,
        }
    }

    /// Create a new WAL entry with flags
    pub fn with_flags(
        txn_id: i64,
        table_name: String,
        row_id: i64,
        operation: WALOperationType,
        data: Vec<u8>,
        flags: WalFlags,
    ) -> Self {
        let mut entry = Self::new(txn_id, table_name, row_id, operation, data);
        entry.flags = flags;
        entry
    }

    /// Create a commit entry (with COMMIT_MARKER flag for two-phase recovery)
    pub fn commit(txn_id: i64) -> Self {
        // Always set COMMIT_MARKER flag so two-phase recovery can identify commits
        Self::with_flags(
            txn_id,
            String::new(),
            0,
            WALOperationType::Commit,
            Vec::new(),
            WalFlags::COMMIT_MARKER,
        )
    }

    /// Create a commit marker entry (explicit commit record for two-phase recovery)
    /// Note: This is now equivalent to commit() - kept for API compatibility
    pub fn commit_marker(txn_id: i64) -> Self {
        Self::commit(txn_id)
    }

    /// Create a rollback entry (with ABORT_MARKER flag for two-phase recovery)
    pub fn rollback(txn_id: i64) -> Self {
        // Always set ABORT_MARKER flag so two-phase recovery can identify aborts
        Self::with_flags(
            txn_id,
            String::new(),
            0,
            WALOperationType::Rollback,
            Vec::new(),
            WalFlags::ABORT_MARKER,
        )
    }

    /// Create an abort marker entry (explicit abort record for two-phase recovery)
    /// Note: This is now equivalent to rollback() - kept for API compatibility
    pub fn abort_marker(txn_id: i64) -> Self {
        Self::rollback(txn_id)
    }

    /// Check if this entry is a commit marker
    pub fn is_commit_marker(&self) -> bool {
        self.flags.contains(WalFlags::COMMIT_MARKER) || self.operation == WALOperationType::Commit
    }

    /// Check if this entry is an abort marker
    pub fn is_abort_marker(&self) -> bool {
        self.flags.contains(WalFlags::ABORT_MARKER) || self.operation == WALOperationType::Rollback
    }

    /// Encode entry to binary format with integrity protection
    ///
    /// Format V2 (32-byte header with extensibility):
    /// ┌─────────────────────────────────────────────────────────────────┐
    /// │ HEADER (32 bytes)                                               │
    /// ├─────────────────────────────────────────────────────────────────┤
    /// │ Magic          (4 bytes)  0x454C4157 "WALE"                     │
    /// │ Version        (1 byte)   Format version (currently 2)          │
    /// │ Flags          (1 byte)   Entry flags                           │
    /// │ Header Size    (2 bytes)  Total header size (32)                │
    /// │ LSN            (8 bytes)  Log sequence number                   │
    /// │ Previous LSN   (8 bytes)  Previous entry LSN (chain link)       │
    /// │ Entry Size     (4 bytes)  Size of data payload                  │
    /// │ Reserved       (4 bytes)  Reserved for future use               │
    /// ├─────────────────────────────────────────────────────────────────┤
    /// │ DATA PORTION (variable):                                        │
    /// │   - TxnID (8 bytes)                                             │
    /// │   - TableNameLen (2 bytes) + TableName                          │
    /// │   - RowID (8 bytes)                                             │
    /// │   - Operation (1 byte)                                          │
    /// │   - Timestamp (8 bytes)                                         │
    /// │   - DataLen (4 bytes) + Data                                    │
    /// ├─────────────────────────────────────────────────────────────────┤
    /// │ CRC32 (4 bytes): checksum of header + data                      │
    /// └─────────────────────────────────────────────────────────────────┘
    pub fn encode(&self) -> Vec<u8> {
        // Determine if we should compress the data payload
        // Only compress if data is large enough to benefit
        let (payload_data, use_compression) = if self.data.len() >= COMPRESSION_THRESHOLD {
            let compressed = lz4_flex::compress_prepend_size(&self.data);
            // Only use compression if it actually reduces size
            if compressed.len() < self.data.len() {
                (compressed, true)
            } else {
                (self.data.clone(), false)
            }
        } else {
            (self.data.clone(), false)
        };

        // Calculate data portion size: txnID(8) + tableNameLen(2) + tableName + rowID(8) + op(1) + ts(8) + dataLen(4) + data
        let data_size = 8 + 2 + self.table_name.len() + 8 + 1 + 8 + 4 + payload_data.len();

        // Total buffer: header(32) + data + CRC(4)
        let mut buf = Vec::with_capacity(WAL_HEADER_SIZE as usize + data_size + 4);

        // ========== HEADER (32 bytes) ==========
        // Magic marker (4 bytes)
        buf.extend_from_slice(&WAL_ENTRY_MAGIC.to_le_bytes());

        // Version (1 byte)
        buf.push(WAL_FORMAT_VERSION);

        // Flags (1 byte) - set COMPRESSED if using compression
        let mut flags = self.flags;
        if use_compression {
            flags.set(WalFlags::COMPRESSED);
        }
        buf.push(flags.as_byte());

        // Header Size (2 bytes)
        buf.extend_from_slice(&WAL_HEADER_SIZE.to_le_bytes());

        // LSN (8 bytes)
        buf.extend_from_slice(&self.lsn.to_le_bytes());

        // Previous LSN (8 bytes)
        buf.extend_from_slice(&self.previous_lsn.to_le_bytes());

        // Entry Size (4 bytes) - size of data portion only
        buf.extend_from_slice(&(data_size as u32).to_le_bytes());

        // Reserved (4 bytes)
        buf.extend_from_slice(&[0u8; 4]);

        // ========== DATA PORTION ==========
        // TxnID (8 bytes)
        buf.extend_from_slice(&self.txn_id.to_le_bytes());

        // Table name length (2 bytes) + table name
        buf.extend_from_slice(&(self.table_name.len() as u16).to_le_bytes());
        buf.extend_from_slice(self.table_name.as_bytes());

        // RowID (8 bytes)
        buf.extend_from_slice(&self.row_id.to_le_bytes());

        // Operation (1 byte)
        buf.push(self.operation as u8);

        // Timestamp (8 bytes)
        buf.extend_from_slice(&self.timestamp.to_le_bytes());

        // Data length (4 bytes) + data (possibly compressed)
        // When compressed, lz4_flex::compress_prepend_size includes the original size
        buf.extend_from_slice(&(payload_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&payload_data);

        // ========== CRC32 (4 bytes) ==========
        // Calculate CRC over data portion only (starting after 32-byte header)
        // This allows decode() to verify integrity without needing the header bytes
        let crc = crc32fast::hash(&buf[WAL_HEADER_SIZE as usize..]);
        buf.extend_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Decode entry from data portion (after header has been parsed)
    ///
    /// Parameters:
    /// - lsn, previous_lsn, flags: extracted from header by caller
    /// - data: data portion + CRC (4 bytes)
    pub fn decode(lsn: u64, previous_lsn: u64, flags: WalFlags, data: &[u8]) -> Result<Self> {
        // Minimum size: txnID(8) + tableNameLen(2) + rowID(8) + op(1) + ts(8) + dataLen(4) + CRC(4) = 35
        if data.len() < 35 {
            return Err(Error::internal(format!(
                "data too short for WAL entry: {} bytes",
                data.len()
            )));
        }

        // Verify CRC32 (last 4 bytes)
        let crc_offset = data.len() - 4;
        let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&data[..crc_offset]);

        if stored_crc != computed_crc {
            return Err(Error::internal(format!(
                "WAL entry checksum mismatch at LSN {}: stored={:#x}, computed={:#x}",
                lsn, stored_crc, computed_crc
            )));
        }

        // Data portion (excluding CRC)
        let data = &data[..crc_offset];
        let mut pos = 0;

        // TxnID (8 bytes)
        if pos + 8 > data.len() {
            return Err(Error::internal("unexpected end of data reading txn_id"));
        }
        let txn_id = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Table name length (2 bytes)
        if pos + 2 > data.len() {
            return Err(Error::internal(
                "unexpected end of data reading table name length",
            ));
        }
        let table_name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        // Table name
        if pos + table_name_len > data.len() {
            return Err(Error::internal("unexpected end of data reading table name"));
        }
        let table_name = String::from_utf8(data[pos..pos + table_name_len].to_vec())
            .map_err(|e| Error::internal(format!("invalid table name: {}", e)))?;
        pos += table_name_len;

        // RowID (8 bytes)
        if pos + 8 > data.len() {
            return Err(Error::internal("unexpected end of data reading row_id"));
        }
        let row_id = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Operation (1 byte)
        if pos + 1 > data.len() {
            return Err(Error::internal("unexpected end of data reading operation"));
        }
        let operation = WALOperationType::from_u8(data[pos])
            .ok_or_else(|| Error::internal(format!("invalid operation type: {}", data[pos])))?;
        pos += 1;

        // Timestamp (8 bytes)
        if pos + 8 > data.len() {
            return Err(Error::internal("unexpected end of data reading timestamp"));
        }
        let timestamp = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Data length (4 bytes)
        if pos + 4 > data.len() {
            return Err(Error::internal(
                "unexpected end of data reading data length",
            ));
        }
        let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        // Data
        if pos + data_len > data.len() {
            return Err(Error::internal("unexpected end of data reading data"));
        }
        let raw_data = &data[pos..pos + data_len];

        // Decompress if COMPRESSED flag is set
        let entry_data = if flags.contains(WalFlags::COMPRESSED) {
            lz4_flex::decompress_size_prepended(raw_data).map_err(|e| {
                Error::internal(format!("failed to decompress WAL entry data: {}", e))
            })?
        } else {
            raw_data.to_vec()
        };

        Ok(WALEntry {
            lsn,
            previous_lsn,
            flags,
            txn_id,
            table_name,
            row_id,
            operation,
            data: entry_data,
            timestamp,
        })
    }

    /// Check if this is a marker entry (used after WAL truncation)
    pub fn is_marker_entry(&self) -> bool {
        self.txn_id == MARKER_TXN_ID
    }
}

/// Checkpoint metadata
/// Committed transaction info for recovery
#[derive(Debug, Clone)]
pub struct CommittedTxnInfo {
    /// Transaction ID
    pub txn_id: i64,
    /// LSN of the commit record
    pub commit_lsn: u64,
}

#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Current WAL file name
    pub wal_file: String,
    /// Previous WAL file name (for rotation)
    pub previous_wal_file: Option<String>,
    /// Last sequence number included in this checkpoint
    pub lsn: u64,
    /// When this checkpoint was created (Unix timestamp in nanoseconds)
    pub timestamp: i64,
    /// Whether the checkpoint represents a consistent state
    pub is_consistent: bool,
    /// List of transaction IDs active at checkpoint time
    pub active_transactions: Vec<i64>,
    /// List of committed transactions since last checkpoint (for two-phase recovery)
    pub committed_transactions: Vec<CommittedTxnInfo>,
}

/// Information returned from two-phase WAL recovery
#[derive(Debug, Clone)]
pub struct TwoPhaseRecoveryInfo {
    /// Last LSN processed
    pub last_lsn: u64,
    /// Number of committed transactions found
    pub committed_transactions: usize,
    /// Number of aborted transactions found
    pub aborted_transactions: usize,
    /// Number of WAL entries applied (from committed transactions)
    pub applied_entries: u64,
    /// Number of WAL entries skipped (from aborted/in-doubt transactions)
    pub skipped_entries: u64,
}

impl CheckpointMetadata {
    /// Read checkpoint metadata from file (section-based v2 format)
    ///
    /// Header format (32 bytes):
    /// - Magic (4 bytes) - CHECKPOINT_MAGIC
    /// - Version (1 byte) - Format version
    /// - Flags (1 byte) - Reserved
    /// - Header Size (2 bytes) - Total header size
    /// - LSN (8 bytes) - Checkpoint LSN
    /// - Timestamp (8 bytes) - Creation timestamp
    /// - Section Count (2 bytes) - Number of sections
    /// - Reserved (6 bytes) - For future use
    ///
    /// Then section headers (8 bytes each):
    /// - Type (2 bytes) - Section type
    /// - Flags (2 bytes) - Section flags
    /// - Size (4 bytes) - Section data size
    ///
    /// Then section data, followed by CRC32
    pub fn read_from_file(path: &Path) -> Result<Self> {
        let data = fs::read(path)
            .map_err(|e| Error::internal(format!("failed to read checkpoint: {}", e)))?;

        if data.len() < CHECKPOINT_HEADER_SIZE as usize + 4 {
            return Err(Error::internal("invalid checkpoint file: too small"));
        }

        // Verify CRC first
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&data[..data.len() - 4]);
        if stored_crc != computed_crc {
            return Err(Error::internal(format!(
                "checkpoint CRC mismatch: stored=0x{:08x}, computed=0x{:08x}",
                stored_crc, computed_crc
            )));
        }

        let mut pos = 0;

        // Parse 32-byte header
        let magic = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        if magic != CHECKPOINT_MAGIC {
            return Err(Error::internal("invalid checkpoint magic number"));
        }
        pos += 4;

        let version = data[pos];
        pos += 1;

        let _flags = data[pos];
        pos += 1;

        let header_size = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        let lsn = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let timestamp = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let section_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        // Skip reserved bytes
        pos += 6;

        // Ensure we're past the header (for future extensibility)
        if header_size > 32 {
            pos = header_size;
        }

        // Read section headers
        struct SectionInfo {
            section_type: u16,
            _flags: u16,
            size: u32,
        }

        let mut sections = Vec::with_capacity(section_count);
        for _ in 0..section_count {
            if pos + 8 > data.len() - 4 {
                break;
            }
            let section_type = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            let flags = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
            let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
            sections.push(SectionInfo {
                section_type,
                _flags: flags,
                size,
            });
            pos += 8;
        }

        // Initialize default values
        let mut wal_file = String::new();
        let mut previous_wal_file = None;
        let mut is_consistent = false;
        let mut active_transactions = Vec::new();
        let mut committed_transactions = Vec::new();

        // Read section data
        for section in sections {
            let section_end = pos + section.size as usize;
            if section_end > data.len() - 4 {
                break;
            }

            match CheckpointSectionType::from_u16(section.section_type) {
                Some(CheckpointSectionType::WalFileInfo) => {
                    // is_consistent (1 byte)
                    is_consistent = data[pos] != 0;
                    let mut spos = pos + 1;

                    // Current WAL filename
                    if spos + 2 <= section_end {
                        let len =
                            u16::from_le_bytes(data[spos..spos + 2].try_into().unwrap()) as usize;
                        spos += 2;
                        if spos + len <= section_end {
                            wal_file = String::from_utf8(data[spos..spos + len].to_vec())
                                .unwrap_or_default();
                            spos += len;
                        }
                    }

                    // Previous WAL filename (optional)
                    if spos + 2 <= section_end {
                        let len =
                            u16::from_le_bytes(data[spos..spos + 2].try_into().unwrap()) as usize;
                        spos += 2;
                        if len > 0 && spos + len <= section_end {
                            previous_wal_file = Some(
                                String::from_utf8(data[spos..spos + len].to_vec())
                                    .unwrap_or_default(),
                            );
                        }
                    }
                }
                Some(CheckpointSectionType::ActiveTransactions) => {
                    // Count (8 bytes) + txn_ids (8 bytes each)
                    if pos + 8 <= section_end {
                        let count =
                            u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                        let mut spos = pos + 8;
                        active_transactions = Vec::with_capacity(count);
                        for _ in 0..count {
                            if spos + 8 > section_end {
                                break;
                            }
                            let txn_id =
                                i64::from_le_bytes(data[spos..spos + 8].try_into().unwrap());
                            active_transactions.push(txn_id);
                            spos += 8;
                        }
                    }
                }
                Some(CheckpointSectionType::CommittedTransactions) => {
                    // Count (8 bytes) + (txn_id(8) + commit_lsn(8)) each
                    if pos + 8 <= section_end {
                        let count =
                            u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
                        let mut spos = pos + 8;
                        committed_transactions = Vec::with_capacity(count);
                        for _ in 0..count {
                            if spos + 16 > section_end {
                                break;
                            }
                            let txn_id =
                                i64::from_le_bytes(data[spos..spos + 8].try_into().unwrap());
                            let commit_lsn =
                                u64::from_le_bytes(data[spos + 8..spos + 16].try_into().unwrap());
                            committed_transactions.push(CommittedTxnInfo { txn_id, commit_lsn });
                            spos += 16;
                        }
                    }
                }
                None => {
                    // Unknown section type - skip (for forward compatibility)
                    if version >= CHECKPOINT_FORMAT_VERSION {
                        // Log warning for future versions
                        eprintln!(
                            "Warning: Unknown checkpoint section type 0x{:04x} (version {})",
                            section.section_type, version
                        );
                    }
                }
            }

            pos = section_end;
        }

        Ok(Self {
            wal_file,
            previous_wal_file,
            lsn,
            timestamp,
            is_consistent,
            active_transactions,
            committed_transactions,
        })
    }

    /// Write checkpoint metadata to file atomically (section-based v2 format)
    ///
    /// Uses temp file + rename pattern to ensure crash safety.
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        use std::io::Write;

        let mut buf = Vec::new();

        // Build section data first to know sizes
        let mut sections: Vec<(u16, Vec<u8>)> = Vec::new();

        // Section 1: WAL file info
        {
            let mut section_data = Vec::new();
            // is_consistent (1 byte)
            section_data.push(if self.is_consistent { 1 } else { 0 });
            // Current WAL filename
            section_data.extend_from_slice(&(self.wal_file.len() as u16).to_le_bytes());
            section_data.extend_from_slice(self.wal_file.as_bytes());
            // Previous WAL filename
            if let Some(ref prev) = self.previous_wal_file {
                section_data.extend_from_slice(&(prev.len() as u16).to_le_bytes());
                section_data.extend_from_slice(prev.as_bytes());
            } else {
                section_data.extend_from_slice(&0u16.to_le_bytes());
            }
            sections.push((CheckpointSectionType::WalFileInfo as u16, section_data));
        }

        // Section 2: Active transactions
        {
            let mut section_data = Vec::new();
            section_data.extend_from_slice(&(self.active_transactions.len() as u64).to_le_bytes());
            for txn_id in &self.active_transactions {
                section_data.extend_from_slice(&txn_id.to_le_bytes());
            }
            sections.push((
                CheckpointSectionType::ActiveTransactions as u16,
                section_data,
            ));
        }

        // Section 3: Committed transactions (for two-phase recovery)
        if !self.committed_transactions.is_empty() {
            let mut section_data = Vec::new();
            section_data
                .extend_from_slice(&(self.committed_transactions.len() as u64).to_le_bytes());
            for info in &self.committed_transactions {
                section_data.extend_from_slice(&info.txn_id.to_le_bytes());
                section_data.extend_from_slice(&info.commit_lsn.to_le_bytes());
            }
            sections.push((
                CheckpointSectionType::CommittedTransactions as u16,
                section_data,
            ));
        }

        // Write 32-byte header
        buf.extend_from_slice(&CHECKPOINT_MAGIC.to_le_bytes()); // Magic (4)
        buf.push(CHECKPOINT_FORMAT_VERSION); // Version (1)
        buf.push(0); // Flags (1)
        buf.extend_from_slice(&CHECKPOINT_HEADER_SIZE.to_le_bytes()); // Header size (2)
        buf.extend_from_slice(&self.lsn.to_le_bytes()); // LSN (8)
        buf.extend_from_slice(&self.timestamp.to_le_bytes()); // Timestamp (8)
        buf.extend_from_slice(&(sections.len() as u16).to_le_bytes()); // Section count (2)
        buf.extend_from_slice(&[0u8; 6]); // Reserved (6)

        // Write section headers (8 bytes each)
        for (section_type, data) in &sections {
            buf.extend_from_slice(&section_type.to_le_bytes()); // Type (2)
            buf.extend_from_slice(&0u16.to_le_bytes()); // Flags (2)
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes()); // Size (4)
        }

        // Write section data
        for (_, data) in &sections {
            buf.extend_from_slice(data);
        }

        // CRC32 of everything
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        // Write atomically using temp file + rename
        let temp_path = path.with_extension("meta.tmp");

        let mut file = File::create(&temp_path).map_err(|e| {
            Error::internal(format!("failed to create checkpoint temp file: {}", e))
        })?;

        file.write_all(&buf)
            .map_err(|e| Error::internal(format!("failed to write checkpoint: {}", e)))?;

        file.sync_all()
            .map_err(|e| Error::internal(format!("failed to sync checkpoint: {}", e)))?;

        // Atomic rename
        fs::rename(&temp_path, path)
            .map_err(|e| Error::internal(format!("failed to rename checkpoint: {}", e)))?;

        // Sync directory to ensure rename is durable
        if let Some(parent) = path.parent() {
            if let Ok(dir_file) = File::open(parent) {
                let _ = dir_file.sync_all();
            }
        }

        Ok(())
    }
}

/// Write-Ahead Log Manager
pub struct WALManager {
    /// Base path for WAL files
    path: PathBuf,
    /// Current WAL file
    wal_file: Mutex<Option<File>>,
    /// Current WAL file name
    current_wal_file: Mutex<String>,
    /// Current Log Sequence Number
    current_lsn: AtomicU64,
    /// Previous LSN for entry chaining (enables backward traversal)
    previous_lsn: AtomicU64,
    /// Write buffer
    buffer: Mutex<Vec<u8>>,
    /// Flush trigger size
    flush_trigger: u64,
    /// Maximum WAL file size
    max_wal_size: u64,
    /// Last checkpoint LSN
    last_checkpoint: AtomicU64,
    /// Sync mode
    sync_mode: SyncMode,
    /// Running flag
    running: AtomicBool,
    /// Pending commits (legacy, kept for API compatibility)
    #[allow(dead_code)]
    pending_commits: AtomicI32,
    /// Last sync time in nanoseconds (legacy, kept for API compatibility)
    #[allow(dead_code)]
    last_sync_time: AtomicI64,
    /// Commit batch size (legacy, kept for API compatibility)
    /// Note: Batching is disabled in Normal mode to ensure durability.
    /// Commits are now always immediately synced to disk.
    #[allow(dead_code)]
    commit_batch_size: i32,
    /// Sync interval in nanoseconds (legacy, kept for API compatibility)
    #[allow(dead_code)]
    sync_interval: i64,
    /// Current file position (for rotation check)
    current_file_position: AtomicU64,
    /// WAL file sequence number (for rotation)
    wal_sequence: AtomicU64,
    /// Count of in-flight writes (entries taken from buffer but not yet written to disk)
    /// Used to prevent race condition during checkpoint where LSN is read but data isn't on disk yet
    in_flight_writes: AtomicU64,
}

impl WALManager {
    /// Create a new WAL manager with default config
    pub fn new(path: impl AsRef<Path>, sync_mode: SyncMode) -> Result<Self> {
        Self::with_config(path, sync_mode, None)
    }

    /// Recover from any interrupted WAL truncation operations
    ///
    /// This is called during WAL manager initialization to handle crash scenarios:
    /// 1. If .bak file exists without corresponding .log file, restore it
    /// 2. If temp files exist, clean them up
    /// 3. If both .bak and new .log exist, the truncation completed - delete .bak
    ///
    /// This ensures no data is lost if a crash occurs during truncation.
    fn recover_interrupted_truncation(wal_dir: &Path) -> Result<()> {
        if !wal_dir.exists() {
            return Ok(());
        }

        let entries = match fs::read_dir(wal_dir) {
            Ok(e) => e,
            Err(_) => return Ok(()), // Directory might not exist yet
        };

        // Collect all files first to avoid iterator invalidation
        let mut backup_files = Vec::new();
        let mut temp_files = Vec::new();
        let mut wal_files = Vec::new();

        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            let path = entry.path();

            if name.ends_with(".log.bak") {
                backup_files.push((name, path));
            } else if name.starts_with("wal-temp-") && name.ends_with(".log") {
                temp_files.push(path);
            } else if name.starts_with("wal-") && name.ends_with(".log") {
                wal_files.push(name);
            }
        }

        // Clean up temp files - they represent incomplete truncations
        for temp_path in temp_files {
            eprintln!(
                "Warning: Removing incomplete truncation temp file: {:?}",
                temp_path
            );
            let _ = fs::remove_file(&temp_path);
        }

        // Process backup files
        for (backup_name, backup_path) in backup_files {
            // Get the original WAL filename (remove .bak suffix)
            let original_name = backup_name.trim_end_matches(".bak");

            // Check if we have any valid WAL files
            let has_valid_wal = wal_files.iter().any(|f| !f.is_empty());

            if !has_valid_wal {
                // No valid WAL files exist - restore from backup
                // This means crash happened after backup but before new file was ready
                let restore_path = wal_dir.join(original_name);
                eprintln!(
                    "Warning: Recovering WAL from backup {:?} -> {:?}",
                    backup_path, restore_path
                );

                if let Err(e) = fs::rename(&backup_path, &restore_path) {
                    return Err(Error::internal(format!(
                        "CRITICAL: Failed to restore WAL from backup {:?}: {}. \
                         Manual intervention required to prevent data loss.",
                        backup_path, e
                    )));
                }

                eprintln!("WAL backup recovery successful");
            } else {
                // Valid WAL file(s) exist - truncation completed successfully
                // The backup is no longer needed, safe to delete
                eprintln!(
                    "Info: Cleaning up stale backup file {:?} (truncation completed)",
                    backup_path
                );
                let _ = fs::remove_file(&backup_path);
            }
        }

        Ok(())
    }

    /// Create a new WAL manager with custom config
    ///
    /// This allows configuring:
    /// - `commit_batch_size`: Number of commits to batch before syncing (SyncNormal mode)
    /// - `sync_interval_ms`: Minimum time between syncs in milliseconds (SyncNormal mode)
    /// - `wal_flush_trigger`: Buffer size that triggers a flush
    /// - `wal_buffer_size`: Initial buffer size
    /// - `wal_max_size`: Maximum WAL file size before rotation
    pub fn with_config(
        path: impl AsRef<Path>,
        sync_mode: SyncMode,
        config: Option<&PersistenceConfig>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create WAL directory if it doesn't exist
        fs::create_dir_all(&path)
            .map_err(|e| Error::internal(format!("failed to create WAL directory: {}", e)))?;

        // CRITICAL: Recover from any interrupted truncation before proceeding
        // This ensures data integrity if a crash happened during WAL truncation
        Self::recover_interrupted_truncation(&path)?;

        let mut wal_file: Option<File> = None;
        let mut initial_lsn: u64 = 0;
        let mut wal_filename = String::new();

        // Check if checkpoint exists
        let checkpoint_path = path.join("checkpoint.meta");
        if let Ok(checkpoint) = CheckpointMetadata::read_from_file(&checkpoint_path) {
            if !checkpoint.wal_file.is_empty() {
                wal_filename = checkpoint.wal_file.clone();
                initial_lsn = checkpoint.lsn;

                let wal_path = path.join(&checkpoint.wal_file);
                if let Ok(file) = OpenOptions::new().read(true).append(true).open(&wal_path) {
                    wal_file = Some(file);
                }
            }
        }

        // If no checkpoint or couldn't open WAL file, look for existing WAL files
        if wal_file.is_none() {
            let mut wal_files: Vec<String> = Vec::new();

            if let Ok(entries) = fs::read_dir(&path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("wal-") && name.ends_with(".log") {
                        wal_files.push(name);
                    }
                }
            }

            wal_files.sort();

            if let Some(newest) = wal_files.last() {
                wal_filename = newest.clone();
                let wal_path = path.join(newest);

                // Try to extract LSN from filename
                if let Some(lsn_start) = wal_filename.find("lsn-") {
                    if let Some(lsn_end) = wal_filename[lsn_start + 4..].find('.') {
                        if let Ok(lsn) =
                            wal_filename[lsn_start + 4..lsn_start + 4 + lsn_end].parse::<u64>()
                        {
                            initial_lsn = lsn;
                        }
                    }
                }

                if let Ok(file) = OpenOptions::new().read(true).append(true).open(&wal_path) {
                    // Find last LSN in file
                    if let Ok(last_lsn) = find_last_lsn(&path.join(newest)) {
                        if last_lsn > initial_lsn {
                            initial_lsn = last_lsn;
                        }
                    }
                    wal_file = Some(file);
                }
            }
        }

        // Create new WAL file if none exists
        if wal_file.is_none() {
            let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
            wal_filename = format!("wal-{}-lsn-0.log", timestamp);
            let wal_path = path.join(&wal_filename);

            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&wal_path)
                .map_err(|e| Error::internal(format!("failed to create WAL file: {}", e)))?;

            wal_file = Some(file);
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        // Extract config values with defaults
        let (commit_batch_size, sync_interval, flush_trigger, buffer_size, max_wal_size) =
            if let Some(cfg) = config {
                (
                    cfg.commit_batch_size as i32,
                    (cfg.sync_interval_ms as i64) * 1_000_000, // ms to ns
                    cfg.wal_flush_trigger as u64,
                    cfg.wal_buffer_size,
                    cfg.wal_max_size as u64,
                )
            } else {
                (
                    100,        // Default batch size
                    10_000_000, // 10ms in nanoseconds
                    DEFAULT_WAL_FLUSH_TRIGGER,
                    DEFAULT_WAL_BUFFER_SIZE,
                    DEFAULT_WAL_MAX_SIZE,
                )
            };

        // Get initial file position if we have an existing WAL file
        let initial_file_position = if let Some(ref file) = wal_file {
            file.metadata().map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        // Extract sequence number from filename (e.g., "wal_00000001.log" -> 1)
        let initial_sequence = Self::extract_sequence_from_filename(&wal_filename).unwrap_or(0);

        Ok(Self {
            path,
            wal_file: Mutex::new(wal_file),
            current_wal_file: Mutex::new(wal_filename),
            current_lsn: AtomicU64::new(initial_lsn),
            previous_lsn: AtomicU64::new(initial_lsn),
            buffer: Mutex::new(Vec::with_capacity(buffer_size)),
            flush_trigger,
            max_wal_size,
            last_checkpoint: AtomicU64::new(initial_lsn),
            sync_mode,
            running: AtomicBool::new(true),
            pending_commits: AtomicI32::new(0),
            last_sync_time: AtomicI64::new(now),
            commit_batch_size,
            sync_interval,
            current_file_position: AtomicU64::new(initial_file_position),
            wal_sequence: AtomicU64::new(initial_sequence),
            in_flight_writes: AtomicU64::new(0),
        })
    }

    /// Extract sequence number from WAL filename
    fn extract_sequence_from_filename(filename: &str) -> Option<u64> {
        // Try new format: wal_00000001.log
        if filename.starts_with("wal_") {
            if let Some(dot_pos) = filename.find('.') {
                return filename[4..dot_pos].parse().ok();
            }
        }
        // Try old format: wal-YYYYMMDD-HHMMSS-lsn-N.log (sequence is implicit from LSN)
        if filename.starts_with("wal-") {
            if let Some(lsn_pos) = filename.find("lsn-") {
                if let Some(dot_pos) = filename[lsn_pos..].find('.') {
                    // Use LSN as a proxy for sequence
                    return filename[lsn_pos + 4..lsn_pos + dot_pos].parse().ok();
                }
            }
        }
        None
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Acquire)
    }

    /// Append a WAL entry
    pub fn append_entry(&self, mut entry: WALEntry) -> Result<u64> {
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        // Get previous LSN and assign new LSN atomically
        let prev_lsn = self.previous_lsn.load(Ordering::Acquire);
        entry.previous_lsn = prev_lsn;

        // Check for LSN overflow before incrementing
        // u64::MAX is ~18 quintillion, practically unreachable, but check for safety
        let current = self.current_lsn.load(Ordering::Acquire);
        if current == u64::MAX {
            return Err(Error::internal(
                "WAL LSN overflow: maximum sequence number reached. Database requires maintenance.",
            ));
        }
        entry.lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst) + 1;

        // Update previous_lsn for next entry's chain link
        self.previous_lsn.store(entry.lsn, Ordering::Release);

        // Encode entry with new V2 format
        let encoded = entry.encode();
        let encoded_len = encoded.len() as u64;

        // Write to buffer
        {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.extend_from_slice(&encoded);

            let needs_flush = buffer.len() >= self.flush_trigger as usize;
            let force_flush = self.sync_mode == SyncMode::Full
                || (self.sync_mode == SyncMode::Normal
                    && (entry.operation.is_transaction_end() || entry.operation.is_ddl()));

            if needs_flush || force_flush {
                let buffer_data = std::mem::take(&mut *buffer);
                // CRITICAL: Increment in-flight counter BEFORE releasing lock
                // This prevents checkpoint from reading LSN before data is on disk
                self.in_flight_writes.fetch_add(1, Ordering::SeqCst);
                drop(buffer); // Release buffer lock before file operations

                // Use a guard pattern to ensure we decrement even on error
                let write_result = self.write_to_file(&buffer_data);
                self.in_flight_writes.fetch_sub(1, Ordering::SeqCst);
                write_result?;

                // Update file position tracking
                self.current_file_position
                    .fetch_add(buffer_data.len() as u64, Ordering::Relaxed);

                if self.should_sync(entry.operation) {
                    self.sync_locked()?;
                }
            }
        }

        // Track that we wrote encoded_len bytes (even if buffered)
        // This is approximate but sufficient for rotation decision
        let _ = encoded_len;

        Ok(entry.lsn)
    }

    /// Get previous LSN (last written entry's LSN)
    pub fn previous_lsn(&self) -> u64 {
        self.previous_lsn.load(Ordering::Acquire)
    }

    /// Write a commit marker for two-phase recovery
    pub fn write_commit_marker(&self, txn_id: i64) -> Result<u64> {
        let entry = WALEntry::commit_marker(txn_id);
        self.append_entry(entry)
    }

    /// Write an abort marker for two-phase recovery
    pub fn write_abort_marker(&self, txn_id: i64) -> Result<u64> {
        let entry = WALEntry::abort_marker(txn_id);
        self.append_entry(entry)
    }

    /// Write data to WAL file
    fn write_to_file(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let mut wal_file = self.wal_file.lock().unwrap();
        if let Some(file) = wal_file.as_mut() {
            file.write_all(data)
                .map_err(|e| Error::internal(format!("failed to write to WAL: {}", e)))?;
        } else {
            return Err(Error::WalFileClosed);
        }

        Ok(())
    }

    /// Sync WAL to disk (assumes lock is held)
    fn sync_locked(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        let wal_file = self.wal_file.lock().unwrap();
        if let Some(file) = wal_file.as_ref() {
            file.sync_all()
                .map_err(|e| Error::internal(format!("failed to sync WAL: {}", e)))?;
        }

        Ok(())
    }

    /// Check if WAL file should be rotated based on size
    ///
    /// Returns true if rotation occurred
    pub fn maybe_rotate(&self) -> Result<bool> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(false);
        }

        let current_size = self.current_file_position.load(Ordering::Relaxed);
        if current_size < self.max_wal_size {
            return Ok(false);
        }

        // Flush and sync before rotation
        self.flush()?;
        self.sync_locked()?;

        // Perform rotation
        self.rotate_wal()?;

        Ok(true)
    }

    /// Rotate WAL to a new file
    ///
    /// This:
    /// 1. Syncs and closes the current WAL file
    /// 2. Creates a new WAL file with incremented sequence number
    /// 3. Updates the checkpoint metadata with the new WAL reference
    fn rotate_wal(&self) -> Result<()> {
        let current_lsn = self.current_lsn.load(Ordering::Acquire);
        let new_sequence = self.wal_sequence.fetch_add(1, Ordering::SeqCst) + 1;

        // Generate new filename with sequence number and LSN
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
        let new_filename = format!(
            "wal_{:08}-{}-lsn-{}.log",
            new_sequence, timestamp, current_lsn
        );
        let new_path = self.path.join(&new_filename);

        // Create new WAL file
        let new_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&new_path)
            .map_err(|e| Error::internal(format!("failed to create rotated WAL file: {}", e)))?;

        // Update current WAL file references
        {
            let old_filename = {
                let mut wal_file = self.wal_file.lock().unwrap();
                let mut current_filename = self.current_wal_file.lock().unwrap();

                // Get old filename for checkpoint update
                let old_filename = current_filename.clone();

                // Replace file handle
                *wal_file = Some(new_file);
                *current_filename = new_filename.clone();

                old_filename
            };

            // Reset file position counter
            self.current_file_position.store(0, Ordering::Release);

            // Update checkpoint with new WAL file reference and previous WAL
            // IMPORTANT: Preserve existing checkpoint LSN (which represents snapshot point)
            // Only update the WAL file references during rotation
            let checkpoint_path = self.path.join("checkpoint.meta");
            let existing_lsn = match CheckpointMetadata::read_from_file(&checkpoint_path) {
                Ok(c) => c.lsn,
                Err(e) => {
                    // Warning: checkpoint read failed during rotation, will use LSN 0
                    // This means a full WAL replay will be needed on next recovery
                    eprintln!(
                        "Warning: Failed to read existing checkpoint during WAL rotation: {}. \
                         Recovery will replay full WAL.",
                        e
                    );
                    0 // If no checkpoint exists, LSN should be 0 for full recovery
                }
            };

            let checkpoint = CheckpointMetadata {
                wal_file: new_filename,
                previous_wal_file: Some(old_filename),
                lsn: existing_lsn, // Preserve existing LSN, don't update to current_lsn
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0),
                is_consistent: true,
                active_transactions: vec![],
                committed_transactions: vec![],
            };
            checkpoint.write_to_file(&checkpoint_path)?;
        }

        Ok(())
    }

    /// Get current WAL file size
    pub fn current_file_size(&self) -> u64 {
        self.current_file_position.load(Ordering::Relaxed)
    }

    /// Get maximum WAL file size
    pub fn max_file_size(&self) -> u64 {
        self.max_wal_size
    }

    /// Get current WAL sequence number
    pub fn current_sequence(&self) -> u64 {
        self.wal_sequence.load(Ordering::Relaxed)
    }

    /// Public sync method
    pub fn sync(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        // First flush buffer
        self.flush()?;

        // Then sync
        self.sync_locked()
    }

    /// Flush buffer to disk without syncing
    pub fn flush(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        let buffer_data = {
            let mut buffer = self.buffer.lock().unwrap();
            if buffer.is_empty() {
                return Ok(());
            }
            let data = std::mem::take(&mut *buffer);
            // CRITICAL: Increment in-flight counter BEFORE releasing lock
            // This prevents checkpoint from reading LSN before data is on disk
            self.in_flight_writes.fetch_add(1, Ordering::SeqCst);
            data
        };

        // Use a guard pattern to ensure we decrement even on error
        let write_result = self.write_to_file(&buffer_data);
        self.in_flight_writes.fetch_sub(1, Ordering::SeqCst);
        write_result
    }

    /// Check if we should sync based on operation type
    fn should_sync(&self, op: WALOperationType) -> bool {
        match self.sync_mode {
            SyncMode::None => false,
            SyncMode::Normal => {
                // CRITICAL: Always sync on DDL operations
                if op.is_ddl() {
                    return true;
                }
                // CRITICAL: Always sync on transaction end (commit/rollback)
                // This ensures durability - a committed transaction is guaranteed
                // to survive crashes. Without this, commit markers could be lost
                // in the OS buffer cache, causing committed data to appear as
                // "in-doubt" (aborted) after recovery.
                //
                // Note: This removes the previous batching optimization for commits.
                // If users need higher write throughput at the cost of durability,
                // they can use SyncMode::None explicitly.
                if op.is_transaction_end() {
                    return true;
                }
                false
            }
            SyncMode::Full => true,
        }
    }

    /// Two-phase WAL replay for crash recovery
    ///
    /// Phase 1 (Analysis): Scan all entries to identify committed/aborted transactions
    ///                     Only stores transaction IDs, not full entries (memory efficient)
    /// Phase 2 (REDO): Re-read WAL and apply only entries from committed transactions
    ///
    /// This ensures that after a crash, only committed transactions are visible.
    /// Uncommitted transactions (those without a COMMIT_MARKER) are discarded.
    ///
    /// Memory optimization: Uses streaming approach with two passes over WAL files
    /// instead of loading all entries into memory. Only txn_id HashSets are kept.
    pub fn replay_two_phase<F>(
        &self,
        from_lsn: u64,
        mut callback: F,
    ) -> Result<TwoPhaseRecoveryInfo>
    where
        F: FnMut(WALEntry) -> Result<()>,
    {
        // Flush buffer first
        self.flush()?;

        let mut from_lsn = from_lsn;

        // Check for checkpoint
        let checkpoint_path = self.path.join("checkpoint.meta");
        if let Ok(checkpoint) = CheckpointMetadata::read_from_file(&checkpoint_path) {
            if checkpoint.lsn > from_lsn {
                from_lsn = checkpoint.lsn;
            }
        }

        // Collect WAL files to replay
        let mut wal_files: Vec<PathBuf> = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name().to_string_lossy().to_string();
                if (name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log")
                {
                    wal_files.push(entry.path());
                }
            }
        }

        wal_files.sort();

        // =====================================================
        // Phase 1: Analysis - Identify transaction outcomes
        // Only collect txn_ids, not full entries (memory efficient)
        // =====================================================
        let mut committed_txns: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut aborted_txns: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut last_lsn = from_lsn;

        for wal_path in &wal_files {
            Self::scan_wal_for_txn_status(
                wal_path,
                from_lsn,
                &mut committed_txns,
                &mut aborted_txns,
                &mut last_lsn,
            )?;
        }

        // =====================================================
        // Phase 2: REDO - Re-read WAL and apply committed entries
        // Streaming approach: read and apply one entry at a time
        // =====================================================
        let mut applied_count = 0u64;
        let mut skipped_count = 0u64;

        for wal_path in &wal_files {
            let mut file = match File::open(wal_path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            loop {
                // Read 32-byte header
                let mut header_buf = [0u8; 32];
                match file.read_exact(&mut header_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break,
                }

                // Check magic marker
                let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
                if magic != WAL_ENTRY_MAGIC {
                    let _ = file.seek(SeekFrom::Current(-32));
                    if !Self::scan_for_magic(&mut file) {
                        break;
                    }
                    continue;
                }

                // Parse header
                let flags = WalFlags::from_byte(header_buf[5]);
                let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
                let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
                let previous_lsn = u64::from_le_bytes(header_buf[16..24].try_into().unwrap());
                let entry_size =
                    u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;

                // Skip any additional header bytes
                if header_size > 32 {
                    let extra = header_size - 32;
                    if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                        break;
                    }
                }

                // Sanity check on size
                let total_data_size = entry_size + 4;
                if entry_size > 64 * 1024 * 1024 {
                    if !Self::scan_for_magic(&mut file) {
                        break;
                    }
                    continue;
                }

                // Skip entries before from_lsn
                if lsn < from_lsn {
                    if file
                        .seek(SeekFrom::Current(total_data_size as i64))
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }

                // Read entry data + CRC
                let mut data = vec![0u8; total_data_size];
                match file.read_exact(&mut data) {
                    Ok(()) => {}
                    Err(_) => break,
                }

                // Decode entry
                match WALEntry::decode(lsn, previous_lsn, flags, &data) {
                    Ok(entry) => {
                        // Skip rotation/snapshot markers (internal WAL management)
                        if entry.is_marker_entry() {
                            continue;
                        }

                        // Skip abort markers entirely - aborted transactions don't need processing
                        if entry.is_abort_marker() {
                            continue;
                        }

                        // For commit markers: pass to callback so registry can be updated
                        // This is crucial for visibility - without this, committed data is invisible
                        if entry.is_commit_marker() {
                            if committed_txns.contains(&entry.txn_id) {
                                callback(entry)?;
                            }
                            continue;
                        }

                        // Apply only committed transactions' data entries
                        if committed_txns.contains(&entry.txn_id) {
                            callback(entry)?;
                            applied_count += 1;
                        } else {
                            // Transaction is aborted or in-doubt (no commit marker)
                            // Treat in-doubt as aborted for safety
                            skipped_count += 1;
                        }
                    }
                    Err(e) => {
                        // Log decode errors (including CRC failures) during recovery
                        // These could indicate WAL corruption or incomplete writes
                        eprintln!(
                            "Warning: WAL entry decode failed at LSN {}: {} (entry skipped)",
                            lsn, e
                        );
                        skipped_count += 1;
                        continue;
                    }
                }
            }
        }

        // Update current LSN if we replayed entries
        if last_lsn > self.current_lsn.load(Ordering::Acquire) {
            self.current_lsn.store(last_lsn, Ordering::Release);
        }

        Ok(TwoPhaseRecoveryInfo {
            last_lsn,
            committed_transactions: committed_txns.len(),
            aborted_transactions: aborted_txns.len(),
            applied_entries: applied_count,
            skipped_entries: skipped_count,
        })
    }

    /// Phase 1 helper: Scan a WAL file for transaction commit/abort status
    ///
    /// This function only extracts transaction IDs and their commit/abort markers,
    /// without storing full entry data. This keeps memory usage minimal during
    /// the analysis phase of two-phase recovery.
    fn scan_wal_for_txn_status(
        wal_path: &Path,
        from_lsn: u64,
        committed_txns: &mut std::collections::HashSet<i64>,
        aborted_txns: &mut std::collections::HashSet<i64>,
        last_lsn: &mut u64,
    ) -> Result<()> {
        let mut file = match File::open(wal_path) {
            Ok(f) => f,
            Err(_) => return Ok(()), // Skip files we can't open
        };

        loop {
            // Read 32-byte header
            let mut header_buf = [0u8; 32];
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(_) => break,
            }

            // Check magic marker
            let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
            if magic != WAL_ENTRY_MAGIC {
                let _ = file.seek(SeekFrom::Current(-32));
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }

            // Parse header - only need flags, header_size, lsn, entry_size
            let flags = WalFlags::from_byte(header_buf[5]);
            let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
            let entry_size = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;

            // Skip any additional header bytes
            if header_size > 32 {
                let extra = header_size - 32;
                if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                    break;
                }
            }

            // Sanity check on size
            let total_data_size = entry_size + 4;
            if entry_size > 64 * 1024 * 1024 {
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }

            // Skip entries before from_lsn
            if lsn < from_lsn {
                if file
                    .seek(SeekFrom::Current(total_data_size as i64))
                    .is_err()
                {
                    break;
                }
                continue;
            }

            // For commit/abort markers, we can identify them from flags without full decode
            // This is the fast path - only read txn_id (first 8 bytes of data)
            if flags.contains(WalFlags::COMMIT_MARKER) || flags.contains(WalFlags::ABORT_MARKER) {
                // Read just the txn_id (first 8 bytes of data portion)
                let mut txn_id_buf = [0u8; 8];
                match file.read_exact(&mut txn_id_buf) {
                    Ok(()) => {
                        let txn_id = i64::from_le_bytes(txn_id_buf);
                        if flags.contains(WalFlags::COMMIT_MARKER) {
                            committed_txns.insert(txn_id);
                        } else {
                            aborted_txns.insert(txn_id);
                        }
                        // Skip rest of entry (entry_size - 8 + CRC 4)
                        let remaining = total_data_size - 8;
                        if file.seek(SeekFrom::Current(remaining as i64)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            } else {
                // Not a commit/abort marker, skip the entire entry
                if file
                    .seek(SeekFrom::Current(total_data_size as i64))
                    .is_err()
                {
                    break;
                }
            }

            // Track last LSN
            if lsn > *last_lsn {
                *last_lsn = lsn;
            }
        }

        Ok(())
    }

    /// Scan forward in the file looking for the next valid magic marker
    ///
    /// The magic marker is stored in little-endian format on disk, so we build
    /// the window by shifting right and inserting new bytes at the high position.
    ///
    /// Uses buffered reads (8KB chunks) for efficiency instead of byte-by-byte syscalls.
    fn scan_for_magic(file: &mut File) -> bool {
        const BUFFER_SIZE: usize = 8192; // 8KB buffer for efficient I/O
        let mut buffer = [0u8; BUFFER_SIZE];
        let mut window: u32 = 0;
        let mut total_scanned: usize = 0;
        const MAX_SCAN: usize = 1024 * 1024; // 1MB limit

        loop {
            // Read a chunk into buffer
            let bytes_read = match file.read(&mut buffer) {
                Ok(0) => return false, // EOF
                Ok(n) => n,
                Err(_) => return false,
            };

            // Scan through the buffer
            for (i, &byte) in buffer[..bytes_read].iter().enumerate() {
                // Build little-endian u32: new byte goes to high position,
                // existing bytes shift down. After reading 4 bytes [b0,b1,b2,b3],
                // window = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
                // which matches how u32::from_le_bytes works.
                window = (window >> 8) | ((byte as u32) << 24);

                total_scanned += 1;
                if total_scanned > MAX_SCAN {
                    return false;
                }

                if window == WAL_ENTRY_MAGIC {
                    // Found magic. Calculate how far back to seek:
                    // We're at position i+1 in the current buffer read
                    // The magic marker started 4 bytes ago
                    // We need to seek back (bytes_read - i - 1) to end of buffer,
                    // plus 4 for the magic marker itself, minus 1 because i is 0-indexed
                    let seek_back = (bytes_read - i - 1 + 4) as i64;
                    if file.seek(SeekFrom::Current(-seek_back)).is_ok() {
                        return true;
                    }
                    return false;
                }
            }
        }
    }

    /// Wait for any in-flight writes to complete with a timeout
    ///
    /// This is critical for checkpoint and truncation safety. The race condition occurs when:
    /// 1. Thread A takes buffer data, releases buffer lock, but hasn't written to disk yet
    /// 2. Checkpoint thread calls flush() which sees empty buffer and returns
    /// 3. Checkpoint reads current_lsn (which includes Thread A's LSN)
    /// 4. Checkpoint/truncation uses that LSN, potentially losing Thread A's data
    ///
    /// By waiting for in_flight_writes to be 0, we ensure all data is on disk
    /// before reading the LSN for checkpoint purposes.
    ///
    /// Returns Ok(()) if all writes completed, Err if timeout was reached.
    /// Default timeout is 30 seconds which should be more than enough for any
    /// reasonable write operation. If timeout is reached, it indicates a serious
    /// problem (hung thread, deadlock, etc.)
    fn wait_for_in_flight_writes(&self) -> Result<()> {
        self.wait_for_in_flight_writes_timeout(std::time::Duration::from_secs(30))
    }

    /// Wait for any in-flight writes to complete with a custom timeout
    ///
    /// Uses exponential backoff to avoid busy-waiting while still being responsive.
    fn wait_for_in_flight_writes_timeout(&self, timeout: std::time::Duration) -> Result<()> {
        use std::time::Instant;

        let deadline = Instant::now() + timeout;
        let mut sleep_duration = std::time::Duration::from_micros(10);
        const MAX_SLEEP: std::time::Duration = std::time::Duration::from_millis(10);

        while self.in_flight_writes.load(Ordering::SeqCst) > 0 {
            if Instant::now() > deadline {
                return Err(Error::internal(format!(
                    "timeout waiting for in-flight WAL writes to complete ({} still pending)",
                    self.in_flight_writes.load(Ordering::SeqCst)
                )));
            }

            // Exponential backoff with cap
            std::thread::sleep(sleep_duration);
            sleep_duration = std::cmp::min(sleep_duration * 2, MAX_SLEEP);
        }

        Ok(())
    }

    /// Create a checkpoint and return the LSN at the checkpoint point
    ///
    /// Returns the LSN that represents the checkpoint point. All data up to
    /// this LSN is guaranteed to be durably written to disk when this returns.
    /// This LSN should be used for snapshot creation to ensure consistency.
    pub fn create_checkpoint(&self, active_transactions: Vec<i64>) -> Result<u64> {
        // CRITICAL: Wait for any in-flight writes before flushing
        // This prevents the race condition where we read current_lsn before
        // all writes at that LSN are actually on disk
        self.wait_for_in_flight_writes()?;

        // Flush and sync
        self.flush()?;
        self.sync_locked()?;

        // Wait again after flush to catch any writes that started during flush
        self.wait_for_in_flight_writes()?;

        // CRITICAL: Capture the LSN atomically after all syncs complete
        // This LSN is the checkpoint point - all data up to this LSN is now on disk
        let checkpoint_lsn = self.current_lsn.load(Ordering::Acquire);
        let wal_file = self.current_wal_file.lock().unwrap().clone();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let checkpoint = CheckpointMetadata {
            wal_file,
            previous_wal_file: None, // Will be set during WAL rotation
            lsn: checkpoint_lsn,
            timestamp: now,
            is_consistent: active_transactions.is_empty(),
            active_transactions,
            committed_transactions: vec![], // Will be populated during two-phase recovery
        };

        let checkpoint_path = self.path.join("checkpoint.meta");
        checkpoint.write_to_file(&checkpoint_path)?;

        self.last_checkpoint
            .store(checkpoint_lsn, Ordering::Release);

        Ok(checkpoint_lsn)
    }

    /// Close the WAL manager
    pub fn close(&self) -> Result<()> {
        // Check if already closed
        if !self.running.load(Ordering::Acquire) {
            return Ok(()); // Already closed
        }

        // Flush buffer (while still running)
        let _ = self.flush();

        // Sync to disk (while still running)
        let _ = self.sync_locked();

        // Now mark as not running
        self.running.store(false, Ordering::SeqCst);

        // Close file
        let mut wal_file = self.wal_file.lock().unwrap();
        *wal_file = None;

        Ok(())
    }

    /// Get the WAL directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get maximum WAL file size before rotation
    pub fn max_wal_size(&self) -> u64 {
        self.max_wal_size
    }

    /// Get last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> u64 {
        self.last_checkpoint.load(Ordering::Acquire)
    }

    /// Get current WAL file name
    pub fn current_wal_file(&self) -> String {
        self.current_wal_file.lock().unwrap().clone()
    }

    /// Truncate the WAL file to remove entries up to the given LSN
    ///
    /// This is used after a successful checkpoint/snapshot to reclaim disk space.
    /// Only entries with LSN > up_to_lsn are kept.
    pub fn truncate_wal(&self, up_to_lsn: u64) -> Result<()> {
        // Skip if not running or if up_to_lsn is zero (no valid checkpoint)
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        if up_to_lsn == 0 {
            return Err(Error::internal(format!(
                "invalid LSN for WAL truncation: {}",
                up_to_lsn
            )));
        }

        // CRITICAL: Wait for any in-flight writes to complete before truncation
        // This prevents the race condition where:
        // 1. Thread A takes buffer data, releases buffer lock, but hasn't written to disk yet
        // 2. truncate_wal() proceeds with truncation
        // 3. Thread A's data targets the old file and gets lost
        self.wait_for_in_flight_writes()?;

        // Lock the WAL file for the entire operation
        let mut wal_file_guard = self.wal_file.lock().unwrap();
        let mut current_wal_name = self.current_wal_file.lock().unwrap();

        // Verify we're still running and have a file
        if !self.running.load(Ordering::Acquire) || wal_file_guard.is_none() {
            return Err(Error::internal(
                "WAL manager is not running or file is closed",
            ));
        }

        // Extract LSN from current WAL filename to check if truncation is needed
        // If upToLSN <= currentFileLSN, there's nothing to truncate
        if let Some(lsn_start) = current_wal_name.find("lsn-") {
            if let Some(lsn_end) = current_wal_name[lsn_start + 4..].find('.') {
                if let Ok(current_file_lsn) =
                    current_wal_name[lsn_start + 4..lsn_start + 4 + lsn_end].parse::<u64>()
                {
                    if up_to_lsn <= current_file_lsn {
                        // All entries in this file are already newer than up_to_lsn
                        return Ok(());
                    }
                }
            }
        }

        // First, flush any pending data to make sure everything is on disk
        {
            let mut buffer = self.buffer.lock().unwrap();
            if !buffer.is_empty() {
                let buffer_data = std::mem::take(&mut *buffer);
                if let Some(file) = wal_file_guard.as_mut() {
                    file.write_all(&buffer_data).map_err(|e| {
                        Error::internal(format!("failed to flush buffer during truncation: {}", e))
                    })?;
                }
            }
        }

        // Sync file to ensure all data is persisted
        if let Some(file) = wal_file_guard.as_ref() {
            file.sync_all().map_err(|e| {
                Error::internal(format!("failed to sync WAL during truncation: {}", e))
            })?;
        }

        // Create a new file for the truncated WAL with LSN-based naming
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
        let new_wal_filename = format!("wal-{}-lsn-{}.log", timestamp, up_to_lsn);
        let temp_wal_path = self.path.join(format!(
            "wal-temp-{}.log",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        let mut temp_wal_file = File::create(&temp_wal_path)
            .map_err(|e| Error::internal(format!("failed to create temporary WAL file: {}", e)))?;

        // Reset the current WAL file position to beginning
        let wal_file_path = self.path.join(&*current_wal_name);
        if let Some(file) = wal_file_guard.as_mut() {
            file.seek(SeekFrom::Start(0))
                .map_err(|e| Error::internal(format!("failed to seek WAL file: {}", e)))?;
        }

        // Copy entries that are newer than up_to_lsn to the temp file
        // 32-byte header: magic(4) + version(1) + flags(1) + header_size(2) + LSN(8) + prev_lsn(8) + entry_size(4) + reserved(4)
        let mut header_buf = [0u8; 32];
        let mut entries_copied = 0u64;
        let mut last_copied_lsn: u64 = up_to_lsn; // Track last LSN for chain continuity
        let mut new_file_size: u64 = 0; // Track new file size for position update

        if let Some(file) = wal_file_guard.as_mut() {
            loop {
                // Try to read entry header (32 bytes)
                match file.read_exact(&mut header_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break,
                }

                // Verify magic marker
                let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
                if magic != WAL_ENTRY_MAGIC {
                    // Corrupted entry, skip
                    break;
                }

                // Parse header fields
                let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
                let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
                let entry_size =
                    u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;

                // Calculate total size after header (including any extra header bytes for future extensibility)
                let extra_header = header_size.saturating_sub(32);
                let total_entry_size = extra_header + entry_size + 4; // extra_header + data + CRC

                // If the entry's LSN is older than or equal to up_to_lsn, skip it
                if lsn <= up_to_lsn {
                    // Skip to next entry
                    if file
                        .seek(SeekFrom::Current(total_entry_size as i64))
                        .is_err()
                    {
                        break;
                    }
                } else {
                    // Write the header to the temp file
                    temp_wal_file.write_all(&header_buf).map_err(|e| {
                        Error::internal(format!("failed to write header to temp file: {}", e))
                    })?;

                    // Copy the rest of the entry (extra header + data + CRC)
                    let mut data = vec![0u8; total_entry_size];
                    file.read_exact(&mut data).map_err(|e| {
                        Error::internal(format!("failed to read entry data: {}", e))
                    })?;
                    temp_wal_file.write_all(&data).map_err(|e| {
                        Error::internal(format!("failed to write entry data to temp file: {}", e))
                    })?;

                    // Track the last copied LSN and accumulate file size
                    last_copied_lsn = lsn;
                    new_file_size += 32 + total_entry_size as u64;
                    entries_copied += 1;
                }
            }
        }

        // If we didn't copy any entries (all entries were old), create a marker entry
        // so the WAL file isn't empty and tracking continues correctly
        //
        // LSN CHAIN BREAK NOTE:
        // The marker entry's previous_lsn points to up_to_lsn which no longer exists
        // in the WAL (it was truncated). This is intentional and safe because:
        // 1. Recovery uses checkpoint metadata to determine the starting point
        // 2. The marker entry serves only to maintain LSN continuity for new entries
        // 3. The snapshot_lsn in checkpoint metadata tracks what was persisted
        // Use CHAIN_BREAK_MARKER (0) as previous_lsn to explicitly indicate this
        if entries_copied == 0 {
            // previous_lsn = 0 indicates a chain break point (truncation occurred)
            // This is more explicit than pointing to a non-existent LSN
            const CHAIN_BREAK_MARKER: u64 = 0;
            let marker_entry = WALEntry {
                lsn: up_to_lsn + 1,
                previous_lsn: CHAIN_BREAK_MARKER, // Explicit chain break marker
                flags: WalFlags::NONE,
                txn_id: MARKER_TXN_ID, // Special marker transaction
                table_name: String::new(),
                row_id: 0,
                operation: WALOperationType::Commit,
                data: Vec::new(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0),
            };
            let encoded = marker_entry.encode();
            temp_wal_file
                .write_all(&encoded)
                .map_err(|e| Error::internal(format!("failed to write marker entry: {}", e)))?;

            // Track marker's LSN and size for chain continuity
            last_copied_lsn = up_to_lsn + 1;
            new_file_size = encoded.len() as u64;
        }

        // Sync the temp file to ensure data is flushed to disk
        temp_wal_file
            .sync_all()
            .map_err(|e| Error::internal(format!("failed to sync temp WAL file: {}", e)))?;

        // ATOMIC WAL TRUNCATION STRATEGY:
        // 1. Sync temp file to disk
        // 2. Close current WAL file
        // 3. Rename old WAL to .bak (backup)
        // 4. Rename temp file to new WAL name
        // 5. Open new WAL file
        // 6. Delete .bak file (only after everything succeeded)
        // On error at any step: restore from .bak if needed

        // Close the current WAL file
        *wal_file_guard = None;

        // Close the temp file (drop it)
        drop(temp_wal_file);

        // Create paths for the operation
        let new_wal_path = self.path.join(&new_wal_filename);
        let backup_wal_path = wal_file_path.with_extension("log.bak");

        // Step 1: Rename old WAL file to .bak (atomic backup)
        if wal_file_path.exists() {
            if let Err(e) = fs::rename(&wal_file_path, &backup_wal_path) {
                // Recovery: reopen original file at the end
                if let Ok(file) = OpenOptions::new()
                    .read(true)
                    .append(true)
                    .open(&wal_file_path)
                {
                    *wal_file_guard = Some(file);
                }
                // Cleanup temp file
                let _ = fs::remove_file(&temp_wal_path);
                return Err(Error::internal(format!(
                    "failed to backup old WAL file: {}",
                    e
                )));
            }
        }

        // Step 2: Rename temp file to new WAL name
        if let Err(e) = fs::rename(&temp_wal_path, &new_wal_path) {
            // Recovery: restore from backup and reopen
            if backup_wal_path.exists() {
                let _ = fs::rename(&backup_wal_path, &wal_file_path);
            }
            if let Ok(file) = OpenOptions::new()
                .read(true)
                .append(true)
                .open(&wal_file_path)
            {
                *wal_file_guard = Some(file);
            }
            return Err(Error::internal(format!(
                "failed to rename temp file to new WAL file: {}",
                e
            )));
        }

        // Step 3: Update current WAL file information
        *current_wal_name = new_wal_filename;

        // Step 4: Open the new WAL file
        let new_file = match OpenOptions::new()
            .read(true)
            .append(true)
            .open(&new_wal_path)
        {
            Ok(f) => f,
            Err(e) => {
                // Critical error: try to restore from backup
                // This is a serious situation but we try our best
                if backup_wal_path.exists() && fs::rename(&backup_wal_path, &wal_file_path).is_ok()
                {
                    *current_wal_name = wal_file_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    if let Ok(file) = OpenOptions::new()
                        .read(true)
                        .append(true)
                        .open(&wal_file_path)
                    {
                        *wal_file_guard = Some(file);
                    }
                }
                return Err(Error::internal(format!(
                    "failed to reopen WAL file after truncation: {}",
                    e
                )));
            }
        };

        *wal_file_guard = Some(new_file);

        // Step 5: Sync directory to ensure renames are durable
        // This is critical on filesystems like ext4 where rename durability
        // requires directory sync. Without this, a crash after rename but
        // before natural sync could result in the old filename persisting.
        if let Ok(dir_file) = File::open(&self.path) {
            let _ = dir_file.sync_all();
        }

        // Step 6: Delete backup file (only after everything succeeded)
        // If this fails, it's just a warning - not critical
        if backup_wal_path.exists() {
            if let Err(e) = fs::remove_file(&backup_wal_path) {
                eprintln!(
                    "Warning: Could not remove backup WAL file {:?}: {}",
                    backup_wal_path, e
                );
            }
        }

        // Step 7: Update WAL manager state to maintain chain continuity
        // CRITICAL: Update previous_lsn to the last entry in the new WAL file
        // This ensures the next append_entry() will correctly chain to the last
        // entry we kept (or the marker entry if all were truncated).
        // Without this, the backward chain would be broken after truncation.
        self.previous_lsn.store(last_copied_lsn, Ordering::Release);

        // Update file position to reflect the new WAL file size
        self.current_file_position
            .store(new_file_size, Ordering::Release);

        Ok(())
    }
}

impl Drop for WALManager {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Find the last LSN in a WAL file (32-byte header format)
fn find_last_lsn(path: &Path) -> Result<u64> {
    let mut file =
        File::open(path).map_err(|e| Error::internal(format!("failed to open WAL file: {}", e)))?;

    let mut last_lsn: u64 = 0;
    // 32-byte header: magic(4) + version(1) + flags(1) + header_size(2) + LSN(8) + prev_lsn(8) + entry_size(4) + reserved(4)
    let mut header_buf = [0u8; 32];

    loop {
        match file.read_exact(&mut header_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(_) => break,
        }

        // Verify magic marker
        let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        if magic != WAL_ENTRY_MAGIC {
            break; // Corrupted or end of valid data
        }

        // Parse header fields
        let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
        let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
        let entry_size = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;

        if lsn > last_lsn {
            last_lsn = lsn;
        }

        // Skip any additional header bytes (for future extensibility)
        if header_size > 32 {
            let extra = header_size - 32;
            if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                break;
            }
        }

        // Skip to next entry (data + CRC)
        let total_entry_size = entry_size + 4;
        if file
            .seek(SeekFrom::Current(total_entry_size as i64))
            .is_err()
        {
            break;
        }
    }

    Ok(last_lsn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_operation_type() {
        assert_eq!(WALOperationType::from_u8(1), Some(WALOperationType::Insert));
        assert_eq!(WALOperationType::from_u8(4), Some(WALOperationType::Commit));
        assert_eq!(WALOperationType::from_u8(0), None);
        assert_eq!(
            WALOperationType::from_u8(11),
            Some(WALOperationType::CreateView)
        );
        assert_eq!(
            WALOperationType::from_u8(12),
            Some(WALOperationType::DropView)
        );
        assert_eq!(WALOperationType::from_u8(13), None); // Invalid value

        assert!(WALOperationType::CreateTable.is_ddl());
        assert!(WALOperationType::CreateView.is_ddl());
        assert!(WALOperationType::DropView.is_ddl());
        assert!(!WALOperationType::Insert.is_ddl());
        assert!(WALOperationType::Commit.is_transaction_end());
        assert!(!WALOperationType::Insert.is_transaction_end());
    }

    #[test]
    fn test_wal_entry_encode_decode() {
        let mut entry = WALEntry::new(
            123,
            "test_table".to_string(),
            456,
            WALOperationType::Insert,
            vec![1, 2, 3, 4],
        );
        entry.lsn = 42; // Set LSN for encoding
        entry.previous_lsn = 41; // Set previous LSN for chaining
        entry.flags = WalFlags::NONE;

        let encoded = entry.encode();
        assert!(!encoded.is_empty());

        // 32-byte header format: magic(4) + version(1) + flags(1) + header_size(2) +
        // LSN(8) + prev_lsn(8) + entry_size(4) + reserved(4) = 32 bytes
        // Verify header
        let magic = u32::from_le_bytes(encoded[0..4].try_into().unwrap());
        assert_eq!(magic, WAL_ENTRY_MAGIC);
        let version = encoded[4];
        assert_eq!(version, WAL_FORMAT_VERSION);
        let flags = WalFlags::from_byte(encoded[5]);
        assert_eq!(flags, WalFlags::NONE);
        let header_size = u16::from_le_bytes(encoded[6..8].try_into().unwrap());
        assert_eq!(header_size, WAL_HEADER_SIZE);
        let lsn = u64::from_le_bytes(encoded[8..16].try_into().unwrap());
        assert_eq!(lsn, 42);
        let previous_lsn = u64::from_le_bytes(encoded[16..24].try_into().unwrap());
        assert_eq!(previous_lsn, 41);

        // Data starts at offset 32, includes CRC at end
        let decoded =
            WALEntry::decode(entry.lsn, entry.previous_lsn, flags, &encoded[32..]).unwrap();

        assert_eq!(decoded.lsn, entry.lsn);
        assert_eq!(decoded.previous_lsn, entry.previous_lsn);
        assert_eq!(decoded.flags, entry.flags);
        assert_eq!(decoded.txn_id, 123);
        assert_eq!(decoded.table_name, "test_table");
        assert_eq!(decoded.row_id, 456);
        assert_eq!(decoded.operation, WALOperationType::Insert);
        assert_eq!(decoded.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_wal_entry_crc_validation() {
        let mut entry = WALEntry::new(
            1,
            "test".to_string(),
            100,
            WALOperationType::Insert,
            vec![1, 2, 3],
        );
        entry.lsn = 1;
        entry.previous_lsn = 0;
        entry.flags = WalFlags::NONE;

        let mut encoded = entry.encode();

        // Corrupt the data portion (after 32-byte header)
        if encoded.len() > 40 {
            encoded[40] ^= 0xFF; // Flip some bits in data portion
        }

        // Decode should fail due to CRC mismatch
        let result = WALEntry::decode(entry.lsn, entry.previous_lsn, entry.flags, &encoded[32..]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_wal_entry_magic_marker() {
        let mut entry = WALEntry::new(1, "test".to_string(), 1, WALOperationType::Insert, vec![]);
        entry.lsn = 1;

        let encoded = entry.encode();

        // Check magic marker at the beginning
        let magic = u32::from_le_bytes(encoded[0..4].try_into().unwrap());
        assert_eq!(magic, WAL_ENTRY_MAGIC);
    }

    #[test]
    fn test_marker_entry_detection() {
        let marker = WALEntry {
            lsn: 100,
            previous_lsn: 99,
            flags: WalFlags::NONE,
            txn_id: MARKER_TXN_ID,
            table_name: String::new(),
            row_id: 0,
            operation: WALOperationType::Commit,
            data: Vec::new(),
            timestamp: 0,
        };
        assert!(marker.is_marker_entry());

        let normal = WALEntry::new(1, "test".to_string(), 1, WALOperationType::Insert, vec![]);
        assert!(!normal.is_marker_entry());
    }

    #[test]
    fn test_wal_entry_commit_rollback() {
        let commit = WALEntry::commit(100);
        assert_eq!(commit.txn_id, 100);
        assert_eq!(commit.operation, WALOperationType::Commit);
        assert!(commit.table_name.is_empty());

        let rollback = WALEntry::rollback(200);
        assert_eq!(rollback.txn_id, 200);
        assert_eq!(rollback.operation, WALOperationType::Rollback);
    }

    #[test]
    fn test_wal_flags() {
        // Test flag operations
        let mut flags = WalFlags::NONE;
        assert_eq!(flags.as_byte(), 0);
        assert!(!flags.contains(WalFlags::COMMIT_MARKER));

        flags.set(WalFlags::COMMIT_MARKER);
        assert!(flags.contains(WalFlags::COMMIT_MARKER));
        assert!(!flags.contains(WalFlags::ABORT_MARKER));

        flags.set(WalFlags::COMPRESSED);
        assert!(flags.contains(WalFlags::COMMIT_MARKER));
        assert!(flags.contains(WalFlags::COMPRESSED));

        flags.clear(WalFlags::COMMIT_MARKER);
        assert!(!flags.contains(WalFlags::COMMIT_MARKER));
        assert!(flags.contains(WalFlags::COMPRESSED));

        // Test union
        let combined = WalFlags::COMMIT_MARKER.union(WalFlags::CHECKPOINT_MARKER);
        assert!(combined.contains(WalFlags::COMMIT_MARKER));
        assert!(combined.contains(WalFlags::CHECKPOINT_MARKER));
        assert!(!combined.contains(WalFlags::ABORT_MARKER));

        // Test from_byte
        let restored = WalFlags::from_byte(combined.as_byte());
        assert_eq!(restored, combined);
    }

    #[test]
    fn test_commit_abort_markers() {
        // Test commit marker
        let commit_marker = WALEntry::commit_marker(42);
        assert_eq!(commit_marker.txn_id, 42);
        assert!(commit_marker.is_commit_marker());
        assert!(!commit_marker.is_abort_marker());
        assert!(commit_marker.flags.contains(WalFlags::COMMIT_MARKER));

        // Test abort marker
        let abort_marker = WALEntry::abort_marker(43);
        assert_eq!(abort_marker.txn_id, 43);
        assert!(!abort_marker.is_commit_marker());
        assert!(abort_marker.is_abort_marker());
        assert!(abort_marker.flags.contains(WalFlags::ABORT_MARKER));

        // Test regular commit (without marker flag)
        let regular_commit = WALEntry::commit(44);
        assert!(regular_commit.is_commit_marker()); // Still recognized via operation type

        // Test regular rollback (without marker flag)
        let regular_rollback = WALEntry::rollback(45);
        assert!(regular_rollback.is_abort_marker()); // Still recognized via operation type
    }

    #[test]
    fn test_previous_lsn_chaining() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

        // Initial previous_lsn should be 0
        assert_eq!(wal.previous_lsn(), 0);

        // Add entries and verify chaining
        let entry1 = WALEntry::new(1, "test".to_string(), 1, WALOperationType::Insert, vec![1]);
        let lsn1 = wal.append_entry(entry1).unwrap();
        assert_eq!(lsn1, 1);
        assert_eq!(wal.previous_lsn(), 1);

        let entry2 = WALEntry::new(1, "test".to_string(), 2, WALOperationType::Insert, vec![2]);
        let lsn2 = wal.append_entry(entry2).unwrap();
        assert_eq!(lsn2, 2);
        assert_eq!(wal.previous_lsn(), 2);

        // Commit both transactions so they show up in two-phase replay
        wal.write_commit_marker(1).unwrap();

        // Verify entries have correct previous_lsn when replayed
        let mut entries = Vec::new();
        let mut commit_markers = Vec::new();
        wal.replay_two_phase(0, |entry| {
            if entry.is_commit_marker() {
                commit_markers.push((entry.lsn, entry.previous_lsn));
            } else {
                entries.push((entry.lsn, entry.previous_lsn));
            }
            Ok(())
        })
        .unwrap();

        // Two data entries
        assert_eq!(entries.len(), 2);
        // First entry's previous_lsn is 0 (initial)
        assert_eq!(entries[0], (1, 0));
        // Second entry's previous_lsn is 1 (links to first)
        assert_eq!(entries[1], (2, 1));
        // Commit marker is also passed to callback
        assert_eq!(commit_markers.len(), 1);

        wal.close().unwrap();
    }

    #[test]
    fn test_checkpoint_metadata() {
        let dir = tempdir().unwrap();
        let checkpoint_path = dir.path().join("checkpoint.meta");

        let checkpoint = CheckpointMetadata {
            wal_file: "wal-test.log".to_string(),
            previous_wal_file: Some("wal-prev.log".to_string()),
            lsn: 12345,
            timestamp: 1234567890,
            is_consistent: true,
            active_transactions: vec![1, 2, 3],
            committed_transactions: vec![
                CommittedTxnInfo {
                    txn_id: 10,
                    commit_lsn: 100,
                },
                CommittedTxnInfo {
                    txn_id: 20,
                    commit_lsn: 200,
                },
            ],
        };

        checkpoint.write_to_file(&checkpoint_path).unwrap();

        let loaded = CheckpointMetadata::read_from_file(&checkpoint_path).unwrap();
        assert_eq!(loaded.wal_file, "wal-test.log");
        assert_eq!(loaded.previous_wal_file, Some("wal-prev.log".to_string()));
        assert_eq!(loaded.lsn, 12345);
        assert!(loaded.is_consistent);
        assert_eq!(loaded.active_transactions, vec![1, 2, 3]);
        assert_eq!(loaded.committed_transactions.len(), 2);
        assert_eq!(loaded.committed_transactions[0].txn_id, 10);
        assert_eq!(loaded.committed_transactions[0].commit_lsn, 100);
        assert_eq!(loaded.committed_transactions[1].txn_id, 20);
        assert_eq!(loaded.committed_transactions[1].commit_lsn, 200);
    }

    #[test]
    fn test_checkpoint_metadata_no_previous_wal() {
        let dir = tempdir().unwrap();
        let checkpoint_path = dir.path().join("checkpoint.meta");

        // Test with no previous WAL file and no committed transactions
        let checkpoint = CheckpointMetadata {
            wal_file: "wal-current.log".to_string(),
            previous_wal_file: None,
            lsn: 999,
            timestamp: 9999999,
            is_consistent: false,
            active_transactions: vec![],
            committed_transactions: vec![],
        };

        checkpoint.write_to_file(&checkpoint_path).unwrap();

        let loaded = CheckpointMetadata::read_from_file(&checkpoint_path).unwrap();
        assert_eq!(loaded.wal_file, "wal-current.log");
        assert_eq!(loaded.previous_wal_file, None);
        assert_eq!(loaded.lsn, 999);
        assert!(!loaded.is_consistent);
        assert!(loaded.active_transactions.is_empty());
        assert!(loaded.committed_transactions.is_empty());
    }

    #[test]
    fn test_wal_manager_creation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Normal).unwrap();
        assert!(wal.is_running());
        assert_eq!(wal.current_lsn(), 0);

        wal.close().unwrap();
        assert!(!wal.is_running());
    }

    #[test]
    fn test_wal_manager_append_entry() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

        let entry = WALEntry::new(
            1,
            "test".to_string(),
            100,
            WALOperationType::Insert,
            vec![1, 2, 3],
        );

        let lsn = wal.append_entry(entry).unwrap();
        assert_eq!(lsn, 1);

        let entry2 = WALEntry::commit(1);
        let lsn2 = wal.append_entry(entry2).unwrap();
        assert_eq!(lsn2, 2);

        wal.close().unwrap();
    }

    #[test]
    fn test_wal_manager_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Write some entries
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            for i in 1..=5 {
                let entry = WALEntry::new(
                    i,
                    format!("table_{}", i),
                    i * 100,
                    WALOperationType::Insert,
                    vec![i as u8],
                );
                wal.append_entry(entry).unwrap();
                // Commit each transaction so it shows up in two-phase replay
                wal.write_commit_marker(i).unwrap();
            }

            wal.close().unwrap();
        }

        // Replay entries using two-phase recovery
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            let mut data_count = 0;
            let mut commit_count = 0;
            wal.replay_two_phase(0, |entry| {
                assert!(entry.lsn > 0);
                if entry.is_commit_marker() {
                    commit_count += 1;
                } else {
                    data_count += 1;
                    assert!(!entry.table_name.is_empty());
                }
                Ok(())
            })
            .unwrap();

            assert_eq!(data_count, 5);
            assert_eq!(commit_count, 5); // 5 commit markers for 5 transactions
        }
    }

    #[test]
    fn test_wal_manager_checkpoint() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

        // Add some entries
        for i in 1..=3 {
            let entry = WALEntry::new(
                i,
                "test".to_string(),
                i * 10,
                WALOperationType::Insert,
                vec![],
            );
            wal.append_entry(entry).unwrap();
        }

        // Create checkpoint
        wal.create_checkpoint(vec![]).unwrap();

        // Verify checkpoint file exists
        let checkpoint_path = wal_path.join("checkpoint.meta");
        assert!(checkpoint_path.exists());

        wal.close().unwrap();
    }

    #[test]
    fn test_wal_manager_sync_modes() {
        let dir = tempdir().unwrap();

        // Test SyncMode::None
        {
            let wal_path = dir.path().join("wal_none");
            let wal = WALManager::new(&wal_path, SyncMode::None).unwrap();
            assert!(!wal.should_sync(WALOperationType::Commit));
            assert!(!wal.should_sync(WALOperationType::Insert));
            wal.close().unwrap();
        }

        // Test SyncMode::Full
        {
            let wal_path = dir.path().join("wal_full");
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();
            assert!(wal.should_sync(WALOperationType::Commit));
            assert!(wal.should_sync(WALOperationType::Insert));
            wal.close().unwrap();
        }
    }

    #[test]
    fn test_wal_manager_multiple_operations() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

        // Transaction 1
        let insert = WALEntry::new(
            1,
            "users".to_string(),
            1,
            WALOperationType::Insert,
            vec![1, 2, 3],
        );
        let lsn1 = wal.append_entry(insert).unwrap();

        let update = WALEntry::new(
            1,
            "users".to_string(),
            1,
            WALOperationType::Update,
            vec![4, 5, 6],
        );
        let lsn2 = wal.append_entry(update).unwrap();

        let commit = WALEntry::commit(1);
        let lsn3 = wal.append_entry(commit).unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(lsn3, 3);

        // Transaction 2
        let insert2 = WALEntry::new(
            2,
            "orders".to_string(),
            100,
            WALOperationType::Insert,
            vec![],
        );
        let lsn4 = wal.append_entry(insert2).unwrap();

        let rollback = WALEntry::rollback(2);
        let lsn5 = wal.append_entry(rollback).unwrap();

        assert_eq!(lsn4, 4);
        assert_eq!(lsn5, 5);

        wal.close().unwrap();
    }

    #[test]
    fn test_wal_ddl_operations() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        let wal = WALManager::new(&wal_path, SyncMode::Normal).unwrap();

        // DDL operations should force sync in Normal mode
        let create_table = WALEntry::new(
            1,
            "new_table".to_string(),
            0,
            WALOperationType::CreateTable,
            vec![],
        );
        assert!(create_table.operation.is_ddl());

        let lsn = wal.append_entry(create_table).unwrap();
        assert_eq!(lsn, 1);

        wal.close().unwrap();
    }

    #[test]
    fn test_find_last_lsn() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with entries
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            for i in 1..=10 {
                let entry = WALEntry::new(
                    i,
                    "test".to_string(),
                    i * 10,
                    WALOperationType::Insert,
                    vec![],
                );
                wal.append_entry(entry).unwrap();
            }

            wal.close().unwrap();
        }

        // Find WAL file and check last LSN
        let mut wal_files: Vec<_> = fs::read_dir(&wal_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with("wal-") && name.ends_with(".log")
            })
            .collect();

        assert!(!wal_files.is_empty());
        wal_files.sort_by_key(|e| e.file_name());

        let last_lsn = find_last_lsn(&wal_files.last().unwrap().path()).unwrap();
        assert_eq!(last_lsn, 10);
    }

    #[test]
    fn test_wal_truncation() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL and add 10 entries
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            for i in 1..=10 {
                let entry = WALEntry::new(
                    i,
                    "test_table".to_string(),
                    i * 10,
                    WALOperationType::Insert,
                    vec![i as u8],
                );
                wal.append_entry(entry).unwrap();
                // Commit each transaction
                wal.write_commit_marker(i).unwrap();
            }

            // Get initial WAL file size
            let wal_files: Vec<_> = fs::read_dir(&wal_path)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with("wal-") && name.ends_with(".log")
                })
                .collect();
            assert_eq!(wal_files.len(), 1);
            let initial_size = wal_files[0].metadata().unwrap().len();

            // Truncate WAL at LSN 10 (keeps entries 11-20, i.e. LSN 11+ which are commit markers for txn 6-10)
            // With commit markers, LSNs are: 1(insert), 2(commit), 3(insert), 4(commit), ...
            // So truncating at LSN 10 keeps the commit markers and data for txn 6-10
            wal.truncate_wal(10).unwrap();

            // Check new WAL file exists with LSN in name
            let new_wal_files: Vec<_> = fs::read_dir(&wal_path)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with("wal-") && name.ends_with(".log")
                })
                .collect();
            assert_eq!(new_wal_files.len(), 1);

            // Check file has LSN-10 in name
            let new_name = new_wal_files[0].file_name().to_string_lossy().to_string();
            assert!(
                new_name.contains("lsn-10"),
                "Expected lsn-10 in filename, got: {}",
                new_name
            );

            // Check that file is smaller (only entries 11-20 remain)
            let truncated_size = new_wal_files[0].metadata().unwrap().len();
            assert!(
                truncated_size < initial_size,
                "Truncated WAL should be smaller"
            );

            // Verify only entries with LSN > 10 can be replayed
            let mut data_count = 0;
            let mut commit_count = 0;
            let mut min_lsn = u64::MAX;
            wal.replay_two_phase(0, |entry| {
                if entry.is_commit_marker() {
                    commit_count += 1;
                } else {
                    data_count += 1;
                }
                if entry.lsn < min_lsn {
                    min_lsn = entry.lsn;
                }
                Ok(())
            })
            .unwrap();

            // Should have 5 data entries (txn 6-10 inserts, LSN 11, 13, 15, 17, 19)
            assert_eq!(data_count, 5, "Expected 5 data entries after truncation");
            assert_eq!(
                commit_count, 5,
                "Expected 5 commit markers after truncation"
            );
            assert!(min_lsn > 10, "Minimum LSN should be > 10");

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_wal_truncation_all_entries() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL and add entries with commit markers
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            for i in 1..=5 {
                let entry = WALEntry::new(
                    i,
                    "test".to_string(),
                    i * 10,
                    WALOperationType::Insert,
                    vec![],
                );
                wal.append_entry(entry).unwrap();
                wal.write_commit_marker(i).unwrap();
            }

            // Truncate all entries (up to LSN 10, which covers all 5 inserts + 5 commits)
            wal.truncate_wal(10).unwrap();

            // Replay should return 0 entries because all data was truncated
            let mut count = 0;
            wal.replay_two_phase(0, |_entry| {
                count += 1;
                Ok(())
            })
            .unwrap();

            // All entries were truncated
            assert_eq!(count, 0, "Expected 0 entries after truncating all");

            // But the WAL file should exist and the LSN should have advanced
            let current_wal_file = wal.current_wal_file();
            assert!(
                current_wal_file.contains("lsn-10"),
                "WAL file should have lsn-10 in name"
            );

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_two_phase_recovery_committed() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with committed transaction
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            // Transaction 1: Insert entries and commit
            let entry1 = WALEntry::new(
                1, // txn_id
                "test".to_string(),
                100,
                WALOperationType::Insert,
                vec![1, 2, 3],
            );
            wal.append_entry(entry1).unwrap();

            let entry2 = WALEntry::new(
                1, // same txn_id
                "test".to_string(),
                101,
                WALOperationType::Insert,
                vec![4, 5, 6],
            );
            wal.append_entry(entry2).unwrap();

            // Write commit marker for transaction 1
            wal.write_commit_marker(1).unwrap();

            wal.close().unwrap();
        }

        // Replay using two-phase recovery
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            let mut applied_entries = Vec::new();
            let mut commit_count = 0;
            let result = wal
                .replay_two_phase(0, |entry| {
                    if entry.is_commit_marker() {
                        commit_count += 1;
                    } else {
                        applied_entries.push(entry.row_id);
                    }
                    Ok(())
                })
                .unwrap();

            // Both data entries should be applied (transaction was committed)
            assert_eq!(applied_entries.len(), 2);
            assert_eq!(applied_entries, vec![100, 101]);
            // Commit marker should also be passed to callback
            assert_eq!(commit_count, 1);
            assert_eq!(result.committed_transactions, 1);
            assert_eq!(result.aborted_transactions, 0);
            assert_eq!(result.applied_entries, 2);
            assert_eq!(result.skipped_entries, 0);

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_two_phase_recovery_uncommitted() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with uncommitted transaction (no commit marker)
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            // Transaction 1: Insert entries but DON'T commit (simulating crash)
            let entry1 = WALEntry::new(
                1, // txn_id
                "test".to_string(),
                100,
                WALOperationType::Insert,
                vec![1, 2, 3],
            );
            wal.append_entry(entry1).unwrap();

            let entry2 = WALEntry::new(
                1, // same txn_id
                "test".to_string(),
                101,
                WALOperationType::Insert,
                vec![4, 5, 6],
            );
            wal.append_entry(entry2).unwrap();

            // NO commit marker - simulating crash before commit

            wal.close().unwrap();
        }

        // Replay using two-phase recovery
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            let mut applied_entries = Vec::new();
            let result = wal
                .replay_two_phase(0, |entry| {
                    applied_entries.push(entry.row_id);
                    Ok(())
                })
                .unwrap();

            // No entries should be applied (transaction was in-doubt/uncommitted)
            assert_eq!(applied_entries.len(), 0);
            assert_eq!(result.committed_transactions, 0);
            assert_eq!(result.aborted_transactions, 0);
            assert_eq!(result.applied_entries, 0);
            assert_eq!(result.skipped_entries, 2); // Both entries skipped

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_two_phase_recovery_aborted() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with explicitly aborted transaction
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            // Transaction 1: Insert entries then abort
            let entry1 = WALEntry::new(
                1, // txn_id
                "test".to_string(),
                100,
                WALOperationType::Insert,
                vec![1, 2, 3],
            );
            wal.append_entry(entry1).unwrap();

            // Write abort marker for transaction 1
            wal.write_abort_marker(1).unwrap();

            wal.close().unwrap();
        }

        // Replay using two-phase recovery
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            let mut applied_entries = Vec::new();
            let result = wal
                .replay_two_phase(0, |entry| {
                    applied_entries.push(entry.row_id);
                    Ok(())
                })
                .unwrap();

            // No entries should be applied (transaction was aborted)
            assert_eq!(applied_entries.len(), 0);
            assert_eq!(result.committed_transactions, 0);
            assert_eq!(result.aborted_transactions, 1);
            assert_eq!(result.applied_entries, 0);
            assert_eq!(result.skipped_entries, 1); // Entry skipped

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_two_phase_recovery_mixed_transactions() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with mixed transactions
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            // Transaction 1: Committed
            let entry1 = WALEntry::new(
                1,
                "test".to_string(),
                100,
                WALOperationType::Insert,
                vec![1],
            );
            wal.append_entry(entry1).unwrap();
            wal.write_commit_marker(1).unwrap();

            // Transaction 2: Aborted
            let entry2 = WALEntry::new(
                2,
                "test".to_string(),
                200,
                WALOperationType::Insert,
                vec![2],
            );
            wal.append_entry(entry2).unwrap();
            wal.write_abort_marker(2).unwrap();

            // Transaction 3: Uncommitted (in-doubt)
            let entry3 = WALEntry::new(
                3,
                "test".to_string(),
                300,
                WALOperationType::Insert,
                vec![3],
            );
            wal.append_entry(entry3).unwrap();

            // Transaction 4: Committed
            let entry4 = WALEntry::new(
                4,
                "test".to_string(),
                400,
                WALOperationType::Insert,
                vec![4],
            );
            wal.append_entry(entry4).unwrap();
            wal.write_commit_marker(4).unwrap();

            wal.close().unwrap();
        }

        // Replay using two-phase recovery
        {
            let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

            let mut applied_entries = Vec::new();
            let mut commit_markers = Vec::new();
            let result = wal
                .replay_two_phase(0, |entry| {
                    if entry.is_commit_marker() {
                        commit_markers.push(entry.txn_id);
                    } else {
                        applied_entries.push(entry.row_id);
                    }
                    Ok(())
                })
                .unwrap();

            // Only transactions 1 and 4 should be applied (data entries)
            assert_eq!(applied_entries.len(), 2);
            assert!(applied_entries.contains(&100)); // from txn 1
            assert!(applied_entries.contains(&400)); // from txn 4
                                                     // Commit markers for txn 1 and 4 should also be passed
            assert_eq!(commit_markers.len(), 2);
            assert!(commit_markers.contains(&1));
            assert!(commit_markers.contains(&4));
            assert_eq!(result.committed_transactions, 2);
            assert_eq!(result.aborted_transactions, 1);
            assert_eq!(result.applied_entries, 2);
            assert_eq!(result.skipped_entries, 2); // txn 2 and txn 3 entries

            wal.close().unwrap();
        }
    }

    #[test]
    fn test_wal_rotation_basic() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with small max size to trigger rotation
        let config = PersistenceConfig {
            wal_max_size: 500, // 500 bytes - very small to trigger rotation
            ..Default::default()
        };

        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config)).unwrap();

        // Initial state
        assert_eq!(wal.current_sequence(), 0);

        // Write entries that should exceed 500 bytes
        for i in 1..=10 {
            let entry = WALEntry::new(
                i,
                "test_table".to_string(),
                i * 100,
                WALOperationType::Insert,
                vec![0u8; 100], // 100 bytes of data
            );
            wal.append_entry(entry).unwrap();
        }

        // Check if rotation would be needed
        let current_size = wal.current_file_size();
        let initial_file = wal.current_wal_file();

        // Manually trigger rotation check
        let rotated = wal.maybe_rotate().unwrap();

        if rotated {
            // Sequence should have incremented
            assert!(
                wal.current_sequence() > 0,
                "Sequence should increment after rotation"
            );

            // File position should have reset
            assert!(
                wal.current_file_size() < current_size,
                "File position should reset after rotation"
            );

            // New WAL file should have different name
            let new_file = wal.current_wal_file();
            assert_ne!(
                new_file, initial_file,
                "WAL filename should change after rotation"
            );
        }

        wal.close().unwrap();
    }

    #[test]
    fn test_wal_rotation_preserves_data() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with larger max size (avoid multiple rotations during writes)
        let config = PersistenceConfig {
            wal_max_size: 2000, // Large enough for initial writes
            ..Default::default()
        };

        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config)).unwrap();

        // Write entries before rotation
        for i in 1..=3 {
            let entry = WALEntry::new(
                i,
                "test".to_string(),
                i * 10,
                WALOperationType::Insert,
                vec![0u8; 50],
            );
            wal.append_entry(entry).unwrap();
            wal.write_commit_marker(i).unwrap();
        }

        // Count files before rotation
        let files_before: Vec<_> = std::fs::read_dir(&wal_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".log"))
            .collect();

        let initial_file = wal.current_wal_file();

        // Force rotation by temporarily modifying the position
        // (in production, this would happen naturally when file exceeds max_wal_size)
        wal.current_file_position
            .store(wal.max_file_size() + 1, Ordering::Release);
        wal.maybe_rotate().unwrap();

        // Verify rotation occurred
        let new_file = wal.current_wal_file();
        assert_ne!(
            initial_file, new_file,
            "WAL file should have changed after rotation"
        );

        // Write more entries after rotation
        for i in 4..=6 {
            let entry = WALEntry::new(
                i,
                "test".to_string(),
                i * 10,
                WALOperationType::Insert,
                vec![0u8; 50],
            );
            wal.append_entry(entry).unwrap();
            wal.write_commit_marker(i).unwrap();
        }

        // Count files after rotation (should be 2)
        let files_after: Vec<_> = std::fs::read_dir(&wal_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".log"))
            .collect();

        assert!(
            files_after.len() > files_before.len(),
            "Should have more WAL files after rotation"
        );

        wal.close().unwrap();

        // Reopen and replay - should get all committed entries from BOTH files
        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config)).unwrap();

        let mut row_ids = Vec::new();
        let mut commit_count = 0;
        let result = wal
            .replay_two_phase(0, |entry| {
                if entry.is_commit_marker() {
                    commit_count += 1;
                } else {
                    row_ids.push(entry.row_id);
                }
                Ok(())
            })
            .unwrap();

        // Should have all 6 data entries (3 from before rotation + 3 from after)
        assert_eq!(
            row_ids.len(),
            6,
            "Should have 6 entries total from both WAL files"
        );
        assert_eq!(commit_count, 6, "Should have 6 commit markers");
        assert_eq!(
            result.committed_transactions, 6,
            "Should have 6 committed transactions"
        );

        // Verify row IDs are in order (entries from all files)
        let expected: Vec<i64> = (1..=6).map(|i| i * 10).collect();
        assert_eq!(row_ids, expected);

        wal.close().unwrap();
    }

    #[test]
    fn test_wal_no_rotation_below_threshold() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");

        // Create WAL with large max size (default)
        let wal = WALManager::new(&wal_path, SyncMode::Full).unwrap();

        // Write a few small entries
        for i in 1..=3 {
            let entry = WALEntry::new(
                i,
                "test".to_string(),
                i * 10,
                WALOperationType::Insert,
                vec![1, 2, 3],
            );
            wal.append_entry(entry).unwrap();
        }

        // Get initial values
        let initial_sequence = wal.current_sequence();
        let initial_file = wal.current_wal_file();

        // Rotation should not occur (file size is below threshold)
        let rotated = wal.maybe_rotate().unwrap();
        assert!(!rotated, "Should not rotate below threshold");

        // Verify nothing changed
        assert_eq!(wal.current_sequence(), initial_sequence);
        assert_eq!(wal.current_wal_file(), initial_file);

        wal.close().unwrap();
    }
}
