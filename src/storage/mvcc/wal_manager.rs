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

use crate::common::time_compat::{SystemTime, UNIX_EPOCH};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::common::I64Set;
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

/// Current WAL entry format version.
///
/// - **v1** (historical): no header magic byte, retired before v0.4.
/// - **v2** (released): 32-byte header with magic + version + flags +
///   LSN + entry size; data portion has txn_id, table_name, row_id,
///   operation, timestamp, data_len + data. CRC32 covers data only.
/// - **v3** (SWMR v2): adds an `i64 commit_seq` to the data portion
///   immediately after the operation byte AND extends CRC32 to cover
///   header + data so the LSN / flags / entry_size are authenticated.
///   For DML entries `commit_seq` is 0 (placeholder; reader joins to
///   the commit marker by txn_id). For commit markers it is the
///   writer's actual commit_seq — what reader's WAL-tail uses to tag
///   tombstones consistently with `snapshot_seq` filtering. v2
///   entries decode with `commit_seq=0` and the data-only CRC scope
///   (selected by the version byte), so v0.4.0 WALs replay in place.
const WAL_FORMAT_VERSION: u8 = 3;

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
    TruncateTable = 13,
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
            13 => Some(WALOperationType::TruncateTable),
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
                | WALOperationType::TruncateTable
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
    /// Commit sequence number (SWMR v2, format v3).
    ///
    /// - On a commit marker: the actual `commit_seq` allocated by
    ///   `TransactionRegistry::start_commit`. Reader's WAL-tail uses
    ///   this to tag tombstones from the corresponding txn's DML so
    ///   `snapshot_seq` filtering matches what the writer's manifest
    ///   stores.
    /// - On DML entries: 0 (placeholder; reader joins to the txn's
    ///   commit marker by `txn_id`).
    /// - On v2 entries decoded under v3 code: 0 (back-compat).
    pub commit_seq: i64,
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
            commit_seq: 0, // 0 for DML; commit markers override via with_commit_seq.
        }
    }

    /// Set `commit_seq` on this entry. Builder-style; used by the commit
    /// marker constructor to record the writer's allocated commit_seq
    /// so reader's WAL-tail can tag tombstones consistently.
    pub fn with_commit_seq(mut self, commit_seq: i64) -> Self {
        self.commit_seq = commit_seq;
        self
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

    /// Create a commit entry (with COMMIT_MARKER flag for two-phase
    /// recovery). `commit_seq` is the value `start_commit` allocated
    /// for this transaction; reader's WAL-tail consumes it (SWMR v2).
    /// Pass 0 for DDL-style synthetic commits that don't participate
    /// in snapshot_seq.
    pub fn commit(txn_id: i64, commit_seq: i64) -> Self {
        Self::with_flags(
            txn_id,
            String::new(),
            0,
            WALOperationType::Commit,
            Vec::new(),
            WalFlags::COMMIT_MARKER,
        )
        .with_commit_seq(commit_seq)
    }

    /// Create a commit marker entry. Equivalent to [`Self::commit`].
    pub fn commit_marker(txn_id: i64, commit_seq: i64) -> Self {
        Self::commit(txn_id, commit_seq)
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

    /// Check if this entry is a commit marker.
    ///
    /// Classify on the `operation` byte ONLY. The
    /// header `flags` byte is NOT covered by the WAL CRC (CRC32
    /// hashes only the data portion starting after the 32-byte
    /// header), so a header-bit corruption could flip
    /// `COMMIT_MARKER` on for a DML entry (silently adding the
    /// txn to `committed_txns`) or off for a real commit marker
    /// (silently losing the commit). The `operation` byte lives
    /// in the CRC-protected data section, so it's the only safe
    /// source of truth for marker classification.
    pub fn is_commit_marker(&self) -> bool {
        self.operation == WALOperationType::Commit
    }

    /// Check if this entry is an abort marker. Same CRC-coverage
    /// reasoning as `is_commit_marker` — classify on the
    /// CRC-protected `operation` byte.
    pub fn is_abort_marker(&self) -> bool {
        self.operation == WALOperationType::Rollback
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
        // Determine if we should compress the data payload.
        // Avoid cloning self.data for the uncompressed case — write it
        // directly into buf via extend_from_slice instead.
        let compressed_data: Option<Vec<u8>>;
        let use_compression;
        if self.data.len() >= COMPRESSION_THRESHOLD {
            let compressed = lz4_flex::compress_prepend_size(&self.data);
            if compressed.len() < self.data.len() {
                compressed_data = Some(compressed);
                use_compression = true;
            } else {
                compressed_data = None;
                use_compression = false;
            }
        } else {
            compressed_data = None;
            use_compression = false;
        }
        let payload: &[u8] = compressed_data.as_deref().unwrap_or(&self.data);

        // Calculate data portion size: txnID(8) + tableNameLen(2) + tableName + rowID(8) + op(1) + commitSeq(8, v3+) + ts(8) + dataLen(4) + data
        let data_size = 8 + 2 + self.table_name.len() + 8 + 1 + 8 + 8 + 4 + payload.len();

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

        // CommitSeq (8 bytes, format v3). Always written by v3 encoders;
        // v2 decoders that see a v3 entry would fail at the size check
        // anyway because of the extra 8 bytes — version byte tells the
        // decoder which layout to expect.
        buf.extend_from_slice(&self.commit_seq.to_le_bytes());

        // Timestamp (8 bytes)
        buf.extend_from_slice(&self.timestamp.to_le_bytes());

        // Data length (4 bytes) + data (possibly compressed)
        // When compressed, lz4_flex::compress_prepend_size includes the original size
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload);

        // ========== CRC32 (4 bytes) ==========
        // v3 CRC covers the FULL buffer (header + data). v2's
        // data-only scope left LSN / flags / entry_size /
        // version unauthenticated; a header-bit corruption
        // could move an in-window scanner entry out of window
        // or change a DML's apparent commit-marker
        // classification. v3 decoders verify the full CRC; v2
        // entries (legacy on-disk WALs from v0.4.0) fall back
        // to data-only CRC for back-compat.
        let crc = crc32fast::hash(&buf[..]);
        buf.extend_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Verify the CRC32 of an encoded entry without
    /// parsing the fields. Cheaper than `decode` (skips
    /// decompression + field decoding) — used by SWMR tail
    /// scanners to authenticate the header (v3) or the data
    /// portion (v2) BEFORE trusting the header LSN for
    /// windowing decisions. Without this, a corrupted header LSN
    /// could move an in-window record out of window, the scanner
    /// would `seek` past it, and the corruption would never
    /// surface as `SwmrOverlayApplyFailed`.
    ///
    /// `header_bytes` is the 32-byte on-disk header; required for
    /// v3 (covered by CRC) and ignored for v2.
    pub fn verify_crc(version: u8, header_bytes: &[u8], data: &[u8]) -> Result<()> {
        if data.len() < 4 {
            return Err(Error::internal(format!(
                "data too short for CRC tail: {} bytes",
                data.len()
            )));
        }
        if version != 2 && version != 3 {
            return Err(Error::internal(format!(
                "unsupported WAL entry format version: {}",
                version
            )));
        }
        let crc_offset = data.len() - 4;
        let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());
        let computed_crc = if version >= 3 {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(header_bytes);
            hasher.update(&data[..crc_offset]);
            hasher.finalize()
        } else {
            crc32fast::hash(&data[..crc_offset])
        };
        if stored_crc != computed_crc {
            return Err(Error::internal(format!(
                "WAL CRC mismatch (version {}): stored={:#x}, computed={:#x}",
                version, stored_crc, computed_crc
            )));
        }
        Ok(())
    }

    /// Cheap classification probe: extract `(txn_id, operation)`
    /// from the data portion WITHOUT building a full `WALEntry`
    /// (no payload allocation, no decompression, no schema/value
    /// parsing). Used by Phase 1 (which only needs to find commit
    /// markers and their txn_id) and Phase 2 in DDL-only mode
    /// (which only needs to filter on `operation.is_ddl()`).
    ///
    /// Caller MUST have CRC-validated `data` against `header_bytes`
    /// already — this function trusts the bytes. Layout matches
    /// `decode`: txn_id(8) + name_len(2) + name(name_len) +
    /// row_id(8) + op(1). For DML the operation byte is enough to
    /// classify; for commit/abort markers the txn_id is the
    /// payload of interest.
    ///
    /// Returns `None` on layout corruption (data too short for
    /// the fixed-offset reads). Caller should fall back to full
    /// `decode` to surface a richer error.
    pub fn peek_classification(data: &[u8]) -> Option<(i64, WALOperationType)> {
        // CRC tail; payload size = data.len() - 4.
        if data.len() < 4 + 8 + 2 + 8 + 1 {
            return None;
        }
        let payload_len = data.len() - 4;
        let mut pos = 0;
        let txn_id = i64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let name_len = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
        pos += 2;
        if pos + name_len + 8 + 1 > payload_len {
            return None;
        }
        pos += name_len; // skip table_name
        pos += 8; // skip row_id
        let op_byte = data[pos];
        let op = WALOperationType::from_u8(op_byte)?;
        Some((txn_id, op))
    }

    /// Decode entry from data portion (after header has been parsed)
    ///
    /// Parameters:
    /// - lsn, previous_lsn, flags, version: extracted from header by caller.
    ///   `version` selects the data-portion layout AND CRC scope:
    ///   - v2 (legacy v0.4.0): no commit_seq, CRC covers data only.
    ///   - v3 (current): adds an 8-byte commit_seq after the operation
    ///     byte AND extends CRC to cover header + data so the header
    ///     bytes (LSN, flags, entry_size) are authenticated.
    ///
    ///   Versions outside the supported range error.
    /// - data: data portion + CRC (4 bytes)
    /// - header_bytes: the 32-byte on-disk header. Required for v3
    ///   CRC validation; v2 ignores it. Pre-existing tests can pass
    ///   an all-zero buffer for v2 entries.
    pub fn decode(
        lsn: u64,
        previous_lsn: u64,
        flags: WalFlags,
        version: u8,
        data: &[u8],
        header_bytes: &[u8],
    ) -> Result<Self> {
        // Minimum size: txnID(8) + tableNameLen(2) + rowID(8) + op(1) + ts(8) + dataLen(4) + CRC(4) = 35
        // (v3 also has commit_seq(8); the per-version size guard fires
        // after we read enough header to know the version.)
        if data.len() < 35 {
            return Err(Error::internal(format!(
                "data too short for WAL entry: {} bytes",
                data.len()
            )));
        }
        if version != 2 && version != 3 {
            return Err(Error::internal(format!(
                "unsupported WAL entry format version: {}",
                version
            )));
        }

        // Verify CRC32 (last 4 bytes). Scope depends on version:
        //   - v2: CRC covers data portion only (offset 32..pre-CRC).
        //   - v3: CRC covers header (32 bytes) + data portion.
        let crc_offset = data.len() - 4;
        let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());
        let computed_crc = if version >= 3 {
            // Hash header + data so header bytes are
            // integrity-protected. Caller provides the original
            // 32-byte header buffer; we hash that followed by the
            // pre-CRC data slice.
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(header_bytes);
            hasher.update(&data[..crc_offset]);
            hasher.finalize()
        } else {
            crc32fast::hash(&data[..crc_offset])
        };

        if stored_crc != computed_crc {
            return Err(Error::internal(format!(
                "WAL entry checksum mismatch at LSN {} (version {}): stored={:#x}, computed={:#x}",
                lsn, version, stored_crc, computed_crc
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

        // CommitSeq (8 bytes, v3 only). v2 entries default to 0 — the
        // synthesize-from-marker-LSN fallback for v2 commit markers is
        // applied by the WAL-tail caller, not here.
        let commit_seq: i64 = if version >= 3 {
            if pos + 8 > data.len() {
                return Err(Error::internal(
                    "unexpected end of data reading commit_seq (v3)",
                ));
            }
            let v = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            v
        } else {
            0
        };

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
            commit_seq,
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

/// Per-reader cursor recording the byte offset just past the
/// last record this reader has CRC-validated AND whose LSN
/// fell within the prior scan's `to_lsn` window. Used by
/// `tail_committed_entries` to skip re-reading + re-CRC-
/// validating bytes the same reader already proved authentic.
///
/// CRC-safe by construction: a record's bytes can only land
/// before `offset` if a prior scan read past it, which means
/// `verify_crc` accepted it then. WAL files are append-only
/// between rotations (the writer never rewrites already-
/// written bytes), so the validated prefix stays stable. On
/// rotation the cursor's `path` no longer matches the current
/// scan target and the cache is bypassed for that file.
///
/// `lsn_at_offset` records the LSN of the last record
/// covered by the cursor. The next scan reuses the cursor
/// ONLY when its window's lower bound is `>= lsn_at_offset` —
/// otherwise it would skip records the new window still
/// needs (e.g. an in-flight transaction's DML below
/// `lsn_at_offset` whose commit marker now lands in the
/// new window, which Phase 2 must rescan from `entry_floor`
/// upward).
///
/// Trade-off note: a hostile process editing already-validated
/// WAL bytes out-of-band could escape detection. Stoolap
/// trusts the WAL directory to be writable only by the
/// running writer, same as the rest of the persistence layer.
#[derive(Clone, Debug)]
pub struct WalScanCursor {
    /// Absolute path of the WAL file the offset belongs to.
    pub path: PathBuf,
    /// Byte position just past the last in-window record this
    /// reader has CRC-validated in `path`.
    pub offset: u64,
    /// LSN of the last record covered by the cursor (i.e. the
    /// largest LSN whose record fully fits in `[0, offset)`).
    /// `0` means "no records yet" — the cursor only holds the
    /// file path, not a real position. Reuse requires the
    /// next scan's lower bound to be `>= lsn_at_offset`.
    pub lsn_at_offset: u64,
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
                        // Cap allocation to what the section can actually hold to prevent
                        // OOM from corrupt checkpoint data
                        let max_entries = (section_end - pos - 8) / 8;
                        let safe_count = count.min(max_entries);
                        let mut spos = pos + 8;
                        active_transactions = Vec::with_capacity(safe_count);
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
                        // Cap allocation to what the section can actually hold to prevent
                        // OOM from corrupt checkpoint data
                        let max_entries = (section_end - pos - 8) / 16;
                        let safe_count = count.min(max_entries);
                        let mut spos = pos + 8;
                        committed_transactions = Vec::with_capacity(safe_count);
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

        // Write atomically using temp file + rename (unique name to avoid races)
        let unique_suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let temp_path = path.with_extension(format!("meta.{}.tmp", unique_suffix));

        let mut file = File::create(&temp_path).map_err(|e| {
            Error::internal(format!("failed to create checkpoint temp file: {}", e))
        })?;

        if let Err(e) = file.write_all(&buf).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temp_path);
            return Err(Error::internal(format!(
                "failed to write checkpoint: {}",
                e
            )));
        }

        // Atomic rename
        fs::rename(&temp_path, path)
            .map_err(|e| Error::internal(format!("failed to rename checkpoint: {}", e)))?;

        // Sync directory to ensure rename is durable.
        // Windows does not support opening directories for fsync.
        #[cfg(not(windows))]
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
    /// Highest WAL LSN whose encoded bytes have been written from the
    /// in-memory WAL buffer to the WAL file. This is intentionally a
    /// file-write watermark, not an fsync durability watermark:
    /// SWMR readers need bytes to be readable from the OS page cache,
    /// while `SyncMode` still controls power-failure durability.
    flushed_lsn: AtomicU64,
    /// Previous LSN for entry chaining (enables backward traversal)
    previous_lsn: AtomicU64,
    /// Write buffer
    buffer: Mutex<Vec<u8>>,
    /// Highest LSN currently present in `buffer`. Guarded by
    /// `buffer`'s mutex; atomic only so the WAL manager can keep the
    /// field adjacent to the other LSN watermarks without introducing
    /// another lock type.
    buffer_highest_lsn: AtomicU64,
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
    /// SWMR v2 P2 perf fix: per-active-txn first DML LSN. Inserted
    /// on the first DML entry the writer appends for a given
    /// `txn_id > 0`; removed on commit/abort marker. The MIN over
    /// values is the "oldest active txn LSN" published to db.shm so
    /// readers can floor their WAL-tail Phase 2 entry scan instead
    /// of starting from LSN 0 every refresh. Synthetic txn_ids
    /// (DDL = -2, RECOVERY, INVALID) are NOT tracked: they're
    /// auto-committed (no in-flight period) or used during recovery.
    active_txn_first_lsn: parking_lot::Mutex<rustc_hash::FxHashMap<i64, u64>>,
    /// Cached MIN of `active_txn_first_lsn` values, refreshed on
    /// every insert/remove. `u64::MAX` when the map is empty (no
    /// active user txns). Engine reads this and publishes to
    /// `db.shm.oldest_active_txn_lsn` after commit/abort.
    oldest_active_lsn_cache: AtomicU64,
    /// Optional `Arc<ShmHandle>` set by the engine after `db.shm`
    /// creation. When set, every active-set change in
    /// `refresh_oldest_active_cache_locked` ALSO does
    /// `shm.oldest_active_txn_lsn.fetch_min(new_oldest, ...)` so
    /// shm tracks a monotonically-decreasing lower bound of the
    /// WAL's actual oldest. This is the deterministic guarantee
    /// the reader's `pre_acquire_swmr_for_read_only_path` relies
    /// on: any txn that has appended its first DML before the
    /// reader samples shm has its first-DML LSN reflected (≤)
    /// in `shm.oldest_active_txn_lsn`, regardless of whether
    /// the writer's commit fast path skipped the seqlock
    /// publish or whether the cleanup loop has had a chance to
    /// run a barrier publish.
    ///
    /// `fetch_min` only — the value never RAISES via this
    /// path. Slow-path commits and `barrier_publish_full_state`
    /// raise via `.store()` under the seqlock when the actual
    /// oldest moves up (e.g., a long txn cleared).
    ///
    /// `OnceLock` (not `Mutex<Option<...>>`) because the engine
    /// wires this once during `MVCCEngine::new` after the shm
    /// handle is created and never changes it for the lifetime
    /// of the WAL manager. `.get()` is a single relaxed atomic
    /// load — eliminates the per-WAL-append + per-commit Mutex
    /// lock that the previous `Mutex<Option<...>>` paid even on
    /// engines with no SWMR readers. Cost on the no-mirror path
    /// (write-only engine, in-memory variants that never wire
    /// the mirror) drops from ~15-25 ns to ~1 ns.
    shm_oldest_mirror: std::sync::OnceLock<Arc<crate::storage::mvcc::shm::ShmHandle>>,
}

impl WALManager {
    /// Create a new WAL manager with default config (writable).
    pub fn new(path: impl AsRef<Path>, sync_mode: SyncMode) -> Result<Self> {
        Self::with_config(path, sync_mode, None, false)
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
        read_only: bool,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create WAL directory only on writable opens. On a read-only
        // mount the dir must already exist (caller verifies); calling
        // create_dir_all would fail with EROFS / EACCES.
        if !read_only {
            fs::create_dir_all(&path)
                .map_err(|e| Error::internal(format!("failed to create WAL directory: {}", e)))?;
        }

        // CRITICAL: Recover from any interrupted truncation before proceeding
        // This ensures data integrity if a crash happened during WAL truncation.
        // Skipped on read-only opens (recovery would write to disk).
        if !read_only {
            Self::recover_interrupted_truncation(&path)?;
        }

        // Pick how to open existing WAL files. Writable opens want
        // `read + append` (and create if missing). Read-only opens want
        // `read` only — no append (would require write perm), no create
        // (would create state on a read-only mount).
        let open_existing_wal = |wal_path: &Path| -> std::io::Result<File> {
            if read_only {
                OpenOptions::new().read(true).open(wal_path)
            } else {
                OpenOptions::new().read(true).append(true).open(wal_path)
            }
        };

        // Read-only opens distinguish "wal/ missing" (acceptable —
        // volumes-only deployments are valid; checkpointed databases
        // shipped without a WAL fall in this bucket) from "wal/ exists
        // but unreadable" (fatal — silently coming up with no WAL would
        // let the engine appear "successfully open" while missing every
        // uncheckpointed change).
        if read_only {
            match fs::read_dir(&path) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // No wal/ at all. Caller relies on volumes / fresh
                    // engine. Continue with no WAL.
                }
                Err(e) => {
                    return Err(Error::internal(format!(
                        "read-only open: cannot read WAL directory '{}': {}",
                        path.display(),
                        e
                    )));
                }
            }
        }

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
                if let Ok(file) = open_existing_wal(&wal_path) {
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
                    if (name.starts_with("wal-") || name.starts_with("wal_"))
                        && name.ends_with(".log")
                    {
                        wal_files.push(name);
                    }
                }
            }

            // Read-only opens with WAL files present that we cannot open:
            // surface the failure rather than silently coming up with no
            // WAL. (For writable opens, the create-new-WAL path below
            // takes over.)
            if read_only && !wal_files.is_empty() && wal_file.is_none() {
                // Re-attempt the open of the newest file to capture the
                // OS error. Sorting first so we pick the same file the
                // happy path would have used.
                let mut sorted = wal_files.clone();
                sorted.sort_by_key(|name| Self::extract_lsn_from_filename(name).unwrap_or(0));
                if let Some(newest) = sorted.last() {
                    let wal_path = path.join(newest);
                    if let Err(e) = open_existing_wal(&wal_path) {
                        return Err(Error::internal(format!(
                            "read-only open: cannot read WAL file '{}': {}",
                            wal_path.display(),
                            e
                        )));
                    }
                }
            }

            // Sort by embedded LSN so we pick the file with the highest LSN
            wal_files.sort_by_key(|name| Self::extract_lsn_from_filename(name).unwrap_or(0));

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

                if let Ok(file) = open_existing_wal(&wal_path) {
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

        // Create new WAL file if none exists. Writable opens only —
        // a read-only mount can't create a file, and a read-only handle
        // has no business appending to the WAL anyway. If no WAL files
        // were found in read-only mode, wal_file stays None and replay
        // is a no-op (volumes alone supply the engine state).
        if wal_file.is_none() && !read_only {
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
            flushed_lsn: AtomicU64::new(initial_lsn),
            previous_lsn: AtomicU64::new(initial_lsn),
            buffer: Mutex::new(Vec::with_capacity(buffer_size)),
            buffer_highest_lsn: AtomicU64::new(0),
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
            active_txn_first_lsn: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            // u64::MAX = "no active user txns" sentinel.
            oldest_active_lsn_cache: AtomicU64::new(u64::MAX),
            shm_oldest_mirror: std::sync::OnceLock::new(),
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

    /// Highest WAL LSN whose bytes have been written to the WAL file
    /// and are therefore safe to advertise to cross-process readers.
    pub fn flushed_lsn(&self) -> u64 {
        self.flushed_lsn.load(Ordering::Acquire)
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

        // SWMR v2 P2 perf fix: track first DML LSN per active user
        // transaction so the engine can publish the MIN as
        // db.shm.oldest_active_txn_lsn. Skip synthetic txn ids:
        //   - DDL (-2): auto-committed, no in-flight period
        //   - INVALID (0) / RECOVERY (negative): not user txns
        // Skip marker entries (commit/abort/rotation) — those don't
        // open a new in-flight DML window.
        if entry.txn_id > 0 && !entry.is_marker_entry() {
            let mut map = self.active_txn_first_lsn.lock();
            // entry_or_insert preserves the FIRST DML LSN we saw for
            // this txn — subsequent DML entries don't overwrite.
            if let std::collections::hash_map::Entry::Vacant(slot) = map.entry(entry.txn_id) {
                slot.insert(entry.lsn);
                self.refresh_oldest_active_cache_locked(&map);
            }
        }

        // Encode entry with new V2 format
        let encoded = entry.encode();
        let encoded_len = encoded.len() as u64;

        // Write to buffer
        {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.extend_from_slice(&encoded);
            self.buffer_highest_lsn
                .fetch_max(entry.lsn, Ordering::AcqRel);

            let needs_flush = buffer.len() >= self.flush_trigger as usize;
            // Full still writes every WAL entry immediately. Normal
            // keeps the historical commit/DDL flush behavior. None
            // intentionally does NOT force-flush transaction-end/DDL
            // markers; SWMR publication is capped by `flushed_lsn`
            // instead, so readers never observe a marker that is
            // still only in this in-memory buffer.
            let force_flush = self.sync_mode == SyncMode::Full
                || (self.sync_mode == SyncMode::Normal
                    && (entry.operation.is_transaction_end() || entry.operation.is_ddl()));

            if needs_flush || force_flush {
                let buffer_data = std::mem::take(&mut *buffer);
                let max_lsn = self.buffer_highest_lsn.swap(0, Ordering::AcqRel);
                // CRITICAL: Increment in-flight counter BEFORE releasing lock
                // This prevents checkpoint from reading LSN before data is on disk
                self.in_flight_writes.fetch_add(1, Ordering::SeqCst);
                drop(buffer); // Release buffer lock before file operations

                // Use a guard pattern to ensure we decrement even on error
                let write_result = self.write_to_file(&buffer_data, max_lsn);
                self.in_flight_writes.fetch_sub(1, Ordering::SeqCst);
                write_result?;

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

    /// Write a commit marker for two-phase recovery. `commit_seq` is
    /// the value `TransactionRegistry::start_commit` allocated for the
    /// txn; reader's WAL-tail uses it to tag tombstones with a
    /// `snapshot_seq`-compatible commit_seq. Pass 0 for synthetic
    /// commits (DDL paths) that don't participate in snapshot ordering.
    pub fn write_commit_marker(&self, txn_id: i64, commit_seq: i64) -> Result<u64> {
        let entry = WALEntry::commit_marker(txn_id, commit_seq);
        let lsn = self.append_entry(entry)?;
        // Do NOT clear active_txn_first_lsn here.
        // The prior version removed the txn from the map
        // immediately after writing the marker — but the txn isn't
        // safely visible until `complete_commit` AND publish run.
        // If a concurrent commit/DDL publish observed the cleared
        // map in that window, it would store
        // `oldest_active_txn_lsn` without this txn → readers would
        // advance `next_entry_floor` above this txn's earlier DML
        // LSNs and lose those rows when the marker eventually
        // becomes visible.
        //
        // Cleanup is deferred to `clear_active_txn`, which the
        // engine's `publish_visible_commit_lsn` calls AFTER the
        // safe-visible publish has fired (so the snapshot it
        // publishes still includes this txn's first DML).
        Ok(lsn)
    }

    /// Write an abort marker for two-phase recovery
    pub fn write_abort_marker(&self, txn_id: i64) -> Result<u64> {
        let entry = WALEntry::abort_marker(txn_id);
        let lsn = self.append_entry(entry)?;
        // Same deferred-cleanup contract as
        // write_commit_marker. The aborted txn's DML stays in the
        // WAL (and won't be applied since txn_id won't be in
        // committed_txns), but its first-DML LSN must remain in
        // `active_txn_first_lsn` until `clear_active_txn` is
        // called by the engine's commit-finalize path.
        Ok(lsn)
    }

    /// Clear a txn's active-DML record. Called by
    /// the engine's `publish_visible_commit_lsn` AFTER the
    /// safe-visible store has been published — so any reader that
    /// observes the new visible_commit_lsn also observed an
    /// `oldest_active_txn_lsn` that still included this txn (and
    /// therefore its earlier DML LSNs). Subsequent publishes won't
    /// include this txn anymore. No-op for synthetic txn_ids
    /// (DDL/INVALID/RECOVERY) that were never tracked on the
    /// entry side.
    pub fn clear_active_txn(&self, txn_id: i64) {
        if txn_id > 0 {
            let mut map = self.active_txn_first_lsn.lock();
            if map.remove(&txn_id).is_some() {
                self.refresh_oldest_active_cache_locked(&map);
            }
        }
    }

    /// Recompute the cached `oldest_active_lsn` from the
    /// `active_txn_first_lsn` map. Caller must hold the map's lock.
    /// `u64::MAX` is the "no active user txns" sentinel.
    ///
    /// Also `fetch_min`-mirrors the new value into
    /// `db.shm.oldest_active_txn_lsn` when the engine has wired
    /// in an `Arc<ShmHandle>`. This is the deterministic
    /// guarantee a fresh reader's pre_acquire relies on:
    /// any txn that has appended its first DML before the
    /// reader samples shm has its first-DML LSN reflected (≤)
    /// in shm.oldest_active_txn_lsn, even when the writer's
    /// commit fast path skipped the seqlock publish during a
    /// `lease_present == false` window. `fetch_min` only
    /// LOWERS — slow-path commits and `barrier_publish_full_state`
    /// raise via `.store()` under the seqlock when the actual
    /// oldest moves up (e.g., a long txn cleared).
    fn refresh_oldest_active_cache_locked(&self, map: &rustc_hash::FxHashMap<i64, u64>) {
        let new = map.values().copied().min().unwrap_or(u64::MAX);
        self.oldest_active_lsn_cache.store(new, Ordering::Release);
        // Mirror to shm if attached. The fetch_min keeps the
        // shm value bounded above by `new`, so a reader sees
        // a valid lower-bound at any time — even mid-update.
        // No seqlock needed: this only DECREASES the value;
        // readers using the (visible, oldest) pair still get a
        // safe (over-pinning) result via Release-Acquire on
        // visible_commit_lsn.
        if let Some(handle) = self.shm_oldest_mirror.get() {
            handle
                .header()
                .oldest_active_txn_lsn
                .fetch_min(new, Ordering::AcqRel);
        }
    }

    /// Wire an `Arc<ShmHandle>` for the
    /// `db.shm.oldest_active_txn_lsn` mirror. Called by the
    /// engine after creating shm so subsequent active-set
    /// changes propagate immediately. Idempotent — only the
    /// first wire-in takes effect (subsequent calls are a no-op
    /// that intentionally drops the new Arc). The engine wires
    /// once during init, so the Err arm is unreachable in
    /// practice; we silently ignore it rather than panic.
    pub fn set_shm_oldest_mirror(&self, shm: Arc<crate::storage::mvcc::shm::ShmHandle>) {
        let _ = self.shm_oldest_mirror.set(shm);
    }

    /// SWMR v2 P2 perf fix: lowest first-DML LSN across active user
    /// transactions. `u64::MAX` when no user txn has open DML in
    /// the WAL — readers may safely skip entries below their
    /// `from_lsn`. Other negative/synthetic txn ids (DDL,
    /// RECOVERY) are NOT counted: DDL is auto-committed; RECOVERY
    /// applies during startup before any reader can attach.
    pub fn oldest_active_txn_lsn(&self) -> u64 {
        self.oldest_active_lsn_cache.load(Ordering::Acquire)
    }

    /// Write data to WAL file
    fn write_to_file(&self, data: &[u8], max_lsn: u64) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        #[cfg(any(test, feature = "test-failpoints"))]
        if crate::test_failpoints::WAL_WRITE_FAIL.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::internal("failpoint: WAL write"));
        }

        let mut wal_file = self.wal_file.lock().unwrap();
        if let Some(file) = wal_file.as_mut() {
            file.write_all(data)
                .map_err(|e| Error::internal(format!("failed to write to WAL: {}", e)))?;
        } else {
            return Err(Error::WalFileClosed);
        }
        self.current_file_position
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        if max_lsn > 0 {
            self.flushed_lsn.fetch_max(max_lsn, Ordering::AcqRel);
        }

        Ok(())
    }

    /// Sync WAL to disk (assumes lock is held)
    fn sync_locked(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(Error::WalNotRunning);
        }

        #[cfg(any(test, feature = "test-failpoints"))]
        if crate::test_failpoints::WAL_SYNC_FAIL.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::internal("failpoint: WAL sync"));
        }

        let wal_file = self.wal_file.lock().unwrap();
        if let Some(file) = wal_file.as_ref() {
            file.sync_all()
                .map_err(|e| Error::internal(format!("failed to sync WAL: {}", e)))?;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        self.last_sync_time.store(now, Ordering::Relaxed);

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
                Err(_) => {
                    // No checkpoint.meta yet (fresh DB or first rotation).
                    // LSN 0 means full WAL replay on recovery, which is correct.
                    0
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
            let max_lsn = self.buffer_highest_lsn.swap(0, Ordering::AcqRel);
            // CRITICAL: Increment in-flight counter BEFORE releasing lock
            // This prevents checkpoint from reading LSN before data is on disk
            self.in_flight_writes.fetch_add(1, Ordering::SeqCst);
            (data, max_lsn)
        };

        // Use a guard pattern to ensure we decrement even on error
        let write_result = self.write_to_file(&buffer_data.0, buffer_data.1);
        self.in_flight_writes.fetch_sub(1, Ordering::SeqCst);
        write_result
    }

    /// Periodically flush `SyncMode::None` buffers so SWMR readers can
    /// observe commits without forcing a write on every commit marker.
    /// Returns true when the reader-visible file-write frontier advanced.
    pub fn flush_for_visibility_if_due(&self) -> Result<bool> {
        if self.sync_mode != SyncMode::None {
            return Ok(false);
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let last = self.last_sync_time.load(Ordering::Acquire);
        if self.sync_interval > 0 && now.saturating_sub(last) < self.sync_interval {
            return Ok(false);
        }

        let before = self.flushed_lsn();
        self.flush()?;
        let after = self.flushed_lsn();
        self.last_sync_time.store(now, Ordering::Release);
        Ok(after > before)
    }

    /// Check if we should sync based on operation type
    fn should_sync(&self, op: WALOperationType) -> bool {
        match self.sync_mode {
            SyncMode::None => false,
            SyncMode::Normal => {
                // Always sync on DDL operations (schema changes must be durable)
                if op.is_ddl() {
                    return true;
                }
                // Time-based sync: fsync at most once per second.
                // Committed data survives in the OS buffer cache for most crashes
                // (power failure is the exception). Checkpoint (every 60s) moves
                // data to fsynced volume files for full durability.
                // Max data loss on power failure: ~1 second of commits.
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                let last = self.last_sync_time.load(Ordering::Relaxed);
                now - last >= self.sync_interval
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
    pub fn replay_two_phase<F>(&self, from_lsn: u64, callback: F) -> Result<TwoPhaseRecoveryInfo>
    where
        F: FnMut(WALEntry) -> Result<()>,
    {
        self.replay_two_phase_capped(from_lsn, u64::MAX, callback)
    }

    /// Same as `replay_two_phase` but caps the
    /// max LSN considered. Read-only opens pass the writer's
    /// published `db.shm.visible_commit_lsn` so unpublished
    /// commit markers (writer wrote marker but hasn't yet
    /// `complete_commit`/published) are skipped during attach.
    /// Writable opens (the writer's own recovery) pass
    /// `u64::MAX` (no cap).
    pub fn replay_two_phase_capped<F>(
        &self,
        from_lsn: u64,
        max_lsn: u64,
        mut callback: F,
    ) -> Result<TwoPhaseRecoveryInfo>
    where
        F: FnMut(WALEntry) -> Result<()>,
    {
        // Flush buffer first
        self.flush()?;

        let mut from_lsn = from_lsn;

        // Check for checkpoint — only use checkpoint.lsn when no snapshots were loaded
        // (from_lsn == 0). When snapshots exist, from_lsn already reflects the safe
        // replay point. Using checkpoint.lsn to override would skip WAL entries needed
        // by tables whose snapshots are older (e.g., after a crash during snapshot rename).
        //
        // Capped read-only attach (max_lsn != u64::MAX): never let
        // `checkpoint.lsn` push the floor PAST the reader's cap.
        // The writer's checkpoint.meta reflects post-attach state
        // (the writer raced past us between our shm sample and this
        // load). Filtering segments above the cap already trimmed
        // our cold view; advancing the floor to `checkpoint.lsn > cap`
        // would then leave the WAL replay range empty and silently
        // drop rows committed in (max_kept_visible, cap]. The
        // pre-acquire WAL pin keeps that range intact, so we'd
        // rather start replay from 0 (or wherever the cold view
        // ends) than skip it entirely.
        if from_lsn == 0 {
            let checkpoint_path = self.path.join("checkpoint.meta");
            if let Ok(checkpoint) = CheckpointMetadata::read_from_file(&checkpoint_path) {
                if checkpoint.lsn > from_lsn && checkpoint.lsn <= max_lsn {
                    from_lsn = checkpoint.lsn;
                }
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

        // Sort by embedded LSN for correct replay order.
        // Lexicographic sort would misorder wal- (truncated) and wal_ (rotated)
        // files when both coexist after a crash.
        wal_files.sort_by_key(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .and_then(Self::extract_lsn_from_filename)
                .unwrap_or(0)
        });

        // =====================================================
        // Phase 1: Analysis - Identify transaction outcomes
        // Only collect txn_ids, not full entries (memory efficient)
        // =====================================================
        let mut committed_txns: I64Set = I64Set::new();
        let mut aborted_txns: I64Set = I64Set::new();
        let mut last_lsn = from_lsn;

        for wal_path in &wal_files {
            Self::scan_wal_for_txn_status(
                wal_path,
                from_lsn,
                max_lsn,
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
                let version = header_buf[4];
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

                // Capped read-only attach: stop once the parsed
                // header LSN passes the published frontier.
                // Records beyond `max_lsn` are the live writer's
                // in-flight tail — body/CRC may still be mid-
                // write, and they're explicitly outside this
                // snapshot. The fatal-on-truncated and
                // fatal-on-CRC checks below would otherwise
                // refuse the open just because the writer was
                // mid-write on an unpublished record. WAL files
                // (and entries within a file) are LSN-monotonic,
                // so a single out-of-cap header is the boundary;
                // the outer file loop iterates the next file
                // which short-circuits at its first read too.
                let capped = max_lsn != u64::MAX;
                if capped && lsn > max_lsn {
                    break;
                }

                // Sanity check on size. In capped read-only attach
                // (`max_lsn != u64::MAX`), the writer has already
                // published a frontier covering this byte range — an
                // oversized record below the cap means the writer
                // wrote bytes a downstream reader cannot trust, and
                // skipping silently leaves the snapshot incomplete.
                // Fail the open instead. The header LSN is unverified
                // pre-CRC, so we can't filter by window beyond the
                // above-cap short-circuit; treat any oversized in
                // capped mode as fatal.
                let total_data_size = entry_size + 4;
                if entry_size > 64 * 1024 * 1024 {
                    if capped {
                        return Err(Error::internal(format!(
                            "WAL recovery (capped at LSN {}): oversized record \
                             entry_size={} at header-LSN {}; refusing to skip in \
                             capped read-only attach",
                            max_lsn, entry_size, lsn
                        )));
                    }
                    if !Self::scan_for_magic(&mut file) {
                        break;
                    }
                    continue;
                }

                // ALWAYS read data + verify CRC
                // BEFORE applying the [from_lsn, max_lsn] window.
                // v3 CRC covers header + data, so a successful
                // verify authenticates the LSN we then use for
                // the window check. Without this, a corrupted v3
                // header LSN could move a committed marker
                // outside the cap and Phase 2 would skip it.
                let mut data = vec![0u8; total_data_size];
                if file.read_exact(&mut data).is_err() {
                    if capped {
                        return Err(Error::internal(format!(
                            "WAL recovery (capped at LSN {}): truncated record \
                             at header-LSN {}; refusing to skip in capped \
                             read-only attach",
                            max_lsn, lsn
                        )));
                    }
                    break;
                }
                if WALEntry::verify_crc(version, &header_buf, &data).is_err() {
                    if capped {
                        // Inside a published frontier, the writer's
                        // contract is "every byte below visible_commit_lsn
                        // is authentic." A CRC failure means either the
                        // header LSN is wrong (so we can't trust the
                        // window check to filter this out) or the data
                        // is corrupted within the published range.
                        // Either way, silently skipping would leave the
                        // reader's snapshot missing committed records.
                        return Err(Error::internal(format!(
                            "WAL recovery (capped at LSN {}): CRC failed at \
                             header-LSN {}; refusing to skip in capped \
                             read-only attach",
                            max_lsn, lsn
                        )));
                    }
                    eprintln!(
                        "Warning: WAL recovery Phase 2 CRC failed at header-LSN {}; \
                         skipping",
                        lsn
                    );
                    skipped_count += 1;
                    continue;
                }
                // CRC validated — apply window AFTER validation.
                if lsn < from_lsn || lsn > max_lsn {
                    continue;
                }

                // Decode entry
                match WALEntry::decode(lsn, previous_lsn, flags, version, &data, &header_buf) {
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
                            if committed_txns.contains(entry.txn_id) {
                                callback(entry)?;
                            }
                            continue;
                        }

                        // Apply only committed transactions' data entries
                        if committed_txns.contains(entry.txn_id) {
                            callback(entry)?;
                            applied_count += 1;
                        } else {
                            // Transaction is aborted or in-doubt (no commit marker)
                            // Treat in-doubt as aborted for safety
                            skipped_count += 1;
                        }
                    }
                    Err(e) => {
                        if capped {
                            // CRC just passed, so the bytes are
                            // authentic — a decode failure means the
                            // WAL format itself is incompatible with
                            // this build. Skipping would leave
                            // committed data unapplied inside the
                            // published frontier.
                            return Err(Error::internal(format!(
                                "WAL recovery (capped at LSN {}): decode failed \
                                 at LSN {}: {}; refusing to skip in capped \
                                 read-only attach",
                                max_lsn, lsn, e
                            )));
                        }
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
        max_lsn: u64,
        committed_txns: &mut I64Set,
        aborted_txns: &mut I64Set,
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

            // Parse header
            let version = header_buf[4];
            let flags = WalFlags::from_byte(header_buf[5]);
            let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
            let previous_lsn = u64::from_le_bytes(header_buf[16..24].try_into().unwrap());
            let entry_size = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;

            // Skip any additional header bytes
            if header_size > 32 {
                let extra = header_size - 32;
                if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                    break;
                }
            }

            // Capped read-only attach: stop once the parsed
            // header LSN passes the published frontier — see
            // the matching block in `replay_two_phase_capped`
            // for the full rationale. In short: records beyond
            // `max_lsn` are the live writer's in-flight tail,
            // their body/CRC may not yet be complete, and the
            // fatal-on-truncated / fatal-on-CRC checks below
            // would otherwise refuse the open just because the
            // writer was mid-write.
            let capped = max_lsn != u64::MAX;
            if capped && lsn > max_lsn {
                break;
            }

            // Sanity check on size. In capped read-only attach
            // (`max_lsn != u64::MAX`), an oversized record below the
            // published frontier means the writer wrote bytes a
            // downstream reader cannot trust — silently skipping
            // would let Phase 2's commit/abort decision miss a
            // committed transaction inside the cap. Treat any
            // oversized in capped mode as fatal: the header LSN is
            // unverified pre-CRC, so we can't filter by window
            // beyond the above-cap short-circuit.
            let total_data_size = entry_size + 4;
            if entry_size > 64 * 1024 * 1024 {
                if capped {
                    return Err(Error::internal(format!(
                        "WAL recovery Phase 1 (capped at LSN {}): oversized \
                         record entry_size={} at header-LSN {}; refusing to \
                         skip in capped read-only attach",
                        max_lsn, entry_size, lsn
                    )));
                }
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }

            // ALWAYS read data + verify CRC BEFORE
            // applying the [from_lsn, max_lsn] window. v3 CRC
            // covers header + data, so a successful verify
            // authenticates the LSN we then use for the window
            // check. Otherwise a corrupted v3 header LSN could
            // move a committed marker outside the cap and recovery
            // would skip it without ever validating the CRC.
            let mut data = vec![0u8; total_data_size];
            if file.read_exact(&mut data).is_err() {
                if capped {
                    return Err(Error::internal(format!(
                        "WAL recovery Phase 1 (capped at LSN {}): truncated \
                         record at header-LSN {}; refusing to skip in capped \
                         read-only attach",
                        max_lsn, lsn
                    )));
                }
                break;
            }
            if WALEntry::verify_crc(version, &header_buf, &data).is_err() {
                if capped {
                    // Inside the writer's published frontier the
                    // bytes must be authentic. A CRC failure means
                    // either the header LSN is corrupted (so we
                    // can't trust the window check to filter this
                    // out) or a real DML/marker is unreadable.
                    // Either way Phase 2 would build a wrong
                    // committed_txns set, dropping a transaction
                    // that's actually visible.
                    return Err(Error::internal(format!(
                        "WAL recovery Phase 1 (capped at LSN {}): CRC failed \
                         at header-LSN {} (flags={:?}); refusing to skip in \
                         capped read-only attach",
                        max_lsn, lsn, flags
                    )));
                }
                // CRC failure during recovery: log and skip.
                // Recovery semantics tolerate corruption (the
                // affected entry is effectively aborted /
                // unrecoverable); the writer's checkpoint_lsn
                // is the durable boundary.
                eprintln!(
                    "Warning: WAL recovery CRC failed at header-LSN {} (flags={:?}); \
                     skipping",
                    lsn, flags
                );
                continue;
            }
            // CRC validated — header LSN is authenticated for v3.
            // Apply window AFTER validation.
            if lsn < from_lsn || lsn > max_lsn {
                continue;
            }

            // classify commit/abort markers via the
            // CRC-protected `operation` byte, NOT the unauthenticated
            // header flags. Marker classification is now identical
            // to the SWMR Phase 1 helper.
            match WALEntry::decode(lsn, previous_lsn, flags, version, &data, &header_buf) {
                Ok(entry) => {
                    if entry.is_commit_marker() {
                        committed_txns.insert(entry.txn_id);
                    } else if entry.is_abort_marker() {
                        aborted_txns.insert(entry.txn_id);
                    }
                    // Non-marker entries (DML/DDL): nothing to do
                    // in Phase 1 — Phase 2 reads them based on
                    // the committed_txns set we built here.
                }
                Err(e) => {
                    if capped {
                        // CRC just passed, so the bytes are
                        // authentic — a decode failure means the
                        // WAL format is incompatible with this
                        // build. Skipping would mis-classify a
                        // committed marker as not committed.
                        return Err(Error::internal(format!(
                            "WAL recovery Phase 1 (capped at LSN {}): decode \
                             failed at LSN {}: {}; refusing to skip in capped \
                             read-only attach",
                            max_lsn, lsn, e
                        )));
                    }
                    // Decode failure (despite CRC pass) — layout
                    // bug, log and skip.
                    eprintln!(
                        "Warning: WAL recovery decode failed at LSN {} (flags={:?}): {}; \
                         skipping",
                        lsn, flags, e
                    );
                }
            }

            // Track last LSN
            if lsn > *last_lsn {
                *last_lsn = lsn;
            }
        }

        Ok(())
    }

    /// SWMR v2 Phase E + H: tail committed entries from the WAL
    /// between LSN bounds, without going through full recovery.
    /// Returns DML entries (Insert/Update/Delete) AND DDL entries
    /// (CreateTable/AlterTable/DropTable/etc., as of Phase H) whose
    /// owning transaction has a commit marker with
    /// `marker_lsn <= to_lsn`. Entries from txns with no commit
    /// marker (in-flight or aborted) are excluded.
    ///
    /// The caller decides what to do with DDL — the overlay stores
    /// DDL events separately so the reader can surface a typed
    /// `Error::SwmrPendingDdl` rather than silently mutate the
    /// read-only handle's schema metadata.
    ///
    /// Two-pass like `replay_two_phase` but pure (no callback, no
    /// state mutation, no side effects). Safe to call concurrently
    /// with writes — the writer only appends, so a tail run sees a
    /// consistent prefix.
    ///
    /// `from_lsn` is exclusive and applies to the commit-marker
    /// scan (which transactions are "newly committed in this
    /// window") and to the DDL entry filter (DDL is auto-committed,
    /// so its entry LSN must be > from_lsn to belong to the new
    /// window). `to_lsn` is inclusive — entries with LSN > to_lsn
    /// are skipped.
    ///
    /// The DML entry floor is a SEPARATE parameter `entry_floor` to
    /// support long-running explicit transactions whose DML was
    /// written before `from_lsn` but whose commit marker lands
    /// inside the window. Pass:
    ///   - `entry_floor = 0`: full DML scan (safe but O(WAL)).
    ///   - `entry_floor = writer.oldest_active_txn_lsn at the time
    ///     of the previous tail` (clamped to `<= from_lsn`): O(delta)
    ///     in steady state. Any DML below the snapshotted watermark
    ///     belongs to a transaction that already committed (and was
    ///     applied) before the previous refresh.
    ///
    /// `to_lsn = u64::MAX` tails every committed entry.
    ///
    /// Returns `Err(Error::SwmrSnapshotExpired { .. })` when the
    /// caller's `from_lsn` (or `entry_floor`) falls below the first
    /// live WAL entry on disk. Without this check, a reader whose
    /// lease was reaped (writer truncated the WAL we needed) would
    /// see an empty delta and silently advance `last_applied_lsn`,
    /// hiding committed rows it should have observed. The error tells
    /// the caller to reopen the handle and re-snapshot.
    /// `include_dml`: when `false`, Phase 2 collects ONLY DDL
    /// entries (skipping all per-row Insert/Update/Delete payload
    /// loading + decode). `last_applied_lsn` and
    /// `next_entry_floor` still advance via the normal scan, so
    /// DDL detection and the WAL pin stay current — but readers
    /// that haven't enabled overlay row materialization don't
    /// pay the per-DML decode + allocation cost on every refresh.
    pub fn tail_committed_entries(
        &self,
        from_lsn: u64,
        entry_floor: u64,
        to_lsn: u64,
        include_dml: bool,
        cursor_hint: Option<&WalScanCursor>,
        out_cursor: Option<&mut Option<WalScanCursor>>,
    ) -> Result<Vec<WALEntry>> {
        // Flush buffer first so the on-disk WAL reflects everything
        // committed so far. Otherwise a reader could miss entries the
        // writer just wrote but hasn't fsynced.
        self.flush()?;

        // Collect WAL files (same logic as replay_two_phase).
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
        wal_files.sort_by_key(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .and_then(Self::extract_lsn_from_filename)
                .unwrap_or(0)
        });

        // Chain-head check: if the writer truncated past our needed
        // range, fail with SwmrSnapshotExpired BEFORE scanning. The
        // earliest LSN still on disk is the first LSN of the oldest
        // WAL file. The lower of `from_lsn + 1` (commit-marker
        // window starts AFTER from_lsn) and the Phase 2 DML floor
        // is what we need to be present — missing either would
        // silently lose data.
        //
        // When `entry_floor == 0` Phase 2 explicitly scans from
        // the BEGINNING (the writer's oldest-active watermark was
        // unknown, so `with_baseline` defers to "scan from 0").
        // Treating that case as needing only `from_lsn + 1` lets a
        // reader whose lease was reaped have WAL truncated to a
        // chain head between 1 and `from_lsn + 1`, still find a
        // later commit marker, and silently miss that
        // transaction's earlier DML. Use `1` instead so a chain
        // head above 1 correctly trips SwmrSnapshotExpired.
        if let Some(chain_head) = Self::first_live_wal_lsn(&wal_files) {
            let phase2_floor = if entry_floor > 0 { entry_floor } else { 1 };
            let needed = phase2_floor.min(from_lsn.saturating_add(1));
            if needed < chain_head {
                return Err(Error::SwmrSnapshotExpired {
                    pinned_lsn: needed,
                    chain_head,
                });
            }
        }

        // Phase 1: which txns have a commit marker with marker_lsn in
        // (from_lsn, to_lsn]? Incremental — only NEWLY committed
        // transactions since the caller's prior tail.
        //
        // P2 perf fix: skip rotated WAL files whose entire LSN range
        // is at or below `from_lsn`. Filenames embed each file's
        // first LSN (`wal-{ts}-lsn-{N}.log`); a file is entirely at
        // or below `from_lsn` iff the NEXT file's start LSN is at
        // or below `from_lsn + 1`. The last (current) file always
        // gets scanned — its tail may have just-written entries
        // with no successor file yet to bound it.
        // Per-reader cursor cache: skip CRC-validating bytes
        // we already validated on a prior call. The cursor's
        // path/offset only matches the LAST file in the
        // wal_files chain (the one still being appended); for
        // older static files the existing
        // `wal_files_needed_for_marker_scan` filename-LSN
        // filter handles cold-start skipping. Older files of
        // the chain that need rescanning still go through the
        // full CRC path because they don't match the cursor.
        let phase1_files = Self::wal_files_needed_for_marker_scan(&wal_files, from_lsn);
        let mut committed_txns: I64Set = I64Set::new();
        let mut last_path_p1: Option<PathBuf> = None;
        let mut last_offset_p1: u64 = 0;
        let mut last_lsn_p1: u64 = 0;
        for wal_path in phase1_files {
            // Reuse the cursor only when its LSN is at or below
            // our window's lower bound (`from_lsn`). Phase 1's
            // window is `(from_lsn, to_lsn]`, so records up to
            // and including LSN `from_lsn` are already
            // consumed and safe to skip. A higher
            // `lsn_at_offset` would mean we'd skip past
            // records still in our window.
            let (start, start_lsn) = match cursor_hint {
                Some(c) if c.path == *wal_path && c.lsn_at_offset <= from_lsn => {
                    (c.offset, c.lsn_at_offset)
                }
                _ => (0, 0),
            };
            let (end, end_lsn) = Self::scan_wal_for_committed_in_range(
                wal_path,
                from_lsn,
                to_lsn,
                &mut committed_txns,
                start,
                start_lsn,
            )?;
            last_path_p1 = Some(wal_path.clone());
            last_offset_p1 = end;
            last_lsn_p1 = end_lsn;
        }
        if committed_txns.is_empty() {
            // No new commits — still publish the cursor
            // advance so the next refresh skips the bytes we
            // just CRC-validated.
            if let (Some(out_c), Some(p)) = (out_cursor, last_path_p1) {
                *out_c = Some(WalScanCursor {
                    path: p,
                    offset: last_offset_p1,
                    lsn_at_offset: last_lsn_p1,
                });
            }
            return Ok(Vec::new());
        }

        // Phase 2: scan DML entries with `entry_lsn <= to_lsn` whose
        // txn_id is in `committed_txns`. The DML floor is
        // `entry_floor`, which the caller computes from the writer's
        // published `oldest_active_txn_lsn` at the time of the prior
        // refresh. Anything below that watermark belongs to a
        // transaction that committed before the prior refresh and
        // was already applied (so we can skip the file walk for
        // those entirely).
        //
        // For DDL entries (which all share the synthetic
        // `DDL_TXN_ID = -2`), the per-entry filter `entry.lsn >
        // from_lsn` is enforced inside the helper — DDL is
        // auto-committed at its own LSN so this distinguishes new
        // DDL from stale re-records.
        //
        // P2 perf fix: pre-compute the smallest LSN we may need to
        // scan, then skip whole WAL files whose range is entirely
        // below it. The helper's `from_lsn` semantics are EXCLUSIVE
        // (skip files whose max LSN <= from_lsn), but `entry_floor`
        // is INCLUSIVE (we want entries with LSN >= entry_floor).
        // Pass `entry_floor.saturating_sub(1)` to translate. For
        // entry_floor == 0, saturating_sub gives 0 and the helper
        // short-circuits "scan all".
        let phase2_files =
            Self::wal_files_needed_for_marker_scan(&wal_files, entry_floor.saturating_sub(1));
        let mut out: Vec<WALEntry> = Vec::new();
        let mut last_path_p2: Option<PathBuf> = None;
        let mut last_offset_p2: u64 = 0;
        let mut last_lsn_p2: u64 = 0;
        for wal_path in phase2_files {
            // Phase 2 needs records with `lsn >= entry_floor`,
            // so cursor reuse requires the cached LSN to be
            // STRICTLY less than `entry_floor` — equality means
            // a record at exactly `entry_floor` is already past
            // the cursor and would be missed. The cold-start
            // case (cursor LSN == 0, entry_floor == 0) falls
            // out naturally since the cursor offset is also 0.
            let (start, start_lsn) = match cursor_hint {
                Some(c) if c.path == *wal_path && c.lsn_at_offset < entry_floor => {
                    (c.offset, c.lsn_at_offset)
                }
                _ => (0, 0),
            };
            let (end, end_lsn) = Self::collect_committed_dml_in_range(
                wal_path,
                from_lsn,
                entry_floor,
                to_lsn,
                &committed_txns,
                &mut out,
                include_dml,
                start,
                start_lsn,
            )?;
            last_path_p2 = Some(wal_path.clone());
            last_offset_p2 = end;
            last_lsn_p2 = end_lsn;
        }
        // Publish the new cursor: pick whichever phase covered
        // more of the last file. Both phases produce in-window
        // cursors, so taking the MAX preserves the most cache
        // for the next refresh.
        if let Some(out_c) = out_cursor {
            *out_c = match (last_path_p2.clone(), last_path_p1) {
                (Some(p2), Some(p1)) if p2 == p1 => {
                    let (offset, lsn) = if last_offset_p2 >= last_offset_p1 {
                        (last_offset_p2, last_lsn_p2)
                    } else {
                        (last_offset_p1, last_lsn_p1)
                    };
                    Some(WalScanCursor {
                        path: p2,
                        offset,
                        lsn_at_offset: lsn,
                    })
                }
                (Some(p), _) => Some(WalScanCursor {
                    path: p,
                    offset: last_offset_p2,
                    lsn_at_offset: last_lsn_p2,
                }),
                (None, Some(p)) => Some(WalScanCursor {
                    path: p,
                    offset: last_offset_p1,
                    lsn_at_offset: last_lsn_p1,
                }),
                (None, None) => None,
            };
        }
        // Stable sort by LSN — writer emits in LSN order per file but
        // multiple files may interleave only at file boundaries, so
        // sorting is cheap (already mostly-sorted).
        out.sort_by_key(|e| e.lsn);
        Ok(out)
    }

    /// SWMR v2 P2 perf fix: pick the subset of WAL files whose LSN
    /// range can possibly contain a record with `lsn > from_lsn`.
    /// The last file is always included — its tail may contain
    /// just-written entries with no successor file yet to bound it.
    ///
    /// The rotation naming convention is
    /// `wal_NNN-ts-lsn-{N}.log` where `N == current_lsn at
    /// rotation time` (see `rotate_wal`). `current_lsn` is the LSN
    /// of the LAST entry written to the OLD file, NOT the first
    /// LSN of the NEW file. So
    /// `extract_lsn_from_filename(files[i+1])` gives the LAST LSN
    /// of `files[i]`. We skip `files[i]` only when ALL its entries
    /// are `<= from_lsn`, i.e. when its last LSN `<= from_lsn`.
    /// The prior `<= from_lsn + 1` threshold was off-by-one and
    /// dropped a file whose last entry was exactly `from_lsn + 1`
    /// — that LSN is in the `(from_lsn, to_lsn]` window and must
    /// be scanned.
    ///
    /// Files must already be sorted by extracted LSN (the
    /// caller's invariant). For the typical SWMR case of a single
    /// live WAL file, this returns the whole slice; for multi-file
    /// WAL chains the marker scan only walks the relevant tail.
    fn wal_files_needed_for_marker_scan(files: &[PathBuf], from_lsn: u64) -> &[PathBuf] {
        if files.len() <= 1 || from_lsn == 0 {
            return files;
        }
        let mut skip_count = 0usize;
        for i in 0..files.len() - 1 {
            let next_file_lsn = files[i + 1]
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(Self::extract_lsn_from_filename)
                .unwrap_or(0);
            // `next_file_lsn` is the last LSN of `files[i]`.
            // Skip `files[i]` only if that last LSN `<= from_lsn`
            // (every entry in `files[i]` is at or below `from_lsn`).
            if next_file_lsn <= from_lsn {
                skip_count = i + 1;
            } else {
                break;
            }
        }
        &files[skip_count..]
    }

    /// SWMR v2 Phase E helper: Phase-1 scan that adds to `out` any
    /// txn_id whose commit marker has `from_lsn < marker_lsn <= to_lsn`.
    ///
    /// `start_offset`: byte offset at which to begin scanning.
    /// Bytes in `[0, start_offset)` are assumed to have been CRC-
    /// validated by a prior scan with the SAME reader cursor. The
    /// caller passes `0` when no cursor is available (cold start
    /// or rotation).
    ///
    /// Returns the byte offset just past the last record this scan
    /// successfully processed (or the start_offset on error / EOF
    /// at start). The caller stores this back into the per-reader
    /// cursor so the next refresh can resume from here.
    fn scan_wal_for_committed_in_range(
        wal_path: &Path,
        from_lsn: u64,
        to_lsn: u64,
        out: &mut I64Set,
        start_offset: u64,
        start_offset_lsn: u64,
    ) -> Result<(u64, u64)> {
        let mut file = match File::open(wal_path) {
            Ok(f) => f,
            Err(_) => return Ok((start_offset, start_offset_lsn)),
        };
        if start_offset > 0 && file.seek(SeekFrom::Start(start_offset)).is_err() {
            // Seek failure (file shorter than cursor) — give up on
            // the cache and rescan from the beginning.
            if file.seek(SeekFrom::Start(0)).is_err() {
                return Ok((0, 0));
            }
        }
        // Cursor advances only for records WITHIN the
        // `(from_lsn, to_lsn]` window. Records read beyond
        // `to_lsn` are CRC-validated this scan (cost paid) but
        // not "owned" by this scan's window — the next refresh
        // may have a higher to_lsn that needs to re-process
        // them, so leaving them outside the cursor preserves
        // correctness. `last_in_window_lsn` tracks the LSN of
        // the last in-window record so the caller can refuse
        // cursor reuse on a window whose lower bound moved
        // backwards (e.g. entry_floor regression for Phase 2).
        let mut last_good_offset = start_offset;
        let mut last_in_window_lsn = start_offset_lsn;
        loop {
            let mut header_buf = [0u8; 32];
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(_) => break,
            }
            let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
            if magic != WAL_ENTRY_MAGIC {
                let _ = file.seek(SeekFrom::Current(-32));
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }
            let version = header_buf[4];
            let flags = WalFlags::from_byte(header_buf[5]);
            let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
            let previous_lsn = u64::from_le_bytes(header_buf[16..24].try_into().unwrap());
            let entry_size = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;
            if header_size > 32 {
                let extra = header_size - 32;
                if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                    break;
                }
            }
            let total_data_size = entry_size + 4;
            // an in-window oversized record is
            // corruption, not "scan forward and resync". Treat
            // out-of-window oversized records as the prior code
            // did (resync), but in-window must error.
            let lsn_in_window = lsn > from_lsn && lsn <= to_lsn;
            if entry_size > 64 * 1024 * 1024 {
                if lsn_in_window {
                    return Err(Error::internal(format!(
                        "WAL Phase-1 oversized record at LSN {} \
                         (entry_size={} > 64 MiB, in window ({}, {}])",
                        lsn, entry_size, from_lsn, to_lsn
                    )));
                }
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }
            // ALWAYS read data + verify CRC BEFORE
            // trusting the header LSN for windowing. Without
            // this, a corrupted header LSN could move an in-window
            // record out of window, the scanner would `seek` past
            // it (cheap-skip), and the corruption would never
            // surface — the overlay would advance past a missing
            // commit/DML.
            //
            // For v3 entries the CRC covers header + data, so a
            // successful verify_crc authenticates the header LSN
            // (and flags, version, etc.) we're about to use. For
            // v2 entries the header remains unauthenticated
            // (a known limitation of those formats); we still
            // verify the data-only CRC.
            //
            // After CRC validates: in-window entries get full
            // decode + commit-marker classification via the
            // CRC-protected `operation` byte; out-of-window
            // entries are skipped.
            let mut data = vec![0u8; total_data_size];
            if let Err(e) = file.read_exact(&mut data) {
                if lsn_in_window {
                    return Err(Error::internal(format!(
                        "WAL Phase-1 short read at LSN {} (in window ({}, {}], \
                         expected {} bytes): {}",
                        lsn, from_lsn, to_lsn, total_data_size, e
                    )));
                }
                break;
            }
            if let Err(e) = WALEntry::verify_crc(version, &header_buf, &data) {
                return Err(Error::internal(format!(
                    "WAL Phase-1 CRC verify failed at header-LSN {} (in window \
                     ({}, {}]={}): {}",
                    lsn, from_lsn, to_lsn, lsn_in_window, e
                )));
            }
            // CRC passed — record bytes are authenticated up to
            // and including this entry. Advance the cursor only
            // when this record is INSIDE our window, so the
            // next scan with a higher `to_lsn` correctly
            // re-processes records that were beyond our cap.
            if lsn_in_window {
                if let Ok(pos) = file.stream_position() {
                    last_good_offset = pos;
                    last_in_window_lsn = lsn;
                }
            }
            // CRC passed — for v3 the header LSN is authenticated.
            // For v2 we trust it under the prior-format risk
            // model.
            //
            // WAL entries are LSN-monotonic within and across
            // files (writer only appends), so once we observe
            // an LSN above `to_lsn` no later record in this
            // file can land back in the window. Break instead
            // of `continue` to avoid re-CRC-validating the
            // entire unpublished tail on every refresh — that
            // also stops surfacing CRC / layout errors from
            // records beyond this snapshot's published cap.
            // Records BELOW `from_lsn` (left half of the
            // window) still need the `continue` path because
            // we may not have started reading at the exact
            // window lower bound.
            if lsn > to_lsn {
                break;
            }
            if !lsn_in_window {
                continue;
            }
            // Phase 1 only needs `(txn_id, operation)` to find
            // commit markers — skip the full `decode` (payload
            // allocation, decompression, schema/value parsing) for
            // every other entry. `peek_classification` reads the
            // CRC-validated bytes at fixed offsets only.
            if let Some((txn_id, op)) = WALEntry::peek_classification(&data) {
                if matches!(op, WALOperationType::Commit) {
                    out.insert(txn_id);
                }
            } else {
                // Layout corruption — fall back to full decode so
                // the error message is precise.
                let _ = WALEntry::decode(lsn, previous_lsn, flags, version, &data, &header_buf)
                    .map_err(|e| {
                        Error::internal(format!(
                            "WAL Phase-1 decode failed at LSN {} (in window \
                             ({}, {}]): {}",
                            lsn, from_lsn, to_lsn, e
                        ))
                    })?;
            }
        }
        Ok((last_good_offset, last_in_window_lsn))
    }

    /// SWMR v2 Phase E helper: Phase-2 scan that decodes each DML
    /// entry whose `txn_id` is in `committed` AND whose `lsn` falls
    /// in `(entry_floor, to_lsn]`. The DDL-specific filter
    /// `entry.lsn > from_lsn` is also applied so stale DDL re-records
    /// don't replay.
    ///
    /// `start_offset` works the same way as in
    /// `scan_wal_for_committed_in_range`: bytes in
    /// `[0, start_offset)` are assumed CRC-validated by a prior
    /// call from the same reader cursor and are skipped without
    /// re-reading. Returns the byte offset just past the last
    /// CRC-validated record so the caller can advance the cursor.
    #[allow(clippy::too_many_arguments)]
    fn collect_committed_dml_in_range(
        wal_path: &Path,
        from_lsn: u64,
        entry_floor: u64,
        to_lsn: u64,
        committed: &I64Set,
        out: &mut Vec<WALEntry>,
        include_dml: bool,
        start_offset: u64,
        start_offset_lsn: u64,
    ) -> Result<(u64, u64)> {
        let mut file = match File::open(wal_path) {
            Ok(f) => f,
            Err(_) => return Ok((start_offset, start_offset_lsn)),
        };
        if start_offset > 0
            && file.seek(SeekFrom::Start(start_offset)).is_err()
            && file.seek(SeekFrom::Start(0)).is_err()
        {
            return Ok((0, 0));
        }
        // See `scan_wal_for_committed_in_range` for the cursor-
        // tracking rationale: only update on records inside
        // `[entry_floor, to_lsn]`, so a future scan with a
        // higher `to_lsn` (or a lower `entry_floor`) can still
        // re-process records the current scan walked past
        // without owning.
        let mut last_good_offset = start_offset;
        let mut last_in_window_lsn = start_offset_lsn;
        loop {
            let mut header_buf = [0u8; 32];
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(_) => break,
            }
            let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
            if magic != WAL_ENTRY_MAGIC {
                let _ = file.seek(SeekFrom::Current(-32));
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }
            let version = header_buf[4];
            let flags = WalFlags::from_byte(header_buf[5]);
            let header_size = u16::from_le_bytes(header_buf[6..8].try_into().unwrap()) as usize;
            let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
            let previous_lsn = u64::from_le_bytes(header_buf[16..24].try_into().unwrap());
            let entry_size = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;
            if header_size > 32 {
                let extra = header_size - 32;
                if file.seek(SeekFrom::Current(extra as i64)).is_err() {
                    break;
                }
            }
            let total_data_size = entry_size + 4;
            // an in-window oversized record is
            // corruption, not "scan forward and resync". Treat
            // out-of-window oversized records as the prior code
            // did (resync), but in-window must error.
            let lsn_in_window = lsn >= entry_floor && lsn <= to_lsn;
            if entry_size > 64 * 1024 * 1024 {
                if lsn_in_window {
                    return Err(Error::internal(format!(
                        "WAL Phase-2 oversized record at LSN {} \
                         (entry_size={} > 64 MiB, in window [{}, {}])",
                        lsn, entry_size, entry_floor, to_lsn
                    )));
                }
                if !Self::scan_for_magic(&mut file) {
                    break;
                }
                continue;
            }
            // ALWAYS read data + verify CRC BEFORE
            // trusting the header LSN for windowing. v3 CRC covers
            // header + data, so a successful verify authenticates
            // the LSN we're about to use for the in-window check.
            // For v2 the header is unauthenticated (prior-format
            // limitation); we still do the data-only CRC.
            //
            // After CRC validates: in-window entries get full
            // decode + commit/abort/DML classification via the
            // CRC-protected `operation` byte; out-of-window
            // entries are skipped.
            let mut data = vec![0u8; total_data_size];
            if let Err(e) = file.read_exact(&mut data) {
                if lsn_in_window {
                    return Err(Error::internal(format!(
                        "WAL Phase-2 short read at LSN {} (in window [{}, {}], \
                         expected {} bytes): {}",
                        lsn, entry_floor, to_lsn, total_data_size, e
                    )));
                }
                break;
            }
            if let Err(e) = WALEntry::verify_crc(version, &header_buf, &data) {
                return Err(Error::internal(format!(
                    "WAL Phase-2 CRC verify failed at header-LSN {} (in window \
                     [{}, {}]={}): {}",
                    lsn, entry_floor, to_lsn, lsn_in_window, e
                )));
            }
            // CRC passed — advance cursor only for in-window
            // records so a future scan with a wider window can
            // re-process records this scan walked past.
            if lsn_in_window {
                if let Ok(pos) = file.stream_position() {
                    last_good_offset = pos;
                    last_in_window_lsn = lsn;
                }
            }
            // CRC passed — for v3 the header LSN is now
            // authenticated. Apply windowing.
            //
            // WAL is append-only and LSN-monotonic, so an
            // LSN above `to_lsn` means every remaining
            // record in this file is also above the window.
            // Break to stop re-CRC-validating the
            // unpublished tail on every refresh — that also
            // prevents surfacing CRC / layout errors from
            // records beyond this snapshot's published cap.
            // Records BELOW `from_lsn` still need `continue`
            // because we may not have started reading at the
            // exact window lower bound.
            if lsn > to_lsn {
                break;
            }
            if !lsn_in_window {
                continue;
            }
            // DDL-only mode: peek txn_id + operation BEFORE the
            // full decode. Skip non-DDL entries (the common case
            // under write-heavy workloads) without paying the
            // payload-allocation + decompression + schema-decode
            // cost. The peek is cheap: it reads CRC-validated
            // bytes at fixed offsets.
            if !include_dml {
                if let Some((txn_id, op)) = WALEntry::peek_classification(&data) {
                    // Skip non-DDL fast: DDL-only mode discards
                    // every Insert/Update/Delete + marker.
                    if !op.is_ddl() {
                        continue;
                    }
                    // In-doubt / aborted txns: skip without full
                    // decode.
                    if !committed.contains(txn_id) {
                        continue;
                    }
                    // DDL stale re-record (lsn <= from_lsn): same
                    // filter as the full-decode branch — applies
                    // ONLY to auto-committed DDL_TXN_ID entries.
                    // User-txn DDL is uniquely tagged by txn_id;
                    // its commit marker may sit in this window
                    // even when the entry itself is below
                    // `from_lsn`, so applying the LSN filter
                    // unconditionally would silently drop a
                    // committed user-txn DDL.
                    if txn_id == crate::storage::mvcc::persistence::DDL_TXN_ID && lsn <= from_lsn {
                        continue;
                    }
                    // Surviving DDL falls through to full decode
                    // so the entry has its payload populated for
                    // `pending_ddl` (CreateIndex needs the
                    // IndexMetadata payload to extract the index
                    // name).
                }
                // peek_classification == None → fall through to
                // full decode for a precise error.
            }
            // ALWAYS read the full payload + decode
            // (CRC validate) for every in-window entry. CRC was
            // already verified above; this decode parses fields.
            //
            // The pre-decode `total_data_size < 8` layout check
            // remains: a valid entry's data section is at least 8
            // bytes (txn_id alone), so anything smaller is layout
            // corruption inside the window.
            if total_data_size < 8 {
                return Err(Error::internal(format!(
                    "WAL Phase-2 layout corruption at LSN {}: total_data_size={} \
                     (less than minimum 8 bytes for txn_id), in window \
                     [{}, {}]",
                    lsn, total_data_size, entry_floor, to_lsn
                )));
            }
            match WALEntry::decode(lsn, previous_lsn, flags, version, &data, &header_buf) {
                Ok(entry) => {
                    if entry.is_marker_entry() {
                        continue;
                    }
                    // also skip commit/abort
                    // markers. With the header-flag pre-skip
                    // removed (so flag-bit corruption can't
                    // silently drop in-window DML), markers now
                    // reach the decode branch. Classify via the
                    // CRC-protected `operation` byte.
                    if entry.is_commit_marker() || entry.is_abort_marker() {
                        continue;
                    }
                    // Skip this committed-set check AFTER decode so
                    // CRC has been validated for every in-window
                    // entry — even ones we end up not applying.
                    if !committed.contains(entry.txn_id) {
                        continue;
                    }
                    // Auto-committed DDL entries reuse
                    // `DDL_TXN_ID`, so the txn_id-set check alone
                    // can't tell "new DDL" from "old DDL" when a
                    // fresh DDL marker arrives. Reject DDL whose
                    // entry LSN is at or below `from_lsn` ONLY for
                    // DDL_TXN_ID entries — every auto-committed
                    // DDL is immediately followed by its own
                    // commit marker, so its entry LSN must be >
                    // from_lsn to belong to this refresh's window.
                    //
                    // Transactional DDL written under a USER txn
                    // id (deferred-DDL Phase 2.5) is uniquely
                    // tagged: the txn_id-set check alone is
                    // sufficient to filter stale entries. Apply
                    // the LSN filter would WRONGLY skip user-txn
                    // DDL whose entry LSN sits below from_lsn but
                    // whose commit marker is in the current
                    // refresh window (legitimate when the user
                    // txn opened many entries before this
                    // refresh's tail boundary).
                    if entry.operation.is_ddl()
                        && entry.txn_id == crate::storage::mvcc::persistence::DDL_TXN_ID
                        && entry.lsn <= from_lsn
                    {
                        continue;
                    }
                    // DDL-only mode: skip non-DDL entries to avoid
                    // the per-row decode + payload allocation cost
                    // for readers that haven't enabled overlay row
                    // materialization. The cursor still advances
                    // (handled in `rebuild_from_wal` via to_lsn) and
                    // DDL surfaces SwmrPendingDdl as before.
                    if !include_dml && !entry.operation.is_ddl() {
                        continue;
                    }
                    // Phase H: non-stale DDL passes through. The
                    // overlay segregates DDL into a separate list so
                    // the reader's refresh can surface SwmrPendingDdl
                    // while keeping DML rows applied to the per-table
                    // overlay.
                    out.push(entry);
                }
                Err(e) => {
                    // every in-window entry must
                    // decode successfully. A CRC failure here is
                    // corruption inside the published window —
                    // silently skipping would drop a possibly
                    // committed row from the overlay.
                    return Err(Error::internal(format!(
                        "WAL Phase-2 decode failed at LSN {} (in window \
                         [{}, {}], entry must validate): {}",
                        lsn, entry_floor, to_lsn, e
                    )));
                }
            }
        }
        Ok((last_good_offset, last_in_window_lsn))
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
        use crate::common::time_compat::Instant;

        let deadline = Instant::now() + timeout;
        #[cfg(not(target_arch = "wasm32"))]
        let mut sleep_duration = std::time::Duration::from_micros(10);
        #[cfg(not(target_arch = "wasm32"))]
        const MAX_SLEEP: std::time::Duration = std::time::Duration::from_millis(10);

        while self.in_flight_writes.load(Ordering::SeqCst) > 0 {
            if Instant::now() > deadline {
                return Err(Error::internal(format!(
                    "timeout waiting for in-flight WAL writes to complete ({} still pending)",
                    self.in_flight_writes.load(Ordering::SeqCst)
                )));
            }

            // Exponential backoff with cap
            #[cfg(not(target_arch = "wasm32"))]
            {
                std::thread::sleep(sleep_duration);
                sleep_duration = std::cmp::min(sleep_duration * 2, MAX_SLEEP);
            }
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

        #[cfg(any(test, feature = "test-failpoints"))]
        if crate::test_failpoints::CHECKPOINT_WRITE_FAIL.load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(Error::internal("failpoint: checkpoint write"));
        }

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

        // Flush buffer to file (while still running)
        self.flush()?;

        // Fsync to ensure all WAL data is durable on disk.
        // Without this, a power failure or kill -9 after close
        // could lose buffered WAL entries.
        self.sync_locked()?;

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

    /// Clean up old rotated WAL files fully covered by a snapshot.
    ///
    /// A WAL file named `lsn-N` contains entries with LSN > N. The file's upper
    /// bound is the NEXT file's start LSN (sorted by embedded LSN). A file is safe
    /// to delete only when ALL its entries are <= `up_to_lsn`, which means the next
    /// file's start LSN must be <= `up_to_lsn`.
    ///
    /// Without this boundary check, files containing entries that straddle
    /// `up_to_lsn` would be deleted, causing data loss if the latest snapshot is
    /// corrupted and recovery falls back to the second-to-last snapshot + WAL.
    fn cleanup_old_wal_files(wal_dir: &Path, current_wal_name: &str, up_to_lsn: u64) {
        let dir_entries = match fs::read_dir(wal_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        // Collect all WAL files with their embedded LSN
        let mut wal_files: Vec<(PathBuf, u64)> = Vec::new();
        for entry in dir_entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if !((name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log")) {
                continue;
            }
            if let Some(lsn) = Self::extract_lsn_from_filename(&name) {
                wal_files.push((entry.path(), lsn));
            }
        }

        // Sort by embedded LSN so we can determine each file's upper bound
        wal_files.sort_by_key(|&(_, lsn)| lsn);

        // Delete file[i] only if its upper bound (= file[i+1].lsn) <= up_to_lsn.
        // The last file in the list is the current WAL — never delete it.
        for i in 0..wal_files.len() {
            let (ref path, _) = wal_files[i];
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            // Never delete the current WAL file
            if name == current_wal_name {
                continue;
            }

            // Need a next file to determine upper bound
            let next_lsn = if i + 1 < wal_files.len() {
                wal_files[i + 1].1
            } else {
                // Last non-current file with no successor — keep it (can't prove coverage)
                continue;
            };

            // The file's entries span (file_lsn, next_lsn]. Safe to delete only if
            // all entries are covered: next_lsn <= up_to_lsn.
            if next_lsn <= up_to_lsn {
                if let Err(e) = fs::remove_file(path) {
                    eprintln!(
                        "Warning: Could not remove old rotated WAL file {:?}: {}",
                        path, e
                    );
                }
            }
        }
    }

    /// First live WAL LSN: the LSN of the first entry in the oldest
    /// WAL file on disk. Returns `None` when there are no WAL files
    /// (fresh DB, never written) or when the oldest file is empty /
    /// unreadable. Used by `tail_committed_entries` to detect WAL
    /// truncation past a reader's needed range.
    ///
    /// `wal_files` MUST be sorted by file-name LSN (caller's
    /// responsibility — same sort the rest of the tail / replay
    /// paths use).
    ///
    /// Reads only the first 16 bytes of the oldest file (magic +
    /// version + flags + header_size + LSN), so this is cheap.
    fn first_live_wal_lsn(wal_files: &[PathBuf]) -> Option<u64> {
        let oldest = wal_files.first()?;
        let mut file = File::open(oldest).ok()?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header).ok()?;
        let magic = u32::from_le_bytes(header[0..4].try_into().ok()?);
        if magic != WAL_ENTRY_MAGIC {
            return None;
        }
        // Header layout (see WALEntry::encode): magic[4] + version[1]
        // + flags[1] + header_size[2] + lsn[8] = 16 bytes.
        Some(u64::from_le_bytes(header[8..16].try_into().ok()?))
    }

    /// Extract the LSN from a WAL filename containing the pattern `lsn-{N}`.
    fn extract_lsn_from_filename(name: &str) -> Option<u64> {
        let lsn_start = name.find("lsn-")?;
        let lsn_str = &name[lsn_start + 4..];
        let dot_pos = lsn_str.find('.')?;
        lsn_str[..dot_pos].parse::<u64>().ok()
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

        // Clean up old rotated WAL files covered by the snapshot.
        // This runs before the early-return check because even if the current WAL file
        // doesn't need truncation, previously-rotated files may be fully covered.
        Self::cleanup_old_wal_files(&self.path, &current_wal_name, up_to_lsn);

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
                let max_lsn = self.buffer_highest_lsn.swap(0, Ordering::AcqRel);
                if let Some(file) = wal_file_guard.as_mut() {
                    file.write_all(&buffer_data).map_err(|e| {
                        Error::internal(format!("failed to flush buffer during truncation: {}", e))
                    })?;
                    self.current_file_position
                        .fetch_add(buffer_data.len() as u64, Ordering::Relaxed);
                    if max_lsn > 0 {
                        self.flushed_lsn.fetch_max(max_lsn, Ordering::AcqRel);
                    }
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
                lsn: up_to_lsn.saturating_add(1),
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
                commit_seq: 0, // chain-break marker, not a real commit
            };
            let encoded = marker_entry.encode();
            temp_wal_file
                .write_all(&encoded)
                .map_err(|e| Error::internal(format!("failed to write marker entry: {}", e)))?;

            // Track marker's LSN and size for chain continuity
            last_copied_lsn = up_to_lsn.saturating_add(1);
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

        // Step 5: Sync directory to ensure renames are durable.
        // This is critical on filesystems like ext4 where rename durability
        // requires directory sync. Without this, a crash after rename but
        // before natural sync could result in the old filename persisting.
        // Windows does not support opening directories for fsync.
        #[cfg(not(windows))]
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
        self.flushed_lsn
            .fetch_max(last_copied_lsn, Ordering::AcqRel);

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
        assert_eq!(
            WALOperationType::from_u8(13),
            Some(WALOperationType::TruncateTable)
        );
        assert_eq!(WALOperationType::from_u8(14), None); // ColdDelete removed

        assert!(WALOperationType::CreateTable.is_ddl());
        assert!(WALOperationType::CreateView.is_ddl());
        assert!(WALOperationType::DropView.is_ddl());
        assert!(WALOperationType::TruncateTable.is_ddl());
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

        // Data starts at offset 32, includes CRC at end. v3 CRC
        // covers header + data, so we pass the encoded[..32]
        // header bytes through to decode.
        let decoded = WALEntry::decode(
            entry.lsn,
            entry.previous_lsn,
            flags,
            WAL_FORMAT_VERSION,
            &encoded[32..],
            &encoded[..32],
        )
        .unwrap();

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

        // Decode should fail due to CRC mismatch. v3 CRC covers
        // header + data; corruption at offset 40 still falls
        // inside the protected region.
        let result = WALEntry::decode(
            entry.lsn,
            entry.previous_lsn,
            entry.flags,
            WAL_FORMAT_VERSION,
            &encoded[32..],
            &encoded[..32],
        );
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
            commit_seq: 0,
        };
        assert!(marker.is_marker_entry());

        let normal = WALEntry::new(1, "test".to_string(), 1, WALOperationType::Insert, vec![]);
        assert!(!normal.is_marker_entry());
    }

    #[test]
    fn test_wal_entry_commit_rollback() {
        let commit = WALEntry::commit(100, 0);
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
        let commit_marker = WALEntry::commit_marker(42, 0);
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
        let regular_commit = WALEntry::commit(44, 0);
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
        wal.write_commit_marker(1, 0).unwrap();

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

        let entry2 = WALEntry::commit(1, 0);
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
                wal.write_commit_marker(i, 0).unwrap();
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
    fn test_sync_none_commit_marker_waits_for_buffer_flush() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal_none_flush_cap");
        let wal = WALManager::new(&wal_path, SyncMode::None).unwrap();

        let marker_lsn = wal.write_commit_marker(1, 7).unwrap();
        assert_eq!(wal.current_lsn(), marker_lsn);
        assert_eq!(
            wal.flushed_lsn(),
            0,
            "SyncMode::None must not force-flush commit markers"
        );

        wal.flush().unwrap();
        assert_eq!(wal.flushed_lsn(), marker_lsn);
        wal.close().unwrap();
    }

    #[test]
    fn test_sync_normal_commit_marker_still_flushes() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal_normal_flush_cap");
        let wal = WALManager::new(&wal_path, SyncMode::Normal).unwrap();

        let marker_lsn = wal.write_commit_marker(1, 7).unwrap();
        assert_eq!(wal.flushed_lsn(), marker_lsn);
        wal.close().unwrap();
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

        let commit = WALEntry::commit(1, 0);
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
                wal.write_commit_marker(i, 0).unwrap();
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
                wal.write_commit_marker(i, 0).unwrap();
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
            wal.write_commit_marker(1, 0).unwrap();

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
            wal.write_commit_marker(1, 0).unwrap();

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
            wal.write_commit_marker(4, 0).unwrap();

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

        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config), false).unwrap();

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

        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config), false).unwrap();

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
            wal.write_commit_marker(i, 0).unwrap();
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
            wal.write_commit_marker(i, 0).unwrap();
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
        let wal = WALManager::with_config(&wal_path, SyncMode::Full, Some(&config), false).unwrap();

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
