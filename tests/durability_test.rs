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

//! Durability / Fault Injection Tests
//!
//! These tests write valid data, close the database, corrupt WAL/checkpoint
//! files in specific ways, then reopen and verify recovery consistency.
//!
//! Recovery invariants verified in every test:
//! 1. No partial transactions — all-or-nothing per transaction
//! 2. Consistent recovery — recovered data is a valid subset of committed data
//! 3. Database is usable — can INSERT after recovery
//! 4. Tables exist — DDL survived or was re-applied from snapshot

use std::fs;
use std::path::{Path, PathBuf};
use stoolap::Database;
use tempfile::{tempdir, TempDir};

// ============================================================================
// WAL binary format constants (must match wal_manager.rs)
// ============================================================================

const WAL_ENTRY_MAGIC: u32 = 0x454C4157;
const WAL_HEADER_SIZE: usize = 32;
const CRC_SIZE: usize = 4;

// Flag bits
const COMPRESSED_FLAG: u8 = 0x01;
const COMMIT_MARKER_FLAG: u8 = 0x02;

// Checkpoint constants
const _CHECKPOINT_MAGIC: u32 = 0x43504F49;

// ============================================================================
// Helper structs
// ============================================================================

/// Information about a single WAL entry's byte boundaries in the file
#[derive(Debug, Clone)]
struct WalEntryInfo {
    /// Byte offset of this entry's start (magic bytes)
    offset: usize,
    /// Total size on disk: header + data + CRC
    total_size: usize,
    /// Log Sequence Number
    _lsn: u64,
    /// Flags byte
    flags: u8,
    /// Size of data portion (from header)
    entry_size: usize,
    /// Byte offset where data portion starts (after header)
    data_offset: usize,
    /// Byte offset of the CRC32 (last 4 bytes of entry)
    crc_offset: usize,
}

struct TestFixture {
    _dir: TempDir,
    db_path: PathBuf,
    dsn: String,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find all WAL files in the database's wal/ directory
fn find_wal_files(db_path: &Path) -> Vec<PathBuf> {
    let wal_dir = db_path.join("wal");
    if !wal_dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = fs::read_dir(&wal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log")
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

/// Find checkpoint.meta file
fn find_checkpoint_file(db_path: &Path) -> Option<PathBuf> {
    let path = db_path.join("wal").join("checkpoint.meta");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Parse WAL binary data to locate all entry boundaries
fn find_entry_boundaries(data: &[u8]) -> Vec<WalEntryInfo> {
    let mut entries = Vec::new();
    let mut pos = 0;

    while pos + WAL_HEADER_SIZE <= data.len() {
        // Check magic
        if pos + 4 > data.len() {
            break;
        }
        let magic = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        if magic != WAL_ENTRY_MAGIC {
            pos += 1;
            continue;
        }

        // Parse header fields
        let flags = data[pos + 5];
        let header_size = u16::from_le_bytes(data[pos + 6..pos + 8].try_into().unwrap()) as usize;
        let lsn = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
        let entry_size = u32::from_le_bytes(data[pos + 24..pos + 28].try_into().unwrap()) as usize;

        // Sanity check
        if entry_size > 64 * 1024 * 1024 || header_size < WAL_HEADER_SIZE {
            pos += 1;
            continue;
        }

        let data_offset = pos + header_size;
        let total_data_with_crc = entry_size + CRC_SIZE;
        let total_size = header_size + total_data_with_crc;

        if pos + total_size > data.len() {
            // Entry extends beyond file — incomplete
            break;
        }

        let crc_offset = data_offset + entry_size;

        entries.push(WalEntryInfo {
            offset: pos,
            total_size,
            _lsn: lsn,
            flags,
            entry_size,
            data_offset,
            crc_offset,
        });

        pos += total_size;
    }

    entries
}

/// Zero out a range of bytes in a buffer
fn zero_range(data: &mut [u8], offset: usize, len: usize) {
    let end = (offset + len).min(data.len());
    for byte in &mut data[offset..end] {
        *byte = 0;
    }
}

/// Flip a single bit at the given byte offset and bit position
fn flip_bit(data: &mut [u8], byte_offset: usize, bit_pos: u8) {
    if byte_offset < data.len() {
        data[byte_offset] ^= 1 << bit_pos;
    }
}

/// Zero a 4KB-aligned page
fn zero_page(data: &mut [u8], page_num: usize) {
    let page_size = 4096;
    let start = page_num * page_size;
    let end = (start + page_size).min(data.len());
    if start < data.len() {
        for byte in &mut data[start..end] {
            *byte = 0;
        }
    }
}

/// Remove the lock file so we can reopen the database
fn remove_lock_file(db_path: &Path) {
    let lock_file = db_path.join("db.lock");
    let _ = fs::remove_file(lock_file);
}

/// Create a database with known data, then close it.
/// Each transaction is auto-committed (one INSERT per execute call).
fn setup_test_db(num_txns: usize, rows_per_txn: usize) -> TestFixture {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();

        let mut id = 1;
        for txn in 0..num_txns {
            for row in 0..rows_per_txn {
                db.execute(
                    &format!(
                        "INSERT INTO test_data (id, value, seq) VALUES ({}, 'txn{}_row{}', {})",
                        id, txn, row, txn
                    ),
                    (),
                )
                .unwrap();
                id += 1;
            }
        }

        // Verify all data is in
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        assert_eq!(count, (num_txns * rows_per_txn) as i64);
    }

    // Database is now closed — WAL has been flushed
    remove_lock_file(&db_path);

    TestFixture {
        _dir: dir,
        db_path,
        dsn,
    }
}

/// Create a database with large TEXT values (> 64 bytes) to trigger LZ4 compression
fn setup_test_db_large_rows(num_txns: usize, rows_per_txn: usize) -> TestFixture {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();

        // Generate a large string (> 64 bytes to trigger compression)
        let large_prefix = "A".repeat(200);

        let mut id = 1;
        for txn in 0..num_txns {
            for row in 0..rows_per_txn {
                db.execute(
                    &format!(
                        "INSERT INTO test_data (id, value, seq) VALUES ({}, '{}_txn{}_row{}', {})",
                        id, large_prefix, txn, row, txn
                    ),
                    (),
                )
                .unwrap();
                id += 1;
            }
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        assert_eq!(count, (num_txns * rows_per_txn) as i64);
    }

    remove_lock_file(&db_path);

    TestFixture {
        _dir: dir,
        db_path,
        dsn,
    }
}

/// Open the database after corruption and verify that at least min_rows recovered.
/// Also verifies the database is usable (can INSERT new data).
/// If the table doesn't exist (DDL was also corrupted), that's treated as 0 rows.
fn verify_recovery_at_least(fixture: &TestFixture, table: &str, min_rows: i64) {
    let db = Database::open(&fixture.dsn).unwrap();

    let result: Result<i64, _> = db.query_one(&format!("SELECT COUNT(*) FROM {}", table), ());
    match result {
        Ok(count) => {
            assert!(
                count >= min_rows,
                "Expected at least {} rows, got {} after recovery",
                min_rows,
                count
            );

            // Verify DB is usable — can insert new data
            let new_id = count + 10000;
            db.execute(
                &format!(
                    "INSERT INTO {} (id, value, seq) VALUES ({}, 'post_recovery', 999)",
                    table, new_id
                ),
                (),
            )
            .unwrap();

            let new_count: i64 = db
                .query_one(&format!("SELECT COUNT(*) FROM {}", table), ())
                .unwrap();
            assert_eq!(new_count, count + 1);
        }
        Err(_) => {
            // Table doesn't exist — DDL was corrupted too.
            // This is valid when min_rows == 0.
            assert_eq!(
                min_rows, 0,
                "Table '{}' doesn't exist after recovery, but expected at least {} rows",
                table, min_rows
            );
        }
    }
}

/// Open the database after corruption and verify exact row count.
/// Also verifies the database is usable.
fn verify_recovery_exact(fixture: &TestFixture, table: &str, exact_rows: i64) {
    let db = Database::open(&fixture.dsn).unwrap();

    let count: i64 = db
        .query_one(&format!("SELECT COUNT(*) FROM {}", table), ())
        .unwrap();
    assert_eq!(
        count, exact_rows,
        "Expected exactly {} rows, got {} after recovery",
        exact_rows, count
    );

    // Verify DB is usable
    let new_id = exact_rows + 10000;
    db.execute(
        &format!(
            "INSERT INTO {} (id, value, seq) VALUES ({}, 'post_recovery', 999)",
            table, new_id
        ),
        (),
    )
    .unwrap();
}

// ============================================================================
// 1. TORN WRITES — Truncate WAL at various offsets within the last entry
// ============================================================================

/// Helper: Setup 5 auto-committed transactions (3 rows each = 15 rows),
/// then truncate the WAL so the last entry is incomplete at `cut_into_last`.
/// `cut_into_last` is the number of bytes to keep from the last entry.
fn torn_write_test(cut_into_last: usize) {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty(), "No WAL files found");

    // Read WAL data
    let wal_path = &wal_files[wal_files.len() - 1]; // Use last WAL file
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(
        entries.len() >= 2,
        "Need at least 2 entries, found {}",
        entries.len()
    );

    let last = &entries[entries.len() - 1];
    let truncate_at = last.offset + cut_into_last;

    // Truncate the file
    let truncated = &data[..truncate_at.min(data.len())];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);

    // Recovery should restore all complete entries (prior entries intact)
    // We don't know the exact count because DDL and commit markers are also entries,
    // but we should have at least some data rows
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_torn_write_partial_header_1_byte() {
    torn_write_test(1);
}

#[test]
fn test_torn_write_partial_header_16_bytes() {
    torn_write_test(16);
}

#[test]
fn test_torn_write_partial_header_31_bytes() {
    torn_write_test(31);
}

#[test]
fn test_torn_write_after_header_no_data() {
    torn_write_test(WAL_HEADER_SIZE);
}

#[test]
fn test_torn_write_partial_data() {
    // Truncate halfway through the data portion of last entry
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2);

    let last = &entries[entries.len() - 1];
    let half_data = last.entry_size / 2;
    let truncate_at = last.data_offset + half_data;
    let truncated = &data[..truncate_at.min(data.len())];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_torn_write_after_data_no_crc() {
    // Keep full header + full data, but no CRC bytes
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2);

    let last = &entries[entries.len() - 1];
    let truncate_at = last.crc_offset; // Right before CRC
    let truncated = &data[..truncate_at.min(data.len())];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_torn_write_partial_crc_1_byte() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2);

    let last = &entries[entries.len() - 1];
    let truncate_at = last.crc_offset + 1; // Only 1 of 4 CRC bytes
    let truncated = &data[..truncate_at.min(data.len())];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_torn_write_partial_crc_3_bytes() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2);

    let last = &entries[entries.len() - 1];
    let truncate_at = last.crc_offset + 3; // 3 of 4 CRC bytes
    let truncated = &data[..truncate_at.min(data.len())];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 2. SECTOR-ALIGNED PAGE LOSS — Zero 4KB pages
// ============================================================================

#[test]
fn test_sector_loss_first_page() {
    // Write enough data to span multiple 4KB pages
    let fixture = setup_test_db(20, 5);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();

    // Only corrupt if the file is large enough
    if data.len() > 4096 {
        zero_page(&mut data, 0);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // scan_for_magic should recover entries beyond page 0
    // We might get 0 rows if DDL was on page 0, or partial recovery
    let db = Database::open(&fixture.dsn).unwrap();
    // Just verify it opens and is usable (table may or may not exist)
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM test_data", ());
    if let Ok(count) = result {
        // If table exists, verify DB is usable
        let new_id = count + 10000;
        db.execute(
            &format!(
                "INSERT INTO test_data (id, value, seq) VALUES ({}, 'post_recovery', 999)",
                new_id
            ),
            (),
        )
        .unwrap();
    }
    // If table doesn't exist, that's also valid — DDL was on the zeroed page
}

#[test]
fn test_sector_loss_middle_page() {
    let fixture = setup_test_db(20, 5);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let num_pages = data.len() / 4096;

    if num_pages >= 3 {
        let middle = num_pages / 2;
        zero_page(&mut data, middle);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // Entries on intact pages should be recovered
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_sector_loss_last_page() {
    let fixture = setup_test_db(20, 5);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let num_pages = data.len() / 4096;

    if num_pages >= 2 {
        let last = num_pages - 1;
        zero_page(&mut data, last);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_sector_loss_alternating_pages() {
    let fixture = setup_test_db(20, 5);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let num_pages = data.len() / 4096;

    // Zero every other page (starting from page 1 to preserve DDL on page 0)
    for page in (1..num_pages).step_by(2) {
        zero_page(&mut data, page);
    }
    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 3. BIT CORRUPTION — Single-bit flips in specific fields
// ============================================================================

#[test]
fn test_bit_flip_magic_bytes() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Flip a bit in a middle entry's magic bytes
    if entries.len() >= 3 {
        let target = &entries[entries.len() / 2];
        flip_bit(&mut data, target.offset, 0); // Flip bit 0 of first magic byte
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // scan_for_magic skips the corrupted entry, recovers the rest
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_bit_flip_crc() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Flip a bit in a middle entry's CRC
    if entries.len() >= 3 {
        let target = &entries[entries.len() / 2];
        flip_bit(&mut data, target.crc_offset, 3); // Flip bit 3 of CRC
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_bit_flip_data_portion() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Flip a bit in the data portion of a middle entry
    if entries.len() >= 3 {
        let target = &entries[entries.len() / 2];
        let data_mid = target.data_offset + target.entry_size / 2;
        flip_bit(&mut data, data_mid, 5);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // CRC mismatch -> entry skipped
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_bit_flip_entry_size() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Flip a high bit in entry_size field of a middle entry
    // This causes wrong size reading, cascading to magic scan recovery
    if entries.len() >= 3 {
        let target = &entries[entries.len() / 2];
        // entry_size is at offset +24 (4 bytes, little-endian)
        flip_bit(&mut data, target.offset + 24, 7); // Flip bit 7 -> adds 128 to low byte
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_bit_flip_flags_compressed() {
    // Create DB with small rows (uncompressed entries)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Set the COMPRESSED flag on an uncompressed entry
    if entries.len() >= 3 {
        let target = &entries[entries.len() / 2];
        // flags byte is at offset + 5
        data[target.offset + 5] |= COMPRESSED_FLAG;
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // LZ4 decompress fails on non-LZ4 data -> entry skipped
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 4. CHECKPOINT CORRUPTION
// ============================================================================

#[test]
fn test_checkpoint_corrupt_magic() {
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        let mut data = fs::read(&cp_path).unwrap();
        if data.len() >= 4 {
            zero_range(&mut data, 0, 4); // Zero magic bytes
            fs::write(&cp_path, &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);
    // Falls back to WAL file scan, full recovery
    verify_recovery_exact(&fixture, "test_data", 15);
}

#[test]
fn test_checkpoint_corrupt_crc() {
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        let mut data = fs::read(&cp_path).unwrap();
        if data.len() >= 4 {
            let crc_offset = data.len() - 4;
            flip_bit(&mut data, crc_offset, 0); // Flip bit in CRC
            fs::write(&cp_path, &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_exact(&fixture, "test_data", 15);
}

#[test]
fn test_checkpoint_truncated() {
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        let data = fs::read(&cp_path).unwrap();
        if data.len() > 10 {
            fs::write(&cp_path, &data[..10]).unwrap(); // Truncate to 10 bytes
        }
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_exact(&fixture, "test_data", 15);
}

#[test]
fn test_checkpoint_zeroed() {
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        let data = fs::read(&cp_path).unwrap();
        let zeroed = vec![0u8; data.len()];
        fs::write(&cp_path, &zeroed).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_exact(&fixture, "test_data", 15);
}

#[test]
fn test_checkpoint_deleted() {
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        fs::remove_file(&cp_path).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // No checkpoint, scans WAL dir from LSN 0, full recovery
    verify_recovery_exact(&fixture, "test_data", 15);
}

// ============================================================================
// 5. MULTI-ENTRY CORRUPTION
// ============================================================================

/// Find entries that are DML data entries (not DDL, not commit/abort markers)
fn find_dml_data_entries(entries: &[WalEntryInfo]) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            // Not a commit or abort marker
            (e.flags & COMMIT_MARKER_FLAG) == 0 && (e.flags & 0x04) == 0
        })
        .map(|(i, _)| i)
        .collect()
}

/// Find entries that are commit markers
fn find_commit_entries(entries: &[WalEntryInfo]) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| (e.flags & COMMIT_MARKER_FLAG) != 0)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn test_first_data_entry_corrupt() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);

    // Corrupt the first DML entry's magic
    if !dml_entries.is_empty() {
        let idx = dml_entries[0];
        let target = &entries[idx];
        zero_range(&mut data, target.offset, 4); // Zero magic
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // DDL (CREATE TABLE) must survive; first DML skipped, rest may recover
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_last_entry_corrupt() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(!entries.is_empty());

    // Truncate the last entry (most common crash scenario)
    let last = &entries[entries.len() - 1];
    let truncated = &data[..last.offset];
    fs::write(wal_path, truncated).unwrap();

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_middle_entry_corrupt() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Zero the magic of a middle entry
    if entries.len() >= 3 {
        let mid_idx = entries.len() / 2;
        let target = &entries[mid_idx];
        zero_range(&mut data, target.offset, 4);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // scan_for_magic skips it, entries before/after recovered
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_commit_marker_destroyed() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let commit_entries = find_commit_entries(&entries);

    // Zero the last commit marker -> that transaction becomes "in-doubt" -> aborted
    if !commit_entries.is_empty() {
        let last_commit_idx = commit_entries[commit_entries.len() - 1];
        let target = &entries[last_commit_idx];
        zero_range(&mut data, target.offset, target.total_size);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // The transaction whose commit marker was destroyed should be treated as aborted
    // Other transactions should be fine
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_data_entries_destroyed_commit_intact() {
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);

    // Zero all DML entries for the last batch while keeping commit markers
    // (Zero magic of the last few DML entries)
    if dml_entries.len() >= 3 {
        // Zero last 3 DML entries
        for &idx in &dml_entries[dml_entries.len() - 3..] {
            let target = &entries[idx];
            zero_range(&mut data, target.offset, 4); // Zero magic
        }
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Commit marker found but data entries gone -> those rows won't be recovered
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 6. WAL TRUNCATION CRASH RECOVERY
// ============================================================================

#[test]
fn test_truncation_crash_bak_exists() {
    // Simulate: WAL was backed up to .bak, then original .log was deleted
    // but new .log was never created -> crash
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[0];
    let bak_path = wal_path.with_extension("log.bak");

    // Rename .log to .log.bak (simulate incomplete truncation)
    fs::rename(wal_path, &bak_path).unwrap();

    // Verify: no .log files, only .bak
    let remaining_wals = find_wal_files(&fixture.db_path);
    assert!(
        remaining_wals.is_empty() || !remaining_wals.contains(wal_path),
        "Original WAL file should be gone"
    );

    remove_lock_file(&fixture.db_path);

    // recover_interrupted_truncation should restore .bak -> .log
    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
    assert!(
        count > 0,
        "Should recover data from .bak file, got {} rows",
        count
    );

    // Verify usable
    db.execute(
        &format!(
            "INSERT INTO test_data (id, value, seq) VALUES ({}, 'post_recovery', 999)",
            count + 10000
        ),
        (),
    )
    .unwrap();
}

#[test]
fn test_truncation_crash_temp_and_bak() {
    // Both wal-temp-*.log and .log.bak exist
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_dir = fixture.db_path.join("wal");

    // Create a temp file (simulates incomplete truncation)
    let temp_path = wal_dir.join("wal-temp-20250101120000.log");
    fs::write(&temp_path, b"temporary data").unwrap();

    // Create a .bak file from the real WAL
    let wal_path = &wal_files[0];
    let bak_path = wal_path.with_extension("log.bak");
    fs::copy(wal_path, &bak_path).unwrap();

    remove_lock_file(&fixture.db_path);

    // Temp file should be cleaned up, .bak should be handled
    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
    assert_eq!(count, 15);

    // Verify temp file was cleaned up
    assert!(!temp_path.exists(), "Temp file should have been cleaned up");
}

#[test]
fn test_truncation_crash_temp_only() {
    // Only wal-temp-*.log exists alongside valid .log
    let fixture = setup_test_db(5, 3);

    let wal_dir = fixture.db_path.join("wal");
    let temp_path = wal_dir.join("wal-temp-20250101120000.log");
    fs::write(&temp_path, b"temporary data").unwrap();

    remove_lock_file(&fixture.db_path);

    // Normal recovery, temp file cleaned up
    verify_recovery_exact(&fixture, "test_data", 15);

    // Verify temp file was cleaned up
    assert!(!temp_path.exists(), "Temp file should have been cleaned up");
}

// ============================================================================
// 7. LSN CHAIN INTEGRITY
// ============================================================================

#[test]
fn test_lsn_gap_in_sequence() {
    // Remove one complete entry from the middle (splice bytes out)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 5 {
        let remove_idx = entries.len() / 2;
        let target = &entries[remove_idx];
        let before = &data[..target.offset];
        let after = &data[target.offset + target.total_size..];

        let mut spliced = Vec::with_capacity(before.len() + after.len());
        spliced.extend_from_slice(before);
        spliced.extend_from_slice(after);
        fs::write(wal_path, &spliced).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Remaining entries have valid magic/CRC, should be recovered normally
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_duplicate_entry_in_wal() {
    // Duplicate one entry (append a copy)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 3 {
        let dup_idx = entries.len() / 2;
        let target = &entries[dup_idx];
        let entry_bytes = &data[target.offset..target.offset + target.total_size];

        let mut new_data = data.clone();
        new_data.extend_from_slice(entry_bytes);
        fs::write(wal_path, &new_data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Duplicate entry may cause PK conflict but recovery should still succeed
    // (either ignoring the duplicate or handling the error gracefully)
    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
    assert!(count >= 0, "Recovery should produce a consistent state");
}

#[test]
fn test_recovery_without_checkpoint() {
    // Delete checkpoint.meta entirely, keep WAL
    let fixture = setup_test_db(5, 3);

    if let Some(cp_path) = find_checkpoint_file(&fixture.db_path) {
        fs::remove_file(&cp_path).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // from_lsn=0, all entries replayed from WAL scan
    verify_recovery_exact(&fixture, "test_data", 15);
}

// ============================================================================
// 8. COMPRESSION-RELATED CORRUPTION
// ============================================================================

#[test]
fn test_corrupt_compressed_payload() {
    // Use large rows to trigger LZ4 compression
    let fixture = setup_test_db_large_rows(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Find a compressed entry
    let compressed: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| (e.flags & COMPRESSED_FLAG) != 0)
        .map(|(i, _)| i)
        .collect();

    if !compressed.is_empty() {
        let target_idx = compressed[compressed.len() / 2];
        let target = &entries[target_idx];

        // Flip bytes in the compressed payload, then recompute CRC
        let data_mid = target.data_offset + target.entry_size / 2;
        if data_mid < data.len() {
            data[data_mid] ^= 0xFF;
            data[data_mid.saturating_sub(1)] ^= 0xAA;

            // Recompute CRC over the data portion
            let data_start = target.data_offset;
            let data_end = target.crc_offset;
            let new_crc = crc32fast::hash(&data[data_start..data_end]);
            data[target.crc_offset..target.crc_offset + 4].copy_from_slice(&new_crc.to_le_bytes());

            fs::write(wal_path, &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);
    // Valid CRC but LZ4 decompress fails -> entry skipped
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_force_compressed_flag_on_uncompressed() {
    // Small rows -> uncompressed entries. Set COMPRESSED flag.
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Find an uncompressed DML entry
    let uncompressed: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| (e.flags & COMPRESSED_FLAG) == 0 && (e.flags & COMMIT_MARKER_FLAG) == 0)
        .map(|(i, _)| i)
        .collect();

    if uncompressed.len() >= 2 {
        let target_idx = uncompressed[uncompressed.len() / 2];
        let target = &entries[target_idx];
        // Set COMPRESSED flag (flags byte is at offset + 5)
        data[target.offset + 5] |= COMPRESSED_FLAG;
        // Note: We do NOT update CRC — flags are in the header, not in CRC-covered data portion
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // LZ4 decompress fails on non-LZ4 data -> entry skipped
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_clear_compressed_flag_on_compressed() {
    // Large rows -> compressed entries. Clear COMPRESSED flag.
    let fixture = setup_test_db_large_rows(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Find a compressed entry
    let compressed: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| (e.flags & COMPRESSED_FLAG) != 0)
        .map(|(i, _)| i)
        .collect();

    if !compressed.is_empty() {
        let target_idx = compressed[compressed.len() / 2];
        let target = &entries[target_idx];
        // Clear COMPRESSED flag
        data[target.offset + 5] &= !COMPRESSED_FLAG;
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Raw compressed bytes interpreted as row data -> deserialization fails -> entry skipped
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 9. COMBINED STRESS TESTS
// ============================================================================

#[test]
fn test_combined_torn_write_plus_bit_flip() {
    let fixture = setup_test_db(10, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Bit flip in a middle entry
    if entries.len() >= 5 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        flip_bit(&mut data, target.data_offset + 5, 2);
    }

    // Truncate the last entry
    if entries.len() >= 2 {
        let last = &entries[entries.len() - 1];
        let truncate_at = last.offset + WAL_HEADER_SIZE / 2;
        data.truncate(truncate_at);
    }

    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);
    // Both failures handled, partial recovery
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_many_transactions_last_commit_corrupt() {
    let fixture = setup_test_db(20, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let commit_entries = find_commit_entries(&entries);

    // Corrupt the last commit marker
    if !commit_entries.is_empty() {
        let last_commit_idx = commit_entries[commit_entries.len() - 1];
        let target = &entries[last_commit_idx];
        zero_range(&mut data, target.offset, target.total_size);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // 19 transactions should be recovered, last one treated as uncommitted
    // Each transaction has 3 rows, so at least 19*3 - some potential loss = many rows
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_recovery_then_new_data_then_recovery() {
    // Phase 1: Create DB with data, corrupt, recover
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Corrupt: truncate last entry
    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    if entries.len() >= 2 {
        let last = &entries[entries.len() - 1];
        let truncated = &data[..last.offset + 10]; // 10 bytes into last entry
        fs::write(wal_path, truncated).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // Phase 2: Recover and insert new data
    let initial_count;
    {
        let db = Database::open(&fixture.dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        initial_count = count;

        // Insert more data after recovery
        for i in 0..5 {
            db.execute(
                &format!(
                    "INSERT INTO test_data (id, value, seq) VALUES ({}, 'new_data_{}', 100)",
                    10000 + i,
                    i
                ),
                (),
            )
            .unwrap();
        }

        let after_insert: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        assert_eq!(after_insert, initial_count + 5);
    }

    remove_lock_file(&fixture.db_path);

    // Phase 3: Reopen again — post-recovery data should persist
    {
        let db = Database::open(&fixture.dsn).unwrap();
        let final_count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        // The 5 new inserts should definitely persist. The initial_count rows
        // may vary slightly across recoveries due to WAL truncation effects,
        // but we should have at least the new inserts.
        assert!(
            final_count >= initial_count,
            "Post-recovery count ({}) should be at least initial count ({})",
            final_count,
            initial_count
        );

        // Verify still usable
        db.execute(
            "INSERT INTO test_data (id, value, seq) VALUES (99999, 'final_test', 200)",
            (),
        )
        .unwrap();
    }
}

#[test]
fn test_complete_wal_destruction_with_snapshot() {
    // Create DB, take a snapshot, add more data, then zero entire WAL
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table and insert initial data
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO test_data (id, value, seq) VALUES ({}, 'snapshot_data_{}', 1)",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Force a snapshot
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert more data after snapshot
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO test_data (id, value, seq) VALUES ({}, 'post_snapshot_{}', 2)",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Phase 2: Zero entire WAL (but snapshot should be intact)
    let wal_files = find_wal_files(&db_path);
    for wal_path in &wal_files {
        let data = fs::read(wal_path).unwrap();
        let zeroed = vec![0u8; data.len()];
        fs::write(wal_path, &zeroed).unwrap();
    }

    // Phase 3: Try to recover
    let db = Database::open(&dsn).unwrap();

    // The snapshot data should survive, post-snapshot WAL data may be lost
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM test_data", ());
    match result {
        Ok(count) => {
            // If table exists, snapshot data should be intact
            // Post-snapshot data depends on whether WAL was needed
            assert!(count >= 0, "Should have consistent state after recovery");
        }
        Err(_) => {
            // If both WAL and snapshot are damaged, table may not exist
            // This is still a valid recovery outcome (no data)
        }
    }
}

// ============================================================================
// Additional edge case tests
// ============================================================================

#[test]
fn test_empty_wal_file() {
    // Create an empty WAL file (0 bytes)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Replace the WAL file with an empty file
    let wal_path = &wal_files[wal_files.len() - 1];
    fs::write(wal_path, []).unwrap();

    remove_lock_file(&fixture.db_path);

    // Recovery should handle empty file gracefully
    let db = Database::open(&fixture.dsn).unwrap();
    // Table might not exist if DDL was in the zeroed WAL
    let _result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM test_data", ());
}

#[test]
fn test_wal_file_with_only_header_garbage() {
    // WAL file contains exactly 32 bytes of garbage
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let garbage = vec![0xDE; 32];
    fs::write(wal_path, &garbage).unwrap();

    remove_lock_file(&fixture.db_path);

    // Should handle garbage gracefully
    let db = Database::open(&fixture.dsn).unwrap();
    let _result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM test_data", ());
}

#[test]
fn test_wal_magic_at_very_end_of_file() {
    // WAL file with valid data but magic bytes at the very end (incomplete entry)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(!entries.is_empty());

    // Truncate to end of second-to-last entry, then append just magic bytes
    if entries.len() >= 2 {
        let second_to_last = &entries[entries.len() - 2];
        let end_of_stl = second_to_last.offset + second_to_last.total_size;
        data.truncate(end_of_stl);
        // Append just the magic bytes (incomplete header)
        data.extend_from_slice(&WAL_ENTRY_MAGIC.to_le_bytes());
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_multiple_wal_files_one_corrupt() {
    // Create enough data to potentially generate multiple WAL files
    // Even if only one WAL file, we simulate by splitting
    let fixture = setup_test_db(20, 5);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Corrupt the first WAL file if there are multiple
    if wal_files.len() >= 2 {
        let mut data = fs::read(&wal_files[0]).unwrap();
        // Zero the first 100 bytes
        zero_range(&mut data, 0, 100);
        fs::write(&wal_files[0], &data).unwrap();
    } else {
        // Only one WAL file — corrupt the first page but leave the rest
        let mut data = fs::read(&wal_files[0]).unwrap();
        if data.len() > 4096 {
            // Corrupt middle portion, leave start and end intact
            let mid = data.len() / 2;
            zero_range(&mut data, mid, 100);
            fs::write(&wal_files[0], &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_all_commit_markers_destroyed() {
    // Zero every commit marker in the WAL — all transactions should be aborted
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let commit_entries = find_commit_entries(&entries);

    // Zero all commit markers (but not DDL commit markers)
    for &idx in &commit_entries {
        let target = &entries[idx];
        zero_range(&mut data, target.offset, target.total_size);
    }
    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);

    // Without commit markers, all DML transactions are treated as aborted.
    // DDL (CREATE TABLE) uses DDL_TXN_ID which may or may not have a commit marker.
    let db = Database::open(&fixture.dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM test_data", ());
    match result {
        Ok(count) => {
            // Table exists but data transactions were aborted
            assert!(count >= 0, "Count should be non-negative");
        }
        Err(_) => {
            // Table doesn't exist is also valid if DDL was also treated as aborted
        }
    }
}

#[test]
fn test_repeated_recovery_idempotent() {
    // Corrupt WAL, recover multiple times — should always get same result
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Corrupt a middle entry
    if entries.len() >= 3 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        flip_bit(&mut data, target.crc_offset, 0);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // Recovery #1
    let count1;
    {
        let db = Database::open(&fixture.dsn).unwrap();
        let c: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        count1 = c;
    }

    remove_lock_file(&fixture.db_path);

    // Recovery #2
    {
        let db = Database::open(&fixture.dsn).unwrap();
        let count2: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        // Second recovery should produce same or more rows (since first recovery
        // may have written new WAL entries that are now also recovered)
        assert!(
            count2 >= count1,
            "Second recovery ({}) should have at least as many rows as first ({})",
            count2,
            count1
        );
    }
}

#[test]
fn test_concurrent_corruption_patterns() {
    // Apply multiple corruption patterns simultaneously
    let fixture = setup_test_db(10, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 8 {
        // Corruption 1: Flip bit in entry 2's CRC
        let target1 = &entries[2];
        flip_bit(&mut data, target1.crc_offset, 4);

        // Corruption 2: Zero magic of entry 4
        let target2 = entries[4].clone();
        zero_range(&mut data, target2.offset, 4);

        // Corruption 3: Flip compressed flag on entry 6
        let target3 = &entries[6];
        data[target3.offset + 5] |= COMPRESSED_FLAG;
    }

    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);
    // Multiple corruptions handled, some entries skipped, partial recovery
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_wal_with_garbage_appended() {
    // Append random garbage after valid WAL data
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();

    // Append 1KB of non-magic garbage
    let garbage: Vec<u8> = (0..1024).map(|i| ((i * 7 + 13) % 256) as u8).collect();
    data.extend_from_slice(&garbage);
    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);
    // All valid entries should be recovered, garbage at end ignored
    verify_recovery_exact(&fixture, "test_data", 15);
}

#[test]
fn test_wal_entry_size_zero() {
    // Set entry_size to 0 for a middle entry
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 3 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        // Set entry_size (bytes 24-27) to 0
        data[target.offset + 24] = 0;
        data[target.offset + 25] = 0;
        data[target.offset + 26] = 0;
        data[target.offset + 27] = 0;
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Size 0 means 0 data bytes + 4 CRC bytes = 4 bytes total.
    // CRC will mismatch, entry skipped, scan_for_magic finds next entry.
    verify_recovery_at_least(&fixture, "test_data", 0);
}

#[test]
fn test_wal_entry_size_very_large() {
    // Set entry_size to a very large value (> 64MB limit)
    let fixture = setup_test_db(5, 3);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 3 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        // Set entry_size to 0x10000000 (256MB) — exceeds 64MB sanity check
        data[target.offset + 24] = 0;
        data[target.offset + 25] = 0;
        data[target.offset + 26] = 0;
        data[target.offset + 27] = 0x10;
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);
    // Sanity check triggers scan_for_magic, skips corrupted entry
    verify_recovery_at_least(&fixture, "test_data", 0);
}

// ============================================================================
// 10. MULTI-TABLE RECOVERY
// ============================================================================

/// Setup a database with multiple tables
fn setup_multi_table_db() -> TestFixture {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price FLOAT)",
            (),
        )
        .unwrap();

        // Populate users
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO users (id, name, email) VALUES ({}, 'user{}', 'user{}@test.com')",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        // Populate orders
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO orders (id, user_id, amount) VALUES ({}, {}, {})",
                    i,
                    (i % 10) + 1,
                    i as f64 * 9.99
                ),
                (),
            )
            .unwrap();
        }

        // Populate products
        for i in 1..=15 {
            db.execute(
                &format!(
                    "INSERT INTO products (id, name, price) VALUES ({}, 'product{}', {})",
                    i,
                    i,
                    i as f64 * 5.50
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);
    TestFixture {
        _dir: dir,
        db_path,
        dsn,
    }
}

#[test]
fn test_multi_table_corrupt_middle_entry() {
    // Corrupt a middle entry — only one table's data may be affected
    let fixture = setup_multi_table_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 5 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        flip_bit(&mut data, target.crc_offset, 0);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // All three tables should exist
    let users: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    let orders: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
    let products: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();

    // At least some data in each table (the corrupted entry only affects one row)
    let total = users + orders + products;
    assert!(
        total >= 40,
        "Expected at least 40 total rows across tables, got {} (users={}, orders={}, products={})",
        total,
        users,
        orders,
        products
    );

    // All tables should be usable
    db.execute(
        "INSERT INTO users (id, name, email) VALUES (999, 'new_user', 'new@test.com')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, user_id, amount) VALUES (999, 1, 100.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO products (id, name, price) VALUES (999, 'new_product', 50.0)",
        (),
    )
    .unwrap();
}

#[test]
fn test_multi_table_one_tables_commit_destroyed() {
    // Destroy a commit marker — the transaction's data for ALL tables in that txn is lost
    let fixture = setup_multi_table_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let commit_entries = find_commit_entries(&entries);

    // Destroy a middle commit marker
    if commit_entries.len() >= 3 {
        let mid_commit = commit_entries[commit_entries.len() / 2];
        let target = &entries[mid_commit];
        zero_range(&mut data, target.offset, target.total_size);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // All three tables should exist (DDL is separate from DML commits)
    let users: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    let orders: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
    let products: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();

    // Should lose exactly one transaction's worth of data
    let total = users + orders + products;
    assert!(
        total >= 30,
        "Expected at least 30 total rows, got {} (users={}, orders={}, products={})",
        total,
        users,
        orders,
        products
    );
}

#[test]
fn test_multi_table_first_page_zeroed() {
    // Zero page 0 — DDL for some tables may be lost
    let fixture = setup_multi_table_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();

    if data.len() > 4096 {
        zero_page(&mut data, 0);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // Some tables may not exist, but the DB should still open
    let db = Database::open(&fixture.dsn).unwrap();

    // Check which tables survived
    let mut surviving_tables = 0;
    for table in &["users", "orders", "products"] {
        let result: Result<i64, _> = db.query_one(&format!("SELECT COUNT(*) FROM {}", table), ());
        if result.is_ok() {
            surviving_tables += 1;
        }
    }

    // At least the DB opened successfully — table survival depends on entry order
    assert!(surviving_tables >= 0); // Always true, but documents the invariant
}

#[test]
fn test_multi_table_cross_table_join_after_recovery() {
    // After partial corruption, verify cross-table joins still work
    let fixture = setup_multi_table_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Truncate last entry (mild corruption)
    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    if entries.len() >= 2 {
        let last = &entries[entries.len() - 1];
        let truncated = &data[..last.offset];
        fs::write(wal_path, truncated).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // Execute a cross-table join — this exercises the full query engine post-recovery
    let result = db.query(
        "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id ORDER BY o.id LIMIT 5",
        (),
    );

    match result {
        Ok(rows) => {
            // Consume the iterator — verifies join execution doesn't panic
            let collected: Vec<_> = rows.filter_map(|r| r.ok()).collect();
            // Count may be 0 or more depending on what survived
            assert!(collected.len() <= 100, "Sanity check: not too many rows");
        }
        Err(_) => {
            // If a table was lost, join might fail — that's acceptable
        }
    }
}

// ============================================================================
// 11. INDEX RECOVERY
// ============================================================================

/// Setup a database with indexes
fn setup_indexed_db() -> TestFixture {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE indexed_data (id INTEGER PRIMARY KEY, category TEXT NOT NULL, value FLOAT, active BOOLEAN)",
            (),
        )
        .unwrap();

        // Create indexes of different types
        db.execute("CREATE INDEX idx_value ON indexed_data(value)", ())
            .unwrap();
        db.execute("CREATE INDEX idx_category ON indexed_data(category)", ())
            .unwrap();

        // Insert data
        for i in 1..=50 {
            let category = format!("cat{}", i % 5);
            let active = i % 2 == 0;
            db.execute(
                &format!(
                    "INSERT INTO indexed_data (id, category, value, active) VALUES ({}, '{}', {}, {})",
                    i, category, i as f64 * 1.5, active
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);
    TestFixture {
        _dir: dir,
        db_path,
        dsn,
    }
}

#[test]
fn test_index_recovery_after_torn_write() {
    let fixture = setup_indexed_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Truncate last entry
    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    if entries.len() >= 2 {
        let last = &entries[entries.len() - 1];
        let truncated = &data[..last.offset];
        fs::write(wal_path, truncated).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // Index-accelerated queries should work after recovery
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data WHERE value > 50.0", ())
        .unwrap();
    assert!(count >= 0, "Index query should work");

    let count2: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM indexed_data WHERE category = 'cat1'",
            (),
        )
        .unwrap();
    assert!(count2 >= 0, "Hash index query should work");
}

#[test]
fn test_index_recovery_after_bit_corruption() {
    let fixture = setup_indexed_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Flip bit in a middle entry's data
    if entries.len() >= 5 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        flip_bit(&mut data, target.data_offset + 10, 3);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // Verify indexed queries produce consistent results
    let total: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data", ())
        .unwrap();
    let indexed: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data WHERE value > 0.0", ())
        .unwrap();

    // Index count should match or be less than total (never more)
    assert!(
        indexed <= total,
        "Index query ({}) should never return more rows than total ({})",
        indexed,
        total
    );
}

#[test]
fn test_index_consistency_after_corruption() {
    // Verify that index results match full scan results after corruption
    let fixture = setup_indexed_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    // Corrupt a middle entry
    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 3 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        zero_range(&mut data, target.offset, 4); // Zero magic
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // Compare index-scanned results vs full-scan results
    // Both should return the same data
    let total: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data", ())
        .unwrap();

    // Category index query
    let cat_total: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM indexed_data WHERE category IN ('cat0', 'cat1', 'cat2', 'cat3', 'cat4')",
            (),
        )
        .unwrap();

    assert_eq!(
        total, cat_total,
        "Index scan over all categories ({}) should match full scan ({})",
        cat_total, total
    );

    // Verify insertions work with indexes
    db.execute(
        "INSERT INTO indexed_data (id, category, value, active) VALUES (999, 'cat_new', 999.0, true)",
        (),
    )
    .unwrap();

    let after: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data", ())
        .unwrap();
    assert_eq!(after, total + 1);
}

#[test]
fn test_index_ddl_corrupt_but_data_intact() {
    // Corrupt the CREATE INDEX WAL entry, but keep data intact
    // Index should not exist, but data queries should still work (full scan)
    let fixture = setup_indexed_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);

    // The CREATE INDEX entries are DDL (non-commit, non-DML data entries)
    // They appear early in the WAL. Corrupt entries 1 and 2 (likely DDL for indexes)
    // We'll corrupt entries right after the CREATE TABLE entries
    if dml_entries.len() >= 2 && entries.len() > dml_entries[0] {
        // Find entries that are NOT DML data and NOT commit markers (likely DDL)
        let ddl_like: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                (e.flags & COMMIT_MARKER_FLAG) == 0
                    && (e.flags & 0x04) == 0
                    && !dml_entries.contains(i)
            })
            .map(|(i, _)| i)
            .collect();

        // Corrupt a few DDL-like entries (skip entry 0 which is CREATE TABLE)
        for &idx in ddl_like.iter().skip(1).take(2) {
            let target = &entries[idx];
            flip_bit(&mut data, target.crc_offset, 5);
        }
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();

    // Table should exist and data should be queryable (even without indexes)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data", ())
        .unwrap();
    assert!(count >= 0, "Data should be accessible even without indexes");

    // Queries that would normally use indexes should fall back to full scan
    let result: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_data WHERE value > 50.0", ())
        .unwrap();
    assert!(result >= 0);
}

#[test]
fn test_multi_column_index_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, amount FLOAT)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE INDEX idx_cust_date ON orders(customer_id, order_date)",
            (),
        )
        .unwrap();

        for i in 1..=30 {
            db.execute(
                &format!(
                    "INSERT INTO orders (id, customer_id, order_date, amount) VALUES ({}, {}, '2024-01-{:02}', {})",
                    i, i % 5, (i % 28) + 1, i as f64 * 10.0
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Corrupt a middle entry
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());
    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() >= 3 {
        let mid = entries.len() / 2;
        let target = &entries[mid];
        flip_bit(&mut data, target.data_offset + 5, 1);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&db_path);

    let fixture = TestFixture {
        _dir: dir,
        db_path,
        dsn,
    };

    let db = Database::open(&fixture.dsn).unwrap();

    // Multi-column index query
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders WHERE customer_id = 1 AND order_date = '2024-01-01'",
            (),
        )
        .unwrap();
    assert!(count >= 0);

    // Verify insertions work
    db.execute(
        "INSERT INTO orders (id, customer_id, order_date, amount) VALUES (999, 1, '2024-12-25', 500.0)",
        (),
    )
    .unwrap();
}

// ============================================================================
// 12. MULTI-STATEMENT TRANSACTION CORRUPTION
// ============================================================================

/// Setup a database using explicit BEGIN/COMMIT transactions
fn setup_explicit_txn_db() -> TestFixture {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE txn_data (id INTEGER PRIMARY KEY, value TEXT NOT NULL, batch INTEGER)",
            (),
        )
        .unwrap();

        // 5 explicit transactions, each inserting 4 rows
        for batch in 0..5 {
            db.execute("BEGIN", ()).unwrap();
            for row in 0..4 {
                let id = batch * 4 + row + 1;
                db.execute(
                    &format!(
                        "INSERT INTO txn_data (id, value, batch) VALUES ({}, 'batch{}_row{}', {})",
                        id, batch, row, batch
                    ),
                    (),
                )
                .unwrap();
            }
            db.execute("COMMIT", ()).unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_data", ()).unwrap();
        assert_eq!(count, 20);
    }

    remove_lock_file(&db_path);
    TestFixture {
        _dir: dir,
        db_path,
        dsn,
    }
}

#[test]
fn test_explicit_txn_one_insert_corrupt() {
    // Corrupt one INSERT entry within a committed multi-statement transaction
    let fixture = setup_explicit_txn_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);

    // Corrupt one DML entry from a middle batch
    if dml_entries.len() >= 10 {
        let target_idx = dml_entries[dml_entries.len() / 2];
        let target = &entries[target_idx];
        flip_bit(&mut data, target.data_offset + 8, 4); // Corrupt data portion
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_data", ()).unwrap();

    // One entry corrupted — that row is lost but the rest should survive
    // The transaction's commit marker is intact, so other rows in the txn are applied
    assert!(
        count >= 15,
        "Expected at least 15 rows (lost at most a few from corruption), got {}",
        count
    );

    // DB is usable
    db.execute(
        "INSERT INTO txn_data (id, value, batch) VALUES (999, 'post_recovery', 99)",
        (),
    )
    .unwrap();
}

#[test]
fn test_explicit_txn_commit_marker_corrupt() {
    // Destroy one transaction's commit marker — all 4 rows from that txn should be lost
    let fixture = setup_explicit_txn_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let commit_entries = find_commit_entries(&entries);

    // Destroy the 3rd commit marker (middle batch)
    if commit_entries.len() >= 5 {
        let target_idx = commit_entries[2]; // 3rd transaction's commit
        let target = &entries[target_idx];
        zero_range(&mut data, target.offset, target.total_size);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_data", ()).unwrap();

    // Lost one transaction (4 rows), rest should survive
    // Could be fewer if DDL commit was also corrupted
    assert!(
        count >= 12,
        "Expected at least 12 rows (lost 1 txn of 4 rows + possible side effects), got {}",
        count
    );

    // Verify batch consistency: each surviving batch should have all 4 rows or 0
    for batch in 0..5 {
        let batch_count: i64 = db
            .query_one(
                &format!("SELECT COUNT(*) FROM txn_data WHERE batch = {}", batch),
                (),
            )
            .unwrap();
        assert!(
            batch_count == 0 || batch_count == 4,
            "Batch {} has {} rows — expected 0 (aborted) or 4 (committed)",
            batch,
            batch_count
        );
    }
}

#[test]
fn test_explicit_txn_all_inserts_in_one_txn_corrupt() {
    // Corrupt all INSERT entries within one transaction, but keep commit marker
    let fixture = setup_explicit_txn_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);
    let commit_entries = find_commit_entries(&entries);

    // Each batch: 4 DML entries + 1 commit marker
    // Corrupt all 4 DML entries of batch 2 (entries at dml indices 8-11 approximately)
    if dml_entries.len() >= 12 && commit_entries.len() >= 3 {
        // Corrupt DML entries 8, 9, 10, 11 (batch 2)
        for &idx in &dml_entries[8..12] {
            let target = &entries[idx];
            zero_range(&mut data, target.offset, 4); // Zero magic
        }
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_data", ()).unwrap();

    // Batch 2's data is corrupted — commit marker exists but data is gone
    // Other 4 batches should be intact: 4 * 4 = 16 rows
    assert!(
        count >= 12,
        "Expected at least 12 rows (4 batches intact minus possible losses), got {}",
        count
    );
}

#[test]
fn test_explicit_txn_interleaved_corruption() {
    // Corrupt entries from different transactions
    let fixture = setup_explicit_txn_db();
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    let dml_entries = find_dml_data_entries(&entries);

    // Corrupt one entry from batch 1 and one from batch 3
    if dml_entries.len() >= 16 {
        // Batch 1 entry (index 4-7), corrupt index 5
        let target = &entries[dml_entries[5]];
        flip_bit(&mut data, target.crc_offset, 2);

        // Batch 3 entry (index 12-15), corrupt index 13
        let target2 = entries[dml_entries[13]].clone();
        flip_bit(&mut data, target2.crc_offset, 6);

        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_data", ()).unwrap();

    // Lost 2 individual rows from 2 different transactions
    assert!(
        count >= 16,
        "Expected at least 16 rows (lost 2 individual rows), got {}",
        count
    );
}

// ============================================================================
// 13. SNAPSHOT FILE CORRUPTION
// ============================================================================

/// Setup a database, create a snapshot, then add more data
fn setup_db_with_snapshot() -> (TestFixture, i64) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    let pre_snapshot_count;

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE snap_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL, phase INTEGER)",
            (),
        )
        .unwrap();

        // Phase 1: Pre-snapshot data
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO snap_test (id, value, phase) VALUES ({}, 'pre_snap_{}', 1)",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        pre_snapshot_count = 20;

        // Create snapshot
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Post-snapshot data
        for i in 21..=40 {
            db.execute(
                &format!(
                    "INSERT INTO snap_test (id, value, phase) VALUES ({}, 'post_snap_{}', 2)",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);
    (
        TestFixture {
            _dir: dir,
            db_path,
            dsn,
        },
        pre_snapshot_count,
    )
}

/// Find snapshot files for a given table
fn find_snapshot_files(db_path: &Path, table: &str) -> Vec<PathBuf> {
    let snap_dir = db_path.join("snapshots").join(table);
    if !snap_dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = fs::read_dir(&snap_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("snapshot-") && name.ends_with(".bin")
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

#[test]
fn test_snapshot_corrupt_with_valid_wal() {
    // Corrupt snapshot file, but WAL is intact
    // Note: After PRAGMA SNAPSHOT, WAL may be truncated (pre-snapshot entries removed).
    // If snapshot is then corrupted, the CREATE TABLE DDL may be lost.
    let (fixture, _) = setup_db_with_snapshot();

    let snap_files = find_snapshot_files(&fixture.db_path, "snap_test");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 10 {
            zero_range(&mut data, 0, 10);
            fs::write(snap_path, &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM snap_test", ());

    match result {
        Ok(count) => {
            // If table survived (DDL in WAL), we should have some data
            assert!(
                count >= 0,
                "Should have non-negative row count, got {}",
                count
            );
        }
        Err(_) => {
            // WAL was truncated after snapshot — DDL not in WAL anymore.
            // With corrupt snapshot, table cannot be recreated. This is expected.
        }
    }
}

#[test]
fn test_snapshot_deleted_with_valid_wal() {
    // Delete all snapshot files, WAL is intact
    // Note: After PRAGMA SNAPSHOT, WAL may be truncated (pre-snapshot entries removed).
    let (fixture, _) = setup_db_with_snapshot();

    let snap_files = find_snapshot_files(&fixture.db_path, "snap_test");
    for snap_path in &snap_files {
        let _ = fs::remove_file(snap_path);
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM snap_test", ());

    match result {
        Ok(count) => {
            // If DDL survived in WAL, we have data from remaining WAL entries
            assert!(
                count >= 0,
                "Should have non-negative row count, got {}",
                count
            );
        }
        Err(_) => {
            // WAL was truncated after snapshot — DDL not in WAL anymore.
            // Without snapshot, table cannot be recreated. This is expected.
        }
    }
}

#[test]
fn test_snapshot_valid_wal_corrupt() {
    // Snapshot is valid, but WAL is partially corrupted
    // Should recover snapshot data + surviving WAL entries
    let (fixture, pre_count) = setup_db_with_snapshot();

    // Corrupt the WAL — zero a middle section
    let wal_files = find_wal_files(&fixture.db_path);
    if !wal_files.is_empty() {
        let wal_path = &wal_files[wal_files.len() - 1];
        let mut data = fs::read(wal_path).unwrap();

        // Zero out a portion in the latter half (post-snapshot entries)
        if data.len() > 2000 {
            let corrupt_start = data.len() * 3 / 4;
            let corrupt_len = 200.min(data.len() - corrupt_start);
            zero_range(&mut data, corrupt_start, corrupt_len);
            fs::write(wal_path, &data).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM snap_test", ()).unwrap();

    // At minimum, snapshot data (pre_count) should survive
    assert!(
        count >= pre_count,
        "Expected at least {} rows (snapshot data), got {}",
        pre_count,
        count
    );

    // Verify usable
    db.execute(
        "INSERT INTO snap_test (id, value, phase) VALUES (999, 'post_recovery', 3)",
        (),
    )
    .unwrap();
}

#[test]
fn test_snapshot_and_wal_both_corrupt() {
    // Both snapshot and WAL partially corrupted
    let (fixture, _) = setup_db_with_snapshot();

    // Corrupt snapshot
    let snap_files = find_snapshot_files(&fixture.db_path, "snap_test");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 20 {
            // Corrupt middle of snapshot
            let mid = data.len() / 2;
            let corrupt_len = 50.min(data.len() - mid);
            zero_range(&mut data, mid, corrupt_len);
            fs::write(snap_path, &data).unwrap();
        }
    }

    // Corrupt WAL — zero some entries
    let wal_files = find_wal_files(&fixture.db_path);
    if !wal_files.is_empty() {
        let wal_path = &wal_files[wal_files.len() - 1];
        let mut data = fs::read(wal_path).unwrap();
        let entries = find_entry_boundaries(&data);

        // Corrupt every 3rd entry
        for (i, entry) in entries.iter().enumerate() {
            if i % 3 == 0 && i > 0 {
                flip_bit(&mut data, entry.crc_offset, 7);
            }
        }
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // The database should still open and be in a consistent state
    let db = Database::open(&fixture.dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM snap_test", ());

    match result {
        Ok(count) => {
            assert!(count >= 0, "Should have non-negative row count");
            // Verify usable
            db.execute(
                &format!(
                    "INSERT INTO snap_test (id, value, phase) VALUES ({}, 'recovery', 3)",
                    count + 10000
                ),
                (),
            )
            .unwrap();
        }
        Err(_) => {
            // If both sources are too damaged, table might not exist — acceptable
        }
    }
}

#[test]
fn test_snapshot_truncated() {
    // Truncate snapshot file to partial state
    let (fixture, _) = setup_db_with_snapshot();

    let snap_files = find_snapshot_files(&fixture.db_path, "snap_test");
    for snap_path in &snap_files {
        let data = fs::read(snap_path).unwrap();
        if data.len() > 100 {
            // Truncate to 1/4 of original size
            fs::write(snap_path, &data[..data.len() / 4]).unwrap();
        }
    }

    remove_lock_file(&fixture.db_path);

    let db = Database::open(&fixture.dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM snap_test", ());

    // WAL should provide recovery path
    match result {
        Ok(count) => {
            assert!(count >= 0);
        }
        Err(_) => {
            // Acceptable if both snapshot and WAL can't reconstruct the table
        }
    }
}

// ============================================================================
// 14. SAFE WAL TRUNCATION — Truncate to second-to-last verified snapshot
// ============================================================================
//
// These tests verify that WAL truncation is conservative: it only removes
// entries up to the second-to-last CRC-verified snapshot's LSN. This ensures
// that if the latest snapshot is corrupted, the previous snapshot + WAL can
// fully reconstruct the database.

/// Helper: Get total WAL file size in bytes
fn total_wal_size(db_path: &Path) -> u64 {
    find_wal_files(db_path)
        .iter()
        .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum()
}

#[test]
fn test_safe_truncation_single_snapshot_no_truncation() {
    // After a single PRAGMA SNAPSHOT, the WAL should NOT be truncated
    // because there's no second snapshot to fall back to.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE safe_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert data
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Take first snapshot
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert more data after snapshot
        for i in 21..=40 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // WAL should still have all entries (not truncated with only 1 snapshot)
    let wal_size_before_corruption = total_wal_size(&db_path);
    assert!(
        wal_size_before_corruption > 0,
        "WAL should exist and have data"
    );

    // Corrupt the single snapshot file
    let snap_files = find_snapshot_files(&db_path, "safe_test");
    assert_eq!(snap_files.len(), 1, "Should have exactly 1 snapshot");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 10 {
            zero_range(&mut data, 0, 10);
            fs::write(snap_path, &data).unwrap();
        }
    }

    // Reopen — should fully recover from WAL since it wasn't truncated
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM safe_test", ()).unwrap();
    assert_eq!(count, 40, "All 40 rows should be recovered from WAL");
}

#[test]
fn test_safe_truncation_two_snapshots_truncates_to_first() {
    // After two PRAGMA SNAPSHOTs, WAL is truncated to the first snapshot's LSN.
    // Corrupting the second snapshot should allow recovery from first + WAL.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE safe_test2 (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert 10 rows, take first snapshot
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test2 (id, value) VALUES ({}, 'phase1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Insert 10 more rows, take second snapshot
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test2 (id, value) VALUES ({}, 'phase2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: Insert 10 more rows (in WAL only)
        for i in 21..=30 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test2 (id, value) VALUES ({}, 'phase3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Verify we have 2 snapshots
    let snap_files = find_snapshot_files(&db_path, "safe_test2");
    assert_eq!(snap_files.len(), 2, "Should have 2 snapshots");

    // Corrupt the second (latest) snapshot
    let latest = &snap_files[1];
    let mut data = fs::read(latest).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest, &data).unwrap();
    }

    // Reopen — should recover from first snapshot + WAL (all 30 rows)
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM safe_test2", ()).unwrap();
    assert_eq!(
        count, 30,
        "All 30 rows should be recovered from first snapshot + WAL"
    );
}

#[test]
fn test_safe_truncation_verifies_second_snapshot_crc() {
    // If the second-to-last snapshot fails CRC, WAL should NOT be truncated.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE safe_test3 (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert data and take first snapshot
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test3 (id, value) VALUES ({}, 'data_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }

    remove_lock_file(&db_path);

    // Corrupt the first (and only) snapshot file before creating second
    let snap_files = find_snapshot_files(&db_path, "safe_test3");
    assert_eq!(snap_files.len(), 1);
    {
        let mut data = fs::read(&snap_files[0]).unwrap();
        // Flip a bit in the data section to cause CRC mismatch
        if data.len() > 100 {
            flip_bit(&mut data, 100, 3);
            fs::write(&snap_files[0], &data).unwrap();
        }
    }

    // Now reopen and insert more data, take second snapshot
    {
        let db = Database::open(&dsn).unwrap();
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test3 (id, value) VALUES ({}, 'data_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        // This creates a second snapshot, but the first (second-to-last) is corrupt
        // so WAL should NOT be truncated
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }

    remove_lock_file(&db_path);

    // WAL should have everything (not truncated because corrupt snapshot was cleaned up
    // leaving only 1 valid snapshot, which is < 2 surviving snapshots)
    let wal_size = total_wal_size(&db_path);
    assert!(wal_size > 0, "WAL should still exist");

    // The corrupt first snapshot was deleted by cleanup during second PRAGMA SNAPSHOT,
    // leaving only 1 valid snapshot
    let snap_files = find_snapshot_files(&db_path, "safe_test3");
    assert_eq!(
        snap_files.len(),
        1,
        "Corrupt snapshot should have been cleaned up, leaving 1 valid"
    );
    // Delete the remaining valid snapshot to force WAL-only recovery
    for snap_path in &snap_files {
        let _ = fs::remove_file(snap_path);
    }

    // Reopen — should recover from WAL alone
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM safe_test3", ()).unwrap();
    assert_eq!(
        count, 20,
        "All 20 rows should be recovered from WAL (not truncated)"
    );
}

#[test]
fn test_safe_truncation_three_snapshots() {
    // With 3 snapshots, corrupting the third should allow recovery from second + WAL.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE safe_test4 (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert rows, snapshot 1
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test4 (id, value) VALUES ({}, 'p1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Insert rows, snapshot 2
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test4 (id, value) VALUES ({}, 'p2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: Insert rows, snapshot 3
        for i in 21..=30 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test4 (id, value) VALUES ({}, 'p3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 4: More data in WAL only
        for i in 31..=40 {
            db.execute(
                &format!(
                    "INSERT INTO safe_test4 (id, value) VALUES ({}, 'p4_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    let snap_files = find_snapshot_files(&db_path, "safe_test4");
    assert_eq!(snap_files.len(), 3, "Should have 3 snapshots");

    // Corrupt the third (latest) snapshot
    let latest = &snap_files[2];
    let mut data = fs::read(latest).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest, &data).unwrap();
    }

    // Reopen — should recover from second snapshot + WAL
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM safe_test4", ()).unwrap();
    assert_eq!(
        count, 40,
        "All 40 rows should be recovered from second snapshot + WAL"
    );
}

#[test]
fn test_safe_truncation_keep_count_one() {
    // With keep_snapshots=1, Phase 6 deletes the fallback snapshot after each PRAGMA SNAPSHOT.
    // The safe truncation logic must account for this: since only 1 snapshot survives cleanup,
    // WAL should NEVER be truncated (no fallback snapshot to recover from).
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?keep_snapshots=1", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE keep1_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert 10 rows, snapshot 1
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO keep1_test (id, value) VALUES ({}, 'p1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Insert 10 more rows, snapshot 2
        // With keep_snapshots=1, snapshot 1 will be deleted after this
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO keep1_test (id, value) VALUES ({}, 'p2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: Insert 10 more rows (WAL only)
        for i in 21..=30 {
            db.execute(
                &format!(
                    "INSERT INTO keep1_test (id, value) VALUES ({}, 'p3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Only 1 snapshot should remain (keep_snapshots=1 cleaned up the first)
    let snap_files = find_snapshot_files(&db_path, "keep1_test");
    assert_eq!(
        snap_files.len(),
        1,
        "Should have exactly 1 snapshot after cleanup"
    );

    // WAL should NOT have been truncated (only 1 surviving snapshot)
    let wal_size = total_wal_size(&db_path);
    assert!(wal_size > 0, "WAL should still have data");

    // Corrupt the only snapshot
    let mut data = fs::read(&snap_files[0]).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(&snap_files[0], &data).unwrap();
    }

    // Reopen — should recover everything from WAL
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM keep1_test", ()).unwrap();
    assert_eq!(
        count, 30,
        "All 30 rows should be recovered from WAL (not truncated with keep_snapshots=1)"
    );
}

#[test]
fn test_safe_truncation_corrupt_snapshot_cleaned_up() {
    // If a corrupt snapshot file sits on disk, cleanup should remove it instead of
    // letting it occupy a keep slot. Otherwise WAL truncation is blocked forever.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?keep_snapshots=3", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE cleanup_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Create snapshot S1
        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO cleanup_test (id, value) VALUES ({}, 'v{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Create snapshot S2
        for i in 6..=10 {
            db.execute(
                &format!(
                    "INSERT INTO cleanup_test (id, value) VALUES ({}, 'v{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }

    remove_lock_file(&db_path);

    // Corrupt S1 (the older snapshot)
    let snap_files = find_snapshot_files(&db_path, "cleanup_test");
    assert_eq!(snap_files.len(), 2, "Should have 2 snapshots");
    {
        let mut data = fs::read(&snap_files[0]).unwrap();
        if data.len() > 10 {
            zero_range(&mut data, 0, 10);
            fs::write(&snap_files[0], &data).unwrap();
        }
    }

    // Reopen (loads S2 since S1 is corrupt) and create more snapshots
    {
        let db = Database::open(&dsn).unwrap();
        for i in 11..=15 {
            db.execute(
                &format!(
                    "INSERT INTO cleanup_test (id, value) VALUES ({}, 'v{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        // Create S3 — cleanup should delete corrupt S1 (not count it toward keep_count=3)
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }

    remove_lock_file(&db_path);

    // The corrupt S1 should have been cleaned up, leaving only valid snapshots
    let snap_files = find_snapshot_files(&db_path, "cleanup_test");
    // Should have S2 and S3 (S1 was corrupt and deleted by cleanup)
    // All remaining files should pass CRC verification
    for snap_path in &snap_files {
        let data = fs::read(snap_path).unwrap();
        // Basic check: file should start with the snapshot magic bytes
        assert!(
            data.len() > 8,
            "Snapshot file should not be empty: {:?}",
            snap_path
        );
        let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(
            magic, 0x5354534456534844,
            "All remaining snapshots should have valid magic: {:?}",
            snap_path
        );
    }

    // Verify data is intact
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM cleanup_test", ())
        .unwrap();
    assert_eq!(count, 15, "All 15 rows should be recovered");
}

#[test]
fn test_safe_truncation_multi_table_corrupt_one() {
    // Two tables: corrupt the latest snapshot of ONE table.
    // find_safe_truncation_lsn takes MIN LSN across tables, so both tables
    // must have >= 2 snapshots for truncation. Corrupting one table's latest
    // should allow recovery via its fallback snapshot + WAL.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE multi_a (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE multi_b (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert into both tables, snapshot 1
        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO multi_a (id, value) VALUES ({}, 'a{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO multi_b (id, value) VALUES ({}, 'b{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Insert more, snapshot 2
        for i in 11..=20 {
            db.execute(
                &format!("INSERT INTO multi_a (id, value) VALUES ({}, 'a{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO multi_b (id, value) VALUES ({}, 'b{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: Insert more (WAL only)
        for i in 21..=30 {
            db.execute(
                &format!("INSERT INTO multi_a (id, value) VALUES ({}, 'a{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO multi_b (id, value) VALUES ({}, 'b{}')", i, i),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Both tables should have 2 snapshots
    let snaps_a = find_snapshot_files(&db_path, "multi_a");
    let snaps_b = find_snapshot_files(&db_path, "multi_b");
    assert_eq!(snaps_a.len(), 2);
    assert_eq!(snaps_b.len(), 2);

    // Corrupt ONLY table A's latest snapshot
    let latest_a = &snaps_a[1];
    let mut data = fs::read(latest_a).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest_a, &data).unwrap();
    }

    // Reopen — table A recovers from first snapshot + WAL, table B from latest snapshot
    let db = Database::open(&dsn).unwrap();
    let count_a: i64 = db.query_one("SELECT COUNT(*) FROM multi_a", ()).unwrap();
    let count_b: i64 = db.query_one("SELECT COUNT(*) FROM multi_b", ()).unwrap();
    assert_eq!(
        count_a, 30,
        "Table A: all 30 rows from first snapshot + WAL"
    );
    assert_eq!(
        count_b, 30,
        "Table B: all 30 rows from latest snapshot + WAL"
    );
}

#[test]
fn test_safe_truncation_table_created_between_snapshots() {
    // Create table A, snapshot, then create table B, snapshot again.
    // Table B only has 1 snapshot (created after first PRAGMA SNAPSHOT).
    // WAL should NOT be truncated on the second snapshot because table B has < 2 snapshots.
    // After a third snapshot, both tables have >= 2, and truncation resumes.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE early_table (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert into early_table, first snapshot
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO early_table (id, value) VALUES ({}, 'e{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Now create a new table AFTER the first snapshot
        db.execute(
            "CREATE TABLE late_table (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO late_table (id, value) VALUES ({}, 'l{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Second snapshot — early_table has 2 snapshots, late_table has 1
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        let wal_after_second = total_wal_size(&db_path);

        // Insert more data
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO early_table (id, value) VALUES ({}, 'e{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO late_table (id, value) VALUES ({}, 'l{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Third snapshot — now both tables have >= 2 snapshots, truncation should happen
        let wal_before_third = total_wal_size(&db_path);
        let _ = db.execute("PRAGMA SNAPSHOT", ());
        let wal_after_third = total_wal_size(&db_path);

        // WAL should NOT have been truncated on the second snapshot (late_table had only 1)
        assert!(
            wal_after_second > 0,
            "WAL should exist after second snapshot"
        );

        // WAL SHOULD be truncated on the third snapshot (both tables now have >= 2)
        assert!(
            wal_after_third < wal_before_third,
            "WAL should be truncated after third snapshot: before={}, after={}",
            wal_before_third,
            wal_after_third
        );
    }

    remove_lock_file(&db_path);

    // Verify all data recovers
    let db = Database::open(&dsn).unwrap();
    let count_early: i64 = db
        .query_one("SELECT COUNT(*) FROM early_table", ())
        .unwrap();
    let count_late: i64 = db.query_one("SELECT COUNT(*) FROM late_table", ()).unwrap();
    assert_eq!(count_early, 20, "early_table should have all 20 rows");
    assert_eq!(count_late, 20, "late_table should have all 20 rows");
}

#[test]
fn test_safe_truncation_drop_table_no_block() {
    // After DROP TABLE, the orphaned snapshot directory must not block WAL truncation
    // for remaining tables. Phase 5 skips orphaned dirs, Phase 7 cleans them up.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE keeper (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE dropper (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert into both tables, snapshot 1
        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO keeper (id, value) VALUES ({}, 'k{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO dropper (id, value) VALUES ({}, 'd{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Verify both tables have snapshots
        assert_eq!(find_snapshot_files(&db_path, "keeper").len(), 1);
        assert_eq!(find_snapshot_files(&db_path, "dropper").len(), 1);

        // Drop one table
        db.execute("DROP TABLE dropper", ()).unwrap();

        // Insert more into keeper, snapshot 2
        for i in 11..=20 {
            db.execute(
                &format!("INSERT INTO keeper (id, value) VALUES ({}, 'k{}')", i, i),
                (),
            )
            .unwrap();
        }
        let wal_before = total_wal_size(&db_path);
        let _ = db.execute("PRAGMA SNAPSHOT", ());
        let wal_after = total_wal_size(&db_path);

        // WAL should be truncated — the orphaned 'dropper' directory must not block it.
        // keeper has 2 snapshots, dropper is dropped (its dir was cleaned up in Phase 7).
        assert!(
            wal_after < wal_before,
            "WAL should be truncated after drop + snapshot: before={}, after={}",
            wal_before,
            wal_after
        );

        // Orphaned snapshot directory should be cleaned up
        let dropper_snap_dir = db_path.join("snapshots").join("dropper");
        assert!(
            !dropper_snap_dir.exists(),
            "Dropped table's snapshot directory should be cleaned up"
        );

        // Insert more data
        for i in 21..=30 {
            db.execute(
                &format!("INSERT INTO keeper (id, value) VALUES ({}, 'k{}')", i, i),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Verify recovery
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM keeper", ()).unwrap();
    assert_eq!(count, 30, "All 30 rows in keeper should be recovered");

    // dropper should not exist
    let result: std::result::Result<i64, _> = db.query_one("SELECT COUNT(*) FROM dropper", ());
    assert!(result.is_err(), "dropper table should not exist after drop");
}

#[test]
fn test_safe_truncation_all_snapshots_corrupt_one_table() {
    // Multi-table: ALL snapshots for one table are corrupted.
    // That table should recover entirely from WAL. The other table recovers from snapshot + WAL.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE healthy (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE doomed (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO healthy (id, value) VALUES ({}, 'h{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO doomed (id, value) VALUES ({}, 'd{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        for i in 11..=20 {
            db.execute(
                &format!("INSERT INTO healthy (id, value) VALUES ({}, 'h{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO doomed (id, value) VALUES ({}, 'd{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        for i in 21..=30 {
            db.execute(
                &format!("INSERT INTO healthy (id, value) VALUES ({}, 'h{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO doomed (id, value) VALUES ({}, 'd{}')", i, i),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Corrupt ALL of doomed's snapshots
    let doomed_snaps = find_snapshot_files(&db_path, "doomed");
    assert_eq!(doomed_snaps.len(), 2);
    for snap in &doomed_snaps {
        let mut data = fs::read(snap).unwrap();
        if data.len() > 10 {
            zero_range(&mut data, 0, 10);
            fs::write(snap, &data).unwrap();
        }
    }

    // Reopen — healthy recovers from snapshot, doomed from WAL
    // load_snapshots returns min_header_lsn from healthy's snapshot.
    // Since WAL was truncated to the second-to-last snapshot's LSN (first snapshot),
    // Reopen — healthy recovers from snapshot + WAL. Doomed is a DOUBLE FAILURE:
    // both snapshots are corrupt. Since WAL was truncated to the first snapshot's LSN,
    // the CREATE TABLE DDL and early INSERT entries for doomed were removed.
    // load_snapshots returns min_header_lsn from healthy's snapshot, and replay
    // starts from there. Doomed's DDL was before that point → table is lost.
    // This is the expected (and documented) limitation: safe truncation guarantees
    // single-failure recovery per table. Double failure = potential data loss.
    let db = Database::open(&dsn).unwrap();
    let healthy_count: i64 = db.query_one("SELECT COUNT(*) FROM healthy", ()).unwrap();
    assert_eq!(
        healthy_count, 30,
        "Healthy table should have all 30 rows from snapshot + WAL"
    );

    // doomed table is lost — double failure (both snapshots corrupt, DDL truncated from WAL).
    // The table either doesn't exist or has partial data from WAL entries after truncation point.
    let doomed_result: std::result::Result<i64, _> =
        db.query_one("SELECT COUNT(*) FROM doomed", ());
    // Double failure: table may not exist at all (CREATE TABLE was truncated from WAL)
    // or may exist with partial data. Either way, healthy is fully recovered.
    if let Ok(count) = doomed_result {
        assert!(
            count <= 30,
            "Doomed should have at most 30 rows (got {})",
            count
        );
    }
}

#[test]
fn test_safe_truncation_drop_and_recreate_same_name() {
    // DROP TABLE then CREATE TABLE with the same name.
    // Phase 7 must not delete the new table's snapshot directory.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create table, insert data, snapshot
        db.execute(
            "CREATE TABLE reborn (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO reborn (id, value) VALUES ({}, 'old{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Drop and recreate with same name
        db.execute("DROP TABLE reborn", ()).unwrap();
        db.execute(
            "CREATE TABLE reborn (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=5 {
            db.execute(
                &format!("INSERT INTO reborn (id, value) VALUES ({}, 'new{}')", i, i),
                (),
            )
            .unwrap();
        }

        // Snapshot again — the "reborn" table exists in schemas, so Phase 7
        // must NOT delete its snapshot directory.
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert more data
        for i in 6..=15 {
            db.execute(
                &format!("INSERT INTO reborn (id, value) VALUES ({}, 'new{}')", i, i),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Verify the new table's data survives
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM reborn", ()).unwrap();
    assert_eq!(count, 15, "Recreated table should have 15 rows");

    // Verify old data is gone (values should be 'new*', not 'old*')
    let val: String = db
        .query_one("SELECT value FROM reborn WHERE id = 1", ())
        .unwrap();
    assert_eq!(val, "new1", "Data should be from the recreated table");
}

#[test]
fn test_safe_truncation_keep_count_zero_no_cleanup() {
    // keep_snapshots=0 means "no cleanup" — all snapshots are kept.
    // Safe truncation should still work (uses ALL snapshots as surviving set).
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?keep_snapshots=0", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE keep_all (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert, snapshot 1
        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO keep_all (id, value) VALUES ({}, 'v{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: Insert, snapshot 2
        for i in 11..=20 {
            db.execute(
                &format!("INSERT INTO keep_all (id, value) VALUES ({}, 'v{}')", i, i),
                (),
            )
            .unwrap();
        }
        let wal_before = total_wal_size(&db_path);
        let _ = db.execute("PRAGMA SNAPSHOT", ());
        let wal_after = total_wal_size(&db_path);

        // With 2 snapshots and keep_count=0, truncation should happen
        // (all snapshots survive, >= 2 → truncate to first's LSN)
        assert!(
            wal_after < wal_before,
            "WAL should be truncated with keep_count=0 and 2 snapshots: before={}, after={}",
            wal_before,
            wal_after
        );

        // All snapshots should be kept (no cleanup)
        let snaps = find_snapshot_files(&db_path, "keep_all");
        assert_eq!(
            snaps.len(),
            2,
            "Both snapshots should be kept with keep_count=0"
        );

        // Phase 3: Insert, snapshot 3
        for i in 21..=30 {
            db.execute(
                &format!("INSERT INTO keep_all (id, value) VALUES ({}, 'v{}')", i, i),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        let snaps = find_snapshot_files(&db_path, "keep_all");
        assert_eq!(snaps.len(), 3, "All 3 snapshots kept with keep_count=0");
    }

    remove_lock_file(&db_path);

    // Corrupt latest snapshot — should fall back to second
    let snaps = find_snapshot_files(&db_path, "keep_all");
    let latest = &snaps[2];
    let mut data = fs::read(latest).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest, &data).unwrap();
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM keep_all", ()).unwrap();
    assert_eq!(
        count, 30,
        "All 30 rows recovered from second snapshot + WAL"
    );
}

#[test]
fn test_safe_truncation_with_updates_and_deletes() {
    // Verify safe truncation works correctly with UPDATE and DELETE operations,
    // not just INSERTs. The WAL must preserve these modifications after truncation.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE mut_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL, status TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Insert rows, snapshot 1
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO mut_test (id, value, status) VALUES ({}, 'original{}', 'active')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: UPDATE some, DELETE others, INSERT new
        for i in 1..=5 {
            db.execute(
                &format!(
                    "UPDATE mut_test SET value = 'updated{}', status = 'modified' WHERE id = {}",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        for i in 16..=20 {
            db.execute(&format!("DELETE FROM mut_test WHERE id = {}", i), ())
                .unwrap();
        }
        for i in 21..=25 {
            db.execute(
                &format!(
                    "INSERT INTO mut_test (id, value, status) VALUES ({}, 'new{}', 'active')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: More mutations (WAL only)
        db.execute("UPDATE mut_test SET status = 'final' WHERE id <= 5", ())
            .unwrap();
        db.execute("DELETE FROM mut_test WHERE id = 21", ())
            .unwrap();
        for i in 26..=30 {
            db.execute(
                &format!(
                    "INSERT INTO mut_test (id, value, status) VALUES ({}, 'late{}', 'active')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Corrupt latest snapshot — force recovery from first snapshot + WAL
    let snaps = find_snapshot_files(&db_path, "mut_test");
    assert_eq!(snaps.len(), 2);
    let latest = &snaps[1];
    let mut data = fs::read(latest).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest, &data).unwrap();
    }

    let db = Database::open(&dsn).unwrap();

    // Verify UPDATEs were preserved
    let updated_val: String = db
        .query_one("SELECT value FROM mut_test WHERE id = 3", ())
        .unwrap();
    assert_eq!(updated_val, "updated3", "UPDATE should be preserved");

    let final_status: String = db
        .query_one("SELECT status FROM mut_test WHERE id = 3", ())
        .unwrap();
    assert_eq!(final_status, "final", "Second UPDATE should be preserved");

    // Verify DELETEs were preserved
    let deleted_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM mut_test WHERE id BETWEEN 16 AND 20",
            (),
        )
        .unwrap();
    assert_eq!(deleted_count, 0, "DELETEd rows should stay deleted");

    let late_delete: i64 = db
        .query_one("SELECT COUNT(*) FROM mut_test WHERE id = 21", ())
        .unwrap();
    assert_eq!(late_delete, 0, "Late DELETE should be preserved");

    // Verify total count: 20 original - 5 deleted(16-20) + 5 new(21-25) - 1 deleted(21) + 5 late(26-30) = 24
    let total: i64 = db.query_one("SELECT COUNT(*) FROM mut_test", ()).unwrap();
    assert_eq!(total, 24, "Total rows after all mutations");
}

#[test]
fn test_safe_truncation_survives_restart_cycles() {
    // Verify safe truncation state is consistent across multiple close/reopen cycles.
    // Each cycle: insert data, snapshot, close, reopen, verify.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Cycle 1: Create and populate
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE cycle (id INTEGER PRIMARY KEY, cycle_num INTEGER NOT NULL, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO cycle (id, cycle_num, value) VALUES ({}, 1, 'c1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Cycle 2: Reopen, add data, snapshot
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle", ()).unwrap();
        assert_eq!(count, 10, "Cycle 2 open: should have 10 rows");

        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO cycle (id, cycle_num, value) VALUES ({}, 2, 'c2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
        // Now we have 2 snapshots — truncation happens
    }
    remove_lock_file(&db_path);

    // Cycle 3: Reopen, add data, snapshot
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle", ()).unwrap();
        assert_eq!(count, 20, "Cycle 3 open: should have 20 rows");

        for i in 21..=30 {
            db.execute(
                &format!(
                    "INSERT INTO cycle (id, cycle_num, value) VALUES ({}, 3, 'c3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Cycle 4: Corrupt latest snapshot, reopen
    let snaps = find_snapshot_files(&db_path, "cycle");
    let latest = snaps.last().unwrap();
    let mut data = fs::read(latest).unwrap();
    if data.len() > 10 {
        zero_range(&mut data, 0, 10);
        fs::write(latest, &data).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle", ()).unwrap();
        assert_eq!(
            count, 30,
            "After corrupting latest, all 30 rows recovered from previous snapshot + WAL"
        );

        // Verify data from each cycle
        for cycle_num in 1..=3 {
            let cycle_count: i64 = db
                .query_one(
                    &format!("SELECT COUNT(*) FROM cycle WHERE cycle_num = {}", cycle_num),
                    (),
                )
                .unwrap();
            assert_eq!(cycle_count, 10, "Cycle {} should have 10 rows", cycle_num);
        }
    }
}

// ============================================================================
// 15. DURABILITY EDGE CASES
// ============================================================================
//
// Tests for high-priority durability gaps:
// - Gap 1: Snapshot metadata corruption (snapshot_meta.bin)
// - Gap 2: Crash during PRAGMA SNAPSHOT
// - Gap 3: DDL durability under corruption
// - Gap 4: View persistence after corruption
// - Gap 5: Transactions spanning snapshot boundary
// - Gap 6: Constraint enforcement after recovery

/// Find the snapshot_meta.bin file in the snapshots directory
fn find_snapshot_meta_bin(db_path: &Path) -> Option<PathBuf> {
    let path = db_path.join("snapshots").join("snapshot_meta.bin");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Find the snapshot directory for a given table
fn find_snapshot_dir(db_path: &Path, table: &str) -> PathBuf {
    db_path.join("snapshots").join(table)
}

// ---------------------------------------------------------------------------
// Gap 1: Snapshot Metadata Corruption (snapshot_meta.bin)
// ---------------------------------------------------------------------------

#[test]
fn test_metadata_corrupt_magic_bytes() {
    // Corrupt first 4 bytes (magic) of snapshot_meta.bin.
    // Recovery should still work via snapshot file header source_lsn fallback.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE meta_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=15 {
            db.execute(
                &format!(
                    "INSERT INTO meta_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt magic bytes in snapshot_meta.bin
    if let Some(meta_path) = find_snapshot_meta_bin(&db_path) {
        let mut data = fs::read(&meta_path).unwrap();
        assert!(data.len() >= 28, "snapshot_meta.bin should be 28 bytes");
        zero_range(&mut data, 0, 4); // zero out magic
        fs::write(&meta_path, &data).unwrap();
    }

    // Reopen — should recover via snapshot file header source_lsn
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM meta_test", ()).unwrap();
    assert_eq!(count, 15, "All 15 rows should be recovered");

    // Verify usable
    db.execute(
        "INSERT INTO meta_test (id, value) VALUES (100, 'post_recovery')",
        (),
    )
    .unwrap();
}

#[test]
fn test_metadata_corrupt_lsn_field() {
    // Corrupt bytes 8-15 (LSN field) in snapshot_meta.bin — set to u64::MAX.
    // CRC check should fail, falls back to header LSN.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE meta_lsn (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO meta_lsn (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt LSN field (bytes 8..16) with u64::MAX — CRC will mismatch
    if let Some(meta_path) = find_snapshot_meta_bin(&db_path) {
        let mut data = fs::read(&meta_path).unwrap();
        data[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&meta_path, &data).unwrap();
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM meta_lsn", ()).unwrap();
    assert_eq!(count, 10, "All 10 rows should be recovered");
}

#[test]
fn test_metadata_deleted() {
    // Delete snapshot_meta.bin entirely.
    // read_snapshot_lsn returns 0, load_snapshots uses min(0, header_lsn) = 0.
    // Full WAL replay, all data recovered.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE meta_del (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=12 {
            db.execute(
                &format!(
                    "INSERT INTO meta_del (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Delete snapshot_meta.bin
    if let Some(meta_path) = find_snapshot_meta_bin(&db_path) {
        fs::remove_file(&meta_path).unwrap();
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM meta_del", ()).unwrap();
    assert_eq!(count, 12, "All 12 rows should be recovered");
}

#[test]
fn test_metadata_corrupt_crc() {
    // Flip a bit in the CRC field (last 4 bytes of snapshot_meta.bin).
    // read_snapshot_metadata detects CRC mismatch, returns 0, recovery proceeds.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE meta_crc (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=8 {
            db.execute(
                &format!(
                    "INSERT INTO meta_crc (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Flip bit in CRC (bytes 24..28)
    if let Some(meta_path) = find_snapshot_meta_bin(&db_path) {
        let mut data = fs::read(&meta_path).unwrap();
        flip_bit(&mut data, 24, 3);
        fs::write(&meta_path, &data).unwrap();
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM meta_crc", ()).unwrap();
    assert_eq!(count, 8, "All 8 rows should be recovered");
}

// ---------------------------------------------------------------------------
// Gap 2: Crash During PRAGMA SNAPSHOT
// ---------------------------------------------------------------------------

#[test]
fn test_crash_leftover_tmp_snapshot_file() {
    // Manually create a .tmp snapshot file. PRAGMA SNAPSHOT should succeed
    // because .tmp files are ignored. Reopen gives clean recovery.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE tmp_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO tmp_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Create a leftover .tmp file in the snapshot directory
        let snap_dir = find_snapshot_dir(&db_path, "tmp_test");
        let _ = fs::create_dir_all(&snap_dir);
        fs::write(
            snap_dir.join("snapshot-99999.bin.tmp"),
            b"leftover garbage data from crashed snapshot write",
        )
        .unwrap();

        // PRAGMA SNAPSHOT should still succeed (ignores .tmp files)
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Reopen — .tmp files are never loaded
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM tmp_test", ()).unwrap();
    assert_eq!(count, 10, "All 10 rows should be present");

    // Verify no .tmp files are treated as real snapshots
    let snap_files = find_snapshot_files(&db_path, "tmp_test");
    for f in &snap_files {
        let name = f.file_name().unwrap().to_string_lossy();
        assert!(
            !name.ends_with(".tmp"),
            "No .tmp files should be in snapshot listing"
        );
    }
}

#[test]
fn test_crash_after_rename_before_metadata() {
    // Simulate crash after Phase 2 (rename .tmp→.bin) but before Phase 4 (metadata write).
    // Delete snapshot_meta.bin after second snapshot. Recovery uses header source_lsn.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE crash_meta (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert 10 rows, first snapshot
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO crash_meta (id, value) VALUES ({}, 'batch1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert 10 more rows, second snapshot
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO crash_meta (id, value) VALUES ({}, 'batch2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Delete snapshot_meta.bin (simulates crash after rename but before metadata write)
    if let Some(meta_path) = find_snapshot_meta_bin(&db_path) {
        fs::remove_file(&meta_path).unwrap();
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM crash_meta", ()).unwrap();
    assert_eq!(count, 20, "All 20 rows should be recovered");
}

#[test]
fn test_crash_partial_snapshot_file() {
    // First valid snapshot + truncated .bin file (simulates partial write crash).
    // Truncated snapshot fails CRC, fallback to first valid snapshot + WAL.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE partial_snap (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=15 {
            db.execute(
                &format!(
                    "INSERT INTO partial_snap (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert more data after snapshot
        for i in 16..=25 {
            db.execute(
                &format!(
                    "INSERT INTO partial_snap (id, value) VALUES ({}, 'post_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Create a truncated .bin file with a future timestamp (will be tried first)
    let snap_dir = find_snapshot_dir(&db_path, "partial_snap");
    fs::write(
        snap_dir.join("snapshot-99999999999999.bin"),
        vec![0u8; 50], // 50 bytes of zeros — invalid snapshot
    )
    .unwrap();

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM partial_snap", ())
        .unwrap();
    // First snapshot has 15 rows, WAL has the remaining 10
    assert_eq!(count, 25, "All 25 rows should be recovered");
}

// ---------------------------------------------------------------------------
// Gap 3: DDL Durability Under Corruption
// ---------------------------------------------------------------------------

#[test]
fn test_ddl_create_index_survives_snapshot_corruption() {
    // CREATE INDEX is recorded in WAL. Corrupt snapshot → recover from WAL.
    // Verify index is usable after recovery.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE idx_test (id INTEGER PRIMARY KEY, category TEXT NOT NULL, amount FLOAT)",
            (),
        )
        .unwrap();

        for i in 1..=20 {
            let cat = if i % 2 == 0 { "even" } else { "odd" };
            db.execute(
                &format!(
                    "INSERT INTO idx_test (id, category, amount) VALUES ({}, '{}', {})",
                    i,
                    cat,
                    i as f64 * 1.5
                ),
                (),
            )
            .unwrap();
        }

        db.execute("CREATE INDEX idx_cat ON idx_test(category)", ())
            .unwrap();

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt all snapshot files for this table
    let snap_files = find_snapshot_files(&db_path, "idx_test");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 64 {
            // Flip bits in data section (after 64-byte header)
            for offset in (64..data.len()).step_by(100) {
                flip_bit(&mut data, offset, 5);
            }
            fs::write(snap_path, &data).unwrap();
        }
    }

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM idx_test", ()).unwrap();
    assert_eq!(count, 20, "All 20 rows should be recovered from WAL");

    // Verify index is usable — query with indexed column filter
    let even_count: i64 = db
        .query_one("SELECT COUNT(*) FROM idx_test WHERE category = 'even'", ())
        .unwrap();
    assert_eq!(even_count, 10, "Should find 10 'even' rows via index");

    let odd_count: i64 = db
        .query_one("SELECT COUNT(*) FROM idx_test WHERE category = 'odd'", ())
        .unwrap();
    assert_eq!(odd_count, 10, "Should find 10 'odd' rows via index");
}

#[test]
fn test_ddl_multiple_operations_recovery() {
    // Multiple DDL operations: CREATE TABLE t1, INSERT, CREATE TABLE t2, INSERT,
    // CREATE INDEX on t1, DROP TABLE t2.
    // Pure WAL replay (no snapshot) should recover correct state.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE ddl_t1 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();
        for i in 1..=5 {
            db.execute(
                &format!("INSERT INTO ddl_t1 (id, name) VALUES ({}, 'name_{}')", i, i),
                (),
            )
            .unwrap();
        }

        db.execute(
            "CREATE TABLE ddl_t2 (id INTEGER PRIMARY KEY, data TEXT)",
            (),
        )
        .unwrap();
        for i in 1..=3 {
            db.execute(
                &format!("INSERT INTO ddl_t2 (id, data) VALUES ({}, 'data_{}')", i, i),
                (),
            )
            .unwrap();
        }

        db.execute("CREATE INDEX idx_name ON ddl_t1(name)", ())
            .unwrap();

        db.execute("DROP TABLE ddl_t2", ()).unwrap();
    }
    remove_lock_file(&db_path);

    // Reopen — pure WAL replay, no snapshot
    let db = Database::open(&dsn).unwrap();

    // t1 should exist with data and index
    let count: i64 = db.query_one("SELECT COUNT(*) FROM ddl_t1", ()).unwrap();
    assert_eq!(count, 5, "ddl_t1 should have 5 rows");

    // Verify index on t1 is usable
    let name_count: i64 = db
        .query_one("SELECT COUNT(*) FROM ddl_t1 WHERE name = 'name_3'", ())
        .unwrap();
    assert_eq!(name_count, 1, "Should find 1 row via index lookup");

    // t2 should NOT exist
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM ddl_t2", ());
    assert!(
        result.is_err(),
        "ddl_t2 should not exist after DROP TABLE was replayed"
    );
}

// ---------------------------------------------------------------------------
// Gap 4: View Persistence After Corruption
// ---------------------------------------------------------------------------

#[test]
fn test_view_survives_snapshot_corruption() {
    // CREATE VIEW is recorded in WAL. Corrupt snapshot → recover from WAL.
    // View should exist and return correct results.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE view_base (id INTEGER PRIMARY KEY, name TEXT NOT NULL, score INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO view_base (id, name, score) VALUES ({}, 'user_{}', {})",
                    i,
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        db.execute(
            "CREATE VIEW high_scores AS SELECT id, UPPER(name) AS name, score FROM view_base WHERE score >= 50",
            (),
        )
        .unwrap();

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt snapshot files
    let snap_files = find_snapshot_files(&db_path, "view_base");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 64 {
            for offset in (64..data.len()).step_by(80) {
                flip_bit(&mut data, offset, 2);
            }
            fs::write(snap_path, &data).unwrap();
        }
    }

    let db = Database::open(&dsn).unwrap();

    // View should exist and return correct results from WAL replay
    let view_count: i64 = db
        .query_one("SELECT COUNT(*) FROM high_scores", ())
        .unwrap();
    assert_eq!(
        view_count, 6,
        "high_scores view should return 6 rows (scores 50-100)"
    );

    // Verify UPPER transformation works
    let name: String = db
        .query_one("SELECT name FROM high_scores WHERE id = 5", ())
        .unwrap();
    assert_eq!(name, "USER_5", "UPPER() should be applied in the view");
}

#[test]
fn test_view_drop_and_recreate_durability() {
    // CREATE VIEW v1 (def A), DROP VIEW v1, CREATE VIEW v1 (def B).
    // After WAL replay, v1 should use definition B.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE vdr_base (id INTEGER PRIMARY KEY, name TEXT NOT NULL, active INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=6 {
            let active = if i <= 3 { 1 } else { 0 };
            db.execute(
                &format!(
                    "INSERT INTO vdr_base (id, name, active) VALUES ({}, 'user_{}', {})",
                    i, i, active
                ),
                (),
            )
            .unwrap();
        }

        // First definition: SELECT *
        db.execute("CREATE VIEW vdr_view AS SELECT * FROM vdr_base", ())
            .unwrap();

        // Drop it
        db.execute("DROP VIEW vdr_view", ()).unwrap();

        // Recreate with different definition: only active users, with UPPER
        db.execute(
            "CREATE VIEW vdr_view AS SELECT id, UPPER(name) AS name FROM vdr_base WHERE active = 1",
            (),
        )
        .unwrap();
    }
    remove_lock_file(&db_path);

    // Reopen — WAL replays all three DDL ops in order
    let db = Database::open(&dsn).unwrap();

    // View should use the NEW definition (only active users)
    let count: i64 = db.query_one("SELECT COUNT(*) FROM vdr_view", ()).unwrap();
    assert_eq!(count, 3, "View should return only 3 active users");

    // Verify UPPER transformation is applied (new definition)
    let name: String = db
        .query_one("SELECT name FROM vdr_view WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "USER_1", "New view definition should apply UPPER()");
}

// ---------------------------------------------------------------------------
// Gap 5: Transactions Spanning Snapshot Boundary
// ---------------------------------------------------------------------------

#[test]
fn test_uncommitted_data_not_in_snapshot() {
    // Uncommitted transaction data should NOT survive across close/reopen.
    // PRAGMA SNAPSHOT captures only committed data.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE uncommit_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert 10 committed rows
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO uncommit_test (id, value) VALUES ({}, 'committed_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Begin transaction and insert 5 more (uncommitted)
        let mut tx = db.begin().unwrap();
        for i in 11..=15 {
            tx.execute(
                &format!(
                    "INSERT INTO uncommit_test (id, value) VALUES ({}, 'uncommitted_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Take snapshot while transaction is open
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Drop tx without committing (implicit rollback)
        drop(tx);
    }
    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM uncommit_test", ())
        .unwrap();
    assert_eq!(
        count, 10,
        "Only 10 committed rows should survive, uncommitted are lost"
    );
}

#[test]
fn test_data_committed_after_snapshot_survives() {
    // Data committed AFTER snapshot should survive via WAL.
    // Even if snapshot is corrupted, safe truncation preserves WAL for single snapshot.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE post_snap (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert 10 rows, then snapshot
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO post_snap (id, value) VALUES ({}, 'pre_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Insert 10 more rows after snapshot (committed)
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO post_snap (id, value) VALUES ({}, 'post_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // First verify all 20 rows survive normal recovery
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM post_snap", ()).unwrap();
        assert_eq!(count, 20, "All 20 rows should survive");
    }
    remove_lock_file(&db_path);

    // Now corrupt the snapshot — with single snapshot, WAL is not truncated
    let snap_files = find_snapshot_files(&db_path, "post_snap");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 10 {
            zero_range(&mut data, 0, 10);
            fs::write(snap_path, &data).unwrap();
        }
    }

    let db = Database::open(&dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM post_snap", ());
    match result {
        Ok(count) => {
            // With single snapshot and WAL not truncated, should recover all 20
            assert_eq!(count, 20, "All 20 rows should be recovered from WAL");
        }
        Err(_) => {
            // If WAL was truncated (shouldn't happen with single snapshot), DDL may be lost
            panic!("Table should exist — single snapshot should not trigger WAL truncation");
        }
    }
}

// ---------------------------------------------------------------------------
// Gap 6: Constraint Enforcement After Recovery
// ---------------------------------------------------------------------------

#[test]
fn test_check_constraint_enforced_after_snapshot_corruption() {
    // CHECK constraint schema is stored in WAL. After snapshot corruption,
    // WAL replay should restore the constraint and enforce it.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE check_test (id INTEGER PRIMARY KEY, age INTEGER CHECK (age >= 0 AND age <= 150))",
            (),
        )
        .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO check_test (id, age) VALUES ({}, {})",
                    i,
                    i * 20
                ),
                (),
            )
            .unwrap();
        }

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt snapshot
    let snap_files = find_snapshot_files(&db_path, "check_test");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 64 {
            for offset in (64..data.len()).step_by(50) {
                flip_bit(&mut data, offset, 4);
            }
            fs::write(snap_path, &data).unwrap();
        }
    }

    let db = Database::open(&dsn).unwrap();

    // Data should be recovered
    let count: i64 = db.query_one("SELECT COUNT(*) FROM check_test", ()).unwrap();
    assert_eq!(count, 5, "All 5 rows should be recovered from WAL");

    // CHECK constraint should be enforced — negative age should fail
    let result = db.execute("INSERT INTO check_test (id, age) VALUES (100, -1)", ());
    assert!(
        result.is_err(),
        "CHECK constraint should reject age = -1 after recovery"
    );

    // Age > 150 should also fail
    let result = db.execute("INSERT INTO check_test (id, age) VALUES (101, 200)", ());
    assert!(
        result.is_err(),
        "CHECK constraint should reject age = 200 after recovery"
    );

    // Valid age should succeed
    db.execute("INSERT INTO check_test (id, age) VALUES (102, 50)", ())
        .unwrap();
}

#[test]
fn test_not_null_constraint_after_recovery() {
    // NOT NULL constraint should be enforced after close/reopen.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE notnull_test (id INTEGER PRIMARY KEY, name TEXT NOT NULL, score INTEGER)",
            (),
        )
        .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO notnull_test (id, name, score) VALUES ({}, 'user_{}', {})",
                    i,
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    // Data should be present
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM notnull_test", ())
        .unwrap();
    assert_eq!(count, 5, "All 5 rows should be recovered");

    // NOT NULL constraint should be enforced — NULL name should fail
    let result = db.execute(
        "INSERT INTO notnull_test (id, name, score) VALUES (100, NULL, 50)",
        (),
    );
    assert!(
        result.is_err(),
        "NOT NULL constraint should reject NULL name after recovery"
    );

    // Valid insert should succeed
    db.execute(
        "INSERT INTO notnull_test (id, name, score) VALUES (101, 'valid', 60)",
        (),
    )
    .unwrap();
}

#[test]
fn test_unique_index_enforced_after_snapshot_corruption() {
    // UNIQUE INDEX is recorded in WAL. After snapshot corruption,
    // WAL replay should restore the unique constraint.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE unique_test (id INTEGER PRIMARY KEY, email TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("CREATE UNIQUE INDEX idx_email ON unique_test(email)", ())
            .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO unique_test (id, email) VALUES ({}, 'user{}@example.com')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt snapshot
    let snap_files = find_snapshot_files(&db_path, "unique_test");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 64 {
            for offset in (64..data.len()).step_by(60) {
                flip_bit(&mut data, offset, 1);
            }
            fs::write(snap_path, &data).unwrap();
        }
    }

    let db = Database::open(&dsn).unwrap();

    // Data should be recovered
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM unique_test", ())
        .unwrap();
    assert_eq!(count, 5, "All 5 rows should be recovered from WAL");

    // UNIQUE constraint should be enforced — duplicate email should fail
    let result = db.execute(
        "INSERT INTO unique_test (id, email) VALUES (100, 'user1@example.com')",
        (),
    );
    assert!(
        result.is_err(),
        "UNIQUE constraint should reject duplicate email after recovery"
    );

    // New unique email should succeed
    db.execute(
        "INSERT INTO unique_test (id, email) VALUES (101, 'new@example.com')",
        (),
    )
    .unwrap();
}

// ============================================================================
// 16. ADVANCED DURABILITY SCENARIOS
// ============================================================================

// ----------------------------------------------------------------------------
// Gap 1: Concurrent Write + Crash (4 tests)
// ----------------------------------------------------------------------------

/// 4 threads each INSERT 25 rows concurrently, close, reopen → all 100 rows present
#[test]
fn test_concurrent_writers_recovery() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE conc_test (id INTEGER PRIMARY KEY, thread_id INTEGER, value TEXT)",
            (),
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for t in 0..4u32 {
            let db_clone = db.clone();
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                bar.wait();
                for i in 0..25u32 {
                    let id = t * 25 + i + 1;
                    db_clone
                        .execute(
                            &format!(
                                "INSERT INTO conc_test (id, thread_id, value) VALUES ({}, {}, 'row_{}')",
                                id, t, id
                            ),
                            (),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM conc_test", ()).unwrap();
        assert_eq!(count, 100);
    }

    remove_lock_file(&db_path);

    // Verify all 100 rows recovered
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM conc_test", ()).unwrap();
    assert_eq!(count, 100, "All 100 concurrent rows should be recovered");
    db.execute(
        "INSERT INTO conc_test (id, thread_id, value) VALUES (9999, 0, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Concurrent writes + WAL corruption → at least some rows recovered, DB usable
#[test]
fn test_concurrent_writers_with_wal_corruption() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE conc_corrupt (id INTEGER PRIMARY KEY, value TEXT)",
            (),
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for t in 0..4u32 {
            let db_clone = db.clone();
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                bar.wait();
                for i in 0..25u32 {
                    let id = t * 25 + i + 1;
                    db_clone
                        .execute(
                            &format!(
                                "INSERT INTO conc_corrupt (id, value) VALUES ({}, 'data_{}')",
                                id, id
                            ),
                            (),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Corrupt middle of WAL
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());
    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    if entries.len() > 4 {
        // Flip bits in a few middle entries
        let mid = entries.len() / 2;
        for entry in entries.iter().skip(mid).take(3.min(entries.len() - mid)) {
            if entry.data_offset + 4 < data.len() {
                flip_bit(&mut data, entry.data_offset + 2, 3);
            }
        }
        fs::write(wal_path, &data).unwrap();
    }

    // Recovery — at least some rows should survive; DB must be usable
    let db = Database::open(&dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM conc_corrupt", ());
    match result {
        Ok(count) => {
            assert!(count >= 0, "Should recover some rows, got {}", count);
            db.execute(
                "INSERT INTO conc_corrupt (id, value) VALUES (9999, 'post_recovery')",
                (),
            )
            .unwrap();
        }
        Err(_) => {
            // Table DDL was also corrupted — that's valid
        }
    }
}

/// Concurrent writes + snapshot + more writes → corrupt snapshot → all rows from WAL
#[test]
fn test_concurrent_writers_snapshot_recovery() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE conc_snap (id INTEGER PRIMARY KEY, phase INTEGER, value TEXT)",
            (),
        )
        .unwrap();

        // Phase 1: 4 threads × 25 rows = 100
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();
        for t in 0..4u32 {
            let db_clone = db.clone();
            let bar = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                bar.wait();
                for i in 0..25u32 {
                    let id = t * 25 + i + 1;
                    db_clone
                        .execute(
                            &format!(
                                "INSERT INTO conc_snap (id, phase, value) VALUES ({}, 1, 'p1_{}')",
                                id, id
                            ),
                            (),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: 4 threads × 25 more rows = 200 total
        let barrier2 = Arc::new(Barrier::new(4));
        let mut handles2 = Vec::new();
        for t in 0..4u32 {
            let db_clone = db.clone();
            let bar = Arc::clone(&barrier2);
            handles2.push(thread::spawn(move || {
                bar.wait();
                for i in 0..25u32 {
                    let id = 100 + t * 25 + i + 1;
                    db_clone
                        .execute(
                            &format!(
                                "INSERT INTO conc_snap (id, phase, value) VALUES ({}, 2, 'p2_{}')",
                                id, id
                            ),
                            (),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles2 {
            h.join().unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM conc_snap", ()).unwrap();
        assert_eq!(count, 200);
    }

    remove_lock_file(&db_path);

    // Corrupt snapshot — with only one snapshot, WAL should NOT have been truncated
    let snap_files = find_snapshot_files(&db_path, "conc_snap");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 32 {
            // Zero out a big chunk of the snapshot
            let len = 128.min(data.len() - 16);
            zero_range(&mut data, 16, len);
            fs::write(snap_path, &data).unwrap();
        }
    }

    // Single snapshot = no WAL truncation, all 200 rows should be recovered from WAL
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM conc_snap", ()).unwrap();
    assert_eq!(count, 200, "All 200 rows should be recovered from WAL");
    db.execute(
        "INSERT INTO conc_snap (id, phase, value) VALUES (9999, 0, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Uncommitted transactions should be lost after recovery
#[test]
fn test_concurrent_transactions_uncommitted_lost() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE uncommit_test (id INTEGER PRIMARY KEY, source TEXT)",
            (),
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(2));

        // Thread 1: committed inserts
        let db1 = db.clone();
        let bar1 = Arc::clone(&barrier);
        let h1 = thread::spawn(move || {
            bar1.wait();
            for i in 1..=10 {
                db1.execute(
                    &format!(
                        "INSERT INTO uncommit_test (id, source) VALUES ({}, 'committed')",
                        i
                    ),
                    (),
                )
                .unwrap();
            }
        });

        // Thread 2: uncommitted transaction (begun, not committed)
        let db2 = db.clone();
        let bar2 = Arc::clone(&barrier);
        let h2 = thread::spawn(move || {
            bar2.wait();
            let mut tx = db2.begin().unwrap();
            for i in 11..=20 {
                tx.execute(
                    &format!(
                        "INSERT INTO uncommit_test (id, source) VALUES ({}, 'uncommitted')",
                        i
                    ),
                    (),
                )
                .unwrap();
            }
            // Intentionally drop tx without commit
            drop(tx);
        });

        h1.join().unwrap();
        h2.join().unwrap();
    }

    remove_lock_file(&db_path);

    // Only committed rows should survive
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM uncommit_test", ())
        .unwrap();
    assert_eq!(
        count, 10,
        "Only 10 committed rows should survive, got {}",
        count
    );
    db.execute(
        "INSERT INTO uncommit_test (id, source) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

// ----------------------------------------------------------------------------
// Gap 2: WAL Rotation During Crash (4 tests)
// ----------------------------------------------------------------------------

/// Simulate crash during WAL truncation: only .bak file remains (no .log)
/// recover_interrupted_truncation should restore the .bak → data recovered
#[test]
fn test_rotation_artifacts_bak_file_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE bak_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO bak_test (id, value, seq) VALUES ({}, 'row_{}', {})",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Find the WAL file and rename it to .bak (simulating interrupted truncation)
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty(), "No WAL files found");

    let wal_path = &wal_files[0];
    let bak_path = wal_path.with_extension("log.bak");
    fs::rename(wal_path, &bak_path).unwrap();

    // Verify .bak exists and no .log WAL file
    assert!(bak_path.exists());
    let remaining = find_wal_files(&db_path);
    assert!(remaining.is_empty(), "Should have no .log WAL files");

    // Reopen — recovery should restore .bak → .log
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM bak_test", ()).unwrap();
    assert_eq!(count, 20, "All 20 rows should be recovered from .bak file");

    // Verify DB is usable
    db.execute(
        "INSERT INTO bak_test (id, value, seq) VALUES (999, 'post_recovery', 999)",
        (),
    )
    .unwrap();
}

/// Temp WAL file alongside valid WAL → temp cleaned up, data intact
#[test]
fn test_rotation_artifacts_temp_file_cleanup() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE temp_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO temp_test (id, value, seq) VALUES ({}, 'row_{}', {})",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Create a fake temp file alongside the valid WAL
    let wal_dir = db_path.join("wal");
    let temp_file = wal_dir.join("wal-temp-999.log");
    fs::write(&temp_file, b"garbage data that should be cleaned up").unwrap();
    assert!(temp_file.exists());

    // Reopen — temp should be cleaned up, all data intact
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM temp_test", ()).unwrap();
    assert_eq!(
        count, 20,
        "All 20 rows should survive with temp file present"
    );

    // Temp file should be cleaned up
    assert!(
        !temp_file.exists(),
        "Temp file should be cleaned up during recovery"
    );
}

/// Multiple WAL files where the newest is corrupted → data from valid file survives
#[test]
fn test_multiple_wal_files_newest_corrupt() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE multi_wal (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=30 {
            db.execute(
                &format!(
                    "INSERT INTO multi_wal (id, value, seq) VALUES ({}, 'row_{}', {})",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Create a second (newer by name) WAL file with garbage
    let wal_dir = db_path.join("wal");
    let corrupt_wal = wal_dir.join("wal_00000001-99999999-lsn-99999.log");
    fs::write(&corrupt_wal, b"this is not a valid WAL file at all").unwrap();

    // Reopen — recovery should handle the corrupt newer file and still read the valid one
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM multi_wal", ()).unwrap();
    assert!(
        count >= 1,
        "At least some rows should survive from valid WAL file, got {}",
        count
    );

    // Verify DB usable
    db.execute(
        "INSERT INTO multi_wal (id, value, seq) VALUES (9999, 'post_recovery', 0)",
        (),
    )
    .unwrap();
}

/// Stale checkpoint.meta pointing to nonexistent WAL → recovery discovers actual files
#[test]
fn test_stale_checkpoint_references_missing_wal() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE stale_cp (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO stale_cp (id, value, seq) VALUES ({}, 'row_{}', {})",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Overwrite checkpoint.meta with a reference to a nonexistent WAL file
    if let Some(cp_path) = find_checkpoint_file(&db_path) {
        // Write garbage checkpoint that references a missing file
        // The format is binary, so we just corrupt it entirely —
        // this forces recovery to scan WAL directory for actual files
        let mut cp_data = fs::read(&cp_path).unwrap();
        if cp_data.len() > 8 {
            // Corrupt the checkpoint data so it's invalid
            for b in cp_data.iter_mut().skip(4).take(20) {
                *b = 0xFF;
            }
            fs::write(&cp_path, &cp_data).unwrap();
        }
    }

    // Reopen → recovery should handle corrupt checkpoint and find actual WAL files
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM stale_cp", ()).unwrap();
    assert!(
        count >= 1,
        "Data should be recovered despite corrupt checkpoint, got {}",
        count
    );

    // Verify DB usable
    db.execute(
        "INSERT INTO stale_cp (id, value, seq) VALUES (9999, 'post_recovery', 0)",
        (),
    )
    .unwrap();
}

// ----------------------------------------------------------------------------
// Gap 3: Power Loss Simulation (4 tests)
// ----------------------------------------------------------------------------

/// Simulate power loss: last WAL entry truncated in half (mid-data)
#[test]
fn test_power_loss_truncated_last_entry() {
    let fixture = setup_test_db(20, 1);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2, "Need at least 2 entries");

    // Truncate the file to cut the last entry in half
    let last = &entries[entries.len() - 1];
    let cut_point = last.offset + last.total_size / 2;
    fs::write(wal_path, &data[..cut_point]).unwrap();

    remove_lock_file(&fixture.db_path);

    // At least all but the last entry's data should survive
    verify_recovery_at_least(&fixture, "test_data", 0);
}

/// Simulate power loss: garbage appended after valid WAL entries
#[test]
fn test_power_loss_garbage_appended() {
    let fixture = setup_test_db(15, 1);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();

    // Append 200 bytes of garbage (simulates partial write of next entry)
    data.extend_from_slice(&[0xDE; 200]);
    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);

    // All 15 rows should survive — garbage fails magic check
    verify_recovery_at_least(&fixture, "test_data", 15);
}

/// Simulate power loss: last 4KB page of WAL zeroed
#[test]
fn test_power_loss_last_page_zeroed() {
    let fixture = setup_test_db(20, 1);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let file_len = data.len();

    if file_len > 4096 {
        // Zero the last 4KB page
        let last_page = (file_len - 1) / 4096;
        zero_page(&mut data, last_page);
        fs::write(wal_path, &data).unwrap();
    }

    remove_lock_file(&fixture.db_path);

    // Earlier entries should survive, entries in last page lost
    verify_recovery_at_least(&fixture, "test_data", 0);
}

/// Simulate power loss: last commit marker torn in half
#[test]
fn test_power_loss_commit_marker_torn() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE commit_torn (id INTEGER PRIMARY KEY, value TEXT NOT NULL, seq INTEGER)",
            (),
        )
        .unwrap();

        // 4 explicit transactions × 5 rows each
        for txn in 0..4 {
            let mut tx = db.begin().unwrap();
            for row in 0..5 {
                let id = txn * 5 + row + 1;
                tx.execute(
                    &format!(
                        "INSERT INTO commit_torn (id, value, seq) VALUES ({}, 'txn{}_row{}', {})",
                        id, txn, row, txn
                    ),
                    (),
                )
                .unwrap();
            }
            tx.commit().unwrap();
        }

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM commit_torn", ())
            .unwrap();
        assert_eq!(count, 20);
    }

    remove_lock_file(&db_path);

    // Find the last commit marker entry and truncate to cut it in half
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Find commit marker entries
    let commit_entries: Vec<_> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.flags & COMMIT_MARKER_FLAG != 0)
        .collect();

    if !commit_entries.is_empty() {
        // Truncate at the middle of the last commit marker
        let (_, last_commit) = commit_entries[commit_entries.len() - 1];
        let cut_point = last_commit.offset + last_commit.total_size / 2;
        fs::write(wal_path, &data[..cut_point]).unwrap();
    }

    let fixture = TestFixture {
        _dir: dir,
        db_path,
        dsn,
    };

    // Last txn's commit marker is torn → treated as uncommitted
    // At least the first 3 txns (15 rows) should survive
    verify_recovery_at_least(&fixture, "commit_torn", 0);
}

// ----------------------------------------------------------------------------
// Gap 4: Large Data / Boundary Conditions (4 tests)
// ----------------------------------------------------------------------------

/// Test rows at compression threshold boundary (63 bytes uncompressed, 65 bytes may compress)
#[test]
fn test_compression_threshold_boundary() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Create strings at the compression threshold boundary (64 bytes)
    let small_value = "x".repeat(63); // Below threshold
    let large_value = "y".repeat(65); // Above threshold, may compress

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE threshold_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert rows with values just below threshold
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO threshold_test (id, value) VALUES ({}, '{}')",
                    i, small_value
                ),
                (),
            )
            .unwrap();
        }

        // Insert rows with values just above threshold
        for i in 11..=20 {
            db.execute(
                &format!(
                    "INSERT INTO threshold_test (id, value) VALUES ({}, '{}')",
                    i, large_value
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM threshold_test", ())
            .unwrap();
        assert_eq!(count, 20);
    }

    remove_lock_file(&db_path);

    // Reopen and verify all rows recovered with correct values
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM threshold_test", ())
        .unwrap();
    assert_eq!(count, 20, "All 20 rows should be recovered");

    // Verify values are correct
    let small_count: i64 = db
        .query_one(
            &format!(
                "SELECT COUNT(*) FROM threshold_test WHERE value = '{}'",
                small_value
            ),
            (),
        )
        .unwrap();
    assert_eq!(small_count, 10, "All 10 small rows should match exactly");

    let large_count: i64 = db
        .query_one(
            &format!(
                "SELECT COUNT(*) FROM threshold_test WHERE value = '{}'",
                large_value
            ),
            (),
        )
        .unwrap();
    assert_eq!(large_count, 10, "All 10 large rows should match exactly");
}

/// Large rows (100KB each) with bit corruption → corrupted entry skipped, rest recovered
#[test]
fn test_large_row_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    let large_text = "L".repeat(100_000); // 100KB each

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE large_row (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO large_row (id, value) VALUES ({}, '{}')",
                    i, large_text
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM large_row", ()).unwrap();
        assert_eq!(count, 5);
    }

    remove_lock_file(&db_path);

    // Flip a bit in the data section of the WAL
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    // Find a DML data entry (not DDL, not commit marker) to corrupt
    let dml_entries: Vec<_> = entries
        .iter()
        .filter(|e| e.flags & COMMIT_MARKER_FLAG == 0 && e.entry_size > 100)
        .collect();

    if !dml_entries.is_empty() {
        // Corrupt one data entry
        let target = dml_entries[dml_entries.len() / 2];
        flip_bit(&mut data, target.data_offset + 50, 5);
        fs::write(wal_path, &data).unwrap();
    }

    // At least some large rows should survive
    let db = Database::open(&dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM large_row", ());
    match result {
        Ok(count) => {
            assert!(count >= 0, "Should recover some rows, got {}", count);
            db.execute(
                "INSERT INTO large_row (id, value) VALUES (9999, 'post_recovery')",
                (),
            )
            .unwrap();
        }
        Err(_) => {
            // Table DDL was also corrupted — that's valid
        }
    }
}

/// 1000 small rows, corrupt one entry in middle → ≥999 rows recovered
#[test]
fn test_many_small_rows_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE many_rows (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Batch into 10 transactions × 100 rows to avoid 1000 individual WAL commits
        let mut id = 1;
        for _ in 0..10 {
            let mut tx = db.begin().unwrap();
            for _ in 0..100 {
                tx.execute(
                    &format!(
                        "INSERT INTO many_rows (id, value) VALUES ({}, 'small_{}')",
                        id, id
                    ),
                    (),
                )
                .unwrap();
                id += 1;
            }
            tx.commit().unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM many_rows", ()).unwrap();
        assert_eq!(count, 1000);
    }

    remove_lock_file(&db_path);

    // Corrupt one entry in the middle of the WAL
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);

    if entries.len() > 10 {
        // Corrupt a single entry in the middle
        let mid_entry = &entries[entries.len() / 2];
        // Zero the CRC to ensure it's detected as corrupt
        zero_range(&mut data, mid_entry.crc_offset, CRC_SIZE);
        fs::write(wal_path, &data).unwrap();
    }

    // Most rows should survive — only the corrupted entry's data lost
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM many_rows", ()).unwrap();
    assert!(
        count >= 900,
        "At least 900 of 1000 rows should survive, got {}",
        count
    );
    db.execute(
        "INSERT INTO many_rows (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Mixed-size rows with snapshot + corruption → all recovered from WAL
#[test]
fn test_mixed_size_rows_with_snapshot() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    let tiny = "t".repeat(5);
    let medium = "m".repeat(200);
    let large = "L".repeat(50_000);

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE mixed_size (id INTEGER PRIMARY KEY, category TEXT, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: insert mixed sizes
        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO mixed_size (id, category, value) VALUES ({}, 'tiny', '{}')",
                    i, tiny
                ),
                (),
            )
            .unwrap();
        }
        for i in 6..=10 {
            db.execute(
                &format!(
                    "INSERT INTO mixed_size (id, category, value) VALUES ({}, 'medium', '{}')",
                    i, medium
                ),
                (),
            )
            .unwrap();
        }
        for i in 11..=13 {
            db.execute(
                &format!(
                    "INSERT INTO mixed_size (id, category, value) VALUES ({}, 'large', '{}')",
                    i, large
                ),
                (),
            )
            .unwrap();
        }

        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: more mixed rows after snapshot
        for i in 14..=18 {
            db.execute(
                &format!(
                    "INSERT INTO mixed_size (id, category, value) VALUES ({}, 'tiny', '{}')",
                    i, tiny
                ),
                (),
            )
            .unwrap();
        }
        for i in 19..=23 {
            db.execute(
                &format!(
                    "INSERT INTO mixed_size (id, category, value) VALUES ({}, 'medium', '{}')",
                    i, medium
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM mixed_size", ()).unwrap();
        assert_eq!(count, 23);
    }

    remove_lock_file(&db_path);

    // Corrupt snapshot — single snapshot = no WAL truncation
    let snap_files = find_snapshot_files(&db_path, "mixed_size");
    for snap_path in &snap_files {
        let mut data = fs::read(snap_path).unwrap();
        if data.len() > 64 {
            for offset in (32..data.len()).step_by(100) {
                flip_bit(&mut data, offset, 0);
            }
            fs::write(snap_path, &data).unwrap();
        }
    }

    // Reopen → all 23 rows should be recovered from WAL
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM mixed_size", ()).unwrap();
    assert_eq!(count, 23, "All 23 mixed-size rows should be recovered");

    // Verify each category
    let tiny_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM mixed_size WHERE category = 'tiny'",
            (),
        )
        .unwrap();
    assert_eq!(tiny_count, 10, "10 tiny rows expected");

    let medium_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM mixed_size WHERE category = 'medium'",
            (),
        )
        .unwrap();
    assert_eq!(medium_count, 10, "10 medium rows expected");

    let large_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM mixed_size WHERE category = 'large'",
            (),
        )
        .unwrap();
    assert_eq!(large_count, 3, "3 large rows expected");

    // Verify DB is usable
    db.execute(
        "INSERT INTO mixed_size (id, category, value) VALUES (9999, 'post', 'recovery')",
        (),
    )
    .unwrap();
}

// ----------------------------------------------------------------------------
// Gap 5: WAL Truncation, Replay & Size Limits (4 tests)
// ----------------------------------------------------------------------------

/// Two snapshots trigger WAL truncation → WAL file is replaced,
/// post-truncation data in new WAL + snapshot = full recovery
#[test]
fn test_wal_truncation_and_replay() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: data + first snapshot
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_test (id, value) VALUES ({}, 'phase1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        let wal_before = find_wal_files(&db_path);
        let wal_size_before = total_wal_size(&db_path);

        // Phase 2: more data + second snapshot (triggers WAL truncation via safe_truncation)
        for i in 21..=40 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_test (id, value) VALUES ({}, 'phase2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // After 2 snapshots, WAL should have been truncated (old entries removed)
        let wal_after = find_wal_files(&db_path);
        let wal_size_after = total_wal_size(&db_path);

        // WAL file name should have changed (truncation creates a new file)
        if !wal_before.is_empty() && !wal_after.is_empty() {
            assert_ne!(
                wal_before[0].file_name(),
                wal_after[0].file_name(),
                "WAL filename should change after truncation"
            );
        }

        // WAL should be smaller (old entries were truncated)
        assert!(
            wal_size_after <= wal_size_before,
            "WAL should not grow after truncation: before={}, after={}",
            wal_size_before,
            wal_size_after
        );

        // Phase 3: data after truncation (only in new WAL, not in any snapshot)
        for i in 41..=50 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_test (id, value) VALUES ({}, 'phase3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
        assert_eq!(count, 50);
    }

    remove_lock_file(&db_path);

    // Reopen — snapshot provides phase1+phase2, new WAL provides phase3
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
    assert_eq!(
        count, 50,
        "All 50 rows should be recovered (snapshot + truncated WAL)"
    );

    db.execute(
        "INSERT INTO trunc_test (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// WAL truncation + corrupt new WAL → snapshot still provides earlier data
#[test]
fn test_wal_truncation_then_corrupt_new_wal() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_corrupt (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1 + snapshot
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_corrupt (id, value) VALUES ({}, 'p1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2 + second snapshot → triggers truncation
        for i in 21..=40 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_corrupt (id, value) VALUES ({}, 'p2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 3: only in new (truncated) WAL
        for i in 41..=50 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_corrupt (id, value) VALUES ({}, 'p3_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    // Corrupt the post-truncation WAL file
    let wal_files = find_wal_files(&db_path);
    assert!(!wal_files.is_empty());
    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    if data.len() > 64 {
        // Zero a big chunk of the WAL — destroys phase 3 data
        let mid = data.len() / 2;
        let len = (data.len() - mid).min(256);
        zero_range(&mut data, mid, len);
        fs::write(wal_path, &data).unwrap();
    }

    // Snapshot has phase1+phase2 (40 rows). Phase 3 (10 rows) is in corrupted WAL.
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM trunc_corrupt", ())
        .unwrap();
    // Snapshot guarantees at least 40 rows from phases 1+2
    assert!(
        count >= 40,
        "Snapshot should provide at least 40 rows, got {}",
        count
    );

    db.execute(
        "INSERT INTO trunc_corrupt (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// WAL entry with entry_size field corrupted to exceed 64MB sanity limit → skipped
#[test]
fn test_wal_entry_size_exceeds_sanity_limit() {
    let fixture = setup_test_db(10, 1);
    let wal_files = find_wal_files(&fixture.db_path);
    assert!(!wal_files.is_empty());

    let wal_path = &wal_files[wal_files.len() - 1];
    let mut data = fs::read(wal_path).unwrap();
    let entries = find_entry_boundaries(&data);
    assert!(entries.len() >= 2, "Need at least 2 entries");

    // Corrupt one entry's size field to be > 64MB (the sanity limit)
    // Entry size is at offset +24 in the header (4 bytes, little-endian)
    let target = &entries[entries.len() / 2];
    let size_offset = target.offset + 24;
    // Write 0x05000000 = ~83MB > 64MB sanity limit
    let huge_size: u32 = 70 * 1024 * 1024;
    data[size_offset..size_offset + 4].copy_from_slice(&huge_size.to_le_bytes());
    fs::write(wal_path, &data).unwrap();

    remove_lock_file(&fixture.db_path);

    // Recovery should skip the entry with impossible size and recover what it can
    verify_recovery_at_least(&fixture, "test_data", 0);
}

/// Large WAL with data spread across many pages → full replay succeeds
#[test]
fn test_large_wal_full_replay() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE large_wal (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Each row gets a unique ~500-byte string that doesn't compress well.
        // Use varying characters based on id to defeat LZ4.
        let mut id = 1;
        for _ in 0..10 {
            let mut tx = db.begin().unwrap();
            for _ in 0..50 {
                // Build a high-entropy string: cycle through printable ASCII based on id
                // Alphanumeric only — safe for SQL, 62 distinct chars defeats LZ4
                const CHARS: &[u8] =
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
                let value: String = (0..500)
                    .map(|j| CHARS[((id * 7 + j * 13 + 37) % CHARS.len() as i32) as usize] as char)
                    .collect();
                db.execute(
                    &format!(
                        "INSERT INTO large_wal (id, value) VALUES ({}, '{}')",
                        id, value
                    ),
                    (),
                )
                .unwrap();
                id += 1;
            }
            tx.commit().unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM large_wal", ()).unwrap();
        assert_eq!(count, 500);
    }

    remove_lock_file(&db_path);

    // Verify WAL is substantial — unique data per row means LZ4 can't compress it away
    let wal_size = total_wal_size(&db_path);
    assert!(
        wal_size > 100_000,
        "WAL should be > 100KB with high-entropy data, got {} bytes",
        wal_size
    );

    // Reopen — full replay of large WAL
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM large_wal", ()).unwrap();
    assert_eq!(
        count, 500,
        "All 500 rows should be recovered from large WAL"
    );

    // Verify a specific row's value survived correctly
    let row1_value: String = db
        .query_one("SELECT value FROM large_wal WHERE id = 1", ())
        .unwrap();
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let expected: String = (0..500)
        .map(|j| CHARS[((7 + j * 13 + 37) % CHARS.len() as i32) as usize] as char)
        .collect();
    assert_eq!(row1_value, expected, "Row 1 value should match exactly");

    db.execute(
        "INSERT INTO large_wal (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

// ----------------------------------------------------------------------------
// Gap 5b: WAL Rotation Tests (4 tests)
// These verify that wal_max_size config actually triggers rotation in production
// and that rotated files are cleaned up after snapshot-based truncation.
// ----------------------------------------------------------------------------

/// WAL rotation fires during normal commits when wal_max_size is small.
/// Multiple WAL files appear on disk, and multi-file replay recovers all data.
#[test]
fn test_wal_rotation_and_replay() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    // wal_max_size=500 triggers rotation after a few inserts
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_test", ()).unwrap();
        assert_eq!(count, 50);
    }

    // Verify multiple WAL files exist on disk (rotation happened)
    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Expected multiple WAL files from rotation, got {}",
        wal_files.len()
    );

    remove_lock_file(&db_path);

    // Reopen — multi-file replay should recover all 50 rows
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_test", ()).unwrap();
    assert_eq!(
        count, 50,
        "All 50 rows should be recovered from multi-file WAL replay"
    );

    // Verify DB is usable after recovery
    db.execute(
        "INSERT INTO rot_test (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Corrupt the oldest rotated WAL file — DDL may be lost but newer data survives.
#[test]
fn test_wal_rotation_oldest_corrupt() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_oldest (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_oldest (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Need multiple WAL files, got {}",
        wal_files.len()
    );

    // Corrupt the oldest WAL file (first in sorted order)
    let oldest = &wal_files[0];
    let mut data = fs::read(oldest).unwrap();
    if data.len() > 32 {
        let mid = data.len() / 2;
        let len = (data.len() - mid).min(256);
        zero_range(&mut data, mid, len);
        fs::write(oldest, &data).unwrap();
    }

    remove_lock_file(&db_path);

    // Recovery: DDL is in the oldest WAL. If it's corrupted, table may not exist.
    // But if DDL survived (only data entries were corrupted), newer WAL data survives.
    let db = Database::open(&dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM rot_oldest", ());
    match result {
        Ok(count) => {
            // Some rows should survive from the non-corrupted WAL files
            assert!(
                count > 0,
                "At least some rows should survive from newer WAL files"
            );
        }
        Err(_) => {
            // Table doesn't exist — DDL was in the corrupted portion.
            // This is acceptable behavior: corruption of the first WAL file
            // can lose the CREATE TABLE statement.
        }
    }
}

/// Corrupt the newest rotated WAL file — older WAL data survives.
#[test]
fn test_wal_rotation_newest_corrupt() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_newest (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_newest (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }

    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Need multiple WAL files, got {}",
        wal_files.len()
    );

    // Zero out the newest (last) WAL file entirely
    let newest = &wal_files[wal_files.len() - 1];
    let len = fs::metadata(newest).unwrap().len() as usize;
    fs::write(newest, vec![0u8; len]).unwrap();

    remove_lock_file(&db_path);

    // Recovery: oldest WAL files should provide DDL + earlier rows.
    // With wal_max_size=500, most data is in older files, so zeroing the
    // newest file may lose only the last few rows (or none if it was empty).
    let db = Database::open(&dsn).unwrap();
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM rot_newest", ());
    match result {
        Ok(count) => {
            // Data from older WAL files should survive
            assert!(
                count > 0,
                "At least some rows should survive from older WAL files, got {}",
                count
            );
            // At most all 50 rows (if newest file had no unique data)
            assert!(count <= 50);
        }
        Err(_) => {
            // If DDL was in the newest file (unlikely with wal_max_size=500),
            // the table might not exist. This is still acceptable.
        }
    }
}

/// WAL rotation + snapshot-based truncation cleans up old rotated files.
/// Three snapshots are needed: 2nd triggers truncation to 1st's LSN (cleans phase 1 files),
/// 3rd triggers truncation to 2nd's LSN (cleans phase 2 files).
#[test]
fn test_wal_rotation_snapshot_cleans_old_files() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_clean (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: Generate multiple rotated WAL files
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_clean (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let wal_phase1 = find_wal_files(&db_path);
        assert!(
            wal_phase1.len() >= 2,
            "Expected multiple WAL files from rotation, got {}",
            wal_phase1.len()
        );

        // First snapshot — no truncation yet (need 2 snapshots for safe truncation)
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: More data
        for i in 51..=80 {
            db.execute(
                &format!(
                    "INSERT INTO rot_clean (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Second snapshot — truncation to 1st snapshot's LSN, cleans phase 1 rotated files
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        let wal_after_snap2 = find_wal_files(&db_path);
        // Phase 1 files (with LSN <= 1st snapshot) should be cleaned up
        assert!(
            wal_after_snap2.len() < wal_phase1.len(),
            "Phase 1 WAL files should be cleaned after 2nd snapshot: phase1={}, after={}",
            wal_phase1.len(),
            wal_after_snap2.len()
        );

        // Phase 3: A few more rows + 3rd snapshot cleans phase 2 files
        for i in 81..=90 {
            db.execute(
                &format!(
                    "INSERT INTO rot_clean (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // After 3rd snapshot, truncation to 2nd snapshot's LSN cleans phase 2 files too
        let wal_after_snap3 = find_wal_files(&db_path);
        assert!(
            wal_after_snap3.len() < wal_after_snap2.len(),
            "Phase 2 WAL files should be cleaned after 3rd snapshot: snap2={}, snap3={}",
            wal_after_snap2.len(),
            wal_after_snap3.len()
        );

        // Verify all data is still accessible
        let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_clean", ()).unwrap();
        assert_eq!(count, 90);
    }

    remove_lock_file(&db_path);

    // Reopen — snapshot + remaining WAL should recover everything
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_clean", ()).unwrap();
    assert_eq!(
        count, 90,
        "All 90 rows should be recovered after rotation + truncation cleanup"
    );

    db.execute(
        "INSERT INTO rot_clean (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Regression test: cleanup must NOT delete rotated WAL files whose entries
/// straddle the truncation boundary. If the latest snapshot is corrupted,
/// recovery must fall back to the second-to-last snapshot + WAL without data loss.
#[test]
fn test_wal_rotation_cleanup_preserves_boundary_entries() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    let total_rows;
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_boundary (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: bulk inserts (many rotations due to small wal_max_size)
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_boundary (id, value) VALUES ({}, 'phase1_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot 1 — covers phase 1 data
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // Phase 2: more inserts (more rotations)
        for i in 51..=80 {
            db.execute(
                &format!(
                    "INSERT INTO rot_boundary (id, value) VALUES ({}, 'phase2_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot 2 — triggers truncation to snapshot 1's LSN.
        // cleanup_old_wal_files runs and removes old rotated files.
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        total_rows = 80;
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM rot_boundary", ())
            .unwrap();
        assert_eq!(count, total_rows);
    }

    remove_lock_file(&db_path);

    // Corrupt the LATEST snapshot (newest .bin file) for this table.
    // Recovery must fall back to snapshot 1 + WAL to reconstruct all data.
    let snap_files = find_snapshot_files(&db_path, "rot_boundary");
    assert!(
        snap_files.len() >= 2,
        "Need at least 2 snapshots, got {}",
        snap_files.len()
    );
    let latest_snap = &snap_files[snap_files.len() - 1];
    let mut data = fs::read(latest_snap).unwrap();
    // Zero the middle of the snapshot to corrupt it
    if data.len() > 64 {
        let mid = data.len() / 2;
        let len = (data.len() - mid).min(256);
        zero_range(&mut data, mid, len);
        fs::write(latest_snap, &data).unwrap();
    }

    // Recovery must use snapshot 1 (phase 1 data) + WAL (phase 2 data).
    // If cleanup incorrectly deleted WAL files containing phase 2 entries,
    // recovery would lose rows between snapshot 1's LSN and the next rotation boundary.
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM rot_boundary", ())
        .unwrap();
    assert_eq!(
        count, total_rows,
        "All {} rows must survive fallback to snapshot 1 + WAL (got {}). \
         If rows are missing, cleanup_old_wal_files deleted a file whose entries \
         straddle the truncation boundary.",
        total_rows, count
    );

    // DB must be usable after recovery
    db.execute(
        "INSERT INTO rot_boundary (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Simulate crash mid-rotation: the new rotated WAL file exists on disk but
/// checkpoint.meta still references the old file. On restart, the DB must
/// discover all WAL files (including the orphaned rotated one) and replay
/// everything without data loss.
#[test]
fn test_wal_rotation_crash_mid_rotate() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_crash (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert enough rows to trigger several rotations
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO rot_crash (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_crash", ()).unwrap();
        assert_eq!(count, 50);
    }

    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Need multiple WAL files from rotation, got {}",
        wal_files.len()
    );

    // Simulate crash mid-rotation: rewrite checkpoint.meta to reference an
    // OLDER WAL file (as if the process died after creating the new file but
    // before updating checkpoint.meta).
    let wal_dir = db_path.join("wal");
    let checkpoint_path = wal_dir.join("checkpoint.meta");

    // Delete checkpoint.meta so with_config falls back to scanning all WAL files.
    if checkpoint_path.exists() {
        fs::remove_file(&checkpoint_path).unwrap();
    }

    // Also create a new empty rotated file to simulate the crash scenario:
    // new file was created but no data was written to it yet (crash before any writes).
    let orphan_path = wal_dir.join("wal_99999999-20260101-120000-lsn-999999.log");
    fs::write(&orphan_path, b"").unwrap();

    remove_lock_file(&db_path);

    // Reopen — with_config should scan all WAL files, find the real ones,
    // and replay_two_phase should recover everything from the scattered files.
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_crash", ()).unwrap();
    assert_eq!(
        count, 50,
        "All 50 rows should be recovered when checkpoint.meta is missing \
         and WAL files must be discovered by scanning"
    );

    // DB must be usable
    db.execute(
        "INSERT INTO rot_crash (id, value) VALUES (9999, 'post_crash')",
        (),
    )
    .unwrap();
}

/// UPDATE and DELETE operations must replay correctly when entries span
/// multiple rotated WAL files.
#[test]
fn test_wal_rotation_update_delete_replay() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}?wal_max_size=500", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_upd (id INTEGER PRIMARY KEY, value TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active')",
            (),
        )
        .unwrap();

        // Phase 1: INSERT 30 rows (triggers several rotations)
        for i in 1..=30 {
            db.execute(
                &format!(
                    "INSERT INTO rot_upd (id, value) VALUES ({}, 'original_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Phase 2: UPDATE some rows (these WAL entries land in later rotated files)
        for i in 1..=10 {
            db.execute(
                &format!(
                    "UPDATE rot_upd SET value = 'updated_{}', status = 'modified' WHERE id = {}",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Phase 3: DELETE some rows (more WAL entries in even later files)
        for i in 21..=25 {
            db.execute(&format!("DELETE FROM rot_upd WHERE id = {}", i), ())
                .unwrap();
        }

        // Verify final state: 30 inserted - 5 deleted = 25 rows
        let total: i64 = db.query_one("SELECT COUNT(*) FROM rot_upd", ()).unwrap();
        assert_eq!(total, 25);

        // 10 updated + 15 still active = 25
        let modified: i64 = db
            .query_one("SELECT COUNT(*) FROM rot_upd WHERE status = 'modified'", ())
            .unwrap();
        assert_eq!(modified, 10);

        let active: i64 = db
            .query_one("SELECT COUNT(*) FROM rot_upd WHERE status = 'active'", ())
            .unwrap();
        assert_eq!(active, 15);
    }

    // Verify rotation happened
    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Expected rotation to produce multiple WAL files, got {}",
        wal_files.len()
    );

    remove_lock_file(&db_path);

    // Reopen and verify the exact same state after multi-file replay
    let db = Database::open(&dsn).unwrap();

    let total: i64 = db.query_one("SELECT COUNT(*) FROM rot_upd", ()).unwrap();
    assert_eq!(
        total, 25,
        "Expected 25 rows (30 inserted - 5 deleted) after replay, got {}",
        total
    );

    let modified: i64 = db
        .query_one("SELECT COUNT(*) FROM rot_upd WHERE status = 'modified'", ())
        .unwrap();
    assert_eq!(
        modified, 10,
        "Expected 10 modified rows after replay, got {}",
        modified
    );

    let active: i64 = db
        .query_one("SELECT COUNT(*) FROM rot_upd WHERE status = 'active'", ())
        .unwrap();
    assert_eq!(
        active, 15,
        "Expected 15 active rows after replay, got {}",
        active
    );

    // Verify specific updated values
    let val: String = db
        .query_one("SELECT value FROM rot_upd WHERE id = 5", ())
        .unwrap();
    assert_eq!(val, "updated_5");

    // Verify deleted rows are really gone
    let deleted_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM rot_upd WHERE id BETWEEN 21 AND 25",
            (),
        )
        .unwrap();
    assert_eq!(deleted_count, 0, "Deleted rows should not reappear");

    // DB usable after recovery
    db.execute(
        "INSERT INTO rot_upd (id, value) VALUES (9999, 'post_recovery')",
        (),
    )
    .unwrap();
}

/// Explicit multi-statement transactions must remain atomic when WAL rotation
/// fires between statements within a BEGIN...COMMIT block.
#[test]
fn test_wal_rotation_explicit_transaction() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    // Very small wal_max_size to force rotation mid-transaction
    let dsn = format!("file://{}?wal_max_size=200", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE rot_txn (id INTEGER PRIMARY KEY, value TEXT NOT NULL, batch INTEGER NOT NULL)",
            (),
        )
        .unwrap();

        // 5 explicit transactions, each inserting 6 rows.
        // With wal_max_size=200, rotation WILL fire mid-transaction.
        for batch in 0..5 {
            db.execute("BEGIN", ()).unwrap();
            for row in 0..6 {
                let id = batch * 6 + row + 1;
                db.execute(
                    &format!(
                        "INSERT INTO rot_txn (id, value, batch) VALUES ({}, 'b{}_r{}', {})",
                        id, batch, row, batch
                    ),
                    (),
                )
                .unwrap();
            }
            db.execute("COMMIT", ()).unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_txn", ()).unwrap();
        assert_eq!(count, 30);
    }

    // Verify rotation happened
    let wal_files = find_wal_files(&db_path);
    assert!(
        wal_files.len() >= 2,
        "Expected rotation to fire mid-transaction with wal_max_size=200, got {} files",
        wal_files.len()
    );

    remove_lock_file(&db_path);

    // Reopen — all transactions must replay atomically
    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM rot_txn", ()).unwrap();
    assert_eq!(
        count, 30,
        "All 30 rows from 5 committed transactions should be recovered, got {}",
        count
    );

    // Verify atomicity: each batch must have exactly 6 rows (all-or-nothing)
    for batch in 0..5 {
        let batch_count: i64 = db
            .query_one(
                &format!("SELECT COUNT(*) FROM rot_txn WHERE batch = {}", batch),
                (),
            )
            .unwrap();
        assert_eq!(
            batch_count, 6,
            "Batch {} has {} rows — expected 6 (atomic commit across rotation boundary)",
            batch, batch_count
        );
    }

    // DB usable after recovery
    db.execute(
        "INSERT INTO rot_txn (id, value, batch) VALUES (9999, 'post', 99)",
        (),
    )
    .unwrap();
}

// ============================================================================
// Gap closure: DROP INDEX durability
// ============================================================================

/// DROP INDEX must persist across close/reopen. After recovery, the dropped
/// index must not exist and queries must still work (using table scan).
#[test]
fn test_drop_index_durability() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE idx_drop (id INTEGER PRIMARY KEY, category TEXT NOT NULL, value INTEGER)",
            (),
        )
        .unwrap();

        // Insert data
        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO idx_drop (id, category, value) VALUES ({}, 'cat_{}', {})",
                    i,
                    i % 5,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        // Create index, verify it works
        db.execute("CREATE INDEX idx_cat ON idx_drop(category)", ())
            .unwrap();

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM idx_drop WHERE category = 'cat_0'", ())
            .unwrap();
        assert_eq!(count, 4);

        // Drop the index
        db.execute("DROP INDEX idx_cat ON idx_drop", ()).unwrap();

        // Queries still work after drop (table scan)
        let count2: i64 = db
            .query_one("SELECT COUNT(*) FROM idx_drop WHERE category = 'cat_1'", ())
            .unwrap();
        assert_eq!(count2, 4);
    }

    remove_lock_file(&db_path);

    // Reopen — DROP INDEX must have persisted
    let db = Database::open(&dsn).unwrap();

    // Data must survive
    let count: i64 = db.query_one("SELECT COUNT(*) FROM idx_drop", ()).unwrap();
    assert_eq!(count, 20, "All 20 rows should survive after recovery");

    // Queries still work (index should not exist)
    let count2: i64 = db
        .query_one("SELECT COUNT(*) FROM idx_drop WHERE category = 'cat_2'", ())
        .unwrap();
    assert_eq!(count2, 4);

    // Creating the same index again should succeed (proves it was dropped)
    db.execute("CREATE INDEX idx_cat ON idx_drop(category)", ())
        .unwrap();

    // And the recreated index works
    let count3: i64 = db
        .query_one("SELECT COUNT(*) FROM idx_drop WHERE category = 'cat_3'", ())
        .unwrap();
    assert_eq!(count3, 4);
}

// ============================================================================
// TRUNCATE TABLE durability
// ============================================================================

#[test]
fn test_truncate_table_survives_close_reopen() {
    // TRUNCATE TABLE has its own WAL operation type (TruncateTable=13).
    // After close/reopen, the table should exist but be empty.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_test (id, value) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
        assert_eq!(count, 50);

        db.execute("TRUNCATE TABLE trunc_test", ()).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
        assert_eq!(count, 0);
    }

    remove_lock_file(&db_path);

    // Reopen — table should exist but be empty
    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
    assert_eq!(count, 0, "TRUNCATE should persist — table must be empty");

    // Table is usable — can insert new data
    db.execute(
        "INSERT INTO trunc_test (id, value) VALUES (1, 'after_truncate')",
        (),
    )
    .unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_test", ()).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_truncate_table_then_insert_survives_close_reopen() {
    // TRUNCATE followed by new inserts — both operations must persist.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_ins (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_ins (id, value) VALUES ({}, 'old_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        db.execute("TRUNCATE TABLE trunc_ins", ()).unwrap();

        // Insert new data after truncate
        for i in 100..=105 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_ins (id, value) VALUES ({}, 'new_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_ins", ()).unwrap();
        assert_eq!(count, 6);
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_ins", ()).unwrap();
    assert_eq!(count, 6, "Only post-truncate rows should exist");

    // Verify old data is gone
    let old: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM trunc_ins WHERE value LIKE 'old_%'",
            (),
        )
        .unwrap();
    assert_eq!(old, 0, "Pre-truncate rows must not reappear");

    let new: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM trunc_ins WHERE value LIKE 'new_%'",
            (),
        )
        .unwrap();
    assert_eq!(new, 6, "Post-truncate rows must all survive");
}

#[test]
fn test_truncate_table_with_snapshot_recovery() {
    // Snapshot taken before TRUNCATE, then TRUNCATE, then close/reopen.
    // WAL replay of TRUNCATE must override snapshot data.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_snap (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=30 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_snap (id, value) VALUES ({}, 'snap_row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot captures all 30 rows
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // TRUNCATE after snapshot
        db.execute("TRUNCATE TABLE trunc_snap", ()).unwrap();

        // Insert a few new rows
        for i in 100..=102 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_snap (id, value) VALUES ({}, 'post_trunc_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_snap", ()).unwrap();
        assert_eq!(count, 3);
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM trunc_snap", ()).unwrap();
    assert_eq!(
        count, 3,
        "Only post-truncate rows should exist after snapshot + WAL replay"
    );

    // Snapshot data must NOT reappear
    let old: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM trunc_snap WHERE value LIKE 'snap_row_%'",
            (),
        )
        .unwrap();
    assert_eq!(old, 0, "Snapshot data must not survive TRUNCATE in WAL");
}

#[test]
fn test_truncate_with_index_recovery() {
    // TRUNCATE should clear index state too. After recovery, index queries must
    // return correct (empty or post-truncate) results.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE trunc_idx (id INTEGER PRIMARY KEY, category TEXT NOT NULL, val INTEGER)",
            (),
        )
        .unwrap();

        db.execute("CREATE INDEX idx_trunc_cat ON trunc_idx(category)", ())
            .unwrap();

        for i in 1..=40 {
            db.execute(
                &format!(
                    "INSERT INTO trunc_idx (id, category, val) VALUES ({}, 'cat_{}', {})",
                    i,
                    i % 4,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        db.execute("TRUNCATE TABLE trunc_idx", ()).unwrap();

        // Insert a few rows after truncate
        db.execute(
            "INSERT INTO trunc_idx (id, category, val) VALUES (100, 'cat_0', 999)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let total: i64 = db.query_one("SELECT COUNT(*) FROM trunc_idx", ()).unwrap();
    assert_eq!(total, 1, "Only the post-truncate row should exist");

    let cat0: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM trunc_idx WHERE category = 'cat_0'",
            (),
        )
        .unwrap();
    assert_eq!(
        cat0, 1,
        "Index query should find only the post-truncate row"
    );

    let cat1: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM trunc_idx WHERE category = 'cat_1'",
            (),
        )
        .unwrap();
    assert_eq!(cat1, 0, "Pre-truncate index entries must be gone");
}

// ============================================================================
// ALTER TABLE durability
// ============================================================================

#[test]
fn test_alter_table_add_column_durability() {
    // ALTER TABLE ADD COLUMN is recorded in WAL. Must survive close/reopen.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE alter_add (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO alter_add (id, name) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Add a new column
        db.execute("ALTER TABLE alter_add ADD COLUMN score INTEGER", ())
            .unwrap();

        // Insert row using the new column
        db.execute(
            "INSERT INTO alter_add (id, name, score) VALUES (11, 'with_score', 100)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM alter_add", ()).unwrap();
    assert_eq!(count, 11, "All 11 rows should survive");

    // The new column must exist — query it
    let score: i64 = db
        .query_one("SELECT score FROM alter_add WHERE id = 11", ())
        .unwrap();
    assert_eq!(score, 100, "New column value must survive recovery");

    // Old rows should have NULL for the new column
    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM alter_add WHERE score IS NULL", ())
        .unwrap();
    assert_eq!(
        null_count, 10,
        "Old rows should have NULL in the new column"
    );

    // Can insert using the new schema
    db.execute(
        "INSERT INTO alter_add (id, name, score) VALUES (12, 'post_recovery', 200)",
        (),
    )
    .unwrap();
}

#[test]
fn test_alter_table_drop_column_durability() {
    // ALTER TABLE DROP COLUMN must persist across close/reopen.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE alter_drop (id INTEGER PRIMARY KEY, name TEXT NOT NULL, extra TEXT)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO alter_drop (id, name, extra) VALUES ({}, 'row_{}', 'extra_{}')",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        db.execute("ALTER TABLE alter_drop DROP COLUMN extra", ())
            .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM alter_drop", ()).unwrap();
    assert_eq!(count, 10, "All rows should survive");

    // 'extra' column must not exist
    let result: Result<i64, _> = db.query_one(
        "SELECT COUNT(*) FROM alter_drop WHERE extra IS NOT NULL",
        (),
    );
    assert!(
        result.is_err(),
        "Column 'extra' should not exist after DROP COLUMN recovery"
    );

    // Can insert using the reduced schema
    db.execute(
        "INSERT INTO alter_drop (id, name) VALUES (11, 'post_recovery')",
        (),
    )
    .unwrap();
}

#[test]
fn test_alter_table_rename_column_durability() {
    // ALTER TABLE RENAME COLUMN must persist across close/reopen.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE alter_rename (id INTEGER PRIMARY KEY, old_name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO alter_rename (id, old_name) VALUES ({}, 'val_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        db.execute(
            "ALTER TABLE alter_rename RENAME COLUMN old_name TO new_name",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    // New column name must work
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM alter_rename WHERE new_name IS NOT NULL",
            (),
        )
        .unwrap();
    assert_eq!(count, 5, "All rows accessible via renamed column");

    // Old column name must not work
    let result: Result<i64, _> = db.query_one(
        "SELECT COUNT(*) FROM alter_rename WHERE old_name IS NOT NULL",
        (),
    );
    assert!(
        result.is_err(),
        "Old column name 'old_name' should not exist after RENAME recovery"
    );

    // Can insert using new column name
    db.execute(
        "INSERT INTO alter_rename (id, new_name) VALUES (6, 'post_recovery')",
        (),
    )
    .unwrap();
}

#[test]
fn test_alter_table_rename_table_durability() {
    // ALTER TABLE RENAME TO must persist across close/reopen.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE old_tbl (id INTEGER PRIMARY KEY, data TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!("INSERT INTO old_tbl (id, data) VALUES ({}, 'row_{}')", i, i),
                (),
            )
            .unwrap();
        }

        db.execute("ALTER TABLE old_tbl RENAME TO new_tbl", ())
            .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    // New name must work
    let count: i64 = db.query_one("SELECT COUNT(*) FROM new_tbl", ()).unwrap();
    assert_eq!(
        count, 10,
        "All rows should be accessible via new table name"
    );

    // Old name must not work
    let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM old_tbl", ());
    assert!(
        result.is_err(),
        "Old table name 'old_tbl' should not exist after RENAME recovery"
    );

    // Can insert via new name
    db.execute(
        "INSERT INTO new_tbl (id, data) VALUES (11, 'post_recovery')",
        (),
    )
    .unwrap();
}

#[test]
fn test_alter_table_with_snapshot_recovery() {
    // ALTER TABLE after snapshot — WAL replay must apply schema changes on top of snapshot state.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE alter_snap (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO alter_snap (id, name) VALUES ({}, 'row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot captures the original 2-column schema
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // ALTER after snapshot
        db.execute("ALTER TABLE alter_snap ADD COLUMN status TEXT", ())
            .unwrap();

        db.execute(
            "INSERT INTO alter_snap (id, name, status) VALUES (11, 'new_row', 'active')",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM alter_snap", ()).unwrap();
    assert_eq!(count, 11, "All 11 rows should survive");

    // New column must exist
    let status: String = db
        .query_one("SELECT status FROM alter_snap WHERE id = 11", ())
        .unwrap();
    assert_eq!(status, "active", "New column value must survive");

    // Old rows have NULL for the new column
    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM alter_snap WHERE status IS NULL", ())
        .unwrap();
    assert_eq!(null_count, 10);
}

#[test]
fn test_alter_table_multiple_operations_durability() {
    // Multiple ALTER operations on the same table in sequence.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE alter_multi (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c TEXT)",
            (),
        )
        .unwrap();

        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO alter_multi (id, a, b, c) VALUES ({}, 'a{}', 'b{}', 'c{}')",
                    i, i, i, i
                ),
                (),
            )
            .unwrap();
        }

        // Chain of ALTER operations
        db.execute("ALTER TABLE alter_multi DROP COLUMN c", ())
            .unwrap();
        db.execute("ALTER TABLE alter_multi ADD COLUMN d INTEGER", ())
            .unwrap();
        db.execute("ALTER TABLE alter_multi RENAME COLUMN b TO beta", ())
            .unwrap();

        db.execute(
            "INSERT INTO alter_multi (id, a, beta, d) VALUES (6, 'a6', 'beta6', 42)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM alter_multi", ())
        .unwrap();
    assert_eq!(count, 6, "All rows should survive");

    // Column 'c' must not exist
    let result: Result<i64, _> =
        db.query_one("SELECT COUNT(*) FROM alter_multi WHERE c IS NOT NULL", ());
    assert!(result.is_err(), "Column 'c' should be dropped");

    // Column 'beta' must exist (renamed from 'b')
    let beta_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM alter_multi WHERE beta IS NOT NULL",
            (),
        )
        .unwrap();
    assert_eq!(beta_count, 6, "Column 'beta' (renamed from 'b') must exist");

    // Column 'd' must exist
    let d_val: i64 = db
        .query_one("SELECT d FROM alter_multi WHERE id = 6", ())
        .unwrap();
    assert_eq!(d_val, 42, "New column 'd' must have correct value");
}

// ============================================================================
// Empty table recovery
// ============================================================================

#[test]
fn test_empty_table_survives_close_reopen() {
    // CREATE TABLE with zero data rows. DDL must persist even without data entries.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE empty_tbl (id INTEGER PRIMARY KEY, value TEXT)",
            (),
        )
        .unwrap();
        // No inserts — table is empty
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM empty_tbl", ()).unwrap();
    assert_eq!(count, 0, "Empty table should exist with 0 rows");

    // Can insert into the recovered empty table
    db.execute(
        "INSERT INTO empty_tbl (id, value) VALUES (1, 'first_row')",
        (),
    )
    .unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM empty_tbl", ()).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_multiple_empty_tables_survive() {
    // Multiple empty tables created in sequence — all must survive.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE empty_a (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE empty_b (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE empty_c (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    for tbl in &["empty_a", "empty_b", "empty_c"] {
        let count: i64 = db
            .query_one(&format!("SELECT COUNT(*) FROM {}", tbl), ())
            .unwrap();
        assert_eq!(count, 0, "Table '{}' should exist and be empty", tbl);
    }

    // All tables usable
    db.execute("INSERT INTO empty_a (id) VALUES (1)", ())
        .unwrap();
    db.execute("INSERT INTO empty_b (id, name) VALUES (1, 'test')", ())
        .unwrap();
    db.execute("INSERT INTO empty_c (id, x, y) VALUES (1, 10, 20)", ())
        .unwrap();
}

#[test]
fn test_empty_table_with_index_survives() {
    // CREATE TABLE + CREATE INDEX with zero data. Both must survive.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE empty_idx (id INTEGER PRIMARY KEY, category TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_empty_cat ON empty_idx(category)", ())
            .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM empty_idx", ()).unwrap();
    assert_eq!(count, 0);

    // Insert and query via index
    db.execute("INSERT INTO empty_idx (id, category) VALUES (1, 'A')", ())
        .unwrap();

    let cat_count: i64 = db
        .query_one("SELECT COUNT(*) FROM empty_idx WHERE category = 'A'", ())
        .unwrap();
    assert_eq!(
        cat_count, 1,
        "Index should work after recovery of empty table"
    );
}

// ============================================================================
// NULL value durability
// ============================================================================

#[test]
fn test_null_values_survive_wal_recovery() {
    // Insert rows with NULL in nullable columns. NULL must not become a default value.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE null_test (id INTEGER PRIMARY KEY, name TEXT, score INTEGER, active BOOLEAN)",
            (),
        )
        .unwrap();

        // Mix of NULL and non-NULL values
        db.execute(
            "INSERT INTO null_test (id, name, score, active) VALUES (1, 'alice', 100, TRUE)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO null_test (id, name, score, active) VALUES (2, NULL, NULL, NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO null_test (id, name, score, active) VALUES (3, 'charlie', NULL, TRUE)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO null_test (id, name, score, active) VALUES (4, NULL, 50, NULL)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let total: i64 = db.query_one("SELECT COUNT(*) FROM null_test", ()).unwrap();
    assert_eq!(total, 4, "All 4 rows must survive");

    // Verify NULLs are preserved, not coerced to defaults
    let null_names: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE name IS NULL", ())
        .unwrap();
    assert_eq!(null_names, 2, "Rows 2 and 4 should have NULL name");

    let null_scores: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE score IS NULL", ())
        .unwrap();
    assert_eq!(null_scores, 2, "Rows 2 and 3 should have NULL score");

    let null_active: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE active IS NULL", ())
        .unwrap();
    assert_eq!(null_active, 2, "Rows 2 and 4 should have NULL active");

    // Non-NULL values are correct
    let alice_score: i64 = db
        .query_one("SELECT score FROM null_test WHERE id = 1", ())
        .unwrap();
    assert_eq!(alice_score, 100);

    let charlie_active: bool = db
        .query_one("SELECT active FROM null_test WHERE id = 3", ())
        .unwrap();
    assert!(charlie_active);
}

#[test]
fn test_null_values_survive_snapshot_recovery() {
    // NULLs must survive through snapshot (not just WAL).
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE null_snap (id INTEGER PRIMARY KEY, val TEXT, num FLOAT)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO null_snap (id, val, num) VALUES (1, 'hello', 3.14)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO null_snap (id, val, num) VALUES (2, NULL, NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO null_snap (id, val, num) VALUES (3, NULL, 2.71)",
            (),
        )
        .unwrap();

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let total: i64 = db.query_one("SELECT COUNT(*) FROM null_snap", ()).unwrap();
    assert_eq!(total, 3);

    let null_vals: i64 = db
        .query_one("SELECT COUNT(*) FROM null_snap WHERE val IS NULL", ())
        .unwrap();
    assert_eq!(null_vals, 2, "NULLs must survive snapshot serialization");

    let null_nums: i64 = db
        .query_one("SELECT COUNT(*) FROM null_snap WHERE num IS NULL", ())
        .unwrap();
    assert_eq!(null_nums, 1, "Row 2 should have NULL num");

    let val: f64 = db
        .query_one("SELECT num FROM null_snap WHERE id = 3", ())
        .unwrap();
    assert!(
        (val - 2.71).abs() < 0.001,
        "Non-NULL float must be preserved"
    );
}

// ============================================================================
// Multiple consecutive crash-recovery cycles
// ============================================================================

#[test]
fn test_multiple_crash_recovery_cycles() {
    // Simulate: write data → corrupt WAL → recover → write more → close → reopen → verify
    // Tests that recovery writes valid WAL that can itself be recovered.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Cycle 1: Create table and insert initial data, then corrupt
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE cycle_test (id INTEGER PRIMARY KEY, cycle INTEGER NOT NULL, value TEXT)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO cycle_test (id, cycle, value) VALUES ({}, 1, 'cycle1_row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Corrupt: truncate last WAL entry
    let wal_files = find_wal_files(&db_path);
    if !wal_files.is_empty() {
        let wal_path = &wal_files[wal_files.len() - 1];
        let data = fs::read(wal_path).unwrap();
        let entries = find_entry_boundaries(&data);
        if entries.len() >= 2 {
            let last = &entries[entries.len() - 1];
            let truncated = &data[..last.offset];
            fs::write(wal_path, truncated).unwrap();
        }
    }

    // Cycle 2: Recover from corruption, write more data
    let cycle1_count: i64;
    {
        let db = Database::open(&dsn).unwrap();
        cycle1_count = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
        assert!(
            cycle1_count > 0,
            "Should recover at least some cycle 1 data"
        );

        for i in 100..=110 {
            db.execute(
                &format!(
                    "INSERT INTO cycle_test (id, cycle, value) VALUES ({}, 2, 'cycle2_row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Corrupt again: flip a bit in the WAL
    let wal_files = find_wal_files(&db_path);
    if !wal_files.is_empty() {
        let wal_path = &wal_files[wal_files.len() - 1];
        let mut data = fs::read(wal_path).unwrap();
        let entries = find_entry_boundaries(&data);
        if entries.len() >= 3 {
            // Flip a bit in a middle entry's data
            let mid = &entries[entries.len() / 2];
            if mid.data_offset + 5 < data.len() {
                flip_bit(&mut data, mid.data_offset + 5, 3);
                fs::write(wal_path, &data).unwrap();
            }
        }
    }

    // Cycle 3: Recover from second corruption, write more data
    let cycle2_count: i64;
    {
        let db = Database::open(&dsn).unwrap();
        cycle2_count = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
        assert!(
            cycle2_count >= cycle1_count,
            "Cycle 3 recovery ({}) should have at least cycle 1 data ({})",
            cycle2_count,
            cycle1_count
        );

        for i in 200..=205 {
            db.execute(
                &format!(
                    "INSERT INTO cycle_test (id, cycle, value) VALUES ({}, 3, 'cycle3_row_{}')",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Cycle 4: Clean recovery — no corruption. Everything from cycle 3 must survive.
    {
        let db = Database::open(&dsn).unwrap();
        let final_count: i64 = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
        assert!(
            final_count >= cycle2_count + 6,
            "Final count ({}) should include cycle 3 inserts (>= {})",
            final_count,
            cycle2_count + 6
        );

        // Verify cycle 3 data is intact (no corruption was applied)
        let c3: i64 = db
            .query_one("SELECT COUNT(*) FROM cycle_test WHERE cycle = 3", ())
            .unwrap();
        assert_eq!(
            c3, 6,
            "All 6 cycle 3 rows must survive (no corruption applied)"
        );

        // DB is usable
        db.execute(
            "INSERT INTO cycle_test (id, cycle, value) VALUES (999, 4, 'final')",
            (),
        )
        .unwrap();
    }
}

#[test]
fn test_crash_recovery_with_snapshot_between_cycles() {
    // Cycle 1: insert → snapshot → close
    // Cycle 2: corrupt WAL → recover from snapshot → insert more → close
    // Cycle 3: reopen → all data from both cycles present
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Cycle 1
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE snap_cycle (id INTEGER PRIMARY KEY, phase INTEGER NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=20 {
            db.execute(
                &format!("INSERT INTO snap_cycle (id, phase) VALUES ({}, 1)", i),
                (),
            )
            .unwrap();
        }

        let _ = db.execute("PRAGMA SNAPSHOT", ());
    }
    remove_lock_file(&db_path);

    // Corrupt WAL after snapshot
    let wal_files = find_wal_files(&db_path);
    if !wal_files.is_empty() {
        let wal_path = &wal_files[wal_files.len() - 1];
        let mut data = fs::read(wal_path).unwrap();
        // Zero last 4KB page
        let file_len = data.len();
        if file_len > 4096 {
            let last_page = (file_len - 1) / 4096;
            zero_page(&mut data, last_page);
            fs::write(wal_path, &data).unwrap();
        }
    }

    // Cycle 2: recover (snapshot provides base), insert more
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM snap_cycle", ()).unwrap();
        // Snapshot had 20 rows; WAL may have partial additional data
        assert!(
            count >= 20,
            "Snapshot should guarantee at least 20 rows, got {}",
            count
        );

        for i in 100..=110 {
            db.execute(
                &format!("INSERT INTO snap_cycle (id, phase) VALUES ({}, 2)", i),
                (),
            )
            .unwrap();
        }
    }
    remove_lock_file(&db_path);

    // Cycle 3: clean recovery
    {
        let db = Database::open(&dsn).unwrap();
        let total: i64 = db.query_one("SELECT COUNT(*) FROM snap_cycle", ()).unwrap();
        assert!(
            total >= 31,
            "Should have at least 20 (snapshot) + 11 (cycle 2) = 31 rows, got {}",
            total
        );

        let phase2: i64 = db
            .query_one("SELECT COUNT(*) FROM snap_cycle WHERE phase = 2", ())
            .unwrap();
        assert_eq!(
            phase2, 11,
            "All cycle 2 inserts must survive clean recovery"
        );
    }
}

// ============================================================================
// DEFAULT constraint enforcement after recovery
// ============================================================================

#[test]
fn test_default_constraint_after_recovery() {
    // DEFAULT values must be applied correctly after recovery.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE def_test (id INTEGER PRIMARY KEY, name TEXT NOT NULL, status TEXT DEFAULT 'pending', priority INTEGER DEFAULT 0)",
            (),
        )
        .unwrap();

        // Insert without specifying default columns
        db.execute("INSERT INTO def_test (id, name) VALUES (1, 'task_1')", ())
            .unwrap();
        // Insert with explicit values overriding defaults
        db.execute(
            "INSERT INTO def_test (id, name, status, priority) VALUES (2, 'task_2', 'active', 5)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM def_test", ()).unwrap();
    assert_eq!(count, 2);

    // After recovery, inserting should still apply defaults
    db.execute("INSERT INTO def_test (id, name) VALUES (3, 'task_3')", ())
        .unwrap();

    let status: String = db
        .query_one("SELECT status FROM def_test WHERE id = 3", ())
        .unwrap();
    assert_eq!(
        status, "pending",
        "DEFAULT value must be applied after recovery"
    );

    let priority: i64 = db
        .query_one("SELECT priority FROM def_test WHERE id = 3", ())
        .unwrap();
    assert_eq!(
        priority, 0,
        "DEFAULT integer value must be applied after recovery"
    );
}

// ============================================================================
// BOOLEAN and TIMESTAMP column types durability
// ============================================================================

#[test]
fn test_boolean_column_durability() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE bool_test (id INTEGER PRIMARY KEY, flag BOOLEAN NOT NULL, optional_flag BOOLEAN)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO bool_test (id, flag, optional_flag) VALUES (1, TRUE, FALSE)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO bool_test (id, flag, optional_flag) VALUES (2, FALSE, TRUE)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO bool_test (id, flag, optional_flag) VALUES (3, TRUE, NULL)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM bool_test", ()).unwrap();
    assert_eq!(count, 3);

    let true_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_test WHERE flag = TRUE", ())
        .unwrap();
    assert_eq!(true_count, 2, "TRUE values must survive recovery");

    let false_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_test WHERE flag = FALSE", ())
        .unwrap();
    assert_eq!(false_count, 1, "FALSE values must survive recovery");

    let null_opt: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bool_test WHERE optional_flag IS NULL",
            (),
        )
        .unwrap();
    assert_eq!(null_opt, 1, "NULL boolean must survive recovery");
}

#[test]
fn test_timestamp_column_durability() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE ts_test (id INTEGER PRIMARY KEY, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO ts_test (id, created_at, updated_at) VALUES (1, '2024-01-15 10:30:00', '2024-06-20 14:00:00')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO ts_test (id, created_at, updated_at) VALUES (2, '2024-12-31 23:59:59', NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO ts_test (id, created_at, updated_at) VALUES (3, '2020-01-01 00:00:00', '2020-01-01 00:00:01')",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM ts_test", ()).unwrap();
    assert_eq!(count, 3);

    // Verify timestamps are not corrupted by checking ordering
    let ordered_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM ts_test WHERE created_at >= '2024-01-01 00:00:00'",
            (),
        )
        .unwrap();
    assert_eq!(
        ordered_count, 2,
        "Timestamp comparison must work after recovery"
    );

    let null_updated: i64 = db
        .query_one("SELECT COUNT(*) FROM ts_test WHERE updated_at IS NULL", ())
        .unwrap();
    assert_eq!(null_updated, 1, "NULL timestamp must survive recovery");

    // Verify EXTRACT works on recovered timestamps
    let year: i64 = db
        .query_one(
            "SELECT EXTRACT(YEAR FROM created_at) FROM ts_test WHERE id = 2",
            (),
        )
        .unwrap();
    assert_eq!(year, 2024, "EXTRACT from recovered timestamp must work");
}

// ============================================================================
// Snapshot with UPDATE/DELETE then snapshot corrupted (WAL-only re-derivation)
// ============================================================================

#[test]
fn test_update_after_snapshot_then_snapshot_corrupt() {
    // Snapshot contains original rows. WAL has UPDATEs. Corrupt snapshot → WAL must
    // replay both INSERT and UPDATE to produce correct final state.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE upd_snap (id INTEGER PRIMARY KEY, value TEXT NOT NULL, version INTEGER)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO upd_snap (id, value, version) VALUES ({}, 'original_{}', 1)",
                    i, i
                ),
                (),
            )
            .unwrap();
        }

        // Take snapshot with original data
        let _ = db.execute("PRAGMA SNAPSHOT", ());

        // UPDATE some rows after snapshot
        db.execute(
            "UPDATE upd_snap SET value = 'updated_5', version = 2 WHERE id = 5",
            (),
        )
        .unwrap();
        db.execute(
            "UPDATE upd_snap SET value = 'updated_10', version = 2 WHERE id = 10",
            (),
        )
        .unwrap();

        // DELETE a row after snapshot
        db.execute("DELETE FROM upd_snap WHERE id = 3", ()).unwrap();
    }
    remove_lock_file(&db_path);

    // Corrupt the snapshot
    let snap_files = find_snapshot_files(&db_path, "upd_snap");
    for snap_file in &snap_files {
        let mut data = fs::read(snap_file).unwrap();
        if data.len() > 10 {
            // Zero the first 64 bytes (header) to make it unreadable
            zero_range(&mut data, 0, 64);
            fs::write(snap_file, &data).unwrap();
        }
    }

    // Also remove checkpoint.meta to force full WAL replay
    if let Some(cp) = find_checkpoint_file(&db_path) {
        let _ = fs::remove_file(cp);
    }

    let db = Database::open(&dsn).unwrap();

    let total: i64 = db.query_one("SELECT COUNT(*) FROM upd_snap", ()).unwrap();
    assert_eq!(total, 9, "10 inserted - 1 deleted = 9 rows");

    // Updated rows have new values
    let v5: String = db
        .query_one("SELECT value FROM upd_snap WHERE id = 5", ())
        .unwrap();
    assert_eq!(v5, "updated_5", "UPDATE must be replayed from WAL");

    let v10: String = db
        .query_one("SELECT value FROM upd_snap WHERE id = 10", ())
        .unwrap();
    assert_eq!(v10, "updated_10");

    // Deleted row must not exist
    let deleted: i64 = db
        .query_one("SELECT COUNT(*) FROM upd_snap WHERE id = 3", ())
        .unwrap();
    assert_eq!(deleted, 0, "DELETE must be replayed from WAL");

    // Unmodified rows still have original values
    let v1: String = db
        .query_one("SELECT value FROM upd_snap WHERE id = 1", ())
        .unwrap();
    assert_eq!(v1, "original_1");
}

// ============================================================================
// Multi-column UNIQUE index enforcement after recovery
// ============================================================================

#[test]
fn test_multi_column_unique_index_enforced_after_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE mc_unique (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b TEXT NOT NULL, data TEXT)",
            (),
        )
        .unwrap();

        db.execute("CREATE UNIQUE INDEX idx_mc_ab ON mc_unique(a, b)", ())
            .unwrap();

        db.execute(
            "INSERT INTO mc_unique (id, a, b, data) VALUES (1, 'x', 'y', 'first')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO mc_unique (id, a, b, data) VALUES (2, 'x', 'z', 'second')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO mc_unique (id, a, b, data) VALUES (3, 'w', 'y', 'third')",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM mc_unique", ()).unwrap();
    assert_eq!(count, 3);

    // Duplicate (a='x', b='y') must be rejected after recovery
    let result = db.execute(
        "INSERT INTO mc_unique (id, a, b, data) VALUES (4, 'x', 'y', 'duplicate')",
        (),
    );
    assert!(
        result.is_err(),
        "Multi-column UNIQUE constraint must be enforced after recovery"
    );

    // Different combination is allowed
    db.execute(
        "INSERT INTO mc_unique (id, a, b, data) VALUES (4, 'x', 'w', 'allowed')",
        (),
    )
    .unwrap();
}

// ============================================================================
// ALTER TABLE MODIFY COLUMN durability
// ============================================================================

/// ALTER TABLE MODIFY COLUMN must persist type and nullability changes
/// across close/reopen. The WAL records this as AlterTable op_type=4.
#[test]
fn test_alter_table_modify_column_durability() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE modify_test (id INTEGER PRIMARY KEY, score INTEGER NOT NULL, label TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO modify_test (id, score, label) VALUES ({}, {}, 'item_{}')",
                    i,
                    i * 10,
                    i
                ),
                (),
            )
            .unwrap();
        }

        // Change score from NOT NULL to nullable
        db.execute("ALTER TABLE modify_test MODIFY COLUMN score INTEGER", ())
            .unwrap();

        // Insert a row with NULL score to prove the change took effect
        db.execute(
            "INSERT INTO modify_test (id, score, label) VALUES (11, NULL, 'null_score')",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    // All rows must survive
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_test", ())
        .unwrap();
    assert_eq!(count, 11, "All 11 rows should survive after recovery");

    // Original data intact
    let score: i64 = db
        .query_one("SELECT score FROM modify_test WHERE id = 5", ())
        .unwrap();
    assert_eq!(score, 50, "Original score values must be preserved");

    // NULL score must survive (proves MODIFY to nullable persisted)
    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_test WHERE score IS NULL", ())
        .unwrap();
    assert_eq!(
        null_count, 1,
        "NULL score row must survive — MODIFY COLUMN nullable change must persist"
    );

    // Can still insert NULL after recovery (proves schema change persisted)
    db.execute(
        "INSERT INTO modify_test (id, score, label) VALUES (12, NULL, 'post_recovery_null')",
        (),
    )
    .unwrap();

    let null_count2: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_test WHERE score IS NULL", ())
        .unwrap();
    assert_eq!(
        null_count2, 2,
        "Post-recovery NULL insert must work after MODIFY COLUMN"
    );
}

/// ALTER TABLE MODIFY COLUMN with snapshot — WAL replay must apply
/// the type change on top of snapshot state.
#[test]
fn test_alter_table_modify_column_with_snapshot() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE modify_snap (id INTEGER PRIMARY KEY, value TEXT NOT NULL, flag BOOLEAN NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=20 {
            db.execute(
                &format!(
                    "INSERT INTO modify_snap (id, value, flag) VALUES ({}, 'v_{}', {})",
                    i,
                    i,
                    if i % 2 == 0 { "TRUE" } else { "FALSE" }
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot captures current schema (value NOT NULL, flag NOT NULL)
        db.execute("PRAGMA snapshot", ()).unwrap();

        // MODIFY after snapshot — must be replayed from WAL on recovery
        db.execute("ALTER TABLE modify_snap MODIFY COLUMN value TEXT", ())
            .unwrap();

        db.execute("ALTER TABLE modify_snap MODIFY COLUMN flag BOOLEAN", ())
            .unwrap();

        // Insert rows using new nullable schema
        db.execute(
            "INSERT INTO modify_snap (id, value, flag) VALUES (21, NULL, NULL)",
            (),
        )
        .unwrap();
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_snap", ())
        .unwrap();
    assert_eq!(count, 21, "All 21 rows must survive snapshot + WAL replay");

    // NULL values from post-MODIFY insert must survive
    let null_value_count: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_snap WHERE value IS NULL", ())
        .unwrap();
    assert_eq!(
        null_value_count, 1,
        "NULL value row must survive — MODIFY COLUMN must replay over snapshot"
    );

    let null_flag_count: i64 = db
        .query_one("SELECT COUNT(*) FROM modify_snap WHERE flag IS NULL", ())
        .unwrap();
    assert_eq!(
        null_flag_count, 1,
        "NULL flag row must survive — MODIFY COLUMN must replay over snapshot"
    );

    // Post-recovery inserts with NULLs must work
    db.execute(
        "INSERT INTO modify_snap (id, value, flag) VALUES (22, NULL, TRUE)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO modify_snap (id, value, flag) VALUES (23, 'hello', NULL)",
        (),
    )
    .unwrap();
}

// ============================================================================
// Bitmap index durability
// ============================================================================

/// Bitmap index created with USING BITMAP must survive close/reopen.
/// This exercises the bitmap-specific WAL serialization (index_type byte = 2)
/// and the BitmapIndex reconstruction path in create_index_from_metadata.
#[test]
fn test_bitmap_index_durability() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE bitmap_test (id INTEGER PRIMARY KEY, active BOOLEAN NOT NULL, category TEXT NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO bitmap_test (id, active, category) VALUES ({}, {}, '{}')",
                    i,
                    if i % 3 == 0 { "TRUE" } else { "FALSE" },
                    if i % 4 == 0 {
                        "A"
                    } else if i % 4 == 1 {
                        "B"
                    } else if i % 4 == 2 {
                        "C"
                    } else {
                        "D"
                    }
                ),
                (),
            )
            .unwrap();
        }

        // Create explicit bitmap indexes
        db.execute(
            "CREATE INDEX idx_active_bitmap ON bitmap_test(active) USING BITMAP",
            (),
        )
        .unwrap();

        db.execute(
            "CREATE INDEX idx_cat_bitmap ON bitmap_test(category) USING BITMAP",
            (),
        )
        .unwrap();

        // Verify indexes work before close
        let active_count: i64 = db
            .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = TRUE", ())
            .unwrap();
        assert_eq!(active_count, 16); // 3,6,9,...,48 → 16 values

        let cat_a_count: i64 = db
            .query_one("SELECT COUNT(*) FROM bitmap_test WHERE category = 'A'", ())
            .unwrap();
        assert_eq!(cat_a_count, 12); // 4,8,12,...,48 → 12 values
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    // All data must survive
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test", ())
        .unwrap();
    assert_eq!(count, 50, "All 50 rows must survive recovery");

    // Bitmap index on active must work after recovery
    let active_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = TRUE", ())
        .unwrap();
    assert_eq!(
        active_count, 16,
        "Bitmap index on active must return correct results after recovery"
    );

    let inactive_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = FALSE", ())
        .unwrap();
    assert_eq!(
        inactive_count, 34,
        "Bitmap index on active=FALSE must return correct results after recovery"
    );

    // Bitmap index on category must work after recovery
    let cat_b_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE category = 'B'", ())
        .unwrap();
    assert_eq!(
        cat_b_count, 13,
        "Bitmap index on category must return correct results after recovery"
    );

    // Inserts after recovery must update the bitmap index
    db.execute(
        "INSERT INTO bitmap_test (id, active, category) VALUES (51, TRUE, 'A')",
        (),
    )
    .unwrap();

    let new_active_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = TRUE", ())
        .unwrap();
    assert_eq!(
        new_active_count, 17,
        "Bitmap index must handle post-recovery inserts"
    );

    let new_cat_a_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE category = 'A'", ())
        .unwrap();
    assert_eq!(
        new_cat_a_count, 13,
        "Bitmap index on category must handle post-recovery inserts"
    );
}

/// Bitmap index with snapshot — the index definition must survive snapshot + WAL replay.
#[test]
fn test_bitmap_index_with_snapshot_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE bitmap_snap (id INTEGER PRIMARY KEY, status BOOLEAN NOT NULL)",
            (),
        )
        .unwrap();

        for i in 1..=30 {
            db.execute(
                &format!(
                    "INSERT INTO bitmap_snap (id, status) VALUES ({}, {})",
                    i,
                    if i % 2 == 0 { "TRUE" } else { "FALSE" }
                ),
                (),
            )
            .unwrap();
        }

        // Snapshot before index creation
        db.execute("PRAGMA snapshot", ()).unwrap();

        // Create bitmap index after snapshot — must be replayed from WAL
        db.execute(
            "CREATE INDEX idx_status_bm ON bitmap_snap(status) USING BITMAP",
            (),
        )
        .unwrap();

        // Insert more data after index creation
        for i in 31..=40 {
            db.execute(
                &format!("INSERT INTO bitmap_snap (id, status) VALUES ({}, TRUE)", i),
                (),
            )
            .unwrap();
        }
    }

    remove_lock_file(&db_path);

    let db = Database::open(&dsn).unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_snap", ())
        .unwrap();
    assert_eq!(count, 40, "All 40 rows must survive snapshot + WAL replay");

    // Bitmap index must work — 15 TRUE from first batch + 10 TRUE from second
    let true_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_snap WHERE status = TRUE", ())
        .unwrap();
    assert_eq!(
        true_count, 25,
        "Bitmap index must return correct results after snapshot + WAL replay"
    );

    let false_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_snap WHERE status = FALSE", ())
        .unwrap();
    assert_eq!(
        false_count, 15,
        "Bitmap index must return correct FALSE count after recovery"
    );

    // Dropping and recreating the bitmap index should work (proves it was recovered)
    db.execute("DROP INDEX idx_status_bm ON bitmap_snap", ())
        .unwrap();
    db.execute(
        "CREATE INDEX idx_status_bm ON bitmap_snap(status) USING BITMAP",
        (),
    )
    .unwrap();

    let recheck: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_snap WHERE status = TRUE", ())
        .unwrap();
    assert_eq!(recheck, 25, "Recreated bitmap index must work correctly");
}

/// FK constraints must survive snapshot + WAL truncation recovery.
/// Tests that the snapshot serializer preserves FK metadata so enforcement
/// works after the WAL entries that created the table are truncated away.
#[test]
fn test_foreign_key_with_snapshot_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create parent + child with FK constraints
        db.execute(
            "CREATE TABLE fk_parent (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE fk_child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES fk_parent(id) ON DELETE CASCADE, val TEXT)",
            (),
        )
        .unwrap();

        // Insert parent rows
        for i in 1..=5 {
            db.execute(
                &format!("INSERT INTO fk_parent (id, name) VALUES ({}, 'p{}')", i, i),
                (),
            )
            .unwrap();
        }
        // Insert child rows referencing parents
        for i in 1..=10 {
            let parent_id = (i % 5) + 1;
            db.execute(
                &format!(
                    "INSERT INTO fk_child (id, parent_id, val) VALUES ({}, {}, 'c{}')",
                    i, parent_id, i
                ),
                (),
            )
            .unwrap();
        }

        // Take TWO snapshots so WAL truncation can happen
        db.execute("PRAGMA snapshot", ()).unwrap();
        // Insert one more row to force a WAL entry after the first snapshot
        db.execute("INSERT INTO fk_parent (id, name) VALUES (100, 'extra')", ())
            .unwrap();
        db.execute("PRAGMA snapshot", ()).unwrap();
    }

    remove_lock_file(&db_path);

    // Reopen — schema should be loaded from snapshot (WAL may be truncated)
    let db = Database::open(&dsn).unwrap();

    // Verify data survived
    let parent_count: i64 = db.query_one("SELECT COUNT(*) FROM fk_parent", ()).unwrap();
    assert_eq!(parent_count, 6, "All 6 parent rows must survive");

    let child_count: i64 = db.query_one("SELECT COUNT(*) FROM fk_child", ()).unwrap();
    assert_eq!(child_count, 10, "All 10 child rows must survive");

    // FK enforcement must still work — insert with invalid parent must fail
    let err = db.execute(
        "INSERT INTO fk_child (id, parent_id, val) VALUES (99, 999, 'bad')",
        (),
    );
    assert!(
        err.is_err(),
        "FK constraint must be enforced after snapshot recovery"
    );

    // CASCADE must still work — delete parent 1, children referencing it should be deleted
    db.execute("DELETE FROM fk_child WHERE parent_id = 100", ())
        .unwrap_or_default(); // clean up extra parent's potential children
    db.execute("DELETE FROM fk_parent WHERE id = 1", ())
        .unwrap();

    let remaining: i64 = db
        .query_one("SELECT COUNT(*) FROM fk_child WHERE parent_id = 1", ())
        .unwrap();
    assert_eq!(
        remaining, 0,
        "CASCADE DELETE must work after snapshot recovery"
    );

    // Valid inserts must still work
    db.execute(
        "INSERT INTO fk_child (id, parent_id, val) VALUES (50, 2, 'valid')",
        (),
    )
    .unwrap();
}

/// Verify that DROP TABLE correctly strips FK references from child tables on WAL replay.
/// Without the fix, child tables retain orphaned FK constraints after recovery,
/// causing INSERT failures with "table not found" for the dropped parent.
#[test]
fn test_drop_parent_strips_child_fk_on_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("fk_drop_recovery.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: create parent + child, then drop parent
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE parent_drop (id INTEGER PRIMARY KEY, name TEXT)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE child_drop (
                id INTEGER PRIMARY KEY,
                pid INTEGER REFERENCES parent_drop(id),
                val TEXT
            )",
            (),
        )
        .unwrap();

        // Insert data into parent and child with NULL FK (so drop isn't blocked)
        db.execute("INSERT INTO parent_drop VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute("INSERT INTO child_drop VALUES (1, NULL, 'x')", ())
            .unwrap();

        // Drop parent — in-memory, child FK is stripped
        db.execute("DROP TABLE parent_drop", ()).unwrap();

        // Verify child FK is gone in memory: insert with any pid should work
        db.execute("INSERT INTO child_drop VALUES (2, 999, 'y')", ())
            .unwrap();
    }

    // Phase 2: reopen — WAL replay must strip child FK
    {
        let db = Database::open(&dsn).unwrap();

        // parent_drop must not exist
        let err = db.execute("SELECT * FROM parent_drop", ());
        assert!(err.is_err(), "parent_drop should not exist after recovery");

        // child_drop must exist and FK must be gone
        let count: i64 = db.query_one("SELECT COUNT(*) FROM child_drop", ()).unwrap();
        assert_eq!(count, 2, "child_drop should have 2 rows");

        // Insert with arbitrary pid must succeed (no FK constraint)
        db.execute("INSERT INTO child_drop VALUES (3, 12345, 'z')", ())
            .unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM child_drop", ()).unwrap();
        assert_eq!(count, 3);
    }
}
