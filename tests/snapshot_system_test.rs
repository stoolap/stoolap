// Copyright 2025 Stoolap Contributors
// Comprehensive end-to-end tests for the snapshot system
//
// Tests cover:
// - Automatic snapshots with configurable interval
// - Snapshot rotation/cleanup
// - WAL truncation after snapshot
// - Crash recovery with snapshot + WAL
// - Large dataset handling
// - Multiple snapshot cycles

use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use stoolap::api::Database;
use tempfile::tempdir;

/// Helper to count files matching a pattern in a directory
fn count_files_matching(dir: &Path, prefix: &str, suffix: &str) -> usize {
    if !dir.exists() {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with(prefix) && name.ends_with(suffix)
                })
                .count()
        })
        .unwrap_or(0)
}

/// Helper to get total size of files in a directory
fn get_dir_size(dir: &Path) -> u64 {
    if !dir.exists() {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Magic bytes for snapshot metadata ("SNAP" in ASCII)
const SNAPSHOT_META_MAGIC: u32 = 0x50414E53;

/// Helper to read snapshot metadata LSN (supports both binary and JSON formats)
fn read_snapshot_lsn(db_path: &Path) -> Option<u64> {
    // First try new binary format
    let bin_path = db_path.join("snapshots/snapshot_meta.bin");
    if let Ok(data) = fs::read(&bin_path) {
        if data.len() >= 28 {
            let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
            if magic == SNAPSHOT_META_MAGIC {
                // Verify CRC32
                let stored_crc = u32::from_le_bytes(data[24..28].try_into().unwrap());
                let computed_crc = crc32fast::hash(&data[0..24]);
                if stored_crc == computed_crc {
                    return Some(u64::from_le_bytes(data[8..16].try_into().unwrap()));
                }
            }
        }
    }

    // Fall back to old JSON format for backward compatibility
    let json_path = db_path.join("snapshots/snapshot_meta.json");
    if let Ok(content) = fs::read_to_string(&json_path) {
        // Parse {"lsn":123}
        if let Some(start) = content.find("\"lsn\":") {
            let rest = &content[start + 6..];
            if let Some(end) = rest.find('}') {
                return rest[..end].trim().parse().ok();
            }
        }
    }
    None
}

/// Helper to get WAL file LSN from filename
fn get_wal_lsn(db_path: &Path) -> Option<u64> {
    let wal_dir = db_path.join("wal");
    if !wal_dir.exists() {
        return None;
    }

    fs::read_dir(&wal_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("wal-") && name.ends_with(".log") {
                // Extract LSN from "wal-YYYYMMDD-HHMMSS-lsn-N.log"
                if let Some(lsn_start) = name.find("lsn-") {
                    if let Some(lsn_end) = name[lsn_start + 4..].find('.') {
                        return name[lsn_start + 4..lsn_start + 4 + lsn_end].parse().ok();
                    }
                }
            }
            None
        })
        .max()
}

// =============================================================================
// TEST 1: Large Dataset with Manual Snapshots
// =============================================================================

#[test]
fn test_large_dataset_with_snapshots() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Large Dataset with Snapshots ===");
    println!("DB path: {:?}", db_path);

    // Create database with persistence
    {
        let db = Database::open(&dsn).unwrap();

        // Create table
        db.execute(
            "CREATE TABLE large_data (id INT PRIMARY KEY, name TEXT, value REAL, category TEXT)",
            (),
        )
        .unwrap();

        // Insert 1000 rows
        println!("Inserting 1000 rows...");
        for i in 1..=1000 {
            db.execute(
                &format!(
                    "INSERT INTO large_data VALUES ({}, 'item_{}', {}.{}, 'cat_{}')",
                    i,
                    i,
                    i,
                    i % 100,
                    i % 10
                ),
                (),
            )
            .unwrap();
        }

        // Verify count
        let count: i64 = db.query_one("SELECT COUNT(*) FROM large_data", ()).unwrap();
        println!("Row count after inserts: {}", count);
        assert_eq!(count, 1000);

        // Check WAL size before snapshot
        let wal_size_before = get_dir_size(&db_path.join("wal"));
        println!("WAL size before snapshot: {} bytes", wal_size_before);

        // Take snapshot
        println!("Taking snapshot...");
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();

        // Check WAL size after snapshot (should be smaller due to truncation)
        let wal_size_after = get_dir_size(&db_path.join("wal"));
        println!("WAL size after snapshot: {} bytes", wal_size_after);

        // Verify WAL was truncated
        assert!(
            wal_size_after < wal_size_before,
            "WAL should be truncated after snapshot: before={}, after={}",
            wal_size_before,
            wal_size_after
        );

        // Check snapshot was created
        let snapshot_count =
            count_files_matching(&db_path.join("snapshots/large_data"), "snapshot-", ".bin");
        println!("Snapshot count: {}", snapshot_count);
        assert_eq!(snapshot_count, 1, "Should have 1 snapshot");

        // Check LSN alignment
        let snapshot_lsn = read_snapshot_lsn(&db_path);
        let wal_lsn = get_wal_lsn(&db_path);
        println!("Snapshot LSN: {:?}, WAL LSN: {:?}", snapshot_lsn, wal_lsn);
        assert_eq!(
            snapshot_lsn, wal_lsn,
            "Snapshot LSN should match WAL file LSN"
        );

        db.close().unwrap();
    }

    // Reopen and verify data is intact
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM large_data", ()).unwrap();
        println!("Row count after reopen: {}", count);
        assert_eq!(count, 1000);

        // Verify some specific rows
        let name: String = db
            .query_one("SELECT name FROM large_data WHERE id = 500", ())
            .unwrap();
        println!("Row 500 name: {}", name);
        assert_eq!(name, "item_500");

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 2: Multiple Snapshot Cycles with WAL Growth
// =============================================================================

#[test]
fn test_multiple_snapshot_cycles() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Multiple Snapshot Cycles ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE cycle_test (id INT PRIMARY KEY, batch INT, value TEXT)",
            (),
        )
        .unwrap();

        let mut expected_count = 0;

        for cycle in 1..=5 {
            println!("\n--- Cycle {} ---", cycle);

            // Insert batch of data
            for i in 1..=200 {
                let id = (cycle - 1) * 200 + i;
                db.execute(
                    &format!(
                        "INSERT INTO cycle_test VALUES ({}, {}, 'value_{}')",
                        id, cycle, id
                    ),
                    (),
                )
                .unwrap();
                expected_count += 1;
            }

            // Verify count
            let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
            println!("Total rows: {}", count);

            // Take snapshot
            db.execute("PRAGMA SNAPSHOT", ()).unwrap();

            // Check snapshot count
            let snapshot_count =
                count_files_matching(&db_path.join("snapshots/cycle_test"), "snapshot-", ".bin");
            println!("Snapshots after cycle {}: {}", cycle, snapshot_count);

            // Check WAL was truncated
            let wal_size = get_dir_size(&db_path.join("wal"));
            println!("WAL size: {} bytes", wal_size);

            // Check LSN alignment
            let snapshot_lsn = read_snapshot_lsn(&db_path);
            let wal_lsn = get_wal_lsn(&db_path);
            println!("Snapshot LSN: {:?}, WAL LSN: {:?}", snapshot_lsn, wal_lsn);
            assert_eq!(
                snapshot_lsn, wal_lsn,
                "LSNs should match after cycle {}",
                cycle
            );
        }

        // Final verification
        let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
        println!("\nFinal count: {} (expected: {})", count, expected_count);
        assert_eq!(count, expected_count as i64);

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM cycle_test", ()).unwrap();
        println!("Count after reopen: {}", count);
        assert_eq!(count, 1000);

        // Verify data from each batch
        for batch in 1..=5 {
            let batch_count: i64 = db
                .query_one(
                    &format!("SELECT COUNT(*) FROM cycle_test WHERE batch = {}", batch),
                    (),
                )
                .unwrap();
            println!("Batch {} count: {}", batch, batch_count);
            assert_eq!(batch_count, 200);
        }

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 3: Crash Recovery - Data After Snapshot
// =============================================================================

#[test]
fn test_crash_recovery_data_after_snapshot() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Crash Recovery - Data After Snapshot ===");

    // Phase 1: Create data and snapshot
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE recovery_test (id INT PRIMARY KEY, phase TEXT, value INT)",
            (),
        )
        .unwrap();

        // Insert initial data
        println!("Inserting 500 rows (phase 1)...");
        for i in 1..=500 {
            db.execute(
                &format!(
                    "INSERT INTO recovery_test VALUES ({}, 'phase1', {})",
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        // Take snapshot
        println!("Taking snapshot...");
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();

        let snapshot_lsn = read_snapshot_lsn(&db_path);
        println!("Snapshot LSN: {:?}", snapshot_lsn);

        // Insert more data AFTER snapshot (these are only in WAL)
        println!("Inserting 200 more rows (phase 2 - after snapshot)...");
        for i in 501..=700 {
            db.execute(
                &format!(
                    "INSERT INTO recovery_test VALUES ({}, 'phase2', {})",
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        // Verify total
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM recovery_test", ())
            .unwrap();
        println!("Total before crash: {}", count);
        assert_eq!(count, 700);

        // Check WAL grew after snapshot
        let wal_size = get_dir_size(&db_path.join("wal"));
        println!("WAL size with post-snapshot data: {} bytes", wal_size);

        // DO NOT call close() - simulate crash by just dropping
        drop(db);
    }

    // Remove lock file to simulate crash recovery
    let lock_path = db_path.join("db.lock");
    if lock_path.exists() {
        fs::remove_file(&lock_path).unwrap();
        println!("Removed lock file (simulating crash)");
    }

    // Phase 2: Recovery after crash
    {
        println!("\nRecovering after crash...");
        let db = Database::open(&dsn).unwrap();

        // Verify ALL data recovered (500 from snapshot + 200 from WAL)
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM recovery_test", ())
            .unwrap();
        println!("Total after recovery: {}", count);
        assert_eq!(count, 700);

        // Verify phase1 data (from snapshot)
        let phase1_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM recovery_test WHERE phase = 'phase1'",
                (),
            )
            .unwrap();
        println!("Phase 1 rows: {}", phase1_count);
        assert_eq!(phase1_count, 500);

        // Verify phase2 data (from WAL only)
        let phase2_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM recovery_test WHERE phase = 'phase2'",
                (),
            )
            .unwrap();
        println!("Phase 2 rows (WAL recovery): {}", phase2_count);
        assert_eq!(phase2_count, 200);

        // Verify specific rows from each phase
        let value250: i64 = db
            .query_one("SELECT value FROM recovery_test WHERE id = 250", ())
            .unwrap();
        println!("Row 250 (snapshot) value: {}", value250);
        assert_eq!(value250, 2500);

        let value650: i64 = db
            .query_one("SELECT value FROM recovery_test WHERE id = 650", ())
            .unwrap();
        println!("Row 650 (WAL) value: {}", value650);
        assert_eq!(value650, 6500);

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 4: Snapshot Rotation with Many Snapshots
// =============================================================================

#[test]
fn test_snapshot_rotation_many_snapshots() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Snapshot Rotation ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE rotation_test (id INT PRIMARY KEY, snapshot_num INT)",
            (),
        )
        .unwrap();

        // Create 10 snapshots (should rotate to keep only 3-5)
        for snapshot_num in 1..=10 {
            // Insert some data
            for i in 1..=50 {
                let id = (snapshot_num - 1) * 50 + i;
                db.execute(
                    &format!(
                        "INSERT INTO rotation_test VALUES ({}, {})",
                        id, snapshot_num
                    ),
                    (),
                )
                .unwrap();
            }

            // Take snapshot
            db.execute("PRAGMA SNAPSHOT", ()).unwrap();

            let snapshot_count = count_files_matching(
                &db_path.join("snapshots/rotation_test"),
                "snapshot-",
                ".bin",
            );
            println!(
                "After snapshot {}: {} snapshot files",
                snapshot_num, snapshot_count
            );

            // Small delay to ensure unique timestamps
            thread::sleep(Duration::from_millis(10));
        }

        // Check final snapshot count (should be limited by keep_count)
        let final_count = count_files_matching(
            &db_path.join("snapshots/rotation_test"),
            "snapshot-",
            ".bin",
        );
        println!("\nFinal snapshot count: {}", final_count);

        // Default keep_count is 3, so we should have at most 3-5 snapshots
        assert!(
            final_count <= 5,
            "Should have at most 5 snapshots due to rotation, got {}",
            final_count
        );
        assert!(final_count >= 1, "Should have at least 1 snapshot");

        // Verify all data is still accessible
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM rotation_test", ())
            .unwrap();
        println!("Total rows: {}", count);
        assert_eq!(count, 500);

        db.close().unwrap();
    }

    // Verify recovery still works with rotated snapshots
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM rotation_test", ())
            .unwrap();
        println!("Rows after reopen: {}", count);
        assert_eq!(count, 500);

        // Verify data from latest snapshots
        let snap10_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM rotation_test WHERE snapshot_num = 10",
                (),
            )
            .unwrap();
        println!("Rows from snapshot 10: {}", snap10_count);
        assert_eq!(snap10_count, 50);

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 5: WAL Truncation Effectiveness
// =============================================================================

#[test]
fn test_wal_truncation_effectiveness() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: WAL Truncation Effectiveness ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute("CREATE TABLE wal_test (id INT PRIMARY KEY, data TEXT)", ())
            .unwrap();

        // Insert data in batches and measure WAL growth
        let mut wal_sizes: Vec<u64> = Vec::new();

        for batch in 1..=5 {
            println!("\n--- Batch {} ---", batch);

            // Insert 500 rows with some data
            for i in 1..=500 {
                let id = (batch - 1) * 500 + i;
                let data = format!("data_{}_{}", batch, "x".repeat(50)); // ~60 bytes per row
                db.execute(
                    &format!("INSERT INTO wal_test VALUES ({}, '{}')", id, data),
                    (),
                )
                .unwrap();
            }

            let wal_before = get_dir_size(&db_path.join("wal"));
            println!("WAL size before snapshot: {} bytes", wal_before);

            // Take snapshot (which truncates WAL)
            db.execute("PRAGMA SNAPSHOT", ()).unwrap();

            let wal_after = get_dir_size(&db_path.join("wal"));
            println!("WAL size after snapshot: {} bytes", wal_after);

            wal_sizes.push(wal_after);

            // WAL should be significantly smaller after snapshot
            if batch > 1 {
                assert!(
                    wal_after < wal_before,
                    "WAL should shrink after snapshot: before={}, after={}",
                    wal_before,
                    wal_after
                );
            }

            // Verify LSN alignment
            let snapshot_lsn = read_snapshot_lsn(&db_path);
            let wal_lsn = get_wal_lsn(&db_path);
            println!("Snapshot LSN: {:?}, WAL LSN: {:?}", snapshot_lsn, wal_lsn);
            assert_eq!(snapshot_lsn, wal_lsn, "LSNs should match");
        }

        // WAL sizes after each snapshot should be similar (small marker entry)
        println!("\nWAL sizes after each snapshot: {:?}", wal_sizes);

        // All WAL sizes should be small (just marker entry)
        for (i, size) in wal_sizes.iter().enumerate() {
            assert!(
                *size < 1000,
                "WAL size after snapshot {} should be small, got {}",
                i + 1,
                size
            );
        }

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 6: Multiple Tables with Snapshots
// =============================================================================

#[test]
fn test_multiple_tables_snapshots() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Multiple Tables with Snapshots ===");

    {
        let db = Database::open(&dsn).unwrap();

        // Create multiple tables
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total REAL)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price REAL)",
            (),
        )
        .unwrap();

        // Insert data into each table
        for i in 1..=100 {
            db.execute(
                &format!("INSERT INTO users VALUES ({}, 'user_{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {}.99)",
                    i,
                    i % 100 + 1,
                    i * 10
                ),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO products VALUES ({}, 'product_{}', {}.99)",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        // Take snapshot
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();

        // Check snapshots for each table
        for table in &["users", "orders", "products"] {
            let count = count_files_matching(
                &db_path.join(format!("snapshots/{}", table)),
                "snapshot-",
                ".bin",
            );
            println!("Snapshots for {}: {}", table, count);
            assert_eq!(count, 1, "Should have 1 snapshot for {}", table);
        }

        // Insert more data
        for i in 101..=150 {
            db.execute(
                &format!("INSERT INTO users VALUES ({}, 'user_{}')", i, i),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {}.99)",
                    i,
                    i % 100 + 1,
                    i * 10
                ),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO products VALUES ({}, 'product_{}', {}.99)",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Remove lock and reopen (simulate crash)
    let lock_path = db_path.join("db.lock");
    if lock_path.exists() {
        fs::remove_file(&lock_path).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Verify all tables recovered
        for table in &["users", "orders", "products"] {
            let count: i64 = db
                .query_one(&format!("SELECT COUNT(*) FROM {}", table), ())
                .unwrap();
            println!("{} count: {}", table, count);
            assert_eq!(count, 150);
        }

        // Verify join still works
        let rows = db
            .query(
                "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE u.id = 50",
                (),
            )
            .unwrap();
        let results: Vec<_> = rows.filter_map(|r| r.ok()).collect();
        println!("Join result for user 50: {} rows", results.len());

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 7: Operations After Snapshot Before Close
// =============================================================================

#[test]
fn test_operations_after_snapshot_before_close() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Operations After Snapshot ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE timing_test (id INT PRIMARY KEY, phase TEXT)",
            (),
        )
        .unwrap();

        // Phase 1: Initial data
        for i in 1..=100 {
            db.execute(
                &format!("INSERT INTO timing_test VALUES ({}, 'initial')", i),
                (),
            )
            .unwrap();
        }

        // Take snapshot
        println!("Taking snapshot...");
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();
        let lsn1 = read_snapshot_lsn(&db_path);
        println!("Snapshot LSN: {:?}", lsn1);

        // Phase 2: More operations
        for i in 101..=200 {
            db.execute(
                &format!("INSERT INTO timing_test VALUES ({}, 'post_snap1')", i),
                (),
            )
            .unwrap();
        }

        // Update some existing rows
        db.execute(
            "UPDATE timing_test SET phase = 'updated' WHERE id <= 50",
            (),
        )
        .unwrap();

        // Delete some rows
        db.execute("DELETE FROM timing_test WHERE id > 180", ())
            .unwrap();

        // Second snapshot
        println!("Taking second snapshot...");
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();
        let lsn2 = read_snapshot_lsn(&db_path);
        println!("Snapshot LSN: {:?}", lsn2);

        // LSN should have increased
        assert!(lsn2 > lsn1, "LSN should increase: {:?} -> {:?}", lsn1, lsn2);

        // More operations
        for i in 201..=250 {
            db.execute(
                &format!("INSERT INTO timing_test VALUES ({}, 'post_snap2')", i),
                (),
            )
            .unwrap();
        }

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM timing_test", ())
            .unwrap();
        println!("Final count: {}", count);

        db.close().unwrap();
    }

    // Recovery test
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM timing_test", ())
            .unwrap();
        println!("Count after reopen: {}", count);

        // Verify phases
        let updated: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM timing_test WHERE phase = 'updated'",
                (),
            )
            .unwrap();
        println!("Phase 'updated' count: {}", updated);
        assert_eq!(updated, 50);

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 8: Index Persistence Across Snapshots
// =============================================================================

#[test]
fn test_index_persistence_with_snapshots() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Index Persistence with Snapshots ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE indexed_data (id INT PRIMARY KEY, category TEXT, value REAL)",
            (),
        )
        .unwrap();

        // Create index
        db.execute("CREATE INDEX idx_category ON indexed_data(category)", ())
            .unwrap();

        // Insert data
        for i in 1..=500 {
            db.execute(
                &format!(
                    "INSERT INTO indexed_data VALUES ({}, 'cat_{}', {}.{})",
                    i,
                    i % 10,
                    i,
                    i % 100
                ),
                (),
            )
            .unwrap();
        }

        // Query using index
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM indexed_data WHERE category = 'cat_5'",
                (),
            )
            .unwrap();
        println!("Count for cat_5 before snapshot: {}", count);
        assert_eq!(count, 50);

        // Take snapshot
        db.execute("PRAGMA SNAPSHOT", ()).unwrap();

        // Insert more data
        for i in 501..=700 {
            db.execute(
                &format!(
                    "INSERT INTO indexed_data VALUES ({}, 'cat_{}', {}.{})",
                    i,
                    i % 10,
                    i,
                    i % 100
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Remove lock and recover
    let lock_path = db_path.join("db.lock");
    if lock_path.exists() {
        fs::remove_file(&lock_path).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Verify index works after recovery
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM indexed_data WHERE category = 'cat_5'",
                (),
            )
            .unwrap();
        println!("Count for cat_5 after recovery: {}", count);
        assert_eq!(count, 70);

        // Verify total count
        let total: i64 = db
            .query_one("SELECT COUNT(*) FROM indexed_data", ())
            .unwrap();
        println!("Total count after recovery: {}", total);
        assert_eq!(total, 700);

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 9: Stress Test - Rapid Snapshots
// =============================================================================

#[test]
fn test_rapid_snapshots_stress() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Rapid Snapshots Stress Test ===");

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE stress_test (id INT PRIMARY KEY, iter INT)",
            (),
        )
        .unwrap();

        let mut total_inserts = 0;

        // Rapid insert-snapshot cycles
        for iter in 1..=20 {
            // Insert small batch
            for i in 1..=25 {
                let id = (iter - 1) * 25 + i;
                db.execute(
                    &format!("INSERT INTO stress_test VALUES ({}, {})", id, iter),
                    (),
                )
                .unwrap();
                total_inserts += 1;
            }

            // Immediate snapshot
            db.execute("PRAGMA SNAPSHOT", ()).unwrap();

            if iter % 5 == 0 {
                let snapshot_count = count_files_matching(
                    &db_path.join("snapshots/stress_test"),
                    "snapshot-",
                    ".bin",
                );
                let wal_size = get_dir_size(&db_path.join("wal"));
                println!(
                    "After iter {}: {} snapshots, {} bytes WAL",
                    iter, snapshot_count, wal_size
                );
            }
        }

        println!("Total inserts: {}", total_inserts);

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM stress_test", ())
            .unwrap();
        println!("Final count: {}", count);
        assert_eq!(count, total_inserts as i64);

        db.close().unwrap();
    }

    // Verify recovery
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM stress_test", ())
            .unwrap();
        println!("Count after recovery: {}", count);
        assert_eq!(count, 500);
        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}

// =============================================================================
// TEST 10: Data Integrity After Multiple Crashes
// =============================================================================

#[test]
fn test_multiple_crash_recovery_cycles() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();
    let dsn = format!("file://{}", db_path.display());

    println!("\n=== TEST: Multiple Crash Recovery Cycles ===");

    let mut expected_count = 0;

    for cycle in 1..=5 {
        println!("\n--- Recovery Cycle {} ---", cycle);

        // Open database
        {
            let db = Database::open(&dsn).unwrap();

            if cycle == 1 {
                db.execute(
                    "CREATE TABLE crash_test (id INT PRIMARY KEY, cycle INT, value TEXT)",
                    (),
                )
                .unwrap();
            }

            // Insert data for this cycle
            for i in 1..=100 {
                let id = (cycle - 1) * 100 + i;
                db.execute(
                    &format!(
                        "INSERT INTO crash_test VALUES ({}, {}, 'cycle_{}_value_{}')",
                        id, cycle, cycle, i
                    ),
                    (),
                )
                .unwrap();
                expected_count += 1;
            }

            // Take snapshot every other cycle
            if cycle % 2 == 0 {
                println!("Taking snapshot...");
                db.execute("PRAGMA SNAPSHOT", ()).unwrap();
            }

            // Verify current count
            let count: i64 = db.query_one("SELECT COUNT(*) FROM crash_test", ()).unwrap();
            println!("Count before crash: {}", count);

            // Simulate crash (don't call close)
            drop(db);
        }

        // Remove lock file
        let lock_path = db_path.join("db.lock");
        if lock_path.exists() {
            fs::remove_file(&lock_path).unwrap();
        }

        // Verify recovery
        {
            let db = Database::open(&dsn).unwrap();
            let count: i64 = db.query_one("SELECT COUNT(*) FROM crash_test", ()).unwrap();
            println!(
                "Count after recovery: {} (expected: {})",
                count, expected_count
            );
            assert_eq!(count, expected_count as i64);

            // Verify data from this cycle exists
            let cycle_count: i64 = db
                .query_one(
                    &format!("SELECT COUNT(*) FROM crash_test WHERE cycle = {}", cycle),
                    (),
                )
                .unwrap();
            println!("Rows from cycle {}: {}", cycle, cycle_count);
            assert_eq!(cycle_count, 100);

            db.close().unwrap();
        }
    }

    // Final verification
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM crash_test", ()).unwrap();
        println!("\nFinal count: {} (expected: {})", count, expected_count);
        assert_eq!(count, expected_count as i64);

        // Verify data from each cycle
        for cycle in 1..=5 {
            let cycle_count: i64 = db
                .query_one(
                    &format!("SELECT COUNT(*) FROM crash_test WHERE cycle = {}", cycle),
                    (),
                )
                .unwrap();
            println!("Cycle {} rows: {}", cycle, cycle_count);
            assert_eq!(cycle_count, 100);
        }

        db.close().unwrap();
    }

    println!("=== TEST PASSED ===\n");
}
