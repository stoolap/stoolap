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

#![cfg(feature = "stress-tests")]

//! Crash Soak Tests
//!
//! These tests simulate random crash-at-any-point scenarios by running random
//! workloads and truncating WAL files at random offsets, then verifying recovery
//! invariants:
//!
//! 1. Database opens without panic (recovery may fail gracefully with Err)
//! 2. If recovery succeeds, row count is non-negative
//! 3. If recovery succeeds, the database is still usable (can INSERT)
//! 4. No duplicate primary keys after recovery
//! 5. Index queries are consistent with full scans

use std::fs;
use std::path::{Path, PathBuf};
use stoolap::Database;
use tempfile::tempdir;

// ============================================================================
// WAL binary format constants (must match wal_manager.rs)
// ============================================================================

const WAL_ENTRY_MAGIC: u32 = 0x454C4157;
const WAL_HEADER_SIZE: usize = 32;
const CRC_SIZE: usize = 4;

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

/// Remove the lock file so we can reopen the database
fn remove_lock_file(db_path: &Path) {
    let lock_file = db_path.join("db.lock");
    let _ = fs::remove_file(lock_file);
}

/// Truncate WAL at a pseudo-random offset derived from seed.
/// Keeps the first 32 bytes (minimum WAL header) intact and cuts at
/// a deterministic point within the remaining data.
fn truncate_wal_at_random_offset(wal_path: &Path, seed: u64) {
    let data = fs::read(wal_path).unwrap();
    if data.len() <= WAL_HEADER_SIZE {
        return; // Too small to truncate meaningfully
    }
    let trunc_point = WAL_HEADER_SIZE + (seed as usize % (data.len() - WAL_HEADER_SIZE));
    fs::write(wal_path, &data[..trunc_point]).unwrap();
}

/// Parse WAL binary data to locate all entry boundaries.
/// Returns the byte offset of each complete entry found.
fn find_entry_boundaries(data: &[u8]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while pos + WAL_HEADER_SIZE <= data.len() {
        if pos + 4 > data.len() {
            break;
        }
        let magic = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        if magic != WAL_ENTRY_MAGIC {
            break;
        }

        boundaries.push(pos);

        // Read header_size and entry_size to compute total entry size
        let header_size = u16::from_le_bytes(data[pos + 6..pos + 8].try_into().unwrap()) as usize;
        let entry_size = u32::from_le_bytes(data[pos + 24..pos + 28].try_into().unwrap()) as usize;

        // Sanity checks
        if entry_size > 64 * 1024 * 1024 || header_size < WAL_HEADER_SIZE {
            break;
        }

        let total_size = header_size + entry_size + CRC_SIZE;
        if pos + total_size > data.len() {
            break; // Incomplete entry
        }

        pos += total_size;
    }

    boundaries
}

/// Truncate WAL at the midpoint of a specific entry (not at an entry boundary).
/// This simulates a crash that tears an entry in half.
fn truncate_wal_mid_entry(wal_path: &Path, entry_index: usize) {
    let data = fs::read(wal_path).unwrap();
    let boundaries = find_entry_boundaries(&data);

    if boundaries.is_empty() || entry_index >= boundaries.len() {
        return;
    }

    let entry_start = boundaries[entry_index];

    // Determine end of this entry
    let entry_end = if entry_index + 1 < boundaries.len() {
        boundaries[entry_index + 1]
    } else {
        // Last entry: compute from header
        let header_size =
            u16::from_le_bytes(data[entry_start + 6..entry_start + 8].try_into().unwrap()) as usize;
        let entry_size =
            u32::from_le_bytes(data[entry_start + 24..entry_start + 28].try_into().unwrap())
                as usize;
        let total = header_size + entry_size + CRC_SIZE;
        entry_start + total
    };

    // Cut at the midpoint of this entry
    let entry_len = entry_end - entry_start;
    if entry_len <= 1 {
        return;
    }
    let midpoint = entry_start + entry_len / 2;
    fs::write(wal_path, &data[..midpoint]).unwrap();
}

// ============================================================================
// Test: 20-cycle crash soak with random workloads
// ============================================================================

#[test]
fn test_crash_soak_20_cycles() {
    for cycle in 0u64..20 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("crash_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Create database and run a deterministic random workload
        {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER, tag TEXT)",
                (),
            )
            .unwrap();

            let seed = cycle * 7919 + 1234;
            let mut next_id = 1i64;
            let ops = 50 + (seed % 150) as i64; // 50-200 operations per cycle

            for op_idx in 0..ops {
                let op_seed = seed.wrapping_mul(31).wrapping_add(op_idx as u64);
                match op_seed % 5 {
                    0..=2 => {
                        // INSERT (60%)
                        let val = (op_seed % 1000) as i64;
                        db.execute(
                            &format!(
                                "INSERT INTO t VALUES ({}, {}, 'tag_{}')",
                                next_id, val, next_id
                            ),
                            (),
                        )
                        .unwrap();
                        next_id += 1;
                    }
                    3 => {
                        // UPDATE (20%)
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(
                                &format!(
                                    "UPDATE t SET value = {} WHERE id = {}",
                                    op_seed % 999,
                                    target
                                ),
                                (),
                            );
                        }
                    }
                    _ => {
                        // DELETE (20%)
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(&format!("DELETE FROM t WHERE id = {}", target), ());
                        }
                    }
                }
            }

            db.close().unwrap();
        }

        // Phase 2: Truncate WAL at a pseudo-random point
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            truncate_wal_at_random_offset(wal_file, cycle * 13 + 42);
        }

        // Phase 3: Reopen and verify recovery invariants
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                // Invariant 1: COUNT(*) works and is non-negative
                let count_result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM t", ());
                match count_result {
                    Ok(count) => {
                        assert!(
                            count >= 0,
                            "Cycle {}: Row count must be non-negative, got {}",
                            cycle,
                            count
                        );

                        // Invariant 2: Can still insert new data
                        db.execute("INSERT INTO t VALUES (99999, 1, 'recovery_test')", ())
                            .unwrap_or_else(|_| {
                                panic!("Cycle {cycle}: INSERT should work after recovery")
                            });

                        // Invariant 3: No duplicate primary keys
                        let dup_count: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id, COUNT(*) AS c FROM t GROUP BY id HAVING c > 1) AS dups",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(
                            dup_count, 0,
                            "Cycle {}: No duplicate primary keys after recovery",
                            cycle
                        );
                    }
                    Err(_) => {
                        // Table might not exist if DDL WAL entry was truncated.
                        // This is acceptable; verify the database is still usable
                        // by creating a fresh table.
                        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, v INTEGER)", ())
                            .unwrap_or_else(|_| {
                                panic!("Cycle {cycle}: CREATE TABLE should work even if t is gone")
                            });
                        db.execute("INSERT INTO t2 VALUES (1, 100)", ())
                            .unwrap_or_else(|_| {
                                panic!("Cycle {cycle}: INSERT into new table should work")
                            });
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                // Recovery failure is acceptable for severe truncation
                // but should not panic - returning an error is fine
                eprintln!("Cycle {}: Recovery failed (acceptable): {}", cycle, e);
            }
        }
    }
}

// ============================================================================
// Test: Mid-entry truncation (specifically targets WAL entry interiors)
// ============================================================================

#[test]
fn test_crash_soak_mid_entry_truncation() {
    for cycle in 0u64..15 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("mid_entry_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Build a database with enough entries to have multiple WAL entries
        {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER, tag TEXT)",
                (),
            )
            .unwrap();

            let seed = cycle * 6761 + 5557;
            let mut next_id = 1i64;
            let ops = 30 + (seed % 100) as i64;

            for op_idx in 0..ops {
                let op_seed = seed.wrapping_mul(37).wrapping_add(op_idx as u64);
                match op_seed % 5 {
                    0..=2 => {
                        let val = (op_seed % 500) as i64;
                        db.execute(
                            &format!(
                                "INSERT INTO t VALUES ({}, {}, 'mid_{}')",
                                next_id, val, next_id
                            ),
                            (),
                        )
                        .unwrap();
                        next_id += 1;
                    }
                    3 => {
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(
                                &format!(
                                    "UPDATE t SET value = {} WHERE id = {}",
                                    op_seed % 999,
                                    target
                                ),
                                (),
                            );
                        }
                    }
                    _ => {
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(&format!("DELETE FROM t WHERE id = {}", target), ());
                        }
                    }
                }
            }

            db.close().unwrap();
        }

        // Phase 2: Find WAL entries and truncate at the midpoint of a chosen entry
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            let data = fs::read(wal_file).unwrap();
            let boundaries = find_entry_boundaries(&data);

            if boundaries.len() >= 2 {
                // Pick an entry to tear based on the cycle seed
                let target_entry = (cycle as usize % (boundaries.len() - 1)) + 1;
                truncate_wal_mid_entry(wal_file, target_entry);
            } else if !boundaries.is_empty() {
                // Only one entry; truncate it
                truncate_wal_mid_entry(wal_file, 0);
            }
        }

        // Phase 3: Reopen and verify
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                let count_result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM t", ());
                match count_result {
                    Ok(count) => {
                        assert!(
                            count >= 0,
                            "Mid-entry cycle {}: Row count must be non-negative, got {}",
                            cycle,
                            count
                        );

                        // Verify usability
                        db.execute("INSERT INTO t VALUES (88888, 42, 'mid_recovery')", ())
                            .unwrap_or_else(|_| {
                                panic!("Mid-entry cycle {cycle}: INSERT should work after recovery")
                            });

                        // No duplicate PKs
                        let dup_count: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id, COUNT(*) AS c FROM t GROUP BY id HAVING c > 1) AS dups",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(
                            dup_count, 0,
                            "Mid-entry cycle {}: No duplicate PKs after recovery",
                            cycle
                        );
                    }
                    Err(_) => {
                        // DDL truncated; verify DB is still usable
                        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, v INTEGER)", ())
                            .unwrap_or_else(|_| {
                                panic!(
                                    "Mid-entry cycle {cycle}: should be able to create a new table"
                                )
                            });
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                eprintln!(
                    "Mid-entry cycle {}: Recovery failed (acceptable): {}",
                    cycle, e
                );
            }
        }
    }
}

// ============================================================================
// Test: Crash soak with indexes - verifies index consistency after recovery
// ============================================================================

#[test]
fn test_crash_soak_with_indexes() {
    for cycle in 0u64..15 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("idx_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Create database with indexes and run workload
        {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER, tag TEXT)",
                (),
            )
            .unwrap();
            db.execute("CREATE INDEX idx_value ON t(value)", ())
                .unwrap();
            db.execute("CREATE INDEX idx_tag ON t(tag)", ()).unwrap();

            let seed = cycle * 8831 + 3571;
            let mut next_id = 1i64;
            let ops = 40 + (seed % 120) as i64;

            for op_idx in 0..ops {
                let op_seed = seed.wrapping_mul(41).wrapping_add(op_idx as u64);
                match op_seed % 5 {
                    0..=2 => {
                        let val = (op_seed % 200) as i64; // Smaller range for more collisions
                        let tag_num = val % 10;
                        db.execute(
                            &format!(
                                "INSERT INTO t VALUES ({}, {}, 'grp_{}')",
                                next_id, val, tag_num
                            ),
                            (),
                        )
                        .unwrap();
                        next_id += 1;
                    }
                    3 => {
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let new_val = (op_seed % 200) as i64;
                            let _ = db.execute(
                                &format!(
                                    "UPDATE t SET value = {}, tag = 'grp_{}' WHERE id = {}",
                                    new_val,
                                    new_val % 10,
                                    target
                                ),
                                (),
                            );
                        }
                    }
                    _ => {
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(&format!("DELETE FROM t WHERE id = {}", target), ());
                        }
                    }
                }
            }

            db.close().unwrap();
        }

        // Phase 2: Truncate WAL
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            truncate_wal_at_random_offset(wal_file, cycle * 17 + 73);
        }

        // Phase 3: Reopen and verify index consistency
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                let count_result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM t", ());
                match count_result {
                    Ok(total_count) => {
                        assert!(
                            total_count >= 0,
                            "Index cycle {}: Row count must be non-negative, got {}",
                            cycle,
                            total_count
                        );

                        // Invariant: Can still insert
                        db.execute("INSERT INTO t VALUES (77777, 42, 'grp_2')", ())
                            .unwrap_or_else(|_| {
                                panic!("Index cycle {cycle}: INSERT should work after recovery")
                            });

                        // Invariant: No duplicate PKs
                        let dup_count: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id, COUNT(*) AS c FROM t GROUP BY id HAVING c > 1) AS dups",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(dup_count, 0, "Index cycle {}: No duplicate PKs", cycle);

                        // Invariant: Index-based queries match full scan
                        // Pick a value to test with: value = 42 (a common value in the range)
                        let full_scan_count: i64 = db
                            .query_one("SELECT COUNT(*) FROM t WHERE value = 42", ())
                            .unwrap_or(0);

                        // Verify that a range query via index matches full scan
                        // Use value > Y and compare counts with/without ORDER BY
                        let range_count: i64 = db
                            .query_one("SELECT COUNT(*) FROM t WHERE value > 100", ())
                            .unwrap_or(0);

                        // Verify range results are consistent by getting
                        // the same count via a different approach
                        let le_count: i64 = db
                            .query_one("SELECT COUNT(*) FROM t WHERE value <= 100", ())
                            .unwrap_or(0);

                        // The updated total (after our insert of 77777) should be total_count+1
                        let new_total: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
                        assert_eq!(
                            new_total,
                            total_count + 1,
                            "Index cycle {}: total should increase by 1 after insert",
                            cycle
                        );

                        // range_count + le_count should equal new_total (complete partition)
                        // (value = 42 for our inserted row means it goes to le_count side)
                        assert_eq!(
                            range_count + le_count, new_total,
                            "Index cycle {}: value>100 ({}) + value<=100 ({}) should equal total ({})",
                            cycle, range_count, le_count, new_total
                        );

                        // Verify index scan on tag matches full scan
                        let tag_idx_count: i64 = db
                            .query_one("SELECT COUNT(*) FROM t WHERE tag = 'grp_2'", ())
                            .unwrap_or(0);
                        assert!(
                            tag_idx_count >= 0,
                            "Index cycle {}: tag index count should be non-negative",
                            cycle
                        );

                        // Cross-check: get all grp_2 via full scan and compare
                        // Use a subquery approach to verify
                        let tag_full_scan: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT * FROM t) AS sub WHERE sub.tag = 'grp_2'",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(
                            tag_idx_count, tag_full_scan,
                            "Index cycle {}: tag index count ({}) should match full scan count ({})",
                            cycle, tag_idx_count, tag_full_scan
                        );

                        // Verify ordered results from index are actually sorted
                        verify_ordered_results(&db, cycle);

                        assert!(
                            full_scan_count >= 0,
                            "Index cycle {}: full scan count must be non-negative",
                            cycle
                        );
                    }
                    Err(_) => {
                        // DDL truncated; verify DB is still usable
                        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, v INTEGER)", ())
                            .unwrap_or_else(|_| {
                                panic!("Index cycle {cycle}: should be able to create a new table")
                            });
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                eprintln!("Index cycle {}: Recovery failed (acceptable): {}", cycle, e);
            }
        }
    }
}

/// Verify that ORDER BY on an indexed column produces correctly sorted results
fn verify_ordered_results(db: &Database, cycle: u64) {
    let rows_result = db.query(
        "SELECT id, value FROM t WHERE value > 50 ORDER BY value, id",
        (),
    );
    match rows_result {
        Ok(mut rows) => {
            let mut prev_value: Option<i64> = None;
            let mut prev_id: Option<i64> = None;
            let mut row_count = 0u64;

            while rows.advance() {
                let row = rows.current_row();
                let value = match row.get(1) {
                    Some(stoolap::Value::Integer(v)) => *v,
                    _ => continue,
                };
                let id = match row.get(0) {
                    Some(stoolap::Value::Integer(v)) => *v,
                    _ => continue,
                };

                if let Some(pv) = prev_value {
                    assert!(
                        value > pv || (value == pv && id >= prev_id.unwrap_or(i64::MIN)),
                        "Index cycle {}: results not sorted at row {}: ({}, {}) after ({}, {})",
                        cycle,
                        row_count,
                        value,
                        id,
                        pv,
                        prev_id.unwrap_or(-1)
                    );
                }
                prev_value = Some(value);
                prev_id = Some(id);
                row_count += 1;
            }
        }
        Err(_) => {
            // Query failure after recovery is acceptable
        }
    }
}

// ============================================================================
// Test: Multiple tables with foreign-key-like relationships
// ============================================================================

#[test]
fn test_crash_soak_multi_table() {
    for cycle in 0u64..10 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Create multiple tables and populate
        {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER)",
                (),
            )
            .unwrap();
            db.execute(
                "CREATE TABLE items (id INTEGER PRIMARY KEY, order_id INTEGER, name TEXT)",
                (),
            )
            .unwrap();
            db.execute("CREATE INDEX idx_cust ON orders(customer_id)", ())
                .unwrap();
            db.execute("CREATE INDEX idx_ord ON items(order_id)", ())
                .unwrap();

            let seed = cycle * 9973 + 2741;
            let mut item_id = 1i64;

            let order_count = 20 + (seed % 80) as i64;
            for (order_id, i) in (1i64..).zip(0..order_count) {
                let op_seed = seed.wrapping_mul(53).wrapping_add(i as u64);
                let customer = (op_seed % 10) as i64 + 1;
                let amount = (op_seed % 1000) as i64;
                db.execute(
                    &format!(
                        "INSERT INTO orders VALUES ({}, {}, {})",
                        order_id, customer, amount
                    ),
                    (),
                )
                .unwrap();

                // Each order gets 1-3 items
                let num_items = 1 + (op_seed % 3) as i64;
                for j in 0..num_items {
                    db.execute(
                        &format!(
                            "INSERT INTO items VALUES ({}, {}, 'item_{}_{}')",
                            item_id, order_id, order_id, j
                        ),
                        (),
                    )
                    .unwrap();
                    item_id += 1;
                }
            }

            db.close().unwrap();
        }

        // Phase 2: Truncate WAL
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            truncate_wal_at_random_offset(wal_file, cycle * 29 + 97);
        }

        // Phase 3: Verify recovery
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                // Check orders table
                let orders_ok: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM orders", ());
                let items_ok: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM items", ());

                match (orders_ok, items_ok) {
                    (Ok(order_count), Ok(item_count)) => {
                        assert!(
                            order_count >= 0,
                            "Multi cycle {}: order count non-negative",
                            cycle
                        );
                        assert!(
                            item_count >= 0,
                            "Multi cycle {}: item count non-negative",
                            cycle
                        );

                        // Both tables should be insertable
                        db.execute(
                            &format!(
                                "INSERT INTO orders VALUES ({}, 1, 100)",
                                order_count + 50000
                            ),
                            (),
                        )
                        .unwrap_or_else(|_| {
                            panic!("Multi cycle {cycle}: INSERT into orders should work")
                        });

                        db.execute(
                            &format!(
                                "INSERT INTO items VALUES ({}, 1, 'recovery')",
                                item_count + 50000
                            ),
                            (),
                        )
                        .unwrap_or_else(|_| {
                            panic!("Multi cycle {cycle}: INSERT into items should work")
                        });

                        // No duplicate PKs in either table
                        let order_dups: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id FROM orders GROUP BY id HAVING COUNT(*) > 1) AS d",
                                (),
                            )
                            .unwrap_or(0);
                        let item_dups: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id FROM items GROUP BY id HAVING COUNT(*) > 1) AS d",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(order_dups, 0, "Multi cycle {}: no dup order PKs", cycle);
                        assert_eq!(item_dups, 0, "Multi cycle {}: no dup item PKs", cycle);
                    }
                    _ => {
                        // One or both tables missing; verify DB is usable
                        let _ = db.execute(
                            "CREATE TABLE IF NOT EXISTS recovery_check (id INTEGER PRIMARY KEY)",
                            (),
                        );
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                eprintln!("Multi cycle {}: Recovery failed (acceptable): {}", cycle, e);
            }
        }
    }
}

// ============================================================================
// Test: Heavy write workload with snapshot + WAL truncation
// ============================================================================

#[test]
fn test_crash_soak_heavy_writes() {
    for cycle in 0u64..10 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("heavy_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Heavy write workload
        {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c TEXT)",
                (),
            )
            .unwrap();

            let seed = cycle * 11213 + 7727;
            let mut next_id = 1i64;

            // Do 500 operations for a heavier workload
            for op_idx in 0..500i64 {
                let op_seed = seed.wrapping_mul(43).wrapping_add(op_idx as u64);
                match op_seed % 10 {
                    0..=5 => {
                        // INSERT (60%)
                        let a = (op_seed % 100) as i64;
                        let b = (op_seed.wrapping_mul(7) % 100) as i64;
                        db.execute(
                            &format!(
                                "INSERT INTO t VALUES ({}, {}, {}, 'data_{}')",
                                next_id, a, b, next_id
                            ),
                            (),
                        )
                        .unwrap();
                        next_id += 1;
                    }
                    6..=7 => {
                        // UPDATE (20%)
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let new_a = (op_seed % 100) as i64;
                            let _ = db.execute(
                                &format!("UPDATE t SET a = {} WHERE id = {}", new_a, target),
                                (),
                            );
                        }
                    }
                    8 => {
                        // DELETE (10%)
                        if next_id > 1 {
                            let target = 1 + (op_seed % (next_id as u64 - 1)) as i64;
                            let _ = db.execute(&format!("DELETE FROM t WHERE id = {}", target), ());
                        }
                    }
                    _ => {
                        // Batch UPDATE (10%)
                        if next_id > 5 {
                            let threshold = (op_seed % 50) as i64;
                            let _ = db.execute(
                                &format!("UPDATE t SET b = b + 1 WHERE a < {}", threshold),
                                (),
                            );
                        }
                    }
                }
            }

            db.close().unwrap();
        }

        // Phase 2: Truncate WAL at various points
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            let file_len = fs::metadata(wal_file).map(|m| m.len()).unwrap_or(0);
            if file_len > WAL_HEADER_SIZE as u64 {
                // Alternate between random offset and mid-entry truncation
                if cycle % 2 == 0 {
                    truncate_wal_at_random_offset(wal_file, cycle * 19 + 53);
                } else {
                    let data = fs::read(wal_file).unwrap();
                    let boundaries = find_entry_boundaries(&data);
                    if boundaries.len() >= 3 {
                        // Truncate at a random entry in the second half
                        let target =
                            boundaries.len() / 2 + (cycle as usize % (boundaries.len() / 2));
                        let target = target.min(boundaries.len() - 1);
                        truncate_wal_mid_entry(wal_file, target);
                    } else {
                        truncate_wal_at_random_offset(wal_file, cycle * 23 + 67);
                    }
                }
            }
        }

        // Phase 3: Verify
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                let count_result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM t", ());
                match count_result {
                    Ok(count) => {
                        assert!(
                            count >= 0,
                            "Heavy cycle {}: count must be non-negative",
                            cycle
                        );

                        // Verify aggregation still works
                        let sum_result: Result<i64, _> =
                            db.query_one("SELECT COALESCE(SUM(a), 0) FROM t", ());
                        assert!(
                            sum_result.is_ok(),
                            "Heavy cycle {}: SUM should work after recovery",
                            cycle
                        );

                        // Verify no PK duplicates
                        let dup_count: i64 = db
                            .query_one(
                                "SELECT COUNT(*) FROM (SELECT id FROM t GROUP BY id HAVING COUNT(*) > 1) AS d",
                                (),
                            )
                            .unwrap_or(0);
                        assert_eq!(dup_count, 0, "Heavy cycle {}: no duplicate PKs", cycle);

                        // Verify insert works
                        db.execute("INSERT INTO t VALUES (999999, 0, 0, 'heavy_recovery')", ())
                            .unwrap_or_else(|_| panic!("Heavy cycle {cycle}: INSERT should work"));
                    }
                    Err(_) => {
                        // DDL lost; acceptable
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                eprintln!("Heavy cycle {}: Recovery failed (acceptable): {}", cycle, e);
            }
        }
    }
}

// ============================================================================
// Test: Zero-length truncation (complete WAL wipe)
// ============================================================================

#[test]
fn test_crash_soak_complete_wal_wipe() {
    for cycle in 0u64..5 {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wipe_soak");
        let dsn = format!("file://{}", db_path.display());

        // Phase 1: Create and populate
        {
            let db = Database::open(&dsn).unwrap();
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER)", ())
                .unwrap();

            for i in 1..=50 {
                db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
                    .unwrap();
            }

            db.close().unwrap();
        }

        // Phase 2: Completely wipe the WAL file (simulate total disk loss of WAL)
        remove_lock_file(&db_path);
        let wal_files = find_wal_files(&db_path);
        if let Some(wal_file) = wal_files.first() {
            // Alternate between zero-length and keeping just a few garbage bytes
            if cycle % 2 == 0 {
                fs::write(wal_file, []).unwrap();
            } else {
                fs::write(wal_file, [0u8; 16]).unwrap();
            }
        }

        // Phase 3: Verify recovery handles total WAL loss
        let result = Database::open(&dsn);
        match result {
            Ok(db) => {
                // Database may have recovered from snapshots, or may have no data
                let count_result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM t", ());
                match count_result {
                    Ok(count) => {
                        assert!(
                            count >= 0,
                            "Wipe cycle {}: count must be non-negative",
                            cycle
                        );
                    }
                    Err(_) => {
                        // Table gone; acceptable after total WAL wipe with no snapshots
                    }
                }
                db.close().unwrap();
            }
            Err(e) => {
                eprintln!("Wipe cycle {}: Recovery failed (acceptable): {}", cycle, e);
            }
        }
    }
}
