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

//! WAL truncation tests.
//!
//! Verifies that WAL files are properly truncated after checkpoint cycles,
//! even under continuous write pressure. Also verifies that no data is lost
//! after WAL truncation and that multiple checkpoint cycles keep the WAL
//! bounded while preserving all committed data.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use stoolap::Database;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the total size in bytes of all WAL files in the wal/ directory.
fn wal_dir_total_size(db_path: &Path) -> u64 {
    let wal_dir = db_path.join("wal");
    if !wal_dir.exists() {
        return 0;
    }
    let mut total: u64 = 0;
    if let Ok(entries) = fs::read_dir(&wal_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if (name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log") {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

fn query_i64(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

fn query_str(db: &Database, sql: &str) -> String {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<String>(0).ok())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// 1. WAL truncates under continuous writes
// ---------------------------------------------------------------------------

#[test]
fn test_wal_truncates_under_continuous_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("wal_trunc_continuous");
    let dsn = format!(
        "file://{}?checkpoint_interval=2&sync_mode=off",
        db_path.display()
    );

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE writes (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    // Signal to stop the writer thread
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // Clone the database handle so the writer thread shares the same engine
    let wdb = db.clone();
    let writer = thread::spawn(move || {
        let mut i = 0i64;
        while !stop_clone.load(Ordering::Relaxed) {
            i += 1;
            // Use auto-commit inserts to keep writes flowing
            let _ = wdb.execute(
                &format!("INSERT INTO writes VALUES ({}, 'row_{}')", i, i),
                (),
            );
            if i % 100 == 0 {
                thread::sleep(Duration::from_millis(1));
            }
        }
        i
    });

    // Let the writer run for 8 seconds (background checkpoint at 2s interval
    // should fire multiple times)
    thread::sleep(Duration::from_secs(8));
    stop.store(true, Ordering::Relaxed);
    let rows_inserted = writer.join().unwrap();

    // Now run an explicit checkpoint to ensure WAL truncation happens
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Check WAL directory size: after checkpoint the WAL should be bounded.
    // With continuous writes producing potentially tens of thousands of rows,
    // an unbounded WAL would be many megabytes. After truncation it should
    // be well under 1 MB.
    let wal_size = wal_dir_total_size(&db_path);
    assert!(
        wal_size < 1_048_576,
        "WAL size after checkpoint should be < 1MB, got {} bytes. \
         {} rows were inserted. WAL truncation may not be working.",
        wal_size,
        rows_inserted,
    );

    // Verify data is queryable (some rows should exist)
    let count = query_i64(&db, "SELECT COUNT(*) FROM writes");
    assert!(
        count > 0,
        "Expected rows in table after continuous writes, got {}",
        count,
    );

    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// 2. No data loss after WAL truncation
// ---------------------------------------------------------------------------

#[test]
fn test_no_data_loss_after_wal_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("wal_trunc_nodataloss");
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&sync_mode=off",
        db_path.display()
    );

    // Phase 1: insert 1000 rows, checkpoint, close
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
            (),
        )
        .unwrap();

        db.execute("BEGIN", ()).unwrap();
        for i in 1..=1000 {
            db.execute(
                &format!("INSERT INTO items VALUES ({}, 'item_{}', {})", i, i, i * 10),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint triggers seal + WAL truncation
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify before close
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 1000);
        assert_eq!(
            query_i64(&db, "SELECT value FROM items WHERE id = 500"),
            5000
        );

        db.close().unwrap();
    }

    // Phase 2: reopen and verify all 1000 rows survived
    {
        let db = Database::open(&dsn).unwrap();
        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(
            count, 1000,
            "Expected 1000 rows after reopen, got {}",
            count
        );

        // Spot-check boundaries
        assert_eq!(
            query_str(&db, "SELECT name FROM items WHERE id = 1"),
            "item_1"
        );
        assert_eq!(query_i64(&db, "SELECT value FROM items WHERE id = 1"), 10);
        assert_eq!(
            query_str(&db, "SELECT name FROM items WHERE id = 1000"),
            "item_1000"
        );
        assert_eq!(
            query_i64(&db, "SELECT value FROM items WHERE id = 1000"),
            10000
        );

        // Spot-check middle
        assert_eq!(
            query_str(&db, "SELECT name FROM items WHERE id = 500"),
            "item_500"
        );
        assert_eq!(
            query_i64(&db, "SELECT value FROM items WHERE id = 500"),
            5000
        );

        // Verify SUM to catch any corrupted values
        assert_eq!(
            query_i64(&db, "SELECT SUM(value) FROM items"),
            (1..=1000i64).map(|i| i * 10).sum::<i64>()
        );

        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 3. WAL truncation after multiple checkpoints
// ---------------------------------------------------------------------------

#[test]
fn test_wal_truncation_after_multiple_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("wal_trunc_multi_ckpt");
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&sync_mode=off",
        db_path.display()
    );

    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE batches (id INTEGER PRIMARY KEY, batch INTEGER, data TEXT)",
        (),
    )
    .unwrap();

    // --- Batch 1: 500 rows, then checkpoint ---
    db.execute("BEGIN", ()).unwrap();
    for i in 1..=500 {
        db.execute(
            &format!("INSERT INTO batches VALUES ({}, 1, 'b1_{}')", i, i),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM batches"), 500);
    let wal_size_after_ckpt1 = wal_dir_total_size(&db_path);

    // --- Batch 2: 500 more rows, then checkpoint ---
    db.execute("BEGIN", ()).unwrap();
    for i in 501..=1000 {
        db.execute(
            &format!("INSERT INTO batches VALUES ({}, 2, 'b2_{}')", i, i),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM batches"), 1000);
    let wal_size_after_ckpt2 = wal_dir_total_size(&db_path);

    // --- Batch 3: 500 more rows, then checkpoint ---
    db.execute("BEGIN", ()).unwrap();
    for i in 1001..=1500 {
        db.execute(
            &format!("INSERT INTO batches VALUES ({}, 3, 'b3_{}')", i, i),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM batches"), 1500);
    let wal_size_after_ckpt3 = wal_dir_total_size(&db_path);

    // The WAL should stay bounded across checkpoints, not grow unboundedly.
    // After each checkpoint + truncation, the WAL only contains entries since
    // the last checkpoint. So it should not accumulate.
    // Allow a generous 512KB threshold; the actual WAL after truncation for
    // 500 small rows should be much smaller.
    let max_wal_size: u64 = 512 * 1024;
    assert!(
        wal_size_after_ckpt3 < max_wal_size,
        "WAL after 3 checkpoints should be bounded (< 512KB), got {} bytes",
        wal_size_after_ckpt3,
    );

    // WAL should not grow monotonically: after each checkpoint it should
    // be roughly the same size (just the post-checkpoint DDL re-record).
    // We check that checkpoint 3 did not accumulate all prior WAL data.
    // (ckpt3 size should not be 3x ckpt1 size, within a reasonable margin)
    if wal_size_after_ckpt1 > 0 {
        assert!(
            wal_size_after_ckpt3 < wal_size_after_ckpt1 * 4,
            "WAL grew too much across checkpoints: ckpt1={}, ckpt2={}, ckpt3={}",
            wal_size_after_ckpt1,
            wal_size_after_ckpt2,
            wal_size_after_ckpt3,
        );
    }

    // Verify all data is present
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM batches WHERE batch = 1"),
        500
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM batches WHERE batch = 2"),
        500
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM batches WHERE batch = 3"),
        500
    );

    // Spot-check values
    assert_eq!(
        query_str(&db, "SELECT data FROM batches WHERE id = 1"),
        "b1_1"
    );
    assert_eq!(
        query_str(&db, "SELECT data FROM batches WHERE id = 750"),
        "b2_750"
    );
    assert_eq!(
        query_str(&db, "SELECT data FROM batches WHERE id = 1500"),
        "b3_1500"
    );

    // Reopen and verify everything survived the truncations
    db.close().unwrap();
    {
        let db2 = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db2, "SELECT COUNT(*) FROM batches"), 1500);
        assert_eq!(
            query_i64(&db2, "SELECT COUNT(*) FROM batches WHERE batch = 1"),
            500
        );
        assert_eq!(
            query_i64(&db2, "SELECT COUNT(*) FROM batches WHERE batch = 2"),
            500
        );
        assert_eq!(
            query_i64(&db2, "SELECT COUNT(*) FROM batches WHERE batch = 3"),
            500
        );
        db2.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 4. Data integrity across checkpoint cycles with multiple tables
// ---------------------------------------------------------------------------

#[test]
fn test_data_integrity_across_checkpoint_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("wal_trunc_integrity");
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&sync_mode=off",
        db_path.display()
    );

    // Phase 1: create tables, insert, checkpoint in multiple rounds
    {
        let db = Database::open(&dsn).unwrap();

        // Create three tables
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY, event TEXT, ts INTEGER)",
            (),
        )
        .unwrap();

        // --- Round 1: insert into all tables, checkpoint ---
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO users VALUES ({}, 'user_{}', {})",
                    i,
                    i,
                    20 + (i % 40)
                ),
                (),
            )
            .unwrap();
        }
        for i in 1..=100 {
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {:.2})",
                    i,
                    (i % 50) + 1,
                    i as f64 * 9.99
                ),
                (),
            )
            .unwrap();
        }
        for i in 1..=200 {
            db.execute(
                &format!(
                    "INSERT INTO logs VALUES ({}, 'event_{}', {})",
                    i,
                    i,
                    1000 + i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 50);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 100);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM logs"), 200);

        // --- Round 2: more inserts + updates, checkpoint ---
        db.execute("BEGIN", ()).unwrap();
        for i in 51..=100 {
            db.execute(
                &format!(
                    "INSERT INTO users VALUES ({}, 'user_{}', {})",
                    i,
                    i,
                    20 + (i % 40)
                ),
                (),
            )
            .unwrap();
        }
        // Update some existing users
        db.execute("UPDATE users SET age = 99 WHERE id <= 10", ())
            .unwrap();
        for i in 101..=200 {
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {:.2})",
                    i,
                    (i % 100) + 1,
                    i as f64 * 9.99
                ),
                (),
            )
            .unwrap();
        }
        for i in 201..=400 {
            db.execute(
                &format!(
                    "INSERT INTO logs VALUES ({}, 'event_{}', {})",
                    i,
                    i,
                    1000 + i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 100);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 200);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM logs"), 400);

        // Verify updates survived checkpoint
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE age = 99"),
            10
        );

        // --- Round 3: deletes + inserts, checkpoint ---
        db.execute("BEGIN", ()).unwrap();
        db.execute("DELETE FROM logs WHERE id <= 50", ()).unwrap();
        for i in 401..=500 {
            db.execute(
                &format!(
                    "INSERT INTO logs VALUES ({}, 'event_{}', {})",
                    i,
                    i,
                    1000 + i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 100);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 200);
        // 400 - 50 deleted + 100 new = 450
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM logs"), 450);

        // Verify WAL is bounded after 3 checkpoints across 3 tables
        let wal_size = wal_dir_total_size(&db_path);
        assert!(
            wal_size < 1_048_576,
            "WAL should be bounded after multiple checkpoints, got {} bytes",
            wal_size,
        );

        db.close().unwrap();
    }

    // Phase 2: reopen and verify all data across all tables
    {
        let db = Database::open(&dsn).unwrap();

        // Table counts
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users"),
            100,
            "users row count mismatch after reopen"
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM orders"),
            200,
            "orders row count mismatch after reopen"
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM logs"),
            450,
            "logs row count mismatch after reopen"
        );

        // Verify updated rows survived
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE age = 99"),
            10,
            "updated users with age=99 lost after reopen"
        );

        // Verify specific values
        assert_eq!(
            query_str(&db, "SELECT name FROM users WHERE id = 1"),
            "user_1"
        );
        assert_eq!(
            query_i64(&db, "SELECT age FROM users WHERE id = 5"),
            99, // was updated in round 2
        );
        assert_eq!(
            query_str(&db, "SELECT name FROM users WHERE id = 100"),
            "user_100"
        );

        // Verify deletes: logs 1-50 should be gone
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM logs WHERE id <= 50"),
            0,
            "deleted log rows reappeared after reopen"
        );

        // Verify logs that should still exist
        assert_eq!(
            query_str(&db, "SELECT event FROM logs WHERE id = 51"),
            "event_51"
        );
        assert_eq!(query_i64(&db, "SELECT ts FROM logs WHERE id = 500"), 1500);

        // Verify orders aggregation
        let order_count = query_i64(&db, "SELECT COUNT(*) FROM orders WHERE user_id = 1");
        assert!(
            order_count > 0,
            "Expected orders for user_id=1, got {}",
            order_count,
        );

        db.close().unwrap();
    }
}
