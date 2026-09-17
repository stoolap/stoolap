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

//! Failpoint I/O Test Matrix
//!
//! Systematically tests I/O failure scenarios by arming failpoint flags
//! in the source code and verifying that:
//!
//! 1. Operations return appropriate errors when failpoints are armed
//! 2. No partial state is left behind (atomicity)
//! 3. The database recovers correctly after failpoint is disarmed
//!
//! Each failpoint is a `#[cfg(test)]`-guarded `AtomicBool` check, so
//! production builds have zero overhead.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::{AtomicBool, Ordering};
use stoolap::test_failpoints;
use stoolap::Database;
use tempfile::tempdir;

/// RAII guard that resets all failpoints on drop (even on panic).
fn failpoint_guard() -> test_failpoints::FailpointGuard {
    test_failpoints::FailpointGuard::new()
}

// ============================================================================
// WAL Write Failpoint Tests
// ============================================================================

#[test]
fn test_wal_write_fail_returns_error() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display()))
        .expect("Failed to open database");

    db.execute("CREATE TABLE fp_wal (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("CREATE should succeed");

    // Arm the failpoint
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);

    // Writes should fail
    let result = db.execute("INSERT INTO fp_wal VALUES (1, 'hello')", ());
    assert!(
        result.is_err(),
        "INSERT should fail with WAL write failpoint armed"
    );

    // Disarm
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);

    // The WAL is poisoned: retrying its buffer could re-write the failed
    // transaction's commit marker and resurrect it at recovery. Writes on
    // this handle must keep failing until the database is reopened.
    let result = db.execute("INSERT INTO fp_wal VALUES (2, 'after_fail')", ());
    assert!(
        result.is_err(),
        "writes must fail on a poisoned WAL until reopen"
    );

    // Reopen: recovery discards the unacknowledged transaction and the
    // database is usable again.
    let dsn = format!("file://{}", dir.path().display());
    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen after poisoned WAL");
    db.execute("INSERT INTO fp_wal VALUES (2, 'after_reopen')", ())
        .expect("INSERT should succeed after reopen");
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_wal", ())
        .expect("COUNT should work");
    assert_eq!(count, 1, "only the post-reopen row exists, got {}", count);
}

#[test]
fn test_wal_write_fail_mid_transaction_atomicity() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display()))
        .expect("Failed to open database");

    db.execute(
        "CREATE TABLE fp_wal_tx (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("CREATE should succeed");

    // Insert initial data
    db.execute("INSERT INTO fp_wal_tx VALUES (1, 100)", ())
        .expect("Initial insert should succeed");

    // Start a transaction with multiple operations
    db.execute("BEGIN", ()).expect("BEGIN should succeed");
    db.execute("UPDATE fp_wal_tx SET val = 200 WHERE id = 1", ())
        .expect("UPDATE should succeed within transaction");

    // Arm failpoint before commit
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);

    // Commit should fail
    let result = db.execute("COMMIT", ());
    // If COMMIT fails, the transaction should be rolled back
    if result.is_err() {
        let _ = db.execute("ROLLBACK", ());
    }

    // Disarm
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);

    // Value should still be original (transaction failed)
    let val: i64 = db
        .query_one("SELECT val FROM fp_wal_tx WHERE id = 1", ())
        .expect("SELECT should work");
    assert_eq!(val, 100, "Value should be unchanged after failed commit");
}

#[test]
fn test_wal_write_fail_recovery_after_disarm() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_wal_rec (id INTEGER PRIMARY KEY, val TEXT)",
            (),
        )
        .expect("CREATE should succeed");
        db.execute("INSERT INTO fp_wal_rec VALUES (1, 'before')", ())
            .expect("INSERT should succeed");
    }

    // Reopen and verify data persisted
    {
        let db = Database::open(&path).expect("Failed to reopen database");
        let val: String = db
            .query_one("SELECT val FROM fp_wal_rec WHERE id = 1", ())
            .expect("SELECT should work");
        assert_eq!(val, "before");

        // Arm failpoint, try to write, fail
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        let _ = db.execute("INSERT INTO fp_wal_rec VALUES (2, 'during_fail')", ());
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);

        // Poisoned until reopen: no write may reuse the stale buffer.
        let result = db.execute("INSERT INTO fp_wal_rec VALUES (3, 'after_fail')", ());
        assert!(
            result.is_err(),
            "writes must fail on a poisoned WAL until reopen"
        );
        let _ = db.close();
    }

    // Reopen after the poisoned handle: writes work again.
    {
        let db = Database::open(&path).expect("reopen after poisoned WAL");
        db.execute("INSERT INTO fp_wal_rec VALUES (3, 'after_reopen')", ())
            .expect("INSERT should succeed after reopen");
    }

    // Reopen and verify consistency
    {
        let db = Database::open(&path).expect("Failed to reopen after failpoint");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_wal_rec", ())
            .expect("COUNT should work");
        // Row 1 (committed), row 2 (may or may not exist - failed write), row 3 (committed)
        assert!(
            count >= 2,
            "At least rows 1 and 3 should exist, got {}",
            count
        );
    }
}

// ============================================================================
// WAL Sync Failpoint Tests
// ============================================================================

#[test]
fn test_wal_sync_fail_returns_error() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display()))
        .expect("Failed to open database");

    db.execute(
        "CREATE TABLE fp_sync (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("CREATE should succeed");

    // Arm sync failpoint
    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);

    // Operations that require sync should fail
    let result = db.execute("INSERT INTO fp_sync VALUES (1, 'test')", ());
    // Sync failures may or may not propagate depending on when sync is called
    // The key invariant is that the database doesn't corrupt

    // Disarm
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);

    if result.is_err() {
        // A failed commit fsync poisons the WAL (the written marker's
        // durability is unknowable); the handle must fail until reopen.
        let retry = db.execute("INSERT INTO fp_sync VALUES (2, 'after')", ());
        assert!(
            retry.is_err(),
            "writes must fail on a poisoned WAL until reopen"
        );
        let dsn = format!("file://{}", dir.path().display());
        let _ = db.close();
        let db = Database::open(&dsn).expect("reopen after poisoned WAL");
        db.execute("INSERT INTO fp_sync VALUES (2, 'after_reopen')", ())
            .expect("INSERT should succeed after reopen");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_sync", ())
            .expect("COUNT should work after reopen");
        assert!(count >= 1, "At least one row should exist");
    } else {
        // Sync not exercised on this path; the handle stays usable.
        db.execute("INSERT INTO fp_sync VALUES (2, 'after')", ())
            .expect("INSERT should succeed");
    }
}

#[test]
fn test_sync_fail_commit_not_durable_after_close() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}?sync_mode=full", dir.path().display());

    {
        let db = Database::open(&path).expect("open");
        db.execute(
            "CREATE TABLE fp_sync_close (id INTEGER PRIMARY KEY, val TEXT)",
            (),
        )
        .expect("CREATE should succeed");
        // Arm only the sync failpoint: the commit marker gets WRITTEN to
        // the fd, then its fsync fails and the commit is aborted.
        test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
        let result = db.execute("INSERT INTO fp_sync_close VALUES (1, 'x')", ());
        test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
        assert!(result.is_err(), "commit must fail when its fsync fails");
        let _ = db.close();
    }

    // Neither close() nor kernel writeback may persist the aborted
    // commit's marker; reopen must not resurrect the row.
    let db = Database::open(&path).expect("reopen");
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_sync_close", ())
        .expect("COUNT should work");
    assert_eq!(count, 0, "aborted commit resurrected after close+reopen");
}

// ============================================================================
// Snapshot Write Failpoint Tests
// ============================================================================

#[test]
fn test_snapshot_write_fail_during_checkpoint() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    let db = Database::open(&path).expect("Failed to open database");

    db.execute(
        "CREATE TABLE fp_snap (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("CREATE should succeed");

    for i in 0..10 {
        db.execute(
            &format!("INSERT INTO fp_snap VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("INSERT should succeed");
    }

    // Arm snapshot write failpoint
    test_failpoints::SNAPSHOT_WRITE_FAIL.store(true, Ordering::Release);

    // Trigger snapshot via VACUUM (which calls create_snapshot internally)
    let vacuum_result = db.execute("VACUUM", ());
    // VACUUM may or may not fail - depends on whether snapshot write is part of it

    // Disarm
    test_failpoints::SNAPSHOT_WRITE_FAIL.store(false, Ordering::Release);

    // Data should still be accessible (WAL has the data even if snapshot failed)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_snap", ())
        .expect("COUNT should work");
    assert_eq!(count, 10, "All 10 rows should be accessible");

    // Sum should be correct
    let sum: f64 = db
        .query_one("SELECT SUM(val) FROM fp_snap", ())
        .expect("SUM should work");
    assert_eq!(sum, 450.0, "Sum should be 0+10+20+...+90 = 450");

    drop(vacuum_result);
}

#[test]
fn test_snapshot_write_fail_recovery_on_reopen() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_snap_rec (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .expect("CREATE should succeed");

        for i in 0..5 {
            db.execute(
                &format!("INSERT INTO fp_snap_rec VALUES ({}, {})", i, i),
                (),
            )
            .expect("INSERT should succeed");
        }

        // Arm and trigger failed snapshot
        test_failpoints::SNAPSHOT_WRITE_FAIL.store(true, Ordering::Release);
        let _ = db.execute("VACUUM", ());
        test_failpoints::SNAPSHOT_WRITE_FAIL.store(false, Ordering::Release);
    }

    // Reopen - should recover from WAL
    {
        let db = Database::open(&path).expect("Recovery should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_snap_rec", ())
            .expect("COUNT should work after recovery");
        assert_eq!(count, 5, "All 5 rows should be recovered from WAL");
    }
}

// ============================================================================
// Snapshot Sync Failpoint Tests
// ============================================================================

#[test]
fn test_snapshot_sync_fail_during_finalize() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    let db = Database::open(&path).expect("Failed to open database");

    db.execute(
        "CREATE TABLE fp_ssync (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("CREATE should succeed");

    for i in 0..5 {
        db.execute(
            &format!("INSERT INTO fp_ssync VALUES ({}, 'row_{}')", i, i),
            (),
        )
        .expect("INSERT should succeed");
    }

    // Arm snapshot sync failpoint
    test_failpoints::SNAPSHOT_SYNC_FAIL.store(true, Ordering::Release);
    let _ = db.execute("VACUUM", ());
    test_failpoints::SNAPSHOT_SYNC_FAIL.store(false, Ordering::Release);

    // Data should still be accessible
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_ssync", ())
        .expect("COUNT should work");
    assert_eq!(count, 5);
}

// ============================================================================
// Snapshot Rename Failpoint Tests
// ============================================================================

#[test]
fn test_snapshot_rename_fail_atomicity() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_rename (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .expect("CREATE should succeed");

        for i in 0..10 {
            db.execute(
                &format!("INSERT INTO fp_rename VALUES ({}, {})", i, i * 100),
                (),
            )
            .expect("INSERT should succeed");
        }

        // Arm rename failpoint
        test_failpoints::SNAPSHOT_RENAME_FAIL.store(true, Ordering::Release);
        let _ = db.execute("VACUUM", ());
        test_failpoints::SNAPSHOT_RENAME_FAIL.store(false, Ordering::Release);

        // Data should still be correct
        let sum: f64 = db
            .query_one("SELECT SUM(val) FROM fp_rename", ())
            .expect("SUM should work");
        assert_eq!(sum, 4500.0);
    }

    // Reopen and verify
    {
        let db = Database::open(&path).expect("Recovery should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_rename", ())
            .expect("COUNT should work");
        assert_eq!(count, 10, "All rows should survive failed snapshot rename");
    }
}

// ============================================================================
// Checkpoint Write Failpoint Tests
// ============================================================================

#[test]
fn test_checkpoint_write_fail() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_ckpt (id INTEGER PRIMARY KEY, val TEXT)",
            (),
        )
        .expect("CREATE should succeed");

        db.execute("INSERT INTO fp_ckpt VALUES (1, 'first')", ())
            .expect("INSERT should succeed");

        // Arm checkpoint write failpoint
        test_failpoints::CHECKPOINT_WRITE_FAIL.store(true, Ordering::Release);

        // More writes (checkpoint is triggered during snapshot creation)
        for i in 2..=5 {
            let _ = db.execute(
                &format!("INSERT INTO fp_ckpt VALUES ({}, 'row_{}')", i, i),
                (),
            );
        }

        test_failpoints::CHECKPOINT_WRITE_FAIL.store(false, Ordering::Release);
    }

    // Reopen - WAL replay should recover everything
    {
        let db = Database::open(&path).expect("Recovery should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_ckpt", ())
            .expect("COUNT should work");
        assert!(
            count >= 1,
            "At least the first row should exist, got {}",
            count
        );
    }
}

// ============================================================================
// Combined failpoint scenarios
// ============================================================================

#[test]
fn test_multiple_failpoints_sequential() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    let db = Database::open(&path).expect("Failed to open database");
    db.execute(
        "CREATE TABLE fp_multi (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("CREATE should succeed");

    // Phase 1: WAL write failure poisons the handle; reopen to continue.
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    let _ = db.execute("INSERT INTO fp_multi VALUES (1, 100)", ());
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    let _ = db.close();
    let db = Database::open(&path).expect("reopen after poisoned WAL");

    // Phase 2: Normal operation
    db.execute("INSERT INTO fp_multi VALUES (2, 200)", ())
        .expect("Should succeed after reopen");

    // Phase 3: Snapshot failure
    test_failpoints::SNAPSHOT_WRITE_FAIL.store(true, Ordering::Release);
    let _ = db.execute("VACUUM", ());
    test_failpoints::SNAPSHOT_WRITE_FAIL.store(false, Ordering::Release);

    // Phase 4: Normal operation again
    db.execute("INSERT INTO fp_multi VALUES (3, 300)", ())
        .expect("Should succeed after disarming snapshot failpoint");

    // Verify consistency
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_multi", ())
        .expect("COUNT should work");
    assert!(
        count >= 2,
        "At least rows 2 and 3 should exist, got {}",
        count
    );
}

#[test]
fn test_failpoint_does_not_corrupt_existing_data() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    // Phase 1: Populate database
    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_preserve (id INTEGER PRIMARY KEY, val TEXT NOT NULL)",
            (),
        )
        .expect("CREATE should succeed");

        for i in 0..20 {
            db.execute(
                &format!("INSERT INTO fp_preserve VALUES ({}, 'data_{}')", i, i),
                (),
            )
            .expect("INSERT should succeed");
        }
    }

    // Phase 2: Arm various failpoints and try operations
    {
        let db = Database::open(&path).expect("Reopen should succeed");

        // Verify initial data
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_preserve", ())
            .expect("COUNT should work");
        assert_eq!(count, 20);

        // Try all failpoints in sequence
        let failpoints: &[&AtomicBool] = &[
            &test_failpoints::WAL_WRITE_FAIL,
            &test_failpoints::WAL_SYNC_FAIL,
            &test_failpoints::SNAPSHOT_WRITE_FAIL,
            &test_failpoints::SNAPSHOT_SYNC_FAIL,
            &test_failpoints::SNAPSHOT_RENAME_FAIL,
            &test_failpoints::CHECKPOINT_WRITE_FAIL,
        ];

        for fp in failpoints {
            fp.store(true, Ordering::Release);
            // Try some operation - may or may not fail
            let _ = db.execute("INSERT INTO fp_preserve VALUES (999, 'fail')", ());
            let _ = db.execute("DELETE FROM fp_preserve WHERE id = 999", ());
            let _ = db.execute("VACUUM", ());
            fp.store(false, Ordering::Release);
        }

        // Original data should be intact
        let count_after: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_preserve WHERE id < 20", ())
            .expect("COUNT should work");
        assert_eq!(count_after, 20, "Original 20 rows should be preserved");
    }

    // Phase 3: Reopen and verify
    {
        let db = Database::open(&path).expect("Final reopen should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_preserve WHERE id < 20", ())
            .expect("COUNT should work");
        assert_eq!(
            count, 20,
            "All original rows should survive failpoint storm"
        );
    }
}

/// A commit that fails at the WAL after its index updates were applied
/// takes those updates back, so the indexes describe the rows that stayed
/// visible and an index-ordered read agrees with the full sort
#[test]
fn test_wal_sync_failure_undoes_the_commit_index_updates() {
    let _guard = failpoint_guard();
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_interval=3600",
        dir.path().display()
    ))
    .expect("open");
    db.execute(
        "CREATE TABLE fp_topk (id INTEGER PRIMARY KEY, k TEXT NOT NULL, t INTEGER NOT NULL, UNIQUE(k, t))",
        (),
    )
    .expect("create");
    db.execute("INSERT INTO fp_topk VALUES (1, 'a', 100), (2, 'a', 50)", ())
        .expect("insert");

    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let update = db.execute("UPDATE fp_topk SET t = 0 WHERE id = 1", ());
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(
        update.is_err(),
        "the UPDATE must abort on the WAL sync failure"
    );

    let ids = |sql: &str| -> Vec<i64> {
        db.query(sql, ())
            .expect("query")
            .map(|r| r.expect("row").get::<i64>(0).expect("id"))
            .collect()
    };
    assert_eq!(
        ids("SELECT id FROM fp_topk WHERE k = 'a' ORDER BY t DESC, id DESC LIMIT 1"),
        vec![1]
    );
    assert_eq!(
        ids("SELECT id FROM fp_topk WHERE k = 'a' ORDER BY t DESC LIMIT 1"),
        vec![1]
    );
    // The old key is back in the index: an equality on it finds the row
    assert_eq!(
        ids("SELECT id FROM fp_topk WHERE k = 'a' AND t = 100"),
        vec![1]
    );
    assert!(ids("SELECT id FROM fp_topk WHERE k = 'a' AND t = 0").is_empty());
}

/// A DELETE aborted at the WAL leaves its delete mark on the row with the
/// aborted transaction's id; the row stays visible, the index gets its key
/// back, and every read path, the index-ordered one included, keeps the row
#[test]
fn test_wal_sync_failure_on_a_delete_keeps_the_row_everywhere() {
    let _guard = failpoint_guard();
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_interval=3600",
        dir.path().display()
    ))
    .expect("open");
    db.execute(
        "CREATE TABLE fp_del (id INTEGER PRIMARY KEY, k TEXT NOT NULL, t INTEGER NOT NULL, UNIQUE(k, t))",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX fp_del_t ON fp_del(t) USING BTREE", ())
        .expect("index");
    db.execute("INSERT INTO fp_del VALUES (1, 'a', 100), (2, 'a', 50)", ())
        .expect("insert");
    let ids = |sql: &str| -> Vec<i64> {
        db.query(sql, ())
            .expect("query")
            .map(|r| r.expect("row").get::<i64>(0).expect("id"))
            .collect()
    };

    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let delete = db.execute("DELETE FROM fp_del WHERE id = 1", ());
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(
        delete.is_err(),
        "the DELETE must abort on the WAL sync failure"
    );

    assert_eq!(
        ids("SELECT id FROM fp_del WHERE k = 'a' ORDER BY t DESC, id DESC LIMIT 3"),
        vec![1, 2]
    );
    assert_eq!(
        ids("SELECT id FROM fp_del WHERE k = 'a' ORDER BY t DESC LIMIT 3"),
        vec![1, 2]
    );
    assert_eq!(
        ids("SELECT id FROM fp_del WHERE k = 'a' AND t = 100"),
        vec![1]
    );
    assert_eq!(ids("SELECT id FROM fp_del WHERE t = 100"), vec![1]);

    // An aborted INSERT leaves no key behind either
    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let insert = db.execute("INSERT INTO fp_del VALUES (3, 'a', 150)", ());
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(insert.is_err());
    assert_eq!(
        ids("SELECT id FROM fp_del WHERE k = 'a' ORDER BY t DESC LIMIT 3"),
        vec![1, 2]
    );
    assert!(ids("SELECT id FROM fp_del WHERE k = 'a' AND t = 150").is_empty());
    assert!(ids("SELECT id FROM fp_del WHERE t = 150").is_empty());
}

// ============================================================================
// Checkpoint entry: the cut, the catalog copies and the recovery boundary
// ============================================================================

fn checkpoint_boundary(dir: &std::path::Path) -> u64 {
    stoolap::storage::mvcc::wal_manager::CheckpointMetadata::read_from_file(
        &dir.join("wal").join("checkpoint.meta"),
    )
    .expect("checkpoint.meta")
    .lsn
}

fn index_names(db: &Database, table: &str) -> Vec<String> {
    db.query(&format!("SHOW INDEXES FROM {}", table), ())
        .expect("SHOW INDEXES")
        .map(|row| row.unwrap().get(1).unwrap())
        .collect()
}

fn catalog(db: &Database, table: &str) {
    db.execute(
        &format!(
            "CREATE TABLE {} (id INTEGER PRIMARY KEY, k TEXT, c INTEGER, UNIQUE(k))",
            table
        ),
        (),
    )
    .expect("CREATE TABLE");
    db.execute(&format!("CREATE INDEX {}_c ON {}(c)", table, table), ())
        .expect("CREATE INDEX");
    db.execute(
        &format!("CREATE VIEW {}_v AS SELECT id, k FROM {}", table, table),
        (),
    )
    .expect("CREATE VIEW");
}

fn assert_catalog_restored(db: &Database, table: &str, rows: i64) {
    let count: i64 = db
        .query_one(&format!("SELECT COUNT(*) FROM {}", table), ())
        .expect("COUNT");
    assert_eq!(count, rows, "rows of {}", table);
    let through_view: i64 = db
        .query_one(&format!("SELECT COUNT(*) FROM {}_v", table), ())
        .expect("the view is back");
    assert_eq!(through_view, rows);
    let names = index_names(db, table);
    assert!(
        names.iter().any(|n| n == &format!("{}_c", table)),
        "the ordinary index is back: {:?}",
        names
    );
    assert!(
        db.execute(&format!("INSERT INTO {} VALUES (1000, 'k1', 1)", table), ())
            .is_err(),
        "the unique index is back"
    );
}

#[test]
fn a_commit_lands_while_the_checkpoint_syncs_its_wal() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).expect("open");
    db.execute("CREATE TABLE fp_cut (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO fp_cut VALUES (1, 'before')", ())
        .unwrap();

    // The checkpoint's WAL sync runs on this thread; the hook commits a
    // row from another thread and waits for it, bounded: a commit held
    // behind the sync by the fence would time out here
    let landed = std::sync::Arc::new(AtomicBool::new(false));
    let seen = std::sync::Arc::clone(&landed);
    let writer = db.clone();
    test_failpoints::before_wal_sync(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = writer.execute("INSERT INTO fp_cut VALUES (2, 'during')", ());
            let _ = tx.send(result.is_ok());
        });
        if let Ok(true) = rx.recv_timeout(std::time::Duration::from_secs(3)) {
            seen.store(true, Ordering::Release);
        }
    });
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");
    assert!(
        landed.load(Ordering::Acquire),
        "a commit must not wait behind the checkpoint's WAL sync"
    );

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM fp_cut", ()).unwrap();
    assert_eq!(count, 2, "the row committed after the cut survives");
}

#[test]
fn a_failed_catalog_copy_leaves_the_recovery_boundary_where_it_was() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    // Full mode: every commit is synced, so the poison that follows the
    // failed write cuts nothing acknowledged
    let dsn = format!("file://{}?sync_mode=full", dir.path().display());
    let db = Database::open(&dsn).expect("open");
    catalog(&db, "fp_bound");
    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO fp_bound VALUES ({}, 'k{}', {})", i, i, i),
            (),
        )
        .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ())
        .expect("first checkpoint");
    let boundary = checkpoint_boundary(dir.path());
    assert!(boundary > 0);

    for i in 4..=6 {
        db.execute(
            &format!("INSERT INTO fp_bound VALUES ({}, 'k{}', {})", i, i, i),
            (),
        )
        .unwrap();
    }
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    let _ = db.execute("PRAGMA CHECKPOINT", ());
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    assert_eq!(
        checkpoint_boundary(dir.path()),
        boundary,
        "the boundary must not move past catalog copies that never became durable"
    );

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    assert_catalog_restored(&db, "fp_bound", 6);
}

#[test]
fn the_catalog_copies_are_durable_when_the_truncation_returns_early() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    // Every record rotates the WAL, so the cut is the file's own LSN and
    // the truncation returns before rewriting anything
    let dsn = format!("file://{}?wal_max_size=1", dir.path().display());
    let db = Database::open(&dsn).expect("open");
    catalog(&db, "fp_rot");
    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO fp_rot VALUES ({}, 'k{}', {})", i, i, i),
            (),
        )
        .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");

    // A failed write poisons the WAL and cuts the file back to its synced
    // length: whatever the checkpoint left unsynced is gone, as in a crash
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    assert!(db
        .execute("INSERT INTO fp_rot VALUES (4, 'k4', 4)", ())
        .is_err());
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    assert_catalog_restored(&db, "fp_rot", 3);
}

#[test]
fn a_write_failing_before_the_checkpoint_sync_fails_the_sync() {
    use stoolap::storage::mvcc::persistence::DDL_TXN_ID;
    use stoolap::storage::mvcc::wal_manager::{WALEntry, WALManager, WALOperationType};
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let wal =
        std::sync::Arc::new(WALManager::new(dir.path().join("wal"), SyncMode::Full).expect("wal"));
    wal.append_entry(WALEntry::new(
        1,
        "t".to_string(),
        1,
        WALOperationType::Insert,
        vec![],
    ))
    .unwrap();
    wal.write_commit_marker(1).unwrap();
    wal.append_catalog_entry(WALEntry::new(
        DDL_TXN_ID,
        "t".to_string(),
        0,
        WALOperationType::CreateTable,
        vec![1, 2, 3],
    ))
    .unwrap();

    // A commit fails right before the sync takes the lock: the WAL is
    // poisoned and cut back to its synced length, the copy with it
    let poisoner = std::sync::Arc::clone(&wal);
    test_failpoints::before_wal_sync(move || {
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        assert!(poisoner.write_commit_marker(2).is_err());
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    });
    assert!(
        wal.sync_for_checkpoint().is_err(),
        "the sync must not report the cut copy durable"
    );
}
