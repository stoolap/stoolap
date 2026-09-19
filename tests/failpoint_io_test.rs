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

#[test]
fn a_failed_add_column_restores_the_readable_schema() {
    assert_failed_column_ddl_restores_schema("ALTER TABLE t ADD COLUMN x INTEGER DEFAULT 4");
}

#[test]
fn a_failed_drop_column_restores_the_readable_schema() {
    assert_failed_column_ddl_restores_schema("ALTER TABLE t DROP COLUMN a");
}

#[test]
fn a_failed_rename_column_restores_the_readable_schema() {
    assert_failed_column_ddl_restores_schema("ALTER TABLE t RENAME COLUMN a TO z");
}

#[test]
fn a_failed_modify_column_restores_the_readable_schema() {
    assert_failed_column_ddl_restores_schema("ALTER TABLE t MODIFY COLUMN b TEXT");
}

fn assert_failed_column_ddl_restores_schema(ddl: &str) {
    let _guard = failpoint_guard();
    for failure in [
        &test_failpoints::WAL_WRITE_FAIL,
        &test_failpoints::WAL_SYNC_FAIL,
    ] {
        for sealed in [false, true] {
            let dir = tempdir().unwrap();
            let dsn = format!(
                "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
                dir.path().display()
            );
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
                (),
            )
            .unwrap();
            db.execute("INSERT INTO t VALUES (1,10,20)", ()).unwrap();
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.execute("INSERT INTO t VALUES (2,11,21)", ()).unwrap();
            let columns = db
                .engine()
                .get_version_store("t")
                .unwrap()
                .schema()
                .columns
                .clone();
            failure.store(true, Ordering::Release);
            let altered = db.execute(ddl, ());
            failure.store(false, Ordering::Release);
            assert!(altered.is_err(), "{ddl}, sealed={sealed}");
            let read = |db: &Database| {
                assert_eq!(
                    db.engine().get_version_store("t").unwrap().schema().columns,
                    columns,
                    "{ddl}, sealed={sealed}"
                );
                db.query("SELECT * FROM t ORDER BY id", ())
                    .unwrap()
                    .map(|row| {
                        let row = row.unwrap();
                        assert_eq!(row.len(), 3, "{ddl}, sealed={sealed}");
                        (
                            row.get::<i64>(0).unwrap(),
                            row.get::<i64>(1).unwrap(),
                            row.get::<i64>(2).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(read(&db), vec![(1, 10, 20), (2, 11, 21)]);
            let error = db
                .execute("ALTER TABLE t ADD COLUMN a INTEGER", ())
                .unwrap_err();
            assert!(matches!(error, stoolap::Error::DuplicateColumn), "{error}");
            assert!(db.execute("INSERT INTO t VALUES (3,12,22)", ()).is_err());
            let _ = db.close();
            let db = Database::open(&dsn).unwrap();
            assert_eq!(read(&db), vec![(1, 10, 20), (2, 11, 21)]);
        }
    }
}

#[test]
fn maintenance_uses_the_restored_schema_after_a_failed_column_drop() {
    let _guard = failpoint_guard();
    for sealed in [false, true] {
        let dir = tempdir().unwrap();
        let dsn = format!(
            "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
            dir.path().display()
        );
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
            (),
        )
        .unwrap();
        for id in 1..=3 {
            db.execute("INSERT INTO t VALUES (?,10,20)", (id,)).unwrap();
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
        }
        assert_eq!(db.engine().volume_stats().len(), if sealed { 3 } else { 0 });
        db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        let altered = db.execute("ALTER TABLE t DROP COLUMN a", ());
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
        assert!(altered.is_err());
        let _ = db.execute("PRAGMA CHECKPOINT", ());
        let read = |db: &Database| {
            db.query("SELECT id, a, b FROM t ORDER BY id", ())
                .unwrap()
                .map(|row| {
                    let row = row.unwrap();
                    (
                        row.get::<i64>(0).unwrap(),
                        row.get::<i64>(1).unwrap(),
                        row.get::<i64>(2).unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(read(&db), vec![(1, 10, 20), (2, 10, 20), (3, 10, 20)]);
        let _ = db.close();
        let db = Database::open(&dsn).unwrap();
        assert_eq!(read(&db), vec![(1, 10, 20), (2, 10, 20), (3, 10, 20)]);
    }
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

// ============================================================================
// WAL truncation by rotation: the swap and what races it
// ============================================================================

fn wal_file_sizes(dir: &std::path::Path) -> Vec<(String, u64)> {
    let mut files: Vec<(String, u64)> = std::fs::read_dir(dir.join("wal"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log")
        })
        .map(|e| {
            // Through a handle: a directory listing reports a stale length
            // for a file that is open for writing on Windows
            let len = std::fs::File::open(e.path())
                .unwrap()
                .metadata()
                .unwrap()
                .len();
            (e.file_name().to_string_lossy().to_string(), len)
        })
        .collect();
    files.sort();
    files
}

/// Commit `sql` from another thread while the hook's thread is inside the
/// truncation, with a bounded wait; true when the commit went through.
fn commit_from_another_thread(db: &Database, sql: &'static str) -> bool {
    let writer = db.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(writer.execute(sql, ()).is_ok());
    });
    matches!(rx.recv_timeout(std::time::Duration::from_secs(3)), Ok(true))
}

#[test]
fn rows_committed_while_the_truncation_prepares_its_file_survive() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).expect("open");
    db.execute("CREATE TABLE fp_swap (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO fp_swap VALUES (1, 'before')", ())
        .unwrap();

    // Between the new file's creation and the swap, two rows commit: they
    // land in the old file above the boundary and nothing copies them
    let landed = std::sync::Arc::new(AtomicBool::new(false));
    let seen = std::sync::Arc::clone(&landed);
    let writer = db.clone();
    test_failpoints::before_wal_swap(move || {
        let a = commit_from_another_thread(&writer, "INSERT INTO fp_swap VALUES (2, 'during')");
        let b = commit_from_another_thread(&writer, "INSERT INTO fp_swap VALUES (3, 'during')");
        seen.store(a && b, Ordering::Release);
    });
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");
    assert!(
        landed.load(Ordering::Acquire),
        "the commits must not wait on the truncation"
    );
    assert_eq!(wal_file_sizes(dir.path()).len(), 2);

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM fp_swap", ()).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn a_rotation_racing_the_truncation_leaves_no_third_file() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    // Every commit rotates, so the commit inside the hook starts a file of
    // its own and the truncation's prepared file must go
    let dsn = format!(
        "file://{}?wal_max_size=1&checkpoint_on_close=off",
        dir.path().display()
    );
    let db = Database::open(&dsn).expect("open");
    db.execute("CREATE TABLE fp_race (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO fp_race VALUES (1, 'a')", ())
        .unwrap();

    let writer = db.clone();
    let wal_dir = dir.path().to_path_buf();
    let rotated = std::sync::Arc::new(AtomicBool::new(false));
    let seen = std::sync::Arc::clone(&rotated);
    test_failpoints::before_wal_swap(move || {
        let before = wal_file_sizes(&wal_dir).len();
        let ok = commit_from_another_thread(&writer, "INSERT INTO fp_race VALUES (2, 'b')");
        let after = wal_file_sizes(&wal_dir).len();
        seen.store(ok && after > before, Ordering::Release);
    });
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");
    assert!(
        rotated.load(Ordering::Acquire),
        "the commit inside the hook rotated"
    );

    let files = wal_file_sizes(dir.path());
    let empty = files.iter().filter(|f| f.1 == 0).count();
    assert!(empty <= 1, "at most the current file is empty: {:?}", files);

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM fp_race", ()).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn a_write_failing_before_the_swap_keeps_the_current_file() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off",
        dir.path().display()
    );
    let db = Database::open(&dsn).expect("open");
    catalog(&db, "fp_noswap");
    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO fp_noswap VALUES ({}, 'k{}', {})", i, i, i),
            (),
        )
        .unwrap();
    }
    let before = wal_file_sizes(dir.path());
    assert_eq!(before.len(), 1);

    // A commit fails while the new file waits: the WAL is poisoned, the
    // truncation must not swap, and the prepared file must not stay
    let writer = db.clone();
    test_failpoints::before_wal_swap(move || {
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        assert!(!commit_from_another_thread(
            &writer,
            "INSERT INTO fp_noswap VALUES (4, 'k4', 4)"
        ));
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    });
    let _ = db.execute("PRAGMA CHECKPOINT", ());
    let after = wal_file_sizes(dir.path());
    assert_eq!(
        after.iter().map(|f| &f.0).collect::<Vec<_>>(),
        before.iter().map(|f| &f.0).collect::<Vec<_>>(),
        "no swap and no leftover: {:?}",
        after
    );

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    assert_catalog_restored(&db, "fp_noswap", 3);
}

#[test]
fn a_commit_synced_before_the_swap_survives_a_poison_of_the_new_file() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off",
        dir.path().display()
    );
    let db = Database::open(&dsn).expect("open");
    db.execute("CREATE TABLE fp_tail (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO fp_tail VALUES (1, 'a')", ())
        .unwrap();

    let writer = db.clone();
    let landed = std::sync::Arc::new(AtomicBool::new(false));
    let seen = std::sync::Arc::clone(&landed);
    test_failpoints::before_wal_swap(move || {
        seen.store(
            commit_from_another_thread(&writer, "INSERT INTO fp_tail VALUES (2, 'tail')"),
            Ordering::Release,
        );
    });
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");
    assert!(landed.load(Ordering::Acquire));

    // The new file is cut back to its synced length by the poison; the
    // old file, holding the row committed before the swap, is untouched
    test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    assert!(db
        .execute("INSERT INTO fp_tail VALUES (3, 'lost')", ())
        .is_err());
    test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);

    let _ = db.close();
    let db = Database::open(&dsn).expect("reopen");
    let ids: Vec<i64> = db
        .query("SELECT id FROM fp_tail ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

// ============================================================================
// A file start's prepared file and the retired file's durability debt
// ============================================================================

fn wal_insert(wal: &stoolap::storage::mvcc::wal_manager::WALManager, id: i64) {
    use stoolap::storage::mvcc::wal_manager::{WALEntry, WALOperationType};
    wal.append_entry(WALEntry::new(
        id,
        "t".to_string(),
        id,
        WALOperationType::Insert,
        vec![1, 2, 3],
    ))
    .unwrap();
}

fn replayed_inserts(dir: &std::path::Path, from: u64) -> Vec<i64> {
    use stoolap::storage::mvcc::wal_manager::{WALManager, WALOperationType};
    use stoolap::storage::SyncMode;
    let reopened = WALManager::new(dir, SyncMode::Full).unwrap();
    let mut rows = Vec::new();
    reopened
        .replay_two_phase(from, |entry| {
            if entry.operation == WALOperationType::Insert {
                rows.push(entry.row_id);
            }
            Ok(())
        })
        .unwrap();
    rows
}

#[test]
fn a_prepared_file_is_no_boundary_for_the_cleanup_of_a_live_file() {
    use stoolap::storage::config::PersistenceConfig;
    use stoolap::storage::mvcc::wal_manager::WALManager;
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let config = PersistenceConfig {
        wal_max_size: 1,
        ..PersistenceConfig::default()
    };
    let wal = std::sync::Arc::new(
        WALManager::with_config(dir.path(), SyncMode::Full, Some(&config)).unwrap(),
    );
    wal_insert(&wal, 1);
    let cut = wal.write_commit_marker(1).unwrap();
    let old_name = wal.current_wal_file();
    let old_path = dir.path().join(&old_name);

    // While a rotation holds its prepared file, a commit lands above the
    // cut in the old file, another rotation wins, and a truncation runs
    // its cleanup: the prepared file must not pass for the old file's
    // upper bound
    let competing = std::sync::Arc::clone(&wal);
    test_failpoints::before_wal_swap(move || {
        wal_insert(&competing, 2);
        assert!(competing.write_commit_marker(2).unwrap() > cut);
        assert!(competing.maybe_rotate().unwrap());
        competing.truncate_wal(cut).unwrap();
        assert!(
            old_path.exists(),
            "the live old file must survive the cleanup"
        );
    });
    wal.maybe_rotate().unwrap();
    wal.close().unwrap();
    assert_eq!(replayed_inserts(dir.path(), cut), vec![2]);
}

#[test]
fn a_full_commit_into_the_new_file_waits_for_the_retired_file_to_be_durable() {
    use stoolap::storage::mvcc::wal_manager::WALManager;
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let wal = std::sync::Arc::new(WALManager::new(dir.path(), SyncMode::Full).unwrap());
    wal_insert(&wal, 1);
    let cut = wal.write_commit_marker(1).unwrap();
    // Buffered, then drained into the old file by the swap
    wal_insert(&wal, 2);

    // Right after the swap, before the retired file is settled, the
    // transaction's commit marker goes into the new file: its sync must
    // settle the retired file first, and here that settlement fails
    let committer = std::sync::Arc::clone(&wal);
    let acknowledged = std::sync::Arc::new(AtomicBool::new(true));
    let seen = std::sync::Arc::clone(&acknowledged);
    test_failpoints::after_wal_swap(move || {
        test_failpoints::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::Release);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(committer.write_commit_marker(2).is_ok());
        });
        let ok = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .expect("the commit must return");
        test_failpoints::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::Release);
        seen.store(ok, Ordering::Release);
    });
    let _ = wal.truncate_wal(cut);
    assert!(
        !acknowledged.load(Ordering::Acquire),
        "a commit must not be acknowledged while its records' file is not durable"
    );
    assert!(wal.write_commit_marker(3).is_err(), "the WAL is poisoned");
    drop(wal);
    assert_eq!(replayed_inserts(dir.path(), 0), vec![1]);
}

#[test]
fn a_full_commit_waiting_on_another_thread_s_settlement_inherits_its_failure() {
    use stoolap::storage::mvcc::wal_manager::WALManager;
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let wal = std::sync::Arc::new(WALManager::new(dir.path(), SyncMode::Full).unwrap());
    wal_insert(&wal, 1);
    let cut = wal.write_commit_marker(1).unwrap();
    wal_insert(&wal, 2);

    // The truncation's thread holds the debt; a commit drains its marker
    // into the new file and waits for the settlement, which then fails
    let committer = std::sync::Arc::clone(&wal);
    let (handle_tx, handle_rx) = std::sync::mpsc::channel();
    test_failpoints::before_retired_settle(move || {
        let (waiting_tx, waiting_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            test_failpoints::before_retired_wait(move || waiting_tx.send(()).unwrap());
            committer.write_commit_marker(2)
        });
        handle_tx.send(handle).unwrap();
        waiting_rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .expect("the commit must reach the wait");
        test_failpoints::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::Release);
    });
    let truncated = wal.truncate_wal(cut);
    let committed = handle_rx.recv().unwrap().join().unwrap();
    test_failpoints::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(truncated.is_err());
    assert!(
        committed.is_err(),
        "a commit that waited on the failed settlement must fail too"
    );
    assert!(wal.write_commit_marker(3).is_err(), "the WAL is poisoned");
}

#[test]
fn a_full_commit_refused_after_a_failed_settlement_does_not_replay() {
    use stoolap::storage::mvcc::wal_manager::WALManager;
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let wal = std::sync::Arc::new(WALManager::new(dir.path(), SyncMode::Full).unwrap());
    wal_insert(&wal, 1);
    let cut = wal.write_commit_marker(1).unwrap();
    wal_insert(&wal, 2);

    // The commit's marker is buffered; before its sync a truncation drains
    // it and the records into the old file, and the settlement fails: the
    // commit is refused, so the marker must not persist in the old file
    let truncator = std::sync::Arc::clone(&wal);
    test_failpoints::before_wal_sync(move || {
        test_failpoints::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::Release);
        let truncated = truncator.truncate_wal(cut);
        test_failpoints::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::Release);
        assert!(truncated.is_err());
    });
    assert!(wal.write_commit_marker(2).is_err());
    drop(wal);
    assert_eq!(replayed_inserts(dir.path(), 0), vec![1]);
}

/// The new name's directory entry is owed by every file start, whether or
/// not the old file has an unsynced tail: the directory is synced before a
/// Full commit into the new file is acknowledged.
fn the_directory_is_synced_before_the_first_full_ack(unsynced_tail: bool, truncate: bool) {
    use stoolap::storage::config::PersistenceConfig;
    use stoolap::storage::mvcc::wal_manager::WALManager;
    use stoolap::storage::SyncMode;

    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let config = PersistenceConfig {
        wal_max_size: 1,
        ..PersistenceConfig::default()
    };
    let wal = WALManager::with_config(dir.path(), SyncMode::Full, Some(&config)).unwrap();
    wal_insert(&wal, 1);
    let cut = wal.write_commit_marker(1).unwrap();
    let old_name = wal.current_wal_file();
    if unsynced_tail {
        wal_insert(&wal, 2);
    }
    let synced = std::sync::Arc::new(AtomicBool::new(false));
    let seen = std::sync::Arc::clone(&synced);
    test_failpoints::after_wal_swap(move || {
        test_failpoints::on_wal_directory_sync(move || seen.store(true, Ordering::Release));
    });
    if truncate {
        wal.truncate_wal(cut).unwrap();
    } else {
        assert!(wal.maybe_rotate().unwrap());
    }
    assert_ne!(wal.current_wal_file(), old_name);
    if !unsynced_tail {
        wal_insert(&wal, 2);
    }
    wal.write_commit_marker(2).unwrap();
    assert!(
        synced.load(Ordering::Acquire),
        "the new name's directory sync must precede the commit's acknowledgement"
    );
}

#[test]
fn a_clean_rotation_syncs_the_new_name_before_a_full_commit_is_acknowledged() {
    the_directory_is_synced_before_the_first_full_ack(false, false);
}

#[test]
fn a_clean_truncation_syncs_the_new_name_before_a_full_commit_is_acknowledged() {
    the_directory_is_synced_before_the_first_full_ack(false, true);
}

#[test]
fn a_rotation_with_an_unsynced_tail_syncs_the_new_name_too() {
    the_directory_is_synced_before_the_first_full_ack(true, false);
}

// ============================================================================
// An empty index probe during a commit's publish window
// ============================================================================

/// A commit updates the shared indexes before its versions are visible.
/// A read in that window finds the index without the old key and the
/// visible version still with it: the empty probe must not be its answer.
#[test]
fn a_lookup_during_a_commit_s_publish_window_still_sees_the_old_key() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k) USING BTREE", ())
        .unwrap();
    for i in 1..=200 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
    }
    let ids = |h: &Database, key: i64| -> Vec<i64> {
        let mut out: Vec<i64> = h
            .query("SELECT id FROM t WHERE k = $1", (key,))
            .unwrap()
            .map(|r| r.unwrap().get(0).unwrap())
            .collect();
        out.sort_unstable();
        out
    };

    // The committing thread stops after its index update; the main
    // thread reads inside that window, then lets the commit finish
    let (in_window_tx, in_window_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let writer = db.clone();
    let committer = std::thread::spawn(move || {
        test_failpoints::after_indexes_published(move || {
            in_window_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("released");
        });
        writer.execute("UPDATE t SET k = 555 WHERE id = 7", ())
    });
    in_window_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the commit reached its publish window");
    let during_old = ids(&db, 70);
    let during_new = ids(&db, 555);
    release_tx.send(()).unwrap();
    committer.join().unwrap().unwrap();
    assert_eq!(
        during_old,
        vec![7],
        "the old key must still find the visible row"
    );
    assert_eq!(
        during_new,
        Vec::<i64>::new(),
        "the new key's version is not visible yet"
    );
    assert_eq!(ids(&db, 70), Vec::<i64>::new());
    assert_eq!(ids(&db, 555), vec![7]);
}
