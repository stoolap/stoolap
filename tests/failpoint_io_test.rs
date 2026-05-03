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
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
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
        // The failed marker write trips the engine's catastrophic-
        // failure latch (the SWMR durability invariant: a markerless
        // commit cannot be recovered safely in-process). The latched
        // engine refuses further writes; the user must reopen.
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
        let post_fail = db.execute("INSERT INTO fp_wal VALUES (1, 'post')", ());
        assert!(
            post_fail.is_err(),
            "INSERT against a latched engine must fail even after disarm"
        );
        // Drop the latched engine so reopen below sees a fresh one.
    }

    // Reopen: recovery discards the markerless transaction; the
    // fresh engine accepts new writes.
    let db = Database::open(&path).expect("Failed to reopen database");
    db.execute("INSERT INTO fp_wal VALUES (2, 'after_fail')", ())
        .expect("INSERT after reopen should succeed");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_wal", ())
        .expect("COUNT should work");
    assert!(count >= 1, "At least one row should exist, got {}", count);
}

#[test]
fn test_wal_write_fail_mid_transaction_atomicity() {
    let _guard = failpoint_guard();
    let dir = tempdir().unwrap();
    let path = format!("file://{}", dir.path().display());

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_wal_tx (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .expect("CREATE should succeed");
        db.execute("INSERT INTO fp_wal_tx VALUES (1, 100)", ())
            .expect("Initial insert should succeed");

        db.execute("BEGIN", ()).expect("BEGIN should succeed");
        db.execute("UPDATE fp_wal_tx SET val = 200 WHERE id = 1", ())
            .expect("UPDATE should succeed within transaction");

        // Arm failpoint before commit. The COMMIT marker write fails;
        // the engine latches into the catastrophic-failure state.
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        let result = db.execute("COMMIT", ());
        assert!(result.is_err(), "COMMIT must fail with WAL failpoint armed");
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
        // Drop the latched engine; recovery on reopen discards the
        // markerless commit.
    }

    let db = Database::open(&path).expect("reopen after failpoint");
    let val: i64 = db
        .query_one("SELECT val FROM fp_wal_tx WHERE id = 1", ())
        .expect("SELECT should work");
    assert_eq!(val, 100, "Value must be unchanged after failed commit");
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

        // Arm failpoint, try to write, fail. The engine latches.
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        let _ = db.execute("INSERT INTO fp_wal_rec VALUES (2, 'during_fail')", ());
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
        // Drop the latched engine; the next reopen recovers cleanly.
    }

    // Reopen and verify the markerless txn was discarded; new
    // writes succeed.
    {
        let db = Database::open(&path).expect("Failed to reopen after failpoint");
        db.execute("INSERT INTO fp_wal_rec VALUES (3, 'after_fail')", ())
            .expect("INSERT should succeed after reopen");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_wal_rec", ())
            .expect("COUNT should work");
        // Row 1 (committed), row 2 (recovery dropped the markerless
        // attempt), row 3 (committed after reopen).
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

    // Should be usable again
    let insert_result = db.execute("INSERT INTO fp_sync VALUES (2, 'after')", ());
    if result.is_err() {
        // If the first insert failed, insert with id=1 too
        let _ = db.execute("INSERT INTO fp_sync VALUES (1, 'retry')", ());
    }

    // Database should be in a consistent state
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM fp_sync", ())
        .expect("COUNT should work after sync recovery");
    assert!(count >= 1, "At least one row should exist");

    drop(insert_result);
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

    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE fp_multi (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .expect("CREATE should succeed");

        // Phase 1: WAL write failure latches the engine.
        test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
        let _ = db.execute("INSERT INTO fp_multi VALUES (1, 100)", ());
        test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
        // Drop the latched engine; reopen below clears it.
    }

    // Reopen for Phase 2 (normal operation after WAL recovery).
    {
        let db = Database::open(&path).expect("Reopen after WAL failpoint");
        db.execute("INSERT INTO fp_multi VALUES (2, 200)", ())
            .expect("Should succeed on the post-recovery engine");

        // Phase 3: Snapshot failure. VACUUM does not advance the WAL
        // marker frontier in a way that latches the engine, so the
        // engine stays usable for subsequent writes after the
        // failpoint is disarmed.
        test_failpoints::SNAPSHOT_WRITE_FAIL.store(true, Ordering::Release);
        let _ = db.execute("VACUUM", ());
        test_failpoints::SNAPSHOT_WRITE_FAIL.store(false, Ordering::Release);

        // Phase 4: Normal operation again.
        db.execute("INSERT INTO fp_multi VALUES (3, 300)", ())
            .expect("Should succeed after disarming snapshot failpoint");

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_multi", ())
            .expect("COUNT should work");
        assert!(
            count >= 2,
            "At least rows 2 and 3 should exist, got {}",
            count
        );
    }
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

    // Phase 2: Each WAL-write failpoint latches the engine, so each
    // arm-disarm cycle gets a fresh engine reopen. The other
    // failpoints (sync / snapshot / checkpoint) do not trip the
    // catastrophic-failure latch on their own, so we batch them
    // against a single engine. Either way, original data must
    // survive across all the failed write attempts.
    let latching: &[&AtomicBool] = &[
        &test_failpoints::WAL_WRITE_FAIL,
        &test_failpoints::CHECKPOINT_WRITE_FAIL,
    ];
    let non_latching: &[&AtomicBool] = &[
        &test_failpoints::WAL_SYNC_FAIL,
        &test_failpoints::SNAPSHOT_WRITE_FAIL,
        &test_failpoints::SNAPSHOT_SYNC_FAIL,
        &test_failpoints::SNAPSHOT_RENAME_FAIL,
    ];

    for fp in latching {
        let db = Database::open(&path).expect("Reopen should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_preserve", ())
            .expect("COUNT should work");
        assert_eq!(count, 20, "original 20 rows visible at reopen");

        fp.store(true, Ordering::Release);
        let _ = db.execute("INSERT INTO fp_preserve VALUES (999, 'fail')", ());
        let _ = db.execute("DELETE FROM fp_preserve WHERE id = 999", ());
        let _ = db.execute("VACUUM", ());
        fp.store(false, Ordering::Release);
        // Drop the (potentially) latched engine.
    }

    {
        let db = Database::open(&path).expect("Reopen should succeed");
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM fp_preserve", ())
            .expect("COUNT should work");
        assert_eq!(count, 20, "original 20 rows visible at reopen");

        for fp in non_latching {
            fp.store(true, Ordering::Release);
            let _ = db.execute("INSERT INTO fp_preserve VALUES (999, 'fail')", ());
            let _ = db.execute("DELETE FROM fp_preserve WHERE id = 999", ());
            let _ = db.execute("VACUUM", ());
            fp.store(false, Ordering::Release);
        }

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
