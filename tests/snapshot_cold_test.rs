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

//! Verify snapshot isolation with cold (volume-backed) rows.
//! After PRAGMA CHECKPOINT seals rows to cold storage, a snapshot
//! transaction must still see pre-update/pre-delete data via
//! versioned tombstones.

use stoolap::{Database, IsolationLevel};

#[test]
fn test_snapshot_isolation_cold_update() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().to_str().unwrap());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'orig')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Start snapshot transaction — should see 'orig'
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let before: String = tx.query_one("SELECT val FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(before, "orig");

    // Auto-commit update from "another session"
    db.execute("UPDATE t SET val = 'new1' WHERE id = 1", ())
        .unwrap();

    // Snapshot should still see 'orig' (point lookup)
    let after: String = tx.query_one("SELECT val FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(after, "orig", "snapshot saw updated value on point lookup");

    // Snapshot should still see 'orig' (full scan)
    let scan_val: String = tx
        .query_one("SELECT val FROM t ORDER BY id LIMIT 1", ())
        .unwrap();
    assert_eq!(scan_val, "orig", "snapshot saw updated value on scan");

    tx.rollback().unwrap();
    db.close().ok();
}

#[test]
fn test_snapshot_isolation_cold_delete() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().to_str().unwrap());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Start snapshot
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();

    // Delete from another session
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();

    // Snapshot should still see both rows
    let count: i64 = tx.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 2, "snapshot lost deleted row");

    let val: String = tx.query_one("SELECT val FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(val, "b", "snapshot can't read deleted row");

    tx.rollback().unwrap();
    db.close().ok();
}

/// Verify that seal proceeds during active snapshot transactions (cutoff-filtered seal).
/// Pre-snapshot rows are sealed to cold; post-snapshot rows stay in hot.
/// Both the snapshot and auto-commit queries see correct data.
#[test]
fn test_seal_proceeds_during_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().to_str().unwrap());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();

    // Insert 200 rows (enough to cross seal threshold)
    for i in 1..=200 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
    }

    // Start snapshot BEFORE checkpoint
    let mut snap = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();

    // Snapshot sees 200 rows
    let count_before: i64 = snap.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count_before, 200);

    // Insert more rows from auto-commit (AFTER snapshot started)
    for i in 201..=250 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
    }

    // Checkpoint: seal should proceed (pre-snapshot rows go cold)
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Snapshot should still see exactly 200 rows (not the 50 new ones)
    let count_after: i64 = snap.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(
        count_after, 200,
        "snapshot should see 200 rows after checkpoint, not {}",
        count_after
    );

    // Auto-commit should see all 250
    let count_auto: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count_auto, 250);

    // Verify snapshot data integrity (spot check)
    let val: i64 = snap
        .query_one("SELECT val FROM t WHERE id = 100", ())
        .unwrap();
    assert_eq!(val, 1000);

    // New rows should not be visible in snapshot
    let new_row_count: i64 = snap
        .query_one("SELECT COUNT(*) FROM t WHERE id > 200", ())
        .unwrap();
    assert_eq!(
        new_row_count, 0,
        "snapshot should not see post-snapshot rows"
    );

    snap.rollback().unwrap();
    db.close().ok();
}

/// Verify that updates to pre-snapshot cold rows are invisible to the snapshot
/// even after the updated rows are sealed again.
#[test]
fn test_snapshot_update_cold_then_reseal() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().to_str().unwrap());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    for i in 1..=200 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'v{}')", i, i), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Start snapshot
    let mut snap = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();

    // Update a cold row from auto-commit
    db.execute("UPDATE t SET val = 'updated' WHERE id = 1", ())
        .unwrap();

    // Second checkpoint (seal the update)
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Snapshot should still see the original value
    let val: String = snap
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        val, "v1",
        "snapshot should see original 'v1', not 'updated'"
    );

    // Auto-commit should see the update
    let auto_val: String = db.query_one("SELECT val FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(auto_val, "updated");

    snap.rollback().unwrap();
    db.close().ok();
}
