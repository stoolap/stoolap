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
