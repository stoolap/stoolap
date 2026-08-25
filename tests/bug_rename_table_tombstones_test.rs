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

//! Regression test for: `ALTER TABLE .. RENAME` reporting success after
//! failing to move a legacy-format `tombstones.dat`. For legacy databases
//! that means the renamed table loads without its tombstones after restart
//! and previously deleted rows resurrect; the rename must fail loud and
//! revert instead.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

#[test]
fn test_rename_fails_loud_when_tombstones_cannot_move() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join("renamedb");
    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for i in 1..=10i64 {
        db.execute("INSERT INTO t (id, v) VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    // Seal first so the deletes target cold rows and produce tombstones.
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM t WHERE id > 5", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Modern manifests carry tombstones inside volumes/<t>/ (moved with the
    // volume dir); fabricate the legacy sidecar the rename path also moves.
    let old_ts = db_dir.join("snapshots").join("t").join("tombstones.dat");
    std::fs::create_dir_all(old_ts.parent().unwrap()).unwrap();
    std::fs::write(&old_ts, b"legacy").unwrap();

    // Sabotage: a DIRECTORY at the rename target makes fs::rename fail.
    let blocked = db_dir.join("snapshots").join("t2").join("tombstones.dat");
    std::fs::create_dir_all(&blocked).unwrap();

    let result = db.execute("ALTER TABLE t RENAME TO t2", ());
    assert!(
        result.is_err(),
        "rename must fail loud when tombstones cannot move"
    );

    // The failed rename must have reverted: t intact, t2 absent.
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 5);
    assert!(db.query("SELECT COUNT(*) FROM t2", ()).is_err());

    // After reopen the deletes must still hold.
    db.close().unwrap();
    std::fs::remove_dir_all(db_dir.join("snapshots").join("t2")).unwrap();
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM t"),
        5,
        "deleted rows must not resurrect after restart"
    );
}
