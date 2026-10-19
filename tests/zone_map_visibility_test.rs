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

//! Zone maps prune a scan only for a statement that reads the committed
//! state they were built from: no writes of its own, no older snapshot,
//! and no commit since

use stoolap::{Database, IsolationLevel};

fn ids(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    rows.and_then(|rows| {
        rows.map(|row| row.map(|row| format!("{:?}", row.as_row().as_slice())))
            .collect::<stoolap::Result<Vec<_>>>()
    })
    .map_err(|e| e.to_string())
}

const READS: [(&str, &str); 3] = [
    ("SELECT id FROM t WHERE k = 9", "[Integer(2)]"),
    (
        "SELECT COUNT(*) FROM t WHERE k = 9 HAVING COUNT(*) > 0",
        "[Integer(1)]",
    ),
    (
        "SELECT id FROM t WHERE k > 8 ORDER BY id LIMIT 5",
        "[Integer(2)]",
    ),
];

fn analyzed(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    db.execute("ANALYZE t", ()).unwrap();
}

#[test]
fn a_transaction_sees_its_own_rows_past_the_zone_maps() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            analyzed(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db.execute("ANALYZE t", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://zone_map_own_rows").unwrap();
            analyzed(&db);
            db
        };
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (2, 9)", ()).unwrap();
        for (sql, row) in READS {
            let got = ids(tx.query(sql, ()));
            if got != Ok(vec![row.to_string()]) {
                wrong.push(format!("sealed {sealed}: {sql} gave {got:?}"));
            }
        }
        let got = ids(tx.query("SELECT id FROM t WHERE k = $1", (9,)));
        if got != Ok(vec!["[Integer(2)]".to_string()]) {
            wrong.push(format!("sealed {sealed}: parameter gave {got:?}"));
        }
        tx.rollback().unwrap();
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn an_older_snapshot_reads_past_zone_maps_built_after_it() {
    let db = Database::open("memory://zone_map_old_snapshot").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 9)", ())
        .unwrap();
    let mut snapshot = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        ids(snapshot.query("SELECT COUNT(*) FROM t", ()))
            .unwrap()
            .len(),
        1
    );
    let other = db.clone();
    other
        .execute("UPDATE t SET k = 1 WHERE id = 2", ())
        .unwrap();
    other.execute("ANALYZE t", ()).unwrap();
    let mut wrong = Vec::new();
    for (sql, row) in READS {
        let got = ids(snapshot.query(sql, ()));
        if got != Ok(vec![row.to_string()]) {
            wrong.push(format!("{sql} gave {got:?}"));
        }
    }
    snapshot.rollback().unwrap();
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn zone_maps_built_inside_a_transaction_leave_out_its_writes() {
    let db = Database::open("memory://zone_map_analyze_in_tx").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 9)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    db.execute("ANALYZE t", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    let mut wrong = Vec::new();
    for (sql, row) in READS {
        let got = ids(db.query(sql, ()));
        if got != Ok(vec![row.to_string()]) {
            wrong.push(format!("{sql} gave {got:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_between_analyzes_read_and_its_maps_leaves_them_unused() {
    let db = Database::open("memory://zone_map_analyze_race").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_analyze_rows_collected(move || {
        let insert = std::thread::spawn(move || other.execute("INSERT INTO t VALUES (2, 9)", ()));
        insert.join().unwrap().unwrap();
    });
    db.execute("ANALYZE t", ()).unwrap();
    let mut wrong = Vec::new();
    for (sql, row) in READS {
        let got = ids(db.query(sql, ()));
        if got != Ok(vec![row.to_string()]) {
            wrong.push(format!("{sql} gave {got:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
