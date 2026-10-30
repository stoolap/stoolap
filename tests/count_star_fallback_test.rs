// Copyright 2026 Stoolap Contributors
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

//! COUNT(*) over sealed and hot rows answers as the rows the statement
//! reads, where the row count cannot come from the stores' counters: under
//! a snapshot, with sealed rows rewritten, deleted, and changed after it

use std::path::Path;
use stoolap::api::Transaction;
use stoolap::Database;

fn open(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.display()
    ))
    .unwrap()
}

fn insert(db: &Database, from: i64, rows: i64) {
    let values: Vec<String> = (from..from + rows)
        .map(|id| format!("({id}, {}, 'n{}')", id % 7, id % 3))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
}

/// COUNT(*) and the number of rows `SELECT id` returns, in `tx`
fn count_and_rows(tx: &mut Transaction) -> (i64, i64) {
    let count: i64 = tx.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    let rows = tx.query("SELECT id FROM t", ()).unwrap().count() as i64;
    (count, rows)
}

#[test]
fn count_star_under_a_snapshot_answers_as_the_rows_it_reads() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    // Five volumes, counted in parallel, then sealed rows rewritten,
    // deleted, and new hot rows
    for volume in 0..5 {
        insert(&db, volume * 1500 + 1, 1500);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let volumes = db
        .query("PRAGMA VOLUME_STATS", ())
        .unwrap()
        .filter(|row| row.as_ref().unwrap().get::<String>(0).unwrap() == "t")
        .count();
    assert_eq!(volumes, 5, "no compaction merged the volumes");
    db.execute("UPDATE t SET k = k + 1 WHERE id % 10 = 0", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id % 13 = 0", ()).unwrap();
    insert(&db, 10_001, 300);
    let expected = 7500 - 7500 / 13 + 300;
    let autocommit: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(autocommit, expected);

    let other = db.clone();
    let mut tx = db
        .begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        count_and_rows(&mut tx),
        (expected, expected),
        "at the snapshot"
    );

    // Commits after the snapshot, a seal among them, stay out of it
    insert(&other, 20_001, 200);
    other
        .execute("DELETE FROM t WHERE id % 17 = 0", ())
        .unwrap();
    other
        .execute("UPDATE t SET k = k + 1 WHERE id % 11 = 0", ())
        .unwrap();
    other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        count_and_rows(&mut tx),
        (expected, expected),
        "after later commits"
    );

    // The transaction's own insert and its own delete of a sealed row count
    tx.execute("INSERT INTO t VALUES (30001, 1, 'x')", ())
        .unwrap();
    tx.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(
        count_and_rows(&mut tx),
        (expected, expected),
        "with its own writes"
    );
    tx.commit().unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    let rows = db.query("SELECT id FROM t", ()).unwrap().count() as i64;
    assert_eq!(count, rows, "after the commit");
}
