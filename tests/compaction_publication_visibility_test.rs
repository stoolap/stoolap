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

//! What a compaction's publication leaves visible around the volumes it
//! rewrites. Debug builds also check every published bitmap against a
//! pass over the whole table

use std::path::Path;

use stoolap::Database;

fn open(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000&target_volume_rows=65536",
        dir.display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db
}

fn seal(db: &Database, sql: &str) {
    db.execute(sql, ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

fn insert_range(db: &Database, from: i64, to: i64, v: i64) {
    let ids: Vec<i64> = (from..=to).collect();
    for chunk in ids.chunks(8_192) {
        let values: Vec<String> = chunk.iter().map(|id| format!("({id}, {v})")).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
}

fn row_counts(db: &Database) -> Vec<usize> {
    db.engine().volume_stats().iter().map(|v| v.3).collect()
}

fn compact(db: &Database) {
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 1000", ()).unwrap();
}

fn value(db: &Database, id: i64) -> Vec<i64> {
    db.query("SELECT v FROM t WHERE id = $1", (id,))
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn an_output_row_a_newer_volume_holds_again_stays_hidden() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal(&db, "INSERT INTO t VALUES (1, 1)");
    seal(&db, "INSERT INTO t VALUES (2, 1)");
    seal(&db, "INSERT INTO t VALUES (3, 1)");
    // A full volume newer than the small ones, holding a newer copy of 1
    db.execute("UPDATE t SET v = 2 WHERE id = 1", ()).unwrap();
    insert_range(&db, 10, 65_544, 0);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(row_counts(&db), vec![1, 1, 1, 65_536]);
    compact(&db);
    assert_eq!(row_counts(&db), vec![3, 65_536], "the small volumes merged");
    assert_eq!(value(&db, 1), vec![2], "the newer copy wins");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_538);
}

#[test]
fn an_older_volume_keeps_hidden_the_rows_a_batch_holds_again() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    insert_range(&db, 1, 65_536, 0);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    seal(&db, "UPDATE t SET v = 1 WHERE id = 1");
    seal(&db, "UPDATE t SET v = 1 WHERE id = 2");
    seal(&db, "INSERT INTO t VALUES (70000, 1)");
    assert_eq!(row_counts(&db), vec![65_536, 1, 1, 1]);
    compact(&db);
    assert_eq!(row_counts(&db), vec![65_536, 3]);
    assert_eq!(value(&db, 1), vec![1]);
    assert_eq!(value(&db, 2), vec![1]);
    assert_eq!(value(&db, 3), vec![0]);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_537);
}

#[test]
fn a_volume_between_the_members_keeps_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal(&db, "INSERT INTO t VALUES (1, 1)");
    // A full volume between the small ones, sharing no id with them
    insert_range(&db, 100, 65_635, 0);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    seal(&db, "INSERT INTO t VALUES (2, 1)");
    seal(&db, "INSERT INTO t VALUES (3, 1)");
    assert_eq!(row_counts(&db), vec![1, 65_536, 1, 1]);
    compact(&db);
    assert_eq!(row_counts(&db).iter().sum::<usize>(), 65_539);
    assert_eq!(row_counts(&db).len(), 2, "the small volumes merged");
    for id in [1, 2, 3] {
        assert_eq!(value(&db, id), vec![1]);
    }
    assert_eq!(value(&db, 100), vec![0]);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_539);
}
