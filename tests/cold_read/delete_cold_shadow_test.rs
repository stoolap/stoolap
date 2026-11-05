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

//! A DELETE inside a transaction skips a sealed row the transaction already
//! updated or deleted: its hot version or pending tombstone shadows the cold copy

use stoolap::Database;

fn sealed_table() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $1)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=2_000i64 {
        insert.execute((id,)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(!db.engine().volume_stats().is_empty());
    (dir, db)
}

#[test]
fn delete_old_value_does_not_delete_own_updated_sealed_row() {
    let (_dir, db) = sealed_table();
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET v = 99 WHERE id = 1", ()).unwrap(),
        1
    );
    assert_eq!(tx.execute("DELETE FROM t WHERE v = 1", ()).unwrap(), 0);
    assert_eq!(
        tx.query("SELECT v FROM t WHERE id = 1", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get::<i64>(0)
            .unwrap(),
        99
    );
    tx.commit().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT v FROM t WHERE id = 1", ())
            .unwrap(),
        99
    );
}

#[test]
fn deleting_a_sealed_row_twice_counts_it_once() {
    let (_dir, db) = sealed_table();
    let mut tx = db.begin().unwrap();
    assert_eq!(tx.execute("DELETE FROM t WHERE v = 1", ()).unwrap(), 1);
    assert_eq!(tx.execute("DELETE FROM t WHERE v = 1", ()).unwrap(), 0);
    tx.rollback().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT v FROM t WHERE id = 1", ())
            .unwrap(),
        1
    );
}
