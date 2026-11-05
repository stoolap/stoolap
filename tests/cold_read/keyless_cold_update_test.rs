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

//! An UPDATE of a sealed row of a table without a primary key keeps the
//! row's id, across a reopen, a new seal, a compaction and an undo

use stoolap::storage::traits::Engine;
use stoolap::Database;

fn dsn(dir: &std::path::Path) -> String {
    format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=2",
        dir.display()
    )
}

/// `(row id, a, v)` of every row the storage API sees, by row id
fn rows(db: &Database) -> Vec<(i64, i64, i64)> {
    let txn = db.engine().begin_transaction().unwrap();
    let table = txn.get_table("k").unwrap();
    let mut rows: Vec<(i64, i64, i64)> = table
        .collect_all_rows(None)
        .unwrap()
        .iter()
        .map(|(row_id, row)| {
            let value = |i: usize| row.get(i).and_then(|v| v.as_int64()).unwrap();
            (*row_id, value(0), value(1))
        })
        .collect();
    rows.sort_unstable();
    rows
}

fn sealed_keyless(dir: &std::path::Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE k (a INTEGER, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO k VALUES (1, 0), (2, 0), (3, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

#[test]
fn a_keyless_sealed_row_keeps_its_id_across_reopen_seal_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_keyless(dir.path());
    let before = rows(&db);
    assert_eq!(db.execute("UPDATE k SET v = 5 WHERE a = 2", ()).unwrap(), 1);
    let expected: Vec<(i64, i64, i64)> = before
        .iter()
        .map(|&(id, a, v)| (id, a, if a == 2 { 5 } else { v }))
        .collect();
    assert_eq!(rows(&db), expected, "the updated row took another id");

    // Reopen with the change only in the log
    drop(db);
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(rows(&db), expected, "after a reopen from the log");

    // A new seal holds the row while the old volume holds it tombstoned
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(rows(&db), expected, "after a new seal");

    // A second change and seal make the volumes compact
    assert_eq!(db.execute("UPDATE k SET v = 7 WHERE a = 3", ()).unwrap(), 1);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let expected: Vec<(i64, i64, i64)> = expected
        .iter()
        .map(|&(id, a, v)| (id, a, if a == 3 { 7 } else { v }))
        .collect();
    assert_eq!(rows(&db), expected, "after compaction");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM k", ()).unwrap();
    assert_eq!(count, 3);

    // New rows after a reopen take ids of their own
    drop(db);
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("INSERT INTO k VALUES (4, 0), (5, 0)", ())
        .unwrap();
    let after = rows(&db);
    assert_eq!(after.len(), 5, "an inserted row took a taken id: {after:?}");
    assert_eq!(&after[..3], &expected[..]);
}

#[test]
fn a_keyless_sealed_row_update_undone_leaves_the_row_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_keyless(dir.path());
    db.execute("CREATE INDEX kv ON k(v)", ()).unwrap();
    let before = rows(&db);

    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE k SET v = 5 WHERE a = 2", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(rows(&db), before, "after a rollback");

    db.execute("BEGIN", ()).unwrap();
    db.execute("SAVEPOINT s", ()).unwrap();
    db.execute("UPDATE k SET v = 5 WHERE a = 2", ()).unwrap();
    db.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(rows(&db), before, "after a rollback to a savepoint");

    // The index answers the committed change, under the row's own id
    db.execute("UPDATE k SET v = 5 WHERE a = 2", ()).unwrap();
    let found: i64 = db.query_one("SELECT a FROM k WHERE v = 5", ()).unwrap();
    assert_eq!(found, 2);
    let old: i64 = db
        .query_one("SELECT COUNT(*) FROM k WHERE v = 0", ())
        .unwrap();
    assert_eq!(old, 2);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM k", ()).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn an_integer_key_cold_update_refuses_a_null_or_a_wrong_type_when_it_runs() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let null = table.update_by_row_ids(&[1], &mut |mut row| {
        row.set(1, stoolap::core::Value::null_unknown()).unwrap();
        Ok((row, true))
    });
    let text = table.update_by_row_ids(&[1], &mut |mut row| {
        row.set(1, stoolap::core::Value::text("not a number"))
            .unwrap();
        Ok((row, true))
    });
    drop(table);
    txn.rollback().unwrap();
    assert!(
        null.is_err(),
        "a NULL went into a NOT NULL column: {null:?}"
    );
    assert!(text.is_err(), "text went into an INTEGER column: {text:?}");
}
