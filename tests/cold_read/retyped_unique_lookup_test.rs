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

//! A unique key sealed under a column's old type is found under its new
//! type: the stored cell is cast to the current type, whatever volume holds
//! it and whether or not a compaction rewrote it

use stoolap::Database;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    ))
    .unwrap()
}

/// The row id a statement's unique refusal names, or None when it ran
fn refused_by(db: &Database, sql: &str) -> Option<i64> {
    match db.execute(sql, ()) {
        Ok(_) => None,
        Err(stoolap::core::Error::UniqueConstraint { row_id, .. }) => Some(row_id),
        Err(e) => panic!("{sql}: {e:?}"),
    }
}

/// `k INTEGER UNIQUE` with 7 and 8 sealed, then retyped to TEXT
fn retyped(dir: &std::path::Path) -> Database {
    let db = open(dir);
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7, 0), (2, 8, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        .unwrap();
    db
}

fn assert_seven_is_held_by_row_one(db: &Database, case: &str) {
    assert_eq!(
        refused_by(db, "INSERT INTO t VALUES (10, '7', 0)"),
        Some(1),
        "{case}: '7' against the sealed 7"
    );
    assert_eq!(
        refused_by(db, "UPDATE t SET k = '7' WHERE id = 2"),
        Some(1),
        "{case}: row 2 set to '7'"
    );
    assert_eq!(
        refused_by(db, "INSERT INTO t VALUES (11, '07', 0)"),
        None,
        "{case}: '07' is another text"
    );
    assert_eq!(
        refused_by(db, "INSERT INTO t VALUES (12, '9', 0)"),
        None,
        "{case}: '9' is free"
    );
}

#[test]
fn a_text_key_equal_to_a_sealed_integer_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = retyped(dir.path());
    assert_seven_is_held_by_row_one(&db, "sealed");
}

#[test]
fn a_text_key_equal_to_a_sealed_integer_is_refused_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    retyped(dir.path()).close().unwrap();
    let db = open(dir.path());
    assert_seven_is_held_by_row_one(&db, "reopened");
}

#[test]
fn a_text_key_equal_to_a_sealed_integer_is_refused_after_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = retyped(dir.path());
    for id in [3, 4] {
        db.execute(&format!("INSERT INTO t VALUES ({id}, 'x{id}', 0)"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let volumes = db.engine().volume_stats().len();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let compacted = db.engine().volume_stats().len();
    assert!(
        compacted < volumes,
        "no compaction: {volumes} -> {compacted}"
    );
    assert_seven_is_held_by_row_one(&db, "compacted");
}

#[test]
fn a_composite_key_with_a_retyped_column_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, UNIQUE(a, b))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7, 'x')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN a TEXT", ())
        .unwrap();
    assert_eq!(
        refused_by(&db, "INSERT INTO t VALUES (2, '7', 'x')"),
        Some(1)
    );
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (3, '7', 'y')"), None);
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (4, '07', 'x')"), None);
}

#[test]
fn a_key_moved_to_a_newer_volume_frees_its_old_value() {
    let dir = tempfile::tempdir().unwrap();
    let db = retyped(dir.path());
    // Row 1's new copy, '70', is sealed in a newer volume than its old 7
    db.execute("UPDATE t SET k = '70' WHERE id = 1", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        refused_by(&db, "INSERT INTO t VALUES (20, '70', 0)"),
        Some(1),
        "the new copy holds '70'"
    );
    assert_eq!(
        refused_by(&db, "INSERT INTO t VALUES (21, '7', 0)"),
        None,
        "the old copy's 7 is gone"
    );
}

#[test]
fn a_value_the_new_type_cannot_hold_is_null_and_holds_no_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT UNIQUE)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'abc'), (2, '5')", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k INTEGER", ())
        .unwrap();
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (3, 5)"), Some(2));
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (4, 6)"), None);
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (5, NULL)"), None);
}
