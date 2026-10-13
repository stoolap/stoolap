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

//! A change of a column's type holds in every layer once it returns: hot
//! versions, indexes and volumes read in the new type, before and after a
//! reopen, and a change that cannot hold is refused with nothing changed

#![cfg(feature = "test-failpoints")]

use stoolap::core::{Row, Value};
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

fn dsn(dir: &std::path::Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// The rows of `sql` as debug strings of their values
fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| format!("{:?}", r.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

/// The row id a statement's unique refusal names, or None when it ran
fn refused_by(db: &Database, sql: &str) -> Option<i64> {
    match db.execute(sql, ()) {
        Ok(_) => None,
        Err(stoolap::core::Error::UniqueConstraint { row_id, .. }) => Some(row_id),
        Err(e) => panic!("{sql}: {e:?}"),
    }
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn table_k(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
}

fn assert_text_layers(db: &Database, case: &str) {
    assert_eq!(
        rows(db, "SELECT id, k FROM t WHERE id IN (1, 2) ORDER BY id"),
        ["Integer(1),Text(\"7\")", "Integer(2),Text(\"8\")"],
        "{case}: values in the new type"
    );
    assert_eq!(
        rows(db, "SELECT id FROM t WHERE k = '7'"),
        ["Integer(1)"],
        "{case}: found by the new type"
    );
    assert_eq!(
        refused_by(db, "INSERT INTO t VALUES (10, '7')"),
        Some(1),
        "{case}: '7' is held"
    );
}

#[test]
fn a_hot_only_table_holds_the_new_type_live_and_after_reopen() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table_k(&db);
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        .unwrap();
    assert_text_layers(&db, "hot live");
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_text_layers(&recovered, "hot recovered");
}

#[test]
fn a_mixed_table_holds_the_new_type_live_and_after_reopen() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table_k(&db);
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        .unwrap();
    assert_text_layers(&db, "mixed live");
    db.close().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_text_layers(&db, "mixed reopened");
}

/// `k TEXT UNIQUE` holding '7' and '07', the first sealed when `seal_first`,
/// both sealed when `seal_both`
fn colliding_texts(dir: &std::path::Path, seal_first: bool, seal_both: bool) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT UNIQUE)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '7')", ()).unwrap();
    if seal_first {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("INSERT INTO t VALUES (2, '07')", ()).unwrap();
    if seal_both {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

#[test]
fn a_cast_that_makes_two_keys_equal_is_refused_with_nothing_changed() {
    let _guard = FailpointGuard::new();
    for (seal_first, seal_both, case) in [
        (false, false, "hot"),
        (true, true, "sealed"),
        (true, false, "hot and sealed"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = colliding_texts(dir.path(), seal_first, seal_both);
        let changed = db.execute("ALTER TABLE t MODIFY COLUMN k INTEGER", ());
        assert!(
            matches!(changed, Err(stoolap::core::Error::UniqueConstraint { .. })),
            "{case}: {changed:?}"
        );
        assert_eq!(
            rows(&db, "SELECT id, k FROM t ORDER BY id"),
            ["Integer(1),Text(\"7\")", "Integer(2),Text(\"07\")"],
            "{case}: rows unchanged"
        );
        assert_eq!(
            refused_by(&db, "INSERT INTO t VALUES (3, '07')"),
            Some(2),
            "{case}: the TEXT index still holds '07'"
        );
    }
}

#[test]
fn a_key_freed_before_the_change_does_not_collide() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table_k(&db);
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET k = 8 WHERE id = 1", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 7)", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        .unwrap();
    assert_eq!(
        rows(&db, "SELECT id, k FROM t ORDER BY id"),
        ["Integer(1),Text(\"8\")", "Integer(2),Text(\"7\")"]
    );
}

#[test]
fn a_type_change_the_policy_forbids_is_refused() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id), pu INTEGER REFERENCES p(u))",
        (),
    )
    .unwrap();
    for (sql, case) in [
        ("ALTER TABLE p MODIFY COLUMN id TEXT", "the primary key"),
        (
            "ALTER TABLE c MODIFY COLUMN pid TEXT",
            "a foreign key column",
        ),
        (
            "ALTER TABLE p MODIFY COLUMN u TEXT",
            "a unique column a foreign key references",
        ),
    ] {
        assert!(db.execute(sql, ()).is_err(), "{case} changed type");
    }
}

#[test]
fn turning_not_null_with_a_sealed_null_is_refused() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, NULL), (2, 5)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert!(
        db.execute("ALTER TABLE t MODIFY COLUMN k INTEGER NOT NULL", ())
            .is_err(),
        "a NULL was sealed"
    );
}

#[test]
fn a_vector_column_with_an_hnsw_index_does_not_become_text() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    db.execute("CREATE INDEX v_e ON v(e) USING HNSW", ())
        .unwrap();
    db.execute("INSERT INTO v VALUES (1, '[1,2,3]')", ())
        .unwrap();
    assert!(
        db.execute("ALTER TABLE v MODIFY COLUMN e TEXT", ())
            .is_err(),
        "the HNSW index cannot hold text"
    );
}

/// The change, in another thread; returns once it closed the table and
/// started to wait, or panics when it never does
fn change_waiting(db: &Database) -> std::thread::JoinHandle<stoolap::core::Result<i64>> {
    let other = db.clone();
    let (reached, waiting) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        fp::on_column_change_waiting(move || reached.send(()).unwrap());
        other.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
    });
    waiting
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the change closed the table and waits");
    handle
}

#[test]
fn a_commit_published_before_the_change_is_cast_when_its_marker_lands() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table_k(&db);
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let change = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&change);
    let db_for_hook = db.clone();
    fp::before_commit_marker(move || {
        *slot.borrow_mut() = Some(change_waiting(&db_for_hook));
    });
    db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
    let changed = change.borrow_mut().take().unwrap().join().unwrap();
    assert!(changed.is_ok(), "{changed:?}");
    assert_text_layers(&db, "marker landed");
}

#[test]
fn a_transaction_that_committed_a_table_directly_is_refused_after_a_type_change() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table_k(&db);
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    let changed = db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ());
    drop(table);
    let committed = txn.commit();
    assert!(
        changed.is_ok(),
        "the table commit applied nothing: {changed:?}"
    );
    assert!(
        matches!(committed, Err(stoolap::Error::SchemaChanged { .. })),
        "{committed:?}"
    );
    assert_eq!(rows(&db, "SELECT id FROM t ORDER BY id"), ["Integer(1)"]);
}

#[test]
fn an_output_prepared_before_the_change_is_not_published_after_it() {
    let _guard = FailpointGuard::new();
    for (case, compaction) in [("seal", false), ("compaction", true)] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        table_k(&db);
        db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
        if compaction {
            for id in [3, 4] {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                db.execute(&format!("INSERT INTO t VALUES ({id}, {})", id * 100), ())
                    .unwrap();
            }
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        }
        db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
        let other = db.clone();
        // The seal or compaction built its volumes in INTEGER; the change
        // lands before they are published
        fp::after_side_files_built(move || {
            other
                .execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
                .unwrap();
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_text_layers(&db, case);
    }
}
