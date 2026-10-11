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

//! A commit skips the recheck of its keys against the volumes only when
//! every pending write of the table was checked at the seal generation the
//! commit sees

use stoolap::Database;

fn open(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", dir.display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db
}

fn holders(db: &Database, k: i64) -> i64 {
    db.query_one(&format!("SELECT COUNT(*) FROM t WHERE k = {k}"), ())
        .unwrap()
}

#[test]
fn a_hot_row_given_a_sealed_rows_key_is_checked_at_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 99)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 10)", ()).unwrap();
    let mut txn = db.begin().unwrap();
    // The hot row takes the sealed row's key; a later statement stamps again
    let updated = txn.execute("UPDATE t SET k = 99 WHERE id = 2", ());
    let inserted = txn.execute("INSERT INTO t VALUES (3, 30)", ());
    let committed = txn.commit();
    assert!(
        updated.is_err() || inserted.is_err() || committed.is_err(),
        "the hot row took the sealed row's key: {updated:?} {inserted:?} {committed:?}"
    );
    assert_eq!(holders(&db, 99), 1, "two rows hold one unique key");
}

#[test]
fn a_write_before_the_first_seal_is_checked_at_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut first = db.begin().unwrap();
    first.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    // Another transaction commits the same key under another id and seals it
    db.execute("INSERT INTO t VALUES (2, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // A later statement of the first transaction stamps the new generation
    first.execute("INSERT INTO t VALUES (3, 8)", ()).unwrap();
    let committed = first.commit();
    assert!(
        committed.is_err(),
        "the first write's key was not checked against the sealed row"
    );
    assert_eq!(holders(&db, 7), 1, "two rows hold one unique key");
}

/// The first-seal case through each storage API insert path: `batch`
/// inserts through `insert_batch`, else through `insert`
fn first_write_before_a_seal_through_the_storage_api(batch: bool) {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let row = |id: i64, k: i64| Row::from(vec![Value::Integer(id), Value::Integer(k)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let write = |table: &mut Box<dyn stoolap::storage::traits::Table>, id, k| {
        if batch {
            table.insert_batch(vec![row(id, k)]).unwrap();
        } else {
            table.insert(row(id, k)).unwrap();
        }
    };
    write(&mut table, 1, 7);
    db.execute("INSERT INTO t VALUES (2, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    write(&mut table, 3, 8);
    drop(table);
    let committed = txn.commit();
    assert!(
        committed.is_err(),
        "batch {batch}: the first write's key was not checked against the sealed row"
    );
    assert_eq!(holders(&db, 7), 1, "two rows hold one unique key");
}

#[test]
fn a_write_before_the_first_seal_is_checked_at_commit_through_insert() {
    first_write_before_a_seal_through_the_storage_api(false);
}

#[test]
fn a_write_before_the_first_seal_is_checked_at_commit_through_insert_batch() {
    first_write_before_a_seal_through_the_storage_api(true);
}

#[cfg(feature = "test-failpoints")]
mod failpoints {
    use std::sync::mpsc;
    use stoolap::test_failpoints;
    use stoolap::Database;

    #[test]
    fn a_commit_during_a_column_change_is_refused_as_a_schema_change() {
        let _guard = test_failpoints::FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER UNIQUE)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 0, 1), (2, 0, 2)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let mut txn = db.begin().unwrap();
        txn.execute("INSERT INTO t VALUES (10, 0, 500)", ())
            .unwrap();
        // A seal after the write, so the commit rechecks against the volumes
        db.execute("INSERT INTO t VALUES (3, 0, 3)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let (paused, is_paused) = mpsc::channel::<()>();
        let (resume, resumed) = mpsc::channel::<()>();
        let ddl_db = db.clone();
        let ddl = std::thread::spawn(move || {
            // The column change holds the table closed here
            test_failpoints::after_drop_hot_published(move || {
                paused.send(()).unwrap();
                resumed.recv().unwrap();
            });
            ddl_db.execute("ALTER TABLE t DROP COLUMN a", ())
        });
        is_paused.recv().unwrap();
        let committed = txn.commit();
        resume.send(()).unwrap();
        ddl.join().unwrap().unwrap();
        assert!(
            matches!(committed, Err(stoolap::core::Error::SchemaChanged { .. })),
            "the commit read the volumes through a half-changed layout: {committed:?}"
        );
    }

    #[test]
    fn only_a_hot_update_that_changes_a_key_makes_the_commit_recheck() {
        let _guard = test_failpoints::FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1, 0)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 2, 0)", ()).unwrap();
        let rechecks = |sql: &str| {
            let before = test_failpoints::cold_rechecks();
            db.execute(sql, ()).unwrap();
            test_failpoints::cold_rechecks() - before
        };
        let kept = rechecks("UPDATE t SET v = 5 WHERE id = 2");
        let upsert = rechecks(
            "INSERT INTO t VALUES (2, 2, 9) ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v",
        );
        let changed = rechecks("UPDATE t SET k = 20 WHERE id = 2");
        assert_eq!(
            (kept, upsert, changed),
            (0, 0, 1),
            "rechecks for a kept key, an upsert keeping it, a changed key"
        );
    }
}
