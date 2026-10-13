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

//! A primary key SELECT inside a transaction reads through the transaction:
//! its own writes, savepoints and isolation level decide what it sees, on
//! the SQL BEGIN path and on the Transaction API

use stoolap::{Database, IsolationLevel};

const SELECT: &str = "SELECT v FROM t WHERE id = $1";

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db
}

/// Asserts the last statement read through the in-transaction fast path
fn assert_fast(what: &str) {
    #[cfg(feature = "test-failpoints")]
    assert!(
        stoolap::test_failpoints::take_read_paths().contains(&"exec.pk_lookup_in_transaction"),
        "{what} took the in-transaction fast path"
    );
    let _ = what;
}

fn begin_v(db: &Database, id: i64) -> Option<i64> {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    let mut rows = db.query(SELECT, (id,)).unwrap();
    rows.next().map(|row| row.unwrap().get::<i64>(0).unwrap())
}

fn tx_v(tx: &mut stoolap::api::Transaction, id: i64) -> Option<i64> {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    let mut rows = tx.query(SELECT, (id,)).unwrap();
    rows.next().map(|row| row.unwrap().get::<i64>(0).unwrap())
}

#[test]
fn test_begin_sees_its_own_writes() {
    let db = table("tx_pk_begin_own_writes");
    // Compiled in autocommit, then used inside the transaction
    assert_eq!(db.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    assert_eq!(begin_v(&db, 1), Some(11));
    assert_fast("own update");
    assert_eq!(begin_v(&db, 2), None);
    assert_fast("own delete");
    assert_eq!(begin_v(&db, 3), Some(30));
    assert_fast("own insert");
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(db.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
}

#[test]
fn test_transaction_api_sees_its_own_writes() {
    let db = table("tx_pk_api_own_writes");
    let mut tx = db.begin().unwrap();
    tx.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    tx.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    assert_eq!(tx_v(&mut tx, 1), Some(11));
    assert_fast("own update");
    assert_eq!(tx_v(&mut tx, 2), None);
    assert_fast("own delete");
    assert_eq!(tx_v(&mut tx, 3), Some(30));
    assert_fast("own insert");
    tx.rollback().unwrap();
}

#[test]
fn test_savepoint_rollback_restores_what_the_lookup_sees() {
    let db = table("tx_pk_savepoint");
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    db.execute("SAVEPOINT s", ()).unwrap();
    db.execute("UPDATE t SET v = 12 WHERE id = 1", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    assert_eq!(begin_v(&db, 1), Some(12));
    db.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    assert_eq!(begin_v(&db, 1), Some(11));
    assert_fast("after rollback to savepoint");
    assert_eq!(begin_v(&db, 3), None);
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_uncommitted_writes_of_others_stay_invisible() {
    let db = table("tx_pk_others_uncommitted");
    let other = db.clone();
    other.execute("BEGIN", ()).unwrap();
    other
        .execute("UPDATE t SET v = 99 WHERE id = 1", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    assert_eq!(begin_v(&db, 1), Some(10));
    assert_fast("row another transaction changed");
    db.execute("COMMIT", ()).unwrap();
    let mut tx = db.begin().unwrap();
    assert_eq!(tx_v(&mut tx, 1), Some(10));
    assert_fast("row another transaction changed");
    tx.rollback().unwrap();
    other.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn test_commit_between_two_lookups_follows_the_isolation_level() {
    let db = table("tx_pk_commit_between");
    let other = db.clone();

    db.execute("BEGIN", ()).unwrap();
    assert_eq!(begin_v(&db, 1), Some(10));
    other
        .execute("UPDATE t SET v = 11 WHERE id = 1", ())
        .unwrap();
    assert_eq!(begin_v(&db, 1), Some(11), "read committed sees the commit");
    assert_fast("read committed");
    db.execute("COMMIT", ()).unwrap();

    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    assert_eq!(begin_v(&db, 1), Some(11));
    other
        .execute("UPDATE t SET v = 12 WHERE id = 1", ())
        .unwrap();
    assert_eq!(begin_v(&db, 1), Some(11), "a snapshot keeps its view");
    assert_fast("snapshot");
    db.execute("COMMIT", ()).unwrap();

    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(tx_v(&mut tx, 1), Some(12));
    other
        .execute("UPDATE t SET v = 13 WHERE id = 1", ())
        .unwrap();
    assert_eq!(tx_v(&mut tx, 1), Some(12), "a snapshot keeps its view");
    assert_fast("snapshot through the Transaction API");
    tx.commit().unwrap();
}

#[test]
fn test_sealed_row_under_a_later_update_in_a_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let other = db.clone();
    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    assert_eq!(begin_v(&db, 1), Some(10));
    other
        .execute("UPDATE t SET v = 11 WHERE id = 1", ())
        .unwrap();
    assert_eq!(begin_v(&db, 1), Some(10));
    assert_fast("sealed row");
    db.execute("COMMIT", ()).unwrap();
}
