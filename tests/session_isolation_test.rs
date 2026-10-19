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

//! The isolation level is the handle's: SET isolation_level and
//! set_default_isolation_level change it for that handle alone, and its
//! BEGIN and begin() start at it unless they name a level

use stoolap::{Database, IsolationLevel};

const READ: &str = "SELECT v FROM t WHERE id = 1";

fn open(name: &str) -> (Database, Database) {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let writer = db.clone();
    (db, writer)
}

fn bump(writer: &Database) {
    writer
        .execute("UPDATE t SET v = v + 1 WHERE id = 1", ())
        .unwrap();
}

/// Whether a transaction from `begin()` sees a commit made after its first read
fn api_sees_later_commit(db: &Database, writer: &Database) -> bool {
    let mut tx = db.begin().unwrap();
    let before: i64 = tx.query_one(READ, ()).unwrap();
    bump(writer);
    let after: i64 = tx.query_one(READ, ()).unwrap();
    tx.commit().unwrap();
    before != after
}

/// Whether a transaction opened by `begin_sql` sees a commit made after its first read
fn sql_sees_later_commit(db: &Database, writer: &Database, begin_sql: &str) -> bool {
    db.execute(begin_sql, ()).unwrap();
    let before: i64 = db.query_one(READ, ()).unwrap();
    bump(writer);
    let after: i64 = db.query_one(READ, ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    before != after
}

#[test]
fn a_handle_starts_at_read_committed() {
    let (db, writer) = open("session_iso_default");
    assert!(api_sees_later_commit(&db, &writer), "begin()");
    assert!(sql_sees_later_commit(&db, &writer, "BEGIN"), "BEGIN");
}

#[test]
fn set_isolation_level_moves_this_handle_s_begin_and_begin_api() {
    let (db, writer) = open("session_iso_set");
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    assert!(!api_sees_later_commit(&db, &writer), "begin()");
    assert!(!sql_sees_later_commit(&db, &writer, "BEGIN"), "BEGIN");
    db.execute("SET isolation_level = 'READ COMMITTED'", ())
        .unwrap();
    assert!(api_sees_later_commit(&db, &writer), "begin() after reset");
    assert!(
        sql_sees_later_commit(&db, &writer, "BEGIN"),
        "BEGIN after reset"
    );
}

#[test]
fn set_default_isolation_level_is_the_same_setting() {
    let (db, writer) = open("session_iso_api");
    db.set_default_isolation_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert!(!api_sees_later_commit(&db, &writer), "begin()");
    assert!(!sql_sees_later_commit(&db, &writer, "BEGIN"), "BEGIN");
}

#[test]
fn another_handle_keeps_its_own_level() {
    let (db, writer) = open("session_iso_other");
    let other = db.clone();
    let reopened = Database::open("memory://session_iso_other").unwrap();
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    for (name, handle) in [("clone", &other), ("reopened", &reopened)] {
        assert!(api_sees_later_commit(handle, &writer), "{name}: begin()");
        assert!(
            sql_sees_later_commit(handle, &writer, "BEGIN"),
            "{name}: BEGIN"
        );
    }
}

#[test]
fn a_named_level_overrides_the_handle_s() {
    let (db, writer) = open("session_iso_named");
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    let mut tx = db
        .begin_with_isolation(IsolationLevel::ReadCommitted)
        .unwrap();
    let before: i64 = tx.query_one(READ, ()).unwrap();
    bump(&writer);
    let after: i64 = tx.query_one(READ, ()).unwrap();
    tx.commit().unwrap();
    assert_ne!(before, after, "begin_with_isolation(ReadCommitted)");
    assert!(
        sql_sees_later_commit(
            &db,
            &writer,
            "BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED"
        ),
        "BEGIN ... READ COMMITTED"
    );
}

#[test]
fn prepared_statements_read_at_the_transaction_s_level() {
    let (db, writer) = open("session_iso_prepared");
    db.set_default_isolation_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let read = db.prepare(READ).unwrap();
    let value = |tx: &mut stoolap::ApiTransaction| -> i64 {
        let mut rows = read.query_in(tx, ()).unwrap();
        assert!(rows.advance());
        rows.current_row().as_slice()[0].as_int64().unwrap()
    };
    let mut tx = db.begin().unwrap();
    let before = value(&mut tx);
    bump(&writer);
    assert_eq!(value(&mut tx), before, "snapshot read through query_in");
    tx.commit().unwrap();
}
