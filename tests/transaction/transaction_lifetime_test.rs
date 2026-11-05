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

//! A transaction keeps its database open until it ends, and a commit on an
//! engine closed under it is refused instead of reported as done.

use stoolap::{Database, Error};
use tempfile::tempdir;

fn count(db: &Database, table: &str) -> i64 {
    db.query_one(&format!("SELECT COUNT(*) FROM {table}"), ())
        .expect("count")
}

#[test]
fn a_transaction_outlives_the_last_memory_handle() {
    let dsn = "memory://tx_outlives_memory_handle";
    let db = Database::open(dsn).expect("open");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .expect("create");
    let mut tx = db.begin().expect("begin");
    tx.execute("INSERT INTO t VALUES (1, 1)", ())
        .expect("insert");
    drop(db);

    let n: i64 = tx
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("read inside the transaction");
    assert_eq!(n, 1);
    let same = Database::open(dsn).expect("reopen while the transaction runs");
    tx.commit().expect("commit");
    assert_eq!(count(&same, "t"), 1, "the committed row is not visible");
}

#[test]
fn a_transaction_outlives_the_last_file_handle() {
    let dir = tempdir().expect("tempdir");
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).expect("open");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .expect("create");
    let mut tx = db.begin().expect("begin");
    tx.execute("INSERT INTO t VALUES (1, 1)", ())
        .expect("insert");
    drop(db);
    tx.commit().expect("commit");
    drop(tx);

    let db = Database::open(&dsn).expect("reopen");
    assert_eq!(count(&db, "t"), 1, "the committed row was lost");
}

/// Under `test-filedb` a memory DSN opens a file database, which keeps its rows
#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_finished_transaction_releases_the_database() {
    for (dsn, commit) in [
        ("memory://tx_finished_releases_commit", true),
        ("memory://tx_finished_releases_rollback", false),
    ] {
        let db = Database::open(dsn).expect("open");
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .expect("create");
        let mut tx = db.begin().expect("begin");
        tx.execute("INSERT INTO t VALUES (1, 1)", ())
            .expect("insert");
        drop(db);
        if commit {
            tx.commit().expect("commit");
        } else {
            tx.rollback().expect("rollback");
        }

        let fresh = Database::open(dsn).expect("reopen");
        assert!(
            fresh.query("SELECT COUNT(*) FROM t", ()).is_err(),
            "{dsn}: the engine stayed open after the transaction ended"
        );
        drop(tx);
    }
}

#[test]
fn a_commit_after_an_explicit_close_is_refused() {
    let db = Database::open("memory://tx_commit_after_close").expect("open");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .expect("create");
    let mut tx = db.begin().expect("begin");
    tx.execute("INSERT INTO t VALUES (1, 1)", ())
        .expect("insert");
    db.close().expect("close");

    let result = tx.commit();
    assert!(
        matches!(result, Err(Error::EngineNotOpen)),
        "a commit on a closed engine returned {result:?}"
    );
}
