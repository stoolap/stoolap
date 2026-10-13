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

//! Every open of a DSN is its own session on the one engine, and the engine
//! stays open until the last handle on it drops

use std::sync::Arc;

use stoolap::Database;

fn table(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> usize {
    db.query("SELECT * FROM t", ()).unwrap().count()
}

#[test]
fn test_two_opens_have_their_own_transactions() {
    let a = table("memory://open_session_transactions");
    let b = Database::open("memory://open_session_transactions").unwrap();
    a.execute("BEGIN", ()).unwrap();
    a.execute("INSERT INTO t VALUES (99)", ()).unwrap();
    assert_eq!(rows(&b), 0, "b read a's uncommitted row");
    let _ = b.execute("ROLLBACK", ());
    a.execute("COMMIT", ()).unwrap();
    assert_eq!(rows(&b), 1, "b's rollback undid a's transaction");
}

#[test]
fn test_two_opens_share_the_engine() {
    let a = table("memory://open_session_engine");
    let b = Database::open("memory://open_session_engine").unwrap();
    assert!(Arc::ptr_eq(a.engine(), b.engine()));
    a.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    assert_eq!(rows(&b), 1);
}

#[test]
fn test_an_open_outlives_the_handle_that_opened_it() {
    let a = table("memory://open_session_outlives_opener");
    let b = Database::open("memory://open_session_outlives_opener").unwrap();
    drop(a);
    b.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    assert_eq!(rows(&b), 1);
}

#[test]
fn test_a_clone_outlives_its_original() {
    let a = table("memory://open_session_clone_outlives");
    let b = a.clone();
    drop(a);
    b.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    assert_eq!(rows(&b), 1);
}

#[test]
fn test_the_engine_closes_when_its_last_handle_drops() {
    let a = table("memory://open_session_last_drop");
    let b = Database::open("memory://open_session_last_drop").unwrap();
    let c = b.clone();
    let engine = Arc::clone(a.engine());
    drop(a);
    drop(b);
    assert!(engine.is_open(), "c still holds the engine");
    drop(c);
    assert!(!engine.is_open());
    let reopened = Database::open("memory://open_session_last_drop").unwrap();
    assert!(!Arc::ptr_eq(reopened.engine(), &engine));
}
