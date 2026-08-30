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

//! Prepared-INSERT fast paths: direct parameter resolution, pre-compiled
//! CHECK constraints, and per-statement DEFAULT evaluation.

use stoolap::api::Database;

#[test]
fn prepared_insert_parameters_roundtrip() {
    let db = Database::open("memory://insert_path_params").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score FLOAT)",
        (),
    )
    .unwrap();

    let stmt = db.prepare("INSERT INTO t VALUES ($1, $2, $3, $4)").unwrap();
    stmt.execute((1i64, "alice", 30i64, 1.5f64)).unwrap();
    stmt.execute((2i64, "bob", Option::<i64>::None, 2.5f64))
        .unwrap();

    let name: String = db.query_one("SELECT name FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(name, "alice");
    let age: Option<i64> = db
        .query_opt("SELECT age FROM t WHERE id = 2 AND age IS NOT NULL", ())
        .unwrap();
    assert!(age.is_none());
    let s: f64 = db
        .query_one("SELECT score FROM t WHERE id = 2", ())
        .unwrap();
    assert_eq!(s, 2.5);
}

#[test]
fn prepared_insert_check_constraints_still_enforced() {
    let db = Database::open("memory://insert_path_check").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, age INTEGER CHECK (age >= 0))",
        (),
    )
    .unwrap();

    let stmt = db.prepare("INSERT INTO t VALUES ($1, $2)").unwrap();
    // Passing row inserts
    stmt.execute((1i64, 30i64)).unwrap();
    // Violation must error and mention the column; nothing inserted
    let err = stmt.execute((2i64, -5i64)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("age"), "error should name the column: {msg}");
    // NULL passes CHECK per the SQL standard
    stmt.execute((3i64, Option::<i64>::None)).unwrap();

    let n: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 2);
    // The compiled plan still enforces on later executions
    assert!(stmt.execute((4i64, -1i64)).is_err());
}

#[test]
fn insert_defaults_apply_from_statement_template() {
    let db = Database::open("memory://insert_path_default").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT DEFAULT 'new', score INTEGER DEFAULT 7)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t (id) VALUES (1), (2), (3)", ())
        .unwrap();
    let n: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE status = 'new' AND score = 7",
            (),
        )
        .unwrap();
    assert_eq!(n, 3);

    // DEFAULT keyword in the VALUES list uses the same template
    db.execute("INSERT INTO t VALUES (4, DEFAULT, DEFAULT)", ())
        .unwrap();
    let s: String = db
        .query_one("SELECT status FROM t WHERE id = 4", ())
        .unwrap();
    assert_eq!(s, "new");
}
