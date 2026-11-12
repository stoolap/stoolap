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

//! A rollback drops the table its transaction created, whatever name that
//! table has by then, and never another table that took its old name

use stoolap::Database;

fn rows(db: &Database, table: &str) -> Option<i64> {
    db.query_one::<i64, _>(&format!("SELECT COUNT(*) FROM {table}"), ())
        .ok()
}

/// Statements of the transaction (A) and of another session (B), each
/// prefixed with its session, then the end of A
const CASES: &[(&[&str], &str)] = &[
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "A:ALTER TABLE n RENAME TO m",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
        ],
        "ROLLBACK",
    ),
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:DROP TABLE n",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
        ],
        "ROLLBACK",
    ),
    (
        &[
            "A:SAVEPOINT s",
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "A:ALTER TABLE n RENAME TO m",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
            "A:ROLLBACK TO SAVEPOINT s",
        ],
        "COMMIT",
    ),
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "A:SAVEPOINT s",
            "A:DROP TABLE n",
            "A:ROLLBACK TO SAVEPOINT s",
            "A:ALTER TABLE n RENAME TO m",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
        ],
        "ROLLBACK",
    ),
];

#[test]
fn a_rollback_drops_its_own_table_under_any_name() {
    let mut wrong = Vec::new();
    for (statements, end) in CASES {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}", dir.path().display());
        let a = Database::open(&dsn).unwrap();
        let b = a.clone();
        a.execute("BEGIN", ()).unwrap();
        for statement in *statements {
            let (session, sql) = statement.split_once(':').unwrap();
            let db = if session == "A" { &a } else { &b };
            db.execute(sql, ()).unwrap();
        }
        if let Err(e) = a.execute(end, ()) {
            wrong.push(format!("{statements:?}: {end} failed: {e}"));
        }
        let live = (rows(&b, "n"), rows(&b, "m"));
        drop(b);
        a.close().unwrap();
        let db = Database::open(&dsn).unwrap();
        let reopened = (rows(&db, "n"), rows(&db, "m"));
        for (when, (n, m)) in [("live", live), ("reopened", reopened)] {
            if n != Some(1) || m.is_some() {
                wrong.push(format!("{statements:?} {end}: {when} n {n:?} m {m:?}"));
            }
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
