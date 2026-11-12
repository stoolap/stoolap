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
/// prefixed with its session, the end of A, then the rows of n and of m
type Case = (
    &'static [&'static str],
    &'static str,
    Option<i64>,
    Option<i64>,
);

const CASES: &[Case] = &[
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "A:ALTER TABLE n RENAME TO m",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
        ],
        "ROLLBACK",
        Some(1),
        None,
    ),
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:DROP TABLE n",
            "B:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:INSERT INTO n VALUES (1)",
        ],
        "ROLLBACK",
        Some(1),
        None,
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
        Some(1),
        None,
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
        Some(1),
        None,
    ),
    (
        &[
            "A:CREATE TABLE n (id INTEGER PRIMARY KEY)",
            "B:BEGIN",
            "B:DROP TABLE n",
            "B:ROLLBACK",
        ],
        "ROLLBACK",
        None,
        None,
    ),
];

#[test]
fn a_rollback_drops_its_own_table_under_any_name() {
    // The failpoint tests here fail WAL syncs process-wide
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let mut wrong = Vec::new();
    for (statements, end, want_n, want_m) in CASES {
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
            if (n, m) != (*want_n, *want_m) {
                wrong.push(format!("{statements:?} {end}: {when} n {n:?} m {m:?}"));
            }
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_create_of_the_old_name_waits_for_the_rollback_that_drops_it() {
    use std::sync::mpsc;
    use std::time::Duration;

    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let a = Database::open(&dsn).unwrap();
    let b = a.clone();
    a.execute("BEGIN", ()).unwrap();
    a.execute("CREATE TABLE n (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let creator = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&creator);
    stoolap::test_failpoints::after_table_unlisted(move || {
        let (created, waiting) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            b.execute("CREATE TABLE n (id INTEGER PRIMARY KEY)", ())?;
            let _ = created.send(());
            b.execute("INSERT INTO n VALUES (1)", ())
        });
        // Long enough for a create that does not wait to land
        let _ = waiting.recv_timeout(Duration::from_millis(500));
        *slot.borrow_mut() = Some(handle);
    });
    a.execute("ROLLBACK", ()).unwrap();
    let written = creator.borrow_mut().take().unwrap().join().unwrap();
    assert!(written.is_ok(), "{written:?}");
    assert_eq!(rows(&a, "n"), Some(1), "the other session's n went");
    a.close().unwrap();
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        rows(&db, "n"),
        Some(1),
        "the other session's n went after a reopen"
    );
    db.close().unwrap();
}
