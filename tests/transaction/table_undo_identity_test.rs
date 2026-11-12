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
        // The create either waits for the rollback's guard or lands now
        let (reached, settled) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let waits = reached.clone();
            stoolap::test_failpoints::on_ddl_guard_waiting(move || {
                let _ = waits.send("waits");
            });
            b.execute("CREATE TABLE n (id INTEGER PRIMARY KEY)", ())?;
            let _ = reached.send("landed");
            b.execute("INSERT INTO n VALUES (1)", ())
        });
        settled.recv().unwrap();
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

thread_local! {
    static IN_SELECT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs the armed callback once, then answers its argument
#[derive(Default)]
struct InSelect;

impl stoolap::functions::ScalarFunction for InSelect {
    fn name(&self) -> &str {
        "TABLE_UNDO_IN_SELECT"
    }
    fn info(&self) -> stoolap::functions::FunctionInfo {
        use stoolap::functions::{FunctionDataType, FunctionInfo, FunctionSignature, FunctionType};
        FunctionInfo::new(
            "TABLE_UNDO_IN_SELECT",
            FunctionType::Scalar,
            "runs the armed callback once",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
        .non_deterministic()
    }
    fn evaluate(
        &self,
        args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        if let Some(run) = IN_SELECT.with(|slot| slot.borrow_mut().take()) {
            run();
        }
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(InSelect)
    }
}

#[test]
fn a_rollback_called_from_a_create_as_select_runs_under_its_guard() {
    use std::sync::mpsc;
    use std::time::Duration;

    // The failpoint tests here fail WAL syncs process-wide
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    stoolap::functions::registry::global_registry()
        .register_scalar::<InSelect>()
        .unwrap();
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let dir = tempfile::tempdir().unwrap();
        let a = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        let b = a.clone();
        a.execute("CREATE TABLE seed (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        a.execute("INSERT INTO seed VALUES (1)", ()).unwrap();
        b.execute("BEGIN", ()).unwrap();
        b.execute("CREATE TABLE n (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        IN_SELECT.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                b.execute("ROLLBACK", ()).unwrap();
            }))
        });
        let created = a
            .execute(
                "CREATE TABLE c AS SELECT TABLE_UNDO_IN_SELECT(id) AS id FROM seed",
                (),
            )
            .map(|_| ());
        let _ = done.send((created, rows(&a, "c"), rows(&a, "n")));
    });
    let (created, c, n) = finished
        .recv_timeout(Duration::from_secs(30))
        .expect("the rollback waited for the guard its own thread holds");
    assert!(created.is_ok(), "{created:?}");
    assert_eq!(c, Some(1), "c holds the selected row");
    assert_eq!(n, None, "the rolled-back table stayed");
}
