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

//! An error a WHERE raises on a row it reaches is the statement's error,
//! wherever the filter is evaluated: OR and NOT follow the expression
//! evaluator, left to right, stopping at the first TRUE

use stoolap::Database;

const ERR: &str = "error";

fn answer(db: &Database, sql: &str) -> String {
    if sql.starts_with("SELECT") {
        match db.query(sql, ()).and_then(|rows| {
            rows.map(|row| row.map(|row| format!("{:?}", row.as_row().as_slice())))
                .collect::<stoolap::Result<Vec<_>>>()
        }) {
            Ok(mut rows) => {
                rows.sort();
                rows.join(" ")
            }
            Err(_) => ERR.to_string(),
        }
    } else {
        match db.execute(sql, ()) {
            Ok(n) => format!("affected {n}"),
            Err(_) => ERR.to_string(),
        }
    }
}

const CASES: [(&str, &str); 17] = [
    ("SELECT id FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE ABS(x) >= 0 LIMIT 2", ERR),
    (
        "SELECT id FROM t WHERE ABS(x) >= 0 ORDER BY id LIMIT 1",
        ERR,
    ),
    ("SELECT COUNT(*) FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE id IN (1, 2) AND ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE k = 1 AND ABS(x) >= 0", ERR),
    (
        "SELECT id FROM t WHERE k = 1 AND ABS(x) >= 0 ORDER BY o LIMIT 1",
        ERR,
    ),
    ("SELECT id FROM t WHERE ABS(x) >= 0 OR id = 1", ERR),
    ("SELECT id FROM t WHERE NOT (ABS(x) < 0)", ERR),
    (
        "SELECT id FROM t WHERE id = 1 OR ABS(x) >= 0 ORDER BY id",
        "[Integer(1)] [Integer(2)]",
    ),
    (
        "SELECT id FROM t WHERE s = 'a' OR ABS(x) >= 0 LIMIT 2",
        "[Integer(1)] [Integer(2)]",
    ),
    (
        "SELECT id FROM t WHERE id = 2 AND ABS(x) >= 0",
        "[Integer(2)]",
    ),
    ("SELECT id FROM t WHERE UPPER(s) = 'A'", "[Integer(1)]"),
    (
        "SELECT id FROM t WHERE CAST(s AS INTEGER) > 0",
        "[Integer(2)]",
    ),
    ("UPDATE t SET k = 0 WHERE ABS(x) >= 0", ERR),
    ("DELETE FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT COUNT(*) FROM t WHERE k > 0", "[Integer(2)]"),
];

fn filled(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, s TEXT, k INTEGER, o INTEGER NOT NULL)",
        "CREATE INDEX tk ON t(k, o)",
        "CREATE INDEX ts ON t(s)",
        "INSERT INTO t VALUES (1, -9223372036854775808, 'a', 1, 1), (2, 5, '2', 2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

#[test]
fn a_filter_error_is_the_statements_on_hot_and_sealed_rows() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!("file://{}", dir.path().display())).unwrap()
        } else {
            Database::open("memory://storage_filter_error_rows").unwrap()
        };
        filled(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, expected) in CASES {
            let got = answer(&db, sql);
            if got != expected {
                wrong.push(format!("sealed {sealed}: {sql} gave {got}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_filter_error_is_the_statements_on_a_transactions_own_rows() {
    let db = Database::open("memory://storage_filter_error_own_rows").unwrap();
    filled(&db);
    db.execute("DELETE FROM t", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, -9223372036854775808, 'a', 1, 1), (2, 5, '2', 2, 2)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (sql, expected) in CASES {
        if expected == ERR && sql.starts_with("SELECT") {
            let got = answer(&db, sql);
            if got != expected {
                wrong.push(format!("{sql} gave {got}"));
            }
        }
    }
    db.execute("ROLLBACK", ()).unwrap();
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

const OWN_WRITES: [(&str, &str, &str); 12] = [
    (
        "DELETE FROM t WHERE id = 1",
        "SELECT id FROM t WHERE ABS(x) >= 0",
        "[Integer(2)]",
    ),
    (
        "DELETE FROM t WHERE id = 1",
        "SELECT id FROM t WHERE ABS(x) >= 0 LIMIT 2",
        "[Integer(2)]",
    ),
    (
        "DELETE FROM t WHERE id = 1",
        "SELECT COUNT(*) FROM t WHERE ABS(x) >= 0",
        "[Integer(1)]",
    ),
    (
        "DELETE FROM t WHERE id = 1",
        "UPDATE t SET o = 9 WHERE ABS(x) >= 0",
        "affected 1",
    ),
    (
        "DELETE FROM t WHERE id = 1",
        "DELETE FROM t WHERE ABS(x) >= 0",
        "affected 1",
    ),
    (
        "UPDATE t SET x = 0 WHERE id = 1",
        "SELECT id FROM t WHERE ABS(x) >= 0",
        "[Integer(1)] [Integer(2)]",
    ),
    (
        "UPDATE t SET x = 0 WHERE id = 1",
        "SELECT COUNT(*) FROM t WHERE ABS(x) >= 0",
        "[Integer(2)]",
    ),
    (
        "UPDATE t SET x = 0 WHERE id = 1",
        "UPDATE t SET o = 9 WHERE ABS(x) >= 0",
        "affected 2",
    ),
    (
        "UPDATE t SET x = 0 WHERE id = 1",
        "DELETE FROM t WHERE ABS(x) >= 0",
        "affected 2",
    ),
    (
        "UPDATE t SET x = 7 WHERE id = 2",
        "SELECT id FROM t WHERE x = 7",
        "[Integer(2)]",
    ),
    (
        "UPDATE t SET x = 7 WHERE id = 2",
        "UPDATE t SET o = 9 WHERE x = 7",
        "affected 1",
    ),
    (
        "UPDATE t SET x = 7 WHERE id = 2",
        "DELETE FROM t WHERE x = 7",
        "affected 1",
    ),
];

#[test]
fn a_filter_judges_the_rows_a_transaction_wrote_by_its_own_versions() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!("file://{}", dir.path().display())).unwrap()
        } else {
            Database::open("memory://storage_filter_error_written").unwrap()
        };
        filled(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (write, sql, expected) in OWN_WRITES {
            db.execute("BEGIN", ()).unwrap();
            db.execute(write, ()).unwrap();
            let got = answer(&db, sql);
            db.execute("ROLLBACK", ()).unwrap();
            if got != expected {
                wrong.push(format!("sealed {sealed}: {write}; {sql} gave {got}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_filter_error_is_the_statements_as_of_a_transaction() {
    let db = Database::open("memory://storage_filter_error_as_of").unwrap();
    filled(&db);
    db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let mut writer = db.begin().unwrap();
    let after_fill = writer.id();
    writer.execute("INSERT INTO z VALUES (1)", ()).unwrap();
    writer.commit().unwrap();
    let points = [
        "TIMESTAMP '2200-01-01 00:00:00'".to_string(),
        format!("TRANSACTION {after_fill}"),
    ];
    for as_of in &points {
        let all = answer(&db, &format!("SELECT id FROM t AS OF {as_of}"));
        let some = answer(
            &db,
            &format!("SELECT id FROM t AS OF {as_of} WHERE ABS(x) >= 0"),
        );
        assert_eq!(
            (all.as_str(), some.as_str()),
            ("[Integer(1)] [Integer(2)]", ERR),
            "{as_of}"
        );
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_row_a_cold_round_judged_is_not_asked_again_after_it_moves_hot() {
    use stoolap::test_failpoints::{after_cold_round_prepared, FailpointGuard};
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE p (x INTEGER)", ()).unwrap();
    db.execute("INSERT INTO p VALUES (1)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    let other = db.clone();
    after_cold_round_prepared(move || {
        let moved =
            std::thread::spawn(move || other.execute("UPDATE p SET x = -9223372036854775808", ()));
        assert_eq!(moved.join().unwrap().ok(), Some(1), "the other update");
    });
    let updated = db.execute("UPDATE p SET x = 2 WHERE ABS(x) >= 0 AND x + x < 0", ());
    assert_eq!(updated.ok(), Some(0));
}

mod written_by_the_filter {
    use std::cell::RefCell;
    use std::sync::Arc;

    use stoolap::core::{Operator, Result, Row, Value};
    use stoolap::functions::scalar::{AbsFunction, SleepFunction};
    use stoolap::functions::{FunctionInfo, ScalarFunction};
    use stoolap::storage::expression::{AndExpr, Expression, FunctionArg, FunctionExpr};
    use stoolap::storage::traits::{Engine, ScanAccess, Table};
    use stoolap::Database;

    thread_local! {
        static OTHER: RefCell<Option<Box<dyn Table>>> = const { RefCell::new(None) };
    }

    type Write = Box<dyn FnOnce(&mut dyn Table) -> Result<()> + Send>;

    thread_local! {
        static WRITE: RefCell<Option<Write>> = const { RefCell::new(None) };
    }

    /// Runs `WRITE` on another handle of the same transaction the first time
    /// it is asked, then answers 1
    struct WritesOnce;

    impl ScalarFunction for WritesOnce {
        fn name(&self) -> &str {
            "WRITES_ONCE"
        }

        fn info(&self) -> FunctionInfo {
            SleepFunction.info()
        }

        fn evaluate(&self, _args: &[Value]) -> Result<Value> {
            let table = OTHER.with(|other| other.borrow_mut().take());
            let write = WRITE.with(|write| write.borrow_mut().take());
            if let (Some(mut table), Some(write)) = (table, write) {
                write(table.as_mut())?;
            }
            Ok(Value::Integer(1))
        }

        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(Self)
        }
    }

    /// Runs `UPDATE w SET v = v` on `w (id, v, x)` holding `rows`, under
    /// `filter`, with `write` armed on a second handle of its transaction
    fn update_writing(
        dsn: &'static str,
        rows: &'static str,
        filter: fn() -> Box<dyn Expression>,
        write: Write,
    ) -> Option<std::result::Result<i32, String>> {
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let db = Database::open(dsn).unwrap();
            db.execute(
                "CREATE TABLE w (id INTEGER PRIMARY KEY, v INTEGER, x INTEGER)",
                (),
            )
            .unwrap();
            db.execute(&format!("INSERT INTO w VALUES {rows}"), ())
                .unwrap();
            let tx = db.engine().begin_transaction().unwrap();
            let mut first = tx.get_table("w").unwrap();
            OTHER.with(|other| *other.borrow_mut() = Some(tx.get_table("w").unwrap()));
            WRITE.with(|slot| *slot.borrow_mut() = Some(write));
            let mut filter = filter();
            filter.prepare_for_schema(first.schema());
            let plan = first.index_plan(filter.as_ref());
            let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
            let updated = first.update(access, Some(filter.as_ref()), &mut |row| Ok((row, true)));
            let _ = done.send(updated.map_err(|e| e.to_string()));
        });
        finished
            .recv_timeout(std::time::Duration::from_secs(10))
            .ok()
    }

    fn writes_once_on_v() -> FunctionExpr {
        FunctionExpr::new(
            Arc::new(WritesOnce),
            vec![FunctionArg::Column("v".to_string())],
            Operator::Eq,
            Value::Integer(1),
        )
    }

    #[test]
    fn a_filter_may_write_through_another_handle_of_its_transaction() {
        let updated = update_writing(
            "memory://storage_filter_writes_through",
            "(1, 1, 0), (2, 1, 0)",
            || Box::new(writes_once_on_v()),
            Box::new(|table| {
                table.insert(Row::from_values(vec![
                    Value::Integer(9),
                    Value::Integer(1),
                    Value::Integer(0),
                ]))?;
                Ok(())
            }),
        );
        assert!(updated.is_some(), "the UPDATE did not finish");
        assert!(
            matches!(updated, Some(Ok(_))),
            "the UPDATE failed: {updated:?}"
        );
    }

    #[test]
    fn a_row_the_filter_deletes_is_not_asked_again() {
        let updated = update_writing(
            "memory://storage_filter_deletes_through",
            "(1, 1, 0), (2, 1, -9223372036854775808)",
            || {
                Box::new(AndExpr::new(vec![
                    Box::new(writes_once_on_v()),
                    Box::new(FunctionExpr::new(
                        Arc::new(AbsFunction),
                        vec![FunctionArg::Column("x".to_string())],
                        Operator::Gte,
                        Value::Integer(0),
                    )),
                ]))
            },
            Box::new(|table| {
                table.delete_by_row_ids(&[2])?;
                Ok(())
            }),
        );
        assert_eq!(updated, Some(Ok(1)));
    }
}
