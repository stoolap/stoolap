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

//! A subquery in the VALUES of an INSERT sees the tables as the statement
//! found them, the target included: before any row the statement writes,
//! after every row its transaction wrote before it

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(result: stoolap::Result<stoolap::Rows>) -> String {
    result
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let b: Option<i64> = r.get(1).unwrap();
            format!(
                "{}:{}",
                r.get::<i64>(0).unwrap(),
                b.map_or("NULL".into(), |b| b.to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Each statement with the rows it adds after `1:10 2:20 3:30 4:40`, in
/// autocommit or a fresh transaction, and after the transaction inserted
/// `(7, 70)` first
const STATEMENTS: &[(&str, &str, &str)] = &[
    (
        "INSERT INTO t VALUES (10, (SELECT MAX(b) FROM t))",
        "10:40",
        "10:70",
    ),
    (
        "INSERT INTO t VALUES (10, (SELECT u.x FROM u WHERE u.id = 1))",
        "10:1",
        "10:1",
    ),
    (
        "INSERT INTO t VALUES (10, (SELECT COUNT(*) FROM t)), (11, (SELECT COUNT(*) FROM t))",
        "10:4 11:4",
        "10:5 11:5",
    ),
    (
        "INSERT INTO t VALUES (10, CASE WHEN EXISTS (SELECT 1 FROM t WHERE id = 10) THEN 1 ELSE 0 END), (11, CASE WHEN EXISTS (SELECT 1 FROM t WHERE id = 10) THEN 1 ELSE 0 END)",
        "10:0 11:0",
        "10:0 11:0",
    ),
    (
        "INSERT INTO t (id, b) VALUES (10, (SELECT MIN(b) FROM t) + 1), (11, 5)",
        "10:11 11:5",
        "10:11 11:5",
    ),
];

#[test]
fn a_values_subquery_reads_the_tables_as_the_statement_found_them() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        for context in ["autocommit", "transaction", "after a write"] {
            for (sql, added, added_after_write) in STATEMENTS {
                let dir = tempfile::tempdir().unwrap();
                let db = if sealed {
                    Database::open(&format!("file://{}", dir.path().display())).unwrap()
                } else {
                    Database::open("memory://prepared_insert_values").unwrap()
                };
                setup(&db);
                if sealed {
                    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                }
                let (got, want) = if context == "autocommit" {
                    let got = db.execute(sql, ()).map_err(|e| e.to_string());
                    (
                        got.map(|_| rows(db.query("SELECT id, b FROM t ORDER BY id", ()))),
                        format!("1:10 2:20 3:30 4:40 {added}"),
                    )
                } else {
                    let mut tx = db.begin().unwrap();
                    let want = if context == "after a write" {
                        tx.execute("INSERT INTO t VALUES (7, 70)", ()).unwrap();
                        format!("1:10 2:20 3:30 4:40 7:70 {added_after_write}")
                    } else {
                        format!("1:10 2:20 3:30 4:40 {added}")
                    };
                    let got = tx.execute(sql, ()).map_err(|e| e.to_string());
                    let got = got.map(|_| rows(tx.query("SELECT id, b FROM t ORDER BY id", ())));
                    tx.rollback().unwrap();
                    (got, want)
                };
                if got.as_deref() != Ok(want.as_str()) {
                    wrong.push(format!("{sql} (sealed: {sealed}, {context}): {got:?}"));
                }
                db.execute("DROP TABLE t", ()).unwrap();
                db.execute("DROP TABLE u", ()).unwrap();
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_two_row_values_subquery_fails_the_insert_and_writes_nothing() {
    let db = Database::open("memory://prepared_insert_two_rows").unwrap();
    setup(&db);
    let sql = "INSERT INTO t VALUES (10, 0), (11, (SELECT x FROM u))";
    let two_rows = |result: stoolap::Result<i64>| matches!(result, Err(e) if e.to_string().contains("more than one row"));
    assert!(
        two_rows(db.execute(sql, ())),
        "{sql} did not fail on its two rows"
    );
    assert_eq!(
        rows(db.query("SELECT id, b FROM t ORDER BY id", ())),
        "1:10 2:20 3:30 4:40"
    );
    let mut tx = db.begin().unwrap();
    assert!(
        two_rows(tx.execute(sql, ())),
        "{sql} did not fail on its two rows"
    );
    assert_eq!(
        rows(tx.query("SELECT id, b FROM t ORDER BY id", ())),
        "1:10 2:20 3:30 4:40",
        "the failed insert wrote inside its transaction"
    );
    tx.rollback().unwrap();
}

#[test]
fn a_prepared_insert_reads_each_execution_s_tables_and_parameters() {
    let db = Database::open("memory://prepared_insert_executions").unwrap();
    setup(&db);
    let insert = db
        .prepare("INSERT INTO t VALUES ($1, (SELECT MAX(b) FROM t) + $2)")
        .unwrap();
    for (id, add) in [(10i64, 1i64), (11, 2), (12, 3)] {
        insert.execute((id, add)).unwrap();
    }
    assert_eq!(
        rows(db.query("SELECT id, b FROM t WHERE id >= 10 ORDER BY id", ())),
        "10:41 11:43 12:46"
    );
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER", ())
        .unwrap();
    insert.execute((13i64, 0i64)).unwrap();
    assert_eq!(
        rows(db.query("SELECT id, b FROM t WHERE id = 13", ())),
        "13:46"
    );
}

#[test]
fn a_values_subquery_fills_its_column_beside_the_defaults() {
    let db = Database::open("memory://prepared_insert_defaults").unwrap();
    setup(&db);
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, b INTEGER DEFAULT 7, c INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO d (id, c) VALUES (1, (SELECT COUNT(*) FROM t)), (2, DEFAULT)",
        (),
    )
    .unwrap();
    let got: Vec<String> = db
        .query("SELECT id, b, c FROM d ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let c: Option<i64> = r.get(2).unwrap();
            format!(
                "{}:{}:{}",
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                c.map_or("NULL".into(), |c| c.to_string())
            )
        })
        .collect();
    assert_eq!(got.join(" "), "1:7:4 2:7:NULL");
}

#[test]
fn a_missing_parameter_fails_beside_a_values_subquery() {
    let db = Database::open("memory://prepared_insert_missing_parameter").unwrap();
    setup(&db);
    for _ in 0..2 {
        let got = db.execute(
            "INSERT INTO t VALUES ($1, (SELECT x + 40 FROM u WHERE id = 1))",
            (),
        );
        let message = got.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("parameters"), "{message:?}");
    }
    assert_eq!(
        rows(db.query("SELECT id, b FROM t ORDER BY id", ())),
        "1:10 2:20 3:30 4:40"
    );
}

#[test]
fn a_values_subquery_reads_no_column_of_the_target() {
    let db = Database::open("memory://prepared_insert_no_target_row").unwrap();
    setup(&db);
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, b TEXT)", ())
        .unwrap();
    for (id, value, want) in [
        (1, "(SELECT \"b\" FROM u WHERE u.id = 1)", Some("b")),
        (2, "(SELECT s.id FROM u WHERE u.id = 1)", None),
        (3, "(SELECT b FROM u WHERE u.id = 1)", None),
        (4, "b || (SELECT 'x')", None),
    ] {
        let insert = db.execute(&format!("INSERT INTO s VALUES ({id}, {value})"), ());
        let got: Option<Option<String>> = db
            .query(&format!("SELECT b FROM s WHERE id = {id}"), ())
            .unwrap()
            .next()
            .map(|r| r.unwrap().get(0).unwrap());
        match want {
            Some(want) => assert_eq!(got, Some(Some(want.to_string())), "{value}: {insert:?}"),
            None => assert!(insert.is_err() && got.is_none(), "{value} inserted {got:?}"),
        }
    }
}

/// Each value with its parameter and the `b` it inserts
const ENTRY_SHAPES: &[(&str, i64, &str)] = &[
    ("LENGTH(\"b\") + (SELECT x FROM u WHERE id = 1)", 0, "2"),
    ("(SELECT x FROM u WHERE u.id = $1 + 1)", 0, "1"),
    ("(SELECT x FROM u WHERE u.id = $1)", 2, "2"),
    ("LENGTH(\"b\") + (SELECT x FROM u WHERE id = 1)", 0, "2"),
    ("$1 + (SELECT COUNT(*) FROM t)", 5, "9"),
];

#[test]
fn a_values_subquery_answers_alike_on_every_entry() {
    for (entry, run) in [("autocommit", 0), ("prepared", 1), ("transaction", 2)] {
        let db = Database::open(&format!("memory://prepared_insert_entry_{entry}")).unwrap();
        setup(&db);
        for (value, param, want) in ENTRY_SHAPES {
            let sql = format!("INSERT INTO t VALUES (10, {value})");
            let inserted = match run {
                0 => db.execute(&sql, (*param,)),
                1 => db.prepare(&sql).and_then(|s| s.execute((*param,))),
                _ => {
                    let mut tx = db.begin().unwrap();
                    let n = tx.execute(&sql, (*param,));
                    tx.commit().unwrap();
                    n
                }
            };
            assert_eq!(
                inserted.as_ref().ok(),
                Some(&1),
                "{entry} {value}: {inserted:?}"
            );
            assert_eq!(
                rows(db.query("SELECT id, b FROM t WHERE id = 10", ())),
                format!("10:{want}"),
                "{entry} {value}"
            );
            db.execute("DELETE FROM t WHERE id = 10", ()).unwrap();
        }
    }
}

#[test]
fn returning_reads_the_parameters_the_statement_was_given() {
    let db = Database::open("memory://prepared_insert_returning_parameters").unwrap();
    setup(&db);
    let returned = |sql: &str| -> Option<i64> {
        let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
        row.get(0).unwrap()
    };
    let literal = returned("INSERT INTO t VALUES (10, 41) RETURNING $1");
    let read =
        returned("INSERT INTO t VALUES (11, (SELECT x + 40 FROM u WHERE id = 1)) RETURNING $1");
    assert_eq!(read, literal);
    assert_eq!(
        rows(db.query("SELECT id, b FROM t WHERE id >= 10 ORDER BY id", ())),
        "10:41 11:41"
    );
}

static WRITE_THEN_FAIL: std::sync::Mutex<Option<Box<dyn stoolap::storage::traits::Table>>> =
    std::sync::Mutex::new(None);

#[derive(Default)]
struct WriteThenFail;

impl stoolap::functions::ScalarFunction for WriteThenFail {
    fn name(&self) -> &str {
        "WRITE_THEN_FAIL"
    }

    fn info(&self) -> stoolap::functions::FunctionInfo {
        stoolap::functions::FunctionInfo {
            name: "WRITE_THEN_FAIL".into(),
            ..stoolap::functions::ScalarFunction::info(&stoolap::functions::scalar::SleepFunction)
        }
    }

    fn evaluate(
        &self,
        _args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        if let Some(mut w) = WRITE_THEN_FAIL.lock().unwrap().take() {
            w.insert(stoolap::core::Row::from_values(vec![
                stoolap::core::Value::Integer(1),
            ]))
            .unwrap();
        }
        Err(stoolap::Error::internal("the value failed after a write"))
    }

    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(Self)
    }
}

#[test]
fn a_values_read_that_fails_takes_back_what_it_wrote_in_the_transaction() {
    let db = Database::open("memory://prepared_insert_values_rollback").unwrap();
    setup(&db);
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let functions = stoolap::functions::FunctionRegistry::new();
    functions.register_scalar::<WriteThenFail>();
    let executor = stoolap::Executor::with_function_registry(
        std::sync::Arc::clone(db.engine()),
        std::sync::Arc::new(functions),
    );
    let tx = executor.begin_transaction().unwrap();
    *WRITE_THEN_FAIL.lock().unwrap() = Some(tx.get_table("w").unwrap());
    executor.install_transaction(tx);
    let statement = stoolap::parser::parse_sql(
        "INSERT INTO t VALUES (10, (SELECT WRITE_THEN_FAIL(x) FROM u WHERE id = 1))",
    )
    .unwrap()
    .remove(0);
    let inserted = executor.execute_statement(&statement, &stoolap::ExecutionContext::new());
    let message = inserted.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(message.contains("failed after a write"), "{message:?}");
    let mut count = executor.execute("SELECT COUNT(*) FROM w").unwrap();
    assert!(count.next());
    let written = format!("{:?}", count.row().get(0));
    drop(count);
    executor.take_transaction().unwrap().rollback().unwrap();
    assert_eq!(
        written, "Some(Integer(0))",
        "the failed statement's write stayed"
    );
}
