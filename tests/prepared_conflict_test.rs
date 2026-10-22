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

//! A subquery in an INSERT's ON CONFLICT clause, or in the VALUES of an
//! INSERT with one, sees the tables as the statement found them

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

const REFUSED: &str = "inside a transaction";

/// Each statement with the table it leaves after `1:10 2:20 3:30 4:40`, and
/// after a transaction inserted `(7, 70)` first; None where a transaction
/// refuses it
const STATEMENTS: &[(&str, &str, Option<&str>)] = &[
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(b) FROM t)",
        "1:40 2:20 3:30 4:40",
        Some("1:70 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, 2) ON CONFLICT (id) DO UPDATE SET b = (SELECT x FROM u WHERE u.id = excluded.b)",
        "1:2 2:20 3:30 4:40",
        Some("1:2 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT COUNT(*) FROM t AS o WHERE o.b > t.b)",
        "1:3 2:20 3:30 4:40",
        None,
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = 99 WHERE EXISTS (SELECT 1 FROM u WHERE x = 2)",
        "1:99 2:20 3:30 4:40",
        Some("1:99 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = 99 WHERE EXISTS (SELECT 1 FROM u WHERE x = 5)",
        "1:10 2:20 3:30 4:40",
        Some("1:10 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, 0), (2, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT SUM(b) FROM t)",
        "1:100 2:100 3:30 4:40",
        Some("1:170 2:170 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, 0), (2, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT SUM(o.b) FROM t AS o WHERE o.id <> t.id)",
        "1:90 2:80 3:30 4:40",
        None,
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON DUPLICATE KEY UPDATE b = (SELECT MAX(b) FROM t)",
        "1:40 2:20 3:30 4:40",
        Some("1:70 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, (SELECT MAX(b) FROM t)) ON CONFLICT (id) DO UPDATE SET b = excluded.b + 1",
        "1:41 2:20 3:30 4:40",
        Some("1:71 2:20 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t VALUES (1, (SELECT x FROM u WHERE id = 1)), (9, (SELECT MAX(b) FROM t)) ON CONFLICT (id) DO NOTHING",
        "1:10 2:20 3:30 4:40 9:40",
        Some("1:10 2:20 3:30 4:40 7:70 9:70"),
    ),
    (
        "INSERT INTO t SELECT id, x FROM u ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(b) FROM t)",
        "1:40 2:40 3:30 4:40",
        Some("1:70 2:70 3:30 4:40 7:70"),
    ),
    (
        "INSERT INTO t SELECT id, x FROM u ON CONFLICT (id) DO UPDATE SET b = excluded.b + (SELECT x FROM u WHERE id = 2)",
        "1:3 2:4 3:30 4:40",
        Some("1:3 2:4 3:30 4:40 7:70"),
    ),
];

#[test]
fn an_on_conflict_subquery_reads_the_tables_as_the_statement_found_them() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        for context in ["autocommit", "transaction", "after a write"] {
            for (sql, left, left_after_write) in STATEMENTS {
                let dir = tempfile::tempdir().unwrap();
                let db = if sealed {
                    Database::open(&format!("file://{}", dir.path().display())).unwrap()
                } else {
                    Database::open("memory://prepared_conflict_statements").unwrap()
                };
                setup(&db);
                if sealed {
                    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                }
                let (got, want) = if context == "autocommit" {
                    let got = db.execute(sql, ()).map_err(|e| e.to_string());
                    (
                        got.map(|_| rows(db.query("SELECT id, b FROM t ORDER BY id", ()))),
                        Ok(left.to_string()),
                    )
                } else {
                    let mut tx = db.begin().unwrap();
                    let want = match (context, left_after_write) {
                        (_, None) => Err(REFUSED),
                        ("after a write", Some(left)) => Ok(left.to_string()),
                        _ => Ok(left.to_string()),
                    };
                    if context == "after a write" {
                        tx.execute("INSERT INTO t VALUES (7, 70)", ()).unwrap();
                    }
                    let got = tx.execute(sql, ()).map_err(|e| e.to_string());
                    let got = got.map(|_| rows(tx.query("SELECT id, b FROM t ORDER BY id", ())));
                    tx.rollback().unwrap();
                    (got, want)
                };
                let right = match (&got, &want) {
                    (Ok(got), Ok(want)) => got == want,
                    (Err(got), Err(want)) => got.contains(want),
                    _ => false,
                };
                if !right {
                    wrong.push(format!(
                        "{sql} (sealed: {sealed}, {context}): {got:?}, want {want:?}"
                    ));
                }
                db.execute("DROP TABLE t", ()).unwrap();
                db.execute("DROP TABLE u", ()).unwrap();
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn table(db: &Database) -> String {
    rows(db.query("SELECT id, b FROM t ORDER BY id", ()))
}

#[test]
fn a_prepared_upsert_reads_each_execution_s_tables_and_parameters() {
    let db = Database::open("memory://prepared_conflict_executions").unwrap();
    setup(&db);
    let upsert = db
        .prepare(
            "INSERT INTO t VALUES ($1, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT x FROM u WHERE u.id = $2) + excluded.b",
        )
        .unwrap();
    upsert.execute((1i64, 1i64)).unwrap();
    upsert.execute((2i64, 2i64)).unwrap();
    assert_eq!(table(&db), "1:1 2:2 3:30 4:40");
    db.execute("UPDATE u SET x = 7 WHERE id = 1", ()).unwrap();
    upsert.execute((3i64, 1i64)).unwrap();
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER", ())
        .unwrap();
    upsert.execute((4i64, 1i64)).unwrap();
    assert_eq!(table(&db), "1:1 2:2 3:7 4:7");
    let missing = db.execute(
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT x FROM u WHERE u.id = $1)",
        (),
    );
    assert!(
        missing
            .as_ref()
            .is_err_and(|e| e.to_string().contains("parameters")),
        "{missing:?}"
    );
    assert_eq!(table(&db), "1:1 2:2 3:7 4:7");
}

#[test]
fn an_on_conflict_subquery_binds_names_as_the_update_does() {
    let db = Database::open("memory://prepared_conflict_names").unwrap();
    setup(&db);
    for (set, want) in [
        ("b = (SELECT b + 1)", Some("1:11")),
        ("b = (SELECT t.b + excluded.b)", Some("1:15")),
        (
            "b = (SELECT MAX(x) FROM u WHERE u.id <= excluded.b)",
            Some("1:2"),
        ),
        ("b = (SELECT zz.b)", None),
        ("b = (SELECT u.b FROM u)", None),
    ] {
        db.execute("UPDATE t SET b = 10 WHERE id = 1", ()).unwrap();
        let sql = format!("INSERT INTO t VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET {set}");
        let got = db.execute(&sql, ()).map_err(|e| e.to_string());
        let row = rows(db.query("SELECT id, b FROM t WHERE id = 1", ()));
        match want {
            Some(want) => assert_eq!((got.is_ok(), row.as_str()), (true, want), "{sql}: {got:?}"),
            None => assert!(got.is_err() && row == "1:10", "{sql}: {got:?}, {row}"),
        }
    }
}

#[test]
fn a_failing_conflict_read_writes_nothing() {
    for context in ["autocommit", "transaction"] {
        let db = Database::open(&format!("memory://prepared_conflict_failing_{context}")).unwrap();
        setup(&db);
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER CHECK (v < 10))",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO c VALUES (1, 0)", ()).unwrap();
        let statements = [
            "INSERT INTO t VALUES (1, 0), (2, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT x FROM u WHERE u.id = 1 OR t.id = 2)",
            "INSERT INTO c VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET v = (SELECT 20)",
        ];
        let mut tx = (context == "transaction").then(|| db.begin().unwrap());
        for sql in statements {
            let got = match tx.as_mut() {
                Some(tx) => tx.execute(sql, ()),
                None => db.execute(sql, ()),
            };
            assert!(got.is_err(), "{sql} ({context}) wrote {got:?}");
        }
        let mut read = |sql: &str| match tx.as_mut() {
            Some(tx) => rows(tx.query(sql, ())),
            None => rows(db.query(sql, ())),
        };
        assert_eq!(
            read("SELECT id, b FROM t ORDER BY id"),
            "1:10 2:20 3:30 4:40",
            "{context}"
        );
        assert_eq!(read("SELECT id, v FROM c"), "1:0", "{context}");
    }
}

#[test]
fn an_upsert_with_a_conflict_subquery_returns_the_row_it_wrote() {
    let db = Database::open("memory://prepared_conflict_returning").unwrap();
    setup(&db);
    let got = rows(db.query(
        "INSERT INTO t VALUES (1, 0), (9, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(x) FROM u) RETURNING id, b",
        (),
    ));
    assert_eq!(got, "1:2 9:0");
    assert_eq!(table(&db), "1:2 2:20 3:30 4:40 9:0");
}

static COMMIT_ON_CALL: std::sync::Mutex<Option<Database>> = std::sync::Mutex::new(None);

#[derive(Default)]
struct CommitOnCall;

impl stoolap::functions::ScalarFunction for CommitOnCall {
    fn name(&self) -> &str {
        "COMMIT_ON_CALL"
    }

    fn info(&self) -> stoolap::functions::FunctionInfo {
        stoolap::functions::FunctionInfo {
            name: "COMMIT_ON_CALL".into(),
            ..stoolap::functions::ScalarFunction::info(&stoolap::functions::scalar::SleepFunction)
        }
    }

    fn evaluate(
        &self,
        args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        if let Some(db) = COMMIT_ON_CALL.lock().unwrap().take() {
            db.execute("UPDATE u SET x = 100", ()).unwrap();
        }
        Ok(args[0].clone())
    }

    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(Self)
    }
}

#[test]
fn a_conflict_subquery_does_not_see_a_commit_made_during_the_statement() {
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        let db = Database::open(&format!("memory://prepared_conflict_commit_{level}")).unwrap();
        setup(&db);
        let functions = stoolap::functions::FunctionRegistry::new();
        functions.register_scalar::<CommitOnCall>();
        let executor = stoolap::Executor::with_function_registry(
            std::sync::Arc::clone(db.engine()),
            std::sync::Arc::new(functions),
        );
        executor
            .execute(&format!("SET isolation_level = '{level}'"))
            .unwrap();
        *COMMIT_ON_CALL.lock().unwrap() = Some(db.clone());
        executor
            .execute(
                "INSERT INTO t VALUES (1, 7) ON CONFLICT (id) DO UPDATE SET b = \
                 COMMIT_ON_CALL(excluded.b) + (SELECT MAX(x) FROM u) \
                 + (SELECT x FROM u WHERE u.id = excluded.id)",
            )
            .unwrap();
        assert_eq!(
            table(&db),
            "1:10 2:20 3:30 4:40",
            "{level}: a subquery read a commit that landed after the statement began"
        );
    }
}

#[test]
fn a_kept_conflict_answer_is_not_one_read_before_a_commit() {
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        let db = Database::open(&format!("memory://prepared_conflict_kept_{level}")).unwrap();
        setup(&db);
        let functions = stoolap::functions::FunctionRegistry::new();
        functions.register_scalar::<CommitOnCall>();
        let executor = stoolap::Executor::with_function_registry(
            std::sync::Arc::clone(db.engine()),
            std::sync::Arc::new(functions),
        );
        executor
            .execute(&format!("SET isolation_level = '{level}'"))
            .unwrap();
        let upsert = "INSERT INTO t VALUES (1, 7) ON CONFLICT (id) DO UPDATE SET b = \
                      COMMIT_ON_CALL(excluded.b) + (SELECT MAX(x) FROM u)";
        *COMMIT_ON_CALL.lock().unwrap() = Some(db.clone());
        executor.execute(upsert).unwrap();
        assert_eq!(table(&db), "1:9 2:20 3:30 4:40", "{level}: first statement");
        executor.execute(upsert).unwrap();
        assert_eq!(
            table(&db),
            "1:107 2:20 3:30 4:40",
            "{level}: the second statement took an answer read before the commit"
        );
    }
}

#[test]
fn excluded_names_the_excluded_row_when_the_target_bears_that_name() {
    let db = Database::open("memory://prepared_conflict_excluded_target").unwrap();
    db.execute(
        "CREATE TABLE excluded (id INTEGER PRIMARY KEY, b INTEGER)",
        (),
    )
    .unwrap();
    for (set, want) in [
        ("b = (SELECT excluded.b)", "1:5"),
        ("b = (SELECT b + excluded.b)", "1:15"),
        ("b = b + excluded.b", "1:15"),
    ] {
        db.execute("DELETE FROM excluded", ()).unwrap();
        db.execute("INSERT INTO excluded VALUES (1, 10)", ())
            .unwrap();
        db.execute(
            &format!("INSERT INTO excluded VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET {set}"),
            (),
        )
        .unwrap();
        assert_eq!(
            rows(db.query("SELECT id, b FROM excluded", ())),
            want,
            "{set}"
        );
    }
}
