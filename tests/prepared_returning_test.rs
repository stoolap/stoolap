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

//! A subquery in RETURNING sees the tables as the statement found them, and
//! a column of the target in it names the row returned

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

fn rows(result: stoolap::Result<stoolap::Rows>) -> stoolap::Result<String> {
    let mut out = Vec::new();
    for r in result? {
        let r = r?;
        let v: Option<i64> = r.get(1)?;
        out.push(format!(
            "{}:{}",
            r.get::<i64>(0)?,
            v.map_or("NULL".into(), |v| v.to_string())
        ));
    }
    out.sort();
    Ok(out.join(" "))
}

const REFUSED: &str = "inside a transaction";

/// Each statement with the rows it returns from `1:10 2:20 3:30 4:40`, and
/// after a transaction inserted `(7, 70)` first; None where a transaction
/// refuses it
const STATEMENTS: &[(&str, &str, Option<&str>)] = &[
    (
        "INSERT INTO t VALUES (5, 1) RETURNING id, (SELECT MAX(x) FROM u)",
        "5:2",
        Some("5:2"),
    ),
    (
        "INSERT INTO t VALUES (5, 2), (6, 1) RETURNING id, (SELECT x FROM u WHERE u.id = t.b)",
        "5:2 6:1",
        Some("5:2 6:1"),
    ),
    (
        "INSERT INTO t VALUES (5, 0), (6, 0) RETURNING id, (SELECT COUNT(*) FROM t)",
        "5:4 6:4",
        Some("5:5 6:5"),
    ),
    (
        "INSERT INTO t VALUES (5, 25) RETURNING id, (SELECT COUNT(*) FROM t AS o WHERE o.b < t.b)",
        "5:2",
        None,
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = 2 RETURNING id, (SELECT x FROM u WHERE u.id = t.b)",
        "1:2",
        Some("1:2"),
    ),
    (
        "INSERT INTO t VALUES (1, 0) ON CONFLICT (id) DO UPDATE SET b = 99 RETURNING id, (SELECT o.b FROM t AS o WHERE o.id = t.id)",
        "1:10",
        None,
    ),
    (
        "INSERT INTO t VALUES (1, 0), (5, 0) ON CONFLICT (id) DO NOTHING RETURNING id, (SELECT MAX(b) FROM t)",
        "5:40",
        Some("5:70"),
    ),
    (
        "INSERT INTO t SELECT id + 4, x FROM u RETURNING id, (SELECT SUM(x) FROM u)",
        "5:3 6:3",
        Some("5:3 6:3"),
    ),
    (
        "UPDATE t SET b = b + 1 WHERE id <= 2 RETURNING id, (SELECT x FROM u WHERE u.id = t.id)",
        "1:1 2:2",
        Some("1:1 2:2"),
    ),
    (
        "UPDATE t SET b = 2 WHERE id = 3 RETURNING id, (SELECT x FROM u WHERE u.id = t.b)",
        "3:2",
        Some("3:2"),
    ),
    (
        "UPDATE t SET b = 0 WHERE id = 1 RETURNING id, (SELECT SUM(b) FROM t)",
        "1:100",
        Some("1:170"),
    ),
    (
        "UPDATE t SET b = 0 WHERE id = 1 RETURNING id, (SELECT o.b FROM t AS o WHERE o.id = t.id)",
        "1:10",
        None,
    ),
    (
        "UPDATE t SET b = (SELECT MAX(x) FROM u) WHERE id = 1 RETURNING id, b + (SELECT MIN(x) FROM u)",
        "1:3",
        Some("1:3"),
    ),
    (
        "DELETE FROM t WHERE id = 1 RETURNING id, (SELECT MAX(x) FROM u)",
        "1:2",
        Some("1:2"),
    ),
    (
        "DELETE FROM t WHERE id <= 2 RETURNING id, (SELECT COUNT(*) FROM t)",
        "1:4 2:4",
        Some("1:5 2:5"),
    ),
    (
        "DELETE FROM t WHERE id = 2 RETURNING id, (SELECT COUNT(*) FROM t AS o WHERE o.b < t.b)",
        "2:1",
        None,
    ),
    (
        "DELETE FROM t WHERE id IN (SELECT id FROM u) RETURNING id, (SELECT x FROM u WHERE u.id = t.id)",
        "1:1 2:2",
        Some("1:1 2:2"),
    ),
];

#[test]
fn a_returning_subquery_reads_the_tables_as_the_statement_found_them() {
    let mut wrong = Vec::new();
    for (sealed, level) in [
        (false, "READ COMMITTED"),
        (false, "SNAPSHOT"),
        (true, "READ COMMITTED"),
    ] {
        for context in ["autocommit", "transaction", "after a write"] {
            for (sql, returned, returned_after_write) in STATEMENTS {
                let dir = tempfile::tempdir().unwrap();
                let db = if sealed {
                    Database::open(&format!("file://{}", dir.path().display())).unwrap()
                } else {
                    Database::open("memory://prepared_returning_statements").unwrap()
                };
                setup(&db);
                if sealed {
                    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                }
                db.execute(&format!("SET isolation_level = '{level}'"), ())
                    .unwrap();
                let (got, want) = if context == "autocommit" {
                    let got = rows(db.query(sql, ())).map_err(|e| e.to_string());
                    (got, Ok(returned.to_string()))
                } else {
                    let mut tx = db.begin().unwrap();
                    let want = match (context, returned_after_write) {
                        (_, None) => Err(REFUSED),
                        ("after a write", Some(after)) => Ok(after.to_string()),
                        _ => Ok(returned.to_string()),
                    };
                    if context == "after a write" {
                        tx.execute("INSERT INTO t VALUES (7, 70)", ()).unwrap();
                    }
                    let before = rows(tx.query("SELECT id, b FROM t", ())).unwrap();
                    let got = rows(tx.query(sql, ())).map_err(|e| e.to_string());
                    if got.is_err() {
                        let after = rows(tx.query("SELECT id, b FROM t", ())).unwrap();
                        if after != before {
                            wrong.push(format!("{sql} ({context}) failed but wrote: {after}"));
                        }
                    }
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
                        "{sql} (sealed: {sealed}, {level}, {context}): {got:?}, want {want:?}"
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
    rows(db.query("SELECT id, b FROM t", ())).unwrap()
}

#[test]
fn a_prepared_returning_reads_each_execution_s_tables_and_parameters() {
    let db = Database::open("memory://prepared_returning_executions").unwrap();
    setup(&db);
    let update = db
        .prepare(
            "UPDATE t SET b = $1 WHERE id = $2 RETURNING id, (SELECT x FROM u WHERE u.id = $1)",
        )
        .unwrap();
    let returned = |b: i64, id: i64| rows(update.query((b, id))).unwrap();
    assert_eq!(returned(1, 1), "1:1");
    assert_eq!(returned(2, 2), "2:2");
    db.execute("UPDATE u SET x = 7 WHERE id = 1", ()).unwrap();
    assert_eq!(returned(1, 3), "3:7");
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER", ())
        .unwrap();
    assert_eq!(returned(1, 4), "4:7");
    assert_eq!(table(&db), "1:1 2:2 3:1 4:1");

    let insert = db
        .prepare(
            "INSERT INTO t (id, b) VALUES ($1, 2) RETURNING *, (SELECT x FROM u WHERE u.id = t.b)",
        )
        .unwrap();
    let names = |result: stoolap::Rows| result.columns().to_vec();
    assert_eq!(
        names(insert.query((5i64,)).unwrap()),
        ["id", "b", "column_1"]
    );
    db.execute("ALTER TABLE t ADD COLUMN c INTEGER", ())
        .unwrap();
    let result = insert.query((6i64,)).unwrap();
    assert_eq!(result.columns(), ["id", "b", "c", "column_1"]);
    let row = result.into_iter().next().unwrap().unwrap();
    let got: (i64, i64, Option<i64>, i64) = (
        row.get(0).unwrap(),
        row.get(1).unwrap(),
        row.get(2).unwrap(),
        row.get(3).unwrap(),
    );
    assert_eq!(got, (6, 2, None, 2));
}

#[test]
fn a_returning_subquery_binds_names_as_the_statement_does() {
    let db = Database::open("memory://prepared_returning_names").unwrap();
    setup(&db);
    for (sql, want) in [
        (
            "UPDATE t SET b = 5 WHERE id = 1 RETURNING id, (SELECT b + 1)",
            Some("1:6"),
        ),
        (
            "UPDATE t SET b = 5 WHERE id = 1 RETURNING id, (SELECT t.b + x FROM u WHERE u.id = 2)",
            Some("1:7"),
        ),
        (
            "DELETE FROM t AS d WHERE id = 1 RETURNING id, (SELECT x FROM u WHERE u.id = d.id)",
            Some("1:1"),
        ),
        (
            "INSERT INTO t VALUES (1, 2) RETURNING id, (SELECT MAX(x) FROM u WHERE u.id <= b)",
            Some("1:2"),
        ),
        (
            "UPDATE t SET b = 5 WHERE id = 1 RETURNING id, (SELECT zz.b)",
            None,
        ),
        (
            "DELETE FROM t WHERE id = 1 RETURNING id, (SELECT u.b FROM u)",
            None,
        ),
    ] {
        db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
        if !sql.starts_with("INSERT") {
            db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
        }
        let before = table(&db);
        let got = rows(db.query(sql, ())).map_err(|e| e.to_string());
        match want {
            Some(want) => assert_eq!(got.as_deref(), Ok(want), "{sql}"),
            None => assert!(got.is_err() && table(&db) == before, "{sql}: {got:?}"),
        }
    }
}

#[test]
fn a_failing_returning_read_writes_nothing() {
    for context in ["autocommit", "transaction"] {
        let db = Database::open(&format!("memory://prepared_returning_failing_{context}")).unwrap();
        setup(&db);
        let statements = [
            "INSERT INTO t VALUES (5, 1), (6, 2) RETURNING id, (SELECT x FROM u WHERE u.id <= t.b)",
            "UPDATE t SET b = 0 RETURNING id, (SELECT x FROM u WHERE u.id <= t.id)",
            "DELETE FROM t RETURNING id, (SELECT x FROM u WHERE u.id <= t.id)",
        ];
        let mut tx = (context == "transaction").then(|| db.begin().unwrap());
        for sql in statements {
            let got = match tx.as_mut() {
                Some(tx) => rows(tx.query(sql, ())),
                None => rows(db.query(sql, ())),
            };
            assert!(got.is_err(), "{sql} ({context}) returned {got:?}");
        }
        let left = match tx.as_mut() {
            Some(tx) => rows(tx.query("SELECT id, b FROM t", ())),
            None => rows(db.query("SELECT id, b FROM t", ())),
        };
        assert_eq!(left.unwrap(), "1:10 2:20 3:30 4:40", "{context}");
    }
}
