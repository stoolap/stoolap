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

//! A subquery in a LIMIT or OFFSET is read once, before the statement reads
//! a row, and counts as its value written as a literal would

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 1)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
        "CREATE VIEW tv AS SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| format!("{:?}", row.get::<stoolap::core::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join(","))
        })
        .collect()
}

/// Each statement with a LIMIT or OFFSET subquery, and the same statement
/// with its value written as a literal
const TWINS: &[(&str, &str)] = &[
    (
        "SELECT id FROM t ORDER BY id LIMIT (SELECT 2)",
        "SELECT id FROM t ORDER BY id LIMIT 2",
    ),
    (
        "SELECT id FROM t ORDER BY id OFFSET (SELECT 2)",
        "SELECT id FROM t ORDER BY id OFFSET 2",
    ),
    (
        "SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u) OFFSET (SELECT MIN(x) FROM u)",
        "SELECT id FROM t ORDER BY id LIMIT 2 OFFSET 1",
    ),
    (
        "SELECT id FROM t ORDER BY id LIMIT 1 + (SELECT 1)",
        "SELECT id FROM t ORDER BY id LIMIT 2",
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a) ORDER BY id LIMIT (SELECT 2)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a) ORDER BY id LIMIT 2",
    ),
    (
        "SELECT id FROM t WHERE id = 1 LIMIT (SELECT 0)",
        "SELECT id FROM t WHERE id = 1 LIMIT 0",
    ),
    (
        "SELECT t.id FROM t JOIN u ON u.x = t.a ORDER BY t.id LIMIT (SELECT 2)",
        "SELECT t.id FROM t JOIN u ON u.x = t.a ORDER BY t.id LIMIT 2",
    ),
    (
        "SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a LIMIT (SELECT 1)",
        "SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a LIMIT 1",
    ),
    (
        "SELECT id FROM t UNION SELECT 9 ORDER BY id LIMIT (SELECT 2)",
        "SELECT id FROM t UNION SELECT 9 ORDER BY id LIMIT 2",
    ),
    (
        "WITH c AS (SELECT id FROM t) SELECT id FROM c ORDER BY id LIMIT (SELECT 2)",
        "WITH c AS (SELECT id FROM t) SELECT id FROM c ORDER BY id LIMIT 2",
    ),
    (
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT (SELECT 2)) d ORDER BY id",
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 2) d ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE id IN (SELECT x FROM u ORDER BY x LIMIT (SELECT 1)) ORDER BY id",
        "SELECT id FROM t WHERE id IN (SELECT x FROM u ORDER BY x LIMIT 1) ORDER BY id",
    ),
    ("SELECT * FROM tv ORDER BY id", "SELECT id FROM t ORDER BY id LIMIT 2"),
    (
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT (SELECT 3)) d ORDER BY id LIMIT (SELECT 2)",
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 3) d ORDER BY id LIMIT 2",
    ),
    (
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT (SELECT 2)) d ORDER BY id DESC LIMIT (SELECT 1)",
        "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 2) d ORDER BY id DESC LIMIT 1",
    ),
];

#[test]
fn a_limit_subquery_answers_as_its_literal() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!("file://{}", dir.path().display())).unwrap()
        } else {
            Database::open("memory://limit_subquery_twins").unwrap()
        };
        setup(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, literal) in TWINS {
            let want = answer(db.query(literal, ()));
            assert!(want.is_ok(), "{literal}: {want:?}");
            let prepared = db.prepare(sql).map_err(|e| e.to_string());
            for context in ["autocommit", "transaction", "prepared", "prepared again"] {
                let got = match context {
                    "autocommit" => answer(db.query(sql, ())),
                    "transaction" => {
                        let mut tx = db.begin().unwrap();
                        let got = answer(tx.query(sql, ()));
                        tx.rollback().unwrap();
                        got
                    }
                    _ => prepared.clone().and_then(|p| answer(p.query(()))),
                };
                if got != want {
                    wrong.push(format!(
                        "{sql} (sealed: {sealed}, {context}): {got:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_limit_subquery_takes_the_rule_of_a_count() {
    let db = Database::open("memory://limit_subquery_errors").unwrap();
    setup(&db);
    for (sql, error) in [
        (
            "SELECT id FROM t LIMIT (SELECT x FROM u)",
            "more than one row",
        ),
        (
            "SELECT id FROM t LIMIT (SELECT x FROM u WHERE id = 9)",
            "LIMIT must be an integer",
        ),
        ("SELECT id FROM t LIMIT (SELECT 'a')", "LIMIT must be an integer"),
        ("SELECT id FROM t LIMIT (SELECT 2.5)", "LIMIT must be an integer"),
        (
            "SELECT id FROM t OFFSET (SELECT -1)",
            "OFFSET must be non-negative",
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT x FROM u ORDER BY x LIMIT (SELECT t.a))",
            "unknown table or alias 't'",
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT x FROM u WHERE u.x = t.a ORDER BY x LIMIT (SELECT 1))",
            "a row of an enclosing statement",
        ),
        ("SELECT id FROM t LIMIT (SELECT a)", "column 'a' not found"),
        (
            "WITH c AS (SELECT id FROM t) SELECT id FROM c LIMIT (SELECT COUNT(*) FROM c)",
            "not found",
        ),
    ] {
        let got = answer(db.query(sql, ()));
        assert!(
            got.as_ref().is_err_and(|e| e.contains(error)),
            "{sql}: {got:?}, want an error with {error:?}"
        );
    }
}

#[test]
fn a_prepared_limit_subquery_reads_each_execution_s_tables_and_parameters() {
    let db = Database::open("memory://limit_subquery_executions").unwrap();
    setup(&db);
    let by_key = db
        .prepare("SELECT id FROM t ORDER BY id LIMIT (SELECT x FROM u WHERE id = $1)")
        .unwrap();
    let count = |rows: Result<Vec<String>, String>| rows.map(|r| r.len());
    assert_eq!(count(answer(by_key.query((1,)))), Ok(1));
    assert_eq!(count(answer(by_key.query((2,)))), Ok(2));
    let by_count = db
        .prepare("SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u)")
        .unwrap();
    assert_eq!(count(answer(by_count.query(()))), Ok(2));
    db.execute("INSERT INTO u VALUES (3, 3)", ()).unwrap();
    assert_eq!(count(answer(by_count.query(()))), Ok(3));
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (4, 4, 0)", ()).unwrap();
    assert_eq!(count(answer(by_count.query(()))), Ok(4));
}

#[test]
fn a_limit_subquery_sees_what_its_transaction_wrote() {
    let db = Database::open("memory://limit_subquery_own_write").unwrap();
    setup(&db);
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (5, 5)", ()).unwrap();
    let got = answer(tx.query(
        "SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM t) - 2",
        (),
    ));
    tx.rollback().unwrap();
    assert_eq!(got.map(|r| r.len()), Ok(3));
}
