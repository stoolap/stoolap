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
    (
        "WITH c(id) AS (SELECT id FROM t ORDER BY id LIMIT (SELECT 1) OFFSET (SELECT 2)) SELECT id FROM c LIMIT (SELECT 1)",
        "WITH c(id) AS (SELECT id FROM t ORDER BY id LIMIT 1 OFFSET 2) SELECT id FROM c LIMIT 1",
    ),
    (
        "WITH c(id) AS (SELECT id FROM t), d(id) AS (SELECT id FROM c LIMIT (SELECT 2)) SELECT id FROM d ORDER BY id",
        "WITH c(id) AS (SELECT id FROM t), d(id) AS (SELECT id FROM c LIMIT 2) SELECT id FROM d ORDER BY id",
    ),
    (
        "WITH c AS (SELECT id FROM t) SELECT id FROM c LIMIT (SELECT 3) OFFSET (SELECT 1)",
        "WITH c AS (SELECT id FROM t) SELECT id FROM c LIMIT 3 OFFSET 1",
    ),
    (
        "WITH c AS (SELECT id FROM t WHERE id < 3) SELECT id FROM c ORDER BY id LIMIT (SELECT COUNT(*) FROM c) - 1",
        "WITH c AS (SELECT id FROM t WHERE id < 3) SELECT id FROM c ORDER BY id LIMIT 1",
    ),
    (
        "WITH u AS (SELECT 1 AS x) SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u)",
        "WITH u AS (SELECT 1 AS x) SELECT id FROM t ORDER BY id LIMIT 1",
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
fn a_block_run_for_each_row_reads_its_limit_subquery_as_a_count() {
    let n = "Null(Null)";
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT id FROM t WHERE id IN (SELECT x FROM u WHERE u.x = t.a ORDER BY x \
             LIMIT (SELECT 1)) ORDER BY id",
            &["Integer(1)", "Integer(2)"],
        ),
        (
            "SELECT (SELECT u.id FROM u WHERE u.x = t.a ORDER BY u.id LIMIT 1 OFFSET (SELECT 1)) \
             FROM t ORDER BY id",
            &["Integer(3)", n, n, "Integer(3)"],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a LIMIT (SELECT 0))",
            &[],
        ),
        (
            "SELECT (SELECT COUNT(*) FROM (SELECT u.id FROM u WHERE u.x = t.a \
             LIMIT (SELECT COUNT(*) FROM r) - 2) d) FROM t ORDER BY id",
            &["Integer(1)", "Integer(1)", "Integer(0)", "Integer(1)"],
        ),
        (
            "SELECT (SELECT u.id FROM u JOIN r ON r.rid = u.id WHERE u.x = t.a ORDER BY u.id \
             LIMIT (SELECT 1)) FROM t ORDER BY id",
            &["Integer(1)", "Integer(2)", n, "Integer(1)"],
        ),
    ];
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}/db", dir.path().display())).unwrap(),
            false => Database::open("memory://limit_subquery_per_row_block").unwrap(),
        };
        setup(&db);
        for sql in [
            "INSERT INTO u VALUES (3, 1)",
            "CREATE TABLE r (rid INTEGER PRIMARY KEY)",
            "INSERT INTO r VALUES (1), (2), (3)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, want) in cases {
            let want: Vec<String> = want.iter().map(|v| v.to_string()).collect();
            assert_eq!(
                answer(db.query(sql, ())),
                Ok(want),
                "sealed {sealed}: {sql}"
            );
        }
    }
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
        (
            "SELECT id FROM t LIMIT (SELECT 'a')",
            "LIMIT must be an integer",
        ),
        (
            "SELECT id FROM t LIMIT (SELECT 2.5)",
            "LIMIT must be an integer",
        ),
        (
            "SELECT id FROM t OFFSET (SELECT -1)",
            "OFFSET must be non-negative",
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT x FROM u ORDER BY x LIMIT (SELECT t.a))",
            "unknown table or alias 't'",
        ),
        ("SELECT id FROM t LIMIT (SELECT a)", "column 'a' not found"),
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

#[test]
fn a_set_operation_block_keeps_its_order_by_s_null_placement() {
    let db = Database::open("memory://limit_subquery_set_operation_nulls").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 0), (5, 1), (6, 2), (7, 3), (8, 0)",
        "INSERT INTO u VALUES (1, 1, 1), (2, 2, 2), (3, 0, 3), (4, 1, 4), (5, 2, 5), (6, 0, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let block = "SELECT x FROM u WHERE id = t.id UNION SELECT y FROM u WHERE id = t.id \
                 ORDER BY 1 DESC LIMIT 1";
    // DESC sorts NULL first, as the block run alone does: row 6 keeps NULL
    let alone: Vec<Option<i64>> = db
        .query(&block.replace("t.id", "6"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<Option<i64>>(0).unwrap())
        .collect();
    assert_eq!(alone, [None]);
    let sql = format!("SELECT id FROM t WHERE a NOT IN ({block}) ORDER BY id");
    let ids: Vec<i64> = db
        .query(&sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(ids, [4, 5, 7, 8], "{sql}");
}
