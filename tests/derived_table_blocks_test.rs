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

//! A subquery block over a derived table runs prepared: one that reads no
//! enclosing scope over its rows, read once per run, and one that does as a
//! statement of its own for each call, its names bound as any block's are

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{prepared_strategy_runs, PreparedStrategy};
use stoolap::Database;

/// The statement-per-call counter is process-wide: tests that read it take turns
static COUNTED: std::sync::Mutex<()> = std::sync::Mutex::new(());

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 5), (2, 2, 6), (3, 9, 7), (4, NULL, 8)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, g TEXT)",
    "INSERT INTO u VALUES (1, 1, 'p'), (2, 3, 'p'), (3, 2, 'q'), (4, NULL, 'q'), (5, 2, 'r')",
    "CREATE VIEW uv AS SELECT id, x FROM u WHERE x IS NOT NULL",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://derived_table_blocks_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn answer(db: &Database, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .map(|v| v.unwrap_or_else(|| "NULL".into()))
                        .unwrap_or_else(|e| format!("<{e}>"))
                })
                .collect(),
        );
    }
    Ok(out)
}

fn ids(values: &[i64]) -> Vec<Vec<String>> {
    values.iter().map(|v| vec![v.to_string()]).collect()
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

/// Each statement, its ids, and whether its derived table reads an enclosing scope
const CASES: &[(&str, &[i64], bool)] = &[
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a) ORDER BY id", &[3, 4], false),
    ("SELECT id FROM t WHERE a IN (SELECT q.x FROM (SELECT x FROM u WHERE x > 1) AS q) ORDER BY id", &[2], false),
    ("SELECT id FROM t WHERE a NOT IN (SELECT q.x FROM (SELECT x FROM u WHERE x IS NOT NULL) AS q) ORDER BY id", &[3], false),
    ("SELECT id FROM t WHERE a NOT IN (SELECT q.x FROM (SELECT x FROM u) AS q) ORDER BY id", &[], false),
    ("SELECT id FROM t WHERE b > (SELECT SUM(q.x) FROM (SELECT x FROM u) AS q) ORDER BY id", &[], false),
    ("SELECT id FROM t WHERE a = ANY (SELECT q.x FROM (SELECT x FROM u) AS q) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE a > ALL (SELECT q.x FROM (SELECT x FROM u WHERE x < 3) AS q) ORDER BY id", &[3], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x, COUNT(*) AS n FROM u GROUP BY x) AS q WHERE q.x = t.a AND q.n = 2) ORDER BY id", &[2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u ORDER BY x DESC LIMIT 2) AS q WHERE q.x = t.a) ORDER BY id", &[], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u UNION SELECT 9) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2, 3], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT DISTINCT x FROM u) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE x > (SELECT MIN(x) FROM u)) AS q WHERE q.x = t.a) ORDER BY id", &[2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM uv) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT column1 AS x FROM (VALUES (1), (9)) AS v) AS q WHERE q.x = t.a) ORDER BY id", &[1, 3], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT value AS x FROM generate_series(1, 2)) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM (SELECT x FROM u) AS r) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x AS y FROM u) AS q WHERE q.y = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x, x + 1 AS y FROM u) AS q WHERE q.y = t.b - 3) ORDER BY id", &[1, 2, 3], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT * FROM (SELECT x FROM u) AS q WHERE q.x = t.a) ORDER BY id", &[1, 2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a AND q.x / (q.x - 1) > 0) ORDER BY id", &[2], false),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE u.x = t.a) AS q) ORDER BY id", &[1, 2], true),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x, (SELECT t.b) AS y FROM u) AS q WHERE q.x = t.a AND q.y > 5) ORDER BY id", &[2], true),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE EXISTS (SELECT 1 FROM t AS t2 WHERE t2.a = u.x AND t2.b > t.b)) AS q) ORDER BY id", &[1], true),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT g, COUNT(*) AS n FROM u WHERE u.x <= t.a GROUP BY g) AS q WHERE q.n > 1) ORDER BY id", &[3], true),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE u.x = t.a UNION SELECT 9 WHERE t.a = 9) AS q) ORDER BY id", &[1, 2, 3], true),
    ("SELECT id FROM t WHERE a IN (SELECT q.x FROM (SELECT x FROM u WHERE u.id > t.id) AS q) ORDER BY id", &[2], true),
    ("SELECT id FROM t WHERE a NOT IN (SELECT q.x FROM (SELECT x FROM u WHERE u.id > t.id) AS q) ORDER BY id", &[], true),
    ("SELECT id FROM t WHERE a < ALL (SELECT q.x FROM (SELECT x FROM u WHERE u.x IS NOT NULL AND u.id > t.id) AS q) ORDER BY id", &[1], true),
    ("SELECT id FROM t WHERE b > (SELECT COUNT(*) FROM (SELECT x FROM u WHERE u.x < t.a) AS q) + 4 ORDER BY id", &[1, 2, 4], true),
    ("SELECT t.id FROM t JOIN u ON u.id = t.id WHERE EXISTS (SELECT 1 FROM (SELECT x FROM uv) AS q WHERE q.x = t.a) ORDER BY t.id", &[1, 2], false),
    ("SELECT t.id FROM t JOIN u ON u.id = t.id WHERE EXISTS (SELECT 1 FROM (SELECT x FROM uv WHERE uv.x = t.a) AS q) ORDER BY t.id", &[1, 2], true),
];

#[test]
fn a_block_over_a_derived_table_runs_prepared_hot_and_sealed() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in [("hot", hot("cases")), ("sealed", sealed(&dir))] {
        for (sql, expected, correlated) in CASES {
            let line = prepared_line(&db, sql);
            if line != "Prepared: yes" {
                wrong.push(format!("{context}: {sql}: {line}"));
            }
            let before = prepared_strategy_runs(PreparedStrategy::BlockStatement);
            let got = answer(&db, sql);
            if got != Ok(ids(expected)) {
                wrong.push(format!("{context}: {sql}: {got:?}"));
            }
            let per_call = prepared_strategy_runs(PreparedStrategy::BlockStatement) > before;
            if per_call != *correlated {
                wrong.push(format!("{context}: {sql}: ran per call: {per_call}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_select_list_block_over_a_derived_table_answers_each_row() {
    let db = hot("select_list");
    for (sql, expected) in [
        (
            "SELECT id, (SELECT COUNT(*) FROM (SELECT x FROM u) AS q WHERE q.x = t.a) FROM t ORDER BY id",
            [["1", "1"], ["2", "2"], ["3", "0"], ["4", "0"]],
        ),
        (
            "SELECT id, (SELECT MAX(q.x) FROM (SELECT x FROM u) AS q) FROM t ORDER BY id",
            [["1", "3"], ["2", "3"], ["3", "3"], ["4", "3"]],
        ),
        (
            "SELECT id, (SELECT MAX(q.x) FROM (SELECT x FROM u WHERE u.x <= t.a) AS q) FROM t ORDER BY id",
            [["1", "1"], ["2", "2"], ["3", "3"], ["4", "NULL"]],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM (SELECT x FROM u WHERE u.x < t.b - 3) AS q WHERE q.x <> t.a) \
             FROM t ORDER BY id",
            [["1", "0"], ["2", "1"], ["3", "4"], ["4", "0"]],
        ),
        (
            "SELECT t.id, (SELECT COUNT(*) FROM (SELECT x FROM uv WHERE uv.x <= u.x) AS q) \
             FROM t JOIN u ON u.id = t.id ORDER BY t.id",
            [["1", "1"], ["2", "4"], ["3", "3"], ["4", "0"]],
        ),
    ] {
        let expected: Vec<Vec<String>> = expected
            .iter()
            .map(|row| row.iter().map(|v| v.to_string()).collect())
            .collect();
        assert_eq!(answer(&db, sql), Ok(expected), "{sql}");
    }
}

#[test]
fn a_name_in_a_block_s_derived_table_binds_the_nearest_scope_with_it() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("names")), ("sealed", sealed(&dir))] {
        for (sql, expected) in [
            // `a` names no column of u: it is t's
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE x = a) AS q) ORDER BY id", ids(&[1, 2])),
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = a) ORDER BY id", ids(&[1, 2])),
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE CASE WHEN x = a THEN 1 END = 1) AS q) ORDER BY id", ids(&[1, 2])),
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x, (SELECT a) AS y FROM u) AS q WHERE q.y = q.x) ORDER BY id", ids(&[1, 2])),
            // `id` and `x` are columns of u and of the derived table: theirs first
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE id = 2) AS q WHERE q.x = t.a + 2) ORDER BY id", ids(&[1])),
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT id AS x FROM u WHERE id = 5) AS q WHERE x = t.id + 4) ORDER BY id", ids(&[1])),
            // An alias of the derived table's own SELECT list is its own
            ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x AS a FROM u ORDER BY a) AS q WHERE q.a = t.a) ORDER BY id", ids(&[1, 2])),
        ] {
            assert_eq!(answer(&db, sql), Ok(expected), "{context}: {sql}");
        }
        let grouped = "SELECT a, COUNT(*) FROM t GROUP BY a \
                       HAVING EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE x = a) AS q) ORDER BY a";
        assert_eq!(
            answer(&db, grouped),
            Ok(vec![
                vec!["1".to_string(), "1".to_string()],
                vec!["2".to_string(), "1".to_string()]
            ]),
            "{context}: {grouped}"
        );
        let with_param = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE x = a AND x > $1) AS q) \
                          ORDER BY id";
        let statement = db.prepare(with_param).unwrap();
        for (param, expected) in [(0i64, vec![1i64, 2]), (1, vec![2])] {
            let got: Vec<i64> = statement
                .query((param,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(got, expected, "{context}: {with_param} with {param}");
        }
    }
}

#[test]
fn a_write_reads_a_block_over_a_derived_table() {
    let db = hot("writes");
    for (sql, expected) in [
        (
            "UPDATE t SET b = 0 WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a) RETURNING id, b",
            vec![["1", "0"], ["2", "0"]],
        ),
        (
            "UPDATE t SET b = (SELECT COUNT(*) FROM (SELECT x FROM u WHERE u.x <= t.a) AS q) RETURNING id, b",
            vec![["1", "1"], ["2", "3"], ["3", "4"], ["4", "0"]],
        ),
        (
            "DELETE FROM t WHERE a IN (SELECT q.x FROM (SELECT x FROM u WHERE u.id > t.id) AS q) RETURNING id, a",
            vec![["2", "2"]],
        ),
    ] {
        db.execute("BEGIN", ()).unwrap();
        let mut got = answer(&db, sql).unwrap();
        got.sort();
        db.execute("ROLLBACK", ()).unwrap();
        let expected: Vec<Vec<String>> = expected
            .iter()
            .map(|row| row.iter().map(|v| v.to_string()).collect())
            .collect();
        assert_eq!(got, expected, "{sql}");
    }
}

#[test]
fn a_correlated_derived_table_raises_its_errors_and_reads_each_statement_s_rows() {
    let db = hot("errors");
    let many =
        "SELECT id FROM t WHERE a = (SELECT q.x FROM (SELECT x FROM u WHERE u.x >= t.a) AS q)";
    let err = answer(&db, many).unwrap_err();
    assert!(err.contains("more than one row"), "{many}: {err}");
    let missing =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE u.x = t.nosuch) AS q)";
    let err = answer(&db, missing).unwrap_err();
    assert!(err.contains("t.nosuch"), "{missing}: {err}");
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE u.x = t.a) AS q) ORDER BY id";
    assert_eq!(answer(&db, sql), Ok(ids(&[1, 2])));
    db.execute("UPDATE u SET x = 9 WHERE id = 1", ()).unwrap();
    assert_eq!(
        answer(&db, sql),
        Ok(ids(&[2, 3])),
        "a write between statements"
    );
    let uncorrelated =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a) ORDER BY id";
    db.execute("UPDATE u SET x = 1 WHERE id = 2", ()).unwrap();
    assert_eq!(
        answer(&db, uncorrelated),
        Ok(ids(&[1, 2, 3])),
        "{uncorrelated}"
    );
}
