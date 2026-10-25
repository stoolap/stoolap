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

use stoolap::test_failpoints::{prepared_strategy_runs, random_draws, PreparedStrategy};
use stoolap::Database;

/// The statement-per-call counter is process-wide: every test here runs
/// statement blocks or reads the counter, so they take turns
static COUNTED: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn turn() -> std::sync::MutexGuard<'static, ()> {
    COUNTED.lock().unwrap_or_else(|e| e.into_inner())
}

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
    let _turn = turn();
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
    let _turn = turn();
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
    let _turn = turn();
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
    let _turn = turn();
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
    let _turn = turn();
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

#[test]
fn a_derived_table_s_names_bind_as_its_rows_and_select_list_name_them() {
    let _turn = turn();
    let db = Database::open("memory://derived_table_blocks_row_names").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        "INSERT INTO t VALUES (1, 999, 10)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (sql, expected) in [
        // The middle `x` is the column CAST(x AS INTEGER) gives c, not t's
        (
            "SELECT (SELECT x FROM (SELECT x FROM (SELECT CAST(x AS INTEGER) FROM u) AS c) AS d) FROM t",
            "2",
        ),
        // A SELECT item reads no alias of its own list: its `a` is t's
        (
            "SELECT (SELECT q.a FROM (SELECT a AS a WHERE t.id = 1) AS q) FROM t",
            "10",
        ),
    ] {
        assert_eq!(answer(&db, sql), Ok(vec![vec![expected.to_string()]]), "{sql}");
    }
}

#[test]
fn a_view_in_a_derived_table_is_read_once_in_the_run_that_prepares_it() {
    let _turn = turn();
    let db = hot("view_once");
    db.execute("CREATE VIEW rv AS SELECT RANDOM() AS n", ())
        .unwrap();
    let sql = "SELECT id FROM t WHERE a > (SELECT MAX(q.n) FROM (SELECT n FROM rv) AS q) - 100 \
               ORDER BY id";
    for run in ["prepares", "keeps its plan"] {
        let before = random_draws();
        assert_eq!(answer(&db, sql), Ok(ids(&[1, 2, 3])), "{run}");
        assert_eq!(
            random_draws() - before,
            1,
            "the run that {run} reads the view once"
        );
    }
}

#[test]
fn an_exists_over_a_correlated_derived_table_reads_no_select_item() {
    let _turn = turn();
    let db = hot("exists_items");
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER, g TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO m VALUES (1, -9223372036854775808, 'p')", ())
        .unwrap();
    const Q: &str = "(SELECT x, g FROM m WHERE m.id = t.id) AS q";
    for (sql, expected) in [
        (format!("SELECT id FROM t WHERE EXISTS (SELECT ABS(q.x) FROM {Q}) ORDER BY id"), ids(&[1])),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT q.g, ABS(MIN(q.x)) FROM {Q} GROUP BY 1 HAVING COUNT(*) > 0) ORDER BY id"),
            ids(&[1]),
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT q.x AS k, ABS(q.x) FROM {Q} GROUP BY k) ORDER BY id"),
            ids(&[1]),
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT ABS(q.x) AS v, q.x AS k FROM {Q} WHERE k < 0) ORDER BY id"),
            ids(&[1]),
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT COUNT(*) AS c, ABS(MIN(q.x)) FROM {Q} HAVING c = 1) ORDER BY id"),
            ids(&[1]),
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT ABS(q.x) FROM {Q} ORDER BY ABS(q.x) LIMIT 1 OFFSET 1) ORDER BY id"),
            ids(&[]),
        ),
    ] {
        assert_eq!(answer(&db, &sql), Ok(expected), "{sql}");
    }
}

#[test]
fn a_block_source_no_call_needs_raises_nothing() {
    let _turn = turn();
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("lazy_sources")), ("sealed", sealed(&dir))] {
        db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO m VALUES (1, -9223372036854775808)", ())
            .unwrap();
        db.execute("CREATE VIEW mv AS SELECT id, ABS(x) AS x FROM m", ())
            .unwrap();
        for source in [
            "mv AS s",
            "(VALUES (1, ABS(-9223372036854775808))) AS s(id, x)",
            "(SELECT id, ABS(x) AS x FROM m) AS s",
            "(SELECT value AS id, value AS x FROM generate_series(1, 2, 0)) AS s",
        ] {
            let block = format!("(SELECT MAX(s.x) FROM {source} WHERE s.id = t.id)");
            for sql in [
                format!("SELECT id FROM t WHERE id > 100 AND a > {block} ORDER BY id"),
                format!("SELECT id FROM t WHERE id > 0 OR a > {block} ORDER BY id"),
                format!("SELECT id FROM t WHERE CASE WHEN id > 0 THEN TRUE ELSE a > {block} END ORDER BY id"),
                format!("SELECT id, CASE WHEN id > 0 THEN 0 ELSE {block} END FROM t ORDER BY id"),
                format!(
                    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 0 OR MAX(id) > {} ORDER BY a",
                    block.replace("t.id", "t.a")
                ),
            ] {
                let got = answer(&db, &sql);
                assert!(got.is_ok(), "{context}: {sql}: {got:?}");
            }
            let with_param = format!("SELECT id FROM t WHERE $1 = 0 OR a > {block} ORDER BY id");
            let statement = db.prepare(&with_param).unwrap();
            for _ in 0..2 {
                let got: Result<Vec<_>, _> = statement.query((0i64,)).unwrap().collect();
                assert!(got.is_ok(), "{context}: {with_param}: {:?}", got.err());
            }
            let needed = statement
                .query((1i64,))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                .map_err(|e| e.to_string());
            assert!(
                needed
                    .as_ref()
                    .is_err_and(|e| e.contains("overflow") || e.contains("step")),
                "{context}: {with_param} with 1 reads the source: {needed:?}"
            );
            let returning = format!(
                "UPDATE t SET b = b WHERE id = 1 RETURNING id, CASE WHEN id > 0 THEN 0 ELSE {block} END"
            );
            let got = answer(&db, &returning);
            assert!(got.is_ok(), "{context}: {returning}: {got:?}");
        }
    }
}

#[test]
fn a_view_that_reads_itself_raises_the_nesting_error() {
    let _turn = turn();
    let db = hot("view_cycle");
    for sql in [
        "CREATE VIEW v AS SELECT x FROM v",
        "CREATE VIEW v1 AS SELECT x FROM u",
        "CREATE VIEW v2 AS SELECT x FROM v1",
        "DROP VIEW v1",
        "CREATE VIEW v1 AS SELECT x FROM v2",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in [
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM v WHERE v.x = t.id)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM v) AS q WHERE q.x = t.id)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM v WHERE x = t.id) AS q)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM v1 WHERE v1.x = t.id)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM v2 WHERE x = t.id) AS q)",
    ] {
        let got = answer(&db, sql);
        assert!(
            got.as_ref()
                .is_err_and(|e| e.contains("Maximum view nesting depth")),
            "{sql}: {got:?}"
        );
    }
}

#[test]
fn an_exists_over_a_correlated_derived_table_binds_the_order_by_it_drops() {
    let _turn = turn();
    let db = hot("exists_order");
    const Q: &str = "(SELECT x FROM u WHERE u.id = t.id) AS q";
    for (sql, missing) in [
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {Q} ORDER BY q.missing LIMIT 1)"),
            "q.missing",
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {Q} ORDER BY missing LIMIT 1)"),
            "missing",
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {Q} ORDER BY q.x, q.missing)"),
            "q.missing",
        ),
        (
            format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {Q} ORDER BY ABS(missing))"),
            "missing",
        ),
    ] {
        let got = answer(&db, &sql);
        assert!(
            got.as_ref()
                .is_err_and(|e| e.contains(&format!("'{missing}' not found"))),
            "{sql}: {got:?}"
        );
    }
    let bound = format!(
        "SELECT id FROM t WHERE EXISTS (SELECT q.x AS k FROM {Q} ORDER BY k, q.x, x, t.a, ABS(q.x - 9223372036854775807 - 10)) ORDER BY id"
    );
    assert_eq!(answer(&db, &bound), Ok(ids(&[1, 2, 3, 4])), "{bound}");
}
