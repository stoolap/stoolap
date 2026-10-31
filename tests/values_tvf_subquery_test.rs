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

use std::sync::{Mutex, MutexGuard};

use stoolap::{Database, Value};

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER)",
        "INSERT INTO t VALUES (1, 30, 1), (2, 10, 2), (3, NULL, 1), (4, 20, 3), (5, 10, 2)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<Value>>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| row.map(|row| (0..row.len()).map(|i| row.get(i).unwrap()).collect()))
                .collect()
        })
        .map_err(|e| e.to_string())
}

/// Each statement with its twin: the same statement with every subquery in
/// a VALUES row or table function argument replaced by the value it reads
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT * FROM (VALUES (1, (SELECT 2)), (3, 4)) x(a, b) ORDER BY a",
        "SELECT * FROM (VALUES (1, 2), (3, 4)) x(a, b) ORDER BY a",
    ),
    (
        "SELECT column1 FROM (VALUES ((SELECT MAX(v) FROM u)), (2)) x ORDER BY column1",
        "SELECT column1 FROM (VALUES (7), (2)) x ORDER BY column1",
    ),
    (
        "SELECT column1 FROM (VALUES ((SELECT MAX(v) FROM u)), (2)) x \
         WHERE column1 > (SELECT 1) ORDER BY column1",
        "SELECT column1 FROM (VALUES (7), (2)) x WHERE column1 > 1 ORDER BY column1",
    ),
    (
        "SELECT c FROM (VALUES (EXISTS (SELECT 1 FROM u WHERE v > 6)), \
         (NOT EXISTS (SELECT 1 FROM u WHERE v > 6))) x(c) ORDER BY c",
        "SELECT c FROM (VALUES (TRUE), (FALSE)) x(c) ORDER BY c",
    ),
    (
        "SELECT c FROM (VALUES (5 IN (SELECT v FROM u)), (6 IN (SELECT v FROM u))) x(c) ORDER BY c",
        "SELECT c FROM (VALUES (TRUE), (NULL)) x(c) ORDER BY c",
    ),
    (
        "SELECT * FROM (VALUES ((SELECT COUNT(*) FROM t), (SELECT MIN(k) FROM u))) x(a, b)",
        "SELECT * FROM (VALUES (5, 1)) x(a, b)",
    ),
    (
        "SELECT c + (SELECT MIN(v) FROM u) FROM (VALUES ((SELECT MAX(v) FROM u))) x(c)",
        "SELECT c + 1 FROM (VALUES (7)) x(c)",
    ),
    (
        "SELECT c, COUNT(*) FROM (VALUES ((SELECT MIN(k) FROM u)), (2), (1)) x(c) \
         GROUP BY c ORDER BY c",
        "SELECT c, COUNT(*) FROM (VALUES (1), (2), (1)) x(c) GROUP BY c ORDER BY c",
    ),
    (
        "WITH c AS (SELECT * FROM (VALUES ((SELECT MAX(v) FROM u))) x(m)) SELECT m FROM c",
        "WITH c AS (SELECT * FROM (VALUES (7)) x(m)) SELECT m FROM c",
    ),
    (
        "SELECT m FROM (SELECT * FROM (VALUES ((SELECT MAX(v) FROM u)), (1)) x(m)) d ORDER BY m",
        "SELECT m FROM (SELECT * FROM (VALUES (7), (1)) x(m)) d ORDER BY m",
    ),
    (
        "SELECT id, (SELECT MAX(c) FROM (VALUES ((SELECT MAX(v) FROM u)), (t.k)) x(c)) \
         FROM t ORDER BY id",
        "SELECT id, (SELECT MAX(c) FROM (VALUES (7), (t.k)) x(c)) FROM t ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE a < (SELECT MAX(c) FROM \
         (VALUES ((SELECT MIN(v) FROM u WHERE u.k = t.k))) x(c)) ORDER BY id",
        "SELECT id FROM t WHERE a < (SELECT MIN(v) FROM u WHERE u.k = t.k) ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE EXISTS \
         (SELECT 1 FROM (VALUES ((SELECT MAX(v) FROM u))) x(c) WHERE c > t.a) ORDER BY id",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (VALUES (7)) x(c) WHERE c > t.a) ORDER BY id",
    ),
    (
        "SELECT value FROM generate_series(1, (SELECT MAX(v) FROM u)) ORDER BY value",
        "SELECT value FROM generate_series(1, 7) ORDER BY value",
    ),
    (
        "SELECT value FROM generate_series(1, CASE WHEN EXISTS (SELECT 1 FROM u) \
         THEN 3 ELSE 1 END) WHERE value > (SELECT 1)",
        "SELECT value FROM generate_series(1, 3) WHERE value > 1",
    ),
    (
        "SELECT value, COUNT(*) FROM generate_series(1, (SELECT COUNT(*) FROM u)) \
         GROUP BY value ORDER BY value",
        "SELECT value, COUNT(*) FROM generate_series(1, 4) GROUP BY value ORDER BY value",
    ),
    (
        "SELECT id, (SELECT SUM(value) FROM generate_series(1, (SELECT MAX(v) FROM u))) \
         FROM t ORDER BY id",
        "SELECT id, (SELECT SUM(value) FROM generate_series(1, 7)) FROM t ORDER BY id",
    ),
    (
        "SELECT id, (SELECT SUM(value) FROM generate_series(1, CASE WHEN EXISTS \
         (SELECT 1 FROM u WHERE u.k = t.k) THEN 2 ELSE 1 END)) FROM t ORDER BY id",
        "SELECT id, CASE WHEN k IN (1, 2) THEN 3 ELSE 1 END FROM t ORDER BY id",
    ),
];

#[test]
fn a_subquery_in_a_values_row_or_a_table_function_argument_reads_its_value() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("values_tvf_subquery_answers", &dir, sealed);
        let mut wrong = Vec::new();
        for (sql, twin) in SHAPES {
            let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            match rows(&db, sql) {
                Ok(got) if got == want => {}
                got => wrong.push(format!(
                    "sealed {sealed}: {sql}\n  got {got:?}\n  want {want:?}"
                )),
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    }
}

#[test]
fn a_parameter_and_a_prepared_statement_read_it_too() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_parameter", &dir, false);
    let read = |sql: &str, param: i64| -> Vec<i64> {
        db.query(sql, (param,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    let values =
        "SELECT c FROM (VALUES ((SELECT MAX(v) FROM u WHERE v < $1)), ($1)) x(c) ORDER BY c";
    assert_eq!(read(values, 6), vec![5, 6]);
    assert_eq!(read(values, 2), vec![1, 2]);
    let series = "SELECT value FROM generate_series(1, (SELECT COUNT(*) FROM u WHERE v > $1))";
    assert_eq!(read(series, 4), vec![1, 2]);
    assert_eq!(read(series, 0), vec![1, 2, 3]);
    let stmt = db.prepare(values).unwrap();
    for (param, want) in [(6, vec![5, 6]), (8, vec![7, 8])] {
        let got: Vec<i64> = stmt
            .query((param,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(got, want, "prepared statement, ${param}");
    }
}

#[test]
fn a_changed_table_changes_the_value_a_cached_statement_reads() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_changes", &dir, false);
    let sql = "SELECT c FROM (VALUES ((SELECT MAX(v) FROM u))) x(c)";
    assert_eq!(rows(&db, sql).unwrap(), vec![vec![Value::Integer(7)]]);
    db.execute("INSERT INTO u VALUES (9, 9, 90)", ()).unwrap();
    assert_eq!(rows(&db, sql).unwrap(), vec![vec![Value::Integer(90)]]);
}

#[test]
fn an_error_in_the_subquery_is_the_statements_error() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_errors", &dir, false);
    for (sql, want) in [
        (
            "SELECT * FROM (VALUES ((SELECT v FROM u))) x(c)",
            "more than one row",
        ),
        (
            "SELECT * FROM (VALUES ((SELECT nocol FROM u))) x(c)",
            "nocol",
        ),
        (
            "SELECT value FROM generate_series(1, (SELECT v FROM u))",
            "more than one row",
        ),
    ] {
        let got = rows(&db, sql).expect_err(sql);
        assert!(got.contains(want), "{sql}: {got}");
    }
}

#[test]
fn a_table_function_a_false_where_keeps_no_row_of_computes_no_argument() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_false_where", &dir, false);
    for (sql, want) in [
        (
            "SELECT value FROM generate_series(ABS(-9223372036854775808), (SELECT 3)) WHERE 1 = 0",
            vec![],
        ),
        (
            "SELECT value FROM generate_series(1, (SELECT v FROM u)) WHERE 1 = 0",
            vec![],
        ),
        (
            "SELECT COUNT(*) FROM generate_series(1, (SELECT v FROM u)) WHERE 1 = 0",
            vec![vec![Value::Integer(0)]],
        ),
    ] {
        assert_eq!(rows(&db, sql), Ok(want), "{sql}");
    }
}

#[test]
fn a_statement_run_without_a_plan_of_its_own_reads_it_too() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_no_plan", &dir, false);
    let values = "SELECT * FROM (VALUES ((SELECT 2))) x(a)";
    let series = "SELECT value FROM generate_series(1, (SELECT 2))";
    for (sql, want) in [(values, vec![2]), (series, vec![1, 2])] {
        let want: Vec<Vec<Value>> = want.into_iter().map(|v| vec![Value::Integer(v)]).collect();
        assert_eq!(
            rows(&db, &format!("SELECT 0; {sql}")),
            Ok(want.clone()),
            "the second statement of a program: {sql}"
        );
        let stmt = stoolap::parser::parse_sql(sql).unwrap().remove(0);
        let mut tx = db.begin().unwrap();
        let got: Vec<Vec<Value>> = tx
            .query_prepared(&stmt, ())
            .unwrap()
            .map(|row| row.map(|row| vec![row.get(0).unwrap()]))
            .collect::<Result<_, _>>()
            .unwrap();
        tx.rollback().unwrap();
        assert_eq!(got, want, "a parsed statement in a transaction: {sql}");
    }
    db.execute(&format!("CREATE VIEW vv AS {values}"), ())
        .unwrap();
    db.execute(&format!("CREATE VIEW vs AS {series}"), ())
        .unwrap();
    for (sql, want) in [
        ("SELECT * FROM vv", vec![2]),
        ("SELECT * FROM vs", vec![1, 2]),
    ] {
        let want: Vec<Vec<Value>> = want.into_iter().map(|v| vec![Value::Integer(v)]).collect();
        assert_eq!(rows(&db, sql), Ok(want), "a view's query: {sql}");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_subquery_in_a_values_row_or_a_table_function_argument_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("values_tvf_subquery_general_path", &dir, false);
    for (sql, _) in SHAPES {
        let before = general_path_total();
        rows(&db, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
