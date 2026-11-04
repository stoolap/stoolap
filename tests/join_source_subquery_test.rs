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

//! A VALUES list or a table function holding a subquery answers as one
//! side of a join, and a VALUES row is evaluated, without the evaluator
//! bridge

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::test_failpoints::{bridge_compiles, tvf_rows};
use stoolap::Database;

// The bridge counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Each statement, its parameters and the rows it answers, every value
/// written out
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT t.id FROM t INNER JOIN (VALUES ((SELECT 3))) AS v(n) ON v.n = t.id",
        &[],
        &["3"],
    ),
    (
        "SELECT a.id, v.n FROM t AS a LEFT JOIN u AS b ON a.id = b.id \
         LEFT JOIN (VALUES ((SELECT COUNT(*) FROM t))) AS v(n) ON v.n = 3 ORDER BY a.id",
        &[],
        &["1|3", "2|3", "3|3"],
    ),
    (
        "SELECT t.id, v.n FROM t INNER JOIN (VALUES (1 + 1, 'b'), ($1 * 3, 'c')) \
         AS v(n, s) ON v.n = t.id ORDER BY t.id",
        &[1],
        &["2|2", "3|3"],
    ),
    (
        "SELECT n, s FROM (VALUES (1 + 1, UPPER('b')), ($1 * 3, 'c')) AS v(n, s) ORDER BY n",
        &[2],
        &["2|B", "6|c"],
    ),
    (
        "SELECT t.id FROM t JOIN generate_series(1, (SELECT 2)) AS g ON g.value = t.id \
         ORDER BY t.id",
        &[],
        &["1", "2"],
    ),
    (
        "SELECT t.id, g.value FROM t LEFT JOIN \
         generate_series(2, (SELECT MAX(id) FROM t WHERE id < $1)) AS g ON g.value = t.id \
         ORDER BY t.id",
        &[3],
        &["1|NULL", "2|2", "3|NULL"],
    ),
    // A GROUP BY join under a LIMIT reads its sides in chunks of the left
    // keys; the first chunk matches nothing
    (
        "SELECT t.id, COUNT(*) FROM t JOIN (VALUES ((SELECT 3))) AS v(n) ON t.id = v.n \
         GROUP BY t.id LIMIT 1",
        &[],
        &["3|1"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN generate_series(3, (SELECT 3)) AS g \
         ON t.id = g.value GROUP BY t.id LIMIT 1",
        &[],
        &["3|1"],
    ),
];

fn rows(db: &Database, sql: &str, params: &[i64]) -> Vec<String> {
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    db.query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|at| row.get::<Value>(at).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

#[test]
fn a_source_holding_a_subquery_answers_as_written_without_the_bridge() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_source_subquery").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1), (2), (3)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO u VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, params, expected) in CASES {
        for run in 0..2 {
            let before = bridge_compiles();
            let answer = rows(&db, sql, params);
            let compiled = bridge_compiles() - before;
            if answer != *expected {
                wrong.push(format!(
                    "{sql}, run {run}: {answer:?}, written {expected:?}"
                ));
            }
            if compiled != 0 {
                wrong.push(format!("{sql}, run {run}: {compiled} bridge compiles"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_table_function_side_is_narrowed_by_its_own_condition_as_it_generates() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_source_subquery_narrowed").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    // Unnarrowed, the series would pass the rows a series may generate
    let sql = "SELECT t.id FROM t JOIN generate_series(1, (SELECT 10000001)) AS g \
               ON g.value = t.id WHERE g.value = 1";
    for run in 0..2 {
        let before = tvf_rows();
        assert_eq!(rows(&db, sql, &[]), ["1"], "{sql}, run {run}");
        assert_eq!(tvf_rows() - before, 1, "{sql}, run {run}: rows generated");
    }
}

#[test]
fn a_side_read_without_its_condition_keeps_its_plan_apart() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_source_subquery_apart").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2)", ()).unwrap();
    // A WHERE false reads each side without its condition, for its columns
    for sql in [
        "SELECT t.id FROM t JOIN (VALUES ((SELECT 1)), ((SELECT 2))) AS v(n) \
         ON t.id = v.n WHERE $1 AND v.n = 1 ORDER BY t.id",
        "SELECT t.id FROM t JOIN generate_series(1, (SELECT 2)) AS g \
         ON t.id = g.value WHERE $1 AND g.value = 1 ORDER BY t.id",
    ] {
        let stmt = db.prepare(sql).unwrap();
        let ids = |keep: bool| -> Vec<i64> {
            stmt.query((keep,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect()
        };
        assert_eq!(ids(false), Vec::<i64>::new(), "{sql} with false");
        assert_eq!(ids(true), [1], "{sql} with true after false");
        assert_eq!(ids(false), Vec::<i64>::new(), "{sql} with false again");
        assert_eq!(ids(true), [1], "{sql} with true again");
    }
}

#[test]
fn a_table_function_side_read_for_its_columns_generates_no_row() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_source_subquery_no_row").unwrap();
    // Read in full, the series would pass the rows a series may generate
    let sql = "SELECT g.value FROM generate_series(1, (SELECT 20000000)) AS g(value) \
               JOIN (VALUES (1)) AS v(x) ON 1 = 1 WHERE 1 = 0 LIMIT 1";
    for run in 0..2 {
        let before = tvf_rows();
        let result = db.query(sql, ()).unwrap();
        assert_eq!(result.columns(), ["value"], "{sql}, run {run}");
        assert_eq!(result.count(), 0, "{sql}, run {run}");
        assert_eq!(tvf_rows() - before, 0, "{sql}, run {run}: rows generated");
    }
}
