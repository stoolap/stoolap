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

//! A sort key the rows of a finished SELECT carry in no column is compiled
//! over the rows' columns without the evaluator bridge

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::test_failpoints::bridge_compiles;
use stoolap::Database;

// The bridge counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Each statement, its parameters and the rows it answers, every value
/// written out
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT src.id, (-k) AS x, ROW_NUMBER() OVER (ORDER BY src.id) AS rn \
         FROM t AS src INNER JOIN s ON s.id = src.id ORDER BY x * 1",
        &[],
        &["1|-30|1", "3|-20|3", "2|-10|2"],
    ),
    (
        "SELECT src.id, (-k) AS x, ROW_NUMBER() OVER (ORDER BY src.id) AS rn \
         FROM t AS src INNER JOIN s ON s.id = src.id ORDER BY x * $1",
        &[-1],
        &["2|-10|2", "3|-20|3", "1|-30|1"],
    ),
    (
        "SELECT src.id, (-k) AS x, ROW_NUMBER() OVER (ORDER BY src.id) AS rn \
         FROM t AS src INNER JOIN s ON s.id = src.id ORDER BY x * $1",
        &[1],
        &["1|-30|1", "3|-20|3", "2|-10|2"],
    ),
    (
        "SELECT (SELECT (ROW_NUMBER() OVER () * 0) + c.a FROM child AS c \
         INNER JOIN side AS d ON d.k = c.a ORDER BY (p.b * 2 - 1) * c.a LIMIT 1) \
         FROM parent AS p ORDER BY p.id",
        &[],
        &["7", "5"],
    ),
    (
        "SELECT id, k FROM t UNION ALL SELECT 9, 0 ORDER BY k + 0",
        &[],
        &["9|0", "2|10", "3|20", "1|30"],
    ),
];

fn open() -> Database {
    let db = Database::open("memory://order_by_keys_bridge").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO t VALUES (1, 30), (2, 10), (3, 20)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY)",
        "INSERT INTO s VALUES (1), (2), (3)",
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO parent VALUES (1, 0), (2, 1)",
        "CREATE TABLE child (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO child VALUES (1, 5), (2, 7)",
        "CREATE TABLE side (k INTEGER PRIMARY KEY)",
        "INSERT INTO side VALUES (5), (7)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

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
fn a_computed_sort_key_answers_as_written_without_the_bridge() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = open();
    let mut wrong = Vec::new();
    for (sql, params, expected) in CASES {
        let before = bridge_compiles();
        let answer = rows(&db, sql, params);
        let compiled = bridge_compiles() - before;
        if answer != *expected {
            wrong.push(format!(
                "{sql} {params:?}: {answer:?}, written {expected:?}"
            ));
        }
        if compiled != 0 {
            wrong.push(format!("{sql} {params:?}: {compiled} bridge compiles"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
