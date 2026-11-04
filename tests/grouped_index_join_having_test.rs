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

//! The HAVING of a grouped join a LIMIT reduces is compiled over the
//! group rows without the evaluator bridge

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::test_failpoints::bridge_compiles;
use stoolap::Database;

// The bridge counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Each statement, its parameters and the rows it answers in some order,
/// every value written out: u 1 has three orders, u 2 one, u 3 two, u 4 none
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT u.id, COUNT(o.id) FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id HAVING COUNT(o.id) > 1 LIMIT 5",
        &[],
        &["1|3", "3|2"],
    ),
    (
        "SELECT u.id, u.name, SUM(o.v) AS s FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id, u.name HAVING SUM(o.v) > $1 LIMIT 5",
        &[25],
        &["1|a|60", "3|c|50"],
    ),
    (
        "SELECT u.id, u.name, SUM(o.v) AS s FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id, u.name HAVING SUM(o.v) > $1 LIMIT 5",
        &[55],
        &["1|a|60"],
    ),
    (
        "SELECT u.id, MAX(o.v) FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id HAVING MAX(o.v) - MIN(o.v) >= 10 LIMIT 5",
        &[],
        &["1|30", "3|30"],
    ),
];

fn rows(db: &Database, sql: &str, params: &[i64]) -> Vec<String> {
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let mut rows = db
        .query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|at| row.get::<Value>(at).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[test]
fn a_reduced_grouped_join_having_answers_as_written_without_the_bridge() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://grouped_index_join_having").unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, name TEXT)",
        "INSERT INTO u VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, uid INTEGER, v INTEGER)",
        "CREATE INDEX o_uid ON o (uid)",
        "INSERT INTO o VALUES (1, 1, 10), (2, 1, 20), (3, 1, 30), (4, 2, 5), (5, 3, 20), (6, 3, 30)",
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
                    "{sql} {params:?}, run {run}: {answer:?}, written {expected:?}"
                ));
            }
            if compiled != 0 {
                wrong.push(format!(
                    "{sql} {params:?}, run {run}: {compiled} bridge compiles"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
