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

//! A prepared statement whose aggregate or window programs its plan holds
//! answers each execution with that execution's parameter

use stoolap::{Database, Value};

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 4, 'w'), (2, 7, 'x'), (3, 2, 'y'), (4, 9, 'z'), (5, 5, 'v')",
        (),
    )
    .unwrap();
    db
}

fn collect(rows: stoolap::Result<stoolap::Rows>) -> stoolap::Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    let mut rows = rows?;
    while rows.advance() {
        out.push(rows.current_row().as_slice().to_vec());
    }
    match rows.error() {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

const SHAPES: &[&str] = &[
    "SELECT SUM(a * $1) FROM t",
    "SELECT SUM(a) FILTER (WHERE a > $1), COUNT(*) FROM t",
    "SELECT a % 2, SUM(a * $1) FROM t GROUP BY a % 2 ORDER BY 1",
    "SELECT a % 2, SUM(a + $1) FROM t GROUP BY ROLLUP(a % 2) ORDER BY 1",
    "SELECT a, SUM(a * $1) OVER (ORDER BY a) FROM t ORDER BY a",
    "SELECT a, LAG(a, 1, a + $1) OVER (ORDER BY a) FROM t ORDER BY a",
    "SELECT a, ROW_NUMBER() OVER (PARTITION BY a % $1 ORDER BY a * $1) FROM t ORDER BY a",
    "SELECT a, a * $1 + ROW_NUMBER() OVER (ORDER BY a) FROM t ORDER BY a",
    "SELECT a * $1, ROW_NUMBER() OVER (ORDER BY a) FROM t ORDER BY 2",
    "SELECT a, SUM(a) FILTER (WHERE a > $1) OVER (ORDER BY a) FROM t ORDER BY a",
    "SELECT a, FIRST_VALUE(a + $1) OVER (ORDER BY a) FROM t ORDER BY a",
    "SELECT a, STRING_AGG(b, ',' ORDER BY a % $1, a) OVER () FROM t ORDER BY a",
];

#[test]
fn held_programs_read_each_execution_parameter() {
    let db = open("held_programs_rebind");
    let mut wrong = Vec::new();
    for sql in SHAPES {
        let stmt = db.prepare(sql).unwrap();
        for param in [2, 3, 5, 2] {
            let twin = sql.replace("$1", &param.to_string());
            let expected =
                collect(db.query(&twin, ())).unwrap_or_else(|e| panic!("literal twin {twin}: {e}"));
            let prepared = collect(stmt.query(vec![Value::Integer(param)]))
                .unwrap_or_else(|e| panic!("{sql} with {param}: {e}"));
            if prepared != expected {
                wrong.push(format!("{sql} with {param}: {prepared:?} != {expected:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn held_update_values_read_each_execution_parameter() {
    let db = open("held_update_rebind");
    let stmt = db.prepare("UPDATE t SET a = id * $1 + 1").unwrap();
    for param in [2, 3, 5] {
        stmt.execute(vec![Value::Integer(param)]).unwrap();
        let rows = collect(db.query("SELECT id, a FROM t ORDER BY id", ())).unwrap();
        let expected: Vec<Vec<Value>> = (1..=5)
            .map(|id| vec![Value::Integer(id), Value::Integer(id * param + 1)])
            .collect();
        assert_eq!(rows, expected, "UPDATE with {param}");
    }
}
