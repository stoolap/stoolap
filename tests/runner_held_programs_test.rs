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

//! A statement the runner reads compiles its programs once, with its plan:
//! a run after the first compiles nothing, through the evaluator bridge or
//! otherwise, and answers as the first did

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{bridge_compiles, program_compiles};
use stoolap::Database;

fn rows(db: &Database, sql: &str, params: &[i64]) -> Vec<String> {
    let params: Vec<stoolap::core::Value> = params
        .iter()
        .map(|p| stoolap::core::Value::Integer(*p))
        .collect();
    db.query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// Each statement, its parameters and its answer
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS v(k, s)",
        &[],
        &["Integer(1)|Text(\"a\")", "Integer(2)|Text(\"b\")"],
    ),
    (
        "SELECT k + 1 FROM (VALUES (1), (2), (3)) AS v(k) WHERE k > 1 ORDER BY k DESC LIMIT 1",
        &[],
        &["Integer(4)"],
    ),
    (
        "SELECT DISTINCT k FROM (VALUES (1), (1), (2)) AS v(k) ORDER BY k",
        &[],
        &["Integer(1)", "Integer(2)"],
    ),
    (
        "SELECT k FROM (VALUES ($1), ($2 * 10)) AS v(k)",
        &[3, 4],
        &["Integer(3)", "Integer(40)"],
    ),
    (
        "SELECT id FROM t WHERE id IN (SELECT k FROM (VALUES (1), (3)) AS v(k)) ORDER BY id",
        &[],
        &["Integer(1)", "Integer(3)"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (VALUES ($1)) AS v(k) WHERE k = t.id)",
        &[2],
        &["Integer(2)"],
    ),
    (
        "SELECT id, (SELECT MAX(k) FROM (VALUES (id), (5)) AS v(k)) FROM t ORDER BY id LIMIT 1",
        &[],
        &["Integer(1)|Integer(5)"],
    ),
];

#[test]
fn a_run_after_the_first_compiles_nothing() {
    let db = Database::open("memory://runner_held_programs").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for &(sql, params, answer) in CASES {
        let first = rows(&db, sql, params);
        let (bridge, programs) = (bridge_compiles(), program_compiles());
        let again = rows(&db, sql, params);
        let compiled = (bridge_compiles() - bridge, program_compiles() - programs);
        if compiled != (0, 0) {
            wrong.push(format!("{sql}: run again, compiled {compiled:?}"));
        }
        if first != answer || again != answer {
            wrong.push(format!(
                "{sql}: {first:?} then {again:?}, expected {answer:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Each statement whose VALUES block holds a value that fails to compile,
/// and its answer, or the error it raises when a row reads the block
const UNREAD: &[(&str, Result<&[&str], &str>)] = &[
    (
        "SELECT CASE WHEN id = 1 THEN 1 ELSE \
         (SELECT k FROM (VALUES ('a' REGEXP '[')) AS v(k) WHERE t.id = 1) END \
         FROM t WHERE id = 1",
        Ok(&["Integer(1)"]),
    ),
    (
        "SELECT id FROM t WHERE id = 9 AND EXISTS (SELECT 1 FROM (VALUES ('a' REGEXP '[')) AS v(k))",
        Ok(&[]),
    ),
    (
        "SELECT CASE WHEN id = 1 THEN 1 ELSE \
         (SELECT k FROM (VALUES ('a' REGEXP '[')) AS v(k) WHERE t.id = 1) END \
         FROM t WHERE id = 2",
        Err("Invalid regex"),
    ),
];

#[test]
fn a_values_block_raises_its_error_only_when_read() {
    let db = Database::open("memory://runner_held_programs_unread").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for &(sql, answer) in UNREAD {
        let got = db.query(sql, ()).and_then(|rows| {
            rows.map(|row| row.map(|row| format!("{:?}", row.get_value(0).unwrap())))
                .collect::<stoolap::Result<Vec<_>>>()
        });
        let right = match (&got, answer) {
            (Ok(rows), Ok(answer)) => rows == answer,
            (Err(e), Err(error)) => e.to_string().contains(error),
            _ => false,
        };
        if !right {
            wrong.push(format!("{sql}: {got:?}, expected {answer:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
