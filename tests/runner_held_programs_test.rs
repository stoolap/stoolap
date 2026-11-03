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
        "SELECT x FROM (VALUES (\"A\"), (\"a\")) AS v(x)",
        &[],
        &["Text(\"A\")", "Text(\"a\")"],
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
    (
        "WITH c AS (SELECT id, a FROM t) SELECT t.id FROM t JOIN c ON c.id = t.id \
         WHERE c.a > 15 ORDER BY t.id",
        &[],
        &["Integer(2)", "Integer(3)"],
    ),
    (
        "SELECT t.id, v.s FROM t JOIN (VALUES (1, 'x'), (3, 'y')) AS v(k, s) ON v.k = t.id \
         WHERE v.s = 'y'",
        &[],
        &["Integer(3)|Text(\"y\")"],
    ),
    (
        "SELECT t.id FROM t JOIN (SELECT id, a FROM t) AS d ON d.id = t.id \
         WHERE d.a < $1 ORDER BY t.id",
        &[25],
        &["Integer(1)", "Integer(2)"],
    ),
    (
        "SELECT t.id FROM t JOIN tv ON tv.id = t.id WHERE tv.a = 30",
        &[],
        &["Integer(3)"],
    ),
    (
        "SELECT t.id FROM generate_series(1, 3) AS s(n) JOIN t ON t.id = s.n \
         WHERE s.n > 1 ORDER BY t.id",
        &[],
        &["Integer(2)", "Integer(3)"],
    ),
    ("SELECT SUM(a * $1) FROM t", &[2], &["Integer(120)"]),
    (
        "SELECT id % 2, SUM(a + $1) FROM t GROUP BY id % 2 ORDER BY 1",
        &[1],
        &["Integer(0)|Integer(21)", "Integer(1)|Integer(42)"],
    ),
    (
        "SELECT id, SUM(a * $1) OVER (ORDER BY id) FROM t ORDER BY id",
        &[2],
        &[
            "Integer(1)|Integer(20)",
            "Integer(2)|Integer(60)",
            "Integer(3)|Integer(120)",
        ],
    ),
    (
        "SELECT id, a * $1 + ROW_NUMBER() OVER (ORDER BY id) FROM t UNION ALL SELECT 9, 9 LIMIT 2",
        &[2],
        &["Integer(1)|Integer(21)", "Integer(2)|Integer(42)"],
    ),
];

/// Statements with an argument after the first that is no literal, which
/// each run reads once, their parameters and their answers
const READ_ARGUMENTS: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT STRING_AGG(CAST(a * $1 AS TEXT), CAST($1 AS TEXT) ORDER BY id) FROM t",
        &[2],
        &["Text(\"20240260\")"],
    ),
    (
        "SELECT id, LAG(a * $1, $1 - 1, a + $1) OVER (ORDER BY id) FROM t ORDER BY id",
        &[2],
        &[
            "Integer(1)|Integer(12)",
            "Integer(2)|Integer(20)",
            "Integer(3)|Integer(40)",
        ],
    ),
];

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
        "CREATE VIEW tv AS SELECT id, a FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// What a second run of `sql` compiled, through the bridge and in all; a
/// wrong answer of either run goes to `wrong`
fn compiled_again(
    db: &Database,
    (sql, params, answer): (&str, &[i64], &[&str]),
    wrong: &mut Vec<String>,
) -> (usize, usize) {
    let first = rows(db, sql, params);
    let (bridge, programs) = (bridge_compiles(), program_compiles());
    let again = rows(db, sql, params);
    let compiled = (bridge_compiles() - bridge, program_compiles() - programs);
    if first != answer || again != answer {
        wrong.push(format!(
            "{sql}: {first:?} then {again:?}, expected {answer:?}"
        ));
    }
    compiled
}

#[test]
fn a_run_after_the_first_compiles_nothing() {
    let db = open("runner_held_programs");
    let mut wrong = Vec::new();
    for &(sql, params, answer) in CASES {
        let compiled = compiled_again(&db, (sql, params, answer), &mut wrong);
        if compiled != (0, 0) {
            wrong.push(format!("{sql}: run again, compiled {compiled:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_run_reading_its_arguments_compiles_nothing_through_the_bridge() {
    let db = open("runner_held_programs_arguments");
    let mut wrong = Vec::new();
    for &(sql, params, answer) in READ_ARGUMENTS {
        let (bridge, _) = compiled_again(&db, (sql, params, answer), &mut wrong);
        if bridge != 0 {
            wrong.push(format!("{sql}: run again, {bridge} through the bridge"));
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

/// Each join holding no subquery, its parameter and its answer
const JOINS: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT t.id + v.a, CASE WHEN v.a > $1 THEN 'big' ELSE 'small' END \
         FROM t JOIN t AS v ON v.id = t.id ORDER BY t.id DESC LIMIT 2",
        &[25],
        &["Integer(33)|Text(\"big\")", "Integer(22)|Text(\"small\")"],
    ),
    (
        "SELECT DISTINCT t.a * 2 FROM t LEFT JOIN t AS v ON v.id = t.id + 1 \
         WHERE v.a IS NULL OR v.a > $1 ORDER BY t.a * 2",
        &[0],
        &["Integer(20)", "Integer(40)", "Integer(60)"],
    ),
];

#[test]
fn a_join_compiles_its_output_without_the_bridge() {
    let db = Database::open("memory://runner_held_programs_joins").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for &(sql, params, answer) in JOINS {
        let bridge = bridge_compiles();
        let first = rows(&db, sql, params);
        let compiled = bridge_compiles() - bridge;
        if compiled != 0 || first != answer {
            wrong.push(format!(
                "{sql}: {first:?}, expected {answer:?}, {compiled} bridge compiles"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
