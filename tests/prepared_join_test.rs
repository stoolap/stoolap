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

//! A join statement's plan holds the residuals and the WHERE its join
//! compiles: run again, the join hands the evaluation bridge nothing

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{bridge_compiles, join_paths_taken, reset_join_paths};
use stoolap::Database;

fn open() -> Database {
    let db = Database::open("memory://prepared_join").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each statement and the join path it takes
const CASES: &[(&str, &str)] = &[
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < r.id",
        "hash",
    ),
    (
        "SELECT l.id, r.id FROM l LEFT JOIN r ON l.g = r.w AND l.id < r.id",
        "hash",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < r.id LIMIT 7",
        "limit_hash",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.g < r.w",
        "nested_loop",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE l.id + r.id > 40",
        "hash",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k AND l.id < r.id LIMIT 40",
        "index_nested_loop",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k WHERE l.id + r.id > 40 LIMIT 40",
        "index_nested_loop",
    ),
];

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{:?}|{:?}",
                row.get_value(0).unwrap(),
                row.get_value(1).unwrap()
            )
        })
        .collect();
    out.sort();
    out
}

#[test]
fn a_join_run_again_hands_the_bridge_nothing() {
    let db = open();
    let mut wrong = Vec::new();
    for &(sql, path) in CASES {
        let first = rows(&db, sql);
        let before = bridge_compiles();
        reset_join_paths();
        let again = rows(&db, sql);
        let compiled = bridge_compiles() - before;
        let taken = join_paths_taken();
        if taken != [path] {
            wrong.push(format!("{sql}: took {taken:?}, expected {path}"));
        }
        if compiled != 0 {
            wrong.push(format!("{sql}: run again, handed the bridge {compiled}"));
        }
        if again != first {
            wrong.push(format!("{sql}: run again gave {again:?}, first {first:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
