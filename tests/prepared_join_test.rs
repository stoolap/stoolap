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

//! A join statement's plan holds what its join compiles: the residuals, the
//! WHERE, the ORDER BY keys and the projection. Run again, it compiles nothing

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{
    bridge_compiles, join_paths_taken, program_compiles, reset_join_paths,
};
use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
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
    ("SELECT l.id + r.id, r.w FROM l JOIN r ON l.g = r.w", "hash"),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w ORDER BY l.id + r.id, r.id",
        "hash",
    ),
    (
        "SELECT l.id * 2 AS x, r.id FROM l JOIN r ON l.g = r.w ORDER BY x, r.id",
        "hash",
    ),
    (
        "SELECT l.id + r.id, r.w FROM l JOIN r ON l.g = r.w LIMIT 7",
        "limit_hash",
    ),
    (
        "SELECT l.id - r.id, CASE WHEN r.w > 2 THEN 1 ELSE 0 END FROM l JOIN r ON l.g < r.w",
        "nested_loop",
    ),
    (
        "SELECT l.id + r.id, r.w FROM l JOIN r ON l.k = r.k LIMIT 40",
        "index_nested_loop",
    ),
    (
        "SELECT l.id, r.id FROM l JOIN r ON l.k = r.id ORDER BY l.id + r.id, r.id LIMIT 40",
        "batch_index_nested_loop",
    ),
    (
        "SELECT l.id + r.w, r.id FROM l JOIN r ON l.k = r.id",
        "batch_index_nested_loop",
    ),
    (
        "SELECT l.id - r.w, r.id FROM r JOIN l ON r.id = l.k WHERE l.g = 1",
        "batch_index_nested_loop",
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
fn a_join_run_again_compiles_nothing() {
    let db = open("prepared_join_compiles_nothing");
    let mut wrong = Vec::new();
    for &(sql, path) in CASES {
        let first = rows(&db, sql);
        let before = bridge_compiles();
        let programs_before = program_compiles();
        reset_join_paths();
        let again = rows(&db, sql);
        let handed = bridge_compiles() - before;
        let compiled = program_compiles() - programs_before;
        let taken = join_paths_taken();
        if taken != [path] {
            wrong.push(format!("{sql}: took {taken:?}, expected {path}"));
        }
        if handed != 0 {
            wrong.push(format!("{sql}: run again, handed the bridge {handed}"));
        }
        if compiled != 0 {
            wrong.push(format!("{sql}: run again, compiled {compiled} programs"));
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

/// The first two columns of each row of a prepared statement, in its order
fn prepared_rows(db: &Database, sql: &str, param: i64) -> Vec<(i64, i64)> {
    db.prepare(sql)
        .unwrap()
        .query((param,))
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

#[test]
fn a_held_projection_reads_each_execution_s_parameter() {
    let db = open("prepared_join_projection_parameter");
    for sql in [
        "SELECT l.id + $1, r.id FROM l JOIN r ON l.g = r.w ORDER BY l.id, r.id",
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w ORDER BY l.id * $1, r.id",
        "SELECT l.id + $1, r.id FROM l JOIN r ON l.k = r.k ORDER BY l.id, r.id LIMIT 40",
        "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k ORDER BY l.id * $1, r.id LIMIT 40",
    ] {
        let one = prepared_rows(&db, sql, 1);
        let other = prepared_rows(&db, sql, -1);
        let one_again = prepared_rows(&db, sql, 1);
        assert_eq!(one, one_again, "{sql}: the same parameter twice");
        assert_ne!(
            one, other,
            "{sql}: another parameter kept the first one's rows"
        );
    }
}

#[test]
fn a_subquery_in_the_select_list_is_read_each_execution() {
    let db = open("prepared_join_select_subquery");
    let sql = "SELECT (SELECT MAX(id) FROM r), l.id FROM l JOIN r ON l.g = r.w LIMIT 1";
    let before: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    db.execute("INSERT INTO r VALUES (1000, 1, 99)", ())
        .unwrap();
    let after: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!((before, after), (vec![50], vec![1000]));
}

#[test]
fn an_index_join_reading_its_left_side_by_key_asks_the_rest_of_the_on() {
    let db = Database::open("memory://prepared_join_swapped_residual").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO l VALUES (1, 999)",
        "INSERT INTO r VALUES (2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in [
        "SELECT l.id, r.id, (SELECT 1) FROM l JOIN r ON l.id = r.k AND l.k = r.id WHERE r.id = 2",
        "SELECT l.id, r.id FROM l JOIN r ON l.id = r.k AND l.k = r.id WHERE r.id = 2",
    ] {
        reset_join_paths();
        assert_eq!(rows(&db, sql), Vec::<String>::new(), "{sql}");
        let taken = join_paths_taken();
        assert!(
            taken.iter().any(|path| path.contains("index")),
            "{sql}: took {taken:?}"
        );
    }
}
