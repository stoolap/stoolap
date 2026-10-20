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

//! Every operator that produces a join's rows gives the rows of the join
//! written so that no operator takes its keys, and every one registered has
//! a case here

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{join_paths, join_paths_taken, reset_join_paths};
use stoolap::Database;

/// l: id 1..30, k = id % 7, g = (id - 1) / 10; r: id 1..50, k = id % 7
/// (indexed), w = id % 5; bl and br: 12000 rows each, u = id unindexed,
/// v = id % 100
fn open() -> Database {
    let db = Database::open("memory://join_path_equivalence").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 50) g",
        "CREATE TABLE bl (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "CREATE TABLE br (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "INSERT INTO bl SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
        "INSERT INTO br SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each path, a statement it answers, and the same statement with keys no
/// path takes. A LIMIT without ORDER BY picks any rows, so those statements
/// count or check their rows instead of listing them
const CASES: &[(&str, &str, &str)] = &[
    (
        "index_nested_loop",
        "SELECT COUNT(*) FROM (SELECT l.id FROM l JOIN r ON l.k = r.k LIMIT 40) j",
        "SELECT COUNT(*) FROM (SELECT l.id FROM l JOIN r ON l.k + 0 = r.k + 0 LIMIT 40) j",
    ),
    (
        "batch_index_nested_loop",
        "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k = r.id",
        "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k + 0 = r.id + 0",
    ),
    (
        "grouped_index_join",
        "SELECT COUNT(*) FROM (SELECT l.id, COUNT(*) FROM l JOIN r ON l.k = r.k GROUP BY l.id LIMIT 5) j",
        "SELECT COUNT(*) FROM (SELECT l.id, COUNT(*) FROM l JOIN r ON l.k + 0 = r.k + 0 GROUP BY l.id LIMIT 5) j",
    ),
    (
        "limit_hash",
        "SELECT COUNT(*) FROM (SELECT l.id FROM l JOIN r ON l.g = r.w LIMIT 7) j",
        "SELECT COUNT(*) FROM (SELECT l.id FROM l JOIN r ON l.g + 0 = r.w + 0 LIMIT 7) j",
    ),
    (
        "parallel_hash",
        "SELECT COUNT(*), SUM(bl.id), SUM(br.id) FROM bl JOIN br ON bl.u = br.u",
        "SELECT COUNT(*), SUM(bl.id), SUM(br.id) FROM bl JOIN br ON bl.u = br.u + 0 AND br.v = bl.v",
    ),
    (
        "hash",
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w",
        "SELECT l.id, r.id FROM l JOIN r ON l.g + 0 = r.w + 0",
    ),
    (
        "nested_loop",
        "SELECT l.id, r.id FROM l JOIN r ON l.g < r.w",
        "SELECT l.id, r.id FROM l JOIN r ON l.g + 0 < r.w + 0",
    ),
];

/// The rows of `sql` as sorted text, or the error
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
}

#[test]
fn every_join_path_answers_as_the_keyless_join() {
    let db = open();
    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for &(path, sql, reference) in CASES {
        reset_join_paths();
        let got = rows(&db, sql);
        let taken = join_paths_taken();
        let want = rows(&db, reference);
        if !taken.contains(&path) {
            wrong.push(format!("{sql}: took {taken:?}, expected {path}"));
        }
        if got != want {
            wrong.push(format!("{sql}: got {got:?}, keyless {want:?}"));
        }
        covered.extend(taken);
    }
    for path in join_paths() {
        if !covered.contains(&path) {
            wrong.push(format!("no case takes the {path} path"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
