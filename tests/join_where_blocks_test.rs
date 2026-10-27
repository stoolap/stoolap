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

//! A join of two tables whose WHERE asks subqueries of its joined rows runs
//! them as the plan's blocks: it answers as its literal twin does, or as
//! recorded when the general path still gave the same answer, hot and
//! sealed, and run again compiles nothing

#![cfg(feature = "test-failpoints")]

mod oracle;

use stoolap::test_failpoints::{
    bridge_compiles, join_paths_taken, program_compiles, reset_join_paths,
};
use stoolap::Database;

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 50) g",
        "INSERT INTO s SELECT g.value, g.value % 31, g.value % 51 FROM generate_series(1, 90) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// The uncorrelated subquery a literal twin writes as its answer, and
/// whether it is an `IN` list
type Twin = Option<(&'static str, bool)>;

/// Each statement, with `{l}` and `{r}` for its sides, the join path it
/// takes over the two tables, and its literal twin's subquery
const CASES: &[(&str, &str, Twin)] = &[
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE r.id IN (SELECT s.b FROM s WHERE s.a = l.g)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE l.id < (SELECT MAX(s.b) FROM s WHERE s.a = r.w)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE l.id > ANY (SELECT s.b FROM s WHERE s.a = r.w)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE l.id < ALL (SELECT s.b FROM s WHERE s.a = r.w + 20)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g < r.w WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        "nested_loop",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} LEFT JOIN {r} ON l.g = r.w AND r.id < 20 WHERE r.id IS NULL OR EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} RIGHT JOIN {r} ON l.g = r.w AND l.id < 12 WHERE l.id IS NULL OR r.id IN (SELECT s.b FROM s WHERE s.a = l.g)",
        "hash",
        None,
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE l.id + r.id > (SELECT MIN(s.b) FROM s WHERE s.a = 5)",
        "hash",
        Some(("SELECT MIN(s.b) FROM s WHERE s.a = 5", false)),
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE l.id + r.id IN (SELECT s.b FROM s)",
        "hash",
        Some(("SELECT s.b FROM s", true)),
    ),
    (
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.id WHERE l.id + r.id > (SELECT MIN(s.b) FROM s WHERE s.a = 5)",
        "batch_index_nested_loop",
        Some(("SELECT MIN(s.b) FROM s WHERE s.a = 5", false)),
    ),
];

/// The answers of the correlated cases, recorded where the general path
/// gave the same ones
const RECORDED: oracle::Answers = answers!("join_where_blocks");

/// The rows of `sql` as sorted text, or the error
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!(
            "{:?}|{:?}",
            row.get_value(0).unwrap(),
            row.get_value(1).unwrap()
        ));
    }
    out.sort();
    Ok(out)
}

#[test]
fn a_join_where_runs_its_subqueries_as_blocks() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://join_where_blocks").unwrap();
            fill(&db);
            db
        };
        for &(shape, path, twin) in CASES {
            let sql = shape.replace("{l}", "l").replace("{r}", "r");
            let first = rows(&db, &sql);
            let bridge = bridge_compiles();
            let programs = program_compiles();
            reset_join_paths();
            let again = rows(&db, &sql);
            let taken = join_paths_taken();
            let compiled = (bridge_compiles() - bridge, program_compiles() - programs);
            if taken != [path] {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: took {taken:?}, expected {path}"
                ));
            }
            if compiled != (0, 0) {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: run again, compiled {compiled:?}"
                ));
            }
            if again != first {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: gave {first:?}, then {again:?}"
                ));
            }
            let answer = first.unwrap_or_else(|e| vec![format!("error: {e}")]);
            match twin {
                Some((subquery, list)) => {
                    let twin = oracle::literal_twin(&db, &sql, subquery, list);
                    let want = rows(&db, &twin);
                    if Ok(&answer) != want.as_ref() {
                        wrong.push(format!(
                            "sealed {sealed}: {sql}: gave {answer:?}, its literal twin {want:?}"
                        ));
                    }
                }
                None => wrong.extend(
                    oracle::check(&RECORDED, &sql, &answer)
                        .map(|why| format!("sealed {sealed}: {why}")),
                ),
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn an_uncorrelated_block_no_joined_row_reaches_is_not_read() {
    let db = Database::open("memory://join_where_blocks_lazy").unwrap();
    fill(&db);
    let shape = "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w AND l.id < 0 WHERE l.id + r.id > (SELECT s.b FROM s)";
    let sql = shape.replace("{l}", "l").replace("{r}", "r");
    let blocks = rows(&db, &sql);
    assert_eq!(blocks, Ok(Vec::new()), "no joined row reads the block");
}

#[test]
fn a_select_alias_the_where_names_is_checked_as_its_site_reads_it() {
    let db = Database::open("memory://join_where_blocks_alias").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO l VALUES (1)",
        "INSERT INTO r VALUES (1)",
        "INSERT INTO s VALUES (1, 10), (2, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT (SELECT b FROM s ORDER BY b LIMIT 1) AS x FROM l JOIN r ON l.id = r.id \
               WHERE x > 0 AND EXISTS (SELECT 1 FROM s WHERE s.id = l.id AND s.id = r.id)";
    for _ in 0..2 {
        let got: Result<Vec<i64>, String> =
            db.query(sql, ())
                .map_err(|e| e.to_string())
                .and_then(|rows| {
                    rows.map(|row| {
                        row.map_err(|e| e.to_string())?
                            .get::<i64>(0)
                            .map_err(|e| e.to_string())
                    })
                    .collect()
                });
        assert_eq!(
            got,
            Ok(vec![10]),
            "the alias's ORDER BY and LIMIT keep one row"
        );
    }
}

#[test]
fn a_column_named_with_a_dot_binds_by_position_inside_a_block() {
    let db = Database::open("memory://join_where_blocks_dotted").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, \"x.y\" INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE s (v INTEGER PRIMARY KEY)",
        "INSERT INTO l VALUES (1, 10)",
        "INSERT INTO r VALUES (1)",
        "INSERT INTO s VALUES (11)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let shape = "SELECT l.id, r.id FROM {l} JOIN {r} ON l.id = r.id \
                 WHERE EXISTS (SELECT 1 FROM s WHERE s.v = l.\"x.y\" + r.id)";
    let sql = shape.replace("{l}", "l").replace("{r}", "r");
    let want = Ok(vec!["Integer(1)|Integer(1)".to_string()]);
    for _ in 0..2 {
        assert_eq!(rows(&db, &sql), want, "{sql}");
    }
}

#[test]
fn a_join_opens_the_where_s_select_aliases_once() {
    let db = Database::open("memory://join_where_aliases_once").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "INSERT INTO l VALUES (1)",
        "INSERT INTO r VALUES (2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT 1 AS x, \"x\" AS y FROM l JOIN r ON l.id < r.id WHERE y = 'x'";
    for _ in 0..2 {
        let got: Vec<String> = db
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
        assert_eq!(
            got.len(),
            1,
            "y opens to the text \"x\", not to the alias x: {got:?}"
        );
    }
}
