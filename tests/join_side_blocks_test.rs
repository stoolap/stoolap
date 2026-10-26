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

//! A join of two tables whose WHERE holds a subquery in a condition on one
//! side reads that side through the plan's blocks, on every join path: it
//! answers as the general path does, hot and sealed, and run again
//! compiles nothing

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::test_failpoints::{
    before_join_side_opened, bound_side_reads, bridge_compiles, join_paths_taken, program_compiles,
    reset_join_paths,
};
use stoolap::Database;

/// l: id 1..60, k = id % 7, g = (id - 1) / 10; r: id 1..80, k = id % 7
/// (indexed), w = id % 5; s: id 1..90, a = id % 31, b = id % 51
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 60) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 80) g",
        "INSERT INTO s SELECT g.value, g.value % 31, g.value % 51 FROM generate_series(1, 90) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Conditions on one side, each with an uncorrelated subquery
const FORMS: &[&str] = &[
    "l.k = (SELECT MIN(s.a) FROM s WHERE s.b < 5)",
    "l.id = (SELECT s.a FROM s WHERE s.b = 50)",
    "l.k IN (SELECT s.a FROM s WHERE s.b < 10)",
    "l.id NOT IN (SELECT s.a FROM s WHERE s.b < 20)",
    "EXISTS (SELECT 1 FROM s WHERE s.b = 50)",
    "l.g > ANY (SELECT s.a FROM s WHERE s.b = 33)",
    "l.k + 1 > (SELECT MIN(s.a) FROM s WHERE s.b < 5)",
    "r.w IN (SELECT s.a FROM s WHERE s.b < 30)",
    "r.k = (SELECT MIN(s.a) FROM s WHERE s.b < 5)",
];

struct Shape {
    sql: &'static str,
    /// The path it takes over hot rows, and over sealed ones
    paths: [&'static str; 2],
    /// The path when the condition does not read `l`, when that differs
    without_l: Option<&'static str>,
    /// The rows a LIMIT stops at, any of the general path's
    limit: Option<usize>,
    /// A side read narrowed by keys the join takes from the other side's
    /// rows compiles that list each run
    keys_from_rows: bool,
}

const fn shape(
    sql: &'static str,
    paths: [&'static str; 2],
    limit: Option<usize>,
    keys_from_rows: bool,
) -> Shape {
    Shape {
        sql,
        paths,
        without_l: None,
        limit,
        keys_from_rows,
    }
}

const SHAPES: &[Shape] = &[
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE {f}",
        ["hash", "hash"],
        None,
        false,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.k WHERE {f} LIMIT 20",
        ["index_nested_loop", "index_nested_loop"],
        Some(20),
        false,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.id WHERE {f}",
        ["batch_index_nested_loop", "batch_index_nested_loop"],
        None,
        false,
    ),
    Shape {
        without_l: Some("limit_hash"),
        ..shape(
            "SELECT r.id, l.id FROM {r} JOIN {l} ON r.id = l.k WHERE {f} LIMIT 20",
            ["index_nested_loop", "index_nested_loop"],
            Some(20),
            false,
        )
    },
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g < r.w WHERE {f}",
        ["nested_loop", "nested_loop"],
        None,
        false,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} LEFT JOIN {r} ON l.g = r.w WHERE {f}",
        ["hash", "hash"],
        None,
        false,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.g = r.w WHERE {f} \
         AND EXISTS (SELECT 1 FROM s WHERE s.a = l.k AND s.b >= r.w)",
        ["hash", "hash"],
        None,
        false,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.k WHERE {f} \
         AND EXISTS (SELECT 1 FROM s WHERE s.a = l.k AND s.b >= r.w) LIMIT 50",
        ["index_nested_loop", "index_nested_loop"],
        Some(50),
        false,
    ),
    shape(
        "SELECT l.id, COUNT(r.id) FROM {l} LEFT JOIN {r} ON l.k = r.k WHERE {f} \
         GROUP BY l.id LIMIT 5",
        ["grouped_index_join", "grouped_index_join"],
        Some(5),
        false,
    ),
    shape(
        "SELECT l.id, SUM(r.w + 1) FROM {l} LEFT JOIN {r} ON l.k = r.k WHERE {f} \
         GROUP BY l.id LIMIT 5",
        ["hash", "hash"],
        Some(5),
        true,
    ),
    shape(
        "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.k WHERE {f}",
        [
            if cfg!(feature = "test-filedb") {
                "hash"
            } else {
                "index_nested_loop"
            },
            "hash",
        ],
        None,
        true,
    ),
];

/// The first two values of each row of `sql` as sorted text, or the error
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

/// `sql` as a derived table's statement, which holds no plan, so the
/// general path answers it
fn general_path(sql: &str) -> String {
    format!("SELECT * FROM ({sql}) AS general")
}

/// Whether every row of `part` is in `whole`, as often as it is there
fn contained(part: &[String], whole: &[String]) -> bool {
    let mut left = whole.to_vec();
    part.iter()
        .all(|row| match left.iter().position(|w| w == row) {
            Some(at) => {
                left.swap_remove(at);
                true
            }
            None => false,
        })
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn a_join_reads_a_side_s_subqueries_through_its_blocks() {
    let mut wrong = Vec::new();
    let mut cases = 0;
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://join_side_blocks").unwrap();
            fill(&db);
            db
        };
        for shape in SHAPES {
            for form in FORMS {
                cases += 1;
                let sql = shape.sql.replace("{f}", form);
                let plain = sql.replace("{l}", "l").replace("{r}", "r");
                let general = match plain.rfind(" LIMIT ") {
                    Some(at) if shape.limit.is_some() => general_path(&plain[..at]),
                    _ => general_path(&plain),
                };
                let first = rows(&db, &plain);
                let compiles = (bridge_compiles(), program_compiles());
                reset_join_paths();
                let again = rows(&db, &plain);
                let taken = join_paths_taken();
                let compiled = (
                    bridge_compiles() - compiles.0,
                    program_compiles() - compiles.1,
                );
                let want = rows(&db, &general);
                if !want.as_ref().is_ok_and(|rows| !rows.is_empty()) {
                    wrong.push(format!("sealed {sealed}: {general}: answers {want:?}"));
                }
                let path = match shape.without_l {
                    Some(path) if !form.contains("l.") => path,
                    _ => shape.paths[usize::from(sealed)],
                };
                if taken != [path] {
                    wrong.push(format!("sealed {sealed}: {plain}: took {taken:?}"));
                }
                let line = prepared_line(&db, &plain);
                if line != "Prepared: yes" {
                    wrong.push(format!("sealed {sealed}: {plain}: {line}"));
                }
                if compiled != (0, 0) && !shape.keys_from_rows {
                    wrong.push(format!(
                        "sealed {sealed}: {plain}: run again compiled {compiled:?}"
                    ));
                }
                let answers = |got: &Result<Vec<String>, String>| match (got, &want, shape.limit) {
                    (Ok(got), Ok(want), Some(limit)) => {
                        got.len() == limit.min(want.len()) && contained(got, want)
                    }
                    (got, want, None) => got == want,
                    _ => false,
                };
                if !answers(&first) || !answers(&again) {
                    wrong.push(format!(
                        "sealed {sealed}: {plain}: gave {first:?} then {again:?}, the general path {want:?}"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {cases} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_condition_on_the_right_side_reads_the_right_side() {
    let db = Database::open("memory://join_side_blocks_right").unwrap();
    fill(&db);
    for (sql, same_as) in [
        (
            "SELECT r.id, l.id FROM r JOIN l ON r.w = l.g WHERE (l.k, l.g) IN ((1, 0), (2, 1))",
            "SELECT r.id, l.id FROM r JOIN l ON r.w = l.g \
             WHERE (l.k = 1 AND l.g = 0) OR (l.k = 2 AND l.g = 1)",
        ),
        (
            "SELECT r.id, l.id FROM r JOIN l ON r.w = l.g \
             WHERE l.g > ANY (SELECT s.a FROM s WHERE s.b = 33)",
            "SELECT * FROM (SELECT r.id, l.id FROM r JOIN l ON r.w = l.g \
             WHERE l.g > ANY (SELECT s.a FROM s WHERE s.b = 33)) AS general",
        ),
    ] {
        let want = rows(&db, same_as);
        assert!(want.as_ref().is_ok_and(|rows| !rows.is_empty()), "{want:?}");
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}

#[test]
fn a_side_s_subquery_errors_by_the_rule_of_its_form() {
    let db = Database::open("memory://join_side_blocks_errors").unwrap();
    fill(&db);
    let compared = "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
                    WHERE l.id < 0 AND l.k = (SELECT s.a FROM s)";
    let got = rows(&db, compared);
    assert!(
        got.as_ref().is_err_and(|e| e.contains("more than one row")),
        "a value a column is compared with is read before the scan: {got:?}"
    );
    let lazy = "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
                WHERE l.id < 0 AND l.k + 1 > (SELECT s.a FROM s)";
    for _ in 0..2 {
        assert_eq!(rows(&db, lazy), Ok(Vec::new()), "no row reads the block");
    }
}

#[test]
fn a_side_s_subquery_reads_the_transaction_s_own_writes() {
    let db = Database::open("memory://join_side_blocks_tx").unwrap();
    fill(&db);
    let sql = "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
               WHERE l.id IN (SELECT s.a FROM s WHERE s.b = 1000)";
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO s VALUES (1000, 3, 1000)", ())
        .unwrap();
    let got: Vec<String> = tx
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
    tx.rollback().unwrap();
    assert_eq!(
        got.len(),
        16,
        "l 3 (g 0) meets the 16 rows of r with w 0: {got:?}"
    );
}

#[test]
fn the_index_joins_answer_with_a_side_subquery_beside_their_own_rules() {
    let db = Database::open("memory://join_side_blocks_index_rules").unwrap();
    fill(&db);
    for (sql, path) in [
        (
            "SELECT l.id, SUM(r.w) FROM {l} LEFT JOIN {r} ON l.k = r.k \
             WHERE l.g > ANY (SELECT s.a FROM s WHERE s.b = 33) \
             AND l.id + r.id > (SELECT MIN(s.b) FROM s WHERE s.a = 3) GROUP BY l.id LIMIT 5",
            "grouped_index_join",
        ),
        (
            "SELECT l.id, r.id FROM {l} JOIN {r} ON l.k = r.k \
             WHERE r.k = 3 AND r.w IN (SELECT s.a FROM s WHERE s.b < 30) \
             AND l.g > ANY (SELECT s.a FROM s WHERE s.b = 33) LIMIT 20",
            "index_nested_loop",
        ),
    ] {
        let plain = sql.replace("{l}", "l").replace("{r}", "r");
        let general = general_path(&plain[..plain.rfind(" LIMIT ").unwrap()]);
        let want = rows(&db, &general).unwrap();
        reset_join_paths();
        let got = rows(&db, &plain).unwrap();
        assert_eq!(join_paths_taken(), [path], "{plain}");
        assert!(
            !got.is_empty() && got.len() <= want.len() && contained(&got, &want),
            "{plain}: {got:?}, the general path {want:?}"
        );
    }
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn an_index_join_fetches_its_inner_rows_through_their_conditions() {
    use stoolap::test_failpoints::inner_fetch_filters;
    let db = Database::open("memory://join_side_blocks_inner_fetch").unwrap();
    fill(&db);
    db.execute("INSERT INTO s VALUES (91, NULL, 77), (92, 2, 77)", ())
        .unwrap();
    for (condition, taken) in [
        ("r.w NOT IN (SELECT s.b FROM s WHERE s.a = 3)", true),
        ("r.w IN (SELECT s.a FROM s WHERE s.b < 30)", true),
        ("r.w NOT IN (SELECT s.a FROM s WHERE s.b = 77)", true),
        ("r.w IN (SELECT s.a FROM s WHERE s.b = 77)", true),
        ("r.w > ANY (SELECT s.a FROM s WHERE s.b = 33)", true),
        ("r.w <= ALL (SELECT s.a FROM s WHERE s.b = 33)", true),
        ("r.w = (SELECT MIN(s.a) FROM s WHERE s.b < 5)", true),
        (
            "r.id > 10 AND r.w NOT IN (SELECT s.b FROM s WHERE s.a = 3)",
            true,
        ),
        ("r.w + 0 > ANY (SELECT s.a FROM s WHERE s.b = 33)", false),
        (
            "r.w IN (SELECT s.a * 9223372036854775807 FROM s WHERE s.b = 33)",
            false,
        ),
    ] {
        let sql = format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON l.k = r.k WHERE {condition}");
        let plain = sql.replace("{l}", "l").replace("{r}", "r");
        let want = rows(&db, &general_path(&plain));
        reset_join_paths();
        let before = inner_fetch_filters();
        let got = rows(&db, &plain);
        assert_eq!(join_paths_taken(), ["index_nested_loop"], "{plain}");
        assert_eq!(
            inner_fetch_filters() > before,
            taken,
            "{plain}: taken by the fetch"
        );
        match (&got, &want) {
            (Err(got), Err(want)) => assert!(
                got.contains("overflow") && want.contains("overflow"),
                "{plain}: {got}, the general path {want}"
            ),
            _ => assert_eq!(got, want, "{plain}"),
        }
    }
}

#[test]
fn a_side_whose_rows_need_the_where_streams_it_bound_to_its_sets() {
    let db = Database::open("memory://join_side_blocks_bound").unwrap();
    fill(&db);
    db.execute("INSERT INTO s VALUES (91, NULL, 77), (92, 2, 77)", ())
        .unwrap();
    db.execute("UPDATE l SET g = 9223372036854775807 WHERE id = 33", ())
        .unwrap();
    for (condition, bound) in [
        ("l.g + 0 > ANY (SELECT s.a FROM s WHERE s.b = 33)", true),
        ("l.g + 1 > ANY (SELECT s.a FROM s WHERE s.b = 33)", true),
        ("l.k + 0 > ANY (SELECT s.a FROM s WHERE s.b = 33)", true),
        ("l.k + 0 NOT IN (SELECT s.a FROM s WHERE s.b < 20)", true),
        ("l.k + 0 NOT IN (SELECT s.a FROM s WHERE s.b = 77)", true),
        ("l.k + 0 IN (SELECT s.a FROM s WHERE s.b = 77)", true),
        ("l.k + 0 <= ALL (SELECT s.a FROM s WHERE s.b = 33)", true),
        (
            "l.k + 0 IN (SELECT s.a * 9223372036854775807 FROM s WHERE s.b = 33)",
            false,
        ),
        (
            "l.g + 0 > 0 AND l.k + 0 IN (SELECT s.a FROM s WHERE s.b < 30)",
            true,
        ),
    ] {
        let sql =
            format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON r.id = l.k WHERE {condition}");
        let plain = sql.replace("{l}", "l").replace("{r}", "r");
        let want = rows(&db, &general_path(&plain));
        let before = bound_side_reads();
        let got = rows(&db, &plain);
        assert_eq!(
            bound_side_reads() > before,
            bound,
            "{plain}: streamed bound"
        );
        match (&got, &want) {
            (Err(got), Err(want)) => assert!(
                got.contains("overflow") && want.contains("overflow"),
                "{plain}: {got}, the general path {want}"
            ),
            _ => assert_eq!(got, want, "{plain}"),
        }
    }
}

#[test]
fn a_side_whose_columns_changed_after_preparing_raises_schema_changed() {
    for (name, condition, table) in [
        (
            "join_side_blocks_drop_outer",
            "a.g IN (SELECT s.x FROM s)",
            "a",
        ),
        (
            "join_side_blocks_drop_inner",
            "b.w IN (SELECT s.x FROM s)",
            "b",
        ),
    ] {
        let db = Database::open(&format!("memory://{name}")).unwrap();
        for sql in [
            "CREATE TABLE a (id INTEGER PRIMARY KEY, pad INTEGER, k INTEGER, g INTEGER)",
            "CREATE TABLE b (id INTEGER PRIMARY KEY, pad INTEGER, k INTEGER, w INTEGER)",
            "CREATE TABLE s (id INTEGER PRIMARY KEY, x INTEGER)",
            "CREATE INDEX b_k ON b (k)",
            "INSERT INTO a SELECT g.value, 0, g.value % 7, g.value % 5 FROM generate_series(1, 60) g",
            "INSERT INTO b SELECT g.value, 0, g.value % 7, g.value % 5 FROM generate_series(1, 80) g",
            "INSERT INTO s VALUES (1, 1), (2, 3)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let sql = format!("SELECT a.id, b.id FROM a JOIN b ON a.k = b.k WHERE {condition}");
        let general = sql.replace(
            "FROM a JOIN b",
            "FROM (SELECT id, k, g FROM a) a JOIN (SELECT id, k, w FROM b) b",
        );
        let want = rows(&db, &general).unwrap();
        assert!(!rows(&db, &sql).unwrap().is_empty(), "{sql}: prepared");
        let writer = db.clone();
        let done = Rc::new(Cell::new(false));
        let flag = Rc::clone(&done);
        let alter = format!("ALTER TABLE {table} DROP COLUMN pad");
        before_join_side_opened(move || {
            writer.execute(&alter, ()).unwrap();
            flag.set(true);
        });
        let got = rows(&db, &sql);
        assert!(done.get(), "{sql}: the column dropped inside the statement");
        assert!(
            got.as_ref().is_err_and(|e| e.contains("schema changed")),
            "{sql}: {got:?}"
        );
        assert_eq!(rows(&db, &sql).unwrap(), want, "{sql}: run again");
    }
}
