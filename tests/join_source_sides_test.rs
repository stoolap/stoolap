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

//! A join with a side that is not a table, a CTE, a view, a derived table,
//! a VALUES list or a table function, runs its subqueries as the plan's
//! blocks and answers as the same join over a table holding the side's rows

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MECHANISMS: [GeneralPath; 15] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::AntiJoin,
    GeneralPath::ExistsSemiJoin,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
    GeneralPath::ScalarSubquery,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedExpression,
    GeneralPath::LiftedSubqueries,
];

fn hits() -> Vec<usize> {
    MECHANISMS
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// The answer of `sql` and the general path mechanisms it ran
fn run(db: &Database, sql: &str) -> (Result<Vec<String>, String>, Vec<GeneralPath>) {
    let before = hits();
    let got = answer(db, sql);
    let ran = MECHANISMS
        .iter()
        .zip(before.iter().zip(hits()))
        .filter(|(_, (before, now))| now > *before)
        .map(|(path, _)| *path)
        .collect();
    (got, ran)
}

/// t joins the side, w is what the subqueries read; the side's rows are
/// u's ids, 1 to 3
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
    "INSERT INTO w VALUES (1, 1, 1), (2, 2, 6), (3, 3, 5), (4, 1, 2), (5, 2, 2)",
    "CREATE VIEW uv AS SELECT id FROM u",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://join_source_sides_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// Each side as `{with}` and `{s}` write it, the side named s
const SIDES: &[(&str, &str, &str)] = &[
    ("cte", "WITH s AS (SELECT id FROM u) ", "s"),
    ("cte not inlined", "WITH s (id) AS (SELECT id FROM u) ", "s"),
    (
        "derived over every column",
        "",
        "(SELECT id, x FROM u) AS s",
    ),
    ("view", "", "uv AS s"),
    ("derived", "", "(SELECT id FROM u) AS s"),
    ("values", "", "(VALUES (1), (2), (3)) AS s (id)"),
    ("table function", "", "generate_series(1, 3) AS s (id)"),
];

/// Each statement over t and the side s
const SHAPES: &[&str] = &[
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a WHERE t.b > (SELECT MIN(y) * 10 FROM w)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a \
     WHERE EXISTS (SELECT 1 FROM w WHERE w.x = s.id AND w.y = t.id)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a WHERE s.id IN (SELECT x FROM w WHERE y > 4)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a \
     WHERE s.id < (SELECT MAX(w.x) FROM w WHERE w.y = s.id)",
    "SELECT t.id, s.id FROM t LEFT JOIN {s} ON s.id = t.a \
     WHERE s.id IS NULL OR s.id IN (SELECT x FROM w WHERE y = 2)",
    "SELECT t.id, s.id FROM t LEFT JOIN {s} ON s.id = t.a AND s.id < (SELECT MAX(x) FROM w)",
    "SELECT t.id, (SELECT COUNT(*) FROM w WHERE w.x = s.id) FROM t JOIN {s} ON s.id = t.a",
    "SELECT s.id, t.id FROM {s} JOIN t ON t.a = s.id WHERE NOT EXISTS (SELECT 1 FROM w WHERE w.y = t.id)",
    "SELECT s.id, t.id FROM {s} LEFT JOIN t ON t.a = s.id AND t.b > 10 \
     WHERE t.id IS NULL OR t.id IN (SELECT y FROM w)",
    "SELECT s.id, t.id FROM {s} JOIN t ON t.a = s.id \
     WHERE s.id + t.id > ALL (SELECT y FROM w WHERE w.x = s.id)",
];

/// The rows of `sql` as sorted text, or the error
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<Option<String>>(i) {
                    Ok(value) => value.unwrap_or_else(|| "NULL".into()),
                    Err(e) => format!("<{e}>"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
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
fn a_join_over_a_side_that_is_not_a_table_runs_its_blocks() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for &(kind, with, side) in SIDES {
            for shape in SHAPES {
                let sql = format!("{with}{}", shape.replace("{s}", side));
                let (want, table_ran) = run(&db, &shape.replace("{s}", "u AS s"));
                let (got, ran) = run(&db, &sql);
                if got != want {
                    wrong.push(format!(
                        "{context} {kind}: {sql}: gave {got:?}, a table side {want:?}"
                    ));
                }
                // The blocks' own projection counts as an evaluator, over a table side too
                let evaluator =
                    |path: &&GeneralPath| !matches!(path, GeneralPath::ProjectionEvaluator);
                let table_ran: Vec<_> = table_ran.iter().filter(evaluator).collect();
                if !table_ran.is_empty() {
                    wrong.push(format!(
                        "{context}: {shape} over a table: ran {table_ran:?}"
                    ));
                }
                let ran: Vec<_> = ran.iter().filter(evaluator).collect();
                if !ran.is_empty() {
                    wrong.push(format!("{context} {kind}: {sql}: ran {ran:?}"));
                }
                let line = prepared_line(&db, &sql);
                if line != "Prepared: yes" {
                    wrong.push(format!("{context} {kind}: {sql}: EXPLAIN {line}"));
                }
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
fn a_join_whose_limit_is_cut_into_a_cte_it_keeps_runs_as_explain_says() {
    let _guard = serial();
    let db = hot("limit_cut");
    let shape = "SELECT c.a, (SELECT MAX(x) FROM w) FROM c LEFT JOIN u ON u.id = c.a \
                 ORDER BY c.a LIMIT 2";
    for (with, line, old) in [
        (
            "WITH c AS (SELECT a, COUNT(*) AS n FROM t GROUP BY a) ",
            "Prepared: yes",
            false,
        ),
        (
            "WITH c (a, n) AS (SELECT a, COUNT(*) FROM t GROUP BY a) ",
            "Prepared: no (a WITH clause)",
            true,
        ),
    ] {
        let sql = format!("{with}{shape}");
        let (got, ran) = run(&db, &sql);
        let evaluator = |path: &&GeneralPath| !matches!(path, GeneralPath::ProjectionEvaluator);
        assert_eq!(got.map(|rows| rows.len()), Ok(2), "{sql}");
        assert_eq!(prepared_line(&db, &sql), line, "{sql}");
        assert_eq!(
            ran.iter().any(|path| evaluator(&path)),
            old,
            "{sql}: ran {ran:?}"
        );
    }
}
