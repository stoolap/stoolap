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

//! COUNT, SUM, AVG, MIN and MAX of arithmetic over number columns, such as
//! `SUM(c * a)`, are computed by the table, in the hot rows and the
//! volumes, and answer as recorded, each answer confirmed by the general
//! path when it was recorded

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

use crate::oracle;

/// NULLs, a zero divisor, negatives, and INTEGER beside FLOAT
const ROWS: &str = "INSERT INTO r VALUES \
    (1, 'x', 3, 2, 1.5, 10, '2024-01-01 01:00:00'), \
    (2, 'x', 5, 0, 2.5, 20, '2024-01-01 02:30:00'), \
    (3, 'y', -4, 3, NULL, 30, '2024-01-02 00:00:00'), \
    (4, 'y', NULL, 7, 0.5, 40, '2024-01-02 05:00:00'), \
    (5, 'z', 6, -2, -1.25, NULL, '2024-01-03 00:00:00'), \
    (6, 'z', 8, 4, 3.0, 60, NULL), \
    (7, 'x', 0, 1, 0.0, 70, '2024-01-01 03:00:00')";

/// The statements of every context; each runs again with `id + 0 = id`
/// added to its WHERE, which the table does not run
const CASES: &[&str] = &[
    "SELECT SUM(a * b), SUM(a + f), COUNT(a / b), MIN(a - b), MAX(-a) FROM r",
    "SELECT AVG((b - a) / a), SUM(a % b), SUM(v * 2 + 1), AVG(f * f) FROM r",
    "SELECT COUNT(*), SUM(a * b) FROM r WHERE v > 15",
    "SELECT AVG(f * a) FROM r WHERE b > a",
    "SELECT d, SUM(a * b), AVG(a + 0.5), COUNT(b / a) FROM r GROUP BY d",
    "SELECT d, MIN(a * f), MAX(v - a) FROM r GROUP BY d HAVING SUM(a * b) > 0",
    "SELECT DATE_TRUNC('day', ts), SUM(v / 3), SUM(v % 3) FROM r GROUP BY DATE_TRUNC('day', ts)",
    "SELECT d, SUM(a * b) FROM r WHERE a > 0 GROUP BY d",
    "SELECT d, a, SUM(v * b), COUNT(*) FROM r GROUP BY d, a",
    "SELECT d, SUM((a - 1) * (b + 1)), SUM(a) FROM r GROUP BY d",
    "SELECT d, FIRST(a ORDER BY v), SUM(a * b), LAST(f ORDER BY v) FROM r GROUP BY d",
];

/// `sql` with `id + 0 = id` in its WHERE, which the table cannot run
fn general(sql: &str) -> String {
    let table = |name: &str| format!(" FROM {name}");
    for name in ["r", "o"] {
        let from = table(name);
        if sql.contains(" WHERE ") {
            return sql.replacen(" WHERE ", " WHERE id + 0 = id AND ", 1);
        }
        if sql.contains(&from) {
            return sql.replacen(&from, &format!("{from} WHERE id + 0 = id"), 1);
        }
    }
    sql.to_string()
}

/// `r` holds `ROWS` and `o` an INTEGER whose double overflows: hot, sealed,
/// or sealed and then written to, a row added, one changed, one deleted
fn open(db: Database, context: &str) -> Database {
    for sql in [
        "CREATE TABLE r (id INTEGER PRIMARY KEY, d TEXT, a INTEGER, b INTEGER, \
         f FLOAT, v INTEGER, ts TIMESTAMP)",
        ROWS,
        "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER)",
        "INSERT INTO o VALUES (1, 1, 9223372036854775807), (2, 1, 1), (3, 2, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if context != "memory" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if context == "mixed" {
        for sql in [
            "INSERT INTO r VALUES (8, 'y', 2, 5, 1.0, 80, '2024-01-02 06:00:00')",
            "UPDATE r SET b = 9 WHERE id = 2",
            "DELETE FROM r WHERE id = 6",
        ] {
            db.execute(sql, ()).unwrap();
        }
    }
    db
}

/// A value with its type, so INTEGER 3 and FLOAT 3.0 differ
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

/// The column names and the sorted rows, or the error, and the paths that
/// answered
fn run(db: &Database, sql: &str) -> (Result<Vec<String>, String>, Vec<&'static str>) {
    reset_aggregate_paths();
    let answer = db
        .query(sql, ())
        .map_err(|e| e.to_string())
        .and_then(|rows| {
            let mut out = vec![format!("columns: {}", rows.columns().join("|"))];
            let mut body = Vec::new();
            for row in rows {
                let row = row.map_err(|e| e.to_string())?;
                body.push(
                    (0..row.len())
                        .map(|i| render(row.get_value(i).unwrap()))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
            body.sort();
            out.extend(body);
            Ok(out)
        });
    (answer, aggregate_paths_taken())
}

const RECORDED: oracle::Answers = answers!("aggregate_arithmetic");

/// Whether the table computes `sql` in `context`: a memory table leaves the
/// rows a WHERE passes to the scan
fn computed(context: &str, sql: &str) -> bool {
    !(context == "memory" && sql.contains(" WHERE "))
}

#[test]
fn arithmetic_over_columns_aggregates_in_the_table_as_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let file = |name: &str| {
        Database::open(&format!(
            "file://{}/{name}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap()
    };
    let contexts = [
        (
            "memory",
            open(
                Database::open("memory://aggregate_arithmetic").unwrap(),
                "memory",
            ),
        ),
        ("sealed", open(file("sealed"), "sealed")),
        ("mixed", open(file("mixed"), "mixed")),
    ];
    // One row's double overflows: both paths fail with it
    let overflows = [
        "SELECT SUM(a * 2) FROM o",
        "SELECT g, SUM(a * 2) FROM o GROUP BY g",
    ];
    let mut wrong = Vec::new();
    for (context, db) in &contexts {
        for &sql in CASES.iter().chain(&overflows) {
            let (got, mut taken) = run(db, sql);
            // A failed statement names no path; the table erred, for it
            // computes the same call where no row overflows, and a table that
            // declines leaves no error
            if got.is_err() {
                taken = run(db, &sql.replace("a * 2", "a * 0")).1;
            }
            let stored = taken.iter().any(|path| path.starts_with("storage_"));
            if computed(context, sql) != stored {
                wrong.push(format!("{context}: {sql}: took {taken:?}"));
            }
            let reference = general(sql);
            let confirm = || {
                let (want, general_taken) = run(db, &reference);
                assert!(
                    !general_taken
                        .iter()
                        .any(|path| path.starts_with("storage_")),
                    "{context}: {reference}: took {general_taken:?}"
                );
                want
            };
            let key = format!("{context}: {sql}");
            if let Some(error) = oracle::check_confirmed(&RECORDED, &key, &got, confirm) {
                wrong.push(error);
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
fn a_parameter_in_the_arithmetic_is_read_at_each_run() {
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = match sealed {
            false => open(
                Database::open("memory://aggregate_arithmetic_parameter").unwrap(),
                "memory",
            ),
            true => open(
                Database::open(&format!(
                    "file://{}/parameter?checkpoint_interval=0",
                    dir.path().display()
                ))
                .unwrap(),
                "sealed",
            ),
        };
        let prepared = db
            .prepare("SELECT d, SUM(a * $1) FROM r GROUP BY d ORDER BY d")
            .unwrap();
        for (factor, want) in [
            (2, ["x|i:16", "y|i:-8", "z|i:28"]),
            (3, ["x|i:24", "y|i:-12", "z|i:42"]),
        ] {
            reset_aggregate_paths();
            let got: Vec<String> = prepared
                .query((factor,))
                .unwrap()
                .map(|row| {
                    let row = row.unwrap();
                    format!(
                        "{}|{}",
                        render(row.get_value(0).unwrap()),
                        render(row.get_value(1).unwrap())
                    )
                })
                .collect();
            assert_eq!(got, want, "sealed {sealed}, $1 = {factor}");
            assert_eq!(
                aggregate_paths_taken(),
                ["storage_grouped"],
                "sealed {sealed}"
            );
        }
    }
}

#[test]
fn a_limit_bounding_the_groups_leaves_the_arithmetic_of_rows_past_them_unrun() {
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = match sealed {
            false => Database::open("memory://aggregate_arithmetic_group_limit").unwrap(),
            true => Database::open(&format!(
                "file://{}/group_limit?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap(),
        };
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO c VALUES (1, 1, 5), (2, 2, 9223372036854775807), (3, 1, 7)",
            (),
        )
        .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let (got, taken) = run(&db, "SELECT g, SUM(a * 2) FROM c GROUP BY g LIMIT 1");
        assert_eq!(
            got,
            Ok(vec![
                "columns: g|SUM(a * 2)".to_string(),
                "i:1|i:24".to_string()
            ]),
            "sealed {sealed}: took {taken:?}"
        );
        assert!(
            !taken.iter().any(|path| path.starts_with("storage_")),
            "sealed {sealed}: took {taken:?}"
        );
    }
}

#[test]
fn an_argument_the_table_cannot_compute_is_left_to_the_general_path() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        "sealed",
    );
    let declined = [
        // A function, CASE, CAST, a timestamp, no column at all
        "SELECT d, SUM(ABS(a)) FROM r GROUP BY d",
        "SELECT d, SUM(CASE WHEN a > 0 THEN a ELSE 0 END) FROM r GROUP BY d",
        "SELECT d, SUM(CAST(a AS FLOAT) * 2) FROM r GROUP BY d",
        "SELECT d, MAX(ts + 1) FROM r GROUP BY d",
        "SELECT d, SUM(2 * 3) FROM r GROUP BY d",
        // DISTINCT, FILTER, a qualified column, a TEXT column
        "SELECT d, SUM(DISTINCT a * b) FROM r GROUP BY d",
        "SELECT d, SUM(a * b) FILTER (WHERE v > 0) FROM r GROUP BY d",
        "SELECT d, SUM(r.a * b) FROM r GROUP BY d",
        "SELECT SUM(a * d) FROM r",
    ];
    for sql in declined {
        let (_, taken) = run(&db, sql);
        assert!(
            !taken.iter().any(|path| path.starts_with("storage_")),
            "{sql}: took {taken:?}"
        );
    }
}
