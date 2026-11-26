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

//! FIRST and LAST of a column by one ORDER BY column are computed by the
//! table, in the hot rows and the volumes, and answer as recorded, each
//! answer confirmed by the hash grouping when it was recorded

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

use crate::oracle;

/// Ties on ts with different values, NULL keys, NULL values, -0.0 beside
/// 0.0, and a group whose keys are all NULL
const ROWS: &str = "INSERT INTO r VALUES \
    (1, 'a', '2024-01-01 00:00:00', 10, 0.0, true), \
    (2, 'a', '2024-01-01 00:00:00', 5, -0.0, false), \
    (3, 'a', '2024-01-02 00:00:00', 7, 1.5, true), \
    (4, 'a', NULL, 99, 2.5, NULL), \
    (5, 'a', '2024-01-03 00:00:00', NULL, NULL, NULL), \
    (6, 'b', '2024-01-01 12:00:00', 20, 3.0, true), \
    (7, 'b', '2024-01-01 12:00:00', 30, 3.0, false), \
    (8, 'b', '2024-01-05 00:00:00', 1, -1.0, true), \
    (9, 'c', NULL, 40, NULL, NULL), \
    (10, 'c', NULL, 41, 0.5, true), \
    (11, 'd', '2024-01-01 00:15:00', 3, 0.25, false), \
    (12, 'd', '2024-01-01 00:20:00', 4, 0.75, true)";

/// The statements of every context; each runs again with `id + 0 = id`
/// added to its WHERE, which the hash grouping answers
const CASES: &[&str] = &[
    "SELECT d, FIRST(v ORDER BY ts), LAST(v ORDER BY ts) FROM r GROUP BY d",
    "SELECT d, FIRST(v ORDER BY ts DESC), LAST(v ORDER BY ts NULLS FIRST), \
     FIRST(v ORDER BY ts DESC NULLS LAST) FROM r GROUP BY d",
    "SELECT d, FIRST(f ORDER BY ts), LAST(f ORDER BY ts DESC NULLS LAST) FROM r GROUP BY d",
    "SELECT d, FIRST(b ORDER BY v), LAST(ts ORDER BY v) FROM r GROUP BY d",
    "SELECT d, FIRST(v ORDER BY f), LAST(v ORDER BY f) FROM r GROUP BY d",
    "SELECT d, FIRST(ts ORDER BY ts), LAST(ts ORDER BY ts) FROM r GROUP BY d",
    "SELECT d, FIRST(DISTINCT v ORDER BY ts), LAST(DISTINCT f ORDER BY ts) FROM r GROUP BY d",
    "SELECT TIME_TRUNC('1h', ts), FIRST(v ORDER BY ts), MAX(v), MIN(v), \
     LAST(v ORDER BY ts), COUNT(*) FROM r GROUP BY TIME_TRUNC('1h', ts)",
    "SELECT d, DATE_TRUNC('day', ts), FIRST(v ORDER BY ts), LAST(f ORDER BY ts) \
     FROM r GROUP BY d, DATE_TRUNC('day', ts)",
    "SELECT d, LAST(v ORDER BY ts) AS l FROM r GROUP BY d HAVING FIRST(v ORDER BY ts) > 4",
    "SELECT d, LAST(v ORDER BY ts) FROM r GROUP BY d HAVING LAST(v ORDER BY ts) < 40",
    "SELECT d AS dev, LAST(v ORDER BY ts) AS last_v, SUM(v) FROM r GROUP BY dev \
     ORDER BY last_v DESC",
];

/// The statements under a WHERE the table runs, which only a table with
/// volumes groups
const FILTERED: &[&str] = &[
    "SELECT d, FIRST(v ORDER BY ts), LAST(v ORDER BY ts) FROM r WHERE v > 4 GROUP BY d",
    "SELECT d, LAST(f ORDER BY ts) FROM r WHERE ts >= '2024-01-01 12:00:00' GROUP BY d",
];

/// `sql` with `id + 0 = id` in its WHERE, which the table cannot run
fn hashed(sql: &str) -> String {
    match sql.contains(" WHERE ") {
        true => sql.replacen(" WHERE ", " WHERE id + 0 = id AND ", 1),
        false => sql.replacen(" FROM r", " FROM r WHERE id + 0 = id", 1),
    }
}

/// A table of `ROWS`: hot, sealed, or sealed and then written to: a row
/// tied with sealed ones added, a group's last row moved to the hot store
/// with another value, and a group's first row deleted
fn open(db: Database, context: &str) -> Database {
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, d TEXT, ts TIMESTAMP, v INTEGER, \
         f FLOAT, b BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(ROWS, ()).unwrap();
    if context != "memory" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if context == "mixed" {
        for sql in [
            "INSERT INTO r VALUES (13, 'a', '2024-01-01 00:00:00', 1, 0.0, true)",
            "UPDATE r SET v = 50 WHERE id = 8",
            "DELETE FROM r WHERE id = 11",
        ] {
            db.execute(sql, ()).unwrap();
        }
    }
    db
}

/// A value with its type, so INTEGER 3 and FLOAT 3.0 differ, and a FLOAT
/// with its sign
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

/// The column names and the rows, sorted unless the statement orders
/// them, or the error, and the paths that answered
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
            // Only the statement's own ORDER BY fixes the order of the groups
            if !sql.contains("ORDER BY last_v") {
                body.sort();
            }
            out.extend(body);
            Ok(out)
        });
    (answer, aggregate_paths_taken())
}

const RECORDED: oracle::Answers = answers!("first_last_storage");

#[test]
fn first_and_last_by_a_column_group_in_the_table_as_recorded() {
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
                Database::open("memory://first_last_storage").unwrap(),
                "memory",
            ),
        ),
        ("sealed", open(file("sealed"), "sealed")),
        ("mixed", open(file("mixed"), "mixed")),
    ];
    let mut wrong = Vec::new();
    for (context, db) in &contexts {
        let filtered = FILTERED.iter().filter(|_| *context != "memory");
        for &sql in CASES.iter().chain(filtered) {
            let (got, taken) = run(db, sql);
            if taken != ["storage_grouped"] {
                wrong.push(format!("{context}: {sql}: took {taken:?}"));
            }
            let reference = hashed(sql);
            let confirm = || {
                let (want, reference_taken) = run(db, &reference);
                assert_eq!(reference_taken, ["hash_groups"], "{context}: {reference}");
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
fn a_first_or_last_the_table_cannot_read_is_grouped_by_hash() {
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
        // No ORDER BY, two keys, an expression key or argument
        "SELECT d, FIRST(v) FROM r GROUP BY d",
        "SELECT d, FIRST(v ORDER BY ts, f) FROM r GROUP BY d",
        "SELECT d, FIRST(v ORDER BY v + 0) FROM r GROUP BY d",
        "SELECT d, LAST(v + 0 ORDER BY ts) FROM r GROUP BY d",
        // A TEXT argument or key, a qualified key, a FILTER
        "SELECT d, LAST(d ORDER BY ts) FROM r GROUP BY d",
        "SELECT ts, FIRST(v ORDER BY d) FROM r GROUP BY ts",
        "SELECT d, FIRST(v ORDER BY r.ts) FROM r GROUP BY d",
        "SELECT d, FIRST(v ORDER BY ts) FILTER (WHERE v > 1) FROM r GROUP BY d",
    ];
    for sql in declined {
        let (_, taken) = run(&db, sql);
        assert!(!taken.contains(&"storage_grouped"), "{sql}: took {taken:?}");
    }
}
