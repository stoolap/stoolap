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

//! A GROUP BY of DATE_TRUNC or TIME_TRUNC of a TIMESTAMP column is grouped
//! by the table, in the hot rows and the volumes, and answers as recorded,
//! each answer confirmed by the hash grouping when it was recorded

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

use crate::oracle;

/// Rows out of time order: week, month, quarter and year edges, a leap
/// day, equal timestamps, NULLs, and timestamps before 1970, where
/// DATE_TRUNC rounds down and TIME_TRUNC toward 1970
const ROWS: &str = "INSERT INTO r VALUES \
    (1, 'a', '2024-01-01 00:00:00', 1), (2, 'a', '2024-01-07 23:59:59', 2), \
    (3, 'b', '2024-01-08 00:00:00', 3), (4, 'b', '2024-03-31 23:59:59.999', 4), \
    (5, 'a', '2024-04-01 00:00:00', 5), (6, 'b', '2024-06-15 10:37:45', 6), \
    (7, 'a', '2024-06-15 10:44:59', 7), (8, 'b', '2024-06-15 10:45:00', 8), \
    (9, 'a', '2024-12-31 23:59:59', 9), (10, 'b', '2025-01-01 00:00:00', 10), \
    (11, 'a', NULL, 11), (12, 'b', NULL, 12), \
    (13, 'a', '1969-12-31 23:59:59.5', 13), (14, 'b', '1969-07-20 20:17:40', 14), \
    (15, 'a', '2024-06-15 10:37:45', 15), (16, 'b', '2024-02-29 12:00:00', 16)";

/// The statements of every context, each with its twin the hash grouping
/// answers
const CASES: &[(&str, &str)] = &[
    (
        "SELECT DATE_TRUNC('day', ts), COUNT(*), SUM(v) FROM r GROUP BY DATE_TRUNC('day', ts)",
        "SELECT DATE_TRUNC('day', ts), COUNT(*), SUM(v) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('day', ts)",
    ),
    (
        "SELECT DATE_TRUNC('year', ts), COUNT(*), SUM(v) FROM r GROUP BY DATE_TRUNC('year', ts)",
        "SELECT DATE_TRUNC('year', ts), COUNT(*), SUM(v) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('year', ts)",
    ),
    (
        "SELECT DATE_TRUNC('quarter', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('quarter', ts)",
        "SELECT DATE_TRUNC('quarter', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('quarter', ts)",
    ),
    (
        "SELECT DATE_TRUNC('Month', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('Month', ts)",
        "SELECT DATE_TRUNC('Month', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('Month', ts)",
    ),
    (
        "SELECT DATE_TRUNC('week', ts), SUM(v) FROM r GROUP BY DATE_TRUNC('week', ts)",
        "SELECT DATE_TRUNC('week', ts), SUM(v) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('week', ts)",
    ),
    (
        "SELECT DATE_TRUNC('hour', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('hour', ts)",
        "SELECT DATE_TRUNC('hour', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('hour', ts)",
    ),
    (
        "SELECT DATE_TRUNC('minute', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('minute', ts)",
        "SELECT DATE_TRUNC('minute', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('minute', ts)",
    ),
    (
        "SELECT DATE_TRUNC('second', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('second', ts)",
        "SELECT DATE_TRUNC('second', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('second', ts)",
    ),
    (
        "SELECT TIME_TRUNC('15m', ts), COUNT(*), MIN(v) FROM r GROUP BY TIME_TRUNC('15m', ts)",
        "SELECT TIME_TRUNC('15m', ts), COUNT(*), MIN(v) FROM r WHERE id + 0 = id \
         GROUP BY TIME_TRUNC('15m', ts)",
    ),
    (
        "SELECT TIME_TRUNC('1h30m', ts), MAX(v) FROM r GROUP BY TIME_TRUNC('1h30m', ts)",
        "SELECT TIME_TRUNC('1h30m', ts), MAX(v) FROM r WHERE id + 0 = id \
         GROUP BY TIME_TRUNC('1h30m', ts)",
    ),
    (
        "SELECT TIME_TRUNC('500ms', ts), COUNT(*) FROM r GROUP BY TIME_TRUNC('500ms', ts)",
        "SELECT TIME_TRUNC('500ms', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY TIME_TRUNC('500ms', ts)",
    ),
    (
        "SELECT TIME_TRUNC('1s', ts), COUNT(*), SUM(v) FROM r GROUP BY TIME_TRUNC('1s', ts)",
        "SELECT TIME_TRUNC('1s', ts), COUNT(*), SUM(v) FROM r WHERE id + 0 = id \
         GROUP BY TIME_TRUNC('1s', ts)",
    ),
    (
        "SELECT DATE_TRUNC('month', ts) AS m, MIN(v), MAX(v), AVG(v) FROM r GROUP BY m ORDER BY m",
        "SELECT DATE_TRUNC('month', ts) AS m, MIN(v), MAX(v), AVG(v) FROM r WHERE id + 0 = id \
         GROUP BY m ORDER BY m",
    ),
    (
        "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM r GROUP BY 1 ORDER BY 1 DESC",
        "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM r WHERE id + 0 = id GROUP BY 1 ORDER BY 1 DESC",
    ),
    (
        "SELECT d, DATE_TRUNC('week', ts), COUNT(*), SUM(v) FROM r \
         GROUP BY d, DATE_TRUNC('week', ts)",
        "SELECT d, DATE_TRUNC('week', ts), COUNT(*), SUM(v) FROM r WHERE id + 0 = id \
         GROUP BY d, DATE_TRUNC('week', ts)",
    ),
    (
        "SELECT TIME_TRUNC('1h', ts), DATE_TRUNC('day', ts), COUNT(*) FROM r \
         GROUP BY TIME_TRUNC('1h', ts), DATE_TRUNC('day', ts)",
        "SELECT TIME_TRUNC('1h', ts), DATE_TRUNC('day', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY TIME_TRUNC('1h', ts), DATE_TRUNC('day', ts)",
    ),
    (
        "SELECT DATE_TRUNC('year', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('year', ts) \
         HAVING COUNT(*) > 1",
        "SELECT DATE_TRUNC('year', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('year', ts) HAVING COUNT(*) > 1",
    ),
    (
        "SELECT DATE_TRUNC('day', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('day', ts) \
         HAVING DATE_TRUNC('day', ts) > '2024-06-01'",
        "SELECT DATE_TRUNC('day', ts), COUNT(*) FROM r WHERE id + 0 = id \
         GROUP BY DATE_TRUNC('day', ts) HAVING DATE_TRUNC('day', ts) > '2024-06-01'",
    ),
    (
        "SELECT DATE_TRUNC('quarter', ts) FROM r GROUP BY 1",
        "SELECT DATE_TRUNC('quarter', ts) FROM r WHERE id + 0 = id GROUP BY 1",
    ),
    (
        "SELECT DATE_TRUNC('day', ts) AS dd, COUNT(*) FROM r GROUP BY dd \
         ORDER BY DATE_TRUNC('day', ts)",
        "SELECT DATE_TRUNC('day', ts) AS dd, COUNT(*) FROM r WHERE id + 0 = id GROUP BY dd \
         ORDER BY DATE_TRUNC('day', ts)",
    ),
];

/// The statements under a WHERE the table runs, which only a table with
/// volumes groups
const FILTERED: &[(&str, &str)] = &[
    (
        "SELECT DATE_TRUNC('month', ts), SUM(v) FROM r WHERE v > 2 GROUP BY DATE_TRUNC('month', ts)",
        "SELECT DATE_TRUNC('month', ts), SUM(v) FROM r WHERE v > 2 AND id + 0 = id \
         GROUP BY DATE_TRUNC('month', ts)",
    ),
    (
        "SELECT d, TIME_TRUNC('1h', ts), COUNT(*) FROM r WHERE ts >= '2024-06-01' \
         GROUP BY d, TIME_TRUNC('1h', ts)",
        "SELECT d, TIME_TRUNC('1h', ts), COUNT(*) FROM r WHERE ts >= '2024-06-01' AND id + 0 = id \
         GROUP BY d, TIME_TRUNC('1h', ts)",
    ),
];

/// A table of `ROWS`: hot, sealed, or sealed and then written to, a row
/// added, one moved to another bucket and one deleted
fn open(db: Database, context: &str) -> Database {
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, d TEXT, ts TIMESTAMP, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(ROWS, ()).unwrap();
    if context != "memory" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if context == "mixed" {
        db.execute(
            "INSERT INTO r VALUES (17, 'a', '2024-06-15 11:00:00', 17)",
            (),
        )
        .unwrap();
        db.execute("UPDATE r SET ts = '2024-01-08 00:00:01' WHERE id = 6", ())
            .unwrap();
        db.execute("DELETE FROM r WHERE id = 9", ()).unwrap();
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

/// The column names and the rows in the order given, or the error, and
/// the paths that answered
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
            // Only an ORDER BY fixes the order of the groups
            if !sql.contains("ORDER BY") {
                body.sort();
            }
            out.extend(body);
            Ok(out)
        });
    (answer, aggregate_paths_taken())
}

const RECORDED: oracle::Answers = answers!("time_bucket_keys");

#[test]
fn a_bucket_of_a_timestamp_groups_in_the_table_as_recorded() {
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
                Database::open("memory://time_bucket_keys").unwrap(),
                "memory",
            ),
        ),
        ("sealed", open(file("sealed"), "sealed")),
        ("mixed", open(file("mixed"), "mixed")),
    ];
    let mut wrong = Vec::new();
    for (context, db) in &contexts {
        let filtered = FILTERED.iter().filter(|_| *context != "memory");
        for &(sql, reference) in CASES.iter().chain(filtered) {
            let (got, taken) = run(db, sql);
            if taken != ["storage_grouped"] {
                wrong.push(format!("{context}: {sql}: took {taken:?}"));
            }
            let confirm = || {
                let (want, reference_taken) = run(db, reference);
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
fn a_bucket_the_table_cannot_read_is_grouped_by_hash() {
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
        // A unit the call rejects errs as the call does
        "SELECT DATE_TRUNC('fortnight', ts), COUNT(*) FROM r GROUP BY 1",
        // A text column, an expression and a qualified column
        "SELECT DATE_TRUNC('day', d), COUNT(*) FROM r GROUP BY 1",
        "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM r GROUP BY TIME_TRUNC('1h', COALESCE(ts, ts))",
        "SELECT DATE_TRUNC('day', r.ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('day', r.ts)",
        // A HAVING naming the column outside its bucket
        "SELECT DATE_TRUNC('day', ts), COUNT(*) FROM r GROUP BY DATE_TRUNC('day', ts) \
         HAVING MAX(ts) > ts",
        // Another function of the column
        "SELECT EXTRACT(YEAR FROM ts), COUNT(*) FROM r GROUP BY EXTRACT(YEAR FROM ts)",
    ];
    for sql in declined {
        let (_, taken) = run(&db, sql);
        assert!(!taken.contains(&"storage_grouped"), "{sql}: took {taken:?}");
    }
    // A unit given as a parameter
    reset_aggregate_paths();
    let rows = db
        .query(
            "SELECT DATE_TRUNC($1, ts), COUNT(*) FROM r GROUP BY DATE_TRUNC($1, ts)",
            ("year",),
        )
        .unwrap()
        .count();
    assert_eq!(rows, 4);
    assert!(!aggregate_paths_taken().contains(&"storage_grouped"));
}
