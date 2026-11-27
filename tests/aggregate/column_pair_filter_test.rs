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

//! A WHERE comparing two columns of the row, such as `b > a`, is run by the
//! table: the volumes' aggregates read it from their columns and every scan
//! asks it of the row, and each answer is recorded only when the statement
//! with the comparison left to the expression VM gave the same rows

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

use crate::oracle;

/// NULLs on either side, an INTEGER equal to a FLOAT, and timestamps
const ROWS: &str = "INSERT INTO r VALUES \
    (1, 'x', 1, 2, 1.0, 2.5, '2024-01-01', '2024-01-02', 10), \
    (2, 'x', 2, 2, 2.0, 2.0, '2024-01-02', '2024-01-02', 20), \
    (3, 'y', 3, 1, 3.5, -1.0, '2024-01-03', '2024-01-01', 30), \
    (4, 'y', NULL, 5, NULL, 1.0, NULL, '2024-01-05', 40), \
    (5, 'z', 4, NULL, 4.0, NULL, '2024-01-04', NULL, 50), \
    (6, 'z', 9, 9, 9.0, 9.0, '2024-01-09', '2024-01-09', 60), \
    (7, 'x', -3, 7, -3.0, 7.5, '2023-12-31', '2024-01-07', 70), \
    (8, 'y', 6, 0, 6.0, 0.0, '2024-01-06', '2023-01-01', 80)";

/// Aggregates the volumes compute under the comparison
const AGGREGATES: &[&str] = &[
    "SELECT COUNT(*), SUM(v) FROM r WHERE b > a",
    "SELECT COUNT(*), MIN(v), MAX(v) FROM r WHERE a = f",
    "SELECT COUNT(*), SUM(v) FROM r WHERE a <> b AND v > 15",
    "SELECT d, COUNT(*), SUM(v) FROM r WHERE b > a GROUP BY d",
    "SELECT d, MIN(v), MAX(v) FROM r WHERE f <= g GROUP BY d",
    "SELECT d, COUNT(*) FROM r WHERE a >= f GROUP BY d",
    "SELECT d, SUM(v) FROM r WHERE ts < u AND d = 'x' GROUP BY d",
    "SELECT DATE_TRUNC('day', ts), COUNT(*) FROM r WHERE ts >= u GROUP BY DATE_TRUNC('day', ts)",
    "SELECT d, a, COUNT(*) FROM r WHERE a <= b GROUP BY d, a",
    // A timestamp and a number are never equal, and order as UNKNOWN
    "SELECT d, COUNT(*) FROM r WHERE ts <> a GROUP BY d",
    "SELECT COUNT(*) FROM r WHERE ts > a",
];

/// Statements whose rows a scan filters by the comparison
const SCANS: &[&str] = &[
    "SELECT id FROM r WHERE b > a ORDER BY id",
    "SELECT id FROM r WHERE NOT (b > a) ORDER BY id",
    "SELECT id FROM r WHERE b > a OR v = 50 ORDER BY id",
    "SELECT id FROM r WHERE NOT (a = b) ORDER BY id",
    "SELECT id FROM r WHERE f < g AND v > 15 ORDER BY id",
    "SELECT id, CASE WHEN v > 30 THEN 'big' ELSE 'small' END FROM r WHERE ts < u ORDER BY id",
    "SELECT d, COUNT(*) FROM r WHERE b > a GROUP BY d HAVING COUNT(*) > 1 ORDER BY d",
];

/// The comparisons, each as the VM answers it: its left column inside a
/// call storage does not take
const PAIRS: &[(&str, &str)] = &[
    ("b > a", "COALESCE(b, b) > a"),
    ("a = f", "COALESCE(a, a) = f"),
    ("a <> b", "COALESCE(a, a) <> b"),
    ("f <= g", "COALESCE(f, f) <= g"),
    ("a >= f", "COALESCE(a, a) >= f"),
    ("ts < u", "COALESCE(ts, ts) < u"),
    ("ts >= u", "COALESCE(ts, ts) >= u"),
    ("a <= b", "COALESCE(a, a) <= b"),
    ("f < g", "COALESCE(f, f) < g"),
    ("a = b", "COALESCE(a, a) = b"),
    ("ts <> a", "COALESCE(ts, ts) <> a"),
    ("ts > a", "COALESCE(ts, ts) > a"),
];

fn reference(sql: &str) -> String {
    PAIRS
        .iter()
        .fold(sql.to_string(), |sql, (pair, vm)| sql.replace(pair, vm))
}

/// A table of `ROWS`: hot, sealed, or sealed and then written to, a row
/// added, one whose comparison changes, and one deleted
fn open(db: Database, context: &str) -> Database {
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, d TEXT, a INTEGER, b INTEGER, \
         f FLOAT, g FLOAT, ts TIMESTAMP, u TIMESTAMP, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(ROWS, ()).unwrap();
    if context != "memory" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if context == "mixed" {
        for sql in [
            "INSERT INTO r VALUES (9, 'x', 5, 8, 5.0, 8.0, '2024-01-10', '2024-01-11', 90)",
            "UPDATE r SET b = 10 WHERE id = 3",
            "DELETE FROM r WHERE id = 1",
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

/// The column names and the rows, sorted unless the statement orders
/// them, or the error, and the aggregate paths that answered
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
            if !sql.contains("ORDER BY") {
                body.sort();
            }
            out.extend(body);
            Ok(out)
        });
    (answer, aggregate_paths_taken())
}

const RECORDED: oracle::Answers = answers!("column_pair_filter");

#[test]
fn a_comparison_of_two_columns_filters_in_the_table_as_recorded() {
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
                Database::open("memory://column_pair_filter").unwrap(),
                "memory",
            ),
        ),
        ("sealed", open(file("sealed"), "sealed")),
        ("mixed", open(file("mixed"), "mixed")),
    ];
    let mut wrong = Vec::new();
    for (context, db) in &contexts {
        for &sql in AGGREGATES.iter().chain(SCANS) {
            let (got, taken) = run(db, sql);
            // The volumes compute an aggregate; the hot store's rows are the scan's
            let stored = taken.iter().any(|path| path.starts_with("storage_"));
            if AGGREGATES.contains(&sql) && (*context != "memory") != stored {
                wrong.push(format!("{context}: {sql}: took {taken:?}"));
            }
            let vm = reference(sql);
            let confirm = || {
                let (want, vm_taken) = run(db, &vm);
                assert!(
                    !vm_taken.iter().any(|path| path.starts_with("storage_")),
                    "{context}: {vm}: took {vm_taken:?}"
                );
                want
            };
            let key = format!("{context}: {sql}");
            if let Some(error) = oracle::check_confirmed(&RECORDED, &key, &got, confirm) {
                wrong.push(error);
            }
        }
        // A parameter beside the comparison, prepared and run twice
        let prepared = db
            .prepare("SELECT d, COUNT(*) FROM r WHERE b > a AND v > $1 GROUP BY d ORDER BY d")
            .unwrap();
        for _ in 0..2 {
            let got: Vec<String> = prepared
                .query((15,))
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
            let want: &[&str] = match *context {
                "mixed" => &["x|i:2", "y|i:1"],
                _ => &["x|i:1"],
            };
            if got != want {
                wrong.push(format!("{context}: prepared: {got:?}"));
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
fn rows_written_by_a_comparison_of_two_columns_are_the_rows_it_holds_for() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dsn = match sealed {
            false => "memory://column_pair_filter_writes".to_string(),
            true => format!(
                "file://{}/writes?checkpoint_interval=0",
                dir.path().display()
            ),
        };
        let db = open(
            Database::open(&dsn).unwrap(),
            if sealed { "sealed" } else { "memory" },
        );
        db.execute("UPDATE r SET v = v + 1 WHERE b > a", ())
            .unwrap();
        db.execute("DELETE FROM r WHERE f >= g", ()).unwrap();
        let got: Vec<String> = db
            .query("SELECT id, v FROM r ORDER BY id", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                format!(
                    "{} {}",
                    render(row.get_value(0).unwrap()),
                    render(row.get_value(1).unwrap())
                )
            })
            .collect();
        // b > a holds for 1 and 7; f >= g for 2, 3, 6 and 8
        let want = ["i:1 i:11", "i:4 i:40", "i:5 i:50", "i:7 i:71"];
        if got != want {
            wrong.push(format!("sealed {sealed}: {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_selective_key_beside_two_columns_compared_walks_its_side_file() {
    use stoolap::executor::DeadlineSite;
    use stoolap::test_failpoints::{count_deadline_charges, deadline_charges, keep_deadlines};
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}/walk?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER, b INTEGER)",
        "CREATE INDEX w_k ON w (k)",
        // k spread over every row group, so no zone map narrows a key
        "INSERT INTO w SELECT g.value, (g.value * 7919) % 600000, g.value % 10, g.value % 7 \
         FROM generate_series(1, 600000) g",
        "PRAGMA CHECKPOINT",
        "PRAGMA query_timeout = 3600000",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Row 123450 holds k = 200550, a = 0 and b = 5; a scan charges each of ten row groups
    count_deadline_charges(DeadlineSite::StorageAggregate);
    let (got, taken) = run(
        &db,
        "SELECT COUNT(*), SUM(id) FROM w WHERE k = 200550 AND a < b",
    );
    let charges = deadline_charges();
    keep_deadlines();
    assert_eq!(
        got,
        Ok(vec![
            "columns: COUNT(*)|SUM(id)".to_string(),
            "i:1|i:123450".to_string()
        ])
    );
    assert_eq!(taken, ["storage_filtered"]);
    assert_eq!(charges, 1, "the side file names one row");
    // Row 123456 holds k = 248064, a = 6 and b = 4: named, then dropped by a < b
    let (got, taken) = run(
        &db,
        "SELECT COUNT(*), SUM(id) FROM w WHERE k = 248064 AND a < b",
    );
    assert_eq!(
        got,
        Ok(vec![
            "columns: COUNT(*)|SUM(id)".to_string(),
            "i:0|NULL".to_string()
        ])
    );
    assert_eq!(taken, ["storage_filtered"]);
}
