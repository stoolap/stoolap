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

//! Every read a statement's scan takes gives the recorded answer, recorded
//! when a twin on another path gave the same, and every path is exercised:
//! one added without a case here fails

#![cfg(all(feature = "test-failpoints", feature = "parallel"))]

use stoolap::test_failpoints::{reset_scan_paths, scan_paths, scan_paths_taken};
use stoolap::Database;

use crate::oracle;

const RECORDED: oracle::Answers = answers!("scan_paths");

/// A statement on `path`, and its twin, which reads the same rows another way
const CASES: &[(&str, &str, &str)] = &[
    (
        "seq",
        "SELECT id, b FROM s WHERE g + 0 = 3",
        "SELECT id, b FROM s WHERE g = 3",
    ),
    (
        "index",
        "SELECT id FROM s WHERE g > 97",
        "SELECT id FROM s WHERE g + 0 > 97",
    ),
    (
        "parallel_filter",
        "SELECT id FROM s WHERE g + 0 > 97 ORDER BY b",
        "SELECT id FROM s WHERE g > 97 ORDER BY b",
    ),
];

/// The paths of the in-memory store's index reads: a table whose rows are
/// all sealed reads its empty store whole, and its volumes as no path names
const HOT_ONLY: &[&str] = &["index"];

fn open(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, g INTEGER, b TEXT)",
        "CREATE INDEX sg ON s (g)",
        "INSERT INTO s SELECT x.value, x.value % 100, 'b' || CAST(x.value % 13 AS TEXT) \
         FROM generate_series(1, 20000) x",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The column names, then the rows, sorted: ties of a case's order fall
/// either way
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let names = format!("{:?}", rows.columns());
    let mut found = rows
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok(format!(
                "{:?}",
                (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    found.sort();
    found.insert(0, names);
    Ok(found)
}

#[test]
fn every_scan_path_answers_as_its_twin_does() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    let layouts = [
        ("hot", open("memory://scan_paths", false)),
        (
            "sealed",
            open(&format!("file://{}", dir.path().display()), true),
        ),
    ];
    for (layout, db) in &layouts {
        for (path, sql, twin) in CASES {
            for run in ["first", "held"] {
                reset_scan_paths();
                let got = answer(db, sql);
                let expected = *layout == "hot" || !HOT_ONLY.contains(path);
                if scan_paths_taken().contains(path) != expected {
                    wrong.push(format!(
                        "{layout} {run}: {sql} took {:?}",
                        scan_paths_taken()
                    ));
                }
                let key = format!("{layout}: {sql}");
                if let Some(why) =
                    oracle::check_confirmed(&RECORDED, &key, &got, || answer(db, twin))
                {
                    wrong.push(format!("{run} {why}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn every_scan_path_has_a_case() {
    let missing: Vec<_> = scan_paths()
        .into_iter()
        .filter(|path| !CASES.iter().any(|(case, _, _)| case == path))
        .collect();
    assert!(missing.is_empty(), "scan paths without a case: {missing:?}");
}
