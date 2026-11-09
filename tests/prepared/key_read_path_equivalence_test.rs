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

//! Every path that reads rows by the keys or row ids a statement names
//! gives the recorded answer, recorded when a twin that scans gave the
//! same, and every path is exercised: one added without a case here fails

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{key_read_paths, key_read_paths_taken, reset_key_read_paths};
use stoolap::Database;

use crate::oracle;

const RECORDED: oracle::Answers = answers!("key_read_paths");

/// A statement on `path`, and its twin, which compares the key computed
/// and so reads by a scan
const CASES: &[(&str, &str, &str)] = &[
    (
        "pk_get",
        "SELECT a, b FROM k WHERE id = 20",
        "SELECT a, b FROM k WHERE id + 0 = 20",
    ),
    (
        "pk_get",
        "SELECT a, b, _rowid_ FROM k WHERE _rowid_ = 30",
        "SELECT a, b, id FROM k WHERE id + 0 = 30",
    ),
    (
        "unique_get",
        "SELECT a, b FROM k WHERE u = 200",
        "SELECT a, b FROM k WHERE u + 0 = 200",
    ),
    (
        "key_members",
        "SELECT a FROM k WHERE id IN (10, 30, 40)",
        "SELECT a FROM k WHERE id + 0 IN (10, 30, 40)",
    ),
    (
        "key_members",
        "SELECT a, _rowid_ FROM k WHERE _rowid_ IN (30, 10)",
        "SELECT a, id FROM k WHERE id + 0 IN (30, 10)",
    ),
    (
        "row_ids",
        "SELECT b FROM n WHERE _rowid_ IN (1, 3, 9)",
        "SELECT b FROM n WHERE _rowid_ + 0 IN (1, 3, 9)",
    ),
    (
        "row_ids",
        "SELECT b, _rowid_ FROM n WHERE _rowid_ = 2",
        "SELECT b, _rowid_ FROM n WHERE _rowid_ + 0 = 2",
    ),
    (
        "block_point_rows",
        "SELECT o.a, (SELECT i.b FROM k i WHERE i.id = o.id AND i.a > 1) FROM k o",
        "SELECT o.a, (SELECT i.b FROM k i WHERE i.id + 0 = o.id AND i.a > 1) FROM k o",
    ),
    (
        "block_point_rows",
        "SELECT a FROM n WHERE a IN (SELECT i.a FROM k i WHERE i.id = n.a * 10)",
        "SELECT a FROM n WHERE a IN (SELECT i.a FROM k i WHERE i.id + 0 = n.a * 10)",
    ),
    (
        "block_point_value",
        "SELECT o.a, (SELECT i.b FROM k i WHERE i.id = o.id) FROM k o",
        "SELECT o.a, (SELECT i.b FROM k i WHERE i.id + 0 = o.id) FROM k o",
    ),
];

fn open(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE k (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, u INTEGER)",
        "CREATE UNIQUE INDEX ku ON k (u)",
        "CREATE TABLE n (a INTEGER, b TEXT)",
        "INSERT INTO k VALUES (10, 1, 'x', 100), (20, 2, 'y', 200), (30, 3, 'z', 300)",
        "INSERT INTO n VALUES (1, 'x'), (2, 'y'), (3, 'z')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The rows, sorted: no case orders them
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
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
    Ok(found)
}

#[test]
fn every_key_read_path_answers_as_a_scan_does() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    let layouts = [
        ("hot", open("memory://key_read_paths", false)),
        (
            "sealed",
            open(&format!("file://{}", dir.path().display()), true),
        ),
    ];
    for (layout, db) in &layouts {
        for (path, sql, twin) in CASES {
            // The first run plans the statement, and one naming `_rowid_` is
            // read by its entry's scan; the second runs the plan held
            for run in ["first", "held"] {
                reset_key_read_paths();
                let got = answer(db, sql);
                if run == "held" && !key_read_paths_taken().contains(path) {
                    wrong.push(format!(
                        "{layout} {run}: {sql} took {:?}",
                        key_read_paths_taken()
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
fn every_key_read_path_has_a_case() {
    let missing: Vec<_> = key_read_paths()
        .into_iter()
        .filter(|path| !CASES.iter().any(|(case, _, _)| case == path))
        .collect();
    assert!(
        missing.is_empty(),
        "key read paths without a case: {missing:?}"
    );
}
