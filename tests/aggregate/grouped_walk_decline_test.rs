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

//! A volume whose side file would walk a WHERE for groups, which the walk
//! cannot answer, sends the statement to the scan before any volume's rows
//! are read; the side files are decided under the statement's deadline

#![cfg(feature = "test-failpoints")]

use stoolap::core::Error;
use stoolap::executor::DeadlineSite;
use stoolap::test_failpoints::{
    after_cold_view_taken, aggregate_paths_taken, count_deadline_charges, deadline_charges,
    keep_deadlines, pass_deadlines, reset_aggregate_paths, side_decisions, FailpointGuard,
};
use stoolap::Database;

/// The sorted rows of `sql`, the paths that answered it, and the charges
/// the volumes' aggregates made
fn run(db: &Database, sql: &str) -> (Vec<String>, Vec<&'static str>, usize) {
    reset_aggregate_paths();
    count_deadline_charges(DeadlineSite::StorageAggregate);
    let mut rows: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    let charges = deadline_charges();
    keep_deadlines();
    rows.sort();
    (rows, aggregate_paths_taken(), charges)
}

/// `w` indexed on `k` in two volumes: the newer, read first, holds k = 7
/// on all of its rows and scans; the older holds it on row 323753 alone
/// among keys spread over it, and walks
fn open(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}/decline?checkpoint_interval=0",
        dir.display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE INDEX w_k ON w (k)",
        "INSERT INTO w SELECT s.value, (s.value * 7919) % 600000, 1 \
         FROM generate_series(1, 600000) s",
        "PRAGMA CHECKPOINT",
        "INSERT INTO w SELECT s.value, 7, 2 FROM generate_series(600001, 700000) s",
        "PRAGMA CHECKPOINT",
        "PRAGMA query_timeout = 3600000",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn a_volume_that_would_walk_for_groups_declines_before_any_volume_is_read() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let (got, taken, charges) = run(&db, "SELECT g, COUNT(*) FROM w WHERE k = 7 GROUP BY g");
    assert_eq!(got, ["1|1", "2|100000"]);
    assert_eq!(taken, ["hash_groups"]);
    assert_eq!(charges, 0, "the scanning volume was read first");
    // A call the walk answers still walks the older volume
    let (got, taken, _) = run(&db, "SELECT COUNT(*) FROM w WHERE k = 7");
    assert_eq!(got, ["100001"]);
    assert_eq!(taken, ["storage_filtered"]);
}

#[test]
fn a_deadline_passed_before_the_side_file_decisions_stops_them() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let before = side_decisions();
    after_cold_view_taken(pass_deadlines);
    let got = db
        .query("SELECT g, COUNT(*) FROM w WHERE k = 7 GROUP BY g", ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    let decided = side_decisions();
    keep_deadlines();
    assert!(
        matches!(got, Err(Error::QueryCancelled)),
        "{:?}",
        got.map(|rows| rows.len())
    );
    assert_eq!(decided, before, "side files decided after the deadline");
}
