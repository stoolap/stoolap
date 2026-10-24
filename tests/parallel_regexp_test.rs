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

//! A REGEXP run by several threads at once, each through its own copy of
//! the program, answers as the pattern matched row by row does.

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

const ROWS: i64 = 30_000;

// The worker row count is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn text(id: i64) -> String {
    format!("v{id}")
}

fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = match sealed {
        false => Database::open(&format!("memory://{name}")).unwrap(),
        true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
    };
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 20, 'v' || CAST(value AS TEXT) \
             FROM generate_series(1, {ROWS})"
        ),
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

#[test]
fn a_regexp_filter_over_many_rows_answers_row_by_row() {
    let _guard = serial();
    let pattern = regex::Regex::new("^v1[0-9]*5$").unwrap();
    let want: Vec<i64> = (1..=ROWS)
        .filter(|id| pattern.is_match(&text(*id)))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = open("parallel_regexp_filter", sealed, &dir);
        for sql in [
            "SELECT id FROM t WHERE s REGEXP '^v1[0-9]*5$' ORDER BY id",
            "SELECT id FROM t WHERE NOT (s NOT REGEXP '^v1[0-9]*5$') ORDER BY id",
            "SELECT id FROM t WHERE s REGEXP '^v1[0-9]*5$' AND s REGEXP '5$' ORDER BY id",
        ] {
            let got: Vec<i64> = db
                .query(sql, ())
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(got, want, "sealed {sealed}: {sql}");
        }
    }
}

#[test]
fn a_regexp_in_a_lag_default_over_many_partitions_answers_row_by_row() {
    let _guard = serial();
    let ends_in_five = regex::Regex::new("5$").unwrap();
    let want: Vec<(i64, i64)> = (1..=ROWS)
        .map(|id| match id > 20 {
            true => (id, id - 20),
            false => (
                id,
                if ends_in_five.is_match(&text(id)) {
                    -1
                } else {
                    -2
                },
            ),
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let db = open("parallel_regexp_lag", false, &dir);
    let sql = "SELECT id, LAG(id, 1, CASE WHEN s REGEXP '5$' THEN -1 ELSE -2 END) \
               OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id";
    let got: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect();
    assert_eq!(got, want, "{sql}");
}

#[test]
fn an_error_beside_a_regexp_over_many_rows_is_the_statements() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open("parallel_regexp_error", false, &dir);
    let rows = db
        .query(OVERFLOWING, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    assert!(rows.is_err(), "{OVERFLOWING} overflows on its second row");
}

/// Overflows on the row with id 2, through the parallel filter's workers
const OVERFLOWING: &str =
    "SELECT id FROM t WHERE s REGEXP '^v' AND id * 9223372036854775807 > 0 ORDER BY id";

#[cfg(all(feature = "parallel", feature = "test-failpoints"))]
#[test]
fn a_worker_reads_no_row_after_the_first_error() {
    use stoolap::test_failpoints::worker_rows_filtered;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open("parallel_regexp_stops", false, &dir);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let before = worker_rows_filtered();
    let failed = pool
        .install(|| {
            db.query(OVERFLOWING, ())
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        })
        .is_err();
    let read = worker_rows_filtered() - before;
    assert!(failed, "{OVERFLOWING} overflows on its second row");
    assert!(
        (1..=2).contains(&read),
        "one worker read {read} rows: none means the filter took no worker, more \
         means it read past the error on row 2"
    );
}
