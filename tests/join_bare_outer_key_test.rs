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

//! An index join probes its inner table by an outer column the ON names
//! bare, its side's name left out, as by one it names qualified; a bare
//! name the inner table has is the inner table's and probes nothing

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::test_failpoints::{hot_scan_step_rows, join_paths_taken, reset_join_paths};
use stoolap::Database;

// The counters are per thread, the join paths process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Each statement, its row count, and whether it reads its tables by key
const CASES: &[(&str, usize, bool)] = &[
    (
        "SELECT t.v FROM (SELECT 1 AS k) AS s JOIN t ON k = t.id",
        1,
        true,
    ),
    (
        "SELECT t.v FROM (SELECT 1 AS k) AS s JOIN t ON t.id = k",
        1,
        true,
    ),
    (
        "SELECT t.v FROM (SELECT 1 AS k) AS s JOIN t ON s.k = t.id",
        1,
        true,
    ),
    (
        "SELECT t.v FROM (VALUES (1)) AS s(k) JOIN t ON k = t.id",
        1,
        true,
    ),
    (
        "SELECT t.v FROM (SELECT 1 AS k) AS s LEFT JOIN t ON k = t.id",
        1,
        true,
    ),
    // id is t's own: the ON reads t.id = t.id
    (
        "SELECT t.v FROM (SELECT 1 AS k) AS s JOIN t ON t.id = id",
        2000,
        false,
    ),
];

// A file database orders this join t JOIN u first, reading t whole
#[cfg(not(feature = "test-filedb"))]
const MEMORY_CASES: &[(&str, usize, bool)] = &[(
    "SELECT t.v FROM (SELECT 1 AS k) AS s JOIN t ON k = t.id JOIN u ON u.tid = t.id",
    4,
    true,
)];
#[cfg(feature = "test-filedb")]
const MEMORY_CASES: &[(&str, usize, bool)] = &[];

#[test]
fn an_index_join_probes_by_a_bare_outer_column() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_bare_outer_key").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX u_tid ON u (tid)", ()).unwrap();
    for i in 1..=2000i64 {
        db.execute("INSERT INTO t VALUES ($1, $2, $3)", (i, i % 7, i * 10))
            .unwrap();
        db.execute(
            "INSERT INTO u VALUES ($1, $2, $3)",
            (i, (i % 500) + 1, format!("name{i}")),
        )
        .unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, count, by_key) in CASES.iter().chain(MEMORY_CASES) {
        for run in 0..2 {
            reset_join_paths();
            let before = hot_scan_step_rows();
            let got = db.query(sql, ()).unwrap().count();
            let scanned = hot_scan_step_rows() - before;
            let paths = join_paths_taken();
            if got != *count {
                wrong.push(format!("{sql}, run {run}: {got} rows, want {count}"));
            }
            let keyed = scanned == 0 && !paths.contains(&"nested_loop");
            if keyed != *by_key {
                wrong.push(format!(
                    "{sql}, run {run}: scanned {scanned} rows over {paths:?}, by key: want {by_key}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
