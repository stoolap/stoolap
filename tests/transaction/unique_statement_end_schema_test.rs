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

//! A statement's end judges its unique keys by indexes and volumes of the
//! layout the transaction's rows are held in

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

/// The outcome of an UPDATE that takes a key row 2 holds, sealed or not,
/// when `at` runs a column drop as the statement ends
fn judged_with_a_column_dropped(sealed: bool, at: fn(Box<dyn FnOnce()>)) -> String {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, pad INTEGER, u INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0, 10), (2, 0, 20)", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    let other = db.clone();
    at(Box::new(move || {
        other.execute("ALTER TABLE t DROP COLUMN pad", ()).unwrap();
    }));
    match db.execute("UPDATE t SET u = 20 WHERE id = 1", ()) {
        Err(stoolap::Error::UniqueConstraint { .. }) => "taken".to_string(),
        other => format!("{other:?}"),
    }
}

#[test]
fn a_column_dropped_as_the_statement_ends_leaves_its_key_judged() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let got = judged_with_a_column_dropped(sealed, |hook| {
            stoolap::test_failpoints::after_statement_indexes(hook)
        });
        if got != "taken" {
            wrong.push(format!("sealed {sealed}: {got}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_column_dropped_before_the_volumes_are_read_leaves_its_key_judged() {
    let got = judged_with_a_column_dropped(true, |hook| {
        stoolap::test_failpoints::before_statement_volumes(hook)
    });
    assert_eq!(got, "taken");
}
