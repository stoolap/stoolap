// Copyright 2026 Stoolap Contributors
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

//! A join side sealed as it is read leaves no rows in the semantic cache
//! that its scan read by need

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::before_join_side_opened;
use stoolap::Database;

/// Arms a checkpoint through `other` at the `at`-th join side opened
fn checkpoint_at(at: usize, other: Database) {
    match at {
        1 => before_join_side_opened(move || {
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }),
        _ => before_join_side_opened(move || checkpoint_at(at - 1, other)),
    }
}

#[test]
fn a_side_sealed_before_its_scan_leaves_no_partial_rows_in_the_cache() {
    let mut wrong = Vec::new();
    for at in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY)",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, qty INTEGER, note TEXT)",
            "INSERT INTO p SELECT value FROM generate_series(1, 5)",
            "INSERT INTO o SELECT value, value % 5 + 1, 'n' FROM generate_series(1, 20)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        checkpoint_at(at, db.clone());
        let joined = db
            .query(
                "SELECT p.id, o.id FROM p JOIN o ON o.qty = p.id WHERE o.id < 10",
                (),
            )
            .unwrap()
            .count();
        let notes: Vec<Option<String>> = db
            .query("SELECT * FROM o WHERE o.id < 10", ())
            .unwrap()
            .map(|row| row.unwrap().get::<Option<String>>(2).unwrap())
            .collect();
        if joined != 9 || notes != vec![Some("n".to_string()); 9] {
            wrong.push(format!(
                "sealed at side {at}: {joined} pairs, notes {notes:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
