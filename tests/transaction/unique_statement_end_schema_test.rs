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

//! A statement's end judges its unique keys by indexes of the layout the
//! transaction's rows are held in

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

#[test]
fn a_column_dropped_as_the_statement_ends_leaves_its_key_judged() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
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
        stoolap::test_failpoints::after_statement_indexes(move || {
            other.execute("ALTER TABLE t DROP COLUMN pad", ()).unwrap();
        });
        let updated = db.execute("UPDATE t SET u = 20 WHERE id = 1", ());
        if !matches!(updated, Err(stoolap::Error::UniqueConstraint { .. })) {
            wrong.push(format!("sealed {sealed}: {updated:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
