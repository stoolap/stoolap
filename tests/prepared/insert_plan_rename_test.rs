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

//! A compiled INSERT follows a column rename another handle commits once
//! the INSERT's transaction has begun

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

fn renamed_while_inserting(name: &str, renames: &'static [&'static str]) -> (Database, bool) {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    let insert = db.prepare("INSERT INTO t (id, a) VALUES ($1, $2)").unwrap();
    insert.execute((1, 10)).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_transaction_begun(move || {
        for sql in renames {
            other.execute(sql, ()).unwrap();
        }
    });
    let inserted = insert.execute((2, 20)).is_ok();
    (db, inserted)
}

#[test]
fn a_cached_insert_writes_the_column_named_after_a_swap() {
    let (db, inserted) = renamed_while_inserting(
        "insert_plan_rename_swap",
        &[
            "ALTER TABLE t RENAME COLUMN a TO c",
            "ALTER TABLE t RENAME COLUMN b TO a",
            "ALTER TABLE t RENAME COLUMN c TO b",
        ],
    );
    assert!(inserted, "the INSERT failed after the swap");
    let row = db
        .query("SELECT a, b FROM t WHERE id = 2", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let got = (
        row.get::<Option<i64>>(0).unwrap(),
        row.get::<Option<i64>>(1).unwrap(),
    );
    assert_eq!(got, (Some(20), None), "(a, b) of the inserted row");
}

#[test]
fn a_cached_insert_refuses_a_column_renamed_away() {
    let (_db, inserted) = renamed_while_inserting(
        "insert_plan_rename_away",
        &["ALTER TABLE t RENAME COLUMN a TO old_a"],
    );
    assert!(!inserted, "the INSERT wrote a column that no longer exists");
}
