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

//! The unique check among an UPDATE's planned rows compares only pairs with
//! a sealed row in them, counted in pairs rather than timed

#![cfg(feature = "test-failpoints")]

use stoolap::storage::traits::Engine;
use stoolap::test_failpoints;
use stoolap::Database;

const HOT_ROWS: i64 = 2000;

#[test]
fn a_run_of_hot_rows_only_is_not_compared_pair_by_pair() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // No committed row holds the key every row is set to
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value + 10000 FROM generate_series(2, {})",
            HOT_ROWS + 1
        ),
        (),
    )
    .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    // The sealed row takes a key of its own; the hot rows share another,
    // which the commit refuses, not the statement
    let updated = table.update(None, &mut |mut row| {
        let sealed = row.get(0).and_then(|v| v.as_int64()) == Some(1);
        let key = if sealed { 7 } else { 5 };
        row.set(1, stoolap::core::Value::Integer(key)).unwrap();
        Ok((row, true))
    });
    let compared = test_failpoints::unique_pairs_compared();
    drop(table);
    txn.rollback().unwrap();
    assert_eq!(updated.unwrap(), HOT_ROWS as i32 + 1);
    assert!(
        compared <= HOT_ROWS as usize,
        "{compared} pairs compared for one sealed row and {HOT_ROWS} hot rows"
    );
}
