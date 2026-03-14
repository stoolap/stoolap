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

//! Regression test for CURRENT_TRANSACTION_ID() returning NULL in WHERE clauses.
//!
//! Previously, RowFilter did not propagate the transaction_id from
//! ExecutionContext, causing CURRENT_TRANSACTION_ID() to return NULL
//! when used inside WHERE predicates evaluated through the parallel filter path.

use stoolap::Database;

/// Insert enough rows to exceed the parallel filter threshold (10,000).
/// This ensures tests exercise the parallel path where the bug was.
fn setup_large_table(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    // Batch insert 10,500 rows to exceed parallel threshold
    for batch_start in (1..=10_500).step_by(500) {
        let batch_end = (batch_start + 499).min(10_500);
        let values: Vec<String> = (batch_start..=batch_end)
            .map(|i| format!("({}, 'row{}')", i, i))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
}

#[test]
fn test_current_transaction_id_in_where_parallel() {
    let db = Database::open("memory://txn_id_parallel").unwrap();
    setup_large_table(&db);

    // CURRENT_TRANSACTION_ID() requires an explicit transaction to return non-NULL.
    // With >10,000 rows this hits the parallel filter path.
    let mut tx = db.begin().unwrap();
    let mut rows = tx
        .query(
            "SELECT COUNT(*) FROM t WHERE CURRENT_TRANSACTION_ID() > 0",
            (),
        )
        .unwrap();
    let count = if let Some(Ok(row)) = rows.next() {
        row.get::<i64>(0).unwrap()
    } else {
        0
    };
    drop(rows);
    tx.rollback().unwrap();

    assert_eq!(
        count, 10_500,
        "CURRENT_TRANSACTION_ID() > 0 should match all 10,500 rows via parallel filter"
    );
}

#[test]
fn test_current_transaction_id_in_where_small() {
    let db = Database::open("memory://txn_id_where").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();

    // Sequential path (small table)
    let mut tx = db.begin().unwrap();
    let mut rows = tx
        .query(
            "SELECT id FROM t WHERE CURRENT_TRANSACTION_ID() > 0 ORDER BY id",
            (),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    drop(rows);
    tx.rollback().unwrap();

    assert_eq!(
        ids,
        vec![1, 2, 3],
        "CURRENT_TRANSACTION_ID() > 0 should match all rows in a transaction"
    );
}

#[test]
fn test_current_transaction_id_is_not_null() {
    let db = Database::open("memory://txn_id_not_null").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();

    // IS NOT NULL check within a transaction
    let mut tx = db.begin().unwrap();
    let mut rows = tx
        .query(
            "SELECT id FROM t WHERE CURRENT_TRANSACTION_ID() IS NOT NULL ORDER BY id",
            (),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    drop(rows);
    tx.rollback().unwrap();

    assert_eq!(
        ids,
        vec![1, 2],
        "CURRENT_TRANSACTION_ID() IS NOT NULL should match all rows in a transaction"
    );
}

#[test]
fn test_current_transaction_id_null_in_autocommit() {
    let db = Database::open("memory://txn_id_autocommit").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();

    // In auto-commit mode (no explicit transaction), CURRENT_TRANSACTION_ID() returns NULL
    let mut rows = db
        .query("SELECT CURRENT_TRANSACTION_ID() IS NULL FROM t", ())
        .unwrap();
    if let Some(Ok(row)) = rows.next() {
        let is_null = row.get::<bool>(0).unwrap();
        assert!(
            is_null,
            "CURRENT_TRANSACTION_ID() should be NULL in auto-commit mode"
        );
    } else {
        panic!("Expected one row");
    }
}
