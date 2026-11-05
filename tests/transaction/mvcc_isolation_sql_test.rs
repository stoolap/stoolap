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

//! MVCC Isolation SQL Tests
//!
//! Tests transaction isolation through the SQL interface
//! Uses BEGIN/COMMIT/ROLLBACK statements (matching working transaction_test.rs pattern)

use stoolap::Database;

/// Test that uncommitted changes are not visible to other connections
/// Note: This test uses the db.begin() API which has known limitations
/// The test verifies rollback works correctly
#[test]
fn test_uncommitted_not_visible() {
    let db = Database::open("memory://mvcc_uncommitted").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE test_isolation (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute(
        "INSERT INTO test_isolation (id, value) VALUES (1, 'initial')",
        (),
    )
    .expect("Failed to insert");

    // Start a transaction and update
    let mut tx = db.begin().expect("Failed to begin transaction");
    tx.execute(
        "UPDATE test_isolation SET value = 'updated' WHERE id = 1",
        (),
    )
    .expect("Failed to update in tx");

    // Query from main connection should still see old value
    let result = db
        .query("SELECT value FROM test_isolation WHERE id = 1", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        // Should see 'initial' since tx hasn't committed
        assert_eq!(value, "initial", "Should see initial value before commit");
    }

    // Rollback the transaction
    tx.rollback().expect("Failed to rollback");
}

/// Test that committed changes are visible (using statement-based transactions)
#[test]
fn test_committed_visible() {
    let db = Database::open("memory://mvcc_committed").expect("Failed to create database");

    // Create table and insert
    db.execute(
        "CREATE TABLE test_isolation (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_isolation (id, value) VALUES (1, 'initial')",
        (),
    )
    .expect("Failed to insert");

    // Start a transaction and update using statement-based transactions
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute(
        "UPDATE test_isolation SET value = 'committed' WHERE id = 1",
        (),
    )
    .expect("Failed to update in tx");

    // Commit the transaction
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Query should now see the committed value
    let result = db
        .query("SELECT value FROM test_isolation WHERE id = 1", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(value, "committed", "Should see committed value");
    }
}

/// Test transaction rollback
#[test]
fn test_transaction_rollback() {
    let db = Database::open("memory://mvcc_rollback").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_rollback (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test_rollback (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert");

    // Start transaction, make changes, then rollback
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("UPDATE test_rollback SET value = 999 WHERE id = 1", ())
        .expect("Failed to update");
    db.execute("INSERT INTO test_rollback (id, value) VALUES (2, 200)", ())
        .expect("Failed to insert in tx");

    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    // Verify original state
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_rollback", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Should have only 1 row after rollback");

    let value: i64 = db
        .query_one("SELECT value FROM test_rollback WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(value, 100, "Value should be unchanged after rollback");
}

/// Test multiple inserts in a transaction
#[test]
fn test_transaction_multiple_inserts() {
    let db = Database::open("memory://mvcc_multi_insert").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_batch (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("INSERT INTO test_batch (id, name) VALUES (1, 'Alice')", ())
        .expect("Failed to insert 1");
    db.execute("INSERT INTO test_batch (id, name) VALUES (2, 'Bob')", ())
        .expect("Failed to insert 2");
    db.execute(
        "INSERT INTO test_batch (id, name) VALUES (3, 'Charlie')",
        (),
    )
    .expect("Failed to insert 3");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_batch", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Should have 3 rows after commit");
}

/// Test delete in transaction
#[test]
fn test_transaction_delete() {
    let db = Database::open("memory://mvcc_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_delete (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_delete (id, value) VALUES (1, 'one'), (2, 'two'), (3, 'three')",
        (),
    )
    .expect("Failed to insert");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("DELETE FROM test_delete WHERE id = 2", ())
        .expect("Failed to delete");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_delete", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Should have 2 rows after delete");

    // Verify id=2 is gone
    let count_2: i64 = db
        .query_one("SELECT COUNT(*) FROM test_delete WHERE id = 2", ())
        .expect("Failed to count");
    assert_eq!(count_2, 0, "Row with id=2 should be deleted");
}

/// Test update in transaction
#[test]
fn test_transaction_update() {
    let db = Database::open("memory://mvcc_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_update (id INTEGER PRIMARY KEY, counter INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test_update (id, counter) VALUES (1, 0)", ())
        .expect("Failed to insert");

    // Multiple updates in one transaction
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "UPDATE test_update SET counter = counter + 1 WHERE id = 1",
        (),
    )
    .expect("Failed to update 1");
    db.execute(
        "UPDATE test_update SET counter = counter + 1 WHERE id = 1",
        (),
    )
    .expect("Failed to update 2");
    db.execute(
        "UPDATE test_update SET counter = counter + 1 WHERE id = 1",
        (),
    )
    .expect("Failed to update 3");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let counter: i64 = db
        .query_one("SELECT counter FROM test_update WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(counter, 3, "Counter should be 3 after three increments");
}

/// Test reading own uncommitted changes within transaction
/// Uses the db.begin() API to verify read-your-own-writes
#[test]
fn test_read_own_changes() {
    let db = Database::open("memory://mvcc_own_changes").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_own (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    let mut tx = db.begin().expect("Failed to begin");

    // Insert in transaction
    tx.execute("INSERT INTO test_own (id, value) VALUES (1, 'new')", ())
        .expect("Failed to insert");

    // Should be able to read our own uncommitted insert
    let result = tx
        .query("SELECT value FROM test_own WHERE id = 1", ())
        .expect("Failed to query in tx");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(value, "new");
        found = true;
    }
    assert!(found, "Should be able to read own uncommitted insert");

    tx.commit().expect("Failed to commit");
}

/// Test aggregate functions in transactions
#[test]
fn test_transaction_aggregates() {
    let db = Database::open("memory://mvcc_agg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_agg (id INTEGER PRIMARY KEY, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_agg (id, amount) VALUES (1, 100), (2, 200), (3, 300)",
        (),
    )
    .expect("Failed to insert");

    // Initial sum
    let initial_sum: i64 = db
        .query_one("SELECT SUM(amount) FROM test_agg", ())
        .expect("Failed to sum");
    assert_eq!(initial_sum, 600, "Initial sum should be 600");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("INSERT INTO test_agg (id, amount) VALUES (4, 400)", ())
        .expect("Failed to insert in tx");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let final_sum: i64 = db
        .query_one("SELECT SUM(amount) FROM test_agg", ())
        .expect("Failed to sum");
    assert_eq!(final_sum, 1000, "Final sum should be 1000");
}
