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

//! Transaction Tests
//!
//! Tests for transaction functionality: BEGIN, COMMIT, ROLLBACK

use stoolap::Database;

#[test]
fn test_basic_commit() {
    let db = Database::open("memory://txn_basic_commit").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert data and commit (implicit transaction)
    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (1, 'test value')",
        (),
    )
    .expect("Failed to insert");

    // Verify data is visible
    let value: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(value, "test value");
}

#[test]
fn test_explicit_transaction_commit() {
    let db = Database::open("memory://txn_explicit_commit").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Start explicit transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    db.execute("INSERT INTO txn_test (id, value) VALUES (1, 'first')", ())
        .expect("Failed to insert first");
    db.execute("INSERT INTO txn_test (id, value) VALUES (2, 'second')", ())
        .expect("Failed to insert second");

    // Commit
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify both rows exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 2);
}

/// Rollback insert test
#[test]
fn test_explicit_transaction_rollback() {
    let db = Database::open("memory://txn_explicit_rollback").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert some initial data
    db.execute("INSERT INTO txn_test (id, value) VALUES (1, 'initial')", ())
        .expect("Failed to insert initial");

    // Start transaction and insert more
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (2, 'should rollback')",
        (),
    )
    .expect("Failed to insert in transaction");

    // Rollback
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    // Only initial row should exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "After rollback, only initial row should exist");

    let value: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(value, "initial");
}

#[test]
fn test_update_in_transaction() {
    let db = Database::open("memory://txn_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (1, 'original')",
        (),
    )
    .expect("Failed to insert");

    // Begin transaction and update
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("UPDATE txn_test SET value = 'modified' WHERE id = 1", ())
        .expect("Failed to update");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let value: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(value, "modified");
}

#[test]
fn test_delete_in_transaction() {
    let db = Database::open("memory://txn_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (1, 'to delete')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (2, 'keep this')",
        (),
    )
    .expect("Failed to insert");

    // Begin transaction and delete
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("DELETE FROM txn_test WHERE id = 1", ())
        .expect("Failed to delete");
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Only row 2 should remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 1);

    let value: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(value, "keep this");
}

#[test]
fn test_multiple_operations_in_transaction() {
    let db = Database::open("memory://txn_multi_ops").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Initial data
    db.execute("INSERT INTO txn_test (id, value) VALUES (1, 'a')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO txn_test (id, value) VALUES (2, 'b')", ())
        .expect("Failed to insert");

    // Transaction with multiple operations
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("INSERT INTO txn_test (id, value) VALUES (3, 'c')", ())
        .expect("Failed to insert");
    db.execute("UPDATE txn_test SET value = 'updated' WHERE id = 1", ())
        .expect("Failed to update");
    db.execute("DELETE FROM txn_test WHERE id = 2", ())
        .expect("Failed to delete");
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Should have 2 rows: id=1 with 'updated', id=3 with 'c'
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 2);

    let value1: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(value1, "updated");

    let value3: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 3", ())
        .expect("Failed to query");
    assert_eq!(value3, "c");
}

/// Rollback update test
#[test]
fn test_rollback_update() {
    let db = Database::open("memory://txn_rollback_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (1, 'original')",
        (),
    )
    .expect("Failed to insert");

    // Begin transaction, update, then rollback
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "UPDATE txn_test SET value = 'should not persist' WHERE id = 1",
        (),
    )
    .expect("Failed to update");
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    // Value should still be original
    let value: String = db
        .query_one("SELECT value FROM txn_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(
        value, "original",
        "Value should be unchanged after rollback"
    );
}

/// Rollback delete test
#[test]
fn test_rollback_delete() {
    let db = Database::open("memory://txn_rollback_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO txn_test (id, value) VALUES (1, 'should exist')",
        (),
    )
    .expect("Failed to insert");

    // Begin transaction, delete, then rollback
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute("DELETE FROM txn_test WHERE id = 1", ())
        .expect("Failed to delete");
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    // Row should still exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Row should still exist after rollback");
}

/// Nested BEGIN test - verifies that nested BEGIN is a no-op when a transaction is already active
#[test]
fn test_nested_begin_is_noop() {
    let db = Database::open("memory://txn_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .unwrap();

    // Start first transaction
    db.execute("BEGIN", ()).expect("First BEGIN should succeed");

    // Insert a row
    db.execute("INSERT INTO txn_test (id, value) VALUES (1, 'first')", ())
        .unwrap();

    // Second BEGIN is a no-op (doesn't start a new nested transaction)
    db.execute("BEGIN", ())
        .expect("Nested BEGIN should succeed as no-op");

    // Insert another row (still in the first transaction)
    db.execute("INSERT INTO txn_test (id, value) VALUES (2, 'second')", ())
        .unwrap();

    // Rollback should undo both inserts (since nested BEGIN was a no-op)
    db.execute("ROLLBACK", ()).expect("ROLLBACK should succeed");

    // Both rows should be gone
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(
        count, 0,
        "Both rows should be rolled back since nested BEGIN was a no-op"
    );
}

#[test]
fn test_commit_without_begin() {
    let db = Database::open("memory://txn_commit_no_begin").expect("Failed to create database");

    db.execute("CREATE TABLE txn_test (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // COMMIT without BEGIN should either succeed (auto-commit) or fail gracefully
    // This tests the behavior, whatever it is
    let result = db.execute("COMMIT", ());
    // Don't assert on success/failure - just ensure it doesn't panic
    let _ = result;
}

#[test]
fn test_transaction_insert_partial_columns() {
    // Regression test: transaction INSERT with partial column list (omitting AUTO_INCREMENT)
    // Previously failed because the transaction path validated param count against total
    // table columns instead of the columns specified in the INSERT statement.
    let db = Database::open("memory://txn_partial_cols").expect("Failed to create database");

    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            country TEXT NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    // Non-transaction insert with partial columns (omitting id) - should work
    db.execute(
        "INSERT INTO customers (name, email, country) VALUES ('Alice', 'alice@example.com', 'US')",
        (),
    )
    .expect("Non-transaction partial column insert should work");

    // Transaction insert with partial columns (omitting id) - this was the bug
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "INSERT INTO customers (name, email, country) VALUES ('Bob', 'bob@example.com', 'UK')",
        (),
    )
    .expect("Transaction partial column insert should work");
    db.execute(
        "INSERT INTO customers (name, email, country) VALUES ('Clara', 'clara@example.com', 'DE')",
        (),
    )
    .expect("Second transaction insert should work");
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify all 3 rows exist with auto-generated IDs
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM customers", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "All 3 rows should be inserted");

    // Verify auto-increment IDs are sequential
    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM customers", ())
        .expect("Failed to get max id");
    assert_eq!(max_id, 3, "Auto-increment IDs should be 1, 2, 3");
}

#[test]
fn test_transaction_insert_with_defaults() {
    // Verify transaction INSERT respects DEFAULT values
    let db = Database::open("memory://txn_defaults").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            category TEXT NOT NULL DEFAULT 'General'
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "INSERT INTO products (name) VALUES ('Widget')",
        (),
    )
    .expect("Transaction insert with defaults should work");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let category: String = db
        .query_one("SELECT category FROM products WHERE name = 'Widget'", ())
        .expect("Failed to query");
    assert_eq!(category, "General", "Default value should be applied");

    let active: bool = db
        .query_one("SELECT is_active FROM products WHERE name = 'Widget'", ())
        .expect("Failed to query");
    assert!(active, "Default boolean should be true");
}

#[test]
fn test_autocommit_mode() {
    let db = Database::open("memory://txn_autocommit").expect("Failed to create database");

    db.execute(
        "CREATE TABLE txn_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Without explicit BEGIN, each statement should auto-commit
    db.execute("INSERT INTO txn_test (id, value) VALUES (1, 'auto1')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO txn_test (id, value) VALUES (2, 'auto2')", ())
        .expect("Failed to insert");

    // Both rows should be visible immediately
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM txn_test", ())
        .expect("Failed to count");
    assert_eq!(count, 2);
}
