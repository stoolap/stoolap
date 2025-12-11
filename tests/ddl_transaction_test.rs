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

//! DDL Transaction Rollback Tests
//!
//! Tests for DDL statements (CREATE TABLE, DROP TABLE) being properly
//! rolled back within transactions (Bug #86)

use stoolap::Database;

#[test]
fn test_create_table_rollback() {
    let db = Database::open("memory://ddl_create_rollback").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create table within transaction
    db.execute("CREATE TABLE rollback_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO rollback_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Verify table exists within transaction
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM rollback_test", ())
        .expect("Table should exist within transaction");
    assert_eq!(count, 1, "Should have 1 row within transaction");

    // Rollback transaction
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify table no longer exists after rollback
    let result = db.query("SELECT * FROM rollback_test", ());
    assert!(result.is_err(), "Table should not exist after rollback");
}

#[test]
fn test_create_table_commit() {
    let db = Database::open("memory://ddl_create_commit").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create table within transaction
    db.execute("CREATE TABLE commit_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO commit_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Commit transaction
    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Verify table exists after commit
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM commit_test", ())
        .expect("Table should exist after commit");
    assert_eq!(count, 1, "Should have 1 row after commit");
}

#[test]
fn test_drop_table_rollback() {
    let db = Database::open("memory://ddl_drop_rollback").expect("Failed to create database");

    // First create a table outside transaction
    db.execute("CREATE TABLE persist_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert some data
    db.execute("INSERT INTO persist_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Verify table exists
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM persist_test", ())
        .expect("Table should exist");
    assert_eq!(count, 1, "Should have 1 row");

    // Begin transaction and drop table
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("DROP TABLE persist_test", ())
        .expect("Failed to drop table");

    // Verify table is gone within transaction
    let result = db.query("SELECT * FROM persist_test", ());
    assert!(
        result.is_err(),
        "Table should not exist after DROP within transaction"
    );

    // Rollback transaction
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify table is restored after rollback (schema only - data may be lost)
    let result = db.query("SELECT * FROM persist_test", ());
    assert!(
        result.is_ok(),
        "Table schema should be restored after rollback"
    );
}

#[test]
fn test_drop_table_commit() {
    let db = Database::open("memory://ddl_drop_commit").expect("Failed to create database");

    // First create a table
    db.execute("CREATE TABLE drop_commit_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Begin transaction and drop table
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("DROP TABLE drop_commit_test", ())
        .expect("Failed to drop table");

    // Commit transaction
    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Verify table is gone after commit
    let result = db.query("SELECT * FROM drop_commit_test", ());
    assert!(
        result.is_err(),
        "Table should not exist after committed DROP"
    );
}

#[test]
fn test_multiple_ddl_in_transaction() {
    let db = Database::open("memory://ddl_multiple").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create multiple tables
    db.execute("CREATE TABLE table1 (id INTEGER)", ())
        .expect("Failed to create table1");
    db.execute("CREATE TABLE table2 (id INTEGER)", ())
        .expect("Failed to create table2");
    db.execute("CREATE TABLE table3 (id INTEGER)", ())
        .expect("Failed to create table3");

    // Verify all tables exist
    db.execute("INSERT INTO table1 VALUES (1)", ())
        .expect("table1 should exist");
    db.execute("INSERT INTO table2 VALUES (2)", ())
        .expect("table2 should exist");
    db.execute("INSERT INTO table3 VALUES (3)", ())
        .expect("table3 should exist");

    // Rollback
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify all tables are gone
    assert!(
        db.query("SELECT * FROM table1", ()).is_err(),
        "table1 should not exist after rollback"
    );
    assert!(
        db.query("SELECT * FROM table2", ()).is_err(),
        "table2 should not exist after rollback"
    );
    assert!(
        db.query("SELECT * FROM table3", ()).is_err(),
        "table3 should not exist after rollback"
    );
}

#[test]
fn test_ddl_outside_transaction_auto_commits() {
    let db = Database::open("memory://ddl_auto_commit").expect("Failed to create database");

    // Create table outside explicit transaction (should auto-commit)
    db.execute("CREATE TABLE auto_commit_test (id INTEGER)", ())
        .expect("Failed to create table");

    // Table should exist
    let result = db.query("SELECT * FROM auto_commit_test", ());
    assert!(result.is_ok(), "Table should exist after auto-commit");
}

#[test]
fn test_mixed_ddl_and_dml_rollback() {
    let db = Database::open("memory://ddl_dml_mixed").expect("Failed to create database");

    // Create a table first
    db.execute("CREATE TABLE existing_table (id INTEGER, value TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO existing_table VALUES (1, 'original')", ())
        .expect("Failed to insert");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create new table
    db.execute("CREATE TABLE new_table (id INTEGER)", ())
        .expect("Failed to create new table");

    // Modify existing table
    db.execute(
        "UPDATE existing_table SET value = 'modified' WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    // Rollback
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // New table should not exist
    assert!(
        db.query("SELECT * FROM new_table", ()).is_err(),
        "New table should not exist after rollback"
    );

    // Existing table should have original value (DML rollback)
    let value: String = db
        .query_one("SELECT value FROM existing_table WHERE id = 1", ())
        .expect("Should be able to query existing table");
    assert_eq!(value, "original", "Value should be rolled back to original");
}
