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

//! Isolation Level Tests
//!
//! Tests transaction isolation level settings

use stoolap::Database;

/// Test session-wide isolation level setting with SET ISOLATIONLEVEL
#[test]
fn test_isolation_level_session() {
    let db = Database::open("memory://iso_session").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE test_iso (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Test session-wide isolation level setting
    db.execute("SET ISOLATIONLEVEL = 'SNAPSHOT'", ())
        .expect("Failed to set session isolation level");

    // Start a transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Insert data
    db.execute(
        "INSERT INTO test_iso (id, value) VALUES (1, 'tx1_data')",
        (),
    )
    .expect("Failed to insert");

    // Commit the transaction
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify data is visible after commit
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_iso", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row after commit");

    // Change back to READ COMMITTED
    db.execute("SET ISOLATIONLEVEL = 'READ COMMITTED'", ())
        .expect("Failed to reset session isolation level");
}

/// Test transaction-specific isolation level setting
#[test]
fn test_isolation_level_transaction() {
    let db = Database::open("memory://iso_tx").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE test_iso_tx (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Set session to READ COMMITTED
    db.execute("SET ISOLATIONLEVEL = 'READ COMMITTED'", ())
        .expect("Failed to set session isolation level");

    // Start a transaction with specific isolation level
    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .expect("Failed to begin transaction with isolation level");

    // Insert data
    db.execute(
        "INSERT INTO test_iso_tx (id, value) VALUES (1, 'tx1_snapshot')",
        (),
    )
    .expect("Failed to insert");

    // Commit
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify data is visible
    let result = db
        .query("SELECT value FROM test_iso_tx WHERE id = 1", ())
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(value, "tx1_snapshot");
        found = true;
    }
    assert!(found, "Expected to find row with id=1");
}

/// Test that transaction-specific isolation levels are reset after transaction
#[test]
fn test_isolation_level_reset() {
    let db = Database::open("memory://iso_reset").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE test_iso_reset (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Set session isolation level to READ COMMITTED
    db.execute("SET ISOLATIONLEVEL = 'READ COMMITTED'", ())
        .expect("Failed to set session isolation level");

    // Start transaction with specific isolation level
    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .expect("Failed to begin transaction");

    // Commit the transaction
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Start new transaction - should use original session level (READ COMMITTED)
    db.execute("BEGIN", ())
        .expect("Failed to begin second transaction");
    db.execute(
        "INSERT INTO test_iso_reset (id, value) VALUES (1, 'test')",
        (),
    )
    .expect("Failed to insert");
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify data was committed
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_iso_reset", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row");
}

/// Test different isolation level settings
#[test]
fn test_isolation_level_variants() {
    let db = Database::open("memory://iso_variants").expect("Failed to create database");

    // Test setting SNAPSHOT isolation
    db.execute("SET ISOLATIONLEVEL = 'SNAPSHOT'", ())
        .expect("Failed to set SNAPSHOT isolation");

    // Test setting READ COMMITTED isolation
    db.execute("SET ISOLATIONLEVEL = 'READ COMMITTED'", ())
        .expect("Failed to set READ COMMITTED isolation");

    // Test setting SERIALIZABLE isolation (if supported)
    let result = db.execute("SET ISOLATIONLEVEL = 'SERIALIZABLE'", ());
    // May or may not be supported - just log the result
    if result.is_err() {
        println!("SERIALIZABLE isolation level not supported");
    }
}

/// Test isolation in concurrent-like scenario
#[test]
fn test_isolation_visibility() {
    let db = Database::open("memory://iso_visibility").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE visibility_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute(
        "INSERT INTO visibility_test (id, value) VALUES (1, 100)",
        (),
    )
    .expect("Failed to insert initial data");

    // Set SNAPSHOT isolation
    db.execute("SET ISOLATIONLEVEL = 'SNAPSHOT'", ())
        .expect("Failed to set isolation level");

    // Begin first transaction
    db.execute("BEGIN", ()).expect("Failed to begin tx1");

    // Read initial value
    let value1: i64 = db
        .query_one("SELECT value FROM visibility_test WHERE id = 1", ())
        .expect("Failed to read in tx1");
    assert_eq!(value1, 100, "Expected initial value 100");

    // Update in same transaction
    db.execute("UPDATE visibility_test SET value = 200 WHERE id = 1", ())
        .expect("Failed to update");

    // Read updated value in same transaction (should see own changes)
    let value2: i64 = db
        .query_one("SELECT value FROM visibility_test WHERE id = 1", ())
        .expect("Failed to read updated value");
    assert_eq!(
        value2, 200,
        "Expected updated value 200 in same transaction"
    );

    // Commit
    db.execute("COMMIT", ()).expect("Failed to commit");

    // Verify final value
    let final_value: i64 = db
        .query_one("SELECT value FROM visibility_test WHERE id = 1", ())
        .expect("Failed to read final value");
    assert_eq!(final_value, 200, "Expected final value 200");
}

/// Test rollback with isolation level
#[test]
fn test_isolation_rollback() {
    let db = Database::open("memory://iso_rollback").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE rollback_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute("INSERT INTO rollback_test (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert");

    // Set isolation level
    db.execute("SET ISOLATIONLEVEL = 'SNAPSHOT'", ())
        .expect("Failed to set isolation level");

    // Begin transaction
    db.execute("BEGIN", ()).expect("Failed to begin");

    // Update value
    db.execute("UPDATE rollback_test SET value = 999 WHERE id = 1", ())
        .expect("Failed to update");

    // Rollback instead of commit
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    // Verify value was not changed
    let value: i64 = db
        .query_one("SELECT value FROM rollback_test WHERE id = 1", ())
        .expect("Failed to read value");
    assert_eq!(value, 100, "Expected original value 100 after rollback");
}
