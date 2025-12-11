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

//! Auto-Increment Tests
//!
//! Tests auto-increment functionality for primary key columns

use stoolap::Database;

/// Test auto-increment functionality for a table with an explicit primary key
#[test]
fn test_auto_increment_pk_table() {
    let db = Database::open("memory://auto_inc_pk").expect("Failed to create database");

    // Create a test table with an INTEGER PRIMARY KEY AUTO_INCREMENT
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            price FLOAT NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert records with explicit ID values
    db.execute(
        "INSERT INTO products (id, name, price) VALUES (1, 'Product A', 10.99)",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO products (id, name, price) VALUES (2, 'Product B', 20.50)",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO products (id, name, price) VALUES (3, 'Product C', 30.75)",
        (),
    )
    .expect("Failed to insert");

    // Test auto-increment behavior by inserting without providing ID
    // Since we've already used IDs 1-3, the auto-generated ID should be 4
    db.execute(
        "INSERT INTO products (name, price) VALUES ('Product D', 40.25)",
        (),
    )
    .expect("Failed to insert without ID");

    // Verify through a query
    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM products", ())
        .expect("Failed to query max ID");
    assert_eq!(max_id, 4, "Expected max ID to be 4");

    // Insert a record with an ID that exceeds the current auto-increment counter
    db.execute(
        "INSERT INTO products (id, name, price) VALUES (10, 'Product E', 50.0)",
        (),
    )
    .expect("Failed to insert with higher ID");

    // Insert another record without ID to check if auto-increment counter was updated
    db.execute(
        "INSERT INTO products (name, price) VALUES ('Product F', 60.50)",
        (),
    )
    .expect("Failed to insert second record without ID");

    // Verify that the auto-increment counter was updated after inserting ID 10
    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM products", ())
        .expect("Failed to query max ID");
    assert_eq!(max_id, 11, "Expected max ID to be 11");
}

/// Test auto-increment with non-sequential inserts
#[test]
fn test_auto_increment_non_sequential() {
    let db = Database::open("memory://auto_inc_nonseq").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert with a high ID first
    db.execute("INSERT INTO items (id, name) VALUES (100, 'First')", ())
        .unwrap();

    // Insert without ID - should get 101
    db.execute("INSERT INTO items (name) VALUES ('Second')", ())
        .unwrap();

    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM items", ())
        .expect("Failed to query");
    assert_eq!(max_id, 101);

    // Insert with a lower explicit ID
    db.execute("INSERT INTO items (id, name) VALUES (50, 'Third')", ())
        .unwrap();

    // Insert without ID - should still get 102 (not 51)
    db.execute("INSERT INTO items (name) VALUES ('Fourth')", ())
        .unwrap();

    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM items", ())
        .expect("Failed to query");
    assert_eq!(max_id, 102);

    // Verify total count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM items", ())
        .expect("Failed to count");
    assert_eq!(count, 4);
}

/// Test that records in a table without explicit primary key can still be inserted
#[test]
fn test_table_without_pk() {
    let db = Database::open("memory://no_pk").expect("Failed to create database");

    // Create a simple table without primary key (no auto-increment column)
    db.execute(
        "CREATE TABLE events (
            event_type TEXT NOT NULL,
            description TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert several records
    db.execute(
        "INSERT INTO events (event_type, description) VALUES ('START', 'System started')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO events (event_type, description) VALUES ('LOG', 'Operation logged')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO events (event_type, description) VALUES ('ALERT', 'Alert triggered')",
        (),
    )
    .expect("Failed to insert");

    // Query to verify the records were inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM events", ())
        .expect("Failed to count");
    assert_eq!(count, 3);

    // Insert another record
    db.execute(
        "INSERT INTO events (event_type, description) VALUES ('LOG', 'New operation logged')",
        (),
    )
    .expect("Failed to insert fourth record");

    // Verify record count is now 4
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM events", ())
        .expect("Failed to count");
    assert_eq!(count, 4);
}

/// Test multiple auto-increment inserts
#[test]
fn test_multiple_auto_increment_inserts() {
    let db = Database::open("memory://multi_auto_inc").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            username TEXT NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert multiple records without specifying ID
    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO users (username) VALUES ('user{}')", i),
            (),
        )
        .expect("Failed to insert");
    }

    // Verify all 10 records were inserted with sequential IDs
    let result = db
        .query("SELECT id, username FROM users ORDER BY id", ())
        .expect("Failed to query");

    let mut expected_id = 1;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let username: String = row.get(1).unwrap();

        assert_eq!(id, expected_id, "Expected sequential ID");
        assert_eq!(username, format!("user{}", expected_id));
        expected_id += 1;
    }

    assert_eq!(expected_id, 11, "Expected 10 records");
}

/// Test auto-increment with gaps
#[test]
fn test_auto_increment_with_gaps() {
    let db = Database::open("memory://auto_inc_gaps").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            product TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert records with gaps
    db.execute("INSERT INTO orders (id, product) VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, product) VALUES (5, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, product) VALUES (10, 'C')", ())
        .unwrap();

    // Insert without ID - should get 11
    db.execute("INSERT INTO orders (product) VALUES ('D')", ())
        .unwrap();

    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM orders", ())
        .expect("Failed to query");
    assert_eq!(max_id, 11);

    // Verify we can still insert in the gaps
    db.execute("INSERT INTO orders (id, product) VALUES (3, 'E')", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, product) VALUES (7, 'F')", ())
        .unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders", ())
        .expect("Failed to count");
    assert_eq!(count, 6);

    // Insert without ID again - should get 12
    db.execute("INSERT INTO orders (product) VALUES ('G')", ())
        .unwrap();

    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM orders", ())
        .expect("Failed to query");
    assert_eq!(max_id, 12);
}
