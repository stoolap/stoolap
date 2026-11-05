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

//! Hash Index SQL Tests
//!
//! Tests Hash INDEX functionality through SQL interface.
//! Hash indexes are auto-selected for TEXT and JSON columns.
//! They excel at equality lookups but do NOT support range queries.

use stoolap::Database;

/// Test hash index on TEXT column (auto-selected)
#[test]
fn test_hash_index_on_text_column() {
    let db = Database::open("memory://hash_idx_test").expect("Failed to create database");

    // Create table with TEXT column
    db.execute(
        "CREATE TABLE hash_test (
            id INTEGER PRIMARY KEY,
            email TEXT,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO hash_test VALUES (1, 'alice@example.com', 'Alice')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hash_test VALUES (2, 'bob@example.com', 'Bob')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hash_test VALUES (3, 'charlie@example.com', 'Charlie')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hash_test VALUES (4, 'alice@example.com', 'Alice2')",
        (),
    )
    .unwrap();

    // Create index on TEXT column (should auto-select Hash index)
    db.execute("CREATE INDEX idx_email ON hash_test (email)", ())
        .expect("Failed to create hash index on email");

    // Test equality lookup (hash index excels at this)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_test WHERE email = 'alice@example.com'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with alice@example.com");

    // Test another equality lookup
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_test WHERE email = 'bob@example.com'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row with bob@example.com");

    // Test with non-existent value
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_test WHERE email = 'nobody@example.com'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 0, "Expected 0 rows with non-existent email");
}

/// Test hash index with updates
#[test]
fn test_hash_index_updates() {
    let db = Database::open("memory://hash_idx_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE hash_update_test (
            id INTEGER PRIMARY KEY,
            username TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO hash_update_test VALUES (1, 'user_a')", ())
        .unwrap();
    db.execute("INSERT INTO hash_update_test VALUES (2, 'user_b')", ())
        .unwrap();
    db.execute("INSERT INTO hash_update_test VALUES (3, 'user_a')", ())
        .unwrap();

    // Create hash index
    db.execute(
        "CREATE INDEX idx_username ON hash_update_test (username)",
        (),
    )
    .expect("Failed to create index");

    // Verify initial counts
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_update_test WHERE username = 'user_a'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 user_a rows initially");

    // Update a row
    db.execute(
        "UPDATE hash_update_test SET username = 'user_c' WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    // Verify counts after update
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_update_test WHERE username = 'user_a'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 user_a row after update");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_update_test WHERE username = 'user_c'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 user_c row after update");
}

/// Test hash index with deletes
#[test]
fn test_hash_index_deletes() {
    let db = Database::open("memory://hash_idx_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE hash_delete_test (
            id INTEGER PRIMARY KEY,
            tag TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO hash_delete_test VALUES (1, 'important')", ())
        .unwrap();
    db.execute("INSERT INTO hash_delete_test VALUES (2, 'important')", ())
        .unwrap();
    db.execute("INSERT INTO hash_delete_test VALUES (3, 'normal')", ())
        .unwrap();

    // Create hash index
    db.execute("CREATE INDEX idx_tag ON hash_delete_test (tag)", ())
        .expect("Failed to create index");

    // Verify initial count
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_delete_test WHERE tag = 'important'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 important rows initially");

    // Delete a row
    db.execute("DELETE FROM hash_delete_test WHERE id = 1", ())
        .expect("Failed to delete");

    // Verify count after delete
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_delete_test WHERE tag = 'important'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 important row after delete");
}

/// Test hash index with NULL values
#[test]
fn test_hash_index_nulls() {
    let db = Database::open("memory://hash_idx_nulls").expect("Failed to create database");

    db.execute(
        "CREATE TABLE hash_null_test (
            id INTEGER PRIMARY KEY,
            category TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO hash_null_test VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO hash_null_test VALUES (2, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO hash_null_test VALUES (3, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO hash_null_test VALUES (4, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO hash_null_test VALUES (5, NULL)", ())
        .unwrap();

    // Create hash index
    db.execute("CREATE INDEX idx_category ON hash_null_test (category)", ())
        .expect("Failed to create index");

    // Test IS NULL with index
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_null_test WHERE category IS NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with NULL category");

    // Test IS NOT NULL with index
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_null_test WHERE category IS NOT NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with non-NULL category");

    // Test equality
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_null_test WHERE category = 'A'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with category = 'A'");
}

/// Test multiple hash indexes on same table
#[test]
fn test_multiple_hash_indexes() {
    let db = Database::open("memory://multi_hash").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_hash_idx (
            id INTEGER PRIMARY KEY,
            email TEXT,
            username TEXT,
            department TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO multi_hash_idx VALUES (1, 'alice@test.com', 'alice', 'engineering')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_hash_idx VALUES (2, 'bob@test.com', 'bob', 'sales')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_hash_idx VALUES (3, 'alice@test.com', 'alice2', 'engineering')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_hash_idx VALUES (4, 'charlie@test.com', 'charlie', 'sales')",
        (),
    )
    .unwrap();

    // Create multiple hash indexes
    db.execute("CREATE INDEX idx_email ON multi_hash_idx (email)", ())
        .expect("Failed to create email index");
    db.execute("CREATE INDEX idx_username ON multi_hash_idx (username)", ())
        .expect("Failed to create username index");
    db.execute("CREATE INDEX idx_dept ON multi_hash_idx (department)", ())
        .expect("Failed to create department index");

    // Test queries using different indexes
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_hash_idx WHERE email = 'alice@test.com'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with alice@test.com");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_hash_idx WHERE username = 'bob'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row with username bob");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_hash_idx WHERE department = 'sales'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows in sales");

    // Test combined conditions
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_hash_idx WHERE email = 'alice@test.com' AND department = 'engineering'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(
        count, 2,
        "Expected 2 rows for alice@test.com in engineering"
    );
}

/// Test hash index with IN clause
#[test]
fn test_hash_index_in_clause() {
    let db = Database::open("memory://hash_in_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE hash_in_test (
            id INTEGER PRIMARY KEY,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO hash_in_test VALUES (1, 'pending')", ())
        .unwrap();
    db.execute("INSERT INTO hash_in_test VALUES (2, 'active')", ())
        .unwrap();
    db.execute("INSERT INTO hash_in_test VALUES (3, 'completed')", ())
        .unwrap();
    db.execute("INSERT INTO hash_in_test VALUES (4, 'pending')", ())
        .unwrap();
    db.execute("INSERT INTO hash_in_test VALUES (5, 'cancelled')", ())
        .unwrap();

    // Create hash index
    db.execute("CREATE INDEX idx_status ON hash_in_test (status)", ())
        .expect("Failed to create index");

    // Test IN clause
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_in_test WHERE status IN ('pending', 'active')",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with pending or active status");

    // Test NOT IN clause
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM hash_in_test WHERE status NOT IN ('completed', 'cancelled')",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows not completed or cancelled");
}
