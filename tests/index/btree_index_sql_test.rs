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

//! B-tree Index SQL Tests
//!
//! Tests B-tree INDEX functionality through SQL interface.
//! B-tree indexes are auto-selected for INTEGER, FLOAT, and TIMESTAMP columns.

use stoolap::Database;

/// Test creating and using B-tree index
#[test]
fn test_btree_index_sql() {
    let db = Database::open("memory://btree_idx").expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE test_btree (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute("INSERT INTO test_btree VALUES (1, 'Alice', 30, true)", ())
        .expect("Failed to insert row 1");
    db.execute("INSERT INTO test_btree VALUES (2, 'Bob', 25, false)", ())
        .expect("Failed to insert row 2");
    db.execute("INSERT INTO test_btree VALUES (3, 'Charlie', 40, true)", ())
        .expect("Failed to insert row 3");

    // Create a B-tree index on the 'age' column (INTEGER -> BTree)
    db.execute("CREATE INDEX idx_age ON test_btree (age)", ())
        .expect("Failed to create index on age");

    // Test querying with the index on age
    let result = db
        .query("SELECT * FROM test_btree WHERE age = 25", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let age: i64 = row.get(2).unwrap();
        let active: bool = row.get(3).unwrap();

        count += 1;
        assert_eq!(id, 2, "Expected id 2");
        assert_eq!(name, "Bob", "Expected name Bob");
        assert_eq!(age, 25, "Expected age 25");
        assert!(!active, "Expected active false");
    }

    assert_eq!(count, 1, "Expected 1 row for age = 25");

    // Drop the index
    db.execute("DROP INDEX idx_age ON test_btree", ())
        .expect("Failed to drop index");

    // The query should still work even after dropping the index
    let result = db
        .query("SELECT * FROM test_btree WHERE age = 25", ())
        .expect("Failed to execute query after dropping index");

    let mut post_drop_count = 0;
    for _ in result {
        post_drop_count += 1;
    }

    assert_eq!(
        post_drop_count, 1,
        "Expected 1 row after dropping index, got {}",
        post_drop_count
    );
}

/// Test B-tree index with range queries
#[test]
fn test_btree_index_range() {
    let db = Database::open("memory://btree_range").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE range_test (
            id INTEGER PRIMARY KEY,
            value INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert data
    for i in 1..=10 {
        db.execute(
            "INSERT INTO range_test (id, value) VALUES (?, ?)",
            (i, i * 10),
        )
        .expect("Failed to insert");
    }

    // Create B-tree index
    db.execute("CREATE INDEX idx_value ON range_test (value)", ())
        .expect("Failed to create index");

    // Test range query with >
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM range_test WHERE value > 50", ())
        .expect("Failed to count");
    assert_eq!(count, 5, "Expected 5 rows with value > 50");

    // Test range query with <
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM range_test WHERE value < 50", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 rows with value < 50");

    // Test range query with BETWEEN
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM range_test WHERE value BETWEEN 30 AND 70",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 5, "Expected 5 rows with value BETWEEN 30 AND 70");

    // Test >= and <=
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM range_test WHERE value >= 50", ())
        .expect("Failed to count");
    assert_eq!(count, 6, "Expected 6 rows with value >= 50");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM range_test WHERE value <= 50", ())
        .expect("Failed to count");
    assert_eq!(count, 5, "Expected 5 rows with value <= 50");
}

/// Test B-tree index with NULL values
#[test]
fn test_btree_index_nulls() {
    let db = Database::open("memory://btree_nulls").expect("Failed to create database");

    // Create table with INTEGER column for BTree
    db.execute(
        "CREATE TABLE null_test (
            id INTEGER PRIMARY KEY,
            score INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert data including NULLs
    db.execute("INSERT INTO null_test VALUES (1, 100)", ())
        .unwrap();
    db.execute("INSERT INTO null_test VALUES (2, 200)", ())
        .unwrap();
    db.execute("INSERT INTO null_test VALUES (3, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO null_test VALUES (4, 100)", ())
        .unwrap();
    db.execute("INSERT INTO null_test VALUES (5, NULL)", ())
        .unwrap();

    // Create B-tree index
    db.execute("CREATE INDEX idx_score ON null_test (score)", ())
        .expect("Failed to create index");

    // Test IS NULL with index
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE score IS NULL", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with NULL score");

    // Test IS NOT NULL with index
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE score IS NOT NULL", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with non-NULL score");

    // Test equality with index
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE score = 100", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with score = 100");
}

/// Test B-tree index with updates
#[test]
fn test_btree_index_updates() {
    let db = Database::open("memory://btree_updates").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE update_test (
            id INTEGER PRIMARY KEY,
            priority INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute("INSERT INTO update_test VALUES (1, 1)", ())
        .unwrap();
    db.execute("INSERT INTO update_test VALUES (2, 1)", ())
        .unwrap();
    db.execute("INSERT INTO update_test VALUES (3, 2)", ())
        .unwrap();

    // Create B-tree index
    db.execute("CREATE INDEX idx_priority ON update_test (priority)", ())
        .expect("Failed to create index");

    // Verify initial counts
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM update_test WHERE priority = 1", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 priority=1 rows initially");

    // Update a row
    db.execute("UPDATE update_test SET priority = 3 WHERE id = 1", ())
        .expect("Failed to update");

    // Verify counts after update
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM update_test WHERE priority = 1", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 priority=1 row after update");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM update_test WHERE priority = 3", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 priority=3 row after update");
}

/// Test B-tree index with deletes
#[test]
fn test_btree_index_deletes() {
    let db = Database::open("memory://btree_deletes").expect("Failed to create database");

    // Create table
    db.execute(
        "CREATE TABLE delete_test (
            id INTEGER PRIMARY KEY,
            category INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO delete_test VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO delete_test VALUES (2, 10)", ())
        .unwrap();
    db.execute("INSERT INTO delete_test VALUES (3, 20)", ())
        .unwrap();
    db.execute("INSERT INTO delete_test VALUES (4, 20)", ())
        .unwrap();

    // Create B-tree index
    db.execute("CREATE INDEX idx_category ON delete_test (category)", ())
        .expect("Failed to create index");

    // Verify initial count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM delete_test WHERE category = 10", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 category=10 rows initially");

    // Delete a row
    db.execute("DELETE FROM delete_test WHERE id = 1", ())
        .expect("Failed to delete");

    // Verify count after delete
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM delete_test WHERE category = 10", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 category=10 row after delete");

    // Delete by category
    db.execute("DELETE FROM delete_test WHERE category = 20", ())
        .expect("Failed to delete");

    let total_count: i64 = db
        .query_one("SELECT COUNT(*) FROM delete_test", ())
        .expect("Failed to count");
    assert_eq!(total_count, 1, "Expected 1 row remaining");
}

/// Test multiple B-tree indexes on same table
#[test]
fn test_multiple_btree_indexes() {
    let db = Database::open("memory://multi_btree").expect("Failed to create database");

    // Create table with multiple INTEGER columns
    db.execute(
        "CREATE TABLE multi_idx (
            id INTEGER PRIMARY KEY,
            score INTEGER,
            level INTEGER,
            priority INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO multi_idx VALUES (1, 100, 1, 1)", ())
        .unwrap();
    db.execute("INSERT INTO multi_idx VALUES (2, 200, 2, 2)", ())
        .unwrap();
    db.execute("INSERT INTO multi_idx VALUES (3, 100, 1, 3)", ())
        .unwrap();
    db.execute("INSERT INTO multi_idx VALUES (4, 300, 2, 1)", ())
        .unwrap();

    // Create multiple B-tree indexes
    db.execute("CREATE INDEX idx_score ON multi_idx (score)", ())
        .expect("Failed to create score index");
    db.execute("CREATE INDEX idx_level ON multi_idx (level)", ())
        .expect("Failed to create level index");
    db.execute("CREATE INDEX idx_priority ON multi_idx (priority)", ())
        .expect("Failed to create priority index");

    // Test queries using different indexes
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM multi_idx WHERE score = 100", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 score=100 rows");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM multi_idx WHERE level = 1", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 level=1 rows");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM multi_idx WHERE priority = 1", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 priority=1 rows");

    // Test combined conditions
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_idx WHERE score = 100 AND level = 1",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows for score=100 AND level=1");

    // Test range with equality
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_idx WHERE score > 100 AND priority = 1",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row for score > 100 AND priority=1");
}
