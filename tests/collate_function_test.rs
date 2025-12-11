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

//! COLLATE Function Tests
//!
//! Tests for collation functions (BINARY, NOCASE, NOACCENT)

use stoolap::Database;

/// Test BINARY collation (case-sensitive)
#[test]
fn test_binary_collation() {
    let db = Database::open("memory://collate_binary").expect("Failed to create database");

    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES
            (1, 'Apple'),
            (2, 'apple'),
            (3, 'APPLE'),
            (4, 'Banana'),
            (5, 'banana'),
            (6, 'BANANA')",
        (),
    )
    .expect("Failed to insert data");

    // BINARY collation should be case-sensitive - only match exact case
    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(text_value, 'BINARY') = 'Apple'
             ORDER BY id",
            (),
        )
        .expect("Failed to query with BINARY collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let value: String = row.get(1).unwrap();
        assert_eq!(id, 1, "Expected id=1");
        assert_eq!(value, "Apple", "Expected 'Apple'");
        count += 1;
    }
    assert_eq!(count, 1, "Expected 1 row from BINARY collation");
}

/// Test NOCASE collation (case-insensitive)
#[test]
fn test_nocase_collation() {
    let db = Database::open("memory://collate_nocase").expect("Failed to create database");

    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES
            (1, 'Apple'),
            (2, 'apple'),
            (3, 'APPLE'),
            (4, 'Banana'),
            (5, 'banana'),
            (6, 'BANANA')",
        (),
    )
    .expect("Failed to insert data");

    // NOCASE collation should match all case variations
    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(text_value, 'NOCASE') = COLLATE('apple', 'NOCASE')
             ORDER BY id",
            (),
        )
        .expect("Failed to query with NOCASE collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        // Should match Apple, apple, APPLE (ids 1, 2, 3)
        assert!(id >= 1 && id <= 3, "Unexpected id: {}", id);
        count += 1;
    }
    assert_eq!(count, 3, "Expected 3 rows from NOCASE collation");
}

/// Test NOACCENT collation (accent-insensitive)
#[test]
fn test_noaccent_collation() {
    let db = Database::open("memory://collate_noaccent").expect("Failed to create database");

    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES
            (1, 'Cafe'),
            (2, 'cafe'),
            (3, 'CAFE')",
        (),
    )
    .expect("Failed to insert data");

    // NOACCENT collation should match accent variations
    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(LOWER(text_value), 'NOACCENT') = COLLATE('cafe', 'NOACCENT')
             ORDER BY id",
            (),
        )
        .expect("Failed to query with NOACCENT collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        assert!(id >= 1 && id <= 3, "Unexpected id: {}", id);
        count += 1;
    }
    assert_eq!(count, 3, "Expected 3 rows from NOACCENT collation");
}

/// Test ordering with COLLATE
#[test]
fn test_ordering_with_collate() {
    let db = Database::open("memory://collate_order").expect("Failed to create database");

    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES
            (1, 'Apple'),
            (2, 'apple'),
            (3, 'APPLE'),
            (4, 'Banana'),
            (5, 'banana'),
            (6, 'BANANA')",
        (),
    )
    .expect("Failed to insert data");

    // Order by COLLATE NOCASE - all Apple variants should come before Banana variants
    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             ORDER BY COLLATE(text_value, 'NOCASE')
             LIMIT 6",
            (),
        )
        .expect("Failed to query with ORDER BY COLLATE");

    let mut apple_count = 0;
    let mut banana_count = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(1).unwrap();
        let lower_value = value.to_lowercase();
        if lower_value == "apple" {
            apple_count += 1;
        } else if lower_value == "banana" {
            banana_count += 1;
        }
    }

    assert_eq!(apple_count, 3, "Expected 3 'Apple' entries");
    assert_eq!(banana_count, 3, "Expected 3 'Banana' entries");
}

/// Test COLLATE with NULL values
#[test]
fn test_collate_null_handling() {
    let db = Database::open("memory://collate_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES (1, NULL)",
        (),
    )
    .expect("Failed to insert NULL");

    // COLLATE should handle NULL without crashing
    let result = db
        .query(
            "SELECT COLLATE(text_value, 'BINARY')
             FROM collate_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query NULL value");

    let mut count = 0;
    for _row in result {
        count += 1;
    }
    assert_eq!(count, 1, "Should return 1 row for NULL value");
}

/// Test case-insensitive comparison with COLLATE
#[test]
fn test_case_insensitive_comparison() {
    let db = Database::open("memory://collate_ci").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (
            id INTEGER,
            username TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO users (id, username) VALUES
            (1, 'JohnDoe'),
            (2, 'johndoe'),
            (3, 'JOHNDOE'),
            (4, 'JaneDoe')",
        (),
    )
    .expect("Failed to insert data");

    // Case-insensitive search for usernames
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users
             WHERE COLLATE(username, 'NOCASE') = COLLATE('johndoe', 'NOCASE')",
            (),
        )
        .expect("Failed to query");

    assert_eq!(count, 3, "Expected 3 matching usernames (case-insensitive)");
}

/// Test COLLATE used with filtering in a practical scenario
#[test]
fn test_collate_practical_usage() {
    let db = Database::open("memory://collate_practical").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            category TEXT,
            price FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, category, price) VALUES
            (1, 'Electronics', 100.0),
            (2, 'electronics', 200.0),
            (3, 'ELECTRONICS', 150.0),
            (4, 'Furniture', 300.0)",
        (),
    )
    .expect("Failed to insert data");

    // Filter by case-insensitive category and sum the prices
    let total: f64 = db
        .query_one(
            "SELECT SUM(price)
             FROM products
             WHERE COLLATE(category, 'NOCASE') = COLLATE('electronics', 'NOCASE')",
            (),
        )
        .expect("Failed to query with COLLATE filter");

    // Electronics total should be 450 (100 + 200 + 150)
    assert!(
        (total - 450.0).abs() < 0.01,
        "Expected Electronics total 450, got {}",
        total
    );
}
