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

//! Simple DISTINCT Tests
//!
//! Tests basic DISTINCT functionality

use stoolap::Database;

/// Test basic DISTINCT with duplicate values
#[test]
fn test_simple_distinct() {
    let db = Database::open("memory://simple_distinct").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
        .expect("Failed to create table");

    // Insert duplicate values
    db.execute("INSERT INTO test (id, value) VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'B'), (5, 'A'), (6, 'C')", ())
        .expect("Failed to insert data");

    // Test DISTINCT
    let result = db
        .query("SELECT DISTINCT value FROM test ORDER BY value", ())
        .expect("Failed to execute query");

    let mut values: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: String = row.get(0).unwrap();
        values.push(val);
    }

    assert_eq!(
        values.len(),
        3,
        "Expected 3 distinct values, got {:?}",
        values
    );
    assert_eq!(values, vec!["A", "B", "C"]);
}

/// Test DISTINCT with integers
#[test]
fn test_distinct_integers() {
    let db = Database::open("memory://distinct_int").expect("Failed to create database");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, num INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO numbers (id, num) VALUES (1, 10), (2, 20), (3, 10), (4, 30), (5, 20), (6, 10)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT DISTINCT num FROM numbers ORDER BY num", ())
        .expect("Failed to execute query");

    let mut nums: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let num: i64 = row.get(0).unwrap();
        nums.push(num);
    }

    assert_eq!(nums.len(), 3, "Expected 3 distinct numbers");
    assert_eq!(nums, vec![10, 20, 30]);
}

/// Test DISTINCT with NULL values
#[test]
fn test_distinct_with_nulls() {
    let db = Database::open("memory://distinct_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE nullable (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO nullable (id, val) VALUES (1, 'A')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO nullable (id, val) VALUES (2, NULL)", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO nullable (id, val) VALUES (3, 'A')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO nullable (id, val) VALUES (4, NULL)", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO nullable (id, val) VALUES (5, 'B')", ())
        .expect("Failed to insert data");

    // DISTINCT should treat all NULLs as one value
    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT val) FROM nullable", ())
        .expect("Failed to execute query");

    // COUNT(DISTINCT) typically doesn't count NULL, so expect 2 (A, B)
    assert_eq!(count, 2, "Expected 2 distinct non-null values");
}

/// Test DISTINCT with multiple columns
#[test]
fn test_distinct_multiple_columns() {
    let db = Database::open("memory://distinct_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE pairs (id INTEGER PRIMARY KEY, col1 TEXT, col2 TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO pairs (id, col1, col2) VALUES (1, 'A', 'X'), (2, 'A', 'Y'), (3, 'A', 'X'), (4, 'B', 'X'), (5, 'B', 'X')", ())
        .expect("Failed to insert data");

    // DISTINCT on two columns
    let result = db
        .query(
            "SELECT DISTINCT col1, col2 FROM pairs ORDER BY col1, col2",
            (),
        )
        .expect("Failed to execute query");

    let mut pairs: Vec<(String, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let c1: String = row.get(0).unwrap();
        let c2: String = row.get(1).unwrap();
        pairs.push((c1, c2));
    }

    // Should have 3 distinct pairs: (A,X), (A,Y), (B,X)
    assert_eq!(pairs.len(), 3, "Expected 3 distinct pairs");
}

/// Test COUNT DISTINCT
#[test]
fn test_count_distinct() {
    let db = Database::open("memory://count_distinct").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items (id, category) VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C'), (5, 'B'), (6, 'A')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM items", ())
        .expect("Failed to execute query");

    assert_eq!(count, 3, "Expected 3 distinct categories");
}

/// Test DISTINCT with WHERE clause
#[test]
fn test_distinct_with_where() {
    let db = Database::open("memory://distinct_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products (id, category, price) VALUES (1, 'A', 100), (2, 'B', 50), (3, 'A', 150), (4, 'C', 75), (5, 'B', 200)", ())
        .expect("Failed to insert data");

    // DISTINCT with WHERE
    let result = db
        .query(
            "SELECT DISTINCT category FROM products WHERE price > 60 ORDER BY category",
            (),
        )
        .expect("Failed to execute query");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: String = row.get(0).unwrap();
        categories.push(cat);
    }

    // A (100, 150), C (75), B (200) have price > 60
    assert_eq!(
        categories.len(),
        3,
        "Expected 3 distinct categories with price > 60"
    );
}
