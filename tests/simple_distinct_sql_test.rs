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
//! Tests SELECT DISTINCT functionality

use std::collections::HashSet;
use stoolap::Database;

/// Test basic SELECT DISTINCT
#[test]
fn test_simple_distinct() {
    let db = Database::open("memory://distinct_basic").expect("Failed to create database");

    // Create test table
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
        .expect("Failed to create table");

    // Insert duplicate values
    db.execute(
        "INSERT INTO test (id, value) VALUES
         (1, 'A'), (2, 'B'), (3, 'A'), (4, 'B'), (5, 'A'), (6, 'C')",
        (),
    )
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
        "Expected 3 distinct values, got {}",
        values.len()
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
        "INSERT INTO numbers (id, num) VALUES
         (1, 10), (2, 20), (3, 10), (4, 30), (5, 20), (6, 10)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT DISTINCT num FROM numbers ORDER BY num", ())
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        values.push(val);
    }

    assert_eq!(values.len(), 3, "Expected 3 distinct values");
    assert_eq!(values, vec![10, 20, 30]);
}

/// Test DISTINCT with NULL values
#[test]
fn test_distinct_with_nulls() {
    let db = Database::open("memory://distinct_nulls").expect("Failed to create database");

    db.execute(
        "CREATE TABLE nulls_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO nulls_test (id, value) VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO nulls_test (id, value) VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO nulls_test (id, value) VALUES (3, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO nulls_test (id, value) VALUES (4, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO nulls_test (id, value) VALUES (5, 'B')", ())
        .unwrap();

    // Query DISTINCT values directly and count
    let result = db
        .query("SELECT DISTINCT value FROM nulls_test", ())
        .expect("Failed to query distinct values");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }

    // Should have 3 distinct values: 'A', 'B', NULL
    assert_eq!(count, 3, "Expected 3 distinct values including NULL");
}

/// Test DISTINCT with multiple columns
#[test]
fn test_distinct_multiple_columns() {
    let db = Database::open("memory://distinct_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi (id INTEGER PRIMARY KEY, col1 TEXT, col2 INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO multi (id, col1, col2) VALUES
         (1, 'A', 1), (2, 'A', 2), (3, 'A', 1), (4, 'B', 1), (5, 'B', 2)",
        (),
    )
    .expect("Failed to insert data");

    // DISTINCT on multiple columns
    let result = db
        .query(
            "SELECT DISTINCT col1, col2 FROM multi ORDER BY col1, col2",
            (),
        )
        .expect("Failed to execute query");

    let mut rows_count = 0;
    let mut combinations: HashSet<(String, i64)> = HashSet::new();

    for row in result {
        let row = row.expect("Failed to get row");
        let col1: String = row.get(0).unwrap();
        let col2: i64 = row.get(1).unwrap();
        combinations.insert((col1, col2));
        rows_count += 1;
    }

    // Should have 4 distinct combinations: (A,1), (A,2), (B,1), (B,2)
    assert_eq!(rows_count, 4, "Expected 4 distinct combinations");
    assert!(combinations.contains(&("A".to_string(), 1)));
    assert!(combinations.contains(&("A".to_string(), 2)));
    assert!(combinations.contains(&("B".to_string(), 1)));
    assert!(combinations.contains(&("B".to_string(), 2)));
}

/// Test COUNT(DISTINCT column)
#[test]
fn test_count_distinct() {
    let db = Database::open("memory://count_distinct").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO items (id, category) VALUES
         (1, 'Electronics'), (2, 'Books'), (3, 'Electronics'),
         (4, 'Clothing'), (5, 'Books'), (6, 'Electronics')",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM items", ())
        .expect("Failed to count distinct categories");

    assert_eq!(count, 3, "Expected 3 distinct categories");
}

/// Test DISTINCT with WHERE clause
#[test]
fn test_distinct_with_where() {
    let db = Database::open("memory://distinct_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, category, active) VALUES
         (1, 'A', true), (2, 'B', true), (3, 'A', false),
         (4, 'C', true), (5, 'B', false), (6, 'A', true)",
        (),
    )
    .expect("Failed to insert data");

    // DISTINCT categories where active = true
    let result = db
        .query(
            "SELECT DISTINCT category FROM products WHERE active = true ORDER BY category",
            (),
        )
        .expect("Failed to execute query");

    let mut values: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: String = row.get(0).unwrap();
        values.push(val);
    }

    // Active categories: A, B, C
    assert_eq!(values.len(), 3, "Expected 3 distinct active categories");
    assert_eq!(values, vec!["A", "B", "C"]);
}

/// Test DISTINCT with ORDER BY
#[test]
fn test_distinct_order_by() {
    let db = Database::open("memory://distinct_order").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO scores (id, score) VALUES
         (1, 85), (2, 92), (3, 85), (4, 78), (5, 92), (6, 100)",
        (),
    )
    .expect("Failed to insert data");

    // DISTINCT with descending order
    let result = db
        .query("SELECT DISTINCT score FROM scores ORDER BY score DESC", ())
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        values.push(val);
    }

    assert_eq!(values.len(), 4, "Expected 4 distinct scores");
    assert_eq!(values, vec![100, 92, 85, 78]);
}

/// Test DISTINCT with GROUP BY
#[test]
fn test_distinct_with_group_by() {
    let db = Database::open("memory://distinct_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, region, product, amount) VALUES
         (1, 'East', 'A', 100), (2, 'East', 'B', 200), (3, 'West', 'A', 150),
         (4, 'East', 'A', 120), (5, 'West', 'B', 180), (6, 'West', 'A', 90)",
        (),
    )
    .expect("Failed to insert data");

    // Count distinct products per region
    let result = db
        .query("SELECT region, COUNT(DISTINCT product) AS unique_products FROM sales GROUP BY region ORDER BY region", ())
        .expect("Failed to execute query");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let region: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        results.push((region, count));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("East".to_string(), 2)); // East has products A and B
    assert_eq!(results[1], ("West".to_string(), 2)); // West has products A and B
}

/// Test DISTINCT on boolean column
#[test]
fn test_distinct_boolean() {
    let db = Database::open("memory://distinct_bool").expect("Failed to create database");

    db.execute(
        "CREATE TABLE flags (id INTEGER PRIMARY KEY, flag BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO flags (id, flag) VALUES
         (1, true), (2, false), (3, true), (4, true), (5, false)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT flag) FROM flags", ())
        .expect("Failed to count distinct flags");

    assert_eq!(count, 2, "Expected 2 distinct boolean values");
}

/// Test DISTINCT with float values
#[test]
fn test_distinct_float() {
    let db = Database::open("memory://distinct_float").expect("Failed to create database");

    db.execute(
        "CREATE TABLE measurements (id INTEGER PRIMARY KEY, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO measurements (id, value) VALUES
         (1, 1.5), (2, 2.5), (3, 1.5), (4, 3.0), (5, 2.5)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT DISTINCT value FROM measurements ORDER BY value", ())
        .expect("Failed to execute query");

    let mut values: Vec<f64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: f64 = row.get(0).unwrap();
        values.push(val);
    }

    assert_eq!(values.len(), 3, "Expected 3 distinct float values");
}
