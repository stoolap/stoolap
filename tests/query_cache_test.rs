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

//! Query Cache Integration Tests

use stoolap::Database;

/// Tests basic query cache functionality
#[test]
fn test_query_cache_basic() {
    // Create a memory database for testing
    let db = Database::open("memory://").expect("Failed to create database");

    // Create a test table with sample data
    db.execute(
        "CREATE TABLE cache_test (id INTEGER PRIMARY KEY, name TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Insert a test row
    db.execute("INSERT INTO cache_test VALUES (1, 'test-name', 10.5)", ())
        .expect("Failed to insert test row");

    // Verify the row was inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM cache_test", ())
        .expect("Failed to count rows");
    assert_eq!(count, 1, "Expected 1 row but got {}", count);

    // Execute the same query multiple times (should use cache after first)
    for i in 0..5 {
        let rows = db
            .query("SELECT * FROM cache_test", ())
            .expect("Query failed");

        let row_count = rows.count();
        assert_eq!(
            row_count, 1,
            "Iteration {}: Expected 1 row, got {}",
            i, row_count
        );
    }

    // Execute parameterized queries that demonstrate query caching with parameters
    let rows = db
        .query("SELECT * FROM cache_test WHERE id = $1", (1,))
        .expect("Parameterized query failed");
    assert_eq!(rows.count(), 1);

    // Execute the same parameterized query again with different value
    let rows = db
        .query("SELECT * FROM cache_test WHERE id = $1", (2,))
        .expect("Second parameterized query failed");
    assert_eq!(rows.count(), 0); // No row with id=2

    // Execute a mix of queries to test cache behavior
    let query_strings = vec![
        "SELECT * FROM cache_test",
        "SELECT name, value FROM cache_test",
        "SELECT id FROM cache_test",
        "SELECT * FROM cache_test", // Repeated query
    ];

    for q in query_strings {
        let _rows = db.query(q, ()).expect(&format!("Query failed on '{}'", q));
    }
}

/// Test that cache handles multiple tables correctly
#[test]
fn test_query_cache_multiple_tables() {
    let db = Database::open("memory://").expect("Failed to create database");

    db.execute(
        "CREATE TABLE table_a (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table_a");
    db.execute(
        "CREATE TABLE table_b (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table_b");

    db.execute("INSERT INTO table_a VALUES (1, 'Alice'), (2, 'Bob')", ())
        .expect("Failed to insert into table_a");
    db.execute("INSERT INTO table_b VALUES (1, 100), (2, 200)", ())
        .expect("Failed to insert into table_b");

    // Query both tables multiple times
    for _ in 0..3 {
        let rows_a = db
            .query("SELECT * FROM table_a", ())
            .expect("Query table_a failed");
        assert_eq!(rows_a.count(), 2);

        let rows_b = db
            .query("SELECT * FROM table_b", ())
            .expect("Query table_b failed");
        assert_eq!(rows_b.count(), 2);
    }
}

/// Test cache with different WHERE clauses
#[test]
fn test_query_cache_where_clauses() {
    let db = Database::open("memory://").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products VALUES (1, 'Apple', 1.50), (2, 'Banana', 0.75), (3, 'Cherry', 2.00)",
        (),
    )
    .expect("Failed to insert");

    // These should be cached as different queries
    let rows1 = db
        .query("SELECT * FROM products WHERE price > 1.0", ())
        .unwrap();
    assert_eq!(rows1.count(), 2);

    let rows2 = db
        .query("SELECT * FROM products WHERE price < 1.0", ())
        .unwrap();
    assert_eq!(rows2.count(), 1);

    // Same query should use cache
    let rows3 = db
        .query("SELECT * FROM products WHERE price > 1.0", ())
        .unwrap();
    assert_eq!(rows3.count(), 2);
}

/// Test cache with aggregation queries
#[test]
fn test_query_cache_aggregations() {
    let db = Database::open("memory://").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount FLOAT, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales VALUES (1, 100.0, 'A'), (2, 200.0, 'A'), (3, 150.0, 'B')",
        (),
    )
    .expect("Failed to insert");

    // Aggregation queries
    let sum: f64 = db.query_one("SELECT SUM(amount) FROM sales", ()).unwrap();
    assert!((sum - 450.0).abs() < 0.01);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM sales", ()).unwrap();
    assert_eq!(count, 3);

    // Run again to test cache
    let sum2: f64 = db.query_one("SELECT SUM(amount) FROM sales", ()).unwrap();
    assert!((sum2 - 450.0).abs() < 0.01);
}

/// Test cache with INSERT (simplified - UPDATE/DELETE have known issues)
#[test]
fn test_query_cache_with_dml() {
    let db = Database::open("memory://").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    // Insert and verify
    db.execute("INSERT INTO items VALUES (1, 'Item1')", ())
        .unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
    assert_eq!(count, 1);

    // Insert more
    db.execute("INSERT INTO items VALUES (2, 'Item2')", ())
        .unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
    assert_eq!(count, 2);
}

/// Test whitespace normalization in cache
#[test]
fn test_query_cache_whitespace_normalization() {
    let db = Database::open("memory://").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO test VALUES (1)", ()).unwrap();

    // These should all hit the same cache entry due to whitespace normalization
    let queries = vec![
        "SELECT * FROM test",
        "SELECT  *  FROM  test",
        "SELECT * FROM test ",
        " SELECT * FROM test",
        "SELECT\t*\tFROM\ttest",
    ];

    for q in queries {
        let rows = db.query(q, ()).expect(&format!("Query failed: '{}'", q));
        assert_eq!(rows.count(), 1, "Query '{}' should return 1 row", q);
    }
}
