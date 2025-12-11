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

//! Column Alias in SELECT Tests
//!
//! Tests SELECT with column aliases functionality

use stoolap::Database;

/// Test basic SELECT with column alias
#[test]
fn test_column_alias_in_select() {
    let db = Database::open("memory://alias_select").expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE items (id INTEGER, price INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO items (id, price, name) VALUES (1, 100, 'Item A')",
        (),
    )
    .expect("Failed to insert data");

    // Test simple SELECT without alias first
    let result = db
        .query("SELECT price FROM items", ())
        .expect("Failed to execute query without alias");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let price: i64 = row.get(0).unwrap();
        assert_eq!(price, 100);
        count += 1;
    }
    assert_eq!(count, 1, "Expected 1 row");

    // Test SELECT with column alias
    let result = db
        .query("SELECT price AS cost FROM items", ())
        .expect("Failed to execute query with alias");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        assert_eq!(cost, 100);
        count += 1;
    }
    assert_eq!(count, 1, "Expected 1 row");
}

/// Test multiple column aliases
#[test]
fn test_multiple_column_aliases() {
    let db = Database::open("memory://alias_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, price, quantity) VALUES (1, 50, 10)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT price AS unit_price, quantity AS qty, price * quantity AS total FROM products",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let unit_price: i64 = row.get(0).unwrap();
        let qty: i64 = row.get(1).unwrap();
        let total: i64 = row.get(2).unwrap();

        assert_eq!(unit_price, 50);
        assert_eq!(qty, 10);
        assert_eq!(total, 500);
    }
}

/// Test alias with expression
#[test]
fn test_alias_with_expression() {
    let db = Database::open("memory://alias_expr").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO numbers (id, value) VALUES (1, 25)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT value * 2 AS doubled, value + 10 AS incremented FROM numbers",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let doubled: i64 = row.get(0).unwrap();
        let incremented: i64 = row.get(1).unwrap();

        assert_eq!(doubled, 50);
        assert_eq!(incremented, 35);
    }
}

/// Test alias with function
#[test]
fn test_alias_with_function() {
    let db = Database::open("memory://alias_func").expect("Failed to create database");

    db.execute("CREATE TABLE names (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO names (id, name) VALUES (1, 'hello')", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT UPPER(name) AS upper_name, LENGTH(name) AS name_len FROM names",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let upper_name: String = row.get(0).unwrap();
        let name_len: i64 = row.get(1).unwrap();

        assert_eq!(upper_name, "HELLO");
        assert_eq!(name_len, 5);
    }
}

/// Test alias with aggregate function
#[test]
fn test_alias_with_aggregate() {
    let db = Database::open("memory://alias_agg").expect("Failed to create database");

    db.execute("CREATE TABLE sales (id INTEGER, amount INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, amount) VALUES (1, 100), (2, 200), (3, 300)",
        (),
    )
    .expect("Failed to insert data");

    let total: i64 = db
        .query_one("SELECT SUM(amount) AS total_sales FROM sales", ())
        .expect("Failed to get sum");

    assert_eq!(total, 600);
}

/// Test alias in GROUP BY result
#[test]
fn test_alias_in_group_by() {
    let db = Database::open("memory://alias_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (id INTEGER, category TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders (id, category, amount) VALUES (1, 'A', 100), (2, 'B', 200), (3, 'A', 150)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT category AS cat, SUM(amount) AS total FROM orders GROUP BY category ORDER BY category", ())
        .expect("Failed to execute query");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: String = row.get(0).unwrap();
        let total: i64 = row.get(1).unwrap();
        results.push((cat, total));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("A".to_string(), 250));
    assert_eq!(results[1], ("B".to_string(), 200));
}

/// Test alias with CAST
#[test]
fn test_alias_with_cast() {
    let db = Database::open("memory://alias_cast").expect("Failed to create database");

    db.execute("CREATE TABLE data (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO data (id, val) VALUES (1, '42')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) AS num FROM data", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let num: i64 = row.get(0).unwrap();
        assert_eq!(num, 42);
    }
}

/// Test alias with DISTINCT
#[test]
fn test_alias_with_distinct() {
    let db = Database::open("memory://alias_distinct").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO items (id, category) VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT DISTINCT category AS cat FROM items ORDER BY category",
            (),
        )
        .expect("Failed to execute query");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: String = row.get(0).unwrap();
        categories.push(cat);
    }

    assert_eq!(categories, vec!["A", "B", "C"]);
}

/// Test alias with ORDER BY (using alias)
#[test]
fn test_alias_in_order_by() {
    let db = Database::open("memory://alias_order").expect("Failed to create database");

    db.execute("CREATE TABLE scores (id INTEGER, score INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO scores (id, score) VALUES (1, 85), (2, 92), (3, 78)",
        (),
    )
    .expect("Failed to insert data");

    // Order by the alias name
    let result = db
        .query(
            "SELECT score AS points FROM scores ORDER BY points DESC",
            (),
        )
        .expect("Failed to execute query");

    let mut points: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let p: i64 = row.get(0).unwrap();
        points.push(p);
    }

    assert_eq!(points, vec![92, 85, 78]);
}

/// Test alias with NULL values
#[test]
fn test_alias_with_null() {
    let db = Database::open("memory://alias_null").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, NULL)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT value AS val FROM test", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let val: Option<i64> = row.get(0).ok();
        assert!(val.is_none());
    }
}

/// Test alias with subquery
#[test]
fn test_alias_with_subquery() {
    let db = Database::open("memory://alias_subquery").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO t1 (id, value) VALUES (1, 10), (2, 20), (3, 30)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT (SELECT MAX(value) FROM t1) AS max_val FROM t1 LIMIT 1",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let max_val: i64 = row.get(0).unwrap();
        assert_eq!(max_val, 30);
    }
}
