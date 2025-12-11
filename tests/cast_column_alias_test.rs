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

//! CAST Column Alias Tests
//!
//! Tests CAST with column aliases

use stoolap::Database;

/// Test simple column aliases
#[test]
fn test_simple_column_alias() {
    let db = Database::open("memory://cast_alias_simple").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_alias (
            id INTEGER,
            val TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test_alias (id, val) VALUES (1, '123')", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT id AS alias_id, val AS alias_val FROM test_alias",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let val: String = row.get(1).unwrap();

        assert_eq!(id, 1);
        assert_eq!(val, "123");
        count += 1;
    }

    assert_eq!(count, 1, "Expected exactly one row");
}

/// Test expression with alias
#[test]
fn test_expression_with_alias() {
    let db = Database::open("memory://cast_alias_expr").expect("Failed to create database");

    db.execute("CREATE TABLE test_alias (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_alias (id, val) VALUES (1, '123')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT (id + 10) AS calculated FROM test_alias", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let calculated: i64 = row.get(0).unwrap();

        // id + 10 = 1 + 10 = 11
        assert_eq!(calculated, 11, "Expected calculated=11, got {}", calculated);
    }
}

/// Test CAST with alias
#[test]
fn test_cast_with_alias() {
    let db = Database::open("memory://cast_alias_cast").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (id, val) VALUES (1, '456')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) AS num_val FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let num_val: i64 = row.get(0).unwrap();

        assert_eq!(num_val, 456, "Expected num_val=456");
    }
}

/// Test multiple aliases
#[test]
fn test_multiple_aliases() {
    let db = Database::open("memory://cast_alias_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER, name TEXT, price INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 100)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT id AS product_id,
                    name AS product_name,
                    price AS unit_price
             FROM products",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let product_id: i64 = row.get(0).unwrap();
        let product_name: String = row.get(1).unwrap();
        let unit_price: i64 = row.get(2).unwrap();

        assert_eq!(product_id, 1);
        assert_eq!(product_name, "Widget");
        assert_eq!(unit_price, 100);
    }
}

/// Test alias with arithmetic expression
#[test]
fn test_alias_with_arithmetic() {
    let db = Database::open("memory://cast_alias_arith").expect("Failed to create database");

    db.execute("CREATE TABLE items (price INTEGER, quantity INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items (price, quantity) VALUES (50, 3)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT price * quantity AS total FROM items", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(0).unwrap();

        assert_eq!(total, 150, "Expected total=150 (50*3)");
    }
}

/// Test alias with function
#[test]
fn test_alias_with_function() {
    let db = Database::open("memory://cast_alias_func").expect("Failed to create database");

    db.execute("CREATE TABLE names (name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO names (name) VALUES ('hello')", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT UPPER(name) AS upper_name, LENGTH(name) AS name_length FROM names",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let upper_name: String = row.get(0).unwrap();
        let name_length: i64 = row.get(1).unwrap();

        assert_eq!(upper_name, "HELLO");
        assert_eq!(name_length, 5);
    }
}

/// Test alias with COALESCE
#[test]
fn test_alias_with_coalesce() {
    let db = Database::open("memory://cast_alias_coalesce").expect("Failed to create database");

    db.execute("CREATE TABLE data (value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO data (value) VALUES (NULL), (42)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT COALESCE(value, 0) AS safe_value FROM data ORDER BY safe_value",
            (),
        )
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let safe_value: i64 = row.get(0).unwrap();
        values.push(safe_value);
    }

    assert_eq!(values.len(), 2);
    assert_eq!(values[0], 0, "NULL should become 0");
    assert_eq!(values[1], 42, "42 should remain 42");
}

/// Test alias in aggregate
#[test]
fn test_alias_in_aggregate() {
    let db = Database::open("memory://cast_alias_agg").expect("Failed to create database");

    db.execute("CREATE TABLE amounts (amount INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO amounts (amount) VALUES (10), (20), (30)", ())
        .expect("Failed to insert data");

    let sum: i64 = db
        .query_one("SELECT SUM(amount) AS total_amount FROM amounts", ())
        .expect("Failed to execute query");

    assert_eq!(sum, 60, "Expected total_amount=60");

    let avg: f64 = db
        .query_one("SELECT AVG(amount) AS avg_amount FROM amounts", ())
        .expect("Failed to execute query");

    assert!((avg - 20.0).abs() < 0.01, "Expected avg_amount=20.0");
}
