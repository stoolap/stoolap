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

//! Named Parameters Tests
//!
//! Tests for named parameter binding with :name syntax.

use stoolap::{named_params, Database, NamedParams};

/// Test basic named parameter insert
#[test]
fn test_named_params_insert() {
    let db = Database::open("memory://named_insert").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert with named params using macro
    db.execute_named(
        "INSERT INTO users VALUES (:id, :name, :age)",
        named_params! { id: 1, name: "Alice", age: 30 },
    )
    .expect("Failed to insert with named params");

    // Verify
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 1);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .expect("Failed to get name");
    assert_eq!(name, "Alice");
}

/// Test named parameter query
#[test]
fn test_named_params_query() {
    let db = Database::open("memory://named_query").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER, name TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'Apple', 1.50)", ())
        .unwrap();
    db.execute("INSERT INTO products VALUES (2, 'Banana', 0.75)", ())
        .unwrap();
    db.execute("INSERT INTO products VALUES (3, 'Cherry', 2.00)", ())
        .unwrap();

    // Query with named params
    let rows = db
        .query_named(
            "SELECT * FROM products WHERE price > :min_price",
            named_params! { min_price: 1.0 },
        )
        .expect("Failed to query");

    let count = rows.count();
    assert_eq!(count, 2, "Expected 2 products with price > 1.0");
}

/// Test named parameter update
#[test]
fn test_named_params_update() {
    let db = Database::open("memory://named_update").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Original')", ())
        .expect("Failed to insert");

    // Update with named params
    db.execute_named(
        "UPDATE users SET name = :new_name WHERE id = :id",
        named_params! { new_name: "Updated", id: 1 },
    )
    .expect("Failed to update with named params");

    // Verify
    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .expect("Failed to get name");
    assert_eq!(name, "Updated");
}

/// Test named parameter delete
#[test]
fn test_named_params_delete() {
    let db = Database::open("memory://named_delete").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (2)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (3)", ()).unwrap();

    // Delete with named params
    db.execute_named(
        "DELETE FROM items WHERE id = :target_id",
        named_params! { target_id: 2 },
    )
    .expect("Failed to delete with named params");

    // Verify
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM items", ())
        .expect("Failed to count");
    assert_eq!(count, 2);
}

/// Test query_one_named
#[test]
fn test_query_one_named() {
    let db = Database::open("memory://query_one_named").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Alice', 30)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob', 25)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie', 35)", ())
        .unwrap();

    // Count users older than a threshold
    let count: i64 = db
        .query_one_named(
            "SELECT COUNT(*) FROM users WHERE age > :min_age",
            named_params! { min_age: 26 },
        )
        .expect("Failed to count with named params");
    assert_eq!(count, 2, "Expected 2 users older than 26");
}

/// Test NamedParams builder pattern
#[test]
fn test_named_params_builder() {
    let db = Database::open("memory://builder").expect("Failed to create database");

    db.execute("CREATE TABLE test (a INTEGER, b TEXT, c FLOAT)", ())
        .expect("Failed to create table");

    // Build params manually
    let params = NamedParams::new()
        .add("val_a", 42)
        .add("val_b", "hello")
        .add("val_c", 3.5);

    db.execute_named("INSERT INTO test VALUES (:val_a, :val_b, :val_c)", params)
        .expect("Failed to insert");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

/// Test named params with multiple occurrences of same param
#[test]
fn test_named_params_multiple_uses() {
    let db = Database::open("memory://multi_use").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (low INTEGER, high INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO numbers VALUES (10, 100)", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (30, 50)", ())
        .unwrap(); // 25 NOT between 30 and 50
    db.execute("INSERT INTO numbers VALUES (5, 30)", ())
        .unwrap();

    // Use same param twice - find rows where val is between low and high
    let rows = db
        .query_named(
            "SELECT * FROM numbers WHERE low <= :val AND high >= :val",
            named_params! { val: 25 },
        )
        .expect("Failed to query");

    let count = rows.count();
    assert_eq!(count, 2, "Expected 2 rows where 25 is between low and high");
}

/// Test mixed data types with named params
#[test]
fn test_named_params_mixed_types() {
    let db = Database::open("memory://mixed_types").expect("Failed to create database");

    db.execute(
        "CREATE TABLE mixed (id INTEGER, name TEXT, price FLOAT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute_named(
        "INSERT INTO mixed VALUES (:id, :name, :price, :active)",
        named_params! {
            id: 1,
            name: "Product",
            price: 99.99,
            active: true
        },
    )
    .expect("Failed to insert");

    let rows = db
        .query_named(
            "SELECT * FROM mixed WHERE active = :is_active AND price < :max_price",
            named_params! { is_active: true, max_price: 100.0 },
        )
        .expect("Failed to query");

    let count = rows.count();
    assert_eq!(count, 1);
}

/// Test empty named params
#[test]
fn test_empty_named_params() {
    let db = Database::open("memory://empty_params").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test VALUES (1)", ()).unwrap();

    // Query with empty named params (no parameters in query)
    let rows = db
        .query_named("SELECT * FROM test", named_params! {})
        .expect("Failed to query");

    let count = rows.count();
    assert_eq!(count, 1);
}
