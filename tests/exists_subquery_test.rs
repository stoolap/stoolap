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

//! EXISTS Subquery Tests
//!
//! Tests EXISTS and NOT EXISTS subqueries

use stoolap::Database;

fn setup_exists_tables(db: &Database) {
    // Create customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER,
            name TEXT,
            country TEXT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Create orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER,
            customer_id INTEGER,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Insert customers
    db.execute(
        "INSERT INTO customers (id, name, country) VALUES
        (1, 'Alice', 'USA'),
        (2, 'Bob', 'UK'),
        (3, 'Charlie', 'USA'),
        (4, 'David', 'Canada')",
        (),
    )
    .expect("Failed to insert customers");

    // Insert orders
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES
        (1, 1, 100.0),
        (2, 1, 200.0),
        (3, 3, 150.0),
        (4, 4, 300.0)",
        (),
    )
    .expect("Failed to insert orders");
}

/// Test EXISTS with subquery that returns rows
#[test]
fn test_exists_with_results() {
    let db = Database::open("memory://exists_results").expect("Failed to create database");
    setup_exists_tables(&db);

    // EXISTS should return all customers when orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT * FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 customers when orders exist");
}

/// Test EXISTS with empty subquery
#[test]
fn test_exists_with_no_results() {
    let db = Database::open("memory://exists_empty").expect("Failed to create database");
    setup_exists_tables(&db);

    // Delete all orders
    db.execute("DELETE FROM orders", ())
        .expect("Failed to delete orders");

    // EXISTS should return no customers when no orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT 1 FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 customers when no orders exist");
}

/// Test NOT EXISTS with empty subquery
#[test]
fn test_not_exists_with_no_results() {
    let db = Database::open("memory://not_exists_empty").expect("Failed to create database");
    setup_exists_tables(&db);

    // Delete all orders
    db.execute("DELETE FROM orders", ())
        .expect("Failed to delete orders");

    // NOT EXISTS should return all customers when no orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE NOT EXISTS (SELECT * FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 customers when no orders exist");
}

/// Test EXISTS with condition in subquery
#[test]
fn test_exists_with_condition() {
    let db = Database::open("memory://exists_condition").expect("Failed to create database");
    setup_exists_tables(&db);

    // EXISTS with WHERE condition in subquery
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT * FROM orders WHERE amount > 150)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    // Since there is at least one order > 150, all customers should be returned
    assert_eq!(count, 4, "Expected 4 customers (order > 150 exists)");
}

/// Test NOT EXISTS with condition in subquery
#[test]
fn test_not_exists_with_condition() {
    let db = Database::open("memory://not_exists_condition").expect("Failed to create database");
    setup_exists_tables(&db);

    // NOT EXISTS with condition that never matches
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE NOT EXISTS (SELECT * FROM orders WHERE amount > 500)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    // Since no order > 500, all customers should be returned
    assert_eq!(count, 4, "Expected 4 customers (no order > 500)");
}

/// Test DELETE with EXISTS
#[test]
fn test_delete_with_exists() {
    let db = Database::open("memory://delete_exists").expect("Failed to create database");

    // Create products and inventory tables
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE inventory (
            id INTEGER,
            product_id INTEGER,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create inventory table");

    // Insert products
    db.execute(
        "INSERT INTO products (id, name, in_stock) VALUES
        (1, 'Laptop', true),
        (2, 'Mouse', true),
        (3, 'Book', true),
        (4, 'Phone', true)",
        (),
    )
    .expect("Failed to insert products");

    // Empty inventory - no rows
    // DELETE with EXISTS on empty table should not delete anything
    db.execute(
        "DELETE FROM products
         WHERE EXISTS (SELECT * FROM inventory)",
        (),
    )
    .expect("Failed to execute DELETE with EXISTS");

    // All 4 products should remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 products (no deletions)");
}

/// Test UPDATE with EXISTS
#[test]
fn test_update_with_exists() {
    let db = Database::open("memory://update_exists").expect("Failed to create database");

    // Create products and inventory tables
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE inventory (
            id INTEGER,
            product_id INTEGER,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create inventory table");

    // Insert products
    db.execute(
        "INSERT INTO products (id, name, in_stock) VALUES
        (1, 'Laptop', true),
        (2, 'Mouse', true),
        (3, 'Book', true),
        (4, 'Phone', true)",
        (),
    )
    .expect("Failed to insert products");

    // Add one inventory record
    db.execute(
        "INSERT INTO inventory (id, product_id, quantity) VALUES (1, 1, 10)",
        (),
    )
    .expect("Failed to insert inventory");

    // UPDATE with EXISTS - should update all products since inventory exists
    db.execute(
        "UPDATE products
         SET in_stock = false
         WHERE EXISTS (SELECT 1 FROM inventory)",
        (),
    )
    .expect("Failed to execute UPDATE with EXISTS");

    // All 4 products should be marked as out of stock
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products WHERE in_stock = false", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected all 4 products to be out of stock");
}
