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

//! Unique Index Bug Tests
//!
//! Tests unique index constraint enforcement

use stoolap::Database;

/// Test that unique constraint is enforced on INSERT
#[test]
fn test_unique_index_basic() {
    let db = Database::open("memory://unique_basic").expect("Failed to create database");

    // Create table with unique index
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, email TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_users_email ON users(email)", ())
        .expect("Failed to create unique index");

    // Insert initial data
    db.execute(
        "INSERT INTO users (id, username, email) VALUES
         (1, 'user1', 'user1@example.com'),
         (2, 'user2', 'user2@example.com'),
         (3, 'user3', 'user3@example.com')",
        (),
    )
    .expect("Failed to insert initial data");

    // Try to insert duplicate email - should fail
    let result = db.execute(
        "INSERT INTO users (id, username, email) VALUES (4, 'duplicate', 'user1@example.com')",
        (),
    );

    assert!(
        result.is_err(),
        "Expected unique constraint violation, but insert succeeded"
    );
}

/// Test unique constraint with NULL values
#[test]
fn test_unique_index_with_nulls() {
    let db = Database::open("memory://unique_nulls").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, code TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_items_code ON items(code)", ())
        .expect("Failed to create unique index");

    // NULL values should not conflict with each other (SQL standard)
    db.execute("INSERT INTO items (id, code) VALUES (1, NULL)", ())
        .expect("Failed to insert first NULL");

    db.execute("INSERT INTO items (id, code) VALUES (2, NULL)", ())
        .expect("Failed to insert second NULL");

    // But non-NULL duplicates should still fail
    db.execute("INSERT INTO items (id, code) VALUES (3, 'ABC')", ())
        .expect("Failed to insert ABC");

    let result = db.execute("INSERT INTO items (id, code) VALUES (4, 'ABC')", ());
    assert!(
        result.is_err(),
        "Expected unique constraint violation for duplicate 'ABC'"
    );
}

/// Test unique constraint on UPDATE
#[test]
fn test_unique_index_on_update() {
    let db = Database::open("memory://unique_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_products_sku ON products(sku)", ())
        .expect("Failed to create unique index");

    db.execute(
        "INSERT INTO products (id, sku) VALUES (1, 'SKU001'), (2, 'SKU002')",
        (),
    )
    .expect("Failed to insert data");

    // Try to update to a duplicate SKU - should fail
    let result = db.execute("UPDATE products SET sku = 'SKU001' WHERE id = 2", ());
    assert!(
        result.is_err(),
        "Expected unique constraint violation on UPDATE"
    );

    // After failed update, try to update id=1 to a new unique value
    // (different row to avoid uncommitted changes issue)
    db.execute("UPDATE products SET sku = 'SKU003' WHERE id = 1", ())
        .expect("Failed to update to unique value");

    // Verify the update worked
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products WHERE sku = 'SKU003'", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

/// Test unique constraint with multiple columns
#[test]
fn test_unique_index_composite() {
    let db = Database::open("memory://unique_composite").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE UNIQUE INDEX idx_orders_customer_date ON orders(customer_id, order_date)",
        (),
    )
    .expect("Failed to create composite unique index");

    // Insert initial data
    db.execute(
        "INSERT INTO orders (id, customer_id, order_date) VALUES
         (1, 100, '2024-01-01'),
         (2, 100, '2024-01-02'),
         (3, 200, '2024-01-01')",
        (),
    )
    .expect("Failed to insert initial data");

    // Same customer_id + order_date combination should fail
    let result = db.execute(
        "INSERT INTO orders (id, customer_id, order_date) VALUES (4, 100, '2024-01-01')",
        (),
    );
    assert!(
        result.is_err(),
        "Expected unique constraint violation for duplicate customer_id + order_date"
    );

    // Different combination should succeed
    db.execute(
        "INSERT INTO orders (id, customer_id, order_date) VALUES (4, 100, '2024-01-03')",
        (),
    )
    .expect("Failed to insert new combination");
}

/// Test dropping and recreating unique index
#[test]
fn test_unique_index_drop_recreate() {
    let db = Database::open("memory://unique_drop").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_test_value ON test(value)", ())
        .expect("Failed to create unique index");

    db.execute("INSERT INTO test (id, value) VALUES (1, 'A')", ())
        .expect("Failed to insert data");

    // Drop the index
    db.execute("DROP INDEX idx_test_value ON test", ())
        .expect("Failed to drop index");

    // Now duplicate should be allowed
    db.execute("INSERT INTO test (id, value) VALUES (2, 'A')", ())
        .expect("Should allow duplicate after index dropped");

    // Verify both rows exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test WHERE value = 'A'", ())
        .expect("Failed to count");
    assert_eq!(count, 2);
}

/// Test unique index with case-sensitive strings
#[test]
fn test_unique_index_case_sensitive() {
    let db = Database::open("memory://unique_case").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE UNIQUE INDEX idx_users_username ON users(username)",
        (),
    )
    .expect("Failed to create unique index");

    // Insert 'Admin'
    db.execute("INSERT INTO users (id, username) VALUES (1, 'Admin')", ())
        .expect("Failed to insert Admin");

    // 'admin' should be different (case-sensitive)
    db.execute("INSERT INTO users (id, username) VALUES (2, 'admin')", ())
        .expect("Failed to insert admin - should be case-sensitive");

    // 'Admin' again should fail
    let result = db.execute("INSERT INTO users (id, username) VALUES (3, 'Admin')", ());
    assert!(
        result.is_err(),
        "Expected unique constraint violation for duplicate 'Admin'"
    );
}

/// Test unique constraint enforcement after DELETE
#[test]
fn test_unique_index_after_delete() {
    let db = Database::open("memory://unique_delete").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, code TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_items_code ON items(code)", ())
        .expect("Failed to create unique index");

    db.execute("INSERT INTO items (id, code) VALUES (1, 'ABC')", ())
        .expect("Failed to insert data");

    // Try to insert duplicate - should fail
    let result = db.execute("INSERT INTO items (id, code) VALUES (2, 'ABC')", ());
    assert!(result.is_err(), "Expected unique constraint violation");

    // Delete the original row
    db.execute("DELETE FROM items WHERE id = 1", ())
        .expect("Failed to delete row");

    // Now the same code should be allowed
    db.execute("INSERT INTO items (id, code) VALUES (2, 'ABC')", ())
        .expect("Should allow 'ABC' after original row deleted");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM items WHERE code = 'ABC'", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

/// Test unique index with empty string
#[test]
fn test_unique_index_empty_string() {
    let db = Database::open("memory://unique_empty").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_test_value ON test(value)", ())
        .expect("Failed to create unique index");

    // Insert empty string
    db.execute("INSERT INTO test (id, value) VALUES (1, '')", ())
        .expect("Failed to insert empty string");

    // Try to insert another empty string - should fail
    let result = db.execute("INSERT INTO test (id, value) VALUES (2, '')", ());
    assert!(
        result.is_err(),
        "Expected unique constraint violation for duplicate empty string"
    );

    // Non-empty should still work
    db.execute("INSERT INTO test (id, value) VALUES (3, 'test')", ())
        .expect("Failed to insert non-empty string");
}

/// Test unique index with numeric values
#[test]
fn test_unique_index_numeric() {
    let db = Database::open("memory://unique_numeric").expect("Failed to create database");

    db.execute(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, account_number INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE UNIQUE INDEX idx_accounts_number ON accounts(account_number)",
        (),
    )
    .expect("Failed to create unique index");

    db.execute(
        "INSERT INTO accounts (id, account_number) VALUES (1, 1000), (2, 2000), (3, 3000)",
        (),
    )
    .expect("Failed to insert data");

    // Try duplicate
    let result = db.execute(
        "INSERT INTO accounts (id, account_number) VALUES (4, 1000)",
        (),
    );
    assert!(
        result.is_err(),
        "Expected unique constraint violation for duplicate account number"
    );
}

/// Test unique index with transaction rollback
#[test]
fn test_unique_index_transaction_rollback() {
    let db = Database::open("memory://unique_rollback").expect("Failed to create database");

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_test_value ON test(value)", ())
        .expect("Failed to create unique index");

    db.execute("INSERT INTO test (id, value) VALUES (1, 'A')", ())
        .expect("Failed to insert initial data");

    // Start a transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Insert a new value in transaction
    db.execute("INSERT INTO test (id, value) VALUES (2, 'B')", ())
        .expect("Failed to insert in transaction");

    // Rollback
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // 'B' should now be available since transaction was rolled back
    db.execute("INSERT INTO test (id, value) VALUES (3, 'B')", ())
        .expect("Should allow 'B' after rollback");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test", ())
        .expect("Failed to count");
    assert_eq!(count, 2); // Original 'A' and new 'B'
}
