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

//! Unique Index Tests
//!
//! Tests unique index creation and constraint enforcement

use stoolap::Database;

/// Test creating a unique index and verifying constraint enforcement
#[test]
fn test_unique_index_constraint() {
    let db = Database::open("memory://unique_index").expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Create a unique index on email
    db.execute("CREATE UNIQUE INDEX idx_users_email ON users (email)", ())
        .expect("Failed to create unique index");

    // Insert a row
    db.execute(
        "INSERT INTO users (id, name, email) VALUES (1, 'User1', 'user1@example.com')",
        (),
    )
    .expect("Failed to insert first row");

    // Try to insert a row with same email - should fail
    let result = db.execute(
        "INSERT INTO users (id, name, email) VALUES (2, 'User2', 'user1@example.com')",
        (),
    );

    assert!(
        result.is_err(),
        "Inserting duplicate email should have failed"
    );
}

/// Test unique index with NULL values
#[test]
fn test_unique_index_with_null() {
    let db = Database::open("memory://unique_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_sku ON products (sku)", ())
        .expect("Failed to create unique index");

    // Insert rows with NULL sku (should be allowed - multiple NULLs are unique)
    db.execute("INSERT INTO products VALUES (1, NULL, 'Product 1')", ())
        .expect("Failed to insert first NULL sku");
    db.execute("INSERT INTO products VALUES (2, NULL, 'Product 2')", ())
        .expect("Failed to insert second NULL sku");

    // Insert rows with actual sku
    db.execute(
        "INSERT INTO products VALUES (3, 'SKU-001', 'Product 3')",
        (),
    )
    .expect("Failed to insert SKU-001");

    // Try to insert duplicate sku - should fail
    let result = db.execute(
        "INSERT INTO products VALUES (4, 'SKU-001', 'Product 4')",
        (),
    );
    assert!(result.is_err(), "Duplicate SKU should have failed");

    // Verify we have 3 rows
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows");
}

/// Test dropping a unique index and verifying constraint is removed
#[test]
fn test_drop_unique_index() {
    let db = Database::open("memory://drop_unique").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, code TEXT)", ())
        .expect("Failed to create table");

    db.execute("CREATE UNIQUE INDEX idx_code ON items (code)", ())
        .expect("Failed to create unique index");

    // Insert initial row
    db.execute("INSERT INTO items VALUES (1, 'CODE-001')", ())
        .expect("Failed to insert first row");

    // Verify unique constraint is enforced
    let result = db.execute("INSERT INTO items VALUES (2, 'CODE-001')", ());
    assert!(
        result.is_err(),
        "Duplicate code should have failed before drop"
    );

    // Drop the unique index
    db.execute("DROP INDEX idx_code ON items", ())
        .expect("Failed to drop index");

    // Now we should be able to insert duplicate code
    db.execute("INSERT INTO items VALUES (2, 'CODE-001')", ())
        .expect("Failed to insert after drop");

    // Verify we have 2 rows with same code
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM items WHERE code = 'CODE-001'", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with same code after drop");
}

/// Test unique index on multiple columns
#[test]
fn test_multi_column_unique_index() {
    let db = Database::open("memory://multi_unique").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    // Create unique index on customer_id + order_date combination
    db.execute(
        "CREATE UNIQUE INDEX idx_customer_date ON orders (customer_id, order_date)",
        (),
    )
    .expect("Failed to create multi-column unique index");

    // Insert valid rows
    db.execute(
        "INSERT INTO orders VALUES (1, 100, '2025-01-01', 99.99)",
        (),
    )
    .expect("Failed to insert first order");
    db.execute(
        "INSERT INTO orders VALUES (2, 100, '2025-01-02', 149.99)",
        (),
    )
    .expect("Failed to insert second order (same customer, different date)");
    db.execute(
        "INSERT INTO orders VALUES (3, 200, '2025-01-01', 199.99)",
        (),
    )
    .expect("Failed to insert third order (different customer, same date)");

    // Try to insert duplicate customer_id + order_date - should fail
    let result = db.execute(
        "INSERT INTO orders VALUES (4, 100, '2025-01-01', 299.99)",
        (),
    );
    assert!(
        result.is_err(),
        "Duplicate customer_id + order_date should have failed"
    );

    // Verify we have 3 rows
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows");
}

/// Test SHOW INDEXES shows unique index correctly
#[test]
fn test_show_unique_index() {
    let db = Database::open("memory://show_unique").expect("Failed to create database");

    db.execute(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, account_no TEXT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE UNIQUE INDEX idx_account_no ON accounts (account_no)",
        (),
    )
    .expect("Failed to create unique index");

    let result = db
        .query("SHOW INDEXES FROM accounts", ())
        .expect("Failed to show indexes");

    let mut found_unique = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let index_name: String = row.get(1).unwrap();
        let is_unique: bool = row.get(4).unwrap();

        if index_name == "idx_account_no" {
            found_unique = true;
            assert!(is_unique, "Index should be marked as unique");
        }
    }

    assert!(found_unique, "Unique index should be found in SHOW INDEXES");
}

/// Test unique index enforcement during UPDATE
#[test]
fn test_unique_index_update() {
    let db = Database::open("memory://unique_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, employee_id TEXT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE UNIQUE INDEX idx_emp_id ON employees (employee_id)",
        (),
    )
    .expect("Failed to create unique index");

    // Insert initial rows
    db.execute("INSERT INTO employees VALUES (1, 'EMP-001', 'Alice')", ())
        .expect("Failed to insert Alice");
    db.execute("INSERT INTO employees VALUES (2, 'EMP-002', 'Bob')", ())
        .expect("Failed to insert Bob");

    // Try to update Bob's employee_id to Alice's - should fail
    let result = db.execute(
        "UPDATE employees SET employee_id = 'EMP-001' WHERE id = 2",
        (),
    );
    assert!(
        result.is_err(),
        "Update to duplicate employee_id should have failed"
    );

    // Verify Bob still has original employee_id
    let result = db
        .query("SELECT employee_id FROM employees WHERE id = 2", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let emp_id: String = row.get(0).unwrap();
        assert_eq!(emp_id, "EMP-002", "Bob's employee_id should be unchanged");
    }
}

/// Test IF NOT EXISTS with unique index
#[test]
fn test_unique_index_if_not_exists() {
    let db = Database::open("memory://unique_if_not_exists").expect("Failed to create database");

    db.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY, tag TEXT)", ())
        .expect("Failed to create table");

    // Create unique index
    db.execute("CREATE UNIQUE INDEX idx_tag ON tags (tag)", ())
        .expect("Failed to create unique index first time");

    // Try to create again without IF NOT EXISTS - should fail
    let result = db.execute("CREATE UNIQUE INDEX idx_tag ON tags (tag)", ());
    assert!(
        result.is_err(),
        "Creating duplicate index without IF NOT EXISTS should fail"
    );

    // Create with IF NOT EXISTS - should succeed (no-op)
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tag ON tags (tag)",
        (),
    )
    .expect("Failed with IF NOT EXISTS");
}
