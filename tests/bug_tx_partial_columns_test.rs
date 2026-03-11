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

//! Regression test for transaction INSERT with partial column lists.
//!
//! Reproduces the exact scenario from reproduce_execute_batch.md:
//! Transaction INSERT omitting AUTO_INCREMENT id column was failing with
//! "invalid column count" because the transaction path validated param count
//! against total table columns instead of the INSERT column list.

use stoolap::Database;

#[test]
fn test_reproduce_execute_batch_bug() {
    let db =
        Database::open("memory://reproduce_execute_batch").expect("Failed to create database");

    // Step 1: Create tables (exact schema from reproduction)
    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            country TEXT NOT NULL,
            segment TEXT NOT NULL DEFAULT 'Standard',
            created_at TIMESTAMP NOT NULL
        )",
        (),
    )
    .expect("Failed to create customers table");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            price FLOAT NOT NULL,
            cost FLOAT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            channel TEXT NOT NULL,
            discount_pct FLOAT NOT NULL DEFAULT 0
        )",
        (),
    )
    .expect("Failed to create orders table");

    db.execute(
        "CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price FLOAT NOT NULL
        )",
        (),
    )
    .expect("Failed to create order_items table");

    // Step 2: Create indexes
    db.execute("CREATE INDEX idx_orders_date ON orders(order_date)", ())
        .expect("Failed to create index");
    db.execute(
        "CREATE INDEX idx_products_category ON products(category)",
        (),
    )
    .expect("Failed to create index");

    // Step 3: Non-transaction INSERT with partial columns (this always worked)
    db.execute(
        "INSERT INTO customers (name, email, country, segment, created_at) VALUES
            ('Alice Johnson', 'alice@example.com', 'US', 'Premium', '2024-01-15 10:30:00'),
            ('Bob Smith', 'bob@example.com', 'US', 'Standard', '2024-02-20 14:00:00'),
            ('Clara Mueller', 'clara@example.com', 'DE', 'Premium', '2024-03-05 09:15:00')",
        (),
    )
    .expect("Non-transaction partial column insert should work");

    db.execute(
        "INSERT INTO products (name, category, subcategory, price, cost, is_active) VALUES
            ('Laptop Pro 15', 'Electronics', 'Laptops', 1299.99, 850, true),
            ('Wireless Mouse', 'Electronics', 'Accessories', 29.99, 12, true),
            ('USB-C Hub', 'Electronics', 'Accessories', 49.99, 22, true)",
        (),
    )
    .expect("Non-transaction products insert should work");

    // Step 4: Transaction INSERT with partial columns (THE BUG)
    // This is what execute_batch does: BEGIN, INSERT x N, COMMIT
    db.execute("BEGIN", ()).expect("Failed to begin");

    db.execute(
        "INSERT INTO customers (name, email, country, segment, created_at) VALUES ('Test User 1', 'test1@example.com', 'US', 'Standard', '2024-10-01 10:00:00')",
        (),
    )
    .expect("Transaction partial column insert 1 should work");

    db.execute(
        "INSERT INTO customers (name, email, country, segment, created_at) VALUES ('Test User 2', 'test2@example.com', 'UK', 'Premium', '2024-10-02 11:00:00')",
        (),
    )
    .expect("Transaction partial column insert 2 should work");

    db.execute(
        "INSERT INTO customers (name, email, country, segment, created_at) VALUES ('Test User 3', 'test3@example.com', 'DE', 'Enterprise', '2024-10-03 12:00:00')",
        (),
    )
    .expect("Transaction partial column insert 3 should work");

    db.execute("COMMIT", ()).expect("Failed to commit");

    // Step 5: Verify all rows were inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM customers", ())
        .expect("Failed to count");
    assert_eq!(
        count, 6,
        "Should have 3 non-tx + 3 tx rows = 6 total customers"
    );

    // Verify the transaction-inserted rows exist
    let test_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM customers WHERE email LIKE 'test%'",
            (),
        )
        .expect("Failed to count test users");
    assert_eq!(
        test_count, 3,
        "All 3 transaction-inserted test users should exist"
    );

    // Verify auto-increment IDs are correct
    let max_id: i64 = db
        .query_one("SELECT MAX(id) FROM customers", ())
        .expect("Failed to get max id");
    assert_eq!(max_id, 6, "Auto-increment should have assigned IDs 1-6");

    // Verify default values were applied
    let segment: String = db
        .query_one(
            "SELECT segment FROM customers WHERE email = 'test1@example.com'",
            (),
        )
        .expect("Failed to query segment");
    assert_eq!(segment, "Standard", "Default segment should be applied");
}

#[test]
fn test_transaction_insert_default_values() {
    // Test that DEFAULT values work in transaction INSERT with partial columns
    let db = Database::open("memory://tx_defaults_bug").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            price FLOAT NOT NULL,
            cost FLOAT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert with partial columns omitting is_active (has DEFAULT TRUE)
    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "INSERT INTO products (name, category, subcategory, price, cost) VALUES ('Widget', 'Tools', 'Hand Tools', 9.99, 5.00)",
        (),
    )
    .expect("Transaction insert omitting default column should work");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let active: bool = db
        .query_one(
            "SELECT is_active FROM products WHERE name = 'Widget'",
            (),
        )
        .expect("Failed to query");
    assert!(active, "DEFAULT TRUE should be applied for omitted column");
}

#[test]
fn test_transaction_insert_all_columns_still_works() {
    // Verify that inserting with ALL columns explicitly listed still works
    let db = Database::open("memory://tx_all_cols").expect("Failed to create database");

    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            country TEXT NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "INSERT INTO customers (id, name, email, country) VALUES (1, 'Alice', 'alice@test.com', 'US')",
        (),
    )
    .expect("Transaction insert with all columns should work");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM customers", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

#[test]
fn test_transaction_insert_rollback_partial_columns() {
    // Verify rollback works correctly with partial column inserts
    let db = Database::open("memory://tx_rollback_partial").expect("Failed to create database");

    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("BEGIN", ()).expect("Failed to begin");
    db.execute(
        "INSERT INTO customers (name, email) VALUES ('Should Not Exist', 'gone@test.com')",
        (),
    )
    .expect("Insert should work");
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM customers", ())
        .expect("Failed to count");
    assert_eq!(count, 0, "Rolled back row should not exist");
}
