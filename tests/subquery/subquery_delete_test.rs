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

//! DELETE with Subquery Tests
//!
//! Tests DELETE statements with IN and NOT IN subqueries

use stoolap::Database;

fn setup_delete_tables(db: &Database) {
    // Create customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            total_spent FLOAT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Create orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Insert customers
    db.execute(
        "INSERT INTO customers (id, name, total_spent) VALUES (1, 'Alice', 1000.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO customers (id, name, total_spent) VALUES (2, 'Bob', 500.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO customers (id, name, total_spent) VALUES (3, 'Charlie', 2000.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO customers (id, name, total_spent) VALUES (4, 'David', 100.0)",
        (),
    )
    .unwrap();

    // Insert orders
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (1, 1, 200.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (2, 2, 100.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (3, 3, 500.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (4, 1, 300.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (5, 3, 400.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES (6, 4, 50.0)",
        (),
    )
    .unwrap();
}

/// Test DELETE with IN subquery
#[test]
fn test_delete_with_in_subquery() {
    let db = Database::open("memory://delete_in").expect("Failed to create database");
    setup_delete_tables(&db);

    // Delete orders for customers with total_spent > 1000 (Charlie only)
    db.execute(
        "DELETE FROM orders WHERE customer_id IN (
            SELECT id FROM customers WHERE total_spent > 1000
        )",
        (),
    )
    .expect("Failed to delete with IN subquery");

    // Verify only orders for customer 3 (Charlie) were deleted
    let result = db
        .query("SELECT id FROM orders ORDER BY id", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Orders 3 and 5 (Charlie's) should be deleted
    assert_eq!(
        ids,
        vec![1, 2, 4, 6],
        "Expected orders 1, 2, 4, 6 to remain"
    );
}

/// Test DELETE with NOT IN subquery
#[test]
fn test_delete_with_not_in_subquery() {
    let db = Database::open("memory://delete_not_in").expect("Failed to create database");
    setup_delete_tables(&db);

    // First delete Charlie's orders (same as previous test)
    db.execute(
        "DELETE FROM orders WHERE customer_id IN (
            SELECT id FROM customers WHERE total_spent > 1000
        )",
        (),
    )
    .expect("Failed to delete with IN subquery");

    // Now delete customers who have no orders
    db.execute(
        "DELETE FROM customers WHERE id NOT IN (
            SELECT DISTINCT customer_id FROM orders
        )",
        (),
    )
    .expect("Failed to delete with NOT IN subquery");

    // Verify customer 3 (Charlie) was deleted
    let result = db
        .query("SELECT id, name FROM customers ORDER BY id", ())
        .expect("Failed to query");

    let mut customers: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        customers.push((id, name));
    }

    assert_eq!(customers.len(), 3, "Expected 3 customers after delete");
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (2, "Bob".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test DELETE all with IN subquery
#[test]
fn test_delete_all_matching() {
    let db = Database::open("memory://delete_all_match").expect("Failed to create database");
    setup_delete_tables(&db);

    // Delete all orders for customers who spent less than 600
    db.execute(
        "DELETE FROM orders WHERE customer_id IN (
            SELECT id FROM customers WHERE total_spent < 600
        )",
        (),
    )
    .expect("Failed to delete");

    // Count remaining orders
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders", ())
        .expect("Failed to count");

    // Bob (500) and David (100) had orders 2 and 6 - those should be deleted
    // Remaining: Alice's orders (1, 4) and Charlie's orders (3, 5)
    assert_eq!(count, 4, "Expected 4 orders to remain");
}

/// Test DELETE with empty subquery result
#[test]
fn test_delete_with_empty_subquery() {
    let db = Database::open("memory://delete_empty_sub").expect("Failed to create database");
    setup_delete_tables(&db);

    // Delete orders for customers with total_spent > 10000 (none match)
    db.execute(
        "DELETE FROM orders WHERE customer_id IN (
            SELECT id FROM customers WHERE total_spent > 10000
        )",
        (),
    )
    .expect("Failed to delete");

    // All orders should remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders", ())
        .expect("Failed to count");

    assert_eq!(count, 6, "Expected all 6 orders to remain");
}

/// Test DELETE with complex subquery condition
#[test]
fn test_delete_with_complex_subquery() {
    let db = Database::open("memory://delete_complex").expect("Failed to create database");
    setup_delete_tables(&db);

    // Delete orders where amount > 200 AND customer spent > 500
    db.execute(
        "DELETE FROM orders WHERE amount > 200 AND customer_id IN (
            SELECT id FROM customers WHERE total_spent > 500
        )",
        (),
    )
    .expect("Failed to delete");

    // Orders affected:
    // - Order 3 (customer 3 Charlie, amount 500, spent 2000) - DELETED
    // - Order 4 (customer 1 Alice, amount 300, spent 1000) - DELETED
    // - Order 5 (customer 3 Charlie, amount 400, spent 2000) - DELETED

    let result = db
        .query("SELECT id FROM orders ORDER BY id", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![1, 2, 6], "Expected orders 1, 2, 6 to remain");
}
