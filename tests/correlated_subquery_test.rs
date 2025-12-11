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

//! Correlated Subquery Tests
//!
//! Tests for correlated subqueries that reference columns from outer queries.

use stoolap::Database;

fn setup_test_tables(db: &Database) {
    // Create customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Create orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount FLOAT,
            status TEXT
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

    // Insert orders - Alice has 2 orders, Charlie has 1, David has 1
    // Bob has no orders
    db.execute(
        "INSERT INTO orders (id, customer_id, amount, status) VALUES
        (1, 1, 100.0, 'completed'),
        (2, 1, 200.0, 'pending'),
        (3, 3, 150.0, 'completed'),
        (4, 4, 300.0, 'completed')",
        (),
    )
    .expect("Failed to insert orders");
}

// ========================
// Correlated EXISTS Tests
// ========================

/// Test correlated EXISTS - customers who have at least one order
#[test]
fn test_correlated_exists_basic() {
    let db = Database::open("memory://correlated_exists_basic").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (1), Charlie (3), David (4) have orders; Bob (2) does not
    assert_eq!(customers.len(), 3, "Expected 3 customers with orders");
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test correlated NOT EXISTS - customers who have no orders
#[test]
fn test_correlated_not_exists() {
    let db = Database::open("memory://correlated_not_exists").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only Bob has no orders
    assert_eq!(customers.len(), 1, "Expected 1 customer without orders");
    assert_eq!(customers[0], (2, "Bob".to_string()));
}

/// Test correlated EXISTS with additional conditions
#[test]
fn test_correlated_exists_with_condition() {
    let db =
        Database::open("memory://correlated_exists_condition").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have at least one completed order
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.id AND o.status = 'completed'
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice, Charlie, David have completed orders
    assert_eq!(
        customers.len(),
        3,
        "Expected 3 customers with completed orders"
    );
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test correlated EXISTS - customers with high-value orders
#[test]
fn test_correlated_exists_high_value() {
    let db =
        Database::open("memory://correlated_exists_high_value").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have at least one order > 150
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.id AND o.amount > 150
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (200), David (300) have orders > 150
    assert_eq!(
        customers.len(),
        2,
        "Expected 2 customers with high-value orders"
    );
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (4, "David".to_string()));
}

// ========================
// Correlated Scalar Subquery Tests
// ========================

/// Test correlated scalar subquery - get order count per customer
#[test]
fn test_correlated_scalar_subquery_count() {
    let db = Database::open("memory://correlated_scalar_count").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String, i64)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (
                row.get::<i64>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<i64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(customers.len(), 4);
    assert_eq!(customers[0], (1, "Alice".to_string(), 2)); // Alice has 2 orders
    assert_eq!(customers[1], (2, "Bob".to_string(), 0)); // Bob has 0 orders
    assert_eq!(customers[2], (3, "Charlie".to_string(), 1)); // Charlie has 1 order
    assert_eq!(customers[3], (4, "David".to_string(), 1)); // David has 1 order
}

/// Test correlated scalar subquery - get total order amount per customer
#[test]
fn test_correlated_scalar_subquery_sum() {
    let db = Database::open("memory://correlated_scalar_sum").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id) AS total_amount
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let total: Option<f64> = row.get(2).ok();
        customers.push((id, name, total));
    }

    assert_eq!(customers.len(), 4);
    assert_eq!(customers[0].0, 1);
    assert_eq!(customers[0].1, "Alice".to_string());
    assert!((customers[0].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // Alice: 100 + 200 = 300

    assert_eq!(customers[1].0, 2);
    assert_eq!(customers[1].1, "Bob".to_string());
    // Bob has NULL total (no orders)

    assert_eq!(customers[2].0, 3);
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie: 150

    assert_eq!(customers[3].0, 4);
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David: 300
}

/// Test correlated scalar subquery - max order amount per customer
#[test]
fn test_correlated_scalar_subquery_max() {
    let db = Database::open("memory://correlated_scalar_max").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT MAX(amount) FROM orders o WHERE o.customer_id = c.id) AS max_order
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let max_order: Option<f64> = row.get(2).ok();
        customers.push((id, name, max_order));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 200.0).abs() < 0.1); // Alice max: 200
    assert!(customers[1].2.is_none() || customers[1].2 == Some(0.0)); // Bob: NULL/0
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie max: 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David max: 300
}

// ========================
// Correlated IN Subquery Tests
// ========================

/// Test correlated IN subquery
#[test]
fn test_correlated_in_subquery() {
    let db = Database::open("memory://correlated_in").expect("Failed to create database");
    setup_test_tables(&db);

    // Find customers whose ID is in their own order's customer_id
    // (This is a contrived example but tests the correlation)
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE c.id IN (SELECT o.customer_id FROM orders o WHERE o.customer_id = c.id)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // All customers with orders should match
    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

// ========================
// Combined Tests
// ========================

/// Test correlated EXISTS combined with other WHERE conditions
#[test]
fn test_correlated_exists_with_outer_where() {
    let db = Database::open("memory://correlated_exists_outer_where")
        .expect("Failed to create database");
    setup_test_tables(&db);

    // USA customers who have orders
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE c.country = 'USA'
               AND EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice and Charlie are from USA and have orders
    assert_eq!(customers.len(), 2);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
}

/// Test deeply nested correlated subquery
#[test]
fn test_nested_correlated_exists() {
    let db = Database::open("memory://nested_correlated").expect("Failed to create database");
    setup_test_tables(&db);

    // Create another related table
    db.execute(
        "CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product TEXT,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create order_items table");

    db.execute(
        "INSERT INTO order_items (id, order_id, product, quantity) VALUES
        (1, 1, 'Widget', 5),
        (2, 1, 'Gadget', 3),
        (3, 3, 'Widget', 2)",
        (),
    )
    .expect("Failed to insert order_items");

    // Customers who have orders that have items
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.id
                  AND EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.id)
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (order 1 has items) and Charlie (order 3 has items)
    assert_eq!(customers.len(), 2);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
}

/// Test correlated subquery with table alias
#[test]
fn test_correlated_with_alias() {
    let db = Database::open("memory://correlated_alias").expect("Failed to create database");
    setup_test_tables(&db);

    // Using different alias names
    let result = db
        .query(
            "SELECT cust.id, cust.name
             FROM customers cust
             WHERE EXISTS (SELECT 1 FROM orders ord WHERE ord.customer_id = cust.id)
             ORDER BY cust.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test that uncorrelated subqueries still work
#[test]
fn test_uncorrelated_still_works() {
    let db = Database::open("memory://uncorrelated_check").expect("Failed to create database");
    setup_test_tables(&db);

    // Uncorrelated EXISTS - all customers if any order exists
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (SELECT 1 FROM orders)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let count = result.count();
    assert_eq!(count, 4, "Expected all 4 customers since orders exist");
}

// ========================
// Additional Comprehensive Tests
// ========================

/// Test multiple correlated scalar subqueries in a single SELECT
#[test]
fn test_multiple_scalar_subqueries_in_select() {
    let db = Database::open("memory://multi_scalar_subquery").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count,
                    (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id) AS total_amount,
                    (SELECT AVG(amount) FROM orders o WHERE o.customer_id = c.id) AS avg_amount
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, i64, Option<f64>, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let count: i64 = row.get(2).unwrap();
        let total: Option<f64> = row.get(3).ok();
        let avg: Option<f64> = row.get(4).ok();
        customers.push((id, name, count, total, avg));
    }

    assert_eq!(customers.len(), 4);

    // Alice: 2 orders, total 300, avg 150
    assert_eq!(customers[0].0, 1);
    assert_eq!(customers[0].2, 2);
    assert!((customers[0].3.unwrap_or(0.0) - 300.0).abs() < 0.1);
    assert!((customers[0].4.unwrap_or(0.0) - 150.0).abs() < 0.1);

    // Bob: 0 orders
    assert_eq!(customers[1].0, 2);
    assert_eq!(customers[1].2, 0);

    // Charlie: 1 order, total 150, avg 150
    assert_eq!(customers[2].0, 3);
    assert_eq!(customers[2].2, 1);
    assert!((customers[2].3.unwrap_or(0.0) - 150.0).abs() < 0.1);

    // David: 1 order, total 300, avg 300
    assert_eq!(customers[3].0, 4);
    assert_eq!(customers[3].2, 1);
    assert!((customers[3].3.unwrap_or(0.0) - 300.0).abs() < 0.1);
}

/// Test correlated scalar subquery with MIN aggregate
#[test]
fn test_correlated_scalar_subquery_min() {
    let db = Database::open("memory://correlated_scalar_min").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT MIN(amount) FROM orders o WHERE o.customer_id = c.id) AS min_order
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let min_order: Option<f64> = row.get(2).ok();
        customers.push((id, name, min_order));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 100.0).abs() < 0.1); // Alice min: 100
    assert!(customers[1].2.is_none() || customers[1].2 == Some(0.0)); // Bob: NULL/0
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie min: 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David min: 300
}

/// Test correlated scalar subquery with AVG aggregate
#[test]
fn test_correlated_scalar_subquery_avg() {
    let db = Database::open("memory://correlated_scalar_avg").expect("Failed to create database");
    setup_test_tables(&db);

    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT AVG(amount) FROM orders o WHERE o.customer_id = c.id) AS avg_order
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let avg_order: Option<f64> = row.get(2).ok();
        customers.push((id, name, avg_order));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Alice avg: (100+200)/2 = 150
                                                                  // Bob: NULL (no orders)
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie avg: 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David avg: 300
}

/// Test correlated subquery with comparison operator (scalar > value)
#[test]
fn test_correlated_scalar_comparison() {
    let db =
        Database::open("memory://correlated_scalar_comparison").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers whose average order amount is greater than 150
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE (SELECT AVG(amount) FROM orders o WHERE o.customer_id = c.id) > 150
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only David has avg > 150 (300)
    // Alice has avg = 150, Charlie has avg = 150
    assert_eq!(customers.len(), 1);
    assert_eq!(customers[0], (4, "David".to_string()));
}

/// Test correlated subquery comparing with another correlated subquery
#[test]
fn test_correlated_scalar_equal_comparison() {
    let db = Database::open("memory://correlated_scalar_equal").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers whose max order equals their total (i.e., they have exactly one order)
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE (SELECT MAX(amount) FROM orders o WHERE o.customer_id = c.id) =
                   (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Charlie (150=150) and David (300=300) have single orders
    // Alice has 2 orders: max=200, sum=300
    assert_eq!(customers.len(), 2);
    assert_eq!(customers[0], (3, "Charlie".to_string()));
    assert_eq!(customers[1], (4, "David".to_string()));
}

/// Test self-referencing correlated subquery (same table in outer and inner)
#[test]
fn test_self_referencing_correlated_subquery() {
    let db = Database::open("memory://self_ref_correlated").expect("Failed to create database");

    // Create employees table with manager relationship
    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            manager_id INTEGER,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create employees table");

    db.execute(
        "INSERT INTO employees (id, name, manager_id, salary) VALUES
        (1, 'CEO', NULL, 200000),
        (2, 'VP Sales', 1, 150000),
        (3, 'VP Eng', 1, 160000),
        (4, 'Sales Rep', 2, 80000),
        (5, 'Engineer', 3, 100000),
        (6, 'Sr Engineer', 3, 120000)",
        (),
    )
    .expect("Failed to insert employees");

    // Find employees who earn more than their manager
    let result = db
        .query(
            "SELECT e.id, e.name, e.salary
             FROM employees e
             WHERE e.salary > (SELECT m.salary FROM employees m WHERE m.id = e.manager_id)
             ORDER BY e.id",
            (),
        )
        .expect("Failed to execute query");

    let employees: Vec<(i64, String, f64)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (
                row.get::<i64>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<f64>(2).unwrap(),
            )
        })
        .collect();

    // No one earns more than their manager in this dataset
    assert_eq!(employees.len(), 0);
}

/// Test self-referencing - employees who have subordinates
#[test]
fn test_self_referencing_exists() {
    let db = Database::open("memory://self_ref_exists").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            manager_id INTEGER,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create employees table");

    db.execute(
        "INSERT INTO employees (id, name, manager_id, salary) VALUES
        (1, 'CEO', NULL, 200000),
        (2, 'VP Sales', 1, 150000),
        (3, 'VP Eng', 1, 160000),
        (4, 'Sales Rep', 2, 80000),
        (5, 'Engineer', 3, 100000),
        (6, 'Sr Engineer', 3, 120000)",
        (),
    )
    .expect("Failed to insert employees");

    // Find managers (employees who have subordinates)
    let result = db
        .query(
            "SELECT e.id, e.name
             FROM employees e
             WHERE EXISTS (SELECT 1 FROM employees sub WHERE sub.manager_id = e.id)
             ORDER BY e.id",
            (),
        )
        .expect("Failed to execute query");

    let managers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // CEO, VP Sales, VP Eng are managers
    assert_eq!(managers.len(), 3);
    assert_eq!(managers[0], (1, "CEO".to_string()));
    assert_eq!(managers[1], (2, "VP Sales".to_string()));
    assert_eq!(managers[2], (3, "VP Eng".to_string()));
}

/// Test correlated subquery with multiple outer column references
#[test]
fn test_multiple_outer_references() {
    let db = Database::open("memory://multi_outer_ref").expect("Failed to create database");
    setup_test_tables(&db);

    // Find customers from USA who have orders with amount > 100
    // referencing both c.id and c.country from outer
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.id
                  AND o.amount > 100
                  AND c.country = 'USA'
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (USA, has order 200) and Charlie would match but Charlie's only order is 150
    assert_eq!(customers.len(), 2);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
}

/// Test correlated EXISTS with OR conditions
#[test]
fn test_correlated_exists_with_or() {
    let db = Database::open("memory://correlated_exists_or").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have either a completed order OR an order > 250
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.id
                  AND (o.status = 'completed' OR o.amount > 250)
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (completed order), Charlie (completed), David (completed AND > 250)
    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test correlated EXISTS with multiple EXISTS in same WHERE
#[test]
fn test_multiple_exists_conditions() {
    let db = Database::open("memory://multi_exists").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have both a completed order AND a pending order
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.status = 'completed')
               AND EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.status = 'pending')
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only Alice has both completed (100) and pending (200) orders
    assert_eq!(customers.len(), 1);
    assert_eq!(customers[0], (1, "Alice".to_string()));
}

/// Test correlated subquery with empty result from subquery
#[test]
fn test_correlated_empty_subquery_result() {
    let db = Database::open("memory://empty_subquery").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have orders with status 'cancelled' (none exist)
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.status = 'cancelled')
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let count = result.count();
    assert_eq!(count, 0, "Expected no customers with cancelled orders");
}

/// Test correlated subquery against empty table
#[test]
fn test_correlated_against_empty_table() {
    let db = Database::open("memory://empty_table_correlated").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE reviews (id INTEGER PRIMARY KEY, product_id INTEGER, rating INTEGER)",
        (),
    )
    .expect("Failed to create reviews table");

    db.execute(
        "INSERT INTO products (id, name) VALUES (1, 'Widget'), (2, 'Gadget')",
        (),
    )
    .expect("Failed to insert products");

    // No reviews exist - all EXISTS should be false
    let result = db
        .query(
            "SELECT p.id, p.name
             FROM products p
             WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.product_id = p.id)
             ORDER BY p.id",
            (),
        )
        .expect("Failed to execute query");

    let count = result.count();
    assert_eq!(count, 0, "Expected no products with reviews");

    // NOT EXISTS should return all products
    let result = db
        .query(
            "SELECT p.id, p.name
             FROM products p
             WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.product_id = p.id)
             ORDER BY p.id",
            (),
        )
        .expect("Failed to execute query");

    let products: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(products.len(), 2);
    assert_eq!(products[0], (1, "Widget".to_string()));
    assert_eq!(products[1], (2, "Gadget".to_string()));
}

/// Test correlated scalar subquery returning NULL for customers without orders
#[test]
fn test_correlated_scalar_returns_null() {
    let db = Database::open("memory://scalar_null").expect("Failed to create database");
    setup_test_tables(&db);

    // Get the minimum order amount for each customer (Bob has none, so NULL)
    // Note: Using unqualified column name in aggregate for correct behavior
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT MIN(amount) FROM orders o WHERE o.customer_id = c.id) AS min_amount
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let min_amount: Option<f64> = row.get(2).ok();
        customers.push((id, name, min_amount));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 100.0).abs() < 0.1); // Alice - min 100
    assert!(customers[1].2.is_none()); // Bob - NULL (no orders)
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie - min 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David - min 300
}

/// Test correlated subquery with NULL values in correlation column
#[test]
fn test_correlated_with_null_in_correlation() {
    let db = Database::open("memory://null_correlation").expect("Failed to create database");

    db.execute(
        "CREATE TABLE parent_table (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create parent table");

    db.execute(
        "CREATE TABLE child_table (id INTEGER PRIMARY KEY, parent_id INTEGER, data TEXT)",
        (),
    )
    .expect("Failed to create child table");

    db.execute(
        "INSERT INTO parent_table (id, value) VALUES (1, 'A'), (2, 'B'), (3, 'C')",
        (),
    )
    .expect("Failed to insert parent data");

    // Insert children - some with NULL parent_id
    db.execute(
        "INSERT INTO child_table (id, parent_id, data) VALUES
        (1, 1, 'Child1'),
        (2, NULL, 'Orphan'),
        (3, 1, 'Child2')",
        (),
    )
    .expect("Failed to insert child data");

    // Parents with children
    let result = db
        .query(
            "SELECT p.id, p.value
             FROM parent_table p
             WHERE EXISTS (SELECT 1 FROM child_table c WHERE c.parent_id = p.id)
             ORDER BY p.id",
            (),
        )
        .expect("Failed to execute query");

    let parents: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only parent 1 has children
    assert_eq!(parents.len(), 1);
    assert_eq!(parents[0], (1, "A".to_string()));
}

/// Test deeply nested correlated subquery (3 levels)
#[test]
fn test_deeply_nested_correlated_three_levels() {
    let db = Database::open("memory://three_level_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create departments table");

    db.execute(
        "CREATE TABLE teams (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create teams table");

    db.execute(
        "CREATE TABLE members (id INTEGER PRIMARY KEY, team_id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create members table");

    db.execute(
        "INSERT INTO departments (id, name) VALUES (1, 'Engineering'), (2, 'Sales'), (3, 'HR')",
        (),
    )
    .expect("Failed to insert departments");

    db.execute(
        "INSERT INTO teams (id, dept_id, name) VALUES
        (1, 1, 'Backend'),
        (2, 1, 'Frontend'),
        (3, 2, 'Inside Sales')",
        (),
    )
    .expect("Failed to insert teams");

    db.execute(
        "INSERT INTO members (id, team_id, name) VALUES
        (1, 1, 'Alice'),
        (2, 1, 'Bob'),
        (3, 2, 'Charlie')",
        (),
    )
    .expect("Failed to insert members");

    // Departments that have teams that have members
    let result = db
        .query(
            "SELECT d.id, d.name
             FROM departments d
             WHERE EXISTS (
                SELECT 1 FROM teams t
                WHERE t.dept_id = d.id
                  AND EXISTS (
                    SELECT 1 FROM members m
                    WHERE m.team_id = t.id
                  )
             )
             ORDER BY d.id",
            (),
        )
        .expect("Failed to execute query");

    let departments: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only Engineering has teams with members (Backend and Frontend)
    // Sales has a team but no members
    // HR has no teams
    assert_eq!(departments.len(), 1);
    assert_eq!(departments[0], (1, "Engineering".to_string()));
}

/// Test correlated NOT IN subquery
#[test]
fn test_correlated_not_in() {
    let db = Database::open("memory://correlated_not_in").expect("Failed to create database");
    setup_test_tables(&db);

    // Find customers whose ID is NOT in the list of customers with completed orders
    // This tests NOT IN with a correlated element
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE c.id NOT IN (
                SELECT DISTINCT o.customer_id FROM orders o WHERE o.status = 'completed'
             )
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Bob has no orders at all
    // Alice, Charlie, David all have completed orders
    assert_eq!(customers.len(), 1);
    assert_eq!(customers[0], (2, "Bob".to_string()));
}

/// Test correlated subquery with arithmetic in correlation
#[test]
fn test_correlated_with_arithmetic() {
    let db = Database::open("memory://correlated_arithmetic").expect("Failed to create database");
    setup_test_tables(&db);

    // Find orders where amount > customer_id * 100
    // Tests using outer column in arithmetic expression
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id AND o.amount > c.id * 50) AS high_orders
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String, i64)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (
                row.get::<i64>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<i64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(customers.len(), 4);
    // Alice (id=1): orders > 50, has 100 and 200, count = 2
    assert_eq!(customers[0], (1, "Alice".to_string(), 2));
    // Bob (id=2): no orders, count = 0
    assert_eq!(customers[1], (2, "Bob".to_string(), 0));
    // Charlie (id=3): orders > 150, has 150, count = 0
    assert_eq!(customers[2], (3, "Charlie".to_string(), 0));
    // David (id=4): orders > 200, has 300, count = 1
    assert_eq!(customers[3], (4, "David".to_string(), 1));
}

/// Test correlated subquery with DISTINCT in non-aggregate context
#[test]
fn test_correlated_with_distinct() {
    let db = Database::open("memory://correlated_distinct").expect("Failed to create database");
    setup_test_tables(&db);

    // Use SELECT DISTINCT in the subquery to count unique customer_ids per country
    // This tests DISTINCT within the subquery itself
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count
             FROM customers c
             WHERE c.id IN (SELECT DISTINCT customer_id FROM orders)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String, i64)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (
                row.get::<i64>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<i64>(2).unwrap(),
            )
        })
        .collect();

    // Only customers with orders: Alice (2), Charlie (1), David (1)
    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string(), 2));
    assert_eq!(customers[1], (3, "Charlie".to_string(), 1));
    assert_eq!(customers[2], (4, "David".to_string(), 1));
}

/// Test correlated EXISTS in OR branch of WHERE clause
#[test]
fn test_correlated_exists_in_or_branch() {
    let db = Database::open("memory://exists_or_branch").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers from Canada OR customers who have high-value orders
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE c.country = 'Canada'
                OR EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 150)
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice (has 200), David (Canada AND has 300)
    assert_eq!(customers.len(), 2);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (4, "David".to_string()));
}

/// Test correlated subquery referencing outer table without alias
#[test]
fn test_correlated_without_outer_alias() {
    let db = Database::open("memory://no_outer_alias").expect("Failed to create database");
    setup_test_tables(&db);

    // Query without alias on outer table
    let result = db
        .query(
            "SELECT customers.id, customers.name
             FROM customers
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = customers.id)
             ORDER BY customers.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test correlated scalar subquery in WHERE with IS NULL check
#[test]
fn test_correlated_scalar_is_null() {
    let db = Database::open("memory://scalar_is_null").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have no orders (scalar subquery returns NULL)
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE (SELECT MAX(amount) FROM orders o WHERE o.customer_id = c.id) IS NULL
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Only Bob has no orders
    assert_eq!(customers.len(), 1);
    assert_eq!(customers[0], (2, "Bob".to_string()));
}

/// Test correlated scalar subquery in WHERE with IS NOT NULL check
#[test]
fn test_correlated_scalar_is_not_null() {
    let db = Database::open("memory://scalar_is_not_null").expect("Failed to create database");
    setup_test_tables(&db);

    // Customers who have at least one order
    let result = db
        .query(
            "SELECT c.id, c.name
             FROM customers c
             WHERE (SELECT MAX(amount) FROM orders o WHERE o.customer_id = c.id) IS NOT NULL
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let customers: Vec<(i64, String)> = result
        .map(|r| {
            let row = r.expect("Failed to get row");
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect();

    // Alice, Charlie, David have orders
    assert_eq!(customers.len(), 3);
    assert_eq!(customers[0], (1, "Alice".to_string()));
    assert_eq!(customers[1], (3, "Charlie".to_string()));
    assert_eq!(customers[2], (4, "David".to_string()));
}

/// Test correlated scalar subquery with qualified column names in aggregate (e.g., o.amount)
#[test]
fn test_correlated_scalar_qualified_column_names() {
    let db = Database::open("memory://scalar_qualified").expect("Failed to create database");
    setup_test_tables(&db);

    // Test MAX with qualified column name
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT MAX(o.amount) FROM orders o WHERE o.customer_id = c.id) AS max_order
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let max_order: Option<f64> = row.get(2).ok();
        customers.push((id, name, max_order));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 200.0).abs() < 0.1); // Alice max: 200
    assert!(customers[1].2.is_none()); // Bob: NULL
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie max: 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David max: 300
}

/// Test correlated scalar subquery with qualified column in MIN
#[test]
fn test_correlated_scalar_qualified_min() {
    let db = Database::open("memory://scalar_qualified_min").expect("Failed to create database");
    setup_test_tables(&db);

    // Test MIN with qualified column name (o.id)
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT MIN(o.id) FROM orders o WHERE o.customer_id = c.id) AS first_order_id
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<i64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let first_order_id: Option<i64> = row.get(2).ok();
        customers.push((id, name, first_order_id));
    }

    assert_eq!(customers.len(), 4);
    assert_eq!(customers[0].2, Some(1)); // Alice - order 1
    assert!(customers[1].2.is_none()); // Bob - NULL (no orders)
    assert_eq!(customers[2].2, Some(3)); // Charlie - order 3
    assert_eq!(customers[3].2, Some(4)); // David - order 4
}

/// Test correlated scalar subquery with qualified column in SUM
#[test]
fn test_correlated_scalar_qualified_sum() {
    let db = Database::open("memory://scalar_qualified_sum").expect("Failed to create database");
    setup_test_tables(&db);

    // Test SUM with qualified column name
    let result = db
        .query(
            "SELECT c.id, c.name,
                    (SELECT SUM(o.amount) FROM orders o WHERE o.customer_id = c.id) AS total_amount
             FROM customers c
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute query");

    let mut customers: Vec<(i64, String, Option<f64>)> = Vec::new();
    for r in result {
        let row = r.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let total: Option<f64> = row.get(2).ok();
        customers.push((id, name, total));
    }

    assert_eq!(customers.len(), 4);
    assert!((customers[0].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // Alice: 100 + 200 = 300
    assert!(customers[1].2.is_none()); // Bob: NULL
    assert!((customers[2].2.unwrap_or(0.0) - 150.0).abs() < 0.1); // Charlie: 150
    assert!((customers[3].2.unwrap_or(0.0) - 300.0).abs() < 0.1); // David: 300
}
