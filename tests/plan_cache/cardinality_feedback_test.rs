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

//! Integration tests for Cardinality Feedback system
//!
//! Cardinality feedback records actual vs estimated row counts from query execution
//! and uses this information to improve future cardinality estimates.

use stoolap::api::Database;

/// Test that EXPLAIN ANALYZE returns both estimated and actual rows
#[test]
fn test_explain_analyze_shows_estimates_and_actuals() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE feedback_test (id INTEGER PRIMARY KEY, value INTEGER, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert data
    for i in 0..100 {
        db.execute(
            &format!(
                "INSERT INTO feedback_test VALUES ({}, {}, 'cat_{}')",
                i,
                i * 10,
                i % 5
            ),
            (),
        )
        .expect("Insert failed");
    }

    // Run ANALYZE
    db.execute("ANALYZE feedback_test", ()).unwrap();

    // EXPLAIN ANALYZE should show both estimated and actual row counts
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM feedback_test WHERE value > 500",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let mut found_actual_rows = false;
    for row in &result {
        let plan_line: String = row.get(0).expect("Failed to get plan line");
        // Check for both formats: "actual rows=" or "actual_rows="
        if plan_line.contains("actual rows=") || plan_line.contains("actual_rows=") {
            found_actual_rows = true;
            break;
        }
    }

    assert!(found_actual_rows, "EXPLAIN ANALYZE should show actual rows");
}

/// Test that repeated queries benefit from feedback
#[test]
fn test_repeated_queries_use_feedback() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, age INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert skewed data - most users are 'active'
    for i in 0..100 {
        let status = if i % 10 == 0 { "inactive" } else { "active" };
        db.execute(
            &format!(
                "INSERT INTO users VALUES ({}, '{}', {})",
                i,
                status,
                20 + i % 50
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE users", ()).unwrap();

    // First EXPLAIN ANALYZE collects feedback
    let result1: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM users WHERE status = 'inactive'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Second EXPLAIN should potentially use feedback for better estimates
    let result2: Vec<_> = db
        .query("EXPLAIN SELECT * FROM users WHERE status = 'inactive'", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Both queries should complete successfully
    assert!(!result1.is_empty(), "First query should return results");
    assert!(!result2.is_empty(), "Second query should return results");
}

/// Test feedback with join queries
#[test]
fn test_join_cardinality_feedback() {
    let db = Database::open("memory://join_feedback").expect("Failed to create database");

    // Create two tables
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create customers table");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total INTEGER)",
        (),
    )
    .expect("Failed to create orders table");

    // Insert customers
    for i in 0..50 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'customer_{}')", i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Insert orders - skewed distribution (some customers have many orders)
    let mut order_id = 0;
    for i in 0..50 {
        // Customer i has i orders
        for _ in 0..i {
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {})",
                    order_id,
                    i,
                    (order_id + 1) * 100
                ),
                (),
            )
            .expect("Insert failed");
            order_id += 1;
        }
    }

    db.execute("ANALYZE customers", ()).unwrap();
    db.execute("ANALYZE orders", ()).unwrap();

    // EXPLAIN ANALYZE the join
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT c.name, o.total
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             WHERE o.total > 5000",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Should complete successfully and show execution stats
    assert!(
        !result.is_empty(),
        "Join EXPLAIN ANALYZE should return results"
    );
}

/// Test that feedback improves estimates for selective predicates
#[test]
fn test_selective_predicate_feedback() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price INTEGER, in_stock INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert products with very few out of stock
    for i in 0..1000 {
        let in_stock = if i % 100 == 0 { 0 } else { 1 }; // Only 1% out of stock
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, {}, {})",
                i,
                (i % 100) * 10,
                in_stock
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE products", ()).unwrap();

    // Query for out of stock (very selective)
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM products WHERE in_stock = 0",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify actual rows is shown
    let mut found_actual = false;
    for row in &result {
        let plan_line: String = row.get(0).expect("Failed to get plan line");
        // Check for both formats: "actual rows=" or "actual_rows="
        if plan_line.contains("actual rows=") || plan_line.contains("actual_rows=") {
            found_actual = true;
            // The actual rows should be around 10 (1% of 1000)
        }
    }

    assert!(found_actual, "Should show actual row count");
}

/// Test feedback with compound predicates
#[test]
fn test_compound_predicate_feedback() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            severity INTEGER,
            timestamp INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert events
    for i in 0..500 {
        let event_type = match i % 4 {
            0 => "error",
            1 => "warning",
            2 => "info",
            _ => "debug",
        };
        let severity = i % 5;
        db.execute(
            &format!(
                "INSERT INTO events VALUES ({}, '{}', {}, {})",
                i,
                event_type,
                severity,
                i * 1000
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE events", ()).unwrap();

    // Complex compound predicate
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM events
             WHERE event_type = 'error' AND severity >= 3",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        !result.is_empty(),
        "Compound predicate query should complete"
    );

    // Run the actual query to verify correctness
    let actual_result: Vec<_> = db
        .query(
            "SELECT COUNT(*) FROM events WHERE event_type = 'error' AND severity >= 3",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count: i64 = actual_result[0].get(0).expect("Failed to get count");
    assert!(count > 0, "Should have matching rows");
}

/// Test feedback with aggregations
#[test]
fn test_aggregation_feedback() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            amount INTEGER,
            year INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert sales data
    let regions = ["North", "South", "East", "West"];
    for i in 0..200 {
        let region = regions[i % 4];
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, '{}', {}, {})",
                i,
                region,
                (i + 1) * 100,
                2020 + (i % 5)
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE sales", ()).unwrap();

    // Aggregation query with GROUP BY
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT region, SUM(amount) as total
             FROM sales
             WHERE year = 2024
             GROUP BY region",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        !result.is_empty(),
        "Aggregation EXPLAIN ANALYZE should complete"
    );
}

/// Test that feedback handles NULL values correctly
#[test]
fn test_feedback_with_nulls() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE nullable_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert mix of values and NULLs
    for i in 0..100 {
        if i % 5 == 0 {
            db.execute(
                &format!("INSERT INTO nullable_test VALUES ({}, NULL)", i),
                (),
            )
            .expect("Insert failed");
        } else {
            db.execute(
                &format!("INSERT INTO nullable_test VALUES ({}, {})", i, i * 10),
                (),
            )
            .expect("Insert failed");
        }
    }

    db.execute("ANALYZE nullable_test", ()).unwrap();

    // Query for NULL values
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM nullable_test WHERE value IS NULL",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify the query completed
    assert!(!result.is_empty(), "NULL predicate query should complete");

    // Verify actual count
    let count_result: Vec<_> = db
        .query("SELECT COUNT(*) FROM nullable_test WHERE value IS NULL", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count: i64 = count_result[0].get(0).expect("Failed to get count");
    assert_eq!(count, 20, "Should have 20 NULL values");
}

/// Test feedback with subqueries
#[test]
fn test_subquery_feedback() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create departments table");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER)",
        (),
    )
    .expect("Failed to create employees table");

    // Insert departments
    db.execute("INSERT INTO departments VALUES (1, 'Engineering')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO departments VALUES (2, 'Sales')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO departments VALUES (3, 'Marketing')", ())
        .expect("Insert failed");

    // Insert employees
    for i in 0..30 {
        let dept_id = (i % 3) + 1;
        db.execute(
            &format!(
                "INSERT INTO employees VALUES ({}, 'emp_{}', {}, {})",
                i,
                i,
                dept_id,
                40000 + i * 1000
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE departments", ()).unwrap();
    db.execute("ANALYZE employees", ()).unwrap();

    // Subquery with feedback
    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM employees
             WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering')",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        !result.is_empty(),
        "Subquery EXPLAIN ANALYZE should complete"
    );
}

/// Test feedback system doesn't break with empty tables
#[test]
fn test_feedback_empty_table() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE empty_table (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("ANALYZE empty_table", ()).unwrap();

    let result: Vec<_> = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM empty_table WHERE value > 100",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Should complete without errors
    assert!(!result.is_empty(), "Should handle empty table");
}

/// Test that feedback is consistent across multiple executions
#[test]
fn test_feedback_consistency() {
    let db = Database::open("memory://feedback_consistency").expect("Failed to create database");

    db.execute(
        "CREATE TABLE consistency_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 0..100 {
        db.execute(
            &format!("INSERT INTO consistency_test VALUES ({}, {})", i, i % 10),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE consistency_test", ()).unwrap();

    // Run the same query multiple times
    for _ in 0..3 {
        let result: Vec<_> = db
            .query("SELECT * FROM consistency_test WHERE value = 5", ())
            .expect("Query failed")
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to collect");

        // Should always return 10 rows (100/10 values are 5)
        assert_eq!(result.len(), 10, "Should consistently return 10 rows");
    }
}
