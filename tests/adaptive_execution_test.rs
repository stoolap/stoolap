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

//! Integration tests for Adaptive Query Execution (AQE)
//!
//! AQE enables runtime adaptation of query plans based on actual cardinalities
//! observed during execution. This helps handle cases where estimated cardinalities
//! are significantly different from actual cardinalities.

use stoolap::api::Database;

/// Test that AQE doesn't break basic join queries
#[test]
fn test_aqe_basic_join() {
    let db = Database::open("memory://aqe_basic").expect("Failed to create database");

    db.execute(
        "CREATE TABLE left_table (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create left table");

    db.execute(
        "CREATE TABLE right_table (id INTEGER PRIMARY KEY, left_id INTEGER, data TEXT)",
        (),
    )
    .expect("Failed to create right table");

    // Insert data
    for i in 0..50 {
        db.execute(
            &format!("INSERT INTO left_table VALUES ({}, 'left_{}')", i, i),
            (),
        )
        .expect("Insert failed");
    }

    for i in 0..100 {
        db.execute(
            &format!(
                "INSERT INTO right_table VALUES ({}, {}, 'right_{}')",
                i,
                i % 50,
                i
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE left_table", ()).unwrap();
    db.execute("ANALYZE right_table", ()).unwrap();

    // Execute join - AQE may adapt the plan
    let result: Vec<_> = db
        .query(
            "SELECT l.value, r.data
             FROM left_table l
             INNER JOIN right_table r ON l.id = r.left_id
             ORDER BY l.id, r.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Each left row joins with 2 right rows (100 right rows / 50 unique left_ids)
    assert_eq!(result.len(), 100, "Should return 100 rows from join");
}

/// Test AQE with heavily skewed data (estimates likely wrong)
#[test]
fn test_aqe_skewed_data() {
    let db = Database::open("memory://aqe_skewed").expect("Failed to create database");

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, region TEXT)",
        (),
    )
    .expect("Failed to create customers table");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total INTEGER)",
        (),
    )
    .expect("Failed to create orders table");

    // Insert customers
    for i in 0..100 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'region_{}')", i, i % 10),
            (),
        )
        .expect("Insert failed");
    }

    // Skewed orders - customer 0 has most orders
    let mut order_id = 0;
    for customer_id in 0..100 {
        // Customer 0 has 100 orders, others have 1-2
        let num_orders = if customer_id == 0 {
            100
        } else {
            1 + (customer_id % 2)
        };
        for _ in 0..num_orders {
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({}, {}, {})",
                    order_id,
                    customer_id,
                    (order_id + 1) * 10
                ),
                (),
            )
            .expect("Insert failed");
            order_id += 1;
        }
    }

    db.execute("ANALYZE customers", ()).unwrap();
    db.execute("ANALYZE orders", ()).unwrap();

    // Query for customer 0's orders (should have 100)
    let result: Vec<_> = db
        .query(
            "SELECT c.region, o.total
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             WHERE c.id = 0
             ORDER BY o.total",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 100, "Should return 100 orders for customer 0");
}

/// Test AQE with LEFT JOIN
#[test]
fn test_aqe_left_join() {
    let db = Database::open("memory://aqe_left").expect("Failed to create database");

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

    // Insert products
    for i in 0..50 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, 'product_{}')", i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Only some products have reviews
    for (review_id, product_id) in (0..25).enumerate() {
        // Only first 25 products have reviews
        db.execute(
            &format!(
                "INSERT INTO reviews VALUES ({}, {}, {})",
                review_id,
                product_id,
                3 + (review_id % 3)
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE products", ()).unwrap();
    db.execute("ANALYZE reviews", ()).unwrap();

    // LEFT JOIN should include products without reviews
    let result: Vec<_> = db
        .query(
            "SELECT p.name, r.rating
             FROM products p
             LEFT JOIN reviews r ON p.id = r.product_id
             ORDER BY p.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Should have all 50 products
    assert_eq!(
        result.len(),
        50,
        "Should return all 50 products in LEFT JOIN"
    );

    // Count products with and without reviews
    let mut with_reviews = 0;
    let mut without_reviews = 0;
    for row in &result {
        let rating: Result<i64, _> = row.get(1);
        if rating.is_ok() {
            with_reviews += 1;
        } else {
            without_reviews += 1;
        }
    }

    assert_eq!(with_reviews, 25, "25 products should have reviews");
    assert_eq!(without_reviews, 25, "25 products should not have reviews");
}

/// Test AQE with multiple joins
#[test]
fn test_aqe_multi_join() {
    let db = Database::open("memory://aqe_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create categories table");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER)",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE inventory (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create inventory table");

    // Insert categories
    for i in 0..5 {
        db.execute(
            &format!("INSERT INTO categories VALUES ({}, 'category_{}')", i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Insert products
    for i in 0..50 {
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, 'product_{}', {})",
                i,
                i,
                i % 5
            ),
            (),
        )
        .expect("Insert failed");
    }

    // Insert inventory
    for i in 0..50 {
        db.execute(
            &format!(
                "INSERT INTO inventory VALUES ({}, {}, {})",
                i,
                i,
                (i + 1) * 10
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE categories", ()).unwrap();
    db.execute("ANALYZE products", ()).unwrap();
    db.execute("ANALYZE inventory", ()).unwrap();

    // Three-way join
    let result: Vec<_> = db
        .query(
            "SELECT c.name as category, p.name as product, i.quantity
             FROM categories c
             INNER JOIN products p ON c.id = p.category_id
             INNER JOIN inventory i ON p.id = i.product_id
             WHERE i.quantity > 200
             ORDER BY c.id, p.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // All rows should have quantity > 200
    for row in &result {
        let quantity: i64 = row.get(2).expect("Failed to get quantity");
        assert!(quantity > 200, "Quantity should be > 200");
    }
}

/// Test AQE with filter after join
#[test]
fn test_aqe_join_with_filter() {
    let db = Database::open("memory://aqe_filter").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)",
        (),
    )
    .expect("Failed to create employees table");

    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER)",
        (),
    )
    .expect("Failed to create departments table");

    // Insert departments
    for i in 0..10 {
        db.execute(
            &format!(
                "INSERT INTO departments VALUES ({}, 'dept_{}', {})",
                i,
                i,
                100000 + i * 50000
            ),
            (),
        )
        .expect("Insert failed");
    }

    // Insert employees
    for i in 0..100 {
        db.execute(
            &format!(
                "INSERT INTO employees VALUES ({}, 'emp_{}', {})",
                i,
                i,
                i % 10
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE employees", ()).unwrap();
    db.execute("ANALYZE departments", ()).unwrap();

    // Join with filter on joined table
    let result: Vec<_> = db
        .query(
            "SELECT e.name, d.name as dept, d.budget
             FROM employees e
             INNER JOIN departments d ON e.dept_id = d.id
             WHERE d.budget > 300000
             ORDER BY e.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Departments 5-9 have budget > 300000 (350k-550k), each has 10 employees
    assert_eq!(
        result.len(),
        50,
        "Should return 50 employees in high-budget departments"
    );
}

/// Test AQE with CROSS JOIN (nested loop only)
#[test]
fn test_aqe_cross_join() {
    let db = Database::open("memory://aqe_cross").expect("Failed to create database");

    db.execute("CREATE TABLE sizes (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create sizes table");

    db.execute(
        "CREATE TABLE colors (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create colors table");

    // Insert sizes
    db.execute(
        "INSERT INTO sizes VALUES (1, 'Small'), (2, 'Medium'), (3, 'Large')",
        (),
    )
    .expect("Insert failed");

    // Insert colors
    db.execute(
        "INSERT INTO colors VALUES (1, 'Red'), (2, 'Blue'), (3, 'Green'), (4, 'Yellow')",
        (),
    )
    .expect("Insert failed");

    // CROSS JOIN should use nested loop
    let result: Vec<_> = db
        .query(
            "SELECT s.name as size, c.name as color
             FROM sizes s
             CROSS JOIN colors c
             ORDER BY s.id, c.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // 3 sizes * 4 colors = 12 combinations
    assert_eq!(result.len(), 12, "CROSS JOIN should produce 12 rows");
}

/// Test AQE with large cardinality difference
#[test]
fn test_aqe_large_small_join() {
    let db = Database::open("memory://aqe_large_small").expect("Failed to create database");

    // Small lookup table
    db.execute(
        "CREATE TABLE status_codes (id INTEGER PRIMARY KEY, description TEXT)",
        (),
    )
    .expect("Failed to create status_codes table");

    // Large fact table
    db.execute(
        "CREATE TABLE transactions (id INTEGER PRIMARY KEY, status_id INTEGER, amount INTEGER)",
        (),
    )
    .expect("Failed to create transactions table");

    // Insert status codes (small table)
    db.execute("INSERT INTO status_codes VALUES (1, 'Pending')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO status_codes VALUES (2, 'Completed')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO status_codes VALUES (3, 'Failed')", ())
        .expect("Insert failed");

    // Insert transactions (large table)
    for i in 0..1000 {
        let status_id = (i % 3) + 1;
        db.execute(
            &format!(
                "INSERT INTO transactions VALUES ({}, {}, {})",
                i,
                status_id,
                (i + 1) * 10
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE status_codes", ()).unwrap();
    db.execute("ANALYZE transactions", ()).unwrap();

    // Join large table with small lookup
    let result: Vec<_> = db
        .query(
            "SELECT s.description, COUNT(*) as count
             FROM transactions t
             INNER JOIN status_codes s ON t.status_id = s.id
             GROUP BY s.description
             ORDER BY s.description",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 3, "Should have 3 status groups");

    // Each status should have ~333 transactions
    for row in &result {
        let count: i64 = row.get(1).expect("Failed to get count");
        assert!(
            (333..=334).contains(&count),
            "Each status should have ~333-334 transactions"
        );
    }
}

/// Test AQE with self-join
#[test]
fn test_aqe_self_join() {
    let db = Database::open("memory://aqe_self").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)",
        (),
    )
    .expect("Failed to create employees table");

    // Insert employees with manager relationships
    db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO employees VALUES (2, 'VP1', 1)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO employees VALUES (3, 'VP2', 1)", ())
        .expect("Insert failed");
    for i in 4..20 {
        let manager = if i < 10 { 2 } else { 3 };
        db.execute(
            &format!(
                "INSERT INTO employees VALUES ({}, 'Employee{}', {})",
                i, i, manager
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE employees", ()).unwrap();

    // Self-join to get employee with manager name
    let result: Vec<_> = db
        .query(
            "SELECT e.name as employee, m.name as manager
             FROM employees e
             INNER JOIN employees m ON e.manager_id = m.id
             ORDER BY e.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // All employees except CEO have managers
    assert_eq!(result.len(), 18, "18 employees should have managers");
}

/// Test AQE with aggregation after join
#[test]
fn test_aqe_join_aggregate() {
    let db = Database::open("memory://aqe_agg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE stores (id INTEGER PRIMARY KEY, name TEXT, region TEXT)",
        (),
    )
    .expect("Failed to create stores table");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, store_id INTEGER, amount INTEGER, date INTEGER)",
        (),
    )
    .expect("Failed to create sales table");

    // Insert stores
    for i in 0..10 {
        let region = match i % 3 {
            0 => "North",
            1 => "South",
            _ => "West",
        };
        db.execute(
            &format!(
                "INSERT INTO stores VALUES ({}, 'Store{}', '{}')",
                i, i, region
            ),
            (),
        )
        .expect("Insert failed");
    }

    // Insert sales
    for i in 0..500 {
        let store_id = i % 10;
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, {}, {}, {})",
                i,
                store_id,
                (i + 1) * 100,
                i
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE stores", ()).unwrap();
    db.execute("ANALYZE sales", ()).unwrap();

    // Join with aggregation
    let result: Vec<_> = db
        .query(
            "SELECT st.region, SUM(sa.amount) as total_sales
             FROM stores st
             INNER JOIN sales sa ON st.id = sa.store_id
             GROUP BY st.region
             ORDER BY st.region",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 3, "Should have 3 regions");
}

/// Test AQE doesn't affect correctness with different data distributions
#[test]
fn test_aqe_correctness_uniform() {
    let db = Database::open("memory://aqe_uniform").expect("Failed to create database");

    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, value INTEGER)", ())
        .expect("Failed to create table a");

    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER, data TEXT)",
        (),
    )
    .expect("Failed to create table b");

    // Uniform distribution
    for i in 0..100 {
        db.execute(&format!("INSERT INTO a VALUES ({}, {})", i, i * 10), ())
            .expect("Insert failed");
    }

    for i in 0..200 {
        db.execute(
            &format!("INSERT INTO b VALUES ({}, {}, 'data_{}')", i, i % 100, i),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE a", ()).unwrap();
    db.execute("ANALYZE b", ()).unwrap();

    // Verify join correctness
    let result: Vec<_> = db
        .query(
            "SELECT a.value, b.data
             FROM a
             INNER JOIN b ON a.id = b.a_id
             WHERE a.value BETWEEN 100 AND 200
             ORDER BY a.id, b.id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // IDs 10-20 have values 100-200, each joins with 2 rows from b
    assert_eq!(result.len(), 22, "Should return 22 rows"); // 11 values * 2 matches each
}

/// Test AQE handles empty join results
#[test]
fn test_aqe_empty_join() {
    let db = Database::open("memory://aqe_empty").expect("Failed to create database");

    db.execute("CREATE TABLE x (id INTEGER PRIMARY KEY, value INTEGER)", ())
        .expect("Failed to create table x");

    db.execute("CREATE TABLE y (id INTEGER PRIMARY KEY, x_id INTEGER)", ())
        .expect("Failed to create table y");

    // Insert into x
    for i in 0..50 {
        db.execute(&format!("INSERT INTO x VALUES ({}, {})", i, i), ())
            .expect("Insert failed");
    }

    // Insert into y with non-overlapping IDs
    for i in 100..150 {
        db.execute(&format!("INSERT INTO y VALUES ({}, {})", i, i), ())
            .expect("Insert failed");
    }

    db.execute("ANALYZE x", ()).unwrap();
    db.execute("ANALYZE y", ()).unwrap();

    // Join should produce empty result
    let result: Vec<_> = db
        .query(
            "SELECT x.value, y.id FROM x INNER JOIN y ON x.id = y.x_id",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 0, "Join with no matches should return empty");
}
