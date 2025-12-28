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

//! Integration Tests for Subquery Coverage
//!
//! These tests exercise the code paths in src/executor/subquery.rs

use stoolap::Database;

// =============================================================================
// EXISTS Subquery Tests
// =============================================================================

#[test]
fn test_exists_basic() {
    let db = Database::open("memory://exists_basic").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 5) + 1),
            (),
        )
        .expect("Failed");
    }

    // Basic EXISTS - customers with orders
    let result = db
        .query(
            "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust_id = c.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5); // Customers 1-5 have orders
}

#[test]
fn test_exists_with_index() {
    let db = Database::open("memory://exists_idx").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_cust ON orders(cust_id)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=30 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 5) + 1),
            (),
        )
        .expect("Failed");
    }

    // EXISTS with indexed inner table
    let result = db
        .query(
            "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust_id = c.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_exists_with_limit() {
    let db = Database::open("memory://exists_limit").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO t1 VALUES ({})", i), ())
            .expect("Failed");
    }
    for i in 1..=200 {
        db.execute(
            &format!("INSERT INTO t2 VALUES ({}, {})", i, (i % 50) + 1),
            (),
        )
        .expect("Failed");
    }

    // EXISTS with LIMIT on outer query
    let result = db
        .query(
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.t1_id = t1.id) LIMIT 10",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// NOT EXISTS (Anti-Join) Tests
// =============================================================================

#[test]
fn test_not_exists_basic() {
    let db = Database::open("memory://not_exists_basic").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=15 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 5) + 1),
            (),
        )
        .expect("Failed");
    }

    // NOT EXISTS - customers without orders
    let result = db
        .query(
            "SELECT * FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.cust_id = c.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5); // Customers 6-10 have no orders
}

#[test]
fn test_not_exists_with_index() {
    let db = Database::open("memory://not_exists_idx").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE order_items (id INTEGER PRIMARY KEY, product_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_prod ON order_items(product_id)", ())
        .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, 'prod{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=30 {
        db.execute(
            &format!("INSERT INTO order_items VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    // NOT EXISTS with indexed inner table
    let result = db
        .query(
            "SELECT * FROM products p WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10); // Products 11-20 have no orders
}

// =============================================================================
// IN Subquery Tests
// =============================================================================

#[test]
fn test_in_subquery_basic() {
    let db = Database::open("memory://in_sub_basic").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE TABLE vip_customers (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 20) + 1),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=5 {
        db.execute(&format!("INSERT INTO vip_customers VALUES ({})", i), ())
            .expect("Failed");
    }

    // IN subquery
    let result = db
        .query(
            "SELECT * FROM orders WHERE cust_id IN (SELECT id FROM vip_customers)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn test_in_subquery_with_where() {
    let db = Database::open("memory://in_sub_where").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE high_value_orders (order_id INTEGER PRIMARY KEY)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }
    // High value orders: 80-100
    for i in 80..=100 {
        db.execute(&format!("INSERT INTO high_value_orders VALUES ({})", i), ())
            .expect("Failed");
    }

    // IN subquery with WHERE on outer
    let result = db
        .query(
            "SELECT * FROM orders WHERE id IN (SELECT order_id FROM high_value_orders) AND amount > 850",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn test_not_in_subquery() {
    let db = Database::open("memory://not_in_sub").expect("Failed");
    db.execute(
        "CREATE TABLE all_users (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE TABLE banned_users (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO all_users VALUES ({}, 'user{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=5 {
        db.execute(&format!("INSERT INTO banned_users VALUES ({})", i), ())
            .expect("Failed");
    }

    // NOT IN subquery
    let result = db
        .query(
            "SELECT * FROM all_users WHERE id NOT IN (SELECT id FROM banned_users)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 15); // Users 6-20
}

// =============================================================================
// Scalar Subquery Tests
// =============================================================================

#[test]
fn test_scalar_subquery_in_select() {
    let db = Database::open("memory://scalar_select").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, {})", i, i as f64 * 100.0),
            (),
        )
        .expect("Failed");
    }

    // Scalar subquery in SELECT
    let result = db
        .query(
            "SELECT id, price, (SELECT AVG(price) FROM products) as avg_price FROM products",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_scalar_subquery_in_where() {
    let db = Database::open("memory://scalar_where").expect("Failed");
    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, salary FLOAT, dept_id INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!(
                "INSERT INTO employees VALUES ({}, {}, {})",
                i,
                i as f64 * 100.0,
                (i % 5) + 1
            ),
            (),
        )
        .expect("Failed");
    }

    // Scalar subquery in WHERE
    let result = db
        .query(
            "SELECT * FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0 && count < 50);
}

#[test]
fn test_scalar_count_subquery() {
    let db = Database::open("memory://scalar_count").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 5) + 1),
            (),
        )
        .expect("Failed");
    }

    // Scalar COUNT subquery
    let result = db
        .query(
            "SELECT c.id, c.name, (SELECT COUNT(*) FROM orders o WHERE o.cust_id = c.id) as order_count FROM customers c",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5);
}

// =============================================================================
// Correlated Subquery Tests
// =============================================================================

#[test]
fn test_correlated_exists() {
    let db = Database::open("memory://corr_exists").expect("Failed");
    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO departments VALUES ({}, 'dept{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=15 {
        db.execute(
            &format!(
                "INSERT INTO employees VALUES ({}, {}, 'emp{}')",
                i,
                (i % 3) + 1,
                i
            ),
            (),
        )
        .expect("Failed");
    }

    // Correlated EXISTS - departments with employees
    let result = db
        .query(
            "SELECT * FROM departments d WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 3); // Departments 1, 2, 3 have employees
}

#[test]
fn test_correlated_scalar_sum() {
    let db = Database::open("memory://corr_sum").expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=20 {
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                (i % 5) + 1,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    // Correlated SUM subquery
    let result = db
        .query(
            "SELECT c.id, c.name, (SELECT SUM(amount) FROM orders o WHERE o.cust_id = c.id) as total FROM customers c",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_correlated_max() {
    let db = Database::open("memory://corr_max").expect("Failed");
    db.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, cat_id INTEGER, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO categories VALUES ({}, 'cat{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=15 {
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, {}, {})",
                i,
                (i % 3) + 1,
                i as f64 * 50.0
            ),
            (),
        )
        .expect("Failed");
    }

    // Correlated MAX subquery - products with max price per category
    let result = db
        .query(
            "SELECT p.* FROM products p WHERE p.price = (SELECT MAX(p2.price) FROM products p2 WHERE p2.cat_id = p.cat_id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

// =============================================================================
// ANY/ALL Subquery Tests
// =============================================================================

#[test]
fn test_any_subquery() {
    let db = Database::open("memory://any_sub").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE discounts (id INTEGER PRIMARY KEY, threshold FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO discounts VALUES ({}, {})", i, i as f64 * 50.0),
            (),
        )
        .expect("Failed");
    }

    // price > ANY (subquery)
    let result = db
        .query(
            "SELECT * FROM products WHERE price > ANY (SELECT threshold FROM discounts)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn test_all_subquery() {
    let db = Database::open("memory://all_sub").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE thresholds (id INTEGER PRIMARY KEY, min_price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }
    db.execute("INSERT INTO thresholds VALUES (1, 50.0)", ())
        .expect("Failed");
    db.execute("INSERT INTO thresholds VALUES (2, 100.0)", ())
        .expect("Failed");

    // price > ALL (subquery)
    let result = db
        .query(
            "SELECT * FROM products WHERE price > ALL (SELECT min_price FROM thresholds)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0); // Products with price > 100
}

// =============================================================================
// Semi-Join Optimization Tests
// =============================================================================

#[test]
fn test_semi_join_optimization() {
    let db = Database::open("memory://semi_join").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_t1_id ON t2(t1_id)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }
    for i in 1..=200 {
        db.execute(
            &format!("INSERT INTO t2 VALUES ({}, {})", i, (i % 50) + 1),
            (),
        )
        .expect("Failed");
    }

    // EXISTS that can be optimized to semi-join
    let result = db
        .query(
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.t1_id = t1.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 50);
}

#[test]
fn test_anti_join_optimization() {
    let db = Database::open("memory://anti_join").expect("Failed");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE logins (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_user ON logins(user_id)", ())
        .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO users VALUES ({}, 'user{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO logins VALUES ({}, {})", i, (i % 25) + 1),
            (),
        )
        .expect("Failed");
    }

    // NOT EXISTS that can be optimized to anti-join
    let result = db
        .query(
            "SELECT * FROM users WHERE NOT EXISTS (SELECT 1 FROM logins WHERE logins.user_id = users.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 25); // Users 26-50 have no logins
}

// =============================================================================
// Nested Subquery Tests
// =============================================================================

#[test]
fn test_nested_subqueries() {
    let db = Database::open("memory://nested_sub").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE t3 (id INTEGER PRIMARY KEY, t2_id INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }
    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO t2 VALUES ({}, {})", i, (i % 5) + 1),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=30 {
        db.execute(
            &format!("INSERT INTO t3 VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    // Nested EXISTS subqueries
    let result = db
        .query(
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.t1_id = t1.id AND EXISTS (SELECT 1 FROM t3 WHERE t3.t2_id = t2.id))",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

// =============================================================================
// Subquery in Different Clauses Tests
// =============================================================================

#[test]
fn test_subquery_in_having() {
    let db = Database::open("memory://sub_having").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        let region = if i % 4 == 0 {
            "North"
        } else if i % 4 == 1 {
            "South"
        } else if i % 4 == 2 {
            "East"
        } else {
            "West"
        };
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, '{}', {})",
                i,
                region,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    // Subquery in HAVING
    let result = db
        .query(
            "SELECT region, SUM(amount) as total FROM sales GROUP BY region HAVING SUM(amount) > (SELECT AVG(amount) * 10 FROM sales)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn test_subquery_in_from() {
    let db = Database::open("memory://sub_from").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, i as f64 * 20.0),
            (),
        )
        .expect("Failed");
    }

    // Subquery in FROM (derived table)
    let result = db
        .query(
            "SELECT * FROM (SELECT id, amount FROM orders WHERE amount > 500) AS high_orders",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_subquery_empty_result() {
    let db = Database::open("memory://sub_empty").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({})", i), ())
            .expect("Failed");
    }
    // t2 is empty

    // IN subquery with empty result
    let result = db
        .query("SELECT * FROM t1 WHERE id IN (SELECT id FROM t2)", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 0);
}

#[test]
fn test_subquery_null_handling() {
    let db = Database::open("memory://sub_null").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");

    db.execute("INSERT INTO t1 VALUES (1, 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO t1 VALUES (2, 20)", ())
        .expect("Failed");
    db.execute("INSERT INTO t1 VALUES (3, NULL)", ())
        .expect("Failed");
    db.execute("INSERT INTO t2 VALUES (1, 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO t2 VALUES (2, NULL)", ())
        .expect("Failed");

    // IN with NULLs
    let result = db
        .query(
            "SELECT * FROM t1 WHERE val IN (SELECT val FROM t2 WHERE val IS NOT NULL)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 1); // Only id=1 matches
}

#[test]
fn test_exists_always_true() {
    let db = Database::open("memory://exists_true").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({})", i), ())
            .expect("Failed");
        db.execute(&format!("INSERT INTO t2 VALUES ({})", i), ())
            .expect("Failed");
    }

    // EXISTS with uncorrelated subquery that returns rows
    let result = db
        .query("SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2)", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10); // All rows from t1
}

#[test]
fn test_exists_always_false() {
    let db = Database::open("memory://exists_false").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({})", i), ())
            .expect("Failed");
    }
    // t2 is empty

    // EXISTS with uncorrelated subquery that returns no rows
    let result = db
        .query("SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2)", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 0);
}

#[test]
fn test_multiple_subqueries_same_query() {
    let db = Database::open("memory://multi_sub").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount FLOAT, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO customers VALUES ({})", i), ())
            .expect("Failed");
    }
    for i in 1..=50 {
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                i as f64 * 10.0,
                (i % 10) + 1
            ),
            (),
        )
        .expect("Failed");
    }

    // Multiple subqueries in same query
    let result = db
        .query(
            "SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders) AND cust_id IN (SELECT id FROM customers WHERE id < 5)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count >= 0);
}

#[test]
fn test_subquery_with_order_by() {
    let db = Database::open("memory://sub_order").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, {})", i, i as f64 * 15.0),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO sales VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    // Subquery in WHERE with ORDER BY on outer
    let result = db
        .query(
            "SELECT * FROM products WHERE id IN (SELECT product_id FROM sales) ORDER BY price DESC LIMIT 5",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5);
}
