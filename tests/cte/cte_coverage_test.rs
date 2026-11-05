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

//! Integration Tests for CTE Coverage
//!
//! These tests exercise the code paths in src/executor/cte.rs

use stoolap::Database;

// =============================================================================
// Basic CTE Tests
// =============================================================================

#[test]
fn test_simple_cte() {
    let db = Database::open("memory://simple_cte").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, 'prod{}', {})",
                i,
                i,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH expensive AS (SELECT * FROM products WHERE price > 50)
             SELECT * FROM expensive",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5); // products with price > 50
}

#[test]
fn test_cte_with_alias() {
    let db = Database::open("memory://cte_alias").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, i as f64 * 5.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH high_orders AS (SELECT id, amount FROM orders WHERE amount > 50)
             SELECT ho.id, ho.amount FROM high_orders ho",
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
fn test_cte_with_column_names() {
    let db = Database::open("memory://cte_cols").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO data VALUES ({}, {})", i, i * 10), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "WITH cte_data(a, b) AS (SELECT id, value FROM data)
             SELECT a, b FROM cte_data",
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
// Multiple CTEs Tests
// =============================================================================

#[test]
fn test_multiple_ctes() {
    let db = Database::open("memory://multi_cte").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=30 {
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, {}, {})",
                i,
                (i % 5) + 1,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH high_sales AS (SELECT * FROM sales WHERE amount > 100),
                  low_sales AS (SELECT * FROM sales WHERE amount <= 100)
             SELECT COUNT(*) FROM high_sales",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn test_cte_references_another_cte() {
    let db = Database::open("memory://cte_chain").expect("Failed");
    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(&format!("INSERT INTO numbers VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "WITH big_nums AS (SELECT * FROM numbers WHERE val > 10),
                  even_big AS (SELECT * FROM big_nums WHERE val % 2 = 0)
             SELECT * FROM even_big",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5); // 12, 14, 16, 18, 20
}

// =============================================================================
// CTE with Aggregation Tests
// =============================================================================

#[test]
fn test_cte_with_group_by() {
    let db = Database::open("memory://cte_group").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                (i % 10) + 1,
                i as f64 * 5.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH customer_totals AS (
                SELECT customer_id, SUM(amount) as total
                FROM orders
                GROUP BY customer_id
             )
             SELECT * FROM customer_totals ORDER BY total DESC",
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
fn test_cte_with_having() {
    let db = Database::open("memory://cte_having").expect("Failed");
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

    let result = db
        .query(
            "WITH region_totals AS (
                SELECT region, SUM(amount) as total
                FROM sales
                GROUP BY region
                HAVING SUM(amount) > 5000
             )
             SELECT * FROM region_totals",
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
// CTE with Joins Tests
// =============================================================================

#[test]
fn test_cte_with_join() {
    let db = Database::open("memory://cte_join").expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
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
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                (i % 10) + 1,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH high_orders AS (SELECT * FROM orders WHERE amount > 150)
             SELECT c.name, ho.amount
             FROM customers c
             JOIN high_orders ho ON c.id = ho.customer_id",
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
fn test_cte_join_with_cte() {
    let db = Database::open("memory://cte_join_cte").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i * 10), ())
            .expect("Failed");
    }
    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO t2 VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH cte1 AS (SELECT * FROM t1 WHERE val > 50),
                  cte2 AS (SELECT * FROM t2)
             SELECT cte1.id, cte2.id FROM cte1 JOIN cte2 ON cte1.id = cte2.t1_id",
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
// Recursive CTE Tests
// =============================================================================

#[test]
fn test_recursive_cte_basic() {
    let db = Database::open("memory://rec_cte").expect("Failed");

    // Generate numbers 1-10 recursively
    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x+1 FROM cnt WHERE x < 10
             )
             SELECT x FROM cnt",
            (),
        )
        .expect("Query failed");

    let mut numbers = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let x: i64 = row.get(0).unwrap();
        numbers.push(x);
    }
    assert_eq!(numbers.len(), 10);
    assert_eq!(numbers[0], 1);
    assert_eq!(numbers[9], 10);
}

#[test]
fn test_recursive_cte_fibonacci() {
    let db = Database::open("memory://rec_fib").expect("Failed");

    // Generate Fibonacci sequence
    let result = db
        .query(
            "WITH RECURSIVE fib(n, a, b) AS (
                SELECT 1, 0, 1
                UNION ALL
                SELECT n+1, b, a+b FROM fib WHERE n < 10
             )
             SELECT n, a FROM fib",
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
fn test_recursive_cte_with_table() {
    let db = Database::open("memory://rec_table").expect("Failed");
    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)", ())
        .expect("Failed");
    db.execute("INSERT INTO employees VALUES (2, 'VP1', 1)", ())
        .expect("Failed");
    db.execute("INSERT INTO employees VALUES (3, 'VP2', 1)", ())
        .expect("Failed");
    db.execute("INSERT INTO employees VALUES (4, 'Mgr1', 2)", ())
        .expect("Failed");
    db.execute("INSERT INTO employees VALUES (5, 'Mgr2', 2)", ())
        .expect("Failed");

    // Get all subordinates of CEO
    let result = db
        .query(
            "WITH RECURSIVE subordinates AS (
                SELECT id, name, manager_id FROM employees WHERE id = 1
                UNION ALL
                SELECT e.id, e.name, e.manager_id
                FROM employees e
                JOIN subordinates s ON e.manager_id = s.id
             )
             SELECT * FROM subordinates",
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
// CTE with Subqueries Tests
// =============================================================================

#[test]
fn test_cte_in_subquery() {
    let db = Database::open("memory://cte_subq").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
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

    let result = db
        .query(
            "WITH avg_price AS (SELECT AVG(price) as avg FROM products)
             SELECT * FROM products WHERE price > (SELECT avg FROM avg_price)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count > 0 && count < 20);
}

#[test]
fn test_cte_with_exists() {
    let db = Database::open("memory://cte_exists").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER)",
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

    let result = db
        .query(
            "WITH active_customers AS (
                SELECT DISTINCT customer_id FROM orders
             )
             SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM active_customers ac WHERE ac.customer_id = c.id)",
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
// CTE with ORDER BY and LIMIT Tests
// =============================================================================

#[test]
fn test_cte_with_order_by() {
    let db = Database::open("memory://cte_order").expect("Failed");
    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO scores VALUES ({}, {})", i, (21 - i) * 5),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH ranked AS (SELECT * FROM scores ORDER BY score DESC)
             SELECT * FROM ranked LIMIT 5",
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
fn test_cte_with_limit_in_definition() {
    let db = Database::open("memory://cte_limit").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO data VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "WITH top10 AS (SELECT * FROM data ORDER BY val DESC LIMIT 10)
             SELECT * FROM top10",
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
// CTE with DISTINCT Tests
// =============================================================================

#[test]
fn test_cte_with_distinct() {
    let db = Database::open("memory://cte_distinct").expect("Failed");
    db.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY, user_id INTEGER, action TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=50 {
        db.execute(
            &format!(
                "INSERT INTO logs VALUES ({}, {}, 'action{}')",
                i,
                (i % 10) + 1,
                i % 5
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH unique_users AS (SELECT DISTINCT user_id FROM logs)
             SELECT * FROM unique_users",
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
// CTE Edge Cases
// =============================================================================

#[test]
fn test_cte_empty_result() {
    let db = Database::open("memory://cte_empty").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "WITH empty AS (SELECT * FROM t1 WHERE val > 100)
             SELECT * FROM empty",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 0);
}

#[test]
fn test_cte_used_multiple_times() {
    let db = Database::open("memory://cte_multi_use").expect("Failed");
    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO numbers VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH nums AS (SELECT * FROM numbers)
             SELECT a.id, b.val FROM nums a, nums b WHERE a.id = b.id",
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
fn test_cte_with_nulls() {
    let db = Database::open("memory://cte_nulls").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO data VALUES (1, 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (2, NULL)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (3, 30)", ())
        .expect("Failed");

    let result = db
        .query(
            "WITH filtered AS (SELECT * FROM data WHERE val IS NOT NULL)
             SELECT * FROM filtered",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn test_cte_with_union() {
    let db = Database::open("memory://cte_union").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");

    for i in 1..=5 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }
    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO t2 VALUES ({}, {})", i + 10, i + 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH combined AS (
                SELECT id, val FROM t1
                UNION ALL
                SELECT id, val FROM t2
             )
             SELECT * FROM combined",
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
fn test_cte_with_case_expression() {
    let db = Database::open("memory://cte_case").expect("Failed");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, i as f64 * 25.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "WITH categorized AS (
                SELECT id, amount,
                       CASE WHEN amount > 300 THEN 'high'
                            WHEN amount > 150 THEN 'medium'
                            ELSE 'low' END as category
                FROM orders
             )
             SELECT category, COUNT(*) FROM categorized GROUP BY category",
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
fn test_cte_single_row() {
    let db = Database::open("memory://cte_single").expect("Failed");

    let result = db
        .query(
            "WITH single AS (SELECT 1 as val)
             SELECT val FROM single",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed");
        let val: i64 = row.get(0).unwrap();
        assert_eq!(val, 1);
        count += 1;
    }
    assert_eq!(count, 1);
}
