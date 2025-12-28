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

//! Query Executor Tests
//!
//! Tests for the main query executor including SELECT variations,
//! JOIN execution, ORDER BY, DISTINCT, LIMIT/OFFSET, and expression evaluation.

use stoolap::Database;

fn setup_tables(db: &Database) {
    // Customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            country TEXT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product TEXT,
            amount FLOAT,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Create indexes
    db.execute(
        "CREATE INDEX idx_orders_customer ON orders(customer_id)",
        (),
    )
    .expect("Failed to create index");
    db.execute("CREATE INDEX idx_customers_city ON customers(city)", ())
        .expect("Failed to create index");

    // Insert customers
    let customers = [
        "INSERT INTO customers VALUES (1, 'Alice', 'New York', 'USA')",
        "INSERT INTO customers VALUES (2, 'Bob', 'Los Angeles', 'USA')",
        "INSERT INTO customers VALUES (3, 'Charlie', 'London', 'UK')",
        "INSERT INTO customers VALUES (4, 'Diana', 'Paris', 'France')",
        "INSERT INTO customers VALUES (5, 'Eve', 'Berlin', 'Germany')",
    ];

    for sql in &customers {
        db.execute(sql, ()).expect("Failed to insert customer");
    }

    // Insert orders
    let orders = [
        "INSERT INTO orders VALUES (1, 1, 'Laptop', 999.99, 'completed')",
        "INSERT INTO orders VALUES (2, 1, 'Phone', 499.99, 'completed')",
        "INSERT INTO orders VALUES (3, 2, 'Tablet', 299.99, 'pending')",
        "INSERT INTO orders VALUES (4, 3, 'Laptop', 899.99, 'completed')",
        "INSERT INTO orders VALUES (5, 3, 'Phone', 599.99, 'shipped')",
        "INSERT INTO orders VALUES (6, 4, 'Monitor', 399.99, 'completed')",
    ];

    for sql in &orders {
        db.execute(sql, ()).expect("Failed to insert order");
    }
}

// =============================================================================
// Basic SELECT Tests
// =============================================================================

#[test]
fn test_select_all_columns() {
    let db = Database::open("memory://select_all").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM customers", ())
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert_eq!(columns.len(), 4, "Expected 4 columns");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }
    assert_eq!(row_count, 5, "Expected 5 customers");
}

#[test]
fn test_select_specific_columns() {
    let db = Database::open("memory://select_specific").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT name, city FROM customers", ())
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert_eq!(columns.len(), 2, "Expected 2 columns");
    assert_eq!(columns[0], "name");
    assert_eq!(columns[1], "city");
}

#[test]
fn test_select_with_expressions() {
    let db = Database::open("memory://select_expr").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT id, amount, amount * 1.1 as with_tax FROM orders",
            (),
        )
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert!(
        columns.contains(&"with_tax".to_string()),
        "Expected with_tax column"
    );
}

#[test]
fn test_select_constant() {
    let db = Database::open("memory://select_const").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT 1 as one, 'hello' as greeting, id FROM customers",
            (),
        )
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert!(columns.contains(&"one".to_string()), "Expected one column");
    assert!(
        columns.contains(&"greeting".to_string()),
        "Expected greeting column"
    );
}

// =============================================================================
// WHERE Clause Tests
// =============================================================================

#[test]
fn test_where_equality() {
    let db = Database::open("memory://where_eq").expect("Failed to create database");
    setup_tables(&db);

    let name: String = db
        .query_one("SELECT name FROM customers WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(name, "Alice");
}

#[test]
fn test_where_comparison() {
    let db = Database::open("memory://where_cmp").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM orders WHERE amount > 500", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(3).unwrap();
        assert!(amount > 500.0, "Expected amount > 500");
        count += 1;
    }

    assert!(count > 0, "Expected some orders > 500");
}

#[test]
fn test_where_and() {
    let db = Database::open("memory://where_and").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT * FROM orders WHERE amount > 400 AND status = 'completed'",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(3).unwrap();
        let status: String = row.get(4).unwrap();
        assert!(amount > 400.0, "Expected amount > 400");
        assert_eq!(status, "completed", "Expected completed status");
        count += 1;
    }

    assert!(count > 0, "Expected some matching orders");
}

#[test]
fn test_where_or() {
    let db = Database::open("memory://where_or").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT * FROM orders WHERE status = 'completed' OR status = 'shipped'",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let status: String = row.get(4).unwrap();
        assert!(
            status == "completed" || status == "shipped",
            "Expected completed or shipped"
        );
        count += 1;
    }

    assert!(count > 0, "Expected some matching orders");
}

#[test]
fn test_where_between() {
    let db = Database::open("memory://where_between").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM orders WHERE amount BETWEEN 300 AND 600", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(3).unwrap();
        assert!(
            (300.0..=600.0).contains(&amount),
            "Expected amount in range"
        );
        count += 1;
    }

    assert!(count > 0, "Expected some orders in range");
}

#[test]
fn test_where_like() {
    let db = Database::open("memory://where_like").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM customers WHERE name LIKE 'A%'", ())
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(1).unwrap();
        names.push(name);
    }

    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
}

#[test]
fn test_where_in_list() {
    let db = Database::open("memory://where_in").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM customers WHERE id IN (1, 3, 5)", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    ids.sort();

    assert_eq!(ids, vec![1, 3, 5], "Expected ids 1, 3, 5");
}

#[test]
fn test_where_is_null() {
    let db = Database::open("memory://where_null").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();

    let result = db
        .query("SELECT id FROM t WHERE val IS NULL", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![2], "Expected only id 2 with NULL");
}

#[test]
fn test_where_is_not_null() {
    let db = Database::open("memory://where_not_null").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();

    let result = db
        .query("SELECT id FROM t WHERE val IS NOT NULL ORDER BY id", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![1, 3], "Expected ids 1, 3 with non-NULL");
}

// =============================================================================
// ORDER BY Tests
// =============================================================================

#[test]
fn test_order_by_asc() {
    let db = Database::open("memory://order_asc").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT name FROM customers ORDER BY name ASC", ())
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(
        names,
        vec!["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "Expected alphabetical order"
    );
}

#[test]
fn test_order_by_desc() {
    let db = Database::open("memory://order_desc").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT amount FROM orders ORDER BY amount DESC", ())
        .expect("Failed to query");

    let mut amounts = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(0).unwrap();
        amounts.push(amount);
    }

    // Verify descending order
    for i in 1..amounts.len() {
        assert!(amounts[i] <= amounts[i - 1], "Expected descending order");
    }
}

#[test]
fn test_order_by_multiple_columns() {
    let db = Database::open("memory://order_multi").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT customer_id, amount FROM orders ORDER BY customer_id ASC, amount DESC",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 6, "Expected all orders");
}

#[test]
fn test_order_by_expression() {
    let db = Database::open("memory://order_expr").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT id, amount FROM orders ORDER BY amount * 2 DESC", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 6, "Expected all orders");
}

#[test]
fn test_order_by_nulls_first() {
    let db = Database::open("memory://order_nulls_first").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();

    let result = db
        .query("SELECT id FROM t ORDER BY val NULLS FIRST", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids[0], 2, "Expected NULL first");
}

#[test]
fn test_order_by_nulls_last() {
    let db = Database::open("memory://order_nulls_last").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();

    let result = db
        .query("SELECT id FROM t ORDER BY val NULLS LAST", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids[2], 2, "Expected NULL last");
}

// =============================================================================
// LIMIT / OFFSET Tests
// =============================================================================

#[test]
fn test_limit() {
    let db = Database::open("memory://limit").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM customers LIMIT 3", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3, "Expected exactly 3 rows");
}

#[test]
fn test_offset() {
    let db = Database::open("memory://offset").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT id FROM customers ORDER BY id LIMIT 2 OFFSET 2", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![3, 4], "Expected ids 3, 4 after offset 2");
}

#[test]
fn test_limit_larger_than_result() {
    let db = Database::open("memory://limit_large").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT * FROM customers LIMIT 100", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 5, "Expected all 5 customers");
}

// =============================================================================
// DISTINCT Tests
// =============================================================================

#[test]
fn test_distinct_single_column() {
    let db = Database::open("memory://distinct_single").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT DISTINCT status FROM orders", ())
        .expect("Failed to query");

    let mut statuses = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let status: String = row.get(0).unwrap();
        statuses.push(status);
    }

    // Should be unique: completed, pending, shipped
    assert_eq!(statuses.len(), 3, "Expected 3 distinct statuses");
}

#[test]
fn test_distinct_multiple_columns() {
    let db = Database::open("memory://distinct_multi").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query("SELECT DISTINCT customer_id, status FROM orders", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert!(count <= 6, "Expected at most 6 distinct combinations");
}

#[test]
fn test_distinct_with_order_by() {
    let db = Database::open("memory://distinct_order").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT DISTINCT country FROM customers ORDER BY country",
            (),
        )
        .expect("Failed to query");

    let mut countries = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let country: String = row.get(0).unwrap();
        countries.push(country);
    }

    // Should be sorted
    let mut sorted = countries.clone();
    sorted.sort();
    assert_eq!(countries, sorted, "Expected sorted distinct values");
}

// =============================================================================
// JOIN Tests
// =============================================================================

#[test]
fn test_inner_join() {
    let db = Database::open("memory://inner_join").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT c.name, o.product, o.amount
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             ORDER BY c.name, o.product",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 6, "Expected 6 joined rows");
}

#[test]
fn test_left_join() {
    let db = Database::open("memory://left_join").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT c.name, o.product
             FROM customers c
             LEFT JOIN orders o ON c.id = o.customer_id
             ORDER BY c.name",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    let mut names = Vec::new();

    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
        count += 1;
    }

    // LEFT JOIN should include all customers, even those without orders
    // Eve (5) and Frank (not in our data but could be) have no orders
    assert!(count >= 6, "Expected at least 6 rows from LEFT JOIN");
    // All customers should appear
    assert!(
        names.contains(&"Alice".to_string()),
        "Expected Alice in results"
    );
    assert!(
        names.contains(&"Eve".to_string()),
        "Expected Eve in results (customer without orders)"
    );
}

#[test]
fn test_right_join() {
    let db = Database::open("memory://right_join").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT c.name, o.product
             FROM customers c
             RIGHT JOIN orders o ON c.id = o.customer_id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 6, "Expected 6 rows (all orders)");
}

#[test]
fn test_cross_join() {
    let db = Database::open("memory://cross_join").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (x INTEGER)", ()).unwrap();
    db.execute("CREATE TABLE t2 (y INTEGER)", ()).unwrap();

    db.execute("INSERT INTO t1 VALUES (1)", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (2)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (10)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (20)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (30)", ()).unwrap();

    let result = db
        .query("SELECT x, y FROM t1 CROSS JOIN t2", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 6, "Expected 2 * 3 = 6 rows from CROSS JOIN");
}

#[test]
fn test_self_join() {
    let db = Database::open("memory://self_join").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)", ())
        .unwrap();
    db.execute("INSERT INTO employees VALUES (2, 'Manager', 1)", ())
        .unwrap();
    db.execute("INSERT INTO employees VALUES (3, 'Developer', 2)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT e.name as employee, m.name as manager
             FROM employees e
             LEFT JOIN employees m ON e.manager_id = m.id
             ORDER BY e.id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 employees");
}

#[test]
fn test_join_with_where() {
    let db = Database::open("memory://join_where").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT c.name, o.product
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             WHERE o.status = 'completed'",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert!(count > 0, "Expected some completed orders");
}

#[test]
fn test_multiple_joins() {
    let db = Database::open("memory://multi_join").expect("Failed to create database");

    // Create three tables
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER, val TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, b_id INTEGER, val TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO a VALUES (1, 'a1')", ()).unwrap();
    db.execute("INSERT INTO b VALUES (1, 1, 'b1')", ()).unwrap();
    db.execute("INSERT INTO c VALUES (1, 1, 'c1')", ()).unwrap();

    let result = db
        .query(
            "SELECT a.val, b.val, c.val
             FROM a
             INNER JOIN b ON a.id = b.a_id
             INNER JOIN c ON b.id = c.b_id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row from multi-join");
}

// =============================================================================
// SELECT Expression Tests
// =============================================================================

#[test]
fn test_case_expression() {
    let db = Database::open("memory://case_expr").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT product, amount,
                    CASE
                        WHEN amount > 800 THEN 'Expensive'
                        WHEN amount > 400 THEN 'Medium'
                        ELSE 'Cheap'
                    END as price_tier
             FROM orders",
            (),
        )
        .expect("Failed to query");

    let mut tiers = std::collections::HashSet::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let tier: String = row.get(2).unwrap();
        tiers.insert(tier);
    }

    assert!(tiers.contains("Expensive"), "Expected 'Expensive' tier");
    assert!(tiers.contains("Medium"), "Expected 'Medium' tier");
    assert!(tiers.contains("Cheap"), "Expected 'Cheap' tier");
}

#[test]
fn test_coalesce() {
    let db = Database::open("memory://coalesce").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL, 'b2')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, NULL, NULL)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT id, COALESCE(a, b, 'default') as val FROM t ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut vals = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: String = row.get(1).unwrap();
        vals.push(val);
    }

    assert_eq!(vals[0], "a1", "Expected 'a1'");
    assert_eq!(vals[1], "b2", "Expected 'b2' (fallback)");
    assert_eq!(vals[2], "default", "Expected 'default' (fallback)");
}

#[test]
fn test_nullif() {
    let db = Database::open("memory://nullif").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, '')", ()).unwrap();

    let result = db
        .query("SELECT id, NULLIF(val, '') as val FROM t ORDER BY id", ())
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        results.push(id);
    }

    // Just verify we get 2 rows back
    assert_eq!(results.len(), 2, "Expected 2 rows");
    assert_eq!(results[0], 1, "Expected first id = 1");
    assert_eq!(results[1], 2, "Expected second id = 2");
}

// =============================================================================
// Qualified Column Tests
// =============================================================================

#[test]
fn test_table_qualified_columns() {
    let db = Database::open("memory://qualified_cols").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT customers.name, customers.city FROM customers WHERE customers.id = 1",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        assert_eq!(name, "Alice");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

#[test]
fn test_alias_qualified_columns() {
    let db = Database::open("memory://alias_cols").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT c.name, c.city FROM customers AS c WHERE c.id = 1",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        assert_eq!(name, "Alice");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

// =============================================================================
// Derived Table / Subquery in FROM Tests
// =============================================================================

#[test]
fn test_derived_table() {
    let db = Database::open("memory://derived_table").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "SELECT d.name, d.total
             FROM (
                 SELECT c.name, SUM(o.amount) as total
                 FROM customers c
                 INNER JOIN orders o ON c.id = o.customer_id
                 GROUP BY c.name
             ) AS d
             ORDER BY d.total DESC",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert!(count > 0, "Expected some rows from derived table");
}

// =============================================================================
// Empty Table Tests
// =============================================================================

#[test]
fn test_select_from_empty_table() {
    let db = Database::open("memory://empty_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE empty_t (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query("SELECT * FROM empty_t", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 rows from empty table");
}

// =============================================================================
// SELECT with Expression List Tests
// =============================================================================

#[test]
fn test_select_expression_list() {
    let db = Database::open("memory://expr_list").expect("Failed to create database");
    setup_tables(&db);

    // Test selecting multiple expressions
    let result = db
        .query(
            "SELECT id, name, UPPER(name) as upper_name, LENGTH(name) as name_len FROM customers ORDER BY id LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _id: i64 = row.get(0).unwrap();
        let _name: String = row.get(1).unwrap();
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows");
}

// =============================================================================
// Table Function Tests (if applicable)
// =============================================================================

#[test]
fn test_select_without_from() {
    let db = Database::open("memory://no_from").expect("Failed to create database");

    let result = db
        .query("SELECT 1 + 2 as sum, 'hello' || ' world' as greeting", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let sum: i64 = row.get(0).unwrap();
        let greeting: String = row.get(1).unwrap();
        assert_eq!(sum, 3, "Expected 3");
        assert_eq!(greeting, "hello world", "Expected 'hello world'");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}
