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

//! Advanced Subquery Tests
//!
//! Tests for EXISTS, IN, ALL/ANY subqueries, semi-join optimization,
//! correlated subqueries, and scalar subqueries.

use stoolap::Database;

fn setup_tables(db: &Database) {
    // Users table
    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create users table");

    // Orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount FLOAT,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Create index on user_id for join optimization
    db.execute("CREATE INDEX idx_orders_user ON orders(user_id)", ())
        .expect("Failed to create index");

    // Insert users
    let users = [
        "INSERT INTO users VALUES (1, 'Alice', 'Engineering', 80000)",
        "INSERT INTO users VALUES (2, 'Bob', 'Engineering', 75000)",
        "INSERT INTO users VALUES (3, 'Charlie', 'Sales', 60000)",
        "INSERT INTO users VALUES (4, 'Diana', 'Sales', 65000)",
        "INSERT INTO users VALUES (5, 'Eve', 'Marketing', 55000)",
        "INSERT INTO users VALUES (6, 'Frank', 'Marketing', 50000)",
    ];

    for sql in &users {
        db.execute(sql, ()).expect("Failed to insert user");
    }

    // Insert orders
    let orders = [
        "INSERT INTO orders VALUES (1, 1, 500.00, 'completed')",
        "INSERT INTO orders VALUES (2, 1, 750.00, 'completed')",
        "INSERT INTO orders VALUES (3, 2, 300.00, 'completed')",
        "INSERT INTO orders VALUES (4, 3, 1000.00, 'pending')",
        "INSERT INTO orders VALUES (5, 3, 200.00, 'completed')",
        "INSERT INTO orders VALUES (6, 4, 150.00, 'cancelled')",
    ];

    for sql in &orders {
        db.execute(sql, ()).expect("Failed to insert order");
    }
}

// =============================================================================
// EXISTS Subquery Tests
// =============================================================================

#[test]
fn test_exists_basic() {
    let db = Database::open("memory://exists_basic").expect("Failed to create database");
    setup_tables(&db);

    // Find users who have at least one order
    let result = db
        .query(
            "SELECT u.name FROM users u
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(
        names,
        vec!["Alice", "Bob", "Charlie", "Diana"],
        "Expected users with orders"
    );
}

#[test]
fn test_not_exists() {
    let db = Database::open("memory://not_exists").expect("Failed to create database");
    setup_tables(&db);

    // Find users who have no orders
    let result = db
        .query(
            "SELECT u.name FROM users u
             WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(names, vec!["Eve", "Frank"], "Expected users without orders");
}

#[test]
fn test_exists_with_additional_condition() {
    let db = Database::open("memory://exists_additional_cond").expect("Failed to create database");
    setup_tables(&db);

    // Find users who have completed orders over $400
    let result = db
        .query(
            "SELECT u.name FROM users u
             WHERE EXISTS (
                 SELECT 1 FROM orders o
                 WHERE o.user_id = u.id
                   AND o.status = 'completed'
                   AND o.amount > 400
             )
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(
        names,
        vec!["Alice"],
        "Expected users with large completed orders"
    );
}

#[test]
fn test_exists_and_condition() {
    let db = Database::open("memory://exists_and").expect("Failed to create database");
    setup_tables(&db);

    // Find Engineering users who have orders
    let result = db
        .query(
            "SELECT u.name FROM users u
             WHERE u.department = 'Engineering'
               AND EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(
        names,
        vec!["Alice", "Bob"],
        "Expected Engineering users with orders"
    );
}

// =============================================================================
// IN Subquery Tests
// =============================================================================

#[test]
fn test_in_subquery_basic() {
    let db = Database::open("memory://in_subquery_basic").expect("Failed to create database");
    setup_tables(&db);

    // Find users who have completed orders
    let result = db
        .query(
            "SELECT name FROM users
             WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed')
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(
        names,
        vec!["Alice", "Bob", "Charlie"],
        "Expected users with completed orders"
    );
}

#[test]
fn test_not_in_subquery() {
    let db = Database::open("memory://not_in_subquery").expect("Failed to create database");
    setup_tables(&db);

    // Find users who don't have any orders
    let result = db
        .query(
            "SELECT name FROM users
             WHERE id NOT IN (SELECT user_id FROM orders)
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(names, vec!["Eve", "Frank"], "Expected users without orders");
}

#[test]
fn test_in_subquery_with_aggregate() {
    let db = Database::open("memory://in_subquery_agg").expect("Failed to create database");
    setup_tables(&db);

    // Find users with total order amount > 500
    let result = db
        .query(
            "SELECT name FROM users
             WHERE id IN (
                 SELECT user_id FROM orders
                 GROUP BY user_id
                 HAVING SUM(amount) > 500
             )
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Alice has 500 + 750 = 1250, Charlie has 1000 + 200 = 1200
    assert_eq!(
        names,
        vec!["Alice", "Charlie"],
        "Expected users with total > 500"
    );
}

// =============================================================================
// ALL/ANY Subquery Tests
// =============================================================================

#[test]
fn test_any_subquery() {
    let db = Database::open("memory://any_subquery").expect("Failed to create database");
    setup_tables(&db);

    // Find orders with amount greater than ANY Sales user's salary / 100
    let result = db
        .query(
            "SELECT id, amount FROM orders
             WHERE amount > ANY (SELECT salary / 100 FROM users WHERE department = 'Sales')
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Sales salaries / 100: 600, 650
    // Orders > 600: 750, 1000 (ids 2, 4)
    assert!(
        !ids.is_empty(),
        "Expected some orders matching ANY condition"
    );
}

#[test]
fn test_all_subquery() {
    let db = Database::open("memory://all_subquery").expect("Failed to create database");
    setup_tables(&db);

    // Find orders with amount greater than ALL Marketing user salaries / 100
    let result = db
        .query(
            "SELECT id, amount FROM orders
             WHERE amount > ALL (SELECT salary / 100 FROM users WHERE department = 'Marketing')
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Marketing salaries / 100: 550, 500
    // Orders > 550: 750, 1000 (ids 2, 4)
    assert!(
        !ids.is_empty(),
        "Expected some orders matching ALL condition"
    );
}

// =============================================================================
// Scalar Subquery Tests
// =============================================================================

#[test]
fn test_scalar_subquery_in_select() {
    let db = Database::open("memory://scalar_subquery_select").expect("Failed to create database");
    setup_tables(&db);

    // Get each user with their total order amount
    let result = db
        .query(
            "SELECT u.name,
                    (SELECT COALESCE(SUM(o.amount), 0) FROM orders o WHERE o.user_id = u.id) as total
             FROM users u
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        results.push((name, total));
    }

    assert_eq!(results.len(), 6, "Expected 6 users");

    // Alice: 500 + 750 = 1250
    let alice = results.iter().find(|(n, _)| n == "Alice").unwrap();
    assert!(
        (alice.1 - 1250.0).abs() < 0.01,
        "Expected Alice total 1250, got {}",
        alice.1
    );

    // Eve: 0 (no orders)
    let eve = results.iter().find(|(n, _)| n == "Eve").unwrap();
    assert!(eve.1.abs() < 0.01, "Expected Eve total 0, got {}", eve.1);
}

#[test]
fn test_scalar_subquery_in_where() {
    let db = Database::open("memory://scalar_subquery_where").expect("Failed to create database");
    setup_tables(&db);

    // Find users with salary above average
    let result = db
        .query(
            "SELECT name, salary FROM users
             WHERE salary > (SELECT AVG(salary) FROM users)
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Average salary: (80000 + 75000 + 60000 + 65000 + 55000 + 50000) / 6 = 64166.67
    // Above average: Alice (80000), Bob (75000), Diana (65000)
    assert_eq!(
        names,
        vec!["Alice", "Bob", "Diana"],
        "Expected users above average salary"
    );
}

#[test]
fn test_scalar_subquery_in_having() {
    let db = Database::open("memory://scalar_subquery_having").expect("Failed to create database");
    setup_tables(&db);

    // Find departments with total salary above overall average salary * 2
    let result = db
        .query(
            "SELECT department, SUM(salary) as total
             FROM users
             GROUP BY department
             HAVING SUM(salary) > (SELECT AVG(salary) FROM users) * 2
             ORDER BY department",
            (),
        )
        .expect("Failed to query");

    let mut depts = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let dept: String = row.get(0).unwrap();
        depts.push(dept);
    }

    // Avg * 2 = 64166.67 * 2 = 128333.33
    // Engineering: 155000, Sales: 125000, Marketing: 105000
    assert!(
        depts.contains(&"Engineering".to_string()),
        "Expected Engineering in results"
    );
}

// =============================================================================
// Multi-column IN Subquery Tests
// =============================================================================

#[test]
fn test_multi_column_in_subquery() {
    let db = Database::open("memory://multi_column_in").expect("Failed to create database");

    // Create tables
    db.execute("CREATE TABLE t1 (a INTEGER, b INTEGER)", ())
        .expect("Failed to create t1");
    db.execute("CREATE TABLE t2 (x INTEGER, y INTEGER)", ())
        .expect("Failed to create t2");

    // Insert data
    db.execute("INSERT INTO t1 VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (2, 20)", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (3, 30)", ()).unwrap();

    db.execute("INSERT INTO t2 VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (2, 25)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (4, 40)", ()).unwrap();

    // Multi-column IN
    let result = db
        .query(
            "SELECT a, b FROM t1 WHERE (a, b) IN (SELECT x, y FROM t2)",
            (),
        )
        .expect("Failed to query");

    let mut matches = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let a: i64 = row.get(0).unwrap();
        let b: i64 = row.get(1).unwrap();
        matches.push((a, b));
    }

    assert_eq!(matches, vec![(1, 10)], "Expected only (1, 10) to match");
}

// =============================================================================
// Correlated Subquery Tests
// =============================================================================

#[test]
fn test_correlated_subquery_count() {
    let db = Database::open("memory://correlated_count").expect("Failed to create database");
    setup_tables(&db);

    // Get order count per user
    let result = db
        .query(
            "SELECT u.name,
                    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count
             FROM users u
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        results.push((name, count));
    }

    // Alice: 2, Bob: 1, Charlie: 2, Diana: 1, Eve: 0, Frank: 0
    let alice = results.iter().find(|(n, _)| n == "Alice").unwrap();
    assert_eq!(alice.1, 2, "Expected Alice to have 2 orders");

    let eve = results.iter().find(|(n, _)| n == "Eve").unwrap();
    assert_eq!(eve.1, 0, "Expected Eve to have 0 orders");
}

#[test]
fn test_correlated_subquery_max() {
    let db = Database::open("memory://correlated_max").expect("Failed to create database");
    setup_tables(&db);

    // Get max order amount per user
    let result = db
        .query(
            "SELECT u.name,
                    (SELECT MAX(o.amount) FROM orders o WHERE o.user_id = u.id) as max_order
             FROM users u
             WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
             ORDER BY u.name",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let max_order: f64 = row.get(1).unwrap();
        results.push((name, max_order));
    }

    // Alice: max(500, 750) = 750
    let alice = results.iter().find(|(n, _)| n == "Alice").unwrap();
    assert!(
        (alice.1 - 750.0).abs() < 0.01,
        "Expected Alice max order 750, got {}",
        alice.1
    );

    // Charlie: max(1000, 200) = 1000
    let charlie = results.iter().find(|(n, _)| n == "Charlie").unwrap();
    assert!(
        (charlie.1 - 1000.0).abs() < 0.01,
        "Expected Charlie max order 1000, got {}",
        charlie.1
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_subquery_with_null() {
    let db = Database::open("memory://subquery_null").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, ref_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (3, 30)", ()).unwrap();

    db.execute("INSERT INTO t2 VALUES (1, 1)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (3, 3)", ()).unwrap();

    // IN with NULL values - NULL never matches
    let result = db
        .query(
            "SELECT id FROM t1 WHERE id IN (SELECT ref_id FROM t2) ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![1, 3], "Expected only non-NULL matches");
}

#[test]
fn test_nested_subquery() {
    let db = Database::open("memory://nested_subquery").expect("Failed to create database");
    setup_tables(&db);

    // Find users in departments that have someone earning > 70000
    let result = db
        .query(
            "SELECT name FROM users
             WHERE department IN (
                 SELECT department FROM users
                 WHERE salary > 70000
             )
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Engineering has Alice (80000) and Bob (75000) > 70000
    assert_eq!(
        names,
        vec!["Alice", "Bob"],
        "Expected Engineering department users"
    );
}

#[test]
fn test_subquery_with_limit() {
    let db = Database::open("memory://subquery_limit").expect("Failed to create database");
    setup_tables(&db);

    // Find users whose id is in top 2 order amounts
    let result = db
        .query(
            "SELECT name FROM users
             WHERE id IN (
                 SELECT user_id FROM orders ORDER BY amount DESC LIMIT 2
             )
             ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Top 2 orders by amount: 1000 (user 3 - Charlie), 750 (user 1 - Alice)
    assert_eq!(names, vec!["Alice", "Charlie"], "Expected top order users");
}
