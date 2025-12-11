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

//! Integration tests for join optimizer

use stoolap::Database;

#[test]
fn test_explain_shows_hash_join() {
    let db = Database::open("memory://join_opt_hash").unwrap();

    // Create tables with equality join condition
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();

    // EXPLAIN should show Hash Join for equality condition
    let result = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();

    let mut found_hash_join = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("Hash Join") {
            found_hash_join = true;
            break;
        }
    }

    assert!(
        found_hash_join,
        "EXPLAIN should show Hash Join for equality condition"
    );
}

#[test]
fn test_explain_shows_nested_loop_for_cross_join() {
    let db = Database::open("memory://join_opt_nested").unwrap();

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // CROSS JOIN should show Nested Loop
    let result = db
        .query("EXPLAIN SELECT * FROM t1 CROSS JOIN t2", ())
        .unwrap();

    let mut found_nested_loop = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("Nested Loop") {
            found_nested_loop = true;
            break;
        }
    }

    assert!(
        found_nested_loop,
        "EXPLAIN should show Nested Loop for CROSS JOIN"
    );
}

#[test]
fn test_explain_shows_cost_and_rows() {
    let db = Database::open("memory://join_opt_cost").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();

    let mut found_cost = false;
    let mut found_rows = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("cost=") {
            found_cost = true;
        }
        if plan_line.contains("rows=") {
            found_rows = true;
        }
    }

    assert!(found_cost, "EXPLAIN should show cost estimate");
    assert!(found_rows, "EXPLAIN should show row estimate");
}

#[test]
fn test_explain_analyze_shows_actual_rows() {
    let db = Database::open("memory://join_opt_analyze").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 1), (3, 2)", ())
        .unwrap();

    let result = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();

    let mut found_actual_rows = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("actual rows=") {
            found_actual_rows = true;
            break;
        }
    }

    assert!(
        found_actual_rows,
        "EXPLAIN ANALYZE should show actual row count"
    );
}

#[test]
fn test_statistics_improve_estimates() {
    let db = Database::open("memory://join_opt_stats").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 1), (3, 2)", ())
        .unwrap();

    // Get estimate without ANALYZE
    let result_before = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();
    let mut cost_before = String::new();
    for row in result_before {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("Join") && plan_line.contains("cost=") {
            cost_before = plan_line;
            break;
        }
    }

    // Run ANALYZE
    db.execute("ANALYZE users", ()).unwrap();
    db.execute("ANALYZE orders", ()).unwrap();

    // Get estimate after ANALYZE
    let result_after = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();
    let mut cost_after = String::new();
    for row in result_after {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("Join") && plan_line.contains("cost=") {
            cost_after = plan_line;
            break;
        }
    }

    // After ANALYZE, the estimates should be different (more accurate)
    assert!(!cost_before.is_empty(), "Should have cost before ANALYZE");
    assert!(!cost_after.is_empty(), "Should have cost after ANALYZE");
    assert_ne!(
        cost_before, cost_after,
        "Cost estimates should change after ANALYZE"
    );
}

#[test]
fn test_hash_join_executes_correctly() {
    let db = Database::open("memory://join_opt_exec").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 150.0)",
        (),
    )
    .unwrap();

    // Execute the join
    let result = db
        .query("SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id ORDER BY o.total", ())
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        rows.push((name, total));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("Alice".to_string(), 100.0));
    assert_eq!(rows[1], ("Bob".to_string(), 150.0));
    assert_eq!(rows[2], ("Alice".to_string(), 200.0));
}

#[test]
fn test_left_join_with_hash_join() {
    let db = Database::open("memory://join_opt_left").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 1)", ())
        .unwrap();

    // LEFT JOIN should include users without orders
    let result = db
        .query(
            "SELECT u.name, o.id FROM users u LEFT JOIN orders o ON u.id = o.user_id ORDER BY u.id",
            (),
        )
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        let order_id: Option<i64> = row.get(1).ok();
        rows.push((name, order_id));
    }

    // Alice has 2 orders, Bob has 0, Charlie has 0
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().any(|(n, id)| n == "Alice" && id.is_some()));
    assert!(rows.iter().any(|(n, id)| n == "Bob" && id.is_none()));
    assert!(rows.iter().any(|(n, id)| n == "Charlie" && id.is_none()));
}

#[test]
fn test_join_with_complex_condition() {
    let db = Database::open("memory://join_opt_complex").unwrap();

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE order_items (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO products VALUES (1, 10.0), (2, 20.0), (3, 30.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO order_items VALUES (1, 1, 2), (2, 2, 1), (3, 1, 3)",
        (),
    )
    .unwrap();

    // Join with equality and additional filter
    let result = db
        .query(
            "SELECT p.price, oi.quantity
             FROM products p
             INNER JOIN order_items oi ON p.id = oi.product_id
             WHERE oi.quantity > 1
             ORDER BY p.price",
            (),
        )
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let price: f64 = row.get(0).unwrap();
        let quantity: i64 = row.get(1).unwrap();
        rows.push((price, quantity));
    }

    // Only items with quantity > 1
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|(_, q)| *q > 1));
}
