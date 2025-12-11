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

//! Column Alias Tests
//!
//! Tests column aliasing in SELECT statements

use stoolap::Database;

/// Test simple SELECT with column alias
#[test]
fn test_column_alias_in_select() {
    let db = Database::open("memory://alias_select").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER, price INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO items (id, price, name) VALUES (1, 100, 'Item A')",
        (),
    )
    .expect("Failed to insert data");

    // Test SELECT with column alias
    let result = db
        .query("SELECT price AS cost FROM items", ())
        .expect("Failed to execute SELECT with alias");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        assert_eq!(cost, 100);
        count += 1;
    }
    assert_eq!(count, 1);
}

/// Test multiple column aliases
#[test]
fn test_multiple_column_aliases() {
    let db = Database::open("memory://alias_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, price, quantity) VALUES (1, 50, 10)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT price AS unit_price, quantity AS qty FROM products",
            (),
        )
        .expect("Failed to execute SELECT with multiple aliases");

    for row in result {
        let row = row.expect("Failed to get row");
        let unit_price: i64 = row.get(0).unwrap();
        let qty: i64 = row.get(1).unwrap();
        assert_eq!(unit_price, 50);
        assert_eq!(qty, 10);
    }
}

/// Test alias with expression
#[test]
fn test_alias_with_expression() {
    let db = Database::open("memory://alias_expr").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, price, quantity) VALUES (1, 20, 5)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT price * quantity AS total FROM sales", ())
        .expect("Failed to execute SELECT with expression alias");

    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(0).unwrap();
        assert_eq!(total, 100); // 20 * 5
    }
}

/// Test alias with aggregation
#[test]
fn test_alias_with_aggregation() {
    let db = Database::open("memory://alias_aggr").expect("Failed to create database");

    db.execute("CREATE TABLE orders (id INTEGER, amount INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO orders (id, amount) VALUES (1, 100)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, amount) VALUES (2, 200)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, amount) VALUES (3, 150)", ())
        .unwrap();

    let result: i64 = db
        .query_one("SELECT SUM(amount) AS total_amount FROM orders", ())
        .expect("Failed to execute aggregate with alias");

    assert_eq!(result, 450);
}

/// Test alias in GROUP BY with HAVING
#[test]
fn test_alias_in_group_by_having() {
    let db = Database::open("memory://alias_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE transactions (id INTEGER, category TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO transactions (id, category, amount) VALUES (1, 'A', 100)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO transactions (id, category, amount) VALUES (2, 'A', 200)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO transactions (id, category, amount) VALUES (3, 'B', 50)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO transactions (id, category, amount) VALUES (4, 'B', 75)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO transactions (id, category, amount) VALUES (5, 'C', 300)",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT category, SUM(amount) AS total FROM transactions GROUP BY category HAVING total > 100 ORDER BY category", ())
        .expect("Failed to execute GROUP BY with HAVING");

    let mut rows: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let total: i64 = row.get(1).unwrap();
        rows.push((category, total));
    }

    // A: 300, B: 125, C: 300 - all > 100
    assert_eq!(rows.len(), 3);
}

/// Test alias with ORDER BY
#[test]
fn test_alias_with_order_by() {
    let db = Database::open("memory://alias_order").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items (id, value) VALUES (1, 30)", ())
        .unwrap();
    db.execute("INSERT INTO items (id, value) VALUES (2, 10)", ())
        .unwrap();
    db.execute("INSERT INTO items (id, value) VALUES (3, 20)", ())
        .unwrap();

    let result = db
        .query("SELECT id, value AS v FROM items ORDER BY v", ())
        .expect("Failed to execute ORDER BY with alias");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let v: i64 = row.get(1).unwrap();
        values.push(v);
    }

    assert_eq!(values, vec![10, 20, 30]);
}

/// Test alias with COUNT
#[test]
fn test_alias_with_count() {
    let db = Database::open("memory://alias_count").expect("Failed to create database");

    db.execute("CREATE TABLE records (id INTEGER, status TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO records (id, status) VALUES (1, 'active')", ())
        .unwrap();
    db.execute("INSERT INTO records (id, status) VALUES (2, 'active')", ())
        .unwrap();
    db.execute(
        "INSERT INTO records (id, status) VALUES (3, 'inactive')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT status, COUNT(*) AS cnt FROM records GROUP BY status ORDER BY status",
            (),
        )
        .expect("Failed to execute COUNT with alias");

    let mut rows: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let status: String = row.get(0).unwrap();
        let cnt: i64 = row.get(1).unwrap();
        rows.push((status, cnt));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], ("active".to_string(), 2));
    assert_eq!(rows[1], ("inactive".to_string(), 1));
}

/// Test table alias
#[test]
fn test_table_alias() {
    let db = Database::open("memory://table_alias").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER, name TEXT, dept_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees (id, name, dept_id) VALUES (1, 'John', 10)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT e.id, e.name FROM employees e WHERE e.dept_id = 10",
            (),
        )
        .expect("Failed to execute query with table alias");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        assert_eq!(id, 1);
        assert_eq!(name, "John");
        count += 1;
    }
    assert_eq!(count, 1);
}
