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

//! Column Alias in WHERE Clause Basic Tests
//!
//! Tests basic column alias functionality in WHERE clauses

use stoolap::Database;

/// Test basic SELECT with column alias
#[test]
fn test_basic_alias_select() {
    let db = Database::open("memory://alias_basic").expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE basic_alias_test_items (id INTEGER, price INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO basic_alias_test_items (id, price, name) VALUES (1, 50, 'Widget')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO basic_alias_test_items (id, price, name) VALUES (2, 150, 'Gadget')",
        (),
    )
    .expect("Failed to insert data");

    // Verify row count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM basic_alias_test_items", ())
        .expect("Failed to count");
    assert_eq!(count, 2);

    // Test SELECT with column alias
    let result = db
        .query("SELECT price AS cost FROM basic_alias_test_items", ())
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let price: i64 = row.get(0).unwrap();
        values.push(price);
    }

    assert_eq!(values.len(), 2);
    assert!(values.contains(&50));
    assert!(values.contains(&150));
}

/// Test basic WHERE filter without alias
#[test]
fn test_basic_where_filter() {
    let db = Database::open("memory://where_basic").expect("Failed to create database");

    db.execute(
        "CREATE TABLE where_test_items (id INTEGER, price INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO where_test_items (id, price, name) VALUES (1, 50, 'Widget'), (2, 150, 'Gadget')", ())
        .expect("Failed to insert data");

    // Normal WHERE with no alias
    let result = db
        .query("SELECT * FROM where_test_items WHERE price > 100", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let price: i64 = row.get(1).unwrap();
        assert!(price > 100);
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row with price > 100");
}

/// Test alias in WHERE clause
#[test]
fn test_alias_in_where() {
    let db = Database::open("memory://alias_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE alias_where_items (id INTEGER, price INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO alias_where_items (id, price, name) VALUES (1, 50, 'Widget'), (2, 150, 'Gadget')", ())
    .expect("Failed to insert data");

    // Column alias in WHERE
    let result = db
        .query(
            "SELECT price AS cost FROM alias_where_items WHERE cost > 100",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        assert!(cost > 100);
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row with cost > 100");
}

/// Test multiple aliases with WHERE
#[test]
fn test_multiple_aliases_with_where() {
    let db = Database::open("memory://multi_alias_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_alias_items (id INTEGER, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO multi_alias_items (id, price, quantity) VALUES
         (1, 10, 5), (2, 20, 3), (3, 15, 10)",
        (),
    )
    .expect("Failed to insert data");

    // Multiple aliases and filter by one
    let result = db
        .query(
            "SELECT price AS unit_price, quantity AS qty FROM multi_alias_items WHERE qty > 4",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let qty: i64 = row.get(1).unwrap();
        assert!(qty > 4);
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 rows with qty > 4");
}

/// Test expression alias in WHERE
#[test]
fn test_expression_alias_in_where() {
    let db = Database::open("memory://expr_alias_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE expr_alias_items (id INTEGER, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Data: (10*5=50), (20*3=60), (15*10=150) - only 150 > 60
    db.execute(
        "INSERT INTO expr_alias_items (id, price, quantity) VALUES
         (1, 10, 5), (2, 20, 3), (3, 15, 10)",
        (),
    )
    .expect("Failed to insert data");

    // Expression alias with WHERE
    let result = db
        .query(
            "SELECT price * quantity AS total FROM expr_alias_items WHERE total > 60",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(0).unwrap();
        assert!(total > 60);
        count += 1;
    }

    // Only one row matches: 15 * 10 = 150 > 60
    assert_eq!(count, 1, "Expected 1 row with total > 60");
}

/// Test alias with ORDER BY
#[test]
fn test_alias_with_where_and_order() {
    let db = Database::open("memory://alias_where_order").expect("Failed to create database");

    db.execute(
        "CREATE TABLE alias_order_items (id INTEGER, price INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO alias_order_items (id, price) VALUES (1, 100), (2, 50), (3, 150), (4, 75)",
        (),
    )
    .expect("Failed to insert data");

    // Alias in SELECT, used in WHERE and ORDER BY
    let result = db
        .query(
            "SELECT price AS cost FROM alias_order_items WHERE cost > 60 ORDER BY cost",
            (),
        )
        .expect("Failed to execute query");

    let mut costs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        costs.push(cost);
    }

    assert_eq!(costs, vec![75, 100, 150], "Expected sorted costs > 60");
}

/// Test alias with aggregate in WHERE (HAVING)
#[test]
fn test_alias_aggregate_having() {
    let db = Database::open("memory://alias_having").expect("Failed to create database");

    db.execute(
        "CREATE TABLE alias_having_sales (id INTEGER, category TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO alias_having_sales (id, category, amount) VALUES
         (1, 'A', 100), (2, 'A', 200), (3, 'B', 50), (4, 'B', 30), (5, 'C', 500)",
        (),
    )
    .expect("Failed to insert data");

    // Use alias in HAVING clause
    let result = db
        .query(
            "SELECT category, SUM(amount) AS total FROM alias_having_sales GROUP BY category HAVING total > 100 ORDER BY category", ())
        .expect("Failed to execute query");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let total: i64 = row.get(1).unwrap();
        results.push((category, total));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("A".to_string(), 300));
    assert_eq!(results[1], ("C".to_string(), 500));
}

/// Test alias with LIKE in WHERE
#[test]
fn test_alias_with_like() {
    let db = Database::open("memory://alias_like").expect("Failed to create database");

    db.execute("CREATE TABLE alias_like_items (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO alias_like_items (id, name) VALUES
         (1, 'apple'), (2, 'banana'), (3, 'apricot'), (4, 'cherry')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT name AS fruit FROM alias_like_items WHERE fruit LIKE 'a%'",
            (),
        )
        .expect("Failed to execute query");

    let mut fruits: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let fruit: String = row.get(0).unwrap();
        fruits.push(fruit);
    }

    assert_eq!(fruits.len(), 2);
    assert!(fruits.contains(&"apple".to_string()));
    assert!(fruits.contains(&"apricot".to_string()));
}

/// Test alias with NULL check in WHERE
#[test]
fn test_alias_with_null_check() {
    let db = Database::open("memory://alias_null_check").expect("Failed to create database");

    db.execute(
        "CREATE TABLE alias_null_items (id INTEGER, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO alias_null_items (id, value) VALUES (1, 10), (2, NULL), (3, 30)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT value AS val FROM alias_null_items WHERE val IS NOT NULL",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _val: i64 = row.get(0).unwrap();
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 rows with non-NULL values");
}
