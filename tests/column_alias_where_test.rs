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

//! Column Alias in WHERE Clause Tests
//!
//! Tests column alias usage in WHERE clauses

use stoolap::Database;

fn setup_products_db(test_name: &str) -> Database {
    let db = Database::open(&format!("memory://col_alias_where_{}", test_name))
        .expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            price INTEGER,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, price, name) VALUES (1, 50, 'Widget')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO products (id, price, name) VALUES (2, 150, 'Gadget')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO products (id, price, name) VALUES (3, 80, 'Thing')",
        (),
    )
    .expect("Failed to insert data");

    db
}

/// Test using column alias in WHERE clause
#[test]
fn test_alias_in_where_clause() {
    let db = setup_products_db("alias_where");

    let result = db
        .query("SELECT price AS cost FROM products WHERE cost > 100", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();

        assert_eq!(cost, 150, "Expected cost = 150");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching WHERE cost > 100");
}

/// Test multiple aliases in SELECT and WHERE
#[test]
fn test_multiple_aliases_in_where() {
    let db = setup_products_db("multi_alias");

    let result = db
        .query(
            "SELECT id, price AS cost, name AS product_name
            FROM products
            WHERE cost > 60 AND product_name = 'Thing'",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let cost: i64 = row.get(1).unwrap();
        let product_name: String = row.get(2).unwrap();

        assert_eq!(id, 3, "Expected id = 3");
        assert_eq!(cost, 80, "Expected cost = 80");
        assert_eq!(product_name, "Thing", "Expected product_name = 'Thing'");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching complex condition");
}

/// Test alias without WHERE (should work)
#[test]
fn test_alias_without_where() {
    let db = setup_products_db("without_where");

    let result = db
        .query("SELECT price AS cost FROM products ORDER BY cost", ())
        .expect("Failed to execute query");

    let mut costs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        costs.push(cost);
    }

    assert_eq!(costs, vec![50, 80, 150], "Expected sorted costs");
}

/// Test alias with ORDER BY (should work)
#[test]
fn test_alias_with_order_by() {
    let db = setup_products_db("order_by");

    let result = db
        .query("SELECT price AS cost FROM products ORDER BY cost DESC", ())
        .expect("Failed to execute query");

    let mut costs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        costs.push(cost);
    }

    assert_eq!(costs, vec![150, 80, 50], "Expected descending sorted costs");
}

/// Test original column in WHERE, alias in SELECT
#[test]
fn test_original_column_in_where() {
    let db = setup_products_db("orig_where");

    let result = db
        .query("SELECT price AS cost FROM products WHERE price > 100", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();

        assert_eq!(cost, 150, "Expected cost = 150");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching WHERE price > 100");
}

/// Test expression alias in SELECT
#[test]
fn test_expression_alias_in_select() {
    let db = setup_products_db("expr_alias");

    let result = db
        .query(
            "SELECT price * 2 AS double_price FROM products ORDER BY double_price",
            (),
        )
        .expect("Failed to execute query");

    let mut prices: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let double_price: i64 = row.get(0).unwrap();
        prices.push(double_price);
    }

    assert_eq!(
        prices,
        vec![100, 160, 300],
        "Expected doubled prices sorted"
    );
}

/// Test alias with aggregate function
#[test]
fn test_alias_with_aggregate() {
    let db = setup_products_db("aggregate");

    let total: i64 = db
        .query_one("SELECT SUM(price) AS total_price FROM products", ())
        .expect("Failed to execute query");

    // 50 + 150 + 80 = 280
    assert_eq!(total, 280, "Expected total price = 280");
}

/// Test alias with COUNT
#[test]
fn test_alias_with_count() {
    let db = setup_products_db("count");

    let count: i64 = db
        .query_one("SELECT COUNT(*) AS product_count FROM products", ())
        .expect("Failed to execute query");

    assert_eq!(count, 3, "Expected product count = 3");
}
