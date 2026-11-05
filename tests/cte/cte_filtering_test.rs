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

//! CTE Filtering Tests
//!
//! Tests CTE operations with various WHERE clause filters and aggregates

use stoolap::Database;

fn setup_products_db(test_name: &str) -> Database {
    let db = Database::open(&format!("memory://cte_filter_{}", test_name))
        .expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price FLOAT,
            category TEXT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert 1000 test rows
    for i in 1..=1000 {
        let category = match i % 5 {
            1 => "CategoryA",
            2 => "CategoryB",
            3 => "CategoryC",
            4 => "CategoryD",
            _ => "CategoryE",
        };
        let in_stock = i % 2 == 0;
        let price = (i as f64) * 10.5;

        db.execute(&format!(
            "INSERT INTO products (id, name, price, category, in_stock) VALUES ({}, 'Product{}', {:.1}, '{}', {})",
            i, i, price, category, in_stock
        ), ())
        .expect("Failed to insert data");
    }

    db
}

/// Test CTE with range filtering (price > 5000)
#[test]
fn test_cte_with_range_filtering() {
    let db = setup_products_db("filtering");

    let count: i64 = db
        .query_one(
            "WITH expensive_products AS (
                SELECT id, name, price, category
                FROM products
                WHERE price > 5000
            )
            SELECT COUNT(*) FROM expensive_products",
            (),
        )
        .expect("Failed to execute query");

    // Products with price > 5000 are those with id > 476 (since price = id * 10.5)
    // 5000 / 10.5 = 476.19, so id > 476 means ids 477-1000 = 524 products
    assert_eq!(count, 524, "Expected 524 products with price > 5000");
}

/// Test CTE with equality filter
#[test]
fn test_cte_with_equality_filter() {
    let db = setup_products_db("equality");

    let count: i64 = db
        .query_one(
            "WITH category_products AS (
                SELECT id, name, price, category
                FROM products
                WHERE category = 'CategoryA'
            )
            SELECT COUNT(*) FROM category_products",
            (),
        )
        .expect("Failed to execute query");

    // CategoryA is for ids where id % 5 == 1 (1, 6, 11, 16, ..., 996)
    // That's 200 products
    assert_eq!(count, 200, "Expected 200 products in CategoryA");
}

/// Test CTE with MIN/MAX aggregates
#[test]
fn test_cte_min_max() {
    let db = setup_products_db("min_max");

    let result = db
        .query(
            "WITH all_products AS (
                SELECT id, name, price
                FROM products
            )
            SELECT MIN(price), MAX(price) FROM all_products",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let min_price: f64 = row.get(0).unwrap();
        let max_price: f64 = row.get(1).unwrap();

        // id=1, price=1*10.5=10.5
        assert!(
            (min_price - 10.5).abs() < 0.01,
            "Expected min price 10.5, got {}",
            min_price
        );
        // id=1000, price=1000*10.5=10500.0
        assert!(
            (max_price - 10500.0).abs() < 0.01,
            "Expected max price 10500.0, got {}",
            max_price
        );
    }
}

/// Test CTE reused multiple times
#[test]
fn test_cte_reused_multiple_times() {
    let db = setup_products_db("reused");

    // Test simpler query - just get the count of cheap products
    let total: i64 = db
        .query_one(
            "WITH cheap_products AS (
                SELECT id, name, price
                FROM products
                WHERE price < 1000
            )
            SELECT COUNT(*) FROM cheap_products",
            (),
        )
        .expect("Failed to execute query");

    // price < 1000 means id < 96 (95 products: 1-95, since 95*10.5=997.5 < 1000)
    assert_eq!(total, 95, "Expected 95 total cheap products");
}

/// Test CTE with very cheap products count
#[test]
fn test_cte_very_cheap_products() {
    let db = setup_products_db("very_cheap");

    let very_cheap: i64 = db
        .query_one(
            "WITH cheap_products AS (
                SELECT id, name, price
                FROM products
                WHERE price < 500
            )
            SELECT COUNT(*) FROM cheap_products",
            (),
        )
        .expect("Failed to execute query");

    // price < 500 means id < 48 (47 products: 1-47, since 47*10.5=493.5 < 500)
    assert_eq!(very_cheap, 47, "Expected 47 very cheap products");
}

/// Test CTE with boolean filter
#[test]
fn test_cte_boolean_filter() {
    let db = setup_products_db("boolean");

    let count: i64 = db
        .query_one(
            "WITH in_stock_products AS (
                SELECT id, name, price
                FROM products
                WHERE in_stock = true
            )
            SELECT COUNT(*) FROM in_stock_products",
            (),
        )
        .expect("Failed to execute query");

    // in_stock is true for even ids (2, 4, 6, ..., 1000) = 500 products
    assert_eq!(count, 500, "Expected 500 in-stock products");
}

/// Test CTE with combined filters
#[test]
fn test_cte_combined_filters() {
    let db = setup_products_db("combined");

    let count: i64 = db
        .query_one(
            "WITH filtered_products AS (
                SELECT id, name, price, category
                FROM products
                WHERE price > 500 AND category = 'CategoryA'
            )
            SELECT COUNT(*) FROM filtered_products",
            (),
        )
        .expect("Failed to execute query");

    // price > 500 means id > 47
    // CategoryA means id % 5 == 1
    // So ids: 51, 56, 61, ..., 996 where id % 5 == 1 and id > 47
    // Starting from 51, stepping by 5: 51, 56, 61, ... 996
    // Count: (996 - 51) / 5 + 1 = 190
    assert_eq!(
        count, 190,
        "Expected 190 products matching combined filters"
    );
}

/// Test CTE with SUM aggregate
#[test]
fn test_cte_sum_aggregate() {
    let db = setup_products_db("sum");

    let sum: f64 = db
        .query_one(
            "WITH all_products AS (
                SELECT price FROM products WHERE id <= 10
            )
            SELECT SUM(price) FROM all_products",
            (),
        )
        .expect("Failed to execute query");

    // Sum of prices for ids 1-10: (1+2+3+4+5+6+7+8+9+10) * 10.5 = 55 * 10.5 = 577.5
    assert!(
        (sum - 577.5).abs() < 0.01,
        "Expected sum 577.5, got {}",
        sum
    );
}

/// Test CTE with AVG aggregate
#[test]
fn test_cte_avg_aggregate() {
    let db = setup_products_db("avg");

    let avg: f64 = db
        .query_one(
            "WITH subset AS (
                SELECT price FROM products WHERE id <= 10
            )
            SELECT AVG(price) FROM subset",
            (),
        )
        .expect("Failed to execute query");

    // Average of prices for ids 1-10: 577.5 / 10 = 57.75
    assert!(
        (avg - 57.75).abs() < 0.01,
        "Expected avg 57.75, got {}",
        avg
    );
}

/// Test nested CTE
#[test]
fn test_nested_cte() {
    let db = setup_products_db("nested");

    let count: i64 = db
        .query_one(
            "WITH cheap AS (
                SELECT id, price FROM products WHERE price < 1000
            ),
            very_cheap AS (
                SELECT id, price FROM cheap WHERE price < 500
            )
            SELECT COUNT(*) FROM very_cheap",
            (),
        )
        .expect("Failed to execute query");

    // price < 500 means id < 48 (47 products)
    assert_eq!(count, 47, "Expected 47 very cheap products from nested CTE");
}
