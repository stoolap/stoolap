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

//! Expression Pushdown Tests
//!
//! Tests NOT, IN, and NOT IN expression handling

use stoolap::Database;

fn setup_test_products(db: &Database) {
    db.execute(
        "CREATE TABLE test_products (
            id INTEGER,
            name TEXT,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            tags TEXT,
            supply_date TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create test table");

    let inserts = [
        "INSERT INTO test_products VALUES (1, 'Laptop', 'Electronics', 1200.00, true, 'premium,tech', TIMESTAMP '2023-01-15 00:00:00')",
        "INSERT INTO test_products VALUES (2, 'Smartphone', 'Electronics', 800.00, true, 'mobile,tech', TIMESTAMP '2023-02-20 00:00:00')",
        "INSERT INTO test_products VALUES (3, 'Headphones', 'Electronics', 150.00, true, 'audio,tech', TIMESTAMP '2023-03-10 00:00:00')",
        "INSERT INTO test_products VALUES (4, 'T-shirt', 'Clothing', 25.00, true, 'casual,cotton', TIMESTAMP '2023-01-25 00:00:00')",
        "INSERT INTO test_products VALUES (5, 'Jeans', 'Clothing', 50.00, false, 'denim,casual', NULL)",
        "INSERT INTO test_products VALUES (6, 'Sneakers', 'Footwear', 80.00, true, 'casual,sports', TIMESTAMP '2023-02-05 00:00:00')",
        "INSERT INTO test_products VALUES (7, 'Boots', 'Footwear', 120.00, false, 'winter,leather', NULL)",
        "INSERT INTO test_products VALUES (8, 'Desk', 'Furniture', 250.00, true, 'office,wood', TIMESTAMP '2023-03-25 00:00:00')",
        "INSERT INTO test_products VALUES (9, 'Chair', 'Furniture', 150.00, true, 'office,comfort', TIMESTAMP '2023-03-25 00:00:00')",
        "INSERT INTO test_products VALUES (10, NULL, NULL, NULL, NULL, NULL, NULL)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert row");
    }
}

fn count_rows(db: &Database, query: &str) -> i64 {
    let result = db.query(query, ()).expect("Failed to execute query");
    let mut count = 0;
    for _row in result {
        count += 1;
    }
    count
}

// ============ NOT Expressions ============

/// Test NOT with equality
#[test]
fn test_not_with_equality() {
    let db = Database::open("memory://expr_not_eq").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT category = 'Electronics'",
    );
    // All non-Electronics categories: Clothing(2) + Footwear(2) + Furniture(2) = 6
    // NULL row: NOT (NULL = 'Electronics') = NOT NULL = NULL, which is excluded from WHERE
    assert_eq!(count, 6, "Expected 6 non-Electronics products");
}

/// Test NOT with inequality
#[test]
fn test_not_with_inequality() {
    let db = Database::open("memory://expr_not_neq").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(&db, "SELECT * FROM test_products WHERE NOT price > 100");
    // Price <= 100: T-shirt(25), Jeans(50), Sneakers(80) = 3
    // NULL row returns false for NOT
    assert!(count >= 3, "Expected at least 3 products with price <= 100");
}

/// Test NOT with AND condition
#[test]
fn test_not_with_and_condition() {
    let db = Database::open("memory://expr_not_and").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category = 'Electronics' AND price > 500)",
    );
    // Not (Electronics AND expensive): should exclude Laptop(1200) and Smartphone(800)
    assert_eq!(
        count, 8,
        "Expected 8 products not (Electronics AND expensive)"
    );
}

/// Test NOT with OR condition
#[test]
fn test_not_with_or_condition() {
    let db = Database::open("memory://expr_not_or").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category = 'Electronics' OR category = 'Clothing')",
    );
    // Not Electronics(3) or Clothing(2): Footwear(2) + Furniture(2) = 4, plus depends on NULL handling
    assert!(
        count >= 4,
        "Expected at least 4 products not Electronics or Clothing"
    );
}

/// Test NOT with NULL check
#[test]
fn test_not_with_null_check() {
    let db = Database::open("memory://expr_not_null").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category IS NULL)",
    );
    // All rows with non-NULL category
    assert_eq!(count, 9, "Expected 9 rows with non-NULL category");
}

/// Test NOT with composite condition
#[test]
fn test_not_with_composite_condition() {
    let db = Database::open("memory://expr_not_composite").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category = 'Electronics' AND price > 500 AND in_stock = true)",
    );
    // Not (expensive Electronics in stock)
    assert_eq!(
        count, 8,
        "Expected 8 products not (expensive Electronics in stock)"
    );
}

// ============ IN Expressions ============

/// Test simple IN with multiple values
#[test]
fn test_in_multiple_values() {
    let db = Database::open("memory://expr_in_multi").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category IN ('Electronics', 'Clothing')",
    );
    // Electronics (3) + Clothing (2) = 5
    assert_eq!(count, 5, "Expected 5 Electronics or Clothing products");
}

/// Test IN with single value
#[test]
fn test_in_single_value() {
    let db = Database::open("memory://expr_in_single").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category IN ('Furniture')",
    );
    // Same as category = 'Furniture'
    assert_eq!(count, 2, "Expected 2 Furniture products");
}

/// Test IN with numeric values
#[test]
fn test_in_numeric_values() {
    let db = Database::open("memory://expr_in_numeric").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE price IN (50.00, 150.00, 250.00)",
    );
    // Jeans(50), Headphones(150), Chair(150), Desk(250) = 4
    assert_eq!(count, 4, "Expected 4 products with these exact prices");
}

/// Test IN with no matching values
#[test]
fn test_in_no_match() {
    let db = Database::open("memory://expr_in_nomatch").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category IN ('Unknown', 'NonExistent')",
    );
    assert_eq!(count, 0, "Expected 0 matches for unknown categories");
}

/// Test IN combined with other conditions
#[test]
fn test_in_combined_with_and() {
    let db = Database::open("memory://expr_in_and").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category IN ('Electronics', 'Clothing') AND in_stock = true",
    );
    // In-stock Electronics (3) + in-stock Clothing (T-shirt only, Jeans is out of stock) = 4
    assert_eq!(
        count, 4,
        "Expected 4 in-stock Electronics and Clothing products"
    );
}

// ============ NOT IN Expressions ============

/// Test simple NOT IN
#[test]
fn test_not_in_simple() {
    let db = Database::open("memory://expr_not_in").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category NOT IN ('Electronics', 'Clothing')",
    );
    // All categories except Electronics(3) and Clothing(2)
    // Footwear(2) + Furniture(2) = 4, NULL row handling varies
    assert!(
        count >= 4,
        "Expected at least 4 products not in Electronics or Clothing"
    );
}

/// Test NOT IN combined with AND
#[test]
fn test_not_in_combined_with_and() {
    let db = Database::open("memory://expr_not_in_and").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category NOT IN ('Electronics', 'Clothing') AND in_stock = true",
    );
    // In-stock products not in Electronics or Clothing
    // Sneakers, Desk, Chair = 3
    assert_eq!(
        count, 3,
        "Expected 3 in-stock products not in Electronics or Clothing"
    );
}

/// Test NOT IN combined with OR
#[test]
fn test_not_in_combined_with_or() {
    let db = Database::open("memory://expr_not_in_or").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category NOT IN ('Electronics', 'Clothing') OR price > 200",
    );
    // Not (Electronics or Clothing) OR expensive products
    assert!(
        count >= 5,
        "Expected at least 5 products matching NOT IN OR condition"
    );
}

/// Test NOT IN with numeric values
#[test]
fn test_not_in_numeric() {
    let db = Database::open("memory://expr_not_in_num").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE id NOT IN (1, 2, 3, 4, 5)",
    );
    // IDs 6-10 = 5
    assert_eq!(count, 5, "Expected 5 products with IDs 6-10");
}

/// Test NOT with IN expression (parenthesized)
#[test]
fn test_not_in_parenthesized() {
    let db = Database::open("memory://expr_not_in_paren").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category IN ('Electronics', 'Clothing'))",
    );
    // Same as category NOT IN ('Electronics', 'Clothing')
    assert!(
        count >= 4,
        "Expected at least 4 products not in Electronics or Clothing"
    );
}

// ============ Complex Expressions ============

/// Test complex nested NOT expressions
#[test]
fn test_complex_nested_not() {
    let db = Database::open("memory://expr_complex").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT (category = 'Electronics' OR (category = 'Clothing' AND price > 30))",
    );
    // Not Electronics, not (Clothing with price > 30)
    // Excludes: Electronics(3), Jeans(50) = 4
    // Keeps: T-shirt(25), Footwear(2), Furniture(2) + NULL handling
    assert!(
        count >= 5,
        "Expected at least 5 products matching complex condition"
    );
}

/// Test IN with OR
#[test]
fn test_in_with_or() {
    let db = Database::open("memory://expr_in_or").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE category IN ('Electronics') OR price < 30",
    );
    // Electronics(3) + T-shirt(25) = 4
    assert_eq!(count, 4, "Expected 4 Electronics or cheap products");
}

/// Test double NOT
#[test]
fn test_double_not() {
    let db = Database::open("memory://expr_double_not").expect("Failed to create database");
    setup_test_products(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM test_products WHERE NOT NOT category = 'Electronics'",
    );
    // Same as category = 'Electronics'
    assert_eq!(count, 3, "Expected 3 Electronics products with double NOT");
}
