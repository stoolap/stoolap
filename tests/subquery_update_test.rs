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

//! UPDATE with Subquery Tests
//!
//! Tests UPDATE statements with IN and NOT IN subqueries

use stoolap::Database;

fn setup_update_tables(db: &Database) {
    // Create products table
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price FLOAT,
            discount FLOAT
        )",
        (),
    )
    .expect("Failed to create products table");

    // Create categories table
    db.execute(
        "CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT,
            is_premium BOOLEAN
        )",
        (),
    )
    .expect("Failed to create categories table");

    // Insert categories
    db.execute(
        "INSERT INTO categories (id, name, is_premium) VALUES (1, 'Electronics', true)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories (id, name, is_premium) VALUES (2, 'Books', false)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories (id, name, is_premium) VALUES (3, 'Clothing', true)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories (id, name, is_premium) VALUES (4, 'Food', false)",
        (),
    )
    .unwrap();

    // Insert products
    db.execute("INSERT INTO products (id, name, category, price, discount) VALUES (1, 'Laptop', 'Electronics', 1000.0, 0.0)", ()).unwrap();
    db.execute("INSERT INTO products (id, name, category, price, discount) VALUES (2, 'Novel', 'Books', 20.0, 0.0)", ()).unwrap();
    db.execute("INSERT INTO products (id, name, category, price, discount) VALUES (3, 'Shirt', 'Clothing', 50.0, 0.0)", ()).unwrap();
    db.execute("INSERT INTO products (id, name, category, price, discount) VALUES (4, 'Phone', 'Electronics', 800.0, 0.0)", ()).unwrap();
    db.execute("INSERT INTO products (id, name, category, price, discount) VALUES (5, 'Bread', 'Food', 5.0, 0.0)", ()).unwrap();
}

/// Test UPDATE with IN subquery
#[test]
fn test_update_with_in_subquery() {
    let db = Database::open("memory://update_in").expect("Failed to create database");
    setup_update_tables(&db);

    // Update discount for products in premium categories
    db.execute(
        "UPDATE products SET discount = 0.15 WHERE category IN (
            SELECT name FROM categories WHERE is_premium = true
        )",
        (),
    )
    .expect("Failed to update with IN subquery");

    // Verify discounts
    let result = db
        .query("SELECT id, name, discount FROM products ORDER BY id", ())
        .expect("Failed to query");

    let expected = vec![
        (1, "Laptop", 0.15), // Electronics - premium
        (2, "Novel", 0.0),   // Books - not premium
        (3, "Shirt", 0.15),  // Clothing - premium
        (4, "Phone", 0.15),  // Electronics - premium
        (5, "Bread", 0.0),   // Food - not premium
    ];

    let mut idx = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let discount: f64 = row.get(2).unwrap();

        assert_eq!(id, expected[idx].0, "ID mismatch at index {}", idx);
        assert_eq!(name, expected[idx].1, "Name mismatch at index {}", idx);
        assert!(
            (discount - expected[idx].2).abs() < 0.001,
            "Discount mismatch at index {}",
            idx
        );
        idx += 1;
    }
    assert_eq!(idx, 5, "Expected 5 products");
}

/// Test UPDATE with NOT IN subquery
#[test]
fn test_update_with_not_in_subquery() {
    let db = Database::open("memory://update_not_in").expect("Failed to create database");
    setup_update_tables(&db);

    // Update price for products NOT in premium categories (10% discount)
    db.execute(
        "UPDATE products SET price = price * 0.9 WHERE category NOT IN (
            SELECT name FROM categories WHERE is_premium = true
        )",
        (),
    )
    .expect("Failed to update with NOT IN subquery");

    // Verify prices
    let result = db
        .query("SELECT id, name, price FROM products ORDER BY id", ())
        .expect("Failed to query");

    let expected = vec![
        (1, "Laptop", 1000.0), // Electronics - premium (unchanged)
        (2, "Novel", 18.0),    // Books - not premium (20 * 0.9)
        (3, "Shirt", 50.0),    // Clothing - premium (unchanged)
        (4, "Phone", 800.0),   // Electronics - premium (unchanged)
        (5, "Bread", 4.5),     // Food - not premium (5 * 0.9)
    ];

    let mut idx = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();

        assert_eq!(id, expected[idx].0, "ID mismatch at index {}", idx);
        assert_eq!(name, expected[idx].1, "Name mismatch at index {}", idx);
        assert!(
            (price - expected[idx].2).abs() < 0.001,
            "Price mismatch at index {}: expected {}, got {}",
            idx,
            expected[idx].2,
            price
        );
        idx += 1;
    }
    assert_eq!(idx, 5, "Expected 5 products");
}

/// Test UPDATE with empty subquery result
#[test]
fn test_update_with_empty_subquery() {
    let db = Database::open("memory://update_empty").expect("Failed to create database");
    setup_update_tables(&db);

    // Update products in categories that don't exist
    db.execute(
        "UPDATE products SET discount = 0.5 WHERE category IN (
            SELECT name FROM categories WHERE id > 100
        )",
        (),
    )
    .expect("Failed to update");

    // All discounts should still be 0.0
    let result = db
        .query("SELECT discount FROM products", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let discount: f64 = row.get(0).unwrap();
        assert!(
            (discount - 0.0).abs() < 0.001,
            "Discount should be unchanged"
        );
    }
}

/// Test UPDATE with complex condition
#[test]
fn test_update_with_complex_condition() {
    let db = Database::open("memory://update_complex").expect("Failed to create database");
    setup_update_tables(&db);

    // Update discount for premium products with price > 100
    db.execute(
        "UPDATE products SET discount = 0.20 WHERE price > 100 AND category IN (
            SELECT name FROM categories WHERE is_premium = true
        )",
        (),
    )
    .expect("Failed to update");

    // Verify - only Laptop and Phone should have 0.20 discount
    let result = db
        .query("SELECT id, discount FROM products ORDER BY id", ())
        .expect("Failed to query");

    let expected_discounts = vec![
        (1, 0.20), // Laptop - premium, price 1000
        (2, 0.0),  // Novel - not premium
        (3, 0.0),  // Shirt - premium but price 50
        (4, 0.20), // Phone - premium, price 800
        (5, 0.0),  // Bread - not premium
    ];

    let mut idx = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let discount: f64 = row.get(1).unwrap();

        assert_eq!(id, expected_discounts[idx].0);
        assert!(
            (discount - expected_discounts[idx].1).abs() < 0.001,
            "Discount mismatch for id {}: expected {}, got {}",
            id,
            expected_discounts[idx].1,
            discount
        );
        idx += 1;
    }
}

/// Test UPDATE multiple columns with subquery
#[test]
fn test_update_multiple_columns() {
    let db = Database::open("memory://update_multi").expect("Failed to create database");
    setup_update_tables(&db);

    // Update both price and discount for non-premium products
    db.execute(
        "UPDATE products SET price = price * 0.95, discount = 0.05 WHERE category NOT IN (
            SELECT name FROM categories WHERE is_premium = true
        )",
        (),
    )
    .expect("Failed to update");

    // Check Books and Food products
    let result = db
        .query(
            "SELECT id, price, discount FROM products WHERE id IN (2, 5) ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut rows: Vec<(i64, f64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let price: f64 = row.get(1).unwrap();
        let discount: f64 = row.get(2).unwrap();
        rows.push((id, price, discount));
    }

    assert_eq!(rows.len(), 2);
    // Novel: 20 * 0.95 = 19.0
    assert_eq!(rows[0].0, 2);
    assert!((rows[0].1 - 19.0).abs() < 0.001);
    assert!((rows[0].2 - 0.05).abs() < 0.001);
    // Bread: 5 * 0.95 = 4.75
    assert_eq!(rows[1].0, 5);
    assert!((rows[1].1 - 4.75).abs() < 0.001);
    assert!((rows[1].2 - 0.05).abs() < 0.001);
}
