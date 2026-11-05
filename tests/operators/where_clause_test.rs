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

//! WHERE Clause Tests
//!
//! Tests various WHERE clause scenarios: comparisons, logical operators, LIKE, IN, etc.

use stoolap::Database;

fn setup_products_table(db: &Database) {
    // Create a test table
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            tags TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    let inserts = [
        "INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 1200.00, true, 'portable,computer')",
        "INSERT INTO products VALUES (2, 'Smartphone', 'Electronics', 800.00, true, 'mobile,phone')",
        "INSERT INTO products VALUES (3, 'Headphones', 'Electronics', 150.00, true, 'audio,accessory')",
        "INSERT INTO products VALUES (4, 'T-shirt', 'Clothing', 25.00, true, 'apparel,cotton')",
        "INSERT INTO products VALUES (5, 'Jeans', 'Clothing', 50.00, false, 'apparel,denim')",
        "INSERT INTO products VALUES (6, 'Sneakers', 'Footwear', 80.00, true, 'shoes,casual')",
        "INSERT INTO products VALUES (7, 'Boots', 'Footwear', 120.00, false, 'shoes,winter')",
        "INSERT INTO products VALUES (8, 'Desk', 'Furniture', 250.00, true, 'home,office')",
        "INSERT INTO products VALUES (9, 'Chair', 'Furniture', 150.00, true, 'home,office')",
        "INSERT INTO products VALUES (10, 'Bookshelf', 'Furniture', 180.00, false, 'home,storage')",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

fn count_rows(db: &Database, query: &str) -> i64 {
    let result = db.query(query, ()).expect("Failed to execute query");
    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    count
}

// Basic comparisons

#[test]
fn test_equality_operator() {
    let db = Database::open("memory://where_eq").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products WHERE category = 'Electronics'");
    assert_eq!(count, 3, "Expected 3 Electronics products");
}

#[test]
fn test_inequality_operator() {
    let db = Database::open("memory://where_neq").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE category != 'Electronics'",
    );
    assert_eq!(count, 7, "Expected 7 non-Electronics products");
}

#[test]
fn test_greater_than_operator() {
    let db = Database::open("memory://where_gt").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products WHERE price > 200");
    assert_eq!(count, 3, "Expected 3 products with price > 200");
}

#[test]
fn test_greater_than_or_equal_operator() {
    let db = Database::open("memory://where_gte").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products WHERE price >= 150");
    assert_eq!(count, 6, "Expected 6 products with price >= 150");
}

#[test]
fn test_less_than_operator() {
    let db = Database::open("memory://where_lt").expect("Failed to create database");
    setup_products_table(&db);

    // T-shirt (25), Jeans (50), Sneakers (80)
    let count = count_rows(&db, "SELECT * FROM products WHERE price < 100");
    assert_eq!(count, 3, "Expected 3 products with price < 100");
}

#[test]
fn test_less_than_or_equal_operator() {
    let db = Database::open("memory://where_lte").expect("Failed to create database");
    setup_products_table(&db);

    // T-shirt (25), Jeans (50), Sneakers (80), Headphones (150), Boots (120), Chair (150)
    let count = count_rows(&db, "SELECT * FROM products WHERE price <= 150");
    assert_eq!(count, 6, "Expected 6 products with price <= 150");
}

// Boolean operators

#[test]
fn test_boolean_true_comparison() {
    let db = Database::open("memory://where_bool_true").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products WHERE in_stock = true");
    assert_eq!(count, 7, "Expected 7 in-stock products");
}

#[test]
fn test_boolean_false_comparison() {
    let db = Database::open("memory://where_bool_false").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products WHERE in_stock = false");
    assert_eq!(count, 3, "Expected 3 out-of-stock products");
}

// Logical operators

#[test]
fn test_and_operator() {
    let db = Database::open("memory://where_and").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE category = 'Electronics' AND price > 500",
    );
    assert_eq!(count, 2, "Expected 2 Electronics products with price > 500");
}

#[test]
fn test_or_operator() {
    let db = Database::open("memory://where_or").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE category = 'Furniture' OR category = 'Footwear'",
    );
    assert_eq!(count, 5, "Expected 5 Furniture or Footwear products");
}

#[test]
fn test_complex_and_or_combination() {
    let db = Database::open("memory://where_complex").expect("Failed to create database");
    setup_products_table(&db);

    // Laptop (Electronics, 1200), Chair (Furniture, 150), Bookshelf (Furniture, 180)
    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE (category = 'Electronics' AND price > 1000) OR (category = 'Furniture' AND price < 200)",
    );
    assert_eq!(count, 3, "Expected 3 products matching complex condition");
}

// LIKE operator

#[test]
fn test_like_starts_with() {
    let db = Database::open("memory://where_like_start").expect("Failed to create database");
    setup_products_table(&db);

    // Boots, Bookshelf
    let count = count_rows(&db, "SELECT * FROM products WHERE name LIKE 'B%'");
    assert_eq!(count, 2, "Expected 2 products starting with 'B'");
}

#[test]
fn test_like_contains() {
    let db = Database::open("memory://where_like_contains").expect("Failed to create database");
    setup_products_table(&db);

    // Laptop, Smartphone, Headphones, Boots, Bookshelf
    let count = count_rows(&db, "SELECT * FROM products WHERE name LIKE '%o%'");
    assert_eq!(count, 5, "Expected 5 products containing 'o'");
}

#[test]
fn test_like_ends_with() {
    let db = Database::open("memory://where_like_end").expect("Failed to create database");
    setup_products_table(&db);

    // Headphones, Jeans, Boots, Sneakers
    let count = count_rows(&db, "SELECT * FROM products WHERE name LIKE '%s'");
    assert_eq!(count, 4, "Expected 4 products ending with 's'");
}

#[test]
fn test_like_single_char() {
    let db = Database::open("memory://where_like_char").expect("Failed to create database");
    setup_products_table(&db);

    // Desk
    let count = count_rows(&db, "SELECT * FROM products WHERE name LIKE 'Des_'");
    assert_eq!(count, 1, "Expected 1 product matching 'Des_'");
}

// NOT operator

#[test]
fn test_not_operator() {
    let db = Database::open("memory://where_not").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE NOT category = 'Electronics'",
    );
    assert_eq!(count, 7, "Expected 7 non-Electronics products");
}

#[test]
fn test_not_with_other_conditions() {
    let db = Database::open("memory://where_not_and").expect("Failed to create database");
    setup_products_table(&db);

    // Boots (120), Bookshelf (180)
    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE NOT in_stock AND price > 100",
    );
    assert_eq!(
        count, 2,
        "Expected 2 out-of-stock products with price > 100"
    );
}

// IN operator

#[test]
fn test_in_operator() {
    let db = Database::open("memory://where_in").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE category IN ('Electronics', 'Furniture')",
    );
    assert_eq!(count, 6, "Expected 6 Electronics or Furniture products");
}

#[test]
fn test_not_in_operator() {
    let db = Database::open("memory://where_not_in").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE category NOT IN ('Electronics', 'Furniture')",
    );
    assert_eq!(count, 4, "Expected 4 non-Electronics/Furniture products");
}

// ORDER BY and LIMIT

#[test]
fn test_order_by_asc_with_limit() {
    let db = Database::open("memory://where_order_asc").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products ORDER BY price ASC LIMIT 3");
    assert_eq!(count, 3, "Expected 3 products with LIMIT");
}

#[test]
fn test_order_by_desc_with_limit() {
    let db = Database::open("memory://where_order_desc").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(&db, "SELECT * FROM products ORDER BY price DESC LIMIT 2");
    assert_eq!(count, 2, "Expected 2 products with LIMIT");
}

#[test]
fn test_where_with_order_by_and_limit() {
    let db = Database::open("memory://where_order_limit").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE price > 100 ORDER BY price DESC LIMIT 3",
    );
    assert_eq!(
        count, 3,
        "Expected 3 products with WHERE, ORDER BY and LIMIT"
    );
}

// Complex combined queries

#[test]
fn test_complex_query_1() {
    let db = Database::open("memory://where_complex1").expect("Failed to create database");
    setup_products_table(&db);

    // Laptop, Smartphone, Desk
    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE (category = 'Electronics' OR category = 'Furniture') AND price > 200 AND in_stock = true",
    );
    assert_eq!(count, 3, "Expected 3 products matching complex query 1");
}

#[test]
fn test_complex_query_2() {
    let db = Database::open("memory://where_complex2").expect("Failed to create database");
    setup_products_table(&db);

    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE name LIKE '%e%' AND price BETWEEN 100 AND 300 AND category != 'Clothing'",
    );
    assert_eq!(count, 3, "Expected 3 products matching complex query 2");
}

#[test]
fn test_complex_query_3() {
    let db = Database::open("memory://where_complex3").expect("Failed to create database");
    setup_products_table(&db);

    // Laptop (1200, true), T-shirt (25, true), Sneakers (80, true)
    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE (price < 100 OR price > 1000) AND in_stock = true ORDER BY price DESC LIMIT 4",
    );
    assert_eq!(count, 3, "Expected 3 products matching complex query 3");
}

// BETWEEN operator

#[test]
fn test_between_operator() {
    let db = Database::open("memory://where_between").expect("Failed to create database");
    setup_products_table(&db);

    // Headphones (150), Boots (120), Chair (150), Bookshelf (180)
    let count = count_rows(
        &db,
        "SELECT * FROM products WHERE price BETWEEN 100 AND 200",
    );
    assert_eq!(
        count, 4,
        "Expected 4 products with price BETWEEN 100 AND 200"
    );
}

// OR with indexed columns - tests OR index union optimization
#[test]
fn test_or_with_indexed_columns() {
    let db = Database::open("memory://where_or_indexed").expect("Failed to create database");

    // Create table with indexes on category and price
    db.execute(
        "CREATE TABLE indexed_products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        )",
        (),
    )
    .expect("Failed to create table");

    // Create indexes for OR optimization
    db.execute(
        "CREATE INDEX idx_category ON indexed_products(category)",
        (),
    )
    .expect("Failed to create category index");
    db.execute("CREATE INDEX idx_price ON indexed_products(price)", ())
        .expect("Failed to create price index");

    // Insert test data (use .0 for REAL column)
    db.execute(
        "INSERT INTO indexed_products VALUES (1, 'Laptop', 'Electronics', 1200.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO indexed_products VALUES (2, 'Phone', 'Electronics', 800.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO indexed_products VALUES (3, 'Chair', 'Furniture', 150.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO indexed_products VALUES (4, 'Desk', 'Furniture', 300.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO indexed_products VALUES (5, 'Shirt', 'Clothing', 50.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO indexed_products VALUES (6, 'Shoes', 'Footwear', 100.0)",
        (),
    )
    .unwrap();

    // Test OR with same indexed column (should use index union)
    let count = count_rows(
        &db,
        "SELECT * FROM indexed_products WHERE category = 'Electronics' OR category = 'Furniture'",
    );
    assert_eq!(count, 4, "Expected 4 products (Electronics + Furniture)");

    // Test OR with different indexed columns (should use index union)
    let count = count_rows(
        &db,
        "SELECT * FROM indexed_products WHERE category = 'Clothing' OR price > 500.0",
    );
    assert_eq!(count, 3, "Expected 3 products (Clothing OR price > 500)");

    // Test OR with range conditions
    let count = count_rows(
        &db,
        "SELECT * FROM indexed_products WHERE price < 100.0 OR price > 1000.0",
    );
    assert_eq!(
        count, 2,
        "Expected 2 products (price < 100 OR price > 1000)"
    );

    // Test multiple OR conditions
    let count = count_rows(
        &db,
        "SELECT * FROM indexed_products WHERE category = 'Electronics' OR category = 'Furniture' OR category = 'Clothing'",
    );
    assert_eq!(count, 5, "Expected 5 products (3 categories)");
}
