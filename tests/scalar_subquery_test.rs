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

//! Scalar Subquery Tests
//!
//! Tests scalar subqueries in WHERE clauses

use stoolap::Database;

fn setup_products_table(db: &Database) {
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price FLOAT,
            category TEXT
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "INSERT INTO products (id, name, price, category) VALUES
        (1, 'Laptop', 1000.0, 'Electronics'),
        (2, 'Mouse', 25.0, 'Electronics'),
        (3, 'Book', 15.0, 'Books'),
        (4, 'Phone', 800.0, 'Electronics'),
        (5, 'Pen', 2.0, 'Stationery')",
        (),
    )
    .expect("Failed to insert products");
}

/// Test WHERE column > (SELECT AVG(...))
#[test]
fn test_scalar_subquery_greater_than_avg() {
    let db = Database::open("memory://scalar_avg").expect("Failed to create database");
    setup_products_table(&db);

    // Average price is (1000 + 25 + 15 + 800 + 2) / 5 = 368.4
    // So we expect products with price > 368.4: Laptop and Phone
    let result = db
        .query(
            "SELECT id, name, price
             FROM products
             WHERE price > (SELECT AVG(price) FROM products)
             ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut products: Vec<(i64, String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();
        products.push((id, name, price));
    }

    assert_eq!(products.len(), 2, "Expected 2 products above average");
    assert_eq!(products[0], (1, "Laptop".to_string(), 1000.0));
    assert_eq!(products[1], (4, "Phone".to_string(), 800.0));
}

/// Test WHERE column = (SELECT MAX(...))
#[test]
fn test_scalar_subquery_equals_max() {
    let db = Database::open("memory://scalar_max").expect("Failed to create database");
    setup_products_table(&db);

    let result = db
        .query(
            "SELECT id, name, price
             FROM products
             WHERE price = (SELECT MAX(price) FROM products)",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        assert_eq!(id, 1, "Expected id 1 for max price product");
        assert_eq!(name, "Laptop", "Expected Laptop with max price");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 product with max price");
}

/// Test WHERE column < (SELECT MIN(...) WHERE condition)
#[test]
fn test_scalar_subquery_less_than_min() {
    let db = Database::open("memory://scalar_min").expect("Failed to create database");
    setup_products_table(&db);

    // Min price in Electronics is 25.0 (Mouse)
    // So we expect products with price < 25.0: Book and Pen
    let result = db
        .query(
            "SELECT id, name, price
             FROM products
             WHERE price < (SELECT MIN(price) FROM products WHERE category = 'Electronics')
             ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut products: Vec<(i64, String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();
        products.push((id, name, price));
    }

    assert_eq!(
        products.len(),
        2,
        "Expected 2 products below Electronics min"
    );
    assert!(products.iter().any(|(_, name, _)| name == "Book"));
    assert!(products.iter().any(|(_, name, _)| name == "Pen"));
}

/// Test scalar subquery returning NULL (no rows)
#[test]
fn test_scalar_subquery_null_result() {
    let db = Database::open("memory://scalar_null").expect("Failed to create database");
    setup_products_table(&db);

    // Subquery returns no rows (NULL), so comparison should return no results
    let result = db
        .query(
            "SELECT id, name
             FROM products
             WHERE price = (SELECT MAX(price) FROM products WHERE category = 'NonExistent')",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 rows for NULL comparison");
}

/// Test scalar subquery in DELETE
#[test]
fn test_scalar_subquery_in_delete() {
    let db = Database::open("memory://scalar_delete").expect("Failed to create database");

    // Create orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            amount FLOAT,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create orders table");

    db.execute(
        "INSERT INTO orders (id, amount, status) VALUES
        (1, 100.0, 'pending'),
        (2, 500.0, 'pending'),
        (3, 200.0, 'completed'),
        (4, 1000.0, 'pending'),
        (5, 50.0, 'completed')",
        (),
    )
    .expect("Failed to insert orders");

    // Average of completed orders: (200 + 50) / 2 = 125
    // Should delete orders with amount < 125: orders 1 and 5
    db.execute(
        "DELETE FROM orders
         WHERE amount < (SELECT AVG(amount) FROM orders WHERE status = 'completed')",
        (),
    )
    .expect("Failed to execute DELETE with scalar subquery");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders", ())
        .expect("Failed to count orders");

    assert_eq!(count, 3, "Expected 3 orders after DELETE");
}

/// Test scalar subquery in UPDATE
#[test]
fn test_scalar_subquery_in_update() {
    let db = Database::open("memory://scalar_update").expect("Failed to create database");

    // Create orders table
    db.execute(
        "CREATE TABLE orders2 (
            id INTEGER PRIMARY KEY,
            amount FLOAT,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create orders table");

    db.execute(
        "INSERT INTO orders2 (id, amount, status) VALUES
        (1, 500.0, 'pending'),
        (2, 200.0, 'pending'),
        (3, 1000.0, 'pending')",
        (),
    )
    .expect("Failed to insert orders");

    // Average = (500 + 200 + 1000) / 3 = 566.67
    // Should update orders with amount > 566.67: order 3 (1000)
    db.execute(
        "UPDATE orders2
         SET status = 'high_value'
         WHERE amount > (SELECT AVG(amount) FROM orders2)",
        (),
    )
    .expect("Failed to execute UPDATE with scalar subquery");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders2 WHERE status = 'high_value'",
            (),
        )
        .expect("Failed to count high_value orders");

    assert_eq!(count, 1, "Expected 1 high_value order");
}

/// Test complex scalar subquery with aggregation
#[test]
fn test_complex_scalar_subquery() {
    let db = Database::open("memory://scalar_complex").expect("Failed to create database");
    setup_products_table(&db);

    // Average price for Electronics: (1000 + 25 + 800) / 3 = 608.33
    // Products with price >= 608.33: Laptop and Phone
    let result = db
        .query(
            "SELECT name, price
             FROM products
             WHERE price >= (
                 SELECT AVG(price)
                 FROM products
                 WHERE category = 'Electronics'
             )
             ORDER BY price DESC",
            (),
        )
        .expect("Failed to execute complex scalar subquery");

    let mut products: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let price: f64 = row.get(1).unwrap();
        products.push((name, price));
    }

    assert_eq!(
        products.len(),
        2,
        "Expected 2 products above Electronics average"
    );
    assert_eq!(products[0].0, "Laptop");
    assert_eq!(products[1].0, "Phone");
}

// ========================
// SELECT Column Scalar Subqueries
// ========================

/// Test scalar subquery in SELECT column (basic)
#[test]
fn test_scalar_subquery_in_select_basic() {
    let db = Database::open("memory://scalar_select_basic").expect("Failed to create database");
    setup_products_table(&db);

    // Get max price alongside product info
    let result = db
        .query(
            "SELECT id, name, price, (SELECT MAX(price) FROM products) AS max_price
             FROM products
             ORDER BY id
             LIMIT 3",
            (),
        )
        .expect("Failed to execute query");

    let mut products: Vec<(i64, String, f64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();
        let max_price: f64 = row.get(3).unwrap();
        products.push((id, name, price, max_price));
    }

    assert_eq!(products.len(), 3, "Expected 3 products");
    // All rows should have the same max_price (1000.0)
    for (_, _, _, max_price) in &products {
        assert_eq!(*max_price, 1000.0, "Max price should be 1000.0");
    }
}

/// Test scalar subquery in SELECT with alias
#[test]
fn test_scalar_subquery_in_select_with_alias() {
    let db = Database::open("memory://scalar_select_alias").expect("Failed to create database");
    setup_products_table(&db);

    let result = db
        .query(
            "SELECT name,
                    (SELECT AVG(price) FROM products) AS avg_price,
                    (SELECT COUNT(*) FROM products) AS total_count
             FROM products
             WHERE id = 1",
            (),
        )
        .expect("Failed to execute query");

    let row = result.into_iter().next().unwrap().unwrap();
    let name: String = row.get(0).unwrap();
    let avg_price: f64 = row.get(1).unwrap();
    let total_count: i64 = row.get(2).unwrap();

    assert_eq!(name, "Laptop");
    // Avg = (1000 + 25 + 15 + 800 + 2) / 5 = 368.4
    assert!((avg_price - 368.4).abs() < 0.1, "Expected avg_price ~368.4");
    assert_eq!(total_count, 5);
}

/// Test scalar subquery in SELECT without FROM clause
#[test]
fn test_scalar_subquery_in_select_no_from() {
    let db = Database::open("memory://scalar_select_no_from").expect("Failed to create database");
    setup_products_table(&db);

    let result = db
        .query(
            "SELECT (SELECT MAX(price) FROM products) AS max_val,
                    (SELECT MIN(price) FROM products) AS min_val",
            (),
        )
        .expect("Failed to execute query");

    let row = result.into_iter().next().unwrap().unwrap();
    let max_val: f64 = row.get(0).unwrap();
    let min_val: f64 = row.get(1).unwrap();

    assert_eq!(max_val, 1000.0);
    assert_eq!(min_val, 2.0);
}

/// Test scalar subquery in SELECT with expression
#[test]
fn test_scalar_subquery_in_select_with_expression() {
    let db = Database::open("memory://scalar_select_expr").expect("Failed to create database");
    setup_products_table(&db);

    // Calculate percentage of max price
    let result = db
        .query(
            "SELECT name, price,
                    price / (SELECT MAX(price) FROM products) * 100 AS pct_of_max
             FROM products
             ORDER BY price DESC
             LIMIT 2",
            (),
        )
        .expect("Failed to execute query");

    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let price: f64 = row.get(1).unwrap();
        let pct: f64 = row.get(2).unwrap();
        rows.push((name, price, pct));
    }

    assert_eq!(rows.len(), 2);
    // Laptop is 100% of max (1000/1000 * 100)
    assert!((rows[0].2 - 100.0).abs() < 0.1);
    // Phone is 80% of max (800/1000 * 100)
    assert!((rows[1].2 - 80.0).abs() < 0.1);
}

/// Test scalar subquery in SELECT returning NULL
#[test]
fn test_scalar_subquery_in_select_null() {
    let db = Database::open("memory://scalar_select_null").expect("Failed to create database");
    setup_products_table(&db);

    // Subquery returns no rows (NULL)
    let result = db
        .query(
            "SELECT id, (SELECT MAX(price) FROM products WHERE category = 'NonExistent') AS max_price
             FROM products
             LIMIT 1",
            (),
        )
        .expect("Failed to execute query");

    let row = result.into_iter().next().unwrap().unwrap();
    let id: i64 = row.get(0).unwrap();
    let max_price: Option<f64> = row.get(1).ok();

    assert_eq!(id, 1);
    assert!(
        max_price.is_none(),
        "Expected NULL for non-existent category"
    );
}

/// Test scalar subquery in SELECT with WHERE on outer query
#[test]
fn test_scalar_subquery_select_with_where() {
    let db = Database::open("memory://scalar_select_where").expect("Failed to create database");
    setup_products_table(&db);

    // Get category average for each product's category (using the same category)
    let result = db
        .query(
            "SELECT name, category,
                    (SELECT AVG(price) FROM products WHERE category = 'Electronics') AS electronics_avg
             FROM products
             WHERE category = 'Electronics'
             ORDER BY name",
            (),
        )
        .expect("Failed to execute query");

    let mut rows: Vec<(String, String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let category: String = row.get(1).unwrap();
        let avg: f64 = row.get(2).unwrap();
        rows.push((name, category, avg));
    }

    assert_eq!(rows.len(), 3, "Expected 3 Electronics products");
    // Electronics average = (1000 + 25 + 800) / 3 = 608.33
    for (_, _, avg) in &rows {
        assert!(
            (*avg - 608.33).abs() < 0.1,
            "Expected electronics avg ~608.33"
        );
    }
}

/// Test multiple scalar subqueries in SELECT
#[test]
fn test_multiple_scalar_subqueries_in_select() {
    let db = Database::open("memory://scalar_select_multi").expect("Failed to create database");
    setup_products_table(&db);

    let result = db
        .query(
            "SELECT
                (SELECT MAX(price) FROM products) AS max_p,
                (SELECT MIN(price) FROM products) AS min_p,
                (SELECT COUNT(*) FROM products) AS cnt,
                (SELECT SUM(price) FROM products) AS total",
            (),
        )
        .expect("Failed to execute query");

    let row = result.into_iter().next().unwrap().unwrap();
    let max_p: f64 = row.get(0).unwrap();
    let min_p: f64 = row.get(1).unwrap();
    let cnt: i64 = row.get(2).unwrap();
    let total: f64 = row.get(3).unwrap();

    assert_eq!(max_p, 1000.0);
    assert_eq!(min_p, 2.0);
    assert_eq!(cnt, 5);
    assert_eq!(total, 1842.0); // 1000 + 25 + 15 + 800 + 2
}
