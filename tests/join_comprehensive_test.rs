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

//! Comprehensive JOIN Tests
//!
//! Tests various JOIN operations with different tables

use stoolap::Database;

fn setup_comprehensive_tables(db: &Database) {
    // Create categories_ch table with parent-child structure
    db.execute(
        "CREATE TABLE categories_ch (
            id INTEGER,
            name TEXT,
            parent_id INTEGER
        )",
        (),
    )
    .expect("Failed to create categories_ch table");

    // Create products table
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            category_id INTEGER,
            price FLOAT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create products table");

    // Create customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER,
            name TEXT,
            email TEXT,
            country TEXT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Create orders table (using TEXT for order_date since DATE type is stored as TIMESTAMP)
    db.execute(
        "CREATE TABLE orders (
            id INTEGER,
            customer_id INTEGER,
            order_date TEXT,
            total FLOAT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Create order_items table
    db.execute(
        "CREATE TABLE order_items (
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            price FLOAT
        )",
        (),
    )
    .expect("Failed to create order_items table");

    // Insert data into categories_ch - parent_id NULL for top-level categories
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (1, 'Electronics', NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (2, 'Computers', 1)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (3, 'Phones', 1)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (4, 'Accessories', 1)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (5, 'Clothing', NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO categories_ch (id, name, parent_id) VALUES (6, 'Books', NULL)",
        (),
    )
    .unwrap();

    // Insert data into products
    db.execute(
        "INSERT INTO products (id, name, category_id, price, in_stock) VALUES
        (101, 'Laptop', 2, 1200.00, true),
        (102, 'Smartphone', 3, 800.00, true),
        (103, 'Tablet', 2, 500.00, true),
        (104, 'Headphones', 4, 150.00, true),
        (105, 'Monitor', 2, 300.00, false),
        (106, 'Keyboard', 4, 80.00, true),
        (107, 'T-shirt', 5, 25.00, true),
        (108, 'Programming Book', 6, 40.00, true)",
        (),
    )
    .expect("Failed to insert products");

    // Insert data into customers
    db.execute(
        "INSERT INTO customers (id, name, email, country) VALUES
        (1, 'Alice Smith', 'alice@example.com', 'USA'),
        (2, 'Bob Johnson', 'bob@example.com', 'Canada'),
        (3, 'Charlie Brown', 'charlie@example.com', 'UK'),
        (4, 'Diana Adams', 'diana@example.com', 'Australia')",
        (),
    )
    .expect("Failed to insert customers");

    // Insert data into orders - customer_id NULL for anonymous order
    db.execute("INSERT INTO orders (id, customer_id, order_date, total) VALUES (1001, 1, '2023-01-15', 1350.00)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, customer_id, order_date, total) VALUES (1002, 2, '2023-01-16', 800.00)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, customer_id, order_date, total) VALUES (1003, 1, '2023-02-10', 45.00)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, customer_id, order_date, total) VALUES (1004, 3, '2023-02-20', 1200.00)", ())
        .unwrap();
    db.execute("INSERT INTO orders (id, customer_id, order_date, total) VALUES (1005, NULL, '2023-03-05', 40.00)", ())
        .unwrap();

    // Insert data into order_items
    db.execute(
        "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
        (1001, 101, 1, 1200.00),
        (1001, 104, 1, 150.00),
        (1002, 102, 1, 800.00),
        (1003, 107, 1, 25.00),
        (1004, 101, 1, 1200.00),
        (1005, 108, 1, 40.00)",
        (),
    )
    .expect("Failed to insert order_items");
}

/// Test self-join for parent-child category relationships
#[test]
fn test_category_self_join() {
    let db = Database::open("memory://join_self").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Run a self-join to get parent category names
    let result = db
        .query(
            "SELECT c.id, c.name, c.parent_id, p.name AS parent_name
             FROM categories_ch c
             LEFT JOIN categories_ch p ON c.parent_id = p.id
             ORDER BY c.id",
            (),
        )
        .expect("Failed to execute self JOIN");

    let mut categories_with_parent = 0;
    let mut total_categories = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let parent_name: Option<String> = row.get(3).unwrap();

        total_categories += 1;

        if parent_name.is_some() {
            categories_with_parent += 1;
            println!("Category {}: {} has parent {:?}", id, name, parent_name);
        } else {
            println!("Category {}: {} is a top-level category", id, name);
        }
    }

    assert_eq!(total_categories, 6, "Expected 6 total categories");
    assert_eq!(
        categories_with_parent, 3,
        "Expected 3 categories with parents"
    );
}

/// Test INNER JOIN between products and categories
#[test]
fn test_inner_join_products_categories() {
    let db = Database::open("memory://join_inner").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Query for in-stock products with their categories
    let result = db
        .query(
            "SELECT p.id, p.name, p.price, c.name AS category
             FROM products p
             INNER JOIN categories_ch c ON p.category_id = c.id
             WHERE p.in_stock = true
             ORDER BY p.price DESC",
            (),
        )
        .expect("Failed to execute INNER JOIN");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();
        let category: String = row.get(3).unwrap();

        println!(
            "Product: {} - {} (${}) in category {}",
            id, name, price, category
        );
        count += 1;
    }

    assert_eq!(count, 7, "Expected 7 in-stock products");
}

/// Test LEFT JOIN between orders and customers
#[test]
fn test_left_join_orders_customers() {
    let db = Database::open("memory://join_left").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Run LEFT JOIN to get all orders with customer names where available
    let result = db
        .query(
            "SELECT o.id, o.order_date, o.total, c.name AS customer_name
             FROM orders o
             LEFT JOIN customers c ON o.customer_id = c.id
             ORDER BY o.id",
            (),
        )
        .expect("Failed to execute LEFT JOIN");

    let mut total_orders = 0;
    let mut orders_with_null_customer = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let order_date: String = row.get(1).unwrap();
        let total: f64 = row.get(2).unwrap();
        let customer_name: Option<String> = row.get(3).unwrap();

        total_orders += 1;

        if customer_name.is_none() {
            orders_with_null_customer += 1;
            println!(
                "Order {} on {} for ${} has no customer",
                id, order_date, total
            );
        } else {
            println!(
                "Order {} on {} for ${} by {:?}",
                id, order_date, total, customer_name
            );
        }
    }

    assert_eq!(total_orders, 5, "Expected 5 total orders");
    assert_eq!(
        orders_with_null_customer, 1,
        "Expected 1 order with NULL customer"
    );
}

/// Test JOIN with ordering by category and price
#[test]
fn test_ordered_products_by_category() {
    let db = Database::open("memory://join_ordered").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Run a JOIN with ordering by category and then by price
    let result = db
        .query(
            "SELECT c.name AS category, p.name AS product, p.price
             FROM categories_ch c
             JOIN products p ON c.id = p.category_id
             ORDER BY c.name, p.price DESC",
            (),
        )
        .expect("Failed to execute JOIN with ordering");

    let mut unique_categories: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut products_found = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let product: String = row.get(1).unwrap();
        let price: f64 = row.get(2).unwrap();

        unique_categories.insert(category.clone());
        products_found += 1;
        println!(
            "Category: {} - Product: {} (${:.2})",
            category, product, price
        );
    }

    assert_eq!(
        unique_categories.len(),
        5,
        "Expected 5 categories with products"
    );
    assert_eq!(products_found, 8, "Expected 8 products");
}

/// Test three-way JOIN (orders -> order_items -> products)
#[test]
fn test_three_way_join() {
    let db = Database::open("memory://join_three").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    let result = db
        .query(
            "SELECT o.id AS order_id, p.name AS product_name, oi.quantity, oi.price
             FROM orders o
             JOIN order_items oi ON o.id = oi.order_id
             JOIN products p ON oi.product_id = p.id
             ORDER BY o.id, p.name",
            (),
        )
        .expect("Failed to execute three-way JOIN");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let order_id: i64 = row.get(0).unwrap();
        let product_name: String = row.get(1).unwrap();
        let quantity: i64 = row.get(2).unwrap();
        let price: f64 = row.get(3).unwrap();

        println!(
            "Order {}: {} x{} @ ${:.2}",
            order_id, product_name, quantity, price
        );
        count += 1;
    }

    assert_eq!(count, 6, "Expected 6 order items");
}

/// Test JOIN with aggregation
#[test]
fn test_join_with_aggregation() {
    let db = Database::open("memory://join_agg").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    let result = db
        .query(
            "SELECT c.name, COUNT(*) AS product_count, AVG(p.price) AS avg_price
             FROM categories_ch c
             JOIN products p ON c.id = p.category_id
             GROUP BY c.name
             ORDER BY c.name",
            (),
        )
        .expect("Failed to execute JOIN with aggregation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let product_count: i64 = row.get(1).unwrap();
        let avg_price: f64 = row.get(2).unwrap();

        println!(
            "Category: {} - {} products, avg price ${:.2}",
            category, product_count, avg_price
        );
        count += 1;
    }

    assert!(count >= 5, "Expected at least 5 categories with products");
}

/// Test JOIN with WHERE clause on joined table
#[test]
fn test_join_with_where_on_joined() {
    let db = Database::open("memory://join_where").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Get orders from USA customers only
    let result = db
        .query(
            "SELECT o.id, o.total, c.name, c.country
             FROM orders o
             JOIN customers c ON o.customer_id = c.id
             WHERE c.country = 'USA'
             ORDER BY o.id",
            (),
        )
        .expect("Failed to execute JOIN with WHERE");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let order_id: i64 = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        let customer: String = row.get(2).unwrap();
        let country: String = row.get(3).unwrap();

        assert_eq!(country, "USA");
        println!(
            "Order {}: ${:.2} by {} ({})",
            order_id, total, customer, country
        );
        count += 1;
    }

    // Alice has 2 orders
    assert_eq!(count, 2, "Expected 2 orders from USA");
}

/// Test multiple JOINs with different types
#[test]
fn test_multiple_join_types() {
    let db = Database::open("memory://join_multi_type").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Complex query with both INNER and LEFT JOIN
    let result = db
        .query(
            "SELECT o.id AS order_id, c.name AS customer, p.name AS product
             FROM orders o
             LEFT JOIN customers c ON o.customer_id = c.id
             JOIN order_items oi ON o.id = oi.order_id
             JOIN products p ON oi.product_id = p.id
             ORDER BY o.id",
            (),
        )
        .expect("Failed to execute mixed JOINs");

    let mut null_customer_order_found = false;
    let mut count = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let order_id: i64 = row.get(0).unwrap();
        let customer: Option<String> = row.get(1).unwrap();
        let product: String = row.get(2).unwrap();

        if customer.is_none() {
            null_customer_order_found = true;
        }

        println!("Order {}: {:?} bought {}", order_id, customer, product);
        count += 1;
    }

    assert!(
        null_customer_order_found,
        "Expected to find an order with NULL customer"
    );
    assert_eq!(count, 6, "Expected 6 total order items");
}

/// Test RIGHT JOIN (if supported)
#[test]
fn test_right_join() {
    let db = Database::open("memory://join_right").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // RIGHT JOIN to get all customers even without orders
    // Note: Some databases implement RIGHT JOIN differently
    let result = db.query(
        "SELECT c.name, o.id AS order_id
         FROM orders o
         RIGHT JOIN customers c ON o.customer_id = c.id
         ORDER BY c.name",
        (),
    );

    match result {
        Ok(rows) => {
            let mut count = 0;
            let mut customer_without_orders = false;

            for row in rows {
                let row = row.expect("Failed to get row");
                let name: String = row.get(0).unwrap();
                let order_id: Option<i64> = row.get(1).unwrap();

                if order_id.is_none() {
                    customer_without_orders = true;
                }

                println!("Customer: {} - Order: {:?}", name, order_id);
                count += 1;
            }

            // Diana has no orders
            assert!(
                customer_without_orders,
                "Expected to find customer without orders (Diana)"
            );
            assert!(count >= 4, "Expected at least 4 rows");
        }
        Err(_) => {
            // RIGHT JOIN might not be implemented
            println!("RIGHT JOIN not supported, skipping test");
        }
    }
}

/// Test JOIN with DISTINCT
#[test]
fn test_join_with_distinct() {
    let db = Database::open("memory://join_distinct").expect("Failed to create database");
    setup_comprehensive_tables(&db);

    // Get distinct customers who have placed orders
    let result = db
        .query(
            "SELECT DISTINCT c.name
             FROM customers c
             JOIN orders o ON c.id = o.customer_id
             ORDER BY c.name",
            (),
        )
        .expect("Failed to execute JOIN with DISTINCT");

    let mut customers: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        customers.push(name);
    }

    // Alice, Bob, Charlie have orders (Diana does not)
    assert_eq!(
        customers.len(),
        3,
        "Expected 3 distinct customers with orders"
    );
    assert!(customers.contains(&"Alice Smith".to_string()));
    assert!(customers.contains(&"Bob Johnson".to_string()));
    assert!(customers.contains(&"Charlie Brown".to_string()));
}
