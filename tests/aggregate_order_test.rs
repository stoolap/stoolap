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

//! Aggregate Ordering Tests
//!
//! Tests aggregation with ORDER BY and LIMIT

use stoolap::Database;

fn setup_sales_table(db: &Database) {
    db.execute(
        "CREATE TABLE sales (
            id INTEGER,
            category TEXT,
            product TEXT,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO sales (id, category, product, amount) VALUES (1, 'Electronics', 'Laptop', 1200.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (2, 'Electronics', 'Phone', 800.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (3, 'Electronics', 'Laptop', 1500.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (4, 'Clothing', 'Shirt', 50.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (5, 'Clothing', 'Pants', 80.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (6, 'Clothing', 'Shirt', 45.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (7, 'Books', 'Fiction', 25.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (8, 'Books', 'Non-Fiction', 35.0)",
        "INSERT INTO sales (id, category, product, amount) VALUES (9, 'Books', 'Fiction', 30.0)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

/// Test basic aggregation with SUM
#[test]
fn test_sum_by_category() {
    let db = Database::open("memory://agg_sum_cat").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount) FROM sales GROUP BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut found_electronics = false;
    let mut found_clothing = false;
    let mut found_books = false;

    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();

        match category.as_str() {
            "Electronics" => {
                assert!(
                    (total - 3500.0).abs() < 0.01,
                    "Electronics total should be 3500.0, got {}",
                    total
                );
                found_electronics = true;
            }
            "Clothing" => {
                assert!(
                    (total - 175.0).abs() < 0.01,
                    "Clothing total should be 175.0, got {}",
                    total
                );
                found_clothing = true;
            }
            "Books" => {
                assert!(
                    (total - 90.0).abs() < 0.01,
                    "Books total should be 90.0, got {}",
                    total
                );
                found_books = true;
            }
            _ => panic!("Unexpected category: {}", category),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 categories");
    assert!(found_electronics, "Electronics not found");
    assert!(found_clothing, "Clothing not found");
    assert!(found_books, "Books not found");
}

/// Test aggregation with AVG
#[test]
fn test_avg_by_product() {
    let db = Database::open("memory://agg_avg_prod").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT product, AVG(amount) FROM sales GROUP BY product",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let product: String = row.get(0).unwrap();
        let avg: f64 = row.get(1).unwrap();

        match product.as_str() {
            "Laptop" => {
                assert!(
                    (avg - 1350.0).abs() < 0.01,
                    "Laptop avg should be 1350.0, got {}",
                    avg
                );
            }
            "Phone" => {
                assert!(
                    (avg - 800.0).abs() < 0.01,
                    "Phone avg should be 800.0, got {}",
                    avg
                );
            }
            "Shirt" => {
                assert!(
                    (avg - 47.5).abs() < 0.01,
                    "Shirt avg should be 47.5, got {}",
                    avg
                );
            }
            "Pants" => {
                assert!(
                    (avg - 80.0).abs() < 0.01,
                    "Pants avg should be 80.0, got {}",
                    avg
                );
            }
            "Fiction" => {
                assert!(
                    (avg - 27.5).abs() < 0.01,
                    "Fiction avg should be 27.5, got {}",
                    avg
                );
            }
            "Non-Fiction" => {
                assert!(
                    (avg - 35.0).abs() < 0.01,
                    "Non-Fiction avg should be 35.0, got {}",
                    avg
                );
            }
            _ => panic!("Unexpected product: {}", product),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 6, "Expected 6 products");
}

/// Test aggregation with LIMIT
#[test]
fn test_aggregation_with_limit() {
    let db = Database::open("memory://agg_limit").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT product, AVG(amount) FROM sales GROUP BY product LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _product: String = row.get(0).unwrap();
        let _avg: f64 = row.get(1).unwrap();
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 rows with LIMIT 3");
}

/// Test COUNT aggregation
#[test]
fn test_count_by_category() {
    let db = Database::open("memory://agg_count_cat").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query("SELECT category, COUNT(*) FROM sales GROUP BY category", ())
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();

        match category.as_str() {
            "Electronics" => {
                assert_eq!(count, 3, "Electronics should have 3 rows");
            }
            "Clothing" => {
                assert_eq!(count, 3, "Clothing should have 3 rows");
            }
            "Books" => {
                assert_eq!(count, 3, "Books should have 3 rows");
            }
            _ => panic!("Unexpected category: {}", category),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 categories");
}

/// Test MIN aggregation
#[test]
fn test_min_by_category() {
    let db = Database::open("memory://agg_min_cat").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, MIN(amount) FROM sales GROUP BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let min_val: f64 = row.get(1).unwrap();

        match category.as_str() {
            "Electronics" => {
                assert!(
                    (min_val - 800.0).abs() < 0.01,
                    "Electronics min should be 800.0, got {}",
                    min_val
                );
            }
            "Clothing" => {
                assert!(
                    (min_val - 45.0).abs() < 0.01,
                    "Clothing min should be 45.0, got {}",
                    min_val
                );
            }
            "Books" => {
                assert!(
                    (min_val - 25.0).abs() < 0.01,
                    "Books min should be 25.0, got {}",
                    min_val
                );
            }
            _ => panic!("Unexpected category: {}", category),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 categories");
}

/// Test MAX aggregation
#[test]
fn test_max_by_category() {
    let db = Database::open("memory://agg_max_cat").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, MAX(amount) FROM sales GROUP BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let max_val: f64 = row.get(1).unwrap();

        match category.as_str() {
            "Electronics" => {
                assert!(
                    (max_val - 1500.0).abs() < 0.01,
                    "Electronics max should be 1500.0, got {}",
                    max_val
                );
            }
            "Clothing" => {
                assert!(
                    (max_val - 80.0).abs() < 0.01,
                    "Clothing max should be 80.0, got {}",
                    max_val
                );
            }
            "Books" => {
                assert!(
                    (max_val - 35.0).abs() < 0.01,
                    "Books max should be 35.0, got {}",
                    max_val
                );
            }
            _ => panic!("Unexpected category: {}", category),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 categories");
}

/// Test multiple aggregates in one query
#[test]
fn test_multiple_aggregates() {
    let db = Database::open("memory://agg_multi").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount)
             FROM sales GROUP BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        let sum: f64 = row.get(2).unwrap();
        let avg: f64 = row.get(3).unwrap();
        let min: f64 = row.get(4).unwrap();
        let max: f64 = row.get(5).unwrap();

        match category.as_str() {
            "Electronics" => {
                assert_eq!(count, 3);
                assert!((sum - 3500.0).abs() < 0.01);
                assert!((avg - 1166.67).abs() < 0.01);
                assert!((min - 800.0).abs() < 0.01);
                assert!((max - 1500.0).abs() < 0.01);
            }
            "Clothing" => {
                assert_eq!(count, 3);
                assert!((sum - 175.0).abs() < 0.01);
                assert!((avg - 58.33).abs() < 0.01);
                assert!((min - 45.0).abs() < 0.01);
                assert!((max - 80.0).abs() < 0.01);
            }
            "Books" => {
                assert_eq!(count, 3);
                assert!((sum - 90.0).abs() < 0.01);
                assert!((avg - 30.0).abs() < 0.01);
                assert!((min - 25.0).abs() < 0.01);
                assert!((max - 35.0).abs() < 0.01);
            }
            _ => panic!("Unexpected category: {}", category),
        }
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 categories");
}

/// Test aggregation without GROUP BY
#[test]
fn test_aggregate_no_group_by() {
    let db = Database::open("memory://agg_no_group").expect("Failed to create database");
    setup_sales_table(&db);

    let total: f64 = db
        .query_one("SELECT SUM(amount) FROM sales", ())
        .expect("Failed to query");

    // Total should be 3500 + 175 + 90 = 3765
    assert!(
        (total - 3765.0).abs() < 0.01,
        "Total should be 3765.0, got {}",
        total
    );

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM sales", ())
        .expect("Failed to query");
    assert_eq!(count, 9, "Total count should be 9");

    let avg: f64 = db
        .query_one("SELECT AVG(amount) FROM sales", ())
        .expect("Failed to query");
    // 3765 / 9 = 418.33...
    assert!(
        (avg - 418.33).abs() < 0.01,
        "Average should be ~418.33, got {}",
        avg
    );
}
