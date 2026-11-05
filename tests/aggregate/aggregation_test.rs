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

//! Aggregation Tests
//!
//! Tests aggregation functions: COUNT, SUM, AVG, MIN, MAX, GROUP BY, HAVING, DISTINCT

use stoolap::Database;

fn setup_sales_table(db: &Database) {
    db.execute("CREATE TABLE sales_aggr (id INTEGER, product TEXT, category TEXT, amount FLOAT, region TEXT)", ())
        .expect("Failed to create table");

    let inserts = [
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (1, 'Laptop', 'Electronics', 1200.0, 'North')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (2, 'Smartphone', 'Electronics', 800.0, 'North')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (3, 'TV', 'Electronics', 1500.0, 'South')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (4, 'Chair', 'Furniture', 150.0, 'East')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (5, 'Table', 'Furniture', 450.0, 'East')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (6, 'Sofa', 'Furniture', 950.0, 'West')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (7, 'Shirt', 'Clothing', 35.0, 'North')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (8, 'Jeans', 'Clothing', 60.0, 'South')",
        "INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (9, 'Shoes', 'Clothing', 90.0, 'West')",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

#[test]
fn test_count_star() {
    let db = Database::open("memory://aggregation_count_star").expect("Failed to create database");
    setup_sales_table(&db);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM sales_aggr", ()).unwrap();
    assert_eq!(count, 9, "Expected count of 9");
}

#[test]
fn test_count_with_group_by() {
    let db = Database::open("memory://aggregation_group_by").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, COUNT(*) FROM sales_aggr GROUP BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;
        let _category: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        assert_eq!(count, 3, "Each category should have 3 items");
    }
    assert_eq!(row_count, 3, "Expected 3 category groups");
}

#[test]
fn test_count_distinct() {
    let db =
        Database::open("memory://aggregation_count_distinct").expect("Failed to create database");
    setup_sales_table(&db);

    // Insert a duplicate product
    db.execute("INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (10, 'Laptop', 'Electronics', 1200.0, 'North')", ())
        .expect("Failed to insert duplicate");

    let distinct_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT product) FROM sales_aggr", ())
        .unwrap();
    assert_eq!(distinct_count, 9, "Expected 9 distinct products");
}

#[test]
fn test_group_by_with_having() {
    let db = Database::open("memory://aggregation_having").expect("Failed to create database");
    setup_sales_table(&db);

    // Insert a duplicate to make Electronics have 4 items
    db.execute("INSERT INTO sales_aggr (id, product, category, amount, region) VALUES (10, 'Laptop', 'Electronics', 1200.0, 'North')", ())
        .expect("Failed to insert duplicate");

    // Use alias in HAVING clause for compatibility
    let result = db
        .query(
            "SELECT category, COUNT(*) as cnt FROM sales_aggr GROUP BY category HAVING cnt > 2",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut categories = std::collections::HashSet::new();

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;
        let category: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        assert!(
            count > 2,
            "Expected count > 2 for category {}, got {}",
            category,
            count
        );
        categories.insert(category);
    }

    assert_eq!(
        row_count, 3,
        "Expected 3 rows (Electronics, Furniture, Clothing)"
    );
    assert!(
        categories.contains("Electronics"),
        "Expected Electronics in results"
    );
    assert!(
        categories.contains("Furniture"),
        "Expected Furniture in results"
    );
    assert!(
        categories.contains("Clothing"),
        "Expected Clothing in results"
    );
}

#[test]
fn test_sum_function() {
    let db = Database::open("memory://aggregation_sum").expect("Failed to create database");
    setup_sales_table(&db);

    let total: f64 = db
        .query_one("SELECT SUM(amount) FROM sales_aggr", ())
        .unwrap();
    // 1200 + 800 + 1500 + 150 + 450 + 950 + 35 + 60 + 90 = 5235
    assert!(
        (total - 5235.0).abs() < 0.01,
        "Expected sum of 5235, got {}",
        total
    );
}

#[test]
fn test_avg_function() {
    let db = Database::open("memory://aggregation_avg").expect("Failed to create database");
    setup_sales_table(&db);

    let avg: f64 = db
        .query_one("SELECT AVG(amount) FROM sales_aggr", ())
        .unwrap();
    // 5235 / 9 = 581.67
    assert!(
        (avg - 581.67).abs() < 0.01,
        "Expected avg of ~581.67, got {}",
        avg
    );
}

#[test]
fn test_min_function() {
    let db = Database::open("memory://aggregation_min").expect("Failed to create database");
    setup_sales_table(&db);

    let min: f64 = db
        .query_one("SELECT MIN(amount) FROM sales_aggr", ())
        .unwrap();
    assert!((min - 35.0).abs() < 0.01, "Expected min of 35, got {}", min);
}

#[test]
fn test_max_function() {
    let db = Database::open("memory://aggregation_max").expect("Failed to create database");
    setup_sales_table(&db);

    let max: f64 = db
        .query_one("SELECT MAX(amount) FROM sales_aggr", ())
        .unwrap();
    assert!(
        (max - 1500.0).abs() < 0.01,
        "Expected max of 1500, got {}",
        max
    );
}

#[test]
fn test_multiple_aggregations() {
    let db = Database::open("memory://aggregation_multiple").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales_aggr",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let count: i64 = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();
        let avg: f64 = row.get(2).unwrap();
        let min: f64 = row.get(3).unwrap();
        let max: f64 = row.get(4).unwrap();

        assert_eq!(count, 9);
        assert!((sum - 5235.0).abs() < 0.01);
        assert!((avg - 581.67).abs() < 0.01);
        assert!((min - 35.0).abs() < 0.01);
        assert!((max - 1500.0).abs() < 0.01);
    }
}

#[test]
fn test_group_by_with_sum() {
    let db = Database::open("memory://aggregation_group_sum").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount) FROM sales_aggr GROUP BY category ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();
        results.push((category, sum));
    }

    // Clothing: 35 + 60 + 90 = 185
    // Electronics: 1200 + 800 + 1500 = 3500
    // Furniture: 150 + 450 + 950 = 1550
    assert_eq!(results.len(), 3);

    for (category, sum) in &results {
        match category.as_str() {
            "Clothing" => assert!((sum - 185.0).abs() < 0.01, "Clothing sum should be 185"),
            "Electronics" => assert!(
                (sum - 3500.0).abs() < 0.01,
                "Electronics sum should be 3500"
            ),
            "Furniture" => assert!((sum - 1550.0).abs() < 0.01, "Furniture sum should be 1550"),
            _ => panic!("Unexpected category: {}", category),
        }
    }
}

#[test]
fn test_group_by_with_avg() {
    let db = Database::open("memory://aggregation_group_avg").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, AVG(amount) FROM sales_aggr GROUP BY category ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let avg: f64 = row.get(1).unwrap();
        results.push((category, avg));
    }

    assert_eq!(results.len(), 3);

    for (category, avg) in &results {
        match category.as_str() {
            "Clothing" => assert!((avg - 61.67).abs() < 0.01, "Clothing avg should be ~61.67"),
            "Electronics" => assert!(
                (avg - 1166.67).abs() < 0.01,
                "Electronics avg should be ~1166.67"
            ),
            "Furniture" => assert!(
                (avg - 516.67).abs() < 0.01,
                "Furniture avg should be ~516.67"
            ),
            _ => panic!("Unexpected category: {}", category),
        }
    }
}

#[test]
fn test_count_with_where() {
    let db = Database::open("memory://aggregation_count_where").expect("Failed to create database");
    setup_sales_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM sales_aggr WHERE amount > 500", ())
        .unwrap();
    // Laptop(1200), Smartphone(800), TV(1500), Sofa(950) = 4 items > 500
    assert_eq!(count, 4, "Expected 4 items with amount > 500");
}

#[test]
fn test_sum_with_where() {
    let db = Database::open("memory://aggregation_sum_where").expect("Failed to create database");
    setup_sales_table(&db);

    let sum: f64 = db
        .query_one(
            "SELECT SUM(amount) FROM sales_aggr WHERE category = 'Electronics'",
            (),
        )
        .unwrap();
    // 1200 + 800 + 1500 = 3500
    assert!(
        (sum - 3500.0).abs() < 0.01,
        "Expected sum of 3500 for Electronics"
    );
}
