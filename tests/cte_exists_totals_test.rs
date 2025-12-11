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

//! CTE Totals Tests
//!
//! Tests CTE with GROUP BY and aggregation for totals

use stoolap::Database;

fn setup_sales_db() -> Database {
    let db = Database::open("memory://cte_totals").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            region TEXT,
            amount FLOAT,
            year INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    db
}

/// Test product totals with GROUP BY
#[test]
fn test_product_totals() {
    let db = setup_sales_db();

    let result = db
        .query(
            "SELECT product, SUM(amount) as total
             FROM sales
             GROUP BY product
             ORDER BY product",
            (),
        )
        .expect("Failed to execute query");

    let mut totals: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let product: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        totals.push((product, total));
    }

    assert_eq!(totals.len(), 2, "Expected 2 products");

    // Gadget total: 2000 + 2500 + 2200 + 2800 = 9500
    // Widget total: 1000 + 1500 + 1200 + 1800 = 5500
    assert_eq!(totals[0].0, "Gadget");
    assert!(
        (totals[0].1 - 9500.0).abs() < 0.01,
        "Expected Gadget total 9500"
    );

    assert_eq!(totals[1].0, "Widget");
    assert!(
        (totals[1].1 - 5500.0).abs() < 0.01,
        "Expected Widget total 5500"
    );
}

/// Test region totals
#[test]
fn test_region_totals() {
    let db = Database::open("memory://cte_region_totals").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, region TEXT, amount FLOAT, year INTEGER)", ())
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT region, SUM(amount) as total
             FROM sales
             GROUP BY region
             ORDER BY region",
            (),
        )
        .expect("Failed to execute query");

    let mut totals: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let region: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        totals.push((region, total));
    }

    assert_eq!(totals.len(), 2, "Expected 2 regions");

    // North total: 1000 + 2000 + 1200 + 2200 = 6400
    // South total: 1500 + 2500 + 1800 + 2800 = 8600
    assert_eq!(totals[0].0, "North");
    assert!(
        (totals[0].1 - 6400.0).abs() < 0.01,
        "Expected North total 6400"
    );

    assert_eq!(totals[1].0, "South");
    assert!(
        (totals[1].1 - 8600.0).abs() < 0.01,
        "Expected South total 8600"
    );
}

/// Test year totals
#[test]
fn test_year_totals() {
    let db = Database::open("memory://cte_year_totals").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, region TEXT, amount FLOAT, year INTEGER)", ())
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT year, SUM(amount) as total
             FROM sales
             GROUP BY year
             ORDER BY year",
            (),
        )
        .expect("Failed to execute query");

    let mut totals: Vec<(i64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let year: i64 = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        totals.push((year, total));
    }

    assert_eq!(totals.len(), 2, "Expected 2 years");

    // 2023 total: 1000 + 1500 + 2000 + 2500 = 7000
    // 2024 total: 1200 + 1800 + 2200 + 2800 = 8000
    assert_eq!(totals[0].0, 2023);
    assert!(
        (totals[0].1 - 7000.0).abs() < 0.01,
        "Expected 2023 total 7000"
    );

    assert_eq!(totals[1].0, 2024);
    assert!(
        (totals[1].1 - 8000.0).abs() < 0.01,
        "Expected 2024 total 8000"
    );
}

/// Test multi-column GROUP BY
#[test]
fn test_multi_column_group_by() {
    let db = Database::open("memory://cte_multi_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, region TEXT, amount FLOAT, year INTEGER)", ())
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT product, year, SUM(amount) as total
             FROM sales
             GROUP BY product, year
             ORDER BY product, year",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _product: String = row.get(0).unwrap();
        let _year: i64 = row.get(1).unwrap();
        let _total: f64 = row.get(2).unwrap();
        count += 1;
    }

    // 2 products * 2 years = 4 groups
    assert_eq!(count, 4, "Expected 4 groups (2 products x 2 years)");
}

/// Test aggregate functions (COUNT, AVG, MIN, MAX)
#[test]
fn test_aggregate_functions() {
    let db = Database::open("memory://cte_agg_funcs").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, region TEXT, amount FLOAT, year INTEGER)", ())
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT product,
                    COUNT(*) as cnt,
                    AVG(amount) as avg_amount,
                    MIN(amount) as min_amount,
                    MAX(amount) as max_amount
             FROM sales
             GROUP BY product
             ORDER BY product",
            (),
        )
        .expect("Failed to execute query");

    let mut rows_data: Vec<(String, i64, f64, f64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let product: String = row.get(0).unwrap();
        let cnt: i64 = row.get(1).unwrap();
        let avg_amount: f64 = row.get(2).unwrap();
        let min_amount: f64 = row.get(3).unwrap();
        let max_amount: f64 = row.get(4).unwrap();
        rows_data.push((product, cnt, avg_amount, min_amount, max_amount));
    }

    assert_eq!(rows_data.len(), 2);

    // Gadget: count=4, avg=2375, min=2000, max=2800
    assert_eq!(rows_data[0].0, "Gadget");
    assert_eq!(rows_data[0].1, 4);
    assert!((rows_data[0].2 - 2375.0).abs() < 0.01);
    assert!((rows_data[0].3 - 2000.0).abs() < 0.01);
    assert!((rows_data[0].4 - 2800.0).abs() < 0.01);

    // Widget: count=4, avg=1375, min=1000, max=1800
    assert_eq!(rows_data[1].0, "Widget");
    assert_eq!(rows_data[1].1, 4);
    assert!((rows_data[1].2 - 1375.0).abs() < 0.01);
    assert!((rows_data[1].3 - 1000.0).abs() < 0.01);
    assert!((rows_data[1].4 - 1800.0).abs() < 0.01);
}

/// Test grand total (no GROUP BY)
#[test]
fn test_grand_total() {
    let db = Database::open("memory://cte_grand_total").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, region TEXT, amount FLOAT, year INTEGER)", ())
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert data");

    let total: f64 = db
        .query_one("SELECT SUM(amount) FROM sales", ())
        .expect("Failed to execute query");

    // Grand total: 1000+1500+2000+2500+1200+1800+2200+2800 = 15000
    assert!((total - 15000.0).abs() < 0.01, "Expected grand total 15000");
}
