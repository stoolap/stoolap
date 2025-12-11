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

//! DISTINCT Tests
//!
//! Tests SELECT DISTINCT functionality: single column, multi-column, with ORDER BY, numeric columns

use std::collections::HashSet;
use stoolap::Database;

fn setup_products_table(db: &Database) {
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, region TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create products table");

    // Insert sample data with duplicates in various columns
    let inserts = [
        "INSERT INTO products (id, category, region, price) VALUES (1, 'Electronics', 'North', 100.0)",
        "INSERT INTO products (id, category, region, price) VALUES (2, 'Electronics', 'South', 200.0)",
        "INSERT INTO products (id, category, region, price) VALUES (3, 'Electronics', 'North', 150.0)",
        "INSERT INTO products (id, category, region, price) VALUES (4, 'Clothing', 'East', 50.0)",
        "INSERT INTO products (id, category, region, price) VALUES (5, 'Clothing', 'West', 75.0)",
        "INSERT INTO products (id, category, region, price) VALUES (6, 'Clothing', 'East', 60.0)",
        "INSERT INTO products (id, category, region, price) VALUES (7, 'Books', 'North', 20.0)",
        "INSERT INTO products (id, category, region, price) VALUES (8, 'Books', 'South', 25.0)",
        "INSERT INTO products (id, category, region, price) VALUES (9, 'Books', 'North', 20.0)",
        "INSERT INTO products (id, category, region, price) VALUES (10, 'Electronics', 'West', 180.0)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

/// Test case 1: Basic SELECT DISTINCT for a single column
#[test]
fn test_single_column_distinct() {
    let db = Database::open("memory://distinct_single").expect("Failed to create database");
    setup_products_table(&db);

    // Query distinct categories
    let result = db
        .query("SELECT DISTINCT category FROM products", ())
        .expect("Failed to execute SELECT DISTINCT");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        categories.push(category);
    }

    // Verify results (should be 3 distinct categories)
    let expected: HashSet<String> = ["Electronics", "Clothing", "Books"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let actual: HashSet<String> = categories.iter().cloned().collect();

    assert_eq!(
        actual.len(),
        3,
        "Expected 3 distinct categories, got {}",
        actual.len()
    );
    assert_eq!(actual, expected, "Category sets don't match");

    // Verify with COUNT DISTINCT for comparison
    let distinct_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM products", ())
        .expect("Failed to execute COUNT DISTINCT");
    assert_eq!(
        distinct_count, 3,
        "COUNT DISTINCT should return 3, got {}",
        distinct_count
    );
}

/// Test case 2: SELECT DISTINCT with multiple columns
#[test]
fn test_multi_column_distinct() {
    let db = Database::open("memory://distinct_multi").expect("Failed to create database");
    setup_products_table(&db);

    // Query distinct category-region combinations
    let result = db
        .query("SELECT DISTINCT category, region FROM products", ())
        .expect("Failed to execute multi-column SELECT DISTINCT");

    let mut combinations: Vec<(String, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let region: String = row.get(1).unwrap();
        combinations.push((category, region));
    }

    // Expected distinct combinations (should be 7 unique combinations)
    let expected: HashSet<(String, String)> = [
        ("Electronics", "North"),
        ("Electronics", "South"),
        ("Electronics", "West"),
        ("Clothing", "East"),
        ("Clothing", "West"),
        ("Books", "North"),
        ("Books", "South"),
    ]
    .iter()
    .map(|(c, r)| (c.to_string(), r.to_string()))
    .collect();

    let actual: HashSet<(String, String)> = combinations.iter().cloned().collect();

    assert_eq!(
        actual.len(),
        7,
        "Expected 7 distinct category-region combinations, got {}",
        actual.len()
    );
    assert_eq!(actual, expected, "Combination sets don't match");
}

/// Test case 3: SELECT DISTINCT with ORDER BY
#[test]
fn test_distinct_with_order_by() {
    let db = Database::open("memory://distinct_order").expect("Failed to create database");
    setup_products_table(&db);

    // Query distinct regions ordered by name
    let result = db
        .query("SELECT DISTINCT region FROM products ORDER BY region", ())
        .expect("Failed to execute SELECT DISTINCT with ORDER BY");

    let mut regions: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let region: String = row.get(0).unwrap();
        regions.push(region);
    }

    // Expected result should be ordered alphabetically
    let expected = vec!["East", "North", "South", "West"];

    assert_eq!(
        regions.len(),
        expected.len(),
        "Expected {} distinct regions, got {}",
        expected.len(),
        regions.len()
    );

    // Check the order
    for (i, region) in regions.iter().enumerate() {
        assert_eq!(
            region, expected[i],
            "Expected region {} at position {}, got {}",
            expected[i], i, region
        );
    }
}

/// Test case 4: SELECT DISTINCT on numeric columns
#[test]
fn test_distinct_on_numeric() {
    let db = Database::open("memory://distinct_numeric").expect("Failed to create database");
    setup_products_table(&db);

    // Query distinct prices
    let result = db
        .query("SELECT DISTINCT price FROM products ORDER BY price", ())
        .expect("Failed to execute SELECT DISTINCT on numeric column");

    let mut prices: Vec<f64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let price: f64 = row.get(0).unwrap();
        prices.push(price);
    }

    // Expected distinct prices (there's a duplicate 20.00)
    let expected_prices = vec![20.0, 25.0, 50.0, 60.0, 75.0, 100.0, 150.0, 180.0, 200.0];

    assert_eq!(
        prices.len(),
        expected_prices.len(),
        "Expected {} distinct prices, got {}",
        expected_prices.len(),
        prices.len()
    );

    // Verify with COUNT DISTINCT for comparison
    let distinct_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT price) FROM products", ())
        .expect("Failed to execute COUNT DISTINCT");
    assert_eq!(
        distinct_count,
        expected_prices.len() as i64,
        "COUNT DISTINCT mismatch"
    );
}

/// Test case 5: Complex scenario with candle data (real-world pattern)
#[test]
fn test_complex_distinct() {
    let db = Database::open("memory://distinct_complex").expect("Failed to create database");

    // Create a table for candle data (simplified without timestamp for this test)
    db.execute(
        "CREATE TABLE candle_data (id INTEGER PRIMARY KEY, symbol TEXT, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT)", ())
    .expect("Failed to create candle_data table");

    // Insert sample data with repeated symbols
    let symbols = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BTC-USD", "ETH-USD", "BTC-USD",
    ];
    for (i, symbol) in symbols.iter().enumerate() {
        let sql = format!(
            "INSERT INTO candle_data (id, symbol, open, high, low, close, volume) VALUES ({}, '{}', 100.0, 105.0, 95.0, 102.0, 1000.0)",
            i + 1,
            symbol
        );
        db.execute(&sql, ()).expect("Failed to insert candle data");
    }

    // Test SELECT DISTINCT on symbol column
    let result = db
        .query(
            "SELECT DISTINCT symbol FROM candle_data ORDER BY symbol",
            (),
        )
        .expect("Failed to execute SELECT DISTINCT on symbols");

    let mut distinct_symbols: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let symbol: String = row.get(0).unwrap();
        distinct_symbols.push(symbol);
    }

    // Expected distinct symbols (ordered)
    let expected = vec!["BTC-USD", "ETH-USD", "SOL-USD"];

    assert_eq!(
        distinct_symbols.len(),
        expected.len(),
        "Expected {} distinct symbols, got {}",
        expected.len(),
        distinct_symbols.len()
    );

    for (i, symbol) in distinct_symbols.iter().enumerate() {
        assert_eq!(
            symbol, expected[i],
            "Expected symbol {} at position {}, got {}",
            expected[i], i, symbol
        );
    }

    // Double-check with COUNT DISTINCT
    let symbol_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT symbol) FROM candle_data", ())
        .expect("Failed to execute COUNT DISTINCT on symbols");
    assert_eq!(symbol_count, 3, "COUNT DISTINCT should return 3");
}

/// Test case 6: DISTINCT with no duplicates (all unique values)
#[test]
fn test_distinct_with_no_duplicates() {
    let db = Database::open("memory://distinct_unique").expect("Failed to create database");

    // Create a table with all unique values
    db.execute("CREATE TABLE unique_data (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create unique_data table");

    // Insert 10 unique values
    for i in 1..=10 {
        let sql = format!("INSERT INTO unique_data (id) VALUES ({})", i);
        db.execute(&sql, ()).expect("Failed to insert data");
    }

    // Test SELECT DISTINCT on a column with all unique values
    let result = db
        .query("SELECT DISTINCT id FROM unique_data ORDER BY id", ())
        .expect("Failed to execute SELECT DISTINCT");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Expect 10 ids
    assert_eq!(ids.len(), 10, "Expected 10 distinct ids, got {}", ids.len());

    // Verify they are in order
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(
            *id,
            (i + 1) as i64,
            "Expected id {} at position {}, got {}",
            i + 1,
            i,
            id
        );
    }
}

/// Test case 7: DISTINCT with a large number of rows
#[test]
fn test_distinct_with_large_dataset() {
    let db = Database::open("memory://distinct_large").expect("Failed to create database");

    // Create a table for this test
    db.execute(
        "CREATE TABLE large_data (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create large_data table");

    // Insert 1000 rows with only 5 distinct categories
    let categories = ["A", "B", "C", "D", "E"];
    for i in 1..=1000 {
        let category = categories[(i - 1) % categories.len()];
        let sql = format!(
            "INSERT INTO large_data (id, category) VALUES ({}, '{}')",
            i, category
        );
        db.execute(&sql, ()).expect("Failed to insert data");
    }

    // Query distinct categories
    let result = db
        .query(
            "SELECT DISTINCT category FROM large_data ORDER BY category",
            (),
        )
        .expect("Failed to execute SELECT DISTINCT on large dataset");

    let mut distinct_categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        distinct_categories.push(category);
    }

    // Should only have 5 distinct categories
    assert_eq!(
        distinct_categories.len(),
        5,
        "Expected 5 distinct categories, got {}",
        distinct_categories.len()
    );

    // Verify they match our expected categories (ordered)
    let expected = vec!["A", "B", "C", "D", "E"];
    for (i, category) in distinct_categories.iter().enumerate() {
        assert_eq!(
            category, expected[i],
            "Expected category {} at position {}, got {}",
            expected[i], i, category
        );
    }

    // Verify with COUNT DISTINCT
    let category_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM large_data", ())
        .expect("Failed to execute COUNT DISTINCT");
    assert_eq!(category_count, 5, "COUNT DISTINCT should return 5");
}

/// Additional test: DISTINCT with WHERE clause
#[test]
fn test_distinct_with_where() {
    let db = Database::open("memory://distinct_where").expect("Failed to create database");
    setup_products_table(&db);

    // Query distinct categories where price > 50
    let result = db
        .query(
            "SELECT DISTINCT category FROM products WHERE price > 50.0 ORDER BY category",
            (),
        )
        .expect("Failed to execute SELECT DISTINCT with WHERE");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        categories.push(category);
    }

    // Books only has prices 20 and 25, so it should be excluded
    // Clothing has 50, 60, 75 - so 60 and 75 are > 50
    // Electronics has 100, 150, 180, 200 - all > 50
    let expected: HashSet<String> = ["Clothing", "Electronics"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let actual: HashSet<String> = categories.iter().cloned().collect();

    assert_eq!(
        actual.len(),
        2,
        "Expected 2 distinct categories with price > 50, got {}",
        actual.len()
    );
    assert_eq!(actual, expected, "Category sets don't match");
}

/// Additional test: DISTINCT with NULL values
#[test]
fn test_distinct_with_nulls() {
    let db = Database::open("memory://distinct_nulls").expect("Failed to create database");

    db.execute(
        "CREATE TABLE nullable_data (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert data including NULLs
    db.execute("INSERT INTO nullable_data (id, value) VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO nullable_data (id, value) VALUES (2, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO nullable_data (id, value) VALUES (3, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO nullable_data (id, value) VALUES (4, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO nullable_data (id, value) VALUES (5, NULL)", ())
        .unwrap();

    // Query distinct values (should include NULL as one distinct value)
    let result = db
        .query("SELECT DISTINCT value FROM nullable_data", ())
        .expect("Failed to execute SELECT DISTINCT with NULLs");

    let mut count = 0;
    let mut has_null = false;
    let mut values: HashSet<String> = HashSet::new();

    for row in result {
        let row = row.expect("Failed to get row");
        count += 1;
        // Use Option<String> to properly detect NULL values
        let value: Option<String> = row.get(0).unwrap();
        match value {
            Some(v) => {
                values.insert(v);
            }
            None => {
                has_null = true;
            }
        }
    }

    // Should have 3 distinct values: 'A', 'B', and NULL
    assert_eq!(count, 3, "Expected 3 distinct values (including NULL)");
    assert!(values.contains("A"), "Should contain 'A'");
    assert!(values.contains("B"), "Should contain 'B'");
    assert!(has_null, "Should contain NULL");
}
