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

//! ROLLUP and CUBE Tests
//!
//! Tests for GROUP BY ROLLUP and GROUP BY CUBE functionality (Bug #84)

use stoolap::Database;

fn setup_sales_table(db: &Database) {
    db.execute(
        "CREATE TABLE sales (region TEXT, product TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO sales VALUES ('East', 'A', 100.0)",
        "INSERT INTO sales VALUES ('East', 'B', 150.0)",
        "INSERT INTO sales VALUES ('West', 'A', 200.0)",
        "INSERT INTO sales VALUES ('West', 'B', 250.0)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

#[test]
fn test_rollup_basic() {
    let db = Database::open("memory://rollup_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY ROLLUP(region, product)",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut has_grand_total = false;
    let mut has_region_subtotal = false;
    let mut has_detail = false;

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();
        let sum: f64 = row.get(2).unwrap();

        if region.is_none() && product.is_none() {
            // Grand total row
            has_grand_total = true;
            assert!(
                (sum - 700.0).abs() < 0.01,
                "Grand total should be 700, got {}",
                sum
            );
        } else if region.is_some() && product.is_none() {
            // Region subtotal
            has_region_subtotal = true;
            let r = region.unwrap();
            if r == "East" {
                assert!(
                    (sum - 250.0).abs() < 0.01,
                    "East subtotal should be 250, got {}",
                    sum
                );
            } else if r == "West" {
                assert!(
                    (sum - 450.0).abs() < 0.01,
                    "West subtotal should be 450, got {}",
                    sum
                );
            }
        } else {
            // Detail row
            has_detail = true;
        }
    }

    assert_eq!(
        row_count, 7,
        "ROLLUP should produce 7 rows (4 detail + 2 subtotal + 1 grand total)"
    );
    assert!(has_grand_total, "Should have a grand total row");
    assert!(has_region_subtotal, "Should have region subtotal rows");
    assert!(has_detail, "Should have detail rows");
}

#[test]
fn test_rollup_single_column() {
    let db = Database::open("memory://rollup_single").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, SUM(amount) FROM sales GROUP BY ROLLUP(region)",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut has_grand_total = false;

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let region: Option<String> = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();

        if region.is_none() {
            has_grand_total = true;
            assert!(
                (sum - 700.0).abs() < 0.01,
                "Grand total should be 700, got {}",
                sum
            );
        }
    }

    assert_eq!(
        row_count, 3,
        "ROLLUP(region) should produce 3 rows (2 regions + 1 grand total)"
    );
    assert!(has_grand_total, "Should have a grand total row");
}

#[test]
fn test_cube_basic() {
    let db = Database::open("memory://cube_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY CUBE(region, product)",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut has_grand_total = false;
    let mut has_region_subtotal = false;
    let mut has_product_subtotal = false;

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();
        let sum: f64 = row.get(2).unwrap();

        if region.is_none() && product.is_none() {
            has_grand_total = true;
            assert!(
                (sum - 700.0).abs() < 0.01,
                "Grand total should be 700, got {}",
                sum
            );
        } else if region.is_some() && product.is_none() {
            has_region_subtotal = true;
        } else if let (None, Some(p)) = (&region, &product) {
            has_product_subtotal = true;
            if p == "A" {
                assert!(
                    (sum - 300.0).abs() < 0.01,
                    "Product A total should be 300, got {}",
                    sum
                );
            } else if p == "B" {
                assert!(
                    (sum - 400.0).abs() < 0.01,
                    "Product B total should be 400, got {}",
                    sum
                );
            }
        }
    }

    assert_eq!(
        row_count, 9,
        "CUBE should produce 9 rows (4 detail + 2 region + 2 product + 1 grand)"
    );
    assert!(has_grand_total, "Should have a grand total row");
    assert!(has_region_subtotal, "Should have region subtotal rows");
    assert!(has_product_subtotal, "Should have product subtotal rows");
}

#[test]
fn test_rollup_with_count() {
    let db = Database::open("memory://rollup_count").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, product, COUNT(*) FROM sales GROUP BY ROLLUP(region, product)",
            (),
        )
        .expect("Failed to query");

    let mut grand_total_count = 0i64;

    for row in result {
        let row = row.expect("Failed to get row");
        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();
        let count: i64 = row.get(2).unwrap();

        if region.is_none() && product.is_none() {
            grand_total_count = count;
        }
    }

    assert_eq!(grand_total_count, 4, "Grand total COUNT(*) should be 4");
}

#[test]
fn test_rollup_with_multiple_aggregates() {
    let db = Database::open("memory://rollup_multi_agg").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, SUM(amount), AVG(amount), COUNT(*) FROM sales GROUP BY ROLLUP(region)",
            (),
        )
        .expect("Failed to query");

    let mut has_grand_total = false;

    for row in result {
        let row = row.expect("Failed to get row");
        let region: Option<String> = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();
        let avg: f64 = row.get(2).unwrap();
        let count: i64 = row.get(3).unwrap();

        if region.is_none() {
            has_grand_total = true;
            assert!((sum - 700.0).abs() < 0.01, "Grand total SUM should be 700");
            assert!((avg - 175.0).abs() < 0.01, "Grand total AVG should be 175");
            assert_eq!(count, 4, "Grand total COUNT should be 4");
        }
    }

    assert!(has_grand_total, "Should have a grand total row");
}

#[test]
fn test_regular_group_by_unchanged() {
    // Ensure regular GROUP BY still works as before
    let db = Database::open("memory://regular_group_by").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query("SELECT region, SUM(amount) FROM sales GROUP BY region", ())
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;
        let region: String = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();

        // Regular GROUP BY should not have NULL region values
        assert!(!region.is_empty(), "Region should not be empty");
        assert!(sum > 0.0, "Sum should be positive");
    }

    assert_eq!(
        row_count, 2,
        "Regular GROUP BY should produce 2 rows (East and West)"
    );
}
