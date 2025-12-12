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

//! ROLLUP, CUBE, and GROUPING SETS Tests
//!
//! Tests for GROUP BY ROLLUP, GROUP BY CUBE, and GROUPING SETS functionality

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

// ============================================================================
// GROUPING SETS Tests
// ============================================================================

#[test]
fn test_grouping_sets_basic() {
    let db = Database::open("memory://grouping_sets_basic").expect("Failed to create database");
    setup_sales_table(&db);

    // GROUPING SETS ((region, product), (region), ()) is equivalent to ROLLUP(region, product)
    let result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region, product), (region), ())",
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
            has_grand_total = true;
            assert!(
                (sum - 700.0).abs() < 0.01,
                "Grand total should be 700, got {}",
                sum
            );
        } else if region.is_some() && product.is_none() {
            has_region_subtotal = true;
        } else {
            has_detail = true;
        }
    }

    assert_eq!(
        row_count, 7,
        "GROUPING SETS should produce 7 rows (4 detail + 2 region subtotal + 1 grand total)"
    );
    assert!(has_grand_total, "Should have a grand total row");
    assert!(has_region_subtotal, "Should have region subtotal rows");
    assert!(has_detail, "Should have detail rows");
}

#[test]
fn test_grouping_sets_single_column() {
    let db = Database::open("memory://grouping_sets_single").expect("Failed to create database");
    setup_sales_table(&db);

    // GROUPING SETS ((region), ()) produces region groups + grand total
    let result = db
        .query(
            "SELECT region, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region), ())",
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
        "GROUPING SETS ((region), ()) should produce 3 rows"
    );
    assert!(has_grand_total, "Should have a grand total row");
}

#[test]
fn test_grouping_sets_no_grand_total() {
    let db = Database::open("memory://grouping_sets_no_grand").expect("Failed to create database");
    setup_sales_table(&db);

    // GROUPING SETS without empty set - no grand total
    let result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region), (product))",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut has_grand_total = false;

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();

        // Should NOT have a row where both are NULL
        if region.is_none() && product.is_none() {
            has_grand_total = true;
        }
    }

    assert_eq!(
        row_count, 4,
        "Should produce 4 rows (2 regions + 2 products)"
    );
    assert!(
        !has_grand_total,
        "Should NOT have grand total (no empty set)"
    );
}

#[test]
fn test_grouping_sets_with_grouping_function() {
    let db = Database::open("memory://grouping_sets_function").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, product, SUM(amount), GROUPING(region), GROUPING(product) FROM sales GROUP BY GROUPING SETS ((region, product), (region), ())",
            (),
        )
        .expect("Failed to query");

    let mut grand_total_groupings = None;
    let mut region_subtotal_groupings = None;
    let mut detail_groupings = None;

    for row in result {
        let row = row.expect("Failed to get row");
        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();
        let g_region: i64 = row.get(3).unwrap();
        let g_product: i64 = row.get(4).unwrap();

        if region.is_none() && product.is_none() {
            grand_total_groupings = Some((g_region, g_product));
        } else if region.is_some() && product.is_none() {
            region_subtotal_groupings = Some((g_region, g_product));
        } else if region.is_some() && product.is_some() {
            detail_groupings = Some((g_region, g_product));
        }
    }

    // Grand total: both columns are rolled up
    assert_eq!(
        grand_total_groupings,
        Some((1, 1)),
        "Grand total should have GROUPING(region)=1, GROUPING(product)=1"
    );

    // Region subtotal: product is rolled up
    assert_eq!(
        region_subtotal_groupings,
        Some((0, 1)),
        "Region subtotal should have GROUPING(region)=0, GROUPING(product)=1"
    );

    // Detail row: neither rolled up
    assert_eq!(
        detail_groupings,
        Some((0, 0)),
        "Detail row should have GROUPING(region)=0, GROUPING(product)=0"
    );
}

#[test]
fn test_grouping_sets_equivalent_to_rollup() {
    let db = Database::open("memory://grouping_sets_equiv").expect("Failed to create database");
    setup_sales_table(&db);

    // Compare GROUPING SETS result with equivalent ROLLUP
    let rollup_result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY ROLLUP(region, product) ORDER BY region, product",
            (),
        )
        .expect("Failed to query ROLLUP");

    let grouping_sets_result = db
        .query(
            "SELECT region, product, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region, product), (region), ()) ORDER BY region, product",
            (),
        )
        .expect("Failed to query GROUPING SETS");

    let rollup_rows: Vec<_> = rollup_result.collect();
    let gs_rows: Vec<_> = grouping_sets_result.collect();

    assert_eq!(
        rollup_rows.len(),
        gs_rows.len(),
        "ROLLUP and equivalent GROUPING SETS should have same row count"
    );

    // Both should produce the same aggregated results
    for (r_row, gs_row) in rollup_rows.iter().zip(gs_rows.iter()) {
        let r_row = r_row.as_ref().expect("ROLLUP row error");
        let gs_row = gs_row.as_ref().expect("GROUPING SETS row error");

        let r_sum: f64 = r_row.get(2).unwrap();
        let gs_sum: f64 = gs_row.get(2).unwrap();

        assert!(
            (r_sum - gs_sum).abs() < 0.01,
            "Sums should match: ROLLUP={}, GROUPING SETS={}",
            r_sum,
            gs_sum
        );
    }
}

#[test]
fn test_grouping_sets_with_multiple_aggregates() {
    let db = Database::open("memory://grouping_sets_multi_agg").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, SUM(amount), AVG(amount), COUNT(*) FROM sales GROUP BY GROUPING SETS ((region), ())",
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
fn test_grouping_sets_only_grand_total() {
    let db =
        Database::open("memory://grouping_sets_grand_only").expect("Failed to create database");
    setup_sales_table(&db);

    // Empty set only - just grand total
    let result = db
        .query(
            "SELECT SUM(amount), COUNT(*) FROM sales GROUP BY GROUPING SETS (())",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let sum: f64 = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();

        assert!((sum - 700.0).abs() < 0.01, "Sum should be 700");
        assert_eq!(count, 4, "Count should be 4");
    }

    assert_eq!(row_count, 1, "Should have exactly 1 row (grand total)");
}

#[test]
fn test_grouping_sets_with_having() {
    let db = Database::open("memory://grouping_sets_having").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region), ()) HAVING SUM(amount) > 300",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let sum: f64 = row.get(1).unwrap();
        assert!(sum > 300.0, "All rows should have SUM > 300 due to HAVING");
    }

    // East=250 (filtered out), West=450, Grand=700
    assert_eq!(
        row_count, 2,
        "Should have 2 rows after HAVING filter (West + Grand)"
    );
}

#[test]
fn test_grouping_sets_with_where() {
    let db = Database::open("memory://grouping_sets_where").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, SUM(amount) FROM sales WHERE amount >= 150 GROUP BY GROUPING SETS ((region), ())",
            (),
        )
        .expect("Failed to query");

    let mut grand_total = 0.0;
    for row in result {
        let row = row.expect("Failed to get row");
        let region: Option<String> = row.get(0).unwrap();
        let sum: f64 = row.get(1).unwrap();

        if region.is_none() {
            grand_total = sum;
        }
    }

    // Only rows with amount >= 150: East-B(150), West-A(200), West-B(250) = 600
    assert!(
        (grand_total - 600.0).abs() < 0.01,
        "Grand total with WHERE should be 600, got {}",
        grand_total
    );
}

#[test]
fn test_grouping_sets_three_columns() {
    let db = Database::open("memory://grouping_sets_three").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (region TEXT, product TEXT, channel TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO orders VALUES ('East', 'A', 'Online', 100.0)",
        "INSERT INTO orders VALUES ('East', 'B', 'Store', 150.0)",
        "INSERT INTO orders VALUES ('West', 'A', 'Online', 200.0)",
        "INSERT INTO orders VALUES ('West', 'B', 'Store', 250.0)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }

    let result = db
        .query(
            "SELECT region, product, channel, SUM(amount) FROM orders GROUP BY GROUPING SETS ((region, product, channel), (region, product), (region), ())",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut has_grand_total = false;
    let mut has_region_only = false;
    let mut has_region_product = false;
    let mut has_all_three = false;

    for row in result {
        let row = row.expect("Failed to get row");
        row_count += 1;

        let region: Option<String> = row.get(0).unwrap();
        let product: Option<String> = row.get(1).unwrap();
        let channel: Option<String> = row.get(2).unwrap();

        match (&region, &product, &channel) {
            (None, None, None) => has_grand_total = true,
            (Some(_), None, None) => has_region_only = true,
            (Some(_), Some(_), None) => has_region_product = true,
            (Some(_), Some(_), Some(_)) => has_all_three = true,
            _ => {}
        }
    }

    // 4 detail + 4 region-product + 2 region + 1 grand = 11
    assert_eq!(
        row_count, 11,
        "Should produce 11 rows for this GROUPING SETS"
    );
    assert!(has_grand_total, "Should have grand total");
    assert!(has_region_only, "Should have region-only rows");
    assert!(has_region_product, "Should have region-product rows");
    assert!(has_all_three, "Should have all-three-columns rows");
}
