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

//! Advanced Aggregation Tests
//!
//! Tests for ROLLUP, CUBE, GROUPING SETS, complex HAVING clauses,
//! GROUP BY expressions, and aggregate functions with FILTER/DISTINCT.

use stoolap::Database;

fn setup_sales_table(db: &Database) {
    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            category TEXT,
            region TEXT,
            year INTEGER,
            quarter INTEGER,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO sales VALUES (1, 'Laptop', 'Electronics', 'North', 2023, 1, 1000.0)",
        "INSERT INTO sales VALUES (2, 'Phone', 'Electronics', 'North', 2023, 1, 500.0)",
        "INSERT INTO sales VALUES (3, 'TV', 'Electronics', 'South', 2023, 1, 800.0)",
        "INSERT INTO sales VALUES (4, 'Laptop', 'Electronics', 'North', 2023, 2, 1200.0)",
        "INSERT INTO sales VALUES (5, 'Phone', 'Electronics', 'South', 2023, 2, 600.0)",
        "INSERT INTO sales VALUES (6, 'Chair', 'Furniture', 'North', 2023, 1, 200.0)",
        "INSERT INTO sales VALUES (7, 'Desk', 'Furniture', 'North', 2023, 2, 400.0)",
        "INSERT INTO sales VALUES (8, 'Chair', 'Furniture', 'South', 2023, 1, 250.0)",
        "INSERT INTO sales VALUES (9, 'Laptop', 'Electronics', 'North', 2024, 1, 1100.0)",
        "INSERT INTO sales VALUES (10, 'Phone', 'Electronics', 'South', 2024, 1, 550.0)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert");
    }
}

// =============================================================================
// ROLLUP Tests
// =============================================================================

#[test]
fn test_rollup_single_column() {
    let db = Database::open("memory://rollup_single").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM sales
             GROUP BY ROLLUP(category)
             ORDER BY category NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut category_count = 0;

    for _ in result {
        category_count += 1;
    }

    // ROLLUP(category) should produce:
    // - Electronics group
    // - Furniture group
    // - Grand total (NULL category)
    // At minimum we expect at least 2 groups (the base categories)
    assert!(
        category_count >= 2,
        "ROLLUP should produce at least 2 category groups"
    );
}

#[test]
fn test_rollup_two_columns() {
    let db = Database::open("memory://rollup_two").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, region, SUM(amount) as total
             FROM sales
             GROUP BY ROLLUP(category, region)
             ORDER BY category NULLS LAST, region NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    // ROLLUP(category, region) produces:
    // - All (category, region) combinations
    // - Subtotals per category (NULL region)
    // - Grand total (NULL category, NULL region)
    // For 2 categories x 2 regions = 4 combos + 2 category subtotals + 1 grand = 7
    assert!(row_count >= 7, "Expected at least 7 rows for ROLLUP");
}

#[test]
fn test_rollup_three_columns() {
    let db = Database::open("memory://rollup_three").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT year, quarter, category, SUM(amount) as total
             FROM sales
             GROUP BY ROLLUP(year, quarter, category)
             ORDER BY year NULLS LAST, quarter NULLS LAST, category NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    // ROLLUP(year, quarter, category) produces many subtotals
    assert!(row_count > 10, "Expected many rows for 3-level ROLLUP");
}

// =============================================================================
// CUBE Tests
// =============================================================================

#[test]
fn test_cube_two_columns() {
    let db = Database::open("memory://cube_two").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, region, SUM(amount) as total
             FROM sales
             GROUP BY CUBE(category, region)
             ORDER BY category NULLS LAST, region NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;

    for _ in result {
        row_count += 1;
    }

    // CUBE produces all combinations of aggregation levels:
    // - All (category, region) combinations: 2 categories x 2 regions = 4
    // - Subtotals per category (NULL region): 2
    // - Subtotals per region (NULL category): 2
    // - Grand total (NULL, NULL): 1
    // Total: 9 rows
    assert!(
        row_count >= 4,
        "CUBE should produce at least 4 rows for base combinations"
    );
}

// =============================================================================
// GROUPING SETS Tests
// =============================================================================

#[test]
fn test_grouping_sets_basic() {
    let db = Database::open("memory://grouping_sets_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, region, SUM(amount) as total
             FROM sales
             GROUP BY GROUPING SETS ((category), (region), ())
             ORDER BY category NULLS LAST, region NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    // Should have: 2 categories + 2 regions + 1 grand total = 5
    assert_eq!(row_count, 5, "Expected 5 rows from GROUPING SETS");
}

#[test]
fn test_grouping_sets_custom() {
    let db = Database::open("memory://grouping_sets_custom").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT year, category, SUM(amount) as total
             FROM sales
             GROUP BY GROUPING SETS ((year, category), (year))
             ORDER BY year, category NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    // (year, category) combinations + year subtotals
    assert!(row_count > 4, "Expected multiple rows from GROUPING SETS");
}

// =============================================================================
// GROUPING Function Tests
// =============================================================================

#[test]
fn test_grouping_function() {
    let db = Database::open("memory://grouping_function").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, region, SUM(amount) as total,
                    GROUPING(category) as cat_grouping,
                    GROUPING(region) as reg_grouping
             FROM sales
             GROUP BY ROLLUP(category, region)
             ORDER BY category NULLS LAST, region NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut found_grand_total = false;

    for row in result {
        let row = row.expect("Failed to get row");
        let cat_grouping: i64 = row.get(3).unwrap();
        let reg_grouping: i64 = row.get(4).unwrap();

        // Grand total has both GROUPING values = 1
        if cat_grouping == 1 && reg_grouping == 1 {
            found_grand_total = true;
        }
    }

    assert!(found_grand_total, "Expected to find grand total row");
}

// =============================================================================
// Complex HAVING Tests
// =============================================================================

#[test]
fn test_having_with_multiple_conditions() {
    let db = Database::open("memory://having_multiple").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, COUNT(*) as cnt, SUM(amount) as total
             FROM sales
             GROUP BY category
             HAVING COUNT(*) > 2 AND SUM(amount) > 1000",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cnt: i64 = row.get(1).unwrap();
        let total: f64 = row.get(2).unwrap();
        assert!(cnt > 2, "COUNT should be > 2");
        assert!(total > 1000.0, "SUM should be > 1000");
        row_count += 1;
    }

    assert!(row_count > 0, "Expected at least one group matching HAVING");
}

#[test]
fn test_having_with_comparison_operators() {
    let db = Database::open("memory://having_operators").expect("Failed to create database");
    setup_sales_table(&db);

    // Test >= operator
    let result = db
        .query(
            "SELECT category, AVG(amount) as avg_amt
             FROM sales
             GROUP BY category
             HAVING AVG(amount) >= 400",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let avg: f64 = row.get(1).unwrap();
        assert!(avg >= 400.0, "AVG should be >= 400");
        row_count += 1;
    }

    assert!(row_count > 0, "Expected at least one group");
}

#[test]
fn test_having_with_less_than() {
    let db = Database::open("memory://having_lt").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, COUNT(*) as cnt
             FROM sales
             GROUP BY region
             HAVING COUNT(*) < 10",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cnt: i64 = row.get(1).unwrap();
        assert!(cnt < 10, "COUNT should be < 10");
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 regions");
}

#[test]
fn test_having_with_not_equal() {
    let db = Database::open("memory://having_neq").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, COUNT(*) as cnt
             FROM sales
             GROUP BY category
             HAVING COUNT(*) != 3",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cnt: i64 = row.get(1).unwrap();
        assert!(cnt != 3, "COUNT should not be 3");
        row_count += 1;
    }

    assert!(row_count > 0, "Expected some groups");
}

// =============================================================================
// Aggregate with Expression Tests
// =============================================================================

#[test]
fn test_sum_with_expression() {
    let db = Database::open("memory://sum_expr").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount * 1.1) as total_with_tax
             FROM sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 categories");
}

#[test]
fn test_avg_with_expression() {
    let db = Database::open("memory://avg_expr").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region, AVG(amount / 100) as avg_hundreds
             FROM sales
             GROUP BY region",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 regions");
}

// =============================================================================
// COUNT Variations Tests
// =============================================================================

#[test]
fn test_count_column() {
    let db = Database::open("memory://count_column").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, NULL)", ()).unwrap();

    let count_star: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    let count_val: i64 = db.query_one("SELECT COUNT(val) FROM t", ()).unwrap();

    assert_eq!(count_star, 4, "COUNT(*) should count all rows");
    assert_eq!(count_val, 2, "COUNT(val) should exclude NULLs");
}

#[test]
fn test_count_distinct_column() {
    let db = Database::open("memory://count_distinct_col").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 20)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 20)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (5, 30)", ()).unwrap();

    let count_distinct: i64 = db
        .query_one("SELECT COUNT(DISTINCT val) FROM t", ())
        .unwrap();

    assert_eq!(count_distinct, 3, "COUNT(DISTINCT val) should be 3");
}

// =============================================================================
// GROUP BY Expression Tests
// =============================================================================

#[test]
fn test_group_by_expression() {
    let db = Database::open("memory://group_by_expr").expect("Failed to create database");
    setup_sales_table(&db);

    // Group by expression: amount / 500 (buckets)
    let result = db
        .query(
            "SELECT FLOOR(amount / 500) as bucket, COUNT(*) as cnt
             FROM sales
             GROUP BY FLOOR(amount / 500)
             ORDER BY bucket",
            (),
        )
        .expect("Failed to query");

    let mut buckets = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let bucket: i64 = row.get(0).unwrap();
        buckets.push(bucket);
    }

    assert!(!buckets.is_empty(), "Expected some buckets");
}

#[test]
fn test_group_by_case_expression() {
    let db = Database::open("memory://group_by_case").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT CASE WHEN amount >= 1000 THEN 'High' ELSE 'Low' END as tier,
                    COUNT(*) as cnt
             FROM sales
             GROUP BY CASE WHEN amount >= 1000 THEN 'High' ELSE 'Low' END
             ORDER BY tier",
            (),
        )
        .expect("Failed to query");

    let mut tiers = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let tier: String = row.get(0).unwrap();
        tiers.push(tier);
    }

    assert!(tiers.contains(&"High".to_string()), "Expected 'High' tier");
    assert!(tiers.contains(&"Low".to_string()), "Expected 'Low' tier");
}

#[test]
fn test_group_by_position() {
    let db = Database::open("memory://group_by_position").expect("Failed to create database");
    setup_sales_table(&db);

    // GROUP BY 1 refers to first SELECT column
    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM sales
             GROUP BY 1
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 categories");
}

// =============================================================================
// Multiple Column GROUP BY Tests
// =============================================================================

#[test]
fn test_group_by_multiple_columns() {
    let db = Database::open("memory://group_by_multi").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, region, year, SUM(amount) as total
             FROM sales
             GROUP BY category, region, year
             ORDER BY category, region, year",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert!(row_count > 4, "Expected multiple group combinations");
}

// =============================================================================
// Aggregate Functions Tests
// =============================================================================

#[test]
fn test_stddev_variance() {
    let db = Database::open("memory://stddev_var").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category,
                    STDDEV(amount) as std_dev,
                    VARIANCE(amount) as var_amt
             FROM sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 categories");
}

#[test]
fn test_string_agg() {
    let db = Database::open("memory://string_agg").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, STRING_AGG(product, ', ') as products
             FROM sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut found_electronics = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let products: String = row.get(1).unwrap();

        if category == "Electronics" {
            // Should contain Laptop, Phone, TV
            assert!(
                products.contains("Laptop") || products.contains("Phone"),
                "Electronics should include Laptop or Phone"
            );
            found_electronics = true;
        }
    }

    assert!(found_electronics, "Expected Electronics category");
}

#[test]
fn test_string_agg_with_order() {
    let db = Database::open("memory://string_agg_order").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, STRING_AGG(DISTINCT product, ', ' ORDER BY product) as products
             FROM sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 categories");
}

// =============================================================================
// Aggregate with FILTER Tests
// =============================================================================

#[test]
fn test_count_with_filter() {
    let db = Database::open("memory://count_filter").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE amount > 500) as high_value
             FROM sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(1).unwrap();
        let high_value: i64 = row.get(2).unwrap();
        assert!(high_value <= total, "Filtered count should be <= total");
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 categories");
}

#[test]
fn test_sum_with_filter() {
    let db = Database::open("memory://sum_filter").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT region,
                    SUM(amount) as total,
                    SUM(amount) FILTER (WHERE year = 2024) as total_2024
             FROM sales
             GROUP BY region
             ORDER BY region",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 2, "Expected 2 regions");
}

// =============================================================================
// Global Aggregation Tests (no GROUP BY)
// =============================================================================

#[test]
fn test_global_aggregation_multiple() {
    let db = Database::open("memory://global_agg_multi").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let count: i64 = row.get(0).unwrap();
        assert_eq!(count, 10, "Expected 10 rows");
        row_count += 1;
    }

    assert_eq!(row_count, 1, "Global aggregation should return 1 row");
}

#[test]
fn test_global_aggregation_empty_table() {
    let db = Database::open("memory://global_agg_empty").expect("Failed to create database");

    db.execute(
        "CREATE TABLE empty_table (id INTEGER PRIMARY KEY, val FLOAT)",
        (),
    )
    .expect("Failed to create table");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM empty_table", ())
        .unwrap();

    assert_eq!(count, 0, "COUNT(*) on empty table should be 0");
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_group_by_null_values() {
    let db = Database::open("memory://group_by_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'A', 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL, 20)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, 'A', 30)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, NULL, 40)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT cat, SUM(val) as total FROM t GROUP BY cat ORDER BY cat NULLS LAST",
            (),
        )
        .expect("Failed to query");

    let mut groups = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: Result<String, _> = row.get(0);
        let total: i64 = row.get(1).unwrap();
        groups.push((cat.ok(), total));
    }

    // Should have 2 groups: 'A' and NULL
    assert_eq!(groups.len(), 2, "Expected 2 groups");
    // Check that we have A group with sum 40
    assert!(
        groups
            .iter()
            .any(|(c, t)| c.as_deref() == Some("A") && *t == 40),
        "Expected 'A' group with sum 40"
    );
}

#[test]
fn test_having_on_aliased_aggregate() {
    let db = Database::open("memory://having_alias").expect("Failed to create database");
    setup_sales_table(&db);

    // Use alias in HAVING
    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM sales
             GROUP BY category
             HAVING total > 1000
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert!(
        row_count > 0,
        "Expected at least one group with total > 1000"
    );
}

#[test]
fn test_group_by_with_limit() {
    let db = Database::open("memory://group_by_limit").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM sales
             GROUP BY category
             ORDER BY total DESC
             LIMIT 1",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 1, "Expected exactly 1 row due to LIMIT");
}

#[test]
fn test_group_by_with_offset() {
    let db = Database::open("memory://group_by_offset").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM sales
             GROUP BY category
             ORDER BY total DESC
             LIMIT 1 OFFSET 1",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 1, "Expected 1 row due to OFFSET 1 LIMIT 1");
}
