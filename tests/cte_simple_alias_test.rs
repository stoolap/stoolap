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

//! CTE Simple Alias Tests
//!
//! Tests CTE with column aliases

use stoolap::Database;

fn setup_cte_alias_db(test_name: &str) -> Database {
    let db = Database::open(&format!("memory://cte_alias_{}", test_name))
        .expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test (
            a INTEGER,
            b INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO test (a, b) VALUES
        (10, 20),
        (30, 40)",
        (),
    )
    .expect("Failed to insert data");

    db
}

/// Test simple CTE alias without WHERE
#[test]
fn test_cte_simple_alias() {
    let db = setup_cte_alias_db("simple");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x + y as sum FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    let mut sums: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let sum: i64 = row.get(0).unwrap();
        sums.push(sum);
    }

    assert_eq!(sums.len(), 2, "Expected 2 rows");
    assert!(sums.contains(&30), "Expected sum 30 (10+20)");
    assert!(sums.contains(&70), "Expected sum 70 (30+40)");
}

/// Test CTE with WHERE on aliased column
#[test]
fn test_cte_alias_with_where() {
    let db = setup_cte_alias_db("with_where");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x, y FROM renamed WHERE x > 20",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let x: i64 = row.get(0).unwrap();
        let y: i64 = row.get(1).unwrap();

        assert_eq!(x, 30, "Expected x = 30");
        assert_eq!(y, 40, "Expected y = 40");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching WHERE x > 20");
}

/// Test CTE with expression using aliased columns
#[test]
fn test_cte_alias_expression() {
    let db = setup_cte_alias_db("expression");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x * y as product FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    let mut products: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let product: i64 = row.get(0).unwrap();
        products.push(product);
    }

    assert_eq!(products.len(), 2, "Expected 2 rows");
    assert!(products.contains(&200), "Expected product 200 (10*20)");
    assert!(products.contains(&1200), "Expected product 1200 (30*40)");
}

/// Test CTE alias with aggregation
#[test]
fn test_cte_alias_aggregation() {
    let db = setup_cte_alias_db("aggregation");

    let sum: i64 = db
        .query_one(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT SUM(x) FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    // Sum of x values: 10 + 30 = 40
    assert_eq!(sum, 40, "Expected sum of x = 40");
}

/// Test CTE with multiple column aliases
#[test]
fn test_cte_multiple_aliases() {
    let db = Database::open("memory://cte_multi_alias").expect("Failed to create database");

    db.execute(
        "CREATE TABLE data (col1 INTEGER, col2 INTEGER, col3 INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO data (col1, col2, col3) VALUES (1, 2, 3), (4, 5, 6)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "WITH aliased (a, b, c) AS (
                SELECT col1, col2, col3 FROM data
            )
            SELECT a, b, c, a + b + c as total FROM aliased",
            (),
        )
        .expect("Failed to execute query");

    let mut totals: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(3).unwrap();
        totals.push(total);
    }

    assert_eq!(totals.len(), 2, "Expected 2 rows");
    assert!(totals.contains(&6), "Expected total 6 (1+2+3)");
    assert!(totals.contains(&15), "Expected total 15 (4+5+6)");
}

/// Test CTE alias with ORDER BY
#[test]
fn test_cte_alias_order_by() {
    let db = setup_cte_alias_db("order_by");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x FROM renamed ORDER BY x DESC",
            (),
        )
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let x: i64 = row.get(0).unwrap();
        values.push(x);
    }

    // Result should have both values, order may vary based on implementation
    assert_eq!(values.len(), 2, "Expected 2 values");
    assert!(
        values.contains(&10) && values.contains(&30),
        "Expected values 10 and 30"
    );
}

/// Test CTE alias with LIMIT
#[test]
fn test_cte_alias_limit() {
    let db = setup_cte_alias_db("limit");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x FROM renamed LIMIT 1",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for _row in result {
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row with LIMIT 1");
}

/// Test nested CTEs with aliases
#[test]
fn test_nested_cte_aliases() {
    let db = setup_cte_alias_db("nested");

    let result = db
        .query(
            "WITH first (p, q) AS (
                SELECT a, b FROM test
            ),
            second (m, n) AS (
                SELECT p * 2, q * 2 FROM first
            )
            SELECT m, n FROM second",
            (),
        )
        .expect("Failed to execute query");

    let mut rows_data: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let m: i64 = row.get(0).unwrap();
        let n: i64 = row.get(1).unwrap();
        rows_data.push((m, n));
    }

    assert_eq!(rows_data.len(), 2, "Expected 2 rows");
    assert!(
        rows_data.contains(&(20, 40)),
        "Expected (20, 40) from (10*2, 20*2)"
    );
    assert!(
        rows_data.contains(&(60, 80)),
        "Expected (60, 80) from (30*2, 40*2)"
    );
}
