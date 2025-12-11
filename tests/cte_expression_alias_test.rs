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

//! CTE Expression Alias Tests
//!
//! Tests CTE with expressions on aliased columns

use stoolap::Database;

/// Test CTE expression with aliased columns and division
#[test]
fn test_cte_expression_alias_division() {
    let db = Database::open("memory://cte_expr_div").expect("Failed to create database");

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
        (10, 2),
        (30, 3)",
        (),
    )
    .expect("Failed to insert data");

    // Test: Expression with aliased columns
    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM test
            )
            SELECT x / y as result FROM renamed WHERE y > 0",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let result_val: i64 = row.get(0).unwrap();
        results.push(result_val);
    }

    assert_eq!(results.len(), 2, "Expected 2 rows");
    // 10/2 = 5, 30/3 = 10
    assert!(results.contains(&5), "Expected result 5 (10/2)");
    assert!(results.contains(&10), "Expected result 10 (30/3)");
}

/// Test CTE expression with multiplication
#[test]
fn test_cte_expression_alias_multiplication() {
    let db = Database::open("memory://cte_expr_mul").expect("Failed to create database");

    db.execute("CREATE TABLE nums (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO nums (a, b) VALUES (5, 3), (7, 4)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "WITH aliased (x, y) AS (
                SELECT a, b FROM nums
            )
            SELECT x * y as product FROM aliased",
            (),
        )
        .expect("Failed to execute query");

    let mut products: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let product: i64 = row.get(0).unwrap();
        products.push(product);
    }

    assert_eq!(products.len(), 2);
    assert!(products.contains(&15), "Expected 15 (5*3)");
    assert!(products.contains(&28), "Expected 28 (7*4)");
}

/// Test CTE expression with addition and subtraction
#[test]
fn test_cte_expression_alias_add_sub() {
    let db = Database::open("memory://cte_expr_add").expect("Failed to create database");

    db.execute("CREATE TABLE vals (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO vals (a, b) VALUES (100, 30), (50, 20)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "WITH renamed (x, y) AS (
                SELECT a, b FROM vals
            )
            SELECT x + y as sum_val, x - y as diff_val FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    let mut sums: Vec<i64> = Vec::new();
    let mut diffs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let sum_val: i64 = row.get(0).unwrap();
        let diff_val: i64 = row.get(1).unwrap();
        sums.push(sum_val);
        diffs.push(diff_val);
    }

    assert_eq!(sums.len(), 2);
    assert!(sums.contains(&130), "Expected sum 130 (100+30)");
    assert!(sums.contains(&70), "Expected sum 70 (50+20)");
    assert!(diffs.contains(&70), "Expected diff 70 (100-30)");
    assert!(diffs.contains(&30), "Expected diff 30 (50-20)");
}

/// Test CTE with complex expression
#[test]
fn test_cte_complex_expression() {
    let db = Database::open("memory://cte_expr_complex").expect("Failed to create database");

    db.execute("CREATE TABLE data (a INTEGER, b INTEGER, c INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO data (a, b, c) VALUES (10, 5, 2)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "WITH aliased (x, y, z) AS (
                SELECT a, b, c FROM data
            )
            SELECT (x + y) * z as result FROM aliased",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let result_val: i64 = row.get(0).unwrap();
        // (10 + 5) * 2 = 30
        assert_eq!(result_val, 30, "Expected (10+5)*2 = 30");
    }
}

/// Test CTE expression with aggregate functions
#[test]
fn test_cte_expression_aggregate() {
    let db = Database::open("memory://cte_expr_agg").expect("Failed to create database");

    db.execute("CREATE TABLE sales (amount INTEGER, quantity INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (amount, quantity) VALUES (100, 2), (200, 3), (150, 4)",
        (),
    )
    .expect("Failed to insert data");

    // Test simpler aggregate - SUM of individual columns
    let sum_amount: i64 = db
        .query_one(
            "WITH renamed (price, qty) AS (
                SELECT amount, quantity FROM sales
            )
            SELECT SUM(price) FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    // 100 + 200 + 150 = 450
    assert_eq!(sum_amount, 450, "Expected sum of prices 450");

    let sum_qty: i64 = db
        .query_one(
            "WITH renamed (price, qty) AS (
                SELECT amount, quantity FROM sales
            )
            SELECT SUM(qty) FROM renamed",
            (),
        )
        .expect("Failed to execute query");

    // 2 + 3 + 4 = 9
    assert_eq!(sum_qty, 9, "Expected sum of quantities 9");
}

/// Test CTE expression with modulo operator
#[test]
fn test_cte_expression_modulo() {
    let db = Database::open("memory://cte_expr_mod").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (num INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO numbers (num) VALUES (10), (15), (20), (25)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "WITH aliased (n) AS (
                SELECT num FROM numbers
            )
            SELECT n, n % 3 as remainder FROM aliased",
            (),
        )
        .expect("Failed to execute query");

    let mut remainders: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let n: i64 = row.get(0).unwrap();
        let rem: i64 = row.get(1).unwrap();
        remainders.push((n, rem));
    }

    assert_eq!(remainders.len(), 4);
    // 10 % 3 = 1, 15 % 3 = 0, 20 % 3 = 2, 25 % 3 = 1
}

/// Test CTE expression with CASE
#[test]
fn test_cte_expression_case() {
    let db = Database::open("memory://cte_expr_case").expect("Failed to create database");

    db.execute("CREATE TABLE scores (value INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO scores (value) VALUES (85), (55), (92), (40)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "WITH aliased (score) AS (
                SELECT value FROM scores
            )
            SELECT score,
                   CASE WHEN score >= 60 THEN 'PASS' ELSE 'FAIL' END as grade
            FROM aliased ORDER BY score",
            (),
        )
        .expect("Failed to execute query");

    let mut grades: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let score: i64 = row.get(0).unwrap();
        let grade: String = row.get(1).unwrap();
        grades.push((score, grade));
    }

    assert_eq!(grades.len(), 4);
}
