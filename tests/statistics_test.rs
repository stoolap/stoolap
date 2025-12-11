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

//! Tests for ANALYZE command and statistics infrastructure

use stoolap::api::Database;

/// Test basic ANALYZE command creates system tables
#[test]
fn test_analyze_creates_system_tables() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Run ANALYZE on the specific table using query (returns result set)
    let result: Vec<_> = db
        .query("ANALYZE test_table", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Check result shows 1 table analyzed
    assert_eq!(result.len(), 1);
    let tables_analyzed: i64 = result[0].get(0).expect("Failed to get value");
    assert_eq!(tables_analyzed, 1);

    // Verify system tables were created using query
    let tables: Vec<_> = db
        .query("SHOW TABLES", ())
        .expect("SHOW TABLES failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let table_names: Vec<String> = tables.iter().map(|r| r.get::<String>(0).unwrap()).collect();

    assert!(
        table_names.iter().any(|t| t == "_sys_table_stats"),
        "System table _sys_table_stats should exist"
    );
    assert!(
        table_names.iter().any(|t| t == "_sys_column_stats"),
        "System table _sys_column_stats should exist"
    );
}

/// Test ANALYZE collects table statistics
#[test]
fn test_analyze_collects_table_stats() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create and populate a test table
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'Apple', 100)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (2, 'Banana', 50)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (3, 'Cherry', 200)", ())
        .expect("Insert failed");

    // Run ANALYZE
    let _: Vec<_> = db
        .query("ANALYZE products", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Check table stats
    let stats: Vec<_> = db
        .query(
            "SELECT table_name, row_count FROM _sys_table_stats WHERE table_name = 'products'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    assert_eq!(stats.len(), 1, "Should have stats for products table");

    let table_name: String = stats[0].get(0).expect("Failed to get table name");
    let row_count: i64 = stats[0].get(1).expect("Failed to get row count");

    assert_eq!(table_name, "products");
    assert_eq!(row_count, 3, "Row count should be 3");
}

/// Test ANALYZE collects column statistics
#[test]
fn test_analyze_collects_column_stats() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create and populate a test table with various values
    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER, dept TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 50000, 'Engineering')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 60000, 'Engineering')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (3, 'Charlie', 55000, 'Sales')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (4, 'Diana', 70000, 'Engineering')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (5, 'Eve', 45000, 'Sales')",
        (),
    )
    .expect("Insert failed");

    // Run ANALYZE
    let _: Vec<_> = db
        .query("ANALYZE employees", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Check column stats for salary column
    let stats: Vec<_> = db
        .query(
            "SELECT column_name, distinct_count, min_value, max_value
             FROM _sys_column_stats
             WHERE table_name = 'employees' AND column_name = 'salary'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    assert_eq!(stats.len(), 1, "Should have stats for salary column");

    let distinct_count: i64 = stats[0].get(1).expect("Failed to get distinct count");
    assert_eq!(distinct_count, 5, "All salary values are distinct");

    // Check stats for dept column (should have 2 distinct values)
    let dept_stats: Vec<_> = db
        .query(
            "SELECT distinct_count FROM _sys_column_stats
             WHERE table_name = 'employees' AND column_name = 'dept'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let dept_distinct: i64 = dept_stats[0].get(0).expect("Failed to get distinct count");
    assert_eq!(dept_distinct, 2, "Should have 2 distinct departments");
}

/// Test ANALYZE handles NULL values correctly
#[test]
fn test_analyze_null_handling() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE nullable_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO nullable_test VALUES (1, 'A')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO nullable_test VALUES (2, NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO nullable_test VALUES (3, 'B')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO nullable_test VALUES (4, NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO nullable_test VALUES (5, 'A')", ())
        .expect("Insert failed");

    let _: Vec<_> = db
        .query("ANALYZE nullable_test", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let stats: Vec<_> = db
        .query(
            "SELECT null_count, distinct_count FROM _sys_column_stats
             WHERE table_name = 'nullable_test' AND column_name = 'value'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let null_count: i64 = stats[0].get(0).expect("Failed to get null count");
    let distinct_count: i64 = stats[0].get(1).expect("Failed to get distinct count");

    assert_eq!(null_count, 2, "Should have 2 NULL values");
    assert_eq!(
        distinct_count, 2,
        "Should have 2 distinct non-NULL values (A, B)"
    );
}

/// Test ANALYZE without table name analyzes all tables
#[test]
fn test_analyze_all_tables() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create multiple tables
    db.execute(
        "CREATE TABLE table1 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table1");
    db.execute(
        "CREATE TABLE table2 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table2");
    db.execute(
        "CREATE TABLE table3 (id INTEGER PRIMARY KEY, score FLOAT)",
        (),
    )
    .expect("Failed to create table3");

    // Insert some data
    db.execute("INSERT INTO table1 VALUES (1, 100)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO table2 VALUES (1, 'test')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO table3 VALUES (1, 3.14)", ())
        .expect("Insert failed");

    // ANALYZE all tables
    let result: Vec<_> = db
        .query("ANALYZE", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let analyzed_count: i64 = result[0].get(0).expect("Failed to get count");
    assert_eq!(analyzed_count, 3, "Should analyze 3 tables");

    // Verify stats exist for all tables
    let stats: Vec<_> = db
        .query(
            "SELECT COUNT(*) FROM _sys_table_stats WHERE table_name IN ('table1', 'table2', 'table3')",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let stats_count: i64 = stats[0].get(0).expect("Failed to get count");
    assert_eq!(stats_count, 3, "Should have stats for all 3 tables");
}

/// Test ANALYZE updates existing statistics
#[test]
fn test_analyze_updates_stats() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE update_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO update_test VALUES (1, 10)", ())
        .expect("Insert failed");

    // First ANALYZE
    let _: Vec<_> = db
        .query("ANALYZE update_test", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let first_stats: Vec<_> = db
        .query(
            "SELECT row_count FROM _sys_table_stats WHERE table_name = 'update_test'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let first_count: i64 = first_stats[0].get(0).expect("Failed to get count");
    assert_eq!(first_count, 1);

    // Add more data
    db.execute("INSERT INTO update_test VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO update_test VALUES (3, 30)", ())
        .expect("Insert failed");

    // Second ANALYZE
    let _: Vec<_> = db
        .query("ANALYZE update_test", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let second_stats: Vec<_> = db
        .query(
            "SELECT row_count FROM _sys_table_stats WHERE table_name = 'update_test'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let second_count: i64 = second_stats[0].get(0).expect("Failed to get count");
    assert_eq!(second_count, 3, "Stats should be updated to 3 rows");
}

/// Test ANALYZE handles empty tables
#[test]
fn test_analyze_empty_table() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE empty_table (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    let _: Vec<_> = db
        .query("ANALYZE empty_table", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let stats: Vec<_> = db
        .query(
            "SELECT row_count, avg_row_size FROM _sys_table_stats WHERE table_name = 'empty_table'",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    assert_eq!(stats.len(), 1, "Should have stats for empty table");

    let row_count: i64 = stats[0].get(0).expect("Failed to get row count");
    assert_eq!(row_count, 0, "Row count should be 0");
}

/// Test ANALYZE handles non-existent table gracefully
#[test]
fn test_analyze_nonexistent_table() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // ANALYZE on non-existent table prints warning but returns 0 tables analyzed
    let result: Vec<_> = db
        .query("ANALYZE nonexistent_table", ())
        .expect("ANALYZE should not error")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Should return 0 tables analyzed (with warning printed)
    let analyzed_count: i64 = result[0].get(0).expect("Failed to get count");
    assert_eq!(analyzed_count, 0, "Should return 0 for non-existent table");
}

/// Test system tables are excluded from ANALYZE all
#[test]
fn test_analyze_excludes_system_tables() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create a user table
    db.execute("CREATE TABLE user_table (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    // Run ANALYZE to create system tables
    let _: Vec<_> = db
        .query("ANALYZE", ())
        .expect("First ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Run ANALYZE again - system tables should be excluded
    let result: Vec<_> = db
        .query("ANALYZE", ())
        .expect("Second ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    let analyzed_count: i64 = result[0].get(0).expect("Failed to get count");

    // Should only analyze user_table, not the _sys_* tables
    assert_eq!(
        analyzed_count, 1,
        "Should only analyze user table, not system tables"
    );
}

/// Test selectivity estimator formulas
#[test]
fn test_selectivity_estimator() {
    use stoolap::storage::SelectivityEstimator;

    // Test equality selectivity
    let eq_sel = SelectivityEstimator::equality(100); // 100 distinct values
    assert!(
        (eq_sel - 0.01).abs() < 0.0001,
        "Equality selectivity should be 1/100"
    );

    // Test range selectivity
    let range_sel = SelectivityEstimator::range();
    assert!(
        (range_sel - 0.33).abs() < 0.01,
        "Range selectivity should be ~0.33"
    );

    // Test LIKE selectivity
    let prefix_sel = SelectivityEstimator::like("test%", 100);
    assert!(
        prefix_sel < 0.5,
        "Prefix LIKE should have reasonable selectivity"
    );

    let contains_sel = SelectivityEstimator::like("%test%", 100);
    assert!(
        contains_sel > prefix_sel,
        "Contains LIKE should be less selective than prefix"
    );

    // Test IN list selectivity
    let in_sel = SelectivityEstimator::in_list(5, 100); // 5 values out of 100 distinct
    assert!(
        (in_sel - 0.05).abs() < 0.0001,
        "IN list selectivity should be 5/100"
    );

    // Test NULL selectivity
    let null_sel = SelectivityEstimator::is_null(20, 100); // 20 nulls out of 100 rows
    assert!(
        (null_sel - 0.2).abs() < 0.0001,
        "IS NULL selectivity should be 20/100"
    );

    let not_null_sel = SelectivityEstimator::is_not_null(20, 100);
    assert!(
        (not_null_sel - 0.8).abs() < 0.0001,
        "IS NOT NULL selectivity should be 80/100"
    );
}

/// Test join cardinality estimation
#[test]
fn test_join_cardinality() {
    use stoolap::storage::SelectivityEstimator;

    // Test join cardinality estimation
    // Function signature: join_cardinality(left_rows, right_rows, left_distinct, right_distinct)
    // left: 1000 rows, 100 distinct
    // right: 500 rows, 50 distinct
    let cardinality = SelectivityEstimator::join_cardinality(1000, 500, 100, 50);

    // Expected: (1000 * 500) / max(100, 50) = 500000 / 100 = 5000
    assert_eq!(cardinality, 5000, "Join cardinality should be 5000");
}
