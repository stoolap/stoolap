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

//! COUNT Aggregation Tests
//!
//! Tests COUNT aggregation with various conditions

use stoolap::Database;

fn setup_count_test_table(db: &Database) {
    db.execute(
        "CREATE TABLE test_count (
            id INTEGER,
            name TEXT,
            value FLOAT,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create test table");

    // Insert 100 test rows
    for i in 1..=100 {
        let active = i % 2 == 0;
        let name = format!("Item {}", (64 + i % 26) as u8 as char);
        let value = i as f64 * 1.5;
        db.execute(
            &format!(
                "INSERT INTO test_count VALUES ({}, '{}', {:.1}, {})",
                i, name, value, active
            ),
            (),
        )
        .expect("Failed to insert data");
    }
}

/// Test COUNT(*) - All rows
#[test]
fn test_count_all_rows() {
    let db = Database::open("memory://count_all").expect("Failed to create database");
    setup_count_test_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count", ())
        .expect("Failed to execute COUNT(*)");

    assert_eq!(count, 100, "Expected count of 100");
}

/// Test COUNT(*) with WHERE clause
#[test]
fn test_count_with_where() {
    let db = Database::open("memory://count_where").expect("Failed to create database");
    setup_count_test_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count WHERE active = true", ())
        .expect("Failed to execute COUNT(*) with WHERE");

    assert_eq!(count, 50, "Expected count of 50 active rows");
}

/// Test COUNT(*) with complex WHERE
#[test]
fn test_count_with_complex_where() {
    let db = Database::open("memory://count_complex").expect("Failed to create database");
    setup_count_test_table(&db);

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_count WHERE id > 50 AND value < 100",
            (),
        )
        .expect("Failed to execute COUNT(*) with complex WHERE");

    // id > 50 means id 51-100 (50 rows)
    // value = id * 1.5, so value < 100 means id < 66.67, so id 51-66 (16 rows)
    assert_eq!(count, 16, "Expected count of 16");
}

/// Test COUNT with column
#[test]
fn test_count_column() {
    let db = Database::open("memory://count_column").expect("Failed to create database");
    setup_count_test_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(id) FROM test_count", ())
        .expect("Failed to execute COUNT(id)");

    assert_eq!(count, 100, "Expected count of 100");
}

/// Test COUNT DISTINCT
#[test]
fn test_count_distinct() {
    let db = Database::open("memory://count_distinct").expect("Failed to create database");
    setup_count_test_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT active) FROM test_count", ())
        .expect("Failed to execute COUNT(DISTINCT)");

    // There are only 2 distinct boolean values: true and false
    assert_eq!(count, 2, "Expected count of 2 distinct boolean values");
}

/// Test multiple aggregates in same query
#[test]
fn test_multiple_aggregates() {
    let db = Database::open("memory://count_multi").expect("Failed to create database");
    setup_count_test_table(&db);

    let result = db
        .query(
            "SELECT COUNT(*), MIN(value), MAX(value) FROM test_count",
            (),
        )
        .expect("Failed to execute multiple aggregates");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let count: i64 = row.get(0).unwrap();
        let min_val: f64 = row.get(1).unwrap();
        let max_val: f64 = row.get(2).unwrap();

        assert_eq!(count, 100);
        assert!((min_val - 1.5).abs() < 0.001); // 1 * 1.5
        assert!((max_val - 150.0).abs() < 0.001); // 100 * 1.5

        row_count += 1;
    }

    // Should return one row with three columns
    assert_eq!(row_count, 1, "Expected 1 row with multiple aggregates");
}

/// Test COUNT with GROUP BY
#[test]
fn test_count_with_group_by() {
    let db = Database::open("memory://count_group").expect("Failed to create database");
    setup_count_test_table(&db);

    let result = db
        .query(
            "SELECT active, COUNT(*) FROM test_count GROUP BY active",
            (),
        )
        .expect("Failed to execute COUNT with GROUP BY");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _active: bool = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();

        // Each group (true/false) should have 50 rows
        assert_eq!(count, 50);
        row_count += 1;
    }

    // One row for each active value (true/false)
    assert_eq!(row_count, 2, "Expected 2 rows from GROUP BY");
}

/// Test COUNT with simple table
#[test]
fn test_count_simple() {
    let db = Database::open("memory://count_simple").expect("Failed to create database");

    db.execute("CREATE TABLE simple_count (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO simple_count VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO simple_count VALUES (2, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO simple_count VALUES (3, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO simple_count VALUES (4, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO simple_count VALUES (5, 'B')", ())
        .unwrap();

    // Total count
    let total: i64 = db
        .query_one("SELECT COUNT(*) FROM simple_count", ())
        .expect("Failed to count");
    assert_eq!(total, 5);

    // Count with WHERE
    let a_count: i64 = db
        .query_one("SELECT COUNT(*) FROM simple_count WHERE category = 'A'", ())
        .expect("Failed to count A");
    assert_eq!(a_count, 2);

    let b_count: i64 = db
        .query_one("SELECT COUNT(*) FROM simple_count WHERE category = 'B'", ())
        .expect("Failed to count B");
    assert_eq!(b_count, 3);
}

/// Test COUNT with NULL values
#[test]
fn test_count_with_nulls() {
    let db = Database::open("memory://count_nulls").expect("Failed to create database");

    db.execute("CREATE TABLE count_nulls (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO count_nulls VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO count_nulls VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO count_nulls VALUES (3, 30)", ())
        .unwrap();
    db.execute("INSERT INTO count_nulls VALUES (4, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO count_nulls VALUES (5, 50)", ())
        .unwrap();

    // COUNT(*) should count all rows including those with NULL
    let total: i64 = db
        .query_one("SELECT COUNT(*) FROM count_nulls", ())
        .expect("Failed to count");
    assert_eq!(total, 5);

    // COUNT(column) should only count non-NULL values
    let value_count: i64 = db
        .query_one("SELECT COUNT(value) FROM count_nulls", ())
        .expect("Failed to count value");
    assert_eq!(value_count, 3, "COUNT(value) should exclude NULLs");
}

/// Test COUNT on empty table
#[test]
fn test_count_empty_table() {
    let db = Database::open("memory://count_empty").expect("Failed to create database");

    db.execute("CREATE TABLE empty_table (id INTEGER)", ())
        .expect("Failed to create table");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM empty_table", ())
        .expect("Failed to count empty table");

    assert_eq!(count, 0, "COUNT(*) on empty table should return 0");
}
