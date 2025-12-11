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

//! Scan with WHERE Clause Tests
//!
//! Tests that scanning works correctly when WHERE clause references columns not in SELECT

use stoolap::Database;

/// Test selecting subset of columns with WHERE on different column
#[test]
fn test_scan_with_where_clause() {
    let db = Database::open("memory://scan_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_scan_table (id INTEGER PRIMARY KEY, val TEXT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_scan_table (id, val, active) VALUES (1, 'value1', true)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO test_scan_table (id, val, active) VALUES (2, 'value2', false)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO test_scan_table (id, val, active) VALUES (3, 'value3', true)",
        (),
    )
    .expect("Failed to insert data");

    // Query only selects 'val' and 'active', but WHERE clause needs 'id'
    let result = db
        .query("SELECT val, active FROM test_scan_table WHERE id = 1", ())
        .expect("Failed to execute query");

    let mut rows_found = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let val: String = row.get(0).unwrap();
        let active: bool = row.get(1).unwrap();

        assert_eq!(val, "value1", "Expected val='value1'");
        assert!(active, "Expected active=true");
        rows_found += 1;
    }

    assert_eq!(rows_found, 1, "Expected exactly 1 row");
}

/// Test complex WHERE with columns not in SELECT
#[test]
fn test_extra_columns_in_where() {
    let db = Database::open("memory://scan_extra_where").expect("Failed to create database");

    db.execute("CREATE TABLE complex_table (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, active BOOLEAN, status TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO complex_table (id, name, age, active, status) VALUES (1, 'John', 25, true, 'employed')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO complex_table (id, name, age, active, status) VALUES (2, 'Jane', 30, false, 'unemployed')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO complex_table (id, name, age, active, status) VALUES (3, 'Bob', 40, true, 'retired')", ())
        .expect("Failed to insert data");

    // SELECT only name and status, WHERE uses id, age, and active
    let result = db
        .query(
            "SELECT name, status FROM complex_table WHERE id = 1 AND age > 20 AND active = true",
            (),
        )
        .expect("Failed to execute query");

    let mut rows_found = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let status: String = row.get(1).unwrap();

        assert_eq!(name, "John", "Expected name='John'");
        assert_eq!(status, "employed", "Expected status='employed'");
        rows_found += 1;
    }

    assert_eq!(rows_found, 1, "Expected exactly 1 row");
}

/// Test SELECT * with WHERE
#[test]
fn test_select_star_with_where() {
    let db = Database::open("memory://scan_star_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE star_test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO star_test (id, name, value) VALUES (1, 'first', 100)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO star_test (id, name, value) VALUES (2, 'second', 200)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO star_test (id, name, value) VALUES (3, 'third', 300)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT * FROM star_test WHERE value > 150", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let value: i64 = row.get(2).unwrap();

        assert!(value > 150, "Expected value > 150");
        assert!(id == 2 || id == 3, "Expected id 2 or 3");
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 rows with value > 150");
}

/// Test WHERE with multiple conditions on same column type
#[test]
fn test_where_multiple_conditions() {
    let db = Database::open("memory://scan_multi_cond").expect("Failed to create database");

    db.execute(
        "CREATE TABLE range_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(
            &format!(
                "INSERT INTO range_test (id, value) VALUES ({}, {})",
                i,
                i * 10
            ),
            (),
        )
        .expect("Failed to insert data");
    }

    // SELECT only id, WHERE uses value
    let result = db
        .query(
            "SELECT id FROM range_test WHERE value >= 30 AND value <= 70",
            (),
        )
        .expect("Failed to execute query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Should get ids 3, 4, 5, 6, 7 (values 30, 40, 50, 60, 70)
    assert_eq!(ids.len(), 5, "Expected 5 rows in range");
}

/// Test WHERE with OR conditions
#[test]
fn test_where_or_conditions() {
    let db = Database::open("memory://scan_or_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE or_test (id INTEGER PRIMARY KEY, category TEXT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO or_test (id, category, active) VALUES (1, 'A', true)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO or_test (id, category, active) VALUES (2, 'B', false)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO or_test (id, category, active) VALUES (3, 'C', true)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO or_test (id, category, active) VALUES (4, 'A', false)",
        (),
    )
    .expect("Failed to insert data");

    // SELECT only id and category, WHERE uses both
    let result = db
        .query(
            "SELECT id, category FROM or_test WHERE category = 'A' OR active = true ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let cat: String = row.get(1).unwrap();
        results.push((id, cat));
    }

    // Should match: id=1 (A, true), id=3 (C, true), id=4 (A, false)
    assert_eq!(results.len(), 3, "Expected 3 rows matching OR condition");
}
