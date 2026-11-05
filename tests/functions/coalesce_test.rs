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

//! COALESCE Function Tests
//!
//! Tests the SQL COALESCE function with literals

use stoolap::Database;

/// Test COALESCE with first value non-null
#[test]
fn test_coalesce_first_non_null() {
    let db = Database::open("memory://coalesce_first").expect("Failed to create database");

    db.execute("CREATE TABLE test_table (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_table (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT COALESCE('Value', 'Default') FROM test_table", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(
            value, "Value",
            "Expected 'Value' when first arg is non-null"
        );
    }
}

/// Test COALESCE with first value null, second non-null
#[test]
fn test_coalesce_first_null() {
    let db = Database::open("memory://coalesce_null").expect("Failed to create database");

    db.execute("CREATE TABLE test_table (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_table (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT COALESCE(NULL, 'Default') FROM test_table", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(
            value, "Default",
            "Expected 'Default' when first arg is NULL"
        );
    }
}

/// Test COALESCE with single value
#[test]
fn test_coalesce_single_value() {
    let db = Database::open("memory://coalesce_single").expect("Failed to create database");

    db.execute("CREATE TABLE test_table (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_table (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT COALESCE('Single') FROM test_table", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(
            value, "Single",
            "Expected 'Single' for single-value COALESCE"
        );
    }
}

/// Test COALESCE with column values
#[test]
fn test_coalesce_with_columns() {
    let db = Database::open("memory://coalesce_cols").expect("Failed to create database");

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, name TEXT, alt_name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO data (id, name, alt_name) VALUES (1, 'Alice', 'A')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO data (id, name, alt_name) VALUES (2, NULL, 'Bob')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO data (id, name, alt_name) VALUES (3, NULL, NULL)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT id, COALESCE(name, alt_name, 'Unknown') as display_name FROM data ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut names: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        names.push((id, name));
    }

    assert_eq!(names.len(), 3);
    assert_eq!(
        names[0],
        (1, "Alice".to_string()),
        "First row should use name"
    );
    assert_eq!(
        names[1],
        (2, "Bob".to_string()),
        "Second row should use alt_name"
    );
    assert_eq!(
        names[2],
        (3, "Unknown".to_string()),
        "Third row should use default"
    );
}

/// Test COALESCE with numeric values
#[test]
fn test_coalesce_numeric() {
    let db = Database::open("memory://coalesce_num").expect("Failed to create database");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, val INTEGER, backup INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO numbers (id, val, backup) VALUES (1, 100, 50)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO numbers (id, val, backup) VALUES (2, NULL, 75)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT id, COALESCE(val, backup, 0) as result FROM numbers ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let result_val: i64 = row.get(1).unwrap();
        results.push((id, result_val));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], (1, 100), "First row should use val");
    assert_eq!(results[1], (2, 75), "Second row should use backup");
}

/// Test COALESCE in WHERE clause
#[test]
fn test_coalesce_in_where() {
    let db = Database::open("memory://coalesce_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items (id, status) VALUES (1, 'active')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO items (id, status) VALUES (2, NULL)", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO items (id, status) VALUES (3, 'inactive')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM items WHERE COALESCE(status, 'unknown') = 'active'",
            (),
        )
        .expect("Failed to execute query");

    assert_eq!(count, 1, "Expected 1 row with active status");
}

/// Test COALESCE with multiple NULL values
#[test]
fn test_coalesce_multiple_nulls() {
    let db = Database::open("memory://coalesce_multi_null").expect("Failed to create database");

    db.execute("CREATE TABLE test_table (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_table (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT COALESCE(NULL, NULL, NULL, 'Found') FROM test_table",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(value, "Found", "Expected 'Found' after multiple NULLs");
    }
}
