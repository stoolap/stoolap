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

//! CAST Expression Tests
//!
//! Tests the SQL CAST expression

use stoolap::Database;

/// Test simple query without CAST to verify table is working
#[test]
fn test_simple_query() {
    let db = Database::open("memory://cast_simple").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast_simple (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_cast_simple (id, val) VALUES (1, '123')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT id, val FROM test_cast_simple", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let val: String = row.get(1).unwrap();
        assert_eq!(id, 1);
        assert_eq!(val, "123");
        count += 1;
    }
    assert_eq!(count, 1);
}

/// Test CAST with literal value
#[test]
fn test_cast_literal() {
    let db = Database::open("memory://cast_literal").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast_simple (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast_simple (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result: i64 = db
        .query_one("SELECT CAST('123' AS INTEGER) FROM test_cast_simple", ())
        .expect("Failed to execute CAST query");

    assert_eq!(result, 123);
}

/// Test CAST with column reference
#[test]
fn test_cast_column() {
    let db = Database::open("memory://cast_column").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast_simple (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_cast_simple (id, val) VALUES (1, '456')",
        (),
    )
    .expect("Failed to insert data");

    let result: i64 = db
        .query_one("SELECT CAST(val AS INTEGER) FROM test_cast_simple", ())
        .expect("Failed to execute CAST query");

    assert_eq!(result, 456);
}

/// Test CAST integer to float
#[test]
fn test_cast_int_to_float() {
    let db = Database::open("memory://cast_int_float").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO numbers (id, value) VALUES (1, 42)", ())
        .expect("Failed to insert data");

    let result: f64 = db
        .query_one("SELECT CAST(value AS FLOAT) FROM numbers", ())
        .expect("Failed to execute CAST query");

    assert!((result - 42.0).abs() < 0.001);
}

/// Test CAST float to integer
#[test]
fn test_cast_float_to_int() {
    let db = Database::open("memory://cast_float_int").expect("Failed to create database");

    db.execute("CREATE TABLE floats (id INTEGER, value FLOAT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO floats (id, value) VALUES (1, 42.7)", ())
        .expect("Failed to insert data");

    let result: i64 = db
        .query_one("SELECT CAST(value AS INTEGER) FROM floats", ())
        .expect("Failed to execute CAST query");

    // CAST truncates toward zero
    assert_eq!(result, 42);
}

/// Test CAST integer to text
#[test]
fn test_cast_int_to_text() {
    let db = Database::open("memory://cast_int_text").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO numbers (id, value) VALUES (1, 123)", ())
        .expect("Failed to insert data");

    let result: String = db
        .query_one("SELECT CAST(value AS TEXT) FROM numbers", ())
        .expect("Failed to execute CAST query");

    assert_eq!(result, "123");
}

/// Test CAST float to text
#[test]
fn test_cast_float_to_text() {
    let db = Database::open("memory://cast_float_text").expect("Failed to create database");

    db.execute("CREATE TABLE floats (id INTEGER, value FLOAT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO floats (id, value) VALUES (1, 3.14)", ())
        .expect("Failed to insert data");

    let result: String = db
        .query_one("SELECT CAST(value AS TEXT) FROM floats", ())
        .expect("Failed to execute CAST query");

    // The result should contain 3.14
    assert!(
        result.starts_with("3.14"),
        "Expected string starting with 3.14, got {}",
        result
    );
}

/// Test CAST in WHERE clause
#[test]
fn test_cast_in_where() {
    let db = Database::open("memory://cast_where").expect("Failed to create database");

    db.execute("CREATE TABLE strings (id INTEGER, value TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO strings (id, value) VALUES (1, '10')", ())
        .unwrap();
    db.execute("INSERT INTO strings (id, value) VALUES (2, '20')", ())
        .unwrap();
    db.execute("INSERT INTO strings (id, value) VALUES (3, '30')", ())
        .unwrap();
    db.execute("INSERT INTO strings (id, value) VALUES (4, '5')", ())
        .unwrap();

    // Use CAST in WHERE to compare as integers
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM strings WHERE CAST(value AS INTEGER) > 15",
            (),
        )
        .expect("Failed to count");

    // 20 and 30 are > 15
    assert_eq!(count, 2);
}

/// Test CAST with arithmetic
#[test]
fn test_cast_with_arithmetic() {
    let db = Database::open("memory://cast_arith").expect("Failed to create database");

    db.execute("CREATE TABLE values_table (id INTEGER, value TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO values_table (id, value) VALUES (1, '100')", ())
        .expect("Failed to insert data");

    let result: i64 = db
        .query_one("SELECT CAST(value AS INTEGER) + 50 FROM values_table", ())
        .expect("Failed to execute CAST with arithmetic");

    assert_eq!(result, 150);
}

/// Test multiple CASTs in same query
#[test]
fn test_multiple_casts() {
    let db = Database::open("memory://cast_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE mixed (id INTEGER, int_val TEXT, float_val TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO mixed (id, int_val, float_val) VALUES (1, '10', '20.5')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT CAST(int_val AS INTEGER), CAST(float_val AS FLOAT) FROM mixed",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let int_result: i64 = row.get(0).unwrap();
        let float_result: f64 = row.get(1).unwrap();
        assert_eq!(int_result, 10);
        assert!((float_result - 20.5).abs() < 0.001);
    }
}
