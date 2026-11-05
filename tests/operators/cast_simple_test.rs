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

//! Simple CAST Expression Tests
//!
//! Tests basic CAST functionality

use stoolap::Database;

/// Test simple query without CAST
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
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let val: String = row.get(1).unwrap();

        assert_eq!(id, 1);
        assert_eq!(val, "123");
        count += 1;
    }

    assert_eq!(count, 1, "Expected exactly one row");
}

/// Test CAST of literal string to INTEGER
#[test]
fn test_cast_literal() {
    let db = Database::open("memory://cast_literal").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (id) VALUES (1)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST('123' AS INTEGER) FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: i64 = row.get(0).unwrap();
        assert_eq!(cast_val, 123);
    }
}

/// Test CAST of column value to INTEGER
#[test]
fn test_cast_column() {
    let db = Database::open("memory://cast_column").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast_simple (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_cast_simple (id, val) VALUES (1, '123')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) FROM test_cast_simple", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: i64 = row.get(0).unwrap();
        assert_eq!(cast_val, 123);
    }
}

/// Test CAST string to FLOAT
#[test]
fn test_cast_string_to_float() {
    let db = Database::open("memory://cast_float").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES ('3.54159')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS FLOAT) FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: f64 = row.get(0).unwrap();
        assert!((cast_val - 3.54159).abs() < 0.00001);
    }
}

/// Test CAST integer to TEXT
#[test]
fn test_cast_int_to_text() {
    let db = Database::open("memory://cast_text").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES (42)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS TEXT) FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: String = row.get(0).unwrap();
        assert_eq!(cast_val, "42");
    }
}

/// Test CAST float to INTEGER (truncation)
#[test]
fn test_cast_float_to_int() {
    let db = Database::open("memory://cast_trunc").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val FLOAT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES (3.7)", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: i64 = row.get(0).unwrap();
        // Float to int should truncate towards zero
        assert_eq!(cast_val, 3);
    }
}

/// Test CAST with alias
#[test]
fn test_cast_with_alias() {
    let db = Database::open("memory://cast_alias").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES ('100')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) AS num FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let num: i64 = row.get(0).unwrap();
        assert_eq!(num, 100);
    }
}

/// Test CAST in WHERE clause
#[test]
fn test_cast_in_where() {
    let db = Database::open("memory://cast_where").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_cast (id, val) VALUES (1, '50'), (2, '100'), (3, '150')",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_cast WHERE CAST(val AS INTEGER) > 75",
            (),
        )
        .expect("Failed to count");

    assert_eq!(count, 2, "Expected 2 rows where CAST(val AS INTEGER) > 75");
}

/// Test CAST in expression
#[test]
fn test_cast_in_expression() {
    let db = Database::open("memory://cast_expr").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES ('10')", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT CAST(val AS INTEGER) * 2 AS doubled FROM test_cast",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let doubled: i64 = row.get(0).unwrap();
        assert_eq!(doubled, 20);
    }
}

/// Test CAST negative numbers
#[test]
fn test_cast_negative() {
    let db = Database::open("memory://cast_neg").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES ('-42')", ())
        .expect("Failed to insert data");

    let result = db
        .query("SELECT CAST(val AS INTEGER) FROM test_cast", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: i64 = row.get(0).unwrap();
        assert_eq!(cast_val, -42);
    }
}

/// Test CAST boolean to integer
#[test]
fn test_cast_bool_to_int() {
    let db = Database::open("memory://cast_bool").expect("Failed to create database");

    db.execute("CREATE TABLE test_cast (val BOOLEAN)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_cast (val) VALUES (true), (false)", ())
        .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT CAST(val AS INTEGER) FROM test_cast ORDER BY val DESC",
            (),
        )
        .expect("Failed to execute query");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cast_val: i64 = row.get(0).unwrap();
        values.push(cast_val);
    }

    assert_eq!(values.len(), 2);
    // true should be 1, false should be 0
    assert!(values.contains(&1));
    assert!(values.contains(&0));
}
