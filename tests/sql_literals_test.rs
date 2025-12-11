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

//! SQL Literals Tests
//!
//! Tests SQL literal values and scalar functions

use stoolap::Database;

/// Test integer literals
#[test]
fn test_integer_literals() {
    let db = Database::open("memory://int_lit").expect("Failed to create database");

    let result = db
        .query("SELECT 1, 2, 3", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: i64 = row.get(0).unwrap();
        let b: i64 = row.get(1).unwrap();
        let c: i64 = row.get(2).unwrap();

        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }
}

/// Test float literals
#[test]
fn test_float_literals() {
    let db = Database::open("memory://float_lit").expect("Failed to create database");

    let result = db
        .query("SELECT 1.5, 2.25, 3.75", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: f64 = row.get(0).unwrap();
        let b: f64 = row.get(1).unwrap();
        let c: f64 = row.get(2).unwrap();

        assert!((a - 1.5).abs() < 0.001);
        assert!((b - 2.25).abs() < 0.001);
        assert!((c - 3.75).abs() < 0.001);
    }
}

/// Test string literals
#[test]
fn test_string_literals() {
    let db = Database::open("memory://str_lit").expect("Failed to create database");

    let result = db
        .query("SELECT 'hello', 'world', 'sql'", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: String = row.get(0).unwrap();
        let b: String = row.get(1).unwrap();
        let c: String = row.get(2).unwrap();

        assert_eq!(a, "hello");
        assert_eq!(b, "world");
        assert_eq!(c, "sql");
    }
}

/// Test boolean literals
#[test]
fn test_boolean_literals() {
    let db = Database::open("memory://bool_lit").expect("Failed to create database");

    let result = db
        .query("SELECT true, false", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: bool = row.get(0).unwrap();
        let b: bool = row.get(1).unwrap();

        assert!(a);
        assert!(!b);
    }
}

/// Test arithmetic expressions
#[test]
fn test_arithmetic_expressions() {
    let db = Database::open("memory://arith").expect("Failed to create database");

    let result = db
        .query("SELECT 1 + 2, 3 - 1, 2 * 3, 10 / 2, 10 % 3", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let add: i64 = row.get(0).unwrap();
        let sub: i64 = row.get(1).unwrap();
        let mul: i64 = row.get(2).unwrap();
        let div: i64 = row.get(3).unwrap();
        let modulo: i64 = row.get(4).unwrap();

        assert_eq!(add, 3, "1 + 2 should be 3");
        assert_eq!(sub, 2, "3 - 1 should be 2");
        assert_eq!(mul, 6, "2 * 3 should be 6");
        assert_eq!(div, 5, "10 / 2 should be 5");
        assert_eq!(modulo, 1, "10 % 3 should be 1");
    }
}

/// Test mixed arithmetic expressions
#[test]
fn test_mixed_arithmetic() {
    let db = Database::open("memory://mixed_arith").expect("Failed to create database");

    let result = db
        .query("SELECT 1 + 2.5, 5.5 - 2, 2 * 3.5", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: f64 = row.get(0).unwrap();
        let b: f64 = row.get(1).unwrap();
        let c: f64 = row.get(2).unwrap();

        assert!((a - 3.5).abs() < 0.001, "1 + 2.5 should be 3.5");
        assert!((b - 3.5).abs() < 0.001, "5.5 - 2 should be 3.5");
        assert!((c - 7.0).abs() < 0.001, "2 * 3.5 should be 7.0");
    }
}

/// Test column aliases
#[test]
fn test_column_aliases() {
    let db = Database::open("memory://aliases").expect("Failed to create database");

    let result = db
        .query("SELECT 1 AS one, 2 AS two, 3 AS three", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let one: i64 = row.get(0).unwrap();
        let two: i64 = row.get(1).unwrap();
        let three: i64 = row.get(2).unwrap();

        assert_eq!(one, 1);
        assert_eq!(two, 2);
        assert_eq!(three, 3);
    }
}

/// Test UPPER function
#[test]
fn test_upper_function() {
    let db = Database::open("memory://upper_fn").expect("Failed to create database");

    let result = db
        .query("SELECT UPPER('hello world') AS greeting", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let greeting: String = row.get(0).unwrap();
        assert_eq!(greeting, "HELLO WORLD");
    }
}

/// Test LOWER function
#[test]
fn test_lower_function() {
    let db = Database::open("memory://lower_fn").expect("Failed to create database");

    let result = db
        .query("SELECT LOWER('HELLO WORLD') AS greeting", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let greeting: String = row.get(0).unwrap();
        assert_eq!(greeting, "hello world");
    }
}

/// Test LENGTH function
#[test]
fn test_length_function() {
    let db = Database::open("memory://length_fn").expect("Failed to create database");

    let result = db
        .query("SELECT LENGTH('hello') AS len", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let len: i64 = row.get(0).unwrap();
        assert_eq!(len, 5);
    }
}

/// Test CONCAT function
#[test]
fn test_concat_function() {
    let db = Database::open("memory://concat_fn").expect("Failed to create database");

    let result = db
        .query("SELECT CONCAT('hello', ' ', 'world') AS message", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let message: String = row.get(0).unwrap();
        assert_eq!(message, "hello world");
    }
}

/// Test ABS function
#[test]
fn test_abs_function() {
    let db = Database::open("memory://abs_fn").expect("Failed to create database");

    let result = db
        .query("SELECT ABS(-5) AS abs_value", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let abs_value: f64 = row.get(0).unwrap();
        assert!((abs_value - 5.0).abs() < 0.001);
    }
}

/// Test ROUND function
#[test]
fn test_round_function() {
    let db = Database::open("memory://round_fn").expect("Failed to create database");

    let result = db
        .query("SELECT ROUND(3.54159, 2) AS rounded", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let rounded: f64 = row.get(0).unwrap();
        assert!((rounded - 3.54).abs() < 0.001);
    }
}

/// Test CEILING function
#[test]
fn test_ceiling_function() {
    let db = Database::open("memory://ceil_fn").expect("Failed to create database");

    let result = db
        .query("SELECT CEILING(3.14) AS ceil_value", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let ceil_value: f64 = row.get(0).unwrap();
        assert!((ceil_value - 4.0).abs() < 0.001);
    }
}

/// Test FLOOR function
#[test]
fn test_floor_function() {
    let db = Database::open("memory://floor_fn").expect("Failed to create database");

    let result = db
        .query("SELECT FLOOR(3.99) AS floor_value", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let floor_value: f64 = row.get(0).unwrap();
        assert!((floor_value - 3.0).abs() < 0.001);
    }
}

/// Test SUBSTRING function
#[test]
fn test_substring_function() {
    let db = Database::open("memory://substr_fn").expect("Failed to create database");

    let result = db
        .query("SELECT SUBSTRING('hello world', 7, 5) AS sub", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let sub: String = row.get(0).unwrap();
        assert_eq!(sub, "world");
    }
}

/// Test COALESCE function
#[test]
fn test_coalesce_function() {
    let db = Database::open("memory://coalesce_fn").expect("Failed to create database");

    let result = db
        .query("SELECT COALESCE(NULL, 'default') AS value", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(0).unwrap();
        assert_eq!(value, "default");
    }
}

/// Test NOW function
#[test]
fn test_now_function() {
    let db = Database::open("memory://now_fn").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() AS current_time", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let timestamp: String = row.get(0).unwrap();
        // Just verify we get a non-empty string (timestamp format may vary)
        assert!(!timestamp.is_empty(), "NOW() should return a timestamp");
    }
}

/// Test mixed functions
#[test]
fn test_mixed_functions() {
    let db = Database::open("memory://mixed_fn").expect("Failed to create database");

    let result = db
        .query(
            "SELECT UPPER(CONCAT('hello', ' ', 'world')) AS message, LENGTH('hello world') AS len",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let message: String = row.get(0).unwrap();
        let len: i64 = row.get(1).unwrap();

        assert_eq!(message, "HELLO WORLD");
        assert_eq!(len, 11);
    }
}

/// Test CAST function - string to int
#[test]
fn test_cast_string_to_int() {
    let db = Database::open("memory://cast_int").expect("Failed to create database");

    let result = db
        .query("SELECT CAST('123' AS INTEGER) AS cast_int", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_int: i64 = row.get(0).unwrap();
        assert_eq!(cast_int, 123);
    }
}

/// Test CAST function - string to float
#[test]
fn test_cast_string_to_float() {
    let db = Database::open("memory://cast_float").expect("Failed to create database");

    let result = db
        .query("SELECT CAST('3.5' AS FLOAT) AS cast_float", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let cast_float: f64 = row.get(0).unwrap();
        assert!((cast_float - 3.5).abs() < 0.001);
    }
}

/// Test many literals in single query
#[test]
fn test_many_literals() {
    let db = Database::open("memory://many_lit").expect("Failed to create database");

    // Build query with 50 literals
    let mut query = String::from("SELECT ");
    for i in 1..=50 {
        query.push_str(&i.to_string());
        if i < 50 {
            query.push_str(", ");
        }
    }

    let result = db.query(&query, ()).expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        // Verify first and last values
        let first: i64 = row.get(0).unwrap();
        let last: i64 = row.get(49).unwrap();

        assert_eq!(first, 1);
        assert_eq!(last, 50);
    }
}

/// Test NULL literal
#[test]
fn test_null_literal() {
    let db = Database::open("memory://null_lit").expect("Failed to create database");

    let result = db
        .query("SELECT NULL AS null_val", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let null_val: Option<String> = row.get(0).unwrap();
        assert!(null_val.is_none(), "Expected NULL value");
    }
}

/// Test negative numbers
#[test]
fn test_negative_numbers() {
    let db = Database::open("memory://negative").expect("Failed to create database");

    let result = db
        .query("SELECT -5, -3.5, -100", ())
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let a: i64 = row.get(0).unwrap();
        let b: f64 = row.get(1).unwrap();
        let c: i64 = row.get(2).unwrap();

        assert_eq!(a, -5);
        assert!((b - (-3.5)).abs() < 0.001);
        assert_eq!(c, -100);
    }
}
