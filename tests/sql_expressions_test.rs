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

//! SQL Expression Capabilities Tests
//!
//! Tests various SQL expressions including literals, arithmetic, string functions, math functions

use stoolap::Database;

fn setup_expression_table(db: &Database) {
    db.execute(
        "CREATE TABLE expression_test (
            id INTEGER PRIMARY KEY,
            int_value INTEGER,
            float_value FLOAT,
            string_value TEXT,
            bool_value BOOLEAN
        )",
        (),
    )
    .expect("Failed to create test table");

    db.execute(
        "INSERT INTO expression_test
        (id, int_value, float_value, string_value, bool_value)
        VALUES
        (1, 100, 1.1, 'hello', true),
        (2, 200, 2.2, 'world', false),
        (3, 300, 3.3, 'SQL', true),
        (4, 400, 4.4, 'test', false),
        (5, 500, 5.5, 'EXPRESSIONS', true)",
        (),
    )
    .expect("Failed to insert test data");
}

// Basic Literals Tests

/// Test integer literal
#[test]
fn test_integer_literal() {
    let db = Database::open("memory://expr_int").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42", ()).expect("Failed to query");
    assert_eq!(result, 42);
}

/// Test float literal
#[test]
fn test_float_literal() {
    let db = Database::open("memory://expr_float").expect("Failed to create database");

    let result: f64 = db.query_one("SELECT 3.5", ()).expect("Failed to query");
    assert!((result - 3.5).abs() < 0.001);
}

/// Test string literal
#[test]
fn test_string_literal() {
    let db = Database::open("memory://expr_string").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT 'hello world'", ())
        .expect("Failed to query");
    assert_eq!(result, "hello world");
}

/// Test boolean literal true
#[test]
fn test_boolean_true() {
    let db = Database::open("memory://expr_bool_true").expect("Failed to create database");

    let result: bool = db.query_one("SELECT true", ()).expect("Failed to query");
    assert!(result);
}

/// Test boolean literal false
#[test]
fn test_boolean_false() {
    let db = Database::open("memory://expr_bool_false").expect("Failed to create database");

    let result: bool = db.query_one("SELECT false", ()).expect("Failed to query");
    assert!(!result);
}

// Arithmetic Expression Tests

/// Test addition
#[test]
fn test_addition() {
    let db = Database::open("memory://expr_add").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 1 + 1", ()).expect("Failed to query");
    assert_eq!(result, 2);
}

/// Test subtraction
#[test]
fn test_subtraction() {
    let db = Database::open("memory://expr_sub").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 5 - 2", ()).expect("Failed to query");
    assert_eq!(result, 3);
}

/// Test multiplication
#[test]
fn test_multiplication() {
    let db = Database::open("memory://expr_mul").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 3 * 4", ()).expect("Failed to query");
    assert_eq!(result, 12);
}

/// Test division
#[test]
fn test_division() {
    let db = Database::open("memory://expr_div").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 10 / 2", ()).expect("Failed to query");
    assert_eq!(result, 5);
}

// String Function Tests

/// Test UPPER function
#[test]
fn test_upper_function() {
    let db = Database::open("memory://expr_upper").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT UPPER('hello')", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO");
}

/// Test LOWER function
#[test]
fn test_lower_function() {
    let db = Database::open("memory://expr_lower").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT LOWER('WORLD')", ())
        .expect("Failed to query");
    assert_eq!(result, "world");
}

/// Test LENGTH function
#[test]
fn test_length_function() {
    let db = Database::open("memory://expr_length").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT LENGTH('test')", ())
        .expect("Failed to query");
    assert_eq!(result, 4);
}

/// Test SUBSTRING function
#[test]
fn test_substring_function() {
    let db = Database::open("memory://expr_substr").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT SUBSTRING('hello world', 1, 5)", ())
        .expect("Failed to query");
    assert_eq!(result, "hello");
}

// Math Function Tests

/// Test ABS function
#[test]
fn test_abs_function() {
    let db = Database::open("memory://expr_abs").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT ABS(-42)", ())
        .expect("Failed to query");
    assert_eq!(result, 42);
}

/// Test CEILING function
#[test]
fn test_ceiling_function() {
    let db = Database::open("memory://expr_ceil").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT CEILING(3.14)", ())
        .expect("Failed to query");
    assert_eq!(result, 4);
}

/// Test FLOOR function
#[test]
fn test_floor_function() {
    let db = Database::open("memory://expr_floor").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT FLOOR(3.99)", ())
        .expect("Failed to query");
    assert_eq!(result, 3);
}

/// Test ROUND function
#[test]
fn test_round_function() {
    let db = Database::open("memory://expr_round").expect("Failed to create database");

    let result: f64 = db
        .query_one("SELECT ROUND(3.54159, 2)", ())
        .expect("Failed to query");
    assert!((result - 3.54).abs() < 0.001);
}

// Date/Time Function Tests

/// Test NOW function
#[test]
fn test_now_function() {
    let db = Database::open("memory://expr_now").expect("Failed to create database");

    let result = db.query("SELECT NOW()", ()).expect("Failed to query");
    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 1);
}

// Column Expression Tests

/// Test column arithmetic
#[test]
fn test_column_arithmetic() {
    let db = Database::open("memory://expr_col_arith").expect("Failed to create database");
    setup_expression_table(&db);

    let result: i64 = db
        .query_one(
            "SELECT int_value + 10 FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 110); // 100 + 10
}

/// Test string concatenation
#[test]
fn test_string_concatenation() {
    let db = Database::open("memory://expr_concat").expect("Failed to create database");
    setup_expression_table(&db);

    let result: String = db
        .query_one(
            "SELECT string_value || ' test' FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "hello test");
}

/// Test string function on column
#[test]
fn test_string_function_on_column() {
    let db = Database::open("memory://expr_col_upper").expect("Failed to create database");
    setup_expression_table(&db);

    let result: String = db
        .query_one(
            "SELECT UPPER(string_value) FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "HELLO");
}

// Aliased Expression Tests

/// Test aliased literal
#[test]
fn test_aliased_literal() {
    let db = Database::open("memory://expr_alias_lit").expect("Failed to create database");

    let result = db
        .query("SELECT 42 AS answer", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        assert_eq!(val, 42);
    }
}

/// Test aliased expression
#[test]
fn test_aliased_expression() {
    let db = Database::open("memory://expr_alias_expr").expect("Failed to create database");
    setup_expression_table(&db);

    let result: i64 = db
        .query_one(
            "SELECT int_value * 2 AS doubled FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 200); // 100 * 2
}

/// Test aliased function
#[test]
fn test_aliased_function() {
    let db = Database::open("memory://expr_alias_func").expect("Failed to create database");
    setup_expression_table(&db);

    let result: String = db
        .query_one(
            "SELECT UPPER(string_value) AS shouting FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "HELLO");
}

// Complex Expression Tests

/// Test nested functions
#[test]
fn test_nested_functions() {
    let db = Database::open("memory://expr_nested").expect("Failed to create database");
    setup_expression_table(&db);

    let result: String = db
        .query_one(
            "SELECT UPPER(SUBSTRING(string_value, 1, 3)) FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "HEL");
}

/// Test compound arithmetic
#[test]
fn test_compound_arithmetic() {
    let db = Database::open("memory://expr_compound").expect("Failed to create database");
    setup_expression_table(&db);

    let result: i64 = db
        .query_one(
            "SELECT (int_value + 5) * 2 FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 210); // (100 + 5) * 2
}

/// Test mixed types arithmetic
#[test]
fn test_mixed_types() {
    let db = Database::open("memory://expr_mixed").expect("Failed to create database");
    setup_expression_table(&db);

    let result: f64 = db
        .query_one(
            "SELECT int_value + float_value FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert!((result - 101.1).abs() < 0.001); // 100 + 1.1
}

/// Test negative numbers
#[test]
fn test_negative_numbers() {
    let db = Database::open("memory://expr_negative").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT -42", ()).expect("Failed to query");
    assert_eq!(result, -42);
}

/// Test complex math expression
#[test]
fn test_complex_math() {
    let db = Database::open("memory://expr_complex_math").expect("Failed to create database");

    let result: f64 = db
        .query_one("SELECT (10.5 + 5) * 2 / 3", ())
        .expect("Failed to query");
    // (15.5 * 2) / 3 = 10.333...
    assert!((result - 10.333333).abs() < 0.001);
}

/// Test modulo operation
#[test]
fn test_modulo() {
    let db = Database::open("memory://expr_mod").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 17 % 5", ()).expect("Failed to query");
    assert_eq!(result, 2);
}

/// Test multiple expressions in SELECT
#[test]
fn test_multiple_expressions() {
    let db = Database::open("memory://expr_multi").expect("Failed to create database");
    setup_expression_table(&db);

    let result = db
        .query(
            "SELECT int_value, int_value * 2, int_value + 50 FROM expression_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let val1: i64 = row.get(0).unwrap();
        let val2: i64 = row.get(1).unwrap();
        let val3: i64 = row.get(2).unwrap();

        assert_eq!(val1, 100);
        assert_eq!(val2, 200);
        assert_eq!(val3, 150);
    }
}
