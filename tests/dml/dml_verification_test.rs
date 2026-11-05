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

//! DML Function Verification Tests
//!
//! Tests to verify that functions work correctly in INSERT statements

use stoolap::Database;

/// Test UPPER function works correctly in INSERT
#[test]
fn test_dml_upper_function() {
    let db = Database::open("memory://dml_upper").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, UPPER('hello world'))",
        (),
    )
    .expect("Failed to insert with UPPER");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "HELLO WORLD", "Expected 'HELLO WORLD'");
}

/// Test CONCAT function works correctly in INSERT
#[test]
fn test_dml_concat_function() {
    let db = Database::open("memory://dml_concat").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, CONCAT('Hello', ' ', 'World', '!'))", ())
    .expect("Failed to insert with CONCAT");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "Hello World!", "Expected 'Hello World!'");
}

/// Test arithmetic expressions work correctly in INSERT
#[test]
fn test_dml_arithmetic() {
    let db = Database::open("memory://dml_arith").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            number_result FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, number_result) VALUES (1, 10.0 + 5.0 * 2.0)",
        (),
    )
    .expect("Failed to insert with arithmetic");

    let result: f64 = db
        .query_one(
            "SELECT number_result FROM verification_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    // Order of operations: 10 + (5 * 2) = 20
    assert!(
        (result - 20.0).abs() < 0.01,
        "Expected 20.0, got {}",
        result
    );
}

/// Test nested functions work correctly in INSERT
#[test]
fn test_dml_nested_functions() {
    let db = Database::open("memory://dml_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, UPPER(CONCAT('nested', ' test')))", ())
    .expect("Failed to insert with nested functions");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "NESTED TEST", "Expected 'NESTED TEST'");
}

/// Test CAST function works correctly in INSERT
#[test]
fn test_dml_cast_function() {
    let db = Database::open("memory://dml_cast").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, CAST(12345 AS TEXT))",
        (),
    )
    .expect("Failed to insert with CAST");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "12345", "Expected '12345'");
}

/// Test CASE expression works correctly in INSERT
#[test]
fn test_dml_case_expression() {
    let db = Database::open("memory://dml_case").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, CASE WHEN 5 > 3 THEN 'greater' ELSE 'less' END)", ())
    .expect("Failed to insert with CASE");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "greater", "Expected 'greater'");
}

/// Test NOW function works correctly in INSERT
#[test]
fn test_dml_now_function() {
    let db = Database::open("memory://dml_now").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            timestamp_result TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, timestamp_result) VALUES (1, NOW())",
        (),
    )
    .expect("Failed to insert with NOW");

    let result: String = db
        .query_one(
            "SELECT timestamp_result FROM verification_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    // Just verify we got a non-empty timestamp
    assert!(
        !result.is_empty(),
        "Expected non-empty timestamp from NOW()"
    );
}

/// Test complex expression with multiple functions in INSERT
#[test]
fn test_dml_complex_expression() {
    let db = Database::open("memory://dml_complex").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // This tests: UPPER(CONCAT('Result: ', CAST((3 + 4) * 2 AS TEXT)))
    // (3 + 4) * 2 = 14
    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, UPPER(CONCAT('Result: ', CAST((3 + 4) * 2 AS TEXT))))", ())
    .expect("Failed to insert with complex expression");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    // The result format might vary - it could be "RESULT: 14" or "RESULT: 14.000000"
    assert!(
        result.starts_with("RESULT: 14"),
        "Expected result starting with 'RESULT: 14', got '{}'",
        result
    );
}

/// Test that INSERT and SELECT produce consistent results for same expression
#[test]
fn test_dml_select_consistency() {
    let db = Database::open("memory://dml_consist").expect("Failed to create database");

    db.execute(
        "CREATE TABLE comparison_test (
            id INTEGER PRIMARY KEY,
            insert_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Test UPPER function consistency
    db.execute(
        "INSERT INTO comparison_test (id, insert_result) VALUES (1, UPPER('hello'))",
        (),
    )
    .expect("Failed to insert");

    let insert_result: String = db
        .query_one("SELECT insert_result FROM comparison_test WHERE id = 1", ())
        .expect("Failed to query insert result");

    let select_result: String = db
        .query_one("SELECT UPPER('hello')", ())
        .expect("Failed to query SELECT result");

    assert_eq!(
        insert_result, select_result,
        "INSERT and SELECT should produce same result for UPPER"
    );

    // Test CONCAT function consistency
    db.execute(
        "INSERT INTO comparison_test (id, insert_result) VALUES (2, CONCAT('a', 'b', 'c'))",
        (),
    )
    .expect("Failed to insert");

    let insert_result: String = db
        .query_one("SELECT insert_result FROM comparison_test WHERE id = 2", ())
        .expect("Failed to query insert result");

    let select_result: String = db
        .query_one("SELECT CONCAT('a', 'b', 'c')", ())
        .expect("Failed to query SELECT result");

    assert_eq!(
        insert_result, select_result,
        "INSERT and SELECT should produce same result for CONCAT"
    );
}

/// Test that invalid function name returns error
#[test]
fn test_dml_invalid_function() {
    let db = Database::open("memory://dml_invalid").expect("Failed to create database");

    db.execute(
        "CREATE TABLE error_test (
            id INTEGER PRIMARY KEY,
            result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    let result = db.execute(
        "INSERT INTO error_test (id, result) VALUES (1, INVALID_FUNCTION('test'))",
        (),
    );

    assert!(result.is_err(), "Expected error for non-existent function");
}

/// Test LOWER function in INSERT
#[test]
fn test_dml_lower_function() {
    let db = Database::open("memory://dml_lower").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, LOWER('HELLO WORLD'))",
        (),
    )
    .expect("Failed to insert with LOWER");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "hello world", "Expected 'hello world'");
}

/// Test LENGTH function in INSERT
#[test]
fn test_dml_length_function() {
    let db = Database::open("memory://dml_length").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            int_result INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, int_result) VALUES (1, LENGTH('hello'))",
        (),
    )
    .expect("Failed to insert with LENGTH");

    let result: i64 = db
        .query_one("SELECT int_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, 5, "Expected length 5");
}

/// Test COALESCE function in INSERT
#[test]
fn test_dml_coalesce_function() {
    let db = Database::open("memory://dml_coalesce").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            text_result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, text_result) VALUES (1, COALESCE(NULL, 'default'))",
        (),
    )
    .expect("Failed to insert with COALESCE");

    let result: String = db
        .query_one("SELECT text_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, "default", "Expected 'default'");
}

/// Test ABS function in INSERT
#[test]
fn test_dml_abs_function() {
    let db = Database::open("memory://dml_abs").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            int_result INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, int_result) VALUES (1, ABS(-42))",
        (),
    )
    .expect("Failed to insert with ABS");

    let result: i64 = db
        .query_one("SELECT int_result FROM verification_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(result, 42, "Expected 42");
}

/// Test ROUND function in INSERT
#[test]
fn test_dml_round_function() {
    let db = Database::open("memory://dml_round").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            float_result FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, float_result) VALUES (1, ROUND(3.54159, 2))",
        (),
    )
    .expect("Failed to insert with ROUND");

    let result: f64 = db
        .query_one(
            "SELECT float_result FROM verification_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    assert!(
        (result - 3.54).abs() < 0.01,
        "Expected 3.54, got {}",
        result
    );
}

/// Test multiple functions in a single INSERT
#[test]
fn test_dml_multiple_functions_single_insert() {
    let db = Database::open("memory://dml_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE verification_test (
            id INTEGER PRIMARY KEY,
            col1 TEXT,
            col2 TEXT,
            col3 INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO verification_test (id, col1, col2, col3) VALUES (1, UPPER('hello'), LOWER('WORLD'), ABS(-100))", ())
    .expect("Failed to insert with multiple functions");

    let result = db
        .query(
            "SELECT col1, col2, col3 FROM verification_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let col1: String = row.get(0).unwrap();
        let col2: String = row.get(1).unwrap();
        let col3: i64 = row.get(2).unwrap();

        assert_eq!(col1, "HELLO");
        assert_eq!(col2, "world");
        assert_eq!(col3, 100);
    }
}
