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

//! DML with Functions Tests
//!
//! Tests INSERT and UPDATE statements with function calls

use stoolap::Database;

/// Test INSERT with UPPER() function
#[test]
fn test_insert_with_upper() {
    let db = Database::open("memory://insert_upper").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_functions (
            id INTEGER PRIMARY KEY,
            name TEXT,
            upper_name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_functions (id, name, upper_name) VALUES (1, 'hello', UPPER('hello'))",
        (),
    )
    .expect("Failed to insert with UPPER()");

    let result: String = db
        .query_one("SELECT upper_name FROM test_functions WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO");
}

/// Test INSERT with LENGTH() function
#[test]
fn test_insert_with_length() {
    let db = Database::open("memory://insert_length").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_functions (
            id INTEGER PRIMARY KEY,
            name TEXT,
            length_name INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_functions (id, name, length_name) VALUES (1, 'world', LENGTH('world'))",
        (),
    )
    .expect("Failed to insert with LENGTH()");

    let result: i64 = db
        .query_one("SELECT length_name FROM test_functions WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 5);
}

/// Test INSERT with CONCAT() function
#[test]
fn test_insert_with_concat() {
    let db = Database::open("memory://insert_concat").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_functions (
            id INTEGER PRIMARY KEY,
            concat_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_functions (id, concat_value) VALUES (1, CONCAT('Hello', ' ', 'World'))",
        (),
    )
    .expect("Failed to insert with CONCAT()");

    let result: String = db
        .query_one("SELECT concat_value FROM test_functions WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "Hello World");
}

/// Test INSERT with arithmetic expression
#[test]
fn test_insert_with_arithmetic() {
    let db = Database::open("memory://insert_arith").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_functions (
            id INTEGER PRIMARY KEY,
            math_result FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_functions (id, math_result) VALUES (1, 10.5 * 2 + 1)",
        (),
    )
    .expect("Failed to insert with arithmetic");

    let result: f64 = db
        .query_one("SELECT math_result FROM test_functions WHERE id = 1", ())
        .expect("Failed to query");
    assert!(
        (result - 22.0).abs() < 0.001,
        "Expected 22.0, got {}",
        result
    );
}

/// Test INSERT with nested functions
#[test]
fn test_insert_with_nested_functions() {
    let db = Database::open("memory://insert_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_functions (
            id INTEGER PRIMARY KEY,
            result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_functions (id, result) VALUES (1, UPPER(CONCAT('hello', ' world')))",
        (),
    )
    .expect("Failed to insert with nested functions");

    let result: String = db
        .query_one("SELECT result FROM test_functions WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO WORLD");
}

/// Test UPDATE with UPPER() function
#[test]
fn test_update_with_upper() {
    let db = Database::open("memory://update_upper").expect("Failed to create database");

    db.execute(
        "CREATE TABLE update_test (
            id INTEGER PRIMARY KEY,
            name TEXT,
            processed_name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO update_test (id, name) VALUES (1, 'hello')", ())
        .expect("Failed to insert");

    db.execute(
        "UPDATE update_test SET processed_name = UPPER(name) WHERE id = 1",
        (),
    )
    .expect("Failed to update with UPPER()");

    let result: String = db
        .query_one("SELECT processed_name FROM update_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO");
}

/// Test UPDATE with arithmetic expression
#[test]
fn test_update_with_arithmetic() {
    let db = Database::open("memory://update_arith").expect("Failed to create database");

    db.execute(
        "CREATE TABLE update_test (
            id INTEGER PRIMARY KEY,
            value INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO update_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert");

    db.execute(
        "UPDATE update_test SET value = value * 2 + 5 WHERE id = 1",
        (),
    )
    .expect("Failed to update with arithmetic");

    let result: i64 = db
        .query_one("SELECT value FROM update_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 25); // (10 * 2) + 5
}

/// Test UPDATE with CONCAT() function
#[test]
fn test_update_with_concat() {
    let db = Database::open("memory://update_concat").expect("Failed to create database");

    db.execute(
        "CREATE TABLE update_test (
            id INTEGER PRIMARY KEY,
            name TEXT,
            processed_name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO update_test (id, name) VALUES (1, 'world')", ())
        .expect("Failed to insert");

    db.execute(
        "UPDATE update_test SET processed_name = CONCAT('Processed: ', UPPER(name)) WHERE id = 1",
        (),
    )
    .expect("Failed to update with CONCAT()");

    let result: String = db
        .query_one("SELECT processed_name FROM update_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "Processed: WORLD");
}

/// Test INSERT with CAST string to integer
#[test]
fn test_insert_with_cast_to_int() {
    let db = Database::open("memory://insert_cast_int").expect("Failed to create database");

    db.execute(
        "CREATE TABLE cast_test (
            id INTEGER PRIMARY KEY,
            int_value INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO cast_test (id, int_value) VALUES (1, CAST('123' AS INTEGER))",
        (),
    )
    .expect("Failed to insert with CAST");

    let result: i64 = db
        .query_one("SELECT int_value FROM cast_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 123);
}

/// Test INSERT with CAST integer to string
#[test]
fn test_insert_with_cast_to_text() {
    let db = Database::open("memory://insert_cast_text").expect("Failed to create database");

    db.execute(
        "CREATE TABLE cast_test (
            id INTEGER PRIMARY KEY,
            str_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO cast_test (id, str_value) VALUES (1, CAST(456 AS TEXT))",
        (),
    )
    .expect("Failed to insert with CAST");

    let result: String = db
        .query_one("SELECT str_value FROM cast_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "456");
}

/// Test INSERT with CAST float to integer
#[test]
fn test_insert_with_cast_float_to_int() {
    let db = Database::open("memory://insert_cast_float").expect("Failed to create database");

    db.execute(
        "CREATE TABLE cast_test (
            id INTEGER PRIMARY KEY,
            int_value INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO cast_test (id, int_value) VALUES (1, CAST(78.9 AS INTEGER))",
        (),
    )
    .expect("Failed to insert with CAST");

    let result: i64 = db
        .query_one("SELECT int_value FROM cast_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 78);
}

/// Test INSERT with simple CASE expression
#[test]
fn test_insert_with_case() {
    let db = Database::open("memory://insert_case").expect("Failed to create database");

    db.execute(
        "CREATE TABLE case_test (
            id INTEGER PRIMARY KEY,
            category TEXT,
            grade TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO case_test (id, category, grade) VALUES (1, 'A',
            CASE 'A'
                WHEN 'A' THEN 'Excellent'
                WHEN 'B' THEN 'Good'
                ELSE 'Average'
            END)",
        (),
    )
    .expect("Failed to insert with CASE");

    let result: String = db
        .query_one("SELECT grade FROM case_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "Excellent");
}

/// Test INSERT with searched CASE expression
#[test]
fn test_insert_with_searched_case() {
    let db = Database::open("memory://insert_searched_case").expect("Failed to create database");

    db.execute(
        "CREATE TABLE case_test (
            id INTEGER PRIMARY KEY,
            score INTEGER,
            grade TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO case_test (id, score, grade) VALUES (1, 85,
            CASE
                WHEN 85 >= 90 THEN 'A'
                WHEN 85 >= 80 THEN 'B'
                WHEN 85 >= 70 THEN 'C'
                ELSE 'F'
            END)",
        (),
    )
    .expect("Failed to insert with searched CASE");

    let result: String = db
        .query_one("SELECT grade FROM case_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "B");
}

/// Test complex arithmetic expression
#[test]
fn test_complex_arithmetic() {
    let db = Database::open("memory://complex_arith").expect("Failed to create database");

    db.execute(
        "CREATE TABLE complex_test (
            id INTEGER PRIMARY KEY,
            number_result FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO complex_test (id, number_result) VALUES (1, (10.5 + 5) * 2 / 3)",
        (),
    )
    .expect("Failed to insert");

    let result: f64 = db
        .query_one("SELECT number_result FROM complex_test WHERE id = 1", ())
        .expect("Failed to query");
    // (15.5 * 2) / 3 = 10.333...
    assert!(
        (result - 10.333333).abs() < 0.001,
        "Expected ~10.333, got {}",
        result
    );
}

/// Test CASE with function calls
#[test]
fn test_case_with_functions() {
    let db = Database::open("memory://case_func").expect("Failed to create database");

    db.execute(
        "CREATE TABLE complex_test (
            id INTEGER PRIMARY KEY,
            result TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO complex_test (id, result) VALUES (1,
            CASE LENGTH('hello')
                WHEN 5 THEN UPPER('correct')
                ELSE LOWER('WRONG')
            END)",
        (),
    )
    .expect("Failed to insert with CASE and functions");

    let result: String = db
        .query_one("SELECT result FROM complex_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "CORRECT");
}
