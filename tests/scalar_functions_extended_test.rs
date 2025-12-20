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

//! Integration tests for extended scalar functions
//! Tests string, math, and datetime functions

use stoolap::Database;

// ============================================================================
// String Function Tests
// ============================================================================

#[test]
fn test_replace_function() {
    let db = Database::open("memory://test_replace").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT REPLACE('hello world', 'world', 'rust')", ())
        .expect("Failed to query");
    assert_eq!(result, "hello rust");

    // Multiple replacements
    let result: String = db
        .query_one("SELECT REPLACE('aaa', 'a', 'b')", ())
        .expect("Failed to query");
    assert_eq!(result, "bbb");

    // No match
    let result: String = db
        .query_one("SELECT REPLACE('hello', 'x', 'y')", ())
        .expect("Failed to query");
    assert_eq!(result, "hello");
}

#[test]
fn test_reverse_function() {
    let db = Database::open("memory://test_reverse").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT REVERSE('hello')", ())
        .expect("Failed to query");
    assert_eq!(result, "olleh");

    // Empty string
    let result: String = db
        .query_one("SELECT REVERSE('')", ())
        .expect("Failed to query");
    assert_eq!(result, "");

    // Single character
    let result: String = db
        .query_one("SELECT REVERSE('a')", ())
        .expect("Failed to query");
    assert_eq!(result, "a");
}

#[test]
fn test_left_function() {
    let db = Database::open("memory://test_left").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT LEFT('hello world', 5)", ())
        .expect("Failed to query");
    assert_eq!(result, "hello");

    // Request more than available
    let result: String = db
        .query_one("SELECT LEFT('hi', 10)", ())
        .expect("Failed to query");
    assert_eq!(result, "hi");

    // Zero length
    let result: String = db
        .query_one("SELECT LEFT('hello', 0)", ())
        .expect("Failed to query");
    assert_eq!(result, "");
}

#[test]
fn test_right_function() {
    let db = Database::open("memory://test_right").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT RIGHT('hello world', 5)", ())
        .expect("Failed to query");
    assert_eq!(result, "world");

    // Request more than available
    let result: String = db
        .query_one("SELECT RIGHT('hi', 10)", ())
        .expect("Failed to query");
    assert_eq!(result, "hi");
}

#[test]
fn test_repeat_function() {
    let db = Database::open("memory://test_repeat").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT REPEAT('ab', 3)", ())
        .expect("Failed to query");
    assert_eq!(result, "ababab");

    // Zero times
    let result: String = db
        .query_one("SELECT REPEAT('ab', 0)", ())
        .expect("Failed to query");
    assert_eq!(result, "");

    // Single repetition
    let result: String = db
        .query_one("SELECT REPEAT('x', 1)", ())
        .expect("Failed to query");
    assert_eq!(result, "x");
}

#[test]
fn test_split_part_function() {
    let db = Database::open("memory://test_split_part").expect("Failed to create database");

    // Standard split
    let result: String = db
        .query_one("SELECT SPLIT_PART('a,b,c', ',', 2)", ())
        .expect("Failed to query");
    assert_eq!(result, "b");

    // First part
    let result: String = db
        .query_one("SELECT SPLIT_PART('a-b-c', '-', 1)", ())
        .expect("Failed to query");
    assert_eq!(result, "a");

    // Last part
    let result: String = db
        .query_one("SELECT SPLIT_PART('a:b:c', ':', 3)", ())
        .expect("Failed to query");
    assert_eq!(result, "c");

    // Out of range returns empty
    let result: String = db
        .query_one("SELECT SPLIT_PART('a,b', ',', 5)", ())
        .expect("Failed to query");
    assert_eq!(result, "");
}

// NOTE: POSITION function supports normal syntax POSITION(substring, string)
// SQL standard syntax POSITION(substring IN string) would require parser update
#[test]
fn test_position_function() {
    let db = Database::open("memory://test_position").expect("Failed to create database");

    // Use normal function syntax instead of SQL standard syntax
    let result: i64 = db
        .query_one("SELECT POSITION('world', 'hello world')", ())
        .expect("Failed to query");
    assert_eq!(result, 7);

    // Not found
    let result: i64 = db
        .query_one("SELECT POSITION('xyz', 'hello')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);

    // At beginning
    let result: i64 = db
        .query_one("SELECT POSITION('hello', 'hello world')", ())
        .expect("Failed to query");
    assert_eq!(result, 1);
}

#[test]
fn test_instr_function() {
    let db = Database::open("memory://test_instr").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT INSTR('hello world', 'world')", ())
        .expect("Failed to query");
    assert_eq!(result, 7);

    // Not found
    let result: i64 = db
        .query_one("SELECT INSTR('hello', 'xyz')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);
}

#[test]
fn test_lpad_function() {
    let db = Database::open("memory://test_lpad").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT LPAD('hi', 5, '0')", ())
        .expect("Failed to query");
    assert_eq!(result, "000hi");

    // String already long enough
    let result: String = db
        .query_one("SELECT LPAD('hello', 3, '0')", ())
        .expect("Failed to query");
    assert_eq!(result, "hel");

    // Multi-char padding
    let result: String = db
        .query_one("SELECT LPAD('x', 5, 'ab')", ())
        .expect("Failed to query");
    assert_eq!(result, "ababx");
}

#[test]
fn test_rpad_function() {
    let db = Database::open("memory://test_rpad").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT RPAD('hi', 5, '0')", ())
        .expect("Failed to query");
    assert_eq!(result, "hi000");

    // String already long enough
    let result: String = db
        .query_one("SELECT RPAD('hello', 3, '0')", ())
        .expect("Failed to query");
    assert_eq!(result, "hel");
}

#[test]
fn test_char_length_function() {
    let db = Database::open("memory://test_char_length").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT CHAR_LENGTH('hello')", ())
        .expect("Failed to query");
    assert_eq!(result, 5);

    // Empty string
    let result: i64 = db
        .query_one("SELECT CHAR_LENGTH('')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);
}

// ============================================================================
// Math Function Tests
// ============================================================================

#[test]
fn test_power_function() {
    let db = Database::open("memory://test_power").expect("Failed to create database");

    let result: f64 = db
        .query_one("SELECT POWER(2, 3)", ())
        .expect("Failed to query");
    assert_eq!(result, 8.0);

    // Float result
    let result: f64 = db
        .query_one("SELECT POWER(2, 0.5)", ())
        .expect("Failed to query");
    assert!((result - 1.414).abs() < 0.01);

    // POW alias
    let result: f64 = db
        .query_one("SELECT POW(3, 2)", ())
        .expect("Failed to query");
    assert_eq!(result, 9.0);
}

#[test]
fn test_sqrt_function() {
    let db = Database::open("memory://test_sqrt").expect("Failed to create database");

    let result: f64 = db
        .query_one("SELECT SQRT(16)", ())
        .expect("Failed to query");
    assert_eq!(result, 4.0);

    let result: f64 = db.query_one("SELECT SQRT(2)", ()).expect("Failed to query");
    assert!((result - 1.414).abs() < 0.01);
}

#[test]
fn test_log_functions() {
    let db = Database::open("memory://test_log").expect("Failed to create database");

    // Natural log
    let result: f64 = db
        .query_one("SELECT LN(2.718281828)", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 0.01);

    // Log base 10
    let result: f64 = db
        .query_one("SELECT LOG10(100)", ())
        .expect("Failed to query");
    assert_eq!(result, 2.0);

    // Log base 2
    let result: f64 = db.query_one("SELECT LOG2(8)", ()).expect("Failed to query");
    assert_eq!(result, 3.0);

    // LOG with base
    let result: f64 = db
        .query_one("SELECT LOG(2, 8)", ())
        .expect("Failed to query");
    assert_eq!(result, 3.0);
}

#[test]
fn test_exp_function() {
    let db = Database::open("memory://test_exp").expect("Failed to create database");

    let result: f64 = db.query_one("SELECT EXP(0)", ()).expect("Failed to query");
    assert_eq!(result, 1.0);

    let result: f64 = db.query_one("SELECT EXP(1)", ()).expect("Failed to query");
    assert!((result - std::f64::consts::E).abs() < 0.0001);
}

#[test]
fn test_sign_function() {
    let db = Database::open("memory://test_sign").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT SIGN(42)", ())
        .expect("Failed to query");
    assert_eq!(result, 1);

    let result: i64 = db
        .query_one("SELECT SIGN(-42)", ())
        .expect("Failed to query");
    assert_eq!(result, -1);

    let result: i64 = db.query_one("SELECT SIGN(0)", ()).expect("Failed to query");
    assert_eq!(result, 0);
}

#[test]
fn test_truncate_function() {
    let db = Database::open("memory://test_truncate").expect("Failed to create database");

    let result: f64 = db
        .query_one("SELECT TRUNCATE(3.789, 2)", ())
        .expect("Failed to query");
    assert_eq!(result, 3.78);

    let result: f64 = db
        .query_one("SELECT TRUNCATE(3.789, 0)", ())
        .expect("Failed to query");
    assert_eq!(result, 3.0);

    // TRUNC alias
    let result: f64 = db
        .query_one("SELECT TRUNC(3.789, 1)", ())
        .expect("Failed to query");
    assert_eq!(result, 3.7);
}

#[test]
fn test_pi_function() {
    let db = Database::open("memory://test_pi").expect("Failed to create database");

    let result: f64 = db.query_one("SELECT PI()", ()).expect("Failed to query");
    assert!((result - std::f64::consts::PI).abs() < 0.0001);
}

#[test]
fn test_random_function() {
    let db = Database::open("memory://test_random").expect("Failed to create database");

    // Just test that it returns something between 0 and 1
    let result: f64 = db
        .query_one("SELECT RANDOM()", ())
        .expect("Failed to query");
    assert!((0.0..1.0).contains(&result));
}

// ============================================================================
// DateTime Function Tests
// ============================================================================

#[test]
fn test_extract_function() {
    let db = Database::open("memory://test_extract").expect("Failed to create database");

    // SQL-standard EXTRACT syntax: EXTRACT(field FROM source)
    let result: i64 = db
        .query_one("SELECT EXTRACT(YEAR FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 2024);

    let result: i64 = db
        .query_one("SELECT EXTRACT(MONTH FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 3);

    let result: i64 = db
        .query_one("SELECT EXTRACT(DAY FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 15);

    let result: i64 = db
        .query_one("SELECT EXTRACT(HOUR FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 10);

    let result: i64 = db
        .query_one("SELECT EXTRACT(MINUTE FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 30);

    let result: i64 = db
        .query_one("SELECT EXTRACT(SECOND FROM '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 45);
}

#[test]
fn test_year_month_day_functions() {
    let db = Database::open("memory://test_year_month_day").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT YEAR('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 2024);

    let result: i64 = db
        .query_one("SELECT MONTH('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 3);

    let result: i64 = db
        .query_one("SELECT DAY('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 15);
}

#[test]
fn test_hour_minute_second_functions() {
    let db = Database::open("memory://test_hour_minute_second").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT HOUR('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 10);

    let result: i64 = db
        .query_one("SELECT MINUTE('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 30);

    let result: i64 = db
        .query_one("SELECT SECOND('2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert_eq!(result, 45);
}

#[test]
fn test_date_add_function() {
    let db = Database::open("memory://test_date_add").expect("Failed to create database");

    // Add days - just verify it doesn't error
    let _result: String = db
        .query_one("SELECT DATE_ADD('2024-03-15 10:30:45', 5, 'day')", ())
        .expect("Failed to query DATE_ADD with days");

    // Add months
    let _result: String = db
        .query_one("SELECT DATE_ADD('2024-03-15 10:30:45', 2, 'month')", ())
        .expect("Failed to query DATE_ADD with months");

    // Add years
    let _result: String = db
        .query_one("SELECT DATE_ADD('2024-03-15 10:30:45', 1, 'year')", ())
        .expect("Failed to query DATE_ADD with years");

    // Add hours
    let _result: String = db
        .query_one("SELECT DATE_ADD('2024-03-15 10:30:45', 3, 'hour')", ())
        .expect("Failed to query DATE_ADD with hours");
}

#[test]
fn test_date_sub_function() {
    let db = Database::open("memory://test_date_sub").expect("Failed to create database");

    // Subtract days - just verify it doesn't error
    let _result: String = db
        .query_one("SELECT DATE_SUB('2024-03-15 10:30:45', 5, 'day')", ())
        .expect("Failed to query DATE_SUB with days");

    // Subtract months
    let _result: String = db
        .query_one("SELECT DATE_SUB('2024-03-15 10:30:45', 2, 'month')", ())
        .expect("Failed to query DATE_SUB with months");
}

#[test]
fn test_datediff_function() {
    let db = Database::open("memory://test_datediff").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT DATEDIFF('2024-03-20', '2024-03-15')", ())
        .expect("Failed to query");
    assert_eq!(result, 5);

    // Negative difference
    let result: i64 = db
        .query_one("SELECT DATEDIFF('2024-03-15', '2024-03-20')", ())
        .expect("Failed to query");
    assert_eq!(result, -5);

    // Same day
    let result: i64 = db
        .query_one("SELECT DATEDIFF('2024-03-15', '2024-03-15')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);
}

#[test]
fn test_current_date_function() {
    let db = Database::open("memory://test_current_date").expect("Failed to create database");

    // Just verify it returns a valid timestamp
    let _result: String = db
        .query_one("SELECT CURRENT_DATE()", ())
        .expect("Failed to query CURRENT_DATE");
}

#[test]
fn test_current_timestamp_function() {
    let db = Database::open("memory://test_current_timestamp").expect("Failed to create database");

    // Just verify it returns a valid timestamp
    let _result: String = db
        .query_one("SELECT CURRENT_TIMESTAMP()", ())
        .expect("Failed to query CURRENT_TIMESTAMP");
}

#[test]
fn test_date_trunc_function() {
    let db = Database::open("memory://test_date_trunc").expect("Failed to create database");

    let result: String = db
        .query_one("SELECT DATE_TRUNC('year', '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert!(result.contains("2024-01-01"));

    let result: String = db
        .query_one("SELECT DATE_TRUNC('month', '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert!(result.contains("2024-03-01"));

    let result: String = db
        .query_one("SELECT DATE_TRUNC('day', '2024-03-15 10:30:45')", ())
        .expect("Failed to query");
    assert!(result.contains("2024-03-15"));
    assert!(result.contains("00:00:00"));
}

// ============================================================================
// Combined Function Tests with Table Data
// ============================================================================

#[test]
fn test_functions_with_table_data() {
    let db = Database::open("memory://test_functions_table").expect("Failed to create database");

    // Create a test table
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, created TIMESTAMP)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, name, price, created) VALUES (1, 'Apple', 1.50, '2024-01-15 09:00:00')",
        (),
    )
    .expect("Failed to insert");

    db.execute(
        "INSERT INTO products (id, name, price, created) VALUES (2, 'Banana', 0.75, '2024-02-20 10:30:00')",
        (),
    )
    .expect("Failed to insert");

    db.execute(
        "INSERT INTO products (id, name, price, created) VALUES (3, 'Cherry', 2.25, '2024-03-25 14:45:00')",
        (),
    )
    .expect("Failed to insert");

    // String operations on name column
    let result: String = db
        .query_one("SELECT UPPER(name) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "APPLE");

    let result: i64 = db
        .query_one("SELECT LENGTH(name) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 5);

    // Math operations on price column
    let result: f64 = db
        .query_one("SELECT ROUND(price, 1) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 1.5);

    let result: f64 = db
        .query_one("SELECT CEIL(price) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 2.0);

    // Date operations on created column
    let result: i64 = db
        .query_one("SELECT YEAR(created) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 2024);

    let result: i64 = db
        .query_one("SELECT MONTH(created) FROM products WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 1);

    let result: i64 = db
        .query_one("SELECT MONTH(created) FROM products WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(result, 2);

    let result: i64 = db
        .query_one("SELECT MONTH(created) FROM products WHERE id = 3", ())
        .expect("Failed to query");
    assert_eq!(result, 3);
}

#[test]
fn test_functions_in_where_clause() {
    let db = Database::open("memory://test_functions_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 25)",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 30)",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO users (id, name, age) VALUES (3, 'Charlie', 35)",
        (),
    )
    .expect("Failed to insert");

    // Use function in WHERE
    let result = db
        .query("SELECT name FROM users WHERE LENGTH(name) > 4", ())
        .expect("Failed to query");
    let count: i64 = result.into_iter().count() as i64;
    assert_eq!(count, 2); // Alice and Charlie

    // Use UPPER in WHERE
    let result: String = db
        .query_one("SELECT name FROM users WHERE UPPER(name) = 'BOB'", ())
        .expect("Failed to query");
    assert_eq!(result, "Bob");
}

#[test]
fn test_nested_string_functions() {
    let db = Database::open("memory://test_nested_string").expect("Failed to create database");

    // UPPER(REPLACE(...))
    let result: String = db
        .query_one("SELECT UPPER(REPLACE('hello world', 'world', 'rust'))", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO RUST");

    // REVERSE(SUBSTRING(...)) - using SUBSTRING instead of LEFT (which is a reserved keyword)
    let result: String = db
        .query_one("SELECT REVERSE(SUBSTRING('hello', 1, 3))", ())
        .expect("Failed to query");
    assert_eq!(result, "leh");
}

#[test]
fn test_nested_math_functions() {
    let db = Database::open("memory://test_nested_math").expect("Failed to create database");

    // SQRT(POWER(...))
    let result: f64 = db
        .query_one("SELECT SQRT(POWER(3, 2))", ())
        .expect("Failed to query");
    assert_eq!(result, 3.0);

    // ABS(SIGN(...))
    let result: i64 = db
        .query_one("SELECT ABS(SIGN(-42))", ())
        .expect("Failed to query");
    assert_eq!(result, 1);
}

// ============================================================================
// Utility Function Tests (GREATEST, LEAST, IIF)
// ============================================================================

#[test]
fn test_greatest_function() {
    let db = Database::open("memory://test_greatest").expect("Failed to create database");

    // With integers
    let result: i64 = db
        .query_one("SELECT GREATEST(1, 5, 3)", ())
        .expect("Failed to query");
    assert_eq!(result, 5);

    // With floats
    let result: f64 = db
        .query_one("SELECT GREATEST(1.5, 2.5, 0.5)", ())
        .expect("Failed to query");
    assert_eq!(result, 2.5);

    // With strings
    let result: String = db
        .query_one("SELECT GREATEST('apple', 'banana', 'cherry')", ())
        .expect("Failed to query");
    assert_eq!(result, "cherry");

    // With two arguments
    let result: i64 = db
        .query_one("SELECT GREATEST(10, 20)", ())
        .expect("Failed to query");
    assert_eq!(result, 20);

    // With negative numbers
    let result: i64 = db
        .query_one("SELECT GREATEST(-5, -10, -1)", ())
        .expect("Failed to query");
    assert_eq!(result, -1);
}

#[test]
fn test_greatest_with_null() {
    let db = Database::open("memory://test_greatest_null").expect("Failed to create database");

    // NULL propagation - if any arg is NULL, result is NULL
    let result = db
        .query("SELECT GREATEST(1, NULL, 3)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 1);
    // The result should be NULL
}

#[test]
fn test_least_function() {
    let db = Database::open("memory://test_least").expect("Failed to create database");

    // With integers
    let result: i64 = db
        .query_one("SELECT LEAST(1, 5, 3)", ())
        .expect("Failed to query");
    assert_eq!(result, 1);

    // With floats
    let result: f64 = db
        .query_one("SELECT LEAST(1.5, 2.5, 0.5)", ())
        .expect("Failed to query");
    assert_eq!(result, 0.5);

    // With strings
    let result: String = db
        .query_one("SELECT LEAST('apple', 'banana', 'cherry')", ())
        .expect("Failed to query");
    assert_eq!(result, "apple");

    // With two arguments
    let result: i64 = db
        .query_one("SELECT LEAST(10, 20)", ())
        .expect("Failed to query");
    assert_eq!(result, 10);

    // With negative numbers
    let result: i64 = db
        .query_one("SELECT LEAST(-5, -10, -1)", ())
        .expect("Failed to query");
    assert_eq!(result, -10);
}

#[test]
fn test_least_with_null() {
    let db = Database::open("memory://test_least_null").expect("Failed to create database");

    // NULL propagation - if any arg is NULL, result is NULL
    let result = db
        .query("SELECT LEAST(1, NULL, 3)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 1);
    // The result should be NULL
}

#[test]
fn test_iif_function() {
    let db = Database::open("memory://test_iif").expect("Failed to create database");

    // True condition
    let result: String = db
        .query_one("SELECT IIF(1 > 0, 'yes', 'no')", ())
        .expect("Failed to query");
    assert_eq!(result, "yes");

    // False condition
    let result: String = db
        .query_one("SELECT IIF(1 < 0, 'yes', 'no')", ())
        .expect("Failed to query");
    assert_eq!(result, "no");

    // With integer result
    let result: i64 = db
        .query_one("SELECT IIF(5 > 3, 100, 200)", ())
        .expect("Failed to query");
    assert_eq!(result, 100);

    // With boolean condition - TRUE
    let result: String = db
        .query_one("SELECT IIF(TRUE, 'yes', 'no')", ())
        .expect("Failed to query");
    assert_eq!(result, "yes");

    // With boolean condition - FALSE
    let result: String = db
        .query_one("SELECT IIF(FALSE, 'yes', 'no')", ())
        .expect("Failed to query");
    assert_eq!(result, "no");

    // With equality check
    let result: String = db
        .query_one("SELECT IIF(2 + 2 = 4, 'correct', 'wrong')", ())
        .expect("Failed to query");
    assert_eq!(result, "correct");
}

#[test]
fn test_iif_with_table_data() {
    let db = Database::open("memory://test_iif_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO scores VALUES (1, 'Alice', 85)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO scores VALUES (2, 'Bob', 72)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO scores VALUES (3, 'Charlie', 90)", ())
        .expect("Failed to insert");

    // Use IIF to categorize scores
    let result: String = db
        .query_one(
            "SELECT IIF(score >= 80, 'Pass', 'Fail') FROM scores WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Pass");

    let result: String = db
        .query_one(
            "SELECT IIF(score >= 80, 'Pass', 'Fail') FROM scores WHERE id = 2",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Fail");
}

// ============================================================================
// Trigonometric Function Tests (SIN, COS, TAN)
// ============================================================================

#[test]
fn test_sin_function() {
    let db = Database::open("memory://test_sin").expect("Failed to create database");

    // sin(0) = 0
    let result: f64 = db.query_one("SELECT SIN(0)", ()).expect("Failed to query");
    assert!(result.abs() < 1e-10);

    // sin(π/2) = 1
    let result: f64 = db
        .query_one("SELECT SIN(PI() / 2)", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10);

    // sin(π) ≈ 0
    let result: f64 = db
        .query_one("SELECT SIN(PI())", ())
        .expect("Failed to query");
    assert!(result.abs() < 1e-10);

    // sin(π/6) = 0.5
    let result: f64 = db
        .query_one("SELECT SIN(PI() / 6)", ())
        .expect("Failed to query");
    assert!((result - 0.5).abs() < 1e-10);
}

#[test]
fn test_cos_function() {
    let db = Database::open("memory://test_cos").expect("Failed to create database");

    // cos(0) = 1
    let result: f64 = db.query_one("SELECT COS(0)", ()).expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10);

    // cos(π/2) ≈ 0
    let result: f64 = db
        .query_one("SELECT COS(PI() / 2)", ())
        .expect("Failed to query");
    assert!(result.abs() < 1e-10);

    // cos(π) = -1
    let result: f64 = db
        .query_one("SELECT COS(PI())", ())
        .expect("Failed to query");
    assert!((result - (-1.0)).abs() < 1e-10);

    // cos(π/3) = 0.5
    let result: f64 = db
        .query_one("SELECT COS(PI() / 3)", ())
        .expect("Failed to query");
    assert!((result - 0.5).abs() < 1e-10);
}

#[test]
fn test_tan_function() {
    let db = Database::open("memory://test_tan").expect("Failed to create database");

    // tan(0) = 0
    let result: f64 = db.query_one("SELECT TAN(0)", ()).expect("Failed to query");
    assert!(result.abs() < 1e-10);

    // tan(π/4) = 1
    let result: f64 = db
        .query_one("SELECT TAN(PI() / 4)", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10);

    // tan(π) ≈ 0
    let result: f64 = db
        .query_one("SELECT TAN(PI())", ())
        .expect("Failed to query");
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_trig_with_null() {
    let db = Database::open("memory://test_trig_null").expect("Failed to create database");

    // SIN(NULL) should return NULL
    let result = db.query("SELECT SIN(NULL)", ()).expect("Failed to query");
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 1);

    // COS(NULL) should return NULL
    let result = db.query("SELECT COS(NULL)", ()).expect("Failed to query");
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 1);

    // TAN(NULL) should return NULL
    let result = db.query("SELECT TAN(NULL)", ()).expect("Failed to query");
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_trig_with_integer_input() {
    let db = Database::open("memory://test_trig_int").expect("Failed to create database");

    // Trig functions should accept integer inputs
    let result: f64 = db.query_one("SELECT SIN(0)", ()).expect("Failed to query");
    assert!(result.abs() < 1e-10);

    let result: f64 = db.query_one("SELECT COS(0)", ()).expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10);

    let result: f64 = db.query_one("SELECT TAN(0)", ()).expect("Failed to query");
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_trig_combined_with_pi() {
    let db = Database::open("memory://test_trig_pi").expect("Failed to create database");

    // Verify sin²(x) + cos²(x) = 1
    let result: f64 = db
        .query_one("SELECT POWER(SIN(1.0), 2) + POWER(COS(1.0), 2)", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10);

    // tan(x) = sin(x) / cos(x)
    let result: f64 = db
        .query_one("SELECT SIN(0.5) / COS(0.5) - TAN(0.5)", ())
        .expect("Failed to query");
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_trig_with_table_data() {
    let db = Database::open("memory://test_trig_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE angles (id INTEGER PRIMARY KEY, radians FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO angles VALUES (1, 0)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO angles VALUES (2, 1.5707963267948966)", ()) // π/2
        .expect("Failed to insert");
    db.execute("INSERT INTO angles VALUES (3, 3.141592653589793)", ()) // π
        .expect("Failed to insert");

    // Query sin values from table
    let result: f64 = db
        .query_one("SELECT SIN(radians) FROM angles WHERE id = 1", ())
        .expect("Failed to query");
    assert!(result.abs() < 1e-10); // sin(0) = 0

    let result: f64 = db
        .query_one("SELECT SIN(radians) FROM angles WHERE id = 2", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10); // sin(π/2) = 1

    // Query cos values from table
    let result: f64 = db
        .query_one("SELECT COS(radians) FROM angles WHERE id = 1", ())
        .expect("Failed to query");
    assert!((result - 1.0).abs() < 1e-10); // cos(0) = 1

    let result: f64 = db
        .query_one("SELECT COS(radians) FROM angles WHERE id = 3", ())
        .expect("Failed to query");
    assert!((result - (-1.0)).abs() < 1e-10); // cos(π) = -1
}
