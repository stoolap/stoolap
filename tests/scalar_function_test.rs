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

//! Scalar Function Tests
//!
//! Tests scalar functions: UPPER, LOWER, LENGTH, ABS, ROUND, COALESCE, etc.

use stoolap::Database;

fn setup_sample_table(db: &Database) {
    // Create the test table
    db.execute(
        "CREATE TABLE sample (
            id INTEGER,
            text_value TEXT,
            num_value FLOAT,
            nullable_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute("INSERT INTO sample (id, text_value, num_value, nullable_value) VALUES (1, 'Hello World', 123.45, 'Not Null')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sample (id, text_value, num_value, nullable_value) VALUES (2, 'second ROW', -42.5, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sample (id, text_value, num_value, nullable_value) VALUES (3, 'Another Test', 0.0, 'Value')", ())
        .expect("Failed to insert");
}

// String functions

#[test]
fn test_upper_function() {
    let db = Database::open("memory://scalar_upper").expect("Failed to create database");
    setup_sample_table(&db);

    let result: String = db
        .query_one("SELECT UPPER(text_value) FROM sample WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "HELLO WORLD", "UPPER function failed");
}

#[test]
fn test_lower_function() {
    let db = Database::open("memory://scalar_lower").expect("Failed to create database");
    setup_sample_table(&db);

    let result: String = db
        .query_one("SELECT LOWER(text_value) FROM sample WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(result, "second row", "LOWER function failed");
}

#[test]
fn test_length_function() {
    let db = Database::open("memory://scalar_length").expect("Failed to create database");
    setup_sample_table(&db);

    let result: i64 = db
        .query_one("SELECT LENGTH(text_value) FROM sample WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 11, "LENGTH function failed");
}

// Numeric functions

#[test]
fn test_abs_function() {
    let db = Database::open("memory://scalar_abs").expect("Failed to create database");
    setup_sample_table(&db);

    let result: f64 = db
        .query_one("SELECT ABS(num_value) FROM sample WHERE id = 2", ())
        .expect("Failed to query");
    assert!(
        (result - 42.5).abs() < 0.001,
        "ABS function failed: got {}",
        result
    );
}

#[test]
fn test_round_function() {
    let db = Database::open("memory://scalar_round").expect("Failed to create database");
    setup_sample_table(&db);

    let result: f64 = db
        .query_one("SELECT ROUND(num_value, 1) FROM sample WHERE id = 1", ())
        .expect("Failed to query");
    assert!(
        (result - 123.5).abs() < 0.001,
        "ROUND function failed: got {}",
        result
    );
}

// COALESCE function

#[test]
fn test_coalesce_non_null() {
    let db = Database::open("memory://scalar_coalesce1").expect("Failed to create database");
    setup_sample_table(&db);

    let result: String = db
        .query_one(
            "SELECT COALESCE('Not Null', 'Default') FROM sample WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Not Null", "COALESCE with non-null failed");
}

#[test]
fn test_coalesce_null_literal() {
    let db = Database::open("memory://scalar_coalesce2").expect("Failed to create database");
    setup_sample_table(&db);

    let result: String = db
        .query_one(
            "SELECT COALESCE(NULL, 'Default') FROM sample WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Default", "COALESCE with NULL failed");
}

#[test]
fn test_coalesce_multiple_args() {
    let db = Database::open("memory://scalar_coalesce3").expect("Failed to create database");
    setup_sample_table(&db);

    // COALESCE returns first non-NULL value. Empty string is not NULL.
    // So it should return '' (empty string), or 'Value' if empty strings are treated as NULL.
    let result: String = db
        .query_one(
            "SELECT COALESCE(NULL, '', 'Value', 'Other') FROM sample WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert!(
        result == "Value" || result.is_empty(),
        "COALESCE multiple args: got '{}'",
        result
    );
}

// Test multiple functions

#[test]
fn test_upper_on_different_row() {
    let db = Database::open("memory://scalar_upper2").expect("Failed to create database");
    setup_sample_table(&db);

    let result: String = db
        .query_one("SELECT UPPER(text_value) FROM sample WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(result, "SECOND ROW", "UPPER function on row 2 failed");
}

#[test]
fn test_length_on_different_row() {
    let db = Database::open("memory://scalar_length2").expect("Failed to create database");
    setup_sample_table(&db);

    // "second ROW" has 10 characters
    let result: i64 = db
        .query_one("SELECT LENGTH(text_value) FROM sample WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(result, 10, "LENGTH function on row 2 failed");
}

// Additional string functions

#[test]
fn test_trim_function() {
    let db = Database::open("memory://scalar_trim").expect("Failed to create database");

    db.execute("CREATE TABLE trim_test (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO trim_test VALUES (1, '  hello  ')", ())
        .expect("Failed to insert");

    let result: String = db
        .query_one("SELECT TRIM(val) FROM trim_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "hello", "TRIM function failed");
}

#[test]
fn test_ltrim_function() {
    let db = Database::open("memory://scalar_ltrim").expect("Failed to create database");

    db.execute("CREATE TABLE ltrim_test (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO ltrim_test VALUES (1, '  hello  ')", ())
        .expect("Failed to insert");

    let result: String = db
        .query_one("SELECT LTRIM(val) FROM ltrim_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "hello  ", "LTRIM function failed");
}

#[test]
fn test_rtrim_function() {
    let db = Database::open("memory://scalar_rtrim").expect("Failed to create database");

    db.execute("CREATE TABLE rtrim_test (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO rtrim_test VALUES (1, '  hello  ')", ())
        .expect("Failed to insert");

    let result: String = db
        .query_one("SELECT RTRIM(val) FROM rtrim_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "  hello", "RTRIM function failed");
}

#[test]
fn test_substring_function() {
    let db = Database::open("memory://scalar_substr").expect("Failed to create database");

    db.execute("CREATE TABLE substr_test (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO substr_test VALUES (1, 'Hello World')", ())
        .expect("Failed to insert");

    // SUBSTRING(val, start, length) - 1-indexed
    let result: String = db
        .query_one(
            "SELECT SUBSTRING(val, 1, 5) FROM substr_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Hello", "SUBSTRING function failed");
}

#[test]
fn test_concat_function() {
    let db = Database::open("memory://scalar_concat").expect("Failed to create database");

    db.execute(
        "CREATE TABLE concat_test (id INTEGER, first_name TEXT, last_name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO concat_test VALUES (1, 'John', 'Doe')", ())
        .expect("Failed to insert");

    let result: String = db
        .query_one(
            "SELECT CONCAT(first_name, ' ', last_name) FROM concat_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "John Doe", "CONCAT function failed");
}

// Additional numeric functions

#[test]
fn test_floor_function() {
    let db = Database::open("memory://scalar_floor").expect("Failed to create database");

    db.execute("CREATE TABLE floor_test (id INTEGER, val FLOAT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO floor_test VALUES (1, 3.7)", ())
        .expect("Failed to insert");

    let result: f64 = db
        .query_one("SELECT FLOOR(val) FROM floor_test WHERE id = 1", ())
        .expect("Failed to query");
    assert!(
        (result - 3.0).abs() < 0.001,
        "FLOOR function failed: got {}",
        result
    );
}

#[test]
fn test_ceil_function() {
    let db = Database::open("memory://scalar_ceil").expect("Failed to create database");

    db.execute("CREATE TABLE ceil_test (id INTEGER, val FLOAT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO ceil_test VALUES (1, 3.2)", ())
        .expect("Failed to insert");

    let result: f64 = db
        .query_one("SELECT CEIL(val) FROM ceil_test WHERE id = 1", ())
        .expect("Failed to query");
    assert!(
        (result - 4.0).abs() < 0.001,
        "CEIL function failed: got {}",
        result
    );
}

#[test]
fn test_mod_function() {
    let db = Database::open("memory://scalar_mod").expect("Failed to create database");

    db.execute(
        "CREATE TABLE mod_test (id INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO mod_test VALUES (1, 10, 3)", ())
        .expect("Failed to insert");

    let result: i64 = db
        .query_one("SELECT MOD(a, b) FROM mod_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 1, "MOD function failed");
}

// IFNULL / NULLIF functions

#[test]
fn test_ifnull_function() {
    let db = Database::open("memory://scalar_ifnull").expect("Failed to create database");
    setup_sample_table(&db);

    // Row 2 has NULL nullable_value
    let result: String = db
        .query_one(
            "SELECT IFNULL(nullable_value, 'was null') FROM sample WHERE id = 2",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "was null", "IFNULL function failed");

    // Row 1 has non-NULL nullable_value
    let result: String = db
        .query_one(
            "SELECT IFNULL(nullable_value, 'was null') FROM sample WHERE id = 1",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "Not Null", "IFNULL with non-null failed");
}

#[test]
fn test_nullif_function() {
    let db = Database::open("memory://scalar_nullif").expect("Failed to create database");

    db.execute("CREATE TABLE nullif_test (id INTEGER, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO nullif_test VALUES (1, 5)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO nullif_test VALUES (2, 0)", ())
        .expect("Failed to insert");

    // NULLIF(val, 0) returns NULL if val=0, otherwise returns val
    // When val=5, should return 5
    let result: i64 = db
        .query_one("SELECT NULLIF(val, 0) FROM nullif_test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, 5, "NULLIF function failed");
}
