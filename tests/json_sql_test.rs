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

//! JSON SQL Tests
//!
//! Tests JSON data type support in SQL statements

use stoolap::Database;

/// Test creating tables with JSON columns
#[test]
fn test_json_create_table() {
    let db = Database::open("memory://json_create").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON,
            required_json JSON NOT NULL
        )",
        (),
    )
    .expect("Failed to create table with JSON columns");
}

/// Test inserting JSON data
#[test]
fn test_json_insert() {
    let db = Database::open("memory://json_insert").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON,
            required_json JSON NOT NULL
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert simple object
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (?, ?, ?)",
        (1, r#"{"name":"John","age":30}"#, r#"{"type":"required"}"#),
    )
    .expect("Failed to insert simple object");

    // Insert array
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (?, ?, ?)",
        (2, "[1,2,3,4]", r#"{"type":"array"}"#),
    )
    .expect("Failed to insert array");

    // Insert nested object
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (?, ?, ?)",
        (
            3,
            r#"{"user":{"name":"Jane","age":25,"roles":["admin","user"]}}"#,
            r#"{"type":"nested"}"#,
        ),
    )
    .expect("Failed to insert nested object");

    // Insert NULL value
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (4, NULL, '{\"type\":\"null_test\"}')", ())
    .expect("Failed to insert NULL value");

    // Insert empty object
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (?, ?, ?)",
        (5, "{}", r#"{"type":"empty"}"#),
    )
    .expect("Failed to insert empty object");

    // Insert empty array
    db.execute(
        "INSERT INTO json_test (id, data, required_json) VALUES (?, ?, ?)",
        (6, "[]", r#"{"type":"empty_array"}"#),
    )
    .expect("Failed to insert empty array");

    // Verify count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test", ())
        .expect("Failed to count");
    assert_eq!(count, 6, "Expected 6 rows inserted");
}

/// Test selecting JSON data
#[test]
fn test_json_select() {
    let db = Database::open("memory://json_select").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (1, '{"name":"John","age":30}')"#,
        (),
    )
    .expect("Failed to insert");

    // Query by ID
    let result = db
        .query("SELECT id, data FROM json_test WHERE id = 1", ())
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let data: String = row.get(1).unwrap();

        assert_eq!(id, 1);
        assert!(data.contains("John"), "Expected data to contain 'John'");
        found = true;
    }
    assert!(found, "Expected to find row with id=1");
}

/// Test updating JSON data
#[test]
fn test_json_update() {
    let db = Database::open("memory://json_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (1, '{"name":"John","age":30}')"#,
        (),
    )
    .expect("Failed to insert");

    // Update the JSON
    db.execute(
        "UPDATE json_test SET data = ? WHERE id = ?",
        (r#"{"updated":true,"name":"UpdatedValue"}"#, 1),
    )
    .expect("Failed to update");

    // Verify update
    let result = db
        .query("SELECT data FROM json_test WHERE id = 1", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let data: String = row.get(0).unwrap();
        assert!(
            data.contains("updated") || data.contains("Updated"),
            "Expected updated data, got: {}",
            data
        );
    }
}

/// Test JSON literal in INSERT
#[test]
fn test_json_literal_insert() {
    let db = Database::open("memory://json_literal").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert with JSON literal
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (100, '{"literal":true}')"#,
        (),
    )
    .expect("Failed to insert with literal");

    // Verify
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test WHERE id = 100", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row with id=100");
}

/// Test parameterized nested JSON
#[test]
fn test_json_nested_param() {
    let db = Database::open("memory://json_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO json_test (id, data) VALUES (?, ?)",
        (101, r#"{"deep":{"nested":{"value":42}}}"#),
    )
    .expect("Failed to insert nested");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test WHERE id = 101", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row with id=101");
}

/// Test JSON with various data types in values
#[test]
fn test_json_various_types() {
    let db = Database::open("memory://json_types").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Boolean values
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (1, '{"active":true,"deleted":false}')"#,
        (),
    )
    .expect("Failed to insert boolean");

    // Numeric values
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (2, '{"integer":42,"float":3.14,"negative":-100}')"#, ())
    .expect("Failed to insert numeric");

    // Null value in JSON
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (3, '{"value":null}')"#,
        (),
    )
    .expect("Failed to insert null value");

    // String with special characters (escaped quotes in JSON)
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (4, '{"text":"Hello \"World\""}')"#,
        (),
    )
    .expect("Failed to insert special chars");

    // Verify all inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 rows");
}

/// Test JSON array operations
#[test]
fn test_json_arrays() {
    let db = Database::open("memory://json_arrays").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Empty array
    db.execute("INSERT INTO json_test (id, data) VALUES (1, '[]')", ())
        .expect("Failed to insert empty array");

    // Number array
    db.execute(
        "INSERT INTO json_test (id, data) VALUES (2, '[1,2,3,4,5]')",
        (),
    )
    .expect("Failed to insert number array");

    // String array
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (3, '["a","b","c"]')"#,
        (),
    )
    .expect("Failed to insert string array");

    // Mixed array
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (4, '[1,"two",true,null]')"#,
        (),
    )
    .expect("Failed to insert mixed array");

    // Nested arrays
    db.execute(
        "INSERT INTO json_test (id, data) VALUES (5, '[[1,2],[3,4]]')",
        (),
    )
    .expect("Failed to insert nested arrays");

    // Array of objects
    db.execute(
        r#"INSERT INTO json_test (id, data) VALUES (6, '[{"name":"John"},{"name":"Jane"}]')"#,
        (),
    )
    .expect("Failed to insert array of objects");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test", ())
        .expect("Failed to count");
    assert_eq!(count, 6, "Expected 6 rows");
}

/// Test JSON_TYPE function
#[test]
fn test_json_type_function() {
    let db = Database::open("memory://json_type").expect("Failed to create database");

    // Test with string literal
    let result: String = db
        .query_one("SELECT JSON_TYPE('\"hello\"')", ())
        .expect("Failed to get JSON_TYPE for string");
    assert_eq!(result, "string");

    // Test with number
    let result: String = db
        .query_one("SELECT JSON_TYPE('123')", ())
        .expect("Failed to get JSON_TYPE for number");
    assert_eq!(result, "number");

    // Test with float
    let result: String = db
        .query_one("SELECT JSON_TYPE('3.14')", ())
        .expect("Failed to get JSON_TYPE for float");
    assert_eq!(result, "number");

    // Test with boolean true
    let result: String = db
        .query_one("SELECT JSON_TYPE('true')", ())
        .expect("Failed to get JSON_TYPE for boolean");
    assert_eq!(result, "boolean");

    // Test with boolean false
    let result: String = db
        .query_one("SELECT JSON_TYPE('false')", ())
        .expect("Failed to get JSON_TYPE for boolean false");
    assert_eq!(result, "boolean");

    // Test with null
    let result: String = db
        .query_one("SELECT JSON_TYPE('null')", ())
        .expect("Failed to get JSON_TYPE for null");
    assert_eq!(result, "null");

    // Test with object
    let result: String = db
        .query_one("SELECT JSON_TYPE('{\"a\":1}')", ())
        .expect("Failed to get JSON_TYPE for object");
    assert_eq!(result, "object");

    // Test with array
    let result: String = db
        .query_one("SELECT JSON_TYPE('[1,2,3]')", ())
        .expect("Failed to get JSON_TYPE for array");
    assert_eq!(result, "array");

    // Test with empty object
    let result: String = db
        .query_one("SELECT JSON_TYPE('{}')", ())
        .expect("Failed to get JSON_TYPE for empty object");
    assert_eq!(result, "object");

    // Test with empty array
    let result: String = db
        .query_one("SELECT JSON_TYPE('[]')", ())
        .expect("Failed to get JSON_TYPE for empty array");
    assert_eq!(result, "array");
}

/// Test JSON_TYPEOF function (alias for JSON_TYPE)
#[test]
fn test_json_typeof_function() {
    let db = Database::open("memory://json_typeof").expect("Failed to create database");

    // Test that JSON_TYPEOF returns same results as JSON_TYPE
    let result: String = db
        .query_one("SELECT JSON_TYPEOF('{\"key\":\"value\"}')", ())
        .expect("Failed to get JSON_TYPEOF for object");
    assert_eq!(result, "object");

    let result: String = db
        .query_one("SELECT JSON_TYPEOF('[1,2,3]')", ())
        .expect("Failed to get JSON_TYPEOF for array");
    assert_eq!(result, "array");

    let result: String = db
        .query_one("SELECT JSON_TYPEOF('\"string\"')", ())
        .expect("Failed to get JSON_TYPEOF for string");
    assert_eq!(result, "string");

    let result: String = db
        .query_one("SELECT JSON_TYPEOF('42')", ())
        .expect("Failed to get JSON_TYPEOF for number");
    assert_eq!(result, "number");
}

/// Test JSON_VALID function
#[test]
fn test_json_valid_function() {
    let db = Database::open("memory://json_valid").expect("Failed to create database");

    // Test valid JSON object
    let result: i64 = db
        .query_one("SELECT JSON_VALID('{\"a\":1}')", ())
        .expect("Failed to get JSON_VALID for valid object");
    assert_eq!(result, 1);

    // Test valid JSON array
    let result: i64 = db
        .query_one("SELECT JSON_VALID('[1,2,3]')", ())
        .expect("Failed to get JSON_VALID for valid array");
    assert_eq!(result, 1);

    // Test valid JSON string
    let result: i64 = db
        .query_one("SELECT JSON_VALID('\"hello\"')", ())
        .expect("Failed to get JSON_VALID for valid string");
    assert_eq!(result, 1);

    // Test valid JSON number
    let result: i64 = db
        .query_one("SELECT JSON_VALID('123')", ())
        .expect("Failed to get JSON_VALID for valid number");
    assert_eq!(result, 1);

    // Test valid JSON boolean
    let result: i64 = db
        .query_one("SELECT JSON_VALID('true')", ())
        .expect("Failed to get JSON_VALID for valid boolean");
    assert_eq!(result, 1);

    // Test valid JSON null
    let result: i64 = db
        .query_one("SELECT JSON_VALID('null')", ())
        .expect("Failed to get JSON_VALID for valid null");
    assert_eq!(result, 1);

    // Test invalid JSON - missing quotes
    let result: i64 = db
        .query_one("SELECT JSON_VALID('invalid')", ())
        .expect("Failed to get JSON_VALID for invalid");
    assert_eq!(result, 0);

    // Test invalid JSON - incomplete object
    let result: i64 = db
        .query_one("SELECT JSON_VALID('{incomplete')", ())
        .expect("Failed to get JSON_VALID for incomplete object");
    assert_eq!(result, 0);

    // Test invalid JSON - trailing comma
    let result: i64 = db
        .query_one("SELECT JSON_VALID('[1,2,3,]')", ())
        .expect("Failed to get JSON_VALID for trailing comma");
    assert_eq!(result, 0);

    // Test invalid JSON - single quotes (JSON requires double quotes)
    let result: i64 = db
        .query_one("SELECT JSON_VALID('{''key'':''value''}')", ())
        .expect("Failed to get JSON_VALID for single quotes");
    assert_eq!(result, 0);

    // Test empty string (not valid JSON)
    let result: i64 = db
        .query_one("SELECT JSON_VALID('')", ())
        .expect("Failed to get JSON_VALID for empty string");
    assert_eq!(result, 0);
}

/// Test JSON_KEYS function
#[test]
fn test_json_keys_function() {
    let db = Database::open("memory://json_keys").expect("Failed to create database");

    // Test with simple object
    let result: String = db
        .query_one("SELECT JSON_KEYS('{\"name\":\"John\",\"age\":30}')", ())
        .expect("Failed to get JSON_KEYS for object");
    // Keys are returned as JSON array, order may vary
    assert!(result.contains("\"name\""));
    assert!(result.contains("\"age\""));

    // Test with nested object - should only return top-level keys
    let result: String = db
        .query_one(
            "SELECT JSON_KEYS('{\"user\":{\"name\":\"John\"},\"active\":true}')",
            (),
        )
        .expect("Failed to get JSON_KEYS for nested object");
    assert!(result.contains("\"user\""));
    assert!(result.contains("\"active\""));
    assert!(!result.contains("\"name\"")); // name is nested, not top-level

    // Test with empty object
    let result: String = db
        .query_one("SELECT JSON_KEYS('{}')", ())
        .expect("Failed to get JSON_KEYS for empty object");
    assert_eq!(result, "[]");

    // Test with single key
    let result: String = db
        .query_one("SELECT JSON_KEYS('{\"key\":\"value\"}')", ())
        .expect("Failed to get JSON_KEYS for single key");
    assert!(result.contains("\"key\""));
}

/// Test JSON_KEYS returns NULL for non-object types
#[test]
fn test_json_keys_non_object() {
    let db = Database::open("memory://json_keys_null").expect("Failed to create database");

    // Test with array - should return NULL
    let result = db
        .query("SELECT JSON_KEYS('[1,2,3]')", ())
        .expect("Failed to query JSON_KEYS for array");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_KEYS on array should return NULL");

    // Test with string - should return NULL
    let result = db
        .query("SELECT JSON_KEYS('\"hello\"')", ())
        .expect("Failed to query JSON_KEYS for string");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_KEYS on string should return NULL");

    // Test with number - should return NULL
    let result = db
        .query("SELECT JSON_KEYS('123')", ())
        .expect("Failed to query JSON_KEYS for number");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_KEYS on number should return NULL");
}

/// Test JSON functions with NULL input
#[test]
fn test_json_functions_with_null() {
    let db = Database::open("memory://json_null_input").expect("Failed to create database");

    // JSON_TYPE with NULL should return NULL
    let result = db
        .query("SELECT JSON_TYPE(NULL)", ())
        .expect("Failed to query JSON_TYPE with NULL");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_TYPE(NULL) should return NULL");

    // JSON_VALID with NULL should return NULL
    let result = db
        .query("SELECT JSON_VALID(NULL)", ())
        .expect("Failed to query JSON_VALID with NULL");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_VALID(NULL) should return NULL");

    // JSON_KEYS with NULL should return NULL
    let result = db
        .query("SELECT JSON_KEYS(NULL)", ())
        .expect("Failed to query JSON_KEYS with NULL");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Failed to get row");
    assert!(row.is_null(0), "JSON_KEYS(NULL) should return NULL");
}

/// Test JSON functions with table data
#[test]
fn test_json_functions_with_table() {
    let db = Database::open("memory://json_func_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_data (
            id INTEGER NOT NULL,
            data JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert various JSON types
    db.execute(
        "INSERT INTO json_data (id, data) VALUES (1, '{\"name\":\"John\",\"age\":30}')",
        (),
    )
    .expect("Failed to insert object");
    db.execute("INSERT INTO json_data (id, data) VALUES (2, '[1,2,3]')", ())
        .expect("Failed to insert array");
    db.execute(
        "INSERT INTO json_data (id, data) VALUES (3, '\"string\"')",
        (),
    )
    .expect("Failed to insert string");
    db.execute("INSERT INTO json_data (id, data) VALUES (4, '42')", ())
        .expect("Failed to insert number");
    db.execute("INSERT INTO json_data (id, data) VALUES (5, NULL)", ())
        .expect("Failed to insert null");

    // Query JSON_TYPE for each row
    let result = db
        .query(
            "SELECT id, JSON_TYPE(data) as type FROM json_data ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 5);

    // Check each type
    let row = rows[0].as_ref().expect("Failed to get row 1");
    let id: i64 = row.get(0).unwrap();
    let json_type: String = row.get(1).unwrap();
    assert_eq!(id, 1);
    assert_eq!(json_type, "object");

    let row = rows[1].as_ref().expect("Failed to get row 2");
    let id: i64 = row.get(0).unwrap();
    let json_type: String = row.get(1).unwrap();
    assert_eq!(id, 2);
    assert_eq!(json_type, "array");

    let row = rows[2].as_ref().expect("Failed to get row 3");
    let id: i64 = row.get(0).unwrap();
    let json_type: String = row.get(1).unwrap();
    assert_eq!(id, 3);
    assert_eq!(json_type, "string");

    let row = rows[3].as_ref().expect("Failed to get row 4");
    let id: i64 = row.get(0).unwrap();
    let json_type: String = row.get(1).unwrap();
    assert_eq!(id, 4);
    assert_eq!(json_type, "number");

    let row = rows[4].as_ref().expect("Failed to get row 5");
    let id: i64 = row.get(0).unwrap();
    assert_eq!(id, 5);
    assert!(row.is_null(1), "JSON_TYPE of NULL should be NULL");
}

/// Test JSON_VALID filtering in WHERE clause
#[test]
fn test_json_valid_in_where() {
    let db = Database::open("memory://json_valid_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_strings (
            id INTEGER NOT NULL,
            content TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert valid and invalid JSON strings
    db.execute(
        "INSERT INTO json_strings (id, content) VALUES (1, '{\"valid\":true}')",
        (),
    )
    .expect("Failed to insert valid JSON");
    db.execute(
        "INSERT INTO json_strings (id, content) VALUES (2, 'invalid json')",
        (),
    )
    .expect("Failed to insert invalid JSON");
    db.execute(
        "INSERT INTO json_strings (id, content) VALUES (3, '[1,2,3]')",
        (),
    )
    .expect("Failed to insert valid array");
    db.execute(
        "INSERT INTO json_strings (id, content) VALUES (4, '{broken')",
        (),
    )
    .expect("Failed to insert broken JSON");

    // Select only valid JSON using JSON_VALID
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM json_strings WHERE JSON_VALID(content) = 1",
            (),
        )
        .expect("Failed to count valid JSON");
    assert_eq!(count, 2, "Expected 2 valid JSON rows");

    // Select only invalid JSON
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM json_strings WHERE JSON_VALID(content) = 0",
            (),
        )
        .expect("Failed to count invalid JSON");
    assert_eq!(count, 2, "Expected 2 invalid JSON rows");
}
