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

//! Data Type Validation Tests
//!
//! Tests data type validation during INSERT operations

use stoolap::Database;

/// Test inserting valid data with all supported data types
#[test]
fn test_insert_all_data_types() {
    let db = Database::open("memory://data_types").expect("Failed to create database");

    // Create a table with all supported data types
    db.execute(
        "CREATE TABLE all_types (
            id INTEGER PRIMARY KEY,
            int_col INTEGER,
            float_col FLOAT,
            string_col TEXT,
            bool_col BOOLEAN,
            timestamp_col TIMESTAMP,
            json_col JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert a valid row with all data types
    db.execute(
        "INSERT INTO all_types VALUES (
            1,
            42,
            3.5,
            'hello',
            true,
            TIMESTAMP '2023-01-15 10:30:00',
            '{\"key\":\"value\"}'
        )",
        (),
    )
    .expect("Failed to insert valid row");

    // Verify the row was inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM all_types", ())
        .expect("Failed to count rows");
    assert_eq!(count, 1, "Expected 1 row inserted");

    // Verify each column value
    let result = db
        .query("SELECT * FROM all_types WHERE id = 1", ())
        .expect("Failed to query row");

    for row in result {
        let row = row.expect("Failed to get row");

        let id: i64 = row.get(0).unwrap();
        assert_eq!(id, 1);

        let int_val: i64 = row.get(1).unwrap();
        assert_eq!(int_val, 42);

        let float_val: f64 = row.get(2).unwrap();
        assert!((float_val - 3.5).abs() < 0.001);

        let string_val: String = row.get(3).unwrap();
        assert_eq!(string_val, "hello");

        let bool_val: bool = row.get(4).unwrap();
        assert!(bool_val);
    }
}

/// Test inserting NULL values in nullable columns
#[test]
fn test_insert_null_values() {
    let db = Database::open("memory://null_types").expect("Failed to create database");

    // Create a table with nullable columns
    db.execute(
        "CREATE TABLE nullable_types (
            id INTEGER PRIMARY KEY,
            int_col INTEGER,
            float_col FLOAT,
            string_col TEXT,
            bool_col BOOLEAN,
            timestamp_col TIMESTAMP,
            json_col JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert a row with all NULL values except primary key
    db.execute(
        "INSERT INTO nullable_types VALUES (1, NULL, NULL, NULL, NULL, NULL, NULL)",
        (),
    )
    .expect("Failed to insert row with NULL values");

    // Verify the row was inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM nullable_types", ())
        .expect("Failed to count rows");
    assert_eq!(count, 1, "Expected 1 row inserted");

    // Verify NULL values
    let result = db
        .query(
            "SELECT int_col, float_col, string_col FROM nullable_types WHERE id = 1",
            (),
        )
        .expect("Failed to query row");

    for row in result {
        let row = row.expect("Failed to get row");

        let int_val: Option<i64> = row.get(0).unwrap();
        assert!(int_val.is_none(), "Expected NULL for int_col");

        let float_val: Option<f64> = row.get(1).unwrap();
        assert!(float_val.is_none(), "Expected NULL for float_col");

        let string_val: Option<String> = row.get(2).unwrap();
        assert!(string_val.is_none(), "Expected NULL for string_col");
    }
}

/// Test non-nullable constraint enforcement
#[test]
fn test_non_nullable_constraint() {
    let db = Database::open("memory://non_null").expect("Failed to create database");

    // Create a table with a NOT NULL constraint
    db.execute(
        "CREATE TABLE required_fields (
            id INTEGER PRIMARY KEY,
            required_text TEXT NOT NULL,
            optional_text TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // This should succeed - required field has value
    db.execute(
        "INSERT INTO required_fields VALUES (1, 'required', 'optional')",
        (),
    )
    .expect("Failed to insert valid row");

    // This should succeed - optional field is NULL
    db.execute(
        "INSERT INTO required_fields VALUES (2, 'required', NULL)",
        (),
    )
    .expect("Failed to insert row with optional NULL");

    // This should fail - required field is NULL
    let result = db.execute(
        "INSERT INTO required_fields VALUES (3, NULL, 'optional')",
        (),
    );
    assert!(
        result.is_err(),
        "Expected error for NULL in NOT NULL column"
    );

    // Verify only 2 rows were inserted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM required_fields", ())
        .expect("Failed to count rows");
    assert_eq!(count, 2, "Expected 2 valid rows");
}

/// Test various integer values
#[test]
fn test_integer_values() {
    let db = Database::open("memory://int_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE int_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Test various integer values
    let test_values: &[i64] = &[0, 1, -1, 100, -100, i64::MAX / 2, i64::MIN / 2];

    for (i, &value) in test_values.iter().enumerate() {
        db.execute(
            "INSERT INTO int_test (id, value) VALUES (?, ?)",
            (i as i64 + 1, value),
        )
        .expect(&format!("Failed to insert value {}", value));
    }

    // Verify all values
    for (i, &expected) in test_values.iter().enumerate() {
        let actual: i64 = db
            .query_one(
                &format!("SELECT value FROM int_test WHERE id = {}", i + 1),
                (),
            )
            .expect("Failed to query value");
        assert_eq!(actual, expected, "Value mismatch for id {}", i + 1);
    }
}

/// Test various float values
#[test]
fn test_float_values() {
    let db = Database::open("memory://float_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE float_test (id INTEGER PRIMARY KEY, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Test various float values
    let test_values: &[f64] = &[
        0.0,
        1.0,
        -1.0,
        3.54159,
        -2.81828,
        1e10,
        1e-10,
        f64::MAX / 2.0,
        f64::MIN / 2.0,
    ];

    for (i, &value) in test_values.iter().enumerate() {
        db.execute(
            "INSERT INTO float_test (id, value) VALUES (?, ?)",
            (i as i64 + 1, value),
        )
        .expect(&format!("Failed to insert value {}", value));
    }

    // Verify all values
    for (i, &expected) in test_values.iter().enumerate() {
        let actual: f64 = db
            .query_one(
                &format!("SELECT value FROM float_test WHERE id = {}", i + 1),
                (),
            )
            .expect("Failed to query value");

        // Use relative tolerance for comparison
        let tolerance = expected.abs() * 1e-10 + 1e-15;
        assert!(
            (actual - expected).abs() < tolerance,
            "Value mismatch for id {}: expected {}, got {}",
            i + 1,
            expected,
            actual
        );
    }
}

/// Test boolean values
#[test]
fn test_boolean_values() {
    let db = Database::open("memory://bool_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE bool_test (id INTEGER PRIMARY KEY, value BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    // Insert true and false values
    db.execute("INSERT INTO bool_test VALUES (1, true)", ())
        .expect("Failed to insert true");
    db.execute("INSERT INTO bool_test VALUES (2, false)", ())
        .expect("Failed to insert false");
    db.execute("INSERT INTO bool_test VALUES (3, TRUE)", ())
        .expect("Failed to insert TRUE");
    db.execute("INSERT INTO bool_test VALUES (4, FALSE)", ())
        .expect("Failed to insert FALSE");

    // Verify values
    let true_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_test WHERE value = true", ())
        .expect("Failed to count true values");
    assert_eq!(true_count, 2, "Expected 2 true values");

    let false_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_test WHERE value = false", ())
        .expect("Failed to count false values");
    assert_eq!(false_count, 2, "Expected 2 false values");
}

/// Test text/string values
#[test]
fn test_text_values() {
    let db = Database::open("memory://text_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE text_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Test various text values
    let long_text = "long text ".repeat(100);
    let test_values: Vec<&str> = vec![
        "hello",
        "world",
        "",                                        // empty string
        "hello world",                             // space
        "hello\nworld",                            // newline
        "hello\tworld",                            // tab
        "unicode: \u{1F600}",                      // emoji
        "special: 'quotes' and \"double quotes\"", // quotes
        "numbers: 12345",                          // mixed
        &long_text,                                // long string
    ];

    for (i, value) in test_values.iter().enumerate() {
        db.execute(
            "INSERT INTO text_test (id, value) VALUES (?, ?)",
            (i as i64 + 1, *value),
        )
        .expect(&format!("Failed to insert value '{}'", value));
    }

    // Verify count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM text_test", ())
        .expect("Failed to count rows");
    assert_eq!(
        count,
        test_values.len() as i64,
        "Expected all rows inserted"
    );
}

/// Test timestamp values
#[test]
fn test_timestamp_values() {
    let db = Database::open("memory://ts_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE ts_test (id INTEGER PRIMARY KEY, value TIMESTAMP)",
        (),
    )
    .expect("Failed to create table");

    // Test various timestamp formats
    db.execute(
        "INSERT INTO ts_test VALUES (1, TIMESTAMP '2023-01-15 10:30:00')",
        (),
    )
    .expect("Failed to insert timestamp");
    db.execute(
        "INSERT INTO ts_test VALUES (2, TIMESTAMP '2023-12-31 23:59:59')",
        (),
    )
    .expect("Failed to insert timestamp");
    db.execute(
        "INSERT INTO ts_test VALUES (3, TIMESTAMP '2000-01-01 00:00:00')",
        (),
    )
    .expect("Failed to insert timestamp");

    // Verify count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM ts_test", ())
        .expect("Failed to count rows");
    assert_eq!(count, 3, "Expected 3 timestamp rows");
}

/// Test JSON values
#[test]
fn test_json_values() {
    let db = Database::open("memory://json_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_test (id INTEGER PRIMARY KEY, value JSON)",
        (),
    )
    .expect("Failed to create table");

    // Test various JSON values
    let json_values = &[
        r#"{"key":"value"}"#,
        r#"[1,2,3]"#,
        r#"{"nested":{"inner":"value"}}"#,
        r#"{"array":[1,2,3],"string":"hello"}"#,
        r#"{}"#,
        r#"[]"#,
        r#"{"bool":true,"null":null,"number":42}"#,
    ];

    for (i, value) in json_values.iter().enumerate() {
        db.execute(
            "INSERT INTO json_test (id, value) VALUES (?, ?)",
            (i as i64 + 1, *value),
        )
        .expect(&format!("Failed to insert JSON: {}", value));
    }

    // Verify count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM json_test", ())
        .expect("Failed to count rows");
    assert_eq!(
        count,
        json_values.len() as i64,
        "Expected all JSON rows inserted"
    );
}

/// Test mixed data types in single table
#[test]
fn test_mixed_data_insert_query() {
    let db = Database::open("memory://mixed_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price FLOAT,
            in_stock BOOLEAN,
            category TEXT,
            metadata JSON
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert mixed data
    db.execute(
        r#"INSERT INTO products VALUES
            (1, 'Laptop', 999.99, true, 'Electronics', '{"brand":"Dell"}'),
            (2, 'Book', 19.99, true, 'Books', '{"author":"Unknown"}'),
            (3, 'Chair', 149.50, false, 'Furniture', NULL)"#,
        (),
    )
    .expect("Failed to insert products");

    // Query with various conditions
    let electronics_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM products WHERE category = 'Electronics'",
            (),
        )
        .expect("Failed to count electronics");
    assert_eq!(electronics_count, 1);

    let expensive_count: i64 = db
        .query_one("SELECT COUNT(*) FROM products WHERE price > 100", ())
        .expect("Failed to count expensive items");
    assert_eq!(expensive_count, 2);

    let in_stock_count: i64 = db
        .query_one("SELECT COUNT(*) FROM products WHERE in_stock = true", ())
        .expect("Failed to count in-stock items");
    assert_eq!(in_stock_count, 2);
}
