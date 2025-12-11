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

//! IS NULL / IS NOT NULL Tests
//!
//! Tests IS NULL and IS NOT NULL expressions through SQL queries

use stoolap::Database;

fn setup_nullable_table(db: &Database) {
    db.execute(
        "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, optional_value INTEGER)",
        (),
    )
    .expect("Failed to create test_table");

    // Insert row with NULL optional_value
    db.execute(
        "INSERT INTO test_table (id, name, optional_value) VALUES (1, 'Alice', NULL)",
        (),
    )
    .expect("Failed to insert Alice");

    // Insert row with non-NULL optional_value
    db.execute(
        "INSERT INTO test_table (id, name, optional_value) VALUES (2, 'Bob', 42)",
        (),
    )
    .expect("Failed to insert Bob");

    // Insert more rows for testing
    db.execute(
        "INSERT INTO test_table (id, name, optional_value) VALUES (3, 'Charlie', NULL)",
        (),
    )
    .expect("Failed to insert Charlie");
    db.execute(
        "INSERT INTO test_table (id, name, optional_value) VALUES (4, 'Diana', 100)",
        (),
    )
    .expect("Failed to insert Diana");
    db.execute(
        "INSERT INTO test_table (id, name, optional_value) VALUES (5, 'Eve', NULL)",
        (),
    )
    .expect("Failed to insert Eve");
}

/// Test IS NULL expression - should find rows where optional_value is NULL
#[test]
fn test_is_null() {
    let db = Database::open("memory://is_null_test").expect("Failed to create database");
    setup_nullable_table(&db);

    // Query rows where optional_value IS NULL
    let result = db
        .query(
            "SELECT id, name FROM test_table WHERE optional_value IS NULL ORDER BY id",
            (),
        )
        .expect("Failed to execute IS NULL query");

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    // Should return Alice (1), Charlie (3), Eve (5)
    assert_eq!(rows.len(), 3, "Expected 3 rows with NULL optional_value");
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (3, "Charlie".to_string()));
    assert_eq!(rows[2], (5, "Eve".to_string()));
}

/// Test IS NOT NULL expression - should find rows where optional_value is not NULL
#[test]
fn test_is_not_null() {
    let db = Database::open("memory://is_not_null_test").expect("Failed to create database");
    setup_nullable_table(&db);

    // Query rows where optional_value IS NOT NULL
    let result = db
        .query("SELECT id, name, optional_value FROM test_table WHERE optional_value IS NOT NULL ORDER BY id", ())
        .expect("Failed to execute IS NOT NULL query");

    let mut rows: Vec<(i64, String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let value: i64 = row.get(2).unwrap();
        rows.push((id, name, value));
    }

    // Should return Bob (2) with value 42, Diana (4) with value 100
    assert_eq!(
        rows.len(),
        2,
        "Expected 2 rows with non-NULL optional_value"
    );
    assert_eq!(rows[0], (2, "Bob".to_string(), 42));
    assert_eq!(rows[1], (4, "Diana".to_string(), 100));
}

/// Test COUNT with IS NULL
#[test]
fn test_count_with_is_null() {
    let db = Database::open("memory://count_is_null_test").expect("Failed to create database");
    setup_nullable_table(&db);

    // Count rows where optional_value IS NULL
    let null_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_table WHERE optional_value IS NULL",
            (),
        )
        .expect("Failed to execute count with IS NULL");
    assert_eq!(null_count, 3, "Expected 3 rows with NULL optional_value");

    // Count rows where optional_value IS NOT NULL
    let not_null_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_table WHERE optional_value IS NOT NULL",
            (),
        )
        .expect("Failed to execute count with IS NOT NULL");
    assert_eq!(
        not_null_count, 2,
        "Expected 2 rows with non-NULL optional_value"
    );
}

/// Test IS NULL on TEXT column
#[test]
fn test_is_null_on_text() {
    let db = Database::open("memory://is_null_text_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE text_test (id INTEGER PRIMARY KEY, description TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO text_test (id, description) VALUES (1, 'Has text')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO text_test (id, description) VALUES (2, NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO text_test (id, description) VALUES (3, 'Another text')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO text_test (id, description) VALUES (4, NULL)",
        (),
    )
    .unwrap();

    // Find rows with NULL description
    let null_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM text_test WHERE description IS NULL",
            (),
        )
        .expect("Failed to count NULL descriptions");
    assert_eq!(null_count, 2);

    // Find rows with non-NULL description
    let not_null_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM text_test WHERE description IS NOT NULL",
            (),
        )
        .expect("Failed to count non-NULL descriptions");
    assert_eq!(not_null_count, 2);
}

/// Test IS NULL on FLOAT column
#[test]
fn test_is_null_on_float() {
    let db = Database::open("memory://is_null_float_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE float_test (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO float_test (id, amount) VALUES (1, 100.5)", ())
        .unwrap();
    db.execute("INSERT INTO float_test (id, amount) VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO float_test (id, amount) VALUES (3, 0.0)", ())
        .unwrap();
    db.execute("INSERT INTO float_test (id, amount) VALUES (4, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO float_test (id, amount) VALUES (5, -50.25)", ())
        .unwrap();

    // Note: 0.0 is NOT NULL - it's a valid float value
    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM float_test WHERE amount IS NULL", ())
        .expect("Failed to count NULL amounts");
    assert_eq!(null_count, 2, "Expected 2 NULL amounts");

    let not_null_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM float_test WHERE amount IS NOT NULL",
            (),
        )
        .expect("Failed to count non-NULL amounts");
    assert_eq!(not_null_count, 3, "Expected 3 non-NULL amounts");
}

/// Test IS NULL combined with other conditions (AND)
#[test]
fn test_is_null_with_and() {
    let db = Database::open("memory://is_null_and_test").expect("Failed to create database");
    setup_nullable_table(&db);

    // Find rows where optional_value IS NULL AND id > 2
    let result = db
        .query(
            "SELECT id, name FROM test_table WHERE optional_value IS NULL AND id > 2 ORDER BY id",
            (),
        )
        .expect("Failed to execute combined query");

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    // Should return Charlie (3), Eve (5)
    assert_eq!(rows.len(), 2, "Expected 2 rows");
    assert_eq!(rows[0], (3, "Charlie".to_string()));
    assert_eq!(rows[1], (5, "Eve".to_string()));
}

/// Test IS NULL combined with other conditions (OR)
#[test]
fn test_is_null_with_or() {
    let db = Database::open("memory://is_null_or_test").expect("Failed to create database");
    setup_nullable_table(&db);

    // Find rows where optional_value IS NULL OR optional_value > 50
    let result = db
        .query("SELECT id, name FROM test_table WHERE optional_value IS NULL OR optional_value > 50 ORDER BY id", ())
        .expect("Failed to execute combined query");

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    // Should return Alice (1), Charlie (3), Diana (4, value=100), Eve (5)
    assert_eq!(rows.len(), 4, "Expected 4 rows");
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (3, "Charlie".to_string()));
    assert_eq!(rows[2], (4, "Diana".to_string()));
    assert_eq!(rows[3], (5, "Eve".to_string()));
}

/// Test IS NULL in SELECT expression (selecting NULL checks)
#[test]
fn test_is_null_in_select() {
    let db = Database::open("memory://is_null_select_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE simple (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO simple (id, value) VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO simple (id, value) VALUES (2, NULL)", ())
        .unwrap();

    // This tests if IS NULL can be used in SELECT to produce boolean results
    // Some databases support: SELECT id, value IS NULL as is_null FROM simple
    // For now, we just verify IS NULL works in WHERE clauses
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM simple WHERE value IS NULL", ())
        .unwrap();
    assert_eq!(count, 1);
}

/// Test that empty string is NOT the same as NULL
#[test]
fn test_empty_string_is_not_null() {
    let db = Database::open("memory://empty_string_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE str_test (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO str_test (id, value) VALUES (1, '')", ())
        .unwrap(); // empty string
    db.execute("INSERT INTO str_test (id, value) VALUES (2, NULL)", ())
        .unwrap(); // actual NULL
    db.execute("INSERT INTO str_test (id, value) VALUES (3, 'text')", ())
        .unwrap(); // regular text

    // Empty string should NOT be NULL
    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM str_test WHERE value IS NULL", ())
        .expect("Failed to count NULLs");
    assert_eq!(null_count, 1, "Only 1 row should have NULL value");

    let not_null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM str_test WHERE value IS NOT NULL", ())
        .expect("Failed to count non-NULLs");
    assert_eq!(
        not_null_count, 2,
        "Empty string and regular text are both NOT NULL"
    );
}

/// Test IS NULL with all rows having NULL
#[test]
fn test_all_nulls() {
    let db = Database::open("memory://all_nulls_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE all_nulls (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO all_nulls (id, value) VALUES (1, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO all_nulls (id, value) VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO all_nulls (id, value) VALUES (3, NULL)", ())
        .unwrap();

    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM all_nulls WHERE value IS NULL", ())
        .unwrap();
    assert_eq!(null_count, 3, "All 3 rows should be NULL");

    let not_null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM all_nulls WHERE value IS NOT NULL", ())
        .unwrap();
    assert_eq!(not_null_count, 0, "No rows should be NOT NULL");
}

/// Test IS NULL with no rows having NULL
#[test]
fn test_no_nulls() {
    let db = Database::open("memory://no_nulls_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE no_nulls (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO no_nulls (id, value) VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO no_nulls (id, value) VALUES (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO no_nulls (id, value) VALUES (3, 30)", ())
        .unwrap();

    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM no_nulls WHERE value IS NULL", ())
        .unwrap();
    assert_eq!(null_count, 0, "No rows should be NULL");

    let not_null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM no_nulls WHERE value IS NOT NULL", ())
        .unwrap();
    assert_eq!(not_null_count, 3, "All 3 rows should be NOT NULL");
}
