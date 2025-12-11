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

//! Parameter Binding Tests
//!
//! Tests SQL parameter binding with ? placeholders

use stoolap::Database;
use stoolap::Value;

/// Test integer parameter binding
#[test]
fn test_integer_parameter() {
    let db = Database::open("memory://param_int").expect("Failed to create database");

    db.execute("CREATE TABLE test_params (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert with parameter
    db.execute("INSERT INTO test_params (id) VALUES (?)", (42,))
        .expect("Failed to insert with integer parameter");

    let result: i64 = db
        .query_one("SELECT id FROM test_params", ())
        .expect("Failed to query");
    assert_eq!(result, 42);
}

/// Test string parameter binding
#[test]
fn test_string_parameter() {
    let db = Database::open("memory://param_str").expect("Failed to create database");

    db.execute("CREATE TABLE test_params (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_params (id, name) VALUES (1, ?)",
        ("John Doe",),
    )
    .expect("Failed to insert with string parameter");

    let result: String = db
        .query_one("SELECT name FROM test_params WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "John Doe");
}

/// Test float parameter binding
#[test]
fn test_float_parameter() {
    let db = Database::open("memory://param_float").expect("Failed to create database");

    db.execute("CREATE TABLE test_params (id INTEGER, salary FLOAT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_params (id, salary) VALUES (1, ?)",
        (75000.50,),
    )
    .expect("Failed to insert with float parameter");

    let result: f64 = db
        .query_one("SELECT salary FROM test_params WHERE id = 1", ())
        .expect("Failed to query");
    assert!((result - 75000.50).abs() < 0.01);
}

/// Test boolean parameter binding
#[test]
fn test_boolean_parameter() {
    let db = Database::open("memory://param_bool").expect("Failed to create database");

    db.execute("CREATE TABLE test_params (id INTEGER, active BOOLEAN)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_params (id, active) VALUES (1, ?)",
        (true,),
    )
    .expect("Failed to insert with boolean parameter");

    let result: bool = db
        .query_one("SELECT active FROM test_params WHERE id = 1", ())
        .expect("Failed to query");
    assert!(result);
}

/// Test multiple parameters
#[test]
fn test_multiple_parameters() {
    let db = Database::open("memory://param_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER, name TEXT, salary FLOAT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees (id, name, salary, active) VALUES (?, ?, ?, ?)",
        (101, "Jane Smith", 85000.75, false),
    )
    .expect("Failed to insert with multiple parameters");

    let name: String = db
        .query_one("SELECT name FROM employees WHERE id = 101", ())
        .expect("Failed to query");
    assert_eq!(name, "Jane Smith");
}

/// Test parameter in WHERE clause
#[test]
fn test_where_clause_parameter() {
    let db = Database::open("memory://param_where").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items (id, value) VALUES (2, 200)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items (id, value) VALUES (3, 300)", ())
        .expect("Failed to insert");

    // Query with parameter
    let result = db
        .query("SELECT id FROM items WHERE value > ?", (150,))
        .expect("Failed to query with parameter");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids.len(), 2, "Expected 2 rows with value > 150");
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}

/// Test UPDATE with parameters
#[test]
fn test_update_with_parameters() {
    let db = Database::open("memory://param_update").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users (id, name) VALUES (1, 'Original')", ())
        .expect("Failed to insert");

    db.execute("UPDATE users SET name = ? WHERE id = ?", ("Updated", 1))
        .expect("Failed to update with parameters");

    let result: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(result, "Updated");
}

/// Test DELETE with parameter
#[test]
fn test_delete_with_parameter() {
    let db = Database::open("memory://param_delete").expect("Failed to create database");

    db.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO records (id) VALUES (1)", ())
        .unwrap();
    db.execute("INSERT INTO records (id) VALUES (2)", ())
        .unwrap();
    db.execute("INSERT INTO records (id) VALUES (3)", ())
        .unwrap();

    db.execute("DELETE FROM records WHERE id = ?", (2,))
        .expect("Failed to delete with parameter");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM records", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Should have 2 records after delete");
}

/// Test NULL parameter using Option
#[test]
fn test_null_parameter() {
    let db = Database::open("memory://param_null").expect("Failed to create database");

    db.execute("CREATE TABLE nullable (id INTEGER, value TEXT)", ())
        .expect("Failed to create table");

    // Use Value::Null directly for explicit NULL
    db.execute(
        "INSERT INTO nullable (id, value) VALUES (?, ?)",
        (1, Value::Null(stoolap::DataType::Text)),
    )
    .expect("Failed to insert with NULL parameter");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM nullable WHERE value IS NULL", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Should have 1 row with NULL value");
}

/// Test parameter in complex expression
#[test]
fn test_complex_expression_parameter() {
    let db = Database::open("memory://param_complex").expect("Failed to create database");

    db.execute("CREATE TABLE data (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO data (id, value) VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO data (id, value) VALUES (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO data (id, value) VALUES (3, 30)", ())
        .unwrap();

    // Query with parameters in complex expression
    let result = db
        .query(
            "SELECT * FROM data WHERE (id = ? OR id = ?) AND value > ?",
            (1, 3, 5),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(
        count, 2,
        "Expected 2 rows matching (id=1 OR id=3) AND value>5"
    );
}

/// Test parameter in LIMIT clause
#[test]
fn test_limit_parameter() {
    let db = Database::open("memory://param_limit").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (id INTEGER)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO numbers (id) VALUES ({})", i), ())
            .unwrap();
    }

    let result = db
        .query("SELECT * FROM numbers ORDER BY id LIMIT ?", (5,))
        .expect("Failed to query with LIMIT parameter");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 5, "LIMIT ? should limit to 5 rows");
}
