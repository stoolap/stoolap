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

//! UPDATE Arithmetic Expression Tests
//!
//! Tests UPDATE statements with arithmetic expressions

use stoolap::Database;

/// Test UPDATE with arithmetic expressions
#[test]
fn test_update_arithmetic_verification() {
    let db = Database::open("memory://update_arith").expect("Failed to create database");

    // Create and populate test table
    db.execute(
        "CREATE TABLE arith_verify (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO arith_verify (id, value) VALUES (1, 10), (2, 20), (3, 5)",
        (),
    )
    .expect("Failed to insert data");

    // Execute UPDATE with arithmetic expression: value * 2 + 5
    let affected = db
        .execute("UPDATE arith_verify SET value = (value * 2 + 5)", ())
        .expect("Failed to execute UPDATE");
    assert_eq!(affected, 3, "Expected 3 rows affected");

    // Verify results
    // ID 1: 10 * 2 + 5 = 25
    let value1: i64 = db
        .query_one("SELECT value FROM arith_verify WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(value1, 25, "ID 1: expected 25, got {}", value1);

    // ID 2: 20 * 2 + 5 = 45
    let value2: i64 = db
        .query_one("SELECT value FROM arith_verify WHERE id = 2", ())
        .expect("Failed to query value");
    assert_eq!(value2, 45, "ID 2: expected 45, got {}", value2);

    // ID 3: 5 * 2 + 5 = 15
    let value3: i64 = db
        .query_one("SELECT value FROM arith_verify WHERE id = 3", ())
        .expect("Failed to query value");
    assert_eq!(value3, 15, "ID 3: expected 15, got {}", value3);
}

/// Test UPDATE arithmetic with zero value
#[test]
fn test_update_arithmetic_zero() {
    let db = Database::open("memory://update_zero").expect("Failed to create database");

    db.execute(
        "CREATE TABLE edge_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO edge_test (id, value) VALUES (1, 0)", ())
        .expect("Failed to insert data");

    // 0 * 2 + 5 = 5
    db.execute(
        "UPDATE edge_test SET value = (value * 2 + 5) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM edge_test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 5, "Expected 5, got {}", result);
}

/// Test UPDATE arithmetic with negative value
#[test]
fn test_update_arithmetic_negative() {
    let db = Database::open("memory://update_neg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE edge_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO edge_test (id, value) VALUES (1, -3)", ())
        .expect("Failed to insert data");

    // -3 * 2 + 5 = -1
    db.execute(
        "UPDATE edge_test SET value = (value * 2 + 5) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM edge_test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, -1, "Expected -1, got {}", result);
}

/// Test UPDATE arithmetic with large value
#[test]
fn test_update_arithmetic_large() {
    let db = Database::open("memory://update_large").expect("Failed to create database");

    db.execute(
        "CREATE TABLE edge_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO edge_test (id, value) VALUES (1, 1000)", ())
        .expect("Failed to insert data");

    // 1000 * 2 + 5 = 2005
    db.execute(
        "UPDATE edge_test SET value = (value * 2 + 5) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM edge_test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 2005, "Expected 2005, got {}", result);
}

/// Test UPDATE with complex arithmetic expression
#[test]
fn test_update_arithmetic_complex() {
    let db = Database::open("memory://update_complex").expect("Failed to create database");

    db.execute(
        "CREATE TABLE edge_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO edge_test (id, value) VALUES (1, 7)", ())
        .expect("Failed to insert data");

    // (7 * 3 - 1) / 2 = 20 / 2 = 10
    db.execute(
        "UPDATE edge_test SET value = (value * 3 - 1) / 2 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM edge_test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 10, "Expected 10, got {}", result);
}

/// Test UPDATE with addition
#[test]
fn test_update_arithmetic_add() {
    let db = Database::open("memory://update_add").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert data");

    db.execute("UPDATE test SET value = value + 50 WHERE id = 1", ())
        .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 150, "Expected 150, got {}", result);
}

/// Test UPDATE with subtraction
#[test]
fn test_update_arithmetic_subtract() {
    let db = Database::open("memory://update_sub").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert data");

    db.execute("UPDATE test SET value = value - 30 WHERE id = 1", ())
        .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 70, "Expected 70, got {}", result);
}

/// Test UPDATE with multiplication
#[test]
fn test_update_arithmetic_multiply() {
    let db = Database::open("memory://update_mul").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 25)", ())
        .expect("Failed to insert data");

    db.execute("UPDATE test SET value = value * 4 WHERE id = 1", ())
        .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 100, "Expected 100, got {}", result);
}

/// Test UPDATE with division
#[test]
fn test_update_arithmetic_divide() {
    let db = Database::open("memory://update_div").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert data");

    db.execute("UPDATE test SET value = value / 4 WHERE id = 1", ())
        .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 25, "Expected 25, got {}", result);
}

/// Test UPDATE with modulo
#[test]
fn test_update_arithmetic_modulo() {
    let db = Database::open("memory://update_mod").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 17)", ())
        .expect("Failed to insert data");

    db.execute("UPDATE test SET value = value % 5 WHERE id = 1", ())
        .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert_eq!(result, 2, "Expected 2, got {}", result);
}

/// Test UPDATE arithmetic with float values
#[test]
fn test_update_arithmetic_float() {
    let db = Database::open("memory://update_float").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO test (id, value) VALUES (1, 10.5)", ())
        .expect("Failed to insert data");

    // 10.5 * 2 + 0.5 = 21.5
    db.execute("UPDATE test SET value = value * 2 + 0.5 WHERE id = 1", ())
        .expect("Failed to update");

    let result: f64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query value");
    assert!(
        (result - 21.5).abs() < 0.001,
        "Expected 21.5, got {}",
        result
    );
}

/// Test UPDATE with conditional WHERE and arithmetic
#[test]
fn test_update_arithmetic_conditional() {
    let db = Database::open("memory://update_cond").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test (id, value, category) VALUES
         (1, 100, 'A'), (2, 200, 'B'), (3, 150, 'A'), (4, 50, 'B')",
        (),
    )
    .expect("Failed to insert data");

    // Only update category A: value = value + 10
    let affected = db
        .execute(
            "UPDATE test SET value = value + 10 WHERE category = 'A'",
            (),
        )
        .expect("Failed to update");
    assert_eq!(affected, 2, "Expected 2 rows affected");

    // Verify category A was updated
    let val1: i64 = db
        .query_one("SELECT value FROM test WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(val1, 110);

    let val3: i64 = db
        .query_one("SELECT value FROM test WHERE id = 3", ())
        .expect("Failed to query");
    assert_eq!(val3, 160);

    // Verify category B was not updated
    let val2: i64 = db
        .query_one("SELECT value FROM test WHERE id = 2", ())
        .expect("Failed to query");
    assert_eq!(val2, 200);

    let val4: i64 = db
        .query_one("SELECT value FROM test WHERE id = 4", ())
        .expect("Failed to query");
    assert_eq!(val4, 50);
}
