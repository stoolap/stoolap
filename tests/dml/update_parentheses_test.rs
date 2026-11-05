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

//! UPDATE Parentheses Tests
//!
//! Tests UPDATE statements with parenthesized expressions

use stoolap::Database;

fn setup_paren_test_db() -> Database {
    let db = Database::open("memory://update_paren").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert data");

    db
}

/// Test UPDATE without parentheses
#[test]
fn test_update_without_parentheses() {
    let db = setup_paren_test_db();

    // Reset value
    db.execute("UPDATE paren_test SET value = 10 WHERE id = 1", ())
        .expect("Failed to reset value");

    // value * 2 + 5 = 10 * 2 + 5 = 25
    db.execute(
        "UPDATE paren_test SET value = value * 2 + 5 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 25, "Expected 25 (10 * 2 + 5)");
}

/// Test UPDATE with parentheses around entire expression
#[test]
fn test_update_with_outer_parentheses() {
    let db = Database::open("memory://update_paren_outer").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // (value * 2 + 5) = (10 * 2 + 5) = 25
    db.execute(
        "UPDATE paren_test SET value = (value * 2 + 5) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 25, "Expected 25 ((10 * 2) + 5)");
}

/// Test UPDATE with parentheses around multiplication
#[test]
fn test_update_with_multiplication_parentheses() {
    let db = Database::open("memory://update_paren_mult").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // (value * 2) + 5 = (10 * 2) + 5 = 25
    db.execute(
        "UPDATE paren_test SET value = (value * 2) + 5 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 25, "Expected 25 ((10 * 2) + 5)");
}

/// Test UPDATE with parentheses around addition - changes order of operations
#[test]
fn test_update_with_addition_parentheses() {
    let db = Database::open("memory://update_paren_add").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // value * (2 + 5) = 10 * 7 = 70
    db.execute(
        "UPDATE paren_test SET value = value * (2 + 5) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 70, "Expected 70 (10 * (2 + 5))");
}

/// Test UPDATE with nested parentheses
#[test]
fn test_update_with_nested_parentheses() {
    let db = Database::open("memory://update_paren_nested").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // ((value + 5) * 2) = ((10 + 5) * 2) = 30
    db.execute(
        "UPDATE paren_test SET value = ((value + 5) * 2) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 30, "Expected 30 (((10 + 5) * 2))");
}

/// Test UPDATE with division and parentheses
#[test]
fn test_update_with_division_parentheses() {
    let db = Database::open("memory://update_paren_div").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 100)", ())
        .expect("Failed to insert data");

    // value / (2 + 3) = 100 / 5 = 20
    db.execute(
        "UPDATE paren_test SET value = value / (2 + 3) WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 20, "Expected 20 (100 / (2 + 3))");
}

/// Test UPDATE with subtraction and parentheses
#[test]
fn test_update_with_subtraction_parentheses() {
    let db = Database::open("memory://update_paren_sub").expect("Failed to create database");

    db.execute(
        "CREATE TABLE paren_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO paren_test (id, value) VALUES (1, 50)", ())
        .expect("Failed to insert data");

    // (value - 10) * 2 = (50 - 10) * 2 = 80
    db.execute(
        "UPDATE paren_test SET value = (value - 10) * 2 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let value: i64 = db
        .query_one("SELECT value FROM paren_test WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(value, 80, "Expected 80 ((50 - 10) * 2)");
}
