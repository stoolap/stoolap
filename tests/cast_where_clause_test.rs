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

//! CAST in WHERE Clause Tests
//!
//! Tests CAST expressions in WHERE clauses

use std::collections::HashSet;
use stoolap::Database;

fn setup_cast_where_db(test_name: &str) -> Database {
    let db = Database::open(&format!("memory://cast_where_{}", test_name))
        .expect("Failed to create database");

    // Create a test table with various data types
    db.execute(
        "CREATE TABLE test_cast_where (
            id INTEGER PRIMARY KEY,
            text_val TEXT,
            int_val INTEGER,
            float_val FLOAT,
            bool_val BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute(
        "INSERT INTO test_cast_where (id, text_val, int_val, float_val, bool_val)
        VALUES
            (1, '123', 123, 123.45, true),
            (2, '456', 456, 456.78, false),
            (3, '789', 789, 789.01, true),
            (4, '0', 0, 0.0, false),
            (5, '-50', -50, -50.5, true),
            (6, NULL, NULL, NULL, NULL)",
        (),
    )
    .expect("Failed to insert data");

    db
}

fn get_ids_from_query(db: &Database, query: &str) -> Vec<i64> {
    let result = db.query(query, ()).expect("Failed to execute query");
    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    ids.sort();
    ids
}

fn assert_ids_match(actual: &[i64], expected: &[i64], test_name: &str) {
    let actual_set: HashSet<_> = actual.iter().collect();
    let expected_set: HashSet<_> = expected.iter().collect();

    assert_eq!(
        actual_set, expected_set,
        "{}: Expected IDs {:?}, got {:?}",
        test_name, expected, actual
    );
}

/// Test CAST text_val to INTEGER with greater than
#[test]
fn test_cast_text_to_int_greater_than() {
    let db = setup_cast_where_db("greater_than");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) > 400",
    );

    assert_ids_match(&ids, &[2, 3], "CAST text to INTEGER > 400");
}

/// Test CAST text_val to INTEGER with less than
#[test]
fn test_cast_text_to_int_less_than() {
    let db = setup_cast_where_db("less_than");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) < 200",
    );

    assert_ids_match(&ids, &[1, 4, 5], "CAST text to INTEGER < 200");
}

/// Test CAST text_val to INTEGER with equals
#[test]
fn test_cast_text_to_int_equals() {
    let db = setup_cast_where_db("equals");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) = 456",
    );

    assert_ids_match(&ids, &[2], "CAST text to INTEGER = 456");
}

/// Test CAST text_val to INTEGER with NOT equals
#[test]
fn test_cast_text_to_int_not_equals() {
    let db = setup_cast_where_db("not_equals");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) <> 123 AND text_val IS NOT NULL",
    );

    assert_ids_match(&ids, &[2, 3, 4, 5], "CAST text to INTEGER <> 123");
}

/// Test CAST int_val to TEXT with LIKE
#[test]
fn test_cast_int_to_text_like() {
    let db = setup_cast_where_db("int_to_text_like");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(int_val AS TEXT) LIKE '4%'",
    );

    assert_ids_match(&ids, &[2], "CAST int to TEXT LIKE '4%'");
}

/// Test CAST float_val to INTEGER
#[test]
fn test_cast_float_to_int() {
    let db = setup_cast_where_db("float_to_int");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(float_val AS INTEGER) = 123",
    );

    assert_ids_match(&ids, &[1], "CAST float to INTEGER = 123");
}

/// Test CAST bool_val to INTEGER
#[test]
fn test_cast_bool_to_int_in_where() {
    let db = setup_cast_where_db("bool_to_int");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(bool_val AS INTEGER) = 1",
    );

    assert_ids_match(&ids, &[1, 3, 5], "CAST bool to INTEGER = 1");
}

/// Test multiple CAST in WHERE
#[test]
fn test_multiple_cast_in_where() {
    let db = setup_cast_where_db("multiple_cast");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) > 100 AND CAST(float_val AS INTEGER) < 500",
    );

    assert_ids_match(&ids, &[1, 2], "Multiple CAST in WHERE");
}

/// Test CAST with BETWEEN
#[test]
fn test_cast_with_between() {
    let db = setup_cast_where_db("between");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) BETWEEN 100 AND 500",
    );

    assert_ids_match(&ids, &[1, 2], "CAST with BETWEEN");
}

/// Test CAST with IN list
#[test]
fn test_cast_with_in_list() {
    let db = setup_cast_where_db("in_list");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) IN (123, 789)",
    );

    assert_ids_match(&ids, &[1, 3], "CAST with IN list");
}

/// Test CAST with OR
#[test]
fn test_cast_with_or() {
    let db = setup_cast_where_db("with_or");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) = 123 OR CAST(text_val AS INTEGER) = 456",
    );

    assert_ids_match(&ids, &[1, 2], "CAST with OR");
}

/// Test CAST with negative values
#[test]
fn test_cast_negative_values() {
    let db = setup_cast_where_db("negative");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) < 0",
    );

    assert_ids_match(&ids, &[5], "CAST negative values");
}

/// Test CAST to FLOAT comparison
#[test]
fn test_cast_to_float_comparison() {
    let db = setup_cast_where_db("float_comparison");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS FLOAT) > 400.0",
    );

    assert_ids_match(&ids, &[2, 3], "CAST to FLOAT > 400.0");
}

/// Test CAST with arithmetic in WHERE
#[test]
fn test_cast_with_arithmetic() {
    let db = setup_cast_where_db("arithmetic");

    let ids = get_ids_from_query(
        &db,
        "SELECT id FROM test_cast_where WHERE CAST(text_val AS INTEGER) * 2 > 800",
    );

    assert_ids_match(&ids, &[2, 3], "CAST with arithmetic * 2 > 800");
}
