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

//! FIRST/LAST Aggregate Function Tests
//!
//! Tests FIRST and LAST aggregate functions

use stoolap::Database;

fn setup_first_last_table(db: &Database) {
    db.execute(
        "CREATE TABLE test_first_last (
            id INTEGER,
            group_id INTEGER,
            value INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_first_last (id, group_id, value, text_value) VALUES
        (1, 1, 10, 'Apple'),
        (2, 1, 20, 'Banana'),
        (3, 1, 30, 'Cherry'),
        (4, 2, 15, 'Grape'),
        (5, 2, 25, 'Lemon'),
        (6, 2, NULL, 'Orange'),
        (7, 3, NULL, NULL),
        (8, 3, 35, 'Pear'),
        (9, 3, 45, 'Plum')",
        (),
    )
    .expect("Failed to insert data");
}

/// Test FIRST function with GROUP BY
#[test]
fn test_first_function() {
    let db = Database::open("memory://first_fn").expect("Failed to create database");
    setup_first_last_table(&db);

    let result = db
        .query(
            "SELECT group_id, FIRST(value), FIRST(text_value)
             FROM test_first_last
             GROUP BY group_id
             ORDER BY group_id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let group_id: i64 = row.get(0).unwrap();
        row_count += 1;

        // FIRST returns the first non-NULL value in the group
        match group_id {
            1 => {
                // Group 1: values are 10, 20, 30 - FIRST should be 10
                let first_val: i64 = row.get(1).unwrap();
                assert_eq!(first_val, 10, "Group 1 FIRST value should be 10");
                let first_text: String = row.get(2).unwrap();
                assert_eq!(first_text, "Apple", "Group 1 FIRST text should be Apple");
            }
            2 => {
                // Group 2: values are 15, 25, NULL - FIRST should be 15
                let first_val: i64 = row.get(1).unwrap();
                assert_eq!(first_val, 15, "Group 2 FIRST value should be 15");
                let first_text: String = row.get(2).unwrap();
                assert_eq!(first_text, "Grape", "Group 2 FIRST text should be Grape");
            }
            3 => {
                // Group 3: values are NULL, 35, 45 - FIRST should be 35
                let first_val: i64 = row.get(1).unwrap();
                assert_eq!(first_val, 35, "Group 3 FIRST value should be 35");
                let first_text: String = row.get(2).unwrap();
                assert_eq!(first_text, "Pear", "Group 3 FIRST text should be Pear");
            }
            _ => panic!("Unexpected group_id: {}", group_id),
        }
    }

    assert_eq!(row_count, 3, "Expected 3 groups");
}

/// Test LAST function with GROUP BY
#[test]
fn test_last_function() {
    let db = Database::open("memory://last_fn").expect("Failed to create database");
    setup_first_last_table(&db);

    let result = db
        .query(
            "SELECT group_id, LAST(value), LAST(text_value)
             FROM test_first_last
             GROUP BY group_id
             ORDER BY group_id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let group_id: i64 = row.get(0).unwrap();
        row_count += 1;

        // LAST returns the last non-NULL value in the group
        match group_id {
            1 => {
                // Group 1: values are 10, 20, 30 - LAST should be 30
                let last_val: i64 = row.get(1).unwrap();
                assert_eq!(last_val, 30, "Group 1 LAST value should be 30");
                let last_text: String = row.get(2).unwrap();
                assert_eq!(last_text, "Cherry", "Group 1 LAST text should be Cherry");
            }
            2 => {
                // Group 2: values are 15, 25, NULL - LAST should be 25
                let last_val: i64 = row.get(1).unwrap();
                assert_eq!(last_val, 25, "Group 2 LAST value should be 25");
                let last_text: String = row.get(2).unwrap();
                assert_eq!(last_text, "Orange", "Group 2 LAST text should be Orange");
            }
            3 => {
                // Group 3: values are NULL, 35, 45 - LAST should be 45
                let last_val: i64 = row.get(1).unwrap();
                assert_eq!(last_val, 45, "Group 3 LAST value should be 45");
                let last_text: String = row.get(2).unwrap();
                assert_eq!(last_text, "Plum", "Group 3 LAST text should be Plum");
            }
            _ => panic!("Unexpected group_id: {}", group_id),
        }
    }

    assert_eq!(row_count, 3, "Expected 3 groups");
}

/// Test FIRST with all NULL values
#[test]
fn test_first_with_all_nulls() {
    let db = Database::open("memory://first_all_nulls").expect("Failed to create database");

    db.execute("CREATE TABLE all_nulls (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO all_nulls (id, value) VALUES (1, NULL), (2, NULL), (3, NULL)",
        (),
    )
    .expect("Failed to insert data");

    // FIRST on all NULLs should return NULL
    let result = db
        .query("SELECT FIRST(value) FROM all_nulls", ())
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        // The result should be NULL, which might be represented as None
        let is_null = row.is_null(0);
        assert!(is_null, "FIRST of all NULLs should be NULL");
        found = true;
    }

    assert!(found, "Expected a result row");
}

/// Test FIRST without GROUP BY
#[test]
fn test_first_without_group_by() {
    let db = Database::open("memory://first_no_group").expect("Failed to create database");

    db.execute("CREATE TABLE simple_first (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO simple_first VALUES (1, 100), (2, 200), (3, 300)",
        (),
    )
    .expect("Failed to insert data");

    // FIRST without GROUP BY returns the first value of the entire result
    let first_val: i64 = db
        .query_one("SELECT FIRST(value) FROM simple_first", ())
        .expect("Failed to query");

    assert_eq!(first_val, 100, "FIRST value should be 100");
}

/// Test LAST without GROUP BY
#[test]
fn test_last_without_group_by() {
    let db = Database::open("memory://last_no_group").expect("Failed to create database");

    db.execute("CREATE TABLE simple_last (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO simple_last VALUES (1, 100), (2, 200), (3, 300)",
        (),
    )
    .expect("Failed to insert data");

    // LAST without GROUP BY returns the last value of the entire result
    let last_val: i64 = db
        .query_one("SELECT LAST(value) FROM simple_last", ())
        .expect("Failed to query");

    assert_eq!(last_val, 300, "LAST value should be 300");
}

/// Test FIRST and LAST with mixed NULL values
#[test]
fn test_first_last_mixed_nulls() {
    let db = Database::open("memory://first_last_mixed").expect("Failed to create database");

    db.execute("CREATE TABLE mixed_nulls (id INTEGER, val INTEGER)", ())
        .expect("Failed to create table");

    // Pattern: NULL, 10, 20, NULL, 30
    db.execute(
        "INSERT INTO mixed_nulls VALUES
        (1, NULL),
        (2, 10),
        (3, 20),
        (4, NULL),
        (5, 30)",
        (),
    )
    .expect("Failed to insert data");

    // FIRST should skip initial NULL and return 10
    let first_val: i64 = db
        .query_one("SELECT FIRST(val) FROM mixed_nulls", ())
        .expect("Failed to query FIRST");

    assert_eq!(first_val, 10, "FIRST should be 10 (first non-NULL)");

    // LAST should skip trailing NULL and return 30
    let last_val: i64 = db
        .query_one("SELECT LAST(val) FROM mixed_nulls", ())
        .expect("Failed to query LAST");

    assert_eq!(last_val, 30, "LAST should be 30 (last non-NULL)");
}
