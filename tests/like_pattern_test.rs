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

//! LIKE Pattern Matching Tests
//!
//! Tests SQL LIKE pattern matching with %, _, and combinations

use stoolap::Database;

fn setup_like_table(db: &Database) {
    db.execute(
        "CREATE TABLE fruits (
            id INTEGER PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data - various fruit names
    db.execute("INSERT INTO fruits (id, name) VALUES (1, 'apple')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (2, 'banana')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (3, 'pineapple')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (4, 'grape')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (5, 'grapefruit')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (6, 'orange')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (7, 'strawberry')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (8, 'blueberry')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (9, 'blackberry')", ())
        .unwrap();
    db.execute("INSERT INTO fruits (id, name) VALUES (10, 'cranberry')", ())
        .unwrap();
}

/// Test exact match (no wildcards)
#[test]
fn test_like_exact_match() {
    let db = Database::open("memory://like_exact").expect("Failed to create database");
    setup_like_table(&db);

    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE 'apple' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![1], "Expected only 'apple' (id=1)");
}

/// Test contains match (%)
#[test]
fn test_like_contains() {
    let db = Database::open("memory://like_contains").expect("Failed to create database");
    setup_like_table(&db);

    // Match all berries
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE '%berry%' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![7, 8, 9, 10], "Expected all berries");
}

/// Test starts with pattern
#[test]
fn test_like_starts_with() {
    let db = Database::open("memory://like_starts").expect("Failed to create database");
    setup_like_table(&db);

    // Match grape and grapefruit
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE 'grape%' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![4, 5], "Expected grape and grapefruit");
}

/// Test ends with pattern
#[test]
fn test_like_ends_with() {
    let db = Database::open("memory://like_ends").expect("Failed to create database");
    setup_like_table(&db);

    // Match apple and pineapple
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE '%apple' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![1, 3], "Expected apple and pineapple");
}

/// Test single character wildcard (_)
#[test]
fn test_like_single_char() {
    let db = Database::open("memory://like_single").expect("Failed to create database");
    setup_like_table(&db);

    // Match orange (6 chars with _range pattern)
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE '_range' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![6], "Expected orange");
}

/// Test match any (%)
#[test]
fn test_like_match_any() {
    let db = Database::open("memory://like_any").expect("Failed to create database");
    setup_like_table(&db);

    // Match everything
    let result = db
        .query("SELECT id FROM fruits WHERE name LIKE '%' ORDER BY id", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(
        ids,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Expected all fruits"
    );
}

/// Test no match
#[test]
fn test_like_no_match() {
    let db = Database::open("memory://like_none").expect("Failed to create database");
    setup_like_table(&db);

    // No mangos in our data
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE 'mango' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0, "Expected no matches for mango");
}

/// Test start and end pattern
#[test]
fn test_like_start_end() {
    let db = Database::open("memory://like_start_end").expect("Failed to create database");
    setup_like_table(&db);

    // Match blueberry and blackberry (start with 'b', end with 'y')
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name LIKE 'b%y' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![8, 9], "Expected blueberry and blackberry");
}

/// Test NOT LIKE
#[test]
fn test_not_like() {
    let db = Database::open("memory://not_like").expect("Failed to create database");
    setup_like_table(&db);

    // Match everything NOT ending in 'berry'
    let result = db
        .query(
            "SELECT id FROM fruits WHERE name NOT LIKE '%berry' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Should exclude: strawberry(7), blueberry(8), blackberry(9), cranberry(10)
    assert_eq!(ids, vec![1, 2, 3, 4, 5, 6], "Expected non-berry fruits");
}

/// Test ILIKE (case-insensitive) - uses LOWER() workaround since ILIKE may not be supported
#[test]
fn test_ilike_case_insensitive() {
    let db = Database::open("memory://ilike_case").expect("Failed to create database");

    db.execute("CREATE TABLE mixed_case (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO mixed_case VALUES (1, 'Apple')", ())
        .unwrap();
    db.execute("INSERT INTO mixed_case VALUES (2, 'BANANA')", ())
        .unwrap();
    db.execute("INSERT INTO mixed_case VALUES (3, 'orange')", ())
        .unwrap();

    // Case-insensitive match using LOWER() function
    let result = db
        .query(
            "SELECT id FROM mixed_case WHERE LOWER(name) LIKE 'apple' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(
        ids,
        vec![1],
        "Expected 'Apple' to match 'apple' case-insensitively"
    );
}

/// Test LIKE with NULL
#[test]
fn test_like_with_null() {
    let db = Database::open("memory://like_null").expect("Failed to create database");

    db.execute("CREATE TABLE nullable (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO nullable VALUES (1, 'apple')", ())
        .unwrap();
    db.execute("INSERT INTO nullable VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO nullable VALUES (3, 'banana')", ())
        .unwrap();

    // LIKE on NULL should not match
    let result = db
        .query(
            "SELECT id FROM nullable WHERE name LIKE '%a%' ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // NULL should not match any pattern
    assert_eq!(ids, vec![1, 3], "NULL should not match LIKE pattern");
}
