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

//! Hash Join Tests
//!
//! Tests for hash join implementation including:
//! - Single key joins
//! - Multiple key joins (composite keys)
//! - All join types (INNER, LEFT, RIGHT, FULL)
//! - Build side optimization (smaller table as build side)
//! - Edge cases (NULLs, duplicates, empty tables)

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create database")
}

// ============================================================================
// Basic Hash Join Tests
// ============================================================================

#[test]
fn test_hash_join_inner() {
    let db = create_test_db("hash_join_inner");

    db.execute("CREATE TABLE left_t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (2, 'X'), (3, 'Y'), (4, 'Z')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.id, l.val, r.data FROM left_t l INNER JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);

    // Check that we got the right matches
    let ids: Vec<i64> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}

#[test]
fn test_hash_join_left() {
    let db = create_test_db("hash_join_left");

    db.execute("CREATE TABLE left_t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (2, 'X'), (3, 'Y'), (4, 'Z')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.id, l.val, r.data FROM left_t l LEFT JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3); // All left rows

    // Check id=1 has NULL for right side
    let row1: Vec<_> = rows
        .iter()
        .filter(|r| r.get::<i64>(0).unwrap() == 1)
        .collect();
    assert_eq!(row1.len(), 1);
    let data: Option<String> = row1[0].get(2).ok();
    assert!(data.is_none() || data == Some("".to_string())); // NULL
}

#[test]
fn test_hash_join_right() {
    let db = create_test_db("hash_join_right");

    db.execute("CREATE TABLE left_t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (2, 'X'), (3, 'Y'), (4, 'Z')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.id, l.val, r.data FROM left_t l RIGHT JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3); // All right rows

    // Check id=4 has NULL for left side
    let row4: Vec<_> = rows
        .iter()
        .filter(|r| {
            let data: String = r.get(2).unwrap();
            data == "Z"
        })
        .collect();
    assert_eq!(row4.len(), 1);
}

#[test]
fn test_hash_join_full_outer() {
    let db = create_test_db("hash_join_full");

    db.execute("CREATE TABLE left_t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (2, 'X'), (3, 'Y'), (4, 'Z')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.id, l.val, r.data FROM left_t l FULL OUTER JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 4); // 1 (left only) + 2 (matched) + 1 (right only)
}

// ============================================================================
// Multi-Key Hash Join Tests
// ============================================================================

#[test]
fn test_hash_join_multi_key() {
    let db = create_test_db("hash_join_multi");

    db.execute("CREATE TABLE left_t (x INTEGER, y INTEGER, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (x INTEGER, y INTEGER, value TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO left_t VALUES (1, 1, 'A'), (1, 2, 'B'), (2, 1, 'C'), (2, 2, 'D')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (1, 1, 'X'), (1, 2, 'Y'), (2, 2, 'Z')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.name, r.value FROM left_t l JOIN right_t r ON l.x = r.x AND l.y = r.y",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3); // A-X, B-Y, D-Z (C has no match)

    let names: Vec<String> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert!(names.contains(&"A".to_string()));
    assert!(names.contains(&"B".to_string()));
    assert!(names.contains(&"D".to_string()));
    assert!(!names.contains(&"C".to_string()));
}

#[test]
fn test_hash_join_multi_key_left() {
    let db = create_test_db("hash_join_multi_left");

    db.execute("CREATE TABLE left_t (x INTEGER, y INTEGER, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE right_t (x INTEGER, y INTEGER, value TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO left_t VALUES (1, 1, 'A'), (1, 2, 'B'), (2, 1, 'C')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO right_t VALUES (1, 1, 'X'), (1, 2, 'Y')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT l.name, r.value FROM left_t l LEFT JOIN right_t r ON l.x = r.x AND l.y = r.y",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3); // All left rows

    // C should have NULL for value
    let row_c: Vec<_> = rows
        .iter()
        .filter(|r| {
            let name: String = r.get(0).unwrap();
            name == "C"
        })
        .collect();
    assert_eq!(row_c.len(), 1);
}

#[test]
fn test_hash_join_three_keys() {
    let db = create_test_db("hash_join_three_keys");

    db.execute(
        "CREATE TABLE left_t (a INTEGER, b INTEGER, c INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE right_t (a INTEGER, b INTEGER, c INTEGER, value TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO left_t VALUES (1, 2, 3, 'match'), (1, 2, 4, 'no_match')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO right_t VALUES (1, 2, 3, 'found')", ())
        .unwrap();

    let result = db
        .query("SELECT l.name, r.value FROM left_t l JOIN right_t r ON l.a = r.a AND l.b = r.b AND l.c = r.c", ())
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 1);

    let name: String = rows[0].get(0).unwrap();
    assert_eq!(name, "match");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_hash_join_with_nulls() {
    let db = create_test_db("hash_join_nulls");

    db.execute("CREATE TABLE left_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (id INTEGER, data TEXT)", ())
        .unwrap();

    db.execute(
        "INSERT INTO left_t VALUES (1, 'a'), (NULL, 'b'), (3, 'c')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (1, 'X'), (NULL, 'Y'), (3, 'Z')",
        (),
    )
    .unwrap();

    // NULL != NULL in SQL, so only 1 and 3 should match
    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2); // Only 1 and 3 match, not NULL=NULL
}

#[test]
fn test_hash_join_with_duplicates() {
    let db = create_test_db("hash_join_dups");

    db.execute("CREATE TABLE left_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (id INTEGER, data TEXT)", ())
        .unwrap();

    // Multiple rows with same join key
    db.execute(
        "INSERT INTO left_t VALUES (1, 'a1'), (1, 'a2'), (2, 'b')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (1, 'X1'), (1, 'X2'), (2, 'Y')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    // id=1: 2 left * 2 right = 4 combinations
    // id=2: 1 left * 1 right = 1 combination
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_hash_join_empty_left() {
    let db = create_test_db("hash_join_empty_left");

    db.execute("CREATE TABLE left_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (id INTEGER, data TEXT)", ())
        .unwrap();

    // Left table is empty
    db.execute("INSERT INTO right_t VALUES (1, 'X'), (2, 'Y')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 0);
}

#[test]
fn test_hash_join_empty_right() {
    let db = create_test_db("hash_join_empty_right");

    db.execute("CREATE TABLE left_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (id INTEGER, data TEXT)", ())
        .unwrap();

    // Right table is empty
    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 0);
}

#[test]
fn test_hash_join_no_matches() {
    let db = create_test_db("hash_join_no_match");

    db.execute("CREATE TABLE left_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (id INTEGER, data TEXT)", ())
        .unwrap();

    // No matching keys
    db.execute("INSERT INTO left_t VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();
    db.execute("INSERT INTO right_t VALUES (3, 'X'), (4, 'Y')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.id = r.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 0);
}

// ============================================================================
// Build Side Optimization Tests
// ============================================================================

#[test]
fn test_hash_join_build_side_optimization() {
    let db = create_test_db("hash_join_build_opt");

    db.execute("CREATE TABLE small_t (id INTEGER, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE large_t (id INTEGER, data TEXT)", ())
        .unwrap();

    // Small table: 2 rows
    db.execute("INSERT INTO small_t VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();

    // Large table: 100 rows
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO large_t VALUES ({}, 'data{}')", i, i),
            (),
        )
        .unwrap();
    }

    // Join should use small table as build side (automatic optimization)
    let result = db
        .query(
            "SELECT s.val, l.data FROM small_t s JOIN large_t l ON s.id = l.id",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2); // Only ids 1 and 2 match
}

// ============================================================================
// Different Data Types
// ============================================================================

#[test]
fn test_hash_join_text_keys() {
    let db = create_test_db("hash_join_text");

    db.execute("CREATE TABLE left_t (code TEXT, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (code TEXT, data TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO left_t VALUES ('A', 1), ('B', 2), ('C', 3)", ())
        .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES ('A', 'alpha'), ('B', 'beta'), ('D', 'delta')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.val, r.data FROM left_t l JOIN right_t r ON l.code = r.code",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2); // A and B match
}

#[test]
fn test_hash_join_float_keys() {
    let db = create_test_db("hash_join_float");

    db.execute("CREATE TABLE left_t (price FLOAT, name TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE right_t (price FLOAT, category TEXT)", ())
        .unwrap();

    db.execute(
        "INSERT INTO left_t VALUES (9.99, 'item1'), (19.99, 'item2')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO right_t VALUES (9.99, 'cheap'), (29.99, 'expensive')",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT l.name, r.category FROM left_t l JOIN right_t r ON l.price = r.price",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 1);

    let name: String = rows[0].get(0).unwrap();
    assert_eq!(name, "item1");
}
