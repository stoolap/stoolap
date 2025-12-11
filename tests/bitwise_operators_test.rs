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

//! Bitwise Operators Tests
//!
//! Tests for bitwise operators: &, |, ^, ~, <<, >>

use stoolap::Database;

// ============================================================================
// Basic Bitwise AND (&)
// ============================================================================

#[test]
fn test_bitwise_and_basic() {
    let db = Database::open("memory://bitwise_and_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 5 & 3", ()).expect("Failed to query");
    assert_eq!(result, 1, "5 & 3 should be 1 (0101 & 0011 = 0001)");
}

#[test]
fn test_bitwise_and_zero() {
    let db = Database::open("memory://bitwise_and_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 255 & 0", ()).expect("Failed to query");
    assert_eq!(result, 0, "255 & 0 should be 0");
}

#[test]
fn test_bitwise_and_same() {
    let db = Database::open("memory://bitwise_and_same").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 & 42", ()).expect("Failed to query");
    assert_eq!(result, 42, "42 & 42 should be 42");
}

// ============================================================================
// Basic Bitwise OR (|)
// ============================================================================

#[test]
fn test_bitwise_or_basic() {
    let db = Database::open("memory://bitwise_or_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 5 | 3", ()).expect("Failed to query");
    assert_eq!(result, 7, "5 | 3 should be 7 (0101 | 0011 = 0111)");
}

#[test]
fn test_bitwise_or_zero() {
    let db = Database::open("memory://bitwise_or_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 | 0", ()).expect("Failed to query");
    assert_eq!(result, 42, "42 | 0 should be 42");
}

#[test]
fn test_bitwise_or_all_ones() {
    let db = Database::open("memory://bitwise_or_all").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 0 | 255", ()).expect("Failed to query");
    assert_eq!(result, 255, "0 | 255 should be 255");
}

// ============================================================================
// Basic Bitwise XOR (^)
// ============================================================================

#[test]
fn test_bitwise_xor_basic() {
    let db = Database::open("memory://bitwise_xor_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 5 ^ 3", ()).expect("Failed to query");
    assert_eq!(result, 6, "5 ^ 3 should be 6 (0101 ^ 0011 = 0110)");
}

#[test]
fn test_bitwise_xor_same() {
    let db = Database::open("memory://bitwise_xor_same").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 ^ 42", ()).expect("Failed to query");
    assert_eq!(result, 0, "42 ^ 42 should be 0 (XOR with self is 0)");
}

#[test]
fn test_bitwise_xor_zero() {
    let db = Database::open("memory://bitwise_xor_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 ^ 0", ()).expect("Failed to query");
    assert_eq!(result, 42, "42 ^ 0 should be 42");
}

// ============================================================================
// Basic Bitwise NOT (~)
// ============================================================================

#[test]
fn test_bitwise_not_basic() {
    let db = Database::open("memory://bitwise_not_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT ~5", ()).expect("Failed to query");
    assert_eq!(result, -6, "~5 should be -6 (two's complement)");
}

#[test]
fn test_bitwise_not_zero() {
    let db = Database::open("memory://bitwise_not_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT ~0", ()).expect("Failed to query");
    assert_eq!(result, -1, "~0 should be -1");
}

#[test]
fn test_bitwise_not_negative() {
    let db = Database::open("memory://bitwise_not_neg").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT ~(-1)", ()).expect("Failed to query");
    assert_eq!(result, 0, "~(-1) should be 0");
}

// ============================================================================
// Left Shift (<<)
// ============================================================================

#[test]
fn test_left_shift_basic() {
    let db = Database::open("memory://left_shift_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 5 << 2", ()).expect("Failed to query");
    assert_eq!(result, 20, "5 << 2 should be 20 (5 * 4)");
}

#[test]
fn test_left_shift_zero() {
    let db = Database::open("memory://left_shift_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 << 0", ()).expect("Failed to query");
    assert_eq!(result, 42, "42 << 0 should be 42");
}

#[test]
fn test_left_shift_one() {
    let db = Database::open("memory://left_shift_one").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 1 << 10", ()).expect("Failed to query");
    assert_eq!(result, 1024, "1 << 10 should be 1024");
}

// ============================================================================
// Right Shift (>>)
// ============================================================================

#[test]
fn test_right_shift_basic() {
    let db = Database::open("memory://right_shift_basic").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 20 >> 2", ()).expect("Failed to query");
    assert_eq!(result, 5, "20 >> 2 should be 5 (20 / 4)");
}

#[test]
fn test_right_shift_zero() {
    let db = Database::open("memory://right_shift_zero").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 42 >> 0", ()).expect("Failed to query");
    assert_eq!(result, 42, "42 >> 0 should be 42");
}

#[test]
fn test_right_shift_large() {
    let db = Database::open("memory://right_shift_large").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT 1024 >> 10", ())
        .expect("Failed to query");
    assert_eq!(result, 1, "1024 >> 10 should be 1");
}

// ============================================================================
// NULL Handling
// ============================================================================

#[test]
fn test_bitwise_and_null() {
    let db = Database::open("memory://bitwise_and_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT 5 & NULL", ())
        .expect("Failed to query");
    assert!(result.is_none(), "5 & NULL should be NULL");
}

#[test]
fn test_bitwise_or_null() {
    let db = Database::open("memory://bitwise_or_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT NULL | 3", ())
        .expect("Failed to query");
    assert!(result.is_none(), "NULL | 3 should be NULL");
}

#[test]
fn test_bitwise_xor_null() {
    let db = Database::open("memory://bitwise_xor_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT 5 ^ NULL", ())
        .expect("Failed to query");
    assert!(result.is_none(), "5 ^ NULL should be NULL");
}

#[test]
fn test_bitwise_not_null() {
    let db = Database::open("memory://bitwise_not_null").expect("Failed to create database");

    let result: Option<i64> = db.query_one("SELECT ~NULL", ()).expect("Failed to query");
    assert!(result.is_none(), "~NULL should be NULL");
}

#[test]
fn test_left_shift_null() {
    let db = Database::open("memory://left_shift_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT 5 << NULL", ())
        .expect("Failed to query");
    assert!(result.is_none(), "5 << NULL should be NULL");
}

#[test]
fn test_right_shift_null() {
    let db = Database::open("memory://right_shift_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT NULL >> 2", ())
        .expect("Failed to query");
    assert!(result.is_none(), "NULL >> 2 should be NULL");
}

// ============================================================================
// Complex Expressions
// ============================================================================

#[test]
fn test_bitwise_complex_expression() {
    let db = Database::open("memory://bitwise_complex").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT (5 & 3) | 8", ())
        .expect("Failed to query");
    assert_eq!(result, 9, "(5 & 3) | 8 should be 9 (1 | 8 = 9)");
}

#[test]
fn test_bitwise_mask_extraction() {
    let db = Database::open("memory://bitwise_mask").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT 255 & (1 << 4)", ())
        .expect("Failed to query");
    assert_eq!(result, 16, "255 & (1 << 4) should be 16");
}

#[test]
fn test_bitwise_toggle() {
    let db = Database::open("memory://bitwise_toggle").expect("Failed to create database");

    // Toggle bit 2 (value 4) in 5: 5 ^ 4 = 1
    let result: i64 = db.query_one("SELECT 5 ^ 4", ()).expect("Failed to query");
    assert_eq!(result, 1, "5 ^ 4 should be 1 (toggle bit 2)");
}

// ============================================================================
// With Table Data
// ============================================================================

#[test]
fn test_bitwise_with_table() {
    let db = Database::open("memory://bitwise_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE flags (id INTEGER PRIMARY KEY, permissions INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO flags VALUES (1, 7)", ()) // rwx = 111
        .expect("Failed to insert");
    db.execute("INSERT INTO flags VALUES (2, 5)", ()) // r-x = 101
        .expect("Failed to insert");
    db.execute("INSERT INTO flags VALUES (3, 4)", ()) // r-- = 100
        .expect("Failed to insert");

    // Check read permission (bit 2, value 4)
    let result = db
        .query("SELECT id, permissions & 4 FROM flags ORDER BY id", ())
        .expect("Failed to query");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let perm: i64 = row.get(1).expect("Failed to get perm");
        rows.push((id, perm));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (1, 4), "Row 1 should have read permission");
    assert_eq!(rows[1], (2, 4), "Row 2 should have read permission");
    assert_eq!(rows[2], (3, 4), "Row 3 should have read permission");
}

#[test]
fn test_bitwise_where_clause() {
    let db = Database::open("memory://bitwise_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE flags (id INTEGER PRIMARY KEY, permissions INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO flags VALUES (1, 7)", ()) // rwx
        .expect("Failed to insert");
    db.execute("INSERT INTO flags VALUES (2, 5)", ()) // r-x
        .expect("Failed to insert");
    db.execute("INSERT INTO flags VALUES (3, 4)", ()) // r--
        .expect("Failed to insert");

    // Find rows with write permission (bit 1, value 2)
    let result = db
        .query(
            "SELECT id FROM flags WHERE (permissions & 2) = 2 ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        ids.push(id);
    }

    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 1, "Only row 1 should have write permission");
}

#[test]
fn test_bitwise_update() {
    let db = Database::open("memory://bitwise_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE flags (id INTEGER PRIMARY KEY, permissions INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO flags VALUES (1, 4)", ()) // r--
        .expect("Failed to insert");

    // Add write permission using OR
    db.execute(
        "UPDATE flags SET permissions = permissions | 2 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT permissions FROM flags WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(
        result, 6,
        "Permissions should be 6 (r-w) after adding write"
    );
}

#[test]
fn test_bitwise_remove_flag() {
    let db = Database::open("memory://bitwise_remove").expect("Failed to create database");

    db.execute(
        "CREATE TABLE flags (id INTEGER PRIMARY KEY, permissions INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO flags VALUES (1, 7)", ()) // rwx
        .expect("Failed to insert");

    // Remove execute permission using AND with NOT
    db.execute(
        "UPDATE flags SET permissions = permissions & ~1 WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    let result: i64 = db
        .query_one("SELECT permissions FROM flags WHERE id = 1", ())
        .expect("Failed to query");

    assert_eq!(
        result, 6,
        "Permissions should be 6 (rw-) after removing execute"
    );
}

// ============================================================================
// Operator Precedence
// ============================================================================

#[test]
fn test_bitwise_precedence_shift_before_and() {
    let db = Database::open("memory://bitwise_prec1").expect("Failed to create database");

    // << should have higher precedence than &
    // 1 << 4 & 255 should be (1 << 4) & 255 = 16 & 255 = 16
    let result: i64 = db
        .query_one("SELECT 1 << 4 & 255", ())
        .expect("Failed to query");
    assert_eq!(result, 16, "1 << 4 & 255 should be 16");
}

#[test]
fn test_bitwise_precedence_and_before_or() {
    let db = Database::open("memory://bitwise_prec2").expect("Failed to create database");

    // & should have higher precedence than |
    // 1 | 2 & 3 should be 1 | (2 & 3) = 1 | 2 = 3
    let result: i64 = db
        .query_one("SELECT 1 | 2 & 3", ())
        .expect("Failed to query");
    assert_eq!(result, 3, "1 | 2 & 3 should be 3");
}

#[test]
fn test_bitwise_precedence_xor_between() {
    let db = Database::open("memory://bitwise_prec3").expect("Failed to create database");

    // ^ should be between & and |
    // 1 | 2 ^ 3 & 7 should be 1 | (2 ^ (3 & 7)) = 1 | (2 ^ 3) = 1 | 1 = 1
    let result: i64 = db
        .query_one("SELECT 1 | 2 ^ 3 & 7", ())
        .expect("Failed to query");
    assert_eq!(result, 1, "1 | 2 ^ 3 & 7 should be 1");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_bitwise_negative_numbers() {
    let db = Database::open("memory://bitwise_negative").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT -1 & 255", ())
        .expect("Failed to query");
    assert_eq!(result, 255, "-1 & 255 should be 255");
}

#[test]
fn test_bitwise_large_numbers() {
    let db = Database::open("memory://bitwise_large").expect("Failed to create database");

    let result: i64 = db.query_one("SELECT 1 << 62", ()).expect("Failed to query");
    assert_eq!(result, 4611686018427387904i64, "1 << 62 should work");
}
