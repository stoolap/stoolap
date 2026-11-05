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

//! Bitmap Index SQL Tests
//!
//! Tests Bitmap INDEX functionality through SQL interface.
//! Bitmap indexes are auto-selected for BOOLEAN columns.
//! They excel at low-cardinality columns with fast AND/OR operations.

use stoolap::Database;

/// Test bitmap index on BOOLEAN column (auto-selected)
#[test]
fn test_bitmap_index_on_boolean_column() {
    let db = Database::open("memory://bitmap_idx_test").expect("Failed to create database");

    // Create table with BOOLEAN column
    db.execute(
        "CREATE TABLE bitmap_test (
            id INTEGER PRIMARY KEY,
            active BOOLEAN,
            verified BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    db.execute("INSERT INTO bitmap_test VALUES (1, true, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_test VALUES (2, true, false)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_test VALUES (3, false, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_test VALUES (4, false, false)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_test VALUES (5, true, true)", ())
        .unwrap();

    // Create index on BOOLEAN column (should auto-select Bitmap index)
    db.execute("CREATE INDEX idx_active ON bitmap_test (active)", ())
        .expect("Failed to create bitmap index on active");

    // Test equality lookup for true
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = true", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with active = true");

    // Test equality lookup for false
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM bitmap_test WHERE active = false", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with active = false");
}

/// Test bitmap index with updates
#[test]
fn test_bitmap_index_updates() {
    let db = Database::open("memory://bitmap_idx_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE bitmap_update_test (
            id INTEGER PRIMARY KEY,
            enabled BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO bitmap_update_test VALUES (1, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_update_test VALUES (2, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_update_test VALUES (3, false)", ())
        .unwrap();

    // Create bitmap index
    db.execute(
        "CREATE INDEX idx_enabled ON bitmap_update_test (enabled)",
        (),
    )
    .expect("Failed to create index");

    // Verify initial counts
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_update_test WHERE enabled = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 enabled rows initially");

    // Update a row
    db.execute(
        "UPDATE bitmap_update_test SET enabled = false WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    // Verify counts after update
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_update_test WHERE enabled = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 enabled row after update");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_update_test WHERE enabled = false",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 disabled rows after update");
}

/// Test bitmap index with deletes
#[test]
fn test_bitmap_index_deletes() {
    let db = Database::open("memory://bitmap_idx_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE bitmap_delete_test (
            id INTEGER PRIMARY KEY,
            premium BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO bitmap_delete_test VALUES (1, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_delete_test VALUES (2, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_delete_test VALUES (3, false)", ())
        .unwrap();

    // Create bitmap index
    db.execute(
        "CREATE INDEX idx_premium ON bitmap_delete_test (premium)",
        (),
    )
    .expect("Failed to create index");

    // Verify initial count
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_delete_test WHERE premium = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 premium rows initially");

    // Delete a row
    db.execute("DELETE FROM bitmap_delete_test WHERE id = 1", ())
        .expect("Failed to delete");

    // Verify count after delete
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_delete_test WHERE premium = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 premium row after delete");
}

/// Test bitmap index with NULL values
#[test]
fn test_bitmap_index_nulls() {
    let db = Database::open("memory://bitmap_idx_nulls").expect("Failed to create database");

    db.execute(
        "CREATE TABLE bitmap_null_test (
            id INTEGER PRIMARY KEY,
            subscribed BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO bitmap_null_test VALUES (1, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_null_test VALUES (2, false)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_null_test VALUES (3, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_null_test VALUES (4, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_null_test VALUES (5, NULL)", ())
        .unwrap();

    // Create bitmap index
    db.execute(
        "CREATE INDEX idx_subscribed ON bitmap_null_test (subscribed)",
        (),
    )
    .expect("Failed to create index");

    // Test IS NULL with index
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_null_test WHERE subscribed IS NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with NULL subscribed");

    // Test IS NOT NULL with index
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_null_test WHERE subscribed IS NOT NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with non-NULL subscribed");

    // Test equality
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_null_test WHERE subscribed = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows with subscribed = true");
}

/// Test multiple bitmap indexes on same table
#[test]
fn test_multiple_bitmap_indexes() {
    let db = Database::open("memory://multi_bitmap").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_bitmap_idx (
            id INTEGER PRIMARY KEY,
            active BOOLEAN,
            verified BOOLEAN,
            premium BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO multi_bitmap_idx VALUES (1, true, true, false)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_bitmap_idx VALUES (2, true, false, true)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_bitmap_idx VALUES (3, false, true, true)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO multi_bitmap_idx VALUES (4, true, true, true)",
        (),
    )
    .unwrap();

    // Create multiple bitmap indexes
    db.execute("CREATE INDEX idx_active ON multi_bitmap_idx (active)", ())
        .expect("Failed to create active index");
    db.execute(
        "CREATE INDEX idx_verified ON multi_bitmap_idx (verified)",
        (),
    )
    .expect("Failed to create verified index");
    db.execute("CREATE INDEX idx_premium ON multi_bitmap_idx (premium)", ())
        .expect("Failed to create premium index");

    // Test queries using different indexes
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_bitmap_idx WHERE active = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 active rows");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_bitmap_idx WHERE verified = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 verified rows");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_bitmap_idx WHERE premium = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 premium rows");

    // Test combined conditions (bitmap AND operations)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_bitmap_idx WHERE active = true AND verified = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows for active AND verified");

    // Test triple condition
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM multi_bitmap_idx WHERE active = true AND verified = true AND premium = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row for all true");
}

/// Test bitmap index with OR conditions
#[test]
fn test_bitmap_index_or_conditions() {
    let db = Database::open("memory://bitmap_or_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE bitmap_or_test (
            id INTEGER PRIMARY KEY,
            status_a BOOLEAN,
            status_b BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO bitmap_or_test VALUES (1, true, false)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_or_test VALUES (2, false, true)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_or_test VALUES (3, false, false)", ())
        .unwrap();
    db.execute("INSERT INTO bitmap_or_test VALUES (4, true, true)", ())
        .unwrap();

    // Create bitmap indexes
    db.execute("CREATE INDEX idx_status_a ON bitmap_or_test (status_a)", ())
        .expect("Failed to create index");
    db.execute("CREATE INDEX idx_status_b ON bitmap_or_test (status_b)", ())
        .expect("Failed to create index");

    // Test OR condition (bitmap OR operations)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_or_test WHERE status_a = true OR status_b = true",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 rows with status_a OR status_b = true");

    // Test NOT condition
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM bitmap_or_test WHERE status_a = false AND status_b = false",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row with both false");
}
