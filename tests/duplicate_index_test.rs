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

//! Duplicate Index Tests
//!
//! Tests behavior when creating duplicate indexes on same column

use stoolap::Database;

/// Test creating unique index first, then trying non-unique
#[test]
fn test_unique_index_first() {
    let db = Database::open("memory://dup_idx_unique_first").expect("Failed to create database");

    db.execute("CREATE TABLE duplicate_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // First create a unique index
    db.execute(
        "CREATE UNIQUE INDEX idx_name_unique ON duplicate_test(name)",
        (),
    )
    .expect("Failed to create unique index");

    // Now try to create a non-unique index on the same column - should fail
    let result = db.execute("CREATE INDEX idx_name ON duplicate_test(name)", ());

    assert!(
        result.is_err(),
        "Expected error when creating duplicate index"
    );

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("cannot create non-unique index")
            || err_msg.contains("already exists")
            || err_msg.contains("index"),
        "Unexpected error message: {}",
        err_msg
    );
}

/// Test creating non-unique index first, then trying unique
#[test]
fn test_non_unique_index_first() {
    let db = Database::open("memory://dup_idx_nonunique_first").expect("Failed to create database");

    db.execute("CREATE TABLE duplicate_test2 (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // First create a non-unique index
    db.execute("CREATE INDEX idx_name2 ON duplicate_test2(name)", ())
        .expect("Failed to create non-unique index");

    // Now try to create a unique index on the same column - should fail
    let result = db.execute(
        "CREATE UNIQUE INDEX idx_name2_unique ON duplicate_test2(name)",
        (),
    );

    assert!(
        result.is_err(),
        "Expected error when creating duplicate index"
    );

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("cannot create unique index")
            || err_msg.contains("already exists")
            || err_msg.contains("index"),
        "Unexpected error message: {}",
        err_msg
    );
}

/// Test IF NOT EXISTS clause with duplicate index
#[test]
fn test_if_not_exists() {
    let db = Database::open("memory://dup_idx_if_not_exists").expect("Failed to create database");

    db.execute("CREATE TABLE duplicate_test3 (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Create a unique index
    db.execute("CREATE UNIQUE INDEX idx_name3 ON duplicate_test3(name)", ())
        .expect("Failed to create unique index");

    // Try to create a non-unique index with IF NOT EXISTS - should succeed without error
    let result = db.execute(
        "CREATE INDEX IF NOT EXISTS idx_name3_nonunique ON duplicate_test3(name)",
        (),
    );

    // IF NOT EXISTS should not return an error
    assert!(
        result.is_ok(),
        "IF NOT EXISTS should succeed: {:?}",
        result.err()
    );
}

/// Test creating same index name twice
#[test]
fn test_same_index_name() {
    let db = Database::open("memory://dup_idx_same_name").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_table (id INTEGER, col1 TEXT, col2 TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Create an index
    db.execute("CREATE INDEX idx_test ON test_table(col1)", ())
        .expect("Failed to create first index");

    // Try to create another index with same name on different column
    let result = db.execute("CREATE INDEX idx_test ON test_table(col2)", ());

    assert!(
        result.is_err(),
        "Expected error when creating index with same name"
    );
}

/// Test creating index on different columns works
#[test]
fn test_different_columns() {
    let db = Database::open("memory://dup_idx_diff_cols").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_col (id INTEGER, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Create index on name
    db.execute("CREATE INDEX idx_name ON multi_col(name)", ())
        .expect("Failed to create index on name");

    // Create index on value - should succeed
    db.execute("CREATE INDEX idx_value ON multi_col(value)", ())
        .expect("Failed to create index on value");

    // Verify both indexes exist by using them
    db.execute(
        "INSERT INTO multi_col (id, name, value) VALUES (1, 'test', 100)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM multi_col WHERE name = 'test'", ())
        .expect("Failed to query by name");
    assert_eq!(count, 1);

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM multi_col WHERE value = 100", ())
        .expect("Failed to query by value");
    assert_eq!(count, 1);
}
