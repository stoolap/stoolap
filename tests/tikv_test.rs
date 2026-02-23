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

//! Integration tests for TiKV storage backend.
//!
//! These tests require a running TiKV cluster. Start one with:
//!
//!   docker compose -f docker-compose.tikv.yml up -d
//!
//! Run tests with:
//!
//!   TIKV_PD_ENDPOINTS=127.0.0.1:2379 cargo test --features tikv --test tikv_test
//!
//! Tests are skipped if TIKV_PD_ENDPOINTS is not set.

#![cfg(feature = "tikv")]

use stoolap::Database;

/// Get TiKV PD endpoints from environment, or return None to skip tests.
fn tikv_dsn() -> Option<String> {
    std::env::var("TIKV_PD_ENDPOINTS")
        .ok()
        .map(|endpoints| format!("tikv://{}", endpoints))
}

/// Helper macro to skip test if TiKV is not available
macro_rules! require_tikv {
    () => {
        match tikv_dsn() {
            Some(dsn) => dsn,
            None => {
                eprintln!("Skipping: TIKV_PD_ENDPOINTS not set");
                return;
            }
        }
    };
}

#[test]
fn test_tikv_connect() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).expect("Failed to connect to TiKV");
    db.close().unwrap();
}

#[test]
fn test_tikv_create_drop_table() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute(
        "CREATE TABLE IF NOT EXISTS tikv_test_create (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_create", ())
        .unwrap();
}

#[test]
fn test_tikv_insert_select() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    // Setup
    db.execute("DROP TABLE IF EXISTS tikv_test_insert", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_insert (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .unwrap();

    // Insert
    db.execute(
        "INSERT INTO tikv_test_insert VALUES (1, 'Alice', 30)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_insert VALUES (2, 'Bob', 25)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_insert VALUES (3, 'Charlie', 35)",
        (),
    )
    .unwrap();

    // Select all
    let mut count = 0;
    for row in db
        .query("SELECT * FROM tikv_test_insert", ())
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 3, "Should have 3 rows");

    // Select with WHERE
    let mut count = 0;
    for row in db
        .query(
            "SELECT name FROM tikv_test_insert WHERE age > 28",
            (),
        )
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 2, "Should have 2 rows with age > 28");

    // Cleanup
    db.execute("DROP TABLE tikv_test_insert", ()).unwrap();
}

#[test]
fn test_tikv_update() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_update", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_update (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_test_update VALUES (1, 100)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_test_update VALUES (2, 200)", ())
        .unwrap();

    // Update
    db.execute(
        "UPDATE tikv_test_update SET value = 999 WHERE id = 1",
        (),
    )
    .unwrap();

    // Verify
    let val: i64 = db
        .query_one(
            "SELECT value FROM tikv_test_update WHERE id = 1",
            (),
        )
        .unwrap();
    assert_eq!(val, 999);

    // Cleanup
    db.execute("DROP TABLE tikv_test_update", ()).unwrap();
}

#[test]
fn test_tikv_delete() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_delete", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_delete (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_test_delete VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_test_delete VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_test_delete VALUES (3, 'Charlie')", ())
        .unwrap();

    // Delete one row
    db.execute("DELETE FROM tikv_test_delete WHERE id = 2", ())
        .unwrap();

    // Verify count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_delete", ())
        .unwrap();
    assert_eq!(count, 2);

    // Cleanup
    db.execute("DROP TABLE tikv_test_delete", ()).unwrap();
}

#[test]
fn test_tikv_transaction() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_txn", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_txn (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    // Transaction commit
    {
        let mut txn = db.begin().unwrap();
        txn.execute("INSERT INTO tikv_test_txn VALUES (1, 100)", ())
            .unwrap();
        txn.execute("INSERT INTO tikv_test_txn VALUES (2, 200)", ())
            .unwrap();
        txn.commit().unwrap();
    }

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_txn", ())
        .unwrap();
    assert_eq!(count, 2);

    // Transaction rollback
    {
        let mut txn = db.begin().unwrap();
        txn.execute("INSERT INTO tikv_test_txn VALUES (3, 300)", ())
            .unwrap();
        txn.rollback().unwrap();
    }

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_txn", ())
        .unwrap();
    assert_eq!(count, 2, "Rolled back insert should not be visible");

    // Cleanup
    db.execute("DROP TABLE tikv_test_txn", ()).unwrap();
}

#[test]
fn test_tikv_aggregation() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_agg", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_agg (id INTEGER PRIMARY KEY, category TEXT, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_test_agg VALUES (1, 'A', 10.5)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_agg VALUES (2, 'B', 20.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_agg VALUES (3, 'A', 30.5)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_agg VALUES (4, 'B', 40.0)",
        (),
    )
    .unwrap();

    // SUM
    let total: f64 = db
        .query_one("SELECT SUM(amount) FROM tikv_test_agg", ())
        .unwrap();
    assert!((total - 101.0).abs() < 0.01);

    // COUNT
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_agg", ())
        .unwrap();
    assert_eq!(count, 4);

    // Cleanup
    db.execute("DROP TABLE tikv_test_agg", ()).unwrap();
}

#[test]
fn test_tikv_multiple_types() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_types", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_types (
            id INTEGER PRIMARY KEY,
            int_val INTEGER,
            float_val FLOAT,
            text_val TEXT,
            bool_val BOOLEAN,
            ts_val TIMESTAMP
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_test_types VALUES (1, 42, 3.14, 'hello', TRUE, '2024-01-15 10:30:00')",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_test_types VALUES (2, NULL, NULL, NULL, NULL, NULL)",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_types", ())
        .unwrap();
    assert_eq!(count, 2);

    // Cleanup
    db.execute("DROP TABLE tikv_test_types", ()).unwrap();
}

#[test]
fn test_tikv_auto_increment() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_autoinc", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_autoinc (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    // Insert without specifying ID (auto-increment)
    db.execute(
        "INSERT INTO tikv_test_autoinc (name) VALUES ('first')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_test_autoinc (name) VALUES ('second')",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_autoinc", ())
        .unwrap();
    assert_eq!(count, 2);

    // Cleanup
    db.execute("DROP TABLE tikv_test_autoinc", ()).unwrap();
}

#[test]
fn test_tikv_views() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();

    db.execute("DROP TABLE IF EXISTS tikv_test_views_t", ())
        .unwrap();
    db.execute(
        "CREATE TABLE tikv_test_views_t (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_test_views_t VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_test_views_t VALUES (2, 20)", ())
        .unwrap();

    // Create view
    db.execute(
        "CREATE VIEW tikv_test_view AS SELECT * FROM tikv_test_views_t WHERE val > 15",
        (),
    )
    .unwrap();

    // Query view
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_test_view", ())
        .unwrap();
    assert_eq!(count, 1);

    // Cleanup
    db.execute("DROP VIEW tikv_test_view", ()).unwrap();
    db.execute("DROP TABLE tikv_test_views_t", ()).unwrap();
}
