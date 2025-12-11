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

//! DATE/TIMESTAMP Formatting Tests
//!
//! Tests TIMESTAMP column formatting and parsing

use stoolap::Database;

/// Test TIMESTAMP column formatting
#[test]
fn test_timestamp_formatting() {
    let db = Database::open("memory://timestamp_format").expect("Failed to create database");

    // Create a simple test table with timestamp column
    db.execute(
        "CREATE TABLE timestamp_test (
            id INTEGER,
            ts_val TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data using SQL with TIMESTAMP literal
    db.execute(
        "INSERT INTO timestamp_test (id, ts_val) VALUES
        (1, TIMESTAMP '2023-01-15 12:30:45')",
        (),
    )
    .expect("Failed to insert data");

    // Verify the row count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM timestamp_test", ())
        .expect("Failed to get count");
    assert_eq!(count, 1, "Table should have 1 row");

    // Query and check the timestamp value - cast to TEXT to get string representation
    let result = db
        .query("SELECT id, CAST(ts_val AS TEXT) FROM timestamp_test", ())
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let ts_val: String = row.get(1).unwrap();

        assert_eq!(id, 1);
        // Timestamp should contain the date part
        assert!(
            ts_val.contains("2023-01-15"),
            "Expected timestamp containing '2023-01-15', got '{}'",
            ts_val
        );
    }
}

/// Test TIMESTAMP column with various values
#[test]
fn test_timestamp_various_values() {
    let db = Database::open("memory://timestamp_various").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timestamps (
            id INTEGER,
            ts_val TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert different timestamp values
    db.execute(
        "INSERT INTO timestamps (id, ts_val) VALUES (1, TIMESTAMP '2023-01-01 00:00:00')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO timestamps (id, ts_val) VALUES (2, TIMESTAMP '2023-12-31 23:59:59')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO timestamps (id, ts_val) VALUES (3, TIMESTAMP '2020-02-29 12:00:00')",
        (),
    )
    .unwrap(); // Leap year
    db.execute(
        "INSERT INTO timestamps (id, ts_val) VALUES (4, TIMESTAMP '1999-12-31 00:00:00')",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM timestamps", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 rows");

    // Query by timestamp
    let result = db
        .query(
            "SELECT id FROM timestamps WHERE ts_val = TIMESTAMP '2020-02-29 12:00:00'",
            (),
        )
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        assert_eq!(id, 3);
        found = true;
    }
    assert!(found, "Expected to find leap year timestamp");
}

/// Test TIMESTAMP comparison
#[test]
fn test_timestamp_comparison() {
    let db = Database::open("memory://timestamp_compare").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER,
            event_time TIMESTAMP,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO events (id, event_time, name) VALUES (1, TIMESTAMP '2023-01-01 08:00:00', 'New Year')", ()).unwrap();
    db.execute("INSERT INTO events (id, event_time, name) VALUES (2, TIMESTAMP '2023-06-15 12:00:00', 'Mid Year')", ()).unwrap();
    db.execute("INSERT INTO events (id, event_time, name) VALUES (3, TIMESTAMP '2023-12-25 09:00:00', 'Christmas')", ()).unwrap();

    // Find events after mid-year
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM events WHERE event_time > TIMESTAMP '2023-06-01 00:00:00'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 events after June 1st");

    // Find events before mid-year
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM events WHERE event_time < TIMESTAMP '2023-06-01 00:00:00'",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 event before June 1st");
}

/// Test TIMESTAMP with NULL values
#[test]
fn test_timestamp_null() {
    let db = Database::open("memory://timestamp_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timed_items (
            id INTEGER,
            created_at TIMESTAMP,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO timed_items (id, created_at, name) VALUES (1, TIMESTAMP '2023-01-01 00:00:00', 'With Timestamp')", ()).unwrap();
    db.execute(
        "INSERT INTO timed_items (id, created_at, name) VALUES (2, NULL, 'Without Timestamp')",
        (),
    )
    .unwrap();

    // Count NULL timestamps
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM timed_items WHERE created_at IS NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 NULL timestamp");

    // Count non-NULL timestamps
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM timed_items WHERE created_at IS NOT NULL",
            (),
        )
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 non-NULL timestamp");
}

/// Test TIMESTAMP ORDER BY
#[test]
fn test_timestamp_order_by() {
    let db = Database::open("memory://timestamp_orderby").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timeline (
            id INTEGER PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO timeline (id, event_time) VALUES (1, TIMESTAMP '2023-03-15 10:00:00')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO timeline (id, event_time) VALUES (2, TIMESTAMP '2023-01-01 08:00:00')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO timeline (id, event_time) VALUES (3, TIMESTAMP '2023-06-30 16:00:00')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO timeline (id, event_time) VALUES (4, TIMESTAMP '2023-02-28 14:00:00')",
        (),
    )
    .unwrap();

    // Verify we have the data
    let count: i64 = db.query_one("SELECT COUNT(*) FROM timeline", ()).unwrap();
    assert_eq!(count, 4);

    // Order by timestamp ascending
    let result = db
        .query(
            "SELECT id, CAST(event_time AS TEXT) FROM timeline ORDER BY event_time ASC",
            (),
        )
        .expect("Failed to query");

    let mut entries: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let ts: String = row.get(1).unwrap();
        entries.push((id, ts));
    }

    // Print for debugging
    for (id, ts) in &entries {
        eprintln!("ID: {}, Timestamp: {}", id, ts);
    }

    // Expected order: Jan 1 (id=2), Feb 28 (id=4), Mar 15 (id=1), Jun 30 (id=3)
    let ids: Vec<i64> = entries.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        vec![2, 4, 1, 3],
        "Timestamps should be ordered ascending"
    );
}
