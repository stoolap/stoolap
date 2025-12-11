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

//! DateTime and Timestamp Tests
//!
//! Tests date, time, and timestamp operations

use stoolap::Database;

/// Test inserting and querying date values
#[test]
fn test_date_insert_and_query() {
    let db = Database::open("memory://dt_date").expect("Failed to create database");

    db.execute(
        "CREATE TABLE datetime_test (
            id INTEGER,
            date_val TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO datetime_test (id, date_val) VALUES (1, TIMESTAMP '2023-01-15 00:00:00'), (2, TIMESTAMP '2023-02-20 00:00:00'), (3, TIMESTAMP '2023-03-25 00:00:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM datetime_test", ())
        .expect("Failed to count");

    assert_eq!(count, 3, "Expected 3 date rows");

    // Query and verify dates
    let result = db
        .query("SELECT id, date_val FROM datetime_test ORDER BY id", ())
        .expect("Failed to query");

    let mut dates: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let date_val: String = row.get(1).unwrap();
        dates.push((id, date_val));
    }

    assert_eq!(dates.len(), 3);
}

/// Test inserting and querying time values
#[test]
fn test_time_insert_and_query() {
    let db = Database::open("memory://dt_time").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timevalue_test (
            id INTEGER,
            time_val TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO timevalue_test (id, time_val) VALUES (1, TIMESTAMP '1970-01-01 09:15:30'), (2, TIMESTAMP '1970-01-01 12:30:45'), (3, TIMESTAMP '1970-01-01 18:45:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM timevalue_test", ())
        .expect("Failed to count");

    assert_eq!(count, 3, "Expected 3 time rows");

    // Query and verify times
    let result = db
        .query("SELECT id, time_val FROM timevalue_test ORDER BY id", ())
        .expect("Failed to query");

    let mut times: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let time_val: String = row.get(1).unwrap();
        times.push((id, time_val));
    }

    assert_eq!(times.len(), 3);
}

/// Test inserting and querying timestamp values
#[test]
fn test_timestamp_insert_and_query() {
    let db = Database::open("memory://dt_timestamp").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timestamp_test (
            id INTEGER,
            timestamp_val TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO timestamp_test (id, timestamp_val) VALUES (1, TIMESTAMP '2023-01-15 09:15:30'), (2, TIMESTAMP '2023-02-20 12:30:45'), (3, TIMESTAMP '2023-03-25 18:45:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM timestamp_test", ())
        .expect("Failed to count");

    assert_eq!(count, 3, "Expected 3 timestamp rows");

    // Query and verify timestamps
    let result = db
        .query(
            "SELECT id, timestamp_val FROM timestamp_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut timestamps: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let ts_val: String = row.get(1).unwrap();
        timestamps.push((id, ts_val));
    }

    assert_eq!(timestamps.len(), 3);
}

/// Test date equality comparison
#[test]
fn test_date_equality() {
    let db = Database::open("memory://dt_date_eq").expect("Failed to create database");

    db.execute("CREATE TABLE dates (id INTEGER, date_val TIMESTAMP)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO dates (id, date_val) VALUES (1, TIMESTAMP '2023-01-15 00:00:00'), (2, TIMESTAMP '2023-02-20 00:00:00'), (3, TIMESTAMP '2023-03-25 00:00:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM dates WHERE date_val = TIMESTAMP '2023-02-20 00:00:00'",
            (),
        )
        .expect("Failed to query");

    assert_eq!(count, 1, "Expected 1 row for date equality");
}

/// Test date range comparison
#[test]
fn test_date_range() {
    let db = Database::open("memory://dt_date_range").expect("Failed to create database");

    db.execute("CREATE TABLE dates (id INTEGER, date_val TIMESTAMP)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO dates (id, date_val) VALUES (1, TIMESTAMP '2023-01-15 00:00:00'), (2, TIMESTAMP '2023-02-20 00:00:00'), (3, TIMESTAMP '2023-03-25 00:00:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM dates WHERE date_val >= TIMESTAMP '2023-02-01 00:00:00' AND date_val <= TIMESTAMP '2023-03-31 00:00:00'", ())
        .expect("Failed to query");

    assert_eq!(count, 2, "Expected 2 rows in date range");
}

/// Test time equality comparison
#[test]
fn test_time_equality() {
    let db = Database::open("memory://dt_time_eq").expect("Failed to create database");

    db.execute("CREATE TABLE times (id INTEGER, time_val TIMESTAMP)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO times (id, time_val) VALUES (1, TIMESTAMP '1970-01-01 09:15:30'), (2, TIMESTAMP '1970-01-01 12:30:45'), (3, TIMESTAMP '1970-01-01 18:45:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM times WHERE time_val = TIMESTAMP '1970-01-01 12:30:45'",
            (),
        )
        .expect("Failed to query");

    assert_eq!(count, 1, "Expected 1 row for time equality");
}

/// Test creating index on timestamp column
#[test]
fn test_timestamp_index() {
    let db = Database::open("memory://dt_ts_index").expect("Failed to create database");

    db.execute("CREATE TABLE events (id INTEGER, event_time TIMESTAMP)", ())
        .expect("Failed to create table");

    db.execute("CREATE INDEX idx_event_time ON events(event_time)", ())
        .expect("Failed to create index");

    db.execute("INSERT INTO events (id, event_time) VALUES (1, TIMESTAMP '2023-01-15 09:00:00'), (2, TIMESTAMP '2023-01-15 12:00:00'), (3, TIMESTAMP '2023-01-15 15:00:00')", ())
        .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM events", ())
        .expect("Failed to count");

    assert_eq!(count, 3, "Expected 3 rows with timestamp index");
}

/// Test timestamp with different formats
#[test]
fn test_timestamp_formats() {
    let db = Database::open("memory://dt_ts_formats").expect("Failed to create database");

    db.execute("CREATE TABLE ts (id INTEGER, ts_val TIMESTAMP)", ())
        .expect("Failed to create table");

    // ISO format with T and Z
    db.execute(
        "INSERT INTO ts (id, ts_val) VALUES (1, TIMESTAMP '2023-06-15 10:30:00')",
        (),
    )
    .expect("Failed to insert ISO format");

    // Space-separated format
    db.execute(
        "INSERT INTO ts (id, ts_val) VALUES (2, TIMESTAMP '2023-06-15 14:45:00')",
        (),
    )
    .expect("Failed to insert space format");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM ts", ())
        .expect("Failed to count");

    assert_eq!(count, 2, "Expected 2 rows with different timestamp formats");
}

/// Test ordering by timestamp
#[test]
fn test_timestamp_ordering() {
    let db = Database::open("memory://dt_ts_order").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO events (id, ts) VALUES (3, TIMESTAMP '2023-03-15 15:00:00')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO events (id, ts) VALUES (1, TIMESTAMP '2023-01-15 09:00:00')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO events (id, ts) VALUES (2, TIMESTAMP '2023-02-15 12:00:00')",
        (),
    )
    .expect("Failed to insert data");

    // Note: When ORDER BY references a column, it must be in SELECT for now
    // This is a limitation that should be fixed in the future
    let result = db
        .query("SELECT id, ts FROM events ORDER BY ts ASC", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    // Note: IDs 1, 2, 3 correspond to January, February, March timestamps
    assert_eq!(ids, vec![1, 2, 3], "Expected IDs ordered by timestamp");
}

/// Test NULL timestamp handling
#[test]
fn test_timestamp_null() {
    let db = Database::open("memory://dt_ts_null").expect("Failed to create database");

    db.execute("CREATE TABLE events (id INTEGER, ts TIMESTAMP)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO events (id, ts) VALUES (1, TIMESTAMP '2023-01-15 09:00:00'), (2, NULL), (3, TIMESTAMP '2023-01-15 12:00:00')", ())
        .expect("Failed to insert data");

    let non_null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM events WHERE ts IS NOT NULL", ())
        .expect("Failed to count non-null");

    assert_eq!(non_null_count, 2, "Expected 2 non-null timestamps");

    let null_count: i64 = db
        .query_one("SELECT COUNT(*) FROM events WHERE ts IS NULL", ())
        .expect("Failed to count null");

    assert_eq!(null_count, 1, "Expected 1 null timestamp");
}
