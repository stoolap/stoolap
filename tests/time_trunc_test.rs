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

//! Time Truncation Function Tests
//!
//! Tests DATE_TRUNC and TIME_TRUNC functions

use stoolap::Database;

/// Test DATE_TRUNC function with year unit
#[test]
fn test_date_trunc_year() {
    let db = Database::open("memory://date_trunc_year").expect("Failed to create database");

    db.execute(
        "CREATE TABLE time_test (
            id INTEGER PRIMARY KEY,
            event_timestamp TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO time_test (id, event_timestamp) VALUES
         (1, TIMESTAMP '2021-03-15 09:15:30'),
         (2, TIMESTAMP '2021-07-20 14:30:00'),
         (3, TIMESTAMP '2022-01-10 08:55:20')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT DATE_TRUNC('year', event_timestamp) AS year_trunc FROM time_test ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3, "Expected 3 rows");
    // First two should be 2021, third should be 2022
    assert!(
        results[0].contains("2021-01-01"),
        "Expected year truncation to 2021-01-01, got {}",
        results[0]
    );
    assert!(
        results[1].contains("2021-01-01"),
        "Expected year truncation to 2021-01-01, got {}",
        results[1]
    );
    assert!(
        results[2].contains("2022-01-01"),
        "Expected year truncation to 2022-01-01, got {}",
        results[2]
    );
}

/// Test DATE_TRUNC function with month unit
#[test]
fn test_date_trunc_month() {
    let db = Database::open("memory://date_trunc_month").expect("Failed to create database");

    db.execute(
        "CREATE TABLE time_test (
            id INTEGER PRIMARY KEY,
            event_timestamp TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO time_test (id, event_timestamp) VALUES
         (1, TIMESTAMP '2021-03-15 09:15:30'),
         (2, TIMESTAMP '2021-03-20 14:30:00'),
         (3, TIMESTAMP '2021-05-10 08:55:20')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT DATE_TRUNC('month', event_timestamp) AS month_trunc FROM time_test ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3, "Expected 3 rows");
    assert!(
        results[0].contains("2021-03-01"),
        "Expected month truncation to 2021-03-01, got {}",
        results[0]
    );
    assert!(
        results[1].contains("2021-03-01"),
        "Expected month truncation to 2021-03-01, got {}",
        results[1]
    );
    assert!(
        results[2].contains("2021-05-01"),
        "Expected month truncation to 2021-05-01, got {}",
        results[2]
    );
}

/// Test TIME_TRUNC function with 1-hour interval
#[test]
fn test_time_trunc_hour() {
    let db = Database::open("memory://time_trunc_hour").expect("Failed to create database");

    db.execute(
        "CREATE TABLE time_test (
            id INTEGER PRIMARY KEY,
            event_timestamp TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO time_test (id, event_timestamp) VALUES
         (1, TIMESTAMP '2021-03-15 09:15:30'),
         (2, TIMESTAMP '2021-03-15 09:45:00'),
         (3, TIMESTAMP '2021-03-15 10:30:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT TIME_TRUNC('1h', event_timestamp) AS hour_trunc FROM time_test ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3, "Expected 3 rows");
    // 09:15:30 and 09:45:00 should truncate to 09:00:00
    assert!(
        results[0].contains("09:00:00"),
        "Expected hour truncation to 09:00:00, got {}",
        results[0]
    );
    assert!(
        results[1].contains("09:00:00"),
        "Expected hour truncation to 09:00:00, got {}",
        results[1]
    );
    // 10:30:00 should truncate to 10:00:00
    assert!(
        results[2].contains("10:00:00"),
        "Expected hour truncation to 10:00:00, got {}",
        results[2]
    );
}

/// Test TIME_TRUNC function with 15-minute interval
#[test]
fn test_time_trunc_15_minutes() {
    let db = Database::open("memory://time_trunc_15m").expect("Failed to create database");

    db.execute(
        "CREATE TABLE time_test (
            id INTEGER PRIMARY KEY,
            event_timestamp TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO time_test (id, event_timestamp) VALUES
         (1, TIMESTAMP '2021-03-15 09:15:30'),
         (2, TIMESTAMP '2021-03-15 09:25:00'),
         (3, TIMESTAMP '2021-03-15 09:35:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT TIME_TRUNC('15m', event_timestamp) AS min_trunc FROM time_test ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3, "Expected 3 rows");
    // 09:15:30 should truncate to 09:15:00
    assert!(
        results[0].contains("09:15:00"),
        "Expected 15m truncation to 09:15:00, got {}",
        results[0]
    );
    // 09:25:00 should truncate to 09:15:00
    assert!(
        results[1].contains("09:15:00"),
        "Expected 15m truncation to 09:15:00, got {}",
        results[1]
    );
    // 09:35:00 should truncate to 09:30:00
    assert!(
        results[2].contains("09:30:00"),
        "Expected 15m truncation to 09:30:00, got {}",
        results[2]
    );
}

/// Test TIME_TRUNC with GROUP BY
#[test]
fn test_time_trunc_group_by() {
    let db = Database::open("memory://time_trunc_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            amount FLOAT,
            transaction_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales (id, amount, transaction_time) VALUES
         (1, 50.0, TIMESTAMP '2023-01-01 09:15:00'),
         (2, 25.0, TIMESTAMP '2023-01-01 09:30:00'),
         (3, 75.0, TIMESTAMP '2023-01-01 10:15:00'),
         (4, 100.0, TIMESTAMP '2023-01-01 10:45:00'),
         (5, 35.0, TIMESTAMP '2023-01-01 11:15:00'),
         (6, 60.0, TIMESTAMP '2023-01-01 11:45:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT TIME_TRUNC('1h', transaction_time) AS hour_bucket, SUM(amount) AS total_sales
             FROM sales
             GROUP BY TIME_TRUNC('1h', transaction_time)
             ORDER BY hour_bucket",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let bucket: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        results.push((bucket, total));
    }

    assert_eq!(results.len(), 3, "Expected 3 hour buckets");

    // Verify totals per hour
    // 9:00-10:00: 50 + 25 = 75
    assert!(
        results[0].0.contains("09:00"),
        "First bucket should be 09:00"
    );
    assert!(
        (results[0].1 - 75.0).abs() < 0.01,
        "9:00 hour should have 75.0 total, got {}",
        results[0].1
    );

    // 10:00-11:00: 75 + 100 = 175
    assert!(
        results[1].0.contains("10:00"),
        "Second bucket should be 10:00"
    );
    assert!(
        (results[1].1 - 175.0).abs() < 0.01,
        "10:00 hour should have 175.0 total, got {}",
        results[1].1
    );

    // 11:00-12:00: 35 + 60 = 95
    assert!(
        results[2].0.contains("11:00"),
        "Third bucket should be 11:00"
    );
    assert!(
        (results[2].1 - 95.0).abs() < 0.01,
        "11:00 hour should have 95.0 total, got {}",
        results[2].1
    );
}

/// Test DATE_TRUNC with day unit
#[test]
fn test_date_trunc_day() {
    let db = Database::open("memory://date_trunc_day").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO events (id, event_time) VALUES
         (1, TIMESTAMP '2023-06-15 09:30:00'),
         (2, TIMESTAMP '2023-06-15 14:45:00'),
         (3, TIMESTAMP '2023-06-16 10:00:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT DATE_TRUNC('day', event_time) AS day_trunc FROM events ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3);
    assert!(
        results[0].contains("2023-06-15 00:00:00") || results[0].contains("2023-06-15T00:00:00")
    );
    assert!(
        results[1].contains("2023-06-15 00:00:00") || results[1].contains("2023-06-15T00:00:00")
    );
    assert!(
        results[2].contains("2023-06-16 00:00:00") || results[2].contains("2023-06-16T00:00:00")
    );
}

/// Test TIME_TRUNC with 5-minute interval
#[test]
fn test_time_trunc_5_minutes() {
    let db = Database::open("memory://time_trunc_5m").expect("Failed to create database");

    db.execute(
        "CREATE TABLE metrics (
            id INTEGER PRIMARY KEY,
            recorded_at TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO metrics (id, recorded_at) VALUES
         (1, TIMESTAMP '2023-01-01 10:03:30'),
         (2, TIMESTAMP '2023-01-01 10:07:00'),
         (3, TIMESTAMP '2023-01-01 10:11:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT TIME_TRUNC('5m', recorded_at) AS bucket FROM metrics ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3);
    // 10:03:30 -> 10:00:00
    assert!(
        results[0].contains("10:00:00"),
        "10:03:30 should truncate to 10:00:00, got {}",
        results[0]
    );
    // 10:07:00 -> 10:05:00
    assert!(
        results[1].contains("10:05:00"),
        "10:07:00 should truncate to 10:05:00, got {}",
        results[1]
    );
    // 10:11:00 -> 10:10:00
    assert!(
        results[2].contains("10:10:00"),
        "10:11:00 should truncate to 10:10:00, got {}",
        results[2]
    );
}

/// Test TIME_TRUNC with 30-minute interval
#[test]
fn test_time_trunc_30_minutes() {
    let db = Database::open("memory://time_trunc_30m").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO events (id, event_time) VALUES
         (1, TIMESTAMP '2023-01-01 10:10:00'),
         (2, TIMESTAMP '2023-01-01 10:40:00'),
         (3, TIMESTAMP '2023-01-01 11:20:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT TIME_TRUNC('30m', event_time) AS bucket FROM events ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3);
    // 10:10:00 -> 10:00:00
    assert!(
        results[0].contains("10:00:00"),
        "10:10:00 should truncate to 10:00:00, got {}",
        results[0]
    );
    // 10:40:00 -> 10:30:00
    assert!(
        results[1].contains("10:30:00"),
        "10:40:00 should truncate to 10:30:00, got {}",
        results[1]
    );
    // 11:20:00 -> 11:00:00
    assert!(
        results[2].contains("11:00:00"),
        "11:20:00 should truncate to 11:00:00, got {}",
        results[2]
    );
}

/// Test DATE_TRUNC with week unit
#[test]
fn test_date_trunc_week() {
    let db = Database::open("memory://date_trunc_week").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    // 2023-06-15 is a Thursday
    // 2023-06-19 is a Monday (start of next week)
    db.execute(
        "INSERT INTO events (id, event_time) VALUES
         (1, TIMESTAMP '2023-06-15 10:00:00'),
         (2, TIMESTAMP '2023-06-17 14:00:00'),
         (3, TIMESTAMP '2023-06-19 09:00:00')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT DATE_TRUNC('week', event_time) AS week_trunc FROM events ORDER BY id",
            (),
        )
        .expect("Failed to execute query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 3);
    // First two should be in the same week, third in the next week
    // Week starts on Monday (2023-06-12)
    assert!(
        results[0].contains("2023-06-12"),
        "First date should truncate to week starting 2023-06-12"
    );
    assert!(
        results[1].contains("2023-06-12"),
        "Second date should truncate to week starting 2023-06-12"
    );
    assert!(
        results[2].contains("2023-06-19"),
        "Third date should truncate to week starting 2023-06-19"
    );
}

/// Test combined DATE_TRUNC and TIME_TRUNC in same query
#[test]
fn test_combined_truncations() {
    let db = Database::open("memory://combined_trunc").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO events (id, event_time) VALUES
         (1, TIMESTAMP '2023-06-15 10:23:45')",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT
             DATE_TRUNC('month', event_time) AS month_trunc,
             TIME_TRUNC('1h', event_time) AS hour_trunc
             FROM events WHERE id = 1",
            (),
        )
        .expect("Failed to execute query");

    for row in result {
        let row = row.expect("Failed to get row");
        let month_trunc: String = row.get(0).unwrap();
        let hour_trunc: String = row.get(1).unwrap();

        assert!(
            month_trunc.contains("2023-06-01"),
            "Month truncation should be 2023-06-01, got {}",
            month_trunc
        );
        assert!(
            hour_trunc.contains("10:00:00"),
            "Hour truncation should contain 10:00:00, got {}",
            hour_trunc
        );
    }
}
