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

//! Interval Tests
//!
//! Tests INTERVAL support for date/time arithmetic

use stoolap::core::Value;
use stoolap::Database;

/// Test NOW() minus 24 hours
#[test]
fn test_interval_now_minus_24_hours() {
    let db = Database::open("memory://interval_24h").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() - INTERVAL '24 hours'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = now.signed_duration_since(timestamp);

            // Check that the result is approximately 24 hours ago (within 1 hour tolerance)
            assert!(
                diff.num_hours() >= 23 && diff.num_hours() <= 25,
                "Expected timestamp ~24 hours ago, got {} hours ago",
                diff.num_hours()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test NOW() plus 1 day
#[test]
fn test_interval_now_plus_1_day() {
    let db = Database::open("memory://interval_1d").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() + INTERVAL '1 day'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = timestamp.signed_duration_since(now);

            // Check that the result is approximately 1 day in the future
            assert!(
                diff.num_hours() >= 23 && diff.num_hours() <= 25,
                "Expected timestamp ~24 hours in future, got {} hours",
                diff.num_hours()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test NOW() minus 30 minutes
#[test]
fn test_interval_now_minus_30_minutes() {
    let db = Database::open("memory://interval_30m").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() - INTERVAL '30 minutes'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = now.signed_duration_since(timestamp);

            // Check that the result is approximately 30 minutes ago
            assert!(
                diff.num_minutes() >= 29 && diff.num_minutes() <= 31,
                "Expected timestamp ~30 minutes ago, got {} minutes ago",
                diff.num_minutes()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test interval with seconds
#[test]
fn test_interval_with_seconds() {
    let db = Database::open("memory://interval_seconds").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() + INTERVAL '90 seconds'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = timestamp.signed_duration_since(now);

            // Check that the result is approximately 90 seconds in the future
            assert!(
                diff.num_seconds() >= 89 && diff.num_seconds() <= 91,
                "Expected timestamp ~90 seconds in future, got {} seconds",
                diff.num_seconds()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test interval with weeks
#[test]
fn test_interval_with_weeks() {
    let db = Database::open("memory://interval_weeks").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() - INTERVAL '2 weeks'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = now.signed_duration_since(timestamp);

            // Check that the result is approximately 14 days ago
            assert!(
                diff.num_days() >= 13 && diff.num_days() <= 15,
                "Expected timestamp ~14 days ago, got {} days ago",
                diff.num_days()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test interval with timestamp literal
#[test]
fn test_interval_with_timestamp_literal() {
    let db = Database::open("memory://interval_literal").expect("Failed to create database");

    let result = db
        .query(
            "SELECT TIMESTAMP '2025-01-01 12:00:00' + INTERVAL '25 hours'",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            // Expected: 2025-01-02 13:00:00 UTC
            let expected = chrono::DateTime::parse_from_rfc3339("2025-01-02T13:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc);

            assert_eq!(
                timestamp, expected,
                "Expected {}, got {}",
                expected, timestamp
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

/// Test interval in WHERE clause
#[test]
fn test_interval_in_where_clause() {
    let db = Database::open("memory://interval_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            name TEXT,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert events at different times
    db.execute("INSERT INTO events VALUES (1, 'Recent Event', NOW())", ())
        .expect("Failed to insert");
    db.execute(
        "INSERT INTO events VALUES (2, 'Old Event', TIMESTAMP '2020-01-01 00:00:00')",
        (),
    )
    .expect("Failed to insert");

    // Query for events in the last week
    let result = db
        .query(
            "SELECT * FROM events WHERE event_time > NOW() - INTERVAL '7 days'",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(1).unwrap();
        assert_eq!(name, "Recent Event", "Expected only recent events");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 recent event");
}

/// Test interval with hours
#[test]
fn test_interval_with_hours() {
    let db = Database::open("memory://interval_hours").expect("Failed to create database");

    let result = db
        .query("SELECT NOW() + INTERVAL '5 hours'", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: Value = row.get(0).unwrap();

        if let Value::Timestamp(timestamp) = value {
            let now = chrono::Utc::now();
            let diff = timestamp.signed_duration_since(now);

            // Check that the result is approximately 5 hours in the future
            assert!(
                diff.num_hours() >= 4 && diff.num_hours() <= 6,
                "Expected timestamp ~5 hours in future, got {} hours",
                diff.num_hours()
            );
        } else {
            panic!("Expected Timestamp value, got {:?}", value);
        }
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}
