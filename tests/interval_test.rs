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

/// Test calendar-aware month addition: Jan 31 + 1 month = Feb 28 (not Mar 2/3)
#[test]
fn test_interval_month_calendar_aware() {
    let db = Database::open("memory://interval_month_cal").expect("Failed to create database");

    let ts: String = db
        .query_one(
            "SELECT TIMESTAMP '2025-01-31 10:30:00' + INTERVAL '1 month'",
            (),
        )
        .expect("Failed to execute query");

    // Jan 31 + 1 month should clamp to Feb 28 (2025 is not a leap year)
    assert_eq!(ts, "2025-02-28T10:30:00Z");
}

/// Test calendar-aware month addition on leap year: Jan 31 + 1 month = Feb 29
#[test]
fn test_interval_month_leap_year() {
    let db = Database::open("memory://interval_month_leap").expect("Failed to create database");

    let ts: String = db
        .query_one(
            "SELECT TIMESTAMP '2024-01-31 15:00:00' + INTERVAL '1 month'",
            (),
        )
        .expect("Failed to execute query");

    // 2024 is a leap year, so Jan 31 + 1 month = Feb 29
    assert_eq!(ts, "2024-02-29T15:00:00Z");
}

/// Test calendar-aware year addition preserves month/day
#[test]
fn test_interval_year_calendar_aware() {
    let db = Database::open("memory://interval_year_cal").expect("Failed to create database");

    let ts: String = db
        .query_one(
            "SELECT TIMESTAMP '2024-02-29 12:00:00' + INTERVAL '1 year'",
            (),
        )
        .expect("Failed to execute query");

    // Feb 29 2024 (leap) + 1 year = Feb 28 2025 (non-leap, day clamped)
    assert_eq!(ts, "2025-02-28T12:00:00Z");
}

/// Test subtracting months works correctly
#[test]
fn test_interval_subtract_months() {
    let db = Database::open("memory://interval_sub_month").expect("Failed to create database");

    let ts: String = db
        .query_one(
            "SELECT TIMESTAMP '2025-03-31 08:00:00' - INTERVAL '1 month'",
            (),
        )
        .expect("Failed to execute query");

    // Mar 31 - 1 month = Feb 28 (2025 is not a leap year)
    assert_eq!(ts, "2025-02-28T08:00:00Z");
}

/// Test adding multiple months crosses year boundary correctly
#[test]
fn test_interval_months_cross_year() {
    let db = Database::open("memory://interval_cross_year").expect("Failed to create database");

    let ts: String = db
        .query_one(
            "SELECT TIMESTAMP '2025-11-15 00:00:00' + INTERVAL '3 months'",
            (),
        )
        .expect("Failed to execute query");

    // Nov 15 + 3 months = Feb 15 next year
    assert_eq!(ts, "2026-02-15T00:00:00Z");
}

/// Test that INTERVAL '12 months' equals INTERVAL '1 year'
#[test]
fn test_interval_12_months_equals_1_year() {
    let db = Database::open("memory://interval_12m_1y").expect("Failed to create database");

    let ts_months: String = db
        .query_one(
            "SELECT TIMESTAMP '2025-06-15 12:00:00' + INTERVAL '12 months'",
            (),
        )
        .expect("Failed");

    let ts_year: String = db
        .query_one(
            "SELECT TIMESTAMP '2025-06-15 12:00:00' + INTERVAL '1 year'",
            (),
        )
        .expect("Failed");

    assert_eq!(ts_months, ts_year);
    assert_eq!(ts_months, "2026-06-15T12:00:00Z");
}
