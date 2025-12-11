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

//! Time Functions Tests
//!
//! Tests DATE_TRUNC, TIME_TRUNC functions and their usage with GROUP BY

use stoolap::Database;

fn setup_time_test_table(db: &Database) {
    db.execute(
        "CREATE TABLE time_functions_test (
            id INTEGER,
            employee_name TEXT,
            department TEXT,
            event_timestamp TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create test table");

    db.execute(
        "INSERT INTO time_functions_test (id, employee_name, department, event_timestamp) VALUES
        (1, 'Alice', 'Engineering', TIMESTAMP '2021-03-15 09:15:30'),
        (2, 'Bob', 'Engineering', TIMESTAMP '2021-03-15 10:25:45'),
        (3, 'Charlie', 'Marketing', TIMESTAMP '2021-03-15 11:35:15'),
        (4, 'Diana', 'Marketing', TIMESTAMP '2021-05-20 14:45:10'),
        (5, 'Eve', 'Finance', TIMESTAMP '2022-01-10 08:55:20')",
        (),
    )
    .expect("Failed to insert test data");
}

fn setup_sales_table(db: &Database) {
    db.execute(
        "CREATE TABLE time_function_sales (
            id INTEGER,
            product_id INTEGER,
            amount FLOAT,
            transaction_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create sales table");

    db.execute(
        "INSERT INTO time_function_sales (id, product_id, amount, transaction_time) VALUES
        (1, 101, 50.0, TIMESTAMP '2023-01-01 09:15:00'),
        (2, 102, 25.0, TIMESTAMP '2023-01-01 09:30:00'),
        (3, 101, 75.0, TIMESTAMP '2023-01-01 10:15:00'),
        (4, 103, 100.0, TIMESTAMP '2023-01-01 10:45:00'),
        (5, 102, 35.0, TIMESTAMP '2023-01-01 11:15:00'),
        (6, 101, 60.0, TIMESTAMP '2023-01-01 11:45:00')",
        (),
    )
    .expect("Failed to insert sales data");
}

/// Test DATE_TRUNC function with year unit
#[test]
fn test_date_trunc_year() {
    let db = Database::open("memory://time_trunc_year").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT DATE_TRUNC('year', event_timestamp) AS year_trunc
             FROM time_functions_test
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 5, "Expected 5 rows");
    // 2021-03-15 should truncate to 2021-01-01
    assert!(
        results[0].contains("2021-01-01"),
        "Expected first date to be truncated to 2021-01-01, got {}",
        results[0]
    );
}

/// Test DATE_TRUNC function with month unit
#[test]
fn test_date_trunc_month() {
    let db = Database::open("memory://time_trunc_month").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT DATE_TRUNC('month', event_timestamp) AS month_trunc
             FROM time_functions_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        // 2021-03-15 should truncate to 2021-03-01
        assert!(
            truncated.contains("2021-03-01"),
            "Expected date truncated to 2021-03-01, got {}",
            truncated
        );
    }
}

/// Test DATE_TRUNC function with day unit
#[test]
fn test_date_trunc_day() {
    let db = Database::open("memory://time_trunc_day").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT DATE_TRUNC('day', event_timestamp) AS day_trunc
             FROM time_functions_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        // 2021-03-15 09:15:30 should truncate to 2021-03-15 00:00:00
        assert!(
            truncated.contains("2021-03-15"),
            "Expected date truncated to 2021-03-15, got {}",
            truncated
        );
    }
}

/// Test TIME_TRUNC function with 1-hour duration
#[test]
fn test_time_trunc_hour() {
    let db = Database::open("memory://time_trunc_hour").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT TIME_TRUNC('1h', event_timestamp) AS hour_trunc
             FROM time_functions_test
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        results.push(truncated);
    }

    assert_eq!(results.len(), 5, "Expected 5 rows");
    // 09:15:30 should truncate to 09:00:00
    assert!(
        results[0].contains("09:00:00"),
        "Expected first time to be truncated to 09:00:00, got {}",
        results[0]
    );
}

/// Test TIME_TRUNC function with 15-minute duration
#[test]
fn test_time_trunc_minutes() {
    let db = Database::open("memory://time_trunc_min").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT TIME_TRUNC('15m', event_timestamp) AS minute_trunc
             FROM time_functions_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        // 09:15:30 should truncate to 09:15:00
        assert!(
            truncated.contains("09:15:00"),
            "Expected time truncated to 09:15:00, got {}",
            truncated
        );
    }
}

/// Test TIME_TRUNC with GROUP BY - hourly buckets
#[test]
fn test_time_trunc_group_by() {
    let db = Database::open("memory://time_trunc_group").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT
                TIME_TRUNC('1h', transaction_time) AS hour_bucket,
                SUM(amount) AS total_sales
             FROM time_function_sales
             GROUP BY TIME_TRUNC('1h', transaction_time)
             ORDER BY hour_bucket",
            (),
        )
        .expect("Failed to query");

    let mut buckets: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let bucket: String = row.get(0).unwrap();
        let sales: f64 = row.get(1).unwrap();
        buckets.push((bucket, sales));
    }

    // Should have 3 different hour buckets
    assert_eq!(buckets.len(), 3, "Expected 3 hour buckets");

    // Verify the totals per hour
    for (bucket, sales) in &buckets {
        if bucket.contains("09:") {
            // 9:00-10:00 bucket should have 75.0 (50.0 + 25.0)
            assert!(
                (*sales - 75.0).abs() < 0.001,
                "Expected 75.0 for 9:00 hour bucket, got {}",
                sales
            );
        } else if bucket.contains("10:") {
            // 10:00-11:00 bucket should have 175.0 (75.0 + 100.0)
            assert!(
                (*sales - 175.0).abs() < 0.001,
                "Expected 175.0 for 10:00 hour bucket, got {}",
                sales
            );
        } else if bucket.contains("11:") {
            // 11:00-12:00 bucket should have 95.0 (35.0 + 60.0)
            assert!(
                (*sales - 95.0).abs() < 0.001,
                "Expected 95.0 for 11:00 hour bucket, got {}",
                sales
            );
        }
    }
}

/// Test DATE_TRUNC with GROUP BY
#[test]
fn test_date_trunc_group_by() {
    let db = Database::open("memory://date_trunc_group").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT
                DATE_TRUNC('year', event_timestamp) AS year_bucket,
                COUNT(*) AS event_count
             FROM time_functions_test
             GROUP BY DATE_TRUNC('year', event_timestamp)
             ORDER BY year_bucket",
            (),
        )
        .expect("Failed to query");

    let mut buckets: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let bucket: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        buckets.push((bucket, count));
    }

    // Should have 2 year buckets: 2021 (4 events) and 2022 (1 event)
    assert_eq!(buckets.len(), 2, "Expected 2 year buckets");

    for (bucket, count) in &buckets {
        if bucket.contains("2021") {
            assert_eq!(*count, 4, "Expected 4 events in 2021");
        } else if bucket.contains("2022") {
            assert_eq!(*count, 1, "Expected 1 event in 2022");
        }
    }
}

/// Test DATE_TRUNC month with GROUP BY
#[test]
fn test_date_trunc_month_group_by() {
    let db = Database::open("memory://date_trunc_month_group").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT
                DATE_TRUNC('month', event_timestamp) AS month_bucket,
                COUNT(*) AS event_count
             FROM time_functions_test
             GROUP BY DATE_TRUNC('month', event_timestamp)
             ORDER BY month_bucket",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }

    // Should have 3 month buckets: 2021-03 (3), 2021-05 (1), 2022-01 (1)
    assert_eq!(count, 3, "Expected 3 month buckets");
}

/// Test TIME_TRUNC with 30-minute intervals
#[test]
fn test_time_trunc_30min() {
    let db = Database::open("memory://time_trunc_30min").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT TIME_TRUNC('30m', transaction_time) AS half_hour_bucket
             FROM time_function_sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut results: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let bucket: String = row.get(0).unwrap();
        results.push(bucket);
    }

    assert_eq!(results.len(), 6, "Expected 6 rows");
    // 09:15:00 should truncate to 09:00:00
    assert!(
        results[0].contains("09:00:00"),
        "Expected 09:15 truncated to 09:00, got {}",
        results[0]
    );
    // 09:30:00 should truncate to 09:30:00
    assert!(
        results[1].contains("09:30:00"),
        "Expected 09:30 to stay at 09:30, got {}",
        results[1]
    );
}

/// Test DATE_TRUNC with week unit
#[test]
fn test_date_trunc_week() {
    let db = Database::open("memory://time_trunc_week").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT DATE_TRUNC('week', event_timestamp) AS week_trunc
             FROM time_functions_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        // 2021-03-15 (Monday) should truncate to 2021-03-15 (week start)
        assert!(
            truncated.contains("2021-03-15") || truncated.contains("2021-03-14"),
            "Expected week start, got {}",
            truncated
        );
    }
}

/// Test DATE_TRUNC with quarter unit
#[test]
fn test_date_trunc_quarter() {
    let db = Database::open("memory://time_trunc_quarter").expect("Failed to create database");
    setup_time_test_table(&db);

    let result = db
        .query(
            "SELECT DATE_TRUNC('quarter', event_timestamp) AS quarter_trunc
             FROM time_functions_test
             WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let truncated: String = row.get(0).unwrap();
        // 2021-03-15 (Q1) should truncate to 2021-01-01
        assert!(
            truncated.contains("2021-01-01"),
            "Expected quarter start 2021-01-01, got {}",
            truncated
        );
    }
}
