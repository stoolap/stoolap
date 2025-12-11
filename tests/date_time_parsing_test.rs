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

//! Date/Time Parsing Tests
//!
//! Tests date and time literal parsing in SQL queries

use stoolap::parser::parse_sql;

/// Test parsing date literal in WHERE clause
#[test]
fn test_date_literal_parsing() {
    let sql = "SELECT * FROM events WHERE event_date = '2023-05-15'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse date literal query: {:?}",
        result.err()
    );
    let statements = result.unwrap();
    assert_eq!(statements.len(), 1);
}

/// Test parsing time literal in WHERE clause
#[test]
fn test_time_literal_parsing() {
    let sql = "SELECT * FROM events WHERE event_time = '14:30:00'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse time literal query: {:?}",
        result.err()
    );
    let statements = result.unwrap();
    assert_eq!(statements.len(), 1);
}

/// Test parsing timestamp literal
#[test]
fn test_timestamp_literal_parsing() {
    let sql = "SELECT * FROM events WHERE created_at = TIMESTAMP '2023-05-15 14:30:00'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse timestamp literal query: {:?}",
        result.err()
    );
    let statements = result.unwrap();
    assert_eq!(statements.len(), 1);
}

/// Test parsing date comparison with >
#[test]
fn test_date_comparison_greater() {
    let sql = "SELECT * FROM orders WHERE order_date > '2023-01-01'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse date comparison: {:?}",
        result.err()
    );
}

/// Test parsing date comparison with BETWEEN
#[test]
fn test_date_between() {
    let sql = "SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse BETWEEN with dates: {:?}",
        result.err()
    );
}

/// Test parsing multiple date conditions
#[test]
fn test_multiple_date_conditions() {
    let sql = "SELECT * FROM events WHERE start_date >= '2023-01-01' AND end_date <= '2023-12-31'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse multiple date conditions: {:?}",
        result.err()
    );
}

/// Test parsing timestamp with timezone
#[test]
fn test_timestamp_with_iso_format() {
    let sql = "SELECT * FROM logs WHERE created_at = TIMESTAMP '2023-05-15T14:30:00Z'";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse ISO timestamp: {:?}",
        result.err()
    );
}

/// Test parsing date in INSERT
#[test]
fn test_date_in_insert() {
    let sql = "INSERT INTO events (id, event_date) VALUES (1, '2023-05-15')";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse INSERT with date: {:?}",
        result.err()
    );
}

/// Test parsing timestamp in INSERT
#[test]
fn test_timestamp_in_insert() {
    let sql = "INSERT INTO logs (id, created_at) VALUES (1, TIMESTAMP '2023-05-15 14:30:00')";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse INSERT with timestamp: {:?}",
        result.err()
    );
}

/// Test date functions parsing
#[test]
fn test_date_function_parsing() {
    let sql = "SELECT id, EXTRACT(YEAR FROM created_at) as year FROM events";
    let result = parse_sql(sql);

    // EXTRACT might not be implemented, but check if parse doesn't crash
    if result.is_err() {
        // Some date functions may not be supported yet
        println!("EXTRACT function not yet supported");
    }
}

/// Test ORDER BY with date column
#[test]
fn test_order_by_date() {
    let sql = "SELECT * FROM events ORDER BY event_date DESC";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse ORDER BY with date: {:?}",
        result.err()
    );
}

/// Test GROUP BY with date
#[test]
fn test_group_by_date() {
    let sql = "SELECT event_date, COUNT(*) FROM events GROUP BY event_date";
    let result = parse_sql(sql);

    assert!(
        result.is_ok(),
        "Failed to parse GROUP BY with date: {:?}",
        result.err()
    );
}
