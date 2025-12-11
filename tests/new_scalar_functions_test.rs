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

//! Tests for new scalar functions: LOCATE, TO_CHAR, EXTRACT DOW/DOY

use stoolap::Database;

// ============================================================================
// LOCATE Function Tests
// ============================================================================

#[test]
fn test_locate_basic() {
    let db = Database::open("memory://locate_basic").expect("Failed to create database");

    // Basic usage
    let result: i64 = db
        .query_one("SELECT LOCATE('o', 'hello world')", ())
        .expect("Failed to execute LOCATE");
    assert_eq!(result, 5); // First 'o' is at position 5

    // Not found
    let result: i64 = db
        .query_one("SELECT LOCATE('x', 'hello world')", ())
        .expect("Failed to execute LOCATE");
    assert_eq!(result, 0);

    // Empty substring
    let result: i64 = db
        .query_one("SELECT LOCATE('', 'hello')", ())
        .expect("Failed to execute LOCATE");
    assert_eq!(result, 1); // Empty string is found at position 1
}

#[test]
fn test_locate_with_start_position() {
    let db = Database::open("memory://locate_start").expect("Failed to create database");

    // Find second 'o' by starting after the first one
    let result: i64 = db
        .query_one("SELECT LOCATE('o', 'hello world', 6)", ())
        .expect("Failed to execute LOCATE with start");
    assert_eq!(result, 8); // Second 'o' is at position 8

    // Start position beyond string length
    let result: i64 = db
        .query_one("SELECT LOCATE('o', 'hello', 10)", ())
        .expect("Failed to execute LOCATE with large start");
    assert_eq!(result, 0);

    // Start position at 1 (default behavior)
    let result: i64 = db
        .query_one("SELECT LOCATE('l', 'hello', 1)", ())
        .expect("Failed to execute LOCATE from 1");
    assert_eq!(result, 3);

    // Start position at exact match position
    let result: i64 = db
        .query_one("SELECT LOCATE('l', 'hello', 3)", ())
        .expect("Failed to execute LOCATE from 3");
    assert_eq!(result, 3);

    // Start after match
    let result: i64 = db
        .query_one("SELECT LOCATE('l', 'hello', 5)", ())
        .expect("Failed to execute LOCATE from 5");
    assert_eq!(result, 0);
}

#[test]
fn test_locate_null_handling() {
    let db = Database::open("memory://locate_null").expect("Failed to create database");

    // NULL substring
    let result = db
        .query("SELECT LOCATE(NULL, 'hello')", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));

    // NULL string
    let result = db
        .query("SELECT LOCATE('o', NULL)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));

    // NULL start position
    let result = db
        .query("SELECT LOCATE('o', 'hello', NULL)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));
}

#[test]
fn test_locate_case_sensitive() {
    let db = Database::open("memory://locate_case").expect("Failed to create database");

    // Case sensitive search
    let result: i64 = db
        .query_one("SELECT LOCATE('H', 'Hello')", ())
        .expect("Failed to execute LOCATE");
    assert_eq!(result, 1);

    let result: i64 = db
        .query_one("SELECT LOCATE('h', 'Hello')", ())
        .expect("Failed to execute LOCATE");
    assert_eq!(result, 0); // lowercase 'h' not found
}

// ============================================================================
// TO_CHAR Function Tests
// ============================================================================

#[test]
fn test_to_char_date_formats() {
    let db = Database::open("memory://to_char_date").expect("Failed to create database");

    // Basic date format
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15 14:30:45', 'YYYY-MM-DD')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "2024-03-15");

    // Full date with month name
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15', 'DD MON YYYY')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "15 MAR 2024");

    // Time format
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15 14:30:45', 'HH24:MI:SS')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "14:30:45");

    // 12-hour format with AM/PM
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15 14:30:45', 'HH12:MI AM')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "02:30 PM");
}

#[test]
fn test_to_char_day_names() {
    let db = Database::open("memory://to_char_day").expect("Failed to create database");

    // Day of week - Friday, March 15, 2024
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15', 'DAY')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "FRIDAY");

    // Abbreviated day name
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15', 'DY')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "FRI");

    // Full month name
    let result: String = db
        .query_one("SELECT TO_CHAR('2024-03-15', 'MONTH')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "MARCH");
}

#[test]
fn test_to_char_number_formats() {
    let db = Database::open("memory://to_char_num").expect("Failed to create database");

    // Number with decimal places
    let result: String = db
        .query_one("SELECT TO_CHAR(12345.678, '999999.99')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "12345.68");

    // Number with thousands separator
    let result: String = db
        .query_one("SELECT TO_CHAR(1234567, '9,999,999')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "1,234,567");

    // Integer
    let result: String = db
        .query_one("SELECT TO_CHAR(42, '999')", ())
        .expect("Failed to execute TO_CHAR");
    assert_eq!(result, "42");
}

#[test]
fn test_to_char_null_handling() {
    let db = Database::open("memory://to_char_null").expect("Failed to create database");

    // NULL value
    let result = db
        .query("SELECT TO_CHAR(NULL, 'YYYY-MM-DD')", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));

    // NULL format
    let result = db
        .query("SELECT TO_CHAR('2024-03-15', NULL)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));
}

// ============================================================================
// EXTRACT DOW/DOY Tests
// ============================================================================

#[test]
fn test_extract_dow() {
    let db = Database::open("memory://extract_dow").expect("Failed to create database");

    // Friday, March 15, 2024 - DOW should be 5 (Sunday=0)
    let result: i64 = db
        .query_one("SELECT EXTRACT(DOW FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT DOW");
    assert_eq!(result, 5);

    // Sunday should be 0
    let result: i64 = db
        .query_one("SELECT EXTRACT(DOW FROM '2024-03-17')", ())
        .expect("Failed to execute EXTRACT DOW");
    assert_eq!(result, 0);

    // DAYOFWEEK alias
    let result: i64 = db
        .query_one("SELECT EXTRACT(DAYOFWEEK FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT DAYOFWEEK");
    assert_eq!(result, 5);
}

#[test]
fn test_extract_isodow() {
    let db = Database::open("memory://extract_isodow").expect("Failed to create database");

    // Friday, March 15, 2024 - ISODOW should be 5 (Monday=1, Sunday=7)
    let result: i64 = db
        .query_one("SELECT EXTRACT(ISODOW FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT ISODOW");
    assert_eq!(result, 5);

    // Sunday should be 7
    let result: i64 = db
        .query_one("SELECT EXTRACT(ISODOW FROM '2024-03-17')", ())
        .expect("Failed to execute EXTRACT ISODOW");
    assert_eq!(result, 7);
}

#[test]
fn test_extract_doy() {
    let db = Database::open("memory://extract_doy").expect("Failed to create database");

    // March 15, 2024 is day 75 of the year (2024 is a leap year)
    let result: i64 = db
        .query_one("SELECT EXTRACT(DOY FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT DOY");
    assert_eq!(result, 75);

    // January 1 should be day 1
    let result: i64 = db
        .query_one("SELECT EXTRACT(DOY FROM '2024-01-01')", ())
        .expect("Failed to execute EXTRACT DOY");
    assert_eq!(result, 1);

    // DAYOFYEAR alias
    let result: i64 = db
        .query_one("SELECT EXTRACT(DAYOFYEAR FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT DAYOFYEAR");
    assert_eq!(result, 75);

    // December 31 in leap year
    let result: i64 = db
        .query_one("SELECT EXTRACT(DOY FROM '2024-12-31')", ())
        .expect("Failed to execute EXTRACT DOY");
    assert_eq!(result, 366);
}

#[test]
fn test_extract_week() {
    let db = Database::open("memory://extract_week").expect("Failed to create database");

    // Week number
    let result: i64 = db
        .query_one("SELECT EXTRACT(WEEK FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT WEEK");
    assert_eq!(result, 11);

    // First week
    let result: i64 = db
        .query_one("SELECT EXTRACT(WEEK FROM '2024-01-01')", ())
        .expect("Failed to execute EXTRACT WEEK");
    assert_eq!(result, 1);
}

#[test]
fn test_extract_quarter() {
    let db = Database::open("memory://extract_quarter").expect("Failed to create database");

    // Q1
    let result: i64 = db
        .query_one("SELECT EXTRACT(QUARTER FROM '2024-03-15')", ())
        .expect("Failed to execute EXTRACT QUARTER");
    assert_eq!(result, 1);

    // Q2
    let result: i64 = db
        .query_one("SELECT EXTRACT(QUARTER FROM '2024-04-15')", ())
        .expect("Failed to execute EXTRACT QUARTER");
    assert_eq!(result, 2);

    // Q3
    let result: i64 = db
        .query_one("SELECT EXTRACT(QUARTER FROM '2024-09-15')", ())
        .expect("Failed to execute EXTRACT QUARTER");
    assert_eq!(result, 3);

    // Q4
    let result: i64 = db
        .query_one("SELECT EXTRACT(QUARTER FROM '2024-12-15')", ())
        .expect("Failed to execute EXTRACT QUARTER");
    assert_eq!(result, 4);
}

#[test]
fn test_extract_epoch() {
    let db = Database::open("memory://extract_epoch").expect("Failed to create database");

    // Unix epoch
    let result: i64 = db
        .query_one("SELECT EXTRACT(EPOCH FROM '1970-01-01 00:00:00')", ())
        .expect("Failed to execute EXTRACT EPOCH");
    assert_eq!(result, 0);

    // Some known timestamp
    let result: i64 = db
        .query_one("SELECT EXTRACT(EPOCH FROM '2024-01-01 00:00:00')", ())
        .expect("Failed to execute EXTRACT EPOCH");
    assert!(result > 0);
}

#[test]
fn test_extract_null_handling() {
    let db = Database::open("memory://extract_null").expect("Failed to create database");

    // NULL timestamp
    let result = db
        .query("SELECT EXTRACT(DOW FROM NULL)", ())
        .expect("Failed to query");
    let rows: Vec<_> = result.collect();
    assert!(rows[0].as_ref().unwrap().is_null(0));
}

// ============================================================================
// Combined Function Tests with Table Data
// ============================================================================

#[test]
fn test_functions_with_table_data() {
    let db = Database::open("memory://func_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (
            id INTEGER NOT NULL,
            name TEXT,
            event_date TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO events (id, name, event_date) VALUES
         (1, 'hello world', '2024-03-15'),
         (2, 'foo bar baz', '2024-06-20'),
         (3, 'test data', '2024-12-25')",
        (),
    )
    .expect("Failed to insert data");

    // LOCATE with table column
    let result: i64 = db
        .query_one("SELECT LOCATE('o', name) FROM events WHERE id = 1", ())
        .expect("Failed to execute LOCATE on table");
    assert_eq!(result, 5);

    // TO_CHAR with table column
    let result: String = db
        .query_one(
            "SELECT TO_CHAR(event_date, 'DD MON YYYY') FROM events WHERE id = 1",
            (),
        )
        .expect("Failed to execute TO_CHAR on table");
    assert_eq!(result, "15 MAR 2024");

    // EXTRACT with table column
    let result: i64 = db
        .query_one(
            "SELECT EXTRACT(QUARTER FROM event_date) FROM events WHERE id = 2",
            (),
        )
        .expect("Failed to execute EXTRACT on table");
    assert_eq!(result, 2);
}
