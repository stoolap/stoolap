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

//! Unquoted Keyword Column Tests
//!
//! Tests for using non-reserved SQL keywords as unquoted column names (Bug #87)
//! Keywords like TIMESTAMP, DATE, TIME, JSON, etc. should be usable as column names
//! without requiring quotes when they are not truly reserved.

use stoolap::Database;

#[test]
fn test_timestamp_as_column_name() {
    let db = Database::open("memory://timestamp_column").expect("Failed to create database");

    // TIMESTAMP should work as unquoted column name
    db.execute(
        "CREATE TABLE events (id INTEGER, timestamp TEXT, description TEXT)",
        (),
    )
    .expect("Failed to create table with TIMESTAMP column");

    db.execute(
        "INSERT INTO events VALUES (1, '2024-01-01 10:00:00', 'Event 1')",
        (),
    )
    .expect("Failed to insert");

    let timestamp: String = db
        .query_one("SELECT timestamp FROM events WHERE id = 1", ())
        .expect("Failed to select timestamp column");

    assert_eq!(timestamp, "2024-01-01 10:00:00");
}

#[test]
fn test_date_as_column_name() {
    let db = Database::open("memory://date_column").expect("Failed to create database");

    // DATE should work as unquoted column name
    db.execute(
        "CREATE TABLE records (id INTEGER, date TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table with DATE column");

    db.execute("INSERT INTO records VALUES (1, '2024-01-01', 100)", ())
        .expect("Failed to insert");

    let date: String = db
        .query_one("SELECT date FROM records WHERE id = 1", ())
        .expect("Failed to select date column");

    assert_eq!(date, "2024-01-01");
}

#[test]
fn test_time_as_column_name() {
    let db = Database::open("memory://time_column").expect("Failed to create database");

    // TIME should work as unquoted column name
    db.execute(
        "CREATE TABLE schedule (id INTEGER, time TEXT, event TEXT)",
        (),
    )
    .expect("Failed to create table with TIME column");

    db.execute("INSERT INTO schedule VALUES (1, '10:30:00', 'Meeting')", ())
        .expect("Failed to insert");

    let time: String = db
        .query_one("SELECT time FROM schedule WHERE id = 1", ())
        .expect("Failed to select time column");

    assert_eq!(time, "10:30:00");
}

#[test]
fn test_json_as_column_name() {
    let db = Database::open("memory://json_column").expect("Failed to create database");

    // JSON should work as unquoted column name
    db.execute("CREATE TABLE documents (id INTEGER, json TEXT)", ())
        .expect("Failed to create table with JSON column");

    db.execute(
        r#"INSERT INTO documents VALUES (1, '{"key": "value"}')"#,
        (),
    )
    .expect("Failed to insert");

    let json: String = db
        .query_one("SELECT json FROM documents WHERE id = 1", ())
        .expect("Failed to select json column");

    assert_eq!(json, r#"{"key": "value"}"#);
}

#[test]
fn test_type_as_column_name() {
    let db = Database::open("memory://type_column").expect("Failed to create database");

    // TYPE should work as unquoted column name
    db.execute("CREATE TABLE items (id INTEGER, type TEXT, name TEXT)", ())
        .expect("Failed to create table with TYPE column");

    db.execute("INSERT INTO items VALUES (1, 'A', 'Item 1')", ())
        .expect("Failed to insert");

    let type_val: String = db
        .query_one("SELECT type FROM items WHERE id = 1", ())
        .expect("Failed to select type column");

    assert_eq!(type_val, "A");
}

#[test]
fn test_value_as_column_name() {
    let db = Database::open("memory://value_column").expect("Failed to create database");

    // VALUE should work as unquoted column name
    db.execute(
        "CREATE TABLE metrics (id INTEGER, value FLOAT, unit TEXT)",
        (),
    )
    .expect("Failed to create table with VALUE column");

    db.execute("INSERT INTO metrics VALUES (1, 42.5, 'meters')", ())
        .expect("Failed to insert");

    let value: f64 = db
        .query_one("SELECT value FROM metrics WHERE id = 1", ())
        .expect("Failed to select value column");

    assert!((value - 42.5).abs() < 0.001);
}

#[test]
fn test_text_as_column_name() {
    let db = Database::open("memory://text_column").expect("Failed to create database");

    // TEXT (data type keyword) should work as column name
    db.execute("CREATE TABLE content (id INTEGER, text TEXT)", ())
        .expect("Failed to create table with TEXT column");

    db.execute("INSERT INTO content VALUES (1, 'Hello World')", ())
        .expect("Failed to insert");

    let text: String = db
        .query_one("SELECT text FROM content WHERE id = 1", ())
        .expect("Failed to select text column");

    assert_eq!(text, "Hello World");
}

#[test]
fn test_multiple_keyword_columns() {
    let db = Database::open("memory://multi_keyword").expect("Failed to create database");

    // Multiple non-reserved keywords as column names
    db.execute(
        "CREATE TABLE complex (id INTEGER, timestamp TEXT, date TEXT, time TEXT, json TEXT, type TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table with multiple keyword columns");

    db.execute(
        r#"INSERT INTO complex VALUES (1, '2024-01-01 10:00:00', '2024-01-01', '10:00:00', '{}', 'A', 99.9)"#,
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "SELECT timestamp, date, time, json, type, value FROM complex WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let ts: String = row.get(0).unwrap();
        let date: String = row.get(1).unwrap();
        let time: String = row.get(2).unwrap();
        let json: String = row.get(3).unwrap();
        let type_val: String = row.get(4).unwrap();
        let value: f64 = row.get(5).unwrap();

        assert_eq!(ts, "2024-01-01 10:00:00");
        assert_eq!(date, "2024-01-01");
        assert_eq!(time, "10:00:00");
        assert_eq!(json, "{}");
        assert_eq!(type_val, "A");
        assert!((value - 99.9).abs() < 0.001);
    }
}

#[test]
fn test_keyword_column_in_where() {
    let db = Database::open("memory://keyword_where").expect("Failed to create database");

    db.execute(
        "CREATE TABLE filtered (id INTEGER, type TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO filtered VALUES (1, 'A', 100)", ())
        .expect("Failed to insert 1");
    db.execute("INSERT INTO filtered VALUES (2, 'B', 200)", ())
        .expect("Failed to insert 2");
    db.execute("INSERT INTO filtered VALUES (3, 'A', 300)", ())
        .expect("Failed to insert 3");

    // Use keyword column in WHERE clause
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM filtered WHERE type = 'A'", ())
        .expect("Failed to count with WHERE on keyword column");

    assert_eq!(count, 2);
}

#[test]
fn test_keyword_column_in_order_by() {
    let db = Database::open("memory://keyword_order").expect("Failed to create database");

    db.execute("CREATE TABLE sorted (id INTEGER, timestamp TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO sorted VALUES (1, '2024-01-03')", ())
        .expect("Failed to insert 1");
    db.execute("INSERT INTO sorted VALUES (2, '2024-01-01')", ())
        .expect("Failed to insert 2");
    db.execute("INSERT INTO sorted VALUES (3, '2024-01-02')", ())
        .expect("Failed to insert 3");

    // Use keyword column in ORDER BY clause
    let result = db
        .query("SELECT id FROM sorted ORDER BY timestamp", ())
        .expect("Failed to query with ORDER BY on keyword column");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get(0).unwrap());
    }

    assert_eq!(ids, vec![2, 3, 1], "Should be ordered by timestamp");
}

#[test]
fn test_keyword_column_in_group_by() {
    let db = Database::open("memory://keyword_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE grouped (id INTEGER, type TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO grouped VALUES (1, 'A', 100)", ())
        .expect("Failed to insert 1");
    db.execute("INSERT INTO grouped VALUES (2, 'A', 200)", ())
        .expect("Failed to insert 2");
    db.execute("INSERT INTO grouped VALUES (3, 'B', 150)", ())
        .expect("Failed to insert 3");

    // Use keyword column in GROUP BY clause
    let result = db
        .query(
            "SELECT type, SUM(value) FROM grouped GROUP BY type ORDER BY type",
            (),
        )
        .expect("Failed to query with GROUP BY on keyword column");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("A".to_string(), 300));
    assert_eq!(results[1], ("B".to_string(), 150));
}

#[test]
fn test_typed_literals_still_work() {
    let db = Database::open("memory://typed_literals").expect("Failed to create database");

    // TIMESTAMP keyword followed by string literal should still work as typed literal
    let result = db
        .query("SELECT TIMESTAMP '2024-01-01 10:00:00'", ())
        .expect("Failed to execute TIMESTAMP literal");

    for row in result {
        let row = row.expect("Failed to get row");
        let _ts: String = row.get(0).unwrap();
        // Just verify it doesn't error
    }

    // DATE keyword followed by string literal should still work as typed literal
    let result = db
        .query("SELECT DATE '2024-01-01'", ())
        .expect("Failed to execute DATE literal");

    for row in result {
        let row = row.expect("Failed to get row");
        let _date: String = row.get(0).unwrap();
        // Just verify it doesn't error
    }

    // TIME keyword followed by string literal should still work as typed literal
    let result = db
        .query("SELECT TIME '10:30:00'", ())
        .expect("Failed to execute TIME literal");

    for row in result {
        let row = row.expect("Failed to get row");
        let _time: String = row.get(0).unwrap();
        // Just verify it doesn't error
    }
}

#[test]
fn test_keyword_column_with_alias() {
    let db = Database::open("memory://keyword_alias").expect("Failed to create database");

    db.execute("CREATE TABLE aliased (id INTEGER, timestamp TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO aliased VALUES (1, '2024-01-01')", ())
        .expect("Failed to insert");

    // Select keyword column with alias
    let result = db
        .query("SELECT timestamp AS ts FROM aliased", ())
        .expect("Failed to query with alias");

    for row in result {
        let row = row.expect("Failed to get row");
        let ts: String = row.get(0).unwrap();
        assert_eq!(ts, "2024-01-01");
    }
}
