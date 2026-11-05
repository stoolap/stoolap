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

//! Tests for compile-time constant folding.
//! Expressions like NOW() - INTERVAL '24 hours' should be evaluated once at
//! compile time, not per-row.

use stoolap::Database;

#[test]
fn test_now_minus_interval_in_where() {
    let db = Database::open("memory://const_fold_now").unwrap();

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        (),
    )
    .unwrap();

    // Insert a recent event
    db.execute(
        "INSERT INTO events (name, created_at) VALUES ('recent', NOW())",
        (),
    )
    .unwrap();

    // Insert an old event (48 hours ago)
    db.execute(
        "INSERT INTO events (name, created_at) VALUES ('old', NOW() - INTERVAL '48 hours')",
        (),
    )
    .unwrap();

    // Query with NOW() - INTERVAL '24 hours' in WHERE
    // This should be folded to a single constant at compile time
    let results: Vec<String> = db
        .query(
            "SELECT name FROM events WHERE created_at > NOW() - INTERVAL '24 hours'",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert_eq!(results, vec!["recent"]);
}

#[test]
fn test_arithmetic_constant_folding() {
    let db = Database::open("memory://const_fold_arith").unwrap();

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO numbers (id, value) VALUES (1, 10), (2, 20), (3, 30)",
        (),
    )
    .unwrap();

    // WHERE value > 2 * 5 + 5 — the right side is fully constant
    let results: Vec<i64> = db
        .query(
            "SELECT value FROM numbers WHERE value > 2 * 5 + 5 ORDER BY value",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();

    assert_eq!(results, vec![20, 30]);
}

#[test]
fn test_function_constant_folding() {
    let db = Database::open("memory://const_fold_func").unwrap();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO items (id, name) VALUES (1, 'HELLO'), (2, 'world')",
        (),
    )
    .unwrap();

    // UPPER('hello') is a column-free function call — should be folded
    let results: Vec<String> = db
        .query("SELECT name FROM items WHERE name = UPPER('hello')", ())
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert_eq!(results, vec!["HELLO"]);
}

#[test]
fn test_current_date_constant_folding() {
    let db = Database::open("memory://const_fold_date").unwrap();

    db.execute(
        "CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            msg TEXT NOT NULL,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        (),
    )
    .unwrap();

    // Insert today's log
    db.execute("INSERT INTO logs (msg) VALUES ('today')", ())
        .unwrap();

    // CURRENT_DATE should be folded to a constant
    let count: i64 = db
        .query("SELECT COUNT(*) FROM logs WHERE ts >= CURRENT_DATE", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    assert_eq!(count, 1);
}

#[test]
fn test_interval_arithmetic_folding() {
    let db = Database::open("memory://const_fold_interval").unwrap();

    db.execute(
        "CREATE TABLE schedules (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            event_name TEXT NOT NULL,
            scheduled_at TIMESTAMP NOT NULL
        )",
        (),
    )
    .unwrap();

    // Insert events at different times
    db.execute(
        "INSERT INTO schedules (event_name, scheduled_at) VALUES
         ('past', NOW() - INTERVAL '2 days'),
         ('soon', NOW() + INTERVAL '1 hour'),
         ('future', NOW() + INTERVAL '7 days')",
        (),
    )
    .unwrap();

    // Find events within next 24 hours
    let results: Vec<String> = db
        .query(
            "SELECT event_name FROM schedules
             WHERE scheduled_at > NOW() AND scheduled_at < NOW() + INTERVAL '24 hours'",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert_eq!(results, vec!["soon"]);
}

#[test]
fn test_select_constant_expression() {
    let db = Database::open("memory://const_fold_select").unwrap();

    // Pure constant expression in SELECT
    let val: i64 = db
        .query("SELECT 1 + 2 + 3", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    assert_eq!(val, 6);
}

#[test]
fn test_mixed_constant_and_column() {
    let db = Database::open("memory://const_fold_mixed").unwrap();

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, x INTEGER NOT NULL)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO data (id, x) VALUES (1, 5), (2, 15), (3, 25)",
        (),
    )
    .unwrap();

    // x + (2 * 3) — the (2 * 3) part should be folded, but x + 6 is not foldable
    let results: Vec<i64> = db
        .query("SELECT x + (2 * 3) FROM data ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();

    assert_eq!(results, vec![11, 21, 31]);
}
