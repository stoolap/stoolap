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

//! SELECT Tests
//!
//! Tests SELECT queries with complex WHERE clauses

use stoolap::Database;

/// Test string comparison in SELECT
#[test]
fn test_select_string_comparison() {
    let db = Database::open("memory://select_str").expect("Failed to create database");

    db.execute("CREATE TABLE str_select (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert test data
    let data = [
        (1, "Apple"),
        (2, "Banana"),
        (3, "Cherry"),
        (4, "Date"),
        (5, "Elderberry"),
    ];

    for (id, name) in &data {
        db.execute(
            &format!("INSERT INTO str_select VALUES ({}, '{}')", id, name),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test string comparison
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM str_select WHERE name = 'Cherry'", ())
        .expect("Failed to query");
    assert_eq!(count, 1, "Expected 1 row with name='Cherry'");

    // Test id lookup
    let name: String = db
        .query_one("SELECT name FROM str_select WHERE id = 3", ())
        .expect("Failed to query");
    assert_eq!(name, "Cherry", "Expected 'Cherry' for id=3");
}

/// Test AND condition in SELECT
#[test]
fn test_select_and_condition() {
    let db = Database::open("memory://select_and").expect("Failed to create database");

    db.execute(
        "CREATE TABLE and_select (id INTEGER, category TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    let data = [
        (1, "A", 10.5),
        (2, "A", 20.0),
        (3, "B", 15.5),
        (4, "B", 25.0),
        (5, "C", 30.5),
        (6, "C", 40.0),
    ];

    for (id, category, value) in &data {
        db.execute(
            &format!(
                "INSERT INTO and_select VALUES ({}, '{}', {:.1})",
                id, category, value
            ),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test simple category filter
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM and_select WHERE category = 'B'", ())
        .expect("Failed to query");
    assert_eq!(count, 2, "Expected 2 rows with category='B'");

    // Test value filter
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM and_select WHERE value > 20.0", ())
        .expect("Failed to query");
    assert_eq!(count, 3, "Expected 3 rows with value > 20.0");

    // Test AND condition
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM and_select WHERE category = 'B' AND value > 20.0",
            (),
        )
        .expect("Failed to query");
    assert_eq!(
        count, 1,
        "Expected 1 row with category='B' AND value > 20.0"
    );

    // Verify the specific row
    let id: i64 = db
        .query_one(
            "SELECT id FROM and_select WHERE category = 'B' AND value > 20.0",
            (),
        )
        .expect("Failed to query");
    assert_eq!(id, 4, "Expected id=4 for category='B' AND value > 20.0");
}

/// Test OR condition in SELECT
#[test]
fn test_select_or_condition() {
    let db = Database::open("memory://select_or").expect("Failed to create database");

    db.execute(
        "CREATE TABLE or_select (id INTEGER, category TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    let data = [
        (1, "A", 10.5),
        (2, "A", 20.0),
        (3, "B", 15.5),
        (4, "B", 25.0),
        (5, "C", 30.5),
        (6, "C", 40.0),
    ];

    for (id, category, value) in &data {
        db.execute(
            &format!(
                "INSERT INTO or_select VALUES ({}, '{}', {:.1})",
                id, category, value
            ),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test OR condition: category='A' (2 rows) OR value>30.0 (2 rows) = 4 total (no overlap)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM or_select WHERE category = 'A' OR value > 30.0",
            (),
        )
        .expect("Failed to query");
    assert_eq!(
        count, 4,
        "Expected 4 rows with category='A' OR value > 30.0"
    );
}

/// Test BETWEEN condition in SELECT
#[test]
fn test_select_between_condition() {
    let db = Database::open("memory://select_between").expect("Failed to create database");

    db.execute("CREATE TABLE between_select (id INTEGER, value FLOAT)", ())
        .expect("Failed to create table");

    // Insert test data: values 10.0, 20.0, 30.0, ..., 100.0
    for i in 1..=10 {
        let value = (i * 10) as f64;
        db.execute(
            &format!("INSERT INTO between_select VALUES ({}, {:.1})", i, value),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test BETWEEN: values 30, 40, 50, 60, 70 (5 rows)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM between_select WHERE value BETWEEN 30 AND 70",
            (),
        )
        .expect("Failed to query");
    assert_eq!(count, 5, "Expected 5 rows with value BETWEEN 30 AND 70");
}

/// Test IN condition in SELECT
#[test]
fn test_select_in_condition() {
    let db = Database::open("memory://select_in").expect("Failed to create database");

    db.execute("CREATE TABLE in_select (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    // Insert test data
    let categories = ["A", "B", "C", "D", "E"];
    for (i, category) in categories.iter().enumerate() {
        db.execute(
            &format!("INSERT INTO in_select VALUES ({}, '{}')", i + 1, category),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test IN condition
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM in_select WHERE category IN ('A', 'C', 'E')",
            (),
        )
        .expect("Failed to query");
    assert_eq!(count, 3, "Expected 3 rows with category IN ('A', 'C', 'E')");
}

/// Test NOT IN condition in SELECT
#[test]
fn test_select_not_in_condition() {
    let db = Database::open("memory://select_not_in").expect("Failed to create database");

    db.execute("CREATE TABLE not_in_select (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    // Insert test data
    let categories = ["A", "B", "C", "D", "E"];
    for (i, category) in categories.iter().enumerate() {
        db.execute(
            &format!(
                "INSERT INTO not_in_select VALUES ({}, '{}')",
                i + 1,
                category
            ),
            (),
        )
        .expect("Failed to insert data");
    }

    // Test NOT IN condition
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM not_in_select WHERE category NOT IN ('A', 'C', 'E')",
            (),
        )
        .expect("Failed to query");
    assert_eq!(
        count, 2,
        "Expected 2 rows with category NOT IN ('A', 'C', 'E')"
    );
}

/// Test SELECT with multiple columns
#[test]
fn test_select_multiple_columns() {
    let db = Database::open("memory://select_multi_col").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_col (id INTEGER, name TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO multi_col VALUES (1, 'first', 10.5)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO multi_col VALUES (2, 'second', 20.5)", ())
        .expect("Failed to insert");

    // Select specific columns
    let result = db
        .query("SELECT id, name FROM multi_col WHERE id = 1", ())
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        assert_eq!(id, 1);
        assert_eq!(name, "first");
        found = true;
    }
    assert!(found, "Expected to find row with id=1");
}

/// Test SELECT * (all columns)
#[test]
fn test_select_star() {
    let db = Database::open("memory://select_star").expect("Failed to create database");

    db.execute(
        "CREATE TABLE star_test (id INTEGER, name TEXT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO star_test VALUES (1, 'test', true)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT * FROM star_test", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        let active: bool = row.get(2).unwrap();
        assert_eq!(id, 1);
        assert_eq!(name, "test");
        assert!(active);
        count += 1;
    }
    assert_eq!(count, 1, "Expected 1 row");
}

/// Test SELECT with ORDER BY
#[test]
fn test_select_order_by() {
    let db = Database::open("memory://select_order").expect("Failed to create database");

    db.execute("CREATE TABLE order_test (id INTEGER, value INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO order_test VALUES (1, 30)", ())
        .unwrap();
    db.execute("INSERT INTO order_test VALUES (2, 10)", ())
        .unwrap();
    db.execute("INSERT INTO order_test VALUES (3, 20)", ())
        .unwrap();

    // Test ORDER BY ASC
    let result = db
        .query("SELECT id FROM order_test ORDER BY value ASC", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get(0).unwrap());
    }
    assert_eq!(ids, vec![2, 3, 1], "Expected order 2, 3, 1 for ASC");

    // Test ORDER BY DESC
    let result = db
        .query("SELECT id FROM order_test ORDER BY value DESC", ())
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get(0).unwrap());
    }
    assert_eq!(ids, vec![1, 3, 2], "Expected order 1, 3, 2 for DESC");
}

/// Test SELECT with LIMIT
#[test]
fn test_select_limit() {
    let db = Database::open("memory://select_limit").expect("Failed to create database");

    db.execute("CREATE TABLE limit_test (id INTEGER)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO limit_test VALUES ({})", i), ())
            .expect("Failed to insert");
    }

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM (SELECT * FROM limit_test LIMIT 5)",
            (),
        )
        .unwrap_or_else(|_| {
            // If subquery doesn't work, count manually
            let result = db
                .query("SELECT * FROM limit_test LIMIT 5", ())
                .expect("Failed to query");
            let mut c = 0i64;
            for _ in result {
                c += 1;
            }
            c
        });

    // Count rows with LIMIT
    let result = db
        .query("SELECT * FROM limit_test LIMIT 5", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }
    assert_eq!(count, 5, "Expected 5 rows with LIMIT 5");
}

/// Test SELECT with OFFSET
#[test]
fn test_select_offset() {
    let db = Database::open("memory://select_offset").expect("Failed to create database");

    db.execute("CREATE TABLE offset_test (id INTEGER)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO offset_test VALUES ({})", i), ())
            .expect("Failed to insert");
    }

    // LIMIT 3 OFFSET 2 should skip first 2 rows
    let result = db
        .query(
            "SELECT id FROM offset_test ORDER BY id LIMIT 3 OFFSET 2",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get(0).unwrap());
    }
    assert_eq!(ids, vec![3, 4, 5], "Expected ids 3, 4, 5 with OFFSET 2");
}
