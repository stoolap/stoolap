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

//! DELETE Statement Tests
//!
//! Tests DELETE operations with various conditions

use stoolap::Database;

/// Test simple DELETE with direct equality condition
#[test]
fn test_direct_delete() {
    let db = Database::open("memory://direct_delete").expect("Failed to create database");

    db.execute("CREATE TABLE test_delete (id INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test_delete VALUES (1)", ())
        .expect("Failed to insert data");

    // Verify the row exists
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_delete", ())
        .expect("Failed to count rows");
    assert_eq!(count, 1, "Expected 1 row before delete");

    // Delete the row
    db.execute("DELETE FROM test_delete WHERE id = 1", ())
        .expect("Failed to delete row");

    // Verify the row was deleted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_delete", ())
        .expect("Failed to count rows after delete");
    assert_eq!(count, 0, "Expected 0 rows after delete");
}

/// Test DELETE with integer comparison operators
#[test]
fn test_integer_comparison_delete() {
    let db = Database::open("memory://int_delete").expect("Failed to create database");

    db.execute("CREATE TABLE int_delete (id INTEGER)", ())
        .expect("Failed to create table");

    // Insert test rows 1-10
    for i in 1..=10 {
        db.execute(&format!("INSERT INTO int_delete VALUES ({})", i), ())
            .expect("Failed to insert data");
    }

    // Delete rows where id > 5
    db.execute("DELETE FROM int_delete WHERE id > 5", ())
        .expect("Failed to delete rows");

    // Verify only rows with id <= 5 remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM int_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 5,
        "Expected 5 rows to remain after DELETE WHERE id > 5"
    );

    // Verify the correct rows were kept (1-5)
    for i in 1..=10 {
        let exists: i64 = db
            .query_one(
                &format!("SELECT COUNT(*) FROM int_delete WHERE id = {}", i),
                (),
            )
            .expect("Failed to check if row exists");

        let expected = if i <= 5 { 1 } else { 0 };
        assert_eq!(exists, expected, "For id={}, expected {} rows", i, expected);
    }
}

/// Test DELETE with boolean conditions
#[test]
fn test_boolean_condition_delete() {
    let db = Database::open("memory://bool_delete").expect("Failed to create database");

    db.execute("CREATE TABLE bool_delete (id INTEGER, active BOOLEAN)", ())
        .expect("Failed to create table");

    // Insert test rows - alternating true/false values
    for i in 1..=10 {
        let active = i % 2 == 0; // Even ids are active, odd are not
        db.execute(
            &format!("INSERT INTO bool_delete VALUES ({}, {})", i, active),
            (),
        )
        .expect("Failed to insert data");
    }

    // Delete active rows
    db.execute("DELETE FROM bool_delete WHERE active = true", ())
        .expect("Failed to delete rows");

    // Verify only inactive rows remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 5,
        "Expected 5 rows to remain after DELETE WHERE active = true"
    );

    // Verify all remaining rows have active = false
    let inactive_count: i64 = db
        .query_one("SELECT COUNT(*) FROM bool_delete WHERE active = false", ())
        .expect("Failed to count inactive rows");
    assert_eq!(inactive_count, 5, "Expected 5 inactive rows to remain");
}

/// Test DELETE with string comparison
#[test]
fn test_string_comparison_delete() {
    let db = Database::open("memory://str_delete").expect("Failed to create database");

    db.execute("CREATE TABLE str_delete (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert test rows with different string values
    let test_data = [
        (1, "Apple"),
        (2, "Banana"),
        (3, "Cherry"),
        (4, "Date"),
        (5, "Elderberry"),
    ];

    for (id, name) in &test_data {
        db.execute(
            &format!("INSERT INTO str_delete VALUES ({}, '{}')", id, name),
            (),
        )
        .expect("Failed to insert data");
    }

    // Delete a specific string
    db.execute("DELETE FROM str_delete WHERE name = 'Cherry'", ())
        .expect("Failed to delete rows");

    // Verify the correct row was deleted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM str_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 4,
        "Expected 4 rows to remain after DELETE WHERE name = 'Cherry'"
    );

    // Verify the 'Cherry' row is gone
    let cherry_count: i64 = db
        .query_one("SELECT COUNT(*) FROM str_delete WHERE name = 'Cherry'", ())
        .expect("Failed to check if Cherry remains");
    assert_eq!(cherry_count, 0, "Expected 0 'Cherry' rows to remain");
}

/// Test DELETE with AND condition
#[test]
fn test_delete_with_and() {
    let db = Database::open("memory://and_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE and_delete (id INTEGER, category TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test rows
    db.execute("INSERT INTO and_delete VALUES (1, 'A', 10.5)", ())
        .unwrap();
    db.execute("INSERT INTO and_delete VALUES (2, 'A', 20.0)", ())
        .unwrap();
    db.execute("INSERT INTO and_delete VALUES (3, 'B', 15.5)", ())
        .unwrap();
    db.execute("INSERT INTO and_delete VALUES (4, 'B', 25.0)", ())
        .unwrap();
    db.execute("INSERT INTO and_delete VALUES (5, 'C', 30.5)", ())
        .unwrap();
    db.execute("INSERT INTO and_delete VALUES (6, 'C', 40.0)", ())
        .unwrap();

    // Delete rows matching both conditions
    db.execute(
        "DELETE FROM and_delete WHERE category = 'B' AND value > 20.0",
        (),
    )
    .expect("Failed to delete rows");

    // Verify the correct row was deleted (should be only id=4)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM and_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 5,
        "Expected 5 rows to remain after DELETE with AND condition"
    );

    // Verify id=4 is gone and id=3 (which is also category B) remains
    let id4_count: i64 = db
        .query_one("SELECT COUNT(*) FROM and_delete WHERE id = 4", ())
        .expect("Failed to check if id=4 remains");
    let id3_count: i64 = db
        .query_one("SELECT COUNT(*) FROM and_delete WHERE id = 3", ())
        .expect("Failed to check if id=3 remains");

    assert_eq!(id4_count, 0, "Expected 0 rows with id=4 to remain");
    assert_eq!(id3_count, 1, "Expected 1 row with id=3 to remain");
}

/// Test deleting all rows in a table without a WHERE clause
#[test]
fn test_delete_all_rows() {
    let db = Database::open("memory://delete_all").expect("Failed to create database");

    db.execute("CREATE TABLE delete_all_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert test rows
    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO delete_all_test VALUES ({}, 'Name-{}')", i, i),
            (),
        )
        .expect("Failed to insert data");
    }

    // Verify initial row count
    let initial_count: i64 = db
        .query_one("SELECT COUNT(*) FROM delete_all_test", ())
        .expect("Failed to count initial rows");
    assert_eq!(initial_count, 10, "Expected 10 initial rows");

    // Delete all rows
    db.execute("DELETE FROM delete_all_test", ())
        .expect("Failed to delete all rows");

    // Verify no rows remain
    let remaining_count: i64 = db
        .query_one("SELECT COUNT(*) FROM delete_all_test", ())
        .expect("Failed to count remaining rows");
    assert_eq!(
        remaining_count, 0,
        "Expected 0 rows to remain after DELETE without WHERE"
    );
}

/// Test DELETE with OR condition
#[test]
fn test_delete_with_or() {
    let db = Database::open("memory://or_delete").expect("Failed to create database");

    db.execute(
        "CREATE TABLE or_delete (id INTEGER, category TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // Insert test rows
    db.execute("INSERT INTO or_delete VALUES (1, 'A', 10.5)", ())
        .unwrap();
    db.execute("INSERT INTO or_delete VALUES (2, 'A', 20.0)", ())
        .unwrap();
    db.execute("INSERT INTO or_delete VALUES (3, 'B', 15.5)", ())
        .unwrap();
    db.execute("INSERT INTO or_delete VALUES (4, 'B', 25.0)", ())
        .unwrap();
    db.execute("INSERT INTO or_delete VALUES (5, 'C', 30.5)", ())
        .unwrap();
    db.execute("INSERT INTO or_delete VALUES (6, 'C', 40.0)", ())
        .unwrap();

    // Delete rows matching either condition
    db.execute(
        "DELETE FROM or_delete WHERE category = 'A' OR value > 30.0",
        (),
    )
    .expect("Failed to delete rows");

    // Verify the correct rows were deleted (should be id=1,2,5,6)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM or_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 2,
        "Expected 2 rows to remain after DELETE with OR condition"
    );

    // Verify only category B rows with value <= 30.0 remain (id=3,4)
    let b_category_count: i64 = db
        .query_one("SELECT COUNT(*) FROM or_delete WHERE category = 'B'", ())
        .expect("Failed to count category B rows");
    assert_eq!(
        b_category_count, 2,
        "Expected 2 rows with category B to remain"
    );
}

/// Test DELETE with BETWEEN condition
#[test]
fn test_delete_with_between() {
    let db = Database::open("memory://between_delete").expect("Failed to create database");

    db.execute("CREATE TABLE between_delete (id INTEGER, value FLOAT)", ())
        .expect("Failed to create table");

    // Insert test rows with float values
    for i in 1..=10 {
        let value = (i * 10) as f64;
        db.execute(
            &format!("INSERT INTO between_delete VALUES ({}, {:.1})", i, value),
            (),
        )
        .expect("Failed to insert data");
    }

    // Delete rows where value is between 30 and 70
    db.execute(
        "DELETE FROM between_delete WHERE value BETWEEN 30 AND 70",
        (),
    )
    .expect("Failed to delete rows");

    // Verify the correct rows were deleted (should be id 3-7)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM between_delete", ())
        .expect("Failed to count rows");
    assert_eq!(
        count, 5,
        "Expected 5 rows to remain after DELETE with BETWEEN"
    );
}

/// Test DELETE with IN condition
#[test]
fn test_delete_with_in() {
    let db = Database::open("memory://in_delete").expect("Failed to create database");

    db.execute("CREATE TABLE in_delete (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    // Insert test rows
    let categories = ["A", "B", "C", "D", "E"];
    for (i, category) in categories.iter().enumerate() {
        db.execute(
            &format!("INSERT INTO in_delete VALUES ({}, '{}')", i + 1, category),
            (),
        )
        .expect("Failed to insert data");
    }

    // Delete rows with category in (A, C, E)
    db.execute(
        "DELETE FROM in_delete WHERE category IN ('A', 'C', 'E')",
        (),
    )
    .expect("Failed to delete rows");

    // Verify the correct rows were deleted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM in_delete", ())
        .expect("Failed to count rows");
    assert_eq!(count, 2, "Expected 2 rows to remain after DELETE with IN");

    // Check only categories B and D remain
    let b_count: i64 = db
        .query_one("SELECT COUNT(*) FROM in_delete WHERE category = 'B'", ())
        .expect("Failed to count category B");
    let d_count: i64 = db
        .query_one("SELECT COUNT(*) FROM in_delete WHERE category = 'D'", ())
        .expect("Failed to count category D");

    assert_eq!(b_count, 1, "Expected 1 row with category B");
    assert_eq!(d_count, 1, "Expected 1 row with category D");
}

/// Test DELETE with NOT IN condition
#[test]
fn test_delete_with_not_in() {
    let db = Database::open("memory://not_in_delete").expect("Failed to create database");

    db.execute("CREATE TABLE not_in_delete (id INTEGER, category TEXT)", ())
        .expect("Failed to create table");

    // Insert test data
    db.execute("INSERT INTO not_in_delete VALUES (1, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO not_in_delete VALUES (2, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO not_in_delete VALUES (3, 'C')", ())
        .unwrap();
    db.execute("INSERT INTO not_in_delete VALUES (4, 'D')", ())
        .unwrap();
    db.execute("INSERT INTO not_in_delete VALUES (5, 'E')", ())
        .unwrap();

    // Delete rows with NOT IN
    db.execute(
        "DELETE FROM not_in_delete WHERE category NOT IN ('B', 'D')",
        (),
    )
    .expect("Failed to execute DELETE");

    // Verify the remaining rows
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM not_in_delete", ())
        .expect("Failed to count remaining rows");
    assert_eq!(count, 2, "Expected 2 rows to remain");

    // Verify that only rows with categories B and D remain
    let result = db
        .query("SELECT category FROM not_in_delete ORDER BY category", ())
        .expect("Failed to query categories");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        categories.push(category);
    }

    assert_eq!(categories.len(), 2);
    assert!(categories.contains(&"B".to_string()));
    assert!(categories.contains(&"D".to_string()));
}
