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

//! Filter Pushdown Tests
//!
//! Tests that filters are properly pushed down to the storage engine

use stoolap::Database;

fn setup_pushdown_table(db: &Database) {
    db.execute(
        "CREATE TABLE test_pushdown (
            id INTEGER,
            name TEXT,
            age INTEGER,
            active BOOLEAN,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_pushdown VALUES
        (1, 'Alice', 30, true, 60000.0),
        (2, 'Bob', 25, true, 55000.0),
        (3, 'Charlie', 35, false, 70000.0),
        (4, 'Dave', 40, true, 80000.0),
        (5, 'Eve', 28, false, 65000.0)",
        (),
    )
    .expect("Failed to insert data");
}

/// Test simple equality filter
#[test]
fn test_simple_equality_filter() {
    let db = Database::open("memory://filter_eq").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query("SELECT * FROM test_pushdown WHERE id = 3", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 1, "Expected 1 row for id = 3");
}

/// Test greater than filter
#[test]
fn test_greater_than_filter() {
    let db = Database::open("memory://filter_gt").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query("SELECT * FROM test_pushdown WHERE age > 30", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 2, "Expected 2 rows for age > 30");
}

/// Test AND condition
#[test]
fn test_and_condition() {
    let db = Database::open("memory://filter_and").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query(
            "SELECT * FROM test_pushdown WHERE active = true AND salary > 60000.0",
            (),
        )
        .expect("Query failed");

    let count = result.count();
    assert_eq!(
        count, 1,
        "Expected 1 row for active = true AND salary > 60000"
    );
}

/// Test IN condition
#[test]
fn test_in_condition() {
    let db = Database::open("memory://filter_in").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query("SELECT * FROM test_pushdown WHERE id IN (1, 3, 5)", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 3, "Expected 3 rows for id IN (1, 3, 5)");
}

/// Test LIKE condition
#[test]
fn test_like_condition() {
    let db = Database::open("memory://filter_like").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query("SELECT * FROM test_pushdown WHERE name LIKE 'A%'", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 1, "Expected 1 row for name LIKE 'A%'");
}

/// Test IS NOT NULL condition
#[test]
fn test_is_not_null_condition() {
    let db = Database::open("memory://filter_notnull").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query("SELECT * FROM test_pushdown WHERE age IS NOT NULL", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 5, "Expected 5 rows for age IS NOT NULL");
}

/// Test BETWEEN condition
#[test]
fn test_between_condition() {
    let db = Database::open("memory://filter_between").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query(
            "SELECT * FROM test_pushdown WHERE age BETWEEN 25 AND 35",
            (),
        )
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 4, "Expected 4 rows for age BETWEEN 25 AND 35");
}

/// Test complex condition with OR and AND
#[test]
fn test_complex_condition() {
    let db = Database::open("memory://filter_complex").expect("Failed to create database");
    setup_pushdown_table(&db);

    let result = db
        .query(
            "SELECT * FROM test_pushdown WHERE (age > 30 OR salary > 60000.0) AND active = true",
            (),
        )
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 1, "Expected 1 row for complex condition");
}

/// Test indexed equality filter
#[test]
fn test_indexed_equality_filter() {
    let db = Database::open("memory://filter_idx_eq").expect("Failed to create database");
    setup_pushdown_table(&db);

    db.execute("CREATE INDEX idx_age ON test_pushdown(age)", ())
        .expect("Failed to create index");

    let result = db
        .query("SELECT * FROM test_pushdown WHERE age = 30", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 1, "Expected 1 row for indexed age = 30");
}

/// Test indexed range filter
#[test]
fn test_indexed_range_filter() {
    let db = Database::open("memory://filter_idx_range").expect("Failed to create database");
    setup_pushdown_table(&db);

    db.execute("CREATE INDEX idx_age ON test_pushdown(age)", ())
        .expect("Failed to create index");

    let result = db
        .query("SELECT * FROM test_pushdown WHERE age > 30", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 2, "Expected 2 rows for indexed age > 30");
}

/// Test indexed BETWEEN filter
#[test]
fn test_indexed_between_filter() {
    let db = Database::open("memory://filter_idx_between").expect("Failed to create database");
    setup_pushdown_table(&db);

    db.execute("CREATE INDEX idx_age ON test_pushdown(age)", ())
        .expect("Failed to create index");

    let result = db
        .query(
            "SELECT * FROM test_pushdown WHERE age BETWEEN 28 AND 35",
            (),
        )
        .expect("Query failed");

    let count = result.count();
    assert_eq!(
        count, 3,
        "Expected 3 rows for indexed age BETWEEN 28 AND 35"
    );
}

/// Test COUNT(*) with no filter
#[test]
fn test_count_no_filter() {
    let db = Database::open("memory://count_nofilter").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_count_pushdown (
            id INTEGER,
            name TEXT,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_count_pushdown VALUES
        (1, 'Alice', true),
        (2, 'Bob', true),
        (3, 'Charlie', false),
        (4, 'Dave', true),
        (5, 'Eve', false)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count_pushdown", ())
        .expect("Query failed");
    assert_eq!(count, 5, "Expected COUNT(*) = 5");
}

/// Test COUNT(*) with simple filter
#[test]
fn test_count_with_filter() {
    let db = Database::open("memory://count_filter").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_count_pushdown (
            id INTEGER,
            name TEXT,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_count_pushdown VALUES
        (1, 'Alice', true),
        (2, 'Bob', true),
        (3, 'Charlie', false),
        (4, 'Dave', true),
        (5, 'Eve', false)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_count_pushdown WHERE active = true",
            (),
        )
        .expect("Query failed");
    assert_eq!(count, 3, "Expected COUNT(*) = 3 for active = true");
}

/// Test COUNT(*) with complex filter
#[test]
fn test_count_with_complex_filter() {
    let db = Database::open("memory://count_complex").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_count_pushdown (
            id INTEGER,
            name TEXT,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_count_pushdown VALUES
        (1, 'Alice', true),
        (2, 'Bob', true),
        (3, 'Charlie', false),
        (4, 'Dave', true),
        (5, 'Eve', false)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_count_pushdown WHERE id > 2 AND active = false",
            (),
        )
        .expect("Query failed");
    assert_eq!(
        count, 2,
        "Expected COUNT(*) = 2 for id > 2 AND active = false"
    );
}

/// Test complex AND with OR
#[test]
fn test_complex_and_with_or() {
    let db = Database::open("memory://complex_and_or").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_complex_pushdown (
            id INTEGER,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            rating INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_complex_pushdown VALUES
        (1, 'Electronics', 499.99, true, 4),
        (2, 'Books', 29.99, true, 5),
        (3, 'Electronics', 1299.99, false, 4),
        (4, 'Clothing', 49.99, true, 3),
        (5, 'Books', 19.99, false, 5),
        (6, 'Electronics', 799.99, true, 2),
        (7, 'Clothing', 89.99, true, 4),
        (8, 'Books', 9.99, true, 3),
        (9, 'Electronics', 999.99, false, 5),
        (10, 'Clothing', 129.99, false, 2)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT * FROM test_complex_pushdown WHERE (category = 'Electronics' OR category = 'Books') AND in_stock = true", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(
        count, 4,
        "Expected 4 rows for Electronics/Books AND in_stock"
    );
}

/// Test OR with multiple ANDs
#[test]
fn test_or_with_multiple_ands() {
    let db = Database::open("memory://or_multi_and").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_complex_pushdown (
            id INTEGER,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            rating INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_complex_pushdown VALUES
        (1, 'Electronics', 499.99, true, 4),
        (2, 'Books', 29.99, true, 5),
        (3, 'Electronics', 1299.99, false, 4),
        (4, 'Clothing', 49.99, true, 3),
        (5, 'Books', 19.99, false, 5),
        (6, 'Electronics', 799.99, true, 2),
        (7, 'Clothing', 89.99, true, 4),
        (8, 'Books', 9.99, true, 3),
        (9, 'Electronics', 999.99, false, 5),
        (10, 'Clothing', 129.99, false, 2)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT * FROM test_complex_pushdown WHERE (category = 'Electronics' AND price > 500) OR (category = 'Books' AND rating = 5)", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 5, "Expected 5 rows for OR with multiple ANDs");
}

/// Test three-way OR with indexed columns
#[test]
fn test_three_way_or() {
    let db = Database::open("memory://three_or").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_complex_pushdown (
            id INTEGER,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            rating INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_complex_pushdown VALUES
        (1, 'Electronics', 499.99, true, 4),
        (2, 'Books', 29.99, true, 5),
        (3, 'Electronics', 1299.99, false, 4),
        (4, 'Clothing', 49.99, true, 3),
        (5, 'Books', 19.99, false, 5),
        (6, 'Electronics', 799.99, true, 2),
        (7, 'Clothing', 89.99, true, 4),
        (8, 'Books', 9.99, true, 3),
        (9, 'Electronics', 999.99, false, 5),
        (10, 'Clothing', 129.99, false, 2)",
        (),
    )
    .expect("Failed to insert data");

    db.execute(
        "CREATE INDEX idx_category ON test_complex_pushdown(category)",
        (),
    )
    .expect("Failed to create index");

    let result = db
        .query("SELECT * FROM test_complex_pushdown WHERE category = 'Electronics' OR category = 'Books' OR in_stock = false", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 8, "Expected 8 rows for three-way OR");
}

/// Test mixed conditions with range query
#[test]
fn test_mixed_with_range() {
    let db = Database::open("memory://mixed_range").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_complex_pushdown (
            id INTEGER,
            category TEXT,
            price FLOAT,
            in_stock BOOLEAN,
            rating INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_complex_pushdown VALUES
        (1, 'Electronics', 499.99, true, 4),
        (2, 'Books', 29.99, true, 5),
        (3, 'Electronics', 1299.99, false, 4),
        (4, 'Clothing', 49.99, true, 3),
        (5, 'Books', 19.99, false, 5),
        (6, 'Electronics', 799.99, true, 2),
        (7, 'Clothing', 89.99, true, 4),
        (8, 'Books', 9.99, true, 3),
        (9, 'Electronics', 999.99, false, 5),
        (10, 'Clothing', 129.99, false, 2)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query("SELECT * FROM test_complex_pushdown WHERE (category = 'Clothing' OR price > 500) AND rating <= 4", ())
        .expect("Query failed");

    let count = result.count();
    assert_eq!(count, 5, "Expected 5 rows for mixed conditions with range");
}
