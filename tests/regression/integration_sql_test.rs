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

//! Integration SQL Tests
//!
//! Comprehensive integration tests for SQL functionality

use stoolap::Database;

/// Test basic aggregations (COUNT, SUM, AVG, MIN, MAX)
#[test]
fn test_basic_aggregations() {
    let db = Database::open("memory://int_agg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales_data (
            id INT PRIMARY KEY,
            product TEXT,
            category TEXT,
            amount FLOAT,
            quantity INT,
            sale_date TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO sales_data VALUES
            (1, 'Laptop', 'Electronics', 1200.50, 2, TIMESTAMP '2024-01-15 10:00:00'),
            (2, 'Mouse', 'Electronics', 25.99, 5, TIMESTAMP '2024-01-15 11:00:00'),
            (3, 'Desk', 'Furniture', 350.00, 1, TIMESTAMP '2024-01-16 09:00:00'),
            (4, 'Chair', 'Furniture', 150.00, 4, TIMESTAMP '2024-01-16 10:00:00'),
            (5, 'Keyboard', 'Electronics', 79.99, 3, TIMESTAMP '2024-01-17 14:00:00')",
        (),
    )
    .expect("Failed to insert data");

    // Test COUNT
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM sales_data", ())
        .expect("Failed to execute COUNT");
    assert_eq!(count, 5, "Expected COUNT(*) = 5");

    // Test SUM
    let total: f64 = db
        .query_one("SELECT SUM(amount) FROM sales_data", ())
        .expect("Failed to execute SUM");
    assert!(
        (total - 1806.48).abs() < 0.01,
        "Expected SUM(amount) = 1806.48, got {}",
        total
    );

    // Test AVG
    let avg: f64 = db
        .query_one("SELECT AVG(amount) FROM sales_data", ())
        .expect("Failed to execute AVG");
    assert!(
        (avg - 361.296).abs() < 0.01,
        "Expected AVG(amount) = 361.296, got {}",
        avg
    );

    // Test MIN/MAX
    let min: f64 = db
        .query_one("SELECT MIN(amount) FROM sales_data", ())
        .expect("Failed to execute MIN");
    let max: f64 = db
        .query_one("SELECT MAX(amount) FROM sales_data", ())
        .expect("Failed to execute MAX");
    assert!(
        (min - 25.99).abs() < 0.01,
        "Expected MIN=25.99, got {}",
        min
    );
    assert!(
        (max - 1200.50).abs() < 0.01,
        "Expected MAX=1200.50, got {}",
        max
    );
}

/// Test GROUP BY operations
#[test]
fn test_group_by_operations() {
    let db = Database::open("memory://int_group").expect("Failed to create database");

    db.execute(
        "CREATE TABLE product_sales (
            id INT PRIMARY KEY,
            category TEXT,
            product TEXT,
            amount FLOAT,
            region TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO product_sales VALUES
            (1, 'Electronics', 'Laptop', 1200.0, 'North'),
            (2, 'Electronics', 'Phone', 800.0, 'North'),
            (3, 'Electronics', 'Tablet', 600.0, 'South'),
            (4, 'Furniture', 'Desk', 400.0, 'North'),
            (5, 'Furniture', 'Chair', 200.0, 'South'),
            (6, 'Furniture', 'Shelf', 150.0, 'South')",
        (),
    )
    .expect("Failed to insert data");

    // Test GROUP BY with aggregation
    let result = db
        .query(
            "SELECT category, COUNT(*) as cnt, SUM(amount) as total
             FROM product_sales
             GROUP BY category
             ORDER BY category",
            (),
        )
        .expect("Failed to execute GROUP BY query");

    let mut results: Vec<(String, i64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        let total: f64 = row.get(2).unwrap();
        results.push((category, count, total));
    }

    assert_eq!(results.len(), 2, "Expected 2 groups");
    assert_eq!(results[0].0, "Electronics");
    assert_eq!(results[0].1, 3);
    assert!((results[0].2 - 2600.0).abs() < 0.01);
    assert_eq!(results[1].0, "Furniture");
    assert_eq!(results[1].1, 3);
    assert!((results[1].2 - 750.0).abs() < 0.01);

    // Test HAVING clause - using alias reference
    let result = db
        .query(
            "SELECT category, SUM(amount) as total
             FROM product_sales
             GROUP BY category
             HAVING total > 1000",
            (),
        )
        .expect("Failed to execute HAVING query");

    let mut having_results: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        having_results.push((category, total));
    }

    assert_eq!(having_results.len(), 1);
    assert_eq!(having_results[0].0, "Electronics");
    assert!((having_results[0].1 - 2600.0).abs() < 0.01);
}

/// Test complex expressions
#[test]
fn test_complex_expressions() {
    let db = Database::open("memory://int_expr").expect("Failed to create database");

    db.execute(
        "CREATE TABLE expr_test (
            id INT PRIMARY KEY,
            a INT,
            b INT,
            c FLOAT,
            d TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO expr_test VALUES
            (1, 10, 5, 2.5, 'hello'),
            (2, 20, 8, 3.5, 'world'),
            (3, 15, 3, 4.5, 'test')",
        (),
    )
    .expect("Failed to insert data");

    // Test arithmetic expressions
    let result: f64 = db
        .query_one("SELECT (a + b) FROM expr_test WHERE id = 1", ())
        .expect("Failed to execute arithmetic expression");
    assert!((result - 15.0).abs() < 0.01, "Expected 15, got {}", result);

    // Test complex WHERE with AND/OR
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM expr_test WHERE (a > 10 AND b < 10) OR c > 4.0",
            (),
        )
        .expect("Failed to execute complex WHERE");
    assert_eq!(count, 2, "Expected 2 rows");
}

/// Test string functions
#[test]
fn test_string_functions() {
    let db = Database::open("memory://int_string").expect("Failed to create database");

    db.execute(
        "CREATE TABLE string_test (
            id INT PRIMARY KEY,
            text1 TEXT,
            text2 TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO string_test VALUES
            (1, 'Hello', 'World'),
            (2, 'STOOLAP', 'Database'),
            (3, 'Test', 'String')",
        (),
    )
    .expect("Failed to insert data");

    // Test UPPER/LOWER
    let result = db
        .query(
            "SELECT UPPER(text1), LOWER(text2) FROM string_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute UPPER/LOWER");
    let mut row_data: Option<(String, String)> = None;
    for row in result {
        let row = row.expect("Failed to get row");
        let upper: String = row.get(0).unwrap();
        let lower: String = row.get(1).unwrap();
        row_data = Some((upper, lower));
    }
    let (upper, lower) = row_data.expect("No rows returned");
    assert_eq!(upper, "HELLO");
    assert_eq!(lower, "world");

    // Test LENGTH
    let length: i64 = db
        .query_one("SELECT LENGTH(text1) FROM string_test WHERE id = 2", ())
        .expect("Failed to execute LENGTH");
    assert_eq!(length, 7, "Expected length 7");

    // Test CONCAT
    let concat: String = db
        .query_one(
            "SELECT CONCAT(text1, ' ', text2) FROM string_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute CONCAT");
    assert_eq!(concat, "Hello World");

    // Test SUBSTRING
    let substr: String = db
        .query_one(
            "SELECT SUBSTRING(text1, 1, 3) FROM string_test WHERE id = 2",
            (),
        )
        .expect("Failed to execute SUBSTRING");
    assert_eq!(substr, "STO");
}

/// Test date/time functions
#[test]
fn test_datetime_functions() {
    let db = Database::open("memory://int_dt").expect("Failed to create database");

    db.execute(
        "CREATE TABLE datetime_test (
            id INT PRIMARY KEY,
            event_time TIMESTAMP
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO datetime_test VALUES
            (1, TIMESTAMP '2024-01-15 14:30:45'),
            (2, TIMESTAMP '2024-02-20 09:15:30'),
            (3, TIMESTAMP '2024-03-25 18:45:00')",
        (),
    )
    .expect("Failed to insert data");

    // Test NOW() function
    let now: String = db
        .query_one("SELECT NOW()", ())
        .expect("Failed to execute NOW()");
    assert!(!now.is_empty(), "NOW() returned empty string");

    // Test DATE_TRUNC
    let truncated: String = db
        .query_one(
            "SELECT DATE_TRUNC('day', event_time) FROM datetime_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute DATE_TRUNC");
    assert!(!truncated.is_empty(), "DATE_TRUNC returned empty string");
}

/// Test CAST operations
#[test]
fn test_cast_operations() {
    let db = Database::open("memory://int_cast").expect("Failed to create database");

    db.execute(
        "CREATE TABLE cast_test (
            id INT PRIMARY KEY,
            int_val INT,
            float_val FLOAT,
            text_val TEXT,
            bool_val BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO cast_test VALUES
            (1, 42, 3.14, '123', true),
            (2, 100, 2.718, '456.78', false),
            (3, 0, 0.0, 'abc', true)",
        (),
    )
    .expect("Failed to insert data");

    // Test INT to FLOAT cast
    let float_result: f64 = db
        .query_one(
            "SELECT CAST(int_val AS FLOAT) FROM cast_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute INT to FLOAT cast");
    assert!((float_result - 42.0).abs() < 0.01, "Expected 42.0");

    // Test TEXT to INT cast
    let int_result: i64 = db
        .query_one(
            "SELECT CAST(text_val AS INTEGER) FROM cast_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute TEXT to INT cast");
    assert_eq!(int_result, 123, "Expected 123");

    // Test FLOAT to TEXT cast
    let text_result: String = db
        .query_one(
            "SELECT CAST(float_val AS TEXT) FROM cast_test WHERE id = 1",
            (),
        )
        .expect("Failed to execute FLOAT to TEXT cast");
    assert!(
        text_result == "3.14" || text_result.starts_with("3.14"),
        "Expected '3.14', got '{}'",
        text_result
    );
}

/// Test NULL handling
#[test]
fn test_null_handling() {
    let db = Database::open("memory://int_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE null_test (
            id INT PRIMARY KEY,
            value INT,
            text_val TEXT,
            float_val FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO null_test VALUES
            (1, 100, 'test', 3.5),
            (2, NULL, NULL, NULL),
            (3, 200, 'value', NULL)",
        (),
    )
    .expect("Failed to insert data");

    // Test IS NULL
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM null_test WHERE value IS NULL", ())
        .expect("Failed to execute IS NULL");
    assert_eq!(count, 1, "Expected 1 NULL value");

    // Test IS NOT NULL
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM null_test WHERE text_val IS NOT NULL",
            (),
        )
        .expect("Failed to execute IS NOT NULL");
    assert_eq!(count, 2, "Expected 2 non-NULL text values");

    // Test COALESCE
    let result: i64 = db
        .query_one(
            "SELECT COALESCE(value, 999) FROM null_test WHERE id = 2",
            (),
        )
        .expect("Failed to execute COALESCE");
    assert_eq!(result, 999, "Expected 999 from COALESCE");

    // Test aggregates with NULL - AVG should only consider non-NULL values
    let avg: f64 = db
        .query_one("SELECT AVG(float_val) FROM null_test", ())
        .expect("Failed to execute AVG with NULL");
    assert!((avg - 3.5).abs() < 0.01, "Expected AVG=3.5, got {}", avg);
}

/// Test transaction support
#[test]
fn test_transaction_support() {
    let db = Database::open("memory://int_tx").expect("Failed to create database");

    db.execute(
        "CREATE TABLE tx_test (
            id INT PRIMARY KEY,
            value INT
        )",
        (),
    )
    .expect("Failed to create table");

    // Test successful transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("INSERT INTO tx_test VALUES (1, 100)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO tx_test VALUES (2, 200)", ())
        .expect("Failed to insert");
    db.execute("COMMIT", ()).expect("Failed to commit");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tx_test", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows after commit");

    // Test rollback
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("INSERT INTO tx_test VALUES (3, 300)", ())
        .expect("Failed to insert");
    db.execute("ROLLBACK", ()).expect("Failed to rollback");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tx_test", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows after rollback");
}

/// Test simple JOINs
#[test]
fn test_simple_joins() {
    let db = Database::open("memory://int_join").expect("Failed to create database");

    db.execute(
        "CREATE TABLE departments (
            id INT PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create departments table");

    db.execute(
        "CREATE TABLE employees (
            id INT PRIMARY KEY,
            name TEXT,
            dept_id INT
        )",
        (),
    )
    .expect("Failed to create employees table");

    db.execute(
        "INSERT INTO departments VALUES
            (1, 'Engineering'),
            (2, 'Sales'),
            (3, 'Marketing')",
        (),
    )
    .expect("Failed to insert departments");

    db.execute(
        "INSERT INTO employees VALUES
            (1, 'Alice', 1),
            (2, 'Bob', 1),
            (3, 'Charlie', 2),
            (4, 'David', NULL)",
        (),
    )
    .expect("Failed to insert employees");

    // Test INNER JOIN
    let result = db
        .query(
            "SELECT e.name, d.name
             FROM employees e
             INNER JOIN departments d ON e.dept_id = d.id
             ORDER BY e.name",
            (),
        )
        .expect("Failed to execute INNER JOIN");

    let mut inner_join_count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        inner_join_count += 1;
    }
    assert_eq!(
        inner_join_count, 3,
        "Expected 3 INNER JOIN results, got {}",
        inner_join_count
    );

    // Test LEFT JOIN
    let result = db
        .query(
            "SELECT e.name, d.name
             FROM employees e
             LEFT JOIN departments d ON e.dept_id = d.id
             ORDER BY e.name",
            (),
        )
        .expect("Failed to execute LEFT JOIN");

    let mut left_join_count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        left_join_count += 1;
    }
    assert_eq!(
        left_join_count, 4,
        "Expected 4 LEFT JOIN results, got {}",
        left_join_count
    );
}

/// Test DISTINCT
#[test]
fn test_distinct() {
    let db = Database::open("memory://int_distinct").expect("Failed to create database");

    db.execute(
        "CREATE TABLE distinct_test (
            id INT PRIMARY KEY,
            category TEXT,
            value INT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO distinct_test VALUES
            (1, 'A', 10),
            (2, 'B', 20),
            (3, 'A', 30),
            (4, 'C', 40),
            (5, 'B', 50)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM distinct_test", ())
        .expect("Failed to execute DISTINCT");
    assert_eq!(count, 3, "Expected 3 distinct categories");
}

/// Test LIMIT and OFFSET
#[test]
fn test_limit_offset() {
    let db = Database::open("memory://int_limit").expect("Failed to create database");

    db.execute(
        "CREATE TABLE limit_test (
            id INT PRIMARY KEY,
            value INT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO limit_test VALUES
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Failed to insert data");

    // Test LIMIT
    let result = db
        .query("SELECT id FROM limit_test ORDER BY id LIMIT 3", ())
        .expect("Failed to execute LIMIT");
    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 3, "Expected 3 rows with LIMIT 3");

    // Test LIMIT with OFFSET
    let result = db
        .query("SELECT id FROM limit_test ORDER BY id LIMIT 2 OFFSET 2", ())
        .expect("Failed to execute LIMIT OFFSET");
    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    assert_eq!(ids, vec![3, 4], "Expected ids 3, 4 with LIMIT 2 OFFSET 2");
}

/// Test IN clause
#[test]
fn test_in_clause() {
    let db = Database::open("memory://int_in").expect("Failed to create database");

    db.execute(
        "CREATE TABLE in_test (
            id INT PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO in_test VALUES
            (1, 'Alice'), (2, 'Bob'), (3, 'Charlie'), (4, 'David')",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM in_test WHERE name IN ('Alice', 'Charlie')",
            (),
        )
        .expect("Failed to execute IN");
    assert_eq!(count, 2, "Expected 2 rows with IN clause");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM in_test WHERE id NOT IN (1, 2)", ())
        .expect("Failed to execute NOT IN");
    assert_eq!(count, 2, "Expected 2 rows with NOT IN clause");
}

/// Test LIKE clause
#[test]
fn test_like_clause() {
    let db = Database::open("memory://int_like").expect("Failed to create database");

    db.execute(
        "CREATE TABLE like_test (
            id INT PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO like_test VALUES
            (1, 'Apple'), (2, 'Banana'), (3, 'Apricot'), (4, 'Avocado')",
        (),
    )
    .expect("Failed to insert data");

    // Test starts with
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM like_test WHERE name LIKE 'A%'", ())
        .expect("Failed to execute LIKE");
    assert_eq!(count, 3, "Expected 3 rows starting with A");

    // Test ends with
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM like_test WHERE name LIKE '%a'", ())
        .expect("Failed to execute LIKE");
    assert_eq!(count, 1, "Expected 1 row ending with a");

    // Test contains
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM like_test WHERE name LIKE '%an%'", ())
        .expect("Failed to execute LIKE");
    assert_eq!(count, 1, "Expected 1 row containing 'an'");
}

/// Test BETWEEN clause
#[test]
fn test_between_clause() {
    let db = Database::open("memory://int_between").expect("Failed to create database");

    db.execute(
        "CREATE TABLE between_test (
            id INT PRIMARY KEY,
            value INT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO between_test VALUES
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Failed to insert data");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM between_test WHERE value BETWEEN 20 AND 40",
            (),
        )
        .expect("Failed to execute BETWEEN");
    assert_eq!(count, 3, "Expected 3 rows between 20 and 40");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM between_test WHERE value NOT BETWEEN 20 AND 40",
            (),
        )
        .expect("Failed to execute NOT BETWEEN");
    assert_eq!(count, 2, "Expected 2 rows not between 20 and 40");
}

/// Test ORDER BY with multiple columns
#[test]
fn test_order_by_multiple() {
    let db = Database::open("memory://int_order").expect("Failed to create database");

    db.execute(
        "CREATE TABLE order_test (
            id INT PRIMARY KEY,
            category TEXT,
            value INT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO order_test VALUES
            (1, 'A', 30),
            (2, 'B', 20),
            (3, 'A', 10),
            (4, 'B', 40),
            (5, 'A', 20)",
        (),
    )
    .expect("Failed to insert data");

    let result = db
        .query(
            "SELECT id, category, value FROM order_test ORDER BY category, value DESC",
            (),
        )
        .expect("Failed to execute ORDER BY");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    // A with values 30, 20, 10 should be 1, 5, 3 (descending)
    // B with values 40, 20 should be 4, 2 (descending)
    assert_eq!(ids, vec![1, 5, 3, 4, 2], "Unexpected order");
}

/// Test DEFAULT values in CREATE TABLE
#[test]
fn test_default_values() {
    let db = Database::open("memory://int_default").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT 'No description',
            price FLOAT DEFAULT 0.0,
            quantity INTEGER DEFAULT 1,
            active BOOLEAN DEFAULT true
        )",
        (),
    )
    .expect("Failed to create table with defaults");

    // Insert with only required columns
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')", ())
        .expect("Failed to insert with defaults");

    // Insert with some defaults overridden
    db.execute(
        "INSERT INTO products (id, name, price, quantity) VALUES (2, 'Gadget', 29.99, 5)",
        (),
    )
    .expect("Failed to insert with partial defaults");

    // Insert with all columns specified
    db.execute(
        "INSERT INTO products VALUES (3, 'Thing', 'A useful thing', 19.99, 10, false)",
        (),
    )
    .expect("Failed to insert with all values");

    // Verify defaults were applied correctly
    let result = db
        .query("SELECT * FROM products ORDER BY id", ())
        .expect("Failed to query");

    let mut rows_data: Vec<(i64, String, String, f64, i64, bool)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        rows_data.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
            row.get(4).unwrap(),
            row.get(5).unwrap(),
        ));
    }

    assert_eq!(rows_data.len(), 3);

    // Row 1: all defaults except id and name
    assert_eq!(rows_data[0].0, 1);
    assert_eq!(rows_data[0].1, "Widget");
    assert_eq!(rows_data[0].2, "No description");
    assert_eq!(rows_data[0].3, 0.0);
    assert_eq!(rows_data[0].4, 1);
    assert!(rows_data[0].5);

    // Row 2: some defaults
    assert_eq!(rows_data[1].0, 2);
    assert_eq!(rows_data[1].1, "Gadget");
    assert_eq!(rows_data[1].2, "No description");
    assert_eq!(rows_data[1].3, 29.99);
    assert_eq!(rows_data[1].4, 5);
    assert!(rows_data[1].5);

    // Row 3: no defaults
    assert_eq!(rows_data[2].0, 3);
    assert_eq!(rows_data[2].1, "Thing");
    assert_eq!(rows_data[2].2, "A useful thing");
    assert_eq!(rows_data[2].3, 19.99);
    assert_eq!(rows_data[2].4, 10);
    assert!(!rows_data[2].5);
}

/// Test CHECK constraints in CREATE TABLE
#[test]
fn test_check_constraints() {
    let db = Database::open("memory://int_check").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER CHECK (age >= 18 AND age <= 100),
            salary FLOAT CHECK (salary > 0),
            department TEXT
        )",
        (),
    )
    .expect("Failed to create table with CHECK constraints");

    // Valid insert
    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 30, 75000.0, 'Engineering')",
        (),
    )
    .expect("Valid insert should succeed");

    // Invalid age (too young)
    let result = db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 15, 50000.0, 'Sales')",
        (),
    );
    assert!(result.is_err(), "Age < 18 should fail CHECK constraint");

    // Invalid age (too old)
    let result = db.execute(
        "INSERT INTO employees VALUES (3, 'Charlie', 150, 60000.0, 'HR')",
        (),
    );
    assert!(result.is_err(), "Age > 100 should fail CHECK constraint");

    // Invalid salary (zero)
    let result = db.execute(
        "INSERT INTO employees VALUES (4, 'David', 25, 0, 'Marketing')",
        (),
    );
    assert!(result.is_err(), "Salary = 0 should fail CHECK constraint");

    // Invalid salary (negative)
    let result = db.execute(
        "INSERT INTO employees VALUES (5, 'Eve', 28, -1000.0, 'Finance')",
        (),
    );
    assert!(
        result.is_err(),
        "Negative salary should fail CHECK constraint"
    );

    // Verify only valid row exists
    let result = db
        .query("SELECT COUNT(*) FROM employees", ())
        .expect("Failed to count");
    let mut count = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        count = row.get(0).unwrap();
    }
    assert_eq!(count, 1, "Only one valid row should exist");
}

/// Test UNIQUE constraints in CREATE TABLE
#[test]
fn test_unique_constraints() {
    let db = Database::open("memory://int_unique").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table with UNIQUE constraints");

    // First user
    db.execute(
        "INSERT INTO users VALUES (1, 'alice', 'alice@example.com', 'Alice Smith')",
        (),
    )
    .expect("First insert should succeed");

    // Second user with different unique values
    db.execute(
        "INSERT INTO users VALUES (2, 'bob', 'bob@example.com', 'Bob Jones')",
        (),
    )
    .expect("Second insert with different values should succeed");

    // Duplicate username
    let result = db.execute(
        "INSERT INTO users VALUES (3, 'alice', 'charlie@example.com', 'Charlie Brown')",
        (),
    );
    assert!(
        result.is_err(),
        "Duplicate username should fail UNIQUE constraint"
    );

    // Duplicate email
    let result = db.execute(
        "INSERT INTO users VALUES (4, 'david', 'alice@example.com', 'David Wilson')",
        (),
    );
    assert!(
        result.is_err(),
        "Duplicate email should fail UNIQUE constraint"
    );

    // Verify only valid rows exist
    let result = db
        .query("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    let mut count = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        count = row.get(0).unwrap();
    }
    assert_eq!(count, 2, "Only two valid rows should exist");
}

/// Test CREATE TABLE AS SELECT
#[test]
fn test_create_table_as_select() {
    let db = Database::open("memory://int_ctas").expect("Failed to create database");

    // Create source table
    db.execute(
        "CREATE TABLE source (
            id INTEGER PRIMARY KEY,
            category TEXT,
            value FLOAT
        )",
        (),
    )
    .expect("Failed to create source table");

    db.execute(
        "INSERT INTO source VALUES
            (1, 'A', 100.0),
            (2, 'B', 200.0),
            (3, 'A', 150.0),
            (4, 'C', 300.0),
            (5, 'B', 250.0)",
        (),
    )
    .expect("Failed to insert data");

    // Create table from SELECT *
    db.execute("CREATE TABLE copy AS SELECT * FROM source", ())
        .expect("Failed to create table from SELECT *");

    let result = db
        .query("SELECT COUNT(*) FROM copy", ())
        .expect("Failed to count copy");
    let mut count = 0i64;
    for row in result {
        count = row.expect("Failed to get row").get(0).unwrap();
    }
    assert_eq!(count, 5, "Copy should have 5 rows");

    // Create table with filter
    db.execute(
        "CREATE TABLE filtered AS SELECT id, category, value FROM source WHERE value > 150.0",
        (),
    )
    .expect("Failed to create filtered table");

    let result = db
        .query("SELECT COUNT(*) FROM filtered", ())
        .expect("Failed to count filtered");
    for row in result {
        count = row.expect("Failed to get row").get(0).unwrap();
    }
    assert_eq!(count, 3, "Filtered should have 3 rows (values > 150)");

    // Create table with aggregation
    db.execute(
        "CREATE TABLE summary AS SELECT category, COUNT(*) as cnt, SUM(value) as total FROM source GROUP BY category",
        (),
    )
    .expect("Failed to create summary table");

    let result = db
        .query("SELECT * FROM summary ORDER BY category", ())
        .expect("Failed to query summary");
    let mut rows: Vec<(String, i64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        rows.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
        ));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("A".to_string(), 2, 250.0));
    assert_eq!(rows[1], ("B".to_string(), 2, 450.0));
    assert_eq!(rows[2], ("C".to_string(), 1, 300.0));
}

/// Test JOIN USING clause
#[test]
fn test_join_using() {
    let db = Database::open("memory://int_using").expect("Failed to create database");

    db.execute(
        "CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create departments table");

    db.execute(
        "CREATE TABLE employees_join (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER
        )",
        (),
    )
    .expect("Failed to create employees table");

    db.execute(
        "INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales'), (3, 'HR')",
        (),
    )
    .expect("Failed to insert departments");

    db.execute(
        "INSERT INTO employees_join VALUES (1, 'Alice', 1), (2, 'Bob', 2), (3, 'Charlie', 1), (4, 'David', 4)",
        (),
    )
    .expect("Failed to insert employees");

    // Create a view-like table to test USING with same column name
    db.execute(
        "CREATE TABLE dept_emp (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT
        )",
        (),
    )
    .expect("Failed to create dept_emp table");

    db.execute(
        "CREATE TABLE emp_dept (
            emp_id INTEGER PRIMARY KEY,
            emp_name TEXT,
            dept_id INTEGER
        )",
        (),
    )
    .expect("Failed to create emp_dept table");

    db.execute("INSERT INTO dept_emp VALUES (1, 'Eng'), (2, 'Sales')", ())
        .expect("Failed to insert dept_emp");

    db.execute(
        "INSERT INTO emp_dept VALUES (1, 'Alice', 1), (2, 'Bob', 2), (3, 'Charlie', 1)",
        (),
    )
    .expect("Failed to insert emp_dept");

    // Test INNER JOIN USING
    let result = db
        .query(
            "SELECT emp_name, dept_name FROM emp_dept JOIN dept_emp USING (dept_id) ORDER BY emp_name",
            (),
        )
        .expect("Failed to execute JOIN USING");

    let mut names: Vec<(String, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        names.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(names.len(), 3);
    assert_eq!(names[0], ("Alice".to_string(), "Eng".to_string()));
    assert_eq!(names[1], ("Bob".to_string(), "Sales".to_string()));
    assert_eq!(names[2], ("Charlie".to_string(), "Eng".to_string()));

    // Test LEFT JOIN USING
    db.execute("INSERT INTO emp_dept VALUES (4, 'David', 99)", ())
        .expect("Failed to insert David");

    let result = db
        .query(
            "SELECT emp_name FROM emp_dept LEFT JOIN dept_emp USING (dept_id) ORDER BY emp_name",
            (),
        )
        .expect("Failed to execute LEFT JOIN USING");

    let mut emp_names: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        emp_names.push(row.get(0).unwrap());
    }
    assert_eq!(emp_names.len(), 4, "LEFT JOIN should include David");
    assert!(emp_names.contains(&"David".to_string()));
}
