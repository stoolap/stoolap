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

// Regression tests for Bug Batch 6
// Tests for bugs #37, #38, #39 discovered during exploratory testing

use std::io::Write;
use stoolap::Database;
use tempfile::NamedTempFile;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG 37: -- line comment after semicolon breaks subsequent statements
// Problem: When reading SQL from file/stdin, line comments after semicolons
//          caused subsequent statements to be swallowed
// =============================================================================

#[test]
fn test_bugs6_line_comment_after_semicolon_piped() {
    let db = setup_db();

    // Create a temp file with SQL statements including line comments
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(
        file,
        "CREATE TABLE t37 (id INTEGER PRIMARY KEY, val INTEGER);"
    )
    .unwrap();
    writeln!(file, "INSERT INTO t37 VALUES (1, 100); -- trailing comment").unwrap();
    writeln!(file, "INSERT INTO t37 VALUES (2, 200);").unwrap();
    writeln!(file, "INSERT INTO t37 VALUES (3, 300); -- another comment").unwrap();
    file.flush().unwrap();

    // Read and execute the file content
    let sql = std::fs::read_to_string(file.path()).expect("Failed to read temp file");

    // Execute statements one by one (simulating what stoolap CLI does)
    for stmt in sql.lines().filter(|l| !l.trim().is_empty()) {
        db.execute(stmt, ())
            .unwrap_or_else(|_| panic!("Failed to execute: {}", stmt));
    }

    // Verify all inserts worked
    let mut rows = db
        .query("SELECT COUNT(*) FROM t37", ())
        .expect("Query failed");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
}

#[test]
fn test_bugs6_empty_line_comment_after_semicolon() {
    let db = setup_db();

    // Test with empty line comment (just --)
    db.execute(
        "CREATE TABLE t37b (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // These should all work even with empty line comments
    db.execute("INSERT INTO t37b VALUES (1, 100)", ())
        .expect("Insert 1 failed");
    db.execute("INSERT INTO t37b VALUES (2, 200)", ())
        .expect("Insert 2 failed");

    let mut rows = db
        .query("SELECT COUNT(*) FROM t37b", ())
        .expect("Query failed");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
}

// =============================================================================
// BUG 38: Aggregate expression returns NULL when same aggregate has alias
// Problem: SELECT SUM(val), SUM(val) AS total returned (30, NULL)
//          The second occurrence returned NULL instead of the computed value
// =============================================================================

#[test]
fn test_bugs6_aggregate_with_alias_second() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t38 (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t38 VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30), (4, 'B', 40)",
        (),
    )
    .expect("Failed to insert data");

    // First without alias, second with alias - both should return correct value
    let mut rows = db
        .query(
            "SELECT category, SUM(val), SUM(val) AS total FROM t38 GROUP BY category ORDER BY category",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30); // SUM(val) without alias
    assert_eq!(row.get::<i64>(2).unwrap(), 30); // SUM(val) AS total

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 70);
    assert_eq!(row.get::<i64>(2).unwrap(), 70);
}

#[test]
fn test_bugs6_aggregate_with_alias_first() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t38b (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t38b VALUES (1, 'X', 100), (2, 'X', 200), (3, 'Y', 300)",
        (),
    )
    .expect("Failed to insert data");

    // First with alias, second without - both should return correct value
    let mut rows = db
        .query(
            "SELECT category, SUM(val) AS total, SUM(val) FROM t38b GROUP BY category ORDER BY category",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "X");
    assert_eq!(row.get::<i64>(1).unwrap(), 300); // SUM(val) AS total
    assert_eq!(row.get::<i64>(2).unwrap(), 300); // SUM(val) without alias

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "Y");
    assert_eq!(row.get::<i64>(1).unwrap(), 300);
    assert_eq!(row.get::<i64>(2).unwrap(), 300);
}

#[test]
fn test_bugs6_triple_same_aggregate() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t38c (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t38c VALUES (1, 'A', 10), (2, 'A', 20)", ())
        .expect("Failed to insert data");

    // Three occurrences: no alias, alias1, alias2
    let mut rows = db
        .query(
            "SELECT category, SUM(val), SUM(val) AS total1, SUM(val) AS total2 FROM t38c GROUP BY category",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30); // SUM(val)
    assert_eq!(row.get::<i64>(2).unwrap(), 30); // SUM(val) AS total1
    assert_eq!(row.get::<i64>(3).unwrap(), 30); // SUM(val) AS total2
}

// =============================================================================
// BUG 39: ORDER BY fails when joining CTEs that contain GROUP BY
// Problem: Joining two CTEs with aggregation, then ORDER BY, returned
//          unsorted results
// =============================================================================

#[test]
fn test_bugs6_cte_join_with_group_by_order_by() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t39 (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t39 VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30), (4, 'B', 40), (5, 'C', 50)",
        (),
    )
    .expect("Failed to insert data");

    // Join two CTEs with GROUP BY, then ORDER BY
    let mut rows = db
        .query(
            "WITH
                cte1 AS (SELECT category, SUM(val) as s FROM t39 GROUP BY category),
                cte2 AS (SELECT category, COUNT(*) as c FROM t39 GROUP BY category)
            SELECT cte1.category, s, c
            FROM cte1
            JOIN cte2 ON cte1.category = cte2.category
            ORDER BY cte1.category",
            (),
        )
        .expect("Query failed");

    // Should be sorted: A, B, C
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30); // SUM for A
    assert_eq!(row.get::<i64>(2).unwrap(), 2); // COUNT for A

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 70); // SUM for B
    assert_eq!(row.get::<i64>(2).unwrap(), 2); // COUNT for B

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 50); // SUM for C
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // COUNT for C
}

#[test]
fn test_bugs6_cte_join_order_by_desc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t39b (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t39b VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30), (4, 'B', 40), (5, 'C', 50)",
        (),
    )
    .expect("Failed to insert data");

    // Join two CTEs with GROUP BY, then ORDER BY aggregate DESC
    let mut rows = db
        .query(
            "WITH
                cte1 AS (SELECT category, SUM(val) as s FROM t39b GROUP BY category),
                cte2 AS (SELECT category, COUNT(*) as c FROM t39b GROUP BY category)
            SELECT cte1.category, s, c
            FROM cte1
            JOIN cte2 ON cte1.category = cte2.category
            ORDER BY s DESC",
            (),
        )
        .expect("Query failed");

    // Should be sorted by s DESC: B=70, C=50, A=30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 70);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 50);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30);
}

#[test]
fn test_bugs6_cte_join_no_aggregation_order_by() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t39c (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t39c VALUES (1, 'Alice', 100), (2, 'Bob', 200), (3, 'Charlie', 50)",
        (),
    )
    .expect("Failed to insert data");

    // Join two CTEs without aggregation, then ORDER BY
    let mut rows = db
        .query(
            "WITH
                cte1 AS (SELECT id, name FROM t39c WHERE value > 75),
                cte2 AS (SELECT id, value FROM t39c WHERE value > 75)
            SELECT cte1.id, name, value
            FROM cte1
            JOIN cte2 ON cte1.id = cte2.id
            ORDER BY name",
            (),
        )
        .expect("Query failed");

    // Should be sorted by name: Alice, Bob (Charlie filtered out)
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "Alice");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "Bob");

    // No more rows
    assert!(rows.next().is_none());
}

// =============================================================================
// Additional edge case tests
// =============================================================================

#[test]
fn test_bugs6_single_cte_with_order_by_still_works() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t39d (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t39d VALUES (1, 'A', 30), (2, 'B', 10), (3, 'C', 20)",
        (),
    )
    .expect("Failed to insert data");

    // Single CTE with ORDER BY should still work
    let mut rows = db
        .query(
            "WITH cte AS (SELECT category, SUM(val) as s FROM t39d GROUP BY category)
            SELECT category, s FROM cte ORDER BY s",
            (),
        )
        .expect("Query failed");

    // Should be sorted by s ASC: B=10, C=20, A=30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 10);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 20);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30);
}

// =============================================================================
// BUG #40: Non-numeric string to INTEGER PRIMARY KEY silently uses auto-increment
// Problem: INSERT INTO t VALUES ('abc', ...) where 'abc' can't convert to INTEGER
//          should fail, not silently auto-increment
// =============================================================================

#[test]
fn test_bugs6_non_numeric_string_to_integer_pk_fails() {
    let db = setup_db();

    db.execute("CREATE TABLE t40 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t40 VALUES (1, 'one')", ())
        .expect("Insert 1 should work");

    // This should fail - 'abc' cannot be converted to INTEGER
    let result = db.execute("INSERT INTO t40 VALUES ('abc', 'two')", ());
    assert!(
        result.is_err(),
        "Should fail when inserting non-numeric string to INTEGER PK"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("type error") || err_msg.contains("cannot convert"),
        "Error should mention type conversion failure: {}",
        err_msg
    );

    // Verify only one row was inserted
    let mut rows = db
        .query("SELECT COUNT(*) FROM t40", ())
        .expect("Query failed");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
}

#[test]
fn test_bugs6_numeric_string_to_integer_pk_works() {
    let db = setup_db();

    db.execute("CREATE TABLE t40b (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    // Numeric string should work fine
    db.execute("INSERT INTO t40b VALUES ('123', 'numeric string')", ())
        .expect("Insert with numeric string should work");

    let mut rows = db
        .query("SELECT id, name FROM t40b", ())
        .expect("Query failed");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 123);
    assert_eq!(row.get::<String>(1).unwrap(), "numeric string");
}

// =============================================================================
// BUG #41: Date arithmetic with subtraction operator fails
// Problem: d - 30 fails while DATE_ADD(d, -30) works
// =============================================================================

#[test]
fn test_bugs6_date_arithmetic_with_integer() {
    let db = setup_db();

    db.execute("CREATE TABLE t41 (id INTEGER PRIMARY KEY, d DATE)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t41 VALUES (1, '2024-06-15')", ())
        .expect("Failed to insert data");

    // Addition with integer (days)
    let mut rows = db
        .query("SELECT d + 30 FROM t41", ())
        .expect("Date addition should work");
    let row = rows.next().unwrap().unwrap();
    let result = row.get::<String>(0).unwrap();
    assert!(
        result.contains("2024-07-15"),
        "Expected 2024-07-15, got {}",
        result
    );

    // Subtraction with integer (days)
    let mut rows = db
        .query("SELECT d - 30 FROM t41", ())
        .expect("Date subtraction should work");
    let row = rows.next().unwrap().unwrap();
    let result = row.get::<String>(0).unwrap();
    assert!(
        result.contains("2024-05-16"),
        "Expected 2024-05-16, got {}",
        result
    );
}

#[test]
fn test_bugs6_date_arithmetic_consistency() {
    let db = setup_db();

    db.execute("CREATE TABLE t41b (id INTEGER PRIMARY KEY, d DATE)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t41b VALUES (1, '2024-06-15')", ())
        .expect("Failed to insert data");

    // DATE_ADD and + operator should give same result
    let mut rows = db
        .query("SELECT DATE_ADD(d, 30), d + 30 FROM t41b", ())
        .expect("Query should work");
    let row = rows.next().unwrap().unwrap();
    let date_add_result = row.get::<String>(0).unwrap();
    let plus_result = row.get::<String>(1).unwrap();
    assert_eq!(
        date_add_result, plus_result,
        "DATE_ADD and + should give same result"
    );

    // DATE_ADD with negative and - operator should give same result
    let mut rows = db
        .query("SELECT DATE_ADD(d, -30), d - 30 FROM t41b", ())
        .expect("Query should work");
    let row = rows.next().unwrap().unwrap();
    let date_add_result = row.get::<String>(0).unwrap();
    let minus_result = row.get::<String>(1).unwrap();
    assert_eq!(
        date_add_result, minus_result,
        "DATE_ADD(-n) and - should give same result"
    );
}

// =============================================================================
// BUG #23: Double negation --val causes parse error
// Problem: --val was being treated as a line comment instead of two minus operators
// =============================================================================

#[test]
fn test_bugs6_double_negation() {
    let db = setup_db();

    db.execute("CREATE TABLE t23 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t23 VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // Double negation should work
    let mut rows = db
        .query(
            "SELECT id, val, -val as neg, --val as double_neg FROM t23",
            (),
        )
        .expect("Double negation query should work");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 10);
    assert_eq!(row.get::<i64>(2).unwrap(), -10);
    assert_eq!(row.get::<i64>(3).unwrap(), 10);
}

#[test]
fn test_bugs6_double_negation_with_literal() {
    let db = setup_db();

    // --5 should evaluate to 5
    let mut rows = db
        .query("SELECT --5", ())
        .expect("Double negation of literal should work");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);

    // --10 + 5 should evaluate to 15
    let mut rows = db
        .query("SELECT --10 + 5", ())
        .expect("Double negation expression should work");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 15);
}

#[test]
fn test_bugs6_real_comment_still_works() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t23b (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Real comments (-- with space) should still work
    db.execute("INSERT INTO t23b VALUES (1, 100) -- this is a comment", ())
        .expect("Statement with trailing comment should work");

    let mut rows = db.query("SELECT val FROM t23b", ()).expect("Query failed");
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 100);
}
