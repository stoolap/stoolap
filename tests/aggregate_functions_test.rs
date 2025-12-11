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

//! Integration tests for new aggregate functions:
//! STRING_AGG, GROUP_CONCAT, STDDEV, VARIANCE, MEDIAN

use stoolap::Database;

fn setup_numbers_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).expect("Failed to create database");

    // Create test table with numeric data
    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value FLOAT, category TEXT)",
        (),
    )
    .unwrap();

    // Insert test data
    // Category A: values 2, 4, 4, 4, 5, 5, 7, 9 (mean=5, pop_var=4, pop_stddev=2)
    db.execute("INSERT INTO numbers VALUES (1, 2, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (2, 4, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (3, 4, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (4, 4, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (5, 5, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (6, 5, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (7, 7, 'A')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (8, 9, 'A')", ())
        .unwrap();

    // Category B: values 1, 2, 3, 4, 5 (mean=3, median=3)
    db.execute("INSERT INTO numbers VALUES (9, 1, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (10, 2, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (11, 3, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (12, 4, 'B')", ())
        .unwrap();
    db.execute("INSERT INTO numbers VALUES (13, 5, 'B')", ())
        .unwrap();

    db
}

fn setup_names_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).expect("Failed to create database");

    db.execute(
        "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, dept TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO people VALUES (1, 'Alice', 'Engineering')", ())
        .unwrap();
    db.execute("INSERT INTO people VALUES (2, 'Bob', 'Engineering')", ())
        .unwrap();
    db.execute("INSERT INTO people VALUES (3, 'Charlie', 'Sales')", ())
        .unwrap();
    db.execute("INSERT INTO people VALUES (4, 'Diana', 'Engineering')", ())
        .unwrap();
    db.execute("INSERT INTO people VALUES (5, 'Eve', 'Sales')", ())
        .unwrap();

    db
}

// ============================================================================
// STRING_AGG / GROUP_CONCAT Tests
// ============================================================================

#[test]
fn test_string_agg_basic() {
    let db = setup_names_db("string_agg_basic");

    let result: String = db
        .query_one("SELECT STRING_AGG(name) FROM people", ())
        .unwrap();

    // Should contain all names separated by commas
    assert!(result.contains("Alice"));
    assert!(result.contains("Bob"));
    assert!(result.contains("Charlie"));
    assert!(result.contains(","));
}

#[test]
fn test_string_agg_with_group_by() {
    let db = setup_names_db("string_agg_group");

    let result = db
        .query(
            "SELECT dept, STRING_AGG(name) FROM people GROUP BY dept ORDER BY dept",
            (),
        )
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);

    // Engineering: Alice, Bob, Diana
    let dept: String = rows[0].get(0).unwrap();
    let names: String = rows[0].get(1).unwrap();
    assert_eq!(dept, "Engineering");
    assert!(names.contains("Alice"));
    assert!(names.contains("Bob"));
    assert!(names.contains("Diana"));

    // Sales: Charlie, Eve
    let dept2: String = rows[1].get(0).unwrap();
    let names2: String = rows[1].get(1).unwrap();
    assert_eq!(dept2, "Sales");
    assert!(names2.contains("Charlie"));
    assert!(names2.contains("Eve"));
}

#[test]
fn test_group_concat() {
    let db = setup_names_db("group_concat");

    let result: String = db
        .query_one(
            "SELECT GROUP_CONCAT(name) FROM people WHERE dept = 'Sales'",
            (),
        )
        .unwrap();

    assert!(result.contains("Charlie"));
    assert!(result.contains("Eve"));
}

#[test]
fn test_string_agg_with_numbers() {
    let db = setup_numbers_db("string_agg_nums");

    let result: String = db
        .query_one(
            "SELECT STRING_AGG(id) FROM numbers WHERE category = 'B'",
            (),
        )
        .unwrap();

    // IDs 9, 10, 11, 12, 13
    assert!(result.contains("9"));
    assert!(result.contains("10"));
    assert!(result.contains(","));
}

// ============================================================================
// STDDEV Tests
// ============================================================================

#[test]
fn test_stddev_pop() {
    let db = setup_numbers_db("stddev_pop");

    // Category A: values 2,4,4,4,5,5,7,9 have stddev_pop = 2.0
    let stddev: f64 = db
        .query_one(
            "SELECT STDDEV_POP(value) FROM numbers WHERE category = 'A'",
            (),
        )
        .unwrap();

    assert!(
        (stddev - 2.0).abs() < 0.0001,
        "Expected 2.0, got {}",
        stddev
    );
}

#[test]
fn test_stddev_alias() {
    let db = setup_numbers_db("stddev_alias");

    // STDDEV should be alias for STDDEV_POP
    let stddev: f64 = db
        .query_one("SELECT STDDEV(value) FROM numbers WHERE category = 'A'", ())
        .unwrap();

    assert!((stddev - 2.0).abs() < 0.0001);
}

#[test]
fn test_stddev_samp() {
    let db = setup_numbers_db("stddev_samp");

    // Sample stddev uses N-1 denominator (Bessel's correction)
    // For values 2,4,4,4,5,5,7,9: sample_var = 32/7 ≈ 4.571, sample_stddev ≈ 2.138
    let stddev: f64 = db
        .query_one(
            "SELECT STDDEV_SAMP(value) FROM numbers WHERE category = 'A'",
            (),
        )
        .unwrap();

    assert!(
        (stddev - 2.138).abs() < 0.01,
        "Expected ~2.138, got {}",
        stddev
    );
}

#[test]
fn test_stddev_with_group_by() {
    let db = setup_numbers_db("stddev_group");

    let result = db
        .query(
            "SELECT category, STDDEV_POP(value) FROM numbers GROUP BY category ORDER BY category",
            (),
        )
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);

    // Category A stddev = 2.0
    let cat_a: String = rows[0].get(0).unwrap();
    let stddev_a: f64 = rows[0].get(1).unwrap();
    assert_eq!(cat_a, "A");
    assert!((stddev_a - 2.0).abs() < 0.0001);

    // Category B: 1,2,3,4,5 -> mean=3, var=2, stddev=sqrt(2)≈1.414
    let cat_b: String = rows[1].get(0).unwrap();
    let stddev_b: f64 = rows[1].get(1).unwrap();
    assert_eq!(cat_b, "B");
    assert!((stddev_b - 1.414).abs() < 0.01);
}

// ============================================================================
// VARIANCE Tests
// ============================================================================

#[test]
fn test_var_pop() {
    let db = setup_numbers_db("var_pop");

    // Category A: variance = 4.0
    let var: f64 = db
        .query_one(
            "SELECT VAR_POP(value) FROM numbers WHERE category = 'A'",
            (),
        )
        .unwrap();

    assert!((var - 4.0).abs() < 0.0001, "Expected 4.0, got {}", var);
}

#[test]
fn test_variance_alias() {
    let db = setup_numbers_db("variance_alias");

    // VARIANCE should be alias for VAR_POP
    let var: f64 = db
        .query_one(
            "SELECT VARIANCE(value) FROM numbers WHERE category = 'A'",
            (),
        )
        .unwrap();

    assert!((var - 4.0).abs() < 0.0001);
}

#[test]
fn test_var_samp() {
    let db = setup_numbers_db("var_samp");

    // Sample variance = 32/7 ≈ 4.571
    let var: f64 = db
        .query_one(
            "SELECT VAR_SAMP(value) FROM numbers WHERE category = 'A'",
            (),
        )
        .unwrap();

    assert!((var - 4.571).abs() < 0.01, "Expected ~4.571, got {}", var);
}

// ============================================================================
// MEDIAN Tests
// ============================================================================

#[test]
fn test_median_odd_count() {
    let db = setup_numbers_db("median_odd");

    // Category B: 1,2,3,4,5 -> median = 3
    let median: f64 = db
        .query_one("SELECT MEDIAN(value) FROM numbers WHERE category = 'B'", ())
        .unwrap();

    assert!(
        (median - 3.0).abs() < 0.0001,
        "Expected 3.0, got {}",
        median
    );
}

#[test]
fn test_median_even_count() {
    let db = setup_numbers_db("median_even");

    // Category A: 2,4,4,4,5,5,7,9 (8 values) -> median = (4+5)/2 = 4.5
    let median: f64 = db
        .query_one("SELECT MEDIAN(value) FROM numbers WHERE category = 'A'", ())
        .unwrap();

    assert!(
        (median - 4.5).abs() < 0.0001,
        "Expected 4.5, got {}",
        median
    );
}

#[test]
fn test_median_with_group_by() {
    let db = setup_numbers_db("median_group");

    let result = db
        .query(
            "SELECT category, MEDIAN(value) FROM numbers GROUP BY category ORDER BY category",
            (),
        )
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);

    // Category A: median = 4.5
    let median_a: f64 = rows[0].get(1).unwrap();
    assert!((median_a - 4.5).abs() < 0.0001);

    // Category B: median = 3.0
    let median_b: f64 = rows[1].get(1).unwrap();
    assert!((median_b - 3.0).abs() < 0.0001);
}

// ============================================================================
// Combined / Edge Case Tests
// ============================================================================

#[test]
fn test_multiple_aggregates_in_query() {
    let db = setup_numbers_db("multi_agg");

    let result = db
        .query(
            "SELECT AVG(value), STDDEV_POP(value), VARIANCE(value), MEDIAN(value)
         FROM numbers WHERE category = 'A'",
            (),
        )
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 1);

    let avg: f64 = rows[0].get(0).unwrap();
    let stddev: f64 = rows[0].get(1).unwrap();
    let var: f64 = rows[0].get(2).unwrap();
    let median: f64 = rows[0].get(3).unwrap();

    assert!((avg - 5.0).abs() < 0.0001);
    assert!((stddev - 2.0).abs() < 0.0001);
    assert!((var - 4.0).abs() < 0.0001);
    assert!((median - 4.5).abs() < 0.0001);
}

#[test]
fn test_aggregates_with_null() {
    let db = Database::open("memory://agg_null").expect("Failed to create database");

    db.execute(
        "CREATE TABLE with_nulls (id INTEGER PRIMARY KEY, value FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO with_nulls VALUES (1, 1)", ())
        .unwrap();
    db.execute("INSERT INTO with_nulls VALUES (2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO with_nulls VALUES (3, 3)", ())
        .unwrap();
    db.execute("INSERT INTO with_nulls VALUES (4, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO with_nulls VALUES (5, 5)", ())
        .unwrap();

    // All aggregates should ignore NULL values
    // Non-null values: 1, 3, 5 -> mean=3, median=3
    let result = db
        .query("SELECT AVG(value), MEDIAN(value) FROM with_nulls", ())
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    let avg: f64 = rows[0].get(0).unwrap();
    let median: f64 = rows[0].get(1).unwrap();

    assert!((avg - 3.0).abs() < 0.0001);
    assert!((median - 3.0).abs() < 0.0001);
}

// ============================================================================
// Multi-Argument STRING_AGG Tests
// ============================================================================

#[test]
fn test_string_agg_custom_separator() {
    let db = setup_names_db("string_agg_custom_sep");

    let result: String = db
        .query_one("SELECT STRING_AGG(name, ' | ') FROM people", ())
        .unwrap();

    // Should contain all names separated by " | "
    assert!(result.contains("Alice"));
    assert!(result.contains("Bob"));
    assert!(result.contains(" | "));
    assert!(!result.contains(",")); // Should NOT have commas
}

#[test]
fn test_string_agg_dash_separator() {
    let db = setup_names_db("string_agg_dash_sep");

    let result: String = db
        .query_one(
            "SELECT STRING_AGG(name, '-') FROM people WHERE dept = 'Sales'",
            (),
        )
        .unwrap();

    // Charlie-Eve or Eve-Charlie (order not guaranteed)
    assert!(result.contains("Charlie"));
    assert!(result.contains("Eve"));
    assert!(result.contains("-"));
    assert!(!result.contains(","));
}

#[test]
fn test_string_agg_custom_separator_with_group_by() {
    let db = setup_names_db("string_agg_group_custom");

    let result = db
        .query(
            "SELECT dept, STRING_AGG(name, '; ') FROM people GROUP BY dept ORDER BY dept",
            (),
        )
        .expect("Query failed");

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);

    // Engineering: Alice; Bob; Diana (order may vary)
    let names: String = rows[0].get(1).unwrap();
    assert!(names.contains("Alice"));
    assert!(names.contains("Bob"));
    assert!(names.contains("Diana"));
    assert!(names.contains("; "));
    assert!(!names.contains(","));

    // Sales: Charlie; Eve
    let names2: String = rows[1].get(1).unwrap();
    assert!(names2.contains("Charlie"));
    assert!(names2.contains("Eve"));
    assert!(names2.contains("; "));
}

#[test]
fn test_group_concat_custom_separator() {
    let db = setup_names_db("group_concat_custom");

    let result: String = db
        .query_one(
            "SELECT GROUP_CONCAT(name, ' and ') FROM people WHERE dept = 'Sales'",
            (),
        )
        .unwrap();

    // Should be "Charlie and Eve" or "Eve and Charlie"
    assert!(result.contains("Charlie"));
    assert!(result.contains("Eve"));
    assert!(result.contains(" and "));
}

#[test]
fn test_string_agg_newline_separator() {
    let db = setup_names_db("string_agg_newline");

    let result: String = db
        .query_one(
            "SELECT STRING_AGG(name, '\n') FROM people WHERE dept = 'Engineering'",
            (),
        )
        .unwrap();

    // Should contain newlines
    assert!(result.contains("Alice"));
    assert!(result.contains("Bob"));
    assert!(result.contains("Diana"));
    assert!(result.contains("\n"));
}

#[test]
fn test_string_agg_empty_separator() {
    let db = setup_names_db("string_agg_empty");

    let result: String = db
        .query_one(
            "SELECT STRING_AGG(name, '') FROM people WHERE dept = 'Sales'",
            (),
        )
        .unwrap();

    // Should be concatenated without separator
    // "CharlieEve" or "EveCharlie"
    assert!(result.contains("Charlie"));
    assert!(result.contains("Eve"));
    assert!(!result.contains(","));
    assert!(!result.contains(" "));
}
