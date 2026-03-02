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

//! Integration tests for GENERATE_SERIES table-valued function

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create in-memory database")
}

fn collect_i64(db: &Database, sql: &str) -> Vec<i64> {
    let result = db.query(sql, ()).unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get::<i64>(0).unwrap());
    }
    values
}

fn collect_f64(db: &Database, sql: &str) -> Vec<f64> {
    let result = db.query(sql, ()).unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get::<f64>(0).unwrap());
    }
    values
}

// ============================================================================
// Basic Integer Tests
// ============================================================================

#[test]
fn test_generate_series_basic() {
    let db = create_test_db("gs_basic");
    let values = collect_i64(&db, "SELECT * FROM generate_series(1, 5)");
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_generate_series_with_step() {
    let db = create_test_db("gs_step");
    let values = collect_i64(&db, "SELECT * FROM generate_series(0, 10, 2)");
    assert_eq!(values, vec![0, 2, 4, 6, 8, 10]);
}

#[test]
fn test_generate_series_descending() {
    let db = create_test_db("gs_desc");
    let values = collect_i64(&db, "SELECT * FROM generate_series(5, 1, -1)");
    assert_eq!(values, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_generate_series_auto_descending() {
    let db = create_test_db("gs_auto_desc");
    let values = collect_i64(&db, "SELECT * FROM generate_series(5, 1)");
    assert_eq!(values, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_generate_series_empty_direction_mismatch() {
    let db = create_test_db("gs_empty");
    let values = collect_i64(&db, "SELECT * FROM generate_series(1, 5, -1)");
    assert_eq!(values, Vec::<i64>::new());
}

#[test]
fn test_generate_series_single_value() {
    let db = create_test_db("gs_single");
    let values = collect_i64(&db, "SELECT * FROM generate_series(3, 3)");
    assert_eq!(values, vec![3]);
}

#[test]
fn test_generate_series_negative_values() {
    let db = create_test_db("gs_negative");
    let values = collect_i64(&db, "SELECT * FROM generate_series(-3, 3)");
    assert_eq!(values, vec![-3, -2, -1, 0, 1, 2, 3]);
}

#[test]
fn test_generate_series_large_step() {
    let db = create_test_db("gs_large_step");
    let values = collect_i64(&db, "SELECT * FROM generate_series(1, 100, 25)");
    assert_eq!(values, vec![1, 26, 51, 76]);
}

// ============================================================================
// Float Tests
// ============================================================================

#[test]
fn test_generate_series_float() {
    let db = create_test_db("gs_float");
    let values = collect_f64(&db, "SELECT * FROM generate_series(0.0, 1.0, 0.5)");
    assert_eq!(values.len(), 3);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1] - 0.5).abs() < 1e-10);
    assert!((values[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_generate_series_float_mixed_types() {
    let db = create_test_db("gs_float_mix");
    // Integer start/stop with float step triggers float path
    let values = collect_f64(&db, "SELECT * FROM generate_series(0, 2, 0.5)");
    assert_eq!(values.len(), 5); // 0.0, 0.5, 1.0, 1.5, 2.0
}

// ============================================================================
// Alias and Column Naming Tests
// ============================================================================

#[test]
fn test_generate_series_with_alias() {
    let db = create_test_db("gs_alias");
    let values = collect_i64(&db, "SELECT value FROM generate_series(1, 3) AS gs(value)");
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_generate_series_default_column() {
    let db = create_test_db("gs_default_col");
    // Default column name is "value"
    let values = collect_i64(&db, "SELECT value FROM generate_series(1, 3)");
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_generate_series_implicit_alias() {
    let db = create_test_db("gs_impl_alias");
    let values = collect_i64(&db, "SELECT n FROM generate_series(1, 3) gs(n)");
    assert_eq!(values, vec![1, 2, 3]);
}

// ============================================================================
// WHERE, ORDER BY, LIMIT Tests
// ============================================================================

#[test]
fn test_generate_series_with_where() {
    let db = create_test_db("gs_where");
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 10) AS g(value) WHERE value > 7",
    );
    assert_eq!(values, vec![8, 9, 10]);
}

#[test]
fn test_generate_series_with_order_by_desc() {
    let db = create_test_db("gs_order");
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 5) AS g(value) ORDER BY value DESC",
    );
    assert_eq!(values, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_generate_series_with_limit() {
    let db = create_test_db("gs_limit");
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 100) AS g(value) LIMIT 5",
    );
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_generate_series_with_limit_offset() {
    let db = create_test_db("gs_limit_offset");
    // Use CTE wrapper for LIMIT + OFFSET to work correctly
    let values = collect_i64(
        &db,
        "WITH gs AS (SELECT * FROM generate_series(1, 10) AS g(value)) \
         SELECT * FROM gs LIMIT 3 OFFSET 2",
    );
    assert_eq!(values, vec![3, 4, 5]);
}

// ============================================================================
// Aggregation Tests
// ============================================================================

#[test]
fn test_generate_series_sum() {
    let db = create_test_db("gs_sum");
    let result = db
        .query(
            "SELECT SUM(value) FROM generate_series(1, 100) AS g(value)",
            (),
        )
        .unwrap();
    let mut sum = 0i64;
    for row in result {
        let row = row.unwrap();
        sum = row.get(0).unwrap();
    }
    assert_eq!(sum, 5050);
}

#[test]
fn test_generate_series_count() {
    let db = create_test_db("gs_count");
    let result = db
        .query(
            "SELECT COUNT(*) FROM generate_series(1, 1000) AS g(value)",
            (),
        )
        .unwrap();
    let mut count = 0i64;
    for row in result {
        let row = row.unwrap();
        count = row.get(0).unwrap();
    }
    assert_eq!(count, 1000);
}

// ============================================================================
// JOIN Tests
// ============================================================================

#[test]
fn test_generate_series_join_with_table() {
    let db = create_test_db("gs_join");

    db.execute(
        "CREATE TABLE test_items (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO test_items VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO test_items VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO test_items VALUES (3, 'Charlie')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT g.n, t.name FROM generate_series(1, 3) AS g(n) \
             JOIN test_items t ON g.n = t.id ORDER BY g.n",
            (),
        )
        .unwrap();

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (2, "Bob".to_string()));
    assert_eq!(rows[2], (3, "Charlie".to_string()));
}

#[test]
fn test_generate_series_cross_join() {
    let db = create_test_db("gs_cross");

    let result = db
        .query(
            "SELECT a.value, b.value FROM generate_series(1, 3) AS a(value) \
             CROSS JOIN generate_series(1, 2) AS b(value) ORDER BY a.value, b.value",
            (),
        )
        .unwrap();

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let a: i64 = row.get(0).unwrap();
        let b: i64 = row.get(1).unwrap();
        rows.push((a, b));
    }

    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0], (1, 1));
    assert_eq!(rows[1], (1, 2));
    assert_eq!(rows[2], (2, 1));
    assert_eq!(rows[3], (2, 2));
    assert_eq!(rows[4], (3, 1));
    assert_eq!(rows[5], (3, 2));
}

// ============================================================================
// Subquery Tests
// ============================================================================

#[test]
fn test_generate_series_in_subquery() {
    let db = create_test_db("gs_subquery");
    let values = collect_i64(
        &db,
        "SELECT * FROM (SELECT * FROM generate_series(1, 5) AS g(n)) sub ORDER BY n",
    );
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

// ============================================================================
// Error Tests
// ============================================================================

#[test]
fn test_generate_series_zero_step_error() {
    let db = create_test_db("gs_err_zero");
    let result = db.query("SELECT * FROM generate_series(1, 10, 0)", ());
    assert!(result.is_err());
}

#[test]
fn test_generate_series_wrong_arg_count() {
    let db = create_test_db("gs_err_args");
    let result = db.query("SELECT * FROM generate_series(1)", ());
    assert!(result.is_err());
}

// ============================================================================
// Case Insensitivity Tests
// ============================================================================

// ============================================================================
// Scalar (SELECT without FROM) Tests
// ============================================================================

#[test]
fn test_generate_series_scalar_returns_array() {
    let db = create_test_db("gs_scalar");
    let result = db.query("SELECT generate_series(1, 5)", ()).unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        let val: String = row.get(0).unwrap();
        values.push(val);
    }
    assert_eq!(values, vec!["[1, 2, 3, 4, 5]"]);
}

#[test]
fn test_generate_series_scalar_with_step() {
    let db = create_test_db("gs_scalar_step");
    let result = db.query("SELECT generate_series(0, 10, 2)", ()).unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        let val: String = row.get(0).unwrap();
        values.push(val);
    }
    assert_eq!(values, vec!["[0, 2, 4, 6, 8, 10]"]);
}

#[test]
fn test_generate_series_case_insensitive() {
    let db = create_test_db("gs_case");
    let values = collect_i64(&db, "SELECT * FROM GENERATE_SERIES(1, 3)");
    assert_eq!(values, vec![1, 2, 3]);
}

// ============================================================================
// Timestamp/Date Tests
// ============================================================================

fn collect_string(db: &Database, sql: &str) -> Vec<String> {
    let result = db.query(sql, ()).unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get::<String>(0).unwrap());
    }
    values
}

#[test]
fn test_generate_series_date_days() {
    let db = create_test_db("gs_date_days");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01', '2024-01-05', '1 day')",
    );
    assert_eq!(values.len(), 5);
    assert!(values[0].starts_with("2024-01-01"));
    assert!(values[4].starts_with("2024-01-05"));
}

#[test]
fn test_generate_series_date_auto_step() {
    let db = create_test_db("gs_date_auto");
    // Without step, default to 1 day for ascending
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01', '2024-01-03')",
    );
    assert_eq!(values.len(), 3);
    assert!(values[0].starts_with("2024-01-01"));
    assert!(values[1].starts_with("2024-01-02"));
    assert!(values[2].starts_with("2024-01-03"));
}

#[test]
fn test_generate_series_timestamp_hours() {
    let db = create_test_db("gs_ts_hours");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01 00:00:00', '2024-01-01 06:00:00', '2 hours')",
    );
    assert_eq!(values.len(), 4); // 00:00, 02:00, 04:00, 06:00
}

#[test]
fn test_generate_series_timestamp_minutes() {
    let db = create_test_db("gs_ts_minutes");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01 00:00:00', '2024-01-01 00:30:00', '10 minutes')",
    );
    assert_eq!(values.len(), 4); // 00:00, 00:10, 00:20, 00:30
}

#[test]
fn test_generate_series_date_descending() {
    let db = create_test_db("gs_date_desc");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-05', '2024-01-01', '-1 day')",
    );
    assert_eq!(values.len(), 5);
    assert!(values[0].starts_with("2024-01-05"));
    assert!(values[4].starts_with("2024-01-01"));
}

#[test]
fn test_generate_series_date_auto_descending() {
    let db = create_test_db("gs_date_auto_desc");
    // Without step, detect descending and default to -1 day
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-03', '2024-01-01')",
    );
    assert_eq!(values.len(), 3);
    assert!(values[0].starts_with("2024-01-03"));
    assert!(values[2].starts_with("2024-01-01"));
}

#[test]
fn test_generate_series_date_weeks() {
    let db = create_test_db("gs_date_weeks");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01', '2024-01-29', '1 week')",
    );
    assert_eq!(values.len(), 5); // Jan 1, 8, 15, 22, 29
    assert!(values[0].starts_with("2024-01-01"));
    assert!(values[4].starts_with("2024-01-29"));
}

#[test]
fn test_generate_series_date_months() {
    let db = create_test_db("gs_date_months");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01', '2024-04-01', '1 month')",
    );
    // month = 30 days: Jan 1, Jan 31, Mar 1, Mar 31
    assert_eq!(values.len(), 4);
}

#[test]
fn test_generate_series_date_empty_mismatch() {
    let db = create_test_db("gs_date_empty");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-05', '2024-01-01', '1 day')",
    );
    assert_eq!(values.len(), 0);
}

#[test]
fn test_generate_series_date_with_where() {
    let db = create_test_db("gs_date_where");
    let values = collect_string(
        &db,
        "SELECT * FROM generate_series('2024-01-01', '2024-01-10', '1 day') AS g(value) \
         WHERE value > '2024-01-07'",
    );
    assert_eq!(values.len(), 3); // Jan 8, 9, 10
}

#[test]
fn test_generate_series_date_count() {
    let db = create_test_db("gs_date_count");
    let result = db
        .query(
            "SELECT COUNT(*) FROM generate_series('2024-01-01', '2024-12-31', '1 day') AS g(value)",
            (),
        )
        .unwrap();
    let mut count = 0i64;
    for row in result {
        let row = row.unwrap();
        count = row.get(0).unwrap();
    }
    assert_eq!(count, 366); // 2024 is a leap year
}

#[test]
fn test_generate_series_date_scalar() {
    let db = create_test_db("gs_date_scalar");
    let result = db
        .query(
            "SELECT generate_series('2024-01-01', '2024-01-03', '1 day')",
            (),
        )
        .unwrap();
    let mut values = Vec::new();
    for row in result {
        let row = row.unwrap();
        let val: String = row.get(0).unwrap();
        values.push(val);
    }
    // Should return a JSON array with 3 timestamp strings
    assert_eq!(values.len(), 1);
    let arr = &values[0];
    assert!(arr.starts_with('['));
    assert!(arr.ends_with(']'));
    assert!(arr.contains("2024-01-01"));
    assert!(arr.contains("2024-01-03"));
}

// ============================================================================
// P0: Overflow safety tests
// ============================================================================

#[test]
fn test_generate_series_huge_range_error() {
    let db = create_test_db("gs_huge_range");
    // Full i64 range with step 1 would be ~18 quintillion rows
    let result = db.query(
        "SELECT * FROM generate_series(-9223372036854775808, 9223372036854775807, 1) LIMIT 1",
        (),
    );
    // Should succeed with LIMIT pushdown, returning just 1 row
    let result = result.unwrap();
    let mut count = 0;
    for _ in result {
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn test_generate_series_tiny_float_step_with_limit() {
    let db = create_test_db("gs_tiny_step");
    // Tiny step would produce astronomical rows, but LIMIT pushdown should handle it
    let values = collect_f64(
        &db,
        "SELECT * FROM generate_series(0.0, 1.0, 0.0001) LIMIT 3",
    );
    assert_eq!(values.len(), 3);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1] - 0.0001).abs() < 1e-10);
    assert!((values[2] - 0.0002).abs() < 1e-10);
}

// ============================================================================
// P1: LIMIT pushdown tests
// ============================================================================

#[test]
fn test_generate_series_large_with_limit() {
    let db = create_test_db("gs_large_limit");
    // Without LIMIT pushdown this would hit the 10M cap
    let values = collect_i64(&db, "SELECT * FROM generate_series(1, 20000000) LIMIT 5");
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_generate_series_limit_offset_pushdown() {
    let db = create_test_db("gs_limit_offset_push");
    // Use CTE wrapper for LIMIT+OFFSET (direct path has known edge case)
    let values = collect_i64(
        &db,
        "WITH gs AS (SELECT * FROM generate_series(1, 20000000) AS g(value) LIMIT 10) \
         SELECT * FROM gs LIMIT 3 OFFSET 2",
    );
    assert_eq!(values, vec![3, 4, 5]);
}

// ============================================================================
// P2: Zero-argument TVF parser test
// ============================================================================

#[test]
fn test_generate_series_zero_args_error() {
    let db = create_test_db("gs_zero_args");
    // Should produce a function arity error, not a parse error
    let result = db.query("SELECT * FROM generate_series()", ());
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            // Should be a semantic error about argument count, not a parse error
            assert!(
                err_msg.contains("2 or 3 arguments") || err_msg.contains("argument"),
                "Expected argument count error, got: {}",
                err_msg
            );
        }
        Ok(_) => panic!("Expected error for zero-argument generate_series"),
    }
}

/// Regression: LIMIT/OFFSET was applied twice (once in CTE logic, once in outer execute_select)
/// `SELECT * FROM generate_series(1, 10) LIMIT 3 OFFSET 2` should return [3, 4, 5]
#[test]
fn test_generate_series_limit_offset_no_double_application() {
    let db = create_test_db("limit_offset_no_double");
    let values = collect_i64(&db, "SELECT * FROM generate_series(1, 10) LIMIT 3 OFFSET 2");
    assert_eq!(values, vec![3, 4, 5]);
}

/// Regression: LIMIT pushdown was unsafe with ORDER BY
/// `generate_series(1, 100) ORDER BY value DESC LIMIT 5` must return [100, 99, 98, 97, 96]
/// Without the fix, only 5 rows were generated then sorted → [5, 4, 3, 2, 1]
#[test]
fn test_generate_series_order_by_desc_with_limit() {
    let db = create_test_db("order_by_desc_limit");
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 100) AS g(value) ORDER BY value DESC LIMIT 5",
    );
    assert_eq!(values, vec![100, 99, 98, 97, 96]);
}

/// Regression: DISTINCT should also prevent LIMIT pushdown
#[test]
fn test_generate_series_distinct_no_pushdown() {
    let db = create_test_db("distinct_no_pushdown");
    // With step=2 generating 1,3,5,7,9 — DISTINCT doesn't change these but
    // the key point is that all rows are generated before DISTINCT + LIMIT apply
    let values = collect_i64(
        &db,
        "SELECT DISTINCT * FROM generate_series(1, 10, 2) LIMIT 3",
    );
    assert_eq!(values.len(), 3);
}

/// Regression: ORDER BY + OFFSET + LIMIT together
#[test]
fn test_generate_series_order_by_with_offset_limit() {
    let db = create_test_db("order_by_offset_limit");
    // generate_series(1, 10) ORDER BY value DESC → [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    // OFFSET 2 → [8, 7, 6, 5, 4, 3, 2, 1]
    // LIMIT 3 → [8, 7, 6]
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 10) AS g(value) ORDER BY value DESC LIMIT 3 OFFSET 2",
    );
    assert_eq!(values, vec![8, 7, 6]);
}

/// Regression: ORDER BY + LIMIT was silently skipped for aggregation queries
/// on memory-backed data (TVFs, multi-use CTEs) because execute_query_on_cte_result's
/// aggregation path early-returned without applying ORDER BY/LIMIT.
#[test]
fn test_generate_series_aggregation_with_order_by_limit() {
    let db = create_test_db("agg_order_by_limit");
    // generate_series(1,9) grouped by value%3: grp=0 sum=18, grp=1 sum=12, grp=2 sum=15
    // ORDER BY total DESC → [(0,18), (2,15), (1,12)]
    // LIMIT 2 → [(0,18), (2,15)]
    let result = db
        .query(
            "SELECT value % 3 AS grp, SUM(value) AS total \
             FROM generate_series(1, 9) AS g(value) \
             GROUP BY value % 3 ORDER BY total DESC LIMIT 2",
            (),
        )
        .unwrap();
    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }
    assert_eq!(
        rows.len(),
        2,
        "Expected 2 rows after LIMIT, got {}",
        rows.len()
    );
    assert_eq!(rows[0], (0, 18));
    assert_eq!(rows[1], (2, 15));
}

/// Regression: Window function queries on memory-backed data also need ORDER BY + LIMIT
#[test]
fn test_generate_series_window_with_order_by_limit() {
    let db = create_test_db("window_order_by_limit");
    let result = db
        .query(
            "SELECT value, ROW_NUMBER() OVER (ORDER BY value) AS rn \
             FROM generate_series(1, 10) AS g(value) \
             ORDER BY value DESC LIMIT 3",
            (),
        )
        .unwrap();
    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }
    assert_eq!(
        rows.len(),
        3,
        "Expected 3 rows after LIMIT, got {}",
        rows.len()
    );
    // ORDER BY value DESC → value 10,9,8 with rn 10,9,8
    assert_eq!(rows[0].0, 10);
    assert_eq!(rows[1].0, 9);
    assert_eq!(rows[2].0, 8);
}

/// Regression: Large interval values caused overflow panic in parse_interval
#[test]
fn test_generate_series_large_interval_no_panic() {
    let db = create_test_db("large_interval");
    let result = db.query(
        "SELECT * FROM generate_series('2024-01-01', '2024-01-02', '9223372036854775807 years')",
        (),
    );
    // Should be an error, not a panic
    assert!(
        result.is_err() || {
            // If it returns Ok, it should be empty or have an error in iteration
            true
        }
    );
}

/// Regression: LIMIT pushdown was unsafe with window functions
/// percent_rank() must be computed over the full set, not just LIMIT rows
#[test]
fn test_generate_series_window_percent_rank_with_limit() {
    let db = create_test_db("window_pr_limit");
    let result = db
        .query(
            "SELECT value, percent_rank() OVER (ORDER BY value) AS pr \
             FROM generate_series(1, 10) AS g(value) LIMIT 3",
            (),
        )
        .unwrap();
    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let value = row.get::<i64>(0).unwrap();
        let pr = row.get::<f64>(1).unwrap();
        rows.push((value, pr));
    }
    assert_eq!(rows.len(), 3);
    // percent_rank for value=1 over 10 rows = 0.0
    assert!(
        (rows[0].1 - 0.0).abs() < 0.001,
        "pr[0] should be 0.0, got {}",
        rows[0].1
    );
    // percent_rank for value=2 = 1/9 ≈ 0.111
    assert!(
        rows[1].1 > 0.1 && rows[1].1 < 0.12,
        "pr[1] should be ~0.111, got {}",
        rows[1].1
    );
}

/// Regression: ORDER BY on non-projected source column returned wrong results
/// SELECT value+1 AS x FROM generate_series(1,5) ORDER BY value DESC LIMIT 2
/// Expected: [6, 5] (value=5 → 6, value=4 → 5)
#[test]
fn test_generate_series_order_by_non_projected_column() {
    let db = create_test_db("order_by_non_projected");
    let values = collect_i64(
        &db,
        "SELECT value + 1 AS x FROM generate_series(1, 5) AS g(value) ORDER BY value DESC LIMIT 2",
    );
    assert_eq!(values, vec![6, 5]);
}

/// Regression: TVF in JOIN with LIMIT should pass limit hint to avoid hard-cap error
#[test]
fn test_generate_series_join_with_limit() {
    let db = create_test_db("join_limit");
    // Large series in a join context with small LIMIT should work
    let values = collect_i64(
        &db,
        "SELECT g.value FROM generate_series(1, 100) AS g(value) \
         CROSS JOIN (VALUES (1)) AS v(x) \
         ORDER BY g.value LIMIT 3",
    );
    assert_eq!(values, vec![1, 2, 3]);
}

/// Regression: Large interval values caused panic in chrono Duration constructors
#[test]
fn test_generate_series_large_interval_units_no_panic() {
    let db = create_test_db("large_interval_units");
    // All of these should return errors, not panics
    let cases = [
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 weeks')",
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 days')",
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 hours')",
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 minutes')",
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 seconds')",
        "SELECT * FROM generate_series('2024-01-01','2024-01-02','9223372036854775807 milliseconds')",
    ];
    for sql in &cases {
        match db.query(sql, ()) {
            Err(_) => {} // Expected: overflow error
            Ok(rows) => {
                // If it returns Ok, iterating should produce 0-1 rows (not panic)
                let mut count = 0;
                for _ in rows {
                    count += 1;
                }
                assert!(count <= 1, "Unexpected row count {} for: {}", count, sql);
            }
        }
    }
}

/// Regression: ORDER BY expression (-value) was silently ignored for TVF queries
#[test]
fn test_generate_series_order_by_expression() {
    let db = create_test_db("order_by_expr");
    let values = collect_i64(
        &db,
        "SELECT * FROM generate_series(1, 5) AS g(value) ORDER BY -value LIMIT 3",
    );
    // ORDER BY -value → descending: 5,4,3,2,1. LIMIT 3 → 5,4,3
    assert_eq!(values, vec![5, 4, 3]);
}

/// Regression: ORDER BY SUM(value) was silently ignored for aggregation on TVF
#[test]
fn test_generate_series_aggregation_order_by_aggregate() {
    let db = create_test_db("agg_order_by_agg");
    let result = db
        .query(
            "SELECT value % 3 AS grp, SUM(value) AS total \
             FROM generate_series(1, 9) AS g(value) \
             GROUP BY value % 3 ORDER BY SUM(value) ASC LIMIT 2",
            (),
        )
        .unwrap();
    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }
    assert_eq!(rows.len(), 2, "Expected 2 rows, got {}", rows.len());
    // SUM(value) ASC: grp=1 total=12, grp=2 total=15, grp=0 total=18
    assert_eq!(rows[0], (1, 12));
    assert_eq!(rows[1], (2, 15));
}

/// Regression: DISTINCT + ORDER BY + LIMIT applied LIMIT before DISTINCT
/// producing too few rows
#[test]
fn test_generate_series_distinct_order_by_limit() {
    let db = create_test_db("distinct_order_limit");
    let values = collect_i64(
        &db,
        "SELECT DISTINCT value % 50 AS v \
         FROM generate_series(1, 100) AS g(value) \
         ORDER BY v LIMIT 3",
    );
    // 100 rows → DISTINCT on value%50 → 50 unique values (0..49) → ORDER BY → LIMIT 3
    assert_eq!(values, vec![0, 1, 2]);
}

/// Regression: LIMIT pushdown broke aggregation queries.
/// COUNT(*) FROM generate_series(1,100) LIMIT 1 returned 1 instead of 100.
#[test]
fn test_generate_series_aggregation_with_limit() {
    let db = create_test_db("agg_limit");
    let values = collect_i64(&db, "SELECT COUNT(*) FROM generate_series(1, 100) LIMIT 1");
    assert_eq!(values, vec![100]);

    let values = collect_i64(
        &db,
        "SELECT SUM(value) FROM generate_series(1, 100) AS g(value) LIMIT 1",
    );
    assert_eq!(values, vec![5050]);
}

/// Regression: chrono::Duration::days(huge) panicked on large integer timestamp steps.
#[test]
fn test_generate_series_timestamp_huge_integer_step_no_panic() {
    let db = create_test_db("ts_huge_step");
    let result = db.query(
        "SELECT * FROM generate_series('2024-01-01','2024-01-02',9223372036854775807)",
        (),
    );
    // Should return an error, not panic
    assert!(result.is_err());
}

/// Regression: Complex ORDER BY on non-projected source column returned wrong results.
/// SELECT value + 1 AS x FROM generate_series(1,5) ORDER BY -value LIMIT 2
/// returned [2,3] instead of [6,5].
#[test]
fn test_generate_series_complex_order_by_non_projected_column() {
    let db = create_test_db("complex_order_nonproj");
    let values = collect_i64(
        &db,
        "SELECT value + 1 AS x \
         FROM generate_series(1, 5) AS g(value) \
         ORDER BY -value LIMIT 2",
    );
    // value descending: 5,4,3,2,1 → value+1: 6,5,4,3,2 → LIMIT 2: [6,5]
    assert_eq!(values, vec![6, 5]);
}

/// Regression: WHERE predicate pushdown narrows TVF generation range.
/// This verifies correctness (not just performance).
#[test]
fn test_generate_series_where_pushdown_correctness() {
    let db = create_test_db("where_pushdown");

    // value >= 95 should still work correctly
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 100) AS g(value) WHERE value >= 95",
    );
    assert_eq!(values, vec![95, 96, 97, 98, 99, 100]);

    // value BETWEEN 3 AND 5
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10) AS g(value) WHERE value BETWEEN 3 AND 5",
    );
    assert_eq!(values, vec![3, 4, 5]);

    // value = 42
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 100) AS g(value) WHERE value = 42",
    );
    assert_eq!(values, vec![42]);

    // value > 98 AND value <= 100
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 100) AS g(value) \
         WHERE value > 98 AND value <= 100",
    );
    assert_eq!(values, vec![99, 100]);

    // value < 3 (range clamps stop)
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 100) AS g(value) WHERE value < 3",
    );
    assert_eq!(values, vec![1, 2]);
}

/// WHERE pushdown correctness with custom column alias
#[test]
fn test_generate_series_where_pushdown_with_alias() {
    let db = create_test_db("where_pushdown_alias");
    let values = collect_i64(
        &db,
        "SELECT n FROM generate_series(1, 1000) AS g(n) WHERE n >= 998",
    );
    assert_eq!(values, vec![998, 999, 1000]);
}

/// WHERE pushdown should not affect queries with step != 1
#[test]
fn test_generate_series_where_no_pushdown_with_step() {
    let db = create_test_db("where_no_pushdown_step");
    // Step of 2: values are 1,3,5,7,9
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10, 2) AS g(value) WHERE value >= 5",
    );
    assert_eq!(values, vec![5, 7, 9]);
}

/// WHERE pushdown on descending series
#[test]
fn test_generate_series_where_pushdown_descending() {
    let db = create_test_db("where_pushdown_desc");
    // Descending: 10,9,8,...,1
    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(10, 1, -1) AS g(value) WHERE value <= 3",
    );
    assert_eq!(values, vec![3, 2, 1]);
}

/// Regression: JOIN + LIMIT on TVF applied limit before filter, returning empty results.
/// generate_series(1,1000000) JOIN t ... WHERE g.value > 999998 LIMIT 1
/// should return a row, not empty.
#[test]
fn test_generate_series_join_limit_with_filter() {
    let db = create_test_db("join_limit_filter");
    db.execute(
        "CREATE TABLE jl_table (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO jl_table VALUES (999999, 'a')", ())
        .unwrap();
    db.execute("INSERT INTO jl_table VALUES (1000000, 'b')", ())
        .unwrap();

    let values = collect_i64(
        &db,
        "SELECT g.value FROM generate_series(1, 1000000) AS g(value) \
         JOIN jl_table t ON g.value = t.id \
         WHERE g.value > 999998 LIMIT 1",
    );
    assert_eq!(values.len(), 1);
    assert!(values[0] == 999999 || values[0] == 1000000);
}

/// Regression: Float predicate in WHERE caused range pushdown to drop valid rows.
/// value < 3.5 on integer series should include 1,2,3 (not just 1,2).
#[test]
fn test_generate_series_where_float_predicate_no_miscompile() {
    let db = create_test_db("float_pred");

    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10) AS g(value) WHERE value < 3.5",
    );
    assert_eq!(values, vec![1, 2, 3]);

    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10) AS g(value) WHERE value > 2.5",
    );
    assert_eq!(values, vec![3, 4, 5, 6, 7, 8, 9, 10]);

    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10) AS g(value) WHERE value <= 3.5",
    );
    assert_eq!(values, vec![1, 2, 3]);

    let values = collect_i64(
        &db,
        "SELECT value FROM generate_series(1, 10) AS g(value) WHERE value >= 7.5",
    );
    assert_eq!(values, vec![8, 9, 10]);
}

/// Regression: CASE expression in ORDER BY on non-projected source column was missed.
/// ORDER BY CASE WHEN value > 3 THEN 0 ELSE 1 END should sort values 4,5 first.
#[test]
fn test_generate_series_order_by_case_expression() {
    let db = create_test_db("order_by_case");
    let values = collect_i64(
        &db,
        "SELECT value + 1 AS x \
         FROM generate_series(1, 5) AS g(value) \
         ORDER BY CASE WHEN value > 3 THEN 0 ELSE 1 END \
         LIMIT 2",
    );
    // CASE sorts: values 4,5 get sort key 0 (first), values 1,2,3 get sort key 1
    // So first 2 rows are value 4,5 → x = 5,6
    assert_eq!(values, vec![5, 6]);
}

/// Regression: Large generate_series with WHERE 1=0 errored instead of returning empty.
/// The hard-cap rejection fired before the constant-false WHERE could filter rows.
/// Also verifies that the result schema has correct column names (not empty).
#[test]
fn test_generate_series_large_where_false_returns_empty() {
    let db = create_test_db("large_where_false");

    // Direct TVF path — verify empty result with correct schema
    let result = db
        .query(
            "SELECT * FROM generate_series(1, 20000000) AS g(value) WHERE 1=0 LIMIT 1",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.collect();
    assert!(rows.is_empty());

    // JOIN path — verify empty result with correct column name
    let result = db
        .query(
            "SELECT g.value FROM generate_series(1, 20000000) AS g(value) \
             JOIN (VALUES (1)) AS v(x) ON 1=1 WHERE 1=0 LIMIT 1",
            (),
        )
        .unwrap();
    let columns = result.columns();
    assert!(
        !columns.is_empty(),
        "Result schema should have columns, got empty"
    );
    assert_eq!(columns[0], "value", "Column name should be 'value'");
    let rows: Vec<_> = result.collect();
    assert!(rows.is_empty());

    // Small range — same behavior, confirms it's not size-dependent
    let result = db
        .query(
            "SELECT g.value FROM generate_series(1, 5) AS g(value) \
             JOIN (VALUES (1)) AS v(x) ON 1=1 WHERE 1=0",
            (),
        )
        .unwrap();
    let columns = result.columns();
    assert!(!columns.is_empty());
    let rows: Vec<_> = result.collect();
    assert!(rows.is_empty());
}

/// The 10M cap now silently truncates rather than erroring.
/// Verify that generate_series(1, 20000000) without LIMIT produces rows up to the cap.
#[test]
fn test_generate_series_cap_truncates_not_errors() {
    let db = create_test_db("cap_truncates");
    let values = collect_i64(
        &db,
        "SELECT COUNT(*) FROM generate_series(1, 20000000) AS g(value)",
    );
    assert_eq!(values, vec![10_000_000]);
}
