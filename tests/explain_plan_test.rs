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

//! Regression tests for: EXPLAIN plan mis-attributes filter to wrong scan alias.
//!
//! The bug: In a JOIN query with `WHERE o.status = 'paid'`, EXPLAIN showed
//! the filter under `Seq Scan on users (u)` instead of `Seq Scan on orders (o)`.
//!
//! Extended to cover unqualified column names with subquery, CTE, and SELECT *
//! join inputs where the filter must be resolved via schema/SELECT-list inspection.

use std::sync::atomic::{AtomicUsize, Ordering};
use stoolap::Database;

static TEST_ID: AtomicUsize = AtomicUsize::new(0);

fn setup_db() -> Database {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    let db = Database::open(&format!("memory://explain_plan_{}", id))
        .expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("create users");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, status TEXT)",
        (),
    )
    .expect("create orders");
    db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())
        .expect("insert users");
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 'paid'), (2, 2, 'pending')",
        (),
    )
    .expect("insert orders");
    db
}

fn get_explain_lines(db: &Database, sql: &str) -> Vec<String> {
    let result = db.query(sql, ()).expect("EXPLAIN should succeed");
    let mut lines = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let line: String = row.get(0).unwrap_or_default();
        lines.push(line);
    }
    lines
}

/// Check that none of the detail lines under a scan node contain `forbidden`.
/// Detail lines are those after `start_idx` until the next `-> ` node line.
fn assert_no_filter_under_scan(lines: &[String], start_idx: usize, forbidden: &str, plan: &str) {
    for line in &lines[start_idx + 1..] {
        if line.contains("-> ") {
            break;
        }
        assert!(
            !line.contains(forbidden),
            "Filter '{}' should NOT appear under scan at line {}.\nFull plan:\n{}",
            forbidden,
            start_idx,
            plan
        );
    }
}

#[test]
fn test_explain_filter_on_correct_table_alias() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'paid'",
    );
    let plan = lines.join("\n");

    let users_idx = lines
        .iter()
        .position(|l| l.contains("users"))
        .expect("Should have users scan in plan");

    // o.status filter must NOT appear under users scan
    assert_no_filter_under_scan(&lines, users_idx, "o.status", &plan);

    // The filter should appear somewhere in the plan
    let has_status_filter = lines
        .iter()
        .any(|l| l.contains("status") && !l.contains("Join Cond"));
    assert!(
        has_status_filter,
        "Filter on status should appear in the plan.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_filter_right_table() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE u.name = 'Alice'",
    );
    let plan = lines.join("\n");

    let orders_idx = lines
        .iter()
        .position(|l| l.contains("orders"))
        .expect("Should have orders scan in plan");

    // u.name filter must NOT appear under orders scan
    assert_no_filter_under_scan(&lines, orders_idx, "u.name", &plan);
}

#[test]
fn test_explain_filters_both_tables() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'paid' AND u.name = 'Alice'",
    );
    let plan = lines.join("\n");

    let users_idx = lines.iter().position(|l| l.contains("users"));
    let orders_idx = lines.iter().position(|l| l.contains("orders"));

    assert!(
        users_idx.is_some(),
        "Should have users scan.\nPlan:\n{}",
        plan
    );
    assert!(
        orders_idx.is_some(),
        "Should have orders scan.\nPlan:\n{}",
        plan
    );

    // o.status must NOT be under users scan
    if let Some(idx) = users_idx {
        assert_no_filter_under_scan(&lines, idx, "o.status", &plan);
    }
    // u.name must NOT be under orders scan
    if let Some(idx) = orders_idx {
        assert_no_filter_under_scan(&lines, idx, "u.name", &plan);
    }
}

#[test]
fn test_explain_analyze_filter_on_correct_table() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN ANALYZE SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'paid'",
    );
    let plan = lines.join("\n");

    let idx = lines
        .iter()
        .position(|l| l.contains("users"))
        .expect("EXPLAIN ANALYZE output should contain a 'users' scan line");
    assert_no_filter_under_scan(&lines, idx, "o.status", &plan);
}

// ============================================================================
// Unqualified column filters with subquery / CTE / SELECT * join inputs
// ============================================================================

/// Helper: find the scan/subquery-scan line for a given alias and assert that
/// the filter text appears in a detail line directly beneath it.
fn assert_filter_under_node(lines: &[String], node_text: &str, filter_text: &str, plan: &str) {
    let idx = lines
        .iter()
        .position(|l| l.contains(node_text))
        .unwrap_or_else(|| panic!("Node '{}' not found in plan:\n{}", node_text, plan));

    for line in &lines[idx + 1..] {
        if line.contains("-> ")
            || line.starts_with("  SELECT")
            || line.trim_start().starts_with("SELECT")
        {
            break;
        }
        if line.contains(filter_text) {
            return; // found it
        }
    }
    panic!(
        "Filter '{}' not found under '{}' node.\nFull plan:\n{}",
        filter_text, node_text, plan
    );
}

#[test]
fn test_explain_unqualified_filter_plain_table() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN orders o ON o.user_id = u.id WHERE status = 'paid'",
    );
    let plan = lines.join("\n");

    // 'status' only exists on orders, must appear under orders scan
    assert_filter_under_node(&lines, "Seq Scan on orders", "status = 'paid'", &plan);

    // Must NOT appear under users scan
    let users_idx = lines
        .iter()
        .position(|l| l.contains("Seq Scan on users"))
        .expect("Should have users scan");
    assert_no_filter_under_scan(&lines, users_idx, "status", &plan);
}

#[test]
fn test_explain_unqualified_filter_named_subquery() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN (SELECT user_id, status FROM orders) o ON o.user_id = u.id WHERE status = 'paid'",
    );
    let plan = lines.join("\n");

    // Filter must appear under the Subquery Scan, not under users
    assert_filter_under_node(&lines, "Subquery Scan AS o", "status = 'paid'", &plan);

    let users_idx = lines
        .iter()
        .position(|l| l.contains("Seq Scan on users"))
        .expect("Should have users scan");
    assert_no_filter_under_scan(&lines, users_idx, "status", &plan);
}

#[test]
fn test_explain_unqualified_filter_star_subquery() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN (SELECT * FROM orders) o ON o.user_id = u.id WHERE status = 'paid'",
    );
    let plan = lines.join("\n");

    // SELECT * inherits orders columns — filter must appear under subquery scan
    assert_filter_under_node(&lines, "Subquery Scan AS o", "status = 'paid'", &plan);

    let users_idx = lines
        .iter()
        .position(|l| l.contains("Seq Scan on users"))
        .expect("Should have users scan");
    assert_no_filter_under_scan(&lines, users_idx, "status", &plan);
}

#[test]
fn test_explain_unqualified_filter_cte() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN WITH o AS (SELECT * FROM orders) SELECT * FROM users u JOIN o ON o.user_id = u.id WHERE status = 'paid'",
    );
    let plan = lines.join("\n");

    // CTE materialized as subquery scan — filter belongs there, not under users
    let users_idx = lines
        .iter()
        .position(|l| l.contains("Seq Scan on users"))
        .expect("Should have users scan");
    assert_no_filter_under_scan(&lines, users_idx, "status", &plan);

    // Filter should appear somewhere in the plan (under the CTE scan or subquery scan)
    let has_filter = lines
        .iter()
        .any(|l| l.contains("status = 'paid'") && l.contains("Filter"));
    assert!(
        has_filter,
        "Filter on status should appear in the plan.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_unqualified_both_sides_star_subquery() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN (SELECT * FROM orders) o ON o.user_id = u.id WHERE name = 'Alice' AND status = 'paid'",
    );
    let plan = lines.join("\n");

    // 'name' only on users, 'status' only on orders
    assert_filter_under_node(&lines, "Seq Scan on users", "name = 'Alice'", &plan);
    assert_filter_under_node(&lines, "Subquery Scan AS o", "status = 'paid'", &plan);
}

// ============================================================================
// Cross-table predicates: must appear as Join Filter, not under a scan
// ============================================================================

#[test]
fn test_explain_cross_table_predicate_qualified() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE u.name = 'Alice' AND o.status = u.name",
    );
    let plan = lines.join("\n");

    // Cross-table predicate o.status = u.name must NOT appear under either scan
    let users_idx = lines
        .iter()
        .position(|l| l.contains("users"))
        .expect("Should have users scan");
    let orders_idx = lines
        .iter()
        .position(|l| l.contains("orders"))
        .expect("Should have orders scan");

    assert_no_filter_under_scan(&lines, users_idx, "o.status = u.name", &plan);
    assert_no_filter_under_scan(&lines, orders_idx, "o.status = u.name", &plan);

    // Cross-table predicate should appear as a Join Filter line
    let has_join_filter = lines.iter().any(|l| l.contains("Join Filter:"));
    assert!(
        has_join_filter,
        "Cross-table predicate should appear as 'Join Filter'.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_cross_table_predicate_equality() {
    let db = setup_db();
    // u.name = o.status references both sides
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE u.name = o.status",
    );
    let plan = lines.join("\n");

    // Must appear as Join Filter, not under either scan
    let has_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("u.name = o.status"));
    assert!(
        has_join_filter,
        "Cross-table predicate should appear as 'Join Filter'.\nFull plan:\n{}",
        plan
    );
}

// ============================================================================
// FunctionTableSource filter rendering
// ============================================================================

#[test]
fn test_explain_function_table_source_filter() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN GENERATE_SERIES(1, 10) AS gs(value) ON gs.value = u.id WHERE u.name = 'Alice'",
    );
    let plan = lines.join("\n");

    // Function scan node should be present
    let has_func_scan = lines.iter().any(|l| l.contains("Function Scan"));
    assert!(
        has_func_scan,
        "Should have Function Scan node.\nFull plan:\n{}",
        plan
    );

    // Filter on u.name should appear under users scan, not function scan
    let func_idx = lines.iter().position(|l| l.contains("Function Scan"));
    if let Some(idx) = func_idx {
        assert_no_filter_under_scan(&lines, idx, "u.name", &plan);
    }
}

#[test]
fn test_explain_function_table_source_no_column_alias() {
    let db = setup_db();
    // No explicit column aliases — TVF default column name is "value"
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN GENERATE_SERIES(1, 3) gs ON u.id = gs.value WHERE value > 1",
    );
    let plan = lines.join("\n");

    // The unqualified predicate "value > 1" should appear under Function Scan,
    // not as a Join Filter (the "value" column belongs to generate_series)
    assert_filter_under_node(&lines, "Function Scan", "value > 1", &plan);

    // Must NOT appear as a Join Filter
    let is_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("value > 1"));
    assert!(
        !is_join_filter,
        "Unqualified 'value > 1' should NOT be a Join Filter.\nFull plan:\n{}",
        plan
    );

    // Must NOT appear under users scan
    let users_idx = lines
        .iter()
        .position(|l| l.contains("users"))
        .expect("Should have users scan");
    assert_no_filter_under_scan(&lines, users_idx, "value > 1", &plan);
}

// ============================================================================
// ValuesSource handling
// ============================================================================

#[test]
fn test_explain_values_source_filter() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN (VALUES (1, 'x'), (2, 'y')) AS v(vid, vname) ON v.vid = u.id WHERE u.name = 'Alice'",
    );
    let plan = lines.join("\n");

    // Values scan node should be present
    let has_values_scan = lines.iter().any(|l| l.contains("Values Scan"));
    assert!(
        has_values_scan,
        "Should have Values Scan node.\nFull plan:\n{}",
        plan
    );

    // u.name filter must NOT appear under Values Scan
    let values_idx = lines.iter().position(|l| l.contains("Values Scan"));
    if let Some(idx) = values_idx {
        assert_no_filter_under_scan(&lines, idx, "u.name", &plan);
    }

    // u.name filter should appear under users scan
    assert_filter_under_node(&lines, "Seq Scan on users", "name = 'Alice'", &plan);
}

// ============================================================================
// LEFT/RIGHT/FULL JOIN nullable-side WHERE predicates
// ============================================================================

#[test]
fn test_explain_left_join_nullable_side_filter() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT u.id FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.status IS NULL ORDER BY u.id",
    );
    let plan = lines.join("\n");

    // o.status IS NULL is a post-join filter on the nullable side — must NOT
    // appear as a scan filter under orders
    let orders_idx = lines.iter().position(|l| l.contains("orders"));
    if let Some(idx) = orders_idx {
        assert_no_filter_under_scan(&lines, idx, "o.status", &plan);
    }

    // Should appear as a Join Filter instead
    let has_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("status"));
    assert!(
        has_join_filter,
        "Nullable-side predicate should appear as 'Join Filter'.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_right_join_nullable_side_filter() {
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT o.id FROM orders o RIGHT JOIN users u ON o.user_id = u.id WHERE o.status IS NULL",
    );
    let plan = lines.join("\n");

    // o is on the nullable (left) side of a RIGHT JOIN
    let orders_idx = lines.iter().position(|l| l.contains("orders"));
    if let Some(idx) = orders_idx {
        assert_no_filter_under_scan(&lines, idx, "o.status", &plan);
    }

    let has_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("status"));
    assert!(
        has_join_filter,
        "Nullable-side predicate should appear as 'Join Filter'.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_inner_join_filter_still_pushed_down() {
    let db = setup_db();
    // INNER JOIN: both sides are non-nullable, filters should still be pushed
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'paid'",
    );
    let plan = lines.join("\n");

    // For INNER JOIN, o.status filter should NOT be a Join Filter
    let is_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("status"));
    assert!(
        !is_join_filter,
        "INNER JOIN filter should be pushed to scan, not Join Filter.\nFull plan:\n{}",
        plan
    );
}

// ============================================================================
// GROUP BY ROLLUP / CUBE / GROUPING SETS in EXPLAIN
// ============================================================================

#[test]
fn test_explain_grouping_sets() {
    let db = Database::open("memory://explain_grouping_sets").expect("db");
    db.execute(
        "CREATE TABLE sales (region TEXT, product TEXT, amount FLOAT)",
        (),
    )
    .expect("create");
    db.execute(
        "INSERT INTO sales VALUES ('US', 'A', 100), ('EU', 'B', 200)",
        (),
    )
    .expect("insert");

    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT region, product, SUM(amount) FROM sales GROUP BY GROUPING SETS ((region, product), (region), ())",
    );
    let plan = lines.join("\n");

    let has_grouping_sets = lines.iter().any(|l| l.contains("GROUPING SETS"));
    assert!(
        has_grouping_sets,
        "EXPLAIN should show GROUPING SETS.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_rollup() {
    let db = Database::open("memory://explain_rollup").expect("db");
    db.execute(
        "CREATE TABLE sales2 (region TEXT, product TEXT, amount FLOAT)",
        (),
    )
    .expect("create");

    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT region, product, SUM(amount) FROM sales2 GROUP BY ROLLUP(region, product)",
    );
    let plan = lines.join("\n");

    let has_rollup = lines.iter().any(|l| l.contains("ROLLUP"));
    assert!(
        has_rollup,
        "EXPLAIN should show ROLLUP.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_cube() {
    let db = Database::open("memory://explain_cube").expect("db");
    db.execute(
        "CREATE TABLE sales3 (region TEXT, product TEXT, amount FLOAT)",
        (),
    )
    .expect("create");

    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT region, product, SUM(amount) FROM sales3 GROUP BY CUBE(region, product)",
    );
    let plan = lines.join("\n");

    let has_cube = lines.iter().any(|l| l.contains("CUBE"));
    assert!(has_cube, "EXPLAIN should show CUBE.\nFull plan:\n{}", plan);
}

// ============================================================================
// Composite index prefix usage
// ============================================================================

#[test]
fn test_explain_composite_index_prefix() {
    let db = Database::open("memory://explain_composite_prefix").expect("db");
    db.execute(
        "CREATE TABLE orders2 (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("create");
    db.execute(
        "CREATE INDEX idx_region_amount ON orders2(region, amount)",
        (),
    )
    .expect("create index");
    db.execute(
        "INSERT INTO orders2 VALUES (1, 'US', 100), (2, 'EU', 200), (3, 'US', 300)",
        (),
    )
    .expect("insert");

    // Single prefix column equality should use the composite index
    let lines = get_explain_lines(&db, "EXPLAIN SELECT * FROM orders2 WHERE region = 'EU'");
    let plan = lines.join("\n");

    let uses_index = lines
        .iter()
        .any(|l| l.contains("Composite Index Scan") || l.contains("Index Scan"));
    assert!(
        uses_index,
        "Single prefix column should use composite index.\nFull plan:\n{}",
        plan
    );

    // Should NOT be a Seq Scan
    let has_seq_scan_with_filter = lines
        .iter()
        .any(|l| l.contains("Seq Scan") && !l.contains("Index"));
    assert!(
        !has_seq_scan_with_filter,
        "Should not fall back to Seq Scan when prefix matches composite index.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_composite_index_trailing_range() {
    let db = Database::open("memory://explain_trailing_range").expect("db");
    db.execute(
        "CREATE TABLE o2 (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("create");
    db.execute(
        "CREATE INDEX idx_o2_region_amount ON o2(region, amount)",
        (),
    )
    .expect("create index");
    db.execute(
        "INSERT INTO o2 VALUES (1, 'US', 100), (2, 'EU', 200), (3, 'EU', 600), (4, 'US', 800)",
        (),
    )
    .expect("insert");

    // Equality prefix + trailing range should show both columns in the index scan
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM o2 WHERE region = 'EU' AND amount > 500",
    );
    let plan = lines.join("\n");

    let uses_composite = lines.iter().any(|l| l.contains("Composite Index Scan"));
    assert!(
        uses_composite,
        "Equality prefix + trailing range should use composite index.\nFull plan:\n{}",
        plan
    );

    // Both columns should appear in the Columns line
    let has_both = lines
        .iter()
        .any(|l| l.contains("region") && l.contains("amount"));
    assert!(
        has_both,
        "Both region and amount should appear in the index scan plan.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_index_scan_residual_filter() {
    let db = Database::open("memory://explain_residual_filter").expect("db");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, region TEXT, day INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_region ON t(region)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO t VALUES (1, 'EU', 1), (2, 'EU', 2), (3, 'US', 3)",
        (),
    )
    .expect("insert");

    // Index on region only; day=2 is a residual filter
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM t WHERE region = 'EU' AND day = 2",
    );
    let plan = lines.join("\n");

    // Should use the index for region
    let uses_index = lines
        .iter()
        .any(|l| l.contains("Index Scan") || l.contains("Composite Index"));
    assert!(
        uses_index,
        "Should use index scan for region.\nFull plan:\n{}",
        plan
    );

    // Residual filter on day should be visible
    let has_day_filter = lines
        .iter()
        .any(|l| l.contains("day") && l.contains("Filter"));
    assert!(
        has_day_filter,
        "Residual predicate 'day = 2' should appear as Filter.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_composite_index_residual_filter() {
    let db = Database::open("memory://explain_composite_residual").expect("db");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT, day INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_region_amount ON t2(region, amount)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO t2 VALUES (1, 'EU', 100, 1), (2, 'EU', 200, 2), (3, 'US', 300, 3)",
        (),
    )
    .expect("insert");

    // Composite index covers region + amount; day is residual
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM t2 WHERE region = 'EU' AND amount > 50 AND day = 2",
    );
    let plan = lines.join("\n");

    let uses_composite = lines.iter().any(|l| l.contains("Composite Index Scan"));
    assert!(
        uses_composite,
        "Should use composite index.\nFull plan:\n{}",
        plan
    );

    // Residual filter on day should be visible
    let has_day_filter = lines
        .iter()
        .any(|l| l.contains("day") && l.contains("Filter"));
    assert!(
        has_day_filter,
        "Residual predicate 'day = 2' should appear as Filter.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_explain_composite_index_between_trailing_range() {
    let db = Database::open("memory://explain_between_trailing").expect("db");
    db.execute(
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_metrics ON metrics(region, amount)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO metrics VALUES (1,'US',100),(2,'EU',200),(3,'EU',550),(4,'EU',700),(5,'US',800)",
        (),
    )
    .expect("insert");

    // BETWEEN should decompose into Gte/Lte and be recognized as trailing range
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT id FROM metrics WHERE region = 'EU' AND amount BETWEEN 500 AND 650",
    );
    let plan = lines.join("\n");

    let uses_composite = lines.iter().any(|l| l.contains("Composite Index Scan"));
    assert!(
        uses_composite,
        "BETWEEN trailing range should use composite index.\nFull plan:\n{}",
        plan
    );

    // Both columns should appear
    let has_both = lines
        .iter()
        .any(|l| l.contains("region") && l.contains("amount"));
    assert!(
        has_both,
        "Both region and amount should appear in composite index scan.\nFull plan:\n{}",
        plan
    );

    // Verify correctness: only row 3 (EU, 550) matches
    let mut rows = db
        .query(
            "SELECT id FROM metrics WHERE region = 'EU' AND amount BETWEEN 500 AND 650",
            (),
        )
        .expect("query");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(
        ids,
        vec![3],
        "Only id=3 (EU, 550) should match BETWEEN 500 AND 650"
    );
}

#[test]
fn test_explain_mixed_or_hybrid_scan() {
    let db = Database::open("memory://explain_mixed_or").expect("db");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT, score INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_status ON items(status)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO items VALUES (1,'active',10),(2,'inactive',20),(3,'active',30),(4,'pending',40),(5,'inactive',50)",
        (),
    )
    .expect("insert");

    // Mixed OR: one indexed, one non-indexed
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM items WHERE status = 'active' OR score > 30",
    );
    let plan = lines.join("\n");

    // Should NOT be a Seq Scan
    assert!(
        !plan.contains("Seq Scan"),
        "Mixed OR should not fall back to Seq Scan.\nFull plan:\n{}",
        plan
    );

    // Should show Multi-Index Scan with indexed branch
    assert!(
        plan.contains("Multi-Index Scan"),
        "Mixed OR should use Multi-Index Scan.\nFull plan:\n{}",
        plan
    );
    assert!(
        plan.contains("idx_status"),
        "Indexed branch should show index name.\nFull plan:\n{}",
        plan
    );

    // Non-indexed branch should appear as Filter
    assert!(
        plan.contains("Filter:"),
        "Non-indexed branch should appear as Filter.\nFull plan:\n{}",
        plan
    );

    // Verify query correctness: ids 1,3 (active) + 4,5 (score>30) = {1,3,4,5}
    let mut rows = db
        .query(
            "SELECT id FROM items WHERE status = 'active' OR score > 30 ORDER BY id",
            (),
        )
        .expect("query");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![1, 3, 4, 5]);
}

#[test]
fn test_composite_index_lower_bound_only_no_panic() {
    // Regression: WHERE a = 5 AND b > 10 on composite index (a, b) caused
    // BTree panic due to inverted range: min_key=[5,10] > max_key=[5]
    let db = Database::open("memory://composite_lower_bound").expect("db");
    db.execute(
        "CREATE TABLE t3 (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_ab ON t3(a, b)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO t3 VALUES (1,5,5),(2,5,10),(3,5,15),(4,5,20),(5,7,1)",
        (),
    )
    .expect("insert");

    // Lower-bound only: b > 10 — must not panic
    let mut rows = db
        .query("SELECT id FROM t3 WHERE a = 5 AND b > 10 ORDER BY id", ())
        .expect("query should not panic");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![3, 4], "a=5 AND b>10 should return ids 3,4");

    // Lower-bound inclusive: b >= 10
    let mut rows = db
        .query("SELECT id FROM t3 WHERE a = 5 AND b >= 10 ORDER BY id", ())
        .expect("query");
    let mut ids2 = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids2.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids2, vec![2, 3, 4], "a=5 AND b>=10 should return ids 2,3,4");

    // Upper-bound only: b < 15 (this was already working)
    let mut rows = db
        .query("SELECT id FROM t3 WHERE a = 5 AND b < 15 ORDER BY id", ())
        .expect("query");
    let mut ids3 = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids3.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids3, vec![1, 2], "a=5 AND b<15 should return ids 1,2");

    // Both bounds: b > 5 AND b < 20 (BETWEEN-like)
    let mut rows = db
        .query(
            "SELECT id FROM t3 WHERE a = 5 AND b > 5 AND b < 20 ORDER BY id",
            (),
        )
        .expect("query");
    let mut ids4 = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids4.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(
        ids4,
        vec![2, 3],
        "a=5 AND b>5 AND b<20 should return ids 2,3"
    );
}

#[test]
fn test_correlated_exists_case_outer_ref_with_index() {
    // Regression: CASE expressions containing outer-table references inside
    // EXISTS subqueries were not detected by predicate_has_outer_refs_or_subqueries,
    // allowing the index-probe fast path to mishandle them. This produced wrong
    // results when an index existed on the inner table's correlation column.
    let db = Database::open("memory://case_outer_ref").expect("db");
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, low INTEGER)", ())
        .expect("create u");
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        (),
    )
    .expect("create o");
    // Index on the correlation column triggers the fast path
    db.execute("CREATE INDEX idx_o_uid ON o(user_id)", ())
        .expect("create index");
    db.execute("INSERT INTO u VALUES (1, 50), (2, 200)", ())
        .expect("insert u");
    db.execute("INSERT INTO o VALUES (1, 1, 100), (2, 2, 100)", ())
        .expect("insert o");

    // u.id=1, u.low=50: o.amount(100) > u.low(50) => CASE=1 => EXISTS true
    // u.id=2, u.low=200: o.amount(100) > u.low(200) => false => CASE=0 => EXISTS false
    // Correct result: [1]
    // Bug result with index: [2] or [1,2] depending on misbinding
    let mut rows = db
        .query(
            "SELECT id FROM u WHERE EXISTS (\
                SELECT 1 FROM o WHERE o.user_id = u.id \
                AND (CASE WHEN o.amount > u.low THEN 1 ELSE 0 END) = 1\
            ) ORDER BY id",
            (),
        )
        .expect("query");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(
        ids,
        vec![1],
        "Only u.id=1 has o.amount(100) > u.low(50). u.id=2 has u.low=200 > o.amount(100)."
    );
}

#[test]
fn test_explain_mixed_qualified_unqualified_cross_table() {
    // Regression: `WHERE o.status = name` has qualifier "o" (right side) but
    // unqualified "name" belongs to left side (users). This is a cross-table
    // predicate and must appear as Join Filter, not under the orders scan.
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN orders o ON o.user_id = u.id WHERE o.status = name",
    );
    let plan = lines.join("\n");

    // Should be a Join Filter (cross-table), not under orders scan
    let orders_idx = lines.iter().position(|l| l.contains("orders"));
    if let Some(idx) = orders_idx {
        assert_no_filter_under_scan(&lines, idx, "name", &plan);
    }

    let has_join_filter = lines
        .iter()
        .any(|l| l.contains("Join Filter:") || l.contains("Filter:"));
    assert!(
        has_join_filter,
        "Mixed qual/unqual cross-table predicate should appear as Join Filter.\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_mixed_or_hybrid_respects_limit() {
    // Regression: mixed-OR phase 2 materialized usize::MAX rows before dedup.
    // Verify correctness with LIMIT.
    let db = Database::open("memory://mixed_or_limit").expect("db");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, status TEXT, score INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_big_status ON big(status)", ())
        .expect("create index");
    // Insert enough rows so the limit matters
    for i in 1..=100 {
        let status = if i % 3 == 0 { "active" } else { "inactive" };
        db.execute(
            &format!("INSERT INTO big VALUES ({}, '{}', {})", i, status, i * 10),
            (),
        )
        .expect("insert");
    }

    // Mixed OR with LIMIT: index handles status='active', scan handles score>800
    let mut rows = db
        .query(
            "SELECT id FROM big WHERE status = 'active' OR score > 800 ORDER BY id LIMIT 5",
            (),
        )
        .expect("query");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert!(
        ids.len() <= 5,
        "LIMIT 5 should return at most 5 rows, got {}",
        ids.len()
    );
    assert!(!ids.is_empty(), "Should return some rows");

    // Verify all returned rows actually match the OR condition
    for &id in &ids {
        let matches_status = id % 3 == 0; // "active" rows
        let matches_score = id * 10 > 800; // score > 800
        assert!(
            matches_status || matches_score,
            "id={} doesn't match either OR branch",
            id
        );
    }
}

#[test]
fn test_explain_case_predicate_pushed_to_correct_scan() {
    // Regression: WHERE CASE WHEN o.status = 'paid' THEN 1 ELSE 0 END = 1
    // references only orders columns (o.status) so it should be pushed to the
    // orders scan, NOT stay as a Join Filter.
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN orders o ON o.user_id = u.id \
         WHERE CASE WHEN o.status = 'paid' THEN 1 ELSE 0 END = 1",
    );
    let plan = lines.join("\n");

    // The CASE predicate should be under the orders scan, not at join level
    let orders_scan_idx = lines.iter().position(|l| l.contains("orders"));
    assert!(
        orders_scan_idx.is_some(),
        "Plan should contain orders scan.\nFull plan:\n{}",
        plan
    );

    // Look for the filter/CASE near the orders scan (within a few lines)
    let idx = orders_scan_idx.unwrap();
    let near_orders = lines[idx..std::cmp::min(idx + 4, lines.len())].join(" ");
    let filter_near_orders = near_orders.contains("Filter:") || near_orders.contains("CASE");
    assert!(
        filter_near_orders,
        "CASE predicate on 'o.status' should be pushed to orders scan, not Join Filter.\nFull plan:\n{}",
        plan
    );

    // The Join Filter line should NOT contain the CASE predicate
    let join_filter_has_case = lines
        .iter()
        .any(|l| l.contains("Join Filter:") && l.contains("CASE"));
    assert!(
        !join_filter_has_case,
        "CASE predicate should NOT appear as Join Filter (it's single-table).\nFull plan:\n{}",
        plan
    );

    // Verify correctness: only orders with status='paid' should appear
    let mut rows = db
        .query(
            "SELECT o.id FROM users u JOIN orders o ON o.user_id = u.id \
             WHERE CASE WHEN o.status = 'paid' THEN 1 ELSE 0 END = 1",
            (),
        )
        .expect("query");
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![1], "Only order id=1 (status='paid') should match");
}

#[test]
fn test_explain_in_cross_table_as_join_filter() {
    // Regression: WHERE o.status IN (name) has qualifier "o" (orders) and
    // unqualified "name" (users column). This is a cross-table predicate
    // and must appear as Join Filter, not under the orders scan.
    let db = setup_db();
    let lines = get_explain_lines(
        &db,
        "EXPLAIN SELECT * FROM users u JOIN orders o ON o.user_id = u.id \
         WHERE o.status IN (name)",
    );
    let plan = lines.join("\n");

    // This should be a Join Filter (cross-table), not under orders scan
    let orders_idx = lines.iter().position(|l| l.contains("orders"));
    if let Some(idx) = orders_idx {
        assert_no_filter_under_scan(&lines, idx, "name", &plan);
    }

    // Should appear at join level
    let has_join_filter = lines.iter().any(|l| {
        (l.contains("Join Filter:") || l.contains("Filter:"))
            && (l.contains("status") || l.contains("name") || l.contains("IN"))
    });
    assert!(
        has_join_filter,
        "Cross-table IN predicate should appear as Join/Filter (not under scan).\nFull plan:\n{}",
        plan
    );
}

#[test]
fn test_limit_zero_returns_no_rows() {
    // Regression: LIMIT 0 leaked one row through index and filter fast paths.
    let db = Database::open("memory://limit_zero").expect("db");
    db.execute(
        "CREATE TABLE lz (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_a ON lz(a)", ()).expect("idx");
    db.execute("INSERT INTO lz VALUES (1,10,'x'),(2,20,'y'),(3,10,'z')", ())
        .expect("insert");

    // Index fast path: WHERE on indexed column
    let mut rows = db
        .query("SELECT id FROM lz WHERE a = 10 LIMIT 0", ())
        .expect("query");
    assert!(
        rows.next().is_none(),
        "LIMIT 0 with index filter must return 0 rows"
    );

    // Seq scan + filter fast path: WHERE on non-indexed column
    let mut rows = db
        .query("SELECT id FROM lz WHERE b = 'x' LIMIT 0", ())
        .expect("query");
    assert!(
        rows.next().is_none(),
        "LIMIT 0 with seq scan filter must return 0 rows"
    );

    // Range scan fast path
    let mut rows = db
        .query("SELECT id FROM lz WHERE id > 0 LIMIT 0", ())
        .expect("query");
    assert!(
        rows.next().is_none(),
        "LIMIT 0 with PK range must return 0 rows"
    );

    // No filter
    let mut rows = db.query("SELECT id FROM lz LIMIT 0", ()).expect("query");
    assert!(
        rows.next().is_none(),
        "LIMIT 0 without filter must return 0 rows"
    );

    // Mixed OR fast path
    let mut rows = db
        .query("SELECT id FROM lz WHERE a = 10 OR b = 'y' LIMIT 0", ())
        .expect("query");
    assert!(
        rows.next().is_none(),
        "LIMIT 0 with mixed OR must return 0 rows"
    );
}
