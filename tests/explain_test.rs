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

//! Tests for EXPLAIN and EXPLAIN ANALYZE functionality

use std::sync::atomic::{AtomicU64, Ordering};
use stoolap::Database;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn setup_test_db() -> Database {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let db = Database::open(&format!("memory://explain_test_{}", id))
        .expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            status TEXT,
            amount DECIMAL(10,2)
        )",
        (),
    )
    .expect("Failed to create table");

    // Create indexes
    db.execute("CREATE INDEX idx_user_id ON orders(user_id)", ())
        .expect("Failed to create user_id index");
    db.execute("CREATE INDEX idx_status ON orders(status)", ())
        .expect("Failed to create status index");

    // Insert test data
    for i in 1..=10 {
        let user_id = (i % 3) + 1; // user_id 1, 2, or 3
        let status = match i % 3 {
            0 => "active",
            1 => "shipped",
            _ => "pending",
        };
        let amount = i as f64 * 10.0;
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, '{}', {})",
                i, user_id, status, amount
            ),
            (),
        )
        .expect("Failed to insert row");
    }

    db
}

fn get_plan_output(db: &Database, query: &str) -> Vec<String> {
    let result = db.query(query, ()).expect("Failed to execute EXPLAIN");
    let mut lines = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let plan_line: String = row.get(0).unwrap_or_default();
        lines.push(plan_line);
    }
    lines
}

#[test]
fn test_explain_seq_scan() {
    let db = setup_test_db();

    // Query on non-indexed column should show Seq Scan
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders WHERE amount > 50");

    let plan = lines.join("\n");
    assert!(
        plan.contains("Seq Scan on orders"),
        "Expected Seq Scan, got:\n{}",
        plan
    );
    assert!(plan.contains("Filter:"), "Expected Filter, got:\n{}", plan);
}

#[test]
fn test_explain_index_scan() {
    let db = setup_test_db();

    // Query on indexed column should show Index Scan
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders WHERE user_id = 1");

    let plan = lines.join("\n");
    assert!(
        plan.contains("Index Scan"),
        "Expected Index Scan, got:\n{}",
        plan
    );
    assert!(
        plan.contains("idx_user_id"),
        "Expected idx_user_id index, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Index Cond:"),
        "Expected Index Cond, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_pk_lookup() {
    let db = setup_test_db();

    // Query on primary key should show PK Lookup
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders WHERE id = 5");

    let plan = lines.join("\n");
    assert!(
        plan.contains("PK Lookup"),
        "Expected PK Lookup, got:\n{}",
        plan
    );
    assert!(plan.contains("id = 5"), "Expected id = 5, got:\n{}", plan);
}

#[test]
fn test_explain_multi_index_or() {
    let db = setup_test_db();

    // OR with multiple indexed columns should show Multi-Index Scan
    let lines = get_plan_output(
        &db,
        "EXPLAIN SELECT * FROM orders WHERE user_id = 1 OR status = 'active'",
    );

    let plan = lines.join("\n");
    assert!(
        plan.contains("Multi-Index Scan"),
        "Expected Multi-Index Scan, got:\n{}",
        plan
    );
    assert!(plan.contains("OR"), "Expected OR operation, got:\n{}", plan);
}

#[test]
fn test_explain_multi_index_and() {
    let db = setup_test_db();

    // AND with multiple indexed columns should show Multi-Index Scan
    let lines = get_plan_output(
        &db,
        "EXPLAIN SELECT * FROM orders WHERE user_id = 1 AND status = 'active'",
    );

    let plan = lines.join("\n");
    // Could be either Multi-Index Scan or single Index Scan depending on optimizer
    assert!(
        plan.contains("Index Scan") || plan.contains("Multi-Index Scan"),
        "Expected Index Scan or Multi-Index Scan, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_range_query() {
    let db = setup_test_db();

    // Range query on indexed column should show Index Scan
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders WHERE user_id > 1");

    let plan = lines.join("\n");
    assert!(
        plan.contains("Index Scan"),
        "Expected Index Scan, got:\n{}",
        plan
    );
    assert!(
        plan.contains("> 1"),
        "Expected > 1 condition, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_analyze_seq_scan() {
    let db = setup_test_db();

    // EXPLAIN ANALYZE on non-indexed column
    let lines = get_plan_output(
        &db,
        "EXPLAIN ANALYZE SELECT * FROM orders WHERE amount > 50",
    );

    let plan = lines.join("\n");
    assert!(
        plan.contains("actual time="),
        "Expected actual time, got:\n{}",
        plan
    );
    assert!(
        plan.contains("actual rows="),
        "Expected actual rows, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Seq Scan"),
        "Expected Seq Scan, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_analyze_index_scan() {
    let db = setup_test_db();

    // EXPLAIN ANALYZE on indexed column
    let lines = get_plan_output(
        &db,
        "EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 2",
    );

    let plan = lines.join("\n");
    assert!(
        plan.contains("actual time="),
        "Expected actual time, got:\n{}",
        plan
    );
    assert!(
        plan.contains("actual rows="),
        "Expected actual rows, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Index Scan"),
        "Expected Index Scan, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_analyze_pk_lookup() {
    let db = setup_test_db();

    // EXPLAIN ANALYZE on primary key
    let lines = get_plan_output(&db, "EXPLAIN ANALYZE SELECT * FROM orders WHERE id = 5");

    let plan = lines.join("\n");
    assert!(
        plan.contains("actual time="),
        "Expected actual time, got:\n{}",
        plan
    );
    assert!(plan.contains("rows=1"), "Expected rows=1, got:\n{}", plan);
    assert!(
        plan.contains("PK Lookup"),
        "Expected PK Lookup, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_analyze_empty_result() {
    let db = setup_test_db();

    // EXPLAIN ANALYZE with no matching rows
    let lines = get_plan_output(&db, "EXPLAIN ANALYZE SELECT * FROM orders WHERE id = 999");

    let plan = lines.join("\n");
    assert!(plan.contains("rows=0"), "Expected rows=0, got:\n{}", plan);
}

#[test]
fn test_explain_order_by() {
    let db = setup_test_db();

    // Query with ORDER BY should show Order clause
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders ORDER BY user_id DESC");

    let plan = lines.join("\n");
    assert!(
        plan.contains("Order By:"),
        "Expected Order By clause, got:\n{}",
        plan
    );
    assert!(
        plan.contains("DESC"),
        "Expected DESC in order, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_limit() {
    let db = setup_test_db();

    // Query with LIMIT should show Limit clause
    let lines = get_plan_output(&db, "EXPLAIN SELECT * FROM orders LIMIT 5");

    let plan = lines.join("\n");
    assert!(
        plan.contains("Limit:"),
        "Expected Limit clause, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_group_by() {
    let db = setup_test_db();

    // Query with GROUP BY should show Group clause
    let lines = get_plan_output(
        &db,
        "EXPLAIN SELECT user_id, COUNT(*) FROM orders GROUP BY user_id",
    );

    let plan = lines.join("\n");
    assert!(
        plan.contains("Group By:"),
        "Expected Group By clause, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_join() {
    let db = setup_test_db();

    // Create another table for join
    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create users table");

    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .expect("Failed to insert users");

    // Query with JOIN
    let lines = get_plan_output(
        &db,
        "EXPLAIN SELECT * FROM orders o JOIN users u ON o.user_id = u.id",
    );

    let plan = lines.join("\n");
    assert!(plan.contains("Join"), "Expected Join, got:\n{}", plan);
}
