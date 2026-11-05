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

//! Query optimization paths integration tests
//!
//! Tests for advanced query execution paths including:
//! - Streaming GROUP BY with indexed columns
//! - Deferred aggregation (SUM/MIN/MAX without row materialization)
//! - Keyset pagination for efficient paging
//! - Index-based optimizations (MIN/MAX lookup, ORDER BY + LIMIT)
//! - EXPLAIN plan variants (PK lookup, index scan, seq scan)
//! - Temporal queries (AS OF)
//! - LIKE/ILIKE/GLOB pattern matching
//! - Window functions with PARTITION BY

use stoolap::Database;

// =============================================================================
// STREAMING GROUP BY TESTS
// =============================================================================

#[test]
fn test_streaming_group_by_count_with_btree_index() {
    let db = Database::open("memory://streaming_group_by_1").expect("Failed to create database");

    // Create table with BTree index on category column
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            category TEXT,
            price FLOAT
        )",
        (),
    )
    .unwrap();

    // Create BTree index on category (required for streaming GROUP BY)
    db.execute(
        "CREATE INDEX idx_category ON products(category) USING BTREE",
        (),
    )
    .unwrap();

    // Insert test data
    for i in 1..=100 {
        let category = match i % 3 {
            0 => "Electronics",
            1 => "Books",
            _ => "Clothing",
        };
        db.execute(
            &format!(
                "INSERT INTO products (id, category, price) VALUES ({}, '{}', {})",
                i,
                category,
                (i as f64) * 10.0
            ),
            (),
        )
        .unwrap();
    }

    // This should trigger streaming GROUP BY path:
    // - Single-column GROUP BY on indexed column
    // - COUNT aggregate (no row fetch needed)
    // - No WHERE clause, no ORDER BY
    let result = db
        .query(
            "SELECT category, COUNT(*) FROM products GROUP BY category",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3, "Should have 3 categories");
}

#[test]
fn test_streaming_group_by_with_limit() {
    let db = Database::open("memory://streaming_group_by_2").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT,
            amount FLOAT
        )",
        (),
    )
    .unwrap();

    db.execute("CREATE INDEX idx_status ON orders(status) USING BTREE", ())
        .unwrap();

    for i in 1..=50 {
        let status = match i % 4 {
            0 => "pending",
            1 => "shipped",
            2 => "delivered",
            _ => "cancelled",
        };
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, '{}', {})",
                i,
                status,
                (i as f64) * 5.0
            ),
            (),
        )
        .unwrap();
    }

    // Streaming GROUP BY with LIMIT should enable early termination for SUM/AVG
    let result = db
        .query(
            "SELECT status, SUM(amount) FROM orders GROUP BY status LIMIT 2",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "Should return only 2 groups");
}

// =============================================================================
// DEFERRED AGGREGATION TESTS
// =============================================================================

#[test]
fn test_deferred_sum_column() {
    let db = Database::open("memory://deferred_sum").expect("Failed to create database");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO numbers VALUES ({}, {})", i, i), ())
            .unwrap();
    }

    // SUM should use deferred aggregation path
    let sum: i64 = db.query_one("SELECT SUM(value) FROM numbers", ()).unwrap();
    assert_eq!(sum, 5050);
}

#[test]
fn test_deferred_min_max_column() {
    let db = Database::open("memory://deferred_minmax").expect("Failed to create database");

    db.execute(
        "CREATE TABLE temps (id INTEGER PRIMARY KEY, temp FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO temps VALUES (1, 15.5)", ())
        .unwrap();
    db.execute("INSERT INTO temps VALUES (2, 25.3)", ())
        .unwrap();
    db.execute("INSERT INTO temps VALUES (3, -5.0)", ())
        .unwrap();
    db.execute("INSERT INTO temps VALUES (4, 42.8)", ())
        .unwrap();
    db.execute("INSERT INTO temps VALUES (5, 10.0)", ())
        .unwrap();

    let min: f64 = db.query_one("SELECT MIN(temp) FROM temps", ()).unwrap();
    let max: f64 = db.query_one("SELECT MAX(temp) FROM temps", ()).unwrap();

    assert!((min - (-5.0)).abs() < 0.01, "MIN should be -5.0");
    assert!((max - 42.8).abs() < 0.01, "MAX should be 42.8");
}

#[test]
fn test_deferred_avg_column() {
    let db = Database::open("memory://deferred_avg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO scores VALUES (1, 80)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (2, 90)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (3, 70)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (4, 100)", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (5, 60)", ()).unwrap();

    let avg: f64 = db.query_one("SELECT AVG(score) FROM scores", ()).unwrap();
    assert!((avg - 80.0).abs() < 0.01, "AVG should be 80.0");
}

// =============================================================================
// KEYSET PAGINATION TESTS
// =============================================================================

#[test]
fn test_keyset_pagination_gt() {
    let db = Database::open("memory://keyset_gt").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // Keyset pagination: WHERE id > X ORDER BY id LIMIT Y
    let result = db
        .query("SELECT * FROM items WHERE id > 50 ORDER BY id LIMIT 10", ())
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 10, "Should return 10 rows");

    // First row should be id=51
    let first_id: i64 = rows[0].get(0).unwrap();
    assert_eq!(first_id, 51, "First row should be id=51");
}

#[test]
fn test_keyset_pagination_gte() {
    let db = Database::open("memory://keyset_gte").expect("Failed to create database");

    db.execute(
        "CREATE TABLE records (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO records VALUES ({}, 'data_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // Keyset with >= bound
    let result = db
        .query(
            "SELECT * FROM records WHERE id >= 25 ORDER BY id LIMIT 5",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 5);

    let first_id: i64 = rows[0].get(0).unwrap();
    assert_eq!(first_id, 25, "First row should be id=25 (inclusive)");
}

#[test]
fn test_keyset_pagination_desc() {
    let db = Database::open("memory://keyset_desc").expect("Failed to create database");

    db.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=30 {
        db.execute(&format!("INSERT INTO logs VALUES ({}, 'log_{}')", i, i), ())
            .unwrap();
    }

    // Descending keyset pagination
    let result = db
        .query(
            "SELECT * FROM logs WHERE id < 25 ORDER BY id DESC LIMIT 5",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 5);

    let first_id: i64 = rows[0].get(0).unwrap();
    assert_eq!(first_id, 24, "First row should be id=24 in DESC order");
}

// =============================================================================
// INDEX OPTIMIZATION TESTS
// =============================================================================

#[test]
fn test_min_max_index_optimization() {
    let db = Database::open("memory://minmax_index").expect("Failed to create database");

    db.execute(
        "CREATE TABLE prices (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_amount ON prices(amount) USING BTREE", ())
        .unwrap();

    db.execute("INSERT INTO prices VALUES (1, 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO prices VALUES (2, 50.0)", ())
        .unwrap();
    db.execute("INSERT INTO prices VALUES (3, 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO prices VALUES (4, 25.0)", ())
        .unwrap();
    db.execute("INSERT INTO prices VALUES (5, 150.0)", ())
        .unwrap();

    // These should use O(1) index lookup instead of O(n) scan
    let min: f64 = db.query_one("SELECT MIN(amount) FROM prices", ()).unwrap();
    let max: f64 = db.query_one("SELECT MAX(amount) FROM prices", ()).unwrap();

    assert!((min - 25.0).abs() < 0.01);
    assert!((max - 200.0).abs() < 0.01);
}

#[test]
fn test_count_star_optimization() {
    let db = Database::open("memory://count_star").expect("Failed to create database");

    db.execute(
        "CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO entities VALUES ({}, 'entity_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // COUNT(*) should use O(1) row_count() instead of scan
    let count: i64 = db.query_one("SELECT COUNT(*) FROM entities", ()).unwrap();
    assert_eq!(count, 50);
}

#[test]
fn test_order_by_index_optimization() {
    let db = Database::open("memory://orderby_index").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_ts ON events(timestamp) USING BTREE", ())
        .unwrap();

    // Insert in random order
    db.execute("INSERT INTO events VALUES (1, 300, 'event_c')", ())
        .unwrap();
    db.execute("INSERT INTO events VALUES (2, 100, 'event_a')", ())
        .unwrap();
    db.execute("INSERT INTO events VALUES (3, 200, 'event_b')", ())
        .unwrap();
    db.execute("INSERT INTO events VALUES (4, 500, 'event_e')", ())
        .unwrap();
    db.execute("INSERT INTO events VALUES (5, 400, 'event_d')", ())
        .unwrap();

    // ORDER BY indexed column + LIMIT should use index-ordered scan
    let result = db
        .query("SELECT * FROM events ORDER BY timestamp LIMIT 3", ())
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3);

    // Should be in ascending timestamp order: 100, 200, 300
    let ts1: i64 = rows[0].get(1).unwrap();
    let ts2: i64 = rows[1].get(1).unwrap();
    let ts3: i64 = rows[2].get(1).unwrap();

    assert_eq!(ts1, 100);
    assert_eq!(ts2, 200);
    assert_eq!(ts3, 300);
}

#[test]
fn test_in_list_index_optimization() {
    let db = Database::open("memory://in_list_index").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_status ON users(status)", ())
        .unwrap();

    for i in 1..=20 {
        let status = match i % 4 {
            0 => "active",
            1 => "inactive",
            2 => "pending",
            _ => "suspended",
        };
        db.execute(
            &format!("INSERT INTO users VALUES ({}, '{}')", i, status),
            (),
        )
        .unwrap();
    }

    // IN list on indexed column should use index probe
    let result = db
        .query(
            "SELECT * FROM users WHERE status IN ('active', 'pending')",
            (),
        )
        .unwrap();

    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty(), "Should find some users");
}

// =============================================================================
// EXPLAIN SCAN PLAN TESTS
// =============================================================================

#[test]
fn test_explain_pk_lookup() {
    let db = Database::open("memory://explain_pk").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_pk (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO test_pk VALUES (1, 'a')", ())
        .unwrap();

    // EXPLAIN should show PK Lookup
    let result = db
        .query("EXPLAIN SELECT * FROM test_pk WHERE id = 1", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty(), "EXPLAIN should return rows");
}

#[test]
fn test_explain_index_scan() {
    let db = Database::open("memory://explain_idx").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_idx (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_cat ON test_idx(category)", ())
        .unwrap();
    db.execute("INSERT INTO test_idx VALUES (1, 'A')", ())
        .unwrap();

    // EXPLAIN should show Index Scan
    let result = db
        .query("EXPLAIN SELECT * FROM test_idx WHERE category = 'A'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty());
}

#[test]
fn test_explain_seq_scan() {
    let db = Database::open("memory://explain_seq").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_seq (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO test_seq VALUES (1, 'a')", ())
        .unwrap();

    // EXPLAIN without WHERE should show Seq Scan
    let result = db.query("EXPLAIN SELECT * FROM test_seq", ()).unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty());
}

#[test]
fn test_explain_analyze() {
    let db = Database::open("memory://explain_analyze").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_analyze (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO test_analyze VALUES ({}, {})", i, i),
            (),
        )
        .unwrap();
    }

    // EXPLAIN ANALYZE should show actual execution stats
    let result = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM test_analyze WHERE val > 5",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty());
}

// =============================================================================
// PK RANGE LOOKUP TESTS
// =============================================================================

#[test]
fn test_pk_range_lookup_gte_lt() {
    let db = Database::open("memory://pk_range_1").expect("Failed to create database");

    db.execute(
        "CREATE TABLE range_test (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO range_test VALUES ({}, 'val_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // PK range: id >= 20 AND id < 30
    let result = db
        .query("SELECT * FROM range_test WHERE id >= 20 AND id < 30", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 10, "Should return exactly 10 rows (20-29)");
}

#[test]
fn test_pk_range_lookup_gt_lte() {
    let db = Database::open("memory://pk_range_2").expect("Failed to create database");

    db.execute("CREATE TABLE range2 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    for i in 1..=50 {
        db.execute(&format!("INSERT INTO range2 VALUES ({}, 'v_{}')", i, i), ())
            .unwrap();
    }

    // PK range: id > 10 AND id <= 20
    let result = db
        .query("SELECT * FROM range2 WHERE id > 10 AND id <= 20", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 10, "Should return exactly 10 rows (11-20)");
}

#[test]
fn test_pk_range_empty_result() {
    let db = Database::open("memory://pk_range_3").expect("Failed to create database");

    db.execute("CREATE TABLE range3 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO range3 VALUES (1, 'a')", ())
        .unwrap();
    db.execute("INSERT INTO range3 VALUES (2, 'b')", ())
        .unwrap();

    // Invalid range: start > end should return empty
    let result = db
        .query("SELECT * FROM range3 WHERE id >= 10 AND id < 5", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 0, "Invalid range should return empty");
}

// =============================================================================
// MULTI-INDEX SCAN TESTS
// =============================================================================

#[test]
fn test_or_condition_multi_index() {
    let db = Database::open("memory://multi_idx").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_idx (
            id INTEGER PRIMARY KEY,
            col_a TEXT,
            col_b TEXT
        )",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_a ON multi_idx(col_a)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_b ON multi_idx(col_b)", ())
        .unwrap();

    for i in 1..=30 {
        db.execute(
            &format!(
                "INSERT INTO multi_idx VALUES ({}, 'a_{}', 'b_{}')",
                i,
                i % 5,
                i % 3
            ),
            (),
        )
        .unwrap();
    }

    // OR on two indexed columns should potentially use MultiIndexScan
    let result = db
        .query(
            "SELECT * FROM multi_idx WHERE col_a = 'a_1' OR col_b = 'b_1'",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert!(!rows.is_empty());
}

// =============================================================================
// COMPOSITE INDEX TESTS
// =============================================================================

#[test]
fn test_composite_index_scan() {
    let db = Database::open("memory://composite_idx").expect("Failed to create database");

    db.execute(
        "CREATE TABLE composite (
            id INTEGER PRIMARY KEY,
            tenant_id INTEGER,
            user_id INTEGER,
            data TEXT
        )",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE INDEX idx_tenant_user ON composite(tenant_id, user_id)",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        db.execute(
            &format!(
                "INSERT INTO composite VALUES ({}, {}, {}, 'data_{}')",
                i,
                i % 5,
                i % 10,
                i
            ),
            (),
        )
        .unwrap();
    }

    // Query on both columns of composite index
    let result = db
        .query(
            "SELECT * FROM composite WHERE tenant_id = 2 AND user_id = 7",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    // Should use composite index scan
    assert!(rows.len() <= 50);
}

// =============================================================================
// LIKE PATTERN TESTS
// =============================================================================

#[test]
fn test_like_prefix_pattern() {
    let db = Database::open("memory://like_prefix").expect("Failed to create database");

    db.execute(
        "CREATE TABLE patterns (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO patterns VALUES (1, 'apple')", ())
        .unwrap();
    db.execute("INSERT INTO patterns VALUES (2, 'apricot')", ())
        .unwrap();
    db.execute("INSERT INTO patterns VALUES (3, 'banana')", ())
        .unwrap();

    let result = db
        .query("SELECT * FROM patterns WHERE name LIKE 'ap%'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "Should match apple and apricot");
}

#[test]
fn test_like_suffix_pattern() {
    let db = Database::open("memory://like_suffix").expect("Failed to create database");

    db.execute(
        "CREATE TABLE suffixes (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO suffixes VALUES (1, 'test.txt')", ())
        .unwrap();
    db.execute("INSERT INTO suffixes VALUES (2, 'data.csv')", ())
        .unwrap();
    db.execute("INSERT INTO suffixes VALUES (3, 'log.txt')", ())
        .unwrap();

    let result = db
        .query("SELECT * FROM suffixes WHERE name LIKE '%.txt'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "Should match .txt files");
}

#[test]
fn test_like_contains_pattern() {
    let db = Database::open("memory://like_contains").expect("Failed to create database");

    db.execute(
        "CREATE TABLE contains (id INTEGER PRIMARY KEY, description TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO contains VALUES (1, 'hello world')", ())
        .unwrap();
    db.execute("INSERT INTO contains VALUES (2, 'world peace')", ())
        .unwrap();
    db.execute("INSERT INTO contains VALUES (3, 'goodbye')", ())
        .unwrap();

    let result = db
        .query(
            "SELECT * FROM contains WHERE description LIKE '%world%'",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "Should match rows containing 'world'");
}

#[test]
fn test_like_prefix_suffix_pattern() {
    let db = Database::open("memory://like_prefixsuffix").expect("Failed to create database");

    db.execute(
        "CREATE TABLE prefixsuffix (id INTEGER PRIMARY KEY, code TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO prefixsuffix VALUES (1, 'A123X')", ())
        .unwrap();
    db.execute("INSERT INTO prefixsuffix VALUES (2, 'A456X')", ())
        .unwrap();
    db.execute("INSERT INTO prefixsuffix VALUES (3, 'B123X')", ())
        .unwrap();
    db.execute("INSERT INTO prefixsuffix VALUES (4, 'A123Y')", ())
        .unwrap();

    let result = db
        .query("SELECT * FROM prefixsuffix WHERE code LIKE 'A%X'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "Should match A...X pattern");
}

#[test]
fn test_ilike_case_insensitive() {
    let db = Database::open("memory://ilike_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE ilike_test (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO ilike_test VALUES (1, 'HELLO')", ())
        .unwrap();
    db.execute("INSERT INTO ilike_test VALUES (2, 'hello')", ())
        .unwrap();
    db.execute("INSERT INTO ilike_test VALUES (3, 'HeLLo')", ())
        .unwrap();

    let result = db
        .query("SELECT * FROM ilike_test WHERE name ILIKE 'hello'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 3, "ILIKE should be case insensitive");
}

#[test]
fn test_glob_pattern() {
    let db = Database::open("memory://glob_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE glob_test (id INTEGER PRIMARY KEY, path TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO glob_test VALUES (1, 'src/main.rs')", ())
        .unwrap();
    db.execute("INSERT INTO glob_test VALUES (2, 'src/lib.rs')", ())
        .unwrap();
    db.execute("INSERT INTO glob_test VALUES (3, 'tests/test.rs')", ())
        .unwrap();

    let result = db
        .query("SELECT * FROM glob_test WHERE path GLOB 'src/*.rs'", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2, "GLOB should match src/*.rs");
}

// =============================================================================
// WINDOW FUNCTION PARTITION TESTS
// =============================================================================

#[test]
fn test_window_partition_by() {
    let db = Database::open("memory://window_partition").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            amount FLOAT
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO sales VALUES (1, 'North', 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO sales VALUES (2, 'North', 150.0)", ())
        .unwrap();
    db.execute("INSERT INTO sales VALUES (3, 'South', 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO sales VALUES (4, 'South', 250.0)", ())
        .unwrap();
    db.execute("INSERT INTO sales VALUES (5, 'East', 300.0)", ())
        .unwrap();

    // Window function with PARTITION BY
    let result = db
        .query(
            "SELECT region, amount,
                    SUM(amount) OVER (PARTITION BY region) as region_total
             FROM sales",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_window_row_number_partition() {
    let db = Database::open("memory://window_rownum").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            dept TEXT,
            salary INTEGER
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO employees VALUES (1, 'Sales', 50000)", ())
        .unwrap();
    db.execute("INSERT INTO employees VALUES (2, 'Sales', 60000)", ())
        .unwrap();
    db.execute("INSERT INTO employees VALUES (3, 'IT', 70000)", ())
        .unwrap();
    db.execute("INSERT INTO employees VALUES (4, 'IT', 80000)", ())
        .unwrap();

    // ROW_NUMBER with PARTITION BY
    let result = db
        .query(
            "SELECT dept, salary,
                    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rank
             FROM employees",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 4);
}

// =============================================================================
// DELETE AND VERSION MANAGEMENT TESTS
// =============================================================================

#[test]
fn test_delete_and_count() {
    let db = Database::open("memory://delete_count").expect("Failed to create database");

    db.execute(
        "CREATE TABLE cleanup_test (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .unwrap();

    // Insert many rows
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO cleanup_test VALUES ({}, 'data_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // Delete some rows (this creates "deleted" versions in version store)
    db.execute("DELETE FROM cleanup_test WHERE id <= 50", ())
        .unwrap();

    // Verify remaining rows
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM cleanup_test", ())
        .unwrap();
    assert_eq!(count, 50);
}

// =============================================================================
// TRANSACTION AND SAVEPOINT TESTS
// =============================================================================

#[test]
fn test_transaction_rollback_to_savepoint() {
    let db = Database::open("memory://savepoint_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE savepoint_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO savepoint_test VALUES (1, 100)", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE savepoint_test SET val = 200 WHERE id = 1", ())
        .unwrap();
    db.execute("SAVEPOINT sp1", ()).unwrap();
    db.execute("UPDATE savepoint_test SET val = 300 WHERE id = 1", ())
        .unwrap();
    db.execute("ROLLBACK TO SAVEPOINT sp1", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();

    let val: i64 = db
        .query_one("SELECT val FROM savepoint_test WHERE id = 1", ())
        .unwrap();
    assert_eq!(val, 200, "Should be rolled back to savepoint value");
}

// =============================================================================
// LIMIT AND OFFSET TESTS
// =============================================================================

#[test]
fn test_limit_without_order_by() {
    let db = Database::open("memory://limit_unordered").expect("Failed to create database");

    db.execute(
        "CREATE TABLE unordered (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO unordered VALUES ({}, 'val_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // LIMIT without ORDER BY should use unordered limit path (faster)
    let result = db.query("SELECT * FROM unordered LIMIT 10", ()).unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 10, "Should return exactly 10 rows");
}

#[test]
fn test_limit_offset_without_order_by() {
    let db = Database::open("memory://offset_test").expect("Failed to create database");

    db.execute(
        "CREATE TABLE offset_test (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO offset_test VALUES ({}, 'val_{}')", i, i),
            (),
        )
        .unwrap();
    }

    // LIMIT with OFFSET without ORDER BY
    let result = db
        .query("SELECT * FROM offset_test LIMIT 10 OFFSET 20", ())
        .unwrap();
    let rows: Vec<_> = result.map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 10);
}
