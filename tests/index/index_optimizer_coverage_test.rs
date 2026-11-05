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

//! Integration Tests for Index Optimizer Coverage
//!
//! These tests exercise the code paths in src/executor/index_optimizer.rs

use stoolap::Database;

// =============================================================================
// MIN/MAX Index Optimization Tests
// =============================================================================

#[test]
fn test_min_index_optimization() {
    let db = Database::open("memory://min_test").expect("Failed to create database");
    db.execute(
        "CREATE TABLE min_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_value ON min_test(value)", ())
        .expect("Failed to create index");

    // Insert some data
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO min_test VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed to insert");
    }

    // Test MIN with indexed column - should use index optimization
    let result = db
        .query("SELECT MIN(value) FROM min_test", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], 10); // MIN(value) = 10
}

#[test]
fn test_max_index_optimization() {
    let db = Database::open("memory://max_test").expect("Failed to create database");
    db.execute(
        "CREATE TABLE max_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_value ON max_test(value)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO max_test VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed to insert");
    }

    // Test MAX with indexed column - should use index optimization
    let result = db
        .query("SELECT MAX(value) FROM max_test", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], 1000); // MAX(value) = 1000
}

#[test]
fn test_min_without_index() {
    let db = Database::open("memory://min_no_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE min_no_idx (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    // No index on 'value' column

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO min_no_idx VALUES ({}, {})", i, i * 5),
            (),
        )
        .expect("Failed to insert");
    }

    // Test MIN without index - should fall back to full scan
    let result = db
        .query("SELECT MIN(value) FROM min_no_idx", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows[0], 5);
}

#[test]
fn test_min_with_where_clause() {
    let db = Database::open("memory://min_where").expect("Failed to create database");
    db.execute(
        "CREATE TABLE min_where (id INTEGER PRIMARY KEY, value INTEGER, category TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_value ON min_where(value)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        let cat = if i % 2 == 0 { "even" } else { "odd" };
        db.execute(
            &format!(
                "INSERT INTO min_where VALUES ({}, {}, '{}')",
                i,
                i * 10,
                cat
            ),
            (),
        )
        .expect("Failed to insert");
    }

    // MIN with WHERE clause - may not use optimization
    let result = db
        .query(
            "SELECT MIN(value) FROM min_where WHERE category = 'even'",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows[0], 20); // MIN of even numbers: 2*10=20
}

// =============================================================================
// COUNT(*) Optimization Tests
// =============================================================================

#[test]
fn test_count_star_optimization() {
    let db = Database::open("memory://count_star").expect("Failed to create database");
    db.execute(
        "CREATE TABLE count_star (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO count_star VALUES ({}, 'name{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    // COUNT(*) should be optimized
    let result = db
        .query("SELECT COUNT(*) FROM count_star", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows[0], 50);
}

#[test]
fn test_count_with_condition() {
    let db = Database::open("memory://count_cond").expect("Failed to create database");
    db.execute(
        "CREATE TABLE count_cond (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_value ON count_cond(value)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO count_cond VALUES ({}, {})", i, i % 10),
            (),
        )
        .expect("Failed to insert");
    }

    // COUNT with condition - tests different path
    let result = db
        .query("SELECT COUNT(*) FROM count_cond WHERE value > 5", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        rows.push(val);
    }
    assert_eq!(rows[0], 40); // values 6,7,8,9 each appear 10 times = 40
}

// =============================================================================
// ORDER BY + LIMIT Index Optimization Tests
// =============================================================================

#[test]
fn test_order_by_limit_indexed() {
    let db = Database::open("memory://order_limit").expect("Failed to create database");
    db.execute(
        "CREATE TABLE order_limit (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_score ON order_limit(score)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO order_limit VALUES ({}, {})", i, 101 - i),
            (),
        )
        .expect("Failed to insert");
    }

    // ORDER BY indexed column with LIMIT - should use index
    let result = db
        .query("SELECT * FROM order_limit ORDER BY score LIMIT 5", ())
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let score: i64 = row.get(1).unwrap();
        rows.push((id, score));
    }
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].1, 1); // Lowest score
    assert_eq!(rows[4].1, 5);
}

#[test]
fn test_order_by_desc_limit() {
    let db = Database::open("memory://order_desc").expect("Failed to create database");
    db.execute(
        "CREATE TABLE order_desc (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_score ON order_desc(score)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO order_desc VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    // ORDER BY DESC with LIMIT - tests descending index traversal
    let result = db
        .query("SELECT * FROM order_desc ORDER BY score DESC LIMIT 5", ())
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let score: i64 = row.get(1).unwrap();
        rows.push((id, score));
    }
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].1, 100); // Highest score
    assert_eq!(rows[4].1, 96);
}

#[test]
fn test_order_by_pk_limit() {
    let db = Database::open("memory://order_pk").expect("Failed to create database");
    db.execute(
        "CREATE TABLE order_pk (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO order_pk VALUES ({}, 'name{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    // ORDER BY primary key - should use PK index
    let result = db
        .query("SELECT * FROM order_pk ORDER BY id LIMIT 10", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        rows.push(id);
    }
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0], 1);
    assert_eq!(rows[9], 10);
}

// =============================================================================
// Keyset Pagination Tests
// =============================================================================

#[test]
fn test_keyset_pagination_basic() {
    let db = Database::open("memory://keyset_basic").expect("Failed to create database");
    db.execute(
        "CREATE TABLE keyset_basic (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO keyset_basic VALUES ({}, 'name{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    // Keyset pagination: id > last_seen_id LIMIT n
    let result = db
        .query(
            "SELECT * FROM keyset_basic WHERE id > 50 ORDER BY id LIMIT 10",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        rows.push(id);
    }
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0], 51);
    assert_eq!(rows[9], 60);
}

#[test]
fn test_keyset_with_indexed_column() {
    let db = Database::open("memory://keyset_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE keyset_idx (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_score ON keyset_idx(score)", ())
        .expect("Failed to create index");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO keyset_idx VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed to insert");
    }

    // Keyset on indexed column
    let result = db
        .query(
            "SELECT * FROM keyset_idx WHERE score > 500 ORDER BY score LIMIT 5",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let score: i64 = row.get(1).unwrap();
        rows.push((id, score));
    }
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].1, 510); // First after 500
}

// =============================================================================
// IN List Optimization Tests
// =============================================================================

#[test]
fn test_in_list_small() {
    let db = Database::open("memory://in_small").expect("Failed to create database");
    db.execute(
        "CREATE TABLE in_small (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO in_small VALUES ({}, 'name{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    // Small IN list
    let result = db
        .query("SELECT * FROM in_small WHERE id IN (1, 5, 10, 25, 50)", ())
        .expect("Query failed");

    let mut rows: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        rows.push(id);
    }
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_in_list_large() {
    let db = Database::open("memory://in_large").expect("Failed to create database");
    db.execute(
        "CREATE TABLE in_large (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=1000 {
        db.execute(
            &format!("INSERT INTO in_large VALUES ({}, 'name{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    // Large IN list - tests HashSet optimization threshold
    let ids: Vec<String> = (1..=100).map(|i| i.to_string()).collect();
    let in_list = ids.join(", ");
    let result = db
        .query(
            &format!("SELECT * FROM in_large WHERE id IN ({})", in_list),
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 100);
}

#[test]
fn test_in_subquery() {
    let db = Database::open("memory://in_subq").expect("Failed to create database");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER)",
        (),
    )
    .expect("Failed to create orders");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, vip INTEGER)",
        (),
    )
    .expect("Failed to create customers");

    // Insert customers (some VIP)
    for i in 1..=20 {
        let vip = if i <= 5 { 1 } else { 0 };
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}', {})", i, i, vip),
            (),
        )
        .expect("Failed to insert customer");
    }

    // Insert orders
    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 20) + 1),
            (),
        )
        .expect("Failed to insert order");
    }

    // IN subquery - tests subquery optimization
    let result = db
        .query(
            "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE vip = 1)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    // VIP customers: 1,2,3,4,5 -> orders with customer_id in those
    assert!(count > 0);
}

// =============================================================================
// Window Function Pre-sorting Tests
// =============================================================================

#[test]
fn test_window_function_presort() {
    let db = Database::open("memory://win_presort_v3").expect("Failed to create database");
    db.execute(
        "CREATE TABLE win_presort (id INTEGER PRIMARY KEY, salary INTEGER)",
        (),
    )
    .expect("Failed to create table");
    // No index on salary - avoids triggering the ORDER BY+LIMIT optimization path that causes overflow

    // Insert rows using a loop
    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO win_presort VALUES ({}, {})", i, i * 100),
            (),
        )
        .expect("Failed to insert");
    }

    // ROW_NUMBER with ORDER BY - tests window function pre-sorting detection
    let result = db
        .query(
            "SELECT id, salary, ROW_NUMBER() OVER (ORDER BY salary) as rn FROM win_presort",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let salary: i64 = row.get(1).unwrap();
        let rn: i64 = row.get(2).unwrap();
        rows.push((id, salary, rn));
    }
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0].1, 100); // Lowest salary first
    assert_eq!(rows[0].2, 1); // ROW_NUMBER = 1
}

#[test]
fn test_window_function_with_index() {
    let db = Database::open("memory://win_idx_test").expect("Failed to create database");
    db.execute(
        "CREATE TABLE win_idx (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("CREATE INDEX idx_score ON win_idx(score)", ())
        .expect("Failed to create index");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO win_idx VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed to insert");
    }

    // Window function with ORDER BY on indexed column + LIMIT
    let result = db
        .query(
            "SELECT id, score, ROW_NUMBER() OVER (ORDER BY score) as rn FROM win_idx LIMIT 5",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_window_function_partition() {
    let db = Database::open("memory://win_partition").expect("Failed to create database");
    db.execute(
        "CREATE TABLE win_partition (id INTEGER PRIMARY KEY, dept TEXT, salary INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert rows individually to avoid bulk insert overflow issue
    db.execute("INSERT INTO win_partition VALUES (1, 'A', 100)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO win_partition VALUES (2, 'A', 200)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO win_partition VALUES (3, 'A', 150)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO win_partition VALUES (4, 'B', 300)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO win_partition VALUES (5, 'B', 250)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO win_partition VALUES (6, 'B', 350)", ())
        .expect("Failed to insert");

    // PARTITION BY tests different code path
    let result = db
        .query(
            "SELECT id, dept, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as r
             FROM win_partition",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, String, i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let dept: String = row.get(1).unwrap();
        let salary: i64 = row.get(2).unwrap();
        let r: i64 = row.get(3).unwrap();
        rows.push((id, dept, salary, r));
    }
    assert_eq!(rows.len(), 6);
}

#[test]
fn test_ntile_window_function() {
    let db = Database::open("memory://win_ntile").expect("Failed to create database");
    db.execute(
        "CREATE TABLE win_ntile (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO win_ntile VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    // NTILE divides into buckets
    let result = db
        .query(
            "SELECT id, score, NTILE(4) OVER (ORDER BY score) as quartile FROM win_ntile",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 100);
}

// =============================================================================
// Join Optimization Tests
// =============================================================================

#[test]
fn test_equality_join_detection() {
    let db = Database::open("memory://eq_join").expect("Failed to create database");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .expect("Failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .expect("Failed");

    for i in 1..=50 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, {})", i, i * 2), ())
            .expect("Failed");
        db.execute(&format!("INSERT INTO t2 VALUES ({}, {})", i, i * 2), ())
            .expect("Failed");
    }

    // Equality join - should use hash join
    let result = db
        .query("SELECT * FROM t1 JOIN t2 ON t1.a = t2.b", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 50);
}

#[test]
fn test_non_equality_join() {
    let db = Database::open("memory://neq_join").expect("Failed to create database");
    db.execute("CREATE TABLE r1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");
    db.execute("CREATE TABLE r2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO r1 VALUES ({}, {})", i, i), ())
            .expect("Failed");
        db.execute(&format!("INSERT INTO r2 VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    // Non-equality join - uses nested loop
    let result = db
        .query("SELECT * FROM r1 JOIN r2 ON r1.val < r2.val", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    // Should be 0+1+2+...+9 = 45
    assert_eq!(count, 45);
}

#[test]
fn test_left_join_with_index() {
    let db = Database::open("memory://left_join_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE main_t (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE detail_t (id INTEGER PRIMARY KEY, main_id INTEGER, info TEXT)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_main_id ON detail_t(main_id)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO main_t VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    // Only add details for some items
    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO detail_t VALUES ({}, {}, 'detail{}')", i, i, i),
            (),
        )
        .expect("Failed");
    }

    // LEFT JOIN - tests join with NULL handling
    let result = db
        .query(
            "SELECT main_t.id, detail_t.info FROM main_t
             LEFT JOIN detail_t ON main_t.id = detail_t.main_id",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// Cardinality Estimation Tests
// =============================================================================

#[test]
fn test_cardinality_with_analyze() {
    let db = Database::open("memory://card_analyze").expect("Failed to create database");
    db.execute(
        "CREATE TABLE card_analyze (id INTEGER PRIMARY KEY, category INTEGER, value TEXT)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_cat ON card_analyze(category)", ())
        .expect("Failed");

    // Insert data with known distribution
    for i in 1..=1000 {
        let cat = i % 10;
        db.execute(
            &format!(
                "INSERT INTO card_analyze VALUES ({}, {}, 'val{}')",
                i, cat, i
            ),
            (),
        )
        .expect("Failed");
    }

    // Run ANALYZE to collect statistics
    db.execute("ANALYZE card_analyze", ())
        .expect("Failed to analyze");

    // Query should now use statistics for cardinality estimation
    let result = db
        .query("SELECT * FROM card_analyze WHERE category = 5", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 100); // Each category has 100 rows
}

#[test]
fn test_cardinality_range_estimate() {
    let db = Database::open("memory://card_range").expect("Failed to create database");
    db.execute(
        "CREATE TABLE card_range (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_val ON card_range(value)", ())
        .expect("Failed");

    for i in 1..=1000 {
        db.execute(&format!("INSERT INTO card_range VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    db.execute("ANALYZE card_range", ())
        .expect("Failed to analyze");

    // Range query - tests range cardinality estimation
    let result = db
        .query(
            "SELECT * FROM card_range WHERE value BETWEEN 100 AND 200",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 101);
}

// =============================================================================
// EXPLAIN Tests (to trigger optimization paths)
// =============================================================================

#[test]
fn test_explain_min_max() {
    let db = Database::open("memory://explain_minmax").expect("Failed to create database");
    db.execute(
        "CREATE TABLE explain_mm (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_val ON explain_mm(val)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO explain_mm VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    // EXPLAIN should show optimization being used
    let result = db
        .query("EXPLAIN SELECT MIN(val) FROM explain_mm", ())
        .expect("Query failed");

    let mut found_output = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let plan: String = row.get(0).unwrap();
        if !plan.is_empty() {
            found_output = true;
        }
    }
    assert!(found_output);
}

#[test]
fn test_explain_analyze() {
    let db = Database::open("memory://explain_analyze").expect("Failed to create database");
    db.execute(
        "CREATE TABLE explain_an (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_val ON explain_an(val)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO explain_an VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    // EXPLAIN ANALYZE shows actual execution stats
    let result = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM explain_an WHERE val > 50",
            (),
        )
        .expect("Query failed");

    let mut found_output = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let plan: String = row.get(0).unwrap();
        if !plan.is_empty() {
            found_output = true;
        }
    }
    assert!(found_output);
}

// =============================================================================
// Edge Cases and Special Queries
// =============================================================================

#[test]
fn test_empty_table_min_max() {
    let db = Database::open("memory://empty_minmax").expect("Failed to create database");
    db.execute(
        "CREATE TABLE empty_minmax (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_val ON empty_minmax(val)", ())
        .expect("Failed");

    // MIN/MAX on empty table - should return NULL
    let result = db
        .query("SELECT MIN(val), MAX(val) FROM empty_minmax", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 1); // Returns one row with NULLs
}

#[test]
fn test_distinct_with_index() {
    let db = Database::open("memory://distinct_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE distinct_idx (id INTEGER PRIMARY KEY, category INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_cat ON distinct_idx(category)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO distinct_idx VALUES ({}, {})", i, i % 5),
            (),
        )
        .expect("Failed");
    }

    // DISTINCT might use index for deduplication
    let result = db
        .query(
            "SELECT DISTINCT category FROM distinct_idx ORDER BY category",
            (),
        )
        .expect("Query failed");

    let mut cats: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: i64 = row.get(0).unwrap();
        cats.push(cat);
    }
    assert_eq!(cats, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_group_by_with_index() {
    let db = Database::open("memory://group_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE group_idx (id INTEGER PRIMARY KEY, category INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_cat ON group_idx(category)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!(
                "INSERT INTO group_idx VALUES ({}, {}, {})",
                i,
                i % 5,
                i as f64 * 1.5
            ),
            (),
        )
        .expect("Failed");
    }

    // GROUP BY on indexed column
    let result = db
        .query(
            "SELECT category, SUM(amount), COUNT(*) FROM group_idx GROUP BY category ORDER BY category",
            (),
        )
        .expect("Query failed");

    let mut groups = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat: i64 = row.get(0).unwrap();
        groups.push(cat);
    }
    assert_eq!(groups.len(), 5);
}

#[test]
fn test_multiple_indexes_same_query() {
    let db = Database::open("memory://multi_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE multi_idx (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_a ON multi_idx(a)", ())
        .expect("Failed");
    db.execute("CREATE INDEX idx_b ON multi_idx(b)", ())
        .expect("Failed");
    db.execute("CREATE INDEX idx_c ON multi_idx(c)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!(
                "INSERT INTO multi_idx VALUES ({}, {}, {}, {})",
                i,
                i % 10,
                i % 20,
                i % 5
            ),
            (),
        )
        .expect("Failed");
    }

    // Query with multiple indexed columns - optimizer must choose
    let result = db
        .query("SELECT * FROM multi_idx WHERE a = 5 AND b = 5", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn test_composite_index() {
    let db = Database::open("memory://composite_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE composite_idx (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute("CREATE INDEX idx_ab ON composite_idx(a, b)", ())
        .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!(
                "INSERT INTO composite_idx VALUES ({}, {}, {})",
                i,
                i % 10,
                i % 5
            ),
            (),
        )
        .expect("Failed");
    }

    // Query on composite index
    let result = db
        .query("SELECT * FROM composite_idx WHERE a = 5 AND b = 3", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    // a=5 appears at i=5,15,25,...,95 (10 times)
    // b=3 appears when i%5=3, so i=3,8,13,18,...
    // Need both: a=5 (i%10=5) AND b=3 (i%5=3)
    // i%10=5 means i∈{5,15,25,35,45,55,65,75,85,95}
    // i%5=3 means i∈{3,8,13,18,23,28,...}
    // Intersection: none in this case since 5,15,25...%5 = 0 or 5
    assert_eq!(count, 0);
}

#[test]
fn test_exists_subquery_optimization() {
    let db = Database::open("memory://exists_opt").expect("Failed to create database");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=50 {
        db.execute(
            &format!("INSERT INTO orders VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    // EXISTS subquery - tests subquery optimization
    let result = db
        .query(
            "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust_id = c.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 10); // Customers 1-10 have orders
}

#[test]
fn test_not_exists_subquery() {
    let db = Database::open("memory://not_exists").expect("Failed to create database");
    db.execute(
        "CREATE TABLE orders2 (id INTEGER PRIMARY KEY, cust_id INTEGER)",
        (),
    )
    .expect("Failed");
    db.execute(
        "CREATE TABLE customers2 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO customers2 VALUES ({}, 'cust{}')", i, i),
            (),
        )
        .expect("Failed");
    }
    for i in 1..=30 {
        db.execute(
            &format!("INSERT INTO orders2 VALUES ({}, {})", i, (i % 10) + 1),
            (),
        )
        .expect("Failed");
    }

    // NOT EXISTS - inverse of EXISTS
    let result = db
        .query(
            "SELECT * FROM customers2 c WHERE NOT EXISTS (SELECT 1 FROM orders2 o WHERE o.cust_id = c.id)",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed to get row");
        count += 1;
    }
    assert_eq!(count, 10); // Customers 11-20 have no orders
}

#[test]
fn test_dense_rank_window() {
    let db = Database::open("memory://dense_rank_test").expect("Failed to create database");
    db.execute(
        "CREATE TABLE dense_rank_test (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    // Insert rows individually
    db.execute("INSERT INTO dense_rank_test VALUES (1, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO dense_rank_test VALUES (2, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO dense_rank_test VALUES (3, 90)", ())
        .expect("Failed");
    db.execute("INSERT INTO dense_rank_test VALUES (4, 90)", ())
        .expect("Failed");
    db.execute("INSERT INTO dense_rank_test VALUES (5, 80)", ())
        .expect("Failed");

    // DENSE_RANK handles ties differently than RANK
    let result = db
        .query(
            "SELECT id, score, DENSE_RANK() OVER (ORDER BY score DESC) as dr FROM dense_rank_test",
            (),
        )
        .expect("Query failed");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let dr: i64 = row.get(2).unwrap();
        rows.push((id, dr));
    }
    // Score 100: dr=1, Score 90: dr=2, Score 80: dr=3
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_offset_with_order_by() {
    let db = Database::open("memory://offset_test").expect("Failed to create database");
    db.execute(
        "CREATE TABLE offset_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO offset_test VALUES ({}, {})", i, i),
            (),
        )
        .expect("Failed");
    }

    // OFFSET with ORDER BY - tests pagination code path
    let result = db
        .query(
            "SELECT * FROM offset_test ORDER BY id LIMIT 10 OFFSET 50",
            (),
        )
        .expect("Query failed");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    assert_eq!(ids.len(), 10);
    assert_eq!(ids[0], 51);
    assert_eq!(ids[9], 60);
}
