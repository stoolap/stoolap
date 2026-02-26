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

//! Complex integration tests for TiKV storage backend.
//!
//! These tests exercise joins, subqueries, concurrent transactions,
//! larger datasets, and schema operations against a real TiKV cluster.
//!
//! Prerequisites:
//!   make tikv-up
//!
//! Run:
//!   make test-tikv-complex
//!   # or directly:
//!   TIKV_PD_ENDPOINTS=127.0.0.1:2379 cargo test --features tikv --test tikv_complex_test -- --test-threads=1

#![cfg(feature = "tikv")]

use stoolap::Database;

fn tikv_dsn() -> Option<String> {
    std::env::var("TIKV_PD_ENDPOINTS")
        .ok()
        .map(|endpoints| format!("tikv://{}", endpoints))
}

macro_rules! require_tikv {
    () => {
        match tikv_dsn() {
            Some(dsn) => dsn,
            None => {
                eprintln!("Skipping: TIKV_PD_ENDPOINTS not set");
                return;
            }
        }
    };
}

/// Helper: drop tables ignoring errors
fn cleanup(db: &Database, tables: &[&str]) {
    for t in tables {
        let _ = db.execute(&format!("DROP TABLE IF EXISTS {}", t), ());
    }
}

// ─── Join Tests ────────────────────────────────────────────────────

#[test]
fn test_tikv_inner_join() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_join_orders", "tikv_join_customers"]);

    db.execute(
        "CREATE TABLE tikv_join_customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_join_orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_join_customers VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_join_customers VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_join_customers VALUES (3, 'Charlie')", ())
        .unwrap();

    db.execute("INSERT INTO tikv_join_orders VALUES (1, 1, 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_join_orders VALUES (2, 1, 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_join_orders VALUES (3, 2, 150.0)", ())
        .unwrap();

    // Inner join
    let mut count = 0;
    for row in db
        .query(
            "SELECT c.name, o.amount FROM tikv_join_customers c
             INNER JOIN tikv_join_orders o ON c.id = o.customer_id",
            (),
        )
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 3, "Inner join should return 3 rows");

    // Join with aggregation
    let total: f64 = db
        .query_one(
            "SELECT SUM(o.amount) FROM tikv_join_customers c
             INNER JOIN tikv_join_orders o ON c.id = o.customer_id
             WHERE c.name = 'Alice'",
            (),
        )
        .unwrap();
    assert!((total - 300.0).abs() < 0.01, "Alice's total should be 300");

    cleanup(&db, &["tikv_join_orders", "tikv_join_customers"]);
}

#[test]
fn test_tikv_left_join() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_lj_items", "tikv_lj_categories"]);

    db.execute(
        "CREATE TABLE tikv_lj_categories (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_lj_items (id INTEGER PRIMARY KEY, cat_id INTEGER, title TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_lj_categories VALUES (1, 'Books')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_lj_categories VALUES (2, 'Movies')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_lj_categories VALUES (3, 'Music')", ())
        .unwrap();

    db.execute("INSERT INTO tikv_lj_items VALUES (1, 1, 'Rust Book')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_lj_items VALUES (2, 1, 'Go Book')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_lj_items VALUES (3, 2, 'Matrix')", ())
        .unwrap();

    // Left join: Music category has no items
    let mut count = 0;
    for row in db
        .query(
            "SELECT c.name, i.title FROM tikv_lj_categories c
             LEFT JOIN tikv_lj_items i ON c.id = i.cat_id",
            (),
        )
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(
        count, 4,
        "Left join should return 4 rows (3 matches + 1 NULL)"
    );

    cleanup(&db, &["tikv_lj_items", "tikv_lj_categories"]);
}

// ─── Subquery Tests ────────────────────────────────────────────────

#[test]
fn test_tikv_subquery_in() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_in_orders", "tikv_in_customers"]);

    db.execute(
        "CREATE TABLE tikv_in_customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_in_orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_in_customers VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_in_customers VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_in_customers VALUES (3, 'Charlie')", ())
        .unwrap();

    db.execute("INSERT INTO tikv_in_orders VALUES (1, 1, 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_in_orders VALUES (2, 2, 200.0)", ())
        .unwrap();

    // IN subquery: find customers who have orders
    let mut names = Vec::new();
    for row in db
        .query(
            "SELECT name FROM tikv_in_customers WHERE id IN (SELECT customer_id FROM tikv_in_orders)",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        names.push(name);
    }
    names.sort();
    assert_eq!(
        names,
        vec!["Alice", "Bob"],
        "Should find customers with orders"
    );

    // NOT IN subquery: find customers without orders
    let mut no_order_names = Vec::new();
    for row in db
        .query(
            "SELECT name FROM tikv_in_customers WHERE id NOT IN (SELECT customer_id FROM tikv_in_orders)",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        no_order_names.push(name);
    }
    assert_eq!(
        no_order_names,
        vec!["Charlie"],
        "Only Charlie has no orders"
    );

    // EXISTS subquery
    let mut exist_names = Vec::new();
    for row in db
        .query(
            "SELECT c.name FROM tikv_in_customers c WHERE EXISTS (SELECT 1 FROM tikv_in_orders o WHERE o.customer_id = c.id)",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        exist_names.push(name);
    }
    exist_names.sort();
    assert_eq!(
        exist_names,
        vec!["Alice", "Bob"],
        "EXISTS should find Alice and Bob"
    );

    cleanup(&db, &["tikv_in_orders", "tikv_in_customers"]);
}

#[test]
fn test_tikv_scalar_subquery() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_scalar_t"]);

    db.execute(
        "CREATE TABLE tikv_scalar_t (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_scalar_t VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_scalar_t VALUES (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_scalar_t VALUES (3, 30)", ())
        .unwrap();

    // Rows above average
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM tikv_scalar_t WHERE val > (SELECT AVG(val) FROM tikv_scalar_t)",
            (),
        )
        .unwrap();
    assert_eq!(count, 1, "Only val=30 is above average 20");

    cleanup(&db, &["tikv_scalar_t"]);
}

// ─── GROUP BY & HAVING ────────────────────────────────────────────

#[test]
fn test_tikv_group_by_having() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_gb_sales"]);

    db.execute(
        "CREATE TABLE tikv_gb_sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_gb_sales VALUES (1, 'East', 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_gb_sales VALUES (2, 'East', 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_gb_sales VALUES (3, 'West', 50.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_gb_sales VALUES (4, 'West', 25.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_gb_sales VALUES (5, 'North', 300.0)", ())
        .unwrap();

    // GROUP BY with HAVING
    // East: 300, West: 75 (below threshold), North: 300
    let mut count = 0;
    for row in db
        .query(
            "SELECT region, SUM(amount) as total FROM tikv_gb_sales
             GROUP BY region HAVING SUM(amount) > 100",
            (),
        )
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 2, "East (300) and North (300) have totals > 100");

    cleanup(&db, &["tikv_gb_sales"]);
}

// ─── Larger Dataset ────────────────────────────────────────────────

#[test]
fn test_tikv_batch_insert_500_rows() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_batch_t"]);

    db.execute(
        "CREATE TABLE tikv_batch_t (id INTEGER PRIMARY KEY, val INTEGER, label TEXT)",
        (),
    )
    .unwrap();

    // Insert 500 rows
    for i in 1..=500 {
        db.execute(
            &format!(
                "INSERT INTO tikv_batch_t VALUES ({}, {}, 'item_{}')",
                i,
                i * 10,
                i
            ),
            (),
        )
        .unwrap();
    }

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_batch_t", ())
        .unwrap();
    assert_eq!(count, 500);

    // Range query
    let range_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM tikv_batch_t WHERE val >= 1000 AND val <= 2000",
            (),
        )
        .unwrap();
    assert_eq!(range_count, 101, "IDs 100..200 have vals 1000..2000");

    // Aggregation on larger data
    let sum: i64 = db
        .query_one("SELECT SUM(val) FROM tikv_batch_t WHERE id <= 10", ())
        .unwrap();
    assert_eq!(sum, 550, "Sum of 10+20+...+100 = 550");

    cleanup(&db, &["tikv_batch_t"]);
}

// ─── Multi-table Schema Operations ────────────────────────────────

#[test]
fn test_tikv_schema_operations() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_schema_a", "tikv_schema_b", "tikv_schema_c"]);

    // Create multiple tables
    db.execute(
        "CREATE TABLE tikv_schema_a (id INTEGER PRIMARY KEY, a TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_schema_b (id INTEGER PRIMARY KEY, b TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_schema_c (id INTEGER PRIMARY KEY, c TEXT)",
        (),
    )
    .unwrap();

    // Insert into each
    db.execute("INSERT INTO tikv_schema_a VALUES (1, 'hello')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_schema_b VALUES (1, 'world')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_schema_c VALUES (1, '!')", ())
        .unwrap();

    // Drop middle table
    db.execute("DROP TABLE tikv_schema_b", ()).unwrap();

    // Verify remaining tables still work
    let a: String = db
        .query_one("SELECT a FROM tikv_schema_a WHERE id = 1", ())
        .unwrap();
    assert_eq!(a, "hello");

    let c: String = db
        .query_one("SELECT c FROM tikv_schema_c WHERE id = 1", ())
        .unwrap();
    assert_eq!(c, "!");

    cleanup(&db, &["tikv_schema_a", "tikv_schema_c"]);
}

// ─── Concurrent Transactions ───────────────────────────────────────

#[test]
fn test_tikv_concurrent_transactions() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_conc_t"]);

    db.execute(
        "CREATE TABLE tikv_conc_t (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    // Transaction 1: insert and commit
    {
        let mut txn1 = db.begin().unwrap();
        txn1.execute("INSERT INTO tikv_conc_t VALUES (1, 100)", ())
            .unwrap();
        txn1.commit().unwrap();
    }

    // Transaction 2: insert and rollback
    {
        let mut txn2 = db.begin().unwrap();
        txn2.execute("INSERT INTO tikv_conc_t VALUES (2, 200)", ())
            .unwrap();
        txn2.rollback().unwrap();
    }

    // Transaction 3: insert and commit
    {
        let mut txn3 = db.begin().unwrap();
        txn3.execute("INSERT INTO tikv_conc_t VALUES (3, 300)", ())
            .unwrap();
        txn3.commit().unwrap();
    }

    // Only txn1 and txn3 should be visible
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_conc_t", ())
        .unwrap();
    assert_eq!(count, 2, "Only committed transactions visible");

    let sum: i64 = db
        .query_one("SELECT SUM(val) FROM tikv_conc_t", ())
        .unwrap();
    assert_eq!(sum, 400, "Sum of committed rows: 100 + 300");

    cleanup(&db, &["tikv_conc_t"]);
}

// ─── ORDER BY and LIMIT ───────────────────────────────────────────

#[test]
fn test_tikv_order_by_limit() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_order_t"]);

    db.execute(
        "CREATE TABLE tikv_order_t (id INTEGER PRIMARY KEY, score INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_order_t VALUES (1, 85, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_order_t VALUES (2, 92, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_order_t VALUES (3, 78, 'Charlie')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_order_t VALUES (4, 95, 'Diana')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_order_t VALUES (5, 88, 'Eve')", ())
        .unwrap();

    // Top 3 by score
    let mut names = Vec::new();
    for row in db
        .query(
            "SELECT name FROM tikv_order_t ORDER BY score DESC LIMIT 3",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        names.push(name);
    }
    assert_eq!(names.len(), 3);
    assert_eq!(names[0], "Diana");
    assert_eq!(names[1], "Bob");
    assert_eq!(names[2], "Eve");

    cleanup(&db, &["tikv_order_t"]);
}

// ─── CASE Expressions ─────────────────────────────────────────────

#[test]
fn test_tikv_case_expression() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_case_t"]);

    db.execute(
        "CREATE TABLE tikv_case_t (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_case_t VALUES (1, 95)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_case_t VALUES (2, 72)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_case_t VALUES (3, 45)", ())
        .unwrap();

    let mut grades = Vec::new();
    for row in db
        .query(
            "SELECT CASE
                WHEN score >= 90 THEN 'A'
                WHEN score >= 70 THEN 'B'
                ELSE 'C'
             END as grade
             FROM tikv_case_t ORDER BY id",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let grade: String = row.get(0).unwrap();
        grades.push(grade);
    }
    assert_eq!(grades, vec!["A", "B", "C"]);

    cleanup(&db, &["tikv_case_t"]);
}

// ─── DISTINCT and COUNT DISTINCT ──────────────────────────────────

#[test]
fn test_tikv_distinct() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_distinct_t"]);

    db.execute(
        "CREATE TABLE tikv_distinct_t (id INTEGER PRIMARY KEY, category TEXT, status TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_distinct_t VALUES (1, 'A', 'active')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_distinct_t VALUES (2, 'B', 'active')", ())
        .unwrap();
    db.execute(
        "INSERT INTO tikv_distinct_t VALUES (3, 'A', 'inactive')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO tikv_distinct_t VALUES (4, 'B', 'active')", ())
        .unwrap();
    db.execute("INSERT INTO tikv_distinct_t VALUES (5, 'A', 'active')", ())
        .unwrap();

    // DISTINCT categories
    let mut count = 0;
    for row in db
        .query("SELECT DISTINCT category FROM tikv_distinct_t", ())
        .unwrap()
    {
        let _row = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 2, "Two distinct categories: A, B");

    // COUNT DISTINCT
    let distinct_count: i64 = db
        .query_one("SELECT COUNT(DISTINCT status) FROM tikv_distinct_t", ())
        .unwrap();
    assert_eq!(distinct_count, 2, "Two distinct statuses: active, inactive");

    cleanup(&db, &["tikv_distinct_t"]);
}

// ─── Re-open Database (Schema Persistence) ────────────────────────

/// Schema persistence test: data survives reconnection.
/// This test may fail if TiKV lock resolution is slow (retry with make test-tikv-complex).
#[test]
fn test_tikv_schema_persistence() {
    let dsn = require_tikv!();

    // Retry wrapper for TiKV lock resolution issues
    let try_op = |db: &Database, sql: &str| -> bool {
        for attempt in 0..5 {
            match db.execute(sql, ()) {
                Ok(_) => return true,
                Err(_) if attempt < 4 => {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(_) => return false,
            }
        }
        false
    };

    // Create table and insert data with first connection
    {
        let db = Database::open(&dsn).unwrap();
        try_op(&db, "DROP TABLE IF EXISTS tikv_persist_t");
        std::thread::sleep(std::time::Duration::from_millis(200));

        db.execute(
            "CREATE TABLE tikv_persist_t (id INTEGER PRIMARY KEY, data TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO tikv_persist_t VALUES (1, 'persisted')", ())
            .unwrap();
        db.close().unwrap();
    }

    // Wait for TiKV to finalize
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Re-open and verify data is still there
    {
        let db = Database::open(&dsn).unwrap();
        let mut last_err = None;
        for attempt in 0..5 {
            match db.query_one::<String, _>("SELECT data FROM tikv_persist_t WHERE id = 1", ()) {
                Ok(data) => {
                    assert_eq!(data, "persisted");
                    last_err = None;
                    break;
                }
                Err(e) => {
                    last_err = Some(e);
                    if attempt < 4 {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                }
            }
        }
        if let Some(e) = last_err {
            panic!("Failed after retries: {}", e);
        }

        // Cleanup
        let _ = db.execute("DROP TABLE tikv_persist_t", ());
    }
}

// ─── Savepoint Tests ────────────────────────────────────────────────

#[test]
fn test_tikv_savepoint_ddl() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_sp_t1", "tikv_sp_t2"]);

    // Use a transaction with savepoints
    let mut tx = db.begin().unwrap();

    tx.execute(
        "CREATE TABLE tikv_sp_t1 (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();
    tx.execute("INSERT INTO tikv_sp_t1 VALUES (1, 'before savepoint')", ())
        .unwrap();

    // Create savepoint
    tx.execute("SAVEPOINT sp1", ()).unwrap();

    // Create another table after savepoint
    tx.execute(
        "CREATE TABLE tikv_sp_t2 (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();
    tx.execute("INSERT INTO tikv_sp_t2 VALUES (1, 'after savepoint')", ())
        .unwrap();

    // Rollback to savepoint — should undo CREATE TABLE tikv_sp_t2
    tx.execute("ROLLBACK TO SAVEPOINT sp1", ()).unwrap();

    // tikv_sp_t1 should still exist
    let count: i64 = tx.query_one("SELECT COUNT(*) FROM tikv_sp_t1", ()).unwrap();
    assert_eq!(count, 1, "tikv_sp_t1 should still have 1 row");

    tx.execute("COMMIT", ()).unwrap();

    // tikv_sp_t2 should NOT exist after commit
    let result = db.query("SELECT * FROM tikv_sp_t2", ());
    assert!(
        result.is_err(),
        "tikv_sp_t2 should not exist after rollback"
    );

    cleanup(&db, &["tikv_sp_t1"]);
}

#[test]
fn test_tikv_savepoint_release() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_sp_rel"]);

    let mut tx = db.begin().unwrap();

    tx.execute(
        "CREATE TABLE tikv_sp_rel (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    tx.execute("SAVEPOINT sp_a", ()).unwrap();
    tx.execute("INSERT INTO tikv_sp_rel VALUES (1, 10)", ())
        .unwrap();

    // Release savepoint (changes stay)
    tx.execute("RELEASE SAVEPOINT sp_a", ()).unwrap();

    tx.execute("COMMIT", ()).unwrap();

    // Data should be there
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_sp_rel", ())
        .unwrap();
    assert_eq!(count, 1, "Released savepoint should keep changes");

    cleanup(&db, &["tikv_sp_rel"]);
}

// ─── Index Tests ────────────────────────────────────────────────

#[test]
fn test_tikv_create_index() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_idx_t"]);

    db.execute(
        "CREATE TABLE tikv_idx_t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .unwrap();

    // Insert data before creating index
    db.execute("INSERT INTO tikv_idx_t VALUES (1, 'Alice', 30)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_idx_t VALUES (2, 'Bob', 25)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_idx_t VALUES (3, 'Charlie', 35)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_idx_t VALUES (4, 'Diana', 28)", ())
        .unwrap();

    // Create index on age column
    db.execute("CREATE INDEX idx_tikv_idx_t_age ON tikv_idx_t(age)", ())
        .unwrap();

    // Insert more data after creating index (should maintain index)
    db.execute("INSERT INTO tikv_idx_t VALUES (5, 'Eve', 22)", ())
        .unwrap();

    // Query that can use the index
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_idx_t WHERE age > 27", ())
        .unwrap();
    assert_eq!(
        count, 3,
        "3 people are older than 27 (Alice=30, Charlie=35, Diana=28)"
    );

    // Test MIN/MAX on indexed column
    let min_age: i64 = db.query_one("SELECT MIN(age) FROM tikv_idx_t", ()).unwrap();
    assert_eq!(min_age, 22, "Min age should be 22 (Eve)");

    let max_age: i64 = db.query_one("SELECT MAX(age) FROM tikv_idx_t", ()).unwrap();
    assert_eq!(max_age, 35, "Max age should be 35 (Charlie)");

    // Test with update (index should be maintained)
    db.execute("UPDATE tikv_idx_t SET age = 40 WHERE name = 'Eve'", ())
        .unwrap();
    let new_max: i64 = db.query_one("SELECT MAX(age) FROM tikv_idx_t", ()).unwrap();
    assert_eq!(new_max, 40, "Max age should now be 40 (Eve)");

    // Test with delete (index should be maintained)
    db.execute("DELETE FROM tikv_idx_t WHERE name = 'Eve'", ())
        .unwrap();
    let count_after: i64 = db.query_one("SELECT COUNT(*) FROM tikv_idx_t", ()).unwrap();
    assert_eq!(count_after, 4, "Should have 4 rows after delete");

    // Drop index
    db.execute("DROP INDEX idx_tikv_idx_t_age ON tikv_idx_t", ())
        .unwrap();

    // Query should still work (just without index optimization)
    let count_final: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_idx_t WHERE age > 27", ())
        .unwrap();
    assert_eq!(
        count_final, 2,
        "2 people are older than 27 now (Alice=30, Charlie=35)"
    );

    cleanup(&db, &["tikv_idx_t"]);
}

#[test]
fn test_tikv_unique_index() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_uidx_t"]);

    db.execute(
        "CREATE TABLE tikv_uidx_t (id INTEGER PRIMARY KEY, email TEXT, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("CREATE UNIQUE INDEX idx_email ON tikv_uidx_t(email)", ())
        .unwrap();

    db.execute(
        "INSERT INTO tikv_uidx_t VALUES (1, 'alice@test.com', 'Alice')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_uidx_t VALUES (2, 'bob@test.com', 'Bob')",
        (),
    )
    .unwrap();

    // Inserting duplicate email should fail
    let result = db.execute(
        "INSERT INTO tikv_uidx_t VALUES (3, 'alice@test.com', 'Alice2')",
        (),
    );
    assert!(
        result.is_err(),
        "Duplicate email should violate unique constraint"
    );

    // Different email should succeed
    db.execute(
        "INSERT INTO tikv_uidx_t VALUES (3, 'charlie@test.com', 'Charlie')",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_uidx_t", ())
        .unwrap();
    assert_eq!(count, 3, "Should have 3 rows");

    cleanup(&db, &["tikv_uidx_t"]);
}

// ─── Isolation Level Tests ──────────────────────────────────────

#[test]
fn test_tikv_read_committed() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_rc_t"]);

    db.execute(
        "CREATE TABLE tikv_rc_t (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO tikv_rc_t VALUES (1, 100)", ())
        .unwrap();

    // Start a Read Committed transaction
    let mut tx = db.begin().unwrap();
    tx.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED", ())
        .unwrap();

    // Read initial value
    let val: i64 = tx
        .query_one("SELECT val FROM tikv_rc_t WHERE id = 1", ())
        .unwrap();
    assert_eq!(val, 100, "Should see initial value");

    // Another connection updates the value and commits
    db.execute("UPDATE tikv_rc_t SET val = 200 WHERE id = 1", ())
        .unwrap();

    // In Read Committed, the transaction should see the committed update
    let val2: i64 = tx
        .query_one("SELECT val FROM tikv_rc_t WHERE id = 1", ())
        .unwrap();
    assert_eq!(val2, 200, "Read Committed should see committed changes");

    tx.execute("COMMIT", ()).unwrap();

    cleanup(&db, &["tikv_rc_t"]);
}

// ─── Window Function Tests ───────────────────────────────────────

#[test]
fn test_tikv_window_row_number() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_wf_sales"]);

    db.execute(
        "CREATE TABLE tikv_wf_sales (id INTEGER PRIMARY KEY, salesperson TEXT, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_wf_sales VALUES (1, 'Alice', 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_sales VALUES (2, 'Alice', 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_sales VALUES (3, 'Bob', 150.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_sales VALUES (4, 'Bob', 300.0)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_sales VALUES (5, 'Alice', 50.0)", ())
        .unwrap();

    // ROW_NUMBER with PARTITION BY
    let rows: Vec<_> = db
        .query(
            "SELECT salesperson, amount, ROW_NUMBER() OVER (PARTITION BY salesperson ORDER BY amount) as rn \
             FROM tikv_wf_sales ORDER BY salesperson, amount",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 5);
    // Alice: 50(1), 100(2), 200(3); Bob: 150(1), 300(2)
    let r = rows[0].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    let rn: i64 = r.get(2).unwrap();
    assert_eq!(name, "Alice");
    assert_eq!(rn, 1);

    let r = rows[2].as_ref().unwrap();
    let rn: i64 = r.get(2).unwrap();
    assert_eq!(rn, 3);

    let r = rows[3].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    let rn: i64 = r.get(2).unwrap();
    assert_eq!(name, "Bob");
    assert_eq!(rn, 1);

    cleanup(&db, &["tikv_wf_sales"]);
}

#[test]
fn test_tikv_window_rank() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_wf_scores"]);

    db.execute(
        "CREATE TABLE tikv_wf_scores (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_wf_scores VALUES (1, 'Alice', 100)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_scores VALUES (2, 'Bob', 100)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_scores VALUES (3, 'Carol', 90)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_scores VALUES (4, 'Dave', 80)", ())
        .unwrap();

    // RANK with ties
    let rows: Vec<_> = db
        .query(
            "SELECT name, score, RANK() OVER (ORDER BY score DESC) as rnk \
             FROM tikv_wf_scores ORDER BY score DESC, name",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 4);
    // 100,100 → rank 1,1; 90 → rank 3; 80 → rank 4
    let r = rows[0].as_ref().unwrap();
    let rnk: i64 = r.get(2).unwrap();
    assert_eq!(rnk, 1);

    let r = rows[1].as_ref().unwrap();
    let rnk: i64 = r.get(2).unwrap();
    assert_eq!(rnk, 1); // tie

    let r = rows[2].as_ref().unwrap();
    let rnk: i64 = r.get(2).unwrap();
    assert_eq!(rnk, 3); // gap after tie

    cleanup(&db, &["tikv_wf_scores"]);
}

#[test]
fn test_tikv_window_sum_over() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_wf_running"]);

    db.execute(
        "CREATE TABLE tikv_wf_running (id INTEGER PRIMARY KEY, amount INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_wf_running VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_running VALUES (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_running VALUES (3, 30)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_wf_running VALUES (4, 40)", ())
        .unwrap();

    // Running sum
    let rows: Vec<_> = db
        .query(
            "SELECT id, amount, SUM(amount) OVER (ORDER BY id) as running_sum \
             FROM tikv_wf_running ORDER BY id",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 4);

    let r = rows[0].as_ref().unwrap();
    let running: i64 = r.get(2).unwrap();
    assert_eq!(running, 10);

    let r = rows[1].as_ref().unwrap();
    let running: i64 = r.get(2).unwrap();
    assert_eq!(running, 30); // 10+20

    let r = rows[3].as_ref().unwrap();
    let running: i64 = r.get(2).unwrap();
    assert_eq!(running, 100); // 10+20+30+40

    cleanup(&db, &["tikv_wf_running"]);
}

// ─── CTE Tests ───────────────────────────────────────────────────

#[test]
fn test_tikv_cte_simple() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_cte_emp"]);

    db.execute(
        "CREATE TABLE tikv_cte_emp (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER, dept TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_cte_emp VALUES (1, 'Alice', 90000, 'Engineering')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_emp VALUES (2, 'Bob', 70000, 'Sales')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_emp VALUES (3, 'Carol', 95000, 'Engineering')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_emp VALUES (4, 'Dave', 60000, 'Sales')",
        (),
    )
    .unwrap();

    // Simple CTE with filtering
    let rows: Vec<_> = db
        .query(
            "WITH high_earners AS (\
               SELECT * FROM tikv_cte_emp WHERE salary > 80000\
             ) \
             SELECT name, salary FROM high_earners ORDER BY salary DESC",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 2);
    let r = rows[0].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    assert_eq!(name, "Carol");

    let r = rows[1].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    assert_eq!(name, "Alice");

    cleanup(&db, &["tikv_cte_emp"]);
}

#[test]
fn test_tikv_cte_multiple() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_cte_emp2", "tikv_cte_dept"]);

    db.execute(
        "CREATE TABLE tikv_cte_dept (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tikv_cte_emp2 (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER, department_id INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_cte_dept VALUES (1, 'Engineering', 500000)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO tikv_cte_dept VALUES (2, 'Sales', 200000)", ())
        .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_emp2 VALUES (1, 'Alice', 90000, 1)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO tikv_cte_emp2 VALUES (2, 'Bob', 70000, 2)", ())
        .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_emp2 VALUES (3, 'Carol', 95000, 1)",
        (),
    )
    .unwrap();

    // Multiple CTEs with JOIN
    let rows: Vec<_> = db
        .query(
            "WITH high_earners AS (\
               SELECT * FROM tikv_cte_emp2 WHERE salary > 80000\
             ), \
             big_departments AS (\
               SELECT * FROM tikv_cte_dept WHERE budget > 300000\
             ) \
             SELECT e.name, e.salary, d.name as dept \
             FROM high_earners e \
             JOIN big_departments d ON e.department_id = d.id \
             ORDER BY e.salary DESC",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 2);
    let r = rows[0].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    let salary: i64 = r.get(1).unwrap();
    assert_eq!(name, "Carol");
    assert_eq!(salary, 95000);

    cleanup(&db, &["tikv_cte_emp2", "tikv_cte_dept"]);
}

#[test]
fn test_tikv_cte_with_aggregation() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_cte_sales"]);

    db.execute(
        "CREATE TABLE tikv_cte_sales (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO tikv_cte_sales VALUES (1, 'East', 100)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_cte_sales VALUES (2, 'East', 200)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_cte_sales VALUES (3, 'West', 300)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_cte_sales VALUES (4, 'West', 400)", ())
        .unwrap();
    db.execute("INSERT INTO tikv_cte_sales VALUES (5, 'East', 150)", ())
        .unwrap();

    // CTE with aggregation, then filter
    let rows: Vec<_> = db
        .query(
            "WITH region_totals AS (\
               SELECT region, SUM(amount) as total, COUNT(*) as cnt \
               FROM tikv_cte_sales \
               GROUP BY region\
             ) \
             SELECT region, total FROM region_totals WHERE total > 500 ORDER BY total DESC",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 1);
    let r = rows[0].as_ref().unwrap();
    let region: String = r.get(0).unwrap();
    let total: i64 = r.get(1).unwrap();
    assert_eq!(region, "West");
    assert_eq!(total, 700);

    cleanup(&db, &["tikv_cte_sales"]);
}

#[test]
fn test_tikv_cte_with_window_function() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_cte_wf"]);

    db.execute(
        "CREATE TABLE tikv_cte_wf (id INTEGER PRIMARY KEY, dept TEXT, name TEXT, salary INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tikv_cte_wf VALUES (1, 'Eng', 'Alice', 90000)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_wf VALUES (2, 'Eng', 'Bob', 85000)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_wf VALUES (3, 'Sales', 'Carol', 80000)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_wf VALUES (4, 'Sales', 'Dave', 75000)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO tikv_cte_wf VALUES (5, 'Eng', 'Eve', 95000)",
        (),
    )
    .unwrap();

    // CTE + window function: top earner per department
    let rows: Vec<_> = db
        .query(
            "WITH ranked AS (\
               SELECT name, dept, salary, \
                 RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rnk \
               FROM tikv_cte_wf\
             ) \
             SELECT name, dept, salary FROM ranked WHERE rnk = 1 ORDER BY dept",
            (),
        )
        .unwrap()
        .collect();

    assert_eq!(rows.len(), 2);
    let r = rows[0].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    let dept: String = r.get(1).unwrap();
    assert_eq!(name, "Eve");
    assert_eq!(dept, "Eng");

    let r = rows[1].as_ref().unwrap();
    let name: String = r.get(0).unwrap();
    let dept: String = r.get(1).unwrap();
    assert_eq!(name, "Carol");
    assert_eq!(dept, "Sales");

    cleanup(&db, &["tikv_cte_wf"]);
}

// ─── Temporal Query Tests ────────────────────────────────────────

#[test]
fn test_tikv_as_of_timestamp() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_asof_t"]);

    db.execute(
        "CREATE TABLE tikv_asof_t (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .unwrap();

    // Insert initial data
    db.execute("INSERT INTO tikv_asof_t VALUES (1, 'original')", ())
        .unwrap();

    // Record a timestamp after the insert
    // Sleep a bit to ensure TiKV timestamps advance
    std::thread::sleep(std::time::Duration::from_millis(100));
    let snap_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let snap_nanos = snap_time.as_nanos() as i64;
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Update the data
    db.execute("UPDATE tikv_asof_t SET val = 'modified' WHERE id = 1", ())
        .unwrap();

    // Current data should show 'modified'
    let val: String = db
        .query_one("SELECT val FROM tikv_asof_t WHERE id = 1", ())
        .unwrap();
    assert_eq!(val, "modified");

    // AS OF TIMESTAMP should show 'original' (before the update)
    let query = format!(
        "SELECT val FROM tikv_asof_t AS OF TIMESTAMP {} WHERE id = 1",
        snap_nanos
    );
    let rows: Vec<_> = db.query(&query, ()).unwrap().collect();
    assert_eq!(rows.len(), 1, "Should find the row at historical timestamp");
    let r = rows[0].as_ref().unwrap();
    let val: String = r.get(0).unwrap();
    assert_eq!(val, "original", "AS OF should see the original value");

    cleanup(&db, &["tikv_asof_t"]);
}

// ─── Large Dataset Test ──────────────────────────────────────────

#[test]
fn test_tikv_large_dataset() {
    let dsn = require_tikv!();
    let db = Database::open(&dsn).unwrap();
    cleanup(&db, &["tikv_large_t"]);

    db.execute(
        "CREATE TABLE tikv_large_t (id INTEGER PRIMARY KEY, category TEXT, value INTEGER, score FLOAT)",
        (),
    )
    .unwrap();

    // Insert 10K rows in batches (100K would be too slow for a unit test against TiKV)
    let total_rows = 10_000;
    let categories = ["alpha", "beta", "gamma", "delta", "epsilon"];

    for i in 1..=total_rows {
        let cat = categories[(i - 1) as usize % categories.len()];
        let val = i * 10;
        let score = (i as f64) * 1.5;
        db.execute(
            &format!(
                "INSERT INTO tikv_large_t VALUES ({}, '{}', {}, {})",
                i, cat, val, score
            ),
            (),
        )
        .unwrap();
    }

    // Verify total count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM tikv_large_t", ())
        .unwrap();
    assert_eq!(count, total_rows, "Should have all rows");

    // Aggregation: SUM and AVG
    let sum: i64 = db
        .query_one("SELECT SUM(value) FROM tikv_large_t", ())
        .unwrap();
    let expected_sum: i64 = (1..=total_rows).map(|i| i * 10).sum();
    assert_eq!(sum, expected_sum, "SUM should match");

    // GROUP BY with HAVING
    let rows: Vec<_> = db
        .query(
            "SELECT category, COUNT(*) as cnt, AVG(value) as avg_val \
             FROM tikv_large_t \
             GROUP BY category \
             HAVING COUNT(*) > 1000 \
             ORDER BY category",
            (),
        )
        .unwrap()
        .collect();
    assert_eq!(rows.len(), 5, "All 5 categories should have > 1000 rows");

    // Each category should have exactly total_rows/5 = 2000 rows
    for row in &rows {
        let r = row.as_ref().unwrap();
        let cnt: i64 = r.get(1).unwrap();
        assert_eq!(cnt, total_rows / 5, "Each category should have equal count");
    }

    // ORDER BY with LIMIT
    let rows: Vec<_> = db
        .query(
            "SELECT id, value FROM tikv_large_t ORDER BY value DESC LIMIT 5",
            (),
        )
        .unwrap()
        .collect();
    assert_eq!(rows.len(), 5);
    let r = rows[0].as_ref().unwrap();
    let top_val: i64 = r.get(1).unwrap();
    assert_eq!(
        top_val,
        total_rows * 10,
        "Top value should be last row * 10"
    );

    // Filtered count
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM tikv_large_t WHERE category = 'alpha'",
            (),
        )
        .unwrap();
    assert_eq!(
        count,
        total_rows / 5,
        "Filtered count should be 1/5 of total"
    );

    // Range query
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM tikv_large_t WHERE id BETWEEN 100 AND 200",
            (),
        )
        .unwrap();
    assert_eq!(count, 101, "Range 100-200 inclusive = 101 rows");

    // Window function on large dataset
    let rows: Vec<_> = db
        .query(
            "SELECT category, value, ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as rn \
             FROM tikv_large_t \
             ORDER BY category, rn \
             LIMIT 10",
            (),
        )
        .unwrap()
        .collect();
    assert_eq!(rows.len(), 10);
    // First row should be alpha with highest value, rn=1
    let r = rows[0].as_ref().unwrap();
    let cat: String = r.get(0).unwrap();
    let rn: i64 = r.get(2).unwrap();
    assert_eq!(cat, "alpha");
    assert_eq!(rn, 1);

    cleanup(&db, &["tikv_large_t"]);
}
