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

//! Integration tests for Semantic Query Caching
//!
//! This module tests the semantic query cache which detects when a new query's
//! results can be computed by filtering cached results from a previous query.
//!
//! # Test Coverage
//!
//! 1. Basic cache operations (insert, lookup, invalidation)
//! 2. Predicate subsumption detection:
//!    - Numeric range tightening (>, <, >=, <=)
//!    - AND conjunction strengthening
//!    - IN list subsumption
//! 3. Cache invalidation on DML operations
//! 4. Performance benefits of subsumption hits

use stoolap::api::Database;

/// Helper function to create an in-memory test database
fn create_test_db() -> Database {
    let db = Database::open_in_memory().expect("Failed to create in-memory database");
    db
}

/// Test basic semantic cache operations: insert and lookup
#[test]
fn test_semantic_cache_basic_operations() {
    let db = create_test_db();

    // Create and populate test table
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount INTEGER, status TEXT)",
        (),
    )
    .unwrap();

    // Insert test data
    db.execute(
        "INSERT INTO orders VALUES
         (1, 100, 'pending'),
         (2, 200, 'pending'),
         (3, 300, 'shipped'),
         (4, 400, 'delivered'),
         (5, 500, 'pending')",
        (),
    )
    .unwrap();

    // Execute a query - this should populate the cache
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders WHERE amount > 200", ())
        .unwrap();
    assert_eq!(count, 3);

    // Execute a stricter query - could potentially use cached result
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM orders WHERE amount > 300", ())
        .unwrap();
    assert_eq!(count, 2);
}

/// Test cache invalidation on INSERT
#[test]
fn test_semantic_cache_invalidation_on_insert() {
    let db = create_test_db();

    // Create table
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO products VALUES (1, 100), (2, 200)", ())
        .unwrap();

    // Verify count
    let count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
    assert_eq!(count, 2);

    // Insert more data - this should invalidate the cache
    db.execute("INSERT INTO products VALUES (3, 300)", ())
        .unwrap();

    // Verify new count
    let count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
    assert_eq!(count, 3);
}

/// Test cache invalidation on UPDATE
#[test]
fn test_semantic_cache_invalidation_on_update() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE inventory (id INTEGER PRIMARY KEY, quantity INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO inventory VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();

    // Initial sum
    let sum: i64 = db
        .query_one("SELECT SUM(quantity) FROM inventory", ())
        .unwrap();
    assert_eq!(sum, 60);

    // Update - should invalidate cache
    db.execute("UPDATE inventory SET quantity = 100 WHERE id = 1", ())
        .unwrap();

    // Verify updated sum
    let sum: i64 = db
        .query_one("SELECT SUM(quantity) FROM inventory", ())
        .unwrap();
    assert_eq!(sum, 150);
}

/// Test cache invalidation on DELETE
#[test]
fn test_semantic_cache_invalidation_on_delete() {
    let db = create_test_db();

    db.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, level TEXT)", ())
        .unwrap();

    db.execute(
        "INSERT INTO logs VALUES (1, 'info'), (2, 'warn'), (3, 'error')",
        (),
    )
    .unwrap();

    // Initial count
    let count: i64 = db.query_one("SELECT COUNT(*) FROM logs", ()).unwrap();
    assert_eq!(count, 3);

    // Delete - should invalidate cache
    db.execute("DELETE FROM logs WHERE level = 'info'", ())
        .unwrap();

    // Verify updated count
    let count: i64 = db.query_one("SELECT COUNT(*) FROM logs", ()).unwrap();
    assert_eq!(count, 2);
}

/// Test cache invalidation on TRUNCATE
#[test]
fn test_semantic_cache_invalidation_on_truncate() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE temp_data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO temp_data VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();

    // Initial count
    let count: i64 = db.query_one("SELECT COUNT(*) FROM temp_data", ()).unwrap();
    assert_eq!(count, 3);

    // Truncate - should invalidate cache
    db.execute("TRUNCATE TABLE temp_data", ()).unwrap();

    // Verify empty
    let count: i64 = db.query_one("SELECT COUNT(*) FROM temp_data", ()).unwrap();
    assert_eq!(count, 0);
}

/// Test semantic cache with range queries
#[test]
fn test_semantic_cache_range_queries() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    // Insert 1000 rows
    for i in 1..=1000 {
        db.execute(
            &format!("INSERT INTO metrics VALUES ({}, {})", i, i * 10),
            (),
        )
        .unwrap();
    }

    // Query 1: Get all rows where value > 5000
    let count1: i64 = db
        .query_one("SELECT COUNT(*) FROM metrics WHERE value > 5000", ())
        .unwrap();
    assert_eq!(count1, 500); // IDs 501-1000 have values > 5000

    // Query 2: Get all rows where value > 8000 (stricter, subset of Query 1)
    let count2: i64 = db
        .query_one("SELECT COUNT(*) FROM metrics WHERE value > 8000", ())
        .unwrap();
    assert_eq!(count2, 200); // IDs 801-1000 have values > 8000
}

/// Test semantic cache with IN list queries
#[test]
fn test_semantic_cache_in_list_queries() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO categories VALUES
         (1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'E')",
        (),
    )
    .unwrap();

    // Query 1: Get categories 1, 2, 3, 4, 5
    let count1: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM categories WHERE id IN (1, 2, 3, 4, 5)",
            (),
        )
        .unwrap();
    assert_eq!(count1, 5);

    // Query 2: Get categories 2, 3 (subset of Query 1)
    let count2: i64 = db
        .query_one("SELECT COUNT(*) FROM categories WHERE id IN (2, 3)", ())
        .unwrap();
    assert_eq!(count2, 2);
}

/// Test semantic cache with AND conjunction queries
#[test]
fn test_semantic_cache_and_conjunction() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            severity INTEGER
        )",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        let event_type = if i % 2 == 0 { "error" } else { "info" };
        let severity = i % 10;
        db.execute(
            &format!(
                "INSERT INTO events VALUES ({}, '{}', {})",
                i, event_type, severity
            ),
            (),
        )
        .unwrap();
    }

    // Query 1: Get high severity events
    let count1: i64 = db
        .query_one("SELECT COUNT(*) FROM events WHERE severity > 5", ())
        .unwrap();
    assert_eq!(count1, 40); // severity 6,7,8,9 = 4 values * 10 cycles = 40

    // Query 2: Get high severity error events (stricter - adds AND condition)
    let count2: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM events WHERE severity > 5 AND event_type = 'error'",
            (),
        )
        .unwrap();
    assert_eq!(count2, 20); // Half of 40 are errors
}

/// Test that different tables don't share cache entries
#[test]
fn test_semantic_cache_table_isolation() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE table_a (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE table_b (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO table_a VALUES (1, 100), (2, 200)", ())
        .unwrap();
    db.execute("INSERT INTO table_b VALUES (1, 999), (2, 888)", ())
        .unwrap();

    // Query table_a
    let sum_a: i64 = db.query_one("SELECT SUM(value) FROM table_a", ()).unwrap();
    assert_eq!(sum_a, 300);

    // Query table_b - should not use table_a's cache
    let sum_b: i64 = db.query_one("SELECT SUM(value) FROM table_b", ()).unwrap();
    assert_eq!(sum_b, 1887);

    // Modify table_a - should not affect table_b's future queries
    db.execute("INSERT INTO table_a VALUES (3, 300)", ())
        .unwrap();

    // Table_b should still return correct results
    let sum_b: i64 = db.query_one("SELECT SUM(value) FROM table_b", ()).unwrap();
    assert_eq!(sum_b, 1887);
}

/// Test semantic cache with complex filter expressions
#[test]
fn test_semantic_cache_complex_filters() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE transactions (
            id INTEGER PRIMARY KEY,
            amount INTEGER,
            fee INTEGER,
            status TEXT
        )",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        let status = if i % 3 == 0 {
            "completed"
        } else if i % 3 == 1 {
            "pending"
        } else {
            "failed"
        };
        db.execute(
            &format!(
                "INSERT INTO transactions VALUES ({}, {}, {}, '{}')",
                i,
                i * 100,
                i * 10,
                status
            ),
            (),
        )
        .unwrap();
    }

    // Query with complex expression
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM transactions WHERE amount > 2000 AND fee < 300",
            (),
        )
        .unwrap();
    // amount > 2000 means id > 20, fee < 300 means id < 30
    // So id in (21, 22, 23, 24, 25, 26, 27, 28, 29) = 9 rows
    assert_eq!(count, 9);
}

/// Test semantic cache with BETWEEN queries
#[test]
fn test_semantic_cache_between_queries() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE readings (id INTEGER PRIMARY KEY, temperature INTEGER)",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO readings VALUES ({}, {})", i, i), ())
            .unwrap();
    }

    // Query with BETWEEN
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM readings WHERE temperature BETWEEN 30 AND 70",
            (),
        )
        .unwrap();
    assert_eq!(count, 41); // 30-70 inclusive

    // Tighter BETWEEN (subset)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM readings WHERE temperature BETWEEN 40 AND 60",
            (),
        )
        .unwrap();
    assert_eq!(count, 21); // 40-60 inclusive
}

/// Test semantic cache performance benefit
#[test]
fn test_semantic_cache_performance() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE large_table (
            id INTEGER PRIMARY KEY,
            category INTEGER,
            value INTEGER
        )",
        (),
    )
    .unwrap();

    // Insert 10000 rows
    for batch in 0..100 {
        let mut sql = String::from("INSERT INTO large_table VALUES ");
        for i in 0..100 {
            let id = batch * 100 + i;
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&format!("({}, {}, {})", id, id % 10, id * 7));
        }
        db.execute(&sql, ()).unwrap();
    }

    // Warm up: Execute query
    let _: i64 = db
        .query_one("SELECT COUNT(*) FROM large_table WHERE category = 5", ())
        .unwrap();

    // Execute same query multiple times - should hit cache
    for _ in 0..5 {
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM large_table WHERE category = 5", ())
            .unwrap();
        assert_eq!(count, 1000); // 10000 / 10 categories = 1000 per category
    }

    // The cache should have been used for subsequent queries
    // (exact stats depend on implementation details)
}

/// Test semantic cache with equality predicates
#[test]
fn test_semantic_cache_equality_predicates() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, level INTEGER)",
        (),
    )
    .unwrap();

    for i in 1..=100 {
        let status = if i % 2 == 0 { "active" } else { "inactive" };
        db.execute(
            &format!("INSERT INTO users VALUES ({}, '{}', {})", i, status, i % 5),
            (),
        )
        .unwrap();
    }

    // Query with equality
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users WHERE status = 'active'", ())
        .unwrap();
    assert_eq!(count, 50);

    // Query with equality + additional condition (stricter)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users WHERE status = 'active' AND level > 2",
            (),
        )
        .unwrap();
    assert_eq!(count, 20); // levels 3,4 = 2/5 of 50 = 20
}

/// Test semantic cache with NULL handling
#[test]
fn test_semantic_cache_null_handling() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE nullable_data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO nullable_data VALUES (1, 100), (2, NULL), (3, 300), (4, NULL), (5, 500)",
        (),
    )
    .unwrap();

    // Query non-null values
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM nullable_data WHERE value IS NOT NULL",
            (),
        )
        .unwrap();
    assert_eq!(count, 3);

    // Query null values
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM nullable_data WHERE value IS NULL", ())
        .unwrap();
    assert_eq!(count, 2);
}

/// Test semantic cache doesn't return stale data after updates
#[test]
fn test_semantic_cache_no_stale_data() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE counter (id INTEGER PRIMARY KEY, count INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO counter VALUES (1, 0)", ()).unwrap();

    // Read initial count
    let count: i64 = db
        .query_one("SELECT count FROM counter WHERE id = 1", ())
        .unwrap();
    assert_eq!(count, 0);

    // Increment multiple times
    for expected in 1..=10 {
        db.execute("UPDATE counter SET count = count + 1 WHERE id = 1", ())
            .unwrap();

        // Each read should see the latest value, not cached
        let count: i64 = db
            .query_one("SELECT count FROM counter WHERE id = 1", ())
            .unwrap();
        assert_eq!(count, expected);
    }
}

/// Test semantic cache with multiple concurrent-style queries
#[test]
fn test_semantic_cache_query_patterns() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            amount INTEGER,
            year INTEGER
        )",
        (),
    )
    .unwrap();

    // Insert sales data
    for i in 1..=200 {
        let region = match i % 4 {
            0 => "North",
            1 => "South",
            2 => "East",
            _ => "West",
        };
        let year = 2020 + (i % 5);
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, '{}', {}, {})",
                i,
                region,
                i * 100,
                year
            ),
            (),
        )
        .unwrap();
    }

    // Various query patterns that could benefit from caching/subsumption

    // Broad query
    let total: i64 = db
        .query_one("SELECT SUM(amount) FROM sales WHERE year >= 2020", ())
        .unwrap();
    assert!(total > 0);

    // Narrower query (same column, stricter condition)
    let recent: i64 = db
        .query_one("SELECT SUM(amount) FROM sales WHERE year >= 2023", ())
        .unwrap();
    assert!(recent < total);

    // Different dimension
    let north: i64 = db
        .query_one("SELECT SUM(amount) FROM sales WHERE region = 'North'", ())
        .unwrap();
    assert!(north > 0);

    // Combined filters
    let north_recent: i64 = db
        .query_one(
            "SELECT SUM(amount) FROM sales WHERE region = 'North' AND year >= 2023",
            (),
        )
        .unwrap();
    assert!(north_recent <= north);
}

/// Test that verifies the semantic cache is actually being used
/// by checking the cache statistics after queries
#[test]
fn test_semantic_cache_stats_verification() {
    let db = create_test_db();

    // Create and populate test table
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price INTEGER, quantity INTEGER)",
        (),
    )
    .unwrap();

    // Insert test data
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO products VALUES ({}, {}, {})", i, i * 10, i * 5),
            (),
        )
        .unwrap();
    }

    // Clear any previous cache state
    db.clear_semantic_cache().unwrap();

    // Get initial stats
    let stats_before = db.semantic_cache_stats().unwrap();
    assert_eq!(stats_before.hits, 0);
    assert_eq!(stats_before.misses, 0);

    // Query 1: SELECT * with WHERE - this should be cached
    let rows: Vec<_> = db
        .query("SELECT * FROM products WHERE price > 500", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), 50); // price > 500 means id > 50

    // Check stats: Should be a cache miss (first query)
    let stats_after_first = db.semantic_cache_stats().unwrap();
    assert_eq!(
        stats_after_first.misses, 1,
        "First query should be a cache miss"
    );

    // Query 2: Exact same query - should be exact hit
    let rows2: Vec<_> = db
        .query("SELECT * FROM products WHERE price > 500", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows2.len(), 50);

    let stats_after_second = db.semantic_cache_stats().unwrap();
    assert_eq!(
        stats_after_second.exact_hits, 1,
        "Second identical query should be an exact hit"
    );

    // Query 3: Stricter query - should be subsumption hit
    let rows3: Vec<_> = db
        .query("SELECT * FROM products WHERE price > 800", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows3.len(), 20); // price > 800 means id > 80

    let stats_after_third = db.semantic_cache_stats().unwrap();
    assert_eq!(
        stats_after_third.subsumption_hits, 1,
        "Stricter query should be a subsumption hit"
    );

    // Verify total hits
    assert_eq!(stats_after_third.hits, 2, "Should have 2 total cache hits");
}

/// Test that cache invalidation on INSERT clears relevant entries
#[test]
fn test_semantic_cache_invalidation_clears_stats() {
    let db = create_test_db();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    for i in 1..=50 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i * 10), ())
            .unwrap();
    }

    db.clear_semantic_cache().unwrap();

    // First query - populates cache
    let _: Vec<_> = db
        .query("SELECT * FROM items WHERE value > 100", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Second query - should hit cache
    let _: Vec<_> = db
        .query("SELECT * FROM items WHERE value > 100", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let stats_before_insert = db.semantic_cache_stats().unwrap();
    assert_eq!(stats_before_insert.exact_hits, 1);

    // Insert new data - should invalidate cache
    db.execute("INSERT INTO items VALUES (51, 510)", ())
        .unwrap();

    // Query again - should be a miss because cache was invalidated
    let _: Vec<_> = db
        .query("SELECT * FROM items WHERE value > 100", ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let stats_after_insert = db.semantic_cache_stats().unwrap();
    // Should have one more miss after invalidation
    assert!(
        stats_after_insert.misses > stats_before_insert.misses,
        "Query after INSERT should be a cache miss due to invalidation"
    );
}
