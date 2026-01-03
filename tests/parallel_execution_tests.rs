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

//! Integration tests for parallel query execution
//!
//! This module tests parallel query execution features including:
//! - Parallel table scan with filtering (10K+ rows threshold)
//! - Parallel hash joins (5K+ rows threshold)
//! - Parallel ORDER BY (50K+ rows threshold)
//! - Parallel DISTINCT
//! - Partial pushdown optimization (index + complex predicate)
//! - Empty table edge cases
//! - Error propagation in parallel execution
//!
//! Note: Cancellation/timeout behavior is not directly tested here as it would
//! require spawning queries in separate threads and implementing a cancellation
//! mechanism. The Rayon work-stealing scheduler handles this gracefully by
//! default - incomplete work is abandoned when the parent scope exits.

use std::time::Instant;
use stoolap::api::Database;

/// Test parallel filtering on a large dataset
#[test]
fn test_parallel_filter_large_table() {
    let db = Database::open_in_memory().unwrap();

    // Create a table with many rows
    db.execute(
        "CREATE TABLE large_test (id INTEGER PRIMARY KEY, value INTEGER, category TEXT)",
        (),
    )
    .unwrap();

    // Insert 50,000 rows (above parallel threshold of 10,000)
    let start = Instant::now();
    db.execute("BEGIN", ()).unwrap();
    for i in 0..50_000 {
        let category = match i % 5 {
            0 => "A",
            1 => "B",
            2 => "C",
            3 => "D",
            _ => "E",
        };
        db.execute(
            &format!(
                "INSERT INTO large_test VALUES ({}, {}, '{}')",
                i,
                i % 100,
                category
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    let insert_time = start.elapsed();
    println!("Insert 50,000 rows: {:?}", insert_time);

    // Query with WHERE clause that requires memory filtering
    // Using UPPER() function forces in-memory filter evaluation
    let start = Instant::now();
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM large_test WHERE value < 50 AND UPPER(category) = 'A'",
            (),
        )
        .unwrap();
    let query_time = start.elapsed();
    println!("Parallel filter query: {:?}", query_time);

    // Verify results: 10,000 category A rows * 50% filter
    assert_eq!(count, 5_000);
}

/// Test parallel filtering compared to small dataset (sequential path)
#[test]
fn test_parallel_vs_sequential_filter() {
    let db = Database::open_in_memory().unwrap();

    // Small table (below parallel threshold)
    db.execute(
        "CREATE TABLE small_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("BEGIN", ()).unwrap();
    for i in 0..1_000 {
        db.execute(
            &format!("INSERT INTO small_test VALUES ({}, {})", i, i % 10),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // This should use sequential filtering
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM small_test WHERE value < 5", ())
        .unwrap();
    assert_eq!(count, 500); // 50% pass

    // Large table (above parallel threshold)
    db.execute(
        "CREATE TABLE big_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("BEGIN", ()).unwrap();
    for i in 0..20_000 {
        db.execute(
            &format!("INSERT INTO big_test VALUES ({}, {})", i, i % 10),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // This should use parallel filtering
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM big_test WHERE value < 5", ())
        .unwrap();
    assert_eq!(count, 10_000); // 50% pass
}

/// Test parallel filtering with complex expressions
#[test]
fn test_parallel_filter_complex_expression() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE complex_test (id INTEGER PRIMARY KEY, name TEXT, score INTEGER, grade TEXT)",
        (),
    )
    .unwrap();

    // Insert 30,000 rows
    db.execute("BEGIN", ()).unwrap();
    for i in 0..30_000 {
        let name = format!("User{}", i);
        let score = i % 100;
        let grade = if score >= 90 {
            "A"
        } else if score >= 80 {
            "B"
        } else if score >= 70 {
            "C"
        } else {
            "D"
        };
        db.execute(
            &format!(
                "INSERT INTO complex_test VALUES ({}, '{}', {}, '{}')",
                i, name, score, grade
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Complex WHERE with multiple conditions and function calls
    // This forces memory-based filtering which can be parallelized
    let start = Instant::now();
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM complex_test
             WHERE score >= 70 AND score < 90
             AND (UPPER(grade) = 'B' OR UPPER(grade) = 'C')",
            (),
        )
        .unwrap();
    let query_time = start.elapsed();
    println!("Complex parallel filter: {:?}", query_time);

    // B: 80-89 (10 values), C: 70-79 (10 values) = 20 values out of 100
    // 30,000 * 20/100 = 6,000
    assert_eq!(count, 6_000);
}

/// Test that parallel execution produces correct results with aggregation
#[test]
fn test_parallel_filter_with_aggregation() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE agg_test (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
        (),
    )
    .unwrap();

    // Insert 25,000 rows
    db.execute("BEGIN", ()).unwrap();
    for i in 0..25_000 {
        let category = match i % 3 {
            0 => "X",
            1 => "Y",
            _ => "Z",
        };
        db.execute(
            &format!("INSERT INTO agg_test VALUES ({}, '{}', {})", i, category, i),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Test with GROUP BY after parallel filter
    let mut rows = db
        .query(
            "SELECT category, COUNT(*), SUM(amount) FROM agg_test
             WHERE LOWER(category) IN ('x', 'y')
             GROUP BY category ORDER BY category",
            (),
        )
        .unwrap();

    // First row: X
    let row = rows.next().unwrap().unwrap();
    let category: String = row.get(0).unwrap();
    assert_eq!(category, "X");
    // ~8333 rows for X (25000 / 3)
    let x_count: i64 = row.get(1).unwrap();
    assert!(x_count > 8000 && x_count < 9000);

    // Second row: Y
    let row = rows.next().unwrap().unwrap();
    let category: String = row.get(0).unwrap();
    assert_eq!(category, "Y");
    let y_count: i64 = row.get(1).unwrap();
    assert!(y_count > 8000 && y_count < 9000);

    // No more rows (Z was filtered out)
    assert!(rows.next().is_none());
}

/// Benchmark: Compare performance with varying data sizes
#[test]
fn test_parallel_scaling() {
    let db = Database::open_in_memory().unwrap();

    let sizes = [5_000, 10_000, 20_000, 50_000];

    for size in sizes {
        let table_name = format!("scale_test_{}", size);
        db.execute(
            &format!(
                "CREATE TABLE {} (id INTEGER PRIMARY KEY, value INTEGER, data TEXT)",
                table_name
            ),
            (),
        )
        .unwrap();

        // Insert rows
        db.execute("BEGIN", ()).unwrap();
        for i in 0..size {
            db.execute(
                &format!(
                    "INSERT INTO {} VALUES ({}, {}, 'data{}')",
                    table_name,
                    i,
                    i % 100,
                    i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Query with function-based filter (forces memory evaluation)
        let start = Instant::now();
        let count: i64 = db
            .query_one(
                &format!(
                    "SELECT COUNT(*) FROM {} WHERE value < 50 AND LENGTH(data) > 4",
                    table_name
                ),
                (),
            )
            .unwrap();
        let query_time = start.elapsed();

        println!(
            "Size {}: {} rows filtered in {:?} ({} passed)",
            size, size, query_time, count
        );

        // Verify roughly 50% passed the value < 50 filter
        // (minus some that have short data strings)
        assert!(count > (size as i64 / 3));
    }
}

/// Test that EXPLAIN ANALYZE shows Parallel Seq Scan for large tables
#[test]
fn test_explain_analyze_shows_parallel() {
    let db = Database::open_in_memory().unwrap();

    // Create a large table that exceeds parallel threshold (10K rows)
    db.execute(
        "CREATE TABLE explain_parallel_test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .unwrap();

    // Insert 15K rows to exceed threshold
    for batch in 0..15 {
        let mut values = Vec::new();
        for i in 0..1000 {
            let id = batch * 1000 + i + 1;
            values.push(format!("({}, 'name_{}', {})", id, id, id % 100));
        }
        let sql = format!(
            "INSERT INTO explain_parallel_test VALUES {}",
            values.join(", ")
        );
        db.execute(&sql, ()).unwrap();
    }

    // Verify row count
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM explain_parallel_test", ())
        .unwrap();
    assert_eq!(count, 15000, "Should have 15000 rows");

    // Execute EXPLAIN ANALYZE with a WHERE clause
    let mut rows = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM explain_parallel_test WHERE value > 50",
            (),
        )
        .unwrap();

    let mut found_parallel = false;
    let mut explain_output = Vec::new();

    while let Some(Ok(row)) = rows.next() {
        let plan_line: String = row.get(0).unwrap();
        explain_output.push(plan_line.clone());

        if plan_line.contains("Parallel Seq Scan") {
            found_parallel = true;
            // Should show workers count
            assert!(
                plan_line.contains("workers="),
                "Parallel scan should show worker count: {}",
                plan_line
            );
        }
    }

    // Print explain output for debugging
    println!("EXPLAIN ANALYZE output:");
    for line in &explain_output {
        println!("  {}", line);
    }

    assert!(
        found_parallel,
        "EXPLAIN ANALYZE should show 'Parallel Seq Scan' for large table with filter"
    );
}

/// Test that EXPLAIN ANALYZE shows regular Seq Scan for small tables
#[test]
fn test_explain_analyze_sequential_for_small() {
    let db = Database::open_in_memory().unwrap();

    // Create a small table (below parallel threshold)
    db.execute(
        "CREATE TABLE small_explain_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    // Insert only 100 rows (well below 10K threshold)
    let values: Vec<String> = (1..=100).map(|i| format!("({}, {})", i, i % 10)).collect();
    let sql = format!(
        "INSERT INTO small_explain_test VALUES {}",
        values.join(", ")
    );
    db.execute(&sql, ()).unwrap();

    // Execute EXPLAIN ANALYZE
    let mut rows = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM small_explain_test WHERE value > 5",
            (),
        )
        .unwrap();

    let mut found_sequential = false;
    let mut found_parallel = false;

    while let Some(Ok(row)) = rows.next() {
        let plan_line: String = row.get(0).unwrap();
        println!("  {}", plan_line);

        if plan_line.contains("Seq Scan") && !plan_line.contains("Parallel") {
            found_sequential = true;
        }
        if plan_line.contains("Parallel") {
            found_parallel = true;
        }
    }

    assert!(
        found_sequential && !found_parallel,
        "Small table should use regular Seq Scan, not Parallel Seq Scan"
    );
}

/// Test parallel hash join on large tables
#[test]
fn test_parallel_hash_join() {
    let db = Database::open_in_memory().unwrap();

    // Create two tables - one larger than the parallel threshold (5000)
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    // Insert 10,000 customers (above parallel threshold)
    db.execute("BEGIN", ()).unwrap();
    for i in 0..10_000 {
        db.execute(
            &format!("INSERT INTO customers VALUES ({}, 'Customer_{}')", i, i),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Insert 20,000 orders
    db.execute("BEGIN", ()).unwrap();
    for i in 0..20_000 {
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                i % 10_000,      // customer_id cycles through all customers
                (i * 17) % 1000  // varied amounts
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Verify counts
    let customer_count: i64 = db.query_one("SELECT COUNT(*) FROM customers", ()).unwrap();
    let order_count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
    assert_eq!(customer_count, 10_000);
    assert_eq!(order_count, 20_000);

    // Execute INNER join - should use parallel hash join
    let start = Instant::now();
    let inner_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders o INNER JOIN customers c ON o.customer_id = c.id",
            (),
        )
        .unwrap();
    let inner_time = start.elapsed();
    println!("INNER JOIN count: {}, time: {:?}", inner_count, inner_time);
    assert_eq!(inner_count, 20_000); // All orders match a customer

    // Execute LEFT join
    let start = Instant::now();
    let left_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON o.customer_id = c.id",
            (),
        )
        .unwrap();
    let left_time = start.elapsed();
    println!("LEFT JOIN count: {}, time: {:?}", left_count, left_time);
    assert_eq!(left_count, 20_000); // All orders, some with NULL customer

    // Execute RIGHT join
    let start = Instant::now();
    let right_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders o RIGHT JOIN customers c ON o.customer_id = c.id",
            (),
        )
        .unwrap();
    let right_time = start.elapsed();
    println!("RIGHT JOIN count: {}, time: {:?}", right_count, right_time);
    // Each customer has 2 orders (20000/10000), so we get all matches
    assert_eq!(right_count, 20_000);

    println!("Parallel hash join test completed successfully!");
}

/// Test parallel hash join with unmatched rows (OUTER joins)
#[test]
fn test_parallel_hash_join_outer() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE left_table (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE right_table (id INTEGER PRIMARY KEY, left_id INTEGER)",
        (),
    )
    .unwrap();

    // Insert 6000 rows in left table (above 5000 threshold)
    db.execute("BEGIN", ()).unwrap();
    for i in 0..6_000 {
        db.execute(
            &format!("INSERT INTO left_table VALUES ({}, {})", i, i * 10),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Insert 8000 rows in right table, but only half match left table
    db.execute("BEGIN", ()).unwrap();
    for i in 0..8_000 {
        // First 3000 match left_table ids 0-2999
        // Next 5000 reference ids 10000-14999 which don't exist in left_table
        let left_id = if i < 3_000 { i } else { 10_000 + i };
        db.execute(
            &format!("INSERT INTO right_table VALUES ({}, {})", i, left_id),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // INNER JOIN: only matching rows
    let inner_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM left_table l INNER JOIN right_table r ON l.id = r.left_id",
            (),
        )
        .unwrap();
    assert_eq!(inner_count, 3_000, "INNER JOIN should have 3000 matches");

    // LEFT JOIN: all left rows + matches
    let left_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM left_table l LEFT JOIN right_table r ON l.id = r.left_id",
            (),
        )
        .unwrap();
    // 3000 matched left rows + 3000 unmatched left rows (6000-3000)
    assert_eq!(
        left_count, 6_000,
        "LEFT JOIN should have 6000 rows (all left rows)"
    );

    // RIGHT JOIN: all right rows + matches
    let right_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM left_table l RIGHT JOIN right_table r ON l.id = r.left_id",
            (),
        )
        .unwrap();
    // All 8000 right rows (3000 matched + 5000 unmatched)
    assert_eq!(
        right_count, 8_000,
        "RIGHT JOIN should have 8000 rows (all right rows)"
    );

    // FULL OUTER JOIN: all rows from both sides
    let full_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM left_table l FULL OUTER JOIN right_table r ON l.id = r.left_id",
            (),
        )
        .unwrap();
    // 3000 matched + 3000 unmatched left + 5000 unmatched right = 11000
    assert_eq!(full_count, 11_000, "FULL OUTER JOIN should have 11000 rows");

    println!("Parallel hash join OUTER test completed successfully!");
}

/// Test that indexed queries still use indexes and don't bypass to parallel path
/// This verifies that the parallel path only activates when storage can't filter
#[test]
fn test_indexed_query_not_parallel_bypassed() {
    let db = Database::open_in_memory().unwrap();

    // Create table with index
    db.execute(
        "CREATE TABLE indexed_test (id INTEGER PRIMARY KEY, value INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    // Create an index on value column
    db.execute("CREATE INDEX idx_value ON indexed_test(value)", ())
        .unwrap();

    // Insert 15,000 rows (above parallel threshold)
    db.execute("BEGIN", ()).unwrap();
    for i in 0..15_000 {
        db.execute(
            &format!(
                "INSERT INTO indexed_test VALUES ({}, {}, 'name_{}')",
                i,
                i % 100, // values 0-99
                i
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Query using indexed column - should use index, NOT parallel full scan
    let start = Instant::now();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM indexed_test WHERE value = 50", ())
        .unwrap();
    let indexed_time = start.elapsed();

    // Should find 150 rows (15000 / 100 = 150 per value)
    assert_eq!(count, 150);

    // Query using non-indexed column - might use parallel path for large datasets
    let start = Instant::now();
    let count2: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM indexed_test WHERE name LIKE 'name_1%'",
            (),
        )
        .unwrap();
    let _non_indexed_time = start.elapsed();

    // Should find rows starting with 'name_1' (1, 10-19, 100-199, 1000-1999, 10000-14999)
    // 1 + 10 + 100 + 1000 + 5000 = 6111
    assert_eq!(count2, 6111);

    println!(
        "Indexed query time: {:?}, should be much faster than full scan",
        indexed_time
    );
}

/// Test parallel ORDER BY at SQL level
#[test]
fn test_parallel_order_by_sql() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE sort_test (id INTEGER PRIMARY KEY, value INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    // Insert 60,000 rows (above ORDER BY parallel threshold of 50,000)
    db.execute("BEGIN", ()).unwrap();
    for i in 0..60_000 {
        db.execute(
            &format!(
                "INSERT INTO sort_test VALUES ({}, {}, 'name_{}')",
                i,
                (60_000 - i) % 1000, // reverse order values
                i
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // ORDER BY should use parallel sort for large dataset
    let start = Instant::now();
    let rows = db
        .query(
            "SELECT id, value FROM sort_test ORDER BY value ASC LIMIT 10",
            (),
        )
        .unwrap();

    let mut results = Vec::new();
    for row in rows {
        let row = row.unwrap();
        let value: i64 = row.get(1).unwrap();
        results.push(value);
    }
    let sort_time = start.elapsed();

    // First 10 should be the smallest values (0)
    assert!(!results.is_empty());
    assert_eq!(results[0], 0, "First value should be 0 (smallest)");

    // Verify sorting worked correctly
    for i in 1..results.len() {
        assert!(
            results[i] >= results[i - 1],
            "Results should be in ascending order"
        );
    }

    println!("Parallel ORDER BY time for 60K rows: {:?}", sort_time);
}

/// Test parallel DISTINCT at SQL level
#[test]
fn test_parallel_distinct_sql() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE distinct_test (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .unwrap();

    // Insert 50,000 rows with only 100 distinct categories
    db.execute("BEGIN", ()).unwrap();
    for i in 0..50_000 {
        db.execute(
            &format!(
                "INSERT INTO distinct_test VALUES ({}, 'cat_{}')",
                i,
                i % 100
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    let start = Instant::now();
    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT category) FROM distinct_test", ())
        .unwrap();
    let distinct_time = start.elapsed();

    assert_eq!(count, 100, "Should have exactly 100 distinct categories");
    println!("Parallel DISTINCT time for 50K rows: {:?}", distinct_time);

    // Also test SELECT DISTINCT
    let start = Instant::now();
    let rows = db
        .query("SELECT DISTINCT category FROM distinct_test", ())
        .unwrap();
    let mut distinct_values = Vec::new();
    for row in rows {
        let row = row.unwrap();
        let cat: String = row.get(0).unwrap();
        distinct_values.push(cat);
    }
    let select_distinct_time = start.elapsed();

    assert_eq!(
        distinct_values.len(),
        100,
        "SELECT DISTINCT should return 100 rows"
    );
    println!(
        "SELECT DISTINCT time for 50K rows: {:?}",
        select_distinct_time
    );
}

/// Test parallel execution with empty tables
#[test]
fn test_parallel_empty_tables() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE empty_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    // Query empty table - should handle gracefully
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM empty_test WHERE value > 50", ())
        .unwrap();
    assert_eq!(count, 0, "Empty table should return 0 count");

    // DISTINCT on empty table
    let count: i64 = db
        .query_one("SELECT COUNT(DISTINCT value) FROM empty_test", ())
        .unwrap();
    assert_eq!(count, 0, "Empty DISTINCT should return 0");

    // ORDER BY on empty table
    let mut rows = db
        .query("SELECT * FROM empty_test ORDER BY value", ())
        .unwrap();
    assert!(
        rows.next().is_none(),
        "Empty ORDER BY should return no rows"
    );

    // JOIN with empty table
    db.execute(
        "CREATE TABLE non_empty (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO non_empty VALUES (1, 100), (2, 200)", ())
        .unwrap();

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM empty_test e JOIN non_empty n ON e.id = n.id",
            (),
        )
        .unwrap();
    assert_eq!(count, 0, "JOIN with empty table should return 0");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM empty_test e LEFT JOIN non_empty n ON e.id = n.id",
            (),
        )
        .unwrap();
    assert_eq!(count, 0, "LEFT JOIN from empty table should return 0");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM non_empty n LEFT JOIN empty_test e ON n.id = e.id",
            (),
        )
        .unwrap();
    assert_eq!(
        count, 2,
        "LEFT JOIN to empty table should preserve non-empty rows"
    );

    println!("Empty table tests passed!");
}

/// Test error propagation during parallel execution
#[test]
fn test_parallel_error_handling() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE error_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    // Insert enough rows to trigger parallel path
    db.execute("BEGIN", ()).unwrap();
    for i in 0..15_000 {
        db.execute(&format!("INSERT INTO error_test VALUES ({}, {})", i, i), ())
            .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Query with division by zero in WHERE - should handle error gracefully
    // Note: This tests that errors during parallel evaluation don't crash
    let result: Result<i64, _> = db.query_one(
        "SELECT COUNT(*) FROM error_test WHERE value / (id - 5000) > 0",
        (),
    );
    // The query might succeed (filtering out problematic rows) or fail
    // Either way, it shouldn't panic or hang
    match result {
        Ok(count) => {
            println!("Query succeeded with count: {}", count);
        }
        Err(e) => {
            println!("Query failed as expected: {:?}", e);
        }
    }

    // Query with type mismatch - should return proper error
    let result = db.query("SELECT * FROM error_test WHERE value = 'not_a_number'", ());
    // This should either error or return empty (no rows match)
    match result {
        Ok(mut rows) => {
            let count = rows.by_ref().count();
            println!("Type mismatch query returned {} rows", count);
        }
        Err(e) => {
            println!("Type mismatch query failed: {:?}", e);
        }
    }

    println!("Error handling tests completed!");
}

/// Test that queries with complex expressions use memory filter path correctly
#[test]
fn test_complex_expression_memory_filter() {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE complex_test (id INTEGER PRIMARY KEY, value INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    // Insert 15,000 rows
    db.execute("BEGIN", ()).unwrap();
    for i in 0..15_000 {
        db.execute(
            &format!(
                "INSERT INTO complex_test VALUES ({}, {}, 'name_{}')",
                i,
                i % 100,
                i
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Query with function that can't be pushed to storage
    // UPPER() is a function that requires memory evaluation
    let start = Instant::now();
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM complex_test WHERE UPPER(name) LIKE 'NAME_1%'",
            (),
        )
        .unwrap();
    let complex_time = start.elapsed();

    // Should find same rows as the LIKE test above
    assert_eq!(count, 6111);
    println!(
        "Complex expression (UPPER) filter time for 15K rows: {:?}",
        complex_time
    );

    // Query combining index-able and non-index-able conditions
    // With partial pushdown: indexed part (value = 50) should use index,
    // non-indexed part (LENGTH(name) > 5) filtered in memory
    let start = Instant::now();
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM complex_test WHERE value = 50 AND LENGTH(name) > 5",
            (),
        )
        .unwrap();
    let combined_time = start.elapsed();

    // value = 50 would be 150 rows, but LENGTH(name) > 5 filters some
    // name_50 to name_99 (50 names) have 7+ chars, plus name_150 to name_199 (50), etc.
    // Actually all names have > 5 chars except 0-9 (name_0 to name_9 = 6 chars)
    // So all 150 rows with value=50 should pass
    assert_eq!(count, 150);
    println!(
        "Combined index + complex filter time for 15K rows: {:?}",
        combined_time
    );
}

/// Test partial pushdown optimization for mixed WHERE clauses
/// This tests that `WHERE indexed_col = X AND complex_func(y)` uses index for first part
#[test]
fn test_partial_pushdown_optimization() {
    let db = Database::open_in_memory().unwrap();

    // Create table with index
    db.execute(
        "CREATE TABLE partial_test (id INTEGER PRIMARY KEY, indexed_val INTEGER, text_col TEXT)",
        (),
    )
    .unwrap();

    // Create index on indexed_val
    db.execute("CREATE INDEX idx_partial ON partial_test(indexed_val)", ())
        .unwrap();

    // Insert 20,000 rows - indexed_val has values 0-99 (200 rows each)
    db.execute("BEGIN", ()).unwrap();
    for i in 0..20_000 {
        db.execute(
            &format!(
                "INSERT INTO partial_test VALUES ({}, {}, 'text_{}')",
                i,
                i % 100,
                i
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // Test 1: Pure indexed query (should be fast - index scan)
    let start = Instant::now();
    let indexed_only: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM partial_test WHERE indexed_val = 42",
            (),
        )
        .unwrap();
    let indexed_time = start.elapsed();
    assert_eq!(indexed_only, 200); // 20000 / 100 = 200 rows per value

    // Test 2: Mixed query - indexed + complex expression
    // With partial pushdown, should use index for indexed_val = 42, then filter LENGTH(text_col)
    let start = Instant::now();
    let mixed: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM partial_test WHERE indexed_val = 42 AND LENGTH(text_col) > 6",
            (),
        )
        .unwrap();
    let mixed_time = start.elapsed();

    // text_42, text_142, text_242, ... up to text_19942
    // text_42 = 7 chars (> 6) ✓
    // text_142 = 8 chars (> 6) ✓
    // etc. All should pass LENGTH > 6
    assert_eq!(mixed, 200);

    // Test 3: Complex-only query (should be slower - full scan)
    let start = Instant::now();
    let complex_only: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM partial_test WHERE LENGTH(text_col) > 6",
            (),
        )
        .unwrap();
    let complex_time = start.elapsed();

    // All text_col values are "text_X" where X goes from 0 to 19999
    // text_0 to text_9 = 6 chars (not > 6) - 10 rows
    // text_10 onwards = 7+ chars (> 6) - 19990 rows
    assert_eq!(complex_only, 19_990);

    println!("Partial pushdown test:");
    println!(
        "  - Pure indexed query (indexed_val = 42): {:?}",
        indexed_time
    );
    println!("  - Mixed query (indexed_val + LENGTH): {:?}", mixed_time);
    println!("  - Complex-only query (LENGTH only): {:?}", complex_time);

    // The mixed query should be much faster than complex-only because it uses the index
    // to reduce from 20K rows to 200 rows before applying LENGTH filter
    // Note: In debug mode timing can be unreliable, so we just verify correctness
    println!(
        "Speedup from partial pushdown: {:.1}x (20K rows → 200 rows via index)",
        complex_time.as_secs_f64() / mixed_time.as_secs_f64().max(0.0001)
    );
}

#[test]
fn test_debug_upper_like_pattern() {
    use stoolap::Database;

    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE test_upper (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO test_upper VALUES (1, 'name_1')", ())
        .unwrap();
    db.execute("INSERT INTO test_upper VALUES (2, 'name_10')", ())
        .unwrap();
    db.execute("INSERT INTO test_upper VALUES (3, 'name_2')", ())
        .unwrap();

    // Check UPPER function works
    let upper_val: String = db
        .query_one("SELECT UPPER(name) FROM test_upper WHERE id = 1", ())
        .unwrap();
    println!("UPPER('name_1'): '{}'", upper_val);

    // Check UPPER(name) = 'NAME_1' works
    let eq_test: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE UPPER(name) = 'NAME_1'",
            (),
        )
        .unwrap();
    println!("UPPER(name) = 'NAME_1': {}", eq_test);

    // Test without underscore wildcard - use pattern without _
    let like_no_underscore: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE UPPER(name) LIKE 'NAME%'",
            (),
        )
        .unwrap();
    println!("UPPER(name) LIKE 'NAME%': {}", like_no_underscore);

    // Test simpler underscore pattern - just one underscore match
    let like_simple_underscore: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE name LIKE 'name__'",
            (),
        )
        .unwrap();
    println!("name LIKE 'name__': {}", like_simple_underscore);

    // Test just the underscore in the middle
    let like_middle_underscore: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE UPPER(name) LIKE 'NAME__'",
            (),
        )
        .unwrap();
    println!("UPPER(name) LIKE 'NAME__': {}", like_middle_underscore);

    // Debug: Get the actual values and filter result
    let debug_query = "SELECT id, name, UPPER(name) as upper_name,
                       CASE WHEN UPPER(name) LIKE 'NAME__' THEN 1 ELSE 0 END as matches
                       FROM test_upper";
    let rows = db.query(debug_query, ()).unwrap();
    println!("Debug output:");
    for row in rows.flatten() {
        println!(
            "  id={:?}, name={:?}, UPPER={:?}, matches={:?}",
            row.get::<i64>(0),
            row.get::<String>(1),
            row.get::<String>(2),
            row.get::<i64>(3)
        );
    }

    // Check simple LIKE works on uppercase literal
    let like_test: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE 'NAME_1' LIKE 'NAME_1%'",
            (),
        )
        .unwrap();
    println!("'NAME_1' LIKE 'NAME_1%': {}", like_test);

    // Direct pattern match test
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE name LIKE 'name_1%'",
            (),
        )
        .unwrap();
    println!("name LIKE 'name_1%': {}", count);

    // UPPER pattern test
    let count2: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_upper WHERE UPPER(name) LIKE 'NAME_1%'",
            (),
        )
        .unwrap();
    println!("UPPER(name) LIKE 'NAME_1%': {}", count2);

    assert_eq!(count, 2, "Direct LIKE should match name_1 and name_10");
    assert_eq!(count2, 2, "UPPER LIKE should match NAME_1 and NAME_10");
}
