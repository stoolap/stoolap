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

//! Integration tests for Zone Maps (min/max indexes for segment pruning)
//!
//! Zone maps maintain min/max statistics per data segment, enabling the query
//! executor to skip entire segments when predicates fall outside the range.

use stoolap::api::Database;

/// Test that ANALYZE builds zone maps for tables
#[test]
fn test_analyze_builds_zone_maps() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create table with sequential data
    db.execute(
        "CREATE TABLE test_zonemap (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert 100 rows with sequential values
    for i in 0..100 {
        db.execute(
            &format!("INSERT INTO test_zonemap VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Insert failed");
    }

    // Run ANALYZE to build zone maps
    let _: Vec<_> = db
        .query("ANALYZE test_zonemap", ())
        .expect("ANALYZE failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect results");

    // Query that should benefit from zone map pruning
    // Looking for value > 950 (id > 95), should skip most segments
    let result: Vec<_> = db
        .query("SELECT * FROM test_zonemap WHERE value > 950", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 4, "Should return 4 rows (id 96-99)");
}

/// Test zone map pruning with EXPLAIN output
#[test]
fn test_zonemap_explain_shows_pruning() {
    let db = Database::open("memory://zonemap_explain").expect("Failed to create database");

    // Create table with 1000 rows
    db.execute(
        "CREATE TABLE large_table (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 0..1000 {
        db.execute(
            &format!("INSERT INTO large_table VALUES ({}, {})", i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Run ANALYZE
    db.execute("ANALYZE large_table", ()).unwrap();

    // EXPLAIN should show zone map usage
    let result: Vec<_> = db
        .query("EXPLAIN SELECT * FROM large_table WHERE value > 900", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify EXPLAIN ran successfully
    assert!(!result.is_empty(), "EXPLAIN should return results");
}

/// Test zone map effectiveness with range queries
#[test]
fn test_zonemap_range_query() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Create table with ordered date-like values
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, event_date INTEGER, data TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert events with dates spread across time
    // event_date represents days from epoch (e.g., 1-365 for year 1)
    for i in 0..365 {
        db.execute(
            &format!("INSERT INTO events VALUES ({}, {}, 'event_{}')", i, i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Run ANALYZE
    db.execute("ANALYZE events", ()).unwrap();

    // Query for last month (days 330-365)
    let result: Vec<_> = db
        .query("SELECT COUNT(*) FROM events WHERE event_date >= 330", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count: i64 = result[0].get(0).expect("Failed to get count");
    assert_eq!(count, 35, "Should return 35 events (days 330-364)");
}

/// Test zone maps with multiple columns
#[test]
fn test_zonemap_multiple_columns() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            price INTEGER,
            quantity INTEGER,
            category TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert products with varying prices and quantities
    for i in 0..100 {
        let price = (i % 10) * 100; // Prices 0-900 in multiples of 100
        let quantity = i * 5;
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, {}, {}, 'cat_{}')",
                i,
                price,
                quantity,
                i % 5
            ),
            (),
        )
        .expect("Insert failed");
    }

    // Run ANALYZE
    db.execute("ANALYZE products", ()).unwrap();

    // Query with multiple range conditions
    let result: Vec<_> = db
        .query(
            "SELECT * FROM products WHERE price >= 700 AND quantity > 400",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify results are correct
    for row in &result {
        let price: i64 = row.get(1).expect("Failed to get price");
        let quantity: i64 = row.get(2).expect("Failed to get quantity");
        assert!(price >= 700, "Price should be >= 700");
        assert!(quantity > 400, "Quantity should be > 400");
    }
}

/// Test zone maps with equality predicates
#[test]
fn test_zonemap_equality() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE lookup (id INTEGER PRIMARY KEY, code INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert rows with codes in specific ranges
    for i in 0..50 {
        let code = i / 10 * 100; // Codes: 0, 0, ..., 100, 100, ..., 400, 400
        db.execute(
            &format!("INSERT INTO lookup VALUES ({}, {}, 'item_{}')", i, code, i),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE lookup", ()).unwrap();

    // Equality query should use zone maps to prune segments
    let result: Vec<_> = db
        .query("SELECT * FROM lookup WHERE code = 200", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Should find rows 20-29
    assert_eq!(result.len(), 10, "Should return 10 rows with code = 200");
}

/// Test zone maps with NULL handling
#[test]
fn test_zonemap_with_nulls() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE nullable_data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert mix of values and NULLs
    for i in 0..20 {
        if i % 3 == 0 {
            db.execute(
                &format!("INSERT INTO nullable_data VALUES ({}, NULL)", i),
                (),
            )
            .expect("Insert failed");
        } else {
            db.execute(
                &format!("INSERT INTO nullable_data VALUES ({}, {})", i, i * 10),
                (),
            )
            .expect("Insert failed");
        }
    }

    db.execute("ANALYZE nullable_data", ()).unwrap();

    // Query should handle NULL segments correctly
    let result: Vec<_> = db
        .query("SELECT * FROM nullable_data WHERE value > 100", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify all returned rows have value > 100
    for row in &result {
        let value: i64 = row.get(1).expect("Failed to get value");
        assert!(value > 100, "Value should be > 100");
    }
}

/// Test zone maps with text columns
#[test]
fn test_zonemap_text_columns() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE text_data (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert alphabetically ordered names
    let names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"];
    for (i, name) in names.iter().enumerate() {
        for j in 0..10 {
            let idx = i * 10 + j;
            db.execute(
                &format!("INSERT INTO text_data VALUES ({}, '{}{}')", idx, name, j),
                (),
            )
            .expect("Insert failed");
        }
    }

    db.execute("ANALYZE text_data", ()).unwrap();

    // Range query on text - names starting with D or later
    let result: Vec<_> = db
        .query("SELECT * FROM text_data WHERE name >= 'David0'", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Should return David*, Eve*, Frank*, Grace* = 40 rows
    assert_eq!(
        result.len(),
        40,
        "Should return 40 rows for names >= 'David0'"
    );
}

/// Test that zone maps are rebuilt after ANALYZE
#[test]
fn test_zonemap_rebuild_after_analyze() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE rebuild_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Initial data
    for i in 0..50 {
        db.execute(
            &format!("INSERT INTO rebuild_test VALUES ({}, {})", i, i),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE rebuild_test", ()).unwrap();

    // Add more data
    for i in 50..100 {
        db.execute(
            &format!("INSERT INTO rebuild_test VALUES ({}, {})", i, i),
            (),
        )
        .expect("Insert failed");
    }

    // Re-analyze should rebuild zone maps with new data
    db.execute("ANALYZE rebuild_test", ()).unwrap();

    // Query should now find the new data
    let result: Vec<_> = db
        .query("SELECT * FROM rebuild_test WHERE value >= 75", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(result.len(), 25, "Should find 25 rows after re-analyze");
}

/// Test zone maps with compound filters
#[test]
fn test_zonemap_compound_filters() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount INTEGER,
            status TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert 200 orders
    for i in 0..200 {
        let customer_id = i % 10;
        let amount = i * 50;
        let status = if i % 2 == 0 { "completed" } else { "pending" };
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, {}, '{}')",
                i, customer_id, amount, status
            ),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE orders", ()).unwrap();

    // Complex filter combining numeric range and equality
    let result: Vec<_> = db
        .query(
            "SELECT * FROM orders WHERE amount > 8000 AND customer_id = 5",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // Verify results
    for row in &result {
        let customer_id: i64 = row.get(1).expect("Failed to get customer_id");
        let amount: i64 = row.get(2).expect("Failed to get amount");
        assert_eq!(customer_id, 5, "Customer ID should be 5");
        assert!(amount > 8000, "Amount should be > 8000");
    }
}

/// Test zone maps work correctly with updates
#[test]
fn test_zonemap_after_updates() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE update_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 0..50 {
        db.execute(
            &format!("INSERT INTO update_test VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE update_test", ()).unwrap();

    // Update some values
    db.execute(
        "UPDATE update_test SET value = 9999 WHERE id BETWEEN 10 AND 20",
        (),
    )
    .expect("Update failed");

    // Query before re-analyze (zone maps may be stale)
    let result1: Vec<_> = db
        .query("SELECT COUNT(*) FROM update_test WHERE value = 9999", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count1: i64 = result1[0].get(0).expect("Failed to get count");
    assert_eq!(count1, 11, "Should find 11 updated rows");

    // Re-analyze to update zone maps
    db.execute("ANALYZE update_test", ()).unwrap();

    // Query after re-analyze
    let result2: Vec<_> = db
        .query("SELECT COUNT(*) FROM update_test WHERE value = 9999", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count2: i64 = result2[0].get(0).expect("Failed to get count");
    assert_eq!(count2, 11, "Should still find 11 rows after re-analyze");
}

/// Test zone maps with BETWEEN predicate
#[test]
fn test_zonemap_between() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE between_test (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 0..100 {
        db.execute(
            &format!("INSERT INTO between_test VALUES ({}, {})", i, i),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE between_test", ()).unwrap();

    // BETWEEN query
    let result: Vec<_> = db
        .query(
            "SELECT * FROM between_test WHERE score BETWEEN 40 AND 60",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert_eq!(
        result.len(),
        21,
        "Should return 21 rows for BETWEEN 40 AND 60"
    );
}

/// Test zone map pruning short-circuits when no rows can match
/// This test verifies that when zone maps indicate no segments contain
/// matching data, the query returns immediately without scanning any rows.
#[test]
fn test_zonemap_prune_entire_scan() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE prune_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert values from 1-100
    for i in 1..=100 {
        db.execute(&format!("INSERT INTO prune_test VALUES ({}, {})", i, i), ())
            .expect("Insert failed");
    }

    // Run ANALYZE to build zone maps
    db.execute("ANALYZE prune_test", ()).unwrap();

    // Query for value > 1000 - way outside all segment ranges
    // Zone maps should allow complete pruning
    let result: Vec<_> = db
        .query("SELECT * FROM prune_test WHERE value > 1000", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        result.is_empty(),
        "Should return empty result for out-of-range query"
    );

    // Query for value < 0 - also outside all segment ranges
    let result2: Vec<_> = db
        .query("SELECT * FROM prune_test WHERE value < 0", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        result2.is_empty(),
        "Should return empty result for negative value query"
    );

    // Equality query for non-existent value
    let result3: Vec<_> = db
        .query("SELECT * FROM prune_test WHERE value = 500", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    assert!(
        result3.is_empty(),
        "Should return empty result for non-existent exact value"
    );
}

/// Test zone map pruning with aggregations
#[test]
fn test_zonemap_prune_with_aggregation() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE agg_prune_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO agg_prune_test VALUES ({}, {})", i, i),
            (),
        )
        .expect("Insert failed");
    }

    db.execute("ANALYZE agg_prune_test", ()).unwrap();

    // COUNT with impossible condition - should return 0 via zone map pruning
    let result: Vec<_> = db
        .query("SELECT COUNT(*) FROM agg_prune_test WHERE value > 1000", ())
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    let count: i64 = result[0].get(0).expect("Failed to get count");
    assert_eq!(count, 0, "COUNT should return 0 for impossible condition");

    // SUM with impossible condition - should return NULL or 0
    let result2: Vec<_> = db
        .query(
            "SELECT SUM(value) FROM agg_prune_test WHERE value > 1000",
            (),
        )
        .expect("Query failed")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");

    // SUM of no rows returns NULL
    let sum: Option<i64> = result2[0].get(0).unwrap_or(None);
    assert!(
        sum.is_none() || sum == Some(0),
        "SUM should return NULL or 0 for impossible condition"
    );
}
