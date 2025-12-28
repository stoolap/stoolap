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

//! Tests for index optimizer functionality
//!
//! Tests MIN/MAX index optimization, COUNT(*) optimization, keyset pagination,
//! IN list/subquery index probing, and ORDER BY index optimization.

use stoolap::Database;

fn setup_indexed_table(db: &Database) {
    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price FLOAT,
            category TEXT,
            stock INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Create indexes
    db.execute("CREATE INDEX idx_price ON products(price)", ())
        .expect("Failed to create price index");
    db.execute("CREATE INDEX idx_category ON products(category)", ())
        .expect("Failed to create category index");
    db.execute("CREATE INDEX idx_stock ON products(stock)", ())
        .expect("Failed to create stock index");

    // Insert test data
    let inserts = [
        "INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics', 50)",
        "INSERT INTO products VALUES (2, 'Phone', 599.99, 'Electronics', 100)",
        "INSERT INTO products VALUES (3, 'Tablet', 399.99, 'Electronics', 75)",
        "INSERT INTO products VALUES (4, 'Chair', 149.99, 'Furniture', 200)",
        "INSERT INTO products VALUES (5, 'Desk', 299.99, 'Furniture', 30)",
        "INSERT INTO products VALUES (6, 'Lamp', 49.99, 'Furniture', 150)",
        "INSERT INTO products VALUES (7, 'Book', 19.99, 'Books', 500)",
        "INSERT INTO products VALUES (8, 'Notebook', 9.99, 'Books', 1000)",
        "INSERT INTO products VALUES (9, 'Pen', 4.99, 'Books', 2000)",
        "INSERT INTO products VALUES (10, 'Monitor', 349.99, 'Electronics', 40)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert data");
    }
}

// =============================================================================
// MIN/MAX Index Optimization Tests
// =============================================================================

#[test]
fn test_min_on_indexed_column() {
    let db = Database::open("memory://min_indexed").expect("Failed to create database");
    setup_indexed_table(&db);

    let min_price: f64 = db
        .query_one("SELECT MIN(price) FROM products", ())
        .expect("Failed to query MIN");

    assert!(
        (min_price - 4.99).abs() < 0.01,
        "Expected MIN(price) = 4.99, got {}",
        min_price
    );
}

#[test]
fn test_max_on_indexed_column() {
    let db = Database::open("memory://max_indexed").expect("Failed to create database");
    setup_indexed_table(&db);

    let max_price: f64 = db
        .query_one("SELECT MAX(price) FROM products", ())
        .expect("Failed to query MAX");

    assert!(
        (max_price - 999.99).abs() < 0.01,
        "Expected MAX(price) = 999.99, got {}",
        max_price
    );
}

#[test]
fn test_min_with_alias() {
    let db = Database::open("memory://min_alias").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query("SELECT MIN(price) AS min_price FROM products", ())
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert_eq!(columns[0], "min_price", "Expected alias 'min_price'");
}

#[test]
fn test_max_with_alias() {
    let db = Database::open("memory://max_alias").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query("SELECT MAX(price) AS max_price FROM products", ())
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert_eq!(columns[0], "max_price", "Expected alias 'max_price'");
}

#[test]
fn test_min_on_integer_column() {
    let db = Database::open("memory://min_integer").expect("Failed to create database");
    setup_indexed_table(&db);

    let min_stock: i64 = db
        .query_one("SELECT MIN(stock) FROM products", ())
        .expect("Failed to query MIN");

    assert_eq!(min_stock, 30, "Expected MIN(stock) = 30");
}

#[test]
fn test_max_on_integer_column() {
    let db = Database::open("memory://max_integer").expect("Failed to create database");
    setup_indexed_table(&db);

    let max_stock: i64 = db
        .query_one("SELECT MAX(stock) FROM products", ())
        .expect("Failed to query MAX");

    assert_eq!(max_stock, 2000, "Expected MAX(stock) = 2000");
}

// =============================================================================
// COUNT(*) Optimization Tests
// =============================================================================

#[test]
fn test_count_star_optimization() {
    let db = Database::open("memory://count_star").expect("Failed to create database");
    setup_indexed_table(&db);

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products", ())
        .expect("Failed to query COUNT");

    assert_eq!(count, 10, "Expected COUNT(*) = 10");
}

#[test]
fn test_count_star_with_alias() {
    let db = Database::open("memory://count_star_alias").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query("SELECT COUNT(*) AS total FROM products", ())
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert_eq!(columns[0], "total", "Expected alias 'total'");
}

// =============================================================================
// Keyset Pagination Tests (id > X ORDER BY id LIMIT Y)
// =============================================================================

#[test]
fn test_keyset_pagination_basic() {
    let db = Database::open("memory://keyset_basic").expect("Failed to create database");
    setup_indexed_table(&db);

    // Get products with id > 5, ordered by id, limit 3
    let result = db
        .query(
            "SELECT id, name FROM products WHERE id > 5 ORDER BY id LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![6, 7, 8], "Expected ids [6, 7, 8]");
}

#[test]
fn test_keyset_pagination_gte() {
    let db = Database::open("memory://keyset_gte").expect("Failed to create database");
    setup_indexed_table(&db);

    // Get products with id >= 8, ordered by id, limit 5
    let result = db
        .query(
            "SELECT id FROM products WHERE id >= 8 ORDER BY id LIMIT 5",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![8, 9, 10], "Expected ids [8, 9, 10]");
}

#[test]
fn test_keyset_pagination_reversed_comparison() {
    let db = Database::open("memory://keyset_reversed").expect("Failed to create database");
    setup_indexed_table(&db);

    // Test reversed comparison: 3 < id (same as id > 3)
    let result = db
        .query(
            "SELECT id FROM products WHERE 3 < id ORDER BY id LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![4, 5, 6], "Expected ids [4, 5, 6]");
}

// =============================================================================
// IN List Index Optimization Tests
// =============================================================================

#[test]
fn test_in_list_with_pk() {
    let db = Database::open("memory://in_list_pk").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id, name FROM products WHERE id IN (1, 3, 5, 7, 9)",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    ids.sort();

    assert_eq!(ids, vec![1, 3, 5, 7, 9], "Expected odd ids");
}

#[test]
fn test_in_list_with_index() {
    let db = Database::open("memory://in_list_index").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id, category FROM products WHERE category IN ('Electronics', 'Books')",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(1).unwrap();
        assert!(
            category == "Electronics" || category == "Books",
            "Unexpected category: {}",
            category
        );
        count += 1;
    }

    assert_eq!(count, 7, "Expected 7 rows (4 Electronics + 3 Books)");
}

#[test]
fn test_in_list_with_limit() {
    let db = Database::open("memory://in_list_limit").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id FROM products WHERE id IN (1, 2, 3, 4, 5) LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows due to LIMIT");
}

#[test]
fn test_in_list_with_offset() {
    let db = Database::open("memory://in_list_offset").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id FROM products WHERE id IN (1, 2, 3, 4, 5) LIMIT 2 OFFSET 2",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 rows due to OFFSET 2 LIMIT 2");
}

#[test]
fn test_in_list_empty() {
    let db = Database::open("memory://in_list_empty").expect("Failed to create database");
    setup_indexed_table(&db);

    // Empty IN list - should return no rows
    let result = db
        .query("SELECT id FROM products WHERE id IN (100, 200, 300)", ())
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 rows for non-existent ids");
}

#[test]
fn test_in_list_with_additional_predicate() {
    let db = Database::open("memory://in_list_additional_pred").expect("Failed to create database");
    setup_indexed_table(&db);

    // IN list with additional WHERE condition
    let result = db
        .query(
            "SELECT id, price FROM products WHERE id IN (1, 2, 3, 4, 5) AND price > 200",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let price: f64 = row.get(1).unwrap();
        assert!(price > 200.0, "Expected price > 200, got {}", price);
        ids.push(id);
    }
    ids.sort();

    // Products with id in (1,2,3,4,5) and price > 200:
    // id=1 (Laptop, 999.99), id=2 (Phone, 599.99), id=3 (Tablet, 399.99), id=5 (Desk, 299.99)
    // All have price > 200
    assert!(!ids.is_empty(), "Expected some ids with price > 200");
}

// =============================================================================
// IN Subquery Index Optimization Tests
// =============================================================================

#[test]
fn test_in_subquery_with_pk() {
    let db = Database::open("memory://in_subquery_pk").expect("Failed to create database");
    setup_indexed_table(&db);

    // Create another table for subquery
    db.execute(
        "CREATE TABLE featured_ids (product_id INTEGER PRIMARY KEY)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO featured_ids VALUES (1)", ())
        .unwrap();
    db.execute("INSERT INTO featured_ids VALUES (3)", ())
        .unwrap();
    db.execute("INSERT INTO featured_ids VALUES (5)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT id, name FROM products WHERE id IN (SELECT product_id FROM featured_ids)",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }
    ids.sort();

    assert_eq!(ids, vec![1, 3, 5], "Expected featured product ids");
}

#[test]
fn test_not_in_subquery_with_pk() {
    let db = Database::open("memory://not_in_subquery_pk").expect("Failed to create database");
    setup_indexed_table(&db);

    // Create table with ids to exclude
    db.execute(
        "CREATE TABLE excluded_ids (product_id INTEGER PRIMARY KEY)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO excluded_ids VALUES (1)", ())
        .unwrap();
    db.execute("INSERT INTO excluded_ids VALUES (2)", ())
        .unwrap();
    db.execute("INSERT INTO excluded_ids VALUES (3)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT id FROM products WHERE id NOT IN (SELECT product_id FROM excluded_ids) ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![4, 5, 6, 7, 8, 9, 10], "Expected non-excluded ids");
}

#[test]
fn test_in_subquery_empty_result() {
    let db = Database::open("memory://in_subquery_empty").expect("Failed to create database");
    setup_indexed_table(&db);

    // Create empty table
    db.execute(
        "CREATE TABLE empty_ids (product_id INTEGER PRIMARY KEY)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query(
            "SELECT id FROM products WHERE id IN (SELECT product_id FROM empty_ids)",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 rows for empty subquery");
}

// =============================================================================
// ORDER BY Index Optimization Tests
// =============================================================================

#[test]
fn test_order_by_indexed_column_asc_limit() {
    let db = Database::open("memory://order_by_asc").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query("SELECT id, price FROM products ORDER BY price LIMIT 3", ())
        .expect("Failed to query");

    let mut prices = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let price: f64 = row.get(1).unwrap();
        prices.push(price);
    }

    assert_eq!(prices.len(), 3, "Expected 3 rows");
    // Verify sorted ascending
    assert!(
        prices[0] <= prices[1] && prices[1] <= prices[2],
        "Expected ascending order"
    );
}

#[test]
fn test_order_by_indexed_column_desc_limit() {
    let db = Database::open("memory://order_by_desc").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id, price FROM products ORDER BY price DESC LIMIT 3",
            (),
        )
        .expect("Failed to query");

    let mut prices = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let price: f64 = row.get(1).unwrap();
        prices.push(price);
    }

    assert_eq!(prices.len(), 3, "Expected 3 rows");
    // Verify sorted descending
    assert!(
        prices[0] >= prices[1] && prices[1] >= prices[2],
        "Expected descending order"
    );
}

#[test]
fn test_order_by_pk_with_offset() {
    let db = Database::open("memory://order_by_offset").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query("SELECT id FROM products ORDER BY id LIMIT 3 OFFSET 5", ())
        .expect("Failed to query");

    let mut ids = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        ids.push(id);
    }

    assert_eq!(ids, vec![6, 7, 8], "Expected ids [6, 7, 8] after OFFSET 5");
}

// =============================================================================
// Window Function Safety Tests
// =============================================================================

#[test]
fn test_window_row_number_with_limit() {
    let db = Database::open("memory://window_row_number").expect("Failed to create database");
    setup_indexed_table(&db);

    let result = db
        .query(
            "SELECT id, name, ROW_NUMBER() OVER (ORDER BY price) as rn FROM products LIMIT 5",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let rn: i64 = row.get(2).unwrap();
        count += 1;
        assert!((1..=5).contains(&rn), "Expected row number 1-5, got {}", rn);
    }

    assert_eq!(count, 5, "Expected 5 rows");
}

#[test]
fn test_window_rank_basic() {
    // Use a fresh, minimal table for RANK test
    let db = Database::open("memory://rank_simple").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO items VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (2, 20)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (3, 20)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (4, 30)", ()).unwrap();

    let result = db
        .query(
            "SELECT id, val, RANK() OVER (ORDER BY val) as rnk FROM items",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let rnk: i64 = row.get(2).unwrap();
        assert!(rnk >= 1, "Expected rank >= 1, got {}", rnk);
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 rows");
}

#[test]
fn test_window_dense_rank_basic() {
    // Use a fresh, minimal table for DENSE_RANK test
    let db = Database::open("memory://dense_rank_simple").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO items VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (2, 20)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (3, 20)", ()).unwrap();
    db.execute("INSERT INTO items VALUES (4, 30)", ()).unwrap();

    let result = db
        .query(
            "SELECT id, val, DENSE_RANK() OVER (ORDER BY val) as drnk FROM items",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let drnk: i64 = row.get(2).unwrap();
        assert!(drnk >= 1, "Expected dense_rank >= 1, got {}", drnk);
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 rows");
}

// =============================================================================
// Equality Condition Detection Tests (for hash join eligibility)
// =============================================================================

#[test]
fn test_join_with_equality() {
    let db = Database::open("memory://join_equality").expect("Failed to create database");
    setup_indexed_table(&db);

    // Create orders table
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 1, 2)", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (2, 2, 1)", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (3, 1, 3)", ())
        .unwrap();

    // Join with equality condition
    let result = db
        .query(
            "SELECT p.name, o.quantity
             FROM products p
             INNER JOIN orders o ON p.id = o.product_id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 joined rows");
}

// =============================================================================
// Cardinality Estimation Tests
// =============================================================================

#[test]
fn test_explain_shows_plan() {
    let db = Database::open("memory://explain_estimates").expect("Failed to create database");
    setup_indexed_table(&db);

    // Run ANALYZE first
    db.execute("ANALYZE products", ())
        .expect("Failed to ANALYZE");

    let result = db
        .query("EXPLAIN SELECT * FROM products WHERE price > 100", ())
        .expect("Failed to EXPLAIN");

    let mut lines = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let plan: String = row.get(0).unwrap();
        lines.push(plan);
    }

    // EXPLAIN should return something about the query plan
    assert!(!lines.is_empty(), "Expected EXPLAIN to return plan lines");
}
