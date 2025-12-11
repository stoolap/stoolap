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

//! Index Operations Tests
//!
//! Tests CREATE INDEX, DROP INDEX, and queries with indexes

use stoolap::Database;

fn setup_index_table(db: &Database) {
    db.execute(
        "CREATE TABLE index_test (
            id INTEGER,
            name TEXT,
            category TEXT,
            value FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO index_test VALUES
        (1, 'Item 1', 'A', 10.5),
        (2, 'Item 2', 'A', 20.1),
        (3, 'Item 3', 'B', 30.7),
        (4, 'Item 4', 'B', 40.2),
        (5, 'Item 5', 'B', 50.9),
        (6, 'Item 6', 'C', 60.3),
        (7, 'Item 7', 'C', 70.8),
        (8, 'Item 8', 'A', 80.4),
        (9, 'Item 9', 'D', 90.6),
        (10, 'Item 10', 'D', 100.1)",
        (),
    )
    .expect("Failed to insert data");
}

/// Test creating an index
#[test]
fn test_create_index() {
    let db = Database::open("memory://index_create").expect("Failed to create database");
    setup_index_table(&db);

    // Create index on category column
    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create index");

    // Verify the index works by querying
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM index_test WHERE category = 'B'", ())
        .expect("Failed to query");
    assert_eq!(count, 3, "Expected 3 rows with category='B'");
}

/// Test creating an index on value column
#[test]
fn test_create_index_on_value() {
    let db = Database::open("memory://index_value").expect("Failed to create database");
    setup_index_table(&db);

    // Create index on value column
    db.execute("CREATE INDEX idx_value ON index_test(value)", ())
        .expect("Failed to create btree index");

    // Query using the indexed column
    let result = db
        .query(
            "SELECT id, name, value FROM index_test WHERE value > 50.0 AND value < 90.0",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let value: f64 = row.get(2).unwrap();
        assert!(
            value > 50.0 && value < 90.0,
            "Value {} should be in range (50, 90)",
            value
        );
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 rows in value range");
}

/// Test DROP INDEX IF EXISTS
#[test]
fn test_drop_index_if_exists() {
    let db = Database::open("memory://index_drop").expect("Failed to create database");
    setup_index_table(&db);

    // Create an index
    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create index");

    // Drop the index
    db.execute("DROP INDEX IF EXISTS idx_category ON index_test", ())
        .expect("Failed to drop index");

    // Try to drop a non-existent index (should not fail with IF EXISTS)
    db.execute("DROP INDEX IF EXISTS idx_nonexistent ON index_test", ())
        .expect("DROP INDEX IF EXISTS should not fail for non-existent index");
}

/// Test unique index
#[test]
fn test_unique_index() {
    let db = Database::open("memory://index_unique").expect("Failed to create database");
    setup_index_table(&db);

    // Create unique index on id column
    db.execute("CREATE UNIQUE INDEX idx_id ON index_test(id)", ())
        .expect("Failed to create unique index");

    // Try to insert a duplicate ID - should fail
    let result = db.execute(
        "INSERT INTO index_test VALUES (1, 'Duplicate', 'X', 999.9)",
        (),
    );
    assert!(result.is_err(), "Should fail due to unique constraint");
}

/// Test query with category filter
#[test]
fn test_query_with_category_filter() {
    let db = Database::open("memory://index_cat_filter").expect("Failed to create database");
    setup_index_table(&db);

    // Create index on category
    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create index");

    let result = db
        .query(
            "SELECT id, name, category FROM index_test WHERE category = 'B'",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(2).unwrap();
        assert_eq!(category, "B", "Expected category 'B'");
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows with category='B'");
}

/// Test query with range filter
#[test]
fn test_query_with_range_filter() {
    let db = Database::open("memory://index_range").expect("Failed to create database");
    setup_index_table(&db);

    // Create index on value
    db.execute("CREATE INDEX idx_value ON index_test(value)", ())
        .expect("Failed to create index");

    // Range query: values between 30 and 70 inclusive
    let result = db
        .query(
            "SELECT id, value FROM index_test WHERE value >= 30.0 AND value <= 70.0",
            (),
        )
        .expect("Failed to query");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let value: f64 = row.get(1).unwrap();
        assert!(
            value >= 30.0 && value <= 70.0,
            "Value {} should be in range [30, 70]",
            value
        );
        ids.push(id);
    }

    // Expected: ids 3, 4, 5, 6 (values 30.7, 40.2, 50.9, 60.3)
    assert_eq!(ids.len(), 4, "Expected 4 rows in range");
}

/// Test query with string range filter
#[test]
fn test_query_with_string_range_filter() {
    let db = Database::open("memory://index_str_range").expect("Failed to create database");
    setup_index_table(&db);

    // Create index on category
    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create index");

    // Range query: categories between 'B' and 'C' inclusive
    let result = db
        .query(
            "SELECT id, category FROM index_test WHERE category >= 'B' AND category <= 'C'",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(1).unwrap();
        assert!(
            category >= "B".to_string() && category <= "C".to_string(),
            "Category {} should be in range ['B', 'C']",
            category
        );
        count += 1;
    }

    // Expected: ids 3,4,5 (B) and 6,7 (C) = 5 total
    assert_eq!(count, 5, "Expected 5 rows in category range");
}

/// Test multiple indexes on same table
#[test]
fn test_multiple_indexes() {
    let db = Database::open("memory://index_multi").expect("Failed to create database");
    setup_index_table(&db);

    // Create multiple indexes
    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create category index");
    db.execute("CREATE INDEX idx_value ON index_test(value)", ())
        .expect("Failed to create value index");

    // Query using category index
    let count1: i64 = db
        .query_one("SELECT COUNT(*) FROM index_test WHERE category = 'A'", ())
        .expect("Failed to query");
    assert_eq!(count1, 3, "Expected 3 rows with category='A'");

    // Query using value index
    let count2: i64 = db
        .query_one("SELECT COUNT(*) FROM index_test WHERE value > 80.0", ())
        .expect("Failed to query");
    assert_eq!(count2, 3, "Expected 3 rows with value > 80");
}

/// Test index with combined conditions
#[test]
fn test_index_combined_conditions() {
    let db = Database::open("memory://index_combined").expect("Failed to create database");
    setup_index_table(&db);

    db.execute("CREATE INDEX idx_category ON index_test(category)", ())
        .expect("Failed to create category index");
    db.execute("CREATE INDEX idx_value ON index_test(value)", ())
        .expect("Failed to create value index");

    // Query using both conditions
    let result = db
        .query(
            "SELECT id, category, value FROM index_test WHERE category = 'B' AND value > 40.0",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let category: String = row.get(1).unwrap();
        let value: f64 = row.get(2).unwrap();
        assert_eq!(category, "B");
        assert!(value > 40.0);
        count += 1;
    }

    // Expected: ids 4, 5 (B with value > 40)
    assert_eq!(count, 2, "Expected 2 rows matching combined conditions");
}

// ============================================================================
// Multi-Column Index Tests
// ============================================================================

/// Test creating a multi-column index
#[test]
fn test_create_multi_column_index() {
    let db = Database::open("memory://multi_col_create").expect("Failed to create database");
    setup_index_table(&db);

    // Create a multi-column index on (category, value)
    db.execute(
        "CREATE INDEX idx_cat_val ON index_test(category, value)",
        (),
    )
    .expect("Failed to create multi-column index");

    // Verify the index works by querying - the index should help with combined conditions
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM index_test WHERE category = 'B' AND value > 40.0",
            (),
        )
        .expect("Failed to query");
    assert_eq!(count, 2, "Expected 2 rows with category='B' AND value > 40");
}

/// Test multi-column index with three columns
#[test]
fn test_multi_column_index_three_columns() {
    let db = Database::open("memory://multi_col_three").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (
            customer_id INTEGER,
            product_id INTEGER,
            status TEXT,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert test data
    for i in 1..=100 {
        let customer_id = (i % 10) + 1;
        let product_id = (i % 20) + 1;
        let status = match i % 3 {
            0 => "pending",
            1 => "shipped",
            _ => "delivered",
        };
        let amount = i as f64 * 10.5;
        db.execute(
            &format!(
                "INSERT INTO orders VALUES ({}, {}, '{}', {})",
                customer_id, product_id, status, amount
            ),
            (),
        )
        .expect("Failed to insert");
    }

    // Create 3-column index
    db.execute(
        "CREATE INDEX idx_customer_product_status ON orders(customer_id, product_id, status)",
        (),
    )
    .expect("Failed to create 3-column index");

    // Query using all 3 columns
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM orders WHERE customer_id = 1 AND product_id = 1 AND status = 'delivered'",
            (),
        )
        .expect("Failed to query");
    // customer_id=1 when i%10=0 (i=10,20,30,...,100) -> 10 rows
    // Of those, product_id=1 when i%20=0 (i=20,40,60,80,100) -> 5 rows
    // Of those, status='delivered' when i%3=2 (i=20,80) -> 2 rows
    assert_eq!(count, 2, "Expected 2 rows matching all 3 conditions");
}

/// Test unique multi-column index
#[test]
fn test_unique_multi_column_index() {
    let db = Database::open("memory://multi_col_unique").expect("Failed to create database");

    db.execute(
        "CREATE TABLE user_roles (
            user_id INTEGER,
            role_id INTEGER,
            assigned_at TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute(
        "INSERT INTO user_roles VALUES (1, 1, '2024-01-01'), (1, 2, '2024-01-02'), (2, 1, '2024-01-03')",
        (),
    )
    .expect("Failed to insert data");

    // Create unique multi-column index on (user_id, role_id)
    db.execute(
        "CREATE UNIQUE INDEX idx_user_role ON user_roles(user_id, role_id)",
        (),
    )
    .expect("Failed to create unique multi-column index");

    // Try to insert duplicate combination - should fail
    let result = db.execute("INSERT INTO user_roles VALUES (1, 1, '2024-02-01')", ());
    assert!(
        result.is_err(),
        "Should fail due to unique constraint on (user_id, role_id)"
    );

    // Insert different combination - should succeed
    db.execute("INSERT INTO user_roles VALUES (1, 3, '2024-02-01')", ())
        .expect("Should succeed with different role_id");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM user_roles WHERE user_id = 1", ())
        .expect("Failed to query");
    assert_eq!(count, 3, "Expected 3 roles for user_id=1");
}

/// Test DROP INDEX on multi-column index
#[test]
fn test_drop_multi_column_index() {
    let db = Database::open("memory://multi_col_drop").expect("Failed to create database");
    setup_index_table(&db);

    // Create multi-column index
    db.execute(
        "CREATE INDEX idx_cat_val ON index_test(category, value)",
        (),
    )
    .expect("Failed to create multi-column index");

    // Drop the index
    db.execute("DROP INDEX idx_cat_val ON index_test", ())
        .expect("Failed to drop multi-column index");

    // Try to drop again with IF EXISTS - should not fail
    db.execute("DROP INDEX IF EXISTS idx_cat_val ON index_test", ())
        .expect("DROP INDEX IF EXISTS should not fail for non-existent index");
}

/// Test multi-column index with IF NOT EXISTS
#[test]
fn test_multi_column_index_if_not_exists() {
    let db = Database::open("memory://multi_col_if_not_exists").expect("Failed to create database");
    setup_index_table(&db);

    // Create multi-column index
    db.execute(
        "CREATE INDEX idx_cat_val ON index_test(category, value)",
        (),
    )
    .expect("Failed to create multi-column index");

    // Try to create again with IF NOT EXISTS - should not fail
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_cat_val ON index_test(category, value)",
        (),
    )
    .expect("CREATE INDEX IF NOT EXISTS should not fail for existing index");
}

/// Test multi-column index preserves data after insert/update/delete
#[test]
fn test_multi_column_index_data_integrity() {
    let db = Database::open("memory://multi_col_integrity").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            category TEXT,
            brand TEXT,
            price FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    // Insert initial data
    db.execute(
        "INSERT INTO products VALUES
            (1, 'Electronics', 'Apple', 999.99),
            (2, 'Electronics', 'Samsung', 799.99),
            (3, 'Electronics', 'Apple', 1299.99),
            (4, 'Clothing', 'Nike', 129.99),
            (5, 'Clothing', 'Adidas', 99.99)",
        (),
    )
    .expect("Failed to insert data");

    // Create multi-column index
    db.execute(
        "CREATE INDEX idx_cat_brand ON products(category, brand)",
        (),
    )
    .expect("Failed to create multi-column index");

    // Verify initial query
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM products WHERE category = 'Electronics' AND brand = 'Apple'",
            (),
        )
        .expect("Failed to query");
    assert_eq!(count, 2, "Expected 2 Apple electronics");

    // Delete one row
    db.execute("DELETE FROM products WHERE id = 1", ())
        .expect("Failed to delete");

    // Verify after delete
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM products WHERE category = 'Electronics' AND brand = 'Apple'",
            (),
        )
        .expect("Failed to query after delete");
    assert_eq!(count, 1, "Expected 1 Apple electronics after delete");

    // Insert new row
    db.execute(
        "INSERT INTO products VALUES (6, 'Electronics', 'Apple', 499.99)",
        (),
    )
    .expect("Failed to insert");

    // Verify after insert
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM products WHERE category = 'Electronics' AND brand = 'Apple'",
            (),
        )
        .expect("Failed to query after insert");
    assert_eq!(count, 2, "Expected 2 Apple electronics after insert");
}

/// Test unique multi-column index with NULLs (NULLs should be allowed as duplicates)
#[test]
fn test_unique_multi_column_index_with_nulls() {
    let db = Database::open("memory://unique_multi_nulls").expect("Failed to create database");

    db.execute(
        "CREATE TABLE user_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            device_id INTEGER,
            session_token TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Create unique index on (user_id, device_id)
    db.execute(
        "CREATE UNIQUE INDEX idx_user_device ON user_sessions(user_id, device_id)",
        (),
    )
    .expect("Failed to create unique multi-column index");

    // Insert row with both columns having values
    db.execute("INSERT INTO user_sessions VALUES (1, 100, 1, 'token1')", ())
        .expect("Failed to insert");

    // Insert row with NULL in one column - should succeed
    db.execute(
        "INSERT INTO user_sessions VALUES (2, 100, NULL, 'token2')",
        (),
    )
    .expect("Should allow NULL in unique multi-column index");

    // Insert another row with NULL in same column - should succeed (NULLs are distinct)
    db.execute(
        "INSERT INTO user_sessions VALUES (3, 100, NULL, 'token3')",
        (),
    )
    .expect("Multiple NULLs should be allowed in unique index");

    // Insert row with NULL in the other column
    db.execute(
        "INSERT INTO user_sessions VALUES (4, NULL, 1, 'token4')",
        (),
    )
    .expect("NULL in first column should be allowed");

    // Try to insert duplicate non-NULL combination - should fail
    let result = db.execute("INSERT INTO user_sessions VALUES (5, 100, 1, 'token5')", ());
    assert!(
        result.is_err(),
        "Duplicate non-NULL combination should fail"
    );

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM user_sessions", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 sessions");
}

/// Test unique multi-column index with UPDATE operations
#[test]
fn test_unique_multi_column_index_update() {
    let db = Database::open("memory://unique_multi_update").expect("Failed to create database");

    db.execute(
        "CREATE TABLE schedules (
            id INTEGER PRIMARY KEY,
            room_id INTEGER,
            time_slot INTEGER,
            event_name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Create unique index
    db.execute(
        "CREATE UNIQUE INDEX idx_room_time ON schedules(room_id, time_slot)",
        (),
    )
    .expect("Failed to create unique index");

    // Insert initial data
    db.execute("INSERT INTO schedules VALUES (1, 101, 1, 'Meeting A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO schedules VALUES (2, 101, 2, 'Meeting B')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO schedules VALUES (3, 102, 1, 'Meeting C')", ())
        .expect("Failed to insert");

    // Update to a new unique combination - should succeed
    db.execute("UPDATE schedules SET time_slot = 3 WHERE id = 1", ())
        .expect("Update to new combination should succeed");

    // Verify the update
    let time: i64 = db
        .query_one("SELECT time_slot FROM schedules WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(time, 3, "time_slot should be updated to 3");

    // Update to conflict with existing - should fail
    let result = db.execute("UPDATE schedules SET time_slot = 2 WHERE id = 1", ());
    assert!(result.is_err(), "Update causing duplicate should fail");

    // The original value should be preserved
    let time: i64 = db
        .query_one("SELECT time_slot FROM schedules WHERE id = 1", ())
        .expect("Failed to query");
    assert_eq!(time, 3, "time_slot should still be 3 after failed update");
}

/// Test unique multi-column index enforcement during bulk insert
#[test]
fn test_unique_multi_column_index_bulk_insert() {
    let db = Database::open("memory://unique_multi_bulk").expect("Failed to create database");

    db.execute(
        "CREATE TABLE inventory (
            id INTEGER PRIMARY KEY,
            warehouse_id INTEGER,
            product_id INTEGER,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Create unique index
    db.execute(
        "CREATE UNIQUE INDEX idx_warehouse_product ON inventory(warehouse_id, product_id)",
        (),
    )
    .expect("Failed to create unique index");

    // Bulk insert with all unique combinations
    db.execute(
        "INSERT INTO inventory VALUES
            (1, 1, 100, 10),
            (2, 1, 101, 20),
            (3, 2, 100, 30),
            (4, 2, 101, 40)",
        (),
    )
    .expect("Bulk insert with unique combinations should succeed");

    // Try bulk insert with duplicate in the batch
    let result = db.execute(
        "INSERT INTO inventory VALUES
            (5, 3, 100, 50),
            (6, 3, 100, 60)",
        (),
    );
    assert!(
        result.is_err(),
        "Bulk insert with duplicate in batch should fail"
    );

    // Try bulk insert conflicting with existing data
    let result = db.execute("INSERT INTO inventory VALUES (7, 1, 100, 70)", ());
    assert!(
        result.is_err(),
        "Insert conflicting with existing should fail"
    );

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM inventory", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected only 4 rows from first bulk insert");
}

/// Test unique multi-column index with 3 columns
#[test]
fn test_unique_multi_column_index_three_columns() {
    let db = Database::open("memory://unique_multi_three").expect("Failed to create database");

    db.execute(
        "CREATE TABLE appointments (
            id INTEGER PRIMARY KEY,
            doctor_id INTEGER,
            patient_id INTEGER,
            appointment_date TEXT,
            notes TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    // Create unique index on 3 columns
    db.execute(
        "CREATE UNIQUE INDEX idx_appointment ON appointments(doctor_id, patient_id, appointment_date)",
        (),
    )
    .expect("Failed to create unique 3-column index");

    // Insert valid data
    db.execute(
        "INSERT INTO appointments VALUES (1, 1, 100, '2025-01-01', 'Checkup')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO appointments VALUES (2, 1, 100, '2025-01-02', 'Follow-up')",
        (),
    )
    .expect("Same doctor+patient, different date should succeed");
    db.execute(
        "INSERT INTO appointments VALUES (3, 1, 101, '2025-01-01', 'New patient')",
        (),
    )
    .expect("Same doctor+date, different patient should succeed");
    db.execute(
        "INSERT INTO appointments VALUES (4, 2, 100, '2025-01-01', 'Second opinion')",
        (),
    )
    .expect("Different doctor, same patient+date should succeed");

    // Try to insert duplicate combination
    let result = db.execute(
        "INSERT INTO appointments VALUES (5, 1, 100, '2025-01-01', 'Duplicate')",
        (),
    );
    assert!(
        result.is_err(),
        "Duplicate 3-column combination should fail"
    );

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM appointments", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 appointments");
}

/// Test that multi-column indexes are used for compound WHERE clauses
#[test]
fn test_multi_column_index_query_optimization() {
    let db = Database::open_in_memory().expect("Failed to open in-memory database");

    // Create table with reasonable amount of data
    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT NOT NULL,
            product TEXT NOT NULL,
            year INTEGER NOT NULL,
            amount DECIMAL(10,2)
        )",
        (),
    )
    .expect("Failed to create table");

    // Create multi-column index on (region, product, year)
    db.execute(
        "CREATE INDEX idx_sales_region_product_year ON sales(region, product, year)",
        (),
    )
    .expect("Failed to create index");

    // Insert test data
    let regions = ["North", "South", "East", "West"];
    let products = ["Widget", "Gadget", "Gizmo"];
    let years = [2020, 2021, 2022, 2023];

    let mut id = 1;
    for region in &regions {
        for product in &products {
            for year in &years {
                db.execute(
                    "INSERT INTO sales (id, region, product, year, amount) VALUES (?, ?, ?, ?, ?)",
                    (id, *region, *product, *year, (id * 100) as f64),
                )
                .expect("Failed to insert");
                id += 1;
            }
        }
    }

    // Query using all 3 columns of the multi-column index
    let amount: f64 = db
        .query_one(
            "SELECT amount FROM sales WHERE region = 'North' AND product = 'Widget' AND year = 2022",
            (),
        )
        .expect("Failed to query");
    assert_eq!(amount, 300.0); // Row 3 (North, Widget, 2022)

    // Query using first 2 columns (prefix match)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM sales WHERE region = 'South' AND product = 'Gadget'",
            (),
        )
        .expect("Failed to query prefix");
    assert_eq!(count, 4); // 4 years for South/Gadget

    // Query using first column only
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM sales WHERE region = 'East'", ())
        .expect("Failed to query single column");
    assert_eq!(count, 12); // 3 products * 4 years

    // Verify SHOW INDEXES shows multi-column index correctly
    // SHOW INDEXES columns: table_name, index_name, column_name, index_type, is_unique
    let results: Vec<(String, String, String)> = db
        .query("SHOW INDEXES FROM sales", ())
        .expect("Failed to show indexes")
        .map(|row| {
            row.map(|r| {
                (
                    r.get::<String>(1).unwrap_or_default(), // index_name
                    r.get::<String>(2).unwrap_or_default(), // column_name
                    r.get::<String>(3).unwrap_or_default(), // index_type
                )
            })
            .unwrap()
        })
        .collect();

    // Should have one index entry showing all columns
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "idx_sales_region_product_year");
    assert!(results[0].1.contains("region"));
    assert!(results[0].1.contains("product"));
    assert!(results[0].1.contains("year"));
    assert_eq!(results[0].2, "MULTICOLUMN");
}
