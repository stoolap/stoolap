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

//! COUNT with Index Tests
//!
//! Tests COUNT(*) with WHERE clause using indexes

use stoolap::Database;

/// Test COUNT(*) with WHERE clause and index
#[test]
fn test_count_with_where_clause_and_index() {
    let db = Database::open("memory://count_idx").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test_count (
            id INTEGER PRIMARY KEY,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Create index on boolean column
    db.execute("CREATE INDEX idx_active ON test_count (active)", ())
        .expect("Failed to create index");

    // Insert records - half true, half false
    let num_true = 500;
    let num_total = 1000;

    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    for i in 1..=num_total {
        let is_active = i <= num_true;
        db.execute(
            "INSERT INTO test_count (id, active) VALUES (?, ?)",
            (i, is_active),
        )
        .expect("Failed to insert record");
    }

    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Test 1: Count all records
    let total_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count", ())
        .expect("Failed to count all records");
    assert_eq!(
        total_count, num_total,
        "Expected {} total records, got {}",
        num_total, total_count
    );

    // Test 2: Count active records
    let active_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count WHERE active = true", ())
        .expect("Failed to count active records");
    assert_eq!(
        active_count, num_true,
        "Expected {} active records, got {}",
        num_true, active_count
    );

    // Test 3: Count inactive records
    let inactive_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_count WHERE active = false", ())
        .expect("Failed to count inactive records");
    assert_eq!(
        inactive_count,
        num_total - num_true,
        "Expected {} inactive records, got {}",
        num_total - num_true,
        inactive_count
    );

    // Test 4: Verify counts add up
    assert_eq!(
        active_count + inactive_count,
        total_count,
        "Sum of active and inactive doesn't equal total"
    );
}

/// Test COUNT with WHERE clause - multiple query consistency
#[test]
fn test_count_with_index_consistency() {
    let db = Database::open("memory://count_consistency").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test_multiple (
            id INTEGER PRIMARY KEY,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Create index
    db.execute("CREATE INDEX idx_active ON test_multiple (active)", ())
        .expect("Failed to create index");

    // Insert records with alternating values
    let num_records = 1000;
    let expected_true = num_records / 2;

    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    for i in 1..=num_records {
        let is_active = i % 2 == 0; // Even IDs are active
        db.execute(
            "INSERT INTO test_multiple (id, active) VALUES (?, ?)",
            (i, is_active),
        )
        .expect("Failed to insert record");
    }

    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Run COUNT queries multiple times to verify consistency
    for run in 0..5 {
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM test_multiple", ())
            .expect("Failed to count records");
        assert_eq!(
            count, num_records,
            "Run {}: Expected {} total records, got {}",
            run, num_records, count
        );
    }

    // Test active counts
    for run in 0..5 {
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM test_multiple WHERE active = true", ())
            .expect("Failed to count active records");
        assert_eq!(
            count, expected_true,
            "Run {}: Expected {} active records, got {}",
            run, expected_true, count
        );
    }

    // Test inactive counts
    for run in 0..5 {
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM test_multiple WHERE active = false",
                (),
            )
            .expect("Failed to count inactive records");
        assert_eq!(
            count,
            num_records - expected_true,
            "Run {}: Expected {} inactive records, got {}",
            run,
            num_records - expected_true,
            count
        );
    }
}

/// Test COUNT after data modifications
#[test]
fn test_count_after_modifications() {
    let db = Database::open("memory://count_modify").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test_bulk (
            id INTEGER PRIMARY KEY,
            active BOOLEAN
        )",
        (),
    )
    .expect("Failed to create table");

    // Create index
    db.execute("CREATE INDEX idx_active ON test_bulk (active)", ())
        .expect("Failed to create index");

    // Insert initial records - all active=true
    let num_records: i64 = 1000;

    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    for i in 1..=num_records {
        db.execute(
            "INSERT INTO test_bulk (id, active) VALUES (?, ?)",
            (i, true),
        )
        .expect("Failed to insert record");
    }

    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Verify initial state
    let initial_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk WHERE active = true", ())
        .expect("Failed to count active records");
    assert_eq!(
        initial_count, num_records,
        "Initial: Expected {} active records, got {}",
        num_records, initial_count
    );

    // Update half the records to active=false
    db.execute(
        "UPDATE test_bulk SET active = ? WHERE id <= ?",
        (false, num_records / 2),
    )
    .expect("Failed to update records");

    // Verify counts after update
    let active_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk WHERE active = true", ())
        .expect("Failed to count active records after update");
    let inactive_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk WHERE active = false", ())
        .expect("Failed to count inactive records after update");

    assert_eq!(
        active_count,
        num_records / 2,
        "After update: Expected {} active records, got {}",
        num_records / 2,
        active_count
    );
    assert_eq!(
        inactive_count,
        num_records / 2,
        "After update: Expected {} inactive records, got {}",
        num_records / 2,
        inactive_count
    );

    // Delete some records
    db.execute(
        "DELETE FROM test_bulk WHERE active = ? AND id > ? AND id <= ?",
        (true, num_records / 2, num_records / 2 + num_records / 4),
    )
    .expect("Failed to delete records");

    // Verify final counts
    let final_total: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk", ())
        .expect("Failed to count total records after delete");
    let expected_total = num_records - num_records / 4;
    assert_eq!(
        final_total, expected_total,
        "Final: Expected {} total records, got {}",
        expected_total, final_total
    );

    let final_active: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk WHERE active = true", ())
        .expect("Failed to count active records after delete");
    let final_inactive: i64 = db
        .query_one("SELECT COUNT(*) FROM test_bulk WHERE active = false", ())
        .expect("Failed to count inactive records after delete");

    assert_eq!(
        final_active + final_inactive,
        final_total,
        "Sum of active and inactive doesn't equal total after delete"
    );
}

/// Test COUNT with complex filters and multiple indexes
#[test]
fn test_count_with_complex_filters() {
    let db = Database::open("memory://count_complex").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test_complex (
            id INTEGER PRIMARY KEY,
            active BOOLEAN,
            category INTEGER,
            value FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    // Create indexes
    db.execute("CREATE INDEX idx_active ON test_complex (active)", ())
        .expect("Failed to create active index");
    db.execute("CREATE INDEX idx_category ON test_complex (category)", ())
        .expect("Failed to create category index");

    // Insert test data
    let num_records = 2000;
    let mut expected_active = 0i64;
    let mut expected_cat_1 = 0i64;
    let mut expected_active_cat_1 = 0i64;
    let mut expected_value_gt_50 = 0i64;

    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    for i in 1..=num_records {
        let is_active = i % 3 != 0; // 2/3 of records are active
        let category = (i % 5) + 1; // Categories 1-5
        let value = ((i % 100) + 1) as f64; // Values 1-100

        if is_active {
            expected_active += 1;
        }
        if category == 1 {
            expected_cat_1 += 1;
            if is_active {
                expected_active_cat_1 += 1;
            }
        }
        if value > 50.0 {
            expected_value_gt_50 += 1;
        }

        db.execute(
            "INSERT INTO test_complex (id, active, category, value) VALUES (?, ?, ?, ?)",
            (i, is_active, category, value),
        )
        .expect("Failed to insert record");
    }

    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Test 1: Basic active filter
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_complex WHERE active = true", ())
        .expect("Failed to count active records");
    assert_eq!(
        count, expected_active,
        "Active count: Expected {}, got {}",
        expected_active, count
    );

    // Test 2: Category filter
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_complex WHERE category = 1", ())
        .expect("Failed to count category 1 records");
    assert_eq!(
        count, expected_cat_1,
        "Category 1 count: Expected {}, got {}",
        expected_cat_1, count
    );

    // Test 3: Combined filters (active AND category)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_complex WHERE active = true AND category = 1",
            (),
        )
        .expect("Failed to count active category 1 records");
    assert_eq!(
        count, expected_active_cat_1,
        "Active category 1 count: Expected {}, got {}",
        expected_active_cat_1, count
    );

    // Test 4: Range filter
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_complex WHERE value > 50", ())
        .expect("Failed to count records with value > 50");
    assert_eq!(
        count, expected_value_gt_50,
        "Value > 50 count: Expected {}, got {}",
        expected_value_gt_50, count
    );

    // Test 5: Total count verification
    let total_count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_complex", ())
        .expect("Failed to get total count");
    assert_eq!(
        total_count, num_records,
        "Total count: Expected {}, got {}",
        num_records, total_count
    );
}

/// Test COUNT with OR conditions
#[test]
fn test_count_with_or_conditions() {
    let db = Database::open("memory://count_or").expect("Failed to create database");

    // Create test table
    db.execute(
        "CREATE TABLE test_or (
            id INTEGER PRIMARY KEY,
            category TEXT,
            status INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    // Create indexes
    db.execute("CREATE INDEX idx_category ON test_or (category)", ())
        .expect("Failed to create category index");
    db.execute("CREATE INDEX idx_status ON test_or (status)", ())
        .expect("Failed to create status index");

    // Insert test data
    db.execute("INSERT INTO test_or VALUES (1, 'A', 1)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (2, 'A', 2)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (3, 'B', 1)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (4, 'B', 2)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (5, 'C', 1)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (6, 'C', 2)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (7, 'A', 3)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (8, 'B', 3)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (9, 'C', 3)", ())
        .unwrap();
    db.execute("INSERT INTO test_or VALUES (10, 'D', 1)", ())
        .unwrap();

    // Test COUNT with OR
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_or WHERE category = 'A' OR category = 'B'",
            (),
        )
        .expect("Failed to count with OR");
    assert_eq!(count, 6, "Expected 6 rows for category A or B");

    // Test COUNT with complex OR/AND
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_or WHERE (category = 'A' OR category = 'B') AND status = 1",
            (),
        )
        .expect("Failed to count with complex condition");
    assert_eq!(count, 2, "Expected 2 rows for (A or B) and status=1");

    // Test COUNT with IN (similar to OR)
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM test_or WHERE category IN ('A', 'B', 'C')",
            (),
        )
        .expect("Failed to count with IN");
    assert_eq!(count, 9, "Expected 9 rows for categories A, B, C");
}
