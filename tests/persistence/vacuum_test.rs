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

//! VACUUM command integration tests

use stoolap::Database;

/// Test basic VACUUM on empty database
#[test]
fn test_vacuum_empty_database() {
    let db = Database::open("memory://vacuum_empty").expect("Failed to create database");

    let result = db.query("VACUUM", ()).expect("VACUUM should succeed");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let deleted: i64 = row.get(0).unwrap();
        let versions: i64 = row.get(1).unwrap();
        let txns: i64 = row.get(2).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(versions, 0);
        assert_eq!(txns, 0);
        found = true;
    }
    assert!(found, "Expected one result row");
}

/// Test VACUUM cleans up deleted rows
#[test]
fn test_vacuum_cleans_deleted_rows() {
    let db = Database::open("memory://vacuum_deleted").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("CREATE TABLE failed");

    // Insert rows
    for i in 1..=100 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'row_{}')", i, i), ())
            .expect("INSERT failed");
    }

    // Delete half the rows
    db.execute("DELETE FROM t WHERE id <= 50", ())
        .expect("DELETE failed");

    // VACUUM should clean deleted rows
    let result = db.query("VACUUM", ()).expect("VACUUM failed");

    let mut deleted_cleaned = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        deleted_cleaned = row.get(0).unwrap();
    }
    assert!(
        deleted_cleaned > 0,
        "Expected VACUUM to clean deleted rows, got {}",
        deleted_cleaned
    );

    // Verify remaining data is intact
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("COUNT failed");
    assert_eq!(count, 50, "Should have 50 remaining rows");

    // Verify correct rows remain
    let min_id: i64 = db
        .query_one("SELECT MIN(id) FROM t", ())
        .expect("MIN failed");
    assert_eq!(min_id, 51, "Smallest remaining id should be 51");
}

/// Test VACUUM on a specific table
#[test]
fn test_vacuum_specific_table() {
    let db = Database::open("memory://vacuum_table").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .expect("CREATE t1 failed");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .expect("CREATE t2 failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({})", i), ())
            .expect("INSERT t1 failed");
        db.execute(&format!("INSERT INTO t2 VALUES ({})", i), ())
            .expect("INSERT t2 failed");
    }

    db.execute("DELETE FROM t1 WHERE id <= 5", ())
        .expect("DELETE t1 failed");
    db.execute("DELETE FROM t2 WHERE id <= 5", ())
        .expect("DELETE t2 failed");

    // Vacuum only t1
    let result = db.query("VACUUM t1", ()).expect("VACUUM t1 failed");

    let mut deleted_cleaned = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        deleted_cleaned = row.get(0).unwrap();
    }
    assert!(
        deleted_cleaned > 0,
        "Expected VACUUM to clean deleted rows from t1"
    );

    // Both tables should still have correct data
    let c1: i64 = db
        .query_one("SELECT COUNT(*) FROM t1", ())
        .expect("COUNT t1 failed");
    let c2: i64 = db
        .query_one("SELECT COUNT(*) FROM t2", ())
        .expect("COUNT t2 failed");
    assert_eq!(c1, 5);
    assert_eq!(c2, 5);
}

/// Test VACUUM on nonexistent table returns error
#[test]
fn test_vacuum_nonexistent_table() {
    let db = Database::open("memory://vacuum_noexist").expect("Failed to create database");

    let result = db.execute("VACUUM nonexistent_table", ());
    assert!(result.is_err(), "VACUUM on nonexistent table should fail");
}

/// Test PRAGMA VACUUM works the same as VACUUM
#[test]
fn test_pragma_vacuum() {
    let db = Database::open("memory://pragma_vacuum").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("CREATE TABLE failed");

    for i in 1..=50 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'row_{}')", i, i), ())
            .expect("INSERT failed");
    }

    db.execute("DELETE FROM t WHERE id <= 25", ())
        .expect("DELETE failed");

    // Use PRAGMA syntax
    let result = db.query("PRAGMA vacuum", ()).expect("PRAGMA vacuum failed");

    let mut deleted_cleaned = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        deleted_cleaned = row.get(0).unwrap();
    }
    assert!(
        deleted_cleaned > 0,
        "Expected PRAGMA vacuum to clean deleted rows"
    );

    // Verify data intact
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("COUNT failed");
    assert_eq!(count, 25);
}

/// Test VACUUM after updates preserves data correctness
#[test]
fn test_vacuum_after_updates() {
    let db = Database::open("memory://vacuum_updates").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("CREATE TABLE failed");

    for i in 1..=20 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .expect("INSERT failed");
    }

    // Update rows multiple times to create old versions
    for _ in 0..5 {
        db.execute("UPDATE t SET val = val + 1", ())
            .expect("UPDATE failed");
    }

    let result = db.query("VACUUM", ()).expect("VACUUM failed");

    let mut versions_cleaned = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        versions_cleaned = row.get(1).unwrap();
    }
    assert!(
        versions_cleaned > 0,
        "Expected VACUUM to clean old versions after updates, got {}",
        versions_cleaned
    );

    // Verify data is correct after vacuum
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("COUNT failed");
    assert_eq!(count, 20);

    let sum: i64 = db
        .query_one("SELECT SUM(val) FROM t", ())
        .expect("SUM failed");
    // Each row started as i, updated 5 times (+1 each) = i + 5
    // Sum of (1+5) + (2+5) + ... + (20+5) = Sum(1..20) + 20*5 = 210 + 100 = 310
    assert_eq!(sum, 310);
}

/// Test multiple VACUUM calls are safe
#[test]
fn test_vacuum_idempotent() {
    let db = Database::open("memory://vacuum_idempotent").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .expect("CREATE TABLE failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t VALUES ({})", i), ())
            .expect("INSERT failed");
    }

    db.execute("DELETE FROM t WHERE id <= 5", ())
        .expect("DELETE failed");

    // First vacuum cleans up
    db.execute("VACUUM", ()).expect("First VACUUM failed");

    // Second vacuum should succeed with nothing to clean
    let result = db.query("VACUUM", ()).expect("Second VACUUM failed");

    let mut deleted_cleaned = 0i64;
    for row in result {
        let row = row.expect("Failed to get row");
        deleted_cleaned = row.get(0).unwrap();
    }
    assert_eq!(
        deleted_cleaned, 0,
        "Second VACUUM should find nothing to clean"
    );

    // Data still intact
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("COUNT failed");
    assert_eq!(count, 5);
}

/// Test VACUUM works correctly with INSERT after DELETE
#[test]
fn test_vacuum_insert_after_delete() {
    let db = Database::open("memory://vacuum_reinsert").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("CREATE TABLE failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO t VALUES ({}, 'original_{}')", i, i),
            (),
        )
        .expect("INSERT failed");
    }

    // Delete and vacuum
    db.execute("DELETE FROM t WHERE id <= 5", ())
        .expect("DELETE failed");
    db.execute("VACUUM", ()).expect("VACUUM failed");

    // Insert new rows (including reusing deleted PKs)
    for i in 1..=5 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'new_{}')", i, i), ())
            .expect("Re-INSERT failed");
    }

    // Verify all 10 rows exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("COUNT failed");
    assert_eq!(count, 10);

    // Verify new rows have correct values
    let val: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("SELECT failed");
    assert_eq!(val, "new_1");
}

/// Test VACUUM fails inside an explicit transaction
#[test]
fn test_vacuum_rejects_inside_transaction() {
    let db = Database::open("memory://vacuum_tx_reject").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .expect("CREATE TABLE failed");

    db.execute("BEGIN", ()).expect("BEGIN failed");

    let result = db.execute("VACUUM", ());
    assert!(
        result.is_err(),
        "VACUUM should fail inside an active transaction"
    );

    // Rollback to clean up
    db.execute("ROLLBACK", ()).expect("ROLLBACK failed");

    // VACUUM should work outside a transaction
    db.execute("VACUUM", ())
        .expect("VACUUM should succeed outside transaction");
}

/// Test PRAGMA VACUUM fails inside an explicit transaction
#[test]
fn test_pragma_vacuum_rejects_inside_transaction() {
    let db = Database::open("memory://pragma_vacuum_tx").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .expect("CREATE TABLE failed");

    db.execute("BEGIN", ()).expect("BEGIN failed");

    let result = db.execute("PRAGMA vacuum", ());
    assert!(
        result.is_err(),
        "PRAGMA VACUUM should fail inside an active transaction"
    );

    db.execute("ROLLBACK", ()).expect("ROLLBACK failed");

    db.execute("PRAGMA vacuum", ())
        .expect("PRAGMA VACUUM should succeed outside transaction");
}
