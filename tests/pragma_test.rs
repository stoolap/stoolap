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

//! PRAGMA Statement Tests
//!
//! Tests PRAGMA statement functionality for database configuration

use stoolap::Database;

/// Test reading snapshot_interval PRAGMA
#[test]
fn test_pragma_read_snapshot_interval() {
    let db = Database::open("memory://pragma_read").expect("Failed to create database");

    // Read default snapshot_interval
    let result = db
        .query("PRAGMA snapshot_interval", ())
        .expect("Failed to execute PRAGMA");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let _value: i64 = row.get(0).unwrap();
        // Just verify we can read it, actual value may vary
        found = true;
    }
    assert!(found, "Expected at least one row in result");
}

/// Test setting snapshot_interval PRAGMA
#[test]
fn test_pragma_set_snapshot_interval() {
    let db = Database::open("memory://pragma_set_snap").expect("Failed to create database");

    // Set snapshot_interval
    let result = db
        .query("PRAGMA snapshot_interval = 60", ())
        .expect("Failed to set PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 60, "Expected snapshot_interval to be 60");
    }

    // Verify the value was set
    let result = db
        .query("PRAGMA snapshot_interval", ())
        .expect("Failed to read PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(
            value, 60,
            "Expected snapshot_interval to be 60 after setting"
        );
    }
}

/// Test setting keep_snapshots PRAGMA
#[test]
fn test_pragma_set_keep_snapshots() {
    let db = Database::open("memory://pragma_keep").expect("Failed to create database");

    // Set keep_snapshots
    let result = db
        .query("PRAGMA keep_snapshots = 10", ())
        .expect("Failed to set PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 10, "Expected keep_snapshots to be 10");
    }

    // Verify the value was set
    let result = db
        .query("PRAGMA keep_snapshots", ())
        .expect("Failed to read PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 10, "Expected keep_snapshots to be 10 after setting");
    }
}

/// Test setting sync_mode PRAGMA
#[test]
fn test_pragma_set_sync_mode() {
    let db = Database::open("memory://pragma_sync").expect("Failed to create database");

    // Set sync_mode
    let result = db
        .query("PRAGMA sync_mode = 2", ())
        .expect("Failed to set PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 2, "Expected sync_mode to be 2");
    }

    // Verify the value was set
    let result = db
        .query("PRAGMA sync_mode", ())
        .expect("Failed to read PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 2, "Expected sync_mode to be 2 after setting");
    }
}

/// Test setting wal_flush_trigger PRAGMA
#[test]
fn test_pragma_set_wal_flush_trigger() {
    let db = Database::open("memory://pragma_wal").expect("Failed to create database");

    // Set wal_flush_trigger
    let result = db
        .query("PRAGMA wal_flush_trigger = 1000", ())
        .expect("Failed to set PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(value, 1000, "Expected wal_flush_trigger to be 1000");
    }

    // Verify the value was set
    let result = db
        .query("PRAGMA wal_flush_trigger", ())
        .expect("Failed to read PRAGMA");

    for row in result {
        let row = row.expect("Failed to get row");
        let value: i64 = row.get(0).unwrap();
        assert_eq!(
            value, 1000,
            "Expected wal_flush_trigger to be 1000 after setting"
        );
    }
}

/// Test PRAGMA interaction with transactions
#[test]
fn test_pragma_with_transactions() {
    let db = Database::open("memory://pragma_tx").expect("Failed to create database");

    // Set a baseline value
    db.execute("PRAGMA snapshot_interval = 60", ())
        .expect("Failed to set baseline snapshot_interval");

    // Verify baseline
    let initial: i64 = db
        .query_one("PRAGMA snapshot_interval", ())
        .expect("Failed to get initial snapshot_interval");
    assert_eq!(initial, 60, "Expected initial snapshot_interval to be 60");

    // Start a transaction and do regular SQL
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table in transaction");
    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Change PRAGMA outside of transaction
    db.execute("PRAGMA snapshot_interval = 90", ())
        .expect("Failed to set snapshot_interval");

    // Verify change
    let new_value: i64 = db
        .query_one("PRAGMA snapshot_interval", ())
        .expect("Failed to read snapshot_interval after setting");
    assert_eq!(new_value, 90, "Expected snapshot_interval to be 90");

    // Test PRAGMA after transaction
    db.execute("PRAGMA snapshot_interval = 120", ())
        .expect("Failed to set snapshot_interval after transaction");

    let final_value: i64 = db
        .query_one("PRAGMA snapshot_interval", ())
        .expect("Failed to read final snapshot_interval");
    assert_eq!(
        final_value, 120,
        "Expected final snapshot_interval to be 120"
    );
}

/// Test multiple PRAGMA settings in sequence
#[test]
fn test_pragma_multiple_settings() {
    let db = Database::open("memory://pragma_multi").expect("Failed to create database");

    // Set multiple PRAGMA values
    db.execute("PRAGMA snapshot_interval = 45", ())
        .expect("Failed to set snapshot_interval");
    db.execute("PRAGMA keep_snapshots = 5", ())
        .expect("Failed to set keep_snapshots");
    db.execute("PRAGMA sync_mode = 1", ())
        .expect("Failed to set sync_mode");
    db.execute("PRAGMA wal_flush_trigger = 500", ())
        .expect("Failed to set wal_flush_trigger");

    // Verify all values
    let snap_interval: i64 = db
        .query_one("PRAGMA snapshot_interval", ())
        .expect("Failed to read snapshot_interval");
    assert_eq!(snap_interval, 45);

    let keep_snap: i64 = db
        .query_one("PRAGMA keep_snapshots", ())
        .expect("Failed to read keep_snapshots");
    assert_eq!(keep_snap, 5);

    let sync_mode: i64 = db
        .query_one("PRAGMA sync_mode", ())
        .expect("Failed to read sync_mode");
    assert_eq!(sync_mode, 1);

    let wal_trigger: i64 = db
        .query_one("PRAGMA wal_flush_trigger", ())
        .expect("Failed to read wal_flush_trigger");
    assert_eq!(wal_trigger, 500);
}
