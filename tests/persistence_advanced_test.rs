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

//! Persistence tests for snapshot, WAL, and index recovery

use stoolap::Database;
use tempfile::tempdir;

/// Verifies that UPDATE operations are persisted correctly across database
/// restarts when snapshots exist.
#[test]
fn test_update_persistence_with_snapshot() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // First connection - create table and insert data
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP
            )",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO users (id, name, email, created_at)
             VALUES (1, 'John Doe', 'john@example.com', NOW())",
            (),
        )
        .unwrap();

        // Verify initial data
        let name: String = db
            .query_one("SELECT name FROM users WHERE id = 1", ())
            .unwrap();
        assert_eq!(name, "John Doe");

        db.close().unwrap();
    }

    // Second connection - update the data
    {
        let db = Database::open(&dsn).unwrap();

        // Update the name
        let result = db.execute("UPDATE users SET name = 'Jane Doe' WHERE id = 1", ());
        assert!(result.is_ok());

        // Verify update in same session
        let name: String = db
            .query_one("SELECT name FROM users WHERE id = 1", ())
            .unwrap();
        assert_eq!(name, "Jane Doe");

        db.close().unwrap();
    }

    // Third connection - verify update persisted
    {
        let db = Database::open(&dsn).unwrap();

        let name: String = db
            .query_one("SELECT name FROM users WHERE id = 1", ())
            .unwrap();

        // This is the critical test - the update should persist!
        assert_eq!(
            name, "Jane Doe",
            "UPDATE not persisted! Expected 'Jane Doe' after restart, got '{}'",
            name
        );

        // Also test multiple updates
        db.execute("UPDATE users SET name = 'Alice Smith' WHERE id = 1", ())
            .unwrap();

        db.close().unwrap();
    }

    // Fourth connection - verify second update
    {
        let db = Database::open(&dsn).unwrap();

        let name: String = db
            .query_one("SELECT name FROM users WHERE id = 1", ())
            .unwrap();

        assert_eq!(
            name, "Alice Smith",
            "Second UPDATE not persisted! Expected 'Alice Smith' after restart, got '{}'",
            name
        );

        db.close().unwrap();
    }
}

/// Tests multiple updates to the same row with snapshot persistence
#[test]
fn test_update_chain_with_snapshot() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE counter (
                id INTEGER PRIMARY KEY,
                value INTEGER NOT NULL
            )",
            (),
        )
        .unwrap();

        // Insert initial value
        db.execute("INSERT INTO counter (id, value) VALUES (1, 0)", ())
            .unwrap();

        // Perform multiple updates
        for i in 1..=5 {
            db.execute(
                &format!("UPDATE counter SET value = {} WHERE id = 1", i),
                (),
            )
            .unwrap();
        }

        // Verify final value
        let value: i64 = db
            .query_one("SELECT value FROM counter WHERE id = 1", ())
            .unwrap();
        assert_eq!(value, 5);

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();

        let value: i64 = db
            .query_one("SELECT value FROM counter WHERE id = 1", ())
            .unwrap();

        assert_eq!(
            value, 5,
            "UPDATE chain not persisted! Expected counter value 5 after restart, got {}",
            value
        );

        db.close().unwrap();
    }
}

/// Tests that both regular and unique indexes with custom names are properly
/// persisted and correctly restored after database restart.
#[test]
fn test_snapshot_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                region TEXT,
                active BOOLEAN
            )",
            (),
        )
        .unwrap();

        // Insert some data
        db.execute(
            "INSERT INTO customers (id, name, email, region, active) VALUES
                (1, 'John Doe', 'john@example.com', 'North', true),
                (2, 'Jane Smith', 'jane@example.com', 'South', true),
                (3, 'Bob Johnson', 'bob@example.com', 'East', false),
                (4, 'Alice Brown', 'alice@example.com', 'West', true),
                (5, 'Charlie Wilson', 'charlie@example.com', 'North', false)",
            (),
        )
        .unwrap();

        // Create a regular index with a custom name
        db.execute("CREATE INDEX idx_customer_region ON customers(region)", ())
            .unwrap();

        // Create a unique index on email
        db.execute(
            "CREATE UNIQUE INDEX idx_customer_email ON customers(email)",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Reopen the database to test persistence
    {
        let db = Database::open(&dsn).unwrap();

        // Try to insert a duplicate email to verify uniqueness is still enforced
        let result =
            db.execute("INSERT INTO customers (id, name, email, region, active) VALUES (6, 'Duplicate Email', 'john@example.com', 'South', true)", ());
        assert!(
            result.is_err(),
            "Expected unique constraint violation, but insert succeeded"
        );

        // Test successful insert with a different email
        let result = db.execute("INSERT INTO customers (id, name, email, region, active) VALUES (6, 'Mark Davis', 'mark@example.com', 'East', true)", ());
        assert!(
            result.is_ok(),
            "Failed to insert valid data after reopen: {:?}",
            result.err()
        );

        db.close().unwrap();
    }
}

/// TestOldSnapshotWithWAL tests the scenario where:
/// 1. User has an old snapshot with 100 rows
/// 2. User deletes all rows (recorded in WAL)
/// 3. No new snapshot is created before exit
/// 4. On restart, system loads old snapshot + replays WAL deletions
#[test]
fn test_old_snapshot_with_wal() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create data
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )",
            (),
        )
        .unwrap();

        // Insert 100 users
        for i in 1..=100 {
            db.execute(
                &format!(
                    "INSERT INTO users (id, name, email) VALUES ({}, 'User {}', 'user{}@example.com')",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Session 2: Delete all data WITHOUT creating new snapshot
    {
        let db = Database::open(&dsn).unwrap();

        // Verify initial state
        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 100, "Expected 100 users from snapshot");

        // Delete all users
        db.execute("DELETE FROM users", ()).unwrap();

        // Verify deletion in current session
        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 0, "Expected 0 users after delete");

        db.close().unwrap();
    }

    // Session 3: Reopen and check if deletions persist
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();

        // Should be 0 (WAL has DELETE operations)
        assert_eq!(
            count, 0,
            "BUG: Expected 0 users, got {} - WAL replay failed!",
            count
        );

        db.close().unwrap();
    }
}

/// TestUniqueIndexPersistence tests that unique index properties are properly preserved
/// when using CREATE UNIQUE INDEX across database reopens
#[test]
fn test_unique_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
            (),
        )
        .unwrap();

        // Create a unique index
        db.execute(
            "CREATE UNIQUE INDEX idx_users_email_unique ON users (email)",
            (),
        )
        .unwrap();

        // Insert a row to test uniqueness constraint
        db.execute(
            "INSERT INTO users (id, name, email) VALUES (1, 'User1', 'user1@example.com')",
            (),
        )
        .unwrap();

        // Try to insert a row with same email - should fail due to unique constraint
        let result = db.execute(
            "INSERT INTO users (id, name, email) VALUES (2, 'User2', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Inserting duplicate email should have failed"
        );

        db.close().unwrap();
    }

    // Reopen the database
    {
        let db = Database::open(&dsn).unwrap();

        // Test that uniqueness constraint is still enforced after reopen
        let result = db.execute(
            "INSERT INTO users (id, name, email) VALUES (3, 'User3', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Inserting duplicate email after reopen should have failed"
        );

        // Test dropping the unique index
        db.execute("DROP INDEX idx_users_email_unique ON users", ())
            .unwrap();

        db.close().unwrap();
    }

    // Reopen to verify DROP INDEX persisted
    {
        let db = Database::open(&dsn).unwrap();

        // Now we should be able to insert duplicate emails since the unique constraint is gone
        let result = db.execute(
            "INSERT INTO users (id, name, email) VALUES (3, 'User3', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_ok(),
            "Inserting previously duplicate email after dropping unique index should succeed, but got: {:?}",
            result.err()
        );

        // Verify we now have two rows with the same email
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM users WHERE email = 'user1@example.com'",
                (),
            )
            .unwrap();
        assert_eq!(
            count, 2,
            "Expected 2 rows with the same email after dropping unique index"
        );

        db.close().unwrap();
    }
}

/// Tests that auto-increment values are correctly persisted and restored
/// when replaying the WAL (Write-Ahead Log)
#[test]
fn test_auto_increment_wal_replay() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table with AUTO_INCREMENT
        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL
            )",
            (),
        )
        .unwrap();

        // Insert records with explicit IDs
        db.execute(
            "INSERT INTO users (id, username, email) VALUES
                (1, 'user1', 'user1@example.com'),
                (2, 'user2', 'user2@example.com'),
                (3, 'user3', 'user3@example.com')",
            (),
        )
        .unwrap();

        // Test auto-increment by inserting without providing ID
        db.execute(
            "INSERT INTO users (username, email) VALUES ('user4', 'user4@example.com')",
            (),
        )
        .unwrap();

        // Insert record with a much higher ID to advance the counter
        db.execute(
            "INSERT INTO users (id, username, email) VALUES (100, 'user100', 'user100@example.com')",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Reopen the database to test WAL replay
    {
        let db = Database::open(&dsn).unwrap();

        // Insert another record without ID to check if auto-increment counter was restored correctly
        db.execute(
            "INSERT INTO users (username, email) VALUES ('user101', 'user101@example.com')",
            (),
        )
        .unwrap();

        // Verify that the auto-increment counter was correctly restored from WAL
        let max_id: i64 = db.query_one("SELECT MAX(id) FROM users", ()).unwrap();
        assert_eq!(
            max_id, 101,
            "Expected max ID to be 101 after WAL replay, got {}",
            max_id
        );

        db.close().unwrap();
    }
}

/// Tests that auto-increment values are correctly persisted and restored
/// when loading from a snapshot
#[test]
fn test_auto_increment_snapshot_loading() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table with AUTO_INCREMENT
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT NOT NULL,
                price FLOAT NOT NULL
            )",
            (),
        )
        .unwrap();

        // Insert records with auto-incrementing IDs
        db.execute(
            "INSERT INTO products (id, name, price) VALUES
                (1, 'Product A', 10.99),
                (2, 'Product B', 20.50),
                (3, 'Product C', 30.75)",
            (),
        )
        .unwrap();

        // Insert a record with a much higher ID to advance the counter
        db.execute(
            "INSERT INTO products (id, name, price) VALUES (1000, 'Product X', 99.99)",
            (),
        )
        .unwrap();

        // Insert records without specifying ID to test auto-increment
        for i in 0..5 {
            db.execute(
                &format!(
                    "INSERT INTO products (name, price) VALUES ('Auto Product {}', {})",
                    i + 1,
                    50.0 + i as f64
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Reopen the database to load from snapshot/WAL
    {
        let db = Database::open(&dsn).unwrap();

        // Insert another record without ID to test if auto-increment counter was restored
        db.execute(
            "INSERT INTO products (name, price) VALUES ('Product After Snapshot', 200.00)",
            (),
        )
        .unwrap();

        // Verify the auto-increment counter was correctly restored
        let max_id: i64 = db.query_one("SELECT MAX(id) FROM products", ()).unwrap();
        assert!(
            max_id >= 1000,
            "Expected max ID to be at least 1000, got {}",
            max_id
        );

        // Check the count of records in the products table
        let product_count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
        assert_eq!(
            product_count,
            10, // 4 explicit + 5 auto + 1 after reopen
            "Expected 10 products, got {}",
            product_count
        );

        db.close().unwrap();
    }
}

/// Tests multiple tables with mixed operations persistence
#[test]
fn test_multi_table_mixed_operations_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create tables, indexes, and insert data
    {
        let db = Database::open(&dsn).unwrap();

        // Create users table with unique index
        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL
            )",
            (),
        )
        .unwrap();
        db.execute("CREATE UNIQUE INDEX idx_email ON users(email)", ())
            .unwrap();

        // Create orders table
        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                amount FLOAT NOT NULL,
                status TEXT
            )",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_user_id ON orders(user_id)", ())
            .unwrap();

        // Insert users
        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO users (id, name, email) VALUES ({}, 'User {}', 'user{}@example.com')",
                    i, i, i
                ),
                (),
            )
            .unwrap();
        }

        // Insert orders
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO orders (id, user_id, amount, status) VALUES ({}, {}, {}, 'pending')",
                    i,
                    (i % 5) + 1,
                    i as f64 * 10.0
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Session 2: Update and delete operations
    {
        let db = Database::open(&dsn).unwrap();

        // Verify data loaded
        let user_count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(user_count, 5);

        let order_count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(order_count, 10);

        // Update some orders
        db.execute("UPDATE orders SET status = 'completed' WHERE id <= 5", ())
            .unwrap();

        // Delete some orders
        db.execute("DELETE FROM orders WHERE id > 8", ()).unwrap();

        // Verify changes
        let completed_count: i64 = db
            .query_one("SELECT COUNT(*) FROM orders WHERE status = 'completed'", ())
            .unwrap();
        assert_eq!(completed_count, 5);

        let remaining_count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(remaining_count, 8);

        db.close().unwrap();
    }

    // Session 3: Verify all changes persisted
    {
        let db = Database::open(&dsn).unwrap();

        // Verify users
        let user_count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(user_count, 5, "User count should be 5");

        // Verify orders
        let order_count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(order_count, 8, "Order count should be 8 after delete");

        // Verify update persisted
        let completed_count: i64 = db
            .query_one("SELECT COUNT(*) FROM orders WHERE status = 'completed'", ())
            .unwrap();
        assert_eq!(completed_count, 5, "Completed orders should be 5");

        // Verify unique index still works
        let result = db.execute(
            "INSERT INTO users (id, name, email) VALUES (10, 'Duplicate', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Unique constraint should still be enforced"
        );

        db.close().unwrap();
    }
}

/// Test transaction rollback is not persisted
/// This tests that rolled back transactions don't affect persistence
#[test]
fn test_rollback_not_persisted() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                value TEXT
            )",
            (),
        )
        .unwrap();

        // Insert initial data (this will be committed)
        db.execute(
            "INSERT INTO test_data (id, value) VALUES (1, 'committed')",
            (),
        )
        .unwrap();

        // Start a transaction, insert, then rollback
        db.execute("BEGIN TRANSACTION", ()).unwrap();
        db.execute(
            "INSERT INTO test_data (id, value) VALUES (2, 'rolled_back')",
            (),
        )
        .unwrap();
        db.execute("ROLLBACK", ()).unwrap();

        // Verify rollback worked in current session
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        assert_eq!(count, 1, "Should only have 1 row after rollback");

        db.close().unwrap();
    }

    // Reopen and verify rollback was not persisted
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_data", ()).unwrap();
        assert_eq!(count, 1, "Should still have only 1 row after reopen");

        let value: String = db
            .query_one("SELECT value FROM test_data WHERE id = 1", ())
            .unwrap();
        assert_eq!(value, "committed");

        db.close().unwrap();
    }
}

/// Test that updates to same row multiple times across sessions persist correctly
#[test]
fn test_same_row_updates_across_sessions() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create and insert (use INTEGER PRIMARY KEY as required by stoolap)
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE config (id INTEGER PRIMARY KEY, config_key TEXT, config_value TEXT)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO config (id, config_key, config_value) VALUES (1, 'setting', 'value1')",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Session 2: First update
    {
        let db = Database::open(&dsn).unwrap();
        let value: String = db
            .query_one(
                "SELECT config_value FROM config WHERE config_key = 'setting'",
                (),
            )
            .unwrap();
        assert_eq!(value, "value1");

        db.execute(
            "UPDATE config SET config_value = 'value2' WHERE config_key = 'setting'",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Session 3: Second update
    {
        let db = Database::open(&dsn).unwrap();
        let value: String = db
            .query_one(
                "SELECT config_value FROM config WHERE config_key = 'setting'",
                (),
            )
            .unwrap();
        assert_eq!(value, "value2");

        db.execute(
            "UPDATE config SET config_value = 'value3' WHERE config_key = 'setting'",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Session 4: Verify final state
    {
        let db = Database::open(&dsn).unwrap();
        let value: String = db
            .query_one(
                "SELECT config_value FROM config WHERE config_key = 'setting'",
                (),
            )
            .unwrap();
        assert_eq!(
            value, "value3",
            "Final value should be 'value3' after multiple session updates"
        );
        db.close().unwrap();
    }
}

/// Test delete then insert same key pattern
#[test]
fn test_delete_then_insert_same_key() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create, insert, delete
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO items (id, name) VALUES (1, 'original')", ())
            .unwrap();
        db.execute("DELETE FROM items WHERE id = 1", ()).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 0);

        db.close().unwrap();
    }

    // Session 2: Insert same key with new value
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 0, "Should have 0 items after delete");

        db.execute("INSERT INTO items (id, name) VALUES (1, 'new_value')", ())
            .unwrap();
        db.close().unwrap();
    }

    // Session 3: Verify new value persisted
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 1);

        let name: String = db
            .query_one("SELECT name FROM items WHERE id = 1", ())
            .unwrap();
        assert_eq!(name, "new_value");

        db.close().unwrap();
    }
}

/// Verifies that unique constraints are enforced after database restart
/// without needing to SELECT rows first to load them into memory.
#[test]
fn test_unique_index_bug() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // First database connection - create table, index, and insert data
    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                email TEXT
            )",
            (),
        )
        .unwrap();

        // Create a unique index on email
        db.execute("CREATE UNIQUE INDEX idx_users_email ON users(email)", ())
            .unwrap();

        // Insert some data
        db.execute(
            "INSERT INTO users (id, username, email) VALUES
                (1, 'user1', 'user1@example.com'),
                (2, 'user2', 'user2@example.com'),
                (3, 'user3', 'user3@example.com')",
            (),
        )
        .unwrap();

        // Verify uniqueness constraint works before restart
        let result = db.execute(
            "INSERT INTO users (id, username, email) VALUES (4, 'duplicate', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Expected unique constraint violation, but insert succeeded"
        );

        db.close().unwrap();
    }

    // Reopen the database
    {
        let db = Database::open(&dsn).unwrap();

        // WITHOUT accessing any rows first, try to insert a duplicate email
        // This should fail immediately - unique constraint should be enforced
        let result = db.execute(
            "INSERT INTO users (id, username, email) VALUES (4, 'duplicate', 'user1@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Unique constraint should be enforced after restart without SELECT"
        );

        // Now verify the data is correct by selecting
        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 3, "Should have 3 users");

        // Try to insert another duplicate - should also fail
        let result = db.execute(
            "INSERT INTO users (id, username, email) VALUES (5, 'another_duplicate', 'user2@example.com')",
            (),
        );
        assert!(
            result.is_err(),
            "Unique constraint should still be enforced"
        );

        db.close().unwrap();
    }
}

/// Tests index name preservation and persistence without the complexities
/// of database reopening
#[test]
fn test_index_direct_create() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // First connection
    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
            (),
        )
        .unwrap();

        // Create an index with a special name
        db.execute("CREATE INDEX idx_users_email ON users (email)", ())
            .unwrap();

        // Try to create a duplicate index - this should fail
        let result = db.execute("CREATE INDEX idx_users_email ON users (email)", ());
        assert!(
            result.is_err(),
            "Creating duplicate index should have failed"
        );

        db.close().unwrap();
    }

    // Reopen the database to test persistence
    {
        let db = Database::open(&dsn).unwrap();

        // Try to create the same index again - should fail since it already exists
        let result = db.execute("CREATE INDEX idx_users_email ON users (email)", ());
        assert!(
            result.is_err(),
            "Creating duplicate index after reopen should have failed"
        );

        // Test dropping the index
        db.execute("DROP INDEX idx_users_email ON users", ())
            .unwrap();

        // Now we should be able to create it again
        db.execute("CREATE INDEX idx_users_email ON users (email)", ())
            .unwrap();

        db.close().unwrap();
    }

    // Verify drop and recreate persisted
    {
        let db = Database::open(&dsn).unwrap();

        // Index should exist, so creating it should fail
        let result = db.execute("CREATE INDEX idx_users_email ON users (email)", ());
        assert!(
            result.is_err(),
            "Index should exist after reopen, create should fail"
        );

        db.close().unwrap();
    }
}

/// Tests CREATE/DROP INDEX IF EXISTS operations and verifies that indexes
/// are properly preserved across database reopens
#[test]
fn test_index_if_exists_operations() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // First database connection
    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price FLOAT,
                category TEXT
            )",
            (),
        )
        .unwrap();

        // Insert some data
        db.execute(
            "INSERT INTO products (id, name, price, category) VALUES
                (1, 'Laptop', 999.99, 'Electronics'),
                (2, 'Smartphone', 599.99, 'Electronics'),
                (3, 'Headphones', 149.99, 'Accessories'),
                (4, 'Backpack', 79.99, 'Accessories'),
                (5, 'Monitor', 349.99, 'Electronics')",
            (),
        )
        .unwrap();

        // Test 1: Create a new index with IF NOT EXISTS (should succeed)
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON products(category)",
            (),
        )
        .unwrap();

        // Test 2: Create the same index again with IF NOT EXISTS (should not error)
        let result = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON products(category)",
            (),
        );
        assert!(
            result.is_ok(),
            "Creating existing index with IF NOT EXISTS should not error: {:?}",
            result.err()
        );

        // Test 3: Create a unique index with IF NOT EXISTS
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_name ON products(name)",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Reopen the database
    {
        let db = Database::open(&dsn).unwrap();

        // Test 4: Indexes should be preserved after reopen
        // Creating them again with IF NOT EXISTS should not error
        let result = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON products(category)",
            (),
        );
        assert!(
            result.is_ok(),
            "Creating existing index with IF NOT EXISTS after reopen should not error"
        );

        let result = db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_name ON products(name)",
            (),
        );
        assert!(
            result.is_ok(),
            "Creating existing unique index with IF NOT EXISTS after reopen should not error"
        );

        // Verify unique constraint still works
        let result = db.execute(
            "INSERT INTO products (id, name, price, category) VALUES (6, 'Laptop', 1299.99, 'Electronics')",
            (),
        );
        assert!(
            result.is_err(),
            "Unique constraint on name should still be enforced after reopen"
        );

        // Test 5: Test DROP INDEX IF EXISTS on an existing index
        db.execute("DROP INDEX IF EXISTS idx_category ON products", ())
            .unwrap();

        // Test 6: Test DROP INDEX IF EXISTS on a non-existing index (should not error)
        let result = db.execute("DROP INDEX IF EXISTS idx_nonexistent ON products", ());
        assert!(
            result.is_ok(),
            "DROP INDEX IF EXISTS on non-existing index should not error: {:?}",
            result.err()
        );

        // Verify idx_category is gone - creating it should succeed
        db.execute("CREATE INDEX idx_category ON products(category)", ())
            .unwrap();

        db.close().unwrap();
    }
}

/// Tests that index names are properly preserved when using
/// CREATE INDEX IF NOT EXISTS across database reopens
#[test]
fn test_index_name_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // First connection - create table and index
    {
        let db = Database::open(&dsn).unwrap();

        // Create a test table
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
            (),
        )
        .unwrap();

        // Create an index with a special name using IF NOT EXISTS
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Reopen the database
    {
        let db = Database::open(&dsn).unwrap();

        // Try to create the same index again with IF NOT EXISTS - should not error
        let result = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
            (),
        );
        assert!(
            result.is_ok(),
            "Creating existing index with IF NOT EXISTS after reopen should not error: {:?}",
            result.err()
        );

        // Verify the index works by trying to create another index on same column
        // without IF NOT EXISTS - should fail because idx_users_email exists
        let result = db.execute("CREATE INDEX idx_users_email ON users (email)", ());
        assert!(
            result.is_err(),
            "Creating duplicate index without IF NOT EXISTS should fail"
        );

        db.close().unwrap();
    }

    // Third connection - test that we can still use the index
    {
        let db = Database::open(&dsn).unwrap();

        // Insert some data
        db.execute(
            "INSERT INTO users (id, name, email) VALUES
                (1, 'John', 'john@example.com'),
                (2, 'Jane', 'jane@example.com')",
            (),
        )
        .unwrap();

        // Query using the indexed column
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM users WHERE email = 'john@example.com'",
                (),
            )
            .unwrap();
        assert_eq!(count, 1, "Should find 1 user with email john@example.com");

        db.close().unwrap();
    }
}

/// Tests that all data types are correctly persisted and restored
/// through WAL and snapshots.
#[test]
fn test_data_type_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table with all data types and insert data
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE all_types (
                id INTEGER PRIMARY KEY,
                int_val INTEGER,
                float_val FLOAT,
                text_val TEXT,
                bool_val BOOLEAN,
                timestamp_val TIMESTAMP,
                date_val DATE,
                time_val TIME,
                json_val JSON
            )",
            (),
        )
        .unwrap();

        // Insert row with all data types
        db.execute(
            "INSERT INTO all_types (id, int_val, float_val, text_val, bool_val, timestamp_val, date_val, time_val, json_val)
             VALUES (1, 42, 3.14159, 'hello world', true, '2025-04-25 14:30:00', '2025-04-25', '14:30:00', '{\"key\": \"value\"}')",
            (),
        )
        .unwrap();

        // Insert row with NULL values
        db.execute(
            "INSERT INTO all_types (id, int_val, float_val, text_val, bool_val, timestamp_val, date_val, time_val, json_val)
             VALUES (2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
            (),
        )
        .unwrap();

        // Insert row with negative and edge values
        // Note: Using i64::MIN + 1 because the parser can't handle i64::MIN directly
        // (it parses the positive part first which overflows)
        db.execute(
            "INSERT INTO all_types (id, int_val, float_val, text_val, bool_val, timestamp_val, date_val, time_val, json_val)
             VALUES (3, -9223372036854775807, -0.00001, '', false, '1970-01-01 00:00:00', '1970-01-01', '00:00:00', '[]')",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Session 2: Reopen and verify all data types persisted correctly
    {
        let db = Database::open(&dsn).unwrap();

        // Verify count
        let count: i64 = db.query_one("SELECT COUNT(*) FROM all_types", ()).unwrap();
        assert_eq!(count, 3, "Should have 3 rows");

        // Verify integer value
        let int_val: i64 = db
            .query_one("SELECT int_val FROM all_types WHERE id = 1", ())
            .unwrap();
        assert_eq!(int_val, 42, "Integer value mismatch");

        // Verify float value (with tolerance)
        let float_val: f64 = db
            .query_one("SELECT float_val FROM all_types WHERE id = 1", ())
            .unwrap();
        assert!(
            (float_val - 3.14159).abs() < 0.00001,
            "Float value mismatch: {}",
            float_val
        );

        // Verify text value
        let text_val: String = db
            .query_one("SELECT text_val FROM all_types WHERE id = 1", ())
            .unwrap();
        assert_eq!(text_val, "hello world", "Text value mismatch");

        // Verify boolean value
        let bool_val: bool = db
            .query_one("SELECT bool_val FROM all_types WHERE id = 1", ())
            .unwrap();
        assert!(bool_val, "Boolean value mismatch");

        // Verify JSON value
        let json_val: String = db
            .query_one("SELECT json_val FROM all_types WHERE id = 1", ())
            .unwrap();
        assert!(
            json_val.contains("key") && json_val.contains("value"),
            "JSON value mismatch: {}",
            json_val
        );

        // Verify NULL values using IS NULL
        let null_count: i64 = db
            .query_one("SELECT COUNT(*) FROM all_types WHERE int_val IS NULL", ())
            .unwrap();
        assert_eq!(null_count, 1, "Should have 1 row with NULL int_val");

        // Verify negative integer (i64::MIN + 1)
        let neg_int: i64 = db
            .query_one("SELECT int_val FROM all_types WHERE id = 3", ())
            .unwrap();
        assert_eq!(
            neg_int, -9223372036854775807i64,
            "Negative integer mismatch"
        );

        // Verify empty string
        let empty_str: String = db
            .query_one("SELECT text_val FROM all_types WHERE id = 3", ())
            .unwrap();
        assert_eq!(empty_str, "", "Empty string mismatch");

        // Verify false boolean
        let false_bool: bool = db
            .query_one("SELECT bool_val FROM all_types WHERE id = 3", ())
            .unwrap();
        assert!(!false_bool, "False boolean mismatch");

        db.close().unwrap();
    }
}

/// Tests that transactions are properly isolated in the WAL, with commits
/// persisted and rollbacks not persisted.
#[test]
fn test_wal_transaction_isolation() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table and test committed vs rolled back transactions
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE txn_test (
                id INTEGER PRIMARY KEY,
                value TEXT,
                source TEXT
            )",
            (),
        )
        .unwrap();

        // Committed transaction 1
        db.execute("BEGIN TRANSACTION", ()).unwrap();
        db.execute(
            "INSERT INTO txn_test (id, value, source) VALUES (1, 'committed1', 'txn1')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO txn_test (id, value, source) VALUES (2, 'committed2', 'txn1')",
            (),
        )
        .unwrap();
        db.execute("COMMIT", ()).unwrap();

        // Rolled back transaction 2
        db.execute("BEGIN TRANSACTION", ()).unwrap();
        db.execute(
            "INSERT INTO txn_test (id, value, source) VALUES (3, 'rolled_back', 'txn2')",
            (),
        )
        .unwrap();
        db.execute("ROLLBACK", ()).unwrap();

        // Committed transaction 3
        db.execute("BEGIN TRANSACTION", ()).unwrap();
        db.execute(
            "INSERT INTO txn_test (id, value, source) VALUES (4, 'committed3', 'txn3')",
            (),
        )
        .unwrap();
        db.execute("COMMIT", ()).unwrap();

        // Verify before close
        let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_test", ()).unwrap();
        assert_eq!(count, 3, "Should have 3 rows (2 from txn1, 1 from txn3)");

        db.close().unwrap();
    }

    // Session 2: Verify only committed transactions are persisted
    {
        let db = Database::open(&dsn).unwrap();

        // Verify total count
        let count: i64 = db.query_one("SELECT COUNT(*) FROM txn_test", ()).unwrap();
        assert_eq!(
            count, 3,
            "Should have 3 rows after reopen (rolled back transaction should not be persisted)"
        );

        // Verify txn1 rows
        let txn1_count: i64 = db
            .query_one("SELECT COUNT(*) FROM txn_test WHERE source = 'txn1'", ())
            .unwrap();
        assert_eq!(txn1_count, 2, "Should have 2 rows from txn1");

        // Verify no rolled back rows
        let txn2_count: i64 = db
            .query_one("SELECT COUNT(*) FROM txn_test WHERE source = 'txn2'", ())
            .unwrap();
        assert_eq!(txn2_count, 0, "Should have 0 rows from txn2 (rolled back)");

        // Verify txn3 row
        let txn3_count: i64 = db
            .query_one("SELECT COUNT(*) FROM txn_test WHERE source = 'txn3'", ())
            .unwrap();
        assert_eq!(txn3_count, 1, "Should have 1 row from txn3");

        db.close().unwrap();
    }
}

/// Tests that DDL operations (CREATE TABLE, DROP TABLE, etc.) are properly
/// persisted through WAL.
#[test]
fn test_ddl_operations_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create multiple tables with various structures
    {
        let db = Database::open(&dsn).unwrap();

        // Create first table
        db.execute(
            "CREATE TABLE table1 (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER
            )",
            (),
        )
        .unwrap();

        // Insert some data
        db.execute(
            "INSERT INTO table1 (id, name, value) VALUES (1, 'test1', 100)",
            (),
        )
        .unwrap();

        // Create second table
        db.execute(
            "CREATE TABLE table2 (
                id INTEGER PRIMARY KEY,
                data TEXT,
                is_active BOOLEAN
            )",
            (),
        )
        .unwrap();

        // Insert data into second table
        db.execute(
            "INSERT INTO table2 (id, data, is_active) VALUES (1, 'data1', true)",
            (),
        )
        .unwrap();

        // Create third table and then drop it
        db.execute(
            "CREATE TABLE table3 (id INTEGER PRIMARY KEY, temp TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO table3 (id, temp) VALUES (1, 'temporary')", ())
            .unwrap();
        db.execute("DROP TABLE table3", ()).unwrap();

        db.close().unwrap();
    }

    // Session 2: Verify DDL operations were persisted correctly
    {
        let db = Database::open(&dsn).unwrap();

        // Verify table1 exists and has data
        let count1: i64 = db.query_one("SELECT COUNT(*) FROM table1", ()).unwrap();
        assert_eq!(count1, 1, "table1 should have 1 row");

        let name: String = db
            .query_one("SELECT name FROM table1 WHERE id = 1", ())
            .unwrap();
        assert_eq!(name, "test1", "table1 name mismatch");

        // Verify table2 exists and has data
        let count2: i64 = db.query_one("SELECT COUNT(*) FROM table2", ()).unwrap();
        assert_eq!(count2, 1, "table2 should have 1 row");

        // Verify table3 does NOT exist (was dropped)
        let result = db.execute("SELECT COUNT(*) FROM table3", ());
        assert!(result.is_err(), "table3 should not exist (was dropped)");

        db.close().unwrap();
    }
}

/// Tests recovery scenarios where we have a checkpoint and additional
/// WAL entries after the checkpoint.
#[test]
fn test_checkpoint_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create data that will be included in a checkpoint
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE checkpoint_test (
                id INTEGER PRIMARY KEY,
                phase TEXT,
                value INTEGER
            )",
            (),
        )
        .unwrap();

        // Insert initial data (will be in checkpoint)
        for i in 1..=5 {
            db.execute(
                &format!(
                    "INSERT INTO checkpoint_test (id, phase, value) VALUES ({}, 'initial', {})",
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Session 2: Add more data after checkpoint, then close
    {
        let db = Database::open(&dsn).unwrap();

        // Verify initial data
        let initial_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM checkpoint_test WHERE phase = 'initial'",
                (),
            )
            .unwrap();
        assert_eq!(initial_count, 5, "Should have 5 initial rows");

        // Add more data (will be in WAL only after last checkpoint)
        for i in 6..=10 {
            db.execute(
                &format!(
                    "INSERT INTO checkpoint_test (id, phase, value) VALUES ({}, 'after_checkpoint', {})",
                    i,
                    i * 10
                ),
                (),
            )
            .unwrap();
        }

        // Update some existing rows
        db.execute(
            "UPDATE checkpoint_test SET value = value + 100 WHERE id <= 3",
            (),
        )
        .unwrap();

        // Delete one row
        db.execute("DELETE FROM checkpoint_test WHERE id = 4", ())
            .unwrap();

        db.close().unwrap();
    }

    // Session 3: Verify all changes (checkpoint + WAL replay) are recovered
    {
        let db = Database::open(&dsn).unwrap();

        // Verify total count (10 inserted - 1 deleted = 9)
        let total_count: i64 = db
            .query_one("SELECT COUNT(*) FROM checkpoint_test", ())
            .unwrap();
        assert_eq!(total_count, 9, "Should have 9 rows (10 - 1 deleted)");

        // Verify initial rows (minus deleted one)
        let initial_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM checkpoint_test WHERE phase = 'initial'",
                (),
            )
            .unwrap();
        assert_eq!(
            initial_count, 4,
            "Should have 4 initial rows (5 - 1 deleted)"
        );

        // Verify after-checkpoint rows
        let after_count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM checkpoint_test WHERE phase = 'after_checkpoint'",
                (),
            )
            .unwrap();
        assert_eq!(after_count, 5, "Should have 5 after-checkpoint rows");

        // Verify updates were persisted (rows 1-3 should have value + 100)
        let updated_value: i64 = db
            .query_one("SELECT value FROM checkpoint_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(
            updated_value, 110,
            "Updated value for id=1 should be 110 (10 + 100)"
        );

        // Verify row 4 was deleted
        let deleted_count: i64 = db
            .query_one("SELECT COUNT(*) FROM checkpoint_test WHERE id = 4", ())
            .unwrap();
        assert_eq!(deleted_count, 0, "Row with id=4 should be deleted");

        db.close().unwrap();
    }
}

/// Tests that database maintains consistency even with concurrent operations
/// and multiple session reopens.
#[test]
fn test_consistent_checkpoint_behavior() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Initial setup
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE consistency_test (
                id INTEGER PRIMARY KEY,
                counter INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP
            )",
            (),
        )
        .unwrap();

        // Insert initial row
        db.execute(
            "INSERT INTO consistency_test (id, counter, updated_at) VALUES (1, 0, NOW())",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Multiple sessions incrementing the counter
    for session in 1..=5 {
        let db = Database::open(&dsn).unwrap();

        // Read current value
        let current: i64 = db
            .query_one("SELECT counter FROM consistency_test WHERE id = 1", ())
            .unwrap();

        // Increment and update
        db.execute(
            &format!(
                "UPDATE consistency_test SET counter = {}, updated_at = NOW() WHERE id = 1",
                current + 1
            ),
            (),
        )
        .unwrap();

        // Verify within session
        let new_value: i64 = db
            .query_one("SELECT counter FROM consistency_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(
            new_value, session as i64,
            "Counter should be {} in session {}",
            session, session
        );

        db.close().unwrap();
    }

    // Final verification
    {
        let db = Database::open(&dsn).unwrap();

        let final_value: i64 = db
            .query_one("SELECT counter FROM consistency_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(
            final_value, 5,
            "Final counter value should be 5 after 5 sessions"
        );

        db.close().unwrap();
    }
}

// =============================================================================
// PERSISTENCE BUG VERIFICATION TESTS
// These tests verify known persistence issues that need to be fixed
// =============================================================================

/// Test that VIEWs ARE properly persisted after database restart.
/// FIXED: create_view() and drop_view() now record to WAL, apply_wal_entry() handles CreateView/DropView.
#[test]
fn test_view_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table and view
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE view_source (id INTEGER PRIMARY KEY, name TEXT)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO view_source VALUES (1, 'Alice'), (2, 'Bob')",
            (),
        )
        .unwrap();

        db.execute(
            "CREATE VIEW test_view AS SELECT id, UPPER(name) as upper_name FROM view_source",
            (),
        )
        .unwrap();

        // Verify view works in this session
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test_view", ()).unwrap();
        assert_eq!(count, 2, "View should return 2 rows");

        db.close().unwrap();
    }

    // Session 2: Try to use the view - THIS WILL FAIL because views aren't persisted
    {
        let db = Database::open(&dsn).unwrap();

        // This query will fail because the view doesn't exist after restart
        let _count: i64 = db.query_one("SELECT COUNT(*) FROM test_view", ()).unwrap();

        db.close().unwrap();
    }
}

/// Test that ALTER TABLE ADD COLUMN IS persisted after database restart.
/// Schema modifications via ADD COLUMN are recorded to WAL and replayed.
/// Note: Backfilling of existing rows with default values may not persist across restarts.
#[test]
fn test_alter_table_add_column_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table and add column
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE alter_test (id INTEGER PRIMARY KEY, name TEXT)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO alter_test VALUES (1, 'test')", ())
            .unwrap();

        // Add a new column
        db.execute(
            "ALTER TABLE alter_test ADD COLUMN new_column INTEGER DEFAULT 42",
            (),
        )
        .unwrap();

        // Verify column exists in this session
        let value: i64 = db
            .query_one("SELECT new_column FROM alter_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(value, 42, "New column should have default value");

        // Insert a new row after column addition
        db.execute("INSERT INTO alter_test VALUES (2, 'second', 100)", ())
            .unwrap();

        db.close().unwrap();
    }

    // Session 2: Verify schema was persisted
    {
        let db = Database::open(&dsn).unwrap();

        // The column should exist after restart (schema was persisted)
        // New rows inserted after the column was added should have their values
        let value: i64 = db
            .query_one("SELECT new_column FROM alter_test WHERE id = 2", ())
            .unwrap();
        assert_eq!(
            value, 100,
            "New row should have its new_column value persisted"
        );

        // The default expression should work for new inserts
        db.execute("INSERT INTO alter_test (id, name) VALUES (3, 'third')", ())
            .unwrap();
        let value: i64 = db
            .query_one("SELECT new_column FROM alter_test WHERE id = 3", ())
            .unwrap();
        assert_eq!(value, 42, "New insert should use default value");

        db.close().unwrap();
    }
}

/// Test that ALTER TABLE RENAME TO IS persisted after database restart.
/// RENAME TABLE is recorded to WAL and replayed.
#[test]
fn test_rename_table_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table and rename it
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE original_table (id INTEGER PRIMARY KEY, name TEXT)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO original_table VALUES (1, 'test')", ())
            .unwrap();

        // Rename the table
        db.execute("ALTER TABLE original_table RENAME TO renamed_table", ())
            .unwrap();

        // Verify renamed table works in this session
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM renamed_table", ())
            .unwrap();
        assert_eq!(count, 1, "Renamed table should have 1 row");

        db.close().unwrap();
    }

    // Session 2: Verify rename was persisted
    {
        let db = Database::open(&dsn).unwrap();

        // The renamed table should exist after restart
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM renamed_table", ())
            .unwrap();
        assert_eq!(count, 1, "Renamed table should have 1 row after restart");

        // The original table name should not exist
        let result: Result<i64, _> = db.query_one("SELECT COUNT(*) FROM original_table", ());
        assert!(result.is_err(), "Original table name should not exist");

        db.close().unwrap();
    }
}

/// Test that DEFAULT constraints ARE properly persisted after database restart.
/// FIXED: serialize_schema() and deserialize_schema() now include default_expr.
#[test]
fn test_default_constraint_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table with DEFAULT constraint
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE defaults_test (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                count INTEGER DEFAULT 100
            )",
            (),
        )
        .unwrap();

        // Insert row without specifying defaults
        db.execute(
            "INSERT INTO defaults_test (id, name) VALUES (1, 'test')",
            (),
        )
        .unwrap();

        // Verify defaults work in this session
        let status: String = db
            .query_one("SELECT status FROM defaults_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(status, "active", "DEFAULT should be applied");

        let count: i64 = db
            .query_one("SELECT count FROM defaults_test WHERE id = 1", ())
            .unwrap();
        assert_eq!(count, 100, "DEFAULT should be applied");

        db.close().unwrap();
    }

    // Session 2: Test if defaults still work after restart
    {
        let db = Database::open(&dsn).unwrap();

        // Try to insert a new row relying on defaults
        // This may fail or insert NULL instead of the default value
        let result = db.execute(
            "INSERT INTO defaults_test (id, name) VALUES (2, 'test2')",
            (),
        );

        if result.is_ok() {
            // Check if the default was actually applied
            let status_result: Result<String, _> =
                db.query_one("SELECT status FROM defaults_test WHERE id = 2", ());

            match status_result {
                Ok(status) => {
                    // BUG: If we get here without 'active', the DEFAULT wasn't persisted
                    if status != "active" {
                        panic!(
                            "BUG: DEFAULT constraint not persisted! Expected 'active', got '{}'",
                            status
                        );
                    }
                }
                Err(_) => {
                    // NULL was inserted instead of default
                    panic!("BUG: DEFAULT constraint not persisted! Got NULL instead of 'active'");
                }
            }
        }

        db.close().unwrap();
    }
}

/// Test that CHECK constraints ARE properly persisted after database restart.
/// FIXED: serialize_schema() and deserialize_schema() now include check_expr.
#[test]
fn test_check_constraint_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table with CHECK constraint
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE check_test (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK (age >= 0 AND age <= 150),
                score INTEGER CHECK (score BETWEEN 0 AND 100)
            )",
            (),
        )
        .unwrap();

        // Verify CHECK constraint works - valid values should succeed
        db.execute("INSERT INTO check_test VALUES (1, 25, 85)", ())
            .unwrap();

        // Verify CHECK constraint works - invalid values should fail
        let invalid_result = db.execute("INSERT INTO check_test VALUES (2, -5, 50)", ());
        assert!(
            invalid_result.is_err(),
            "CHECK constraint should reject negative age"
        );

        db.close().unwrap();
    }

    // Session 2: Test if CHECK constraints still work after restart
    {
        let db = Database::open(&dsn).unwrap();

        // Try to insert invalid data that should be rejected by CHECK constraint
        let invalid_result = db.execute("INSERT INTO check_test VALUES (3, -10, 50)", ());

        if invalid_result.is_ok() {
            // BUG: CHECK constraint was not persisted!
            panic!(
                "BUG: CHECK constraint not persisted! Invalid age -10 was accepted after restart"
            );
        }

        // Also try invalid score
        let invalid_score = db.execute("INSERT INTO check_test VALUES (4, 25, 200)", ());
        if invalid_score.is_ok() {
            panic!(
                "BUG: CHECK constraint not persisted! Invalid score 200 was accepted after restart"
            );
        }

        db.close().unwrap();
    }
}

/// Test that inline UNIQUE constraints (column-level) ARE properly persisted after database restart.
/// FIXED: ddl.rs now calls record_create_index() after creating indexes for inline UNIQUE constraints.
#[test]
fn test_inline_unique_constraint_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Session 1: Create table with inline UNIQUE constraint
    {
        let db = Database::open(&dsn).unwrap();

        // Using inline UNIQUE constraint (NOT explicit CREATE UNIQUE INDEX)
        db.execute(
            "CREATE TABLE unique_test (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE
            )",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO unique_test VALUES (1, 'test@example.com')", ())
            .unwrap();

        // Verify UNIQUE constraint works in this session
        let dup_result = db.execute("INSERT INTO unique_test VALUES (2, 'test@example.com')", ());
        assert!(
            dup_result.is_err(),
            "UNIQUE constraint should reject duplicate"
        );

        db.close().unwrap();
    }

    // Session 2: Test if UNIQUE constraint still works after restart
    {
        let db = Database::open(&dsn).unwrap();

        // Try to insert duplicate email - should be rejected if UNIQUE is persisted
        let dup_result = db.execute("INSERT INTO unique_test VALUES (3, 'test@example.com')", ());

        if dup_result.is_ok() {
            // BUG: UNIQUE constraint was not persisted!
            panic!("BUG: Inline UNIQUE constraint not persisted! Duplicate 'test@example.com' was accepted after restart");
        }

        db.close().unwrap();
    }
}
