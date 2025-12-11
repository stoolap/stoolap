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

// Snapshot recovery integration tests
// These tests verify that snapshots can be created and loaded correctly

use stoolap::Database;
use tempfile::tempdir;

/// Test basic snapshot creation and loading
#[test]
fn test_snapshot_create_and_load() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Phase 1: Create database, add data, create snapshot
    {
        let db = Database::open(&dsn).expect("Failed to open database");

        // Create a table with some data
        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                balance FLOAT DEFAULT 0.0
            )",
            (),
        )
        .expect("Failed to create table");

        // Insert some rows
        db.execute(
            "INSERT INTO users (name, email, balance) VALUES ('Alice', 'alice@test.com', 100.50)",
            (),
        )
        .expect("Insert failed");
        db.execute(
            "INSERT INTO users (name, email, balance) VALUES ('Bob', 'bob@test.com', 250.75)",
            (),
        )
        .expect("Insert failed");
        db.execute(
            "INSERT INTO users (name, email, balance) VALUES ('Charlie', NULL, 0.0)",
            (),
        )
        .expect("Insert failed");

        // Create a snapshot
        db.create_snapshot().expect("Failed to create snapshot");

        // Verify data before closing
        let rows: Vec<_> = db
            .query("SELECT id, name, email, balance FROM users ORDER BY id", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(rows.len(), 3);

        // Close database
        drop(db);
    }

    // Phase 2: Reopen and verify snapshot data is recovered
    {
        let db = Database::open(&dsn).expect("Failed to reopen database");

        // Query the data - should be recovered from snapshot
        let rows: Vec<_> = db
            .query("SELECT id, name, email, balance FROM users ORDER BY id", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(rows.len(), 3, "Expected 3 rows after recovery");

        // Verify first row (Alice)
        let id: i64 = rows[0].get(0).unwrap();
        let name: String = rows[0].get(1).unwrap();
        assert_eq!(id, 1);
        assert_eq!(name, "Alice");

        // Verify second row (Bob)
        let id: i64 = rows[1].get(0).unwrap();
        let name: String = rows[1].get(1).unwrap();
        assert_eq!(id, 2);
        assert_eq!(name, "Bob");

        // Verify third row (Charlie)
        let id: i64 = rows[2].get(0).unwrap();
        let name: String = rows[2].get(1).unwrap();
        assert_eq!(id, 3);
        assert_eq!(name, "Charlie");
    }
}

/// Test snapshot with subsequent WAL updates
#[test]
fn test_snapshot_with_subsequent_wal() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db_wal");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Phase 1: Create database, add initial data, create snapshot
    {
        let db = Database::open(&dsn).expect("Failed to open database");

        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, quantity INTEGER)",
            (),
        )
        .expect("Create failed");

        db.execute("INSERT INTO orders VALUES (1, 'Widget', 10)", ())
            .expect("Insert failed");
        db.execute("INSERT INTO orders VALUES (2, 'Gadget', 5)", ())
            .expect("Insert failed");

        // Create snapshot - captures 2 rows
        db.create_snapshot().expect("Failed to create snapshot");

        // Add more data AFTER snapshot (these go to WAL only)
        db.execute("INSERT INTO orders VALUES (3, 'Doohickey', 20)", ())
            .expect("Insert failed");
        db.execute("UPDATE orders SET quantity = 15 WHERE id = 1", ())
            .expect("Update failed");

        drop(db);
    }

    // Phase 2: Reopen - should load snapshot + replay WAL
    {
        let db = Database::open(&dsn).expect("Failed to reopen database");

        let rows: Vec<_> = db
            .query("SELECT id, product, quantity FROM orders ORDER BY id", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(rows.len(), 3, "Expected 3 rows after recovery");

        // Row 1 was updated after snapshot
        let id: i64 = rows[0].get(0).unwrap();
        let quantity: i64 = rows[0].get(2).unwrap();
        assert_eq!(id, 1);
        assert_eq!(quantity, 15, "Row 1 should have updated quantity");

        // Row 2 unchanged
        let id: i64 = rows[1].get(0).unwrap();
        let quantity: i64 = rows[1].get(2).unwrap();
        assert_eq!(id, 2);
        assert_eq!(quantity, 5);

        // Row 3 was inserted after snapshot
        let id: i64 = rows[2].get(0).unwrap();
        let product: String = rows[2].get(1).unwrap();
        assert_eq!(id, 3);
        assert_eq!(product, "Doohickey");
    }
}

/// Test snapshot preserves auto-increment counter
#[test]
fn test_snapshot_auto_increment() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db_auto");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Phase 1: Create table with auto-increment, add data, snapshot
    {
        let db = Database::open(&dsn).expect("Failed to open database");

        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
            (),
        )
        .expect("Create failed");

        db.execute("INSERT INTO items (name) VALUES ('Item A')", ())
            .expect("Insert failed");
        db.execute("INSERT INTO items (name) VALUES ('Item B')", ())
            .expect("Insert failed");
        db.execute("INSERT INTO items (name) VALUES ('Item C')", ())
            .expect("Insert failed");

        // Create snapshot
        db.create_snapshot().expect("Failed to create snapshot");

        drop(db);
    }

    // Phase 2: Reopen and add more items - should continue auto-increment
    {
        let db = Database::open(&dsn).expect("Failed to reopen database");

        // Insert new item
        db.execute("INSERT INTO items (name) VALUES ('Item D')", ())
            .expect("Insert failed");

        let rows: Vec<_> = db
            .query("SELECT id, name FROM items ORDER BY id", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(rows.len(), 4);

        // Verify IDs
        let id1: i64 = rows[0].get(0).unwrap();
        let id2: i64 = rows[1].get(0).unwrap();
        let id3: i64 = rows[2].get(0).unwrap();
        let id4: i64 = rows[3].get(0).unwrap();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        // New item should get ID 4, not 1
        assert_eq!(id4, 4, "Auto-increment should continue from 4");
    }
}

/// Test snapshot with multiple tables
#[test]
fn test_snapshot_multiple_tables() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db_multi");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Phase 1: Create multiple tables and snapshot
    {
        let db = Database::open(&dsn).expect("Failed to open database");

        db.execute(
            "CREATE TABLE table_a (id INTEGER PRIMARY KEY, val TEXT)",
            (),
        )
        .expect("Create failed");
        db.execute(
            "CREATE TABLE table_b (id INTEGER PRIMARY KEY, num INTEGER)",
            (),
        )
        .expect("Create failed");
        db.execute(
            "CREATE TABLE table_c (id INTEGER PRIMARY KEY, flag BOOLEAN)",
            (),
        )
        .expect("Create failed");

        db.execute("INSERT INTO table_a VALUES (1, 'Hello')", ())
            .expect("Insert failed");
        db.execute("INSERT INTO table_b VALUES (1, 42)", ())
            .expect("Insert failed");
        db.execute("INSERT INTO table_c VALUES (1, TRUE)", ())
            .expect("Insert failed");

        db.create_snapshot().expect("Failed to create snapshot");

        drop(db);
    }

    // Phase 2: Verify all tables recovered
    {
        let db = Database::open(&dsn).expect("Failed to reopen database");

        let rows_a: Vec<_> = db
            .query("SELECT val FROM table_a WHERE id = 1", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();
        let val_a: String = rows_a[0].get(0).unwrap();
        assert_eq!(val_a, "Hello");

        let rows_b: Vec<_> = db
            .query("SELECT num FROM table_b WHERE id = 1", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();
        let val_b: i64 = rows_b[0].get(0).unwrap();
        assert_eq!(val_b, 42);

        let rows_c: Vec<_> = db
            .query("SELECT flag FROM table_c WHERE id = 1", ())
            .expect("Query failed")
            .map(|r| r.unwrap())
            .collect();
        let val_c: bool = rows_c[0].get(0).unwrap();
        assert!(val_c);
    }
}

/// Test snapshot with multi-column indexes
#[test]
fn test_snapshot_multi_column_index() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Phase 1: Create database with multi-column index, add data, create snapshot
    {
        let db = Database::open(&dsn).expect("Failed to open database");

        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date TEXT,
                amount FLOAT
            )",
            (),
        )
        .expect("Failed to create table");

        // Create multi-column index
        db.execute(
            "CREATE INDEX idx_customer_date ON orders(customer_id, order_date)",
            (),
        )
        .expect("Failed to create multi-column index");

        // Create unique multi-column index
        db.execute(
            "CREATE UNIQUE INDEX idx_customer_date_unique ON orders(customer_id, order_date, id)",
            (),
        )
        .expect("Failed to create unique multi-column index");

        // Insert data
        db.execute(
            "INSERT INTO orders VALUES (1, 100, '2025-01-01', 99.99)",
            (),
        )
        .expect("Insert failed");
        db.execute(
            "INSERT INTO orders VALUES (2, 100, '2025-01-02', 149.99)",
            (),
        )
        .expect("Insert failed");
        db.execute(
            "INSERT INTO orders VALUES (3, 200, '2025-01-01', 199.99)",
            (),
        )
        .expect("Insert failed");

        // Create snapshot
        db.create_snapshot().expect("Failed to create snapshot");

        db.close().expect("Failed to close");
    }

    // Phase 2: Reopen from snapshot and verify multi-column index works
    {
        let db = Database::open(&dsn).expect("Failed to reopen database");

        // Verify data
        let count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(count, 3, "Expected 3 orders after snapshot restore");

        // Query using both indexed columns
        let count: i64 = db
            .query_one(
                "SELECT COUNT(*) FROM orders WHERE customer_id = 100 AND order_date = '2025-01-01'",
                (),
            )
            .unwrap();
        assert_eq!(count, 1, "Expected 1 order for customer 100 on 2025-01-01");

        // Query using first indexed column only
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM orders WHERE customer_id = 100", ())
            .unwrap();
        assert_eq!(count, 2, "Expected 2 orders for customer 100");

        // Verify we can insert new data (index still working)
        db.execute(
            "INSERT INTO orders VALUES (4, 100, '2025-01-03', 299.99)",
            (),
        )
        .expect("Insert after snapshot restore failed");

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM orders WHERE customer_id = 100", ())
            .unwrap();
        assert_eq!(count, 3, "Expected 3 orders for customer 100 after insert");

        db.close().expect("Failed to close");
    }
}
