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

use stoolap::Database;
use tempfile::tempdir;

#[test]
fn test_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table with index
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_name ON products(name)", ())
            .unwrap();
        db.execute(
            "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 19.99)",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Phase 2: Reopen and verify index works
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM products WHERE name = 'Widget'", ())
            .unwrap();
        assert_eq!(count, 1, "Expected 1 row for Widget");

        let count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
        assert_eq!(count, 2, "Expected 2 products");
        db.close().unwrap();
    }
}

#[test]
fn test_unique_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table with unique index
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)",
            (),
        )
        .unwrap();
        db.execute("CREATE UNIQUE INDEX idx_email ON users(email)", ())
            .unwrap();
        db.execute(
            "INSERT INTO users (id, email) VALUES (1, 'test@example.com')",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Phase 2: Verify unique constraint still works after reopen
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 1, "Expected 1 user");

        // Try to insert duplicate - should fail
        let result = db.execute(
            "INSERT INTO users (id, email) VALUES (2, 'test@example.com')",
            (),
        );
        assert!(result.is_err(), "Duplicate insert should fail after reopen");
        db.close().unwrap();
    }
}

#[test]
fn test_update_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create and update
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO items (id, value) VALUES (1, 100)", ())
            .unwrap();
        db.execute("UPDATE items SET value = 200 WHERE id = 1", ())
            .unwrap();
        db.close().unwrap();
    }

    // Phase 2: Verify update persisted
    {
        let db = Database::open(&dsn).unwrap();
        let value: i64 = db
            .query_one("SELECT value FROM items WHERE id = 1", ())
            .unwrap();
        assert_eq!(value, 200, "Expected updated value 200");
        db.close().unwrap();
    }
}

#[test]
fn test_delete_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create, insert, and delete
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO items (id, value) VALUES (1, 100), (2, 200), (3, 300)",
            (),
        )
        .unwrap();
        db.execute("DELETE FROM items WHERE id = 2", ()).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 2, "Expected 2 items after delete");
        db.close().unwrap();
    }

    // Phase 2: Verify delete persisted
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 2, "Expected 2 items after reopen");

        let rows = db.query("SELECT id FROM items ORDER BY id", ()).unwrap();
        let ids: Vec<i64> = rows.map(|r| r.unwrap().get(0).unwrap()).collect();
        assert_eq!(ids, vec![1, 3], "Expected ids 1 and 3");
        db.close().unwrap();
    }
}

#[test]
fn test_multiple_tables_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create multiple tables
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 100)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)",
            (),
        )
        .unwrap();
        db.close().unwrap();
    }

    // Phase 2: Verify all tables persisted
    {
        let db = Database::open(&dsn).unwrap();

        let users: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(users, 2, "Expected 2 users");

        let orders: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(orders, 1, "Expected 1 order");

        let products: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
        assert_eq!(products, 1, "Expected 1 product");
        db.close().unwrap();
    }
}

#[test]
fn test_auto_increment_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table with auto-increment and insert rows
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO items (name) VALUES ('First')", ())
            .unwrap();
        db.execute("INSERT INTO items (name) VALUES ('Second')", ())
            .unwrap();

        let last_id: i64 = db.query_one("SELECT MAX(id) FROM items", ()).unwrap();
        assert_eq!(last_id, 2, "Expected last id to be 2");
        db.close().unwrap();
    }

    // Phase 2: Verify auto-increment continues properly
    {
        let db = Database::open(&dsn).unwrap();

        // Insert another row - id should be 3 or higher
        db.execute("INSERT INTO items (name) VALUES ('Third')", ())
            .unwrap();

        let last_id: i64 = db.query_one("SELECT MAX(id) FROM items", ()).unwrap();
        assert!(last_id >= 3, "Expected last id to be >= 3, got {}", last_id);

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 3, "Expected 3 items");
        db.close().unwrap();
    }
}

#[test]
fn test_transaction_rollback_not_persisted() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table and rollback a transaction
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
            .unwrap();

        // Start transaction, insert, then rollback
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO test (id, value) VALUES (2, 200)", ())
            .unwrap();
        tx.rollback().unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 1, "Should have only 1 row after rollback");
        db.close().unwrap();
    }

    // Phase 2: Verify rolled back data is NOT persisted
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 1, "Should still have only 1 row after reopen");
        db.close().unwrap();
    }
}

#[test]
fn test_create_table_as_select_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create source table and CREATE TABLE AS SELECT
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO source (id, name, value) VALUES (1, 'A', 100), (2, 'B', 200), (3, 'C', 300)", ()).unwrap();

        // Create table from SELECT
        db.execute("CREATE TABLE derived AS SELECT id, name, value * 2 AS doubled FROM source WHERE value >= 200", ()).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM derived", ()).unwrap();
        assert_eq!(count, 2, "Expected 2 rows in derived table");
        db.close().unwrap();
    }

    // Phase 2: Verify both tables and data persist
    {
        let db = Database::open(&dsn).unwrap();

        // Source table should still exist
        let source_count: i64 = db.query_one("SELECT COUNT(*) FROM source", ()).unwrap();
        assert_eq!(source_count, 3, "Source should have 3 rows");

        // Derived table should exist with correct data
        let derived_count: i64 = db.query_one("SELECT COUNT(*) FROM derived", ()).unwrap();
        assert_eq!(derived_count, 2, "Derived should have 2 rows");

        // Check derived values are correct
        let doubled_sum: i64 = db
            .query_one("SELECT SUM(doubled) FROM derived", ())
            .unwrap();
        assert_eq!(
            doubled_sum, 1000,
            "Expected sum of doubled values to be 1000 (400 + 600)"
        );

        db.close().unwrap();
    }
}

#[test]
fn test_truncate_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table, insert, then truncate
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO items (id, value) VALUES (1, 100), (2, 200), (3, 300)",
            (),
        )
        .unwrap();

        let before: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(before, 3, "Expected 3 items before truncate");

        // Truncate the table
        db.execute("TRUNCATE TABLE items", ()).unwrap();

        let after: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(after, 0, "Expected 0 items after truncate");
        db.close().unwrap();
    }

    // Phase 2: Verify truncation persisted
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 0, "Table should still be empty after reopen");

        // Can still insert into the table
        db.execute("INSERT INTO items (id, value) VALUES (10, 1000)", ())
            .unwrap();
        let new_count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(new_count, 1, "Should have 1 row after new insert");
        db.close().unwrap();
    }
}

#[test]
fn test_multi_column_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table with multi-column index and insert data
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date TEXT,
                amount FLOAT
            )",
            (),
        )
        .unwrap();

        // Create multi-column index
        db.execute(
            "CREATE INDEX idx_customer_date ON orders(customer_id, order_date)",
            (),
        )
        .unwrap();

        // Insert test data
        db.execute(
            "INSERT INTO orders VALUES (1, 100, '2025-01-01', 99.99)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO orders VALUES (2, 100, '2025-01-02', 149.99)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO orders VALUES (3, 200, '2025-01-01', 199.99)",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Phase 2: Reopen and verify multi-column index works
    {
        let db = Database::open(&dsn).unwrap();

        // Verify data persisted
        let count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        assert_eq!(count, 3, "Expected 3 orders after reopen");

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

        db.close().unwrap();
    }
}

#[test]
fn test_unique_multi_column_index_persistence() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table with unique multi-column index
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE reservations (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                date TEXT,
                guest_name TEXT
            )",
            (),
        )
        .unwrap();

        // Create unique multi-column index (room_id + date should be unique)
        db.execute(
            "CREATE UNIQUE INDEX idx_room_date ON reservations(room_id, date)",
            (),
        )
        .unwrap();

        // Insert test data
        db.execute(
            "INSERT INTO reservations VALUES (1, 101, '2025-06-01', 'Alice')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO reservations VALUES (2, 101, '2025-06-02', 'Bob')",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO reservations VALUES (3, 102, '2025-06-01', 'Charlie')",
            (),
        )
        .unwrap();

        db.close().unwrap();
    }

    // Phase 2: Verify unique constraint still works after reopen
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db
            .query_one("SELECT COUNT(*) FROM reservations", ())
            .unwrap();
        assert_eq!(count, 3, "Expected 3 reservations after reopen");

        // Try to insert duplicate room_id + date - should fail
        let result = db.execute(
            "INSERT INTO reservations VALUES (4, 101, '2025-06-01', 'Eve')",
            (),
        );
        assert!(
            result.is_err(),
            "Duplicate room_id + date should fail after reopen"
        );

        // Different combination should succeed
        db.execute(
            "INSERT INTO reservations VALUES (4, 101, '2025-06-03', 'Eve')",
            (),
        )
        .unwrap();

        let final_count: i64 = db
            .query_one("SELECT COUNT(*) FROM reservations", ())
            .unwrap();
        assert_eq!(final_count, 4, "Expected 4 reservations after new insert");

        db.close().unwrap();
    }
}
