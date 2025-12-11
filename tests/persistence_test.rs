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

// Persistence tests for WAL and recovery

use stoolap::Database;
use tempfile::tempdir;

#[test]
fn test_persistence_create_table_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create database and table
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();

        // Verify table exists
        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 0);
    }

    // Phase 2: Reopen and verify table still exists
    {
        let db = Database::open(&dsn).unwrap();

        // Table should still exist
        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn test_persistence_insert_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table and insert data
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
            (),
        )
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

        // Verify data
        let count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
        assert_eq!(count, 2);
    }

    // Phase 2: Reopen and verify data persists
    {
        let db = Database::open(&dsn).unwrap();

        // Data should still exist
        let count: i64 = db.query_one("SELECT COUNT(*) FROM products", ()).unwrap();
        assert_eq!(count, 2);

        // Verify rows
        let rows = db
            .query("SELECT id, name, price FROM products ORDER BY id", ())
            .unwrap();
        let rows: Vec<_> = rows.map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 2);

        // Verify first row
        assert_eq!(rows[0].get::<i64>(0).unwrap(), 1);
        assert_eq!(rows[0].get::<String>(1).unwrap(), "Widget");

        // Verify second row
        assert_eq!(rows[1].get::<i64>(0).unwrap(), 2);
        assert_eq!(rows[1].get::<String>(1).unwrap(), "Gadget");
    }
}

#[test]
fn test_persistence_transaction_commit() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("Test DSN: {}", dsn);
    eprintln!("Test path: {:?}", db_path);

    // Phase 1: Create table and commit transaction
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount INTEGER)",
            (),
        )
        .unwrap();

        // Use explicit transaction
        let mut tx = db.begin().unwrap();
        eprintln!("Transaction ID: {}", tx.id());
        tx.execute("INSERT INTO orders (id, amount) VALUES (1, 100)", ())
            .unwrap();
        tx.execute("INSERT INTO orders (id, amount) VALUES (2, 200)", ())
            .unwrap();
        tx.commit().unwrap();
        eprintln!("Transaction committed");

        // Verify
        let count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        eprintln!("Phase 1 count: {}", count);
        assert_eq!(count, 2);
    }

    eprintln!("Phase 1 complete, database scope closed");

    // Check if WAL file exists
    let wal_path = db_path.with_extension("db.wal");
    eprintln!("WAL path: {:?}", wal_path);
    eprintln!("WAL exists: {}", wal_path.exists());
    if wal_path.exists() {
        eprintln!(
            "WAL size: {} bytes",
            std::fs::metadata(&wal_path).unwrap().len()
        );
    }

    // Phase 2: Reopen and verify committed data
    {
        eprintln!("Opening database again...");
        let db = Database::open(&dsn).unwrap();
        eprintln!("Database reopened");

        let count: i64 = db.query_one("SELECT COUNT(*) FROM orders", ()).unwrap();
        eprintln!("Phase 2 count: {}", count);
        assert_eq!(count, 2);

        let rows = db
            .query("SELECT id, amount FROM orders ORDER BY id", ())
            .unwrap();
        let rows: Vec<_> = rows.map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<i64>(1).unwrap(), 100);
        assert_eq!(rows[1].get::<i64>(1).unwrap(), 200);
    }
}

#[test]
fn test_persistence_drop_table() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create table then drop it
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE temp_table (id INTEGER PRIMARY KEY)", ())
            .unwrap();

        // Verify it exists
        let count: i64 = db.query_one("SELECT COUNT(*) FROM temp_table", ()).unwrap();
        assert_eq!(count, 0);

        // Drop it
        db.execute("DROP TABLE temp_table", ()).unwrap();
    }

    // Phase 2: Reopen and verify table is gone
    {
        let db = Database::open(&dsn).unwrap();

        // Table should not exist
        let result = db.query("SELECT COUNT(*) FROM temp_table", ());
        assert!(result.is_err());
    }
}

#[test]
fn test_memory_mode_no_persistence() {
    // Memory mode should work without persistence
    let db = Database::open("memory://").unwrap();
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO test (id) VALUES (1)", ()).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
    assert_eq!(count, 1);
}
