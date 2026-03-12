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

//! Regression test: UNIQUE constraints lost after snapshot + reopen.
//! Snapshots store schema + rows but not index metadata, and WAL truncation
//! after snapshot removes the CreateIndex entries. This caused UNIQUE
//! constraints to silently disappear after database reopen.

use stoolap::Database;

#[test]
fn test_unique_constraint_survives_snapshot_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    // Create table with composite UNIQUE, insert data, force snapshot, close
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE test (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                time TIMESTAMP NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                val FLOAT,
                UNIQUE(exchange, symbol, time)
            )",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO test (time, exchange, symbol, val) VALUES ('2026-01-01 00:00:00', 'ex', 'SYM', 1.0)",
            (),
        )
        .unwrap();

        // Verify UNIQUE works before close
        let dup = db.execute(
            "INSERT INTO test (time, exchange, symbol, val) VALUES ('2026-01-01 00:00:00', 'ex', 'SYM', 2.0)",
            (),
        );
        assert!(dup.is_err(), "Duplicate should be rejected before close");

        // Force snapshot
        db.query("PRAGMA snapshot", ()).unwrap();

        // Close (drops db)
    }

    // Reopen and verify UNIQUE constraint still works
    {
        let db = Database::open(&dsn).unwrap();

        let count: i64 = db
            .query("SELECT COUNT(*) FROM test", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 1, "Should have 1 row after reopen");

        // This should fail — UNIQUE(exchange, symbol, time) must still hold
        let dup = db.execute(
            "INSERT INTO test (time, exchange, symbol, val) VALUES ('2026-01-01 00:00:00', 'ex', 'SYM', 3.0)",
            (),
        );
        assert!(
            dup.is_err(),
            "Duplicate should be rejected after reopen — UNIQUE constraint must survive snapshot"
        );

        // Verify count didn't change
        let count: i64 = db
            .query("SELECT COUNT(*) FROM test", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 1, "Should still have 1 row");
    }
}

#[test]
fn test_single_column_unique_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE emails (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                email TEXT NOT NULL UNIQUE
            )",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO emails (email) VALUES ('alice@example.com')",
            (),
        )
        .unwrap();

        // Verify unique constraint
        let dup = db.execute(
            "INSERT INTO emails (email) VALUES ('alice@example.com')",
            (),
        );
        assert!(dup.is_err(), "Duplicate email should be rejected");

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        let dup = db.execute(
            "INSERT INTO emails (email) VALUES ('alice@example.com')",
            (),
        );
        assert!(
            dup.is_err(),
            "Duplicate email should be rejected after reopen"
        );

        // Non-duplicate should succeed
        db.execute("INSERT INTO emails (email) VALUES ('bob@example.com')", ())
            .unwrap();

        let count: i64 = db
            .query("SELECT COUNT(*) FROM emails", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 2);
    }
}

#[test]
fn test_explicit_index_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL
            )",
            (),
        )
        .unwrap();

        db.execute("CREATE INDEX idx_category ON products (category)", ())
            .unwrap();

        db.execute(
            "INSERT INTO products VALUES (1, 'Widget', 'tools'), (2, 'Gadget', 'electronics')",
            (),
        )
        .unwrap();

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Verify data survived
        let count: i64 = db
            .query("SELECT COUNT(*) FROM products", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 2);

        // Verify index-accelerated query works
        let names: Vec<String> = db
            .query("SELECT name FROM products WHERE category = 'tools'", ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        assert_eq!(names, vec!["Widget"]);
    }
}

#[test]
fn test_upsert_works_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE ohlcv (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                open FLOAT NOT NULL,
                close FLOAT NOT NULL,
                UNIQUE(exchange, symbol, time)
            )",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO ohlcv (exchange, symbol, time, open, close) VALUES
             ('binance', 'BTC/USD', '2024-01-01 00:00:00', 42000.0, 42100.0)",
            (),
        )
        .unwrap();

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Upsert should update the existing row, not fail or create duplicates
        db.execute(
            "INSERT INTO ohlcv (exchange, symbol, time, open, close) VALUES
             ('binance', 'BTC/USD', '2024-01-01 00:00:00', 41000.0, 43000.0)
             ON CONFLICT (exchange, symbol, time) DO UPDATE SET
                 open = EXCLUDED.open, close = EXCLUDED.close",
            (),
        )
        .unwrap();

        let count: i64 = db
            .query("SELECT COUNT(*) FROM ohlcv", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 1, "Upsert should update, not duplicate");

        // Verify values were updated
        let open: f64 = db
            .query("SELECT open FROM ohlcv", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(
            (open - 41000.0).abs() < 0.01,
            "open should be updated to 41000"
        );
    }
}

#[test]
fn test_view_survives_snapshot_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, active BOOLEAN NOT NULL)",
            (),
        )
        .unwrap();

        db.execute(
            "INSERT INTO users VALUES (1, 'Alice', true), (2, 'Bob', false), (3, 'Carol', true)",
            (),
        )
        .unwrap();

        db.execute(
            "CREATE VIEW active_users AS SELECT id, name FROM users WHERE active = true",
            (),
        )
        .unwrap();

        // Verify view works
        let names: Vec<String> = db
            .query("SELECT name FROM active_users ORDER BY id", ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        assert_eq!(names, vec!["Alice", "Carol"]);

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // View should still exist and work after reopen
        let names: Vec<String> = db
            .query("SELECT name FROM active_users ORDER BY id", ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            names,
            vec!["Alice", "Carol"],
            "View should survive snapshot + reopen"
        );
    }
}

#[test]
fn test_check_constraint_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE scores (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT NOT NULL,
                score INTEGER NOT NULL CHECK (score >= 0 AND score <= 100)
            )",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO scores (name, score) VALUES ('Alice', 85)", ())
            .unwrap();

        // Check constraint should reject invalid value
        let bad = db.execute("INSERT INTO scores (name, score) VALUES ('Bob', 150)", ());
        assert!(bad.is_err(), "CHECK constraint should reject score > 100");

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Check constraint must still hold after reopen
        let bad = db.execute("INSERT INTO scores (name, score) VALUES ('Carol', -5)", ());
        assert!(
            bad.is_err(),
            "CHECK constraint should reject score < 0 after reopen"
        );

        // Valid insert should still work
        db.execute("INSERT INTO scores (name, score) VALUES ('Dave', 90)", ())
            .unwrap();

        let count: i64 = db
            .query("SELECT COUNT(*) FROM scores", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 2);
    }
}

#[test]
fn test_describe_shows_composite_unique_key_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE comp_desc (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                val FLOAT,
                UNIQUE(exchange, symbol, time)
            )",
            (),
        )
        .unwrap();

        // Check DESCRIBE before close — MySQL convention: composite unique shows MUL on first col
        let rows: Vec<(String, String)> = db
            .query("DESCRIBE comp_desc", ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<String>(0).unwrap(), r.get::<String>(3).unwrap())
            })
            .collect();

        let exchange_key = rows
            .iter()
            .find(|(f, _)| f == "exchange")
            .map(|(_, k)| k.as_str());
        assert_eq!(
            exchange_key,
            Some("MUL"),
            "Composite UNIQUE columns should show MUL before close"
        );

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        let rows: Vec<(String, String)> = db
            .query("DESCRIBE comp_desc", ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<String>(0).unwrap(), r.get::<String>(3).unwrap())
            })
            .collect();

        let exchange_key = rows
            .iter()
            .find(|(f, _)| f == "exchange")
            .map(|(_, k)| k.as_str());
        assert_eq!(
            exchange_key,
            Some("MUL"),
            "Composite UNIQUE columns should show MUL after reopen"
        );

        // The constraint must still be enforced
        db.execute(
            "INSERT INTO comp_desc (exchange, symbol, time, val) VALUES ('ex', 'SYM', '2026-01-01 00:00:00', 1.0)",
            (),
        )
        .unwrap();

        let dup = db.execute(
            "INSERT INTO comp_desc (exchange, symbol, time, val) VALUES ('ex', 'SYM', '2026-01-01 00:00:00', 2.0)",
            (),
        );
        assert!(
            dup.is_err(),
            "Composite UNIQUE must be enforced after reopen"
        );
    }
}

#[test]
fn test_describe_shows_unique_key_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE describe_test (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL
            )",
            (),
        )
        .unwrap();

        // Verify DESCRIBE shows UNI before close
        let rows: Vec<(String, String)> = db
            .query("DESCRIBE describe_test", ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<String>(0).unwrap(), r.get::<String>(3).unwrap())
            })
            .collect();

        let email_key = rows
            .iter()
            .find(|(f, _)| f == "email")
            .map(|(_, k)| k.as_str());
        assert_eq!(email_key, Some("UNI"), "email should show UNI before close");

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        let rows: Vec<(String, String)> = db
            .query("DESCRIBE describe_test", ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<String>(0).unwrap(), r.get::<String>(3).unwrap())
            })
            .collect();

        let email_key = rows
            .iter()
            .find(|(f, _)| f == "email")
            .map(|(_, k)| k.as_str());
        assert_eq!(email_key, Some("UNI"), "email should show UNI after reopen");
    }
}

#[test]
fn test_default_value_survives_alter_table_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO items VALUES (1, 'Widget')", ())
            .unwrap();

        // Add column with a default value
        db.execute(
            "ALTER TABLE items ADD COLUMN active BOOLEAN DEFAULT true",
            (),
        )
        .unwrap();

        // Old row should get the default value
        let active: bool = db
            .query("SELECT active FROM items WHERE id = 1", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(active, "Old row should get DEFAULT true before snapshot");

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Old row (written before ALTER TABLE) should still get the default
        let active: bool = db
            .query("SELECT active FROM items WHERE id = 1", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(active, "Old row should get DEFAULT true after reopen");
    }
}

/// Expression defaults (CURRENT_TIMESTAMP) must survive snapshot + reopen.
/// Previously, only literal defaults were reconstructed during recovery;
/// expression defaults like CURRENT_TIMESTAMP were lost, causing old rows
/// to get NULL instead of the default value after reopen.
/// Also verifies timestamp STABILITY: the same value must be returned before
/// and after reopen (default_value is persisted, not re-evaluated).
#[test]
fn test_expression_default_survives_alter_table_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=normal&snapshot_interval=300&keep_snapshots=5",
        dir.path().display()
    );

    let ts_before;
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY, msg TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO logs VALUES (1, 'hello')", ())
            .unwrap();

        // Add column with an expression default
        db.execute(
            "ALTER TABLE logs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            (),
        )
        .unwrap();

        // Old row should get a timestamp before snapshot
        ts_before = db
            .query("SELECT created_at FROM logs WHERE id = 1", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap();
        assert!(
            !ts_before.is_empty(),
            "Old row should get CURRENT_TIMESTAMP default before snapshot"
        );

        db.query("PRAGMA snapshot", ()).unwrap();
    }

    // Sleep to ensure any re-evaluation would produce a different timestamp
    std::thread::sleep(std::time::Duration::from_millis(1100));

    {
        let db = Database::open(&dsn).unwrap();

        // Old row should get the SAME timestamp as before reopen
        let ts_after: String = db
            .query("SELECT created_at FROM logs WHERE id = 1", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(
            ts_before, ts_after,
            "Timestamp must be stable across reopen (was {} before, {} after)",
            ts_before, ts_after
        );
    }
}
