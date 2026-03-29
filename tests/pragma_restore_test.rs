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

//! Tests for PRAGMA RESTORE behavior with volume-backed data.
//! Verifies that restore correctly handles volumes, indexes, and data.

use stoolap::Database;

fn query_i64(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

fn query_f64(db: &Database, sql: &str) -> f64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<f64>(0).ok())
        .unwrap_or(f64::NAN)
}

fn query_str(db: &Database, sql: &str) -> Option<String> {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<String>(0).ok())
}

/// Full lifecycle: insert -> snapshot -> modify -> checkpoint -> restore -> verify
#[test]
fn test_restore_clears_volumes_and_restores_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_vol", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    // Create table with data
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price FLOAT NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO items VALUES (1, 'Alpha', 10.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (2, 'Beta', 20.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (3, 'Gamma', 30.0)", ())
        .unwrap();

    // Create backup snapshot (state A: 3 rows)
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Checkpoint to create volumes
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Modify data after snapshot (state B: different data)
    db.execute("INSERT INTO items VALUES (4, 'Delta', 40.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (5, 'Epsilon', 50.0)", ())
        .unwrap();
    db.execute("UPDATE items SET price = 99.0 WHERE id = 1", ())
        .unwrap();
    db.execute("DELETE FROM items WHERE id = 3", ()).unwrap();

    // Checkpoint again (volumes now have state B)
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Verify state B
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 4);
    assert!((query_f64(&db, "SELECT price FROM items WHERE id = 1") - 99.0).abs() < 0.01);

    // Restore to state A
    db.execute("PRAGMA RESTORE", ()).unwrap();

    // Verify state A is back
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM items"),
        3,
        "Should have 3 rows after restore"
    );
    let price1 = query_f64(&db, "SELECT price FROM items WHERE id = 1");
    assert!(
        (price1 - 10.0).abs() < 0.01,
        "Alpha price should be 10.0 (original), got {}",
        price1
    );
    assert_eq!(
        query_str(&db, "SELECT name FROM items WHERE id = 3").as_deref(),
        Some("Gamma"),
        "Gamma should exist after restore"
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM items WHERE id = 4"),
        0,
        "Delta should not exist after restore"
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM items WHERE id = 5"),
        0,
        "Epsilon should not exist after restore"
    );

    db.close().unwrap();

    // Reopen and verify restore persisted
    let db = Database::open(&dsn).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 3);
    let price1 = query_f64(&db, "SELECT price FROM items WHERE id = 1");
    assert!(
        (price1 - 10.0).abs() < 0.01,
        "Alpha price should survive reopen after restore, got {}",
        price1
    );
    db.close().unwrap();
}

/// Verify indexes survive PRAGMA RESTORE
#[test]
fn test_restore_preserves_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_idx", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, age INTEGER, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX idx_email ON users (email)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_age ON users (age)", ())
        .unwrap();

    db.execute(
        "INSERT INTO users VALUES (1, 'alice@test.com', 30, true)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'bob@test.com', 25, true)", ())
        .unwrap();
    db.execute(
        "INSERT INTO users VALUES (3, 'charlie@test.com', 35, false)",
        (),
    )
    .unwrap();

    // Snapshot (state A)
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Modify after snapshot
    db.execute(
        "INSERT INTO users VALUES (4, 'diana@test.com', 28, true)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Restore to state A
    db.execute("PRAGMA RESTORE", ()).unwrap();

    // Verify data
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 3);

    // Verify index-assisted lookups work
    let email_lookup = query_str(&db, "SELECT email FROM users WHERE email = 'bob@test.com'");
    assert_eq!(email_lookup.as_deref(), Some("bob@test.com"));

    let age_range = query_i64(
        &db,
        "SELECT COUNT(*) FROM users WHERE age BETWEEN 25 AND 35",
    );
    assert_eq!(age_range, 3);

    // Verify UNIQUE constraint still enforced
    let dup = db.execute(
        "INSERT INTO users VALUES (99, 'alice@test.com', 40, true)",
        (),
    );
    assert!(
        dup.is_err(),
        "UNIQUE index on email should prevent duplicate after restore"
    );

    // Verify PK constraint still enforced
    let dup_pk = db.execute("INSERT INTO users VALUES (1, 'new@test.com', 40, true)", ());
    assert!(
        dup_pk.is_err(),
        "PRIMARY KEY should prevent duplicate after restore"
    );

    // Verify new writes work with indexes after restore
    db.execute(
        "INSERT INTO users VALUES (10, 'new@test.com', 40, true)",
        (),
    )
    .unwrap();
    let new_lookup = query_str(&db, "SELECT email FROM users WHERE email = 'new@test.com'");
    assert_eq!(new_lookup.as_deref(), Some("new@test.com"));

    db.close().unwrap();

    // Reopen and verify indexes persist
    let db = Database::open(&dsn).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 4);
    let email_lookup = query_str(&db, "SELECT email FROM users WHERE email = 'bob@test.com'");
    assert_eq!(email_lookup.as_deref(), Some("bob@test.com"));

    // UNIQUE still enforced after reopen
    let dup = db.execute(
        "INSERT INTO users VALUES (99, 'alice@test.com', 40, true)",
        (),
    );
    assert!(dup.is_err(), "UNIQUE index should survive restore + reopen");
    db.close().unwrap();
}

/// Verify PRAGMA RESTORE with multiple tables
#[test]
fn test_restore_multiple_tables() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_multi", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, num FLOAT)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 'a')", ()).unwrap();
    db.execute("INSERT INTO t1 VALUES (2, 'b')", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (1, 100.0)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (2, 200.0)", ()).unwrap();

    // Snapshot
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Modify both tables
    db.execute("INSERT INTO t1 VALUES (3, 'c')", ()).unwrap();
    db.execute("DELETE FROM t2 WHERE id = 1", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t1"), 3);
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t2"), 1);

    // Restore
    db.execute("PRAGMA RESTORE", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t1"), 2);
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t2"), 2);

    let sum = query_f64(&db, "SELECT SUM(num) FROM t2");
    assert!((sum - 300.0).abs() < 0.01);

    db.close().unwrap();
}

/// Verify PRAGMA RESTORE with specific timestamp
#[test]
fn test_restore_specific_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_ts", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    // State A: 2 rows
    db.execute("INSERT INTO t VALUES (1, 'first')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'second')", ())
        .unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Wait a moment so timestamps differ
    std::thread::sleep(std::time::Duration::from_millis(100));

    // State B: 4 rows
    db.execute("INSERT INTO t VALUES (3, 'third')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 'fourth')", ())
        .unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // State C: modified
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

    // Restore latest snapshot (state B)
    db.execute("PRAGMA RESTORE", ()).unwrap();
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM t"),
        4,
        "Latest snapshot should have 4 rows"
    );

    db.close().unwrap();
}

/// Verify PRAGMA RESTORE fails gracefully with no snapshots
#[test]
fn test_restore_no_snapshots_error() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_none", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    // No PRAGMA SNAPSHOT was called, so restore should fail
    let result = db.query("PRAGMA RESTORE", ());
    assert!(
        result.is_err(),
        "PRAGMA RESTORE should fail without snapshots"
    );

    // Data should be unaffected
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 1);

    db.close().unwrap();
}

/// Verify PRAGMA RESTORE cleans up old volume files
#[test]
fn test_restore_removes_volume_files() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_clean", dir.path().display());
    let db_dir = dir.path().join("restore_clean");

    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    for i in 0..100 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'row_{}')", i, i), ())
            .unwrap();
    }

    // Snapshot then checkpoint to create volumes
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Verify volumes exist
    let vol_dir = db_dir.join("volumes").join("t");
    assert!(vol_dir.exists(), "volumes/t should exist before restore");

    // Add more data and checkpoint again
    for i in 100..200 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'row_{}')", i, i), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 200);

    // Restore (should clear volumes from state B and recreate from snapshot)
    db.execute("PRAGMA RESTORE", ()).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 100);

    // Volumes should still exist (restore checkpoints after loading)
    assert!(
        db_dir.join("volumes").exists(),
        "volumes/ should exist after restore (checkpoint recreates them)"
    );

    db.close().unwrap();
}

/// Verify writes work correctly after PRAGMA RESTORE
#[test]
fn test_restore_then_write_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_write", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 200)", ()).unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Modify and checkpoint
    db.execute("INSERT INTO t VALUES (3, 300)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Restore
    db.execute("PRAGMA RESTORE", ()).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

    // Write new data after restore
    db.execute("INSERT INTO t VALUES (10, 1000)", ()).unwrap();
    db.execute("UPDATE t SET val = 150 WHERE id = 1", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2); // 1 + 10
    assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 1"), 150);
    assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 10"), 1000);

    // Checkpoint and reopen
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();

    let db = Database::open(&dsn).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);
    assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 1"), 150);
    assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 10"), 1000);
    db.close().unwrap();
}

/// Verify PRAGMA RESTORE preserves indexes and views via ddl.bin
#[test]
fn test_pragma_restore_indexes_and_views() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/restore_ddl", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    // Create tables, indexes, views
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX idx_name ON users (name)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_age ON users (age)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_user ON orders (user_id)", ())
        .unwrap();
    db.execute(
        "CREATE VIEW active_users AS SELECT id, name, age FROM users WHERE active = true",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE VIEW user_totals AS SELECT u.name, SUM(o.amount) as total FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        (),
    )
    .unwrap();

    // Insert data
    db.execute("INSERT INTO users VALUES (1, 'Alice', 30, true)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob', 25, true)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie', 35, false)", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1, 99.99)", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (2, 1, 49.50)", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (3, 2, 200.00)", ())
        .unwrap();

    // Snapshot (saves ddl.bin with indexes + views)
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();

    // Modify after snapshot
    db.execute("INSERT INTO users VALUES (4, 'Diana', 28, true)", ())
        .unwrap();
    db.execute("DELETE FROM users WHERE id = 3", ()).unwrap();
    db.execute("INSERT INTO orders VALUES (4, 4, 300.00)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // PRAGMA RESTORE
    db.execute("PRAGMA RESTORE", ()).unwrap();

    // Verify data restored to snapshot state
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 3);
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 3);
    assert_eq!(
        query_str(&db, "SELECT name FROM users WHERE id = 3").as_deref(),
        Some("Charlie"),
        "Charlie should exist after restore"
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 4"),
        0,
        "Diana should not exist after restore"
    );

    // Verify indexes restored
    let email_hit = query_str(&db, "SELECT name FROM users WHERE age = 25");
    assert_eq!(
        email_hit.as_deref(),
        Some("Bob"),
        "Index lookup should work"
    );

    // Verify UNIQUE constraint
    let dup = db.execute("INSERT INTO users VALUES (99, 'Alice', 40, true)", ());
    assert!(
        dup.is_err(),
        "UNIQUE index should be enforced after restore"
    );

    // Verify views restored
    let active_count = query_i64(&db, "SELECT COUNT(*) FROM active_users");
    assert_eq!(active_count, 2, "active_users view should show Alice + Bob");

    let alice_total = query_f64(&db, "SELECT total FROM user_totals WHERE name = 'Alice'");
    assert!(
        (alice_total - 149.49).abs() < 0.01,
        "Alice total should be 149.49, got {}",
        alice_total
    );

    let bob_total = query_f64(&db, "SELECT total FROM user_totals WHERE name = 'Bob'");
    assert!(
        (bob_total - 200.0).abs() < 0.01,
        "Bob total should be 200.0, got {}",
        bob_total
    );

    // Verify new writes work after restore
    db.execute("INSERT INTO users VALUES (10, 'Eve', 22, true)", ())
        .unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM active_users"), 3);

    db.close().unwrap();

    // Reopen and verify persistence
    let db = Database::open(&dsn).unwrap();
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 4);
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM active_users"), 3);
    let dup = db.execute("INSERT INTO users VALUES (99, 'Bob', 40, true)", ());
    assert!(
        dup.is_err(),
        "UNIQUE index should survive reopen after restore"
    );
    db.close().unwrap();
}
