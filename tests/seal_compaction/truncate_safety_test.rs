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

/// Basic TRUNCATE correctness: empties the table, table still usable.
#[test]
fn test_truncate_basic() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 3);

    db.execute("TRUNCATE TABLE t", ()).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0, "table should be empty after TRUNCATE");

    // Table still usable
    db.execute("INSERT INTO t VALUES (4, 40)", ()).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 1);
}

/// TRUNCATE inside an explicit transaction (no other writers) should succeed.
#[test]
fn test_truncate_in_own_transaction() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    // The TRUNCATE's own transaction has no uncommitted writes on the *version store*
    // (it hasn't done any INSERT/UPDATE/DELETE yet), so this should succeed.
    db.execute("TRUNCATE TABLE t", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

/// TRUNCATE must fail when another transaction has uncommitted writes.
///
/// Scenario:
///   1. Tx A begins and inserts a row (uncommitted)
///   2. Tx B (auto-commit) tries TRUNCATE → must fail
///   3. Tx A commits
///   4. TRUNCATE now succeeds
#[test]
fn test_truncate_blocked_by_active_writer() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();

    // Start an explicit transaction and insert (uncommitted write)
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();

    // A second connection tries TRUNCATE — should fail because tx A has uncommitted writes.
    // Since stoolap is embedded (single-process), we use a second Database on the same path.
    // For in-memory DBs, the same Database object shares the engine, so the active
    // transaction's uncommitted writes are visible to the TRUNCATE path.
    //
    // However, TRUNCATE TABLE goes through the executor which creates its own transaction,
    // and the same Database's executor shares the version store. So attempting TRUNCATE
    // while the explicit transaction has uncommitted writes should hit the guard.
    //
    // But wait — TRUNCATE and the explicit transaction both go through the same executor.
    // Let's verify: if we try TRUNCATE while we have an active BEGIN...INSERT:
    // The executor's execute_truncate creates a new table handle. The version store's
    // uncommitted_writes will have the INSERT's row claim from our BEGIN transaction.
    // So truncate_all() should see non-empty uncommitted_writes and fail.

    // Try TRUNCATE — should fail
    let result = db.execute("TRUNCATE TABLE t", ());
    assert!(
        result.is_err(),
        "TRUNCATE should fail when active transaction has uncommitted writes"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("active transactions have uncommitted changes"),
        "Error should mention active transactions, got: {}",
        err
    );

    // Commit the active transaction
    db.execute("COMMIT", ()).unwrap();

    // Now TRUNCATE should succeed
    db.execute("TRUNCATE TABLE t", ()).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

/// TRUNCATE must fail when another transaction has uncommitted UPDATE.
#[test]
fn test_truncate_blocked_by_active_updater() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();

    // Begin transaction and update a row (uncommitted)
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET val = 99 WHERE id = 2", ())
        .unwrap();

    // TRUNCATE should fail
    let result = db.execute("TRUNCATE TABLE t", ());
    assert!(
        result.is_err(),
        "TRUNCATE should fail when active transaction has uncommitted UPDATE"
    );

    // Rollback the active transaction
    db.execute("ROLLBACK", ()).unwrap();

    // Now TRUNCATE should succeed
    db.execute("TRUNCATE TABLE t", ()).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

/// TRUNCATE must fail when another transaction has uncommitted DELETE.
#[test]
fn test_truncate_blocked_by_active_deleter() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();

    let result = db.execute("TRUNCATE TABLE t", ());
    assert!(
        result.is_err(),
        "TRUNCATE should fail when active transaction has uncommitted DELETE"
    );

    db.execute("COMMIT", ()).unwrap();

    db.execute("TRUNCATE TABLE t", ()).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

/// After a failed TRUNCATE, the original data is intact.
#[test]
fn test_truncate_failure_preserves_data() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 40)", ()).unwrap();

    // TRUNCATE fails
    let _ = db.execute("TRUNCATE TABLE t", ());

    // Rollback
    db.execute("ROLLBACK", ()).unwrap();

    // All original rows should still exist (the INSERT in the rolled-back tx is gone)
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(
        count, 3,
        "original data should be intact after failed TRUNCATE"
    );

    let sum: i64 = db.query_one("SELECT SUM(val) FROM t", ()).unwrap();
    assert_eq!(sum, 60);
}

/// TRUNCATE on empty table is a no-op.
#[test]
fn test_truncate_empty_table() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();

    db.execute("TRUNCATE TABLE t", ()).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

/// Persistence: TRUNCATE survives database reopen.
#[test]
fn test_truncate_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_truncate_persist.db");
    let dsn = format!("file://{}", db_path.display());

    // Phase 1: Create, Insert, Truncate
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO items VALUES (1, 100), (2, 200)", ())
            .unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 2);

        db.execute("TRUNCATE TABLE items", ()).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 0);
    }

    // Phase 2: Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 0, "TRUNCATE should be persisted across reopen");

        db.execute("INSERT INTO items VALUES (3, 300)", ()).unwrap();
        let count: i64 = db.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 1);
    }
}
