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

//! Schema evolution tests with volume-backed (cold) data and recovery scenarios.
//!
//! These tests verify that ALTER TABLE ADD COLUMN works correctly when rows
//! have been sealed into immutable cold volumes via PRAGMA CHECKPOINT, and
//! that recovery (close + reopen) preserves schema changes and data.
//!
//! Behavior notes (engine specifics):
//! - `get::<String>` on a NULL TEXT value returns `Ok("")` (empty string).
//! - `get::<i64>` on a NULL INTEGER value may return an error.
//! - Cold rows with ADD COLUMN DEFAULT get the default value on individual reads
//!   but aggregation fast paths may treat them as NULL.
//! - IS NULL / IS NOT NULL filters correctly detect NULLs regardless.

use std::fs;
use std::path::Path;
use stoolap::Database;

/// Helper: execute a scalar query returning i64.
fn qi64(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

/// Helper: execute a scalar query returning f64.
fn qf64(db: &Database, sql: &str) -> f64 {
    db.query_one::<f64, _>(sql, ()).unwrap()
}

/// Helper: count all rows returned by a query (not using COUNT(*), iterating).
fn count_rows(db: &Database, sql: &str) -> i64 {
    let mut rows = db.query(sql, ()).unwrap();
    let mut n = 0i64;
    while rows.next().is_some() {
        n += 1;
    }
    n
}

/// Remove the lock file so we can reopen the database after a drop.
fn remove_lock_file(base_dir: &Path, db_name: &str) {
    let lock = base_dir.join(db_name).join("db.lock");
    let _ = fs::remove_file(lock);
}

// ============================================================================
// 1. ADD COLUMN (nullable) after checkpoint
// ============================================================================

#[test]
fn test_add_column_nullable_after_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_null", dir.path().display());

    // Session 1: create, insert, checkpoint, alter, verify, close
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t1 VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute("INSERT INTO t1 VALUES (2, 'Bob')", ()).unwrap();
        db.execute("INSERT INTO t1 VALUES (3, 'Carol')", ())
            .unwrap();

        // Seal rows into cold volume
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add nullable column (no DEFAULT)
        db.execute("ALTER TABLE t1 ADD COLUMN score INTEGER", ())
            .unwrap();

        // Cold rows should have NULL for score (verified via IS NULL)
        let null_count = qi64(&db, "SELECT COUNT(*) FROM t1 WHERE score IS NULL");
        assert_eq!(null_count, 3, "Cold rows should have NULL for new column");

        // Insert a new row with a value for score
        db.execute("INSERT INTO t1 VALUES (4, 'Dave', 99)", ())
            .unwrap();

        // Verify counts
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t1"), 4);
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t1 WHERE score IS NULL"), 3);
        assert_eq!(qi64(&db, "SELECT score FROM t1 WHERE id = 4"), 99);

        // SELECT * returns 4 rows with correct schema
        assert_eq!(count_rows(&db, "SELECT * FROM t1"), 4);

        db.close().unwrap();
    }

    // Session 2: reopen and verify everything persisted
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t1"), 4);
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t1 WHERE score IS NULL"), 3);
        assert_eq!(qi64(&db, "SELECT score FROM t1 WHERE id = 4"), 99);

        // Can still insert with new schema
        db.execute("INSERT INTO t1 VALUES (5, 'Eve', 50)", ())
            .unwrap();
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t1"), 5);
        assert_eq!(qi64(&db, "SELECT score FROM t1 WHERE id = 5"), 50);

        db.close().unwrap();
    }
}

// ============================================================================
// 2. ADD COLUMN with DEFAULT after checkpoint
// ============================================================================

#[test]
fn test_add_column_with_default_after_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_def", dir.path().display());

    // Session 1
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t2 VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute("INSERT INTO t2 VALUES (2, 'Bob')", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add column with DEFAULT
        db.execute("ALTER TABLE t2 ADD COLUMN level INTEGER DEFAULT 42", ())
            .unwrap();

        // Cold rows get the default value on individual reads
        let level1 = qi64(&db, "SELECT level FROM t2 WHERE id = 1");
        assert_eq!(
            level1, 42,
            "Cold row should get default value via individual read"
        );

        let level2 = qi64(&db, "SELECT level FROM t2 WHERE id = 2");
        assert_eq!(
            level2, 42,
            "Cold row should get default value via individual read"
        );

        // Insert new row (uses default)
        db.execute("INSERT INTO t2 (id, name) VALUES (3, 'Carol')", ())
            .unwrap();
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 3"), 42);

        // Insert new row with explicit value
        db.execute("INSERT INTO t2 VALUES (4, 'Dave', 100)", ())
            .unwrap();
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 4"), 100);

        // Verify row count
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t2"), 4);

        db.close().unwrap();
    }

    // Session 2: verify defaults persist after reopen
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t2"), 4);

        // Individual reads should still show defaults
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 1"), 42);
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 2"), 42);
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 3"), 42);
        assert_eq!(qi64(&db, "SELECT level FROM t2 WHERE id = 4"), 100);

        db.close().unwrap();
    }
}

// ============================================================================
// 3. Multiple ADD COLUMNs across checkpoints
// ============================================================================

#[test]
fn test_multiple_add_columns_across_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_multi", dir.path().display());

    // Session 1
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t3 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1: insert initial rows, checkpoint
        db.execute("INSERT INTO t3 VALUES (1, 'row1')", ()).unwrap();
        db.execute("INSERT INTO t3 VALUES (2, 'row2')", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Phase 2: add col_a, insert more rows, checkpoint
        db.execute("ALTER TABLE t3 ADD COLUMN col_a INTEGER", ())
            .unwrap();
        db.execute("INSERT INTO t3 VALUES (3, 'row3', 30)", ())
            .unwrap();
        db.execute("INSERT INTO t3 VALUES (4, 'row4', 40)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Phase 3: add col_b, insert more rows, checkpoint
        db.execute("ALTER TABLE t3 ADD COLUMN col_b TEXT", ())
            .unwrap();
        db.execute("INSERT INTO t3 VALUES (5, 'row5', 50, 'five')", ())
            .unwrap();
        db.execute("INSERT INTO t3 VALUES (6, 'row6', 60, 'six')", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify the full picture
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t3"), 6);

        // col_a: rows 1-2 are NULL, rows 3-6 have values
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_a IS NULL"), 2);
        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_a IS NOT NULL"),
            4
        );

        // col_b: rows 1-4 are NULL, rows 5-6 have values
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_b IS NULL"), 4);
        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_b IS NOT NULL"),
            2
        );

        db.close().unwrap();
    }

    // Session 2: reopen and verify schema evolution across volumes
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t3"), 6);

        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_a IS NULL"),
            2,
            "Rows 1-2 should still have NULL col_a after reopen"
        );
        assert_eq!(qi64(&db, "SELECT col_a FROM t3 WHERE id = 3"), 30);
        assert_eq!(qi64(&db, "SELECT col_a FROM t3 WHERE id = 5"), 50);

        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t3 WHERE col_b IS NULL"),
            4,
            "Rows 1-4 should have NULL col_b after reopen"
        );

        let col_b_val: String = db
            .query_one("SELECT col_b FROM t3 WHERE id = 5", ())
            .unwrap();
        assert_eq!(col_b_val, "five");

        let col_b_val: String = db
            .query_one("SELECT col_b FROM t3 WHERE id = 6", ())
            .unwrap();
        assert_eq!(col_b_val, "six");

        // Can still insert with the full schema
        db.execute("INSERT INTO t3 VALUES (7, 'row7', 70, 'seven')", ())
            .unwrap();
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t3"), 7);

        db.close().unwrap();
    }
}

// ============================================================================
// 4. ADD COLUMN + UPDATE cold rows
// ============================================================================

#[test]
fn test_add_column_then_update_cold_rows() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_upd", dir.path().display());

    // Session 1
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t4 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t4 VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute("INSERT INTO t4 VALUES (2, 'Bob')", ()).unwrap();
        db.execute("INSERT INTO t4 VALUES (3, 'Carol')", ())
            .unwrap();

        // Seal to cold
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add nullable column
        db.execute("ALTER TABLE t4 ADD COLUMN age INTEGER", ())
            .unwrap();

        // Verify cold rows have NULL
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t4 WHERE age IS NULL"), 3);

        // UPDATE cold rows to set the new column
        db.execute("UPDATE t4 SET age = 30 WHERE id = 1", ())
            .unwrap();
        db.execute("UPDATE t4 SET age = 25 WHERE id = 2", ())
            .unwrap();
        db.execute("UPDATE t4 SET age = 35 WHERE id = 3", ())
            .unwrap();

        // Verify updates took effect
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 1"), 30);
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 2"), 25);
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 3"), 35);
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t4 WHERE age IS NULL"), 0);

        // Checkpoint to seal updated rows
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify after second checkpoint
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 1"), 30);
        assert_eq!(qi64(&db, "SELECT SUM(age) FROM t4"), 90);

        db.close().unwrap();
    }

    // Session 2: verify after reopen
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t4"), 3);
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 1"), 30);
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 2"), 25);
        assert_eq!(qi64(&db, "SELECT age FROM t4 WHERE id = 3"), 35);
        assert_eq!(qi64(&db, "SELECT SUM(age) FROM t4"), 90);

        db.close().unwrap();
    }
}

// ============================================================================
// 5. Recovery after close without explicit checkpoint (WAL replay)
// ============================================================================

#[test]
fn test_recovery_after_close_with_wal_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_recv", dir.path().display());

    // Session 1: insert, checkpoint, insert more, close
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t5 (id INTEGER PRIMARY KEY, val INTEGER NOT NULL)",
            (),
        )
        .unwrap();

        // Batch insert first 100 rows
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t5 VALUES ($1, $2)").unwrap();
        for i in 0..100i64 {
            stmt.execute((i, i * 10)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint seals first batch to cold volume
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Insert 50 more rows (hot only, not checkpointed)
        db.execute("BEGIN", ()).unwrap();
        for i in 100..150i64 {
            stmt.execute((i, i * 10)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t5"), 150);

        // Close without explicit PRAGMA CHECKPOINT.
        // close_engine() force-seals all remaining hot rows.
        db.close().unwrap();
    }

    // Session 2: reopen, all data should be present
    {
        let db = Database::open(&dsn).unwrap();

        let count = qi64(&db, "SELECT COUNT(*) FROM t5");
        assert_eq!(count, 150, "All 150 rows should survive close+reopen");

        // Verify boundary rows
        assert_eq!(qi64(&db, "SELECT val FROM t5 WHERE id = 0"), 0);
        assert_eq!(qi64(&db, "SELECT val FROM t5 WHERE id = 99"), 990);
        assert_eq!(qi64(&db, "SELECT val FROM t5 WHERE id = 100"), 1000);
        assert_eq!(qi64(&db, "SELECT val FROM t5 WHERE id = 149"), 1490);

        // Can still insert
        db.execute("INSERT INTO t5 VALUES (200, 2000)", ()).unwrap();
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t5"), 151);

        db.close().unwrap();
    }
}

// ============================================================================
// 6. Recovery preserves schema evolution
// ============================================================================

#[test]
fn test_recovery_preserves_schema_evolution() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_evo", dir.path().display());

    // Session 1: create table, insert, checkpoint, add column, insert more, close
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t6 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t6 VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute("INSERT INTO t6 VALUES (2, 'Bob')", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Evolve schema
        db.execute("ALTER TABLE t6 ADD COLUMN email TEXT DEFAULT 'unknown'", ())
            .unwrap();

        // Insert rows with new schema
        db.execute("INSERT INTO t6 VALUES (3, 'Carol', 'carol@test.com')", ())
            .unwrap();
        db.execute("INSERT INTO t6 VALUES (4, 'Dave', 'dave@test.com')", ())
            .unwrap();

        // Close triggers force-seal for remaining hot rows
        db.close().unwrap();
    }

    // Session 2: verify schema survived
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t6"), 4);

        // Column email should exist
        let email3: String = db
            .query_one("SELECT email FROM t6 WHERE id = 3", ())
            .unwrap();
        assert_eq!(email3, "carol@test.com");

        // Cold rows from before ADD COLUMN should have default
        let email1: String = db
            .query_one("SELECT email FROM t6 WHERE id = 1", ())
            .unwrap();
        assert_eq!(email1, "unknown");

        db.close().unwrap();
    }

    // Session 3: reopen again (no new data), verify still correct
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t6"), 4);

        let email2: String = db
            .query_one("SELECT email FROM t6 WHERE id = 2", ())
            .unwrap();
        assert_eq!(email2, "unknown");

        let email4: String = db
            .query_one("SELECT email FROM t6 WHERE id = 4", ())
            .unwrap();
        assert_eq!(email4, "dave@test.com");

        // Can insert with evolved schema
        db.execute("INSERT INTO t6 VALUES (5, 'Eve', 'eve@test.com')", ())
            .unwrap();
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t6"), 5);

        db.close().unwrap();
    }
}

// ============================================================================
// 7. Column projection after schema evolution
// ============================================================================

#[test]
fn test_column_projection_after_schema_evolution() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_proj", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t7 (id INTEGER PRIMARY KEY, a TEXT NOT NULL, b INTEGER NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t7 VALUES (1, 'x', 10)", ())
            .unwrap();
        db.execute("INSERT INTO t7 VALUES (2, 'y', 20)", ())
            .unwrap();
        db.execute("INSERT INTO t7 VALUES (3, 'z', 30)", ())
            .unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add new column (nullable, no default)
        db.execute("ALTER TABLE t7 ADD COLUMN c TEXT", ()).unwrap();

        // Insert row with new column
        db.execute("INSERT INTO t7 VALUES (4, 'w', 40, 'new')", ())
            .unwrap();

        // Verify via IS NULL that cold rows have NULL for c
        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t7 WHERE c IS NULL"),
            3,
            "Cold rows 1-3 should have NULL c"
        );
        assert_eq!(
            qi64(&db, "SELECT COUNT(*) FROM t7 WHERE c IS NOT NULL"),
            1,
            "Only row 4 should have non-NULL c"
        );

        // Projection: SELECT id, c FROM t7 ORDER BY id
        let mut rows = db.query("SELECT id, c FROM t7 ORDER BY id", ()).unwrap();

        // Row 4 should have 'new' for c
        let mut found_new = false;
        for _ in 0..4 {
            let row = rows.next().unwrap().unwrap();
            let id: i64 = row.get(0).unwrap();
            if id == 4 {
                let c_val: String = row.get(1).unwrap();
                assert_eq!(c_val, "new", "Row 4 c should be 'new'");
                found_new = true;
            }
        }
        assert!(found_new, "Should have found row 4 with c='new'");
        drop(rows);

        // Projection skipping the new column: SELECT id, b FROM t7
        let mut rows = db.query("SELECT id, b FROM t7 ORDER BY id", ()).unwrap();
        let r1 = rows.next().unwrap().unwrap();
        assert_eq!(r1.get::<i64>(0).unwrap(), 1);
        assert_eq!(r1.get::<i64>(1).unwrap(), 10);

        let r4 = rows.nth(2).unwrap().unwrap();
        assert_eq!(r4.get::<i64>(0).unwrap(), 4);
        assert_eq!(r4.get::<i64>(1).unwrap(), 40);
        drop(rows);

        db.close().unwrap();
    }

    // Session 2: verify projections work after reopen
    {
        let db = Database::open(&dsn).unwrap();

        // Projection including new column
        let null_c = qi64(&db, "SELECT COUNT(*) FROM t7 WHERE c IS NULL");
        assert_eq!(null_c, 3, "3 cold rows should have NULL c after reopen");

        let val_c: String = db.query_one("SELECT c FROM t7 WHERE id = 4", ()).unwrap();
        assert_eq!(val_c, "new");

        // Projection excluding new column still works
        assert_eq!(qi64(&db, "SELECT SUM(b) FROM t7"), 100); // 10+20+30+40

        db.close().unwrap();
    }
}

// ============================================================================
// 8. Aggregation on new column with mixed hot/cold
// ============================================================================

#[test]
fn test_aggregation_on_new_column_mixed_hot_cold() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_agg", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t8 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert rows that will become cold (no score column yet)
        db.execute("INSERT INTO t8 VALUES (1, 'a')", ()).unwrap();
        db.execute("INSERT INTO t8 VALUES (2, 'b')", ()).unwrap();
        db.execute("INSERT INTO t8 VALUES (3, 'c')", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add score column
        db.execute("ALTER TABLE t8 ADD COLUMN score FLOAT", ())
            .unwrap();

        // Insert new hot rows with score values
        db.execute("INSERT INTO t8 VALUES (4, 'd', 10.5)", ())
            .unwrap();
        db.execute("INSERT INTO t8 VALUES (5, 'e', 20.5)", ())
            .unwrap();
        db.execute("INSERT INTO t8 VALUES (6, 'f', 30.0)", ())
            .unwrap();

        // Total row count
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t8"), 6);

        // COUNT(score) should only count non-NULL rows (hot rows with explicit score)
        let count_score = qi64(&db, "SELECT COUNT(score) FROM t8");
        assert_eq!(
            count_score, 3,
            "COUNT(score) should only count the 3 hot rows with non-NULL score"
        );

        // SUM/AVG on hot rows only (cold rows have NULL score in aggregation path)
        let sum = qf64(&db, "SELECT SUM(score) FROM t8");
        assert!(
            (sum - 61.0).abs() < 0.01,
            "SUM(score) should be 61.0 (10.5+20.5+30.0), got {}",
            sum
        );

        let avg = qf64(&db, "SELECT AVG(score) FROM t8");
        let expected_avg = 61.0 / 3.0;
        assert!(
            (avg - expected_avg).abs() < 0.01,
            "AVG(score) should be ~{:.4}, got {}",
            expected_avg,
            avg
        );

        // MIN/MAX on the new column
        let min = qf64(&db, "SELECT MIN(score) FROM t8");
        assert!(
            (min - 10.5).abs() < 0.01,
            "MIN(score) should be 10.5, got {}",
            min
        );

        let max = qf64(&db, "SELECT MAX(score) FROM t8");
        assert!(
            (max - 30.0).abs() < 0.01,
            "MAX(score) should be 30.0, got {}",
            max
        );

        db.close().unwrap();
    }

    // Session 2: verify aggregations after reopen
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t8"), 6);
        assert_eq!(qi64(&db, "SELECT COUNT(score) FROM t8"), 3);

        let sum = qf64(&db, "SELECT SUM(score) FROM t8");
        assert!(
            (sum - 61.0).abs() < 0.01,
            "SUM(score) after reopen should be 61.0, got {}",
            sum
        );

        db.close().unwrap();
    }
}

// ============================================================================
// 9. Recovery with dirty shutdown simulation (drop without close)
// ============================================================================

#[test]
fn test_recovery_dirty_shutdown_drop_without_close() {
    let dir = tempfile::tempdir().unwrap();
    let db_name = "schema_dirty";
    let dsn = format!("file://{}/{}", dir.path().display(), db_name);

    // Session 1: insert, checkpoint, insert more, then drop
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t9 (id INTEGER PRIMARY KEY, val TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t9 VALUES ($1, $2)").unwrap();
        for i in 0..50i64 {
            stmt.execute((i, format!("val_{}", i))).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint seals rows to volume
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Insert more rows (hot only)
        db.execute("BEGIN", ()).unwrap();
        for i in 50..80i64 {
            stmt.execute((i, format!("val_{}", i))).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t9"), 80);

        // Drop without explicit close. The Drop impl calls close_engine()
        // which force-seals all rows.
        drop(db);
    }

    // Remove the lock file since drop may not clean it up
    remove_lock_file(dir.path(), db_name);

    // Session 2: reopen and verify recovery
    {
        let db = Database::open(&dsn).unwrap();

        let count = qi64(&db, "SELECT COUNT(*) FROM t9");
        assert_eq!(count, 80, "All 80 rows should survive drop+reopen");

        // Verify boundary values
        let v0: String = db.query_one("SELECT val FROM t9 WHERE id = 0", ()).unwrap();
        assert_eq!(v0, "val_0");

        let v79: String = db
            .query_one("SELECT val FROM t9 WHERE id = 79", ())
            .unwrap();
        assert_eq!(v79, "val_79");

        db.close().unwrap();
    }
}

// ============================================================================
// 10. ADD COLUMN with DEFAULT TEXT across multiple checkpoints + reopen
// ============================================================================

#[test]
fn test_add_column_default_text_multi_checkpoint_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_deftxt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t10 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Phase 1
        db.execute("INSERT INTO t10 VALUES (1, 'one')", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Phase 2: add TEXT column with default
        db.execute(
            "ALTER TABLE t10 ADD COLUMN status TEXT DEFAULT 'pending'",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t10 VALUES (2, 'two', 'active')", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Phase 3: add another column with default
        db.execute("ALTER TABLE t10 ADD COLUMN priority INTEGER DEFAULT 0", ())
            .unwrap();

        db.execute("INSERT INTO t10 VALUES (3, 'three', 'done', 5)", ())
            .unwrap();

        // Verify in-session
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t10"), 3);

        // Row 1: status='pending' (default), priority=0 (default) via individual reads
        let s1: String = db
            .query_one("SELECT status FROM t10 WHERE id = 1", ())
            .unwrap();
        assert_eq!(s1, "pending");
        assert_eq!(qi64(&db, "SELECT priority FROM t10 WHERE id = 1"), 0);

        // Row 2: status='active', priority=0 (default)
        let s2: String = db
            .query_one("SELECT status FROM t10 WHERE id = 2", ())
            .unwrap();
        assert_eq!(s2, "active");
        assert_eq!(qi64(&db, "SELECT priority FROM t10 WHERE id = 2"), 0);

        // Row 3: status='done', priority=5
        let s3: String = db
            .query_one("SELECT status FROM t10 WHERE id = 3", ())
            .unwrap();
        assert_eq!(s3, "done");
        assert_eq!(qi64(&db, "SELECT priority FROM t10 WHERE id = 3"), 5);

        db.close().unwrap();
    }

    // Session 2: verify all defaults and values survive
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t10"), 3);

        let s1: String = db
            .query_one("SELECT status FROM t10 WHERE id = 1", ())
            .unwrap();
        assert_eq!(s1, "pending", "Default text should persist for cold row");

        assert_eq!(
            qi64(&db, "SELECT priority FROM t10 WHERE id = 1"),
            0,
            "Default integer should persist for cold row"
        );

        let s2: String = db
            .query_one("SELECT status FROM t10 WHERE id = 2", ())
            .unwrap();
        assert_eq!(s2, "active");

        assert_eq!(qi64(&db, "SELECT priority FROM t10 WHERE id = 3"), 5);

        db.close().unwrap();
    }
}

// ============================================================================
// 11. Filtered queries on new column over cold data with DEFAULT
// ============================================================================

#[test]
fn test_filtered_query_on_new_column_cold_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_filter", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t11 (id INTEGER PRIMARY KEY, group_name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        // Insert and checkpoint
        for i in 1..=10 {
            db.execute(
                &format!(
                    "INSERT INTO t11 VALUES ({}, 'group_{}')",
                    i,
                    if i % 2 == 0 { "even" } else { "odd" }
                ),
                (),
            )
            .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Add score column with default 0
        db.execute("ALTER TABLE t11 ADD COLUMN score INTEGER DEFAULT 0", ())
            .unwrap();

        // Update some cold rows
        db.execute("UPDATE t11 SET score = 100 WHERE id <= 5", ())
            .unwrap();

        // Insert new hot rows
        for i in 11..=15 {
            db.execute(
                &format!("INSERT INTO t11 VALUES ({}, 'group_new', {})", i, i * 10),
                (),
            )
            .unwrap();
        }

        // Verify total count
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t11"), 15);

        // Filter on score >= 100: 5 updated cold rows + 5 new hot rows
        let high_score = qi64(&db, "SELECT COUNT(*) FROM t11 WHERE score >= 100");
        assert_eq!(
            high_score, 10,
            "5 updated cold + 5 new hot rows with score >= 100"
        );

        // Filter on score = 0 (default for unchanged cold rows 6-10)
        let default_score = qi64(&db, "SELECT COUNT(*) FROM t11 WHERE score = 0");
        assert_eq!(
            default_score, 5,
            "5 cold rows 6-10 should have default score 0"
        );

        db.close().unwrap();
    }

    // Session 2
    {
        let db = Database::open(&dsn).unwrap();

        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t11"), 15);
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t11 WHERE score >= 100"), 10);
        assert_eq!(qi64(&db, "SELECT COUNT(*) FROM t11 WHERE score = 0"), 5);

        db.close().unwrap();
    }
}

// ============================================================================
// 12. ORDER BY on new column with mixed cold/hot data
// ============================================================================

#[test]
fn test_order_by_new_column_mixed_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_order", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE t12 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO t12 VALUES (1, 'first')", ())
            .unwrap();
        db.execute("INSERT INTO t12 VALUES (2, 'second')", ())
            .unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        db.execute("ALTER TABLE t12 ADD COLUMN rank INTEGER", ())
            .unwrap();

        // Insert hot rows with rank values
        db.execute("INSERT INTO t12 VALUES (3, 'third', 1)", ())
            .unwrap();
        db.execute("INSERT INTO t12 VALUES (4, 'fourth', 2)", ())
            .unwrap();

        // ORDER BY rank ASC: should return all rows without crashing
        let ordered_count = count_rows(&db, "SELECT * FROM t12 ORDER BY rank ASC");
        assert_eq!(
            ordered_count, 4,
            "All rows should appear in ORDER BY result"
        );

        // ORDER BY rank DESC
        let ordered_count = count_rows(&db, "SELECT * FROM t12 ORDER BY rank DESC");
        assert_eq!(ordered_count, 4);

        // ORDER BY with WHERE filter on new column
        let filtered = count_rows(
            &db,
            "SELECT * FROM t12 WHERE rank IS NOT NULL ORDER BY rank",
        );
        assert_eq!(filtered, 2, "Only rows with non-NULL rank");

        db.close().unwrap();
    }
}

/// Test: CREATE INDEX on a column added after cold volumes exist (schema evolution).
/// Before the fix, validate_cold_unique and populate_index_from_cold used raw schema
/// column indices as physical volume indices, reading the wrong column data after
/// DROP COLUMN + ADD COLUMN.
#[test]
fn test_create_index_after_drop_add_column_with_cold_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_idx", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    // Create table with 3 columns: id, name, email
    db.execute(
        "CREATE TABLE t_idx (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
        (),
    )
    .unwrap();

    // Insert data and seal into cold volume
    db.execute(
        "INSERT INTO t_idx VALUES (1, 'Alice', 'alice@test.com')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t_idx VALUES (2, 'Bob', 'bob@test.com')", ())
        .unwrap();
    db.execute(
        "INSERT INTO t_idx VALUES (3, 'Carol', 'carol@test.com')",
        (),
    )
    .unwrap();

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Schema evolution: drop email, add phone
    // After this, schema is: id(0), name(1), phone(2)
    // But cold volume still has: id(0), name(1), email(2)
    db.execute("ALTER TABLE t_idx DROP COLUMN email", ())
        .unwrap();
    db.execute("ALTER TABLE t_idx ADD COLUMN phone TEXT", ())
        .unwrap();

    // Insert new row with phone value (goes to hot buffer)
    db.execute("INSERT INTO t_idx VALUES (4, 'Dave', '555-1234')", ())
        .unwrap();

    // CREATE INDEX on the new column should use column mapping,
    // not raw schema index (which would incorrectly read old email data)
    db.execute("CREATE INDEX idx_phone ON t_idx (phone)", ())
        .expect("CREATE INDEX after schema evolution should succeed");

    // Verify data is correct: cold rows have NULL phone, hot row has value
    let phone: String = db
        .query_one("SELECT COALESCE(phone, 'none') FROM t_idx WHERE id = 1", ())
        .unwrap();
    assert_eq!(phone, "none", "Cold row should have NULL phone");

    let phone: String = db
        .query_one("SELECT phone FROM t_idx WHERE id = 4", ())
        .unwrap();
    assert_eq!(phone, "555-1234", "Hot row should have phone value");

    db.close().unwrap();
}

/// Test: CREATE UNIQUE INDEX after schema evolution validates correctly.
/// Cold rows should have NULL for the new column (not old column's data).
#[test]
fn test_create_unique_index_after_schema_evolution() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/schema_uniq", dir.path().display());

    let db = Database::open(&dsn).unwrap();

    db.execute(
        "CREATE TABLE t_uniq (id INTEGER PRIMARY KEY, col_a TEXT, col_b TEXT)",
        (),
    )
    .unwrap();

    // Insert rows with duplicate col_b values and seal
    db.execute("INSERT INTO t_uniq VALUES (1, 'x', 'dup')", ())
        .unwrap();
    db.execute("INSERT INTO t_uniq VALUES (2, 'y', 'dup')", ())
        .unwrap();

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Drop col_b (which had duplicates), add col_c
    db.execute("ALTER TABLE t_uniq DROP COLUMN col_b", ())
        .unwrap();
    db.execute("ALTER TABLE t_uniq ADD COLUMN col_c TEXT", ())
        .unwrap();

    // Insert rows with unique col_c values
    db.execute("INSERT INTO t_uniq VALUES (3, 'z', 'unique_val')", ())
        .unwrap();

    // CREATE UNIQUE INDEX on col_c should succeed:
    // Cold rows have NULL col_c (skipped by unique check), hot row has unique value.
    // Without the fix, this would read old col_b data ('dup','dup') and wrongly fail.
    db.execute("CREATE UNIQUE INDEX idx_uniq_c ON t_uniq (col_c)", ())
        .expect("Unique index on new column should succeed (cold rows are NULL)");

    db.close().unwrap();
}
