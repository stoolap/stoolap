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

//! Correctness tests for the segment storage architecture.
//!
//! Validates that the new hot/cold separation preserves ACID guarantees:
//! - PK/UNIQUE constraints enforced across hot + cold
//! - FK references across hot + cold
//! - Snapshot isolation with concurrent segment reads
//! - Recovery after crash (WAL replay with cold data)
//! - DDL (rename/truncate/drop) with segment state
//! - Hot-only index behavior with cold data

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
        .unwrap_or(-1.0)
}

fn exec(db: &Database, sql: &str) {
    db.execute(sql, ()).unwrap();
}

fn exec_err(db: &Database, sql: &str) -> String {
    db.execute(sql, ()).unwrap_err().to_string()
}

// =========================================================================
// 1. PK / UNIQUE constraints across hot + cold
// =========================================================================

#[test]
fn test_pk_insert_duplicate_into_cold_data() {
    // Setup: create table, insert rows, snapshot (creates volume), reopen
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/pk_dup", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        exec(&db, "INSERT INTO t VALUES (1, 'cold_a')");
        exec(&db, "INSERT INTO t VALUES (2, 'cold_b')");
        exec(&db, "INSERT INTO t VALUES (3, 'cold_c')");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Reopen: data loads as volume (close() force-seals all rows into .vol files)
    {
        let db = Database::open(&dsn).unwrap();

        // Verify cold data is readable
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

        // PK duplicate against cold data must fail
        let err = exec_err(&db, "INSERT INTO t VALUES (1, 'dup')");
        assert!(
            err.contains("primary key") || err.contains("Primary key") || err.contains("PK"),
            "Expected PK constraint error, got: {}",
            err
        );

        // Non-duplicate should succeed
        exec(&db, "INSERT INTO t VALUES (4, 'hot_d')");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 4);

        db.close().unwrap();
    }
}

#[test]
fn test_pk_insert_after_cold_row_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/pk_del", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        exec(&db, "INSERT INTO t VALUES (1, 'a')");
        exec(&db, "INSERT INTO t VALUES (2, 'b')");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

        // Delete cold row
        exec(&db, "DELETE FROM t WHERE id = 1");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 1);

        // Re-insert same PK should now succeed (delete vector marks it gone)
        exec(&db, "INSERT INTO t VALUES (1, 'reinserted')");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

        db.close().unwrap();
    }
}

#[test]
fn test_unique_constraint_across_hot_cold() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/uniq", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE t (id INTEGER PRIMARY KEY, email TEXT NOT NULL)",
        );
        exec(&db, "CREATE UNIQUE INDEX idx_email ON t (email)");
        exec(&db, "INSERT INTO t VALUES (1, 'alice@example.com')");
        exec(&db, "INSERT INTO t VALUES (2, 'bob@example.com')");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Recreate index (it was persisted via WAL)
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

        // Unique duplicate against cold data must fail
        let err = exec_err(&db, "INSERT INTO t VALUES (3, 'alice@example.com')");
        assert!(
            err.to_lowercase().contains("unique"),
            "Expected UNIQUE constraint error, got: {}",
            err
        );

        // Different email should succeed
        exec(&db, "INSERT INTO t VALUES (3, 'carol@example.com')");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

        db.close().unwrap();
    }
}

#[test]
fn test_upsert_no_pk_unique_across_hot_cold() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/upsert_nopk_cold", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE metrics (
                host TEXT NOT NULL,
                ts TIMESTAMP NOT NULL,
                value INTEGER NOT NULL,
                UNIQUE(host, ts)
            )",
        );
        exec(
            &db,
            "INSERT INTO metrics VALUES ('server1', '2024-01-01 00:00:00', 10)",
        );
        exec(
            &db,
            "INSERT INTO metrics VALUES ('server1', '2024-01-01 01:00:00', 20)",
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        exec(
            &db,
            "INSERT INTO metrics VALUES ('server1', '2024-01-01 00:00:00', 99)
             ON CONFLICT (host, ts) DO UPDATE SET value = EXCLUDED.value",
        );

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 2);
        assert_eq!(
            query_i64(
                &db,
                "SELECT value FROM metrics WHERE host = 'server1' AND ts = '2024-01-01 00:00:00'"
            ),
            99
        );

        exec(
            &db,
            "INSERT INTO metrics VALUES ('server2', '2024-01-01 00:00:00', 7)
             ON CONFLICT (host, ts) DO UPDATE SET value = EXCLUDED.value",
        );
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 3);

        db.close().unwrap();
    }
}

// =========================================================================
// 2. UPDATE / DELETE on cold rows
// =========================================================================

#[test]
fn test_update_cold_row() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/upd", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 100)");
        exec(&db, "INSERT INTO t VALUES (2, 200)");
        exec(&db, "INSERT INTO t VALUES (3, 300)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Update a cold row
        exec(&db, "UPDATE t SET val = 999 WHERE id = 2");

        // Verify update
        assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 2"), 999);

        // Other rows unchanged
        assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 1"), 100);
        assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 3"), 300);

        // Count should stay the same
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

        // SUM should reflect the update
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 100 + 999 + 300);

        db.close().unwrap();
    }
}

#[test]
fn test_delete_cold_row_then_aggregate() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/del_agg", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        for i in 1..=10 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}0)"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 10);

        // Delete rows 3, 5, 7
        exec(&db, "DELETE FROM t WHERE id IN (3, 5, 7)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 7);

        // SUM should exclude deleted rows: total - 30 - 50 - 70 = 550 - 150 = 400
        // Total of 10+20+30+40+50+60+70+80+90+100 = 550
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 400);

        // MIN/MAX should still work
        assert_eq!(query_i64(&db, "SELECT MIN(val) FROM t"), 10);
        assert_eq!(query_i64(&db, "SELECT MAX(val) FROM t"), 100);

        db.close().unwrap();
    }
}

// =========================================================================
// 3. Recovery: WAL replay with cold data
// =========================================================================

#[test]
fn test_recovery_preserves_cold_updates() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/recovery", dir.path().display());

    // Session 1: Create and snapshot
    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        exec(&db, "INSERT INTO t VALUES (1, 'original')");
        exec(&db, "INSERT INTO t VALUES (2, 'original')");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Session 2: Update cold data (creates WAL entry), no snapshot
    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "UPDATE t SET val = 'updated' WHERE id = 1");
        exec(&db, "INSERT INTO t VALUES (3, 'new_hot')");
        // Close without snapshot — WAL has the update
        db.close().unwrap();
    }

    // Session 3: Reopen — WAL replay should restore the update
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

        let mut r = db.query("SELECT val FROM t WHERE id = 1", ()).unwrap();
        let val: String = r.next().unwrap().unwrap().get::<String>(0).unwrap();
        assert_eq!(val, "updated", "WAL replay should preserve cold row update");

        let mut r = db.query("SELECT val FROM t WHERE id = 3", ()).unwrap();
        let val: String = r.next().unwrap().unwrap().get::<String>(0).unwrap();
        assert_eq!(val, "new_hot", "WAL replay should preserve hot row insert");

        db.close().unwrap();
    }
}

#[test]
fn test_recovery_preserves_cold_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/rec_del", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 10)");
        exec(&db, "INSERT INTO t VALUES (2, 20)");
        exec(&db, "INSERT INTO t VALUES (3, 30)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "DELETE FROM t WHERE id = 2");
        db.close().unwrap(); // No snapshot — WAL only
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t"),
            2,
            "WAL replay should mark cold row as deleted"
        );
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 40); // 10 + 30
        db.close().unwrap();
    }
}

// =========================================================================
// 4. DDL with segment state: TRUNCATE, DROP, RENAME (via ALTER TABLE)
// =========================================================================

#[test]
fn test_truncate_clears_cold_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/trunc", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        for i in 1..=100 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i})"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 100);

        exec(&db, "TRUNCATE TABLE t");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 0);

        // Insert after truncate should work (PK 1 was in cold, now gone)
        exec(&db, "INSERT INTO t VALUES (1, 999)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 1);
        assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 1"), 999);

        db.close().unwrap();
    }
}

#[test]
fn test_drop_table_clears_cold_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/drop", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        for i in 1..=50 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i})"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 50);

        exec(&db, "DROP TABLE t");

        // Recreate with same name
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 0);

        exec(&db, "INSERT INTO t VALUES (1, 100)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 1);

        db.close().unwrap();
    }
}

// =========================================================================
// 5. Query correctness: hot + cold merge
// =========================================================================

#[test]
fn test_scan_merges_hot_and_cold_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/merge", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        // Cold data: ids 1-5
        for i in 1..=5 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}00)"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Hot data: ids 6-10
        for i in 6..=10 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}00)"));
        }

        // Total should be 10
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 10);

        // ORDER BY should work across hot + cold
        let mut r = db.query("SELECT id FROM t ORDER BY id", ()).unwrap();
        let mut ids = Vec::new();
        while let Some(Ok(row)) = r.next() {
            ids.push(row.get::<i64>(0).unwrap());
        }
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // WHERE filter should work on cold data
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t WHERE val >= 500"), 6); // ids 5-10

        // Aggregation across hot + cold
        assert_eq!(
            query_i64(&db, "SELECT SUM(val) FROM t"),
            (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) * 100
        );

        db.close().unwrap();
    }
}

#[test]
fn test_cold_data_with_where_filter() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/filter", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE t (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        );
        exec(&db, "INSERT INTO t VALUES (1, 'books', 10.0)");
        exec(&db, "INSERT INTO t VALUES (2, 'electronics', 500.0)");
        exec(&db, "INSERT INTO t VALUES (3, 'books', 25.0)");
        exec(&db, "INSERT INTO t VALUES (4, 'electronics', 999.0)");
        exec(&db, "INSERT INTO t VALUES (5, 'books', 15.0)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Text equality filter on cold data (uses dictionary pre-filter)
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t WHERE category = 'books'"),
            3
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t WHERE category = 'electronics'"),
            2
        );

        // Range filter on cold data
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t WHERE price > 20.0"),
            3
        );

        // Combined filter
        assert_eq!(
            query_i64(
                &db,
                "SELECT COUNT(*) FROM t WHERE category = 'books' AND price > 12.0"
            ),
            2
        );

        db.close().unwrap();
    }
}

// =========================================================================
// 6. Hot-only index behavior
// =========================================================================

#[test]
fn test_index_scan_with_mixed_hot_cold() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/idx_mix", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "CREATE INDEX idx_val ON t (val)");
        for i in 1..=5 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}0)"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Add hot rows
        for i in 6..=10 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}0)"));
        }

        // Query using index-friendly predicate: val = 30 (cold) and val = 80 (hot)
        assert_eq!(
            query_i64(&db, "SELECT id FROM t WHERE val = 30"),
            3,
            "Should find cold row via segment scan"
        );
        assert_eq!(
            query_i64(&db, "SELECT id FROM t WHERE val = 80"),
            8,
            "Should find hot row via index"
        );

        // Range query across hot + cold
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t WHERE val >= 40 AND val <= 70"),
            4
        );

        db.close().unwrap();
    }
}

// =========================================================================
// 7. Edge cases
// =========================================================================

#[test]
fn test_empty_table_with_segments_after_full_delete() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/empty_seg", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 10)");
        exec(&db, "INSERT INTO t VALUES (2, 20)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

        // Delete all cold rows
        exec(&db, "DELETE FROM t WHERE id IN (1, 2)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 0);

        // Insert new rows (should not conflict with deleted cold PKs)
        exec(&db, "INSERT INTO t VALUES (1, 100)");
        exec(&db, "INSERT INTO t VALUES (2, 200)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 300);

        db.close().unwrap();
    }
}

#[test]
fn test_multiple_reopens_preserve_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/multi_reopen", dir.path().display());

    // Session 1
    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 10)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Session 2: add more data
    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "INSERT INTO t VALUES (2, 20)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Session 3: add more data
    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "INSERT INTO t VALUES (3, 30)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Session 4: verify all data
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 60);
        db.close().unwrap();
    }
}

// =========================================================================
// 8. FK references across hot + cold
// =========================================================================

#[test]
fn test_fk_insert_child_referencing_cold_parent() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/fk_cold", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        );
        exec(
            &db,
            "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parents(id))",
        );
        exec(&db, "INSERT INTO parents VALUES (1, 'alice')");
        exec(&db, "INSERT INTO parents VALUES (2, 'bob')");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Parents are now in cold storage. FK insert referencing cold parent
        // should succeed because the parent row exists in segments.
        exec(&db, "INSERT INTO children VALUES (10, 1)");
        exec(&db, "INSERT INTO children VALUES (20, 2)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM children"), 2);

        // FK insert referencing non-existent parent should fail
        let err = exec_err(&db, "INSERT INTO children VALUES (30, 999)");
        assert!(
            err.to_lowercase().contains("foreign key") || err.to_lowercase().contains("constraint"),
            "Expected FK constraint error, got: {}",
            err
        );

        db.close().unwrap();
    }
}

#[test]
fn test_fk_delete_cold_parent_with_children() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/fk_del", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        );
        exec(
            &db,
            "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parents(id))",
        );
        exec(&db, "INSERT INTO parents VALUES (1, 'alice')");
        exec(&db, "INSERT INTO parents VALUES (2, 'bob')");
        exec(&db, "INSERT INTO children VALUES (10, 1)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Deleting a cold parent with children should fail
        let err = exec_err(&db, "DELETE FROM parents WHERE id = 1");
        assert!(
            err.to_lowercase().contains("foreign key")
                || err.to_lowercase().contains("constraint")
                || err.to_lowercase().contains("referenced"),
            "Expected FK constraint error on delete, got: {}",
            err
        );

        // Deleting a cold parent without children should succeed
        exec(&db, "DELETE FROM parents WHERE id = 2");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM parents"), 1);

        db.close().unwrap();
    }
}

// =========================================================================
// 9. Snapshot isolation with cold data
// =========================================================================

#[test]
fn test_snapshot_txn_reads_cold_data_correctly() {
    // Verifies that a SNAPSHOT isolation transaction can read cold (volume) data.
    // True concurrent isolation testing requires multi-threaded access;
    // this test validates cold data visibility within a snapshot transaction.
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/snap_iso", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        for i in 1..=5 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}0)"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 5);

        // Snapshot transaction reads cold data
        exec(&db, "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT");
        let snap_count = query_i64(&db, "SELECT COUNT(*) FROM t");
        let snap_sum = query_i64(&db, "SELECT SUM(val) FROM t");
        assert_eq!(snap_count, 5);
        assert_eq!(snap_sum, 150); // 10+20+30+40+50
        exec(&db, "COMMIT");

        // Hot insert after snapshot commit is visible in auto-commit reads
        exec(&db, "INSERT INTO t VALUES (6, 60)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 6);

        db.close().unwrap();
    }
}

#[test]
fn test_cold_data_visible_after_delete() {
    // Verifies that cold data deletion is reflected in subsequent reads.
    // True concurrent isolation testing requires multi-threaded access;
    // this test validates cold delete visibility in auto-commit reads.
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/snap_del", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        for i in 1..=5 {
            exec(&db, &format!("INSERT INTO t VALUES ({i}, {i}0)"));
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Verify initial cold data
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 5);

        // Delete a cold row
        exec(&db, "DELETE FROM t WHERE id = 3");

        // Subsequent read should see the delete
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 4);

        db.close().unwrap();
    }
}

// =========================================================================
// 10. ALTER TABLE RENAME with segment state
// =========================================================================

#[test]
fn test_rename_table_with_cold_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/rename", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE old_name (id INTEGER PRIMARY KEY, val INTEGER)",
        );
        exec(&db, "INSERT INTO old_name VALUES (1, 10)");
        exec(&db, "INSERT INTO old_name VALUES (2, 20)");
        exec(&db, "INSERT INTO old_name VALUES (3, 30)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        // Cold data is under "old_name"
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM old_name"), 3);

        // Rename
        exec(&db, "ALTER TABLE old_name RENAME TO new_name");

        // Old name should not work
        let err = exec_err(&db, "SELECT COUNT(*) FROM old_name");
        assert!(
            err.to_lowercase().contains("not found")
                || err.to_lowercase().contains("does not exist")
                || err.to_lowercase().contains("no such table"),
            "Expected table-not-found error, got: {}",
            err
        );

        // New name should have the cold data
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM new_name"), 3);
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM new_name"), 60);

        // Insert into renamed table should respect PK constraints from cold data
        let err = exec_err(&db, "INSERT INTO new_name VALUES (1, 99)");
        assert!(
            err.contains("primary key") || err.contains("Primary key") || err.contains("PK"),
            "Expected PK constraint error after rename, got: {}",
            err
        );

        // New PK should work
        exec(&db, "INSERT INTO new_name VALUES (4, 40)");
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM new_name"), 4);

        db.close().unwrap();
    }
}

#[test]
fn test_rename_table_with_cold_data_persists_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/rename_persist", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(
            &db,
            "CREATE TABLE original (id INTEGER PRIMARY KEY, val INTEGER)",
        );
        exec(&db, "INSERT INTO original VALUES (1, 10)");
        exec(&db, "INSERT INTO original VALUES (2, 20)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "ALTER TABLE original RENAME TO renamed");
        exec(&db, "INSERT INTO renamed VALUES (3, 30)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Reopen and verify renamed table has all data
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM renamed"), 3);
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM renamed"), 60);
        db.close().unwrap();
    }
}

// =========================================================================
// 11. ROLLBACK correctness with cold data
// =========================================================================

#[test]
fn test_rollback_restores_cold_row_visibility() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/rollback", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 10)");
        exec(&db, "INSERT INTO t VALUES (2, 20)");
        exec(&db, "INSERT INTO t VALUES (3, 30)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 3);

        // Delete a cold row inside a transaction, then rollback
        exec(&db, "BEGIN");
        exec(&db, "DELETE FROM t WHERE id = 2");
        // Within the transaction, the row should be gone
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);
        exec(&db, "ROLLBACK");

        // After rollback, the cold row should be visible again
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM t"),
            3,
            "ROLLBACK should restore cold row visibility"
        );
        assert_eq!(query_i64(&db, "SELECT SUM(val) FROM t"), 60);

        db.close().unwrap();
    }
}

#[test]
fn test_rollback_update_on_cold_row() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/rb_upd", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        exec(&db, "INSERT INTO t VALUES (1, 100)");
        exec(&db, "INSERT INTO t VALUES (2, 200)");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        exec(&db, "BEGIN");
        exec(&db, "UPDATE t SET val = 999 WHERE id = 1");
        assert_eq!(query_i64(&db, "SELECT val FROM t WHERE id = 1"), 999);
        exec(&db, "ROLLBACK");

        // Original cold value should be restored
        assert_eq!(
            query_i64(&db, "SELECT val FROM t WHERE id = 1"),
            100,
            "ROLLBACK should restore original cold row value"
        );
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM t"), 2);

        db.close().unwrap();
    }
}

// =========================================================================
// Regression: ON CONFLICT DO UPDATE on cold rows with UNIQUE indexes
// =========================================================================

/// Regression: insert_discard(old_row) during cold-row upsert raises
/// UniqueConstraint (not PrimaryKeyConstraint) when the hot index already
/// has the unique key values. Only PrimaryKeyConstraint was caught.
#[test]
fn test_upsert_on_cold_unique_constraint() {
    let db = Database::open("memory://upsert_cold_unique").unwrap();

    exec(
        &db,
        "CREATE TABLE candles (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        time TIMESTAMP NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        open FLOAT,
        close FLOAT,
        volume FLOAT
    )",
    );
    exec(
        &db,
        "CREATE UNIQUE INDEX idx_unique ON candles (exchange, symbol, time)",
    );

    // Insert initial data
    exec(
        &db,
        "INSERT INTO candles (time, exchange, symbol, open, close, volume) VALUES
         ('2026-01-01T00:00:00Z', 'binance', 'BTCUSDT', 100.0, 105.0, 1000.0),
         ('2026-01-01T04:00:00Z', 'binance', 'BTCUSDT', 110.0, 115.0, 1100.0)",
    );

    // Seal into cold volumes
    exec(&db, "PRAGMA CHECKPOINT");

    // Upsert targeting a cold row
    let result = db.execute(
        "INSERT INTO candles (time, exchange, symbol, open, close, volume)
         VALUES ('2026-01-01T00:00:00Z', 'binance', 'BTCUSDT', 200.0, 205.0, 2000.0)
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
            open = EXCLUDED.open, close = EXCLUDED.close, volume = EXCLUDED.volume",
        (),
    );
    assert!(
        result.is_ok(),
        "Upsert on cold row with UNIQUE index should succeed: {:?}",
        result.err()
    );

    // Upsert the same row again (now hot)
    let result2 = db.execute(
        "INSERT INTO candles (time, exchange, symbol, open, close, volume)
         VALUES ('2026-01-01T00:00:00Z', 'binance', 'BTCUSDT', 300.0, 305.0, 3000.0)
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
            open = EXCLUDED.open, close = EXCLUDED.close, volume = EXCLUDED.volume",
        (),
    );
    assert!(
        result2.is_ok(),
        "Second upsert on same row should succeed: {:?}",
        result2.err()
    );

    assert_eq!(
        query_f64(&db, "SELECT open FROM candles WHERE exchange = 'binance' AND symbol = 'BTCUSDT' AND time = '2026-01-01T00:00:00Z'"),
        300.0
    );
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM candles"), 2);
    db.close().unwrap();
}

/// INSERT ... SELECT ... ON CONFLICT targeting cold rows. Simulates the
/// exact production scenario: aggregating raw data into a summary table.
#[test]
fn test_insert_select_upsert_on_cold_unique() {
    let db = Database::open("memory://insert_select_upsert_cold").unwrap();

    exec(
        &db,
        "CREATE TABLE raw (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        time TIMESTAMP NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        val FLOAT
    )",
    );
    exec(
        &db,
        "CREATE TABLE agg (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        time TIMESTAMP NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        total FLOAT
    )",
    );
    exec(
        &db,
        "CREATE UNIQUE INDEX idx_agg ON agg (exchange, symbol, time)",
    );

    // Insert raw data and aggregate
    exec(
        &db,
        "INSERT INTO raw (time, exchange, symbol, val) VALUES
         ('2026-01-01T01:00:00Z', 'binance', 'BTC', 100.0),
         ('2026-01-01T02:00:00Z', 'binance', 'BTC', 200.0)",
    );
    exec(
        &db,
        "INSERT INTO agg (time, exchange, symbol, total)
         SELECT '2026-01-01T00:00:00Z', exchange, symbol, SUM(val)
         FROM raw WHERE exchange = 'binance' AND symbol = 'BTC'
         GROUP BY exchange, symbol
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET total = EXCLUDED.total",
    );

    // Seal into cold
    exec(&db, "PRAGMA CHECKPOINT");

    // Add more raw data and re-aggregate (upsert on cold)
    exec(
        &db,
        "INSERT INTO raw (time, exchange, symbol, val) VALUES
         ('2026-01-01T03:00:00Z', 'binance', 'BTC', 300.0)",
    );

    let result = db.execute(
        "INSERT INTO agg (time, exchange, symbol, total)
         SELECT '2026-01-01T00:00:00Z', exchange, symbol, SUM(val)
         FROM raw WHERE exchange = 'binance' AND symbol = 'BTC'
         GROUP BY exchange, symbol
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET total = EXCLUDED.total",
        (),
    );
    assert!(
        result.is_ok(),
        "INSERT...SELECT upsert on cold row should succeed: {:?}",
        result.err()
    );

    assert_eq!(
        query_f64(
            &db,
            "SELECT total FROM agg WHERE exchange = 'binance' AND symbol = 'BTC'"
        ),
        600.0
    );
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM agg"), 1);
    db.close().unwrap();
}
