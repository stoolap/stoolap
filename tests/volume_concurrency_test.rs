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

//! Concurrent operations on cold (volume-backed) data.
//!
//! After PRAGMA CHECKPOINT seals rows into immutable volumes, concurrent
//! transactions must coordinate via try_claim_row to prevent lost updates
//! and lost deletes. Snapshot isolation must honour versioned tombstones.
//!
//! Test categories:
//! 1. Cold-row write conflict (try_claim_row)
//! 2. Snapshot isolation with versioned tombstones
//! 3. Pending tombstone isolation
//! 4. Cold row scan during concurrent modifications

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use stoolap::{Database, IsolationLevel};

// ============================================================================
// Helpers
// ============================================================================

/// Create a file-backed database, populate a table, and checkpoint to cold.
fn setup_cold_db(name: &str, rows: &[(i64, &str)]) -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let dsn = format!("file://{}/{}", dir.path().display(), name);
    let db = Database::open(&dsn).expect("failed to open database");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT NOT NULL)",
        (),
    )
    .expect("failed to create table");

    for &(id, val) in rows {
        db.execute(&format!("INSERT INTO t VALUES ({}, '{}')", id, val), ())
            .expect("failed to insert row");
    }

    db.execute("PRAGMA CHECKPOINT", ())
        .expect("PRAGMA CHECKPOINT failed");

    (dir, db)
}

/// Query a single i64 value (returns -1 on no rows).
fn q_i64(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .expect("query failed")
        .next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

/// Query a single String value via a Transaction.
fn tx_q_str(tx: &mut stoolap::api::transaction::Transaction, sql: &str) -> Option<String> {
    let mut rows = tx.query(sql, ()).expect("tx query failed");
    rows.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<String>(0).ok())
}

/// Query a single i64 value via a Transaction.
fn tx_q_i64(tx: &mut stoolap::api::transaction::Transaction, sql: &str) -> i64 {
    tx.query_one::<i64, _>(sql, ()).unwrap_or(-1)
}

// ============================================================================
// 1. Cold-row write conflict (try_claim_row)
// ============================================================================

/// Two threads both try to UPDATE the same cold row. One succeeds, one gets a
/// conflict error (row has uncommitted changes from another transaction).
#[test]
fn test_cold_update_vs_update_conflict() {
    let (_dir, db) = setup_cold_db("upd_upd", &[(1, "orig")]);

    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));

    let handles: Vec<_> = (0..2)
        .map(|i| {
            let db_clone = db.clone();
            let sc = Arc::clone(&success_count);
            let fc = Arc::clone(&failure_count);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                let new_val = format!("thread{}", i);
                db_clone.execute("BEGIN", ()).expect("BEGIN failed");

                // Synchronize so both threads attempt the UPDATE close together.
                bar.wait();

                let result = db_clone.execute(
                    &format!("UPDATE t SET val = '{}' WHERE id = 1", new_val),
                    (),
                );

                match result {
                    Ok(_) => {
                        // Try to commit; commit itself may also fail on conflict.
                        match db_clone.execute("COMMIT", ()) {
                            Ok(_) => {
                                sc.fetch_add(1, Ordering::SeqCst);
                            }
                            Err(_) => {
                                let _ = db_clone.execute("ROLLBACK", ());
                                fc.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                    Err(_) => {
                        let _ = db_clone.execute("ROLLBACK", ());
                        fc.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);

    assert_eq!(
        successes + failures,
        2,
        "expected exactly 2 outcomes, got {} successes + {} failures",
        successes,
        failures
    );
    assert!(
        successes >= 1,
        "at least one thread must succeed, but got {} successes",
        successes
    );
    // At most one should succeed because they contend on the same cold row.
    // However, if scheduling means one fully commits before the other even
    // starts its UPDATE, both may succeed sequentially. So we only assert
    // successes >= 1 and the total is 2.

    // Final state: exactly one committed value should be visible.
    let count = q_i64(&db, "SELECT COUNT(*) FROM t WHERE id = 1");
    assert_eq!(count, 1, "row id=1 must still exist exactly once");
    db.close().ok();
}

/// Two threads both try to DELETE the same cold row.
#[test]
fn test_cold_delete_vs_delete_conflict() {
    let (_dir, db) = setup_cold_db("del_del", &[(1, "victim"), (2, "survivor")]);

    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));

    let handles: Vec<_> = (0..2)
        .map(|_| {
            let db_clone = db.clone();
            let sc = Arc::clone(&success_count);
            let fc = Arc::clone(&failure_count);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                db_clone.execute("BEGIN", ()).expect("BEGIN failed");
                bar.wait();

                let result = db_clone.execute("DELETE FROM t WHERE id = 1", ());
                match result {
                    Ok(_) => match db_clone.execute("COMMIT", ()) {
                        Ok(_) => {
                            sc.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(_) => {
                            let _ = db_clone.execute("ROLLBACK", ());
                            fc.fetch_add(1, Ordering::SeqCst);
                        }
                    },
                    Err(_) => {
                        let _ = db_clone.execute("ROLLBACK", ());
                        fc.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);
    assert_eq!(
        successes + failures,
        2,
        "expected 2 outcomes, got {} + {}",
        successes,
        failures
    );
    assert!(
        successes >= 1,
        "at least one delete must succeed, got {} successes",
        successes
    );

    // Row 1 must be gone, row 2 must survive.
    let c1 = q_i64(&db, "SELECT COUNT(*) FROM t WHERE id = 1");
    assert_eq!(c1, 0, "row id=1 should have been deleted");
    let c2 = q_i64(&db, "SELECT COUNT(*) FROM t WHERE id = 2");
    assert_eq!(c2, 1, "row id=2 should survive");
    db.close().ok();
}

/// One thread UPDATEs a cold row while another DELETEs the same row.
/// Only one should succeed.
#[test]
fn test_cold_update_vs_delete_conflict() {
    let (_dir, db) = setup_cold_db("upd_del", &[(1, "orig")]);

    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));

    // Thread 0: UPDATE, Thread 1: DELETE
    let handles: Vec<_> = (0..2)
        .map(|i| {
            let db_clone = db.clone();
            let sc = Arc::clone(&success_count);
            let fc = Arc::clone(&failure_count);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                db_clone.execute("BEGIN", ()).expect("BEGIN failed");
                bar.wait();

                let result = if i == 0 {
                    db_clone.execute("UPDATE t SET val = 'updated' WHERE id = 1", ())
                } else {
                    db_clone.execute("DELETE FROM t WHERE id = 1", ())
                };

                match result {
                    Ok(_) => match db_clone.execute("COMMIT", ()) {
                        Ok(_) => {
                            sc.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(_) => {
                            let _ = db_clone.execute("ROLLBACK", ());
                            fc.fetch_add(1, Ordering::SeqCst);
                        }
                    },
                    Err(_) => {
                        let _ = db_clone.execute("ROLLBACK", ());
                        fc.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);
    assert_eq!(
        successes + failures,
        2,
        "expected 2 outcomes, got {} + {}",
        successes,
        failures
    );
    assert!(
        successes >= 1,
        "at least one operation must succeed, got {} successes",
        successes
    );

    // The row is either updated or deleted, but it must be consistent.
    let count = q_i64(&db, "SELECT COUNT(*) FROM t WHERE id = 1");
    assert!(
        count == 0 || count == 1,
        "row count for id=1 must be 0 or 1, got {}",
        count
    );
    db.close().ok();
}

// ============================================================================
// 2. Snapshot isolation with versioned tombstones
// ============================================================================

/// A snapshot transaction started before a DELETE on a cold row should still
/// see the row.
#[test]
fn test_snapshot_sees_cold_row_before_delete() {
    let (_dir, db) = setup_cold_db("snap_del", &[(1, "alive"), (2, "also_alive")]);

    // Start snapshot txn before any modifications.
    let mut snap_tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("failed to begin snapshot txn");

    // Auto-commit DELETE from another session.
    db.execute("DELETE FROM t WHERE id = 1", ())
        .expect("auto-commit DELETE failed");

    // Snapshot should still see row id=1.
    let count = tx_q_i64(&mut snap_tx, "SELECT COUNT(*) FROM t");
    assert_eq!(count, 2, "snapshot should see both rows, but got {}", count);

    let val = tx_q_str(&mut snap_tx, "SELECT val FROM t WHERE id = 1");
    assert_eq!(
        val.as_deref(),
        Some("alive"),
        "snapshot should see 'alive' for id=1, got {:?}",
        val
    );

    snap_tx.rollback().expect("rollback failed");

    // After snapshot ends, auto-commit sees the deletion.
    let count_after = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(
        count_after, 1,
        "auto-commit should see 1 row after deletion"
    );
    db.close().ok();
}

/// A snapshot transaction started before an UPDATE on a cold row should see
/// the old value.
#[test]
fn test_snapshot_sees_old_cold_value_before_update() {
    let (_dir, db) = setup_cold_db("snap_upd", &[(1, "old_value")]);

    let mut snap_tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("failed to begin snapshot txn");

    // Auto-commit UPDATE from another session.
    db.execute("UPDATE t SET val = 'new_value' WHERE id = 1", ())
        .expect("auto-commit UPDATE failed");

    // Snapshot should see old value.
    let val = tx_q_str(&mut snap_tx, "SELECT val FROM t WHERE id = 1");
    assert_eq!(
        val.as_deref(),
        Some("old_value"),
        "snapshot should see 'old_value', got {:?}",
        val
    );

    snap_tx.rollback().expect("rollback failed");

    // Auto-commit sees updated value.
    let current: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query_one failed");
    assert_eq!(current, "new_value", "auto-commit should see 'new_value'");
    db.close().ok();
}

/// Multiple snapshot transactions at different points in time see different
/// states of the same cold rows.
#[test]
fn test_multiple_snapshots_different_states() {
    let (_dir, db) = setup_cold_db("multi_snap", &[(1, "v1"), (2, "v1"), (3, "v1")]);

    // Snapshot A: sees all 3 rows with 'v1'.
    let mut snap_a = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("snap_a begin failed");

    // Modify: delete row 3, update row 1.
    db.execute("DELETE FROM t WHERE id = 3", ())
        .expect("delete id=3 failed");
    db.execute("UPDATE t SET val = 'v2' WHERE id = 1", ())
        .expect("update id=1 failed");

    // Snapshot B: sees rows 1 (v2) and 2 (v1), no row 3.
    let mut snap_b = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("snap_b begin failed");

    // Modify again: delete row 2.
    db.execute("DELETE FROM t WHERE id = 2", ())
        .expect("delete id=2 failed");

    // Verify Snapshot A: all 3 rows, all 'v1'.
    let count_a = tx_q_i64(&mut snap_a, "SELECT COUNT(*) FROM t");
    assert_eq!(count_a, 3, "snap_a should see 3 rows, got {}", count_a);
    let val_a1 = tx_q_str(&mut snap_a, "SELECT val FROM t WHERE id = 1");
    assert_eq!(val_a1.as_deref(), Some("v1"), "snap_a id=1 should be 'v1'");
    let val_a3 = tx_q_str(&mut snap_a, "SELECT val FROM t WHERE id = 3");
    assert_eq!(val_a3.as_deref(), Some("v1"), "snap_a id=3 should be 'v1'");

    // Verify Snapshot B: 2 rows (1 and 2), id=1 is 'v2'.
    let count_b = tx_q_i64(&mut snap_b, "SELECT COUNT(*) FROM t");
    assert_eq!(count_b, 2, "snap_b should see 2 rows, got {}", count_b);
    let val_b1 = tx_q_str(&mut snap_b, "SELECT val FROM t WHERE id = 1");
    assert_eq!(val_b1.as_deref(), Some("v2"), "snap_b id=1 should be 'v2'");
    let val_b2 = tx_q_str(&mut snap_b, "SELECT val FROM t WHERE id = 2");
    assert_eq!(val_b2.as_deref(), Some("v1"), "snap_b id=2 should be 'v1'");

    // Verify auto-commit (latest): only row 1 with 'v2'.
    let count_latest = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(
        count_latest, 1,
        "auto-commit should see 1 row, got {}",
        count_latest
    );
    let val_latest: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query_one failed");
    assert_eq!(val_latest, "v2", "auto-commit id=1 should be 'v2'");

    snap_a.rollback().expect("snap_a rollback failed");
    snap_b.rollback().expect("snap_b rollback failed");
    db.close().ok();
}

/// Auto-commit transactions always see the latest committed state.
#[test]
fn test_autocommit_always_sees_latest() {
    let (_dir, db) = setup_cold_db("autocommit", &[(1, "orig")]);

    // Update cold row.
    db.execute("UPDATE t SET val = 'step1' WHERE id = 1", ())
        .expect("update step1 failed");
    let v1: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query failed");
    assert_eq!(v1, "step1", "auto-commit should see 'step1' after update");

    // Update again.
    db.execute("UPDATE t SET val = 'step2' WHERE id = 1", ())
        .expect("update step2 failed");
    let v2: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query failed");
    assert_eq!(
        v2, "step2",
        "auto-commit should see 'step2' after second update"
    );

    // Delete.
    db.execute("DELETE FROM t WHERE id = 1", ())
        .expect("delete failed");
    let count = q_i64(&db, "SELECT COUNT(*) FROM t WHERE id = 1");
    assert_eq!(count, 0, "auto-commit should see 0 rows after delete");

    db.close().ok();
}

// ============================================================================
// 3. Pending tombstone isolation
// ============================================================================

/// Txn A deletes a cold row (pending tombstone). Txn A sees the row as gone,
/// but Txn B (auto-commit) still sees it. After A commits, new reads see
/// the deletion.
#[test]
fn test_pending_tombstone_invisible_to_others() {
    let (_dir, db) = setup_cold_db("pending_ts", &[(1, "visible"), (2, "also_visible")]);

    // Txn A: begin, delete row 1.
    let db_a = db.clone();
    db_a.execute("BEGIN", ()).expect("BEGIN failed");
    db_a.execute("DELETE FROM t WHERE id = 1", ())
        .expect("DELETE in txn A failed");

    // From Txn A's perspective, row 1 is gone.
    let count_a = db_a
        .query("SELECT COUNT(*) FROM t", ())
        .expect("query in txn A failed")
        .next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1);
    assert_eq!(
        count_a, 1,
        "txn A should see 1 row (id=2 only), got {}",
        count_a
    );

    // From Txn B (auto-commit via the original handle), row 1 is still visible.
    let count_b = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(
        count_b, 2,
        "auto-commit should still see 2 rows while txn A is uncommitted, got {}",
        count_b
    );

    // Commit Txn A.
    db_a.execute("COMMIT", ()).expect("COMMIT failed");

    // After commit, auto-commit sees the deletion.
    let count_after = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(
        count_after, 1,
        "auto-commit should see 1 row after txn A commits, got {}",
        count_after
    );

    db.close().ok();
}

/// A pending UPDATE tombstone on a cold row should not be visible to other
/// transactions until committed.
#[test]
fn test_pending_update_invisible_to_others() {
    let (_dir, db) = setup_cold_db("pending_upd", &[(1, "original")]);

    // Txn A: begin, update row 1.
    let db_a = db.clone();
    db_a.execute("BEGIN", ()).expect("BEGIN failed");
    db_a.execute("UPDATE t SET val = 'modified' WHERE id = 1", ())
        .expect("UPDATE in txn A failed");

    // From Txn A, row 1 should have the new value.
    let val_a: String = db_a
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query in txn A failed");
    assert_eq!(
        val_a, "modified",
        "txn A should see its own update 'modified', got '{}'",
        val_a
    );

    // From another auto-commit handle, row 1 should still show the original.
    let val_b: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("auto-commit query failed");
    assert_eq!(
        val_b, "original",
        "auto-commit should see 'original' while txn A is uncommitted, got '{}'",
        val_b
    );

    // Commit txn A.
    db_a.execute("COMMIT", ()).expect("COMMIT failed");

    // Now auto-commit sees the updated value.
    let val_after: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("auto-commit query after commit failed");
    assert_eq!(
        val_after, "modified",
        "auto-commit should see 'modified' after commit, got '{}'",
        val_after
    );
    db.close().ok();
}

/// A pending DELETE that is rolled back should leave the cold row visible.
#[test]
fn test_pending_tombstone_rollback_preserves_row() {
    let (_dir, db) = setup_cold_db("pending_rb", &[(1, "keep_me")]);

    let db_a = db.clone();
    db_a.execute("BEGIN", ()).expect("BEGIN failed");
    db_a.execute("DELETE FROM t WHERE id = 1", ())
        .expect("DELETE failed");

    // Row should appear gone to txn A.
    let count_a = db_a
        .query("SELECT COUNT(*) FROM t WHERE id = 1", ())
        .expect("query failed")
        .next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1);
    assert_eq!(count_a, 0, "txn A should see 0 for deleted row");

    // Rollback.
    db_a.execute("ROLLBACK", ()).expect("ROLLBACK failed");

    // Row should be fully visible again.
    let val: String = db
        .query_one("SELECT val FROM t WHERE id = 1", ())
        .expect("query after rollback failed");
    assert_eq!(
        val, "keep_me",
        "row should be visible after rollback, got '{}'",
        val
    );
    db.close().ok();
}

// ============================================================================
// 4. Cold row scan during concurrent modifications
// ============================================================================

/// Start collecting scan results, while a concurrent thread updates/deletes
/// cold rows. The scan (auto-commit) should return a consistent snapshot
/// at its start time.
#[test]
fn test_scan_consistency_during_concurrent_mods() {
    // Insert more rows to make the scan longer and increase overlap probability.
    let rows: Vec<(i64, String)> = (1..=100).map(|i| (i, format!("row_{}", i))).collect();
    let row_refs: Vec<(i64, &str)> = rows.iter().map(|(id, val)| (*id, val.as_str())).collect();

    let (_dir, db) = setup_cold_db("scan_conc", &row_refs);

    // Pre-modification: verify all 100 rows present.
    let pre_count = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(pre_count, 100, "setup should have 100 rows");

    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: scanner. Reads all rows with ORDER BY id.
    let db_scan = db.clone();
    let bar1 = Arc::clone(&barrier);
    let scanner = thread::spawn(move || {
        bar1.wait();
        let result = db_scan
            .query("SELECT id, val FROM t ORDER BY id", ())
            .expect("scan query failed");
        let mut scanned_ids: Vec<i64> = Vec::new();
        for row_result in result {
            let row = row_result.expect("row iteration failed");
            let id: i64 = row.get(0).expect("get id failed");
            scanned_ids.push(id);
        }
        scanned_ids
    });

    // Thread 2: modifier. Deletes and updates some cold rows.
    let db_mod = db.clone();
    let bar2 = Arc::clone(&barrier);
    let modifier = thread::spawn(move || {
        bar2.wait();
        // Delete rows 10, 20, 30.
        for &id in &[10, 20, 30] {
            let _ = db_mod.execute(&format!("DELETE FROM t WHERE id = {}", id), ());
        }
        // Update rows 50, 60, 70.
        for &id in &[50, 60, 70] {
            let _ = db_mod.execute(
                &format!("UPDATE t SET val = 'changed_{}' WHERE id = {}", id, id),
                (),
            );
        }
    });

    let scanned_ids = scanner.join().expect("scanner thread panicked");
    modifier.join().expect("modifier thread panicked");

    // The scan should return a consistent set of IDs: either it saw
    // the pre-modification state (100 rows) or some committed subset,
    // but the IDs must be in sorted order and have no duplicates.
    let mut sorted = scanned_ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        scanned_ids.len(),
        "scan returned duplicate IDs: {:?}",
        scanned_ids
    );

    // The IDs must be monotonically increasing (we used ORDER BY id).
    for window in scanned_ids.windows(2) {
        assert!(
            window[0] < window[1],
            "scan result not sorted: {} >= {}",
            window[0],
            window[1]
        );
    }

    db.close().ok();
}

/// A snapshot-isolation scan must see a consistent view even when concurrent
/// auto-commit transactions modify cold rows.
#[test]
fn test_snapshot_scan_during_concurrent_deletes() {
    let rows: Vec<(i64, String)> = (1..=50).map(|i| (i, format!("val_{}", i))).collect();
    let row_refs: Vec<(i64, &str)> = rows.iter().map(|(id, val)| (*id, val.as_str())).collect();

    let (_dir, db) = setup_cold_db("snap_scan", &row_refs);

    // Start a snapshot transaction.
    let mut snap_tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .expect("snapshot begin failed");

    // Concurrent auto-commit deletes.
    for id in 1..=25 {
        db.execute(&format!("DELETE FROM t WHERE id = {}", id), ())
            .expect("concurrent delete failed");
    }

    // Snapshot scan should still see all 50 rows.
    let snap_count = tx_q_i64(&mut snap_tx, "SELECT COUNT(*) FROM t");
    assert_eq!(
        snap_count, 50,
        "snapshot should see all 50 rows despite concurrent deletes, got {}",
        snap_count
    );

    // Verify we can iterate and get the correct IDs.
    let rows_iter = snap_tx
        .query("SELECT id FROM t ORDER BY id", ())
        .expect("snapshot scan query failed");
    let mut ids: Vec<i64> = Vec::new();
    for row_result in rows_iter {
        let row = row_result.expect("row iteration failed");
        let id: i64 = row.get(0).expect("get id failed");
        ids.push(id);
    }
    assert_eq!(
        ids.len(),
        50,
        "snapshot scan should yield 50 rows, got {}",
        ids.len()
    );
    assert_eq!(ids[0], 1, "first id should be 1");
    assert_eq!(ids[49], 50, "last id should be 50");

    snap_tx.rollback().expect("rollback failed");

    // After ending the snapshot, auto-commit sees only 25 rows.
    let final_count = q_i64(&db, "SELECT COUNT(*) FROM t");
    assert_eq!(
        final_count, 25,
        "auto-commit should see 25 rows after deletes, got {}",
        final_count
    );
    db.close().ok();
}

/// Multiple concurrent writers updating different cold rows should all
/// succeed (no conflict since they target different row_ids).
#[test]
fn test_concurrent_updates_different_cold_rows() {
    let rows: Vec<(i64, String)> = (1..=10).map(|i| (i, format!("orig_{}", i))).collect();
    let row_refs: Vec<(i64, &str)> = rows.iter().map(|(id, val)| (*id, val.as_str())).collect();

    let (_dir, db) = setup_cold_db("diff_rows", &row_refs);

    let barrier = Arc::new(Barrier::new(10));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (1..=10)
        .map(|id| {
            let db_clone = db.clone();
            let bar = Arc::clone(&barrier);
            let sc = Arc::clone(&success_count);
            thread::spawn(move || {
                db_clone.execute("BEGIN", ()).expect("BEGIN failed");
                bar.wait();
                db_clone
                    .execute(
                        &format!("UPDATE t SET val = 'updated_{}' WHERE id = {}", id, id),
                        (),
                    )
                    .expect("UPDATE should succeed for different rows");
                db_clone
                    .execute("COMMIT", ())
                    .expect("COMMIT should succeed");
                sc.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::SeqCst),
        10,
        "all 10 updates on different rows should succeed"
    );

    // Verify all rows updated correctly.
    for id in 1..=10 {
        let val: String = db
            .query_one(&format!("SELECT val FROM t WHERE id = {}", id), ())
            .expect("query_one failed");
        assert_eq!(
            val,
            format!("updated_{}", id),
            "row {} should be 'updated_{}', got '{}'",
            id,
            id,
            val
        );
    }
    db.close().ok();
}
