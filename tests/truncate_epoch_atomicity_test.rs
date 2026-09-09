// Copyright 2026 Stoolap Contributors
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

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering;
use std::sync::Arc;
use stoolap::core::IsolationLevel;
use stoolap::storage::traits::Engine;
use stoolap::{test_failpoints, Database};

fn fixture() -> (tempfile::TempDir, String, Database) {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=full&checkpoint_interval=3600",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, code TEXT UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();
    (dir, dsn, db)
}

fn ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn wal_write_and_sync_failure_preserve_both_layers_and_reopen() {
    let _guard = test_failpoints::FailpointGuard::new();
    for sync_failure in [false, true] {
        let (_dir, dsn, db) = fixture();
        let flag = if sync_failure {
            &test_failpoints::WAL_SYNC_FAIL
        } else {
            &test_failpoints::WAL_WRITE_FAIL
        };
        flag.store(true, Ordering::Release);
        let failed = db.execute("TRUNCATE TABLE t", ());
        flag.store(false, Ordering::Release);
        assert!(failed.is_err(), "TRUNCATE must report the WAL error");
        assert_eq!(ids(&db), vec![1, 2, 3]);
        for (code, expected) in [("a", 1i64), ("b", 2), ("c", 3)] {
            assert_eq!(
                db.query_one::<i64, _>("SELECT id FROM t WHERE code = ?", (code,))
                    .unwrap(),
                expected
            );
        }
        // The existing WAL failure policy requires reopening for further writes.
        let _ = db.close();
        let db = Database::open(&dsn).unwrap();
        assert_eq!(ids(&db), vec![1, 2, 3]);
        assert_eq!(db.execute("TRUNCATE TABLE t", ()).unwrap(), 3);
        db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
        db.close().unwrap();
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            ids(&db),
            vec![1],
            "no duplicate TRUNCATE WAL record may erase later INSERT"
        );
        db.close().unwrap();
    }
}

#[test]
fn external_epoch_and_delayed_snapshot_block_before_any_table_capture() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, _dsn, db) = fixture();
    let epoch = db.engine().registry().capture_read_epoch();
    assert!(db.execute("TRUNCATE TABLE t", ()).is_err());
    assert_eq!(ids(&db), vec![1, 2, 3]);
    drop(epoch);
    let mut delayed = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert!(db.execute("TRUNCATE TABLE t", ()).is_err());
    delayed.rollback().unwrap();
    assert_eq!(db.execute("TRUNCATE TABLE t", ()).unwrap(), 3);
    assert!(ids(&db).is_empty());
}

#[test]
fn direct_table_private_binding_is_exempt_but_external_strong_and_weak_views_are_not() {
    let _guard = test_failpoints::FailpointGuard::new();
    // The in-memory engine exposes MVCCTable's view hook directly; persistent
    // SegmentedTable uses that hook internally without exposing the inner Arc.
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, code TEXT UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    let view = table.captured_hot_view().unwrap();
    let weak = Arc::downgrade(&view);
    assert!(table.truncate().is_err());
    assert_eq!(table.row_count().unwrap(), 3);
    drop(view);
    assert!(
        table.truncate().is_err(),
        "a Weak owner can later upgrade the retained view"
    );
    drop(weak);
    assert_eq!(table.truncate().unwrap(), 3);
    assert_eq!(table.row_count().unwrap(), 0);
    drop(table);
    tx.rollback().unwrap();
    assert!(
        ids(&db).is_empty(),
        "successful physical TRUNCATE remains nonrollbackable"
    );
}

#[test]
fn direct_truncate_checks_current_foreign_keys_after_its_private_epoch() {
    let _guard = test_failpoints::FailpointGuard::new();
    for nullable_replacement in [false, true] {
        let (_dir, _dsn, db) = fixture();
        db.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES t(id))",
            (),
        )
        .unwrap();
        if nullable_replacement {
            db.execute("INSERT INTO child VALUES (1, 1)", ()).unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let mut tx = db
            .engine()
            .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
            .unwrap();
        let mut table = tx.get_table("t").unwrap();
        if nullable_replacement {
            db.execute("UPDATE child SET parent_id = NULL WHERE id = 1", ())
                .unwrap();
            assert_eq!(
                table.truncate().unwrap(),
                3,
                "current HOT NULL must suppress the older cold reference"
            );
        } else {
            db.execute("INSERT INTO child VALUES (1, 1)", ()).unwrap();
            let error = table.truncate().unwrap_err();
            assert!(error.to_string().contains("foreign key"), "{error}");
            assert_eq!(table.row_count().unwrap(), 3);
        }
        drop(table);
        tx.rollback().unwrap();
        assert_eq!(ids(&db).len(), if nullable_replacement { 0 } else { 3 });
    }
}
