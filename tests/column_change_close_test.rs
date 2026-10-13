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

//! A change of a column's definition closes the table, waits for what was
//! being published into it, and lets nothing prepared before it publish
//! after it

#![cfg(feature = "test-failpoints")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::Duration;

use stoolap::core::{DataType, Row, Value};
use stoolap::storage::mvcc::version_store::TransactionVersionStore;
use stoolap::storage::mvcc::MVCCTable;
use stoolap::storage::traits::{Engine, Table, Transaction};
use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

const MODIFY: &str = "ALTER TABLE t MODIFY COLUMN k TEXT";

fn dsn(dir: &std::path::Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

fn open(dir: &std::path::Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db
}

fn count(db: &Database) -> i64 {
    db.query_one("SELECT COUNT(*) FROM t", ()).unwrap()
}

fn k_type(db: &Database) -> DataType {
    db.engine().get_table_schema("t").unwrap().columns[1].data_type
}

/// The change, in another thread; returns once it closed the table and
/// started to wait, or panics when it never does
fn change_waiting(db: &Database) -> std::thread::JoinHandle<stoolap::core::Result<i64>> {
    let other = db.clone();
    let (reached, waiting) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        fp::on_column_change_waiting(move || reached.send(()).unwrap());
        other.execute(MODIFY, ())
    });
    waiting
        .recv_timeout(Duration::from_secs(5))
        .expect("the change closed the table and waits");
    handle
}

/// A handle bound to an open transaction, writing straight into the table's
/// version store
fn store_handle(db: &Database, txn: &dyn Transaction) -> MVCCTable {
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn.id());
    MVCCTable::new(txn.id(), store, local)
}

/// A transaction left open after applying row (`id`, `k`) to the store
fn applied_and_open(db: &Database, id: i64, k: i64) -> Box<dyn Transaction> {
    let txn = db.engine().begin_transaction().unwrap();
    let mut table = store_handle(db, txn.as_ref());
    table
        .insert(Row::from(vec![Value::Integer(id), Value::Integer(k)]))
        .unwrap();
    table.txn_versions().write_store().commit().unwrap();
    txn
}

/// A seal that built its volumes, then `change` before it publishes them
fn seal_across(change: impl FnOnce(&Database) + 'static) {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    let reached = Rc::new(Cell::new(false));
    let seen = Rc::clone(&reached);
    let other = db.clone();
    fp::after_seal_volumes_built(move || {
        seen.set(true);
        change(&other);
    });
    let before = fp::stale_publications();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(reached.get(), "the seal built its volumes");
    assert_eq!(db.engine().volume_stats().len(), 0, "nothing was published");
    assert_eq!(count(&db), 2, "the rows stayed hot");
    assert_eq!(fp::stale_publications() - before, 1, "the volumes went");
}

#[test]
fn a_seal_built_before_a_column_change_is_not_published_after_it() {
    seal_across(|db| {
        db.execute(MODIFY, ()).unwrap();
    });
}

#[test]
fn a_seal_built_before_an_engine_column_change_is_not_published_after_it() {
    seal_across(|db| {
        db.engine()
            .modify_column("t", "k", DataType::Text, true)
            .unwrap();
    });
}

#[test]
fn a_seal_built_before_a_transaction_column_change_is_not_published_after_it() {
    seal_across(|db| {
        let mut column = db.engine().get_table_schema("t").unwrap().columns[1].clone();
        column.data_type = DataType::Text;
        let mut txn = db.engine().begin_transaction().unwrap();
        txn.modify_table_column("t", column).unwrap();
        txn.commit().unwrap();
    });
}

#[test]
fn a_compaction_built_before_a_column_change_is_not_published_after_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    for id in 1..=3 {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {})", id * 10), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let volumes = db.engine().volume_stats().len();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let reached = Rc::new(Cell::new(false));
    let seen = Rc::clone(&reached);
    let other = db.clone();
    fp::after_compaction_outputs_built(move || {
        seen.set(true);
        other.execute(MODIFY, ()).unwrap();
    });
    let before = fp::stale_publications();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(reached.get(), "the compaction built its outputs");
    assert_eq!(
        db.engine().volume_stats().len(),
        volumes,
        "the old volumes stayed"
    );
    assert_eq!(count(&db), 3);
    assert_eq!(fp::stale_publications() - before, 1, "the outputs went");
}

#[test]
fn a_change_waits_for_a_commit_counted_before_it_and_refused_by_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let change = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&change);
    let other = db.clone();
    fp::after_commit_counted(move || {
        let handle = change_waiting(&other);
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !handle.is_finished(),
            "the change ran past a counted commit"
        );
        *slot.borrow_mut() = Some(handle);
    });
    let inserted = db.execute("INSERT INTO t VALUES (2, 8)", ());
    assert!(
        matches!(inserted, Err(stoolap::core::Error::SchemaChanged { .. })),
        "the change moved the schema under the commit: {inserted:?}"
    );
    let changed = change.borrow_mut().take().unwrap().join().unwrap();
    assert!(changed.is_ok(), "{changed:?}");
    assert_eq!(count(&db), 1, "the commit was undone");
    assert_eq!(k_type(&db), DataType::Text);
}

#[test]
fn a_change_waits_for_a_commit_published_before_it_until_its_marker_lands() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let change = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&change);
    let other = db.clone();
    fp::before_commit_marker(move || {
        let handle = change_waiting(&other);
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !handle.is_finished(),
            "the change ran past a commit in publication"
        );
        *slot.borrow_mut() = Some(handle);
    });
    db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
    let changed = change.borrow_mut().take().unwrap().join().unwrap();
    assert!(changed.is_ok(), "{changed:?}");
    assert_eq!(count(&db), 2);
    assert_eq!(k_type(&db), DataType::Text);
}

#[test]
fn a_change_waiting_on_a_commit_whose_marker_fails_keeps_the_old_schema() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let change = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&change);
    let other = db.clone();
    fp::before_commit_marker(move || {
        *slot.borrow_mut() = Some(change_waiting(&other));
        fp::WAL_WRITE_FAIL.store(true, Ordering::Release);
    });
    let inserted = db.execute("INSERT INTO t VALUES (2, 8)", ());
    let changed = change.borrow_mut().take().unwrap().join().unwrap();
    fp::WAL_WRITE_FAIL.store(false, Ordering::Release);
    assert!(inserted.is_err(), "the marker failed: {inserted:?}");
    // The failed write poisons the log: the change's own record fails too
    assert!(changed.is_err(), "{changed:?}");
    assert_eq!(k_type(&db), DataType::Integer, "the old schema stays");
    assert_eq!(count(&db), 1, "the commit was undone");
}

#[test]
fn a_change_is_refused_while_a_transaction_that_applied_rows_is_open() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let mut txn = applied_and_open(&db, 2, 8);
    let changed = db.execute(MODIFY, ());
    assert!(
        matches!(
            changed,
            Err(stoolap::core::Error::TableHasOpenTransactionRows { .. })
        ),
        "{changed:?}"
    );
    assert_eq!(k_type(&db), DataType::Integer, "the old schema stays");
    txn.commit().unwrap();
    let row: i64 = db.query_one("SELECT k FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(row, 8);
    assert_eq!(
        db.query_one::<i64, _>("SELECT id FROM t WHERE k = 8", ())
            .unwrap(),
        2,
        "the index holds the row"
    );
}

#[test]
fn a_change_waits_for_a_gc_pass_to_finish_its_cleanup() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    let change = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&change);
    let other = db.clone();
    fp::after_gc_rows_removed(move || {
        let handle = change_waiting(&other);
        std::thread::sleep(Duration::from_millis(100));
        assert!(!handle.is_finished(), "the change ran past a GC pass");
        *slot.borrow_mut() = Some(handle);
    });
    db.execute("VACUUM", ()).unwrap();
    let handle = change
        .borrow_mut()
        .take()
        .expect("the GC pass removed a row");
    let changed = handle.join().unwrap();
    assert!(changed.is_ok(), "{changed:?}");
    assert_eq!(k_type(&db), DataType::Text);
}

#[test]
fn dropping_the_table_waits_for_a_change_in_progress() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let mut txn = applied_and_open(&db, 2, 8);
    // The change waits on the open transaction, holding the DDL guard
    let changing = change_waiting(&db);
    let other = db.clone();
    let dropping = std::thread::spawn(move || other.execute("DROP TABLE t", ()));
    std::thread::sleep(Duration::from_millis(200));
    assert!(!dropping.is_finished(), "the drop ran inside the change");
    assert!(changing.join().unwrap().is_err());
    txn.rollback().unwrap();
    let _ = dropping.join().unwrap();
}

#[test]
fn a_commit_begun_while_a_change_waits_is_refused() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let mut holder = applied_and_open(&db, 2, 8);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(3), Value::Integer(9)]))
        .unwrap();
    drop(table);
    let changing = change_waiting(&db);
    let committed = txn.commit();
    assert!(
        matches!(committed, Err(stoolap::core::Error::SchemaChanged { .. })),
        "{committed:?}"
    );
    assert!(changing.join().unwrap().is_err());
    holder.rollback().unwrap();
    let refused: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 3", ())
        .unwrap();
    assert_eq!(refused, 0, "the refused commit left nothing");
}

#[test]
fn a_store_commit_begun_while_a_change_waits_is_refused() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let mut holder = applied_and_open(&db, 2, 8);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = store_handle(&db, txn.as_ref());
    table
        .insert(Row::from(vec![Value::Integer(3), Value::Integer(9)]))
        .unwrap();
    let changing = change_waiting(&db);
    let committed = table.txn_versions().write_store().commit();
    assert!(
        matches!(committed, Err(stoolap::core::Error::SchemaChanged { .. })),
        "{committed:?}"
    );
    assert!(changing.join().unwrap().is_err());
    txn.rollback().unwrap();
    holder.rollback().unwrap();
    let refused: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 3", ())
        .unwrap();
    assert_eq!(refused, 0, "the refused commit left nothing");
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_commit_rolled_back_to_a_savepoint_is_refused_while_a_change_waits() {
    let _guard = FailpointGuard::new();
    let db = Database::open(
        "memory://a_commit_rolled_back_to_a_savepoint_is_refused_while_a_change_waits",
    )
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    let mut holder = applied_and_open(&db, 2, 8);
    let writer = db.clone();
    writer.execute("BEGIN", ()).unwrap();
    writer.execute("INSERT INTO t VALUES (3, 9)", ()).unwrap();
    writer.execute("SAVEPOINT s", ()).unwrap();
    writer.execute("INSERT INTO t VALUES (4, 10)", ()).unwrap();
    writer.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    let changing = change_waiting(&db);
    let committed = writer.execute("COMMIT", ());
    assert!(
        matches!(committed, Err(stoolap::core::Error::SchemaChanged { .. })),
        "{committed:?}"
    );
    assert!(changing.join().unwrap().is_err());
    holder.rollback().unwrap();
    let refused: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 3", ())
        .unwrap();
    assert_eq!(refused, 0, "the refused commit left nothing");
}

#[test]
fn a_seal_begun_inside_a_transaction_column_change_takes_the_new_schema() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    let sealer = db.clone();
    let seal = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&seal);
    fp::on_column_change_waiting(move || {
        let (seen, saw) = mpsc::channel();
        let passed = seen.clone();
        let handle = std::thread::spawn(move || {
            fp::on_seal_passed_changing_table(move || {
                let _ = passed.send("passed the table by");
            });
            fp::after_maintenance_schema_taken(move || {
                let _ = seen.send("took the schema");
            });
            sealer.execute("PRAGMA CHECKPOINT", ())
        });
        let what = saw
            .recv_timeout(Duration::from_secs(5))
            .expect("the seal reached the table");
        assert_eq!(what, "passed the table by", "the seal inside the change");
        *slot.borrow_mut() = Some(handle);
    });
    let mut column = db.engine().get_table_schema("t").unwrap().columns[1].clone();
    column.data_type = DataType::Text;
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.modify_table_column("t", column).unwrap();
    txn.commit().unwrap();
    let sealed = seal.borrow_mut().take().unwrap().join().unwrap();
    assert!(sealed.is_ok(), "{sealed:?}");
    assert_eq!(
        db.engine().volume_stats().len(),
        0,
        "a seal begun inside the change published under the old schema"
    );
    assert_eq!(
        k_type(&db),
        DataType::Text,
        "the schema cache was refreshed"
    );
}
