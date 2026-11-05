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

//! Visibility answered by the transaction status table: rows of a
//! transaction that never committed stay hidden, however it ended

use std::sync::Arc;

use stoolap::core::{Row, Value};
use stoolap::storage::mvcc::version_store::TransactionVersionStore;
use stoolap::storage::mvcc::MVCCTable;
use stoolap::storage::traits::{Engine, Table};
use stoolap::storage::TransactionRegistry;
use stoolap::Database;

fn table(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> usize {
    db.query("SELECT * FROM t", ()).unwrap().count()
}

#[test]
fn rows_published_before_a_rollback_stay_hidden() {
    let db = table("memory://txn_status_published_rollback");
    let mut txn = db.engine().begin_transaction().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn.id());
    let mut handle = MVCCTable::new(txn.id(), store, local);
    handle
        .insert(Row::from(vec![Value::Integer(1), Value::Integer(7)]))
        .unwrap();
    handle
        .txn_versions()
        .unwrap()
        .write_store()
        .commit()
        .unwrap();
    drop(handle);
    txn.rollback().unwrap();
    assert_eq!(
        rows(&db),
        0,
        "a rolled back transaction's row became visible"
    );
}

#[test]
fn a_store_held_past_its_rollback_cannot_publish() {
    let db = table("memory://txn_status_publish_after_rollback");
    let mut txn = db.engine().begin_transaction().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn.id());
    let mut handle = MVCCTable::new(txn.id(), store, local);
    handle
        .insert(Row::from(vec![Value::Integer(1), Value::Integer(7)]))
        .unwrap();
    txn.rollback().unwrap();
    assert!(handle
        .txn_versions()
        .unwrap()
        .write_store()
        .commit()
        .is_err());
    assert_eq!(rows(&db), 0, "an ended transaction published a row");
}

#[test]
fn a_store_held_past_its_drop_cannot_publish() {
    let db = table("memory://txn_status_publish_after_drop");
    let txn = db.engine().begin_transaction().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn.id());
    let mut handle = MVCCTable::new(txn.id(), store, local);
    handle
        .insert(Row::from(vec![Value::Integer(1), Value::Integer(7)]))
        .unwrap();
    drop(txn);
    assert!(handle
        .txn_versions()
        .unwrap()
        .write_store()
        .commit()
        .is_err());
    assert_eq!(rows(&db), 0, "an ended transaction published a row");
}

#[test]
fn rows_of_a_transaction_dropped_after_publishing_stay_hidden() {
    let db = table("memory://txn_status_published_drop");
    let txn = db.engine().begin_transaction().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn.id());
    let mut handle = MVCCTable::new(txn.id(), store, local);
    handle
        .insert(Row::from(vec![Value::Integer(1), Value::Integer(7)]))
        .unwrap();
    handle
        .txn_versions()
        .unwrap()
        .write_store()
        .commit()
        .unwrap();
    drop(handle);
    drop(txn);
    assert_eq!(rows(&db), 0, "a dropped transaction's row became visible");
}

#[test]
fn many_read_only_transactions_leave_old_commits_visible() {
    let db = table("memory://txn_status_read_only_churn");
    for id in 1..=100i64 {
        db.execute("INSERT INTO t VALUES ($1, 0)", (id,)).unwrap();
    }
    for _ in 0..40_000 {
        db.query("SELECT COUNT(*) FROM t WHERE k = 1", ())
            .unwrap()
            .count();
    }
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1000, 0)", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(rows(&db), 100);
}

#[test]
fn an_id_from_another_registry_is_not_taken_for_committed() {
    let first = TransactionRegistry::new();
    let (id, _) = first.begin_transaction();
    first.commit_transaction(id);
    let second = TransactionRegistry::new();
    let (open, _) = second.begin_transaction();
    assert_eq!(open, id);
    assert!(first.is_directly_visible(id));
    assert!(!second.is_directly_visible(open));
    second.forget_transaction(open);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn old_rows_are_checked_without_the_lock_after_many_read_only_transactions() {
    let db = table("memory://txn_status_no_locked_check");
    for id in 1..=100i64 {
        db.execute("INSERT INTO t VALUES ($1, 0)", (id,)).unwrap();
    }
    for _ in 0..40_000 {
        db.query("SELECT COUNT(*) FROM t WHERE k = 1", ())
            .unwrap()
            .count();
    }
    stoolap::test_failpoints::take_read_paths();
    assert_eq!(rows(&db), 100);
    let locked = stoolap::test_failpoints::take_read_paths()
        .into_iter()
        .filter(|path| *path == "registry.check_committed.locked")
        .count();
    assert_eq!(locked, 0, "old rows were checked under the registry lock");
}

/// Transaction ids one status segment holds
#[cfg(feature = "test-failpoints")]
const SEGMENT_IDS: i64 = 16_384;

#[cfg(feature = "test-failpoints")]
#[test]
fn a_late_publication_cannot_take_the_slot_of_the_transaction_that_reused_it() {
    use std::cell::RefCell;
    use std::rc::Rc;
    use stoolap::storage::traits::Transaction;

    let db = table("memory://txn_status_slot_reuse");
    let mut txn = db.engine().begin_transaction().unwrap();
    let ended = txn.id();
    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), ended);
    let mut handle = MVCCTable::new(ended, store, local);
    handle
        .insert(Row::from(vec![Value::Integer(1), Value::Integer(7)]))
        .unwrap();
    txn.rollback().unwrap();

    let engine = Arc::clone(db.engine());
    let reuser: Rc<RefCell<Option<Box<dyn Transaction>>>> = Rc::new(RefCell::new(None));
    let hook_reuser = Rc::clone(&reuser);
    stoolap::test_failpoints::after_status_run_checked(move || loop {
        let next = engine.begin_transaction().unwrap();
        if next.id() == ended + SEGMENT_IDS {
            *hook_reuser.borrow_mut() = Some(next);
            break;
        }
    });
    let published = handle.txn_versions().unwrap().write_store().commit();
    let reused = reuser.borrow_mut().take();
    assert!(reused.is_some(), "the segment was handed to a later run");
    assert!(published.is_err(), "the ended transaction was let in");
    assert_eq!(rows(&db), 0, "an ended transaction published a row");
    let mut reused = reused.unwrap();
    assert!(reused.rollback().is_ok());
}
