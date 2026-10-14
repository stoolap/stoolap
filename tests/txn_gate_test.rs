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

//! A transaction takes no write once its commit, rollback or drop starts,
//! and those wait for the statements its handles already began

#![cfg(feature = "test-failpoints")]

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use stoolap::core::{Error, Row, Value};
use stoolap::executor::context::StatementSnapshot;
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{
    after_batch_rows, after_hot_publish, after_statement_admitted, after_table_wal_recorded,
    before_commit_marker, cold_rechecks, on_txn_gate_held_off, on_txn_gate_wait,
    txn_gate_admissions, txn_gate_lookups, FailpointGuard, WAL_WRITE_FAIL,
};
use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(10);

fn dsn(dir: &Path) -> String {
    format!("file://{}?checkpoint_interval=3600", dir.display())
}

fn one(db: &Database, sql: &str) -> Option<i64> {
    db.query(sql, ())
        .unwrap()
        .next()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The database the files as they stand recover to, from the WAL alone
fn recover(dir: &Path) -> (tempfile::TempDir, Database) {
    let image = tempfile::tempdir().unwrap();
    crash_image(dir, image.path());
    let db = Database::open(&dsn(image.path())).unwrap();
    (image, db)
}

fn recovered(dir: &Path, sql: &str) -> Option<i64> {
    one(&recover(dir).1, sql)
}

fn row(values: &[i64]) -> Row {
    Row::from(
        values
            .iter()
            .map(|&v| Value::Integer(v))
            .collect::<Vec<_>>(),
    )
}

fn set(column: usize, value: i64) -> impl FnMut(Row) -> stoolap::Result<(Row, bool)> {
    move |mut row| {
        row.set(column, Value::Integer(value))?;
        Ok((row, true))
    }
}

/// A hook that reports it was reached and waits to be let go; a statement
/// never let go fails the test rather than going on
struct Pause {
    reached: Receiver<()>,
    go: Sender<()>,
}

fn pause() -> (impl FnOnce() + 'static, Pause) {
    let (reached_tx, reached) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let hook = move || {
        let _ = reached_tx.send(());
        go_rx
            .recv_timeout(WATCHDOG)
            .expect("the test let the statement go");
    };
    (hook, Pause { reached, go })
}

impl Pause {
    fn reached(&self) {
        self.reached
            .recv_timeout(WATCHDOG)
            .expect("the statement reached its pause");
    }
}

enum Event<T> {
    Waiting,
    Finished(T),
}

/// A commit, rollback, drop or savepoint running on its own thread
struct Finishing<T> {
    events: Receiver<Event<T>>,
}

fn finish_on_thread<T: Send + 'static>(
    finish: impl FnOnce() -> T + Send + 'static,
) -> Finishing<T> {
    let (events_tx, events) = mpsc::channel();
    let waiting = events_tx.clone();
    thread::spawn(move || {
        on_txn_gate_wait(move || {
            let _ = waiting.send(Event::Waiting);
        });
        let _ = events_tx.send(Event::Finished(finish()));
    });
    Finishing { events }
}

impl<T> Finishing<T> {
    /// Lets `paused` go once the finish is seen waiting for it, or seen
    /// done without it, and returns what the finish returned
    fn after_letting_go(self, paused: &Pause) -> T {
        let mut event = self
            .events
            .recv_timeout(WATCHDOG)
            .expect("the finish waited or finished");
        let _ = paused.go.send(());
        loop {
            match event {
                Event::Finished(result) => return result,
                Event::Waiting => {
                    event = self
                        .events
                        .recv_timeout(WATCHDOG)
                        .expect("the finish finished");
                }
            }
        }
    }
}

fn join<T: Send + 'static>(handle: JoinHandle<T>) -> T {
    let (done_tx, done) = mpsc::channel();
    let watcher = thread::spawn(move || {
        let result = handle.join();
        let _ = done_tx.send(());
        result
    });
    done.recv_timeout(WATCHDOG).expect("no deadlock");
    watcher.join().unwrap().unwrap()
}

/// Runs `test` on a thread of its own under the watchdog
fn watched(test: impl FnOnce() + Send + 'static) {
    join(thread::spawn(test));
}

fn kv_table(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    db
}

fn row_one_is_free(db: &Database, holder: &str) {
    db.execute("UPDATE t SET v = 6 WHERE id = 1", ())
        .unwrap_or_else(|e| panic!("the {holder} transaction still holds row 1: {e}"));
}

#[test]
fn a_write_through_another_handle_during_the_commit_is_refused() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.get_table("t").unwrap().insert(row(&[2, 1, 0])).unwrap();
    let mut second = txn.get_table("t").unwrap();
    let written = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&written);
    after_table_wal_recorded(move || {
        *slot.borrow_mut() = Some(second.update_by_row_ids(&[2], &mut set(2, 5)));
    });
    let committed = txn.commit();
    let written = written.borrow_mut().take().expect("the hook ran");
    assert!(
        matches!(written, Err(Error::TransactionEnded)),
        "{written:?}"
    );
    assert!(committed.is_ok(), "{committed:?}");
    let sql = "SELECT v FROM t WHERE id = 2";
    assert_eq!(one(&db, sql), Some(0), "live");
    assert_eq!(recovered(dir.path(), sql), Some(0), "recovered");
}

#[test]
fn a_statement_begun_before_the_commit_is_in_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.get_table("t").unwrap().insert(row(&[2, 0])).unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        handle.update_by_row_ids(&[1], &mut set(1, 5))
    });
    paused.reached();
    finish_on_thread(move || txn.commit())
        .after_letting_go(&paused)
        .unwrap();
    assert_eq!(join(writer).unwrap(), 1);
    let sql = "SELECT v FROM t WHERE id = 1";
    assert_eq!(one(&db, sql), Some(5), "live");
    assert_eq!(recovered(dir.path(), sql), Some(5), "recovered");
}

#[test]
fn a_statement_begun_before_any_write_keeps_the_commit_from_being_read_only() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        handle.update_by_row_ids(&[1], &mut set(1, 5))
    });
    paused.reached();
    finish_on_thread(move || txn.commit())
        .after_letting_go(&paused)
        .unwrap();
    assert_eq!(join(writer).unwrap(), 1);
    let sql = "SELECT v FROM t WHERE id = 1";
    assert_eq!(one(&db, sql), Some(5), "live");
    assert_eq!(recovered(dir.path(), sql), Some(5), "recovered");
}

#[test]
fn a_batch_the_commit_meets_halfway_is_committed_whole() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_batch_rows(500, hook);
        handle.insert_batch((1000..2000).map(|id| row(&[id, 1])).collect())
    });
    paused.reached();
    finish_on_thread(move || txn.commit())
        .after_letting_go(&paused)
        .unwrap();
    join(writer).unwrap();
    let sql = "SELECT COUNT(*) FROM t WHERE v = 1";
    assert_eq!(one(&db, sql), Some(1000), "live");
    assert_eq!(recovered(dir.path(), sql), Some(1000), "recovered");
}

#[test]
fn a_statement_tail_the_commit_meets_marks_the_key_it_did_not_check() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    // Stamps the transaction at the current seal generation
    txn.get_table("t").unwrap().insert(row(&[3, 30])).unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_hot_publish(hook);
        // A hot row's new key meets the volumes only at commit
        handle.update_by_row_ids(&[2], &mut set(1, 7))
    });
    paused.reached();
    let (committed, rechecks) =
        finish_on_thread(move || (txn.commit(), cold_rechecks())).after_letting_go(&paused);
    assert_eq!(join(writer).unwrap(), 1);
    assert_eq!(rechecks, 1, "the commit did not recheck the changed key");
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { .. })),
        "{committed:?}"
    );
    assert_eq!(one(&db, "SELECT COUNT(*) FROM t WHERE k = 7"), Some(1));
}

#[test]
fn a_statement_tail_the_commit_meets_hides_the_sealed_copy_it_replaced() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    db.execute("INSERT INTO t VALUES (2, 0)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_hot_publish(hook);
        handle.update_by_row_ids(&[1], &mut set(1, 5))
    });
    paused.reached();
    finish_on_thread(move || txn.commit())
        .after_letting_go(&paused)
        .unwrap();
    assert_eq!(join(writer).unwrap(), 1);
    let check = |db: &Database, when: &str| {
        assert_eq!(one(db, "SELECT COUNT(*) FROM t"), Some(2), "{when}");
        assert_eq!(
            one(db, "SELECT COUNT(*) FROM t WHERE id = 1"),
            Some(1),
            "{when}"
        );
        assert_eq!(
            one(db, "SELECT COUNT(*) FROM t WHERE v = 0"),
            Some(1),
            "{when}"
        );
        assert_eq!(one(db, "SELECT v FROM t WHERE id = 1"), Some(5), "{when}");
    };
    check(&db, "live");
    let (_image, reopened) = recover(dir.path());
    check(&reopened, "recovered");
    reopened.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&reopened, "recovered and sealed");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}

#[test]
fn a_waiting_commit_holds_no_lock_a_running_statement_needs() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let mut other = txn.get_table("u").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        handle.update_by_row_ids(&[1], &mut |mut row| {
            other.insert(self::row(&[9]))?;
            row.set(1, Value::Integer(5))?;
            Ok((row, true))
        })
    });
    paused.reached();
    finish_on_thread(move || txn.commit())
        .after_letting_go(&paused)
        .unwrap();
    assert_eq!(join(writer).unwrap(), 1);
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(5));
    assert_eq!(one(&db, "SELECT COUNT(*) FROM u"), Some(1));
}

#[test]
fn a_finished_transaction_opens_no_table_and_leaves_no_entry() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let engine = db.engine().clone();
    let active = engine.registry().active_count();
    let entries = engine.txn_store_entries();
    for commit in [true, false] {
        for _ in 0..200 {
            let mut txn = engine.begin_transaction().unwrap();
            let id = txn.id();
            txn.get_table("t")
                .unwrap()
                .insert(row(&[id + 10_000, 1]))
                .unwrap();
            if commit {
                txn.commit().unwrap();
            } else {
                txn.rollback().unwrap();
            }
            assert!(txn.get_table("t").is_err(), "commit={commit}");
        }
    }
    assert_eq!(engine.registry().active_count(), active);
    assert_eq!(engine.txn_store_entries(), entries);
}

#[test]
fn a_rollback_to_a_timestamp_through_another_handle_during_the_commit_does_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.get_table("t").unwrap().insert(row(&[2, 3])).unwrap();
    let second = txn.get_table("t").unwrap();
    after_table_wal_recorded(move || second.rollback_to_timestamp(0));
    txn.commit().unwrap();
    let sql = "SELECT v FROM t WHERE id = 2";
    assert_eq!(one(&db, sql), Some(3), "live");
    assert_eq!(recovered(dir.path(), sql), Some(3), "recovered");
}

#[test]
fn a_handle_closed_after_the_commit_changes_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    handle.update_by_row_ids(&[1], &mut set(1, 5)).unwrap();
    txn.commit().unwrap();
    assert!(handle.close().is_ok());
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(5));
    db.execute("UPDATE t SET v = 6 WHERE id = 1", ()).unwrap();
}

#[test]
fn a_rollback_lets_its_rows_go_while_a_handle_is_kept() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut kept = txn.get_table("t").unwrap();
    kept.update_by_row_ids(&[1], &mut set(1, 5)).unwrap();
    txn.rollback().unwrap();
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    row_one_is_free(&db, "rolled back");
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(6));
}

#[test]
fn a_failed_commit_lets_its_rows_go_while_a_handle_is_kept() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut kept = txn.get_table("t").unwrap();
    // A hot row's new key meets the volumes only at commit, which refuses it
    kept.update_by_row_ids(&[2], &mut set(1, 7)).unwrap();
    let committed = txn.commit();
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { .. })),
        "{committed:?}"
    );
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    db.execute("UPDATE t SET k = 21 WHERE id = 2", ())
        .expect("the failed transaction still holds row 2");
}

#[test]
fn a_commit_whose_marker_fails_leaves_no_entry_and_lets_a_kept_handle_hold_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let engine = db.engine().clone();
    let entries = engine.txn_store_entries();
    let mut txn = engine.begin_transaction().unwrap();
    let mut kept = txn.get_table("t").unwrap();
    kept.update_by_row_ids(&[1], &mut set(1, 5)).unwrap();
    before_commit_marker(|| WAL_WRITE_FAIL.store(true, Ordering::SeqCst));
    let committed = txn.commit();
    WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
    assert!(committed.is_err(), "the marker write fails the commit");
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    assert_eq!(engine.txn_store_entries(), entries);
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_commit_lets_the_claims_of_a_table_it_did_not_write_go_while_a_handle_is_kept() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://txn_gate_claim_only_commit").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let kept = txn.get_table("t").unwrap();
    kept.try_claim_row(1).unwrap();
    txn.get_table("u").unwrap().insert(row(&[1, 1])).unwrap();
    txn.commit().unwrap();
    row_one_is_free(&db, "committed");
    drop(kept);
}

#[test]
fn a_rollback_waits_for_a_statement_begun_before_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        let written = handle.update_by_row_ids(&[1], &mut set(1, 5));
        (written, handle)
    });
    paused.reached();
    finish_on_thread(move || txn.rollback())
        .after_letting_go(&paused)
        .unwrap();
    let (written, kept) = join(writer);
    assert_eq!(written.unwrap(), 1);
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(0));
    row_one_is_free(&db, "rolled back");
}

#[test]
fn a_row_written_while_the_rollback_waits_is_cleaned_up() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let mut other = txn.get_table("u").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        let written = handle.update_by_row_ids(&[1], &mut |mut row| {
            other.insert(self::row(&[9]))?;
            row.set(1, Value::Integer(5))?;
            Ok((row, true))
        });
        (written, other)
    });
    paused.reached();
    finish_on_thread(move || txn.rollback())
        .after_letting_go(&paused)
        .unwrap();
    let (written, other) = join(writer);
    assert_eq!(written.unwrap(), 1);
    assert!(!other.has_local_changes(), "u's kept handle holds its row");
    assert_eq!(one(&db, "SELECT COUNT(*) FROM u"), Some(0));
    db.execute("INSERT INTO u VALUES (9)", ())
        .expect("the rolled back transaction still holds u's row");
}

#[test]
fn a_transaction_cannot_finish_inside_its_own_statement() {
    let _guard = FailpointGuard::new();
    watched(|| {
        let dir = tempfile::tempdir().unwrap();
        let db = kv_table(dir.path());
        let mut txn = db.engine().begin_transaction().unwrap();
        txn.create_savepoint("s").unwrap();
        let mut handle = txn.get_table("t").unwrap();
        let mut refused = Vec::new();
        let written = handle.update_by_row_ids(&[1], &mut |mut row| {
            refused.push(txn.commit());
            refused.push(txn.rollback());
            refused.push(txn.rollback_to_savepoint("s"));
            row.set(1, Value::Integer(5))?;
            Ok((row, true))
        });
        assert_eq!(written.unwrap(), 1);
        for (step, result) in ["commit", "rollback", "rollback to savepoint"]
            .iter()
            .zip(&refused)
        {
            assert!(
                matches!(result, Err(Error::InvalidArgument(_))),
                "{step}: {result:?}"
            );
        }
        txn.commit().unwrap();
        assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(5));
    });
}

#[test]
fn a_drop_waits_for_a_statement_begun_before_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        let written = handle.update_by_row_ids(&[1], &mut set(1, 5));
        (written, handle)
    });
    paused.reached();
    finish_on_thread(move || drop(txn)).after_letting_go(&paused);
    let (written, kept) = join(writer);
    assert_eq!(written.unwrap(), 1);
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(0));
    row_one_is_free(&db, "dropped");
}

#[test]
fn a_table_from_a_statement_snapshot_admits_through_the_gate() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let snapshot = StatementSnapshot::new(db.engine().begin_transaction().unwrap());
    let mut handle = snapshot.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_statement_admitted(hook);
        let written = handle.update_by_row_ids(&[1], &mut set(1, 5));
        (written, handle)
    });
    paused.reached();
    finish_on_thread(move || drop(snapshot)).after_letting_go(&paused);
    let (written, kept) = join(writer);
    assert_eq!(written.unwrap(), 1);
    assert!(!kept.has_local_changes(), "the kept handle holds its rows");
    assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(0));
    row_one_is_free(&db, "dropped");
}

#[test]
fn a_transaction_dropped_inside_its_own_statement_cleans_up_when_it_ends() {
    let _guard = FailpointGuard::new();
    watched(|| {
        let dir = tempfile::tempdir().unwrap();
        let db = kv_table(dir.path());
        let txn = db.engine().begin_transaction().unwrap();
        let mut kept = txn.get_table("t").unwrap();
        let mut owner = Some(txn);
        let written = kept.update_by_row_ids(&[1], &mut |mut row| {
            drop(owner.take());
            row.set(1, Value::Integer(5))?;
            Ok((row, true))
        });
        assert_eq!(
            written.unwrap(),
            1,
            "the admitted statement did not complete"
        );
        assert!(!kept.has_local_changes(), "the kept handle holds its rows");
        assert_eq!(one(&db, "SELECT v FROM t WHERE id = 1"), Some(0));
        row_one_is_free(&db, "dropped");
    });
}

#[test]
fn a_savepoint_is_never_taken_in_the_middle_of_a_statement() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_batch_rows(500, hook);
        handle.insert_batch((1000..2000).map(|id| row(&[id, 1])).collect())
    });
    paused.reached();
    let (taken, mut txn) = finish_on_thread(move || {
        let taken = txn.create_savepoint("s");
        (taken, txn)
    })
    .after_letting_go(&paused);
    join(writer).unwrap();
    taken.unwrap();
    txn.get_table("t").unwrap().insert(row(&[5000, 2])).unwrap();
    txn.rollback_to_savepoint("s").unwrap();
    txn.commit().unwrap();
    assert_eq!(one(&db, "SELECT COUNT(*) FROM t WHERE v = 1"), Some(1000));
    assert_eq!(one(&db, "SELECT COUNT(*) FROM t WHERE v = 2"), Some(0));
}

#[test]
fn a_rollback_to_a_savepoint_waits_for_a_statement_in_flight() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.create_savepoint("s").unwrap();
    let mut handle = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_batch_rows(500, hook);
        handle.insert_batch((1000..2000).map(|id| row(&[id, 1])).collect())
    });
    paused.reached();
    let (done, mut txn) = finish_on_thread(move || {
        let done = txn.rollback_to_savepoint("s");
        (done, txn)
    })
    .after_letting_go(&paused);
    join(writer).unwrap();
    done.unwrap();
    txn.commit().unwrap();
    assert_eq!(
        one(&db, "SELECT COUNT(*) FROM t WHERE v = 1"),
        Some(0),
        "half of the batch survived the rollback"
    );
}

#[test]
fn a_statement_of_one_engine_may_finish_a_transaction_of_another_with_the_same_id() {
    let _guard = FailpointGuard::new();
    watched(|| {
        let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (db_a, db_b) = (kv_table(dir_a.path()), kv_table(dir_b.path()));
        let txn_a = db_a.engine().begin_transaction().unwrap();
        let mut txn_b = db_b.engine().begin_transaction().unwrap();
        assert_eq!(txn_a.id(), txn_b.id(), "the two engines gave the same id");
        txn_b.get_table("t").unwrap().insert(row(&[2, 2])).unwrap();
        let mut handle = txn_a.get_table("t").unwrap();
        let mut finished = None;
        handle
            .update_by_row_ids(&[1], &mut |mut row| {
                finished = Some(txn_b.commit());
                row.set(1, Value::Integer(5))?;
                Ok((row, true))
            })
            .unwrap();
        assert!(finished.unwrap().is_ok());
        assert_eq!(one(&db_b, "SELECT v FROM t WHERE id = 2"), Some(2));
    });
}

#[test]
fn a_statement_started_during_a_savepoint_rollback_waits_and_then_runs() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    txn.create_savepoint("s").unwrap();
    let mut batch = txn.get_table("t").unwrap();
    let mut late = txn.get_table("t").unwrap();
    let (hook, paused) = pause();
    let writer = thread::spawn(move || {
        after_batch_rows(500, hook);
        batch.insert_batch((1000..2000).map(|id| row(&[id, 1])).collect())
    });
    paused.reached();
    let rollback = finish_on_thread(move || {
        let done = txn.rollback_to_savepoint("s");
        (done, txn)
    });
    let waiting = rollback
        .events
        .recv_timeout(WATCHDOG)
        .expect("the rollback waited or finished");
    assert!(
        matches!(waiting, Event::Waiting),
        "the rollback did not wait for the batch"
    );
    let (held_tx, held) = mpsc::channel();
    let third = thread::spawn(move || {
        on_txn_gate_held_off(move || {
            let _ = held_tx.send(());
        });
        late.insert(row(&[3, 3]))
    });
    held.recv_timeout(WATCHDOG)
        .expect("the third statement was held off");
    let _ = paused.go.send(());
    join(writer).unwrap();
    let (done, mut txn) = match rollback.events.recv_timeout(WATCHDOG).expect("finished") {
        Event::Finished(result) => result,
        Event::Waiting => panic!("the rollback waited twice"),
    };
    done.unwrap();
    join(third).unwrap();
    txn.commit().unwrap();
    assert_eq!(one(&db, "SELECT COUNT(*) FROM t WHERE v = 1"), Some(0));
    assert_eq!(
        one(&db, "SELECT v FROM t WHERE id = 3"),
        Some(3),
        "the statement held off did not run after the rollback"
    );
}

#[test]
fn a_chain_of_nine_transactions_nested_on_one_thread_works() {
    let _guard = FailpointGuard::new();
    watched(|| {
        let dirs: Vec<_> = (0..9).map(|_| tempfile::tempdir().unwrap()).collect();
        let dbs: Vec<_> = dirs.iter().map(|dir| kv_table(dir.path())).collect();
        let mut txns: Vec<_> = dbs
            .iter()
            .map(|db| db.engine().begin_transaction().unwrap())
            .collect();
        let mut handles: Vec<_> = txns
            .iter_mut()
            .map(|txn| txn.get_table("t").unwrap())
            .collect();
        fn nest(handles: &mut [Box<dyn stoolap::storage::traits::Table>]) -> stoolap::Result<()> {
            let Some((first, rest)) = handles.split_first_mut() else {
                return Ok(());
            };
            first
                .update_by_row_ids(&[1], &mut |mut row| {
                    nest(rest)?;
                    row.set(1, Value::Integer(5))?;
                    Ok((row, true))
                })
                .map(|_| ())
        }
        nest(&mut handles).unwrap();
        for txn in &mut txns {
            txn.commit().unwrap();
        }
        for db in &dbs {
            assert_eq!(one(db, "SELECT v FROM t WHERE id = 1"), Some(5));
        }
    });
}

#[test]
fn a_statement_is_admitted_once_whatever_its_layers_rows_and_claims() {
    let _guard = FailpointGuard::new();
    let measure = |work: &mut dyn FnMut()| {
        let (admissions, lookups) = (txn_gate_admissions(), txn_gate_lookups());
        work();
        (
            txn_gate_admissions() - admissions,
            txn_gate_lookups() - lookups,
        )
    };
    let batch = || (1000..2000).map(|id| row(&[id, 1])).collect::<Vec<_>>();

    let dir = tempfile::tempdir().unwrap();
    let db = kv_table(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut segmented = txn.get_table("t").unwrap();
    let counted = measure(&mut || segmented.insert_batch(batch()).unwrap());
    assert_eq!(counted, (1, 1), "a batch through a segmented handle");
    txn.commit().unwrap();

    let memory = Database::open("memory://a_statement_is_admitted_once").unwrap();
    memory
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut txn = memory.engine().begin_transaction().unwrap();
    let mut bare = txn.get_table("t").unwrap();
    let counted = measure(&mut || bare.insert_batch(batch()).unwrap());
    assert_eq!(counted, (1, 1), "a batch through a bare hot handle");
    txn.commit().unwrap();

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut cold = txn.get_table("t").unwrap();
    let ids: Vec<i64> = (1000..2000).collect();
    let counted = measure(&mut || {
        assert_eq!(cold.update_by_row_ids(&ids, &mut set(1, 2)).unwrap(), 1000);
    });
    assert_eq!(counted, (1, 1), "a cold update claiming each row");
    let mut other = txn.get_table("t").unwrap();
    let counted = measure(&mut || {
        cold.update_by_row_ids(&[1], &mut |mut row| {
            other.insert(self::row(&[5000, 9]))?;
            row.set(1, Value::Integer(7))?;
            Ok((row, true))
        })
        .unwrap();
    });
    assert_eq!(counted, (1, 2), "a statement with one nested in its setter");
    txn.commit().unwrap();
}
