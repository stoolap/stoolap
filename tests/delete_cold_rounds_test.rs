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

//! A DELETE reads and selects its sealed rows outside the seal fence, and
//! under it claims and changes them from what it read

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use stoolap::core::{Error, IsolationLevel};
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{
    after_cold_round_prepared, after_delete_rows_scanned, block_file_reads,
    block_store_drops_by_fence, volume_file_drops_by_fence, volume_file_reads_by_fence,
    FailpointGuard,
};
use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(10);

fn open(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db
}

/// Commits `rows` as `(id, 0, v)` and seals them into one volume
fn seal_rows(db: &Database, rows: &[(i64, i64)]) {
    let values: Vec<String> = rows
        .iter()
        .map(|(id, v)| format!("({id}, 0, {v})"))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db.engine().seal_for_test().unwrap();
}

/// Seals `(id, 0, v)` from another thread, so its counts stay off this one
fn seal_row_apart(db: &Database, id: i64, v: i64) {
    let db = db.clone();
    on_thread(move || seal_rows(&db, &[(id, v)])).expect("the seal finished");
}

/// Runs `sql` from another thread; true when it succeeded in time
fn execute_apart(db: &Database, sql: &'static str) -> bool {
    let db = db.clone();
    on_thread(move || db.execute(sql, ()).is_ok()) == Some(true)
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

fn is_write_conflict<T: std::fmt::Debug>(result: &stoolap::core::Result<T>) -> bool {
    matches!(result, Err(Error::Internal { message }) if message.contains("write conflict"))
}

/// File reads on this thread, whole volumes and blocks, as (outside, under)
/// the seal fence
fn file_reads() -> (usize, usize) {
    let (volumes, blocks) = (volume_file_reads_by_fence(), block_file_reads());
    (volumes.0 + blocks.0, volumes.1 + blocks.1)
}

fn delta(after: (usize, usize), before: (usize, usize)) -> (usize, usize) {
    (after.0 - before.0, after.1 - before.1)
}

/// Runs `work` on its own thread and waits for it; None when it did not
/// finish in time
fn on_thread<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(work());
    });
    finished.recv_timeout(WATCHDOG).ok()
}

/// The process-wide group cache budget at zero while held, restored on
/// drop; held one at a time, since tests share the process under cargo test
struct NoGroupCache<'a> {
    db: &'a Database,
    budget: i64,
    _serial: MutexGuard<'static, ()>,
}

static GROUP_CACHE_BUDGET: Mutex<()> = Mutex::new(());

impl<'a> NoGroupCache<'a> {
    fn set(db: &'a Database) -> Self {
        let serial = GROUP_CACHE_BUDGET
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let budget: i64 = db.query_one("PRAGMA GROUP_CACHE_MB", ()).unwrap();
        db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
        Self {
            db,
            budget,
            _serial: serial,
        }
    }
}

impl Drop for NoGroupCache<'_> {
    fn drop(&mut self) {
        let _ = self
            .db
            .execute(&format!("PRAGMA GROUP_CACHE_MB = {}", self.budget), ());
    }
}

/// Seals a row in each of the next three DELETE rounds
fn seal_in_three_rounds(db: Database, rounds: Rc<Cell<usize>>) {
    after_cold_round_prepared(move || {
        let round = rounds.get() + 1;
        rounds.set(round);
        seal_row_apart(&db, 100 + round as i64, 100 + round as i64);
        if round < 3 {
            seal_in_three_rounds(db, rounds);
        }
    });
}

/// Rows 1, 2 and 3 sealed together and cold, the group cache empty, then
/// `delete` run; returns what it deleted and its file reads, (outside, under)
fn reads_of(delete: impl FnOnce(&Database) -> i64) -> (i64, (usize, usize)) {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20), (3, 30)]);
    let _no_cache = NoGroupCache::set(&db);
    db.engine().cold_volumes_for_test("t");
    let reads = file_reads();
    let deleted = delete(&db);
    (deleted, delta(file_reads(), reads))
}

fn assert_reads_outside(form: &str, deleted: i64, expected: i64, reads: (usize, usize)) {
    assert_eq!(deleted, expected, "{form}: rows deleted");
    assert_eq!(reads.1, 0, "{form}: file reads under the seal fence");
    assert!(reads.0 > 0, "{form}: the DELETE read no file");
}

#[test]
fn a_delete_by_primary_key_reads_no_file_under_the_fence() {
    let _guard = FailpointGuard::new();
    let (deleted, reads) = reads_of(|db| db.execute("DELETE FROM t WHERE id = 2", ()).unwrap());
    assert_reads_outside("pk", deleted, 1, reads);
}

#[test]
fn a_delete_with_a_filter_reads_no_file_under_the_fence() {
    let _guard = FailpointGuard::new();
    let (deleted, reads) = reads_of(|db| db.execute("DELETE FROM t WHERE v >= 20", ()).unwrap());
    assert_reads_outside("where", deleted, 2, reads);
}

#[test]
fn a_delete_by_row_ids_reads_no_file_under_the_fence() {
    let _guard = FailpointGuard::new();
    let (deleted, reads) = reads_of(|db| {
        let mut txn = db.engine().begin_transaction().unwrap();
        let mut table = txn.get_table("t").unwrap();
        let deleted = table.delete_by_row_ids(&[1, 3]).unwrap();
        drop(table);
        txn.commit().unwrap();
        i64::from(deleted)
    });
    assert_reads_outside("ids", deleted, 2, reads);
}

#[test]
fn a_delete_of_scanned_rows_reads_no_file_under_the_fence() {
    let _guard = FailpointGuard::new();
    let (deleted, reads) = reads_of(|db| {
        let mut txn = db.engine().begin_transaction().unwrap();
        let mut table = txn.get_table("t").unwrap();
        let deleted = table.delete_scanned_rows(&mut vec![1, 3]).unwrap();
        drop(table);
        txn.commit().unwrap();
        i64::from(deleted)
    });
    assert_reads_outside("scanned", deleted, 2, reads);
}

#[test]
fn a_delete_whose_volumes_keep_changing_deletes_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (50, 0, 50)", ()).unwrap();
    let rounds = Rc::new(Cell::new(0));
    seal_in_three_rounds(db.clone(), Rc::clone(&rounds));
    let deleted = txn.execute("DELETE FROM t WHERE v >= 10 AND v <= 20", ());
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    assert_eq!(rounds.get(), 3, "the DELETE did not prepare three times");
    txn.commit().unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 50"), 1);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id <= 2"), 2);
    // The refused statement holds no claim
    assert_eq!(db.execute("DELETE FROM t WHERE id = 1", ()).unwrap(), 1);
}

#[test]
fn a_row_compacted_between_rounds_is_deleted_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10)]);
    seal_rows(&db, &[(3, 30)]);
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let mover = db.clone();
    after_cold_round_prepared(move || {
        seal_row_apart(&mover, 60, 60);
        let compacted = on_thread(move || mover.engine().compact_for_test().is_ok());
        assert_eq!(compacted, Some(true), "the compaction did not finish");
    });
    let deleted = db.execute("DELETE FROM t WHERE v = 10", ());
    assert_eq!(deleted.ok(), Some(1));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 1"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 2);
}

#[test]
fn a_row_another_transaction_deletes_while_prepared_is_a_write_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let other = db.clone();
    let done = Rc::new(Cell::new(false));
    let slot = Rc::clone(&done);
    after_cold_round_prepared(move || {
        slot.set(execute_apart(&other, "DELETE FROM t WHERE id = 1"))
    });
    let mut txn = db.begin().unwrap();
    let deleted = txn.execute("DELETE FROM t WHERE v <= 20", ());
    assert!(
        done.get(),
        "the other delete did not commit inside the round"
    );
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    txn.rollback().unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 2"), 1);
}

#[test]
fn a_row_another_transaction_updates_while_prepared_is_a_write_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let other = db.clone();
    let done = Rc::new(Cell::new(false));
    let slot = Rc::clone(&done);
    after_cold_round_prepared(move || {
        slot.set(execute_apart(&other, "UPDATE t SET x = 9 WHERE id = 1"))
    });
    let deleted = db.execute("DELETE FROM t WHERE v <= 20", ());
    assert!(
        done.get(),
        "the other update did not commit inside the round"
    );
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    assert_eq!(count(&db, "SELECT x FROM t WHERE id = 1"), 9);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 2"), 1);
}

#[test]
fn a_row_another_transaction_deleted_and_collected_is_a_write_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        txn.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        2
    );
    assert!(execute_apart(&db, "DELETE FROM t WHERE id = 1"));
    db.execute("VACUUM", ()).unwrap();
    let deleted = txn.execute("DELETE FROM t WHERE id = 1", ());
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    txn.rollback().unwrap();
}

/// Seals a row in each of the next two DELETE rounds, then drops column
/// `x` and seals another in the third
fn seal_then_change_a_column(db: Database, rounds: Rc<Cell<usize>>, done: Rc<Cell<Option<bool>>>) {
    after_cold_round_prepared(move || {
        let round = rounds.get() + 1;
        rounds.set(round);
        if round < 3 {
            seal_row_apart(&db, 100 + round as i64, 100 + round as i64);
            return seal_then_change_a_column(db, rounds, done);
        }
        let (ddl, sealer) = (db.clone(), db);
        let dropped = on_thread(move || ddl.execute("ALTER TABLE t DROP COLUMN x", ()).is_ok());
        let sealed = on_thread(move || {
            sealer.execute("INSERT INTO t VALUES (60, 60)", ()).is_ok()
                && sealer.engine().seal_for_test().is_ok()
        });
        done.set(Some(dropped == Some(true) && sealed == Some(true)));
    });
}

#[test]
fn a_column_change_and_a_seal_in_the_last_round_of_a_delete_are_schema_changed() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let (rounds, done) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(None)));
    seal_then_change_a_column(db.clone(), Rc::clone(&rounds), Rc::clone(&done));
    let deleted = db.execute("DELETE FROM t WHERE v <= 20", ());
    assert!(
        matches!(deleted, Err(Error::SchemaChanged { .. })),
        "{deleted:?}"
    );
    assert_eq!(rounds.get(), 3, "the DELETE did not prepare three times");
    assert_eq!(
        done.take(),
        Some(true),
        "the column change and the seal did not both finish inside the third round"
    );
}

#[test]
fn a_refused_storage_delete_lets_go_of_the_claims_it_took() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20), (3, 30)]);
    // Another transaction holds row 2, uncommitted
    let mut holder = db.begin().unwrap();
    holder
        .execute("UPDATE t SET x = 7 WHERE id = 2", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    assert_eq!(table.delete_by_row_ids(&[3]).unwrap(), 1);
    // Row 1 is claimed first, then row 2 refuses the statement
    let refused = table.delete_by_row_ids(&[1, 2]);
    let free = execute_apart(&db, "UPDATE t SET x = 5 WHERE id = 1");
    let kept = execute_apart(&db, "UPDATE t SET x = 5 WHERE id = 3");
    drop(table);
    txn.rollback().unwrap();
    holder.rollback().unwrap();
    assert!(
        refused.is_err(),
        "row 2 is another transaction's: {refused:?}"
    );
    assert!(free, "row 1 is still claimed");
    assert!(!kept, "the earlier statement's claim on row 3 went");
}

#[test]
fn a_hot_half_refused_after_the_cold_claims_lets_go_of_them() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    db.execute("INSERT INTO t VALUES (2, 0, 20)", ()).unwrap();
    // Another transaction holds the hot row 2, uncommitted
    let mut holder = db.begin().unwrap();
    holder
        .execute("UPDATE t SET x = 7 WHERE id = 2", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    assert_eq!(table.delete_by_row_ids(&[3]).unwrap(), 1);
    let refused = table.delete_by_row_ids(&[1, 2]);
    let free = execute_apart(&db, "UPDATE t SET x = 5 WHERE id = 1");
    let kept = execute_apart(&db, "UPDATE t SET x = 5 WHERE id = 3");
    drop(table);
    txn.rollback().unwrap();
    holder.rollback().unwrap();
    assert!(
        refused.is_err(),
        "row 2 is another transaction's: {refused:?}"
    );
    assert!(free, "the cold row 1 is still claimed");
    assert!(!kept, "the earlier statement's claim on row 3 went");
}

#[test]
fn a_row_deleted_between_the_scan_and_the_delete_is_not_returned() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20), (3, 30)]);
    let other = db.clone();
    let done = Rc::new(Cell::new(false));
    let slot = Rc::clone(&done);
    after_delete_rows_scanned(move || {
        slot.set(execute_apart(&other, "DELETE FROM t WHERE id = 2"))
    });
    let returned: Vec<i64> = db
        .query("DELETE FROM t WHERE v >= 10 RETURNING id", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert!(done.get(), "the other delete did not commit after the scan");
    assert_eq!(returned, vec![1, 3]);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 0);
}

#[test]
fn a_scanned_id_named_twice_is_deleted_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20), (3, 30)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let mut ids = vec![1, 3, 1];
    let deleted = table.delete_scanned_rows(&mut ids).unwrap();
    drop(table);
    txn.commit().unwrap();
    assert_eq!(deleted, 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 1);
}

#[test]
fn a_volume_retired_during_a_delete_is_let_go_outside_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    seal_rows(&db, &[(5, 50)]);
    db.engine().cold_volumes_for_test("t");
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let compactor = db.clone();
    after_cold_round_prepared(move || {
        let compacted = on_thread(move || compactor.engine().compact_for_test().is_ok());
        assert_eq!(compacted, Some(true), "the compaction did not finish");
    });
    let (files, stores) = (volume_file_drops_by_fence(), block_store_drops_by_fence());
    let deleted = db.execute("DELETE FROM t WHERE v = 30", ()).unwrap();
    let (files, stores) = (
        delta(volume_file_drops_by_fence(), files),
        delta(block_store_drops_by_fence(), stores),
    );
    assert_eq!(deleted, 1);
    assert_eq!(
        db.engine().volume_stats().len(),
        1,
        "the volumes were not merged"
    );
    assert_eq!(files.1, 0, "a volume file let go under the seal fence");
    assert_eq!(stores.1, 0, "a block store let go under the seal fence");
    assert!(
        stores.0 > 0,
        "the DELETE let go of no loaded volume it held last"
    );
}

#[test]
fn a_prepared_row_deleted_and_sealed_again_with_other_values_is_a_write_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let other = db.clone();
    let done = Rc::new(Cell::new(false));
    let slot = Rc::clone(&done);
    after_cold_round_prepared(move || {
        let deleted = execute_apart(&other, "DELETE FROM t WHERE id = 1");
        let inserted = execute_apart(&other, "INSERT INTO t VALUES (1, 0, 99)");
        let sealed = on_thread(move || other.engine().seal_for_test().is_ok());
        slot.set(deleted && inserted && sealed == Some(true));
    });
    let deleted = db.execute("DELETE FROM t WHERE v <= 20", ());
    assert!(
        done.get(),
        "the delete, insert and seal did not finish in the round"
    );
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    assert_eq!(count(&db, "SELECT v FROM t WHERE id = 1"), 99);
}

#[test]
fn a_volume_retired_after_the_round_stood_is_let_go_outside_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    seal_rows(&db, &[(5, 50)]);
    db.engine().cold_volumes_for_test("t");
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let compactor = db.clone();
    let done = Rc::new(Cell::new(None));
    let slot = Rc::clone(&done);
    stoolap::test_failpoints::after_cold_round_settled(move || {
        slot.set(on_thread(move || {
            compactor.engine().compact_for_test().is_ok()
        }));
    });
    let (files, stores) = (volume_file_drops_by_fence(), block_store_drops_by_fence());
    let deleted = db.execute("DELETE FROM t WHERE v = 30", ()).unwrap();
    let (files, stores) = (
        delta(volume_file_drops_by_fence(), files),
        delta(block_store_drops_by_fence(), stores),
    );
    assert_eq!(done.get(), Some(true), "the compaction did not finish");
    assert_eq!(deleted, 1);
    assert_eq!(files.1, 0, "a volume file let go under the seal fence");
    assert_eq!(stores.1, 0, "a block store let go under the seal fence");
    assert!(
        stores.0 > 0,
        "the DELETE let go of no loaded volume it held last"
    );
}

#[test]
fn a_row_another_commit_made_hot_before_its_tombstone_fails_the_delete_before_any_change() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20), (3, 30)]);
    let (paused_tx, paused) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let other = db.clone();
    // The other commit stops after its hot versions and before its tombstones
    let committing = std::thread::spawn(move || {
        stoolap::test_failpoints::before_commit_marker(move || {
            let _ = paused_tx.send(());
            let _ = go_rx.recv_timeout(WATCHDOG);
        });
        other.execute("UPDATE t SET x = 9 WHERE id = 3", ())
    });
    let started = Rc::new(Cell::new(false));
    let slot = Rc::clone(&started);
    after_cold_round_prepared(move || {
        slot.set(paused.recv_timeout(WATCHDOG).is_ok());
    });
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    // Rows 1 and 2 come before row 3: a refusal at row 3's change would leave them changed
    let deleted = table.delete_by_row_ids(&[1, 2, 3]);
    let _ = go.send(());
    let updated = committing.join().unwrap();
    let free = execute_apart(&db, "UPDATE t SET x = 5 WHERE id = 1");
    drop(table);
    txn.rollback().unwrap();
    assert!(
        started.get(),
        "the other commit did not pause inside the round"
    );
    assert!(is_write_conflict(&deleted), "{deleted:?}");
    assert_eq!(updated.ok(), Some(1));
    assert!(
        free,
        "row 1 is still claimed: the DELETE changed rows before it failed"
    );
    assert_eq!(count(&db, "SELECT x FROM t WHERE id = 3"), 9);
}

/// Sets `x` to 9 on the rows `ids` names through `other`, a second handle
/// of the same transaction, once the next DELETE round is prepared; true
/// when it changed them all
fn update_through_another_handle(
    mut other: Box<dyn stoolap::storage::traits::Table>,
    ids: &'static [i64],
) -> Rc<Cell<Option<bool>>> {
    let done = Rc::new(Cell::new(None));
    let slot = Rc::clone(&done);
    after_cold_round_prepared(move || {
        let mut set = |mut row: stoolap::core::Row| {
            row.set(1, stoolap::core::Value::Integer(9))?;
            Ok((row, true))
        };
        let updated = other.update_by_row_ids(ids, &mut set).ok();
        slot.set(Some(updated == Some(ids.len() as i32)));
    });
    done
}

#[test]
fn a_named_row_its_transaction_changed_since_the_round_is_deleted_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let done = update_through_another_handle(txn.get_table("t").unwrap(), &[1]);
    let deleted = deleter.delete_by_row_ids(&[1]);
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        done.get(),
        Some(true),
        "the other handle did not update row 1"
    );
    assert_eq!(deleted.ok(), Some(1), "row 1 not deleted exactly once");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 1);
}

#[test]
fn every_row_with_one_its_transaction_changed_since_the_round_is_deleted_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let done = update_through_another_handle(txn.get_table("t").unwrap(), &[1]);
    let deleted = deleter.delete(None);
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        done.get(),
        Some(true),
        "the other handle did not update row 1"
    );
    assert_eq!(deleted.ok(), Some(2), "a row counted twice");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 0);
}

#[test]
fn a_row_its_transaction_changed_since_the_round_meets_the_filter_again() {
    use stoolap::core::{Operator, Value};
    use stoolap::storage::expression::{ComparisonExpr, Expression};
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let done = update_through_another_handle(txn.get_table("t").unwrap(), &[1]);
    let mut filter = ComparisonExpr::new("x", Operator::Eq, Value::Integer(0));
    filter.prepare_for_schema(deleter.schema());
    let deleted = deleter.delete(Some(&filter));
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        done.get(),
        Some(true),
        "the other handle did not update row 1"
    );
    assert_eq!(deleted.ok(), Some(1), "row 1 no longer has x = 0");
    assert_eq!(count(&db, "SELECT x FROM t WHERE id = 1"), 9);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 2"), 0);
}

#[test]
fn a_key_its_transaction_changed_since_the_round_is_deleted_once() {
    use stoolap::core::{Operator, Value};
    use stoolap::storage::expression::{ComparisonExpr, Expression};
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (2, 20)]);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let done = update_through_another_handle(txn.get_table("t").unwrap(), &[1]);
    let mut key = ComparisonExpr::new("id", Operator::Eq, Value::Integer(1));
    key.prepare_for_schema(deleter.schema());
    let deleted = deleter.delete(Some(&key));
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        done.get(),
        Some(true),
        "the other handle did not update row 1"
    );
    assert_eq!(deleted.ok(), Some(1), "one row counted twice");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 1);
}

#[test]
fn more_rows_than_fit_inline_its_transaction_changed_since_the_round_are_deleted_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(
        &db,
        &[
            (1, 10),
            (2, 20),
            (3, 30),
            (4, 40),
            (5, 50),
            (6, 60),
            (7, 70),
            (8, 80),
        ],
    );
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let done = update_through_another_handle(txn.get_table("t").unwrap(), &[6, 1, 5, 2, 4, 3]);
    let deleted = deleter.delete(None);
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        done.get(),
        Some(true),
        "the other handle did not update six rows"
    );
    assert_eq!(
        deleted.ok(),
        Some(8),
        "a changed row counted twice or not at all"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 0);
}
