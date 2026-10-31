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

//! AS OF reads a table as it stood when the commits completed by then had
//! completed, and fails when the database no longer holds what that needs.

#![cfg(feature = "test-failpoints")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use stoolap::core::{Row, Value};
use stoolap::storage::mvcc::MVCCTable;
use stoolap::storage::traits::{Engine, Table};
use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

type Rows = Result<Vec<(i64, i64)>, String>;

fn memory(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db
}

fn file(dir: &tempfile::TempDir) -> Database {
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db
}

/// The wall clock as AS OF TIMESTAMP reads it, apart from the commits
/// around it
fn now(db: &Database) -> String {
    std::thread::sleep(Duration::from_millis(3));
    let time: String = db
        .query("SELECT CAST(NOW() AS TEXT)", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    std::thread::sleep(Duration::from_millis(3));
    time
}

fn rows(db: &Database, sql: &str) -> Rows {
    let mut out: Vec<(i64, i64)> = db
        .query(sql, ())
        .map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()))
        })
        .collect::<Result<_, String>>()?;
    out.sort_unstable();
    Ok(out)
}

fn at(db: &Database, time: &str) -> Rows {
    rows(db, &format!("SELECT id, v FROM t AS OF TIMESTAMP '{time}'"))
}

fn after(db: &Database, txn: i64) -> Rows {
    rows(db, &format!("SELECT id, v FROM t AS OF TRANSACTION {txn}"))
}

fn refused(answer: Rows, why: &str) {
    match answer {
        Err(error) => assert!(error.contains("AS OF"), "{why}: {error}"),
        Ok(rows) => panic!("{why}: answered {rows:?}"),
    }
}

#[test]
fn a_row_written_before_a_time_and_committed_after_it_is_absent_at_it() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_written_before");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let before = now(&db);
    tx.commit().unwrap();
    assert_eq!(at(&db, &before), Ok(vec![]), "the commit completed after");
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10)]));
}

#[test]
fn a_point_beyond_the_kept_history_fails_instead_of_losing_a_row() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_beyond");
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let before = now(&db);
    for v in 0..15 {
        db.execute("UPDATE t SET v = $1 WHERE id = 2", (100 + v,))
            .unwrap();
    }
    refused(at(&db, &before), "row 2's version at that time was cut");
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10), (2, 114)]));
}

#[test]
fn a_delete_counts_from_its_commit() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_delete");
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let before = now(&db);
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let after_delete = now(&db);
    assert_eq!(at(&db, &before), Ok(vec![(1, 10), (2, 20)]));
    assert_eq!(at(&db, &after_delete), Ok(vec![(2, 20)]));
}

#[test]
fn a_rolled_back_write_is_never_visible() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_rolled_back");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    tx.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    assert_eq!(at(&db, "2200-01-01"), Ok(vec![(1, 10)]), "uncommitted");
    tx.rollback().unwrap();
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10)]), "rolled back");
}

#[test]
fn as_of_transaction_reads_right_after_it_completed() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_transaction");
    let mut first = db.begin().unwrap();
    let first_id = first.id();
    let mut second = db.begin().unwrap();
    let second_id = second.id();
    assert!(first_id < second_id);
    second.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    second.commit().unwrap();
    first.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    first.commit().unwrap();
    assert_eq!(
        after(&db, second_id),
        Ok(vec![(2, 20)]),
        "begun later, completed first"
    );
    assert_eq!(after(&db, first_id), Ok(vec![(1, 10), (2, 20)]));

    let mut active = db.begin().unwrap();
    active.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    refused(after(&db, active.id()), "an active transaction");
    let id = active.id();
    active.rollback().unwrap();
    refused(after(&db, id), "a rolled back transaction");
    refused(after(&db, 999_999), "an unknown transaction");
    let mut reader = db.begin().unwrap();
    let reader_id = reader.id();
    reader.query("SELECT * FROM t", ()).unwrap();
    reader.commit().unwrap();
    refused(after(&db, reader_id), "a read-only transaction");
}

#[test]
fn a_commit_waiting_to_become_visible_stays_out_of_a_point_taken_meanwhile() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_waiting");
    let other = db.clone();
    let seen = Rc::new(RefCell::new(None));
    let noted = Rc::clone(&seen);
    fp::before_commit_visible(move || {
        other.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
        let time = now(&other);
        *noted.borrow_mut() = Some((time.clone(), at(&other, &time)));
    });
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let (time, while_waiting) = seen.borrow_mut().take().unwrap();
    assert_eq!(while_waiting, Ok(vec![(2, 20)]), "A waits, B completed");
    assert_eq!(at(&db, &time), Ok(vec![(2, 20)]), "A completed later");
}

#[test]
fn a_point_that_needs_an_evicted_entry_is_refused() {
    let _guard = FailpointGuard::new();
    fp::COMMIT_LOG_CAPACITY.store(4, Ordering::Release);
    let db = memory("as_of_history_evicted");
    let other = db.clone();
    let seen = Rc::new(RefCell::new(None));
    let noted = Rc::clone(&seen);
    fp::before_commit_visible(move || {
        for id in 2..10 {
            other
                .execute("INSERT INTO t VALUES ($1, $1)", (id,))
                .unwrap();
        }
        let time = now(&other);
        *noted.borrow_mut() = Some(at(&other, &time));
    });
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    let while_waiting = seen.borrow_mut().take().unwrap();
    refused(
        while_waiting,
        "the point's own entry is kept, row 2's is not",
    );
}

#[test]
fn a_seal_refuses_a_point_that_has_not_seen_its_cutoff() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = file(&dir);
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let mut writer = db.begin().unwrap();
    let before = now(&db);
    writer
        .execute("UPDATE t SET v = 11 WHERE id = 1", ())
        .unwrap();
    writer.commit().unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    refused(at(&db, &before), "the seal moved the update and dropped 10");
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 11)]));
}

#[test]
fn a_point_after_the_newest_seal_reads_the_volumes() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = file(&dir);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let sealed = now(&db);
    db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    let updated = now(&db);
    db.execute("UPDATE t SET v = 12 WHERE id = 1", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(at(&db, &sealed), Ok(vec![(1, 10), (2, 20), (3, 30)]));
    assert_eq!(
        at(&db, &updated),
        Ok(vec![(1, 11), (2, 20), (3, 30)]),
        "tombstoned at the first update, again at the second"
    );
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 12), (3, 30)]));
}

#[test]
fn a_truncate_refuses_the_points_before_it() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_truncate");
    let other = db.clone();
    db.execute("BEGIN", ()).unwrap();
    db.query("SELECT 1", ()).unwrap();
    other.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let before = now(&other);
    db.execute("TRUNCATE TABLE t", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    refused(at(&other, &before), "the truncate took the rows away");
    assert_eq!(
        at(&other, &now(&other)),
        Ok(vec![]),
        "no later write needed"
    );
}

#[test]
fn a_failed_truncate_still_completes_its_event() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_failed_truncate");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let writer = db.clone();
    writer.execute("BEGIN", ()).unwrap();
    writer
        .execute("UPDATE t SET v = 11 WHERE id = 1", ())
        .unwrap();
    assert!(db.execute("TRUNCATE TABLE t", ()).is_err());
    writer.execute("ROLLBACK", ()).unwrap();
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10)]));
}

#[test]
fn an_alter_refuses_the_points_before_it() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_alter");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let before = now(&db);
    db.execute("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7", ())
        .unwrap();
    refused(at(&db, &before), "no historical schema");
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10)]));
}

#[test]
fn a_clock_step_back_does_not_move_a_passed_time() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_clock");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let passed = now(&db);
    assert_eq!(at(&db, &passed), Ok(vec![(1, 10)]));
    fp::COMMIT_CLOCK_SHIFT.store(-3_600_000_000_000, Ordering::Release);
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    assert_eq!(at(&db, &passed), Ok(vec![(1, 10)]), "row 2 stays after it");
}

#[test]
fn the_open_point_reads_what_was_created_before_the_first_write() {
    let _guard = FailpointGuard::new();
    fp::COMMIT_LOG_CAPACITY.store(4, Ordering::Release);
    let db = memory("as_of_history_open");
    let created = now(&db);
    assert_eq!(at(&db, &created), Ok(vec![]), "before the first write");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    assert_eq!(at(&db, &created), Ok(vec![]), "after it");
    for id in 2..10 {
        db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
    }
    refused(at(&db, &created), "the first entries were evicted");
    refused(at(&db, "1999-01-01"), "before the open");
}

#[test]
fn a_ring_overwritten_while_read_voids_the_read() {
    let _guard = FailpointGuard::new();
    fp::COMMIT_LOG_CAPACITY.store(4, Ordering::Release);
    let db = memory("as_of_history_overwritten");
    for id in 1..4 {
        db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
    }
    let other = db.clone();
    fp::before_commit_log_slots_read(move || {
        for id in 10..20 {
            other
                .execute("INSERT INTO t VALUES ($1, $1)", (id,))
                .unwrap();
        }
    });
    refused(at(&db, "2200-01-01"), "the newest slot was overwritten");
}

#[test]
fn vacuum_keeps_the_points_after_the_versions_it_drops() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_vacuum");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let first = now(&db);
    db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    let second = now(&db);
    db.execute("VACUUM", ()).unwrap();
    refused(at(&db, &first), "10 was dropped");
    assert_eq!(at(&db, &second), Ok(vec![(1, 11)]));
}

#[test]
fn a_reopened_database_starts_its_history_at_the_open() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let before = {
        let db = file(&dir);
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
        let before = now(&db);
        db.close().unwrap();
        before
    };
    let db = file(&dir);
    refused(at(&db, &before), "before the open");
    let opened = now(&db);
    assert_eq!(at(&db, &opened), Ok(vec![(1, 10)]), "the recovered state");
    db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    assert_eq!(at(&db, &opened), Ok(vec![(1, 10)]));
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 11)]));
}

#[test]
fn a_time_not_passed_yet_reads_the_newest_state() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_future");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    assert_eq!(at(&db, "2200-01-01"), Ok(vec![(1, 10)]));
}

#[test]
fn a_compaction_during_a_read_refuses_the_point_it_overtook() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = file(&dir);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 40), (5, 50), (6, 60)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let point = now(&db);
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    let other = db.clone();
    fp::after_as_of_hot_read(move || other.engine().compact_for_test().unwrap());
    let answer = at(&db, &point);
    assert!(
        answer.is_err() || answer.as_ref().is_ok_and(|rows| rows.len() == 6),
        "row 2 stood at the point: {answer:?}"
    );
}

#[test]
fn a_row_published_before_its_transaction_committed_is_never_history() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_early_publication");
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut writer = early_writer(&db, txn.id());
    writer.insert(row(1, 10)).unwrap();
    publish(&mut writer);
    assert_eq!(at(&db, "2200-01-01"), Ok(vec![(2, 20)]), "still active");
    txn.rollback().unwrap();
    assert_eq!(at(&db, "2200-01-01"), Ok(vec![(2, 20)]), "rolled back");

    let mut txn = db.engine().begin_transaction().unwrap();
    let mut writer = early_writer(&db, txn.id());
    writer.insert(row(3, 30)).unwrap();
    publish(&mut writer);
    // Nothing local is left, so the commit takes no sequence
    txn.commit().unwrap();
    assert_eq!(rows(&db, "SELECT id, v FROM t"), Ok(vec![(2, 20), (3, 30)]));
    refused(at(&db, "2200-01-01"), "a commit without a sequence");
}

/// A handle on table t for transaction `txn_id`, outside the transaction's own tables
fn early_writer(db: &Database, txn_id: i64) -> MVCCTable {
    use std::sync::Arc;
    use stoolap::storage::mvcc::version_store::TransactionVersionStore;

    let store = db.engine().get_version_store("t").unwrap();
    let local = TransactionVersionStore::new(Arc::clone(&store), txn_id);
    MVCCTable::new(txn_id, store, local)
}

/// Publishes what `handle` wrote to the shared store before its transaction commits
fn publish(handle: &mut MVCCTable) {
    handle
        .txn_versions()
        .unwrap()
        .write_store()
        .commit()
        .unwrap();
}

fn row(id: i64, v: i64) -> Row {
    Row::from(vec![Value::Integer(id), Value::Integer(v)])
}

#[test]
fn a_chain_cut_by_early_published_writes_keeps_the_committed_version() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_unstamped_cut");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut writer = early_writer(&db, txn.id());
    for v in 11..=22 {
        writer
            .txn_versions()
            .unwrap()
            .write_store()
            .put(1, row(1, v), false)
            .unwrap();
        publish(&mut writer);
    }
    assert_eq!(at(&db, &now(&db)), Ok(vec![(1, 10)]));
    txn.rollback().unwrap();
}

#[test]
fn retention_places_an_early_published_delete_by_its_commit() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_unstamped_retention");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut writer = early_writer(&db, txn.id());
    writer.delete_by_row_ids(&[1]).unwrap();
    publish(&mut writer);
    txn.get_table("t").unwrap().insert(row(2, 20)).unwrap();
    txn.commit().unwrap();
    now(&db);
    let store = db.engine().get_version_store("t").unwrap();
    assert_eq!(store.cleanup_deleted_rows(Duration::ZERO), 1);
}

#[test]
fn transaction_zero_is_no_history_event() {
    let _guard = FailpointGuard::new();
    let db = memory("as_of_history_transaction_zero");
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("ALTER TABLE t ADD COLUMN w INTEGER", ())
        .unwrap();
    refused(after(&db, 0), "no transaction has id 0");
    refused(after(&db, -1), "nor a negative one");
}
