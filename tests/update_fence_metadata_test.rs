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

//! An UPDATE reads its sealed rows outside the seal fence and, under it,
//! checks only what memory holds; three rounds that do not stand end in a
//! write conflict before any change

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::core::{Operator, Row, Value};
use stoolap::storage::expression::ComparisonExpr;
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{
    after_cold_round_prepared, after_cold_round_settled, block_file_reads, cold_rounds,
    side_file_reads, volume_file_reads_by_fence, FailpointGuard,
};
use stoolap::Database;

/// Four sealed volumes of 1,000 rows (ids 1..=4000), k = id * 10 unique
fn sealed_table(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX t_k ON t(k)", ()).unwrap();
    for b in 0..4 {
        let values: Vec<String> = (b * 1000 + 1..=(b + 1) * 1000)
            .map(|id| format!("({id}, {}, 0)", id * 10))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn v_of(db: &Database, id: i64) -> Option<i64> {
    db.query(&format!("SELECT v FROM t WHERE id = {id}"), ())
        .unwrap()
        .next()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
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

/// Lands a seal in each of the next three cold rounds; the third seals
/// row 9001 (k = 5) and leaves every volume cold, so a fourth attempt
/// finds 9001 in a volume it has to read from its file
fn seal_in_three_rounds(db: &Database) {
    fn seal(db: &Database, id: i64, k: i64) {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {k}, 0)"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let (a, b, c) = (db.clone(), db.clone(), db.clone());
    after_cold_round_prepared(move || {
        seal(&a, 7001, 70010);
        after_cold_round_prepared(move || {
            seal(&b, 8001, 80010);
            after_cold_round_prepared(move || {
                seal(&c, 9001, 5);
                c.engine().cold_volumes_for_test("t");
            });
        });
    });
}

fn delta(after: (usize, usize), before: (usize, usize)) -> (usize, usize) {
    (after.0 - before.0, after.1 - before.1)
}

/// File reads under the seal fence so far: whole volumes, side files
fn other_reads_under_fence() -> (usize, usize) {
    (volume_file_reads_by_fence().1, side_file_reads().1)
}

fn set_v(calls: &mut Vec<i64>) -> impl FnMut(Row) -> stoolap::core::Result<(Row, bool)> + '_ {
    move |mut row| {
        if let Some(Value::Integer(id)) = row.get(0) {
            calls.push(*id);
        }
        row.set(2, Value::Integer(4))?;
        Ok((row, true))
    }
}

#[test]
fn three_rounds_a_seal_moved_end_in_a_write_conflict_with_no_read_under_the_fence_by_id() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    db.engine().cold_volumes_for_test("t");
    let _cache = NoGroupCache::set(&db);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    seal_in_three_rounds(&db);
    let before = block_file_reads();
    let other_before = other_reads_under_fence();
    let mut calls = Vec::new();
    let result = table.update_by_row_ids(&[5, 9001], &mut set_v(&mut calls));
    let (outside, under) = delta(block_file_reads(), before);
    let other_under = delta(other_reads_under_fence(), other_before);
    // The claims this statement took are gone: another transaction sets row 5
    let other = db.execute("UPDATE t SET v = 9 WHERE id = 5", ());
    drop(table);
    txn.rollback().unwrap();
    assert_eq!(under, 0, "blocks read under the fence ({outside} outside)");
    assert_eq!(
        other_under,
        (0, 0),
        "volumes, side files read under the fence"
    );
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.to_string().contains("kept changing")),
        "{result:?}"
    );
    assert!(other.is_ok(), "row 5 stayed claimed: {other:?}");
    calls.sort_unstable();
    calls.dedup();
    assert_eq!(calls, [5], "the setter ran once, on row 5 only");
}

#[test]
fn three_rounds_a_seal_moved_end_in_a_write_conflict_with_no_read_under_the_fence_by_filter() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    db.engine().cold_volumes_for_test("t");
    let _cache = NoGroupCache::set(&db);
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    seal_in_three_rounds(&db);
    let before = block_file_reads();
    let other_before = other_reads_under_fence();
    let mut calls = Vec::new();
    // Rows 1..=5 (k 10..50), and 9001 (k 5) once sealed
    let filter = ComparisonExpr::new("k", Operator::Lt, Value::Integer(60));
    let result = table.update(Some(&filter), &mut set_v(&mut calls));
    let (outside, under) = delta(block_file_reads(), before);
    let other_under = delta(other_reads_under_fence(), other_before);
    drop(table);
    txn.rollback().unwrap();
    assert_eq!(under, 0, "blocks read under the fence ({outside} outside)");
    assert_eq!(
        other_under,
        (0, 0),
        "volumes, side files read under the fence"
    );
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.to_string().contains("kept changing")),
        "{result:?}"
    );
    assert!(!calls.contains(&9001), "the setter ran on 9001: {calls:?}");
    assert_eq!(v_of(&db, 5), Some(0), "row 5 changed");
}

#[test]
fn a_column_change_met_in_the_last_round_is_a_schema_change_not_a_write_conflict() {
    use std::sync::mpsc;
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    db.execute("ALTER TABLE t ADD COLUMN w INTEGER", ())
        .unwrap();
    let (resume, resumed) = mpsc::channel::<()>();
    let ddl_db = db.clone();
    let ddl = std::rc::Rc::new(std::cell::RefCell::new(None));
    let ddl_slot = std::rc::Rc::clone(&ddl);
    let (a, b, c) = (db.clone(), db.clone(), db.clone());
    let seal = |db: &Database, id: i64| {
        db.execute(
            &format!("INSERT INTO t VALUES ({id}, {}, 0, 0)", id * 10),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    };
    // A seal in every round; in the last one a column change also starts
    // and holds the table closed
    after_cold_round_prepared(move || {
        seal(&a, 7001);
        after_cold_round_prepared(move || {
            seal(&b, 8001);
            after_cold_round_prepared(move || {
                seal(&c, 9001);
                let (paused, is_paused) = mpsc::channel::<()>();
                *ddl_slot.borrow_mut() = Some(std::thread::spawn(move || {
                    stoolap::test_failpoints::after_drop_hot_published(move || {
                        paused.send(()).unwrap();
                        resumed.recv().unwrap();
                    });
                    ddl_db.execute("ALTER TABLE t DROP COLUMN w", ())
                }));
                is_paused.recv().unwrap();
            });
        });
    });
    let updated = db.execute("UPDATE t SET v = 4 WHERE id = 5", ());
    resume.send(()).unwrap();
    let handle = ddl.borrow_mut().take().expect("the column change started");
    handle.join().unwrap().unwrap();
    assert!(
        matches!(updated, Err(stoolap::core::Error::SchemaChanged { .. })),
        "{updated:?}"
    );
}

#[test]
fn a_round_reads_its_sealed_rows_from_the_file_outside_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    db.engine().cold_volumes_for_test("t");
    let _cache = NoGroupCache::set(&db);
    let before = block_file_reads();
    db.execute("UPDATE t SET v = 4 WHERE id = 5", ()).unwrap();
    let (outside, under) = delta(block_file_reads(), before);
    assert!(outside > 0, "no block read from the file");
    assert_eq!(under, 0, "blocks read under the fence");
    assert_eq!(v_of(&db, 5), Some(4));
}

#[test]
fn a_sealed_row_deleted_after_its_round_is_a_write_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    let other = db.clone();
    // Committed with no seal: the seal generation does not move
    after_cold_round_prepared(move || {
        other.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    });
    let updated = db.execute("UPDATE t SET v = 4 WHERE id = 5", ());
    assert!(
        updated
            .as_ref()
            .is_err_and(|e| e.to_string().contains("write conflict: row 5")),
        "{updated:?}"
    );
    assert_eq!(v_of(&db, 5), None, "the deleted row came back");
}

#[test]
fn a_sealed_row_compaction_moved_after_its_round_is_read_again_in_a_new_round() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    let other = db.clone();
    // A compaction alone moves the rows without moving the seal generation
    after_cold_round_prepared(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let before = db.engine().volume_stats().len();
    let rounds = cold_rounds();
    let updated = db.execute("UPDATE t SET v = 4 WHERE id = 5", ());
    let rounds = cold_rounds() - rounds;
    let after = db.engine().volume_stats().len();
    assert!(after < before, "the compaction ran: {before} -> {after}");
    assert!(matches!(updated, Ok(1)), "{updated:?}");
    assert_eq!(v_of(&db, 5), Some(4));
    assert_eq!(rounds, 2, "the moved row was read again in a second round");
}

#[test]
fn a_seal_between_the_settled_generation_and_the_snapshot_repeats_the_round() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    let other = db.clone();
    // The snapshot takes the map this seal published, so only the
    // generation read before it shows the seal
    stoolap::test_failpoints::before_cold_round_snapshot(move || {
        other
            .execute("INSERT INTO t VALUES (9001, 90010, 0)", ())
            .unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let rounds = cold_rounds();
    let updated = db.execute("UPDATE t SET v = 4 WHERE id = 5", ());
    let rounds = cold_rounds() - rounds;
    assert!(matches!(updated, Ok(1)), "{updated:?}");
    assert_eq!(v_of(&db, 5), Some(4));
    assert_eq!(rounds, 2, "the round stood across the seal");
}

#[test]
fn a_compaction_after_the_fence_check_keeps_one_updated_version() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(dir.path());
    let other = db.clone();
    // Compaction publishes without the seal write fence: it can land after
    // the round stood its checks and before the rows are claimed
    after_cold_round_settled(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let before = db.engine().volume_stats().len();
    let updated = db.execute("UPDATE t SET v = 4 WHERE id = 5", ());
    let after = db.engine().volume_stats().len();
    assert!(after < before, "the compaction ran: {before} -> {after}");
    assert!(matches!(updated, Ok(1)), "{updated:?}");
    let rows: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 5", ())
        .unwrap();
    assert_eq!((rows, v_of(&db, 5)), (1, Some(4)));
}
