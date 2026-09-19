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

//! A LIMIT read takes its hot rows and its cold view at different moments.
//! A seal landing between them must neither lose the rows it moved nor show
//! them twice, and a retry the next seal lands in is held to the same rule.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use stoolap::core::RowVec;
use stoolap::storage::traits::{Engine, Table};
use stoolap::Database;

const BATCH: i64 = 1000;

fn insert_batch(db: &Database, batch: i64) {
    let values = (batch * BATCH + 1..=batch * BATCH + BATCH)
        .map(|id| format!("({id},{})", id % 7))
        .collect::<Vec<_>>()
        .join(",");
    db.execute(&format!("INSERT INTO t VALUES {values}"), ())
        .unwrap();
}

fn seal(db: &Database, volumes_after: usize) {
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        db.engine().volume_stats().len(),
        volumes_after,
        "the seal landed inside the read's window"
    );
}

/// Two sealed volumes and a third batch still hot.
fn fixture() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    for batch in 0..3 {
        insert_batch(&db, batch);
        if batch < 2 {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }
    assert_eq!(db.engine().volume_stats().len(), 2);
    (dir, db)
}

fn collect(table: &dyn Table, unordered: bool) -> RowVec {
    if unordered {
        table.collect_rows_with_limit_unordered(None, 100_000, 0)
    } else {
        table.collect_rows_with_limit(None, 100_000, 0)
    }
    .unwrap()
}

/// The ids a read returned, in order, with the count it returned them in.
fn ids(rows: &RowVec) -> (usize, Vec<i64>) {
    let mut ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    let count = ids.len();
    ids.dedup();
    (count, ids)
}

fn every_id_up_to(batches: i64) -> Vec<i64> {
    (1..=batches * BATCH).collect()
}

/// A hook a read never reached would otherwise drop its database handle in
/// the thread's own destructor, which aborts the process instead of failing
/// the test.
fn disarm() {
    stoolap::test_failpoints::after_cold_volumes_taken(|| {});
    stoolap::test_failpoints::after_hot_rows_taken(|| {});
}

#[test]
fn an_ordered_limit_keeps_the_rows_a_seal_moves_after_its_cold_view() {
    let (_dir, db) = fixture();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || seal(&other, 3));

    let mut tx = db.engine().begin_transaction().unwrap();
    let rows = collect(&*tx.get_table("t").unwrap(), false);
    tx.rollback().unwrap();
    disarm();

    assert_eq!(ids(&rows), (3000, every_id_up_to(3)));
}

#[test]
fn an_unordered_limit_shows_once_the_rows_a_seal_moves_after_its_hot_read() {
    let (_dir, db) = fixture();
    let other = db.clone();
    stoolap::test_failpoints::after_hot_rows_taken(move || seal(&other, 3));

    let mut tx = db.engine().begin_transaction().unwrap();
    let rows = collect(&*tx.get_table("t").unwrap(), true);
    tx.rollback().unwrap();
    disarm();

    assert_eq!(ids(&rows), (3000, every_id_up_to(3)));
}

/// The first attempt is dropped for the seal that landed in it; the second
/// attempt is hit by another seal, which moved a batch inserted meanwhile,
/// and is dropped the same way rather than trusted.
#[test]
fn a_retry_a_seal_lands_in_again_is_verified_too() {
    let (_dir, db) = fixture();
    let seals = Arc::new(AtomicUsize::new(0));
    let other = db.clone();
    let counted = Arc::clone(&seals);
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        seal(&other, 3);
        counted.fetch_add(1, Ordering::Relaxed);
        insert_batch(&other, 3);
        let again = other.clone();
        let counted = Arc::clone(&counted);
        stoolap::test_failpoints::after_cold_volumes_taken(move || {
            seal(&again, 4);
            counted.fetch_add(1, Ordering::Relaxed);
        });
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let rows = collect(&*tx.get_table("t").unwrap(), false);
    tx.rollback().unwrap();
    disarm();

    assert_eq!(
        seals.load(Ordering::Relaxed),
        2,
        "both seals landed in a window"
    );
    assert_eq!(ids(&rows), (4000, every_id_up_to(4)));
}

/// Five attempts in a row are each hit by a seal that moved a fresh batch,
/// and each is dropped for it, whatever its number; the sixth, which no seal
/// lands in, answers with every batch.
#[test]
fn every_attempt_a_seal_lands_in_is_dropped_however_many_in_a_row() {
    const SEALS: usize = 5;
    let (_dir, db) = fixture();
    let seals = Arc::new(AtomicUsize::new(0));

    fn arm(db: Database, seals: Arc<AtomicUsize>) {
        stoolap::test_failpoints::after_hot_rows_taken(move || {
            let landed = seals.fetch_add(1, Ordering::Relaxed) + 1;
            seal(&db, 2 + landed);
            if landed < SEALS {
                insert_batch(&db, 2 + landed as i64);
                arm(db, seals);
            }
        });
    }
    arm(db.clone(), Arc::clone(&seals));

    let mut tx = db.engine().begin_transaction().unwrap();
    let rows = collect(&*tx.get_table("t").unwrap(), true);
    tx.rollback().unwrap();
    disarm();

    assert_eq!(seals.load(Ordering::Relaxed), SEALS);
    assert_eq!(ids(&rows), (7000, every_id_up_to(7)));
}
