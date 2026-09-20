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

//! Prototype: a secondary index on one integer column answers equality and
//! short ranges the same before a seal, after it, after a reopen and from
//! metadata-only volumes, against rows computed from the fixture's formula.

#![cfg(feature = "test-failpoints")]

use std::collections::BTreeMap;
use std::sync::Mutex;

use stoolap::api::Transaction;
use stoolap::storage::volume::secondary::COUNTERS;
use stoolap::{Database, IsolationLevel};

static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 20_000;
/// Eight rows share a key; the key is uncorrelated with the row order.
fn key_of(id: i64) -> i64 {
    ((id * 7919) % ROWS) / 8
}

fn expected_eq(k: i64) -> Vec<i64> {
    (1..=ROWS).filter(|&id| key_of(id) == k).collect()
}

fn expected_range(low: i64, high: i64) -> Vec<i64> {
    (1..=ROWS)
        .filter(|&id| (low..=high).contains(&key_of(id)))
        .collect()
}

fn open(dir: &std::path::Path, extra: &str) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100{extra}",
        dir.display()
    ))
    .unwrap()
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
}

fn insert(db: &Database, from: i64, to: i64) {
    for chunk_start in (from..=to).step_by(2000) {
        let chunk_end = (chunk_start + 1999).min(to);
        let values = (chunk_start..=chunk_end)
            .map(|id| format!("({id},{},{})", key_of(id), id * 3))
            .collect::<Vec<_>>()
            .join(",");
        db.execute(&format!("INSERT INTO t VALUES {values}"), ())
            .unwrap();
    }
}

fn ids_db(db: &Database, sql: &str, params: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = match params {
        [a] => db.query(sql, (*a,)).unwrap(),
        [a, b] => db.query(sql, (*a, *b)).unwrap(),
        _ => unreachable!(),
    }
    .map(|r| r.unwrap().get(0).unwrap())
    .collect();
    out.sort_unstable();
    out
}

fn ids_tx(tx: &mut Transaction, sql: &str, params: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = match params {
        [a] => tx.query(sql, (*a,)).unwrap(),
        [a, b] => tx.query(sql, (*a, *b)).unwrap(),
        _ => unreachable!(),
    }
    .map(|r| r.unwrap().get(0).unwrap())
    .collect();
    out.sort_unstable();
    out
}

const EQ: &str = "SELECT id FROM t WHERE k = $1";
const RANGE: &str = "SELECT id FROM t WHERE k >= $1 AND k <= $2";
const OPEN_RANGE: &str = "SELECT id FROM t WHERE k > $1 AND k < $2";

fn counters() -> BTreeMap<&'static str, u64> {
    COUNTERS.snapshot().into_iter().collect()
}

/// Every shape against the formula, on one database handle.
fn check_all(db: &Database, what: &str) {
    for k in [0, 1, 1234, 2498, 2499] {
        assert_eq!(ids_db(db, EQ, &[k]), expected_eq(k), "{what}: k = {k}");
    }
    assert_eq!(
        ids_db(db, EQ, &[2500]),
        Vec::<i64>::new(),
        "{what}: absent key"
    );
    assert_eq!(
        ids_db(db, RANGE, &[100, 103]),
        expected_range(100, 103),
        "{what}: range"
    );
    assert_eq!(
        ids_db(db, OPEN_RANGE, &[100, 104]),
        expected_range(101, 103),
        "{what}: open range"
    );
    assert_eq!(
        ids_db(db, RANGE, &[2499, 2600]),
        expected_eq(2499),
        "{what}: range past the end"
    );
}

#[test]
fn an_indexed_equality_and_range_answer_the_same_through_seal_reopen_and_eviction() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    check_all(&db, "hot");

    let before = counters();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let built = counters();
    assert_eq!(
        built["builds"] - before["builds"],
        1,
        "the seal built one index"
    );
    check_all(&db, "sealed");
    let after = counters();
    assert!(
        after["probes"] > built["probes"],
        "sealed lookups went through the index"
    );
    assert_eq!(after["unindexed_volumes"], built["unindexed_volumes"]);
    assert_eq!(
        after["candidates"] - built["candidates"],
        after["visible"] - built["visible"],
        "every candidate of an untouched volume is visible"
    );

    db.close().unwrap();
    let db = open(dir.path(), "");
    let reopened = counters();
    check_all(&db, "reopened");
    let after = counters();
    assert_eq!(
        after["loads"] - reopened["loads"],
        1,
        "the side file was read once"
    );
    assert!(after["probes"] > reopened["probes"]);

    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((volumes, cold), (1, 1), "the volume is metadata-only");
    let evicted = counters();
    check_all(&db, "metadata-only");
    let after = counters();
    assert!(after["probes"] > evicted["probes"]);
    assert_eq!(
        after["loads"], evicted["loads"],
        "the index stayed with the volume across tiers"
    );
}

#[test]
fn a_volume_sealed_before_its_index_existed_is_scanned_and_later_ones_are_probed() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    insert(&db, 1, ROWS / 2);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, ROWS / 2 + 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let before = counters();
    check_all(&db, "mixed");
    let after = counters();
    assert!(
        after["unindexed_volumes"] > before["unindexed_volumes"],
        "the older volume went to the scan"
    );
    assert!(
        after["probes"] > before["probes"],
        "the newer volume was probed"
    );
}

#[test]
fn a_snapshot_keeps_seeing_the_old_key_after_a_commit_moves_a_sealed_row() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let moved = expected_eq(700)[3];
    let mut snapshot = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(ids_tx(&mut snapshot, EQ, &[700]), expected_eq(700));

    db.execute("UPDATE t SET k = 2600 WHERE id = $1", (moved,))
        .unwrap();

    assert_eq!(
        ids_tx(&mut snapshot, EQ, &[700]),
        expected_eq(700),
        "the snapshot still sees the row under its old key"
    );
    assert_eq!(
        ids_tx(&mut snapshot, EQ, &[2600]),
        Vec::<i64>::new(),
        "and not under the new one"
    );
    snapshot.rollback().unwrap();

    let mut without = expected_eq(700);
    without.retain(|&id| id != moved);
    assert_eq!(
        ids_db(&db, EQ, &[700]),
        without,
        "a fresh read sees the move"
    );
    assert_eq!(ids_db(&db, EQ, &[2600]), vec![moved]);
    assert_eq!(ids_db(&db, RANGE, &[2599, 2601]), vec![moved]);
}

#[test]
fn a_deleted_sealed_row_is_hidden_from_its_own_transaction_first_and_from_everyone_after_commit() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let gone = expected_eq(900)[0];
    let mut without = expected_eq(900);
    without.retain(|&id| id != gone);

    let mut tx = db.begin().unwrap();
    tx.execute("DELETE FROM t WHERE id = $1", (gone,)).unwrap();
    assert_eq!(
        ids_tx(&mut tx, EQ, &[900]),
        without,
        "the transaction does not see its own deletion"
    );
    assert_eq!(
        ids_db(&db, EQ, &[900]),
        expected_eq(900),
        "others still do until it commits"
    );
    tx.commit().unwrap();
    assert_eq!(ids_db(&db, EQ, &[900]), without);
    assert_eq!(
        ids_db(&db, RANGE, &[899, 901]),
        expected_range(899, 901)
            .into_iter()
            .filter(|&id| id != gone)
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_compaction_inside_the_lookup_window_does_not_change_the_answer_and_the_output_is_indexed() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    for batch in 0..4 {
        insert(&db, batch * (ROWS / 4) + 1, (batch + 1) * (ROWS / 4));
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), 4);

    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(
            other.engine().volume_stats().len(),
            1,
            "the compaction merged the volumes"
        );
    });
    let mut tx = db.begin().unwrap();
    assert_eq!(
        ids_tx(&mut tx, RANGE, &[100, 103]),
        expected_range(100, 103)
    );
    tx.rollback().unwrap();
    stoolap::test_failpoints::after_cold_volumes_taken(|| {});

    let before = counters();
    check_all(&db, "compacted");
    let after = counters();
    assert!(
        after["probes"] > before["probes"],
        "the merged volume carries an index"
    );
    assert_eq!(after["unindexed_volumes"], before["unindexed_volumes"]);
}

#[test]
fn sixteen_volumes_answer_every_shape_through_their_indexes() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    for batch in 0..16 {
        insert(&db, batch * (ROWS / 16) + 1, (batch + 1) * (ROWS / 16));
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), 16);
    let before = counters();
    check_all(&db, "sixteen volumes");
    let after = counters();
    assert_eq!(after["unindexed_volumes"], before["unindexed_volumes"]);
    assert!(
        after["probes"] - before["probes"] >= 16,
        "each present key probed several volumes"
    );
}
