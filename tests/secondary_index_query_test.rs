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

//! The query path over a volume's secondary index side file: an equality
//! or a range on an indexed INTEGER column answers the same rows hot,
//! sealed, reopened and from a metadata-only volume, against rows computed
//! from the fixture's formula, and every visibility rule of the scan holds
//! for a row named by the index: the hot version, the tombstone under the
//! snapshot rule, the transaction's own deletion, a seal or a compaction
//! inside the window. A refused working reservation and an ineligible side
//! file send the volume through the scan with the same answer.

use std::collections::BTreeMap;
use std::sync::Mutex;

use stoolap::api::Transaction;
use stoolap::{Database, IsolationLevel};

/// The budgets and counters are process-wide, so the tests run one at a
/// time whatever the runner does.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// The read counters of `PRAGMA INDEX_READ_STATS`, by column name
fn reads(db: &Database) -> BTreeMap<String, i64> {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let columns: Vec<String> = rows.columns().to_vec();
    let row = rows.into_iter().next().unwrap().unwrap();
    columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), row.get::<i64>(i).unwrap()))
        .collect()
}

fn delta(after: &BTreeMap<String, i64>, before: &BTreeMap<String, i64>, key: &str) -> i64 {
    after[key] - before[key]
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
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    check_all(&db, "hot");

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = reads(&db);
    check_all(&db, "sealed");
    let after = reads(&db);
    assert!(
        delta(&after, &before, "probes") > 0,
        "sealed lookups went through the side file"
    );
    assert_eq!(delta(&after, &before, "ineligible"), 0);
    assert_eq!(delta(&after, &before, "refused"), 0);
    assert_eq!(
        delta(&after, &before, "candidates"),
        delta(&after, &before, "rows"),
        "every candidate of an untouched volume is visible"
    );

    db.close().unwrap();
    drop(db);
    let db = open(dir.path(), "");
    let before = reads(&db);
    check_all(&db, "reopened");
    let after = reads(&db);
    assert!(delta(&after, &before, "probes") > 0);
}

/// A metadata-only volume is probed on its side file before any data is
/// reloaded: a miss reloads nothing, a hit reloads the volume once.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_metadata_only_volume_is_probed_before_any_reload() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    // Key 1000 is taken out, so a range on it is inside the zone map yet
    // empty in the side file
    db.execute("DELETE FROM t WHERE k = 1000", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((volumes, cold), (1, 1), "the volume is metadata-only");
    // An absent key: the bloom filter or the side file answers, and no
    // data is reloaded either way
    let before = reads(&db);
    assert_eq!(ids_db(&db, EQ, &[2500]), Vec::<i64>::new());
    let after = reads(&db);
    assert_eq!(
        delta(&after, &before, "reloads"),
        0,
        "a miss on a metadata-only volume reloads no data"
    );
    // A range inside the zone map (ranges take no bloom filter) that the
    // side file finds empty: probed, missed, nothing reloaded
    let before = reads(&db);
    assert_eq!(ids_db(&db, RANGE, &[1000, 1000]), Vec::<i64>::new());
    let after = reads(&db);
    assert_eq!(delta(&after, &before, "probes"), 1);
    assert_eq!(delta(&after, &before, "misses"), 1);
    assert_eq!(delta(&after, &before, "reloads"), 0);
    // A hit reloads the volume once, after the probe
    let before = reads(&db);
    assert_eq!(ids_db(&db, EQ, &[1234]), expected_eq(1234));
    let after = reads(&db);
    assert_eq!(delta(&after, &before, "probes"), 1);
    assert_eq!(
        delta(&after, &before, "reloads"),
        1,
        "a hit reloads the volume"
    );
    for k in [0, 1, 1234, 2499] {
        assert_eq!(
            ids_db(&db, EQ, &[k]),
            expected_eq(k),
            "metadata-only: k = {k}"
        );
    }
}

#[test]
fn a_volume_sealed_before_its_index_existed_is_scanned_and_later_ones_are_probed() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    insert(&db, 1, ROWS / 2);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, ROWS / 2 + 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let before = reads(&db);
    check_all(&db, "mixed");
    let after = reads(&db);
    assert!(
        delta(&after, &before, "ineligible") > 0,
        "the older volume went to the scan"
    );
    assert!(
        delta(&after, &before, "probes") > 0,
        "the newer volume was probed"
    );
}

/// The hot version is authoritative: a sealed row updated in hot, whether
/// its key stays in the filter or leaves it, is answered from hot or not
/// at all, never as its cold copy too.
#[test]
fn a_hot_version_shadows_the_cold_copy_whether_or_not_it_still_matches() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let same = expected_eq(300)[1];
    let moved = expected_eq(300)[5];
    // One row keeps its key and changes its payload; one leaves the key
    db.execute("UPDATE t SET v = -1 WHERE id = $1", (same,))
        .unwrap();
    db.execute("UPDATE t SET k = 2600 WHERE id = $1", (moved,))
        .unwrap();
    let mut want = expected_eq(300);
    want.retain(|&id| id != moved);
    assert_eq!(
        ids_db(&db, EQ, &[300]),
        want,
        "once each, the moved row gone"
    );
    let payload: i64 = db
        .query_one("SELECT v FROM t WHERE k = 300 AND id = $1", (same,))
        .unwrap();
    assert_eq!(
        payload, -1,
        "the hot version's payload, not the cold copy's"
    );
    assert_eq!(ids_db(&db, EQ, &[2600]), vec![moved]);
    assert_eq!(
        ids_db(&db, RANGE, &[299, 301]),
        expected_range(299, 301)
            .into_iter()
            .filter(|&id| id != moved)
            .collect::<Vec<_>>()
    );
    // Sealed again, the hot versions become cold rows and the old copies
    // carry tombstones: the same answer
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(ids_db(&db, EQ, &[300]), want);
    assert_eq!(ids_db(&db, EQ, &[2600]), vec![moved]);
}

#[test]
fn a_snapshot_keeps_seeing_the_old_key_after_a_commit_moves_a_sealed_row() {
    let _serial = serial();
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
    let _serial = serial();
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

/// A refused working reservation sends the volume through the scan with
/// the same rows; an ineligible side file (its index dropped and created
/// again) does the same until the next compaction covers the volume.
#[test]
fn a_refused_reservation_and_an_ineligible_side_file_answer_through_the_scan() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    db.execute("PRAGMA INDEX_CACHE_MB = 0", ()).unwrap();
    let before = reads(&db);
    check_all(&db, "no cache budget");
    let after = reads(&db);
    assert!(
        delta(&after, &before, "refused") > 0,
        "the reservation was refused"
    );
    assert_eq!(
        delta(&after, &before, "candidates"),
        0,
        "no side file was walked"
    );
    db.execute("PRAGMA INDEX_CACHE_MB = 16", ()).unwrap();

    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let before = reads(&db);
    check_all(&db, "recreated index");
    let after = reads(&db);
    assert!(
        delta(&after, &before, "ineligible") > 0,
        "the old side file is not eligible for the recreated index"
    );
    assert_eq!(delta(&after, &before, "probes"), 0);
    // A volume sealed under the recreated index is eligible while the old
    // one still goes to the scan; the extra rows carry keys outside the
    // formula's range
    let values = (ROWS + 1..ROWS + 101)
        .map(|id| format!("({id},3000,{})", id * 3))
        .collect::<Vec<_>>()
        .join(",");
    db.execute(&format!("INSERT INTO t VALUES {values}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = reads(&db);
    check_all(&db, "one covered volume, one not");
    let after = reads(&db);
    assert!(delta(&after, &before, "ineligible") > 0, "the old volume");
    let before = reads(&db);
    assert_eq!(
        ids_db(&db, EQ, &[3000]),
        (ROWS + 1..ROWS + 101).collect::<Vec<_>>()
    );
    let after = reads(&db);
    assert!(delta(&after, &before, "probes") > 0, "the new volume");
    assert_eq!(
        delta(&after, &before, "ineligible"),
        0,
        "the old volume is pruned by its zone map"
    );
}

/// A wide result is served by the scan: the directory's bound admits the
/// index only for a small share of the volume's rows.
#[test]
fn a_wide_range_goes_to_the_scan_and_a_narrow_one_to_the_index() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = reads(&db);
    assert_eq!(
        ids_db(&db, RANGE, &[0, 2000]),
        expected_range(0, 2000),
        "eighty percent of the rows"
    );
    let after = reads(&db);
    assert!(
        delta(&after, &before, "cost_scans") > 0,
        "the cost decision chose the scan"
    );
    assert_eq!(delta(&after, &before, "candidates"), 0);
    let before = reads(&db);
    assert_eq!(ids_db(&db, RANGE, &[100, 103]), expected_range(100, 103));
    let after = reads(&db);
    assert_eq!(delta(&after, &before, "cost_scans"), 0);
    assert!(delta(&after, &before, "candidates") > 0);
}

/// LIMIT stops the fetch after the window that satisfies it.
#[test]
fn a_limit_stops_after_the_window_that_satisfies_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = reads(&db);
    let rows: Vec<i64> = db
        .query("SELECT id FROM t WHERE k >= 100 AND k <= 150 LIMIT 3", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    let want = expected_range(100, 150);
    assert!(rows.iter().all(|id| want.contains(id)));
    let after = reads(&db);
    assert!(
        delta(&after, &before, "rows") < want.len() as i64,
        "not every candidate was fetched: {} of {}",
        delta(&after, &before, "rows"),
        want.len()
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_inside_the_lookup_window_does_not_change_the_answer_and_the_output_is_indexed() {
    let _serial = serial();
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

    let before = reads(&db);
    check_all(&db, "compacted");
    let after = reads(&db);
    assert!(
        delta(&after, &before, "probes") > 0,
        "the merged volume carries a side file"
    );
    assert_eq!(delta(&after, &before, "ineligible"), 0);
}

/// A seal inside the lookup window: the rows sealed meanwhile come once,
/// from hot or from cold, never twice and never lost.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_seal_inside_the_lookup_window_answers_every_row_once() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS / 2);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert(&db, ROWS / 2 + 1, ROWS);

    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(other.engine().volume_stats().len(), 2, "the seal landed");
    });
    let mut tx = db.begin().unwrap();
    assert_eq!(
        ids_tx(&mut tx, RANGE, &[100, 103]),
        expected_range(100, 103)
    );
    tx.rollback().unwrap();
    stoolap::test_failpoints::after_cold_volumes_taken(|| {});
    check_all(&db, "after the seal");
}

#[test]
fn sixteen_volumes_answer_every_shape_through_their_side_files() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    for batch in 0..16 {
        insert(&db, batch * (ROWS / 16) + 1, (batch + 1) * (ROWS / 16));
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), 16);
    let before = reads(&db);
    check_all(&db, "sixteen volumes");
    let after = reads(&db);
    assert_eq!(delta(&after, &before, "ineligible"), 0);
    assert!(delta(&after, &before, "probes") >= 16);
}
