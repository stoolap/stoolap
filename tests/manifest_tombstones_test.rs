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

//! A table's committed tombstones live once, in the tombstone map, and a
//! persist writes the map as it stands with the manifest

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rustc_hash::FxHashMap;
use stoolap::storage::volume::manifest::SegmentManager;
use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.with(|l| l.set(l.get() + layout.size() as isize));
        ALLOCATED.with(|a| a.set(a.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.with(|l| l.set(l.get() - layout.size() as isize));
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.with(|l| l.set(l.get() + new_size as isize - layout.size() as isize));
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const TOMBSTONES: i64 = 100_000;

fn live() -> isize {
    LIVE.with(Cell::get)
}

/// What the tombstones cost held in a map alone, built the way the
/// manager builds its map
fn map_alone(ids: &[i64]) -> isize {
    let before = live();
    let mut map: FxHashMap<i64, u64> = FxHashMap::default();
    for &id in ids {
        map.insert(id, 1);
    }
    let held = live() - before;
    drop(map);
    held
}

fn assert_one_copy(held: isize, map: isize, when: &str) {
    // The manager's own fields are small; a second list of 100,000
    // (row id, seq) pairs is 1.6 MB
    assert!(
        held < map + 256 * 1024,
        "{when}: the manager holds {held} bytes, the map alone {map}"
    );
}

#[test]
fn committed_tombstones_are_held_once() {
    let ids: Vec<i64> = (0..TOMBSTONES).map(|i| i * 3 - 7).collect();
    let restamp: Vec<i64> = ids.iter().copied().step_by(2).collect();
    let map = map_alone(&ids);

    let before = live();
    let mgr = SegmentManager::new("t", None);
    mgr.add_tombstones(&ids, 1);
    assert_one_copy(live() - before, map, "after the commits");
    mgr.add_tombstones(&restamp, 2);
    assert_one_copy(live() - before, map, "after a new stamp on half of them");
    assert_eq!(mgr.tombstone_count(), TOMBSTONES as usize);
}

#[test]
fn tombstones_read_back_at_open_are_held_once() {
    let dir = tempfile::tempdir().unwrap();
    let ids: Vec<i64> = (0..TOMBSTONES).map(|i| i * 3 - 7).collect();
    {
        let mgr = SegmentManager::new("t", Some(dir.path().to_path_buf()));
        mgr.add_tombstones(&ids, 1);
        mgr.persist_manifest_only().unwrap();
    }
    let map = map_alone(&ids);

    let before = live();
    let mgr = SegmentManager::load_from_disk("t", dir.path())
        .unwrap()
        .unwrap();
    assert_one_copy(live() - before, map, "after the open");
    assert_eq!(mgr.tombstone_count(), TOMBSTONES as usize);
}

fn reopened(dir: &std::path::Path) -> Vec<(i64, u64)> {
    let mgr = SegmentManager::load_from_disk("t", dir).unwrap().unwrap();
    let mut read: Vec<(i64, u64)> = mgr
        .tombstone_set_arc()
        .iter()
        .map(|(&id, &seq)| (id, seq))
        .collect();
    read.sort_unstable();
    read
}

#[test]
fn a_new_stamp_is_written_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    mgr.add_tombstones(&[1, 2, 3], 1);
    mgr.add_tombstones(&[2], 5);
    mgr.persist_manifest_only().unwrap();
    assert_eq!(reopened(dir.path()), vec![(1, 1), (2, 5), (3, 1)]);
}

#[test]
fn cleanup_by_an_old_snapshot_keeps_a_newer_stamp_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    mgr.add_tombstones(&[1, 2, 3], 1);
    let applied: Vec<(i64, u64)> = [1, 2, 3]
        .iter()
        .map(|rid| (*rid, mgr.tombstone_set_arc()[rid]))
        .collect();
    mgr.add_tombstones(&[2], 7);
    mgr.remove_applied_tombstones(&applied);
    mgr.persist_manifest_only().unwrap();
    assert_eq!(reopened(dir.path()), vec![(2, 7)]);
}

#[test]
fn cleared_tombstones_stay_cleared_after_a_persist() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    mgr.add_tombstones(&[1, 2, 3], 1);
    mgr.persist_manifest_only().unwrap();
    mgr.clear();
    mgr.persist_manifest_only().unwrap();
    assert_eq!(reopened(dir.path()), vec![]);
}

#[test]
fn truncate_after_cold_deletes_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=0",
        dir.path().join("db").display()
    );
    let count = |db: &Database| -> i64 {
        db.query("SELECT COUNT(*) FROM t", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap()
    };
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for id in 1..=2_000 {
            db.execute("INSERT INTO t VALUES ($1, $2)", (id, id))
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("DELETE FROM t WHERE id % 2 = 0", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(count(&db), 1_000);
        db.execute("TRUNCATE TABLE t", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(count(&db), 1);
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(count(&db), 1, "only the row inserted after TRUNCATE");
    let v: i64 = db
        .query("SELECT v FROM t WHERE id = 2", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(v, 20);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn an_older_persist_never_lands_after_a_newer_one() {
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let mgr = Arc::new(SegmentManager::new("t", Some(dir.path().to_path_buf())));
    mgr.add_tombstones(&[1], 1);

    let (reached_tx, reached_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let older = {
        let mgr = Arc::clone(&mgr);
        std::thread::spawn(move || {
            stoolap::test_failpoints::after_manifest_captured(move || {
                reached_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            mgr.persist_manifest_only().unwrap();
        })
    };
    reached_rx.recv().unwrap();

    mgr.add_tombstones(&[1], 2);
    let (done_tx, done_rx) = mpsc::channel();
    let newer = {
        let mgr = Arc::clone(&mgr);
        std::thread::spawn(move || {
            mgr.persist_manifest_only().unwrap();
            done_tx.send(()).unwrap();
        })
    };
    // The newer persist either finishes here or waits for the older one
    let _ = done_rx.recv_timeout(Duration::from_millis(300));
    release_tx.send(()).unwrap();
    older.join().unwrap();
    newer.join().unwrap();

    assert_eq!(
        reopened(dir.path()),
        vec![(1, 2)],
        "the older capture was written last"
    );
}

fn open_file(dir: &std::path::Path, extra: &str) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0{extra}",
        dir.display()
    ))
    .unwrap()
}

/// Inserts `rows` rows with ids from `from` and seals them into a volume
fn seal_rows(db: &Database, from: i64, rows: i64) {
    for start in (from..from + rows).step_by(10_000) {
        let end = (start + 10_000).min(from + rows);
        let values: Vec<String> = (start..end).map(|id| format!("({id},{id})")).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

fn count(db: &Database) -> i64 {
    db.query_one("SELECT COUNT(*) FROM t", ()).unwrap()
}

/// The committed tombstones the table's manifest holds on disk
fn tombstones_on_disk(dir: &std::path::Path) -> Vec<(i64, u64)> {
    let path = dir.join("volumes").join("t").join("manifest.bin");
    let mut on_disk = stoolap::storage::volume::manifest::TableManifest::read_from_disk(&path)
        .unwrap()
        .tombstones;
    on_disk.sort_unstable();
    on_disk
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_during_a_compaction_does_not_copy_the_map() {
    use std::sync::mpsc;

    fn delete_one(db: &Database, id: i64) -> usize {
        let db = db.clone();
        std::thread::spawn(move || {
            let before = ALLOCATED.with(Cell::get);
            db.execute("DELETE FROM t WHERE id = $1", (id,)).unwrap();
            ALLOCATED.with(Cell::get) - before
        })
        .join()
        .unwrap()
    }

    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path(), "");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    seal_rows(&db, 1, 100_000);
    seal_rows(&db, 100_001, 100_000);
    // 100,000 tombstones: a copy of the map is over 2 MB
    db.execute("DELETE FROM t WHERE id % 2 = 0", ()).unwrap();
    let alone = delete_one(&db, 1);
    assert!(alone < 1 << 20, "a lone commit allocated {alone} bytes");

    let (sent, received) = mpsc::channel();
    let other = db.clone();
    stoolap::test_failpoints::after_compaction_dedup(move |_| {
        sent.send(delete_one(&other, 3)).unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let during = received
        .try_recv()
        .expect("the compaction reached its dedup");
    assert!(
        during < alone + (1 << 20),
        "a commit during the compaction allocated {during} bytes, {alone} alone"
    );
    assert_eq!(count(&db), 200_000 - 100_000 - 2);
}

#[test]
fn a_compaction_under_a_snapshot_keeps_later_tombstones_and_unselected_volumes() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open_file(dir.path(), "&target_volume_rows=100000");
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        seal_rows(&db, 1, 50_000);
        seal_rows(&db, 50_001, 50_000);
        seal_rows(&db, 100_001, 100_000);
        db.execute("DELETE FROM t WHERE id = 10", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 60010", ()).unwrap();

        let snapshot = db.clone();
        snapshot
            .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
            .unwrap();
        assert_eq!(count(&snapshot), 199_998);
        // After the snapshot began: one in a volume the compaction merges,
        // one in the at-target volume it leaves alone
        db.execute("DELETE FROM t WHERE id = 20", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 150000", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        let kept: Vec<i64> = tombstones_on_disk(dir.path())
            .into_iter()
            .map(|(rid, _)| rid)
            .collect();
        assert_eq!(kept, vec![20, 150_000], "the applied tombstones went");
        assert_eq!(
            count(&snapshot),
            199_998,
            "the snapshot still sees its rows"
        );
        assert_eq!(count(&db), 199_996);
        snapshot.execute("COMMIT", ()).unwrap();
        db.close().unwrap();
    }
    let db = open_file(dir.path(), "&target_volume_rows=100000");
    assert_eq!(count(&db), 199_996);
    let gone: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE id IN (10, 20, 60010, 150000)",
            (),
        )
        .unwrap();
    assert_eq!(gone, 0);
}

#[test]
fn a_compaction_that_empties_its_volumes_clears_their_tombstones() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open_file(dir.path(), "");
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        seal_rows(&db, 1, 1_000);
        seal_rows(&db, 1_001, 1_000);
        db.execute("DELETE FROM t", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(tombstones_on_disk(dir.path()), vec![]);
        assert_eq!(count(&db), 0);
        db.close().unwrap();
    }
    let db = open_file(dir.path(), "");
    assert_eq!(count(&db), 0);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_takes_one_pair_per_deleted_row_over_overlapping_volumes() {
    use std::sync::mpsc;
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path(), "&compact_threshold=100");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    seal_rows(&db, 1, 1_000);
    for _ in 0..4 {
        db.execute("UPDATE t SET v = v + 1", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(
        db.engine().volume_stats().len(),
        5,
        "every round sealed another copy of the same ids"
    );
    db.execute("DELETE FROM t", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let (sent, received) = mpsc::channel();
    stoolap::test_failpoints::after_compaction_dedup(move |pairs| sent.send(pairs).unwrap());
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        received
            .try_recv()
            .expect("the compaction reached its dedup"),
        1_000,
        "one pair per deleted row, not per copy"
    );
    assert_eq!(count(&db), 0);
    assert_eq!(tombstones_on_disk(dir.path()), vec![]);
}
