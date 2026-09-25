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

use rustc_hash::{FxHashMap, FxHashSet};
use stoolap::storage::volume::manifest::SegmentManager;
use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.with(|l| l.set(l.get() + layout.size() as isize));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.with(|l| l.set(l.get() - layout.size() as isize));
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.with(|l| l.set(l.get() + new_size as isize - layout.size() as isize));
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
    let snapshot = mgr.tombstone_set_arc();
    mgr.add_tombstones(&[2], 7);
    let compacted: FxHashSet<i64> = [1, 2, 3].into_iter().collect();
    mgr.remove_tombstones_matching_snapshot(&snapshot, &compacted);
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
