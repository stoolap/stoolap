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

//! A reader that takes its cold volume list must read that same view to the
//! end: a compaction running inside its window may neither bring a deleted
//! row back nor take a live one away.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use stoolap::core::{RowVec, Value};
use stoolap::storage::traits::Engine;
use stoolap::Database;

const ROWS: i64 = 2_000;
const DELETED_UP_TO: i64 = 10;

/// The rows a scan returned, as `(row id, key column)`, ordered by row id.
fn ids_and_keys(rows: &RowVec) -> Vec<(i64, i64)> {
    let mut seen: Vec<(i64, i64)> = rows
        .iter()
        .map(|(id, row)| {
            let key = match row.get(1) {
                Some(Value::Integer(k)) => *k,
                other => panic!("row {id} has no integer key: {other:?}"),
            };
            (*id, key)
        })
        .collect();
    seen.sort_unstable();
    seen
}

/// What the fixture holds once its low ids are deleted: every id from 11 up,
/// once each, with the key the insert gave it.
fn expected() -> Vec<(i64, i64)> {
    (DELETED_UP_TO + 1..=ROWS).map(|id| (id, id % 7)).collect()
}

/// The decode cache is process-wide, so the tests here that read it, and the
/// tests whose reads fill it, run one at a time whatever the runner does.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The volume files the table has on disk.
fn vol_files(dir: &std::path::Path) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir.join("volumes").join("t")) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".vol") {
                    names.insert(name.to_string());
                }
            }
        }
    }
    names
}

fn fixture(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();

    // Two small volumes, so the next checkpoint has something to merge
    for batch in 0..2 {
        let mut values = String::new();
        for i in 0..ROWS / 2 {
            let id = batch * (ROWS / 2) + i + 1;
            values.push_str(&format!("({id},{},1.0),", id % 7));
        }
        values.pop();
        db.execute(&format!("INSERT INTO t (id, k, v) VALUES {values}"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    // The tombstones stay uncompacted, so the compaction lands inside the
    // reader's window rather than before it
    db.execute(&format!("DELETE FROM t WHERE id <= {DELETED_UP_TO}"), ())
        .unwrap();
    db
}

/// A compaction that consumes the tombstones of the captured rows must not
/// let those rows back into a reader that took its view before it. The rows
/// are resident when it runs, so the volume itself is still readable: what
/// the reader loses is the record that they were deleted.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_during_a_cold_read_does_not_restore_a_deleted_row() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = fixture(dir.path());

    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    assert_eq!(
        ids_and_keys(&rows),
        expected(),
        "the reader's view of what was deleted must survive the compaction"
    );
}

/// A volume whose blocks are in RAM has no file-backed store to carry its
/// file's lifetime, so the cleanup that retires it takes the branch that
/// removes the file outright rather than the one a holder defers. A reader
/// that pinned the file keeps it anyway, which is what the routing is for.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_retirement_of_a_memory_backed_volume_keeps_the_file_a_reader_holds() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=3",
        dir.path().display()
    );
    let batch = |from: i64, count: i64| {
        let mut values = String::new();
        for id in from..from + count {
            values.push_str(&format!("({id},{},1.0),", id % 7));
        }
        values.pop();
        format!("INSERT INTO t (id, k, v) VALUES {values}")
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    // Resident volumes: nothing is evicted, so their blocks are in RAM when
    // the compaction replaces them
    for b in 0..3 {
        db.execute(&batch(b * 1_000 + 1, 1_000), ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }

    let before = vol_files(dir.path());
    assert_eq!(before.len(), 3, "the fixture's volumes are on disk");

    let held = Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
    let kept = Arc::clone(&held);
    let table_dir = dir.path().to_path_buf();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute(&batch(3_001, 1_000), ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        *kept.lock().unwrap() = vol_files(&table_dir);
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    assert_eq!(
        rows.len(),
        3_000,
        "the reader answers its own three volumes"
    );
    let during = held.lock().unwrap().clone();
    for name in &before {
        assert!(
            during.contains(name),
            "{name} survives the cleanup that removes files outright"
        );
    }
    assert!(
        during.len() > before.len(),
        "the compaction merged the volumes the reader holds"
    );
}

/// A rename between the capture and the reload moves the directory the
/// volume's file lives in. The reader reads through the handle it pinned,
/// which moves with the directory.
///
/// This is a control, not a guard: it passes with the pin bypassed too,
/// because the path a reader would resolve on its own is built from the
/// manager's current table name and the rename updates that. What the pin
/// adds is the window between resolving a path and opening it, which no test
/// here can place a rename inside.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_during_a_cold_read_does_not_break_the_reload() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,10.0),(2,1,20.0),(3,2,30.0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((volumes, cold), (1, 1), "the volume is metadata-only");

    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("ALTER TABLE t RENAME TO t2", ()).unwrap();
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    assert_eq!(
        ids_and_keys(&rows),
        vec![(1, 1), (2, 1), (3, 2)],
        "the reader reloads the volume the rename moved"
    );
}

/// A reader that takes its view while a rename has moved the directory but
/// not yet the name reads the volume it took. The view carries the handles
/// its segments were registered with, so no path is resolved from a name
/// that is mid-change.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_in_flight_does_not_break_the_view_a_reader_takes() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,10.0),(2,1,20.0),(3,2,30.0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((volumes, cold), (1, 1), "the volume is metadata-only");

    let (moved, moved_rx) = std::sync::mpsc::channel();
    let (taken, taken_rx) = std::sync::mpsc::channel();
    let (done, done_rx) = std::sync::mpsc::channel();

    let rename = {
        let other = db.clone();
        std::thread::spawn(move || {
            stoolap::test_failpoints::in_rename_after_the_move(move || {
                moved.send(()).unwrap();
                taken_rx.recv().unwrap();
            });
            other.execute("ALTER TABLE t RENAME TO t2", ()).unwrap();
            done.send(()).unwrap();
        })
    };

    let reader = {
        let db = db.clone();
        std::thread::spawn(move || {
            moved_rx.recv().unwrap();
            stoolap::test_failpoints::after_cold_volumes_taken(move || {
                taken.send(()).unwrap();
            });
            // Mid-rename: the engine knows the new name, the manager's
            // directory has moved and its name has not
            let mut tx = db.engine().begin_transaction().unwrap();
            let table = tx.get_table("t2").unwrap();
            let rows = table.collect_all_rows(None).unwrap();
            tx.rollback().unwrap();
            ids_and_keys(&rows)
        })
    };

    let seen = reader.join().unwrap();
    rename.join().unwrap();
    done_rx.recv().unwrap();
    assert_eq!(
        seen,
        vec![(1, 1), (2, 1), (3, 2)],
        "the reader reads the volume its view carries"
    );
}

/// A metadata-only volume read twice. The first read installs what it loaded,
/// so the second shares that volume rather than reading the file again, and
/// the groups the first decoded come back from the cache instead of the disk.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_second_read_of_a_metadata_only_volume_reuses_the_first() {
    let _serial = serial();
    use stoolap::storage::volume::group_cache::DECODED_GROUPS;

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,1.0),(2,2,2.0),(3,3,3.0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((volumes, cold), (1, 1), "the volume is metadata-only");

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let first = table.collect_all_rows(None).unwrap().len();
    let after_first = DECODED_GROUPS.stats().misses;
    let second = table.collect_all_rows(None).unwrap().len();
    let after_second = DECODED_GROUPS.stats().misses;
    tx.rollback().unwrap();

    assert_eq!(first, 3, "the first read answers every row");
    assert_eq!(second, 3, "and so does the second");
    assert!(after_first > 0, "the first read decodes the volume");
    assert_eq!(
        after_second, after_first,
        "the second read decodes nothing the first already had"
    );
}

/// A compaction that retires a captured volume must not take its live rows
/// with it. The volumes are metadata-only when the reader takes them, so they
/// have to be reloaded from disk to be read at all, and that reload must not
/// depend on the segment still being in the manifest.
///
/// The threshold is set so three volumes do not merge: the fourth, sealed
/// inside the reader's window, is what takes the count over it.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_during_a_cold_read_does_not_lose_a_live_row() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=3",
        dir.path().display()
    );
    let batch = |from: i64, count: i64| {
        let mut values = String::new();
        for id in from..from + count {
            values.push_str(&format!("({id},{},1.0),", id % 7));
        }
        values.pop();
        format!("INSERT INTO t (id, k, v) VALUES {values}")
    };
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
        for b in 0..3 {
            db.execute(&batch(b * 1_000 + 1, 1_000), ()).unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }

    // A reopened volume is resident, and the read only reloads a metadata-only
    // one, so the test drives the volumes down the tiers itself. The local
    // epochs leave the global eviction epoch alone.
    let db = Database::open(&dsn).unwrap();
    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!(volumes, 3, "the fixture sealed three volumes");
    assert_eq!(cold, 3, "all three are metadata-only before the read");

    let before = vol_files(dir.path());
    assert_eq!(before.len(), 3, "the fixture's volumes are on disk");

    let retired = Arc::new(AtomicUsize::new(usize::MAX));
    let held = Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
    let seen = Arc::clone(&retired);
    let kept = Arc::clone(&held);
    let table_dir = dir.path().to_path_buf();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute(&batch(3_001, 1_000), ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        seen.store(other.engine().volume_stats().len(), Ordering::Relaxed);
        // The reader holds this view, so every file it names must still be
        // there: the merged volume is added on top of them, not in place
        *kept.lock().unwrap() = vol_files(&table_dir);
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    // The fourth volume takes the count over the threshold, so the compaction
    // runs and the three captured segments leave the manifest
    assert!(
        retired.load(Ordering::Relaxed) < 3,
        "the compaction must retire the captured volumes for this test to guard anything"
    );
    let during = held.lock().unwrap().clone();
    for name in &before {
        assert!(
            during.contains(name),
            "{name} is the reader's to read until it lets go"
        );
    }
    // The reader is the last holder of the files it pinned, so the retired
    // ones go now that it has let go
    let after = vol_files(dir.path());
    for name in &before {
        assert!(
            !after.contains(name),
            "{name} is cleaned up once its last holder lets go"
        );
    }

    // The row the hook inserts commits after this statement's view, so the
    // reader's answer is the three captured volumes and nothing else
    let expected: Vec<(i64, i64)> = (1..=3_000).map(|id| (id, id % 7)).collect();
    assert_eq!(
        ids_and_keys(&rows),
        expected,
        "a captured volume keeps the rows it held"
    );
}
