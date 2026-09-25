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

#[cfg(feature = "test-failpoints")]
#[test]
fn a_column_drop_cannot_pair_a_new_schema_with_an_old_cold_mapping() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1,10,20)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let (paused, paused_rx) = std::sync::mpsc::channel();
    let (captured, captured_rx) = std::sync::mpsc::channel();
    let (done, done_rx) = std::sync::mpsc::channel();
    let other = db.clone();
    let writer = std::thread::spawn(move || {
        stoolap::test_failpoints::before_wal_sync(move || {
            paused.send(()).unwrap();
            captured_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        });
        other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        done.send(()).unwrap();
    });
    paused_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let at_capture = captured.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        at_capture.send(()).unwrap();
        done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
    });
    let result = db.query("SELECT b FROM t ORDER BY b", ());
    let _ = captured.send(());
    writer.join().unwrap();
    stoolap::test_failpoints::after_cold_volumes_taken(|| {});
    let error = match result {
        Ok(mut rows) => {
            while rows.advance() {}
            rows.error().expect("a mismatched schema must be rejected")
        }
        Err(error) => error,
    };
    assert!(error.to_string().contains("schema changed"), "{error}");
    let mut rows = db.query("SELECT b FROM t ORDER BY b", ()).unwrap();
    assert!(rows.advance());
    assert_eq!(rows.current_row()[0], Value::Integer(20));
    assert!(!rows.advance());
    assert!(rows.error().is_none());
    db.close().unwrap();
}

#[cfg(feature = "test-failpoints")]
#[test]
fn maintenance_retries_when_ddl_changes_its_captured_schema() {
    let _serial = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
            dir.path().display()
        ))
        .unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
            (),
        )
        .unwrap();
        for id in 1..=3 {
            db.execute("INSERT INTO t VALUES (?,10,20)", (id,)).unwrap();
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
        }
        let mut before: Vec<_> = db.engine().volume_stats().iter().map(|v| v.1).collect();
        before.sort_unstable();
        assert_eq!(before.len(), if sealed { 3 } else { 0 });
        db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        let other = db.clone();
        let fired = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&fired);
        stoolap::test_failpoints::after_maintenance_schema_taken(move || {
            other
                .execute(
                    if sealed {
                        "ALTER TABLE t DROP COLUMN a"
                    } else {
                        "ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 4"
                    },
                    (),
                )
                .unwrap();
            observed.fetch_add(1, Ordering::Relaxed);
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(fired.load(Ordering::Relaxed), 1);
        let mut after: Vec<_> = db.engine().volume_stats().iter().map(|v| v.1).collect();
        after.sort_unstable();
        assert_eq!(
            after, before,
            "maintenance used a stale schema, sealed={sealed}"
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let value: i64 = db.query_one("SELECT SUM(b) FROM t", ()).unwrap();
        assert_eq!(value, 60, "sealed={sealed}");
        if !sealed {
            let value: i64 = db.query_one("SELECT SUM(c) FROM t", ()).unwrap();
            assert_eq!(value, 12);
        }
        db.close().unwrap();
    }
}

#[test]
fn readers_share_a_retired_volume_without_keeping_its_payload_alive() {
    use stoolap::core::{DataType, Row, SchemaBuilder};
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use stoolap::storage::volume::{io, writer::VolumeBuilder};

    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
    let volume = builder.finish().unwrap();
    let id = io::next_volume_id();
    let path = io::write_volume_to_disk(dir.path(), "t", id, &volume).unwrap();
    let cold = Arc::new(volume.to_cold());
    drop(volume);
    let mgr = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    mgr.register_segment(
        id,
        cold,
        SegmentMeta {
            segment_id: id,
            file_path: path.clone(),
            row_count: 1,
            min_row_id: 1,
            max_row_id: 1,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    let view = mgr.cold_snapshot();
    let owner = view.segs[&id].file.as_ref().unwrap();
    mgr.replace_segments_atomic_remove_only(&[id]);
    owner.retire();
    let barrier = std::sync::Barrier::new(8);
    let loaded = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    mgr.ensure_pinned_volume(id, owner).unwrap().unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(loaded.iter().all(|v| Arc::ptr_eq(v, &loaded[0])));
    let before = stoolap::storage::volume::group_cache::DECODED_GROUPS
        .stats()
        .misses;
    for volume in &loaded {
        assert_eq!(volume.get_row(0).unwrap()[0], Value::Integer(1));
    }
    let after = stoolap::storage::volume::group_cache::DECODED_GROUPS
        .stats()
        .misses;
    assert_eq!(after - before, 1);
    let weak = Arc::downgrade(&loaded[0]);
    drop(loaded);
    assert!(
        weak.upgrade().is_none(),
        "file ownership must not retain the loaded volume"
    );
    assert!(path.exists());
    let again = mgr.ensure_pinned_volume(id, owner).unwrap().unwrap();
    assert_eq!(again.get_row(0).unwrap()[0], Value::Integer(1));
    drop(again);
    drop(view);
    assert!(!path.exists());
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_captured_mapping_finishes_after_a_column_rename() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,20)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other
            .execute("ALTER TABLE t RENAME COLUMN b TO c", ())
            .unwrap();
    });
    let value: i64 = db.query_one("SELECT b FROM t ORDER BY b", ()).unwrap();
    assert_eq!(value, 20);
    let value: i64 = db.query_one("SELECT c FROM t ORDER BY c", ()).unwrap();
    assert_eq!(value, 20);
}

#[test]
fn a_cold_reload_can_reuse_a_readers_unique_index() {
    use stoolap::core::{DataType, Row, SchemaBuilder};
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use stoolap::storage::volume::{io, writer::VolumeBuilder};

    let _serial = serial();
    for bulk in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let mut volume = builder.finish().unwrap();
        let id = io::next_volume_id();
        let path = io::write_volume_to_disk(dir.path(), "t", id, &volume).unwrap();
        let (_, store) = io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        volume.prebuild_unique_index(&[0]).unwrap();
        let mgr = Arc::new(SegmentManager::new("t", Some(dir.path().to_path_buf())));
        mgr.register_segment(
            id,
            Arc::new(volume),
            SegmentMeta {
                segment_id: id,
                file_path: path,
                row_count: 1,
                min_row_id: 1,
                max_row_id: 1,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(&schema),
        );
        mgr.evict_idle_volumes(0);
        mgr.evict_idle_volumes(3);
        let held = Arc::clone(&mgr.segments_raw()[&id].volume);
        assert!(held.is_warm());
        mgr.evict_idle_volumes(6);
        mgr.evict_idle_volumes(9);
        assert!(mgr.segments_raw()[&id].volume.is_cold());
        let (done, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let loaded = if bulk {
                Arc::clone(&mgr.segments_snapshot().unwrap()[&id].volume)
            } else {
                mgr.ensure_volume(id).unwrap().unwrap()
            };
            done.send(loaded).unwrap();
        });
        let loaded = done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("reload must not lock the same unique index twice");
        worker.join().unwrap();
        assert!(Arc::ptr_eq(&held, &loaded));
        assert_eq!(loaded.get_row(0).unwrap()[0], Value::Integer(1));
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn dropping_a_pinned_table_persists_an_empty_manifest() {
    use stoolap::storage::volume::manifest::TableManifest;

    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,10)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().cold_volumes_for_test("t"), (1, 1));
    let old_names = vol_files(dir.path());
    let root = dir.path().to_path_buf();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("DROP TABLE t", ()).unwrap();
        other
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(old_names.is_subset(&vol_files(&root)));
        let manifest = TableManifest::read_from_disk(&root.join("volumes/t/manifest.bin"))
            .unwrap()
            .manifest;
        assert!(
            manifest.segments.is_empty(),
            "recovery must not discover pinned, dropped rows"
        );
    });
    let value: i64 = db.query_one("SELECT k FROM t ORDER BY k", ()).unwrap();
    assert_eq!(value, 10);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn a_stale_table_handle_cannot_scan_a_new_column_mapping() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,20)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let old = tx.get_table("t").unwrap();
    db.clone()
        .execute("ALTER TABLE T ADD COLUMN c INTEGER DEFAULT 4", ())
        .unwrap();
    let error = old.collect_all_rows(None).unwrap_err();
    assert!(matches!(error, stoolap::Error::SchemaChanged { .. }));
    tx.rollback().unwrap();
    let value: i64 = db.query_one("SELECT c FROM t ORDER BY c", ()).unwrap();
    assert_eq!(value, 4);
    assert!(db
        .execute("ALTER TABLE t ADD COLUMN c INTEGER", ())
        .is_err());
    let value: i64 = db.query_one("SELECT c FROM t ORDER BY c", ()).unwrap();
    assert_eq!(
        value, 4,
        "a validation error must leave the schema readable"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn destructive_ddl_keeps_captured_files_without_restoring_them_on_reopen() {
    let _serial = serial();
    for drop_table in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!(
            "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        );
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1,10)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(db.engine().cold_volumes_for_test("t"), (1, 1));
        let old_names = vol_files(dir.path());
        let held_names = old_names.clone();
        let root = dir.path().to_path_buf();
        let other = db.clone();
        stoolap::test_failpoints::after_cold_volumes_taken(move || {
            if drop_table {
                other.execute("DROP TABLE t", ()).unwrap();
                other
                    .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
                    .unwrap();
            } else {
                other.execute("TRUNCATE TABLE t", ()).unwrap();
            }
            other.execute("INSERT INTO t VALUES (2,20)", ()).unwrap();
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
            let present = vol_files(&root);
            assert!(
                held_names.is_subset(&present),
                "captured files must survive destructive DDL"
            );
        });
        let value: i64 = db.query_one("SELECT k FROM t ORDER BY k", ()).unwrap();
        assert_eq!(value, 10);
        let after = vol_files(dir.path());
        assert!(old_names.is_disjoint(&after));
        let value: i64 = db.query_one("SELECT k FROM t ORDER BY k", ()).unwrap();
        assert_eq!(value, 20);
        db.close().unwrap();
        let db = Database::open(&dsn).unwrap();
        let value: i64 = db.query_one("SELECT k FROM t ORDER BY k", ()).unwrap();
        assert_eq!(value, 20);
        let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(count, 1);
    }
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
