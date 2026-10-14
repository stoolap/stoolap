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

//! An INSERT checks its keys against the volumes outside the seal fence,
//! through the files its snapshot captured, and decides under it from
//! memory only

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use stoolap::core::{DataType, Error, Row, SchemaBuilder, Value};
use stoolap::storage::traits::Engine;
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::manifest::{ColdLoads, SegmentManager, SegmentMeta};
use stoolap::storage::volume::writer::VolumeBuilder;
use stoolap::test_failpoints::{
    after_insert_cold_snapshot, block_file_reads, block_store_drops_by_fence, live_volume_loads,
    volume_file_drops_by_fence, volume_file_reads_by_fence, volume_loads, FailpointGuard,
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
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db
}

/// Commits `rows` as `(id, 0, k, 0)` and seals them into one volume
fn seal_rows(db: &Database, rows: &[(i64, i64)]) {
    let values: Vec<String> = rows
        .iter()
        .map(|(id, k)| format!("({id}, 0, {k}, 0)"))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db.engine().seal_for_test().unwrap();
}

/// Seals `(id, 0, k, 0)` from another thread, so its counts stay off this one
fn seal_row_apart(db: &Database, id: i64, k: i64) {
    let db = db.clone();
    on_thread(move || seal_rows(&db, &[(id, k)])).expect("the seal finished");
}

/// `(id, 0, k, 0)` for the storage API
fn row(id: i64, k: i64) -> Row {
    Row::from_values(vec![
        Value::Integer(id),
        Value::Integer(0),
        Value::Integer(k),
        Value::Integer(0),
    ])
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

fn delta(after: (usize, usize), before: (usize, usize)) -> (usize, usize) {
    (after.0 - before.0, after.1 - before.1)
}

/// File reads on this thread, whole volumes and blocks, as (outside, under)
/// the seal fence
fn file_reads() -> (usize, usize) {
    let (volumes, blocks) = (volume_file_reads_by_fence(), block_file_reads());
    (volumes.0 + blocks.0, volumes.1 + blocks.1)
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

/// Compacts the table's volumes into one right after the next INSERT took
/// its snapshot, with the volumes cold
fn compact_after_the_snapshot(db: &Database) {
    db.engine().cold_volumes_for_test("t");
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let compactor = db.clone();
    after_insert_cold_snapshot(move || {
        let compacted = on_thread(move || compactor.engine().compact_for_test().is_ok());
        assert_eq!(compacted, Some(true), "the compaction did not finish");
    });
}

/// Seals a row in each of the next three INSERT rounds; after the third the
/// volumes are cold, so a check after it reads them from their files
fn seal_in_three_rounds(db: Database, rounds: Rc<Cell<usize>>) {
    after_insert_cold_snapshot(move || {
        let round = rounds.get() + 1;
        rounds.set(round);
        seal_row_apart(&db, 100 + round as i64, 100 + round as i64);
        if round < 3 {
            seal_in_three_rounds(db, rounds);
        } else {
            db.engine().cold_volumes_for_test("t");
        }
    });
}

#[test]
fn a_primary_key_in_a_volume_compacted_after_the_snapshot_is_found() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10)]);
    seal_rows(&db, &[(3, 30)]);
    compact_after_the_snapshot(&db);
    let inserted = db.execute("INSERT INTO t VALUES (1, 0, 99, 0)", ());
    assert_eq!(
        db.engine().volume_stats().len(),
        1,
        "the volumes were not merged"
    );
    assert!(
        matches!(inserted, Err(Error::PrimaryKeyConstraint { row_id: 1 })),
        "{inserted:?}"
    );
    assert_eq!(count(&db, "SELECT k FROM t WHERE id = 1"), 10);
}

#[test]
fn a_unique_key_in_a_volume_compacted_after_the_snapshot_is_found() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10)]);
    seal_rows(&db, &[(3, 30)]);
    compact_after_the_snapshot(&db);
    // Id 5 lies outside every volume's range: only the unique key is read
    let inserted = db.execute("INSERT INTO t VALUES (5, 0, 30, 0)", ());
    assert_eq!(
        db.engine().volume_stats().len(),
        1,
        "the volumes were not merged"
    );
    assert!(
        matches!(inserted, Err(Error::UniqueConstraint { row_id: 3, .. })),
        "{inserted:?}"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 30"), 1);
}

#[test]
fn a_batch_key_in_a_volume_compacted_after_the_snapshot_is_found() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10)]);
    seal_rows(&db, &[(3, 30)]);
    compact_after_the_snapshot(&db);
    let mut txn = db.engine().begin_transaction().unwrap();
    let inserted = txn
        .get_table("t")
        .unwrap()
        .insert_batch(vec![row(7, 70), row(1, 99)]);
    assert!(
        matches!(inserted, Err(Error::PrimaryKeyConstraint { row_id: 1 })),
        "{inserted:?}"
    );
    txn.rollback().unwrap();
    assert_eq!(count(&db, "SELECT k FROM t WHERE id = 1"), 10);
}

#[test]
fn a_check_loads_no_volume_through_the_live_map() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    db.engine().cold_volumes_for_test("t");
    let (live, loads) = (live_volume_loads(), volume_loads().0);
    db.execute("INSERT INTO t VALUES (2, 0, 20, 0)", ())
        .unwrap();
    assert!(volume_loads().0 > loads, "the check loaded no volume");
    assert_eq!(
        live_volume_loads() - live,
        0,
        "volumes loaded through the live map"
    );
}

#[test]
fn a_cold_batch_loads_each_volume_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (9, 90)]);
    db.engine().cold_volumes_for_test("t");
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let loads = volume_loads().0;
    table
        .insert_batch(vec![row(2, 20), row(3, 30), row(4, 40), row(5, 50)])
        .unwrap();
    drop(table);
    txn.commit().unwrap();
    assert_eq!(
        volume_loads().0 - loads,
        1,
        "volume loads for a four-row batch"
    );
}

#[test]
fn a_volume_retired_during_the_check_is_let_go_outside_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    seal_rows(&db, &[(5, 50)]);
    compact_after_the_snapshot(&db);
    let (files, stores) = (volume_file_drops_by_fence(), block_store_drops_by_fence());
    db.execute("INSERT INTO t VALUES (2, 0, 20, 0)", ())
        .unwrap();
    assert_eq!(
        db.engine().volume_stats().len(),
        1,
        "the volumes were not merged"
    );
    let (files, stores) = (
        delta(volume_file_drops_by_fence(), files),
        delta(block_store_drops_by_fence(), stores),
    );
    assert_eq!(files.1, 0, "a volume file let go under the seal fence");
    assert_eq!(stores.1, 0, "a block store let go under the seal fence");
    assert!(
        stores.0 > 0,
        "the INSERT let go of no loaded volume it held last"
    );
}

#[test]
fn an_insert_whose_volumes_keep_changing_reads_no_file_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    let _no_cache = NoGroupCache::set(&db);
    let rounds = Rc::new(Cell::new(0));
    seal_in_three_rounds(db.clone(), Rc::clone(&rounds));
    db.engine().cold_volumes_for_test("t");
    let reads = file_reads();
    let inserted = db.execute("INSERT INTO t VALUES (2, 0, 20, 0)", ());
    let (outside, under) = delta(file_reads(), reads);
    assert_eq!(rounds.get(), 3, "the INSERT did not check three times");
    assert_eq!(under, 0, "file reads under the seal fence");
    assert!(outside > 0, "the rounds read no volume");
    assert!(
        matches!(&inserted, Err(Error::Internal { message }) if message.contains("write conflict")),
        "{inserted:?}"
    );
}

#[test]
fn an_insert_whose_volumes_keep_changing_writes_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10), (3, 30)]);
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (50, 0, 50, 0)", ())
        .unwrap();
    let rounds = Rc::new(Cell::new(0));
    seal_in_three_rounds(db.clone(), Rc::clone(&rounds));
    let inserted = txn.execute("INSERT INTO t VALUES (2, 0, 20, 0)", ());
    assert!(
        matches!(&inserted, Err(Error::Internal { message }) if message.contains("write conflict")),
        "{inserted:?}"
    );
    txn.commit().unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 50"), 1);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE id = 2"), 0);
}

/// Seals a row in each of the next two INSERT rounds, then drops a column
/// and seals in the third
fn seal_then_change_a_column(db: Database, rounds: Rc<Cell<usize>>, done: Rc<Cell<Option<bool>>>) {
    after_insert_cold_snapshot(move || {
        let round = rounds.get() + 1;
        rounds.set(round);
        if round < 3 {
            seal_row_apart(&db, 100 + round as i64, 100 + round as i64);
            return seal_then_change_a_column(db, rounds, done);
        }
        let (ddl, sealer) = (db.clone(), db);
        let dropped = on_thread(move || ddl.execute("ALTER TABLE t DROP COLUMN x", ()).is_ok());
        let sealed = on_thread(move || {
            sealer
                .execute("INSERT INTO t VALUES (60, 60, 0)", ())
                .is_ok()
                && sealer.engine().seal_for_test().is_ok()
        });
        done.set(Some(dropped == Some(true) && sealed == Some(true)));
    });
}

#[test]
fn a_column_change_and_a_seal_in_the_last_round_of_a_conflicting_check_are_schema_changed() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    seal_rows(&db, &[(1, 10)]);
    seal_rows(&db, &[(3, 30)]);
    let (rounds, done) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(None)));
    seal_then_change_a_column(db.clone(), Rc::clone(&rounds), Rc::clone(&done));
    // k = 30 is row 3's: every round finds a real conflict
    let inserted = db.execute("INSERT INTO t VALUES (5, 0, 30, 0)", ());
    assert_eq!(rounds.get(), 3, "the INSERT did not check three times");
    assert_eq!(
        done.take(),
        Some(true),
        "the column change and the seal did not both finish inside the third round"
    );
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "{inserted:?}"
    );
}

/// A manager of table `name` whose segment 1 holds `ids`, evicted cold
fn cold_manager(dir: &Path, name: &str, ids: [i64; 2]) -> Arc<SegmentManager> {
    let schema = SchemaBuilder::new(name)
        .column("id", DataType::Integer, false, true)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for id in ids {
        builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
    }
    let path = write_volume_to_disk(dir, name, 1, &builder.finish().unwrap()).unwrap();
    let volume = Arc::new(read_volume_from_disk(&path).unwrap());
    volume.mark_accessed();
    let manager = Arc::new(SegmentManager::new(name, Some(dir.to_path_buf())));
    manager
        .register_segment(
            1,
            volume,
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("vol_0000000000000001.vol"),
                row_count: 2,
                min_row_id: ids[0],
                max_row_id: ids[1],
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(&schema),
        )
        .unwrap();
    manager.evict_idle_volumes(0, true);
    manager.evict_idle_volumes(3, true);
    assert!(manager.segments_raw()[&1].volume.is_cold());
    manager
}

#[test]
fn a_load_list_shared_by_two_tables_keeps_their_volumes_apart() {
    let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a = cold_manager(dir_a.path(), "a", [1, 3]);
    let b = cold_manager(dir_b.path(), "b", [2, 4]);
    let loads = ColdLoads::default();
    let found = |mgr: &SegmentManager, id: i64| {
        mgr.check_value_exists_with_snapshot(&mgr.cold_snapshot(), &loads, 0, &Value::Integer(id))
            .unwrap()
    };
    assert_eq!(found(&a, 1), Some(1));
    assert_eq!(found(&b, 3), None, "table b found table a's key");
    assert_eq!(found(&b, 2), Some(2));
}
