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

//! A commit's cold recheck reads the volumes outside the seal fence and
//! decides under it from memory only

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use stoolap::core::Error;
use stoolap::test_failpoints::{
    after_commit_round_moved, after_seal_generation_moved, block_file_reads,
    block_store_drops_by_fence, cold_rechecks, live_volume_loads, on_cold_recheck, on_commit_fence,
    table_wal_records, volume_file_drops_by_fence, volume_file_reads_by_fence, FailpointGuard,
};
use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(10);

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
}

fn seal(db: &Database) {
    db.engine().seal_for_test().unwrap();
}

/// Commits `(id, 0, k, 0)` through `db` and seals it
fn seal_row(db: &Database, id: i64, k: i64) {
    db.execute(&format!("INSERT INTO t VALUES ({id}, 0, {k}, 0)"), ())
        .unwrap();
    seal(db);
}

/// Seals `(id, 0, k, 0)` from another thread, so its statement's counts
/// stay off this one
fn seal_row_apart(db: &Database, id: i64, k: i64) {
    let db = db.clone();
    on_thread(move || seal_row(&db, id, k)).expect("the seal finished");
}

/// `(5, 0, 50, 0)` sealed; `(1, 0, 10, 0)` committed; a transaction inserts
/// `(2, 0, 7, 0)`; then `(3, 0, k3, 0)` is committed and sealed with row 1,
/// so the commit rechecks, and id 2 falls inside that volume's PK range
fn pending(dir: &Path, k3: i64) -> (Database, stoolap::ApiTransaction) {
    let db = Database::open(&dsn(dir)).unwrap();
    create(&db);
    seal_row(&db, 5, 50);
    db.execute("INSERT INTO t VALUES (1, 0, 10, 0)", ())
        .unwrap();
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 0, 7, 0)", ())
        .unwrap();
    seal_row(&db, 3, k3);
    (db, txn)
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

/// Runs `work` on its own thread and waits for it, so what it lets go is
/// counted on that thread; None when it did not finish in time
fn on_thread<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(work());
    });
    finished.recv_timeout(WATCHDOG).ok()
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_recheck_reads_no_volume_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (db, mut txn) = pending(dir.path(), 30);
    db.engine().cold_volumes_for_test("t");
    let _no_cache = NoGroupCache::set(&db);
    let (rechecks, reads) = (cold_rechecks(), file_reads());
    txn.commit().unwrap();
    let (outside, under) = delta(file_reads(), reads);
    assert!(cold_rechecks() > rechecks, "the commit did not recheck");
    assert_eq!(under, 0, "file reads under the seal fence");
    assert!(outside > 0, "the recheck read no volume");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 7"), 1);
}

#[test]
fn a_conflict_found_by_the_recheck_reads_no_volume_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (db, mut txn) = pending(dir.path(), 7);
    db.engine().cold_volumes_for_test("t");
    let _no_cache = NoGroupCache::set(&db);
    let reads = file_reads();
    let committed = txn.commit();
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { row_id: 3, .. })),
        "{committed:?}"
    );
    assert_eq!(
        delta(file_reads(), reads).1,
        0,
        "file reads under the seal fence"
    );
}

#[test]
fn a_seal_between_the_recheck_and_the_fence_brings_its_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (db, mut txn) = pending(dir.path(), 30);
    let other = db.clone();
    let mut sealed = false;
    on_commit_fence(move || {
        if !std::mem::replace(&mut sealed, true) {
            seal_row_apart(&other, 4, 7);
        }
    });
    let records = table_wal_records();
    let committed = txn.commit();
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { row_id: 4, .. })),
        "{committed:?}"
    );
    assert_eq!(
        table_wal_records() - records,
        1,
        "the table went to the WAL once"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 7"), 1);
}

#[test]
fn a_seal_after_a_skipped_recheck_brings_its_conflict() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    create(&db);
    seal_row(&db, 1, 10);
    // Written after the last seal, so its record is the commit's generation
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 0, 7, 0)", ())
        .unwrap();
    let other = db.clone();
    let mut sealed = false;
    on_commit_fence(move || {
        if !std::mem::replace(&mut sealed, true) {
            seal_row_apart(&other, 4, 7);
        }
    });
    let committed = txn.commit();
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { row_id: 4, .. })),
        "{committed:?}"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 7"), 1);
}

#[test]
fn a_first_seal_that_moved_the_generation_is_decided_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 0, 10, 0), (3, 0, 30, 0)", ())
        .unwrap();
    // A delete records no generation: the commit decides from the segments
    let mut txn = db.begin().unwrap();
    txn.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let (paused_tx, paused) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let sealer = db.clone();
    let sealing = std::thread::spawn(move || {
        after_seal_generation_moved(move || {
            let _ = paused_tx.send(());
            go_rx
                .recv_timeout(WATCHDOG)
                .expect("the commit reached its fence while the seal was paused");
        });
        seal(&sealer);
    });
    paused
        .recv_timeout(WATCHDOG)
        .expect("the first seal moved the generation");
    let mut go = Some(go);
    on_commit_fence(move || {
        if let Some(go) = go.take() {
            let _ = go.send(());
        }
    });
    let rechecks = cold_rechecks();
    txn.commit().unwrap();
    sealing.join().unwrap();
    assert_eq!(cold_rechecks() - rechecks, 1, "the commit did not recheck");
}

/// Whether the column change and the seal finished, None for one that did
/// not in time
type Finished = Rc<Cell<Option<(Option<bool>, Option<bool>)>>>;

/// Seals a row in each of the next two rechecks and, in the third, drops
/// column `x` and seals another; `done` gets what the third saw finish
fn change_in_the_third_recheck(db: Database, rounds: Rc<Cell<usize>>, done: Finished) {
    on_cold_recheck(move || {
        let round = rounds.get() + 1;
        rounds.set(round);
        if round < 3 {
            seal_row_apart(&db, 100 + round as i64, 100 + round as i64);
            change_in_the_third_recheck(db, rounds, done);
            return;
        }
        let (ddl, sealer) = (db.clone(), db);
        let dropped = on_thread(move || ddl.execute("ALTER TABLE t DROP COLUMN x", ()).is_ok());
        let sealed = on_thread(move || {
            sealer
                .execute("INSERT INTO t VALUES (60, 60, 0)", ())
                .is_ok()
                && sealer.engine().seal_for_test().is_ok()
        });
        done.set(Some((dropped, sealed)));
    });
}

#[test]
fn a_column_change_and_a_seal_in_the_third_recheck_are_schema_changed() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    // Every round finds k = 7 sealed, so each keeps a constraint error
    let (db, mut txn) = pending(dir.path(), 7);
    let rounds = Rc::new(Cell::new(0));
    let done = Rc::new(Cell::new(None));
    change_in_the_third_recheck(db.clone(), Rc::clone(&rounds), Rc::clone(&done));
    let committed = txn.commit();
    assert_eq!(rounds.get(), 3, "the commit did not recheck three times");
    let done = done.take().expect("the third recheck ran");
    assert_eq!(
        done,
        (Some(true), Some(true)),
        "the column change and the seal did not both finish inside the third recheck"
    );
    assert!(
        matches!(committed, Err(Error::SchemaChanged { .. })),
        "{committed:?}"
    );
}

#[test]
fn a_column_change_between_two_rechecks_is_schema_changed() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    // Both rounds find k = 7 sealed, so each keeps a constraint error
    let (db, mut txn) = pending(dir.path(), 7);
    let sealer = db.clone();
    on_cold_recheck(move || seal_row_apart(&sealer, 101, 101));
    let ddl = db.clone();
    let dropped = Rc::new(Cell::new(None));
    let slot = Rc::clone(&dropped);
    after_commit_round_moved(move || {
        slot.set(on_thread(move || {
            ddl.execute("ALTER TABLE t DROP COLUMN x", ()).is_ok()
        }));
    });
    let committed = txn.commit();
    assert_eq!(
        dropped.get(),
        Some(true),
        "the column change did not finish between the rounds"
    );
    assert!(
        matches!(committed, Err(Error::SchemaChanged { .. })),
        "{committed:?}"
    );
}

/// `pending`, with `conflict` committed and sealed in a volume of its own
/// when given, and the volumes compacted into one while the recheck runs,
/// after its snapshot was taken
fn compacted_during_recheck(k3: i64, conflict: Option<&str>) -> Compacted {
    let dir = tempfile::tempdir().unwrap();
    let (db, mut txn) = pending(dir.path(), k3);
    if let Some(sql) = conflict {
        db.execute(sql, ()).unwrap();
        seal(&db);
    }
    assert!(db.engine().volume_stats().len() >= 2);
    db.engine().cold_volumes_for_test("t");
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let compactor = db.clone();
    let compacted = Rc::new(Cell::new(None));
    let slot = Rc::clone(&compacted);
    on_cold_recheck(move || {
        slot.set(on_thread(move || {
            compactor.engine().compact_for_test().is_ok()
        }));
    });
    let (loads, files, stores) = (
        live_volume_loads(),
        volume_file_drops_by_fence(),
        block_store_drops_by_fence(),
    );
    let committed = txn.commit();
    let measured = Compacted {
        live_loads: live_volume_loads() - loads,
        file_drops: delta(volume_file_drops_by_fence(), files),
        store_drops: delta(block_store_drops_by_fence(), stores),
        committed,
        db,
        _dir: dir,
    };
    assert_eq!(compacted.get(), Some(true), "the compaction did not finish");
    assert_eq!(
        measured.db.engine().volume_stats().len(),
        1,
        "the volumes were not merged"
    );
    measured
}

/// What a commit did while its volumes were compacted under its recheck
struct Compacted {
    committed: stoolap::Result<()>,
    /// Volumes the commit loaded through the live segment map
    live_loads: usize,
    /// Volume files and block stores the commit let go, (outside, under)
    file_drops: (usize, usize),
    store_drops: (usize, usize),
    db: Database,
    _dir: tempfile::TempDir,
}

#[test]
fn a_unique_conflict_in_a_volume_compacted_during_the_recheck_is_found() {
    let _guard = FailpointGuard::new();
    let run = compacted_during_recheck(7, None);
    assert!(
        matches!(
            run.committed,
            Err(Error::UniqueConstraint { row_id: 3, .. })
        ),
        "{:?}",
        run.committed
    );
    assert_eq!(count(&run.db, "SELECT COUNT(*) FROM t WHERE k = 7"), 1);
}

#[test]
fn a_primary_key_conflict_in_a_volume_compacted_during_the_recheck_is_found() {
    let _guard = FailpointGuard::new();
    let run = compacted_during_recheck(30, Some("INSERT INTO t VALUES (2, 0, 70, 0)"));
    assert!(
        matches!(
            run.committed,
            Err(Error::PrimaryKeyConstraint { row_id: 2 })
        ),
        "{:?}",
        run.committed
    );
    assert_eq!(count(&run.db, "SELECT k FROM t WHERE id = 2"), 70);
}

#[test]
fn a_primary_key_recheck_loads_no_volume_through_the_live_map() {
    let _guard = FailpointGuard::new();
    let run = compacted_during_recheck(30, Some("INSERT INTO t VALUES (2, 0, 70, 0)"));
    assert!(run.committed.is_err(), "{:?}", run.committed);
    assert_eq!(run.live_loads, 0, "volumes loaded through the live map");
}

#[test]
fn a_volume_retired_during_the_recheck_is_let_go_outside_the_fence() {
    let _guard = FailpointGuard::new();
    let run = compacted_during_recheck(30, None);
    run.committed.as_ref().unwrap();
    assert_eq!(
        run.file_drops.1, 0,
        "a volume file let go under the seal fence"
    );
    assert_eq!(
        run.store_drops.1, 0,
        "a block store let go under the seal fence"
    );
    assert!(
        run.store_drops.0 > 0,
        "the commit let go of no loaded volume it held last"
    );
    assert_eq!(count(&run.db, "SELECT COUNT(*) FROM t"), 4);
}

#[test]
fn a_recheck_whose_volumes_keep_changing_fails_the_commit_and_undoes_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    create(&db);
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, w INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX u_w ON u(w)", ()).unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    seal_row(&db, 1, 10);
    // u is committed first; only t's recheck meets the seals
    let mut txn = db.begin().unwrap();
    txn.execute("UPDATE u SET w = 5 WHERE id = 1", ()).unwrap();
    txn.execute("INSERT INTO u VALUES (2, 20)", ()).unwrap();
    txn.execute("INSERT INTO t VALUES (2, 0, 7, 0)", ())
        .unwrap();
    seal_row(&db, 3, 30);
    let other = db.clone();
    let fences = Rc::new(Cell::new(0));
    let seen = Rc::clone(&fences);
    on_commit_fence(move || {
        let n = seen.get();
        seen.set(n + 1);
        if n > 0 {
            seal_row_apart(&other, 100 + n, 100 + n);
        }
    });
    let records = table_wal_records();
    let committed = txn.commit();
    assert!(
        matches!(&committed, Err(Error::Internal { message }) if message.contains("write conflict")),
        "{committed:?}"
    );
    assert_eq!(fences.get(), 4, "u once, then three rounds of t");
    assert_eq!(
        table_wal_records() - records,
        2,
        "each table went to the WAL once"
    );
    let check = |db: &Database, at: &str| {
        assert_eq!(
            count(db, "SELECT w FROM u WHERE id = 1"),
            1,
            "{at}: u's update stayed"
        );
        assert_eq!(
            count(db, "SELECT COUNT(*) FROM u WHERE w = 20"),
            0,
            "{at}: u's insert stayed"
        );
        assert_eq!(count(db, "SELECT COUNT(*) FROM u"), 1, "{at}: u's rows");
        assert_eq!(
            count(db, "SELECT COUNT(*) FROM t WHERE k = 7"),
            0,
            "{at}: t's insert stayed"
        );
    };
    check(&db, "live");
    db.execute("UPDATE u SET w = 6 WHERE id = 1", ())
        .expect("the failed transaction still holds u's row");
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let reopened = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(
        count(&reopened, "SELECT w FROM u WHERE id = 1"),
        6,
        "recovered"
    );
    assert_eq!(
        count(&reopened, "SELECT COUNT(*) FROM u WHERE w = 20"),
        0,
        "recovered"
    );
    assert_eq!(
        count(&reopened, "SELECT COUNT(*) FROM t WHERE k = 7"),
        0,
        "recovered"
    );
}
