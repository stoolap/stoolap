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

//! Backfill of the secondary index side files: a volume the query path
//! cannot serve although the catalog indexes one of its columns gets its
//! side file without waiting for a compaction, one volume at a time, under
//! half the builds budget, published under the DDL guard once it covers
//! every current identity and its volume is still registered.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use stoolap::Database;

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

fn expected_range(low: i64, high: i64) -> Vec<i64> {
    (1..=ROWS)
        .filter(|&id| (low..=high).contains(&key_of(id)))
        .collect()
}

fn open(dir: &Path, extra: &str) -> Database {
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

/// Seals `volumes` volumes of the fixture's rows, in order
fn seal_volumes(db: &Database, volumes: i64) {
    for batch in 0..volumes {
        insert(
            db,
            batch * (ROWS / volumes) + 1,
            (batch + 1) * (ROWS / volumes),
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
}

fn range(db: &Database, low: i64, high: i64) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query("SELECT id FROM t WHERE k >= $1 AND k <= $2", (low, high))
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

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

/// The row of `PRAGMA INDEX_BACKFILL`, by column name
fn backfill(db: &Database, limit: Option<i64>) -> BTreeMap<String, i64> {
    let sql = match limit {
        Some(n) => format!("PRAGMA INDEX_BACKFILL = {n}"),
        None => "PRAGMA INDEX_BACKFILL".to_string(),
    };
    let rows = db.query(&sql, ()).unwrap();
    let columns: Vec<String> = rows.columns().to_vec();
    let row = rows.into_iter().next().unwrap().unwrap();
    columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), row.get::<i64>(i).unwrap()))
        .collect()
}

/// A range read: the rows, and how the volumes were served
fn served(db: &Database, low: i64, high: i64) -> (i64, i64) {
    let before = reads(db);
    assert_eq!(range(db, low, high), expected_range(low, high));
    let after = reads(db);
    (
        delta(&after, &before, "probes"),
        delta(&after, &before, "ineligible"),
    )
}

/// The volumes with a file of `ext` beside them: a side file of a later
/// generation is named `vol_<id>.g<generation>.sidx`
fn files(dir: &Path, table: &str, ext: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir.join("volumes").join(table)) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                names.insert(stoolap::storage::volume::secondary::side_stem(&path).unwrap());
            }
        }
    }
    names
}

/// The side files beside `table`'s volumes, by name
fn side_names(dir: &Path, table: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("volumes").join(table))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("sidx"))
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn leftovers(dir: &Path, table: &str) -> Vec<String> {
    std::fs::read_dir(dir.join("volumes").join(table))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.contains(".build-"))
                .collect()
        })
        .unwrap_or_default()
}

/// Volumes sealed before their index existed are scanned; a pass builds
/// their side files, oldest first, and they are probed from then on.
#[test]
fn volumes_sealed_before_the_index_are_covered_by_a_pass() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 4);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "no side file yet"
    );
    assert_eq!(served(&db, 100, 103), (0, 4), "every volume is ineligible");
    // Two volumes per pass, oldest first
    let first = backfill(&db, Some(2));
    assert_eq!(
        (first["examined"], first["built"], first["left"]),
        (2, 2, 2)
    );
    assert_eq!(served(&db, 100, 103), (2, 2));
    let second = backfill(&db, None);
    assert_eq!((second["built"], second["left"]), (2, 0));
    assert_eq!(files(dir.path(), "t", "sidx").len(), 4);
    assert_eq!(served(&db, 100, 103), (4, 0), "every volume is probed");
    assert_eq!(served(&db, 0, 2499), (0, 0), "the wide range still scans");
    let third = backfill(&db, None);
    assert_eq!(third["examined"], 0, "nothing left to cover");
    assert!(leftovers(dir.path(), "t").is_empty());
}

/// A build the budget refused at seal leaves the volume uncovered; with
/// the budget back, a pass covers it.
#[test]
fn a_volume_whose_build_was_refused_at_seal_is_covered_by_a_pass() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    insert(&db, 1, ROWS / 2);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA INDEX_BUILD_MB = 0", ()).unwrap();
    insert(&db, ROWS / 2 + 1, ROWS);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA INDEX_BUILD_MB = 64", ()).unwrap();
    assert_eq!(
        files(dir.path(), "t", "sidx").len(),
        1,
        "the second volume is uncovered"
    );
    assert_eq!(served(&db, 100, 103), (1, 1));
    let pass = backfill(&db, None);
    assert_eq!((pass["examined"], pass["built"]), (1, 1));
    assert_eq!(files(dir.path(), "t", "sidx").len(), 2);
    assert_eq!(served(&db, 100, 103), (2, 0));
}

/// A side file that does not open leaves its volume uncovered at reopen;
/// a pass replaces it.
#[test]
fn a_side_file_that_does_not_open_is_replaced_by_a_pass() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path(), "");
        create(&db);
        db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
        seal_volumes(&db, 2);
        db.close().unwrap();
    }
    let table_dir = dir.path().join("volumes").join("t");
    let mut sides: Vec<_> = std::fs::read_dir(&table_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("sidx"))
        .collect();
    sides.sort();
    // The first volume's file is cut short
    let bytes = std::fs::read(&sides[0]).unwrap();
    std::fs::write(&sides[0], &bytes[..bytes.len() / 2]).unwrap();
    let db = open(dir.path(), "");
    assert_eq!(
        served(&db, 100, 103),
        (1, 1),
        "the cut file leaves its volume uncovered"
    );
    let pass = backfill(&db, None);
    assert_eq!((pass["examined"], pass["built"]), (1, 1));
    assert_eq!(served(&db, 100, 103), (2, 0));
    let db = {
        db.close().unwrap();
        open(dir.path(), "")
    };
    assert_eq!(served(&db, 100, 103), (2, 0), "the rebuilt file reopens");
}

/// A backfill takes at most half the builds budget, and sizes its
/// workspace to that share: a volume whose least workspace does not fit
/// is refused and left, and one that fits is built in runs merged under
/// the share.
#[test]
fn a_build_is_sized_to_half_the_budget_and_refused_only_below_its_least() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    // A single volume of 100,000 rows: its workspace at the engine's share
    // is above half a megabyte, its least is not
    for chunk_start in (1..=100_000i64).step_by(2000) {
        let values = (chunk_start..chunk_start + 2000)
            .map(|id| format!("({id},{},{})", (id * 7919) % 100_000, id))
            .collect::<Vec<_>>()
            .join(",");
        db.execute(&format!("INSERT INTO t VALUES {values}"), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("PRAGMA INDEX_BUILD_MB = 0", ()).unwrap();
    let pass = backfill(&db, None);
    assert_eq!(
        (pass["examined"], pass["refused"], pass["built"]),
        (1, 1, 0),
        "{pass:?}"
    );
    assert!(files(dir.path(), "t", "sidx").is_empty());
    db.execute("PRAGMA INDEX_BUILD_MB = 1", ()).unwrap();
    let pass = backfill(&db, None);
    db.execute("PRAGMA INDEX_BUILD_MB = 64", ()).unwrap();
    assert_eq!((pass["examined"], pass["built"]), (1, 1), "{pass:?}");
    let before = reads(&db);
    let rows: Vec<i64> = db
        .query("SELECT id FROM t WHERE k = 7919", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(rows, vec![1]);
    assert_eq!(delta(&reads(&db), &before, "probes"), 1);
}

/// Index DDL between the build and the publication: the built file does
/// not stand for the recreated index, so it is discarded, nothing is
/// attached, and the next pass builds under the new identity.
#[cfg(feature = "test-failpoints")]
#[test]
fn index_ddl_between_the_build_and_the_publication_discards_the_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_backfilled(move || {
        other.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
        other.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    });
    let pass = backfill(&db, None);
    assert_eq!(
        (pass["examined"], pass["built"], pass["discarded"]),
        (1, 0, 1),
        "{pass:?}"
    );
    assert_eq!(served(&db, 100, 103), (0, 1), "nothing was attached");
    let pass = backfill(&db, None);
    assert_eq!((pass["examined"], pass["built"]), (1, 1), "{pass:?}");
    assert_eq!(served(&db, 100, 103), (1, 0));
    assert!(leftovers(dir.path(), "t").is_empty());
}

/// A compaction between the build and the publication takes the volume
/// away: the file goes with it and nothing is left on disk.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_between_the_build_and_the_publication_retires_the_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 4);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    assert_eq!(files(dir.path(), "t", "vol").len(), 4);
    let other = db.clone();
    stoolap::test_failpoints::after_side_backfilled(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(other.engine().volume_stats().len(), 1, "compacted");
    });
    let pass = backfill(&db, Some(1));
    assert_eq!(
        (pass["examined"], pass["built"], pass["discarded"]),
        (1, 0, 1),
        "{pass:?}"
    );
    // The compaction's output carries its own side file; the input's is gone
    let vols = files(dir.path(), "t", "vol");
    assert_eq!(vols.len(), 1);
    assert_eq!(files(dir.path(), "t", "sidx"), vols);
    assert!(leftovers(dir.path(), "t").is_empty());
    assert_eq!(served(&db, 100, 103), (1, 0));
}

/// A file built beside a volume but never attached is attached at reopen
/// and served, judged per use by its identity.
#[test]
fn a_built_file_left_unattached_is_attached_at_reopen() {
    use stoolap::storage::volume::secondary::{build_side_file, next_generation, ColumnInput};
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let identity;
    {
        let db = open(dir.path(), "");
        create(&db);
        seal_volumes(&db, 1);
        db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
        identity = db.engine().index_identity("t", "idx_t_k").unwrap();
        assert_eq!(served(&db, 100, 103), (0, 1));
        db.close().unwrap();
    }
    // The file the pass would have written: position i holds row i + 1
    let table_dir = dir.path().join("volumes").join("t");
    let volume = std::fs::read_dir(&table_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("vol"))
        .unwrap();
    let side = volume.with_extension("sidx");
    build_side_file(
        &side,
        next_generation(),
        vec![ColumnInput {
            column: 1,
            identity,
            pairs: Box::new((0..ROWS as u32).map(|i| (i, key_of(i as i64 + 1)))),
        }],
        4 * 1024 * 1024,
    )
    .unwrap();
    let db = open(dir.path(), "");
    assert_eq!(served(&db, 100, 103), (1, 0), "attached at reopen");
    let pass = backfill(&db, None);
    assert_eq!(pass["examined"], 0);
}

/// The maintenance thread covers one volume per cycle on its own: the
/// volumes stay the ones sealed, so the coverage is the backfill's.
#[test]
fn the_maintenance_thread_covers_the_volumes_without_a_pragma() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    // A one-second cycle from the start
    let db = open(dir.path(), "&checkpoint_interval=1");
    create(&db);
    seal_volumes(&db, 2);
    let volumes = files(dir.path(), "t", "vol");
    assert_eq!(volumes.len(), 2);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    assert_eq!(served(&db, 100, 103), (0, 2));
    let start = std::time::Instant::now();
    loop {
        if served(&db, 100, 103) == (2, 0) {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "the thread did not cover the volumes in time"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    assert_eq!(
        files(dir.path(), "t", "vol"),
        volumes,
        "no rewrite happened"
    );
    assert_eq!(files(dir.path(), "t", "sidx"), volumes);
}

/// A side file that covers one index and not another the catalog has now
/// is rebuilt for both: the pass asks every current identity of a volume.
#[test]
fn a_file_covering_one_index_of_two_is_rebuilt_for_both() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    seal_volumes(&db, 1);
    assert_eq!(served(&db, 100, 103), (1, 0), "covered for k");
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    let on_v = |db: &Database| -> (i64, i64) {
        let before = reads(db);
        let rows: Vec<i64> = db
            .query("SELECT id FROM t WHERE v = 300", ())
            .unwrap()
            .map(|r| r.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows, vec![100]);
        let after = reads(db);
        (
            delta(&after, &before, "probes"),
            delta(&after, &before, "ineligible"),
        )
    };
    assert_eq!(on_v(&db), (0, 1), "not covered for v");
    let pass = backfill(&db, None);
    assert_eq!((pass["examined"], pass["built"]), (1, 1), "{pass:?}");
    assert_eq!(on_v(&db), (1, 0));
    assert_eq!(served(&db, 100, 103), (1, 0), "still covered for k");
    // The replacement has its generation's name; the file before goes
    // with its last holder
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);
    assert_eq!(
        side_names(dir.path(), "t").len(),
        1,
        "{:?}",
        side_names(dir.path(), "t")
    );
}

/// A compaction that takes the volume away before its build: the file is
/// built beside a retired volume, the attach finds no segment, and the
/// file is retired rather than left an orphan.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_before_the_build_leaves_no_orphan_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 4);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_backfill_volume_loaded(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(other.engine().volume_stats().len(), 1, "compacted");
    });
    let pass = backfill(&db, Some(1));
    assert_eq!(
        (pass["examined"], pass["built"], pass["discarded"]),
        (1, 0, 1),
        "{pass:?}"
    );
    let vols = files(dir.path(), "t", "vol");
    assert_eq!(vols.len(), 1);
    assert_eq!(files(dir.path(), "t", "sidx"), vols, "no orphan side file");
    assert!(leftovers(dir.path(), "t").is_empty());
    assert_eq!(served(&db, 100, 103), (1, 0));
}

/// A pass replacing a volume's side file leaves a reader of the old file
/// on its own bytes: a read through another handle inside the pass, with
/// the page cache cleared, answers and probes as before.
#[cfg(feature = "test-failpoints")]
#[test]
fn replacing_a_side_file_leaves_its_readers_on_their_own_bytes() {
    use stoolap::storage::volume::secondary::INDEX_PAGES;
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    seal_volumes(&db, 1);
    assert_eq!(served(&db, 100, 103), (1, 0), "covered for k");
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    let other = db.clone();
    let inside = std::sync::Arc::new(std::sync::Mutex::new(None));
    let seen = std::sync::Arc::clone(&inside);
    stoolap::test_failpoints::after_side_backfilled(move || {
        INDEX_PAGES.clear();
        *seen.lock().unwrap() = Some(served(&other, 100, 103));
    });
    let pass = backfill(&db, None);
    assert_eq!((pass["examined"], pass["built"]), (1, 1), "{pass:?}");
    assert_eq!(
        inside.lock().unwrap().take(),
        Some((1, 0)),
        "the read inside the pass came through the old file"
    );
    INDEX_PAGES.clear();
    assert_eq!(
        served(&db, 100, 103),
        (1, 0),
        "and through the new one after"
    );
}

/// A replacement discarded at publication leaves the attached file and a
/// later replacement alone: the file count holds through close and reopen.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_discarded_replacement_does_not_retire_the_file_a_retry_publishes() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_backfilled(move || {
        other.execute("DROP INDEX idx_t_v ON t", ()).unwrap();
        other.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    });
    let pass = backfill(&db, None);
    assert_eq!((pass["built"], pass["discarded"]), (0, 1), "{pass:?}");
    assert_eq!(served(&db, 100, 103), (1, 0), "k stays covered");
    let pass = backfill(&db, None);
    assert_eq!((pass["built"], pass["discarded"]), (1, 0), "{pass:?}");
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);
    assert!(leftovers(dir.path(), "t").is_empty());
    db.close().unwrap();
    assert_eq!(
        files(dir.path(), "t", "sidx").len(),
        1,
        "the file survives close"
    );
    let db = open(dir.path(), "");
    assert_eq!(served(&db, 100, 103), (1, 0));
}

/// Close waits for a pass that is running and starts no other; a pass
/// paused inside its build holds close until it has let its volume go.
#[cfg(feature = "test-failpoints")]
#[test]
fn close_waits_for_a_running_pass() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let (loaded_tx, loaded_rx) = mpsc::channel::<()>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    // The hook is the pass thread's own
    let pass_db = db.clone();
    let pass = std::thread::spawn(move || {
        stoolap::test_failpoints::after_backfill_volume_loaded(move || {
            loaded_tx.send(()).unwrap();
            go_rx.recv().unwrap();
        });
        backfill(&pass_db, None)
    });
    loaded_rx.recv().unwrap();
    let closed = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&closed);
    let close_db = db.clone();
    let closer = std::thread::spawn(move || {
        close_db.close().unwrap();
        flag.store(true, Ordering::SeqCst);
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(!closed.load(Ordering::SeqCst), "close waited for the pass");
    go_tx.send(()).unwrap();
    let report = pass.join().unwrap();
    closer.join().unwrap();
    assert!(closed.load(Ordering::SeqCst));
    assert_eq!(
        report["built"], 0,
        "nothing published into a closing engine: {report:?}"
    );
    drop(db);
    // The next engine's own pass covers the volume
    let db = open(dir.path(), "");
    let pass = backfill(&db, None);
    assert_eq!(pass["built"], 1, "{pass:?}");
    assert_eq!(served(&db, 100, 103), (1, 0));
}

/// A table renamed while its volume's side file is being built, its old
/// name taken by a new table: the file is published where the volume is
/// now, and reopens with it.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_during_the_build_publishes_where_the_volume_is_now() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_backfill_volume_loaded(move || {
        other
            .execute("ALTER TABLE t RENAME TO archived", ())
            .unwrap();
        other
            .execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
                (),
            )
            .unwrap();
        other.execute("INSERT INTO t VALUES (1, 1, 1)", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let pass = backfill(&db, None);
    assert_eq!(pass["built"], 1, "{pass:?}");
    let archived = files(dir.path(), "archived", "vol");
    assert_eq!(archived.len(), 1);
    assert_eq!(files(dir.path(), "archived", "sidx"), archived);
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "the new table has no side file"
    );
    let probed = |db: &Database| -> (i64, i64) {
        let before = reads(db);
        let rows: Vec<i64> = db
            .query("SELECT id FROM archived WHERE k >= 100 AND k <= 103", ())
            .unwrap()
            .map(|r| r.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows.len(), expected_range(100, 103).len());
        let after = reads(db);
        (
            delta(&after, &before, "probes"),
            delta(&after, &before, "ineligible"),
        )
    };
    assert_eq!(probed(&db), (1, 0));
    db.close().unwrap();
    let db = open(dir.path(), "");
    assert_eq!(probed(&db), (1, 0), "after reopen");
}

/// A small volume fits half of a small budget: its workspace is sized to
/// the share, not to the engine's default.
#[test]
fn a_small_volume_is_built_under_a_small_budget() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    insert(&db, 1, 1000);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("PRAGMA INDEX_BUILD_MB = 8", ()).unwrap();
    let pass = backfill(&db, None);
    db.execute("PRAGMA INDEX_BUILD_MB = 64", ()).unwrap();
    assert_eq!((pass["built"], pass["refused"]), (1, 0), "{pass:?}");
    let before = reads(&db);
    let rows: Vec<i64> = db
        .query("SELECT id FROM t WHERE k = 100", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(
        rows,
        (1..=1000)
            .filter(|&id| key_of(id) == 100)
            .collect::<Vec<_>>()
    );
    assert_eq!(delta(&reads(&db), &before, "probes"), 1);
}

/// Makes a volume file unreadable, so its build fails, and readable again
#[cfg(unix)]
fn readable(path: &Path, yes: bool) {
    use std::os::unix::fs::PermissionsExt;
    let mode = if yes { 0o644 } else { 0o000 };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// A volume whose build keeps failing does not keep the newer ones from
/// their turn: the next pass starts after it.
#[cfg(unix)]
#[test]
fn a_failing_volume_does_not_starve_the_newer_ones() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 2);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.close().unwrap();
    let db = open(dir.path(), "");
    let mut vols: Vec<String> = files(dir.path(), "t", "vol").into_iter().collect();
    vols.sort();
    // The oldest volume cannot be read: its build fails
    let blocked = dir
        .path()
        .join("volumes")
        .join("t")
        .join(format!("{}.vol", vols[0]));
    readable(&blocked, false);
    let first = backfill(&db, Some(1));
    assert_eq!(
        (first["examined"], first["failed"], first["left"]),
        (1, 1, 1),
        "{first:?}"
    );
    let second = backfill(&db, Some(1));
    assert_eq!((second["examined"], second["built"]), (1, 1), "{second:?}");
    readable(&blocked, true);
    assert_eq!(served(&db, 100, 103), (1, 1), "the newer volume is probed");
    let third = backfill(&db, Some(1));
    assert_eq!(third["built"], 1, "{third:?}");
    assert_eq!(served(&db, 100, 103), (2, 0));
}

/// A volume that cannot be built in one table does not keep another
/// table's volumes from their turn either: the cursor spans the tables.
#[cfg(unix)]
#[test]
fn a_failing_volume_does_not_starve_another_table() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    // Table a sorts first; its only volume cannot be read
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO a VALUES (1, 1), (2, 2)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_a_k ON a(k)", ()).unwrap();
    create(&db);
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.close().unwrap();
    let db = open(dir.path(), "");
    let a_volume = std::fs::read_dir(dir.path().join("volumes").join("a"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("vol"))
        .unwrap();
    readable(&a_volume, false);
    let first = backfill(&db, Some(1));
    assert_eq!((first["failed"], first["left"]), (1, 1), "{first:?}");
    let second = backfill(&db, Some(1));
    assert_eq!(second["built"], 1, "t's volume takes its turn: {second:?}");
    assert_eq!(served(&db, 100, 103), (1, 0));
    readable(&a_volume, true);
    let third = backfill(&db, Some(1));
    assert_eq!(third["built"], 1, "{third:?}");
}

/// A table renamed after its volume's side file was staged: the staged
/// file is found where the directory went, published there, and nothing
/// is left behind; a discard after such a rename leaves nothing either.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_after_the_staging_publishes_where_the_volume_is_now() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_volumes(&db, 1);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_backfilled(move || {
        other
            .execute("ALTER TABLE t RENAME TO archived", ())
            .unwrap();
    });
    let pass = backfill(&db, None);
    assert_eq!(pass["built"], 1, "{pass:?}");
    let archived = files(dir.path(), "archived", "vol");
    assert_eq!(files(dir.path(), "archived", "sidx"), archived);
    assert!(
        leftovers(dir.path(), "archived").is_empty(),
        "no build directory left"
    );
    // Renamed back and the index recreated inside the pass: discarded, nothing left
    db.execute("ALTER TABLE archived RENAME TO t", ()).unwrap();
    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_backfilled(move || {
        other
            .execute("ALTER TABLE t RENAME TO archived", ())
            .unwrap();
        other.execute("DROP INDEX idx_t_k ON archived", ()).unwrap();
        other
            .execute("CREATE INDEX idx_t_k ON archived(k)", ())
            .unwrap();
    });
    let pass = backfill(&db, None);
    assert_eq!((pass["built"], pass["discarded"]), (0, 1), "{pass:?}");
    assert!(
        leftovers(dir.path(), "archived").is_empty(),
        "the discarded build left nothing"
    );
    db.close().unwrap();
    assert!(leftovers(dir.path(), "archived").is_empty());
}

/// A table renamed and then truncated or dropped while its volume's side
/// file is staged: the staged file is found through the volume's file
/// owner wherever the rename moved it, discarded, and nothing is left.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_then_truncate_or_drop_after_the_staging_leaves_nothing() {
    for drop_table in [false, true] {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path(), "");
        create(&db);
        seal_volumes(&db, 1);
        db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
        let other = db.clone();
        stoolap::test_failpoints::after_side_backfilled(move || {
            other
                .execute("ALTER TABLE t RENAME TO archived", ())
                .unwrap();
            if drop_table {
                other.execute("DROP TABLE archived", ()).unwrap();
            } else {
                other.execute("TRUNCATE TABLE archived", ()).unwrap();
            }
        });
        let pass = backfill(&db, None);
        assert_eq!((pass["built"], pass["discarded"]), (0, 1), "{pass:?}");
        assert!(
            leftovers(dir.path(), "archived").is_empty(),
            "drop {drop_table}: nothing left in the moved directory"
        );
        db.close().unwrap();
        assert!(
            leftovers(dir.path(), "archived").is_empty(),
            "drop {drop_table}: after close"
        );
    }
}
