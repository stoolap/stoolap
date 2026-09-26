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

//! A row a seal extracted and that changed or was claimed before the seal's
//! fence keeps its hot state; the copy the seal built never gets authority,
//! for any reader, at any snapshot, across a crash and a compaction

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::{mpsc, Arc};
use std::time::Duration;

use stoolap::storage::volume::manifest::{StoredManifest, TableManifest};
use stoolap::Database;

fn open(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.display()
    ))
    .unwrap()
}

/// 1,000 hot rows `(id, id * 10)`, the table's columns followed by `extra`
fn table(dir: &Path, extra: &str) -> Database {
    let db = open(dir);
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER{extra})"),
        (),
    )
    .unwrap();
    let values: Vec<String> = (1..=1_000)
        .map(|id| format!("({id},{})", id * 10))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db
}

/// Seals the table with `during` run on another thread after the seal
/// extracted the rows and before its fence; then a snapshot begins on
/// `snapshot` and another commit moves the sequence on. The snapshot, held
/// open, keeps the compaction off the new volume. Returns what the hook saw.
fn seal_while(
    db: &Database,
    snapshot: &Arc<Database>,
    during: impl FnOnce() + Send + 'static,
) -> Vec<String> {
    let (writer, s) = (db.clone(), Arc::clone(snapshot));
    let (events, seen) = mpsc::channel::<String>();
    stoolap::test_failpoints::after_side_files_built(move || {
        events.send("seal hook".into()).unwrap();
        std::thread::spawn(move || {
            during();
            s.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
                .unwrap();
            let _: i64 = s.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
            writer
                .execute("INSERT INTO t VALUES (5000, 1)", ())
                .unwrap();
        })
        .join()
        .unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    seen.try_iter().collect()
}

fn manifest_of(dir: &Path) -> StoredManifest {
    TableManifest::read_from_disk(&dir.join("volumes").join("t").join("manifest.bin")).unwrap()
}

fn dead_copies_on_disk(dir: &Path) -> usize {
    manifest_of(dir)
        .manifest
        .dead_copies
        .values()
        .map(|d| d.len())
        .sum()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

// Bug guard: fails on main, where the seal tombstoned the skipped copy at
// the current sequence and the open snapshot did not see the tombstone
#[test]
fn a_snapshot_never_sees_the_sealed_copy_of_a_row_updated_before_it_began() {
    let mut wrong = Vec::new();
    for extra in ["", ", UNIQUE (v)"] {
        let dir = tempfile::tempdir().unwrap();
        let db = table(dir.path(), extra);
        let snapshot = Arc::new(db.clone());
        let writer = db.clone();
        let log = seal_while(&db, &snapshot, move || {
            writer
                .execute("UPDATE t SET v = 15 WHERE id = 1", ())
                .unwrap();
        });
        assert_eq!(log, vec!["seal hook"]);
        let stale: i64 = snapshot
            .query_one("SELECT COUNT(*) FROM t WHERE v = 10", ())
            .unwrap();
        let ids: Vec<i64> = snapshot
            .query("SELECT id FROM t WHERE v = 10", ())
            .unwrap()
            .map(|r| r.unwrap().get(0).unwrap())
            .collect();
        let current: i64 = snapshot
            .query_one("SELECT v FROM t WHERE id = 1", ())
            .unwrap();
        snapshot.execute("COMMIT", ()).unwrap();
        if (stale, &ids, current) != (0, &vec![], 15) {
            wrong.push(format!("{extra:?}: count={stale} ids={ids:?} v={current}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "the snapshot saw the sealed copy: {wrong:?}"
    );
}

// Bug guard: fails when the seal copies the version behind a head still
// becoming visible, and moves that copy once the head commits
#[test]
fn a_seal_never_moves_the_version_behind_a_head_that_commits_during_it() {
    let mut wrong = Vec::new();
    for ((statement, expected), cutoff) in [
        ("DELETE FROM t WHERE id = 1", (999, vec![])),
        ("UPDATE t SET v = 15 WHERE id = 1", (1_000, vec![15])),
    ]
    .into_iter()
    .flat_map(|case| [(case.clone(), false), (case, true)])
    {
        let dir = tempfile::tempdir().unwrap();
        let db = table(dir.path(), "");
        // An open snapshot gives the seal its commit cutoff
        let snapshot = db.clone();
        if cutoff {
            snapshot
                .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
                .unwrap();
            let _: i64 = snapshot.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        }
        let writer = db.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel::<()>();
        let writer = std::thread::spawn(move || {
            stoolap::test_failpoints::before_commit_visible(move || {
                ready_tx.send(()).unwrap();
                resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            });
            writer.execute(statement, ()).unwrap();
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        stoolap::test_failpoints::after_side_files_built(move || {
            resume_tx.send(()).unwrap();
            writer.join().unwrap();
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        let v: Vec<i64> = db
            .query("SELECT v FROM t WHERE id = 1", ())
            .unwrap()
            .map(|r| r.unwrap().get(0).unwrap())
            .collect();
        if cutoff {
            snapshot.execute("COMMIT", ()).unwrap();
        }
        if (rows, &v) != (expected.0, &expected.1) {
            wrong.push(format!("{statement} cutoff={cutoff}: rows={rows} v={v:?}"));
        }
    }
    assert!(wrong.is_empty(), "the seal undid a commit: {wrong:?}");
}

// Bug guard: fails when a compaction takes the tombstones between a seal's
// hot removal and its tombstone cleanup, and applies a moved row's old
// tombstone to its new copy
#[test]
fn a_compaction_never_applies_a_moved_rows_old_tombstone_to_its_new_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for start in [1, 1_001, 2_001] {
        let values: Vec<String> = (start..start + 1_000)
            .map(|id| format!("({id},{})", id * 10))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("PRAGMA compact_threshold = 2", ()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel::<()>();
    let (captured_tx, captured_rx) = mpsc::channel();
    let compact = db.clone();
    let compact = std::thread::spawn(move || {
        stoolap::test_failpoints::after_maintenance_schema_taken(move || {
            stoolap::test_failpoints::after_compaction_tombstones_taken(move |_| {
                let _ = captured_tx.send(());
            });
            started_tx.send(()).unwrap();
            resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        });
        compact.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    db.execute("UPDATE t SET v = 20 WHERE id = 1", ()).unwrap();
    let (seal_tx, seal_rx) = mpsc::channel();
    stoolap::test_failpoints::in_seal_after_rows_removed(move || {
        resume_tx.send(()).unwrap();
        let captured = captured_rx.recv_timeout(Duration::from_secs(1)).is_ok();
        seal_tx.send(captured).unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    compact.join().unwrap();
    let captured_during_seal = seal_rx.try_recv().unwrap();
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    let v: Vec<i64> = db
        .query("SELECT v FROM t WHERE id = 1", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!((rows, v), (3_000, vec![20]), "the moved row was lost");
    assert!(
        !captured_during_seal,
        "the compaction captured a seal halfway"
    );
}

// Contract: a compaction takes its tombstones under the DDL guard a seal
// holds through its transfer; checked where they are taken, no timing
#[test]
fn a_compaction_takes_its_tombstones_under_the_ddl_guard() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for start in [1, 11] {
        let values: Vec<String> = (start..start + 10)
            .map(|id| format!("({id},{id})"))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let (held_tx, held_rx) = mpsc::channel();
    stoolap::test_failpoints::after_compaction_tombstones_taken(move |held| {
        held_tx.send(held).unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        held_rx.try_recv(),
        Ok(true),
        "the guard was free at the capture"
    );
}

// Bug guard: fails when a snapshot that begins after the seal read its
// cutoff loses the version it read to the seal's hot removal
#[test]
fn a_snapshot_begun_after_the_seal_cutoff_keeps_its_version() {
    let dir = tempfile::tempdir().unwrap();
    let db = table(dir.path(), "");
    let snapshot = Arc::new(db.clone());
    let (s, writer) = (Arc::clone(&snapshot), db.clone());
    let (before_tx, before_rx) = mpsc::channel();
    stoolap::test_failpoints::after_seal_cutoff_read(move || {
        s.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
            .unwrap();
        let _: i64 = s.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        std::thread::spawn(move || {
            writer
                .execute("UPDATE t SET v = 15 WHERE id = 1", ())
                .unwrap();
        })
        .join()
        .unwrap();
        let before: i64 = s.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
        before_tx.send(before).unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = before_rx.try_recv().unwrap();
    let after: i64 = snapshot
        .query_one("SELECT v FROM t WHERE id = 1", ())
        .unwrap();
    snapshot.execute("COMMIT", ()).unwrap();
    let current: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(
        (before, after, current, rows),
        (10, 10, 15, 1_000),
        "the seal changed an open snapshot's value"
    );
}

// Contract: the mask is in the manifest the checkpoint wrote, and a crash
// image of that state, reopened before any compaction, replays the newer
// value and never shows the dead copy
#[test]
fn a_dead_copy_stays_dead_across_a_crash_before_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = table(dir.path(), "");
    let snapshot = Arc::new(db.clone());
    let writer = db.clone();
    seal_while(&db, &snapshot, move || {
        writer
            .execute("UPDATE t SET v = 20 WHERE id = 1", ())
            .unwrap();
    });
    // A second volume, so the reopen recomputes visibility across segments;
    // the open snapshot keeps the compaction off both
    let values: Vec<String> = (3001..=3010).map(|id| format!("({id},{id})")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(manifest_of(dir.path()).manifest.segments.len(), 2);
    assert_eq!(dead_copies_on_disk(dir.path()), 1, "the mask was persisted");
    assert_eq!(manifest_of(dir.path()).tombstones, vec![]);

    let image = tempfile::tempdir().unwrap();
    copy_dir(dir.path(), image.path());
    let reopened = open(image.path());
    let current: i64 = reopened
        .query_one("SELECT v FROM t WHERE id = 1", ())
        .unwrap();
    let stale: i64 = reopened
        .query_one("SELECT COUNT(*) FROM t WHERE v = 10", ())
        .unwrap();
    let rows: i64 = reopened.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!((current, stale, rows), (20, 0, 1_011));
    // With the hot row gone, only the reopened mask keeps the copy dead
    reopened.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    reopened
        .engine()
        .vacuum(Some("t"), std::time::Duration::ZERO)
        .unwrap();
    let found: i64 = reopened
        .query_one("SELECT COUNT(*) FROM t WHERE id = 1 OR v = 10", ())
        .unwrap();
    assert_eq!(found, 0, "the dead copy came back after the reopen");
    snapshot.execute("COMMIT", ()).unwrap();
}

// Contract: a claimed row is deferred: no tombstone, its copy masked, and
// the claimant's rollback or commit decides what the next seal moves
#[test]
fn a_claimed_row_is_deferred_without_a_tombstone() {
    for commit in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table(dir.path(), "");
        let snapshot = Arc::new(db.clone());
        let claimant = Arc::new(db.clone());
        let c = Arc::clone(&claimant);
        seal_while(&db, &snapshot, move || {
            c.execute("BEGIN", ()).unwrap();
            c.execute("UPDATE t SET v = 999 WHERE id = 2", ()).unwrap();
        });
        assert_eq!(
            manifest_of(dir.path()).tombstones,
            vec![],
            "commit={commit}"
        );
        assert_eq!(dead_copies_on_disk(dir.path()), 1, "commit={commit}");
        claimant
            .execute(if commit { "COMMIT" } else { "ROLLBACK" }, ())
            .unwrap();
        snapshot.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let v: i64 = db.query_one("SELECT v FROM t WHERE id = 2", ()).unwrap();
        let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(
            (v, rows),
            (if commit { 999 } else { 20 }, 1_001),
            "commit={commit}"
        );
    }
}

// Contract: the dead copy's old value is no unique conflict; the hot value is
#[test]
fn a_dead_copy_is_no_unique_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let db = table(dir.path(), ", UNIQUE (v)");
    let snapshot = Arc::new(db.clone());
    let writer = db.clone();
    seal_while(&db, &snapshot, move || {
        writer
            .execute("UPDATE t SET v = 7 WHERE id = 1", ())
            .unwrap();
    });
    snapshot.execute("COMMIT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (6000, 10)", ())
        .expect("the dead copy's 10 is no conflict");
    assert!(db.execute("INSERT INTO t VALUES (6001, 7)", ()).is_err());
}

// Contract: once the hot row is deleted and collected, the dead copy does
// not come back
#[test]
fn a_dead_copy_does_not_return_after_its_hot_row_is_collected() {
    let dir = tempfile::tempdir().unwrap();
    let db = table(dir.path(), "");
    let snapshot = Arc::new(db.clone());
    let writer = db.clone();
    seal_while(&db, &snapshot, move || {
        writer
            .execute("UPDATE t SET v = 20 WHERE id = 1", ())
            .unwrap();
    });
    snapshot.execute("COMMIT", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    db.engine()
        .vacuum(Some("t"), std::time::Duration::ZERO)
        .unwrap();
    let found: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 1 OR v = 10", ())
        .unwrap();
    assert_eq!(found, 0);
}

// Contract: a compaction drops the dead copy instead of publishing it
#[test]
fn a_compaction_drops_a_dead_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = table(dir.path(), "");
    let snapshot = Arc::new(db.clone());
    let writer = db.clone();
    seal_while(&db, &snapshot, move || {
        writer
            .execute("UPDATE t SET v = 20 WHERE id = 1", ())
            .unwrap();
    });
    snapshot.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(dead_copies_on_disk(dir.path()), 0, "the rewrite dropped it");
    // A crash image after the rewrite replays the newer value: a published
    // dead copy would pass for the sealed row and the replay would skip it
    let image = tempfile::tempdir().unwrap();
    copy_dir(dir.path(), image.path());
    let replayed: i64 = open(image.path())
        .query_one("SELECT v FROM t WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        replayed, 20,
        "the replay took the published copy for the row"
    );
    let stale: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE v = 10", ())
        .unwrap();
    let current: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!((stale, current), (0, 20));
    // With the hot row gone, a published copy would answer
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    db.engine()
        .vacuum(Some("t"), std::time::Duration::ZERO)
        .unwrap();
    let found: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 1 OR v = 10", ())
        .unwrap();
    assert_eq!(found, 0, "the compaction published the dead copy");
}

// Contract: a volume left with only dead copies and deleted rows goes by
// the compaction's remove-only branch, taking its mask with it
#[test]
fn a_volume_of_dead_copies_and_deleted_rows_is_removed_whole() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = table(dir.path(), "");
        let snapshot = Arc::new(db.clone());
        let writer = db.clone();
        seal_while(&db, &snapshot, move || {
            writer
                .execute("UPDATE t SET v = 20 WHERE id = 1", ())
                .unwrap();
        });
        snapshot.execute("COMMIT", ()).unwrap();
        db.execute("DELETE FROM t", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let stored = manifest_of(dir.path());
        assert_eq!(
            (
                stored.manifest.segments.len(),
                stored.manifest.dead_copies.len()
            ),
            (0, 0)
        );
        let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(rows, 0);
        db.close().unwrap();
    }
    let db = open(dir.path());
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, 0);
}

// Contract: extraction with a cutoff reserves for the live rows, not for the
// deleted heads the root keeps
#[test]
fn cutoff_extraction_does_not_reserve_for_deleted_heads() {
    use stoolap::core::{DataType, Row, SchemaBuilder, Value};
    use stoolap::storage::mvcc::registry::TransactionRegistry;
    use stoolap::storage::mvcc::version_store::{RowVersion, VersionStore};
    let registry = Arc::new(TransactionRegistry::new());
    let (txn, _) = registry.begin_transaction();
    registry.commit_transaction(txn);
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .build();
    let mut store = VersionStore::new("t", schema);
    store.set_visibility_checker(Arc::clone(&registry));
    for id in 1..=100_000 {
        let row = Row::from_values(vec![Value::Integer(id)]);
        let version = if id <= 1_000 {
            RowVersion::new(txn, row)
        } else {
            RowVersion::new_deleted(txn, row)
        };
        store.add_version(id, version);
    }
    assert_eq!(store.committed_row_count(), 1_000);
    let (rows, _root) = store.extract_for_seal_with_cutoff(registry.current_commit_sequence() + 1);
    assert_eq!(rows.len(), 1_000);
    assert!(
        rows.capacity() <= 2_000,
        "live rows={}, reserved rows={}",
        rows.len(),
        rows.capacity()
    );
}
