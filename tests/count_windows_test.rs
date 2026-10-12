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

//! COUNT(*) stays exact while a compaction publishes and cleans up its
//! tombstones and while a truncate clears the segment set

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use stoolap::test_failpoints as fp;
use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// `volumes` volumes of ten rows each, ids from 1
fn volumes_of_ten(dir: &Path, volumes: i64) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for volume in 0..volumes {
        let values: Vec<String> = (volume * 10 + 1..=volume * 10 + 10)
            .map(|id| format!("({id}, {id})"))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), volumes as usize);
    db
}

fn count(db: &Database) -> i64 {
    db.query_one("SELECT COUNT(*) FROM t", ()).unwrap()
}

#[test]
fn count_between_a_compaction_publishing_and_its_tombstone_cleanup_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let db = volumes_of_ten(dir.path(), 3);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(count(&db), 29);

    let seen = Rc::new(Cell::new(None));
    let window = Rc::clone(&seen);
    let other = db.clone();
    fp::after_compaction_published(move || window.set(Some(count(&other))));
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    assert_eq!(db.engine().volume_stats().len(), 1, "one output");
    assert_eq!(seen.get(), Some(29), "COUNT(*) inside the window");
    assert_eq!(count(&db), 29);
}

#[test]
fn count_whose_segments_predate_a_whole_compaction_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let db = volumes_of_ten(dir.path(), 3);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();

    let ran = Rc::new(Cell::new(false));
    let compacted = Rc::clone(&ran);
    let other = db.clone();
    fp::between_count_captures(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        compacted.set(true);
    });
    let n = count(&db);
    assert!(ran.get());
    assert_eq!(db.engine().volume_stats().len(), 1, "one output");
    assert_eq!(n, 29, "one map read before a whole compaction, one after");
}

#[test]
fn count_that_began_before_a_compaction_published_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let db = volumes_of_ten(dir.path(), 3);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();

    let (published_tx, published) = std::sync::mpsc::channel();
    let (resume, resume_rx) = std::sync::mpsc::channel::<()>();
    let compactor = db.clone();
    let compaction = std::thread::spawn(move || {
        fp::after_compaction_published(move || {
            published_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
        });
        compactor.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let start = Rc::new(Cell::new(Some(published)));
    let waiting = Rc::clone(&start);
    fp::between_count_captures(move || {
        waiting.take().unwrap().recv().unwrap();
    });
    let n = count(&db);
    resume.send(()).unwrap();
    compaction.join().unwrap();

    assert_eq!(db.engine().volume_stats().len(), 1, "one output");
    assert_eq!(n, 29, "the outputs published while the count read its maps");
}

/// The files as they stand, without the lock file
fn copy_files(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_files(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn count_after_a_crash_inside_the_cleanup_window_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let image = tempfile::tempdir().unwrap();
    let db = volumes_of_ten(dir.path(), 3);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();

    let (from, to) = (dir.path().to_path_buf(), image.path().to_path_buf());
    let checkpointer = db.clone();
    let checkpoint = Rc::new(Cell::new(None));
    let joined = Rc::clone(&checkpoint);
    fp::after_compaction_published(move || {
        let (persisted_tx, persisted) = std::sync::mpsc::channel();
        let (go, go_rx) = std::sync::mpsc::channel::<()>();
        joined.set(Some(std::thread::spawn(move || {
            fp::after_checkpoint_manifests_persisted(move || {
                persisted_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            checkpointer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        })));
        persisted.recv().unwrap();
        copy_files(&from, &to);
        go.send(()).unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    checkpoint
        .take()
        .expect("the compaction published")
        .join()
        .unwrap();
    assert_eq!(count(&db), 29);

    let reopened = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(
        reopened.engine().volume_stats().len(),
        1,
        "the output alone"
    );
    assert_eq!(
        count(&reopened),
        29,
        "the window's tombstone came back from disk"
    );
}

#[test]
fn count_of_one_volume_with_many_tombstones_reads_no_row_ids() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let values: Vec<String> = (1..=2000).map(|id| format!("({id}, {id})")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1);
    db.execute("DELETE FROM t WHERE id % 2 = 0", ()).unwrap();

    let walked = fp::count_ids_walked();
    assert_eq!(count(&db), 1000);
    assert_eq!(fp::count_ids_walked() - walked, 0, "no row id read");
}

#[test]
fn a_reopened_volume_counts_without_row_ids_once_its_tombstones_checked_out() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(&dsn(dir.path())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        let values: Vec<String> = (1..=2000).map(|id| format!("({id}, {id})")).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("DELETE FROM t WHERE id % 2 = 0", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1);
    assert_eq!(
        count(&db),
        1000,
        "the tombstones are checked against the volume"
    );
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();

    let walked = fp::count_ids_walked();
    assert_eq!(count(&db), 999);
    assert_eq!(fp::count_ids_walked() - walked, 0, "no row id read");
}

#[test]
fn count_halfway_through_a_truncate_does_not_bring_deleted_rows_back() {
    let dir = tempfile::tempdir().unwrap();
    let db = volumes_of_ten(dir.path(), 2);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();

    let seen = Rc::new(Cell::new(None));
    let halfway = Rc::clone(&seen);
    let other = db.clone();
    fp::during_segments_clearing(move || halfway.set(Some(count(&other))));
    db.execute("TRUNCATE TABLE t", ()).unwrap();

    let n = seen.get().expect("the truncate cleared the segment set");
    assert!(n == 0 || n == 19, "{n} rows halfway through the truncate");
    assert_eq!(count(&db), 0);
}
