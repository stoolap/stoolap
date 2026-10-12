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

//! A compaction applies a tombstone only where every copy of its row goes
//! with it: a copy left in a volume outside the batch must stay hidden

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000&target_volume_rows=65536",
        dir.display()
    )
}

fn values_of(db: &Database, id: i64) -> Vec<i64> {
    db.query("SELECT v FROM t WHERE id = $1", (id,))
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

fn insert(db: &Database, ids: impl Iterator<Item = i64>, k: impl Fn(i64) -> i64) {
    let ids: Vec<i64> = ids.collect();
    for chunk in ids.chunks(8_192) {
        let values: Vec<String> = chunk
            .iter()
            .map(|&id| format!("({id}, {}, 0)", k(id)))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
}

fn row_counts(db: &Database) -> Vec<usize> {
    db.engine().volume_stats().iter().map(|v| v.3).collect()
}

fn tombstones(db: &Database) -> usize {
    db.engine().volume_stats().first().map_or(0, |v| v.6)
}

/// Three small volumes: newer copies of `first` and `second`, then a new
/// row 1,000,001, enough for the batch rules to merge them
fn small_copies(db: &Database, first: i64, second: i64) {
    db.execute(&format!("UPDATE t SET v = 1 WHERE id = {first}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(&format!("UPDATE t SET v = 1 WHERE id = {second}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert(db, std::iter::once(1_000_001), |id| -id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

/// Compacts the small volumes, deleting `id` between the plan and the dedup
fn compact_deleting(db: &Database, id: i64) {
    let reached = Rc::new(Cell::new(false));
    let seen = Rc::clone(&reached);
    let other = db.clone();
    fp::after_compaction_planned(move || {
        seen.set(true);
        other
            .execute(&format!("DELETE FROM t WHERE id = {id}"), ())
            .unwrap();
    });
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 1000", ()).unwrap();
    assert!(reached.get(), "the compaction planned a batch");
}

/// The files as they stand, opened without the close's own checkpoint
fn crash_image(dir: &Path) -> (tempfile::TempDir, Database) {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy(&entry.path(), &target);
            } else if entry.file_name() != "db.lock" {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }
    let image = tempfile::tempdir().unwrap();
    copy(dir, image.path());
    let db = Database::open(&dsn(image.path())).unwrap();
    (image, db)
}

fn table(db: &Database, cluster: bool) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    if cluster {
        db.execute("ALTER TABLE t CLUSTER BY (k)", ()).unwrap();
    }
}

#[test]
fn a_delete_committed_after_the_batch_was_planned_stays_deleted() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table(&db, false);
    insert(&db, 1..=65_536, |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    small_copies(&db, 1, 2);
    assert_eq!(row_counts(&db), vec![65_536, 1, 1, 1]);
    compact_deleting(&db, 1);
    assert_eq!(row_counts(&db), vec![65_536, 3], "the small volumes merged");
    assert_eq!(values_of(&db, 1), Vec::<i64>::new(), "the delete stands");
    assert_eq!(values_of(&db, 2), vec![1]);
    assert_eq!(tombstones(&db), 1, "kept for the copy outside the batch");
    let (_image, reopened) = crash_image(dir.path());
    assert_eq!(
        values_of(&reopened, 1),
        Vec::<i64>::new(),
        "and after reopen"
    );
    let count: i64 = reopened.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_536);
    // The next compaction sees the tombstone and takes the full volume in
    for id in [1_000_002, 1_000_003] {
        insert(&db, std::iter::once(id), |id| id);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        tombstones(&db),
        0,
        "the row and its tombstone went together"
    );
    assert_eq!(values_of(&db, 1), Vec::<i64>::new());
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_538);
}

#[test]
fn a_tombstone_whose_every_copy_is_in_the_batch_is_cleaned() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table(&db, false);
    insert(&db, 1..=2, |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    small_copies(&db, 1, 2);
    assert_eq!(row_counts(&db), vec![2, 1, 1, 1]);
    compact_deleting(&db, 1);
    assert_eq!(row_counts(&db), vec![2], "one volume left");
    assert_eq!(values_of(&db, 1), Vec::<i64>::new());
    assert_eq!(tombstones(&db), 0, "the row and its tombstone went");
    let (_image, reopened) = crash_image(dir.path());
    assert_eq!(values_of(&reopened, 1), Vec::<i64>::new());
}

#[test]
fn a_volume_whose_range_covers_the_id_without_holding_it_keeps_nothing() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    table(&db, false);
    insert(&db, (1..=65_536).map(|n| n * 2), |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert(&db, std::iter::once(3), |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert(&db, std::iter::once(5), |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert(&db, std::iter::once(7), |id| id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(row_counts(&db), vec![65_536, 1, 1, 1]);
    compact_deleting(&db, 3);
    assert_eq!(row_counts(&db), vec![65_536, 2]);
    assert_eq!(values_of(&db, 3), Vec::<i64>::new());
    assert_eq!(tombstones(&db), 0, "no copy outside, the tombstone went");
}

#[test]
fn a_clustered_volume_outside_the_batch_is_read_without_its_order() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(&dsn(dir.path())).unwrap();
        table(&db, true);
        // k falls as id rises, so the volume holds its ids descending
        insert(&db, (1..=65_536).map(|n| n * 2), |id| -id);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    // Read back from disk, its order not decided yet
    let db = Database::open(&dsn(dir.path())).unwrap();
    for id in [2, 4] {
        // Through k, so no lookup by id decides the volume's order
        db.execute(&format!("UPDATE t SET v = 1 WHERE k = -{id}"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    insert(&db, std::iter::once(3), |id| -id);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(row_counts(&db), vec![65_536, 1, 1, 1]);
    let reached = Rc::new(Cell::new(false));
    let seen = Rc::clone(&reached);
    let other = db.clone();
    fp::after_compaction_planned(move || {
        seen.set(true);
        other.execute("DELETE FROM t WHERE k = -2", ()).unwrap();
        other.execute("DELETE FROM t WHERE k = -3", ()).unwrap();
    });
    db.execute("PRAGMA COMPACT_THRESHOLD = 1", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 1000", ()).unwrap();
    assert!(reached.get());
    assert_eq!(row_counts(&db).len(), 2, "the small volumes merged");
    assert_eq!(
        values_of(&db, 2),
        Vec::<i64>::new(),
        "held outside, stays deleted"
    );
    assert_eq!(values_of(&db, 3), Vec::<i64>::new());
    assert_eq!(tombstones(&db), 1, "only id 2 has a copy outside");
    let (_image, reopened) = crash_image(dir.path());
    assert_eq!(values_of(&reopened, 2), Vec::<i64>::new());
}

#[test]
fn a_volume_read_back_from_disk_holds_its_copy_by_its_metadata() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(&dsn(dir.path())).unwrap();
        table(&db, false);
        insert(&db, 1..=65_536, |id| id);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(db.engine().volume_stats()[0].2, "file", "read back lazily");
    small_copies(&db, 1, 2);
    assert_eq!(row_counts(&db), vec![65_536, 1, 1, 1]);
    compact_deleting(&db, 1);
    assert_eq!(row_counts(&db), vec![65_536, 3]);
    assert_eq!(values_of(&db, 1), Vec::<i64>::new());
    assert_eq!(tombstones(&db), 1);
    let (_image, reopened) = crash_image(dir.path());
    assert_eq!(values_of(&reopened, 1), Vec::<i64>::new());
}
