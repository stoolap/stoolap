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
