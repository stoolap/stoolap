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

//! A checkpoint's WAL truncation starts a new file and leaves the old one
//! to the next truncation that covers it; nothing is copied.

use std::path::Path;
use stoolap::Database;
use tempfile::tempdir;

fn wal_files(dir: &Path) -> Vec<(String, u64)> {
    let mut files: Vec<(String, u64)> = std::fs::read_dir(dir.join("wal"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name.starts_with("wal-") || name.starts_with("wal_")) && name.ends_with(".log")
        })
        .map(|e| {
            // Through a handle: a directory listing reports a stale length
            // for a file that is open for writing on Windows
            let len = std::fs::File::open(e.path())
                .unwrap()
                .metadata()
                .unwrap()
                .len();
            (e.file_name().to_string_lossy().to_string(), len)
        })
        .collect();
    files.sort();
    files
}

fn lsn_in(name: &str) -> u64 {
    let at = name.find("lsn-").unwrap() + 4;
    name[at..name[at..].find('.').unwrap() + at]
        .parse()
        .unwrap()
}

#[test]
fn a_checkpoint_under_writes_leaves_the_old_file_until_the_next_one_covers_it() {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    for i in 1..=20 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'r{}')", i, i), ())
            .unwrap();
    }
    let before = wal_files(dir.path());
    assert_eq!(before.len(), 1);

    // The first checkpoint truncates: the old file stays whole, a new one
    // starts at the last record the old one holds
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let after = wal_files(dir.path());
    assert_eq!(after.len(), 2, "{:?}", after);
    let (old, new) = if after[0].0 == before[0].0 {
        (&after[0], &after[1])
    } else {
        (&after[1], &after[0])
    };
    // The old file grew by the checkpoint's catalog copies only, appended
    // above the boundary before the truncation; the new file is empty
    assert!(old.1 >= before[0].1, "{:?} vs {:?}", old, before[0]);
    assert_eq!(new.1, 0);
    assert!(lsn_in(&new.0) > lsn_in(&old.0));

    // Writes after it land in the new file
    for i in 21..=25 {
        db.execute(&format!("INSERT INTO t VALUES ({}, 'r{}')", i, i), ())
            .unwrap();
    }
    let written = wal_files(dir.path());
    assert_eq!(written.len(), 2);
    assert!(written.iter().any(|f| f.0 == new.0 && f.1 > new.1));

    // The next checkpoint covers the old file and removes it
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let next = wal_files(dir.path());
    assert!(
        next.iter().all(|f| f.0 != old.0),
        "the old file goes once covered: {:?}",
        next
    );

    // A checkpoint with nothing new starts no file; its catalog copies go
    // into the current one, and the file it covers is removed
    let current = next.iter().max_by_key(|f| lsn_in(&f.0)).unwrap().0.clone();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let idle = wal_files(dir.path());
    assert_eq!(idle.len(), 1, "{:?}", idle);
    assert_eq!(idle[0].0, current);

    let _ = db.close();
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 25);
}

#[test]
fn rows_committed_across_checkpoints_are_recovered_from_both_files() {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // The new file starts empty; this row is its first record and the old
    // file's tail above the boundary is empty
    db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();
    assert_eq!(wal_files(dir.path()).len(), 2);

    let _ = db.close();
    let db = Database::open(&dsn).unwrap();
    let ids: Vec<i64> = db
        .query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();
    let _ = db.close();
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 3);
}
