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

//! A recovery seals the log it replayed once: opening the database it left
//! again, after another crash, replays nothing it already sealed

use std::path::Path;

use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
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

fn volumes(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("volumes").join("t"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "vol")
        })
        .count()
}

fn sum(db: &Database) -> (i64, i64) {
    let row = db
        .query("SELECT COUNT(*), SUM(n) FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    (row.get(0).unwrap(), row.get(1).unwrap())
}

/// A sealed table with an UPDATE and a DELETE left in the WAL, crashed into
/// `root/1`; the rows it holds
fn crashed(root: &Path) -> (i64, i64) {
    let first = root.join("0");
    let db = Database::open(&dsn(&first)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
        "INSERT INTO t SELECT value, value FROM generate_series(1, 1000)",
        "PRAGMA CHECKPOINT",
        "UPDATE t SET n = n + 1 WHERE id % 10 = 0",
        "DELETE FROM t WHERE id % 10 = 1",
    ] {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    let want = sum(&db);
    crash_image(&first, &root.join("1"));
    db.close().unwrap();
    want
}

#[test]
fn a_database_reopened_after_crash_after_crash_seals_its_log_once() {
    // The failpoint tests here fail WAL syncs process-wide
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let root = tempfile::tempdir().unwrap();
    let want = crashed(root.path());
    let mut counts = Vec::new();
    for crash in 1..=3 {
        let dir = root.path().join(crash.to_string());
        let db = Database::open(&dsn(&dir)).unwrap();
        assert_eq!(sum(&db), want, "rows after crash {crash}");
        counts.push(volumes(&dir));
        crash_image(&dir, &root.path().join((crash + 1).to_string()));
        db.close().unwrap();
    }
    assert!(
        counts.windows(2).all(|w| w[0] == w[1]),
        "volumes after each recovery {counts:?}: a recovery sealed again what the one before sealed"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_recovery_whose_checkpoint_poisons_the_wal_fails_the_open() {
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;
    let _guard = test_failpoints::FailpointGuard::new();
    let root = tempfile::tempdir().unwrap();
    let want = crashed(root.path());
    let dir = root.path().join("1");
    test_failpoints::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::Release);
    let opened = Database::open(&dsn(&dir));
    test_failpoints::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(
        opened.is_err(),
        "an open whose WAL refuses every write fails: {:?}",
        opened.as_ref().err()
    );
    drop(opened);
    let db = Database::open(&dsn(&dir)).unwrap();
    assert_eq!(sum(&db), want, "the next open recovers the rows");
    db.close().unwrap();
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_recovery_whose_wal_rotation_poisons_the_wal_fails_the_open_when_no_truncation_follows() {
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;
    let _guard = test_failpoints::FailpointGuard::new();
    let root = tempfile::tempdir().unwrap();
    crashed(root.path());
    let dir = root.path().join("1");
    // The catalog copies rotate the WAL past its size, the rotation's sync of
    // the retired file fails, and a failed manifest write skips the truncation
    test_failpoints::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::Release);
    test_failpoints::MANIFEST_WRITE_FAIL.store(true, Ordering::Release);
    let opened = Database::open(&format!("{}&wal_max_size=1", dsn(&dir)));
    test_failpoints::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::Release);
    test_failpoints::MANIFEST_WRITE_FAIL.store(false, Ordering::Release);
    assert!(
        opened.is_err(),
        "an open whose WAL refuses every write fails: {:?}",
        opened.as_ref().err()
    );
}
