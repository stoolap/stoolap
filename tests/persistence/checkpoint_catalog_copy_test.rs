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

//! A checkpoint's catalog copies, replayed after a crash that left the
//! manifests behind them, keep the rows the replay put back before them

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::atomic::Ordering;

use stoolap::test_failpoints;
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

/// A sealed, indexed table with an UPDATE, a DELETE and new rows since
fn written(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
        "CREATE INDEX t_n ON t (n)",
        "INSERT INTO t SELECT value, value FROM generate_series(1, 1000)",
        "PRAGMA CHECKPOINT",
        "UPDATE t SET n = n + 1 WHERE id % 10 = 0",
        "DELETE FROM t WHERE id % 10 = 1",
        "INSERT INTO t SELECT value, value FROM generate_series(1001, 1010)",
    ] {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    db
}

fn assert_rows(dir: &Path) {
    let db = Database::open(&dsn(dir)).unwrap();
    let row = db
        .query("SELECT COUNT(*), SUM(n) FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let rows: (i64, i64) = (row.get(0).unwrap(), row.get(1).unwrap());
    assert_eq!(rows, (910, 461055), "rows after the crash");
    let ids: Vec<i64> = db
        .query("SELECT id FROM t WHERE n IN (11, 1005) ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![10, 1005], "rows the index finds after the crash");
    db.close().unwrap();
}

#[test]
fn a_crash_after_the_catalog_copies_before_the_manifests_keeps_the_rows() {
    let _guard = test_failpoints::FailpointGuard::new();
    let root = tempfile::tempdir().unwrap();
    let (first, crashed) = (root.path().join("0"), root.path().join("1"));
    let db = written(&first);
    let (from, to) = (first.clone(), crashed.clone());
    test_failpoints::after_checkpoint_managers_taken(move || crash_image(&from, &to));
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    assert_rows(&crashed);
}

#[test]
fn a_crash_after_a_checkpoint_whose_manifests_failed_keeps_the_rows() {
    let _guard = test_failpoints::FailpointGuard::new();
    let root = tempfile::tempdir().unwrap();
    let (first, crashed) = (root.path().join("0"), root.path().join("1"));
    let db = written(&first);
    test_failpoints::MANIFEST_WRITE_FAIL.store(true, Ordering::Release);
    let checkpoint = db.execute("PRAGMA CHECKPOINT", ());
    test_failpoints::MANIFEST_WRITE_FAIL.store(false, Ordering::Release);
    drop(checkpoint);
    crash_image(&first, &crashed);
    db.close().unwrap();
    assert_rows(&crashed);
}
