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

//! A COMMIT that fails takes back the tables its transaction created or
//! dropped, as a ROLLBACK does, now and after a crash

#![cfg(feature = "test-failpoints")]

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

fn dsn(path: &Path) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_interval=3600",
        path.display()
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

fn exists(db: &Database, table: &str) -> bool {
    db.query_one::<i64, _>(&format!("SELECT COUNT(*) FROM {table}"), ())
        .is_ok()
}

/// A COMMIT that a column change of t, begun once the commit is counted,
/// refuses before its marker
fn commit_refused_by_a_change(db: &Database) -> stoolap::core::Result<i64> {
    let other = db.clone();
    let change = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&change);
    fp::after_commit_counted(move || {
        let (reached, waiting) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            fp::on_column_change_waiting(move || reached.send(()).unwrap());
            other.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        *slot.borrow_mut() = Some(handle);
    });
    let committed = db.execute("COMMIT", ());
    let changed = change.borrow_mut().take().unwrap().join().unwrap();
    assert!(changed.is_ok(), "{changed:?}");
    committed
}

/// A COMMIT whose marker cannot be made durable
fn commit_without_marker(db: &Database) -> stoolap::core::Result<i64> {
    fp::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let committed = db.execute("COMMIT", ());
    fp::WAL_SYNC_FAIL.store(false, Ordering::Release);
    committed
}

#[test]
fn a_failed_commit_takes_back_the_table_it_created() {
    let _guard = FailpointGuard::new();
    let mut wrong = Vec::new();
    for (failure, fail) in [
        ("change", commit_refused_by_a_change as fn(&Database) -> _),
        ("marker", commit_without_marker),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let image = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
        db.execute("BEGIN", ()).unwrap();
        db.execute("CREATE TABLE n (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
        assert!(fail(&db).is_err(), "{failure}: the commit failed");
        if exists(&db, "n") {
            wrong.push(format!("{failure}: n stayed"));
        }
        if failure == "change" {
            if db.execute("INSERT INTO n VALUES (1)", ()).is_ok() {
                wrong.push(format!("{failure}: an insert into n committed"));
            }
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        crash_image(dir.path(), image.path());
        drop(db);
        let db = Database::open(&dsn(image.path())).unwrap();
        if exists(&db, "n") {
            wrong.push(format!("{failure}: n came back"));
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_failed_commit_brings_back_the_table_it_dropped() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let image = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE x (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("DROP TABLE x", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 8)", ()).unwrap();
    assert!(commit_refused_by_a_change(&db).is_err());
    assert!(exists(&db, "x"), "x was not brought back");
    crash_image(dir.path(), image.path());
    drop(db);
    let db = Database::open(&dsn(image.path())).unwrap();
    assert!(exists(&db, "x"), "x was gone after a crash");
    db.close().unwrap();
}
