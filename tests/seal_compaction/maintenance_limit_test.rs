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

//! Maintenance acts only on commits below the registry's maintenance limit:
//! complete, and seen by every snapshot and statement view

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(15);

fn ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

/// Runs `sql` in a transaction on another handle and stops its commit after
/// it published, before it becomes visible; `during` runs there
fn while_committing(db: &Database, sql: &'static str, during: impl FnOnce()) {
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        tx.execute(sql, ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(WATCHDOG).unwrap();
        });
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(WATCHDOG).unwrap();
    during();
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
}

fn open_file(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=2",
        dir.display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db
}

#[test]
fn vacuum_keeps_a_row_whose_delete_is_committing() {
    let db = Database::open("memory://maintenance_limit_vacuum_delete").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut seen = Vec::new();
    while_committing(&db, "DELETE FROM t WHERE id = 1", || {
        std::thread::sleep(Duration::from_millis(20));
        db.execute("VACUUM", ()).unwrap();
        seen = ids(&db);
    });
    assert_eq!(
        seen,
        vec![1],
        "VACUUM removed a row its delete had not made invisible"
    );
    assert_eq!(ids(&db), Vec::<i64>::new());
}

#[test]
fn a_seal_keeps_hot_a_row_a_view_reads_below_a_commit_it_excludes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut view = None;
    while_committing(&db, "UPDATE t SET v = 1 WHERE id = 1", || {
        view = Some(db.engine().registry().capture_view());
    });
    let store = db.engine().get_version_store("t").unwrap();
    db.engine().seal_for_test().unwrap();
    assert!(
        store.has_committed_row(1),
        "the seal moved a row whose earlier version the view reads"
    );
    drop(view);
    db.engine().seal_for_test().unwrap();
    assert!(
        !store.has_committed_row(1),
        "the row stayed hot after the view"
    );
}

#[test]
fn a_seal_keeps_hot_a_row_a_view_taken_after_its_cutoff_reads() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let view = std::rc::Rc::new(std::cell::RefCell::new(None));
    let (writer, taken) = (db.clone(), std::rc::Rc::clone(&view));
    stoolap::test_failpoints::after_seal_cutoff_read(move || {
        while_committing(&writer, "UPDATE t SET v = 1 WHERE id = 1", || {
            *taken.borrow_mut() = Some(writer.engine().registry().capture_view());
        });
    });
    let store = db.engine().get_version_store("t").unwrap();
    db.engine().seal_for_test().unwrap();
    assert!(view.borrow().is_some(), "the seal read its cutoff");
    assert!(
        store.has_committed_row(1),
        "the seal moved a row whose earlier version a view taken after its cutoff reads"
    );
    view.borrow_mut().take();
    db.engine().seal_for_test().unwrap();
    assert!(
        !store.has_committed_row(1),
        "the row stayed hot after the view"
    );
}

#[test]
fn a_compaction_keeps_a_row_whose_delete_a_view_excludes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    let seal_volume = |first: i64| {
        let values: Vec<String> = (first..first + 100)
            .map(|id| format!("({id}, 0)"))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.engine().seal_for_test().unwrap();
    };
    seal_volume(1);
    seal_volume(101);
    let mut view = None;
    while_committing(&db, "DELETE FROM t WHERE id = 1", || {
        view = Some(db.engine().registry().capture_view());
    });
    let applied = || {
        let (sent, received) = mpsc::channel();
        stoolap::test_failpoints::after_compaction_dedup(move |pairs| sent.send(pairs).unwrap());
        db.engine().compact_for_test().unwrap();
        received.try_recv().unwrap_or(0)
    };
    assert_eq!(
        applied(),
        0,
        "the compaction applied a delete the view excludes"
    );
    drop(view);
    seal_volume(201);
    assert_eq!(applied(), 1, "the delete applies after the view");
}
