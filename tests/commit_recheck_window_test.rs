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

//! A commit's cold recheck answers only for the layout it read through

#![cfg(feature = "test-failpoints")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::core::Error;
use stoolap::test_failpoints::{on_cold_recheck, FailpointGuard};
use stoolap::Database;

fn dsn(dir: &std::path::Path) -> String {
    format!("file://{}?checkpoint_interval=3600", dir.display())
}

/// `t(id, x, a UNIQUE, b)` with (1, 0, 10, 7) sealed, and a transaction that
/// inserted `a = 7` before (3, 0, `sealed_a`, 0) was sealed, so its commit
/// rechecks
fn recheck_pending(dir: &std::path::Path, sealed_a: i64) -> (Database, stoolap::ApiTransaction) {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER UNIQUE, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0, 10, 7)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 0, 7, 0)", ())
        .unwrap();
    db.execute(&format!("INSERT INTO t VALUES (3, 0, {sealed_a}, 0)"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    (db, txn)
}

#[test]
fn a_column_dropped_during_the_cold_recheck_is_never_a_unique_violation() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (db, mut txn) = recheck_pending(dir.path(), 30);
    let ddl_db = db.clone();
    let dropped = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&dropped);
    on_cold_recheck(move || {
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = done.send(ddl_db.execute("ALTER TABLE t DROP COLUMN x", ()));
        });
        *slot.borrow_mut() = Some(finished.recv_timeout(Duration::from_secs(10)));
    });
    let committed = txn.commit();
    let dropped = dropped.borrow_mut().take().expect("the recheck ran");
    assert!(
        matches!(dropped, Ok(Ok(_))),
        "the drop did not complete inside the recheck: {dropped:?}"
    );
    assert!(
        matches!(committed, Err(Error::SchemaChanged { .. })),
        "{committed:?}"
    );
}

#[test]
fn a_conflict_the_recheck_finds_with_no_column_change_is_a_unique_violation() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (_db, mut txn) = recheck_pending(dir.path(), 7);
    let ran = Rc::new(RefCell::new(false));
    let slot = Rc::clone(&ran);
    on_cold_recheck(move || *slot.borrow_mut() = true);
    let committed = txn.commit();
    assert!(*ran.borrow(), "the commit did not recheck");
    assert!(
        matches!(committed, Err(Error::UniqueConstraint { row_id: 3, .. })),
        "{committed:?}"
    );
}
