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

//! Statements on one handle run at once outside a transaction; one that
//! changes the session waits for them, and a program runs alone

#![cfg(feature = "test-failpoints")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use stoolap::Database;

fn table(dsn: &str) -> Arc<Database> {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    Arc::new(db)
}

/// Starts `sql` on `db` from another thread, then reports whether it
/// finished within `wait` while the caller is still inside its own statement
fn finishes_meanwhile(
    db: &Arc<Database>,
    sql: &'static str,
    wait: Duration,
    joined: &Rc<RefCell<Option<JoinHandle<()>>>>,
) -> bool {
    let db = Arc::clone(db);
    let (done, finished) = mpsc::channel();
    *joined.borrow_mut() = Some(thread::spawn(move || {
        let _ = db.execute(sql, ());
        let _ = done.send(());
    }));
    finished.recv_timeout(wait).is_ok()
}

fn join(joined: &Rc<RefCell<Option<JoinHandle<()>>>>) {
    if let Some(handle) = joined.borrow_mut().take() {
        handle.join().unwrap();
    }
}

#[test]
fn test_two_statements_on_one_handle_run_at_once() {
    let db = table("memory://gate_order_parallel");
    let joined = Rc::new(RefCell::new(None));
    let overlapped = Rc::new(Cell::new(false));
    let (hook_db, hook_joined, hook_overlapped) =
        (Arc::clone(&db), Rc::clone(&joined), Rc::clone(&overlapped));
    stoolap::test_failpoints::after_select_table_opened(move || {
        hook_overlapped.set(finishes_meanwhile(
            &hook_db,
            "UPDATE t SET v = 21 WHERE id = 2",
            Duration::from_secs(2),
            &hook_joined,
        ));
    });
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 2);
    join(&joined);
    assert!(
        overlapped.get(),
        "the second statement waited for the first"
    );
}

#[test]
fn test_begin_waits_for_a_running_statement() {
    let db = table("memory://gate_order_begin");
    let joined = Rc::new(RefCell::new(None));
    let overlapped = Rc::new(Cell::new(true));
    let (hook_db, hook_joined, hook_overlapped) =
        (Arc::clone(&db), Rc::clone(&joined), Rc::clone(&overlapped));
    stoolap::test_failpoints::after_select_table_opened(move || {
        hook_overlapped.set(finishes_meanwhile(
            &hook_db,
            "BEGIN",
            Duration::from_millis(300),
            &hook_joined,
        ));
    });
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 2);
    join(&joined);
    assert!(!overlapped.get(), "BEGIN ran inside a running statement");
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn test_no_statement_runs_between_the_statements_of_a_program() {
    let db = table("memory://gate_order_program");
    let joined = Rc::new(RefCell::new(None));
    let overlapped = Rc::new(Cell::new(true));
    let (hook_db, hook_joined, hook_overlapped) =
        (Arc::clone(&db), Rc::clone(&joined), Rc::clone(&overlapped));
    stoolap::test_failpoints::after_program_statement(move || {
        hook_overlapped.set(finishes_meanwhile(
            &hook_db,
            "ROLLBACK",
            Duration::from_millis(300),
            &hook_joined,
        ));
    });
    db.execute(
        "BEGIN; INSERT INTO t VALUES (3, 30); INSERT INTO t VALUES (4, 40); COMMIT",
        (),
    )
    .unwrap();
    join(&joined);
    assert!(!overlapped.get(), "a ROLLBACK ran inside the program");
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 4);
}
