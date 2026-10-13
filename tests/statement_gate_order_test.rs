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

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(10);

#[derive(Debug, PartialEq)]
enum Seen {
    Blocked,
    Done(Result<i64, String>),
}

fn table(dsn: &str) -> Arc<Database> {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    Arc::new(db)
}

/// Runs `sql` on `db` from another thread, which reports whether it found
/// the statement gate held and then how its statement ended
fn start(db: &Arc<Database>, sql: &'static str) -> mpsc::Receiver<Seen> {
    let db = Arc::clone(db);
    let (seen, events) = mpsc::channel();
    thread::spawn(move || {
        let blocked = seen.clone();
        stoolap::test_failpoints::on_statement_gate_blocked(move || {
            let _ = blocked.send(Seen::Blocked);
        });
        let _ = seen.send(Seen::Done(db.execute(sql, ()).map_err(|e| e.to_string())));
    });
    events
}

/// The first thing `events` reports, from inside the caller's statement
fn first(events: &mpsc::Receiver<Seen>) -> Seen {
    events
        .recv_timeout(WATCHDOG)
        .expect("the other statement neither ran nor reached the gate")
}

type Shared = Rc<RefCell<Option<(Seen, mpsc::Receiver<Seen>)>>>;

#[test]
fn test_two_statements_on_one_handle_run_at_once() {
    let db = table("memory://gate_order_parallel");
    let observed: Shared = Rc::new(RefCell::new(None));
    let (hook_db, hook_observed) = (Arc::clone(&db), Rc::clone(&observed));
    stoolap::test_failpoints::after_select_table_opened(move || {
        let events = start(&hook_db, "UPDATE t SET v = 21 WHERE id = 2");
        *hook_observed.borrow_mut() = Some((first(&events), events));
    });
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 2);
    let (seen, _) = observed.borrow_mut().take().unwrap();
    assert_eq!(seen, Seen::Done(Ok(1)), "the update waited for the scan");
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(v, 21);
}

#[test]
fn test_begin_waits_for_a_running_statement() {
    let db = table("memory://gate_order_begin");
    let observed: Shared = Rc::new(RefCell::new(None));
    let (hook_db, hook_observed) = (Arc::clone(&db), Rc::clone(&observed));
    stoolap::test_failpoints::after_select_table_opened(move || {
        let events = start(&hook_db, "BEGIN");
        *hook_observed.borrow_mut() = Some((first(&events), events));
    });
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 2);
    let (seen, events) = observed.borrow_mut().take().unwrap();
    assert_eq!(seen, Seen::Blocked, "BEGIN ran inside a running statement");
    assert_eq!(first(&events), Seen::Done(Ok(0)));
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn test_no_statement_runs_between_the_statements_of_a_program() {
    let db = table("memory://gate_order_program");
    let observed: Shared = Rc::new(RefCell::new(None));
    let (hook_db, hook_observed) = (Arc::clone(&db), Rc::clone(&observed));
    stoolap::test_failpoints::after_program_statement(move || {
        let events = start(&hook_db, "ROLLBACK");
        *hook_observed.borrow_mut() = Some((first(&events), events));
    });
    db.execute(
        "BEGIN; INSERT INTO t VALUES (3, 30); INSERT INTO t VALUES (4, 40); COMMIT",
        (),
    )
    .unwrap();
    let (seen, events) = observed.borrow_mut().take().unwrap();
    assert_eq!(seen, Seen::Blocked, "a ROLLBACK ran inside the program");
    assert!(matches!(first(&events), Seen::Done(_)));
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 4);
}
