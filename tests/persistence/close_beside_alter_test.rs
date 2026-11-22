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

//! A close from another thread waits for a DDL statement in flight, holding
//! no lock an open needs, so the next open finds what the statement answered

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use stoolap::core::Value;
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

// One test runs at a time
static SERIAL: Mutex<()> = Mutex::new(());

/// The handle another thread closes while the CHECK judges
static CLOSED_BESIDE: Mutex<Option<Database>> = Mutex::new(None);
/// The closing thread, once the CHECK's function has started it
static CLOSER: Mutex<Option<JoinHandle<stoolap::Result<()>>>> = Mutex::new(None);
/// The thread that opens another database from inside the statement
static OPENER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// Whether the function lets the row pass
static PASSES: AtomicBool = AtomicBool::new(true);
/// Whether the close was seen waiting for the DDL guard before the function
/// went on, and whether an open of another database then returned
static WAITED: AtomicBool = AtomicBool::new(false);
static OPENED: AtomicBool = AtomicBool::new(false);

const PATIENCE: Duration = Duration::from_secs(20);

/// Starts a close on another thread, goes on once that thread waits for the
/// DDL guard, opens another database, then judges
#[derive(Default)]
struct ClosedBeside;

impl ScalarFunction for ClosedBeside {
    fn name(&self) -> &str {
        "ALTER_CHECK_CLOSED_BESIDE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ALTER_CHECK_CLOSED_BESIDE",
            FunctionType::Scalar,
            "starts a close on another thread, then judges its row",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, _args: &[Value]) -> stoolap::core::Result<Value> {
        let handle = CLOSED_BESIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(db) = handle {
            let (waiting, waits) = channel();
            let closer = std::thread::spawn(move || {
                stoolap::test_failpoints::on_ddl_guard_waiting(move || {
                    let _ = waiting.send(());
                });
                db.close()
            });
            *CLOSER.lock().unwrap_or_else(|e| e.into_inner()) = Some(closer);
            WAITED.store(waits.recv_timeout(PATIENCE).is_ok(), Ordering::SeqCst);
            let (opened, opens) = channel();
            let opener = std::thread::spawn(move || {
                let other = Database::open("memory://close_beside_alter_other");
                let _ = opened.send(other.is_ok());
            });
            *OPENER.lock().unwrap_or_else(|e| e.into_inner()) = Some(opener);
            OPENED.store(
                opens.recv_timeout(PATIENCE).unwrap_or(false),
                Ordering::SeqCst,
            );
        }
        Ok(Value::Boolean(PASSES.load(Ordering::SeqCst)))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ClosedBeside)
    }
}

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// Runs `alter` over a sealed table while another thread closes the
/// database, and returns whether the close waited for the statement, an
/// open inside the statement returned, the statement was taken, and the
/// table's definition at the next open
fn reopened_after(alter: &str, passes: bool) -> (bool, bool, bool, String) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = global_registry().register_scalar::<ClosedBeside>();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 5), (2, 50)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    PASSES.store(passes, Ordering::SeqCst);
    WAITED.store(false, Ordering::SeqCst);
    OPENED.store(false, Ordering::SeqCst);
    *CLOSED_BESIDE.lock().unwrap_or_else(|e| e.into_inner()) = Some(db.clone());
    let taken = db.execute(alter, ()).is_ok();
    let closer = CLOSER.lock().unwrap_or_else(|e| e.into_inner()).take();
    closer
        .expect("the CHECK's function never ran")
        .join()
        .unwrap()
        .unwrap();
    let opener = OPENER.lock().unwrap_or_else(|e| e.into_inner()).take();
    opener.expect("the opener never started").join().unwrap();
    drop(db);

    let reopened = Database::open(&dsn(dir.path())).unwrap();
    let definition = reopened
        .query("SHOW CREATE TABLE t", ())
        .unwrap()
        .map(|row| row.unwrap().get_value(1).unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    (
        WAITED.load(Ordering::SeqCst),
        OPENED.load(Ordering::SeqCst),
        taken,
        definition,
    )
}

const ADD: &str =
    "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 4 CHECK (ALTER_CHECK_CLOSED_BESIDE(z))";

#[test]
fn a_column_refused_while_another_thread_closes_does_not_come_back() {
    assert_eq!(
        reopened_after(ADD, false),
        (
            true,
            true,
            false,
            "CREATE TABLE t (id Integer PRIMARY KEY, v Integer)".to_string()
        )
    );
}

#[test]
fn a_column_added_while_another_thread_closes_is_there_at_the_next_open() {
    assert_eq!(
        reopened_after(ADD, true),
        (
            true,
            true,
            true,
            "CREATE TABLE t (id Integer PRIMARY KEY, v Integer, z Integer DEFAULT 4 CHECK (ALTER_CHECK_CLOSED_BESIDE(z)))".to_string()
        )
    );
}

#[test]
fn a_second_close_returns_once_the_first_is_done() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2)", ()).unwrap();

    let (began, begins) = channel();
    let (release, released) = channel::<()>();
    let first = db.clone();
    let first = std::thread::spawn(move || {
        stoolap::test_failpoints::after_close_began(move || {
            let _ = began.send(());
            let _ = released.recv_timeout(PATIENCE);
        });
        first.close()
    });
    begins
        .recv_timeout(PATIENCE)
        .expect("the first close began");

    let (done, dones) = channel();
    let second = db.clone();
    let second = std::thread::spawn(move || {
        let closed = second.close();
        let _ = done.send(());
        closed
    });
    let early = dones.recv_timeout(Duration::from_millis(500)).is_ok();
    release.send(()).unwrap();
    first.join().unwrap().unwrap();
    second.join().unwrap().unwrap();
    assert!(
        !early,
        "the second close returned while the first had only begun"
    );

    drop(db);
    let reopened = Database::open(&dsn(dir.path())).unwrap();
    let rows: i64 = reopened.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, 2);
}

#[test]
fn a_close_forgets_no_database_opened_under_its_dsn_since() {
    let dsn = "memory://close_beside_alter_reopened";
    let closed = Database::open(dsn).unwrap();
    let late = closed.clone();
    closed.close().unwrap();
    let opened = Database::open(dsn).unwrap();
    opened
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    late.close().unwrap();
    let again = Database::open(dsn).unwrap();
    again
        .execute("INSERT INTO t VALUES (1)", ())
        .expect("the database opened since the close is the one the DSN names");
}
