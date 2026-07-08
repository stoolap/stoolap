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

//! Regression tests for re-entrant DDL-fence acquisition deadlocking behind a
//! parked checkpoint writer (parking_lot fair RwLock). A transaction's DDL op
//! holds the fence shared until COMMIT; a later statement on the same thread
//! must acquire the fence recursively or it queues behind checkpoint's parked
//! write() and hangs forever.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use stoolap::Database;

/// BEGIN; CREATE TABLE a; (checkpoint parks on the fence); run `second_stmt`;
/// COMMIT. Fails fast via watchdog instead of hanging if the deadlock returns.
fn run_ddl_fence_scenario(second_stmt: &'static str) {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/ddl_fence", dir.path().display());
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE VIEW pre_v AS SELECT id FROM t", ())
        .unwrap();

    let (ready_tx, ready_rx) = mpsc::channel();
    let (proceed_tx, proceed_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel();

    let session_db = db.clone();
    let session = thread::spawn(move || {
        session_db.execute("BEGIN", ()).unwrap();
        session_db
            .execute("CREATE TABLE a (id INTEGER)", ())
            .unwrap();
        ready_tx.send(()).unwrap();
        proceed_rx.recv().unwrap();
        session_db.execute(second_stmt, ()).unwrap();
        session_db.execute("COMMIT", ()).unwrap();
        done_tx.send(()).unwrap();
    });

    ready_rx.recv().unwrap();

    // Cloned handle has its own executor, so PRAGMA CHECKPOINT is not
    // rejected as in-transaction; it parks on the fence write lock.
    let (ckpt_done_tx, ckpt_done_rx) = mpsc::channel();
    let ckpt_db = db.clone();
    let checkpoint = thread::spawn(move || {
        ckpt_db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        ckpt_done_tx.send(()).unwrap();
    });

    thread::sleep(Duration::from_millis(300));
    proceed_tx.send(()).unwrap();

    if done_rx.recv_timeout(Duration::from_secs(10)).is_err() {
        panic!(
            "deadlock: '{}' queued behind the parked checkpoint writer \
             while the transaction still held the DDL fence shared",
            second_stmt
        );
    }
    ckpt_done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("checkpoint did not complete after COMMIT released the DDL fence");

    session.join().unwrap();
    checkpoint.join().unwrap();
}

#[test]
fn test_truncate_after_in_txn_ddl_with_parked_checkpoint() {
    run_ddl_fence_scenario("TRUNCATE TABLE t");
}

#[test]
fn test_second_create_table_with_parked_checkpoint() {
    run_ddl_fence_scenario("CREATE TABLE b (id INTEGER)");
}

#[test]
fn test_ctas_with_parked_checkpoint() {
    run_ddl_fence_scenario("CREATE TABLE c AS SELECT id FROM t");
}

#[test]
fn test_create_view_with_parked_checkpoint() {
    run_ddl_fence_scenario("CREATE VIEW v AS SELECT id FROM t");
}

#[test]
fn test_drop_view_with_parked_checkpoint() {
    run_ddl_fence_scenario("DROP VIEW pre_v");
}
