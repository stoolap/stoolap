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

//! A unique check reads the transaction's own rows without holding the
//! table's index map, so a commit of the same transaction on another
//! thread and an index change queued between them all finish.

#![cfg(feature = "test-failpoints")]

use std::sync::mpsc;
use std::time::Duration;

use stoolap::storage::traits::Engine;
use stoolap::{Database, Row, Value};

fn row(id: i64, k: i64, u: i64) -> Row {
    Row::from_values(vec![
        Value::Integer(id),
        Value::Integer(k),
        Value::Integer(u),
    ])
}

#[test]
fn a_unique_check_a_commit_and_an_index_change_all_finish() {
    let db = Database::open("memory://unique_check_lock_order").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, u INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    tx.get_table("t").unwrap().insert(row(1, 1, 1)).unwrap();
    let mut checking = tx.get_table("t").unwrap();

    let limit = Duration::from_secs(10);
    let (ready, at_point) = mpsc::channel::<()>();
    let (done, finished) = mpsc::channel::<()>();
    let (go_check, check_go) = mpsc::channel::<()>();
    let (go_commit, commit_go) = mpsc::channel::<()>();

    let (ready_check, done_check) = (ready.clone(), done.clone());
    std::thread::spawn(move || {
        stoolap::test_failpoints::after_unique_index_found(move || {
            ready_check.send(()).unwrap();
            check_go.recv().unwrap();
        });
        let _ = checking.insert(row(2, 2, 2));
        done_check.send(()).unwrap();
    });
    at_point
        .recv_timeout(limit)
        .expect("the check reaches its point");
    std::thread::spawn(move || {
        stoolap::test_failpoints::before_commit_index_capture(move || {
            ready.send(()).unwrap();
            commit_go.recv().unwrap();
        });
        let _ = tx.commit();
        done.send(()).unwrap();
    });
    at_point
        .recv_timeout(limit)
        .expect("the commit reaches its point");

    let store = db.engine().get_version_store("t").unwrap();
    let (indexed, index_done) = mpsc::channel::<Result<(), String>>();
    let indexing = db.clone();
    std::thread::spawn(move || {
        let created = indexing.execute("CREATE INDEX ik ON t(k)", ());
        indexed
            .send(created.map(|_| ()).map_err(|e| format!("{e:?}")))
            .unwrap();
    });
    // The index change either queues for the index map, behind a check
    // that holds it, or takes the map and is done
    let deadline = std::time::Instant::now() + limit;
    let mut created = None;
    while created.is_none() && !store.index_map_writer_queued() {
        created = index_done.try_recv().ok();
        assert!(
            std::time::Instant::now() < deadline,
            "the index change neither queued nor finished"
        );
        std::thread::yield_now();
    }
    go_check.send(()).unwrap();
    go_commit.send(()).unwrap();
    for _ in 0..2 {
        finished
            .recv_timeout(limit)
            .expect("the check and the commit finish");
    }
    let created = match created {
        Some(created) => created,
        None => index_done
            .recv_timeout(limit)
            .expect("the index change finishes"),
    };
    created.unwrap();
}
