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

    let (ready, at_point) = mpsc::channel::<&str>();
    let (done, finished) = mpsc::channel::<&str>();
    let (go_check, check_go) = mpsc::channel::<()>();
    let (go_commit, commit_go) = mpsc::channel::<()>();

    let (ready_check, done_check) = (ready.clone(), done.clone());
    std::thread::spawn(move || {
        stoolap::test_failpoints::after_unique_index_found(move || {
            ready_check.send("check").unwrap();
            check_go.recv().unwrap();
        });
        let _ = checking.insert(row(2, 2, 2));
        done_check.send("check").unwrap();
    });
    let (ready_commit, done_commit) = (ready, done.clone());
    std::thread::spawn(move || {
        stoolap::test_failpoints::before_commit_index_capture(move || {
            ready_commit.send("commit").unwrap();
            commit_go.recv().unwrap();
        });
        let _ = tx.commit();
        done_commit.send("commit").unwrap();
    });
    for _ in 0..2 {
        at_point
            .recv_timeout(Duration::from_secs(10))
            .expect("the check and the commit reach their points");
    }
    let indexing = db.clone();
    std::thread::spawn(move || {
        let _ = indexing.execute("CREATE INDEX ik ON t(k)", ());
        done.send("index").unwrap();
    });
    // The index change queues for the index map behind the check
    std::thread::sleep(Duration::from_millis(300));
    go_check.send(()).unwrap();
    go_commit.send(()).unwrap();
    for _ in 0..3 {
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("the check, the commit and the index change finish");
    }
}
