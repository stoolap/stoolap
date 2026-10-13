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

//! A statement that runs others, a half-read result and a second thread on a
//! session with an open transaction all finish under the statement gate

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use stoolap::Database;

fn table(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db
}

/// Runs `body` on its own thread and fails if it does not finish in time
fn finishes(body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        body();
        let _ = done.send(());
    });
    assert!(
        finished.recv_timeout(Duration::from_secs(10)).is_ok(),
        "the session deadlocked"
    );
}

#[test]
fn test_explain_analyze_and_analyze_inside_a_transaction() {
    finishes(|| {
        let db = table("memory://gate_explain_in_tx");
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        let plan: Vec<_> = db
            .query("EXPLAIN ANALYZE SELECT * FROM t", ())
            .unwrap()
            .collect();
        assert!(!plan.is_empty());
        db.execute("ANALYZE t", ()).unwrap();
        db.execute("ROLLBACK", ()).unwrap();
        db.execute("ANALYZE t", ()).unwrap();
        db.query("EXPLAIN ANALYZE BEGIN", ()).unwrap().count();
        db.execute("ROLLBACK", ()).unwrap();
    });
}

#[test]
fn test_a_half_read_result_does_not_hold_the_session() {
    finishes(|| {
        let db = table("memory://gate_half_read");
        let mut rows = db.query("SELECT * FROM t", ()).unwrap();
        assert!(rows.next().is_some());
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        assert!(rows.next().is_some());
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 3);
    });
}

#[test]
fn test_another_thread_runs_inside_the_open_transaction() {
    let db = table("memory://gate_shared_tx");
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    let seen = thread::scope(|s| {
        s.spawn(|| db.query("SELECT * FROM t", ()).unwrap().count())
            .join()
            .unwrap()
    });
    assert_eq!(
        seen, 3,
        "the other thread read outside the session's transaction"
    );
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(db.query("SELECT * FROM t", ()).unwrap().count(), 2);
}
