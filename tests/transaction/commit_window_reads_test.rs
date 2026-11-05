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

//! A read committed read while another handle's commit publishes sees the
//! state before that commit, in the hot store and in the volumes

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(15);

const QUERIES: &[&str] = &[
    "SELECT id, v FROM t ORDER BY id",
    "SELECT v FROM t WHERE id = 1",
    "SELECT id FROM t WHERE v >= 10 ORDER BY id",
    "SELECT COUNT(*) FROM t",
    "SELECT COUNT(v) FROM t",
    "SELECT SUM(v) FROM t",
    "SELECT MIN(v) FROM t",
    "SELECT MAX(v) FROM t",
    "SELECT v, COUNT(*) FROM t GROUP BY v ORDER BY v",
];

fn answers(db: &Database) -> Vec<String> {
    QUERIES
        .iter()
        .map(|sql| {
            let rows: Vec<String> = db
                .query(sql, ())
                .unwrap()
                .map(|row| {
                    let row = row.unwrap();
                    (0..row.len())
                        .map(|i| format!("{:?}", row.get_value(i)))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            format!("{sql} -> {}", rows.join(" | "))
        })
        .collect()
}

enum Stop {
    BeforeMarker,
    BeforeVisible,
}

/// The answers while `sql` commits on another handle, stopped at `stop`
fn answers_while_committing(db: &Database, sql: &'static str, stop: Stop) -> Vec<String> {
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        tx.execute(sql, ()).unwrap();
        let hook = move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(WATCHDOG).unwrap();
        };
        match stop {
            Stop::BeforeMarker => stoolap::test_failpoints::before_commit_marker(hook),
            Stop::BeforeVisible => stoolap::test_failpoints::before_commit_visible(hook),
        }
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(WATCHDOG).unwrap();
    let seen = answers(db);
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
    seen
}

fn table(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
}

fn open_file(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

/// The rows are sealed before each commit when `sealed`
fn check(db: &Database, sealed: bool) {
    let mut wrong = Vec::new();
    for sql in [
        "UPDATE t SET v = 99 WHERE id = 1",
        "DELETE FROM t WHERE id = 1",
    ] {
        for (stop, name) in [
            (Stop::BeforeMarker, "marker"),
            (Stop::BeforeVisible, "visible"),
        ] {
            if sealed {
                db.engine().seal_for_test().unwrap();
            }
            let before = answers(db);
            let during = answers_while_committing(db, sql, stop);
            for (seen, expected) in during.iter().zip(&before) {
                if seen != expected {
                    wrong.push(format!(
                        "{sql}, before the {name}: {seen}, expected {expected}"
                    ));
                }
            }
            db.execute("DELETE FROM t", ()).unwrap();
            db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
                .unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_hot_read_during_a_commit_sees_the_state_before_it() {
    let db = Database::open("memory://commit_window_reads_hot").unwrap();
    table(&db);
    check(&db, false);
}

#[test]
fn a_cold_read_during_a_commit_sees_the_state_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    table(&db);
    check(&db, true);
}
