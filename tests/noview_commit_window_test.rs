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

//! A read without a statement view, and a DML statement's target, while a
//! commit that changes a sealed row publishes: the row is seen once, and a
//! statement that would change it fails instead of passing over it

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::storage::traits::Engine;
use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(15);

const WRITERS: &[&str] = &[
    "UPDATE c SET v = 11 WHERE id = 1",
    "DELETE FROM c WHERE id = 1",
];

#[derive(Clone, Copy, Debug)]
enum Stop {
    BeforeMarker,
    BeforeVisible,
}

/// Row 1 sealed, row 2 hot
fn open(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER, f TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 10, 'old')", ())
        .unwrap();
    db.engine().seal_for_test().unwrap();
    db.execute("INSERT INTO c VALUES (2, 20, 'old')", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> Vec<(i64, i64, String)> {
    db.query("SELECT id, v, f FROM c ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<i64>(1).unwrap(),
                row.get::<String>(2).unwrap(),
            )
        })
        .collect()
}

/// `during`'s answer while `writer` commits on another handle, held at `stop`
fn while_committing<R>(
    db: &Database,
    writer: &'static str,
    stop: Stop,
    during: impl FnOnce() -> R,
) -> R {
    let (held_tx, held_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let handle = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = handle.begin().unwrap();
        tx.execute(writer, ()).unwrap();
        let hook = move || {
            held_tx.send(()).unwrap();
            go_rx.recv_timeout(WATCHDOG).unwrap();
        };
        match stop {
            Stop::BeforeMarker => stoolap::test_failpoints::before_commit_marker(hook),
            Stop::BeforeVisible => stoolap::test_failpoints::before_commit_visible(hook),
        }
        tx.commit().unwrap();
    });
    held_rx.recv_timeout(WATCHDOG).unwrap();
    let answer = during();
    go_tx.send(()).unwrap();
    commit.join().unwrap();
    answer
}

#[test]
fn a_storage_read_while_a_sealed_row_s_change_publishes_sees_the_table_before_it() {
    let mut wrong = Vec::new();
    for &writer in WRITERS {
        for stop in [Stop::BeforeMarker, Stop::BeforeVisible] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(dir.path());
            let tx = db.engine().begin_transaction().unwrap();
            let table = tx.get_table("c").unwrap();
            let seen = while_committing(&db, writer, stop, || {
                let mut ids: Vec<i64> = table
                    .collect_all_rows(None)
                    .unwrap()
                    .iter()
                    .map(|(id, _)| *id)
                    .collect();
                ids.sort_unstable();
                let mut active = table.get_active_row_ids().unwrap();
                active.sort_unstable();
                (
                    table.row_count().unwrap(),
                    ids,
                    table.collect_all_rows_unsorted().unwrap().len(),
                    table.collect_rows_with_limit(None, 10, 0).unwrap().len(),
                    active,
                )
            });
            let before = (2, vec![1, 2], 2, 2, vec![1, 2]);
            if seen != before {
                wrong.push(format!("{writer}, held {stop:?}: {seen:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_storage_read_a_commit_completes_inside_sees_the_sealed_row_once() {
    for &writer in WRITERS {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        let tx = db.engine().begin_transaction().unwrap();
        let table = tx.get_table("c").unwrap();
        let (held_tx, held_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let handle = db.clone();
        let commit = std::thread::spawn(move || {
            let mut tx = handle.begin().unwrap();
            tx.execute(writer, ()).unwrap();
            stoolap::test_failpoints::before_commit_visible(move || {
                held_tx.send(()).unwrap();
                go_rx.recv_timeout(WATCHDOG).unwrap();
            });
            tx.commit().unwrap();
            done_tx.send(()).unwrap();
        });
        held_rx.recv_timeout(WATCHDOG).unwrap();
        // The hot rows are read with the commit held; it completes before
        // the volumes are read
        stoolap::test_failpoints::after_merged_read_collected(move || {
            go_tx.send(()).unwrap();
            done_rx.recv_timeout(WATCHDOG).unwrap();
        });
        let mut ids: Vec<i64> = table
            .collect_all_rows(None)
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        commit.join().unwrap();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2], "{writer}: row 1 in neither layer");
    }
}

#[test]
fn a_statement_that_would_change_a_sealed_row_another_commit_publishes_fails() {
    let mut wrong = Vec::new();
    for &writer in WRITERS {
        for stop in [Stop::BeforeMarker, Stop::BeforeVisible] {
            for dml in [
                "DELETE FROM c",
                "DELETE FROM c WHERE id = 1",
                "DELETE FROM c WHERE v > 0",
                "UPDATE c SET f = 'new'",
                "UPDATE c SET f = 'new' WHERE id = 1",
                "UPDATE c SET f = 'new' WHERE v > 0",
            ] {
                let dir = tempfile::tempdir().unwrap();
                let db = open(dir.path());
                let other = db.clone();
                let answer = while_committing(&db, writer, stop, || {
                    other.execute(dml, ()).map_err(|e| e.to_string())
                });
                let committed = rows(&db);
                let expected: Vec<(i64, i64, String)> = if writer.starts_with("UPDATE") {
                    vec![(1, 11, "old".into()), (2, 20, "old".into())]
                } else {
                    vec![(2, 20, "old".into())]
                };
                let refused = answer.as_ref().is_err_and(|e| e.contains("write conflict"));
                if !refused || committed != expected {
                    wrong.push(format!(
                        "{dml} while {writer} held {stop:?}: {answer:?}, then {committed:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
