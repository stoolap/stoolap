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

//! A row count asked while a seal counts its moved rows in both layers.

#![cfg(feature = "test-failpoints")]

use std::sync::mpsc;
use std::time::Duration;

use stoolap::storage::traits::Engine;
use stoolap::test_failpoints;
use stoolap::Database;

fn file_db(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap()
}

#[derive(Debug)]
enum Asked {
    Sql,
    RowCount,
    FastRowCount,
}

fn count(db: &Database, how: &Asked) -> i64 {
    match how {
        Asked::Sql => db.query_one("SELECT COUNT(*) FROM t", ()).unwrap(),
        Asked::RowCount | Asked::FastRowCount => {
            let mut tx = db.engine().begin_transaction().unwrap();
            let table = tx.get_table("t").unwrap();
            let n = match how {
                Asked::RowCount => table.row_count().unwrap(),
                _ => table.fast_row_count().unwrap().unwrap_or(usize::MAX),
            };
            tx.rollback().unwrap();
            n as i64
        }
    }
}

#[test]
fn a_count_asked_inside_a_seal_s_overlap_counts_every_row_once() {
    for how in [Asked::Sql, Asked::RowCount, Asked::FastRowCount] {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        for id in 1..=10i64 {
            db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        for id in 11..=20i64 {
            db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
        }
        let (reached_tx, reached_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let sealer = db.clone();
        let writer = db.clone();
        let seal = std::thread::spawn(move || {
            // Committed after the seal took its rows: they stay hot
            test_failpoints::after_side_files_built(move || {
                for id in 21..=30i64 {
                    writer
                        .execute("INSERT INTO t VALUES ($1, $1)", (id,))
                        .unwrap();
                }
            });
            test_failpoints::in_seal_after_rows_removed(move || {
                reached_tx.send(()).unwrap();
                go_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            });
            sealer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        reached_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the seal reached its hold");
        let (counted_tx, counted_rx) = mpsc::channel();
        let reader = db.clone();
        let read = std::thread::spawn(move || {
            counted_tx.send(count(&reader, &how)).unwrap();
            how
        });
        let inside = counted_rx.recv_timeout(Duration::from_millis(500)).ok();
        go_tx.send(()).unwrap();
        seal.join().unwrap();
        let how = read.join().unwrap();
        let counted = inside.unwrap_or_else(|| counted_rx.recv().unwrap());
        assert_eq!(counted, 30, "{how:?} asked inside the seal's overlap");
        assert_eq!(count(&db, &how), 30, "{how:?} once the seal is done");
    }
}
