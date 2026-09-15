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

//! DML statements prepare their cold reads outside the seal fence and take
//! the fence only to apply; a seal that lands in between makes the
//! statement prepare again. Under seals racing point and batch updates,
//! inserts and deletes, no update is lost, no row doubles and every
//! constraint holds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use stoolap::Database;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?target_volume_rows=65536&compact_threshold=2",
        dir.display()
    ))
    .unwrap()
}

#[test]
fn dml_on_cold_rows_stays_correct_while_seals_race_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX uk ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT g.value, g.value, 0 FROM generate_series(1, 20000) g",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Seals keep landing while the statements run
    let stop = Arc::new(AtomicBool::new(false));
    let sealer = {
        let db = db.clone();
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut seals = 0;
            while !stop.load(Ordering::Acquire) {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                seals += 1;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            seals
        })
    };
    // Point updates of cold rows, batch updates over a cold range,
    // inserts that must find the unique key in cold rows, deletes
    for round in 0..40i64 {
        let id = 1 + round * 400;
        assert_eq!(
            db.execute(&format!("UPDATE t SET v = v + 1 WHERE id = {id}"), ())
                .unwrap(),
            1
        );
        assert_eq!(
            db.execute(
                &format!(
                    "UPDATE t SET v = v + 1 WHERE id > {} AND id <= {}",
                    id + 1,
                    id + 5
                ),
                ()
            )
            .unwrap(),
            4
        );
        // The key is taken by a cold row: refused
        assert!(db
            .execute(
                &format!("INSERT INTO t VALUES ({}, {id}, 0)", 100_000 + round),
                ()
            )
            .is_err());
        // A fresh key: accepted
        assert_eq!(
            db.execute(
                &format!(
                    "INSERT INTO t VALUES ({}, {}, 0)",
                    100_000 + round,
                    100_000 + round
                ),
                ()
            )
            .unwrap(),
            1
        );
        assert_eq!(
            db.execute(&format!("DELETE FROM t WHERE id = {}", id + 6), ())
                .unwrap(),
            1
        );
    }
    stop.store(true, Ordering::Release);
    let seals = sealer.join().unwrap();
    assert!(seals > 0);

    // Every statement's effect, exactly once
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 20_000 + 40 - 40);
    let bumped: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE v = 1", ())
        .unwrap();
    assert_eq!(bumped, 40 * 5);
    let untouched: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE v = 0", ())
        .unwrap();
    assert_eq!(untouched, 20_000 - 40 * 6 + 40);
    let keys: i64 = db.query_one("SELECT COUNT(DISTINCT k) FROM t", ()).unwrap();
    assert_eq!(keys, count);
    let gone: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE id % 400 = 7 AND id < 16000",
            (),
        )
        .unwrap();
    assert_eq!(gone, 0);
}
