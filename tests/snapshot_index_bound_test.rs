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

//! MIN and MAX in a snapshot transaction answer for the snapshot, not from
//! an index that holds the commits after it

use stoolap::{Database, IsolationLevel};

#[test]
fn a_snapshot_reads_min_and_max_as_of_its_start_over_sealed_and_hot_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX ia ON t(a)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 20, 1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 10, 0)", ()).unwrap();
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let mut bounds = || -> (i64, i64) {
        let mut one = |sql: &str| -> i64 {
            tx.query(sql, ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get(0)
                .unwrap()
        };
        (
            one("SELECT MIN(a) FROM t LIMIT 1"),
            one("SELECT MAX(a) FROM t LIMIT 1"),
        )
    };
    assert_eq!(bounds(), (10, 20));
    db.execute("INSERT INTO t VALUES (3, 5, 0), (4, 50, 0)", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(bounds(), (10, 20), "the bounds as of the snapshot's start");
}
