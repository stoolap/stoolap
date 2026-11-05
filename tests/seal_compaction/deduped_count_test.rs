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

//! COUNT(*) after a delete of a cold row reads only the volumes the
//! tombstone can fall in

#![cfg(feature = "test-failpoints")]

use std::path::Path;

use stoolap::test_failpoints as fp;
use stoolap::Database;

const PER_VOLUME: i64 = 20_000;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// Two volumes of `PER_VOLUME` rows each, ids ascending
fn two_volumes(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    for volume in 0..2 {
        let ids: Vec<i64> = (volume * PER_VOLUME..(volume + 1) * PER_VOLUME).collect();
        for chunk in ids.chunks(5_000) {
            let values: Vec<String> = chunk.iter().map(|id| format!("({id}, {id})")).collect();
            db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), 2);
    db
}

fn count_walking(db: &Database) -> (i64, usize) {
    let walked = fp::count_ids_walked();
    let n: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    (n, fp::count_ids_walked() - walked)
}

#[test]
fn count_after_a_cold_delete_searches_a_volume_whose_order_is_decided() {
    let dir = tempfile::tempdir().unwrap();
    let db = two_volumes(dir.path());
    assert_eq!(count_walking(&db).0, 2 * PER_VOLUME);
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(
        count_walking(&db),
        (2 * PER_VOLUME - 1, 0),
        "recounted without walking any volume"
    );
}

#[test]
fn count_after_a_cold_delete_walks_only_the_undecided_volume_it_falls_in() {
    let dir = tempfile::tempdir().unwrap();
    two_volumes(dir.path()).close().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(count_walking(&db).0, 2 * PER_VOLUME);
    db.execute("DELETE FROM t WHERE v = 5", ()).unwrap();
    assert_eq!(
        count_walking(&db),
        (2 * PER_VOLUME - 1, PER_VOLUME as usize),
        "the second volume is out of range and not read"
    );
}
