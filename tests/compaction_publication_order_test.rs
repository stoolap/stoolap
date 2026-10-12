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

//! A compaction's publication masks its outputs without deciding the row
//! order of a newer volume read back from disk

#![cfg(feature = "test-failpoints")]

use std::path::Path;

use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000&target_volume_rows=65536",
        dir.display()
    )
}

fn insert(db: &Database, ids: impl Iterator<Item = i64>, k: impl Fn(i64) -> i64) {
    let ids: Vec<i64> = ids.collect();
    for chunk in ids.chunks(8_192) {
        let values: Vec<String> = chunk
            .iter()
            .map(|&id| format!("({id}, {}, 0)", k(id)))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
}

fn row_counts(db: &Database) -> Vec<usize> {
    db.engine().volume_stats().iter().map(|v| v.3).collect()
}

/// Three small volumes of ids 1, 2 and 3, then a newer full volume of ids
/// `newer_from..` held descending; reopened, so that volume's order is not
/// decided. With `shadow`, the full volume also holds a newer copy of 2
fn layout(dir: &Path, newer_from: i64, shadow: bool) -> Database {
    {
        let db = Database::open(&dsn(dir)).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute("ALTER TABLE t CLUSTER BY (k)", ()).unwrap();
        for id in 1..=3 {
            insert(&db, std::iter::once(id), |id| id);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let extra = i64::from(shadow);
        if shadow {
            db.execute("UPDATE t SET k = -2, v = 9 WHERE k = 2", ())
                .unwrap();
        }
        insert(&db, newer_from..newer_from + 65_536 - extra, |id| -id);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    Database::open(&dsn(dir)).unwrap()
}

fn compact(db: &Database) -> usize {
    let built = fp::row_orders_built();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    fp::row_orders_built() - built
}

#[test]
fn a_newer_volume_whose_range_misses_the_outputs_is_not_read() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = layout(dir.path(), 100, false);
    assert_eq!(row_counts(&db), vec![1, 1, 1, 65_536]);
    assert_eq!(compact(&db), 0, "no order built for the newer volume");
    assert_eq!(row_counts(&db), vec![3, 65_536], "the small volumes merged");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_539);
}

#[test]
fn a_newer_volume_shadowing_an_output_row_is_walked_without_an_order() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = layout(dir.path(), 100, true);
    assert_eq!(row_counts(&db), vec![1, 1, 1, 65_536]);
    assert_eq!(compact(&db), 0, "no order built for the newer volume");
    assert_eq!(row_counts(&db), vec![3, 65_536]);
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(v, 9, "the newer copy wins");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 65_538);
}
