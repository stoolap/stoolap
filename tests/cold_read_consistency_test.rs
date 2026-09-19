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

//! A reader that takes its cold volume list must read that same view to the
//! end: a compaction running inside its window may neither bring a deleted
//! row back nor take a live one away.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use stoolap::core::{RowVec, Value};
use stoolap::storage::traits::Engine;
use stoolap::Database;

const ROWS: i64 = 2_000;
const DELETED_UP_TO: i64 = 10;

/// The rows a scan returned, as `(row id, key column)`, ordered by row id.
fn ids_and_keys(rows: &RowVec) -> Vec<(i64, i64)> {
    let mut seen: Vec<(i64, i64)> = rows
        .iter()
        .map(|(id, row)| {
            let key = match row.get(1) {
                Some(Value::Integer(k)) => *k,
                other => panic!("row {id} has no integer key: {other:?}"),
            };
            (*id, key)
        })
        .collect();
    seen.sort_unstable();
    seen
}

/// What the fixture holds once its low ids are deleted: every id from 11 up,
/// once each, with the key the insert gave it.
fn expected() -> Vec<(i64, i64)> {
    (DELETED_UP_TO + 1..=ROWS).map(|id| (id, id % 7)).collect()
}

fn fixture(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();

    // Two small volumes, so the next checkpoint has something to merge
    for batch in 0..2 {
        let mut values = String::new();
        for i in 0..ROWS / 2 {
            let id = batch * (ROWS / 2) + i + 1;
            values.push_str(&format!("({id},{},1.0),", id % 7));
        }
        values.pop();
        db.execute(&format!("INSERT INTO t (id, k, v) VALUES {values}"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    // The tombstones stay uncompacted, so the compaction lands inside the
    // reader's window rather than before it
    db.execute(&format!("DELETE FROM t WHERE id <= {DELETED_UP_TO}"), ())
        .unwrap();
    db
}

/// A compaction that consumes the tombstones of the captured rows must not
/// let those rows back into a reader that took its view before it. The rows
/// are resident when it runs, so the volume itself is still readable: what
/// the reader loses is the record that they were deleted.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_during_a_cold_read_does_not_restore_a_deleted_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = fixture(dir.path());

    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    assert_eq!(
        ids_and_keys(&rows),
        expected(),
        "the reader's view of what was deleted must survive the compaction"
    );
}

/// A compaction that retires a captured volume must not take its live rows
/// with it. The volumes are metadata-only when the reader takes them, so they
/// have to be reloaded from disk to be read at all, and that reload must not
/// depend on the segment still being in the manifest.
///
/// The threshold is set so three volumes do not merge: the fourth, sealed
/// inside the reader's window, is what takes the count over it.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_during_a_cold_read_does_not_lose_a_live_row() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=3",
        dir.path().display()
    );
    let batch = |from: i64, count: i64| {
        let mut values = String::new();
        for id in from..from + count {
            values.push_str(&format!("({id},{},1.0),", id % 7));
        }
        values.pop();
        format!("INSERT INTO t (id, k, v) VALUES {values}")
    };
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
        for b in 0..3 {
            db.execute(&batch(b * 1_000 + 1, 1_000), ()).unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }

    // A reopened volume is resident, and the read only reloads a metadata-only
    // one, so the test drives the volumes down the tiers itself. The local
    // epochs leave the global eviction epoch alone.
    let db = Database::open(&dsn).unwrap();
    let (volumes, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!(volumes, 3, "the fixture sealed three volumes");
    assert_eq!(cold, 3, "all three are metadata-only before the read");

    let retired = Arc::new(AtomicUsize::new(usize::MAX));
    let seen = Arc::clone(&retired);
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute(&batch(3_001, 1_000), ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        seen.store(other.engine().volume_stats().len(), Ordering::Relaxed);
    });

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();

    // The fourth volume takes the count over the threshold, so the compaction
    // runs and the three captured segments leave the manifest
    assert!(
        retired.load(Ordering::Relaxed) < 3,
        "the compaction must retire the captured volumes for this test to guard anything"
    );

    // The row the hook inserts commits after this statement's view, so the
    // reader's answer is the three captured volumes and nothing else
    let expected: Vec<(i64, i64)> = (1..=3_000).map(|id| (id, id % 7)).collect();
    assert_eq!(
        ids_and_keys(&rows),
        expected,
        "a captured volume keeps the rows it held"
    );
}
