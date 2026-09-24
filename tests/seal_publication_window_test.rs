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

//! A read that merges a sealed table's hot rows with its volumes, begun
//! before a seal took its fence and reading while the seal is inside it.

#![cfg(feature = "test-failpoints")]

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use stoolap::core::Value;
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

#[derive(Clone, Copy)]
enum Hold {
    RowsRemoved,
    IndexesCleaned,
}

type Handle = Arc<Mutex<Option<std::thread::JoinHandle<()>>>>;

/// Runs `read` on this thread with a checkpoint on another handle held at
/// `hold` from the moment the read begins until it has taken its hot rows
/// and its cold view; `during_build` runs on the sealing thread once the
/// volume is built, before the fence
fn inside_seal<T>(
    db: &Database,
    hold: Hold,
    during_build: Option<Box<dyn FnOnce() + Send>>,
    read: impl FnOnce() -> T,
) -> T {
    let (reached_tx, reached_rx) = mpsc::channel::<()>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let go = Arc::new(Mutex::new(Some(go_tx)));
    let sealer: Handle = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&sealer);
    let other = db.clone();
    test_failpoints::after_merged_read_began(move || {
        let handle = std::thread::spawn(move || {
            if let Some(f) = during_build {
                test_failpoints::after_side_files_built(f);
            }
            let pause = move || {
                reached_tx.send(()).unwrap();
                go_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            };
            match hold {
                Hold::RowsRemoved => test_failpoints::in_seal_after_rows_removed(pause),
                Hold::IndexesCleaned => test_failpoints::in_seal_after_indexes_cleaned(pause),
            }
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        *slot.lock().unwrap() = Some(handle);
        reached_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the seal reached its hold");
    });
    let release = Arc::clone(&go);
    let collected = Arc::new(Mutex::new(false));
    let seen = Arc::clone(&collected);
    test_failpoints::after_merged_read_collected(move || {
        *seen.lock().unwrap() = true;
        if let Some(tx) = release.lock().unwrap().take() {
            tx.send(()).unwrap();
        }
    });
    let out = read();
    if let Some(tx) = go.lock().unwrap().take() {
        let _ = tx.send(());
    }
    let handle = sealer
        .lock()
        .unwrap()
        .take()
        .expect("the read began a merged read of a sealed table");
    handle.join().unwrap();
    assert!(
        *collected.lock().unwrap(),
        "the read took its hot rows and cold view"
    );
    out
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    let mut rows: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
        })
        .collect();
    rows.sort_unstable();
    rows
}

fn row_pairs(rows: &stoolap::core::RowVec) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = rows
        .iter()
        .map(|(_, row)| match (row.get(0), row.get(1)) {
            (Some(Value::Integer(id)), Some(Value::Integer(v))) => (*id, *v),
            other => panic!("unexpected row {:?}", other),
        })
        .collect();
    out.sort_unstable();
    out
}

/// 20 rows sealed, then row 5 updated to 105: its cold copy is tombstoned
/// and its new version is hot until the next seal moves it
fn cold_row_updated(dir: &tempfile::TempDir, indexed: bool) -> Database {
    let db = file_db(dir);
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    if indexed {
        db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    }
    for id in 1..=20i64 {
        db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET v = 105 WHERE id = 5", ()).unwrap();
    db
}

fn after_update() -> Vec<(i64, i64)> {
    (1..=20)
        .map(|id| (id, if id == 5 { 105 } else { id }))
        .collect()
}

#[test]
fn a_row_updated_during_a_seal_does_not_come_back_from_its_sealed_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    for id in 1..=10i64 {
        db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for id in 11..=20i64 {
        db.execute("INSERT INTO t VALUES ($1, $1)", (id,)).unwrap();
    }
    let writer = db.clone();
    let during_build: Box<dyn FnOnce() + Send> = Box::new(move || {
        writer
            .execute("UPDATE t SET v = -1 WHERE id = 15", ())
            .unwrap();
    });
    let rows = inside_seal(&db, Hold::RowsRemoved, Some(during_build), || {
        pairs(&db, "SELECT id, v FROM t WHERE v > 0")
    });
    let expected: Vec<(i64, i64)> = (1..=20).filter(|&id| id != 15).map(|id| (id, id)).collect();
    assert_eq!(
        pairs(&db, "SELECT id, v FROM t WHERE v > 0"),
        expected,
        "once the seal is done"
    );
    assert_eq!(rows, expected, "row 15 no longer matches, inside the seal");
}

#[test]
fn a_scan_inside_a_seal_keeps_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        pairs(&db, "SELECT id, v FROM t WHERE v > 0")
    });
    assert_eq!(
        pairs(&db, "SELECT id, v FROM t WHERE v > 0"),
        after_update(),
        "once the seal is done"
    );
    assert_eq!(rows, after_update(), "row 5 inside the seal");
}

#[test]
fn collect_all_rows_inside_a_seal_keeps_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        row_pairs(&table.collect_all_rows(None).unwrap())
    });
    assert_eq!(rows, after_update(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn collect_all_rows_unsorted_inside_a_seal_keeps_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        row_pairs(&table.collect_all_rows_unsorted().unwrap())
    });
    assert_eq!(rows, after_update(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn active_row_ids_inside_a_seal_keep_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut ids = inside_seal(&db, Hold::IndexesCleaned, None, || {
        table.get_active_row_ids().unwrap()
    });
    ids.sort_unstable();
    assert_eq!(ids, (1..=20).collect::<Vec<i64>>(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn ordered_pk_rows_inside_a_seal_keep_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        let rows = table
            .collect_rows_ordered_by_index("id", true, 100, 0)
            .unwrap()
            .expect("the ordered primary key path answers");
        row_pairs(&rows)
    });
    assert_eq!(rows, after_update(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn top_k_inside_a_seal_keeps_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        let rows = table
            .scan_top_k(None, "v", true, 100, 0)
            .unwrap()
            .expect("the top-k path answers");
        row_pairs(&rows)
    });
    assert_eq!(rows, after_update(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn current_as_of_inside_a_seal_keeps_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, false);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = inside_seal(&db, Hold::IndexesCleaned, None, || {
        let mut result = table
            .select_as_of(&["id", "v"], None, "CURRENT", 0)
            .unwrap();
        let mut rows = stoolap::core::RowVec::new();
        while result.next() {
            rows.push((0, result.take_row()));
        }
        row_pairs(&rows)
    });
    assert_eq!(rows, after_update(), "row 5 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn partition_values_inside_a_seal_keep_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, true);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut values: Vec<i64> = inside_seal(&db, Hold::IndexesCleaned, None, || {
        table
            .get_partition_values("v")
            .unwrap()
            .expect("the indexed column's values answer")
            .into_iter()
            .map(|v| v.as_int64().unwrap())
            .collect()
    });
    values.sort_unstable();
    let expected: Vec<i64> = after_update()
        .into_iter()
        .map(|(_, v)| v)
        .collect::<Vec<_>>();
    let mut expected = expected;
    expected.sort_unstable();
    assert_eq!(values, expected, "value 105 inside the seal");
    tx.rollback().unwrap();
}

#[test]
fn distinct_values_inside_a_seal_keep_a_cold_row_updated_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row_updated(&dir, true);
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut values: Vec<i64> = inside_seal(&db, Hold::IndexesCleaned, None, || {
        table
            .compute_distinct_values(1)
            .unwrap()
            .expect("the indexed column's values answer")
            .into_iter()
            .map(|v| v.as_int64().unwrap())
            .collect()
    });
    values.sort_unstable();
    let mut expected: Vec<i64> = after_update().into_iter().map(|(_, v)| v).collect();
    expected.sort_unstable();
    assert_eq!(values, expected, "value 105 inside the seal");
    tx.rollback().unwrap();
}
