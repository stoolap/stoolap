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

fn cold_row(dir: &std::path::Path) -> Database {
    let db = open(dir);
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn bump(row: &mut stoolap::core::Row) -> (i64, i64) {
    let id = row.get(0).and_then(|v| v.as_int64()).unwrap();
    let old = row.get(1).and_then(|v| v.as_int64()).unwrap();
    row.set(1, stoolap::core::Value::Integer(old + 1)).unwrap();
    (id, old + 1)
}

#[test]
fn a_cold_unique_shift_lands() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Row 2's new key is row 1's old one, which the statement takes away
    assert_eq!(
        db.execute("UPDATE t SET k = k - 1 WHERE id IN (1, 2)", ())
            .unwrap(),
        2
    );
    let sum: i64 = db.query_one("SELECT SUM(k) FROM t", ()).unwrap();
    assert_eq!(sum, 1);
    assert_eq!(db.execute("UPDATE t SET k = k - 1", ()).unwrap(), 2);
    let sum: i64 = db.query_one("SELECT SUM(k) FROM t", ()).unwrap();
    assert_eq!(sum, -1);
}

#[test]
fn a_seal_landing_while_a_row_is_prepared_runs_the_setter_once() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    db.execute("INSERT INTO t VALUES (2, 0)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let other = db.clone();
    let mut calls = 0;
    let mut returned = Vec::new();
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        returned.push(bump(&mut row));
        if calls == 1 {
            // A seal lands between the row's preparation and the fence
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        Ok((row, true))
    };
    assert_eq!(table.update_by_row_ids(&[1], &mut setter).unwrap(), 1);
    drop(table);
    txn.commit().unwrap();
    assert_eq!(calls, 1);
    assert_eq!(returned, vec![(1, 1)]);
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, 1);
}

#[test]
fn a_row_another_transaction_changed_while_prepared_is_a_write_conflict() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let other = db.clone();
    let mut calls = 0;
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        bump(&mut row);
        if calls == 1 {
            other
                .execute("UPDATE t SET v = 10 WHERE id = 1", ())
                .unwrap();
        }
        Ok((row, true))
    };
    let err = table.update(None, &mut setter).unwrap_err().to_string();
    assert!(err.contains("write conflict"), "{err}");
    assert_eq!(calls, 1);
    drop(table);
    txn.rollback().unwrap();
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, 10);
}

#[test]
fn a_prepared_row_sealed_again_by_another_transaction_is_a_write_conflict() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let other = db.clone();
    let mut calls = 0;
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        bump(&mut row);
        if calls == 1 {
            // The row's newer version lands in a newer volume before the fence
            other
                .execute("UPDATE t SET v = 10 WHERE id = 1", ())
                .unwrap();
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        Ok((row, true))
    };
    let err = table
        .update_by_row_ids(&[1], &mut setter)
        .unwrap_err()
        .to_string();
    assert!(err.contains("write conflict"), "{err}");
    assert_eq!(calls, 1);
    drop(table);
    txn.rollback().unwrap();
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, 10);
}

#[test]
fn a_prepared_row_sealed_again_outside_the_filter_is_a_write_conflict() {
    use stoolap::storage::expression::{ComparisonExpr, Expression};
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let mut filter = ComparisonExpr::eq("v", stoolap::core::Value::Integer(0));
    filter.prepare_for_schema(table.schema());
    let other = db.clone();
    let mut calls = 0;
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        bump(&mut row);
        if calls == 1 {
            // The newer version no longer matches the filter, so the next
            // round's walk never reaches its volume
            other
                .execute("UPDATE t SET v = 10 WHERE id = 1", ())
                .unwrap();
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        Ok((row, true))
    };
    let err = table
        .update(Some(&filter), &mut setter)
        .unwrap_err()
        .to_string();
    assert!(err.contains("write conflict"), "{err}");
    assert_eq!(calls, 1);
    drop(table);
    txn.rollback().unwrap();
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, 10);
}

#[test]
fn a_compaction_moving_a_prepared_row_is_no_conflict() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    db.execute("INSERT INTO t VALUES (3, 0)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 0)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let other = db.clone();
    let mut calls = 0;
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        bump(&mut row);
        if calls == 1 {
            // The seal of row 2 makes three volumes, which compact into one
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        Ok((row, true))
    };
    assert_eq!(table.update_by_row_ids(&[1], &mut setter).unwrap(), 1);
    assert_eq!(calls, 1);
    drop(table);
    txn.commit().unwrap();
    let volumes = db.query("PRAGMA VOLUME_STATS", ()).unwrap().count();
    assert_eq!(volumes, 1, "the three volumes compacted into one");
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, 1);
    let sum: i64 = db.query_one("SELECT SUM(v) FROM t", ()).unwrap();
    assert_eq!(sum, 1);
}

#[test]
fn a_prepared_row_another_transaction_holds_fails_before_any_hot_write() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = cold_row(dir.path());
    db.execute("INSERT INTO t VALUES (2, 0)", ()).unwrap();
    // Another transaction holds row 1, uncommitted
    let mut holder = db.engine().begin_transaction().unwrap();
    let mut held = holder.get_table("t").unwrap();
    assert_eq!(
        held.update_by_row_ids(&[1], &mut |mut row| {
            bump(&mut row);
            Ok((row, true))
        })
        .unwrap(),
        1
    );
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let mut calls = 0;
    let mut setter = |mut row: stoolap::core::Row| {
        calls += 1;
        bump(&mut row);
        Ok((row, true))
    };
    let err = table.update(None, &mut setter).unwrap_err().to_string();
    assert!(err.contains("uncommitted changes"), "{err}");
    // The cold row was prepared; the hot row was never set
    assert_eq!(calls, 1);
    drop(table);
    txn.rollback().unwrap();
    drop(held);
    holder.rollback().unwrap();
    let sum: i64 = db.query_one("SELECT SUM(v) FROM t", ()).unwrap();
    assert_eq!(sum, 0);
}
