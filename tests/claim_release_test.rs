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

//! A refused cold update lets go of the claims it took, but never of one
//! another handle of the same transaction relies on

#![cfg(feature = "test-failpoints")]

use std::sync::mpsc;
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints;
use stoolap::Database;

#[test]
fn a_claim_another_handle_relies_on_survives_a_refused_update() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0), (2, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Another transaction holds row 2, uncommitted
    let mut holder = db.begin().unwrap();
    holder
        .execute("UPDATE t SET v = 7 WHERE id = 2", ())
        .unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut updater = txn.get_table("t").unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    let (start, started) = mpsc::channel::<()>();
    let (claimed, has_claimed) = mpsc::channel::<()>();
    let (resume, resumed) = mpsc::channel::<()>();
    let (updated, deleted) = std::thread::scope(|scope| {
        let delete = scope.spawn(move || {
            started.recv().unwrap();
            // The delete holds row 1's claim and has not written yet
            test_failpoints::after_cold_only_rows_claimed(move || {
                claimed.send(()).unwrap();
                resumed.recv().unwrap();
            });
            deleter.delete_by_row_ids(&[1])
        });
        // The update takes row 1's claim first, then row 2 refuses it
        test_failpoints::after_cold_claim_taken(move || {
            start.send(()).unwrap();
            has_claimed.recv().unwrap();
        });
        let updated = updater.update_by_row_ids(&[1, 2], &mut |row| Ok((row, true)));
        resume.send(()).unwrap();
        (updated, delete.join().unwrap())
    });
    let mut other = db.begin().unwrap();
    let taken = other.execute("UPDATE t SET v = 9 WHERE id = 1", ());
    other.rollback().unwrap();
    drop(updater);
    txn.rollback().unwrap();
    holder.rollback().unwrap();
    assert!(updated.is_err(), "row 2 is held: {updated:?}");
    assert_eq!(deleted.unwrap(), 1);
    assert!(
        taken.is_err(),
        "another transaction wrote a row the uncommitted delete holds: {taken:?}"
    );
}

#[test]
fn a_prepared_row_another_handle_wrote_is_not_set_again() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0), (2, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut updater = txn.get_table("t").unwrap();
    let mut writer = txn.get_table("t").unwrap();
    let reader = txn.get_table("t").unwrap();
    // After the update claims row 1, the other handle writes it locally
    test_failpoints::after_cold_claim_taken(move || {
        let mut set = |mut row: stoolap::core::Row| {
            row.set(1, stoolap::core::Value::Integer(5)).unwrap();
            Ok((row, true))
        };
        writer.update_by_row_ids(&[1], &mut set).unwrap();
    });
    let mut calls = Vec::new();
    let updated = updater.update(None, &mut |mut row| {
        calls.push(row.get(0).and_then(|v| v.as_int64()).unwrap());
        row.set(1, stoolap::core::Value::Integer(7)).unwrap();
        Ok((row, true))
    });
    let mut rows: Vec<(i64, i64)> = reader
        .collect_all_rows(None)
        .unwrap()
        .iter()
        .map(|(_, row)| {
            let value = |i: usize| row.get(i).and_then(|v| v.as_int64()).unwrap();
            (value(0), value(1))
        })
        .collect();
    rows.sort_unstable();
    drop(updater);
    drop(reader);
    txn.rollback().unwrap();
    calls.sort_unstable();
    assert_eq!(calls, vec![1, 2], "the setter ran again: {updated:?}");
    let conflict = matches!(&updated, Err(e) if e.to_string().contains("write conflict"));
    assert!(
        conflict,
        "the other handle wrote row 1 after it was prepared: {updated:?}"
    );
    assert_eq!(
        rows,
        vec![(1, 5), (2, 0)],
        "the refused statement overwrote the other handle's write or left part of its own"
    );
}

#[test]
fn a_late_conflict_on_the_second_row_leaves_no_part_of_the_statement() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0), (2, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut updater = txn.get_table("t").unwrap();
    let mut writer = txn.get_table("t").unwrap();
    let reader = txn.get_table("t").unwrap();
    // Row 1 is claimed; the other handle writes row 2 before it is applied
    test_failpoints::after_cold_claim_taken(move || {
        let mut set = |mut row: stoolap::core::Row| {
            row.set(1, stoolap::core::Value::Integer(5)).unwrap();
            Ok((row, true))
        };
        writer.update_by_row_ids(&[2], &mut set).unwrap();
    });
    let mut calls = Vec::new();
    let updated = updater.update(None, &mut |mut row| {
        calls.push(row.get(0).and_then(|v| v.as_int64()).unwrap());
        row.set(1, stoolap::core::Value::Integer(7)).unwrap();
        Ok((row, true))
    });
    let rows = rows_of(reader.as_ref());
    drop(updater);
    drop(reader);
    txn.rollback().unwrap();
    calls.sort_unstable();
    assert_eq!(calls, vec![1, 2], "the setter ran again: {updated:?}");
    let conflict = matches!(&updated, Err(e) if e.to_string().contains("write conflict"));
    assert!(
        conflict,
        "the other handle wrote row 2 after it was prepared: {updated:?}"
    );
    assert_eq!(
        rows,
        vec![(1, 0), (2, 5)],
        "the refused statement left row 1 changed or overwrote the other handle's write"
    );
}

fn rows_of(table: &dyn stoolap::storage::traits::Table) -> Vec<(i64, i64)> {
    let mut rows: Vec<(i64, i64)> = table
        .collect_all_rows(None)
        .unwrap()
        .iter()
        .map(|(_, row)| {
            let value = |i: usize| row.get(i).and_then(|v| v.as_int64()).unwrap();
            (value(0), value(1))
        })
        .collect();
    rows.sort_unstable();
    rows
}
