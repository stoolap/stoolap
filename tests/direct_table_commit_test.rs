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

//! A table handle's own commit inside a transaction: what the transaction's
//! commit, rollback and recovery make of the rows it applied

use std::path::Path;

use stoolap::core::{Operator, Row, Value};
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::traits::{Engine, Table};
use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

fn open(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7)", ()).unwrap();
    db
}

fn ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn delete_row_1(table: &mut dyn Table) {
    let mut filter = ComparisonExpr::new("id", Operator::Eq, Value::Integer(1));
    filter.prepare_for_schema(table.schema());
    assert_eq!(table.delete(Some(&filter)).unwrap(), 1);
}

fn recovered(dir: &Path) -> Vec<i64> {
    let image = tempfile::tempdir().unwrap();
    crash_image(dir, image.path());
    ids(&Database::open(&dsn(image.path())).unwrap())
}

#[test]
fn rows_a_table_commit_applied_survive_a_crash_after_the_transaction_commits() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    drop(table);
    txn.commit().unwrap();
    assert_eq!(ids(&db), vec![1, 2]);
    assert_eq!(
        recovered(dir.path()),
        vec![1, 2],
        "the committed row recovers"
    );
}

#[test]
fn rows_written_before_and_after_a_table_commit_recover_together() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    table
        .insert(Row::from(vec![Value::Integer(3), Value::Integer(9)]))
        .unwrap();
    drop(table);
    txn.commit().unwrap();
    assert_eq!(ids(&db), vec![1, 2, 3]);
    assert_eq!(recovered(dir.path()), vec![1, 2, 3], "both rows recover");
}

#[test]
fn a_rollback_takes_back_rows_a_table_commit_applied() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    drop(table);
    txn.rollback().unwrap();
    assert_eq!(ids(&db), vec![1], "the rolled back row is gone");
    assert_eq!(recovered(dir.path()), vec![1]);
}

#[test]
fn a_table_commit_is_not_seen_by_others_before_the_transaction_commits() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    assert_eq!(
        ids(&db),
        vec![1],
        "not visible before the transaction commits"
    );
    drop(table);
    txn.commit().unwrap();
    assert_eq!(ids(&db), vec![1, 2]);
}

#[test]
fn a_table_commit_deleting_a_sealed_row_is_not_seen_before_the_transaction_commits() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1, "the row is sealed");
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    delete_row_1(table.as_mut());
    table.commit().unwrap();
    assert_eq!(
        ids(&db),
        vec![1],
        "not deleted before the transaction commits"
    );
    drop(table);
    txn.rollback().unwrap();
    assert_eq!(ids(&db), vec![1], "the rollback keeps the row");
    assert_eq!(recovered(dir.path()), vec![1]);
}

#[test]
fn a_table_commit_deleting_a_sealed_row_survives_a_crash_after_the_transaction_commits() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    delete_row_1(table.as_mut());
    table.commit().unwrap();
    drop(table);
    txn.commit().unwrap();
    assert_eq!(ids(&db), Vec::<i64>::new());
    assert_eq!(
        recovered(dir.path()),
        Vec::<i64>::new(),
        "the delete recovers"
    );
}

fn ids_with_k(table: &dyn Table, k: i64) -> Vec<i64> {
    let mut filter = ComparisonExpr::new("k", Operator::Eq, Value::Integer(k));
    filter.prepare_for_schema(table.schema());
    let mut rows = table.select(&["id"], Some(&filter)).unwrap();
    let mut ids = Vec::new();
    while rows.next() {
        ids.push(rows.row().get(0).unwrap().as_int64().unwrap());
    }
    ids
}

fn sql_ids_with_k(db: &Database, k: i64) -> Vec<i64> {
    db.query("SELECT id FROM t WHERE k = $1 ORDER BY id", (k,))
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn a_savepoint_rollback_takes_back_an_indexed_change_two_table_commits_kept() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table.commit().unwrap();
    txn.create_savepoint("s").unwrap();
    table
        .update_by_row_ids(&[2], &mut |_| {
            Ok((Row::from(vec![Value::Integer(2), Value::Integer(9)]), true))
        })
        .unwrap();
    table.commit().unwrap();
    let other = txn.get_table("t").unwrap();
    assert_eq!(ids_with_k(other.as_ref(), 9), vec![2], "its own change");
    assert_eq!(ids_with_k(other.as_ref(), 8), Vec::<i64>::new());
    assert_eq!(ids(&db), vec![1], "nothing visible to others yet");
    txn.rollback_to_savepoint("s").unwrap();
    assert_eq!(
        ids_with_k(other.as_ref(), 8),
        vec![2],
        "back to the savepoint"
    );
    assert_eq!(ids_with_k(other.as_ref(), 9), Vec::<i64>::new());
    drop(other);
    drop(table);
    txn.commit().unwrap();
    assert_eq!(sql_ids_with_k(&db, 8), vec![2]);
    assert_eq!(sql_ids_with_k(&db, 9), Vec::<i64>::new());
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let reopened = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(ids(&reopened), vec![1, 2]);
    assert_eq!(sql_ids_with_k(&reopened, 8), vec![2], "recovered");
    assert_eq!(sql_ids_with_k(&reopened, 9), Vec::<i64>::new());
}

#[test]
fn a_unique_failure_at_the_transaction_commit_takes_back_every_table_commit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("CREATE UNIQUE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1, "row 1 is sealed");
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    delete_row_1(table.as_mut());
    table
        .insert(Row::from(vec![Value::Integer(2), Value::Integer(8)]))
        .unwrap();
    table
        .insert(Row::from(vec![Value::Integer(3), Value::Integer(9)]))
        .unwrap();
    table.commit().unwrap();
    drop(table);
    db.execute("INSERT INTO t VALUES (4, 9)", ()).unwrap();
    let committed = txn.commit();
    assert!(committed.is_err(), "k = 9 is taken: {committed:?}");
    assert_eq!(
        ids(&db),
        vec![1, 4],
        "the delete and the inserts are undone"
    );
    assert_eq!(recovered(dir.path()), vec![1, 4]);
}
