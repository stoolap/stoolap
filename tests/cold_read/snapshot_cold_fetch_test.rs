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

//! A snapshot reads a sealed row through its transaction's table handle as
//! of its begin: a newer hot version committed after the snapshot began, or
//! a delete committed after it, must not hide the sealed copy

use stoolap::storage::expression::ConstBoolExpr;
use stoolap::{Database, Engine, IsolationLevel, Row, Value};

fn sealed_row(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'orig')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1, "row 1 is sealed");
    db
}

fn fetched_val(tx: &dyn stoolap::Transaction) -> Option<Value> {
    let table = tx.get_table("t").unwrap();
    let rows = table
        .fetch_rows_by_ids(&[1], &ConstBoolExpr::true_expr())
        .unwrap();
    rows.into_iter()
        .next()
        .map(|(_, row)| row.get(1).cloned().unwrap())
}

#[test]
fn test_snapshot_fetch_reads_sealed_row_under_later_update() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("UPDATE t SET val = 'new' WHERE id = 1", ())
        .unwrap();
    assert_eq!(fetched_val(&*tx), Some(Value::text("orig")));
}

#[test]
fn test_snapshot_fetch_reads_sealed_row_under_later_delete() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(fetched_val(&*tx), Some(Value::text("orig")));
}

#[test]
fn test_snapshot_sql_point_read_under_later_update_and_delete() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("UPDATE t SET val = 'new' WHERE id = 1", ())
        .unwrap();
    let val: String = tx.query_one("SELECT val FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(val, "orig", "after a later update");
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let val: String = tx.query_one("SELECT val FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(val, "orig", "after a later delete");
}

#[test]
fn test_snapshot_join_reads_sealed_inner_row_under_later_update() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, tid INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (10, 1)", ()).unwrap();
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("UPDATE t SET val = 'new' WHERE id = 1", ())
        .unwrap();
    let val: String = tx
        .query_one("SELECT t.val FROM o JOIN t ON o.tid = t.id", ())
        .unwrap();
    assert_eq!(val, "orig");
}

#[test]
fn test_has_visible_row_counts_own_uncommitted_insert() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let mut table = tx.get_table("t").unwrap();
    table
        .insert(Row::from_values(vec![
            Value::Integer(5),
            Value::text("mine"),
        ]))
        .unwrap();
    assert!(table.has_visible_row(5).unwrap(), "own insert");
    assert!(table.has_visible_row(1).unwrap(), "sealed row");
}

#[test]
fn test_has_visible_row_skips_rows_committed_after_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_row(&dir);
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("INSERT INTO t VALUES (7, 'later')", ()).unwrap();
    db.execute("UPDATE t SET val = 'new' WHERE id = 1", ())
        .unwrap();
    let table = tx.get_table("t").unwrap();
    assert!(!table.has_visible_row(7).unwrap(), "insert after begin");
    assert!(
        table.has_visible_row(1).unwrap(),
        "sealed copy under a later update"
    );
}
