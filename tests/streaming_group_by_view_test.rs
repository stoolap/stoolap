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

//! A GROUP BY served by the index walk answers the statement's own rows, not
//! the rows of every committed transaction: a local insert, delete or key
//! update inside the transaction, and a snapshot that predates another
//! handle's key move, all reach the same answer as the scan.

use stoolap::Database;

fn seeded(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10.0), (2, 1, 20.0), (3, 2, 30.0)",
        (),
    )
    .unwrap();
    db
}

fn groups(db: &Database, sql: &str) -> Vec<(i64, f64)> {
    let mut out: Vec<(i64, f64)> = Vec::new();
    for row in db.query(sql, ()).unwrap() {
        let row = row.unwrap();
        out.push((row.get::<i64>(0).unwrap(), row.get::<f64>(1).unwrap()));
    }
    out.sort_by_key(|(k, _)| *k);
    out
}

fn value_for(groups: &[(i64, f64)], key: i64) -> f64 {
    groups
        .iter()
        .find(|(k, _)| *k == key)
        .unwrap_or_else(|| panic!("no group {key} in {groups:?}"))
        .1
}

/// The index walk carries the row to its new key and drops it from the old one.
#[test]
fn a_local_insert_is_counted_by_the_group_by() {
    let db = seeded("local_insert_sum");
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 1, 100.0)", ())
        .unwrap();
    let sums = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&sums, 1), 130.0, "the local row is in group 1");
    assert_eq!(value_for(&sums, 2), 30.0);
    db.execute("ROLLBACK", ()).unwrap();
}

/// COUNT reads the walk's id list without fetching a row per id, so it needs
/// the same guard as the aggregates that do fetch.
#[test]
fn a_local_insert_is_counted_by_a_count_only_group_by() {
    let db = seeded("local_insert_count");
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 1, 100.0)", ())
        .unwrap();
    let counts = groups(&db, "SELECT k, COUNT(*) FROM t GROUP BY k");
    assert_eq!(value_for(&counts, 1), 3.0, "the local row is in group 1");
    assert_eq!(value_for(&counts, 2), 1.0);
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn a_local_delete_leaves_the_group() {
    let db = seeded("local_delete");
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let sums = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&sums, 1), 20.0, "the deleted row is gone");
    db.execute("ROLLBACK", ()).unwrap();
}

/// The SUM above survives an unguarded walk because the row fetch drops the
/// deleted row; COUNT reads the id list and does not.
#[test]
fn a_local_delete_is_not_counted_by_a_count_only_group_by() {
    let db = seeded("local_delete_count");
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let counts = groups(&db, "SELECT k, COUNT(*) FROM t GROUP BY k");
    assert_eq!(value_for(&counts, 1), 1.0, "the deleted row is not counted");
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn a_local_update_moves_the_row_between_groups() {
    let db = seeded("local_update");
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET k = 2 WHERE id = 1", ()).unwrap();
    let sums = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&sums, 1), 20.0, "the moved row left group 1");
    assert_eq!(value_for(&sums, 2), 40.0, "and joined group 2");
    db.execute("ROLLBACK", ()).unwrap();
}

/// The index is updated at commit, before the versions are visible, so a
/// snapshot taken before the commit still groups the row under its old key.
#[test]
fn a_snapshot_keeps_the_group_key_it_read() {
    let db = seeded("snapshot_key");
    let other = db.clone();
    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    let before = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&before, 1), 30.0);

    other
        .execute("UPDATE t SET k = 2 WHERE id = 1", ())
        .unwrap();

    let after = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&after, 1), 30.0, "the snapshot still sees key 1");
    assert_eq!(value_for(&after, 2), 30.0);
    db.execute("ROLLBACK", ()).unwrap();
}

/// Without a local view to reconcile, the committed answer is unchanged.
#[test]
fn a_committed_group_by_counts_every_row() {
    let db = seeded("committed");
    let sums = groups(&db, "SELECT k, SUM(v) FROM t GROUP BY k LIMIT 10");
    assert_eq!(value_for(&sums, 1), 30.0);
    assert_eq!(value_for(&sums, 2), 30.0);
    let counts = groups(&db, "SELECT k, COUNT(*) FROM t GROUP BY k");
    assert_eq!(value_for(&counts, 1), 2.0);
    assert_eq!(value_for(&counts, 2), 1.0);
}
