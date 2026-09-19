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

/// Under `test-filedb` a `memory://` DSN is a file database, whose tables go
/// through the segment wrapper. That wrapper cannot hand out its index, so it
/// captures a prefix of it, and the answer must not change when a seal turns
/// the rows that prefix described into volumes.
///
/// The HAVING clause keeps the query off the storage-aggregation path, which
/// would otherwise answer it before the walk is reached.
#[cfg(feature = "test-filedb")]
#[test]
fn a_group_by_answers_the_same_before_and_after_a_seal() {
    let db = seeded("seal_transition");
    let before = groups(
        &db,
        "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 0 LIMIT 10",
    );
    assert_eq!(value_for(&before, 1), 30.0);
    assert_eq!(value_for(&before, 2), 30.0);

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let after = groups(
        &db,
        "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 0 LIMIT 10",
    );
    assert_eq!(value_for(&after, 1), 30.0, "a sealed row is still grouped");
    assert_eq!(value_for(&after, 2), 30.0);
    let counts = groups(
        &db,
        "SELECT k, COUNT(*) FROM t GROUP BY k HAVING COUNT(*) > 0",
    );
    assert_eq!(value_for(&counts, 1), 2.0);
    assert_eq!(value_for(&counts, 2), 1.0);
}

/// SUM(v) and SUM(w) share a function name, so a HAVING that binds by name
/// alone tests the wrong aggregate and drops groups it should keep.
#[test]
fn a_having_on_the_second_aggregate_tests_that_aggregate() {
    let db = Database::open("memory://having_second").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL, w REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1,1,10,100),(2,1,20,200),(3,2,30,300)",
        (),
    )
    .unwrap();

    // Both aggregates are in the select list, and the HAVING names the second
    let mut rows: Vec<(i64, f64, f64)> = Vec::new();
    for row in db
        .query(
            "SELECT k, SUM(v), SUM(w) FROM t GROUP BY k HAVING SUM(w) > 100 LIMIT 10",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        rows.push((
            row.get::<i64>(0).unwrap(),
            row.get::<f64>(1).unwrap(),
            row.get::<f64>(2).unwrap(),
        ));
    }
    rows.sort_by_key(|(k, _, _)| *k);
    assert_eq!(rows, vec![(1, 30.0, 300.0), (2, 30.0, 300.0)]);
}

/// ROLLUP adds a total row that has no group key, which a walk over the key
/// order cannot produce.
#[test]
fn a_rollup_group_by_keeps_its_total_row() {
    use stoolap::Value;

    let db = seeded("rollup_total");
    let mut rows: Vec<(Option<i64>, f64)> = Vec::new();
    for row in db
        .query(
            "SELECT k, SUM(v) FROM t GROUP BY ROLLUP(k) HAVING SUM(v) > 0 LIMIT 10",
            (),
        )
        .unwrap()
    {
        let row = row.unwrap();
        let key = match row.get_value(0) {
            Some(Value::Integer(k)) => Some(*k),
            _ => None,
        };
        let total = match row.get_value(1) {
            Some(Value::Float(v)) => *v,
            other => panic!("total is not a float: {other:?}"),
        };
        rows.push((key, total));
    }
    rows.sort_by_key(|(key, _)| *key);
    assert_eq!(
        rows,
        vec![(None, 60.0), (Some(1), 30.0), (Some(2), 30.0)],
        "two groups and the total row"
    );
}

/// The walk keeps no distinct state, so a HAVING that asks for one must not
/// be answered by it: SUM(DISTINCT v) is not SUM(v).
#[test]
fn a_having_on_a_distinct_aggregate_falls_back() {
    let db = Database::open("memory://having_distinct").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,10),(2,1,10),(3,2,30)", ())
        .unwrap();

    let sums = groups(
        &db,
        "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(DISTINCT v) > 15 LIMIT 10",
    );
    assert_eq!(sums, vec![(2, 30.0)], "group 1 has a distinct sum of 10");
}

/// A primary-key index collects and sorts its overflow ids before it calls
/// back, so the capture's bounds would not bound that work.
#[test]
fn a_primary_key_group_index_is_not_captured() {
    use stoolap::storage::traits::Engine;

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}?sync_mode=none", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,10.0),(2,1,20.0),(3,2,30.0)", ())
        .unwrap();

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let captured = table
        .walk_btree_groups("id", 4096, 1024 * 1024, &mut |_, _| Ok(true))
        .unwrap();
    assert!(captured.is_none(), "a primary-key index is declined");
    let captured = table
        .walk_btree_groups("k", 4096, 1024 * 1024, &mut |_, _| Ok(true))
        .unwrap();
    assert!(captured.is_some(), "a b-tree index is captured");
    tx.rollback().unwrap();
}

/// A destructive operation between the capture and the rows it names must
/// invalidate the walk. TRUNCATE clears the segments and the hot rows without
/// registering a volume, so the seal generation is what has to carry it.
#[test]
fn a_truncate_during_the_walk_does_not_answer() {
    use stoolap::storage::expression::logical::ConstBoolExpr;
    use stoolap::storage::traits::Engine;

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}?sync_mode=none", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1,1,10.0),(2,1,20.0),(3,2,30.0)", ())
        .unwrap();

    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let other = db.clone();
    let mut truncated = false;
    let accepted = table
        .walk_btree_groups("k", 4096, 1024 * 1024, &mut |_key, ids| {
            if !truncated {
                other.execute("TRUNCATE TABLE t", ()).unwrap();
                truncated = true;
            }
            let mut rows = stoolap::core::RowVec::new();
            table.fetch_rows_by_ids_into(ids, &ConstBoolExpr::true_expr(), &mut rows)?;
            assert!(rows.is_empty(), "the truncated rows are gone");
            Ok(true)
        })
        .unwrap();
    assert!(
        accepted.is_none(),
        "a capture a truncate outran is not an answer"
    );
    tx.rollback().unwrap();
}

/// A table holding volumes captures a prefix of its index while the lock is
/// held. A query that needs more groups than the prefix holds must fall back
/// and still answer every group.
///
/// The HAVING clause is what puts the query on the streaming path: without
/// one, storage-level aggregation answers it before this walk is reached.
#[test]
fn a_group_by_past_the_capture_bound_answers_every_group() {
    let db = Database::open("memory://capture_bound").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v REAL NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();

    // 6,000 single-row groups: more groups than a capture holds rows, so the
    // prefix runs out before the end of the index
    for chunk in 0..10 {
        let mut values = String::new();
        for i in 0..600 {
            let id = chunk * 600 + i + 1;
            values.push_str(&format!("({id}, {id}, 1.0),"));
        }
        values.pop();
        db.execute(&format!("INSERT INTO t (id, k, v) VALUES {values}"), ())
            .unwrap();
    }

    let sums = groups(
        &db,
        "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 0 LIMIT 100000",
    );
    assert_eq!(sums.len(), 6000, "every group is answered");
    assert!(sums.iter().all(|(_, total)| *total == 1.0));
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
