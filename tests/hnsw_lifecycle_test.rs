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

//! The HNSW graph covers hot and sealed rows alike, so it must follow a
//! sealed row's update, answer an older snapshot as the snapshot sees the
//! table, be filled from the sealed rows whenever it is created, be rebuilt
//! through each volume's column mapping, and keep the NULL rules of ORDER BY.

use stoolap::api::Transaction;
use stoolap::{Database, IsolationLevel};

fn file_db(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_interval=0&checkpoint_on_close=off",
        dir.display()
    ))
    .unwrap()
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

fn ids_tx(tx: &mut Transaction, sql: &str) -> Vec<i64> {
    tx.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

/// (id, distance) pairs of a query projecting both
fn distances(db: &Database, sql: &str) -> Vec<(i64, f64)> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

fn distances_tx(tx: &mut Transaction, sql: &str) -> Vec<(i64, f64)> {
    tx.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

const NEAREST_100: &str = "SELECT id, VEC_DISTANCE_L2(v, '[100,0]') AS d FROM t ORDER BY d LIMIT 2";
const NEAREST_1: &str = "SELECT id, VEC_DISTANCE_L2(v, '[1,0]') AS d FROM t ORDER BY d LIMIT 3";
const NEAREST_0: &str = "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[0,0]') LIMIT 1";

/// A sealed row updated in place: the graph follows, so the row's distance
/// to its own new vector is zero and it comes first.
#[test]
fn a_sealed_vectors_update_reaches_the_graph() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(dir.path());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '[1,0]'), (2, '[10,0]')", ())
        .unwrap();
    db.execute("CREATE INDEX idx_v ON t(v) USING HNSW", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET v = '[100,0]' WHERE id = 1", ())
        .unwrap();
    assert_eq!(distances(&db, NEAREST_100), vec![(1, 0.0), (2, 90.0)]);
    // Updated again while hot, then sealed again
    db.execute("UPDATE t SET v = '[10,1]' WHERE id = 1", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[10,1]') LIMIT 1"
        ),
        vec![1]
    );
}

fn seed(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, '[1,0]'), (2, '[3,0]'), (3, '[10,0]')",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_v ON t(v) USING HNSW", ())
        .unwrap();
}

/// An older snapshot keeps its own vectors and distances after another
/// transaction updates a row, hot and sealed alike.
#[test]
fn an_older_snapshot_sees_its_own_vectors_after_an_update() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            file_db(dir.path())
        } else {
            Database::open("memory://hnsw_lifecycle_snapshot_update").unwrap()
        };
        seed(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let want = vec![(1, 0.0), (2, 2.0), (3, 9.0)];
        let mut snapshot = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(
            distances_tx(&mut snapshot, NEAREST_1),
            want,
            "sealed {sealed}"
        );
        db.execute("UPDATE t SET v = '[100,0]' WHERE id = 1", ())
            .unwrap();
        assert_eq!(
            distances_tx(&mut snapshot, NEAREST_1),
            want,
            "after the update, sealed {sealed}"
        );
        snapshot.rollback().unwrap();
        assert_eq!(
            distances(&db, NEAREST_1),
            vec![(2, 2.0), (3, 9.0), (1, 99.0)],
            "the current view, sealed {sealed}"
        );
    }
}

/// An older snapshot answers from the rows it sees when another
/// transaction inserts a nearer row it must not see.
#[test]
fn an_older_snapshot_answers_from_its_visible_rows_after_an_insert() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            file_db(dir.path())
        } else {
            Database::open("memory://hnsw_lifecycle_snapshot_insert").unwrap()
        };
        seed(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let mut snapshot = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(ids_tx(&mut snapshot, NEAREST_0), vec![1], "sealed {sealed}");
        db.execute("INSERT INTO t VALUES (4, '[0,0]')", ()).unwrap();
        assert_eq!(
            ids_tx(&mut snapshot, NEAREST_0),
            vec![1],
            "after the insert, sealed {sealed}"
        );
        snapshot.rollback().unwrap();
        assert_eq!(
            ids(&db, NEAREST_0),
            vec![4],
            "the current view, sealed {sealed}"
        );
    }
}

/// An index created without USING on a vector column becomes an HNSW
/// index; created after a seal, it holds the sealed rows too.
#[test]
fn an_implicit_hnsw_index_created_after_a_seal_holds_the_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(dir.path());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '[1,0]'), (2, '[10,0]')", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX idx_v ON t(v)", ()).unwrap();
    // A hot row keeps the graph non-empty, so a graph missing the sealed
    // rows would answer with it instead of falling back
    db.execute("INSERT INTO t VALUES (3, '[50,0]')", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 1"
        ),
        vec![1]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[10,0]') LIMIT 2"
        ),
        vec![2, 1]
    );
}

/// Reopening rebuilds the graph from the volumes through each volume's
/// column mapping, so a column dropped after the seal does not shift the
/// vector column.
#[test]
fn reopening_rebuilds_the_graph_through_the_volumes_column_mapping() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = file_db(dir.path());
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, obsolete TEXT, v VECTOR(2))",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO t VALUES (1, 'a', '[1,0]'), (2, 'b', '[3,0]'), (3, 'c', '[10,0]')",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE t DROP COLUMN obsolete", ())
            .unwrap();
        db.execute("CREATE INDEX idx_v ON t(v) USING HNSW", ())
            .unwrap();
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 1"
            ),
            vec![1]
        );
        db.close().unwrap();
    }
    let db = file_db(dir.path());
    // A hot row keeps the graph non-empty, so a graph rebuilt from the
    // wrong column would answer with it instead of falling back
    db.execute("INSERT INTO t VALUES (4, '[50,0]')", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 1"
        ),
        vec![1]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[10,0]') LIMIT 3"
        ),
        vec![3, 2, 1]
    );
}

/// NULL vectors keep the ORDER BY rules: NULLS FIRST puts them first, and
/// the default order lists them after every distance, so a LIMIT past the
/// non-null rows reaches them.
#[test]
fn null_vectors_keep_the_order_by_rules_with_and_without_the_index() {
    for indexed in [false, true] {
        let db = Database::open(&format!("memory://hnsw_lifecycle_nulls_{indexed}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, NULL), (2, '[1,0]')", ())
            .unwrap();
        if indexed {
            db.execute("CREATE INDEX idx_v ON t(v) USING HNSW", ())
                .unwrap();
        }
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') ASC NULLS FIRST LIMIT 1"
            ),
            vec![1],
            "nulls first, indexed {indexed}"
        );
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 2"
            ),
            vec![2, 1],
            "the null tail, indexed {indexed}"
        );
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 1"
            ),
            vec![2],
            "the nearest, indexed {indexed}"
        );
    }
}

/// The column's spelling in the query does not decide whether its index
/// is found.
#[test]
fn the_index_is_found_whatever_the_columns_spelling_in_the_query() {
    let db = Database::open("memory://hnsw_lifecycle_spelling").unwrap();
    seed(&db);
    for column in ["v", "V"] {
        let plan: Vec<String> = db
            .query(
                &format!(
                    "EXPLAIN SELECT id FROM t ORDER BY VEC_DISTANCE_L2({column}, '[1,0]') LIMIT 1"
                ),
                (),
            )
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(
            plan.iter().any(|line| line.contains("HNSW Index Scan")),
            "{column}: {plan:?}"
        );
        assert_eq!(
            ids(
                &db,
                &format!("SELECT id FROM t ORDER BY VEC_DISTANCE_L2({column}, '[1,0]') LIMIT 1")
            ),
            vec![1]
        );
    }
}
