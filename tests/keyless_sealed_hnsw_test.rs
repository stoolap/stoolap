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

//! A table without an integer key updates a sealed row under a new row id
//! and deletes one by a tombstone alone: the vector index that keeps the
//! sealed rows must still lose the old row's node at commit, and take it
//! back when the commit fails

use stoolap::storage::index::hnsw::HnswIndex;
use stoolap::Database;

fn dsn(dir: &tempfile::TempDir) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    )
}

fn nearest(db: &Database, k: usize) -> Vec<(String, f64)> {
    db.query(
        &format!("SELECT name, VEC_DISTANCE_L2(v, '[1,0]') AS d FROM t ORDER BY d LIMIT {k}"),
        (),
    )
    .unwrap()
    .map(|r| {
        let r = r.unwrap();
        (r.get(0).unwrap(), r.get(1).unwrap())
    })
    .collect()
}

/// The ids the graph itself answers for `[1,0]`, nearest first
fn graph_ids(db: &Database, k: usize) -> Vec<i64> {
    let store = db.engine().get_version_store("t").unwrap();
    let index = store.get_index("idx_t_v").unwrap();
    let graph = index.as_any().downcast_ref::<HnswIndex>().unwrap();
    let query: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    graph
        .search_nearest(&query, k, 100)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

fn sealed_table(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (name TEXT, v VECTOR(2))", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_v ON t(v) USING HNSW", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_name ON t(name)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES ('near', '[1,0]'), ('mid', '[2,0]'), ('far', '[9,0]')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// A sealed row of a table without a key, replaced or deleted: the graph
/// answers with the rows that are, at their distances
#[test]
fn a_keyless_sealed_row_replaced_or_deleted_leaves_the_graph() {
    // The failing-commit guard below sets a process-wide failpoint
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    // The delete by a predicate walks the volume; the one with RETURNING
    // scans first and deletes the rows by id
    for statement in [
        "UPDATE t SET v = '[7,0]' WHERE name = 'near'",
        "DELETE FROM t WHERE name = 'near'",
        "DELETE FROM t WHERE name = 'near' RETURNING name",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = sealed_table(&dir);
        assert_eq!(
            graph_ids(&db, 3).len(),
            3,
            "{statement}: the sealed rows are in the graph"
        );
        if statement.contains("RETURNING") {
            assert_eq!(db.query(statement, ()).unwrap().count(), 1, "{statement}");
        } else {
            db.execute(statement, ()).unwrap();
        }
        let expected = if statement.starts_with("UPDATE") {
            vec![
                ("mid".to_string(), 1.0),
                ("near".to_string(), 6.0),
                ("far".to_string(), 8.0),
            ]
        } else {
            vec![("mid".to_string(), 1.0), ("far".to_string(), 8.0)]
        };
        assert_eq!(nearest(&db, 3), expected, "{statement}");
        assert_eq!(
            graph_ids(&db, 3).len(),
            expected.len(),
            "{statement}: the old node left the graph"
        );
        // And after reopen, the graph rebuilt from the volumes and the log
        db.close().unwrap();
        drop(db);
        let db = Database::open(&dsn(&dir)).unwrap();
        assert_eq!(nearest(&db, 3), expected, "{statement}: after reopen");
        assert_eq!(
            graph_ids(&db, 3).len(),
            expected.len(),
            "{statement}: after reopen"
        );
    }
}

#[cfg(feature = "test-failpoints")]
mod failpoints {
    use super::*;
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;

    /// A commit that fails after replacing or deleting a keyless sealed
    /// row leaves the old node in the graph, as the row is still there
    #[test]
    fn a_failed_commit_keeps_the_keyless_sealed_vector_in_the_graph() {
        let _guard = test_failpoints::FailpointGuard::new();
        for statement in [
            "UPDATE t SET v = '[7,0]' WHERE name = 'near'",
            "DELETE FROM t WHERE name = 'near'",
            "DELETE FROM t WHERE name = 'near' RETURNING name",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = sealed_table(&dir);
            test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
            let result = if statement.contains("RETURNING") {
                db.query(statement, ()).map(|rows| rows.count() as i64)
            } else {
                db.execute(statement, ())
            };
            test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
            assert!(result.is_err(), "{statement}: the commit failed");
            let before = vec![
                ("near".to_string(), 0.0),
                ("mid".to_string(), 1.0),
                ("far".to_string(), 8.0),
            ];
            assert_eq!(nearest(&db, 3), before, "{statement}");
            assert_eq!(
                graph_ids(&db, 3).len(),
                3,
                "{statement}: the old node is back"
            );
            // The hot B-tree never held the sealed row: the failed commit
            // leaves it without a key for it, NULL included
            use stoolap::core::{DataType, Value};
            let store = db.engine().get_version_store("t").unwrap();
            let names = store.get_index("idx_t_name").unwrap();
            assert!(
                names
                    .get_row_ids_equal(&[Value::Null(DataType::Text)])
                    .is_empty(),
                "{statement}: no NULL key"
            );
            assert!(
                names.get_row_ids_equal(&[Value::text("near")]).is_empty(),
                "{statement}: no key for the sealed row"
            );
        }
    }
}
