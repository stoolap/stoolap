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

//! Two readers that trust an index's answer while the rows move under
//! them: the memory engine's group walk while a TRUNCATE empties the
//! table, and a fetch of rows by id on a persistent table while a seal
//! moves the hot rows into a volume.

use std::time::{Duration, Instant};

use stoolap::storage::expression::logical::ConstBoolExpr;
use stoolap::storage::traits::Engine;
use stoolap::Database;

/// A group walk on the memory engine that a TRUNCATE outruns is not an
/// answer: the callback sees the rows leave before the walk ends, and the
/// walk reports that it cannot answer instead of the groups it captured
#[test]
fn a_group_walk_a_truncate_outran_is_not_an_answer() {
    let db = Database::open("memory://truncate_walk").unwrap();
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
    let store = db.engine().get_version_store("t").unwrap();
    let other = db.clone();
    let mut truncating: Option<std::thread::JoinHandle<()>> = None;
    let accepted = table
        .walk_btree_groups("k", 4096, 1024 * 1024, &mut |_key, ids| {
            if truncating.is_none() {
                // The walk holds the index, so the truncate clears the rows
                // and then waits for the index: the rows are gone before
                // this callback goes on
                let other = other.clone();
                truncating = Some(std::thread::spawn(move || {
                    other.execute("TRUNCATE TABLE t", ()).unwrap();
                }));
                let deadline = Instant::now() + Duration::from_secs(5);
                while store.committed_row_count() > 0 {
                    assert!(Instant::now() < deadline, "the truncate cleared the rows");
                    std::thread::yield_now();
                }
            }
            let mut rows = stoolap::core::RowVec::new();
            for page in ids.pages() {
                table.fetch_rows_by_ids_into(page, &ConstBoolExpr::true_expr(), &mut rows)?;
            }
            assert!(rows.is_empty(), "the truncated rows are gone");
            Ok(true)
        })
        .unwrap();
    assert!(
        accepted.is_none(),
        "a walk the truncate outran is not an answer"
    );
    tx.rollback().unwrap();
    truncating.unwrap().join().unwrap();
    let count: i64 = db
        .query("SELECT COUNT(*) FROM t", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .next()
        .unwrap();
    assert_eq!(count, 0);
}

#[cfg(feature = "test-failpoints")]
mod sealed {
    use super::*;
    use stoolap::test_failpoints;

    fn file_db(dir: &tempfile::TempDir) -> Database {
        Database::open(&format!(
            "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap()
    }

    /// A seal that lands after the vector search decided which of its
    /// candidates are hot and before it read them: every row comes back,
    /// with the distance of the vector it carries
    #[test]
    fn a_seal_inside_the_vector_fetch_loses_no_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, '[1,0]'), (2, '[10,0]')", ())
            .unwrap();
        db.execute("CREATE INDEX idx_t_v ON t(v) USING HNSW", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, '[2,0]'), (4, '[3,0]')", ())
            .unwrap();

        let other = db.clone();
        test_failpoints::after_row_ids_classified(move || {
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        let rows: Vec<(i64, f64)> = db
            .query(
                "SELECT id, VEC_DISTANCE_L2(v, '[1,0]') AS d FROM t ORDER BY d LIMIT 4",
                (),
            )
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<i64>(0).unwrap(), r.get::<f64>(1).unwrap())
            })
            .collect();
        assert_eq!(rows, vec![(1, 0.0), (3, 1.0), (4, 2.0), (2, 9.0)]);
        assert_eq!(db.engine().volume_stats().len(), 2, "the seal landed");
    }

    /// A seal that lands inside a join's fetch of its inner rows: the rows
    /// decided hot before it are read from the volume after it
    #[test]
    fn a_seal_inside_a_join_fetch_loses_no_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (3, 30), (4, 40)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        // The first probe of the join is a hot row, and the seal lands
        // once that row is decided hot
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO p VALUES (1), (2), (3), (4)", ())
            .unwrap();

        let other = db.clone();
        test_failpoints::after_row_ids_classified(move || {
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        let mut rows: Vec<(i64, i64)> = db
            .query(
                "SELECT p.id, t.v FROM p INNER JOIN t ON p.id = t.id LIMIT 10",
                (),
            )
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
            })
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![(1, 10), (2, 20), (3, 30), (4, 40)]);
    }
}
